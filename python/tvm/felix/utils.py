import logging
import typing as ty
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import onnx
import pytorch_lightning as pl
import torch
import tvm
from torch import nn
from tvm import auto_scheduler as ansor
from tvm import relay
from tvm.auto_scheduler.loop_state import StateObject
from tvm.relay.backend.executor_factory import GraphExecutorFactoryModule

logger = logging.getLogger(__file__)
__all__ = [
    "benchmark_tvm",
    "transpose2",
    "transpose3",
    "coalesce_dicts",
    "parse_task",
    "parse_task_name",
    "TARGET",
    "HW_PARAMS",
    "MEASURER",
]

TARGET = tvm.target.cuda(model="a5000", arch="sm_80")
HW_PARAMS = ansor.HardwareParams(
    num_cores=-1,
    vector_unit_bytes=16,
    cache_line_bytes=64,
    max_shared_memory_per_block=49152,
    max_local_memory_per_block=2147483647,
    max_threads_per_block=1024,
    max_vthread_extent=8,
    warp_size=32,
)
MEASURER = ansor.LocalRunner()

PathLike = ty.Union[Path, str]
InputSpec = Union[torch.Tensor, List[torch.Tensor]]


@dataclass
class ModelInput:
    DTYPE_MAP = {
        torch.float32: "float",
        torch.long: "long",
    }

    tensor: torch.Tensor
    name: Optional[str] = None

    @classmethod
    def from_input_arrays(cls, arrays):
        if isinstance(arrays, torch.Tensor):
            return [cls(arrays)]
        elif isinstance(arrays, (list, tuple)):
            for a in arrays:
                assert isinstance(a, torch.Tensor)
            return [cls(a) for a in arrays]
        elif isinstance(arrays, dict):
            for v in arrays.values():
                assert isinstance(v, torch.Tensor)
            return [cls(v, k) for k, v in arrays.items()]
        else:
            raise ValueError("Invalid input array")

    @classmethod
    def get_input_infos(cls, inputs: List["ModelInput"]):
        ret = []
        for idx, input_ in enumerate(inputs):
            name = f"input{idx}" if input_.name is None else input_.name
            dtype = cls.DTYPE_MAP.get(input_.tensor.dtype)
            if dtype is None:
                raise ValueError(f"Unsupported input dtype: {input_.tensor.dtype}")
            ret.append((name, (input_.tensor.shape, dtype)))
        return ret


def get_tvm_mod(
    model: ty.Union[nn.Module, onnx.ModelProto],  # type: ignore
    example_inputs: Optional[InputSpec] = None,
    to_nhwc: bool = True,
    entry_pt_name: Optional[str] = None,
):
    if isinstance(model, onnx.ModelProto):  # type: ignore
        mod, params = relay.frontend.from_onnx(model)
        if to_nhwc:
            mod = _convert_to_nhwc(mod)
        return mod, params

    if isinstance(model, nn.Module):
        if isinstance(model, pl.LightningModule):
            if entry_pt_name is not None:
                raise ValueError("Must trace forward() for pl.LightningModule")
            graph_model = model.to_torchscript(method="trace")
            example_inputs_ = ModelInput.from_input_arrays(model.example_input_array)
        else:
            if example_inputs is None:
                raise ValueError(
                    "Example input must be provided unless model is pl.LightningModule"
                )
            elif isinstance(example_inputs, torch.Tensor):
                example_inputs_ = [ModelInput(example_inputs)]
            else:
                example_inputs_ = [ModelInput(t) for t in example_inputs]
            inputs = [i.tensor for i in example_inputs_]
            graph_model = (
                torch.jit.trace(model, inputs)  # type: ignore
                if entry_pt_name is None
                else torch.jit.trace_module(model, {entry_pt_name: inputs})  # type: ignore
            )
        mod, params = relay.frontend.from_pytorch(
            graph_model, ModelInput.get_input_infos(example_inputs_)
        )
        if to_nhwc and _check_nhwc_on_torch(model):
            mod = _convert_to_nhwc(mod)
        return mod, params

    raise ValueError(f"Model type {type(model)} unsupported")


def _check_nhwc_on_torch(model: nn.Module):
    from torch.nn.intrinsic.quantized import ConvReLU2d as QConvReLU2d
    from torch.nn.quantized import Conv2d as QConv2d

    for module in model.modules():
        if not isinstance(module, (nn.Conv2d, QConv2d, QConvReLU2d)):
            continue
        cin = module.in_channels
        groups = module.groups
        if groups != 1 and cin != groups:
            logger.warning("Cannot apply NHWC on grouped, non-depthwise convolution")
            return False
    return True


def _convert_to_nhwc(mod):
    from tvm import transform

    # Convert the layout to NHWC
    # RemoveUnunsedFunctions is used to clean up the graph.
    converter = relay.transform.ConvertLayout(
        {
            "nn.conv2d": ["NHWC", "default"],
            "nn.max_pool2d": ["NHWC", "default"],
            "qnn.conv2d": ["NHWC", "default"],
        },
    )
    seq = transform.Sequential([relay.transform.RemoveUnusedFunctions(), converter])
    with transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod


def benchmark_tvm(
    built_mod: GraphExecutorFactoryModule,
    input_shape: Sequence[int],
    target_name: str,
    dtype: str = "float32",
    repeat: int = 500,
    gpu_id: int = 0,
):
    import tvm.contrib.graph_executor as runtime

    rt_device = tvm.device(target_name, gpu_id)
    module = runtime.GraphModule(built_mod["default"](rt_device))
    data_tvm = tvm.nd.array(np.random.uniform(size=input_shape).astype(dtype))
    module.set_input("input", data_tvm)
    timing_results = module.benchmark(rt_device, repeat=repeat)
    return timing_results


def get_cuda_code(task: ansor.SearchTask, state: StateObject):
    from tvm.driver.build_module import build

    sch, args = task.compute_dag.apply_steps_from_state(state, task.layout_rewrite_option)
    func = build(sch, args, "cuda")
    return func.imported_modules[0].get_source()


T1 = ty.TypeVar("T1")
T2 = ty.TypeVar("T2")
T3 = ty.TypeVar("T3")


def transpose2(xs: Iterable[Tuple[T1, T2]]) -> Tuple[List[T1], List[T2]]:
    if not xs:
        return [], []
    return tuple([list(xs_) for xs_ in zip(*xs)])  # type: ignore


def transpose3(xs: Iterable[Tuple[T1, T2, T3]]) -> Tuple[List[T1], List[T2], List[T3]]:
    if not xs:
        return [], [], []
    return tuple([list(xs_) for xs_ in zip(*xs)])  # type: ignore


def coalesce_dicts(dicts: List[Dict[str, ty.Any]]):
    all_keys = set([tuple(sorted(d.keys())) for d in dicts])
    if len(all_keys) != 1:
        raise ValueError(f"Conflicting keys: {all_keys}")
    keys = list(all_keys.pop())
    values = [[d[k] for k in keys] for d in dicts]
    return keys, values


def parse_task(task: ansor.SearchTask, keep_all_inputs: bool = True):
    import ast

    # Workload key is a list [hash, *in_shapes, out_shape]; we remove the hash from the list
    shapes: List[tuple] = ast.literal_eval(task.workload_key)[1:]
    task_desc, n_args = _parse_tvm_func_name(task.desc)
    shapes = shapes[:-1] if keep_all_inputs else shapes[:n_args]
    return task_desc, shapes


def parse_task_name(task_desc: str):
    task_desc_, _ = _parse_tvm_func_name(task_desc)
    return short_print_names(task_desc_)


def short_print_names(types_indices: List[Tuple[str, int]]):
    if len(types_indices) == 1:
        ty, indice = types_indices[0]
        return f"{ty}[{indice}]"
    types, indices = transpose2(types_indices)
    types_ = set(types)
    if len(types) == 1:
        ty = types_.pop()
        return f"{ty}{sorted(indices)}"
    return ",".join(f"{ty}[{indice}]" for ty, indice in types_indices)


def _parse_tvm_func_name(name: str) -> Tuple[List[Tuple[str, int]], int]:
    import re

    OPERATORS = {
        "nn_contrib_conv2d_NCHWc": ("ConvNCHWc", 2),
        "nn_contrib_conv2d_winograd_without_weight_transform": ("ConvW", 2),
        "nn_conv2d": ("Conv", 2),
        "nn_dense": ("Dense", 2),
        "nn_batch_matmul": ("BatchMatmul", 2),
        "nn_adaptive_avg_pool2d": ("AdaAvgPool", 1),
        "nn_avg_pool2d": ("AvgPool", 1),
        "nn_max_pool2d": ("MaxPool", 1),
        "fixed_point": ("Int", 1),
        "mean": ("Mean", 1),
        "add": ("+", 2),
        "subtract": ("-", 2),
        "multiply": ("Mul", 2),
        "divide": ("Div", 2),
        "power": ("Pow", 2),
        "cast": ("Cast", 1),
        "clip": ("Clip", 1),
        "nn_max": ("Max", 1),
        "nn_relu": ("ReLU", 1),
        "sigmoid": ("Sigmoid", 1),
        "nn_softmax": ("Softmax", 1),
        "variance": ("Variance", 2),
        "c": ("Unknown", 0),
    }

    def parse_typename(typename: str) -> Optional[Tuple[List[str], int]]:
        if typename == "vm_mod_fused":
            return [], 0
        for k, (to_name, op_n_args) in OPERATORS.items():
            suffix = f"_{k}"
            if not typename.endswith(suffix):
                continue
            ret = parse_typename(typename[: -len(suffix)])
            if ret is None:  # Some parsing failed downstream.
                continue
            ops, n_args = ret
            if not n_args:
                n_args = op_n_args
            return ops + [to_name], n_args
        # Try taking a hash from the end
        match = re.match(r"(.*)_[0-9a-f]+(_?)$", typename)
        if match:
            rest = match.group(1)
            if rest.endswith("_"):
                rest = rest[:-1]
            ret = parse_typename(rest)
            if ret is None:
                return None
            ops, n_args = ret
            return ops + ["Unknown"], n_args
        return None

    segments = name.split(",")
    ops_indices = []
    n_args = set()
    for s in segments:
        try:
            ss, i_s = s.rsplit("_", 1)
            idx = int(i_s)
        except ValueError:
            ss, idx = s, 0
        ret = parse_typename(ss)
        if ret is None:
            ops_name, n_args_ = ss, 0
        else:
            ops, n_args_ = ret
            ops_name = "/".join(ops)
        ops_indices.append((ops_name, idx))
        n_args.add(n_args_)
    assert len(n_args) == 1
    return ops_indices, list(n_args)[0]
