# mypy: disable-error-code="override, method-assign"
import abc
import logging
from collections import defaultdict
from functools import reduce
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

import networkx as nx
import symengine as sp
from tvm import auto_scheduler as ansor
from tvm import te, tir, topi

from .utils import transpose2

SymShapeT = List[sp.Expr]
# TVM-conventional dimension names.
# Used for specifying layout to TVM and for our variable names.
DIM_NAMES = {1: ["W"], 2: ["H", "W"], 3: ["D", "H", "W"]}

logger = logging.getLogger(__file__)


class SymbolicDAG:
    def __init__(self, nx_graph: nx.DiGraph) -> None:
        self.nx_graph = nx_graph
        # Sort nodes in the graph by topological order, but break ties by a
        # deterministic ordering between nodes.
        # (Let's use the string repr of the node.)
        self.sorted_nodes: List[RelayOpBuilder] = []
        for gen in nx.topological_generations(nx_graph):
            self.sorted_nodes.extend(sorted(gen, key=lambda x: str(x)))

    @classmethod
    def from_ansor(cls, dag: ansor.ComputeDAG, require_hash_match: bool = True):
        __old_tensor_eq = te.Tensor.__eq__
        te.Tensor.__eq__ = tensor_equal
        # Assuming only placeholders and compute for now...
        # Split into placeholders and compute ops and add all tensors into the env.
        env = _TensorEnv()
        compute_ops: List[te.ComputeOp] = []
        idx = 0
        for op in dag.ops:
            assert op.num_outputs == 1
            if isinstance(op, te.ComputeOp):
                compute_ops.append(op)
                continue
            assert isinstance(op, te.PlaceholderOp)
            tensor = op.output(0)
            tb = Const.from_te_tensor(tensor, idx)
            env.register_op_(tb, tensor, tb.dims, [])
            idx += 1
        # Start building a graph of nx.DiGraph format
        graph = nx.DiGraph()
        while compute_ops:
            if (matched := cls._match_op(compute_ops)) is None:
                return None
            # Here `attrs` are attributes of the op such as strides, padding, etc.
            nops, builder, attrs = matched
            env.add_to_varmap_(tvm_data_to_python(attrs))
            these_ops, compute_ops = compute_ops[:nops], compute_ops[nops:]
            inputs, (output,) = get_inputs_outputs(these_ops)
            input_shapes = []
            for idx, input in enumerate(inputs):
                provider, shape = env.get_by_tensor(input)
                graph.add_edge(provider, builder, input_i=idx)
                input_shapes.append(shape)
            oshape, cons = builder.infer_shape(input_shapes)
            env.register_op_(builder, output, oshape, cons)
        params = env.rename_vars_clear_all_(graph)
        te.Tensor.__eq__ = __old_tensor_eq
        self = cls(graph)
        if require_hash_match:
            remade_dag = self.make_ansor_compute_dag(params)
            if remade_dag is None:
                logger.warning("Failed to remake Ansor dag from symbolic dag")
                return None
            if remade_dag.workload_key() != dag.workload_key():
                logger.warning(
                    "Workload key check failed: %s != %s",
                    remade_dag.workload_key(),
                    dag.workload_key(),
                )
                return None
        return self, params

    def make_ansor_compute_dag(self, config: Optional[dict]) -> Optional[ansor.ComputeDAG]:
        from tvm import autotvm

        nodes = self.sorted_nodes
        graph = self.nx_graph
        # Turn off AutoTVM config not found warnings
        if config is None:
            config = self._make_sym_config(nodes)
        env: Dict[RelayOpBuilder, te.Tensor] = {}
        dag_inputs = []

        old_autotvm_silent = autotvm.GLOBAL_SCOPE.silent
        autotvm.GLOBAL_SCOPE.silent = True
        for node in nodes:
            assert isinstance(node, RelayOpBuilder)
            in_edges = graph.in_edges(node, data="input_i")  # type: ignore
            indices, inputs = transpose2(
                sorted(
                    [(idx, env[from_]) for from_, _, idx in in_edges],
                    key=lambda x: x[0],
                )
            )
            # Must be consecutive from 0 to n-1.
            assert indices == list(range(len(inputs)))
            if (out := node.make_te(config, *inputs)) is None:
                # Some node in the graph is unsupported yet.
                return None
            if isinstance(node, Const):
                dag_inputs.append((node.op_idx, out))
            env[node] = out
        autotvm.GLOBAL_SCOPE.silent = old_autotvm_silent

        graph_ret_nodes = [n for n in graph if graph.out_degree(n) == 0]
        if len(graph_ret_nodes) != 1:
            raise ValueError(f"Graph must have exactly one return node: {graph_ret_nodes}")
        ret_val = env[graph_ret_nodes[0]]
        # NOTE: must give both inputs and outputs.
        # With only outputs given, ComputeDAG can be constructed
        # but problem occurs when lowering to TE.
        # NOTE: this order is important as it affects workload_key.
        dag_inputs = [tensor for _, tensor in sorted(dag_inputs, key=lambda p: p[0])]
        return ansor.ComputeDAG(dag_inputs + [ret_val])

    def describe(self, unique_identify: bool = True) -> str:
        nodes = self.sorted_nodes
        if unique_identify:
            strs = [repr(node) for node in nodes]
            node_to_idx = {n: i for i, n in enumerate(nodes)}
            inputs = [
                (node_to_idx[from_], node_to_idx[to_], idx)
                for from_, to_, idx in self.nx_graph.in_edges(data="input_i")  # type: ignore
            ]
            strs.append(str(sorted(inputs)))
        else:
            strs = [str(node) for node in nodes]
        return ",".join(strs)

    def __str__(self) -> str:
        return self.describe(False)

    def __repr__(self) -> str:
        return self.describe(True)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, SymbolicDAG):
            return False
        return self.describe() == __o.describe()

    def __hash__(self) -> int:
        return hash(self.describe())

    @staticmethod
    def _match_op(
        ops: List[te.ComputeOp],
    ) -> Optional[Tuple[int, "RelayComputeOpBuilder", Dict]]:
        matches = []
        for build_ty in RELAY_BUILDERS:
            if (n_ops := build_ty.lookahead()) > len(ops):
                continue
            if (match_res := build_ty.try_from_ops(*ops[:n_ops])) is not None:
                builder, attrs = match_res
                matches.append((n_ops, builder, attrs))
        if len(matches) > 1:
            cls_names = [m.__class__.__name__ for _, m, _ in matches]
            raise ValueError(f"Multiple matches for {ops}: {cls_names}")
        if not matches:
            return None
        return matches[0]

    @staticmethod
    def _make_sym_config(nodes: List["RelayOpBuilder"]):
        config = {}
        for b in nodes:
            for symbol in b.free_vars():
                name = symbol.name
                if name in config:
                    continue
                config[name] = tir_config_var(name)
        return config


def tensor_equal(self: te.Tensor, other: te.Tensor) -> bool:
    """Taken from Tensor::operator== -- how is this not the default in Python?"""
    return self.op == other.op and self.value_index == other.value_index


class _TensorEnv:
    def __init__(self) -> None:
        self.env: Dict[te.Tensor, Tuple[RelayOpBuilder, SymShapeT]] = {}
        self.varmap: Dict[sp.Symbol, int] = {}
        self._rename_vars: Dict[str, sp.Symbol] = {}
        self.renamed_exprs: Dict[sp.Expr, sp.Symbol] = {}
        self.constraints: List[sp.Eq] = []

    def register_op_(
        self,
        builder: "RelayOpBuilder",
        output: te.Tensor,
        sym_shape: SymShapeT,
        cons: List[Tuple[sp.Expr, Union[str, sp.Expr]]],
    ):
        self.env[output] = builder, sym_shape
        concrete_shape = tvm_data_to_python(output.shape)
        assert isinstance(concrete_shape, list)
        if len(sym_shape) != len(concrete_shape):
            raise ValueError(f"Number of dimensions mismatch: {sym_shape} vs {concrete_shape}")
        for i in range(len(sym_shape)):
            sym, con = sym_shape[i], concrete_shape[i]
            if isinstance(sym, sp.Symbol) and sym not in self.varmap:
                self.varmap[sym] = con
            elif sym.subs(self.varmap) != con:
                raise ValueError(f"Shape mismatch in {i}-th dimension: {sym} vs {con}")
        self.register_cons_and_rename_(cons)

    def get_by_tensor(self, tensor: te.Tensor):
        return self.env[tensor]

    def add_to_varmap_(self, varmap: Dict[sp.Symbol, int]):
        for key, val in varmap.items():
            if key in self.varmap:
                assert self.varmap[key] == val
            self.varmap[key] = val

    def register_cons_and_rename_(self, cons: List[Tuple[sp.Expr, Union[str, sp.Expr]]]):
        for from_expr, to_val in cons:
            if isinstance(to_val, str):
                to_symbol = self._rename_vars.get(to_val)
                if to_symbol is None:
                    self._rename_vars[to_val] = to_symbol = sp.Symbol(to_val)
                self.constraints.append(sp.Eq(from_expr, to_symbol))
                self.renamed_exprs[from_expr] = to_symbol
                continue
            if from_expr not in self.renamed_exprs or not isinstance(to_val, sp.Integer):
                # Renamed expression shouldn't be set again to constant.
                # This happens sometimes with broadcast
                # which doesn't have the full context.
                self.constraints.append(sp.Eq(from_expr, to_val))

    def rename_vars_clear_all_(self, graph: nx.DiGraph):
        import sympy

        def sympy_solve(constraints, *target_vars: list):
            solutions = sympy.solve(constraints, *target_vars, dict=True)
            if len(solutions) != 1:
                raise ValueError(f"Cannot solve constraints: {constraints} with {target_vars}")
            return {k.name: sp.sympify(v) for k, v in solutions[0].items()}

        def solve(constraints, vars: Optional[set] = None):
            if vars is None:
                vars = set().union(*[c.free_symbols for c in constraints])
            vars_ = list(vars)
            if len(vars) != len(constraints):
                return sympy_solve(constraints, *vars_)
            solution: tuple = sp.linsolve(constraints, vars_)
            unsolved = any(expr.free_symbols.intersection(vars) for expr in solution)
            if unsolved:
                return sympy_solve(constraints, *vars_)
            return {k.name: v for k, v in zip(vars_, solution)}

        if self.constraints:
            # Express shape vars in terms of free vars in self.constraints,
            # and relabel the graph with new nodes.
            consts = [cb for cb, _ in self.env.values() if isinstance(cb, Const)]
            shape_vars = set().union(*[cb.dims for cb in consts])
            assert all(isinstance(size, sp.Symbol) for size in shape_vars)
            subs = solve(self.constraints, shape_vars)
            relabeling = {
                const: Const(const.op_idx, [size.subs(subs) for size in const.dims], const.dtype)
                for const in consts
            }
            nx.relabel_nodes(graph, relabeling, copy=False)
            # Solve the free vars into concrete values (integers).
            eqns = [sympy.Eq(k.subs(subs), v) for k, v in self.varmap.items()]
            varmap = {k: int(v) for k, v in solve(eqns).items()}
        else:
            varmap = self.varmap.copy()
        self.env.clear()
        self.varmap.clear()
        self._rename_vars.clear()
        self.renamed_exprs.clear()
        self.constraints.clear()
        return varmap


class RelayOpBuilder(abc.ABC):
    @abc.abstractmethod
    def make_te(
        self, config: Dict[str, Union[int, tir.SizeVar]], *inputs: te.Tensor
    ) -> Optional[te.Tensor]:
        pass

    @abc.abstractmethod
    def make_attrs(self, configs: Optional[dict]) -> List[List[sp.Expr]]:
        pass

    def free_vars(self):
        vars = [
            attr
            for attr_group in self.make_attrs(None)
            for attr in attr_group
            if isinstance(attr, sp.Symbol)
        ]
        return set(vars)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        def to_hashable(v):
            if isinstance(v, list):
                return tuple(to_hashable(x) for x in v)
            if isinstance(v, dict):
                return tuple((k, to_hashable(v)) for k, v in v.items())
            return v

        return hash(to_hashable(self.__dict__))

    # Short for make_or_get, used internally.
    @classmethod
    def _mog(
        cls, name: str, config: Optional[Dict[str, Union[int, tir.SizeVar]]]
    ) -> Union[tir.SizeVar, sp.Symbol, int]:
        return sp.Symbol(name) if config is None else config[name]


class Const(RelayOpBuilder):
    def __init__(self, op_idx: int, dims: List[sp.Expr], dtype: str):
        super().__init__()
        self.op_idx = op_idx
        self.dims = dims
        self.dtype = dtype

    @classmethod
    def from_te_tensor(cls, op: te.Tensor, op_idx: int) -> "Const":
        shape: List[sp.Expr] = [sp.Symbol(f"n_{op_idx}_{i}") for i in range(len(op.shape))]
        return cls(op_idx, shape, op.dtype)

    def make_te(self, config: Dict[str, Union[int, tir.SizeVar]]) -> Optional[te.Tensor]:
        if any(isinstance(v, tir.SizeVar) for v in config.values()):
            assert all(isinstance(v, tir.SizeVar) for v in config.values())
            converter = SymEngineToTir(cast(Dict[str, tir.SizeVar], config))
            shape = [converter(dim) for dim in self.dims]
        else:
            shape = [tir_int(int(dim.subs(config))) for dim in self.dims]
        return te.placeholder(shape=shape, dtype=self.dtype)

    def make_attrs(self, _):
        return [list(dim.free_symbols) for dim in self.dims]

    def __str__(self) -> str:
        return "Const"

    def __repr__(self) -> str:
        return f"Const({self.dims})"


TBuilder = TypeVar("TBuilder", bound="RelayComputeOpBuilder")


class RelayComputeOpBuilder(RelayOpBuilder):
    @classmethod
    @abc.abstractmethod
    def lookahead(cls) -> int:
        pass

    @classmethod
    @abc.abstractmethod
    def try_from_ops(cls: Type[TBuilder], *ops: te.ComputeOp) -> Optional[Tuple[TBuilder, dict]]:
        pass

    @abc.abstractmethod
    def infer_shape(
        self, input_shapes: List[SymShapeT]
    ) -> Tuple[SymShapeT, List[Tuple[sp.Expr, Union[str, sp.Expr]]]]:
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        # Requires this be defined for every subclass.
        pass

    def __repr__(self) -> str:
        return str(self)


class _GeneralConv(RelayComputeOpBuilder, abc.ABC):
    OPS = {1: topi.nn.conv1d_nwc, 2: topi.nn.conv2d_nhwc, 3: topi.nn.conv3d_ndhwc}

    def __init__(self, n_dims: int, n_groups: int, is_depthwise: bool) -> None:
        super().__init__()
        self.n_dims = n_dims
        self.n_groups = n_groups
        self.is_depthwise = is_depthwise

    def make_te(
        self,
        config: Dict[str, Union[int, tir.SizeVar]],
        input: te.Tensor,
        weight: te.Tensor,
    ) -> Optional[te.Tensor]:
        strides, paddings = self.make_attrs(config)
        # Repeat twice as a list; left/right, top/bottom, front/back
        paddings = paddings * 2
        dilation = [1] * self.n_dims
        # TVM grouped convolution has this limitation that the number of groups
        # must actually be an integer. We cannot use a symbolic variable here.
        if self.n_groups > 1:
            return None
        op = topi.nn.depthwise_conv2d_nhwc if self.is_depthwise else self.OPS[self.n_dims]
        return op(input, weight, strides, paddings, dilation=dilation)  # type: ignore

    def infer_shape(self, input_shapes: List[SymShapeT]):
        input, weight = input_shapes
        ni, *isizes, ci = input
        *wsizes, ci_w, co_w = weight
        assert len(isizes) == len(wsizes)
        if self.is_depthwise:
            cons = [(ni, "N"), (ci, "C"), (ci_w, "C"), (co_w, 1)]
            co = ci
        else:
            cons = [(ni, "N"), (ci, "Cin"), (ci_w, "Cin"), (co_w, "Cout")]
            co = co_w
        dim_names = DIM_NAMES[self.n_dims]
        assert len(dim_names) == self.n_dims
        for i in range(self.n_dims):
            cons.append((isizes[i], dim_names[i]))
            cons.append((wsizes[i], f"K{dim_names[i]}"))
        sym_strides, sym_pads = self.make_attrs()
        output_shape = conv_like_shape_infer(isizes, wsizes, sym_pads, sym_strides)
        output_shape = [ni, *output_shape, co]
        return output_shape, cons

    def make_attrs(self, config=None):
        ds = DIM_NAMES[self.n_dims]
        strides = [self._mog(f"Stride{ds[i]}", config) for i in range(self.n_dims)]
        paddings = [self._mog(f"Pad{ds[i]}", config) for i in range(self.n_dims)]
        return strides, paddings

    def _make_attrs_dict(self, strides, paddings, dilations, precision="float32"):
        dims = self.n_dims
        if len(paddings) != 2 * dims:
            raise ValueError(f"Invalid padding {paddings}")
        if paddings[:dims] != paddings[dims:]:
            raise ValueError(f"Unsupported unequal padding {paddings}")
        if any(x != 1 for x in dilations):
            raise ValueError(f"Unsupported dilation {dilations}")
        if precision != "float32":
            raise ValueError(f"Unsupported precision {precision}")
        attrs = {}
        sym_strides, sym_pads = self.make_attrs()
        for i in range(dims):
            attrs[sym_strides[i]] = strides[i]
            attrs[sym_pads[i]] = paddings[i]
        return attrs


class Conv(_GeneralConv):
    @classmethod
    def lookahead(cls):
        return 2

    @classmethod
    def try_from_ops(cls, op0: te.ComputeOp, op1: te.ComputeOp):
        if (paddings := detect_pad_op(op0)) is None:
            return None
        out1 = op1.output(0).name
        if out1 == "Conv1dOutput":
            dims = 1
        elif out1 == "Conv2dOutput":
            dims = 2
        elif out1 == "Conv3dOutput":
            dims = 3
        else:
            return None
        ret = cls(dims, 1, False)
        strides, groups = parse_conv_stride_group(op1)
        assert groups == 1
        dilations = [1] * dims
        attrs = ret._make_attrs_dict(strides, paddings, dilations)
        return ret, attrs

    def __str__(self) -> str:
        return f"Conv{self.n_dims}d"

    __repr__ = __str__


class DepthwiseConv2d(_GeneralConv):
    @classmethod
    def lookahead(cls):
        return 2

    @classmethod
    def try_from_ops(cls, op0: te.ComputeOp, op1: te.ComputeOp):
        if (paddings := detect_pad_op(op0)) is None:
            return None
        if op1.output(0).name != "DepthwiseConv2d":
            return None
        ret = cls(2, 1, True)
        strides, groups = parse_conv_stride_group(op1)
        assert groups == 1
        attrs = ret._make_attrs_dict(strides, paddings, dilations=[1, 1])
        return ret, attrs

    def __str__(self) -> str:
        return f"DepthwiseConv{self.n_dims}d"

    __repr__ = __str__


class GroupConv2d(_GeneralConv):
    @classmethod
    def lookahead(cls):
        return 2

    @classmethod
    def try_from_ops(cls, op0: te.ComputeOp, op1: te.ComputeOp):
        # See _GeneralConv.make_te; we don't have a way to deal with Group Conv yet.
        return None
        # if op0.output(0).name != "pad_temp" or op1.tag != "group_conv2d_nhwc":
        #     return None
        # strides, groups = parse_conv_stride_group(op1)
        # ret = cls(2, groups, False)
        # paddings = parse_pad_op(op0)
        # attrs = ret._make_attrs_dict(strides, paddings, dilations=[1, 1])
        # return ret, attrs

    def __str__(self) -> str:
        return f"GroupConv{self.n_dims}d"

    __repr__ = __str__


class Conv2dTensorCore(_GeneralConv):
    @classmethod
    def lookahead(cls):
        return 4

    def make_te(
        self,
        config: Dict[str, Union[int, tir.SizeVar]],
        input: te.Tensor,
        weight: te.Tensor,
    ) -> Optional[te.Tensor]:
        strides, paddings = self.make_attrs(config)
        # Repeat twice as a list; left/right, top/bottom, front/back
        paddings = paddings * 2
        dilation = [1] * self.n_dims
        return topi.cuda.conv2d_nhwc_tensorcore(
            input, weight, strides, paddings, dilation, "float32"
        )  # type: ignore

    @classmethod
    def try_from_ops(cls, op0, op1, op2, op3):
        if op3.tag != "conv2d_nhwc_tensorcore":
            return None
        attrs = tvm_data_to_python(op3.attrs["workload"])
        assert isinstance(attrs, list)
        strides, paddings, dilations, precision = attrs[3:]
        if len(paddings) != 4:
            raise ValueError(f"Invalid padding {paddings}")
        if paddings[:2] != paddings[2:]:
            raise ValueError(f"Unsupported unequal padding {paddings}")
        if len(strides) != 2:
            raise ValueError(f"Invalid stride {strides}")
        if any(x != 1 for x in dilations):
            raise ValueError(f"Unsupported dilation {dilations}")
        if precision != "float32":
            raise ValueError(f"Unsupported precision {precision}")
        ret = cls(2, 1, False)
        strideh, stridew = strides
        padh, padw = paddings[:2]
        (sstrideh, sstridew), (spadh, spadw) = ret.make_attrs()
        attrs = {sstrideh: strideh, sstridew: stridew, spadh: padh, spadw: padw}
        return ret, attrs

    def __str__(self) -> str:
        return "Conv2dTensorCore"


# Conv2dWinograd and TransposeConv2d are really different from the general conv op.


class Conv2dWinograd(RelayComputeOpBuilder):
    @classmethod
    def lookahead(cls) -> int:
        return 8

    def make_te(
        self,
        _: Dict[str, Union[int, tir.SizeVar]],
        input: te.Tensor,
        weight: te.Tensor,
    ) -> Optional[te.Tensor]:
        return topi.nn.conv2d_winograd_nhwc_without_weight_transform(
            input,
            weight,
            strides=[1, 1],
            padding=[1, 1, 1, 1],
            dilation=[1, 1],
            out_dtype="float32",
        )  # type: ignore

    def make_attrs(self, _):
        return []

    @classmethod
    def try_from_ops(cls, *ops: te.ComputeOp):
        assert len(ops) == 8
        outputs = [op.output(0).name for op in ops]
        # fmt: off
        to_match = [
            "data_pad", "input_tile", "B", "data_pack",
            "bgemm", "A", "inverse", "conv2d_winograd",
        ]
        # fmt: on
        if outputs != to_match:
            return None
        pack_kh, pack_kw = ops[4].input_tensors[1].shape[0:2]
        in_h, in_w = ops[0].input_tensors[0].shape[1:-1]
        out_h, out_w = ops[-1].output(0).shape[1:-1]
        assert in_h == in_w and out_h == out_w and (out_h - in_h) % 2 == 0
        size_diff = (out_h - in_h) // 2
        # 1. Winograd algorithm packs its kernel, but regardless, the kernel must be square.
        # 2. Winograd conv must be 1x1 stride.
        # 3. pack_hk = tile_size + kernel_size - 1. tile_size is 4 in TVM now
        #    (NOTE: could it be other value in other cases?)
        # 4. Kernel size could otherwise be 1x1 but not supported in TVM yet.
        if not (pack_kh == pack_kw == 6 and size_diff == 0):
            return None
        return cls(), {}

    def infer_shape(self, input_shapes: List[SymShapeT]):
        input, weight = input_shapes[:2]
        ni, h, w, ci = input
        # WARNING: Winograd Conv is HWOI not HWIO.
        pack_kh, pack_kw, co_w, ci_w = weight
        pad, pack_khw = 1, 6
        # fmt: off
        cons = [
            (ni, "N"), (ci, "Cin"), (ci_w, "Cin"), (co_w, "Cout"),
            (h / 4, "QHW"), (w / 4, "QHW"),
            (pack_kh, pack_khw), (pack_kw, pack_khw),
        ]
        # fmt: on
        return [ni, h, h, co_w], cons

    def __str__(self) -> str:
        return "Conv2dWinograd"

    __repr__ = __str__


class TransposeConv2d(RelayComputeOpBuilder):
    @classmethod
    def lookahead(cls) -> int:
        return 2

    def make_te(
        self,
        config: Dict[str, Union[int, tir.SizeVar]],
        input: te.Tensor,
        weight: te.Tensor,
    ) -> Optional[te.Tensor]:
        (sh, sw), (pad_h, pad_w) = self.make_attrs(config)
        n, h, w, c = input.shape
        kh, kw, ci, co = weight.shape
        assert c == ci

        # padding stage
        bpad_h, bpad_w = kh - 1 - pad_h, kw - 1 - pad_w
        ipadding = [0, (bpad_h + sh - 1) // sh, (bpad_w + sw - 1) // sw, 0]
        padded = topi.nn.pad(input, ipadding, ipadding)
        # remove extra padding introduced by dilatation
        idx_div = te.indexdiv
        idx_mod = te.indexmod
        border_h = idx_mod(sh - idx_mod(bpad_h, sh), sh)
        border_w = idx_mod(sw - idx_mod(bpad_w, sw), sw)

        # dilation stage
        strides = [1, sh, sw, 1]

        # We should embed this dilation directly into te.compute rather than creating a new te.compute.
        # Only in this way can we use unroll to eliminate the multiplication of zeros.
        def _dilate(*indices):
            not_zero = []
            index_tuple = []
            for i in range(len(padded.shape)):
                if not strides[i] == 1:
                    index_tuple.append(idx_div(indices[i], strides[i]))
                    not_zero.append(idx_mod(indices[i], strides[i]).equal(0))
                else:
                    index_tuple.append(indices[i])
            if not_zero:
                not_zero = te.all(*not_zero)
                return te.if_then_else(not_zero, padded(*index_tuple), tir.const(0.0, padded.dtype))
            return padded(*index_tuple)

        # convolution stage
        out_h = (h - 1) * sh - 2 * pad_h + kh
        out_w = (w - 1) * sw - 2 * pad_w + kw
        rc = te.reduce_axis((0, c), name="rc")
        rh = te.reduce_axis((0, kh), name="rh")
        rw = te.reduce_axis((0, kw), name="rw")
        return te.compute(
            (n, out_h, out_w, co),
            lambda n, h, w, co: te.sum(
                _dilate(n, h + rh + border_h, w + rw + border_w, rc)
                * weight[kh - 1 - rh, kw - 1 - rw, rc, co],
                axis=[rh, rw, rc],
            ),
            name="conv2d_transpose_nhwc",
        )  # type: ignore

    @classmethod
    def try_from_ops(cls, op0: te.ComputeOp, op1: te.ComputeOp):
        if op1.output(0).name != "conv2d_transpose_nhwc":
            return None
        if (strides := parse_deconv_stride(op1)) is None:
            return None
        ishape = op0.input_tensors[0].shape
        kshape = op1.input_tensors[1].shape
        oshape = op1.output(0).shape
        # NOTE: input is NHWC and kernel is HWIO
        padh = (ishape[1] - 1) * strides[0] + kshape[0] - oshape[1]
        padw = (ishape[2] - 1) * strides[1] + kshape[1] - oshape[2]
        assert padh % 2 == padw % 2 == 0
        padh, padw = padh // 2, padw // 2
        ret = cls()
        (sym_sh, sym_sw), (sym_ph, sym_pw) = ret.make_attrs()
        attrs = {sym_sh: strides[0], sym_sw: strides[1], sym_ph: padh, sym_pw: padw}
        return ret, attrs

    def infer_shape(self, input_shapes: List[SymShapeT]):
        (ni, *isizes, ci), (*wsizes, ci_w, co_w) = input_shapes
        # fmt: off
        cons = [
            (ni, "N"), (ci, "Cin"), (ci_w, "Cin"), (co_w, "Cout"),
            (isizes[0], "H"), (isizes[1], "W"),
            (wsizes[0], "KH"), (wsizes[1], "KW")
        ]
        # fmt: on
        sym_strides, sym_pads = self.make_attrs()
        output_shape = [
            (isizes[i] - 1) * sym_strides[i] - 2 * sym_pads[i] + (wsizes[i] - 1) + 1
            for i in range(2)
        ]
        return [ni, *output_shape, co_w], cons

    def make_attrs(self, config=None):
        strides = [self._mog(f"Stride{s}", config) for s in ("H", "W")]
        paddings = [self._mog(f"Pad{s}", config) for s in ("H", "W")]
        return strides, paddings

    def __str__(self) -> str:
        return "TransposeConv2d"

    __repr__ = __str__


class Dense(RelayComputeOpBuilder):
    @classmethod
    def lookahead(cls):
        return 1

    def make_attrs(self, _):
        return []

    def make_te(
        self, _: Optional[Dict[str, int]], input: te.Tensor, weight: te.Tensor
    ) -> Optional[te.Tensor]:
        return topi.nn.dense(input, weight)  # type: ignore

    @classmethod
    def try_from_ops(cls, op0: te.ComputeOp):
        if op0.tag != "dense":
            return None
        return cls(), {}

    def infer_shape(self, input_shapes: List[SymShapeT]):
        input, weight = input_shapes
        n, ki = input
        mw, kw = weight
        cons = [(ki, "K"), (kw, "K"), (n, "N"), (mw, "M")]
        output_shape = [n, mw]
        return output_shape, cons

    def __str__(self) -> str:
        return "Dense"

    __repr__ = __str__


class BatchMatmul(RelayComputeOpBuilder):
    @classmethod
    def lookahead(cls) -> int:
        return 1

    def make_attrs(self, _):
        return []

    def make_te(
        self, _: Optional[Dict[str, int]], input: te.Tensor, weight: te.Tensor
    ) -> Optional[te.Tensor]:
        return topi.nn.batch_matmul(input, weight)  # type: ignore

    @classmethod
    def try_from_ops(cls, op0: te.ComputeOp):
        def get_name(idx):
            return idx.name if isinstance(idx, tir.Var) else None

        if op0.tag != "batch_matmul":
            return None
        expr = op0.body[0].source[0]
        assert isinstance(expr, tir.Mul)
        lhs, rhs = expr.a, expr.b
        lhs_names = [get_name(x) for x in lhs.indices]
        rhs_names = [get_name(x) for x in rhs.indices]
        if lhs_names != ["b", "i", "k"] or rhs_names != ["b", "j", "k"]:
            return None
        return cls(), {}

    def infer_shape(self, input_shapes: List[SymShapeT]):
        input, weight = input_shapes
        bi, n, ki = input
        bw, m, kw = weight
        cons = [(bi, "B"), (bw, "B"), (ki, "K"), (kw, "K"), (n, "N"), (m, "M")]
        output_shape = [bi, n, m]
        return output_shape, cons

    def __str__(self) -> str:
        return "BatchMatmul"

    __repr__ = __str__


class FixedSizePool(RelayComputeOpBuilder, abc.ABC):
    OPS = {1: topi.nn.pool1d, 2: topi.nn.pool2d, 3: topi.nn.pool3d}

    def __init__(self, n_dims: int, is_avg: bool, has_paddings: bool) -> None:
        super().__init__()
        self.n_dims = n_dims
        self.is_avg = is_avg
        self.has_paddings = has_paddings

    def make_te(
        self, config: Dict[str, Union[int, tir.SizeVar]], input: te.Tensor
    ) -> Optional[te.Tensor]:
        op = self.OPS[self.n_dims]  # type: ignore
        layout = "".join(DIM_NAMES[self.n_dims])
        layout = f"N{layout}C"
        strides, pool_sizes, paddings = self.make_attrs(config)
        paddings = paddings * 2  # Padding on both sides.
        dilations = [1] * self.n_dims
        pool_type = "avg" if self.is_avg else "max"
        # fmt: off
        return op(
            input, pool_sizes, strides, dilations, paddings,
            pool_type=pool_type, layout=layout
        )
        # fmt: on

    def infer_shape(self, input_shapes: List[SymShapeT]):
        ((ni, *isizes, c),) = input_shapes
        cons = [(ni, "N"), (c, "C")]
        for i in range(self.n_dims):
            cons.append((isizes[i], f"In{i}"))
        strides, pool_sizes, paddings = self.make_attrs()
        output_shape = conv_like_shape_infer(isizes, pool_sizes, paddings, strides)
        output_shape = [ni, *output_shape, c]
        return output_shape, cons

    def _make_attrs_dict(self, strides, kernel_sizes, paddings=None):
        sym_strides, sym_ksizes, sym_paddings = self.make_attrs()
        attrs = {}
        for i in range(len(strides)):
            attrs[sym_strides[i]] = strides[i]
            attrs[sym_ksizes[i]] = kernel_sizes[i]
        if self.has_paddings:
            assert paddings is not None
            for i in range(len(strides)):
                attrs[sym_paddings[i]] = paddings[i]
        else:
            assert paddings is None
        return attrs

    def make_attrs(self, config=None):
        ds = DIM_NAMES[self.n_dims]
        strides = [self._mog(f"Stride{ds[i]}", config) for i in range(self.n_dims)]
        kernel_sizes = [self._mog(f"KSize{i}", config) for i in range(self.n_dims)]
        if self.has_paddings:
            paddings = [self._mog(f"Pad{ds[i]}", config) for i in range(self.n_dims)]
        else:
            paddings = [0] * self.n_dims
        return strides, kernel_sizes, paddings

    @classmethod
    def _parse_pool_attrs(cls, op0: te.ComputeOp):
        def parse_extent(ivar: tir.IterVar) -> int:
            extent = ivar.dom.extent
            if isinstance(extent, tir.Sub):
                return maybe_mul(extent.b)
            if isinstance(extent, tir.IntImm):
                return extent.value
            raise ValueError(f"Unsupported pooling extent {extent}")

        dim_indices = op0.body[0].source[0].indices[1:-1]
        reduce_axes = {
            itervar.var: parse_extent(itervar) for itervar in op0.reduce_axis  # type: ignore
        }
        strides, kernel_sizes = [], []
        for index in dim_indices:
            if not isinstance(index, tir.Add):
                raise ValueError(f"Unsupported pooling index {index}")
            lhs, rhs = index.a, index.b
            if isinstance(lhs, tir.Mul) and isinstance(lhs.b, tir.IntImm):
                stride, reduce_var = lhs.b, rhs
            elif isinstance(lhs, tir.Var):
                stride, reduce_var = tir_int(1), rhs
            else:
                raise ValueError(f"Unsupported pooling index {index}")
            kernel_sizes.append(reduce_axes[reduce_var])
            strides.append(stride)
        return strides, kernel_sizes

    @classmethod
    def lookahead(cls) -> int:
        return 0

    @classmethod
    def try_from_ops(cls, *ops):
        raise NotImplementedError()

    def __str__(self) -> str:
        ret = f"{'Avg' if self.is_avg else 'Max'}{self.n_dims}DPool"
        if self.has_paddings:
            ret += "[Padded]"
        return ret

    __repr__ = __str__


class OneOPPool(RelayComputeOpBuilder):
    @classmethod
    def lookahead(cls) -> int:
        return 1

    @classmethod
    def try_from_ops(cls, op0: te.ComputeOp):
        if op0.tag != "pool_max":
            return None
        strides, kernel_sizes = FixedSizePool._parse_pool_attrs(op0)
        ret = FixedSizePool(len(strides), is_avg=False, has_paddings=False)
        attrs = ret._make_attrs_dict(strides, kernel_sizes)
        return ret, attrs


class TwoOPsPool(RelayComputeOpBuilder):
    @classmethod
    def lookahead(cls) -> int:
        return 2

    @classmethod
    def try_from_ops(cls, op0, op1):
        if (paddings := detect_pad_op(op0)) is not None and op1.tag == "pool_max":
            strides, kernel_sizes = FixedSizePool._parse_pool_attrs(op1)
            if paddings[: len(strides)] != paddings[len(strides) :]:
                raise ValueError("Unsupported asymmetric padding")
            inst = FixedSizePool(len(strides), is_avg=False, has_paddings=True)
            attrs = inst._make_attrs_dict(strides, kernel_sizes, paddings)
            return inst, attrs
        if op0.tag == "pool_sum" and op1.tag == "elemwise":
            strides, kernel_sizes = FixedSizePool._parse_pool_attrs(op0)
            inst = FixedSizePool(len(strides), is_avg=True, has_paddings=False)
            attrs = inst._make_attrs_dict(strides, kernel_sizes)
            return inst, attrs
        return None


class ThreeOPsPool(RelayComputeOpBuilder):
    @classmethod
    def lookahead(cls) -> int:
        return 3

    @classmethod
    def try_from_ops(cls, op0, op1, op2):
        if (
            (paddings := detect_pad_op(op0)) is not None
            and op1.tag == "pool_sum"
            and op2.tag == "elemwise"
        ):
            strides, kernel_sizes = FixedSizePool._parse_pool_attrs(op1)
            if paddings[: len(strides)] != paddings[len(strides) :]:
                raise ValueError("Unsupported asymmetric padding")
            inst = FixedSizePool(len(strides), is_avg=True, has_paddings=True)
            attrs = inst._make_attrs_dict(strides, kernel_sizes, paddings)
            return inst, attrs
        return None


class AdaptiveAvgPool(RelayComputeOpBuilder):
    def __init__(self, n_dims: int) -> None:
        super().__init__()
        self.n_dims = n_dims

    @classmethod
    def lookahead(cls) -> int:
        return 2

    @classmethod
    def try_from_ops(cls, op0, op1):
        if op0.tag != "adaptive_pool_sum" or op1.tag != "elemwise":
            return None
        output_sizes = op0.output(0).shape[1:-1]
        if len(output_sizes) not in (2, 3):
            return None
        self = cls(len(output_sizes))
        (osizes,) = self.make_attrs()
        return self, dict(zip(osizes, output_sizes))

    def make_te(
        self, config: Dict[str, Union[int, tir.SizeVar]], input: te.Tensor
    ) -> Optional[te.Tensor]:
        layout = f"N{DIM_NAMES[self.n_dims]}C"
        (osizes,) = self.make_attrs(config)
        if self.n_dims == 2:
            return topi.nn.adaptive_pool(input, osizes, "avg", layout)
        elif self.n_dims == 3:
            return topi.nn.adaptive_pool3d(input, osizes, "avg", layout)
        assert False

    def infer_shape(self, input_shapes: List[SymShapeT]):
        (ishape,) = input_shapes
        (n, *_, c) = ishape
        (oshape,) = self.make_attrs()
        idims_name = ["N"] + [f"In{c}" for c in DIM_NAMES[self.n_dims]] + ["C"]
        return [n, *oshape, c], list(zip(ishape, idims_name))

    def make_attrs(self, config=None):
        return [[self._mog(f"Out{c}", config) for c in DIM_NAMES[self.n_dims]]]

    def __str__(self) -> str:
        return f"AdaptiveAvgPool{self.n_dims}D"

    __repr__ = __str__


class Softmax(RelayComputeOpBuilder):
    def __init__(self, n_dims: int, axis: int) -> None:
        super().__init__()
        self.n_dims = n_dims
        self.axis = axis

    @classmethod
    def lookahead(cls) -> int:
        return 4

    def make_attrs(self, _):
        return []

    def make_te(
        self, _: Dict[str, Union[int, tir.SizeVar]], input: te.Tensor
    ) -> Optional[te.Tensor]:
        return topi.nn.softmax(input, self.axis)  # type: ignore

    @classmethod
    def try_from_ops(cls, *ops: te.ComputeOp):
        assert len(ops) == 4
        if any(op.tag != "softmax_output" for op in ops):
            return None
        n_dims = len(ops[0].input_tensors[0].shape)
        body = ops[0].body[0]
        if len(body.axis) != 1:
            return None
        axis_var = body.axis[0].var
        axis = list(body.source[0].indices).index(axis_var)
        if axis == -1:
            return None
        return cls(n_dims, axis), {}

    def infer_shape(self, input_shapes: List[SymShapeT]):
        (shape,) = input_shapes
        assert len(shape) == self.n_dims
        cons = [(shape[i], "Nx" if i == self.axis else f"N{i}") for i in range(self.n_dims)]
        return shape, cons

    def __str__(self) -> str:
        return "Softmax"

    __repr__ = __str__


class ConstScalar(RelayComputeOpBuilder):
    def __init__(self, value: Union[tir.IntImm, tir.FloatImm]) -> None:
        super().__init__()
        self.value = value

    @classmethod
    def lookahead(cls) -> int:
        return 1

    def make_attrs(self, _):
        return []

    def make_te(self, config: Dict[str, Union[int, tir.SizeVar]]) -> Optional[te.Tensor]:
        return te.compute([], lambda: self.value, "compile_engine_const", "broadcast")  # type: ignore

    @classmethod
    def try_from_ops(cls, op):
        if op.name != "compile_engine_const" or len(op.body) != 1:
            return None
        return cls(op.body[0]), {}

    def infer_shape(self, input_shapes: List[SymShapeT]):
        assert len(input_shapes) == 0
        return [], []

    def __str__(self) -> str:
        return "ConstScalar"

    def __repr__(self) -> str:
        return f"ConstScalar({self.value})"


class Mean(RelayComputeOpBuilder):
    def __init__(self, n_dims: int) -> None:
        super().__init__()
        self.n_dims = n_dims

    def make_te(
        self, _: Dict[str, Union[int, tir.SizeVar]], input: te.Tensor
    ) -> Optional[te.Tensor]:
        sum_ = topi.sum(input, axis=self.n_dims - 1, keepdims=True)
        return topi.divide(sum_, input.shape[-1])

    @classmethod
    def lookahead(cls) -> int:
        return 2

    def make_attrs(self, _):
        return []

    @classmethod
    def try_from_ops(cls, op0, op1):
        if op0.tag != "comm_reduce":
            return None
        body = op0.body[0]
        assert isinstance(body, tir.Reduce)
        if not isinstance(body.combiner.result[0], tir.Add):
            return None
        if len(op0.body[0].source) != 1:
            return None
        if op1.name != "T_divide" or op1.output(0).shape[-1] != 1:
            return None
        input0 = op0.body[0].source[0]
        return cls(len(input0.indices)), {}

    def infer_shape(self, input_shapes: List[SymShapeT]):
        (shape,) = input_shapes
        assert len(shape) == self.n_dims
        cons = [(shape[i], f"N{i}") for i in range(self.n_dims)]
        out_shape = shape.copy()
        out_shape[-1] = sp.Integer(1)
        return out_shape, cons

    def __repr__(self) -> str:
        return f"Mean({self.n_dims})"

    def __str__(self) -> str:
        return "Mean"


class Elemwise(RelayComputeOpBuilder):
    def __init__(self, topi_f, kwargs) -> None:
        super().__init__()
        self.topi_f = topi_f
        self.kwargs = kwargs

    def make_te(self, _: Optional[Dict[str, int]], input: te.Tensor) -> Optional[te.Tensor]:
        return self.topi_f(input, **self.kwargs)

    @classmethod
    def lookahead(cls) -> int:
        return 1

    def make_attrs(self, _):
        return []

    @classmethod
    def try_from_ops(cls, op: te.ComputeOp):
        if op.tag != "elemwise":
            return None
        if (func := cls.find_te_func(op)) is None:
            return None
        te_func, kwargs = func
        return cls(te_func, kwargs), {}

    def infer_shape(self, input_shapes: List[SymShapeT]):
        if len(input_shapes) > 1:
            raise NotImplementedError("Broadcasting not supported")
        return input_shapes[0], []

    @classmethod
    def find_te_func(cls, op: te.ComputeOp):
        def is_atom(expr):
            return isinstance(expr, (tir.ProducerLoad, tir.IntImm, tir.FloatImm))

        def get_number(expr):
            if isinstance(expr, (tir.IntImm, tir.FloatImm)):
                return expr.value
            return None

        def simple_binop(expr, opcode):
            if not isinstance(expr, opcode):
                return False
            return is_atom(expr.a) and is_atom(expr.b)

        expr = op.body[0]
        if (
            isinstance(expr, tir.Max)
            and isinstance(expr.a, tir.Min)
            and is_atom(expr.a.a)
            and (min := get_number(expr.b)) is not None
            and (max := get_number(expr.a.b)) is not None
        ):
            return topi.clip, {"a_min": min, "a_max": max}
        elif op.name == "T_relu":
            return topi.nn.relu, {}
        elif op.name == "T_sigmoid":
            return topi.sigmoid, {}
        elif op.name == "T_leaky_relu":
            alpha = expr.false_value.b.value
            return topi.nn.leaky_relu, {"alpha": alpha}
        elif simple_binop(expr, tir.Max):
            return topi.maximum, {"rhs": expr.b}
        elif simple_binop(expr, tir.Min):
            return topi.minimum, {"rhs": expr.b}
        elif simple_binop(expr, tir.Div):
            return topi.divide, {"rhs": expr.b}
        elif (
            isinstance(expr, tir.Call)
            and len(expr.args) == 1
            and is_atom(expr.args[0])
            and expr.op.name == "tir.tanh"
        ):
            return topi.tanh, {}
        elif op.name == "T_fast_tanh":
            return topi.tanh, {}
        return None

    def __str__(self) -> str:
        return f"Elemwise({self.topi_f.__name__}...)"

    def __repr__(self) -> str:
        if self.kwargs:
            kwargs = "; " + ",".join(f"{k}={v}" for k, v in self.kwargs.items())
        else:
            kwargs = ""
        return f"Elemwise({self.topi_f.__name__}{kwargs})"


class Broadcast(RelayComputeOpBuilder):
    OP2OP = {
        "Add": topi.add,
        "Sub": topi.subtract,
        "Mul": topi.multiply,
        "Div": topi.divide,
        "tir.pow": topi.power,
    }

    def __init__(
        self,
        op_code: str,
        n_dims: List[int],
        iter_vars: Dict[str, List[Tuple[int, int]]],
        one_sized_dims: List[Tuple[int, int]],
        output_shape: List[str],
    ) -> None:
        super().__init__()
        self.op_code = op_code
        self.n_dims = n_dims
        self.iter_vars = iter_vars
        self.one_sized_dims = one_sized_dims
        self.output_shape = output_shape

    @classmethod
    def lookahead(cls) -> int:
        return 1

    def make_te(self, _: Optional[Dict[str, int]], *args: te.Tensor) -> Optional[te.Tensor]:
        assert len(args) == len(self.n_dims)
        op = self.OP2OP[self.op_code]
        if len(args) == 1:
            return op(args[0], args[0])
        if len(args) == 2:
            lhs, rhs = args
            return op(lhs, rhs)
        raise ValueError(f"Unsupported number of args {len(args)}")

    def make_attrs(self, _):
        return []

    @classmethod
    def try_from_ops(cls, op: te.ComputeOp):
        if op.tag != "broadcast" or len(op.body) != 1:
            return None
        if len(op.input_tensors) == 0:
            return None  # Evade cases that ConstTensor will handle.
        stmt0 = op.body[0]
        if isinstance(stmt0, tir.Call):
            # For example, pow(a, b).
            input_acc_indices = [arg.indices for arg in stmt0.args]
            op_code = stmt0.op.name
        elif len(op.input_tensors) == 1:
            # This means lhs and rhs are the same.
            # Can happen when e.g. it's a square op (x*x).
            input_acc_indices = [stmt0.a.indices]
            op_code = type(stmt0).__name__
        elif len(op.input_tensors) == 2:
            input_acc_indices = [stmt0.a.indices, stmt0.b.indices]
            op_code = type(stmt0).__name__
        else:
            raise ValueError(f"Unsupported broadcast operator {stmt0}")
        iter_var_occur = defaultdict(list)
        one_sized_dims = []
        for input_idx, acc_indices in enumerate(input_acc_indices):
            for dim_idx, index in enumerate(acc_indices):
                key = input_idx, dim_idx
                if isinstance(index, tir.IntImm):
                    # This is a 1-sized dimension
                    assert index.value == 0
                    one_sized_dims.append(key)
                elif isinstance(index, tir.Var):
                    iter_var_occur[index.name].append(key)
                else:
                    raise ValueError(f"Unsupported broadcast index {index}")
        n_dims = [len(acc_indices) for acc_indices in input_acc_indices]
        output_shape = [axis.var.name for axis in op.axis]  # type: ignore
        ret = cls(op_code, n_dims, iter_var_occur, one_sized_dims, output_shape)
        return ret, {}

    def infer_shape(self, input_shapes: List[SymShapeT]):
        assert len(input_shapes) == len(self.n_dims)
        for input_shape, n_dims in zip(input_shapes, self.n_dims):
            assert len(input_shape) == n_dims
        cons = []
        for input_idx, dim_idx in self.one_sized_dims:
            size = input_shapes[input_idx][dim_idx]
            cons.append((size, 1))
        for indices in self.iter_vars.values():
            for (iidx0, didx0), (iidx1, didx1) in zip(indices, indices[1:]):
                size0 = input_shapes[iidx0][didx0]
                size1 = input_shapes[iidx1][didx1]
                cons.append((size0, size1))
        output_shape = []
        for var in self.output_shape:
            # We have requested all expressions in self.iter_vars[var]
            # be equal, so we can just pick the first one.
            iidx, didx = self.iter_vars[var][0]
            output_shape.append(input_shapes[iidx][didx])
        return output_shape, cons

    def __str__(self) -> str:
        return f"Broadcast({self.op_code}...)"

    def __repr__(self) -> str:
        def print_list(xs):
            return "[" + ",".join(str(x) for x in xs) + "]"

        input_shapes: List[List[Any]] = [[None] * n for n in self.n_dims]
        for ax, indices in self.iter_vars.items():
            for input_idx, dim_idx in indices:
                input_shapes[input_idx][dim_idx] = ax
        for input_idx, dim_idx in self.one_sized_dims:
            input_shapes[input_idx][dim_idx] = 1
        printed = []
        for shape in input_shapes:
            assert all(x is not None for x in shape)
            printed.append(print_list(shape))
        printed = " x ".join(printed) + " -> " + print_list(self.output_shape)
        return f"Broadcast({self.op_code}; {printed})"


# fmt: off
RELAY_BUILDERS: List[Type[RelayComputeOpBuilder]] = [
    Conv, DepthwiseConv2d, GroupConv2d, TransposeConv2d, Conv2dTensorCore, Conv2dWinograd,
    Dense, BatchMatmul,
    # OneOPPool, TwoOPsPool, ThreeOPsPool are just used as parsers
    # and do not return instances of themselves.
    # trunk-ignore(mypy/type-abstract)
    OneOPPool, TwoOPsPool, ThreeOPsPool, AdaptiveAvgPool,
    Softmax, Mean,
    ConstScalar, Elemwise, Broadcast,
]
# fmt: on


def get_inputs_outputs(
    ops: List[te.ComputeOp],
) -> Tuple[List[te.Tensor], List[te.Tensor]]:
    all_ios = [
        (list(op.input_tensors), [op.output(i) for i in range(op.num_outputs)]) for op in ops
    ]
    inputs, input_set, output_set = [], set(), set()
    for inputs_, outputs_ in all_ios:
        for input in inputs_:
            if input not in input_set and input not in output_set:
                inputs.append(input)
        input_set.update(inputs_)
        output_set.update(outputs_)
    outputs, input_set, output_set = [], set(), set()
    for inputs_, outputs_ in reversed(all_ios):
        for output in outputs_:
            if output not in output_set and output not in input_set:
                outputs.append(output)
        input_set.update(inputs_)
        output_set.update(outputs)
    return inputs, outputs


def _detect_total_padding(op: te.ComputeOp):
    def is_vsc(x):  # var - const
        return isinstance(x, tir.Sub) and isinstance(x.a, tir.Var) and isinstance(x.b, tir.IntImm)

    def check_stride(x):
        if isinstance(x, tir.Var) or is_vsc(x):
            return 1
        if isinstance(x, tir.FloorDiv) and is_vsc(x.a) and isinstance(x.b, tir.IntImm):
            return x.b.value
        return None

    def total_padding():
        output_shape = op.output(0).shape
        input_shape = op.input_tensors[0].shape
        return [o - i for i, o in zip(input_shape, output_shape)]

    body = op.body[0]
    if isinstance(body, tir.ProducerLoad):
        if not all(isinstance(x, tir.Var) for x in body.indices):
            return None
        padding = total_padding()
        assert [p == 0 for p in padding]
        return [1] * len(padding), [0] * len(padding)  # deconv_stride, padding
    if isinstance(body, tir.Call) and body.op.name == "tir.if_then_else":
        strides = [check_stride(index) for index in body.args[1].indices]
        if any(s is None for s in strides):
            return None
        padding = total_padding()
        return strides, padding


def detect_pad_op(op: te.ComputeOp):
    if (detection := _detect_total_padding(op)) is None:
        return None
    strides, padding = detection
    if any(s != 1 for s in strides):
        return None
    # Assume equal padding on both sides
    if not all(p % 2 == 0 for p in padding):
        return None
    padding = [p // 2 for p in padding]
    if padding[0] == 0 and padding[-1] == 0:
        # N(*D)C: NWC, NHWC, NCDHW, etc.
        return padding[1:-1] + padding[1:-1]
    elif padding[0] == 0 and padding[1] == 0:
        # NC(*D): NCW, NCHW, NCDHW, etc.
        return padding[2:] + padding[2:]
    else:
        logger.warning(f"Unsupported format detected from padding {padding}")
        return None


def detect_deconv_pad_op(op: te.ComputeOp):
    if (detection := _detect_total_padding(op)) is None:
        return None
    strides, _ = detection
    return strides


def parse_conv_stride_group(op: te.ComputeOp):
    def parse_add_mul(x):
        if isinstance(x, tir.Add):
            return maybe_mul(x.a)
        return maybe_mul(x)

    pad_input_idxs = op.body[0].source[0].a.indices
    hw_idxs, ch_idx = pad_input_idxs[1:-1], pad_input_idxs[-1]
    strides = [parse_add_mul(i) for i in hw_idxs]
    return strides, parse_add_mul(ch_idx)


def parse_deconv_stride(op: te.ComputeOp):
    def parse_add_div(x):
        if isinstance(x, tir.FloorDiv):
            assert isinstance(x.b, tir.IntImm)
            return x.b.value
        return 1

    if not isinstance((src0 := op.body[0].source[0]), tir.Mul):
        return None
    if not isinstance((lhs := src0.a), tir.Call):
        return None
    pad_input_idxs = lhs.args[1].indices
    hw_idxs = pad_input_idxs[1:-1]
    return [parse_add_div(i) for i in hw_idxs]


def conv_like_shape_infer(isizes, wsizes, paddings, strides):
    assert len(isizes) == len(wsizes) == len(paddings) == len(strides)
    return [
        sp.floor((isizes[i] + 2 * paddings[i] - wsizes[i]) / strides[i] + 1)
        for i in range(len(isizes))
    ]


def maybe_mul(x) -> int:
    if isinstance(x, tir.Mul) and isinstance(x.b, tir.IntImm):
        return x.b.value
    elif isinstance(x, tir.Var):
        return 1
    raise ValueError(f"Unsupported expr {x}")


def tvm_data_to_python(data) -> Any:
    from tvm.ir.container import Array
    from tvm.runtime.container import String
    from tvm.tir import IntImm

    do = tvm_data_to_python
    if isinstance(data, IntImm):
        return int(data)
    if isinstance(data, String):
        return str(data)
    if isinstance(data, Array):
        return [do(x) for x in data]
    if isinstance(data, dict):
        return {do(k): do(v) for k, v in data.items()}
    return data


def tir_config_var(name: str):
    return tir.SizeVar(name, "int32", None, True)


def tir_int(v: int):
    return tir.IntImm("int32", v)


def _visit_sym_binary(ctor):
    def visit(self, e):
        a, b = e.args
        return ctor(self(a), self(b))

    return visit


class SymEngineToTir:
    def __init__(self, name_to_vars: Dict[str, tir.SizeVar]) -> None:
        super().__init__()
        self.name_to_vars = name_to_vars

    def is_type_relevant(self, t: Type) -> bool:
        return issubclass(t, sp.Basic)

    def __call__(self, e: sp.Expr) -> tir.PrimExpr:
        def rec(t: Type):
            if not self.is_type_relevant(t):
                return []
            return [t] + [rty for bty in t.__bases__ for rty in rec(bty)]

        bases = set(rec(type(e)))
        visitor_names = [f"visit_{base.__name__.lower()}" for base in bases]
        visitor = None
        for name in visitor_names:
            visitor = getattr(self, name, None)
            if visitor is not None:
                break
        if visitor is None:
            raise ValueError(
                f"""Unsupported value {e} of type {type(e)}.
Tried (and didn't find): {visitor_names}"""
            )
        return visitor(e)

    def visit_symbol(self, e) -> tir.PrimExpr:
        return self.name_to_vars[e.name]

    def visit_rational(self, e) -> tir.PrimExpr:
        return tir.div(tir_int(e.p), tir_int(e.q))

    def visit_integer(self, e) -> tir.IntImm:
        return tir_int(e.p)

    def visit_add(self, e) -> tir.PrimExpr:
        adds, subs = [], []
        for arg in e.args:
            if isinstance(arg, sp.Mul) and arg.args[0] == -1:
                subs.append(self(arg.args[1]))
            else:
                adds.append(self(arg))
        zero = tir_int(0)
        add = self._reduce(tir.Add, adds, zero)
        if subs:
            sub = self._reduce(tir.Add, subs, zero)
            return tir.Sub(add, sub)
        else:
            return add

    def visit_mul(self, e) -> tir.PrimExpr:
        numers, denoms = [], []
        for arg in e.args:
            if isinstance(arg, sp.Pow) and arg.args[1] == -1:
                denoms.append(self(arg.args[0]))
            else:
                numers.append(self(arg))
        one = tir_int(1)
        numer = self._reduce(tir.Mul, numers, one)
        if denoms:
            denom = self._reduce(tir.Mul, denoms, one)
            return tir.Div(numer, denom)
        else:
            return numer

    def visit_floor(self, e) -> tir.PrimExpr:
        (a,) = e.args
        a = self(a)  # Easier to switch on converted value
        if isinstance(a, tir.Mod):
            return tir.FloorMod(a.a, a.b)
        elif isinstance(a, tir.Div):
            return tir.FloorDiv(a.a, a.b)
        else:
            return tir.floor(a)

    visit_equality = _visit_sym_binary(tir.EQ)
    visit_unequality = _visit_sym_binary(tir.NE)
    visit_strictlessthan = _visit_sym_binary(tir.LT)
    visit_lessthan = _visit_sym_binary(tir.LE)
    visit_strictgreaterthan = _visit_sym_binary(tir.GT)
    visit_greaterthan = _visit_sym_binary(tir.GE)

    def _reduce(
        self, op: Type, args: List[tir.PrimExpr], unit: Optional[tir.IntImm] = None
    ) -> tir.PrimExpr:
        if len(args) == 0:
            if unit is None:
                raise ValueError("Cannot reduce empty list without unit")
            return unit
        if len(args) == 1:
            return args[0]
        return reduce(op, args)
