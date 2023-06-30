import sys
from typing import Dict, List, NamedTuple, Union

import pytest
import torch
import tvm
import tvm.testing
from torchvision.models import mobilenet_v2, resnet50
from torchvision.models.video import r3d_18
from tvm import felix
from tvm.felix import ffi


def dcgan():
    from tvm.felix import nn

    network, shape = nn.dcgan()
    return network, [1, *shape]


NETWORKS = {
    "resnet50": lambda: (resnet50(), (1, 3, 224, 224)),
    "mobilenet_v2": lambda: (mobilenet_v2(), (1, 3, 224, 224)),
    "r3d_18": lambda: (r3d_18(), (1, 3, 16, 112, 112)),
    "dcgan": dcgan,
}
# fmt: off
NETWORK_TASKS = {
    "resnet50": {
        "Const,AdaptiveAvgPool2D": [0],
        "Const,Const,Const,Conv2d,Broadcast(Add...)": list(range(1, 4 + 1)),
        "Const,Const,Const,Const,Conv2d,Broadcast(Add...),Broadcast(Add...),Elemwise(relu...)": list(range(5, 8 + 1)),
        "Const,Const,Const,Conv2d,Broadcast(Add...),Elemwise(relu...)": list(range(9, 24 + 1)),
        "Const,Const,Const,Dense,Broadcast(Add...)": [25],
        "Const,Max2DPool[Padded]": [26],
    },
    "mobilenet_v2": {
        "Const,AdaptiveAvgPool2D": [0],
        "Const,Const,Const,Conv2d,Broadcast(Add...)": list(range(1, 7 + 1)),
        "Const,Const,Const,Const,Conv2d,Broadcast(Add...),Broadcast(Add...)": [8, 9, 10, 11, 12],
        "Const,Const,Const,Conv2d,Broadcast(Add...),Elemwise(clip...)": [13, 15, 17, 20, 23, 25, 28, 30],
        "Const,Const,Const,DepthwiseConv2d,Broadcast(Add...),Elemwise(clip...)": [14, 16, 18, 19, 21, 22, 24, 26, 27, 29],
        "Const,Const,Const,Dense,Broadcast(Add...)": [31],
    },
    "r3d_18": {
        "Const,Const,Const,Dense,Broadcast(Add...)": [0],
        "Const,AdaptiveAvgPool3D": [1],
        "Const,Const,Const,Const,Conv3d,Broadcast(Mul...),Broadcast(Add...)": [2, 3, 4],
        "Const,Const,Const,Const,Const,Conv3d,Broadcast(Mul...),Broadcast(Add...),Broadcast(Add...),Elemwise(relu...)": [5, 6, 7, 8],
        "Const,Const,Const,Const,Conv3d,Broadcast(Mul...),Broadcast(Add...),Elemwise(relu...)": list(range(9, 16 + 1))
    },
    "dcgan": {
        "Const,Const,Conv2d,Elemwise(sigmoid...)": [0],
        "Const,Const,Const,Conv2d,Broadcast(Add...),Elemwise(leaky_relu...)": [1, 2, 3],
        "Const,Const,Conv2d,Elemwise(leaky_relu...)": [4],
        "Const,Const,Const,Const,TransposeConv2d,Broadcast(Mul...),Broadcast(Add...),Elemwise(relu...)": [5, 6, 7, 8],
        "Const,Const,TransposeConv2d,Elemwise(tanh...)": [9],
    }
}
# fmt: on


class SketchCodeDesc:
    def __init__(self, backbone: tuple, loop_bounds: Dict[str, str]):
        self.backbone = backbone
        self.loop_bounds = {k: ffi.parse_expr_preorder(v) for k, v in loop_bounds.items()}

    def make_desc(self):
        return self


class SketchCodeRef:
    def __init__(
        self,
        base_op: str,
        base_sketch: int,
        prepend_steps: List[str],
        overwrite_loop_bounds: Dict[str, str],
    ):
        self.base_op = base_op
        self.base_sketch = base_sketch
        self.prepend_steps = prepend_steps
        self.overwrite_loop_bounds = {
            k: ffi.parse_expr_preorder(v) for k, v in overwrite_loop_bounds.items()
        }

    def make_desc(self):
        sketch = TASK_SKETCHES[self.base_op][self.base_sketch].make_desc()
        assert isinstance(sketch, SketchCodeDesc)
        backbone = tuple(self.prepend_steps + list(sketch.backbone))
        ret = SketchCodeDesc(backbone, {})
        ret.loop_bounds = {**sketch.loop_bounds, **self.overwrite_loop_bounds}
        return ret


TASK_SKETCHES: Dict[str, List[Union[SketchCodeDesc, SketchCodeRef]]] = {
    "Const,Max2DPool[Padded]": [
        SketchCodeDesc(
            ("FU", "SP", "AN", "AN", "PR"),
            {
                "v0": "(/ (* (* (* N:i (+ (// (- (+ (* PadH:i 2) In0:i) KSize0:i) StrideH:i) 1)) (+ (// (- (+ (* PadW:i 2) In1:i) KSize1:i) StrideW:i) 1)) C:i) sp_0_0:i)",
                "v1": "(+ N:i (* N:i (// (- (+ (* PadH:i 2) In0:i) KSize0:i) StrideH:i)))",
                "v2": "(+ v1:i (* v1:i (// (- (+ (* PadW:i 2) In1:i) KSize1:i) StrideW:i)))",
                "blockIdx.x": "(+ (// (- (* v2:i C:i) 1) sp_0_0:i) 1)",
                "threadIdx.x": "sp_0_0:i",
                "tensor/rv0": "KSize0:i",
                "tensor/rv1": "KSize1:i",
            },
        )
    ],
    # AdaptiveAvgPool2D
    "Const,AdaptiveAvgPool2D": [
        SketchCodeDesc(
            ("FU", "SP", "AN", "AN", "FU", "SP", "AN", "AN", "PR"),
            {
                "v0": "(/ (* (* (* N:i OutH:i) OutW:i) C:i) sp_0_0:i)",
                "v1": "(/ (* (* (* N:i OutH:i) OutW:i) C:i) sp_1_0:i)",
                "blockIdx.x": "v0:i",
                "threadIdx.x": "sp_0_0:i",
                "tensor/rv0": "(- (select (== (mod (* (+ (mod (// (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) OutH:i) 1) InH:i) OutH:i) 0) (// (* (+ (mod (// (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) OutH:i) 1) InH:i) OutH:i) (+ (// (* (+ (mod (// (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) OutH:i) 1) InH:i) OutH:i) 1)) (// (* (mod (// (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) OutH:i) InH:i) OutH:i))",
                "tensor/rv1": "(- (select (== (mod (* (+ (mod (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) 1) InW:i) OutW:i) 0) (// (* (+ (mod (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) 1) InW:i) OutW:i) (+ (// (* (+ (mod (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) 1) InW:i) OutW:i) 1)) (// (* (mod (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) InW:i) OutW:i))",
            },
        ),
    ],
    # Conv2d/+
    "Const,Const,Const,Conv2d,Broadcast(Add...)": [
        SketchCodeDesc(
            ("FU", "SP", "AN", "FSP", "AN", "CA", "CI", "FU", "AN", "PR"),
            {
                "v0": "(/ Cout:i sp_0_0:i)",
                "v1": "(+ N:i (* N:i (// (- (+ (* PadH:i 2) H:i) KH:i) StrideH:i)))",
                "v2": "(+ v1:i (* v1:i (// (- (+ (* PadW:i 2) W:i) KW:i) StrideW:i)))",
                "v3": "(// (- (+ (* (* KH:i KW:i) Cin:i) sp_0_0:i) 1) sp_0_0:i)",
                "blockIdx.x": "(* v2:i v0:i)",
                "Conv2dOutput/ff": "sp_0_0:i",
                "threadIdx.x": "sp_0_0:i",
                "Conv2dOutput/ry.rx.fused.rc.fused.outer": "v3:i",
            },
        ),
        SketchCodeDesc(
            # fmt: off
            ("SP", "SP", "SP", "SP", "SP", "SP", "SP", "RE", "FSP", "FSP", "FSP", "FSP", "RE", "CA", "CHR", "CA", "CHR", "CA", "CI", "FU", "AN", "FU", "AN", "FU", "AN", "FU", "SP", "AN", "FFSP", "AN", "FU", "SP", "AN", "FFSP", "AN", "PR"),
            # fmt: on
            {
                "v0": "(/ N:i (* (* (* sp_0_0:i sp_0_1:i) sp_0_2:i) sp_0_3:i))",
                "v1": "(/ (+ (// (- (+ (* PadH:i 2) H:i) KH:i) StrideH:i) 1) (* (* (* sp_1_0:i sp_1_1:i) sp_1_2:i) sp_1_3:i))",
                "v2": "(/ (+ (// (- (+ (* PadW:i 2) W:i) KW:i) StrideW:i) 1) (* (* (* sp_2_0:i sp_2_1:i) sp_2_2:i) sp_2_3:i))",
                "v3": "(/ Cout:i (* (* (* sp_3_0:i sp_3_1:i) sp_3_2:i) sp_3_3:i))",
                "v4": "(/ KH:i (* sp_4_0:i sp_4_1:i))",
                "v5": "(/ KW:i (* sp_5_0:i sp_5_1:i))",
                "v6": "(/ Cin:i (* sp_6_0:i sp_6_1:i))",
                "v7": "(/ (* (* (* (* (min KH:i sp_4_1:i) (min (+ (// (- KH:i 1) sp_4_1:i) 1) sp_4_0:i)) (* (min KW:i sp_5_1:i) (min (+ (// (- KW:i 1) sp_5_1:i) 1) sp_5_0:i))) (* (min Cin:i sp_6_1:i) (min (+ (// (- Cin:i 1) sp_6_1:i) 1) sp_6_0:i))) (* (min (min Cout:i (* sp_3_2:i sp_3_3:i)) sp_3_3:i) (min (+ (// (- (min Cout:i (* sp_3_2:i sp_3_3:i)) 1) sp_3_3:i) 1) sp_3_2:i))) sp_7_0:i)",
                "v8": "(* sp_4_1:i (* sp_5_1:i (* sp_4_0:i sp_5_0:i)))",
                "blockIdx.x": "(* (* (* v0:i v1:i) v2:i) v3:i)",
                "vthread": "(* (* (* sp_0_0:i sp_1_0:i) sp_2_0:i) sp_3_0:i)",
                "threadIdx.x": "(* (* (* sp_3_1:i sp_2_1:i) sp_1_1:i) sp_0_1:i)",
                "Conv2dOutput/nn.outer.inner.init": "sp_0_2:i",
                "Conv2dOutput/yy.outer.inner.init": "sp_1_2:i",
                "Conv2dOutput/xx.outer.inner.init": "sp_2_2:i",
                "Conv2dOutput/ff.outer.inner.init": "sp_3_2:i",
                "Conv2dOutput/nn.inner.init": "sp_0_3:i",
                "Conv2dOutput/yy.inner.init": "sp_1_3:i",
                "Conv2dOutput/xx.inner.init": "sp_2_3:i",
                "Conv2dOutput/ff.inner.init": "sp_3_3:i",
                "Conv2dOutput/ry.outer.outer": "v4:i",
                "Conv2dOutput/rx.outer.outer": "v5:i",
                "Conv2dOutput/rc.outer.outer": "v6:i",
                "PaddedInput.shared/ax0.ax1.fused.ax2.fused.ax3.fused.outer.outer": "(+ 1 (// (// (+ -1 (* (* sp_0_2:i sp_0_3:i) (* (* (+ (+ 1 (// (// (// -1 sp_3_1:i) sp_2_1:i) sp_1_1:i)) (* sp_0_1:i (+ sp_0_0:i (+ 1 (// (// (// -1 sp_3_0:i) sp_2_0:i) sp_1_0:i))))) (+ (* sp_4_1:i sp_4_0:i) (- (* sp_1_0:i (* (* sp_1_1:i StrideH:i) (* sp_1_2:i sp_1_3:i))) StrideH:i))) (* (+ (* sp_5_1:i sp_5_0:i) (- (* sp_2_0:i (* sp_2_1:i (* sp_2_2:i (* sp_2_3:i StrideW:i)))) StrideW:i)) (* sp_6_1:i sp_6_0:i))))) sp_8_0:i) (* sp_0_1:i (* sp_3_1:i (* sp_2_1:i sp_1_1:i)))))",
                "PaddedInput.shared/ax0.ax1.fused.ax2.fused.ax3.fused.inner": "(min (* (+ 1 (+ (// (// (// -1 sp_3_1:i) sp_2_1:i) sp_1_1:i) (* sp_0_1:i (+ sp_0_0:i (+ 1 (// (// (// -1 sp_3_0:i) sp_2_0:i) sp_1_0:i)))))) (* (* sp_0_2:i (* sp_0_3:i (+ (* sp_4_1:i sp_4_0:i) (- (* StrideH:i (* (* (* sp_1_1:i sp_1_3:i) sp_1_0:i) sp_1_2:i)) StrideH:i)))) (* (+ (* sp_5_1:i sp_5_0:i) (- (* (* sp_2_1:i StrideW:i) (* (* sp_2_0:i sp_2_3:i) sp_2_2:i)) StrideW:i)) (* sp_6_1:i sp_6_0:i)))) sp_8_0:i)",
                "placeholder.shared/ax0.ax1.fused.ax2.fused.ax3.fused.outer.outer": "(+ (// (// (- (* (* v8:i (* sp_6_1:i sp_6_0:i)) (* (* sp_3_2:i sp_3_3:i) (* sp_3_1:i sp_3_0:i))) 1) sp_7_0:i) (* (* (* sp_3_1:i sp_2_1:i) sp_1_1:i) sp_0_1:i)) 1)",
                "placeholder.shared/ax0.ax1.fused.ax2.fused.ax3.fused.inner": "(min (* (* v8:i (* sp_6_1:i sp_6_0:i)) (* (* sp_3_2:i sp_3_3:i) (* sp_3_1:i sp_3_0:i))) sp_7_0:i)",
                "placeholder.shared/ry.outer.inner": "sp_4_0:i",
                "placeholder.shared/rx.outer.inner": "sp_5_0:i",
                "placeholder.shared/rc.outer.inner": "sp_6_0:i",
                "placeholder.shared/nn.outer.inner": "sp_0_2:i",
                "placeholder.shared/yy.outer.inner": "sp_1_2:i",
                "placeholder.shared/xx.outer.inner": "sp_2_2:i",
                "placeholder.shared/ff.outer.inner": "sp_3_2:i",
                "placeholder.shared/ry.inner": "sp_4_1:i",
                "placeholder.shared/rx.inner": "sp_5_1:i",
                "placeholder.shared/rc.inner": "sp_6_1:i",
                "placeholder.shared/nn.inner": "sp_0_3:i",
                "placeholder.shared/yy.inner": "sp_1_3:i",
                "placeholder.shared/xx.inner": "sp_2_3:i",
                "placeholder.shared/ff.inner": "sp_3_3:i",
                "Conv2dOutput/ax0.inner": "(* sp_0_2:i sp_0_3:i)",
                "Conv2dOutput/ax1.inner": "(* sp_1_2:i sp_1_3:i)",
                "Conv2dOutput/ax2.inner": "(* sp_2_2:i sp_2_3:i)",
                "Conv2dOutput/ax3.inner": "(* sp_3_2:i sp_3_3:i)",
            }
        ),
    ],
    "Const,Const,Const,Dense,Broadcast(Add...)": [
        SketchCodeDesc(
            ("SP", "AN", "FSP", "AN", "CA", "FU", "AN", "PR"),
            {
                "v0": "(/ M:i sp_0_0:i)",
                "blockIdx.x": "(* N:i v0:i)",
                "T_matmul_NT/j": "sp_0_0:i",
                "threadIdx.x": "sp_0_0:i",
                "T_matmul_NT/k.outer": "(+ (// (- K:i 1) sp_0_0:i) 1)",
            },
        ),
        SketchCodeDesc(
            # fmt: off
            ("SP", "SP", "SP", "RE", "FSP", "FSP", "RE", "CA", "CHR", "CA", "CHR", "CA", "FU", "AN", "FU", "AN", "FU", "AN", "FU", "SP", "AN", "FFSP", "AN", "FU", "SP", "AN", "FFSP", "AN", "PR"),
            # fmt: on
            {
                "v0": "(/ N:i (* (* (* sp_0_0:i sp_0_1:i) sp_0_2:i) sp_0_3:i))",
                "v1": "(/ M:i (* (* (* sp_1_0:i sp_1_1:i) sp_1_2:i) sp_1_3:i))",
                "v2": "(/ K:i (* sp_2_0:i sp_2_1:i))",
                "v3": "(/ (* (* (min (min M:i (* sp_1_2:i sp_1_3:i)) sp_1_3:i) (min (+ (// (- (min M:i (* sp_1_2:i sp_1_3:i)) 1) sp_1_3:i) 1) sp_1_2:i)) (* (min K:i sp_2_1:i) (min (+ (// (- K:i 1) sp_2_1:i) 1) sp_2_0:i))) sp_3_0:i)",
                "v4": "(/ (* (* (min (min N:i (* sp_0_2:i sp_0_3:i)) sp_0_3:i) (min (+ (// (- (min N:i (* sp_0_2:i sp_0_3:i)) 1) sp_0_3:i) 1) sp_0_2:i)) (* (min K:i sp_2_1:i) (min (+ (// (- K:i 1) sp_2_1:i) 1) sp_2_0:i))) sp_4_0:i)",
                "blockIdx.x": "(* v0:i v1:i)",
                "vthread": "(* sp_0_0:i sp_1_0:i)",
                "threadIdx.x": "(* sp_1_1:i sp_0_1:i)",
                "T_matmul_NT/i.outer.inner.init": "sp_0_2:i",
                "T_matmul_NT/j.outer.inner.init": "sp_1_2:i",
                "T_matmul_NT/i.inner.init": "sp_0_3:i",
                "T_matmul_NT/j.inner.init": "sp_1_3:i",
                "T_matmul_NT/k.outer.outer": "v2:i",
                "placeholder.d.shared/ax0.ax1.fused.outer.outer": "(+ 1 (// (// (+ -1 (* sp_0_3:i (* (+ (// -1 sp_1_1:i) (+ (+ sp_0_1:i 1) (- (* sp_0_1:i (+ (* sp_0_0:i (// blockIdx.x:i v1:i)) (+ sp_0_0:i (// -1 sp_1_0:i)))) (* sp_0_1:i (* sp_0_0:i (// blockIdx.x:i v1:i)))))) (* sp_0_2:i (* sp_2_1:i sp_2_0:i))))) sp_4_0:i) (* sp_0_1:i sp_1_1:i)))",
                "placeholder.d.shared/ax0.ax1.fused.inner": "(min (* sp_0_2:i (* (+ 1 (+ (// -1 sp_1_1:i) (+ (* sp_0_1:i (+ (* sp_0_0:i (// blockIdx.x:i v1:i)) (+ sp_0_0:i (// -1 sp_1_0:i)))) (- sp_0_1:i (* sp_0_1:i (* sp_0_0:i (// blockIdx.x:i v1:i))))))) (* sp_0_3:i (* sp_2_1:i sp_2_0:i)))) sp_4_0:i)",
                "placeholder.shared/ax0.ax1.fused.outer.outer": "(+ (// (// (- (* (* (* sp_1_2:i sp_1_3:i) (* sp_1_1:i sp_1_0:i)) (* sp_2_1:i sp_2_0:i)) 1) sp_3_0:i) (* sp_1_1:i sp_0_1:i)) 1)",
                "placeholder.shared/ax0.ax1.fused.inner": "(min (* (* (* sp_1_2:i sp_1_3:i) (* sp_1_1:i sp_1_0:i)) (* sp_2_1:i sp_2_0:i)) sp_3_0:i)",
                "placeholder.shared/k.outer.inner": "sp_2_0:i",
                "placeholder.shared/i.outer.inner": "sp_0_2:i",
                "placeholder.shared/j.outer.inner": "sp_1_2:i",
                "placeholder.shared/k.inner": "sp_2_1:i",
                "placeholder.shared/i.inner": "sp_0_3:i",
                "placeholder.shared/j.inner": "sp_1_3:i",
                "T_matmul_NT/ax0.inner": "(* sp_0_2:i sp_0_3:i)",
                "T_matmul_NT/ax1.inner": "(* sp_1_2:i sp_1_3:i)",
            },
        ),
    ],
    "Const,Const,Const,DepthwiseConv2d,Broadcast(Add...),Elemwise(clip...)": [
        SketchCodeDesc(
            ("CI", "FU", "SP", "AN", "FSP", "AN", "CA", "CI", "FU", "AN", "PR"),
            {
                "v0": "(/ C:i sp_0_0:i)",
                "v1": "(+ N:i (* N:i (// (- (+ (* PadH:i 2) H:i) KH:i) StrideH:i)))",
                "v2": "(+ v1:i (* v1:i (// (- (+ (* PadW:i 2) W:i) KW:i) StrideW:i)))",
                "blockIdx.x": "(* v2:i v0:i)",
                "DepthwiseConv2d/c": "sp_0_0:i",
                "threadIdx.x": "sp_0_0:i",
                "DepthwiseConv2d/di.dj.fused.outer": "(+ (// (- (* KH:i KW:i) 1) sp_0_0:i) 1)",
            },
        ),
        SketchCodeDesc(
            # fmt: off
            ('CI', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'RE', 'FSP', 'FSP', 'FSP', 'FSP', 'RE', 'CA', 'CHR', 'CA', 'CHR', 'CA', 'CI', 'FU', 'AN', 'FU', 'AN', 'FU', 'AN', 'FU', 'SP', 'AN', 'FFSP', 'AN', 'FU', 'SP', 'AN', 'FFSP', 'AN', 'PR'),
            # fmt: on
            {
                "v0": "(/ N:i (* (* (* sp_0_0:i sp_0_1:i) sp_0_2:i) sp_0_3:i))",
                "v1": "(/ (+ (// (- (+ (* PadH:i 2) H:i) KH:i) StrideH:i) 1) (* (* (* sp_1_0:i sp_1_1:i) sp_1_2:i) sp_1_3:i))",
                "v2": "(/ (+ (// (- (+ (* PadW:i 2) W:i) KW:i) StrideW:i) 1) (* (* (* sp_2_0:i sp_2_1:i) sp_2_2:i) sp_2_3:i))",
                "v3": "(/ C:i (* (* (* sp_3_0:i sp_3_1:i) sp_3_2:i) sp_3_3:i))",
                "v4": "(/ KH:i (* sp_4_0:i sp_4_1:i))",
                "v5": "(/ KW:i (* sp_5_0:i sp_5_1:i))",
                "v6": "(/ (* (* (* (min KH:i sp_4_1:i) (min (+ (// (- KH:i 1) sp_4_1:i) 1) sp_4_0:i)) (* (min KW:i sp_5_1:i) (min (+ (// (- KW:i 1) sp_5_1:i) 1) sp_5_0:i))) (* (min (min C:i (* sp_3_2:i sp_3_3:i)) sp_3_3:i) (min (+ (// (- (min C:i (* sp_3_2:i sp_3_3:i)) 1) sp_3_3:i) 1) sp_3_2:i))) sp_6_0:i)",
                "v7": "(* sp_4_1:i (* sp_5_1:i (* sp_4_0:i sp_5_0:i)))",
                "blockIdx.x": "(* (* (* v0:i v1:i) v2:i) v3:i)",
                "vthread": "(* (* (* sp_0_0:i sp_1_0:i) sp_2_0:i) sp_3_0:i)",
                "threadIdx.x": "(* (* (* sp_3_1:i sp_2_1:i) sp_1_1:i) sp_0_1:i)",
                "DepthwiseConv2d/b.outer.inner.init": "sp_0_2:i",
                "DepthwiseConv2d/i.outer.inner.init": "sp_1_2:i",
                "DepthwiseConv2d/j.outer.inner.init": "sp_2_2:i",
                "DepthwiseConv2d/c.outer.inner.init": "sp_3_2:i",
                "DepthwiseConv2d/b.inner.init": "sp_0_3:i",
                "DepthwiseConv2d/i.inner.init": "sp_1_3:i",
                "DepthwiseConv2d/j.inner.init": "sp_2_3:i",
                "DepthwiseConv2d/c.inner.init": "sp_3_3:i",
                "DepthwiseConv2d/di.outer.outer": "v4:i",
                "DepthwiseConv2d/dj.outer.outer": "v5:i",
                "PaddedInput.shared/ax0.ax1.fused.ax2.fused.ax3.fused.outer.outer": "(+ 1 (// (// (+ -1 (* (+ (// (// (// -1 sp_3_1:i) sp_2_1:i) sp_1_1:i) (+ 1 (* sp_0_1:i (+ (// (/ (// -1 sp_3_0:i) sp_2_0:i) sp_1_0:i) (+ sp_0_0:i 1))))) (* (* (* sp_3_0:i (* sp_3_1:i (* sp_3_2:i sp_3_3:i))) (* (* sp_0_2:i sp_0_3:i) (+ (* sp_1_0:i (* (* sp_1_2:i StrideH:i) (* sp_1_1:i sp_1_3:i))) (- (* sp_4_1:i sp_4_0:i) StrideH:i)))) (+ (* sp_5_1:i sp_5_0:i) (- (* sp_2_0:i (* sp_2_1:i (* sp_2_2:i (* sp_2_3:i StrideW:i)))) StrideW:i))))) sp_7_0:i) (* sp_0_1:i (* sp_2_1:i (* sp_3_1:i sp_1_1:i)))))",
                "PaddedInput.shared/ax0.ax1.fused.ax2.fused.ax3.fused.inner": "(min (* (+ (// (// (/ -1 sp_3_1:i) sp_2_1:i) sp_1_1:i) (+ 1 (* sp_0_1:i (+ sp_0_0:i (+ 1 (// (/ (/ -1 sp_3_0:i) sp_2_0:i) sp_1_0:i)))))) (* (* (* sp_0_2:i sp_0_3:i) (+ (* sp_4_1:i sp_4_0:i) (- (* (* (* sp_1_1:i StrideH:i) sp_1_0:i) (* sp_1_2:i sp_1_3:i)) StrideH:i))) (* (+ (* sp_5_1:i sp_5_0:i) (- (* sp_2_0:i (* sp_2_1:i (* sp_2_2:i (* sp_2_3:i StrideW:i)))) StrideW:i)) (* sp_3_0:i (* sp_3_1:i (* sp_3_2:i sp_3_3:i)))))) sp_7_0:i)",
                "placeholder.shared/ax0.ax1.fused.ax2.fused.ax3.fused.outer.outer": "(+ (// (// (- (* v7:i (* (* sp_3_2:i sp_3_3:i) (* sp_3_1:i sp_3_0:i))) 1) sp_6_0:i) (* (* (* sp_3_1:i sp_2_1:i) sp_1_1:i) sp_0_1:i)) 1)",
                "placeholder.shared/ax0.ax1.fused.ax2.fused.ax3.fused.inner": "(min (* v7:i (* (* sp_3_2:i sp_3_3:i) (* sp_3_1:i sp_3_0:i))) sp_6_0:i)",
                "placeholder.shared/di.outer.inner": "sp_4_0:i",
                "placeholder.shared/dj.outer.inner": "sp_5_0:i",
                "placeholder.shared/b.outer.inner": "sp_0_2:i",
                "placeholder.shared/i.outer.inner": "sp_1_2:i",
                "placeholder.shared/j.outer.inner": "sp_2_2:i",
                "placeholder.shared/c.outer.inner": "sp_3_2:i",
                "placeholder.shared/di.inner": "sp_4_1:i",
                "placeholder.shared/dj.inner": "sp_5_1:i",
                "placeholder.shared/b.inner": "sp_0_3:i",
                "placeholder.shared/i.inner": "sp_1_3:i",
                "placeholder.shared/j.inner": "sp_2_3:i",
                "placeholder.shared/c.inner": "sp_3_3:i",
                "DepthwiseConv2d/ax0.inner": "(* sp_0_2:i sp_0_3:i)",
                "DepthwiseConv2d/ax1.inner": "(* sp_1_2:i sp_1_3:i)",
                "DepthwiseConv2d/ax2.inner": "(* sp_2_2:i sp_2_3:i)",
                "DepthwiseConv2d/ax3.inner": "(* sp_3_2:i sp_3_3:i)",
            }
        ),
    ],
    "Const,AdaptiveAvgPool3D": [
        SketchCodeDesc(
            ("FU", "SP", "AN", "AN", "FU", "SP", "AN", "AN", "PR"),
            {
                "v0": "(/ (* (* (* (* N:i OutD:i) OutH:i) OutW:i) C:i) sp_0_0:i)",
                "v1": "(/ (* (* (* (* N:i OutD:i) OutH:i) OutW:i) C:i) sp_1_0:i)",
                "blockIdx.x": "v0:i",
                "threadIdx.x": "sp_0_0:i",
                "tensor/rv0": "(- (select (== (mod (* (+ (mod (// (// (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) OutH:i) OutD:i) 1) InD:i) OutD:i) 0) (// (* (+ (mod (// (// (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) OutH:i) OutD:i) 1) InD:i) OutD:i) (+ (// (* (+ (mod (// (// (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) OutH:i) OutD:i) 1) InD:i) OutD:i) 1)) (// (* (mod (// (// (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) OutH:i) OutD:i) InD:i) OutD:i))",
                "tensor/rv1": "(- (select (== (mod (* (+ (mod (// (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) OutH:i) 1) InH:i) OutH:i) 0) (// (* (+ (mod (// (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) OutH:i) 1) InH:i) OutH:i) (+ (// (* (+ (mod (// (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) OutH:i) 1) InH:i) OutH:i) 1)) (// (* (mod (// (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) OutH:i) InH:i) OutH:i))",
                "tensor/rv2": "(- (select (== (mod (* (+ (mod (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) 1) InW:i) OutW:i) 0) (// (* (+ (mod (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) 1) InW:i) OutW:i) (+ (// (* (+ (mod (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) 1) InW:i) OutW:i) 1)) (// (* (mod (// (+ threadIdx.x:i (* blockIdx.x:i sp_1_0:i)) C:i) OutW:i) InW:i) OutW:i))",
            },
        )
    ],
    "Const,Const,Const,Const,Conv3d,Broadcast(Mul...),Broadcast(Add...)": [
        SketchCodeDesc(
            ("CI", "FU", "SP", "AN", "FSP", "AN", "CA", "CI", "FU", "AN", "PR"),
            {
                "v0": "(/ Cout:i sp_0_0:i)",
                "v1": "(+ N:i (* N:i (// (- (+ (* PadD:i 2) D:i) KD:i) StrideD:i)))",
                "v2": "(+ v1:i (* v1:i (// (- (+ (* PadH:i 2) H:i) KH:i) StrideH:i)))",
                "v3": "(+ v2:i (* v2:i (// (- (+ (* PadW:i 2) W:i) KW:i) StrideW:i)))",
                "v4": "(// (- (+ (* (* (* KD:i KH:i) KW:i) Cin:i) sp_0_0:i) 1) sp_0_0:i)",
                "blockIdx.x": "(* v3:i v0:i)",
                "Conv3dOutput/cc": "sp_0_0:i",
                "threadIdx.x": "sp_0_0:i",
                "Conv3dOutput/rd.rh.fused.rw.fused.rc.fused.outer": "v4:i",
            },
        ),
        SketchCodeDesc(
            # fmt: off
            ('CI', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'RE', 'FSP', 'FSP', 'FSP', 'FSP', 'FSP', 'RE', 'CA', 'CHR', 'CA', 'CHR', 'CA', 'CI', 'FU', 'AN', 'FU', 'AN', 'FU', 'AN', 'FU', 'SP', 'AN', 'FFSP', 'AN', 'FU', 'SP', 'AN', 'FFSP', 'AN', 'PR'),
            # fmt: on
            {
                "v0": "(/ N:i (* (* (* sp_0_0:i sp_0_1:i) sp_0_2:i) sp_0_3:i))",
                "v1": "(/ (+ (// (- (+ (* PadD:i 2) D:i) KD:i) StrideD:i) 1) (* (* (* sp_1_0:i sp_1_1:i) sp_1_2:i) sp_1_3:i))",
                "v2": "(/ (+ (// (- (+ (* PadH:i 2) H:i) KH:i) StrideH:i) 1) (* (* (* sp_2_0:i sp_2_1:i) sp_2_2:i) sp_2_3:i))",
                "v3": "(/ (+ (// (- (+ (* PadW:i 2) W:i) KW:i) StrideW:i) 1) (* (* (* sp_3_0:i sp_3_1:i) sp_3_2:i) sp_3_3:i))",
                "v4": "(/ Cout:i (* (* (* sp_4_0:i sp_4_1:i) sp_4_2:i) sp_4_3:i))",
                "v5": "(/ KD:i (* sp_5_0:i sp_5_1:i))",
                "v6": "(/ KH:i (* sp_6_0:i sp_6_1:i))",
                "v7": "(/ KW:i (* sp_7_0:i sp_7_1:i))",
                "v8": "(/ Cin:i (* sp_8_0:i sp_8_1:i))",
                "v9": "(/ (* (* (* (* (* (min KD:i sp_5_1:i) (min (+ (// (- KD:i 1) sp_5_1:i) 1) sp_5_0:i)) (* (min KH:i sp_6_1:i) (min (+ (// (- KH:i 1) sp_6_1:i) 1) sp_6_0:i))) (* (min KW:i sp_7_1:i) (min (+ (// (- KW:i 1) sp_7_1:i) 1) sp_7_0:i))) (* (min Cin:i sp_8_1:i) (min (+ (// (- Cin:i 1) sp_8_1:i) 1) sp_8_0:i))) (* (min (min Cout:i (* sp_4_2:i sp_4_3:i)) sp_4_3:i) (min (+ (// (- (min Cout:i (* sp_4_2:i sp_4_3:i)) 1) sp_4_3:i) 1) sp_4_2:i))) sp_9_0:i)",
                "v10": "(* sp_5_1:i (* sp_6_1:i (* sp_5_0:i sp_6_0:i)))",
                "v11": "(* v10:i (* sp_7_0:i (* sp_7_1:i (* sp_8_1:i sp_8_0:i))))",
                "blockIdx.x": "(* (* (* (* v0:i v1:i) v2:i) v3:i) v4:i)",
                "vthread": "(* (* (* (* sp_0_0:i sp_1_0:i) sp_2_0:i) sp_3_0:i) sp_4_0:i)",
                "threadIdx.x": "(* (* (* (* sp_4_1:i sp_3_1:i) sp_2_1:i) sp_1_1:i) sp_0_1:i)",
                "Conv3dOutput/nn.outer.inner.init": "sp_0_2:i",
                "Conv3dOutput/dd.outer.inner.init": "sp_1_2:i",
                "Conv3dOutput/hh.outer.inner.init": "sp_2_2:i",
                "Conv3dOutput/ww.outer.inner.init": "sp_3_2:i",
                "Conv3dOutput/cc.outer.inner.init": "sp_4_2:i",
                "Conv3dOutput/nn.inner.init": "sp_0_3:i",
                "Conv3dOutput/dd.inner.init": "sp_1_3:i",
                "Conv3dOutput/hh.inner.init": "sp_2_3:i",
                "Conv3dOutput/ww.inner.init": "sp_3_3:i",
                "Conv3dOutput/cc.inner.init": "sp_4_3:i",
                "Conv3dOutput/rd.outer.outer": "v5:i",
                "Conv3dOutput/rh.outer.outer": "v6:i",
                "Conv3dOutput/rw.outer.outer": "v7:i",
                "Conv3dOutput/rc.outer.outer": "v8:i",
                "PaddedInput.shared/ax0.ax1.fused.ax2.fused.ax3.fused.ax4.fused.outer.outer": "(+ 1 (// (/ (+ -1 (* (* sp_0_3:i (+ sp_0_2:i (* sp_0_2:i (+ sp_0_1:i (+ (// (// (/ (/ -1 sp_4_1:i) sp_3_1:i) sp_2_1:i) sp_1_1:i) (- (* sp_0_1:i (+ (+ sp_0_0:i (* sp_0_0:i (// (/ (// (// blockIdx.x:i v4:i) v3:i) v2:i) v1:i))) (// (// (// (// -1 sp_4_0:i) sp_3_0:i) sp_2_0:i) sp_1_0:i))) (* sp_0_1:i (* sp_0_0:i (// (/ (// (// blockIdx.x:i v4:i) v3:i) v2:i) v1:i))))))))) (* (* (+ (* sp_5_1:i sp_5_0:i) (- (* sp_1_0:i (* (* sp_1_2:i StrideD:i) (* sp_1_1:i sp_1_3:i))) StrideD:i)) (+ (* sp_6_1:i sp_6_0:i) (- (* sp_2_0:i (* sp_2_1:i (* sp_2_2:i (* sp_2_3:i StrideH:i)))) StrideH:i))) (* (+ (* sp_7_1:i sp_7_0:i) (- (* sp_3_0:i (* sp_3_1:i (* sp_3_3:i (* sp_3_2:i StrideW:i)))) StrideW:i)) (* sp_8_1:i sp_8_0:i))))) sp_10_0:i) (* sp_0_1:i (* sp_4_1:i (* sp_3_1:i (* sp_2_1:i sp_1_1:i))))))",
                "PaddedInput.shared/ax0.ax1.fused.ax2.fused.ax3.fused.ax4.fused.inner": "(min (* (* sp_0_3:i (* sp_0_2:i (+ (* sp_0_1:i (+ (* sp_0_0:i (// (/ (// (// blockIdx.x:i v4:i) v3:i) v2:i) v1:i)) (+ (// (/ (/ (/ -1 sp_4_0:i) sp_3_0:i) sp_2_0:i) sp_1_0:i) (+ sp_0_0:i 1)))) (- (+ 1 (// (/ (/ (// -1 sp_4_1:i) sp_3_1:i) sp_2_1:i) sp_1_1:i)) (* sp_0_1:i (* sp_0_0:i (// (/ (// (// blockIdx.x:i v4:i) v3:i) v2:i) v1:i))))))) (* (* (+ (* sp_7_1:i sp_7_0:i) (- (* sp_3_0:i (* (* sp_3_1:i (* sp_3_2:i StrideW:i)) sp_3_3:i)) StrideW:i)) (* (+ (* sp_5_1:i sp_5_0:i) (- (* sp_1_0:i (* sp_1_1:i (* sp_1_2:i (* sp_1_3:i StrideD:i)))) StrideD:i)) (+ (* sp_6_1:i sp_6_0:i) (- (* sp_2_0:i (* sp_2_1:i (* sp_2_2:i (* sp_2_3:i StrideH:i)))) StrideH:i)))) (* sp_8_1:i sp_8_0:i))) sp_10_0:i)",
                "placeholder.shared/ax0.ax1.fused.ax2.fused.ax3.fused.ax4.fused.outer.outer": "(+ (// (// (- (* v11:i (* (* sp_4_2:i sp_4_3:i) (* sp_4_1:i sp_4_0:i))) 1) sp_9_0:i) (* (* (* (* sp_4_1:i sp_3_1:i) sp_2_1:i) sp_1_1:i) sp_0_1:i)) 1)",
                "placeholder.shared/ax0.ax1.fused.ax2.fused.ax3.fused.ax4.fused.inner": "(min (* v11:i (* (* sp_4_2:i sp_4_3:i) (* sp_4_1:i sp_4_0:i))) sp_9_0:i)",
                "placeholder.shared/rd.outer.inner": "sp_5_0:i",
                "placeholder.shared/rh.outer.inner": "sp_6_0:i",
                "placeholder.shared/rw.outer.inner": "sp_7_0:i",
                "placeholder.shared/rc.outer.inner": "sp_8_0:i",
                "placeholder.shared/nn.outer.inner": "sp_0_2:i",
                "placeholder.shared/dd.outer.inner": "sp_1_2:i",
                "placeholder.shared/hh.outer.inner": "sp_2_2:i",
                "placeholder.shared/ww.outer.inner": "sp_3_2:i",
                "placeholder.shared/cc.outer.inner": "sp_4_2:i",
                "placeholder.shared/rd.inner": "sp_5_1:i",
                "placeholder.shared/rh.inner": "sp_6_1:i",
                "placeholder.shared/rw.inner": "sp_7_1:i",
                "placeholder.shared/rc.inner": "sp_8_1:i",
                "placeholder.shared/nn.inner": "sp_0_3:i",
                "placeholder.shared/dd.inner": "sp_1_3:i",
                "placeholder.shared/hh.inner": "sp_2_3:i",
                "placeholder.shared/ww.inner": "sp_3_3:i",
                "placeholder.shared/cc.inner": "sp_4_3:i",
                "Conv3dOutput/ax0.inner": "(* sp_0_2:i sp_0_3:i)",
                "Conv3dOutput/ax1.inner": "(* sp_1_2:i sp_1_3:i)",
                "Conv3dOutput/ax2.inner": "(* sp_2_2:i sp_2_3:i)",
                "Conv3dOutput/ax3.inner": "(* sp_3_2:i sp_3_3:i)",
                "Conv3dOutput/ax4.inner": "(* sp_4_2:i sp_4_3:i)",
            }
        ),
    ],
    "Const,Const,TransposeConv2d,Elemwise(tanh...)": [
        SketchCodeDesc(
            ("FU", "SP", "AN", "FSP", "AN", "CA", "CI", "FU", "AN", "PR"),
            {
                "v0": "(/ (- (+ (* (- W:i 1) StrideW:i) KW:i) (* PadW:i 2)) sp_0_0:i)",
                "v1": "(* (* N:i Cout:i) (- (+ (* (- H:i 1) StrideH:i) KH:i) (* PadH:i 2)))",
                "v2": "(// (- (+ (* (* Cin:i KH:i) KW:i) sp_0_0:i) 1) sp_0_0:i)",
                "blockIdx.x": "(* v1:i v0:i)",
                "compute/w": "sp_0_0:i",
                "threadIdx.x": "sp_0_0:i",
                "compute/dc.dh.fused.dw.fused.outer": "v2:i",
            },
        ),
        SketchCodeDesc(
            # fmt: off
            ('SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'RE', 'FSP', 'FSP', 'FSP', 'FSP', 'RE', 'CA', 'CHR', 'CA', 'CHR', 'CA', 'CI', 'FU', 'AN', 'FU', 'AN', 'FU', 'AN', 'FU', 'SP', 'AN', 'FFSP', 'AN', 'FU', 'SP', 'AN', 'FFSP', 'AN', 'PR'),
            # fmt: on
            {
                "v0": "(/ N:i (* (* (* sp_0_0:i sp_0_1:i) sp_0_2:i) sp_0_3:i))",
                "v1": "(/ Cout:i (* (* (* sp_1_0:i sp_1_1:i) sp_1_2:i) sp_1_3:i))",
                "v2": "(/ (- (+ (* (- H:i 1) StrideH:i) KH:i) (* PadH:i 2)) (* (* (* sp_2_0:i sp_2_1:i) sp_2_2:i) sp_2_3:i))",
                "v3": "(/ (- (+ (* (- W:i 1) StrideW:i) KW:i) (* PadW:i 2)) (* (* (* sp_3_0:i sp_3_1:i) sp_3_2:i) sp_3_3:i))",
                "v4": "(/ Cin:i (* sp_4_0:i sp_4_1:i))",
                "v5": "(/ KH:i (* sp_5_0:i sp_5_1:i))",
                "v6": "(/ KW:i (* sp_6_0:i sp_6_1:i))",
                "v7": "(/ (* (* (* (* (min Cin:i sp_4_1:i) (min (+ (// (- Cin:i 1) sp_4_1:i) 1) sp_4_0:i)) (* (min (min Cout:i (* sp_1_2:i sp_1_3:i)) sp_1_3:i) (min (+ (// (- (min Cout:i (* sp_1_2:i sp_1_3:i)) 1) sp_1_3:i) 1) sp_1_2:i))) (* (min KH:i sp_5_1:i) (min (+ (// (- KH:i 1) sp_5_1:i) 1) sp_5_0:i))) (* (min KW:i sp_6_1:i) (min (+ (// (- KW:i 1) sp_6_1:i) 1) sp_6_0:i))) sp_7_0:i)",
                "blockIdx.x": "(* (* (* v0:i v1:i) v2:i) v3:i)",
                "vthread": "(* (* (* sp_0_0:i sp_1_0:i) sp_2_0:i) sp_3_0:i)",
                "threadIdx.x": "(* (* (* sp_3_1:i sp_2_1:i) sp_1_1:i) sp_0_1:i)",
                "compute/b.outer.inner.init": "sp_0_2:i",
                "compute/c.outer.inner.init": "sp_1_2:i",
                "compute/h.outer.inner.init": "sp_2_2:i",
                "compute/w.outer.inner.init": "sp_3_2:i",
                "compute/b.inner.init": "sp_0_3:i",
                "compute/c.inner.init": "sp_1_3:i",
                "compute/h.inner.init": "sp_2_3:i",
                "compute/w.inner.init": "sp_3_3:i",
                "compute/dc.outer.outer": "v4:i",
                "compute/dh.outer.outer": "v5:i",
                "compute/dw.outer.outer": "v6:i",
                "data_pad.shared/ax0.ax1.fused.ax2.fused.ax3.fused.outer.outer": "(+ 1 (// (// (+ -1 (* (* sp_0_2:i sp_0_3:i) (* (* (* (* sp_4_0:i (+ -1 (+ (* sp_6_1:i sp_6_0:i) (* sp_3_1:i (* (* sp_3_0:i sp_3_2:i) sp_3_3:i))))) (+ (* sp_5_1:i sp_5_0:i) (+ -1 (* sp_2_0:i (* sp_2_1:i (* sp_2_2:i sp_2_3:i)))))) sp_4_1:i) (+ 1 (+ (// (// (// -1 sp_3_1:i) sp_2_1:i) sp_1_1:i) (* sp_0_1:i (+ sp_0_0:i (+ 1 (// (// (// -1 sp_3_0:i) sp_2_0:i) sp_1_0:i))))))))) sp_8_0:i) (* sp_0_1:i (* sp_2_1:i (* sp_3_1:i sp_1_1:i)))))",
                "data_pad.shared/ax0.ax1.fused.ax2.fused.ax3.fused.inner": "(min (* (* sp_0_2:i sp_0_3:i) (* (* (+ 1 (+ (// (// (// -1 sp_3_1:i) sp_2_1:i) sp_1_1:i) (* sp_0_1:i (+ (// (// (// -1 sp_3_0:i) sp_2_0:i) sp_1_0:i) (+ sp_0_0:i 1))))) (* sp_4_1:i sp_4_0:i)) (* (+ -1 (+ (* sp_5_1:i sp_5_0:i) (* sp_2_0:i (* sp_2_1:i (* sp_2_2:i sp_2_3:i))))) (+ -1 (+ (* sp_6_1:i sp_6_0:i) (* sp_3_1:i (* sp_3_0:i (* sp_3_2:i sp_3_3:i)))))))) sp_8_0:i)",
                "placeholder.shared/ax0.ax1.fused.ax2.fused.ax3.fused.outer.outer": "(+ (// (// (- (* (* (* (* sp_4_1:i sp_4_0:i) (* (* sp_1_2:i sp_1_3:i) (* sp_1_1:i sp_1_0:i))) (* sp_5_1:i sp_5_0:i)) (* sp_6_1:i sp_6_0:i)) 1) sp_7_0:i) (* (* (* sp_3_1:i sp_2_1:i) sp_1_1:i) sp_0_1:i)) 1)",
                "placeholder.shared/ax0.ax1.fused.ax2.fused.ax3.fused.inner": "(min (* (* (* (* sp_4_1:i sp_4_0:i) (* (* sp_1_2:i sp_1_3:i) (* sp_1_1:i sp_1_0:i))) (* sp_5_1:i sp_5_0:i)) (* sp_6_1:i sp_6_0:i)) sp_7_0:i)",
                "placeholder.shared/dc.outer.inner": "sp_4_0:i",
                "placeholder.shared/dh.outer.inner": "sp_5_0:i",
                "placeholder.shared/dw.outer.inner": "sp_6_0:i",
                "placeholder.shared/b.outer.inner": "sp_0_2:i",
                "placeholder.shared/c.outer.inner": "sp_1_2:i",
                "placeholder.shared/h.outer.inner": "sp_2_2:i",
                "placeholder.shared/w.outer.inner": "sp_3_2:i",
                "placeholder.shared/dc.inner": "sp_4_1:i",
                "placeholder.shared/dh.inner": "sp_5_1:i",
                "placeholder.shared/dw.inner": "sp_6_1:i",
                "placeholder.shared/b.inner": "sp_0_3:i",
                "placeholder.shared/c.inner": "sp_1_3:i",
                "placeholder.shared/h.inner": "sp_2_3:i",
                "placeholder.shared/w.inner": "sp_3_3:i",
                "compute/ax0.inner": "(* sp_0_2:i sp_0_3:i)",
                "compute/ax1.inner": "(* sp_1_2:i sp_1_3:i)",
                "compute/ax2.inner": "(* sp_2_2:i sp_2_3:i)",
                "compute/ax3.inner": "(* sp_3_2:i sp_3_3:i)",
            }
        ),
    ],
    "Const,Const,Conv2d,Elemwise(sigmoid...)": [
        SketchCodeRef("Const,Const,Const,Conv2d,Broadcast(Add...)", 0, [], {}),
        SketchCodeRef("Const,Const,Const,Conv2d,Broadcast(Add...)", 1, [], {}),
    ],
    "Const,Const,Conv2d,Elemwise(leaky_relu...)": [
        SketchCodeRef("Const,Const,Const,Conv2d,Broadcast(Add...)", 0, [], {}),
        SketchCodeRef("Const,Const,Const,Conv2d,Broadcast(Add...)", 1, [], {}),
    ],
    "Const,Const,Const,Conv2d,Broadcast(Add...),Elemwise(leaky_relu...)": [
        SketchCodeRef("Const,Const,Const,Conv2d,Broadcast(Add...)", 0, ["CI"], {}),
        SketchCodeRef("Const,Const,Const,Conv2d,Broadcast(Add...)", 1, ["CI"], {}),
    ],
    "Const,Const,Const,Conv2d,Broadcast(Add...),Elemwise(relu...)": [
        SketchCodeRef("Const,Const,Const,Conv2d,Broadcast(Add...)", 0, ["CI"], {}),
        SketchCodeRef("Const,Const,Const,Conv2d,Broadcast(Add...)", 1, ["CI"], {}),
    ],
    "Const,Const,Const,Conv2d,Broadcast(Add...),Elemwise(clip...)": [
        SketchCodeRef("Const,Const,Const,Conv2d,Broadcast(Add...)", 0, ["CI"], {}),
        SketchCodeRef("Const,Const,Const,Conv2d,Broadcast(Add...)", 1, ["CI"], {}),
    ],
    "Const,Const,Const,Const,Conv2d,Broadcast(Add...),Broadcast(Add...)": [
        SketchCodeRef("Const,Const,Const,Conv2d,Broadcast(Add...)", 0, ["CI"], {}),
        SketchCodeRef("Const,Const,Const,Conv2d,Broadcast(Add...)", 1, ["CI"], {}),
    ],
    "Const,Const,Const,Const,Conv2d,Broadcast(Add...),Broadcast(Add...),Elemwise(relu...)": [
        SketchCodeRef("Const,Const,Const,Conv2d,Broadcast(Add...)", 0, ["CI", "CI"], {}),
        SketchCodeRef("Const,Const,Const,Conv2d,Broadcast(Add...)", 1, ["CI", "CI"], {}),
    ],
    "Const,Const,Const,Const,Conv3d,Broadcast(Mul...),Broadcast(Add...),Elemwise(relu...)": [
        SketchCodeRef(
            "Const,Const,Const,Const,Conv3d,Broadcast(Mul...),Broadcast(Add...)", 0, ["CI"], {}
        ),
        SketchCodeRef(
            "Const,Const,Const,Const,Conv3d,Broadcast(Mul...),Broadcast(Add...)", 1, ["CI"], {}
        ),
    ],
    "Const,Const,Const,Const,Const,Conv3d,Broadcast(Mul...),Broadcast(Add...),Broadcast(Add...),Elemwise(relu...)": [
        # fmt: off
        SketchCodeRef(
            "Const,Const,Const,Const,Conv3d,Broadcast(Mul...),Broadcast(Add...)", 0, ["CI", "CI"], {}
        ),
        SketchCodeRef(
            "Const,Const,Const,Const,Conv3d,Broadcast(Mul...),Broadcast(Add...)", 1, ["CI", "CI"], {}
        ),
        # fmt: on
    ],
    "Const,Const,Const,Const,TransposeConv2d,Broadcast(Mul...),Broadcast(Add...),Elemwise(relu...)": [
        SketchCodeRef("Const,Const,TransposeConv2d,Elemwise(tanh...)", 0, ["CI", "CI"], {}),
        SketchCodeRef("Const,Const,TransposeConv2d,Elemwise(tanh...)", 1, ["CI", "CI"], {}),
    ],
}


def check_tasks(tasks: List[felix.SymTaskAndInstances], expected: Dict[str, List[int]]):
    assert len(tasks) == len(expected)
    for task, instances in tasks:
        desc = str(task)
        assert desc in expected
        expected_indices = expected[desc]
        actual_indices = [inst.idx for inst in instances]
        assert expected_indices == actual_indices
        assert desc in TASK_SKETCHES
        sketches = TASK_SKETCHES[desc]
        assert len(task.sketches) == len(sketches)
        print(desc)
        for in_sk, ref_sk in zip(task.sketches, sketches):
            ref_sk_ = ref_sk.make_desc()
            assert in_sk.backbone == ref_sk_.backbone
            vardefs = {repr(k): v for k, v in in_sk.context.to_varmap().items()}
            vardefs.update(ffi.get_loop_bounds(in_sk.code))
            assert len(ref_sk_.loop_bounds) == len(vardefs)
            for k, expr in vardefs.items():
                print(f"{k}: {expr} ; {ref_sk_.loop_bounds[k]}")
                assert ffi.is_expr_equivalent(expr, ref_sk_.loop_bounds[k])
            # vardefs = {
            #     repr(k): ffi.print_expr_preorder(v) for k, v in in_sk.context.to_varmap().items()
            # }
            # bounds = {k: ffi.print_expr_preorder(v) for k, v in ffi.get_loop_bounds(in_sk.code)}


@tvm.testing.requires_gpu
@pytest.mark.parametrize("network_name", ["resnet50", "mobilenet_v2", "r3d_18", "dcgan"])
def test_extract_task(network_name: str):
    network, shape = NETWORKS[network_name]()
    inputs = torch.randn(shape)
    tasks = felix.extract_tasks(network, inputs, save_to=f"/tmp/{network_name}.pkl")
    check_tasks(tasks, NETWORK_TASKS[network_name])
    tasks_ = felix.load_and_register_tasks(f"/tmp/{network_name}.pkl")
    check_tasks(tasks_, NETWORK_TASKS[network_name])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
