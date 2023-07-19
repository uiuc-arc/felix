import sys
from typing import Dict, List, Union

import pytest
import torch
import tvm
import tvm.testing
from torchvision.models import mobilenet_v2, resnet50
from torchvision.models.video import r3d_18
from tvm.felix import ffi, nn, sym_task


def dcgan():
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
        self.vardefs = {k: ffi.parse_expr_preorder(v) for k, v in loop_bounds.items()}

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
        ret.vardefs = {**sketch.vardefs, **self.overwrite_loop_bounds}
        return ret


TASK_SKETCHES: Dict[str, List[Union[SketchCodeDesc, SketchCodeRef]]] = {
    "Const,Max2DPool[Padded]": [
        SketchCodeDesc(
            ("CI", "FU", "SP", "AN", "AN", "PR"),
            {
                "v0": "(+ N:s (* N:s (// (+ In0:s (- (* PadH:s 2) KSize0:s)) StrideH:s)))",
                "v1": "(+ v0:v (* v0:v (// (+ In1:s (- (* PadW:s 2) KSize1:s)) StrideW:s)))",
                "blockIdx.x": "(+ (// (- (* v1:v C:s) 1) sp_0_0:k) 1)",
                "threadIdx.x": "sp_0_0:k",
                "tensor/rv0": "KSize0:s",
                "tensor/rv1": "KSize1:s",
            },
        )
    ],
    # AdaptiveAvgPool2D
    "Const,AdaptiveAvgPool2D": [
        SketchCodeDesc(
            ("FU", "SP", "AN", "AN", "FU", "SP", "AN", "AN", "PR"),
            {
                "blockIdx.x": "(+ (// (- (* (* (* N:s OutH:s) OutW:s) C:s) 1) sp_0_0:k) 1)",
                "threadIdx.x": "sp_0_0:k",
                "tensor/rv0": "(- (select (== (mod (* (+ (mod (// (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) OutH:s) 1) InH:s) OutH:s) 0) (// (* (+ (mod (// (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) OutH:s) 1) InH:s) OutH:s) (+ (// (* (+ (mod (// (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) OutH:s) 1) InH:s) OutH:s) 1)) (// (* (mod (// (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) OutH:s) InH:s) OutH:s))",
                "tensor/rv1": "(- (select (== (mod (* (+ (mod (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) 1) InW:s) OutW:s) 0) (// (* (+ (mod (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) 1) InW:s) OutW:s) (+ (// (* (+ (mod (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) 1) InW:s) OutW:s) 1)) (// (* (mod (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) InW:s) OutW:s))",
            },
        ),
    ],
    # Conv2d/+
    "Const,Const,Const,Conv2d,Broadcast(Add...)": [
        SketchCodeDesc(
            ("FU", "SP", "AN", "FSP", "AN", "CA", "CI", "FU", "AN", "PR"),
            {
                "E0": "Cout:s",
                "v1": "(+ N:s (* N:s (// (+ H:s (- (* PadH:s 2) KH:s)) StrideH:s)))",
                "v2": "(+ v1:v (* v1:v (// (+ W:s (- (* PadW:s 2) KW:s)) StrideW:s)))",
                "blockIdx.x": "(* v2:v q0:v)",
                "Conv2dOutput/ff": "sp_0_0:k",
                "threadIdx.x": "sp_0_0:k",
                "Conv2dOutput/ry.rx.fused.rc.fused.outer": "(+ (// (- (* (* KH:s KW:s) Cin:s) 1) sp_0_0:k) 1)",
            },
        ),
        SketchCodeDesc(
            # fmt: off
            ("SP", "SP", "SP", "SP", "SP", "SP", "SP", "RE", "FSP", "FSP", "FSP", "FSP", "RE", "CA", "CHR", "CA", "CHR", "CA", "CI", "FU", "AN", "FU", "AN", "FU", "AN", "FU", "SP", "AN", "FFSP", "AN", "FU", "SP", "AN", "FFSP", "AN", "PR"),
            # fmt: on
            {
                "E0": "N:s",
                "E1": "(+ (// (- (+ (* PadH:s 2) H:s) KH:s) StrideH:s) 1)",
                "E2": "(+ (// (- (+ (* PadW:s 2) W:s) KW:s) StrideW:s) 1)",
                "E3": "Cout:s",
                "E4": "KH:s",
                "E5": "KW:s",
                "E6": "Cin:s",
                "v7": "(* sp_4_1:k (* sp_5_1:k (* sp_4_0:k sp_5_0:k)))",
                "blockIdx.x": "(* (* (* q0:v q1:v) q2:v) q3:v)",
                "vthread": "(* (* (* sp_0_0:k sp_1_0:k) sp_2_0:k) sp_3_0:k)",
                "threadIdx.x": "(* (* (* sp_3_1:k sp_2_1:k) sp_1_1:k) sp_0_1:k)",
                "Conv2dOutput/nn.outer.inner.init": "sp_0_2:k",
                "Conv2dOutput/yy.outer.inner.init": "sp_1_2:k",
                "Conv2dOutput/xx.outer.inner.init": "sp_2_2:k",
                "Conv2dOutput/ff.outer.inner.init": "sp_3_2:k",
                "Conv2dOutput/nn.inner.init": "sp_0_3:k",
                "Conv2dOutput/yy.inner.init": "sp_1_3:k",
                "Conv2dOutput/xx.inner.init": "sp_2_3:k",
                "Conv2dOutput/ff.inner.init": "sp_3_3:k",
                "Conv2dOutput/ry.outer.outer": "q4:v",
                "Conv2dOutput/rx.outer.outer": "q5:v",
                "Conv2dOutput/rc.outer.outer": "q6:v",
                "PaddedInput.shared/ax0.ax1.fused.ax2.fused.ax3.fused.outer.outer": "(+ 1 (// (// (+ -1 (* (* sp_0_2:k sp_0_3:k) (* (* (+ sp_0_1:k (+ 1 (+ (// (// (// -1 sp_3_1:k) sp_2_1:k) sp_1_1:k) (* sp_0_1:k (+ sp_0_0:k (// (// (// -1 sp_3_0:k) sp_2_0:k) sp_1_0:k)))))) (+ (* (* sp_1_2:k sp_1_3:k) (* StrideH:s (* sp_1_0:k sp_1_1:k))) (- (* sp_4_1:k sp_4_0:k) StrideH:s))) (* (+ (* sp_5_1:k sp_5_0:k) (- (* sp_2_0:k (* StrideW:s (* sp_2_3:k (* sp_2_1:k sp_2_2:k)))) StrideW:s)) (* sp_6_1:k sp_6_0:k))))) sp_8_0:k) (* sp_0_1:k (* sp_3_1:k (* sp_2_1:k sp_1_1:k)))))",
                "PaddedInput.shared/ax0.ax1.fused.ax2.fused.ax3.fused.inner": "(min (* (* sp_0_2:k sp_0_3:k) (* (* (+ 1 (+ sp_0_1:k (+ (// (// (// -1 sp_3_1:k) sp_2_1:k) sp_1_1:k) (* sp_0_1:k (+ sp_0_0:k (// (// (// -1 sp_3_0:k) sp_2_0:k) sp_1_0:k)))))) (+ (* sp_1_0:k (* StrideH:s (* sp_1_1:k (* sp_1_2:k sp_1_3:k)))) (- (* sp_4_1:k sp_4_0:k) StrideH:s))) (* (+ (* sp_2_0:k (* StrideW:s (* sp_2_2:k (* sp_2_1:k sp_2_3:k)))) (- (* sp_5_1:k sp_5_0:k) StrideW:s)) (* sp_6_1:k sp_6_0:k)))) sp_8_0:k)",
                "placeholder.shared/ax0.ax1.fused.ax2.fused.ax3.fused.outer.outer": "(+ (// (// (- (* (* v7:v (* sp_6_1:k sp_6_0:k)) (* (* sp_3_2:k sp_3_3:k) (* sp_3_1:k sp_3_0:k))) 1) sp_7_0:k) (* (* (* sp_3_1:k sp_2_1:k) sp_1_1:k) sp_0_1:k)) 1)",
                "placeholder.shared/ax0.ax1.fused.ax2.fused.ax3.fused.inner": "(min (* (* v7:v (* sp_6_1:k sp_6_0:k)) (* (* sp_3_2:k sp_3_3:k) (* sp_3_1:k sp_3_0:k))) sp_7_0:k)",
                "placeholder.shared/ry.outer.inner": "sp_4_0:k",
                "placeholder.shared/rx.outer.inner": "sp_5_0:k",
                "placeholder.shared/rc.outer.inner": "sp_6_0:k",
                "placeholder.shared/nn.outer.inner": "sp_0_2:k",
                "placeholder.shared/yy.outer.inner": "sp_1_2:k",
                "placeholder.shared/xx.outer.inner": "sp_2_2:k",
                "placeholder.shared/ff.outer.inner": "sp_3_2:k",
                "placeholder.shared/ry.inner": "sp_4_1:k",
                "placeholder.shared/rx.inner": "sp_5_1:k",
                "placeholder.shared/rc.inner": "sp_6_1:k",
                "placeholder.shared/nn.inner": "sp_0_3:k",
                "placeholder.shared/yy.inner": "sp_1_3:k",
                "placeholder.shared/xx.inner": "sp_2_3:k",
                "placeholder.shared/ff.inner": "sp_3_3:k",
                "Conv2dOutput/ax0.inner": "(* sp_0_2:k sp_0_3:k)",
                "Conv2dOutput/ax1.inner": "(* sp_1_2:k sp_1_3:k)",
                "Conv2dOutput/ax2.inner": "(* sp_2_2:k sp_2_3:k)",
                "Conv2dOutput/ax3.inner": "(* sp_3_2:k sp_3_3:k)",
            }
        ),
    ],
    "Const,Const,Const,Dense,Broadcast(Add...)": [
        SketchCodeDesc(
            ("SP", "AN", "FSP", "AN", "CA", "FU", "AN", "PR"),
            {
                "E0": "M:s",
                "blockIdx.x": "(* N:s q0:v)",
                "T_matmul_NT/j": "sp_0_0:k",
                "threadIdx.x": "sp_0_0:k",
                "T_matmul_NT/k.outer": "(+ (// (- K:s 1) sp_0_0:k) 1)",
            },
        ),
        SketchCodeDesc(
            # fmt: off
            ("SP", "SP", "SP", "RE", "FSP", "FSP", "RE", "CA", "CHR", "CA", "CHR", "CA", "FU", "AN", "FU", "AN", "FU", "AN", "FU", "SP", "AN", "FFSP", "AN", "FU", "SP", "AN", "FFSP", "AN", "PR"),
            # fmt: on
            {
                "E0": "N:s",
                "E1": "M:s",
                "E2": "K:s",
                "blockIdx.x": "(* q0:v q1:v)",
                "vthread": "(* sp_0_0:k sp_1_0:k)",
                "threadIdx.x": "(* sp_1_1:k sp_0_1:k)",
                "T_matmul_NT/i.outer.inner.init": "sp_0_2:k",
                "T_matmul_NT/j.outer.inner.init": "sp_1_2:k",
                "T_matmul_NT/i.inner.init": "sp_0_3:k",
                "T_matmul_NT/j.inner.init": "sp_1_3:k",
                "T_matmul_NT/k.outer.outer": "q2:v",
                "placeholder.d.shared/ax0.ax1.fused.outer.outer": "(+ 1 (// (// (+ -1 (* sp_0_3:k (* (+ sp_0_1:k (+ (+ 1 (// -1 sp_1_1:k)) (* sp_0_1:k (+ sp_0_0:k (// -1 sp_1_0:k))))) (* sp_0_2:k (* sp_2_1:k sp_2_0:k))))) sp_4_0:k) (* sp_0_1:k sp_1_1:k)))",
                "placeholder.d.shared/ax0.ax1.fused.inner": "(min (* sp_0_2:k (* (+ 1 (+ (// -1 sp_1_1:k) (* sp_0_1:k (+ (+ 1 (// -1 sp_1_0:k)) (* sp_0_0:k (- (+ (// blockIdx.x q1:v) 1) (// blockIdx.x q1:v))))))) (* sp_0_3:k (* sp_2_1:k sp_2_0:k)))) sp_4_0:k)",
                "placeholder.shared/ax0.ax1.fused.outer.outer": "(+ (// (// (- (* (* (* sp_1_2:k sp_1_3:k) (* sp_1_1:k sp_1_0:k)) (* sp_2_1:k sp_2_0:k)) 1) sp_3_0:k) (* sp_1_1:k sp_0_1:k)) 1)",
                "placeholder.shared/ax0.ax1.fused.inner": "(min (* (* (* sp_1_2:k sp_1_3:k) (* sp_1_1:k sp_1_0:k)) (* sp_2_1:k sp_2_0:k)) sp_3_0:k)",
                "placeholder.shared/k.outer.inner": "sp_2_0:k",
                "placeholder.shared/i.outer.inner": "sp_0_2:k",
                "placeholder.shared/j.outer.inner": "sp_1_2:k",
                "placeholder.shared/k.inner": "sp_2_1:k",
                "placeholder.shared/i.inner": "sp_0_3:k",
                "placeholder.shared/j.inner": "sp_1_3:k",
                "T_matmul_NT/ax0.inner": "(* sp_0_2:k sp_0_3:k)",
                "T_matmul_NT/ax1.inner": "(* sp_1_2:k sp_1_3:k)",
            },
        ),
    ],
    "Const,Const,Const,DepthwiseConv2d,Broadcast(Add...),Elemwise(clip...)": [
        SketchCodeDesc(
            ("CI", "FU", "SP", "AN", "FSP", "AN", "CA", "CI", "FU", "AN", "PR"),
            {
                "E0": "C:s",
                "v1": "(+ N:s (* N:s (// (+ H:s (- (* PadH:s 2) KH:s)) StrideH:s)))",
                "v2": "(+ v1:v (* v1:v (// (+ W:s (- (* PadW:s 2) KW:s)) StrideW:s)))",
                "blockIdx.x": "(* v2:v q0:v)",
                "DepthwiseConv2d/c": "sp_0_0:k",
                "threadIdx.x": "sp_0_0:k",
                "DepthwiseConv2d/di.dj.fused.outer": "(+ (// (- (* KH:s KW:s) 1) sp_0_0:k) 1)",
            },
        ),
        SketchCodeDesc(
            # fmt: off
            ('CI', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'RE', 'FSP', 'FSP', 'FSP', 'FSP', 'RE', 'CA', 'CHR', 'CA', 'CHR', 'CA', 'CI', 'FU', 'AN', 'FU', 'AN', 'FU', 'AN', 'FU', 'SP', 'AN', 'FFSP', 'AN', 'FU', 'SP', 'AN', 'FFSP', 'AN', 'PR'),
            # fmt: on
            {
                "E0": "N:s",
                "E1": "(+ (// (- (+ (* PadH:s 2) H:s) KH:s) StrideH:s) 1)",
                "E2": "(+ (// (- (+ (* PadW:s 2) W:s) KW:s) StrideW:s) 1)",
                "E3": "C:s",
                "E4": "KH:s",
                "E5": "KW:s",
                "v6": "(* sp_4_1:k (* sp_5_1:k (* sp_4_0:k sp_5_0:k)))",
                "blockIdx.x": "(* (* (* q0:v q1:v) q2:v) q3:v)",
                "vthread": "(* (* (* sp_0_0:k sp_1_0:k) sp_2_0:k) sp_3_0:k)",
                "threadIdx.x": "(* (* (* sp_3_1:k sp_2_1:k) sp_1_1:k) sp_0_1:k)",
                "DepthwiseConv2d/b.outer.inner.init": "sp_0_2:k",
                "DepthwiseConv2d/i.outer.inner.init": "sp_1_2:k",
                "DepthwiseConv2d/j.outer.inner.init": "sp_2_2:k",
                "DepthwiseConv2d/c.outer.inner.init": "sp_3_2:k",
                "DepthwiseConv2d/b.inner.init": "sp_0_3:k",
                "DepthwiseConv2d/i.inner.init": "sp_1_3:k",
                "DepthwiseConv2d/j.inner.init": "sp_2_3:k",
                "DepthwiseConv2d/c.inner.init": "sp_3_3:k",
                "DepthwiseConv2d/di.outer.outer": "q4:v",
                "DepthwiseConv2d/dj.outer.outer": "q5:v",
                "PaddedInput.shared/ax0.ax1.fused.ax2.fused.ax3.fused.outer.outer": "(+ 1 (// (// (+ (* sp_0_1:k (* (* sp_0_3:k sp_0_0:k) (* (+ (* sp_4_1:k sp_4_0:k) (- (* StrideH:s (* sp_1_1:k (* sp_1_2:k (* sp_1_0:k sp_1_3:k)))) StrideH:s)) (* (+ (* sp_5_1:k sp_5_0:k) (* StrideW:s (+ (* sp_2_0:k (* sp_2_1:k (* sp_2_2:k sp_2_3:k))) -1))) (* sp_0_2:k (* sp_3_2:k (* sp_3_1:k (* sp_3_3:k sp_3_0:k)))))))) -1) sp_7_0:k) (* sp_2_1:k (* sp_0_1:k (* sp_3_1:k sp_1_1:k)))))",
                "PaddedInput.shared/ax0.ax1.fused.ax2.fused.ax3.fused.inner": "(min (* (* sp_0_2:k (* (* sp_0_3:k (* sp_0_0:k sp_0_1:k)) (+ (* sp_5_1:k sp_5_0:k) (- (* StrideW:s (* sp_2_2:k (* sp_2_3:k (* sp_2_0:k sp_2_1:k)))) StrideW:s)))) (* (+ (* sp_4_1:k sp_4_0:k) (- (* StrideH:s (* (* sp_1_2:k sp_1_3:k) (* sp_1_0:k sp_1_1:k))) StrideH:s)) (* sp_3_2:k (* sp_3_3:k (* sp_3_0:k sp_3_1:k))))) sp_7_0:k)",
                "placeholder.shared/ax0.ax1.fused.ax2.fused.ax3.fused.outer.outer": "(+ 1 (// (// (+ (* sp_3_3:k (* sp_3_0:k (* sp_3_2:k (* v6:v sp_3_1:k)))) -1) sp_6_0:k) (* sp_3_1:k (* sp_0_1:k (* sp_1_1:k sp_2_1:k)))))",
                "placeholder.shared/ax0.ax1.fused.ax2.fused.ax3.fused.inner": "(min (* v6:v (* (* sp_3_2:k sp_3_3:k) (* sp_3_1:k sp_3_0:k))) sp_6_0:k)",
                "placeholder.shared/di.outer.inner": "sp_4_0:k",
                "placeholder.shared/dj.outer.inner": "sp_5_0:k",
                "placeholder.shared/b.outer.inner": "sp_0_2:k",
                "placeholder.shared/i.outer.inner": "sp_1_2:k",
                "placeholder.shared/j.outer.inner": "sp_2_2:k",
                "placeholder.shared/c.outer.inner": "sp_3_2:k",
                "placeholder.shared/di.inner": "sp_4_1:k",
                "placeholder.shared/dj.inner": "sp_5_1:k",
                "placeholder.shared/b.inner": "sp_0_3:k",
                "placeholder.shared/i.inner": "sp_1_3:k",
                "placeholder.shared/j.inner": "sp_2_3:k",
                "placeholder.shared/c.inner": "sp_3_3:k",
                "DepthwiseConv2d/ax0.inner": "(* sp_0_2:k sp_0_3:k)",
                "DepthwiseConv2d/ax1.inner": "(* sp_1_2:k sp_1_3:k)",
                "DepthwiseConv2d/ax2.inner": "(* sp_2_2:k sp_2_3:k)",
                "DepthwiseConv2d/ax3.inner": "(* sp_3_2:k sp_3_3:k)",
            }
        ),
    ],
    "Const,AdaptiveAvgPool3D": [
        SketchCodeDesc(
            ("FU", "SP", "AN", "AN", "FU", "SP", "AN", "AN", "PR"),
            {
                "blockIdx.x": "(+ (// (- (* (* (* (* N:s OutD:s) OutH:s) OutW:s) C:s) 1) sp_0_0:k) 1)",
                "threadIdx.x": "sp_0_0:k",
                "tensor/rv0": "(- (select (== (mod (* (+ (mod (// (// (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) OutH:s) OutD:s) 1) InD:s) OutD:s) 0) (// (* (+ (mod (// (// (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) OutH:s) OutD:s) 1) InD:s) OutD:s) (+ (// (* (+ (mod (// (// (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) OutH:s) OutD:s) 1) InD:s) OutD:s) 1)) (// (* (mod (// (// (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) OutH:s) OutD:s) InD:s) OutD:s))",
                "tensor/rv1": "(- (select (== (mod (* (+ (mod (// (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) OutH:s) 1) InH:s) OutH:s) 0) (// (* (+ (mod (// (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) OutH:s) 1) InH:s) OutH:s) (+ (// (* (+ (mod (// (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) OutH:s) 1) InH:s) OutH:s) 1)) (// (* (mod (// (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) OutH:s) InH:s) OutH:s))",
                "tensor/rv2": "(- (select (== (mod (* (+ (mod (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) 1) InW:s) OutW:s) 0) (// (* (+ (mod (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) 1) InW:s) OutW:s) (+ (// (* (+ (mod (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) 1) InW:s) OutW:s) 1)) (// (* (mod (// (+ threadIdx.x (* blockIdx.x sp_1_0:k)) C:s) OutW:s) InW:s) OutW:s))",
            },
        )
    ],
    "Const,Const,Const,Const,Conv3d,Broadcast(Mul...),Broadcast(Add...)": [
        SketchCodeDesc(
            ("CI", "FU", "SP", "AN", "FSP", "AN", "CA", "CI", "FU", "AN", "PR"),
            {
                "E0": "Cout:s",
                "v1": "(+ N:s (* N:s (// (+ D:s (- (* PadD:s 2) KD:s)) StrideD:s)))",
                "v2": "(+ v1:v (* v1:v (// (+ H:s (- (* PadH:s 2) KH:s)) StrideH:s)))",
                "v3": "(+ v2:v (* v2:v (// (+ W:s (- (* PadW:s 2) KW:s)) StrideW:s)))",
                "blockIdx.x": "(* v3:v q0:v)",
                "Conv3dOutput/cc": "sp_0_0:k",
                "threadIdx.x": "sp_0_0:k",
                "Conv3dOutput/rd.rh.fused.rw.fused.rc.fused.outer": "(+ (// (- (* (* (* KD:s KH:s) KW:s) Cin:s) 1) sp_0_0:k) 1)",
            },
        ),
        SketchCodeDesc(
            # fmt: off
            ('CI', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'RE', 'FSP', 'FSP', 'FSP', 'FSP', 'FSP', 'RE', 'CA', 'CHR', 'CA', 'CHR', 'CA', 'CI', 'FU', 'AN', 'FU', 'AN', 'FU', 'AN', 'FU', 'SP', 'AN', 'FFSP', 'AN', 'FU', 'SP', 'AN', 'FFSP', 'AN', 'PR'),
            # fmt: on
            {
                "E0": "N:s",
                "E1": "(+ (// (- (+ (* PadD:s 2) D:s) KD:s) StrideD:s) 1)",
                "E2": "(+ (// (- (+ (* PadH:s 2) H:s) KH:s) StrideH:s) 1)",
                "E3": "(+ (// (- (+ (* PadW:s 2) W:s) KW:s) StrideW:s) 1)",
                "E4": "Cout:s",
                "E5": "KD:s",
                "E6": "KH:s",
                "E7": "KW:s",
                "E8": "Cin:s",
                "v9": "(* sp_5_1:k (* sp_6_1:k (* sp_5_0:k sp_6_0:k)))",
                "v10": "(* v9:v (* sp_7_0:k (* sp_7_1:k (* sp_8_1:k sp_8_0:k))))",
                "blockIdx.x": "(* (* (* (* q0:v q1:v) q2:v) q3:v) q4:v)",
                "vthread": "(* (* (* (* sp_0_0:k sp_1_0:k) sp_2_0:k) sp_3_0:k) sp_4_0:k)",
                "threadIdx.x": "(* (* (* (* sp_4_1:k sp_3_1:k) sp_2_1:k) sp_1_1:k) sp_0_1:k)",
                "Conv3dOutput/nn.outer.inner.init": "sp_0_2:k",
                "Conv3dOutput/dd.outer.inner.init": "sp_1_2:k",
                "Conv3dOutput/hh.outer.inner.init": "sp_2_2:k",
                "Conv3dOutput/ww.outer.inner.init": "sp_3_2:k",
                "Conv3dOutput/cc.outer.inner.init": "sp_4_2:k",
                "Conv3dOutput/nn.inner.init": "sp_0_3:k",
                "Conv3dOutput/dd.inner.init": "sp_1_3:k",
                "Conv3dOutput/hh.inner.init": "sp_2_3:k",
                "Conv3dOutput/ww.inner.init": "sp_3_3:k",
                "Conv3dOutput/cc.inner.init": "sp_4_3:k",
                "Conv3dOutput/rd.outer.outer": "q5:v",
                "Conv3dOutput/rh.outer.outer": "q6:v",
                "Conv3dOutput/rw.outer.outer": "q7:v",
                "Conv3dOutput/rc.outer.outer": "q8:v",
                "PaddedInput.shared/ax0.ax1.fused.ax2.fused.ax3.fused.ax4.fused.outer.outer": "(+ 1 (// (// (+ -1 (* (* sp_0_3:k (* (+ (* sp_5_1:k sp_5_0:k) (* StrideD:s (+ -1 (* (* sp_1_2:k sp_1_3:k) (* sp_1_0:k sp_1_1:k))))) (+ sp_0_2:k (* sp_0_2:k (+ (// (// (// (// -1 sp_4_1:k) sp_3_1:k) sp_2_1:k) sp_1_1:k) (* sp_0_1:k (+ sp_0_0:k (+ 1 (// (// (// (// -1 sp_4_0:k) sp_3_0:k) sp_2_0:k) sp_1_0:k))))))))) (* (* sp_8_1:k sp_8_0:k) (* (+ (* sp_6_1:k sp_6_0:k) (- (* (* sp_2_0:k sp_2_1:k) (* sp_2_3:k (* sp_2_2:k StrideH:s))) StrideH:s)) (+ (* sp_7_1:k sp_7_0:k) (- (* StrideW:s (* sp_3_0:k (* sp_3_3:k (* sp_3_1:k sp_3_2:k)))) StrideW:s)))))) sp_10_0:k) (* sp_0_1:k (* sp_2_1:k (* sp_3_1:k (* sp_4_1:k sp_1_1:k))))))",
                "PaddedInput.shared/ax0.ax1.fused.ax2.fused.ax3.fused.ax4.fused.inner": "(min (* (* sp_0_3:k (+ sp_0_2:k (* sp_0_2:k (+ (// (// (// (// -1 sp_4_1:k) sp_3_1:k) sp_2_1:k) sp_1_1:k) (* sp_0_1:k (+ (// (// (// (// -1 sp_4_0:k) sp_3_0:k) sp_2_0:k) sp_1_0:k) (+ sp_0_0:k 1))))))) (* (* (+ (* sp_5_1:k sp_5_0:k) (* StrideD:s (+ -1 (* sp_1_0:k (* sp_1_1:k (* sp_1_2:k sp_1_3:k)))))) (+ (* sp_6_1:k sp_6_0:k) (- (* StrideH:s (* sp_2_0:k (* sp_2_3:k (* sp_2_1:k sp_2_2:k)))) StrideH:s))) (* (+ (* sp_7_1:k sp_7_0:k) (- (* StrideW:s (* (* sp_3_2:k sp_3_3:k) (* sp_3_0:k sp_3_1:k))) StrideW:s)) (* sp_8_1:k sp_8_0:k)))) sp_10_0:k)",
                "placeholder.shared/ax0.ax1.fused.ax2.fused.ax3.fused.ax4.fused.outer.outer": "(+ (// (// (- (* v10:v (* (* sp_4_2:k sp_4_3:k) (* sp_4_1:k sp_4_0:k))) 1) sp_9_0:k) (* (* (* (* sp_4_1:k sp_3_1:k) sp_2_1:k) sp_1_1:k) sp_0_1:k)) 1)",
                "placeholder.shared/ax0.ax1.fused.ax2.fused.ax3.fused.ax4.fused.inner": "(min (* v10:v (* (* sp_4_2:k sp_4_3:k) (* sp_4_1:k sp_4_0:k))) sp_9_0:k)",
                "placeholder.shared/rd.outer.inner": "sp_5_0:k",
                "placeholder.shared/rh.outer.inner": "sp_6_0:k",
                "placeholder.shared/rw.outer.inner": "sp_7_0:k",
                "placeholder.shared/rc.outer.inner": "sp_8_0:k",
                "placeholder.shared/nn.outer.inner": "sp_0_2:k",
                "placeholder.shared/dd.outer.inner": "sp_1_2:k",
                "placeholder.shared/hh.outer.inner": "sp_2_2:k",
                "placeholder.shared/ww.outer.inner": "sp_3_2:k",
                "placeholder.shared/cc.outer.inner": "sp_4_2:k",
                "placeholder.shared/rd.inner": "sp_5_1:k",
                "placeholder.shared/rh.inner": "sp_6_1:k",
                "placeholder.shared/rw.inner": "sp_7_1:k",
                "placeholder.shared/rc.inner": "sp_8_1:k",
                "placeholder.shared/nn.inner": "sp_0_3:k",
                "placeholder.shared/dd.inner": "sp_1_3:k",
                "placeholder.shared/hh.inner": "sp_2_3:k",
                "placeholder.shared/ww.inner": "sp_3_3:k",
                "placeholder.shared/cc.inner": "sp_4_3:k",
                "Conv3dOutput/ax0.inner": "(* sp_0_2:k sp_0_3:k)",
                "Conv3dOutput/ax1.inner": "(* sp_1_2:k sp_1_3:k)",
                "Conv3dOutput/ax2.inner": "(* sp_2_2:k sp_2_3:k)",
                "Conv3dOutput/ax3.inner": "(* sp_3_2:k sp_3_3:k)",
                "Conv3dOutput/ax4.inner": "(* sp_4_2:k sp_4_3:k)",
            }
        ),
    ],
    "Const,Const,TransposeConv2d,Elemwise(tanh...)": [
        SketchCodeDesc(
            ("FU", "SP", "AN", "FSP", "AN", "CA", "CI", "FU", "AN", "PR"),
            {
                "E0": "Cout:s",
                "v1": "(* N:s (- (+ (* (- H:s 1) StrideH:s) KH:s) (* PadH:s 2)))",
                "v2": "(* v1:v (- (+ (* (- W:s 1) StrideW:s) KW:s) (* PadW:s 2)))",
                "blockIdx.x": "(* v2:v q0:v)",
                "conv2d_transpose_nhwc/co": "sp_0_0:k",
                "threadIdx.x": "sp_0_0:k",
                "conv2d_transpose_nhwc/rh.rw.fused.rc.fused.outer": "(+ 1 (// (+ (* KW:s (* KH:s Cin:s)) -1) sp_0_0:k))",
            },
        ),
        SketchCodeDesc(
            # fmt: off
            ('SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'SP', 'RE', 'FSP', 'FSP', 'FSP', 'FSP', 'RE', 'CA', 'CHR', 'CA', 'CHR', 'CA', 'CI', 'FU', 'AN', 'FU', 'AN', 'FU', 'AN', 'FU', 'SP', 'AN', 'FFSP', 'AN', 'FU', 'SP', 'AN', 'FFSP', 'AN', 'PR'),
            # fmt: on
            {
                "E0": "N:s",
                "E1": "(+ (* (- H:s 1) StrideH:s) (+ KH:s (* PadH:s -2)))",
                "E2": "(+ (* (- W:s 1) StrideW:s) (+ KW:s (* PadW:s -2)))",
                "E3": "Cout:s",
                "E4": "KH:s",
                "E5": "KW:s",
                "E6": "Cin:s",
                "blockIdx.x": "(* q0:v (* q2:v (* q1:v q3:v)))",
                "vthread": "(* sp_0_0:k (* sp_2_0:k (* sp_1_0:k sp_3_0:k)))",
                "threadIdx.x": "(* sp_3_1:k (* sp_2_1:k (* sp_1_1:k sp_0_1:k)))",
                "conv2d_transpose_nhwc/n.outer.inner.init": "sp_0_2:k",
                "conv2d_transpose_nhwc/h.outer.inner.init": "sp_1_2:k",
                "conv2d_transpose_nhwc/w.outer.inner.init": "sp_2_2:k",
                "conv2d_transpose_nhwc/co.outer.inner.init": "sp_3_2:k",
                "conv2d_transpose_nhwc/n.inner.init": "sp_0_3:k",
                "conv2d_transpose_nhwc/h.inner.init": "sp_1_3:k",
                "conv2d_transpose_nhwc/w.inner.init": "sp_2_3:k",
                "conv2d_transpose_nhwc/co.inner.init": "sp_3_3:k",
                "conv2d_transpose_nhwc/rh.outer.outer": "q4:v",
                "conv2d_transpose_nhwc/rw.outer.outer": "q5:v",
                "conv2d_transpose_nhwc/rc.outer.outer": "q6:v",
                "PadInput.shared/ax0.ax1.fused.ax2.fused.ax3.fused.outer.outer": "(+ 1 (// (// (+ (* (+ 1 (- (// (+ (* sp_4_1:k (+ sp_4_0:k (* sp_4_0:k rh.outer.outer))) (+ (mod (- StrideH:s (mod (+ KH:s (- -1 PadH:s)) StrideH:s)) StrideH:s) (+ (* (+ 1 (mod (// (// blockIdx.x q3:v) q2:v) q1:v)) (* sp_1_2:k (* sp_1_0:k (* sp_1_1:k sp_1_3:k)))) -2))) StrideH:s) (// (+ (* sp_1_0:k (* (mod (// (// blockIdx.x q3:v) q2:v) q1:v) (* sp_1_1:k (* sp_1_2:k sp_1_3:k)))) (+ (mod (- StrideH:s (mod (+ KH:s (- -1 PadH:s)) StrideH:s)) StrideH:s) (* sp_4_1:k (* sp_4_0:k rh.outer.outer)))) StrideH:s))) (* sp_0_2:k (* (+ (// (+ (+ (* sp_5_0:k (+ sp_5_1:k (* sp_5_1:k rw.outer.outer))) (* (* sp_2_2:k (* sp_2_3:k sp_2_1:k)) (+ sp_2_0:k (* sp_2_0:k (mod (// blockIdx.x q3:v) q2:v))))) (+ (mod (- StrideW:s (mod (+ (- KW:s PadW:s) -1) StrideW:s)) StrideW:s) -2)) StrideW:s) (- 1 (// (+ (* sp_2_0:k (* sp_2_2:k (* (mod (// blockIdx.x q3:v) q2:v) (* sp_2_3:k sp_2_1:k)))) (+ (mod (- StrideW:s (mod (+ (- KW:s PadW:s) -1) StrideW:s)) StrideW:s) (* sp_5_0:k (* sp_5_1:k rw.outer.outer)))) StrideW:s))) (* sp_6_1:k (* sp_0_0:k (* sp_0_3:k (* sp_6_0:k sp_0_1:k))))))) -1) sp_8_0:k) (* sp_1_1:k (* sp_2_1:k (* sp_0_1:k sp_3_1:k)))))",
                "PadInput.shared/ax0.ax1.fused.ax2.fused.ax3.fused.inner": "(min (* sp_0_2:k (* (+ 1 (- (// (+ (* sp_4_1:k (+ sp_4_0:k (* sp_4_0:k rh.outer.outer))) (+ (* sp_1_0:k (* (+ 1 (mod (// (// blockIdx.x q3:v) q2:v) q1:v)) (* sp_1_1:k (* sp_1_2:k sp_1_3:k)))) (+ (mod (- StrideH:s (mod (- (+ KH:s -1) PadH:s) StrideH:s)) StrideH:s) -2))) StrideH:s) (// (+ (mod (- StrideH:s (mod (- (+ KH:s -1) PadH:s) StrideH:s)) StrideH:s) (+ (* (mod (// (// blockIdx.x q3:v) q2:v) q1:v) (* sp_1_3:k (* sp_1_2:k (* sp_1_0:k sp_1_1:k)))) (* sp_4_1:k (* sp_4_0:k rh.outer.outer)))) StrideH:s))) (* (+ 1 (- (// (+ (* sp_2_0:k (* (* sp_2_2:k sp_2_3:k) (+ sp_2_1:k (* sp_2_1:k (mod (// blockIdx.x q3:v) q2:v))))) (+ (mod (- StrideW:s (mod (+ KW:s (- -1 PadW:s)) StrideW:s)) StrideW:s) (+ (* sp_5_0:k (+ sp_5_1:k (* sp_5_1:k rw.outer.outer))) -2))) StrideW:s) (// (+ (mod (- StrideW:s (mod (+ KW:s (- -1 PadW:s)) StrideW:s)) StrideW:s) (+ (* sp_2_2:k (* (* sp_2_1:k (mod (// blockIdx.x q3:v) q2:v)) (* sp_2_0:k sp_2_3:k))) (* sp_5_1:k (* sp_5_0:k rw.outer.outer)))) StrideW:s))) (* sp_6_1:k (* sp_6_0:k (* sp_0_0:k (* sp_0_3:k sp_0_1:k))))))) sp_8_0:k)",
                "placeholder.shared/ax0.ax1.fused.ax2.fused.ax3.fused.outer.outer": "(+ 1 (// (// (+ (* sp_4_1:k (* sp_4_0:k (* sp_5_0:k (* sp_6_1:k (* (* sp_5_1:k sp_3_3:k) (* sp_3_1:k (* sp_3_2:k (* sp_3_0:k sp_6_0:k)))))))) -1) sp_7_0:k) (* sp_3_1:k (* sp_1_1:k (* sp_2_1:k sp_0_1:k)))))",
                "placeholder.shared/ax0.ax1.fused.ax2.fused.ax3.fused.inner": "(min (* sp_4_1:k (* (* sp_3_0:k (* sp_3_1:k sp_6_1:k)) (* (* sp_5_1:k (* sp_4_0:k sp_5_0:k)) (* sp_3_2:k (* sp_6_0:k sp_3_3:k))))) sp_7_0:k)",
                "placeholder.shared/rh.outer.inner": "sp_4_0:k",
                "placeholder.shared/rw.outer.inner": "sp_5_0:k",
                "placeholder.shared/rc.outer.inner": "sp_6_0:k",
                "placeholder.shared/n.outer.inner": "sp_0_2:k",
                "placeholder.shared/h.outer.inner": "sp_1_2:k",
                "placeholder.shared/w.outer.inner": "sp_2_2:k",
                "placeholder.shared/co.outer.inner": "sp_3_2:k",
                "placeholder.shared/rh.inner": "sp_4_1:k",
                "placeholder.shared/rw.inner": "sp_5_1:k",
                "placeholder.shared/rc.inner": "sp_6_1:k",
                "placeholder.shared/n.inner": "sp_0_3:k",
                "placeholder.shared/h.inner": "sp_1_3:k",
                "placeholder.shared/w.inner": "sp_2_3:k",
                "placeholder.shared/co.inner": "sp_3_3:k",
                "conv2d_transpose_nhwc/ax0.inner": "(* sp_0_2:k sp_0_3:k)",
                "conv2d_transpose_nhwc/ax1.inner": "(* sp_1_2:k sp_1_3:k)",
                "conv2d_transpose_nhwc/ax2.inner": "(* sp_2_2:k sp_2_3:k)",
                "conv2d_transpose_nhwc/ax3.inner": "(* sp_3_2:k sp_3_3:k)",
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


def check_tasks(tasks: List[sym_task.SymTaskAndInstances], expected: Dict[str, List[int]]):
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
            assert len(ref_sk_.vardefs) == len(vardefs)
            # vardefs_ = {k: ffi.print_expr_preorder(v) for k, v in vardefs.items()}
            # print(vardefs_)
            for k, expr in vardefs.items():
                assert ffi.is_expr_equivalent(expr, ref_sk_.vardefs[k])


@tvm.testing.requires_gpu
@pytest.mark.parametrize("network_name", ["resnet50", "mobilenet_v2", "r3d_18", "dcgan"])
def test_extract_task(network_name: str):
    network, shape = NETWORKS[network_name]()
    inputs = torch.randn(shape)
    tasks = sym_task.extract_tasks_(network, inputs, save_to=f"/tmp/{network_name}.pkl")
    check_tasks(tasks, NETWORK_TASKS[network_name])
    tasks_ = sym_task.load_and_register_tasks_(f"/tmp/{network_name}.pkl")
    check_tasks(tasks_, NETWORK_TASKS[network_name])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
