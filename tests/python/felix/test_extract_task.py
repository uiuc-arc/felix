import sys

import pytest
import torch
import tvm
import tvm.testing
from torchvision.models import mobilenet_v2, resnet50
from torchvision.models.video import r3d_18
from tvm import felix
from tvm.felix import nn


def check_tasks(tasks, expected):
    assert len(tasks) == len(expected)
    for task, instances in tasks:
        desc = str(task)
        assert desc in expected
        expected_indices = expected[desc]
        actual_indices = [idx for (idx, _, _) in instances]
        assert expected_indices == actual_indices


# fmt: off

@tvm.testing.requires_gpu
def test_extract_resnet50():
    network = resnet50()
    inputs = torch.randn(1, 3, 256, 256)
    tasks = felix.extract_tasks(network, inputs)
    check_tasks(tasks, {
        "Const,AdaptiveAvgPool2D": [0],
        "Const,Const,Const,Conv2d,Broadcast(Add...)": list(range(1, 4 + 1)),
        "Const,Const,Const,Const,Conv2d,Broadcast(Add...),Broadcast(Add...),Elemwise(relu...)": list(range(5, 8 + 1)),
        "Const,Const,Const,Conv2d,Broadcast(Add...),Elemwise(relu...)": list(range(9, 24 + 1)),
        "Const,Const,Const,Dense,Broadcast(Add...)": [25],
        "Const,Max2DPool[Padded]": [26],
    })

@tvm.testing.requires_gpu
def test_extract_mobilenet_v2():
    network = mobilenet_v2()
    inputs = torch.randn(1, 3, 256, 256)
    tasks = felix.extract_tasks(network, inputs)
    check_tasks(tasks, {
        "Const,AdaptiveAvgPool2D": [0],
        "Const,Const,Const,Conv2d,Broadcast(Add...)": list(range(1, 7 + 1)),
        "Const,Const,Const,Const,Conv2d,Broadcast(Add...),Broadcast(Add...)": [8, 9, 10, 11, 12],
        "Const,Const,Const,Conv2d,Broadcast(Add...),Elemwise(clip...)": [13, 15, 17, 20, 23, 25, 28, 30],
        "Const,Const,Const,DepthwiseConv2d,Broadcast(Add...),Elemwise(clip...)": [14, 16, 18, 19, 21, 22, 24, 26, 27, 29],
        "Const,Const,Const,Dense,Broadcast(Add...)": [31],
    })


@tvm.testing.requires_gpu
def test_extract_r3d_18():
    network = r3d_18()
    inputs = torch.randn(1, 3, 16, 112, 112)
    tasks = felix.extract_tasks(network, inputs)
    check_tasks(tasks, {
        "Const,Const,Const,Dense,Broadcast(Add...)": [0],
        "Const,AdaptiveAvgPool3D": [1],
        "Const,Const,Const,Const,Conv3d,Broadcast(Mul...),Broadcast(Add...)": [2, 3, 4],
        "Const,Const,Const,Const,Const,Conv3d,Broadcast(Mul...),Broadcast(Add...),Broadcast(Add...),Elemwise(relu...)": [5, 6, 7, 8],
        "Const,Const,Const,Const,Conv3d,Broadcast(Mul...),Broadcast(Add...),Elemwise(relu...)": list(range(9, 16 + 1))
    })


@tvm.testing.requires_gpu
def test_extract_dcgan():
    network, input_shape = nn.dcgan()
    inputs = torch.randn(1, *input_shape)
    tasks = felix.extract_tasks(network, inputs)
    check_tasks(tasks, {
        "Const,Const,Conv2d,Elemwise(sigmoid...)": [0],
        "Const,Const,Const,Conv2d,Broadcast(Add...),Elemwise(leaky_relu...)": [1, 2, 3],
        "Const,Const,Conv2d,Elemwise(leaky_relu...)": [4],
        "Const,Const,Const,Const,TransposeConv2d,Broadcast(Mul...),Broadcast(Add...),Elemwise(relu...)": [5, 6, 7, 8],
        "Const,Const,TransposeConv2d,Elemwise(tanh...)": [9],
    })


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
