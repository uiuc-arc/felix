import logging
from pathlib import Path

import torch
from torchvision.models import mobilenet_v2, resnet50
from torchvision.models.video import r3d_18
from tvm import felix
from tvm.felix import nn

logger = logging.getLogger(__name__)


def get_network(name: str, batch_size: int):
    if name == "resnet50":
        network = resnet50()
        inputs = torch.randn(batch_size, 3, 256, 256)
    elif name == "mobilenet_v2":
        network = mobilenet_v2()
        inputs = torch.randn(batch_size, 3, 256, 256)
    elif name == "r3d_18":
        network = r3d_18()
        inputs = torch.randn(batch_size, 3, 16, 112, 112)
    elif name == "dcgan":
        network, input_size = nn.dcgan()
        inputs = torch.randn(batch_size, *input_size)
    elif name == "vit":
        network, input_size = nn.vit()
        inputs = torch.randn(batch_size, *input_size)
    elif name == "llama_100":
        network, inputs = nn.llama()
    else:
        raise ValueError(f"Invalid network: {name}")
    return network, inputs


def main():
    network, batch_size = "resnet50", 1
    work_dir = Path(f"/tmp/{network}_{batch_size}")
    felix.init_logging(work_dir, True)
    pkl = work_dir / "felix_tasks.pkl"
    if pkl.is_file():
        tasks = felix.load_and_register_tasks(pkl)
    else:
        model, inputs = get_network(network, batch_size)
        tasks = felix.extract_tasks(model, inputs, pkl)
        del model
    perf_model = felix.MLPModelPLWrapper.load_from_checkpoint(
        "_cost_models/epoch=144-loss=0.0665.ckpt"
    )
    optim = felix.Optimizer(tasks, perf_model)
    optim.tune(len(tasks) * 512, 16, 128, 16, work_dir / f"felix_configs.json")


if __name__ == "__main__":
    main()
