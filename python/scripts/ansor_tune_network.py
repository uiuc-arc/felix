import logging
from pathlib import Path

import torch
from torchvision.models import mobilenet_v2, resnet50
from torchvision.models.video import r3d_18
from tvm import felix

logger = logging.getLogger(__name__)


def get_network(name: str, batch_size: int):
    from tvm.felix import nn

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
    else:
        raise ValueError(f"Invalid network: {name}")
    return network, inputs


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--network", required=True, type=str)
    parser.add_argument("-b", "--batch-size", required=True, type=int)
    parser.add_argument("--iters", required=True, type=int)
    parser.add_argument("--per-layer", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = str(felix.utils.TARGET.model)
    work_dir = Path(
        "./lightning_logs", device, "tuning_results", f"{args.network}_{args.batch_size}"
    )
    cost_model = f"lightning_logs/{device}/cost_models/xgb.pkl"
    felix.init_logging(work_dir, True)
    tasks_pkl = work_dir / "ansor_tasks.pkl"
    if tasks_pkl.is_file():
        tasks = felix.utils.load_and_register_ansor_tasks(tasks_pkl, True)
    else:
        model, inputs = get_network(args.network, args.batch_size)
        tasks = felix.utils.extract_ansor_tasks(model, inputs, save_to=tasks_pkl)
        del model
    if args.per_layer:
        for idx in range(len(tasks)):
            felix.ansor_tune_full(
                tasks[idx : idx + 1], cost_model, work_dir / f"ansor_t{idx}.json", args.iters
            )
    else:
        felix.ansor_tune_full(tasks, cost_model, work_dir / "ansor_configs.json", args.iters)


if __name__ == "__main__":
    main()
