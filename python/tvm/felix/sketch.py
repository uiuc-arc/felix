import logging
from pathlib import Path
from typing import Dict, Tuple

from torch import Tensor, nn
from tvm.auto_scheduler.cost_model import MLPCostModel
from tvm.auto_scheduler.loop_state import StateObject
from tvm.tir import Stmt

from . import ffi
from .features import TorchFeatures
from .utils import HW_PARAMS

_logger = logging.getLogger(__name__)
__all__ = ["Sketch"]
FEATURE_CACHE_PATH = Path(Path("~").expanduser(), ".tvm", "felix", "features")


class Sketch:
    def __init__(self, task, sym_state: StateObject):
        from .sym_task import SymTask

        self.parent_task: SymTask = task
        self.state_repr = str(sym_state)
        self.code, self.context = ffi.generate_code_for_state(task.ansor_task, sym_state, True)
        self.backbone = ffi.extract_backbone(sym_state)

        _logger.debug("Sketch transformation steps: %s", ffi.print_state_tr_steps(sym_state))
        _logger.debug("Code: %s", self.code)
        _logger.debug("With loop sizes:\n%s", self.context.to_varmap())

    def state_hash(self) -> str:
        import hashlib

        md5 = hashlib.md5()
        md5.update(self.state_repr.encode())
        return md5.hexdigest()

    def get_flops(self, sizes):
        return self.parent_task.get_flops(sizes)

    def generate_code(self, size_info: dict, state: StateObject) -> Stmt:
        task, _ = self.parent_task.make_concrete_task(size_info)
        return ffi.generate_code_for_state(task, state, False)[0]

    def save_path(self) -> Path:
        return FEATURE_CACHE_PATH / f"{self.state_hash()}.json"

    def fetch_features(
        self,
        sizes: Dict[str, int],
        prime_factorize: bool = True,
        max_n_buf: int = 5,
        cache_line_size: int = 64,
    ):
        path = self.save_path()
        path.parent.mkdir(exist_ok=True, parents=True)
        features = ffi.get_feature_pack(
            self.code,
            self.context,
            HW_PARAMS,
            sizes,
            cache_line_size,
            max_n_buf,
            prime_factorize,
            path.as_posix(),
        )
        return TorchFeatures.from_feat_pack(features)

    def __str__(self) -> str:
        return f"Sketch({self.backbone} from {self.parent_task})"

    __repr__ = __str__


class SketchPerfFunc(nn.Module):
    def __init__(self, sketch: Sketch, features: TorchFeatures, cost_f: MLPCostModel) -> None:
        super().__init__()
        self.sketch = sketch
        self.features = features
        self.cost_f = cost_f

    @classmethod
    def from_sketch(cls, sketch: Sketch, sizes: dict, cost_f: MLPCostModel) -> "SketchPerfFunc":
        features = sketch.fetch_features(sizes)
        return cls(sketch, features, cost_f)

    def forward(self, configs: Tensor) -> Tuple[Tensor, Tensor]:
        feats, constraints = self.features(configs)
        perf = self.cost_f.forward(feats)
        return perf, constraints
