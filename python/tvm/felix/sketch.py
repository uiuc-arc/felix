import logging
from pathlib import Path

from tvm.auto_scheduler.loop_state import StateObject
from tvm.tir import Stmt

from . import ffi

_logger = logging.getLogger(__name__)
__all__ = ["Sketch"]
CACHE_PATH = (Path(__file__).parent / "../../lightning_logs/features").resolve()


class Sketch:
    def __init__(self, task, sym_state: StateObject):
        from .sym_task import SymTask

        self.parent_task: SymTask = task
        self.state = sym_state
        self.code, self.context = ffi.generate_code_for_state(task.ansor_task, sym_state, True)
        self.print_transform_and_code()
        self.backbone = ffi.extract_backbone(sym_state)

    def state_hash(self) -> str:
        import hashlib

        md5 = hashlib.md5()
        md5.update(str(self.state).encode())
        return md5.hexdigest()

    def get_flops(self, sizes):
        return self.parent_task.get_flops(sizes)

    def generate_code(self, size_info: dict, state: StateObject) -> Stmt:
        task, _ = self.parent_task.make_concrete_task(size_info)
        return ffi.generate_code_for_state(task, state, False)[0]

    def save_path(self) -> Path:
        return CACHE_PATH / f"{self.state_hash()}.json"

    def __str__(self) -> str:
        return f"Sketch({self.backbone} from {self.parent_task})"

    __repr__ = __str__

    # Debug API below.

    def print_transform_and_code(self):
        steps_str = ffi.print_state_tr_steps(self.state)
        _logger.debug(f"Sketch transformation steps: {steps_str}")
        _logger.debug(f"Code: {self.code}")
        _logger.debug("With loop sizes:\n%s", self.context.to_str())
