# mypy: disable-error-code="attr-defined"
from typing import Dict, List, Tuple

import tvm
from tvm import auto_scheduler as ansor
from tvm import tir
from tvm.arith import _ffi_api as _arith
from tvm.auto_scheduler import _ffi_api as _ansor
from tvm.auto_scheduler.loop_state import StateObject
from tvm.auto_scheduler.search_policy import SketchPolicy
from tvm.tir import _ffi_api as _tir


@tvm._ffi.register_object("arith.VarContext")
class VarContext(tvm.Object):
    def to_str(self) -> str:
        return _arith.VarMapToStr(self)


# TIR Expr Utils


def subst_by_name(expr: tir.PrimExpr, kvs: Dict[str, tir.PrimExpr]) -> tir.PrimExpr:
    return _tir.SubstByName(expr, kvs)


# Felix main functionality entry points


def generate_all_sym_sketches(policy: SketchPolicy) -> List[StateObject]:
    return _ansor.GenerateAllSymSketches(policy)


def extract_backbone(state: StateObject) -> tuple[str, ...]:
    return tuple(_ansor.ExtractBackbone(state.transform_steps))


def generate_code_for_state(
    task: ansor.SearchTask, state: StateObject, is_symbolic: bool
) -> Tuple[tir.Stmt, VarContext]:
    return _ansor.GenerateCodeForState(task, state, is_symbolic)


def print_state_tr_steps(state: StateObject) -> str:
    return "\n".join(_ansor.PrintTrStep(s) for s in state.transform_steps)
