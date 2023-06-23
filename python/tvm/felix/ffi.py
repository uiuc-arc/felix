# mypy: disable-error-code="attr-defined"
from typing import Dict, List

from tvm import tir
from tvm.auto_scheduler import _ffi_api as _ansor
from tvm.auto_scheduler.loop_state import StateObject
from tvm.auto_scheduler.search_policy import SketchPolicy
from tvm.tir import _ffi_api as _tir

# TIR Expr Utils


def subst_by_name(expr: tir.PrimExpr, kvs: Dict[str, tir.PrimExpr]) -> tir.PrimExpr:
    return _tir.SubstByName(expr, kvs)


# Felix main functionality entry points


def generate_all_sym_sketches(policy: SketchPolicy) -> List[StateObject]:
    return _ansor.GenerateAllSymSketches(policy)
