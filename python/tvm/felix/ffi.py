# mypy: disable-error-code="attr-defined"
from typing import Dict, List, Tuple

import tvm
from tvm import auto_scheduler as ansor
from tvm import tir
from tvm.arith import _ffi_api as _arith
from tvm.auto_scheduler import _ffi_api as _ansor
from tvm.auto_scheduler.loop_state import StateObject
from tvm.auto_scheduler.search_policy import SketchPolicy
from tvm.felix import _ffi_api as _felix
from tvm.tir import _ffi_api as _tir


@tvm._ffi.register_object("arith.VarContext")
class VarContext(tvm.Object):
    def to_varmap(self) -> Dict[tir.Var, tir.PrimExpr]:
        return dict(_arith.VarContextGetVarDefs(self))


@tvm._ffi.register_object("felix.LinearExpr")
class LinearExpr(tvm.Object):
    lin_terms: Dict[tir.Var, float]
    constant: float

    def as_primexpr(self) -> tir.PrimExpr:
        return _felix.LinearExprAsPrimExpr(self)


@tvm._ffi.register_object("felix.FeaturePackPy")
class FeaturePack(tvm.Object):
    expressions: List[Tuple[str, tir.PrimExpr]]
    free_vars: List[str]
    linear_cons: List[LinearExpr]
    var_decomp: Dict[str, Dict[int, tir.SizeVar]]


# PrimExpr Utils


def subst_by_name(expr: tir.PrimExpr, kvs: Dict[str, tir.PrimExpr]) -> tir.PrimExpr:
    return _tir.SubstByName(expr, kvs)


def is_expr_equivalent(e1: tir.PrimExpr, e2: tir.PrimExpr) -> bool:
    return _arith.IsExprEquivalent(e1, e2, 30, 10000, False)


def print_expr_preorder(expr: tir.PrimExpr) -> str:
    return _arith.PrintExprPreorder(expr)


def parse_expr_preorder(expr: str) -> tir.PrimExpr:
    return _arith.ParseExprPreorder(expr)


# Felix main functionality entry points


def generate_all_sym_sketches(policy: SketchPolicy) -> List[StateObject]:
    return _felix.GenerateAllSymSketches(policy)


def extract_backbone(state: StateObject) -> tuple[str, ...]:
    return tuple(_felix.ExtractBackbone(state.transform_steps))


def generate_code_for_state(
    task: ansor.SearchTask, state: StateObject, is_symbolic: bool
) -> Tuple[tir.Stmt, VarContext]:
    return _felix.GenerateCodeForState(task, state, is_symbolic)


def print_state_tr_steps(state: StateObject) -> str:
    return "\n".join(_felix.PrintTrStep(s) for s in state.transform_steps)


def get_feature_pack(
    code: tir.Stmt,
    context: VarContext,
    hw_params: ansor.HardwareParams,
    sizes: Dict[str, int],
    cache_line_size: int,
    max_n_buf: int,
    factorize: bool,
    save_load_path: str,
) -> FeaturePack:
    return _felix.GetFeaturePack(
        code,
        context,
        hw_params,
        sizes,
        cache_line_size,
        max_n_buf,
        factorize,
        save_load_path,
    )


def get_loop_bounds(code: tir.Stmt) -> List[Tuple[str, tir.PrimExpr]]:
    return list(_felix.GetLoopBounds(code))


# Additional Felix utils to interface with Ansor data structures


def extract_config_dict(state: StateObject) -> Dict[str, int]:
    return {str(k): int(v) for k, v in _felix.ExtractConfigDict(state.transform_steps).items()}
