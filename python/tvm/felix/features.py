import logging
import typing as ty
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
from sympy.ntheory import factorint
from torch import Tensor, nn
from torch.fx._symbolic_trace import symbolic_trace
from torch.fx.graph_module import GraphModule
from tvm import tir

from . import ffi
from .utils import transpose2

__all__ = ["TorchFeatures"]
logger = logging.getLogger(__name__)

Number = ty.Union[int, float]
T = ty.TypeVar("T")


class TorchFeatures(nn.Module):
    def __init__(
        self,
        features_f,
        constraints_f,
        lin_cons: list,
        var_order: Dict[str, int],
        var_factors: Dict[str, Dict[int, tir.SizeVar]],
        n_feats: int,
        n_bufs: int,
    ):
        super().__init__()
        self.features_f = features_f
        self.constraints_f = constraints_f
        self.lin_cons = lin_cons
        self.var_order = var_order
        self.var_factors = {
            k: {int(prime): v.name for prime, v in ds.items()} for k, ds in var_factors.items()
        }
        self.n_feats, self.n_bufs = n_feats, n_bufs
        self.n_consts = 0
        self._dummy_param = nn.Parameter(torch.empty(0))  # type: ignore
        self._conf_maker: Optional[RandConfigMaker] = None

    @classmethod
    def from_feat_pack(cls, feat: ffi.FeaturePack) -> "TorchFeatures":
        var_order = {var: i for i, var in enumerate(sorted(feat.free_vars))}
        features = defaultdict(list)
        other_cons = []
        vdefs = {}
        for vname, expr in feat.expressions:
            if vname.startswith("BS"):
                bufstore_idx = int(vname.split(".")[0][2:])
                features[bufstore_idx].append(expr)
            elif vname.startswith("con_"):
                other_cons.append(expr)
            else:
                vdefs[vname] = expr
        features_ = np.array([features[i] for i in range(len(features))])
        n_bufs, n_feats = features_.shape
        features_f = TorchExprRunner(features_, var_order, vdefs).get_traced()
        lin_cons = list(feat.linear_cons)
        all_cons = np.array([c.as_primexpr() for c in lin_cons] + other_cons)
        cons_f = TorchExprRunner(all_cons, var_order, vdefs).get_traced()
        return cls(features_f, cons_f, lin_cons, var_order, feat.var_factors, n_feats, n_bufs)

    @property
    def device(self):
        return self._dummy_param.device

    def rand_configs(self, n_configs: int):
        if self._conf_maker is None:
            self._conf_maker = RandConfigMaker(
                list(self.var_order.keys()),
                self.lin_cons,
                self.constraints_f,
                self.device,
            )
        return self._conf_maker.rand_configs(n_configs)

    def forward(self, params: Tensor) -> Tuple[Tensor, Tensor]:
        # params: [batch_size, n_vars]
        # features: [batch_size, n_bufs, n_feats]
        features = self.features_f(params)
        # leq_0s: [batch_size, n_constraints]
        leq_0s = self.constraints_f(params)
        return features, leq_0s

    def run_on_initial_configs(self, initial_configs: Sequence[dict]):
        configs = [self.transform_config(c) for c in initial_configs]
        return self(torch.stack(configs, dim=0))

    def transform_config(self, config: Dict[str, Number]):
        stage1: Dict[str, Number] = {}
        for var_name, value in config.items():
            if var_name not in self.var_factors:
                stage1[var_name] = value
                continue
            decomposed = self.var_factors[var_name]
            if value == 0:
                for vname in decomposed.values():
                    stage1[vname] = -10  # HACK
                continue
            if len(decomposed) == 1:
                ((b, vname),) = decomposed.items()
                if b < 2:
                    raise ValueError()
                stage1[vname] = np.log(value) / np.log(b)
                continue
            factors: Dict[int, int] = factorint(value)
            uncovered_ps = set(factors.keys()) - set(decomposed.keys())
            if uncovered_ps:
                bases = list(decomposed.keys())
                raise ValueError(f"Cannot factorize {value} into {bases}")
            for basis, vname in decomposed.items():
                power = factors.get(basis, 0)
                stage1[vname] = power
        stage2: List[Optional[float]] = [None] * len(self.var_order)
        for name, value in stage1.items():
            if name not in self.var_order:
                continue
            idx = self.var_order[name]
            stage2[idx] = value
        if any(x is None for x in stage2):
            required = set(self.var_order.keys())
            got = set(stage1.keys())
            raise ValueError(f"Missing value for some symbols {required - got}")
        return torch.tensor(cast(List[float], stage2)).float()

    def inv_transform_config(self, config: Tensor):
        def expr_to_int(config, d: Dict[int, str]):
            v = np.prod([prime ** config[vname] for prime, vname in d.items()])
            if not np.isclose(v, int(v)):
                raise ValueError(f"Cannot convert {v} to int")
            return int(v)

        stage1 = {k: config[idx].item() for k, idx in self.var_order.items()}
        return {var: expr_to_int(stage1, d) for var, d in self.var_factors.items()}

    def _concat_knobs(self, consts: Tensor, knobs: Tensor):
        assert consts.shape[0] == knobs.shape[0]
        assert consts.shape[1] == self.n_consts
        assert consts.shape[1] + knobs.shape[1] == self.n_vars
        return torch.cat([consts.float(), knobs.float()], dim=1)


class RandConfigMaker:
    def __init__(
        self,
        variables: List[str],
        lin_cons: List[ffi.LinearExpr],
        constraints_f,
        device,
    ) -> None:
        self.variables = variables
        self.lin_cons = lin_cons
        self.constraints_f = constraints_f
        self.device = device
        coefs, biases, self.bounds = self._estimate_var_bounds()
        # Make configs in [0, 1]^n but zoom them before returning.
        # Consider a single constraint: \sum A_ij x_j + b <= 0.
        # Now x_i \in [0, bound_i]; make x'_i := x_i / bound_i, then x'_i \in [0, 1].
        # And the constraint becomes: \sum (A_ij * bound_j) x'_j + b <= 0.
        # Therefore:
        self.coefs, self.biases = coefs * self.bounds.unsqueeze(0), biases

    def rand_configs(self, n_configs: int):
        # This point (0) should be in the polyhedron (on the boundary).
        current = torch.zeros((1, self.coefs.shape[1]), device=self.device)
        while len(current) < n_configs + 1:
            next = self.rand_next_points(current)
            # Don't insert scale-restored points into `configs` yet.
            # We need to work with [0, 1]^n points here.
            current = torch.cat([current, next], dim=0)
        # ret: [n_configs, n_vars] (discard the first point which is (0))
        return current[1 : n_configs + 1] * self.bounds

    def rand_next_points(self, xs: Tensor):
        # https://mathoverflow.net/questions/9854
        assert xs.shape[1] == self.coefs.shape[1]
        alpha = torch.rand_like(xs, device=self.device)
        alpha = alpha / torch.linalg.norm(alpha, dim=1, keepdim=True)
        # Solve A_i . (x + t_i alpha) + b_i <= 0 (. is for dot prod)
        #   -> t_i ? (-b - A_i . x) / (A_i . alpha)
        #      (whether t_i is a lower or upper bound depends on the sign of A_i . alpha)
        # alpha, xs: [n_configs, n_vars]; self.coefs: [n_constraints, n_vars]
        denoms = torch.tensordot(alpha, self.coefs, dims=([1], [1]))  # type: ignore
        a_dot_x = torch.tensordot(xs, self.coefs, dims=([1], [1]))  # type: ignore
        # denoms, a_dot_x: [n_configs, n_constraints]; self.biases: [n_constraints]
        ts: Tensor = (-self.biases - a_dot_x) / denoms
        # ts: [n_configs, n_constraints]
        # NOTE: for each row, we would have at least one column where denoms > 0
        # and at least one column where denoms < 0.
        # Otherwise the problem would be unbounded.
        # (If that actually happens, this code doesn't detect it.)
        min_ubounds = torch.where(denoms > 0, ts, +1e9).min(dim=1).values
        max_lbounds = torch.where(denoms < 0, ts, -1e9).max(dim=1).values
        t_rands = []
        for i in range(len(xs)):
            t_min, t_max = max_lbounds[i], min_ubounds[i]
            assert t_min <= 0 <= t_max
            # Pick a uniformly random value in [t_min, -d] \union [d, t_max]
            # (where d == 0.05) if there is such space, to prevent too small
            # of a step from previous configuration.
            # (If space is not enough for this operation, t is then just uniformly
            # sampled from [t_min, t_max]).
            t_rands.append(self._rand_uniform_with_gap(t_min, t_max))
        return xs + torch.tensor(t_rands).unsqueeze(1) * alpha

    @staticmethod
    def _rand_uniform_with_gap(left: float, right: float, half_gap: float = 0.05):
        assert left <= 0 <= right

        def rand_range_f(l0, r0):
            return np.random.rand() * (r0 - l0) + l0

        if right <= half_gap and left >= -half_gap:
            # Not enough space to evade [-half_gap, half_gap].
            return rand_range_f(left, right)
        l_size = max(0, -left - half_gap)
        r_size = max(0, right - half_gap)
        x = rand_range_f(-l_size, r_size)
        return -half_gap + x if x < 0 else half_gap + x

    def _estimate_var_bounds(self):
        from scipy.optimize import linprog

        def _get_coefs(poly: ffi.LinearExpr):
            lin_terms = {repr(k): v for k, v in poly.lin_terms.items()}
            coefs = [float(lin_terms.get(s, 0)) for s in self.variables]
            return coefs, float(poly.constant)

        coefs_biases = [_get_coefs(poly) for poly in self.lin_cons]
        coefs, biases = transpose2(coefs_biases)
        coefs, biases = np.array(coefs), np.array(biases)
        # Run linear optimization with scipy.
        vars = self.variables
        var_bounds = []
        for i in range(len(vars)):
            coef_c = np.zeros(len(vars))
            coef_c[i] = -1  # To maximize x[i]
            result = linprog(coef_c, coefs, -biases)
            if result.success:
                var_bounds.append(result.x[i])
            else:
                raise RuntimeError(f"Bound inference failed. Diagnostics: \n{result}")

        def np2torch(x):
            return torch.tensor(x, device=self.device).float()

        return np2torch(coefs), np2torch(biases), np2torch(var_bounds)


def safe_log(x: torch.Tensor):
    # To prevent Infs and NaNs in the result, use this:
    return torch.clamp_min(x, 1).log()


def safe_div(x: torch.Tensor, y: torch.Tensor):
    y = y.sign() * torch.clamp_min(y.abs(), 1e-6)
    return x / y


class TorchExprRunner:
    TIR_OP_TORCH_FN = {
        tir.Add: (torch.add, ["a", "b"]),
        tir.Sub: (torch.sub, ["a", "b"]),
        tir.Mul: (torch.mul, ["a", "b"]),
        tir.Div: (safe_div, ["a", "b"]),
        tir.Min: (torch.minimum, ["a", "b"]),
        tir.Max: (torch.maximum, ["a", "b"]),
        tir.Cast: (lambda x: x, ["value"]),
    }
    TIR_FN_TORCH_FN = {
        "tir.pow": torch.pow,
        "tir.exp": torch.exp,
        "tir.log": safe_log,
        "tir.logk": lambda base, x: safe_log(x) / torch.log(base),
        "tir.sigmoid": lambda x: x * torch.pow(x**2 + 1, -0.5) / 2 + 0.5,
        "tir.hump": lambda x: torch.pow(x**2 + 1, -0.5),
    }

    def __init__(self, exprs: np.ndarray, var2idx: dict, var2expr: dict) -> None:
        self.exprs = exprs.ravel().tolist()
        self.shape = exprs.shape
        self.var2idx = var2idx
        self.var2expr = var2expr
        self.memo: Dict[str, Tensor] = {}

    def __call__(self, inputs: Tensor):
        self.memo.clear()
        output = self.run(inputs)
        self.memo.clear()
        if torch.any(torch.isnan(output)):
            raise RuntimeError("NaN detected in the output")
        return output

    def get_traced(self) -> GraphModule:
        self.memo.clear()
        # `self` is coded in the graph.
        # Use a lambda to convince symbolic_trace that it's a function, not a member
        ret = symbolic_trace(lambda input: self.run(input))
        self.memo.clear()
        return ret

    def _run_expr_memoized(self, expr: tir.PrimExpr, inputs: Tensor, batch_size: int):
        estr = repr(expr)
        if (result := self.memo.get(estr)) is not None:
            return result
        self.memo[estr] = result = self._run_expr(expr, inputs, batch_size)
        return result

    def _run_expr(self, expr: tir.PrimExpr, inputs: Tensor, nb: int) -> Tensor:
        if isinstance(expr, tir.Var):
            var_idx = self.var2idx.get(expr.name)
            if var_idx is not None:
                return inputs[:, var_idx]
            var_expr = self.var2expr.get(expr.name)
            if var_expr is not None:
                return self._run_expr_memoized(var_expr, inputs, nb)
            raise KeyError(f"Stray variable {expr.name}")
        if isinstance(expr, tir.IntImm):
            dtype = torch.bool if expr.dtype == "bool" else torch.int64
            return inputs.new_full((nb,), expr.value, dtype=dtype)
        if isinstance(expr, tir.FloatImm):
            return inputs.new_full((nb,), expr.value, dtype=torch.float32)
        if isinstance(expr, tir.Call):
            if (func := self.TIR_FN_TORCH_FN.get(expr.op.name)) is None:
                raise ValueError(f"Function call {expr.op.name} is unsupported")
            args = expr.args
        else:
            if (func_fields := self.TIR_OP_TORCH_FN.get(type(expr))) is None:
                raise ValueError(f"Operator {type(expr)} is unsupported")
            func, fields = func_fields
            args = [getattr(expr, f) for f in fields]
        args = [self._run_expr_memoized(arg, inputs, nb) for arg in args]
        return func(*args)

    def run(self, inputs: Tensor):
        batch_size = inputs.shape[0]
        if not self.exprs:
            return inputs.new_zeros(batch_size, *self.shape)
        # inputs: [batch_size, n_vars]
        ret = []
        for expr in self.exprs:
            result = self._run_expr_memoized(expr, inputs, batch_size)
            if result.dtype is torch.float and (result.abs() > 1e4).any():
                raise ValueError(f"Result too large: {result} in {expr}")
            ret.append(result)
        ret_ = torch.stack(ret, dim=1)
        # return [batch_size, *shape] (shape = [n_buffers, n_exprs])
        return ret_.view(-1, *self.shape)
