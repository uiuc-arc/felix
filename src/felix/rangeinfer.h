/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include <algorithm>

namespace tvm {
namespace felix {

using namespace tir;

inline FloatImm ToFloatImm(double e) { return FloatImm(DataType::Float(32), e); }

inline PrimExpr CastToFloat(const PrimExpr& value) {
  return value->dtype.is_float() ? value : cast(DataType::Float(32), value);
}

inline bool ToConstNumber(const PrimExpr& x, double& val) {
  if (!x.defined()) {
    return false;
  }
  if (const auto op = x.as<IntImmNode>()) {
    val = static_cast<float>(op->value);
    return true;
  } else if (const auto op = x.as<FloatImmNode>()) {
    val = op->value;
    return true;
  } else {
    return false;
  }
}

inline bool ToConstRange(const Range& range, double& min, double& max) {
  double extent;
  if (!ToConstNumber(range->min, min) || !ToConstNumber(range->extent, extent)) {
    return false;
  }
  max = min + extent;
  return true;
}

inline bool ToConstNumber(const Range& range, double& x) {
  double extent;
  return ToConstNumber(range->min, x) && ToConstNumber(range->extent, extent) && extent == 0;
}

template <typename Ret>
class MemoizedExprFunctor : public ExprFunctor<Ret(const PrimExpr&)> {
 public:
  Ret VisitExpr(const PrimExpr& expr) override {
    auto it = this->memo.find(expr);
    if (it != this->memo.end()) {
      return it->second;
    }
    return this->memo[expr] = ExprFunctor<Ret(const PrimExpr&)>::VisitExpr(expr);
  }

 protected:
  std::unordered_map<PrimExpr, Ret, StructuralHash, StructuralEqual> memo;
};

template <>
class MemoizedExprFunctor<PrimExpr> : public ExprMutator {
 public:
  PrimExpr VisitExpr(const PrimExpr& expr) override {
    auto it = this->memo.find(expr);
    if (it != this->memo.end()) {
      return it->second;
    }
    return this->memo[expr] = ExprMutator::VisitExpr(expr);
  }

 protected:
  std::unordered_map<PrimExpr, PrimExpr, StructuralHash, StructuralEqual> memo;
};

class RangeInfer : public MemoizedExprFunctor<Range> {
 public:
  RangeInfer(Range range_for_sizevar = Range()) : range_for_sizevar(range_for_sizevar) {}

  void BindLoop(const ForNode* loop, bool allow_override) {
    // To tighten the bound, we adopt an [a, b] convension for the range instead of [a, b).
    Bind(loop->loop_var, Range::FromMinExtent(loop->min, loop->extent - 1), allow_override);
  }

  void Bind(Var var, Range range, bool allow_override) {
    ICHECK(arith::ExprIsConstant(range->min) && arith::ExprIsConstant(range->extent))
        << "Cannot bind non-constant range for variable " << var << " = " << range;
    if (this->var_bind.count(var->name_hint)) {
      if (allow_override) {
        this->memo.clear();
      } else {
        LOG_FATAL << "Cannot override range for " << var->name_hint << " (already defined)";
      }
    }
    this->var_bind[var->name_hint] = range;
  }

  PrimExpr GetMax(PrimExpr e) {
    auto range = VisitExpr(e);
    return range->min + range->extent;
  }

 private:
#define VisitBinOpSameMono(Type, Func)                                     \
  Range VisitExpr_(const Type##Node* op) final {                           \
    Range lhs = VisitExpr(op->a), rhs = VisitExpr(op->b);                  \
    PrimExpr lmax = lhs->min + lhs->extent, rmax = rhs->min + rhs->extent; \
    PrimExpr begin = Func(lhs->min, rhs->min), end = Func(lmax, rmax);     \
    return Range(begin, end);                                              \
  }

#define VisitBinOpRevMono(Type, Func)                                      \
  Range VisitExpr_(const Type##Node* op) final {                           \
    Range lhs = VisitExpr(op->a), rhs = VisitExpr(op->b);                  \
    PrimExpr lmax = lhs->min + lhs->extent, rmax = rhs->min + rhs->extent; \
    PrimExpr begin = Func(lhs->min, rmax), end = Func(lmax, rhs->min);     \
    return Range(begin, end);                                              \
  }

#define VisitMods(Type)                                              \
  Range VisitExpr_(const Type##Node* op) final {                     \
    Range lhs = VisitExpr(op->a), rhs = VisitExpr(op->b);            \
    PrimExpr zero = Integer(0);                                      \
    if (is_const_int(lhs->min, 0) && is_const_int(lhs->extent, 0)) { \
      return Range(zero, zero);                                      \
    }                                                                \
    return Range(zero, rhs->min + rhs->extent - 1);                  \
  }

  VisitBinOpSameMono(Add, add);
  VisitBinOpSameMono(Mul, mul);
  VisitBinOpRevMono(Sub, sub);
  VisitBinOpRevMono(Div, div);
  VisitBinOpRevMono(FloorDiv, floordiv);
  VisitBinOpSameMono(Min, min);
  VisitBinOpSameMono(Max, max);
  VisitMods(Mod);
  VisitMods(FloorMod);

  Range VisitExpr_(const IntImmNode* op) final {
    return Range::FromMinExtent(GetRef<IntImm>(op), Integer(0));
  }
  Range VisitExpr_(const FloatImmNode* op) final {
    return Range::FromMinExtent(GetRef<FloatImm>(op), ToFloatImm(0.0));
  }
  // SizeVar that we insert as schedule vars are seen as constants
  // as they eventually will be constants for each given configuration.
  Range VisitExpr_(const SizeVarNode* op) final {
    if (this->range_for_sizevar.defined()) {
      return this->range_for_sizevar;
    }
    return Range::FromMinExtent(GetRef<SizeVar>(op), 0);
  }
  Range VisitExpr_(const VarNode* op) final {
    auto it = this->var_bind.find(op->name_hint);
    if (it != this->var_bind.end()) {
      return it->second;
    } else {
      LOG_FATAL << "Cannot find var \'" << op->name_hint << "\' in range inference";
      return Range();
    }
  }

  Range VisitExpr_(const SelectNode* op) final {
    // Don't have much use for range of condition.
    Range lhs = VisitExpr(op->true_value), rhs = VisitExpr(op->false_value);
    PrimExpr lmax = lhs->min + lhs->extent, rmax = rhs->min + rhs->extent;
    return Range(min(lhs->min, rhs->min), max(lmax, rmax));
  }

  Range VisitExpr_(const CastNode* op) final { return VisitExpr(op->value); }

  Range VisitExpr_(const CallNode* op) final {
    auto* pop = op->op.as<OpNode>();
    ICHECK(pop != nullptr);
    if (pop->name == "tir.exp") {
      Range r = VisitExpr(op->args[0]);
      return Range(exp(r->min), exp(r->min + r->extent));
    } else if (pop->name == "tir.log") {
      Range r = VisitExpr(op->args[0]);
      return Range(log(r->min), log(r->min + r->extent));
    } else if (pop->name == "tir.logk") {
      double base_ = 0;
      if (!ToConstNumber(VisitExpr(op->args[0]), base_)) {
        LOG_FATAL << "logk only supports constant base";
      }
      Range x = VisitExpr(op->args[1]);
      double x_min = 0, x_max = 0;
      if (ToConstRange(x, x_min, x_max)) {
        return Range(ToFloatImm(std::log(x_min) / std::log(base_)),
                     ToFloatImm(std::log(x_max) / std::log(base_)));
      } else {
        return Range(div(log(x->min), std::log(base_)),
                     div(log(x->min + x->extent), std::log(base_)));
      }
    } else if (pop->name == "tir.pow") {
      Range r1 = VisitExpr(op->args[0]), r2 = VisitExpr(op->args[1]);
      double r1_min = 0, r1_max = 0, r2_min = 0, r2_max = 0;
      bool is_r1_const = ToConstRange(r1, r1_min, r1_max),
           is_r2_const = ToConstRange(r2, r2_min, r2_max);
      if (is_r1_const && is_r2_const) {
        return Range(ToFloatImm(std::pow(r1_min, r2_min)), ToFloatImm(std::pow(r1_max, r2_max)));
      } else if (is_r1_const && r1_min == r1_max) {
        FloatImm r1f = ToFloatImm(r1_min);
        if (r1_min <= 0) {
          LOG_FATAL << "only pow with base >= 1 is supported; got base " << r1 << " in "
                    << GetRef<PrimExpr>(op);
        } else if (r1_min < 1) {
          return Range(pow(r1f, r2->min + r2->extent), pow(r1f, r2->min));
        } else {
          return Range(pow(r1f, r2->min), pow(r1f, r2->min + r2->extent));
        }
      } else if (is_r2_const && r2_min == r2_max) {
        FloatImm r2f = ToFloatImm(r2_min);
        if (r2_min < 0) {
          return Range(pow(r1->min + r1->extent, r2f), pow(r1->min, r2f));
        } else {
          return Range(pow(r1->min, r2f), pow(r1->min + r1->extent, r2f));
        }
      } else {
        LOG_FATAL << "pow with non-constant base and exponent is unsupported: "
                  << GetRef<PrimExpr>(op);
      }
    } else {
      LOG_FATAL << "Call to " << pop->name << " not supported";
    }
    return Range();  // unreachable
  }

  Range VisitExprDefault_(const Object* op) final {
    LOG_FATAL << "Expression of type " << op->GetTypeKey() << " is unsupported in RangeInfer";
    return Range();
  }

 public:
  std::unordered_map<std::string, Range> var_bind;
  Range range_for_sizevar;
};

}  // namespace felix
}  // namespace tvm
