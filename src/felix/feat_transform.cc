#include <tvm/arith/egg_simpl.h>
#include <tvm/arith/var_context.h>
#include <tvm/auto_scheduler/feature.h>
#include <tvm/support/parallel_for.h>

#include <optional>
#include <variant>

#include "features.h"
#include "utils.h"

namespace tvm {
namespace felix {
using namespace tvm::arith;

auto Factorize(uint64_t n) {
  std::unordered_map<uint64_t, uint64_t> factors;
  for (uint64_t i = 2; i <= n; ++i) {
    while (n % i == 0) {
      factors[i] += 1;
      n /= i;
    }
  }
  std::unordered_map<uint64_t, PrimExpr> ret;
  for (auto& [factor, count] : factors) {
    ret[factor] = Integer(count);
  }
  return ret;
}

template <typename T>
void CollectSameOps(const T* e, std::vector<PrimExpr>& ret) {
  auto *lhs = e->a.template as<T>(), *rhs = e->b.template as<T>();
  if (lhs) {
    CollectSameOps(lhs, ret);
  } else {
    ret.push_back(e->a);
  }
  if (rhs) {
    CollectSameOps(rhs, ret);
  } else {
    ret.push_back(e->b);
  }
}

template <typename T>
std::vector<PrimExpr> CollectSameOps(const PrimExpr& e) {
  std::vector<PrimExpr> ret;
  auto* node = e.as<T>();
  if (node) {
    CollectSameOps(node, ret);
  } else {
    ret.push_back(e);
  }
  return ret;
}

class LinearExprNode : public Object {
 public:
  Map<SizeVar, FloatImm> lin_terms;
  FloatImm constant;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("lin_terms", &lin_terms);
    v->Visit("constant", &constant);
  }

  static constexpr const char* _type_key = "felix.LinearExpr";
  TVM_DECLARE_FINAL_OBJECT_INFO(LinearExprNode, Object);
};

TVM_REGISTER_NODE_TYPE(LinearExprNode);

class LinearExpr : public ObjectRef {
 public:
  explicit LinearExpr(double constant) {
    auto node = make_object<LinearExprNode>();
    node->constant = ToFloatImm(constant);
    this->data_ = std::move(node);
  }

  explicit LinearExpr(SizeVar var) {
    auto node = make_object<LinearExprNode>();
    node->constant = ToFloatImm(0.0f);
    node->lin_terms.Set(var, ToFloatImm(1.0f));
    this->data_ = std::move(node);
  }

  LinearExpr(double constant, const std::unordered_map<std::string, double>& name2coef) {
    auto node = make_object<LinearExprNode>();
    node->constant = ToFloatImm(constant);
    for (auto& [name, coef] : name2coef) {
      node->lin_terms.Set(SizeVar(name, SizeVarKind::kScheduleKnob), ToFloatImm(coef));
    }
    this->data_ = std::move(node);
  }

  PrimExpr ToPrimExpr() const {
    auto node = this->operator->();
    PrimExpr ret = node->constant;
    for (auto& [var, coef] : node->lin_terms) {
      ret = ret + var * coef;
    }
    return ret;
  }

  LinearExpr& operator+=(const LinearExpr& other) {
    ICHECK(this->defined() && other.defined());
    auto this_ = this->CopyOnWrite();
    this_->constant = ToFloatImm(this_->constant->value + other->constant->value);
    for (auto& [var, coef1] : other->lin_terms) {
      auto coef2 = this_->lin_terms.Get(var).value_or(ToFloatImm(0.0f));
      this_->lin_terms.Set(var, ToFloatImm(coef1->value + coef2->value));
    }
    return *this;
  }
  LinearExpr& operator*=(double other) {
    ICHECK(this->defined());
    auto this_ = this->CopyOnWrite();
    this_->constant = ToFloatImm(this_->constant->value * other);
    for (auto& [var, coef] : this_->lin_terms) {
      this_->lin_terms.Set(var, ToFloatImm(coef->value * other));
    }
    return *this;
  }
  LinearExpr& operator-=(const LinearExpr& other) { return *this += LinearExpr(-1.0f) * other; }
  LinearExpr& operator/=(double other) { return *this *= (1.0f / other); }

#define DEF_BINARY_OP(def, with, other_t, check)        \
  LinearExpr operator def(const other_t& other) const { \
    if (!this->defined() || check) {                    \
      return LinearExpr();                              \
    }                                                   \
    LinearExpr ret = *this;                             \
    ret with other;                                     \
    return ret;                                         \
  }

  DEF_BINARY_OP(+, +=, LinearExpr, !other.defined())
  DEF_BINARY_OP(-, -=, LinearExpr, !other.defined())
  DEF_BINARY_OP(*, *=, double, false)
  DEF_BINARY_OP(/, /=, double, false)

  LinearExpr operator*(LinearExpr other) {
    if (!this->defined() || !other.defined()) {
      return LinearExpr();
    } else if ((*this)->lin_terms.empty()) {
      return other * (*this)->constant->value;
    } else if (other->lin_terms.empty()) {
      return (*this) * other->constant->value;
    } else {
      return LinearExpr();
    }
  }
  LinearExpr operator/(LinearExpr other) {
    if (!this->defined() || !other.defined()) {
      return LinearExpr();
    } else if (other->lin_terms.empty()) {
      return (*this) / other->constant->value;
    } else {
      return LinearExpr();
    }
  }

  TVM_DEFINE_OBJECT_REF_METHODS(LinearExpr, ObjectRef, LinearExprNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(LinearExprNode);
};

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LinearExprNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* expr = static_cast<const LinearExprNode*>(node.get());
      auto& os = p->stream;
      os << "(" << expr->constant << "; ";
      bool first = true;
      for (auto& [var, coef] : expr->lin_terms) {
        if (!first) {
          os << ", ";
        }
        os << "(" << coef << " * " << var << ")";
        first = false;
      }
      os << ")";
    });

class LinExprExtractor : public ExprFunctor<LinearExpr(const PrimExpr&)> {
  LinearExpr VisitExpr_(const SizeVarNode* e) override { return LinearExpr(GetRef<SizeVar>(e)); }
  LinearExpr VisitExpr_(const IntImmNode* e) override { return LinearExpr((double)e->value); }
  LinearExpr VisitExpr_(const FloatImmNode* e) override { return LinearExpr(e->value); }
  LinearExpr VisitExpr_(const VarNode* e) override {
    LOG_FATAL << "Do not use LinExprExtractor on expressions with non-sizevar variable; got "
              << GetRef<Var>(e);
    return LinearExpr();
  }

  LinearExpr VisitExpr_(const AddNode* e) override { return VisitExpr(e->a) + VisitExpr(e->b); }
  LinearExpr VisitExpr_(const SubNode* e) override { return VisitExpr(e->a) - VisitExpr(e->b); }
  LinearExpr VisitExpr_(const MulNode* e) override { return VisitExpr(e->a) * VisitExpr(e->b); }
  LinearExpr VisitExpr_(const DivNode* e) override { return VisitExpr(e->a) / VisitExpr(e->b); }
  LinearExpr VisitExpr_(const CastNode* e) override { return VisitExpr(e->value); }

  LinearExpr VisitExprDefault_(const Object* e) override { return LinearExpr(); }
};

class FloorRemover : public ExprMutator {
  PrimExpr VisitExpr_(const FloorDivNode* e) final {
    // Simply drop the floor() operator; it's not differentiable.
    return div(VisitExpr(e->a), VisitExpr(e->b));
  }
  PrimExpr VisitExpr_(const FloorModNode* e) final {
    LOG_WARNING << "Mod operator is not differentiable and will be dropped.";
    return Integer(0);
  }
};

class DiffableApprox : public MemoizedExprFunctor<PrimExpr> {
 public:
  DiffableApprox(const DiffableApprox& other) = delete;
  DiffableApprox() = delete;

  // * Important to use Float values (Range(1.0, 5.0)) for range inf default range.
  DiffableApprox(const std::unordered_set<std::string>& new_knobs, const VarMapT& exp_subst,
                 const VarMapT& shorthands, const VarMapT& quotients)
      : rinf(Range(ToFloatImm(1.0), ToFloatImm(5.0))),
        new_knobs(new_knobs),
        exp_subst(exp_subst),
        shorthands(shorthands),
        quotients(quotients) {}

  // Simplification strategies:
  // 1. `K < sp_i_j_b + sp_i_j_b + ...` (K is actual constant)
  //    - We're good; just return sigmoid(RHS - LHS).
  // 2. `1 < sp_i_j`
  //    - Replace sp_i_j with its exp decomposition, simplify, should give us Form 1.
  // 3. Anything expr with shorthand variables
  //    - Substitute with shorthand vars and simplify; if changed, revisit.
  // 4. Now safe to use special simplification (safe on expressions that only contain `sp_i_j` and
  //    quotient vars `d{i}`, and shape constants)
  //    - This helps reduce forms such as `1 < sp_i_j * sp_i'_j' * ...` to smaller forms.
  //    - If expr contains quotients d{i}, try twice: expanding d{i} = E{i} / sp_i_j / sp_i_j' ... ,
  //      see which one is shorter.
  // 5. If everything fall through, try taking log on both sides (with range inference to help with
  // safety).
  //
  // Step 3-5 also applies to VisitExpr_(EQ).

  PrimExpr VisitExpr_(const LTNode* e) final {
    auto expr = GetRef<PrimExpr>(e);
    LinearExpr diff = IsTermLinear(e->b - e->a);
    if (diff.defined()) {
      return MakeDiffableCond(diff.ToPrimExpr(), /* sigmoid_or_hump */ true);
    }
    PrimExpr response;
    if ((response = ConstVsSingleSchedVar(e->a, e->b)).defined()) {
      response = SimplifyExpr(LT(e->a, response), 20, 1000, true);
      return VisitExpr(response);
    }
    if ((response = SubstShorthandsAway(expr)).defined()) {
      return VisitExpr(response);
    }
    if ((response = SpecialSimplAndVisit(expr)).defined()) {
      return response;
    }
    if ((response = TakeLogDiffIfSafe(e->b, e->a)).defined()) {
      return MakeDiffableCond(response, /* sigmoid_or_hump */ true);
    }
    LOG_FATAL << "Cannot simplify " << e->a << " < " << e->b;
    return PrimExpr();
  }

  PrimExpr VisitExpr_(const EQNode* e) final {
    auto expr = GetRef<PrimExpr>(e);
    PrimExpr response;
    if ((response = SubstShorthandsAway(expr)).defined()) {  // Step 1
      return VisitExpr(response);
    }
    if ((response = SpecialSimplAndVisit(expr)).defined()) {
      return response;
    }
    if ((response = TakeLogDiffIfSafe(e->a, e->b)).defined()) {
      return MakeDiffableCond(response, /* sigmoid_or_hump */ false);
    }
    LOG_ERROR << "Cannot simplify " << e->a << " == " << e->b;
    return PrimExpr();
  }

  PrimExpr VisitExpr_(const IntImmNode* e) final {
    // Converts true into 1 and false into 0.
    return Integer(e->value);
  }

  PrimExpr VisitExpr_(const SelectNode* e) final {
    if (ExprIsShapeConstant(e->condition)) {
      return select(SimplifyExpr(e->condition, 20, 1000), VisitExpr(e->true_value),
                    VisitExpr(e->false_value));
    }
    auto cond = VisitExpr(e->condition), tv = VisitExpr(e->true_value),
         fv = VisitExpr(e->false_value);
    return cond * (tv - fv) + fv;
  }

  // 1 - x below stands for `not x`
  PrimExpr VisitExpr_(const LENode* e) final { return 1 - this->VisitExpr(LT(e->b, e->a)); }
  PrimExpr VisitExpr_(const GENode* e) final { return 1 - this->VisitExpr(LT(e->a, e->b)); }
  PrimExpr VisitExpr_(const GTNode* e) final { return this->VisitExpr(LT(e->b, e->a)); }
  PrimExpr VisitExpr_(const NENode* e) final { return 1 - this->VisitExpr(EQ(e->a, e->b)); }

  PrimExpr VisitExpr_(const AndNode* e) final { return VisitExpr(e->a) * VisitExpr(e->b); }
  PrimExpr VisitExpr_(const OrNode* e) final {
    std::vector<PrimExpr> chained_ors;
    CollectSameOps(e, chained_ors);
    // Using de Morgan's law:
    PrimExpr ret = Integer(1);
    for (auto& expr : chained_ors) {
      ret *= 1 - VisitExpr(expr);
    }
    return 1 - ret;
  }
  PrimExpr VisitExpr_(const NotNode* e) final { return 1 - VisitExpr(e->a); }

  LinearExpr IsTermLinear(const PrimExpr& diff) {
    auto lin_diff = LinExprExtractor()(diff);
    if (!lin_diff.defined()) {
      return LinearExpr();
    }
    bool all_vars_allowed = true;
    for (auto& [var, _] : lin_diff->lin_terms) {
      if (!this->new_knobs.count(var->name_hint)) {
        all_vars_allowed = false;
      }
    }
    return all_vars_allowed ? lin_diff : LinearExpr();
  }

 private:
  PrimExpr ConstVsSingleSchedVar(const PrimExpr& lhs, const PrimExpr& rhs) {
    auto* rhs_v = rhs.as<SizeVarNode>();
    if (lhs->IsInstance<IntImmNode>() && rhs_v) {
      auto it = this->exp_subst.find(rhs_v->name_hint);
      if (it != this->exp_subst.end()) {
        return it->second;
      }
    }
    return PrimExpr();
  }

  PrimExpr SubstShorthandsAway(const PrimExpr& expr) {
    t4.Start();
    bool changed = false;
    auto ret = SubstByName(expr, this->shorthands, &changed);
    ret = changed ? SimplifyExpr(ret) : PrimExpr();
    t4.Stop();
    return ret;
  }

  PrimExpr SpecialSimplAndVisit(const PrimExpr& expr) {
    if (!this->progressed) {
      t5.Start();
      auto direct_simpl = SpecialSimplIfSafe(expr);
      t5.Stop();
      if (direct_simpl.defined()) {
        this->progressed = true;
        auto ret = VisitExpr(direct_simpl);
        this->progressed = false;
        return ret;
      }
    }
    return PrimExpr();
  }

  PrimExpr SpecialSimplIfSafe(const PrimExpr& expr) {
    bool safe_vars_only = true, has_quotients = false;
    PostOrderVisit(expr, [this, &safe_vars_only, &has_quotients](const ObjectRef& obj) {
      if (auto* var = obj.as<VarNode>()) {
        if (!obj->IsInstance<SizeVarNode>() || this->shorthands.count(var->name_hint)) {
          safe_vars_only = false;
        } else if (this->quotients.count(var->name_hint)) {
          has_quotients = true;
        }
      }
    });
    if (!safe_vars_only) {
      return PrimExpr();
    }
    auto simpl = SimplifyExpr(expr, 30, 10000, true);
    if (has_quotients) {
      auto subbed = SubAndSimplify(expr, this->quotients);
      return CountOps(subbed) < CountOps(expr) ? subbed : simpl;
    } else {
      return simpl;
    }
  }

  PrimExpr TakeLogDiffIfSafe(PrimExpr a, PrimExpr b) {
    double amin, amax, bmin, bmax;
    ICHECK(ToConstRange(this->rinf(a), amin, amax));
    ICHECK(ToConstRange(this->rinf(b), bmin, bmax));
    if (amin <= 0 || bmin <= 0) {
      LOG_WARNING << "Cannot take diff of " << PrintExprPrefix(a) << " (" << amin << ") and "
                  << PrintExprPrefix(b) << " (" << bmin << ") to a range stable form";
    }
    auto ret = SimplifyExpr(log(a) - log(b), 20, 5000, true);
    return ret;
  }

  PrimExpr MakeDiffableCond(const PrimExpr& e, bool sigmoid_or_hump) {
    auto* e_int = e.as<IntImmNode>();
    if (e_int) {
      if (sigmoid_or_hump) {
        return e_int->value > 0 ? 1 : 0;
      } else {
        return e_int->value == 0 ? 1 : 0;
      }
    }
    String var_name = "da_" + std::to_string(this->vdefs_out.size());
    this->vdefs_out[var_name] = sigmoid_or_hump ? sigmoid(e) : hump(e);
    return SizeVar(var_name, SizeVarKind::kShorthand);
  }

 public:
  VarMapT vdefs_out;

 private:
  Timer t4{"SubstShorthandsAway"}, t5{"SpecialSimplAndVisit"};
  RangeInfer rinf;
  std::unordered_set<std::string> new_knobs;
  const VarMapT &exp_subst, &shorthands, &quotients;
  bool progressed{};
};

class FeaturePackPyNode : public Object {
 public:
  Array<Array<ObjectRef>> expressions;
  Array<String> free_vars;
  Array<LinearExpr> linear_cons;
  Map<String, Map<Integer, SizeVar>> var_factors;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("expressions", &expressions);
    v->Visit("free_vars", &free_vars);
    v->Visit("linear_cons", &linear_cons);
    v->Visit("var_factors", &var_factors);
  }

  static constexpr const char* _type_key = "felix.FeaturePackPy";
  TVM_DECLARE_FINAL_OBJECT_INFO(FeaturePackPyNode, Object);
};

TVM_REGISTER_NODE_TYPE(FeaturePackPyNode);

class FeaturePackPy : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(FeaturePackPy, ObjectRef, FeaturePackPyNode);
};

class FeaturePack {
 public:
  static constexpr double MANUAL_UPPER_BOUND = 10.0;

  FeaturePack() = default;

  FeaturePack(VarDefStack vdefs, VarDefStack features, VarDefStack constraints,
              Array<SplitGroup> sp_groups)
      : variables({{"vdefs", std::move(vdefs)},
                   {"features", std::move(features)},
                   {"constraints", std::move(constraints)}}),
        split_groups(std::move(sp_groups)) {}

  void RunRollingSimplify() {
    // For every expression `e` in vdefs,
    // for each variable `v` in `e` that's defined in vdefs as `v = e_v`,
    // if substituting `v = e_v` into `e` (and simplifying) makes `e` smaller, do it.
    auto& vdefs = this->variables.at("vdefs");
    vdefs->MapExprs([&vdefs](const PrimExpr& expr) {
      std::unordered_set<const VarNode*> vars;
      PostOrderVisit(expr, [&vdefs, &vars](const ObjectRef& node) {
        auto* var = node.as<SizeVarNode>();
        if (var && vdefs->Contains(var->name_hint)) {
          vars.insert(node.as<VarNode>());
        }
      });
      auto expr_ = expr;
      for (auto& var : vars) {
        auto& to_sub = vdefs->GetExprAt(var->name_hint);
        auto subbed = arith::SubAndSimplify(expr, {{var->name_hint, to_sub}});
        if (CountOps(subbed) < CountOps(expr)) {
          expr_ = subbed;
        }
      }
      return expr_;
    });
    // Concat `features` to the end of `vdefs` to form `vi_and_features`,
    // and split out E{i} variables from vdefs into `ei_vars`, the rest into `vi_vars`.
    VarDefStack &vi_and_features = this->variables["vi_and_features"],
                &vi_vars = this->variables["vi_vars"], &ei_vars = this->variables["ei_vars"];
    for (auto& pair : this->variables.at("vdefs")->GetExprs()) {
      if (pair->var->kind == SizeVarKind::kShapeVar) {
        ei_vars->Append(pair->var, pair->expr);
      } else {
        vi_vars->Append(pair->var, pair->expr);
        vi_and_features->Append(pair->var, pair->expr);
      }
    }
    for (auto& pair : this->variables.at("features")->GetExprs()) {
      vi_and_features->Append(pair->var, pair->expr);
    }
    this->variables.erase("vdefs");
    this->variables.erase("features");
  }

  void RunExpDecomposeNoFactoring() {
    // If no prime factoring is requested, still do all of the work but with prime_bases = {2},
    // because we do need all the variables to be at least defined.
    this->no_factoring = true;
    this->RunExpDecompose({2});
  }

  void RunExpDecompose(const std::vector<size_t>& prime_bases) {
    // List all schedule vars (such as sp_i_j), and create exp decomposition for them,
    // inserting into this->var_factors and `exp_decomp`.
    VarDefStack &exp_decomp = this->variables["exp_decomp"],
                &vi_and_features = this->variables.at("vi_and_features");
    auto all_var_names = vi_and_features->GetAllUsedVars(SizeVarKind::kScheduleKnob);
    // 1. For variables in `SplitGroup`s, we create a new variable for each prime base p.
    auto DecomposeOneVar = [this, &prime_bases](const std::string& vname, SizeVarKind kind) {
      PrimExpr subst_expr = Integer(1);
      for (size_t prime : prime_bases) {
        SizeVar var(vname + "_" + std::to_string(prime), kind);
        this->var_factors[vname][prime] = var;
        // Added variable should be greater than 0 (i.e., -sv <= 0).
        this->linear_cons.push_back(LinearExpr(var) * (-1));
        subst_expr *= pow(Integer(prime), var);
      }
      return subst_expr;
    };
    for (auto& group : this->split_groups) {
      auto extent_vname = group->extent->name_hint;
      // Create sp_i_j_2, sp_i_j_3, ... for each sp_i_j and prime base p.
      for (auto& vname : group->vars) {
        exp_decomp->Append(vname, DecomposeOneVar(vname, SizeVarKind::kScheduleKnob));
        all_var_names.erase(vname);
      }
      // Create Ei_2, Ei_3, ... for the group's extent Ei and each prime base p.
      exp_decomp->Append(extent_vname, DecomposeOneVar(extent_vname, SizeVarKind::kShapeVar));
      // Create representation of the quotient Ei / (sp_i_0 * sp_i_1 * ...)
      // as a sum in the power: 2**(Ei_2 - sp_i_0_2 - sp_i_1_2 - ...) * 3**(...) * ...
      // Also insert the constraints that Ei_b >= sp_i_0_b + sp_i_1_b + ...
      // (essentially meaning that each power consisting q{i} is non-negative)
      auto quotient = Integer(1);
      for (size_t prime : prime_bases) {
        LinearExpr q_total_power(this->var_factors[extent_vname][prime]);
        for (auto& vname : group->vars) {
          q_total_power -= LinearExpr(this->var_factors[vname][prime]);
        }
        // NOTE: it's -q_total_power because we want something lower-better.
        this->linear_cons.push_back(q_total_power * (-1));
        quotient *= pow(Integer(prime), q_total_power.ToPrimExpr());
      }
      exp_decomp->Append(group->quotient, quotient);
    }
    // For all other variables, just do a log2, and don't put it in this->var_factors.
    for (auto& vname : all_var_names) {
      SizeVar sv(vname + "_2", SizeVarKind::kScheduleKnob);
      this->var_factors[vname][2] = sv;
      exp_decomp->Append(vname, pow(Integer(2), sv));
      // Added variable should be greater than 0 (i.e., -sv <= 0)
      this->linear_cons.push_back(LinearExpr(sv) * (-1));
      // Then also put an arbitrary upperbound to the variable (which would otherwise be unbounded):
      this->linear_cons.push_back(LinearExpr(sv) - LinearExpr(MANUAL_UPPER_BOUND));
    }
  }

  void RunDiffTransform() {
    // Remove floordiv and floormod from all variables except Ei variables.
    for (auto& [table_name, vdefs] : this->variables) {
      if (table_name != "ei_vars") {
        vdefs->MapExprs(FloorRemover());
      }
    }
    // Prepare 2 substitution maps for the differentiability transformation:
    // 1. exp_decomp: sp_i_j = 2**sp_i_j_2 * 3**sp_i_j_3 * ..., Ei = 2**Ei_2 * 3**Ei_3 * ..., qi =
    //    2**qi_2 * 3**qi_3 * ...
    //    - Also get a list of all variables on the power (sp_i_j_2, Ei_2, qi_2, ...) from
    //    exp_decomp.
    // 2. shorthands: vi = ... (all variables in vi_vars)
    // 3. quotient: all the qi variables in non-exponential form:
    //    q0 = E0 / (sp_0_0 * sp_0_1 * ...), q1 = E1 / (sp_1_0 * sp_1_1 * ...), ...
    //    - This can help simplify expressions like (q0 * sp_0_0 * sp_0_1), without having to expand
    //    everything into exp form.
    auto& exp_decomp = this->variables.at("exp_decomp");
    auto exp_sched_vars = exp_decomp->GetAllUsedVars(std::nullopt);
    VarMapT exp_subst = exp_decomp->IntoUnwindedVarMap(),
            shorthands = this->variables.at("vi_vars")->IntoUnwindedVarMap();
    this->variables.erase("vi_vars");
    VarMapT quotient;
    for (auto& group : this->split_groups) {
      PrimExpr extent = group->extent;
      for (auto& vname : group->vars) {
        extent = div(extent, SizeVar(vname, SizeVarKind::kScheduleKnob));
      }
      quotient[group->quotient->name_hint] = extent;
    }
    auto& features = this->variables.at("vi_and_features");
    DiffableApprox da(exp_sched_vars, exp_subst, shorthands, quotient);
    {
      Timer timer("DA");
      features->MapExprs([&da](const PrimExpr& e) { return da(e); });
    }
    auto& diff_approx = this->variables["diff_approx"];
    for (auto& [vname, expr] : da.vdefs_out) {
      diff_approx->Append(vname, expr);
    }
    VarDefStack constraints;
    for (auto& pair : this->variables.at("constraints")->GetExprs()) {
      auto* expr_lt = pair->expr.as<LENode>();
      ICHECK(expr_lt);
      // NOTE: it's a - b because we want something lower-better.
      auto diff = log(expr_lt->a) - log(expr_lt->b);
      auto linexpr =
          LinExprExtractor()(SimplifyExpr(SubstByName(SubstByName(diff, shorthands), exp_subst)));
      if (linexpr.defined()) {
        this->linear_cons.push_back(linexpr);
      } else {
        constraints->Append(pair->var, diff);
      }
    }
    this->variables["constraints"] = constraints;
  }

  void RunSizeSubstitution(const Map<String, Integer>& size_subst) {
    VarMapT size_subst_(size_subst.begin(), size_subst.end());
    // 1. Set up size_subst_ with the values of all Ei and Ei_b variables.
    //    * size_subst_ values are all integers when this->no_factoring is false,
    //      and can be floats (results of log2(x)) otherwise.
    for (auto& pair : this->variables.at("ei_vars")->GetExprs()) {
      auto& vname = pair->var->name_hint;
      auto expr = SubAndSimplify(pair->expr, size_subst_);
      auto* expr_int = expr.as<IntImmNode>();
      ICHECK(expr_int) << "All shape variables must be substituted to integers; got " << pair->var
                       << " = " << expr;
      size_subst_[vname] = expr;
      std::unordered_map<uint64_t, PrimExpr> factors;
      if (this->no_factoring) {
        factors[2] = ToFloatImm(std::log2(expr_int->value));
      } else {
        factors = Factorize(expr_int->value);
      }
      this->InsertShapeVarFactors(vname, factors, size_subst_);
      // Drop shape vars from var_factors to not send them to Python side
      // (if TorchFeatures.inv_transform_config see Ei variables it will fail to find values for
      // them).
      this->var_factors.erase(vname);
    }
    // 2. Some (many) knobs can become trivial during size concretization; add all that is 0 to
    // size_subst_ as well.
    if (!this->no_factoring) {
      this->ComputeSplitGroupsFactors(size_subst_);
    }
    // 3. Concatenate all variables into one VarDefStack and substitute all sizes.
    VarDefStack concated;
    for (auto& [table_name, vdefs] : this->variables) {
      if (table_name != "ei_vars") {
        for (auto& pair : vdefs->GetExprs()) {
          concated->Append(pair->var, pair->expr);
        }
      }
    }
    concated->MapExprsParallel([this, &size_subst_](const PrimExpr& expr) {
      auto ret = SubAndSimplify(expr, size_subst_, true);
      ICHECK(CheckNoShapeVars(ret)) << "Expression " << expr << " substituted to " << ret
                                    << " still contains shape variables.";
      return ret;
    });
    this->variables.clear();
    this->variables["vdefs"] = concated;
    // 4. Substitute and simplify all linear constraints.
    std::vector<LinearExpr> new_linear_cons = std::move(this->linear_cons);
    this->linear_cons.clear();
    for (auto& linexpr : new_linear_cons) {
      LinearExpr linexpr_ =
          LinExprExtractor()(SubAndSimplify(linexpr.ToPrimExpr(), size_subst_, true));
      ICHECK(linexpr.defined());
      if (linexpr_->lin_terms.empty()) {
        ICHECK(linexpr_->constant->value <= 0)
            << "Constraint " << linexpr << " simplified to positive value " << linexpr_
            << " (unfeasible).";
      } else {
        this->linear_cons.push_back(linexpr_);
      }
    }
  }

  FeaturePackPy IntoPythonFeaturePack() const {
    auto node = make_object<FeaturePackPyNode>();
    auto& vdefs = this->variables.at("vdefs");
    for (auto& pair : vdefs->GetExprs()) {
      node->expressions.push_back({pair->var->name_hint, pair->expr});
    }
    node->linear_cons = this->linear_cons;
    std::unordered_set<std::string> free_vars;
    for (auto& [k1, vs] : this->var_factors) {
      Map<Integer, SizeVar> m;
      for (auto& [k2, v] : vs) {
        if (v->kind == SizeVarKind::kScheduleKnob) {
          free_vars.insert(v->name_hint);
        }
        m.Set(k2, v);
      }
      node->var_factors.Set(k1, m);
    }
    for (auto& vname : free_vars) {
      node->free_vars.push_back(vname);
    }
    return FeaturePackPy(node);
  }

  static std::optional<FeaturePack> LoadFromJsonReader(dmlc::JSONReader& reader) {
    bool is_defined;
    reader.BeginArray();
    ICHECK(reader.NextArrayItem());
    reader.Read(&is_defined);
    if (!is_defined) {
      return std::nullopt;
    }
    FeaturePack fp;
    ICHECK(reader.NextArrayItem());
    reader.Read(&fp.variables);
    ICHECK(reader.NextArrayItem());
    reader.Read(&fp.linear_cons);
    ICHECK(reader.NextArrayItem());
    reader.Read(&fp.split_groups);
    ICHECK(reader.NextArrayItem());
    reader.Read(&fp.var_factors);
    ICHECK(reader.NextArrayItem());
    reader.Read(&fp.no_factoring);
    ICHECK(!reader.NextArrayItem());
    return fp;
  }

  static void SaveAsJson(const String& filepath, const std::optional<FeaturePack>& fp) {
    std::ofstream fout(filepath);
    dmlc::JSONWriter writer(&fout);
    writer.BeginArray(true);
    writer.WriteArrayItem((bool)fp);
    if (fp) {
      writer.WriteArrayItem(fp->variables);
      writer.WriteArrayItem(fp->linear_cons);
      writer.WriteArrayItem(fp->split_groups);
      writer.WriteArrayItem(fp->var_factors);
      writer.WriteArrayItem(fp->no_factoring);
    }
    writer.EndArray();
  }

  void InsertShapeVarFactors(const std::string& vname,
                             const std::unordered_map<uint64_t, PrimExpr>& factors,
                             VarMapT& size_subst) const {
    auto& shape_var_decomp = this->var_factors.at(vname);
    for (auto& [prime, power] : factors) {
      auto it = shape_var_decomp.find(prime);
      ICHECK(it != shape_var_decomp.end()) << "Shape variable " << vname << " contains factor "
                                           << prime << " that the features weren't factorized for.";
      size_subst[it->second->name_hint] = power;
    }
    for (auto& [k, v] : shape_var_decomp) {
      if (!size_subst.count(v->name_hint)) {
        size_subst[v->name_hint] = Integer(0);
      }
    }
  }

  void ComputeSplitGroupsFactors(VarMapT& size_subst) {
    for (auto& group : this->split_groups) {
      auto& ename = group->extent->name_hint;
      for (auto& vname : group->vars) {
        auto& inner_map = this->var_factors.at(vname);
        for (auto it = inner_map.begin(); it != inner_map.end();) {
          auto& [prime, decomp_var] = *it;
          auto e_prime_name = ename + "_" + std::to_string(prime);
          auto* extent_prime_power = size_subst.at(e_prime_name).as<IntImmNode>();
          ICHECK(extent_prime_power);
          // If for example E2_3 is 0 (meaning E2 is not divisible by 3),
          // we can set sp_2_i_3 to 0 for all i.
          if (extent_prime_power->value == 0) {
            size_subst[decomp_var->name_hint] = Integer(0);
            it = inner_map.erase(it);
          } else {
            ++it;
          }
        }
      }
    }
  }

  bool CheckNoShapeVars(const PrimExpr& expr) {
    bool no_shape_vars{true};
    PostOrderVisit(expr, [&no_shape_vars](const ObjectRef& node) {
      auto* op = node.as<SizeVarNode>();
      if (op && op->kind == SizeVarKind::kShapeVar) {
        no_shape_vars = false;
      }
    });
    return no_shape_vars;
  }

  std::unordered_map<std::string, VarDefStack> variables;
  std::vector<LinearExpr> linear_cons{};
  Array<SplitGroup> split_groups;
  // size_t is not (de)serializable, so we use uint64_t instead
  std::unordered_map<std::string, std::unordered_map<uint64_t, SizeVar>> var_factors{};
  bool no_factoring{false};
};

class StmtSimplifier : public StmtExprMutator {
 public:
  StmtSimplifier(RangeInfer& rinf, const VarDefStack& vdefs) : rinf(rinf) {
    for (auto& pair : vdefs->GetExprs()) {
      if (pair->var->kind == SizeVarKind::kShapeVar) {
        this->e_vars.emplace_back(pair->var, pair->expr);
      }
    }
  }

  Stmt VisitStmt_(const ForNode* node) final {
    PrimExpr extent = GetRangeMax(node->extent);
    this->rinf.Bind(node->loop_var, Range::FromMinExtent(0, extent), true);
    Stmt body = StmtExprMutator::VisitStmt(node->body);
    return For(node->loop_var, 0, extent, node->kind, body);
  }

  Stmt VisitStmt_(const AttrStmtNode* node) final {
    if (node->attr_key == tir::attr::thread_extent || node->attr_key == tir::attr::virtual_thread) {
      PrimExpr extent = GetRangeMax(node->value);
      const Var& var = node->node.as<IterVarNode>()->var;
      this->rinf.Bind(var, Range::FromMinExtent(0, extent), true);
      Stmt body = StmtExprMutator::VisitStmt(node->body);
      return AttrStmt(node->node, node->attr_key, extent, body);
    } else {
      return StmtExprMutator::VisitStmt_(node);
    }
  }

  // Remove tir.likely() for if-then-else
  // which doesn't do anything for feature extraction
  // and is hard to read.
  Stmt VisitStmt_(const IfThenElseNode* node) final {
    auto* call = node->condition.as<CallNode>();
    static auto op_likely = Op::Get("tir.likely");
    if (!call || !call->op.same_as(op_likely)) {
      return StmtExprMutator::VisitStmt_(node);
    }
    return StmtExprMutator::VisitStmt(node->then_case);
  }

  Stmt VisitStmt_(const AllocateNode* op) final {
    Array<PrimExpr> extents;
    for (const auto& x : op->extents) {
      extents.push_back(GetRangeMax(x));
    }
    return Allocate(op->buffer_var, op->dtype, extents, op->condition,
                    StmtExprMutator::VisitStmt(op->body), op->annotations);
  }

  Stmt VisitStmt_(const BufferRealizeNode* node) final {
    TryChangingBufferShape(node->buffer);
    Array<Range> bounds;
    for (auto& r : node->bounds) {
      bounds.push_back(Range::FromMinExtent(0, GetRangeMax(r->extent)));
    }
    return BufferRealize(node->buffer, bounds, node->condition,
                         StmtExprMutator::VisitStmt(node->body));
  }

  Stmt VisitStmt_(const BufferStoreNode* node) final {
    TryChangingBufferShape(node->buffer);
    Array<PrimExpr> indices;
    for (auto& index : node->indices) {
      indices.push_back(this->SimplifyExpr(index));
    }
    return BufferStore(node->buffer, StmtExprMutator::VisitExpr(node->value), indices);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* node) final {
    TryChangingBufferShape(node->buffer);
    Array<PrimExpr> indices;
    for (auto& index : node->indices) {
      indices.push_back(this->SimplifyExpr(index));
    }
    return BufferLoad(node->buffer, indices);
  }

 private:
  void TryChangingBufferShape(const Buffer& buf) {
    auto* buf_ = buf.get();
    if (this->touched_bufs.count(buf_)) {
      return;
    }
    Array<PrimExpr>& buf_shape = *const_cast<Array<PrimExpr>*>(&buf_->shape);
    for (size_t i = 0; i < buf_shape.size(); ++i) {
      // HACK: pattern-match buffer shape stored in Buffer and BufferRealize
      // against the expressions of the E{i} variables we defined in VarContext.
      // If we were able to define these variables earlier, we wouldn't need to do this.
      if (auto size_var = FindEquivalentMatch(buf_shape[i])) {
        buf_shape.Set(i, size_var.value());
      }
    }
    this->touched_bufs.insert(buf_);
  }

  std::optional<SizeVar> FindEquivalentMatch(const PrimExpr& expr) {
    for (auto& [v, e] : this->e_vars) {
      if (arith::IsExprEquivalent(e, expr)) {
        return v;
      }
    }
    return std::nullopt;
  }

  PrimExpr SimplifyExpr(PrimExpr expr) {
    auto it = this->_memo.find(expr);
    if (it != this->_memo.end()) {
      return it->second;
    }
    return this->_memo[expr] = arith::SimplifyExpr(expr);
  }

  PrimExpr GetRangeMax(const PrimExpr& expr) {
    return this->SimplifyExpr(this->rinf.GetMax(this->SimplifyExpr(expr)));
  }

  RangeInfer& rinf;
  std::unordered_set<const BufferNode*> touched_bufs;
  std::vector<std::pair<SizeVar, PrimExpr>> e_vars;
  std::unordered_map<PrimExpr, PrimExpr, StructuralHash, StructuralEqual> _memo;
};

std::optional<FeaturePack> GetFeaturePack(Stmt& stmt, VarContext& context,
                                          const auto_scheduler::HardwareParams& hw_params,
                                          bool factoring, size_t cache_line_size,
                                          size_t max_n_bufs) {
  auto st_vdefs = context->var_defs;
  FeaturePack fp;
  try {
    RangeInfer rinf;
    {
      Timer timer("StmtSimplifier");
      stmt = StmtSimplifier(rinf, st_vdefs)(stmt);
    }
    // Feature and constraint extraction
    Timer timer("FeatureExtraction");
    VarDefStack features =
        GetPerStoreFeatureExpr(stmt, *(st_vdefs.get()), rinf, cache_line_size, max_n_bufs);
    auto constraints_ = GetConstraints(stmt, hw_params);
    VarDefStack constraints;
    for (size_t i = 0; i < constraints_.size(); ++i) {
      auto name = "con_" + std::to_string(i);
      constraints->Append("con_" + std::to_string(i), constraints_[i]);
    }
    fp = FeaturePack(std::move(context->var_defs), std::move(features), std::move(constraints),
                     std::move(context->split_groups));
  } catch (const std::exception& e) {
    LOG_WARNING << "Feature extraction failed: " << e.what();
    return std::nullopt;
  }
  // Simplify features that just came out from feature extraction
  fp.RunRollingSimplify();
  if (factoring) {
    fp.RunExpDecompose({2, 3, 5, 7});
  } else {
    fp.RunExpDecomposeNoFactoring();
  }
  fp.RunDiffTransform();
  return fp;
}

TVM_REGISTER_GLOBAL("felix.GetFeaturePack")
    .set_body_typed([](Stmt stmt, VarContext context, auto_scheduler::HardwareParams hw_params,
                       Map<String, Integer> sizes, size_t cache_line_size, size_t max_n_bufs,
                       bool factoring, String save_load_prefix) {
      std::string save_load_path = save_load_prefix + (factoring ? ".json" : "_nofactor.json");
      std::ifstream fin(save_load_path);
      FeaturePack fp;
      if (fin.is_open()) {
        dmlc::JSONReader reader(&fin);
        auto fp_opt = FeaturePack::LoadFromJsonReader(reader);
        if (!fp_opt) {
          LOG_WARNING << "File " << save_load_path << " notes previous feature extraction failure";
          return FeaturePackPy();  // None
        }
        fp = fp_opt.value();
      } else {
        LOG_INFO << "Extracting features to save to " << save_load_path;
        auto fp_opt =
            GetFeaturePack(stmt, context, hw_params, factoring, cache_line_size, max_n_bufs);
        if (fp_opt) {
          fp = fp_opt.value();
          FeaturePack::SaveAsJson(save_load_path, fp);
        } else {
          FeaturePack::SaveAsJson(save_load_path, std::nullopt);
          return FeaturePackPy();  // None
        }
      }
      fp.RunSizeSubstitution(sizes);
      return fp.IntoPythonFeaturePack();
    });

TVM_REGISTER_GLOBAL("felix.LinearExprAsPrimExpr").set_body_typed([](LinearExpr e) {
  return e.ToPrimExpr();
});

}  // namespace felix
}  // namespace tvm

namespace dmlc {
namespace json {

template <typename V>
struct Handler<std::unordered_map<uint64_t, V>> {
  static void Write(JSONWriter* writer, const std::unordered_map<uint64_t, V>& map) {
    writer->BeginArray(map.size() > 1);
    for (auto& [k, v] : map) {
      writer->WriteArraySeperator();
      writer->BeginArray(false);
      writer->WriteArrayItem(k);
      writer->WriteArrayItem(v);
      writer->EndArray();
    }
    writer->EndArray();
  }

  static void Read(JSONReader* reader, std::unordered_map<uint64_t, V>* map) {
    map->clear();
    reader->BeginArray();
    while (reader->NextArrayItem()) {
      uint64_t k;
      V v;
      reader->BeginArray();
      ICHECK(reader->NextArrayItem());
      reader->Read(&k);
      ICHECK(reader->NextArrayItem());
      reader->Read(&v);
      ICHECK(!reader->NextArrayItem());
      map->emplace(k, v);
    }
  }
};

template <>
struct Handler<::tvm::felix::LinearExpr> {
  static void Write(JSONWriter* writer, const tvm::felix::LinearExpr& e) {
    writer->BeginArray();
    writer->WriteArrayItem(e->constant->value);
    std::unordered_map<std::string, double> name2coef;
    for (auto& [var, coef] : e->lin_terms) {
      name2coef[var->name_hint] = coef->value;
    }
    writer->WriteArrayItem(name2coef);
    writer->EndArray();
  }
  static void Read(JSONReader* reader, tvm::felix::LinearExpr* e) {
    reader->BeginArray();
    ICHECK(reader->NextArrayItem());
    double constant;
    reader->Read(&constant);
    ICHECK(reader->NextArrayItem());
    std::unordered_map<std::string, double> name2coef;
    reader->Read(&name2coef);
    ICHECK(!reader->NextArrayItem());
    *e = tvm::felix::LinearExpr(constant, name2coef);
  }
};
}  // namespace json
}  // namespace dmlc