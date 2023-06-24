#include <tvm/arith/analyzer.h>
#include <tvm/arith/egg_simpl.h>
#include <tvm/arith/var_context.h>
#include <tvm/support/parallel_for.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <numeric>

namespace tvm {
namespace arith {

using namespace tir;

TVM_REGISTER_NODE_TYPE(VarContextNode);

inline bool HasDiv(const PrimExpr& expr) {
  bool found = false;
  PostOrderVisit(expr, [&expr, &found](const ObjectRef& node) {
    if (node->IsInstance<DivNode>()) {
      found = true;
    }
  });
  return found;
}

Var VarDefStack::Append(const std::string& vname, const PrimExpr& expr) {
  auto it = this->var2idx.find(vname);
  if (it == this->var2idx.end()) {
    this->var2idx.emplace(vname, this->exprs.size());
    this->expr2idx.emplace(expr, this->exprs.size());
    tir::Var var(vname, expr->dtype);
    this->exprs.emplace_back(var, expr);
    return var;
  } else {
    auto& [v, e] = this->exprs[it->second];
    e = expr;
    return v;
  }
}

Var VarDefStack::FindOrAppend(const std::string& vname, const PrimExpr& expr) {
  auto it = this->expr2idx.find(expr);
  if (it != this->expr2idx.end()) {
    return this->exprs[it->second].first;
  }
  return this->Append(vname, expr);
}

VarDefStack VarDefStack::Prepend(const VarMapT& vmap) const {
  VarDefStack ret;
  for (auto& [vname, expr] : vmap) {
    ret.Append(vname, expr);
  }
  for (auto& [var, expr] : this->exprs) {
    ret.Append(var->name_hint, expr);
  }
  return ret;
}

VarMapT VarDefStack::IntoVarMap() const {
  VarMapT vmap;
  for (const auto& [vname, expr] : this->exprs) {
    vmap.emplace(vname->name_hint, tir::SubstByName(expr, vmap));
  }
  return vmap;
}

std::vector<Var> VarDefStack::FreeVars() {
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> vars;
  auto CollectVars = [&vars](const ObjectRef& node) {
    if (const VarNode* op = node.as<VarNode>()) {
      vars.insert(GetRef<Var>(op));
    }
  };
  for (const auto& [_, expr] : this->GetExprs()) {
    tir::PostOrderVisit(expr, CollectVars);
  }
  for (const auto& [var, _] : this->GetExprs()) {
    vars.erase(var);
  }
  return std::vector<Var>(vars.begin(), vars.end());
}

bool VarDefStack::HasUndefVars(const PrimExpr& expr) {
  // SizeVar is seen as constant and doesn't count.
  bool has_undef = false;
  auto CheckUndef = [&has_undef, this](const ObjectRef& obj) {
    auto* vnode = obj.as<VarNode>();
    auto* svnode = obj.as<SizeVarNode>();
    if (vnode && this->var2idx.count(vnode->name_hint) == 0 &&
        !(svnode && svnode->is_const_symbol)) {
      has_undef = true;
    }
  };
  tir::PostOrderVisit(expr, CheckUndef);
  return has_undef;
}

inline std::pair<PrimExpr, PrimExpr> ConservativeDiv(PrimExpr extent, PrimExpr factor,
                                                     bool no_tighten_factor) {
  auto min_factor = no_tighten_factor ? factor : min(extent, factor);
  auto divided = indexdiv(extent + (factor - 1), factor);
  return std::make_pair(min_factor, divided);
}

Array<SizeVar> VarContextNode::GetSplitVars(PrimExpr extent, size_t n_splits, bool whole_div) {
  Array<SizeVar> vars;
  Array<String> var_names;
  PrimExpr product = 1;
  for (size_t i = 0; i < n_splits; i++) {
    String name = "sp_" + std::to_string(this->split_counter) + "_" + std::to_string(i);
    SizeVar var(name, DataType::Int(32), Span(), true);
    vars.push_back(var);
    var_names.push_back(name);
    product *= var;
  }
  ++this->split_counter;
  extent = Analyzer().canonical_simplify(extent);
  this->split_groups.push_back(SplitGroup(extent, var_names, whole_div));
  if (this->var_defs.HasUndefVars(extent)) {
    // Don't put non-const extents in the map. They can contain loop variables that are removed in
    // later transformation steps, and we can't keep track of that.
    ICHECK(n_splits == 1 && !whole_div);
    return vars;
  }
  Var quotient = AllocVarForExpr(extent / product, false);
  this->div_map.emplace(extent, product * quotient);
  return vars;
}

std::pair<PrimExpr, PrimExpr> VarContextNode::GetSplitSizes(PrimExpr extent, PrimExpr factor,
                                                            bool no_tighten_factor) {
  // Fast path for when the inputs are not symbolic.
  if (is_const_int(extent) && is_const_int(factor)) {
    // Use conservative division; consts will properly simplify anyway.
    return ConservativeDiv(extent, factor, no_tighten_factor);
  }
  // A special case that can happen in loop splitting.
  if (is_const_int(extent, 1)) {
    // 1 splitted by anything is still (1, 1).
    return {extent, extent};
  }
  extent = Analyzer().canonical_simplify(extent);
  auto [min_factor, divided] = this->SymbolicDiv(extent, factor, no_tighten_factor);
  return {this->DefineConstShorthand(min_factor), this->DefineConstShorthand(divided)};
}

void VarContextNode::DefineVar(const std::string& name, PrimExpr expr) {
  this->var_defs.Append(name, expr);
}

PrimExpr VarContextNode::DefineConstShorthand(PrimExpr expr) {
  if (!this->var_defs.HasUndefVars(expr) && CountOps(expr) >= 10) {
    expr = this->AllocVarForExpr(arith::SimplifyExpr(expr), true);
  }
  return expr;
}

Var VarContextNode::AllocVarForExpr(PrimExpr expr, bool is_shorthand) {
  std::string name = (is_shorthand ? "v" : "d") + std::to_string(this->var_defs.Size());
  return this->var_defs.FindOrAppend(name, expr);
}

std::pair<PrimExpr, PrimExpr> VarContextNode::SymbolicDiv(PrimExpr numer, PrimExpr denom,
                                                          bool no_tighten_factor) {
  auto [begin, end] = this->div_map.equal_range(numer);
  for (; begin != end; ++begin) {
    PrimExpr simpl = SimplifyExpr(begin->second / denom);
    if (!HasDiv(simpl)) {
      return {denom, simpl};
    }
  }
  PrimExpr simpl = SimplifyExpr(numer / denom);
  if (HasDiv(simpl)) {
    return ConservativeDiv(numer, denom, no_tighten_factor);
  } else {
    return {denom, simpl};
  }
}

TVM_REGISTER_GLOBAL("arith.VarMapToStr").set_body_typed([](VarContext context) {
  std::ostringstream os;
  for (auto& [k, v] : context->var_defs.GetExprs()) {
    os << k << " = " << v << "\n";
  }
  return String(os.str());
});

}  // namespace arith
}  // namespace tvm