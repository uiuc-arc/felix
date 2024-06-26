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

TVM_REGISTER_NODE_TYPE(VarExprPairNode);
TVM_REGISTER_NODE_TYPE(VarDefStackNode);
TVM_REGISTER_NODE_TYPE(SplitGroupNode);
TVM_REGISTER_NODE_TYPE(VarContextNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<VarDefStackNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const VarDefStackNode*>(node.get());
      auto vname_expr = op->GetExprs();
      p->stream << "{";
      for (size_t i = 0; i < vname_expr.size(); i++) {
        auto& pair = vname_expr[i];
        if (i != 0) {
          p->stream << ",\n";
        }
        p->stream << pair->var << ": " << pair->expr;
      }
      p->stream << "}";
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SplitGroupNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SplitGroupNode*>(node.get());
      if (op->quotient.defined()) {
        p->stream << "SplitGroup(" << op->extent << " / " << op->vars << " = " << op->quotient
                  << ")";
      } else {
        p->stream << "SplitGroup(" << op->extent << " // " << op->vars << ")";
      }
    });

bool HasDiv(const PrimExpr& expr) {
  bool found = false;
  PostOrderVisit(expr, [&expr, &found](const ObjectRef& node) {
    if (node->IsInstance<DivNode>()) {
      found = true;
    }
  });
  return found;
}

bool ExprIsConstant(const PrimExpr& expr) {
  bool is_const = true;
  PostOrderVisit(expr, [&is_const](const ObjectRef& node) {
    if (node->IsInstance<VarNode>()) {
      auto* svnode = node.as<SizeVarNode>();
      if (!svnode || svnode->kind == SizeVarKind::kOther) {
        is_const = false;
      }
    }
  });
  return is_const;
}

bool ExprIsShapeConstant(const PrimExpr& expr) {
  bool shape_var_only = true;
  PostOrderVisit(expr, [&shape_var_only](const ObjectRef& node) {
    if (node->IsInstance<VarNode>()) {
      auto* svnode = node.as<tir::SizeVarNode>();
      if (!svnode || svnode->kind != SizeVarKind::kShapeVar) {
        shape_var_only = false;
      }
    }
  });
  return shape_var_only;
}

SizeVar VarDefStackNode::Append(const std::string& vname, const PrimExpr& expr) {
  SizeVar ret(vname, SizeVarKind::kShorthand, expr.dtype());
  this->Append(ret, expr);
  return ret;
}

void VarDefStackNode::Append(const SizeVar& var, const PrimExpr& expr) {
  ICHECK(expr.defined());
  ICHECK(ExprIsConstant(expr)) << "Expression " << expr
                               << " is not constant and cannot be inserted into VarDefStack";
  this->var2idx.Set(var->name_hint, this->exprs.size());
  this->expr2idx.emplace(expr, this->exprs.size());
  this->exprs.push_back(VarExprPair(var, expr));
}

PrimExpr VarDefStackNode::DefineConstShorthand(PrimExpr expr) {
  if (ExprIsConstant(expr) && CountOps(expr) >= 10) {
    std::string name = "v" + std::to_string(this->exprs.size());
    expr = Analyzer().canonical_simplify(arith::SimplifyExpr(expr));
    auto it = this->expr2idx.find(expr);
    if (it == this->expr2idx.end()) {
      expr = this->Append(name, expr);
    } else {
      expr = this->exprs[it->second]->var;
    }
  }
  return expr;
}

VarMapT VarDefStackNode::IntoUnwindedVarMap() const {
  VarMapT vmap;
  for (const auto& pair : this->exprs) {
    vmap.emplace(pair->var->name_hint, SubstByName(pair->expr, vmap));
  }
  return vmap;
}

std::unordered_set<std::string> VarDefStackNode::GetAllUsedVars(
    std::optional<SizeVarKind> kind) const {
  std::unordered_set<std::string> ret;
  for (auto& pair : this->exprs) {
    PostOrderVisit(pair->expr, [this, &ret, kind](const ObjectRef& node) {
      if (auto* svnode = node.as<SizeVarNode>()) {
        if (!kind || svnode->kind == kind) {
          ret.insert(svnode->name_hint);
        }
      }
    });
  }
  return ret;
}

void VarDefStackNode::MapExprs(ExprMutator func) {
  for (size_t i = 0; i < this->exprs.size(); i++) {
    auto& pair = this->exprs[i];
    this->exprs.Set(i, VarExprPair(pair->var, func(pair->expr)));
  }
}

void VarDefStackNode::MapExprsParallel(ExprMutator func) {
  support::parallel_for(0, this->exprs.size(), [this, &func](int i) {
    auto& pair = this->exprs[i];
    this->exprs.Set(i, VarExprPair(pair->var, func(pair->expr)));
  });
}

inline std::pair<PrimExpr, PrimExpr> ConservativeDiv(PrimExpr extent, PrimExpr factor,
                                                     bool no_tighten_factor) {
  auto min_factor = no_tighten_factor ? factor : min(extent, factor);
  auto divided = indexdiv(extent + (factor - 1), factor);
  return std::make_pair(min_factor, divided);
}

Array<SizeVar> VarContextNode::GetSplitVars(PrimExpr extent, size_t n_splits, bool whole_div) {
  std::string group_idx = std::to_string(this->split_counter++);
  Array<SizeVar> vars;
  Array<String> var_names;
  PrimExpr product = 1;
  for (size_t i = 0; i < n_splits; i++) {
    String name = "sp_" + group_idx + "_" + std::to_string(i);
    SizeVar var(name, SizeVarKind::kScheduleKnob);
    vars.push_back(var);
    var_names.push_back(name);
    product *= var;
  }
  if (!whole_div) {
    // No need to add a SplitGroup if not whole_div.
    return vars;
  }
  extent = SimplifyExpr(extent);
  // Don't put non-const extents in the map. They can contain loop variables that are removed in
  // later transformation steps, and we can't keep track of that.
  ICHECK(ExprIsShapeConstant(extent));
  // Replace the extent with a (possibly) shorter symbol E{i}.
  // * Consider this a shape variable even though it can be a shape expression. This will be
  //   important later.
  SizeVar extent_var("E" + group_idx, SizeVarKind::kShapeVar);
  this->var_defs->Append(extent_var, extent);
  // Declare a quotient variable which would be equal to Ei / (sp_i_0*sp_i_1*...sp_i_j), as
  // qi. For example q3 could be Cout / (sp_3_0 * sp_3_1 * sp_3_2...) for group 3.
  // * Don't even define this variable in this->var_defs. We'll need to delay the expansion of q{i}
  //   as much as possible, and when finally it's needed it can be derived from the SplitGroup.
  SizeVar quotient("q" + group_idx, SizeVarKind::kShorthand);
  // Say that Extent_i and qi * (sp_i_0 * sp_i_1 * sp_i_2...) are equivalent.
  // This is useful when we figure out divisibility later.
  this->div_extents.emplace_back(extent, product * quotient);
  this->split_groups.push_back(SplitGroup(extent_var, quotient, var_names));
  return vars;
}

std::pair<PrimExpr, PrimExpr> VarContextNode::GetSplitSizes(const PrimExpr& extent, PrimExpr factor,
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
  return this->SymbolicDiv(extent, factor, no_tighten_factor);
}

std::pair<PrimExpr, PrimExpr> VarContextNode::SymbolicDiv(PrimExpr numer, PrimExpr denom,
                                                          bool no_tighten_factor) {
  numer = SimplifyExpr(numer), denom = SimplifyExpr(denom);
  PrimExpr simpl = SimplifyExpr(numer / denom);
  if (!HasDiv(simpl)) {
    return {denom, simpl};
  }
  for (auto& [extent, subst] : this->div_extents) {
    if (IsExprEquivalent(numer, extent)) {
      PrimExpr simpl = SimplifyExpr(subst / denom);
      if (!HasDiv(simpl)) {
        return {denom, simpl};
      }
    }
  }
  return ConservativeDiv(numer, denom, no_tighten_factor);
}

TVM_REGISTER_GLOBAL("arith.VarContextGetVarDefs").set_body_typed([](VarContext context) {
  Array<Array<ObjectRef>> ret;
  for (const auto& pair : context->var_defs->GetExprs()) {
    ret.push_back(Array<ObjectRef>({pair->var, pair->expr}));
  }
  return ret;
});

}  // namespace arith
}  // namespace tvm