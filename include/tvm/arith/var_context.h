#ifndef TVM_ARITH_VAR_CONTEXT_H_
#define TVM_ARITH_VAR_CONTEXT_H_

#include <tvm/arith/egg_simpl.h>
#include <tvm/te/schedule.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace arith {

using VarMapT = std::unordered_map<std::string, PrimExpr>;

bool ExprIsConstant(const PrimExpr& expr);
bool ExprIsShapeConstant(const PrimExpr& expr);

class VarExprPairNode : public Object {
 public:
  VarExprPairNode(tir::SizeVar var, PrimExpr expr) : var(std::move(var)), expr(std::move(expr)) {}
  VarExprPairNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("var", &var);
    v->Visit("expr", &expr);
  }

  static constexpr const char* _type_key = "arith.VarExprPair";
  TVM_DECLARE_FINAL_OBJECT_INFO(VarExprPairNode, Object);

  tir::SizeVar var;
  PrimExpr expr;
};

class VarExprPair : public ObjectRef {
 public:
  VarExprPair(tir::SizeVar var, PrimExpr expr)
      : VarExprPair(make_object<VarExprPairNode>(var, expr)) {}
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(VarExprPair, ObjectRef, VarExprPairNode);
};

class VarDefStackNode : public Object {
  using ContainerT = Array<VarExprPair>;
  using ExprVisitor = std::function<PrimExpr(const PrimExpr& e)>;
  using ThreadedExprVisitor = std::function<PrimExpr(const PrimExpr& e, size_t)>;
  using VarExprVisitor = std::function<PrimExpr(const Var& v, const PrimExpr& e)>;

 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("exprs", &exprs);
    v->Visit("var2idx", &var2idx);
    // No need to save expr2idx (just a memo table)
  }

  tir::SizeVar Append(const std::string& vname, const PrimExpr& expr);
  void Append(const tir::SizeVar& var, const PrimExpr& expr);
  tir::SizeVar FindOrAppend(const std::string& vname, const PrimExpr& expr);

  VarDefStackNode Prepend(const VarMapT& vmap) const;
  VarMapT IntoVarMap() const;

  bool Contains(const std::string& vname) const {
    return this->var2idx.find(vname) != this->var2idx.end();
  }
  size_t Size() const { return this->exprs.size(); }
  const ContainerT& GetExprs() const { return this->exprs; }
  PrimExpr& GetExprAt(const std::string& vname) {
    auto it = this->var2idx.find(vname);
    ICHECK(it != this->var2idx.end()) << "Var " << vname << " not found in VarDefStack";
    return this->exprs[(*it).second]->expr;
  }

  std::vector<Var> FreeVars() const;
  bool HasUndefVars(const PrimExpr& expr) const;

  static constexpr const char* _type_key = "arith.VarDefStack";
  TVM_DECLARE_FINAL_OBJECT_INFO(VarDefStackNode, Object);

 private:
  ContainerT exprs;
  Map<String, Integer> var2idx;
  std::unordered_map<PrimExpr, size_t, StructuralHash, StructuralEqual> expr2idx;
};

class VarDefStack : public ObjectRef {
 public:
  VarDefStack() : VarDefStack(make_object<VarDefStackNode>()) {}
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(VarDefStack, ObjectRef, VarDefStackNode);
};

class SplitGroupNode : public Object {
 public:
  SplitGroupNode(tir::SizeVar extent, tir::SizeVar quotient, Array<String> vars)
      : extent(std::move(extent)), quotient(std::move(quotient)), vars(std::move(vars)) {}
  SplitGroupNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("extent", &extent);
    v->Visit("quotient", &quotient);
    v->Visit("vars", &vars);
  }

  static constexpr const char* _type_key = "arith.SplitGroupNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(SplitGroupNode, Object);

  tir::SizeVar extent, quotient;
  Array<String> vars;
};

class SplitGroup : public ObjectRef {
 public:
  SplitGroup(tir::SizeVar extent, tir::SizeVar quotient, Array<String> vars)
      : SplitGroup(make_object<SplitGroupNode>(extent, quotient, vars)) {}
  SplitGroup() : SplitGroup(make_object<SplitGroupNode>()) {}

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(SplitGroup, ObjectRef, SplitGroupNode);
};

class VarContextNode : public Object {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("split_groups", &this->split_groups);
    v->Visit("var_defs", &this->var_defs);
  }

  Array<tir::SizeVar> GetSplitVars(const PrimExpr& extent, size_t n_splits, bool whole_div);
  std::pair<PrimExpr, PrimExpr> GetSplitSizes(const PrimExpr& extent, PrimExpr factor,
                                              bool no_tighten_factor);

  void DefineVar(const std::string& name, PrimExpr expr);

  PrimExpr DefineConstShorthand(PrimExpr expr);

 private:
  Var AllocVarForExpr(PrimExpr expr);

  std::pair<PrimExpr, PrimExpr> SymbolicDiv(PrimExpr numer, PrimExpr denom, bool no_tighten_factor);

 public:
  static constexpr const char* _type_key = "arith.VarContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(VarContextNode, Object);

  Array<SplitGroup> split_groups{};
  VarDefStack var_defs{};

 private:
  std::vector<std::pair<PrimExpr, PrimExpr>> div_extents{};
  size_t split_counter{};
};

class VarContext : public ObjectRef {
 public:
  static VarContext MakeContext() {
    auto node = make_object<VarContextNode>();
    return VarContext(node);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(VarContext, ObjectRef, VarContextNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(VarContextNode);
};
}  // namespace arith
}  // namespace tvm

namespace dmlc {
namespace json {

template <typename T>
struct Handler<tvm::Array<T>> {
  inline static void Write(JSONWriter* writer, const tvm::Array<T>& array) {
    writer->BeginArray();
    for (const auto& item : array) {
      writer->WriteArrayItem(item);
    }
    writer->EndArray();
  }

  inline static void Read(JSONReader* reader, tvm::Array<T>* array) {
    array->clear();
    reader->BeginArray();
    while (reader->NextArrayItem()) {
      T item;
      reader->Read(&item);
      array->push_back(item);
    }
  }
};

template <>
struct Handler<tvm::tir::SizeVar> {
  static void Write(JSONWriter* writer, const tvm::tir::SizeVar& var) {
    writer->Write((tvm::PrimExpr)var);
  }
  static void Read(JSONReader* reader, tvm::tir::SizeVar* var) {
    tvm::PrimExpr expr;
    reader->Read(&expr);
    auto* sv = expr.as<tvm::tir::SizeVarNode>();
    ICHECK(sv) << "Expected SizeVar, got " << expr;
    *var = tvm::GetRef<tvm::tir::SizeVar>(sv);
  }
};

template <>
struct Handler<tvm::arith::SplitGroup> {
  inline static void Write(JSONWriter* writer, const tvm::arith::SplitGroup& group) {
    writer->BeginArray();
    writer->WriteArrayItem(group->extent);
    writer->WriteArrayItem(group->quotient);
    writer->WriteArrayItem(group->vars);
    writer->EndArray();
  }

  inline static void Read(JSONReader* reader, tvm::arith::SplitGroup* group) {
    tvm::tir::SizeVar extent, quotient;
    tvm::Array<tvm::String> vars;
    reader->BeginArray();
    ICHECK(reader->NextArrayItem());
    reader->Read(&extent);
    ICHECK(reader->NextArrayItem());
    reader->Read(&quotient);
    ICHECK(reader->NextArrayItem());
    reader->Read(&vars);
    ICHECK(!reader->NextArrayItem());
    *group = tvm::arith::SplitGroup(extent, quotient, vars);
  }
};

template <>
struct Handler<tvm::arith::VarDefStack> {
  inline static void Write(JSONWriter* writer, const tvm::arith::VarDefStack& vmap) {
    writer->BeginObject();
    for (const auto& pair : vmap->GetExprs()) {
      writer->WriteObjectKeyValue(pair->var->name_hint, pair->expr);
    }
    writer->EndObject();
  }

  inline static void Read(JSONReader* reader, tvm::arith::VarDefStack* group) {
    *group = tvm::arith::VarDefStack();
    reader->BeginObject();
    std::string vname;
    tvm::PrimExpr expr;
    while (reader->NextObjectItem(&vname)) {
      reader->Read(&expr);
      (*group)->Append(vname, expr);
    }
  }
};

}  // namespace json
}  // namespace dmlc
#endif
