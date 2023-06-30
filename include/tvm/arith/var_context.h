#ifndef TVM_ARITH_VAR_CONTEXT_H_
#define TVM_ARITH_VAR_CONTEXT_H_

#include <tvm/arith/egg_simpl.h>
#include <tvm/te/schedule.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace arith {

using VarMapT = std::unordered_map<std::string, PrimExpr>;

class VarExprPairNode : public Object {
 public:
  VarExprPairNode(Var var, PrimExpr expr) : var(std::move(var)), expr(std::move(expr)) {}
  VarExprPairNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("var", &var);
    v->Visit("expr", &expr);
  }

  static constexpr const char* _type_key = "arith.VarExprPair";
  TVM_DECLARE_FINAL_OBJECT_INFO(VarExprPairNode, Object);

  Var var;
  PrimExpr expr;
};

class VarExprPair : public ObjectRef {
 public:
  VarExprPair(Var var, PrimExpr expr) : VarExprPair(make_object<VarExprPairNode>(var, expr)) {}
  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(VarExprPair, ObjectRef, VarExprPairNode);
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

  Var Append(const std::string& vname, const PrimExpr& expr);
  Var FindOrAppend(const std::string& vname, const PrimExpr& expr);

  VarDefStackNode Prepend(const VarMapT& vmap) const;
  VarMapT IntoVarMap() const;

  bool Contains(const std::string& vname) const {
    return this->var2idx.find(vname) != this->var2idx.end();
  }
  size_t Size() const { return this->exprs.size(); }
  const ContainerT& GetExprs() const { return this->exprs; }
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
  SplitGroupNode(PrimExpr extent, Array<String> vars, bool whole_div)
      : extent(std::move(extent)), vars(std::move(vars)), whole_div(whole_div) {}
  SplitGroupNode() = default;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("extent", &extent);
    v->Visit("vars", &vars);
    v->Visit("whole_div", &whole_div);
  }

  static constexpr const char* _type_key = "arith.SplitGroupNode";
  TVM_DECLARE_FINAL_OBJECT_INFO(SplitGroupNode, Object);

  PrimExpr extent;
  Array<String> vars;
  bool whole_div;
};

class SplitGroup : public ObjectRef {
 public:
  SplitGroup(PrimExpr extent, Array<String> vars, bool whole_div)
      : SplitGroup(make_object<SplitGroupNode>(extent, vars, whole_div)) {}

  TVM_DEFINE_NOTNULLABLE_OBJECT_REF_METHODS(SplitGroup, ObjectRef, SplitGroupNode);
};

class VarContextNode : public Object {
 public:
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("split_groups", &this->split_groups);
    v->Visit("var_defs", &this->var_defs);
  }

  Array<tir::SizeVar> GetSplitVars(PrimExpr extent, size_t n_splits, bool whole_div);
  std::pair<PrimExpr, PrimExpr> GetSplitSizes(PrimExpr extent, PrimExpr factor,
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
  std::unordered_multimap<PrimExpr, PrimExpr, StructuralHash, StructuralEqual> div_map{};
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
  inline static void Write(dmlc::JSONWriter* writer, const tvm::Array<T>& array) {
    writer->BeginArray();
    for (const auto& item : array) {
      writer->WriteArrayItem(item);
    }
    writer->EndArray();
  }

  inline static void Read(dmlc::JSONReader* reader, tvm::Array<T>* array) {
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
struct Handler<tvm::arith::SplitGroup> {
  inline static void Write(dmlc::JSONWriter* writer, const tvm::arith::SplitGroup& group) {
    writer->BeginArray();
    writer->WriteArrayItem(group->extent);
    writer->WriteArrayItem(group->vars);
    writer->WriteArrayItem(group->whole_div);
    writer->EndArray();
  }

  inline static void Read(dmlc::JSONReader* reader, tvm::arith::SplitGroup* group) {
    tvm::PrimExpr extent;
    tvm::Array<tvm::String> vars;
    bool whole_div;
    reader->BeginArray();
    ICHECK(reader->NextArrayItem());
    reader->Read(&extent);
    ICHECK(reader->NextArrayItem());
    reader->Read(&vars);
    ICHECK(reader->NextArrayItem());
    reader->Read(&whole_div);
    ICHECK(!reader->NextArrayItem());
    *group = tvm::arith::SplitGroup(std::move(extent), std::move(vars), whole_div);
  }
};

template <>
struct Handler<tvm::arith::VarDefStack> {
  inline static void Write(dmlc::JSONWriter* writer, const tvm::arith::VarDefStack& vmap) {
    writer->BeginObject();
    for (const auto& pair : vmap->GetExprs()) {
      writer->WriteObjectKeyValue(pair->var->name_hint, pair->expr);
    }
    writer->EndObject();
  }

  inline static void Read(dmlc::JSONReader* reader, tvm::arith::VarDefStack* group) {
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
