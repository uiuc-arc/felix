#ifndef TVM_ARITH_EGG_SIMPL_H
#define TVM_ARITH_EGG_SIMPL_H

#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace arith {

inline size_t CountOps(const PrimExpr& expr) {
  size_t count = 0;
  tir::PostOrderVisit(expr, [&count](const ObjectRef& node) {
    if (!node->IsInstance<tir::CastNode>()) {
      count++;
    }
  });
  return count;
}

PrimExpr SimplifyExpr(const PrimExpr& expr, size_t max_n_iters = 30, size_t max_n_nodes = 10000,
                      bool diff_approx = false);

PrimExpr SubAndSimplify(const PrimExpr& expr,
                        const std::unordered_map<std::string, PrimExpr>& subst,
                        size_t max_n_iters = 30, size_t max_n_nodes = 10000);

bool IsExprEquivalent(const PrimExpr& lhs, const PrimExpr& rhs, size_t max_n_iters = 30,
                      size_t max_n_nodes = 10000, bool diff_approx = false);

String PrintExprPrefix(const PrimExpr& expr);

PrimExpr ParseExprPrefix(const String& str);

}  // namespace arith
}  // namespace tvm

namespace dmlc {
namespace json {

template <>
struct Handler<tvm::String> {
  inline static void Write(JSONWriter* writer, const tvm::String& e) {
    writer->Write(std::string(e));
  }
  inline static void Read(JSONReader* reader, tvm::String* out) {
    std::string str;
    reader->Read(&str);
    *out = tvm::String(str);
  }
};

template <>
struct Handler<tvm::PrimExpr> {
  inline static void Write(JSONWriter* writer, const tvm::PrimExpr& e) {
    writer->Write(tvm::arith::PrintExprPrefix(e));
  }
  inline static void Read(JSONReader* reader, tvm::PrimExpr* out) {
    std::string str;
    reader->Read(&str);
    *out = tvm::arith::ParseExprPrefix(str);
  }
};

}  // namespace json
}  // namespace dmlc

#endif
