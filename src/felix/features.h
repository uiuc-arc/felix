#ifndef TVM_AUTO_SCHEDULER_FEATURES_H_
#define TVM_AUTO_SCHEDULER_FEATURES_H_

#include <tvm/auto_scheduler/search_task.h>
#include <tvm/tir/stmt.h>

#include <unordered_map>

#include "rangeinfer.h"

namespace tvm {
namespace felix {

template <class T>
using BufferMap = std::unordered_map<tir::Buffer, T, ObjectHash, ObjectEqual>;
template <class T>
using ExprMap = std::unordered_map<PrimExpr, T, StructuralHash, StructuralEqual>;

arith::VarDefStack GetPerStoreFeatureExpr(const tir::Stmt& stmt, arith::VarDefStackNode& vdefs,
                                          RangeInfer& rinf, size_t cache_line_size,
                                          size_t max_n_bufs);

std::vector<PrimExpr> GetConstraints(const tir::Stmt& code,
                                     const auto_scheduler::HardwareParams& hw_params);

}  // namespace felix
}  // namespace tvm

#endif  // TVM_AUTO_SCHEDULER_FEATURE_H_
