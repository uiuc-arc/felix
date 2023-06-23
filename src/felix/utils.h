#ifndef FELIX_UTILS_H_
#define FELIX_UTILS_H_

#include <tvm/auto_scheduler/loop_state.h>
#include <tvm/auto_scheduler/transform_step.h>
#include <tvm/runtime/container/string.h>

namespace tvm {
namespace felix {

using auto_scheduler::IteratorKind;
using auto_scheduler::Stage;
using auto_scheduler::Step;

String PrintTrStep(const Step& step);

inline std::pair<PrimExpr, PrimExpr> GetCumulativeSpaceAndReductionLength_(const Stage& stage) {
  PrimExpr cum_space_len = 1, cum_reduce_len = 1;
  for (const auto& iter : stage->iters) {
    if (iter->iter_kind == IteratorKind::kSpatial) {
      cum_space_len *= iter->range->extent;
    } else if (iter->iter_kind == IteratorKind::kReduction) {
      cum_reduce_len *= iter->range->extent;
    }
  }
  return std::make_pair(cum_space_len, cum_reduce_len);
}

}  // namespace felix
}  // namespace tvm

#endif