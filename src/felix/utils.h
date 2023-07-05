#ifndef FELIX_UTILS_H_
#define FELIX_UTILS_H_

#include <tvm/auto_scheduler/loop_state.h>
#include <tvm/auto_scheduler/transform_step.h>
#include <tvm/runtime/container/string.h>

#include <numeric>

namespace tvm {
namespace felix {

using auto_scheduler::IteratorKind;
using auto_scheduler::Stage;
using auto_scheduler::Step;

String PrintTrStep(const Step& step);

/*! \brief Compute the product of all elements in a vector */
inline PrimExpr ElementProduct(const std::vector<PrimExpr>& array) {
  return std::accumulate(array.begin(), array.end(), PrimExpr(1),
                         [](const PrimExpr& a, const PrimExpr& b) { return a * b; });
}

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

#define LOG_TIME

class Timer {
 public:
  Timer(std::string event_name)
      : event_name(std::move(event_name)),
        duration(),
        now(std::chrono::high_resolution_clock::now()),
        stopped(false) {}

  void Start() {
#ifdef LOG_TIME
    this->now = std::chrono::high_resolution_clock::now();
    this->stopped = false;
#endif
  }

  void Stop() {
#ifdef LOG_TIME
    if (!this->stopped) {
      this->duration += std::chrono::high_resolution_clock::now() - this->now;
      this->stopped = true;
    }
#endif
  }

  ~Timer() {
#ifdef LOG_TIME
    Stop();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(this->duration);
    LOG_INFO << this->event_name << " -- " << duration.count() << " ms";
#endif
  }

 private:
  std::string event_name;
  std::chrono::duration<double> duration;
  std::chrono::time_point<std::chrono::high_resolution_clock> now;
  bool stopped;
};

}  // namespace felix
}  // namespace tvm

#endif