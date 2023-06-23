#ifndef FELIX_SKETCH_RULES_
#define FELIX_SKETCH_RULES_

#include <tvm/auto_scheduler/loop_state.h>
#include <tvm/auto_scheduler/search_task.h>
#include <tvm/auto_scheduler/sketch_policy.h>

#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace felix {

using auto_scheduler::SearchTask;
using auto_scheduler::State;

// We reuse most of the initial sketch generation rules from sketch_policy_rules.h
// but redefine this one to make it more symbolic.
using auto_scheduler::SketchGenerationRule;
using auto_scheduler::SketchPolicyNode;
DEFINE_SKETCH_GENERATION_RULE(SketchFelixCrossThreadReduction);

class ExtendSketchRule {
 public:
  virtual std::vector<State> Apply(const SketchPolicyNode* policy, State state) const = 0;
  virtual ~ExtendSketchRule() = default;
};

#define DEFINE_EXT_SKETCH_RULE(rule_name)                                                      \
  class rule_name : public ExtendSketchRule {                                                  \
   public:                                                                                     \
    virtual std::vector<State> Apply(const SketchPolicyNode* policy, State state) const final; \
  };

/*! \brief The rule that fills the incomplete SplitSteps. */
DEFINE_EXT_SKETCH_RULE(InitFillTileSize);

/*! \brief The rule that randomly changes the computation location for some stages that do not
 * need tiling and are not strictly inlineable(e.g. data padding). */
DEFINE_EXT_SKETCH_RULE(InitChangeComputeLocation);

/*! \brief The rule that annotates parallel for CPU. */
DEFINE_EXT_SKETCH_RULE(InitParallel);

/*! \brief The rule that annotates unroll. */
DEFINE_EXT_SKETCH_RULE(InitUnroll);

/*! \brief The rule that annotates vectorization. */
DEFINE_EXT_SKETCH_RULE(InitVectorization);

/*! \brief The rule that annotates thread binding for GPU. */
DEFINE_EXT_SKETCH_RULE(InitThreadBind);

}  // namespace felix
}  // namespace tvm

#endif  // FELIX_SKETCH_RULES_
