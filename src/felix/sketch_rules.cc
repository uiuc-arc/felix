#include "sketch_rules.h"

#include <tvm/arith/var_context.h>
#include <tvm/auto_scheduler/search_policy.h>
#include <tvm/auto_scheduler/search_policy_utils.h>

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "utils.h"

namespace tvm {
namespace felix {

using namespace tvm::auto_scheduler;

/* Ansor sketch rules (copied from sketch_policy.cc) */
// Except cross_thread_reduction because we'll redefine it.
static RuleSkipStage rule_skip_stage;
static RuleAlwaysInline rule_always_inline;
static RuleMultiLevelTiling rule_multi_level_tiling;
static RuleMultiLevelTilingWithFusion rule_multi_level_tiling_with_fusion;
static RuleAddCacheRead rule_add_cache_read_stage;
static RuleAddCacheWrite rule_add_cache_write_stage;
static RuleAddRfactor rule_add_rfactor;
static RuleSimplifyComputeWithConstTensor rule_simplify_compute_with_const_tensor;
static RuleSpecialComputeLocationGPU rule_special_compute_location_gpu;

/* Felix sketch rules */
static SketchFelixCrossThreadReduction rule_felix_cross_thread_reduction;

/* Felix extended sketch rules */
static InitFillTileSize init_fill_tile_size;
static InitChangeComputeLocation init_change_compute_location;
static InitParallel init_parallel;
static InitUnroll init_unroll;
static InitVectorization init_vectorization;
static InitThreadBind init_thread_bind;

SketchGenerationRule::ConditionKind SketchFelixCrossThreadReduction::MeetCondition(
    const SketchPolicyNode& policy, const State& state, int stage_id) const {
  ICHECK(IsGPUTask(policy.search_task));

  // If it is an intermediate state created by RuleAddCacheWrite,
  // we just skip it.
  if (HasCacheWriteStage(state, stage_id)) {
    return ConditionKind::kSkip;
  }

  const auto& op = state->stages[stage_id]->op;
  if (op->IsInstance<te::ComputeOpNode>()) {
    // Compute the product of lengths of all space iters and all reduce iters
    auto [cum_space_len, cum_reduce_len] =
        GetCumulativeSpaceAndReductionLength_(state->stages[stage_id]);
    const auto nthreads = policy.search_task->hardware_params->max_threads_per_block;
    const auto warp_size = policy.search_task->hardware_params->warp_size;
    auto* cum_space_len_int = cum_space_len.as<IntImmNode>();
    auto* cum_reduce_len_int = cum_reduce_len.as<IntImmNode>();

    if (NeedsMultilevelTiling(policy.search_task, state, stage_id)) {
      // Avoid rfactor if we have enough parallelism on space iters
      if (cum_space_len_int && cum_space_len_int->value > nthreads) {
        return ConditionKind::kSkip;
      }
      if (cum_space_len_int && cum_reduce_len_int &&
          cum_space_len_int->value >= cum_reduce_len_int->value) {
        return ConditionKind::kSkip;
      }
      return ConditionKind::kApply;
    } else if (cum_reduce_len_int && cum_reduce_len_int->value > 1) {
      // Try rfactor for other reduction operators
      return cum_reduce_len_int->value > warp_size ? ConditionKind::kApply : ConditionKind::kSkip;
    }
  }

  return ConditionKind::kSkip;
}

std::vector<std::pair<State, int>> SketchFelixCrossThreadReduction::Apply(
    const SketchPolicyNode& policy, const State& state, int stage_id) const {
  const SearchTask& task = policy.search_task;
  State tmp_s = state;

  // fuse all reduction iters
  Array<Iterator> space_iters, reduce_iters;
  Iterator fused_reduce_iter;
  tmp_s =
      FuseAllReductionIterators(tmp_s, stage_id, &fused_reduce_iter, &space_iters, &reduce_iters);

  // Check the opportunity for kernel fusion
  bool fusible = false;
  int target_stage_id = GetSingleConsumerId(task, tmp_s, stage_id);
  int num_common_outer = -1;
  if (target_stage_id >= 0) {
    num_common_outer = GetNumCommonOuterIterator(task, tmp_s, stage_id, target_stage_id);
    if (num_common_outer > 0 && !NeedsMultilevelTiling(task, state, target_stage_id)) {
      fusible = true;
    }
  }

  if (fusible) {
    const Stage& target_stage = state->stages[target_stage_id];
    std::vector<int> split_step_ids;

    GetSplitStepIds(tmp_s, target_stage_id, &split_step_ids);

    if (split_step_ids.size() == 0) {
      // If the target stage does not have split step,
      // it must be a simple stage without reduce iters.
      // We then should do a split for it.
      ICHECK(!HasReduceIter(target_stage));
      const auto& split_res =
          tmp_s.split(target_stage_id, target_stage->iters.back(), {PrimExpr()});
      tmp_s.bind(target_stage_id, split_res[1], IteratorAnnotation::kThreadX);
      split_step_ids.push_back(tmp_s->transform_steps.size() - 2);
    }

    ICHECK_EQ(split_step_ids.size(), 1);

    const Iterator& target_iter = tmp_s->stages[target_stage_id]->iters[num_common_outer - 1];
    const auto& split_res = tmp_s.follow_split(stage_id, fused_reduce_iter, split_step_ids[0], 1);
    tmp_s.bind(stage_id, split_res[1], IteratorAnnotation::kThreadX);
    tmp_s.compute_at(stage_id, target_stage_id, target_iter);
  } else {
    const auto& split_res = tmp_s.split(stage_id, fused_reduce_iter, {PrimExpr()});
    tmp_s.bind(stage_id, split_res[1], IteratorAnnotation::kThreadX);
  }

  return {std::make_pair(std::move(tmp_s), stage_id - 1)};
}

/********** Init Population **********/

std::vector<State> InitFillTileSize::Apply(const SketchPolicyNode* policy, State state) const {
  // TODO: remember variable range constraints
  // int max_innermost_split_factor =
  //     GetIntParam(policy->params, SketchParamKey::max_innermost_split_factor);
  // Scan the transformation history and randomly fill tiles size for all SplitStep
  Array<Step> new_steps;
  for (auto& step : state->transform_steps) {
    if (auto ps = step.as<SplitStepNode>()) {
      ICHECK(ps->extent);
      bool any_defined = std::any_of(ps->lengths.begin(), ps->lengths.end(),
                                     [](const ObjectRef& item) { return item.defined(); });
      auto vars = state.GetVarContext().GetSplitVars(ps->extent.value(), ps->lengths.size(), true);
      Array<PrimExpr> new_split_lengths(vars.begin(), vars.end());
      if (any_defined) {
        LOG_WARNING << "Replacing already defined split sizes " << ps->lengths << " with "
                    << new_split_lengths;
      }
      new_steps.push_back(
          SplitStep(ps->stage_id, ps->iter_id, ps->extent, new_split_lengths, ps->inner_to_outer));
    } else {
      new_steps.push_back(step);
    }
  }
  StateNode* ret_node = state.CopyOnWrite();
  ret_node->concrete = true;
  ret_node->transform_steps = new_steps;
  return {GetRef<State>(ret_node)};
}

// CCL -> Change Compute Location
void CCLGetNextStates(const SearchTask& task, const State& state, int stage_id,
                      std::vector<State>& ret) {
  const Stage& stage = state->stages[stage_id];
  // Skip the inlined stages and placeholders
  if (stage->op_type == StageKind::kPlaceholder || stage->compute_at == ComputeAtKind::kInlined) {
    ret.push_back(state);
    return;
  }
  // Skip the tiled stages
  if (IsTiled(stage) || NeedsMultilevelTiling(task, state, stage_id)) {
    ret.push_back(state);
    return;
  }

  std::vector<std::pair<int, int>> candidates = GetComputeLocationCandidates(task, state, stage_id);
  if (!HasReduceIter(stage)) {
    State inline_ = state;
    const auto& map = inline_->attach_map->stage_to_attach_iter;
    if (map.find(stage_id) != map.end()) {
      inline_.compute_inline(stage_id);
      ret.push_back(inline_);
    }
  }
  State root = state;
  root.compute_root(stage_id);
  ret.push_back(root);
  for (auto [target_stage_id, iter_id] : candidates) {
    State compute_at = state;
    auto& stage = compute_at->stages[target_stage_id];
    compute_at.compute_at(stage_id, target_stage_id, stage->iters[iter_id]);
    ret.push_back(compute_at);
  }
}

std::vector<State> InitChangeComputeLocation::Apply(const SketchPolicyNode* policy,
                                                    State state) const {
  if (GetIntParam(policy->params, SketchParamKey::disable_change_compute_location)) {
    return {state};
  }
  // ChangeComputeLocation does not change the number of stages.
  // We are essentially doing a BFS on a tree with depth n_stages,
  // using a for-loop that ticks down: `stage_id in range(n_stages - 1, -1, -1)`.
  int n_stages = static_cast<int>(state->stages.size());
  auto& task = policy->search_task;
  std::vector<State> cur_states{state}, next_states{};  // All states at current stage_id
  for (int stage_id = n_stages - 1; stage_id >= 0; --stage_id) {
    for (auto& state : cur_states) {
      auto& stages = state->stages;
      ICHECK(static_cast<int>(stages.size()) == n_stages);
      const Stage& stage = stages[stage_id];
      CCLGetNextStates(task, state, stage_id, next_states);
    }
    cur_states.swap(next_states);
    next_states.clear();
  }

  std::vector<State> ret;
  // InferBound has a vector<State> -> vector<State> variant
  // but it is parallelized and our VarContext isn't thread-safe.
  for (auto& state : cur_states) {
    try {
      ret.push_back(task->compute_dag.InferBound(state));
    } catch (Error& e) {
      LOG_WARNING << "InferBound fails on the state:\n"
                  << state << "\n"
                  << "with: " << e.what() << std::endl;
    }
  }
  return ret;
}

void AnnotateParallel(const SearchTask& task, State& state, int stage_id, int iter_offset) {
  const Stage& stage = state->stages[stage_id];
  auto& attached_stages = state->attach_map->iter_to_attached_stages;
  Array<Iterator> to_fuse;
  PrimExpr parallel_degree = Integer(1);
  bool any_parallel = false;
  // Try to fuse and parallel the outermost n iterators
  // Stop if we meet reduce iterator or we have enough parallel degree
  size_t iter_id = iter_offset;
  for (; iter_id < stage->iters.size(); ++iter_id) {
    const Iterator& it = stage->iters[iter_id];
    if (it->iter_kind == IteratorKind::kReduction || it->annotation != IteratorAnnotation::kNone) {
      break;
    }
    to_fuse.push_back(it);
    parallel_degree *= it->range->extent;
    any_parallel = true;
    // TODO: try to generate multiple sketches from this condition?
    // if (parallel_degree > policy.search_task->hardware_params->num_cores * 16) {
    //   break;
    // }
    if (attached_stages.count({stage_id, iter_id})) {
      break;
    }
  }

  if (!any_parallel) {
    auto res = attached_stages.find({stage_id, iter_id});
    if (res != attached_stages.end()) {
      for (int attached_stage_id : res->second) {
        AnnotateParallel(task, state, attached_stage_id, 0);
      }
      AnnotateParallel(task, state, stage_id, iter_id + 1);
    }
  }

  if (!to_fuse.empty()) {
    State ret = state;
    if (to_fuse.size() == 1) {
      ret.parallel(stage_id, to_fuse[0]);
    } else {
      Iterator fused_iter = ret.fuse(stage_id, to_fuse);
      ret.parallel(stage_id, fused_iter);
    }
  }
}

std::vector<State> InitParallel::Apply(const SketchPolicyNode* policy, State state) const {
  for (size_t stage_id = 0; stage_id < state->stages.size(); ++stage_id) {
    const Stage& stage = state->stages[stage_id];
    if (stage->compute_at != ComputeAtKind::kRoot || stage->op_type == StageKind::kPlaceholder) {
      continue;
    }
    AnnotateParallel(policy->search_task, state, stage_id, 0);
  }
  return {state};
}

std::vector<State> InitUnroll::Apply(const SketchPolicyNode* policy, State state) const {
  for (size_t stage_id = 0; stage_id < state->stages.size(); ++stage_id) {
    const Stage& stage = state->stages[stage_id];
    // Skip the inlined stage and placeholder stage
    if (stage->compute_at == ComputeAtKind::kInlined || stage->op_type == StageKind::kPlaceholder) {
      continue;
    }

    // Handle always_unroll_inner attr
    if (stage->op->attrs.count(SearchPolicyKey::always_unroll_inner)) {
      const auto& to_unroll_name_set =
          GetIterNameSetParam(stage->op->attrs, SearchPolicyKey::always_unroll_inner);

      // Unroll the space iterators and reduce iterators listed in the attrs in the innermost
      // tile
      std::set<std::string> visited_names;
      for (int n = static_cast<int>(stage->iters.size()) - 1; n >= 0; n--) {
        const Iterator& it = stage->iters[n];

        // If we meet two iterators that come from a same original iterator,
        // then we are out of the innermost tile
        size_t size_before = visited_names.size();
        ExtractOriginalIterators(it->name, &visited_names);
        if (size_before == visited_names.size()) {
          break;
        }

        std::set<std::string> name;
        ExtractOriginalIterators(it->name, &name);
        if (name.size() == 1 && to_unroll_name_set.count(*name.begin())) {
          if (it->annotation == IteratorAnnotation::kNone) {
            state.unroll(stage_id, it);
          }
        }
      }
    }

    if (HasReduceIter(stage)) {
      // Use auto unroll for multi level tiled stage
      auto var_name = "ur_" + std::to_string(stage_id);
      state.pragma(stage_id, state->stages[stage_id]->iters[0],
                   std::string("auto_unroll_max_step") + "$" + var_name);
    }
  }

  return {state};
}

void VecGetNextStates(const SketchPolicyNode* policy, const State& state, int stage_id,
                      std::vector<State>& out) {
  const Stage& stage = state->stages[stage_id];
  // Skip the inlined stage and placeholder stage
  if (stage->compute_at == ComputeAtKind::kInlined || stage->op_type == StageKind::kPlaceholder) {
    out.push_back(state);
    return;
  }
  // Try to fuse and vectorize the space iterators in the inner most tile
  PrimExpr cum_length_prod = Integer(1);
  int num_fusible = 0;
  int n_iters = static_cast<int>(stage->iters.size());
  while (num_fusible < n_iters) {
    int iter_id = n_iters - 1 - num_fusible;
    // Stop if this iterator has been a compute at attach point
    if (state->attach_map->iter_to_attached_stages.count(std::make_pair(stage_id, iter_id))) {
      break;
    }
    const Iterator& it = stage->iters[iter_id];
    // Stop if we meet a reduce iterator or annotated iterator
    if (it->iter_kind == IteratorKind::kReduction || it->annotation != IteratorAnnotation::kNone) {
      break;
    }
    // Stop if the memory access is not continuous (vectorizable)
    // Note: The check is too hard, so we use heuristic here
    if (IsTiled(stage) && num_fusible != 0) {
      // If the stage is tiled, then the memory access must not be continuous
      // for the innermost two iterators
      break;
    }
    cum_length_prod *= it->range->extent;
    // if (cum_length_prod > GetIntParam(policy->params, SketchParamKey::max_vectorize_size)) {
    //   break;
    // }
    num_fusible++;
  }

  if (num_fusible == 0) {
    out.push_back(state);
    return;
  }
  State fuse1 = state;
  fuse1.vectorize(stage_id, stage->iters.back());
  out.push_back(fuse1);
  for (int fuse_i = 2; fuse_i <= num_fusible; ++fuse_i) {
    State fuse_n = state;
    Array<Iterator> to_fuse(stage->iters.end() + (-fuse_i), stage->iters.end());
    fuse_n.vectorize(stage_id, fuse_n.fuse(stage_id, to_fuse));
    out.push_back(fuse_n);
  }
}

std::vector<State> InitVectorization::Apply(const SketchPolicyNode* policy, State state) const {
  std::vector<State> cur_states{state}, next_states{};  // All states at current 0
  size_t n_stages = state->stages.size();
  for (size_t stage_id = 0; stage_id < n_stages; ++stage_id) {
    for (auto& state : cur_states) {
      auto& stages = state->stages;
      ICHECK(stages.size() == n_stages);
      const Stage& stage = stages[stage_id];
      VecGetNextStates(policy, state, stage_id, next_states);
    }
    cur_states.swap(next_states);
    next_states.clear();
  }
  return cur_states;
}

std::vector<State> InitThreadBind::Apply(const SketchPolicyNode* policy, State state) const {
  // Collect all stages that are roots of stages that perform multi-level tiling.
  auto& task = policy->search_task;
  std::set<int> multi_level_tiling_root_set;
  for (size_t stage_id = 0; stage_id < state->stages.size(); ++stage_id) {
    if (NeedsMultilevelTiling(task, state, stage_id)) {
      const Stage& stage = state->stages[stage_id];
      if (stage->compute_at == ComputeAtKind::kInlined) {
        continue;
      } else if (stage->compute_at != ComputeAtKind::kIter) {
        // This stage is not multi-level tiled,
        // so it must be produced by RuleCrossThreadReduction.
        ICHECK(HasCrossThreadReduction(state, stage_id));
      } else {
        const auto res = state->attach_map->stage_to_attach_iter.find(stage_id);
        ICHECK(res != state->attach_map->stage_to_attach_iter.end());
        multi_level_tiling_root_set.insert(res->second.first);
      }
    }
  }

  state = task->compute_dag.InferBound(state);
  std::vector<std::pair<size_t, String>> thread_bound_iters, vectorized_iters;
  for (int stage_id = state->stages.size() - 1; stage_id >= 0; --stage_id) {
    const Stage& stage = state->stages[stage_id];

    if (stage->compute_at == ComputeAtKind::kInlined || stage->op_type == StageKind::kPlaceholder) {
      continue;
    }

    // Deal with the cross-thread reduction generated by RuleCrossThreadReduction
    if (HasCrossThreadReduction(state, stage_id)) {
      if (stage->compute_at != ComputeAtKind::kRoot) {
        continue;
      }

      Iterator fused_it;
      state = std::move(FuseAllOuterSpaceIterators(state, stage_id, &fused_it));
      state.bind(stage_id, fused_it, IteratorAnnotation::kBlockX);
      continue;
    }

    // Skip if this stage has already been annotaed with threadIdx.x
    if (HasAnnotatedIter(stage, IteratorAnnotation::kThreadX)) {
      continue;
    }

    if (stage->compute_at == ComputeAtKind::kRoot) {
      // This stage has not been tiled, but in GPU schedule, we must tile the root stage
      // to do thread binding
      if (!multi_level_tiling_root_set.count(stage_id)) {
        Iterator fused_it;
        state = FuseAllOuterSpaceIterators(state, stage_id, &fused_it);
        auto size_var = state.GetVarContext().GetSplitVars(fused_it->range->extent, 1, false)[0];
        const auto& split_its = state.split(stage_id, fused_it, {size_var});
        state.bind(stage_id, split_its[0], IteratorAnnotation::kBlockX);
        state.bind(stage_id, split_its[1], IteratorAnnotation::kThreadX);
        thread_bound_iters.emplace_back(stage_id, split_its[1]->name);
        continue;
      }

      // Otherwise, this is a tiled root stage, we assume it should be tiled with 3 space level
      // in the outer iterators.
      // The remaining part deals with the thread binding for multi-level tiled stages
      auto pop = stage->op.as<te::ComputeOpNode>();
      std::vector<Iterator> to_fuse;

      // Fuse the outermost space tile as blockIdx
      for (size_t i = 0; i < pop->axis.size(); i++) {
        const auto& it = state->stages[stage_id]->iters[i];
        // There may be some iterators that are marked with no split, stop if reaches next
        // tiling level
        if (!StrEndsWith(it->name, ".0")) {
          break;
        }
        to_fuse.push_back(it);
      }
      const auto& blockidx_it = state.fuse(stage_id, to_fuse);
      state.bind(stage_id, blockidx_it, IteratorAnnotation::kBlockX);

      // Fuse the second outermost space tile as vthread
      to_fuse.clear();
      for (size_t i = 1; i < pop->axis.size() + 1; i++) {
        const auto& it = state->stages[stage_id]->iters[i];
        // There may be some iterators that are marked with no split, stop if reaches next
        // tiling level
        if (!StrEndsWith(it->name, ".1")) {
          break;
        }
        to_fuse.push_back(state->stages[stage_id]->iters[i]);
      }
      const auto& vthread_it = state.fuse(stage_id, to_fuse);
      state.bind(stage_id, vthread_it, IteratorAnnotation::kVThread);

      // Fuse the third outermost space tile as threadIdx
      to_fuse.clear();
      for (size_t i = 2; i < pop->axis.size() + 2; i++) {
        const auto& it = state->stages[stage_id]->iters[i];
        // There may be some iterators that are marked with no split, stop if reaches next
        // tiling level
        if (!StrEndsWith(it->name, ".2")) {
          break;
        }
        to_fuse.push_back(state->stages[stage_id]->iters[i]);
      }
      const auto& threadidx_it = state.fuse(stage_id, to_fuse);
      state.bind(stage_id, threadidx_it, IteratorAnnotation::kThreadX);
      thread_bound_iters.emplace_back(stage_id, threadidx_it->name);
    } else if (stage->compute_at == ComputeAtKind::kIter &&
               StrEndsWith(stage->op->name, ".shared")) {
      // Do cooperative fetching for the cache read stage.
      // Get spatial_split_step_ids from the root stage
      const auto& it = state->attach_map->stage_to_attach_iter.find(stage_id);
      ICHECK(it != state->attach_map->stage_to_attach_iter.end());
      Array<Integer> spatial_split_step_ids = GetSpatialSplitStepIds(state, it->second.first);

      // Fuse all iterators to do cooperative fetching
      Iterator fused = state.fuse(stage_id, state->stages[stage_id]->iters);
      // Split out an extra iterator for vectorization
      auto size_var = state.GetVarContext().GetSplitVars(fused->range->extent, 1, false)[0];
      const auto& iters0 = state.split(stage_id, fused, {size_var});
      state.vectorize(stage_id, iters0[1]);
      vectorized_iters.emplace_back(stage_id, iters0[1]->name);
      // Follow split to keep a same thread extent with the root stage
      const auto& iters1 =
          state.follow_fused_split(stage_id, iters0[0], spatial_split_step_ids, 1, true);
      state.bind(stage_id, iters1[1], IteratorAnnotation::kThreadX);
      // This is followsplit, which has the same extent as the root stage.
      // No need to put this in thread_bound_iters.
    }
  }
  return {state};
}

void PrintSketch(const State& sketch_) {
  auto& info = LOG_INFO;
  info << "Initial sketch " << sketch_ << "\n\n";
  info << "Transform steps: \n";
  for (auto& step : sketch_->transform_steps) {
    info << "  " << PrintTrStep(step) << "\n";
  }
  info << "\n";
}

SketchPolicyNode PatchSketchPolicySketchRules(const SketchPolicyNode& node) {
  auto ret = node;
  ret.sketch_rules.clear();
  if (IsCPUTask(node.search_task)) {
    // Sketch Generation Rules
    ret.sketch_rules.push_back(&rule_always_inline);
    ret.sketch_rules.push_back(&rule_simplify_compute_with_const_tensor);
    ret.sketch_rules.push_back(&rule_add_rfactor);
    ret.sketch_rules.push_back(&rule_add_cache_write_stage);
    ret.sketch_rules.push_back(&rule_multi_level_tiling_with_fusion);
    ret.sketch_rules.push_back(&rule_multi_level_tiling);
    ret.sketch_rules.push_back(&rule_skip_stage);
  } else if (IsGPUTask(node.search_task)) {
    // Sketch Generation Rules
    if (node.search_task->target->GetAttr<String>("device", "") == "mali") {
      ret.sketch_rules.push_back(&rule_always_inline);
      ret.sketch_rules.push_back(&rule_simplify_compute_with_const_tensor);
      ret.sketch_rules.push_back(&rule_add_rfactor);
      ret.sketch_rules.push_back(&rule_add_cache_write_stage);
      ret.sketch_rules.push_back(&rule_multi_level_tiling_with_fusion);
      ret.sketch_rules.push_back(&rule_multi_level_tiling);
      ret.sketch_rules.push_back(&rule_skip_stage);
    } else {
      ret.sketch_rules.push_back(&rule_add_cache_read_stage);
      ret.sketch_rules.push_back(&rule_special_compute_location_gpu);
      ret.sketch_rules.push_back(&rule_always_inline);
      ret.sketch_rules.push_back(&rule_simplify_compute_with_const_tensor);
      ret.sketch_rules.push_back(&rule_felix_cross_thread_reduction);
      ret.sketch_rules.push_back(&rule_add_cache_write_stage);
      ret.sketch_rules.push_back(&rule_multi_level_tiling_with_fusion);
      ret.sketch_rules.push_back(&rule_multi_level_tiling);
      ret.sketch_rules.push_back(&rule_skip_stage);
    }
  }
  return ret;
}

Array<State> GenerateAllSymSketches(const SketchPolicy& sketch_policy) {
  // 1. Patch sketch policy with our rules, and we can reuse their code for initial sketch
  // generation.
  auto sp_node = PatchSketchPolicySketchRules(*sketch_policy.operator->());
  Array<State> sketches_ = sp_node.GenerateSketches();
  std::vector<State> sketches(sketches_.begin(), sketches_.end());

  std::vector<ExtendSketchRule*> ext_sketch_rules;
  auto& task = sp_node.search_task;
  if (IsCPUTask(task)) {
    ext_sketch_rules.push_back(&init_fill_tile_size);
    ext_sketch_rules.push_back(&init_change_compute_location);
    ext_sketch_rules.push_back(&init_parallel);
    ext_sketch_rules.push_back(&init_unroll);
    ext_sketch_rules.push_back(&init_vectorization);
  } else if (IsGPUTask(task)) {
    ext_sketch_rules.push_back(&init_fill_tile_size);
    ext_sketch_rules.push_back(&init_thread_bind);
    ext_sketch_rules.push_back(&init_unroll);
  } else {
    LOG_FATAL << "Unsupported target " << task->target;
  }

  // Every sketch must try each rule once.
  std::vector<State> next_sketches{};
  // std::cout << "!/ Symbolic config generation from sketch: \n";
  // int i = 0;
  // PrintSketch(sketch);
  for (const auto* rule : ext_sketch_rules) {
    // std::cout << "Applying rule " << i << "\n";
    for (const auto& sketch : sketches) {
      auto ret = rule->Apply(&sp_node, sketch);
      next_sketches.insert(next_sketches.end(), ret.begin(), ret.end());
    }
    sketches = std::move(next_sketches);
    next_sketches = {};
    // std::cout << "Now has " << sketches.size() << " sketches\n";
    // ++i;
    // std::cout << "\n\n";
  }

  return Array<State>(sketches.begin(), sketches.end());
}

TVM_REGISTER_GLOBAL("auto_scheduler.GenerateAllSymSketches").set_body_typed(GenerateAllSymSketches);

}  // namespace felix
}  // namespace tvm
