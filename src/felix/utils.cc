#include "utils.h"

namespace tvm {
namespace felix {

using namespace tvm::auto_scheduler;

String PrintTrStep(const Step& step) {
  std::ostringstream os;
  if (auto ps = step.as<AnnotationStepNode>()) {
    os << "Annotation(stage_id=" << ps->stage_id << ", loop=" << ps->iter_id << ", annotation=\""
       << IteratorAnnotationString[static_cast<int>(ps->annotation)] << "\")";
  } else if (auto ps = step.as<FuseStepNode>()) {
    os << "Fuse(stage_id=" << ps->stage_id << ", fused_ids=" << ps->fused_ids << ")";
  } else if (auto ps = step.as<PragmaStepNode>()) {
    os << "Pragma(stage_id=" << ps->stage_id << ", loop=" << ps->iter_id
       << ", pragma=" << ps->pragma_type << ")";
  } else if (auto ps = step.as<ReorderStepNode>()) {
    os << "Reorder(stage_id=" << ps->stage_id << ", order_after=" << ps->after_ids << ")";
  } else if (auto ps = step.as<SplitStepNode>()) {
    os << "Split(stage_id=" << ps->stage_id << ", loop=" << ps->iter_id << ", extent=" << ps->extent
       << ", " << ps->lengths << ", inner_to_outer=" << ps->inner_to_outer << ")";
  } else if (auto ps = step.as<FollowSplitStepNode>()) {
    os << "FollowSplit(stage_id=" << ps->stage_id << ", loop=" << ps->iter_id
       << ", src_step_id=" << ps->src_step_id << ", n_split=" << ps->n_split << ")";
  } else if (auto ps = step.as<FollowFusedSplitStepNode>()) {
    os << "FollowFusedSplit(stage_id=" << ps->stage_id << ", loop=" << ps->iter_id
       << ", src_step_ids=" << ps->src_step_ids << ", level=" << ps->level
       << ", factor_or_nparts=" << ps->factor_or_nparts << ")";
  } else if (auto ps = step.as<StorageAlignStepNode>()) {
    os << "StorageAlign(stage_id=" << ps->stage_id << ", loop=" << ps->iter_id
       << ", factor=" << ps->factor << ", offset=" << ps->offset << ")";
  } else if (auto ps = step.as<ComputeAtStepNode>()) {
    os << "ComputeAt(stage_id=" << ps->stage_id << ", target_stage_id=" << ps->target_stage_id
       << ", loop=" << ps->target_iter_id << ")";
  } else if (auto ps = step.as<ComputeInlineStepNode>()) {
    os << "ComputeInline(stage_id=" << ps->stage_id << ")";
  } else if (auto ps = step.as<ComputeRootStepNode>()) {
    os << "ComputeRoot(stage_id=" << ps->stage_id << ")";
  } else if (auto ps = step.as<CacheReadStepNode>()) {
    os << "CacheRead(stage_id=" << ps->stage_id << ", scope_name=" << ps->scope_name
       << ", reader_stage_ids=" << ps->reader_stage_ids << ")";
  } else if (auto ps = step.as<CacheWriteStepNode>()) {
    os << "CacheWrite(stage_id=" << ps->stage_id << ", scope_name=" << ps->scope_name << ")";
  } else if (auto ps = step.as<RfactorStepNode>()) {
    os << "RFactor(stage_id=" << ps->stage_id << ", from_loop=" << ps->iter_id
       << ", to_loop=" << ps->factor_iter_id << ")";
  } else {
    LOG(FATAL) << "Invalid step: " << step;
  }
  return os.str();
}

}  // namespace felix
}  // namespace tvm