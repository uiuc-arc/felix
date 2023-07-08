#include "utils.h"

#include <tvm/auto_scheduler/search_policy_utils.h>
#include <tvm/auto_scheduler/search_task.h>
#include <tvm/driver/driver_api.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace felix {

using namespace tvm::auto_scheduler;
using namespace tvm::tir::transform;
using namespace tvm::arith;

Sequential GetGPUCodeGenPasses(const HardwareParams& hw_params, bool verify) {
  auto pass_list = Array<Pass>();
  auto pass_ctx = PassContext::Current();
  bool disable_vectorize = pass_ctx->GetConfig<Bool>("tir.disable_vectorize", Bool(false)).value();
  bool instrument_bound_checkers =
      pass_ctx->GetConfig<Bool>("tir.instrument_bound_checkers", Bool(false)).value();
  Map<String, PrimExpr> gpu_params{
      {"max_shared_memory_per_block", hw_params->max_shared_memory_per_block},
      {"max_local_memory_per_block", hw_params->max_local_memory_per_block},
      {"max_threads_per_block", hw_params->max_threads_per_block},
      {"max_vector_bytes", hw_params->vector_unit_bytes},
      {"max_vthread", hw_params->max_vthread_extent},
  };
  pass_list = Array<Pass>({// Phase 0
                           InjectPrefetch(), StorageFlatten(64, instrument_bound_checkers),
                           // Phase 1
                           NarrowDataType(32), Simplify(), VectorizeLoop(!disable_vectorize),
                           InjectVirtualThread(), StorageRewrite(), Simplify()});
  if (verify) {
    pass_list.push_back(VerifyGPUCode(gpu_params));
  }
  return Sequential(pass_list);
}

Stmt GenerateCodeForState(const SearchTask& task, const State& state, bool is_symbolic,
                          bool print_error, VarContextNode* var_context) {
  te::Schedule sch;
  Array<te::Tensor> tensors;
  std::tie(sch, tensors) = task->compute_dag.ApplySteps(state->transform_steps);
  // When inlining, replace const matrices with const values.
  // Produces wrong IR, but good enough for feature extraction, and
  // can improve the speed of feature extraction/search.  Must be
  // called before ScheduleToModule to have an effect.
  sch = sch.normalize_for_feature_extraction();

  try {
    const std::string& name = "main";
    auto mod = ScheduleToModule(sch, Array<ObjectRef>{tensors.begin(), tensors.end()}, name,
                                std::unordered_map<te::Tensor, te::Buffer>(), var_context);
    if (IsGPUTask(task)) {
      auto optimize = GetGPUCodeGenPasses(task->hardware_params, !is_symbolic);
      optimize(mod);
    }
    const auto& optimize =
        tir::transform::Sequential(Array<tvm::transform::Pass>{tir::transform::Simplify()});
    mod = optimize(std::move(mod));
    PrimFunc prim_func = Downcast<PrimFunc>(mod->Lookup(name));
    return prim_func->body;
  } catch (Error& e) {
    if (print_error) LOG_WARNING << "Failed to generate code: " << e.what() << "\n";
    return Stmt();
  }
}

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

TVM_REGISTER_GLOBAL("auto_scheduler.ExtractBackbone").set_body_typed([](const Array<Step>& steps) {
  Array<String> ret;
  for (auto& step : steps) {
    if (step.as<AnnotationStepNode>()) {
      ret.push_back(AnnotationStepNode::record_prefix_str);
    } else if (step.as<FuseStepNode>()) {
      ret.push_back(FuseStepNode::record_prefix_str);
    } else if (step.as<PragmaStepNode>()) {
      ret.push_back(PragmaStepNode::record_prefix_str);
    } else if (step.as<ReorderStepNode>()) {
      ret.push_back(ReorderStepNode::record_prefix_str);
    } else if (step.as<SplitStepNode>()) {
      ret.push_back(SplitStepNode::record_prefix_str);
    } else if (step.as<FollowSplitStepNode>()) {
      ret.push_back(FollowSplitStepNode::record_prefix_str);
    } else if (step.as<FollowFusedSplitStepNode>()) {
      ret.push_back(FollowFusedSplitStepNode::record_prefix_str);
    } else if (step.as<StorageAlignStepNode>()) {
      ret.push_back(StorageAlignStepNode::record_prefix_str);
    } else if (step.as<ComputeAtStepNode>()) {
      ret.push_back(ComputeAtStepNode::record_prefix_str);
    } else if (step.as<ComputeInlineStepNode>()) {
      ret.push_back(ComputeInlineStepNode::record_prefix_str);
    } else if (step.as<ComputeRootStepNode>()) {
      ret.push_back(ComputeRootStepNode::record_prefix_str);
    } else if (step.as<CacheReadStepNode>()) {
      ret.push_back(CacheReadStepNode::record_prefix_str);
    } else if (step.as<CacheWriteStepNode>()) {
      ret.push_back(CacheWriteStepNode::record_prefix_str);
    } else if (step.as<RfactorStepNode>()) {
      ret.push_back(RfactorStepNode::record_prefix_str);
    } else {
      LOG(FATAL) << "Invalid step: " << step;
    }
  }
  return ret;
});

class ForLoopCollector : public StmtExprVisitor {
 public:
  void VisitStmt_(const BufferRealizeNode* node) final {
    this->bufreal_nodes.push_back(node);
    StmtExprVisitor::VisitStmt_(node);
    this->bufreal_nodes.pop_back();
  }

  void VisitStmt_(const ForNode* node) final {
    ICHECK(!this->bufreal_nodes.empty());
    auto* last_buf_node = this->bufreal_nodes.back();
    String name = last_buf_node->buffer->name + "/" + node->loop_var->name_hint;
    this->for_loops.emplace_back(name, SimplifyExpr(node->extent));
    StmtExprVisitor::VisitStmt_(node);
  }

  void VisitStmt_(const AttrStmtNode* node) final {
    if (node->attr_key == tir::attr::thread_extent || node->attr_key == tir::attr::virtual_thread) {
      const Var& var = node->node.as<IterVarNode>()->var;
      this->for_loops.emplace_back(var->name_hint, SimplifyExpr(node->value));
    }
    StmtExprVisitor::VisitStmt_(node);
  }

  std::vector<std::pair<String, PrimExpr>> for_loops;
  std::vector<const BufferRealizeNode*> bufreal_nodes;
};

TVM_REGISTER_GLOBAL("auto_scheduler.PrintTrStep").set_body_typed(PrintTrStep);

TVM_REGISTER_GLOBAL("auto_scheduler.GenerateCodeForState")
    .set_body_typed([](const SearchTask& task, State state, Bool symbolic) {
      bool is_sym = symbolic->value;
      VarContextNode* var_context = is_sym ? &state.GetVarContext() : nullptr;
      auto code = GenerateCodeForState(
          task, state, is_sym, /* print error for symbolic generation */ is_sym, var_context);
      return Array<ObjectRef>{code, GetRef<VarContext>(var_context)};
    });

TVM_REGISTER_GLOBAL("auto_scheduler.GetLoopBounds").set_body_typed([](Stmt stmt) {
  ForLoopCollector collector;
  collector(stmt);
  Array<Array<ObjectRef>> ret;
  for (auto& kv : collector.for_loops) {
    ret.push_back({kv.first, kv.second});
  }
  return ret;
});

TVM_REGISTER_GLOBAL("auto_scheduler.ExtractConfigDict")
    .set_body_typed([](const Array<Step>& steps) {
      size_t split_count = 0;
      Map<String, Integer> config;
      for (auto& step : steps) {
        if (auto* split = step.as<SplitStepNode>()) {
          for (size_t i = 0; i < split->lengths.size(); ++i) {
            // Name must match the one in sketch_policy_rule.cc
            auto var_name = "sp_" + std::to_string(split_count) + "_" + std::to_string(i);
            auto len = split->lengths[i];
            auto* len_int = len.as<IntImmNode>();
            ICHECK(len_int) << "Split length must be a constant integer";
            config.Set(var_name, Integer(len_int->value));
          }
          split_count += 1;
        } else if (auto* pragma = step.as<PragmaStepNode>()) {
          std::string pstring = pragma->pragma_type, pat = "auto_unroll_max_step";
          if (pstring.substr(0, pat.length()) == pat) {
            auto unroll_size = std::stoi(pstring.substr(pat.length() + 1));  // skip "$"
            auto var_name = "ur_" + std::to_string(pragma->stage_id);
            config.Set(var_name, Integer(unroll_size));
          }
        }
      }
      return config;
    });

}  // namespace felix
}  // namespace tvm