#include <tvm/auto_scheduler/search_task.h>
#include <tvm/tir/stmt_functor.h>

#include <numeric>

#include "../tir/transforms/ir_utils.h"
#include "features.h"

namespace tvm {
namespace felix {

using namespace tvm::tir;

class GPUConstraintsMaker : public StmtExprVisitor {
  // TODO(jcf94): Add support of detecting CUDA Misaligned Address error
 public:
  explicit GPUConstraintsMaker(size_t max_local_memory_per_block,
                               size_t max_shared_memory_per_block, size_t max_threads_per_block,
                               size_t max_vthread, size_t max_vector_size, size_t max_vector_bytes)
      : max_local_memory_per_block(max_local_memory_per_block),
        max_shared_memory_per_block(max_shared_memory_per_block),
        max_threads_per_block(max_threads_per_block),
        max_vthread(max_vthread),
        max_vector_size(max_vector_size),
        max_vector_bytes(max_vector_bytes) {}

  void RunOnStmt(Stmt stmt) {
    Reset_();
    this->VisitStmt(stmt);
  }

  void VisitStmt_(const AllocateNode* op) final {
    StmtVisitor::VisitStmt_(op);
    // visit an allocation of a buffer in shared memory, record its size
    auto scope = GetPtrStorageScope(op->buffer_var);
    CountBufferSize_(op->extents, op->dtype, scope);
    if (op->dtype.lanes() > 1) {
      CheckVectorBytes_(op->dtype);
    }
  }

  void VisitStmt_(const BufferRealizeNode* op) final {
    StmtVisitor::VisitStmt_(op);
    auto scope = GetPtrStorageScope(op->buffer->data);
    Array<PrimExpr> extents;
    for (auto& range : op->bounds) {
      extents.push_back(range->extent);
    }
    CountBufferSize_(extents, op->buffer->dtype, scope);
    if (op->buffer->dtype.lanes() > 1) {
      CheckVectorBytes_(op->buffer->dtype);
    }
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    auto attr = op->attr_key;
    bool is_thread = attr == tir::attr::thread_extent;
    bool is_vthread = attr == tir::attr::virtual_thread;
    bool is_unroll = attr == tir::attr::pragma_auto_unroll_max_step;
    if (!is_thread && !is_vthread && !is_unroll) {
      StmtVisitor::VisitStmt_(op);
      return;
    }

    if (this->nest_level == 0) {
      // enter a new kernel, reset statistics
      Reset_();
      kernels_launched++;
    }

    // record the number of threads in a block
    Var var = op->node.as<IterVarNode>()->var;
    std::string name = var.get()->name_hint;
    PrimExpr extent = op->value;
    if (name == "threadIdx.x" || name == "threadIdx.y" || name == "threadIdx.z" ||
        name == "vthread") {
      // record the number of threads in a block
      if (!this->visited_threads.count(name)) {
        this->visited_threads.insert(name);
        this->thread_per_block *= extent;
      }
      // else: the thread should be bound to axes with the same length
      // but we don't check this here, as it can be difficult to
      // compare the equality of two expressions.
    }

    this->nest_level++;
    StmtVisitor::VisitStmt_(op);
    this->nest_level--;

    if (this->nest_level == 0) {
      // exit a kernel, check the validity
      AddConstraint_(this->thread_per_block, this->max_threads_per_block);
      AddConstraint_(this->local_memory_per_block, this->max_local_memory_per_block);
      AddConstraint_(this->shared_memory_per_block, this->max_shared_memory_per_block);
    }
  }

  void VisitStmt_(const ForNode* op) {
    if (op->kind == ForKind::kVectorized) {
      AddConstraint_(op->extent, this->max_vector_size);
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const LoadNode* op) {
    if (op->dtype.lanes() > 1) {
      CheckVectorBytes_(op->dtype);
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const StoreNode* op) {
    if (op->value->dtype.lanes() > 1) {
      CheckVectorBytes_(op->value->dtype);
    }
    StmtVisitor::VisitStmt_(op);
  }

 private:
  size_t max_local_memory_per_block;
  size_t max_shared_memory_per_block;
  size_t max_threads_per_block;
  size_t max_vthread;
  size_t max_vector_size;
  size_t max_vector_bytes;

  std::unordered_set<std::string> visited_threads{};
  PrimExpr local_memory_per_block = 0;
  PrimExpr shared_memory_per_block = 0;
  PrimExpr thread_per_block = 1;
  size_t kernels_launched = 0;
  size_t nest_level = 0;

 public:
  std::vector<PrimExpr> constraints{};
  std::vector<String> errors{};

 private:
  void AddConstraint_(PrimExpr lhs, size_t rhs) {
    auto con = lhs <= Integer(rhs);
    if (auto* simp_bool = con.as<IntImmNode>()) {
      if (simp_bool->value == 0) {
        std::stringstream s;
        s << "Constraint " << lhs << " <= " << rhs << " is trivially False";
        this->errors.push_back(s.str());
      }
    } else {
      this->constraints.push_back(con);
    }
  }

  void CountBufferSize_(const Array<PrimExpr>& extents, DataType dtype, const String& scope) {
    PrimExpr one = Integer(1);
    PrimExpr alloc_count =
        std::accumulate(extents.begin(), extents.end(), one,
                        [](const PrimExpr& a, const PrimExpr& b) { return a * b; });
    PrimExpr alloc_size = alloc_count * dtype.bytes() * dtype.lanes();
    if (scope == "local") {
      this->local_memory_per_block += alloc_size;
    } else if (scope == "shared") {
      this->shared_memory_per_block += alloc_size;
    }
  }

  void CheckVectorBytes_(DataType dtype) {
    if (static_cast<size_t>(dtype.lanes() * dtype.bytes()) > this->max_vector_bytes) {
      std::stringstream s;
      s << "Number of lanes (" << dtype.lanes() << ") times number of bytes (" << dtype.bytes()
        << ") for dtype " << dtype << " is greater than the maximum number of vector bytes ("
        << this->max_vector_bytes << ")";
      this->errors.push_back(s.str());
    }
  }

  void Reset_() {
    this->local_memory_per_block = 0;
    this->shared_memory_per_block = 0;
    this->visited_threads.clear();
  }
};

std::vector<PrimExpr> GetConstraints(const Stmt& code,
                                     const auto_scheduler::HardwareParams& hw_params) {
  // Run GPU verification pass to inject constraints.
  // HACK: size of 4 is based on src/target/source/codegen_cuda.cc.
  //   - line 246: bool vector is only supported when size is less than 4
  //   - line 351: vector of int/uint is only supported when size is less than 8.
  //   While this is true for all CUDA devices and will stay so for a while,
  //   it's not a good idea to hardcode it here.
  GPUConstraintsMaker cmaker(hw_params->max_local_memory_per_block,
                             hw_params->max_shared_memory_per_block,
                             hw_params->max_threads_per_block, hw_params->max_vthread_extent,
                             /*max_vector_size*/ 4, hw_params->vector_unit_bytes);
  cmaker.RunOnStmt(code);
  if (!cmaker.errors.empty()) {
    LOG_WARNING << "Code constraint check failed: ";
    for (auto& err : cmaker.errors) {
      LOG_WARNING << "  " << err;
    }
    LOG_WARNING << "Failed code: " << code;
    return {};
  }
  return cmaker.constraints;
}
}  // namespace felix
}  // namespace tvm