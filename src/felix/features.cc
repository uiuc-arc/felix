/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "features.h"

#include <tvm/arith/analyzer.h>
#include <tvm/tir/analysis.h>

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <vector>

#include "utils.h"

namespace tvm {
namespace felix {

using namespace tvm::tir;
using namespace tvm::arith;

// The number of samples to extract for arithmetic intensity curves
// static const constexpr int ARITH_INTENSITY_CURVE_SAMPLE_N = 10;

// Annotation position encoding
enum class AnnotationPosType : int {
  kPosNone = 0,           // Does not have this kind of annotation
  kPosInnerSpatial = 1,   // The annotated iterator is the innermost spatial iterator
  kPosMiddleSpatial = 2,  // The annotated iterator is a middle spatial iterator
  kPosOuterSpatial = 3,   // The annotated iterator is the outermost spatial iterator
  kPosInnerReduce = 4,    // The annotated iterator is the innermost reduce iterator
  kPosMiddleReduce = 5,   // The annotated iterator is a middle reduce iterator
  kPosOuterReduce = 6,    // The annotated iterator is the outermost reduce iterator
  kPosMixed = 7           // The annotated iterator is a mixed space and reduce iterator
};

// Buffer access type
enum class BufferAccessType : int { kRead = 0, kWrite = 1, kReadWrite = 2, kUnknownRW = 3 };

// Accesses to a buffer
struct BufferAccess {
  // data reuse type
  BufferAccessType acc_type{BufferAccessType::kUnknownRW};
  // Use a two-dimensional array to store multiple multi-dimensional accesses.
  // The innermost vector stores the multi-dimensional indices of one access.
  std::vector<std::vector<PrimExpr>> indices;
};

// Feature for an access of a buffer
struct BufferAccessFeature {
  std::string buffer_name;    // The name of the buffer
  BufferAccessType acc_type;  // The type of the access
  PrimExpr bytes;             // The touched memory in bytes
  PrimExpr unique_bytes;      // The touched unique memory in bytes
  PrimExpr lines;             // The number of touched cache lines
  PrimExpr unique_lines;      // The number touched unique cache lines
  // Types of data reuse
  PrimExpr multi_read_cond, serial_multi_rw_cond, no_reuse_cond;
  PrimExpr reuse_dis_iter;           // The reuse distance in iterator number
  PrimExpr reuse_dis_bytes;          // The reuse distance in total touched bytes
  PrimExpr reuse_ct;                 // The reuse ratio
  PrimExpr bytes_d_reuse_ct;         // bytes / reuse_ct
  PrimExpr unique_bytes_d_reuse_ct;  // unique_bytes / reuse_ct
  PrimExpr lines_d_reuse_ct;         // lines / reuse_ct
  PrimExpr unique_lines_d_reuse_ct;  // unique_lines / reuse_ct
  PrimExpr stride;                   // The stride in access
};

// Feature set of a BufferStore statement
struct FeatureSet {
  // Group 1: Computation related features
  PrimExpr float_mad;               // The number of float MAD (Multiply–add) ops
  PrimExpr float_addsub;            // The number of float add and sub ops
  PrimExpr float_mul;               // The number of float multiply ops
  PrimExpr float_divmod;            // The number of float div and mod ops
  PrimExpr float_cmp;               // The number of float comparison ops
  PrimExpr float_math_func;         // The number of float math func calls
  PrimExpr float_other_func;        // The number of other float func calls
  PrimExpr int_mad;                 // The number of integer MAD (Multiply–add) ops
  PrimExpr int_addsub;              // The number of integer add and sub ops
  PrimExpr int_mul;                 // The number of float multiply ops
  PrimExpr int_divmod;              // The number of float div and mod ops
  PrimExpr int_cmp;                 // The number of float comparison ops
  PrimExpr int_math_func;           // The number of float math func calls
  PrimExpr int_other_func;          // The number of other float func calls
  PrimExpr bool_op;                 // The number of bool ops
  PrimExpr select_op;               // The number of select ops
  PrimExpr vec_num;                 // The number of vectorized iterators
  PrimExpr vec_prod;                // The product of the lengths of vectorized iterators
  PrimExpr vec_len;                 // The length of the innermost vectorized iterator
  AnnotationPosType vec_type;       // The type of vectorization position
  PrimExpr unroll_num;              // The number of unrolled iterators
  PrimExpr unroll_prod;             // The product of the lengths of vectorized iterators
  PrimExpr unroll_len;              // The length of the innermost unrolled iterator
  AnnotationPosType unroll_type;    // The type of unroll position
  PrimExpr parallel_num;            // The number of paralleled iterators
  PrimExpr parallel_prod;           // The product of the lengths of paralleled iterators
  PrimExpr parallel_len;            // The length of the innermost paralleled iterators
  AnnotationPosType parallel_type;  // The type of parallel position
  PrimExpr is_gpu;                  // Whether it is a GPU task
  PrimExpr blockIdx_x_len;          // The length of blockIdx.x
  PrimExpr blockIdx_y_len;          // The length of blockIdx.y
  PrimExpr blockIdx_z_len;          // The length of blockIdx.z
  PrimExpr threadIdx_x_len;         // The length of threadIdx.x
  PrimExpr threadIdx_y_len;         // The length of threadIdx.y
  PrimExpr threadIdx_z_len;         // The length of threadIdx.z
  PrimExpr vthread_len;             // The length of virtual thread

  // Group 2: Buffer access related features (per buffer)
  std::vector<BufferAccessFeature> access_feas;

  // Group 3: Arithmetic intensity related features
  // PrimExpr arith_intensity_curve[ARITH_INTENSITY_CURVE_SAMPLE_N];  // points sampled from the
  //                                                                  // arithmetic intensity curve

  // Group 4: Allocation related features
  PrimExpr alloc_size;        // The size of allocated buffer in bytes
  PrimExpr alloc_outer_prod;  // The product of lengths of loops outside the scope of the allocation
  PrimExpr alloc_inner_prod;  // The product of lengths of loops inside the score of the allocation
  PrimExpr alloc_prod;        // alloc_outer_prod * alloc_inner_prod

  // Group 5: Outer scope related features
  PrimExpr outer_prod;            // The product of lengths of outer loops
  PrimExpr num_loops;             // The number of outer loops
  PrimExpr auto_unroll_max_step;  // The value of pragma "auto_unroll_max_step"
};

namespace {

// Return whether a var is in an expr
bool VarInExpr(const Var& var, const PrimExpr& expr) {
  bool found = false;

  // Find by name, because TVM duplicates some loops such as threadIdx.x,
  // creating 2 loop variables that have the same name but are different as objects.
  PostOrderVisit(expr, [&found, &var](const ObjectRef& node) {
    const VarNode* op = node.as<VarNode>();
    if (op && op->name_hint == var->name_hint) {
      found = true;
    }
  });

  return found;
}

PrimExpr SelectNonZero(const PrimExpr& expr, PrimExpr non_zero) {
  auto as_select = expr.as<SelectNode>();
  if (as_select) {
    auto false_value = as_select->false_value.as<IntImmNode>();
    if (false_value && false_value->value == 0) {
      return select(as_select->condition, CastToFloat(as_select->true_value), non_zero);
    }
  }
  return select(expr == 0, non_zero, CastToFloat(expr));
}

PrimExpr SelectLogOr0(PrimExpr cond, PrimExpr value) { return select(cond, log(value), 0); }

// Count math ops in an expr
class MathOpCounter : public ExprVisitor {
 public:
  MathOpCounter(RangeInfer& rinf) : rinf(rinf) {}

  PrimExpr FromExprMap(const ExprMap<size_t>& expr_map) const {
    PrimExpr result = 0;
    for (auto& [cond, count] : expr_map) {
      result += select(cond, 0, Integer(count));
    }
    return result;
  }

 private:
  PrimExpr ConstCond(Range lhs, Range rhs, Integer const_goal) {
    bool lhs_const = this->ana.CanProveEqual(lhs->extent, 0),
         rhs_const = this->ana.CanProveEqual(rhs->extent, 0);
    if (lhs_const && rhs_const) {
      return const_true();
    } else if (lhs_const && const_goal.defined()) {
      return lhs->min == const_goal;
    } else if (rhs_const && const_goal.defined()) {
      return rhs->min == const_goal;
    } else {
      return const_false();
    }
  }

#define DefineVisitBinOp(Type, float_ct, int_ct, const_goal)  \
  void VisitExpr_(const Type* op) override {                  \
    if (op->a.dtype().is_float()) {                           \
      float_ct++;                                             \
    } else {                                                  \
      Range lhs = this->rinf(op->a), rhs = this->rinf(op->b); \
      PrimExpr const_cond = ConstCond(lhs, rhs, const_goal);  \
      int_ct[const_cond]++;                                   \
    }                                                         \
    ExprVisitor::VisitExpr_(op);                              \
  }
  DefineVisitBinOp(AddNode, float_addsub, int_addsub, 0);
  DefineVisitBinOp(SubNode, float_addsub, int_addsub, 0);
  DefineVisitBinOp(MulNode, float_mul, int_mul, 1);
  DefineVisitBinOp(DivNode, float_divmod, int_divmod, 1);
  DefineVisitBinOp(FloorDivNode, float_divmod, int_divmod, 1);
  DefineVisitBinOp(ModNode, float_divmod, int_divmod, 1);
  DefineVisitBinOp(FloorModNode, float_divmod, int_divmod, 1);
  DefineVisitBinOp(MaxNode, float_cmp, int_cmp, Integer());
  DefineVisitBinOp(MinNode, float_cmp, int_cmp, Integer());

#define BoolOp(Type)                         \
  void VisitExpr_(const Type* op) override { \
    bool_op++;                               \
    ExprVisitor::VisitExpr_(op);             \
  }
  BoolOp(AndNode);
  BoolOp(OrNode);
  BoolOp(NotNode);
#define NumToBoolCmpOp(Type)                 \
  void VisitExpr_(const Type* op) override { \
    if (op->a.dtype().is_float()) {          \
      float_cmp++;                           \
    } else {                                 \
      int_cmp[const_false()]++;              \
    }                                        \
    ExprVisitor::VisitExpr_(op);             \
  }
  NumToBoolCmpOp(EQNode);
  NumToBoolCmpOp(NENode);
  NumToBoolCmpOp(LTNode);
  NumToBoolCmpOp(LENode);
  NumToBoolCmpOp(GTNode);
  NumToBoolCmpOp(GENode);

#undef DefineVisitBinOp
#undef BoolOp
#undef NumToBoolCmpOp

  void VisitExpr_(const SelectNode* op) override {
    select_op++;
    ExprVisitor::VisitExpr_(op);
  }

  // Returning empty range as we have no idea what the range could be.
  void VisitExpr_(const CallNode* op) override {
    auto* pop = op->op.as<OpNode>();
    ICHECK(pop != nullptr);
    auto effect_kind = op_call_effect_[GetRef<Op>(pop)];
    bool is_pure =
        effect_kind == CallEffectKind::kPure || effect_kind == CallEffectKind::kExprAnnotation;

    if (is_pure) {
      if (op->dtype.is_float()) {
        float_math_func++;
      } else {
        int_math_func++;
      }
    } else {
      if (op->dtype.is_float()) {
        float_other_func++;
      } else {
        int_other_func++;
      }
    }
    ExprVisitor::VisitExpr_(op);
  }

  RangeInfer& rinf;
  Analyzer ana;
  OpAttrMap<TCallEffectKind> op_call_effect_ = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");

 public:
  size_t float_mad{0};         // The number of float MAD (Multiply–add) ops
  size_t float_addsub{0};      // The number of float add and sub ops
  size_t float_mul{0};         // The number of float multiply ops
  size_t float_divmod{0};      // The number of float div and mod ops
  size_t float_cmp{0};         // The number of float comparison ops
  size_t float_math_func{0};   // The number of float math func calls
  size_t float_other_func{0};  // The number of other float func calls
  size_t int_mad{0};           // The number of integer MAD (Multiply–add) ops
  ExprMap<size_t> int_addsub;  // The number of integer add and sub ops
  ExprMap<size_t> int_mul;     // The number of integer multiply ops
  ExprMap<size_t> int_divmod;  // The number of integer div and mod ops
  ExprMap<size_t> int_cmp;     // The number of integer comparison ops
  size_t int_math_func{0};     // The number of float math func calls
  size_t int_other_func{0};    // The number of other float func calls
  size_t bool_op{0};           // The number of bool ops
  size_t select_op{0};         // The number of select ops
};

// Extract all buffer accesses in an expr
class BufferAccessExtractor : public StmtExprVisitor {
 public:
  void ExtractReads(const PrimExpr& expr) { this->VisitExpr(expr); }

  void InsertAccess(const Buffer& buf, BufferAccessType acc_type, const Array<PrimExpr>& indices) {
    BufferAccess& acc = buf_accesses[buf];
    acc.acc_type = acc_type;
    acc.indices.push_back(std::vector<PrimExpr>(indices.begin(), indices.end()));
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    BufferAccess& acc = buf_accesses[op->buffer];
    switch (acc.acc_type) {
      case BufferAccessType::kRead:
        break;
      case BufferAccessType::kWrite:
        acc.acc_type = BufferAccessType::kReadWrite;
        break;
      case BufferAccessType::kReadWrite:
        break;
      case BufferAccessType::kUnknownRW:
      default:
        acc.acc_type = BufferAccessType::kRead;
        break;
    }

    if (acc.acc_type != BufferAccessType::kReadWrite) {
      // If a buffer is both read and written, in the tvm DSL, it must be a update,
      // so the indices should be the same. Then we can skip appending indices for it.
      // Otherwise we do the following.
      buf_accesses[op->buffer].indices.push_back(
          std::vector<PrimExpr>(op->indices.begin(), op->indices.end()));
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  BufferMap<BufferAccess> buf_accesses;
};

// Compute the coefficient for an loop iterator in an expression
// Note: we use an approximation strategy to find coefficient.
// Hopefully, it is faster than DetectLinearEquation and can handle more cases (non-linear)
class CoefficientExtractor : public StmtExprVisitor {
 public:
  void VisitExpr_(const MulNode* node) final {
    StmtExprVisitor::VisitExpr_(node);
    if (visited_var) {
      if (!visited_add) {
        if (auto a = node->a.as<IntImmNode>()) {
          visited_mul = true;
          stride = a->value;
        } else if (auto b = node->b.as<IntImmNode>()) {
          visited_mul = true;
          stride = b->value;
        }
      }
    }
  }

  void VisitExpr_(const AddNode* node) final {
    StmtExprVisitor::VisitExpr_(node);
    if (visited_var) {
      if (!visited_mul) {
        visited_add = true;
        stride = 1;
      }
    }
  }

  void VisitExpr_(const VarNode* node) final {
    if (node == var_) {
      visited_var = true;
      // This is a magic default stride in case our approximation strategy fails
      stride = 2;
    }
  }

  int ExtractCoefficient(const PrimExpr& expr, const VarNode* var) {
    visited_var = visited_mul = visited_add = false;
    var_ = var;

    this->VisitExpr(expr);

    if (visited_var && !visited_mul && !visited_add) {
      return 1;
    } else {
      return stride;
    }
  }

  bool visited_var{false};
  bool visited_mul{false};
  bool visited_add{false};
  int stride{0};

 private:
  const VarNode* var_{nullptr};
};

// Compute stride for the accesses to a buffer
std::pair<bool, PrimExpr> ComputeStride(const std::vector<std::vector<PrimExpr>>& indices,
                                        const Array<PrimExpr>& shape, const VarNode* stride_var) {
  PrimExpr min_stride{};
  bool found = false;
  CoefficientExtractor extractor;

  for (const auto& index : indices) {
    PrimExpr shape_stride = 1;
    for (int i = static_cast<int>(index.size()) - 1; i >= 0; i--) {
      int coefficient = extractor.ExtractCoefficient(index[i], stride_var);
      if (extractor.visited_var) {
        found = true;
        if (min_stride.defined()) {
          min_stride = min(min_stride, std::abs(coefficient) * shape_stride);
        } else {
          min_stride = std::abs(coefficient) * shape_stride;
        }
        break;
      }
      shape_stride *= shape[i];
    }
  }
  return {found, min_stride};
}

PrimExpr LoopNonTrivialCond(const ForNode* loop) {
  std::string name = loop->loop_var->name_hint;
  if (name.substr(0, 8) == "blockIdx" || name.substr(0, 9) == "threadIdx" || name == "vthread") {
    // These loops are always there no matter what the loop size is.
    return Bool(true);
  }
  return loop->extent > 1;
}

std::tuple<PrimExpr, PrimExpr, PrimExpr> ComputeStrideForLoops(
    const std::vector<std::vector<PrimExpr>>& indices, const Array<PrimExpr>& shape,
    const std::vector<const ForNode*> loops_reversed) {
  PrimExpr reduce_ratio_acc = 1;
  PrimExpr reduce_ratio = 0, stride = 0, innermost_stride = 0;
  PrimExpr found = Bool(false), in_loop = Bool(true), first_loop = Bool(true);
  for (const auto& loop : loops_reversed) {
    PrimExpr non_trivial_loop = LoopNonTrivialCond(loop);
    // If loop is trivial, then the following don't happen and we effectively have a `continue;`.
    auto [found_, stride_] = ComputeStride(indices, shape, loop->loop_var.get());
    PrimExpr found_this = in_loop && non_trivial_loop && Bool(found_);
    reduce_ratio_acc *= loop->extent;
    if (found_) {
      // innermost_stride is non-zero only when the stride is found from the innermost loop.
      innermost_stride += SelectLogOr0(found_this && first_loop, stride_);
      stride += SelectLogOr0(found_this, stride_);
    }
    reduce_ratio += SelectLogOr0(found_this, reduce_ratio_acc);
    found = found || found_this;
    // Breaks out when we actually find something.
    in_loop = in_loop && (!non_trivial_loop || Bool(!found_));
    first_loop = first_loop && (!non_trivial_loop);
  }
  // Default value. Can also use !found here, but that expression is more complex.
  reduce_ratio += SelectLogOr0(in_loop, reduce_ratio_acc);
  ICHECK(innermost_stride.defined());
  return {exp(stride), exp(innermost_stride), exp(reduce_ratio)};
}

// Compute touched bytes and cache lines for accesses to a buffer
std::vector<PrimExpr> ComputeRegion(const std::vector<std::vector<PrimExpr>>& indices,
                                    RangeInfer& rinf, VarDefStackNode& vdefs) {
  std::vector<PrimExpr> ret;
  if (indices.empty()) return ret;
  if (indices.size() == 1) {
    for (const auto& index : indices[0]) {
      Range range = rinf(index);
      ret.push_back(vdefs.DefineConstShorthand(range->extent + 1));
    }
  } else {
    for (const auto& indices_ : indices) {
      Range range = rinf(indices_[0]);
      PrimExpr size = range->extent + 1;
      for (size_t i = 1; i < indices_.size(); ++i) {
        size = max(size, rinf(indices_[i])->extent + 1);
      }
      ret.push_back(vdefs.DefineConstShorthand(size));
    }
  }
  return ret;
}

using BufferInfo3 = std::tuple<BufferAccessType, PrimExpr, int>;
using ForTouchRegionT = std::unordered_map<const ForNode*, BufferMap<std::vector<BufferInfo3>>>;

bool LoopIterInIndices(Var for_var, const std::vector<std::vector<PrimExpr>>& indices) {
  for (size_t j = 0; j < indices.size(); j++) {
    for (size_t k = 0; k < indices[j].size(); k++) {
      if (VarInExpr(for_var, indices[j][k])) {
        return true;
      }
    }
  }
  return false;
}

PrimExpr ReuseDistInBytes(const BufferMap<std::vector<BufferInfo3>>& this_for_region,
                          bool include_n_elems) {
  PrimExpr reuse_dis_bytes = 0;
  for (const auto& iter : this_for_region) {
    for (auto& [_, n_elem, dtype_bytes] : iter.second) {
      if (include_n_elems) {
        reuse_dis_bytes += n_elem * dtype_bytes;
      } else {
        reuse_dis_bytes += dtype_bytes;
      }
    }
  }
  return reuse_dis_bytes;
}

PrimExpr MultiRWReuseDistance(const std::vector<BufferInfo3>& buffers, PrimExpr for_extent) {
  ICHECK(!buffers.empty());
  PrimExpr reuse_dis_iter = std::get<1>(buffers[0]);
  for (size_t i = 1; i < buffers.size(); ++i) {
    reuse_dis_iter = min(reuse_dis_iter, std::get<1>(buffers[i]));
  }
  return div(reuse_dis_iter, for_extent);
}

// Compute reuse distance and reuse ratio for accesses to a buffer
// return values: reuse_type, reuse_dis_iter, reuse_dis_bytes, reuse_ct
std::tuple<PrimExpr, PrimExpr, PrimExpr, PrimExpr, PrimExpr> ComputeReuse(
    const Buffer& buf, const std::vector<std::vector<PrimExpr>>& indices,
    const std::vector<const ForNode*>& for_loops, const ForTouchRegionT& for_touch_regions) {
  // for (i = 0; i < N; i++) {
  //   if (trivial_loop[i]) continue;
  //   if (has_loop_iter[i]) { ... }
  //   else return x_i;                     // kLoopMultipleRead
  //   if (has_serial_uses[i]) return y_i;  // kSerialMultipleReadWrite
  // }
  // return 0;                              // NoReuse
  //
  // Denote the condition that we're still in the loop for i-th iteration by `in_loop[i]`. Then
  //   multi_read_reuse[i] := in_loop[i] && !trivial_loop[i] && !has_loop_iter[i]
  //   serial_rw_reuse[i] = in_loop[i] && !trivial_loop[i] && has_serial_uses[i]
  // Also
  //   in_loop[i + 1] := in_loop[i] && (!multi_read_reuse[i] && !serial_rw_reuse[i])
  //     in_loop[0] = true
  // The return value R from this function is then
  //   \sum_i select(multi_read_reuse[i], x_i, 0) + select(serial_rw_reuse[i], y_i, 0)
  // This function has multiple return values, they all follow this same idea.
  int n_loops = static_cast<int>(for_loops.size());
  PrimExpr in_loop = Bool(true), multi_read_reuse = Bool(false), serial_rw_reuse = Bool(false);
  PrimExpr read_reuse_dist_iter = 1;
  PrimExpr reuse_dist_iter = 0, reuse_dist_bytes = 0, reuse_count = 0;
  for (int i = n_loops - 1; i >= 0; --i) {
    auto* loop = for_loops[i];
    PrimExpr extent = loop->extent;
    const auto& this_for_region = for_touch_regions.at(loop);
    const auto& this_buffer = this_for_region.at(buf);
    int serial_reuse = (int)this_buffer.size() - 1;
    PrimExpr has_loop_iter = Bool(LoopIterInIndices(loop->loop_var, indices));
    // Use extent > 1 here (instead of LoopNonTrivialCond) to skip _all_ loops with extent 1
    // including threadIdx/blockIdx, because that's what the concrete version does
    // (and it makes more sense because if extent is 1 then there won't really be a "reuse").
    PrimExpr non_trivial_loop = extent > 1, has_serial_uses = Bool(serial_reuse > 0);
    PrimExpr multi_read_reuse_ = in_loop && non_trivial_loop && !has_loop_iter,
             serial_rw_reuse_ = in_loop && non_trivial_loop && has_serial_uses;
    PrimExpr no_exit = !non_trivial_loop || (has_loop_iter && !has_serial_uses);
    in_loop = in_loop && no_exit;

    // accumulate/update reuse distance
    PrimExpr rw_reuse_dist_iter = MultiRWReuseDistance(this_buffer, extent);
    PrimExpr reuse_dist_iter_ = SelectLogOr0(multi_read_reuse_, read_reuse_dist_iter) +
                                SelectLogOr0(serial_rw_reuse_, rw_reuse_dist_iter);
    read_reuse_dist_iter *= extent;  // This after reuse_dist_iter_
    // For multi read, reuse_dist_bytes is computed based on the previous (1-level inner) loop.
    // When this is the innermost loop, it's computed from this loop with a slightly different
    // algorithm (`n_elems` is not counted).
    PrimExpr read_reuse_dist_bytes_ =
        i == n_loops - 1 ? ReuseDistInBytes(this_for_region, false)
                         : ReuseDistInBytes(for_touch_regions.at(for_loops[i + 1]), true);
    PrimExpr rw_reuse_dist_bytes_ = div(ReuseDistInBytes(this_for_region, true), extent);
    PrimExpr reuse_dist_bytes_ = SelectLogOr0(multi_read_reuse_, read_reuse_dist_bytes_) +
                                 SelectLogOr0(serial_rw_reuse_, rw_reuse_dist_bytes_);
    PrimExpr reuse_count_ =
        SelectLogOr0(multi_read_reuse_, extent) + SelectLogOr0(serial_rw_reuse_, serial_reuse);

    multi_read_reuse = multi_read_reuse || multi_read_reuse_;
    serial_rw_reuse = serial_rw_reuse || serial_rw_reuse_;
    reuse_dist_bytes += reuse_dist_bytes_;
    reuse_dist_iter += reuse_dist_iter_;
    reuse_count += reuse_count_;
  }
  return std::make_tuple(multi_read_reuse, serial_rw_reuse, exp(reuse_dist_iter),
                         exp(reuse_dist_bytes), exp(reuse_count));
}

// Extract features for every BufferStore statement
class PerStoreFeatureExtractor : public StmtExprVisitor {
 public:
  explicit PerStoreFeatureExtractor(VarDefStackNode& vdefs, RangeInfer& rinf, int cache_line_size)
      : vdefs(vdefs), rinf(rinf), cache_line_size_(cache_line_size) {}

  void VisitStmt_(const AttrStmtNode* node) final {
    if (node->attr_key == tir::attr::thread_extent || node->attr_key == tir::attr::virtual_thread) {
      const Var& var = node->node.as<IterVarNode>()->var;
      this->is_gpu = true;

      // make a fake for node for blockIdx.x or threadIdx.x
      For fake_for(var, 0, node->value, ForKind::kParallel, node->body);
      auto fake_node = fake_for.as<ForNode>();
      this->for_loop_stack.push_back(fake_node);
      StmtExprVisitor::VisitStmt_(node);
      for_loop_stack.pop_back();
    } else if (node->attr_key == "pragma_auto_unroll_max_step") {
      PrimExpr old_value = cur_auto_unroll_max_step_;
      cur_auto_unroll_max_step_ = node->value;
      StmtExprVisitor::VisitStmt_(node);
      cur_auto_unroll_max_step_ = old_value;
    } else {
      StmtExprVisitor::VisitStmt_(node);
    }
  }

  void VisitStmt_(const ForNode* node) final {
    for_loop_stack.push_back(node);
    StmtExprVisitor::VisitStmt(node->body);
    for_loop_stack.pop_back();
  }

  void VisitStmt_(const BufferStoreNode* node) final {
    PrimExpr loop_prod = LoopProd(FilterForLoops(std::nullopt));

    // Group 1: Computation related features
    MathOpCounter moc(this->rinf);
    moc(node->value);
    ExtractComputationFeature(node, moc, loop_prod);

    // Group 2: Buffer access related features (per buffer)
    std::vector<PrimExpr> mem_bytes_list, compute_ops_list;
    PrimExpr cur_compute_ops;
    ExtractBufferAccessFeature(node, moc, loop_prod, &cur_compute_ops, &compute_ops_list,
                               &mem_bytes_list);

    // Group 3: Arithmetic intensity related features
    // LOG_WARNING << "ExtractArithmeticIntensityFeature is unsupported yet";
    // ExtractArithmeticIntensityFeature(node, cur_compute_ops, compute_ops_list, mem_bytes_list);

    // Group 5: Outer scope related features
    ExtractOuterScopeFeature(node, loop_prod);
  }

  void VisitStmt_(const BufferRealizeNode* node) final {
    StmtExprVisitor::VisitStmt_(node);

    // Group 4: Allocation related features
    ExtractAllocationFeature(node);
  }

  // Extract computation related features (group 1)
  void ExtractComputationFeature(const BufferStoreNode* node, const MathOpCounter& moc,
                                 const PrimExpr& loop_prod) {
    // Computation related features
    FeatureSet& fea = bufstore_feats[node->buffer];

    fea.float_mad = loop_prod * (int)moc.float_mad;
    fea.float_addsub = loop_prod * (int)moc.float_addsub;
    fea.float_mul = loop_prod * (int)moc.float_mul;
    fea.float_divmod = loop_prod * (int)moc.float_divmod;
    fea.float_cmp = loop_prod * (int)moc.float_cmp;
    fea.float_math_func = loop_prod * (int)moc.float_math_func;
    fea.float_other_func = loop_prod * (int)moc.float_other_func;
    fea.int_mad = loop_prod * (int)moc.int_mad;
    fea.int_addsub = loop_prod * moc.FromExprMap(moc.int_addsub);
    fea.int_mul = loop_prod * moc.FromExprMap(moc.int_mul);
    fea.int_divmod = loop_prod * moc.FromExprMap(moc.int_divmod);
    fea.int_math_func = loop_prod * (int)moc.int_math_func;
    fea.int_cmp = loop_prod * moc.FromExprMap(moc.int_cmp);
    fea.int_other_func = loop_prod * (int)moc.int_other_func;
    fea.bool_op = loop_prod * (int)moc.bool_op;
    fea.select_op = loop_prod * (int)moc.select_op;

    FillLoopFeatures(ForKind::kVectorized, fea.vec_num, fea.vec_len, fea.vec_prod, fea.vec_type);
    FillLoopFeatures(ForKind::kUnrolled, fea.unroll_num, fea.unroll_len, fea.unroll_prod,
                     fea.unroll_type);
    FillLoopFeatures(ForKind::kParallel, fea.parallel_num, fea.parallel_len, fea.parallel_prod,
                     fea.parallel_type);

    // GPU threads
    fea.is_gpu = Bool(this->is_gpu);
    Map<String, PrimExpr> loop_sizes;
    for (const auto& loop : this->for_loop_stack) {
      loop_sizes.Set(loop->loop_var->name_hint, loop->extent);
    }
    fea.blockIdx_x_len = loop_sizes.Get("blockIdx.x").value_or(1);
    fea.blockIdx_y_len = loop_sizes.Get("blockIdx.y").value_or(1);
    fea.blockIdx_z_len = loop_sizes.Get("blockIdx.z").value_or(1);
    fea.threadIdx_x_len = loop_sizes.Get("threadIdx.x").value_or(1);
    fea.threadIdx_y_len = loop_sizes.Get("threadIdx.y").value_or(1);
    fea.threadIdx_z_len = loop_sizes.Get("threadIdx.z").value_or(1);
    fea.vthread_len = loop_sizes.Get("vthread").value_or(1);
  }

  // Extract buffer access related features (group 2)
  void ExtractBufferAccessFeature(const BufferStoreNode* node, const MathOpCounter& moc,
                                  PrimExpr loop_prod, PrimExpr* cur_compute_ops,
                                  std::vector<PrimExpr>* compute_ops_list,
                                  std::vector<PrimExpr>* mem_bytes_list) {
    std::vector<BufferAccessFeature>& acc_feas = bufstore_feats[node->buffer].access_feas;
    // We may have multiple bufferstore nodes for the same buffer (e.g., 1 for initializing an
    // array, and 1 for computing it). In that case, delibrately overwrite the previous result.
    acc_feas.clear();

    BufferAccessExtractor buf_extractor;
    buf_extractor.InsertAccess(node->buffer, BufferAccessType::kWrite, node->indices);
    buf_extractor.ExtractReads(node->value);
    auto for_loops = FilterForLoops(std::nullopt), for_loops_rev = for_loops;
    std::reverse(for_loops_rev.begin(), for_loops_rev.end());

    // Compute touched region for all outer loops
    // * Make a copy of our global RangeInfer and override all loop variables to be [min, min]
    RangeInfer rangeinf = this->rinf;
    for (auto* loop : for_loops) {
      // Using [a, b] convension for range, this means [x->min, x->min].
      rangeinf.Bind(loop->loop_var, Range::FromMinExtent(loop->min, 0), true);
    }

    mem_bytes_list->reserve(for_loops.size());
    compute_ops_list->reserve(for_loops.size());

    *cur_compute_ops = (int)(moc.float_mad + moc.float_addsub + moc.float_mul + moc.float_divmod +
                             moc.float_cmp + moc.float_math_func + moc.float_other_func);

    // std::cout << "In BufferStoreNode " << node->buffer->name << "\n";
    std::vector<PrimExpr> tmp_region;
    for (auto* loop : for_loops_rev) {
      rangeinf.BindLoop(loop, true);
      // std::cout << "  in for loop " << loop->loop_var->name_hint << "\n";
      // Note, here we do overwrite.
      // So if there are multiple BufferStoreNode, the last one will overwrite the first few.
      // e.g. The update part in gemm will overwrite the init part.
      BufferMap<std::vector<BufferInfo3>>& buffer_regions_map = for_touch_regions_[loop];
      PrimExpr mem_bytes = 0;
      for (const auto& x : buf_extractor.buf_accesses) {
        const Buffer& t = x.first;
        const BufferAccess& acc = x.second;
        // std::cout << "    in buffer access name " << t->name << "\n";
        tmp_region = ComputeRegion(acc.indices, rangeinf, this->vdefs);
        PrimExpr touched_size = ElementProduct(tmp_region);
        buffer_regions_map[t].emplace_back(acc.acc_type, touched_size, t->dtype.bytes());
        mem_bytes += touched_size * t->dtype.bytes();
      }

      mem_bytes_list->push_back(log2(mem_bytes));
      *cur_compute_ops *= loop->extent;
      compute_ops_list->push_back(log2(*cur_compute_ops));
    }

    //  Buffer access related features (per buffer)
    auto bufmap = buf_extractor.buf_accesses;
    std::vector<std::pair<Buffer, BufferAccess>> buf_accs(bufmap.begin(), bufmap.end());
    for (size_t i = 0; i < buf_accs.size(); ++i) {
      auto [buf, acc] = buf_accs[i];
      Integer ele_bytes = buf->dtype.bytes();
      // calculate bytes
      PrimExpr bytes = loop_prod * ele_bytes, unique_bytes;
      // calculate cache lines
      PrimExpr stride, lines, unique_lines;
      if (for_loops.empty()) {
        unique_bytes = ele_bytes;
        stride = 0;
        lines = 1.0f;
        unique_lines = 1.0f;
      } else {
        unique_bytes = this->vdefs.DefineConstShorthand(
            std::get<1>(for_touch_regions_[for_loops.front()][buf].front()) * ele_bytes);
        auto [stride_, innermost_stride, reduce_ratio] =
            ComputeStrideForLoops(acc.indices, buf->shape, for_loops_rev);
        // convert `stride` back to the stride of the innermost iterator
        stride = innermost_stride;
        auto term1 = min(1.0f, div(CastToFloat(stride_ * ele_bytes), (float)cache_line_size_));
        lines = max(div(CastToFloat(loop_prod), CastToFloat(reduce_ratio)) * term1, 1.0f);

        // Modeled after this:
        // PrimExpr n_continuous = ele_bytes;
        // for (int i = std::min(tmp_region.size() - 1, t->shape.size() - 1); i >= 0; i--) {
        //   if (this->ana_.CanProveEqual(tmp_region[i], t->shape[i])) {
        //     n_continuous *= tmp_region[i];
        //     break;
        //   }
        // }
        PrimExpr n_continuous = 0, in_loop = Bool(true);
        for (int i = std::min(tmp_region.size() - 1, buf->shape.size() - 1); i >= 0; i--) {
          PrimExpr is_equal = tmp_region[i] == buf->shape[i];
          n_continuous += SelectLogOr0(in_loop && is_equal, ele_bytes * tmp_region[i]);
          in_loop = in_loop && (!is_equal);
        }
        // If we've done the whole loop without `is_equal == True`, then the value
        // should just be `ele_bytes`.
        n_continuous += SelectLogOr0(in_loop, ele_bytes);
        unique_lines =
            max(div(CastToFloat(unique_bytes), min(exp(n_continuous), cache_line_size_)), 1.0f);
      }

      auto [multi_read_cond, serial_multi_rw_cond, reuse_dis_iter, reuse_dis_bytes, reuse_ct] =
          ComputeReuse(buf, acc.indices, for_loops, for_touch_regions_);
      multi_read_cond = SimplifyExpr(multi_read_cond);
      serial_multi_rw_cond = SimplifyExpr(serial_multi_rw_cond);
      reuse_dis_iter = SimplifyExpr(reuse_dis_iter);
      reuse_dis_bytes = SimplifyExpr(reuse_dis_bytes);
      reuse_ct = SimplifyExpr(reuse_ct);
      PrimExpr no_reuse_cond = SimplifyExpr(!(serial_multi_rw_cond || multi_read_cond));

      acc_feas.emplace_back();
      BufferAccessFeature& acc_fea = acc_feas.back();

      acc_fea.buffer_name = buf->name;
      acc_fea.acc_type = acc.acc_type;
      acc_fea.stride = stride;
      acc_fea.bytes = bytes;
      acc_fea.unique_bytes = unique_bytes;
      acc_fea.lines = lines;
      acc_fea.unique_lines = unique_lines;
      acc_fea.multi_read_cond = multi_read_cond;
      acc_fea.serial_multi_rw_cond = serial_multi_rw_cond;
      acc_fea.no_reuse_cond = no_reuse_cond;
      acc_fea.reuse_dis_iter = reuse_dis_iter;
      acc_fea.reuse_dis_bytes = reuse_dis_bytes;
      acc_fea.reuse_ct = reuse_ct;
      // no reuse, multiply by a magic number '2'
      PrimExpr coef = SelectNonZero(reuse_ct, 0.5f);
      acc_fea.bytes_d_reuse_ct = bytes / coef;
      acc_fea.unique_bytes_d_reuse_ct = unique_bytes / coef;
      acc_fea.lines_d_reuse_ct = lines / coef;
      acc_fea.unique_lines_d_reuse_ct = unique_lines / coef;
    }
  }

  // Extract allocation related features (group 4)
  void ExtractAllocationFeature(const BufferRealizeNode* node) {
    FeatureSet& fea = bufstore_feats[node->buffer];
    PrimExpr allocation_size = 1;
    for (const auto& x : node->bounds) {
      allocation_size *= this->rinf.GetMax(x->extent);
    }
    // allocation feature
    allocation_size = this->vdefs.DefineConstShorthand(allocation_size);
    auto loop_prod = LoopProd(FilterForLoops(std::nullopt));
    fea.alloc_size = allocation_size * node->buffer->dtype.bytes();
    fea.alloc_prod = allocation_size * loop_prod;
    fea.alloc_outer_prod = loop_prod;
    fea.alloc_inner_prod = div(fea.outer_prod, loop_prod);
  }

  // Extract outer scope related features (group 5)
  void ExtractOuterScopeFeature(const BufferStoreNode* node, const PrimExpr& loop_prod) {
    FeatureSet& fea = bufstore_feats[node->buffer];
    fea.outer_prod = loop_prod;
    fea.num_loops = CountLoops(for_loop_stack);
    fea.auto_unroll_max_step = cur_auto_unroll_max_step_;
  }

  void FillLoopFeatures(ForKind kind, PrimExpr& num, PrimExpr& len, PrimExpr& prod,
                        AnnotationPosType& type) {
    auto loops = FilterForLoops(kind);
    num = CountLoops(loops);
    if (loops.empty()) {
      len = prod = 0;
      type = AnnotationPosType::kPosNone;
    } else {
      len = loops.back()->extent;
      prod = 1;
      for (auto* loop : loops) {
        prod *= loop->extent;
      }
      type = AnnotationPosType::kPosMixed;
    }
  }

  std::vector<const ForNode*> FilterForLoops(std::optional<ForKind> kind) {
    std::unordered_set<std::string> var_registered;
    std::vector<const ForNode*> loops;
    for (auto* loop : this->for_loop_stack) {
      if (kind && kind.value() != loop->kind) {
        continue;
      }
      std::string var_name = loop->loop_var->name_hint;
      if (var_registered.count(var_name)) {
        ICHECK(var_name.substr(0, 8) == "blockIdx" || var_name.substr(0, 9) == "threadIdx")
            << "Duplicate non-gpu-grid loop var: " << var_name;
        continue;
      }
      loops.push_back(loop);
    }
    return loops;
  }

  PrimExpr CountLoops(const std::vector<const ForNode*>& loops) {
    PrimExpr num = 0;
    for (auto* loop : loops) {
      num += select(LoopNonTrivialCond(loop), 1, 0);
    }
    return num;
  }

  PrimExpr LoopProd(const std::vector<const ForNode*>& loops) {
    PrimExpr ret = 1.0f;
    for (auto* loop : loops) {
      ret *= loop->extent;
    }
    return ret;
  }

 public:
  BufferMap<FeatureSet> bufstore_feats;

 private:
  // The shared arithmetic analyzers
  VarDefStackNode& vdefs;
  RangeInfer& rinf;
  VarMapT flatmap;

  std::vector<const ForNode*> for_loop_stack;
  PrimExpr previous_outer;
  // GPU-related features
  bool is_gpu;
  PrimExpr cur_auto_unroll_max_step_{0};

  // Store touch region information for all for loops. The format of this nested map:
  // For a loop, for all its touched buffers, for all different accesses to the buffers,
  // its (access type, number of touched elements, number of bytes of single element)
  ForTouchRegionT for_touch_regions_;

  // The default cache line size in bytes
  const int cache_line_size_ = 64;
};

}  // namespace

VarDefStack GetPerStoreFeatureExpr(const Stmt& stmt, VarDefStackNode& vdefs, RangeInfer& rinf,
                                   size_t cache_line_size, size_t max_n_bufs) {
  // Extract features
  PerStoreFeatureExtractor extractor(vdefs, rinf, cache_line_size);
  extractor(stmt);
  std::vector<std::pair<Buffer, FeatureSet>> buffer_features(extractor.bufstore_feats.begin(),
                                                             extractor.bufstore_feats.end());
  std::sort(buffer_features.begin(), buffer_features.end(),
            [](auto& a, auto& b) { return a.first->name < b.first->name; });

  // Define features in context, and put the resulted variable names in ret.
  VarDefStack feats;
  for (size_t i = 0; i < buffer_features.size(); ++i) {
    auto& [buf, fea_set] = buffer_features[i];
    auto PushFeature = [&i, &feats](const std::string& name, const PrimExpr& val,
                                    PrimExpr default_) {
      auto name_ = "BS" + std::to_string(i) + "." + name;
      if (!val.defined()) {
        if (default_.defined()) {
          feats->Append(name_, default_);
        } else {
          LOG_FATAL << "Feature " << name_ << " is not defined";
        }
      } else if (val.dtype().is_bool()) {
        feats->Append(name_, val);
      } else {
        feats->Append(name_, log(val + 1));
      }
    };
    auto PushEnumFeature = [&i, &feats](const std::string& field,
                                        const std::vector<std::string>& kind_names, auto val) {
      for (size_t j = 0; j < kind_names.size(); j++) {
        auto name_ = "BS" + std::to_string(i) + "." + field + "." + kind_names[j];
        feats->Append(name_, Bool(static_cast<size_t>(val) == j));
      }
    };

#define PUSH_FEATURE(feature) PushFeature(#feature, fea_set.feature, PrimExpr());
#define PUSH_ENUM_FEATURE(feature, names) PushEnumFeature(#feature, names, fea_set.feature);
    /***** Group 1: Computation related features *****/
    PUSH_FEATURE(float_mad);
    PUSH_FEATURE(float_addsub);
    PUSH_FEATURE(float_mul);
    PUSH_FEATURE(float_divmod);
    PUSH_FEATURE(float_cmp);
    PUSH_FEATURE(float_math_func);
    PUSH_FEATURE(float_other_func);
    PUSH_FEATURE(int_mad);
    PUSH_FEATURE(int_addsub);
    PUSH_FEATURE(int_mul);
    PUSH_FEATURE(int_divmod);
    PUSH_FEATURE(int_cmp);
    PUSH_FEATURE(int_math_func);
    PUSH_FEATURE(int_other_func);
    PUSH_FEATURE(bool_op);
    PUSH_FEATURE(select_op);

    static const std::vector<std::string> annot_pos_names = {
        "kPosNone",        "kPosInnerSpatial", "kPosMiddleSpatial", "kPosOuterSpatial",
        "kPosInnerReduce", "kPosMiddleReduce", "kPosOuterReduce",   "kPosMixed",
    };
    PUSH_FEATURE(vec_num);
    PUSH_FEATURE(vec_prod);
    PUSH_FEATURE(vec_len);
    PUSH_ENUM_FEATURE(vec_type, annot_pos_names);
    PUSH_FEATURE(unroll_num);
    PUSH_FEATURE(unroll_prod);
    PUSH_FEATURE(unroll_len);
    PUSH_ENUM_FEATURE(unroll_type, annot_pos_names);
    PUSH_FEATURE(parallel_num);
    PUSH_FEATURE(parallel_prod);
    PUSH_FEATURE(parallel_len);
    PUSH_ENUM_FEATURE(parallel_type, annot_pos_names);

    PUSH_FEATURE(is_gpu);
    PUSH_FEATURE(blockIdx_x_len);
    PUSH_FEATURE(blockIdx_y_len);
    PUSH_FEATURE(blockIdx_z_len);
    PUSH_FEATURE(threadIdx_x_len);
    PUSH_FEATURE(threadIdx_y_len);
    PUSH_FEATURE(threadIdx_z_len);
    PUSH_FEATURE(vthread_len);

    /***** Group 2: Buffer access related features *****/
    static const std::vector<std::string> acc_type_names = {"kRead", "kWrite", "kReadWrite"};
    auto& buf_feats = fea_set.access_feas;
    std::sort(buf_feats.begin(), buf_feats.end(),
              [](auto& a, auto& b) { return a.buffer_name < b.buffer_name; });
    buf_feats.resize(max_n_bufs);
    for (size_t j = 0; j < max_n_bufs; ++j) {
      const auto& acc_fea = buf_feats[j];

#define PUSH_BUF_FEATURE(feature) \
  PushFeature("B" + std::to_string(j) + "." + #feature, acc_fea.feature, Integer(0));

      PushEnumFeature("B" + std::to_string(j) + ".acc_type", acc_type_names, acc_fea.acc_type);
      PUSH_BUF_FEATURE(bytes);
      PUSH_BUF_FEATURE(unique_bytes);
      PUSH_BUF_FEATURE(lines);
      PUSH_BUF_FEATURE(unique_lines);
      PUSH_BUF_FEATURE(multi_read_cond);
      PUSH_BUF_FEATURE(serial_multi_rw_cond);
      PUSH_BUF_FEATURE(no_reuse_cond);
      PUSH_BUF_FEATURE(reuse_dis_iter);
      PUSH_BUF_FEATURE(reuse_dis_bytes);
      PUSH_BUF_FEATURE(reuse_ct);
      PUSH_BUF_FEATURE(bytes_d_reuse_ct);
      PUSH_BUF_FEATURE(unique_bytes_d_reuse_ct);
      PUSH_BUF_FEATURE(lines_d_reuse_ct);
      PUSH_BUF_FEATURE(unique_lines_d_reuse_ct);
      PUSH_BUF_FEATURE(stride);
    }

    /***** Group 4: Allocation related features *****/
    PUSH_FEATURE(alloc_size);
    PUSH_FEATURE(alloc_prod);
    PUSH_FEATURE(alloc_outer_prod);
    PUSH_FEATURE(alloc_inner_prod);

    /***** Group 5: Outer scope related features *****/
    PUSH_FEATURE(outer_prod);
    PUSH_FEATURE(num_loops);
    PUSH_FEATURE(auto_unroll_max_step);
  }
  return feats;
}

}  // namespace felix
}  // namespace tvm
