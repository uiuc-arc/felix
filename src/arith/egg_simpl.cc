#include <tvm/arith/egg_simpl.h>
#include <tvm/ir/span.h>
#include <tvm/tir/stmt_functor.h>

using namespace tvm::tir;

// See egg_simpl/src/lib.rs for the Rust implementation of this function.
extern "C" {
char* simplify_expr(const char* str, uint64_t n_iters, uint64_t n_nodes, bool diff_approx);
bool is_equivalent(const char* str1, const char* str2, bool explain, uint64_t n_iters,
                   uint64_t n_nodes, bool diff_approx);
void free_str(char* str);
}

namespace tvm {
namespace arith {

class PreorderPrinter : public ExprFunctor<void(const PrimExpr&)> {
 public:
  std::string Print(const PrimExpr& expr) {
    ss.clear(), ss.str("");
    VisitExpr(expr);
    auto ret = ss.str();
    ss.clear(), ss.str("");
    return ret;
  }

  void VisitExpr_(const SizeVarNode* op) override {
    this->var_map.Set(op->name_hint, GetRef<SizeVar>(op));
    if (op->kind == SizeVarKind::kShapeVar) {
      ss << op->name_hint << ":s";
    } else if (op->kind == SizeVarKind::kScheduleKnob) {
      ss << op->name_hint << ":k";
    } else if (op->kind == SizeVarKind::kShorthand) {
      ss << op->name_hint << ":v";
    } else {
      LOG_FATAL << "Unknown SizeVarKind: " << (int)op->kind;
    }
  }

  void VisitExpr_(const VarNode* op) override {
    this->var_map.Set(op->name_hint, GetRef<Var>(op));
    ss << op->name_hint;
    if (op->dtype.is_bool()) {
      ss << ":b";
    }
  }

  void VisitExpr_(const IntImmNode* op) override {
    if (op->dtype.is_bool()) {
      ss << (op->value ? "true" : "false");
    } else {
      ss << op->value;
    }
  }
  void VisitExpr_(const FloatImmNode* op) override { ss << op->value; }

#define DefineVisitUOp(Type, Op)                   \
  void VisitExpr_(const Type##Node* op) override { \
    ss << "(" << Op << " ";                        \
    VisitExpr(op->a);                              \
    ss << ")";                                     \
  }
#define DefineVisitBinOp(Type, Op)                 \
  void VisitExpr_(const Type##Node* op) override { \
    ss << "(" << Op << " ";                        \
    VisitExpr(op->a);                              \
    ss << " ";                                     \
    VisitExpr(op->b);                              \
    ss << ")";                                     \
  }
  DefineVisitBinOp(Add, "+");
  DefineVisitBinOp(Sub, "-");
  DefineVisitBinOp(Mul, "*");
  DefineVisitBinOp(Div, "/");
  DefineVisitBinOp(FloorDiv, "//");
  // For all our purposes we don't distinguish between Mod and FloorMod
  DefineVisitBinOp(Mod, "mod");
  DefineVisitBinOp(FloorMod, "mod");
  DefineVisitBinOp(Min, "min");
  DefineVisitBinOp(Max, "max");
  DefineVisitBinOp(EQ, "==");
  DefineVisitBinOp(NE, "!=");
  DefineVisitBinOp(LT, "<");
  DefineVisitBinOp(LE, "<=");
  DefineVisitBinOp(GT, ">");
  DefineVisitBinOp(GE, ">=");
  DefineVisitBinOp(And, "&&");
  DefineVisitBinOp(Or, "||");
  DefineVisitUOp(Not, "!");
#undef DefineVisitUOp
#undef DefineVisitBinOp

  void VisitExpr_(const SelectNode* op) override {
    ss << "(select ";
    VisitExpr(op->condition);
    ss << " ";
    VisitExpr(op->true_value);
    ss << " ";
    VisitExpr(op->false_value);
    ss << ")";
  }

  void VisitExpr_(const CallNode* op) override {
    auto* pop = op->op.as<OpNode>();
    ICHECK(pop != nullptr);
    std::string name = pop->name;
    if (name.substr(0, 4) == "tir.") {
      name = name.substr(4);
    }
    ss << "(" << name << " ";
    for (size_t i = 0; i < op->args.size(); ++i) {
      VisitExpr(op->args[i]);
      if (i != op->args.size() - 1) {
        ss << " ";
      }
    }
    ss << ")";
  }

  void VisitExpr_(const CastNode* op) override { VisitExpr(op->value); }

  Map<String, Var> var_map;

 private:
  std::ostringstream ss;
};

inline bool IsFloat32(const PrimExpr& arg) {
  return arg.dtype().is_float() && arg.dtype().bits() == 32;
}
void CheckArgs(std::vector<PrimExpr>& args, size_t N, const std::string& op_name) {
  if (args.size() != N) {
    std::ostringstream ss;
    ss << "Expected " << N << " arguments for " << op_name << ", got " << args.size() << ": ";
    for (const auto& arg : args) {
      ss << arg << ", ";
    }
    throw std::runtime_error(ss.str());
  }
}

PrimExpr MakeUnOp(const std::string& op_name, PrimExpr (*func)(PrimExpr, Span),
                  std::vector<PrimExpr> args) {
  CheckArgs(args, 1, op_name);
  return func(std::move(args[0]), Span());
}
PrimExpr MakeBinOp(const std::string& op_name, PrimExpr (*func)(PrimExpr, PrimExpr, Span),
                   std::vector<PrimExpr> args) {
  CheckArgs(args, 2, op_name);
  return func(std::move(args[0]), std::move(args[1]), Span());
}
PrimExpr MakeSelect(std::vector<PrimExpr> args) {
  CheckArgs(args, 3, "select");
  return select(std::move(args[0]), std::move(args[1]), std::move(args[2]));
}

std::pair<std::string, char> SplitVarParts(const std::string& str) {
  if (str.size() > 2 && str[str.size() - 2] == ':') {
    return {str.substr(0, str.size() - 2), str.back()};
  } else {
    return {str, (char)0};
  }
}

std::pair<PrimExpr, size_t> ParseExprPreorder(const std::string& str,
                                              const Optional<Map<String, Var>>& var_map,
                                              size_t loc = 0) {
  auto ParseToken = [&var_map](const std::string& str) -> PrimExpr {
    ICHECK(!str.empty());
    if (str == "true") {
      return Bool(true);
    } else if (str == "false") {
      return Bool(false);
    } else if (str == "inf" || str == "-inf") {
      throw std::runtime_error("Inf not supported");
    } else if (std::isalpha(str[0]) || str[0] == '_') {
      auto [var_name, annot] = SplitVarParts(str);
      if (var_map) {
        auto var = var_map.value().Get(var_name);
        ICHECK(var.defined()) << "Undefined variable: " << str;
        return var.value();
      }
      switch (annot) {
        case 'b':
          return Var(var_name, DataType::Bool());
        case 's':
          return SizeVar(var_name, SizeVarKind::kShapeVar);
        case 'k':
          return SizeVar(var_name, SizeVarKind::kScheduleKnob);
        case 'v':
          return SizeVar(var_name, SizeVarKind::kShorthand);
        case 0:
          return Var(var_name, DataType::Int(32));
        default:
          throw std::runtime_error("Unknown annotation: " + str);
      }
    }
    auto is_digit = [](char c) { return std::isdigit(c); };
    bool all_digits = std::all_of(str.begin(), str.end(), is_digit);
    bool neg_all_digits = str[0] == '-' && std::all_of(str.begin() + 1, str.end(), is_digit);
    if (all_digits || neg_all_digits) {
      return Integer(std::stoll(str));
    }

    return FloatImm(DataType::Float(32), std::stof(str));
  };

  if (str[loc] != '(') {
    if (loc == 0) {
      return {ParseToken(str), loc};
    }
    throw std::runtime_error("Expected '(' at position " + std::to_string(loc) + ": " + str);
  }
  std::string op_str;
  std::vector<PrimExpr> args;
  size_t last_sep = loc;
  ++loc;
  for (; loc < str.size(); ++loc) {
    if (str[loc] == '(') {
      if (op_str.empty()) {
        throw std::runtime_error("Expected operator at position " + std::to_string(loc) + ": " +
                                 str);
      }
      auto [subexpr, new_loc] = ParseExprPreorder(str, var_map, loc);
      args.push_back(std::move(subexpr));
      loc = new_loc;  // Will be inc'd by loop
      last_sep = new_loc;
    } else if (str[loc] == ' ' || str[loc] == ')') {
      if (last_sep < loc - 1) {
        std::string token = str.substr(last_sep + 1, loc - last_sep - 1);
        if (op_str.empty()) {
          op_str = std::move(token);
        } else {
          args.push_back(ParseToken(token));
        }
      }
      last_sep = loc;
      if (str[loc] == ')') {
        break;
      }
    }
  }
  if (str[loc] != ')') {
    throw std::runtime_error("Expected ')' at end of expression");
  }
  if (op_str.empty()) {
    throw std::runtime_error("Expected operator at position " + std::to_string(loc) + ": " + str);
  }
  if (op_str == "+") {
    return {MakeBinOp("Add", add, args), loc};
  } else if (op_str == "-") {
    return {MakeBinOp("Sub", sub, args), loc};
  } else if (op_str == "*") {
    return {MakeBinOp("Mul", mul, args), loc};
  } else if (op_str == "/") {
    return {MakeBinOp("Div", div, args), loc};
  } else if (op_str == "mod") {
    return {MakeBinOp("FloorMod", floormod, args), loc};
  } else if (op_str == "//") {
    return {MakeBinOp("FloorDiv", floordiv, args), loc};
  } else if (op_str == "min") {
    return {MakeBinOp("Min", min, args), loc};
  } else if (op_str == "max") {
    return {MakeBinOp("Max", max, args), loc};
  } else if (op_str == "pow") {
    return {MakeBinOp("Pow", pow, args), loc};
  } else if (op_str == "==") {
    return {MakeBinOp("EQ", equal, args), loc};
  } else if (op_str == "!=") {
    return {MakeBinOp("NE", not_equal, args), loc};
  } else if (op_str == "<") {
    return {MakeBinOp("LT", less, args), loc};
  } else if (op_str == "<=") {
    return {MakeBinOp("LE", less_equal, args), loc};
  } else if (op_str == ">") {
    return {MakeBinOp("GT", greater, args), loc};
  } else if (op_str == ">=") {
    return {MakeBinOp("GE", greater_equal, args), loc};
  } else if (op_str == "&&") {
    return {MakeBinOp("And", logical_and, args), loc};
  } else if (op_str == "||") {
    return {MakeBinOp("Or", logical_or, args), loc};
  } else if (op_str == "!") {
    return {MakeUnOp("Not", logical_not, args), loc};
  } else if (op_str == "select") {
    return {MakeSelect(args), loc};
  } else if (op_str == "log") {
    return {MakeUnOp(op_str, log, args), loc};
  } else if (op_str == "logk") {
    return {MakeBinOp(op_str, logk, args), loc};
  } else if (op_str == "exp") {
    return {MakeUnOp(op_str, exp, args), loc};
  } else if (op_str == "sigmoid") {
    return {MakeUnOp(op_str, sigmoid, args), loc};
  } else {
    throw std::runtime_error("Unknown operator " + op_str + " in " + str);
  }
}

String SimplifyExprStr(const String& str, size_t max_n_iters, size_t max_n_nodes,
                       bool diff_approx) {
  char* sp = simplify_expr(str.c_str(), max_n_iters, max_n_nodes, diff_approx);
  std::string simplified(sp);
  free_str(sp);
  return simplified;
}

PrimExpr SimplifyExpr(const PrimExpr& expr, size_t max_n_iters, size_t max_n_nodes,
                      bool diff_approx) {
  PreorderPrinter printer;
  auto expr_str = printer.Print(expr);
  char* simpl_str = simplify_expr(expr_str.c_str(), max_n_iters, max_n_nodes, diff_approx);
  PrimExpr simplified = ParseExprPreorder(simpl_str, printer.var_map).first;
  free_str(simpl_str);
  // If not really simplified, don't return the simplified version
  // because the order of the variables etc. may be different.
  return CountOps(simplified) < CountOps(expr) ? simplified : expr;
}

PrimExpr SubAndSimplify(const PrimExpr& expr,
                        const std::unordered_map<std::string, PrimExpr>& subst,
                        bool simpl_only_on_change, size_t max_n_iters, size_t max_n_nodes) {
  bool changed = false;
  auto expr_ = SubstByName(expr, subst, &changed);
  if (!changed && simpl_only_on_change) {
    return expr;
  }
  return SimplifyExpr(expr_, max_n_iters, max_n_nodes, true);
}

bool IsExprEquivalent(const PrimExpr& lhs, const PrimExpr& rhs, size_t max_n_iters,
                      size_t max_n_nodes, bool diff_approx) {
  PreorderPrinter printer;
  auto lhs_str = printer.Print(lhs), rhs_str = printer.Print(rhs);
  return is_equivalent(lhs_str.c_str(), rhs_str.c_str(), false, max_n_iters, max_n_nodes,
                       diff_approx);
}

String PrintExprPrefix(const PrimExpr& expr) {
  PreorderPrinter printer;
  return printer.Print(expr);
}

PrimExpr ParseExprPrefix(const String& str) {
  return ParseExprPreorder(str, Optional<Map<String, Var>>()).first;
}

TVM_REGISTER_GLOBAL("arith.SimplifyExprStr").set_body_typed(SimplifyExprStr);
TVM_REGISTER_GLOBAL("arith.IsExprEquivalent").set_body_typed(IsExprEquivalent);
TVM_REGISTER_GLOBAL("arith.IsExprStrEquivalent")
    .set_body_typed([](String se1, String se2, size_t n_iters, size_t n_nodes, bool diff_approx) {
      return is_equivalent(se1.c_str(), se2.c_str(), true, n_iters, n_nodes, diff_approx);
    });
TVM_REGISTER_GLOBAL("arith.PrintExprPreorder").set_body_typed([](const PrimExpr& expr) {
  PreorderPrinter printer;
  return printer.Print(expr);
});
TVM_REGISTER_GLOBAL("arith.ParseExprPreorder").set_body_typed([](const String& str) {
  return ParseExprPreorder(str, Optional<Map<String, Var>>()).first;
});

}  // namespace arith
}  // namespace tvm
