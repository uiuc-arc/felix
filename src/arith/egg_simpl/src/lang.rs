use egg::*;
use once_cell::sync::Lazy;
use ordered_float::{Float, OrderedFloat};
use std::cmp;
use std::fmt::{Display, Error, Formatter};
use std::ops::{Add, Mul, Sub};
use std::str::FromStr;

#[cfg(feature = "count")]
use std::collections::HashMap;
#[cfg(feature = "count")]
use std::sync::Mutex;

type EGraph = egg::EGraph<Math, ConstantFold>;
type Rewrite = egg::Rewrite<Math, ConstantFold>;
type Runner = egg::Runner<Math, ConstantFold>;
type F64 = OrderedFloat<f64>;

// trunk-ignore(clippy/derive_hash_xor_eq)
#[derive(Clone, Copy, PartialOrd, Ord, Debug, Hash)]
pub enum Const {
    Float(F64),
    Bool(bool),
}

impl PartialEq for Const {
    fn eq(&self, other: &Self) -> bool {
        const EPS: f64 = 1e-7;
        match (self, other) {
            (Const::Float(OrderedFloat(f1)), Const::Float(OrderedFloat(f2))) => {
                (f1 - f2).abs() < EPS
            }
            (Const::Bool(b1), Const::Bool(b2)) => b1 == b2,
            _ => false,
        }
    }
}
impl Eq for Const {}

impl FromStr for Const {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Ok(f) = s.parse::<f64>() {
            Ok(Const::Float(OrderedFloat(f)))
        } else if let Ok(b) = s.parse::<bool>() {
            Ok(Const::Bool(b))
        } else {
            Err(())
        }
    }
}

impl Display for Const {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        match self {
            Const::Float(fl) => write!(f, "{}", fl),
            Const::Bool(b) => write!(f, "{}", b),
        }
    }
}

define_language! {
    pub enum Math {
        Const(Const),
        Symbol(Symbol),
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "//" = FloorDiv([Id; 2]),
        "mod" = FloorMod([Id; 2]),
        "min" = Min([Id; 2]),
        "max" = Max([Id; 2]),
        "pow" = Pow([Id; 2]),
        "log" = Log([Id; 1]),
        "logk" = LogK([Id; 2]),
        "exp" = Exp([Id; 1]),
        "==" = Eq([Id; 2]),
        "!=" = Ne([Id; 2]),
        "<" = Lt([Id; 2]),
        "<=" = Le([Id; 2]),
        ">" = Gt([Id; 2]),
        ">=" = Ge([Id; 2]),
        "&&" = And([Id; 2]),
        "||" = Or([Id; 2]),
        "!" = Not([Id; 1]),
        "select" = Select([Id; 3]),
        "sigmoid" = Sigmoid([Id; 1]),
        "hump" = Hump([Id; 1]),
    }
}

macro_rules! _extract_ast {
    ($op:expr, $x:ident, $e:expr) => {{
        let e = $x($e)?;
        let ast = format!(concat!("(", $op, " {})"), e).parse().unwrap();
        (e, ast)
    }};
    ($op:expr, $x:ident, $lhs:expr, $rhs:expr) => {{
        let (lhs, rhs) = ($x($lhs)?, $x($rhs)?);
        let ast = format!(concat!("(", $op, " {} {})"), lhs, rhs)
            .parse()
            .unwrap();
        (lhs, rhs, ast)
    }};
}
macro_rules! _opt_op {
    ($var_in:path, $op:expr, $x:ident, $arg:expr, $op_func:expr) => {{
        let (arg, ast) = _extract_ast!($op, $x, $arg);
        if let $var_in(arg) = arg {
            ($var_in($op_func(arg)?), ast)
        } else {
            panic!(concat!("expected a ", stringify!($var_in)))
        }
    }};
    ($var_in:path, $var_out:path, $op:expr, $x:ident, $lhs:expr, $rhs:expr, $op_func:expr) => {{
        let (lhs, rhs, ast) = _extract_ast!($op, $x, $lhs, $rhs);
        let value = match (lhs, rhs) {
            ($var_in(l), $var_in(r)) => $var_out($op_func(l, r)?),
            _ => panic!("expected two floats"),
        };
        (value, ast)
    }};
}
macro_rules! _general_op {
    ($var_in:path, $op:expr, $x:ident, $arg:expr, $op_func:expr) => {
        _opt_op!($var_in, $op, $x, $arg, |x| Some($op_func(x)))
    };
    ($var_in:path, $var_out:path, $op:expr, $x:ident, $lhs:expr, $rhs:expr, $op_func:expr) => {{
        _opt_op!($var_in, $var_out, $op, $x, $lhs, $rhs, |l, r| {
            Some($op_func(l, r))
        })
    }};
}
macro_rules! float_op {
    ($op:expr, $x:ident, $arg:expr, $op_func:expr) => {
        _general_op!(Const::Float, $op, $x, $arg, $op_func)
    };
    ($op:expr, $x:ident, $lhs:expr, $rhs:expr, $op_func:expr) => {
        _general_op!(Const::Float, Const::Float, $op, $x, $lhs, $rhs, $op_func)
    };
}
macro_rules! float_opt_op {
    ($op:expr, $x:ident, $arg:expr, $op_func:expr) => {
        _opt_op!(Const::Float, $op, $x, $arg, $op_func)
    };
    ($op:expr, $x:ident, $lhs:expr, $rhs:expr, $op_func:expr) => {
        _opt_op!(Const::Float, Const::Float, $op, $x, $lhs, $rhs, $op_func)
    };
}
macro_rules! bool_op {
    ($op:expr, $x:ident, $arg:expr, $op_func:expr) => {
        _general_op!(Const::Bool, $op, $x, $arg, $op_func)
    };
    ($op:expr, $x:ident, $lhs:expr, $rhs:expr, $op_func:expr) => {
        _general_op!(Const::Bool, Const::Bool, $op, $x, $lhs, $rhs, $op_func)
    };
}
macro_rules! cmp_op {
    ($op:expr, $x:ident, $lhs:expr, $rhs:expr, $op_func:expr) => {
        _general_op!(Const::Float, Const::Bool, $op, $x, $lhs, $rhs, $op_func)
    };
}

#[derive(Default, Clone)]
pub struct ConstantFold;
impl egg::Analysis<Math> for ConstantFold {
    type Data = Option<(Const, egg::PatternAst<Math>)>;

    fn make(egraph: &EGraph, enode: &Math) -> Self::Data {
        let x = |i: &Id| egraph[*i].data.as_ref().map(|d| d.0);
        let safe_div = |l: F64, r: F64| {
            if *r == 0.0 {
                None
            } else {
                Some(l / r)
            }
        };
        let safe_log = |base: F64, val: F64| {
            if *val <= 0.0 || *base <= 0.0 || *base == 1.0 {
                None
            } else {
                Some(val.log(base))
            }
        };
        let safe_ln = |x: F64| {
            if *x <= 0.0 {
                None
            } else {
                Some(x.ln())
            }
        };
        Some(match enode {
            Math::Const(c) => (*c, format!("{}", c).parse().unwrap()),
            Math::Add([lhs, rhs]) => float_op!("+", x, lhs, rhs, F64::add),
            Math::Sub([lhs, rhs]) => float_op!("-", x, lhs, rhs, F64::sub),
            Math::Mul([lhs, rhs]) => float_op!("*", x, lhs, rhs, F64::mul),
            Math::Div([lhs, rhs]) => float_opt_op!("/", x, lhs, rhs, safe_div),
            Math::FloorDiv([lhs, rhs]) => {
                float_opt_op!("//", x, lhs, rhs, |l: F64, r: F64| {
                    Some(safe_div(l, r)?.floor())
                })
            }
            Math::FloorMod([lhs, rhs]) => {
                float_opt_op!("mod", x, lhs, rhs, |l: F64, r: F64| {
                    Some(l - safe_div(l, r)?.floor() * r)
                })
            }
            Math::Min([lhs, rhs]) => float_op!("min", x, lhs, rhs, cmp::min),
            Math::Max([lhs, rhs]) => float_op!("max", x, lhs, rhs, cmp::max),
            Math::Pow([lhs, rhs]) => float_op!("pow", x, lhs, rhs, F64::powf),
            Math::Log([arg]) => float_opt_op!("log", x, arg, safe_ln),
            Math::Exp([arg]) => float_op!("exp", x, arg, F64::exp),
            Math::LogK([lhs, rhs]) => {
                float_opt_op!("logk", x, lhs, rhs, safe_log)
            }
            Math::Eq([lhs, rhs]) => cmp_op!("==", x, lhs, rhs, |a, b| a == b),
            Math::Ne([lhs, rhs]) => cmp_op!("!=", x, lhs, rhs, |a, b| a != b),
            Math::Lt([lhs, rhs]) => cmp_op!("<", x, lhs, rhs, |a, b| a < b),
            Math::Le([lhs, rhs]) => cmp_op!("<=", x, lhs, rhs, |a, b| a <= b),
            Math::Gt([lhs, rhs]) => cmp_op!(">", x, lhs, rhs, |a, b| a > b),
            Math::Ge([lhs, rhs]) => cmp_op!(">=", x, lhs, rhs, |a, b| a >= b),
            Math::And([lhs, rhs]) => bool_op!("&&", x, lhs, rhs, |a, b| a && b),
            Math::Or([lhs, rhs]) => bool_op!("||", x, lhs, rhs, |a, b| a || b),
            Math::Not([arg]) => bool_op!("!", x, arg, |a: bool| !a),
            // Math::Select([cond, lhs, rhs])
            // `select b x y` is almost never fully constant. Let's use rules to simplify it instead.
            _ => return None,
        })
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> egg::DidMerge {
        egg::merge_option(to, from, |a, b| {
            if a.0 != b.0 {
                println!(
                    "Merged non-equal constants:\n  {}, from {};\n  {}, from {}",
                    a.0, a.1, b.0, b.1
                );
                panic!("Merged non-equal constants");
            }
            egg::DidMerge(false, false)
        })
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        let data = egraph[id].data.clone();
        if let Some((c, pat)) = data {
            if egraph.are_explanations_enabled() {
                egraph.union_instantiations(
                    &pat,
                    &format!("{}", c).parse().unwrap(),
                    &Default::default(),
                    "constant_fold".to_string(),
                );
            } else {
                let added = egraph.add(Math::Const(c));
                egraph.union(id, added);
            }
            // to not prune, comment this out
            egraph[id].nodes.retain(|n| n.is_leaf());

            #[cfg(debug_assertions)]
            egraph[id].assert_unique_leaves();
        }
    }
}

fn _is_const_with_pred(
    var: &str,
    fop: impl Fn(f64) -> bool,
) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| {
        if let Some(n) = &egraph[subst[var]].data {
            if let Const::Float(OrderedFloat(f)) = n.0 {
                return fop(f);
            }
        }
        false
    }
}

fn is_not_zero(var: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| {
        if let Some(n) = &egraph[subst[var]].data {
            if let Const::Float(OrderedFloat(f)) = n.0 {
                return f != 0.0;
            }
        }
        true
    }
}

fn is_symbol(var: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| {
        egraph[subst[var]]
            .nodes
            .iter()
            .any(|n| matches!(n, Math::Symbol(..)))
    }
}

fn _is_pow_of(egraph: &mut EGraph, subst: &Subst, x: Var, base: Var) -> Option<i64> {
    let (x, base) = (
        egraph[subst[x]].data.as_ref()?.0,
        egraph[subst[base]].data.as_ref()?.0,
    );
    match (x, base) {
        (Const::Float(OrderedFloat(x)), Const::Float(OrderedFloat(base))) => {
            let log_ = x.log(base);
            if (base.powf(log_.round()) - x).abs() > 1e-6 {
                return None;
            }
            Some(log_.round() as i64)
        }
        _ => None,
    }
}

fn is_pow_of(x: &str, base: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let (x, base) = (x.parse().unwrap(), base.parse().unwrap());
    move |egraph, _, subst| _is_pow_of(egraph, subst, x, base).is_some()
}

pub static BASIC_RULES: Lazy<Vec<Rewrite>> = Lazy::new(|| {
    vec![
        rewrite!("mul-comm";    "(* ?a ?b)"             => "(* ?b ?a)"),
        rewrite!("mul-assoc";   "(* ?a (* ?b ?c))"      => "(* (* ?a ?b) ?c)"),
        rewrite!("add-comm";    "(+ ?a ?b)"             => "(+ ?b ?a)"),
        rewrite!("add-assoc";   "(+ ?a (+ ?b ?c))"      => "(+ (+ ?a ?b) ?c)"),
        rewrite!("mul-1";       "(* ?a 1)"              => "?a"),
        rewrite!("mul-0";       "(* ?a 0)"              => "0"),
        rewrite!("add-0";       "(+ ?a 0)"              => "?a"),
        rewrite!("sub-canon";   "(- ?a ?b)"             => "(+ ?a (* -1 ?b))"),
        rewrite!("sub-intro";   "(+ ?a (* -1 ?b))"      => "(- ?a ?b)"),
        rewrite!("sub-cancel";  "(+ ?a (* -1 ?a))"      => "0"),
        rewrite!("div-canon";   "(/ ?a ?b)"             => "(* ?a (pow ?b -1))"),
        rewrite!("div-intro";   "(* ?a (pow ?b -1))"    => "(/ ?a ?b)"  if is_not_zero("?b")),
        rewrite!("div-cancel";  "(* ?a (pow ?a -1))"    => "1"),
        rewrite!("add-mul-distrib";     "(* ?a (+ ?b ?c))"              => "(+ (* ?a ?b) (* ?a ?c))"),
        rewrite!("add-mul-factor";      "(+ (* ?a ?b) (* ?a ?c))"       => "(* ?a (+ ?b ?c))"),
        rewrite!("pow-mul-factor";      "(* (pow ?a ?b) (pow ?a ?c))"   => "(pow ?a (+ ?b ?c))"),
        rewrite!("pow-1";       "(pow ?a 1)"            => "?a"),
        rewrite!("pow-pow";     "(pow (pow ?a ?b) ?c)"  => "(pow ?a (* ?b ?c))"),
        rewrite!("const-mul-pow";      "(* ?a (pow ?b ?c))"     => "(pow ?b (+ (logk ?b ?a) ?c))" if is_pow_of("?a", "?b")),
        rewrite!("const-div-pow";      "(/ ?a (pow ?b ?c))"     => "(pow ?b (- (logk ?b ?a) ?c))" if is_pow_of("?a", "?b")),
        rewrite!("max-self";    "(max ?a ?a)"           => "?a"),
        rewrite!("max-a-add-b"; "(max ?a (+ ?a ?b))"    => "(+ ?a ?b)"  if _is_const_with_pred("?b", |x| x >= 0.0)),
        rewrite!("min-self";    "(max ?a ?a)"           => "?a"),
        rewrite!("min-pow";     "(min (pow ?a ?b) (pow ?a ?c))" => "(pow ?a (min ?b ?c))"),
        rewrite!("logk-canon";  "(logk ?a ?b)"          => "(/ (log ?b) (log ?a))"),
        rewrite!("log-prod";    "(log (* ?a ?b))"       => "(+ (log ?a) (log ?b))"),
        rewrite!("log-pow";     "(log (pow ?a ?b))"     => "(* ?b (log ?a))"),
        rewrite!("floordiv-cancel"; "(// (* ?a ?b) ?a)"     => "?b"),
        rewrite!("floordiv-merge";  "(// (// ?a ?b) ?c)"    => "(// (/ ?a ?b) ?c)"),
        rewrite!("floordiv-neg-1";  "(// -1 ?a)"            => "-1"),
        rewrite!("select-same";     "(select ?a ?b ?b)"     => "?b"),
        rewrite!("select-true";     "(select true ?a ?b)"   => "?a"),
        rewrite!("select-false";    "(select false ?a ?b)"  => "?b"),
        rewrite!("exp-never-eq-0";  "(== (exp ?a) 0)"       => "false"),
        // Boolean rules:
        rewrite!("and-comm";    "(&& ?a ?b)"            => "(&& ?b ?a)"),
        rewrite!("and-assoc";   "(&& ?a (&& ?b ?c))"    => "(&& (&& ?a ?b) ?c)"),
        rewrite!("or-comm";     "(|| ?a ?b)"            => "(|| ?b ?a)"),
        rewrite!("or-assoc";    "(|| ?a (|| ?b ?c))"    => "(|| (|| ?a ?b) ?c)"),
        rewrite!("or-dup";      "(|| ?a ?a)"            => "?a"),
        rewrite!("or-and-dup";  "(|| ?a (&& ?a ?b))"    => "?a"),
        rewrite!("or-and-dist"; "(|| ?a (&& ?b ?c))"    => "(&& (|| ?a ?b) (|| ?a ?c))"),
        rewrite!("and-or-dist"; "(&& ?a (|| ?b ?c))"    => "(|| (&& ?a ?b) (&& ?a ?c))"),
        rewrite!("and-true";    "(&& ?a true)"          => "?a"),
        rewrite!("and-false";   "(&& ?a false)"         => "false"),
        rewrite!("or-true";     "(|| ?a true)"          => "true"),
        rewrite!("or-false";    "(|| ?a false)"         => "?a"),
        // Comparison rules:
        // Attempt to boil everything down to == and <
        rewrite!("ne-canon";    "(!= ?a ?b)"    => "(! (== ?a ?b))"),
        rewrite!("ge-canon";    "(>= ?a ?b)"    => "(! (< ?a ?b))"),
        rewrite!("gt-canon";    "(> ?a ?b)"     => "(< ?b ?a)"),
        rewrite!("le-canon";    "(<= ?a ?b)"    => "(! (< ?b ?a))"),
        rewrite!("eq-comm";     "(== ?a ?b)"    => "(== ?b ?a)"),
        rewrite!("lt-min";      "(< ?a (min ?b ?c))"    => "(&& (< ?a ?b) (< ?a ?c))"),
        rewrite!("eq-min";      "(== ?a (min ?b ?c))"   => "(|| (== ?a ?b) (== ?a ?c))"),
    ]
});

pub static DIFF_APPROX_RULES: Lazy<Vec<Rewrite>> = Lazy::new(|| {
    let mut ret_rules = vec![];
    ret_rules.extend_from_slice(BASIC_RULES.as_slice());
    ret_rules.extend(vec![
        // Differentiability-approx-specific simplification rules
        rewrite!("1-lt-prod";   "(< 1 (* ?a ?b))"       => "(|| (< 1 ?a) (< 1 ?b))"),
        rewrite!("lt-pow-1";    "(< (pow ?a ?b) ?c)"    => "(< ?b (logk ?a ?c))"    if _is_const_with_pred("?a", |x| x >= 2.0)),
        rewrite!("lt-pow-2";    "(< ?c (pow ?a ?b))"    => "(< (logk ?a ?c) ?b)"    if _is_const_with_pred("?a", |x| x >= 2.0)),
        rewrite!("eq-to-lt";    "(== 1 ?a)"             => "(! (< 1 ?a))"           if is_symbol("?a")),
        rewrite!("1-eq-prod";   "(== 1 (* ?a ?b))"      => "(&& (== 1 ?a) (== 1 ?b))"),
        rewrite!("neg-1-never-0";   "(== 0 (* -1 ?a))"  => "false"),
        rewrite!("hump-0";      "(hump 0)"              => "1"),
        rewrite!("hump-const";  "(hump ?a)"             => "0" if _is_const_with_pred("?a", |x| x != 0.0)),
    ]);
    ret_rules
});

pub struct DiffApproxSimplCostFn;
impl egg::CostFunction<Math> for DiffApproxSimplCostFn {
    type Cost = usize;
    fn cost<C>(&mut self, enode: &Math, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        // Prefer the form (|| (< 1 ?a) (< 1 ?b)) over (< 1 (* a b)).
        let op_cost = match enode {
            // Thus all logical and lt / eq operators are cheap,
            // and everything else is expensive.
            Math::Const(..) => 1,
            Math::Symbol(..) => 1,
            Math::And(..) => 1,
            Math::Or(..) => 1,
            Math::Not(..) => 1,
            Math::Lt(..) => 1,
            // Prefer Lt over any other comparators
            Math::Eq(..) => 5,
            Math::Ne(..) => 5,
            Math::Le(..) => 5,
            Math::Gt(..) => 5,
            Math::Ge(..) => 5,
            _ => 10,
        };
        enode.fold(op_cost, |sum, i| sum + costs(i))
    }
}

#[cfg(feature = "count")]
pub static GLOBAL_RULE_COUNTER: Lazy<Mutex<HashMap<Symbol, usize>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[cfg(feature = "count")]
macro_rules! find_fexp_rule_ {
    ($rule_field:ident, $func:ident) => {
        fn $func(flat: &FlatTerm<Math>) -> Option<Symbol> {
            if let Some(rule_name) = flat.$rule_field {
                return Some(rule_name);
            }
            let mut rule = None;
            for child in flat.children.iter() {
                rule = rule.or_else(|| $func(child));
            }
            rule
        }
    };
}
#[cfg(feature = "count")]
find_fexp_rule_!(forward_rule, find_fterm_fwd_rule);
#[cfg(feature = "count")]
find_fexp_rule_!(backward_rule, find_fterm_bwd_rule);

#[cfg(feature = "count")]
fn add_rules_to_counter(expl: &mut Explanation<Math>) {
    let mut rules = vec![];
    for e in expl.make_flat_explanation().iter() {
        if let Some(rule_name) = find_fterm_bwd_rule(&e) {
            rules.push(rule_name);
        }
        if let Some(rule_name) = find_fterm_fwd_rule(&e) {
            rules.push(rule_name);
        }
    }

    let mut counter = GLOBAL_RULE_COUNTER.lock().unwrap();
    for rule in rules {
        let count = counter.entry(rule).or_insert(0);
        *count += 1;
    }
}
#[cfg(not(feature = "count"))]
fn add_rules_to_counter(_expl: &mut Explanation<Math>) {}

#[cfg(feature = "count")]
pub fn print_rule_counter() {
    let counter = GLOBAL_RULE_COUNTER.lock().unwrap();
    let mut rules: Vec<_> = counter.iter().collect();
    rules.sort_by_key(|(_, count)| *count);
    for (rule, count) in rules.iter().rev() {
        println!("{}: {}", rule, count);
    }
}
#[cfg(not(feature = "count"))]
pub fn print_rule_counter() {}

fn make_runner(
    n_iters: usize,
    n_nodes: usize,
    diff_approx: bool,
    explain: bool,
) -> (Runner, &'static Vec<Rewrite>) {
    let rules = if diff_approx {
        &*DIFF_APPROX_RULES
    } else {
        &*BASIC_RULES
    };
    let mut runner = Runner::default()
        .with_iter_limit(n_iters)
        .with_node_limit(n_nodes);
    if explain {
        runner = runner.with_explanations_enabled();
    }
    (runner, rules)
}

pub fn simplify(expr: &str, n_iters: usize, n_nodes: usize, diff_approx: bool) -> String {
    let expr: RecExpr<Math> = expr.parse().expect("failed to parse expression");
    let (mut runner, rules) = make_runner(n_iters, n_nodes, diff_approx, cfg!(feature = "count"));
    runner = runner.with_expr(&expr).run(rules);
    let root = runner.roots[0];
    let (_, best) = if diff_approx {
        Extractor::new(&runner.egraph, DiffApproxSimplCostFn).find_best(root)
    } else {
        Extractor::new(&runner.egraph, AstSize).find_best(root)
    };
    if cfg!(feature = "count") {
        add_rules_to_counter(&mut runner.explain_equivalence(&expr, &best));
    }
    best.to_string()
}

pub fn is_equivalent(
    lhs: &str,
    rhs: &str,
    explain: bool,
    n_iters: usize,
    n_nodes: usize,
    diff_approx: bool,
) -> bool {
    let lhs: RecExpr<Math> = lhs.parse().expect("failed to parse expression");
    let rhs: RecExpr<Math> = rhs.parse().expect("failed to parse expression");
    let (mut runner, rules) = make_runner(n_iters, n_nodes, diff_approx, explain);
    runner = runner.with_expr(&lhs).with_expr(&rhs).run(rules);
    let (lhs_id, rhs_id) = (runner.roots[0], runner.roots[1]);
    let equivalent = runner.egraph.find(lhs_id) == runner.egraph.find(rhs_id);
    if equivalent && explain {
        let explanation = runner.explain_equivalence(&lhs, &rhs);
        println!("{}", explanation);
    }
    equivalent
}
