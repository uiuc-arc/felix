mod lang;
use std::convert::TryInto;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

fn str_from_ptr(s_raw: *const c_char) -> String {
    let raw = unsafe { CStr::from_ptr(s_raw) };
    match raw.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => {
            panic!("failed to convert CStr to str");
        }
    }
}

#[no_mangle]
pub extern "C" fn simplify_expr(
    s_raw: *const c_char,
    n_iters: u64,
    n_nodes: u64,
    simpl_rel: bool,
) -> *mut c_char {
    let s = str_from_ptr(s_raw);
    let simplified = lang::simplify(
        &s,
        n_iters.try_into().unwrap(),
        n_nodes.try_into().unwrap(),
        simpl_rel,
    );
    let c_str = CString::new(simplified).unwrap();
    c_str.into_raw()
}

#[no_mangle]
// trunk-ignore(clippy/missing_safety_doc)
pub unsafe extern "C" fn free_str(s_raw: *mut c_char) {
    unsafe {
        if s_raw.is_null() {
            return;
        }
        CString::from_raw(s_raw)
    };
}

#[no_mangle]
pub extern "C" fn is_equivalent(
    lhs_raw: *const c_char,
    rhs_raw: *const c_char,
    explain: bool,
    n_iters: u64,
    n_nodes: u64,
    simpl_rel: bool,
) -> bool {
    let lhs = str_from_ptr(lhs_raw);
    let rhs = str_from_ptr(rhs_raw);
    lang::is_equivalent(
        &lhs,
        &rhs,
        explain,
        n_iters.try_into().unwrap(),
        n_nodes.try_into().unwrap(),
        simpl_rel,
    )
}

#[no_mangle]
pub extern "C" fn print_rule_counter() {
    lang::print_rule_counter();
}
