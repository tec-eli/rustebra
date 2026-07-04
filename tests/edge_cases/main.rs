// The shared Krylov test harness lives with the property suite; only a subset of it is used
// per test target, hence the dead_code allowance.
#[path = "../property/krylov/common.rs"]
#[allow(dead_code)]
mod common;

mod krylov;
