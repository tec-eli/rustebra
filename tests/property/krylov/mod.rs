// The shared harness also serves the edge-case and stress targets; the fixed-matrix helpers
// are unused in this one.
mod arnoldi;
#[allow(dead_code)]
mod common;
mod gmres;
mod inverse_power_iteration;
mod lanczos;
mod power_iteration;
