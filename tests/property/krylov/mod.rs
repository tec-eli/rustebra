// The shared harness also serves the edge-case and stress targets; the fixed-matrix helpers
// are unused in this one.
#[allow(dead_code)]
mod common;
mod inverse_power_iteration;
mod lanczos;
mod power_iteration;
