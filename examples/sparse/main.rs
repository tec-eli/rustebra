//! Tour every public function and type in the `sparse` module.
//!
//! Run with: `cargo run --example sparse --features alloc`

mod add;
mod convert;
mod coo;
mod csc;
mod csr;
mod linear_op;
mod matmat;
mod matvec;
mod prune;
mod scale;
mod sorted_csr;
mod spmm;

fn main() {
    coo::run();
    csr::run();
    csc::run();
    sorted_csr::run();
    convert::run();
    scale::run();
    matvec::run();
    linear_op::run();
    matmat::run();
    add::run();
    spmm::run();
    prune::run();
}
