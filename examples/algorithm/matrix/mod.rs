mod arithmetic;
mod cholesky;
mod condition;
mod determinant;
mod lu;
mod qr;
mod rank;
mod svd;

pub(crate) fn run() {
    arithmetic::run();
    determinant::run();
    rank::run();
    lu::run();
    qr::run();
    cholesky::run();
    svd::run();
    condition::run();
}
