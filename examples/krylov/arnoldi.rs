use rustebra::krylov::arnoldi;
use rustebra::storage::{Basis, StaticStorage};

pub(crate) fn run() {
    println!("\n== Arnoldi iteration ==");
    // Non-symmetric on purpose: Arnoldi handles general operators, unlike Lanczos.
    let a = StaticStorage::new([4.0, 1.0, 2.0, 3.0, 3.0, 1.0, 5.0, 1.0, 5.0]);
    let v0 = StaticStorage::new([1.0, 1.0, 1.0]);
    let mut buffer = [0.0; 9];
    let mut basis = Basis::<f64, 3>::new(&mut buffer, 3).unwrap();
    let mut scratch = [0.0; 3];

    let (h, reached) = arnoldi(&a, 3, &v0, 1e-12, &mut basis, &mut scratch).unwrap();
    println!("reached = {reached}");
    for r in 0..3 {
        let row: Vec<f64> = (0..3).map(|c| h.entry(r, c).unwrap()).collect();
        println!("h[{r}] = {row:?}");
    }

    // Requesting fewer basis vectors than the matrix dimension (K < n) still produces the
    // leading block of the same Hessenberg form, at a fraction of the memory: only `K`
    // vectors of the basis are ever stored.
    let mut partial_buffer = [0.0; 6];
    let mut partial_basis = Basis::<f64, 2>::new(&mut partial_buffer, 3).unwrap();
    let mut partial_scratch = [0.0; 3];
    let (partial_h, partial_reached) =
        arnoldi(&a, 3, &v0, 1e-12, &mut partial_basis, &mut partial_scratch).unwrap();
    println!("partial reached (K = 2) = {partial_reached}");
    println!("partial h[0][0] = {}", partial_h.entry(0, 0).unwrap());

    // A rank-1 matrix started from its own range breaks down after one vector — a good
    // outcome (the invariant subspace was found), reported as `Ok`, not an error.
    let rank_one = StaticStorage::new([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    let v_rank_one = StaticStorage::new([1.0, 1.0, 1.0]);
    let mut breakdown_buffer = [0.0; 9];
    let mut breakdown_basis = Basis::<f64, 3>::new(&mut breakdown_buffer, 3).unwrap();
    let mut breakdown_scratch = [0.0; 3];
    let (_, breakdown_reached) = arnoldi(
        &rank_one,
        3,
        &v_rank_one,
        1e-10,
        &mut breakdown_basis,
        &mut breakdown_scratch,
    )
    .unwrap();
    println!("breakdown reached (rank-1 input) = {breakdown_reached}");
}
