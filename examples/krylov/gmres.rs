use rustebra::krylov::gmres;
use rustebra::sparse::CsrMatrix;
use rustebra::storage::Basis;

pub(crate) fn run() {
    println!("\n== GMRES(m) ==");

    // Non-symmetric, non-SPD system, unlike Conjugate Gradient's requirements:
    // [[4, 1], [2, 3]] x = [1, 2]. Solution: x = [0.1, 0.6].
    let a = CsrMatrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![4.0_f64, 1.0, 2.0, 3.0],
    )
    .expect("valid CSR");
    let b = [1.0, 2.0];
    let x0 = [0.0, 0.0];
    let mut out_x = [0.0; 2];
    let mut buffer = [0.0; 4];
    let mut basis = Basis::<f64, 2>::new(&mut buffer, 2).unwrap();
    let mut scratch = [0.0; 2];

    gmres(&a, &b, &x0, 10, 1e-10, &mut out_x, &mut basis, &mut scratch).expect("converges");
    println!("full-basis solve: x = {out_x:?}");

    // Restart size smaller than the problem dimension (M = 1 < n = 3): the solution still
    // emerges, just carried forward across several restart cycles instead of one.
    let a3 = CsrMatrix::new(
        3,
        3,
        vec![0, 2, 5, 7],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![5.0_f64, 1.0, 1.0, 4.0, 1.0, 1.0, 3.0],
    )
    .expect("valid CSR");
    let b3 = [6.0, 6.0, 4.0];
    let x0_3 = [0.0, 0.0, 0.0];
    let mut out_x3 = [0.0; 3];
    let mut buffer3 = [0.0; 3];
    let mut basis3 = Basis::<f64, 1>::new(&mut buffer3, 3).unwrap();
    let mut scratch3 = [0.0; 3];

    gmres(
        &a3,
        &b3,
        &x0_3,
        500,
        1e-10,
        &mut out_x3,
        &mut basis3,
        &mut scratch3,
    )
    .expect("converges across restarts");
    println!("restarted GMRES(1) solve: x = {out_x3:?}");
}
