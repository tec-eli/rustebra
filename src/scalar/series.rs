use core::ops::{Add, Div, Mul};

const ITERATIONS: u32 = 20;

pub(super) fn taylor_series<T>(
    init_term: T,
    num_factor: T,
    mut a: T,
    mut b: T,
    step_a: T,
    step_b: T,
) -> T
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    let mut term = init_term;
    let mut sum = term;

    for _ in 0..ITERATIONS {
        term = term.mul(num_factor).div(a.mul(b));
        sum = sum.add(term);
        a = a.add(step_a);
        b = b.add(step_b);
    }
    sum
}
