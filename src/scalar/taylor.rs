use core::ops::{Add, Div, Mul};

/// Fixed iteration count for [`taylor_series`].
///
/// A higher count increases precision for inputs further away from zero, but
/// adds a linear computation cost.
const ITERATIONS: u32 = 20;

/// Computes a generic power series expansion around zero using an efficient recurrence relation.
///
/// Instead of recomputing exponents and factorials from scratch for every term, this function
/// progresses iteratively by multiplying the previous term by a constant numerator factor
/// and dividing it by two stepping denominator components.
///
/// # Recurrence Relation
/// In each iteration $k$, the next term of the series is computed as:
/// `term_{k+1} = term_k * num_factor / (a * b)`
///
/// After updating the sum, the denominator components advance linearly:
/// `a = a + step_a`
/// `b = b + step_b`
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

#[cfg(test)]
mod tests {
    use super::taylor_series;

    #[test]
    fn sine_parameterization_matches_known_angle() {
        let value: f64 = core::f64::consts::FRAC_PI_2;
        let neg_x2 = -(value * value);

        let result = taylor_series(value, neg_x2, 2.0, 3.0, 2.0, 2.0);
        assert!((result - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_parameterization_matches_known_angle() {
        let value: f64 = core::f64::consts::PI;
        let neg_x2 = -(value * value);

        let result = taylor_series(1.0, neg_x2, 1.0, 2.0, 2.0, 2.0);
        assert!((result - (-1.0)).abs() < 1e-9);
    }
}
