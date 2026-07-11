//! Verifies `Scalar::sqrt`/`sin`/`cos` against a pre-generated high-precision reference,
//! targeting a relative error below `1e-14` for f64 and below `1e-6` for f32, over
//! `[-2*pi, 2*pi]`.
//!
//! The reference fixtures (`fixtures/elementary_reference_{f64,f32}.csv`) were generated
//! offline by a standalone program using the host platform's `f64` libm (correctly rounded
//! to within about 1 ulp, i.e. ~2e-16 relative error) — not the crate's own fixed-iteration
//! implementations, and not part of this test binary's build. That reference is ~100x
//! tighter than the f64 target and ~1e10x tighter than the f32 target, so it is precise
//! enough to certify both bounds despite not being arbitrary-precision. Fixtures are frozen,
//! static data rather than being computed by the test itself, so results stay reproducible
//! across platforms whose system libm may round differently than the one used to generate
//! the fixture.
//!
//! The f32 fixture's inputs are pre-rounded to f32 (then widened back to f64) before the
//! expected value is computed, so `expected` is the correct high-precision result for the
//! exact f32 input the f32 test feeds in, not for the unrounded f64 sample point.
//!
//! `sqrt`'s reference only covers `[0, 2*pi]`: negative inputs are contract-defined to
//! return `0` (see `Scalar::sqrt`'s doc comment), not a precision question, so they are out
//! of scope for an accuracy fixture.

use rustebra::scalar::Scalar;

const FIXTURE_F64: &str = include_str!("fixtures/elementary_reference_f64.csv");
const FIXTURE_F32: &str = include_str!("fixtures/elementary_reference_f32.csv");

/// Below this magnitude, `expected` is treated as a zero crossing: relative error is
/// undefined (or numerically meaningless) there, so the row is checked against the same
/// tolerance as an absolute error instead.
const ZERO_CROSSING_FLOOR: f64 = 1e-9;

struct Row {
    function: &'static str,
    input: f64,
    expected: f64,
}

fn parse_fixture(fixture: &'static str) -> Vec<Row> {
    fixture
        .lines()
        .skip(1)
        .filter(|line| !line.is_empty())
        .map(|line| {
            let mut fields = line.split(',');
            let function = fields.next().unwrap();
            let input: f64 = fields.next().unwrap().parse().unwrap();
            let expected: f64 = fields.next().unwrap().parse().unwrap();
            Row {
                function,
                input,
                expected,
            }
        })
        .collect()
}

/// Relative error against `expected`, falling back to absolute error near zero crossings
/// where relative error is undefined or dominated by rounding noise in the reference itself.
fn error(actual: f64, expected: f64) -> f64 {
    if expected.abs() < ZERO_CROSSING_FLOOR {
        (actual - expected).abs()
    } else {
        (actual - expected).abs() / expected.abs()
    }
}

fn check_f64(function: &str, input: f64, expected: f64, tol: f64) -> Option<String> {
    let actual = match function {
        "sqrt" => Scalar::sqrt(input),
        "sin" => Scalar::sin(input),
        "cos" => Scalar::cos(input),
        other => panic!("unknown function in fixture: {other}"),
    };
    let err = error(actual, expected);
    if err > tol {
        Some(format!(
            "f64::{function}({input:e}): got {actual:e}, expected {expected:e}, error {err:e} > {tol:e}"
        ))
    } else {
        None
    }
}

fn check_f32(function: &str, input: f32, expected: f32, tol: f32) -> Option<String> {
    let actual = match function {
        "sqrt" => Scalar::sqrt(input),
        "sin" => Scalar::sin(input),
        "cos" => Scalar::cos(input),
        other => panic!("unknown function in fixture: {other}"),
    };
    let err = if expected.abs() < ZERO_CROSSING_FLOOR as f32 {
        (actual - expected).abs()
    } else {
        (actual - expected).abs() / expected.abs()
    };
    if err > tol {
        Some(format!(
            "f32::{function}({input:e}): got {actual:e}, expected {expected:e}, error {err:e} > {tol:e}"
        ))
    } else {
        None
    }
}

#[test]
fn f64_elementary_functions_meet_the_relative_error_target_over_the_documented_domain() {
    const TOL: f64 = 1e-14;
    let rows = parse_fixture(FIXTURE_F64);
    let failures: Vec<String> = rows
        .iter()
        .filter_map(|row| check_f64(row.function, row.input, row.expected, TOL))
        .collect();

    assert!(
        failures.is_empty(),
        "{} of {} f64 samples exceeded the 1e-14 target:\n{}",
        failures.len(),
        rows.len(),
        failures.join("\n")
    );
}

#[test]
fn f32_elementary_functions_meet_the_relative_error_target_over_the_documented_domain() {
    const TOL: f32 = 1e-6;
    let rows = parse_fixture(FIXTURE_F32);
    let failures: Vec<String> = rows
        .iter()
        .filter_map(|row| check_f32(row.function, row.input as f32, row.expected as f32, TOL))
        .collect();

    assert!(
        failures.is_empty(),
        "{} of {} f32 samples exceeded the 1e-6 target:\n{}",
        failures.len(),
        rows.len(),
        failures.join("\n")
    );
}
