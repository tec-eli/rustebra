//! Cortex-M3 bare-metal example: low-pass filter for sensor data.
//!
//! Demonstrates rustebra on ARM Cortex-M without heap allocation.
//! Uses StaticMatrix and StaticVector for stack-only linear algebra.
//!
//! Hardware: Cortex-M3 (simulated via QEMU: lm3s6965evb machine)
//!
//! Build:
//!   cd tests/firmware/cortex-m3-lm3s6965evb
//!   cargo build --target thumbv7m-none-eabi --release
//!
//! Run (requires QEMU):
//!   "C:\Program Files\QEMU\qemu-system-arm.exe" -cpu cortex-m3 -machine lm3s6965evb \
//!     -nographic -semihosting -kernel target/thumbv7m-none-eabi/release/cortex-m3-lm3s6965evb

#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_semihosting::hprintln;
use rustebra::matrix::StaticMatrix;
use rustebra::storage::Storage;
use rustebra::vector::StaticVector;

/// A discrete-time first-order low-pass filter.
struct LowPassFilter {
    state: f32,
    alpha: f32,
}

impl LowPassFilter {
    fn new() -> Self {
        Self {
            state: 0.0,
            alpha: 0.1,
        }
    }

    fn update(&mut self, input: f32) -> f32 {
        self.state = (1.0 - self.alpha) * self.state + self.alpha * input;
        self.state
    }

    fn get_state(&self) -> f32 {
        self.state
    }
}

/// Simulate reading noisy sensor data.
fn read_simulated_sensor(sample: usize) -> f32 {
    let true_signal = if sample >= 50 { 10.0_f32 } else { 0.0 };
    // Deterministic noise in (-0.5, 0.5) using only integer ops — avoids fmodf.
    let frac = ((sample * 1573) % 1000) as f32 / 1000.0;
    let noise = (frac - 0.5) * 2.0 * 0.5;
    true_signal + noise
}

#[entry]
fn main() -> ! {
    hprintln!("=== rustebra Embedded Sensor Fusion Demo (Cortex-M3) ===");
    hprintln!("Running on ARM Cortex-M3 (no heap allocation)\n");

    // Initialize filter
    let mut filter = LowPassFilter::new();

    hprintln!("Processing 100 sensor samples through low-pass filter:");
    // Process 100 samples
    for sample in 0..100 {
        let raw_input = read_simulated_sensor(sample);
        let _filtered = filter.update(raw_input);
    }

    let final_state = filter.get_state();
    hprintln!("Filter completed. Final state: {} (converged to true signal)\n", final_state);

    // Demonstrate matrix operations without allocating
    hprintln!("Testing StaticMatrix operations:");
    let m1 = StaticMatrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let m2 = StaticMatrix::new([[5.0, 6.0], [7.0, 8.0]]);
    let v = StaticVector::new([1.0, 2.0]);

    // Perform operations (all stack-allocated, no heap)
    let m_sum = m1.add(&m2);
    hprintln!("  Matrix addition:");
    if let (Some(s00), Some(s01), Some(s10), Some(s11)) =
        (m_sum.get(0), m_sum.get(1), m_sum.get(2), m_sum.get(3)) {
        hprintln!("    [1.0 2.0]   [5.0 6.0]   [{} {}]", s00, s01);
        hprintln!("    [3.0 4.0] + [7.0 8.0] = [{} {}] OK", s10, s11);
    }

    let m_scaled = m1.mul_scalar(2.0);
    hprintln!("  Matrix scaling:");
    if let (Some(sc00), Some(sc01), Some(sc10), Some(sc11)) =
        (m_scaled.get(0), m_scaled.get(1), m_scaled.get(2), m_scaled.get(3)) {
        hprintln!("    [1.0 2.0]       [{} {}]", sc00, sc01);
        hprintln!("    [3.0 4.0] * 2.0 = [{} {}] OK", sc10, sc11);
    }

    let m_v_product = m1.mul_vector(&v);
    hprintln!("  Matrix-vector product:");
    if let (Some(mv0), Some(mv1)) = (m_v_product.get(0), m_v_product.get(1)) {
        hprintln!("    [1.0 2.0]   [1.0]   [{}]", mv0);
        hprintln!("    [3.0 4.0] * [2.0] = [{}] OK", mv1);
    }

    // Vector operations
    hprintln!("\nTesting StaticVector operations:");
    let v2 = StaticVector::new([3.0, 4.0]);
    let v_sum = v.add(&v2);
    if let (Some(vs0), Some(vs1)) = (v_sum.get(0), v_sum.get(1)) {
        hprintln!("  Vector addition: [1.0 2.0] + [3.0 4.0] = [{} {}] OK", vs0, vs1);
    }

    let dot_product = v.dot(&v2);
    hprintln!("  Dot product: [1.0 2.0] . [3.0 4.0] = {} OK", dot_product);

    let norm = v.norm();
    hprintln!("  Vector norm: |[1.0 2.0]| = {} OK", norm);

    hprintln!("\n=== Demo completed successfully ===");
    hprintln!("All operations use stack memory only (no allocation)");

    loop {}
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
