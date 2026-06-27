//! Stack-allocated PID controller for a first-order system.
//!
//! This example demonstrates:
//! - No heap allocation (no `alloc` feature needed)
//! - PID control law using only stack variables
//! - Matrix-vector multiplication for system dynamics
//! - Real-time control loop
//!
//! Run with: `cargo run --example pid_controller`

use rustebra::matrix::StaticMatrix;

/// A simple first-order system: dx/dt = -0.1*x + u, y = x
struct System {
    state: f32, // x
}

impl System {
    fn new(initial_state: f32) -> Self {
        Self {
            state: initial_state,
        }
    }

    /// Simulate one step with input u, return output y
    fn step(&mut self, u: f32, dt: f32) -> f32 {
        let a = -0.1;
        let b = 1.0;
        self.state = self.state + dt * (a * self.state + b * u);
        self.state
    }
}

/// Stack-allocated PID controller
struct PidController {
    kp: f32,
    ki: f32,
    kd: f32,
    integral: f32,
    prev_error: f32,
}

impl PidController {
    fn new(kp: f32, ki: f32, kd: f32) -> Self {
        Self {
            kp,
            ki,
            kd,
            integral: 0.0,
            prev_error: 0.0,
        }
    }

    /// Compute control input: u = Kp*e + Ki*∫e + Kd*de/dt
    fn compute(&mut self, setpoint: f32, measured: f32, dt: f32) -> f32 {
        let error = setpoint - measured;
        self.integral += error * dt;
        let derivative = (error - self.prev_error) / dt;
        self.prev_error = error;

        self.kp * error + self.ki * self.integral + self.kd * derivative
    }
}

fn main() {
    println!("=== PID Controller Example ===\n");

    let mut system = System::new(0.0);
    let mut controller = PidController::new(1.0, 0.1, 0.5);

    let setpoint = 5.0;
    let dt = 0.01;
    let steps = 500;

    println!("Setpoint: {}", setpoint);
    println!("Time(s)\t\tOutput\t\tControl");
    println!("{}", "-".repeat(45));

    for step in 0..steps {
        let time = step as f32 * dt;
        let measured = system.state;
        let control_input = controller.compute(setpoint, measured, dt);
        let _output = system.step(control_input, dt);

        if step % 100 == 0 {
            println!("{:.2}\t\t{:.3}\t\t{:.3}", time, system.state, control_input);
        }
    }

    println!(
        "\nFinal state: {:.3} (target: {:.3})",
        system.state, setpoint
    );

    // Demonstrate matrix operations: system state transition matrix
    let a_matrix = StaticMatrix::new([[0.999], [0.001]]);
    let b_matrix = StaticMatrix::new([[0.01]]);
    println!(
        "\nState transition demo (no allocation):\n  A = {:?}\n  B = {:?}",
        a_matrix, b_matrix
    );
}
