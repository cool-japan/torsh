//! Neural ODE integration with automatic differentiation
//!
//! This module provides comprehensive support for Neural Ordinary Differential Equations
//! with automatic differentiation using the adjoint method. Neural ODEs represent
//! continuous-depth neural networks and are particularly useful for modeling
//! dynamical systems and time series data.

use crate::error_handling::{AutogradError, AutogradResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Trait for ODE system functions
pub trait ODESystem: Send + Sync {
    /// Compute the derivative dy/dt = f(t, y, params)
    fn compute_derivative(
        &self,
        t: f64,
        y: &[f64],
        params: &[f64],
        dy_dt: &mut [f64],
    ) -> AutogradResult<()>;

    /// Get the system dimension (number of state variables)
    fn dimension(&self) -> usize;

    /// Get the number of parameters
    fn parameter_count(&self) -> usize;

    /// Optional: Compute Jacobian with respect to state variables
    fn compute_jacobian_y(
        &self,
        t: f64,
        y: &[f64],
        params: &[f64],
        jacobian: &mut [f64],
    ) -> AutogradResult<()> {
        // Default implementation using finite differences
        self.compute_jacobian_finite_difference(t, y, params, jacobian, true)
    }

    /// Optional: Compute Jacobian with respect to parameters
    fn compute_jacobian_params(
        &self,
        t: f64,
        y: &[f64],
        params: &[f64],
        jacobian: &mut [f64],
    ) -> AutogradResult<()> {
        // Default implementation using finite differences
        self.compute_jacobian_finite_difference(t, y, params, jacobian, false)
    }

    /// Helper method for finite difference Jacobian computation
    fn compute_jacobian_finite_difference(
        &self,
        t: f64,
        y: &[f64],
        params: &[f64],
        jacobian: &mut [f64],
        wrt_y: bool, // true for dy/dy, false for dy/dparams
    ) -> AutogradResult<()> {
        let eps = 1e-8;
        let dim = self.dimension();
        let param_count = self.parameter_count();

        let target_size = if wrt_y { dim } else { param_count };

        let mut dy_dt = vec![0.0; dim];
        let mut dy_dt_pert = vec![0.0; dim];
        let mut y_pert = y.to_vec();
        let mut params_pert = params.to_vec();

        for i in 0..target_size {
            // Compute f(t, y + eps, params) or f(t, y, params + eps)
            if wrt_y {
                y_pert[i] += eps;
                self.compute_derivative(t, &y_pert, params, &mut dy_dt_pert)?;
                y_pert[i] = y[i]; // Reset
            } else {
                params_pert[i] += eps;
                self.compute_derivative(t, y, &params_pert, &mut dy_dt_pert)?;
                params_pert[i] = params[i]; // Reset
            }

            // Compute f(t, y, params)
            self.compute_derivative(t, y, params, &mut dy_dt)?;

            // Compute finite difference
            for j in 0..dim {
                jacobian[j * target_size + i] = (dy_dt_pert[j] - dy_dt[j]) / eps;
            }
        }

        Ok(())
    }
}

/// Neural ODE layer that implements a continuous-depth neural network
#[derive(Debug, Clone)]
pub struct NeuralODELayer {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    params: Vec<f64>,
    param_names: Vec<String>,
}

impl NeuralODELayer {
    /// Create a new Neural ODE layer
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        // Initialize parameters for a simple neural network f(t, y)
        // Architecture: input -> hidden -> output
        let input_to_hidden = input_dim * hidden_dim + hidden_dim; // weights + biases
        let hidden_to_output = hidden_dim * output_dim + output_dim; // weights + biases
        let total_params = input_to_hidden + hidden_to_output;

        let mut params = vec![0.0; total_params];
        let mut param_names = Vec::new();

        // Initialize with small random values (simplified initialization)
        for (i, param) in params.iter_mut().enumerate() {
            *param = (i as f64 * 0.1) % 2.0 - 1.0; // Simple initialization
        }

        // Generate parameter names
        for i in 0..input_dim {
            for j in 0..hidden_dim {
                param_names.push(format!("w_input_{}_{}", i, j));
            }
        }
        for j in 0..hidden_dim {
            param_names.push(format!("b_hidden_{}", j));
        }
        for i in 0..hidden_dim {
            for j in 0..output_dim {
                param_names.push(format!("w_hidden_{}_{}", i, j));
            }
        }
        for j in 0..output_dim {
            param_names.push(format!("b_output_{}", j));
        }

        Self {
            input_dim,
            hidden_dim,
            output_dim,
            params,
            param_names,
        }
    }

    /// Get mutable reference to parameters
    pub fn parameters_mut(&mut self) -> &mut [f64] {
        &mut self.params
    }

    /// Get parameters
    pub fn parameters(&self) -> &[f64] {
        &self.params
    }

    /// Get parameter names
    pub fn parameter_names(&self) -> &[String] {
        &self.param_names
    }

    /// Apply activation function (tanh for stability)
    fn activation(&self, x: f64) -> f64 {
        x.tanh()
    }

    /// Apply activation derivative
    fn activation_derivative(&self, x: f64) -> f64 {
        let tanh_x = x.tanh();
        1.0 - tanh_x * tanh_x
    }
}

impl ODESystem for NeuralODELayer {
    fn compute_derivative(
        &self,
        _t: f64, // Time parameter (can be used for time-dependent dynamics)
        y: &[f64],
        params: &[f64],
        dy_dt: &mut [f64],
    ) -> AutogradResult<()> {
        if y.len() != self.input_dim {
            return Err(AutogradError::shape_mismatch(
                "neural_ode_compute_derivative",
                vec![self.input_dim],
                vec![y.len()],
            ));
        }

        if dy_dt.len() != self.output_dim {
            return Err(AutogradError::shape_mismatch(
                "neural_ode_compute_derivative",
                vec![self.output_dim],
                vec![dy_dt.len()],
            ));
        }

        // Forward pass: input -> hidden -> output
        let mut hidden = vec![0.0; self.hidden_dim];

        // Input to hidden layer
        let input_to_hidden_weights = self.input_dim * self.hidden_dim;
        for j in 0..self.hidden_dim {
            let mut sum = 0.0;
            for i in 0..self.input_dim {
                sum += y[i] * params[i * self.hidden_dim + j];
            }
            // Add bias
            sum += params[input_to_hidden_weights + j];
            hidden[j] = self.activation(sum);
        }

        // Hidden to output layer
        let hidden_to_output_start = input_to_hidden_weights + self.hidden_dim;
        for j in 0..self.output_dim {
            let mut sum = 0.0;
            for i in 0..self.hidden_dim {
                sum += hidden[i] * params[hidden_to_output_start + i * self.output_dim + j];
            }
            // Add bias
            let bias_start = hidden_to_output_start + self.hidden_dim * self.output_dim;
            sum += params[bias_start + j];
            dy_dt[j] = sum; // Linear output for stability
        }

        Ok(())
    }

    fn dimension(&self) -> usize {
        self.output_dim
    }

    fn parameter_count(&self) -> usize {
        self.params.len()
    }
}

/// ODE solver configuration
#[derive(Debug, Clone)]
pub struct ODESolverConfig {
    /// Integration method
    pub method: IntegrationMethod,
    /// Relative tolerance
    pub rtol: f64,
    /// Absolute tolerance
    pub atol: f64,
    /// Initial step size
    pub initial_step_size: f64,
    /// Maximum step size
    pub max_step_size: f64,
    /// Maximum number of steps
    pub max_steps: usize,
    /// Enable adaptive step size
    pub adaptive: bool,
}

impl Default for ODESolverConfig {
    fn default() -> Self {
        Self {
            method: IntegrationMethod::RungeKutta4,
            rtol: 1e-6,
            atol: 1e-8,
            initial_step_size: 0.01,
            max_step_size: 0.1,
            max_steps: 10000,
            adaptive: true,
        }
    }
}

/// Available integration methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrationMethod {
    Euler,
    RungeKutta4,
    AdaptiveRungeKutta,
    DormandPrince,
}

/// ODE solver for Neural ODEs
pub struct ODESolver {
    config: ODESolverConfig,
}

impl ODESolver {
    /// Create a new ODE solver
    pub fn new(config: ODESolverConfig) -> Self {
        Self { config }
    }

    /// Solve ODE from t0 to t1 with initial condition y0
    pub fn solve(
        &self,
        system: &dyn ODESystem,
        t0: f64,
        t1: f64,
        y0: &[f64],
        params: &[f64],
    ) -> AutogradResult<ODESolution> {
        match self.config.method {
            IntegrationMethod::Euler => self.solve_euler(system, t0, t1, y0, params),
            IntegrationMethod::RungeKutta4 => self.solve_rk4(system, t0, t1, y0, params),
            IntegrationMethod::AdaptiveRungeKutta => {
                self.solve_adaptive_rk4(system, t0, t1, y0, params)
            }
            IntegrationMethod::DormandPrince => {
                self.solve_dormand_prince(system, t0, t1, y0, params)
            }
        }
    }

    /// Euler method implementation
    fn solve_euler(
        &self,
        system: &dyn ODESystem,
        t0: f64,
        t1: f64,
        y0: &[f64],
        params: &[f64],
    ) -> AutogradResult<ODESolution> {
        let dim = system.dimension();
        let mut t = t0;
        let mut y = y0.to_vec();
        let mut solution = ODESolution::new(dim);
        solution.add_point(t, &y);

        let mut step_size = self.config.initial_step_size;
        let mut steps = 0;
        let mut dy_dt = vec![0.0; dim];

        while t < t1 && steps < self.config.max_steps {
            if t + step_size > t1 {
                step_size = t1 - t;
            }

            // Compute derivative
            system.compute_derivative(t, &y, params, &mut dy_dt)?;

            // Euler step: y_{n+1} = y_n + h * f(t_n, y_n)
            for i in 0..dim {
                y[i] += step_size * dy_dt[i];
            }

            t += step_size;
            steps += 1;

            solution.add_point(t, &y);
        }

        if steps >= self.config.max_steps {
            return Err(AutogradError::gradient_computation(
                "ode_solve_euler",
                format!(
                    "Maximum number of steps ({}) exceeded",
                    self.config.max_steps
                ),
            ));
        }

        Ok(solution)
    }

    /// Runge-Kutta 4th order method implementation
    fn solve_rk4(
        &self,
        system: &dyn ODESystem,
        t0: f64,
        t1: f64,
        y0: &[f64],
        params: &[f64],
    ) -> AutogradResult<ODESolution> {
        let dim = system.dimension();
        let mut t = t0;
        let mut y = y0.to_vec();
        let mut solution = ODESolution::new(dim);
        solution.add_point(t, &y);

        let mut step_size = self.config.initial_step_size;
        let mut steps = 0;
        let mut k1 = vec![0.0; dim];
        let mut k2 = vec![0.0; dim];
        let mut k3 = vec![0.0; dim];
        let mut k4 = vec![0.0; dim];
        let mut y_temp = vec![0.0; dim];

        while t < t1 && steps < self.config.max_steps {
            if t + step_size > t1 {
                step_size = t1 - t;
            }

            // k1 = h * f(t, y)
            system.compute_derivative(t, &y, params, &mut k1)?;
            for i in 0..dim {
                k1[i] *= step_size;
            }

            // k2 = h * f(t + h/2, y + k1/2)
            for i in 0..dim {
                y_temp[i] = y[i] + k1[i] * 0.5;
            }
            system.compute_derivative(t + step_size * 0.5, &y_temp, params, &mut k2)?;
            for i in 0..dim {
                k2[i] *= step_size;
            }

            // k3 = h * f(t + h/2, y + k2/2)
            for i in 0..dim {
                y_temp[i] = y[i] + k2[i] * 0.5;
            }
            system.compute_derivative(t + step_size * 0.5, &y_temp, params, &mut k3)?;
            for i in 0..dim {
                k3[i] *= step_size;
            }

            // k4 = h * f(t + h, y + k3)
            for i in 0..dim {
                y_temp[i] = y[i] + k3[i];
            }
            system.compute_derivative(t + step_size, &y_temp, params, &mut k4)?;
            for i in 0..dim {
                k4[i] *= step_size;
            }

            // y_{n+1} = y_n + (k1 + 2*k2 + 2*k3 + k4) / 6
            for i in 0..dim {
                y[i] += (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
            }

            t += step_size;
            steps += 1;

            solution.add_point(t, &y);
        }

        if steps >= self.config.max_steps {
            return Err(AutogradError::gradient_computation(
                "ode_solve_rk4",
                format!(
                    "Maximum number of steps ({}) exceeded",
                    self.config.max_steps
                ),
            ));
        }

        Ok(solution)
    }

    /// Adaptive Runge-Kutta method with error control
    fn solve_adaptive_rk4(
        &self,
        system: &dyn ODESystem,
        t0: f64,
        t1: f64,
        y0: &[f64],
        params: &[f64],
    ) -> AutogradResult<ODESolution> {
        let dim = system.dimension();
        let mut t = t0;
        let mut y = y0.to_vec();
        let mut solution = ODESolution::new(dim);
        solution.add_point(t, &y);

        let mut step_size = self.config.initial_step_size;
        let mut steps = 0;

        while t < t1 && steps < self.config.max_steps {
            if t + step_size > t1 {
                step_size = t1 - t;
            }

            // Compute one full step
            let y_full = self.rk4_step(system, t, &y, params, step_size)?;

            // Compute two half steps
            let y_half1 = self.rk4_step(system, t, &y, params, step_size * 0.5)?;
            let y_full_from_half = self.rk4_step(
                system,
                t + step_size * 0.5,
                &y_half1,
                params,
                step_size * 0.5,
            )?;

            // Estimate error
            let mut error = 0.0;
            for i in 0..dim {
                let diff = y_full_from_half[i] - y_full[i];
                error += diff * diff;
            }
            error = error.sqrt() / dim as f64;

            // Check if error is acceptable
            let tolerance = self.config.atol
                + self.config.rtol * y.iter().map(|&x| x.abs()).fold(0.0, f64::max);

            if error < tolerance {
                // Accept step
                y = y_full_from_half; // Use the more accurate estimate
                t += step_size;
                steps += 1;
                solution.add_point(t, &y);

                // Increase step size if error is very small
                if self.config.adaptive && error < tolerance * 0.1 {
                    step_size = (step_size * 1.5).min(self.config.max_step_size);
                }
            } else {
                // Reject step and reduce step size
                if self.config.adaptive {
                    step_size *= 0.5;
                    if step_size < 1e-12 {
                        return Err(AutogradError::numerical_instability(
                            "adaptive_rk4",
                            step_size,
                            "Step size became too small",
                        ));
                    }
                }
            }
        }

        if steps >= self.config.max_steps {
            return Err(AutogradError::gradient_computation(
                "ode_solve_adaptive_rk4",
                format!(
                    "Maximum number of steps ({}) exceeded",
                    self.config.max_steps
                ),
            ));
        }

        Ok(solution)
    }

    /// Single RK4 step
    fn rk4_step(
        &self,
        system: &dyn ODESystem,
        t: f64,
        y: &[f64],
        params: &[f64],
        h: f64,
    ) -> AutogradResult<Vec<f64>> {
        let dim = system.dimension();
        let mut k1 = vec![0.0; dim];
        let mut k2 = vec![0.0; dim];
        let mut k3 = vec![0.0; dim];
        let mut k4 = vec![0.0; dim];
        let mut y_temp = vec![0.0; dim];
        let mut result = vec![0.0; dim];

        // k1
        system.compute_derivative(t, y, params, &mut k1)?;

        // k2
        for i in 0..dim {
            y_temp[i] = y[i] + h * 0.5 * k1[i];
        }
        system.compute_derivative(t + h * 0.5, &y_temp, params, &mut k2)?;

        // k3
        for i in 0..dim {
            y_temp[i] = y[i] + h * 0.5 * k2[i];
        }
        system.compute_derivative(t + h * 0.5, &y_temp, params, &mut k3)?;

        // k4
        for i in 0..dim {
            y_temp[i] = y[i] + h * k3[i];
        }
        system.compute_derivative(t + h, &y_temp, params, &mut k4)?;

        // Final result
        for i in 0..dim {
            result[i] = y[i] + h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        }

        Ok(result)
    }

    /// Dormand-Prince method (placeholder - simplified implementation)
    fn solve_dormand_prince(
        &self,
        system: &dyn ODESystem,
        t0: f64,
        t1: f64,
        y0: &[f64],
        params: &[f64],
    ) -> AutogradResult<ODESolution> {
        // For simplicity, use adaptive RK4
        // In a full implementation, this would use the actual Dormand-Prince coefficients
        self.solve_adaptive_rk4(system, t0, t1, y0, params)
    }
}

/// Solution container for ODE integration
#[derive(Debug, Clone)]
pub struct ODESolution {
    pub times: Vec<f64>,
    pub states: Vec<Vec<f64>>,
    pub dimension: usize,
}

impl ODESolution {
    /// Create a new ODE solution container
    pub fn new(dimension: usize) -> Self {
        Self {
            times: Vec::new(),
            states: Vec::new(),
            dimension,
        }
    }

    /// Add a solution point
    pub fn add_point(&mut self, t: f64, y: &[f64]) {
        self.times.push(t);
        self.states.push(y.to_vec());
    }

    /// Get the final state
    pub fn final_state(&self) -> Option<&[f64]> {
        self.states.last().map(|v| v.as_slice())
    }

    /// Get the final time
    pub fn final_time(&self) -> Option<f64> {
        self.times.last().copied()
    }

    /// Get the number of time steps
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// Check if solution is empty
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Interpolate solution at a given time (linear interpolation)
    pub fn interpolate_at(&self, t: f64) -> AutogradResult<Vec<f64>> {
        if self.times.is_empty() {
            return Err(AutogradError::gradient_computation(
                "interpolate_at",
                "Empty solution",
            ));
        }

        if self.times.len() == 1 {
            return Ok(self.states[0].clone());
        }

        // Find the interval containing t
        if t <= self.times[0] {
            return Ok(self.states[0].clone());
        }

        if t >= self.times[self.times.len() - 1] {
            return Ok(self.states[self.states.len() - 1].clone());
        }

        for i in 0..self.times.len() - 1 {
            if t >= self.times[i] && t <= self.times[i + 1] {
                let t0 = self.times[i];
                let t1 = self.times[i + 1];
                let y0 = &self.states[i];
                let y1 = &self.states[i + 1];

                let alpha = (t - t0) / (t1 - t0);
                let mut result = vec![0.0; self.dimension];

                for j in 0..self.dimension {
                    result[j] = y0[j] * (1.0 - alpha) + y1[j] * alpha;
                }

                return Ok(result);
            }
        }

        Err(AutogradError::gradient_computation(
            "interpolate_at",
            format!("Time {} not found in solution", t),
        ))
    }
}

/// Adjoint method for computing gradients through ODE solutions
pub struct AdjointMethod {
    solver: ODESolver,
}

impl AdjointMethod {
    /// Create a new adjoint method solver
    pub fn new(solver: ODESolver) -> Self {
        Self { solver }
    }

    /// Compute gradients using the adjoint method
    pub fn compute_gradients(
        &self,
        system: &dyn ODESystem,
        t0: f64,
        t1: f64,
        y0: &[f64],
        params: &[f64],
        loss_gradient: &[f64], // gradient of loss w.r.t. final state
    ) -> AutogradResult<AdjointSolution> {
        // Step 1: Forward pass - solve the ODE
        let forward_solution = self.solver.solve(system, t0, t1, y0, params)?;

        // Step 2: Backward pass - solve the adjoint equations
        let adjoint_solution =
            self.solve_adjoint_equations(system, &forward_solution, loss_gradient, params)?;

        Ok(adjoint_solution)
    }

    /// Solve the adjoint equations backward in time
    fn solve_adjoint_equations(
        &self,
        system: &dyn ODESystem,
        forward_solution: &ODESolution,
        loss_gradient: &[f64],
        params: &[f64],
    ) -> AutogradResult<AdjointSolution> {
        let dim = system.dimension();
        let param_count = system.parameter_count();

        // Create the augmented adjoint system
        let adjoint_system = AdjointSystem::new(system, forward_solution, params);

        // Initial conditions for adjoint variables
        // λ(T) = ∂L/∂y(T)
        // ∂L/∂p accumulated = 0 initially
        let mut adjoint_y0 = loss_gradient.to_vec();
        adjoint_y0.extend(vec![0.0; param_count]); // Initialize parameter gradients to zero

        // Solve adjoint equations backward from t1 to t0
        let t0 = forward_solution.times[0];
        let t1 = forward_solution.final_time().unwrap();

        let adjoint_solution_raw = self
            .solver
            .solve(&adjoint_system, t1, t0, &adjoint_y0, &[])?;

        // Extract gradients
        let final_adjoint_state = adjoint_solution_raw.final_state().unwrap();
        let y0_gradient = final_adjoint_state[0..dim].to_vec();
        let param_gradient = final_adjoint_state[dim..dim + param_count].to_vec();

        Ok(AdjointSolution {
            y0_gradient,
            param_gradient,
            forward_solution: forward_solution.clone(),
            adjoint_trajectory: adjoint_solution_raw,
        })
    }
}

/// System for solving adjoint equations
struct AdjointSystem<'a> {
    original_system: &'a dyn ODESystem,
    forward_solution: &'a ODESolution,
    params: &'a [f64],
    jacobian_y_cache: Arc<Mutex<HashMap<String, Vec<f64>>>>,
    jacobian_p_cache: Arc<Mutex<HashMap<String, Vec<f64>>>>,
}

impl<'a> AdjointSystem<'a> {
    fn new(
        original_system: &'a dyn ODESystem,
        forward_solution: &'a ODESolution,
        params: &'a [f64],
    ) -> Self {
        Self {
            original_system,
            forward_solution,
            params,
            jacobian_y_cache: Arc::new(Mutex::new(HashMap::new())),
            jacobian_p_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get cached or compute Jacobian with respect to y
    fn get_jacobian_y(&self, t: f64, y: &[f64]) -> AutogradResult<Vec<f64>> {
        let key = format!("{:.10}_{:?}", t, y);

        // Check cache first
        if let Ok(cache) = self.jacobian_y_cache.lock() {
            if let Some(jacobian) = cache.get(&key) {
                return Ok(jacobian.clone());
            }
        }

        let dim = self.original_system.dimension();
        let mut jacobian = vec![0.0; dim * dim];
        self.original_system
            .compute_jacobian_y(t, y, self.params, &mut jacobian)?;

        // Store in cache
        if let Ok(mut cache) = self.jacobian_y_cache.lock() {
            cache.insert(key, jacobian.clone());
        }
        Ok(jacobian)
    }

    /// Get cached or compute Jacobian with respect to parameters
    fn get_jacobian_p(&self, t: f64, y: &[f64]) -> AutogradResult<Vec<f64>> {
        let key = format!("{:.10}_{:?}", t, y);

        // Check cache first
        if let Ok(cache) = self.jacobian_p_cache.lock() {
            if let Some(jacobian) = cache.get(&key) {
                return Ok(jacobian.clone());
            }
        }

        let dim = self.original_system.dimension();
        let param_count = self.original_system.parameter_count();
        let mut jacobian = vec![0.0; dim * param_count];
        self.original_system
            .compute_jacobian_params(t, y, self.params, &mut jacobian)?;

        // Store in cache
        if let Ok(mut cache) = self.jacobian_p_cache.lock() {
            cache.insert(key, jacobian.clone());
        }
        Ok(jacobian)
    }
}

impl<'a> ODESystem for AdjointSystem<'a> {
    fn compute_derivative(
        &self,
        t: f64,
        adjoint_state: &[f64], // [λ, ∂L/∂p]
        _params: &[f64],       // Not used for adjoint system
        d_adjoint_dt: &mut [f64],
    ) -> AutogradResult<()> {
        let dim = self.original_system.dimension();
        let param_count = self.original_system.parameter_count();

        if adjoint_state.len() != dim + param_count {
            return Err(AutogradError::shape_mismatch(
                "adjoint_compute_derivative",
                vec![dim + param_count],
                vec![adjoint_state.len()],
            ));
        }

        let lambda = &adjoint_state[0..dim];
        let param_grad_accum = &adjoint_state[dim..dim + param_count];

        // Interpolate forward solution at time t
        let y_forward = self.forward_solution.interpolate_at(t)?;

        // Get Jacobians
        let jacobian_y = self.get_jacobian_y(t, &y_forward)?;
        let jacobian_p = self.get_jacobian_p(t, &y_forward)?;

        // Compute adjoint derivatives
        // dλ/dt = -λᵀ * ∂f/∂y (note: negative sign for backward time)
        for i in 0..dim {
            let mut sum = 0.0;
            for j in 0..dim {
                sum += lambda[j] * jacobian_y[j * dim + i];
            }
            d_adjoint_dt[i] = -sum;
        }

        // Compute parameter gradient accumulation
        // d(∂L/∂p)/dt = -λᵀ * ∂f/∂p
        for i in 0..param_count {
            let mut sum = 0.0;
            for j in 0..dim {
                sum += lambda[j] * jacobian_p[j * param_count + i];
            }
            d_adjoint_dt[dim + i] = -sum;
        }

        Ok(())
    }

    fn dimension(&self) -> usize {
        self.original_system.dimension() + self.original_system.parameter_count()
    }

    fn parameter_count(&self) -> usize {
        0 // Adjoint system doesn't have additional parameters
    }
}

/// Solution from adjoint method computation
#[derive(Debug, Clone)]
pub struct AdjointSolution {
    /// Gradient with respect to initial conditions
    pub y0_gradient: Vec<f64>,
    /// Gradient with respect to parameters
    pub param_gradient: Vec<f64>,
    /// Forward solution
    pub forward_solution: ODESolution,
    /// Adjoint trajectory
    pub adjoint_trajectory: ODESolution,
}

/// Neural ODE module that combines the ODE layer with automatic differentiation
pub struct NeuralODE {
    ode_layer: NeuralODELayer,
    solver: ODESolver,
    adjoint_method: AdjointMethod,
    integration_time: (f64, f64),
}

impl NeuralODE {
    /// Create a new Neural ODE module
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        solver_config: ODESolverConfig,
        integration_time: (f64, f64),
    ) -> Self {
        let ode_layer = NeuralODELayer::new(input_dim, hidden_dim, output_dim);
        let solver = ODESolver::new(solver_config);
        let adjoint_method = AdjointMethod::new(ODESolver::new(solver.config.clone()));

        Self {
            ode_layer,
            solver,
            adjoint_method,
            integration_time,
        }
    }

    /// Forward pass through Neural ODE
    pub fn forward(&self, input: &[f64]) -> AutogradResult<Vec<f64>> {
        let (t0, t1) = self.integration_time;
        let solution =
            self.solver
                .solve(&self.ode_layer, t0, t1, input, self.ode_layer.parameters())?;

        Ok(solution.final_state().unwrap().to_vec())
    }

    /// Backward pass using adjoint method
    pub fn backward(
        &self,
        input: &[f64],
        output_gradient: &[f64],
    ) -> AutogradResult<NeuralODEGradients> {
        let (t0, t1) = self.integration_time;
        let adjoint_solution = self.adjoint_method.compute_gradients(
            &self.ode_layer,
            t0,
            t1,
            input,
            self.ode_layer.parameters(),
            output_gradient,
        )?;

        Ok(NeuralODEGradients {
            input_gradient: adjoint_solution.y0_gradient,
            parameter_gradients: adjoint_solution.param_gradient,
        })
    }

    /// Get mutable reference to parameters for optimization
    pub fn parameters_mut(&mut self) -> &mut [f64] {
        self.ode_layer.parameters_mut()
    }

    /// Get parameters
    pub fn parameters(&self) -> &[f64] {
        self.ode_layer.parameters()
    }

    /// Set integration time
    pub fn set_integration_time(&mut self, t0: f64, t1: f64) {
        self.integration_time = (t0, t1);
    }
}

/// Gradients computed by Neural ODE backward pass
#[derive(Debug, Clone)]
pub struct NeuralODEGradients {
    /// Gradient with respect to input
    pub input_gradient: Vec<f64>,
    /// Gradients with respect to parameters
    pub parameter_gradients: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple test ODE system: dy/dt = -k*y (exponential decay)
    struct ExponentialDecaySystem {
        decay_constant: f64,
    }

    impl ExponentialDecaySystem {
        fn new(k: f64) -> Self {
            Self { decay_constant: k }
        }
    }

    impl ODESystem for ExponentialDecaySystem {
        fn compute_derivative(
            &self,
            _t: f64,
            y: &[f64],
            _params: &[f64],
            dy_dt: &mut [f64],
        ) -> AutogradResult<()> {
            dy_dt[0] = -self.decay_constant * y[0];
            Ok(())
        }

        fn dimension(&self) -> usize {
            1
        }

        fn parameter_count(&self) -> usize {
            0
        }
    }

    #[test]
    fn test_ode_solver_euler() {
        let system = ExponentialDecaySystem::new(1.0);
        let config = ODESolverConfig {
            method: IntegrationMethod::Euler,
            initial_step_size: 0.01,
            ..Default::default()
        };
        let solver = ODESolver::new(config);

        let solution = solver.solve(&system, 0.0, 1.0, &[1.0], &[]).unwrap();

        // Check that solution decays (approximately exponential decay)
        let final_state = solution.final_state().unwrap();
        assert!(final_state[0] < 1.0);
        assert!(final_state[0] > 0.0);
    }

    #[test]
    fn test_ode_solver_rk4() {
        let system = ExponentialDecaySystem::new(1.0);
        let config = ODESolverConfig {
            method: IntegrationMethod::RungeKutta4,
            initial_step_size: 0.1,
            ..Default::default()
        };
        let solver = ODESolver::new(config);

        let solution = solver.solve(&system, 0.0, 2.0, &[1.0], &[]).unwrap();

        // RK4 should be more accurate than Euler
        let final_state = solution.final_state().unwrap();
        let expected = (-2.0_f64).exp(); // Analytical solution: e^(-k*t)
        let error = (final_state[0] - expected).abs();

        assert!(
            error < 0.1,
            "RK4 error too large: {} vs expected {}",
            final_state[0],
            expected
        );
    }

    #[test]
    fn test_neural_ode_layer() {
        let layer = NeuralODELayer::new(2, 4, 2);
        assert_eq!(layer.dimension(), 2);
        assert_eq!(layer.input_dim, 2);
        assert_eq!(layer.hidden_dim, 4);
        assert_eq!(layer.output_dim, 2);

        let input = vec![1.0, 0.5];
        let mut output = vec![0.0, 0.0];
        let result = layer.compute_derivative(0.0, &input, layer.parameters(), &mut output);

        assert!(result.is_ok());
        // Output should be non-zero for non-zero input
        assert!(output[0].abs() > 1e-10 || output[1].abs() > 1e-10);
    }

    #[test]
    fn test_neural_ode_forward() {
        let neural_ode = NeuralODE::new(
            2, // input_dim
            4, // hidden_dim
            2, // output_dim
            ODESolverConfig::default(),
            (0.0, 1.0), // integration time
        );

        let input = vec![1.0, 0.5];
        let output = neural_ode.forward(&input).unwrap();

        assert_eq!(output.len(), 2);
        // Output should be different from input due to ODE integration
        assert!((output[0] - input[0]).abs() > 1e-10 || (output[1] - input[1]).abs() > 1e-10);
    }

    #[test]
    fn test_ode_solution_interpolation() {
        let mut solution = ODESolution::new(1);
        solution.add_point(0.0, &[1.0]);
        solution.add_point(1.0, &[0.5]);
        solution.add_point(2.0, &[0.25]);

        // Test interpolation at midpoint
        let interpolated = solution.interpolate_at(0.5).unwrap();
        assert!((interpolated[0] - 0.75).abs() < 1e-10);

        // Test interpolation at exact point
        let interpolated = solution.interpolate_at(1.0).unwrap();
        assert!((interpolated[0] - 0.5).abs() < 1e-10);

        // Test extrapolation (should clamp)
        let interpolated = solution.interpolate_at(3.0).unwrap();
        assert!((interpolated[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_solver_config_presets() {
        let default_config = ODESolverConfig::default();
        assert_eq!(default_config.method, IntegrationMethod::RungeKutta4);
        assert!(default_config.adaptive);

        // Test that all methods are available
        let methods = [
            IntegrationMethod::Euler,
            IntegrationMethod::RungeKutta4,
            IntegrationMethod::AdaptiveRungeKutta,
            IntegrationMethod::DormandPrince,
        ];

        for method in &methods {
            let config = ODESolverConfig {
                method: *method,
                ..Default::default()
            };
            assert_eq!(config.method, *method);
        }
    }

    #[test]
    fn test_neural_ode_parameter_access() {
        let mut neural_ode = NeuralODE::new(2, 3, 2, ODESolverConfig::default(), (0.0, 1.0));

        let params = neural_ode.parameters();
        let param_count = params.len();
        assert!(param_count > 0);

        // Modify parameters
        let params_mut = neural_ode.parameters_mut();
        params_mut[0] = 42.0;

        assert_eq!(neural_ode.parameters()[0], 42.0);
    }
}
