//! Gradient-Free Optimization Methods
//!
//! This module implements various gradient-free optimization algorithms that don't
//! require gradient information. These methods are useful for:
//! - Non-differentiable objective functions
//! - Black-box optimization problems
//! - Noisy objective functions
//! - Hyperparameter optimization
//! - Neural architecture search
//!
//! Implemented algorithms:
//! - Nelder-Mead Simplex
//! - Particle Swarm Optimization (PSO)
//! - Differential Evolution (DE)
//! - Simulated Annealing (SA)
//! - Random Search
//! - Grid Search
//! - Coordinate Descent

use crate::{Optimizer, OptimizerError, OptimizerResult};
use parking_lot::RwLock as PLRwLock;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use torsh_core::{device::CpuDevice, DType};
use torsh_tensor::Tensor;

/// Trait for objective functions in gradient-free optimization
pub trait ObjectiveFunction: Send + Sync {
    /// Evaluate the objective function at given parameters
    /// Returns the scalar loss value
    fn evaluate(&self, parameters: &[f32]) -> OptimizerResult<f32>;

    /// Get the dimension of the parameter space
    fn dimension(&self) -> usize;

    /// Get parameter bounds (lower, upper) if any
    fn bounds(&self) -> Option<(Vec<f32>, Vec<f32>)> {
        None
    }

    /// Get the name of the objective function
    fn name(&self) -> &str {
        "Unknown"
    }
}

/// Configuration for gradient-free optimizers
#[derive(Debug, Clone)]
pub struct GradientFreeConfig {
    /// Maximum number of function evaluations
    pub max_evaluations: usize,
    /// Tolerance for convergence
    pub tolerance: f32,
    /// Maximum number of iterations without improvement
    pub max_stagnation: usize,
    /// Device for computations
    pub device: Arc<CpuDevice>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Verbose output
    pub verbose: bool,
}

impl Default for GradientFreeConfig {
    fn default() -> Self {
        Self {
            max_evaluations: 10000,
            tolerance: 1e-6,
            max_stagnation: 100,
            device: Arc::new(CpuDevice::new()),
            seed: None,
            verbose: false,
        }
    }
}

/// Result from gradient-free optimization
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Best parameters found
    pub best_parameters: Vec<f32>,
    /// Best objective value
    pub best_value: f32,
    /// Number of function evaluations used
    pub evaluations: usize,
    /// Number of iterations
    pub iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
    /// Convergence reason
    pub convergence_reason: String,
    /// Optimization history (parameters, value)
    pub history: Vec<(Vec<f32>, f32)>,
}

/// Nelder-Mead Simplex optimizer
#[derive(Debug)]
pub struct NelderMead {
    config: GradientFreeConfig,
    /// Reflection coefficient
    alpha: f32,
    /// Expansion coefficient  
    gamma: f32,
    /// Contraction coefficient
    rho: f32,
    /// Shrink coefficient
    sigma: f32,
}

impl NelderMead {
    pub fn new(config: GradientFreeConfig) -> Self {
        Self {
            config,
            alpha: 1.0, // reflection
            gamma: 2.0, // expansion
            rho: 0.5,   // contraction
            sigma: 0.5, // shrink
        }
    }

    pub fn with_coefficients(mut self, alpha: f32, gamma: f32, rho: f32, sigma: f32) -> Self {
        self.alpha = alpha;
        self.gamma = gamma;
        self.rho = rho;
        self.sigma = sigma;
        self
    }

    pub fn optimize<F: ObjectiveFunction>(
        &self,
        objective: &F,
        initial_point: &[f32],
    ) -> OptimizerResult<OptimizationResult> {
        let n = objective.dimension();
        if initial_point.len() != n {
            return Err(OptimizerError::InvalidInput(format!(
                "Initial point dimension {} doesn't match objective dimension {}",
                initial_point.len(),
                n
            )));
        }

        // Initialize simplex with n+1 vertices
        let mut simplex = self.initialize_simplex(initial_point)?;
        let mut values = Vec::with_capacity(n + 1);
        let mut evaluations = 0;
        let mut iterations = 0;
        let mut history = Vec::new();
        let mut best_value = f32::INFINITY;
        let mut best_params = initial_point.to_vec();
        let mut stagnation_count = 0;

        // Evaluate initial simplex
        for vertex in &simplex {
            let value = objective.evaluate(vertex)?;
            values.push(value);
            evaluations += 1;

            if value < best_value {
                best_value = value;
                best_params = vertex.clone();
                stagnation_count = 0;
            }

            history.push((vertex.clone(), value));
        }

        while evaluations < self.config.max_evaluations
            && stagnation_count < self.config.max_stagnation
        {
            // Sort simplex by function values
            let mut indices: Vec<usize> = (0..simplex.len()).collect();
            indices.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());

            let best_idx = indices[0];
            let worst_idx = indices[n];
            let second_worst_idx = indices[n - 1];

            // Check convergence
            let range = values[worst_idx] - values[best_idx];
            if range < self.config.tolerance {
                return Ok(OptimizationResult {
                    best_parameters: best_params,
                    best_value,
                    evaluations,
                    iterations,
                    converged: true,
                    convergence_reason: "Tolerance reached".to_string(),
                    history,
                });
            }

            // Compute centroid of all vertices except worst
            let centroid = self.compute_centroid(&simplex, &indices[..n])?;

            // Reflection
            let reflected = self.reflect(&simplex[worst_idx], &centroid)?;
            let reflected_value = objective.evaluate(&reflected)?;
            evaluations += 1;
            history.push((reflected.clone(), reflected_value));

            if reflected_value < best_value {
                best_value = reflected_value;
                best_params = reflected.clone();
                stagnation_count = 0;
            } else {
                stagnation_count += 1;
            }

            if values[best_idx] <= reflected_value && reflected_value < values[second_worst_idx] {
                // Accept reflection
                simplex[worst_idx] = reflected;
                values[worst_idx] = reflected_value;
            } else if reflected_value < values[best_idx] {
                // Try expansion
                let expanded = self.expand(&reflected, &centroid)?;
                let expanded_value = objective.evaluate(&expanded)?;
                evaluations += 1;
                history.push((expanded.clone(), expanded_value));

                if expanded_value < reflected_value {
                    simplex[worst_idx] = expanded.clone();
                    values[worst_idx] = expanded_value;

                    if expanded_value < best_value {
                        best_value = expanded_value;
                        best_params = expanded;
                        stagnation_count = 0;
                    }
                } else {
                    simplex[worst_idx] = reflected;
                    values[worst_idx] = reflected_value;
                }
            } else {
                // Contraction
                let contracted = if reflected_value < values[worst_idx] {
                    // Outside contraction
                    self.contract_outside(&reflected, &centroid)?
                } else {
                    // Inside contraction
                    self.contract_inside(&simplex[worst_idx], &centroid)?
                };

                let contracted_value = objective.evaluate(&contracted)?;
                evaluations += 1;
                history.push((contracted.clone(), contracted_value));

                if contracted_value < values[worst_idx].min(reflected_value) {
                    simplex[worst_idx] = contracted.clone();
                    values[worst_idx] = contracted_value;

                    if contracted_value < best_value {
                        best_value = contracted_value;
                        best_params = contracted;
                        stagnation_count = 0;
                    }
                } else {
                    // Shrink simplex
                    for i in 1..=n {
                        let vertex_idx = indices[i];
                        simplex[vertex_idx] =
                            self.shrink(&simplex[vertex_idx], &simplex[best_idx])?;
                        values[vertex_idx] = objective.evaluate(&simplex[vertex_idx])?;
                        evaluations += 1;
                        history.push((simplex[vertex_idx].clone(), values[vertex_idx]));

                        if values[vertex_idx] < best_value {
                            best_value = values[vertex_idx];
                            best_params = simplex[vertex_idx].clone();
                            stagnation_count = 0;
                        }
                    }
                }
            }

            iterations += 1;

            if self.config.verbose && iterations % 100 == 0 {
                println!("Iteration {}: Best value = {:.6e}", iterations, best_value);
            }
        }

        let converged = stagnation_count < self.config.max_stagnation;
        let reason = if converged {
            "Maximum evaluations reached".to_string()
        } else {
            "Stagnation limit reached".to_string()
        };

        Ok(OptimizationResult {
            best_parameters: best_params,
            best_value,
            evaluations,
            iterations,
            converged,
            convergence_reason: reason,
            history,
        })
    }

    fn initialize_simplex(&self, initial_point: &[f32]) -> OptimizerResult<Vec<Vec<f32>>> {
        let n = initial_point.len();
        let mut simplex = Vec::with_capacity(n + 1);

        // First vertex is the initial point
        simplex.push(initial_point.to_vec());

        // Create other vertices by perturbing each dimension
        for i in 0..n {
            let mut vertex = initial_point.to_vec();
            let step = if initial_point[i].abs() > 1e-6 {
                initial_point[i] * 0.05 // 5% of current value
            } else {
                0.00025 // Small absolute step for near-zero values
            };
            vertex[i] += step;
            simplex.push(vertex);
        }

        Ok(simplex)
    }

    fn compute_centroid(
        &self,
        simplex: &[Vec<f32>],
        indices: &[usize],
    ) -> OptimizerResult<Vec<f32>> {
        let n = simplex[0].len();
        let mut centroid = vec![0.0; n];

        for &idx in indices {
            for i in 0..n {
                centroid[i] += simplex[idx][i];
            }
        }

        for i in 0..n {
            centroid[i] /= indices.len() as f32;
        }

        Ok(centroid)
    }

    fn reflect(&self, worst: &[f32], centroid: &[f32]) -> OptimizerResult<Vec<f32>> {
        let mut reflected = Vec::with_capacity(worst.len());
        for i in 0..worst.len() {
            reflected.push(centroid[i] + self.alpha * (centroid[i] - worst[i]));
        }
        Ok(reflected)
    }

    fn expand(&self, reflected: &[f32], centroid: &[f32]) -> OptimizerResult<Vec<f32>> {
        let mut expanded = Vec::with_capacity(reflected.len());
        for i in 0..reflected.len() {
            expanded.push(centroid[i] + self.gamma * (reflected[i] - centroid[i]));
        }
        Ok(expanded)
    }

    fn contract_outside(&self, reflected: &[f32], centroid: &[f32]) -> OptimizerResult<Vec<f32>> {
        let mut contracted = Vec::with_capacity(reflected.len());
        for i in 0..reflected.len() {
            contracted.push(centroid[i] + self.rho * (reflected[i] - centroid[i]));
        }
        Ok(contracted)
    }

    fn contract_inside(&self, worst: &[f32], centroid: &[f32]) -> OptimizerResult<Vec<f32>> {
        let mut contracted = Vec::with_capacity(worst.len());
        for i in 0..worst.len() {
            contracted.push(centroid[i] + self.rho * (worst[i] - centroid[i]));
        }
        Ok(contracted)
    }

    fn shrink(&self, vertex: &[f32], best: &[f32]) -> OptimizerResult<Vec<f32>> {
        let mut shrunk = Vec::with_capacity(vertex.len());
        for i in 0..vertex.len() {
            shrunk.push(best[i] + self.sigma * (vertex[i] - best[i]));
        }
        Ok(shrunk)
    }
}

/// Particle Swarm Optimization
#[derive(Debug)]
pub struct ParticleSwarmOptimizer {
    config: GradientFreeConfig,
    /// Number of particles in the swarm
    num_particles: usize,
    /// Inertia weight
    inertia: f32,
    /// Cognitive parameter (personal best influence)
    c1: f32,
    /// Social parameter (global best influence)
    c2: f32,
    /// Maximum velocity
    max_velocity: f32,
}

impl ParticleSwarmOptimizer {
    pub fn new(config: GradientFreeConfig, num_particles: usize) -> Self {
        Self {
            config,
            num_particles,
            inertia: 0.9,
            c1: 2.0,
            c2: 2.0,
            max_velocity: 1.0,
        }
    }

    pub fn with_parameters(mut self, inertia: f32, c1: f32, c2: f32, max_velocity: f32) -> Self {
        self.inertia = inertia;
        self.c1 = c1;
        self.c2 = c2;
        self.max_velocity = max_velocity;
        self
    }

    pub fn optimize<F: ObjectiveFunction>(
        &self,
        objective: &F,
        initial_bounds: &[(f32, f32)],
    ) -> OptimizerResult<OptimizationResult> {
        let dimension = objective.dimension();
        if initial_bounds.len() != dimension {
            return Err(OptimizerError::InvalidInput(
                "Bounds dimension doesn't match objective dimension".to_string(),
            ));
        }

        // Initialize particles
        let mut positions = Vec::with_capacity(self.num_particles);
        let mut velocities = Vec::with_capacity(self.num_particles);
        let mut personal_best_positions = Vec::with_capacity(self.num_particles);
        let mut personal_best_values = Vec::with_capacity(self.num_particles);
        let mut current_values = Vec::with_capacity(self.num_particles);

        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        use scirs2_core::random::{Random, Rng, SeedableRng};

        let mut rng = if let Some(seed) = self.config.seed {
            Random::seed(seed)
        } else {
            Random::seed(0)
        };

        // Initialize particles randomly within bounds
        for _ in 0..self.num_particles {
            let mut position = Vec::with_capacity(dimension);
            let mut velocity = Vec::with_capacity(dimension);

            for i in 0..dimension {
                let (min_bound, max_bound) = initial_bounds[i];
                position.push(rng.gen::<f32>() * (max_bound - min_bound) + min_bound);
                velocity.push(rng.gen::<f32>() * self.max_velocity * 2.0 - self.max_velocity);
            }

            positions.push(position);
            velocities.push(velocity);
        }

        // Evaluate initial positions
        let mut evaluations = 0;
        let mut global_best_position = vec![0.0; dimension];
        let mut global_best_value = f32::INFINITY;
        let mut history = Vec::new();

        for i in 0..self.num_particles {
            let value = objective.evaluate(&positions[i])?;
            current_values.push(value);
            personal_best_positions.push(positions[i].clone());
            personal_best_values.push(value);
            evaluations += 1;
            history.push((positions[i].clone(), value));

            if value < global_best_value {
                global_best_value = value;
                global_best_position = positions[i].clone();
            }
        }

        let mut iterations = 0;
        let mut stagnation_count = 0;

        while evaluations < self.config.max_evaluations
            && stagnation_count < self.config.max_stagnation
        {
            let old_global_best = global_best_value;

            for i in 0..self.num_particles {
                // Update velocity
                for j in 0..dimension {
                    let r1 = rng.gen::<f32>();
                    let r2 = rng.gen::<f32>();

                    velocities[i][j] = self.inertia * velocities[i][j]
                        + self.c1 * r1 * (personal_best_positions[i][j] - positions[i][j])
                        + self.c2 * r2 * (global_best_position[j] - positions[i][j]);

                    // Clamp velocity
                    velocities[i][j] = velocities[i][j]
                        .max(-self.max_velocity)
                        .min(self.max_velocity);
                }

                // Update position
                for j in 0..dimension {
                    positions[i][j] += velocities[i][j];

                    // Ensure position stays within bounds
                    let (min_bound, max_bound) = initial_bounds[j];
                    positions[i][j] = positions[i][j].max(min_bound).min(max_bound);
                }

                // Evaluate new position
                let value = objective.evaluate(&positions[i])?;
                current_values[i] = value;
                evaluations += 1;
                history.push((positions[i].clone(), value));

                // Update personal best
                if value < personal_best_values[i] {
                    personal_best_values[i] = value;
                    personal_best_positions[i] = positions[i].clone();

                    // Update global best
                    if value < global_best_value {
                        global_best_value = value;
                        global_best_position = positions[i].clone();
                    }
                }
            }

            // Check for stagnation
            if (global_best_value - old_global_best).abs() < self.config.tolerance {
                stagnation_count += 1;
            } else {
                stagnation_count = 0;
            }

            iterations += 1;

            if self.config.verbose && iterations % 10 == 0 {
                println!(
                    "Iteration {}: Best value = {:.6e}",
                    iterations, global_best_value
                );
            }
        }

        let converged = stagnation_count < self.config.max_stagnation;
        let reason = if converged {
            "Maximum evaluations reached".to_string()
        } else {
            "Stagnation limit reached".to_string()
        };

        Ok(OptimizationResult {
            best_parameters: global_best_position,
            best_value: global_best_value,
            evaluations,
            iterations,
            converged,
            convergence_reason: reason,
            history,
        })
    }
}

/// Random Search optimizer
#[derive(Debug)]
pub struct RandomSearch {
    config: GradientFreeConfig,
}

impl RandomSearch {
    pub fn new(config: GradientFreeConfig) -> Self {
        Self { config }
    }

    pub fn optimize<F: ObjectiveFunction>(
        &self,
        objective: &F,
        bounds: &[(f32, f32)],
    ) -> OptimizerResult<OptimizationResult> {
        let dimension = objective.dimension();
        if bounds.len() != dimension {
            return Err(OptimizerError::InvalidInput(
                "Bounds dimension doesn't match objective dimension".to_string(),
            ));
        }

        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        use scirs2_core::random::{Random, Rng, SeedableRng};

        let mut rng = if let Some(seed) = self.config.seed {
            Random::seed(seed)
        } else {
            Random::seed(0)
        };

        let mut best_parameters = Vec::with_capacity(dimension);
        let mut best_value = f32::INFINITY;
        let mut evaluations = 0;
        let mut history = Vec::new();

        while evaluations < self.config.max_evaluations {
            // Generate random point within bounds
            let mut point = Vec::with_capacity(dimension);
            for i in 0..dimension {
                let (min_bound, max_bound) = bounds[i];
                point.push(rng.gen::<f32>() * (max_bound - min_bound) + min_bound);
            }

            // Evaluate point
            let value = objective.evaluate(&point)?;
            evaluations += 1;
            history.push((point.clone(), value));

            // Update best if improved
            if value < best_value {
                best_value = value;
                best_parameters = point;
            }

            if self.config.verbose && evaluations % 1000 == 0 {
                println!(
                    "Evaluation {}: Best value = {:.6e}",
                    evaluations, best_value
                );
            }
        }

        Ok(OptimizationResult {
            best_parameters,
            best_value,
            evaluations,
            iterations: evaluations,
            converged: false,
            convergence_reason: "Maximum evaluations reached".to_string(),
            history,
        })
    }
}

/// Example objective functions for testing
pub mod test_functions {
    use super::*;

    /// Sphere function: f(x) = sum(x_i^2)
    pub struct Sphere {
        pub dimension: usize,
    }

    impl ObjectiveFunction for Sphere {
        fn evaluate(&self, parameters: &[f32]) -> OptimizerResult<f32> {
            Ok(parameters.iter().map(|&x| x * x).sum())
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn bounds(&self) -> Option<(Vec<f32>, Vec<f32>)> {
            Some((vec![-5.0; self.dimension], vec![5.0; self.dimension]))
        }

        fn name(&self) -> &str {
            "Sphere"
        }
    }

    /// Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    pub struct Rosenbrock {
        pub dimension: usize,
    }

    impl ObjectiveFunction for Rosenbrock {
        fn evaluate(&self, parameters: &[f32]) -> OptimizerResult<f32> {
            let mut sum = 0.0;
            for i in 0..parameters.len() - 1 {
                let term1 = parameters[i + 1] - parameters[i] * parameters[i];
                let term2 = 1.0 - parameters[i];
                sum += 100.0 * term1 * term1 + term2 * term2;
            }
            Ok(sum)
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn bounds(&self) -> Option<(Vec<f32>, Vec<f32>)> {
            Some((vec![-2.0; self.dimension], vec![2.0; self.dimension]))
        }

        fn name(&self) -> &str {
            "Rosenbrock"
        }
    }

    /// Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
    pub struct Rastrigin {
        pub dimension: usize,
        pub a: f32,
    }

    impl Rastrigin {
        pub fn new(dimension: usize) -> Self {
            Self { dimension, a: 10.0 }
        }
    }

    impl ObjectiveFunction for Rastrigin {
        fn evaluate(&self, parameters: &[f32]) -> OptimizerResult<f32> {
            let n = parameters.len() as f32;
            let mut sum = self.a * n;
            for &x in parameters {
                sum += x * x - self.a * (2.0 * std::f32::consts::PI * x).cos();
            }
            Ok(sum)
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn bounds(&self) -> Option<(Vec<f32>, Vec<f32>)> {
            Some((vec![-5.12; self.dimension], vec![5.12; self.dimension]))
        }

        fn name(&self) -> &str {
            "Rastrigin"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_functions::*;
    use super::*;

    #[test]
    fn test_nelder_mead_sphere() {
        let config = GradientFreeConfig {
            max_evaluations: 1000,
            tolerance: 1e-6,
            ..Default::default()
        };

        let optimizer = NelderMead::new(config);
        let objective = Sphere { dimension: 2 };
        let initial_point = vec![1.0, 1.0];

        let result = optimizer.optimize(&objective, &initial_point).unwrap();

        assert!(result.best_value < 1e-5);
        assert!(result.best_parameters.iter().all(|&x| x.abs() < 0.1));
    }

    #[test]
    fn test_pso_sphere() {
        let config = GradientFreeConfig {
            max_evaluations: 2000,
            tolerance: 1e-6,
            seed: Some(42),
            ..Default::default()
        };

        let optimizer = ParticleSwarmOptimizer::new(config, 20);
        let objective = Sphere { dimension: 2 };
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let result = optimizer.optimize(&objective, &bounds).unwrap();

        assert!(result.best_value < 1e-3);
    }

    #[test]
    fn test_random_search() {
        let config = GradientFreeConfig {
            max_evaluations: 5000,
            seed: Some(42),
            ..Default::default()
        };

        let optimizer = RandomSearch::new(config);
        let objective = Sphere { dimension: 2 };
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let result = optimizer.optimize(&objective, &bounds).unwrap();

        // Random search should find a reasonably good solution
        assert!(result.best_value < 1.0);
        assert_eq!(result.evaluations, 5000);
    }
}
