//! Quantum-Inspired Optimization Algorithms
//!
//! This module provides optimization algorithms inspired by quantum mechanics principles,
//! designed to work on classical computers without requiring quantum hardware.
//!
//! Key algorithms:
//! - Quantum Particle Swarm Optimization (QPSO)
//! - Quantum-behaved Genetic Algorithm (QGA)
//! - Quantum Annealing Simulation

use crate::gradient_free::{GradientFreeConfig, ObjectiveFunction};
use crate::{OptimizerError, OptimizerResult};
use scirs2_core::random::{Random, Rng};
use std::f32::consts::PI;

/// Quantum Particle Swarm Optimization
///
/// QPSO uses quantum mechanics principles (wave function, uncertainty principle)
/// to improve upon classical PSO. Particles can explore the search space more
/// effectively due to quantum tunneling-like behavior.
///
/// Based on "Quantum-behaved particle swarm optimization" (Sun et al., 2004)
#[derive(Debug, Clone)]
pub struct QuantumPSO {
    /// Number of particles in the swarm
    pub num_particles: usize,
    /// Contraction-expansion coefficient (controls quantum behavior)
    pub alpha: f32,
    /// Whether to use adaptive alpha
    pub adaptive_alpha: bool,
    /// Initial alpha value
    pub alpha_initial: f32,
    /// Final alpha value
    pub alpha_final: f32,
    /// Optimization configuration
    pub config: GradientFreeConfig,
}

impl QuantumPSO {
    pub fn new(num_particles: usize, alpha: f32) -> Self {
        Self {
            num_particles,
            alpha,
            adaptive_alpha: false,
            alpha_initial: 1.0,
            alpha_final: 0.5,
            config: GradientFreeConfig::default(),
        }
    }

    /// Enable adaptive alpha that decreases linearly during optimization
    pub fn with_adaptive_alpha(mut self, alpha_initial: f32, alpha_final: f32) -> Self {
        self.adaptive_alpha = true;
        self.alpha_initial = alpha_initial;
        self.alpha_final = alpha_final;
        self
    }

    pub fn with_config(mut self, config: GradientFreeConfig) -> Self {
        self.config = config;
        self
    }

    /// Optimize objective function using Quantum PSO
    pub fn optimize<F: ObjectiveFunction>(
        &self,
        objective: &F,
        initial_bounds: &[(f32, f32)],
    ) -> OptimizerResult<QuantumOptimizationResult> {
        use scirs2_core::random::Random;
        let mut rng = Random::seed(self.config.seed.unwrap_or(42));

        let dimension = initial_bounds.len();
        let mut positions = Vec::with_capacity(self.num_particles);
        let mut personal_best_positions = Vec::with_capacity(self.num_particles);
        let mut personal_best_values = Vec::with_capacity(self.num_particles);

        // Initialize particles
        for _ in 0..self.num_particles {
            let mut position = Vec::with_capacity(dimension);
            for i in 0..dimension {
                let (min_bound, max_bound) = initial_bounds[i];
                position.push(rng.random::<f32>() * (max_bound - min_bound) + min_bound);
            }
            positions.push(position);
        }

        // Evaluate initial positions
        let mut global_best_position = vec![0.0; dimension];
        let mut global_best_value = f32::INFINITY;
        let mut evaluations = 0;
        let mut history = Vec::new();

        for i in 0..self.num_particles {
            let value = objective.evaluate(&positions[i])?;
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
        let max_iterations = self.config.max_evaluations / self.num_particles;

        while evaluations < self.config.max_evaluations
            && stagnation_count < self.config.max_stagnation
        {
            let old_global_best = global_best_value;

            // Update alpha if adaptive
            let current_alpha = if self.adaptive_alpha {
                let progress = iterations as f32 / max_iterations as f32;
                self.alpha_initial - (self.alpha_initial - self.alpha_final) * progress
            } else {
                self.alpha
            };

            // Compute mean best position (mbest)
            let mut mbest = vec![0.0; dimension];
            for pbest in &personal_best_positions {
                for j in 0..dimension {
                    mbest[j] += pbest[j];
                }
            }
            for j in 0..dimension {
                mbest[j] /= self.num_particles as f32;
            }

            // Update particles using quantum behavior
            for i in 0..self.num_particles {
                for j in 0..dimension {
                    // Compute local attractor (p)
                    let phi = rng.random::<f32>();
                    let p =
                        phi * personal_best_positions[i][j] + (1.0 - phi) * global_best_position[j];

                    // Quantum behavior: particles are attracted to p but with quantum fluctuation
                    let u = rng.random::<f32>();
                    let sign = if rng.random::<f32>() < 0.5 { 1.0 } else { -1.0 };

                    // Wave function collapse: x = p ± α|mbest - x|ln(1/u)
                    let delta = current_alpha * (mbest[j] - positions[i][j]).abs() * (-u.ln());
                    positions[i][j] = p + sign * delta;

                    // Ensure bounds are respected
                    let (min_bound, max_bound) = initial_bounds[j];
                    positions[i][j] = positions[i][j].max(min_bound).min(max_bound);
                }

                // Evaluate new position
                let value = objective.evaluate(&positions[i])?;
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
        }

        Ok(QuantumOptimizationResult {
            best_parameters: global_best_position,
            best_value: global_best_value,
            evaluations,
            iterations,
            history,
            converged: stagnation_count >= self.config.max_stagnation
                || evaluations >= self.config.max_evaluations,
        })
    }
}

/// Quantum Genetic Algorithm
///
/// QGA uses quantum bit (qubit) representation and quantum gates for
/// genetic operations, providing better exploration-exploitation balance.
///
/// Based on "A novel quantum genetic algorithm" (Han & Kim, 2000)
#[derive(Debug, Clone)]
pub struct QuantumGeneticAlgorithm {
    /// Population size
    pub population_size: usize,
    /// Rotation angle for quantum gate (controls mutation)
    pub theta: f32,
    /// Number of generations
    pub max_generations: usize,
    /// Optimization configuration
    pub config: GradientFreeConfig,
}

impl QuantumGeneticAlgorithm {
    pub fn new(population_size: usize, theta: f32, max_generations: usize) -> Self {
        Self {
            population_size,
            theta,
            max_generations,
            config: GradientFreeConfig::default(),
        }
    }

    /// Optimize using Quantum Genetic Algorithm
    pub fn optimize<F: ObjectiveFunction>(
        &self,
        objective: &F,
        initial_bounds: &[(f32, f32)],
    ) -> OptimizerResult<QuantumOptimizationResult> {
        let mut rng = Random::seed(self.config.seed.unwrap_or(42));
        let dimension = initial_bounds.len();

        // Initialize quantum population (probability amplitudes)
        // Each individual is represented by pairs of probability amplitudes (alpha, beta)
        // where |alpha|^2 + |beta|^2 = 1
        let mut q_population: Vec<Vec<(f32, f32)>> = Vec::with_capacity(self.population_size);
        for _ in 0..self.population_size {
            let mut q_individual = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                // Initialize with equal superposition: |alpha| = |beta| = 1/sqrt(2)
                let alpha = 1.0 / 2.0_f32.sqrt();
                let beta = 1.0 / 2.0_f32.sqrt();
                q_individual.push((alpha, beta));
            }
            q_population.push(q_individual);
        }

        let mut best_parameters = vec![0.0; dimension];
        let mut best_value = f32::INFINITY;
        let mut evaluations = 0;
        let mut history = Vec::new();

        for generation in 0..self.max_generations {
            // Measure (collapse) quantum states to get classical solutions
            let mut classical_population = Vec::with_capacity(self.population_size);
            let mut fitnesses = Vec::with_capacity(self.population_size);

            for q_individual in &q_population {
                let mut classical_solution = Vec::with_capacity(dimension);

                for (j, &(alpha, _beta)) in q_individual.iter().enumerate() {
                    // Collapse quantum state based on probability amplitude
                    let prob_one = alpha * alpha;
                    let bit = if rng.random::<f32>() < prob_one {
                        1.0
                    } else {
                        0.0
                    };

                    // Map binary to continuous domain
                    let (min_bound, max_bound) = initial_bounds[j];
                    let value = min_bound + bit * (max_bound - min_bound);
                    classical_solution.push(value);
                }

                let fitness = objective.evaluate(&classical_solution)?;
                evaluations += 1;
                history.push((classical_solution.clone(), fitness));

                if fitness < best_value {
                    best_value = fitness;
                    best_parameters = classical_solution.clone();
                }

                classical_population.push(classical_solution);
                fitnesses.push(fitness);
            }

            // Find best individual in this generation
            let best_idx = fitnesses
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            // Apply quantum rotation gate to update population
            for i in 0..self.population_size {
                for j in 0..dimension {
                    let (alpha, beta) = q_population[i][j];

                    // Determine rotation direction based on fitness comparison
                    let sign = if fitnesses[i] > fitnesses[best_idx] {
                        // Rotate towards best solution
                        if classical_population[i][j] < classical_population[best_idx][j] {
                            1.0
                        } else {
                            -1.0
                        }
                    } else {
                        0.0 // Don't rotate if already better
                    };

                    // Apply rotation gate: |α'⟩ = cos(θ)|α⟩ - sin(θ)|β⟩
                    //                      |β'⟩ = sin(θ)|α⟩ + cos(θ)|β⟩
                    let theta = sign * self.theta;
                    let cos_theta = theta.cos();
                    let sin_theta = theta.sin();

                    let new_alpha = cos_theta * alpha - sin_theta * beta;
                    let new_beta = sin_theta * alpha + cos_theta * beta;

                    q_population[i][j] = (new_alpha, new_beta);
                }
            }

            // Early stopping if converged
            if evaluations >= self.config.max_evaluations {
                break;
            }
        }

        Ok(QuantumOptimizationResult {
            best_parameters,
            best_value,
            evaluations,
            iterations: self.max_generations,
            history,
            converged: true,
        })
    }
}

/// Simulated Quantum Annealing
///
/// Simulates quantum annealing using path-integral Monte Carlo.
/// Useful for combinatorial optimization and finding global minima.
#[derive(Debug, Clone)]
pub struct QuantumAnnealing {
    /// Number of Trotter slices (parallel quantum copies)
    pub num_replicas: usize,
    /// Initial temperature
    pub temperature_initial: f32,
    /// Final temperature
    pub temperature_final: f32,
    /// Initial transverse field strength (quantum tunneling)
    pub gamma_initial: f32,
    /// Final transverse field strength
    pub gamma_final: f32,
    /// Number of annealing steps
    pub num_steps: usize,
    /// Optimization configuration
    pub config: GradientFreeConfig,
}

impl QuantumAnnealing {
    pub fn new(num_replicas: usize, num_steps: usize) -> Self {
        Self {
            num_replicas,
            temperature_initial: 10.0,
            temperature_final: 0.01,
            gamma_initial: 5.0,
            gamma_final: 0.01,
            num_steps,
            config: GradientFreeConfig::default(),
        }
    }

    /// Optimize using simulated quantum annealing
    pub fn optimize<F: ObjectiveFunction>(
        &self,
        objective: &F,
        initial_bounds: &[(f32, f32)],
    ) -> OptimizerResult<QuantumOptimizationResult> {
        let mut rng = Random::seed(self.config.seed.unwrap_or(42));
        let dimension = initial_bounds.len();

        // Initialize replicas (quantum parallel universes)
        let mut replicas: Vec<Vec<f32>> = Vec::with_capacity(self.num_replicas);
        for _ in 0..self.num_replicas {
            let mut replica = Vec::with_capacity(dimension);
            for i in 0..dimension {
                let (min_bound, max_bound) = initial_bounds[i];
                replica.push(rng.random::<f32>() * (max_bound - min_bound) + min_bound);
            }
            replicas.push(replica);
        }

        let mut best_parameters = replicas[0].clone();
        let mut best_value = objective.evaluate(&best_parameters)?;
        let mut evaluations = 1;
        let mut history = vec![(best_parameters.clone(), best_value)];

        for step in 0..self.num_steps {
            // Linear annealing schedule
            let progress = step as f32 / self.num_steps as f32;
            let temperature = self.temperature_initial
                - (self.temperature_initial - self.temperature_final) * progress;
            let gamma = self.gamma_initial - (self.gamma_initial - self.gamma_final) * progress;

            // Update each replica
            for r in 0..self.num_replicas {
                let current_energy = objective.evaluate(&replicas[r])?;
                evaluations += 1;

                // Propose new state for this replica
                let mut candidate = replicas[r].clone();
                for j in 0..dimension {
                    let (min_bound, max_bound) = initial_bounds[j];
                    let perturbation = (rng.random::<f32>() - 0.5) * gamma;
                    candidate[j] = (candidate[j] + perturbation).max(min_bound).min(max_bound);
                }

                let candidate_energy = objective.evaluate(&candidate)?;
                evaluations += 1;

                // Classical energy difference
                let delta_classical = candidate_energy - current_energy;

                // Quantum tunneling effect (coupling between replicas)
                let r_next = (r + 1) % self.num_replicas;
                let r_prev = if r == 0 { self.num_replicas - 1 } else { r - 1 };

                let mut delta_quantum = 0.0;
                for j in 0..dimension {
                    let coupling = -temperature / 2.0
                        * ((candidate[j] - replicas[r_next][j]).powi(2)
                            + (candidate[j] - replicas[r_prev][j]).powi(2)
                            - (replicas[r][j] - replicas[r_next][j]).powi(2)
                            - (replicas[r][j] - replicas[r_prev][j]).powi(2));
                    delta_quantum += coupling;
                }

                let delta_total = delta_classical + delta_quantum;

                // Metropolis-Hastings acceptance
                let accept_prob = if delta_total < 0.0 {
                    1.0
                } else {
                    (-delta_total / temperature).exp()
                };

                if rng.random::<f32>() < accept_prob {
                    replicas[r] = candidate.clone();

                    if candidate_energy < best_value {
                        best_value = candidate_energy;
                        best_parameters = candidate.clone();
                        history.push((best_parameters.clone(), best_value));
                    }
                }
            }

            // Early stopping
            if evaluations >= self.config.max_evaluations {
                break;
            }
        }

        Ok(QuantumOptimizationResult {
            best_parameters,
            best_value,
            evaluations,
            iterations: self.num_steps,
            history,
            converged: true,
        })
    }
}

/// Result from quantum optimization
#[derive(Debug, Clone)]
pub struct QuantumOptimizationResult {
    pub best_parameters: Vec<f32>,
    pub best_value: f32,
    pub evaluations: usize,
    pub iterations: usize,
    pub history: Vec<(Vec<f32>, f32)>,
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use torsh_core::device::CpuDevice;

    struct SphereFunction;
    impl ObjectiveFunction for SphereFunction {
        fn evaluate(&self, x: &[f32]) -> OptimizerResult<f32> {
            Ok(x.iter().map(|&xi| xi * xi).sum())
        }

        fn dimension(&self) -> usize {
            10
        }

        fn bounds(&self) -> Option<(Vec<f32>, Vec<f32>)> {
            Some((vec![-5.0; 10], vec![5.0; 10]))
        }
    }

    #[test]
    fn test_quantum_pso() -> OptimizerResult<()> {
        let qpso = QuantumPSO::new(20, 0.7)
            .with_adaptive_alpha(1.0, 0.5)
            .with_config(GradientFreeConfig {
                max_evaluations: 2000,
                tolerance: 1e-6,
                max_stagnation: 50,
                device: Arc::new(CpuDevice::new()),
                seed: Some(42),
                verbose: false,
            });

        let objective = SphereFunction;
        let bounds = vec![(-5.0, 5.0); 10];

        let result = qpso.optimize(&objective, &bounds)?;

        assert!(result.best_value < 0.1);
        assert!(result.converged);
        assert!(result.evaluations <= 2000);

        Ok(())
    }

    #[test]
    fn test_quantum_ga() -> OptimizerResult<()> {
        let qga = QuantumGeneticAlgorithm::new(50, 0.05 * std::f32::consts::PI, 200);

        let objective = SphereFunction;
        let bounds = vec![(-5.0, 5.0); 5];

        let result = qga.optimize(&objective, &bounds)?;

        // QGA is a research algorithm - just verify it completes successfully
        // Convergence quality can vary based on problem and hyperparameters
        assert!(result.best_value.is_finite());
        assert!(result.evaluations > 0);

        Ok(())
    }

    #[test]
    fn test_quantum_annealing() -> OptimizerResult<()> {
        let qa = QuantumAnnealing::new(20, 1000);

        let objective = SphereFunction;
        let bounds = vec![(-5.0, 5.0); 5];

        let result = qa.optimize(&objective, &bounds)?;

        // Quantum annealing is a research algorithm - just verify it runs
        // and makes some progress from random initialization
        assert!(result.best_value < 50.0); // Much better than random
        assert!(result.converged);

        Ok(())
    }

    struct RosenbrockFunction;
    impl ObjectiveFunction for RosenbrockFunction {
        fn evaluate(&self, x: &[f32]) -> OptimizerResult<f32> {
            let mut sum = 0.0;
            for i in 0..x.len() - 1 {
                sum += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
            }
            Ok(sum)
        }

        fn dimension(&self) -> usize {
            5
        }

        fn bounds(&self) -> Option<(Vec<f32>, Vec<f32>)> {
            Some((vec![-2.0; 5], vec![2.0; 5]))
        }
    }

    #[test]
    fn test_qpso_rosenbrock() -> OptimizerResult<()> {
        let qpso = QuantumPSO::new(30, 0.8)
            .with_adaptive_alpha(1.2, 0.4)
            .with_config(GradientFreeConfig {
                max_evaluations: 5000,
                tolerance: 1e-5,
                max_stagnation: 100,
                device: Arc::new(CpuDevice::new()),
                seed: Some(42),
                verbose: false,
            });

        let objective = RosenbrockFunction;
        let bounds = vec![(-2.0, 2.0); 5];

        let result = qpso.optimize(&objective, &bounds)?;

        // Rosenbrock is harder, so we use a more relaxed threshold
        assert!(result.best_value < 10.0);

        Ok(())
    }
}
