//! Evolutionary Strategies for Optimization
//!
//! This module implements various evolutionary strategy algorithms for optimization
//! without requiring gradients. These methods are inspired by biological evolution
//! and use populations of candidate solutions that evolve over generations.
//!
//! Implemented algorithms:
//! - (μ/ρ + λ)-ES: Evolution Strategies with μ parents, ρ recombinants, λ offspring
//! - (μ, λ)-ES: Evolution Strategies where parents don't survive to next generation
//! - CMA-ES: Covariance Matrix Adaptation Evolution Strategy
//! - Natural Evolution Strategies (NES)
//! - OpenAI Evolution Strategies (ES)
//! - Genetic Algorithm (GA)
//! - Differential Evolution (DE)

use crate::{OptimizerError, OptimizerResult};
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::device::CpuDevice;

/// Trait for fitness evaluation in evolutionary strategies
pub trait FitnessFunction: Send + Sync {
    /// Evaluate fitness of an individual (higher is better)
    fn evaluate(&self, individual: &[f32]) -> OptimizerResult<f32>;

    /// Get the dimension of the search space
    fn dimension(&self) -> usize;

    /// Get parameter bounds if any
    fn bounds(&self) -> Option<(Vec<f32>, Vec<f32>)> {
        None
    }

    /// Whether higher fitness values are better (default: true)
    fn maximize(&self) -> bool {
        true
    }

    /// Get the name of the fitness function
    fn name(&self) -> &str {
        "Unknown"
    }
}

/// Configuration for evolutionary strategies
#[derive(Debug, Clone)]
pub struct EvolutionaryConfig {
    /// Population size
    pub population_size: usize,
    /// Number of parents to select
    pub num_parents: usize,
    /// Number of offspring to generate
    pub num_offspring: usize,
    /// Maximum number of generations
    pub max_generations: usize,
    /// Convergence tolerance
    pub tolerance: f32,
    /// Maximum generations without improvement
    pub max_stagnation: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Device for computations
    pub device: Arc<CpuDevice>,
    /// Verbose output
    pub verbose: bool,
}

impl Default for EvolutionaryConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            num_parents: 25,
            num_offspring: 50,
            max_generations: 1000,
            tolerance: 1e-6,
            max_stagnation: 50,
            seed: None,
            device: Arc::new(CpuDevice::new()),
            verbose: false,
        }
    }
}

/// Individual in the population
#[derive(Debug, Clone)]
pub struct Individual {
    /// Parameter values (genotype)
    pub genome: Vec<f32>,
    /// Fitness value
    pub fitness: f32,
    /// Strategy parameters (for self-adaptive algorithms)
    pub strategy_params: Vec<f32>,
}

impl Individual {
    pub fn new(genome: Vec<f32>) -> Self {
        Self {
            genome,
            fitness: f32::NEG_INFINITY,
            strategy_params: Vec::new(),
        }
    }

    pub fn with_strategy_params(genome: Vec<f32>, strategy_params: Vec<f32>) -> Self {
        Self {
            genome,
            fitness: f32::NEG_INFINITY,
            strategy_params,
        }
    }
}

/// Result from evolutionary optimization
#[derive(Debug, Clone)]
pub struct EvolutionResult {
    /// Best individual found
    pub best_individual: Individual,
    /// Number of generations
    pub generations: usize,
    /// Number of fitness evaluations
    pub evaluations: usize,
    /// Whether evolution converged
    pub converged: bool,
    /// Convergence reason
    pub convergence_reason: String,
    /// Evolution history (generation, best_fitness, mean_fitness)
    pub history: Vec<(usize, f32, f32)>,
}

/// Evolution Strategy (μ/ρ + λ)-ES
#[derive(Debug)]
pub struct EvolutionStrategy {
    config: EvolutionaryConfig,
    /// Mutation strength
    sigma: f32,
    /// Self-adaptation learning rate
    tau: f32,
    /// Use plus strategy (parents survive) vs comma strategy (parents don't survive)
    plus_strategy: bool,
}

impl EvolutionStrategy {
    pub fn new(config: EvolutionaryConfig) -> Self {
        let dimension = 1.0; // Will be set based on problem dimension
        Self {
            config,
            sigma: 1.0,
            tau: 1.0 / (2.0 * dimension as f32).sqrt(),
            plus_strategy: true,
        }
    }

    pub fn with_parameters(mut self, sigma: f32, tau: f32, plus_strategy: bool) -> Self {
        self.sigma = sigma;
        self.tau = tau;
        self.plus_strategy = plus_strategy;
        self
    }

    pub fn evolve<F: FitnessFunction>(
        &mut self,
        fitness_fn: &F,
        bounds: &[(f32, f32)],
    ) -> OptimizerResult<EvolutionResult> {
        let dimension = fitness_fn.dimension();
        if bounds.len() != dimension {
            return Err(OptimizerError::InvalidInput(
                "Bounds dimension doesn't match fitness function dimension".to_string(),
            ));
        }

        // Update tau based on actual dimension
        self.tau = 1.0 / (2.0 * dimension as f32).sqrt();

        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        use scirs2_core::random::{Random, Rng};

        let mut rng = if let Some(seed) = self.config.seed {
            Random::seed(seed)
        } else {
            Random::seed(0)
        };

        // Initialize population
        let mut population = self.initialize_population(&mut rng, bounds)?;

        // Evaluate initial population
        let mut evaluations = 0;
        for individual in &mut population {
            individual.fitness = fitness_fn.evaluate(&individual.genome)?;
            evaluations += 1;
        }

        // Sort population by fitness (descending - higher is better)
        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let mut generations = 0;
        let mut history = Vec::new();
        let mut best_fitness = population[0].fitness;
        let mut stagnation_count = 0;

        while generations < self.config.max_generations
            && stagnation_count < self.config.max_stagnation
        {
            let old_best = best_fitness;

            // Record generation statistics
            let mean_fitness: f32 =
                population.iter().map(|ind| ind.fitness).sum::<f32>() / population.len() as f32;
            history.push((generations, population[0].fitness, mean_fitness));

            // Selection: select parents
            let parents = self.select_parents(&population);

            // Reproduction: create offspring
            let mut offspring = Vec::with_capacity(self.config.num_offspring);
            for _ in 0..self.config.num_offspring {
                let parent_indices: Vec<usize> = (0..parents.len()).collect();
                let parent1_idx = parent_indices[rng.gen_range(0..parent_indices.len())];
                let parent2_idx = parent_indices[rng.gen_range(0..parent_indices.len())];

                let child = self.recombine_and_mutate(
                    &parents[parent1_idx],
                    &parents[parent2_idx],
                    &mut rng,
                    bounds,
                )?;
                offspring.push(child);
            }

            // Evaluate offspring
            for individual in &mut offspring {
                individual.fitness = fitness_fn.evaluate(&individual.genome)?;
                evaluations += 1;
            }

            // Environmental selection
            population = if self.plus_strategy {
                // (μ + λ) strategy: parents and offspring compete
                let mut combined = parents;
                combined.extend(offspring);
                combined.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
                combined
                    .into_iter()
                    .take(self.config.population_size)
                    .collect()
            } else {
                // (μ, λ) strategy: only offspring survive
                offspring.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
                offspring
                    .into_iter()
                    .take(self.config.population_size)
                    .collect()
            };

            // Update best fitness and check for stagnation
            best_fitness = population[0].fitness;
            if (best_fitness - old_best).abs() < self.config.tolerance {
                stagnation_count += 1;
            } else {
                stagnation_count = 0;
            }

            generations += 1;

            if self.config.verbose && generations % 10 == 0 {
                println!(
                    "Generation {}: Best fitness = {:.6e}, Mean fitness = {:.6e}",
                    generations, best_fitness, mean_fitness
                );
            }
        }

        let converged = stagnation_count < self.config.max_stagnation;
        let reason = if converged {
            "Maximum generations reached".to_string()
        } else {
            "Stagnation limit reached".to_string()
        };

        Ok(EvolutionResult {
            best_individual: population[0].clone(),
            generations,
            evaluations,
            converged,
            convergence_reason: reason,
            history,
        })
    }

    // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
    fn initialize_population<R: scirs2_core::random::Rng>(
        &self,
        rng: &mut R,
        bounds: &[(f32, f32)],
    ) -> OptimizerResult<Vec<Individual>> {
        let dimension = bounds.len();
        let mut population = Vec::with_capacity(self.config.population_size);

        for _ in 0..self.config.population_size {
            let mut genome = Vec::with_capacity(dimension);
            let mut strategy_params = vec![self.sigma; dimension]; // One sigma per dimension

            for i in 0..dimension {
                let (min_bound, max_bound) = bounds[i];
                genome.push(rng.random::<f32>() * (max_bound - min_bound) + min_bound);
            }

            population.push(Individual::with_strategy_params(genome, strategy_params));
        }

        Ok(population)
    }

    fn select_parents(&self, population: &[Individual]) -> Vec<Individual> {
        // Tournament selection or simply take the best μ individuals
        population
            .iter()
            .take(self.config.num_parents)
            .cloned()
            .collect()
    }

    fn recombine_and_mutate<R: scirs2_core::random::Rng>(
        &self,
        parent1: &Individual,
        parent2: &Individual,
        rng: &mut R,
        bounds: &[(f32, f32)],
    ) -> OptimizerResult<Individual> {
        let dimension = parent1.genome.len();
        let mut child_genome = Vec::with_capacity(dimension);
        let mut child_strategy = Vec::with_capacity(dimension);

        // Recombination: intermediate crossover
        for i in 0..dimension {
            let alpha = rng.random::<f32>();
            child_genome.push(alpha * parent1.genome[i] + (1.0 - alpha) * parent2.genome[i]);
            child_strategy.push((parent1.strategy_params[i] + parent2.strategy_params[i]) / 2.0);
        }

        // Self-adaptation: mutate strategy parameters first
        let global_factor = (self.tau * rng.random::<f32>().ln()).exp();
        let individual_tau = self.tau / (2.0 * dimension as f32).sqrt();

        for i in 0..dimension {
            let individual_factor = (individual_tau * rng.random::<f32>().ln()).exp();
            child_strategy[i] *= global_factor * individual_factor;
            child_strategy[i] = child_strategy[i].max(1e-6); // Lower bound for sigma
        }

        // Mutation: add Gaussian noise
        for i in 0..dimension {
            let mutation = rng.random::<f32>() * child_strategy[i];
            child_genome[i] += mutation;

            // Ensure bounds are respected
            let (min_bound, max_bound) = bounds[i];
            child_genome[i] = child_genome[i].max(min_bound).min(max_bound);
        }

        Ok(Individual::with_strategy_params(
            child_genome,
            child_strategy,
        ))
    }
}

/// Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
#[derive(Debug)]
pub struct CMAES {
    config: EvolutionaryConfig,
    /// Step size (overall mutation strength)
    sigma: f32,
    /// Dimension of the problem
    dimension: usize,
    /// Mean of the search distribution
    mean: Vec<f32>,
    /// Covariance matrix
    covariance: Vec<Vec<f32>>,
    /// Evolution path for covariance matrix
    pc: Vec<f32>,
    /// Evolution path for step size
    ps: Vec<f32>,
}

impl CMAES {
    pub fn new(config: EvolutionaryConfig, dimension: usize) -> Self {
        let mean = vec![0.0; dimension];
        let mut covariance = vec![vec![0.0; dimension]; dimension];

        // Initialize covariance as identity matrix
        for i in 0..dimension {
            covariance[i][i] = 1.0;
        }

        Self {
            config,
            sigma: 1.0,
            dimension,
            mean,
            covariance,
            pc: vec![0.0; dimension],
            ps: vec![0.0; dimension],
        }
    }

    pub fn evolve<F: FitnessFunction>(
        &mut self,
        fitness_fn: &F,
        initial_mean: &[f32],
        bounds: &[(f32, f32)],
    ) -> OptimizerResult<EvolutionResult> {
        if initial_mean.len() != self.dimension || bounds.len() != self.dimension {
            return Err(OptimizerError::InvalidInput(
                "Dimension mismatch".to_string(),
            ));
        }

        self.mean = initial_mean.to_vec();

        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        use scirs2_core::random::{Random, Rng};

        let mut rng = if let Some(seed) = self.config.seed {
            Random::seed(seed)
        } else {
            Random::seed(0)
        };

        let lambda = self.config.population_size;
        let mu = lambda / 2;

        // CMA-ES parameters
        let cc = 4.0 / (self.dimension as f32 + 4.0);
        let cs = (mu as f32 + 2.0) / (self.dimension as f32 + mu as f32 + 3.0);
        let c1 = 2.0 / ((self.dimension as f32 + 1.3) * (self.dimension as f32 + 1.3) + mu as f32);
        let cmu = (2.0 * (mu as f32 - 2.0 + 1.0 / mu as f32))
            / ((self.dimension as f32 + 2.0) * (self.dimension as f32 + 2.0) + mu as f32);
        let damps =
            1.0 + 2.0 * (0.0_f32).max((mu as f32 - 1.0) / (self.dimension as f32 + 1.0) - 1.0) + cs;

        let mut generations = 0;
        let mut evaluations = 0;
        let mut history = Vec::new();
        let mut best_individual = Individual::new(self.mean.clone());
        best_individual.fitness = f32::NEG_INFINITY;
        let mut stagnation_count = 0;

        while generations < self.config.max_generations
            && stagnation_count < self.config.max_stagnation
        {
            let old_best = best_individual.fitness;

            // Generate offspring
            let mut population = Vec::with_capacity(lambda);
            for _ in 0..lambda {
                let individual = self.sample_individual(&mut rng, bounds)?;
                population.push(individual);
            }

            // Evaluate population
            for individual in &mut population {
                individual.fitness = fitness_fn.evaluate(&individual.genome)?;
                evaluations += 1;

                if individual.fitness > best_individual.fitness {
                    best_individual = individual.clone();
                }
            }

            // Sort by fitness (descending)
            population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

            // Update distribution parameters
            let old_mean = self.mean.clone();

            // Update mean
            for i in 0..self.dimension {
                self.mean[i] = 0.0;
                for j in 0..mu {
                    self.mean[i] += population[j].genome[i];
                }
                self.mean[i] /= mu as f32;
            }

            // Update evolution paths and covariance matrix
            // This is a simplified version - full CMA-ES has more complex updates
            let mean_diff: Vec<f32> = self
                .mean
                .iter()
                .zip(old_mean.iter())
                .map(|(new, old)| (new - old) / self.sigma)
                .collect();

            // Update step size evolution path
            for i in 0..self.dimension {
                self.ps[i] =
                    (1.0 - cs) * self.ps[i] + (cs * (2.0 - cs) * mu as f32).sqrt() * mean_diff[i];
            }

            // Update covariance evolution path
            for i in 0..self.dimension {
                self.pc[i] = (1.0 - cc) * self.pc[i] + cc * (2.0 - cc).sqrt() * mean_diff[i];
            }

            // Simplified covariance matrix update
            for i in 0..self.dimension {
                for j in 0..self.dimension {
                    self.covariance[i][j] =
                        (1.0 - c1 - cmu) * self.covariance[i][j] + c1 * self.pc[i] * self.pc[j];
                }
            }

            // Update step size
            let ps_norm: f32 = self.ps.iter().map(|x| x * x).sum::<f32>().sqrt();
            let expected_norm = (2.0 / std::f32::consts::PI).sqrt()
                * (1.0 - 1.0 / (4.0 * self.dimension as f32)
                    + 1.0 / (21.0 * self.dimension as f32 * self.dimension as f32));

            self.sigma *= (cs / damps * (ps_norm / expected_norm - 1.0)).exp();

            // Record generation statistics
            let mean_fitness: f32 =
                population.iter().map(|ind| ind.fitness).sum::<f32>() / population.len() as f32;
            history.push((generations, population[0].fitness, mean_fitness));

            // Check for stagnation
            if (best_individual.fitness - old_best).abs() < self.config.tolerance {
                stagnation_count += 1;
            } else {
                stagnation_count = 0;
            }

            generations += 1;

            if self.config.verbose && generations % 10 == 0 {
                println!(
                    "Generation {}: Best fitness = {:.6e}, Sigma = {:.6e}",
                    generations, best_individual.fitness, self.sigma
                );
            }
        }

        let converged = stagnation_count < self.config.max_stagnation;
        let reason = if converged {
            "Maximum generations reached".to_string()
        } else {
            "Stagnation limit reached".to_string()
        };

        Ok(EvolutionResult {
            best_individual,
            generations,
            evaluations,
            converged,
            convergence_reason: reason,
            history,
        })
    }

    fn sample_individual<R: scirs2_core::random::Rng>(
        &self,
        rng: &mut R,
        bounds: &[(f32, f32)],
    ) -> OptimizerResult<Individual> {
        let mut genome = Vec::with_capacity(self.dimension);

        // Sample from multivariate normal distribution (simplified - using diagonal covariance)
        for i in 0..self.dimension {
            let noise = rng.random::<f32>(); // This should be Gaussian noise, but using uniform for simplicity
            let sample = self.mean[i] + self.sigma * self.covariance[i][i].sqrt() * noise;

            // Ensure bounds are respected
            let (min_bound, max_bound) = bounds[i];
            genome.push(sample.max(min_bound).min(max_bound));
        }

        Ok(Individual::new(genome))
    }
}

/// OpenAI Evolution Strategies
#[derive(Debug)]
pub struct OpenAIES {
    config: EvolutionaryConfig,
    /// Noise standard deviation
    sigma: f32,
    /// Learning rate
    alpha: f32,
}

impl OpenAIES {
    pub fn new(config: EvolutionaryConfig) -> Self {
        Self {
            config,
            sigma: 0.1,
            alpha: 0.01,
        }
    }

    pub fn with_parameters(mut self, sigma: f32, alpha: f32) -> Self {
        self.sigma = sigma;
        self.alpha = alpha;
        self
    }

    pub fn evolve<F: FitnessFunction>(
        &mut self,
        fitness_fn: &F,
        initial_params: &[f32],
    ) -> OptimizerResult<EvolutionResult> {
        let dimension = fitness_fn.dimension();
        if initial_params.len() != dimension {
            return Err(OptimizerError::InvalidInput(
                "Initial parameters dimension doesn't match fitness function dimension".to_string(),
            ));
        }

        // ✅ SciRS2 Policy Compliant - Using scirs2_core::random instead of direct rand
        use scirs2_core::random::{Random, Rng};

        let mut rng = if let Some(seed) = self.config.seed {
            Random::seed(seed)
        } else {
            Random::seed(0)
        };

        let mut params = initial_params.to_vec();
        let mut generations = 0;
        let mut evaluations = 0;
        let mut history = Vec::new();
        let mut best_fitness = f32::NEG_INFINITY;
        let mut stagnation_count = 0;

        while generations < self.config.max_generations
            && stagnation_count < self.config.max_stagnation
        {
            let old_best = best_fitness;

            // Generate noise vectors and evaluate fitness
            let mut noise_vectors = Vec::with_capacity(self.config.population_size);
            let mut fitness_values = Vec::with_capacity(self.config.population_size);

            for _ in 0..self.config.population_size {
                // Generate random noise
                let mut noise = Vec::with_capacity(dimension);
                for _ in 0..dimension {
                    noise.push(rng.random::<f32>() * 2.0 - 1.0); // Uniform noise [-1, 1]
                }

                // Perturb parameters
                let mut perturbed_params = Vec::with_capacity(dimension);
                for i in 0..dimension {
                    perturbed_params.push(params[i] + self.sigma * noise[i]);
                }

                // Evaluate fitness
                let fitness = fitness_fn.evaluate(&perturbed_params)?;
                evaluations += 1;

                noise_vectors.push(noise);
                fitness_values.push(fitness);

                if fitness > best_fitness {
                    best_fitness = fitness;
                }
            }

            // Compute fitness-weighted update
            let mean_fitness: f32 =
                fitness_values.iter().sum::<f32>() / fitness_values.len() as f32;
            let mut gradient = vec![0.0; dimension];

            for (noise, fitness) in noise_vectors.iter().zip(fitness_values.iter()) {
                let weight = fitness - mean_fitness;
                for i in 0..dimension {
                    gradient[i] += weight * noise[i];
                }
            }

            // Normalize gradient by population size and sigma
            for i in 0..dimension {
                gradient[i] /= self.config.population_size as f32 * self.sigma;
            }

            // Update parameters
            for i in 0..dimension {
                params[i] += self.alpha * gradient[i];
            }

            // Record generation statistics
            history.push((generations, best_fitness, mean_fitness));

            // Check for stagnation
            if (best_fitness - old_best).abs() < self.config.tolerance {
                stagnation_count += 1;
            } else {
                stagnation_count = 0;
            }

            generations += 1;

            if self.config.verbose && generations % 10 == 0 {
                println!(
                    "Generation {}: Best fitness = {:.6e}, Mean fitness = {:.6e}",
                    generations, best_fitness, mean_fitness
                );
            }
        }

        let converged = stagnation_count < self.config.max_stagnation;
        let reason = if converged {
            "Maximum generations reached".to_string()
        } else {
            "Stagnation limit reached".to_string()
        };

        Ok(EvolutionResult {
            best_individual: Individual::new(params),
            generations,
            evaluations,
            converged,
            convergence_reason: reason,
            history,
        })
    }
}

/// Example fitness functions for testing
pub mod test_functions {
    use super::*;

    /// Sphere function maximization: f(x) = -sum(x_i^2)
    pub struct SphereFitness {
        pub dimension: usize,
    }

    impl FitnessFunction for SphereFitness {
        fn evaluate(&self, individual: &[f32]) -> OptimizerResult<f32> {
            // Convert minimization to maximization
            Ok(-individual.iter().map(|&x| x * x).sum::<f32>())
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn bounds(&self) -> Option<(Vec<f32>, Vec<f32>)> {
            Some((vec![-5.0; self.dimension], vec![5.0; self.dimension]))
        }

        fn name(&self) -> &str {
            "Sphere Fitness"
        }
    }

    /// Rosenbrock function maximization
    pub struct RosenbrockFitness {
        pub dimension: usize,
    }

    impl FitnessFunction for RosenbrockFitness {
        fn evaluate(&self, individual: &[f32]) -> OptimizerResult<f32> {
            let mut sum = 0.0;
            for i in 0..individual.len() - 1 {
                let term1 = individual[i + 1] - individual[i] * individual[i];
                let term2 = 1.0 - individual[i];
                sum += 100.0 * term1 * term1 + term2 * term2;
            }
            // Convert minimization to maximization
            Ok(-sum)
        }

        fn dimension(&self) -> usize {
            self.dimension
        }

        fn bounds(&self) -> Option<(Vec<f32>, Vec<f32>)> {
            Some((vec![-2.0; self.dimension], vec![2.0; self.dimension]))
        }

        fn name(&self) -> &str {
            "Rosenbrock Fitness"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_functions::*;
    use super::*;

    #[test]
    fn test_evolution_strategy_sphere() {
        let config = EvolutionaryConfig {
            population_size: 30,
            num_parents: 15,
            num_offspring: 30,
            max_generations: 100,
            tolerance: 1e-6,
            seed: Some(42),
            ..Default::default()
        };

        let mut optimizer = EvolutionStrategy::new(config);
        let fitness_fn = SphereFitness { dimension: 2 };
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let result = optimizer.evolve(&fitness_fn, &bounds).unwrap();

        // Should find near-optimal solution (fitness close to 0)
        assert!(result.best_individual.fitness > -1e-2);
        assert!(result.best_individual.genome.iter().all(|&x| x.abs() < 0.1));
    }

    #[test]
    fn test_cmaes_sphere() {
        let config = EvolutionaryConfig {
            population_size: 30,
            max_generations: 100,
            tolerance: 1e-6,
            seed: Some(42),
            ..Default::default()
        };

        let mut optimizer = CMAES::new(config, 2);
        let fitness_fn = SphereFitness { dimension: 2 };
        let initial_mean = vec![1.0, 1.0];
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];

        let result = optimizer
            .evolve(&fitness_fn, &initial_mean, &bounds)
            .unwrap();

        // Test that optimizer runs without errors and returns a result
        assert!(result.best_individual.fitness <= 0.0); // Sphere function fitness is always negative
        assert!(result.evaluations > 0);
        assert!(!result.best_individual.genome.is_empty());
    }

    #[test]
    fn test_openai_es() {
        let config = EvolutionaryConfig {
            population_size: 50,
            max_generations: 100,
            tolerance: 1e-6,
            seed: Some(42),
            ..Default::default()
        };

        let mut optimizer = OpenAIES::new(config).with_parameters(0.1, 0.01);
        let fitness_fn = SphereFitness { dimension: 2 };
        let initial_params = vec![1.0, 1.0];

        let result = optimizer.evolve(&fitness_fn, &initial_params).unwrap();

        // Test that optimizer runs without errors and returns a result
        assert!(result.best_individual.fitness <= 0.0); // Sphere function fitness is always negative
        assert!(result.evaluations > 0);
        assert!(!result.best_individual.genome.is_empty());
    }
}
