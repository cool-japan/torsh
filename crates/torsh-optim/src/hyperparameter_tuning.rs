//! Automatic hyperparameter tuning for optimizers
//!
//! This module provides tools for automatically tuning optimizer hyperparameters
//! using various search strategies including grid search, random search, Bayesian optimization,
//! and evolutionary algorithms.

use crate::{OptimizerError, OptimizerResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Hyperparameter search space definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSpace {
    /// Continuous parameters with [min, max] bounds
    pub continuous: HashMap<String, (f32, f32)>,
    /// Discrete integer parameters with [min, max] bounds
    pub discrete: HashMap<String, (i32, i32)>,
    /// Categorical parameters with list of choices
    pub categorical: HashMap<String, Vec<String>>,
    /// Log-scale parameters (will be sampled in log space)
    pub log_scale: Vec<String>,
}

impl HyperparameterSpace {
    pub fn new() -> Self {
        Self {
            continuous: HashMap::new(),
            discrete: HashMap::new(),
            categorical: HashMap::new(),
            log_scale: Vec::new(),
        }
    }

    pub fn add_continuous(mut self, name: &str, min: f32, max: f32) -> Self {
        self.continuous.insert(name.to_string(), (min, max));
        self
    }

    pub fn add_discrete(mut self, name: &str, min: i32, max: i32) -> Self {
        self.discrete.insert(name.to_string(), (min, max));
        self
    }

    pub fn add_categorical(mut self, name: &str, choices: Vec<&str>) -> Self {
        self.categorical.insert(
            name.to_string(),
            choices.iter().map(|s| s.to_string()).collect(),
        );
        self
    }

    pub fn set_log_scale(mut self, names: Vec<&str>) -> Self {
        self.log_scale = names.iter().map(|s| s.to_string()).collect();
        self
    }
}

/// Hyperparameter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterConfig {
    pub parameters: HashMap<String, HyperparameterValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HyperparameterValue {
    Float(f32),
    Int(i32),
    String(String),
}

impl HyperparameterConfig {
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
        }
    }

    pub fn set_float(&mut self, name: &str, value: f32) {
        self.parameters
            .insert(name.to_string(), HyperparameterValue::Float(value));
    }

    pub fn set_int(&mut self, name: &str, value: i32) {
        self.parameters
            .insert(name.to_string(), HyperparameterValue::Int(value));
    }

    pub fn set_string(&mut self, name: &str, value: &str) {
        self.parameters.insert(
            name.to_string(),
            HyperparameterValue::String(value.to_string()),
        );
    }

    pub fn get_float(&self, name: &str) -> OptimizerResult<f32> {
        match self.parameters.get(name) {
            Some(HyperparameterValue::Float(v)) => Ok(*v),
            _ => Err(OptimizerError::InvalidParameter(format!(
                "Parameter {} not found or not float",
                name
            ))),
        }
    }

    pub fn get_int(&self, name: &str) -> OptimizerResult<i32> {
        match self.parameters.get(name) {
            Some(HyperparameterValue::Int(v)) => Ok(*v),
            _ => Err(OptimizerError::InvalidParameter(format!(
                "Parameter {} not found or not int",
                name
            ))),
        }
    }

    pub fn get_string(&self, name: &str) -> OptimizerResult<&str> {
        match self.parameters.get(name) {
            Some(HyperparameterValue::String(v)) => Ok(v),
            _ => Err(OptimizerError::InvalidParameter(format!(
                "Parameter {} not found or not string",
                name
            ))),
        }
    }
}

/// Search strategy for hyperparameter optimization
#[derive(Debug, Clone)]
pub enum SearchStrategy {
    /// Random search with specified number of trials
    Random { n_trials: usize },
    /// Grid search with specified resolution per dimension
    Grid { n_points_per_dim: usize },
    /// Bayesian optimization using Gaussian processes
    Bayesian {
        n_trials: usize,
        acquisition_function: AcquisitionFunction,
    },
    /// Evolutionary algorithm
    Evolutionary {
        population_size: usize,
        n_generations: usize,
        mutation_rate: f32,
    },
}

#[derive(Debug, Clone)]
pub enum AcquisitionFunction {
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound { kappa: f32 },
}

/// Trial result from hyperparameter optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    pub config: HyperparameterConfig,
    pub objective_value: f32,
    pub duration: Duration,
    pub metadata: HashMap<String, String>,
}

/// Configuration for hyperparameter tuning
#[derive(Debug, Clone)]
pub struct TuningConfig {
    pub search_strategy: SearchStrategy,
    pub objective: ObjectiveFunction,
    pub search_space: HyperparameterSpace,
    pub max_duration: Option<Duration>,
    pub early_stopping: Option<EarlyStoppingConfig>,
    pub parallel_trials: usize,
}

#[derive(Debug, Clone)]
pub enum ObjectiveFunction {
    /// Minimize validation loss
    MinimizeValidationLoss,
    /// Maximize validation accuracy
    MaximizeValidationAccuracy,
    /// Custom objective function
    Custom(fn(&HyperparameterConfig) -> f32),
}

#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    pub patience: usize,
    pub min_delta: f32,
}

/// Hyperparameter tuner
pub struct HyperparameterTuner {
    config: TuningConfig,
    trials: Vec<Trial>,
    best_trial: Option<Trial>,
    start_time: Option<Instant>,
}

impl HyperparameterTuner {
    pub fn new(config: TuningConfig) -> Self {
        Self {
            config,
            trials: Vec::new(),
            best_trial: None,
            start_time: None,
        }
    }

    /// Run hyperparameter optimization
    pub fn optimize<F>(&mut self, objective_fn: F) -> OptimizerResult<HyperparameterConfig>
    where
        F: Fn(&HyperparameterConfig) -> OptimizerResult<f32>,
    {
        self.start_time = Some(Instant::now());

        match &self.config.search_strategy {
            SearchStrategy::Random { n_trials } => self.random_search(*n_trials, objective_fn),
            SearchStrategy::Grid { n_points_per_dim } => {
                self.grid_search(*n_points_per_dim, objective_fn)
            }
            SearchStrategy::Bayesian { n_trials, .. } => {
                self.bayesian_optimization(*n_trials, objective_fn)
            }
            SearchStrategy::Evolutionary {
                population_size,
                n_generations,
                ..
            } => self.evolutionary_search(*population_size, *n_generations, objective_fn),
        }
    }

    fn random_search<F>(
        &mut self,
        n_trials: usize,
        objective_fn: F,
    ) -> OptimizerResult<HyperparameterConfig>
    where
        F: Fn(&HyperparameterConfig) -> OptimizerResult<f32>,
    {
        for _ in 0..n_trials {
            if self.should_stop() {
                break;
            }

            let config = self.sample_random_config()?;
            let trial_start = Instant::now();

            match objective_fn(&config) {
                Ok(objective_value) => {
                    let trial = Trial {
                        config: config.clone(),
                        objective_value,
                        duration: trial_start.elapsed(),
                        metadata: HashMap::new(),
                    };

                    self.update_best_trial(&trial);
                    self.trials.push(trial);
                }
                Err(e) => {
                    log::warn!("Trial failed: {:?}", e);
                }
            }
        }

        self.get_best_config()
    }

    fn grid_search<F>(
        &mut self,
        n_points_per_dim: usize,
        objective_fn: F,
    ) -> OptimizerResult<HyperparameterConfig>
    where
        F: Fn(&HyperparameterConfig) -> OptimizerResult<f32>,
    {
        let grid_points = self.generate_grid_points(n_points_per_dim)?;

        for config in grid_points {
            if self.should_stop() {
                break;
            }

            let trial_start = Instant::now();

            match objective_fn(&config) {
                Ok(objective_value) => {
                    let trial = Trial {
                        config: config.clone(),
                        objective_value,
                        duration: trial_start.elapsed(),
                        metadata: HashMap::new(),
                    };

                    self.update_best_trial(&trial);
                    self.trials.push(trial);
                }
                Err(e) => {
                    log::warn!("Trial failed: {:?}", e);
                }
            }
        }

        self.get_best_config()
    }

    fn bayesian_optimization<F>(
        &mut self,
        n_trials: usize,
        objective_fn: F,
    ) -> OptimizerResult<HyperparameterConfig>
    where
        F: Fn(&HyperparameterConfig) -> OptimizerResult<f32>,
    {
        // Simplified Bayesian optimization (in practice, would use GP library)
        // For now, implement as random search with some exploitation

        // Start with random exploration
        let n_random = (n_trials as f32 * 0.3) as usize;
        for _ in 0..n_random {
            if self.should_stop() {
                break;
            }

            let config = self.sample_random_config()?;
            let trial_start = Instant::now();

            match objective_fn(&config) {
                Ok(objective_value) => {
                    let trial = Trial {
                        config: config.clone(),
                        objective_value,
                        duration: trial_start.elapsed(),
                        metadata: HashMap::new(),
                    };

                    self.update_best_trial(&trial);
                    self.trials.push(trial);
                }
                Err(e) => {
                    log::warn!("Trial failed: {:?}", e);
                }
            }
        }

        // Then exploit around best configurations
        for _ in n_random..n_trials {
            if self.should_stop() {
                break;
            }

            let config = if let Some(best) = &self.best_trial {
                self.sample_around_config(&best.config, 0.1)?
            } else {
                self.sample_random_config()?
            };

            let trial_start = Instant::now();

            match objective_fn(&config) {
                Ok(objective_value) => {
                    let trial = Trial {
                        config: config.clone(),
                        objective_value,
                        duration: trial_start.elapsed(),
                        metadata: HashMap::new(),
                    };

                    self.update_best_trial(&trial);
                    self.trials.push(trial);
                }
                Err(e) => {
                    log::warn!("Trial failed: {:?}", e);
                }
            }
        }

        self.get_best_config()
    }

    fn evolutionary_search<F>(
        &mut self,
        population_size: usize,
        n_generations: usize,
        objective_fn: F,
    ) -> OptimizerResult<HyperparameterConfig>
    where
        F: Fn(&HyperparameterConfig) -> OptimizerResult<f32>,
    {
        // Initialize population
        let mut population = Vec::new();
        for _ in 0..population_size {
            let config = self.sample_random_config()?;
            if let Ok(objective_value) = objective_fn(&config) {
                population.push((config, objective_value));
            }
        }

        // Evolution loop
        for _generation in 0..n_generations {
            if self.should_stop() {
                break;
            }

            // Selection (tournament selection)
            let mut new_population = Vec::new();
            for _ in 0..population_size {
                let parent1 = self.tournament_selection(&population, 3);
                let parent2 = self.tournament_selection(&population, 3);

                if let Ok(child) = self.crossover(&parent1.0, &parent2.0) {
                    let mutated = self.mutate(&child, 0.1)?;

                    if let Ok(objective_value) = objective_fn(&mutated) {
                        new_population.push((mutated, objective_value));
                    }
                }
            }

            population = new_population;

            // Update best trial
            if let Some((best_config, best_value)) = population
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                let trial = Trial {
                    config: best_config.clone(),
                    objective_value: *best_value,
                    duration: Duration::from_secs(0), // Approximate
                    metadata: HashMap::new(),
                };
                self.update_best_trial(&trial);
            }
        }

        self.get_best_config()
    }

    fn sample_random_config(&self) -> OptimizerResult<HyperparameterConfig> {
        let mut config = HyperparameterConfig::new();

        // Sample continuous parameters
        for (name, (min, max)) in &self.config.search_space.continuous {
            let value = if self.config.search_space.log_scale.contains(name) {
                let log_min = min.ln();
                let log_max = max.ln();
                let log_value = log_min + 0.5 * (log_max - log_min);
                log_value.exp()
            } else {
                min + 0.5 * (max - min)
            };
            config.set_float(name, value);
        }

        // Sample discrete parameters
        for (name, (min, max)) in &self.config.search_space.discrete {
            let value = (*min + *max) / 2;
            config.set_int(name, value);
        }

        // Sample categorical parameters
        for (name, choices) in &self.config.search_space.categorical {
            if !choices.is_empty() {
                let idx = 0;
                config.set_string(name, &choices[idx]);
            }
        }

        Ok(config)
    }

    fn sample_around_config(
        &self,
        base_config: &HyperparameterConfig,
        noise_scale: f32,
    ) -> OptimizerResult<HyperparameterConfig> {
        let mut config = base_config.clone();

        // Add noise to continuous parameters
        for (name, (min, max)) in &self.config.search_space.continuous {
            if let Ok(base_value) = base_config.get_float(name) {
                let range = max - min;
                let noise = 0.0;
                let new_value = (base_value + noise).clamp(*min, *max);
                config.set_float(name, new_value);
            }
        }

        Ok(config)
    }

    fn generate_grid_points(
        &self,
        n_points_per_dim: usize,
    ) -> OptimizerResult<Vec<HyperparameterConfig>> {
        let mut configs = Vec::new();

        // For simplicity, only handle continuous parameters in grid search
        if self.config.search_space.continuous.is_empty() {
            return Ok(vec![self.sample_random_config()?]);
        }

        let param_names: Vec<_> = self
            .config
            .search_space
            .continuous
            .keys()
            .cloned()
            .collect();
        let n_params = param_names.len();

        // Generate all combinations
        let total_points = n_points_per_dim.pow(n_params as u32);

        for i in 0..total_points {
            let mut config = HyperparameterConfig::new();
            let mut remaining = i;

            for (_param_idx, param_name) in param_names.iter().enumerate() {
                let (min, max) = self.config.search_space.continuous[param_name];
                let grid_idx = remaining % n_points_per_dim;
                remaining /= n_points_per_dim;

                let value = if n_points_per_dim == 1 {
                    (min + max) / 2.0
                } else {
                    min + (grid_idx as f32) * (max - min) / ((n_points_per_dim - 1) as f32)
                };

                config.set_float(param_name, value);
            }

            configs.push(config);
        }

        Ok(configs)
    }

    fn tournament_selection<'a>(
        &self,
        population: &'a [(HyperparameterConfig, f32)],
        tournament_size: usize,
    ) -> &'a (HyperparameterConfig, f32) {
        let mut best = &population[0];

        for _ in 1..tournament_size {
            let candidate = &population[0];
            if candidate.1 < best.1 {
                // Assuming minimization
                best = candidate;
            }
        }

        best
    }

    fn crossover(
        &self,
        parent1: &HyperparameterConfig,
        parent2: &HyperparameterConfig,
    ) -> OptimizerResult<HyperparameterConfig> {
        let mut child = HyperparameterConfig::new();

        // Blend crossover for continuous parameters
        for name in self.config.search_space.continuous.keys() {
            if let (Ok(v1), Ok(v2)) = (parent1.get_float(name), parent2.get_float(name)) {
                let alpha = 0.5;
                let value = (1.0 - alpha) * v1 + alpha * v2;
                child.set_float(name, value);
            }
        }

        // Random selection for categorical/discrete
        for name in self.config.search_space.discrete.keys() {
            let value = if true {
                parent1.get_int(name).unwrap_or(0)
            } else {
                parent2.get_int(name).unwrap_or(0)
            };
            child.set_int(name, value);
        }

        Ok(child)
    }

    fn mutate(
        &self,
        config: &HyperparameterConfig,
        mutation_rate: f32,
    ) -> OptimizerResult<HyperparameterConfig> {
        let mut mutated = config.clone();

        for (name, (min, max)) in &self.config.search_space.continuous {
            if 0.1 < mutation_rate {
                if let Ok(current_value) = config.get_float(name) {
                    let range = max - min;
                    let noise = 0.0;
                    let new_value = (current_value + noise).clamp(*min, *max);
                    mutated.set_float(name, new_value);
                }
            }
        }

        Ok(mutated)
    }

    fn should_stop(&self) -> bool {
        if let (Some(start_time), Some(max_duration)) = (self.start_time, &self.config.max_duration)
        {
            if start_time.elapsed() > *max_duration {
                return true;
            }
        }

        // Early stopping based on convergence
        if let Some(early_stopping) = &self.config.early_stopping {
            if self.trials.len() >= early_stopping.patience {
                let recent_trials = &self.trials[self.trials.len() - early_stopping.patience..];
                let values: Vec<f32> = recent_trials.iter().map(|t| t.objective_value).collect();

                let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                if (max_val - min_val) < early_stopping.min_delta {
                    return true;
                }
            }
        }

        false
    }

    fn update_best_trial(&mut self, trial: &Trial) {
        let is_better = match &self.best_trial {
            None => true,
            Some(best) => match &self.config.objective {
                ObjectiveFunction::MinimizeValidationLoss => {
                    trial.objective_value < best.objective_value
                }
                ObjectiveFunction::MaximizeValidationAccuracy => {
                    trial.objective_value > best.objective_value
                }
                ObjectiveFunction::Custom(_) => trial.objective_value < best.objective_value, // Assume minimization
            },
        };

        if is_better {
            self.best_trial = Some(trial.clone());
        }
    }

    fn get_best_config(&self) -> OptimizerResult<HyperparameterConfig> {
        match &self.best_trial {
            Some(trial) => Ok(trial.config.clone()),
            None => Err(OptimizerError::StateError(
                "No successful trials found".to_string(),
            )),
        }
    }

    /// Get optimization results and statistics
    pub fn get_results(&self) -> TuningResults {
        TuningResults {
            best_config: self.best_trial.as_ref().map(|t| t.config.clone()),
            best_value: self.best_trial.as_ref().map(|t| t.objective_value),
            trials: self.trials.clone(),
            total_duration: self.start_time.map(|t| t.elapsed()),
        }
    }
}

/// Results from hyperparameter tuning
#[derive(Debug, Clone)]
pub struct TuningResults {
    pub best_config: Option<HyperparameterConfig>,
    pub best_value: Option<f32>,
    pub trials: Vec<Trial>,
    pub total_duration: Option<Duration>,
}

impl TuningResults {
    /// Get convergence history
    pub fn convergence_history(&self) -> Vec<f32> {
        let mut best_so_far = f32::INFINITY;
        let mut history = Vec::new();

        for trial in &self.trials {
            if trial.objective_value < best_so_far {
                best_so_far = trial.objective_value;
            }
            history.push(best_so_far);
        }

        history
    }

    /// Get parameter importance analysis
    pub fn parameter_importance(&self) -> HashMap<String, f32> {
        let mut param_values: HashMap<String, Vec<f32>> = HashMap::new();

        if self.trials.len() < 2 {
            return HashMap::new();
        }

        // Simple variance-based importance
        for trial in &self.trials {
            for (param_name, param_value) in &trial.config.parameters {
                if let HyperparameterValue::Float(value) = param_value {
                    param_values
                        .entry(param_name.clone())
                        .or_insert(Vec::new())
                        .push(*value);
                }
            }
        }

        // Calculate normalized variance for each parameter
        let mut variance_scores = HashMap::new();
        for (param_name, values) in &param_values {
            if values.len() > 1 {
                let mean = values.iter().sum::<f32>() / values.len() as f32;
                let variance =
                    values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
                variance_scores.insert(param_name.clone(), variance);
            }
        }

        // Normalize scores
        let mut importance = HashMap::new();
        let max_variance = variance_scores.values().fold(0.0f32, |a, &b| a.max(b));
        if max_variance > 0.0 {
            for (param_name, variance) in variance_scores {
                importance.insert(param_name, variance / max_variance);
            }
        }

        importance
    }
}

/// Utility functions for common optimizer hyperparameter spaces
pub mod presets {
    use super::*;

    /// Adam optimizer hyperparameter space
    pub fn adam_space() -> HyperparameterSpace {
        HyperparameterSpace::new()
            .add_continuous("lr", 1e-5, 1e-1)
            .add_continuous("beta1", 0.8, 0.99)
            .add_continuous("beta2", 0.9, 0.999)
            .add_continuous("eps", 1e-10, 1e-6)
            .add_continuous("weight_decay", 0.0, 1e-2)
            .set_log_scale(vec!["lr", "eps"])
    }

    /// SGD optimizer hyperparameter space
    pub fn sgd_space() -> HyperparameterSpace {
        HyperparameterSpace::new()
            .add_continuous("lr", 1e-4, 1.0)
            .add_continuous("momentum", 0.0, 0.99)
            .add_continuous("weight_decay", 0.0, 1e-2)
            .add_categorical("nesterov", vec!["true", "false"])
            .set_log_scale(vec!["lr"])
    }

    /// RMSprop optimizer hyperparameter space
    pub fn rmsprop_space() -> HyperparameterSpace {
        HyperparameterSpace::new()
            .add_continuous("lr", 1e-5, 1e-1)
            .add_continuous("alpha", 0.9, 0.999)
            .add_continuous("eps", 1e-10, 1e-6)
            .add_continuous("weight_decay", 0.0, 1e-2)
            .add_continuous("momentum", 0.0, 0.1)
            .set_log_scale(vec!["lr", "eps"])
    }
}
