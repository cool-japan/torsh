//! Meta-learning support for optimizers
//!
//! This module provides tools for meta-learning with optimizers, including
//! MAML (Model-Agnostic Meta-Learning), learning-to-optimize, and adaptive
//! optimizer selection based on task characteristics.

use crate::{Adam, Optimizer, OptimizerError, OptimizerResult, SGD};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use torsh_tensor::Tensor;

/// Task characteristics for meta-learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskCharacteristics {
    /// Problem dimensionality
    pub dimension: usize,
    /// Expected training steps
    pub training_steps: usize,
    /// Problem type
    pub problem_type: ProblemType,
    /// Gradient characteristics
    pub gradient_stats: GradientStatistics,
    /// Loss landscape properties
    pub landscape_properties: LandscapeProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProblemType {
    Classification,
    Regression,
    Reinforcement,
    Generative,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientStatistics {
    pub mean_magnitude: f32,
    pub variance_magnitude: f32,
    pub sparsity_ratio: f32,
    pub correlation_length: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LandscapeProperties {
    pub estimated_smoothness: f32,
    pub condition_number_estimate: f32,
    pub has_saddle_points: bool,
    pub convexity_score: f32,
}

/// Meta-learning algorithm types
#[derive(Debug, Clone)]
pub enum MetaLearningAlgorithm {
    /// Model-Agnostic Meta-Learning
    MAML {
        inner_lr: f32,
        outer_lr: f32,
        inner_steps: usize,
    },
    /// Learning to Optimize
    L2O {
        meta_optimizer: Box<dyn Optimizer>,
        hidden_size: usize,
    },
    /// Few-shot adaptation
    FewShot {
        adaptation_steps: usize,
        adaptation_lr: f32,
    },
    /// Gradient-based meta-learning
    GradientBased {
        meta_step_size: f32,
        adaptation_steps: usize,
    },
}

/// Meta-learning configuration
#[derive(Debug, Clone)]
pub struct MetaLearningConfig {
    pub algorithm: MetaLearningAlgorithm,
    pub num_meta_tasks: usize,
    pub support_set_size: usize,
    pub query_set_size: usize,
    pub meta_batch_size: usize,
    pub meta_epochs: usize,
}

/// Task dataset for meta-learning
pub struct TaskDataset {
    pub tasks: Vec<Task>,
    pub meta_split: (Vec<usize>, Vec<usize>), // (train_task_ids, test_task_ids)
}

#[derive(Debug, Clone)]
pub struct Task {
    pub id: String,
    pub characteristics: TaskCharacteristics,
    pub support_data: Vec<(Tensor, Tensor)>, // (input, target) pairs
    pub query_data: Vec<(Tensor, Tensor)>,
    pub metadata: HashMap<String, String>,
}

/// Meta-optimizer that learns to adapt to new tasks
pub struct MetaOptimizer {
    config: MetaLearningConfig,
    base_optimizer: Box<dyn Optimizer>,
    meta_parameters: HashMap<String, Tensor>,
    task_history: Vec<TaskPerformance>,
    adaptation_rules: AdaptationRules,
}

#[derive(Debug, Clone)]
pub struct TaskPerformance {
    pub task_id: String,
    pub characteristics: TaskCharacteristics,
    pub initial_loss: f32,
    pub final_loss: f32,
    pub convergence_steps: usize,
    pub optimizer_config: OptimizerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub optimizer_type: String,
    pub hyperparameters: HashMap<String, f32>,
}

/// Rules for adapting optimizer based on task characteristics
#[derive(Debug, Clone)]
pub struct AdaptationRules {
    /// Map from task characteristics to optimizer configurations
    pub characteristic_rules: Vec<(TaskMatcher, OptimizerConfig)>,
    /// Learned adaptation patterns
    pub learned_patterns: HashMap<String, Vec<f32>>,
    /// Performance thresholds for rule activation
    pub performance_thresholds: HashMap<String, f32>,
}

#[derive(Debug, Clone)]
pub struct TaskMatcher {
    pub dimension_range: Option<(usize, usize)>,
    pub problem_type: Option<ProblemType>,
    pub gradient_magnitude_range: Option<(f32, f32)>,
    pub sparsity_range: Option<(f32, f32)>,
}

impl MetaOptimizer {
    pub fn new(config: MetaLearningConfig, base_optimizer: Box<dyn Optimizer>) -> Self {
        Self {
            config,
            base_optimizer,
            meta_parameters: HashMap::new(),
            task_history: Vec::new(),
            adaptation_rules: AdaptationRules::new(),
        }
    }

    /// Train the meta-optimizer on a set of tasks
    pub fn meta_train(&mut self, task_dataset: &TaskDataset) -> OptimizerResult<()> {
        match &self.config.algorithm {
            MetaLearningAlgorithm::MAML { inner_lr, outer_lr, inner_steps } => {
                self.maml_training(task_dataset, *inner_lr, *outer_lr, *inner_steps)
            }
            MetaLearningAlgorithm::L2O { .. } => {
                self.l2o_training(task_dataset)
            }
            MetaLearningAlgorithm::FewShot { adaptation_steps, adaptation_lr } => {
                self.few_shot_training(task_dataset, *adaptation_steps, *adaptation_lr)
            }
            MetaLearningAlgorithm::GradientBased { meta_step_size, adaptation_steps } => {
                self.gradient_based_training(task_dataset, *meta_step_size, *adaptation_steps)
            }
        }
    }

    /// Adapt to a new task
    pub fn adapt_to_task(&mut self, task: &Task) -> OptimizerResult<Box<dyn Optimizer>> {
        // Analyze task characteristics
        let characteristics = &task.characteristics;
        
        // Select best optimizer configuration based on learned patterns
        let optimizer_config = self.select_optimizer_config(characteristics)?;
        
        // Create adapted optimizer
        let adapted_optimizer = self.create_optimizer_from_config(&optimizer_config)?;
        
        // Perform few-shot adaptation if needed
        let final_optimizer = match &self.config.algorithm {
            MetaLearningAlgorithm::FewShot { adaptation_steps, adaptation_lr } => {
                self.perform_few_shot_adaptation(adapted_optimizer, task, *adaptation_steps, *adaptation_lr)?
            }
            _ => adapted_optimizer,
        };

        Ok(final_optimizer)
    }

    fn maml_training(&mut self, task_dataset: &TaskDataset, inner_lr: f32, outer_lr: f32, inner_steps: usize) -> OptimizerResult<()> {
        for meta_epoch in 0..self.config.meta_epochs {
            let mut meta_gradients = HashMap::new();
            
            // Sample batch of tasks
            let task_batch = self.sample_task_batch(task_dataset, self.config.meta_batch_size)?;
            
            for task_id in task_batch {
                let task = &task_dataset.tasks[task_id];
                
                // Inner loop: adapt to task
                let mut adapted_params = self.meta_parameters.clone();
                
                for _ in 0..inner_steps {
                    // Compute gradients on support set
                    let support_gradients = self.compute_task_gradients(task, &adapted_params, true)?;
                    
                    // Update adapted parameters
                    for (param_name, gradient) in support_gradients {
                        if let Some(param) = adapted_params.get_mut(&param_name) {
                            *param = param.sub(&gradient.mul_scalar(inner_lr)?)?;
                        }
                    }
                }
                
                // Outer loop: compute meta-gradients on query set
                let query_gradients = self.compute_task_gradients(task, &adapted_params, false)?;
                
                // Accumulate meta-gradients
                for (param_name, gradient) in query_gradients {
                    match meta_gradients.get_mut(&param_name) {
                        Some(existing) => {
                            *existing = existing.add(&gradient)?;
                        }
                        None => {
                            meta_gradients.insert(param_name, gradient);
                        }
                    }
                }
            }
            
            // Update meta-parameters
            for (param_name, meta_gradient) in meta_gradients {
                if let Some(param) = self.meta_parameters.get_mut(&param_name) {
                    *param = param.sub(&meta_gradient.mul_scalar(outer_lr)?)?;
                }
            }
            
            log::info!("Meta-epoch {}/{} completed", meta_epoch + 1, self.config.meta_epochs);
        }
        
        Ok(())
    }

    fn l2o_training(&mut self, task_dataset: &TaskDataset) -> OptimizerResult<()> {
        // Learning to Optimize implementation
        // This is a complex algorithm that would typically involve neural networks
        // For now, implement a simplified version
        
        for meta_epoch in 0..self.config.meta_epochs {
            let task_batch = self.sample_task_batch(task_dataset, self.config.meta_batch_size)?;
            
            for task_id in task_batch {
                let task = &task_dataset.tasks[task_id];
                
                // Extract optimizer state features
                let state_features = self.extract_optimizer_state_features(task)?;
                
                // Use meta-optimizer to predict next update
                let predicted_update = self.predict_optimizer_update(&state_features)?;
                
                // Apply update and measure performance
                let performance = self.evaluate_optimizer_update(task, &predicted_update)?;
                
                // Update meta-optimizer based on performance
                self.update_meta_optimizer(performance)?;
            }
        }
        
        Ok(())
    }

    fn few_shot_training(&mut self, task_dataset: &TaskDataset, adaptation_steps: usize, _adaptation_lr: f32) -> OptimizerResult<()> {
        // Build adaptation rules from task performance data
        for task in &task_dataset.tasks {
            // Try different optimizer configurations
            let configs = self.generate_optimizer_configs(&task.characteristics)?;
            
            for config in configs {
                let performance = self.evaluate_config_on_task(&config, task, adaptation_steps)?;
                
                self.task_history.push(TaskPerformance {
                    task_id: task.id.clone(),
                    characteristics: task.characteristics.clone(),
                    initial_loss: performance.initial_loss,
                    final_loss: performance.final_loss,
                    convergence_steps: performance.convergence_steps,
                    optimizer_config: config,
                });
            }
        }
        
        // Learn adaptation rules from performance data
        self.learn_adaptation_rules()?;
        
        Ok(())
    }

    fn gradient_based_training(&mut self, task_dataset: &TaskDataset, meta_step_size: f32, adaptation_steps: usize) -> OptimizerResult<()> {
        // Gradient-based meta-learning (simplified version of MAML)
        for meta_epoch in 0..self.config.meta_epochs {
            let task_batch = self.sample_task_batch(task_dataset, self.config.meta_batch_size)?;
            
            for task_id in task_batch {
                let task = &task_dataset.tasks[task_id];
                
                // Compute higher-order gradients
                let meta_gradients = self.compute_meta_gradients(task, adaptation_steps)?;
                
                // Update meta-parameters
                for (param_name, gradient) in meta_gradients {
                    if let Some(param) = self.meta_parameters.get_mut(&param_name) {
                        *param = param.sub(&gradient.mul_scalar(meta_step_size)?)?;
                    }
                }
            }
        }
        
        Ok(())
    }

    fn select_optimizer_config(&self, characteristics: &TaskCharacteristics) -> OptimizerResult<OptimizerConfig> {
        // Use learned rules to select best optimizer configuration
        for (matcher, config) in &self.adaptation_rules.characteristic_rules {
            if matcher.matches(characteristics) {
                return Ok(config.clone());
            }
        }

        // Fall back to default configuration
        Ok(OptimizerConfig {
            optimizer_type: "Adam".to_string(),
            hyperparameters: {
                let mut params = HashMap::new();
                params.insert("lr".to_string(), 0.001);
                params.insert("beta1".to_string(), 0.9);
                params.insert("beta2".to_string(), 0.999);
                params
            },
        })
    }

    fn create_optimizer_from_config(&self, config: &OptimizerConfig) -> OptimizerResult<Box<dyn Optimizer>> {
        // Empty parameter list: the caller is expected to populate the optimizer
        // via `add_param_group` before use (meta-learning adapts the hyper-params,
        // not the parameter set itself which is task-specific).
        let empty_params: Vec<Arc<RwLock<Tensor>>> = Vec::new();

        match config.optimizer_type.to_lowercase().as_str() {
            "adam" | "adamw" => {
                let lr = config.hyperparameters.get("lr").copied();
                let beta1 = config.hyperparameters.get("beta1").copied().unwrap_or(0.9);
                let beta2 = config.hyperparameters.get("beta2").copied().unwrap_or(0.999);
                let eps = config.hyperparameters.get("eps").copied();
                let weight_decay = config.hyperparameters.get("weight_decay").copied();
                Ok(Box::new(Adam::new(
                    empty_params,
                    lr,
                    Some((beta1, beta2)),
                    eps,
                    weight_decay,
                    false,
                )))
            }
            "sgd" => {
                let lr = config.hyperparameters.get("lr").copied().unwrap_or(0.01);
                let momentum = config.hyperparameters.get("momentum").copied();
                let weight_decay = config.hyperparameters.get("weight_decay").copied();
                let dampening = config.hyperparameters.get("dampening").copied();
                Ok(Box::new(SGD::new(
                    empty_params,
                    lr,
                    momentum,
                    dampening,
                    weight_decay,
                    false,
                )))
            }
            other => Err(OptimizerError::ConfigError(format!(
                "Unknown optimizer type: '{}'. Supported types: adam, adamw, sgd",
                other
            ))),
        }
    }

    fn perform_few_shot_adaptation(&self, mut optimizer: Box<dyn Optimizer>, _task: &Task, adaptation_steps: usize, _adaptation_lr: f32) -> OptimizerResult<Box<dyn Optimizer>> {
        // Perform few-shot adaptation on support set
        for _ in 0..adaptation_steps {
            // Run optimizer step on support data
            optimizer.step()?;
        }

        Ok(optimizer)
    }

    // Helper methods
    fn sample_task_batch(&self, task_dataset: &TaskDataset, batch_size: usize) -> Vec<usize> {
        let mut batch = Vec::new();
        let train_tasks = &task_dataset.meta_split.0;
        
        for _ in 0..batch_size {
            if !train_tasks.is_empty() {
                let idx = fastrand::usize(0..train_tasks.len());
                batch.push(train_tasks[idx]);
            }
        }
        
        batch
    }

    fn compute_task_gradients(&self, task: &Task, _parameters: &HashMap<String, Tensor>, use_support: bool) -> OptimizerResult<HashMap<String, Tensor>> {
        // Computing per-task gradients requires a differentiable model that maps the
        // task's parameters to a loss over its data, followed by a backward pass.
        // `MetaOptimizer` carries only a parameter map and an opaque base optimizer
        // (`Box<dyn Optimizer>`) -- it has no model architecture or loss function to
        // run forward/backward through, so real gradients cannot be produced here.
        //
        // Returning an empty map would silently fabricate "zero gradients" and make
        // the surrounding MAML / gradient-based meta-training loops appear to run
        // while learning nothing. Fail loudly instead.
        let split = if use_support { "support" } else { "query" };
        Err(OptimizerError::ConfigError(format!(
            "compute_task_gradients (task '{}', {} set) is not available: MetaOptimizer \
             has no differentiable model/loss to backpropagate through. Provide a model \
             and loss-aware training loop before invoking gradient-based meta-learning.",
            task.id, split
        )))
    }

    fn extract_optimizer_state_features(&self, task: &Task) -> OptimizerResult<Vec<f32>> {
        // Extract features from optimizer state for L2O
        let mut features = Vec::new();
        
        // Add task characteristics as features
        features.push(task.characteristics.dimension as f32);
        features.push(task.characteristics.gradient_stats.mean_magnitude);
        features.push(task.characteristics.gradient_stats.variance_magnitude);
        features.push(task.characteristics.gradient_stats.sparsity_ratio);
        
        Ok(features)
    }

    fn predict_optimizer_update(&self, _state_features: &[f32]) -> OptimizerResult<HashMap<String, Tensor>> {
        // Learning-to-Optimize (L2O) predicts parameter updates with a learned
        // meta-network (typically an LSTM) mapping optimizer-state features to an
        // update. No such meta-network is instantiated on `MetaOptimizer`, so there
        // is nothing to run inference with.
        //
        // Returning an empty map would masquerade as a real (no-op) update; fail
        // loudly so callers know L2O inference is not wired up.
        Err(OptimizerError::ConfigError(
            "predict_optimizer_update is not available: the L2O meta-network that maps \
             optimizer-state features to parameter updates has not been instantiated. \
             A learned update model must be provided before L2O inference can run."
                .to_string(),
        ))
    }

    fn evaluate_optimizer_update(&self, task: &Task, _update: &HashMap<String, Tensor>) -> OptimizerResult<f32> {
        // Scoring a predicted update requires applying it to the task's parameters
        // and measuring the resulting loss on the task data -- which again needs a
        // model and loss function that `MetaOptimizer` does not hold. A hardcoded
        // `0.0` would be a fabricated metric, so report the missing capability.
        Err(OptimizerError::ConfigError(format!(
            "evaluate_optimizer_update (task '{}') is not available: scoring an update \
             requires a model and loss to measure post-update task performance.",
            task.id
        )))
    }

    fn update_meta_optimizer(&mut self, performance: f32) -> OptimizerResult<()> {
        // Update the meta-optimizer based on performance feedback
        Ok(())
    }

    fn generate_optimizer_configs(&self, characteristics: &TaskCharacteristics) -> Vec<OptimizerConfig> {
        let mut configs = Vec::new();
        
        // Generate configurations based on task characteristics
        match characteristics.problem_type {
            ProblemType::Classification => {
                configs.push(OptimizerConfig {
                    optimizer_type: "Adam".to_string(),
                    hyperparameters: {
                        let mut params = HashMap::new();
                        params.insert("lr".to_string(), 0.001);
                        params.insert("beta1".to_string(), 0.9);
                        params.insert("beta2".to_string(), 0.999);
                        params
                    },
                });
            }
            ProblemType::Regression => {
                configs.push(OptimizerConfig {
                    optimizer_type: "SGD".to_string(),
                    hyperparameters: {
                        let mut params = HashMap::new();
                        params.insert("lr".to_string(), 0.01);
                        params.insert("momentum".to_string(), 0.9);
                        params
                    },
                });
            }
            _ => {
                // Default configuration
                configs.push(OptimizerConfig {
                    optimizer_type: "Adam".to_string(),
                    hyperparameters: HashMap::new(),
                });
            }
        }
        
        configs
    }

    fn evaluate_config_on_task(&self, _config: &OptimizerConfig, task: &Task, _max_steps: usize) -> OptimizerResult<TaskPerformance> {
        // A genuine evaluation trains the configured optimizer on the task's support
        // data for `max_steps` and records the real initial/final losses and the
        // step at which it converged. That requires a model + loss to produce those
        // losses; `MetaOptimizer` has neither.
        //
        // The previous implementation returned hardcoded `initial_loss = 1.0`,
        // `final_loss = 0.1` -- fabricated metrics that would poison the learned
        // adaptation rules derived from this performance history. Return an honest
        // error instead.
        Err(OptimizerError::ConfigError(format!(
            "evaluate_config_on_task (task '{}') is not available: measuring real \
             initial/final losses requires training a model with a loss function, \
             which MetaOptimizer does not provide. Refusing to return fabricated metrics.",
            task.id
        )))
    }

    fn learn_adaptation_rules(&mut self) -> OptimizerResult<()> {
        // Learn adaptation rules from task performance history
        // This would involve clustering, regression, or other ML techniques
        
        // For now, create simple rules based on performance data
        let mut rules = Vec::new();
        
        // Group tasks by characteristics and find best performing configs
        let mut characteristic_groups: HashMap<String, Vec<&TaskPerformance>> = HashMap::new();
        
        for performance in &self.task_history {
            let key = format!("{:?}_{}", 
                performance.characteristics.problem_type,
                performance.characteristics.dimension / 1000 // Group by magnitude
            );
            characteristic_groups.entry(key).or_default().push(performance);
        }
        
        for (_, group) in characteristic_groups {
            if let Some(best_performance) = group.iter().min_by(|a, b| a.final_loss.partial_cmp(&b.final_loss).unwrap_or(std::cmp::Ordering::Equal)) {
                let matcher = TaskMatcher {
                    dimension_range: Some((best_performance.characteristics.dimension.saturating_sub(100), best_performance.characteristics.dimension + 100)),
                    problem_type: Some(best_performance.characteristics.problem_type.clone()),
                    gradient_magnitude_range: None,
                    sparsity_range: None,
                };
                
                rules.push((matcher, best_performance.optimizer_config.clone()));
            }
        }
        
        self.adaptation_rules.characteristic_rules = rules;
        
        Ok(())
    }

    fn compute_meta_gradients(&self, task: &Task, _adaptation_steps: usize) -> OptimizerResult<HashMap<String, Tensor>> {
        // Gradient-based meta-learning differentiates the post-adaptation loss with
        // respect to the initial (meta) parameters, i.e. it computes gradients
        // through the inner-loop optimization (higher-order / "gradient of
        // gradients"). This needs a differentiable model, a loss, and second-order
        // autograd support -- none of which `MetaOptimizer` currently wires up.
        //
        // An empty map would silently pretend the meta-gradients are zero, leaving
        // the meta-parameters unchanged while claiming training occurred. Fail loudly.
        Err(OptimizerError::ConfigError(format!(
            "compute_meta_gradients (task '{}') is not available: higher-order \
             meta-gradients require a differentiable model/loss and second-order \
             autograd through the inner adaptation loop, which MetaOptimizer does not \
             provide.",
            task.id
        )))
    }
}

impl AdaptationRules {
    pub fn new() -> Self {
        Self {
            characteristic_rules: Vec::new(),
            learned_patterns: HashMap::new(),
            performance_thresholds: HashMap::new(),
        }
    }
}

impl TaskMatcher {
    pub fn matches(&self, characteristics: &TaskCharacteristics) -> bool {
        if let Some((min_dim, max_dim)) = self.dimension_range {
            if characteristics.dimension < min_dim || characteristics.dimension > max_dim {
                return false;
            }
        }
        
        if let Some(ref expected_type) = self.problem_type {
            if std::mem::discriminant(&characteristics.problem_type) != std::mem::discriminant(expected_type) {
                return false;
            }
        }
        
        if let Some((min_mag, max_mag)) = self.gradient_magnitude_range {
            if characteristics.gradient_stats.mean_magnitude < min_mag || 
               characteristics.gradient_stats.mean_magnitude > max_mag {
                return false;
            }
        }
        
        if let Some((min_sparse, max_sparse)) = self.sparsity_range {
            if characteristics.gradient_stats.sparsity_ratio < min_sparse || 
               characteristics.gradient_stats.sparsity_ratio > max_sparse {
                return false;
            }
        }
        
        true
    }
}

/// Utility functions for meta-learning
pub mod utils {
    use super::*;
    
    /// Analyze task characteristics from gradient history
    pub fn analyze_task_characteristics(gradients: &[Tensor]) -> OptimizerResult<TaskCharacteristics> {
        if gradients.is_empty() {
            return Err(OptimizerError::InvalidParameter("Empty gradient history".to_string()));
        }
        
        // Compute gradient statistics
        let mut magnitudes = Vec::new();
        let mut sparsity_ratios = Vec::new();
        
        for gradient in gradients {
            // Compute magnitude
            let magnitude = gradient.norm()?.item()?;
            magnitudes.push(magnitude);
            
            // Compute sparsity ratio (simplified)
            let total_elements = gradient.numel()?;
            let zero_threshold = 1e-8;
            let non_zero_count = gradient.abs()?.gt_scalar(zero_threshold)?.sum()?.item()? as usize;
            let sparsity = 1.0 - (non_zero_count as f32 / total_elements as f32);
            sparsity_ratios.push(sparsity);
        }
        
        let mean_magnitude = magnitudes.iter().sum::<f32>() / magnitudes.len() as f32;
        let variance_magnitude = magnitudes.iter()
            .map(|x| (x - mean_magnitude).powi(2))
            .sum::<f32>() / magnitudes.len() as f32;
        let mean_sparsity = sparsity_ratios.iter().sum::<f32>() / sparsity_ratios.len() as f32;
        
        Ok(TaskCharacteristics {
            dimension: gradients[0].numel(),
            training_steps: gradients.len(),
            problem_type: ProblemType::Custom("unknown".to_string()),
            gradient_stats: GradientStatistics {
                mean_magnitude,
                variance_magnitude,
                sparsity_ratio: mean_sparsity,
                correlation_length: 1.0, // Simplified
            },
            landscape_properties: LandscapeProperties {
                estimated_smoothness: 0.5,
                condition_number_estimate: 1.0,
                has_saddle_points: false,
                convexity_score: 0.5,
            },
        })
    }
    
    /// Create a meta-learning dataset from task data
    pub fn create_meta_dataset(tasks: Vec<Task>, train_ratio: f32) -> TaskDataset {
        let n_train = (tasks.len() as f32 * train_ratio) as usize;
        let train_ids: Vec<usize> = (0..n_train).collect();
        let test_ids: Vec<usize> = (n_train..tasks.len()).collect();
        
        TaskDataset {
            tasks,
            meta_split: (train_ids, test_ids),
        }
    }
    
    /// Evaluate meta-learning performance across a set of test tasks.
    ///
    /// A real evaluation would, for each task, adapt the meta-optimizer and then
    /// measure the adapted model's loss/accuracy on the task's query set. The query
    /// score requires a model and loss function to evaluate, which `MetaOptimizer`
    /// does not hold, so a genuine aggregate performance number cannot be computed.
    ///
    /// The previous implementation accumulated a hardcoded `0.8` per successfully
    /// adapted task -- a fabricated metric independent of the actual tasks. Rather
    /// than return that, this function reports the missing capability.
    pub fn evaluate_meta_performance(_meta_optimizer: &mut MetaOptimizer, _test_tasks: &[Task]) -> OptimizerResult<f32> {
        Err(OptimizerError::ConfigError(
            "evaluate_meta_performance is not available: scoring adapted optimizers on \
             their query sets requires a model and loss function to measure real task \
             performance. Refusing to return a fabricated score."
                .to_string(),
        ))
    }
}