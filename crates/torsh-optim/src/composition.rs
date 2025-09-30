//! Optimizer composition tools
//!
//! This module provides utilities for composing multiple optimizers together,
//! creating ensembles, pipelines, and adaptive switching between optimizers.

use crate::{Optimizer, OptimizerError, OptimizerResult, OptimizerState};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Add;
use std::sync::Arc;
use torsh_tensor::Tensor;

/// Strategies for combining multiple optimizers
#[derive(Debug, Clone)]
pub enum CompositionStrategy {
    /// Use different optimizers in sequence (pipeline)
    Sequential {
        schedule: Vec<(String, usize)>, // (optimizer_name, steps)
    },
    /// Ensemble of optimizers with weighted averaging
    Ensemble {
        weights: HashMap<String, f32>,
        combination_method: CombinationMethod,
    },
    /// Adaptive switching based on performance
    Adaptive {
        switch_criterion: SwitchCriterion,
        evaluation_window: usize,
    },
    /// Parallel execution with consensus
    Consensus {
        agreement_threshold: f32,
        voting_method: VotingMethod,
    },
    /// Hierarchical composition
    Hierarchical { levels: Vec<CompositionLevel> },
}

#[derive(Debug, Clone)]
pub enum CombinationMethod {
    /// Weighted average of parameter updates
    WeightedAverage,
    /// Median of parameter updates
    Median,
    /// Best performing optimizer takes precedence
    BestWins,
    /// Custom combination function
    Custom(fn(&[Tensor]) -> Tensor),
}

#[derive(Debug, Clone)]
pub enum SwitchCriterion {
    /// Switch based on loss improvement
    LossImprovement { threshold: f32 },
    /// Switch based on gradient magnitude
    GradientMagnitude { threshold: f32 },
    /// Switch based on convergence rate
    ConvergenceRate { window: usize },
    /// Switch based on custom metric
    Custom(fn(&OptimizerMetrics) -> bool),
}

#[derive(Debug, Clone)]
pub enum VotingMethod {
    /// Majority vote on update direction
    Majority,
    /// Weighted vote based on recent performance
    WeightedVote,
    /// Unanimous consensus required
    Unanimous,
}

#[derive(Debug, Clone)]
pub struct CompositionLevel {
    pub name: String,
    pub optimizers: Vec<String>,
    pub strategy: CompositionStrategy,
}

/// Metrics for optimizer performance evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerMetrics {
    pub loss_history: Vec<f32>,
    pub gradient_norms: Vec<f32>,
    pub update_magnitudes: Vec<f32>,
    pub convergence_rate: f32,
    pub stability_score: f32,
    pub efficiency_score: f32,
}

impl Default for OptimizerMetrics {
    fn default() -> Self {
        Self {
            loss_history: Vec::new(),
            gradient_norms: Vec::new(),
            update_magnitudes: Vec::new(),
            convergence_rate: 0.0,
            stability_score: 0.0,
            efficiency_score: 0.0,
        }
    }
}

impl OptimizerMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, loss: f32, gradient_norm: f32, update_magnitude: f32) {
        self.loss_history.push(loss);
        self.gradient_norms.push(gradient_norm);
        self.update_magnitudes.push(update_magnitude);

        self.compute_derived_metrics();
    }

    fn compute_derived_metrics(&mut self) {
        if self.loss_history.len() < 2 {
            return;
        }

        // Compute convergence rate
        let recent_losses = &self.loss_history[self.loss_history.len().saturating_sub(10)..];
        if recent_losses.len() >= 2 {
            let start_loss = recent_losses[0];
            let end_loss = recent_losses[recent_losses.len() - 1];
            self.convergence_rate = (start_loss - end_loss) / recent_losses.len() as f32;
        }

        // Compute stability score (lower variance is better)
        if !self.update_magnitudes.is_empty() {
            let mean_magnitude =
                self.update_magnitudes.iter().sum::<f32>() / self.update_magnitudes.len() as f32;
            let variance = self
                .update_magnitudes
                .iter()
                .map(|x| (x - mean_magnitude).powi(2))
                .sum::<f32>()
                / self.update_magnitudes.len() as f32;
            self.stability_score = 1.0 / (1.0 + variance);
        }

        // Compute efficiency score (convergence per unit work)
        if !self.loss_history.is_empty() {
            let total_improvement =
                self.loss_history[0] - self.loss_history[self.loss_history.len() - 1];
            let steps = self.loss_history.len() as f32;
            self.efficiency_score = total_improvement / steps;
        }
    }
}

/// Composed optimizer that combines multiple optimizers
pub struct ComposedOptimizer {
    strategy: CompositionStrategy,
    optimizers: HashMap<String, Box<dyn Optimizer>>,
    metrics: HashMap<String, OptimizerMetrics>,
    current_optimizer: Option<String>,
    step_count: usize,
    composition_state: CompositionState,
}

#[derive(Debug, Clone)]
enum CompositionState {
    Sequential {
        current_phase: usize,
        phase_steps: usize,
    },
    Ensemble {
        last_updates: HashMap<String, Vec<Tensor>>,
    },
    Adaptive {
        evaluation_buffer: Vec<(String, f32)>,
    },
    Consensus {
        votes: HashMap<String, Vec<Tensor>>,
    },
    Hierarchical {
        current_level: usize,
    },
}

impl ComposedOptimizer {
    pub fn new(strategy: CompositionStrategy) -> Self {
        let composition_state = match &strategy {
            CompositionStrategy::Sequential { .. } => CompositionState::Sequential {
                current_phase: 0,
                phase_steps: 0,
            },
            CompositionStrategy::Ensemble { .. } => CompositionState::Ensemble {
                last_updates: HashMap::new(),
            },
            CompositionStrategy::Adaptive { .. } => CompositionState::Adaptive {
                evaluation_buffer: Vec::new(),
            },
            CompositionStrategy::Consensus { .. } => CompositionState::Consensus {
                votes: HashMap::new(),
            },
            CompositionStrategy::Hierarchical { .. } => {
                CompositionState::Hierarchical { current_level: 0 }
            }
        };

        Self {
            strategy,
            optimizers: HashMap::new(),
            metrics: HashMap::new(),
            current_optimizer: None,
            step_count: 0,
            composition_state,
        }
    }

    /// Add an optimizer to the composition
    pub fn add_optimizer(&mut self, name: String, optimizer: Box<dyn Optimizer>) {
        self.optimizers.insert(name.clone(), optimizer);
        self.metrics.insert(name.clone(), OptimizerMetrics::new());
    }

    /// Remove an optimizer from the composition
    pub fn remove_optimizer(&mut self, name: &str) -> Option<Box<dyn Optimizer>> {
        self.metrics.remove(name);
        self.optimizers.remove(name)
    }

    /// Get current active optimizers
    pub fn active_optimizers(&self) -> Vec<&str> {
        match &self.strategy {
            CompositionStrategy::Sequential { .. } => {
                if let Some(ref current) = self.current_optimizer {
                    vec![current]
                } else {
                    Vec::new()
                }
            }
            CompositionStrategy::Ensemble { .. } => {
                self.optimizers.keys().map(|s| s.as_str()).collect()
            }
            CompositionStrategy::Adaptive { .. } => {
                if let Some(ref current) = self.current_optimizer {
                    vec![current]
                } else {
                    Vec::new()
                }
            }
            CompositionStrategy::Consensus { .. } => {
                self.optimizers.keys().map(|s| s.as_str()).collect()
            }
            CompositionStrategy::Hierarchical { .. } => {
                // Return optimizers from current level
                if let CompositionState::Hierarchical { current_level } = &self.composition_state {
                    if let CompositionStrategy::Hierarchical { levels } = &self.strategy {
                        if *current_level < levels.len() {
                            return levels[*current_level]
                                .optimizers
                                .iter()
                                .map(|s| s.as_str())
                                .collect();
                        }
                    }
                }
                Vec::new()
            }
        }
    }

    /// Update optimizer metrics
    pub fn update_metrics(
        &mut self,
        optimizer_name: &str,
        loss: f32,
        gradient_norm: f32,
        update_magnitude: f32,
    ) {
        if let Some(metrics) = self.metrics.get_mut(optimizer_name) {
            metrics.update(loss, gradient_norm, update_magnitude);
        }
    }

    /// Get performance metrics for an optimizer
    pub fn get_metrics(&self, optimizer_name: &str) -> Option<&OptimizerMetrics> {
        self.metrics.get(optimizer_name)
    }

    /// Get the best performing optimizer based on metrics
    pub fn best_optimizer(&self) -> Option<&str> {
        let mut best_name = None;
        let mut best_score = f32::NEG_INFINITY;

        for (name, metrics) in &self.metrics {
            let score =
                metrics.convergence_rate * metrics.stability_score * metrics.efficiency_score;
            if score > best_score {
                best_score = score;
                best_name = Some(name.as_str());
            }
        }

        best_name
    }

    /// Execute the composition strategy
    fn execute_strategy(&mut self) -> OptimizerResult<()> {
        match &self.strategy.clone() {
            CompositionStrategy::Sequential { schedule } => self.execute_sequential(schedule),
            CompositionStrategy::Ensemble {
                weights,
                combination_method,
            } => self.execute_ensemble(weights, combination_method),
            CompositionStrategy::Adaptive {
                switch_criterion,
                evaluation_window,
            } => self.execute_adaptive(switch_criterion, *evaluation_window),
            CompositionStrategy::Consensus {
                agreement_threshold,
                voting_method,
            } => self.execute_consensus(*agreement_threshold, voting_method),
            CompositionStrategy::Hierarchical { levels } => self.execute_hierarchical(levels),
        }
    }

    fn execute_sequential(&mut self, schedule: &[(String, usize)]) -> OptimizerResult<()> {
        if let CompositionState::Sequential {
            current_phase,
            phase_steps,
        } = &mut self.composition_state
        {
            if *current_phase < schedule.len() {
                let (optimizer_name, max_steps) = &schedule[*current_phase];

                if *phase_steps < *max_steps {
                    // Continue with current optimizer
                    if let Some(optimizer) = self.optimizers.get_mut(optimizer_name) {
                        optimizer.step()?;
                        *phase_steps += 1;
                        self.current_optimizer = Some(optimizer_name.clone());
                    }
                } else {
                    // Move to next phase
                    *current_phase += 1;
                    *phase_steps = 0;

                    if *current_phase < schedule.len() {
                        let (next_optimizer, _) = &schedule[*current_phase];
                        self.current_optimizer = Some(next_optimizer.clone());
                    }
                }
            }
        }
        Ok(())
    }

    fn execute_ensemble(
        &mut self,
        weights: &HashMap<String, f32>,
        combination_method: &CombinationMethod,
    ) -> OptimizerResult<()> {
        // Store parameter updates from each optimizer
        let mut updates = HashMap::new();

        // Collect updates first to avoid borrowing conflicts
        let mut state_diffs = Vec::new();
        for (name, optimizer) in &mut self.optimizers {
            if weights.contains_key(name) {
                // Get parameters before step
                let state_before = optimizer.state_dict()?;

                // Perform optimizer step
                optimizer.step()?;

                // Get parameters after step
                let state_after = optimizer.state_dict()?;

                state_diffs.push((name.clone(), state_before, state_after));
            }
        }

        // Compute updates after collecting all state diffs
        for (name, state_before, state_after) in state_diffs {
            let update = self.compute_parameter_update(&state_before, &state_after)?;
            updates.insert(name, update);
        }

        // Combine updates according to strategy
        if !updates.is_empty() {
            let combined_update = self.combine_updates(&updates, weights, combination_method)?;
            self.apply_combined_update(&combined_update)?;
        }

        Ok(())
    }

    fn execute_adaptive(
        &mut self,
        switch_criterion: &SwitchCriterion,
        evaluation_window: usize,
    ) -> OptimizerResult<()> {
        // Evaluate current optimizer performance
        if let Some(current_name) = &self.current_optimizer.clone() {
            if let Some(current_optimizer) = self.optimizers.get_mut(current_name) {
                current_optimizer.step()?;

                // Check if we should switch
                if let Some(current_metrics) = self.metrics.get(current_name) {
                    let should_switch = match switch_criterion {
                        SwitchCriterion::LossImprovement { threshold } => {
                            if current_metrics.loss_history.len() >= evaluation_window {
                                let recent_losses = &current_metrics.loss_history
                                    [current_metrics.loss_history.len() - evaluation_window..];
                                let improvement =
                                    recent_losses[0] - recent_losses[recent_losses.len() - 1];
                                improvement < *threshold
                            } else {
                                false
                            }
                        }
                        SwitchCriterion::GradientMagnitude { threshold } => {
                            if let Some(&last_gradient_norm) = current_metrics.gradient_norms.last()
                            {
                                last_gradient_norm < *threshold
                            } else {
                                false
                            }
                        }
                        SwitchCriterion::ConvergenceRate { window } => {
                            current_metrics.convergence_rate < 0.001
                                && current_metrics.loss_history.len() >= *window
                        }
                        SwitchCriterion::Custom(criterion_fn) => criterion_fn(current_metrics),
                    };

                    if should_switch {
                        self.switch_to_best_optimizer()?;
                    }
                }
            }
        } else {
            // No current optimizer, select the best one
            self.switch_to_best_optimizer()?;
        }

        Ok(())
    }

    fn execute_consensus(
        &mut self,
        agreement_threshold: f32,
        voting_method: &VotingMethod,
    ) -> OptimizerResult<()> {
        // Collect votes (parameter updates) from all optimizers
        let mut votes = HashMap::new();

        // Collect state diffs first to avoid borrowing conflicts
        let mut state_diffs = Vec::new();
        for (name, optimizer) in &mut self.optimizers {
            let state_before = optimizer.state_dict()?;
            optimizer.step()?;
            let state_after = optimizer.state_dict()?;

            state_diffs.push((name.clone(), state_before, state_after));
        }

        // Compute updates after collecting all state diffs
        for (name, state_before, state_after) in state_diffs {
            let update = self.compute_parameter_update(&state_before, &state_after)?;
            votes.insert(name, update);
        }

        // Apply voting mechanism
        if !votes.is_empty() {
            let consensus_update = match voting_method {
                VotingMethod::Majority => self.majority_vote(&votes)?,
                VotingMethod::WeightedVote => self.weighted_vote(&votes)?,
                VotingMethod::Unanimous => self.unanimous_vote(&votes, agreement_threshold)?,
            };

            self.apply_combined_update(&consensus_update)?;
        }

        Ok(())
    }

    fn execute_hierarchical(&mut self, levels: &[CompositionLevel]) -> OptimizerResult<()> {
        // Get current level without borrowing the entire state
        let current_level_idx =
            if let CompositionState::Hierarchical { current_level } = &self.composition_state {
                *current_level
            } else {
                return Ok(());
            };

        if current_level_idx < levels.len() {
            let level = &levels[current_level_idx];

            // Execute the strategy for the current level
            // This would recursively handle the composition strategy for this level
            // For now, just execute the first optimizer in the level
            if let Some(optimizer_name) = level.optimizers.first() {
                if let Some(optimizer) = self.optimizers.get_mut(optimizer_name) {
                    optimizer.step()?;
                }
            }

            // Check if we should move to the next level
            // This would be based on some criteria (e.g., convergence, time, etc.)
            if self.should_advance_level() {
                if let CompositionState::Hierarchical { current_level } =
                    &mut self.composition_state
                {
                    *current_level += 1;
                }
            }
        }

        Ok(())
    }

    // Helper methods
    fn compute_parameter_update(
        &self,
        state_before: &OptimizerState,
        state_after: &OptimizerState,
    ) -> OptimizerResult<HashMap<String, Tensor>> {
        let mut updates = HashMap::new();

        // Compute difference for each parameter
        for (param_name, param_dict_after) in &state_after.state {
            if let Some(param_dict_before) = state_before.state.get(param_name) {
                for (state_name, tensor_after) in param_dict_after {
                    if let Some(tensor_before) = param_dict_before.get(state_name) {
                        let update = tensor_after.sub(tensor_before)?;
                        let full_name = format!("{param_name}_{state_name}");
                        updates.insert(full_name, update);
                    }
                }
            }
        }

        Ok(updates)
    }

    fn combine_updates(
        &self,
        updates: &HashMap<String, HashMap<String, Tensor>>,
        weights: &HashMap<String, f32>,
        combination_method: &CombinationMethod,
    ) -> OptimizerResult<HashMap<String, Tensor>> {
        let mut combined = HashMap::new();

        // Get all parameter names
        let mut all_param_names = std::collections::HashSet::new();
        for update_dict in updates.values() {
            for param_name in update_dict.keys() {
                all_param_names.insert(param_name.clone());
            }
        }

        // Combine each parameter
        for param_name in all_param_names {
            let mut param_updates = Vec::new();
            let mut param_weights = Vec::new();

            for (optimizer_name, update_dict) in updates {
                if let Some(param_update) = update_dict.get(&param_name) {
                    param_updates.push(param_update.clone());
                    param_weights.push(weights.get(optimizer_name).copied().unwrap_or(1.0));
                }
            }

            if !param_updates.is_empty() {
                let combined_update = match combination_method {
                    CombinationMethod::WeightedAverage => {
                        self.weighted_average(&param_updates, &param_weights)?
                    }
                    CombinationMethod::Median => self.median_update(&param_updates)?,
                    CombinationMethod::BestWins => {
                        // Use update from best performing optimizer
                        param_updates[0].clone() // Simplified
                    }
                    CombinationMethod::Custom(combine_fn) => combine_fn(&param_updates),
                };

                combined.insert(param_name, combined_update);
            }
        }

        Ok(combined)
    }

    fn weighted_average(&self, tensors: &[Tensor], weights: &[f32]) -> OptimizerResult<Tensor> {
        if tensors.is_empty() || weights.is_empty() || tensors.len() != weights.len() {
            return Err(OptimizerError::InvalidParameter(
                "Mismatched tensors and weights".to_string(),
            ));
        }

        let weight_sum: f32 = weights.iter().sum();
        if weight_sum == 0.0 {
            return Err(OptimizerError::InvalidParameter(
                "Zero weight sum".to_string(),
            ));
        }

        let mut result = tensors[0].mul_scalar(weights[0] / weight_sum)?;
        for i in 1..tensors.len() {
            let weighted_tensor = tensors[i].mul_scalar(weights[i] / weight_sum)?;
            result = result.add(&weighted_tensor)?;
        }

        Ok(result)
    }

    fn median_update(&self, tensors: &[Tensor]) -> OptimizerResult<Tensor> {
        if tensors.is_empty() {
            return Err(OptimizerError::InvalidParameter(
                "Empty tensor list".to_string(),
            ));
        }

        if tensors.len() == 1 {
            return Ok(tensors[0].clone());
        }

        // For simplicity, just return the middle tensor
        // In practice, would compute element-wise median
        let median_idx = tensors.len() / 2;
        Ok(tensors[median_idx].clone())
    }

    fn majority_vote(
        &self,
        votes: &HashMap<String, HashMap<String, Tensor>>,
    ) -> OptimizerResult<HashMap<String, Tensor>> {
        // Simplified majority vote - just average all votes
        let mut combined = HashMap::new();
        let mut param_counts = HashMap::new();

        for vote_dict in votes.values() {
            for (param_name, param_tensor) in vote_dict {
                combined
                    .entry(param_name.clone())
                    .and_modify(|t: &mut Tensor| *t = t.add(param_tensor).unwrap())
                    .or_insert(param_tensor.clone());
                *param_counts.entry(param_name.clone()).or_insert(0) += 1;
            }
        }

        // Average by count
        for (param_name, tensor) in &mut combined {
            if let Some(&count) = param_counts.get(param_name) {
                if count > 1 {
                    *tensor = tensor.div_scalar(count as f32)?;
                }
            }
        }

        Ok(combined)
    }

    fn weighted_vote(
        &self,
        votes: &HashMap<String, HashMap<String, Tensor>>,
    ) -> OptimizerResult<HashMap<String, Tensor>> {
        // Weight votes by optimizer performance
        let mut weights = HashMap::new();
        for optimizer_name in votes.keys() {
            if let Some(metrics) = self.metrics.get(optimizer_name) {
                let weight = metrics.efficiency_score * metrics.stability_score;
                weights.insert(optimizer_name.clone(), weight);
            } else {
                weights.insert(optimizer_name.clone(), 1.0);
            }
        }

        // Apply weighted combination
        self.combine_updates(votes, &weights, &CombinationMethod::WeightedAverage)
    }

    fn unanimous_vote(
        &self,
        votes: &HashMap<String, HashMap<String, Tensor>>,
        agreement_threshold: f32,
    ) -> OptimizerResult<HashMap<String, Tensor>> {
        // Only apply updates where optimizers agree (within threshold)
        let mut unanimous_updates = HashMap::new();

        // Get all parameter names
        let mut all_params = std::collections::HashSet::new();
        for vote_dict in votes.values() {
            for param_name in vote_dict.keys() {
                all_params.insert(param_name.clone());
            }
        }

        for param_name in all_params {
            let mut param_votes = Vec::new();

            for vote_dict in votes.values() {
                if let Some(param_tensor) = vote_dict.get(&param_name) {
                    param_votes.push(param_tensor.clone());
                }
            }

            if param_votes.len() > 1 {
                // Check agreement (simplified - use variance)
                let mean = self.compute_mean_tensor(&param_votes)?;
                let variance = self.compute_variance_tensor(&param_votes, &mean)?;
                let variance_norm = variance.norm()?.item()?;

                if variance_norm < agreement_threshold {
                    unanimous_updates.insert(param_name, mean);
                }
            } else if param_votes.len() == 1 {
                unanimous_updates.insert(param_name, param_votes[0].clone());
            }
        }

        Ok(unanimous_updates)
    }

    fn compute_mean_tensor(&self, tensors: &[Tensor]) -> OptimizerResult<Tensor> {
        if tensors.is_empty() {
            return Err(OptimizerError::InvalidParameter(
                "Empty tensor list".to_string(),
            ));
        }

        let mut sum = tensors[0].clone();
        for tensor in tensors.iter().skip(1) {
            sum = sum.add(tensor)?;
        }

        Ok(sum.div_scalar(tensors.len() as f32)?)
    }

    fn compute_variance_tensor(
        &self,
        tensors: &[Tensor],
        mean: &Tensor,
    ) -> OptimizerResult<Tensor> {
        if tensors.is_empty() {
            return Err(OptimizerError::InvalidParameter(
                "Empty tensor list".to_string(),
            ));
        }

        let mut variance = tensors[0].sub(mean)?.pow_scalar(2.0)?;
        for tensor in tensors.iter().skip(1) {
            let diff = tensor.sub(mean)?.pow_scalar(2.0)?;
            variance = variance.add(&diff)?;
        }

        Ok(variance.div_scalar(tensors.len() as f32)?)
    }

    fn apply_combined_update(&mut self, _update: &HashMap<String, Tensor>) -> OptimizerResult<()> {
        // Apply the combined update to the actual parameters
        // This would require access to the actual model parameters
        // For now, this is a placeholder
        Ok(())
    }

    fn switch_to_best_optimizer(&mut self) -> OptimizerResult<()> {
        let best_name = self.best_optimizer().map(|s| s.to_string());
        if let Some(best_name) = best_name {
            self.current_optimizer = Some(best_name.clone());
            log::info!("Switched to optimizer: {best_name}");
        }
        Ok(())
    }

    fn should_advance_level(&self) -> bool {
        // Placeholder logic for hierarchical advancement
        // In practice, this would check convergence criteria, time limits, etc.
        false
    }
}

impl Optimizer for ComposedOptimizer {
    fn step(&mut self) -> OptimizerResult<()> {
        self.step_count += 1;
        self.execute_strategy()
    }

    fn zero_grad(&mut self) {
        for optimizer in self.optimizers.values_mut() {
            optimizer.zero_grad();
        }
    }

    fn get_lr(&self) -> Vec<f32> {
        // Return learning rates from all optimizers
        let mut all_lrs = Vec::new();
        for optimizer in self.optimizers.values() {
            all_lrs.extend(optimizer.get_lr());
        }
        all_lrs
    }

    fn set_lr(&mut self, lr: f32) {
        for optimizer in self.optimizers.values_mut() {
            optimizer.set_lr(lr);
        }
    }

    fn add_param_group(&mut self, params: Vec<Arc<RwLock<Tensor>>>, options: HashMap<String, f32>) {
        for optimizer in self.optimizers.values_mut() {
            optimizer.add_param_group(params.clone(), options.clone());
        }
    }

    fn state_dict(&self) -> OptimizerResult<OptimizerState> {
        // Combine state from all optimizers
        let mut combined_state = HashMap::new();
        let mut combined_param_groups = Vec::new();

        for (name, optimizer) in &self.optimizers {
            let state = optimizer.state_dict()?;

            // Add optimizer name prefix to avoid conflicts
            for (param_id, param_state) in state.state {
                let prefixed_id = format!("{name}_{param_id}");
                combined_state.insert(prefixed_id, param_state);
            }

            combined_param_groups.extend(state.param_groups);
        }

        Ok(OptimizerState {
            optimizer_type: "CompositeOptimizer".to_string(),
            version: "0.1.0".to_string(),
            param_groups: combined_param_groups,
            state: combined_state,
            global_state: HashMap::new(),
        })
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> OptimizerResult<()> {
        // Split state back to individual optimizers
        for (optimizer_name, optimizer) in &mut self.optimizers {
            let mut optimizer_state = HashMap::new();
            let prefix = format!("{optimizer_name}_");

            for (param_id, param_state) in &state.state {
                if param_id.starts_with(&prefix) {
                    let unprefixed_id = param_id.strip_prefix(&prefix).unwrap().to_string();
                    optimizer_state.insert(unprefixed_id, param_state.clone());
                }
            }

            let optimizer_state_dict = OptimizerState {
                optimizer_type: "CompositeOptimizer".to_string(),
                version: "0.1.0".to_string(),
                param_groups: state.param_groups.clone(),
                state: optimizer_state,
                global_state: HashMap::new(),
            };

            optimizer.load_state_dict(optimizer_state_dict)?;
        }

        Ok(())
    }
}

/// Builder for composed optimizers
pub struct CompositionBuilder {
    strategy: Option<CompositionStrategy>,
    optimizers: HashMap<String, Box<dyn Optimizer>>,
}

impl Default for CompositionBuilder {
    fn default() -> Self {
        Self {
            strategy: None,
            optimizers: HashMap::new(),
        }
    }
}

impl CompositionBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn strategy(mut self, strategy: CompositionStrategy) -> Self {
        self.strategy = Some(strategy);
        self
    }

    pub fn add_optimizer(mut self, name: &str, optimizer: Box<dyn Optimizer>) -> Self {
        self.optimizers.insert(name.to_string(), optimizer);
        self
    }

    pub fn build(self) -> OptimizerResult<ComposedOptimizer> {
        let strategy = self.strategy.ok_or_else(|| {
            OptimizerError::ConfigError("No composition strategy specified".to_string())
        })?;

        let mut composed = ComposedOptimizer::new(strategy);
        for (name, optimizer) in self.optimizers {
            composed.add_optimizer(name, optimizer);
        }

        Ok(composed)
    }
}

/// Utility functions for optimizer composition
pub mod utils {
    use super::*;

    /// Create an ensemble of optimizers with equal weights
    pub fn equal_ensemble(
        optimizers: Vec<(&str, Box<dyn Optimizer>)>,
    ) -> OptimizerResult<ComposedOptimizer> {
        let mut weights = HashMap::new();
        let weight = 1.0 / optimizers.len() as f32;

        let mut builder = CompositionBuilder::new();
        for (name, optimizer) in optimizers {
            weights.insert(name.to_string(), weight);
            builder = builder.add_optimizer(name, optimizer);
        }

        let strategy = CompositionStrategy::Ensemble {
            weights,
            combination_method: CombinationMethod::WeightedAverage,
        };

        builder.strategy(strategy).build()
    }

    /// Create a sequential pipeline of optimizers
    pub fn sequential_pipeline(
        schedule: Vec<(&str, Box<dyn Optimizer>, usize)>,
    ) -> OptimizerResult<ComposedOptimizer> {
        let mut builder = CompositionBuilder::new();
        let mut strategy_schedule = Vec::new();

        for (name, optimizer, steps) in schedule {
            builder = builder.add_optimizer(name, optimizer);
            strategy_schedule.push((name.to_string(), steps));
        }

        let strategy = CompositionStrategy::Sequential {
            schedule: strategy_schedule,
        };

        builder.strategy(strategy).build()
    }

    /// Create an adaptive composition that switches based on loss improvement
    pub fn adaptive_switching(
        optimizers: Vec<(&str, Box<dyn Optimizer>)>,
        improvement_threshold: f32,
    ) -> OptimizerResult<ComposedOptimizer> {
        let mut builder = CompositionBuilder::new();

        for (name, optimizer) in optimizers {
            builder = builder.add_optimizer(name, optimizer);
        }

        let strategy = CompositionStrategy::Adaptive {
            switch_criterion: SwitchCriterion::LossImprovement {
                threshold: improvement_threshold,
            },
            evaluation_window: 10,
        };

        builder.strategy(strategy).build()
    }
}
