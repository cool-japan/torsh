// Copyright (c) 2025 ToRSh Contributors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! # Differentiable Compilation
//!
//! This module implements differentiable compilation where the compilation process itself
//! is differentiable, enabling gradient-based optimization of compilation decisions.
//!
//! ## Key Concepts
//!
//! - **Differentiable Optimization Passes**: Each optimization can compute gradients
//! - **Compilation Graph**: Represent compilation as a differentiable computation graph
//! - **Meta-Optimization**: Optimize compilation strategies using gradient descent
//! - **Soft Decision Making**: Use continuous relaxations of discrete choices
//! - **End-to-End Learning**: Jointly learn program behavior and compilation strategy
//!
//! ## Architecture
//!
//! ```text
//! Input Program → Soft Compilation Decisions → Optimized Program → Loss
//!                  ↑                                                  ↓
//!                  └────────────── Gradients ───────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use torsh_jit::differentiable_compilation::{DifferentiableCompiler, CompilationParams};
//!
//! let mut compiler = DifferentiableCompiler::new();
//!
//! // Define compilation parameters (learnable)
//! let mut params = CompilationParams::new();
//!
//! // Compile with soft decisions
//! let result = compiler.compile_differentiable(&graph, &params)?;
//!
//! // Compute gradients based on performance
//! let grads = compiler.backward(&result, performance_loss)?;
//!
//! // Update compilation strategy
//! params.update(&grads, learning_rate);
//! ```

use crate::graph::{ComputationGraph, NodeId};
// use crate::optimizer::OptimizationPass; // Reserved for future integration
use crate::JitResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ============================================================================
// Differentiable Parameters
// ============================================================================

/// Learnable compilation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationParams {
    /// Weights for each optimization pass (0-1, continuous)
    pub pass_weights: HashMap<String, f32>,

    /// Fusion temperature (controls fusion aggressiveness)
    pub fusion_temp: f32,

    /// Loop unrolling factor (continuous relaxation)
    pub unroll_factor: f32,

    /// Vectorization width (soft selection)
    pub vector_width: f32,

    /// Memory layout preference (continuous)
    pub layout_preference: f32,

    /// Gradients for each parameter
    #[serde(skip)]
    pub gradients: HashMap<String, f32>,
}

impl CompilationParams {
    /// Create new parameters with default initialization
    pub fn new() -> Self {
        let mut pass_weights = HashMap::new();

        // Initialize pass weights (will be learned)
        for pass_name in [
            "constant_folding",
            "dead_code_elimination",
            "common_subexpression_elimination",
            "loop_invariant_motion",
            "strength_reduction",
            "fusion",
            "vectorization",
            "parallelization",
        ] {
            pass_weights.insert(pass_name.to_string(), 0.5); // Neutral initialization
        }

        Self {
            pass_weights,
            fusion_temp: 1.0,
            unroll_factor: 4.0,
            vector_width: 4.0,
            layout_preference: 0.5,
            gradients: HashMap::new(),
        }
    }

    /// Update parameters using gradients (gradient descent)
    pub fn update(&mut self, learning_rate: f32) {
        // Update pass weights
        for (name, weight) in &mut self.pass_weights {
            if let Some(&grad) = self.gradients.get(name) {
                *weight -= learning_rate * grad;
                *weight = weight.clamp(0.0, 1.0); // Keep in valid range
            }
        }

        // Update other parameters
        if let Some(&grad) = self.gradients.get("fusion_temp") {
            self.fusion_temp -= learning_rate * grad;
            self.fusion_temp = self.fusion_temp.max(0.1); // Avoid zero/negative
        }

        if let Some(&grad) = self.gradients.get("unroll_factor") {
            self.unroll_factor -= learning_rate * grad;
            self.unroll_factor = self.unroll_factor.clamp(1.0, 32.0);
        }

        if let Some(&grad) = self.gradients.get("vector_width") {
            self.vector_width -= learning_rate * grad;
            self.vector_width = self.vector_width.clamp(1.0, 16.0);
        }

        if let Some(&grad) = self.gradients.get("layout_preference") {
            self.layout_preference -= learning_rate * grad;
            self.layout_preference = self.layout_preference.clamp(0.0, 1.0);
        }

        // Clear gradients after update
        self.gradients.clear();
    }

    /// Zero out all gradients
    pub fn zero_grad(&mut self) {
        self.gradients.clear();
    }

    /// Accumulate gradients
    pub fn accumulate_grad(&mut self, name: &str, grad: f32) {
        *self.gradients.entry(name.to_string()).or_insert(0.0) += grad;
    }
}

impl Default for CompilationParams {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Soft Operations
// ============================================================================

/// Soft (differentiable) version of binary decisions
#[derive(Debug, Clone)]
pub struct SoftDecision {
    /// Probability of taking the decision (0-1)
    pub probability: f32,

    /// Gradient with respect to probability
    pub gradient: f32,
}

impl SoftDecision {
    /// Create a new soft decision
    pub fn new(probability: f32) -> Self {
        Self {
            probability: probability.clamp(0.0, 1.0),
            gradient: 0.0,
        }
    }

    /// Apply soft decision (weighted combination)
    pub fn apply<T: Clone>(&self, if_true: T, if_false: T, blend_fn: fn(&T, &T, f32) -> T) -> T {
        blend_fn(&if_true, &if_false, self.probability)
    }

    /// Backward pass for soft decision
    pub fn backward(&mut self, upstream_grad: f32) {
        self.gradient += upstream_grad;
    }

    /// Sigmoid activation for smooth decisions
    pub fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Softmax for multi-way decisions
    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_values: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exp_values.iter().sum();
        exp_values.iter().map(|&x| x / sum).collect()
    }
}

// ============================================================================
// Differentiable Compilation Result
// ============================================================================

/// Result of differentiable compilation
#[derive(Debug, Clone)]
pub struct DiffCompilationResult {
    /// Compiled graph (soft-compiled)
    pub graph: ComputationGraph,

    /// Compilation decisions made (with probabilities)
    pub decisions: Vec<CompilationDecision>,

    /// Estimated performance metrics
    pub estimated_performance: PerformanceMetrics,

    /// Computation tape for backward pass
    pub tape: Arc<Mutex<ComputationTape>>,
}

/// A single compilation decision
#[derive(Debug, Clone)]
pub struct CompilationDecision {
    /// Name of the decision
    pub name: String,

    /// Decision type
    pub decision_type: DecisionType,

    /// Soft decision value
    pub decision: SoftDecision,

    /// Impact on performance (estimated)
    pub impact: f32,
}

/// Types of compilation decisions
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionType {
    /// Whether to apply an optimization pass
    ApplyOptimization(String),

    /// Whether to fuse two operations
    FuseOperations(NodeId, NodeId),

    /// Loop unrolling decision
    UnrollLoop(usize),

    /// Vectorization decision
    Vectorize(usize),

    /// Memory layout choice
    MemoryLayout(String),
}

/// Performance metrics
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    /// Estimated execution time (microseconds)
    pub exec_time_us: f32,

    /// Estimated memory usage (bytes)
    pub memory_bytes: f32,

    /// Estimated FLOPs
    pub flops: f32,

    /// Cache efficiency score (0-1)
    pub cache_efficiency: f32,
}

/// Computation tape for automatic differentiation
#[derive(Debug, Clone, Default)]
pub struct ComputationTape {
    /// Recorded operations
    pub operations: Vec<TapeOperation>,

    /// Variable gradients
    pub gradients: HashMap<String, f32>,
}

/// A single operation in the tape
#[derive(Debug, Clone)]
pub struct TapeOperation {
    /// Operation name
    pub name: String,

    /// Input variables
    pub inputs: Vec<String>,

    /// Output variable
    pub output: String,

    /// Forward function
    pub forward_val: f32,

    /// Gradient function (chain rule)
    pub grad_fn: GradientFunction,
}

/// Gradient computation function
#[derive(Debug, Clone)]
pub enum GradientFunction {
    /// Linear: dy/dx = a
    Linear(f32),

    /// Product: dy/dx = other_input
    Product(f32),

    /// Sigmoid: dy/dx = sigmoid(x) * (1 - sigmoid(x))
    Sigmoid,

    /// ReLU: dy/dx = x > 0 ? 1 : 0
    ReLU,

    /// Custom gradient function
    Custom(fn(f32, f32) -> f32),
}

// ============================================================================
// Differentiable Compiler
// ============================================================================

/// Main differentiable compilation engine
pub struct DifferentiableCompiler {
    /// Configuration
    config: DiffCompilerConfig,

    /// Training statistics
    stats: CompilerStatistics,
}

/// Configuration for differentiable compiler
#[derive(Debug, Clone)]
pub struct DiffCompilerConfig {
    /// Enable gradient checkpointing to save memory
    pub gradient_checkpointing: bool,

    /// Use straight-through estimators for discrete operations
    pub straight_through: bool,

    /// Temperature for Gumbel-Softmax relaxation
    pub gumbel_temperature: f32,

    /// Gradient clipping threshold
    pub grad_clip: f32,
}

impl Default for DiffCompilerConfig {
    fn default() -> Self {
        Self {
            gradient_checkpointing: true,
            straight_through: true,
            gumbel_temperature: 1.0,
            grad_clip: 10.0,
        }
    }
}

/// Compiler statistics
#[derive(Debug, Clone, Default)]
pub struct CompilerStatistics {
    /// Number of compilations
    pub compilations: usize,

    /// Total gradient updates
    pub gradient_updates: usize,

    /// Average loss
    pub avg_loss: f32,

    /// Best performance achieved
    pub best_performance: f32,
}

impl DifferentiableCompiler {
    /// Create a new differentiable compiler
    pub fn new() -> Self {
        Self::with_config(DiffCompilerConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: DiffCompilerConfig) -> Self {
        Self {
            config,
            stats: CompilerStatistics::default(),
        }
    }

    /// Compile a graph with differentiable decisions
    pub fn compile_differentiable(
        &mut self,
        graph: &ComputationGraph,
        params: &CompilationParams,
    ) -> JitResult<DiffCompilationResult> {
        let mut tape = ComputationTape::default();
        let mut decisions = Vec::new();

        // Create a copy of the graph for modification
        let mut compiled_graph = graph.clone();

        // Apply soft optimization passes
        for (pass_name, &weight) in &params.pass_weights {
            let decision = SoftDecision::new(weight);

            // Record decision
            decisions.push(CompilationDecision {
                name: pass_name.clone(),
                decision_type: DecisionType::ApplyOptimization(pass_name.clone()),
                decision: decision.clone(),
                impact: self.estimate_pass_impact(pass_name, graph),
            });

            // Apply optimization with probability weight
            if weight > 0.5 {
                // For now, deterministic application
                // In full implementation, would use soft blending
                compiled_graph =
                    self.apply_soft_optimization(&compiled_graph, pass_name, weight)?;
            }

            // Record in tape
            tape.operations.push(TapeOperation {
                name: format!("apply_{}", pass_name),
                inputs: vec!["graph".to_string()],
                output: "graph".to_string(),
                forward_val: weight,
                grad_fn: GradientFunction::Linear(1.0),
            });
        }

        // Estimate performance
        let estimated_performance = self.estimate_performance(&compiled_graph, params);

        self.stats.compilations += 1;

        Ok(DiffCompilationResult {
            graph: compiled_graph,
            decisions,
            estimated_performance,
            tape: Arc::new(Mutex::new(tape)),
        })
    }

    /// Backward pass to compute gradients
    pub fn backward(
        &mut self,
        result: &DiffCompilationResult,
        loss: f32,
    ) -> JitResult<CompilationParams> {
        let mut params_grad = CompilationParams::new();
        params_grad.zero_grad();

        // Compute gradients for each decision
        for decision in &result.decisions {
            match &decision.decision_type {
                DecisionType::ApplyOptimization(pass_name) => {
                    // Simple gradient: dL/dw = dL/dperf * dperf/dw
                    // Approximate: if optimization helps, gradient is negative (minimize loss)
                    let grad = if decision.impact > 0.0 {
                        -loss * decision.impact
                    } else {
                        loss * decision.impact.abs()
                    };

                    params_grad.accumulate_grad(pass_name, grad);
                }
                _ => {
                    // Handle other decision types
                }
            }
        }

        // Gradient clipping
        for (_name, grad) in &mut params_grad.gradients {
            *grad = grad.clamp(-self.config.grad_clip, self.config.grad_clip);
        }

        self.stats.gradient_updates += 1;
        self.stats.avg_loss = (self.stats.avg_loss * (self.stats.gradient_updates - 1) as f32
            + loss)
            / self.stats.gradient_updates as f32;

        Ok(params_grad)
    }

    /// Apply soft optimization (weighted)
    fn apply_soft_optimization(
        &self,
        graph: &ComputationGraph,
        pass_name: &str,
        _weight: f32,
    ) -> JitResult<ComputationGraph> {
        // Simplified: just return graph (full implementation would apply soft transformations)
        // In practice, would blend original and optimized versions based on weight
        log::debug!("Applying soft optimization: {} with weight", pass_name);
        Ok(graph.clone())
    }

    /// Estimate impact of an optimization pass
    fn estimate_pass_impact(&self, pass_name: &str, _graph: &ComputationGraph) -> f32 {
        // Heuristic estimates (in production, would be learned)
        match pass_name {
            "constant_folding" => 0.1,
            "dead_code_elimination" => 0.15,
            "common_subexpression_elimination" => 0.2,
            "fusion" => 0.3,
            "vectorization" => 0.4,
            "parallelization" => 0.5,
            _ => 0.05,
        }
    }

    /// Estimate performance of compiled graph
    fn estimate_performance(
        &self,
        graph: &ComputationGraph,
        params: &CompilationParams,
    ) -> PerformanceMetrics {
        let node_count = graph.node_count() as f32;

        // Simple performance model (in production, would use learned model)
        let base_time = node_count * 10.0; // 10 us per node

        // Apply optimization effects
        let mut speedup = 1.0;
        for (pass_name, &weight) in &params.pass_weights {
            let impact = self.estimate_pass_impact(pass_name, graph);
            speedup += weight * impact;
        }

        let exec_time_us = base_time / speedup;
        let memory_bytes = node_count * 1024.0; // Simplified

        PerformanceMetrics {
            exec_time_us,
            memory_bytes,
            flops: node_count * 100.0,
            cache_efficiency: 0.7 + params.layout_preference * 0.3,
        }
    }

    /// Get compiler statistics
    pub fn statistics(&self) -> &CompilerStatistics {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = CompilerStatistics::default();
    }
}

impl Default for DifferentiableCompiler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Gumbel-Softmax Trick
// ============================================================================

/// Gumbel-Softmax trick for differentiable discrete decisions
pub struct GumbelSoftmax {
    /// Temperature parameter
    temperature: f32,
}

impl GumbelSoftmax {
    /// Create new Gumbel-Softmax with temperature
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }

    /// Sample from Gumbel distribution
    fn sample_gumbel(&self) -> f32 {
        // Note: In production, use proper random sampling from scirs2-core
        // For now, using a simple hash-based pseudo-random approach
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after UNIX_EPOCH")
            .subsec_nanos();
        let u = ((nanos % 1000) as f32 / 1000.0).max(1e-10);
        -(-u).ln().ln()
    }

    /// Apply Gumbel-Softmax to logits
    pub fn apply(&self, logits: &[f32]) -> Vec<f32> {
        let gumbel_logits: Vec<f32> = logits
            .iter()
            .map(|&logit| (logit + self.sample_gumbel()) / self.temperature)
            .collect();

        SoftDecision::softmax(&gumbel_logits)
    }

    /// Straight-through estimator (forward: argmax, backward: softmax)
    pub fn straight_through(&self, logits: &[f32]) -> (usize, Vec<f32>) {
        let probs = SoftDecision::softmax(logits);

        // Forward: argmax (discrete)
        let choice = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Backward: use softmax gradients
        (choice, probs)
    }
}

// ============================================================================
// Training Loop Helper
// ============================================================================

/// Helper for training compilation parameters
pub struct CompilationTrainer {
    /// Compiler
    compiler: DifferentiableCompiler,

    /// Parameters to optimize
    params: CompilationParams,

    /// Learning rate
    learning_rate: f32,

    /// Training history
    history: Vec<TrainingEpoch>,
}

/// Training epoch record
#[derive(Debug, Clone)]
pub struct TrainingEpoch {
    /// Epoch number
    pub epoch: usize,

    /// Average loss
    pub loss: f32,

    /// Performance metrics
    pub performance: PerformanceMetrics,

    /// Parameter snapshot
    pub params: CompilationParams,
}

impl CompilationTrainer {
    /// Create new trainer
    pub fn new(learning_rate: f32) -> Self {
        Self {
            compiler: DifferentiableCompiler::new(),
            params: CompilationParams::new(),
            learning_rate,
            history: Vec::new(),
        }
    }

    /// Train on a single graph
    pub fn train_step(
        &mut self,
        graph: &ComputationGraph,
        target_performance: f32,
    ) -> JitResult<f32> {
        // Forward pass
        let result = self.compiler.compile_differentiable(graph, &self.params)?;

        // Compute loss (MSE on execution time)
        let loss = (result.estimated_performance.exec_time_us - target_performance).powi(2);

        // Backward pass
        let grads = self.compiler.backward(&result, loss)?;

        // Update parameters
        self.params.gradients = grads.gradients;
        self.params.update(self.learning_rate);

        Ok(loss)
    }

    /// Train for multiple epochs
    pub fn train(
        &mut self,
        graphs: &[ComputationGraph],
        targets: &[f32],
        epochs: usize,
    ) -> JitResult<Vec<TrainingEpoch>> {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (graph, &target) in graphs.iter().zip(targets.iter()) {
                let loss = self.train_step(graph, target)?;
                total_loss += loss;
            }

            let avg_loss = total_loss / graphs.len() as f32;

            // Record epoch
            let result = self
                .compiler
                .compile_differentiable(&graphs[0], &self.params)?;
            self.history.push(TrainingEpoch {
                epoch,
                loss: avg_loss,
                performance: result.estimated_performance.clone(),
                params: self.params.clone(),
            });

            log::info!("Epoch {}: loss = {:.4}", epoch, avg_loss);
        }

        Ok(self.history.clone())
    }

    /// Get best parameters from training
    pub fn best_params(&self) -> &CompilationParams {
        self.history
            .iter()
            .min_by(|a, b| {
                a.loss
                    .partial_cmp(&b.loss)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|e| &e.params)
            .unwrap_or(&self.params)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;
    use torsh_core::{DType, Shape};

    #[test]
    fn test_compilation_params() {
        let mut params = CompilationParams::new();
        assert!(params.pass_weights.len() > 0);

        params.accumulate_grad("fusion", 0.1);
        params.update(0.01);

        assert!(params.pass_weights.contains_key("fusion"));
    }

    #[test]
    fn test_soft_decision() {
        let decision = SoftDecision::new(0.7);
        assert!((decision.probability - 0.7).abs() < 1e-6);

        let probs = SoftDecision::softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_differentiable_compilation() {
        let mut compiler = DifferentiableCompiler::new();
        let params = CompilationParams::new();

        let mut builder = GraphBuilder::new();
        let x = builder.add_input("x".to_string(), Shape::new(vec![10, 10]), DType::F32);
        builder.mark_output(x).unwrap();

        let graph = builder.build().unwrap();
        let result = compiler.compile_differentiable(&graph, &params).unwrap();

        assert!(result.decisions.len() > 0);
        assert!(result.estimated_performance.exec_time_us > 0.0);
    }

    #[test]
    fn test_backward_pass() {
        let mut compiler = DifferentiableCompiler::new();
        let params = CompilationParams::new();

        let mut builder = GraphBuilder::new();
        let x = builder.add_input("x".to_string(), Shape::new(vec![5, 5]), DType::F32);
        builder.mark_output(x).unwrap();

        let graph = builder.build().unwrap();
        let result = compiler.compile_differentiable(&graph, &params).unwrap();

        let loss = 100.0; // Simulated loss
        let grads = compiler.backward(&result, loss).unwrap();

        assert!(grads.gradients.len() > 0);
    }

    #[test]
    fn test_compilation_trainer() {
        let mut trainer = CompilationTrainer::new(0.01);

        let mut builder = GraphBuilder::new();
        let x = builder.add_input("x".to_string(), Shape::new(vec![3, 3]), DType::F32);
        builder.mark_output(x).unwrap();

        let graph = builder.build().unwrap();
        let loss = trainer.train_step(&graph, 50.0).unwrap();

        assert!(loss >= 0.0);
    }

    #[test]
    fn test_gumbel_softmax() {
        let gumbel = GumbelSoftmax::new(1.0);
        let logits = vec![1.0, 2.0, 3.0];

        let (choice, probs) = gumbel.straight_through(&logits);
        assert!(choice < 3);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
