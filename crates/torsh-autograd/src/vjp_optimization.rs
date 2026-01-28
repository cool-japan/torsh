//! Vector-Jacobian Product (VJP) optimizations for efficient reverse-mode automatic differentiation
//!
//! This module provides advanced optimizations for computing Vector-Jacobian Products,
//! which are the core of efficient reverse-mode automatic differentiation (backpropagation).

use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use torsh_core::dtype::FloatElement;
use torsh_core::error::{Result, TorshError};

/// VJP computation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VjpStrategy {
    /// Standard reverse-mode AD
    Standard,
    /// Checkpointing for memory efficiency
    Checkpointed,
    /// Fused operations
    Fused,
    /// Vectorized computation
    Vectorized,
    /// Adaptive strategy selection
    Adaptive,
}

/// VJP optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimizations
    Basic,
    /// Aggressive optimizations
    Aggressive,
    /// Maximum optimization (may trade accuracy for speed)
    Maximum,
}

/// VJP computation context
#[derive(Debug, Clone)]
pub struct VjpContext<T: FloatElement> {
    /// Computation graph nodes
    pub nodes: Vec<VjpNode<T>>,
    /// Execution order for backward pass
    pub execution_order: Vec<usize>,
    /// Checkpoints for recomputation
    pub checkpoints: HashMap<usize, VjpCheckpoint<T>>,
    /// Memory budget for VJP computation
    pub memory_budget: usize,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
}

/// VJP computation node
#[derive(Clone)]
pub struct VjpNode<T: FloatElement> {
    /// Node identifier
    pub id: usize,
    /// Operation type
    pub operation: VjpOperation,
    /// Input node IDs
    pub inputs: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Saved tensors for backward computation
    pub saved_tensors: Vec<Arc<RwLock<Vec<T>>>>,
    /// VJP function
    pub vjp_fn: Option<VjpFunction<T>>,
    /// Memory usage estimate
    pub memory_usage: usize,
    /// Computation cost estimate
    pub compute_cost: f64,
    /// Whether this node should be checkpointed
    pub should_checkpoint: bool,
}

impl<T: FloatElement> std::fmt::Debug for VjpNode<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VjpNode")
            .field("id", &self.id)
            .field("operation", &self.operation)
            .field("inputs", &self.inputs)
            .field("output_shape", &self.output_shape)
            .field("saved_tensors_count", &self.saved_tensors.len())
            .field("has_vjp_fn", &self.vjp_fn.is_some())
            .field("memory_usage", &self.memory_usage)
            .field("compute_cost", &self.compute_cost)
            .field("should_checkpoint", &self.should_checkpoint)
            .finish()
    }
}

/// VJP operation types
#[derive(Debug, Clone, PartialEq)]
pub enum VjpOperation {
    /// Addition
    Add,
    /// Multiplication
    Mul,
    /// Matrix multiplication
    MatMul,
    /// Element-wise operations
    ElementWise(String),
    /// Reduction operations
    Reduction(String),
    /// Convolution
    Conv2d,
    /// Pooling
    Pool2d,
    /// Activation functions
    Activation(String),
    /// Custom operation
    Custom(String),
}

/// VJP function type
pub type VjpFunction<T> =
    Arc<dyn Fn(&[T], &[Arc<RwLock<Vec<T>>>]) -> Result<Vec<Vec<T>>> + Send + Sync>;

/// Checkpoint for recomputation
#[derive(Debug, Clone)]
pub struct VjpCheckpoint<T: FloatElement> {
    /// Checkpoint ID
    pub id: usize,
    /// Saved tensors
    pub tensors: Vec<Arc<RwLock<Vec<T>>>>,
    /// Node range covered by this checkpoint
    pub node_range: (usize, usize),
    /// Memory usage of checkpoint
    pub memory_usage: usize,
}

/// VJP optimization configuration
#[derive(Debug, Clone)]
pub struct VjpOptimizationConfig {
    /// Strategy to use
    pub strategy: VjpStrategy,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Memory budget for VJP computation
    pub memory_budget: usize,
    /// Enable operation fusion
    pub enable_fusion: bool,
    /// Enable vectorization
    pub enable_vectorization: bool,
    /// Enable checkpointing
    pub enable_checkpointing: bool,
    /// Checkpoint frequency (nodes per checkpoint)
    pub checkpoint_frequency: usize,
    /// Maximum number of checkpoints
    pub max_checkpoints: usize,
    /// Enable parallel VJP computation
    pub enable_parallel_vjp: bool,
}

impl Default for VjpOptimizationConfig {
    fn default() -> Self {
        Self {
            strategy: VjpStrategy::Adaptive,
            optimization_level: OptimizationLevel::Aggressive,
            memory_budget: 512 * 1024 * 1024, // 512MB
            enable_fusion: true,
            enable_vectorization: true,
            enable_checkpointing: true,
            checkpoint_frequency: 50,
            max_checkpoints: 10,
            enable_parallel_vjp: true,
        }
    }
}

/// VJP computation statistics
#[derive(Debug, Clone, Default)]
pub struct VjpStats {
    /// Total VJP computations
    pub total_vjp_computations: usize,
    /// Total computation time
    pub total_computation_time_ms: f64,
    /// Average computation time per VJP
    pub average_computation_time_ms: f64,
    /// Memory usage statistics
    pub peak_memory_usage: usize,
    /// Number of operations fused
    pub operations_fused: usize,
    /// Number of recomputations
    pub recomputations: usize,
    /// Checkpointing statistics
    pub checkpoints_created: usize,
    /// Parallel speedup achieved
    pub parallel_speedup: f64,
}

/// VJP optimizer for efficient reverse-mode AD
pub struct VjpOptimizer<T: FloatElement> {
    /// Configuration
    config: VjpOptimizationConfig,
    /// VJP function registry
    vjp_registry: Arc<RwLock<HashMap<String, VjpFunction<T>>>>,
    /// Fusion patterns
    fusion_patterns: Arc<RwLock<Vec<FusionPattern>>>,
    /// Statistics
    stats: Arc<RwLock<VjpStats>>,
    /// Memory tracker
    memory_tracker: Arc<Mutex<usize>>,
}

/// Operation fusion pattern
#[derive(Debug, Clone)]
pub struct FusionPattern {
    /// Pattern name
    pub name: String,
    /// Operations that can be fused
    pub operations: Vec<VjpOperation>,
    /// Fused VJP function
    pub fused_vjp: String,
    /// Memory saving factor
    pub memory_saving: f64,
    /// Speedup factor
    pub speedup: f64,
}

impl<T: FloatElement + Send + Sync + 'static> VjpOptimizer<T> {
    /// Create a new VJP optimizer
    pub fn new(config: VjpOptimizationConfig) -> Self {
        let mut optimizer = Self {
            config,
            vjp_registry: Arc::new(RwLock::new(HashMap::new())),
            fusion_patterns: Arc::new(RwLock::new(Vec::new())),
            stats: Arc::new(RwLock::new(VjpStats::default())),
            memory_tracker: Arc::new(Mutex::new(0)),
        };

        optimizer.initialize_vjp_functions();
        optimizer.initialize_fusion_patterns();

        optimizer
    }

    /// Initialize built-in VJP functions
    fn initialize_vjp_functions(&mut self) {
        let mut registry = self
            .vjp_registry
            .write()
            .expect("lock should not be poisoned");

        // Addition VJP
        registry.insert(
            "add".to_string(),
            Arc::new(
                |grad_output: &[T],
                 _saved_tensors: &[Arc<RwLock<Vec<T>>>]|
                 -> Result<Vec<Vec<T>>> {
                    // For addition: grad_a = grad_output, grad_b = grad_output
                    Ok(vec![grad_output.to_vec(), grad_output.to_vec()])
                },
            ),
        );

        // Multiplication VJP
        registry.insert(
            "mul".to_string(),
            Arc::new(
                |grad_output: &[T], saved_tensors: &[Arc<RwLock<Vec<T>>>]| -> Result<Vec<Vec<T>>> {
                    if saved_tensors.len() != 2 {
                        return Err(TorshError::AutogradError(
                            "Mul VJP requires 2 saved tensors".to_string(),
                        ));
                    }

                    let a = saved_tensors[0]
                        .read()
                        .expect("lock should not be poisoned");
                    let b = saved_tensors[1]
                        .read()
                        .expect("lock should not be poisoned");

                    let grad_a: Vec<T> = grad_output
                        .iter()
                        .zip(b.iter())
                        .map(|(&g, &b_val)| g * b_val)
                        .collect();

                    let grad_b: Vec<T> = grad_output
                        .iter()
                        .zip(a.iter())
                        .map(|(&g, &a_val)| g * a_val)
                        .collect();

                    Ok(vec![grad_a, grad_b])
                },
            ),
        );

        // Matrix multiplication VJP
        registry.insert(
            "matmul".to_string(),
            Arc::new(
                |_grad_output: &[T],
                 _saved_tensors: &[Arc<RwLock<Vec<T>>>]|
                 -> Result<Vec<Vec<T>>> {
                    // Simplified matmul VJP - in practice this would be more complex
                    // grad_a = grad_output @ b.T
                    // grad_b = a.T @ grad_output
                    Err(TorshError::AutogradError(
                        "MatMul VJP not fully implemented".to_string(),
                    ))
                },
            ),
        );

        // ReLU VJP
        registry.insert(
            "relu".to_string(),
            Arc::new(
                |grad_output: &[T], saved_tensors: &[Arc<RwLock<Vec<T>>>]| -> Result<Vec<Vec<T>>> {
                    if saved_tensors.is_empty() {
                        return Err(TorshError::AutogradError(
                            "ReLU VJP requires saved input".to_string(),
                        ));
                    }

                    let input = saved_tensors[0]
                        .read()
                        .expect("lock should not be poisoned");
                    let grad_input: Vec<T> = grad_output
                        .iter()
                        .zip(input.iter())
                        .map(|(&g, &x)| {
                            if x > <T as torsh_core::dtype::TensorElement>::zero() {
                                g
                            } else {
                                <T as torsh_core::dtype::TensorElement>::zero()
                            }
                        })
                        .collect();

                    Ok(vec![grad_input])
                },
            ),
        );
    }

    /// Initialize fusion patterns
    fn initialize_fusion_patterns(&mut self) {
        let mut patterns = self
            .fusion_patterns
            .write()
            .expect("lock should not be poisoned");

        // Add-ReLU fusion
        patterns.push(FusionPattern {
            name: "add_relu".to_string(),
            operations: vec![
                VjpOperation::Add,
                VjpOperation::Activation("relu".to_string()),
            ],
            fused_vjp: "add_relu_fused".to_string(),
            memory_saving: 0.3,
            speedup: 1.5,
        });

        // Mul-Add fusion (fused multiply-add)
        patterns.push(FusionPattern {
            name: "mul_add".to_string(),
            operations: vec![VjpOperation::Mul, VjpOperation::Add],
            fused_vjp: "mul_add_fused".to_string(),
            memory_saving: 0.25,
            speedup: 1.3,
        });

        // Conv-ReLU fusion
        patterns.push(FusionPattern {
            name: "conv_relu".to_string(),
            operations: vec![
                VjpOperation::Conv2d,
                VjpOperation::Activation("relu".to_string()),
            ],
            fused_vjp: "conv_relu_fused".to_string(),
            memory_saving: 0.4,
            speedup: 1.8,
        });
    }

    /// Optimize VJP computation for a given context
    pub fn optimize_vjp(&self, mut context: VjpContext<T>) -> Result<VjpContext<T>> {
        let start_time = Instant::now();

        match self.config.strategy {
            VjpStrategy::Standard => self.optimize_standard_vjp(&mut context)?,
            VjpStrategy::Checkpointed => self.optimize_checkpointed_vjp(&mut context)?,
            VjpStrategy::Fused => self.optimize_fused_vjp(&mut context)?,
            VjpStrategy::Vectorized => self.optimize_vectorized_vjp(&mut context)?,
            VjpStrategy::Adaptive => self.optimize_adaptive_vjp(&mut context)?,
        }

        // Update statistics
        {
            let mut stats = self.stats.write().expect("lock should not be poisoned");
            stats.total_vjp_computations += 1;
            let computation_time = start_time.elapsed().as_millis() as f64;
            stats.total_computation_time_ms += computation_time;
            stats.average_computation_time_ms =
                stats.total_computation_time_ms / stats.total_vjp_computations as f64;
        }

        Ok(context)
    }

    /// Standard VJP optimization
    fn optimize_standard_vjp(&self, context: &mut VjpContext<T>) -> Result<()> {
        // Basic reverse topological sort for execution order
        context.execution_order = self.compute_reverse_topological_order(&context.nodes)?;

        // Estimate memory usage
        for node in &mut context.nodes {
            node.memory_usage = self.estimate_node_memory_usage(node);
        }

        Ok(())
    }

    /// Checkpointed VJP optimization
    fn optimize_checkpointed_vjp(&self, context: &mut VjpContext<T>) -> Result<()> {
        // First do standard optimization
        self.optimize_standard_vjp(context)?;

        // Create checkpoints based on memory budget
        self.create_checkpoints(context)?;

        // Update stats
        {
            let mut stats = self.stats.write().expect("lock should not be poisoned");
            stats.checkpoints_created = context.checkpoints.len();
        }

        Ok(())
    }

    /// Fused VJP optimization
    fn optimize_fused_vjp(&self, context: &mut VjpContext<T>) -> Result<()> {
        // Identify fusion opportunities
        let fusion_opportunities = self.identify_fusion_opportunities(&context.nodes)?;

        // Apply fusions
        let fused_count = self.apply_fusions(context, fusion_opportunities)?;

        // Update execution order after fusion
        context.execution_order = self.compute_reverse_topological_order(&context.nodes)?;

        // Update stats
        {
            let mut stats = self.stats.write().expect("lock should not be poisoned");
            stats.operations_fused += fused_count;
        }

        Ok(())
    }

    /// Vectorized VJP optimization
    fn optimize_vectorized_vjp(&self, context: &mut VjpContext<T>) -> Result<()> {
        // Identify vectorizable operations
        self.identify_vectorizable_operations(context)?;

        // Apply vectorization
        self.apply_vectorization(context)?;

        // Standard optimization after vectorization
        self.optimize_standard_vjp(context)?;

        Ok(())
    }

    /// Adaptive VJP optimization
    fn optimize_adaptive_vjp(&self, context: &mut VjpContext<T>) -> Result<()> {
        let node_count = context.nodes.len();
        let estimated_memory = self.estimate_total_memory_usage(&context.nodes);

        // Choose strategy based on problem characteristics
        if estimated_memory > self.config.memory_budget {
            // High memory usage - use checkpointing
            self.optimize_checkpointed_vjp(context)?;
        } else if node_count > 100 && self.config.enable_fusion {
            // Large graph - use fusion for efficiency
            self.optimize_fused_vjp(context)?;
        } else if self.config.enable_vectorization {
            // Small to medium graph - use vectorization
            self.optimize_vectorized_vjp(context)?;
        } else {
            // Simple case - use standard
            self.optimize_standard_vjp(context)?;
        }

        Ok(())
    }

    /// Compute reverse topological order for backward pass
    fn compute_reverse_topological_order(&self, nodes: &[VjpNode<T>]) -> Result<Vec<usize>> {
        let mut in_degree = vec![0; nodes.len()];
        let mut adj_list: Vec<Vec<usize>> = vec![vec![]; nodes.len()];

        // Build adjacency list and compute in-degrees
        for (i, node) in nodes.iter().enumerate() {
            for &input_id in &node.inputs {
                if input_id < nodes.len() {
                    adj_list[input_id].push(i);
                    in_degree[i] += 1;
                }
            }
        }

        // Kahn's algorithm for topological sort
        let mut queue = VecDeque::new();
        for (i, &degree) in in_degree.iter().enumerate() {
            if degree == 0 {
                queue.push_back(i);
            }
        }

        let mut topo_order = Vec::new();
        while let Some(node_id) = queue.pop_front() {
            topo_order.push(node_id);

            for &neighbor in &adj_list[node_id] {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push_back(neighbor);
                }
            }
        }

        if topo_order.len() != nodes.len() {
            return Err(TorshError::AutogradError(
                "Cycle detected in computation graph".to_string(),
            ));
        }

        // Reverse for backward pass
        topo_order.reverse();
        Ok(topo_order)
    }

    /// Estimate memory usage for a node
    fn estimate_node_memory_usage(&self, node: &VjpNode<T>) -> usize {
        let element_size = std::mem::size_of::<T>();
        let output_elements: usize = node.output_shape.iter().product();

        // Base memory for output
        let base_memory = output_elements * element_size;

        // Additional memory based on operation type
        let additional_memory = match &node.operation {
            VjpOperation::MatMul => base_memory * 2, // Need to store both inputs
            VjpOperation::Conv2d => base_memory * 3, // Weights, inputs, and intermediate
            VjpOperation::Pool2d => base_memory / 2, // Pooling reduces size
            _ => base_memory,
        };

        base_memory + additional_memory
    }

    /// Estimate total memory usage for all nodes
    fn estimate_total_memory_usage(&self, nodes: &[VjpNode<T>]) -> usize {
        nodes
            .iter()
            .map(|node| self.estimate_node_memory_usage(node))
            .sum()
    }

    /// Create checkpoints for memory-efficient recomputation
    fn create_checkpoints(&self, context: &mut VjpContext<T>) -> Result<()> {
        let checkpoint_interval = self.config.checkpoint_frequency;
        let mut checkpoint_id = 0;

        for i in (0..context.nodes.len()).step_by(checkpoint_interval) {
            if checkpoint_id >= self.config.max_checkpoints {
                break;
            }

            let end_idx = std::cmp::min(i + checkpoint_interval, context.nodes.len());
            let memory_usage = context.nodes[i..end_idx]
                .iter()
                .map(|n| n.memory_usage)
                .sum();

            let checkpoint = VjpCheckpoint {
                id: checkpoint_id,
                tensors: Vec::new(), // Would be populated during forward pass
                node_range: (i, end_idx),
                memory_usage,
            };

            context.checkpoints.insert(checkpoint_id, checkpoint);

            // Mark nodes in this range for checkpointing
            for node in &mut context.nodes[i..end_idx] {
                node.should_checkpoint = true;
            }

            checkpoint_id += 1;
        }

        Ok(())
    }

    /// Identify opportunities for operation fusion
    fn identify_fusion_opportunities(&self, nodes: &[VjpNode<T>]) -> Result<Vec<(usize, usize)>> {
        let mut opportunities = Vec::new();
        let patterns = self
            .fusion_patterns
            .read()
            .expect("lock should not be poisoned");

        for i in 0..nodes.len().saturating_sub(1) {
            let current_op = &nodes[i].operation;
            let next_op = &nodes[i + 1].operation;

            // Check if consecutive operations can be fused
            for pattern in patterns.iter() {
                if pattern.operations.len() == 2
                    && pattern.operations[0] == *current_op
                    && pattern.operations[1] == *next_op
                {
                    opportunities.push((i, i + 1));
                    break;
                }
            }
        }

        Ok(opportunities)
    }

    /// Apply fusion optimizations
    fn apply_fusions(
        &self,
        _context: &mut VjpContext<T>,
        opportunities: Vec<(usize, usize)>,
    ) -> Result<usize> {
        // In a full implementation, this would:
        // 1. Merge the operations
        // 2. Create fused VJP functions
        // 3. Update the computation graph

        // For now, just return the count of fusion opportunities
        Ok(opportunities.len())
    }

    /// Identify vectorizable operations
    fn identify_vectorizable_operations(&self, _context: &mut VjpContext<T>) -> Result<()> {
        // Identify operations that can be vectorized (element-wise ops, reductions, etc.)
        // This would analyze the graph for vectorization opportunities
        Ok(())
    }

    /// Apply vectorization optimizations
    fn apply_vectorization(&self, _context: &mut VjpContext<T>) -> Result<()> {
        // Apply SIMD vectorization to suitable operations
        // This would integrate with the SIMD operations module
        Ok(())
    }

    /// Compute VJP for a given gradient output
    pub fn compute_vjp(
        &self,
        context: &VjpContext<T>,
        grad_output: &[T],
        node_id: usize,
    ) -> Result<Vec<Vec<T>>> {
        if node_id >= context.nodes.len() {
            return Err(TorshError::AutogradError("Invalid node ID".to_string()));
        }

        let node = &context.nodes[node_id];

        // Get VJP function for this operation
        let vjp_fn = if let Some(ref vjp_fn) = node.vjp_fn {
            vjp_fn.clone()
        } else {
            // Try to get from registry
            let op_name = self.operation_to_string(&node.operation);
            let registry = self
                .vjp_registry
                .read()
                .expect("lock should not be poisoned");
            let vjp_fn = registry.get(&op_name).ok_or_else(|| {
                TorshError::AutogradError(format!("No VJP function for operation: {op_name}"))
            })?;
            vjp_fn.clone()
        };

        // Compute VJP
        let result = vjp_fn(grad_output, &node.saved_tensors)?;

        // Update memory tracking
        {
            let mut memory_tracker = self.memory_tracker.lock();
            *memory_tracker += std::mem::size_of_val(grad_output);
        }

        Ok(result)
    }

    /// Convert operation to string for registry lookup
    fn operation_to_string(&self, op: &VjpOperation) -> String {
        match op {
            VjpOperation::Add => "add".to_string(),
            VjpOperation::Mul => "mul".to_string(),
            VjpOperation::MatMul => "matmul".to_string(),
            VjpOperation::ElementWise(name) => name.clone(),
            VjpOperation::Reduction(name) => name.clone(),
            VjpOperation::Conv2d => "conv2d".to_string(),
            VjpOperation::Pool2d => "pool2d".to_string(),
            VjpOperation::Activation(name) => name.clone(),
            VjpOperation::Custom(name) => name.clone(),
        }
    }

    /// Register a custom VJP function
    pub fn register_vjp_function(&self, name: String, vjp_fn: VjpFunction<T>) {
        let mut registry = self
            .vjp_registry
            .write()
            .expect("lock should not be poisoned");
        registry.insert(name, vjp_fn);
    }

    /// Get VJP statistics
    pub fn get_stats(&self) -> VjpStats {
        self.stats
            .read()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Get current memory usage
    pub fn get_memory_usage(&self) -> usize {
        *self.memory_tracker.lock()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write().expect("lock should not be poisoned");
        *stats = VjpStats::default();

        let mut memory_tracker = self.memory_tracker.lock();
        *memory_tracker = 0;
    }
}

/// Helper functions for VJP operations
pub mod vjp_ops {
    use super::*;

    /// Create a VJP node for addition
    pub fn create_add_node<T: FloatElement>(
        id: usize,
        inputs: Vec<usize>,
        output_shape: Vec<usize>,
    ) -> VjpNode<T> {
        VjpNode {
            id,
            operation: VjpOperation::Add,
            inputs,
            output_shape,
            saved_tensors: Vec::new(),
            vjp_fn: None,
            memory_usage: 0,
            compute_cost: 1.0,
            should_checkpoint: false,
        }
    }

    /// Create a VJP node for multiplication
    pub fn create_mul_node<T: FloatElement>(
        id: usize,
        inputs: Vec<usize>,
        output_shape: Vec<usize>,
        saved_tensors: Vec<Arc<RwLock<Vec<T>>>>,
    ) -> VjpNode<T> {
        VjpNode {
            id,
            operation: VjpOperation::Mul,
            inputs,
            output_shape,
            saved_tensors,
            vjp_fn: None,
            memory_usage: 0,
            compute_cost: 2.0,
            should_checkpoint: false,
        }
    }

    /// Create a VJP node for ReLU activation
    pub fn create_relu_node<T: FloatElement>(
        id: usize,
        inputs: Vec<usize>,
        output_shape: Vec<usize>,
        input_tensor: Arc<RwLock<Vec<T>>>,
    ) -> VjpNode<T> {
        VjpNode {
            id,
            operation: VjpOperation::Activation("relu".to_string()),
            inputs,
            output_shape,
            saved_tensors: vec![input_tensor],
            vjp_fn: None,
            memory_usage: 0,
            compute_cost: 1.5,
            should_checkpoint: false,
        }
    }
}

/// Utilities for VJP optimization
pub mod utils {
    use super::*;

    /// Analyze VJP computation efficiency
    pub fn analyze_vjp_efficiency(stats: &VjpStats) -> f64 {
        if stats.total_vjp_computations == 0 {
            return 0.0;
        }

        // Efficiency based on various factors
        let time_efficiency = 1000.0 / stats.average_computation_time_ms.max(1.0);
        let fusion_efficiency = 1.0 + (stats.operations_fused as f64 * 0.1);
        let parallel_efficiency = stats.parallel_speedup;

        time_efficiency * fusion_efficiency * parallel_efficiency
    }

    /// Get VJP optimization recommendations
    pub fn get_vjp_recommendations(
        stats: &VjpStats,
        config: &VjpOptimizationConfig,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if stats.average_computation_time_ms > 100.0 {
            recommendations.push("Consider increasing optimization level".to_string());
        }

        if stats.operations_fused == 0 && config.enable_fusion {
            recommendations.push("Enable operation fusion for better performance".to_string());
        }

        if stats.peak_memory_usage > config.memory_budget {
            recommendations
                .push("Increase checkpointing frequency to reduce memory usage".to_string());
        }

        if stats.parallel_speedup < 1.2 && config.enable_parallel_vjp {
            recommendations.push("Check parallel VJP configuration".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("VJP optimization is performing well".to_string());
        }

        recommendations
    }

    /// Format VJP statistics
    pub fn format_vjp_stats(stats: &VjpStats) -> String {
        format!(
            "VJP Stats: {} computations, {:.2}ms avg time, {} ops fused, {:.1}x speedup",
            stats.total_vjp_computations,
            stats.average_computation_time_ms,
            stats.operations_fused,
            stats.parallel_speedup
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_vjp_optimizer_creation() {
        let config = VjpOptimizationConfig::default();
        let optimizer = VjpOptimizer::<f32>::new(config);

        let stats = optimizer.get_stats();
        assert_eq!(stats.total_vjp_computations, 0);
        assert_eq!(optimizer.get_memory_usage(), 0);
    }

    #[test]
    fn test_vjp_context_optimization() {
        let config = VjpOptimizationConfig::default();
        let optimizer = VjpOptimizer::<f32>::new(config);

        // Create simple computation graph: a + b * c
        let nodes = vec![
            vjp_ops::create_mul_node(
                0,
                vec![],
                vec![10],
                vec![
                    Arc::new(RwLock::new(vec![1.0; 10])),
                    Arc::new(RwLock::new(vec![2.0; 10])),
                ],
            ),
            vjp_ops::create_add_node(1, vec![0], vec![10]),
        ];

        let context = VjpContext {
            nodes,
            execution_order: Vec::new(),
            checkpoints: HashMap::new(),
            memory_budget: 1024 * 1024,
            optimization_level: OptimizationLevel::Basic,
        };

        let optimized_context = optimizer.optimize_vjp(context).unwrap();
        assert!(!optimized_context.execution_order.is_empty());
    }

    #[test]
    fn test_vjp_computation() {
        let config = VjpOptimizationConfig::default();
        let optimizer = VjpOptimizer::<f32>::new(config);

        // Test addition VJP
        let grad_output = vec![1.0, 2.0, 3.0];
        let context = VjpContext {
            nodes: vec![vjp_ops::create_add_node(0, vec![], vec![3])],
            execution_order: vec![0],
            checkpoints: HashMap::new(),
            memory_budget: 1024,
            optimization_level: OptimizationLevel::Basic,
        };

        let result = optimizer.compute_vjp(&context, &grad_output, 0).unwrap();
        assert_eq!(result.len(), 2); // Two gradients for addition
        assert_eq!(result[0], grad_output); // grad_a = grad_output
        assert_eq!(result[1], grad_output); // grad_b = grad_output
    }

    #[test]
    fn test_multiplication_vjp() {
        let config = VjpOptimizationConfig::default();
        let optimizer = VjpOptimizer::<f32>::new(config);

        let grad_output = vec![1.0, 1.0];
        let a_vals = vec![2.0, 3.0];
        let b_vals = vec![4.0, 5.0];

        let context = VjpContext {
            nodes: vec![vjp_ops::create_mul_node(
                0,
                vec![],
                vec![2],
                vec![
                    Arc::new(RwLock::new(a_vals.clone())),
                    Arc::new(RwLock::new(b_vals.clone())),
                ],
            )],
            execution_order: vec![0],
            checkpoints: HashMap::new(),
            memory_budget: 1024,
            optimization_level: OptimizationLevel::Basic,
        };

        let result = optimizer.compute_vjp(&context, &grad_output, 0).unwrap();
        assert_eq!(result.len(), 2);

        // grad_a = grad_output * b
        assert_eq!(result[0], vec![4.0, 5.0]);
        // grad_b = grad_output * a
        assert_eq!(result[1], vec![2.0, 3.0]);
    }

    #[test]
    fn test_relu_vjp() {
        let config = VjpOptimizationConfig::default();
        let optimizer = VjpOptimizer::<f32>::new(config);

        let grad_output = vec![1.0, 1.0, 1.0];
        let input_vals = vec![-1.0, 0.0, 2.0]; // negative, zero, positive

        let context = VjpContext {
            nodes: vec![vjp_ops::create_relu_node(
                0,
                vec![],
                vec![3],
                Arc::new(RwLock::new(input_vals)),
            )],
            execution_order: vec![0],
            checkpoints: HashMap::new(),
            memory_budget: 1024,
            optimization_level: OptimizationLevel::Basic,
        };

        let result = optimizer.compute_vjp(&context, &grad_output, 0).unwrap();
        assert_eq!(result.len(), 1);

        // ReLU gradient: 0 for negative, 0 for zero, 1 for positive
        assert_eq!(result[0], vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_fusion_pattern_identification() {
        let config = VjpOptimizationConfig {
            enable_fusion: true,
            ..Default::default()
        };
        let optimizer = VjpOptimizer::<f32>::new(config);

        let nodes = vec![
            vjp_ops::create_add_node(0, vec![], vec![10]),
            vjp_ops::create_relu_node(1, vec![0], vec![10], Arc::new(RwLock::new(vec![1.0; 10]))),
        ];

        let opportunities = optimizer.identify_fusion_opportunities(&nodes).unwrap();
        assert!(!opportunities.is_empty()); // Should find add-relu fusion
    }

    #[test]
    fn test_checkpointing() {
        let config = VjpOptimizationConfig {
            enable_checkpointing: true,
            checkpoint_frequency: 2,
            max_checkpoints: 5,
            ..Default::default()
        };
        let optimizer = VjpOptimizer::<f32>::new(config);

        let nodes = vec![
            vjp_ops::create_add_node(0, vec![], vec![10]),
            vjp_ops::create_mul_node(
                1,
                vec![0],
                vec![10],
                vec![
                    Arc::new(RwLock::new(vec![1.0; 10])),
                    Arc::new(RwLock::new(vec![2.0; 10])),
                ],
            ),
            vjp_ops::create_relu_node(2, vec![1], vec![10], Arc::new(RwLock::new(vec![1.0; 10]))),
            vjp_ops::create_add_node(3, vec![2], vec![10]),
        ];

        let mut context = VjpContext {
            nodes,
            execution_order: Vec::new(),
            checkpoints: HashMap::new(),
            memory_budget: 512, // Small budget to force checkpointing
            optimization_level: OptimizationLevel::Aggressive,
        };

        optimizer.create_checkpoints(&mut context).unwrap();
        assert!(!context.checkpoints.is_empty());
    }

    #[test]
    fn test_memory_estimation() {
        let config = VjpOptimizationConfig::default();
        let optimizer = VjpOptimizer::<f32>::new(config);

        let node = vjp_ops::create_add_node(0, vec![], vec![100]);
        let memory_usage = optimizer.estimate_node_memory_usage(&node);

        // Should be at least the size of 100 f32s
        assert!(memory_usage >= 100 * std::mem::size_of::<f32>());
    }

    #[test]
    fn test_topological_sorting() {
        let config = VjpOptimizationConfig::default();
        let optimizer = VjpOptimizer::<f32>::new(config);

        // Create DAG: 0 -> 1 -> 2
        let nodes = vec![
            vjp_ops::create_add_node(0, vec![], vec![10]),
            vjp_ops::create_mul_node(
                1,
                vec![0],
                vec![10],
                vec![
                    Arc::new(RwLock::new(vec![1.0; 10])),
                    Arc::new(RwLock::new(vec![2.0; 10])),
                ],
            ),
            vjp_ops::create_relu_node(2, vec![1], vec![10], Arc::new(RwLock::new(vec![1.0; 10]))),
        ];

        let order = optimizer.compute_reverse_topological_order(&nodes).unwrap();
        assert_eq!(order.len(), 3);
        // For backward pass, should be reverse: 2, 1, 0
        assert_eq!(order, vec![2, 1, 0]);
    }

    #[test]
    fn test_utility_functions() {
        let stats = VjpStats {
            total_vjp_computations: 100,
            average_computation_time_ms: 50.0,
            operations_fused: 10,
            parallel_speedup: 1.5,
            ..Default::default()
        };

        let efficiency = utils::analyze_vjp_efficiency(&stats);
        assert!(efficiency > 0.0);

        let config = VjpOptimizationConfig::default();
        let recommendations = utils::get_vjp_recommendations(&stats, &config);
        assert!(!recommendations.is_empty());

        let formatted = utils::format_vjp_stats(&stats);
        assert!(formatted.contains("100 computations"));
        assert!(formatted.contains("50.00ms"));
    }
}
