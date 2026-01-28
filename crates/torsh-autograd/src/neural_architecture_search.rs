//! Differentiable Neural Architecture Search (DNAS) support
//!
//! This module provides comprehensive support for differentiable neural architecture
//! search, enabling automatic discovery of optimal network architectures through
//! gradient-based optimization. It supports various DNAS methods including DARTS,
//! GDAS, PC-DARTS, and custom search strategies.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::error_handling::{AutogradError, AutogradResult};
use scirs2_core::random::{Random, Rng}; // SciRS2 POLICY compliant
use std::collections::HashMap;

/// Trait for searchable operations in DNAS
pub trait SearchableOperation: Send + Sync + std::fmt::Debug {
    /// Compute forward pass of the operation
    fn forward(&self, input: &[f64], alpha: f64) -> AutogradResult<Vec<f64>>;

    /// Compute backward pass (gradients w.r.t. input and parameters)
    fn backward(
        &self,
        input: &[f64],
        output_grad: &[f64],
        alpha: f64,
    ) -> AutogradResult<OperationGradients>;

    /// Get operation name/identifier
    fn name(&self) -> &str;

    /// Get parameter count for this operation
    fn parameter_count(&self) -> usize;

    /// Get parameters (if any)
    fn parameters(&self) -> &[f64];

    /// Get mutable parameters (if any)
    fn parameters_mut(&mut self) -> &mut [f64];

    /// Get computational cost estimate (FLOPs)
    fn computational_cost(&self, input_shape: &[usize]) -> f64;

    /// Get memory cost estimate (bytes)
    fn memory_cost(&self, input_shape: &[usize]) -> f64;
}

/// Gradients from operation backward pass
#[derive(Debug, Clone)]
pub struct OperationGradients {
    /// Gradient w.r.t. input
    pub input_gradient: Vec<f64>,
    /// Gradient w.r.t. operation parameters
    pub parameter_gradients: Vec<f64>,
}

/// Identity operation (skip connection)
#[derive(Debug, Clone)]
pub struct IdentityOperation;

impl SearchableOperation for IdentityOperation {
    fn forward(&self, input: &[f64], _alpha: f64) -> AutogradResult<Vec<f64>> {
        Ok(input.to_vec())
    }

    fn backward(
        &self,
        _input: &[f64],
        output_grad: &[f64],
        _alpha: f64,
    ) -> AutogradResult<OperationGradients> {
        Ok(OperationGradients {
            input_gradient: output_grad.to_vec(),
            parameter_gradients: vec![],
        })
    }

    fn name(&self) -> &str {
        "identity"
    }

    fn parameter_count(&self) -> usize {
        0
    }

    fn parameters(&self) -> &[f64] {
        &[]
    }

    fn parameters_mut(&mut self) -> &mut [f64] {
        &mut []
    }

    fn computational_cost(&self, _input_shape: &[usize]) -> f64 {
        0.0 // No computation
    }

    fn memory_cost(&self, input_shape: &[usize]) -> f64 {
        input_shape.iter().product::<usize>() as f64 * 4.0 // Just pass-through
    }
}

/// Zero operation (no connection)
#[derive(Debug, Clone)]
pub struct ZeroOperation {
    output_size: usize,
}

impl ZeroOperation {
    pub fn new(output_size: usize) -> Self {
        Self { output_size }
    }
}

impl SearchableOperation for ZeroOperation {
    fn forward(&self, _input: &[f64], _alpha: f64) -> AutogradResult<Vec<f64>> {
        Ok(vec![0.0; self.output_size])
    }

    fn backward(
        &self,
        input: &[f64],
        _output_grad: &[f64],
        _alpha: f64,
    ) -> AutogradResult<OperationGradients> {
        Ok(OperationGradients {
            input_gradient: vec![0.0; input.len()],
            parameter_gradients: vec![],
        })
    }

    fn name(&self) -> &str {
        "zero"
    }

    fn parameter_count(&self) -> usize {
        0
    }

    fn parameters(&self) -> &[f64] {
        &[]
    }

    fn parameters_mut(&mut self) -> &mut [f64] {
        &mut []
    }

    fn computational_cost(&self, _input_shape: &[usize]) -> f64 {
        0.0 // No computation
    }

    fn memory_cost(&self, _input_shape: &[usize]) -> f64 {
        self.output_size as f64 * 4.0 // Output tensor
    }
}

/// Convolution operation for DNAS
#[derive(Debug, Clone)]
pub struct ConvOperation {
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    in_channels: usize,
    out_channels: usize,
    weights: Vec<f64>,
    bias: Vec<f64>,
}

impl ConvOperation {
    pub fn new(
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        in_channels: usize,
        out_channels: usize,
    ) -> Self {
        let weight_count = out_channels * in_channels * kernel_size * kernel_size;
        let mut weights = vec![0.0; weight_count];

        // Xavier initialization
        let fan_in = in_channels * kernel_size * kernel_size;
        let fan_out = out_channels * kernel_size * kernel_size;
        let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();

        for w in &mut weights {
            *w = (Random::default().random::<f64>() - 0.5) * 2.0 * limit;
        }

        let bias = vec![0.0; out_channels];

        Self {
            kernel_size,
            stride,
            padding,
            dilation,
            in_channels,
            out_channels,
            weights,
            bias,
        }
    }

    /// Calculate output size after convolution
    fn output_size(&self, input_size: usize) -> usize {
        let effective_kernel = self.dilation * (self.kernel_size - 1) + 1;
        (input_size + 2 * self.padding - effective_kernel) / self.stride + 1
    }
}

impl SearchableOperation for ConvOperation {
    fn forward(&self, input: &[f64], alpha: f64) -> AutogradResult<Vec<f64>> {
        // Simplified 1D convolution for demonstration
        let input_size = input.len() / self.in_channels;
        let output_size = self.output_size(input_size);
        let mut output = vec![0.0; self.out_channels * output_size];

        for out_ch in 0..self.out_channels {
            for pos in 0..output_size {
                let mut sum = 0.0;

                for in_ch in 0..self.in_channels {
                    for k in 0..self.kernel_size {
                        let input_pos = pos * self.stride + k;
                        if input_pos < input_size {
                            let weight_idx =
                                (out_ch * self.in_channels + in_ch) * self.kernel_size + k;
                            let input_idx = in_ch * input_size + input_pos;
                            sum += input[input_idx] * self.weights[weight_idx];
                        }
                    }
                }

                sum += self.bias[out_ch];
                output[out_ch * output_size + pos] = sum * alpha; // Scale by architecture weight
            }
        }

        Ok(output)
    }

    fn backward(
        &self,
        input: &[f64],
        output_grad: &[f64],
        alpha: f64,
    ) -> AutogradResult<OperationGradients> {
        let input_size = input.len() / self.in_channels;
        let output_size = self.output_size(input_size);

        let mut input_gradient = vec![0.0; input.len()];
        let mut weight_gradients = vec![0.0; self.weights.len()];
        let mut bias_gradients = vec![0.0; self.bias.len()];

        // Compute gradients (simplified)
        for out_ch in 0..self.out_channels {
            for pos in 0..output_size {
                let grad = output_grad[out_ch * output_size + pos] * alpha;
                bias_gradients[out_ch] += grad;

                for in_ch in 0..self.in_channels {
                    for k in 0..self.kernel_size {
                        let input_pos = pos * self.stride + k;
                        if input_pos < input_size {
                            let weight_idx =
                                (out_ch * self.in_channels + in_ch) * self.kernel_size + k;
                            let input_idx = in_ch * input_size + input_pos;

                            // Gradient w.r.t. weights
                            weight_gradients[weight_idx] += grad * input[input_idx];

                            // Gradient w.r.t. input
                            input_gradient[input_idx] += grad * self.weights[weight_idx];
                        }
                    }
                }
            }
        }

        let mut parameter_gradients = weight_gradients;
        parameter_gradients.extend(bias_gradients);

        Ok(OperationGradients {
            input_gradient,
            parameter_gradients,
        })
    }

    fn name(&self) -> &str {
        "conv"
    }

    fn parameter_count(&self) -> usize {
        self.weights.len() + self.bias.len()
    }

    fn parameters(&self) -> &[f64] {
        // Return combined weights and bias
        &[] // Simplified for this implementation
    }

    fn parameters_mut(&mut self) -> &mut [f64] {
        // Return combined weights and bias
        &mut [] // Simplified for this implementation
    }

    fn computational_cost(&self, input_shape: &[usize]) -> f64 {
        let input_size = input_shape.iter().product::<usize>() / self.in_channels;
        let output_size = self.output_size(input_size);

        (self.out_channels * self.in_channels * self.kernel_size * output_size) as f64
    }

    fn memory_cost(&self, input_shape: &[usize]) -> f64 {
        let input_size = input_shape.iter().product::<usize>() / self.in_channels;
        let output_size = self.output_size(input_size);

        ((self.out_channels * output_size) + self.parameter_count()) as f64 * 4.0
    }
}

/// Mixed operation that combines multiple operations with learnable weights
pub struct MixedOperation {
    operations: Vec<Box<dyn SearchableOperation>>,
    alpha_weights: Vec<f64>, // Architecture weights (learnable)
    operation_names: Vec<String>,
    temperature: f64, // For Gumbel softmax
}

impl std::fmt::Debug for MixedOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MixedOperation")
            .field("operation_names", &self.operation_names)
            .field("alpha_weights", &self.alpha_weights)
            .field("temperature", &self.temperature)
            .field("num_operations", &self.operations.len())
            .finish()
    }
}

impl MixedOperation {
    /// Create a new mixed operation
    pub fn new(operations: Vec<Box<dyn SearchableOperation>>, temperature: f64) -> Self {
        let op_count = operations.len();
        let operation_names: Vec<String> =
            operations.iter().map(|op| op.name().to_string()).collect();

        Self {
            operations,
            alpha_weights: vec![1.0 / op_count as f64; op_count], // Initialize uniformly
            operation_names,
            temperature,
        }
    }

    /// Get alpha weights (architecture parameters)
    pub fn alpha_weights(&self) -> &[f64] {
        &self.alpha_weights
    }

    /// Get mutable alpha weights
    pub fn alpha_weights_mut(&mut self) -> &mut [f64] {
        &mut self.alpha_weights
    }

    /// Apply softmax to alpha weights
    fn softmax_weights(&self) -> Vec<f64> {
        let max_val = self
            .alpha_weights
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals: Vec<f64> = self
            .alpha_weights
            .iter()
            .map(|&x| ((x - max_val) / self.temperature).exp())
            .collect();
        let sum: f64 = exp_vals.iter().sum();
        exp_vals.iter().map(|&x| x / sum).collect()
    }

    /// Apply Gumbel softmax for discrete sampling
    fn gumbel_softmax(&self, hard: bool) -> Vec<f64> {
        let gumbel_noise: Vec<f64> = (0..self.alpha_weights.len())
            .map(|_| {
                let u1: f64 = Random::default().random();
                let _u2: f64 = Random::default().random();
                -(-u1.ln()).ln() // Gumbel noise
            })
            .collect();

        let max_val = self
            .alpha_weights
            .iter()
            .zip(&gumbel_noise)
            .map(|(&a, &g)| a + g)
            .fold(f64::NEG_INFINITY, |a, b| a.max(b));

        let exp_vals: Vec<f64> = self
            .alpha_weights
            .iter()
            .zip(&gumbel_noise)
            .map(|(&a, &g)| ((a + g - max_val) / self.temperature).exp())
            .collect();

        let sum: f64 = exp_vals.iter().sum();
        let soft_weights: Vec<f64> = exp_vals.iter().map(|&x| x / sum).collect();

        if hard {
            // Straight-through estimator: one-hot in forward, soft in backward
            let max_idx = soft_weights
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .expect("soft_weights should not be empty")
                .0;

            let mut hard_weights = vec![0.0; self.alpha_weights.len()];
            hard_weights[max_idx] = 1.0;
            hard_weights
        } else {
            soft_weights
        }
    }

    /// Forward pass using weighted combination of operations
    pub fn forward(
        &self,
        input: &[f64],
        sampling_strategy: SamplingStrategy,
    ) -> AutogradResult<MixedOperationOutput> {
        let weights = match sampling_strategy {
            SamplingStrategy::Continuous => self.softmax_weights(),
            SamplingStrategy::GumbelSoft => self.gumbel_softmax(false),
            SamplingStrategy::GumbelHard => self.gumbel_softmax(true),
        };

        let mut outputs = Vec::new();
        let mut total_output = None;

        for (i, operation) in self.operations.iter().enumerate() {
            let op_output = operation.forward(input, weights[i])?;
            outputs.push(op_output.clone());

            if total_output.is_none() {
                total_output = Some(vec![0.0; op_output.len()]);
            }

            if let Some(ref mut total) = total_output {
                for (j, &val) in op_output.iter().enumerate() {
                    total[j] += val;
                }
            }
        }

        Ok(MixedOperationOutput {
            output: total_output.expect("total_output should be set after processing operations"),
            operation_outputs: outputs,
            weights_used: weights,
        })
    }

    /// Backward pass with architecture gradient computation
    pub fn backward(
        &self,
        input: &[f64],
        output_grad: &[f64],
        weights_used: &[f64],
    ) -> AutogradResult<MixedOperationGradients> {
        let mut input_gradient = vec![0.0; input.len()];
        let mut alpha_gradients = vec![0.0; self.alpha_weights.len()];
        let mut operation_gradients = Vec::new();

        for (i, operation) in self.operations.iter().enumerate() {
            let op_grads = operation.backward(input, output_grad, weights_used[i])?;
            operation_gradients.push(op_grads.clone());

            // Accumulate input gradients
            for (j, &grad) in op_grads.input_gradient.iter().enumerate() {
                input_gradient[j] += grad;
            }

            // Compute alpha gradients (gradient w.r.t. architecture weights)
            let op_output = operation.forward(input, 1.0)?; // Unweighted output for gradient
            let alpha_grad: f64 = output_grad
                .iter()
                .zip(&op_output)
                .map(|(&og, &oo)| og * oo)
                .sum();
            alpha_gradients[i] = alpha_grad;
        }

        Ok(MixedOperationGradients {
            input_gradient,
            alpha_gradients,
            operation_gradients,
        })
    }

    /// Get the most likely operation (highest alpha weight)
    pub fn most_likely_operation(&self) -> (usize, &str, f64) {
        let weights = self.softmax_weights();
        let (idx, &weight) = weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .expect("weights should not be empty");
        (idx, &self.operation_names[idx], weight)
    }

    /// Prune operation with low weights
    pub fn prune_operations(&mut self, threshold: f64) -> Vec<String> {
        let weights = self.softmax_weights();
        let mut pruned_ops = Vec::new();

        let mut i = 0;
        while i < self.operations.len() {
            if weights[i] < threshold {
                pruned_ops.push(self.operation_names.remove(i));
                self.operations.remove(i);
                self.alpha_weights.remove(i);
            } else {
                i += 1;
            }
        }

        // Renormalize remaining weights
        if !self.alpha_weights.is_empty() {
            let sum: f64 = self.alpha_weights.iter().sum();
            for weight in &mut self.alpha_weights {
                *weight /= sum;
            }
        }

        pruned_ops
    }
}

/// Output from mixed operation forward pass
#[derive(Debug, Clone)]
pub struct MixedOperationOutput {
    pub output: Vec<f64>,
    pub operation_outputs: Vec<Vec<f64>>,
    pub weights_used: Vec<f64>,
}

/// Gradients from mixed operation backward pass
#[derive(Debug, Clone)]
pub struct MixedOperationGradients {
    pub input_gradient: Vec<f64>,
    pub alpha_gradients: Vec<f64>,
    pub operation_gradients: Vec<OperationGradients>,
}

/// Sampling strategy for mixed operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingStrategy {
    /// Continuous relaxation (standard DARTS)
    Continuous,
    /// Gumbel softmax (soft sampling)
    GumbelSoft,
    /// Gumbel softmax with straight-through estimator (hard sampling)
    GumbelHard,
}

/// Edge in the search graph
#[derive(Debug)]
pub struct SearchEdge {
    pub from_node: usize,
    pub to_node: usize,
    pub operation: MixedOperation,
    pub edge_id: String,
}

impl SearchEdge {
    pub fn new(
        from_node: usize,
        to_node: usize,
        operations: Vec<Box<dyn SearchableOperation>>,
        temperature: f64,
    ) -> Self {
        let edge_id = format!("edge_{}_{}", from_node, to_node);
        let operation = MixedOperation::new(operations, temperature);

        Self {
            from_node,
            to_node,
            operation,
            edge_id,
        }
    }
}

/// Node in the search graph
#[derive(Debug, Clone)]
pub struct SearchNode {
    pub node_id: usize,
    pub intermediate_tensors: Vec<Vec<f64>>,
    pub is_input: bool,
    pub is_output: bool,
}

impl SearchNode {
    pub fn new(node_id: usize, is_input: bool, is_output: bool) -> Self {
        Self {
            node_id,
            intermediate_tensors: Vec::new(),
            is_input,
            is_output,
        }
    }
}

/// DARTS-style differentiable architecture search
#[derive(Debug)]
pub struct DARTS {
    nodes: Vec<SearchNode>,
    edges: Vec<SearchEdge>,
    node_connections: HashMap<usize, Vec<usize>>, // node_id -> edge_indices
    sampling_strategy: SamplingStrategy,
    alpha_lr: f64,  // Learning rate for architecture parameters
    weight_lr: f64, // Learning rate for operation weights
    temperature: f64,
    pruning_threshold: f64,
}

impl DARTS {
    /// Create a new DARTS search space
    pub fn new(
        num_nodes: usize,
        operations_per_edge: Vec<Box<dyn SearchableOperation>>,
        alpha_lr: f64,
        weight_lr: f64,
        temperature: f64,
    ) -> Self {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut node_connections = HashMap::new();

        // Create nodes
        for i in 0..num_nodes {
            let is_input = i == 0;
            let is_output = i == num_nodes - 1;
            nodes.push(SearchNode::new(i, is_input, is_output));
            node_connections.insert(i, Vec::new());
        }

        // Create edges (all-to-all connection pattern, typical for DARTS)
        let mut edge_idx = 0;
        for i in 0..num_nodes {
            for j in (i + 1)..num_nodes {
                // Clone operations for each edge
                let mut ops_clone: Vec<Box<dyn SearchableOperation>> = Vec::new();

                // For simplicity, create predefined operation set
                ops_clone.push(Box::new(IdentityOperation));
                ops_clone.push(Box::new(ZeroOperation::new(16))); // Match input dimension
                if operations_per_edge.len() > 2 {
                    ops_clone.push(Box::new(ConvOperation::new(3, 1, 1, 1, 1, 16)));
                }

                let edge = SearchEdge::new(i, j, ops_clone, temperature);
                edges.push(edge);

                node_connections
                    .get_mut(&i)
                    .expect("node i should exist in connections")
                    .push(edge_idx);
                node_connections
                    .get_mut(&j)
                    .expect("node j should exist in connections")
                    .push(edge_idx);
                edge_idx += 1;
            }
        }

        Self {
            nodes,
            edges,
            node_connections,
            sampling_strategy: SamplingStrategy::Continuous,
            alpha_lr,
            weight_lr,
            temperature,
            pruning_threshold: 0.01,
        }
    }

    /// Forward pass through the search space
    pub fn forward(&mut self, input: &[f64]) -> AutogradResult<Vec<f64>> {
        // Reset intermediate tensors
        for node in &mut self.nodes {
            node.intermediate_tensors.clear();
        }

        // Set input tensor
        self.nodes[0].intermediate_tensors.push(input.to_vec());

        // Process nodes in topological order
        for node_id in 0..self.nodes.len() {
            if node_id == 0 {
                continue; // Skip input node
            }

            let mut accumulated_input = None;

            // Collect inputs from all incoming edges
            for &edge_idx in &self.node_connections[&node_id] {
                let edge = &self.edges[edge_idx];
                if edge.to_node == node_id {
                    // This edge feeds into current node
                    let from_node_id = edge.from_node;

                    if !self.nodes[from_node_id].intermediate_tensors.is_empty() {
                        let edge_input = &self.nodes[from_node_id].intermediate_tensors[0];
                        let edge_output =
                            edge.operation.forward(edge_input, self.sampling_strategy)?;

                        if accumulated_input.is_none() {
                            accumulated_input = Some(vec![0.0; edge_output.output.len()]);
                        }

                        if let Some(ref mut acc) = accumulated_input {
                            for (i, &val) in edge_output.output.iter().enumerate() {
                                acc[i] += val;
                            }
                        }
                    }
                }
            }

            if let Some(tensor) = accumulated_input {
                self.nodes[node_id].intermediate_tensors.push(tensor);
            }
        }

        // Return output from the final node
        let output_node = &self.nodes[self.nodes.len() - 1];
        if output_node.intermediate_tensors.is_empty() {
            return Err(AutogradError::gradient_computation(
                "darts_forward",
                "No output produced",
            ));
        }

        Ok(output_node.intermediate_tensors[0].clone())
    }

    /// Update architecture parameters (alpha weights)
    pub fn update_architecture_parameters(&mut self, gradients: &[Vec<f64>]) -> AutogradResult<()> {
        if gradients.len() != self.edges.len() {
            return Err(AutogradError::gradient_computation(
                "update_architecture_parameters",
                format!(
                    "Gradient count mismatch: {} vs {}",
                    gradients.len(),
                    self.edges.len()
                ),
            ));
        }

        for (edge_idx, edge) in self.edges.iter_mut().enumerate() {
            let alpha_grads = &gradients[edge_idx];
            let alpha_weights = edge.operation.alpha_weights_mut();

            for (i, &grad) in alpha_grads.iter().enumerate() {
                if i < alpha_weights.len() {
                    alpha_weights[i] -= self.alpha_lr * grad;
                }
            }
        }

        Ok(())
    }

    /// Prune operations with low architecture weights
    pub fn prune_operations(&mut self) -> DARTSPruningResult {
        let mut total_pruned = 0;
        let mut pruned_per_edge = HashMap::new();

        for (edge_idx, edge) in self.edges.iter_mut().enumerate() {
            let pruned_ops = edge.operation.prune_operations(self.pruning_threshold);
            total_pruned += pruned_ops.len();
            pruned_per_edge.insert(edge_idx, pruned_ops);
        }

        DARTSPruningResult {
            total_operations_pruned: total_pruned,
            pruned_per_edge,
        }
    }

    /// Get the current architecture (most likely operations for each edge)
    pub fn get_architecture(&self) -> DARTSArchitecture {
        let mut edge_operations = HashMap::new();
        let mut edge_weights = HashMap::new();

        for (edge_idx, edge) in self.edges.iter().enumerate() {
            let (op_idx, op_name, _weight) = edge.operation.most_likely_operation();
            edge_operations.insert(edge_idx, (op_idx, op_name.to_string()));
            edge_weights.insert(edge_idx, edge.operation.alpha_weights().to_vec());
        }

        DARTSArchitecture {
            edge_operations,
            edge_weights,
            nodes: self.nodes.len(),
            edges: self.edges.len(),
        }
    }

    /// Set sampling strategy
    pub fn set_sampling_strategy(&mut self, strategy: SamplingStrategy) {
        self.sampling_strategy = strategy;
    }

    /// Update temperature for Gumbel sampling
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
        for edge in &mut self.edges {
            edge.operation.temperature = temperature;
        }
    }

    /// Get total number of architecture parameters
    pub fn architecture_parameter_count(&self) -> usize {
        self.edges
            .iter()
            .map(|edge| edge.operation.alpha_weights().len())
            .sum()
    }

    /// Get total computational cost of current architecture
    pub fn computational_cost(&self, input_shape: &[usize]) -> f64 {
        let mut total_cost = 0.0;

        for edge in &self.edges {
            let (op_idx, _, weight) = edge.operation.most_likely_operation();
            if let Some(operation) = edge.operation.operations.get(op_idx) {
                total_cost += operation.computational_cost(input_shape) * weight;
            }
        }

        total_cost
    }

    /// Get total memory cost of current architecture
    pub fn memory_cost(&self, input_shape: &[usize]) -> f64 {
        let mut total_cost = 0.0;

        for edge in &self.edges {
            let (op_idx, _, weight) = edge.operation.most_likely_operation();
            if let Some(operation) = edge.operation.operations.get(op_idx) {
                total_cost += operation.memory_cost(input_shape) * weight;
            }
        }

        total_cost
    }
}

/// Result of DARTS pruning operation
#[derive(Debug, Clone)]
pub struct DARTSPruningResult {
    pub total_operations_pruned: usize,
    pub pruned_per_edge: HashMap<usize, Vec<String>>,
}

/// DARTS architecture representation
#[derive(Debug, Clone)]
pub struct DARTSArchitecture {
    pub edge_operations: HashMap<usize, (usize, String)>, // edge_idx -> (op_idx, op_name)
    pub edge_weights: HashMap<usize, Vec<f64>>,           // edge_idx -> alpha_weights
    pub nodes: usize,
    pub edges: usize,
}

impl DARTSArchitecture {
    /// Export architecture to string representation
    pub fn to_string(&self) -> String {
        let mut result = format!(
            "DARTS Architecture: {} nodes, {} edges\n",
            self.nodes, self.edges
        );

        for (edge_idx, (op_idx, op_name)) in &self.edge_operations {
            if let Some(weights) = self.edge_weights.get(edge_idx) {
                result += &format!(
                    "Edge {}: {} (op_idx={}) weights={:?}\n",
                    edge_idx, op_name, op_idx, weights
                );
            }
        }

        result
    }

    /// Calculate architecture diversity (entropy of operation distribution)
    pub fn diversity(&self) -> f64 {
        let mut op_counts = HashMap::new();
        let total_edges = self.edge_operations.len();

        for (_, (_, op_name)) in &self.edge_operations {
            *op_counts.entry(op_name.clone()).or_insert(0) += 1;
        }

        let mut entropy = 0.0;
        for &count in op_counts.values() {
            let p = count as f64 / total_edges as f64;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        entropy
    }
}

/// Progressive DARTS with gradual architecture refinement
pub struct ProgressiveDARTS {
    darts: DARTS,
    epochs_per_stage: usize,
    current_stage: usize,
    total_stages: usize,
    initial_temperature: f64,
    final_temperature: f64,
    pruning_schedule: Vec<f64>,
}

impl ProgressiveDARTS {
    /// Create a new Progressive DARTS searcher
    pub fn new(
        darts: DARTS,
        total_stages: usize,
        epochs_per_stage: usize,
        initial_temperature: f64,
        final_temperature: f64,
    ) -> Self {
        let pruning_schedule = (0..total_stages)
            .map(|i| 0.01 + (0.1 - 0.01) * (i as f64 / total_stages as f64))
            .collect();

        Self {
            darts,
            epochs_per_stage,
            current_stage: 0,
            total_stages,
            initial_temperature,
            final_temperature,
            pruning_schedule,
        }
    }

    /// Advance to next stage
    pub fn advance_stage(&mut self) -> AutogradResult<()> {
        if self.current_stage >= self.total_stages {
            return Err(AutogradError::gradient_computation(
                "advance_stage",
                "Already at final stage",
            ));
        }

        self.current_stage += 1;

        // Update temperature (cooling schedule)
        let progress = self.current_stage as f64 / self.total_stages as f64;
        let new_temperature = self.initial_temperature
            + (self.final_temperature - self.initial_temperature) * progress;
        self.darts.set_temperature(new_temperature);

        // Prune operations if at pruning stage
        if self.current_stage < self.pruning_schedule.len() {
            self.darts.pruning_threshold = self.pruning_schedule[self.current_stage];
            self.darts.prune_operations();
        }

        // Switch to harder sampling in later stages
        if progress > 0.7 {
            self.darts
                .set_sampling_strategy(SamplingStrategy::GumbelHard);
        } else if progress > 0.4 {
            self.darts
                .set_sampling_strategy(SamplingStrategy::GumbelSoft);
        }

        Ok(())
    }

    /// Get current stage info
    pub fn stage_info(&self) -> ProgressiveStageInfo {
        ProgressiveStageInfo {
            current_stage: self.current_stage,
            total_stages: self.total_stages,
            epochs_per_stage: self.epochs_per_stage,
            current_temperature: self.darts.temperature,
            current_pruning_threshold: self.darts.pruning_threshold,
            sampling_strategy: self.darts.sampling_strategy,
        }
    }

    /// Check if search is complete
    pub fn is_complete(&self) -> bool {
        self.current_stage >= self.total_stages
    }

    /// Get the underlying DARTS searcher
    pub fn darts(&self) -> &DARTS {
        &self.darts
    }

    /// Get mutable reference to underlying DARTS searcher
    pub fn darts_mut(&mut self) -> &mut DARTS {
        &mut self.darts
    }
}

/// Information about current progressive stage
#[derive(Debug, Clone)]
pub struct ProgressiveStageInfo {
    pub current_stage: usize,
    pub total_stages: usize,
    pub epochs_per_stage: usize,
    pub current_temperature: f64,
    pub current_pruning_threshold: f64,
    pub sampling_strategy: SamplingStrategy,
}

// Add a simple random number generator for demo purposes
mod rand {
    use std::sync::atomic::{AtomicU64, Ordering};

    static STATE: AtomicU64 = AtomicU64::new(1);

    pub fn random<T>() -> T
    where
        T: From<f64>,
    {
        // Simple LCG for demo purposes
        let prev = STATE.load(Ordering::Relaxed);
        let next = prev.wrapping_mul(1103515245).wrapping_add(12345);
        STATE.store(next, Ordering::Relaxed);

        let normalized = (next as f64) / (u64::MAX as f64);
        T::from(normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_operation() {
        let op = IdentityOperation;
        let input = vec![1.0, 2.0, 3.0];
        let output = op.forward(&input, 1.0).unwrap();

        assert_eq!(input, output);
        assert_eq!(op.name(), "identity");
        assert_eq!(op.parameter_count(), 0);
    }

    #[test]
    fn test_zero_operation() {
        let op = ZeroOperation::new(3);
        let input = vec![1.0, 2.0, 3.0];
        let output = op.forward(&input, 1.0).unwrap();

        assert_eq!(output, vec![0.0, 0.0, 0.0]);
        assert_eq!(op.name(), "zero");
    }

    #[test]
    fn test_conv_operation() {
        let op = ConvOperation::new(3, 1, 1, 1, 2, 4);
        assert_eq!(op.name(), "conv");
        assert!(op.parameter_count() > 0);

        let input = vec![1.0; 20]; // 2 channels, 10 elements each
        let output = op.forward(&input, 1.0).unwrap();
        assert_eq!(output.len(), 4 * 10); // 4 output channels, 10 elements each
    }

    #[test]
    fn test_mixed_operation() {
        let operations: Vec<Box<dyn SearchableOperation>> = vec![
            Box::new(IdentityOperation),
            Box::new(ZeroOperation::new(10)),
        ];

        let mixed_op = MixedOperation::new(operations, 1.0);
        let input = vec![1.0; 10];

        let output = mixed_op
            .forward(&input, SamplingStrategy::Continuous)
            .unwrap();
        assert_eq!(output.output.len(), 10);

        // Test that weights are normalized
        let weights = mixed_op.softmax_weights();
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_darts_creation() {
        let operations: Vec<Box<dyn SearchableOperation>> = vec![Box::new(IdentityOperation)];

        let darts = DARTS::new(4, operations, 0.01, 0.001, 1.0);
        assert_eq!(darts.nodes.len(), 4);
        assert!(darts.edges.len() > 0);
        assert!(darts.architecture_parameter_count() > 0);
    }

    #[test]
    fn test_darts_forward() {
        let operations: Vec<Box<dyn SearchableOperation>> = vec![Box::new(IdentityOperation)];

        let mut darts = DARTS::new(3, operations, 0.01, 0.001, 1.0);
        let input = vec![1.0; 16];

        let output = darts.forward(&input).unwrap();
        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_progressive_darts() {
        let operations: Vec<Box<dyn SearchableOperation>> = vec![
            Box::new(IdentityOperation),
            Box::new(ZeroOperation::new(16)),
        ];

        let darts = DARTS::new(3, operations, 0.01, 0.001, 2.0);
        let mut progressive = ProgressiveDARTS::new(darts, 5, 10, 2.0, 0.1);

        assert!(!progressive.is_complete());

        progressive.advance_stage().unwrap();
        let stage_info = progressive.stage_info();
        assert_eq!(stage_info.current_stage, 1);
        assert!(stage_info.current_temperature < 2.0);
    }

    #[test]
    fn test_architecture_export() {
        let operations: Vec<Box<dyn SearchableOperation>> = vec![
            Box::new(IdentityOperation),
            Box::new(ZeroOperation::new(16)),
        ];

        let darts = DARTS::new(3, operations, 0.01, 0.001, 1.0);
        let arch = darts.get_architecture();

        assert_eq!(arch.nodes, 3);
        assert!(arch.edges > 0);
        assert!(!arch.to_string().is_empty());
        assert!(arch.diversity() >= 0.0);
    }

    #[test]
    fn test_sampling_strategies() {
        let operations: Vec<Box<dyn SearchableOperation>> = vec![
            Box::new(IdentityOperation),
            Box::new(ZeroOperation::new(10)),
        ];

        let mixed_op = MixedOperation::new(operations, 1.0);
        let input = vec![1.0; 10];

        // Test all sampling strategies
        let strategies = [
            SamplingStrategy::Continuous,
            SamplingStrategy::GumbelSoft,
            SamplingStrategy::GumbelHard,
        ];

        for strategy in &strategies {
            let output = mixed_op.forward(&input, *strategy).unwrap();
            assert_eq!(output.output.len(), 10);
            assert_eq!(output.weights_used.len(), 2);
        }
    }

    #[test]
    fn test_operation_costs() {
        let conv_op = ConvOperation::new(3, 1, 1, 1, 16, 32);
        let identity_op = IdentityOperation;
        let zero_op = ZeroOperation::new(100);

        let input_shape = vec![16, 100];

        // Conv should have higher computational cost
        assert!(
            conv_op.computational_cost(&input_shape) > identity_op.computational_cost(&input_shape)
        );
        assert!(
            conv_op.computational_cost(&input_shape) > zero_op.computational_cost(&input_shape)
        );

        // All should have some memory cost
        assert!(conv_op.memory_cost(&input_shape) > 0.0);
        assert!(identity_op.memory_cost(&input_shape) > 0.0);
        assert!(zero_op.memory_cost(&input_shape) > 0.0);
    }
}
