//! Research neural network layers and components
//!
//! This module contains implementations of cutting-edge research architectures
//! including Neural ODEs, Differentiable NAS, Meta-learning, and more.

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::{boxed::Box, collections::HashMap, string::String, vec::Vec};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, sync::Arc, vec::Vec};

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

/// Neural Ordinary Differential Equation (NODE) layer
///
/// Neural ODEs model the hidden state as a continuous-time dynamical system
/// defined by an ODE: dh/dt = f(h(t), t, θ) where f is a neural network.
///
/// This allows for more memory-efficient training and variable-depth computation.
pub struct NeuralODE {
    base: ModuleBase,
    func: Box<dyn Module>,
    solver: ODESolver,
    #[allow(dead_code)]
    atol: f32,
    #[allow(dead_code)]
    rtol: f32,
    max_steps: usize,
}

/// ODE solver methods
#[derive(Debug, Clone, Copy)]
pub enum ODESolver {
    /// Euler's method (simplest, first-order)
    Euler,
    /// Runge-Kutta 4th order (more accurate)
    RK4,
    /// Adaptive step size methods
    Dopri5,
}

impl NeuralODE {
    /// Create a new Neural ODE layer
    ///
    /// # Arguments
    /// * `func` - The neural network that defines the ODE dynamics
    /// * `solver` - The numerical ODE solver to use
    /// * `atol` - Absolute tolerance for adaptive solvers
    /// * `rtol` - Relative tolerance for adaptive solvers
    /// * `max_steps` - Maximum number of integration steps
    pub fn new(
        func: Box<dyn Module>,
        solver: ODESolver,
        atol: f32,
        rtol: f32,
        max_steps: usize,
    ) -> Self {
        Self {
            base: ModuleBase::new(),
            func,
            solver,
            atol,
            rtol,
            max_steps,
        }
    }

    /// Solve the ODE from t0 to t1 using the specified solver
    fn solve_ode(&self, y0: &Tensor, t0: f32, t1: f32) -> Result<Tensor> {
        match self.solver {
            ODESolver::Euler => self.euler_solve(y0, t0, t1),
            ODESolver::RK4 => self.rk4_solve(y0, t0, t1),
            ODESolver::Dopri5 => self.dopri5_solve(y0, t0, t1),
        }
    }

    /// Euler's method for ODE solving
    fn euler_solve(&self, y0: &Tensor, t0: f32, t1: f32) -> Result<Tensor> {
        let h = (t1 - t0) / self.max_steps as f32;
        let mut y = y0.clone();
        let mut t = t0;

        for _ in 0..self.max_steps {
            let dy_dt = self.func.forward(&y)?;
            let h_tensor = full(y.shape().dims(), h)?;
            let delta_y = dy_dt.mul_op(&h_tensor)?;
            y = y.add_op(&delta_y)?;
            t += h;

            if t >= t1 {
                break;
            }
        }

        Ok(y)
    }

    /// Runge-Kutta 4th order method
    fn rk4_solve(&self, y0: &Tensor, t0: f32, t1: f32) -> Result<Tensor> {
        let h = (t1 - t0) / self.max_steps as f32;
        let mut y = y0.clone();
        let mut t = t0;

        for _ in 0..self.max_steps {
            let k1 = self.func.forward(&y)?;

            let h_tensor = full(y.shape().dims(), h)?;
            let half_tensor = full(y.shape().dims(), 0.5)?;
            let two_tensor = full(y.shape().dims(), 2.0)?;
            let six_tensor = full(y.shape().dims(), 6.0)?;

            let k1_half_h = k1.mul_op(&h_tensor)?.mul_op(&half_tensor)?;
            let y_k1 = y.add_op(&k1_half_h)?;
            let k2 = self.func.forward(&y_k1)?;

            let k2_half_h = k2.mul_op(&h_tensor)?.mul_op(&half_tensor)?;
            let y_k2 = y.add_op(&k2_half_h)?;
            let k3 = self.func.forward(&y_k2)?;

            let k3_h = k3.mul_op(&h_tensor)?;
            let y_k3 = y.add_op(&k3_h)?;
            let k4 = self.func.forward(&y_k3)?;

            // y = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            let k2_times_2 = k2.mul_op(&two_tensor)?;
            let k3_times_2 = k3.mul_op(&two_tensor)?;
            let sum = k1.add_op(&k2_times_2)?.add_op(&k3_times_2)?.add_op(&k4)?;
            let weighted_sum = sum.mul_op(&h_tensor)?.div(&six_tensor)?;
            y = y.add_op(&weighted_sum)?;

            t += h;
            if t >= t1 {
                break;
            }
        }

        Ok(y)
    }

    /// Dormand-Prince 5th order adaptive method (simplified)
    fn dopri5_solve(&self, y0: &Tensor, t0: f32, t1: f32) -> Result<Tensor> {
        // For simplicity, use RK4 with adaptive step size
        // In a full implementation, this would use the Dormand-Prince coefficients
        self.rk4_solve(y0, t0, t1)
    }
}

impl Module for NeuralODE {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Integrate from t=0 to t=1
        self.solve_ode(input, 0.0, 1.0)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Differentiable Neural Architecture Search (DARTS) cell
///
/// DARTS enables gradient-based architecture optimization by using
/// continuous relaxation of the architecture search space.
pub struct DARTSCell {
    base: ModuleBase,
    operations: Vec<Box<dyn Module>>,
    alpha: Parameter, // Architecture parameters
    #[allow(dead_code)]
    num_nodes: usize,
    #[allow(dead_code)]
    num_ops: usize,
}

impl DARTSCell {
    /// Create a new DARTS cell
    ///
    /// # Arguments
    /// * `operations` - Set of candidate operations
    /// * `num_nodes` - Number of intermediate nodes
    pub fn new(operations: Vec<Box<dyn Module>>, num_nodes: usize) -> Result<Self> {
        let num_ops = operations.len();
        let mut base = ModuleBase::new();

        // Initialize architecture parameters (logits)
        let alpha_size = num_nodes * (num_nodes + 1) / 2 * num_ops;
        let alpha_data = zeros(&[alpha_size])?;
        base.register_parameter("alpha".to_string(), Parameter::new(alpha_data));

        // Get alpha parameter before moving base
        let alpha = base.parameters["alpha"].clone();

        Ok(Self {
            base,
            operations,
            alpha,
            num_nodes,
            num_ops,
        })
    }

    /// Apply softmax to architecture parameters to get weights
    fn get_architecture_weights(&self) -> Result<Tensor> {
        let alpha_tensor = self.alpha.tensor().read().clone();
        alpha_tensor.softmax(-1)
    }
}

impl Module for DARTSCell {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let weights = self.get_architecture_weights()?;
        let weight_data = weights.to_vec()?;

        // For each edge, compute weighted sum of operations
        // This is a simplified implementation
        let mut output = input.clone();

        for (i, op) in self.operations.iter().enumerate() {
            let op_output = op.forward(input)?;
            let weight = weight_data[i % weight_data.len()];
            let weight_tensor = full(op_output.shape().dims(), weight)?;
            let weighted_output = op_output.mul_op(&weight_tensor)?;

            if i == 0 {
                output = weighted_output;
            } else {
                output = output.add_op(&weighted_output)?;
            }
        }

        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = self.base.parameters.clone();

        // Add parameters from all operations
        for (i, op) in self.operations.iter().enumerate() {
            let op_params = op.parameters();
            for (name, param) in op_params {
                params.insert(format!("op_{}_{}", i, name), param);
            }
        }

        params
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Model-Agnostic Meta-Learning (MAML) module
///
/// MAML trains a model to quickly adapt to new tasks with minimal examples
/// by optimizing for fast learning rather than task-specific performance.
pub struct MAMLModule {
    base: ModuleBase,
    inner_model: Box<dyn Module>,
    #[allow(dead_code)]
    inner_lr: f32,
    inner_steps: usize,
}

impl MAMLModule {
    /// Create a new MAML module
    ///
    /// # Arguments
    /// * `inner_model` - The base model to meta-learn
    /// * `inner_lr` - Learning rate for inner loop adaptation
    /// * `inner_steps` - Number of gradient steps in inner loop
    pub fn new(inner_model: Box<dyn Module>, inner_lr: f32, inner_steps: usize) -> Self {
        Self {
            base: ModuleBase::new(),
            inner_model,
            inner_lr,
            inner_steps,
        }
    }

    /// Perform inner loop adaptation on a support set
    pub fn adapt(&mut self, support_x: &Tensor, support_y: &Tensor) -> Result<()> {
        // Perform gradient descent on the support set
        // This is a simplified implementation - real MAML would use higher-order gradients

        for _ in 0..self.inner_steps {
            let prediction = self.inner_model.forward(support_x)?;

            // Compute loss (MSE for simplicity)
            let diff = prediction.sub(support_y)?;
            let squared_diff = diff.mul_op(&diff)?;
            // Placeholder: would compute mean in real implementation
            let loss = squared_diff; // .mean() - mean function not available yet

            // In a full implementation, we would compute gradients and update parameters
            // For now, this is a placeholder
            let _ = loss; // Suppress warning
        }

        Ok(())
    }

    /// Meta-forward pass: adapt on support set, then evaluate on query set
    pub fn meta_forward(
        &mut self,
        support_x: &Tensor,
        support_y: &Tensor,
        query_x: &Tensor,
    ) -> Result<Tensor> {
        // Save original parameters
        let original_params = self.inner_model.parameters();

        // Adapt on support set
        self.adapt(support_x, support_y)?;

        // Evaluate on query set
        let query_prediction = self.inner_model.forward(query_x)?;

        // Restore original parameters for next task
        // In practice, we'd use the gradient information for meta-learning
        let _ = original_params; // Suppress warning

        Ok(query_prediction)
    }
}

impl Module for MAMLModule {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.inner_model.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.inner_model.parameters()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
        self.inner_model.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.inner_model.named_parameters()
    }
}

/// Capsule Network layer
///
/// Capsules are groups of neurons that represent the instantiation parameters
/// of a specific type of entity (e.g., pose, lighting, deformation).
pub struct CapsuleLayer {
    base: ModuleBase,
    in_capsules: usize,
    out_capsules: usize,
    #[allow(dead_code)]
    in_dim: usize,
    #[allow(dead_code)]
    out_dim: usize,
    num_routing: usize,
}

impl CapsuleLayer {
    /// Create a new Capsule layer
    ///
    /// # Arguments
    /// * `in_capsules` - Number of input capsules
    /// * `out_capsules` - Number of output capsules  
    /// * `in_dim` - Dimension of input capsules
    /// * `out_dim` - Dimension of output capsules
    /// * `num_routing` - Number of routing iterations
    pub fn new(
        in_capsules: usize,
        out_capsules: usize,
        in_dim: usize,
        out_dim: usize,
        num_routing: usize,
    ) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Weight tensor for transformation: [out_capsules, in_capsules, out_dim, in_dim]
        let weight_shape = vec![out_capsules, in_capsules, out_dim, in_dim];
        let weight = randn(&weight_shape)?;
        base.register_parameter("weight".to_string(), Parameter::new(weight));

        Ok(Self {
            base,
            in_capsules,
            out_capsules,
            in_dim,
            out_dim,
            num_routing,
        })
    }

    /// Squash function to ensure capsule length is between 0 and 1
    fn squash(&self, tensor: &Tensor) -> Result<Tensor> {
        // ||s||² / (1 + ||s||²) * s / ||s||
        let squared = tensor.mul_op(tensor)?;
        let squared_sum = squared.sum()?;
        let norm = squared_sum.sqrt()?;
        let norm_squared = squared_sum.clone();

        let one = ones(&[1])?;
        let denominator = one.add_op(&norm_squared)?;
        let scale = norm_squared.div(&denominator)?;

        let unit_vector = tensor.div(&norm)?;
        scale.mul_op(&unit_vector)
    }

    /// Dynamic routing algorithm
    fn routing(&self, u_hat: &Tensor) -> Result<Tensor> {
        // Initialize routing logits b_ij to 0
        let batch_size = u_hat.shape().dims()[0];
        let mut b = zeros(&[batch_size, self.in_capsules, self.out_capsules])?;

        for _ in 0..self.num_routing {
            // Softmax over output capsules
            let _c = b.softmax(-1)?;

            // Weighted sum: s_j = Σ c_ij * u_hat_j|i
            // This is a simplified implementation
            let s = u_hat.clone(); // Placeholder

            // Squash to get output capsules
            let _v = self.squash(&s)?;

            // Update routing logits: b_ij += u_hat_j|i · v_j
            // Simplified: just update b with a small value
            let update = full(b.shape().dims(), 0.1)?;
            b = b.add_op(&update)?;
        }

        let _c_final = b.softmax(-1)?;
        let s_final = u_hat.clone(); // Placeholder
        self.squash(&s_final)
    }
}

impl Module for CapsuleLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Input shape: [batch_size, in_capsules, in_dim]
        let weight = self.base.parameters["weight"].tensor().read().clone();

        // Compute prediction vectors u_hat = W_ij * u_i
        // This is a simplified implementation
        let u_hat = input.matmul(&weight)?;

        // Apply dynamic routing
        self.routing(&u_hat)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Graph Convolutional Network layer
///
/// GCNs operate on graph-structured data by aggregating information
/// from neighboring nodes to update node representations.
pub struct GraphConvLayer {
    base: ModuleBase,
    #[allow(dead_code)]
    in_features: usize,
    #[allow(dead_code)]
    out_features: usize,
    use_bias: bool,
}

impl GraphConvLayer {
    /// Create a new Graph Convolution layer
    ///
    /// # Arguments
    /// * `in_features` - Number of input features per node
    /// * `out_features` - Number of output features per node
    /// * `use_bias` - Whether to use bias term
    pub fn new(in_features: usize, out_features: usize, use_bias: bool) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Weight matrix
        let weight = randn(&[in_features, out_features])?;
        base.register_parameter("weight".to_string(), Parameter::new(weight));

        // Bias vector
        if use_bias {
            let bias = zeros(&[out_features])?;
            base.register_parameter("bias".to_string(), Parameter::new(bias));
        }

        Ok(Self {
            base,
            in_features,
            out_features,
            use_bias,
        })
    }

    /// Normalize adjacency matrix (add self-loops and compute D^(-1/2) A D^(-1/2))
    #[allow(dead_code)]
    fn normalize_adjacency(&self, adj: &Tensor) -> Result<Tensor> {
        // Add self-loops: A = A + I
        let num_nodes = adj.shape().dims()[0];
        let identity = eye(num_nodes)?;
        let adj_with_self_loops = adj.add_op(&identity)?;

        // Compute degree matrix D
        let degrees = adj_with_self_loops.sum_dim(&[1], false)?;

        // D^(-1/2)
        let _degrees_sqrt = degrees.pow(-0.5)?;

        // Create diagonal matrix from degrees_sqrt
        // This is simplified - real implementation would use proper diagonal matrix ops
        let normalized_adj = adj_with_self_loops.clone(); // Placeholder

        Ok(normalized_adj)
    }
}

impl Module for GraphConvLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // input should be a tuple of (node_features, adjacency_matrix)
        // For simplicity, we'll assume input is just node features
        // and the adjacency matrix is passed separately or stored

        let weight = self.base.parameters["weight"].tensor().read().clone();

        // Linear transformation: X' = X * W
        let transformed = input.matmul(&weight)?;

        // In a real implementation, we would multiply by normalized adjacency matrix
        // A_norm * X' where A_norm is the normalized adjacency matrix
        let mut output = transformed;

        // Add bias if present
        if self.use_bias {
            let bias = self.base.parameters["bias"].tensor().read().clone();
            output = output.add_op(&bias)?;
        }

        Ok(output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

/// Graph Attention Network layer
///
/// GAT uses attention mechanisms to learn the relative importance
/// of neighboring nodes when aggregating information.
pub struct GraphAttentionLayer {
    base: ModuleBase,
    #[allow(dead_code)]
    in_features: usize,
    #[allow(dead_code)]
    out_features: usize,
    num_heads: usize,
    #[allow(dead_code)]
    dropout: f32,
    #[allow(dead_code)]
    alpha: f32, // LeakyReLU negative slope
}

impl GraphAttentionLayer {
    /// Create a new Graph Attention layer
    ///
    /// # Arguments
    /// * `in_features` - Number of input features per node
    /// * `out_features` - Number of output features per node
    /// * `num_heads` - Number of attention heads
    /// * `dropout` - Dropout probability for attention weights
    /// * `alpha` - Negative slope for LeakyReLU in attention
    pub fn new(
        in_features: usize,
        out_features: usize,
        num_heads: usize,
        dropout: f32,
        alpha: f32,
    ) -> Result<Self> {
        let mut base = ModuleBase::new();

        // Linear transformation weights for each head
        for h in 0..num_heads {
            let weight = randn(&[in_features, out_features])?;
            base.register_parameter(format!("weight_{}", h), Parameter::new(weight));

            // Attention parameters a^T [W h_i || W h_j]
            let att = randn(&[2 * out_features, 1])?;
            base.register_parameter(format!("att_{}", h), Parameter::new(att));
        }

        Ok(Self {
            base,
            in_features,
            out_features,
            num_heads,
            dropout,
            alpha,
        })
    }

    /// Compute attention coefficients
    #[allow(dead_code)]
    fn attention(&self, h_i: &Tensor, h_j: &Tensor, head: usize) -> Result<Tensor> {
        let att = self.base.parameters[&format!("att_{}", head)]
            .tensor()
            .read()
            .clone();

        // Concatenate h_i and h_j
        let concat = Tensor::cat(&[h_i, h_j], -1)?;

        // Compute attention: a^T [W h_i || W h_j]
        let e = concat.matmul(&att)?;

        // Apply LeakyReLU
        let alpha_tensor = full(e.shape().dims(), self.alpha)?;
        let zero = zeros(e.shape().dims())?;
        let positive = e.maximum(&zero)?;
        let negative = e.minimum(&zero)?;
        let leaky_negative = negative.mul_op(&alpha_tensor)?;
        positive.add_op(&leaky_negative)
    }
}

impl Module for GraphAttentionLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Multi-head attention
        let mut head_outputs = Vec::new();

        for h in 0..self.num_heads {
            let weight = self.base.parameters[&format!("weight_{}", h)]
                .tensor()
                .read()
                .clone();

            // Linear transformation
            let h_transformed = input.matmul(&weight)?;

            // For simplicity, we'll just return the transformed features
            // Real GAT would compute attention weights and aggregate neighbors
            head_outputs.push(h_transformed);
        }

        // Concatenate or average multi-head outputs
        if head_outputs.len() == 1 {
            Ok(head_outputs.into_iter().next().unwrap())
        } else {
            // Average the heads
            let mut sum = head_outputs[0].clone();
            for head in head_outputs.iter().skip(1) {
                sum = sum.add_op(head)?;
            }
            let num_heads_tensor = full(sum.shape().dims(), self.num_heads as f32)?;
            sum.div(&num_heads_tensor)
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::linear::Linear;

    #[test]
    fn test_neural_ode_creation() {
        let inner_model = Box::new(Linear::new(10, 10, true));
        let node = NeuralODE::new(inner_model, ODESolver::Euler, 1e-3, 1e-3, 100);

        assert_eq!(node.max_steps, 100);
        assert!((node.atol - 1e-3).abs() < 1e-6);
    }

    #[test]
    fn test_capsule_layer_creation() {
        let capsule = CapsuleLayer::new(10, 5, 8, 16, 3).unwrap();

        assert_eq!(capsule.in_capsules, 10);
        assert_eq!(capsule.out_capsules, 5);
        assert_eq!(capsule.in_dim, 8);
        assert_eq!(capsule.out_dim, 16);
        assert_eq!(capsule.num_routing, 3);
    }

    #[test]
    fn test_graph_conv_creation() {
        let gcn = GraphConvLayer::new(64, 32, true).unwrap();

        assert_eq!(gcn.in_features, 64);
        assert_eq!(gcn.out_features, 32);
        assert!(gcn.use_bias);
    }

    #[test]
    fn test_graph_attention_creation() {
        let gat = GraphAttentionLayer::new(64, 32, 8, 0.1, 0.2).unwrap();

        assert_eq!(gat.in_features, 64);
        assert_eq!(gat.out_features, 32);
        assert_eq!(gat.num_heads, 8);
        assert!((gat.dropout - 0.1).abs() < 1e-6);
        assert!((gat.alpha - 0.2).abs() < 1e-6);
    }
}
