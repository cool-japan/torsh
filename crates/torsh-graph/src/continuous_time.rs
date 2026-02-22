//! Continuous-Time Graph Neural Networks
//!
//! This module implements neural networks for continuous-time dynamic graphs where
//! nodes and edges have associated timestamps and can evolve continuously.
//!
//! # Key Features:
//! - Temporal Graph Networks (TGN) with memory modules
//! - Neural Ordinary Differential Equations (Neural ODE) for graphs
//! - Continuous-time message passing
//! - Temporal attention mechanisms
//! - Time encoding for irregular events
//!
//! # Applications:
//! - Social network evolution modeling
//! - Traffic flow prediction
//! - Financial transaction analysis
//! - Biological system dynamics
//!
//! # References:
//! - Rossi et al. "Temporal Graph Networks for Deep Learning on Dynamic Graphs" (ICML 2020)
//! - Chen et al. "Neural Ordinary Differential Equations" (NeurIPS 2018)
//! - Xu et al. "Inductive Representation Learning on Temporal Graphs" (ICLR 2020)

use crate::{GraphData, GraphLayer};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{thread_rng, Normal};
use std::collections::HashMap;
use torsh_core::device::DeviceType;
use torsh_tensor::{
    creation::{from_vec, zeros},
    Tensor,
};

/// Time encoding module
///
/// Encodes continuous time values into fixed-dimensional representations
/// using Fourier features or learnable embeddings.
#[derive(Debug, Clone)]
pub struct TimeEncoder {
    /// Dimensionality of time encoding
    dim: usize,
    /// Frequency scaling factor
    frequency_scale: f32,
    /// Learnable weights for time encoding
    weight: Tensor,
    bias: Tensor,
}

impl TimeEncoder {
    /// Create a new time encoder
    ///
    /// # Arguments:
    /// * `dim` - Output dimension
    /// * `frequency_scale` - Scaling factor for frequencies
    ///
    /// # Example:
    /// ```rust
    /// use torsh_graph::continuous_time::TimeEncoder;
    ///
    /// let encoder = TimeEncoder::new(64, 1.0).unwrap();
    /// ```
    pub fn new(dim: usize, frequency_scale: f32) -> Result<Self, Box<dyn std::error::Error>> {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.1)?;

        // Initialize learnable parameters
        let weight_values: Vec<f32> = (0..dim)
            .map(|_| normal.sample(&mut rng) as f32)
            .collect();
        let weight = from_vec(weight_values, &[dim], DeviceType::Cpu)?;

        let bias_values: Vec<f32> = (0..dim)
            .map(|_| normal.sample(&mut rng) as f32)
            .collect();
        let bias = from_vec(bias_values, &[dim], DeviceType::Cpu)?;

        Ok(Self {
            dim,
            frequency_scale,
            weight,
            bias,
        })
    }

    /// Encode time values
    ///
    /// Uses combination of sin/cos features (Fourier features) and learnable transformation
    ///
    /// # Arguments:
    /// * `times` - Time values [batch_size]
    ///
    /// # Returns:
    /// Time encodings [batch_size, dim]
    pub fn encode(&self, times: &[f32]) -> Result<Tensor, Box<dyn std::error::Error>> {
        let batch_size = times.len();
        let mut encoding = Vec::with_capacity(batch_size * self.dim);

        let weight_data = self.weight.to_vec()?;
        let bias_data = self.bias.to_vec()?;

        for &t in times {
            for i in 0..self.dim {
                // Fourier features: alternate sin and cos
                let freq = self.frequency_scale * weight_data[i];
                let phase = bias_data[i];

                let value = if i % 2 == 0 {
                    (freq * t + phase).sin()
                } else {
                    (freq * t + phase).cos()
                };
                encoding.push(value);
            }
        }

        from_vec(encoding, &[batch_size, self.dim], DeviceType::Cpu)
    }

    /// Get parameters
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

/// Memory module for temporal graph networks
///
/// Maintains a learnable memory state for each node that gets updated
/// when the node participates in interactions.
#[derive(Debug, Clone)]
pub struct NodeMemory {
    /// Number of nodes
    num_nodes: usize,
    /// Memory dimension
    memory_dim: usize,
    /// Current memory states [num_nodes, memory_dim]
    memory: Tensor,
    /// Last update timestamps for each node
    last_update: Vec<f32>,
    /// Memory update function type
    update_type: MemoryUpdateType,
}

/// Type of memory update function
#[derive(Debug, Clone, Copy)]
pub enum MemoryUpdateType {
    /// GRU-based update
    GRU,
    /// RNN-based update
    RNN,
    /// Simple moving average
    MovingAverage { alpha: f32 },
}

impl NodeMemory {
    /// Create a new node memory module
    ///
    /// # Arguments:
    /// * `num_nodes` - Number of nodes
    /// * `memory_dim` - Dimension of memory vectors
    /// * `update_type` - Type of memory update
    pub fn new(
        num_nodes: usize,
        memory_dim: usize,
        update_type: MemoryUpdateType,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let memory = zeros(&[num_nodes, memory_dim], DeviceType::Cpu)?;
        let last_update = vec![0.0; num_nodes];

        Ok(Self {
            num_nodes,
            memory_dim,
            memory,
            last_update,
            update_type,
        })
    }

    /// Get memory for specific nodes
    ///
    /// # Arguments:
    /// * `node_ids` - IDs of nodes to retrieve memory for
    ///
    /// # Returns:
    /// Memory states [len(node_ids), memory_dim]
    pub fn get_memory(&self, node_ids: &[usize]) -> Result<Tensor, Box<dyn std::error::Error>> {
        let memory_data = self.memory.to_vec()?;
        let mut result = Vec::with_capacity(node_ids.len() * self.memory_dim);

        for &node_id in node_ids {
            for d in 0..self.memory_dim {
                result.push(memory_data[node_id * self.memory_dim + d]);
            }
        }

        from_vec(result, &[node_ids.len(), self.memory_dim], DeviceType::Cpu)
    }

    /// Update memory for specific nodes
    ///
    /// # Arguments:
    /// * `node_ids` - IDs of nodes to update
    /// * `messages` - Update messages [len(node_ids), memory_dim]
    /// * `timestamps` - Timestamps of updates
    pub fn update_memory(
        &mut self,
        node_ids: &[usize],
        messages: &Tensor,
        timestamps: &[f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let message_data = messages.to_vec()?;
        let mut memory_data = self.memory.to_vec()?;

        match self.update_type {
            MemoryUpdateType::GRU => {
                // Simplified GRU update (full implementation would use gates)
                for (i, &node_id) in node_ids.iter().enumerate() {
                    for d in 0..self.memory_dim {
                        let message = message_data[i * self.memory_dim + d];
                        let old_mem = memory_data[node_id * self.memory_dim + d];

                        // Simplified: weighted average
                        memory_data[node_id * self.memory_dim + d] = 0.5 * old_mem + 0.5 * message;
                    }
                    self.last_update[node_id] = timestamps[i];
                }
            }
            MemoryUpdateType::RNN => {
                // Simple RNN update: h_t = tanh(W_h * h_{t-1} + W_x * x_t)
                for (i, &node_id) in node_ids.iter().enumerate() {
                    for d in 0..self.memory_dim {
                        let message = message_data[i * self.memory_dim + d];
                        let old_mem = memory_data[node_id * self.memory_dim + d];

                        memory_data[node_id * self.memory_dim + d] = (old_mem + message).tanh();
                    }
                    self.last_update[node_id] = timestamps[i];
                }
            }
            MemoryUpdateType::MovingAverage { alpha } => {
                // Exponential moving average
                for (i, &node_id) in node_ids.iter().enumerate() {
                    for d in 0..self.memory_dim {
                        let message = message_data[i * self.memory_dim + d];
                        let old_mem = memory_data[node_id * self.memory_dim + d];

                        memory_data[node_id * self.memory_dim + d] = alpha * old_mem + (1.0 - alpha) * message;
                    }
                    self.last_update[node_id] = timestamps[i];
                }
            }
        }

        self.memory = from_vec(memory_data, &[self.num_nodes, self.memory_dim], DeviceType::Cpu)?;
        Ok(())
    }

    /// Reset memory to zero
    pub fn reset(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.memory = zeros(&[self.num_nodes, self.memory_dim], DeviceType::Cpu)?;
        self.last_update = vec![0.0; self.num_nodes];
        Ok(())
    }
}

/// Temporal Graph Network (TGN) layer
///
/// Processes continuous-time dynamic graphs by maintaining node memories
/// and using time-aware message passing.
#[derive(Debug, Clone)]
pub struct TGNLayer {
    /// Input feature dimension
    in_features: usize,
    /// Output feature dimension
    out_features: usize,
    /// Memory dimension
    memory_dim: usize,
    /// Time encoder
    time_encoder: TimeEncoder,
    /// Message function MLP
    message_weight: Tensor,
    message_bias: Tensor,
    /// Embedding function MLP
    embedding_weight: Tensor,
    embedding_bias: Tensor,
}

impl TGNLayer {
    /// Create a new TGN layer
    ///
    /// # Arguments:
    /// * `in_features` - Input feature dimension
    /// * `out_features` - Output feature dimension
    /// * `memory_dim` - Memory dimension
    /// * `time_dim` - Time encoding dimension
    pub fn new(
        in_features: usize,
        out_features: usize,
        memory_dim: usize,
        time_dim: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.1)?;

        let time_encoder = TimeEncoder::new(time_dim, 1.0)?;

        // Message function: [src_feat, dst_feat, edge_feat, time_enc] -> memory_dim
        let message_input_dim = in_features * 2 + time_dim;
        let message_weight = Self::init_weight(message_input_dim, memory_dim, &normal, &mut rng)?;
        let message_bias = zeros(&[memory_dim], DeviceType::Cpu)?;

        // Embedding function: [feat, memory, time_enc] -> out_features
        let embed_input_dim = in_features + memory_dim + time_dim;
        let embedding_weight = Self::init_weight(embed_input_dim, out_features, &normal, &mut rng)?;
        let embedding_bias = zeros(&[out_features], DeviceType::Cpu)?;

        Ok(Self {
            in_features,
            out_features,
            memory_dim,
            time_encoder,
            message_weight,
            message_bias,
            embedding_weight,
            embedding_bias,
        })
    }

    /// Initialize weight tensor
    fn init_weight(
        in_dim: usize,
        out_dim: usize,
        normal: &Normal<f64>,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let std = (2.0 / (in_dim + out_dim) as f64).sqrt();
        let values: Vec<f32> = (0..in_dim * out_dim)
            .map(|_| (normal.sample(rng) * std) as f32)
            .collect();
        from_vec(values, &[in_dim, out_dim], DeviceType::Cpu)
    }

    /// Compute messages for edges
    ///
    /// # Arguments:
    /// * `src_features` - Source node features
    /// * `dst_features` - Destination node features
    /// * `timestamps` - Edge timestamps
    ///
    /// # Returns:
    /// Messages for each edge
    pub fn compute_messages(
        &self,
        src_features: &Tensor,
        dst_features: &Tensor,
        timestamps: &[f32],
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let num_edges = timestamps.len();

        // Encode timestamps
        let time_enc = self.time_encoder.encode(timestamps)?;
        let time_data = time_enc.to_vec()?;

        // Concatenate features and time encoding
        let src_data = src_features.to_vec()?;
        let dst_data = dst_features.to_vec()?;

        let time_dim = time_enc.shape().dims()[1];
        let mut message_input = Vec::with_capacity(num_edges * (self.in_features * 2 + time_dim));

        for e in 0..num_edges {
            // Source features
            for f in 0..self.in_features {
                message_input.push(src_data[e * self.in_features + f]);
            }
            // Destination features
            for f in 0..self.in_features {
                message_input.push(dst_data[e * self.in_features + f]);
            }
            // Time encoding
            for t in 0..time_dim {
                message_input.push(time_data[e * time_dim + t]);
            }
        }

        let message_tensor = from_vec(
            message_input,
            &[num_edges, self.in_features * 2 + time_dim],
            DeviceType::Cpu,
        )?;

        // Apply MLP
        let messages = message_tensor.matmul(&self.message_weight)?;
        let messages = messages.add(&self.message_bias.unsqueeze(0)?)?;
        Self::relu(&messages)
    }

    /// Compute node embeddings
    ///
    /// # Arguments:
    /// * `features` - Node features
    /// * `memory` - Node memory states
    /// * `current_time` - Current timestamp
    ///
    /// # Returns:
    /// Node embeddings
    pub fn compute_embeddings(
        &self,
        features: &Tensor,
        memory: &Tensor,
        current_time: f32,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let num_nodes = features.shape().dims()[0];

        // Encode current time
        let times = vec![current_time; num_nodes];
        let time_enc = self.time_encoder.encode(&times)?;

        // Concatenate features, memory, and time encoding
        let feat_data = features.to_vec()?;
        let mem_data = memory.to_vec()?;
        let time_data = time_enc.to_vec()?;
        let time_dim = time_enc.shape().dims()[1];

        let mut embed_input = Vec::with_capacity(num_nodes * (self.in_features + self.memory_dim + time_dim));

        for n in 0..num_nodes {
            for f in 0..self.in_features {
                embed_input.push(feat_data[n * self.in_features + f]);
            }
            for m in 0..self.memory_dim {
                embed_input.push(mem_data[n * self.memory_dim + m]);
            }
            for t in 0..time_dim {
                embed_input.push(time_data[n * time_dim + t]);
            }
        }

        let embed_tensor = from_vec(
            embed_input,
            &[num_nodes, self.in_features + self.memory_dim + time_dim],
            DeviceType::Cpu,
        )?;

        // Apply MLP
        let embeddings = embed_tensor.matmul(&self.embedding_weight)?;
        let embeddings = embeddings.add(&self.embedding_bias.unsqueeze(0)?)?;
        Self::relu(&embeddings)
    }

    /// ReLU activation
    fn relu(x: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let data = x.to_vec()?;
        let result: Vec<f32> = data.iter().map(|&v| v.max(0.0)).collect();
        from_vec(result, x.shape().dims(), DeviceType::Cpu)
    }

    /// Get parameters
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![
            self.message_weight.clone(),
            self.message_bias.clone(),
            self.embedding_weight.clone(),
            self.embedding_bias.clone(),
        ];
        params.extend(self.time_encoder.parameters());
        params
    }
}

/// Neural ODE layer for continuous graph dynamics
///
/// Models graph evolution as a continuous-time dynamical system using
/// neural ordinary differential equations.
#[derive(Debug, Clone)]
pub struct GraphNeuralODE {
    /// Feature dimension
    feature_dim: usize,
    /// ODE function network
    ode_weight1: Tensor,
    ode_bias1: Tensor,
    ode_weight2: Tensor,
    ode_bias2: Tensor,
    /// Integration method
    solver: ODESolver,
}

/// ODE solver type
#[derive(Debug, Clone, Copy)]
pub enum ODESolver {
    /// Euler method (simple, first-order)
    Euler { step_size: f32 },
    /// Runge-Kutta 4th order (more accurate)
    RK4 { step_size: f32 },
}

impl GraphNeuralODE {
    /// Create a new Graph Neural ODE layer
    ///
    /// # Arguments:
    /// * `feature_dim` - Dimension of node features
    /// * `hidden_dim` - Hidden dimension for ODE function
    /// * `solver` - ODE solver configuration
    pub fn new(
        feature_dim: usize,
        hidden_dim: usize,
        solver: ODESolver,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 0.1)?;

        let ode_weight1 = TGNLayer::init_weight(feature_dim, hidden_dim, &normal, &mut rng)?;
        let ode_bias1 = zeros(&[hidden_dim], DeviceType::Cpu)?;
        let ode_weight2 = TGNLayer::init_weight(hidden_dim, feature_dim, &normal, &mut rng)?;
        let ode_bias2 = zeros(&[feature_dim], DeviceType::Cpu)?;

        Ok(Self {
            feature_dim,
            ode_weight1,
            ode_bias1,
            ode_weight2,
            ode_bias2,
            solver,
        })
    }

    /// Compute derivative dh/dt = f(h, t)
    ///
    /// The ODE function is implemented as a 2-layer MLP
    fn ode_func(&self, h: &Tensor, _t: f32) -> Result<Tensor, Box<dyn std::error::Error>> {
        // Layer 1
        let hidden = h.matmul(&self.ode_weight1)?;
        let hidden = hidden.add(&self.ode_bias1.unsqueeze(0)?)?;
        let hidden = TGNLayer::relu(&hidden)?;

        // Layer 2
        let output = hidden.matmul(&self.ode_weight2)?;
        let output = output.add(&self.ode_bias2.unsqueeze(0)?)?;

        Ok(output)
    }

    /// Integrate from time t0 to t1
    ///
    /// # Arguments:
    /// * `h0` - Initial state at time t0
    /// * `t0` - Start time
    /// * `t1` - End time
    ///
    /// # Returns:
    /// State at time t1
    pub fn integrate(
        &self,
        h0: &Tensor,
        t0: f32,
        t1: f32,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        match self.solver {
            ODESolver::Euler { step_size } => {
                self.euler_integrate(h0, t0, t1, step_size)
            }
            ODESolver::RK4 { step_size } => {
                self.rk4_integrate(h0, t0, t1, step_size)
            }
        }
    }

    /// Euler method integration
    fn euler_integrate(
        &self,
        h0: &Tensor,
        t0: f32,
        t1: f32,
        step_size: f32,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut h = h0.clone();
        let mut t = t0;

        while t < t1 {
            let dt = step_size.min(t1 - t);
            let dh = self.ode_func(&h, t)?;
            h = h.add(&dh.mul_scalar(dt)?)?;
            t += dt;
        }

        Ok(h)
    }

    /// Runge-Kutta 4th order integration
    fn rk4_integrate(
        &self,
        h0: &Tensor,
        t0: f32,
        t1: f32,
        step_size: f32,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut h = h0.clone();
        let mut t = t0;

        while t < t1 {
            let dt = step_size.min(t1 - t);

            // k1 = f(h, t)
            let k1 = self.ode_func(&h, t)?;

            // k2 = f(h + dt/2 * k1, t + dt/2)
            let h2 = h.add(&k1.mul_scalar(dt / 2.0)?)?;
            let k2 = self.ode_func(&h2, t + dt / 2.0)?;

            // k3 = f(h + dt/2 * k2, t + dt/2)
            let h3 = h.add(&k2.mul_scalar(dt / 2.0)?)?;
            let k3 = self.ode_func(&h3, t + dt / 2.0)?;

            // k4 = f(h + dt * k3, t + dt)
            let h4 = h.add(&k3.mul_scalar(dt)?)?;
            let k4 = self.ode_func(&h4, t + dt)?;

            // h(t+dt) = h(t) + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            let update = k1
                .add(&k2.mul_scalar(2.0)?)?
                .add(&k3.mul_scalar(2.0)?)?
                .add(&k4)?
                .mul_scalar(dt / 6.0)?;

            h = h.add(&update)?;
            t += dt;
        }

        Ok(h)
    }

    /// Get parameters
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![
            self.ode_weight1.clone(),
            self.ode_bias1.clone(),
            self.ode_weight2.clone(),
            self.ode_bias2.clone(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_encoder_creation() {
        let encoder = TimeEncoder::new(32, 1.0);
        assert!(encoder.is_ok());
        let encoder = encoder.unwrap();
        assert_eq!(encoder.dim, 32);
    }

    #[test]
    fn test_time_encoder_encode() {
        let encoder = TimeEncoder::new(16, 1.0).unwrap();
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let encoded = encoder.encode(&times);

        assert!(encoded.is_ok());
        let encoded = encoded.unwrap();
        assert_eq!(encoded.shape().dims(), &[4, 16]);
    }

    #[test]
    fn test_node_memory_creation() {
        let memory = NodeMemory::new(10, 64, MemoryUpdateType::GRU);
        assert!(memory.is_ok());
        let memory = memory.unwrap();
        assert_eq!(memory.num_nodes, 10);
        assert_eq!(memory.memory_dim, 64);
    }

    #[test]
    fn test_node_memory_get() {
        let memory = NodeMemory::new(10, 8, MemoryUpdateType::GRU).unwrap();
        let node_ids = vec![0, 1, 2];
        let result = memory.get_memory(&node_ids);

        assert!(result.is_ok());
        let mem = result.unwrap();
        assert_eq!(mem.shape().dims(), &[3, 8]);
    }

    #[test]
    fn test_node_memory_update() {
        let mut memory = NodeMemory::new(5, 4, MemoryUpdateType::MovingAverage { alpha: 0.5 }).unwrap();

        let node_ids = vec![0, 1];
        let messages = from_vec(vec![1.0; 2 * 4], &[2, 4], DeviceType::Cpu).unwrap();
        let timestamps = vec![1.0, 1.5];

        let result = memory.update_memory(&node_ids, &messages, &timestamps);
        assert!(result.is_ok());

        assert_eq!(memory.last_update[0], 1.0);
        assert_eq!(memory.last_update[1], 1.5);
    }

    #[test]
    fn test_tgn_layer_creation() {
        let layer = TGNLayer::new(32, 64, 128, 16);
        assert!(layer.is_ok());
        let layer = layer.unwrap();
        assert_eq!(layer.in_features, 32);
        assert_eq!(layer.out_features, 64);
        assert_eq!(layer.memory_dim, 128);
    }

    #[test]
    fn test_tgn_compute_messages() {
        let layer = TGNLayer::new(8, 16, 32, 8).unwrap();

        let src_features = from_vec(vec![1.0; 3 * 8], &[3, 8], DeviceType::Cpu).unwrap();
        let dst_features = from_vec(vec![0.5; 3 * 8], &[3, 8], DeviceType::Cpu).unwrap();
        let timestamps = vec![0.0, 1.0, 2.0];

        let messages = layer.compute_messages(&src_features, &dst_features, &timestamps);
        assert!(messages.is_ok());
        let messages = messages.unwrap();
        assert_eq!(messages.shape().dims()[0], 3);
        assert_eq!(messages.shape().dims()[1], 32);
    }

    #[test]
    fn test_tgn_compute_embeddings() {
        let layer = TGNLayer::new(8, 16, 32, 8).unwrap();

        let features = from_vec(vec![1.0; 5 * 8], &[5, 8], DeviceType::Cpu).unwrap();
        let memory = from_vec(vec![0.5; 5 * 32], &[5, 32], DeviceType::Cpu).unwrap();

        let embeddings = layer.compute_embeddings(&features, &memory, 1.5);
        assert!(embeddings.is_ok());
        let embeddings = embeddings.unwrap();
        assert_eq!(embeddings.shape().dims()[0], 5);
        assert_eq!(embeddings.shape().dims()[1], 16);
    }

    #[test]
    fn test_graph_neural_ode_creation() {
        let ode = GraphNeuralODE::new(32, 64, ODESolver::Euler { step_size: 0.1 });
        assert!(ode.is_ok());
    }

    #[test]
    fn test_neural_ode_integration_euler() {
        let ode = GraphNeuralODE::new(8, 16, ODESolver::Euler { step_size: 0.1 }).unwrap();
        let h0 = from_vec(vec![1.0; 4 * 8], &[4, 8], DeviceType::Cpu).unwrap();

        let h1 = ode.integrate(&h0, 0.0, 1.0);
        assert!(h1.is_ok());
        let h1 = h1.unwrap();
        assert_eq!(h1.shape().dims(), &[4, 8]);
    }

    #[test]
    fn test_neural_ode_integration_rk4() {
        let ode = GraphNeuralODE::new(8, 16, ODESolver::RK4 { step_size: 0.1 }).unwrap();
        let h0 = from_vec(vec![1.0; 4 * 8], &[4, 8], DeviceType::Cpu).unwrap();

        let h1 = ode.integrate(&h0, 0.0, 0.5);
        assert!(h1.is_ok());
        let h1 = h1.unwrap();
        assert_eq!(h1.shape().dims(), &[4, 8]);
    }

    #[test]
    fn test_memory_update_types() {
        let update_types = vec![
            MemoryUpdateType::GRU,
            MemoryUpdateType::RNN,
            MemoryUpdateType::MovingAverage { alpha: 0.9 },
        ];

        for update_type in update_types {
            let mut memory = NodeMemory::new(10, 16, update_type).unwrap();
            let node_ids = vec![0, 1, 2];
            let messages = from_vec(vec![1.0; 3 * 16], &[3, 16], DeviceType::Cpu).unwrap();
            let timestamps = vec![1.0, 2.0, 3.0];

            let result = memory.update_memory(&node_ids, &messages, &timestamps);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_memory_reset() {
        let mut memory = NodeMemory::new(10, 16, MemoryUpdateType::GRU).unwrap();

        // Update some memory
        let node_ids = vec![0, 1];
        let messages = from_vec(vec![1.0; 2 * 16], &[2, 16], DeviceType::Cpu).unwrap();
        let timestamps = vec![1.0, 2.0];
        memory.update_memory(&node_ids, &messages, &timestamps).unwrap();

        // Reset
        let result = memory.reset();
        assert!(result.is_ok());

        // Check that last_update is reset
        assert_eq!(memory.last_update[0], 0.0);
        assert_eq!(memory.last_update[1], 0.0);
    }
}
