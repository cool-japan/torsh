//! Graph Neural Network implementations for ToRSh
//!
//! This module provides various graph neural network architectures including
//! Graph Convolutional Networks (GCN), GraphSAGE, Graph Attention Networks (GAT),
//! and Graph Isomorphism Networks (GIN).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::{DType, DeviceType};
use torsh_nn::prelude::{Dropout, Linear};
use torsh_nn::{Module, Parameter};
use torsh_tensor::{stats::StatMode, Tensor};

/// Configuration for Graph Convolutional Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCNConfig {
    pub input_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub output_dim: usize,
    pub num_layers: usize,
    pub dropout_rate: f64,
    pub activation: String,
    pub bias: bool,
    pub normalize: bool,
}

impl Default for GCNConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            hidden_dims: vec![128, 128],
            output_dim: 10,
            num_layers: 2,
            dropout_rate: 0.5,
            activation: "relu".to_string(),
            bias: true,
            normalize: true,
        }
    }
}

/// Graph Convolutional Network layer
#[derive(Debug)]
pub struct GCNLayer {
    linear: Linear,
    dropout: Dropout,
    use_bias: bool,
    normalize: bool,
}

impl GCNLayer {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        dropout_rate: f64,
        bias: bool,
        normalize: bool,
    ) -> Self {
        Self {
            linear: Linear::new(input_dim, output_dim, bias),
            dropout: Dropout::new(dropout_rate as f32),
            use_bias: bias,
            normalize,
        }
    }

    pub fn forward(&self, x: &Tensor, adjacency: &Tensor) -> torsh_core::error::Result<Tensor> {
        // Graph convolution: A * X * W
        let ax = adjacency.matmul(x)?;
        let mut output = self.linear.forward(&ax)?;

        if self.normalize {
            output = self.normalize_features(&output)?;
        }

        self.dropout.forward(&output)
    }

    fn normalize_features(&self, x: &Tensor) -> torsh_core::error::Result<Tensor> {
        let mean = x.mean(Some(&[1]), true)?;
        let std = x.std(Some(&[1]), true, StatMode::Sample)?;
        let eps = torsh_tensor::creation::full(&[1], 1e-8)?;
        let std_eps = std.add(&eps)?;
        x.sub(&mean)?.div(&std_eps)
    }
}

impl Module for GCNLayer {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        let adjacency = &input;
        let x = &input;
        self.forward(x, adjacency)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = self.linear.parameters();
        params.extend(self.dropout.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.linear.named_parameters() {
            params.insert(format!("linear.{}", name), param);
        }
        for (name, param) in self.dropout.named_parameters() {
            params.insert(format!("dropout.{}", name), param);
        }
        params
    }

    fn training(&self) -> bool {
        self.dropout.training()
    }

    fn train(&mut self) {
        self.linear.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.linear.eval();
        self.dropout.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        self.linear.to_device(device)?;
        self.dropout.to_device(device)
    }
}

/// Graph Convolutional Network
#[derive(Debug)]
pub struct GCN {
    layers: Vec<GCNLayer>,
    config: GCNConfig,
}

impl GCN {
    pub fn new(config: GCNConfig) -> Self {
        let mut layers = Vec::new();
        let mut input_dim = config.input_dim;

        for &hidden_dim in &config.hidden_dims {
            layers.push(GCNLayer::new(
                input_dim,
                hidden_dim,
                config.dropout_rate,
                config.bias,
                config.normalize,
            ));
            input_dim = hidden_dim;
        }

        // Output layer
        layers.push(GCNLayer::new(
            input_dim,
            config.output_dim,
            0.0, // No dropout in output layer
            config.bias,
            false, // No normalization in output layer
        ));

        Self { layers, config }
    }

    pub fn forward_with_adjacency(
        &self,
        x: &Tensor,
        adjacency: &Tensor,
    ) -> torsh_core::error::Result<Tensor> {
        let mut output = x.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            output = layer.forward(&output, adjacency)?;

            // Apply activation except for the last layer
            if i < self.layers.len() - 1 {
                output = match self.config.activation.as_str() {
                    "relu" => output.relu()?,
                    "gelu" => output.gelu()?,
                    "tanh" => output.tanh()?,
                    "sigmoid" => output.sigmoid()?,
                    _ => output,
                };
            }
        }

        Ok(output)
    }
}

impl Module for GCN {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        let adjacency = input;
        let x = input;
        self.forward_with_adjacency(x, adjacency)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("layer_{}.{}", i, name), param);
            }
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.named_parameters() {
                params.insert(format!("layer_{}.{}", i, name), param);
            }
        }
        params
    }

    fn training(&self) -> bool {
        self.layers.iter().any(|layer| layer.training())
    }

    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

/// Configuration for GraphSAGE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSAGEConfig {
    pub input_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub output_dim: usize,
    pub aggregator_type: String, // mean, max, lstm, pool
    pub dropout_rate: f64,
    pub normalize: bool,
    pub concat: bool,
}

impl Default for GraphSAGEConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            hidden_dims: vec![128, 128],
            output_dim: 10,
            aggregator_type: "mean".to_string(),
            dropout_rate: 0.5,
            normalize: true,
            concat: true,
        }
    }
}

/// GraphSAGE layer
#[derive(Debug)]
pub struct GraphSAGELayer {
    self_linear: Linear,
    neighbor_linear: Linear,
    dropout: Dropout,
    aggregator_type: String,
    normalize: bool,
    concat: bool,
}

impl GraphSAGELayer {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        aggregator_type: String,
        dropout_rate: f64,
        normalize: bool,
        concat: bool,
    ) -> Self {
        let self_dim = if concat { output_dim / 2 } else { output_dim };
        let neighbor_dim = if concat { output_dim / 2 } else { output_dim };

        Self {
            self_linear: Linear::new(input_dim, self_dim, true),
            neighbor_linear: Linear::new(input_dim, neighbor_dim, true),
            dropout: Dropout::new(dropout_rate as f32),
            aggregator_type,
            normalize,
            concat,
        }
    }

    pub fn forward(&self, x: &Tensor, adjacency: &Tensor) -> torsh_core::error::Result<Tensor> {
        // Self features
        let self_features = self.self_linear.forward(x)?;

        // Aggregate neighbor features
        let neighbor_features = self.aggregate_neighbors(x, adjacency)?;
        let neighbor_features = self.neighbor_linear.forward(&neighbor_features)?;

        // Combine self and neighbor features
        let output = if self.concat {
            Tensor::cat(&[&self_features, &neighbor_features], 1)?
        } else {
            self_features.add(&neighbor_features)?
        };

        let output = if self.normalize {
            self.l2_normalize(&output)?
        } else {
            output
        };

        self.dropout.forward(&output)
    }

    fn aggregate_neighbors(
        &self,
        x: &Tensor,
        adjacency: &Tensor,
    ) -> torsh_core::error::Result<Tensor> {
        match self.aggregator_type.as_str() {
            "mean" => adjacency.matmul(x),
            "max" => {
                // For simplicity, use mean aggregation
                adjacency.matmul(x)
            }
            _ => adjacency.matmul(x),
        }
    }

    fn l2_normalize(&self, x: &Tensor) -> torsh_core::error::Result<Tensor> {
        let norm = x.norm()?;
        let eps = torsh_tensor::creation::full(&[1], 1e-8)?;
        let norm_eps = norm.add(&eps)?;
        x.div(&norm_eps)
    }
}

impl Module for GraphSAGELayer {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        let adjacency = input;
        let x = input;
        self.forward(x, adjacency)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = self.self_linear.parameters();
        params.extend(self.neighbor_linear.parameters());
        params.extend(self.dropout.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.self_linear.named_parameters() {
            params.insert(format!("self_linear.{}", name), param);
        }
        for (name, param) in self.neighbor_linear.named_parameters() {
            params.insert(format!("neighbor_linear.{}", name), param);
        }
        for (name, param) in self.dropout.named_parameters() {
            params.insert(format!("dropout.{}", name), param);
        }
        params
    }

    fn training(&self) -> bool {
        self.dropout.training()
    }

    fn train(&mut self) {
        self.self_linear.train();
        self.neighbor_linear.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.self_linear.eval();
        self.neighbor_linear.eval();
        self.dropout.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        self.self_linear.to_device(device)?;
        self.neighbor_linear.to_device(device)?;
        self.dropout.to_device(device)
    }
}

/// GraphSAGE model
#[derive(Debug)]
pub struct GraphSAGE {
    layers: Vec<GraphSAGELayer>,
    config: GraphSAGEConfig,
}

impl GraphSAGE {
    pub fn new(config: GraphSAGEConfig) -> Self {
        let mut layers = Vec::new();
        let mut input_dim = config.input_dim;

        for &hidden_dim in &config.hidden_dims {
            layers.push(GraphSAGELayer::new(
                input_dim,
                hidden_dim,
                config.aggregator_type.clone(),
                config.dropout_rate,
                config.normalize,
                config.concat,
            ));
            input_dim = hidden_dim;
        }

        // Output layer
        layers.push(GraphSAGELayer::new(
            input_dim,
            config.output_dim,
            config.aggregator_type.clone(),
            0.0,
            false,
            false,
        ));

        Self { layers, config }
    }

    pub fn forward_with_adjacency(
        &self,
        x: &Tensor,
        adjacency: &Tensor,
    ) -> torsh_core::error::Result<Tensor> {
        let mut output = x.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            output = layer.forward(&output, adjacency)?;

            // Apply ReLU activation except for the last layer
            if i < self.layers.len() - 1 {
                output = output.relu()?;
            }
        }

        Ok(output)
    }
}

impl Module for GraphSAGE {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        let adjacency = input;
        let x = input;
        self.forward_with_adjacency(x, adjacency)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("layer_{}.{}", i, name), param);
            }
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.named_parameters() {
                params.insert(format!("layer_{}.{}", i, name), param);
            }
        }
        params
    }

    fn training(&self) -> bool {
        self.layers.iter().any(|layer| layer.training())
    }

    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

/// Configuration for Graph Attention Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GATConfig {
    pub input_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub output_dim: usize,
    pub num_heads: usize,
    pub dropout_rate: f64,
    pub alpha: f64, // LeakyReLU negative slope
    pub concat_heads: bool,
}

impl Default for GATConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            hidden_dims: vec![128],
            output_dim: 10,
            num_heads: 8,
            dropout_rate: 0.6,
            alpha: 0.2,
            concat_heads: true,
        }
    }
}

/// Graph Attention Network layer
#[derive(Debug)]
pub struct GATLayer {
    linear: Linear,
    attention_weights: Parameter,
    dropout: Dropout,
    num_heads: usize,
    head_dim: usize,
    alpha: f64,
    concat_heads: bool,
}

impl GATLayer {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        num_heads: usize,
        dropout_rate: f64,
        alpha: f64,
        concat_heads: bool,
    ) -> Self {
        let head_dim = if concat_heads {
            output_dim / num_heads
        } else {
            output_dim
        };

        let linear_output_dim = head_dim * num_heads;

        Self {
            linear: Linear::new(input_dim, linear_output_dim, false),
            attention_weights: Parameter::new(
                torsh_tensor::creation::randn(&[2 * head_dim, 1]).unwrap(),
            ),
            dropout: Dropout::new(dropout_rate as f32),
            num_heads,
            head_dim,
            alpha,
            concat_heads,
        }
    }

    pub fn forward(&self, x: &Tensor, adjacency: &Tensor) -> torsh_core::error::Result<Tensor> {
        let batch_size = x.size(0)?;
        let num_nodes = x.size(1)?;

        // Linear transformation
        let h = self.linear.forward(x)?;

        // Reshape for multi-head attention
        let h = h.view(&[
            batch_size as i32,
            num_nodes as i32,
            self.num_heads as i32,
            self.head_dim as i32,
        ])?;

        // Compute attention coefficients
        let attention_scores = self.compute_attention(&h)?;

        // Apply adjacency mask (only attend to connected nodes)
        let masked_attention = self.apply_adjacency_mask(&attention_scores, adjacency)?;

        // Apply softmax
        let attention_weights = masked_attention.softmax(-1)?;

        // Apply dropout to attention weights
        let attention_weights = self.dropout.forward(&attention_weights)?;

        // Apply attention to features
        let output = attention_weights.matmul(&h)?;

        // Concatenate or average heads
        let output = if self.concat_heads {
            output.view(&[batch_size as i32, num_nodes as i32, -1])?
        } else {
            output.mean(Some(&[2]), false)?
        };

        Ok(output)
    }

    fn compute_attention(&self, h: &Tensor) -> torsh_core::error::Result<Tensor> {
        let batch_size = h.size(0)?;
        let num_nodes = h.size(1)?;
        let num_heads = h.size(2)?;

        // Compute pairwise attention scores
        let h_i = h.unsqueeze(3)?; // [B, N, H, 1, D]
        let h_j = h.unsqueeze(2)?; // [B, N, 1, H, D]
        let h_i = h_i.expand(&[batch_size, num_nodes, num_heads, num_nodes, self.head_dim])?;
        let h_j = h_j.expand(&[batch_size, num_nodes, num_heads, num_nodes, self.head_dim])?;

        // Concatenate h_i and h_j
        let concat_h = Tensor::cat(&[&h_i, &h_j], -1)?;

        // Apply attention weights
        let attention_weights_tensor = self.attention_weights.tensor();
        let attention_weights_guard = attention_weights_tensor.read();
        let attention_scores = concat_h.matmul(&*attention_weights_guard)?;

        // Apply LeakyReLU
        attention_scores.leaky_relu(self.alpha as f32)
    }

    fn apply_adjacency_mask(
        &self,
        attention: &Tensor,
        adjacency: &Tensor,
    ) -> torsh_core::error::Result<Tensor> {
        // Expand adjacency matrix to match attention dimensions
        let mask = adjacency.unsqueeze(2)?; // Add head dimension
        let neg_inf = torsh_tensor::creation::full(&[1], f32::NEG_INFINITY)?;
        // Implement conditional selection: where mask is true, use attention; where false, use neg_inf
        mask.mul(&attention)?
            .add(&mask.mul_scalar(-1.0)?.add_scalar(1.0)?.mul(&neg_inf)?)
    }
}

impl Module for GATLayer {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        let adjacency = input;
        let x = input;
        self.forward(x, adjacency)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = self.linear.parameters();
        params.insert(
            "attention_weights".to_string(),
            self.attention_weights.clone(),
        );
        params.extend(self.dropout.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.linear.named_parameters() {
            params.insert(format!("linear.{}", name), param);
        }
        params.insert(
            "attention_weights".to_string(),
            self.attention_weights.clone(),
        );
        for (name, param) in self.dropout.named_parameters() {
            params.insert(format!("dropout.{}", name), param);
        }
        params
    }

    fn training(&self) -> bool {
        self.dropout.training()
    }

    fn train(&mut self) {
        self.linear.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.linear.eval();
        self.dropout.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        self.linear.to_device(device)?;
        self.attention_weights.to_device(device)?;
        self.dropout.to_device(device)
    }
}

/// Graph Attention Network
#[derive(Debug)]
pub struct GAT {
    layers: Vec<GATLayer>,
    config: GATConfig,
}

impl GAT {
    pub fn new(config: GATConfig) -> Self {
        let mut layers = Vec::new();
        let mut input_dim = config.input_dim;

        for &hidden_dim in &config.hidden_dims {
            layers.push(GATLayer::new(
                input_dim,
                hidden_dim,
                config.num_heads,
                config.dropout_rate,
                config.alpha,
                config.concat_heads,
            ));
            input_dim = hidden_dim;
        }

        // Output layer (usually single head)
        layers.push(GATLayer::new(
            input_dim,
            config.output_dim,
            1,
            0.0,
            config.alpha,
            false,
        ));

        Self { layers, config }
    }

    pub fn forward_with_adjacency(
        &self,
        x: &Tensor,
        adjacency: &Tensor,
    ) -> torsh_core::error::Result<Tensor> {
        let mut output = x.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            output = layer.forward(&output, adjacency)?;

            // Apply ELU activation except for the last layer
            if i < self.layers.len() - 1 {
                output = output.relu()?;
            }
        }

        Ok(output)
    }
}

impl Module for GAT {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        let adjacency = input;
        let x = input;
        self.forward_with_adjacency(x, adjacency)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("layer_{}.{}", i, name), param);
            }
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.named_parameters() {
                params.insert(format!("layer_{}.{}", i, name), param);
            }
        }
        params
    }

    fn training(&self) -> bool {
        self.layers.iter().any(|layer| layer.training())
    }

    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

/// Configuration for Graph Isomorphism Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GINConfig {
    pub input_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub output_dim: usize,
    pub num_layers: usize,
    pub eps: f64,
    pub learn_eps: bool,
    pub dropout_rate: f64,
}

impl Default for GINConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            hidden_dims: vec![128, 128],
            output_dim: 10,
            num_layers: 2,
            eps: 0.0,
            learn_eps: false,
            dropout_rate: 0.5,
        }
    }
}

/// Graph Isomorphism Network layer
#[derive(Debug)]
pub struct GINLayer {
    mlp: Vec<Linear>,
    eps: Parameter,
    dropout: Dropout,
    learn_eps: bool,
}

impl GINLayer {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        eps: f64,
        learn_eps: bool,
        dropout_rate: f64,
    ) -> Self {
        let mlp = vec![
            Linear::new(input_dim, output_dim, true),
            Linear::new(output_dim, output_dim, true),
        ];

        let eps_param = if learn_eps {
            Parameter::new(torsh_tensor::creation::full(&[1], eps as f32).unwrap())
        } else {
            Parameter::new(torsh_tensor::creation::full(&[1], eps as f32).unwrap())
        };

        Self {
            mlp,
            eps: eps_param,
            dropout: Dropout::new(dropout_rate as f32),
            learn_eps,
        }
    }

    pub fn forward(&self, x: &Tensor, adjacency: &Tensor) -> torsh_core::error::Result<Tensor> {
        // Aggregate neighbor features
        let neighbor_sum = adjacency.matmul(x)?;

        // Add self-loop with learnable epsilon
        let one_tensor = Tensor::ones(&[1], x.device())?;
        let eps_tensor = self.eps.tensor();
        let eps_guard = eps_tensor.read();
        let eps_plus_one = eps_guard.add(&one_tensor)?;
        let self_contribution = x.mul(&eps_plus_one)?;

        // Combine self and neighbor contributions
        let combined = self_contribution.add(&neighbor_sum)?;

        // Apply MLP
        let mut output = combined;
        for (i, linear) in self.mlp.iter().enumerate() {
            output = linear.forward(&output)?;
            if i < self.mlp.len() - 1 {
                output = output.relu()?;
                output = self.dropout.forward(&output)?;
            }
        }

        Ok(output)
    }
}

impl Module for GINLayer {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        let adjacency = input;
        let x = input;
        self.forward(x, adjacency)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, linear) in self.mlp.iter().enumerate() {
            for (name, param) in linear.parameters() {
                params.insert(format!("mlp_{}.{}", i, name), param);
            }
        }
        if self.learn_eps {
            params.insert("eps".to_string(), self.eps.clone());
        }
        params.extend(self.dropout.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, linear) in self.mlp.iter().enumerate() {
            for (name, param) in linear.named_parameters() {
                params.insert(format!("mlp_{}.{}", i, name), param);
            }
        }
        if self.learn_eps {
            params.insert("eps".to_string(), self.eps.clone());
        }
        for (name, param) in self.dropout.named_parameters() {
            params.insert(format!("dropout.{}", name), param);
        }
        params
    }

    fn training(&self) -> bool {
        self.dropout.training()
    }

    fn train(&mut self) {
        for linear in &mut self.mlp {
            linear.train();
        }
        self.dropout.train();
    }

    fn eval(&mut self) {
        for linear in &mut self.mlp {
            linear.eval();
        }
        self.dropout.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        for linear in &mut self.mlp {
            linear.to_device(device)?;
        }
        self.eps.to_device(device)?;
        self.dropout.to_device(device)
    }
}

/// Graph Isomorphism Network
#[derive(Debug)]
pub struct GIN {
    layers: Vec<GINLayer>,
    config: GINConfig,
}

impl GIN {
    pub fn new(config: GINConfig) -> Self {
        let mut layers = Vec::new();
        let mut input_dim = config.input_dim;

        for &hidden_dim in &config.hidden_dims {
            layers.push(GINLayer::new(
                input_dim,
                hidden_dim,
                config.eps,
                config.learn_eps,
                config.dropout_rate,
            ));
            input_dim = hidden_dim;
        }

        // Output layer
        layers.push(GINLayer::new(
            input_dim,
            config.output_dim,
            config.eps,
            config.learn_eps,
            0.0,
        ));

        Self { layers, config }
    }

    pub fn forward_with_adjacency(
        &self,
        x: &Tensor,
        adjacency: &Tensor,
    ) -> torsh_core::error::Result<Tensor> {
        let mut output = x.clone();

        for layer in &self.layers {
            output = layer.forward(&output, adjacency)?;
        }

        Ok(output)
    }
}

impl Module for GIN {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        let adjacency = input;
        let x = input;
        self.forward_with_adjacency(x, adjacency)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("layer_{}.{}", i, name), param);
            }
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.named_parameters() {
                params.insert(format!("layer_{}.{}", i, name), param);
            }
        }
        params
    }

    fn training(&self) -> bool {
        self.layers.iter().any(|layer| layer.training())
    }

    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

/// GNN Architecture enum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GNNArchitecture {
    GCN,
    GraphSAGE,
    GAT,
    GIN,
}

/// Unified GNN model type
#[derive(Debug)]
pub enum GNNModel {
    GCN(GCN),
    GraphSAGE(GraphSAGE),
    GAT(GAT),
    GIN(GIN),
}

impl Module for GNNModel {
    fn forward(&self, input: &Tensor) -> torsh_core::error::Result<Tensor> {
        match self {
            GNNModel::GCN(model) => model.forward(input),
            GNNModel::GraphSAGE(model) => model.forward(input),
            GNNModel::GAT(model) => model.forward(input),
            GNNModel::GIN(model) => model.forward(input),
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        match self {
            GNNModel::GCN(model) => model.parameters(),
            GNNModel::GraphSAGE(model) => model.parameters(),
            GNNModel::GAT(model) => model.parameters(),
            GNNModel::GIN(model) => model.parameters(),
        }
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        match self {
            GNNModel::GCN(model) => model.named_parameters(),
            GNNModel::GraphSAGE(model) => model.named_parameters(),
            GNNModel::GAT(model) => model.named_parameters(),
            GNNModel::GIN(model) => model.named_parameters(),
        }
    }

    fn training(&self) -> bool {
        match self {
            GNNModel::GCN(model) => model.training(),
            GNNModel::GraphSAGE(model) => model.training(),
            GNNModel::GAT(model) => model.training(),
            GNNModel::GIN(model) => model.training(),
        }
    }

    fn train(&mut self) {
        match self {
            GNNModel::GCN(model) => model.train(),
            GNNModel::GraphSAGE(model) => model.train(),
            GNNModel::GAT(model) => model.train(),
            GNNModel::GIN(model) => model.train(),
        }
    }

    fn eval(&mut self) {
        match self {
            GNNModel::GCN(model) => model.eval(),
            GNNModel::GraphSAGE(model) => model.eval(),
            GNNModel::GAT(model) => model.eval(),
            GNNModel::GIN(model) => model.eval(),
        }
    }

    fn to_device(&mut self, device: DeviceType) -> torsh_core::error::Result<()> {
        match self {
            GNNModel::GCN(model) => model.to_device(device),
            GNNModel::GraphSAGE(model) => model.to_device(device),
            GNNModel::GAT(model) => model.to_device(device),
            GNNModel::GIN(model) => model.to_device(device),
        }
    }
}
