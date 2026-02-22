//! CLIP transformer layers

use super::super::multimodal_common::activations::QuickGELU;
use super::config::{CLIPTextConfig, CLIPVisionConfig};
use std::collections::HashMap;
use torsh_core::{error::Result, DeviceType};
use torsh_nn::prelude::*;
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

/// CLIP MLP layer
pub struct CLIPMLP {
    fc1: Linear,
    fc2: Linear,
    activation: QuickGELU,
    dropout: Dropout,
}

impl CLIPMLP {
    pub fn new(config: &CLIPVisionConfig) -> Self {
        let fc1 = Linear::new(config.hidden_size, config.intermediate_size, true);
        let fc2 = Linear::new(config.intermediate_size, config.hidden_size, true);
        let activation = QuickGELU::new();
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Self {
            fc1,
            fc2,
            activation,
            dropout,
        }
    }

    pub fn new_for_text(config: &CLIPTextConfig) -> Self {
        let fc1 = Linear::new(config.hidden_size, config.intermediate_size, true);
        let fc2 = Linear::new(config.intermediate_size, config.hidden_size, true);
        let activation = QuickGELU::new();
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Self {
            fc1,
            fc2,
            activation,
            dropout,
        }
    }
}

impl Module for CLIPMLP {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.fc1.forward(hidden_states)?;
        let hidden_states = self.activation.forward(&hidden_states)?;
        let hidden_states = self.fc2.forward(&hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        Ok(hidden_states)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.fc1.parameters() {
            params.insert(format!("fc1.{}", name), param);
        }
        for (name, param) in self.fc2.parameters() {
            params.insert(format!("fc2.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.dropout.training()
    }

    fn train(&mut self) {
        self.fc1.train();
        self.fc2.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.fc1.eval();
        self.fc2.eval();
        self.dropout.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.fc1.to_device(device)?;
        self.fc2.to_device(device)?;
        Ok(())
    }
}

/// CLIP Attention layer
pub struct CLIPAttention {
    attention: MultiheadAttention,
    scale: f32,
    dropout: Dropout,
}

impl CLIPAttention {
    pub fn new(config: &CLIPVisionConfig) -> Self {
        let attention = MultiheadAttention::new(config.hidden_size, config.num_attention_heads);
        let scale = 1.0 / (config.attention_head_size() as f32).sqrt();
        let dropout = Dropout::new(config.attention_dropout);

        Self {
            attention,
            scale,
            dropout,
        }
    }

    pub fn new_for_text(config: &CLIPTextConfig) -> Self {
        let attention = MultiheadAttention::new(config.hidden_size, config.num_attention_heads);
        let scale = 1.0 / (config.attention_head_size() as f32).sqrt();
        let dropout = Dropout::new(config.attention_dropout);

        Self {
            attention,
            scale,
            dropout,
        }
    }
}

impl Module for CLIPAttention {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states)?;
        self.dropout.forward(&attention_output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.attention.parameters()
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.attention.named_parameters()
    }

    fn training(&self) -> bool {
        self.attention.training() && self.dropout.training()
    }

    fn train(&mut self) {
        self.attention.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.attention.eval();
        self.dropout.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.attention.to_device(device)
    }
}

/// CLIP Encoder Layer
pub struct CLIPEncoderLayer {
    self_attn: CLIPAttention,
    layer_norm1: LayerNorm,
    mlp: CLIPMLP,
    layer_norm2: LayerNorm,
}

impl CLIPEncoderLayer {
    pub fn new(config: CLIPVisionConfig) -> Result<Self> {
        let self_attn = CLIPAttention::new(&config);
        let layer_norm1 = LayerNorm::new(
            vec![config.hidden_size],
            config.layer_norm_eps as f64,
            true,
            torsh_core::DeviceType::Cpu,
        )?;
        let mlp = CLIPMLP::new(&config);
        let layer_norm2 = LayerNorm::new(
            vec![config.hidden_size],
            config.layer_norm_eps as f64,
            true,
            torsh_core::DeviceType::Cpu,
        )?;

        Ok(Self {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    pub fn new_for_text(config: CLIPTextConfig) -> Result<Self> {
        let self_attn = CLIPAttention::new_for_text(&config);
        let layer_norm1 = LayerNorm::new(
            vec![config.hidden_size],
            config.layer_norm_eps as f64,
            true,
            torsh_core::DeviceType::Cpu,
        )?;
        let mlp = CLIPMLP::new_for_text(&config);
        let layer_norm2 = LayerNorm::new(
            vec![config.hidden_size],
            config.layer_norm_eps as f64,
            true,
            torsh_core::DeviceType::Cpu,
        )?;

        Ok(Self {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }
}

impl Module for CLIPEncoderLayer {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Self-attention with residual connection
        let normed_hidden_states = self.layer_norm1.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(&normed_hidden_states)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        // Feed-forward with residual connection
        let normed_hidden_states = self.layer_norm2.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed_hidden_states)?;
        let hidden_states = hidden_states.add(&mlp_output)?;

        Ok(hidden_states)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.self_attn.parameters() {
            params.insert(format!("self_attn.{}", name), param);
        }
        for (name, param) in self.layer_norm1.parameters() {
            params.insert(format!("layer_norm1.{}", name), param);
        }
        for (name, param) in self.mlp.parameters() {
            params.insert(format!("mlp.{}", name), param);
        }
        for (name, param) in self.layer_norm2.parameters() {
            params.insert(format!("layer_norm2.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.self_attn.training()
            && self.layer_norm1.training()
            && self.mlp.training()
            && self.layer_norm2.training()
    }

    fn train(&mut self) {
        self.self_attn.train();
        self.layer_norm1.train();
        self.mlp.train();
        self.layer_norm2.train();
    }

    fn eval(&mut self) {
        self.self_attn.eval();
        self.layer_norm1.eval();
        self.mlp.eval();
        self.layer_norm2.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.self_attn.to_device(device)?;
        self.layer_norm1.to_device(device)?;
        self.mlp.to_device(device)?;
        self.layer_norm2.to_device(device)?;
        Ok(())
    }
}
