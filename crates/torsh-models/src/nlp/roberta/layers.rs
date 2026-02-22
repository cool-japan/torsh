//! RoBERTa transformer layers

use super::{attention::RobertaAttention, config::RobertaConfig};
use torsh_core::error::Result;
use torsh_nn::prelude::*;
use torsh_nn::Module;
use torsh_tensor::Tensor;

/// RoBERTa Intermediate Layer (Feed-Forward Network)
pub struct RobertaIntermediate {
    dense: Linear,
    activation_fn: String, // Store activation function name
}

impl RobertaIntermediate {
    pub fn new(config: &RobertaConfig) -> Self {
        let dense = Linear::new(config.hidden_size, config.intermediate_size, true);
        let activation_fn = match config.hidden_act.as_str() {
            "gelu" => "gelu".to_string(),
            "relu" => "relu".to_string(),
            _ => "gelu".to_string(), // Default to GELU
        };

        Self {
            dense,
            activation_fn,
        }
    }

    fn apply_activation(&self, input: &Tensor) -> Result<Tensor> {
        match self.activation_fn.as_str() {
            "gelu" => {
                // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                let x_cubed = input.pow_scalar(3.0)?;
                let inner = input.add(&x_cubed.mul_scalar(0.044715)?)?;
                let inner = inner.mul_scalar((2.0 / std::f32::consts::PI).sqrt())?;
                let tanh_val = inner.tanh()?;
                let one_plus_tanh = tanh_val.add_scalar(1.0)?;
                input.mul(&one_plus_tanh)?.mul_scalar(0.5)
            }
            "relu" => input.relu(),
            _ => {
                // Default to GELU
                let x_cubed = input.pow_scalar(3.0)?;
                let inner = input.add(&x_cubed.mul_scalar(0.044715)?)?;
                let inner = inner.mul_scalar((2.0 / std::f32::consts::PI).sqrt())?;
                let tanh_val = inner.tanh()?;
                let one_plus_tanh = tanh_val.add_scalar(1.0)?;
                input.mul(&one_plus_tanh)?.mul_scalar(0.5)
            }
        }
    }
}

impl Module for RobertaIntermediate {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        self.apply_activation(&hidden_states)
    }
}

/// RoBERTa Output Layer
pub struct RobertaOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl RobertaOutput {
    pub fn new(config: &RobertaConfig) -> Result<Self> {
        let dense = Linear::new(config.intermediate_size, config.hidden_size, true);
        let layer_norm = LayerNorm::new(
            vec![config.hidden_size],
            config.layer_norm_eps as f64,
            true,
            torsh_core::DeviceType::Cpu,
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Ok(Self {
            dense,
            layer_norm,
            dropout,
        })
    }
}

impl Module for RobertaOutput {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        // Note: residual connection would be added here in full implementation
        let hidden_states = self.layer_norm.forward(&hidden_states)?;
        Ok(hidden_states)
    }
}

/// RoBERTa Transformer Layer
pub struct RobertaLayer {
    attention: RobertaAttention,
    intermediate: RobertaIntermediate,
    output: RobertaOutput,
}

impl RobertaLayer {
    pub fn new(config: RobertaConfig) -> Result<Self> {
        let attention = RobertaAttention::new(config.clone())?;
        let intermediate = RobertaIntermediate::new(&config);
        let output = RobertaOutput::new(&config)?;

        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }
}

impl Module for RobertaLayer {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Self-attention
        let attention_output = self.attention.forward(hidden_states)?;

        // Feed-forward network
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self.output.forward(&intermediate_output)?;

        Ok(layer_output)
    }
}

/// RoBERTa Encoder (stack of transformer layers)
pub struct RobertaEncoder {
    layers: Vec<RobertaLayer>,
}

impl RobertaEncoder {
    pub fn new(config: RobertaConfig) -> Result<Self> {
        let layers: Result<Vec<_>> = (0..config.num_hidden_layers)
            .map(|_| RobertaLayer::new(config.clone()))
            .collect();
        let layers = layers?;

        Ok(Self { layers })
    }
}

impl Module for RobertaEncoder {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut current_states = hidden_states.clone();

        for layer in &self.layers {
            current_states = layer.forward(&current_states)?;
        }

        Ok(current_states)
    }
}
