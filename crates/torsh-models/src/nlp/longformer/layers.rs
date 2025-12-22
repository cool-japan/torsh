//! Longformer Transformer Layers

use crate::nlp::longformer::attention::LongformerSlidingWindowAttention;
use crate::nlp::longformer::config::LongformerConfig;
use std::collections::HashMap;
use torsh_core::error::Result;
use torsh_nn::prelude::*;
use torsh_tensor::Tensor;

/// Longformer transformer layer
pub struct LongformerLayer {
    /// Sliding window attention
    attention: LongformerSlidingWindowAttention,
    /// Intermediate (feed-forward first layer)
    intermediate: Linear,
    /// Output (feed-forward second layer)
    output: Linear,
    /// Layer norms
    attention_layer_norm: LayerNorm,
    output_layer_norm: LayerNorm,
    /// Dropout
    dropout: Dropout,
    /// Configuration
    config: LongformerConfig,
}

impl LongformerLayer {
    pub fn new(config: LongformerConfig, layer_id: usize) -> Result<Self> {
        Ok(Self {
            attention: LongformerSlidingWindowAttention::new(config.clone(), layer_id)?,
            intermediate: Linear::new(config.hidden_size, config.intermediate_size, true),
            output: Linear::new(config.intermediate_size, config.hidden_size, true),
            attention_layer_norm: LayerNorm::new(
                vec![config.hidden_size],
                config.layer_norm_eps as f64,
                true,
                torsh_core::DeviceType::Cpu,
            )?,
            output_layer_norm: LayerNorm::new(
                vec![config.hidden_size],
                config.layer_norm_eps as f64,
                true,
                torsh_core::DeviceType::Cpu,
            )?,
            dropout: Dropout::new(config.hidden_dropout_prob),
            config,
        })
    }

    pub fn config(&self) -> &LongformerConfig {
        &self.config
    }
}

impl Module for LongformerLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Self-attention with residual
        let attention_output = self.attention.forward(input)?;
        let attention_output = self.dropout.forward(&attention_output)?;
        let hidden_states = input.add(&attention_output)?;
        let hidden_states = self.attention_layer_norm.forward(&hidden_states)?;

        // Feed-forward with residual
        let intermediate_output = self.intermediate.forward(&hidden_states)?;
        let intermediate_output = intermediate_output.gelu()?;
        let layer_output = self.output.forward(&intermediate_output)?;
        let layer_output = self.dropout.forward(&layer_output)?;
        let layer_output = hidden_states.add(&layer_output)?;
        self.output_layer_norm.forward(&layer_output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.attention.parameters() {
            params.insert(format!("attention.{}", name), param);
        }
        for (name, param) in self.intermediate.parameters() {
            params.insert(format!("intermediate.{}", name), param);
        }
        for (name, param) in self.output.parameters() {
            params.insert(format!("output.{}", name), param);
        }
        for (name, param) in self.attention_layer_norm.parameters() {
            params.insert(format!("attention_layer_norm.{}", name), param);
        }
        for (name, param) in self.output_layer_norm.parameters() {
            params.insert(format!("output_layer_norm.{}", name), param);
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
        self.attention.train();
        self.intermediate.train();
        self.output.train();
        self.attention_layer_norm.train();
        self.output_layer_norm.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.attention.eval();
        self.intermediate.eval();
        self.output.eval();
        self.attention_layer_norm.eval();
        self.output_layer_norm.eval();
        self.dropout.eval();
    }

    fn to_device(&mut self, device: torsh_core::DeviceType) -> Result<()> {
        self.attention.to_device(device)?;
        self.intermediate.to_device(device)?;
        self.output.to_device(device)?;
        self.attention_layer_norm.to_device(device)?;
        self.output_layer_norm.to_device(device)?;
        self.dropout.to_device(device)?;
        Ok(())
    }
}

/// Longformer encoder
pub struct LongformerEncoder {
    layers: Vec<LongformerLayer>,
    config: LongformerConfig,
}

impl LongformerEncoder {
    pub fn new(config: LongformerConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            layers.push(LongformerLayer::new(config.clone(), i)?);
        }
        Ok(Self { layers, config })
    }

    pub fn config(&self) -> &LongformerConfig {
        &self.config
    }
}

impl Module for LongformerEncoder {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut hidden_states = input.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }
        Ok(hidden_states)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("layer.{}.{}", i, name), param);
            }
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.layers.first().map_or(false, |l| l.training())
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

    fn to_device(&mut self, device: torsh_core::DeviceType) -> Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_longformer_layer_creation() {
        let config = LongformerConfig::longformer_base();
        let layer = LongformerLayer::new(config, 0);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_longformer_encoder_creation() {
        let config = LongformerConfig::longformer_base();
        let encoder = LongformerEncoder::new(config);
        assert!(encoder.is_ok());
    }
}
