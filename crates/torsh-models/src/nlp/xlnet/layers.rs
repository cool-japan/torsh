//! XLNet Transformer Layers
//!
//! This module implements the transformer layers for XLNet models.

use crate::nlp::xlnet::attention::XLNetRelativeAttention;
use crate::nlp::xlnet::config::XLNetConfig;
use std::collections::HashMap;
use torsh_core::error::Result;
use torsh_nn::prelude::*;
use torsh_tensor::Tensor;

/// XLNet feed-forward layer
pub struct XLNetFeedForward {
    /// First linear layer
    dense_1: Linear,
    /// Second linear layer
    dense_2: Linear,
    /// Dropout
    dropout: Dropout,
    /// Layer normalization
    layer_norm: LayerNorm,
    /// Configuration
    config: XLNetConfig,
}

impl XLNetFeedForward {
    /// Create new feed-forward layer
    pub fn new(config: XLNetConfig) -> Result<Self> {
        Ok(Self {
            dense_1: Linear::new(config.hidden_size, config.intermediate_size, true),
            dense_2: Linear::new(config.intermediate_size, config.hidden_size, true),
            dropout: Dropout::new(config.hidden_dropout_prob),
            layer_norm: LayerNorm::new(
                vec![config.hidden_size],
                config.layer_norm_eps as f64,
                true,
                torsh_core::DeviceType::Cpu,
            )?,
            config,
        })
    }

    /// Get configuration
    pub fn config(&self) -> &XLNetConfig {
        &self.config
    }
}

impl Module for XLNetFeedForward {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let residual = input;

        // First linear + GELU
        let hidden = self.dense_1.forward(input)?;
        let hidden = hidden.gelu()?;

        // Second linear
        let hidden = self.dense_2.forward(&hidden)?;

        // Dropout
        let hidden = self.dropout.forward(&hidden)?;

        // Residual connection
        let output = hidden.add(residual)?;

        // Layer normalization
        self.layer_norm.forward(&output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.dense_1.parameters() {
            params.insert(format!("dense_1.{}", name), param);
        }
        for (name, param) in self.dense_2.parameters() {
            params.insert(format!("dense_2.{}", name), param);
        }
        for (name, param) in self.layer_norm.parameters() {
            params.insert(format!("layer_norm.{}", name), param);
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
        self.dense_1.train();
        self.dense_2.train();
        self.dropout.train();
        self.layer_norm.train();
    }

    fn eval(&mut self) {
        self.dense_1.eval();
        self.dense_2.eval();
        self.dropout.eval();
        self.layer_norm.eval();
    }

    fn to_device(&mut self, device: torsh_core::DeviceType) -> Result<()> {
        self.dense_1.to_device(device)?;
        self.dense_2.to_device(device)?;
        self.dropout.to_device(device)?;
        self.layer_norm.to_device(device)?;
        Ok(())
    }
}

/// XLNet transformer layer
pub struct XLNetLayer {
    /// Self-attention mechanism
    attention: XLNetRelativeAttention,
    /// Feed-forward network
    feed_forward: XLNetFeedForward,
    /// Layer normalization for attention
    attention_layer_norm: LayerNorm,
    /// Dropout for attention
    attention_dropout: Dropout,
    /// Configuration
    config: XLNetConfig,
}

impl XLNetLayer {
    /// Create new XLNet layer
    pub fn new(config: XLNetConfig) -> Result<Self> {
        Ok(Self {
            attention: XLNetRelativeAttention::new(config.clone())?,
            feed_forward: XLNetFeedForward::new(config.clone())?,
            attention_layer_norm: LayerNorm::new(
                vec![config.hidden_size],
                config.layer_norm_eps as f64,
                true,
                torsh_core::DeviceType::Cpu,
            )?,
            attention_dropout: Dropout::new(config.attention_probs_dropout_prob),
            config,
        })
    }

    /// Get configuration
    pub fn config(&self) -> &XLNetConfig {
        &self.config
    }
}

impl Module for XLNetLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let residual = input;

        // Self-attention
        let attention_output = self.attention.forward(input)?;

        // Dropout
        let attention_output = self.attention_dropout.forward(&attention_output)?;

        // Residual connection
        let attention_output = attention_output.add(residual)?;

        // Layer normalization
        let attention_output = self.attention_layer_norm.forward(&attention_output)?;

        // Feed-forward
        self.feed_forward.forward(&attention_output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.attention.parameters() {
            params.insert(format!("attention.{}", name), param);
        }
        for (name, param) in self.feed_forward.parameters() {
            params.insert(format!("feed_forward.{}", name), param);
        }
        for (name, param) in self.attention_layer_norm.parameters() {
            params.insert(format!("attention_layer_norm.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.attention.training()
    }

    fn train(&mut self) {
        self.attention.train();
        self.feed_forward.train();
        self.attention_layer_norm.train();
        self.attention_dropout.train();
    }

    fn eval(&mut self) {
        self.attention.eval();
        self.feed_forward.eval();
        self.attention_layer_norm.eval();
        self.attention_dropout.eval();
    }

    fn to_device(&mut self, device: torsh_core::DeviceType) -> Result<()> {
        self.attention.to_device(device)?;
        self.feed_forward.to_device(device)?;
        self.attention_layer_norm.to_device(device)?;
        self.attention_dropout.to_device(device)?;
        Ok(())
    }
}

/// XLNet encoder (stack of transformer layers)
pub struct XLNetEncoder {
    /// Transformer layers
    layers: Vec<XLNetLayer>,
    /// Configuration
    config: XLNetConfig,
}

impl XLNetEncoder {
    /// Create new XLNet encoder
    pub fn new(config: XLNetConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(XLNetLayer::new(config.clone())?);
        }

        Ok(Self { layers, config })
    }

    /// Get configuration
    pub fn config(&self) -> &XLNetConfig {
        &self.config
    }
}

impl Module for XLNetEncoder {
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
    fn test_feed_forward_creation() {
        let config = XLNetConfig::xlnet_base();
        let ff = XLNetFeedForward::new(config);
        assert!(ff.is_ok());
    }

    #[test]
    fn test_layer_creation() {
        let config = XLNetConfig::xlnet_base();
        let layer = XLNetLayer::new(config);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_encoder_creation() {
        let config = XLNetConfig::xlnet_base();
        let encoder = XLNetEncoder::new(config);
        assert!(encoder.is_ok());
    }
}
