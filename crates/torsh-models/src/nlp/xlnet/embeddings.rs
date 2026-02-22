//! XLNet Embeddings
//!
//! This module implements word embeddings for XLNet models.
//! XLNet uses word embeddings without position embeddings in the input,
//! as it uses relative positional encodings in the attention mechanism.

use crate::nlp::xlnet::config::XLNetConfig;
use std::collections::HashMap;
use torsh_core::error::Result;
use torsh_nn::prelude::*;

/// XLNet embeddings layer
///
/// Unlike BERT, XLNet does not use absolute position embeddings in the input.
/// It only uses word embeddings and optionally segment embeddings.
pub struct XLNetEmbeddings {
    /// Word embeddings
    word_embeddings: Embedding,
    /// Layer normalization
    layer_norm: LayerNorm,
    /// Dropout
    dropout: Dropout,
    /// Configuration
    config: XLNetConfig,
}

impl XLNetEmbeddings {
    /// Create new XLNet embeddings
    pub fn new(config: XLNetConfig) -> Result<Self> {
        let word_embeddings = Embedding::new(config.vocab_size, config.hidden_size);
        let layer_norm = LayerNorm::new(
            vec![config.hidden_size],
            config.layer_norm_eps as f64,
            true,
            torsh_core::DeviceType::Cpu,
        );
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Ok(Self {
            word_embeddings,
            layer_norm: layer_norm?,
            dropout,
            config,
        })
    }

    /// Get configuration
    pub fn config(&self) -> &XLNetConfig {
        &self.config
    }
}

impl Module for XLNetEmbeddings {
    fn forward(&self, input: &torsh_tensor::Tensor) -> Result<torsh_tensor::Tensor> {
        // Get word embeddings
        let embeddings = self.word_embeddings.forward(input)?;

        // Apply layer normalization
        let embeddings = self.layer_norm.forward(&embeddings)?;

        // Apply dropout
        let embeddings = self.dropout.forward(&embeddings)?;

        Ok(embeddings)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        // Word embeddings parameters
        for (name, param) in self.word_embeddings.parameters() {
            params.insert(format!("word_embeddings.{}", name), param);
        }

        // Layer norm parameters
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
        self.word_embeddings.train();
        self.layer_norm.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.word_embeddings.eval();
        self.layer_norm.eval();
        self.dropout.eval();
    }

    fn to_device(&mut self, device: torsh_core::DeviceType) -> Result<()> {
        self.word_embeddings.to_device(device)?;
        self.layer_norm.to_device(device)?;
        self.dropout.to_device(device)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xlnet_embeddings_creation() {
        let config = XLNetConfig::xlnet_base();
        let embeddings = XLNetEmbeddings::new(config);
        assert!(embeddings.is_ok());
    }

    #[test]
    fn test_embeddings_parameters() {
        let config = XLNetConfig::xlnet_base();
        let embeddings = XLNetEmbeddings::new(config).unwrap();
        let params = embeddings.parameters();
        assert!(params.contains_key("word_embeddings.weight"));
        assert!(params.contains_key("layer_norm.weight"));
        assert!(params.contains_key("layer_norm.bias"));
    }
}
