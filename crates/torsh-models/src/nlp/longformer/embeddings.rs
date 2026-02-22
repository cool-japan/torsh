//! Longformer Embeddings
//!
//! This module implements embeddings for Longformer models.
//! Longformer extends BERT's embeddings to support longer sequences (up to 4096 tokens).

use crate::nlp::longformer::config::LongformerConfig;
use std::collections::HashMap;
use torsh_core::error::Result;
use torsh_nn::prelude::*;

/// Longformer embeddings layer
///
/// Similar to BERT but with extended position embeddings for long documents
pub struct LongformerEmbeddings {
    /// Word embeddings
    word_embeddings: Embedding,
    /// Position embeddings (extended to 4096)
    position_embeddings: Embedding,
    /// Token type embeddings
    token_type_embeddings: Embedding,
    /// Layer normalization
    layer_norm: LayerNorm,
    /// Dropout
    dropout: Dropout,
    /// Configuration
    config: LongformerConfig,
}

impl LongformerEmbeddings {
    /// Create new Longformer embeddings
    pub fn new(config: LongformerConfig) -> Result<Self> {
        let word_embeddings = Embedding::new(config.vocab_size, config.hidden_size);
        let position_embeddings =
            Embedding::new(config.max_position_embeddings, config.hidden_size);
        let token_type_embeddings = Embedding::new(config.type_vocab_size, config.hidden_size);
        let layer_norm = LayerNorm::new(
            vec![config.hidden_size],
            config.layer_norm_eps as f64,
            true,
            torsh_core::DeviceType::Cpu,
        );
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm: layer_norm?,
            dropout,
            config,
        })
    }

    /// Get configuration
    pub fn config(&self) -> &LongformerConfig {
        &self.config
    }
}

impl Module for LongformerEmbeddings {
    fn forward(&self, input: &torsh_tensor::Tensor) -> Result<torsh_tensor::Tensor> {
        let seq_length = input.shape().dims()[1];

        // Get word embeddings
        let mut embeddings = self.word_embeddings.forward(input)?;

        // Create position IDs (0, 1, 2, ..., seq_length-1)
        let position_ids = torsh_tensor::creation::arange(0.0, seq_length as f32, 1.0)?;
        let position_ids = position_ids.unsqueeze(0)?;

        // Add position embeddings
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;
        embeddings = embeddings.add(&position_embeddings)?;

        // Create token type IDs (all zeros for single sequence)
        let token_type_ids = torsh_tensor::creation::zeros(input.shape().dims())?;
        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids)?;
        embeddings = embeddings.add(&token_type_embeddings)?;

        // Apply layer normalization
        let embeddings = self.layer_norm.forward(&embeddings)?;

        // Apply dropout
        self.dropout.forward(&embeddings)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.word_embeddings.parameters() {
            params.insert(format!("word_embeddings.{}", name), param);
        }
        for (name, param) in self.position_embeddings.parameters() {
            params.insert(format!("position_embeddings.{}", name), param);
        }
        for (name, param) in self.token_type_embeddings.parameters() {
            params.insert(format!("token_type_embeddings.{}", name), param);
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
        self.word_embeddings.train();
        self.position_embeddings.train();
        self.token_type_embeddings.train();
        self.layer_norm.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.word_embeddings.eval();
        self.position_embeddings.eval();
        self.token_type_embeddings.eval();
        self.layer_norm.eval();
        self.dropout.eval();
    }

    fn to_device(&mut self, device: torsh_core::DeviceType) -> Result<()> {
        self.word_embeddings.to_device(device)?;
        self.position_embeddings.to_device(device)?;
        self.token_type_embeddings.to_device(device)?;
        self.layer_norm.to_device(device)?;
        self.dropout.to_device(device)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_longformer_embeddings_creation() {
        let config = LongformerConfig::longformer_base();
        let embeddings = LongformerEmbeddings::new(config);
        assert!(embeddings.is_ok());
    }

    #[test]
    fn test_embeddings_parameters() {
        let config = LongformerConfig::longformer_base();
        let embeddings = LongformerEmbeddings::new(config).unwrap();
        let params = embeddings.parameters();
        assert!(params.contains_key("word_embeddings.weight"));
        assert!(params.contains_key("position_embeddings.weight"));
        assert!(params.contains_key("token_type_embeddings.weight"));
    }
}
