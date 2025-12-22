//! RoBERTa embedding layers

use super::config::RobertaConfig;
use torsh_core::error::Result;
use torsh_nn::prelude::*;
use torsh_nn::Module;
use torsh_tensor::{creation, Tensor};

/// RoBERTa Embeddings
pub struct RobertaEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    config: RobertaConfig,
}

impl RobertaEmbeddings {
    pub fn new(config: RobertaConfig) -> Result<Self> {
        let word_embeddings = Embedding::new(config.vocab_size, config.hidden_size);
        let position_embeddings =
            Embedding::new(config.max_position_embeddings, config.hidden_size);
        let token_type_embeddings = Embedding::new(config.type_vocab_size, config.hidden_size);
        let layer_norm = LayerNorm::new(
            vec![config.hidden_size],
            config.layer_norm_eps as f64,
            true,
            torsh_core::DeviceType::Cpu,
        )?;
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
            config,
        })
    }

    /// Create position IDs tensor for RoBERTa (offset by pad_token_id + 1)
    pub fn create_position_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.size(1)?;
        let position_offset = (self.config.pad_token_id + 1) as f32;

        let position_ids: Vec<f32> = (0..seq_length)
            .map(|i| i as f32 + position_offset)
            .collect();

        creation::from_vec(position_ids, &[1, seq_length], torsh_core::DeviceType::Cpu)
    }
}

impl Module for RobertaEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.size(1)?;

        // Create position ids (0, 1, 2, ..., seq_length-1) offset by pad_token_id + 1
        let position_ids = self.create_position_ids(input_ids)?;

        // Create token type ids (all zeros for single sequence)
        let token_type_ids = Tensor::zeros(
            &[input_ids.size(0)?, seq_length],
            torsh_core::DeviceType::Cpu,
        )?;

        // Get embeddings
        let words_embeddings = self.word_embeddings.forward(input_ids)?;
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids)?;

        // Sum all embeddings
        let embeddings = words_embeddings
            .add(&position_embeddings)?
            .add(&token_type_embeddings)?;

        // Apply layer norm and dropout
        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings)?;

        Ok(embeddings)
    }
}
