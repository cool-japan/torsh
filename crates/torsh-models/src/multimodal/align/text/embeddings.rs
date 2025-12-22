//! ALIGN Text Embeddings
//!
//! Word, position, and token type embeddings for BERT-style text encoder.

use std::collections::HashMap;
use torsh_core::{error::Result, DeviceType};
use torsh_nn::prelude::{Dropout, Embedding, LayerNorm};
use torsh_nn::{Module, Parameter};
use torsh_tensor::{creation, Tensor};

use crate::multimodal::align::config::ALIGNTextConfig;

/// ALIGN Text Embeddings (word, position, and token type embeddings)
pub struct ALIGNTextEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    config: ALIGNTextConfig,
}

impl ALIGNTextEmbeddings {
    pub fn new(config: ALIGNTextConfig) -> Result<Self> {
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
}

impl Module for ALIGNTextEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.size(1)?;
        let batch_size = input_ids.size(0)?;

        let words_embeddings = self.word_embeddings.forward(input_ids)?;

        let position_ids = creation::arange(0i64, seq_length as i64, 1i64)?
            .unsqueeze(0)?
            .expand(&[batch_size, seq_length])?;
        let position_ids_f32 = position_ids.to_f32_simd()?;
        let position_embeddings = self.position_embeddings.forward(&position_ids_f32)?;

        let token_type_ids = creation::zeros(&[batch_size, seq_length])?;
        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids)?;

        let mut embeddings = words_embeddings.add(&position_embeddings)?;
        embeddings = embeddings.add(&token_type_embeddings)?;
        embeddings = self.layer_norm.forward(&embeddings)?;
        embeddings = self.dropout.forward(&embeddings)?;

        Ok(embeddings)
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
            params.insert(format!("LayerNorm.{}", name), param);
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
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.dropout.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.word_embeddings.to_device(device)?;
        self.position_embeddings.to_device(device)?;
        self.token_type_embeddings.to_device(device)?;
        self.layer_norm.to_device(device)?;
        Ok(())
    }
}
