//! BigBird Embeddings

use crate::nlp::bigbird::config::BigBirdConfig;
use std::collections::HashMap;
use torsh_core::error::Result;
use torsh_nn::prelude::*;

pub struct BigBirdEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    config: BigBirdConfig,
}

impl BigBirdEmbeddings {
    pub fn new(config: BigBirdConfig) -> Result<Self> {
        Ok(Self {
            word_embeddings: Embedding::new(config.vocab_size, config.hidden_size),
            position_embeddings: Embedding::new(config.max_position_embeddings, config.hidden_size),
            token_type_embeddings: Embedding::new(config.type_vocab_size, config.hidden_size),
            layer_norm: LayerNorm::new(
                vec![config.hidden_size],
                config.layer_norm_eps as f64,
                true,
                torsh_core::DeviceType::Cpu,
            )?,
            dropout: Dropout::new(config.hidden_dropout_prob),
            config,
        })
    }

    pub fn config(&self) -> &BigBirdConfig {
        &self.config
    }
}

impl Module for BigBirdEmbeddings {
    fn forward(&self, input: &torsh_tensor::Tensor) -> Result<torsh_tensor::Tensor> {
        let seq_length = input.shape().dims()[1];
        let mut embeddings = self.word_embeddings.forward(input)?;

        let position_ids = torsh_tensor::creation::arange(0.0, seq_length as f32, 1.0)?;
        let position_ids = position_ids.unsqueeze(0)?;
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;
        embeddings = embeddings.add(&position_embeddings)?;

        let token_type_ids = torsh_tensor::creation::zeros(input.shape().dims())?;
        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids)?;
        embeddings = embeddings.add(&token_type_embeddings)?;

        let embeddings = self.layer_norm.forward(&embeddings)?;
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
