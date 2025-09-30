//! ALIGN Text Encoder
//!
//! BERT-based text encoder for ALIGN model with embeddings, transformer layers,
//! and pooling for text feature extraction.

use std::collections::HashMap;
use torsh_core::{error::Result, DeviceType};
use torsh_nn::prelude::Linear;
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

use super::bert::ALIGNBertEncoder;
use super::embeddings::ALIGNTextEmbeddings;
use crate::multimodal::align::config::ALIGNTextConfig;

/// ALIGN Text Encoder (BERT-based architecture)
pub struct ALIGNTextEncoder {
    embeddings: ALIGNTextEmbeddings,
    encoder: ALIGNBertEncoder,
    pooler: Linear,
    config: ALIGNTextConfig,
}

impl ALIGNTextEncoder {
    pub fn new(config: ALIGNTextConfig) -> Result<Self> {
        let embeddings = ALIGNTextEmbeddings::new(config.clone())?;
        let encoder = ALIGNBertEncoder::new(config.clone())?;
        let pooler = Linear::new(config.hidden_size, config.hidden_size, true);

        Ok(Self {
            embeddings,
            encoder,
            pooler,
            config,
        })
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters().len()
    }
}

impl Module for ALIGNTextEncoder {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden_states = self.embeddings.forward(input_ids)?;
        let encoder_outputs = self.encoder.forward(&hidden_states)?;

        // Pool the representation using [CLS] token
        let first_token_tensor = encoder_outputs.select(1, 0)?;
        let pooled_output = self.pooler.forward(&first_token_tensor)?;
        let pooled_output = pooled_output.tanh()?;

        Ok(pooled_output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.embeddings.parameters() {
            params.insert(format!("embeddings.{}", name), param);
        }

        for (name, param) in self.encoder.parameters() {
            params.insert(format!("encoder.{}", name), param);
        }

        for (name, param) in self.pooler.parameters() {
            params.insert(format!("pooler.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.embeddings.training() && self.encoder.training()
    }

    fn train(&mut self) {
        self.embeddings.train();
        self.encoder.train();
    }

    fn eval(&mut self) {
        self.embeddings.eval();
        self.encoder.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.embeddings.to_device(device)?;
        self.encoder.to_device(device)?;
        self.pooler.to_device(device)?;
        Ok(())
    }
}
