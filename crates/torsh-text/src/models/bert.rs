use super::transformer::TransformerEncoder;
use crate::{TextModel, TextModelConfig};
use std::collections::HashMap;
use torsh_core::{device::DeviceType, Result};
use torsh_nn::{prelude::*, Module, Parameter};
use torsh_tensor::creation::*;
use torsh_tensor::Tensor;

/// BERT embeddings (token + position + token_type)
pub struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    max_position_embeddings: usize,
    is_training: bool,
}

impl BertEmbeddings {
    pub fn new(config: &TextModelConfig, _device: DeviceType) -> Result<Self> {
        Ok(Self {
            word_embeddings: Embedding::new(config.vocab_size, config.hidden_dim),
            position_embeddings: Embedding::new(config.max_position_embeddings, config.hidden_dim),
            token_type_embeddings: Embedding::new(2, config.hidden_dim), // 2 token types for BERT
            layer_norm: LayerNorm::new(vec![config.hidden_dim]),
            dropout: Dropout::new(config.dropout),
            max_position_embeddings: config.max_position_embeddings,
            is_training: true,
        })
    }

    pub fn forward_with_type_ids(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _seq_len = input_ids.size(1)?;

        // Get word embeddings
        let word_embeddings = self.word_embeddings.forward(input_ids)?;

        // Get position embeddings - simplified implementation
        let position_ids = if let Some(pos_ids) = position_ids {
            pos_ids.clone()
        } else {
            // For now, use a dummy tensor - proper position encoding needs implementation
            input_ids.clone() // This is a workaround
        };
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;

        // Get token type embeddings
        let token_type_ids = if let Some(type_ids) = token_type_ids {
            type_ids.clone()
        } else {
            // Default to all zeros (single segment)
            zeros_like(input_ids)
        };
        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids)?;

        // Combine embeddings
        let embeddings = word_embeddings
            .add(&position_embeddings)?
            .add(&token_type_embeddings)?;
        let embeddings = self.layer_norm.forward(&embeddings)?;
        self.dropout.forward(&embeddings)
    }
}

impl Module for BertEmbeddings {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_with_type_ids(input, None, None)
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

    fn train(&mut self) {
        self.is_training = true;
        // Note: The individual layers need to be mutable to call train()
        // This is a simplified implementation
    }

    fn eval(&mut self) {
        self.is_training = false;
        // Note: The individual layers need to be mutable to call eval()
        // This is a simplified implementation
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.word_embeddings.to_device(device)?;
        self.position_embeddings.to_device(device)?;
        self.token_type_embeddings.to_device(device)?;
        self.layer_norm.to_device(device)?;
        Ok(())
    }
}

/// BERT pooler for extracting sentence representations
pub struct BertPooler {
    dense: Linear,
    is_training: bool,
}

impl BertPooler {
    pub fn new(hidden_dim: usize, _device: DeviceType) -> Result<Self> {
        Ok(Self {
            dense: Linear::new(hidden_dim, hidden_dim, true),
            is_training: true,
        })
    }
}

impl Module for BertPooler {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Take the hidden state of the first token ([CLS])
        let first_token = input.narrow(1, 0, 1).unwrap().squeeze();
        let pooled = self.dense.forward(&first_token)?;
        pooled.tanh()
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.dense.parameters() {
            params.insert(format!("dense.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.is_training = true;
        self.dense.train();
    }

    fn eval(&mut self) {
        self.is_training = false;
        self.dense.eval();
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.dense.to_device(device)
    }
}

/// BERT model
pub struct BertModel {
    embeddings: BertEmbeddings,
    encoder: TransformerEncoder,
    pooler: Option<BertPooler>,
    config: TextModelConfig,
    is_training: bool,
}

impl BertModel {
    pub fn new(config: TextModelConfig, device: DeviceType) -> Result<Self> {
        let add_pooling_layer = true; // Default to true for compatibility
        let pooler = if add_pooling_layer {
            Some(BertPooler::new(config.hidden_dim, device)?)
        } else {
            None
        };

        Ok(Self {
            embeddings: BertEmbeddings::new(&config, device)?,
            encoder: TransformerEncoder::new(&config, device)?,
            pooler,
            config,
            is_training: true,
        })
    }

    pub fn forward_with_type_ids(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Get embeddings
        let hidden_states =
            self.embeddings
                .forward_with_type_ids(input_ids, token_type_ids, None)?;

        // Convert attention mask to attention bias if provided
        let attention_mask = if let Some(mask) = attention_mask {
            // Convert binary mask (1 for attend, 0 for not attend) to attention bias
            // (0 for attend, -inf for not attend)
            let inverted_mask = mask.sub_scalar(1.0)?.mul_scalar(-1.0)?;
            let attention_bias = inverted_mask.mul_scalar(f32::NEG_INFINITY)?;
            Some(attention_bias)
        } else {
            None
        };

        // Pass through encoder with attention mask
        let encoder_outputs = self
            .encoder
            .forward_with_mask(&hidden_states, attention_mask.as_ref())?;

        // Apply pooler if available
        let pooled_output = if let Some(ref pooler) = self.pooler {
            Some(pooler.forward(&encoder_outputs)?)
        } else {
            None
        };

        Ok((encoder_outputs, pooled_output))
    }
}

impl Module for BertModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (sequence_output, _) = self.forward_with_type_ids(input, None, None)?;
        Ok(sequence_output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.embeddings.parameters() {
            params.insert(format!("embeddings.{}", name), param);
        }
        for (name, param) in self.encoder.parameters() {
            params.insert(format!("encoder.{}", name), param);
        }
        if let Some(ref pooler) = self.pooler {
            for (name, param) in pooler.parameters() {
                params.insert(format!("pooler.{}", name), param);
            }
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.is_training = true;
        self.embeddings.train();
        self.encoder.train();
        if let Some(ref mut pooler) = self.pooler {
            pooler.train();
        }
    }

    fn eval(&mut self) {
        self.is_training = false;
        self.embeddings.eval();
        self.encoder.eval();
        if let Some(ref mut pooler) = self.pooler {
            pooler.eval();
        }
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.embeddings.to_device(device)?;
        self.encoder.to_device(device)?;
        if let Some(ref mut pooler) = self.pooler {
            pooler.to_device(device)?;
        }
        Ok(())
    }
}

impl TextModel for BertModel {
    fn name(&self) -> &str {
        "BERT"
    }

    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }

    fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    fn max_seq_length(&self) -> usize {
        self.config.max_position_embeddings
    }
}

/// BERT for sequence classification
pub struct BertForSequenceClassification {
    bert: BertModel,
    classifier: Linear,
    dropout: Dropout,
    num_labels: usize,
    is_training: bool,
}

impl BertForSequenceClassification {
    pub fn new(config: TextModelConfig, num_labels: usize, device: DeviceType) -> Result<Self> {
        Ok(Self {
            bert: BertModel::new(config.clone(), device)?,
            classifier: Linear::new(config.hidden_dim, num_labels, true),
            dropout: Dropout::new(config.dropout),
            num_labels,
            is_training: true,
        })
    }
}

impl Module for BertForSequenceClassification {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (_, pooled_output) = self.bert.forward_with_type_ids(input, None, None)?;
        let pooled_output = pooled_output.ok_or_else(|| {
            torsh_core::error::TorshError::InvalidArgument("BERT pooler output is None".to_string())
        })?;

        let pooled_output = self.dropout.forward(&pooled_output)?;
        self.classifier.forward(&pooled_output)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.bert.parameters() {
            params.insert(format!("bert.{}", name), param);
        }
        for (name, param) in self.classifier.parameters() {
            params.insert(format!("classifier.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn train(&mut self) {
        self.is_training = true;
        self.bert.train();
        self.classifier.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.is_training = false;
        self.bert.eval();
        self.classifier.eval();
        self.dropout.eval();
    }

    fn training(&self) -> bool {
        self.is_training
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.bert.to_device(device)?;
        self.classifier.to_device(device)?;
        Ok(())
    }
}
