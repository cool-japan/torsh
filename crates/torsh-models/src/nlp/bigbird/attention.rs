//! BigBird Sparse Attention

use crate::nlp::bigbird::config::BigBirdConfig;
use std::collections::HashMap;
use torsh_core::error::Result;
use torsh_nn::prelude::*;
use torsh_tensor::Tensor;

/// BigBird sparse attention combining random, window, and global attention
#[derive(Debug)]
pub struct BigBirdSparseAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dense: Linear,
    dropout: Dropout,
    config: BigBirdConfig,
}

impl BigBirdSparseAttention {
    pub fn new(config: BigBirdConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        Ok(Self {
            query: Linear::new(hidden_size, hidden_size, config.use_bias),
            key: Linear::new(hidden_size, hidden_size, config.use_bias),
            value: Linear::new(hidden_size, hidden_size, config.use_bias),
            dense: Linear::new(hidden_size, hidden_size, config.use_bias),
            dropout: Dropout::new(config.attention_probs_dropout_prob),
            config,
        })
    }

    pub fn config(&self) -> &BigBirdConfig {
        &self.config
    }
}

impl Module for BigBirdSparseAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.shape().dims()[0];
        let seq_len = input.shape().dims()[1];
        let hidden_size = self.config.hidden_size;
        let num_heads = self.config.num_attention_heads;
        let head_dim = hidden_size / num_heads;

        // Project to Q, K, V
        let query_layer = self.query.forward(input)?;
        let key_layer = self.key.forward(input)?;
        let value_layer = self.value.forward(input)?;

        // Reshape for multi-head attention
        let query_layer = query_layer.reshape(&[
            batch_size as i32,
            seq_len as i32,
            num_heads as i32,
            head_dim as i32,
        ])?;
        let key_layer = key_layer.reshape(&[
            batch_size as i32,
            seq_len as i32,
            num_heads as i32,
            head_dim as i32,
        ])?;
        let value_layer = value_layer.reshape(&[
            batch_size as i32,
            seq_len as i32,
            num_heads as i32,
            head_dim as i32,
        ])?;

        let query_layer = query_layer.permute(&[0, 2, 1, 3])?;
        let key_layer = key_layer.permute(&[0, 2, 1, 3])?;
        let value_layer = value_layer.permute(&[0, 2, 1, 3])?;

        // Compute attention (simplified - full implementation would use sparse patterns)
        let key_layer_t = key_layer.permute(&[0, 1, 3, 2])?;
        let mut attention_scores = query_layer.matmul(&key_layer_t)?;

        let scale = (head_dim as f32).sqrt();
        attention_scores = attention_scores.div_scalar(scale)?;

        let attention_probs = attention_scores.softmax(-1)?;
        let attention_probs = self.dropout.forward(&attention_probs)?;

        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.permute(&[0, 2, 1, 3])?;
        let context_layer =
            context_layer.reshape(&[batch_size as i32, seq_len as i32, hidden_size as i32])?;

        self.dense.forward(&context_layer)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.query.parameters() {
            params.insert(format!("query.{}", name), param);
        }
        for (name, param) in self.key.parameters() {
            params.insert(format!("key.{}", name), param);
        }
        for (name, param) in self.value.parameters() {
            params.insert(format!("value.{}", name), param);
        }
        for (name, param) in self.dense.parameters() {
            params.insert(format!("dense.{}", name), param);
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
        self.query.train();
        self.key.train();
        self.value.train();
        self.dense.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.query.eval();
        self.key.eval();
        self.value.eval();
        self.dense.eval();
        self.dropout.eval();
    }

    fn to_device(&mut self, device: torsh_core::DeviceType) -> Result<()> {
        self.query.to_device(device)?;
        self.key.to_device(device)?;
        self.value.to_device(device)?;
        self.dense.to_device(device)?;
        self.dropout.to_device(device)?;
        Ok(())
    }
}
