//! RoBERTa (Robustly Optimized BERT Pretraining Approach) Models
//!
//! This module contains comprehensive implementations of RoBERTa transformer models,
//! extracted from the massive monolithic nlp.rs file as part of Phase 14 systematic refactoring.
//!
//! # RoBERTa Architecture Components
//!
//! - `RobertaConfig` - Model configuration with base/large variants
//! - `RobertaEmbeddings` - Token, position, and type embeddings
//! - `RobertaSelfAttention` - Multi-head self-attention mechanism
//! - `RobertaSelfOutput` - Self-attention output projection
//! - `RobertaAttention` - Complete attention layer (self-attention + output)
//! - `RobertaIntermediate` - Feed-forward intermediate layer
//! - `RobertaOutput` - Feed-forward output layer with residual connections
//! - `RobertaLayer` - Complete transformer layer (attention + feed-forward)
//! - `RobertaEncoder` - Stack of RoBERTa layers
//! - `RobertaPooler` - Pooling layer for sequence-level representations
//! - `RobertaModel` - Complete RoBERTa base model
//! - `RobertaForSequenceClassification` - RoBERTa with classification head
//!
//! # Usage Examples
//!
//! ```rust
//! use crate::models::roberta::*;
//!
//! // Create RoBERTa-base model
//! let config = RobertaConfig::roberta_base();
//! let model = RobertaModel::new(config);
//!
//! // Create RoBERTa for sequence classification
//! let classifier = RobertaForSequenceClassification::roberta_base_for_classification(2);
//!
//! // Forward pass
//! let input_ids = Tensor::new(&[batch_size, seq_len])?;
//! let output = model.forward(&input_ids)?;
//! ```

use crate::ModelResult;
use torsh_core::{
    error::{Result, TorshError},
    DeviceType,
};
use torsh_nn::prelude::*;
use torsh_nn::{Module, Parameter};
use torsh_tensor::{creation, Tensor};
use std::collections::HashMap;

// ========================================
// ROBERTA CONFIGURATION
// ========================================

/// Configuration for RoBERTa model
#[derive(Debug, Clone)]
pub struct RobertaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    pub pad_token_id: usize,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub position_embedding_type: String,
}

impl Default for RobertaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50265,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 514,
            type_vocab_size: 1,
            initializer_range: 0.02,
            layer_norm_eps: 1e-5,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            position_embedding_type: "absolute".to_string(),
        }
    }
}

impl RobertaConfig {
    /// Create RoBERTa-base configuration
    pub fn roberta_base() -> Self {
        Self::default()
    }

    /// Create RoBERTa-large configuration
    pub fn roberta_large() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            ..Self::default()
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(TorshError::InvalidArgument(
                "hidden_size must be divisible by num_attention_heads".to_string()
            ));
        }

        if self.num_hidden_layers == 0 {
            return Err(TorshError::InvalidArgument(
                "num_hidden_layers must be greater than 0".to_string()
            ));
        }

        Ok(())
    }

    /// Get attention head size
    pub fn attention_head_size(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ========================================
// ROBERTA EMBEDDINGS
// ========================================

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
    pub fn new(config: RobertaConfig) -> Self {
        let word_embeddings = Embedding::new(config.vocab_size, config.hidden_size);
        let position_embeddings =
            Embedding::new(config.max_position_embeddings, config.hidden_size);
        let token_type_embeddings = Embedding::new(config.type_vocab_size, config.hidden_size);
        let layer_norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps, true);
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
            config,
        }
    }

    /// Create position IDs for the input
    fn create_position_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.size(1)?;
        let position_offset = (self.config.pad_token_id + 1) as f32;

        let position_ids = Tensor::arange_with_step(
            position_offset,
            position_offset + seq_length as f32,
            1.0,
            input_ids.device(),
        )?;

        Ok(position_ids.unsqueeze(0)?.expand(&[input_ids.size(0)?, seq_length])?)
    }
}

impl Module for RobertaEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.size(1)?;

        // Create position ids (0, 1, 2, ..., seq_length-1) offset by pad_token_id + 1
        let position_ids = self.create_position_ids(input_ids)?;

        // Get embeddings
        let word_embeddings = self.word_embeddings.forward(input_ids)?;
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;

        // Token type embeddings (for RoBERTa, usually all zeros)
        let token_type_ids = Tensor::zeros(&[input_ids.size(0)?, seq_length])?;
        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids)?;

        // Combine embeddings
        let embeddings = word_embeddings
            .add(&position_embeddings)?
            .add(&token_type_embeddings)?;

        // Apply layer norm and dropout
        let embeddings = self.layer_norm.forward(&embeddings)?;
        let embeddings = self.dropout.forward(&embeddings)?;

        Ok(embeddings)
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

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.word_embeddings.parameters());
        params.extend(self.position_embeddings.parameters());
        params.extend(self.token_type_embeddings.parameters());
        params.extend(self.layer_norm.parameters());
        params
    }
}

// ========================================
// ROBERTA SELF-ATTENTION
// ========================================

/// RoBERTa Self-Attention mechanism
pub struct RobertaSelfAttention {
    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    config: RobertaConfig,
}

impl RobertaSelfAttention {
    pub fn new(config: RobertaConfig) -> Self {
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / num_attention_heads;
        let all_head_size = num_attention_heads * attention_head_size;

        let query = Linear::new(config.hidden_size, all_head_size, true);
        let key = Linear::new(config.hidden_size, all_head_size, true);
        let value = Linear::new(config.hidden_size, all_head_size, true);
        let dropout = Dropout::new(config.attention_probs_dropout_prob);

        Self {
            num_attention_heads,
            attention_head_size,
            all_head_size,
            query,
            key,
            value,
            dropout,
            config,
        }
    }

    /// Transpose for attention scores computation
    fn transpose_for_scores(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.size(0)?;
        let seq_length = x.size(1)?;

        // Reshape to [batch_size, seq_length, num_heads, head_size]
        let x = x.view(&[batch_size, seq_length, self.num_attention_heads, self.attention_head_size])?;

        // Transpose to [batch_size, num_heads, seq_length, head_size]
        x.permute(&[0, 2, 1, 3])
    }
}

impl Module for RobertaSelfAttention {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        // Compute attention scores
        let attention_scores = query_layer.matmul(&key_layer.transpose(-1, -2)?)?;
        let attention_scores = attention_scores.div_scalar((self.attention_head_size as f32).sqrt())?;

        // Apply softmax
        let attention_probs = attention_scores.softmax(-1)?;
        let attention_probs = self.dropout.forward(&attention_probs)?;

        // Apply attention to values
        let context_layer = attention_probs.matmul(&value_layer)?;

        // Transpose back to [batch_size, seq_length, num_heads, head_size]
        let context_layer = context_layer.permute(&[0, 2, 1, 3])?;

        // Reshape to [batch_size, seq_length, all_head_size]
        let batch_size = context_layer.size(0)?;
        let seq_length = context_layer.size(2)?;
        context_layer.contiguous()?.view(&[batch_size, seq_length, self.all_head_size])
    }

    fn train(&mut self) {
        self.query.train();
        self.key.train();
        self.value.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.query.eval();
        self.key.eval();
        self.value.eval();
        self.dropout.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.query.parameters());
        params.extend(self.key.parameters());
        params.extend(self.value.parameters());
        params
    }
}

// ========================================
// ROBERTA SELF-ATTENTION OUTPUT
// ========================================

/// RoBERTa Self-Attention Output projection
pub struct RobertaSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl RobertaSelfOutput {
    pub fn new(config: &RobertaConfig) -> Self {
        let dense = Linear::new(config.hidden_size, config.hidden_size, true);
        let layer_norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps, true);
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Self {
            dense,
            layer_norm,
            dropout,
        }
    }
}

impl Module for RobertaSelfOutput {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        Ok(hidden_states)
    }

    fn train(&mut self) {
        self.dense.train();
        self.layer_norm.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.dense.eval();
        self.layer_norm.eval();
        self.dropout.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.dense.parameters());
        params.extend(self.layer_norm.parameters());
        params
    }
}

// ========================================
// ROBERTA ATTENTION LAYER
// ========================================

/// Complete RoBERTa Attention layer (self-attention + output)
pub struct RobertaAttention {
    self_attention: RobertaSelfAttention,
    output: RobertaSelfOutput,
}

impl RobertaAttention {
    pub fn new(config: RobertaConfig) -> Self {
        let self_attention = RobertaSelfAttention::new(config.clone());
        let output = RobertaSelfOutput::new(&config);

        Self {
            self_attention,
            output,
        }
    }
}

impl Module for RobertaAttention {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let self_attention_output = self.self_attention.forward(hidden_states)?;
        let attention_output = self.output.forward(&self_attention_output)?;

        // Add residual connection
        attention_output.add(hidden_states)
    }

    fn train(&mut self) {
        self.self_attention.train();
        self.output.train();
    }

    fn eval(&mut self) {
        self.self_attention.eval();
        self.output.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.self_attention.parameters());
        params.extend(self.output.parameters());
        params
    }
}

// ========================================
// ROBERTA FEED-FORWARD LAYERS
// ========================================

/// RoBERTa Intermediate (Feed-Forward) layer
pub struct RobertaIntermediate {
    dense: Linear,
    activation: String,
}

impl RobertaIntermediate {
    pub fn new(config: &RobertaConfig) -> Self {
        let dense = Linear::new(config.hidden_size, config.intermediate_size, true);
        let activation = config.hidden_act.clone();

        Self { dense, activation }
    }

    fn apply_activation(&self, input: &Tensor) -> Result<Tensor> {
        match self.activation.as_str() {
            "gelu" => input.gelu(),
            "relu" => input.relu(),
            "tanh" => Ok(input.tanh()),
            _ => input.gelu(), // Default to GELU
        }
    }
}

impl Module for RobertaIntermediate {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        self.apply_activation(&hidden_states)
    }

    fn train(&mut self) {
        self.dense.train();
    }

    fn eval(&mut self) {
        self.dense.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.dense.parameters()
    }
}

/// RoBERTa Output (Feed-Forward output) layer
pub struct RobertaOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl RobertaOutput {
    pub fn new(config: &RobertaConfig) -> Self {
        let dense = Linear::new(config.intermediate_size, config.hidden_size, true);
        let layer_norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps, true);
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Self {
            dense,
            layer_norm,
            dropout,
        }
    }
}

impl Module for RobertaOutput {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        Ok(hidden_states)
    }

    fn train(&mut self) {
        self.dense.train();
        self.layer_norm.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.dense.eval();
        self.layer_norm.eval();
        self.dropout.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.dense.parameters());
        params.extend(self.layer_norm.parameters());
        params
    }
}

// ========================================
// ROBERTA TRANSFORMER LAYER
// ========================================

/// Complete RoBERTa Transformer layer
pub struct RobertaLayer {
    attention: RobertaAttention,
    intermediate: RobertaIntermediate,
    output: RobertaOutput,
}

impl RobertaLayer {
    pub fn new(config: RobertaConfig) -> Self {
        let attention = RobertaAttention::new(config.clone());
        let intermediate = RobertaIntermediate::new(&config);
        let output = RobertaOutput::new(&config);

        Self {
            attention,
            intermediate,
            output,
        }
    }
}

impl Module for RobertaLayer {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Self-attention
        let attention_output = self.attention.forward(hidden_states)?;

        // Feed-forward
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self.output.forward(&intermediate_output)?;

        // Add residual connection
        layer_output.add(&attention_output)
    }

    fn train(&mut self) {
        self.attention.train();
        self.intermediate.train();
        self.output.train();
    }

    fn eval(&mut self) {
        self.attention.eval();
        self.intermediate.eval();
        self.output.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.attention.parameters());
        params.extend(self.intermediate.parameters());
        params.extend(self.output.parameters());
        params
    }
}

// ========================================
// ROBERTA ENCODER
// ========================================

/// RoBERTa Encoder (stack of transformer layers)
pub struct RobertaEncoder {
    layers: Vec<RobertaLayer>,
}

impl RobertaEncoder {
    pub fn new(config: RobertaConfig) -> Self {
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(RobertaLayer::new(config.clone()));
        }

        Self { layers }
    }
}

impl Module for RobertaEncoder {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        Ok(hidden_states)
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

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("layer_{}.{}", i, name), param);
            }
        }
        params
    }
}

// ========================================
// ROBERTA POOLER
// ========================================

/// RoBERTa Pooler for sequence-level representations
pub struct RobertaPooler {
    dense: Linear,
    activation: String,
}

impl RobertaPooler {
    pub fn new(config: &RobertaConfig) -> Self {
        let dense = Linear::new(config.hidden_size, config.hidden_size, true);
        let activation = "tanh".to_string();

        Self { dense, activation }
    }
}

impl Module for RobertaPooler {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Take the hidden state of the first token (CLS token)
        let first_token_tensor = hidden_states.select(1, 0)?;
        let pooled_output = self.dense.forward(&first_token_tensor)?;

        // Apply tanh activation
        Ok(pooled_output.tanh())
    }

    fn train(&mut self) {
        self.dense.train();
    }

    fn eval(&mut self) {
        self.dense.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.dense.parameters()
    }
}

// ========================================
// ROBERTA BASE MODEL
// ========================================

/// Complete RoBERTa base model
pub struct RobertaModel {
    config: RobertaConfig,
    embeddings: RobertaEmbeddings,
    encoder: RobertaEncoder,
    pooler: RobertaPooler,
}

impl RobertaModel {
    pub fn new(config: RobertaConfig) -> Self {
        config.validate().expect("Invalid RoBERTa configuration");

        let embeddings = RobertaEmbeddings::new(config.clone());
        let encoder = RobertaEncoder::new(config.clone());
        let pooler = RobertaPooler::new(&config);

        Self {
            config,
            embeddings,
            encoder,
            pooler,
        }
    }

    /// Create RoBERTa-base model
    pub fn roberta_base() -> Self {
        Self::new(RobertaConfig::roberta_base())
    }

    /// Create RoBERTa-large model
    pub fn roberta_large() -> Self {
        Self::new(RobertaConfig::roberta_large())
    }

    /// Get model configuration
    pub fn config(&self) -> &RobertaConfig {
        &self.config
    }
}

impl Module for RobertaModel {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embedding_output = self.embeddings.forward(input_ids)?;
        let encoder_output = self.encoder.forward(&embedding_output)?;
        let pooled_output = self.pooler.forward(&encoder_output)?;
        Ok(pooled_output)
    }

    fn train(&mut self) {
        self.embeddings.train();
        self.encoder.train();
        self.pooler.train();
    }

    fn eval(&mut self) {
        self.embeddings.eval();
        self.encoder.eval();
        self.pooler.eval();
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
}

// ========================================
// ROBERTA FOR SEQUENCE CLASSIFICATION
// ========================================

/// RoBERTa for sequence classification tasks
pub struct RobertaForSequenceClassification {
    roberta: RobertaModel,
    dropout: Dropout,
    classifier: Linear,
    num_labels: usize,
}

impl RobertaForSequenceClassification {
    pub fn new(config: RobertaConfig, num_labels: usize) -> Self {
        let roberta = RobertaModel::new(config.clone());
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let classifier = Linear::new(config.hidden_size, num_labels, true);

        Self {
            roberta,
            dropout,
            classifier,
            num_labels,
        }
    }

    /// Create RoBERTa-base for classification
    pub fn roberta_base_for_classification(num_labels: usize) -> Self {
        Self::new(RobertaConfig::roberta_base(), num_labels)
    }

    /// Create RoBERTa-large for classification
    pub fn roberta_large_for_classification(num_labels: usize) -> Self {
        Self::new(RobertaConfig::roberta_large(), num_labels)
    }

    /// Get number of classification labels
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }
}

impl Module for RobertaForSequenceClassification {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let pooled_output = self.roberta.forward(input_ids)?;
        let pooled_output = self.dropout.forward(&pooled_output)?;
        let logits = self.classifier.forward(&pooled_output)?;
        Ok(logits)
    }

    fn train(&mut self) {
        self.roberta.train();
        self.classifier.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.roberta.eval();
        self.classifier.eval();
        self.dropout.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.roberta.parameters() {
            params.insert(format!("roberta.{}", name), param);
        }

        for (name, param) in self.classifier.parameters() {
            params.insert(format!("classifier.{}", name), param);
        }

        params
    }
}

// ========================================
// ROBERTA UTILITIES AND FACTORY FUNCTIONS
// ========================================

/// RoBERTa model factory and utilities
pub struct RobertaFactory;

impl RobertaFactory {
    /// Create a RoBERTa model from configuration
    pub fn create_model(config: RobertaConfig) -> RobertaModel {
        RobertaModel::new(config)
    }

    /// Create a RoBERTa classification model
    pub fn create_classification_model(config: RobertaConfig, num_labels: usize) -> RobertaForSequenceClassification {
        RobertaForSequenceClassification::new(config, num_labels)
    }

    /// Get available RoBERTa model variants
    pub fn available_models() -> Vec<&'static str> {
        vec!["roberta-base", "roberta-large"]
    }

    /// Create model by name
    pub fn create_by_name(model_name: &str, num_labels: Option<usize>) -> Result<Box<dyn Module>> {
        match model_name {
            "roberta-base" => {
                if let Some(labels) = num_labels {
                    Ok(Box::new(RobertaForSequenceClassification::roberta_base_for_classification(labels)))
                } else {
                    Ok(Box::new(RobertaModel::roberta_base()))
                }
            },
            "roberta-large" => {
                if let Some(labels) = num_labels {
                    Ok(Box::new(RobertaForSequenceClassification::roberta_large_for_classification(labels)))
                } else {
                    Ok(Box::new(RobertaModel::roberta_large()))
                }
            },
            _ => Err(TorshError::InvalidArgument(
                format!("Unknown RoBERTa model: {}", model_name)
            ))
        }
    }
}

// ========================================
// COMPREHENSIVE TESTING FRAMEWORK
// ========================================

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::Device;

    #[test]
    fn test_roberta_config() {
        let config = RobertaConfig::roberta_base();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.num_attention_heads, 12);
        assert!(config.validate().is_ok());

        let large_config = RobertaConfig::roberta_large();
        assert_eq!(large_config.hidden_size, 1024);
        assert_eq!(large_config.num_hidden_layers, 24);
        assert!(large_config.validate().is_ok());
    }

    #[test]
    fn test_roberta_model_creation() {
        let config = RobertaConfig::roberta_base();
        let model = RobertaModel::new(config);
        assert_eq!(model.config().hidden_size, 768);

        let base_model = RobertaModel::roberta_base();
        assert_eq!(base_model.config().hidden_size, 768);

        let large_model = RobertaModel::roberta_large();
        assert_eq!(large_model.config().hidden_size, 1024);
    }

    #[test]
    fn test_roberta_classification() {
        let classifier = RobertaForSequenceClassification::roberta_base_for_classification(2);
        assert_eq!(classifier.num_labels(), 2);

        let large_classifier = RobertaForSequenceClassification::roberta_large_for_classification(5);
        assert_eq!(large_classifier.num_labels(), 5);
    }

    #[test]
    fn test_roberta_factory() {
        let available_models = RobertaFactory::available_models();
        assert!(available_models.contains(&"roberta-base"));
        assert!(available_models.contains(&"roberta-large"));

        // Test model creation by name
        let base_model = RobertaFactory::create_by_name("roberta-base", None);
        assert!(base_model.is_ok());

        let classification_model = RobertaFactory::create_by_name("roberta-base", Some(3));
        assert!(classification_model.is_ok());

        let invalid_model = RobertaFactory::create_by_name("invalid-model", None);
        assert!(invalid_model.is_err());
    }

    #[test]
    fn test_attention_head_size() {
        let config = RobertaConfig::roberta_base();
        assert_eq!(config.attention_head_size(), 64); // 768 / 12

        let large_config = RobertaConfig::roberta_large();
        assert_eq!(large_config.attention_head_size(), 64); // 1024 / 16
    }

    #[test]
    fn test_roberta_forward_pass() {
        let model = RobertaModel::roberta_base();

        // Create dummy input
        let batch_size = 2;
        let seq_length = 10;
        let input_ids = Tensor::zeros(&[batch_size, seq_length]).unwrap();

        // Test forward pass shape
        let output = model.forward(&input_ids);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.size(0).unwrap(), batch_size);
        assert_eq!(output.size(1).unwrap(), 768); // hidden_size
    }
}

/// Re-export commonly used RoBERTa components
pub use {
    RobertaConfig, RobertaModel, RobertaForSequenceClassification,
    RobertaFactory, RobertaEmbeddings, RobertaEncoder, RobertaLayer,
    RobertaAttention, RobertaSelfAttention
};