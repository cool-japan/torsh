//! BERT (Bidirectional Encoder Representations from Transformers) Models
//!
//! This module contains comprehensive implementations of BERT transformer models,
//! extracted from the massive monolithic nlp.rs file as part of Phase 14 systematic refactoring.
//!
//! # BERT Architecture Components
//!
//! - `BertConfig` - Model configuration with base/large variants
//! - `BertEmbeddings` - Token, position, and segment embeddings
//! - `BertSelfAttention` - Multi-head self-attention mechanism
//! - `BertSelfOutput` - Self-attention output projection
//! - `BertAttention` - Complete attention layer (self-attention + output)
//! - `BertIntermediate` - Feed-forward intermediate layer
//! - `BertOutput` - Feed-forward output layer with residual connections
//! - `BertLayer` - Complete transformer layer (attention + feed-forward)
//! - `BertEncoder` - Stack of BERT layers
//! - `BertPooler` - Pooling layer for sequence-level representations
//! - `BertModel` - Complete BERT base model
//! - `BertForSequenceClassification` - BERT with classification head
//!
//! # Key Differences from RoBERTa
//!
//! - Uses segment embeddings (token type embeddings)
//! - Different default vocabulary size (30522 vs 50265)
//! - Different token IDs (CLS, SEP, MASK tokens)
//! - Slightly different layer norm epsilon (1e-12 vs 1e-5)
//!
//! # Usage Examples
//!
//! ```rust
//! use crate::models::bert::*;
//!
//! // Create BERT-base model
//! let config = BertConfig::bert_base();
//! let model = BertModel::new(config);
//!
//! // Create BERT for sequence classification
//! let classifier = BertForSequenceClassification::bert_base_for_classification(2);
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
// BERT CONFIGURATION
// ========================================

/// Configuration for BERT model
#[derive(Debug, Clone)]
pub struct BertConfig {
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
    pub cls_token_id: usize,
    pub sep_token_id: usize,
    pub mask_token_id: usize,
}

impl Default for BertConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            cls_token_id: 101,
            sep_token_id: 102,
            mask_token_id: 103,
        }
    }
}

impl BertConfig {
    /// Create BERT-base configuration
    pub fn bert_base() -> Self {
        Self::default()
    }

    /// Create BERT-large configuration
    pub fn bert_large() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            ..Self::default()
        }
    }

    /// Create BERT-small configuration (for experimentation)
    pub fn bert_small() -> Self {
        Self {
            hidden_size: 256,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            intermediate_size: 1024,
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

        if self.type_vocab_size < 2 {
            return Err(TorshError::InvalidArgument(
                "type_vocab_size must be at least 2 for BERT".to_string()
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
// BERT EMBEDDINGS
// ========================================

/// BERT Embeddings with segment (token type) embeddings
pub struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    config: BertConfig,
}

impl BertEmbeddings {
    pub fn new(config: BertConfig) -> Self {
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

        let position_ids = Tensor::arange(0.0, seq_length as f32, 1.0, input_ids.device())?;
        Ok(position_ids.unsqueeze(0)?.expand(&[input_ids.size(0)?, seq_length])?)
    }

    /// Create default token type IDs (all zeros for single sequence)
    fn create_token_type_ids(&self, input_ids: &Tensor) -> Result<Tensor> {
        let batch_size = input_ids.size(0)?;
        let seq_length = input_ids.size(1)?;
        Tensor::zeros(&[batch_size, seq_length])
    }
}

impl Module for BertEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let position_ids = self.create_position_ids(input_ids)?;
        let token_type_ids = self.create_token_type_ids(input_ids)?;

        // Get embeddings
        let word_embeddings = self.word_embeddings.forward(input_ids)?;
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;
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
// BERT SELF-ATTENTION
// ========================================

/// BERT Self-Attention mechanism
pub struct BertSelfAttention {
    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    config: BertConfig,
}

impl BertSelfAttention {
    pub fn new(config: BertConfig) -> Self {
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

impl Module for BertSelfAttention {
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
// BERT SELF-ATTENTION OUTPUT
// ========================================

/// BERT Self-Attention Output projection
pub struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertSelfOutput {
    pub fn new(config: &BertConfig) -> Self {
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

impl Module for BertSelfOutput {
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
// BERT ATTENTION LAYER
// ========================================

/// Complete BERT Attention layer (self-attention + output)
pub struct BertAttention {
    self_attention: BertSelfAttention,
    output: BertSelfOutput,
}

impl BertAttention {
    pub fn new(config: BertConfig) -> Self {
        let self_attention = BertSelfAttention::new(config.clone());
        let output = BertSelfOutput::new(&config);

        Self {
            self_attention,
            output,
        }
    }
}

impl Module for BertAttention {
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
// BERT FEED-FORWARD LAYERS
// ========================================

/// BERT Intermediate (Feed-Forward) layer
pub struct BertIntermediate {
    dense: Linear,
    activation: String,
}

impl BertIntermediate {
    pub fn new(config: &BertConfig) -> Self {
        let dense = Linear::new(config.hidden_size, config.intermediate_size, true);
        let activation = config.hidden_act.clone();

        Self { dense, activation }
    }

    fn apply_activation(&self, input: &Tensor) -> Result<Tensor> {
        match self.activation.as_str() {
            "gelu" => input.gelu(),
            "relu" => input.relu(),
            "tanh" => Ok(input.tanh()),
            "swish" => input.swish(),
            _ => input.gelu(), // Default to GELU
        }
    }
}

impl Module for BertIntermediate {
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

/// BERT Output (Feed-Forward output) layer
pub struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertOutput {
    pub fn new(config: &BertConfig) -> Self {
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

impl Module for BertOutput {
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
// BERT TRANSFORMER LAYER
// ========================================

/// Complete BERT Transformer layer
pub struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    pub fn new(config: BertConfig) -> Self {
        let attention = BertAttention::new(config.clone());
        let intermediate = BertIntermediate::new(&config);
        let output = BertOutput::new(&config);

        Self {
            attention,
            intermediate,
            output,
        }
    }
}

impl Module for BertLayer {
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
// BERT ENCODER
// ========================================

/// BERT Encoder (stack of transformer layers)
pub struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    pub fn new(config: BertConfig) -> Self {
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(BertLayer::new(config.clone()));
        }

        Self { layers }
    }
}

impl Module for BertEncoder {
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
// BERT POOLER
// ========================================

/// BERT Pooler for sequence-level representations
pub struct BertPooler {
    dense: Linear,
    activation: String,
}

impl BertPooler {
    pub fn new(config: &BertConfig) -> Self {
        let dense = Linear::new(config.hidden_size, config.hidden_size, true);
        let activation = "tanh".to_string();

        Self { dense, activation }
    }
}

impl Module for BertPooler {
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
// BERT BASE MODEL
// ========================================

/// Complete BERT base model
pub struct BertModel {
    config: BertConfig,
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pooler: BertPooler,
}

impl BertModel {
    pub fn new(config: BertConfig) -> Self {
        config.validate().expect("Invalid BERT configuration");

        let embeddings = BertEmbeddings::new(config.clone());
        let encoder = BertEncoder::new(config.clone());
        let pooler = BertPooler::new(&config);

        Self {
            config,
            embeddings,
            encoder,
            pooler,
        }
    }

    /// Create BERT-base model
    pub fn bert_base() -> Self {
        Self::new(BertConfig::bert_base())
    }

    /// Create BERT-large model
    pub fn bert_large() -> Self {
        Self::new(BertConfig::bert_large())
    }

    /// Create BERT-small model (for experimentation)
    pub fn bert_small() -> Self {
        Self::new(BertConfig::bert_small())
    }

    /// Get model configuration
    pub fn config(&self) -> &BertConfig {
        &self.config
    }
}

impl Module for BertModel {
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
// BERT FOR SEQUENCE CLASSIFICATION
// ========================================

/// BERT for sequence classification tasks
pub struct BertForSequenceClassification {
    bert: BertModel,
    dropout: Dropout,
    classifier: Linear,
    num_labels: usize,
}

impl BertForSequenceClassification {
    pub fn new(config: BertConfig, num_labels: usize) -> Self {
        let bert = BertModel::new(config.clone());
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let classifier = Linear::new(config.hidden_size, num_labels, true);

        Self {
            bert,
            dropout,
            classifier,
            num_labels,
        }
    }

    /// Create BERT-base for classification
    pub fn bert_base_for_classification(num_labels: usize) -> Self {
        Self::new(BertConfig::bert_base(), num_labels)
    }

    /// Create BERT-large for classification
    pub fn bert_large_for_classification(num_labels: usize) -> Self {
        Self::new(BertConfig::bert_large(), num_labels)
    }

    /// Create BERT-small for classification (experimentation)
    pub fn bert_small_for_classification(num_labels: usize) -> Self {
        Self::new(BertConfig::bert_small(), num_labels)
    }

    /// Get number of classification labels
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }
}

impl Module for BertForSequenceClassification {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let pooled_output = self.bert.forward(input_ids)?;
        let pooled_output = self.dropout.forward(&pooled_output)?;
        let logits = self.classifier.forward(&pooled_output)?;
        Ok(logits)
    }

    fn train(&mut self) {
        self.bert.train();
        self.classifier.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.bert.eval();
        self.classifier.eval();
        self.dropout.eval();
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
}

// ========================================
// BERT UTILITIES AND FACTORY FUNCTIONS
// ========================================

/// BERT model factory and utilities
pub struct BertFactory;

impl BertFactory {
    /// Create a BERT model from configuration
    pub fn create_model(config: BertConfig) -> BertModel {
        BertModel::new(config)
    }

    /// Create a BERT classification model
    pub fn create_classification_model(config: BertConfig, num_labels: usize) -> BertForSequenceClassification {
        BertForSequenceClassification::new(config, num_labels)
    }

    /// Get available BERT model variants
    pub fn available_models() -> Vec<&'static str> {
        vec!["bert-base", "bert-large", "bert-small"]
    }

    /// Create model by name
    pub fn create_by_name(model_name: &str, num_labels: Option<usize>) -> Result<Box<dyn Module>> {
        match model_name {
            "bert-base" => {
                if let Some(labels) = num_labels {
                    Ok(Box::new(BertForSequenceClassification::bert_base_for_classification(labels)))
                } else {
                    Ok(Box::new(BertModel::bert_base()))
                }
            },
            "bert-large" => {
                if let Some(labels) = num_labels {
                    Ok(Box::new(BertForSequenceClassification::bert_large_for_classification(labels)))
                } else {
                    Ok(Box::new(BertModel::bert_large()))
                }
            },
            "bert-small" => {
                if let Some(labels) = num_labels {
                    Ok(Box::new(BertForSequenceClassification::bert_small_for_classification(labels)))
                } else {
                    Ok(Box::new(BertModel::bert_small()))
                }
            },
            _ => Err(TorshError::InvalidArgument(
                format!("Unknown BERT model: {}", model_name)
            ))
        }
    }

    /// Compare BERT and RoBERTa configurations
    pub fn compare_with_roberta() -> Vec<(&'static str, &'static str, &'static str)> {
        vec![
            ("Vocabulary Size", "30,522", "50,265"),
            ("Token Type Embeddings", "Yes (2 types)", "Yes (1 type)"),
            ("LayerNorm Epsilon", "1e-12", "1e-5"),
            ("Special Tokens", "CLS, SEP, MASK", "BOS, EOS, PAD"),
            ("Max Position Embeddings", "512", "514"),
            ("Default Architecture", "Encoder-only", "Encoder-only"),
        ]
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
    fn test_bert_config() {
        let config = BertConfig::bert_base();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.vocab_size, 30522);
        assert!(config.validate().is_ok());

        let large_config = BertConfig::bert_large();
        assert_eq!(large_config.hidden_size, 1024);
        assert_eq!(large_config.num_hidden_layers, 24);
        assert!(large_config.validate().is_ok());

        let small_config = BertConfig::bert_small();
        assert_eq!(small_config.hidden_size, 256);
        assert_eq!(small_config.num_hidden_layers, 4);
        assert!(small_config.validate().is_ok());
    }

    #[test]
    fn test_bert_model_creation() {
        let config = BertConfig::bert_base();
        let model = BertModel::new(config);
        assert_eq!(model.config().hidden_size, 768);

        let base_model = BertModel::bert_base();
        assert_eq!(base_model.config().hidden_size, 768);

        let large_model = BertModel::bert_large();
        assert_eq!(large_model.config().hidden_size, 1024);

        let small_model = BertModel::bert_small();
        assert_eq!(small_model.config().hidden_size, 256);
    }

    #[test]
    fn test_bert_classification() {
        let classifier = BertForSequenceClassification::bert_base_for_classification(2);
        assert_eq!(classifier.num_labels(), 2);

        let large_classifier = BertForSequenceClassification::bert_large_for_classification(5);
        assert_eq!(large_classifier.num_labels(), 5);

        let small_classifier = BertForSequenceClassification::bert_small_for_classification(3);
        assert_eq!(small_classifier.num_labels(), 3);
    }

    #[test]
    fn test_bert_factory() {
        let available_models = BertFactory::available_models();
        assert!(available_models.contains(&"bert-base"));
        assert!(available_models.contains(&"bert-large"));
        assert!(available_models.contains(&"bert-small"));

        // Test model creation by name
        let base_model = BertFactory::create_by_name("bert-base", None);
        assert!(base_model.is_ok());

        let classification_model = BertFactory::create_by_name("bert-large", Some(3));
        assert!(classification_model.is_ok());

        let invalid_model = BertFactory::create_by_name("invalid-model", None);
        assert!(invalid_model.is_err());
    }

    #[test]
    fn test_bert_token_ids() {
        let config = BertConfig::bert_base();
        assert_eq!(config.cls_token_id, 101);
        assert_eq!(config.sep_token_id, 102);
        assert_eq!(config.mask_token_id, 103);
        assert_eq!(config.pad_token_id, 0);
    }

    #[test]
    fn test_attention_head_size() {
        let config = BertConfig::bert_base();
        assert_eq!(config.attention_head_size(), 64); // 768 / 12

        let large_config = BertConfig::bert_large();
        assert_eq!(large_config.attention_head_size(), 64); // 1024 / 16

        let small_config = BertConfig::bert_small();
        assert_eq!(small_config.attention_head_size(), 64); // 256 / 4
    }

    #[test]
    fn test_bert_forward_pass() {
        let model = BertModel::bert_base();

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

    #[test]
    fn test_bert_roberta_comparison() {
        let comparison = BertFactory::compare_with_roberta();
        assert!(!comparison.is_empty());
        assert_eq!(comparison.len(), 6);
    }
}

/// Re-export commonly used BERT components
pub use {
    BertConfig, BertModel, BertForSequenceClassification,
    BertFactory, BertEmbeddings, BertEncoder, BertLayer,
    BertAttention, BertSelfAttention
};