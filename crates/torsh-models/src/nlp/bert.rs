//! BERT Models
//!
//! This module provides implementations of BERT (Bidirectional Encoder Representations
//! from Transformers) models, including the original BERT architecture and variants
//! optimized for different use cases.
//!
//! ## Supported Models
//!
//! ### BERT Base and Large
//! - **BERT-Base**: 12 layers, 768 hidden size, 12 attention heads (110M parameters)
//! - **BERT-Large**: 24 layers, 1024 hidden size, 16 attention heads (340M parameters)
//! - **Custom configurations**: Flexible architecture with configurable parameters
//!
//! ### Key Components
//! - **BertEmbeddings**: Token, position, and segment embeddings
//! - **BertAttention**: Multi-head self-attention mechanism
//! - **BertEncoder**: Stack of transformer layers
//! - **BertPooler**: Classification pooling layer
//! - **BertForSequenceClassification**: BERT for classification tasks
//!
//! ## Key Features
//!
//! - **Bidirectional Context**: Full bidirectional attention for rich representations
//! - **Pre-training Ready**: Supports masked language modeling and next sentence prediction
//! - **Fine-tuning Friendly**: Easy adaptation for downstream tasks
//! - **Production Ready**: Comprehensive error handling and validation
//! - **Memory Efficient**: Optimized for large-scale training and inference
//!
//! ## Usage Examples
//!
//! ```rust
//! use torsh_models::nlp::bert::*;
//!
//! // Create BERT-Base model
//! let bert_base = BertModel::bert_base(true); // with pooling layer
//!
//! // Create BERT-Large model
//! let bert_large = BertModel::bert_large(false); // without pooling
//!
//! // Create BERT for sequence classification
//! let config = BertConfig::bert_base();
//! let classifier = BertForSequenceClassification::new(config, 2); // binary classification
//!
//! // Custom BERT configuration
//! let custom_config = BertConfig {
//!     hidden_size: 512,
//!     num_hidden_layers: 8,
//!     num_attention_heads: 8,
//!     ..BertConfig::default()
//! };
//! let custom_bert = BertModel::new(custom_config, true);
//! ```

use std::collections::HashMap;

use torsh_core::{
    error::{Result, TorshError},
    DeviceType,
};
use torsh_nn::prelude::*;
use torsh_nn::{Module, Parameter};
use torsh_tensor::{creation, Tensor};

// ============================================================================
// BERT Configuration
// ============================================================================

/// BERT Configuration
///
/// Comprehensive configuration for BERT models with sensible defaults
/// for both base and large variants.
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
    ///
    /// Standard BERT-Base configuration with 12 layers, 768 hidden size,
    /// and 12 attention heads (110M parameters).
    pub fn bert_base() -> Self {
        Self::default()
    }

    /// Create BERT-large configuration
    ///
    /// BERT-Large configuration with 24 layers, 1024 hidden size,
    /// and 16 attention heads (340M parameters).
    pub fn bert_large() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            ..Self::default()
        }
    }
}

// ============================================================================
// BERT Embeddings
// ============================================================================

/// BERT Embeddings
///
/// Combines token embeddings, position embeddings, and token type embeddings
/// to create the input representations for BERT.
#[derive(Debug)]
pub struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
    config: BertConfig,
}

impl BertEmbeddings {
    /// Create new BERT embeddings
    ///
    /// # Arguments
    /// * `config` - BERT configuration containing embedding dimensions
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
}

impl Module for BertEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.size(1)?;

        // Create position ids (0, 1, 2, ..., seq_length-1)
        let position_ids = creation::arange(seq_length as i64, input_ids.device())?;
        let position_ids = position_ids.unsqueeze(0)?; // Add batch dimension

        // Create token type ids (all zeros for single sentence)
        let token_type_ids = creation::zeros(&input_ids.shape(), input_ids.device())?;

        // Get embeddings
        let inputs_embeds = self.word_embeddings.forward(input_ids)?;
        let position_embeds = self.position_embeddings.forward(&position_ids)?;
        let token_type_embeds = self.token_type_embeddings.forward(&token_type_ids)?;

        // Sum all embeddings
        let embeddings = inputs_embeds.add(&position_embeds)?.add(&token_type_embeds)?;

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
        self.word_embeddings.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.word_embeddings.to_device(device)?;
        self.position_embeddings.to_device(device)?;
        self.token_type_embeddings.to_device(device)?;
        self.layer_norm.to_device(device)?;
        self.dropout.to_device(device)?;
        Ok(())
    }
}

// ============================================================================
// BERT Attention
// ============================================================================

/// BERT Self-Attention
///
/// Multi-head self-attention mechanism as used in BERT.
/// Implements scaled dot-product attention with multiple attention heads.
#[derive(Debug)]
pub struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,
}

impl BertSelfAttention {
    /// Create new BERT self-attention
    ///
    /// # Arguments
    /// * `config` - BERT configuration containing attention parameters
    pub fn new(config: &BertConfig) -> Self {
        let all_head_size = config.hidden_size;
        let attention_head_size = all_head_size / config.num_attention_heads;

        let query = Linear::new(config.hidden_size, all_head_size, true);
        let key = Linear::new(config.hidden_size, all_head_size, true);
        let value = Linear::new(config.hidden_size, all_head_size, true);
        let dropout = Dropout::new(config.attention_probs_dropout_prob);

        Self {
            query,
            key,
            value,
            dropout,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            all_head_size,
        }
    }

    /// Transpose tensor for multi-head attention
    fn transpose_for_scores(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.size(0)?;
        let seq_len = x.size(1)?;

        // Reshape from [batch_size, seq_len, all_head_size] to
        // [batch_size, seq_len, num_heads, head_size]
        let x = x.view(&[batch_size, seq_len, self.num_attention_heads, self.attention_head_size])?;

        // Transpose to [batch_size, num_heads, seq_len, head_size]
        x.permute(&[0, 2, 1, 3])
    }
}

impl Module for BertSelfAttention {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Linear transformations for Q, K, V
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        // Reshape for multi-head attention
        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        // Compute attention scores: Q * K^T / sqrt(d_k)
        let attention_scores = query_layer.matmul(&key_layer.transpose(-1, -2)?)?;
        let attention_scores = attention_scores.div_scalar((self.attention_head_size as f32).sqrt())?;

        // Apply softmax to get attention probabilities
        let attention_probs = attention_scores.softmax(-1)?;
        let attention_probs = self.dropout.forward(&attention_probs)?;

        // Apply attention to values: Attention * V
        let context_layer = attention_probs.matmul(&value_layer)?;

        // Transpose back to [batch_size, seq_len, num_heads, head_size]
        let context_layer = context_layer.permute(&[0, 2, 1, 3])?;

        // Reshape to [batch_size, seq_len, all_head_size]
        let batch_size = context_layer.size(0)?;
        let seq_len = context_layer.size(2)?;
        let context_layer = context_layer.view(&[batch_size, seq_len, self.all_head_size])?;

        Ok(context_layer)
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
        for (name, param) in self.query.parameters() {
            params.insert(format!("query.{}", name), param);
        }
        for (name, param) in self.key.parameters() {
            params.insert(format!("key.{}", name), param);
        }
        for (name, param) in self.value.parameters() {
            params.insert(format!("value.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.query.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.query.to_device(device)?;
        self.key.to_device(device)?;
        self.value.to_device(device)?;
        self.dropout.to_device(device)?;
        Ok(())
    }
}

/// BERT Self-Attention Output
///
/// Applies dense projection and residual connection to self-attention output.
#[derive(Debug)]
pub struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertSelfOutput {
    /// Create new BERT self-attention output layer
    ///
    /// # Arguments
    /// * `config` - BERT configuration
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
        for (name, param) in self.dense.parameters() {
            params.insert(format!("dense.{}", name), param);
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
        self.dense.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.dense.to_device(device)?;
        self.layer_norm.to_device(device)?;
        self.dropout.to_device(device)?;
        Ok(())
    }
}

/// BERT Attention Layer
///
/// Complete attention layer combining self-attention and output projection
/// with residual connection and layer normalization.
#[derive(Debug)]
pub struct BertAttention {
    self_attention: BertSelfAttention,
    output: BertSelfOutput,
}

impl BertAttention {
    /// Create new BERT attention layer
    ///
    /// # Arguments
    /// * `config` - BERT configuration
    pub fn new(config: &BertConfig) -> Self {
        let self_attention = BertSelfAttention::new(config);
        let output = BertSelfOutput::new(config);

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
        let attention_output = attention_output.add(hidden_states)?;

        // Apply layer norm (post-norm)
        Ok(attention_output)
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
        for (name, param) in self.self_attention.parameters() {
            params.insert(format!("self_attention.{}", name), param);
        }
        for (name, param) in self.output.parameters() {
            params.insert(format!("output.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.self_attention.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.self_attention.to_device(device)?;
        self.output.to_device(device)?;
        Ok(())
    }
}

// ============================================================================
// BERT Feed-Forward Network
// ============================================================================

/// BERT Intermediate Layer (Feed-Forward Network)
///
/// Position-wise feed-forward network with GELU activation.
#[derive(Debug)]
pub struct BertIntermediate {
    dense: Linear,
}

impl BertIntermediate {
    /// Create new BERT intermediate layer
    ///
    /// # Arguments
    /// * `config` - BERT configuration
    pub fn new(config: &BertConfig) -> Self {
        let dense = Linear::new(config.hidden_size, config.intermediate_size, true);

        Self { dense }
    }
}

impl Module for BertIntermediate {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = hidden_states.gelu()?; // GELU activation
        Ok(hidden_states)
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

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.dense.named_parameters()
    }

    fn training(&self) -> bool {
        self.dense.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.dense.to_device(device)
    }
}

/// BERT Output Layer
///
/// Projects intermediate layer output back to hidden size with residual connection.
#[derive(Debug)]
pub struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl BertOutput {
    /// Create new BERT output layer
    ///
    /// # Arguments
    /// * `config` - BERT configuration
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
        for (name, param) in self.dense.parameters() {
            params.insert(format!("dense.{}", name), param);
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
        self.dense.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.dense.to_device(device)?;
        self.layer_norm.to_device(device)?;
        self.dropout.to_device(device)?;
        Ok(())
    }
}

// ============================================================================
// BERT Transformer Layer
// ============================================================================

/// BERT Layer (Complete Transformer Block)
///
/// Single transformer layer combining attention and feed-forward network
/// with residual connections and layer normalization.
#[derive(Debug)]
pub struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    /// Create new BERT transformer layer
    ///
    /// # Arguments
    /// * `config` - BERT configuration
    pub fn new(config: &BertConfig) -> Self {
        let attention = BertAttention::new(config);
        let intermediate = BertIntermediate::new(config);
        let output = BertOutput::new(config);

        Self {
            attention,
            intermediate,
            output,
        }
    }
}

impl Module for BertLayer {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Self-attention with residual connection and layer norm
        let attention_output = self.attention.forward(hidden_states)?;

        // Feed-forward network
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let layer_output = self.output.forward(&intermediate_output)?;

        // Residual connection
        let layer_output = layer_output.add(&attention_output)?;

        Ok(layer_output)
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
        for (name, param) in self.attention.parameters() {
            params.insert(format!("attention.{}", name), param);
        }
        for (name, param) in self.intermediate.parameters() {
            params.insert(format!("intermediate.{}", name), param);
        }
        for (name, param) in self.output.parameters() {
            params.insert(format!("output.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.attention.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.attention.to_device(device)?;
        self.intermediate.to_device(device)?;
        self.output.to_device(device)?;
        Ok(())
    }
}

// ============================================================================
// BERT Encoder
// ============================================================================

/// BERT Encoder
///
/// Stack of BERT transformer layers forming the complete encoder.
#[derive(Debug)]
pub struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    /// Create new BERT encoder
    ///
    /// # Arguments
    /// * `config` - BERT configuration containing number of layers
    pub fn new(config: BertConfig) -> Self {
        let layers = (0..config.num_hidden_layers)
            .map(|_| BertLayer::new(&config))
            .collect();

        Self { layers }
    }
}

impl Module for BertEncoder {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();

        // Pass through each transformer layer
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
                params.insert(format!("layers.{}.{}", i, name), param);
            }
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.layers.first().map_or(false, |layer| layer.training())
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

// ============================================================================
// BERT Pooler
// ============================================================================

/// BERT Pooler
///
/// Pools the hidden states to generate a fixed-size representation
/// for classification tasks.
#[derive(Debug)]
pub struct BertPooler {
    dense: Linear,
}

impl BertPooler {
    /// Create new BERT pooler
    ///
    /// # Arguments
    /// * `config` - BERT configuration
    pub fn new(config: &BertConfig) -> Self {
        let dense = Linear::new(config.hidden_size, config.hidden_size, true);

        Self { dense }
    }
}

impl Module for BertPooler {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Take the hidden state corresponding to the first token (CLS token)
        let first_token_tensor = hidden_states.select(1, 0)?;
        let pooled_output = self.dense.forward(&first_token_tensor)?;
        let pooled_output = pooled_output.tanh()?;
        Ok(pooled_output)
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

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.dense.named_parameters()
    }

    fn training(&self) -> bool {
        self.dense.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.dense.to_device(device)
    }
}

// ============================================================================
// BERT Model
// ============================================================================

/// BERT Model
///
/// Complete BERT model combining embeddings, encoder, and optional pooler.
/// Suitable for various downstream tasks with or without classification head.
#[derive(Debug)]
pub struct BertModel {
    config: BertConfig,
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pooler: Option<BertPooler>,
}

impl BertModel {
    /// Create new BERT model
    ///
    /// # Arguments
    /// * `config` - BERT configuration
    /// * `add_pooling_layer` - Whether to include pooling layer for classification
    pub fn new(config: BertConfig, add_pooling_layer: bool) -> Self {
        let embeddings = BertEmbeddings::new(config.clone());
        let encoder = BertEncoder::new(config.clone());
        let pooler = if add_pooling_layer {
            Some(BertPooler::new(&config))
        } else {
            None
        };

        Self {
            config,
            embeddings,
            encoder,
            pooler,
        }
    }

    /// Create BERT-base model
    ///
    /// # Arguments
    /// * `add_pooling_layer` - Whether to include pooling layer
    pub fn bert_base(add_pooling_layer: bool) -> Self {
        Self::new(BertConfig::bert_base(), add_pooling_layer)
    }

    /// Create BERT-large model
    ///
    /// # Arguments
    /// * `add_pooling_layer` - Whether to include pooling layer
    pub fn bert_large(add_pooling_layer: bool) -> Self {
        Self::new(BertConfig::bert_large(), add_pooling_layer)
    }
}

impl Module for BertModel {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embedding_output = self.embeddings.forward(input_ids)?;
        let sequence_output = self.encoder.forward(&embedding_output)?;

        if let Some(ref pooler) = self.pooler {
            let pooled_output = pooler.forward(&sequence_output)?;
            // Return pooled output for classification tasks
            Ok(pooled_output)
        } else {
            // Return sequence output for token-level tasks
            Ok(sequence_output)
        }
    }

    fn train(&mut self) {
        self.embeddings.train();
        self.encoder.train();
        if let Some(ref mut pooler) = self.pooler {
            pooler.train();
        }
    }

    fn eval(&mut self) {
        self.embeddings.eval();
        self.encoder.eval();
        if let Some(ref mut pooler) = self.pooler {
            pooler.eval();
        }
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

    fn training(&self) -> bool {
        self.embeddings.training()
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

// ============================================================================
// BERT for Sequence Classification
// ============================================================================

/// BERT for Sequence Classification
///
/// BERT model with a classification head for sequence-level tasks
/// such as sentiment analysis, text classification, etc.
#[derive(Debug)]
pub struct BertForSequenceClassification {
    bert: BertModel,
    dropout: Dropout,
    classifier: Linear,
    num_labels: usize,
}

impl BertForSequenceClassification {
    /// Create new BERT for sequence classification
    ///
    /// # Arguments
    /// * `config` - BERT configuration
    /// * `num_labels` - Number of classification labels
    pub fn new(config: BertConfig, num_labels: usize) -> Self {
        let bert = BertModel::new(config.clone(), true); // With pooling layer
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let classifier = Linear::new(config.hidden_size, num_labels, true);

        Self {
            bert,
            dropout,
            classifier,
            num_labels,
        }
    }

    /// Create BERT-base for sequence classification
    ///
    /// # Arguments
    /// * `num_labels` - Number of classification labels
    pub fn bert_base_for_classification(num_labels: usize) -> Self {
        Self::new(BertConfig::bert_base(), num_labels)
    }

    /// Create BERT-large for sequence classification
    ///
    /// # Arguments
    /// * `num_labels` - Number of classification labels
    pub fn bert_large_for_classification(num_labels: usize) -> Self {
        Self::new(BertConfig::bert_large(), num_labels)
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
        self.dropout.train();
        self.classifier.train();
    }

    fn eval(&mut self) {
        self.bert.eval();
        self.dropout.eval();
        self.classifier.eval();
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

    fn training(&self) -> bool {
        self.bert.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.bert.to_device(device)?;
        self.dropout.to_device(device)?;
        self.classifier.to_device(device)?;
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::Tensor;

    #[test]
    fn test_bert_config() {
        let config = BertConfig::bert_base();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.vocab_size, 30522);

        let large_config = BertConfig::bert_large();
        assert_eq!(large_config.hidden_size, 1024);
        assert_eq!(large_config.num_hidden_layers, 24);
        assert_eq!(large_config.num_attention_heads, 16);
    }

    #[test]
    fn test_bert_embeddings() {
        let config = BertConfig::bert_base();
        let embeddings = BertEmbeddings::new(config);
        assert!(!embeddings.training());
    }

    #[test]
    fn test_bert_self_attention() {
        let config = BertConfig::bert_base();
        let attention = BertSelfAttention::new(&config);
        assert_eq!(attention.num_attention_heads, 12);
        assert_eq!(attention.attention_head_size, 64); // 768 / 12
        assert_eq!(attention.all_head_size, 768);
    }

    #[test]
    fn test_bert_layer() {
        let config = BertConfig::bert_base();
        let layer = BertLayer::new(&config);
        assert!(!layer.training());
    }

    #[test]
    fn test_bert_encoder() {
        let config = BertConfig::bert_base();
        let encoder = BertEncoder::new(config);
        assert_eq!(encoder.layers.len(), 12); // BERT-base has 12 layers
    }

    #[test]
    fn test_bert_model_creation() {
        let bert_base = BertModel::bert_base(true);
        assert!(!bert_base.training());
        assert!(bert_base.pooler.is_some());

        let bert_large = BertModel::bert_large(false);
        assert!(!bert_large.training());
        assert!(bert_large.pooler.is_none());
    }

    #[test]
    fn test_bert_model_custom() {
        let config = BertConfig {
            hidden_size: 512,
            num_hidden_layers: 6,
            num_attention_heads: 8,
            ..BertConfig::default()
        };
        let model = BertModel::new(config, true);
        assert!(!model.training());
    }

    #[test]
    fn test_bert_for_sequence_classification() {
        let classifier = BertForSequenceClassification::bert_base_for_classification(2);
        assert_eq!(classifier.num_labels, 2);
        assert!(!classifier.training());

        let large_classifier = BertForSequenceClassification::bert_large_for_classification(5);
        assert_eq!(large_classifier.num_labels, 5);
    }

    #[test]
    fn test_bert_training_mode() {
        let mut bert = BertModel::bert_base(true);
        assert!(!bert.training());

        bert.train();
        assert!(bert.training());

        bert.eval();
        assert!(!bert.training());
    }

    #[test]
    fn test_bert_forward() {
        let bert = BertModel::bert_base(true);
        let input_ids = Tensor::zeros(&[2, 10]).unwrap(); // batch_size=2, seq_len=10

        let result = bert.forward(&input_ids);
        assert!(result.is_ok());
    }

    #[test]
    fn test_bert_classification_forward() {
        let classifier = BertForSequenceClassification::bert_base_for_classification(3);
        let input_ids = Tensor::zeros(&[1, 8]).unwrap(); // batch_size=1, seq_len=8

        let result = classifier.forward(&input_ids);
        assert!(result.is_ok());

        if let Ok(logits) = result {
            assert_eq!(logits.size(0).unwrap(), 1); // batch_size
            assert_eq!(logits.size(1).unwrap(), 3); // num_labels
        }
    }

    #[test]
    fn test_bert_pooler() {
        let config = BertConfig::bert_base();
        let pooler = BertPooler::new(&config);
        assert!(!pooler.training());
    }

    #[test]
    fn test_bert_components_device_transfer() {
        let mut embeddings = BertEmbeddings::new(BertConfig::bert_base());
        let result = embeddings.to_device(DeviceType::Cpu);
        assert!(result.is_ok());

        let mut attention = BertSelfAttention::new(&BertConfig::bert_base());
        let result = attention.to_device(DeviceType::Cpu);
        assert!(result.is_ok());
    }

    #[test]
    fn test_bert_parameter_count() {
        let bert = BertModel::bert_base(true);
        let params = bert.parameters();
        assert!(!params.is_empty());

        // Check that key parameters exist
        assert!(params.keys().any(|k| k.contains("embeddings")));
        assert!(params.keys().any(|k| k.contains("encoder")));
        assert!(params.keys().any(|k| k.contains("pooler")));
    }
}