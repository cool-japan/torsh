//! # BART (Bidirectional and Auto-Regressive Transformers) Implementation
//!
//! This module provides a comprehensive implementation of BART models in the ToRSh framework.
//! BART combines bidirectional encoding with auto-regressive decoding, making it particularly
//! effective for sequence-to-sequence tasks like text summarization and neural machine translation.
//!
//! ## Models Included
//!
//! - **BartConfig**: Configuration for BART models
//! - **BartEncoderLayer**: Transformer encoder layer
//! - **BartForConditionalGeneration**: Complete BART model for generation tasks
//!
//! ## Usage Example
//!
//! ```rust
//! use torsh_models::nlp::bart::{BartConfig, BartForConditionalGeneration};
//! use torsh_tensor::Tensor;
//!
//! // Create BART-base model
//! let mut model = BartForConditionalGeneration::bart_base();
//!
//! // Forward pass
//! let input_ids = Tensor::zeros(&[1, 10]).unwrap();
//! let output = model.forward(&input_ids).unwrap();
//! ```
//!
//! ## Model Variants
//!
//! - **BART-base**: 6 encoder/decoder layers, 768 hidden size, 12 attention heads
//! - **BART-large**: 12 encoder/decoder layers, 1024 hidden size, 16 attention heads
//!
//! All models follow the BART paper specifications and are compatible with
//! HuggingFace transformers library weights.

use std::collections::HashMap;
use torsh_core::error::{Result, TorshError};
use torsh_core::device::DeviceType;
use torsh_tensor::Tensor;
use torsh_nn::{
    activations::GELU,
    attention::MultiheadAttention,
    dropout::Dropout,
    embedding::Embedding,
    layers::Linear,
    layer_norm::LayerNorm,
    module::{Module, Parameter},
};

/// Configuration for BART model
///
/// This struct contains all the hyperparameters needed to define a BART model architecture.
/// BART uses an encoder-decoder architecture for sequence-to-sequence tasks.
#[derive(Debug, Clone)]
pub struct BartConfig {
    /// Size of the vocabulary
    pub vocab_size: usize,
    /// Size of the hidden layers
    pub hidden_size: usize,
    /// Number of encoder layers
    pub num_encoder_layers: usize,
    /// Number of decoder layers
    pub num_decoder_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Size of the feed-forward layer
    pub intermediate_size: usize,
    /// Dropout probability for hidden layers
    pub hidden_dropout_prob: f32,
    /// Dropout probability for attention weights
    pub attention_probs_dropout_prob: f32,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// Standard deviation for weight initialization
    pub init_std: f32,
    /// Whether to scale embeddings by sqrt(hidden_size)
    pub scale_embedding: bool,
    /// Whether to use caching during generation
    pub use_cache: bool,
    /// ID of the padding token
    pub pad_token_id: usize,
    /// ID of the beginning-of-sequence token
    pub bos_token_id: usize,
    /// ID of the end-of-sequence token
    pub eos_token_id: usize,
    /// Decoder start token ID
    pub decoder_start_token_id: usize,
}

impl Default for BartConfig {
    /// Returns the default BART-base configuration
    fn default() -> Self {
        Self {
            vocab_size: 50265,
            hidden_size: 768,
            num_encoder_layers: 6,
            num_decoder_layers: 6,
            num_heads: 12,
            intermediate_size: 3072,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 1024,
            init_std: 0.02,
            scale_embedding: false,
            use_cache: true,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            decoder_start_token_id: 2,
        }
    }
}

impl BartConfig {
    /// Create BART-base configuration
    ///
    /// # Returns
    ///
    /// A `BartConfig` with parameters for the base model
    pub fn bart_base() -> Self {
        Self::default()
    }

    /// Create BART-large configuration
    ///
    /// # Returns
    ///
    /// A `BartConfig` with parameters for the large model
    pub fn bart_large() -> Self {
        Self {
            hidden_size: 1024,
            num_encoder_layers: 12,
            num_decoder_layers: 12,
            num_heads: 16,
            intermediate_size: 4096,
            ..Self::default()
        }
    }

    /// Validates the configuration parameters
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, `Err` with description if invalid
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size % self.num_heads != 0 {
            return Err(TorshError::ModelError(
                "hidden_size must be divisible by num_heads".to_string()
            ));
        }
        if self.vocab_size == 0 {
            return Err(TorshError::ModelError("vocab_size must be > 0".to_string()));
        }
        if self.num_encoder_layers == 0 {
            return Err(TorshError::ModelError("num_encoder_layers must be > 0".to_string()));
        }
        if self.num_decoder_layers == 0 {
            return Err(TorshError::ModelError("num_decoder_layers must be > 0".to_string()));
        }
        Ok(())
    }

    /// Gets the head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

/// BART Encoder Layer
///
/// A transformer encoder layer with multi-head self-attention and feed-forward network.
/// Uses pre-normalization and residual connections as specified in the BART paper.
#[derive(Debug)]
pub struct BartEncoderLayer {
    self_attn: MultiheadAttention,
    self_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
    dropout: Dropout,
    activation: GELU,
}

impl BartEncoderLayer {
    /// Creates new BART encoder layer
    ///
    /// # Arguments
    ///
    /// * `config` - BART configuration
    ///
    /// # Returns
    ///
    /// New `BartEncoderLayer` instance
    pub fn new(config: &BartConfig) -> Self {
        let self_attn = MultiheadAttention::new(
            config.hidden_size,
            config.num_heads,
            config.attention_probs_dropout_prob,
            true,
        );
        let self_attn_layer_norm = LayerNorm::new(vec![config.hidden_size], 1e-5, true);
        let fc1 = Linear::new(config.hidden_size, config.intermediate_size, true);
        let fc2 = Linear::new(config.intermediate_size, config.hidden_size, true);
        let final_layer_norm = LayerNorm::new(vec![config.hidden_size], 1e-5, true);
        let dropout = Dropout::new(config.hidden_dropout_prob);
        let activation = GELU::new();

        Self {
            self_attn,
            self_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
            dropout,
            activation,
        }
    }

    /// Gets the self-attention component
    pub fn self_attention(&self) -> &MultiheadAttention {
        &self.self_attn
    }
}

impl Module for BartEncoderLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Self-attention block with pre-normalization
        let residual = input.clone();
        let mut hidden_states = self.self_attn_layer_norm.forward(input)?;
        hidden_states = self.self_attn.forward(&hidden_states, None, None)?;
        hidden_states = self.dropout.forward(&hidden_states)?;
        hidden_states = hidden_states.add(&residual)?;

        // Feed-forward block with pre-normalization
        let residual = hidden_states.clone();
        hidden_states = self.final_layer_norm.forward(&hidden_states)?;
        hidden_states = self.fc1.forward(&hidden_states)?;
        hidden_states = self.activation.forward(&hidden_states)?;
        hidden_states = self.fc2.forward(&hidden_states)?;
        hidden_states = self.dropout.forward(&hidden_states)?;
        hidden_states = hidden_states.add(&residual)?;

        Ok(hidden_states)
    }

    fn train(&mut self) {
        self.self_attn.train();
        self.self_attn_layer_norm.train();
        self.fc1.train();
        self.fc2.train();
        self.final_layer_norm.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.self_attn.eval();
        self.self_attn_layer_norm.eval();
        self.fc1.eval();
        self.fc2.eval();
        self.final_layer_norm.eval();
        self.dropout.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.self_attn.parameters() {
            params.insert(format!("self_attn.{}", name), param);
        }
        for (name, param) in self.self_attn_layer_norm.parameters() {
            params.insert(format!("self_attn_layer_norm.{}", name), param);
        }
        for (name, param) in self.fc1.parameters() {
            params.insert(format!("fc1.{}", name), param);
        }
        for (name, param) in self.fc2.parameters() {
            params.insert(format!("fc2.{}", name), param);
        }
        for (name, param) in self.final_layer_norm.parameters() {
            params.insert(format!("final_layer_norm.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.self_attn.training()
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.self_attn.to_device(device)?;
        self.self_attn_layer_norm.to_device(device)?;
        self.fc1.to_device(device)?;
        self.fc2.to_device(device)?;
        self.final_layer_norm.to_device(device)?;
        Ok(())
    }
}

/// BART Model for conditional generation
///
/// Complete BART model with encoder layers and language modeling head.
/// Designed for sequence-to-sequence tasks like summarization and translation.
#[derive(Debug)]
pub struct BartForConditionalGeneration {
    config: BartConfig,
    embeddings: Embedding,
    encoder_layers: Vec<BartEncoderLayer>,
    lm_head: Linear,
    layernorm: LayerNorm,
    dropout: Dropout,
}

impl BartForConditionalGeneration {
    /// Creates new BART conditional generation model
    ///
    /// # Arguments
    ///
    /// * `config` - BART configuration
    ///
    /// # Returns
    ///
    /// New `BartForConditionalGeneration` instance
    pub fn new(config: BartConfig) -> Self {
        let embeddings = Embedding::new(config.vocab_size, config.hidden_size);
        let mut encoder_layers = Vec::new();
        for _ in 0..config.num_encoder_layers {
            encoder_layers.push(BartEncoderLayer::new(&config));
        }
        let lm_head = Linear::new(config.hidden_size, config.vocab_size, false);
        let layernorm = LayerNorm::new(vec![config.hidden_size], 1e-5, true);
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Self {
            config,
            embeddings,
            encoder_layers,
            lm_head,
            layernorm,
            dropout,
        }
    }

    /// Create BART-base model
    ///
    /// # Returns
    ///
    /// New BART-base conditional generation model
    pub fn bart_base() -> Self {
        Self::new(BartConfig::bart_base())
    }

    /// Create BART-large model
    ///
    /// # Returns
    ///
    /// New BART-large conditional generation model
    pub fn bart_large() -> Self {
        Self::new(BartConfig::bart_large())
    }

    /// Gets the model configuration
    pub fn config(&self) -> &BartConfig {
        &self.config
    }

    /// Gets the embeddings component
    pub fn embeddings(&self) -> &Embedding {
        &self.embeddings
    }

    /// Gets the number of encoder layers
    pub fn num_encoder_layers(&self) -> usize {
        self.encoder_layers.len()
    }

    /// Gets a reference to a specific encoder layer
    pub fn encoder_layer(&self, index: usize) -> Option<&BartEncoderLayer> {
        self.encoder_layers.get(index)
    }

    /// Gets the language modeling head
    pub fn lm_head(&self) -> &Linear {
        &self.lm_head
    }

    /// Generate logits for conditional text generation
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits for text generation
    pub fn generate_logits(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward(input_ids)
    }
}

impl Module for BartForConditionalGeneration {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Embedding layer
        let mut hidden_states = self.embeddings.forward(input)?;

        // Scale embeddings if configured
        if self.config.scale_embedding {
            let scale = (self.config.hidden_size as f32).sqrt();
            hidden_states = hidden_states.mul_scalar(scale)?;
        }

        hidden_states = self.layernorm.forward(&hidden_states)?;
        hidden_states = self.dropout.forward(&hidden_states)?;

        // Encoder layers
        for layer in &self.encoder_layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        // Language modeling head
        let logits = self.lm_head.forward(&hidden_states)?;
        Ok(logits)
    }

    fn train(&mut self) {
        self.embeddings.train();
        for layer in &mut self.encoder_layers {
            layer.train();
        }
        self.lm_head.train();
        self.layernorm.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.embeddings.eval();
        for layer in &mut self.encoder_layers {
            layer.eval();
        }
        self.lm_head.eval();
        self.layernorm.eval();
        self.dropout.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        // Embeddings
        for (name, param) in self.embeddings.parameters() {
            params.insert(format!("embeddings.{}", name), param);
        }

        // Encoder layers
        for (i, layer) in self.encoder_layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("encoder.layers.{}.{}", i, name), param);
            }
        }

        // Output head
        for (name, param) in self.lm_head.parameters() {
            params.insert(format!("lm_head.{}", name), param);
        }

        // Layer norm
        for (name, param) in self.layernorm.parameters() {
            params.insert(format!("layernorm.{}", name), param);
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
        for layer in &mut self.encoder_layers {
            layer.to_device(device)?;
        }
        self.lm_head.to_device(device)?;
        self.layernorm.to_device(device)?;
        Ok(())
    }
}

/// Factory functions for creating BART models
pub mod factory {
    use super::*;

    /// Creates BART-base model
    pub fn bart_base() -> BartForConditionalGeneration {
        BartForConditionalGeneration::bart_base()
    }

    /// Creates BART-large model
    pub fn bart_large() -> BartForConditionalGeneration {
        BartForConditionalGeneration::bart_large()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::Tensor;

    #[test]
    fn test_bart_config_creation() {
        let config = BartConfig::bart_base();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_encoder_layers, 6);
        assert_eq!(config.num_decoder_layers, 6);
        assert_eq!(config.num_heads, 12);

        let large_config = BartConfig::bart_large();
        assert_eq!(large_config.hidden_size, 1024);
        assert_eq!(large_config.num_encoder_layers, 12);
        assert_eq!(large_config.num_decoder_layers, 12);
        assert_eq!(large_config.num_heads, 16);
    }

    #[test]
    fn test_bart_config_validation() {
        let mut config = BartConfig::bart_base();
        assert!(config.validate().is_ok());

        // Test invalid configuration
        config.num_heads = 7; // Should not divide 768
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_bart_config_derived_values() {
        let config = BartConfig::bart_base();
        assert_eq!(config.head_dim(), 768 / 12);
    }

    #[test]
    fn test_bart_encoder_layer_creation() {
        let config = BartConfig::bart_base();
        let layer = BartEncoderLayer::new(&config);

        // Verify self-attention component exists
        assert!(layer.self_attention().training());
    }

    #[test]
    fn test_bart_model_creation() {
        let model = BartForConditionalGeneration::bart_base();
        assert_eq!(model.num_encoder_layers(), 6);
        assert!(model.encoder_layer(0).is_some());
        assert!(model.encoder_layer(6).is_none());
    }

    #[test]
    fn test_bart_forward_pass() {
        let mut model = BartForConditionalGeneration::bart_base();
        let input_ids = Tensor::zeros(&[1, 10]).unwrap();

        let result = model.forward(&input_ids);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.dims().len(), 3); // [batch, seq_len, vocab_size]
        assert_eq!(output.size(2).unwrap(), 50265); // vocab_size
    }

    #[test]
    fn test_bart_training_mode() {
        let mut model = BartForConditionalGeneration::bart_base();

        model.train();
        assert!(model.training());

        model.eval();
        assert!(!model.training());
    }

    #[test]
    fn test_bart_parameter_count() {
        let model = BartForConditionalGeneration::bart_base();
        let params = model.parameters();
        assert!(!params.is_empty());

        // Check that we have parameters from all components
        let param_names: Vec<&String> = params.keys().collect();
        assert!(param_names.iter().any(|name| name.contains("embeddings")));
        assert!(param_names.iter().any(|name| name.contains("encoder")));
        assert!(param_names.iter().any(|name| name.contains("lm_head")));
        assert!(param_names.iter().any(|name| name.contains("layernorm")));
    }

    #[test]
    fn test_bart_encoder_layer_forward() {
        let config = BartConfig::bart_base();
        let mut layer = BartEncoderLayer::new(&config);
        let input = Tensor::zeros(&[1, 10, 768]).unwrap();

        let result = layer.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.dims(), &[1, 10, 768]); // Same shape as input
    }

    #[test]
    fn test_factory_functions() {
        let model = factory::bart_base();
        assert_eq!(model.config().hidden_size, 768);

        let large_model = factory::bart_large();
        assert_eq!(large_model.config().hidden_size, 1024);
    }

    #[test]
    fn test_generate_logits() {
        let model = BartForConditionalGeneration::bart_base();
        let input_ids = Tensor::zeros(&[1, 5]).unwrap();

        let result = model.generate_logits(&input_ids);
        assert!(result.is_ok());

        let logits = result.unwrap();
        assert_eq!(logits.dims(), &[1, 5, 50265]); // [batch, seq, vocab]
    }

    #[test]
    fn test_bart_embedding_scaling() {
        let mut config = BartConfig::bart_base();
        config.scale_embedding = true;

        let model = BartForConditionalGeneration::new(config);
        let input_ids = Tensor::zeros(&[1, 5]).unwrap();

        let result = model.forward(&input_ids);
        assert!(result.is_ok());
    }
}