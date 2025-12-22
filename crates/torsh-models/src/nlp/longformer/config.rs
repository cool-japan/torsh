//! Longformer Configuration
//!
//! This module defines configuration structures for Longformer models.
//! Longformer extends BERT with efficient attention for long documents.

use serde::{Deserialize, Serialize};

/// Configuration for Longformer models
///
/// # Architecture
///
/// Longformer introduces:
/// - **Sliding Window Attention**: Local attention with fixed window size
/// - **Global Attention**: Select tokens attend to all positions
/// - **Dilated Sliding Window**: Multi-scale attention patterns
/// - **Extended Position Embeddings**: Up to 4096 tokens (vs BERT's 512)
///
/// # Variants
///
/// - Base: 12 layers, 768 hidden, 12 attention heads, 4096 max positions
/// - Large: 24 layers, 1024 hidden, 16 attention heads, 4096 max positions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongformerConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Feed-forward intermediate size
    pub intermediate_size: usize,
    /// Hidden activation function
    pub hidden_act: String,
    /// Dropout probability
    pub hidden_dropout_prob: f32,
    /// Attention dropout probability
    pub attention_probs_dropout_prob: f32,
    /// Maximum sequence length (extended for long documents)
    pub max_position_embeddings: usize,
    /// Type vocabulary size
    pub type_vocab_size: usize,
    /// Initializer range
    pub initializer_range: f32,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
    /// Number of classes for sequence classification
    pub num_labels: Option<usize>,
    /// Attention window size (one-sided)
    pub attention_window: Vec<usize>,
    /// Whether to use autoregressive attention
    pub autoregressive: bool,
}

impl LongformerConfig {
    /// Create Longformer-base configuration
    ///
    /// Parameters:
    /// - 12 layers
    /// - 768 hidden dimensions
    /// - 12 attention heads
    /// - 3072 intermediate size
    /// - 4096 max positions (8x BERT's 512)
    /// - 50265 vocabulary (RoBERTa vocab)
    pub fn longformer_base() -> Self {
        Self {
            vocab_size: 50265,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 4096,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-5,
            num_labels: None,
            attention_window: vec![512; 12], // 512 tokens window per layer
            autoregressive: false,
        }
    }

    /// Create Longformer-large configuration
    ///
    /// Parameters:
    /// - 24 layers
    /// - 1024 hidden dimensions
    /// - 16 attention heads
    /// - 4096 intermediate size
    /// - 4096 max positions
    /// - 50265 vocabulary
    pub fn longformer_large() -> Self {
        Self {
            vocab_size: 50265,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 4096,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-5,
            num_labels: None,
            attention_window: vec![512; 24], // 512 tokens window per layer
            autoregressive: false,
        }
    }

    /// Create Longformer configuration for sequence classification
    pub fn longformer_base_for_classification(num_labels: usize) -> Self {
        Self {
            num_labels: Some(num_labels),
            ..Self::longformer_base()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> crate::ModelResult<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(crate::ModelError::ValidationError {
                reason: format!(
                    "hidden_size ({}) must be divisible by num_attention_heads ({})",
                    self.hidden_size, self.num_attention_heads
                ),
            });
        }
        if self.hidden_dropout_prob < 0.0 || self.hidden_dropout_prob > 1.0 {
            return Err(crate::ModelError::ValidationError {
                reason: format!(
                    "hidden_dropout_prob must be in [0, 1], got {}",
                    self.hidden_dropout_prob
                ),
            });
        }
        if self.attention_window.len() != self.num_hidden_layers {
            return Err(crate::ModelError::ValidationError {
                reason: format!(
                    "attention_window length ({}) must equal num_hidden_layers ({})",
                    self.attention_window.len(),
                    self.num_hidden_layers
                ),
            });
        }
        Ok(())
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_longformer_base_config() {
        let config = LongformerConfig::longformer_base();
        assert_eq!(config.vocab_size, 50265);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.max_position_embeddings, 4096);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_longformer_large_config() {
        let config = LongformerConfig::longformer_large();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_attention_heads, 16);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_head_dim() {
        let config = LongformerConfig::longformer_base();
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_attention_window_validation() {
        let mut config = LongformerConfig::longformer_base();
        config.attention_window = vec![512; 10]; // Wrong length
        assert!(config.validate().is_err());
    }
}
