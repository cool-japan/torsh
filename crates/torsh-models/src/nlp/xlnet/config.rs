//! XLNet Configuration
//!
//! This module defines configuration structures for XLNet models.
//! XLNet uses permutation language modeling and two-stream self-attention.

use serde::{Deserialize, Serialize};

/// Configuration for XLNet models
///
/// # Architecture
///
/// XLNet introduces:
/// - Permutation language modeling for bidirectional context
/// - Two-stream self-attention mechanism
/// - Relative positional encodings (Transformer-XL style)
/// - Segment-level recurrence mechanism
///
/// # Variants
///
/// - Base: 12 layers, 768 hidden, 12 attention heads
/// - Large: 24 layers, 1024 hidden, 16 attention heads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XLNetConfig {
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
    /// Dropout probability
    pub hidden_dropout_prob: f32,
    /// Attention dropout probability
    pub attention_probs_dropout_prob: f32,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
    /// Number of classes for sequence classification
    pub num_labels: Option<usize>,
    /// Initialization standard deviation
    pub initializer_range: f32,
    /// Summary activation function
    pub summary_activation: String,
    /// Memory length for segment-level recurrence
    pub mem_len: usize,
    /// Whether to use relative attention
    pub use_relative_attention: bool,
    /// Attention type (bi, uni)
    pub attn_type: String,
    /// Bi data pipeline
    pub bi_data: bool,
    /// Clamp length for relative position
    pub clamp_len: isize,
    /// Whether to use two-stream attention
    pub use_mems_train: bool,
    /// Whether to use same_length attention
    pub same_length: bool,
}

impl XLNetConfig {
    /// Create XLNet-base configuration
    ///
    /// Parameters:
    /// - 12 layers
    /// - 768 hidden dimensions
    /// - 12 attention heads
    /// - 3072 intermediate size
    /// - 32000 vocabulary
    pub fn xlnet_base() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            layer_norm_eps: 1e-12,
            num_labels: None,
            initializer_range: 0.02,
            summary_activation: "tanh".to_string(),
            mem_len: 512,
            use_relative_attention: true,
            attn_type: "bi".to_string(),
            bi_data: false,
            clamp_len: -1,
            use_mems_train: true,
            same_length: false,
        }
    }

    /// Create XLNet-large configuration
    ///
    /// Parameters:
    /// - 24 layers
    /// - 1024 hidden dimensions
    /// - 16 attention heads
    /// - 4096 intermediate size
    /// - 32000 vocabulary
    pub fn xlnet_large() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            layer_norm_eps: 1e-12,
            num_labels: None,
            initializer_range: 0.02,
            summary_activation: "tanh".to_string(),
            mem_len: 512,
            use_relative_attention: true,
            attn_type: "bi".to_string(),
            bi_data: false,
            clamp_len: -1,
            use_mems_train: true,
            same_length: false,
        }
    }

    /// Create XLNet configuration for sequence classification
    pub fn xlnet_base_for_classification(num_labels: usize) -> Self {
        Self {
            num_labels: Some(num_labels),
            ..Self::xlnet_base()
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
    fn test_xlnet_base_config() {
        let config = XLNetConfig::xlnet_base();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.num_attention_heads, 12);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_xlnet_large_config() {
        let config = XLNetConfig::xlnet_large();
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 24);
        assert_eq!(config.num_attention_heads, 16);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_head_dim() {
        let config = XLNetConfig::xlnet_base();
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn test_validation() {
        let mut config = XLNetConfig::xlnet_base();
        config.hidden_size = 100;
        config.num_attention_heads = 12;
        assert!(config.validate().is_err());
    }
}
