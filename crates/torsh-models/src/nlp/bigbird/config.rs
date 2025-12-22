//! BigBird Configuration

use serde::{Deserialize, Serialize};

/// Configuration for BigBird models
///
/// BigBird uses sparse attention combining:
/// - Random attention
/// - Window attention
/// - Global attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BigBirdConfig {
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
    pub num_labels: Option<usize>,
    /// Attention type: "original_full" or "block_sparse"
    pub attention_type: String,
    /// Block size for sparse attention
    pub block_size: usize,
    /// Number of random blocks
    pub num_random_blocks: usize,
    /// Use separate query/value projections
    pub use_bias: bool,
}

impl BigBirdConfig {
    pub fn bigbird_base() -> Self {
        Self {
            vocab_size: 50358,
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
            layer_norm_eps: 1e-12,
            num_labels: None,
            attention_type: "block_sparse".to_string(),
            block_size: 64,
            num_random_blocks: 3,
            use_bias: true,
        }
    }

    pub fn bigbird_large() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            ..Self::bigbird_base()
        }
    }

    pub fn bigbird_base_for_classification(num_labels: usize) -> Self {
        Self {
            num_labels: Some(num_labels),
            ..Self::bigbird_base()
        }
    }

    pub fn validate(&self) -> crate::ModelResult<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(crate::ModelError::ValidationError {
                reason: format!(
                    "hidden_size ({}) must be divisible by num_attention_heads ({})",
                    self.hidden_size, self.num_attention_heads
                ),
            });
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bigbird_base_config() {
        let config = BigBirdConfig::bigbird_base();
        assert_eq!(config.hidden_size, 768);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_head_dim() {
        let config = BigBirdConfig::bigbird_base();
        assert_eq!(config.head_dim(), 64);
    }
}
