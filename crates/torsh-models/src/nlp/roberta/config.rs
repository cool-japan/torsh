//! RoBERTa model configuration

/// RoBERTa configuration parameters
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

    /// Create custom RoBERTa configuration
    pub fn custom(
        vocab_size: usize,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
    ) -> Self {
        Self {
            vocab_size,
            hidden_size,
            num_hidden_layers: num_layers,
            num_attention_heads: num_heads,
            intermediate_size: hidden_size * 4, // Standard ratio
            ..Self::default()
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(format!(
                "Hidden size ({}) must be divisible by number of attention heads ({})",
                self.hidden_size, self.num_attention_heads
            ));
        }

        if self.vocab_size == 0 {
            return Err("Vocabulary size must be greater than 0".to_string());
        }

        if self.num_hidden_layers == 0 {
            return Err("Number of hidden layers must be greater than 0".to_string());
        }

        if self.hidden_dropout_prob < 0.0 || self.hidden_dropout_prob > 1.0 {
            return Err("Hidden dropout probability must be between 0.0 and 1.0".to_string());
        }

        if self.attention_probs_dropout_prob < 0.0 || self.attention_probs_dropout_prob > 1.0 {
            return Err("Attention dropout probability must be between 0.0 and 1.0".to_string());
        }

        Ok(())
    }

    /// Get the size of each attention head
    pub fn attention_head_size(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get total parameters count (approximate)
    pub fn parameter_count(&self) -> usize {
        // Embeddings: vocab_size * hidden_size + max_pos * hidden_size + type_vocab * hidden_size
        let embedding_params = self.vocab_size * self.hidden_size
            + self.max_position_embeddings * self.hidden_size
            + self.type_vocab_size * self.hidden_size;

        // Per layer: attention (4 * hidden^2) + FFN (2 * hidden * intermediate) + layer norms
        let per_layer_params = 4 * self.hidden_size * self.hidden_size // attention weights
            + 2 * self.hidden_size * self.intermediate_size // FFN weights
            + 4 * self.hidden_size; // layer norm parameters

        embedding_params + self.num_hidden_layers * per_layer_params
    }
}
