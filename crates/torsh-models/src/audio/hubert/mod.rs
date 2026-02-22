//! HuBERT models for self-supervised speech representation learning
//!
//! Implementation of HuBERT architecture for speech representation learning.
//!
//! Reference: [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447)

/// HuBERT Configuration
#[derive(Debug, Clone)]
pub struct HuBERTConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_dropout: f32,
    pub attention_dropout: f32,
    pub feat_proj_dropout: f32,
    pub layerdrop: f32,
    pub conv_dim: Vec<usize>,
    pub conv_stride: Vec<usize>,
    pub conv_kernel: Vec<usize>,
    pub conv_bias: bool,
    pub num_conv_pos_embeddings: usize,
    pub num_conv_pos_embedding_groups: usize,
    pub feat_extract_norm: String,
    pub feat_extract_activation: String,
    pub conv_pos_embeddings_kernel_size: usize,
    pub apply_spec_augment: bool,
    pub mask_time_prob: f32,
    pub mask_time_length: usize,
    pub mask_feature_prob: f32,
    pub mask_feature_length: usize,
    pub label_smoothing: f32,
}

impl Default for HuBERTConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_dropout: 0.1,
            attention_dropout: 0.1,
            feat_proj_dropout: 0.0,
            layerdrop: 0.1,
            conv_dim: vec![512, 512, 512, 512, 512, 512, 512],
            conv_stride: vec![5, 2, 2, 2, 2, 2, 2],
            conv_kernel: vec![10, 3, 3, 3, 3, 2, 2],
            conv_bias: false,
            num_conv_pos_embeddings: 128,
            num_conv_pos_embedding_groups: 16,
            feat_extract_norm: "group".to_string(),
            feat_extract_activation: "gelu".to_string(),
            conv_pos_embeddings_kernel_size: 128,
            apply_spec_augment: true,
            mask_time_prob: 0.05,
            mask_time_length: 10,
            mask_feature_prob: 0.0,
            mask_feature_length: 10,
            label_smoothing: 0.1,
        }
    }
}

impl HuBERTConfig {
    /// Create configuration for HuBERT Base model
    pub fn base() -> Self {
        Self::default()
    }

    /// Create configuration for HuBERT Large model
    pub fn large() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            ..Self::default()
        }
    }

    /// Create configuration for HuBERT X-Large model
    pub fn xlarge() -> Self {
        Self {
            hidden_size: 1280,
            num_hidden_layers: 48,
            num_attention_heads: 16,
            intermediate_size: 5120,
            ..Self::default()
        }
    }
}

// Forward declarations for the component modules that will be implemented
pub struct HuBERTFeatureExtractor;
pub struct HuBERTFeatureProjection;
pub struct HuBERTPositionalConvEmbedding;
pub struct HuBERTSelfAttention;
pub struct HuBERTFeedForward;
pub struct HuBERTEncoderLayer;
pub struct HuBERTEncoder;
pub struct HuBERTForSequenceClassification;
pub struct HuBERTModel;

// Note: Key types (HuBERTConfig) are already public
// Removed redundant re-export to fix duplicate definition errors
