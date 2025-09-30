//! ALIGN Model Configuration
//!
//! Contains configuration structures for ALIGN (Large-scale Vision-Language Learning) models
//! including vision encoder (EfficientNet-style), text encoder (BERT-style), and overall model configs.

use torsh_core::error::{Result, TorshError};

/// ALIGN Configuration for Large-scale Vision-Language Learning
#[derive(Debug, Clone)]
pub struct ALIGNConfig {
    /// Vision encoder configuration (EfficientNet-based)
    pub vision_config: ALIGNVisionConfig,
    /// Text encoder configuration (BERT-based)
    pub text_config: ALIGNTextConfig,
    /// Projection dimension for contrastive learning
    pub projection_dim: usize,
    /// Temperature for contrastive learning
    pub temperature: f32,
    /// Whether to use learnable temperature
    pub learnable_temperature: bool,
}

impl Default for ALIGNConfig {
    fn default() -> Self {
        Self {
            vision_config: ALIGNVisionConfig::default(),
            text_config: ALIGNTextConfig::default(),
            projection_dim: 640,
            temperature: 0.07,
            learnable_temperature: true,
        }
    }
}

impl ALIGNConfig {
    /// Create ALIGN-Large configuration
    pub fn align_large() -> Self {
        Self::default()
    }

    /// Create ALIGN-Small configuration
    pub fn align_small() -> Self {
        Self {
            vision_config: ALIGNVisionConfig::efficientnet_b3(),
            text_config: ALIGNTextConfig::bert_base(),
            projection_dim: 512,
            temperature: 0.07,
            learnable_temperature: true,
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.projection_dim == 0 {
            return Err(TorshError::InvalidArgument(
                "projection_dim must be greater than 0".to_string(),
            ));
        }
        if self.temperature <= 0.0 {
            return Err(TorshError::InvalidArgument(
                "temperature must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// ALIGN Vision Encoder Configuration (EfficientNet-style)
#[derive(Debug, Clone)]
pub struct ALIGNVisionConfig {
    pub width_coefficient: f32,
    pub depth_coefficient: f32,
    pub image_size: usize,
    pub dropout_rate: f32,
    pub drop_connect_rate: f32,
    pub depth_divisor: usize,
    pub min_depth: usize,
    pub blocks_args: Vec<MBConvBlockArgs>,
    pub num_classes: usize,
    pub stem_size: usize,
    pub head_size: usize,
}

impl Default for ALIGNVisionConfig {
    fn default() -> Self {
        // EfficientNet-B7 like configuration for ALIGN
        Self {
            width_coefficient: 2.0,
            depth_coefficient: 3.1,
            image_size: 600,
            dropout_rate: 0.5,
            drop_connect_rate: 0.2,
            depth_divisor: 8,
            min_depth: 8,
            blocks_args: vec![
                MBConvBlockArgs {
                    kernel_size: 3,
                    num_repeat: 1,
                    input_filters: 32,
                    output_filters: 16,
                    expand_ratio: 1,
                    se_ratio: 0.25,
                    stride: 1,
                },
                MBConvBlockArgs {
                    kernel_size: 3,
                    num_repeat: 2,
                    input_filters: 16,
                    output_filters: 24,
                    expand_ratio: 6,
                    se_ratio: 0.25,
                    stride: 2,
                },
                MBConvBlockArgs {
                    kernel_size: 5,
                    num_repeat: 2,
                    input_filters: 24,
                    output_filters: 40,
                    expand_ratio: 6,
                    se_ratio: 0.25,
                    stride: 2,
                },
                MBConvBlockArgs {
                    kernel_size: 3,
                    num_repeat: 3,
                    input_filters: 40,
                    output_filters: 80,
                    expand_ratio: 6,
                    se_ratio: 0.25,
                    stride: 2,
                },
                MBConvBlockArgs {
                    kernel_size: 5,
                    num_repeat: 3,
                    input_filters: 80,
                    output_filters: 112,
                    expand_ratio: 6,
                    se_ratio: 0.25,
                    stride: 1,
                },
                MBConvBlockArgs {
                    kernel_size: 5,
                    num_repeat: 4,
                    input_filters: 112,
                    output_filters: 192,
                    expand_ratio: 6,
                    se_ratio: 0.25,
                    stride: 2,
                },
                MBConvBlockArgs {
                    kernel_size: 3,
                    num_repeat: 1,
                    input_filters: 192,
                    output_filters: 320,
                    expand_ratio: 6,
                    se_ratio: 0.25,
                    stride: 1,
                },
            ],
            num_classes: 1000,
            stem_size: 32,
            head_size: 2560,
        }
    }
}

impl ALIGNVisionConfig {
    /// EfficientNet-B3 configuration for smaller models
    pub fn efficientnet_b3() -> Self {
        Self {
            width_coefficient: 1.2,
            depth_coefficient: 1.4,
            image_size: 300,
            dropout_rate: 0.3,
            drop_connect_rate: 0.2,
            depth_divisor: 8,
            min_depth: 8,
            blocks_args: vec![
                MBConvBlockArgs {
                    kernel_size: 3,
                    num_repeat: 1,
                    input_filters: 32,
                    output_filters: 16,
                    expand_ratio: 1,
                    se_ratio: 0.25,
                    stride: 1,
                },
                MBConvBlockArgs {
                    kernel_size: 3,
                    num_repeat: 2,
                    input_filters: 16,
                    output_filters: 24,
                    expand_ratio: 6,
                    se_ratio: 0.25,
                    stride: 2,
                },
                MBConvBlockArgs {
                    kernel_size: 5,
                    num_repeat: 2,
                    input_filters: 24,
                    output_filters: 40,
                    expand_ratio: 6,
                    se_ratio: 0.25,
                    stride: 2,
                },
                MBConvBlockArgs {
                    kernel_size: 3,
                    num_repeat: 3,
                    input_filters: 40,
                    output_filters: 80,
                    expand_ratio: 6,
                    se_ratio: 0.25,
                    stride: 2,
                },
                MBConvBlockArgs {
                    kernel_size: 5,
                    num_repeat: 3,
                    input_filters: 80,
                    output_filters: 112,
                    expand_ratio: 6,
                    se_ratio: 0.25,
                    stride: 1,
                },
                MBConvBlockArgs {
                    kernel_size: 5,
                    num_repeat: 4,
                    input_filters: 112,
                    output_filters: 192,
                    expand_ratio: 6,
                    se_ratio: 0.25,
                    stride: 2,
                },
                MBConvBlockArgs {
                    kernel_size: 3,
                    num_repeat: 1,
                    input_filters: 192,
                    output_filters: 320,
                    expand_ratio: 6,
                    se_ratio: 0.25,
                    stride: 1,
                },
            ],
            num_classes: 1000,
            stem_size: 32,
            head_size: 1536,
        }
    }
}

/// ALIGN Text Encoder Configuration (BERT-style)
#[derive(Debug, Clone)]
pub struct ALIGNTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_dropout_prob: f32,
    pub attention_dropout: f32,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    pub hidden_act: String,
}

impl Default for ALIGNTextConfig {
    fn default() -> Self {
        // BERT-Large like configuration for ALIGN
        Self {
            vocab_size: 30522,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            hidden_dropout_prob: 0.1,
            attention_dropout: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".to_string(),
        }
    }
}

impl ALIGNTextConfig {
    /// BERT-Base configuration for smaller models
    pub fn bert_base() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_dropout_prob: 0.1,
            attention_dropout: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".to_string(),
        }
    }

    /// Get attention head size
    pub fn attention_head_size(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// MBConv (Mobile Inverted Bottleneck Convolution) Block Arguments
#[derive(Debug, Clone)]
pub struct MBConvBlockArgs {
    pub kernel_size: usize,
    pub num_repeat: usize,
    pub input_filters: usize,
    pub output_filters: usize,
    pub expand_ratio: usize,
    pub se_ratio: f32,
    pub stride: usize,
}
