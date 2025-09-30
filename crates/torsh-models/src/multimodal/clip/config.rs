//! CLIP model configuration

use super::super::common::types::{TextEncoderConfig, VisionEncoderConfig};

/// CLIP Configuration
#[derive(Debug, Clone)]
pub struct CLIPConfig {
    /// Vision encoder config
    pub vision_config: CLIPVisionConfig,
    /// Text encoder config
    pub text_config: CLIPTextConfig,
    /// Projection dimension for contrastive learning
    pub projection_dim: usize,
    /// Logit scale for contrastive loss
    pub logit_scale_init_value: f32,
}

impl Default for CLIPConfig {
    fn default() -> Self {
        Self {
            vision_config: CLIPVisionConfig::default(),
            text_config: CLIPTextConfig::default(),
            projection_dim: 512,
            logit_scale_init_value: 2.6592, // ln(1/0.07)
        }
    }
}

impl CLIPConfig {
    /// Create CLIP-ViT-B/32 configuration
    pub fn vit_b_32() -> Self {
        Self {
            vision_config: CLIPVisionConfig::vit_b_32(),
            text_config: CLIPTextConfig::default(),
            projection_dim: 512,
            logit_scale_init_value: 2.6592,
        }
    }

    /// Create CLIP-ViT-B/16 configuration
    pub fn vit_b_16() -> Self {
        Self {
            vision_config: CLIPVisionConfig::vit_b_16(),
            text_config: CLIPTextConfig::default(),
            projection_dim: 512,
            logit_scale_init_value: 2.6592,
        }
    }

    /// Create CLIP-ViT-L/14 configuration
    pub fn vit_l_14() -> Self {
        Self {
            vision_config: CLIPVisionConfig::vit_l_14(),
            text_config: CLIPTextConfig::large(),
            projection_dim: 768,
            logit_scale_init_value: 2.6592,
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        self.vision_config.validate()?;
        self.text_config.validate()?;

        if self.projection_dim == 0 {
            return Err("Projection dimension must be greater than 0".to_string());
        }

        Ok(())
    }
}

/// CLIP Vision Encoder Configuration
#[derive(Debug, Clone)]
pub struct CLIPVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_dropout_prob: f32,
    pub attention_dropout: f32,
    pub hidden_act: String,
    pub layer_norm_eps: f32,
    pub initializer_range: f32,
    pub initializer_factor: f32,
}

impl Default for CLIPVisionConfig {
    fn default() -> Self {
        Self::vit_b_32()
    }
}

impl CLIPVisionConfig {
    /// ViT-B/32 configuration
    pub fn vit_b_32() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            num_channels: 3,
            image_size: 224,
            patch_size: 32,
            hidden_dropout_prob: 0.0,
            attention_dropout: 0.0,
            hidden_act: "quick_gelu".to_string(),
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
            initializer_factor: 1.0,
        }
    }

    /// ViT-B/16 configuration
    pub fn vit_b_16() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            num_channels: 3,
            image_size: 224,
            patch_size: 16,
            hidden_dropout_prob: 0.0,
            attention_dropout: 0.0,
            hidden_act: "quick_gelu".to_string(),
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
            initializer_factor: 1.0,
        }
    }

    /// ViT-L/14 configuration
    pub fn vit_l_14() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            num_channels: 3,
            image_size: 224,
            patch_size: 14,
            hidden_dropout_prob: 0.0,
            attention_dropout: 0.0,
            hidden_act: "quick_gelu".to_string(),
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
            initializer_factor: 1.0,
        }
    }

    /// Convert to generic VisionEncoderConfig
    pub fn to_vision_config(&self) -> VisionEncoderConfig {
        VisionEncoderConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_channels: self.num_channels,
            image_size: self.image_size,
            patch_size: self.patch_size,
            hidden_dropout_prob: self.hidden_dropout_prob,
            attention_dropout: self.attention_dropout,
            hidden_act: self.hidden_act.clone(),
            layer_norm_eps: self.layer_norm_eps,
            initializer_range: self.initializer_range,
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

        if self.image_size % self.patch_size != 0 {
            return Err(format!(
                "Image size ({}) must be divisible by patch size ({})",
                self.image_size, self.patch_size
            ));
        }

        if self.num_hidden_layers == 0 {
            return Err("Number of hidden layers must be greater than 0".to_string());
        }

        if self.hidden_dropout_prob < 0.0 || self.hidden_dropout_prob > 1.0 {
            return Err("Hidden dropout probability must be between 0.0 and 1.0".to_string());
        }

        if self.attention_dropout < 0.0 || self.attention_dropout > 1.0 {
            return Err("Attention dropout probability must be between 0.0 and 1.0".to_string());
        }

        Ok(())
    }

    /// Get number of patches
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    /// Get sequence length (including class token)
    pub fn sequence_length(&self) -> usize {
        self.num_patches() + 1
    }

    /// Get attention head size
    pub fn attention_head_size(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// CLIP Text Encoder Configuration
#[derive(Debug, Clone)]
pub struct CLIPTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub hidden_dropout_prob: f32,
    pub attention_dropout: f32,
    pub hidden_act: String,
    pub layer_norm_eps: f32,
    pub initializer_range: f32,
    pub initializer_factor: f32,
    pub pad_token_id: usize,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
}

impl Default for CLIPTextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 49408,
            hidden_size: 512,
            intermediate_size: 2048,
            num_hidden_layers: 12,
            num_attention_heads: 8,
            max_position_embeddings: 77,
            hidden_dropout_prob: 0.0,
            attention_dropout: 0.0,
            hidden_act: "quick_gelu".to_string(),
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
            initializer_factor: 1.0,
            pad_token_id: 1,
            bos_token_id: 49406,
            eos_token_id: 49407,
        }
    }
}

impl CLIPTextConfig {
    /// Large text encoder configuration
    pub fn large() -> Self {
        Self {
            vocab_size: 49408,
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            max_position_embeddings: 77,
            hidden_dropout_prob: 0.0,
            attention_dropout: 0.0,
            hidden_act: "quick_gelu".to_string(),
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
            initializer_factor: 1.0,
            pad_token_id: 1,
            bos_token_id: 49406,
            eos_token_id: 49407,
        }
    }

    /// Convert to generic TextEncoderConfig
    pub fn to_text_config(&self) -> TextEncoderConfig {
        TextEncoderConfig {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            max_position_embeddings: self.max_position_embeddings,
            hidden_dropout_prob: self.hidden_dropout_prob,
            attention_dropout: self.attention_dropout,
            hidden_act: self.hidden_act.clone(),
            layer_norm_eps: self.layer_norm_eps,
            initializer_range: self.initializer_range,
            pad_token_id: self.pad_token_id,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
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

        if self.attention_dropout < 0.0 || self.attention_dropout > 1.0 {
            return Err("Attention dropout probability must be between 0.0 and 1.0".to_string());
        }

        Ok(())
    }

    /// Get attention head size
    pub fn attention_head_size(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}
