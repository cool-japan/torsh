//! Common types for multimodal models

use std::collections::HashMap;
use torsh_core::{
    error::{Result, TorshError},
    DeviceType,
};
use torsh_nn::prelude::*;
use torsh_nn::{Module, Parameter};
use torsh_tensor::{creation, Tensor};

/// Multimodal model architectures supported
#[derive(Debug, Clone, PartialEq)]
pub enum MultimodalArchitecture {
    CLIP,
    ALIGN,
    Flamingo,
    DallE,
    BLIP,
    LLaVA,
    InstructBLIP,
}

impl MultimodalArchitecture {
    pub fn as_str(&self) -> &'static str {
        match self {
            MultimodalArchitecture::CLIP => "CLIP",
            MultimodalArchitecture::ALIGN => "ALIGN",
            MultimodalArchitecture::Flamingo => "Flamingo",
            MultimodalArchitecture::DallE => "DALL-E",
            MultimodalArchitecture::BLIP => "BLIP",
            MultimodalArchitecture::LLaVA => "LLaVA",
            MultimodalArchitecture::InstructBLIP => "InstructBLIP",
        }
    }
}

/// Common multimodal task types
#[derive(Debug, Clone, PartialEq)]
pub enum MultimodalTask {
    /// Zero-shot image classification
    ZeroShotImageClassification,
    /// Text-to-image generation
    TextToImage,
    /// Image-to-text generation
    ImageToText,
    /// Visual question answering
    VisualQuestionAnswering,
    /// Image captioning
    ImageCaptioning,
    /// Few-shot learning with vision and language
    FewShotVisionLanguage,
    /// Cross-modal retrieval
    CrossModalRetrieval,
    /// Instruction following with vision
    InstructionFollowing,
}

impl MultimodalTask {
    pub fn as_str(&self) -> &'static str {
        match self {
            MultimodalTask::ZeroShotImageClassification => "zero_shot_image_classification",
            MultimodalTask::TextToImage => "text_to_image",
            MultimodalTask::ImageToText => "image_to_text",
            MultimodalTask::VisualQuestionAnswering => "visual_question_answering",
            MultimodalTask::ImageCaptioning => "image_captioning",
            MultimodalTask::FewShotVisionLanguage => "few_shot_vision_language",
            MultimodalTask::CrossModalRetrieval => "cross_modal_retrieval",
            MultimodalTask::InstructionFollowing => "instruction_following",
        }
    }
}

/// Cross-modal projection configuration
#[derive(Debug, Clone)]
pub struct CrossModalProjectionConfig {
    /// Input dimension from vision encoder
    pub vision_dim: usize,
    /// Input dimension from text encoder
    pub text_dim: usize,
    /// Output projection dimension
    pub projection_dim: usize,
    /// Whether to normalize the projections
    pub normalize: bool,
    /// Dropout probability for projections
    pub dropout: f32,
}

impl Default for CrossModalProjectionConfig {
    fn default() -> Self {
        Self {
            vision_dim: 768,
            text_dim: 512,
            projection_dim: 512,
            normalize: true,
            dropout: 0.0,
        }
    }
}

/// Attention pooling configuration for multimodal models
#[derive(Debug, Clone)]
pub struct AttentionPoolingConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of latent tokens for pooling
    pub num_latents: usize,
    /// Dropout probability
    pub dropout: f32,
}

impl Default for AttentionPoolingConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_heads: 12,
            num_latents: 64,
            dropout: 0.0,
        }
    }
}

/// Vision encoder configuration template
#[derive(Debug, Clone)]
pub struct VisionEncoderConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Intermediate/FFN dimension
    pub intermediate_size: usize,
    /// Number of transformer layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Input image channels
    pub num_channels: usize,
    /// Image size (assumed square)
    pub image_size: usize,
    /// Patch size for vision transformer
    pub patch_size: usize,
    /// Hidden dropout probability
    pub hidden_dropout_prob: f32,
    /// Attention dropout probability
    pub attention_dropout: f32,
    /// Activation function name
    pub hidden_act: String,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
    /// Weight initialization range
    pub initializer_range: f32,
}

impl Default for VisionEncoderConfig {
    fn default() -> Self {
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
            hidden_act: "gelu".to_string(),
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
        }
    }
}

/// Text encoder configuration template
#[derive(Debug, Clone)]
pub struct TextEncoderConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Intermediate/FFN dimension
    pub intermediate_size: usize,
    /// Number of transformer layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// Hidden dropout probability
    pub hidden_dropout_prob: f32,
    /// Attention dropout probability
    pub attention_dropout: f32,
    /// Activation function name
    pub hidden_act: String,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
    /// Weight initialization range
    pub initializer_range: f32,
    /// Padding token ID
    pub pad_token_id: usize,
    /// Beginning of sequence token ID
    pub bos_token_id: usize,
    /// End of sequence token ID
    pub eos_token_id: usize,
}

impl Default for TextEncoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            hidden_size: 512,
            intermediate_size: 2048,
            num_hidden_layers: 12,
            num_attention_heads: 8,
            max_position_embeddings: 77,
            hidden_dropout_prob: 0.0,
            attention_dropout: 0.0,
            hidden_act: "gelu".to_string(),
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
        }
    }
}
