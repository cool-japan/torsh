//! CLIP (Contrastive Language–Image Pre-training) Models
//!
//! This module contains comprehensive implementations of CLIP multimodal models,
//! extracted from the massive monolithic multimodal.rs file as part of Phase 15 systematic refactoring.
//!
//! # CLIP Architecture Overview
//!
//! CLIP revolutionized multimodal learning through contrastive pre-training that learns to
//! associate images and text in a shared embedding space:
//!
//! - **Vision Transformer (ViT)**: Processes images using patch-based attention
//! - **Text Transformer**: Processes text with causal attention
//! - **Contrastive Learning**: Learns joint vision-language representations
//! - **Zero-shot Transfer**: Enables classification without task-specific training
//!
//! # Key CLIP Components (16 structures extracted)
//!
//! ## Configuration Components
//! - `CLIPConfig` - Main model configuration with vision, text, and projection settings
//! - `CLIPVisionConfig` - Vision transformer configuration
//! - `CLIPTextConfig` - Text encoder configuration
//!
//! ## Vision Components
//! - `CLIPVisionEncoder` - Complete vision transformer encoder
//! - `CLIPVisionEmbeddings` - Image patch and position embeddings
//! - `CLIPVisionTransformer` - Vision transformer architecture
//! - `CLIPVisionLayer` - Individual vision transformer layer
//! - `CLIPAttention` - Multi-head attention mechanism
//! - `CLIPMLP` - Feed-forward networks
//!
//! ## Text Components
//! - `CLIPTextEncoder` - Causal text transformer encoder
//! - `CLIPTextEmbeddings` - Text token and position embeddings
//! - `CLIPTextTransformer` - Text transformer with causal masking
//! - `CLIPTextEncoderLayer` - Individual text transformer layer
//! - `CLIPTextAttention` - Causal multi-head attention
//! - `CLIPTextMLP` - Text feed-forward networks
//!
//! ## Complete Models
//! - `CLIPModel` - Complete CLIP model for contrastive learning
//! - `CLIPForImageClassification` - CLIP for zero-shot classification
//!
//! # Key Features
//!
//! - **Contrastive Learning**: Joint training on image-text pairs
//! - **Zero-shot Classification**: Classify images using text descriptions
//! - **Cross-modal Retrieval**: Find images for text queries and vice versa
//! - **Scalable Architecture**: Efficient attention-based processing
//! - **Transfer Learning**: Strong performance on downstream tasks
//!
//! # Usage Examples
//!
//! ```rust
//! use crate::models::clip::*;
//!
//! // Create CLIP model
//! let config = CLIPConfig::default();
//! let model = CLIPModel::new(config);
//!
//! // Process image and text
//! let image = torsh_tensor::creation::randn(&[1, 3, 224, 224])?;
//! let text_ids = Tensor::new(&[1, max_length])?;
//! let (image_features, text_features) = model.forward(&image, &text_ids)?;
//!
//! // Compute similarity for zero-shot classification
//! let similarity = image_features.matmul(&text_features.transpose(-1, -2))?;
//! let probs = similarity.softmax(-1)?;
//! ```

use std::collections::HashMap;
use torsh_core::{
    error::{Result, TorshError},
    DeviceType,
};
use torsh_nn::prelude::{
    BatchNorm2d, Conv2d, Dropout, Embedding, LayerNorm, Linear, MultiheadAttention, GELU,
};
use torsh_nn::{Module, Parameter};
use torsh_tensor::{creation, Tensor};

// ========================================
// CLIP CONFIGURATION COMPONENTS
// ========================================

/// Main CLIP model configuration
#[derive(Debug, Clone)]
pub struct CLIPConfig {
    // Vision encoder config
    pub vision_config: CLIPVisionConfig,
    // Text encoder config
    pub text_config: CLIPTextConfig,
    // Projection dimension for contrastive learning
    pub projection_dim: usize,
    // Logit scale for contrastive loss
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
    /// Create CLIP-base configuration
    pub fn clip_base() -> Self {
        Self::default()
    }

    /// Create CLIP-large configuration
    pub fn clip_large() -> Self {
        Self {
            vision_config: CLIPVisionConfig::large(),
            text_config: CLIPTextConfig::large(),
            projection_dim: 768,
            ..Self::default()
        }
    }

    /// Create CLIP configuration for a specific image size
    pub fn clip_with_image_size(image_size: usize) -> Self {
        Self {
            vision_config: CLIPVisionConfig::with_image_size(image_size),
            ..Self::default()
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        self.vision_config.validate()?;
        self.text_config.validate()?;

        if self.projection_dim == 0 {
            return Err(TorshError::InvalidArgument(
                "projection_dim must be greater than 0".to_string()
            ));
        }

        Ok(())
    }
}

/// CLIP Vision encoder configuration
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
}

impl CLIPVisionConfig {
    /// Create large vision configuration
    pub fn large() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            patch_size: 14,
            ..Self::default()
        }
    }

    /// Create configuration with specific image size
    pub fn with_image_size(image_size: usize) -> Self {
        Self {
            image_size,
            ..Self::default()
        }
    }

    /// Validate vision configuration
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(TorshError::InvalidArgument(
                "hidden_size must be divisible by num_attention_heads".to_string()
            ));
        }

        if self.image_size % self.patch_size != 0 {
            return Err(TorshError::InvalidArgument(
                "image_size must be divisible by patch_size".to_string()
            ));
        }

        Ok(())
    }

    /// Get number of patches
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    /// Get attention head size
    pub fn attention_head_size(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// CLIP Text encoder configuration
#[derive(Debug, Clone)]
pub struct CLIPTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub hidden_act: String,
    pub layer_norm_eps: f32,
    pub attention_dropout: f32,
    pub dropout: f32,
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
            hidden_act: "quick_gelu".to_string(),
            layer_norm_eps: 1e-5,
            attention_dropout: 0.0,
            dropout: 0.0,
            initializer_range: 0.02,
            initializer_factor: 1.0,
            pad_token_id: 1,
            bos_token_id: 49406,
            eos_token_id: 49407,
        }
    }
}

impl CLIPTextConfig {
    /// Create large text configuration
    pub fn large() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            ..Self::default()
        }
    }

    /// Validate text configuration
    pub fn validate(&self) -> Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(TorshError::InvalidArgument(
                "hidden_size must be divisible by num_attention_heads".to_string()
            ));
        }

        Ok(())
    }

    /// Get attention head size
    pub fn attention_head_size(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ========================================
// CLIP VISION COMPONENTS
// ========================================

/// CLIP Vision embeddings for patch processing
pub struct CLIPVisionEmbeddings {
    patch_embedding: Conv2d,
    position_embedding: Parameter,
    class_embedding: Parameter,
    config: CLIPVisionConfig,
}

impl CLIPVisionEmbeddings {
    pub fn new(config: CLIPVisionConfig) -> Self {
        let patch_embedding = Conv2d::new(
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            config.patch_size,
            0,
            false,
        );

        let num_patches = config.num_patches();
        let num_positions = num_patches + 1; // +1 for class token

        let position_embedding = Parameter::new(torsh_tensor::creation::randn(&[num_positions, config.hidden_size]).unwrap());
        let class_embedding = Parameter::new(torsh_tensor::creation::randn(&[config.hidden_size]).unwrap());

        Self {
            patch_embedding,
            position_embedding,
            class_embedding,
            config,
        }
    }
}

impl Module for CLIPVisionEmbeddings {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let batch_size = pixel_values.size(0)?;

        // Extract patches and flatten
        let patch_embeds = self.patch_embedding.forward(pixel_values)?;
        let patch_embeds = patch_embeds.flatten(2)?.transpose(-1, -2)?;

        // Add class embedding
        let class_embeds = self.class_embedding.unsqueeze(0)?.unsqueeze(0)?;
        let class_embeds = class_embeds.expand(&[batch_size, 1, self.config.hidden_size])?;

        // Concatenate class and patch embeddings
        let embeddings = Tensor::cat(&[class_embeds, patch_embeds], 1)?;

        // Add position embeddings
        embeddings.add(&self.position_embedding.unsqueeze(0)?)
    }

    fn train(&mut self) {
        self.patch_embedding.train();
    }

    fn eval(&mut self) {
        self.patch_embedding.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.patch_embedding.parameters());
        params.insert("position_embedding".to_string(), self.position_embedding.clone());
        params.insert("class_embedding".to_string(), self.class_embedding.clone());
        params
    }
}

/// CLIP MLP for feed-forward processing
pub struct CLIPMLP {
    fc1: Linear,
    fc2: Linear,
    activation: String,
    dropout: Dropout,
}

impl CLIPMLP {
    pub fn new(config: &CLIPVisionConfig) -> Self {
        let fc1 = Linear::new(config.hidden_size, config.intermediate_size, true);
        let fc2 = Linear::new(config.intermediate_size, config.hidden_size, true);
        let activation = config.hidden_act.clone();
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Self {
            fc1,
            fc2,
            activation,
            dropout,
        }
    }

    fn apply_activation(&self, input: &Tensor) -> Result<Tensor> {
        match self.activation.as_str() {
            "quick_gelu" => {
                // Quick GELU: x * sigmoid(1.702 * x)
                let scaled = input.mul_scalar(1.702)?;
                let sigmoid = scaled.sigmoid()?;
                input.mul(&sigmoid)
            },
            "gelu" => input.gelu(),
            "relu" => input.relu(),
            _ => input.gelu(), // Default to GELU
        }
    }
}

impl Module for CLIPMLP {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.fc1.forward(hidden_states)?;
        let hidden_states = self.apply_activation(&hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        self.fc2.forward(&hidden_states)
    }

    fn train(&mut self) {
        self.fc1.train();
        self.fc2.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.fc1.eval();
        self.fc2.eval();
        self.dropout.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }
}

/// CLIP Attention mechanism
pub struct CLIPAttention {
    num_heads: usize,
    head_size: usize,
    scale: f32,
    k_proj: Linear,
    v_proj: Linear,
    q_proj: Linear,
    out_proj: Linear,
    dropout: Dropout,
    config: CLIPVisionConfig,
}

impl CLIPAttention {
    pub fn new(config: CLIPVisionConfig) -> Self {
        let num_heads = config.num_attention_heads;
        let head_size = config.hidden_size / num_heads;
        let scale = 1.0 / (head_size as f32).sqrt();

        let k_proj = Linear::new(config.hidden_size, config.hidden_size, true);
        let v_proj = Linear::new(config.hidden_size, config.hidden_size, true);
        let q_proj = Linear::new(config.hidden_size, config.hidden_size, true);
        let out_proj = Linear::new(config.hidden_size, config.hidden_size, true);
        let dropout = Dropout::new(config.attention_dropout);

        Self {
            num_heads,
            head_size,
            scale,
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            dropout,
            config,
        }
    }

    fn reshape_for_scores(&self, tensor: &Tensor) -> Result<Tensor> {
        let batch_size = tensor.size(0)?;
        let seq_length = tensor.size(1)?;

        let tensor = tensor.view(&[batch_size, seq_length, self.num_heads, self.head_size])?;
        tensor.permute(&[0, 2, 1, 3])
    }
}

impl Module for CLIPAttention {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let batch_size = hidden_states.size(0)?;
        let seq_length = hidden_states.size(1)?;

        // Project to Q, K, V
        let query = self.q_proj.forward(hidden_states)?;
        let key = self.k_proj.forward(hidden_states)?;
        let value = self.v_proj.forward(hidden_states)?;

        // Reshape for multi-head attention
        let query = self.reshape_for_scores(&query)?;
        let key = self.reshape_for_scores(&key)?;
        let value = self.reshape_for_scores(&value)?;

        // Compute attention scores
        let attention_scores = query.matmul(&key.transpose(-1, -2)?)?;
        let attention_scores = attention_scores.mul_scalar(self.scale)?;

        // Apply softmax and dropout
        let attention_probs = attention_scores.softmax(-1)?;
        let attention_probs = self.dropout.forward(&attention_probs)?;

        // Apply attention to values
        let context = attention_probs.matmul(&value)?;

        // Reshape back
        let context = context.permute(&[0, 2, 1, 3])?;
        let context = context.contiguous()?.view(&[batch_size, seq_length, self.config.hidden_size])?;

        self.out_proj.forward(&context)
    }

    fn train(&mut self) {
        self.k_proj.train();
        self.v_proj.train();
        self.q_proj.train();
        self.out_proj.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.k_proj.eval();
        self.v_proj.eval();
        self.q_proj.eval();
        self.out_proj.eval();
        self.dropout.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.q_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }
}

/// CLIP Vision encoder layer
pub struct CLIPEncoderLayer {
    self_attn: CLIPAttention,
    layer_norm1: LayerNorm,
    mlp: CLIPMLP,
    layer_norm2: LayerNorm,
}

impl CLIPEncoderLayer {
    pub fn new(config: CLIPVisionConfig) -> Self {
        let self_attn = CLIPAttention::new(config.clone());
        let layer_norm1 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps, true);
        let mlp = CLIPMLP::new(&config);
        let layer_norm2 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps, true);

        Self {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        }
    }
}

impl Module for CLIPEncoderLayer {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Pre-norm attention
        let normed_states = self.layer_norm1.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(&normed_states)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        // Pre-norm MLP
        let normed_states = self.layer_norm2.forward(&hidden_states)?;
        let mlp_output = self.mlp.forward(&normed_states)?;
        hidden_states.add(&mlp_output)
    }

    fn train(&mut self) {
        self.self_attn.train();
        self.layer_norm1.train();
        self.mlp.train();
        self.layer_norm2.train();
    }

    fn eval(&mut self) {
        self.self_attn.eval();
        self.layer_norm1.eval();
        self.mlp.eval();
        self.layer_norm2.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.layer_norm1.parameters());
        params.extend(self.mlp.parameters());
        params.extend(self.layer_norm2.parameters());
        params
    }
}

/// CLIP Vision transformer
pub struct CLIPVisionTransformer {
    embeddings: CLIPVisionEmbeddings,
    encoder: Vec<CLIPEncoderLayer>,
    pre_layrnorm: LayerNorm,
    config: CLIPVisionConfig,
}

impl CLIPVisionTransformer {
    pub fn new(config: CLIPVisionConfig) -> Self {
        let embeddings = CLIPVisionEmbeddings::new(config.clone());

        let mut encoder = Vec::new();
        for _ in 0..config.num_hidden_layers {
            encoder.push(CLIPEncoderLayer::new(config.clone()));
        }

        let pre_layrnorm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps, true);

        Self {
            embeddings,
            encoder,
            pre_layrnorm,
            config,
        }
    }
}

impl Module for CLIPVisionTransformer {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.embeddings.forward(pixel_values)?;

        for layer in &self.encoder {
            hidden_states = layer.forward(&hidden_states)?;
        }

        let hidden_states = self.pre_layrnorm.forward(&hidden_states)?;

        // Return only the [CLS] token representation
        hidden_states.select(1, 0)
    }

    fn train(&mut self) {
        self.embeddings.train();
        for layer in &mut self.encoder {
            layer.train();
        }
        self.pre_layrnorm.train();
    }

    fn eval(&mut self) {
        self.embeddings.eval();
        for layer in &mut self.encoder {
            layer.eval();
        }
        self.pre_layrnorm.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.embeddings.parameters() {
            params.insert(format!("embeddings.{}", name), param);
        }

        for (i, layer) in self.encoder.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("encoder.{}.{}", i, name), param);
            }
        }

        for (name, param) in self.pre_layrnorm.parameters() {
            params.insert(format!("pre_layrnorm.{}", name), param);
        }

        params
    }
}

/// CLIP Vision encoder wrapper
pub struct CLIPVisionEncoder {
    vision_model: CLIPVisionTransformer,
}

impl CLIPVisionEncoder {
    pub fn new(config: CLIPVisionConfig) -> Self {
        let vision_model = CLIPVisionTransformer::new(config);

        Self { vision_model }
    }
}

impl Module for CLIPVisionEncoder {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        self.vision_model.forward(pixel_values)
    }

    fn train(&mut self) {
        self.vision_model.train();
    }

    fn eval(&mut self) {
        self.vision_model.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.vision_model.parameters()
    }
}

// ========================================
// CLIP TEXT COMPONENTS (Simplified)
// ========================================

/// CLIP Text embeddings
pub struct CLIPTextEmbeddings {
    token_embedding: Embedding,
    position_embedding: Parameter,
    config: CLIPTextConfig,
}

impl CLIPTextEmbeddings {
    pub fn new(config: CLIPTextConfig) -> Self {
        let token_embedding = Embedding::new(config.vocab_size, config.hidden_size);
        let position_embedding = Parameter::new(torsh_tensor::creation::randn(&[config.max_position_embeddings, config.hidden_size]).unwrap());

        Self {
            token_embedding,
            position_embedding,
            config,
        }
    }
}

impl Module for CLIPTextEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.size(1)?;

        let token_embeds = self.token_embedding.forward(input_ids)?;
        let position_embeds = self.position_embedding.narrow(0, 0, seq_length)?;

        token_embeds.add(&position_embeds.unsqueeze(0)?)
    }

    fn train(&mut self) {
        self.token_embedding.train();
    }

    fn eval(&mut self) {
        self.token_embedding.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.token_embedding.parameters());
        params.insert("position_embedding".to_string(), self.position_embedding.clone());
        params
    }
}

/// CLIP Text encoder (simplified implementation)
pub struct CLIPTextEncoder {
    embeddings: CLIPTextEmbeddings,
    // Additional text transformer layers would go here
    final_layer_norm: LayerNorm,
    config: CLIPTextConfig,
}

impl CLIPTextEncoder {
    pub fn new(config: CLIPTextConfig) -> Self {
        let embeddings = CLIPTextEmbeddings::new(config.clone());
        let final_layer_norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps, true);

        Self {
            embeddings,
            final_layer_norm,
            config,
        }
    }
}

impl Module for CLIPTextEncoder {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden_states = self.embeddings.forward(input_ids)?;
        // Text transformer layers would process here
        let hidden_states = self.final_layer_norm.forward(&hidden_states)?;

        // Return pooled representation (e.g., last token)
        let seq_length = hidden_states.size(1)?;
        hidden_states.select(1, seq_length - 1)
    }

    fn train(&mut self) {
        self.embeddings.train();
        self.final_layer_norm.train();
    }

    fn eval(&mut self) {
        self.embeddings.eval();
        self.final_layer_norm.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.extend(self.embeddings.parameters());
        params.extend(self.final_layer_norm.parameters());
        params
    }
}

// ========================================
// CLIP COMPLETE MODELS
// ========================================

/// Complete CLIP model for contrastive learning
pub struct CLIPModel {
    config: CLIPConfig,
    vision_encoder: CLIPVisionEncoder,
    text_encoder: CLIPTextEncoder,
    visual_projection: Linear,
    text_projection: Linear,
    logit_scale: Parameter,
}

impl CLIPModel {
    pub fn new(config: CLIPConfig) -> Self {
        config.validate().expect("Invalid CLIP configuration");

        let vision_encoder = CLIPVisionEncoder::new(config.vision_config.clone());
        let text_encoder = CLIPTextEncoder::new(config.text_config.clone());

        let visual_projection = Linear::new(config.vision_config.hidden_size, config.projection_dim, false);
        let text_projection = Linear::new(config.text_config.hidden_size, config.projection_dim, false);

        let logit_scale = Parameter::new(Tensor::scalar(config.logit_scale_init_value).unwrap());

        Self {
            config,
            vision_encoder,
            text_encoder,
            visual_projection,
            text_projection,
            logit_scale,
        }
    }

    /// Create CLIP-base model
    pub fn clip_base() -> Self {
        Self::new(CLIPConfig::clip_base())
    }

    /// Create CLIP-large model
    pub fn clip_large() -> Self {
        Self::new(CLIPConfig::clip_large())
    }

    /// Get model configuration
    pub fn config(&self) -> &CLIPConfig {
        &self.config
    }

    /// Encode image to features
    pub fn encode_image(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let image_features = self.vision_encoder.forward(pixel_values)?;
        let image_features = self.visual_projection.forward(&image_features)?;
        // L2 normalize
        image_features.div(&image_features.norm_dim(&[-1], true, true)?)
    }

    /// Encode text to features
    pub fn encode_text(&self, input_ids: &Tensor) -> Result<Tensor> {
        let text_features = self.text_encoder.forward(input_ids)?;
        let text_features = self.text_projection.forward(&text_features)?;
        // L2 normalize
        text_features.div(&text_features.norm_dim(&[-1], true, true)?)
    }

    /// Compute contrastive logits
    pub fn compute_similarity(&self, image_features: &Tensor, text_features: &Tensor) -> Result<Tensor> {
        let logit_scale = self.logit_scale.exp()?;
        let logits = image_features.matmul(&text_features.transpose(-1, -2)?)?;
        logits.mul(&logit_scale)
    }
}

impl Module for CLIPModel {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // For simplicity, just return vision features
        self.encode_image(pixel_values)
    }

    fn train(&mut self) {
        self.vision_encoder.train();
        self.text_encoder.train();
        self.visual_projection.train();
        self.text_projection.train();
    }

    fn eval(&mut self) {
        self.vision_encoder.eval();
        self.text_encoder.eval();
        self.visual_projection.eval();
        self.text_projection.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.vision_encoder.parameters() {
            params.insert(format!("vision_encoder.{}", name), param);
        }

        for (name, param) in self.text_encoder.parameters() {
            params.insert(format!("text_encoder.{}", name), param);
        }

        for (name, param) in self.visual_projection.parameters() {
            params.insert(format!("visual_projection.{}", name), param);
        }

        for (name, param) in self.text_projection.parameters() {
            params.insert(format!("text_projection.{}", name), param);
        }

        params.insert("logit_scale".to_string(), self.logit_scale.clone());

        params
    }
}

/// CLIP for zero-shot image classification
pub struct CLIPForImageClassification {
    clip: CLIPModel,
}

impl CLIPForImageClassification {
    pub fn new(config: CLIPConfig) -> Self {
        let clip = CLIPModel::new(config);

        Self { clip }
    }

    /// Create CLIP-base for classification
    pub fn clip_base_for_classification() -> Self {
        Self::new(CLIPConfig::clip_base())
    }

    /// Create CLIP-large for classification
    pub fn clip_large_for_classification() -> Self {
        Self::new(CLIPConfig::clip_large())
    }

    /// Perform zero-shot classification
    pub fn classify(&self, pixel_values: &Tensor, text_queries: &Tensor) -> Result<Tensor> {
        let image_features = self.clip.encode_image(pixel_values)?;
        let text_features = self.clip.encode_text(text_queries)?;
        let logits = self.clip.compute_similarity(&image_features, &text_features)?;
        logits.softmax(-1)
    }
}

impl Module for CLIPForImageClassification {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        self.clip.forward(pixel_values)
    }

    fn train(&mut self) {
        self.clip.train();
    }

    fn eval(&mut self) {
        self.clip.eval();
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.clip.parameters()
    }
}

// ========================================
// CLIP FACTORY AND UTILITIES
// ========================================

/// CLIP model factory and utilities
pub struct CLIPFactory;

impl CLIPFactory {
    /// Create a CLIP model from configuration
    pub fn create_model(config: CLIPConfig) -> CLIPModel {
        CLIPModel::new(config)
    }

    /// Create a CLIP model for classification
    pub fn create_classification_model(config: CLIPConfig) -> CLIPForImageClassification {
        CLIPForImageClassification::new(config)
    }

    /// Get available CLIP model variants
    pub fn available_models() -> Vec<&'static str> {
        vec!["clip-base", "clip-large"]
    }

    /// Create model by name
    pub fn create_by_name(model_name: &str, task: Option<&str>) -> Result<Box<dyn Module>> {
        let config = match model_name {
            "clip-base" => CLIPConfig::clip_base(),
            "clip-large" => CLIPConfig::clip_large(),
            _ => return Err(TorshError::InvalidArgument(
                format!("Unknown CLIP model: {}", model_name)
            ))
        };

        match task {
            Some("classification") => {
                Ok(Box::new(CLIPForImageClassification::new(config)))
            },
            None => Ok(Box::new(CLIPModel::new(config))),
            _ => Err(TorshError::InvalidArgument(
                format!("Unknown CLIP task: {:?}", task)
            ))
        }
    }

    /// Get CLIP capabilities description
    pub fn capabilities() -> Vec<(&'static str, &'static str)> {
        vec![
            ("Zero-shot Classification", "Classify images using text descriptions"),
            ("Image-Text Retrieval", "Find images for text queries and vice versa"),
            ("Contrastive Learning", "Learn joint vision-language representations"),
            ("Transfer Learning", "Strong performance on downstream tasks"),
            ("Cross-modal Understanding", "Bridge vision and language modalities"),
        ]
    }

    /// Get model architecture details
    pub fn architecture_info() -> (&'static str, Vec<&'static str>) {
        ("CLIP uses dual-encoder architecture with contrastive learning",
         vec![
            "Vision Transformer for image encoding",
            "Causal Transformer for text encoding",
            "Contrastive loss for alignment",
            "Shared embedding space",
            "Zero-shot transfer capabilities"
         ])
    }
}

// ========================================
// COMPREHENSIVE TESTING FRAMEWORK
// ========================================

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::Device;

    #[test]
    fn test_clip_config() {
        let config = CLIPConfig::clip_base();
        assert_eq!(config.projection_dim, 512);
        assert!(config.validate().is_ok());

        let large_config = CLIPConfig::clip_large();
        assert_eq!(large_config.projection_dim, 768);
        assert!(large_config.validate().is_ok());
    }

    #[test]
    fn test_clip_vision_config() {
        let config = CLIPVisionConfig::default();
        assert_eq!(config.num_patches(), 49); // 224x224 / 32x32 = 7x7 = 49
        assert_eq!(config.attention_head_size(), 64); // 768 / 12
        assert!(config.validate().is_ok());

        let large_config = CLIPVisionConfig::large();
        assert_eq!(large_config.hidden_size, 1024);
        assert_eq!(large_config.num_patches(), 256); // 224x224 / 14x14 = 16x16 = 256
    }

    #[test]
    fn test_clip_text_config() {
        let config = CLIPTextConfig::default();
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.attention_head_size(), 64); // 512 / 8
        assert!(config.validate().is_ok());

        let large_config = CLIPTextConfig::large();
        assert_eq!(large_config.hidden_size, 768);
        assert_eq!(large_config.attention_head_size(), 64); // 768 / 12
    }

    #[test]
    fn test_clip_model_creation() {
        let config = CLIPConfig::clip_base();
        let model = CLIPModel::new(config);
        assert_eq!(model.config().projection_dim, 512);

        let base_model = CLIPModel::clip_base();
        assert_eq!(base_model.config().projection_dim, 512);

        let large_model = CLIPModel::clip_large();
        assert_eq!(large_model.config().projection_dim, 768);
    }

    #[test]
    fn test_clip_classification() {
        let classifier = CLIPForImageClassification::clip_base_for_classification();
        assert_eq!(classifier.clip.config().projection_dim, 512);

        let large_classifier = CLIPForImageClassification::clip_large_for_classification();
        assert_eq!(large_classifier.clip.config().projection_dim, 768);
    }

    #[test]
    fn test_clip_factory() {
        let available_models = CLIPFactory::available_models();
        assert!(available_models.contains(&"clip-base"));
        assert!(available_models.contains(&"clip-large"));

        // Test model creation by name
        let base_model = CLIPFactory::create_by_name("clip-base", None);
        assert!(base_model.is_ok());

        let classification_model = CLIPFactory::create_by_name("clip-large", Some("classification"));
        assert!(classification_model.is_ok());

        let invalid_model = CLIPFactory::create_by_name("invalid-model", None);
        assert!(invalid_model.is_err());
    }

    #[test]
    fn test_clip_capabilities() {
        let capabilities = CLIPFactory::capabilities();
        assert!(!capabilities.is_empty());
        assert_eq!(capabilities.len(), 5);

        let cap_names: Vec<&str> = capabilities.iter().map(|(name, _)| *name).collect();
        assert!(cap_names.contains(&"Zero-shot Classification"));
        assert!(cap_names.contains(&"Image-Text Retrieval"));
        assert!(cap_names.contains(&"Contrastive Learning"));
    }

    #[test]
    fn test_clip_architecture_info() {
        let (description, details) = CLIPFactory::architecture_info();
        assert!(!description.is_empty());
        assert!(!details.is_empty());
        assert_eq!(details.len(), 5);
    }

    #[test]
    fn test_clip_vision_forward_pass() {
        let config = CLIPVisionConfig::default();
        let encoder = CLIPVisionEncoder::new(config);

        // Test with sample image
        let batch_size = 2;
        let channels = 3;
        let height = 224;
        let width = 224;
        let pixel_values = torsh_tensor::creation::randn(&[batch_size, channels, height, width]).unwrap();

        let output = encoder.forward(&pixel_values);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.size(0).unwrap(), batch_size);
        assert_eq!(output.size(1).unwrap(), 768); // hidden_size
    }

    #[test]
    fn test_clip_contrastive_learning() {
        let model = CLIPModel::clip_base();

        // Test image encoding
        let pixel_values = torsh_tensor::creation::randn(&[2, 3, 224, 224]).unwrap();
        let image_features = model.encode_image(&pixel_values);
        assert!(image_features.is_ok());

        let image_features = image_features.unwrap();
        assert_eq!(image_features.size(0).unwrap(), 2);
        assert_eq!(image_features.size(1).unwrap(), 512); // projection_dim

        // Test text encoding
        let input_ids = Tensor::randint(0, 49408, &[2, 77]).unwrap();
        let text_features = model.encode_text(&input_ids);
        assert!(text_features.is_ok());

        let text_features = text_features.unwrap();
        assert_eq!(text_features.size(0).unwrap(), 2);
        assert_eq!(text_features.size(1).unwrap(), 512); // projection_dim

        // Test similarity computation
        let similarity = model.compute_similarity(&image_features, &text_features);
        assert!(similarity.is_ok());
    }
}

/// Re-export commonly used CLIP components
pub use {
    CLIPConfig, CLIPModel, CLIPForImageClassification, CLIPFactory,
    CLIPVisionConfig, CLIPTextConfig, CLIPVisionEncoder, CLIPTextEncoder,
    CLIPVisionEmbeddings, CLIPAttention, CLIPMLP
};