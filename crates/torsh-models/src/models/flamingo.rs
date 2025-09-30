//! Flamingo: Few-Shot Learning with Frozen Multimodal Models
//!
//! Flamingo is DeepMind's breakthrough multimodal model that enables few-shot
//! learning by combining a frozen pretrained vision encoder with a language model
//! enhanced with cross-attention mechanisms and a Perceiver Resampler.
//!
//! Key Features:
//! - Frozen CLIP-style vision encoder for extracting visual features
//! - Perceiver Resampler for reducing variable-length visual sequences to fixed tokens
//! - Gated cross-attention layers interleaved in the language model
//! - Support for in-context learning with interleaved vision-language examples
//! - Few-shot adaptation without updating vision encoder weights
//!
//! Architecture Components:
//! - Vision Encoder: Frozen CLIP vision transformer
//! - Perceiver Resampler: Cross-attention based pooling to fixed visual tokens
//! - Language Model: GPT-style decoder with gated cross-attention layers
//! - Cross-Attention: Gated mechanism to inject visual information into text
//!
//! References:
//! - Flamingo: a Visual Language Model for Few-Shot Learning (Alayrac et al.)
//! - Perceiver: General Perception with Iterative Attention (Jaegle et al.)

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

// Import CLIP vision config from the clip module
use super::clip::CLIPVisionConfig;

/// Flamingo Configuration for Few-Shot Learning
#[derive(Debug, Clone)]
pub struct FlamingoConfig {
    /// Vision encoder configuration (frozen CLIP-like encoder)
    pub vision_config: CLIPVisionConfig,
    /// Language model configuration
    pub language_config: FlamingoLanguageConfig,
    /// Perceiver resampler configuration
    pub resampler_config: PerceiverResamplerConfig,
    /// Cross-attention layer configuration
    pub cross_attention_config: CrossAttentionConfig,
    /// Number of visual tokens after resampling
    pub num_visual_tokens: usize,
    /// Whether to freeze the vision encoder
    pub freeze_vision_encoder: bool,
}

impl Default for FlamingoConfig {
    fn default() -> Self {
        Self {
            vision_config: CLIPVisionConfig::default(),
            language_config: FlamingoLanguageConfig::default(),
            resampler_config: PerceiverResamplerConfig::default(),
            cross_attention_config: CrossAttentionConfig::default(),
            num_visual_tokens: 64,
            freeze_vision_encoder: true,
        }
    }
}

impl FlamingoConfig {
    /// Create Flamingo-3B configuration
    pub fn flamingo_3b() -> Self {
        Self {
            vision_config: CLIPVisionConfig::clip_vit_base_patch32(),
            language_config: FlamingoLanguageConfig::gpt_3b(),
            resampler_config: PerceiverResamplerConfig::default(),
            cross_attention_config: CrossAttentionConfig::default(),
            num_visual_tokens: 64,
            freeze_vision_encoder: true,
        }
    }

    /// Create Flamingo-9B configuration
    pub fn flamingo_9b() -> Self {
        Self {
            vision_config: CLIPVisionConfig::clip_vit_large_patch14(),
            language_config: FlamingoLanguageConfig::gpt_9b(),
            resampler_config: PerceiverResamplerConfig::large(),
            cross_attention_config: CrossAttentionConfig::large(),
            num_visual_tokens: 64,
            freeze_vision_encoder: true,
        }
    }

    /// Create Flamingo-80B configuration
    pub fn flamingo_80b() -> Self {
        Self {
            vision_config: CLIPVisionConfig::clip_vit_large_patch14(),
            language_config: FlamingoLanguageConfig::gpt_80b(),
            resampler_config: PerceiverResamplerConfig::large(),
            cross_attention_config: CrossAttentionConfig::large(),
            num_visual_tokens: 64,
            freeze_vision_encoder: true,
        }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.num_visual_tokens == 0 {
            return Err(TorshError::InvalidArgument(
                "num_visual_tokens must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }
}

/// Flamingo Language Model Configuration
#[derive(Debug, Clone)]
pub struct FlamingoLanguageConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f32,
    pub hidden_dropout_prob: f32,
    pub attention_dropout: f32,
    pub use_cache: bool,
}

impl Default for FlamingoLanguageConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50257, // GPT-2 vocab size
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 1024,
            layer_norm_eps: 1e-5,
            hidden_dropout_prob: 0.1,
            attention_dropout: 0.1,
            use_cache: false,
        }
    }
}

impl FlamingoLanguageConfig {
    /// GPT-3B style configuration
    pub fn gpt_3b() -> Self {
        Self {
            vocab_size: 50257,
            hidden_size: 2048,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 8192,
            max_position_embeddings: 2048,
            layer_norm_eps: 1e-5,
            hidden_dropout_prob: 0.1,
            attention_dropout: 0.1,
            use_cache: false,
        }
    }

    /// GPT-9B style configuration
    pub fn gpt_9b() -> Self {
        Self {
            vocab_size: 50257,
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            intermediate_size: 16384,
            max_position_embeddings: 2048,
            layer_norm_eps: 1e-5,
            hidden_dropout_prob: 0.1,
            attention_dropout: 0.1,
            use_cache: false,
        }
    }

    /// GPT-80B style configuration
    pub fn gpt_80b() -> Self {
        Self {
            vocab_size: 50257,
            hidden_size: 8192,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            intermediate_size: 32768,
            max_position_embeddings: 2048,
            layer_norm_eps: 1e-5,
            hidden_dropout_prob: 0.1,
            attention_dropout: 0.1,
            use_cache: false,
        }
    }

    /// Get attention head size
    pub fn attention_head_size(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// Perceiver Resampler Configuration
#[derive(Debug, Clone)]
pub struct PerceiverResamplerConfig {
    pub num_queries: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub ff_mult: usize,
}

impl Default for PerceiverResamplerConfig {
    fn default() -> Self {
        Self {
            num_queries: 64,
            hidden_size: 768,
            num_heads: 12,
            num_layers: 6,
            ff_mult: 4,
        }
    }
}

impl PerceiverResamplerConfig {
    /// Large resampler configuration
    pub fn large() -> Self {
        Self {
            num_queries: 64,
            hidden_size: 1024,
            num_heads: 16,
            num_layers: 8,
            ff_mult: 4,
        }
    }
}

/// Cross-Attention Configuration for Gated Cross-Attention
#[derive(Debug, Clone)]
pub struct CrossAttentionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub dropout: f32,
    pub gate_activation: String, // e.g., "tanh", "sigmoid"
}

impl Default for CrossAttentionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_heads: 12,
            dropout: 0.1,
            gate_activation: "tanh".to_string(),
        }
    }
}

impl CrossAttentionConfig {
    /// Large cross-attention configuration
    pub fn large() -> Self {
        Self {
            hidden_size: 1024,
            num_heads: 16,
            dropout: 0.1,
            gate_activation: "tanh".to_string(),
        }
    }
}

/// Perceiver Resampler for processing visual features into fixed number of tokens
#[derive(Debug)]
pub struct PerceiverResampler {
    learned_queries: Parameter,
    layers: Vec<PerceiverLayer>,
    norm: LayerNorm,
    config: PerceiverResamplerConfig,
}

impl PerceiverResampler {
    pub fn new(config: PerceiverResamplerConfig) -> Result<Self> {
        let learned_queries =
            Parameter::new(creation::randn(&[config.num_queries, config.hidden_size])?);

        let mut layers = Vec::new();
        for _ in 0..config.num_layers {
            layers.push(PerceiverLayer::new(config.clone()));
        }

        let norm = LayerNorm::new(config.hidden_size, 1e-6);

        Ok(Self {
            learned_queries,
            layers,
            norm,
            config,
        })
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters().len()
    }
}

impl Module for PerceiverResampler {
    fn forward(&self, visual_features: &Tensor) -> Result<Tensor> {
        let batch_size = visual_features.size(0)?;

        // Expand learned queries for batch
        let mut x = self.learned_queries.tensor().read().unsqueeze(0)?.expand(&[
            batch_size,
            self.config.num_queries,
            self.config.hidden_size,
        ])?;

        // Process through perceiver layers
        for layer in &self.layers {
            x = layer.forward(&x, visual_features)?;
        }

        // Final normalization
        x = self.norm.forward(&x)?;

        Ok(x)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        params.insert("learned_queries".to_string(), self.learned_queries.clone());

        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("layers.{}.{}", i, name), param);
            }
        }

        for (name, param) in self.norm.parameters() {
            params.insert(format!("norm.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.layers.iter().all(|layer| layer.training())
    }

    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        self.norm.to_device(device)?;
        Ok(())
    }
}

/// Perceiver Layer with cross-attention and feedforward
#[derive(Debug)]
pub struct PerceiverLayer {
    cross_attention: MultiheadAttention,
    self_attention: MultiheadAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    ff: PerceiverFeedForward,
    config: PerceiverResamplerConfig,
}

impl PerceiverLayer {
    pub fn new(config: PerceiverResamplerConfig) -> Self {
        let cross_attention =
            MultiheadAttention::new(config.hidden_size, config.num_heads, 0.0, true);

        let self_attention =
            MultiheadAttention::new(config.hidden_size, config.num_heads, 0.0, true);

        let norm1 = LayerNorm::new(config.hidden_size, 1e-6);
        let norm2 = LayerNorm::new(config.hidden_size, 1e-6);
        let norm3 = LayerNorm::new(config.hidden_size, 1e-6);

        let ff = PerceiverFeedForward::new(config.clone());

        Self {
            cross_attention,
            self_attention,
            norm1,
            norm2,
            norm3,
            ff,
            config,
        }
    }
}

impl Module for PerceiverLayer {
    fn forward(&self, queries: &Tensor, visual_features: &Tensor) -> Result<Tensor> {
        // Cross-attention: queries attend to visual features
        let residual = queries.clone();
        let queries_norm = self.norm1.forward(&residual)?;
        let cross_attn_out = self.cross_attention.forward(
            &queries_norm,
            Some(visual_features),
            Some(visual_features),
            None,
        )?;
        let queries = residual.add(&cross_attn_out)?;

        // Self-attention within queries
        let residual = queries.clone();
        let queries_norm = self.norm2.forward(&residual)?;
        let self_attn_out = self.self_attention.forward(
            &queries_norm,
            Some(&queries_norm),
            Some(&queries_norm),
            None,
        )?;
        let queries = residual.add(&self_attn_out)?;

        // Feedforward
        let residual = queries.clone();
        let queries_norm = self.norm3.forward(&residual)?;
        let ff_out = self.ff.forward(&queries_norm)?;
        let queries = residual.add(&ff_out)?;

        Ok(queries)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.cross_attention.parameters() {
            params.insert(format!("cross_attention.{}", name), param);
        }

        for (name, param) in self.self_attention.parameters() {
            params.insert(format!("self_attention.{}", name), param);
        }

        for (name, param) in self.norm1.parameters() {
            params.insert(format!("norm1.{}", name), param);
        }

        for (name, param) in self.norm2.parameters() {
            params.insert(format!("norm2.{}", name), param);
        }

        for (name, param) in self.norm3.parameters() {
            params.insert(format!("norm3.{}", name), param);
        }

        for (name, param) in self.ff.parameters() {
            params.insert(format!("ff.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.cross_attention.training() && self.self_attention.training() && self.ff.training()
    }

    fn train(&mut self) {
        self.cross_attention.train();
        self.self_attention.train();
        self.ff.train();
    }

    fn eval(&mut self) {
        self.cross_attention.eval();
        self.self_attention.eval();
        self.ff.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.cross_attention.to_device(device)?;
        self.self_attention.to_device(device)?;
        self.norm1.to_device(device)?;
        self.norm2.to_device(device)?;
        self.norm3.to_device(device)?;
        self.ff.to_device(device)?;
        Ok(())
    }
}

/// Feedforward network for Perceiver layer
#[derive(Debug)]
pub struct PerceiverFeedForward {
    fc1: Linear,
    fc2: Linear,
    dropout: Dropout,
    config: PerceiverResamplerConfig,
}

impl PerceiverFeedForward {
    pub fn new(config: PerceiverResamplerConfig) -> Self {
        let intermediate_size = config.hidden_size * config.ff_mult;
        let fc1 = Linear::new(config.hidden_size, intermediate_size, true);
        let fc2 = Linear::new(intermediate_size, config.hidden_size, true);
        let dropout = Dropout::new(0.0);

        Self {
            fc1,
            fc2,
            dropout,
            config,
        }
    }
}

impl Module for PerceiverFeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu()?;
        let x = self.fc2.forward(&x)?;
        let x = self.dropout.forward(&x)?;
        Ok(x)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.fc1.parameters() {
            params.insert(format!("fc1.{}", name), param);
        }
        for (name, param) in self.fc2.parameters() {
            params.insert(format!("fc2.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.dropout.training()
    }

    fn train(&mut self) {
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.dropout.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.fc1.to_device(device)?;
        self.fc2.to_device(device)?;
        Ok(())
    }
}

/// Main Flamingo Model for Few-Shot Learning
#[derive(Debug)]
pub struct FlamingoModel {
    // Note: We'd import this from CLIP module
    // vision_encoder: CLIPVisionTransformer,
    perceiver_resampler: PerceiverResampler,
    language_model: FlamingoLanguageModel,
    config: FlamingoConfig,
}

impl FlamingoModel {
    pub fn new(config: FlamingoConfig) -> Result<Self> {
        config.validate()?;

        // Note: In a complete implementation, we'd create the vision encoder:
        // let vision_encoder = CLIPVisionTransformer::new(config.vision_config.clone())?;

        let perceiver_resampler = PerceiverResampler::new(config.resampler_config.clone())?;
        let language_model = FlamingoLanguageModel::new(
            config.language_config.clone(),
            config.cross_attention_config.clone(),
        );

        Ok(Self {
            // vision_encoder,
            perceiver_resampler,
            language_model,
            config,
        })
    }

    /// Create Flamingo-3B model
    pub fn flamingo_3b() -> Result<Self> {
        Self::new(FlamingoConfig::flamingo_3b())
    }

    /// Create Flamingo-9B model
    pub fn flamingo_9b() -> Result<Self> {
        Self::new(FlamingoConfig::flamingo_9b())
    }

    /// Create Flamingo-80B model
    pub fn flamingo_80b() -> Result<Self> {
        Self::new(FlamingoConfig::flamingo_80b())
    }

    /// Extract visual features from images (simplified without vision encoder)
    pub fn get_visual_features(&self, images: &Tensor) -> Result<Tensor> {
        // Note: In a complete implementation:
        // let visual_features = self.vision_encoder.forward(images)?;
        // For now, assume visual features are already provided

        let visual_features = images; // Placeholder

        // Add sequence dimension for resampler if needed
        let visual_features = if visual_features.dim() == 3 {
            visual_features.unsqueeze(1)?
        } else {
            visual_features.clone()
        };

        // Resample to fixed number of visual tokens
        let visual_tokens = self.perceiver_resampler.forward(&visual_features)?;

        Ok(visual_tokens)
    }

    /// Generate text conditioned on visual input
    pub fn generate(&self, input_ids: &Tensor, images: Option<&Tensor>) -> Result<Tensor> {
        // Extract visual features if images are provided
        let visual_features = if let Some(images) = images {
            Some(self.get_visual_features(images)?)
        } else {
            None
        };

        // Generate text using language model
        self.language_model
            .forward_with_vision(input_ids, visual_features.as_ref())
    }

    /// Get configuration
    pub fn config(&self) -> &FlamingoConfig {
        &self.config
    }

    /// Get model size information
    pub fn model_size_info(&self) -> String {
        let params = self.language_model.parameters().len();
        format!(
            "Flamingo Model: ~{}M parameters, {} layers, {} heads",
            params / 1_000_000,
            self.config.language_config.num_hidden_layers,
            self.config.language_config.num_attention_heads
        )
    }
}

impl Module for FlamingoModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified forward that assumes input is text tokens
        self.language_model.forward(input)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        // Include vision encoder parameters only if not frozen
        // Note: In complete implementation:
        // if !self.config.freeze_vision_encoder {
        //     for (name, param) in self.vision_encoder.parameters() {
        //         params.insert(format!("vision_encoder.{}", name), param);
        //     }
        // }

        for (name, param) in self.perceiver_resampler.parameters() {
            params.insert(format!("perceiver_resampler.{}", name), param);
        }

        for (name, param) in self.language_model.parameters() {
            params.insert(format!("language_model.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.perceiver_resampler.training() && self.language_model.training()
    }

    fn train(&mut self) {
        if !self.config.freeze_vision_encoder {
            // Note: self.vision_encoder.train();
        }
        self.perceiver_resampler.train();
        self.language_model.train();
    }

    fn eval(&mut self) {
        // Always eval vision encoder (frozen)
        // Note: self.vision_encoder.eval();
        self.perceiver_resampler.eval();
        self.language_model.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        // Note: self.vision_encoder.to_device(device)?;
        self.perceiver_resampler.to_device(device)?;
        self.language_model.to_device(device)?;
        Ok(())
    }
}

/// Flamingo Language Model with cross-attention layers
#[derive(Debug)]
pub struct FlamingoLanguageModel {
    embeddings: Embedding,
    position_embeddings: Embedding,
    layers: Vec<FlamingoLanguageLayer>,
    final_norm: LayerNorm,
    lm_head: Linear,
    config: FlamingoLanguageConfig,
}

impl FlamingoLanguageModel {
    pub fn new(
        config: FlamingoLanguageConfig,
        cross_attention_config: CrossAttentionConfig,
    ) -> Self {
        let embeddings = Embedding::new(config.vocab_size, config.hidden_size);
        let position_embeddings =
            Embedding::new(config.max_position_embeddings, config.hidden_size);

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            // Add cross-attention to every few layers (e.g., every 4th layer)
            let cross_attn_config = if i % 4 == 0 {
                Some(cross_attention_config.clone())
            } else {
                None
            };
            layers.push(FlamingoLanguageLayer::new(&config, cross_attn_config));
        }

        let final_norm = LayerNorm::new(config.hidden_size, config.layer_norm_eps);
        let lm_head = Linear::new(config.hidden_size, config.vocab_size, false);

        Self {
            embeddings,
            position_embeddings,
            layers,
            final_norm,
            lm_head,
            config,
        }
    }

    /// Forward pass with optional visual features
    pub fn forward_with_vision(
        &self,
        input_ids: &Tensor,
        visual_features: Option<&Tensor>,
    ) -> Result<Tensor> {
        let seq_length = input_ids.size(1)?;
        let batch_size = input_ids.size(0)?;

        // Token embeddings
        let mut hidden_states = self.embeddings.forward(input_ids)?;

        // Position embeddings
        let position_ids = creation::arange(0i64, seq_length as i64, 1i64)?
            .unsqueeze(0)?
            .expand(&[batch_size, seq_length])?;
        let position_embeds = self.position_embeddings.forward(&position_ids)?;

        hidden_states = hidden_states.add(&position_embeds)?;

        // Process through layers
        for layer in &self.layers {
            // Note: In a full implementation, we'd pass visual_features to layers with cross-attention
            // For now, we only do text processing
            hidden_states = layer.forward(&hidden_states)?;
        }

        // Final normalization and language modeling head
        hidden_states = self.final_norm.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&hidden_states)?;

        Ok(logits)
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        self.parameters().len()
    }
}

impl Module for FlamingoLanguageModel {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_with_vision(input_ids, None)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.embeddings.parameters() {
            params.insert(format!("embeddings.{}", name), param);
        }

        for (name, param) in self.position_embeddings.parameters() {
            params.insert(format!("position_embeddings.{}", name), param);
        }

        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("layers.{}.{}", i, name), param);
            }
        }

        for (name, param) in self.final_norm.parameters() {
            params.insert(format!("final_norm.{}", name), param);
        }

        for (name, param) in self.lm_head.parameters() {
            params.insert(format!("lm_head.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.layers.iter().all(|layer| layer.training())
    }

    fn train(&mut self) {
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.embeddings.to_device(device)?;
        self.position_embeddings.to_device(device)?;
        for layer in &mut self.layers {
            layer.to_device(device)?;
        }
        self.final_norm.to_device(device)?;
        self.lm_head.to_device(device)?;
        Ok(())
    }
}

/// Gated Cross-Attention for injecting visual information into language model
#[derive(Debug)]
pub struct GatedCrossAttention {
    attention: MultiheadAttention,
    gate_linear: Linear,
    gate_activation: String,
    dropout: Dropout,
    norm: LayerNorm,
    config: CrossAttentionConfig,
}

impl GatedCrossAttention {
    pub fn new(config: CrossAttentionConfig) -> Self {
        let attention =
            MultiheadAttention::new(config.hidden_size, config.num_heads, config.dropout, true);

        let gate_linear = Linear::new(config.hidden_size, config.hidden_size, true);
        let dropout = Dropout::new(config.dropout);
        let norm = LayerNorm::new(config.hidden_size, 1e-5);

        Self {
            attention,
            gate_linear,
            gate_activation: config.gate_activation.clone(),
            dropout,
            norm,
            config,
        }
    }

    /// Forward pass with text and visual features
    pub fn forward_with_vision(&self, text_features: &Tensor, visual_features: &Tensor) -> Result<Tensor> {
        // Cross-attention: text attends to visual features
        let attn_output = self.attention.forward(
            text_features,
            Some(visual_features),
            Some(visual_features),
            None,
        )?;
        let attn_output = self.dropout.forward(&attn_output)?;

        // Gating mechanism
        let gate = self.gate_linear.forward(text_features)?;
        let gate = match self.gate_activation.as_str() {
            "tanh" => gate.tanh()?,
            "sigmoid" => gate.sigmoid()?,
            _ => gate.tanh()?, // default to tanh
        };

        // Apply gating
        let gated_output = attn_output.mul(&gate)?;

        // Residual connection and normalization
        let output = text_features.add(&gated_output)?;
        self.norm.forward(&output)
    }
}

impl Module for GatedCrossAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified forward without visual features
        // In practice, this would be called with both text and visual features
        input.clone()
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.attention.parameters() {
            params.insert(format!("attention.{}", name), param);
        }

        for (name, param) in self.gate_linear.parameters() {
            params.insert(format!("gate_linear.{}", name), param);
        }

        for (name, param) in self.norm.parameters() {
            params.insert(format!("norm.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.attention.training() && self.dropout.training()
    }

    fn train(&mut self) {
        self.attention.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.attention.eval();
        self.dropout.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.attention.to_device(device)?;
        self.gate_linear.to_device(device)?;
        self.norm.to_device(device)?;
        Ok(())
    }
}

/// Flamingo Language Model Layer with optional cross-attention
#[derive(Debug)]
pub struct FlamingoLanguageLayer {
    self_attention: MultiheadAttention,
    cross_attention: Option<GatedCrossAttention>,
    feedforward: FlamingoFeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    has_cross_attention: bool,
}

impl FlamingoLanguageLayer {
    pub fn new(
        config: &FlamingoLanguageConfig,
        cross_attention_config: Option<CrossAttentionConfig>,
    ) -> Self {
        let self_attention = MultiheadAttention::new(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_dropout,
            true,
        );

        let cross_attention = cross_attention_config.map(GatedCrossAttention::new);
        let has_cross_attention = cross_attention.is_some();

        let feedforward = FlamingoFeedForward::new(config);

        let norm1 = LayerNorm::new(config.hidden_size, config.layer_norm_eps);
        let norm2 = LayerNorm::new(config.hidden_size, config.layer_norm_eps);
        let norm3 = LayerNorm::new(config.hidden_size, config.layer_norm_eps);

        Self {
            self_attention,
            cross_attention,
            feedforward,
            norm1,
            norm2,
            norm3,
            has_cross_attention,
        }
    }

    /// Forward pass with optional visual features
    pub fn forward_with_vision(&self, hidden_states: &Tensor, visual_features: Option<&Tensor>) -> Result<Tensor> {
        // Self-attention
        let residual = hidden_states.clone();
        let hidden_states = self.norm1.forward(&residual)?;
        let hidden_states = self.self_attention.forward(
            &hidden_states,
            Some(&hidden_states),
            Some(&hidden_states),
            None,
        )?;
        let hidden_states = residual.add(&hidden_states)?;

        // Cross-attention (if layer has it and visual features are provided)
        let hidden_states = if let (Some(cross_attn), Some(visual_features)) = (&self.cross_attention, visual_features) {
            let residual = hidden_states.clone();
            let norm_hidden = self.norm2.forward(&residual)?;
            let cross_output = cross_attn.forward_with_vision(&norm_hidden, visual_features)?;
            residual.add(&cross_output)?
        } else {
            hidden_states
        };

        // Feedforward
        let residual = hidden_states.clone();
        let hidden_states = self.norm3.forward(&residual)?;
        let hidden_states = self.feedforward.forward(&hidden_states)?;
        let hidden_states = residual.add(&hidden_states)?;

        Ok(hidden_states)
    }
}

impl Module for FlamingoLanguageLayer {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Simplified forward pass for text-only
        self.forward_with_vision(hidden_states, None)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.self_attention.parameters() {
            params.insert(format!("self_attention.{}", name), param);
        }

        if let Some(cross_attention) = &self.cross_attention {
            for (name, param) in cross_attention.parameters() {
                params.insert(format!("cross_attention.{}", name), param);
            }
        }

        for (name, param) in self.feedforward.parameters() {
            params.insert(format!("feedforward.{}", name), param);
        }

        for (name, param) in self.norm1.parameters() {
            params.insert(format!("norm1.{}", name), param);
        }

        for (name, param) in self.norm2.parameters() {
            params.insert(format!("norm2.{}", name), param);
        }

        for (name, param) in self.norm3.parameters() {
            params.insert(format!("norm3.{}", name), param);
        }

        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        let mut training = self.self_attention.training() && self.feedforward.training();
        if let Some(cross_attention) = &self.cross_attention {
            training = training && cross_attention.training();
        }
        training
    }

    fn train(&mut self) {
        self.self_attention.train();
        if let Some(cross_attention) = &mut self.cross_attention {
            cross_attention.train();
        }
        self.feedforward.train();
    }

    fn eval(&mut self) {
        self.self_attention.eval();
        if let Some(cross_attention) = &mut self.cross_attention {
            cross_attention.eval();
        }
        self.feedforward.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.self_attention.to_device(device)?;
        if let Some(cross_attention) = &mut self.cross_attention {
            cross_attention.to_device(device)?;
        }
        self.feedforward.to_device(device)?;
        self.norm1.to_device(device)?;
        self.norm2.to_device(device)?;
        self.norm3.to_device(device)?;
        Ok(())
    }
}

/// Feedforward network for Flamingo language layer
#[derive(Debug)]
pub struct FlamingoFeedForward {
    fc1: Linear,
    fc2: Linear,
    dropout: Dropout,
}

impl FlamingoFeedForward {
    pub fn new(config: &FlamingoLanguageConfig) -> Self {
        let fc1 = Linear::new(config.hidden_size, config.intermediate_size, true);
        let fc2 = Linear::new(config.intermediate_size, config.hidden_size, true);
        let dropout = Dropout::new(config.hidden_dropout_prob);

        Self { fc1, fc2, dropout }
    }
}

impl Module for FlamingoFeedForward {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.fc1.forward(hidden_states)?;
        let hidden_states = hidden_states.gelu()?;
        let hidden_states = self.fc2.forward(&hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        Ok(hidden_states)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.fc1.parameters() {
            params.insert(format!("fc1.{}", name), param);
        }
        for (name, param) in self.fc2.parameters() {
            params.insert(format!("fc2.{}", name), param);
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.dropout.training()
    }

    fn train(&mut self) {
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.dropout.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.fc1.to_device(device)?;
        self.fc2.to_device(device)?;
        Ok(())
    }
}

/// Flamingo Factory Functions and Utilities
pub struct FlamingoFactory;

impl FlamingoFactory {
    /// Create Flamingo model by name
    pub fn create_by_name(model_name: &str) -> Result<FlamingoModel> {
        match model_name.to_lowercase().as_str() {
            "flamingo-3b" | "flamingo_3b" => FlamingoModel::flamingo_3b(),
            "flamingo-9b" | "flamingo_9b" => FlamingoModel::flamingo_9b(),
            "flamingo-80b" | "flamingo_80b" => FlamingoModel::flamingo_80b(),
            _ => Err(TorshError::InvalidArgument(format!(
                "Unknown Flamingo model: {}. Available: flamingo-3b, flamingo-9b, flamingo-80b",
                model_name
            ))),
        }
    }

    /// Get model information
    pub fn model_info(model_name: &str) -> Result<String> {
        match model_name.to_lowercase().as_str() {
            "flamingo-3b" | "flamingo_3b" => Ok(format!(
                "Flamingo-3B: ~3B parameters, 24 layers, 16 heads, frozen CLIP vision encoder"
            )),
            "flamingo-9b" | "flamingo_9b" => Ok(format!(
                "Flamingo-9B: ~9B parameters, 32 layers, 32 heads, frozen CLIP-Large vision encoder"
            )),
            "flamingo-80b" | "flamingo_80b" => Ok(format!(
                "Flamingo-80B: ~80B parameters, 80 layers, 64 heads, frozen CLIP-Large vision encoder"
            )),
            _ => Err(TorshError::InvalidArgument(format!(
                "Unknown Flamingo model: {}",
                model_name
            ))),
        }
    }

    /// Compare Flamingo with other multimodal models
    pub fn comparison_info() -> String {
        format!(
            "Flamingo vs Other Models:\n\
            - Flamingo: Few-shot learning, frozen vision encoder, gated cross-attention\n\
            - CLIP: Contrastive learning, joint training, dual encoders\n\
            - BLIP: Bootstrapped learning, unified encoder-decoder architecture\n\
            - ALIGN: Large-scale noisy data, EfficientNet + BERT"
        )
    }

    /// Get few-shot learning capabilities info
    pub fn few_shot_info() -> String {
        format!(
            "Flamingo Few-Shot Learning:\n\
            - In-context learning with interleaved vision-language examples\n\
            - Frozen vision encoder preserves pretrained visual representations\n\
            - Gated cross-attention allows selective information flow\n\
            - Perceiver resampler handles variable-length visual sequences\n\
            - Supports various tasks: VQA, captioning, classification"
        )
    }
}

// Comprehensive test suite for Flamingo models
#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation;

    #[test]
    fn test_flamingo_config_creation() {
        let config = FlamingoConfig::flamingo_3b();
        assert_eq!(config.num_visual_tokens, 64);
        assert!(config.freeze_vision_encoder);

        let config_9b = FlamingoConfig::flamingo_9b();
        assert_eq!(config_9b.language_config.hidden_size, 4096);

        let config_80b = FlamingoConfig::flamingo_80b();
        assert_eq!(config_80b.language_config.num_hidden_layers, 80);
    }

    #[test]
    fn test_flamingo_config_validation() {
        let mut config = FlamingoConfig::default();
        assert!(config.validate().is_ok());

        config.num_visual_tokens = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_language_config_variants() {
        let gpt_3b = FlamingoLanguageConfig::gpt_3b();
        assert_eq!(gpt_3b.hidden_size, 2048);
        assert_eq!(gpt_3b.num_hidden_layers, 24);

        let gpt_9b = FlamingoLanguageConfig::gpt_9b();
        assert_eq!(gpt_9b.hidden_size, 4096);
        assert_eq!(gpt_9b.num_hidden_layers, 32);

        let gpt_80b = FlamingoLanguageConfig::gpt_80b();
        assert_eq!(gpt_80b.hidden_size, 8192);
        assert_eq!(gpt_80b.num_hidden_layers, 80);
    }

    #[test]
    fn test_perceiver_resampler_config() {
        let config = PerceiverResamplerConfig::default();
        assert_eq!(config.num_queries, 64);
        assert_eq!(config.hidden_size, 768);

        let large_config = PerceiverResamplerConfig::large();
        assert_eq!(large_config.hidden_size, 1024);
    }

    #[test]
    fn test_cross_attention_config() {
        let config = CrossAttentionConfig::default();
        assert_eq!(config.gate_activation, "tanh");
        assert_eq!(config.num_heads, 12);

        let large_config = CrossAttentionConfig::large();
        assert_eq!(large_config.hidden_size, 1024);
    }

    #[test]
    fn test_flamingo_model_creation() {
        let model = FlamingoModel::flamingo_3b();
        assert!(model.is_ok());

        let model_9b = FlamingoModel::flamingo_9b();
        assert!(model_9b.is_ok());
    }

    #[test]
    fn test_perceiver_resampler_creation() {
        let config = PerceiverResamplerConfig::default();
        let resampler = PerceiverResampler::new(config);
        assert!(resampler.is_ok());
    }

    #[test]
    fn test_flamingo_factory() {
        let model = FlamingoFactory::create_by_name("flamingo-3b");
        assert!(model.is_ok());

        let model_9b = FlamingoFactory::create_by_name("flamingo-9b");
        assert!(model_9b.is_ok());

        let invalid_model = FlamingoFactory::create_by_name("invalid");
        assert!(invalid_model.is_err());
    }

    #[test]
    fn test_model_info() {
        let info = FlamingoFactory::model_info("flamingo-3b");
        assert!(info.is_ok());
        assert!(info.unwrap().contains("3B"));

        let info_9b = FlamingoFactory::model_info("flamingo-9b");
        assert!(info_9b.is_ok());
        assert!(info_9b.unwrap().contains("9B"));
    }

    #[test]
    fn test_comparison_info() {
        let comparison = FlamingoFactory::comparison_info();
        assert!(comparison.contains("Flamingo"));
        assert!(comparison.contains("few-shot"));
        assert!(comparison.contains("frozen"));
    }

    #[test]
    fn test_few_shot_info() {
        let info = FlamingoFactory::few_shot_info();
        assert!(info.contains("in-context"));
        assert!(info.contains("frozen"));
        assert!(info.contains("cross-attention"));
    }

    #[test]
    fn test_attention_head_size() {
        let config = FlamingoLanguageConfig::gpt_3b();
        assert_eq!(config.attention_head_size(), 128); // 2048 / 16

        let config_9b = FlamingoLanguageConfig::gpt_9b();
        assert_eq!(config_9b.attention_head_size(), 128); // 4096 / 32
    }

    #[test]
    fn test_gated_cross_attention() {
        let config = CrossAttentionConfig::default();
        let gca = GatedCrossAttention::new(config);

        // Test that parameters are created
        let params = gca.parameters();
        assert!(params.contains_key("attention.in_proj_weight"));
        assert!(params.contains_key("gate_linear.weight"));
    }

    #[test]
    fn test_flamingo_language_layer() {
        let lang_config = FlamingoLanguageConfig::default();
        let cross_config = CrossAttentionConfig::default();

        // Test layer with cross-attention
        let layer_with_cross = FlamingoLanguageLayer::new(&lang_config, Some(cross_config));
        assert!(layer_with_cross.has_cross_attention);

        // Test layer without cross-attention
        let layer_without_cross = FlamingoLanguageLayer::new(&lang_config, None);
        assert!(!layer_without_cross.has_cross_attention);
    }
}