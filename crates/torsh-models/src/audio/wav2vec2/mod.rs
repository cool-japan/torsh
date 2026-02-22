//! Wav2Vec2 models for self-supervised speech representation learning
//!
//! Implementation of Wav2Vec2 architecture for speech representation learning.
//!
//! Reference: [Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)

use std::collections::HashMap;
use torsh_core::{error::Result, DeviceType};
use torsh_nn::prelude::{Conv1d, Dropout, LayerNorm, Linear};
use torsh_nn::{Module, Parameter};
use torsh_tensor::Tensor;

/// Wav2Vec2 Configuration
#[derive(Debug, Clone)]
pub struct Wav2Vec2Config {
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
    pub ctc_loss_reduction: String,
    pub ctc_zero_infinity: bool,
    pub use_weighted_layer_sum: bool,
}

impl Default for Wav2Vec2Config {
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
            ctc_loss_reduction: "mean".to_string(),
            ctc_zero_infinity: false,
            use_weighted_layer_sum: false,
        }
    }
}

impl Wav2Vec2Config {
    /// Create configuration for Wav2Vec2 Base model
    pub fn base() -> Self {
        Self::default()
    }

    /// Create configuration for Wav2Vec2 Large model
    pub fn large() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            ..Self::default()
        }
    }

    /// Create configuration for fine-tuning
    pub fn for_finetuning(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            ..Self::default()
        }
    }
}

/// Wav2Vec2 Feature Extractor - processes raw audio waveforms
pub struct Wav2Vec2FeatureExtractor {
    conv_layers: Vec<Conv1d>,
    config: Wav2Vec2Config,
}

impl Wav2Vec2FeatureExtractor {
    pub fn new(config: Wav2Vec2Config) -> Self {
        let mut conv_layers = Vec::new();
        let mut in_dim = 1; // Raw audio input

        for i in 0..config.conv_dim.len() {
            let out_dim = config.conv_dim[i];
            let kernel_size = config.conv_kernel[i];
            let stride = config.conv_stride[i];

            let conv = Conv1d::new(
                in_dim,
                out_dim,
                kernel_size,
                stride,
                kernel_size / 2,  // padding
                1,                // dilation
                config.conv_bias, // bias
                1,                // groups
            );
            conv_layers.push(conv);
            in_dim = out_dim;
        }

        Self {
            conv_layers,
            config,
        }
    }
}

impl Module for Wav2Vec2FeatureExtractor {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut hidden_states = input.clone();

        for conv_layer in &self.conv_layers {
            hidden_states = conv_layer.forward(&hidden_states)?;

            // Apply activation function
            match self.config.feat_extract_activation.as_str() {
                "gelu" => hidden_states = hidden_states.gelu()?,
                "relu" => hidden_states = hidden_states.relu()?,
                _ => hidden_states = hidden_states.gelu()?,
            }
        }

        Ok(hidden_states)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (i, layer) in self.conv_layers.iter().enumerate() {
            for (name, param) in layer.parameters() {
                params.insert(format!("conv_layers.{}.{}", i, name), param);
            }
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.parameters()
    }

    fn training(&self) -> bool {
        self.conv_layers.iter().all(|layer| layer.training())
    }

    fn train(&mut self) {
        for layer in &mut self.conv_layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        for layer in &mut self.conv_layers {
            layer.eval();
        }
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        for layer in &mut self.conv_layers {
            layer.to_device(device)?;
        }
        Ok(())
    }
}

/// Wav2Vec2 Feature Projection - projects extracted features to hidden size
pub struct Wav2Vec2FeatureProjection {
    layer_norm: LayerNorm,
    projection: Linear,
    dropout: Dropout,
}

impl Wav2Vec2FeatureProjection {
    pub fn new(config: &Wav2Vec2Config) -> Self {
        let in_dim = *config.conv_dim.last().unwrap_or(&512);
        let layer_norm = LayerNorm::new(vec![in_dim], 1e-5, false, DeviceType::Cpu)
            .expect("Failed to create LayerNorm");
        let projection = Linear::new(in_dim, config.hidden_size, true);
        let dropout = Dropout::new(config.feat_proj_dropout);

        Self {
            layer_norm,
            projection,
            dropout,
        }
    }
}

impl Module for Wav2Vec2FeatureProjection {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden_states = self.layer_norm.forward(input)?;
        let hidden_states = self.projection.forward(&hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states)?;
        Ok(hidden_states)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (name, param) in self.layer_norm.parameters() {
            params.insert(format!("layer_norm.{}", name), param);
        }
        for (name, param) in self.projection.parameters() {
            params.insert(format!("projection.{}", name), param);
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
        self.layer_norm.to_device(device)?;
        self.projection.to_device(device)?;
        Ok(())
    }
}

// Forward declarations for remaining components (to be implemented)
pub struct Wav2Vec2PositionalConvEmbedding;
pub struct Wav2Vec2Attention;
pub struct Wav2Vec2FeedForward;
pub struct Wav2Vec2EncoderLayer;
pub struct Wav2Vec2Encoder;
pub struct Wav2Vec2Model;

/// Wav2Vec2 model for Connectionist Temporal Classification (CTC)
pub struct Wav2Vec2ForCTC {
    feature_extractor: Wav2Vec2FeatureExtractor,
    feature_projection: Wav2Vec2FeatureProjection,
    lm_head: Linear,
    config: Wav2Vec2Config,
}

impl Wav2Vec2ForCTC {
    pub fn new(config: Wav2Vec2Config) -> Result<Self> {
        let feature_extractor = Wav2Vec2FeatureExtractor::new(config.clone());
        let feature_projection = Wav2Vec2FeatureProjection::new(&config);
        let lm_head = Linear::new(config.hidden_size, config.vocab_size, true);

        Ok(Self {
            feature_extractor,
            feature_projection,
            lm_head,
            config,
        })
    }
}

impl Module for Wav2Vec2ForCTC {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let features = self.feature_extractor.forward(input)?;
        let projected = self.feature_projection.forward(&features)?;
        let logits = self.lm_head.forward(&projected)?;
        Ok(logits)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.feature_extractor.parameters() {
            params.insert(format!("feature_extractor.{}", name), param);
        }

        for (name, param) in self.feature_projection.parameters() {
            params.insert(format!("feature_projection.{}", name), param);
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
        self.feature_extractor.training()
            && self.feature_projection.training()
            && self.lm_head.training()
    }

    fn train(&mut self) {
        self.feature_extractor.train();
        self.feature_projection.train();
        self.lm_head.train();
    }

    fn eval(&mut self) {
        self.feature_extractor.eval();
        self.feature_projection.eval();
        self.lm_head.eval();
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.feature_extractor.to_device(device)?;
        self.feature_projection.to_device(device)?;
        self.lm_head.to_device(device)?;
        Ok(())
    }
}

// Note: Key types (Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2FeatureProjection) are already public
// Removed redundant re-export to fix duplicate definition errors
