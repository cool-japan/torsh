//! Model configuration system for parameterizing architectures

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Base model configuration trait
pub trait ModelConfig: Clone + Default {
    /// Get the model name
    fn model_name(&self) -> String;

    /// Get model variant/size description
    fn variant(&self) -> String;

    /// Validate configuration parameters
    fn validate(&self) -> Result<(), String>;

    /// Get estimated parameter count
    fn estimated_parameters(&self) -> u64;
}

/// Vision model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionModelConfig {
    /// Model architecture type
    pub architecture: VisionArchitecture,
    /// Input image size (height, width)
    pub input_size: (usize, usize),
    /// Number of input channels (usually 3 for RGB)
    pub in_channels: usize,
    /// Number of output classes
    pub num_classes: usize,
    /// Architecture-specific parameters
    pub arch_params: VisionArchParams,
    /// Training hyperparameters
    pub training: TrainingConfig,
}

/// Vision architecture types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisionArchitecture {
    ResNet,
    EfficientNet,
    VisionTransformer,
    MobileNet,
    DenseNet,
    ConvNeXt,
    Swin,
}

/// Architecture-specific parameters for vision models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VisionArchParams {
    ResNet(ResNetConfig),
    EfficientNet(EfficientNetConfig),
    VisionTransformer(ViTConfig),
    MobileNet(MobileNetConfig),
    DenseNet(DenseNetConfig),
}

/// ResNet configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResNetConfig {
    /// Number of layers in each stage [stage1, stage2, stage3, stage4]
    pub layers: [usize; 4],
    /// Use bottleneck blocks (for ResNet-50+)
    pub bottleneck: bool,
    /// Width multiplier for channels
    pub width_mult: f32,
    /// Stem convolution type
    pub stem_type: StemType,
}

/// EfficientNet configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficientNetConfig {
    /// Width scaling factor
    pub width_mult: f32,
    /// Depth scaling factor
    pub depth_mult: f32,
    /// Resolution scaling factor
    pub resolution_mult: f32,
    /// Dropout rate
    pub dropout_rate: f32,
    /// Squeeze-and-Excitation ratio
    pub se_ratio: f32,
}

/// Vision Transformer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViTConfig {
    /// Patch size (assumed square)
    pub patch_size: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Number of transformer layers
    pub depth: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// MLP expansion ratio
    pub mlp_ratio: f32,
    /// Dropout rate
    pub dropout_rate: f32,
    /// Attention dropout rate
    pub attn_dropout_rate: f32,
    /// Position embedding type
    pub pos_embed_type: PositionEmbedType,
}

/// MobileNet configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileNetConfig {
    /// MobileNet version
    pub version: MobileNetVersion,
    /// Width multiplier
    pub width_mult: f32,
    /// Minimum channel divisor
    pub min_ch: usize,
    /// Dropout rate
    pub dropout_rate: f32,
}

/// DenseNet configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseNetConfig {
    /// Growth rate (number of channels added per layer)
    pub growth_rate: usize,
    /// Number of layers in each dense block
    pub block_config: Vec<usize>,
    /// Number of initial features
    pub num_init_features: usize,
    /// Bottleneck width multiplier
    pub bn_size: usize,
    /// Compression factor in transition layers
    pub compression: f32,
}

/// NLP model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NlpModelConfig {
    /// Model architecture type
    pub architecture: NlpArchitecture,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_length: usize,
    /// Architecture-specific parameters
    pub arch_params: NlpArchParams,
    /// Training hyperparameters
    pub training: TrainingConfig,
}

/// NLP architecture types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NlpArchitecture {
    BERT,
    GPT,
    T5,
    RoBERTa,
    BART,
    ELECTRA,
    DeBERTa,
}

/// Architecture-specific parameters for NLP models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NlpArchParams {
    BERT(BERTConfig),
    GPT(GPTConfig),
    T5(T5Config),
    RoBERTa(RoBERTaConfig),
    BART(BARTConfig),
}

/// BERT configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BERTConfig {
    /// Hidden size
    pub hidden_size: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Intermediate size in feed-forward layers
    pub intermediate_size: usize,
    /// Hidden activation function
    pub hidden_act: String,
    /// Hidden dropout probability
    pub hidden_dropout_prob: f32,
    /// Attention dropout probability
    pub attention_probs_dropout_prob: f32,
    /// Layer norm epsilon
    pub layer_norm_eps: f32,
    /// Use absolute position embeddings
    pub use_absolute_pos: bool,
}

/// GPT configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPTConfig {
    /// Embedding/hidden size
    pub n_embd: usize,
    /// Number of layers
    pub n_layer: usize,
    /// Number of attention heads
    pub n_head: usize,
    /// Context length
    pub n_ctx: usize,
    /// Residual dropout
    pub resid_pdrop: f32,
    /// Embedding dropout
    pub embd_pdrop: f32,
    /// Attention dropout
    pub attn_pdrop: f32,
    /// Use bias in linear layers
    pub use_bias: bool,
}

/// T5 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T5Config {
    /// Model dimension
    pub d_model: usize,
    /// Key/value dimension
    pub d_kv: usize,
    /// Feed-forward dimension
    pub d_ff: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of heads
    pub num_heads: usize,
    /// Relative attention bucket size
    pub relative_attention_num_buckets: usize,
    /// Dropout rate
    pub dropout_rate: f32,
    /// Layer norm epsilon
    pub layer_norm_epsilon: f32,
}

/// RoBERTa configuration (extends BERT)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoBERTaConfig {
    /// Base BERT configuration
    pub bert_config: BERTConfig,
    /// Use different layer norm
    pub use_alternate_layernorm: bool,
}

/// BART configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BARTConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Model dimension
    pub d_model: usize,
    /// Encoder layers
    pub encoder_layers: usize,
    /// Decoder layers
    pub decoder_layers: usize,
    /// Encoder attention heads
    pub encoder_attention_heads: usize,
    /// Decoder attention heads  
    pub decoder_attention_heads: usize,
    /// Feed-forward dimension
    pub encoder_ffn_dim: usize,
    pub decoder_ffn_dim: usize,
    /// Dropout
    pub dropout: f32,
    /// Attention dropout
    pub attention_dropout: f32,
    /// Activation function
    pub activation_function: String,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Batch size
    pub batch_size: usize,
    /// Number of epochs
    pub epochs: usize,
    /// Weight decay
    pub weight_decay: f32,
    /// Optimizer type
    pub optimizer: OptimizerType,
    /// Learning rate scheduler
    pub lr_scheduler: LRSchedulerType,
    /// Gradient clipping
    pub max_grad_norm: Option<f32>,
    /// Warmup steps
    pub warmup_steps: Option<usize>,
}

/// Optimizer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    SGD { momentum: f32, nesterov: bool },
    Adam { beta1: f32, beta2: f32, eps: f32 },
    AdamW { beta1: f32, beta2: f32, eps: f32 },
    RMSprop { alpha: f32, eps: f32 },
}

/// Learning rate scheduler types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LRSchedulerType {
    Constant,
    Linear { total_steps: usize },
    Cosine { total_steps: usize, min_lr: f32 },
    StepLR { step_size: usize, gamma: f32 },
    ExponentialLR { gamma: f32 },
}

/// Stem convolution types for ResNet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StemType {
    /// Standard 7x7 conv with stride 2
    Standard,
    /// Deep stem with multiple 3x3 convs
    Deep,
    /// Patch-like stem for better fine-grained features
    Patch,
}

/// Position embedding types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionEmbedType {
    Learned,
    Sinusoidal,
    Relative,
    Rotary,
}

/// MobileNet version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MobileNetVersion {
    V1,
    V2,
    V3Small,
    V3Large,
}

impl Default for VisionModelConfig {
    fn default() -> Self {
        Self {
            architecture: VisionArchitecture::ResNet,
            input_size: (224, 224),
            in_channels: 3,
            num_classes: 1000,
            arch_params: VisionArchParams::ResNet(ResNetConfig::default()),
            training: TrainingConfig::default(),
        }
    }
}

impl Default for ResNetConfig {
    fn default() -> Self {
        Self {
            layers: [2, 2, 2, 2], // ResNet-18
            bottleneck: false,
            width_mult: 1.0,
            stem_type: StemType::Standard,
        }
    }
}

impl Default for EfficientNetConfig {
    fn default() -> Self {
        Self {
            width_mult: 1.0,
            depth_mult: 1.0,
            resolution_mult: 1.0,
            dropout_rate: 0.2,
            se_ratio: 0.25,
        }
    }
}

impl Default for ViTConfig {
    fn default() -> Self {
        Self {
            patch_size: 16,
            embed_dim: 768,
            depth: 12,
            num_heads: 12,
            mlp_ratio: 4.0,
            dropout_rate: 0.1,
            attn_dropout_rate: 0.0,
            pos_embed_type: PositionEmbedType::Learned,
        }
    }
}

impl Default for MobileNetConfig {
    fn default() -> Self {
        Self {
            version: MobileNetVersion::V2,
            width_mult: 1.0,
            min_ch: 8,
            dropout_rate: 0.2,
        }
    }
}

impl Default for DenseNetConfig {
    fn default() -> Self {
        Self {
            growth_rate: 32,
            block_config: vec![6, 12, 24, 16], // DenseNet-121
            num_init_features: 64,
            bn_size: 4,
            compression: 0.5,
        }
    }
}

impl Default for NlpModelConfig {
    fn default() -> Self {
        Self {
            architecture: NlpArchitecture::BERT,
            vocab_size: 30522,
            max_length: 512,
            arch_params: NlpArchParams::BERT(BERTConfig::default()),
            training: TrainingConfig::default(),
        }
    }
}

impl Default for BERTConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            layer_norm_eps: 1e-12,
            use_absolute_pos: true,
        }
    }
}

impl Default for GPTConfig {
    fn default() -> Self {
        Self {
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            n_ctx: 1024,
            resid_pdrop: 0.1,
            embd_pdrop: 0.1,
            attn_pdrop: 0.1,
            use_bias: true,
        }
    }
}

impl Default for T5Config {
    fn default() -> Self {
        Self {
            d_model: 512,
            d_kv: 64,
            d_ff: 2048,
            num_layers: 6,
            num_heads: 8,
            relative_attention_num_buckets: 32,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
        }
    }
}

impl Default for RoBERTaConfig {
    fn default() -> Self {
        Self {
            bert_config: BERTConfig::default(),
            use_alternate_layernorm: false,
        }
    }
}

impl Default for BARTConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50265,
            d_model: 768,
            encoder_layers: 6,
            decoder_layers: 6,
            encoder_attention_heads: 12,
            decoder_attention_heads: 12,
            encoder_ffn_dim: 3072,
            decoder_ffn_dim: 3072,
            dropout: 0.1,
            attention_dropout: 0.0,
            activation_function: "gelu".to_string(),
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            batch_size: 32,
            epochs: 10,
            weight_decay: 0.01,
            optimizer: OptimizerType::AdamW {
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
            },
            lr_scheduler: LRSchedulerType::Constant,
            max_grad_norm: Some(1.0),
            warmup_steps: None,
        }
    }
}

impl ModelConfig for VisionModelConfig {
    fn model_name(&self) -> String {
        format!("{:?}", self.architecture)
    }

    fn variant(&self) -> String {
        match &self.arch_params {
            VisionArchParams::ResNet(config) => {
                let total_layers: usize = config.layers.iter().sum::<usize>() * 2 + 2;
                format!("resnet{}", total_layers)
            }
            VisionArchParams::EfficientNet(config) => {
                if config.width_mult == 1.0 && config.depth_mult == 1.0 {
                    "efficientnet_b0".to_string()
                } else {
                    format!(
                        "efficientnet_w{:.1}_d{:.1}",
                        config.width_mult, config.depth_mult
                    )
                }
            }
            VisionArchParams::VisionTransformer(config) => {
                if config.embed_dim == 768 {
                    format!("vit_base_patch{}", config.patch_size)
                } else if config.embed_dim == 1024 {
                    format!("vit_large_patch{}", config.patch_size)
                } else {
                    format!("vit_{}d_patch{}", config.embed_dim, config.patch_size)
                }
            }
            VisionArchParams::MobileNet(config) => {
                format!("mobilenet_{:?}_w{:.1}", config.version, config.width_mult)
            }
            VisionArchParams::DenseNet(config) => {
                let total_layers: usize = config.block_config.iter().sum::<usize>() * 2 + 1;
                format!("densenet{}", total_layers)
            }
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.input_size.0 == 0 || self.input_size.1 == 0 {
            return Err("Input size must be positive".to_string());
        }
        if self.in_channels == 0 {
            return Err("Input channels must be positive".to_string());
        }
        if self.num_classes == 0 {
            return Err("Number of classes must be positive".to_string());
        }

        match &self.arch_params {
            VisionArchParams::VisionTransformer(config) => {
                if self.input_size.0 % config.patch_size != 0
                    || self.input_size.1 % config.patch_size != 0
                {
                    return Err("Image size must be divisible by patch size".to_string());
                }
                if config.embed_dim % config.num_heads != 0 {
                    return Err(
                        "Embedding dimension must be divisible by number of heads".to_string()
                    );
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn estimated_parameters(&self) -> u64 {
        match &self.arch_params {
            VisionArchParams::ResNet(config) => {
                let base_params = if config.bottleneck {
                    25_000_000
                } else {
                    11_000_000
                };
                (base_params as f32 * config.width_mult * config.width_mult) as u64
            }
            VisionArchParams::EfficientNet(config) => {
                let base_params = 5_300_000u64;
                (base_params as f32 * config.width_mult * config.width_mult * config.depth_mult)
                    as u64
            }
            VisionArchParams::VisionTransformer(config) => {
                let patch_embed_params =
                    self.in_channels * config.embed_dim * config.patch_size * config.patch_size;
                let transformer_params = config.depth
                    * (4 * config.embed_dim * config.embed_dim
                        + 4 * config.embed_dim * (config.embed_dim * config.mlp_ratio as usize));
                let head_params = config.embed_dim * self.num_classes;
                (patch_embed_params + transformer_params + head_params) as u64
            }
            VisionArchParams::MobileNet(_) => 3_500_000,
            VisionArchParams::DenseNet(config) => {
                let base_params = config.growth_rate as u64
                    * config.block_config.iter().sum::<usize>() as u64
                    * 1000;
                base_params
            }
        }
    }
}

impl ModelConfig for NlpModelConfig {
    fn model_name(&self) -> String {
        format!("{:?}", self.architecture)
    }

    fn variant(&self) -> String {
        match &self.arch_params {
            NlpArchParams::BERT(config) => {
                if config.hidden_size == 768 {
                    "bert_base".to_string()
                } else if config.hidden_size == 1024 {
                    "bert_large".to_string()
                } else {
                    format!("bert_{}h_{}l", config.hidden_size, config.num_hidden_layers)
                }
            }
            NlpArchParams::GPT(config) => {
                if config.n_embd == 768 {
                    "gpt_base".to_string()
                } else {
                    format!("gpt_{}d_{}l", config.n_embd, config.n_layer)
                }
            }
            NlpArchParams::T5(config) => {
                if config.d_model == 512 {
                    "t5_small".to_string()
                } else if config.d_model == 768 {
                    "t5_base".to_string()
                } else {
                    format!("t5_{}d_{}l", config.d_model, config.num_layers)
                }
            }
            NlpArchParams::RoBERTa(config) => {
                if config.bert_config.hidden_size == 768 {
                    "roberta_base".to_string()
                } else {
                    "roberta_large".to_string()
                }
            }
            NlpArchParams::BART(config) => {
                if config.d_model == 768 {
                    "bart_base".to_string()
                } else {
                    format!("bart_{}d", config.d_model)
                }
            }
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0 {
            return Err("Vocabulary size must be positive".to_string());
        }
        if self.max_length == 0 {
            return Err("Maximum length must be positive".to_string());
        }

        match &self.arch_params {
            NlpArchParams::BERT(config) => {
                if config.hidden_size % config.num_attention_heads != 0 {
                    return Err(
                        "Hidden size must be divisible by number of attention heads".to_string()
                    );
                }
            }
            NlpArchParams::GPT(config) => {
                if config.n_embd % config.n_head != 0 {
                    return Err("Embedding size must be divisible by number of heads".to_string());
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn estimated_parameters(&self) -> u64 {
        match &self.arch_params {
            NlpArchParams::BERT(config) => {
                let embedding_params =
                    self.vocab_size * config.hidden_size + self.max_length * config.hidden_size;
                let transformer_params = config.num_hidden_layers
                    * (4 * config.hidden_size * config.hidden_size
                        + config.intermediate_size * config.hidden_size);
                (embedding_params + transformer_params) as u64
            }
            NlpArchParams::GPT(config) => {
                let embedding_params =
                    self.vocab_size * config.n_embd + config.n_ctx * config.n_embd;
                let transformer_params = config.n_layer * (4 * config.n_embd * config.n_embd);
                (embedding_params + transformer_params) as u64
            }
            NlpArchParams::T5(config) => {
                let embedding_params = self.vocab_size * config.d_model;
                let encoder_params = config.num_layers
                    * (4 * config.d_model * config.d_model + config.d_ff * config.d_model);
                let decoder_params = config.num_layers
                    * (4 * config.d_model * config.d_model + config.d_ff * config.d_model);
                (embedding_params + encoder_params + decoder_params) as u64
            }
            NlpArchParams::RoBERTa(config) => {
                let embedding_params = self.vocab_size * config.bert_config.hidden_size
                    + self.max_length * config.bert_config.hidden_size;
                let transformer_params = config.bert_config.num_hidden_layers
                    * (4 * config.bert_config.hidden_size * config.bert_config.hidden_size
                        + config.bert_config.intermediate_size * config.bert_config.hidden_size);
                (embedding_params + transformer_params) as u64
            }
            NlpArchParams::BART(config) => {
                let embedding_params = config.vocab_size * config.d_model;
                let encoder_params = config.encoder_layers
                    * (4 * config.d_model * config.d_model
                        + config.encoder_ffn_dim * config.d_model);
                let decoder_params = config.decoder_layers
                    * (4 * config.d_model * config.d_model
                        + config.decoder_ffn_dim * config.d_model);
                (embedding_params + encoder_params + decoder_params) as u64
            }
        }
    }
}

/// Predefined model configurations
pub struct ModelConfigs;

impl ModelConfigs {
    /// Get ResNet configurations
    pub fn resnet_configs() -> HashMap<String, VisionModelConfig> {
        let mut configs = HashMap::new();

        // ResNet-18
        configs.insert(
            "resnet18".to_string(),
            VisionModelConfig {
                architecture: VisionArchitecture::ResNet,
                arch_params: VisionArchParams::ResNet(ResNetConfig {
                    layers: [2, 2, 2, 2],
                    bottleneck: false,
                    width_mult: 1.0,
                    stem_type: StemType::Standard,
                }),
                ..Default::default()
            },
        );

        // ResNet-50
        configs.insert(
            "resnet50".to_string(),
            VisionModelConfig {
                architecture: VisionArchitecture::ResNet,
                arch_params: VisionArchParams::ResNet(ResNetConfig {
                    layers: [3, 4, 6, 3],
                    bottleneck: true,
                    width_mult: 1.0,
                    stem_type: StemType::Standard,
                }),
                ..Default::default()
            },
        );

        configs
    }

    /// Get EfficientNet configurations
    pub fn efficientnet_configs() -> HashMap<String, VisionModelConfig> {
        let mut configs = HashMap::new();

        let variants = [
            ("efficientnet_b0", 1.0, 1.0, 1.0, 0.2),
            ("efficientnet_b1", 1.0, 1.1, 1.15, 0.2),
            ("efficientnet_b2", 1.1, 1.2, 1.3, 0.3),
            ("efficientnet_b3", 1.2, 1.4, 1.5, 0.3),
            ("efficientnet_b4", 1.4, 1.8, 1.8, 0.4),
        ];

        for (name, width, depth, res, dropout) in variants {
            configs.insert(
                name.to_string(),
                VisionModelConfig {
                    architecture: VisionArchitecture::EfficientNet,
                    input_size: ((224.0 * res) as usize, (224.0 * res) as usize),
                    arch_params: VisionArchParams::EfficientNet(EfficientNetConfig {
                        width_mult: width,
                        depth_mult: depth,
                        resolution_mult: res,
                        dropout_rate: dropout,
                        se_ratio: 0.25,
                    }),
                    ..Default::default()
                },
            );
        }

        configs
    }

    /// Get Vision Transformer configurations
    pub fn vit_configs() -> HashMap<String, VisionModelConfig> {
        let mut configs = HashMap::new();

        // ViT-Base/16
        configs.insert(
            "vit_base_patch16_224".to_string(),
            VisionModelConfig {
                architecture: VisionArchitecture::VisionTransformer,
                arch_params: VisionArchParams::VisionTransformer(ViTConfig {
                    patch_size: 16,
                    embed_dim: 768,
                    depth: 12,
                    num_heads: 12,
                    mlp_ratio: 4.0,
                    dropout_rate: 0.1,
                    attn_dropout_rate: 0.0,
                    pos_embed_type: PositionEmbedType::Learned,
                }),
                ..Default::default()
            },
        );

        // ViT-Large/16
        configs.insert(
            "vit_large_patch16_224".to_string(),
            VisionModelConfig {
                architecture: VisionArchitecture::VisionTransformer,
                arch_params: VisionArchParams::VisionTransformer(ViTConfig {
                    patch_size: 16,
                    embed_dim: 1024,
                    depth: 24,
                    num_heads: 16,
                    mlp_ratio: 4.0,
                    dropout_rate: 0.1,
                    attn_dropout_rate: 0.0,
                    pos_embed_type: PositionEmbedType::Learned,
                }),
                ..Default::default()
            },
        );

        configs
    }

    /// Get BERT configurations
    pub fn bert_configs() -> HashMap<String, NlpModelConfig> {
        let mut configs = HashMap::new();

        // BERT-Base
        configs.insert(
            "bert_base_uncased".to_string(),
            NlpModelConfig {
                architecture: NlpArchitecture::BERT,
                vocab_size: 30522,
                max_length: 512,
                arch_params: NlpArchParams::BERT(BERTConfig {
                    hidden_size: 768,
                    num_hidden_layers: 12,
                    num_attention_heads: 12,
                    intermediate_size: 3072,
                    hidden_act: "gelu".to_string(),
                    hidden_dropout_prob: 0.1,
                    attention_probs_dropout_prob: 0.1,
                    layer_norm_eps: 1e-12,
                    use_absolute_pos: true,
                }),
                ..Default::default()
            },
        );

        // BERT-Large
        configs.insert(
            "bert_large_uncased".to_string(),
            NlpModelConfig {
                architecture: NlpArchitecture::BERT,
                vocab_size: 30522,
                max_length: 512,
                arch_params: NlpArchParams::BERT(BERTConfig {
                    hidden_size: 1024,
                    num_hidden_layers: 24,
                    num_attention_heads: 16,
                    intermediate_size: 4096,
                    hidden_act: "gelu".to_string(),
                    hidden_dropout_prob: 0.1,
                    attention_probs_dropout_prob: 0.1,
                    layer_norm_eps: 1e-12,
                    use_absolute_pos: true,
                }),
                ..Default::default()
            },
        );

        configs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vision_config_validation() {
        let mut config = VisionModelConfig::default();
        assert!(config.validate().is_ok());

        config.input_size = (0, 224);
        assert!(config.validate().is_err());

        config.input_size = (224, 224);
        config.num_classes = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_vit_config_validation() {
        let mut config = VisionModelConfig {
            input_size: (224, 224),
            arch_params: VisionArchParams::VisionTransformer(ViTConfig {
                patch_size: 16,
                embed_dim: 768,
                num_heads: 12,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(config.validate().is_ok());

        // Test invalid patch size
        config.input_size = (225, 225);
        assert!(config.validate().is_err());

        // Test invalid head count
        config.input_size = (224, 224);
        if let VisionArchParams::VisionTransformer(ref mut vit_config) = config.arch_params {
            vit_config.embed_dim = 770; // Not divisible by 12 heads
        }
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_model_variants() {
        let resnet_config = VisionModelConfig {
            arch_params: VisionArchParams::ResNet(ResNetConfig {
                layers: [2, 2, 2, 2],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(resnet_config.variant(), "resnet18");

        let vit_config = VisionModelConfig {
            arch_params: VisionArchParams::VisionTransformer(ViTConfig {
                embed_dim: 768,
                patch_size: 16,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(vit_config.variant(), "vit_base_patch16");
    }

    #[test]
    fn test_predefined_configs() {
        let resnet_configs = ModelConfigs::resnet_configs();
        assert!(resnet_configs.contains_key("resnet18"));
        assert!(resnet_configs.contains_key("resnet50"));

        let efficientnet_configs = ModelConfigs::efficientnet_configs();
        assert!(efficientnet_configs.contains_key("efficientnet_b0"));

        let vit_configs = ModelConfigs::vit_configs();
        assert!(vit_configs.contains_key("vit_base_patch16_224"));

        let bert_configs = ModelConfigs::bert_configs();
        assert!(bert_configs.contains_key("bert_base_uncased"));
    }

    #[test]
    fn test_parameter_estimation() {
        let resnet_configs = ModelConfigs::resnet_configs();
        let resnet18_config = resnet_configs.get("resnet18").unwrap();
        let params = resnet18_config.estimated_parameters();
        assert!(params > 10_000_000 && params < 15_000_000); // ResNet-18 has ~11M params

        let bert_configs = ModelConfigs::bert_configs();
        let bert_base_config = bert_configs.get("bert_base_uncased").unwrap();
        let bert_params = bert_base_config.estimated_parameters();
        assert!(bert_params > 50_000_000 && bert_params < 200_000_000); // BERT-base has ~110M params (allow wider range for estimation)
    }
}
