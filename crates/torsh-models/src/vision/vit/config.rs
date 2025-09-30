//! Vision Transformer (ViT) configuration

use super::super::common::types::ModelInitConfig;

/// Vision Transformer variants
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViTVariant {
    /// ViT-Tiny (5.7M parameters)
    Tiny,
    /// ViT-Small (22M parameters)
    Small,
    /// ViT-Base (86M parameters)
    Base,
    /// ViT-Large (307M parameters)
    Large,
    /// ViT-Huge (632M parameters)
    Huge,
}

impl ViTVariant {
    /// Get the embed dimension for each variant
    pub fn embed_dim(&self) -> usize {
        match self {
            ViTVariant::Tiny => 192,
            ViTVariant::Small => 384,
            ViTVariant::Base => 768,
            ViTVariant::Large => 1024,
            ViTVariant::Huge => 1280,
        }
    }

    /// Get the number of transformer layers
    pub fn depth(&self) -> usize {
        match self {
            ViTVariant::Tiny => 12,
            ViTVariant::Small => 12,
            ViTVariant::Base => 12,
            ViTVariant::Large => 24,
            ViTVariant::Huge => 32,
        }
    }

    /// Get the number of attention heads
    pub fn num_heads(&self) -> usize {
        match self {
            ViTVariant::Tiny => 3,
            ViTVariant::Small => 6,
            ViTVariant::Base => 12,
            ViTVariant::Large => 16,
            ViTVariant::Huge => 16,
        }
    }

    /// Get the MLP ratio (hidden_dim = embed_dim * mlp_ratio)
    pub fn mlp_ratio(&self) -> f32 {
        4.0 // Standard ratio for all variants
    }

    /// Get the variant name as string
    pub fn name(&self) -> &'static str {
        match self {
            ViTVariant::Tiny => "vit_tiny",
            ViTVariant::Small => "vit_small",
            ViTVariant::Base => "vit_base",
            ViTVariant::Large => "vit_large",
            ViTVariant::Huge => "vit_huge",
        }
    }

    /// Get approximate parameter count
    pub fn parameter_count(&self) -> u64 {
        match self {
            ViTVariant::Tiny => 5_700_000,
            ViTVariant::Small => 22_000_000,
            ViTVariant::Base => 86_600_000,
            ViTVariant::Large => 307_000_000,
            ViTVariant::Huge => 632_000_000,
        }
    }

    /// Get typical ImageNet accuracy
    pub fn imagenet_accuracy(&self) -> f32 {
        match self {
            ViTVariant::Tiny => 75.5,
            ViTVariant::Small => 81.1,
            ViTVariant::Base => 81.8,
            ViTVariant::Large => 82.6,
            ViTVariant::Huge => 83.1,
        }
    }
}

/// Vision Transformer configuration
#[derive(Debug, Clone)]
pub struct ViTConfig {
    /// Model variant
    pub variant: ViTVariant,
    /// Input image size
    pub img_size: usize,
    /// Patch size
    pub patch_size: usize,
    /// Input channels
    pub in_channels: usize,
    /// Number of output classes
    pub num_classes: usize,
    /// Embedding dimension
    pub embed_dim: usize,
    /// Number of transformer layers
    pub depth: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// MLP ratio (hidden_dim = embed_dim * mlp_ratio)
    pub mlp_ratio: f32,
    /// QKV bias in attention
    pub qkv_bias: bool,
    /// Representation size (for pre-training)
    pub representation_size: Option<usize>,
    /// Attention dropout rate
    pub attn_dropout: f32,
    /// Projection dropout rate
    pub proj_dropout: f32,
    /// Path dropout rate (stochastic depth)
    pub path_dropout: f32,
    /// Norm layer epsilon
    pub norm_eps: f32,
    /// Whether to use global average pooling instead of class token
    pub global_pool: bool,
    /// Patch embedding strategy
    pub patch_embed_strategy: PatchEmbedStrategy,
}

/// Patch embedding strategies
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatchEmbedStrategy {
    /// Standard convolution with patch_size stride
    Convolution,
    /// Linear projection after reshaping
    Linear,
    /// Hybrid approach with initial convolution layers
    Hybrid,
}

impl Default for ViTConfig {
    fn default() -> Self {
        Self::vit_base_patch16_224()
    }
}

impl ViTConfig {
    /// Create ViT-Base/16 configuration for 224x224 images
    pub fn vit_base_patch16_224() -> Self {
        let variant = ViTVariant::Base;
        Self {
            variant: variant.clone(),
            img_size: 224,
            patch_size: 16,
            in_channels: 3,
            num_classes: 1000,
            embed_dim: variant.embed_dim(),
            depth: variant.depth(),
            num_heads: variant.num_heads(),
            mlp_ratio: variant.mlp_ratio(),
            qkv_bias: true,
            representation_size: None,
            attn_dropout: 0.0,
            proj_dropout: 0.0,
            path_dropout: 0.0,
            norm_eps: 1e-6,
            global_pool: false,
            patch_embed_strategy: PatchEmbedStrategy::Convolution,
        }
    }

    /// Create ViT-Large/16 configuration
    pub fn vit_large_patch16_224() -> Self {
        let variant = ViTVariant::Large;
        Self {
            variant: variant.clone(),
            embed_dim: variant.embed_dim(),
            depth: variant.depth(),
            num_heads: variant.num_heads(),
            ..Self::vit_base_patch16_224()
        }
    }

    /// Create ViT-Small/16 configuration
    pub fn vit_small_patch16_224() -> Self {
        let variant = ViTVariant::Small;
        Self {
            variant: variant.clone(),
            embed_dim: variant.embed_dim(),
            depth: variant.depth(),
            num_heads: variant.num_heads(),
            ..Self::vit_base_patch16_224()
        }
    }

    /// Create ViT-Tiny/16 configuration
    pub fn vit_tiny_patch16_224() -> Self {
        let variant = ViTVariant::Tiny;
        Self {
            variant: variant.clone(),
            embed_dim: variant.embed_dim(),
            depth: variant.depth(),
            num_heads: variant.num_heads(),
            ..Self::vit_base_patch16_224()
        }
    }

    /// Create ViT-Base/32 configuration
    pub fn vit_base_patch32_224() -> Self {
        Self {
            patch_size: 32,
            ..Self::vit_base_patch16_224()
        }
    }

    /// Create configuration for custom image size
    pub fn with_image_size(mut self, img_size: usize) -> Self {
        self.img_size = img_size;
        self
    }

    /// Create configuration for custom number of classes
    pub fn with_num_classes(mut self, num_classes: usize) -> Self {
        self.num_classes = num_classes;
        self
    }

    /// Create configuration with dropout
    pub fn with_dropout(mut self, attn_dropout: f32, proj_dropout: f32, path_dropout: f32) -> Self {
        self.attn_dropout = attn_dropout;
        self.proj_dropout = proj_dropout;
        self.path_dropout = path_dropout;
        self
    }

    /// Create configuration from ModelInitConfig
    pub fn from_init_config(variant: ViTVariant, init_config: ModelInitConfig) -> Self {
        let mut config = match variant {
            ViTVariant::Tiny => Self::vit_tiny_patch16_224(),
            ViTVariant::Small => Self::vit_small_patch16_224(),
            ViTVariant::Base => Self::vit_base_patch16_224(),
            ViTVariant::Large => Self::vit_large_patch16_224(),
            ViTVariant::Huge => {
                let variant = ViTVariant::Huge;
                Self {
                    variant: variant.clone(),
                    embed_dim: variant.embed_dim(),
                    depth: variant.depth(),
                    num_heads: variant.num_heads(),
                    ..Self::vit_base_patch16_224()
                }
            }
        };

        config.num_classes = init_config.num_classes;
        if init_config.dropout > 0.0 {
            config = config.with_dropout(
                init_config.dropout,
                init_config.dropout,
                init_config.dropout * 0.1,
            );
        }

        config
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.img_size == 0 {
            return Err("Image size must be greater than 0".to_string());
        }

        if self.patch_size == 0 {
            return Err("Patch size must be greater than 0".to_string());
        }

        if self.img_size % self.patch_size != 0 {
            return Err(format!(
                "Image size ({}) must be divisible by patch size ({})",
                self.img_size, self.patch_size
            ));
        }

        if self.in_channels == 0 {
            return Err("Input channels must be greater than 0".to_string());
        }

        if self.num_classes == 0 {
            return Err("Number of classes must be greater than 0".to_string());
        }

        if self.embed_dim == 0 {
            return Err("Embedding dimension must be greater than 0".to_string());
        }

        if self.embed_dim % self.num_heads != 0 {
            return Err(format!(
                "Embedding dimension ({}) must be divisible by number of heads ({})",
                self.embed_dim, self.num_heads
            ));
        }

        if self.depth == 0 {
            return Err("Depth must be greater than 0".to_string());
        }

        if self.num_heads == 0 {
            return Err("Number of heads must be greater than 0".to_string());
        }

        if self.mlp_ratio <= 0.0 {
            return Err("MLP ratio must be greater than 0".to_string());
        }

        if self.attn_dropout < 0.0 || self.attn_dropout >= 1.0 {
            return Err("Attention dropout must be in range [0.0, 1.0)".to_string());
        }

        if self.proj_dropout < 0.0 || self.proj_dropout >= 1.0 {
            return Err("Projection dropout must be in range [0.0, 1.0)".to_string());
        }

        if self.path_dropout < 0.0 || self.path_dropout >= 1.0 {
            return Err("Path dropout must be in range [0.0, 1.0)".to_string());
        }

        Ok(())
    }

    /// Get the number of patches
    pub fn num_patches(&self) -> usize {
        (self.img_size / self.patch_size).pow(2)
    }

    /// Get the sequence length (number of patches + class token)
    pub fn seq_length(&self) -> usize {
        if self.global_pool {
            self.num_patches()
        } else {
            self.num_patches() + 1
        }
    }

    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }

    /// Get the MLP hidden dimension
    pub fn mlp_hidden_dim(&self) -> usize {
        (self.embed_dim as f32 * self.mlp_ratio) as usize
    }

    /// Get the expected input shape
    pub fn input_shape(&self) -> (usize, usize, usize) {
        (self.in_channels, self.img_size, self.img_size)
    }
}
