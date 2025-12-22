//! Vision Transformer (ViT) model family
//!
//! This module provides implementations of Vision Transformer architectures.
//!
//! # Model Variants
//! - ViT-Tiny: 5.7M parameters, 192 embed dim
//! - ViT-Small: 22M parameters, 384 embed dim
//! - ViT-Base: 86M parameters, 768 embed dim
//! - ViT-Large: 307M parameters, 1024 embed dim
//! - ViT-Huge: 632M parameters, 1280 embed dim
//!
//! # Example Usage
//! ```rust,no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use torsh_models::vision::vit::{ViTConfig, ViTVariant};
//! use torsh_models::vision::VisionTransformer;
//!
//! // Create ViT-Base/16 for ImageNet
//! let config = ViTConfig::vit_base_patch16_224();
//! let model = VisionTransformer::new(config)?;
//!
//! // Create ViT-Small/16 for CIFAR-100
//! let config = ViTConfig::vit_small_patch16_224().with_num_classes(100);
//! let model = VisionTransformer::new(config)?;
//! # Ok(())
//! # }
//! ```

pub mod config;
pub mod embeddings;

// Re-export main types
pub use config::{PatchEmbedStrategy, ViTConfig, ViTVariant};
pub use embeddings::{ClassToken, PatchEmbedding, PositionalEmbedding};

// Placeholder for full VisionTransformer implementation
// This would be implemented in models.rs in the complete refactoring

/// Placeholder Vision Transformer model
///
/// In the complete refactoring, this would be a full implementation
/// with transformer encoder blocks, attention mechanisms, etc.
pub struct VisionTransformer {
    config: ViTConfig,
}

impl VisionTransformer {
    pub fn new(config: ViTConfig) -> torsh_core::error::Result<Self> {
        config
            .validate()
            .map_err(|e| torsh_core::error::TorshError::Other(e))?;
        Ok(Self { config })
    }

    pub fn config(&self) -> &ViTConfig {
        &self.config
    }
}

// Note: Full Module implementation would be in models.rs
// This is a demonstration of the modular structure

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vit_variants() {
        assert_eq!(ViTVariant::Base.embed_dim(), 768);
        assert_eq!(ViTVariant::Large.depth(), 24);
        assert_eq!(ViTVariant::Small.num_heads(), 6);
    }

    #[test]
    fn test_vit_configs() {
        let config = ViTConfig::vit_base_patch16_224();
        assert_eq!(config.img_size, 224);
        assert_eq!(config.patch_size, 16);
        assert_eq!(config.num_patches(), 196);

        let config = ViTConfig::vit_large_patch16_224();
        assert_eq!(config.embed_dim, 1024);
        assert_eq!(config.depth, 24);
    }

    #[test]
    fn test_config_validation() {
        let mut config = ViTConfig::vit_base_patch16_224();
        assert!(config.validate().is_ok());

        config.img_size = 225; // Not divisible by patch_size
        assert!(config.validate().is_err());
    }
}
