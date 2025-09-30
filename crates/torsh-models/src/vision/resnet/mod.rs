//! ResNet (Residual Networks) model family
//!
//! This module provides implementations of ResNet architectures for image classification.
//!
//! # Model Variants
//! - ResNet-18: 18-layer network with basic blocks
//! - ResNet-34: 34-layer network with basic blocks
//! - ResNet-50: 50-layer network with bottleneck blocks
//! - ResNet-101: 101-layer network with bottleneck blocks
//! - ResNet-152: 152-layer network with bottleneck blocks
//!
//! # Extensions
//! - SE-ResNet: ResNet with Squeeze-and-Excitation blocks
//! - ResNeXt: ResNet with grouped convolutions
//! - Wide ResNet: ResNet with increased width
//!
//! # Example Usage
//! ```rust,no_run
//! use torsh_models::vision::resnet::{ResNet, ResNetBuilder, ResNetVariant};
//!
//! // Create ResNet-50 for ImageNet classification
//! let model = ResNet::resnet50(1000)?;
//!
//! // Create ResNet-18 for CIFAR-10 classification
//! let model = ResNet::resnet18(10)?;
//!
//! // Create SE-ResNet-50 with custom configuration
//! let model = ResNetBuilder::new(ResNetVariant::ResNet50)
//!     .num_classes(100)
//!     .with_se(16)
//!     .dropout(0.1)
//!     .build()?;
//!
//! // Create ResNeXt-50 32x4d
//! let model = ResNetBuilder::new(ResNetVariant::ResNet50)
//!     .num_classes(1000)
//!     .resnext(32, 4)
//!     .build()?;
//! ```

pub mod blocks;
pub mod config;
pub mod models;

// Re-export main types for easy access
pub use blocks::{BasicBlock, BottleneckBlock, SEBlock};
pub use config::{ResNetConfig, ResNetVariant};
pub use models::{ResNet, ResNetBuilder};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resnet_variants() {
        assert_eq!(ResNetVariant::ResNet18.name(), "resnet18");
        assert_eq!(ResNetVariant::ResNet50.name(), "resnet50");
        assert!(!ResNetVariant::ResNet18.uses_bottleneck());
        assert!(ResNetVariant::ResNet50.uses_bottleneck());
    }

    #[test]
    fn test_resnet_layer_configs() {
        assert_eq!(ResNetVariant::ResNet18.layer_config(), &[2, 2, 2, 2]);
        assert_eq!(ResNetVariant::ResNet50.layer_config(), &[3, 4, 6, 3]);
        assert_eq!(ResNetVariant::ResNet152.layer_config(), &[3, 8, 36, 3]);
    }

    #[test]
    fn test_config_creation() {
        let config = ResNetConfig::resnet18(10);
        assert_eq!(config.variant, ResNetVariant::ResNet18);
        assert_eq!(config.num_classes, 10);

        let config = ResNetConfig::se_resnet(ResNetVariant::ResNet50, 1000);
        assert!(config.use_se);
        assert_eq!(config.se_reduction_ratio, 16);
    }

    #[test]
    fn test_resnext_config() {
        let config = ResNetConfig::resnext50_32x4d(1000);
        assert_eq!(config.groups, 32);
        assert_eq!(config.width_per_group, 4);
    }
}
