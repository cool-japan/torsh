//! Vision models for ToRSh deep learning framework
//!
//! This module provides computer vision models including CNNs and Transformers:
//! - **ResNet** family for residual learning
//! - **Vision Transformer (ViT)** family for transformer-based vision
//! - **EfficientNet** family for efficient architectures (placeholder)
//! - **Swin Transformer** family for hierarchical vision transformers (placeholder)
//! - **ConvNeXt** family for modern ConvNets (placeholder)
//! - **DETR** for object detection (placeholder)
//! - **MaskRCNN** for instance segmentation (placeholder)
//! - **YOLO** for real-time object detection (placeholder)
//! - **MobileNet** family for mobile-optimized models (placeholder)
//! - **DenseNet** family for dense connections (placeholder)
//!
//! # Example Usage
//!
//! ```rust,no_run
//! use torsh_models::vision::{ResNet, ViTConfig, VisionTransformer, VisionArchitecture};
//! use torsh_tensor::creation;
//!
//! // Create ResNet-50 for ImageNet classification
//! let resnet = ResNet::resnet50(1000)?;
//!
//! // Create Vision Transformer for custom task
//! let vit_config = ViTConfig::vit_base_patch16_224().with_num_classes(100);
//! let vit = VisionTransformer::new(vit_config)?;
//!
//! // Process images
//! let images = creation::randn(&[4, 3, 224, 224])?;
//! let resnet_output = resnet.forward(&images)?;
//! ```
//!
//! # Architecture Overview
//!
//! The vision module is organized into model families:
//!
//! - `common/` - Shared components and utilities
//! - `resnet/` - ResNet model family (complete implementation)
//! - `vit/` - Vision Transformer family (partial implementation)
//! - `efficientnet/` - EfficientNet family (placeholder)
//! - `swin/` - Swin Transformer family (placeholder)
//! - `convnext/` - ConvNeXt family (placeholder)
//! - `detr/` - DETR family (placeholder)
//! - `maskrcnn/` - Mask R-CNN family (placeholder)
//! - `yolo/` - YOLO family (placeholder)
//! - `mobilenet/` - MobileNet family (placeholder)
//! - `densenet/` - DenseNet family (placeholder)

// Common components
pub mod common;

// Model families - Complete implementations
pub mod resnet;
pub mod vision_transformer;
pub mod vit;

// Model families - Placeholder modules for systematic refactoring
pub mod efficientnet {
    //! EfficientNet model family (placeholder)
    //!
    //! This module will contain the EfficientNet model implementations
    //! once migrated from the monolithic vision.rs file.
    //!
    //! Key components to migrate:
    //! - EfficientNet config and variants
    //! - MBConv blocks with Squeeze-and-Excitation
    //! - Depthwise separable convolutions
    //! - Compound scaling strategy
}

pub mod swin {
    //! Swin Transformer model family (placeholder)
    //!
    //! This module will contain the Swin Transformer implementations
    //! including hierarchical vision transformers with shifted windows.
}

pub mod convnext {
    //! ConvNeXt model family (placeholder)
    //!
    //! This module will contain modern ConvNet architectures
    //! that compete with vision transformers.
}

pub mod detr {
    //! DETR (Detection Transformer) model family (placeholder)
    //!
    //! This module will contain transformer-based object detection models.
}

pub mod maskrcnn {
    //! Mask R-CNN model family (placeholder)
    //!
    //! This module will contain instance segmentation models.
}

pub mod yolo {
    //! YOLO model family (placeholder)
    //!
    //! This module will contain real-time object detection models.
}

pub mod mobilenet {
    //! MobileNet model family (placeholder)
    //!
    //! This module will contain mobile-optimized architectures.
}

pub mod densenet {
    //! DenseNet model family (placeholder)
    //!
    //! This module will contain densely connected networks.
}

// Re-export common types for backward compatibility
pub use common::preprocessing::{AugmentationConfig, ImagePreprocessor, PreprocessingPipeline};
pub use common::types::{
    ImageNormalization, ModelInitConfig, VisionActivation, VisionArchitecture, VisionModelVariant,
    VisionTask,
};
pub use common::utils::{get_common_vision_models, VisionModelUtils};

// Re-export ResNet models for easy access
pub use resnet::{
    BasicBlock, BottleneckBlock, ResNet, ResNetBuilder, ResNetConfig, ResNetVariant, SEBlock,
};

// Re-export Vision Transformer models
pub use vit::{
    ClassToken, PatchEmbedStrategy, PatchEmbedding, PositionalEmbedding, ViTConfig, ViTVariant,
};

// Re-export the complete VisionTransformer implementation
pub use vision_transformer::VisionTransformer;

// EfficientNet family - re-export from torsh-nn
pub use torsh_nn::prelude::{EfficientNet, EfficientNetConfig};

/// EfficientNet variants supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfficientNetVariant {
    B0,
    B1,
    B2,
    B3,
    B4,
    B5,
    B6,
    B7,
}

impl EfficientNetVariant {
    /// Get the string representation of the variant
    pub fn as_str(&self) -> &'static str {
        match self {
            EfficientNetVariant::B0 => "b0",
            EfficientNetVariant::B1 => "b1",
            EfficientNetVariant::B2 => "b2",
            EfficientNetVariant::B3 => "b3",
            EfficientNetVariant::B4 => "b4",
            EfficientNetVariant::B5 => "b5",
            EfficientNetVariant::B6 => "b6",
            EfficientNetVariant::B7 => "b7",
        }
    }
}

// Placeholder re-exports for other model families
// These will be uncommented as the models are migrated

/*
// Swin Transformer family
pub use swin::{
    SwinTransformer, SwinConfig,
    WindowAttention, SwinTransformerBlock,
    // ... other Swin exports
};

// ConvNeXt family
pub use convnext::{
    ConvNeXt, ConvNeXtConfig,
    ConvNeXtBlock, ConvNeXtStage,
    // ... other ConvNeXt exports
};

// And so on for other families...
*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vision_architecture_enum() {
        let arch = VisionArchitecture::ResNet;
        assert_eq!(arch.name(), "ResNet");

        let arch = VisionArchitecture::VisionTransformer;
        assert_eq!(arch.name(), "Vision Transformer");
    }

    #[test]
    fn test_vision_task_enum() {
        let task = VisionTask::ImageClassification;
        assert_eq!(task.as_str(), "image_classification");

        let task = VisionTask::ObjectDetection;
        assert_eq!(task.as_str(), "object_detection");
    }

    #[test]
    fn test_model_variant_access() {
        let variant = VisionModelUtils::get_model_variant("resnet50");
        assert!(variant.is_some());
        assert_eq!(variant.unwrap().architecture, VisionArchitecture::ResNet);
    }

    #[test]
    fn test_preprocessor_creation() {
        let preprocessor = ImagePreprocessor::imagenet();
        assert_eq!(preprocessor.output_size(), (224, 224));

        let preprocessor = ImagePreprocessor::for_architecture(VisionArchitecture::YOLO);
        assert_eq!(preprocessor.output_size(), (640, 640));
    }

    #[test]
    fn test_resnet_integration() {
        // Test that ResNet models are accessible through the re-exports
        let config = ResNetConfig::resnet18(10);
        assert_eq!(config.variant, ResNetVariant::ResNet18);
    }

    #[test]
    fn test_vit_integration() {
        // Test that ViT models are accessible through the re-exports
        let config = ViTConfig::vit_base_patch16_224();
        assert_eq!(config.variant, ViTVariant::Base);
    }
}

/// Utility function to create a vision model by architecture name
pub fn create_model_by_architecture(
    architecture: VisionArchitecture,
    num_classes: usize,
) -> Result<Box<dyn torsh_nn::Module>, torsh_core::error::TorshError> {
    match architecture {
        VisionArchitecture::ResNet => {
            let model = ResNet::resnet50(num_classes)?;
            Ok(Box::new(model))
        }
        VisionArchitecture::VisionTransformer => {
            let config = ViTConfig::vit_base_patch16_224().with_num_classes(num_classes);
            let model = VisionTransformer::new(config)?;
            Ok(Box::new(model))
        }
        _ => Err(torsh_core::error::TorshError::Other(format!(
            "Vision architecture {} not yet implemented",
            architecture.name()
        ))),
    }
}

/// Utility function to list supported vision architectures
pub fn supported_architectures() -> Vec<VisionArchitecture> {
    vec![
        VisionArchitecture::ResNet,
        VisionArchitecture::VisionTransformer,
        // Other architectures will be added as they are implemented
    ]
}

/// Utility function to check if an architecture is supported
pub fn is_architecture_supported(architecture: &VisionArchitecture) -> bool {
    matches!(
        architecture,
        VisionArchitecture::ResNet | VisionArchitecture::VisionTransformer
    )
}

/// Get recommended model for specific requirements
pub fn recommend_model(
    task: VisionTask,
    accuracy_target: f32,
    efficiency_priority: bool,
) -> Option<VisionArchitecture> {
    match task {
        VisionTask::ImageClassification => {
            if efficiency_priority {
                Some(VisionArchitecture::ResNet) // ResNet-18/34 for efficiency
            } else if accuracy_target > 82.0 {
                Some(VisionArchitecture::VisionTransformer) // ViT for high accuracy
            } else {
                Some(VisionArchitecture::ResNet) // ResNet-50/101 for balanced performance
            }
        }
        VisionTask::ObjectDetection => {
            // Would recommend DETR or YOLO when implemented
            None
        }
        VisionTask::InstanceSegmentation => {
            // Would recommend Mask R-CNN when implemented
            None
        }
        _ => None,
    }
}
