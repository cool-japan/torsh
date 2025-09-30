//! Computer vision models
//!
//! This module contains implementations of popular computer vision models
//! including ResNet, VGG, EfficientNet, Vision Transformer, and object detection models.

/// Configuration for vision models
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub num_classes: usize,
    pub dropout: f32,
    pub pretrained: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            num_classes: 1000,
            dropout: 0.0,
            pretrained: false,
        }
    }
}

impl ModelConfig {
    pub fn new(num_classes: usize) -> Self {
        Self {
            num_classes,
            dropout: 0.0,
            pretrained: false,
        }
    }

    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn with_pretrained(mut self, pretrained: bool) -> Self {
        self.pretrained = pretrained;
        self
    }
}

/// Common trait for all vision models
pub trait VisionModel {
    /// Get the number of classes this model predicts
    fn num_classes(&self) -> usize;

    /// Get the expected input size (height, width)
    fn input_size(&self) -> (usize, usize);

    /// Get the model name
    fn name(&self) -> &str;
}

pub mod advanced_architectures;
pub mod advanced_cnns;
pub mod alexnet;
pub mod densenet;
pub mod detection;
pub mod efficientnet;
pub mod mobilenet;
pub mod registry;
pub mod resnet;
pub mod style_transfer;
pub mod super_resolution;
pub mod vgg;
pub mod vision_transformer;

// Re-export common types for convenience
pub use advanced_architectures::*;
pub use advanced_cnns::*;
pub use alexnet::*;
pub use densenet::*;
pub use detection::*;
pub use efficientnet::*;
pub use mobilenet::*;
pub use registry::*;
pub use resnet::*;
pub use style_transfer::*;
pub use super_resolution::*;
pub use vgg::*;
pub use vision_transformer::*;
