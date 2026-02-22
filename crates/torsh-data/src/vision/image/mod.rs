//! Image processing components for computer vision
//!
//! This module provides datasets and transformations for working with images in computer vision tasks.

pub mod datasets;
pub mod transforms;

// Re-export core image types
pub use datasets::ImageFolder;
pub use transforms::{
    transforms::{CenterCrop, Normalize, Resize},
    Compose, ImageToTensor, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip,
    TensorToImage,
};
