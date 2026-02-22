//! Standard vision datasets
//!
//! This module provides implementations of commonly used computer vision datasets
//! including MNIST, CIFAR-10, and ImageNet.

pub mod cifar;
pub mod imagenet;
pub mod mnist;

// Re-export dataset types
pub use cifar::CIFAR10;
pub use imagenet::ImageNet;
pub use mnist::MNIST;
