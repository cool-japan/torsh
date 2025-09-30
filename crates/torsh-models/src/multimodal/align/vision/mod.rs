//! ALIGN Vision Components
//!
//! EfficientNet-based vision encoder with MBConv blocks, Squeeze-and-Excitation attention,
//! and mobile-optimized convolution architecture for ALIGN model.

pub mod encoder;
pub mod mbconv;

// Re-export key components
pub use encoder::ALIGNVisionEncoder;
pub use mbconv::{GlobalAveragePooling2d, MBConvBlock, SqueezeExcitation};
