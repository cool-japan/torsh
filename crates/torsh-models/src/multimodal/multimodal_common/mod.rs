//! Common components for multimodal models

pub mod activations;
pub mod types;
pub mod utils;

// Re-export commonly used items
pub use activations::{create_activation, QuickGELU, SiLU};
pub use types::{
    AttentionPoolingConfig, CrossModalProjectionConfig, MultimodalArchitecture, MultimodalTask,
    TextEncoderConfig, VisionEncoderConfig,
};
pub use utils::{
    contrastive_loss, create_sinusoidal_position_embeddings, CrossModalProjection,
    GlobalAveragePooling2d, SqueezeExcitation,
};
