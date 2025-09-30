//! CLIP (Contrastive Language-Image Pre-training) models
//!
//! This module provides implementations of CLIP models for contrastive vision-language learning.
//!
//! # Model Variants
//! - CLIP-ViT-B/32: Vision Transformer Base with 32x32 patches
//! - CLIP-ViT-B/16: Vision Transformer Base with 16x16 patches
//! - CLIP-ViT-L/14: Vision Transformer Large with 14x14 patches
//!
//! # Example Usage
//! ```rust,no_run
//! use torsh_models::multimodal::clip::{CLIPModel, CLIPConfig};
//! use torsh_tensor::creation;
//!
//! // Create a CLIP-ViT-B/32 model
//! let model = CLIPModel::vit_b_32()?;
//!
//! // Encode images and text
//! let images = creation::randn(&[4, 3, 224, 224])?; // batch of 4 images
//! let texts = creation::randint(0, 49408, &[4, 77])?; // batch of 4 text sequences
//!
//! let image_features = model.encode_image(&images)?;
//! let text_features = model.encode_text(&texts)?;
//!
//! // Compute similarity
//! let similarity = model.compute_similarity(&images, &texts)?;
//! ```

pub mod config;
pub mod embeddings;
pub mod encoders;
pub mod layers;
pub mod models;

// Re-export main types for easy access
pub use config::{CLIPConfig, CLIPTextConfig, CLIPVisionConfig};
pub use embeddings::{CLIPTextEmbeddings, CLIPVisionEmbeddings};
pub use encoders::{
    CLIPTextEncoder, CLIPTextTransformer, CLIPVisionEncoder, CLIPVisionTransformer,
};
pub use layers::{CLIPAttention, CLIPEncoderLayer, CLIPMLP};
pub use models::{CLIPModel, CLIPModelWithOutput, CLIPOutput};

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation;

    #[test]
    fn test_clip_config_creation() {
        let config = CLIPConfig::vit_b_32();
        assert_eq!(config.projection_dim, 512);
        assert_eq!(config.vision_config.hidden_size, 768);
        assert_eq!(config.text_config.hidden_size, 512);
    }

    #[test]
    fn test_clip_config_validation() {
        let config = CLIPConfig::vit_b_32();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_vision_config_variants() {
        let vit_b_32 = CLIPVisionConfig::vit_b_32();
        let vit_b_16 = CLIPVisionConfig::vit_b_16();
        let vit_l_14 = CLIPVisionConfig::vit_l_14();

        assert_eq!(vit_b_32.patch_size, 32);
        assert_eq!(vit_b_16.patch_size, 16);
        assert_eq!(vit_l_14.patch_size, 14);

        assert_eq!(vit_b_32.hidden_size, 768);
        assert_eq!(vit_b_16.hidden_size, 768);
        assert_eq!(vit_l_14.hidden_size, 1024);
    }

    #[test]
    fn test_vision_config_calculations() {
        let config = CLIPVisionConfig::vit_b_32();
        assert_eq!(config.num_patches(), 49); // (224/32)^2 = 49
        assert_eq!(config.sequence_length(), 50); // 49 patches + 1 CLS token
        assert_eq!(config.attention_head_size(), 64); // 768 / 12 = 64
    }

    #[test]
    fn test_text_config_calculations() {
        let config = CLIPTextConfig::default();
        assert_eq!(config.attention_head_size(), 64); // 512 / 8 = 64
    }

    // Note: Actual model tests would require proper tensor operations and are commented out
    // as they need the full ToRSh infrastructure to be available.

    /*
    #[test]
    fn test_clip_model_creation() -> Result<(), Box<dyn std::error::Error>> {
        let model = CLIPModel::vit_b_32()?;
        assert!(model.get_logit_scale()? > 0.0);
        Ok(())
    }

    #[test]
    fn test_clip_forward_pass() -> Result<(), Box<dyn std::error::Error>> {
        let model = CLIPModel::vit_b_32()?;

        let images = creation::randn(&[2, 3, 224, 224])?;
        let texts = creation::randint(0, 49408, &[2, 77])?;

        let image_features = model.encode_image(&images)?;
        let text_features = model.encode_text(&texts)?;

        assert_eq!(image_features.shape(), &[2, 512]);
        assert_eq!(text_features.shape(), &[2, 512]);

        let similarity = model.compute_similarity(&images, &texts)?;
        assert_eq!(similarity.shape(), &[2, 2]);

        Ok(())
    }
    */
}
