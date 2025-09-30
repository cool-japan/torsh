//! Multimodal models for ToRSh deep learning framework
//!
//! This module provides multimodal models that work across vision and language:
//! - **CLIP** for contrastive vision-language learning
//! - **ALIGN** for large-scale alignment (placeholder)
//! - **Flamingo** for few-shot learning (placeholder)
//! - **DALL-E** for text-to-image generation (placeholder)
//! - **BLIP** for bootstrapped vision-language understanding (placeholder)
//! - **LLaVA** for large language and vision assistant (placeholder)
//! - **InstructBLIP** for instructional multimodal learning (placeholder)
//!
//! # Example Usage
//!
//! ```rust,no_run
//! use torsh_models::multimodal::{CLIPModel, MultimodalArchitecture};
//! use torsh_tensor::creation;
//!
//! // Create a CLIP model
//! let model = CLIPModel::vit_b_32()?;
//!
//! // Process vision and language inputs
//! let images = creation::randn(&[4, 3, 224, 224])?;
//! let texts = creation::randint(0, 49408, &[4, 77])?;
//!
//! let similarity = model.compute_similarity(&images, &texts)?;
//! ```
//!
//! # Architecture Overview
//!
//! The multimodal module is organized into model families:
//!
//! - `common/` - Shared components and utilities
//! - `clip/` - CLIP model family implementation
//! - `align/` - ALIGN model family (placeholder)
//! - `flamingo/` - Flamingo model family (placeholder)
//! - `dalle/` - DALL-E model family (placeholder)
//! - `blip/` - BLIP model family (placeholder)
//! - `llava/` - LLaVA model family (placeholder)
//! - `instructblip/` - InstructBLIP model family (placeholder)

// Common components
pub mod common;

// Model families
pub mod clip;

// Model families - ALIGN implemented with modular architecture
pub mod align;

pub mod flamingo {
    //! Flamingo model family (placeholder)
    //!
    //! This module will contain the Flamingo model implementations
    //! once migrated from the monolithic multimodal.rs file.
}

pub mod dalle {
    //! DALL-E model family (placeholder)
    //!
    //! This module will contain the DALL-E model implementations
    //! once migrated from the monolithic multimodal.rs file.
}

pub mod blip {
    //! BLIP model family (placeholder)
    //!
    //! This module will contain the BLIP model implementations
    //! once migrated from the monolithic multimodal.rs file.
}

pub mod llava {
    //! LLaVA model family (placeholder)
    //!
    //! This module will contain the LLaVA model implementations
    //! once migrated from the monolithic multimodal.rs file.
}

pub mod instructblip {
    //! InstructBLIP model family (placeholder)
    //!
    //! This module will contain the InstructBLIP model implementations
    //! once migrated from the monolithic multimodal.rs file.
}

// Re-export common types for backward compatibility
pub use common::activations::{QuickGELU, SiLU};
pub use common::types::{MultimodalArchitecture, MultimodalTask};
pub use common::utils::{CrossModalProjection, GlobalAveragePooling2d, SqueezeExcitation};

// Re-export CLIP models for easy access
pub use clip::{
    CLIPAttention, CLIPConfig, CLIPEncoderLayer, CLIPModel, CLIPModelWithOutput, CLIPOutput,
    CLIPTextConfig, CLIPTextEmbeddings, CLIPTextTransformer, CLIPVisionConfig,
    CLIPVisionEmbeddings, CLIPVisionTransformer, CLIPMLP,
};

// ALIGN model family - fully implemented with modular architecture
pub use align::{
    ALIGNBertAttention, ALIGNBertEncoder, ALIGNBertLayer, ALIGNConfig, ALIGNFactory, ALIGNModel,
    ALIGNTextConfig, ALIGNTextEmbeddings, ALIGNTextEncoder, ALIGNVisionConfig, ALIGNVisionEncoder,
    MBConvBlock, MBConvBlockArgs,
};

// Placeholder re-exports for other model families
// These will be uncommented as the models are migrated
/*
// Flamingo model family
pub use flamingo::{
    FlamingoModel, FlamingoConfig,
    PerceiverResampler, GatedCrossAttention,
    // ... other Flamingo exports
};

// DALL-E model family
pub use dalle::{
    DallEModel, DallEConfig,
    DallEVisionDecoder, DallETextEncoder,
    VectorQuantizer, VQConfig,
    // ... other DALL-E exports
};

// BLIP model family
pub use blip::{
    BLIPModel, BLIPConfig,
    BLIPVisionTransformer, BLIPQFormer,
    // ... other BLIP exports
};

// LLaVA model family
pub use llava::{
    LLaVAModel, LLaVAConfig,
    LLaVAVisionTransformer, LLaVAMultiModalProjector,
    // ... other LLaVA exports
};

// InstructBLIP model family
pub use instructblip::{
    InstructBLIPModel, InstructBLIPConfig,
    InstructBLIPQFormer,
    // ... other InstructBLIP exports
};
*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multimodal_architecture_enum() {
        let arch = MultimodalArchitecture::CLIP;
        assert_eq!(arch.as_str(), "CLIP");

        let arch = MultimodalArchitecture::ALIGN;
        assert_eq!(arch.as_str(), "ALIGN");
    }

    #[test]
    fn test_multimodal_task_enum() {
        let task = MultimodalTask::ZeroShotImageClassification;
        assert_eq!(task.as_str(), "zero_shot_image_classification");

        let task = MultimodalTask::TextToImage;
        assert_eq!(task.as_str(), "text_to_image");
    }

    // Test that CLIP models are accessible through the re-exports
    #[test]
    fn test_clip_model_accessibility() {
        let config = CLIPConfig::vit_b_32();
        assert_eq!(config.projection_dim, 512);
    }
}

/// Utility function to create a multimodal model by architecture name
pub fn create_model_by_architecture(
    architecture: MultimodalArchitecture,
) -> Result<Box<dyn torsh_nn::Module>, torsh_core::error::TorshError> {
    match architecture {
        MultimodalArchitecture::CLIP => {
            let model = CLIPModel::vit_b_32()?;
            Ok(Box::new(model))
        }
        MultimodalArchitecture::ALIGN => {
            let model = ALIGNModel::align_large()?;
            Ok(Box::new(model))
        }
        _ => Err(torsh_core::error::TorshError::Other(format!(
            "Model architecture {} not yet implemented",
            architecture.as_str()
        ))),
    }
}

/// Utility function to list supported architectures
pub fn supported_architectures() -> Vec<MultimodalArchitecture> {
    vec![
        MultimodalArchitecture::CLIP,
        MultimodalArchitecture::ALIGN,
        // Other architectures will be added as they are implemented
    ]
}

/// Utility function to check if an architecture is supported
pub fn is_architecture_supported(architecture: &MultimodalArchitecture) -> bool {
    matches!(
        architecture,
        MultimodalArchitecture::CLIP | MultimodalArchitecture::ALIGN
    )
}
