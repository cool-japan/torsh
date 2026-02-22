//! ALIGN Factory Functions and Utilities
//!
//! Factory methods for creating ALIGN models and utility functions
//! for model information and comparisons.

use torsh_core::error::{Result, TorshError};

use super::model::ALIGNModel;

/// ALIGN Factory Functions and Utilities
pub struct ALIGNFactory;

impl ALIGNFactory {
    /// Create ALIGN model by name
    pub fn create_by_name(model_name: &str) -> Result<ALIGNModel> {
        match model_name.to_lowercase().as_str() {
            "align-large" | "align_large" => ALIGNModel::align_large(),
            "align-small" | "align_small" => ALIGNModel::align_small(),
            _ => Err(TorshError::InvalidArgument(format!(
                "Unknown ALIGN model: {}. Available: align-large, align-small",
                model_name
            ))),
        }
    }

    /// Get model information
    pub fn model_info(model_name: &str) -> Result<String> {
        match model_name.to_lowercase().as_str() {
            "align-large" | "align_large" => Ok(format!(
                "ALIGN-Large: Vision=EfficientNet-B7 (2560 dim), Text=BERT-Large (1024 dim), Projection=640 dim"
            )),
            "align-small" | "align_small" => Ok(format!(
                "ALIGN-Small: Vision=EfficientNet-B3 (1536 dim), Text=BERT-Base (768 dim), Projection=512 dim"
            )),
            _ => Err(TorshError::InvalidArgument(format!(
                "Unknown ALIGN model: {}",
                model_name
            ))),
        }
    }

    /// Compare ALIGN with other multimodal models
    pub fn comparison_info() -> String {
        format!(
            "ALIGN vs Other Models:\n\
            - ALIGN: Large-scale noisy web data training, EfficientNet + BERT\n\
            - CLIP: Curated dataset training, ResNet/ViT + Transformer\n\
            - BLIP: Bootstrapped vision-language understanding, ViT + BERT\n\
            - Flamingo: Few-shot learning, frozen CLIP + language model"
        )
    }
}
