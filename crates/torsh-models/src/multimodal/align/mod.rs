//! ALIGN (Large-scale Vision-Language Learning) Model Implementation
//!
//! ALIGN is Google's large-scale vision-language model that learns from
//! noisy web data. It uses an EfficientNet-style vision encoder and BERT-style
//! text encoder to create aligned representations for contrastive learning.
//!
//! Key Features:
//! - EfficientNet-B7 based vision encoder with MobileNet Inverted Residual blocks
//! - BERT-Large based text encoder with 24 layers
//! - Contrastive learning framework for vision-text alignment
//! - Support for large-scale web data training
//! - Advanced mobile-optimized convolution blocks (MBConv)
//! - Squeeze-and-Excitation attention in vision encoder
//!
//! References:
//! - ALIGN: Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision
//! - EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

pub mod config;
pub mod factory;
pub mod model;
pub mod text;
pub mod vision;

// Re-export main components for backward compatibility
pub use config::{ALIGNConfig, ALIGNTextConfig, ALIGNVisionConfig, MBConvBlockArgs};
pub use factory::ALIGNFactory;
pub use model::ALIGNModel;

// Re-export vision components
pub use vision::{ALIGNVisionEncoder, GlobalAveragePooling2d, MBConvBlock, SqueezeExcitation};

// Re-export text components
pub use text::{
    ALIGNBertAttention, ALIGNBertEncoder, ALIGNBertIntermediate, ALIGNBertLayer, ALIGNBertOutput,
    ALIGNBertSelfOutput, ALIGNTextEmbeddings, ALIGNTextEncoder,
};

// Comprehensive test suite for ALIGN models
#[cfg(test)]
mod tests {
    use super::*;
    use torsh_nn::Module;

    #[test]
    fn test_align_config_creation() {
        let config = ALIGNConfig::align_large();
        assert_eq!(config.projection_dim, 640);
        assert_eq!(config.temperature, 0.07);
        assert!(config.learnable_temperature);

        let small_config = ALIGNConfig::align_small();
        assert_eq!(small_config.projection_dim, 512);
    }

    #[test]
    fn test_align_config_validation() {
        let mut config = ALIGNConfig::default();
        assert!(config.validate().is_ok());

        config.projection_dim = 0;
        assert!(config.validate().is_err());

        config.projection_dim = 640;
        config.temperature = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_align_model_creation() {
        let model = ALIGNModel::align_large();
        assert!(model.is_ok());

        let small_model = ALIGNModel::align_small();
        assert!(small_model.is_ok());
    }

    #[test]
    fn test_vision_config_variants() {
        let b7_config = ALIGNVisionConfig::default();
        assert_eq!(b7_config.image_size, 600);
        assert_eq!(b7_config.head_size, 2560);

        let b3_config = ALIGNVisionConfig::efficientnet_b3();
        assert_eq!(b3_config.image_size, 300);
        assert_eq!(b3_config.head_size, 1536);
    }

    #[test]
    fn test_text_config_variants() {
        let large_config = ALIGNTextConfig::default();
        assert_eq!(large_config.hidden_size, 1024);
        assert_eq!(large_config.num_hidden_layers, 24);

        let base_config = ALIGNTextConfig::bert_base();
        assert_eq!(base_config.hidden_size, 768);
        assert_eq!(base_config.num_hidden_layers, 12);
    }

    #[test]
    fn test_mbconv_block_args() {
        let args = MBConvBlockArgs {
            kernel_size: 3,
            num_repeat: 2,
            input_filters: 32,
            output_filters: 64,
            expand_ratio: 6,
            se_ratio: 0.25,
            stride: 2,
        };

        assert_eq!(args.kernel_size, 3);
        assert_eq!(args.output_filters, 64);
        assert_eq!(args.expand_ratio, 6);
    }

    #[test]
    fn test_align_factory() {
        let model = ALIGNFactory::create_by_name("align-large");
        assert!(model.is_ok());

        let small_model = ALIGNFactory::create_by_name("align-small");
        assert!(small_model.is_ok());

        let invalid_model = ALIGNFactory::create_by_name("invalid");
        assert!(invalid_model.is_err());
    }

    #[test]
    fn test_model_info() {
        let info = ALIGNFactory::model_info("align-large");
        assert!(info.is_ok());
        assert!(info.unwrap().contains("EfficientNet-B7"));

        let small_info = ALIGNFactory::model_info("align-small");
        assert!(small_info.is_ok());
        assert!(small_info.unwrap().contains("EfficientNet-B3"));
    }

    #[test]
    fn test_comparison_info() {
        let comparison = ALIGNFactory::comparison_info();
        assert!(comparison.contains("ALIGN"));
        assert!(comparison.contains("CLIP"));
        assert!(comparison.contains("BLIP"));
        assert!(comparison.contains("Flamingo"));
    }

    #[test]
    fn test_squeeze_excitation() {
        let se = SqueezeExcitation::new(64, 0.25);

        // Test that parameters are created
        let params = se.parameters();
        assert!(params.contains_key("reduce.weight"));
        assert!(params.contains_key("expand.weight"));
    }

    #[test]
    fn test_global_average_pooling() {
        let gap = GlobalAveragePooling2d::new();

        // Should have no parameters
        let params = gap.parameters();
        assert!(params.is_empty());
    }

    #[test]
    fn test_attention_head_size() {
        let config = ALIGNTextConfig::default();
        assert_eq!(config.attention_head_size(), 64); // 1024 / 16

        let base_config = ALIGNTextConfig::bert_base();
        assert_eq!(base_config.attention_head_size(), 64); // 768 / 12
    }
}
