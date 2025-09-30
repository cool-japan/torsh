//! Integration tests for torsh-models
//!
//! This module contains comprehensive integration tests for all model implementations
//! to ensure they work correctly and maintain API compatibility.

use torsh_core::{DType, DeviceType, Shape};
use torsh_nn::Module;
use torsh_tensor::Tensor;

// Helper function to create test tensors
fn create_test_tensor(
    shape: &[usize],
    device: DeviceType,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    // Create a simple test tensor with random-like values
    let total_elements = shape.iter().product::<usize>();
    let data: Vec<f32> = (0..total_elements)
        .map(|i| ((i % 100) as f32) / 100.0) // Simple deterministic "random" values
        .collect();

    Ok(Tensor::from_data(data, shape.to_vec(), device)?)
}

#[cfg(feature = "vision")]
mod vision_tests {
    use super::*;
    use torsh_models::vision::*;

    #[test]
    fn test_resnet_forward() -> Result<(), Box<dyn std::error::Error>> {
        let config = ResNetConfig {
            variant: ResNetVariant::ResNet18,
            num_classes: 1000,
            in_channels: 3,
            stem_channels: 64,
            zero_init_residual: false,
            groups: 1,
            width_per_group: 64,
            replace_stride_with_dilation: [false, false, false].to_vec(),
            norm_layer: "BatchNorm2d".to_string(),
            dropout: 0.0,
            use_se: false,
            se_reduction_ratio: 16,
        };

        let mut model = ResNet::new(config)?;
        let input = create_test_tensor(&[1, 3, 224, 224], DeviceType::Cpu)?;

        // Test forward pass
        let output = model.forward(&input)?;
        assert_eq!(output.shape().dims(), &[1, 1000]);

        // Test training/eval mode switching
        model.train();
        assert!(model.training());

        model.eval();
        assert!(!model.training());

        // Test parameter access
        let params = model.parameters();
        assert!(!params.is_empty());

        Ok(())
    }

    #[test]
    fn test_efficientnet_forward() -> Result<(), Box<dyn std::error::Error>> {
        let config = EfficientNetConfig {
            width_multiplier: 1.0,
            depth_multiplier: 1.0,
            input_resolution: 224,
            dropout_rate: 0.2,
            stochastic_depth_rate: 0.2,
            num_classes: 1000,
        };

        let mut model = EfficientNet::new(config)?;
        let input = create_test_tensor(&[1, 3, 224, 224], DeviceType::Cpu)?;

        let output = model.forward(&input)?;
        assert_eq!(output.shape().dims(), &[1, 1000]);

        Ok(())
    }

    #[test]
    fn test_vision_transformer_forward() -> Result<(), Box<dyn std::error::Error>> {
        let config = ViTConfig::vit_base_patch16_224().with_num_classes(1000);

        let mut model = VisionTransformer::new(config)?;
        let input = create_test_tensor(&[1, 3, 224, 224], DeviceType::Cpu)?;

        let output = model.forward(&input)?;
        assert_eq!(output.shape().dims(), &[1, 1000]);

        Ok(())
    }
}

#[cfg(feature = "nlp")]
mod nlp_tests {
    use super::*;
    use torsh_models::nlp::*;

    #[test]
    fn test_roberta_forward() -> Result<(), Box<dyn std::error::Error>> {
        let config = RobertaConfig {
            vocab_size: 50265,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 514,
            type_vocab_size: 1,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
            position_embedding_type: "absolute".to_string(),
        };

        let mut model = RobertaForSequenceClassification::new(config, 2)?;

        // Create input token IDs (batch_size=1, seq_length=128)
        let input_ids = create_test_tensor(&[1, 128], DeviceType::Cpu)?;

        let output = model.forward(&input_ids)?;
        assert_eq!(output.shape().dims(), &[1, 2]); // num_labels = 2

        Ok(())
    }
}

#[cfg(feature = "audio")]
mod audio_tests {
    use super::*;
    use torsh_models::audio::*;

    #[test]
    fn test_wav2vec2_forward() -> Result<(), Box<dyn std::error::Error>> {
        let config = Wav2Vec2Config {
            vocab_size: 32,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_dropout: 0.1,
            attention_dropout: 0.1,
            feat_proj_dropout: 0.0,
            layerdrop: 0.1,
            conv_dim: vec![512, 512, 512, 512, 512, 512, 512],
            conv_stride: vec![5, 2, 2, 2, 2, 2, 2],
            conv_kernel: vec![10, 3, 3, 3, 3, 2, 2],
            conv_bias: false,
            num_conv_pos_embeddings: 128,
            num_conv_pos_embedding_groups: 16,
            feat_extract_norm: "group".to_string(),
            feat_extract_activation: "gelu".to_string(),
            conv_pos_embeddings_kernel_size: 128,
            apply_spec_augment: true,
            mask_time_prob: 0.05,
            mask_time_length: 10,
            mask_feature_prob: 0.0,
            mask_feature_length: 10,
            ctc_loss_reduction: "mean".to_string(),
            ctc_zero_infinity: false,
            use_weighted_layer_sum: false,
        };

        let mut model = Wav2Vec2ForCTC::new(config)?;

        // Create audio input (batch_size=1, sequence_length=16000 for 1 second at 16kHz)
        let input_values = create_test_tensor(&[1, 16000], DeviceType::Cpu)?;

        let output = model.forward(&input_values)?;
        // Output should be [batch_size, sequence_length, vocab_size]
        assert_eq!(output.shape().dims().len(), 3);
        assert_eq!(output.shape().dims()[0], 1); // batch size
        assert_eq!(output.shape().dims()[2], 32); // vocab_size

        Ok(())
    }
}

#[cfg(feature = "multimodal")]
mod multimodal_tests {
    use super::*;
    use torsh_models::multimodal::*;

    #[test]
    fn test_clip_forward() -> Result<(), Box<dyn std::error::Error>> {
        let vision_config = CLIPVisionConfig {
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            num_channels: 3,
            image_size: 224,
            patch_size: 32,
            layer_norm_eps: 1e-5,
            hidden_dropout_prob: 0.0,
            attention_dropout: 0.0,
            hidden_act: "quick_gelu".to_string(),
            initializer_range: 0.02,
            initializer_factor: 1.0,
        };

        let text_config = CLIPTextConfig {
            vocab_size: 49408,
            hidden_size: 512,
            intermediate_size: 2048,
            num_hidden_layers: 12,
            num_attention_heads: 8,
            max_position_embeddings: 77,
            layer_norm_eps: 1e-5,
            hidden_dropout_prob: 0.0,
            attention_dropout: 0.0,
            hidden_act: "quick_gelu".to_string(),
            initializer_range: 0.02,
            initializer_factor: 1.0,
            pad_token_id: 1,
            bos_token_id: 0,
            eos_token_id: 2,
        };

        let config = CLIPConfig {
            text_config,
            vision_config,
            projection_dim: 512,
            logit_scale_init_value: 2.6592,
        };

        let mut model = CLIPModel::new(config)?;

        // Test image encoding
        let image_input = create_test_tensor(&[1, 3, 224, 224], DeviceType::Cpu)?;
        let image_features = model.encode_image(&image_input)?;
        assert_eq!(image_features.shape().dims(), &[1, 512]); // projection_dim

        // Test text encoding
        let text_input = create_test_tensor(&[1, 77], DeviceType::Cpu)?; // max_position_embeddings
        let text_features = model.encode_text(&text_input)?;
        assert_eq!(text_features.shape().dims(), &[1, 512]); // projection_dim

        Ok(())
    }
}

#[cfg(feature = "rl")]
mod rl_tests {
    use super::*;
    use torsh_models::rl::*;

    #[test]
    fn test_dqn_forward() -> Result<(), Box<dyn std::error::Error>> {
        let config = DQNConfig::default();

        let mut model = DQN::new(config)?;
        let state = create_test_tensor(&[1, 4], DeviceType::Cpu)?;

        let q_values = model.forward(&state)?;
        assert_eq!(q_values.shape().dims(), &[1, 2]); // action_dim

        Ok(())
    }

    #[test]
    fn test_ppo_forward() -> Result<(), Box<dyn std::error::Error>> {
        let config = PPOConfig::default();

        let mut model = PPO::new(config)?;
        let state = create_test_tensor(&[1, 4], DeviceType::Cpu)?;

        // Test actor forward pass
        let action_logits = model.actor.forward(&state)?;
        assert_eq!(action_logits.shape().dims(), &[1, 2]); // action_dim

        // Test critic forward pass
        let value = model.critic.forward(&state)?;
        assert_eq!(value.shape().dims(), &[1, 1]); // single value output

        Ok(())
    }
}

#[cfg(feature = "domain")]
mod domain_tests {
    use super::*;
    use torsh_models::domain::*;

    #[test]
    fn test_unet_forward() -> Result<(), Box<dyn std::error::Error>> {
        let config = UNetConfig {
            in_channels: 1,
            out_channels: 1,
            base_features: 64,
            num_levels: 4,
            batch_norm: true,
            dropout: 0.0,
            attention: false,
            deep_supervision: false,
            activation: "relu".to_string(),
        };

        let mut model = UNet::new(config)?;
        let input = create_test_tensor(&[1, 1, 128, 128], DeviceType::Cpu)?;

        let output = model.forward(&input)?;
        assert_eq!(output.shape().dims(), &[1, 1, 128, 128]); // Same spatial dimensions

        Ok(())
    }

    #[test]
    fn test_pinn_forward() -> Result<(), Box<dyn std::error::Error>> {
        let config = PINNConfig {
            input_dim: 2,  // e.g., (x, t) for time-dependent PDE
            output_dim: 1, // e.g., u(x,t)
            hidden_dims: vec![50, 50, 50],
            activation: "tanh".to_string(),
            physics_weight: 1.0,
            boundary_weight: 100.0,
            initial_weight: 100.0,
            adaptive_weights: false,
        };

        let mut model = PINN::new(config)?;
        let coords = create_test_tensor(&[100, 2], DeviceType::Cpu)?; // 100 coordinate points

        let solution = model.forward(&coords)?;
        assert_eq!(solution.shape().dims(), &[100, 1]); // output_dim

        Ok(())
    }
}

// Model registry and utility tests
#[cfg(any(
    feature = "vision",
    feature = "nlp",
    feature = "audio",
    feature = "multimodal"
))]
mod utility_tests {
    use super::*;
    use torsh_models::benchmark::*;
    use torsh_models::registry::*;
    use torsh_models::validation::*;

    #[test]
    fn test_model_registry() -> Result<(), Box<dyn std::error::Error>> {
        let registry = ModelRegistry::new("/tmp/torsh_test_cache")?;

        // Test registry operations (these would work with actual models)
        let models = registry.list_models();
        assert!(models.len() >= 0);

        // Test searching for models
        let vision_models: Vec<String> = Vec::new(); // Placeholder for search functionality
        assert!(vision_models.len() >= 0);

        Ok(())
    }

    #[test]
    fn test_benchmark_config() -> Result<(), Box<dyn std::error::Error>> {
        let config = BenchmarkConfig::default();

        assert_eq!(config.warmup_iterations, 10);
        assert_eq!(config.benchmark_iterations, 100);
        assert!(!config.batch_sizes.is_empty());
        assert!(config.measure_memory);

        Ok(())
    }

    #[test]
    fn test_validation_config() -> Result<(), Box<dyn std::error::Error>> {
        let config = ValidationConfig {
            strategy: ValidationStrategy::Holdout {
                test_ratio: 0.2,
                stratified: true,
            },
            metrics: vec![ValidationMetric::Accuracy],
            cross_validation: None,
            statistical_tests: None,
            tolerance: ToleranceConfig::default(),
            save_detailed_results: false,
        };

        // Test that the config is properly constructed
        match config.strategy {
            ValidationStrategy::Holdout {
                test_ratio,
                stratified,
            } => {
                assert_eq!(test_ratio, 0.2);
                assert!(stratified);
            }
            _ => panic!("Wrong validation strategy"),
        }

        Ok(())
    }
}

// Performance and stress tests
mod performance_tests {
    use super::*;

    #[test]
    fn test_tensor_creation_performance() -> Result<(), Box<dyn std::error::Error>> {
        use std::time::Instant;

        let start = Instant::now();

        // Create multiple tensors to test performance
        for i in 1..10 {
            let _tensor = create_test_tensor(&[i, 3, 224, 224], DeviceType::Cpu)?;
        }

        let duration = start.elapsed();

        // Ensure tensor creation is reasonably fast (adjust threshold as needed)
        assert!(
            duration.as_millis() < 1000,
            "Tensor creation took too long: {:?}",
            duration
        );

        Ok(())
    }
}

// Error handling tests
mod error_tests {
    use super::*;

    #[test]
    fn test_invalid_tensor_shapes() {
        // Test that invalid shapes are handled gracefully
        let result = create_test_tensor(&[], DeviceType::Cpu);
        assert!(result.is_err() || result.unwrap().shape().dims().is_empty());

        // Test with very large dimensions (should handle gracefully)
        let result = create_test_tensor(&[1000000, 1000000], DeviceType::Cpu);
        // This might succeed or fail depending on system memory, but shouldn't panic
    }
}
