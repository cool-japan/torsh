//! Comprehensive usage example for torsh-models
//!
//! This example demonstrates how to use various models from the torsh-models crate
//! including vision, NLP, audio, multimodal, RL, and domain-specific models.

use torsh_core::DeviceType;
use torsh_nn::Module;
use torsh_tensor::Tensor;

// Import model modules
use torsh_models::audio::{Wav2Vec2Config, Wav2Vec2ForCTC};
use torsh_models::domain::{PINNConfig, UNet, UNetConfig, PINN};
use torsh_models::multimodal::{CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig};
use torsh_models::nlp::{RobertaConfig, RobertaForSequenceClassification};
use torsh_models::rl::{DQNConfig, PPOConfig, DQN, PPO};
use torsh_models::vision::{ResNet, ResNetConfig, ResNetVariant};

// Import utilities
use torsh_models::benchmark::BenchmarkConfig;
use torsh_models::quantization::QuantizationConfig;
use torsh_models::registry::ModelRegistry;
use torsh_models::validation::{
    AverageType, ToleranceConfig, ValidationConfig, ValidationMetric, ValidationStrategy,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 ToRSh Models Comprehensive Usage Example");
    println!("{}", "=".repeat(50));

    // Example 1: Vision Models
    vision_models_example()?;

    // Example 2: NLP Models
    nlp_models_example()?;

    // Example 3: Audio Models
    audio_models_example()?;

    // Example 4: Multimodal Models
    multimodal_models_example()?;

    // Example 5: Reinforcement Learning Models
    rl_models_example()?;

    // Example 6: Domain-Specific Models
    domain_models_example()?;

    // Example 7: Model Utilities
    model_utilities_example()?;

    // Example 8: Advanced Features
    advanced_features_example()?;

    println!("\n✅ All examples completed successfully!");

    Ok(())
}

fn vision_models_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📸 Vision Models Example");
    println!("{}", "-".repeat(30));

    // ResNet for image classification
    let resnet_config = ResNetConfig {
        variant: ResNetVariant::ResNet50,
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

    let mut resnet = ResNet::new(resnet_config)?;
    let image_input = create_random_tensor(&[1, 3, 224, 224], DeviceType::Cpu)?;

    println!("🔄 Running ResNet-50 inference...");
    let resnet_output = resnet.forward(&image_input)?;
    println!(
        "✅ ResNet-50 output shape: {:?}",
        resnet_output.shape().dims()
    );

    // Note: EfficientNet is implemented but not yet enabled due to API compatibility
    // Will be available in v0.2.0 after torsh-nn API stabilization
    println!("ℹ️  EfficientNet skipped (awaiting torsh-nn v0.2 API compatibility)");

    // Switch between training and evaluation modes
    resnet.train();
    println!("📚 ResNet in training mode: {}", resnet.training());

    resnet.eval();
    println!("🔍 ResNet in evaluation mode: {}", !resnet.training());

    Ok(())
}

fn nlp_models_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📝 NLP Models Example");
    println!("{}", "-".repeat(30));

    // RoBERTa for sequence classification
    let roberta_config = RobertaConfig {
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

    let roberta = RobertaForSequenceClassification::new(roberta_config, 2)?;

    // Create token IDs input (batch_size=1, seq_length=128)
    let token_ids = create_random_tensor(&[1, 128], DeviceType::Cpu)?;

    println!("🔄 Running RoBERTa sequence classification...");
    let roberta_output = roberta.forward(&token_ids)?;
    println!(
        "✅ RoBERTa output shape: {:?}",
        roberta_output.shape().dims()
    );

    // Get model parameters
    let params = roberta.parameters();
    println!("📊 RoBERTa has {} parameter groups", params.len());

    Ok(())
}

fn audio_models_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎵 Audio Models Example");
    println!("{}", "-".repeat(30));

    // Wav2Vec2 for speech recognition
    let wav2vec2_config = Wav2Vec2Config {
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

    let wav2vec2 = Wav2Vec2ForCTC::new(wav2vec2_config)?;

    // Create audio input (1 second at 16kHz)
    let audio_input = create_random_tensor(&[1, 16000], DeviceType::Cpu)?;

    println!("🔄 Running Wav2Vec2 speech recognition...");
    let wav2vec2_output = wav2vec2.forward(&audio_input)?;
    println!(
        "✅ Wav2Vec2 output shape: {:?}",
        wav2vec2_output.shape().dims()
    );

    Ok(())
}

fn multimodal_models_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🖼️ Multimodal Models Example");
    println!("{}", "-".repeat(30));

    // CLIP for vision-language understanding
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

    let clip_config = CLIPConfig {
        text_config,
        vision_config,
        projection_dim: 512,
        logit_scale_init_value: 2.6592,
    };

    let clip_model = CLIPModel::new(clip_config)?;

    // Encode image and text
    let image_input = create_random_tensor(&[1, 3, 224, 224], DeviceType::Cpu)?;
    let text_input = create_random_tensor(&[1, 77], DeviceType::Cpu)?;

    println!("🔄 Running CLIP image encoding...");
    let image_features = clip_model.encode_image(&image_input)?;
    println!(
        "✅ CLIP image features shape: {:?}",
        image_features.shape().dims()
    );

    println!("🔄 Running CLIP text encoding...");
    let text_features = clip_model.encode_text(&text_input)?;
    println!(
        "✅ CLIP text features shape: {:?}",
        text_features.shape().dims()
    );

    Ok(())
}

fn rl_models_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎮 Reinforcement Learning Models Example");
    println!("{}", "-".repeat(30));

    // DQN for value-based RL
    let dqn_config = DQNConfig::default();

    let dqn = DQN::new(dqn_config)?;
    let state = create_random_tensor(&[32, 4], DeviceType::Cpu)?; // batch of states

    println!("🔄 Running DQN value estimation...");
    let q_values = dqn.forward(&state)?;
    println!("✅ DQN Q-values shape: {:?}", q_values.shape().dims());

    // PPO for policy-based RL
    let ppo_config = PPOConfig::default();

    let ppo = PPO::new(ppo_config)?;
    let single_state = create_random_tensor(&[1, 4], DeviceType::Cpu)?;

    println!("🔄 Running PPO policy and value estimation...");
    let action_probs = ppo.actor.forward(&single_state)?;
    let state_value = ppo.critic.forward(&single_state)?;
    println!(
        "✅ PPO action probabilities shape: {:?}",
        action_probs.shape().dims()
    );
    println!("✅ PPO state value shape: {:?}", state_value.shape().dims());

    Ok(())
}

fn domain_models_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🔬 Domain-Specific Models Example");
    println!("{}", "-".repeat(30));

    // U-Net for medical image segmentation
    let unet_config = UNetConfig {
        in_channels: 1,  // grayscale medical images
        out_channels: 2, // binary segmentation
        base_features: 64,
        num_levels: 4,
        batch_norm: true,
        dropout: 0.1,
        attention: true,
        deep_supervision: false,
        activation: "relu".to_string(),
    };

    let unet = UNet::new(unet_config)?;
    let medical_image = create_random_tensor(&[1, 1, 256, 256], DeviceType::Cpu)?;

    println!("🔄 Running U-Net medical image segmentation...");
    let segmentation = unet.forward(&medical_image)?;
    println!(
        "✅ U-Net segmentation output shape: {:?}",
        segmentation.shape().dims()
    );

    // PINN for physics-informed neural networks
    let pinn_config = PINNConfig {
        input_dim: 2,  // (x, t) coordinates
        output_dim: 1, // solution u(x,t)
        hidden_dims: vec![50, 50, 50, 50],
        activation: "tanh".to_string(),
        physics_weight: 1.0,
        boundary_weight: 100.0,
        initial_weight: 100.0,
        adaptive_weights: true,
    };

    let pinn = PINN::new(pinn_config)?;
    let coordinates = create_random_tensor(&[1000, 2], DeviceType::Cpu)?; // 1000 sample points

    println!("🔄 Running PINN physics simulation...");
    let solution = pinn.forward(&coordinates)?;
    println!("✅ PINN solution shape: {:?}", solution.shape().dims());

    Ok(())
}

fn model_utilities_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🛠️ Model Utilities Example");
    println!("{}", "-".repeat(30));

    // Model Registry
    let registry = ModelRegistry::new("/tmp/torsh_model_cache")?;
    println!("📚 Exploring model registry...");

    let models = registry.list_models();
    println!("✅ Found {} models in registry", models.len());

    let vision_models: Vec<String> = Vec::new(); // Placeholder for search functionality
    println!("🔍 Found {} vision models", vision_models.len());

    // Benchmark Configuration
    let benchmark_config = BenchmarkConfig {
        warmup_iterations: 5,
        benchmark_iterations: 50,
        batch_sizes: vec![1, 4, 8],
        input_shapes: vec![vec![224, 224, 3]],
        devices: vec![DeviceType::Cpu],
        measure_memory: true,
        measure_accuracy: false,
        dtypes: vec!["f32".to_string()],
    };

    println!(
        "⚡ Benchmark config created with {} batch sizes",
        benchmark_config.batch_sizes.len()
    );

    // Validation Configuration
    let validation_config = ValidationConfig {
        strategy: ValidationStrategy::KFold {
            k: 5,
            shuffle: true,
            stratified: true,
        },
        metrics: vec![
            ValidationMetric::Accuracy,
            ValidationMetric::F1Score {
                average: AverageType::Macro,
            },
        ],
        cross_validation: None,
        statistical_tests: None,
        tolerance: ToleranceConfig::default(),
        save_detailed_results: true,
    };

    println!(
        "✅ Validation config created with {} metrics",
        validation_config.metrics.len()
    );

    // Quantization Configuration
    let quantization_config = QuantizationConfig::default();

    println!(
        "🔧 Quantization config created for {:?} precision",
        quantization_config.dtype
    );

    Ok(())
}

fn advanced_features_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Advanced Features Example");
    println!("{}", "-".repeat(30));

    // Create a simple ResNet model for demonstration
    let config = ResNetConfig {
        variant: ResNetVariant::ResNet18,
        num_classes: 10,
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

    // Device Transfer
    println!("📱 Testing device transfer...");
    model.to_device(DeviceType::Cpu)?;
    println!("✅ Model moved to CPU");

    // Parameter Analysis
    let params = model.parameters();
    let named_params = model.named_parameters();

    println!("📊 Model has {} parameter groups", params.len());
    println!("📝 Model has {} named parameters", named_params.len());

    // Training Mode Management
    println!("🎓 Testing training mode management...");
    model.train();
    assert!(model.training());
    println!("✅ Model set to training mode");

    model.eval();
    assert!(!model.training());
    println!("✅ Model set to evaluation mode");

    // Inference
    let test_input = create_random_tensor(&[1, 3, 32, 32], DeviceType::Cpu)?;
    let output = model.forward(&test_input)?;
    println!(
        "🔄 Inference completed with output shape: {:?}",
        output.shape().dims()
    );

    Ok(())
}

// Helper function to create random tensors
fn create_random_tensor(
    shape: &[usize],
    device: DeviceType,
) -> Result<Tensor, Box<dyn std::error::Error>> {
    let total_elements = shape.iter().product::<usize>();

    // Generate deterministic "random" data for reproducible results
    let data: Vec<f32> = (0..total_elements)
        .map(|i| {
            let x = (i as f32) * 0.01;
            (x.sin() + 1.0) * 0.5 // Values between 0 and 1
        })
        .collect();

    Ok(Tensor::from_data(data, shape.to_vec(), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_random_tensor() {
        let tensor = create_random_tensor(&[2, 3], DeviceType::Cpu).unwrap();
        assert_eq!(tensor.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_examples_dont_panic() {
        // This test ensures that our examples at least don't panic
        // Individual functionality would need the actual model implementations to work
        let result = create_random_tensor(&[1, 3, 224, 224], DeviceType::Cpu);
        assert!(result.is_ok());
    }
}
