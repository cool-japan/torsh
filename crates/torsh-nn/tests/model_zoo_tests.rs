//! Integration tests for the model zoo
//!
//! Tests that verify model zoo architectures work correctly with the broader
//! torsh-nn ecosystem including training, serialization, and optimization.

use std::collections::HashMap;
use torsh_nn::functional::*;
use torsh_nn::model_zoo::{ModelConfig, ModelZoo, PretrainedWeights};
use torsh_nn::Module;
use torsh_tensor::creation::{randn, tensor_1d, tensor_2d};

/// Test that all model zoo models can be created and run forward passes
#[test]
fn test_all_models_forward_pass() {
    let config = ModelConfig {
        num_classes: 10,
        dropout: 0.1,
        ..ModelConfig::default()
    };

    let models = ModelZoo::list_models();

    for model_name in &models {
        match model_name.as_str() {
            "mnist_mlp" => {
                let (model, metadata) = ModelZoo::create_model("mnist_mlp", &config, None).unwrap();
                let input = randn::<f32>(&[2, 784]).unwrap();
                let output = model.forward(&input).unwrap();
                assert_eq!(output.shape().dims(), &[2, 10]);
                assert!(metadata.num_parameters > 0);
            }
            "lenet5" => {
                let (model, metadata) = ModelZoo::create_model("lenet5", &config, None).unwrap();
                let input = randn::<f32>(&[2, 1, 32, 32]).unwrap();
                let output = model.forward(&input).unwrap();
                assert_eq!(output.shape().dims(), &[2, 10]);
                assert!(metadata.num_parameters > 0);
            }
            "cifar10_cnn" => {
                let (model, metadata) =
                    ModelZoo::create_model("cifar10_cnn", &config, None).unwrap();
                let input = randn::<f32>(&[2, 3, 32, 32]).unwrap();
                let output = model.forward(&input).unwrap();
                assert_eq!(output.shape().dims(), &[2, 10]);
                assert!(metadata.num_parameters > 0);
            }
            "resnet_basic" => {
                let (model, metadata) =
                    ModelZoo::create_model("resnet_basic", &config, None).unwrap();
                let input = randn::<f32>(&[1, 3, 224, 224]).unwrap(); // Smaller batch for memory
                let output = model.forward(&input).unwrap();
                assert_eq!(output.shape().dims(), &[1, 10]);
                assert!(metadata.num_parameters > 0);
            }
            "transformer_classifier" => {
                let mut extra_params = HashMap::new();
                extra_params.insert("seq_len".to_string(), 64);
                extra_params.insert("d_model".to_string(), 128);

                let (model, metadata) =
                    ModelZoo::create_model("transformer_classifier", &config, Some(extra_params))
                        .unwrap();
                let input = randn::<f32>(&[2, 64]).unwrap();
                let output = model.forward(&input).unwrap();
                assert_eq!(output.shape().dims(), &[2, 10]);
                assert!(metadata.num_parameters > 0);
            }
            "autoencoder" => {
                let mut extra_params = HashMap::new();
                extra_params.insert("input_dim".to_string(), 256);
                extra_params.insert("latent_dim".to_string(), 32);

                let (model, metadata) =
                    ModelZoo::create_model("autoencoder", &config, Some(extra_params)).unwrap();
                let input = randn::<f32>(&[2, 256]).unwrap();
                let output = model.forward(&input).unwrap();
                assert_eq!(output.shape().dims(), &[2, 256]);
                assert!(metadata.num_parameters > 0);
            }
            _ => {
                panic!("Unknown model in zoo: {}", model_name);
            }
        }
    }
}

/// Test model configuration variations
#[test]
fn test_model_configurations() {
    // Test with different number of classes
    let configs = vec![
        ModelConfig {
            num_classes: 2,
            ..ModelConfig::default()
        },
        ModelConfig {
            num_classes: 100,
            ..ModelConfig::default()
        },
        ModelConfig {
            num_classes: 1000,
            ..ModelConfig::default()
        },
    ];

    for (i, config) in configs.iter().enumerate() {
        let (model, _) = ModelZoo::mnist_mlp(config).unwrap();
        let input = randn::<f32>(&[1, 784]).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(
            output.shape().dims()[1],
            config.num_classes,
            "Config {} failed",
            i
        );
    }

    // Test with different dropout rates
    let dropout_configs = vec![
        ModelConfig {
            dropout: 0.0,
            ..ModelConfig::default()
        },
        ModelConfig {
            dropout: 0.5,
            ..ModelConfig::default()
        },
        ModelConfig {
            dropout: 0.9,
            ..ModelConfig::default()
        },
    ];

    for config in &dropout_configs {
        let (model, _) = ModelZoo::cifar10_cnn(config).unwrap();
        let input = randn::<f32>(&[1, 3, 32, 32]).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[1, config.num_classes]);
    }
}

/// Test training-like workflow with model zoo models
#[test]
fn test_training_workflow() {
    let config = ModelConfig {
        num_classes: 5,
        dropout: 0.2,
        ..ModelConfig::default()
    };

    let (model, _) = ModelZoo::mnist_mlp(&config).unwrap();

    // Simulate a training step
    let batch_size = 4;
    let input = randn::<f32>(&[batch_size, 784]).unwrap();
    let target = tensor_1d(&[0i64, 1, 2, 3]).unwrap(); // Class labels

    // Forward pass
    let logits = model.forward(&input).unwrap();
    assert_eq!(logits.shape().dims(), &[batch_size, 5]);

    // Apply softmax to get probabilities
    let probs = softmax(&logits, Some(1)).unwrap();
    let prob_data = probs.to_vec().unwrap();

    // Check that probabilities sum to 1 for each sample
    for batch in 0..batch_size {
        let mut batch_sum = 0.0;
        for class in 0..5 {
            batch_sum += prob_data[batch * 5 + class];
        }
        assert!(
            (batch_sum - 1.0).abs() < 1e-4,
            "Probabilities don't sum to 1: {}",
            batch_sum
        );
    }
}

/// Test model metadata accuracy
#[test]
fn test_model_metadata() {
    let config = ModelConfig::default();

    // Test LeNet-5 metadata
    let (_, metadata) = ModelZoo::lenet5(&config).unwrap();
    assert_eq!(metadata.name, "LeNet-5");
    assert_eq!(metadata.input_shape, vec![1, 32, 32]);
    assert!(metadata.num_parameters > 50000 && metadata.num_parameters < 100000);
    assert!(metadata.model_size_mb > 0.0);

    // Test CIFAR-10 CNN metadata
    let (_, metadata) = ModelZoo::cifar10_cnn(&config).unwrap();
    assert_eq!(metadata.name, "CIFAR-10 CNN");
    assert_eq!(metadata.input_shape, vec![3, 32, 32]);
    assert!(metadata.num_parameters > 100000);

    // Test autoencoder metadata
    let (_, metadata) = ModelZoo::autoencoder(1000, 50).unwrap();
    assert_eq!(metadata.name, "Autoencoder");
    assert_eq!(metadata.input_shape, vec![1000]);

    // Model size should be reasonable
    assert!(metadata.model_size_mb > 0.0 && metadata.model_size_mb < 100.0);
}

/// Test batch size scaling for different models
#[test]
fn test_batch_scaling() {
    let config = ModelConfig {
        num_classes: 10,
        ..ModelConfig::default()
    };
    let batch_sizes = vec![1, 4, 16];

    // Test MNIST MLP scaling
    let (model, _) = ModelZoo::mnist_mlp(&config).unwrap();
    for &batch_size in &batch_sizes {
        let input = randn::<f32>(&[batch_size, 784]).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[batch_size, 10]);
    }

    // Test CIFAR-10 CNN scaling
    let (model, _) = ModelZoo::cifar10_cnn(&config).unwrap();
    for &batch_size in &batch_sizes {
        let input = randn::<f32>(&[batch_size, 3, 32, 32]).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[batch_size, 10]);
    }
}

/// Test autoencoder reconstruction quality
#[test]
fn test_autoencoder_reconstruction() {
    let input_dim = 100;
    let latent_dim = 20;

    let (model, metadata) = ModelZoo::autoencoder(input_dim, latent_dim).unwrap();
    assert_eq!(metadata.input_shape, vec![input_dim]);

    // Test with a simple pattern
    let mut input_data = vec![0.0f32; input_dim];
    // Create a simple pattern (first half 1s, second half 0s)
    for i in 0..input_dim / 2 {
        input_data[i] = 1.0;
    }

    let input = tensor_2d(&[&input_data]).unwrap();
    let reconstructed = model.forward(&input).unwrap();

    // Check output shape
    assert_eq!(reconstructed.shape().dims(), &[1, input_dim]);

    // Check that output is in valid range [0, 1]
    let output_data = reconstructed.to_vec().unwrap();
    for &val in &output_data {
        assert!(
            val >= 0.0 && val <= 1.0,
            "Autoencoder output out of range: {}",
            val
        );
    }

    // The reconstruction might not be perfect, but it should be reasonable
    // (we can't expect perfect reconstruction without training)
}

/// Test transformer classifier with different sequence lengths
#[test]
fn test_transformer_variations() {
    let config = ModelConfig {
        num_classes: 3,
        ..ModelConfig::default()
    };

    let test_cases = vec![
        (32, 64),   // Small sequence, small model
        (128, 128), // Medium sequence, medium model
        (256, 256), // Large sequence, large model
    ];

    for (seq_len, d_model) in test_cases {
        let (model, metadata) =
            ModelZoo::transformer_classifier(&config, seq_len, d_model).unwrap();

        assert_eq!(metadata.input_shape, vec![seq_len]);
        assert!(metadata.num_parameters > 0);

        // Test forward pass
        let input = randn::<f32>(&[2, seq_len]).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape().dims(), &[2, 3]);
    }
}

/// Test model creation with invalid parameters
#[test]
fn test_invalid_model_creation() {
    let config = ModelConfig::default();

    // Test invalid model name
    let result = ModelZoo::create_model("nonexistent_model", &config, None);
    assert!(result.is_err());

    // Test invalid extra parameters (should not crash, should use defaults)
    let mut invalid_params = HashMap::new();
    invalid_params.insert("invalid_param".to_string(), 999);

    let result = ModelZoo::create_model("transformer_classifier", &config, Some(invalid_params));
    assert!(result.is_ok()); // Should work with default parameters
}

/// Test pretrained weights functionality (placeholder tests)
#[test]
fn test_pretrained_weights() {
    let weights = PretrainedWeights::new();

    // Test that no pretrained weights are currently available
    assert!(!weights.is_available("mnist_mlp"));
    assert!(!weights.is_available("lenet5"));
    assert!(!weights.is_available("nonexistent_model"));

    // Test that loading/saving returns appropriate errors
    let config = ModelConfig::default();
    let (mut model, _) = ModelZoo::mnist_mlp(&config).unwrap();

    let load_result = weights.load_weights(model.as_mut(), "mnist_mlp");
    assert!(load_result.is_err());

    let save_result = weights.save_weights(model.as_ref(), "mnist_mlp", "test_weights.pth");
    assert!(save_result.is_err());
}

/// Test that models work with loss functions
#[test]
fn test_models_with_loss_functions() {
    let config = ModelConfig {
        num_classes: 3,
        ..ModelConfig::default()
    };
    let (model, _) = ModelZoo::mnist_mlp(&config).unwrap();

    let input = randn(&[2, 784]).unwrap();
    let logits = model.forward(&input).unwrap();

    // Create targets
    let target_probs = tensor_2d(&[&[1.0, 0.0, 0.0], &[0.0, 1.0, 0.0]]).unwrap();

    // Test with different loss functions
    let mse_loss_result = mse_loss(&logits, &target_probs, "mean").unwrap();
    assert_eq!(mse_loss_result.shape().dims(), &[0usize; 0]); // Scalar loss

    // Apply sigmoid for binary cross entropy
    let sigmoid_logits = sigmoid(&logits).unwrap();
    let bce_loss_result =
        binary_cross_entropy(&sigmoid_logits, &target_probs, None, "mean").unwrap();
    assert_eq!(bce_loss_result.shape().dims(), &[0usize; 0]); // Scalar loss

    // Both losses should be positive
    let mse_data = mse_loss_result.to_vec().unwrap();
    let bce_data = bce_loss_result.to_vec().unwrap();

    assert!(mse_data[0] >= 0.0, "MSE loss should be non-negative");
    assert!(bce_data[0] >= 0.0, "BCE loss should be non-negative");
}

/// Performance test for model zoo models
#[test]
fn test_model_performance() {
    use std::time::Instant;

    let config = ModelConfig {
        num_classes: 10,
        ..ModelConfig::default()
    };

    // Test that models can handle reasonably sized inputs in reasonable time
    let (model, _) = ModelZoo::cifar10_cnn(&config).unwrap();
    let input = randn::<f32>(&[8, 3, 32, 32]).unwrap(); // Batch of 8 images

    let start = Instant::now();
    let output = model.forward(&input).unwrap();
    let duration = start.elapsed();

    assert_eq!(output.shape().dims(), &[8, 10]);
    assert!(
        duration.as_millis() < 5000,
        "Model should complete forward pass within 5 seconds"
    );
}
