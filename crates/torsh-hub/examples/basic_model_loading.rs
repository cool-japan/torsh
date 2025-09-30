//! Basic Model Loading Example
//!
//! This example demonstrates how to load models from various sources using ToRSh Hub.

use torsh_core::error::Result;
use torsh_hub::*;

fn main() -> Result<()> {
    println!("=== ToRSh Hub Basic Model Loading Example ===\n");

    // Example 1: Load a model from GitHub repository
    println!("1. Loading model from GitHub repository...");
    match load("pytorch/vision", "resnet18", true, None) {
        Ok(model) => {
            println!("✓ Successfully loaded ResNet-18 from pytorch/vision");
            println!("  Model type: {}", std::any::type_name_of_val(&*model));
        }
        Err(e) => println!("✗ Failed to load model: {}", e),
    }

    // Example 2: Load model with custom configuration
    println!("\n2. Loading model with custom configuration...");
    let config = HubConfig {
        cache_dir: std::env::temp_dir().join("torsh_hub_examples"),
        verbose: true,
        force_reload: false,
        timeout_seconds: 120,
        ..Default::default()
    };

    match load(
        "huggingface/transformers",
        "bert-base-uncased",
        true,
        Some(config),
    ) {
        Ok(model) => {
            println!("✓ Successfully loaded BERT with custom config");

            // Get model parameters
            let params = model.parameters();
            println!("  Number of parameters: {}", params.len());
        }
        Err(e) => println!("✗ Failed to load model: {}", e),
    }

    // Example 3: Load ONNX model
    println!("\n3. Loading ONNX model...");

    // First, let's check if we have an example ONNX model
    let onnx_path = "examples/models/resnet18.onnx";
    if std::path::Path::new(onnx_path).exists() {
        match load_onnx_model(onnx_path, None) {
            Ok(model) => {
                println!("✓ Successfully loaded ONNX model");

                // Model is successfully loaded as a Box<dyn Module>
                println!("  Model loaded successfully as ToRSh Module");

                // For detailed metadata, you would need to access the wrapper directly
                // This demonstrates the model is working as a ToRSh Module
            }
            Err(e) => println!("✗ Failed to load ONNX model: {}", e),
        }
    } else {
        println!(
            "ℹ ONNX model file not found at {}, skipping example",
            onnx_path
        );
    }

    // Example 4: Load TensorFlow model (if feature is enabled)
    #[cfg(feature = "tensorflow")]
    {
        println!("\n4. Loading TensorFlow model...");
        let tf_model_path = "examples/models/saved_model";

        if std::path::Path::new(tf_model_path).exists() {
            use torsh_hub::tensorflow::{TfConfig, TfModel, TfToTorshWrapper};
            let config = TfConfig::default();
            let tags = vec!["serve"];
            match TfModel::from_saved_model(tf_model_path, &tags, Some(config)) {
                Ok(tf_model) => {
                    let wrapper = TfToTorshWrapper::new(tf_model);
                    println!("✓ Successfully loaded TensorFlow model");

                    // Display model information
                    let metadata = wrapper.metadata();
                    println!("  Model metadata: {:?}", metadata.model_name);
                }
                Err(e) => println!("✗ Failed to load TensorFlow model: {}", e),
            }
        } else {
            println!(
                "ℹ TensorFlow model directory not found at {}, skipping example",
                tf_model_path
            );
        }
    }

    // Example 5: List available models in a repository
    println!("\n5. Listing available models in repository...");
    match list("pytorch/vision", None) {
        Ok(models) => {
            println!("✓ Available models in pytorch/vision:");
            for (i, model) in models.iter().enumerate() {
                println!("  {}. {}", i + 1, model);
            }
        }
        Err(e) => println!("✗ Failed to list models: {}", e),
    }

    // Example 6: Get help for a specific model
    println!("\n6. Getting help for a specific model...");
    match help("pytorch/vision", "resnet18", None) {
        Ok(doc) => {
            println!("✓ Documentation for resnet18:");
            println!("{}", doc);
        }
        Err(e) => println!("✗ Failed to get model help: {}", e),
    }

    // Example 7: Load state dict from URL
    println!("\n7. Loading state dict from URL...");
    let state_dict_url = "https://download.pytorch.org/models/resnet18-5c106cde.pth";

    match load_state_dict_from_url(state_dict_url, None, None, true) {
        Ok(state_dict) => {
            println!("✓ Successfully loaded state dict from URL");
            println!("  Number of parameters: {}", state_dict.len());

            // Display some parameter names
            let param_names: Vec<&String> = state_dict.keys().take(5).collect();
            println!("  First 5 parameters: {:?}", param_names);
        }
        Err(e) => println!("✗ Failed to load state dict: {}", e),
    }

    // Example 8: Authentication (optional)
    println!("\n8. Authentication status...");
    if is_authenticated() {
        println!("✓ {}", auth_status());
    } else {
        println!("ℹ Not authenticated. To access private models, set your token:");
        println!("  export TORSH_HUB_TOKEN=your_token_here");
        println!("  or use: torsh_hub::set_auth_token(\"your_token\")?;");
    }

    // Example 9: Configure hub directory
    println!("\n9. Hub directory configuration...");
    let current_dir = get_dir();
    println!("✓ Current hub directory: {}", current_dir.display());

    // Set custom directory (optional)
    let custom_dir = std::env::temp_dir().join("custom_torsh_hub");
    set_dir(&custom_dir)?;
    println!("✓ Set custom hub directory: {}", custom_dir.display());

    // Reset to default
    std::env::remove_var("TORSH_HUB_DIR");
    let default_dir = get_dir();
    println!("✓ Reset to default directory: {}", default_dir.display());

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

// Helper function to demonstrate model usage
fn demonstrate_model_inference(model: &dyn torsh_nn::Module) -> Result<()> {
    use torsh_tensor::Tensor;

    println!("Demonstrating model inference...");

    // Create dummy input (adjust shape based on your model)
    use torsh_tensor::creation::randn;
    let input = randn(&[1, 3, 224, 224])?; // Common input shape for vision models

    // Run inference
    let output = model.forward(&input)?;

    println!("✓ Input shape: {:?}", input.shape());
    println!("✓ Output shape: {:?}", output.shape());

    Ok(())
}

// Helper function to save and load a model locally
fn save_and_load_model_example() -> Result<()> {
    use std::collections::HashMap;
    use torsh_nn::prelude::{Linear, Module};

    println!("Creating and saving a simple model...");

    // Create a simple model
    let model = Linear::new(10, 5, true);

    // Get state dict
    let state_dict = model.state_dict();

    // Save to file (you would implement this based on your serialization format)
    let save_path = std::env::temp_dir().join("example_model.json");
    save_state_dict(&state_dict, &save_path)?;
    println!("✓ Model saved to: {}", save_path.display());

    // Load from file
    let loaded_state_dict = load_state_dict(&save_path)?;
    println!("✓ Model loaded from file");

    // Create new model and load state
    let mut new_model = Linear::new(10, 5, true);
    new_model.load_state_dict(&loaded_state_dict, true)?;
    println!("✓ State dict loaded into new model");

    Ok(())
}

// Placeholder implementations - replace with actual serialization
fn save_state_dict(
    state_dict: &std::collections::HashMap<String, torsh_tensor::Tensor<f32>>,
    path: &std::path::Path,
) -> Result<()> {
    // This is a placeholder - implement actual serialization
    std::fs::write(path, format!("{{\"tensors\": {}}}", state_dict.len()))?;
    Ok(())
}

fn load_state_dict(
    path: &std::path::Path,
) -> Result<std::collections::HashMap<String, torsh_tensor::Tensor<f32>>> {
    // This is a placeholder - implement actual deserialization
    let _content = std::fs::read_to_string(path)?;
    Ok(std::collections::HashMap::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hub_config_creation() {
        let config = HubConfig::default();
        assert!(config.cache_dir.exists() || config.cache_dir.to_string_lossy().contains("torsh"));
        assert!(!config.force_reload);
        assert!(config.verbose);
    }

    #[test]
    fn test_directory_operations() {
        let temp_dir = std::env::temp_dir().join("test_torsh_hub");

        // Test setting custom directory
        set_dir(&temp_dir).unwrap();
        let current = get_dir();
        assert_eq!(current, temp_dir);

        // Clean up
        std::env::remove_var("TORSH_HUB_DIR");
    }

    #[test]
    fn test_authentication_status() {
        // Test without token
        std::env::remove_var("TORSH_HUB_TOKEN");
        assert!(!is_authenticated());

        // Test with token
        std::env::set_var("TORSH_HUB_TOKEN", "test_token");
        assert!(is_authenticated());

        let status = auth_status();
        assert!(status.contains("test"));

        // Clean up
        std::env::remove_var("TORSH_HUB_TOKEN");
    }
}
