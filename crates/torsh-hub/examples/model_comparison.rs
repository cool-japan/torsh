//! Example demonstrating model comparison and ensemble creation utilities
//!
//! This example shows how to use the new model_ops module to:
//! - Compare two model versions
//! - Analyze differences in parameters
//! - Create model ensembles
//! - Estimate memory footprints

use std::collections::HashMap;
use torsh_core::DeviceType;
use torsh_hub::{
    compare_models, create_model_ensemble, ComparisonOptions, EnsembleConfig, VotingStrategy,
};
use torsh_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh Hub Model Operations Example ===\n");

    // Create two models with similar but different parameters
    let model_v1 = create_sample_model(1.0)?;
    let model_v2 = create_sample_model(1.1)?;

    println!("1. Comparing Model Versions");
    println!("---------------------------");

    // Compare the models
    let diff = compare_models(&model_v1, &model_v2, Some(ComparisonOptions::default()))?;

    println!("Common parameters: {}", diff.common_parameters.len());
    println!("Parameters only in v1: {}", diff.only_in_first.len());
    println!("Parameters only in v2: {}", diff.only_in_second.len());
    println!("Shape differences: {}", diff.shape_differences.len());
    println!(
        "Value differences analyzed: {}",
        diff.value_differences.len()
    );

    // Display detailed value differences
    if !diff.value_differences.is_empty() {
        println!("\nDetailed Value Differences:");
        for vd in &diff.value_differences {
            println!("  {}", vd.parameter_name);
            println!("    Mean abs diff: {:.6}", vd.mean_absolute_diff);
            println!("    Max abs diff: {:.6}", vd.max_absolute_diff);
            println!("    Relative diff: {:.2}%", vd.relative_diff_percent);
            println!("    Cosine similarity: {:.6}", vd.cosine_similarity);
        }
    }

    // Display memory footprints
    println!("\nMemory Footprints:");
    println!("  Model v1: {} bytes", diff.memory_footprints.0);
    println!("  Model v2: {} bytes", diff.memory_footprints.1);

    println!("\n2. Creating Model Ensemble");
    println!("---------------------------");

    // Create an ensemble of the two models
    let models = vec![model_v1.clone(), model_v2.clone()];

    let ensemble_config = EnsembleConfig {
        weights: vec![0.7, 0.3], // 70% v1, 30% v2
        normalize_weights: true,
        voting_strategy: VotingStrategy::WeightedAverage,
    };

    let ensemble = create_model_ensemble(&models, Some(ensemble_config))?;

    println!("Created ensemble with {} parameters", ensemble.len());
    println!("Ensemble combines v1 (70%) and v2 (30%)");

    // Verify ensemble parameter
    if let Some(weight) = ensemble.get("layer1.weight") {
        let data = weight.to_vec()?;
        println!("\nExample ensemble parameter (layer1.weight):");
        println!("  First value: {:.4}", data[0]);
        println!("  Shape: {:?}", weight.shape().dims());
    }

    println!("\n3. Creating Balanced Ensemble");
    println!("-------------------------------");

    // Create a balanced ensemble (equal weights)
    let balanced_config = EnsembleConfig {
        weights: vec![0.5, 0.5],
        normalize_weights: true,
        voting_strategy: VotingStrategy::Average,
    };

    let balanced_ensemble = create_model_ensemble(&models, Some(balanced_config))?;
    println!("Created balanced ensemble with equal weights");

    // Compare original and ensemble parameters
    if let (Some(orig), Some(ens)) = (
        model_v1.get("layer1.weight"),
        balanced_ensemble.get("layer1.weight"),
    ) {
        let orig_data = orig.to_vec()?;
        let ens_data = ens.to_vec()?;

        println!("\nParameter comparison (layer1.weight):");
        println!("  Original v1 first value: {:.4}", orig_data[0]);
        println!("  Ensemble first value: {:.4}", ens_data[0]);
    }

    println!("\n=== Example Complete ===");

    Ok(())
}

/// Create a sample model with simple parameters
fn create_sample_model(
    scale: f32,
) -> Result<HashMap<String, Tensor<f32>>, Box<dyn std::error::Error>> {
    let mut model = HashMap::new();

    // Layer 1: 4x4 weight matrix
    let layer1_data: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * scale).collect();
    let layer1_weight = Tensor::from_data(layer1_data, vec![4, 4], DeviceType::Cpu)?;
    model.insert("layer1.weight".to_string(), layer1_weight);

    // Layer 1: bias (4 elements)
    let layer1_bias = Tensor::from_data(
        vec![0.1 * scale, 0.2 * scale, 0.3 * scale, 0.4 * scale],
        vec![4],
        DeviceType::Cpu,
    )?;
    model.insert("layer1.bias".to_string(), layer1_bias);

    // Layer 2: 3x4 weight matrix
    let layer2_data: Vec<f32> = (0..12).map(|i| (i as f32 + 1.0) * scale * 0.5).collect();
    let layer2_weight = Tensor::from_data(layer2_data, vec![3, 4], DeviceType::Cpu)?;
    model.insert("layer2.weight".to_string(), layer2_weight);

    // Layer 2: bias (3 elements)
    let layer2_bias = Tensor::from_data(
        vec![0.5 * scale, 0.6 * scale, 0.7 * scale],
        vec![3],
        DeviceType::Cpu,
    )?;
    model.insert("layer2.bias".to_string(), layer2_bias);

    Ok(model)
}
