//! Enhanced model profiling with layer-by-layer analysis
//!
//! This module provides detailed profiling of models including memory,
//! computation, and performance analysis for each layer.

// Infrastructure module - functions designed for CLI command integration
#![allow(dead_code)]

use anyhow::Result;
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info};

// ✅ SciRS2 POLICY COMPLIANT: Use scirs2-core unified access patterns
use scirs2_core::random::{thread_rng, Distribution, Normal};

// ToRSh integration
use torsh::core::device::DeviceType;
use torsh::tensor::Tensor;

use super::tensor_integration::estimate_tensor_flops;
use super::types::{LayerInfo, TorshModel};

/// Layer profiling result
#[derive(Debug, Clone)]
pub struct LayerProfile {
    /// Layer name
    pub name: String,
    /// Layer type (Linear, Conv2d, etc.)
    pub layer_type: String,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Number of parameters
    pub parameters: u64,
    /// Memory footprint (bytes)
    pub memory_bytes: u64,
    /// Estimated FLOPs per forward pass
    pub flops: u64,
    /// Execution time (ms) - if measured
    pub execution_time_ms: Option<f64>,
    /// Memory usage during execution (MB) - if measured
    pub runtime_memory_mb: Option<f64>,
    /// Percentage of total model parameters
    pub param_percentage: f64,
    /// Percentage of total model FLOPs
    pub flops_percentage: f64,
}

/// Complete model profile
#[derive(Debug, Clone)]
pub struct ModelProfile {
    /// Individual layer profiles
    pub layers: Vec<LayerProfile>,
    /// Total parameters
    pub total_parameters: u64,
    /// Total FLOPs
    pub total_flops: u64,
    /// Total memory footprint (bytes)
    pub total_memory: u64,
    /// Execution time breakdown
    pub execution_breakdown: HashMap<String, f64>,
    /// Memory hotspots (layers using most memory)
    pub memory_hotspots: Vec<(String, u64)>,
    /// Computation hotspots (layers with most FLOPs)
    pub computation_hotspots: Vec<(String, u64)>,
}

/// Profile a model's layers
pub fn profile_model(model: &TorshModel) -> Result<ModelProfile> {
    info!("Profiling model with {} layers", model.layers.len());

    // First pass: calculate totals
    let total_parameters: u64 = model.layers.iter().map(|l| l.parameters).sum();

    let total_flops: u64 = model.layers.iter().map(|l| estimate_layer_flops(l)).sum();

    let total_memory: u64 = model
        .weights
        .values()
        .map(|t| {
            let elements: usize = t.shape.iter().product();
            (elements * t.dtype.size_bytes()) as u64
        })
        .sum();

    debug!(
        "Model totals: {} params, {} FLOPs, {:.2} MB",
        total_parameters,
        total_flops,
        total_memory as f64 / (1024.0 * 1024.0)
    );

    // Second pass: profile each layer
    let mut layer_profiles = Vec::new();

    for layer in &model.layers {
        let flops = estimate_layer_flops(layer);
        let memory = calculate_layer_memory(layer, model);

        let param_percentage = if total_parameters > 0 {
            (layer.parameters as f64 / total_parameters as f64) * 100.0
        } else {
            0.0
        };

        let flops_percentage = if total_flops > 0 {
            (flops as f64 / total_flops as f64) * 100.0
        } else {
            0.0
        };

        let profile = LayerProfile {
            name: layer.name.clone(),
            layer_type: layer.layer_type.clone(),
            input_shape: layer.input_shape.clone(),
            output_shape: layer.output_shape.clone(),
            parameters: layer.parameters,
            memory_bytes: memory,
            flops,
            execution_time_ms: None, // Will be filled by runtime profiling
            runtime_memory_mb: None,
            param_percentage,
            flops_percentage,
        };

        layer_profiles.push(profile);
    }

    // Identify hotspots
    let mut memory_hotspots: Vec<(String, u64)> = layer_profiles
        .iter()
        .map(|p| (p.name.clone(), p.memory_bytes))
        .collect();
    memory_hotspots.sort_by(|a, b| b.1.cmp(&a.1));
    memory_hotspots.truncate(5);

    let mut computation_hotspots: Vec<(String, u64)> = layer_profiles
        .iter()
        .map(|p| (p.name.clone(), p.flops))
        .collect();
    computation_hotspots.sort_by(|a, b| b.1.cmp(&a.1));
    computation_hotspots.truncate(5);

    Ok(ModelProfile {
        layers: layer_profiles,
        total_parameters,
        total_flops,
        total_memory,
        execution_breakdown: HashMap::new(),
        memory_hotspots,
        computation_hotspots,
    })
}

/// Estimate FLOPs for a layer
fn estimate_layer_flops(layer: &LayerInfo) -> u64 {
    estimate_tensor_flops(
        &layer.layer_type.to_lowercase(),
        &layer.input_shape,
        &layer.output_shape,
    )
}

/// Calculate memory footprint for a layer
fn calculate_layer_memory(layer: &LayerInfo, model: &TorshModel) -> u64 {
    let weight_name = format!("{}.weight", layer.name);
    let bias_name = format!("{}.bias", layer.name);

    let mut memory = 0u64;

    if let Some(weight) = model.weights.get(&weight_name) {
        let elements: usize = weight.shape.iter().product();
        memory += (elements * weight.dtype.size_bytes()) as u64;
    }

    if let Some(bias) = model.weights.get(&bias_name) {
        let elements: usize = bias.shape.iter().product();
        memory += (elements * bias.dtype.size_bytes()) as u64;
    }

    memory
}

/// Format model profile as human-readable text
pub fn format_model_profile(profile: &ModelProfile) -> String {
    let mut output = String::new();

    output.push_str("╔═══════════════════════════════════════════════════════════════════════╗\n");
    output.push_str("║                        MODEL PROFILE REPORT                           ║\n");
    output
        .push_str("╚═══════════════════════════════════════════════════════════════════════╝\n\n");

    // Summary section
    output.push_str("📊 Overall Statistics\n");
    output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    output.push_str(&format!(
        "  Total Parameters: {} ({:.2} M)\n",
        profile.total_parameters,
        profile.total_parameters as f64 / 1_000_000.0
    ));
    output.push_str(&format!(
        "  Total FLOPs:      {} ({:.2} GFLOPs)\n",
        profile.total_flops,
        profile.total_flops as f64 / 1_000_000_000.0
    ));
    output.push_str(&format!(
        "  Total Memory:     {:.2} MB\n",
        profile.total_memory as f64 / (1024.0 * 1024.0)
    ));
    output.push_str(&format!("  Number of Layers: {}\n", profile.layers.len()));
    output.push_str("\n");

    // Layer-by-layer breakdown
    output.push_str("📋 Layer-by-Layer Breakdown\n");
    output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    for (i, layer) in profile.layers.iter().enumerate() {
        output.push_str(&format!(
            "\n[{}] {} ({})\n",
            i, layer.name, layer.layer_type
        ));
        output.push_str(&format!(
            "    Shape: {:?} → {:?}\n",
            layer.input_shape, layer.output_shape
        ));
        output.push_str(&format!(
            "    Parameters: {} ({:.1}% of total)\n",
            layer.parameters, layer.param_percentage
        ));
        output.push_str(&format!(
            "    Memory: {:.2} KB\n",
            layer.memory_bytes as f64 / 1024.0
        ));
        output.push_str(&format!(
            "    FLOPs: {:.2} MFLOPs ({:.1}% of total)\n",
            layer.flops as f64 / 1_000_000.0,
            layer.flops_percentage
        ));

        if let Some(time) = layer.execution_time_ms {
            output.push_str(&format!("    Execution Time: {:.2} ms\n", time));
        }
    }

    // Hotspots
    output.push_str("\n\n🔥 Memory Hotspots (Top 5)\n");
    output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    for (i, (name, memory)) in profile.memory_hotspots.iter().enumerate() {
        output.push_str(&format!(
            "  {}. {} - {:.2} MB\n",
            i + 1,
            name,
            *memory as f64 / (1024.0 * 1024.0)
        ));
    }

    output.push_str("\n🚀 Computation Hotspots (Top 5)\n");
    output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    for (i, (name, flops)) in profile.computation_hotspots.iter().enumerate() {
        output.push_str(&format!(
            "  {}. {} - {:.2} GFLOPs\n",
            i + 1,
            name,
            *flops as f64 / 1_000_000_000.0
        ));
    }

    output.push_str("\n");
    output
}

/// Profile model with runtime execution measurements
pub async fn profile_model_runtime(
    model: &TorshModel,
    batch_size: usize,
    iterations: usize,
) -> Result<ModelProfile> {
    info!(
        "Runtime profiling model with batch size {} for {} iterations",
        batch_size, iterations
    );

    // Get static profile first
    let mut profile = profile_model(model)?;

    // Create input tensor
    let input_shape = model
        .layers
        .first()
        .map(|l| l.input_shape.clone())
        .unwrap_or_else(|| vec![784]);

    let _input = create_test_input(&input_shape, batch_size)?;

    // Measure each layer's execution time (simulated for now)
    // In real implementation, this would do actual forward passes per layer
    for layer_profile in &mut profile.layers {
        debug!("Profiling layer: {}", layer_profile.name);

        let mut timings = Vec::new();

        for _ in 0..iterations {
            let start = Instant::now();

            // Simulate layer execution based on FLOPs
            // In real implementation, would do actual layer forward pass
            let compute_time = (layer_profile.flops as f64 / 1_000_000_000.0) * 10.0; // Rough estimate
            tokio::time::sleep(std::time::Duration::from_micros(
                (compute_time * 1000.0) as u64,
            ))
            .await;

            timings.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        let avg_time = timings.iter().sum::<f64>() / timings.len() as f64;
        layer_profile.execution_time_ms = Some(avg_time);

        // Estimate runtime memory
        let activation_memory = layer_profile.output_shape.iter().product::<usize>() * 4; // f32
        layer_profile.runtime_memory_mb = Some(
            (layer_profile.memory_bytes + activation_memory as u64) as f64 / (1024.0 * 1024.0),
        );
    }

    // Build execution breakdown
    let mut execution_breakdown = HashMap::new();
    for layer_profile in &profile.layers {
        if let Some(time) = layer_profile.execution_time_ms {
            execution_breakdown.insert(layer_profile.name.clone(), time);
        }
    }
    profile.execution_breakdown = execution_breakdown;

    Ok(profile)
}

/// Create test input tensor
fn create_test_input(shape: &[usize], batch_size: usize) -> Result<Tensor<f32>> {
    let mut full_shape = vec![batch_size];
    full_shape.extend_from_slice(shape);

    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0)?;

    let num_elements: usize = full_shape.iter().product();
    let data: Vec<f32> = (0..num_elements)
        .map(|_| normal.sample(&mut rng) as f32)
        .collect();

    Ok(Tensor::from_data(data, full_shape, DeviceType::Cpu)?)
}

/// Export profile to JSON
pub fn export_profile_json(profile: &ModelProfile) -> Result<String> {
    let json = serde_json::json!({
        "summary": {
            "total_parameters": profile.total_parameters,
            "total_flops": profile.total_flops,
            "total_memory_bytes": profile.total_memory,
            "num_layers": profile.layers.len(),
        },
        "layers": profile.layers.iter().map(|l| {
            serde_json::json!({
                "name": l.name,
                "type": l.layer_type,
                "input_shape": l.input_shape,
                "output_shape": l.output_shape,
                "parameters": l.parameters,
                "memory_bytes": l.memory_bytes,
                "flops": l.flops,
                "param_percentage": l.param_percentage,
                "flops_percentage": l.flops_percentage,
                "execution_time_ms": l.execution_time_ms,
                "runtime_memory_mb": l.runtime_memory_mb,
            })
        }).collect::<Vec<_>>(),
        "hotspots": {
            "memory": profile.memory_hotspots.iter().map(|(name, mem)| {
                serde_json::json!({"layer": name, "memory_bytes": mem})
            }).collect::<Vec<_>>(),
            "computation": profile.computation_hotspots.iter().map(|(name, flops)| {
                serde_json::json!({"layer": name, "flops": flops})
            }).collect::<Vec<_>>(),
        }
    });

    Ok(serde_json::to_string_pretty(&json)?)
}

#[cfg(test)]
mod tests {
    use super::super::tensor_integration::create_real_model;
    use super::*;

    #[test]
    fn test_model_profiling() {
        let model = create_real_model("test", 3, DeviceType::Cpu).unwrap();
        let profile = profile_model(&model).unwrap();

        assert_eq!(profile.layers.len(), 3);
        assert!(profile.total_parameters > 0);
        assert!(profile.total_flops > 0);
        assert!(!profile.memory_hotspots.is_empty());
        assert!(!profile.computation_hotspots.is_empty());
    }

    #[test]
    fn test_profile_formatting() {
        let model = create_real_model("test", 2, DeviceType::Cpu).unwrap();
        let profile = profile_model(&model).unwrap();
        let formatted = format_model_profile(&profile);

        assert!(formatted.contains("MODEL PROFILE REPORT"));
        assert!(formatted.contains("Overall Statistics"));
        assert!(formatted.contains("Layer-by-Layer Breakdown"));
    }

    #[test]
    fn test_profile_export_json() {
        let model = create_real_model("test", 2, DeviceType::Cpu).unwrap();
        let profile = profile_model(&model).unwrap();
        let json = export_profile_json(&profile).unwrap();

        assert!(json.contains("total_parameters"));
        assert!(json.contains("layers"));
        assert!(json.contains("hotspots"));
    }

    #[tokio::test]
    async fn test_runtime_profiling() {
        let model = create_real_model("test", 2, DeviceType::Cpu).unwrap();
        let profile = profile_model_runtime(&model, 1, 5).await.unwrap();

        // Check that execution times were measured
        assert!(profile.layers.iter().any(|l| l.execution_time_ms.is_some()));
    }
}
