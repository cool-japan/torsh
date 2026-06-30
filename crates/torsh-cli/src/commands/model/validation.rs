//! Comprehensive model validation with real inference and gradient checking
//!
//! This module provides tools for validating ToRSh models including:
//! - Real forward pass inference
//! - Gradient checking for correctness
//! - Numerical stability analysis
//! - Performance validation

// Infrastructure module - functions designed for CLI command integration
#![allow(dead_code)]

use anyhow::Result;
use tracing::{debug, info, warn};

// ✅ SciRS2 POLICY COMPLIANT: Use scirs2-core unified access patterns
use scirs2_core::random::{thread_rng, Distribution, Uniform};

// ToRSh integration
use torsh::core::device::DeviceType;
use torsh::tensor::Tensor;

use super::types::{LayerInfo, TorshModel};

/// Validation result for a model
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the model passed validation
    pub passed: bool,
    /// Validation accuracy (if applicable)
    pub accuracy: Option<f64>,
    /// Top-5 accuracy (if applicable)
    pub top5_accuracy: Option<f64>,
    /// Number of samples tested
    pub num_samples: usize,
    /// Number of successful inferences
    pub successful_inferences: usize,
    /// Number of failed inferences
    pub failed_inferences: usize,
    /// Average inference time (ms)
    pub avg_inference_time_ms: f64,
    /// Peak memory usage (MB)
    pub peak_memory_mb: f64,
    /// Gradient check results (if performed)
    pub gradient_check_passed: Option<bool>,
    /// Numerical stability score (0-1, higher is better)
    pub numerical_stability: f64,
    /// Validation errors
    pub errors: Vec<String>,
    /// Warnings
    pub warnings: Vec<String>,
}

/// Gradient checking result
#[derive(Debug, Clone)]
pub struct GradientCheckResult {
    /// Whether gradient check passed
    pub passed: bool,
    /// Maximum relative error
    pub max_relative_error: f64,
    /// Average relative error
    pub avg_relative_error: f64,
    /// Number of gradients checked
    pub num_gradients_checked: usize,
    /// Failed gradient locations
    pub failed_locations: Vec<String>,
}

/// Numerical stability analysis result
#[derive(Debug, Clone)]
pub struct StabilityAnalysis {
    /// Presence of NaN values
    pub has_nan: bool,
    /// Presence of Inf values
    pub has_inf: bool,
    /// Very large values (>1e6)
    pub has_large_values: bool,
    /// Very small values (<1e-6)
    pub has_tiny_values: bool,
    /// Gradient magnitude statistics. `None` when they cannot be computed —
    /// a metadata-only model has no autograd graph to derive gradients from.
    pub gradient_magnitude: Option<GradientStatistics>,
    /// Activation statistics
    pub activation_stats: ActivationStatistics,
}

/// Gradient magnitude statistics
#[derive(Debug, Clone)]
pub struct GradientStatistics {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    /// Percentage of gradients near zero (<1e-7)
    pub vanishing_percentage: f64,
    /// Percentage of large gradients (>10)
    pub exploding_percentage: f64,
}

/// Activation statistics
#[derive(Debug, Clone)]
pub struct ActivationStatistics {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    /// Percentage of dead neurons (always output 0)
    pub dead_neurons_percentage: f64,
}

/// Perform comprehensive model validation
pub async fn validate_model(
    model: &TorshModel,
    num_samples: usize,
    check_gradients: bool,
) -> Result<ValidationResult> {
    info!(
        "Validating model with {} samples (gradient check: {})",
        num_samples, check_gradients
    );

    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    // Step 1: Basic structure validation
    if let Err(e) = validate_model_structure(model) {
        errors.push(format!("Model structure validation failed: {}", e));
    }

    // Step 2: Run inference tests
    let (successful, failed, avg_time, peak_memory) =
        run_inference_tests(model, num_samples).await?;

    // Step 3: Gradient checking (if requested)
    let gradient_check_result = if check_gradients {
        match perform_gradient_check(model).await {
            Ok(result) => Some(result.passed),
            Err(e) => {
                warnings.push(format!("Gradient check failed: {}", e));
                None
            }
        }
    } else {
        None
    };

    // Step 4: Numerical stability analysis
    let stability = analyze_numerical_stability(model).await?;
    let numerical_stability = calculate_stability_score(&stability);

    if stability.has_nan {
        errors.push("Model contains NaN values".to_string());
    }
    if stability.has_inf {
        errors.push("Model contains Inf values".to_string());
    }

    if let Some(gradient_magnitude) = &stability.gradient_magnitude {
        if gradient_magnitude.vanishing_percentage > 50.0 {
            warnings.push(format!(
                "High vanishing gradient rate: {:.1}%",
                gradient_magnitude.vanishing_percentage
            ));
        }

        if gradient_magnitude.exploding_percentage > 10.0 {
            warnings.push(format!(
                "High exploding gradient rate: {:.1}%",
                gradient_magnitude.exploding_percentage
            ));
        }
    }

    let passed = errors.is_empty() && successful > 0;

    Ok(ValidationResult {
        passed,
        accuracy: None, // Would be calculated with real dataset
        top5_accuracy: None,
        num_samples,
        successful_inferences: successful,
        failed_inferences: failed,
        avg_inference_time_ms: avg_time,
        peak_memory_mb: peak_memory,
        gradient_check_passed: gradient_check_result,
        numerical_stability,
        errors,
        warnings,
    })
}

/// Validate model structure
fn validate_model_structure(model: &TorshModel) -> Result<()> {
    debug!("Validating model structure");

    // Check layers exist
    if model.layers.is_empty() {
        anyhow::bail!("Model has no layers");
    }

    // Check each layer has valid shapes
    for layer in &model.layers {
        if layer.input_shape.is_empty() {
            anyhow::bail!("Layer {} has empty input shape", layer.name);
        }
        if layer.output_shape.is_empty() {
            anyhow::bail!("Layer {} has empty output shape", layer.name);
        }

        // Verify weight tensor exists for trainable layers
        if layer.trainable {
            let weight_name = format!("{}.weight", layer.name);
            if !model.weights.contains_key(&weight_name) {
                anyhow::bail!("Trainable layer {} missing weight tensor", layer.name);
            }
        }
    }

    // Check layer connectivity (input/output shapes should match)
    for i in 0..model.layers.len() - 1 {
        let current = &model.layers[i];
        let next = &model.layers[i + 1];

        if current.output_shape != next.input_shape {
            warn!(
                "Shape mismatch between layers {} and {}: {:?} != {:?}",
                current.name, next.name, current.output_shape, next.input_shape
            );
        }
    }

    Ok(())
}

/// Run inference tests on random inputs
async fn run_inference_tests(
    model: &TorshModel,
    num_samples: usize,
) -> Result<(usize, usize, f64, f64)> {
    info!("Running {} inference tests", num_samples);

    let input_shape = model
        .layers
        .first()
        .map(|l| l.input_shape.clone())
        .unwrap_or_else(|| vec![784]);

    let mut successful = 0;
    let mut failed = 0;
    let mut total_time = 0.0;
    let mut peak_memory = 0.0f64;

    for i in 0..num_samples {
        let input = create_random_input(&input_shape)?;

        let start = std::time::Instant::now();

        match perform_forward_pass(model, &input).await {
            Ok(output) => {
                successful += 1;
                total_time += start.elapsed().as_secs_f64() * 1000.0;

                // Estimate memory usage
                let memory = estimate_inference_memory(model, &output);
                peak_memory = peak_memory.max(memory);

                debug!(
                    "Inference {}: successful, output shape: {:?}",
                    i,
                    output.shape().dims()
                );
            }
            Err(e) => {
                failed += 1;
                warn!("Inference {} failed: {}", i, e);
            }
        }
    }

    let avg_time = if successful > 0 {
        total_time / successful as f64
    } else {
        0.0
    };

    Ok((successful, failed, avg_time, peak_memory))
}

/// Create random input tensor
fn create_random_input(shape: &[usize]) -> Result<Tensor<f32>> {
    let mut rng = thread_rng();
    let uniform = Uniform::new(-1.0f64, 1.0f64)?;

    let num_elements: usize = shape.iter().product();
    let data: Vec<f32> = (0..num_elements)
        .map(|_| uniform.sample(&mut rng) as f32)
        .collect();

    Ok(Tensor::from_data(data, shape.to_vec(), DeviceType::Cpu)?)
}

/// Perform a real forward pass through the model.
///
/// Delegates to [`super::tensor_integration::forward_pass`], which threads the
/// input through each layer with real `matmul`/activation tensor kernels. This
/// is a genuine computation — not a zero-filled placeholder.
async fn perform_forward_pass(model: &TorshModel, input: &Tensor<f32>) -> Result<Tensor<f32>> {
    debug!("Performing forward pass");
    super::tensor_integration::forward_pass(model, input)
}

/// Estimate FLOPs for a layer
fn estimate_layer_flops(layer: &LayerInfo) -> u64 {
    let input_size: u64 = layer.input_shape.iter().map(|&x| x as u64).product();
    let output_size: u64 = layer.output_shape.iter().map(|&x| x as u64).product();

    match layer.layer_type.as_str() {
        "Linear" | "Dense" => 2 * input_size * output_size,
        "Conv2d" => {
            let kernel_size = 9; // Assume 3x3
            2 * kernel_size * output_size
        }
        "ReLU" | "Sigmoid" | "Tanh" => output_size,
        _ => output_size,
    }
}

/// Estimate memory usage for inference
fn estimate_inference_memory(model: &TorshModel, _output: &Tensor<f32>) -> f64 {
    let param_memory: u64 = model
        .weights
        .values()
        .map(|t| {
            let elements: usize = t.shape.iter().product();
            (elements * t.dtype.size_bytes()) as u64
        })
        .sum();

    let activation_memory: u64 = model
        .layers
        .iter()
        .map(|l| {
            let output_elements: u64 = l.output_shape.iter().map(|&x| x as u64).product();
            output_elements * 4 // f32
        })
        .sum();

    (param_memory + activation_memory) as f64 / (1024.0 * 1024.0)
}

/// Attempt gradient checking.
///
/// A meaningful gradient check compares finite-difference (numerical) gradients
/// with analytical (autograd) gradients for the same parameters. A
/// [`TorshModel`] carries only tensor metadata — it has no autograd-tracked
/// parameters or live computation graph — so analytical gradients cannot be
/// obtained. Rather than compare fabricated random vectors (which would always
/// "pass"), this delegates to the honest primitive in
/// [`super::tensor_integration::gradient_check`], which returns an error
/// describing what is required to gradient-check a model.
async fn perform_gradient_check(model: &TorshModel) -> Result<GradientCheckResult> {
    info!("Performing gradient check");

    let input_shape = model
        .layers
        .first()
        .map(|l| l.input_shape.clone())
        .unwrap_or_else(|| vec![784]);
    let input = create_random_input(&input_shape)?;

    // Errors for metadata-only models; yields a real verdict once
    // autograd-tracked models are supported.
    let passed = super::tensor_integration::gradient_check(model, &input, 1e-5)?;

    Ok(GradientCheckResult {
        passed,
        max_relative_error: 0.0,
        avg_relative_error: 0.0,
        num_gradients_checked: model.weights.len(),
        failed_locations: Vec::new(),
    })
}

/// Analyze numerical stability of the model from a real forward pass.
///
/// A [`TorshModel`] holds only tensor metadata (shapes/dtypes) — it has no
/// trained weight values to inspect — so stability is assessed from the
/// activations of a genuinely-computed forward pass rather than from fabricated
/// samples. Gradient-magnitude statistics require an autograd graph that a
/// metadata-only model does not provide, so they are honestly reported as
/// absent (`None`).
async fn analyze_numerical_stability(model: &TorshModel) -> Result<StabilityAnalysis> {
    info!("Analyzing numerical stability");

    let input_shape = model
        .layers
        .first()
        .map(|l| l.input_shape.clone())
        .unwrap_or_else(|| vec![784]);
    let input = create_random_input(&input_shape)?;
    let output = super::tensor_integration::forward_pass(model, &input)?;
    let values: Vec<f32> = output.to_vec()?;

    let mut has_nan = false;
    let mut has_inf = false;
    let mut has_large_values = false;
    let mut has_tiny_values = false;
    for &val in &values {
        if val.is_nan() {
            has_nan = true;
        }
        if val.is_infinite() {
            has_inf = true;
        }
        if val.abs() > 1e6 {
            has_large_values = true;
        }
        if val != 0.0 && val.abs() < 1e-6 {
            has_tiny_values = true;
        }
    }

    let activation_stats = compute_activation_statistics(&values);

    Ok(StabilityAnalysis {
        has_nan,
        has_inf,
        has_large_values,
        has_tiny_values,
        gradient_magnitude: None,
        activation_stats,
    })
}

/// Compute activation statistics from real activation values.
fn compute_activation_statistics(activations: &[f32]) -> ActivationStatistics {
    if activations.is_empty() {
        return ActivationStatistics {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            dead_neurons_percentage: 0.0,
        };
    }

    let count = activations.len() as f64;
    let mean = activations.iter().map(|&x| x as f64).sum::<f64>() / count;
    let variance = activations
        .iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>()
        / count;
    let std = variance.sqrt();
    let min = activations.iter().copied().fold(f32::INFINITY, f32::min) as f64;
    let max = activations
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max) as f64;

    let dead_count = activations.iter().filter(|&&x| x == 0.0).count();
    let dead_neurons_percentage = (dead_count as f64 / count) * 100.0;

    ActivationStatistics {
        mean,
        std,
        min,
        max,
        dead_neurons_percentage,
    }
}

/// Calculate overall stability score (0-1)
fn calculate_stability_score(analysis: &StabilityAnalysis) -> f64 {
    let mut score = 1.0f64;

    // Penalize for NaN/Inf values
    if analysis.has_nan {
        score -= 0.5;
    }
    if analysis.has_inf {
        score -= 0.5;
    }

    // Penalize for extreme values
    if analysis.has_large_values {
        score -= 0.1;
    }
    if analysis.has_tiny_values {
        score -= 0.05;
    }

    // Penalize for gradient issues (only when gradient statistics are available)
    if let Some(gradient_magnitude) = &analysis.gradient_magnitude {
        if gradient_magnitude.vanishing_percentage > 50.0 {
            score -= 0.2;
        }
        if gradient_magnitude.exploding_percentage > 10.0 {
            score -= 0.2;
        }
    }

    // Penalize for dead neurons
    if analysis.activation_stats.dead_neurons_percentage > 50.0 {
        score -= 0.1;
    }

    score.max(0.0)
}

/// Format validation result as human-readable text
pub fn format_validation_result(result: &ValidationResult) -> String {
    let mut output = String::new();

    output.push_str("╔═══════════════════════════════════════════════════════════════════════╗\n");
    output.push_str("║                     MODEL VALIDATION REPORT                           ║\n");
    output
        .push_str("╚═══════════════════════════════════════════════════════════════════════╝\n\n");

    // Overall status
    let status = if result.passed {
        "✅ PASSED"
    } else {
        "❌ FAILED"
    };
    output.push_str(&format!("Status: {}\n\n", status));

    // Inference results
    output.push_str("📊 Inference Testing\n");
    output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    output.push_str(&format!("  Samples tested:       {}\n", result.num_samples));
    output.push_str(&format!(
        "  Successful:           {}\n",
        result.successful_inferences
    ));
    output.push_str(&format!(
        "  Failed:               {}\n",
        result.failed_inferences
    ));
    output.push_str(&format!(
        "  Avg inference time:   {:.2} ms\n",
        result.avg_inference_time_ms
    ));
    output.push_str(&format!(
        "  Peak memory:          {:.2} MB\n",
        result.peak_memory_mb
    ));

    if let Some(acc) = result.accuracy {
        output.push_str(&format!("  Accuracy:             {:.2}%\n", acc * 100.0));
    }
    if let Some(top5) = result.top5_accuracy {
        output.push_str(&format!("  Top-5 Accuracy:       {:.2}%\n", top5 * 100.0));
    }

    output.push_str("\n");

    // Gradient check results
    if let Some(grad_passed) = result.gradient_check_passed {
        output.push_str("🔍 Gradient Checking\n");
        output
            .push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        output.push_str(&format!(
            "  Status:               {}\n",
            if grad_passed {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            }
        ));
        output.push_str("\n");
    }

    // Numerical stability
    output.push_str("📈 Numerical Stability\n");
    output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    output.push_str(&format!(
        "  Stability score:      {:.2}/1.00\n",
        result.numerical_stability
    ));
    output.push_str("\n");

    // Errors
    if !result.errors.is_empty() {
        output.push_str("❌ Errors\n");
        output
            .push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        for error in &result.errors {
            output.push_str(&format!("  • {}\n", error));
        }
        output.push_str("\n");
    }

    // Warnings
    if !result.warnings.is_empty() {
        output.push_str("⚠️  Warnings\n");
        output
            .push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        for warning in &result.warnings {
            output.push_str(&format!("  • {}\n", warning));
        }
        output.push_str("\n");
    }

    output
}

#[cfg(test)]
mod tests {
    use super::super::tensor_integration::create_real_model;
    use super::*;

    #[tokio::test]
    async fn test_model_validation() {
        let model = create_real_model("test", 3, DeviceType::Cpu)
            .expect("create real model should succeed");
        let result = validate_model(&model, 10, false)
            .await
            .expect("operation should succeed");

        assert!(result.num_samples == 10);
        assert!(result.successful_inferences > 0);
    }

    #[test]
    fn test_structure_validation() {
        let model = create_real_model("test", 2, DeviceType::Cpu)
            .expect("create real model should succeed");
        assert!(validate_model_structure(&model).is_ok());
    }

    #[tokio::test]
    async fn test_gradient_check_is_honest_error() {
        let model = create_real_model("test", 2, DeviceType::Cpu)
            .expect("create real model should succeed");
        // A metadata-only model has no autograd graph, so gradient checking
        // must return an honest error rather than fabricate a passing result.
        assert!(perform_gradient_check(&model).await.is_err());
    }

    #[tokio::test]
    async fn test_stability_analysis() {
        let model = create_real_model("test", 2, DeviceType::Cpu)
            .expect("create real model should succeed");
        let analysis = analyze_numerical_stability(&model)
            .await
            .expect("operation should succeed");

        assert!(!analysis.has_nan);
        assert!(!analysis.has_inf);
    }

    #[test]
    fn test_validation_formatting() {
        let result = ValidationResult {
            passed: true,
            accuracy: Some(0.95),
            top5_accuracy: Some(0.99),
            num_samples: 100,
            successful_inferences: 98,
            failed_inferences: 2,
            avg_inference_time_ms: 5.5,
            peak_memory_mb: 125.3,
            gradient_check_passed: Some(true),
            numerical_stability: 0.92,
            errors: vec![],
            warnings: vec!["High memory usage".to_string()],
        };

        let formatted = format_validation_result(&result);
        assert!(formatted.contains("VALIDATION REPORT"));
        assert!(formatted.contains("PASSED"));
    }
}
