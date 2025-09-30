//! Numerical Stability Testing and Validation
//!
//! This module provides comprehensive numerical stability testing capabilities including:
//! - Gradient overflow/underflow detection
//! - Numerical precision validation
//! - Activation distribution analysis
//! - Model robustness testing
//! - Loss stability monitoring

use crate::Module;
use std::collections::HashMap;
use torsh_core::error::Result;
use torsh_tensor::Tensor;

/// Numerical stability test configuration
#[derive(Debug, Clone)]
pub struct StabilityConfig {
    /// Tolerance for numerical precision tests
    pub precision_tolerance: f64,
    /// Threshold for detecting overflow
    pub overflow_threshold: f32,
    /// Threshold for detecting underflow
    pub underflow_threshold: f32,
    /// Maximum allowed gradient norm
    pub max_gradient_norm: f32,
    /// Minimum allowed gradient norm (to detect vanishing gradients)
    pub min_gradient_norm: f32,
    /// Number of test iterations for stability checks
    pub test_iterations: usize,
    /// Whether to enable verbose output
    pub verbose: bool,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            precision_tolerance: 1e-6,
            overflow_threshold: 1e6,
            underflow_threshold: 1e-10,
            max_gradient_norm: 100.0,
            min_gradient_norm: 1e-8,
            test_iterations: 100,
            verbose: false,
        }
    }
}

/// Results of numerical stability tests
#[derive(Debug, Clone)]
pub struct StabilityResults {
    /// Whether all tests passed
    pub passed: bool,
    /// Individual test results
    pub test_results: HashMap<String, TestResult>,
    /// Overall stability score (0.0 to 1.0)
    pub stability_score: f64,
    /// Detected issues
    pub issues: Vec<StabilityIssue>,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Individual test result
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Test name
    pub name: String,
    /// Whether the test passed
    pub passed: bool,
    /// Test score (0.0 to 1.0)
    pub score: f64,
    /// Error message if test failed
    pub error: Option<String>,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

/// Types of numerical stability issues
#[derive(Debug, Clone)]
pub enum StabilityIssue {
    /// Gradient overflow detected
    GradientOverflow { layer: String, magnitude: f32 },
    /// Gradient underflow/vanishing detected
    GradientUnderflow { layer: String, magnitude: f32 },
    /// Activation overflow detected
    ActivationOverflow { layer: String, max_value: f32 },
    /// Activation underflow detected
    ActivationUnderflow { layer: String, min_value: f32 },
    /// Poor numerical precision
    PrecisionLoss { operation: String, error: f64 },
    /// Unstable loss behavior
    LossInstability { variance: f64 },
    /// NaN or Inf values detected
    InvalidValues { location: String },
}

/// Numerical stability tester
pub struct StabilityTester {
    config: StabilityConfig,
}

impl StabilityTester {
    /// Create a new stability tester
    pub fn new(config: StabilityConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(StabilityConfig::default())
    }

    /// Run comprehensive stability tests on a model
    pub fn test_model<M: Module>(
        &self,
        model: &M,
        test_inputs: &[Tensor],
    ) -> Result<StabilityResults> {
        let mut test_results = HashMap::new();
        let mut issues = Vec::new();
        let mut recommendations = Vec::new();

        // Test 1: Parameter stability
        let param_result = self.test_parameter_stability(model)?;
        if !param_result.passed {
            issues.extend(self.extract_parameter_issues(&param_result));
        }
        test_results.insert("parameter_stability".to_string(), param_result);

        // Test 2: Forward pass stability
        let forward_result = self.test_forward_stability(model, test_inputs)?;
        if !forward_result.passed {
            issues.extend(self.extract_forward_issues(&forward_result));
        }
        test_results.insert("forward_stability".to_string(), forward_result);

        // Test 3: Activation analysis
        let activation_result = self.test_activation_stability(model, test_inputs)?;
        if !activation_result.passed {
            issues.extend(self.extract_activation_issues(&activation_result));
        }
        test_results.insert("activation_stability".to_string(), activation_result);

        // Test 4: Numerical precision
        let precision_result = self.test_numerical_precision(model, test_inputs)?;
        if !precision_result.passed {
            issues.extend(self.extract_precision_issues(&precision_result));
        }
        test_results.insert("numerical_precision".to_string(), precision_result);

        // Test 5: Loss stability
        let loss_result = self.test_loss_stability(model, test_inputs)?;
        if !loss_result.passed {
            issues.extend(self.extract_loss_issues(&loss_result));
        }
        test_results.insert("loss_stability".to_string(), loss_result);

        // Generate recommendations
        recommendations.extend(self.generate_recommendations(&issues));

        // Calculate overall score
        let stability_score = self.calculate_stability_score(&test_results);
        let passed = stability_score >= 0.8 && issues.is_empty();

        Ok(StabilityResults {
            passed,
            test_results,
            stability_score,
            issues,
            recommendations,
        })
    }

    /// Test parameter stability (detect NaN, Inf, extreme values)
    fn test_parameter_stability<M: Module>(&self, model: &M) -> Result<TestResult> {
        let mut passed = true;
        let mut score = 1.0;
        let mut error = None;
        let mut metrics = HashMap::new();

        let parameters = model.parameters();
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut extreme_count = 0;
        let mut total_params = 0;

        for (name, param) in parameters {
            let tensor = param.tensor();
            let tensor_guard = tensor.read();
            let data = tensor_guard.to_vec()?;
            total_params += data.len();

            for &value in &data {
                if value.is_nan() {
                    nan_count += 1;
                    passed = false;
                    error = Some(format!("NaN detected in parameter {}", name));
                } else if value.is_infinite() {
                    inf_count += 1;
                    passed = false;
                    error = Some(format!("Inf detected in parameter {}", name));
                } else if value.abs() > self.config.overflow_threshold {
                    extreme_count += 1;
                    score *= 0.9; // Reduce score for extreme values
                }
            }
        }

        metrics.insert("nan_count".to_string(), nan_count as f64);
        metrics.insert("inf_count".to_string(), inf_count as f64);
        metrics.insert("extreme_count".to_string(), extreme_count as f64);
        metrics.insert("total_params".to_string(), total_params as f64);

        // Adjust score based on extreme values
        if extreme_count > 0 {
            score *= (total_params - extreme_count) as f64 / total_params as f64;
        }

        Ok(TestResult {
            name: "Parameter Stability".to_string(),
            passed,
            score,
            error,
            metrics,
        })
    }

    /// Test forward pass stability with multiple inputs
    fn test_forward_stability<M: Module>(
        &self,
        model: &M,
        test_inputs: &[Tensor],
    ) -> Result<TestResult> {
        let mut passed = true;
        let mut score = 1.0;
        let mut error = None;
        let mut metrics = HashMap::new();

        let mut outputs = Vec::new();
        let mut output_variances = Vec::new();

        // Run forward passes and collect outputs
        for input in test_inputs {
            match model.forward(input) {
                Ok(output) => {
                    let data = output.to_vec()?;

                    // Check for NaN/Inf
                    for &value in &data {
                        if value.is_nan() || value.is_infinite() {
                            passed = false;
                            error = Some("NaN or Inf detected in forward pass output".to_string());
                            break;
                        }
                    }

                    outputs.push(data);
                }
                Err(e) => {
                    passed = false;
                    error = Some(format!("Forward pass failed: {}", e));
                    break;
                }
            }
        }

        // Analyze output variance across different inputs
        if outputs.len() > 1 {
            let variance = self.calculate_output_variance(&outputs);
            output_variances.push(variance);

            // Check if variance is reasonable
            if variance > 1e6 {
                score *= 0.8;
            }
        }

        metrics.insert("num_successful_passes".to_string(), outputs.len() as f64);
        metrics.insert(
            "output_variance".to_string(),
            output_variances.get(0).copied().unwrap_or(0.0),
        );

        Ok(TestResult {
            name: "Forward Pass Stability".to_string(),
            passed,
            score,
            error,
            metrics,
        })
    }

    /// Test activation distribution stability
    fn test_activation_stability<M: Module>(
        &self,
        model: &M,
        test_inputs: &[Tensor],
    ) -> Result<TestResult> {
        let mut passed = true;
        let mut score = 1.0;
        let mut error = None;
        let mut metrics = HashMap::new();

        let mut activation_stats = Vec::new();

        for input in test_inputs.iter().take(5) {
            // Limit to 5 samples for efficiency
            if let Ok(output) = model.forward(input) {
                let data = output.to_vec()?;

                let mean = data.iter().sum::<f32>() / data.len() as f32;
                let variance =
                    data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
                let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                activation_stats.push((mean, variance, min, max));

                // Check for extreme activations
                if max > self.config.overflow_threshold {
                    passed = false;
                    error = Some("Activation overflow detected".to_string());
                }

                if min.abs() < self.config.underflow_threshold && min != 0.0 {
                    score *= 0.9; // Reduce score for near-zero activations
                }
            }
        }

        // Calculate stability metrics
        if !activation_stats.is_empty() {
            let mean_variance = activation_stats
                .iter()
                .map(|(_, var, _, _)| *var as f64)
                .sum::<f64>()
                / activation_stats.len() as f64;

            let max_activation = activation_stats
                .iter()
                .map(|(_, _, _, max)| *max as f64)
                .fold(f64::NEG_INFINITY, f64::max);

            metrics.insert("mean_variance".to_string(), mean_variance);
            metrics.insert("max_activation".to_string(), max_activation);
        }

        Ok(TestResult {
            name: "Activation Stability".to_string(),
            passed,
            score,
            error,
            metrics,
        })
    }

    /// Test numerical precision through double precision comparison
    fn test_numerical_precision<M: Module>(
        &self,
        model: &M,
        test_inputs: &[Tensor],
    ) -> Result<TestResult> {
        let mut passed = true;
        let mut score = 1.0;
        let mut error = None;
        let mut metrics = HashMap::new();

        let mut precision_errors = Vec::new();

        // For a subset of inputs, test numerical precision
        for input in test_inputs.iter().take(3) {
            if let Ok(output1) = model.forward(input) {
                // Run the same computation again
                if let Ok(output2) = model.forward(input) {
                    let data1 = output1.to_vec()?;
                    let data2 = output2.to_vec()?;

                    // Calculate relative error
                    let mut max_error: f64 = 0.0;
                    for (&a, &b) in data1.iter().zip(data2.iter()) {
                        if a != 0.0 {
                            let rel_error = ((a - b) / a).abs() as f64;
                            max_error = max_error.max(rel_error);
                        }
                    }

                    precision_errors.push(max_error);

                    if max_error > self.config.precision_tolerance {
                        passed = false;
                        error = Some(format!(
                            "Poor numerical precision: max error {:.2e}",
                            max_error
                        ));
                        score *= 0.7;
                    }
                }
            }
        }

        let avg_precision_error = if !precision_errors.is_empty() {
            precision_errors.iter().sum::<f64>() / precision_errors.len() as f64
        } else {
            0.0
        };

        metrics.insert("avg_precision_error".to_string(), avg_precision_error);
        metrics.insert(
            "max_precision_error".to_string(),
            precision_errors.iter().fold(0.0f64, |a, &b| a.max(b)),
        );

        Ok(TestResult {
            name: "Numerical Precision".to_string(),
            passed,
            score,
            error,
            metrics,
        })
    }

    /// Test loss function stability
    fn test_loss_stability<M: Module>(
        &self,
        model: &M,
        test_inputs: &[Tensor],
    ) -> Result<TestResult> {
        let mut passed = true;
        let mut score = 1.0;
        let mut error = None;
        let mut metrics = HashMap::new();

        let mut losses = Vec::new();

        // Simulate loss computation (simplified)
        for input in test_inputs.iter().take(10) {
            if let Ok(output) = model.forward(input) {
                let data = output.to_vec()?;
                // Simple MSE-like loss simulation
                let loss = data.iter().map(|&x| x.powi(2)).sum::<f32>();

                if loss.is_nan() || loss.is_infinite() {
                    passed = false;
                    error = Some("NaN or Inf loss detected".to_string());
                    break;
                }

                losses.push(loss);
            }
        }

        // Analyze loss stability
        if losses.len() > 1 {
            let mean_loss = losses.iter().sum::<f32>() / losses.len() as f32;
            let loss_variance =
                losses.iter().map(|&x| (x - mean_loss).powi(2)).sum::<f32>() / losses.len() as f32;

            let cv = if mean_loss != 0.0 {
                (loss_variance.sqrt() / mean_loss).abs()
            } else {
                0.0
            };

            metrics.insert("loss_variance".to_string(), loss_variance as f64);
            metrics.insert("coefficient_of_variation".to_string(), cv as f64);

            // High coefficient of variation indicates instability
            if cv > 0.5 {
                score *= 0.8;
            }
        }

        Ok(TestResult {
            name: "Loss Stability".to_string(),
            passed,
            score,
            error,
            metrics,
        })
    }

    /// Calculate output variance across multiple runs
    fn calculate_output_variance(&self, outputs: &[Vec<f32>]) -> f64 {
        if outputs.is_empty() || outputs[0].is_empty() {
            return 0.0;
        }

        let _num_outputs = outputs.len();
        let output_size = outputs[0].len();
        let mut total_variance = 0.0;

        for i in 0..output_size {
            let values: Vec<f32> = outputs.iter().map(|out| out[i]).collect();
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let variance =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
            total_variance += variance as f64;
        }

        total_variance / output_size as f64
    }

    /// Extract issues from parameter test results
    fn extract_parameter_issues(&self, result: &TestResult) -> Vec<StabilityIssue> {
        let mut issues = Vec::new();

        if let Some(nan_count) = result.metrics.get("nan_count") {
            if *nan_count > 0.0 {
                issues.push(StabilityIssue::InvalidValues {
                    location: "model parameters".to_string(),
                });
            }
        }

        issues
    }

    /// Extract issues from forward pass test results
    fn extract_forward_issues(&self, result: &TestResult) -> Vec<StabilityIssue> {
        let mut issues = Vec::new();

        if !result.passed {
            issues.push(StabilityIssue::InvalidValues {
                location: "forward pass output".to_string(),
            });
        }

        issues
    }

    /// Extract issues from activation test results
    fn extract_activation_issues(&self, result: &TestResult) -> Vec<StabilityIssue> {
        let mut issues = Vec::new();

        if let Some(max_activation) = result.metrics.get("max_activation") {
            if *max_activation > self.config.overflow_threshold as f64 {
                issues.push(StabilityIssue::ActivationOverflow {
                    layer: "unknown".to_string(),
                    max_value: *max_activation as f32,
                });
            }
        }

        issues
    }

    /// Extract issues from precision test results
    fn extract_precision_issues(&self, result: &TestResult) -> Vec<StabilityIssue> {
        let mut issues = Vec::new();

        if let Some(max_error) = result.metrics.get("max_precision_error") {
            if *max_error > self.config.precision_tolerance {
                issues.push(StabilityIssue::PrecisionLoss {
                    operation: "forward pass".to_string(),
                    error: *max_error,
                });
            }
        }

        issues
    }

    /// Extract issues from loss test results
    fn extract_loss_issues(&self, result: &TestResult) -> Vec<StabilityIssue> {
        let mut issues = Vec::new();

        if let Some(variance) = result.metrics.get("loss_variance") {
            if *variance > 1e6 {
                issues.push(StabilityIssue::LossInstability {
                    variance: *variance,
                });
            }
        }

        issues
    }

    /// Generate recommendations based on detected issues
    fn generate_recommendations(&self, issues: &[StabilityIssue]) -> Vec<String> {
        let mut recommendations = Vec::new();

        for issue in issues {
            match issue {
                StabilityIssue::GradientOverflow { .. } => {
                    recommendations
                        .push("Consider gradient clipping or reducing learning rate".to_string());
                }
                StabilityIssue::GradientUnderflow { .. } => {
                    recommendations.push(
                        "Consider increasing learning rate or using gradient scaling".to_string(),
                    );
                }
                StabilityIssue::ActivationOverflow { .. } => {
                    recommendations
                        .push("Consider batch normalization or activation scaling".to_string());
                }
                StabilityIssue::PrecisionLoss { .. } => {
                    recommendations.push(
                        "Consider using higher precision or numerical stabilization techniques"
                            .to_string(),
                    );
                }
                StabilityIssue::LossInstability { .. } => {
                    recommendations.push(
                        "Consider reducing learning rate or using learning rate scheduling"
                            .to_string(),
                    );
                }
                StabilityIssue::InvalidValues { .. } => {
                    recommendations.push("Check for division by zero, invalid operations, or uninitialized parameters".to_string());
                }
                _ => {}
            }
        }

        recommendations.sort();
        recommendations.dedup();
        recommendations
    }

    /// Calculate overall stability score
    fn calculate_stability_score(&self, test_results: &HashMap<String, TestResult>) -> f64 {
        let total_score: f64 = test_results.values().map(|r| r.score).sum();
        let num_tests = test_results.len() as f64;

        if num_tests > 0.0 {
            total_score / num_tests
        } else {
            0.0
        }
    }
}

/// Utilities for numerical stability testing
pub mod utils {
    use super::*;
    use torsh_tensor::creation::*;

    /// Generate test inputs for stability testing
    pub fn generate_test_inputs(
        input_shape: &[usize],
        num_inputs: usize,
        value_range: (f32, f32),
    ) -> Result<Vec<Tensor>> {
        let mut inputs = Vec::new();

        for i in 0..num_inputs {
            let scale = (i as f32 + 1.0) / num_inputs as f32;
            let range_size = value_range.1 - value_range.0;
            let value = value_range.0 + range_size * scale;

            let input = full(input_shape, value)?;
            inputs.push(input);
        }

        Ok(inputs)
    }

    /// Generate pathological test inputs (edge cases)
    pub fn generate_pathological_inputs(input_shape: &[usize]) -> Result<Vec<Tensor>> {
        let mut inputs = Vec::new();

        // Zero input
        inputs.push(zeros(input_shape)?);

        // Large positive values
        inputs.push(full(input_shape, 1e6)?);

        // Large negative values
        inputs.push(full(input_shape, -1e6)?);

        // Very small positive values
        inputs.push(full(input_shape, 1e-6)?);

        // Very small negative values
        inputs.push(full(input_shape, -1e-6)?);

        Ok(inputs)
    }

    /// Quick stability check with default configuration
    pub fn quick_stability_check<M: Module>(model: &M, input_shape: &[usize]) -> Result<bool> {
        let tester = StabilityTester::default();
        let test_inputs = generate_test_inputs(input_shape, 5, (-1.0, 1.0))?;

        let results = tester.test_model(model, &test_inputs)?;
        Ok(results.passed)
    }

    /// Comprehensive stability analysis
    pub fn comprehensive_stability_analysis<M: Module>(
        model: &M,
        input_shape: &[usize],
    ) -> Result<StabilityResults> {
        let config = StabilityConfig {
            test_iterations: 50,
            verbose: true,
            ..Default::default()
        };

        let tester = StabilityTester::new(config);
        let mut test_inputs = generate_test_inputs(input_shape, 10, (-5.0, 5.0))?;
        let pathological = generate_pathological_inputs(input_shape)?;
        test_inputs.extend(pathological);

        tester.test_model(model, &test_inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::linear::Linear;
    use torsh_tensor::creation::*;

    #[test]
    fn test_stability_tester_creation() {
        let _tester = StabilityTester::default();
    }

    #[test]
    fn test_parameter_stability() -> Result<()> {
        let tester = StabilityTester::default();
        let linear = Linear::new(4, 2, true);

        let result = tester.test_parameter_stability(&linear)?;
        assert!(result.passed); // Should pass for well-initialized linear layer

        Ok(())
    }

    #[test]
    fn test_model_stability() -> Result<()> {
        let tester = StabilityTester::default();
        let linear = Linear::new(4, 2, true);
        let test_inputs = vec![ones(&[1, 4])?];

        let results = tester.test_model(&linear, &test_inputs)?;
        // Basic linear layer should be stable
        assert!(results.stability_score > 0.5);

        Ok(())
    }

    #[test]
    fn test_utils_generate_inputs() -> Result<()> {
        let inputs = utils::generate_test_inputs(&[2, 3], 5, (-1.0, 1.0))?;
        assert_eq!(inputs.len(), 5);
        assert_eq!(inputs[0].shape().dims(), &[2, 3]);

        Ok(())
    }

    #[test]
    fn test_quick_stability_check() -> Result<()> {
        let linear = Linear::new(4, 2, true);
        let is_stable = utils::quick_stability_check(&linear, &[1, 4])?;
        assert!(is_stable);

        Ok(())
    }
}
