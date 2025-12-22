//! Cross-Framework Gradient Verification
//!
//! This module provides comprehensive gradient verification capabilities across different
//! machine learning frameworks, ensuring torsh-autograd produces correct and consistent
//! gradients compared to established frameworks like PyTorch, JAX, and TensorFlow.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::error_handling::{AutogradError, AutogradResult};
use scirs2_core::ndarray::{Array, ArrayView, IxDyn};
use scirs2_core::random::quick::random_f64;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::Path;

/// Supported frameworks for gradient verification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SupportedFramework {
    PyTorch,
    JAX,
    TensorFlow,
    Torch,
    Custom(u32), // For user-defined frameworks
}

impl fmt::Display for SupportedFramework {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SupportedFramework::PyTorch => write!(f, "PyTorch"),
            SupportedFramework::JAX => write!(f, "JAX"),
            SupportedFramework::TensorFlow => write!(f, "TensorFlow"),
            SupportedFramework::Torch => write!(f, "Torch"),
            SupportedFramework::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

/// Gradient data structure for cross-framework comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientData {
    pub values: Vec<f64>,
    pub shape: Vec<usize>,
    pub metadata: HashMap<String, String>,
    pub framework: SupportedFramework,
    pub computation_time: Option<f64>, // in milliseconds
    pub memory_usage: Option<usize>,   // in bytes
}

impl GradientData {
    pub fn new(values: Vec<f64>, shape: Vec<usize>, framework: SupportedFramework) -> Self {
        Self {
            values,
            shape,
            metadata: HashMap::new(),
            framework,
            computation_time: None,
            memory_usage: None,
        }
    }

    pub fn from_array(array: &ArrayView<f64, IxDyn>, framework: SupportedFramework) -> Self {
        let values = array.iter().cloned().collect();
        let shape = array.shape().to_vec();
        Self::new(values, shape, framework)
    }

    pub fn to_array(&self) -> AutogradResult<Array<f64, IxDyn>> {
        let array =
            Array::from_shape_vec(self.shape.as_slice(), self.values.clone()).map_err(|e| {
                AutogradError::gradient_computation(
                    "array_creation",
                    format!("Failed to create array: {}", e),
                )
            })?;
        Ok(array)
    }

    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    pub fn total_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_compatible_shape(&self, other: &GradientData) -> bool {
        self.shape == other.shape
    }
}

/// Detailed comparison statistics between two gradients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientComparisonResult {
    pub framework_a: SupportedFramework,
    pub framework_b: SupportedFramework,
    pub operation_name: String,
    pub absolute_error: GradientErrorStats,
    pub relative_error: GradientErrorStats,
    pub correlation_coefficient: f64,
    pub cosine_similarity: f64,
    pub max_deviation_index: Option<usize>,
    pub passed_tolerance: bool,
    pub comparison_metadata: HashMap<String, String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Statistical measures of gradient errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientErrorStats {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub percentile_95: f64,
    pub percentile_99: f64,
    pub l1_norm: f64,
    pub l2_norm: f64,
    pub inf_norm: f64,
}

impl GradientErrorStats {
    pub fn compute(errors: &[f64]) -> Self {
        let mut sorted_errors = errors.to_vec();
        sorted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = errors.len() as f64;
        let mean = errors.iter().sum::<f64>() / n;
        let variance = errors.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let min = sorted_errors.first().copied().unwrap_or(0.0);
        let max = sorted_errors.last().copied().unwrap_or(0.0);
        let median = Self::percentile(&sorted_errors, 50.0);
        let percentile_95 = Self::percentile(&sorted_errors, 95.0);
        let percentile_99 = Self::percentile(&sorted_errors, 99.0);

        let l1_norm = errors.iter().map(|x| x.abs()).sum::<f64>();
        let l2_norm = errors.iter().map(|x| x * x).sum::<f64>().sqrt();
        let inf_norm = errors.iter().map(|x| x.abs()).fold(0.0, f64::max);

        Self {
            mean,
            std_dev,
            min,
            max,
            median,
            percentile_95,
            percentile_99,
            l1_norm,
            l2_norm,
            inf_norm,
        }
    }

    fn percentile(sorted_data: &[f64], p: f64) -> f64 {
        if sorted_data.is_empty() {
            return 0.0;
        }

        let index = (p / 100.0) * (sorted_data.len() as f64 - 1.0);
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            sorted_data[lower]
        } else {
            let weight = index - lower as f64;
            sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
        }
    }
}

/// Tolerance settings for gradient comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonTolerance {
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub correlation_threshold: f64,
    pub cosine_similarity_threshold: f64,
    pub ignore_near_zero: bool,
    pub near_zero_threshold: f64,
}

impl Default for ComparisonTolerance {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1e-6,
            relative_tolerance: 1e-4,
            correlation_threshold: 0.99,
            cosine_similarity_threshold: 0.99,
            ignore_near_zero: true,
            near_zero_threshold: 1e-8,
        }
    }
}

impl ComparisonTolerance {
    pub fn strict() -> Self {
        Self {
            absolute_tolerance: 1e-8,
            relative_tolerance: 1e-6,
            correlation_threshold: 0.999,
            cosine_similarity_threshold: 0.999,
            ignore_near_zero: true,
            near_zero_threshold: 1e-10,
        }
    }

    pub fn relaxed() -> Self {
        Self {
            absolute_tolerance: 1e-4,
            relative_tolerance: 1e-2,
            correlation_threshold: 0.95,
            cosine_similarity_threshold: 0.95,
            ignore_near_zero: true,
            near_zero_threshold: 1e-6,
        }
    }
}

/// Cross-framework gradient verification engine
pub struct CrossFrameworkVerifier {
    tolerance: ComparisonTolerance,
    reference_gradients: HashMap<String, GradientData>,
    verification_history: Vec<GradientComparisonResult>,
    framework_adapters: HashMap<SupportedFramework, Box<dyn FrameworkAdapter>>,
}

impl CrossFrameworkVerifier {
    pub fn new(tolerance: ComparisonTolerance) -> Self {
        Self {
            tolerance,
            reference_gradients: HashMap::new(),
            verification_history: Vec::new(),
            framework_adapters: HashMap::new(),
        }
    }

    pub fn with_default_tolerance() -> Self {
        Self::new(ComparisonTolerance::default())
    }

    pub fn register_framework_adapter(
        &mut self,
        framework: SupportedFramework,
        adapter: Box<dyn FrameworkAdapter>,
    ) {
        self.framework_adapters.insert(framework, adapter);
    }

    pub fn add_reference_gradient(&mut self, operation_name: String, gradient: GradientData) {
        self.reference_gradients.insert(operation_name, gradient);
    }

    pub fn compare_gradients(
        &mut self,
        operation_name: String,
        gradient_a: &GradientData,
        gradient_b: &GradientData,
    ) -> AutogradResult<GradientComparisonResult> {
        if !gradient_a.is_compatible_shape(gradient_b) {
            return Err(AutogradError::shape_mismatch(
                "gradient_comparison",
                gradient_a.shape.clone(),
                gradient_b.shape.clone(),
            ));
        }

        let absolute_errors = self.compute_absolute_errors(&gradient_a.values, &gradient_b.values);
        let relative_errors = self.compute_relative_errors(&gradient_a.values, &gradient_b.values);

        let absolute_error_stats = GradientErrorStats::compute(&absolute_errors);
        let relative_error_stats = GradientErrorStats::compute(&relative_errors);

        let correlation = self.compute_correlation(&gradient_a.values, &gradient_b.values);
        let cosine_sim = self.compute_cosine_similarity(&gradient_a.values, &gradient_b.values);

        let max_deviation_index = absolute_errors
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx);

        let passed_tolerance = self.check_tolerance(
            &absolute_error_stats,
            &relative_error_stats,
            correlation,
            cosine_sim,
        );

        let result = GradientComparisonResult {
            framework_a: gradient_a.framework,
            framework_b: gradient_b.framework,
            operation_name,
            absolute_error: absolute_error_stats,
            relative_error: relative_error_stats,
            correlation_coefficient: correlation,
            cosine_similarity: cosine_sim,
            max_deviation_index,
            passed_tolerance,
            comparison_metadata: HashMap::new(),
            timestamp: chrono::Utc::now(),
        };

        self.verification_history.push(result.clone());
        Ok(result)
    }

    pub fn verify_against_reference(
        &mut self,
        operation_name: &str,
        gradient: &GradientData,
    ) -> AutogradResult<GradientComparisonResult> {
        let reference = self
            .reference_gradients
            .get(operation_name)
            .ok_or_else(|| {
                AutogradError::gradient_computation(
                    "reference_verification",
                    format!(
                        "No reference gradient found for operation: {}",
                        operation_name
                    ),
                )
            })?
            .clone();

        self.compare_gradients(operation_name.to_string(), &reference, gradient)
    }

    pub fn batch_verify(
        &mut self,
        gradients: Vec<(String, GradientData, GradientData)>,
    ) -> AutogradResult<Vec<GradientComparisonResult>> {
        let mut results = Vec::new();

        for (operation_name, gradient_a, gradient_b) in gradients {
            match self.compare_gradients(operation_name.clone(), &gradient_a, &gradient_b) {
                Ok(result) => results.push(result),
                Err(e) => {
                    eprintln!(
                        "Failed to verify gradients for operation {}: {}",
                        operation_name, e
                    );
                    continue;
                }
            }
        }

        Ok(results)
    }

    pub fn cross_framework_verification(
        &mut self,
        operation_name: &str,
        input_data: &[f64],
        input_shape: &[usize],
    ) -> AutogradResult<HashMap<SupportedFramework, GradientData>> {
        let mut results = HashMap::new();

        for (framework, adapter) in &self.framework_adapters {
            match adapter.compute_gradient(operation_name, input_data, input_shape) {
                Ok(gradient) => {
                    results.insert(*framework, gradient);
                }
                Err(e) => {
                    eprintln!(
                        "Failed to compute gradient for framework {}: {}",
                        framework, e
                    );
                    continue;
                }
            }
        }

        Ok(results)
    }

    pub fn generate_verification_report(&self) -> VerificationReport {
        VerificationReport::new(&self.verification_history, &self.tolerance)
    }

    pub fn export_verification_results(&self, file_path: &Path) -> AutogradResult<()> {
        let json_data = serde_json::to_string_pretty(&self.verification_history).map_err(|e| {
            AutogradError::gradient_computation(
                "json_serialization",
                format!("Failed to serialize results: {}", e),
            )
        })?;

        std::fs::write(file_path, json_data).map_err(|e| {
            AutogradError::gradient_computation(
                "file_write",
                format!("Failed to write file: {}", e),
            )
        })?;

        Ok(())
    }

    pub fn import_verification_results(&mut self, file_path: &Path) -> AutogradResult<()> {
        let json_data = std::fs::read_to_string(file_path).map_err(|e| {
            AutogradError::gradient_computation("file_read", format!("Failed to read file: {}", e))
        })?;

        let results: Vec<GradientComparisonResult> =
            serde_json::from_str(&json_data).map_err(|e| {
                AutogradError::gradient_computation(
                    "json_parsing",
                    format!("Failed to parse results: {}", e),
                )
            })?;

        self.verification_history.extend(results);
        Ok(())
    }

    fn compute_absolute_errors(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).collect()
    }

    fn compute_relative_errors(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                if self.tolerance.ignore_near_zero && x.abs() < self.tolerance.near_zero_threshold {
                    0.0
                } else {
                    ((x - y) / (x.abs() + 1e-10)).abs()
                }
            })
            .collect()
    }

    fn compute_correlation(&self, a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let mean_a = a.iter().sum::<f64>() / n;
        let mean_b = b.iter().sum::<f64>() / n;

        let numerator: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - mean_a) * (y - mean_b))
            .sum();

        let var_a: f64 = a.iter().map(|x| (x - mean_a).powi(2)).sum();
        let var_b: f64 = b.iter().map(|y| (y - mean_b).powi(2)).sum();

        let denominator = (var_a * var_b).sqrt();

        if denominator.abs() < 1e-10 {
            if numerator.abs() < 1e-10 {
                1.0
            } else {
                0.0
            }
        } else {
            numerator / denominator
        }
    }

    fn compute_cosine_similarity(&self, a: &[f64], b: &[f64]) -> f64 {
        let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|y| y * y).sum::<f64>().sqrt();

        let denominator = norm_a * norm_b;

        if denominator.abs() < 1e-10 {
            if dot_product.abs() < 1e-10 {
                1.0
            } else {
                0.0
            }
        } else {
            dot_product / denominator
        }
    }

    fn check_tolerance(
        &self,
        abs_stats: &GradientErrorStats,
        rel_stats: &GradientErrorStats,
        correlation: f64,
        cosine_sim: f64,
    ) -> bool {
        abs_stats.max <= self.tolerance.absolute_tolerance
            && rel_stats.max <= self.tolerance.relative_tolerance
            && correlation >= self.tolerance.correlation_threshold
            && cosine_sim >= self.tolerance.cosine_similarity_threshold
    }
}

/// Framework adapter trait for different ML frameworks
pub trait FrameworkAdapter: Send + Sync {
    fn compute_gradient(
        &self,
        operation_name: &str,
        input_data: &[f64],
        input_shape: &[usize],
    ) -> AutogradResult<GradientData>;
    fn framework_name(&self) -> SupportedFramework;
    fn supported_operations(&self) -> Vec<String>;
    fn framework_version(&self) -> String;
}

/// Torsh-specific framework adapter
pub struct TorshFrameworkAdapter {
    // Integration with torsh-autograd would go here
}

impl TorshFrameworkAdapter {
    pub fn new() -> Self {
        Self {}
    }
}

impl FrameworkAdapter for TorshFrameworkAdapter {
    fn compute_gradient(
        &self,
        operation_name: &str,
        _input_data: &[f64],
        input_shape: &[usize],
    ) -> AutogradResult<GradientData> {
        // This would integrate with actual torsh autograd computation
        // For now, we'll create a mock gradient for demonstration
        let total_elements = input_shape.iter().product();
        let gradient_values: Vec<f64> = (0..total_elements)
            .map(|_| {
                let rand_val: f64 = random_f64();
                -1.0 + 2.0 * rand_val // Scale to [-1.0, 1.0]
            })
            .collect();

        let mut gradient = GradientData::new(
            gradient_values,
            input_shape.to_vec(),
            SupportedFramework::Torch,
        );
        gradient.add_metadata("operation".to_string(), operation_name.to_string());
        gradient.add_metadata("framework_version".to_string(), "0.1.0".to_string());

        Ok(gradient)
    }

    fn framework_name(&self) -> SupportedFramework {
        SupportedFramework::Torch
    }

    fn supported_operations(&self) -> Vec<String> {
        vec![
            "add".to_string(),
            "mul".to_string(),
            "sin".to_string(),
            "cos".to_string(),
            "exp".to_string(),
            "log".to_string(),
            "pow".to_string(),
            "relu".to_string(),
            "sigmoid".to_string(),
            "tanh".to_string(),
            "softmax".to_string(),
        ]
    }

    fn framework_version(&self) -> String {
        "torsh-autograd-0.1.0-alpha.2".to_string()
    }
}

/// Mock PyTorch adapter for testing purposes
pub struct MockPyTorchAdapter {
    // In a real implementation, this would use PyTorch's Python API
    // through pyo3 or similar
}

impl MockPyTorchAdapter {
    pub fn new() -> Self {
        Self {}
    }
}

impl FrameworkAdapter for MockPyTorchAdapter {
    fn compute_gradient(
        &self,
        operation_name: &str,
        input_data: &[f64],
        input_shape: &[usize],
    ) -> AutogradResult<GradientData> {
        // Mock PyTorch gradient computation
        // In reality, this would call into PyTorch
        let total_elements = input_shape.iter().product();
        let gradient_values: Vec<f64> = (0..total_elements)
            .map(|i| {
                // Create slightly different gradients to test comparison
                let base_value = input_data.get(i).unwrap_or(&0.0);
                match operation_name {
                    "add" => 1.0,
                    "mul" => *base_value,
                    "sin" => base_value.cos(),
                    "cos" => -base_value.sin(),
                    "exp" => base_value.exp(),
                    "log" => 1.0 / base_value.max(1e-10),
                    "pow" => 2.0 * base_value, // x^2 derivative
                    "relu" => {
                        if *base_value > 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    _ => 0.0,
                }
            })
            .collect();

        let mut gradient = GradientData::new(
            gradient_values,
            input_shape.to_vec(),
            SupportedFramework::PyTorch,
        );
        gradient.add_metadata("operation".to_string(), operation_name.to_string());
        gradient.add_metadata("framework_version".to_string(), "2.1.0".to_string());

        Ok(gradient)
    }

    fn framework_name(&self) -> SupportedFramework {
        SupportedFramework::PyTorch
    }

    fn supported_operations(&self) -> Vec<String> {
        vec![
            "add".to_string(),
            "mul".to_string(),
            "sin".to_string(),
            "cos".to_string(),
            "exp".to_string(),
            "log".to_string(),
            "pow".to_string(),
            "relu".to_string(),
            "sigmoid".to_string(),
            "tanh".to_string(),
            "softmax".to_string(),
        ]
    }

    fn framework_version(&self) -> String {
        "pytorch-2.1.0".to_string()
    }
}

/// Comprehensive verification report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    pub total_comparisons: usize,
    pub passed_comparisons: usize,
    pub failed_comparisons: usize,
    pub success_rate: f64,
    pub framework_comparison_matrix: HashMap<(SupportedFramework, SupportedFramework), usize>,
    pub operation_success_rates: HashMap<String, f64>,
    pub average_errors: GradientErrorStats,
    pub tolerance_settings: ComparisonTolerance,
    pub generation_timestamp: chrono::DateTime<chrono::Utc>,
    pub detailed_failures: Vec<FailureAnalysis>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureAnalysis {
    pub operation_name: String,
    pub framework_a: SupportedFramework,
    pub framework_b: SupportedFramework,
    pub failure_reason: String,
    pub severity: FailureSeverity,
    pub suggested_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureSeverity {
    Low,      // Minor numerical differences
    Medium,   // Significant but acceptable differences
    High,     // Major differences that need investigation
    Critical, // Completely wrong gradients
}

impl VerificationReport {
    pub fn new(history: &[GradientComparisonResult], tolerance: &ComparisonTolerance) -> Self {
        let total_comparisons = history.len();
        let passed_comparisons = history.iter().filter(|r| r.passed_tolerance).count();
        let failed_comparisons = total_comparisons - passed_comparisons;
        let success_rate = if total_comparisons > 0 {
            passed_comparisons as f64 / total_comparisons as f64
        } else {
            0.0
        };

        let mut framework_comparison_matrix = HashMap::new();
        let mut operation_counts = HashMap::new();
        let mut operation_successes = HashMap::new();

        for result in history {
            let key = (result.framework_a, result.framework_b);
            *framework_comparison_matrix.entry(key).or_insert(0) += 1;

            *operation_counts
                .entry(result.operation_name.clone())
                .or_insert(0) += 1;
            if result.passed_tolerance {
                *operation_successes
                    .entry(result.operation_name.clone())
                    .or_insert(0) += 1;
            }
        }

        let operation_success_rates = operation_counts
            .into_iter()
            .map(|(op, count)| {
                let successes = operation_successes.get(&op).copied().unwrap_or(0);
                let rate = successes as f64 / count as f64;
                (op, rate)
            })
            .collect();

        let all_absolute_errors: Vec<f64> = history
            .iter()
            .flat_map(|r| vec![r.absolute_error.mean, r.absolute_error.max])
            .collect();

        let average_errors = if all_absolute_errors.is_empty() {
            GradientErrorStats::compute(&[0.0])
        } else {
            GradientErrorStats::compute(&all_absolute_errors)
        };

        let detailed_failures = history
            .iter()
            .filter(|r| !r.passed_tolerance)
            .map(|r| FailureAnalysis {
                operation_name: r.operation_name.clone(),
                framework_a: r.framework_a,
                framework_b: r.framework_b,
                failure_reason: Self::determine_failure_reason(r, tolerance),
                severity: Self::determine_failure_severity(r, tolerance),
                suggested_actions: Self::generate_suggested_actions(r, tolerance),
            })
            .collect();

        Self {
            total_comparisons,
            passed_comparisons,
            failed_comparisons,
            success_rate,
            framework_comparison_matrix,
            operation_success_rates,
            average_errors,
            tolerance_settings: tolerance.clone(),
            generation_timestamp: chrono::Utc::now(),
            detailed_failures,
        }
    }

    fn determine_failure_reason(
        result: &GradientComparisonResult,
        tolerance: &ComparisonTolerance,
    ) -> String {
        let mut reasons = Vec::new();

        if result.absolute_error.max > tolerance.absolute_tolerance {
            reasons.push(format!(
                "Absolute error too high: {:.2e} > {:.2e}",
                result.absolute_error.max, tolerance.absolute_tolerance
            ));
        }

        if result.relative_error.max > tolerance.relative_tolerance {
            reasons.push(format!(
                "Relative error too high: {:.2e} > {:.2e}",
                result.relative_error.max, tolerance.relative_tolerance
            ));
        }

        if result.correlation_coefficient < tolerance.correlation_threshold {
            reasons.push(format!(
                "Correlation too low: {:.4} < {:.4}",
                result.correlation_coefficient, tolerance.correlation_threshold
            ));
        }

        if result.cosine_similarity < tolerance.cosine_similarity_threshold {
            reasons.push(format!(
                "Cosine similarity too low: {:.4} < {:.4}",
                result.cosine_similarity, tolerance.cosine_similarity_threshold
            ));
        }

        reasons.join("; ")
    }

    fn determine_failure_severity(
        result: &GradientComparisonResult,
        tolerance: &ComparisonTolerance,
    ) -> FailureSeverity {
        let abs_ratio = result.absolute_error.max / tolerance.absolute_tolerance;
        let rel_ratio = result.relative_error.max / tolerance.relative_tolerance;
        let corr_diff = tolerance.correlation_threshold - result.correlation_coefficient;

        if abs_ratio > 1000.0 || rel_ratio > 1000.0 || corr_diff > 0.5 {
            FailureSeverity::Critical
        } else if abs_ratio > 100.0 || rel_ratio > 100.0 || corr_diff > 0.1 {
            FailureSeverity::High
        } else if abs_ratio > 10.0 || rel_ratio > 10.0 || corr_diff > 0.05 {
            FailureSeverity::Medium
        } else {
            FailureSeverity::Low
        }
    }

    fn generate_suggested_actions(
        result: &GradientComparisonResult,
        tolerance: &ComparisonTolerance,
    ) -> Vec<String> {
        let mut actions = Vec::new();

        if result.absolute_error.max > tolerance.absolute_tolerance {
            actions.push("Check numerical precision and computation algorithms".to_string());
            actions.push("Verify input data preprocessing consistency".to_string());
        }

        if result.correlation_coefficient < 0.9 {
            actions.push("Investigate algorithmic differences between frameworks".to_string());
            actions.push("Check for different random number generation seeds".to_string());
        }

        if result.cosine_similarity < 0.9 {
            actions.push("Verify gradient normalization and scaling".to_string());
            actions.push("Check for different regularization or optimization settings".to_string());
        }

        actions.push("Run additional test cases with different input data".to_string());
        actions
            .push("Consider relaxing tolerance settings if differences are acceptable".to_string());

        actions
    }

    pub fn print_summary(&self) {
        println!("=== Cross-Framework Gradient Verification Report ===");
        println!("Total Comparisons: {}", self.total_comparisons);
        println!(
            "Passed: {} ({:.2}%)",
            self.passed_comparisons,
            self.success_rate * 100.0
        );
        println!(
            "Failed: {} ({:.2}%)",
            self.failed_comparisons,
            (1.0 - self.success_rate) * 100.0
        );
        println!();

        println!("Operation Success Rates:");
        for (operation, rate) in &self.operation_success_rates {
            println!("  {}: {:.2}%", operation, rate * 100.0);
        }
        println!();

        if !self.detailed_failures.is_empty() {
            println!("Critical Failures:");
            for failure in &self.detailed_failures {
                if matches!(failure.severity, FailureSeverity::Critical) {
                    println!(
                        "  {} ({} vs {}): {}",
                        failure.operation_name,
                        failure.framework_a,
                        failure.framework_b,
                        failure.failure_reason
                    );
                }
            }
        }
    }

    pub fn export_detailed_report(&self, file_path: &Path) -> AutogradResult<()> {
        let json_data = serde_json::to_string_pretty(self).map_err(|e| {
            AutogradError::gradient_computation(
                "report_serialization",
                format!("Failed to serialize report: {}", e),
            )
        })?;

        std::fs::write(file_path, json_data).map_err(|e| {
            AutogradError::gradient_computation(
                "report_write",
                format!("Failed to write report: {}", e),
            )
        })?;

        Ok(())
    }
}

/// Global cross-framework verifier instance
static GLOBAL_VERIFIER: std::sync::OnceLock<std::sync::Mutex<CrossFrameworkVerifier>> =
    std::sync::OnceLock::new();

pub fn get_global_verifier() -> &'static std::sync::Mutex<CrossFrameworkVerifier> {
    GLOBAL_VERIFIER
        .get_or_init(|| std::sync::Mutex::new(CrossFrameworkVerifier::with_default_tolerance()))
}

pub fn initialize_verification_frameworks() {
    let verifier = get_global_verifier();
    let mut verifier_lock = verifier.lock().unwrap();
    verifier_lock.register_framework_adapter(
        SupportedFramework::Torch,
        Box::new(TorshFrameworkAdapter::new()),
    );
    verifier_lock.register_framework_adapter(
        SupportedFramework::PyTorch,
        Box::new(MockPyTorchAdapter::new()),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    // Array macro is not available from scirs2_core::ndarray_ext, using Vec instead

    #[test]
    fn test_gradient_data_creation() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let gradient =
            GradientData::new(values.clone(), shape.clone(), SupportedFramework::PyTorch);

        assert_eq!(gradient.values, values);
        assert_eq!(gradient.shape, shape);
        assert_eq!(gradient.framework, SupportedFramework::PyTorch);
        assert_eq!(gradient.total_elements(), 4);
    }

    #[test]
    fn test_gradient_data_from_array() {
        let arr = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let gradient = GradientData::from_array(&arr.view().into_dyn(), SupportedFramework::JAX);

        assert_eq!(gradient.values, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(gradient.shape, vec![2, 2]);
        assert_eq!(gradient.framework, SupportedFramework::JAX);
    }

    #[test]
    fn test_gradient_comparison_identical() {
        let mut verifier = CrossFrameworkVerifier::with_default_tolerance();

        let values = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let gradient_a =
            GradientData::new(values.clone(), shape.clone(), SupportedFramework::PyTorch);
        let gradient_b = GradientData::new(values, shape, SupportedFramework::JAX);

        let result = verifier
            .compare_gradients("test_op".to_string(), &gradient_a, &gradient_b)
            .unwrap();

        assert!(result.passed_tolerance);
        assert_eq!(result.correlation_coefficient, 1.0);
        assert_eq!(result.cosine_similarity, 1.0);
        assert!(result.absolute_error.max < 1e-10);
    }

    #[test]
    fn test_gradient_comparison_different() {
        let mut verifier = CrossFrameworkVerifier::with_default_tolerance();

        let values_a = vec![1.0, 2.0, 3.0, 4.0];
        let values_b = vec![1.1, 2.1, 3.1, 4.1]; // Small differences
        let shape = vec![2, 2];

        let gradient_a = GradientData::new(values_a, shape.clone(), SupportedFramework::PyTorch);
        let gradient_b = GradientData::new(values_b, shape, SupportedFramework::JAX);

        let result = verifier
            .compare_gradients("test_op".to_string(), &gradient_a, &gradient_b)
            .unwrap();

        assert!(!result.passed_tolerance);
        assert!(result.correlation_coefficient > 0.9);
        assert!(result.absolute_error.max > 0.0);
    }

    #[test]
    fn test_framework_adapters() {
        let torsh_adapter = TorshFrameworkAdapter::new();
        let pytorch_adapter = MockPyTorchAdapter::new();

        assert_eq!(torsh_adapter.framework_name(), SupportedFramework::Torch);
        assert_eq!(
            pytorch_adapter.framework_name(),
            SupportedFramework::PyTorch
        );

        assert!(!torsh_adapter.supported_operations().is_empty());
        assert!(!pytorch_adapter.supported_operations().is_empty());
    }

    #[test]
    fn test_cross_framework_verification() {
        let mut verifier = CrossFrameworkVerifier::with_default_tolerance();
        verifier.register_framework_adapter(
            SupportedFramework::Torch,
            Box::new(TorshFrameworkAdapter::new()),
        );
        verifier.register_framework_adapter(
            SupportedFramework::PyTorch,
            Box::new(MockPyTorchAdapter::new()),
        );

        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input_shape = vec![2, 2];

        let results = verifier
            .cross_framework_verification("add", &input_data, &input_shape)
            .unwrap();

        assert_eq!(results.len(), 2);
        assert!(results.contains_key(&SupportedFramework::Torch));
        assert!(results.contains_key(&SupportedFramework::PyTorch));
    }

    #[test]
    fn test_error_statistics() {
        let errors = vec![0.1, 0.2, 0.05, 0.15, 0.3, 0.01, 0.25];
        let stats = GradientErrorStats::compute(&errors);

        assert!(stats.min <= stats.max);
        assert!(stats.median >= stats.min && stats.median <= stats.max);
        assert!(stats.percentile_95 >= stats.median);
        assert!(stats.l1_norm > 0.0);
        assert!(stats.l2_norm > 0.0);
        assert!(stats.inf_norm == stats.max);
    }

    #[test]
    fn test_tolerance_presets() {
        let default = ComparisonTolerance::default();
        let strict = ComparisonTolerance::strict();
        let relaxed = ComparisonTolerance::relaxed();

        assert!(strict.absolute_tolerance < default.absolute_tolerance);
        assert!(default.absolute_tolerance < relaxed.absolute_tolerance);
        assert!(strict.correlation_threshold > default.correlation_threshold);
        assert!(default.correlation_threshold > relaxed.correlation_threshold);
    }

    #[test]
    fn test_verification_report() {
        let mut verifier = CrossFrameworkVerifier::with_default_tolerance();

        // Create some mock comparison results
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let gradient_a =
            GradientData::new(values.clone(), shape.clone(), SupportedFramework::PyTorch);
        let gradient_b = GradientData::new(values, shape, SupportedFramework::JAX);

        let _result = verifier
            .compare_gradients("test_op".to_string(), &gradient_a, &gradient_b)
            .unwrap();

        let report = verifier.generate_verification_report();

        assert_eq!(report.total_comparisons, 1);
        assert!(report.success_rate >= 0.0 && report.success_rate <= 1.0);
        assert!(report.operation_success_rates.contains_key("test_op"));
    }

    #[test]
    fn test_global_verifier() {
        initialize_verification_frameworks();
        let verifier = get_global_verifier();
        let verifier_lock = verifier.lock().unwrap();

        assert!(verifier_lock
            .framework_adapters
            .contains_key(&SupportedFramework::Torch));
        assert!(verifier_lock
            .framework_adapters
            .contains_key(&SupportedFramework::PyTorch));
    }

    #[test]
    fn test_batch_verification() {
        let mut verifier = CrossFrameworkVerifier::with_default_tolerance();

        let batch_data = vec![
            (
                "op1".to_string(),
                GradientData::new(vec![1.0, 2.0], vec![2], SupportedFramework::PyTorch),
                GradientData::new(vec![1.0, 2.0], vec![2], SupportedFramework::JAX),
            ),
            (
                "op2".to_string(),
                GradientData::new(vec![3.0, 4.0], vec![2], SupportedFramework::PyTorch),
                GradientData::new(vec![3.0, 4.0], vec![2], SupportedFramework::JAX),
            ),
        ];

        let results = verifier.batch_verify(batch_data).unwrap();
        assert_eq!(results.len(), 2);

        for result in results {
            assert!(result.passed_tolerance);
        }
    }

    #[test]
    fn test_import_export_functionality() {
        let mut verifier = CrossFrameworkVerifier::with_default_tolerance();

        // Create a test result
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let gradient_a =
            GradientData::new(values.clone(), shape.clone(), SupportedFramework::PyTorch);
        let gradient_b = GradientData::new(values, shape, SupportedFramework::JAX);

        let _result = verifier
            .compare_gradients("test_op".to_string(), &gradient_a, &gradient_b)
            .unwrap();

        // Test export
        let temp_file = std::env::temp_dir().join("test_verification_results.json");
        verifier.export_verification_results(&temp_file).unwrap();

        // Test import
        let mut new_verifier = CrossFrameworkVerifier::with_default_tolerance();
        new_verifier
            .import_verification_results(&temp_file)
            .unwrap();

        assert_eq!(new_verifier.verification_history.len(), 1);

        // Cleanup
        std::fs::remove_file(temp_file).ok();
    }
}
