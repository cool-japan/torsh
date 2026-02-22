//! Debugging tools for quantization
//!
//! This module provides comprehensive debugging capabilities for quantization including:
//! - Quantization debugger with step-by-step analysis
//! - Error analysis and distribution tracking
//! - Range tracking and monitoring
//! - Overflow detection and reporting
//! - Comparison tools for different quantization schemes

use crate::{QScheme, QuantConfig, TorshResult};
use std::collections::HashMap;
use torsh_core::{DType, TorshError};
use torsh_tensor::Tensor;

/// Comprehensive quantization debugger
#[derive(Debug)]
pub struct QuantizationDebugger {
    /// Debug mode enabled
    pub debug_enabled: bool,
    /// Step-by-step execution trace
    pub execution_trace: Vec<DebugStep>,
    /// Error statistics
    pub error_stats: ErrorStatistics,
    /// Range tracking information
    pub range_tracker: RangeTracker,
    /// Overflow detector
    pub overflow_detector: OverflowDetector,
}

/// Individual debug step information
#[derive(Debug, Clone)]
pub struct DebugStep {
    /// Step name/description
    pub name: String,
    /// Input tensor statistics
    pub input_stats: TensorStatistics,
    /// Output tensor statistics
    pub output_stats: TensorStatistics,
    /// Quantization parameters used
    pub quant_params: QuantParams,
    /// Error metrics
    pub error_metrics: ErrorMetrics,
    /// Timestamp
    pub timestamp: std::time::Instant,
}

/// Tensor statistics for debugging
#[derive(Debug, Clone)]
pub struct TensorStatistics {
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// Shape
    pub shape: Vec<usize>,
    /// Number of elements
    pub num_elements: usize,
    /// Data type
    pub dtype: DType,
}

/// Quantization parameters for debugging
#[derive(Debug, Clone)]
pub struct QuantParams {
    /// Scale factor
    pub scale: f32,
    /// Zero point
    pub zero_point: i32,
    /// Quantization scheme
    pub scheme: QScheme,
    /// Quantization range
    pub qint_range: (i32, i32),
}

/// Error metrics for quantization debugging
#[derive(Debug, Clone)]
pub struct ErrorMetrics {
    /// Mean absolute error
    pub mae: f32,
    /// Mean squared error
    pub mse: f32,
    /// Root mean squared error
    pub rmse: f32,
    /// Signal-to-noise ratio
    pub snr: f32,
    /// Peak signal-to-noise ratio
    pub psnr: f32,
    /// Cosine similarity
    pub cosine_similarity: f32,
}

/// Error statistics accumulator
#[derive(Debug)]
pub struct ErrorStatistics {
    /// Total number of operations
    pub total_ops: usize,
    /// Error distribution histogram
    pub error_histogram: Vec<usize>,
    /// Error bins
    pub error_bins: Vec<f32>,
    /// Cumulative error metrics
    pub cumulative_mae: f32,
    /// Cumulative MSE
    pub cumulative_mse: f32,
    /// Per-layer error tracking
    pub layer_errors: HashMap<String, Vec<f32>>,
}

/// Range tracking for tensors
#[derive(Debug)]
pub struct RangeTracker {
    /// Per-tensor range history
    pub tensor_ranges: HashMap<String, Vec<(f32, f32)>>,
    /// Range violations
    pub range_violations: Vec<RangeViolation>,
    /// Expected ranges
    pub expected_ranges: HashMap<String, (f32, f32)>,
    /// Range stability metrics
    pub stability_metrics: HashMap<String, f32>,
}

/// Range violation information
#[derive(Debug, Clone)]
pub struct RangeViolation {
    /// Tensor name
    pub tensor_name: String,
    /// Expected range
    pub expected_range: (f32, f32),
    /// Actual range
    pub actual_range: (f32, f32),
    /// Violation severity
    pub severity: ViolationSeverity,
    /// Timestamp
    pub timestamp: std::time::Instant,
}

/// Severity levels for violations
#[derive(Debug, Clone, PartialEq)]
pub enum ViolationSeverity {
    /// Minor violation (< 10% outside range)
    Minor,
    /// Moderate violation (10-50% outside range)
    Moderate,
    /// Severe violation (> 50% outside range)
    Severe,
}

/// Overflow detection and reporting
#[derive(Debug)]
pub struct OverflowDetector {
    /// Overflow events
    pub overflow_events: Vec<OverflowEvent>,
    /// Underflow events
    pub underflow_events: Vec<OverflowEvent>,
    /// Overflow thresholds
    pub overflow_threshold: f32,
    /// Underflow threshold
    pub underflow_threshold: f32,
    /// Detection enabled
    pub detection_enabled: bool,
}

/// Overflow/underflow event information
#[derive(Debug, Clone)]
pub struct OverflowEvent {
    /// Tensor name
    pub tensor_name: String,
    /// Value that caused overflow
    pub overflow_value: f32,
    /// Threshold that was exceeded
    pub threshold: f32,
    /// Number of elements that overflowed
    pub num_elements: usize,
    /// Position in tensor (first occurrence)
    pub position: Vec<usize>,
    /// Timestamp
    pub timestamp: std::time::Instant,
}

impl QuantizationDebugger {
    /// Create a new quantization debugger
    pub fn new() -> Self {
        Self {
            debug_enabled: true,
            execution_trace: Vec::new(),
            error_stats: ErrorStatistics::new(),
            range_tracker: RangeTracker::new(),
            overflow_detector: OverflowDetector::new(),
        }
    }

    /// Enable or disable debugging
    pub fn set_debug_enabled(&mut self, enabled: bool) {
        self.debug_enabled = enabled;
    }

    /// Debug a quantization operation
    pub fn debug_quantization(
        &mut self,
        name: &str,
        input: &Tensor,
        output: &Tensor,
        config: &QuantConfig,
        scale: f32,
        zero_point: i32,
    ) -> TorshResult<()> {
        if !self.debug_enabled {
            return Ok(());
        }

        let input_stats = self.compute_tensor_statistics(input)?;
        let output_stats = self.compute_tensor_statistics(output)?;

        let quant_params = QuantParams {
            scale,
            zero_point,
            scheme: config.scheme,
            qint_range: config.get_qint_range(),
        };

        let error_metrics = self.compute_error_metrics(input, output)?;

        let debug_step = DebugStep {
            name: name.to_string(),
            input_stats,
            output_stats,
            quant_params,
            error_metrics: error_metrics.clone(),
            timestamp: std::time::Instant::now(),
        };

        self.execution_trace.push(debug_step);
        self.error_stats.update(&error_metrics, name);
        self.range_tracker.track_range(name, input)?;
        self.overflow_detector.detect_overflow(name, input)?;

        Ok(())
    }

    /// Compute tensor statistics
    fn compute_tensor_statistics(&self, tensor: &Tensor) -> TorshResult<TensorStatistics> {
        let data = tensor.data()?;
        let num_elements = data.len();

        if num_elements == 0 {
            return Err(TorshError::InvalidArgument("Empty tensor".to_string()));
        }

        let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mean = data.iter().sum::<f32>() / num_elements as f32;

        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / num_elements as f32;
        let std = variance.sqrt();

        Ok(TensorStatistics {
            min,
            max,
            mean,
            std,
            shape: tensor.shape().dims().to_vec(),
            num_elements,
            dtype: tensor.dtype(),
        })
    }

    /// Compute error metrics between original and quantized tensors
    fn compute_error_metrics(
        &self,
        original: &Tensor,
        quantized: &Tensor,
    ) -> TorshResult<ErrorMetrics> {
        let orig_data = original.data()?;
        let quant_data = quantized.data()?;

        if orig_data.len() != quant_data.len() {
            return Err(TorshError::InvalidArgument(
                "Tensor size mismatch".to_string(),
            ));
        }

        let n = orig_data.len() as f32;

        // Mean Absolute Error
        let mae = orig_data
            .iter()
            .zip(quant_data.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f32>()
            / n;

        // Mean Squared Error
        let mse = orig_data
            .iter()
            .zip(quant_data.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            / n;

        // Root Mean Squared Error
        let rmse = mse.sqrt();

        // Signal-to-Noise Ratio
        let signal_power = orig_data.iter().map(|&x| x.powi(2)).sum::<f32>() / n;
        let noise_power = mse;
        let snr = if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f32::INFINITY
        };

        // Peak Signal-to-Noise Ratio
        let max_val = orig_data.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let psnr = if mse > 0.0 {
            20.0 * (max_val / rmse).log10()
        } else {
            f32::INFINITY
        };

        // Cosine Similarity
        let dot_product = orig_data
            .iter()
            .zip(quant_data.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f32>();

        let orig_norm = orig_data.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt();
        let quant_norm = quant_data.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt();

        let cosine_similarity = if orig_norm > 0.0 && quant_norm > 0.0 {
            dot_product / (orig_norm * quant_norm)
        } else {
            0.0
        };

        Ok(ErrorMetrics {
            mae,
            mse,
            rmse,
            snr,
            psnr,
            cosine_similarity,
        })
    }

    /// Generate comprehensive debug report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== QUANTIZATION DEBUG REPORT ===\n\n");

        // Execution summary
        report.push_str(&format!(
            "Total Operations: {}\n",
            self.execution_trace.len()
        ));
        report.push_str(&format!(
            "Debug Mode: {}\n\n",
            if self.debug_enabled {
                "Enabled"
            } else {
                "Disabled"
            }
        ));

        // Error statistics
        report.push_str("--- ERROR STATISTICS ---\n");
        report.push_str(&format!(
            "Total Operations: {}\n",
            self.error_stats.total_ops
        ));
        report.push_str(&format!(
            "Cumulative MAE: {:.6}\n",
            self.error_stats.cumulative_mae
        ));
        report.push_str(&format!(
            "Cumulative MSE: {:.6}\n",
            self.error_stats.cumulative_mse
        ));
        report.push_str(&format!(
            "Cumulative RMSE: {:.6}\n",
            self.error_stats.cumulative_mse.sqrt()
        ));
        report.push('\n');

        // Per-layer errors
        if !self.error_stats.layer_errors.is_empty() {
            report.push_str("--- PER-LAYER ERRORS ---\n");
            for (layer, errors) in &self.error_stats.layer_errors {
                let avg_error = errors.iter().sum::<f32>() / errors.len() as f32;
                report.push_str(&format!(
                    "{}: {:.6} (avg over {} ops)\n",
                    layer,
                    avg_error,
                    errors.len()
                ));
            }
            report.push('\n');
        }

        // Range violations
        if !self.range_tracker.range_violations.is_empty() {
            report.push_str("--- RANGE VIOLATIONS ---\n");
            for violation in &self.range_tracker.range_violations {
                report.push_str(&format!(
                    "{}: Expected [{:.3}, {:.3}], Got [{:.3}, {:.3}] - {:?}\n",
                    violation.tensor_name,
                    violation.expected_range.0,
                    violation.expected_range.1,
                    violation.actual_range.0,
                    violation.actual_range.1,
                    violation.severity
                ));
            }
            report.push('\n');
        }

        // Overflow events
        if !self.overflow_detector.overflow_events.is_empty()
            || !self.overflow_detector.underflow_events.is_empty()
        {
            report.push_str("--- OVERFLOW/UNDERFLOW EVENTS ---\n");
            for event in &self.overflow_detector.overflow_events {
                report.push_str(&format!(
                    "OVERFLOW in {}: {:.3} > {:.3} ({} elements)\n",
                    event.tensor_name, event.overflow_value, event.threshold, event.num_elements
                ));
            }
            for event in &self.overflow_detector.underflow_events {
                report.push_str(&format!(
                    "UNDERFLOW in {}: {:.3} < {:.3} ({} elements)\n",
                    event.tensor_name, event.overflow_value, event.threshold, event.num_elements
                ));
            }
            report.push('\n');
        }

        // Recent execution trace (last 10 steps)
        if !self.execution_trace.is_empty() {
            report.push_str("--- RECENT EXECUTION TRACE ---\n");
            let start_idx = self.execution_trace.len().saturating_sub(10);
            for (i, step) in self.execution_trace[start_idx..].iter().enumerate() {
                report.push_str(&format!(
                    "{}. {} - MAE: {:.6}, SNR: {:.2} dB\n",
                    start_idx + i + 1,
                    step.name,
                    step.error_metrics.mae,
                    step.error_metrics.snr
                ));
            }
        }

        report
    }

    /// Clear all debug information
    pub fn clear(&mut self) {
        self.execution_trace.clear();
        self.error_stats = ErrorStatistics::new();
        self.range_tracker = RangeTracker::new();
        self.overflow_detector = OverflowDetector::new();
    }

    /// Export debug data to JSON format
    pub fn export_to_json(&self) -> TorshResult<String> {
        // Simplified JSON export (in practice, would use serde)
        let mut json = String::new();
        json.push_str("{\n");
        json.push_str(&format!(
            "  \"total_operations\": {},\n",
            self.execution_trace.len()
        ));
        json.push_str(&format!("  \"debug_enabled\": {},\n", self.debug_enabled));
        json.push_str(&format!(
            "  \"cumulative_mae\": {},\n",
            self.error_stats.cumulative_mae
        ));
        json.push_str(&format!(
            "  \"cumulative_mse\": {},\n",
            self.error_stats.cumulative_mse
        ));
        json.push_str(&format!(
            "  \"range_violations\": {},\n",
            self.range_tracker.range_violations.len()
        ));
        json.push_str(&format!(
            "  \"overflow_events\": {}\n",
            self.overflow_detector.overflow_events.len()
        ));
        json.push_str("}\n");
        Ok(json)
    }
}

impl Default for QuantizationDebugger {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ErrorStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorStatistics {
    /// Create new error statistics
    pub fn new() -> Self {
        Self {
            total_ops: 0,
            error_histogram: vec![0; 100], // 100 bins
            error_bins: (0..100).map(|i| i as f32 * 0.01).collect(),
            cumulative_mae: 0.0,
            cumulative_mse: 0.0,
            layer_errors: HashMap::new(),
        }
    }

    /// Update statistics with new error metrics
    pub fn update(&mut self, metrics: &ErrorMetrics, layer_name: &str) {
        self.total_ops += 1;
        self.cumulative_mae += metrics.mae;
        self.cumulative_mse += metrics.mse;

        // Update per-layer errors
        self.layer_errors
            .entry(layer_name.to_string())
            .or_default()
            .push(metrics.mae);

        // Update histogram
        let bin_idx = (metrics.mae / 0.01).floor() as usize;
        if bin_idx < self.error_histogram.len() {
            self.error_histogram[bin_idx] += 1;
        }
    }

    /// Get average error metrics
    pub fn get_averages(&self) -> (f32, f32) {
        if self.total_ops > 0 {
            (
                self.cumulative_mae / self.total_ops as f32,
                self.cumulative_mse / self.total_ops as f32,
            )
        } else {
            (0.0, 0.0)
        }
    }
}

impl Default for RangeTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl RangeTracker {
    /// Create new range tracker
    pub fn new() -> Self {
        Self {
            tensor_ranges: HashMap::new(),
            range_violations: Vec::new(),
            expected_ranges: HashMap::new(),
            stability_metrics: HashMap::new(),
        }
    }

    /// Track tensor range
    pub fn track_range(&mut self, name: &str, tensor: &Tensor) -> TorshResult<()> {
        let data = tensor.data()?;
        if data.is_empty() {
            return Ok(());
        }

        let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Store range history
        self.tensor_ranges
            .entry(name.to_string())
            .or_default()
            .push((min, max));

        // Check for violations if expected range is set
        if let Some(&expected_range) = self.expected_ranges.get(name) {
            let violation = self.check_range_violation(name, (min, max), expected_range);
            if let Some(v) = violation {
                self.range_violations.push(v);
            }
        }

        // Update stability metrics
        self.update_stability_metric(name);

        Ok(())
    }

    /// Set expected range for a tensor
    pub fn set_expected_range(&mut self, name: &str, range: (f32, f32)) {
        self.expected_ranges.insert(name.to_string(), range);
    }

    /// Check for range violations
    fn check_range_violation(
        &self,
        name: &str,
        actual: (f32, f32),
        expected: (f32, f32),
    ) -> Option<RangeViolation> {
        let (min_actual, max_actual) = actual;
        let (min_expected, max_expected) = expected;

        let min_violation = (min_actual - min_expected) / (max_expected - min_expected).abs();
        let max_violation = (max_actual - max_expected) / (max_expected - min_expected).abs();

        let max_violation_pct = min_violation.abs().max(max_violation.abs());

        if max_violation_pct > 0.01 {
            // 1% threshold
            let severity = if max_violation_pct < 0.1 {
                ViolationSeverity::Minor
            } else if max_violation_pct < 0.5 {
                ViolationSeverity::Moderate
            } else {
                ViolationSeverity::Severe
            };

            Some(RangeViolation {
                tensor_name: name.to_string(),
                expected_range: expected,
                actual_range: actual,
                severity,
                timestamp: std::time::Instant::now(),
            })
        } else {
            None
        }
    }

    /// Update stability metric for a tensor
    fn update_stability_metric(&mut self, name: &str) {
        if let Some(ranges) = self.tensor_ranges.get(name) {
            if ranges.len() < 2 {
                return;
            }

            // Calculate coefficient of variation for range stability
            let range_sizes: Vec<f32> = ranges.iter().map(|(min, max)| max - min).collect();
            let mean_range = range_sizes.iter().sum::<f32>() / range_sizes.len() as f32;
            let variance = range_sizes
                .iter()
                .map(|&x| (x - mean_range).powi(2))
                .sum::<f32>()
                / range_sizes.len() as f32;
            let std_dev = variance.sqrt();

            let stability = if mean_range > 0.0 {
                1.0 - (std_dev / mean_range) // Higher is more stable
            } else {
                0.0
            };

            self.stability_metrics.insert(name.to_string(), stability);
        }
    }

    /// Get stability metric for a tensor
    pub fn get_stability(&self, name: &str) -> f32 {
        self.stability_metrics.get(name).copied().unwrap_or(0.0)
    }
}

impl Default for OverflowDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl OverflowDetector {
    /// Create new overflow detector
    pub fn new() -> Self {
        Self {
            overflow_events: Vec::new(),
            underflow_events: Vec::new(),
            overflow_threshold: 1e6,   // Default overflow threshold
            underflow_threshold: -1e6, // Default underflow threshold
            detection_enabled: true,
        }
    }

    /// Set overflow thresholds
    pub fn set_thresholds(&mut self, overflow: f32, underflow: f32) {
        self.overflow_threshold = overflow;
        self.underflow_threshold = underflow;
    }

    /// Detect overflow/underflow in tensor
    pub fn detect_overflow(&mut self, name: &str, tensor: &Tensor) -> TorshResult<()> {
        if !self.detection_enabled {
            return Ok(());
        }

        let data = tensor.data()?;
        let binding = tensor.shape();
        let shape = binding.dims();

        for (i, &value) in data.iter().enumerate() {
            if value > self.overflow_threshold {
                let position = self.linear_to_nd_index(i, shape);
                self.overflow_events.push(OverflowEvent {
                    tensor_name: name.to_string(),
                    overflow_value: value,
                    threshold: self.overflow_threshold,
                    num_elements: data
                        .iter()
                        .filter(|&&x| x > self.overflow_threshold)
                        .count(),
                    position,
                    timestamp: std::time::Instant::now(),
                });
                break;
            }

            if value < self.underflow_threshold {
                let position = self.linear_to_nd_index(i, shape);
                self.underflow_events.push(OverflowEvent {
                    tensor_name: name.to_string(),
                    overflow_value: value,
                    threshold: self.underflow_threshold,
                    num_elements: data
                        .iter()
                        .filter(|&&x| x < self.underflow_threshold)
                        .count(),
                    position,
                    timestamp: std::time::Instant::now(),
                });
                break;
            }
        }

        Ok(())
    }

    /// Convert linear index to n-dimensional index
    fn linear_to_nd_index(&self, linear_idx: usize, shape: &[usize]) -> Vec<usize> {
        let mut indices = Vec::with_capacity(shape.len());
        let mut remaining = linear_idx;

        for &dim_size in shape.iter().rev() {
            indices.push(remaining % dim_size);
            remaining /= dim_size;
        }

        indices.reverse();
        indices
    }

    /// Enable or disable overflow detection
    pub fn set_detection_enabled(&mut self, enabled: bool) {
        self.detection_enabled = enabled;
    }

    /// Clear all overflow events
    pub fn clear_events(&mut self) {
        self.overflow_events.clear();
        self.underflow_events.clear();
    }

    /// Get overflow event count
    pub fn get_overflow_count(&self) -> usize {
        self.overflow_events.len()
    }

    /// Get underflow event count
    pub fn get_underflow_count(&self) -> usize {
        self.underflow_events.len()
    }
}

/// Comparison tools for different quantization schemes
#[derive(Debug)]
pub struct QuantizationComparator {
    /// Comparison results
    pub comparison_results: Vec<ComparisonResult>,
    /// Reference tensor (original)
    pub reference: Option<Tensor>,
}

/// Comparison result between two quantization schemes
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    /// Scheme A name
    pub scheme_a: String,
    /// Scheme B name
    pub scheme_b: String,
    /// Error metrics for scheme A
    pub metrics_a: ErrorMetrics,
    /// Error metrics for scheme B
    pub metrics_b: ErrorMetrics,
    /// Winner (better scheme)
    pub winner: String,
    /// Improvement metrics
    pub improvement: ImprovementMetrics,
}

/// Improvement metrics between schemes
#[derive(Debug, Clone)]
pub struct ImprovementMetrics {
    /// MAE improvement percentage
    pub mae_improvement: f32,
    /// SNR improvement in dB
    pub snr_improvement: f32,
    /// PSNR improvement in dB
    pub psnr_improvement: f32,
    /// Cosine similarity improvement
    pub cosine_improvement: f32,
}

impl QuantizationComparator {
    /// Create new comparator
    pub fn new() -> Self {
        Self {
            comparison_results: Vec::new(),
            reference: None,
        }
    }

    /// Set reference tensor
    pub fn set_reference(&mut self, reference: Tensor) {
        self.reference = Some(reference);
    }

    /// Compare two quantization schemes
    pub fn compare_schemes(
        &mut self,
        scheme_a_name: &str,
        quantized_a: &Tensor,
        scheme_b_name: &str,
        quantized_b: &Tensor,
    ) -> TorshResult<ComparisonResult> {
        let reference = self
            .reference
            .as_ref()
            .ok_or_else(|| TorshError::InvalidArgument("Reference tensor not set".to_string()))?;

        // Compute error metrics for both schemes
        let metrics_a = self.compute_error_metrics(reference, quantized_a)?;
        let metrics_b = self.compute_error_metrics(reference, quantized_b)?;

        // Determine winner based on multiple criteria
        let winner = self.determine_winner(&metrics_a, &metrics_b, scheme_a_name, scheme_b_name);

        // Calculate improvement metrics
        let improvement = ImprovementMetrics {
            mae_improvement: ((metrics_a.mae - metrics_b.mae) / metrics_a.mae) * 100.0,
            snr_improvement: metrics_b.snr - metrics_a.snr,
            psnr_improvement: metrics_b.psnr - metrics_a.psnr,
            cosine_improvement: metrics_b.cosine_similarity - metrics_a.cosine_similarity,
        };

        let result = ComparisonResult {
            scheme_a: scheme_a_name.to_string(),
            scheme_b: scheme_b_name.to_string(),
            metrics_a,
            metrics_b,
            winner,
            improvement,
        };

        self.comparison_results.push(result.clone());
        Ok(result)
    }

    /// Compute error metrics (similar to debugger implementation)
    fn compute_error_metrics(
        &self,
        original: &Tensor,
        quantized: &Tensor,
    ) -> TorshResult<ErrorMetrics> {
        let orig_data = original.data()?;
        let quant_data = quantized.data()?;

        if orig_data.len() != quant_data.len() {
            return Err(TorshError::InvalidArgument(
                "Tensor size mismatch".to_string(),
            ));
        }

        let n = orig_data.len() as f32;

        let mae = orig_data
            .iter()
            .zip(quant_data.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f32>()
            / n;

        let mse = orig_data
            .iter()
            .zip(quant_data.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            / n;

        let rmse = mse.sqrt();

        let signal_power = orig_data.iter().map(|&x| x.powi(2)).sum::<f32>() / n;
        let snr = if mse > 0.0 {
            10.0 * (signal_power / mse).log10()
        } else {
            f32::INFINITY
        };

        let max_val = orig_data.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let psnr = if mse > 0.0 {
            20.0 * (max_val / rmse).log10()
        } else {
            f32::INFINITY
        };

        let dot_product = orig_data
            .iter()
            .zip(quant_data.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f32>();

        let orig_norm = orig_data.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt();
        let quant_norm = quant_data.iter().map(|&x| x.powi(2)).sum::<f32>().sqrt();

        let cosine_similarity = if orig_norm > 0.0 && quant_norm > 0.0 {
            dot_product / (orig_norm * quant_norm)
        } else {
            0.0
        };

        Ok(ErrorMetrics {
            mae,
            mse,
            rmse,
            snr,
            psnr,
            cosine_similarity,
        })
    }

    /// Determine winner based on multiple criteria
    fn determine_winner(
        &self,
        metrics_a: &ErrorMetrics,
        metrics_b: &ErrorMetrics,
        name_a: &str,
        name_b: &str,
    ) -> String {
        let mut score_a = 0;
        let mut score_b = 0;

        // Lower MAE is better
        if metrics_a.mae < metrics_b.mae {
            score_a += 1;
        } else {
            score_b += 1;
        }

        // Higher SNR is better
        if metrics_a.snr > metrics_b.snr {
            score_a += 1;
        } else {
            score_b += 1;
        }

        // Higher PSNR is better
        if metrics_a.psnr > metrics_b.psnr {
            score_a += 1;
        } else {
            score_b += 1;
        }

        // Higher cosine similarity is better
        if metrics_a.cosine_similarity > metrics_b.cosine_similarity {
            score_a += 1;
        } else {
            score_b += 1;
        }

        if score_a > score_b {
            name_a.to_string()
        } else if score_b > score_a {
            name_b.to_string()
        } else {
            "Tie".to_string()
        }
    }

    /// Generate comparison report
    pub fn generate_comparison_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== QUANTIZATION SCHEME COMPARISON REPORT ===\n\n");

        for (i, result) in self.comparison_results.iter().enumerate() {
            report.push_str(&format!(
                "Comparison {}: {} vs {}\n",
                i + 1,
                result.scheme_a,
                result.scheme_b
            ));
            report.push_str(&format!("Winner: {}\n", result.winner));
            report.push_str(&format!(
                "MAE Improvement: {:.2}%\n",
                result.improvement.mae_improvement
            ));
            report.push_str(&format!(
                "SNR Improvement: {:.2} dB\n",
                result.improvement.snr_improvement
            ));
            report.push_str(&format!(
                "PSNR Improvement: {:.2} dB\n",
                result.improvement.psnr_improvement
            ));
            report.push_str(&format!(
                "Cosine Similarity Improvement: {:.4}\n",
                result.improvement.cosine_improvement
            ));
            report.push('\n');
        }

        report
    }
}

impl Default for QuantizationComparator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_quantization_debugger() {
        let mut debugger = QuantizationDebugger::new();
        assert!(debugger.debug_enabled);

        let input = tensor_1d(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let output = tensor_1d(&[1.1, 2.1, 2.9, 3.9]).unwrap();
        let config = QuantConfig::int8();

        debugger
            .debug_quantization("test_op", &input, &output, &config, 0.1, 0)
            .unwrap();

        assert_eq!(debugger.execution_trace.len(), 1);
        assert_eq!(debugger.error_stats.total_ops, 1);

        let report = debugger.generate_report();
        assert!(report.contains("QUANTIZATION DEBUG REPORT"));
        assert!(report.contains("Total Operations: 1"));

        debugger.clear();
        assert_eq!(debugger.execution_trace.len(), 0);
    }

    #[test]
    fn test_error_statistics() {
        let mut stats = ErrorStatistics::new();
        assert_eq!(stats.total_ops, 0);

        let metrics = ErrorMetrics {
            mae: 0.1,
            mse: 0.01,
            rmse: 0.1,
            snr: 20.0,
            psnr: 30.0,
            cosine_similarity: 0.95,
        };

        stats.update(&metrics, "test_layer");

        assert_eq!(stats.total_ops, 1);
        assert_eq!(stats.cumulative_mae, 0.1);
        assert!(stats.layer_errors.contains_key("test_layer"));

        let (avg_mae, avg_mse) = stats.get_averages();
        assert_eq!(avg_mae, 0.1);
        assert_eq!(avg_mse, 0.01);
    }

    #[test]
    fn test_range_tracker() {
        let mut tracker = RangeTracker::new();

        let tensor = tensor_1d(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        tracker.track_range("test_tensor", &tensor).unwrap();

        assert!(tracker.tensor_ranges.contains_key("test_tensor"));
        assert_eq!(tracker.tensor_ranges["test_tensor"].len(), 1);
        assert_eq!(tracker.tensor_ranges["test_tensor"][0], (1.0, 4.0));

        // Set expected range and test violation detection
        tracker.set_expected_range("test_tensor", (0.0, 3.0));

        let tensor2 = tensor_1d(&[0.0, 1.0, 2.0, 5.0]).unwrap(); // Max violates expected range
        tracker.track_range("test_tensor", &tensor2).unwrap();

        assert!(!tracker.range_violations.is_empty());
        assert_eq!(tracker.range_violations[0].tensor_name, "test_tensor");
    }

    #[test]
    fn test_overflow_detector() {
        let mut detector = OverflowDetector::new();
        assert!(detector.detection_enabled);

        detector.set_thresholds(10.0, -10.0);

        let tensor = tensor_1d(&[1.0, 2.0, 15.0, 4.0]).unwrap(); // 15.0 exceeds threshold
        detector.detect_overflow("test_tensor", &tensor).unwrap();

        assert_eq!(detector.get_overflow_count(), 1);
        assert_eq!(detector.overflow_events[0].tensor_name, "test_tensor");
        assert_eq!(detector.overflow_events[0].overflow_value, 15.0);

        detector.clear_events();
        assert_eq!(detector.get_overflow_count(), 0);
    }

    #[test]
    fn test_quantization_comparator() {
        let mut comparator = QuantizationComparator::new();

        let reference = tensor_1d(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        comparator.set_reference(reference);

        let quantized_a = tensor_1d(&[1.1, 2.1, 2.9, 3.9]).unwrap();
        let quantized_b = tensor_1d(&[1.0, 2.0, 3.0, 4.0]).unwrap();

        let result = comparator
            .compare_schemes("INT8", &quantized_a, "FP32", &quantized_b)
            .unwrap();

        assert_eq!(result.scheme_a, "INT8");
        assert_eq!(result.scheme_b, "FP32");
        assert_eq!(result.winner, "FP32"); // Perfect match should win

        assert_eq!(comparator.comparison_results.len(), 1);

        let report = comparator.generate_comparison_report();
        assert!(report.contains("QUANTIZATION SCHEME COMPARISON REPORT"));
        assert!(report.contains("INT8 vs FP32"));
    }

    #[test]
    fn test_debug_step() {
        let input_stats = TensorStatistics {
            min: 1.0,
            max: 4.0,
            mean: 2.5,
            std: 1.29,
            shape: vec![4],
            num_elements: 4,
            dtype: DType::F32,
        };

        let output_stats = input_stats.clone();

        let quant_params = QuantParams {
            scale: 0.1,
            zero_point: 0,
            scheme: QScheme::PerTensorAffine,
            qint_range: (-128, 127),
        };

        let error_metrics = ErrorMetrics {
            mae: 0.1,
            mse: 0.01,
            rmse: 0.1,
            snr: 20.0,
            psnr: 30.0,
            cosine_similarity: 0.95,
        };

        let debug_step = DebugStep {
            name: "test_step".to_string(),
            input_stats,
            output_stats,
            quant_params,
            error_metrics,
            timestamp: std::time::Instant::now(),
        };

        assert_eq!(debug_step.name, "test_step");
        assert_eq!(debug_step.quant_params.scale, 0.1);
        assert_eq!(debug_step.error_metrics.mae, 0.1);
    }

    #[test]
    fn test_violation_severity() {
        assert_eq!(ViolationSeverity::Minor, ViolationSeverity::Minor);
        assert_ne!(ViolationSeverity::Minor, ViolationSeverity::Severe);
    }
}
