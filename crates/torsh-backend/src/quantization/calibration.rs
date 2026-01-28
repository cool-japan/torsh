//! Quantization calibration methods and utilities
//!
//! This module provides sophisticated calibration techniques for determining
//! optimal quantization parameters from sample data. It supports various
//! calibration methods including statistical approaches, entropy-based methods,
//! and error minimization techniques.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::quantization::{QuantizationParams, QuantizationScheme, QuantizedDType};
use crate::{BackendResult, Device};
use std::collections::HashMap;
use std::sync::Arc;
use torsh_core::error::TorshError;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

/// Quantization calibration utility
///
/// The calibrator analyzes sample data to determine optimal quantization
/// parameters that balance accuracy and performance. It supports multiple
/// calibration methods to suit different use cases and accuracy requirements.
#[derive(Debug, Clone)]
pub struct QuantizationCalibrator {
    /// Sample data for calibration
    samples: Vec<Vec<f32>>,
    /// Calibration method to use
    method: CalibrationMethod,
    /// Device for calibration computations
    device: Device,
    /// Cache for previously computed parameters
    parameter_cache: HashMap<String, QuantizationParams>,
}

/// Calibration methods for quantization parameter optimization
///
/// Different calibration methods offer trade-offs between computational cost,
/// robustness to outliers, and final quantization accuracy.
#[derive(Debug)]
pub enum CalibrationMethod {
    /// Simple min-max calibration
    ///
    /// Uses the minimum and maximum values in the data to set quantization
    /// range. Fast but sensitive to outliers.
    MinMax,

    /// Percentile-based calibration
    ///
    /// Uses a specified percentile to clip outliers before determining range.
    /// More robust than min-max but requires tuning the percentile parameter.
    Percentile(f32),

    /// Entropy-based calibration (KL divergence minimization)
    ///
    /// Minimizes the KL divergence between original and quantized distributions.
    /// Provides good accuracy but is computationally expensive.
    Entropy,

    /// Mean squared error minimization
    ///
    /// Finds parameters that minimize MSE between original and quantized values.
    /// Good balance between accuracy and computational cost.
    MSE,

    /// Adaptive method selection
    ///
    /// Automatically selects the best method based on data characteristics.
    /// Uses multiple methods and picks the one with best validation score.
    Adaptive,

    /// Custom calibration with user-defined function
    ///
    /// Allows users to provide their own calibration logic for specialized
    /// use cases or domain-specific optimization.
    Custom(Arc<dyn CalibrationFunction>),
}

impl Clone for CalibrationMethod {
    fn clone(&self) -> Self {
        match self {
            CalibrationMethod::MinMax => CalibrationMethod::MinMax,
            CalibrationMethod::Percentile(percentile) => CalibrationMethod::Percentile(*percentile),
            CalibrationMethod::Entropy => CalibrationMethod::Entropy,
            CalibrationMethod::MSE => CalibrationMethod::MSE,
            CalibrationMethod::Adaptive => CalibrationMethod::Adaptive,
            CalibrationMethod::Custom(func) => CalibrationMethod::Custom(Arc::clone(func)),
        }
    }
}

/// Trait for custom calibration functions
pub trait CalibrationFunction: Send + Sync + std::fmt::Debug {
    /// Compute quantization parameters from sample data
    fn calibrate(
        &self,
        samples: &[Vec<f32>],
        dtype: QuantizedDType,
    ) -> BackendResult<QuantizationParams>;
}

impl QuantizationCalibrator {
    /// Create a new calibrator with the specified method
    ///
    /// # Arguments
    ///
    /// * `method` - The calibration method to use
    /// * `device` - Device for performing calibration computations
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use torsh_backend::quantization::calibration::{QuantizationCalibrator, CalibrationMethod};
    /// use torsh_core::DeviceType;
    ///
    /// let device = DeviceType::Cpu;
    /// let calibrator = QuantizationCalibrator::new(CalibrationMethod::MinMax, device);
    /// ```
    pub fn new(method: CalibrationMethod, device: Device) -> Self {
        Self {
            samples: Vec::new(),
            method,
            device,
            parameter_cache: HashMap::new(),
        }
    }

    /// Add calibration sample
    ///
    /// Adds a sample of data that will be used to determine optimal
    /// quantization parameters. More samples generally lead to better
    /// parameter estimation.
    ///
    /// # Arguments
    ///
    /// * `data` - Sample data vector
    pub fn add_sample(&mut self, data: Vec<f32>) {
        self.samples.push(data);
    }

    /// Add multiple calibration samples at once
    pub fn add_samples(&mut self, samples: Vec<Vec<f32>>) {
        self.samples.extend(samples);
    }

    /// Clear all calibration samples
    pub fn clear_samples(&mut self) {
        self.samples.clear();
        self.parameter_cache.clear();
    }

    /// Get the number of calibration samples
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Set the calibration method
    pub fn set_method(&mut self, method: CalibrationMethod) {
        self.method = method;
        self.parameter_cache.clear(); // Clear cache when method changes
    }

    /// Compute optimal quantization parameters
    ///
    /// Analyzes all collected samples to determine the best quantization
    /// parameters for the specified data type using the configured method.
    ///
    /// # Arguments
    ///
    /// * `dtype` - Target quantization data type
    ///
    /// # Returns
    ///
    /// Optimized quantization parameters
    ///
    /// # Errors
    ///
    /// Returns an error if no samples have been added or if calibration fails
    pub fn calibrate(&self, dtype: QuantizedDType) -> BackendResult<QuantizationParams> {
        if self.samples.is_empty() {
            return Err(TorshError::BackendError(
                "No samples available for calibration".to_string(),
            ));
        }

        // Check cache first
        let cache_key = format!("{:?}_{:?}", dtype, self.method);
        if let Some(cached_params) = self.parameter_cache.get(&cache_key) {
            return Ok(cached_params.clone());
        }

        // Perform calibration based on method
        let params = match &self.method {
            CalibrationMethod::MinMax => self.calibrate_minmax(dtype),
            CalibrationMethod::Percentile(percentile) => {
                self.calibrate_percentile(dtype, *percentile)
            }
            CalibrationMethod::Entropy => self.calibrate_entropy(dtype),
            CalibrationMethod::MSE => self.calibrate_mse(dtype),
            CalibrationMethod::Adaptive => self.calibrate_adaptive(dtype),
            CalibrationMethod::Custom(func) => func.calibrate(&self.samples, dtype),
        };

        params
    }

    /// Min-max calibration implementation
    fn calibrate_minmax(&self, dtype: QuantizedDType) -> BackendResult<QuantizationParams> {
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        // Find global min and max across all samples
        for sample in &self.samples {
            for &val in sample {
                if val.is_finite() {
                    min_val = min_val.min(val);
                    max_val = max_val.max(val);
                }
            }
        }

        if min_val.is_infinite() || max_val.is_infinite() {
            return Err(TorshError::BackendError(
                "No finite values found in calibration data".to_string(),
            ));
        }

        let mut params = QuantizationParams {
            dtype,
            scheme: QuantizationScheme::Asymmetric,
            scale: vec![1.0],
            zero_point: vec![0],
            block_size: None,
            min_val: Some(min_val),
            max_val: Some(max_val),
        };

        params.from_statistics(min_val, max_val)?;
        Ok(params)
    }

    /// Percentile-based calibration implementation
    fn calibrate_percentile(
        &self,
        dtype: QuantizedDType,
        percentile: f32,
    ) -> BackendResult<QuantizationParams> {
        if !(0.0..=100.0).contains(&percentile) {
            return Err(TorshError::BackendError(
                "Percentile must be between 0 and 100".to_string(),
            ));
        }

        // Collect all values
        let mut all_values = Vec::new();
        for sample in &self.samples {
            for &val in sample {
                if val.is_finite() {
                    all_values.push(val);
                }
            }
        }

        if all_values.is_empty() {
            return Err(TorshError::BackendError(
                "No finite values found in calibration data".to_string(),
            ));
        }

        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate percentile bounds
        let lower_percentile = (100.0 - percentile) / 2.0;
        let upper_percentile = (100.0 + percentile) / 2.0;

        let lower_idx = ((lower_percentile / 100.0) * (all_values.len() - 1) as f32) as usize;
        let upper_idx = ((upper_percentile / 100.0) * (all_values.len() - 1) as f32) as usize;

        let min_val = all_values[lower_idx];
        let max_val = all_values[upper_idx];

        let mut params = QuantizationParams {
            dtype,
            scheme: if min_val >= 0.0 {
                QuantizationScheme::Asymmetric
            } else {
                QuantizationScheme::Symmetric
            },
            scale: vec![1.0],
            zero_point: vec![0],
            block_size: None,
            min_val: Some(min_val),
            max_val: Some(max_val),
        };

        params.from_statistics(min_val, max_val)?;
        Ok(params)
    }

    /// Entropy-based calibration (KL divergence minimization)
    fn calibrate_entropy(&self, dtype: QuantizedDType) -> BackendResult<QuantizationParams> {
        // Collect all values for histogram computation
        let mut all_values = Vec::new();
        for sample in &self.samples {
            for &val in sample {
                if val.is_finite() {
                    all_values.push(val);
                }
            }
        }

        if all_values.is_empty() {
            return Err(TorshError::BackendError(
                "No finite values found for entropy calibration".to_string(),
            ));
        }

        // Find reasonable initial bounds
        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let global_min = all_values[0];
        let global_max = all_values[all_values.len() - 1];

        // Try different clipping thresholds and find the one with minimum KL divergence
        let mut best_kl_div = f64::INFINITY;
        let mut best_min = global_min;
        let mut best_max = global_max;

        // Search over different percentile thresholds
        for percentile in [90.0, 95.0, 97.0, 99.0, 99.5, 99.9, 100.0] {
            let threshold_idx = ((percentile / 100.0) * (all_values.len() - 1) as f32) as usize;
            let threshold_max = all_values[threshold_idx];
            let threshold_min = -threshold_max; // Symmetric for simplicity

            // Compute KL divergence for this threshold
            if let Ok(kl_div) =
                self.compute_kl_divergence(&all_values, threshold_min, threshold_max, &dtype)
            {
                if kl_div < best_kl_div {
                    best_kl_div = kl_div;
                    best_min = threshold_min;
                    best_max = threshold_max;
                }
            }
        }

        let mut params = QuantizationParams {
            dtype,
            scheme: QuantizationScheme::Symmetric,
            scale: vec![1.0],
            zero_point: vec![0],
            block_size: None,
            min_val: Some(best_min),
            max_val: Some(best_max),
        };

        params.from_statistics(best_min, best_max)?;
        Ok(params)
    }

    /// MSE-based calibration implementation
    fn calibrate_mse(&self, dtype: QuantizedDType) -> BackendResult<QuantizationParams> {
        // Collect all values
        let mut all_values = Vec::new();
        for sample in &self.samples {
            for &val in sample {
                if val.is_finite() {
                    all_values.push(val);
                }
            }
        }

        if all_values.is_empty() {
            return Err(TorshError::BackendError(
                "No finite values found for MSE calibration".to_string(),
            ));
        }

        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let global_min = all_values[0];
        let global_max = all_values[all_values.len() - 1];

        let mut best_mse = f64::INFINITY;
        let mut best_min = global_min;
        let mut best_max = global_max;

        // Grid search over different clipping thresholds
        for percentile in [95.0, 97.0, 99.0, 99.5, 99.9, 100.0] {
            let threshold_idx = ((percentile / 100.0) * (all_values.len() - 1) as f32) as usize;
            let threshold_max = all_values[threshold_idx];
            let threshold_min = if global_min >= 0.0 {
                0.0
            } else {
                -threshold_max
            };

            // Compute MSE for this threshold
            if let Ok(mse) = self.compute_mse(&all_values, threshold_min, threshold_max, &dtype) {
                if mse < best_mse {
                    best_mse = mse;
                    best_min = threshold_min;
                    best_max = threshold_max;
                }
            }
        }

        let mut params = QuantizationParams {
            dtype,
            scheme: if best_min >= 0.0 {
                QuantizationScheme::Asymmetric
            } else {
                QuantizationScheme::Symmetric
            },
            scale: vec![1.0],
            zero_point: vec![0],
            block_size: None,
            min_val: Some(best_min),
            max_val: Some(best_max),
        };

        params.from_statistics(best_min, best_max)?;
        Ok(params)
    }

    /// Adaptive calibration that tries multiple methods
    fn calibrate_adaptive(&self, dtype: QuantizedDType) -> BackendResult<QuantizationParams> {
        // Try different methods and evaluate their quality
        let methods = vec![
            CalibrationMethod::MinMax,
            CalibrationMethod::Percentile(99.0),
            CalibrationMethod::Percentile(95.0),
            CalibrationMethod::MSE,
        ];

        let mut best_score = f64::INFINITY;
        let mut best_params = None;

        for method in methods {
            // Create temporary calibrator with this method
            let mut temp_calibrator = self.clone();
            temp_calibrator.set_method(method);

            if let Ok(params) = temp_calibrator.calibrate(dtype.clone()) {
                // Evaluate quality of these parameters
                if let Ok(score) = self.evaluate_quantization_quality(&params) {
                    if score < best_score {
                        best_score = score;
                        best_params = Some(params);
                    }
                }
            }
        }

        best_params.ok_or_else(|| {
            TorshError::BackendError(
                "No suitable quantization parameters found in adaptive mode".to_string(),
            )
        })
    }

    /// Compute KL divergence between original and quantized distributions
    fn compute_kl_divergence(
        &self,
        values: &[f32],
        min_val: f32,
        max_val: f32,
        dtype: &QuantizedDType,
    ) -> BackendResult<f64> {
        const NUM_BINS: usize = 256;

        // Create histogram of original values
        let mut original_hist = vec![0usize; NUM_BINS];
        let range = max_val - min_val;

        if range <= 0.0 {
            return Ok(f64::INFINITY);
        }

        for &val in values {
            let clipped_val = val.clamp(min_val, max_val);
            let bin = ((clipped_val - min_val) / range * (NUM_BINS - 1) as f32) as usize;
            let bin = bin.min(NUM_BINS - 1);
            original_hist[bin] += 1;
        }

        // Simulate quantization and create quantized histogram
        let mut quantized_hist = vec![0usize; NUM_BINS];
        let (qmin, qmax) = dtype.value_range();
        let scale = range / (qmax - qmin) as f32;

        for &val in values {
            let clipped_val = val.clamp(min_val, max_val);
            // Simulate quantization
            let quantized = ((clipped_val - min_val) / scale)
                .round()
                .clamp(qmin as f32, qmax as f32);
            let dequantized = quantized * scale + min_val;

            let bin = ((dequantized - min_val) / range * (NUM_BINS - 1) as f32) as usize;
            let bin = bin.min(NUM_BINS - 1);
            quantized_hist[bin] += 1;
        }

        // Compute KL divergence
        let total_samples = values.len() as f64;
        let mut kl_div = 0.0;

        for i in 0..NUM_BINS {
            let p = (original_hist[i] as f64 + 1e-10) / total_samples; // Add small epsilon
            let q = (quantized_hist[i] as f64 + 1e-10) / total_samples;

            if p > 0.0 && q > 0.0 {
                kl_div += p * (p / q).ln();
            }
        }

        Ok(kl_div)
    }

    /// Compute MSE between original and quantized values
    fn compute_mse(
        &self,
        values: &[f32],
        min_val: f32,
        max_val: f32,
        dtype: &QuantizedDType,
    ) -> BackendResult<f64> {
        let (qmin, qmax) = dtype.value_range();
        let range = max_val - min_val;

        if range <= 0.0 {
            return Ok(f64::INFINITY);
        }

        let scale = range / (qmax - qmin) as f32;
        let mut total_error = 0.0;

        for &val in values {
            let clipped_val = val.clamp(min_val, max_val);
            // Simulate quantization
            let quantized = ((clipped_val - min_val) / scale)
                .round()
                .clamp(qmin as f32, qmax as f32);
            let dequantized = quantized * scale + min_val;

            let error = (val - dequantized).powi(2);
            total_error += error as f64;
        }

        Ok(total_error / values.len() as f64)
    }

    /// Evaluate the quality of quantization parameters
    fn evaluate_quantization_quality(&self, params: &QuantizationParams) -> BackendResult<f64> {
        // Use a subset of samples for evaluation to avoid overfitting
        let eval_samples = if self.samples.len() > 1000 {
            &self.samples[..1000]
        } else {
            &self.samples
        };

        let mut total_error = 0.0;
        let mut total_count = 0;

        for sample in eval_samples {
            for &val in sample {
                if !val.is_finite() {
                    continue;
                }

                // Simulate quantization
                let scale = params.scale[0];
                let zero_point = params.zero_point[0] as f32;
                let (qmin, qmax) = params.dtype.value_range();

                let quantized = ((val / scale + zero_point)
                    .round()
                    .clamp(qmin as f32, qmax as f32)) as i32;
                let dequantized = (quantized - params.zero_point[0]) as f32 * scale;

                let error = (val - dequantized).powi(2);
                total_error += error as f64;
                total_count += 1;
            }
        }

        if total_count == 0 {
            Ok(f64::INFINITY)
        } else {
            Ok(total_error / total_count as f64)
        }
    }
}

/// Percentile-based calibration method
///
/// A specialized calibrator that focuses on percentile-based methods
/// with additional features for robust outlier handling.
#[derive(Debug, Clone)]
pub struct PercentileCalibrator {
    /// Percentile threshold for calibration
    pub percentile: f32,
    /// Whether to use symmetric clipping
    pub symmetric: bool,
    /// Device for calibration computations
    device: Device,
}

impl PercentileCalibrator {
    /// Create a new percentile calibrator
    ///
    /// # Arguments
    ///
    /// * `percentile` - Percentile threshold (0-100)
    /// * `symmetric` - Whether to use symmetric clipping around zero
    /// * `device` - Device for computations
    pub fn new(percentile: f32, symmetric: bool, device: Device) -> BackendResult<Self> {
        if !(0.0..=100.0).contains(&percentile) {
            return Err(TorshError::BackendError(
                "Percentile must be between 0 and 100".to_string(),
            ));
        }

        Ok(Self {
            percentile,
            symmetric,
            device,
        })
    }

    /// Calibrate using percentile method with enhanced outlier detection
    pub fn calibrate_percentile(
        &self,
        samples: &[Vec<f32>],
        dtype: QuantizedDType,
    ) -> BackendResult<QuantizationParams> {
        // Collect all values from samples
        let mut all_values = Vec::new();
        for sample in samples {
            for &val in sample {
                if val.is_finite() {
                    all_values.push(val);
                }
            }
        }

        if all_values.is_empty() {
            return Err(TorshError::BackendError(
                "No finite values found in calibration data".to_string(),
            ));
        }

        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let (min_val, max_val) = if self.symmetric {
            // Symmetric percentile clipping
            let threshold_idx =
                ((self.percentile / 100.0) * (all_values.len() - 1) as f32) as usize;
            let max_abs = all_values[threshold_idx]
                .abs()
                .max(all_values[all_values.len() - 1 - threshold_idx].abs());
            (-max_abs, max_abs)
        } else {
            // Asymmetric percentile clipping
            let lower_percentile = (100.0 - self.percentile) / 2.0;
            let upper_percentile = (100.0 + self.percentile) / 2.0;

            let lower_idx = ((lower_percentile / 100.0) * (all_values.len() - 1) as f32) as usize;
            let upper_idx = ((upper_percentile / 100.0) * (all_values.len() - 1) as f32) as usize;

            (all_values[lower_idx], all_values[upper_idx])
        };

        let mut params = QuantizationParams {
            dtype,
            scheme: if self.symmetric {
                QuantizationScheme::Symmetric
            } else {
                QuantizationScheme::Asymmetric
            },
            scale: vec![1.0],
            zero_point: vec![0],
            block_size: None,
            min_val: Some(min_val),
            max_val: Some(max_val),
        };

        params.from_statistics(min_val, max_val)?;
        Ok(params)
    }

    /// Calibrate with entropy validation
    ///
    /// Uses percentile clipping but validates the result using entropy measures
    /// to ensure the clipping doesn't lose too much information.
    pub fn calibrate_entropy_validated(
        &self,
        samples: &[Vec<f32>],
        dtype: QuantizedDType,
        max_entropy_loss: f64,
    ) -> BackendResult<QuantizationParams> {
        // Try different percentile values and pick the highest one that
        // doesn't exceed the entropy loss threshold
        let mut best_params = None;
        let mut _best_percentile = 0.0;

        for test_percentile in [50.0, 70.0, 80.0, 90.0, 95.0, 97.0, 99.0, 99.5] {
            if test_percentile > self.percentile {
                break;
            }

            let mut temp_calibrator = self.clone();
            temp_calibrator.percentile = test_percentile;

            if let Ok(params) = temp_calibrator.calibrate_percentile(samples, dtype.clone()) {
                // Estimate entropy loss (simplified)
                let entropy_loss = self.estimate_entropy_loss(samples, &params)?;

                if entropy_loss <= max_entropy_loss {
                    best_params = Some(params);
                    _best_percentile = test_percentile;
                }
            }
        }

        best_params.ok_or_else(|| {
            TorshError::BackendError(format!(
                "No percentile found that meets entropy loss requirement of {}",
                max_entropy_loss
            ))
        })
    }

    /// Estimate entropy loss from quantization
    fn estimate_entropy_loss(
        &self,
        samples: &[Vec<f32>],
        params: &QuantizationParams,
    ) -> BackendResult<f64> {
        // Simplified entropy loss estimation
        // In practice, would compute actual entropy of original vs quantized data
        let min_val = params.min_val.expect("min_val should be set in params");
        let max_val = params.max_val.expect("max_val should be set in params");

        let mut clipped_count = 0;
        let mut total_count = 0;

        for sample in samples {
            for &val in sample {
                if val.is_finite() {
                    total_count += 1;
                    if val < min_val || val > max_val {
                        clipped_count += 1;
                    }
                }
            }
        }

        if total_count == 0 {
            return Ok(0.0);
        }

        // Simple approximation: entropy loss ≈ fraction of clipped values
        Ok(clipped_count as f64 / total_count as f64)
    }
}

/// Calibration statistics and analysis
#[derive(Debug, Clone)]
pub struct CalibrationStatistics {
    /// Total number of samples processed
    pub num_samples: usize,
    /// Total number of values processed
    pub num_values: usize,
    /// Minimum value encountered
    pub min_value: f32,
    /// Maximum value encountered
    pub max_value: f32,
    /// Mean value
    pub mean_value: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Percentage of outliers (beyond 3 standard deviations)
    pub outlier_percentage: f32,
    /// Recommended calibration methods
    pub recommended_methods: Vec<CalibrationMethod>,
}

impl CalibrationStatistics {
    /// Compute statistics from calibration samples
    pub fn from_samples(samples: &[Vec<f32>]) -> BackendResult<Self> {
        let mut all_values = Vec::new();
        for sample in samples {
            for &val in sample {
                if val.is_finite() {
                    all_values.push(val);
                }
            }
        }

        if all_values.is_empty() {
            return Err(TorshError::BackendError(
                "No finite values found in samples".to_string(),
            ));
        }

        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let num_values = all_values.len();
        let min_value = all_values[0];
        let max_value = all_values[num_values - 1];

        // Compute mean
        let sum: f64 = all_values.iter().map(|&x| x as f64).sum();
        let mean_value = (sum / num_values as f64) as f32;

        // Compute standard deviation
        let variance: f64 = all_values
            .iter()
            .map(|&x| (x as f64 - mean_value as f64).powi(2))
            .sum::<f64>()
            / num_values as f64;
        let std_dev = variance.sqrt() as f32;

        // Count outliers (beyond 3 standard deviations)
        let outlier_threshold = 3.0 * std_dev;
        let outlier_count = all_values
            .iter()
            .filter(|&&x| (x - mean_value).abs() > outlier_threshold)
            .count();
        let outlier_percentage = (outlier_count as f32 / num_values as f32) * 100.0;

        // Recommend calibration methods based on statistics
        let recommended_methods =
            Self::recommend_methods(outlier_percentage, std_dev, min_value, max_value);

        Ok(Self {
            num_samples: samples.len(),
            num_values,
            min_value,
            max_value,
            mean_value,
            std_dev,
            outlier_percentage,
            recommended_methods,
        })
    }

    /// Recommend calibration methods based on data characteristics
    fn recommend_methods(
        outlier_percentage: f32,
        std_dev: f32,
        min_value: f32,
        max_value: f32,
    ) -> Vec<CalibrationMethod> {
        let mut recommendations = Vec::new();

        // If many outliers, recommend percentile methods
        if outlier_percentage > 5.0 {
            recommendations.push(CalibrationMethod::Percentile(99.0));
            recommendations.push(CalibrationMethod::Percentile(95.0));
        }

        // If high variance, recommend entropy-based methods
        if std_dev > (max_value - min_value) * 0.2 {
            recommendations.push(CalibrationMethod::Entropy);
            recommendations.push(CalibrationMethod::MSE);
        }

        // Always include adaptive as a fallback
        recommendations.push(CalibrationMethod::Adaptive);

        // If no outliers and low variance, min-max is fine
        if outlier_percentage < 1.0 && std_dev < (max_value - min_value) * 0.1 {
            recommendations.push(CalibrationMethod::MinMax);
        }

        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_samples() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 4.0, 6.0, 8.0, 10.0],
            vec![-1.0, -2.0, 0.0, 1.0, 2.0],
        ]
    }

    #[test]
    fn test_calibrator_creation() {
        let device = Device::cpu().unwrap();
        let calibrator = QuantizationCalibrator::new(CalibrationMethod::MinMax, device);

        assert_eq!(calibrator.num_samples(), 0);
        assert!(matches!(calibrator.method, CalibrationMethod::MinMax));
    }

    #[test]
    fn test_sample_management() {
        let device = Device::cpu().unwrap();
        let mut calibrator = QuantizationCalibrator::new(CalibrationMethod::MinMax, device);

        calibrator.add_sample(vec![1.0, 2.0, 3.0]);
        assert_eq!(calibrator.num_samples(), 1);

        calibrator.add_samples(vec![vec![4.0, 5.0], vec![6.0, 7.0]]);
        assert_eq!(calibrator.num_samples(), 3);

        calibrator.clear_samples();
        assert_eq!(calibrator.num_samples(), 0);
    }

    #[test]
    fn test_minmax_calibration() {
        let device = Device::cpu().unwrap();
        let mut calibrator = QuantizationCalibrator::new(CalibrationMethod::MinMax, device);

        let samples = create_test_samples();
        calibrator.add_samples(samples);

        let result = calibrator.calibrate(QuantizedDType::Int8);
        assert!(result.is_ok());

        let params = result.unwrap();
        assert_eq!(params.dtype, QuantizedDType::Int8);
        assert!(params.scale[0] > 0.0);
        assert!(params.min_val.is_some());
        assert!(params.max_val.is_some());
    }

    #[test]
    fn test_percentile_calibration() {
        let device = Device::cpu().unwrap();
        let mut calibrator =
            QuantizationCalibrator::new(CalibrationMethod::Percentile(95.0), device);

        let samples = create_test_samples();
        calibrator.add_samples(samples);

        let result = calibrator.calibrate(QuantizedDType::UInt8);
        assert!(result.is_ok());

        let params = result.unwrap();
        assert_eq!(params.dtype, QuantizedDType::UInt8);
    }

    #[test]
    fn test_mse_calibration() {
        let device = Device::cpu().unwrap();
        let mut calibrator = QuantizationCalibrator::new(CalibrationMethod::MSE, device);

        let samples = create_test_samples();
        calibrator.add_samples(samples);

        let result = calibrator.calibrate(QuantizedDType::Int8);
        assert!(result.is_ok());
    }

    #[test]
    fn test_adaptive_calibration() {
        let device = Device::cpu().unwrap();
        let mut calibrator = QuantizationCalibrator::new(CalibrationMethod::Adaptive, device);

        let samples = create_test_samples();
        calibrator.add_samples(samples);

        let result = calibrator.calibrate(QuantizedDType::Int8);
        assert!(result.is_ok());
    }

    #[test]
    fn test_calibration_with_outliers() {
        let device = Device::cpu().unwrap();
        let mut calibrator =
            QuantizationCalibrator::new(CalibrationMethod::Percentile(90.0), device);

        // Add samples with outliers - using more samples for better percentile calculation
        let samples = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 1000.0], // 1000.0 is an outlier
            vec![2.0, 4.0, 6.0, 8.0, 10.0],
            vec![-1.0, -2.0, 0.0, 1.0, 2.0, -1000.0], // -1000.0 is an outlier
            vec![1.5, 2.5, 3.5, 4.5, 5.5],
            vec![0.5, 1.0, 1.5, 2.0, 2.5],
            vec![3.0, 3.5, 4.0, 4.5, 5.0],
            vec![-0.5, -1.0, 0.5, 1.0, 1.5],
        ];
        calibrator.add_samples(samples);

        let result = calibrator.calibrate(QuantizedDType::Int8);
        assert!(result.is_ok());

        let params = result.unwrap();

        // Percentile method should handle outliers better than min-max
        assert!(params.min_val.unwrap() > -100.0); // More reasonable bound with more data
        assert!(params.max_val.unwrap() < 100.0);
    }

    #[test]
    fn test_percentile_calibrator() {
        let device = Device::cpu().unwrap();
        let calibrator = PercentileCalibrator::new(95.0, false, device);
        assert!(calibrator.is_ok());

        let calibrator = calibrator.unwrap();
        let samples = create_test_samples();

        let result = calibrator.calibrate_percentile(&samples, QuantizedDType::Int8);
        assert!(result.is_ok());

        let params = result.unwrap();
        assert_eq!(params.dtype, QuantizedDType::Int8);
        assert_eq!(params.scheme, QuantizationScheme::Asymmetric);
    }

    #[test]
    fn test_symmetric_percentile_calibrator() {
        let device = Device::cpu().unwrap();
        let calibrator = PercentileCalibrator::new(95.0, true, device).unwrap();
        let samples = create_test_samples();

        let result = calibrator.calibrate_percentile(&samples, QuantizedDType::Int8);
        assert!(result.is_ok());

        let params = result.unwrap();
        assert_eq!(params.scheme, QuantizationScheme::Symmetric);
    }

    #[test]
    fn test_calibration_statistics() {
        let samples = create_test_samples();
        let stats = CalibrationStatistics::from_samples(&samples);
        assert!(stats.is_ok());

        let stats = stats.unwrap();
        assert_eq!(stats.num_samples, 3);
        assert_eq!(stats.num_values, 15);
        assert!(stats.min_value <= stats.max_value);
        assert!(stats.std_dev >= 0.0);
        assert!(!stats.recommended_methods.is_empty());
    }

    #[test]
    fn test_invalid_percentile() {
        let device = Device::cpu().unwrap();

        // Test invalid percentile values
        let result = PercentileCalibrator::new(101.0, false, device.clone());
        assert!(result.is_err());

        let result = PercentileCalibrator::new(-1.0, false, device);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_samples_error() {
        let device = Device::cpu().unwrap();
        let calibrator = QuantizationCalibrator::new(CalibrationMethod::MinMax, device);

        let result = calibrator.calibrate(QuantizedDType::Int8);
        assert!(result.is_err());
    }

    #[test]
    fn test_method_switching() {
        let device = Device::cpu().unwrap();
        let mut calibrator = QuantizationCalibrator::new(CalibrationMethod::MinMax, device);

        calibrator.add_samples(create_test_samples());

        // Test switching methods
        calibrator.set_method(CalibrationMethod::Percentile(95.0));
        let result1 = calibrator.calibrate(QuantizedDType::Int8);
        assert!(result1.is_ok());

        calibrator.set_method(CalibrationMethod::MSE);
        let result2 = calibrator.calibrate(QuantizedDType::Int8);
        assert!(result2.is_ok());

        // Results might be different due to different methods
        let params1 = result1.unwrap();
        let params2 = result2.unwrap();
        // Both should be valid but may have different parameters
        assert!(params1.scale[0] > 0.0);
        assert!(params2.scale[0] > 0.0);
    }

    #[test]
    fn test_calibration_with_infinite_values() {
        let device = Device::cpu().unwrap();
        let mut calibrator = QuantizationCalibrator::new(CalibrationMethod::MinMax, device);

        // Add samples with infinite values (should be filtered out)
        let samples = vec![
            vec![1.0, 2.0, f32::INFINITY, 4.0, 5.0],
            vec![2.0, f32::NEG_INFINITY, 6.0, 8.0, 10.0],
            vec![-1.0, -2.0, 0.0, 1.0, f32::NAN],
        ];
        calibrator.add_samples(samples);

        let result = calibrator.calibrate(QuantizedDType::Int8);
        assert!(result.is_ok());

        let params = result.unwrap();
        assert!(params.min_val.unwrap().is_finite());
        assert!(params.max_val.unwrap().is_finite());
    }
}
