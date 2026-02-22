//! Observer implementations for quantization parameter calibration
//!
//! This module provides various observer types for collecting statistics from tensors
//! during the calibration phase of quantization. Observers track tensor distributions
//! and calculate optimal quantization parameters.
//!
//! # Features
//!
//! - **MinMax Observer**: Simple min/max range tracking
//! - **MovingAverage Observer**: Exponential moving average of ranges
//! - **Histogram Observer**: Distribution-based quantization with outlier removal
//! - **Percentile Observer**: Percentile-based range estimation
//! - **Parallel Processing**: Optimized for large tensors using Rayon
//! - **Outlier Detection**: IQR-based outlier detection and removal
//! - **Memory Management**: Efficient memory usage for large datasets

use crate::config::ObserverType;

#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{collections::BTreeMap as HashMap, string::String, vec::Vec};

use torsh_core::{
    dtype::DType,
    error::{Result as TorshResult, TorshError},
};
use torsh_tensor::Tensor;

/// Observer for tracking tensor statistics during calibration
#[derive(Debug)]
pub struct Observer {
    observer_type: ObserverType,
    min_val: f32,
    max_val: f32,
    num_batches: usize,
    // For moving average observer
    #[allow(dead_code)]
    avg_min: f32,
    #[allow(dead_code)]
    avg_max: f32,
    // For histogram observer
    histogram: Vec<usize>,
    hist_min: f32,
    hist_max: f32,
    num_bins: usize,
    // For percentile observer
    values: Vec<f32>,
    percentile: f32,
}

impl Observer {
    /// Create a new observer
    pub fn new(observer_type: ObserverType) -> Self {
        Self {
            observer_type,
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            num_batches: 0,
            avg_min: 0.0,
            avg_max: 0.0,
            histogram: vec![0; 256], // Default 256 bins
            hist_min: f32::INFINITY,
            hist_max: f32::NEG_INFINITY,
            num_bins: 256,
            values: Vec::new(),
            percentile: 99.99, // Default percentile
        }
    }

    /// Create a new histogram observer with specified number of bins
    pub fn new_histogram(num_bins: usize) -> Self {
        Self {
            observer_type: ObserverType::Histogram,
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            num_batches: 0,
            avg_min: 0.0,
            avg_max: 0.0,
            histogram: vec![0; num_bins],
            hist_min: f32::INFINITY,
            hist_max: f32::NEG_INFINITY,
            num_bins,
            values: Vec::new(),
            percentile: 99.99,
        }
    }

    /// Create a new percentile observer with specified percentile
    pub fn new_percentile(percentile: f32) -> Self {
        Self {
            observer_type: ObserverType::Percentile,
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            num_batches: 0,
            avg_min: 0.0,
            avg_max: 0.0,
            histogram: Vec::new(),
            hist_min: f32::INFINITY,
            hist_max: f32::NEG_INFINITY,
            num_bins: 0,
            values: Vec::new(),
            percentile,
        }
    }

    /// Update observer with new tensor (optimized with parallel processing)
    pub fn update(&mut self, tensor: &Tensor) -> TorshResult<()> {
        let data = tensor.data()?;

        // Always count as a batch, even if data is empty
        self.num_batches += 1;

        if data.is_empty() {
            return Ok(());
        }

        // Validate data for NaN/infinity
        if data.iter().any(|&x| !x.is_finite()) {
            return Err(TorshError::InvalidArgument(
                "Tensor contains non-finite values (NaN or infinity)".to_string(),
            ));
        }

        // Use parallel processing for large tensors
        let (batch_min, batch_max) = if data.len() > 10000 {
            #[cfg(feature = "std")]
            {
                use scirs2_core::parallel_ops::*;
                data.par_iter().map(|&x| (x, x)).reduce(
                    || (f32::INFINITY, f32::NEG_INFINITY),
                    |(min1, max1), (min2, max2)| (min1.min(min2), max1.max(max2)),
                )
            }
            #[cfg(not(feature = "std"))]
            {
                data.iter()
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &val| {
                        (min.min(val), max.max(val))
                    })
            }
        } else {
            data.iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &val| {
                    (min.min(val), max.max(val))
                })
        };

        match self.observer_type {
            ObserverType::MinMax => {
                self.min_val = self.min_val.min(batch_min);
                self.max_val = self.max_val.max(batch_max);
            }
            ObserverType::MovingAverage => {
                if self.num_batches == 0 {
                    self.min_val = batch_min;
                    self.max_val = batch_max;
                    self.avg_min = batch_min;
                    self.avg_max = batch_max;
                } else {
                    let alpha = 0.01; // Moving average factor
                    self.avg_min = alpha * batch_min + (1.0 - alpha) * self.avg_min;
                    self.avg_max = alpha * batch_max + (1.0 - alpha) * self.avg_max;
                    // Keep global min/max for reference
                    self.min_val = self.min_val.min(batch_min);
                    self.max_val = self.max_val.max(batch_max);
                }
            }
            ObserverType::Histogram => {
                // Update global min/max first
                self.min_val = self.min_val.min(batch_min);
                self.max_val = self.max_val.max(batch_max);

                // Update histogram range if this is the first batch
                if self.num_batches == 0 {
                    self.hist_min = batch_min;
                    self.hist_max = batch_max;
                } else {
                    self.hist_min = self.hist_min.min(batch_min);
                    self.hist_max = self.hist_max.max(batch_max);
                }

                // Add values to histogram with improved binning
                if data.len() > 5000 {
                    // Use parallel histogram update for large tensors
                    #[cfg(feature = "std")]
                    {
                        use scirs2_core::parallel_ops::*;
                        let local_histograms: Vec<Vec<usize>> = data
                            .par_chunks(1000)
                            .map(|chunk| {
                                let mut local_hist = vec![0; self.num_bins];
                                for &value in chunk {
                                    let bin_idx = self.value_to_bin_index(value);
                                    if bin_idx < local_hist.len() {
                                        local_hist[bin_idx] += 1;
                                    }
                                }
                                local_hist
                            })
                            .collect();

                        // Merge local histograms
                        for local_hist in local_histograms {
                            for (i, count) in local_hist.iter().enumerate() {
                                self.histogram[i] += count;
                            }
                        }
                    }
                    #[cfg(not(feature = "std"))]
                    {
                        for &value in data.iter() {
                            let bin_idx = self.value_to_bin_index(value);
                            if bin_idx < self.histogram.len() {
                                self.histogram[bin_idx] += 1;
                            }
                        }
                    }
                } else {
                    for &value in data.iter() {
                        let bin_idx = self.value_to_bin_index(value);
                        if bin_idx < self.histogram.len() {
                            self.histogram[bin_idx] += 1;
                        }
                    }
                }
            }
            ObserverType::Percentile => {
                // Update global min/max
                self.min_val = self.min_val.min(batch_min);
                self.max_val = self.max_val.max(batch_max);

                // Limit memory usage for percentile calculation
                if self.values.len() + data.len() > 100_000 {
                    // Sample the data to avoid memory explosion
                    let sample_rate = 100_000.0 / (self.values.len() + data.len()) as f32;
                    let sampled_data: Vec<f32> = data
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| (*i as f32 * sample_rate) % 1.0 < sample_rate)
                        .map(|(_, &val)| val)
                        .collect();
                    self.values.extend(sampled_data);
                } else {
                    self.values.extend(data.iter().cloned());
                }
            }
            _ => {
                // For other observer types, fall back to min-max
                self.min_val = self.min_val.min(batch_min);
                self.max_val = self.max_val.max(batch_max);
            }
        }

        Ok(())
    }

    /// Calculate quantization parameters from observed statistics
    pub fn calculate_qparams(&self, dtype: DType) -> TorshResult<(f32, i32)> {
        let (qmin, qmax) = match dtype {
            DType::I8 => (-128, 127),
            DType::U8 => (0, 255),
            _ => {
                return Err(TorshError::InvalidArgument(
                    "Unsupported quantization dtype".to_string(),
                ))
            }
        };

        // Use observer-specific range calculation
        let (min_val, max_val) = match self.observer_type {
            ObserverType::Histogram => {
                if !self.histogram.is_empty() {
                    self.calculate_histogram_range()
                } else {
                    (self.min_val.min(0.0), self.max_val.max(0.0))
                }
            }
            ObserverType::Percentile => {
                if !self.values.is_empty() {
                    self.calculate_percentile_range()
                } else {
                    (self.min_val.min(0.0), self.max_val.max(0.0))
                }
            }
            _ => (self.min_val.min(0.0), self.max_val.max(0.0)),
        };

        let scale = (max_val - min_val) / (qmax - qmin) as f32;
        let scale = if scale == 0.0 { 1.0 } else { scale };

        let zero_point = (qmin as f32 - min_val / scale)
            .round()
            .max(qmin as f32)
            .min(qmax as f32) as i32;

        Ok((scale, zero_point))
    }

    /// Convert a value to histogram bin index with improved stability
    fn value_to_bin_index(&self, value: f32) -> usize {
        // Use hist_min/hist_max for more accurate binning
        let range_min = if self.hist_min.is_finite() {
            self.hist_min
        } else {
            self.min_val
        };
        let range_max = if self.hist_max.is_finite() {
            self.hist_max
        } else {
            self.max_val
        };

        if range_max <= range_min || !value.is_finite() {
            return 0;
        }

        let ratio = ((value - range_min) / (range_max - range_min)).clamp(0.0, 1.0);
        let idx = (ratio * self.num_bins as f32).floor() as usize;
        idx.min(self.num_bins - 1)
    }

    /// Calculate optimal range from histogram with enhanced outlier removal
    fn calculate_histogram_range(&self) -> (f32, f32) {
        if self.histogram.is_empty() || self.num_bins == 0 {
            return (self.min_val, self.max_val);
        }

        let total_samples: usize = self.histogram.iter().sum();
        if total_samples == 0 {
            return (self.min_val, self.max_val);
        }

        // Use adaptive threshold based on data distribution
        let outlier_threshold = if total_samples > 10000 {
            0.001 // 0.1% for large datasets
        } else if total_samples > 1000 {
            0.005 // 0.5% for medium datasets
        } else {
            0.01 // 1% for small datasets
        };

        let threshold_count = (total_samples as f32 * outlier_threshold) as usize;
        let mut cumsum = 0;
        let mut start_bin = 0;
        let mut end_bin = self.num_bins - 1;

        // Find start bin (skip outliers from the beginning)
        for (i, &count) in self.histogram.iter().enumerate() {
            cumsum += count;
            if cumsum > threshold_count {
                start_bin = i;
                break;
            }
        }

        // Find end bin (skip outliers from the end)
        cumsum = 0;
        for (i, &count) in self.histogram.iter().enumerate().rev() {
            cumsum += count;
            if cumsum > threshold_count {
                end_bin = i;
                break;
            }
        }

        // Ensure we have a valid range
        if start_bin >= end_bin {
            return (self.min_val, self.max_val);
        }

        let range_min = if self.hist_min.is_finite() {
            self.hist_min
        } else {
            self.min_val
        };
        let range_max = if self.hist_max.is_finite() {
            self.hist_max
        } else {
            self.max_val
        };

        if range_max <= range_min {
            return (self.min_val, self.max_val);
        }

        let bin_width = (range_max - range_min) / self.num_bins as f32;
        let min_val = range_min + start_bin as f32 * bin_width;
        let max_val = range_min + (end_bin + 1) as f32 * bin_width;

        // Ensure the calculated range is valid
        if min_val >= max_val {
            (self.min_val, self.max_val)
        } else {
            (min_val.max(self.min_val), max_val.min(self.max_val))
        }
    }

    /// Calculate percentile-based range
    fn calculate_percentile_range(&self) -> (f32, f32) {
        if self.values.is_empty() {
            return (self.min_val, self.max_val);
        }

        let mut sorted_values = self.values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

        let n = sorted_values.len();
        let lower_percentile = 100.0 - self.percentile;
        let upper_percentile = self.percentile;

        let lower_idx = ((lower_percentile / 100.0) * n as f32) as usize;
        let upper_idx = ((upper_percentile / 100.0) * n as f32) as usize;

        let lower_idx = lower_idx.min(n - 1);
        let upper_idx = upper_idx.min(n - 1);

        (sorted_values[lower_idx], sorted_values[upper_idx])
    }

    /// Detect and remove outliers using IQR method
    pub fn detect_outliers(&self, data: &[f32], factor: f32) -> (Vec<f32>, usize) {
        if data.is_empty() {
            return (Vec::new(), 0);
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

        let n = sorted_data.len();

        // Use proper percentile calculation for quartiles
        let q1 = if n >= 4 {
            let idx = (n as f32 * 0.25) as usize;
            if idx > 0 {
                sorted_data[idx.min(n - 1)]
            } else {
                sorted_data[0]
            }
        } else {
            sorted_data[0]
        };

        let q3 = if n >= 4 {
            let idx = (n as f32 * 0.75) as usize;
            sorted_data[idx.min(n - 1)]
        } else {
            sorted_data[n - 1]
        };

        let iqr = q3 - q1;

        // If IQR is too small, use a more conservative approach
        if iqr < 1e-6 {
            return (sorted_data, 0);
        }

        let lower_bound = q1 - factor * iqr;
        let upper_bound = q3 + factor * iqr;

        let original_len = data.len();
        let cleaned_data: Vec<f32> = data
            .iter()
            .filter(|&&x| x >= lower_bound && x <= upper_bound)
            .cloned()
            .collect();

        let outliers_removed = original_len - cleaned_data.len();

        (cleaned_data, outliers_removed)
    }

    /// Get comprehensive statistics from the observer
    pub fn get_statistics(&self) -> HashMap<String, f32> {
        let mut stats = HashMap::new();

        stats.insert("min_val".to_string(), self.min_val);
        stats.insert("max_val".to_string(), self.max_val);
        stats.insert("range".to_string(), self.max_val - self.min_val);
        stats.insert("num_batches".to_string(), self.num_batches as f32);

        match self.observer_type {
            ObserverType::Histogram => {
                stats.insert("num_bins".to_string(), self.num_bins as f32);
                stats.insert(
                    "total_samples".to_string(),
                    self.histogram.iter().sum::<usize>() as f32,
                );
                if !self.histogram.is_empty() {
                    let max_bin_count = *self.histogram.iter().max().unwrap_or(&0);
                    stats.insert("max_bin_count".to_string(), max_bin_count as f32);
                }
            }
            ObserverType::Percentile => {
                stats.insert("total_values".to_string(), self.values.len() as f32);
                stats.insert("percentile".to_string(), self.percentile);
            }
            _ => {}
        }

        stats
    }

    /// Get the observer type
    pub fn observer_type(&self) -> ObserverType {
        self.observer_type
    }

    /// Get the current min/max values
    pub fn get_min_max(&self) -> (f32, f32) {
        (self.min_val, self.max_val)
    }

    /// Get number of processed batches
    pub fn num_batches(&self) -> usize {
        self.num_batches
    }

    /// Reset the observer state
    pub fn reset(&mut self) {
        self.min_val = f32::INFINITY;
        self.max_val = f32::NEG_INFINITY;
        self.num_batches = 0;
        self.avg_min = 0.0;
        self.avg_max = 0.0;
        self.hist_min = f32::INFINITY;
        self.hist_max = f32::NEG_INFINITY;
        self.histogram.iter_mut().for_each(|x| *x = 0);
        self.values.clear();
    }
}

/// Factory functions for creating observers
impl Observer {
    /// Create a MinMax observer
    pub fn min_max() -> Self {
        Self::new(ObserverType::MinMax)
    }

    /// Create a MovingAverage observer
    pub fn moving_average() -> Self {
        Self::new(ObserverType::MovingAverage)
    }

    /// Create a Histogram observer with default bins
    pub fn histogram() -> Self {
        Self::new(ObserverType::Histogram)
    }

    /// Create a Histogram observer with custom number of bins
    pub fn histogram_with_bins(num_bins: usize) -> Self {
        Self::new_histogram(num_bins)
    }

    /// Create a Percentile observer with default percentile
    pub fn percentile() -> Self {
        Self::new(ObserverType::Percentile)
    }

    /// Create a Percentile observer with custom percentile
    pub fn percentile_with_value(percentile: f32) -> Self {
        Self::new_percentile(percentile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_observer_creation() {
        let minmax_observer = Observer::min_max();
        assert_eq!(minmax_observer.observer_type(), ObserverType::MinMax);

        let histogram_observer = Observer::histogram_with_bins(128);
        assert_eq!(histogram_observer.observer_type(), ObserverType::Histogram);
        assert_eq!(histogram_observer.num_bins, 128);

        let percentile_observer = Observer::percentile_with_value(95.0);
        assert_eq!(
            percentile_observer.observer_type(),
            ObserverType::Percentile
        );
        assert_eq!(percentile_observer.percentile, 95.0);
    }

    #[test]
    fn test_minmax_observer() {
        let mut observer = Observer::min_max();

        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let tensor1 = tensor_1d(&data1).unwrap();
        observer.update(&tensor1).unwrap();

        let (min, max) = observer.get_min_max();
        assert_eq!(min, 1.0);
        assert_eq!(max, 4.0);

        let data2 = vec![0.5, 5.0];
        let tensor2 = tensor_1d(&data2).unwrap();
        observer.update(&tensor2).unwrap();

        let (min, max) = observer.get_min_max();
        assert_eq!(min, 0.5);
        assert_eq!(max, 5.0);
    }

    #[test]
    fn test_histogram_observer() {
        let mut observer = Observer::histogram_with_bins(10);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let tensor = tensor_1d(&data).unwrap();
        observer.update(&tensor).unwrap();

        let stats = observer.get_statistics();
        assert_eq!(stats.get("total_samples"), Some(&5.0));
        assert_eq!(stats.get("num_bins"), Some(&10.0));
    }

    #[test]
    fn test_percentile_observer() {
        let mut observer = Observer::percentile_with_value(90.0);

        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let tensor = tensor_1d(&data).unwrap();
        observer.update(&tensor).unwrap();

        let stats = observer.get_statistics();
        assert_eq!(stats.get("total_values"), Some(&100.0));
        assert_eq!(stats.get("percentile"), Some(&90.0));
    }

    #[test]
    fn test_calculate_qparams() {
        let mut observer = Observer::min_max();

        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let tensor = tensor_1d(&data).unwrap();
        observer.update(&tensor).unwrap();

        let (scale, zero_point) = observer.calculate_qparams(DType::I8).unwrap();
        assert!(scale > 0.0);
        assert!(zero_point >= -128 && zero_point <= 127);
    }

    #[test]
    fn test_outlier_detection() {
        let observer = Observer::min_max();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100.0 is an outlier

        let (cleaned_data, outliers_removed) = observer.detect_outliers(&data, 1.5);
        assert!(outliers_removed > 0);
        assert!(cleaned_data.len() < data.len());
        assert!(!cleaned_data.contains(&100.0));
    }

    #[test]
    fn test_observer_reset() {
        let mut observer = Observer::min_max();

        let data = vec![1.0, 2.0, 3.0];
        let tensor = tensor_1d(&data).unwrap();
        observer.update(&tensor).unwrap();

        assert_eq!(observer.num_batches(), 1);

        observer.reset();
        assert_eq!(observer.num_batches(), 0);

        let (min, max) = observer.get_min_max();
        assert!(min.is_infinite() && min > 0.0);
        assert!(max.is_infinite() && max < 0.0);
    }

    #[test]
    fn test_invalid_tensor_data() {
        let mut observer = Observer::min_max();

        let data = vec![f32::NAN, 1.0, 2.0];
        let tensor = tensor_1d(&data).unwrap();

        let result = observer.update(&tensor);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_tensor() {
        let mut observer = Observer::min_max();

        let data: Vec<f32> = vec![];
        let tensor = tensor_1d(&data).unwrap();

        let result = observer.update(&tensor);
        assert!(result.is_ok());
        assert_eq!(observer.num_batches(), 1);
    }

    #[test]
    fn test_unsupported_dtype() {
        let mut observer = Observer::min_max();

        let data = vec![1.0, 2.0, 3.0];
        let tensor = tensor_1d(&data).unwrap();
        observer.update(&tensor).unwrap();

        let result = observer.calculate_qparams(DType::F32);
        assert!(result.is_err());
    }
}
