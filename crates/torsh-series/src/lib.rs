//! Time series analysis components for ToRSh
//!
//! This module provides PyTorch-compatible time series models and operations,
//! built on top of SciRS2's time series decomposition and state-space methods.

pub mod anomaly;
pub mod decomposition;
pub mod forecast;
pub mod state_space;
pub mod utils;

use torsh_tensor::Tensor;
// scirs2_series will be used when specific APIs are available

/// Time series data container
#[derive(Debug, Clone)]
pub struct TimeSeries {
    /// Data values (time x features)
    pub values: Tensor,
    /// Timestamps (optional)
    pub timestamps: Option<Vec<f64>>,
    /// Sampling frequency
    pub frequency: Option<f64>,
    /// Feature names
    pub features: Option<Vec<String>>,
}

impl TimeSeries {
    /// Create a new time series
    pub fn new(values: Tensor) -> Self {
        Self {
            values,
            timestamps: None,
            frequency: None,
            features: None,
        }
    }

    /// Set timestamps
    pub fn with_timestamps(mut self, timestamps: Vec<f64>) -> Self {
        self.timestamps = Some(timestamps);
        self
    }

    /// Set sampling frequency
    pub fn with_frequency(mut self, frequency: f64) -> Self {
        self.frequency = Some(frequency);
        self
    }

    /// Set feature names
    pub fn with_features(mut self, features: Vec<String>) -> Self {
        self.features = Some(features);
        self
    }

    /// Get number of time points
    pub fn len(&self) -> usize {
        self.values.shape().dims()[0]
    }

    /// Check if series is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        let shape = self.values.shape();
        let dims = shape.dims();
        if dims.len() > 1 {
            dims[1]
        } else {
            1
        }
    }

    /// Slice time series
    pub fn slice(
        &self,
        start: usize,
        end: usize,
    ) -> Result<TimeSeries, torsh_core::error::TorshError> {
        // Validate bounds
        if start > end || end > self.len() {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Invalid slice bounds: start={}, end={}, len={}",
                start,
                end,
                self.len()
            )));
        }

        // Create a sliced tensor by extracting elements using flat indexing
        let slice_len = end - start;
        let num_features = self.num_features();

        // Extract data for the time slice
        let mut sliced_data = Vec::with_capacity(slice_len * num_features);
        for t in start..end {
            for f in 0..num_features {
                let flat_idx = t * num_features + f;
                let value = self.values.get_item_flat(flat_idx).map_err(|e| {
                    torsh_core::error::TorshError::InvalidArgument(format!(
                        "Failed to slice tensor: {}",
                        e
                    ))
                })?;
                sliced_data.push(value);
            }
        }

        let shape = if num_features == 1 {
            vec![slice_len]
        } else {
            vec![slice_len, num_features]
        };
        let values = Tensor::from_vec(sliced_data, &shape).map_err(|e| {
            torsh_core::error::TorshError::InvalidArgument(format!(
                "Failed to create sliced tensor: {}",
                e
            ))
        })?;

        let timestamps = self.timestamps.as_ref().map(|ts| ts[start..end].to_vec());

        Ok(TimeSeries {
            values,
            timestamps,
            frequency: self.frequency,
            features: self.features.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::creation::*;

    #[test]
    fn test_timeseries_creation() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_vec(data, &[5]).unwrap();
        let ts = TimeSeries::new(tensor);

        assert_eq!(ts.len(), 5);
        assert_eq!(ts.num_features(), 1);
        assert!(!ts.is_empty());
    }

    #[test]
    fn test_timeseries_with_metadata() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_vec(data, &[5]).unwrap();
        let timestamps = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let features = vec!["value".to_string()];

        let ts = TimeSeries::new(tensor)
            .with_timestamps(timestamps.clone())
            .with_frequency(1.0)
            .with_features(features.clone());

        assert_eq!(ts.timestamps, Some(timestamps));
        assert_eq!(ts.frequency, Some(1.0));
        assert_eq!(ts.features, Some(features));
    }

    #[test]
    fn test_timeseries_multivariate() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, &[3, 2]).unwrap();
        let ts = TimeSeries::new(tensor);

        assert_eq!(ts.len(), 3);
        assert_eq!(ts.num_features(), 2);
    }

    #[test]
    fn test_timeseries_slice() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_vec(data, &[5]).unwrap();
        let ts = TimeSeries::new(tensor);

        let sliced = ts.slice(1, 4).unwrap();
        assert_eq!(sliced.len(), 3);
    }

    #[test]
    fn test_empty_timeseries() {
        let tensor = zeros(&[0]).unwrap();
        let ts = TimeSeries::new(tensor);

        assert_eq!(ts.len(), 0);
        assert!(ts.is_empty());
    }
}
