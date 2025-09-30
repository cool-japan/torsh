//! Data preprocessing utilities for clustering

use crate::error::{ClusterError, ClusterResult};
use torsh_tensor::Tensor;

/// Data preprocessing methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreprocessingMethod {
    /// Standardize features to zero mean and unit variance
    StandardScaler,
    /// Scale features to [0, 1] range
    MinMaxScaler,
    /// Normalize to unit norm
    Normalizer,
    /// Robust scaling using median and IQR
    RobustScaler,
}

/// Standardize features to have zero mean and unit variance
pub fn standardize_features(data: &Tensor) -> ClusterResult<Tensor> {
    let n_samples = data.shape().dims()[0];
    let n_features = data.shape().dims()[1];
    let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;

    let mut standardized_data = Vec::with_capacity(data_vec.len());

    // Compute mean and std for each feature
    for j in 0..n_features {
        let mut feature_values = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            feature_values.push(data_vec[i * n_features + j] as f64);
        }

        // Compute mean
        let mean = feature_values.iter().sum::<f64>() / n_samples as f64;

        // Compute standard deviation
        let variance = feature_values
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / n_samples as f64;
        let std = variance.sqrt();

        // Standardize feature values
        for &feature_val in feature_values.iter() {
            let standardized_val = if std > 0.0 {
                (feature_val - mean) / std
            } else {
                0.0
            };
            standardized_data.push(standardized_val as f32);
        }
    }

    // Reorder data back to row-major format
    let mut reordered_data = vec![0.0; n_samples * n_features];
    for i in 0..n_samples {
        for j in 0..n_features {
            reordered_data[i * n_features + j] = standardized_data[j * n_samples + i];
        }
    }

    Tensor::from_vec(reordered_data, &[n_samples, n_features]).map_err(ClusterError::TensorError)
}

/// Normalize features to [0, 1] range
pub fn normalize_features(data: &Tensor) -> ClusterResult<Tensor> {
    let n_samples = data.shape().dims()[0];
    let n_features = data.shape().dims()[1];
    let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;

    let mut normalized_data = Vec::with_capacity(data_vec.len());

    // Compute min and max for each feature
    for j in 0..n_features {
        let mut feature_values = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            feature_values.push(data_vec[i * n_features + j] as f64);
        }

        let min_val = feature_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = feature_values
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        // Normalize feature values
        for &feature_val in feature_values.iter() {
            let normalized_val = if range > 0.0 {
                (feature_val - min_val) / range
            } else {
                0.0
            };
            normalized_data.push(normalized_val as f32);
        }
    }

    // Reorder data back to row-major format
    let mut reordered_data = vec![0.0; n_samples * n_features];
    for i in 0..n_samples {
        for j in 0..n_features {
            reordered_data[i * n_features + j] = normalized_data[j * n_samples + i];
        }
    }

    Tensor::from_vec(reordered_data, &[n_samples, n_features]).map_err(ClusterError::TensorError)
}

/// Normalize each sample to unit norm
pub fn unit_normalize(data: &Tensor) -> ClusterResult<Tensor> {
    let n_samples = data.shape().dims()[0];
    let n_features = data.shape().dims()[1];
    let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;

    let mut normalized_data = Vec::with_capacity(data_vec.len());

    for i in 0..n_samples {
        // Compute L2 norm for this sample
        let mut norm = 0.0;
        for j in 0..n_features {
            let val = data_vec[i * n_features + j] as f64;
            norm += val * val;
        }
        norm = norm.sqrt();

        // Normalize the sample
        for j in 0..n_features {
            let normalized_val = if norm > 0.0 {
                data_vec[i * n_features + j] as f64 / norm
            } else {
                0.0
            };
            normalized_data.push(normalized_val as f32);
        }
    }

    Tensor::from_vec(normalized_data, &[n_samples, n_features]).map_err(ClusterError::TensorError)
}

/// Apply robust scaling using median and interquartile range
pub fn robust_scale(data: &Tensor) -> ClusterResult<Tensor> {
    let n_samples = data.shape().dims()[0];
    let n_features = data.shape().dims()[1];
    let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;

    let mut scaled_data = Vec::with_capacity(data_vec.len());

    // Compute median and IQR for each feature
    for j in 0..n_features {
        let mut feature_values = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            feature_values.push(data_vec[i * n_features + j] as f64);
        }

        // Sort to compute quantiles
        feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if n_samples % 2 == 0 {
            (feature_values[n_samples / 2 - 1] + feature_values[n_samples / 2]) / 2.0
        } else {
            feature_values[n_samples / 2]
        };

        let q1_idx = n_samples / 4;
        let q3_idx = 3 * n_samples / 4;
        let iqr = feature_values[q3_idx] - feature_values[q1_idx];

        // Scale feature values
        for i in 0..n_samples {
            let original_val = data_vec[i * n_features + j] as f64;
            let scaled_val = if iqr > 0.0 {
                (original_val - median) / iqr
            } else {
                0.0
            };
            scaled_data.push(scaled_val as f32);
        }
    }

    // Reorder data back to row-major format
    let mut reordered_data = vec![0.0; n_samples * n_features];
    for i in 0..n_samples {
        for j in 0..n_features {
            reordered_data[i * n_features + j] = scaled_data[j * n_samples + i];
        }
    }

    Tensor::from_vec(reordered_data, &[n_samples, n_features]).map_err(ClusterError::TensorError)
}
