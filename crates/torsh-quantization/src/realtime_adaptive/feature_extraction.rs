//! Feature extraction for tensor analysis
//!
//! This module provides comprehensive feature extraction capabilities including
//! statistical, spectral, and spatial features for adaptive quantization optimization.

use super::config::FeatureExtractor;
use crate::TorshResult;
use torsh_tensor::Tensor;

impl FeatureExtractor {
    /// Extract comprehensive features from tensor
    pub fn extract_features(&self, tensor: &Tensor) -> TorshResult<Vec<f32>> {
        let data = tensor.data()?;
        let mut features = Vec::new();

        // Extract statistical features
        if self.enable_statistical {
            features.extend(self.extract_statistical_features(&data)?);
        }

        // Extract spectral features
        if self.enable_spectral {
            features.extend(self.extract_spectral_features(&data)?);
        }

        // Extract spatial features
        if self.enable_spatial {
            features.extend(self.extract_spatial_features(&data, tensor.shape().dims())?);
        }

        // Pad or truncate to fixed size (16 features)
        let target_size = 16;
        if features.len() < target_size {
            features.resize(target_size, 0.0);
        } else if features.len() > target_size {
            features.truncate(target_size);
        }

        Ok(features)
    }

    /// Extract statistical features
    fn extract_statistical_features(&self, data: &[f32]) -> TorshResult<Vec<f32>> {
        if data.is_empty() {
            return Ok(vec![0.0; 6]);
        }

        // Mean
        let mean = data.iter().sum::<f32>() / data.len() as f32;

        // Variance and standard deviation
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std_dev = variance.sqrt();

        // Min and max
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Skewness (third moment)
        let skewness = if std_dev > 1e-10 {
            let sum_cubed = data
                .iter()
                .map(|x| ((x - mean) / std_dev).powi(3))
                .sum::<f32>();
            sum_cubed / data.len() as f32
        } else {
            0.0
        };

        Ok(vec![mean, std_dev, min_val, max_val, variance, skewness])
    }

    /// Extract spectral features using simplified frequency analysis
    fn extract_spectral_features(&self, data: &[f32]) -> TorshResult<Vec<f32>> {
        if data.len() < 4 {
            return Ok(vec![0.0; 4]);
        }

        // Simplified spectral analysis
        let mut spectral_features = Vec::new();

        // Energy in different frequency bands (simplified)
        let n = data.len();
        let quarter = n / 4;

        // Low frequency energy (0-25%)
        let low_freq_energy: f32 = data[0..quarter].iter().map(|x| x.powi(2)).sum();
        spectral_features.push(low_freq_energy);

        // Mid frequency energy (25-75%)
        let mid_freq_energy: f32 = data[quarter..3 * quarter].iter().map(|x| x.powi(2)).sum();
        spectral_features.push(mid_freq_energy);

        // High frequency energy (75-100%)
        let high_freq_energy: f32 = data[3 * quarter..].iter().map(|x| x.powi(2)).sum();
        spectral_features.push(high_freq_energy);

        // Spectral centroid (center of mass of spectrum)
        let total_energy = low_freq_energy + mid_freq_energy + high_freq_energy;
        let spectral_centroid = if total_energy > 1e-10 {
            (0.125 * low_freq_energy + 0.5 * mid_freq_energy + 0.875 * high_freq_energy)
                / total_energy
        } else {
            0.5
        };
        spectral_features.push(spectral_centroid);

        Ok(spectral_features)
    }

    /// Extract spatial features based on tensor dimensions
    fn extract_spatial_features(&self, data: &[f32], dims: &[usize]) -> TorshResult<Vec<f32>> {
        let mut spatial_features = Vec::new();

        if dims.len() >= 2 && data.len() >= dims[0] * dims[1] {
            // Treat as 2D for spatial analysis
            let height = dims[0];
            let width = dims[1];

            // Horizontal gradient energy
            let mut h_gradient_energy = 0.0;
            for i in 0..height {
                for j in 0..(width - 1) {
                    let idx1 = i * width + j;
                    let idx2 = i * width + j + 1;
                    if idx1 < data.len() && idx2 < data.len() {
                        let diff = data[idx2] - data[idx1];
                        h_gradient_energy += diff * diff;
                    }
                }
            }
            spatial_features.push(h_gradient_energy);

            // Vertical gradient energy
            let mut v_gradient_energy = 0.0;
            for i in 0..(height - 1) {
                for j in 0..width {
                    let idx1 = i * width + j;
                    let idx2 = (i + 1) * width + j;
                    if idx1 < data.len() && idx2 < data.len() {
                        let diff = data[idx2] - data[idx1];
                        v_gradient_energy += diff * diff;
                    }
                }
            }
            spatial_features.push(v_gradient_energy);

            // Spatial correlation (simplified)
            let mut correlation = 0.0;
            let mut count = 0;
            for i in 0..(height - 1) {
                for j in 0..(width - 1) {
                    let idx = i * width + j;
                    let idx_right = i * width + j + 1;
                    let idx_down = (i + 1) * width + j;
                    if idx < data.len() && idx_right < data.len() && idx_down < data.len() {
                        correlation += data[idx] * data[idx_right] + data[idx] * data[idx_down];
                        count += 2;
                    }
                }
            }
            if count > 0 {
                correlation /= count as f32;
            }
            spatial_features.push(correlation);

            // Edge density (simplified)
            let edge_threshold = 0.1;
            let mut edge_count = 0;
            let total_gradients = (h_gradient_energy + v_gradient_energy).sqrt();
            if total_gradients > edge_threshold {
                edge_count = 1;
            }
            spatial_features.push(edge_count as f32);
        } else {
            // For 1D data or when spatial analysis isn't applicable
            spatial_features = vec![0.0, 0.0, 0.0, 0.0];
        }

        // Ensure we have exactly 4 spatial features
        if spatial_features.len() < 4 {
            spatial_features.resize(4, 0.0);
        } else if spatial_features.len() > 4 {
            spatial_features.truncate(4);
        }

        Ok(spatial_features)
    }

    /// Get feature dimension count
    pub fn get_feature_dimension(&self) -> usize {
        let mut dim = 0;
        if self.enable_statistical {
            dim += 6;
        }
        if self.enable_spectral {
            dim += 4;
        }
        if self.enable_spatial {
            dim += 4;
        }
        16.max(dim) // Always return at least 16 features
    }

    /// Extract simplified features for quick analysis
    pub fn extract_quick_features(&self, data: &[f32]) -> Vec<f32> {
        if data.is_empty() {
            return vec![0.0; 4];
        }

        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        vec![mean, variance.sqrt(), min_val, max_val]
    }

    /// Calculate feature importance weights
    pub fn calculate_feature_weights(&self, _examples: &[Vec<f32>]) -> Vec<f32> {
        // Simplified feature weighting - in practice would use more sophisticated methods
        let total_features = self.get_feature_dimension();
        let uniform_weight = 1.0 / total_features as f32;
        vec![uniform_weight; total_features]
    }

    /// Normalize features to standard range
    pub fn normalize_features(&self, features: &mut [f32]) {
        if features.is_empty() {
            return;
        }

        // Simple min-max normalization to [0, 1]
        let min_val = features.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = features.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        if (max_val - min_val).abs() > 1e-10 {
            for feature in features {
                *feature = (*feature - min_val) / (max_val - min_val);
            }
        }
    }
}
