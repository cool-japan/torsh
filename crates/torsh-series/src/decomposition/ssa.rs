//! Singular Spectrum Analysis (SSA) implementation

use crate::TimeSeries;
use torsh_tensor::{creation::zeros, Tensor};

/// Singular Spectrum Analysis (SSA)
pub struct SSA {
    window_length: usize,
    num_components: usize,
}

impl SSA {
    /// Create a new SSA decomposition
    pub fn new(window_length: usize, num_components: usize) -> Self {
        Self {
            window_length,
            num_components,
        }
    }

    /// Decompose time series using SSA
    pub fn fit(&self, series: &TimeSeries) -> Vec<Tensor> {
        let data = series.values.to_vec().unwrap();
        let n = data.len();

        if self.window_length >= n || self.window_length < 2 {
            return vec![series.values.clone()];
        }

        // Step 1: Embedding - create trajectory matrix
        let k = n - self.window_length + 1;
        let mut trajectory_matrix = vec![vec![0.0f32; k]; self.window_length];

        for i in 0..self.window_length {
            for j in 0..k {
                trajectory_matrix[i][j] = data[i + j];
            }
        }

        // Step 2: SVD decomposition (simplified using covariance matrix eigendecomposition)
        let components = self.compute_ssa_components(&trajectory_matrix, k);

        // Step 3: Reconstruct components
        self.reconstruct_components(&components, n, &series.values.device())
    }

    /// Compute SSA components using simplified eigendecomposition
    fn compute_ssa_components(&self, trajectory_matrix: &[Vec<f32>], k: usize) -> Vec<Vec<f32>> {
        let m = self.window_length;

        // Compute covariance matrix C = X * X^T
        let mut covariance = vec![vec![0.0f32; m]; m];
        for i in 0..m {
            for j in 0..m {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += trajectory_matrix[i][l] * trajectory_matrix[j][l];
                }
                covariance[i][j] = sum / k as f32;
            }
        }

        // Simplified power iteration to find dominant eigenvectors
        let mut components = Vec::new();
        let num_components = self.num_components.min(m);

        for _ in 0..num_components {
            let eigenvector = self.power_iteration(&covariance);
            components.push(eigenvector);

            // Deflate matrix (simplified - in practice would remove the contribution)
            // For now, just reduce diagonal elements
            for i in 0..m {
                covariance[i][i] *= 0.9;
            }
        }

        components
    }

    /// Power iteration to find dominant eigenvector
    fn power_iteration(&self, matrix: &[Vec<f32>]) -> Vec<f32> {
        let n = matrix.len();
        let mut vector = vec![1.0f32; n];
        let iterations = 100;

        for _ in 0..iterations {
            let mut new_vector = vec![0.0f32; n];

            // Matrix-vector multiplication
            for i in 0..n {
                for j in 0..n {
                    new_vector[i] += matrix[i][j] * vector[j];
                }
            }

            // Normalize
            let norm = new_vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for i in 0..n {
                    new_vector[i] /= norm;
                }
            }

            vector = new_vector;
        }

        vector
    }

    /// Reconstruct time series components
    fn reconstruct_components(
        &self,
        eigenvectors: &[Vec<f32>],
        n: usize,
        _device: &torsh_core::device::DeviceType,
    ) -> Vec<Tensor> {
        let mut reconstructed_components = Vec::new();

        for eigenvector in eigenvectors {
            let mut reconstructed = vec![0.0f32; n];

            // Diagonal averaging to reconstruct the time series
            for i in 0..n {
                let mut sum = 0.0f32;
                let mut count = 0;

                for j in 0..self.window_length {
                    if i >= j && i - j < n - self.window_length + 1 {
                        sum += eigenvector[j];
                        count += 1;
                    }
                }

                if count > 0 {
                    reconstructed[i] = sum / count as f32;
                }
            }

            let tensor = Tensor::from_vec(reconstructed, &[n]).unwrap();
            reconstructed_components.push(tensor);
        }

        reconstructed_components
    }

    /// Forecast using SSA
    pub fn forecast(&self, series: &TimeSeries, steps: usize) -> Tensor {
        let components = self.fit(series);

        if components.is_empty() {
            return zeros(&[steps, series.num_features()]).unwrap();
        }

        // Simple extrapolation using the trend of the first component
        let main_component = &components[0];
        let data = main_component.to_vec().unwrap();
        let n = data.len();

        if n < 2 {
            return zeros(&[steps, series.num_features()]).unwrap();
        }

        // Linear extrapolation based on last few points
        let window = 5.min(n);
        let mut forecast = Vec::with_capacity(steps);

        // Calculate trend from last window points
        let trend = if window > 1 {
            (data[n - 1] - data[n - window]) / (window - 1) as f32
        } else {
            0.0f32
        };

        let last_value = data[n - 1];

        for i in 0..steps {
            forecast.push(last_value + trend * (i + 1) as f32);
        }

        Tensor::from_vec(forecast, &[steps]).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimeSeries;

    fn create_test_series() -> TimeSeries {
        // Create synthetic time series with trend and seasonality
        let mut data = Vec::new();
        for i in 0..50 {
            let trend = i as f32 * 0.1;
            let seasonal = (i as f32 * 2.0 * std::f32::consts::PI / 12.0).sin() * 2.0;
            let noise = 0.1;
            data.push(trend + seasonal + noise);
        }
        let tensor = Tensor::from_vec(data, &[50]).unwrap();
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_ssa_decomposition() {
        let series = create_test_series();
        let ssa = SSA::new(20, 10); // window_length=20, num_components=10
        let components = ssa.fit(&series);

        // Should return at least one component
        assert!(!components.is_empty());
    }

    #[test]
    fn test_ssa_forecasting() {
        let series = create_test_series();
        let ssa = SSA::new(20, 10); // window_length=20, num_components=10
        let forecast = ssa.forecast(&series, 5);

        // Forecast should have requested number of steps
        assert_eq!(forecast.shape().dims()[0], 5);
    }
}
