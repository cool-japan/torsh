//! STL (Seasonal-Trend decomposition using LOESS) implementation

use crate::TimeSeries;
use scirs2_series::decomposition::{stl_decomposition, STLOptions};
use torsh_tensor::Tensor;

/// STL decomposition result
#[derive(Debug, Clone)]
pub struct STLResult {
    /// Trend component
    pub trend: Tensor,
    /// Seasonal component
    pub seasonal: Tensor,
    /// Residual component
    pub residual: Tensor,
}

/// Seasonal-Trend decomposition using LOESS (STL)
pub struct STLDecomposition {
    period: usize,
    seasonal_deg: usize,
    _trend_deg: usize,
    robust: bool,
}

impl STLDecomposition {
    /// Create a new STL decomposition
    pub fn new(period: usize) -> Self {
        Self {
            period,
            seasonal_deg: 1,
            _trend_deg: 1,
            robust: false,
        }
    }

    /// Set whether to use robust fitting
    pub fn robust(mut self, robust: bool) -> Self {
        self.robust = robust;
        self
    }

    /// Decompose time series using scirs2-series STL implementation
    pub fn fit(&self, series: &TimeSeries) -> Result<STLResult, torsh_core::error::TorshError> {
        use scirs2_core::ndarray::Array1;

        // Convert TimeSeries to Array1 for scirs2-series
        let data = series.values.to_vec().map_err(|e| {
            torsh_core::error::TorshError::InvalidArgument(format!(
                "Failed to convert tensor to vec: {}",
                e
            ))
        })?;
        let ts_array = Array1::from_vec(data);

        // Configure STL options
        let options = STLOptions {
            trend_window: ((series.len() / 10).max(3) | 1).max(3), // Ensure odd and >= 3
            seasonal_window: ((self.period / 2) | 1).max(3),       // Ensure odd and >= 3
            n_inner: if self.seasonal_deg > 0 { 2 } else { 1 },
            n_outer: if self.robust { 15 } else { 1 },
            robust: self.robust,
        };

        // Perform STL decomposition using scirs2-series
        let result = stl_decomposition(&ts_array, self.period, &options).map_err(|e| {
            torsh_core::error::TorshError::InvalidArgument(format!(
                "STL decomposition failed: {:?}",
                e
            ))
        })?;

        // Convert results back to tensors
        let trend_data: Vec<f32> = result.trend.to_vec();
        let seasonal_data: Vec<f32> = result.seasonal.to_vec();
        let residual_data: Vec<f32> = result.residual.to_vec();

        let n = trend_data.len();
        let trend_tensor = Tensor::from_vec(trend_data, &[n]).map_err(|e| {
            torsh_core::error::TorshError::InvalidArgument(format!(
                "Failed to create trend tensor: {}",
                e
            ))
        })?;
        let seasonal_tensor = Tensor::from_vec(seasonal_data, &[n]).map_err(|e| {
            torsh_core::error::TorshError::InvalidArgument(format!(
                "Failed to create seasonal tensor: {}",
                e
            ))
        })?;
        let residual_tensor = Tensor::from_vec(residual_data, &[n]).map_err(|e| {
            torsh_core::error::TorshError::InvalidArgument(format!(
                "Failed to create residual tensor: {}",
                e
            ))
        })?;

        Ok(STLResult {
            trend: trend_tensor,
            seasonal: seasonal_tensor,
            residual: residual_tensor,
        })
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
    fn test_stl_decomposition_creation() {
        let stl = STLDecomposition::new(12);
        assert_eq!(stl.period, 12);
        assert_eq!(stl.seasonal_deg, 1);
        assert!(!stl.robust);
    }

    #[test]
    fn test_stl_decomposition_with_options() {
        let stl = STLDecomposition::new(7).robust(true);
        assert_eq!(stl.period, 7);
        assert!(stl.robust);
    }

    #[test]
    fn test_stl_decomposition_fit() {
        let series = create_test_series();
        let stl = STLDecomposition::new(12);
        let result = stl.fit(&series).unwrap();

        // Check that components have the same length as input
        assert_eq!(result.trend.shape().dims()[0], series.len());
        assert_eq!(result.seasonal.shape().dims()[0], series.len());
        assert_eq!(result.residual.shape().dims()[0], series.len());
    }

    #[test]
    fn test_stl_decomposition_short_series() {
        // Create a longer series to satisfy the STL requirements (>= 2 * period)
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let tensor = Tensor::from_vec(data, &[10]).unwrap();
        let series = TimeSeries::new(tensor);

        // Use a smaller period that works with the series length
        let stl = STLDecomposition::new(3);
        let result = stl.fit(&series).unwrap();

        assert_eq!(result.trend.shape().dims()[0], series.len());
        assert_eq!(result.seasonal.shape().dims()[0], series.len());
        assert_eq!(result.residual.shape().dims()[0], series.len());
    }
}
