//! Multiple seasonal decomposition methods

use crate::TimeSeries;
use torsh_tensor::{creation::zeros, Tensor};

/// MSTL decomposition result
#[derive(Debug, Clone)]
pub struct MSTLResult {
    /// Trend component
    pub trend: Tensor,
    /// Seasonal components (one per period)
    pub seasonal_components: Vec<Tensor>,
    /// Residual component
    pub residual: Tensor,
}

/// Multiple STL decomposition for multiple seasonalities
pub struct MSTLDecomposition {
    periods: Vec<usize>,
    iterations: usize,
}

impl MSTLDecomposition {
    /// Create a new MSTL decomposition
    pub fn new(periods: Vec<usize>) -> Self {
        Self {
            periods,
            iterations: 2,
        }
    }

    /// Set number of iterations
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    /// Decompose time series with multiple seasonalities
    pub fn fit(&self, series: &TimeSeries) -> MSTLResult {
        // TODO: Implement actual MSTL algorithm
        // For now, return placeholder implementation
        MSTLResult {
            trend: series.values.clone(),
            seasonal_components: vec![zeros(series.values.shape().dims()).unwrap()],
            residual: zeros(series.values.shape().dims()).unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TimeSeries;
    use torsh_tensor::creation::*;

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
    fn test_mstl_decomposition() {
        let series = create_test_series();
        let mstl = MSTLDecomposition::new(vec![7, 12]);
        let result = mstl.fit(&series);

        assert_eq!(result.trend.shape().dims()[0], series.len());
        assert_eq!(result.residual.shape().dims()[0], series.len());
        assert!(!result.seasonal_components.is_empty());
    }

    #[test]
    fn test_mstl_with_iterations() {
        let mstl = MSTLDecomposition::new(vec![12]).with_iterations(5);
        assert_eq!(mstl.iterations, 5);
    }
}
