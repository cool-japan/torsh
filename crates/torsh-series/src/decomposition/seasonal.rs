//! Multiple seasonal decomposition methods

use crate::TimeSeries;
use torsh_core::error::{Result, TorshError};
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
    ///
    /// # Algorithm
    ///
    /// MSTL (Multiple Seasonal-Trend decomposition using Loess) iteratively decomposes
    /// a time series with multiple seasonal periods:
    ///
    /// 1. Initialize: remaining_series = original series
    /// 2. For each seasonal period in ascending order:
    ///    a. Apply STL decomposition on remaining_series with current period
    ///    b. Extract seasonal component
    ///    c. Update remaining_series by removing the seasonal component
    /// 3. Final trend and residual come from the last STL decomposition
    ///
    /// # Returns
    ///
    /// - trend: Overall trend component
    /// - seasonal_components: One seasonal component per period (in same order as input)
    /// - residual: Remaining unexplained variation
    ///
    /// # Errors
    ///
    /// Returns an error if `periods` is empty: MSTL is meaningless without at
    /// least one seasonal period, and silently returning the input as "trend"
    /// with a zero residual would be a fabricated decomposition.
    pub fn fit(&self, series: &TimeSeries) -> Result<MSTLResult> {
        use super::stl::STLDecomposition;

        let n = series.len();

        // MSTL requires at least one seasonal period. Refuse to fabricate a
        // decomposition rather than return the input unchanged as "trend".
        if self.periods.is_empty() {
            return Err(TorshError::InvalidArgument(
                "MSTL requires at least one seasonal period".to_string(),
            ));
        }

        // Sort periods in ascending order for processing
        let mut sorted_periods = self.periods.clone();
        sorted_periods.sort();

        // Initialize with original series
        let mut remaining_data = series.values.to_vec()?;
        let mut seasonal_components = Vec::new();
        let mut final_trend = series.values.clone();
        let mut final_residual = zeros(&[n])?;

        // Iterate for the specified number of iterations to refine decomposition
        for _iter in 0..self.iterations {
            let mut iter_seasonals = Vec::new();
            let mut current_remaining = remaining_data.clone();

            // Process each seasonal period
            for &period in &sorted_periods {
                if period >= n {
                    // Period too large, skip and use zeros
                    iter_seasonals.push(vec![0.0; n]);
                    continue;
                }

                // Create TimeSeries from current remaining data
                let remaining_tensor = match Tensor::from_vec(current_remaining.clone(), &[n]) {
                    Ok(t) => t,
                    Err(_) => {
                        iter_seasonals.push(vec![0.0; n]);
                        continue;
                    }
                };
                let remaining_series = TimeSeries::new(remaining_tensor);

                // Apply STL decomposition for this period
                let stl = STLDecomposition::new(period);
                let stl_result = match stl.fit(&remaining_series) {
                    Ok(result) => result,
                    Err(_) => {
                        // If STL fails, use placeholder
                        iter_seasonals.push(vec![0.0; n]);
                        continue;
                    }
                };

                // Extract seasonal component (STL succeeded, so this conversion
                // must succeed too — propagate any genuine failure).
                let seasonal_data = stl_result.seasonal.to_vec()?;
                iter_seasonals.push(seasonal_data.clone());

                // Remove seasonal component from remaining data
                for i in 0..n {
                    current_remaining[i] -= seasonal_data[i];
                }

                // Store final trend and residual from last decomposition
                final_trend = stl_result.trend;
                final_residual = stl_result.residual;
            }

            // Average seasonal components across iterations
            if _iter == 0 {
                seasonal_components = iter_seasonals;
            } else {
                // Average with previous iterations for smoothing
                for (i, new_seasonal) in iter_seasonals.iter().enumerate() {
                    if i < seasonal_components.len() {
                        for j in 0..n.min(new_seasonal.len()) {
                            seasonal_components[i][j] =
                                (seasonal_components[i][j] + new_seasonal[j]) / 2.0;
                        }
                    }
                }
            }

            // Prepare for next iteration: start with original series minus all seasonals
            remaining_data = series.values.to_vec()?;
            for seasonal in &seasonal_components {
                for i in 0..n.min(seasonal.len()) {
                    remaining_data[i] -= seasonal[i];
                }
            }
        }

        // Convert seasonal components to tensors (in original order from self.periods)
        let mut seasonal_tensors: Vec<Tensor> = Vec::with_capacity(self.periods.len());
        for (i, _period) in self.periods.iter().enumerate() {
            // Find the index in sorted_periods
            let sorted_idx = sorted_periods
                .iter()
                .position(|&p| p == self.periods[i])
                .unwrap_or(i);

            let tensor = if sorted_idx < seasonal_components.len() {
                Tensor::from_vec(seasonal_components[sorted_idx].clone(), &[n])?
            } else {
                zeros(&[n])?
            };
            seasonal_tensors.push(tensor);
        }

        Ok(MSTLResult {
            trend: final_trend,
            seasonal_components: seasonal_tensors,
            residual: final_residual,
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
        let tensor = Tensor::from_vec(data, &[50]).expect("Tensor should succeed");
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_mstl_decomposition() {
        let series = create_test_series();
        let mstl = MSTLDecomposition::new(vec![7, 12]);
        let result = mstl.fit(&series).expect("mstl fit should succeed");

        assert_eq!(result.trend.shape().dims()[0], series.len());
        assert_eq!(result.residual.shape().dims()[0], series.len());
        assert!(!result.seasonal_components.is_empty());
    }

    #[test]
    fn test_mstl_with_iterations() {
        let mstl = MSTLDecomposition::new(vec![12]).with_iterations(5);
        assert_eq!(mstl.iterations, 5);
    }

    #[test]
    fn test_mstl_empty_periods_is_error() {
        // No periods => honest error, not a fabricated "trend = input" result.
        let series = create_test_series();
        let mstl = MSTLDecomposition::new(vec![]);
        assert!(
            mstl.fit(&series).is_err(),
            "MSTL with no periods must return an error"
        );
    }
}
