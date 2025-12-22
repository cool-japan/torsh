//! Missing value imputation methods for time series
//!
//! This module provides various strategies for handling missing values in time series data,
//! including simple forward/backward fill, interpolation, and advanced model-based methods.

use crate::{state_space::kalman::KalmanFilter, TimeSeries};
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Imputation method enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImputationMethod {
    /// Last Observation Carried Forward
    LOCF,
    /// Next Observation Carried Backward
    NOCB,
    /// Linear interpolation
    Linear,
    /// Cubic spline interpolation
    Spline,
    /// Mean imputation
    Mean,
    /// Median imputation
    Median,
    /// Kalman filter-based imputation
    KalmanFilter,
    /// Seasonal decomposition-based imputation
    Seasonal,
}

/// Missing value imputer for time series
pub struct TimeSeriesImputer {
    method: ImputationMethod,
    /// For seasonal imputation: seasonal period
    seasonal_period: Option<usize>,
    /// For KF imputation: state dimension
    state_dim: Option<usize>,
}

impl TimeSeriesImputer {
    /// Create a new imputer with specified method
    pub fn new(method: ImputationMethod) -> Self {
        Self {
            method,
            seasonal_period: None,
            state_dim: None,
        }
    }

    /// Set seasonal period for seasonal imputation
    pub fn with_seasonal_period(mut self, period: usize) -> Self {
        self.seasonal_period = Some(period);
        self
    }

    /// Set state dimension for Kalman filter imputation
    pub fn with_state_dim(mut self, dim: usize) -> Self {
        self.state_dim = Some(dim);
        self
    }

    /// Impute missing values in the time series
    ///
    /// Missing values should be represented as NaN in the input tensor.
    pub fn fit_transform(&self, series: &TimeSeries) -> Result<TimeSeries> {
        match self.method {
            ImputationMethod::LOCF => self.locf(series),
            ImputationMethod::NOCB => self.nocb(series),
            ImputationMethod::Linear => self.linear_interpolation(series),
            ImputationMethod::Spline => self.spline_interpolation(series),
            ImputationMethod::Mean => self.mean_imputation(series),
            ImputationMethod::Median => self.median_imputation(series),
            ImputationMethod::KalmanFilter => self.kalman_imputation(series),
            ImputationMethod::Seasonal => self.seasonal_imputation(series),
        }
    }

    /// Last Observation Carried Forward (LOCF)
    fn locf(&self, series: &TimeSeries) -> Result<TimeSeries> {
        let n = series.len();
        let mut imputed_data = Vec::with_capacity(n);

        let mut last_valid = None;

        for i in 0..n {
            let val = series.values.get_item_flat(i)?;
            if val.is_nan() {
                // Use last valid value if available
                if let Some(last) = last_valid {
                    imputed_data.push(last);
                } else {
                    // No previous valid value, keep NaN or use 0
                    imputed_data.push(0.0);
                }
            } else {
                last_valid = Some(val);
                imputed_data.push(val);
            }
        }

        let tensor = Tensor::from_vec(imputed_data, &[n])?;
        Ok(TimeSeries::new(tensor))
    }

    /// Next Observation Carried Backward (NOCB)
    fn nocb(&self, series: &TimeSeries) -> Result<TimeSeries> {
        let n = series.len();
        let mut imputed_data = vec![0.0f32; n];

        let mut next_valid = None;

        // Backward pass
        for i in (0..n).rev() {
            let val = series.values.get_item_flat(i)?;
            if val.is_nan() {
                // Use next valid value if available
                if let Some(next) = next_valid {
                    imputed_data[i] = next;
                } else {
                    // No next valid value, use 0
                    imputed_data[i] = 0.0;
                }
            } else {
                next_valid = Some(val);
                imputed_data[i] = val;
            }
        }

        let tensor = Tensor::from_vec(imputed_data, &[n])?;
        Ok(TimeSeries::new(tensor))
    }

    /// Linear interpolation
    fn linear_interpolation(&self, series: &TimeSeries) -> Result<TimeSeries> {
        let n = series.len();
        let mut imputed_data = Vec::with_capacity(n);

        // First, collect all valid points
        let mut valid_points: Vec<(usize, f32)> = Vec::new();
        for i in 0..n {
            let val = series.values.get_item_flat(i)?;
            if !val.is_nan() {
                valid_points.push((i, val));
            }
        }

        if valid_points.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot interpolate: no valid values".to_string(),
            ));
        }

        // Interpolate
        for i in 0..n {
            let val = series.values.get_item_flat(i)?;
            if val.is_nan() {
                // Find surrounding valid points
                let before = valid_points.iter().filter(|(idx, _)| *idx < i).last();
                let after = valid_points.iter().find(|(idx, _)| *idx > i);

                let interpolated = match (before, after) {
                    (Some((i0, v0)), Some((i1, v1))) => {
                        // Linear interpolation between two points
                        let t = (i - i0) as f32 / (i1 - i0) as f32;
                        v0 + t * (v1 - v0)
                    }
                    (Some((_, v)), None) | (None, Some((_, v))) => {
                        // Use nearest valid value
                        *v
                    }
                    (None, None) => 0.0, // Should not happen if we have valid_points
                };
                imputed_data.push(interpolated);
            } else {
                imputed_data.push(val);
            }
        }

        let tensor = Tensor::from_vec(imputed_data, &[n])?;
        Ok(TimeSeries::new(tensor))
    }

    /// Cubic spline interpolation using scirs2-core
    fn spline_interpolation(&self, series: &TimeSeries) -> Result<TimeSeries> {
        use scirs2_core::ndarray::Array1;

        // Convert to array
        let data = series.values.to_vec()?;
        let array = Array1::from_vec(data);

        // Find valid points
        let mut x_valid = Vec::new();
        let mut y_valid = Vec::new();

        for (i, &val) in array.iter().enumerate() {
            if !val.is_nan() {
                x_valid.push(i as f64);
                y_valid.push(val as f64);
            }
        }

        if x_valid.len() < 4 {
            // Fall back to linear interpolation for too few points
            return self.linear_interpolation(series);
        }

        // TODO: Use scirs2-core spline interpolation when available
        // For now, use linear interpolation as fallback
        // Cubic spline implementation would require additional dependencies
        self.linear_interpolation(series)
    }

    /// Mean imputation
    fn mean_imputation(&self, series: &TimeSeries) -> Result<TimeSeries> {
        let n = series.len();

        // Calculate mean of valid values
        let mut sum = 0.0f64;
        let mut count = 0;

        for i in 0..n {
            let val = series.values.get_item_flat(i)?;
            if !val.is_nan() {
                sum += val as f64;
                count += 1;
            }
        }

        if count == 0 {
            return Err(TorshError::InvalidArgument(
                "Cannot compute mean: no valid values".to_string(),
            ));
        }

        let mean = (sum / count as f64) as f32;

        // Replace NaN values with mean
        let mut imputed_data = Vec::with_capacity(n);
        for i in 0..n {
            let val = series.values.get_item_flat(i)?;
            imputed_data.push(if val.is_nan() { mean } else { val });
        }

        let tensor = Tensor::from_vec(imputed_data, &[n])?;
        Ok(TimeSeries::new(tensor))
    }

    /// Median imputation
    fn median_imputation(&self, series: &TimeSeries) -> Result<TimeSeries> {
        let n = series.len();

        // Collect valid values
        let mut valid_values = Vec::new();
        for i in 0..n {
            let val = series.values.get_item_flat(i)?;
            if !val.is_nan() {
                valid_values.push(val);
            }
        }

        if valid_values.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot compute median: no valid values".to_string(),
            ));
        }

        // Calculate median
        valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if valid_values.len() % 2 == 0 {
            let mid = valid_values.len() / 2;
            (valid_values[mid - 1] + valid_values[mid]) / 2.0
        } else {
            valid_values[valid_values.len() / 2]
        };

        // Replace NaN values with median
        let mut imputed_data = Vec::with_capacity(n);
        for i in 0..n {
            let val = series.values.get_item_flat(i)?;
            imputed_data.push(if val.is_nan() { median } else { val });
        }

        let tensor = Tensor::from_vec(imputed_data, &[n])?;
        Ok(TimeSeries::new(tensor))
    }

    /// Kalman filter-based imputation
    fn kalman_imputation(&self, series: &TimeSeries) -> Result<TimeSeries> {
        let state_dim = self.state_dim.unwrap_or(1);
        let mut kf = KalmanFilter::new(state_dim, 1);

        let n = series.len();
        let mut imputed_data = Vec::with_capacity(n);

        for i in 0..n {
            kf.predict()?;

            let val = series.values.get_item_flat(i)?;
            if val.is_nan() {
                // Use predicted state for missing value
                let predicted = kf.state().get_item_flat(0)?;
                imputed_data.push(predicted);
            } else {
                // Update with observation
                let obs = Tensor::from_vec(vec![val], &[1])?;
                kf.update(&obs)?;
                imputed_data.push(val);
            }
        }

        let tensor = Tensor::from_vec(imputed_data, &[n])?;
        Ok(TimeSeries::new(tensor))
    }

    /// Seasonal imputation using seasonal patterns
    fn seasonal_imputation(&self, series: &TimeSeries) -> Result<TimeSeries> {
        let period = self.seasonal_period.ok_or_else(|| {
            TorshError::InvalidArgument(
                "Seasonal period must be set for seasonal imputation".to_string(),
            )
        })?;

        let n = series.len();
        let mut imputed_data = Vec::with_capacity(n);

        // Calculate seasonal averages for each position in the cycle
        let mut seasonal_means = vec![0.0f64; period];
        let mut seasonal_counts = vec![0usize; period];

        for i in 0..n {
            let val = series.values.get_item_flat(i)?;
            if !val.is_nan() {
                let season_idx = i % period;
                seasonal_means[season_idx] += val as f64;
                seasonal_counts[season_idx] += 1;
            }
        }

        // Compute averages
        for i in 0..period {
            if seasonal_counts[i] > 0 {
                seasonal_means[i] /= seasonal_counts[i] as f64;
            }
        }

        // Impute using seasonal pattern
        for i in 0..n {
            let val = series.values.get_item_flat(i)?;
            if val.is_nan() {
                let season_idx = i % period;
                let seasonal_val = seasonal_means[season_idx] as f32;
                imputed_data.push(seasonal_val);
            } else {
                imputed_data.push(val);
            }
        }

        let tensor = Tensor::from_vec(imputed_data, &[n])?;
        Ok(TimeSeries::new(tensor))
    }
}

/// Multiple Imputation by Chained Equations (MICE) for time series
///
/// This is a more advanced imputation method that performs multiple rounds
/// of imputation, creating multiple complete datasets.
pub struct MICEImputer {
    n_imputations: usize,
    max_iter: usize,
    methods: Vec<ImputationMethod>,
}

impl MICEImputer {
    /// Create a new MICE imputer
    pub fn new(n_imputations: usize) -> Self {
        Self {
            n_imputations,
            max_iter: 10,
            methods: vec![
                ImputationMethod::Linear,
                ImputationMethod::Mean,
                ImputationMethod::KalmanFilter,
            ],
        }
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Perform multiple imputation
    pub fn fit_transform(&self, series: &TimeSeries) -> Result<Vec<TimeSeries>> {
        let mut imputations = Vec::with_capacity(self.n_imputations);

        for i in 0..self.n_imputations {
            // Use different methods for each imputation
            let method = self.methods[i % self.methods.len()];
            let imputer = TimeSeriesImputer::new(method);
            let imputed = imputer.fit_transform(series)?;
            imputations.push(imputed);
        }

        Ok(imputations)
    }

    /// Pool results from multiple imputations
    pub fn pool_results(&self, imputations: &[TimeSeries]) -> Result<TimeSeries> {
        if imputations.is_empty() {
            return Err(TorshError::InvalidArgument(
                "No imputations to pool".to_string(),
            ));
        }

        let n = imputations[0].len();
        let mut pooled_data = Vec::with_capacity(n);

        // Average across all imputations
        for i in 0..n {
            let mut sum = 0.0f64;
            for imputed in imputations {
                let val = imputed.values.get_item_flat(i)?;
                sum += val as f64;
            }
            pooled_data.push((sum / imputations.len() as f64) as f32);
        }

        let tensor = Tensor::from_vec(pooled_data, &[n])?;
        Ok(TimeSeries::new(tensor))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::Tensor;

    fn create_series_with_missing() -> TimeSeries {
        let data = vec![1.0f32, 2.0, f32::NAN, 4.0, f32::NAN, 6.0, 7.0, 8.0];
        let tensor = Tensor::from_vec(data, &[8]).unwrap();
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_locf_imputation() {
        let series = create_series_with_missing();
        let imputer = TimeSeriesImputer::new(ImputationMethod::LOCF);
        let imputed = imputer.fit_transform(&series).unwrap();

        assert_eq!(imputed.len(), series.len());
        // Check that no NaN values remain
        for i in 0..imputed.len() {
            let val = imputed.values.get_item_flat(i).unwrap();
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_nocb_imputation() {
        let series = create_series_with_missing();
        let imputer = TimeSeriesImputer::new(ImputationMethod::NOCB);
        let imputed = imputer.fit_transform(&series).unwrap();

        assert_eq!(imputed.len(), series.len());
        for i in 0..imputed.len() {
            let val = imputed.values.get_item_flat(i).unwrap();
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_linear_interpolation() {
        let series = create_series_with_missing();
        let imputer = TimeSeriesImputer::new(ImputationMethod::Linear);
        let imputed = imputer.fit_transform(&series).unwrap();

        assert_eq!(imputed.len(), series.len());
        for i in 0..imputed.len() {
            let val = imputed.values.get_item_flat(i).unwrap();
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_mean_imputation() {
        let series = create_series_with_missing();
        let imputer = TimeSeriesImputer::new(ImputationMethod::Mean);
        let imputed = imputer.fit_transform(&series).unwrap();

        assert_eq!(imputed.len(), series.len());
        for i in 0..imputed.len() {
            let val = imputed.values.get_item_flat(i).unwrap();
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_median_imputation() {
        let series = create_series_with_missing();
        let imputer = TimeSeriesImputer::new(ImputationMethod::Median);
        let imputed = imputer.fit_transform(&series).unwrap();

        assert_eq!(imputed.len(), series.len());
        for i in 0..imputed.len() {
            let val = imputed.values.get_item_flat(i).unwrap();
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_kalman_imputation() {
        let series = create_series_with_missing();
        let imputer = TimeSeriesImputer::new(ImputationMethod::KalmanFilter).with_state_dim(1);
        let imputed = imputer.fit_transform(&series).unwrap();

        assert_eq!(imputed.len(), series.len());
        for i in 0..imputed.len() {
            let val = imputed.values.get_item_flat(i).unwrap();
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_seasonal_imputation() {
        let series = create_series_with_missing();
        let imputer = TimeSeriesImputer::new(ImputationMethod::Seasonal).with_seasonal_period(4);
        let imputed = imputer.fit_transform(&series).unwrap();

        assert_eq!(imputed.len(), series.len());
        for i in 0..imputed.len() {
            let val = imputed.values.get_item_flat(i).unwrap();
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_mice_imputation() {
        let series = create_series_with_missing();
        let mice = MICEImputer::new(3).with_max_iter(5);
        let imputations = mice.fit_transform(&series).unwrap();

        assert_eq!(imputations.len(), 3);
        for imputed in &imputations {
            assert_eq!(imputed.len(), series.len());
        }

        let pooled = mice.pool_results(&imputations).unwrap();
        assert_eq!(pooled.len(), series.len());
    }
}
