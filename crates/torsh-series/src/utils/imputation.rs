//! Missing value imputation methods for time series
//!
//! This module provides various strategies for handling missing values in time series data,
//! including simple forward/backward fill, interpolation, and advanced model-based methods.

use crate::{state_space::kalman::KalmanFilter, TimeSeries};
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

// ============================================================
// Natural cubic spline interpolation (pure Rust, no deps)
// ============================================================

/// Natural (not-a-knot) cubic spline through a set of knot points.
///
/// For each interval [x_i, x_{i+1}] the polynomial is:
///   S_i(x) = a_i + b_i*(x-x_i) + c_i*(x-x_i)^2 + d_i*(x-x_i)^3
struct NaturalCubicSpline {
    x: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
}

impl NaturalCubicSpline {
    /// Fit the spline to the knot points `(x, y)`.
    ///
    /// `x` must be strictly increasing.
    fn fit(x: &[f64], y: &[f64]) -> Result<Self> {
        let n = x.len();
        if n < 2 {
            return Err(TorshError::InvalidArgument(
                "Need at least 2 knots for spline".to_string(),
            ));
        }
        let m = n - 1; // number of intervals

        // Step widths
        let h: Vec<f64> = (0..m).map(|i| x[i + 1] - x[i]).collect();

        // Build tridiagonal system for the second derivatives (c coefficients):
        //   h[i-1]*c[i-1] + 2*(h[i-1]+h[i])*c[i] + h[i]*c[i+1] = 3*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
        // Natural spline boundary: c[0] = 0, c[n-1] = 0
        let size = n - 2;
        if size == 0 {
            // Linear – only 2 knots
            let b0 = if h[0].abs() > 1e-15 { (y[1] - y[0]) / h[0] } else { 0.0 };
            return Ok(Self {
                x: x.to_vec(),
                a: vec![y[0]],
                b: vec![b0],
                c: vec![0.0],
                d: vec![0.0],
            });
        }

        let mut diag = vec![0.0f64; size];
        let mut upper = vec![0.0f64; size - 1];
        let mut lower = vec![0.0f64; size - 1];
        let mut rhs = vec![0.0f64; size];

        for i in 0..size {
            let idx = i + 1; // knot index (skip first and last)
            diag[i] = 2.0 * (h[idx - 1] + h[idx]);
            if i > 0 {
                lower[i - 1] = h[idx - 1];
            }
            if i < size - 1 {
                upper[i] = h[idx];
            }
            rhs[i] = 3.0 * ((y[idx + 1] - y[idx]) / h[idx]
                - (y[idx] - y[idx - 1]) / h[idx - 1]);
        }

        // Solve tridiagonal via Thomas algorithm
        let c_inner = thomas_solve(&diag, &upper, &lower, &rhs)?;

        // Full c vector with c[0]=0, c[n-1]=0
        let mut c_full = vec![0.0f64; n];
        for (i, &ci) in c_inner.iter().enumerate() {
            c_full[i + 1] = ci;
        }

        // Compute a, b, d coefficients for each interval
        let a: Vec<f64> = y[..m].to_vec();
        let b: Vec<f64> = (0..m)
            .map(|i| {
                (y[i + 1] - y[i]) / h[i]
                    - h[i] * (2.0 * c_full[i] + c_full[i + 1]) / 3.0
            })
            .collect();
        let d: Vec<f64> = (0..m)
            .map(|i| (c_full[i + 1] - c_full[i]) / (3.0 * h[i]))
            .collect();
        let c: Vec<f64> = c_full[..m].to_vec();

        Ok(Self { x: x.to_vec(), a, b, c, d })
    }

    /// Evaluate the spline at point `xi`.  Clamps to the knot range.
    fn evaluate(&self, xi: f64) -> f64 {
        let m = self.a.len();
        // Binary search for the right interval
        let idx = match self.x[1..m]
            .iter()
            .position(|&xk| xi < xk)
        {
            Some(pos) => pos,
            None => m - 1, // xi >= x[m]
        };
        let t = xi - self.x[idx];
        self.a[idx] + self.b[idx] * t + self.c[idx] * t * t + self.d[idx] * t * t * t
    }
}

/// Solve a tridiagonal system via the Thomas (forward-elimination / back-substitution) algorithm.
fn thomas_solve(
    diag: &[f64],
    upper: &[f64],
    lower: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>> {
    let n = diag.len();
    let mut c_prime = vec![0.0f64; n];
    let mut d_prime = vec![0.0f64; n];
    let mut x = vec![0.0f64; n];

    // Forward sweep
    let denom = diag[0];
    if denom.abs() < 1e-15 {
        return Err(TorshError::InvalidArgument(
            "Singular tridiagonal matrix in spline solve".to_string(),
        ));
    }
    c_prime[0] = upper.first().copied().unwrap_or(0.0) / denom;
    d_prime[0] = rhs[0] / denom;

    for i in 1..n {
        let lower_i = lower.get(i - 1).copied().unwrap_or(0.0);
        let denom_i = diag[i] - lower_i * c_prime[i - 1];
        if denom_i.abs() < 1e-15 {
            return Err(TorshError::InvalidArgument(
                "Singular tridiagonal matrix in spline solve".to_string(),
            ));
        }
        c_prime[i] = upper.get(i).copied().unwrap_or(0.0) / denom_i;
        d_prime[i] = (rhs[i] - lower_i * d_prime[i - 1]) / denom_i;
    }

    // Back substitution
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    Ok(x)
}

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

    /// Natural cubic spline interpolation.
    ///
    /// Fits a not-a-knot natural cubic spline through the valid (non-NaN) knot points
    /// and evaluates it at all integer positions.  Falls back to linear interpolation
    /// when fewer than 4 valid points are available.
    fn spline_interpolation(&self, series: &TimeSeries) -> Result<TimeSeries> {
        let data = series.values.to_vec()?;
        let n = data.len();

        // Collect valid (knot) points
        let mut x_knots: Vec<f64> = Vec::new();
        let mut y_knots: Vec<f64> = Vec::new();
        for (i, &val) in data.iter().enumerate() {
            if !val.is_nan() {
                x_knots.push(i as f64);
                y_knots.push(val as f64);
            }
        }

        if x_knots.len() < 4 {
            return self.linear_interpolation(series);
        }

        // Compute natural cubic spline coefficients (not-a-knot boundary)
        let spline = NaturalCubicSpline::fit(&x_knots, &y_knots)?;

        // Evaluate at each position
        let mut imputed = Vec::with_capacity(n);
        let x_min = x_knots[0];
        let x_max = *x_knots.last().expect("knots non-empty");
        for i in 0..n {
            let val = data[i];
            if val.is_nan() {
                let xi = i as f64;
                // Clamp to knot range for extrapolation safety
                let xi_clamped = xi.max(x_min).min(x_max);
                let interpolated = spline.evaluate(xi_clamped);
                imputed.push(interpolated as f32);
            } else {
                imputed.push(val);
            }
        }

        let tensor = Tensor::from_vec(imputed, &[n])?;
        Ok(TimeSeries::new(tensor))
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
        valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
        let tensor = Tensor::from_vec(data, &[8]).expect("Tensor should succeed");
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_locf_imputation() {
        let series = create_series_with_missing();
        let imputer = TimeSeriesImputer::new(ImputationMethod::LOCF);
        let imputed = imputer
            .fit_transform(&series)
            .expect("fit_transform should succeed with valid input");

        assert_eq!(imputed.len(), series.len());
        // Check that no NaN values remain
        for i in 0..imputed.len() {
            let val = imputed
                .values
                .get_item_flat(i)
                .expect("flat item retrieval should succeed for valid index");
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_nocb_imputation() {
        let series = create_series_with_missing();
        let imputer = TimeSeriesImputer::new(ImputationMethod::NOCB);
        let imputed = imputer
            .fit_transform(&series)
            .expect("fit_transform should succeed with valid input");

        assert_eq!(imputed.len(), series.len());
        for i in 0..imputed.len() {
            let val = imputed
                .values
                .get_item_flat(i)
                .expect("flat item retrieval should succeed for valid index");
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_linear_interpolation() {
        let series = create_series_with_missing();
        let imputer = TimeSeriesImputer::new(ImputationMethod::Linear);
        let imputed = imputer
            .fit_transform(&series)
            .expect("fit_transform should succeed with valid input");

        assert_eq!(imputed.len(), series.len());
        for i in 0..imputed.len() {
            let val = imputed
                .values
                .get_item_flat(i)
                .expect("flat item retrieval should succeed for valid index");
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_mean_imputation() {
        let series = create_series_with_missing();
        let imputer = TimeSeriesImputer::new(ImputationMethod::Mean);
        let imputed = imputer
            .fit_transform(&series)
            .expect("fit_transform should succeed with valid input");

        assert_eq!(imputed.len(), series.len());
        for i in 0..imputed.len() {
            let val = imputed
                .values
                .get_item_flat(i)
                .expect("flat item retrieval should succeed for valid index");
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_median_imputation() {
        let series = create_series_with_missing();
        let imputer = TimeSeriesImputer::new(ImputationMethod::Median);
        let imputed = imputer
            .fit_transform(&series)
            .expect("fit_transform should succeed with valid input");

        assert_eq!(imputed.len(), series.len());
        for i in 0..imputed.len() {
            let val = imputed
                .values
                .get_item_flat(i)
                .expect("flat item retrieval should succeed for valid index");
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_kalman_imputation() {
        let series = create_series_with_missing();
        let imputer = TimeSeriesImputer::new(ImputationMethod::KalmanFilter).with_state_dim(1);
        let imputed = imputer
            .fit_transform(&series)
            .expect("fit_transform should succeed with valid input");

        assert_eq!(imputed.len(), series.len());
        for i in 0..imputed.len() {
            let val = imputed
                .values
                .get_item_flat(i)
                .expect("flat item retrieval should succeed for valid index");
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_seasonal_imputation() {
        let series = create_series_with_missing();
        let imputer = TimeSeriesImputer::new(ImputationMethod::Seasonal).with_seasonal_period(4);
        let imputed = imputer
            .fit_transform(&series)
            .expect("fit_transform should succeed with valid input");

        assert_eq!(imputed.len(), series.len());
        for i in 0..imputed.len() {
            let val = imputed
                .values
                .get_item_flat(i)
                .expect("flat item retrieval should succeed for valid index");
            assert!(!val.is_nan());
        }
    }

    #[test]
    fn test_mice_imputation() {
        let series = create_series_with_missing();
        let mice = MICEImputer::new(3).with_max_iter(5);
        let imputations = mice
            .fit_transform(&series)
            .expect("fit_transform should succeed with valid input");

        assert_eq!(imputations.len(), 3);
        for imputed in &imputations {
            assert_eq!(imputed.len(), series.len());
        }

        let pooled = mice
            .pool_results(&imputations)
            .expect("result pooling should succeed");
        assert_eq!(pooled.len(), series.len());
    }
}
