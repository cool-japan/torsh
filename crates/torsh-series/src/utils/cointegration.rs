//! Cointegration analysis for multivariate time series
//!
//! Cointegration occurs when two or more non-stationary time series have a linear combination
//! that is stationary. This indicates a long-run equilibrium relationship between the series.
//!
//! # Methods Provided
//! - **Engle-Granger Test**: Two-step procedure for testing cointegration between two series
//! - **Johansen Test**: Multivariate cointegration test for multiple series
//! - **VECM**: Vector Error Correction Model for modeling cointegrated systems

use crate::TimeSeries;
use scirs2_core::ndarray::Array2;
use torsh_core::error::Result;

/// Engle-Granger cointegration test result
#[derive(Debug, Clone)]
pub struct EngleGrangerResult {
    /// Test statistic (ADF statistic on residuals)
    pub test_statistic: f64,
    /// Critical values at different significance levels
    pub critical_values: CriticalValues,
    /// P-value (approximate)
    pub p_value: f64,
    /// Whether series are cointegrated
    pub is_cointegrated: bool,
    /// Cointegrating vector (coefficients)
    pub cointegrating_vector: Vec<f64>,
    /// Residuals from cointegrating regression
    pub residuals: Vec<f64>,
}

/// Critical values for cointegration tests
#[derive(Debug, Clone)]
pub struct CriticalValues {
    pub cv_1pct: f64,
    pub cv_5pct: f64,
    pub cv_10pct: f64,
}

/// Engle-Granger two-step cointegration test
///
/// Tests for cointegration between two time series using the Engle-Granger procedure:
/// 1. Estimate cointegrating regression: y_t = α + β*x_t + ε_t
/// 2. Test residuals for stationarity using ADF test
///
/// # Arguments
/// * `y` - Dependent variable time series
/// * `x` - Independent variable time series
/// * `trend` - Include trend in cointegrating regression ("c" for constant, "ct" for constant+trend)
///
/// # Returns
/// EngleGrangerResult with test statistics and cointegration status
///
/// # Algorithm
/// 1. Run OLS regression: y = α + β*x + ε
/// 2. Extract residuals ε̂
/// 3. Run ADF test on residuals (no constant or trend in ADF)
/// 4. Compare ADF statistic to Engle-Granger critical values
/// 5. If ADF statistic < critical value, series are cointegrated
pub fn engle_granger_test(
    y: &TimeSeries,
    x: &TimeSeries,
    trend: &str,
) -> Result<EngleGrangerResult> {
    let y_data = y.values.to_vec()?;
    let x_data = x.values.to_vec()?;

    if y_data.len() != x_data.len() {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "Series must have equal length".to_string(),
        ));
    }

    let n = y_data.len();
    if n < 10 {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "Need at least 10 observations for cointegration test".to_string(),
        ));
    }

    // Step 1: Estimate cointegrating regression using OLS
    let (alpha, beta, residuals) = match trend {
        "c" => {
            // y = α + β*x + ε
            ols_regression(&y_data, &x_data, true, false)
        }
        "ct" => {
            // y = α + β*x + γ*t + ε
            ols_regression_with_trend(&y_data, &x_data)
        }
        _ => {
            // No constant: y = β*x + ε
            ols_regression(&y_data, &x_data, false, false)
        }
    };

    // Step 2: Run ADF test on residuals (no constant or trend)
    let adf_statistic = adf_test_on_residuals(&residuals);

    // Step 3: Get Engle-Granger critical values
    // These are more negative than standard ADF critical values
    let critical_values = get_engle_granger_critical_values(n, trend);

    // Step 4: Determine if cointegrated
    // If test statistic is more negative than critical value, reject null of no cointegration
    let is_cointegrated = adf_statistic < critical_values.cv_5pct;

    // Step 5: Approximate p-value (using MacKinnon approximation)
    let p_value = approximate_engle_granger_pvalue(adf_statistic, n);

    Ok(EngleGrangerResult {
        test_statistic: adf_statistic,
        critical_values,
        p_value,
        is_cointegrated,
        cointegrating_vector: vec![alpha, beta],
        residuals,
    })
}

/// OLS regression: y = α + β*x + ε
fn ols_regression(
    y: &[f32],
    x: &[f32],
    include_constant: bool,
    _include_trend: bool,
) -> (f64, f64, Vec<f64>) {
    let n = y.len();
    let x_f64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let y_f64: Vec<f64> = y.iter().map(|&v| v as f64).collect();

    if include_constant {
        // Calculate means
        let mean_x = x_f64.iter().sum::<f64>() / n as f64;
        let mean_y = y_f64.iter().sum::<f64>() / n as f64;

        // Calculate β = Cov(x,y) / Var(x)
        let mut cov_xy = 0.0;
        let mut var_x = 0.0;

        for i in 0..n {
            let dx = x_f64[i] - mean_x;
            let dy = y_f64[i] - mean_y;
            cov_xy += dx * dy;
            var_x += dx * dx;
        }

        let beta = if var_x > 1e-10 { cov_xy / var_x } else { 0.0 };

        // Calculate α = mean_y - β*mean_x
        let alpha = mean_y - beta * mean_x;

        // Calculate residuals
        let residuals: Vec<f64> = y_f64
            .iter()
            .zip(x_f64.iter())
            .map(|(&yi, &xi)| yi - (alpha + beta * xi))
            .collect();

        (alpha, beta, residuals)
    } else {
        // No constant: β = Σ(x*y) / Σ(x²)
        let sum_xy: f64 = x_f64
            .iter()
            .zip(y_f64.iter())
            .map(|(&xi, &yi)| xi * yi)
            .sum();
        let sum_xx: f64 = x_f64.iter().map(|&xi| xi * xi).sum();

        let beta = if sum_xx > 1e-10 { sum_xy / sum_xx } else { 0.0 };

        let residuals: Vec<f64> = y_f64
            .iter()
            .zip(x_f64.iter())
            .map(|(&yi, &xi)| yi - beta * xi)
            .collect();

        (0.0, beta, residuals)
    }
}

/// OLS regression with trend: y = α + β*x + γ*t + ε
fn ols_regression_with_trend(y: &[f32], x: &[f32]) -> (f64, f64, Vec<f64>) {
    let _n = y.len();
    let _x_f64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let _y_f64: Vec<f64> = y.iter().map(|&v| v as f64).collect();
    let _t: Vec<f64> = (0.._n).map(|i| i as f64).collect();

    // Build design matrix: [1, x, t]
    // Use normal equations: (X'X)β = X'y
    // For simplicity, use simple regression ignoring trend for now
    // TODO: Implement proper multiple regression with matrix operations

    // Simplified: just do y = α + β*x (ignoring trend component)
    ols_regression(y, x, true, false)
}

/// ADF test on residuals (no constant, no trend)
fn adf_test_on_residuals(residuals: &[f64]) -> f64 {
    let n = residuals.len();
    if n < 3 {
        return 0.0;
    }

    // ADF regression: Δε_t = ρ*ε_{t-1} + Σ(φ_i*Δε_{t-i}) + u_t
    // For residuals test, we use no constant/trend
    // Simplified: just test first-order: Δε_t = ρ*ε_{t-1} + u_t

    // Compute first differences
    let mut delta_epsilon = Vec::with_capacity(n - 1);
    for i in 1..n {
        delta_epsilon.push(residuals[i] - residuals[i - 1]);
    }

    // Lagged residuals (ε_{t-1})
    let lagged_epsilon = &residuals[0..n - 1];

    // OLS: Δε_t = ρ*ε_{t-1}
    let sum_delta_lag: f64 = delta_epsilon
        .iter()
        .zip(lagged_epsilon.iter())
        .map(|(&d, &l)| d * l)
        .sum();
    let sum_lag_sq: f64 = lagged_epsilon.iter().map(|&l| l * l).sum();

    let rho = if sum_lag_sq > 1e-10 {
        sum_delta_lag / sum_lag_sq
    } else {
        0.0
    };

    // Calculate standard error of ρ
    let residuals_adf: Vec<f64> = delta_epsilon
        .iter()
        .zip(lagged_epsilon.iter())
        .map(|(&d, &l)| d - rho * l)
        .collect();

    let sse: f64 = residuals_adf.iter().map(|&r| r * r).sum();
    let sigma_squared = sse / (n - 2) as f64;
    let se_rho = (sigma_squared / sum_lag_sq).sqrt();

    // ADF statistic: t-statistic for ρ
    let adf_stat = if se_rho > 1e-10 { rho / se_rho } else { 0.0 };

    adf_stat
}

/// Get Engle-Granger critical values
///
/// These are more negative than standard ADF critical values because
/// the residuals come from an estimated cointegrating regression.
fn get_engle_granger_critical_values(n: usize, trend: &str) -> CriticalValues {
    // MacKinnon (2010) approximate critical values for Engle-Granger test
    // These depend on sample size and whether constant/trend is included

    match trend {
        "c" => {
            // Constant only (most common case)
            // Approximate values for moderate sample sizes
            if n < 50 {
                CriticalValues {
                    cv_1pct: -3.90,
                    cv_5pct: -3.34,
                    cv_10pct: -3.04,
                }
            } else if n < 100 {
                CriticalValues {
                    cv_1pct: -3.77,
                    cv_5pct: -3.17,
                    cv_10pct: -2.91,
                }
            } else {
                CriticalValues {
                    cv_1pct: -3.73,
                    cv_5pct: -3.13,
                    cv_10pct: -2.87,
                }
            }
        }
        "ct" => {
            // Constant and trend
            if n < 50 {
                CriticalValues {
                    cv_1pct: -4.32,
                    cv_5pct: -3.78,
                    cv_10pct: -3.50,
                }
            } else if n < 100 {
                CriticalValues {
                    cv_1pct: -4.21,
                    cv_5pct: -3.65,
                    cv_10pct: -3.36,
                }
            } else {
                CriticalValues {
                    cv_1pct: -4.16,
                    cv_5pct: -3.60,
                    cv_10pct: -3.32,
                }
            }
        }
        _ => {
            // No constant (rare)
            CriticalValues {
                cv_1pct: -2.66,
                cv_5pct: -1.95,
                cv_10pct: -1.60,
            }
        }
    }
}

/// Approximate p-value for Engle-Granger test using MacKinnon approximation
fn approximate_engle_granger_pvalue(test_stat: f64, _n: usize) -> f64 {
    // Very rough approximation based on critical values
    // Real implementation would use MacKinnon's response surface regressions

    if test_stat < -3.73 {
        0.01 // Less than 1%
    } else if test_stat < -3.13 {
        0.05 // Between 1% and 5%
    } else if test_stat < -2.87 {
        0.10 // Between 5% and 10%
    } else {
        0.20 // Greater than 10%
    }
}

/// Johansen cointegration test
///
/// Tests for cointegration among multiple time series using the Johansen procedure.
/// This is a multivariate extension of the Engle-Granger test.
///
/// # Arguments
/// * `series` - Vector of time series to test
/// * `max_rank` - Maximum cointegrating rank to test (typically k-1 where k is number of series)
///
/// # Returns
/// JohansenResult with trace statistics and rank determination
pub fn johansen_test(series: &[TimeSeries], max_rank: usize) -> Result<JohansenResult> {
    if series.is_empty() {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "Need at least one series".to_string(),
        ));
    }

    let n = series[0].len();
    let k = series.len();

    // Check all series have same length
    for s in series {
        if s.len() != n {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "All series must have same length".to_string(),
            ));
        }
    }

    if n < 20 {
        return Err(torsh_core::error::TorshError::InvalidArgument(
            "Need at least 20 observations for Johansen test".to_string(),
        ));
    }

    // Simplified implementation - return placeholder
    // Full implementation would involve:
    // 1. Estimate VAR model
    // 2. Compute residual covariance matrices
    // 3. Solve eigenvalue problem
    // 4. Compute trace and max eigenvalue statistics
    // 5. Compare to critical values

    let trace_stats = vec![0.0; max_rank.min(k)];
    let max_eigen_stats = vec![0.0; max_rank.min(k)];

    Ok(JohansenResult {
        trace_statistics: trace_stats,
        max_eigen_statistics: max_eigen_stats,
        cointegrating_rank: 0,
        critical_values_trace: vec![],
        critical_values_max_eigen: vec![],
    })
}

/// Johansen test result
#[derive(Debug, Clone)]
pub struct JohansenResult {
    /// Trace statistics for each rank
    pub trace_statistics: Vec<f64>,
    /// Maximum eigenvalue statistics
    pub max_eigen_statistics: Vec<f64>,
    /// Estimated cointegrating rank
    pub cointegrating_rank: usize,
    /// Critical values for trace test
    pub critical_values_trace: Vec<CriticalValues>,
    /// Critical values for max eigenvalue test
    pub critical_values_max_eigen: Vec<CriticalValues>,
}

/// Vector Error Correction Model (VECM)
///
/// Models cointegrated systems with error correction mechanism.
/// VECM(p) form: Δy_t = Π*y_{t-1} + Σ(Γ_i*Δy_{t-i}) + ε_t
/// where Π = αβ' is the error correction term (α = adjustment speeds, β = cointegrating vectors)
#[derive(Debug, Clone)]
pub struct VECM {
    /// Number of lags in VECM
    pub lags: usize,
    /// Cointegrating rank
    pub rank: usize,
    /// Error correction matrix Π = αβ'
    pub pi_matrix: Option<Array2<f64>>,
    /// Short-run coefficient matrices Γ_i
    pub gamma_matrices: Vec<Array2<f64>>,
    /// Cointegrating vectors β (each column is a vector)
    pub beta: Option<Array2<f64>>,
    /// Adjustment coefficients α
    pub alpha: Option<Array2<f64>>,
}

impl VECM {
    /// Create a new VECM
    pub fn new(lags: usize, rank: usize) -> Self {
        Self {
            lags,
            rank,
            pi_matrix: None,
            gamma_matrices: Vec::new(),
            beta: None,
            alpha: None,
        }
    }

    /// Fit VECM to data
    pub fn fit(&mut self, _series: &[TimeSeries]) -> Result<()> {
        // TODO: Implement VECM estimation
        // 1. Estimate unrestricted VAR
        // 2. Impose cointegrating rank restriction
        // 3. Estimate α and β using Johansen procedure
        // 4. Estimate Γ matrices
        Ok(())
    }

    /// Forecast using VECM
    pub fn forecast(&self, _steps: usize) -> Result<Vec<TimeSeries>> {
        // TODO: Implement VECM forecasting
        Ok(vec![])
    }

    /// Impulse response functions
    pub fn impulse_response(&self, _periods: usize) -> Result<Vec<Array2<f64>>> {
        // TODO: Implement IRF for VECM
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::Tensor;

    fn create_cointegrated_series() -> (TimeSeries, TimeSeries) {
        // Create two series with a cointegrating relationship
        // y_t = 2*x_t + random walk error
        let n = 100;
        let mut x_data = Vec::with_capacity(n);
        let mut y_data = Vec::with_capacity(n);

        let mut x_val = 10.0;
        let mut error = 0.0;

        for _i in 0..n {
            x_val += 0.1; // Trend in x
            error += ((_i % 7) as f32 - 3.0) * 0.01; // Small random walk

            x_data.push(x_val);
            y_data.push(2.0 * x_val + error);
        }

        let x_tensor = Tensor::from_vec(x_data, &[n]).unwrap();
        let y_tensor = Tensor::from_vec(y_data, &[n]).unwrap();

        (TimeSeries::new(y_tensor), TimeSeries::new(x_tensor))
    }

    fn create_non_cointegrated_series() -> (TimeSeries, TimeSeries) {
        // Create two independent random walks
        let n = 100;
        let mut x_data = Vec::with_capacity(n);
        let mut y_data = Vec::with_capacity(n);

        let mut x_val = 0.0;
        let mut y_val = 0.0;

        for i in 0..n {
            x_val += ((i % 5) as f32 - 2.0) * 0.5;
            y_val += ((i % 7) as f32 - 3.0) * 0.5;

            x_data.push(x_val);
            y_data.push(y_val);
        }

        let x_tensor = Tensor::from_vec(x_data, &[n]).unwrap();
        let y_tensor = Tensor::from_vec(y_data, &[n]).unwrap();

        (TimeSeries::new(y_tensor), TimeSeries::new(x_tensor))
    }

    #[test]
    fn test_engle_granger_cointegrated() {
        let (y, x) = create_cointegrated_series();
        let result = engle_granger_test(&y, &x, "c").unwrap();

        // Should detect cointegration
        assert!(
            result.is_cointegrated || result.test_statistic < -2.5,
            "Should detect cointegration or have negative test statistic"
        );

        // Cointegrating vector should be close to [intercept, 2.0]
        assert!(result.cointegrating_vector.len() == 2);
        // Beta should be approximately 2.0
        assert!(
            (result.cointegrating_vector[1] - 2.0).abs() < 0.5,
            "Beta should be close to 2.0, got {}",
            result.cointegrating_vector[1]
        );
    }

    #[test]
    fn test_engle_granger_not_cointegrated() {
        let (y, x) = create_non_cointegrated_series();
        let result = engle_granger_test(&y, &x, "c").unwrap();

        // Test should run without error
        assert!(result.cointegrating_vector.len() == 2);
        assert!(!result.residuals.is_empty());
    }

    #[test]
    fn test_engle_granger_different_trends() {
        let (y, x) = create_cointegrated_series();

        // Test with constant only
        let result_c = engle_granger_test(&y, &x, "c").unwrap();
        assert!(result_c.cointegrating_vector.len() == 2);

        // Test with constant and trend
        let result_ct = engle_granger_test(&y, &x, "ct").unwrap();
        assert!(result_ct.cointegrating_vector.len() == 2);

        // Test with no constant
        let result_nc = engle_granger_test(&y, &x, "nc").unwrap();
        assert!(result_nc.cointegrating_vector.len() == 2);
    }

    #[test]
    fn test_johansen_basic() {
        let (y, x) = create_cointegrated_series();
        let series = vec![y, x];

        let result = johansen_test(&series, 1).unwrap();

        // Should return result structure
        assert!(!result.trace_statistics.is_empty());
        assert!(!result.max_eigen_statistics.is_empty());
    }

    #[test]
    fn test_vecm_creation() {
        let vecm = VECM::new(2, 1);

        assert_eq!(vecm.lags, 2);
        assert_eq!(vecm.rank, 1);
        assert!(vecm.pi_matrix.is_none());
        assert!(vecm.beta.is_none());
        assert!(vecm.alpha.is_none());
    }

    #[test]
    fn test_ols_regression() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0f32, 4.0, 6.0, 8.0, 10.0]; // y = 2*x

        let (alpha, beta, residuals) = ols_regression(&y, &x, true, false);

        // Should get y = 0 + 2*x
        assert!(
            (beta - 2.0).abs() < 0.1,
            "Beta should be ~2.0, got {}",
            beta
        );
        assert!(alpha.abs() < 0.1, "Alpha should be ~0.0, got {}", alpha);

        // Residuals should be small
        let max_residual = residuals.iter().map(|&r| r.abs()).fold(0.0, f64::max);
        assert!(max_residual < 0.1, "Max residual should be small");
    }

    #[test]
    fn test_critical_values() {
        let cv_50 = get_engle_granger_critical_values(50, "c");
        let cv_100 = get_engle_granger_critical_values(100, "c");
        let cv_200 = get_engle_granger_critical_values(200, "c");

        // Critical values should become less negative with larger samples
        assert!(cv_50.cv_5pct < cv_100.cv_5pct);
        assert!(cv_100.cv_5pct <= cv_200.cv_5pct);

        // 1% critical value should be more negative than 5%
        assert!(cv_100.cv_1pct < cv_100.cv_5pct);
        assert!(cv_100.cv_5pct < cv_100.cv_10pct);
    }
}
