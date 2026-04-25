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
///
/// Solves the 3×3 normal equations (X'X)β = X'y where the design matrix
/// columns are [1, x_i, i] using Gauss-Jordan elimination with partial
/// pivoting. Falls back to constant-only OLS if the system is singular.
fn ols_regression_with_trend(y: &[f32], x: &[f32]) -> (f64, f64, Vec<f64>) {
    let n = y.len();
    let x_f64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let y_f64: Vec<f64> = y.iter().map(|&v| v as f64).collect();

    // Build X'X (3×3) and X'y (3-vector); columns of X are [1, x_i, t_i]
    let mut xtx = [[0.0_f64; 3]; 3];
    let mut xty = [0.0_f64; 3];
    for i in 0..n {
        let row = [1.0_f64, x_f64[i], i as f64];
        for j in 0..3 {
            for k in 0..3 {
                xtx[j][k] += row[j] * row[k];
            }
            xty[j] += row[j] * y_f64[i];
        }
    }

    match solve_3x3(&xtx, &xty) {
        Some(beta) => {
            let (alpha, b, gamma) = (beta[0], beta[1], beta[2]);
            let residuals: Vec<f64> = (0..n)
                .map(|i| y_f64[i] - (alpha + b * x_f64[i] + gamma * i as f64))
                .collect();
            (alpha, b, residuals)
        }
        None => {
            // Singular system — fall back to constant-only OLS
            ols_regression(y, x, true, false)
        }
    }
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
    /// Last observed levels y_{T}, y_{T-1}, ..., y_{T-p+1} (stored by fit)
    /// Outer index 0 = most recent, 1 = one step back, etc.
    last_obs: Vec<Vec<f64>>,
    /// Last observed differences Δy_{T}, Δy_{T-1}, ..., Δy_{T-p+1} (stored by fit)
    last_deltas: Vec<Vec<f64>>,
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
            last_obs: Vec::new(),
            last_deltas: Vec::new(),
        }
    }

    /// Fit VECM to data
    ///
    /// Estimation procedure:
    /// 1. Compute first differences Δy
    /// 2. Estimate unrestricted VAR(p-1) in differences via OLS on each equation
    ///    — build lagged-difference design matrix, solve normal equations
    /// 3. Compute long-run impact matrix Π = αβ' from residual structure using SVD:
    ///    stack the OLS coefficient sum A(1) = I - Σ A_i, then factor via Jacobi SVD
    ///    to obtain the rank-r approximation β (right singular vectors) and α
    /// 4. Store fitted matrices
    pub fn fit(&mut self, series: &[TimeSeries]) -> Result<()> {
        let k = series.len();
        if k == 0 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Need at least one series for VECM fit".to_string(),
            ));
        }
        if self.rank == 0 || self.rank > k {
            return Err(torsh_core::error::TorshError::InvalidArgument(format!(
                "Cointegrating rank must be in 1..={k}, got {}",
                self.rank
            )));
        }

        // Convert each series to Vec<f64>
        let mut data: Vec<Vec<f64>> = Vec::with_capacity(k);
        for s in series {
            let vals = s.values.to_vec()?;
            data.push(vals.iter().map(|&v| v as f64).collect());
        }

        let n = data[0].len();
        for d in &data {
            if d.len() != n {
                return Err(torsh_core::error::TorshError::InvalidArgument(
                    "All series must have the same length".to_string(),
                ));
            }
        }

        let p = self.lags;
        if n <= p + 1 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Not enough observations for the specified lag order".to_string(),
            ));
        }

        // Compute first differences: delta[j][t] = data[j][t+1] - data[j][t], t=0..n-2
        let t_diff = n - 1; // number of differenced observations
        let delta: Vec<Vec<f64>> = (0..k)
            .map(|j| (0..t_diff).map(|t| data[j][t + 1] - data[j][t]).collect())
            .collect();

        // Reorder to align: we use t = p..t_diff as effective sample
        //   dependent:   delta_t  for t in p..t_diff
        //   regressors:  delta_{t-1}, ..., delta_{t-(p-1)}  (p-1 lags of differences)
        //                y_{t-1}  (levels lag for ECT)
        // For simplicity we build the VAR(p-1) in differences for the Γ matrices, then
        // compute the level-lag coefficient matrix for Π.
        let t_eff = t_diff - p; // effective sample after consuming p lags
        if t_eff < k + 1 {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Effective sample too small after consuming lags".to_string(),
            ));
        }

        // Build the OLS design matrix for each equation:
        //   columns: [1, delta_{t-1}_1, ..., delta_{t-1}_k, delta_{t-2}_1, ...,
        //             delta_{t-(p-1)}_k, y_{t-1}_1, ..., y_{t-1}_k]
        // Number of regressors: 1 + k*(p-1) + k = 1 + k*p
        let n_reg = 1 + k * p; // intercept + k*(p-1) diff lags + k level lags
        // Build regressor matrix row by row; row index s corresponds to time t = p + s
        let mut reg_matrix: Vec<Vec<f64>> = Vec::with_capacity(t_eff);
        for s in 0..t_eff {
            let t = p + s; // time index in delta array (0-based)
            let mut row = vec![1.0_f64]; // intercept
            // Lagged differences: lags 1..p-1
            for lag in 1..p {
                for j in 0..k {
                    let idx = t.checked_sub(lag).unwrap_or(0);
                    row.push(if idx < delta[j].len() {
                        delta[j][idx]
                    } else {
                        0.0
                    });
                }
            }
            // Level lag y_{t-1} — t in delta corresponds to original index t+1,
            // so y_{t-1} in original = data[j][t] (0-based)
            for j in 0..k {
                row.push(data[j][t]);
            }
            reg_matrix.push(row);
        }

        // Solve OLS for each dependent variable using normal equations
        // coef_matrix[j] = coefficient vector (length n_reg) for variable j
        let mut coef_matrix: Vec<Vec<f64>> = Vec::with_capacity(k);
        let mut residual_matrix: Vec<Vec<f64>> = (0..k).map(|_| vec![0.0_f64; t_eff]).collect();

        for j in 0..k {
            let y_dep: Vec<f64> = (0..t_eff).map(|s| delta[j][p + s]).collect();
            match ols_multivariate(&reg_matrix, &y_dep) {
                Some(coef) => {
                    // Compute residuals
                    for s in 0..t_eff {
                        let fitted: f64 = coef
                            .iter()
                            .zip(reg_matrix[s].iter())
                            .map(|(c, x)| c * x)
                            .sum();
                        residual_matrix[j][s] = y_dep[s] - fitted;
                    }
                    coef_matrix.push(coef);
                }
                None => {
                    // Singular: zero coefficients, residuals = y
                    coef_matrix.push(vec![0.0_f64; n_reg]);
                    for s in 0..t_eff {
                        residual_matrix[j][s] = delta[j][p + s];
                    }
                }
            }
        }

        // Extract Γ matrices (short-run coefficients) from coef_matrix
        // Γ_i is k×k matrix of coefficients for lag i of Δy
        // coef layout: [intercept, diff_lag_1 (k cols), diff_lag_2 (k cols), ..., level_lag (k cols)]
        let mut gamma_matrices: Vec<Array2<f64>> = Vec::new();
        for lag in 0..(p.saturating_sub(1)) {
            let col_start = 1 + lag * k;
            let mut gamma = Array2::zeros((k, k));
            for j in 0..k {
                for m in 0..k {
                    gamma[[j, m]] = coef_matrix[j][col_start + m];
                }
            }
            gamma_matrices.push(gamma);
        }

        // Extract long-run level coefficients → form Π (k×k)
        // Level-lag columns start at: 1 + (p-1)*k
        let level_start = 1 + (p.saturating_sub(1)) * k;
        let mut pi_flat: Vec<f64> = vec![0.0_f64; k * k];
        for j in 0..k {
            for m in 0..k {
                pi_flat[j * k + m] = if level_start + m < coef_matrix[j].len() {
                    coef_matrix[j][level_start + m]
                } else {
                    0.0
                };
            }
        }

        // Factor Π ≈ α β' using rank-r SVD via Jacobi decomposition on Π'Π
        // Π is k×k; result: alpha is k×r, beta is k×r
        let pi_mat: Vec<Vec<f64>> = (0..k)
            .map(|j| pi_flat[j * k..(j + 1) * k].to_vec())
            .collect();
        let (u_mat, sigma, v_mat) = svd_small(&pi_mat, self.rank);

        // α = U * Σ  (k×r),  β = V  (k×r)
        let mut alpha_arr = Array2::zeros((k, self.rank));
        let mut beta_arr = Array2::zeros((k, self.rank));
        let mut pi_arr = Array2::zeros((k, k));

        for j in 0..k {
            for r in 0..self.rank {
                alpha_arr[[j, r]] = u_mat[j][r] * sigma[r];
                beta_arr[[j, r]] = v_mat[j][r];
            }
            for m in 0..k {
                pi_arr[[j, m]] = pi_flat[j * k + m];
            }
        }

        self.alpha = Some(alpha_arr);
        self.beta = Some(beta_arr);
        self.pi_matrix = Some(pi_arr);
        self.gamma_matrices = gamma_matrices;

        // Store last p levels and last p differences for forecast() initialisation.
        // last_obs[0] = y_{T} (most recent level), last_obs[1] = y_{T-1}, etc.
        // last_deltas[0] = Δy_{T} (most recent difference), etc.
        let last_p = p.max(1);
        self.last_obs = (0..last_p)
            .map(|lag| {
                // original index of y for lag `lag` back: n-1-lag
                let idx = n.saturating_sub(1 + lag);
                (0..k).map(|j| data[j][idx]).collect()
            })
            .collect();
        self.last_deltas = (0..last_p)
            .map(|lag| {
                // delta[j][t_diff-1-lag] where t_diff = n-1
                let idx = t_diff.saturating_sub(1 + lag);
                (0..k)
                    .map(|j| if idx < delta[j].len() { delta[j][idx] } else { 0.0 })
                    .collect()
            })
            .collect();

        Ok(())
    }

    /// Forecast using VECM recursion
    ///
    /// Uses VECM form:
    ///   Δy_t = c + α β' y_{t-1} + Σ_i Γ_i Δy_{t-i} + ε
    ///
    /// Initialises from the last observed level and differences and iterates
    /// forward for `steps` periods. Returns one TimeSeries per variable.
    pub fn forecast(&self, steps: usize) -> Result<Vec<TimeSeries>> {
        use torsh_tensor::Tensor;
        let pi = self
            .pi_matrix
            .as_ref()
            .ok_or_else(|| torsh_core::error::TorshError::NotImplemented("VECM not fitted".to_string()))?;
        let k = pi.shape()[0];

        // Initialise from the last observed values stored during fit().
        // If fit() has not been called (last_obs empty), fall back to zero-level initialisation.
        let mut y_level = if self.last_obs.is_empty() {
            vec![0.0_f64; k]
        } else {
            self.last_obs[0].clone() // most recent level y_T
        };
        let p = self.lags;
        let mut delta_history: Vec<Vec<f64>> = if self.last_deltas.is_empty() {
            vec![vec![0.0_f64; k]; p.max(1)]
        } else {
            // last_deltas[0] = Δy_T (most recent), reverse so oldest is first in ring buffer
            let mut h: Vec<Vec<f64>> = self.last_deltas.iter().cloned().rev().collect();
            h.truncate(p.max(1));
            while h.len() < p.max(1) {
                h.insert(0, vec![0.0_f64; k]);
            }
            h
        };

        // Collect paths per variable
        let mut paths: Vec<Vec<f64>> = (0..k).map(|_| Vec::with_capacity(steps)).collect();

        for _step in 0..steps {
            // Compute Δy = c + Π y_{t-1} + Σ Γ_i Δy_{t-i}
            let mut delta_new = vec![0.0_f64; k];

            // Π y_{t-1}
            for j in 0..k {
                for m in 0..k {
                    delta_new[j] += pi[[j, m]] * y_level[m];
                }
            }

            // Γ_i Δy_{t-i}
            for (i, gamma) in self.gamma_matrices.iter().enumerate() {
                let hist_idx = delta_history.len().saturating_sub(i + 1);
                let lag_delta = if hist_idx < delta_history.len() {
                    &delta_history[hist_idx]
                } else {
                    continue;
                };
                for j in 0..k {
                    for m in 0..k {
                        delta_new[j] += gamma[[j, m]] * lag_delta[m];
                    }
                }
            }

            // y_t = y_{t-1} + Δy
            let y_new: Vec<f64> = y_level.iter().zip(delta_new.iter()).map(|(y, d)| y + d).collect();

            for j in 0..k {
                paths[j].push(y_new[j]);
            }

            delta_history.push(delta_new);
            if delta_history.len() > p {
                delta_history.remove(0);
            }
            y_level = y_new;
        }

        // Convert paths to TimeSeries
        let result: Result<Vec<_>> = paths
            .into_iter()
            .map(|path| {
                let n = path.len();
                let f32_path: Vec<f32> = path.iter().map(|&v| v as f32).collect();
                Tensor::from_vec(f32_path, &[n])
                    .map(TimeSeries::new)
                    .map_err(|e| torsh_core::error::TorshError::InvalidArgument(e.to_string()))
            })
            .collect();
        result
    }

    /// Impulse response functions
    ///
    /// Computes MA(∞) coefficient matrices Φ_h for h = 0..n_periods where
    /// Φ_h[i,j] is the response of variable i at horizon h to a unit shock in
    /// variable j at h=0. Uses the VECM companion-form recursion.
    ///
    /// Returns one k×k Array2 per horizon (n_periods+1 matrices total).
    pub fn impulse_response(&self, periods: usize) -> Result<Vec<Array2<f64>>> {
        let pi = self
            .pi_matrix
            .as_ref()
            .ok_or_else(|| torsh_core::error::TorshError::NotImplemented("VECM not fitted".to_string()))?;
        let k = pi.shape()[0];
        let p = self.lags;

        // Build the level-form companion VAR coefficient matrix A_comp
        // for the companion system of dimension k*p:
        //   A_comp = I + Π  (first k×k block, from VECM → VAR(1) transformation)
        //          + Γ_1 ... Γ_{p-1} in remaining columns
        // Full companion matrix is k*p × k*p
        let kp = k * p;
        let mut a_comp: Vec<Vec<f64>> = vec![vec![0.0_f64; kp]; kp];

        // First k rows: A_1 = I + Π + Γ_1, A_2 = Γ_2 - Γ_1, etc. (VECM→VAR conversion)
        // Simplified companion: use A_i directly from the VAR representation A(L) = I - Π
        // A_1 = I + Π + Γ_1, A_i = Γ_i - Γ_{i-1} for 1<i<p, A_p = -Γ_{p-1}
        // For i-th lag block (0-indexed), the VAR coefficient is:
        //   A_1 = I + Π + (Γ_1 if p>1 else 0)
        //   A_i (1<i<p) = Γ_i - Γ_{i-1}  (using 1-indexed Γ)
        //   A_p = -Γ_{p-1}
        for j in 0..k {
            for m in 0..k {
                // A_1 block: I + Π
                let pi_val = pi[[j, m]];
                let i_val = if j == m { 1.0_f64 } else { 0.0_f64 };
                let gamma1_val = if !self.gamma_matrices.is_empty() {
                    self.gamma_matrices[0][[j, m]]
                } else {
                    0.0
                };
                a_comp[j][m] = i_val + pi_val + gamma1_val;

                // A_2..A_{p-1} blocks
                for lag in 1..(p.saturating_sub(1)) {
                    let g_curr = if lag < self.gamma_matrices.len() {
                        self.gamma_matrices[lag][[j, m]]
                    } else {
                        0.0
                    };
                    let g_prev = self.gamma_matrices[lag - 1][[j, m]];
                    a_comp[j][m + (lag) * k] = g_curr - g_prev;
                }

                // A_p block: -Γ_{p-1}
                if p > 1 {
                    let g_last = if p - 2 < self.gamma_matrices.len() {
                        self.gamma_matrices[p - 2][[j, m]]
                    } else {
                        0.0
                    };
                    a_comp[j][m + (p - 1) * k] = -g_last;
                }
            }
        }

        // Identity block in companion: rows k..kp, col -(k) shift
        for j in k..kp {
            if j - k < kp {
                a_comp[j][j - k] = 1.0;
            }
        }

        // Impulse responses: iterate Φ_h = A_comp^h applied to e_j for each shock
        // We accumulate Φ_h matrices (k×k) by repeated companion multiplication.
        // State vector is kp-dimensional; extract top-k rows for responses.
        let mut irf: Vec<Array2<f64>> = Vec::with_capacity(periods + 1);

        // Φ_0 = I (contemporaneous unit shock identity for the k variables)
        let phi0 = Array2::eye(k);
        irf.push(phi0);

        // State matrix: columns are unit shocks; rows track kp-dimensional companion state
        // We use kp × k matrix; each column is a companion-space state for one shock
        let mut state: Vec<Vec<f64>> = vec![vec![0.0_f64; k]; kp];
        // Initialise: top-k rows = I_k (unit shocks), rest zero
        for j in 0..k {
            state[j][j] = 1.0;
        }

        for _h in 0..periods {
            // state_new = A_comp * state
            let mut state_new: Vec<Vec<f64>> = vec![vec![0.0_f64; k]; kp];
            for row in 0..kp {
                for col in 0..k {
                    let val: f64 = (0..kp).map(|c| a_comp[row][c] * state[c][col]).sum();
                    state_new[row][col] = val;
                }
            }
            // Extract top k rows → Φ_{h+1} (k×k)
            let mut phi_h = Array2::zeros((k, k));
            for j in 0..k {
                for m in 0..k {
                    phi_h[[j, m]] = state_new[j][m];
                }
            }
            irf.push(phi_h);
            state = state_new;
        }

        Ok(irf)
    }
}

// ─── Local matrix utilities (pure Rust, no external deps) ──────────────────

/// Solve 3×3 linear system A x = b via Gauss-Jordan with partial pivoting.
/// Returns `None` if the matrix is singular (max pivot < 1e-12).
fn solve_3x3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> Option<[f64; 3]> {
    let mut aug = [[0.0_f64; 4]; 3];
    for i in 0..3 {
        aug[i][0] = a[i][0];
        aug[i][1] = a[i][1];
        aug[i][2] = a[i][2];
        aug[i][3] = b[i];
    }
    for col in 0..3 {
        // Partial pivot
        let max_row = (col..3)
            .max_by(|&r1, &r2| aug[r1][col].abs().partial_cmp(&aug[r2][col].abs()).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(col);
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        if pivot.abs() < 1e-12 {
            return None;
        }
        // Scale pivot row
        let inv_pivot = 1.0 / pivot;
        for k in col..4 {
            aug[col][k] *= inv_pivot;
        }
        // Eliminate column in all other rows
        for row in 0..3 {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for k in col..4 {
                aug[row][k] -= factor * aug[col][k];
            }
        }
    }
    Some([aug[0][3], aug[1][3], aug[2][3]])
}

/// Solve n×n linear system A x = b via Gauss-Jordan with partial pivoting.
/// Returns `None` if the matrix is singular (max pivot < 1e-12).
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    debug_assert_eq!(a.len(), n);
    // Build augmented matrix [A | b]
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &rhs)| {
            let mut r = row.clone();
            r.push(rhs);
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivot
        let max_row = (col..n)
            .max_by(|&r1, &r2| {
                aug[r1][col]
                    .abs()
                    .partial_cmp(&aug[r2][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        if pivot.abs() < 1e-12 {
            return None;
        }
        let inv_pivot = 1.0 / pivot;
        for k in col..=n {
            aug[col][k] *= inv_pivot;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for k in col..=n {
                aug[row][k] -= factor * aug[col][k];
            }
        }
    }
    Some((0..n).map(|i| aug[i][n]).collect())
}

/// Solve normal equations (X'X) β = X'y using `solve_linear_system`.
/// `x_rows` is a list of row vectors (design matrix rows).
/// Returns the coefficient vector or `None` if singular.
fn ols_multivariate(x_rows: &[Vec<f64>], y: &[f64]) -> Option<Vec<f64>> {
    let n_obs = x_rows.len();
    debug_assert_eq!(y.len(), n_obs);
    if n_obs == 0 {
        return None;
    }
    let n_reg = x_rows[0].len();
    // Build X'X and X'y
    let mut xtx: Vec<Vec<f64>> = vec![vec![0.0_f64; n_reg]; n_reg];
    let mut xty: Vec<f64> = vec![0.0_f64; n_reg];
    for (row, &yi) in x_rows.iter().zip(y.iter()) {
        for j in 0..n_reg {
            for k in 0..n_reg {
                xtx[j][k] += row[j] * row[k];
            }
            xty[j] += row[j] * yi;
        }
    }
    solve_linear_system(&xtx, &xty)
}

/// One-sided Jacobi sweep on a symmetric matrix to compute eigenvalues/vectors.
/// Returns `(eigenvalues, eigenvectors)` after up to `max_iter` sweeps.
/// Eigenvalues are sorted descending by magnitude.
fn jacobi_eig_symmetric(mat: &[Vec<f64>], max_iter: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = mat.len();
    let mut a: Vec<Vec<f64>> = mat.to_vec();
    // Identity eigenvector matrix
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();

    for _ in 0..max_iter {
        // Find max off-diagonal
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break;
        }
        // Compute Jacobi rotation angle
        let theta = if (a[p][p] - a[q][q]).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * a[p][q]) / (a[p][p] - a[q][q])).atan()
        };
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Apply Jacobi rotation J' A J
        let a_pp = a[p][p];
        let a_qq = a[q][q];
        let a_pq = a[p][q];
        a[p][p] = cos_t * cos_t * a_pp + 2.0 * cos_t * sin_t * a_pq + sin_t * sin_t * a_qq;
        a[q][q] = sin_t * sin_t * a_pp - 2.0 * cos_t * sin_t * a_pq + cos_t * cos_t * a_qq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for r in 0..n {
            if r == p || r == q {
                continue;
            }
            let a_rp = a[r][p];
            let a_rq = a[r][q];
            a[r][p] = cos_t * a_rp + sin_t * a_rq;
            a[p][r] = a[r][p];
            a[r][q] = -sin_t * a_rp + cos_t * a_rq;
            a[q][r] = a[r][q];
        }
        // Update eigenvector matrix
        for r in 0..n {
            let v_rp = v[r][p];
            let v_rq = v[r][q];
            v[r][p] = cos_t * v_rp + sin_t * v_rq;
            v[r][q] = -sin_t * v_rp + cos_t * v_rq;
        }
    }

    // Collect and sort eigenvalues descending by magnitude
    let mut pairs: Vec<(f64, usize)> = (0..n).map(|i| (a[i][i], i)).collect();
    pairs.sort_by(|(a, _), (b, _)| b.abs().partial_cmp(&a.abs()).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<f64> = pairs.iter().map(|&(ev, _)| ev).collect();
    let eigenvectors: Vec<Vec<f64>> = pairs
        .iter()
        .map(|&(_, idx)| (0..n).map(|r| v[r][idx]).collect())
        .collect();
    (eigenvalues, eigenvectors)
}

/// Compute the thin SVD of a k×k matrix `a`, returning at most `rank`
/// singular values and vectors. Uses Jacobi eigen-decomposition on A'A.
/// Returns `(U, sigma, V)` where U is k×r, sigma is r-vec, V is k×r.
fn svd_small(a: &[Vec<f64>], rank: usize) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let k = a.len();
    if k == 0 {
        return (vec![], vec![], vec![]);
    }
    let r = rank.min(k);

    // Compute A'A (k×k symmetric)
    let mut ata: Vec<Vec<f64>> = vec![vec![0.0_f64; k]; k];
    for i in 0..k {
        for j in 0..k {
            for row in 0..k {
                ata[i][j] += a[row][i] * a[row][j];
            }
        }
    }

    let max_jacobi = 200 * k;
    let (eigenvalues, v_cols) = jacobi_eig_symmetric(&ata, max_jacobi);

    // sigma_i = sqrt(max(0, eigenvalue_i))
    let sigma: Vec<f64> = eigenvalues[..r].iter().map(|&ev| ev.max(0.0).sqrt()).collect();

    // V matrix (k×r): columns are right singular vectors
    let v_mat: Vec<Vec<f64>> = (0..k).map(|row| (0..r).map(|col| v_cols[col][row]).collect()).collect();

    // U_i = A V_i / sigma_i  (with fallback for zero singular values)
    let mut u_mat: Vec<Vec<f64>> = vec![vec![0.0_f64; r]; k];
    for col in 0..r {
        if sigma[col] < 1e-14 {
            // Zero singular value — set corresponding left vector to zero
            continue;
        }
        let inv_s = 1.0 / sigma[col];
        // Compute A * v_col
        for row in 0..k {
            let av: f64 = (0..k).map(|m| a[row][m] * v_cols[col][m]).sum();
            u_mat[row][col] = av * inv_s;
        }
    }

    (u_mat, sigma, v_mat)
}

// ─── Tests ──────────────────────────────────────────────────────────────────

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

        let x_tensor = Tensor::from_vec(x_data, &[n]).expect("x tensor");
        let y_tensor = Tensor::from_vec(y_data, &[n]).expect("y tensor");

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

        let x_tensor = Tensor::from_vec(x_data, &[n]).expect("x tensor");
        let y_tensor = Tensor::from_vec(y_data, &[n]).expect("y tensor");

        (TimeSeries::new(y_tensor), TimeSeries::new(x_tensor))
    }

    #[test]
    fn test_engle_granger_cointegrated() {
        let (y, x) = create_cointegrated_series();
        let result = engle_granger_test(&y, &x, "c").expect("engle granger test");

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
        let result = engle_granger_test(&y, &x, "c").expect("engle granger test");

        // Test should run without error
        assert!(result.cointegrating_vector.len() == 2);
        assert!(!result.residuals.is_empty());
    }

    #[test]
    fn test_engle_granger_different_trends() {
        let (y, x) = create_cointegrated_series();

        // Test with constant only
        let result_c = engle_granger_test(&y, &x, "c").expect("constant test");
        assert!(result_c.cointegrating_vector.len() == 2);

        // Test with constant and trend
        let result_ct = engle_granger_test(&y, &x, "ct").expect("constant+trend test");
        assert!(result_ct.cointegrating_vector.len() == 2);

        // Test with no constant
        let result_nc = engle_granger_test(&y, &x, "nc").expect("no-constant test");
        assert!(result_nc.cointegrating_vector.len() == 2);
    }

    #[test]
    fn test_johansen_basic() {
        let (y, x) = create_cointegrated_series();
        let series = vec![y, x];

        let result = johansen_test(&series, 1).expect("johansen test");

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

    #[test]
    fn test_ols_regression_with_trend() {
        // Synthetic: y_i = 1 + 2*x_i + 0.5*i  (exact, no noise)
        // x must NOT be a linear function of i to avoid perfect collinearity.
        // Use x_i = sin(i) + 3 so that x and t=i are not collinear.
        let n = 30usize;
        let x: Vec<f32> = (0..n)
            .map(|i| ((i as f32).sin() + 3.0))
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|i| 1.0 + 2.0 * x[i] + 0.5 * (i as f32))
            .collect();

        let (alpha, beta, residuals) = ols_regression_with_trend(&y, &x);

        assert!(
            (alpha - 1.0).abs() < 1.0,
            "alpha should be ~1.0, got {alpha}"
        );
        assert!(
            (beta - 2.0).abs() < 0.5,
            "beta should be ~2.0, got {beta}"
        );
        let max_res = residuals.iter().map(|&r| r.abs()).fold(0.0_f64, f64::max);
        assert!(max_res < 1.0, "max residual should be small, got {max_res}");
    }

    #[test]
    fn test_solve_3x3_identity() {
        let a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b = [3.0, 5.0, 7.0];
        let sol = solve_3x3(&a, &b).expect("identity system should be solvable");
        assert!((sol[0] - 3.0).abs() < 1e-10);
        assert!((sol[1] - 5.0).abs() < 1e-10);
        assert!((sol[2] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_3x3_singular_returns_none() {
        // Singular: row 2 = row 1
        let a = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0.0, 0.0, 1.0]];
        let b = [1.0, 2.0, 3.0];
        assert!(solve_3x3(&a, &b).is_none());
    }

    #[test]
    fn test_jacobi_eig_2x2() {
        // Known: [[2, 1], [1, 2]] has eigenvalues 3, 1
        let mat = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let (eigenvalues, _) = jacobi_eig_symmetric(&mat, 100);
        assert_eq!(eigenvalues.len(), 2);
        // Sorted descending by magnitude: 3 first, then 1
        assert!(
            (eigenvalues[0] - 3.0).abs() < 1e-10,
            "expected 3.0, got {}",
            eigenvalues[0]
        );
        assert!(
            (eigenvalues[1] - 1.0).abs() < 1e-10,
            "expected 1.0, got {}",
            eigenvalues[1]
        );
    }

    #[test]
    fn test_svd_small_rank1() {
        // Rank-1 matrix: A = u * v' where u=[1,0], v=[1,0] → sigma=1
        let a = vec![vec![1.0, 0.0], vec![0.0, 0.0]];
        let (u, sigma, v) = svd_small(&a, 1);
        assert_eq!(sigma.len(), 1);
        assert!(
            (sigma[0] - 1.0).abs() < 1e-6,
            "singular value should be ~1.0, got {}",
            sigma[0]
        );
        assert_eq!(u.len(), 2);
        assert_eq!(v.len(), 2);
    }

    fn create_vecm_test_series() -> Vec<TimeSeries> {
        // Two cointegrated random walks: y2_t = 2*y1_t + stationary noise
        let n = 60usize;
        let mut y1_data = Vec::with_capacity(n);
        let mut y2_data = Vec::with_capacity(n);
        let mut level = 10.0_f32;
        for i in 0..n {
            level += ((i % 3) as f32 - 1.0) * 0.5;
            y1_data.push(level);
            y2_data.push(2.0 * level + ((i % 5) as f32 - 2.0) * 0.1);
        }
        let t1 = Tensor::from_vec(y1_data, &[n]).expect("tensor y1");
        let t2 = Tensor::from_vec(y2_data, &[n]).expect("tensor y2");
        vec![TimeSeries::new(t1), TimeSeries::new(t2)]
    }

    #[test]
    fn test_vecm_fit_basic() {
        let series = create_vecm_test_series();
        let mut vecm = VECM::new(2, 1);
        vecm.fit(&series).expect("fit should succeed");
        assert!(vecm.pi_matrix.is_some(), "pi_matrix should be set after fit");
        assert!(vecm.alpha.is_some(), "alpha should be set after fit");
        assert!(vecm.beta.is_some(), "beta should be set after fit");
        let pi = vecm.pi_matrix.as_ref().expect("pi");
        assert_eq!(pi.shape(), [2, 2]);
    }

    #[test]
    fn test_vecm_forecast_shape() {
        let series = create_vecm_test_series();
        let mut vecm = VECM::new(2, 1);
        vecm.fit(&series).expect("fit should succeed");
        let steps = 10;
        let forecasts = vecm.forecast(steps).expect("forecast should succeed");
        assert_eq!(forecasts.len(), 2, "should return one series per variable");
        for ts in &forecasts {
            assert_eq!(ts.len(), steps, "each series should have {steps} points");
        }
    }

    #[test]
    fn test_vecm_impulse_response_shape() {
        let series = create_vecm_test_series();
        let mut vecm = VECM::new(2, 1);
        vecm.fit(&series).expect("fit should succeed");
        let periods = 5;
        let irf = vecm.impulse_response(periods).expect("irf should succeed");
        assert_eq!(
            irf.len(),
            periods + 1,
            "should have periods+1 matrices (including h=0)"
        );
        for (h, phi) in irf.iter().enumerate() {
            assert_eq!(phi.shape(), [2, 2], "matrix at h={h} should be 2x2");
        }
    }

    #[test]
    fn test_vecm_fit_invalid_rank() {
        let series = create_vecm_test_series();
        // rank > k should fail
        let mut vecm = VECM::new(2, 5);
        assert!(vecm.fit(&series).is_err());
    }
}
