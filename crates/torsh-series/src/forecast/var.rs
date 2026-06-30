//! Vector Autoregression (VAR) models for multivariate time series
//!
//! VAR models are used to capture the linear interdependencies among multiple time series.
//! Each variable is modeled as a linear function of past values of itself and past values
//! of the other variables.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::TimeSeries;
use scirs2_core::ndarray::{Array1, Array2};
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

// ============================================================
// Pure-Rust math helpers for p-value computation
// ============================================================

fn regularised_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let lbeta = ln_gamma_var(a) + ln_gamma_var(b) - ln_gamma_var(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - lbeta).exp() / a;
    front * beta_cf_var(a, b, x)
}

fn beta_cf_var(a: f64, b: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 3e-10;
    let fpmin = f64::MIN_POSITIVE / eps;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < fpmin {
        d = fpmin;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=max_iter {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        let mut aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        h *= d * c;
        aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = 1.0 + aa / c;
        if c.abs() < fpmin {
            c = fpmin;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < eps {
            break;
        }
    }
    h
}

/// ln(Gamma(x)) via Lanczos approximation (g=7, n=9).
///
/// Uses the recurrence `ln Γ(x) = ln Γ(x+1) − ln(x)` for 0 < x < 1
/// so the Lanczos kernel is always evaluated at x ≥ 1 where it is
/// numerically stable.  No recursion — the loop terminates in at most
/// one iteration for the arguments that arise in beta-function p-values.
fn ln_gamma_var(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_3,
        676.520_368_121_885_1,
        -1_259.139_216_722_403_0,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_904_8,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];
    // Shift x up until xr ≥ 1, accumulating ln(x), ln(x+1), ...
    let mut shift = 0.0;
    let mut xr = x;
    while xr < 1.0 {
        shift += xr.ln();
        xr += 1.0;
    }
    let z = xr - 1.0;
    let mut ser = C[0];
    for (k, &ck) in C[1..].iter().enumerate() {
        ser += ck / (z + (k + 1) as f64);
    }
    let t = z + G + 0.5;
    let lanczos = (2.0 * std::f64::consts::PI).sqrt().ln() + (z + 0.5) * t.ln() - t + ser.ln();
    lanczos - shift
}

// ============================================================
// Cholesky helpers for OLS estimation
// ============================================================

/// Cholesky-Banachiewicz decomposition (lower triangular, row-major).
/// Returns `None` if the matrix is not positive-definite.
fn var_cholesky_lower(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i * n + j] = sum.sqrt();
            } else {
                let diag = l[j * n + j];
                if diag.abs() < 1e-15 {
                    return None;
                }
                l[i * n + j] = sum / diag;
            }
        }
    }
    Some(l)
}

/// Solve the symmetric PD system A x = b using the precomputed Cholesky lower L.
fn var_cholesky_solve_with_l(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    // Forward substitution L y = b
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * y[j];
        }
        y[i] = s / l[i * n + i];
    }
    // Back substitution L^T x = y
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j];
        }
        x[i] = s / l[i * n + i];
    }
    x
}

/// Solve A x = b for symmetric PD A.
/// Falls back to ridge-regularised diagonal solve if A is not PD.
fn var_cholesky_solve(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    // Try exact Cholesky first
    if let Some(l) = var_cholesky_lower(a, n) {
        return var_cholesky_solve_with_l(&l, b, n);
    }
    // Ridge fallback: use diagonal of A + λI
    let lambda = 1e-6;
    (0..n)
        .map(|i| b[i] / (a[i * n + i] + lambda).max(1e-15))
        .collect()
}

/// Vector Autoregression (VAR) model
///
/// A VAR(p) model expresses each variable as a linear combination of:
/// - Its own lagged values up to lag p
/// - Lagged values of all other variables up to lag p
/// - A constant term and potentially exogenous variables
pub struct VAR {
    /// Model order (number of lags)
    order: usize,
    /// Number of variables
    n_vars: usize,
    /// Coefficient matrices: coefficients[lag][var_to][var_from]
    coefficients: Vec<Array2<f64>>,
    /// Intercept vector
    intercept: Option<Array1<f64>>,
    /// Fitted flag
    is_fitted: bool,
    /// Residuals from fitted model
    residuals: Option<Array2<f64>>,
}

impl VAR {
    /// Create a new VAR model
    ///
    /// # Arguments
    /// * `order` - Number of lags (p in VAR(p))
    pub fn new(order: usize) -> Self {
        Self {
            order,
            n_vars: 0,
            coefficients: Vec::new(),
            intercept: None,
            is_fitted: false,
            residuals: None,
        }
    }

    /// Get model order
    pub fn order(&self) -> usize {
        self.order
    }

    /// Get number of variables
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Check if model is fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Fit VAR model to multivariate time series
    ///
    /// Uses Ordinary Least Squares (OLS) to estimate coefficients.
    /// For each variable, we solve: y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + c + e_t
    pub fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        // Validate input dimensions
        let shape = series.values.shape();
        let dims = shape.dims();

        if dims.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "VAR requires 2D time series (time x variables)".to_string(),
            ));
        }

        let n_obs = dims[0];
        let n_vars = dims[1];

        if n_obs <= self.order {
            return Err(TorshError::InvalidArgument(format!(
                "Insufficient observations: {} observations for VAR({}) model",
                n_obs, self.order
            )));
        }

        self.n_vars = n_vars;

        // Convert to ndarray for easier manipulation
        let data = series.values.to_vec()?;
        let y_matrix = Array2::from_shape_vec((n_obs, n_vars), data)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to create matrix: {}", e)))?;

        // Construct design matrix X and response matrix Y
        let n_effective = n_obs - self.order;

        // Y: response variables (from time t=order to end)
        let mut y_response = Array2::<f64>::zeros((n_effective, n_vars));
        for i in 0..n_effective {
            for j in 0..n_vars {
                y_response[[i, j]] = y_matrix[[i + self.order, j]] as f64;
            }
        }

        // X: lagged variables (design matrix)
        // Each row: [y_{t-1}, y_{t-2}, ..., y_{t-p}, 1] flattened
        let n_features = n_vars * self.order + 1; // +1 for intercept
        let mut x_design = Array2::<f64>::zeros((n_effective, n_features));

        for t in 0..n_effective {
            let mut col_idx = 0;

            // Add lagged values
            for lag in 1..=self.order {
                for var in 0..n_vars {
                    let time_idx = t + self.order - lag;
                    x_design[[t, col_idx]] = y_matrix[[time_idx, var]] as f64;
                    col_idx += 1;
                }
            }

            // Add intercept term
            x_design[[t, col_idx]] = 1.0;
        }

        // Solve OLS for each variable: β = (X'X)^(-1) X'y
        // Using proper Cholesky decomposition of the (n_features × n_features)
        // Gram matrix X'X, which is symmetric positive-semi-definite.
        let xt_x = x_design.t().dot(&x_design);
        let xt_y = x_design.t().dot(&y_response);

        // Flatten X'X into a row-major Vec<f64> for the Cholesky solver
        let xtx_flat: Vec<f64> = xt_x.iter().copied().collect();

        let mut all_coefficients = Array2::<f64>::zeros((n_vars, n_features));
        for var in 0..n_vars {
            // Extract the right-hand side column X'y[:,var]
            let rhs: Vec<f64> = (0..n_features).map(|i| xt_y[[i, var]]).collect();
            // Solve (X'X) β = X'y[:,var]  via Cholesky (with ridge fallback)
            let beta = var_cholesky_solve(&xtx_flat, &rhs, n_features);
            for i in 0..n_features {
                all_coefficients[[var, i]] = beta[i];
            }
        }

        // Extract coefficient matrices and intercept
        self.coefficients = Vec::with_capacity(self.order);
        for lag in 0..self.order {
            let mut coef_matrix = Array2::<f64>::zeros((n_vars, n_vars));
            for var_to in 0..n_vars {
                for var_from in 0..n_vars {
                    let col_idx = lag * n_vars + var_from;
                    coef_matrix[[var_to, var_from]] = all_coefficients[[var_to, col_idx]];
                }
            }
            self.coefficients.push(coef_matrix);
        }

        // Extract intercept
        let mut intercept = Array1::<f64>::zeros(n_vars);
        for var in 0..n_vars {
            intercept[var] = all_coefficients[[var, n_features - 1]];
        }
        self.intercept = Some(intercept);

        // Calculate residuals
        let predictions = self.predict_in_sample(&x_design)?;
        let mut residuals = Array2::<f64>::zeros((n_effective, n_vars));
        for i in 0..n_effective {
            for j in 0..n_vars {
                residuals[[i, j]] = y_response[[i, j]] - predictions[[i, j]];
            }
        }
        self.residuals = Some(residuals);

        self.is_fitted = true;
        Ok(())
    }

    /// Predict using design matrix (internal helper)
    fn predict_in_sample(&self, x_design: &Array2<f64>) -> Result<Array2<f64>> {
        let n_obs = x_design.nrows();
        let n_features = x_design.ncols();
        let n_vars = self.n_vars;

        let mut predictions = Array2::<f64>::zeros((n_obs, n_vars));

        // Reconstruct coefficient matrix
        let mut all_coefficients = Array2::<f64>::zeros((n_vars, n_features));

        for lag in 0..self.order {
            for var_to in 0..n_vars {
                for var_from in 0..n_vars {
                    let col_idx = lag * n_vars + var_from;
                    all_coefficients[[var_to, col_idx]] =
                        self.coefficients[lag][[var_to, var_from]];
                }
            }
        }

        // Add intercept
        if let Some(ref intercept) = self.intercept {
            for var in 0..n_vars {
                all_coefficients[[var, n_features - 1]] = intercept[var];
            }
        }

        // Compute predictions: Y_hat = X * β
        for obs in 0..n_obs {
            for var in 0..n_vars {
                let mut pred = 0.0;
                for feat in 0..n_features {
                    pred += x_design[[obs, feat]] * all_coefficients[[var, feat]];
                }
                predictions[[obs, var]] = pred;
            }
        }

        Ok(predictions)
    }

    /// Forecast h steps ahead
    ///
    /// Uses recursive forecasting: each prediction is used as input for the next step.
    pub fn forecast(&self, series: &TimeSeries, steps: usize) -> Result<TimeSeries> {
        if !self.is_fitted {
            return Err(TorshError::InvalidArgument(
                "Model must be fitted before forecasting".to_string(),
            ));
        }

        let shape = series.values.shape();
        let dims = shape.dims();
        let n_obs = dims[0];
        let n_vars = dims[1];

        if n_vars != self.n_vars {
            return Err(TorshError::InvalidArgument(format!(
                "Series has {} variables but model was fitted with {} variables",
                n_vars, self.n_vars
            )));
        }

        if n_obs < self.order {
            return Err(TorshError::InvalidArgument(
                "Insufficient observations for forecasting".to_string(),
            ));
        }

        // Get recent history
        let data = series.values.to_vec()?;
        let y_matrix = Array2::from_shape_vec((n_obs, n_vars), data)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to create matrix: {}", e)))?;

        // Initialize with last 'order' observations
        let mut history = Vec::with_capacity(self.order);
        for i in 0..self.order {
            let idx = n_obs - self.order + i;
            let mut obs = Array1::<f64>::zeros(n_vars);
            for j in 0..n_vars {
                obs[j] = y_matrix[[idx, j]] as f64;
            }
            history.push(obs);
        }

        // Forecast recursively
        let mut forecasts = Vec::with_capacity(steps * n_vars);

        for _step in 0..steps {
            // Construct current state vector
            let mut y_pred = if let Some(ref intercept) = self.intercept {
                intercept.clone()
            } else {
                Array1::<f64>::zeros(n_vars)
            };

            // Add contributions from lagged values
            for (lag_idx, lag_obs) in history.iter().rev().enumerate() {
                let coef_matrix = &self.coefficients[lag_idx];
                for var_to in 0..n_vars {
                    for var_from in 0..n_vars {
                        y_pred[var_to] += coef_matrix[[var_to, var_from]] * lag_obs[var_from];
                    }
                }
            }

            // Add to forecasts
            for var in 0..n_vars {
                forecasts.push(y_pred[var] as f32);
            }

            // Update history (sliding window)
            history.remove(0);
            history.push(y_pred);
        }

        let tensor = Tensor::from_vec(forecasts, &[steps, n_vars])?;
        Ok(TimeSeries::new(tensor))
    }

    /// Get Akaike Information Criterion (AIC)
    pub fn aic(&self) -> Result<f64> {
        if !self.is_fitted {
            return Err(TorshError::InvalidArgument(
                "Model must be fitted to compute AIC".to_string(),
            ));
        }

        let residuals = self
            .residuals
            .as_ref()
            .ok_or_else(|| TorshError::InvalidArgument("No residuals available".to_string()))?;

        let _n_obs = residuals.nrows() as f64;
        let n_params = (self.n_vars * self.n_vars * self.order + self.n_vars) as f64;

        // Calculate log-likelihood (assuming Gaussian errors)
        let mut log_likelihood = 0.0;
        let two_pi = 2.0 * std::f64::consts::PI;

        for i in 0..residuals.nrows() {
            for j in 0..residuals.ncols() {
                let residual = residuals[[i, j]];
                log_likelihood -= 0.5 * (two_pi.ln() + residual * residual);
            }
        }

        // AIC = -2*log(L) + 2*k
        Ok(-2.0 * log_likelihood + 2.0 * n_params)
    }

    /// Get Bayesian Information Criterion (BIC)
    pub fn bic(&self) -> Result<f64> {
        if !self.is_fitted {
            return Err(TorshError::InvalidArgument(
                "Model must be fitted to compute BIC".to_string(),
            ));
        }

        let residuals = self
            .residuals
            .as_ref()
            .ok_or_else(|| TorshError::InvalidArgument("No residuals available".to_string()))?;

        let n_obs = residuals.nrows() as f64;
        let n_params = (self.n_vars * self.n_vars * self.order + self.n_vars) as f64;

        // Calculate log-likelihood
        let mut log_likelihood = 0.0;
        let two_pi = 2.0 * std::f64::consts::PI;

        for i in 0..residuals.nrows() {
            for j in 0..residuals.ncols() {
                let residual = residuals[[i, j]];
                log_likelihood -= 0.5 * (two_pi.ln() + residual * residual);
            }
        }

        // BIC = -2*log(L) + k*log(n)
        Ok(-2.0 * log_likelihood + n_params * n_obs.ln())
    }

    /// Get Hannan-Quinn Information Criterion (HQIC)
    pub fn hqic(&self) -> Result<f64> {
        if !self.is_fitted {
            return Err(TorshError::InvalidArgument(
                "Model must be fitted to compute HQIC".to_string(),
            ));
        }

        let residuals = self
            .residuals
            .as_ref()
            .ok_or_else(|| TorshError::InvalidArgument("No residuals available".to_string()))?;

        let n_obs = residuals.nrows() as f64;
        let n_params = (self.n_vars * self.n_vars * self.order + self.n_vars) as f64;

        // Calculate log-likelihood
        let mut log_likelihood = 0.0;
        let two_pi = 2.0 * std::f64::consts::PI;

        for i in 0..residuals.nrows() {
            for j in 0..residuals.ncols() {
                let residual = residuals[[i, j]];
                log_likelihood -= 0.5 * (two_pi.ln() + residual * residual);
            }
        }

        // HQIC = -2*log(L) + 2*k*log(log(n))
        Ok(-2.0 * log_likelihood + 2.0 * n_params * n_obs.ln().ln())
    }

    /// Get coefficient matrix for a specific lag
    pub fn coefficients(&self, lag: usize) -> Result<&Array2<f64>> {
        if lag == 0 || lag > self.order {
            return Err(TorshError::InvalidArgument(format!(
                "Lag must be between 1 and {}",
                self.order
            )));
        }

        Ok(&self.coefficients[lag - 1])
    }

    /// Get intercept vector
    pub fn intercept(&self) -> Option<&Array1<f64>> {
        self.intercept.as_ref()
    }

    /// Get residuals
    pub fn residuals(&self) -> Option<&Array2<f64>> {
        self.residuals.as_ref()
    }
}

/// Granger causality test
///
/// Tests whether one time series is useful in forecasting another.
/// The null hypothesis is that x does not Granger-cause y.
pub struct GrangerCausality {
    max_lags: usize,
}

impl GrangerCausality {
    /// Create a new Granger causality test
    pub fn new(max_lags: usize) -> Self {
        Self { max_lags }
    }

    /// Test if x Granger-causes y
    ///
    /// Returns F-statistic and p-value for each lag up to max_lags
    pub fn test(&self, x: &TimeSeries, y: &TimeSeries) -> Result<Vec<(usize, f64, f64)>> {
        if x.len() != y.len() {
            return Err(TorshError::InvalidArgument(
                "Time series must have equal length".to_string(),
            ));
        }

        let mut results = Vec::new();

        for lag in 1..=self.max_lags {
            // Fit restricted model: y ~ lags(y)
            let mut restricted_series_data = Vec::new();
            for i in 0..y.len() {
                let y_val = y.values.get_item_flat(i)?;
                restricted_series_data.push(y_val);
            }
            let restricted_tensor = Tensor::from_vec(restricted_series_data, &[y.len(), 1])?;
            let restricted_series = TimeSeries::new(restricted_tensor);

            let mut restricted_model = VAR::new(lag);
            restricted_model.fit(&restricted_series)?;

            // Fit unrestricted model: y ~ lags(y) + lags(x)
            let mut unrestricted_series_data = Vec::new();
            for i in 0..y.len() {
                let y_val = y.values.get_item_flat(i)?;
                let x_val = x.values.get_item_flat(i)?;
                unrestricted_series_data.push(y_val);
                unrestricted_series_data.push(x_val);
            }
            let unrestricted_tensor = Tensor::from_vec(unrestricted_series_data, &[y.len(), 2])?;
            let unrestricted_series = TimeSeries::new(unrestricted_tensor);

            let mut unrestricted_model = VAR::new(lag);
            unrestricted_model.fit(&unrestricted_series)?;

            // Granger causality F-test on the y-equation only.
            //
            // Restricted model:   y_t = c + Σ a_i y_{t-i} + e_t                  (own lags only)
            // Unrestricted model: y_t = c + Σ a_i y_{t-i} + Σ b_i x_{t-i} + u_t  (+ lags of x)
            //
            //     F = ((RSS_r - RSS_u) / q) / (RSS_u / (n - k))
            //
            // The unrestricted VAR is fitted on the bivariate series [y, x]; only the
            // y-equation residuals (column 0) enter the test — the x-equation is irrelevant
            // to the hypothesis "x Granger-causes y". The restricted model is univariate,
            // so column 0 is again the y-equation. Because the unrestricted model nests the
            // restricted one (same response, same own lags, plus the lagged-x regressors),
            // RSS_u <= RSS_r holds in exact arithmetic.
            let rss_restricted = self.equation_rss(&restricted_model, 0)?;
            let rss_unrestricted = self.equation_rss(&unrestricted_model, 0)?;

            // q = number of restrictions = lagged-x coefficients zeroed under the null.
            let q = lag;
            // k = parameters in the unrestricted y-equation: own lags + x lags + intercept.
            let k = 2 * lag + 1;
            // n = effective sample size after lagging.
            let n_effective = y.len() - lag;

            // The denominator degrees of freedom must be positive for the test to exist.
            if n_effective <= k {
                return Err(TorshError::InvalidArgument(format!(
                    "Granger causality F-test at lag {lag} requires n > k: effective sample \
                     n = {n_effective} but the unrestricted model has k = {k} parameters; \
                     supply more observations or reduce max_lags"
                )));
            }
            let df_denominator = n_effective - k;

            // Data-relative tolerance for detecting a (numerically) perfect fit, so that
            // deterministic inputs stay well-defined instead of yielding a 0/0 statistic.
            let rss_floor = self.effective_response_tss(y, lag)?.max(1.0) * 1e-12;

            let f_stat = if rss_unrestricted <= rss_floor {
                // The unrestricted model already explains y to machine precision.
                let improvement = (rss_restricted - rss_unrestricted).max(0.0);
                if improvement <= rss_floor {
                    // Restricted model is also (numerically) perfect: the lagged-x terms add
                    // no detectable forecasting power, so there is no evidence of causality.
                    0.0
                } else {
                    // Lagged x removes essentially all remaining error from a non-trivial
                    // restricted RSS: the statistic diverges, collapsing the p-value to 0.
                    f64::INFINITY
                }
            } else {
                // Clamp away floating-point noise that could make the numerator negative.
                let numerator = (rss_restricted - rss_unrestricted).max(0.0) / q as f64;
                let denominator = rss_unrestricted / df_denominator as f64;
                numerator / denominator
            };

            // Upper-tail p-value of the F(q, n-k) distribution.
            let p_value = self.f_distribution_pvalue(f_stat, q, df_denominator);

            results.push((lag, f_stat.max(0.0), p_value));
        }

        Ok(results)
    }

    /// Residual sum of squares of a single fitted equation (target variable column).
    ///
    /// Granger causality compares the forecasting error of one equation (the y-equation),
    /// so the residuals of every other equation in a multivariate VAR must be excluded —
    /// summing over all columns (as a naive RSS would) corrupts the F-statistic.
    fn equation_rss(&self, model: &VAR, target_col: usize) -> Result<f64> {
        let residuals = model
            .residuals()
            .ok_or_else(|| TorshError::InvalidArgument("No residuals available".to_string()))?;

        if target_col >= residuals.ncols() {
            return Err(TorshError::InvalidArgument(format!(
                "target equation column {} out of range for a {}-equation model",
                target_col,
                residuals.ncols()
            )));
        }

        let mut rss = 0.0;
        for i in 0..residuals.nrows() {
            let r = residuals[[i, target_col]];
            rss += r * r;
        }

        Ok(rss)
    }

    /// Total (centred) sum of squares of the response actually modelled at the given lag.
    ///
    /// Both the restricted and unrestricted models predict `y_t` for `t = lag..n`, so the
    /// variation in that sub-series sets the natural scale of the residual sums of squares.
    /// It is used purely to derive a data-relative perfect-fit tolerance — never as a
    /// substitute for the real residuals.
    fn effective_response_tss(&self, y: &TimeSeries, lag: usize) -> Result<f64> {
        let n = y.len();
        if n <= lag {
            return Err(TorshError::InvalidArgument(format!(
                "series of length {n} is too short for lag {lag}"
            )));
        }

        let mut sum = 0.0;
        for i in lag..n {
            sum += y.values.get_item_flat(i)? as f64;
        }
        let count = (n - lag) as f64;
        let mean = sum / count;

        let mut tss = 0.0;
        for i in lag..n {
            let centred = y.values.get_item_flat(i)? as f64 - mean;
            tss += centred * centred;
        }

        Ok(tss)
    }

    /// Approximate F-distribution p-value using regularised incomplete beta function.
    fn f_distribution_pvalue(&self, f_stat: f64, df1: usize, df2: usize) -> f64 {
        if f_stat <= 0.0 || df1 == 0 || df2 == 0 {
            return 1.0;
        }
        // P(F_{df1,df2} > f_stat) = I_{z}(df2/2, df1/2)
        // where z = df2 / (df2 + df1 * f_stat)
        let d1 = df1 as f64;
        let d2 = df2 as f64;
        let z = d2 / (d2 + d1 * f_stat);
        regularised_incomplete_beta(z, d2 / 2.0, d1 / 2.0).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::Tensor;

    fn create_multivariate_series() -> TimeSeries {
        // Create a simple 2-variable time series
        let data = vec![
            1.0f32, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
        ];
        let tensor = Tensor::from_vec(data, &[8, 2]).expect("Tensor should succeed");
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_var_creation() {
        let var = VAR::new(2);
        assert_eq!(var.order(), 2);
        assert!(!var.is_fitted());
    }

    #[test]
    fn test_var_fit() {
        let series = create_multivariate_series();
        let mut var = VAR::new(2);
        var.fit(&series)
            .expect("fit operation should succeed with valid input");

        assert!(var.is_fitted());
        assert_eq!(var.n_vars(), 2);
    }

    #[test]
    fn test_var_forecast() {
        let series = create_multivariate_series();
        let mut var = VAR::new(1);
        var.fit(&series)
            .expect("fit operation should succeed with valid input");

        let forecast = var
            .forecast(&series, 3)
            .expect("forecast computation should succeed");
        assert_eq!(forecast.len(), 3);
        assert_eq!(forecast.num_features(), 2);
    }

    #[test]
    fn test_var_information_criteria() {
        let series = create_multivariate_series();
        let mut var = VAR::new(1);
        var.fit(&series)
            .expect("fit operation should succeed with valid input");

        let aic = var.aic().expect("AIC computation should succeed");
        let bic = var.bic().expect("BIC computation should succeed");
        let hqic = var.hqic().expect("HQIC computation should succeed");

        assert!(aic.is_finite());
        assert!(bic.is_finite());
        assert!(hqic.is_finite());
    }

    #[test]
    fn test_granger_causality() {
        // Create two simple series
        let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y_data = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let x_tensor = Tensor::from_vec(x_data, &[8]).expect("Tensor should succeed");
        let y_tensor = Tensor::from_vec(y_data, &[8]).expect("Tensor should succeed");

        let x_series = TimeSeries::new(x_tensor);
        let y_series = TimeSeries::new(y_tensor);

        let gc = GrangerCausality::new(2);
        let results = gc
            .test(&x_series, &y_series)
            .expect("statistical test should succeed");

        assert_eq!(results.len(), 2);
        for (lag, f_stat, p_value) in results {
            assert!(lag > 0);
            assert!(f_stat >= 0.0);
            assert!(p_value >= 0.0 && p_value <= 1.0);
        }
    }

    /// Build a synthetic `(x, y)` pair and return the lag-1 Granger F-statistic and
    /// p-value for the hypothesis "x Granger-causes y".
    ///
    /// `x` is always an independent AR(1) driver. `y` always has its own AR(1) dynamics;
    /// only when `caused` is true does `y_t` additionally depend on `x_{t-1}`. The random
    /// draws are identical for both settings (the causal term consumes no RNG), so the two
    /// scenarios share the exact same noise — a clean controlled comparison.
    fn granger_lag1(seed: u64, caused: bool) -> (f64, f64) {
        use scirs2_core::random::{Normal, Random};

        let n = 400usize;
        let mut rng = Random::seed(seed);
        let noise = Normal::new(0.0, 1.0).expect("normal distribution should construct");

        let mut x = vec![0.0f64; n];
        let mut y = vec![0.0f64; n];

        // Independent AR(1) driver for x.
        for t in 1..n {
            x[t] = 0.3 * x[t - 1] + rng.sample(noise);
        }
        // y has AR(1) dynamics; the causal case feeds in the previous value of x.
        for t in 1..n {
            let x_term = if caused { 0.8 * x[t - 1] } else { 0.0 };
            y[t] = 0.5 * y[t - 1] + x_term + rng.sample(noise);
        }

        let x_series = TimeSeries::new(
            Tensor::from_vec(x.iter().map(|&v| v as f32).collect::<Vec<f32>>(), &[n])
                .expect("x tensor should construct"),
        );
        let y_series = TimeSeries::new(
            Tensor::from_vec(y.iter().map(|&v| v as f32).collect::<Vec<f32>>(), &[n])
                .expect("y tensor should construct"),
        );

        let gc = GrangerCausality::new(2);
        let results = gc
            .test(&x_series, &y_series)
            .expect("granger causality test should succeed");

        let (lag, f_stat, p_value) = results[0];
        assert_eq!(lag, 1);
        (f_stat, p_value)
    }

    #[test]
    fn test_granger_causality_strong_when_x_causes_y() {
        // Y_t genuinely depends on X_{t-1}: X must show strong Granger causality on Y.
        let (f_caused, p_caused) = granger_lag1(20240617, true);

        assert!(
            f_caused.is_finite(),
            "F-statistic must be finite, got {f_caused}"
        );
        assert!(
            f_caused > 20.0,
            "genuine causation should yield a large F-statistic, got {f_caused}"
        );
        assert!(
            p_caused < 0.01,
            "genuine causation should yield a tiny p-value, got {p_caused}"
        );
    }

    #[test]
    fn test_granger_causality_weak_when_x_independent_of_y() {
        // Identical noise; the only difference is whether Y actually depends on X's lag.
        let (f_caused, _) = granger_lag1(20240617, true);
        let (f_independent, p_independent) = granger_lag1(20240617, false);

        // An independent X carries no forecasting power for Y -> small, non-significant F.
        assert!(
            f_independent < 10.0,
            "independent series should yield a small F-statistic, got {f_independent}"
        );
        assert!(
            p_independent > 0.05,
            "independent series should not be significant, got p = {p_independent}"
        );

        // Robust directional inequality: genuine causation dominates the independent case
        // by a wide margin (no exact value hard-coded).
        assert!(
            f_caused > 5.0 * f_independent,
            "causal F ({f_caused}) should greatly exceed independent F ({f_independent})"
        );
    }
}
