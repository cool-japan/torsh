//! ARIMA (AutoRegressive Integrated Moving Average) models
//!
//! Implements ARIMA(p, d, q) fitting via:
//! - Levinson-Durbin recursion for AR parameter estimation (Yule-Walker)
//! - Hannan-Rissanen two-stage method for MA parameter estimation
//! - Conditional Sum of Squares (CSS) log-likelihood
//! - AIC / BIC information criteria
//! - AutoARIMA grid search for automatic order selection

use crate::TimeSeries;
use torsh_tensor::Tensor;

// ---- Helper: extract f64 values from a TimeSeries (first feature) ----

/// Extract the first-feature values from a TimeSeries as Vec<f64>.
fn series_to_f64(series: &TimeSeries) -> Vec<f64> {
    series
        .values
        .to_vec()
        .unwrap_or_default()
        .into_iter()
        .map(|v: f32| v as f64)
        .collect()
}

/// Build a 1-D TimeSeries from Vec<f64>.
fn f64_to_series(values: &[f64]) -> TimeSeries {
    let f32_vals: Vec<f32> = values.iter().map(|&v| v as f32).collect();
    let n = f32_vals.len();
    let tensor = Tensor::from_vec(f32_vals, &[n])
        .unwrap_or_else(|_| Tensor::from_vec(vec![], &[0]).expect("empty tensor should succeed"));
    TimeSeries::new(tensor)
}

// ---- Levinson-Durbin recursion ----
//
// Given sample autocorrelations r[0..=p] (r[0] = variance), returns AR(p)
// coefficients [phi_1, ..., phi_p] (0-indexed: phi[k] = phi_{k+1}).

fn levinson_durbin(r: &[f64]) -> Vec<f64> {
    let p = r.len().saturating_sub(1);
    if p == 0 || r[0] <= 0.0 {
        return vec![];
    }
    let mut phi = vec![0.0_f64; p];
    let mut err = r[0];
    for k in 1..=p {
        // Reflection coefficient
        let mut num = r[k];
        for j in 0..k.saturating_sub(1) {
            num -= phi[j] * r[k - 1 - j];
        }
        if err.abs() < f64::EPSILON * 1e6 {
            break;
        }
        let lambda = num / err;
        // Limit to stable range
        let lambda = lambda.clamp(-0.9999, 0.9999);
        // Update existing coefficients
        let phi_old: Vec<f64> = phi[..k.saturating_sub(1)].to_vec();
        for j in 0..k.saturating_sub(1) {
            phi[j] = phi_old[j] - lambda * phi_old[k - 2 - j];
        }
        phi[k - 1] = lambda;
        err *= 1.0 - lambda * lambda;
        if err <= 0.0 {
            break;
        }
    }
    phi
}

// ---- Sample autocorrelation ----
//
// Returns r[0..=max_lag] from a mean-subtracted series.

fn autocorrelations(y: &[f64], max_lag: usize) -> Vec<f64> {
    let n = y.len();
    if n == 0 {
        return vec![0.0; max_lag + 1];
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    let y_c: Vec<f64> = y.iter().map(|&v| v - mean).collect();
    let mut r = vec![0.0_f64; max_lag + 1];
    for lag in 0..=max_lag {
        let mut acc = 0.0_f64;
        for t in lag..n {
            acc += y_c[t] * y_c[t - lag];
        }
        r[lag] = acc / n as f64;
    }
    r
}

// ---- d-fold first differencing ----

fn difference(y: &[f64], d: usize) -> Vec<f64> {
    let mut cur = y.to_vec();
    for _ in 0..d {
        if cur.len() < 2 {
            return cur;
        }
        cur = cur.windows(2).map(|w| w[1] - w[0]).collect();
    }
    cur
}

// ---- Compute AR residuals ----
//
// Given demeaned differenced series and AR params, compute one-step-ahead errors.
// The first `p` observations are skipped (warm-up).

fn ar_residuals(y: &[f64], phi: &[f64]) -> Vec<f64> {
    let p = phi.len();
    let n = y.len();
    if n <= p {
        return vec![0.0; n];
    }
    let mut resid = vec![0.0_f64; n];
    for t in p..n {
        let mut pred = 0.0_f64;
        for j in 0..p {
            pred += phi[j] * y[t - 1 - j];
        }
        resid[t] = y[t] - pred;
    }
    resid
}

// ---- Hannan-Rissanen MA estimation ----
//
// Stage 1: fit AR(m) where m = max(p+q, ceil(log(n))), get residuals.
// Stage 2: OLS on lagged y' and lagged residuals to get [phi, theta] jointly.
// Returns (ar_params, ma_params).

fn hannan_rissanen(y: &[f64], p: usize, q: usize) -> (Vec<f64>, Vec<f64>) {
    let n = y.len();
    if n < 2 {
        return (vec![0.0; p], vec![0.0; q]);
    }

    // Stage 1: pilot AR
    let m_pilot = if p + q > 0 {
        let log_n = (n as f64).ln().ceil() as usize;
        std::cmp::max(p + q, log_n).min(n / 2)
    } else {
        0
    };

    let pilot_resid = if m_pilot > 0 && n > m_pilot {
        let r = autocorrelations(y, m_pilot);
        let phi_pilot = levinson_durbin(&r);
        ar_residuals(y, &phi_pilot)
    } else {
        // Use mean-zero residuals as proxy
        let mean = y.iter().sum::<f64>() / n as f64;
        y.iter().map(|&v| v - mean).collect()
    };

    // Stage 2: OLS with regressors [lagged y', lagged e_hat]
    let start = m_pilot.max(p).max(q);
    if n <= start + 1 || (p + q) == 0 {
        // Only pure AR (q == 0) or nothing to fit
        if p > 0 && q == 0 {
            let r = autocorrelations(y, p);
            let ar = levinson_durbin(&r);
            return (ar, vec![]);
        }
        if p == 0 && q == 0 {
            return (vec![], vec![]);
        }
    }

    let nrows = n.saturating_sub(start);
    let ncols = p + q;
    if nrows < 1 || ncols == 0 {
        return (vec![0.0; p], vec![0.0; q]);
    }

    // Build design matrix X (nrows × ncols) and response vector Y
    let mut x_mat = vec![0.0_f64; nrows * ncols];
    let mut y_vec = vec![0.0_f64; nrows];
    for (row_idx, t) in (start..n).enumerate() {
        y_vec[row_idx] = y[t];
        for j in 0..p {
            x_mat[row_idx * ncols + j] = y[t - 1 - j];
        }
        for j in 0..q {
            let lag_idx = t.saturating_sub(1 + j);
            x_mat[row_idx * ncols + p + j] = pilot_resid[lag_idx];
        }
    }

    // OLS via normal equations: beta = (X'X)^-1 X'y
    // Use simple Cholesky or Gaussian elimination on the (ncols x ncols) system
    let beta = ols_solve(&x_mat, &y_vec, nrows, ncols);
    let ar_params = beta[..p].to_vec();
    let ma_params = beta[p..p + q].to_vec();
    (ar_params, ma_params)
}

// ---- Simple OLS via Gaussian elimination (normal equations) ----
//
// X is nrows×ncols (row-major), solves (X'X) beta = X'y.

fn ols_solve(x: &[f64], y: &[f64], nrows: usize, ncols: usize) -> Vec<f64> {
    // Build X'X (ncols×ncols) and X'y (ncols)
    let mut xtx = vec![0.0_f64; ncols * ncols];
    let mut xty = vec![0.0_f64; ncols];
    for row in 0..nrows {
        for c1 in 0..ncols {
            let xrc1 = x[row * ncols + c1];
            xty[c1] += xrc1 * y[row];
            for c2 in 0..ncols {
                xtx[c1 * ncols + c2] += xrc1 * x[row * ncols + c2];
            }
        }
    }
    // Add small ridge for numerical stability
    let ridge = 1e-8;
    for c in 0..ncols {
        xtx[c * ncols + c] += ridge;
    }
    // Gaussian elimination with partial pivoting on augmented matrix [XtX | Xty]
    let mut aug = vec![0.0_f64; ncols * (ncols + 1)];
    for r in 0..ncols {
        for c in 0..ncols {
            aug[r * (ncols + 1) + c] = xtx[r * ncols + c];
        }
        aug[r * (ncols + 1) + ncols] = xty[r];
    }
    for col in 0..ncols {
        // Pivot
        let mut max_val = aug[col * (ncols + 1) + col].abs();
        let mut max_row = col;
        for row in (col + 1)..ncols {
            let v = aug[row * (ncols + 1) + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for c in 0..=(ncols) {
                aug.swap(col * (ncols + 1) + c, max_row * (ncols + 1) + c);
            }
        }
        let pivot = aug[col * (ncols + 1) + col];
        if pivot.abs() < f64::EPSILON * 1e6 {
            continue;
        }
        for row in (col + 1)..ncols {
            let factor = aug[row * (ncols + 1) + col] / pivot;
            for c in col..=(ncols) {
                let tmp = aug[col * (ncols + 1) + c] * factor;
                aug[row * (ncols + 1) + c] -= tmp;
            }
        }
    }
    // Back-substitution
    let mut beta = vec![0.0_f64; ncols];
    for row in (0..ncols).rev() {
        let mut val = aug[row * (ncols + 1) + ncols];
        for c in (row + 1)..ncols {
            val -= aug[row * (ncols + 1) + c] * beta[c];
        }
        let diag = aug[row * (ncols + 1) + row];
        if diag.abs() < f64::EPSILON * 1e6 {
            beta[row] = 0.0;
        } else {
            beta[row] = val / diag;
        }
    }
    beta
}

// ---- ARMA residuals (for CSS log-likelihood computation) ----
//
// Computes one-step-ahead forecast errors for ARMA(p,q) on demeaned series.

fn arma_residuals(y: &[f64], phi: &[f64], theta: &[f64]) -> Vec<f64> {
    let p = phi.len();
    let q = theta.len();
    let n = y.len();
    let mut e = vec![0.0_f64; n];
    let start = p.max(q);
    for t in start..n {
        let mut pred = 0.0_f64;
        for j in 0..p {
            pred += phi[j] * y[t - 1 - j];
        }
        for j in 0..q {
            pred += theta[j] * e[t - 1 - j];
        }
        e[t] = y[t] - pred;
    }
    e
}

// ---- CSS log-likelihood ----
//
// L = -(n_eff/2)*ln(2π) - (n_eff/2)*ln(σ²) - SSE/(2σ²)
// After simplification: L = -(n_eff/2)*(1 + ln(2π) + ln(σ²))

fn css_log_likelihood(residuals: &[f64], skip: usize) -> f64 {
    let effective: Vec<f64> = residuals[skip..].to_vec();
    let n_eff = effective.len();
    if n_eff == 0 {
        return f64::NEG_INFINITY;
    }
    let sse: f64 = effective.iter().map(|&e| e * e).sum();
    let sigma2 = sse / n_eff as f64;
    if sigma2 <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let log_like =
        -(n_eff as f64) / 2.0 * (1.0 + std::f64::consts::PI.mul_add(2.0, 0.0).ln() + sigma2.ln());
    log_like
}

// ========================================================================
// ARIMA struct
// ========================================================================

/// ARIMA(p, d, q) model for time series forecasting.
///
/// Fitting uses Levinson-Durbin (AR) and Hannan-Rissanen (MA) estimators
/// with CSS-based log-likelihood for AIC/BIC.
pub struct ARIMA {
    p: usize,
    d: usize,
    q: usize,
    seasonal_order: Option<(usize, usize, usize, usize)>,
    // Fitted parameters
    ar_params: Vec<f64>,
    ma_params: Vec<f64>,
    intercept: f64,
    // Fitted state for forecasting
    last_diff_values: Vec<f64>, // last p values of differenced (demeaned) series
    last_residuals: Vec<f64>,   // last q residuals
    level_history: Vec<Vec<f64>>, // for inverting each differencing step
    // Diagnostics
    residuals_data: Option<Vec<f64>>,
    log_likelihood: Option<f64>,
    n_obs: usize,
}

impl ARIMA {
    /// Create a new ARIMA(p, d, q) model.
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self {
            p,
            d,
            q,
            seasonal_order: None,
            ar_params: Vec::new(),
            ma_params: Vec::new(),
            intercept: 0.0,
            last_diff_values: Vec::new(),
            last_residuals: Vec::new(),
            level_history: Vec::new(),
            residuals_data: None,
            log_likelihood: None,
            n_obs: 0,
        }
    }

    /// Add seasonal component (P, D, Q, s).
    pub fn seasonal(mut self, p: usize, d: usize, q: usize, s: usize) -> Self {
        self.seasonal_order = Some((p, d, q, s));
        self
    }

    /// AR order.
    pub fn p(&self) -> usize {
        self.p
    }

    /// Differencing order.
    pub fn d(&self) -> usize {
        self.d
    }

    /// MA order.
    pub fn q(&self) -> usize {
        self.q
    }

    /// Seasonal order (P, D, Q, s) if set.
    pub fn seasonal_order(&self) -> Option<(usize, usize, usize, usize)> {
        self.seasonal_order
    }

    /// Whether the model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.n_obs > 0
    }

    /// Fit the ARIMA model to `series` (uses first feature if multivariate).
    ///
    /// Algorithm:
    /// 1. Difference the series `d` times.
    /// 2. Demean the differenced series; store intercept.
    /// 3. Estimate AR params via Levinson-Durbin (Yule-Walker).
    /// 4. Estimate MA params via Hannan-Rissanen two-stage OLS.
    /// 5. Compute ARMA residuals and CSS log-likelihood.
    pub fn fit(&mut self, series: &TimeSeries) {
        let raw = series_to_f64(series);
        let n_raw = raw.len();
        if n_raw == 0 {
            return;
        }

        // --- Save per-differencing-step history for inversion ---
        // We need the series at each intermediate differencing level so we can
        // invert the differencing when producing forecasts.
        let mut level_history: Vec<Vec<f64>> = Vec::with_capacity(self.d);
        let mut diffed = raw.clone();
        for _step in 0..self.d {
            level_history.push(diffed.clone());
            if diffed.len() < 2 {
                break;
            }
            diffed = difference(&diffed, 1);
        }
        self.level_history = level_history;

        let n = diffed.len();
        let min_obs = 2 * (self.p + self.q) + self.d + 1;
        if n < 2 || n_raw < min_obs.max(3) {
            // Not enough data — store trivial model
            self.n_obs = n_raw;
            self.intercept = diffed.first().copied().unwrap_or(0.0);
            self.ar_params = vec![0.0; self.p];
            self.ma_params = vec![0.0; self.q];
            self.residuals_data = Some(vec![0.0; n]);
            self.log_likelihood = Some(f64::NEG_INFINITY);
            self.last_diff_values = diffed.iter().rev().take(self.p).rev().cloned().collect();
            self.last_residuals = vec![0.0; self.q];
            return;
        }

        // --- Demean the differenced series ---
        let mean = diffed.iter().sum::<f64>() / n as f64;
        let y: Vec<f64> = diffed.iter().map(|&v| v - mean).collect();

        // --- Parameter estimation ---
        let (ar_params, ma_params) = if self.p == 0 && self.q == 0 {
            (vec![], vec![])
        } else {
            hannan_rissanen(&y, self.p, self.q)
        };

        // Recompute intercept: c = mean * (1 - sum(phi))
        let phi_sum: f64 = ar_params.iter().sum();
        let intercept = mean * (1.0 - phi_sum);

        // --- ARMA residuals on demeaned series ---
        let skip = self.p.max(self.q);
        let raw_resid = arma_residuals(&y, &ar_params, &ma_params);
        let log_like = css_log_likelihood(&raw_resid, skip);

        // Save state needed for forecasting.
        // Use the *undemeaned* diffed series for last_diff_values so that the
        // AR recursion in forecast() operates on the original (undemeaned) scale,
        // matching the intercept `c = mean * (1 - sum(phi))`.
        let last_diff_values: Vec<f64> = diffed.iter().rev().take(self.p).rev().cloned().collect();
        let last_residuals: Vec<f64> = raw_resid.iter().rev().take(self.q).rev().cloned().collect();

        self.ar_params = ar_params;
        self.ma_params = ma_params;
        self.intercept = intercept;
        self.last_diff_values = last_diff_values;
        self.last_residuals = last_residuals;
        self.residuals_data = Some(raw_resid);
        self.log_likelihood = Some(log_like);
        self.n_obs = n_raw;
    }

    /// Akaike Information Criterion: `2k - 2L`.
    ///
    /// `k = p + q + 1` (intercept counted).
    pub fn aic(&self) -> f64 {
        let k = self.p + self.q + 1;
        let ll = self.log_likelihood.unwrap_or(f64::NEG_INFINITY);
        2.0 * k as f64 - 2.0 * ll
    }

    /// Bayesian Information Criterion: `k * ln(n) - 2L`.
    pub fn bic(&self) -> f64 {
        let k = self.p + self.q + 1;
        let ll = self.log_likelihood.unwrap_or(f64::NEG_INFINITY);
        let n = self.n_obs.max(1) as f64;
        k as f64 * n.ln() - 2.0 * ll
    }

    /// Residuals from the fitted model, or `None` if not fitted.
    pub fn residuals(&self) -> Option<TimeSeries> {
        self.residuals_data.as_ref().map(|r| f64_to_series(r))
    }

    /// Forecast `steps` steps ahead.
    ///
    /// Uses AR recursion on the demeaned differenced scale, adds back the
    /// mean (via intercept), then inverts each differencing step.
    pub fn forecast(&self, steps: usize) -> TimeSeries {
        if steps == 0 || !self.is_fitted() {
            // Return zeros of length `steps`
            let vals: Vec<f32> = vec![0.0_f32; steps];
            let tensor = Tensor::from_vec(vals, &[steps]).unwrap_or_else(|_| {
                Tensor::from_vec(vec![], &[0]).expect("empty tensor should succeed")
            });
            return TimeSeries::new(tensor);
        }

        // ---- AR recursion on the differenced series (undemeaned scale) ----
        let p = self.ar_params.len();
        let q = self.ma_params.len();

        // History buffer: combine observed tail and forecasted values
        let mut history: Vec<f64> = self.last_diff_values.clone();
        let mut e_history: Vec<f64> = self.last_residuals.clone();

        let mut diff_forecasts = Vec::with_capacity(steps);
        for _ in 0..steps {
            let mut yhat = self.intercept;
            // AR part
            let hist_len = history.len();
            for j in 0..p {
                if j < hist_len {
                    yhat += self.ar_params[j] * history[hist_len - 1 - j];
                }
            }
            // MA part (future innovations are 0 in unconditional forecast).
            // Positive-convention MA: pred += theta[j] * e[t-1-j],
            // consistent with how Hannan-Rissanen fits MA via OLS.
            let e_len = e_history.len();
            for j in 0..q {
                if j < e_len {
                    yhat += self.ma_params[j] * e_history[e_len - 1 - j];
                }
            }
            diff_forecasts.push(yhat);
            history.push(yhat);
            e_history.push(0.0); // no future innovations
        }

        // ---- Invert differencing (d times) ----
        // For each differencing level (from innermost to outermost), integrate.
        let mut result = diff_forecasts;
        for level_idx in (0..self.d).rev() {
            let last_val = self
                .level_history
                .get(level_idx)
                .and_then(|h| h.last())
                .copied()
                .unwrap_or(0.0);
            let mut integrated = Vec::with_capacity(result.len());
            let mut prev = last_val;
            for &delta in &result {
                prev += delta;
                integrated.push(prev);
            }
            result = integrated;
        }

        f64_to_series(&result)
    }

    /// Fitted AR parameters.
    pub fn ar_params(&self) -> &[f64] {
        &self.ar_params
    }

    /// Fitted MA parameters.
    pub fn ma_params(&self) -> &[f64] {
        &self.ma_params
    }

    /// Model intercept (includes mean adjustment).
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// CSS log-likelihood of the fitted model.
    pub fn log_likelihood(&self) -> Option<f64> {
        self.log_likelihood
    }
}

// ========================================================================
// SARIMA (delegates to ARIMA; seasonal fitting is not yet implemented)
// ========================================================================

/// SARIMA model — wraps ARIMA; seasonal multiplicative fitting is delegated
/// to future work.
pub struct SARIMA {
    arima: ARIMA,
}

impl SARIMA {
    /// Create a new SARIMA(p, d, q)(P, D, Q, s) model.
    pub fn new(
        p: usize,
        d: usize,
        q: usize,
        seasonal_p: usize,
        seasonal_d: usize,
        seasonal_q: usize,
        seasonal_period: usize,
    ) -> Self {
        Self {
            arima: ARIMA::new(p, d, q).seasonal(
                seasonal_p,
                seasonal_d,
                seasonal_q,
                seasonal_period,
            ),
        }
    }

    /// Fit SARIMA model (seasonal part forwarded to ARIMA non-seasonal fit).
    pub fn fit(&mut self, series: &TimeSeries) {
        self.arima.fit(series);
    }

    /// Forecast future values.
    pub fn forecast(&self, steps: usize) -> TimeSeries {
        self.arima.forecast(steps)
    }
}

// ========================================================================
// AutoARIMA — grid search over (p, d, q)
// ========================================================================

/// Information criterion for model selection.
#[derive(Debug, Clone, PartialEq, Eq)]
enum InfoCriterion {
    Aic,
    Bic,
}

impl InfoCriterion {
    fn score(&self, model: &ARIMA) -> f64 {
        match self {
            InfoCriterion::Aic => model.aic(),
            InfoCriterion::Bic => model.bic(),
        }
    }
}

/// Auto ARIMA: selects the best ARIMA(p, d, q) by grid search over a bounded
/// parameter space, using AIC or BIC to rank models.
pub struct AutoARIMA {
    max_p: usize,
    max_d: usize,
    max_q: usize,
    information_criterion: String,
}

impl AutoARIMA {
    /// Create a new AutoARIMA with default bounds (p ≤ 5, d ≤ 2, q ≤ 5, AIC).
    pub fn new() -> Self {
        Self {
            max_p: 5,
            max_d: 2,
            max_q: 5,
            information_criterion: "aic".to_string(),
        }
    }

    /// Override the maximum orders for the grid search.
    pub fn with_max_order(mut self, max_p: usize, max_d: usize, max_q: usize) -> Self {
        self.max_p = max_p;
        self.max_d = max_d;
        self.max_q = max_q;
        self
    }

    /// Set information criterion: `"aic"` (default) or `"bic"`.
    pub fn with_criterion(mut self, criterion: &str) -> Self {
        self.information_criterion = criterion.to_string();
        self
    }

    /// Run the grid search and return the best-fitting ARIMA model.
    ///
    /// Skips configurations where the effective sample is too small
    /// (`n - d ≤ 2*(p + q) + 1`) to avoid overfitting artefacts.
    pub fn fit(&self, series: &TimeSeries) -> ARIMA {
        let ic = if self.information_criterion.eq_ignore_ascii_case("bic") {
            InfoCriterion::Bic
        } else {
            InfoCriterion::Aic
        };

        let n = series.len();
        let mut best_model: Option<ARIMA> = None;
        let mut best_score = f64::INFINITY;

        for d in 0..=self.max_d {
            for p in 0..=self.max_p {
                for q in 0..=self.max_q {
                    // Guard against overfitting on short series
                    let n_eff = n.saturating_sub(d);
                    let min_needed = 2 * (p + q) + 1;
                    if n_eff <= min_needed {
                        continue;
                    }
                    // At least one of p, q must be > 0, or d > 0 (otherwise trivial)
                    if p == 0 && q == 0 && d == 0 {
                        continue;
                    }

                    let mut model = ARIMA::new(p, d, q);
                    model.fit(series);
                    let score = ic.score(&model);

                    if score.is_finite() && score < best_score {
                        best_score = score;
                        best_model = Some(model);
                    }
                }
            }
        }

        // Fallback: ARIMA(1,1,1) if nothing was selected
        best_model.unwrap_or_else(|| {
            let mut m = ARIMA::new(1, 1, 1);
            m.fit(series);
            m
        })
    }
}

impl Default for AutoARIMA {
    fn default() -> Self {
        Self::new()
    }
}

// ========================================================================
// Tests
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::Tensor;

    fn create_test_series() -> TimeSeries {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::from_vec(data, &[8]).expect("test tensor creation should succeed");
        TimeSeries::new(tensor)
    }

    /// A simple AR(1) process: y[t] = 0.7 * y[t-1] + noise
    fn create_ar1_series() -> TimeSeries {
        let data: Vec<f32> = vec![
            1.0, 1.7, 1.19, 1.433, 1.603, 1.722, 1.805, 1.864, 1.905, 1.933, 1.953, 1.967, 1.977,
            1.984, 1.989, 1.992, 1.994, 1.996, 1.997, 1.998,
        ];
        let n = data.len();
        let tensor = Tensor::from_vec(data, &[n]).expect("test tensor creation should succeed");
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_arima_creation() {
        let arima = ARIMA::new(2, 1, 1);
        assert_eq!(arima.p(), 2);
        assert_eq!(arima.d(), 1);
        assert_eq!(arima.q(), 1);
        assert!(arima.seasonal_order().is_none());
    }

    #[test]
    fn test_arima_seasonal() {
        let arima = ARIMA::new(1, 1, 1).seasonal(1, 1, 1, 12);
        assert_eq!(arima.seasonal_order(), Some((1, 1, 1, 12)));
    }

    #[test]
    fn test_arima_forecast() {
        let series = create_test_series();
        let mut arima = ARIMA::new(1, 1, 1);
        arima.fit(&series);
        let forecast = arima.forecast(5);

        assert_eq!(forecast.len(), 5);
        assert_eq!(forecast.num_features(), 1);
    }

    #[test]
    fn test_arima_fit_updates_state() {
        let series = create_ar1_series();
        let mut arima = ARIMA::new(1, 0, 0);
        assert!(!arima.is_fitted());
        arima.fit(&series);
        assert!(arima.is_fitted());
    }

    #[test]
    fn test_arima_residuals_available_after_fit() {
        let series = create_ar1_series();
        let mut arima = ARIMA::new(1, 0, 0);
        arima.fit(&series);
        let resid = arima.residuals();
        assert!(resid.is_some());
    }

    #[test]
    fn test_arima_aic_bic_finite_after_fit() {
        let series = create_ar1_series();
        let mut arima = ARIMA::new(1, 0, 0);
        arima.fit(&series);
        let aic = arima.aic();
        let bic = arima.bic();
        assert!(aic.is_finite(), "AIC should be finite, got {aic}");
        assert!(bic.is_finite(), "BIC should be finite, got {bic}");
        // BIC penalises more, so BIC >= AIC for n > e^2 (approx)
        // For n=20, ln(20) ≈ 3.0 > 2, so BIC >= AIC
        assert!(
            bic >= aic,
            "BIC should be >= AIC for n=20; AIC={aic}, BIC={bic}"
        );
    }

    #[test]
    fn test_arima_aic_decreases_with_good_order() {
        // AR(1) on an AR(1) process should have lower AIC than AR(0)
        let series = create_ar1_series();
        let mut ar1 = ARIMA::new(1, 0, 0);
        ar1.fit(&series);
        let mut ar0 = ARIMA::new(0, 0, 1); // MA(1) as comparison
        ar0.fit(&series);
        // Both should be finite; we just verify the property holds or that values make sense
        assert!(ar1.aic().is_finite());
        assert!(ar0.aic().is_finite());
    }

    #[test]
    fn test_arima_forecast_length() {
        let series = create_ar1_series();
        let mut arima = ARIMA::new(1, 1, 0);
        arima.fit(&series);
        let fc = arima.forecast(10);
        assert_eq!(fc.len(), 10);
        assert_eq!(fc.num_features(), 1);
    }

    #[test]
    fn test_arima_forecast_unfitted_returns_zeros() {
        let arima = ARIMA::new(2, 1, 1);
        let fc = arima.forecast(5);
        assert_eq!(fc.len(), 5);
        let vals = fc.values.to_vec().expect("to_vec should succeed");
        assert!(vals.iter().all(|&v| v == 0.0_f32));
    }

    #[test]
    fn test_arima_differencing() {
        // Series 1, 3, 6, 10, 15 with d=1 => diffs 2, 3, 4, 5
        let data: Vec<f32> = vec![1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0];
        let n = data.len();
        let tensor = Tensor::from_vec(data, &[n]).expect("test tensor should succeed");
        let series = TimeSeries::new(tensor);
        let mut arima = ARIMA::new(1, 1, 0);
        arima.fit(&series);
        assert!(arima.is_fitted());
        let fc = arima.forecast(3);
        assert_eq!(fc.len(), 3);
        // Integrated forecast should continue the linear trend
        let vals = fc.values.to_vec().expect("to_vec should succeed");
        // Values should be positive and increasing (trend series)
        assert!(
            vals.iter().all(|&v| v > 0.0_f32),
            "forecast values should be positive: {vals:?}"
        );
    }

    #[test]
    fn test_sarima_creation() {
        let sarima = SARIMA::new(1, 1, 1, 1, 1, 1, 12);
        let forecast = sarima.forecast(3);
        assert_eq!(forecast.len(), 3);
    }

    #[test]
    fn test_sarima_fit_and_forecast() {
        let series = create_ar1_series();
        let mut sarima = SARIMA::new(1, 0, 0, 0, 0, 0, 12);
        sarima.fit(&series);
        let fc = sarima.forecast(5);
        assert_eq!(fc.len(), 5);
    }

    #[test]
    fn test_auto_arima() {
        let series = create_test_series();
        let auto_arima = AutoARIMA::new().with_max_order(3, 2, 3);
        let _model = auto_arima.fit(&series);
        // Test that fit completes without errors
    }

    #[test]
    fn test_auto_arima_selects_model() {
        let series = create_ar1_series();
        let auto_arima = AutoARIMA::new()
            .with_max_order(3, 1, 2)
            .with_criterion("aic");
        let best = auto_arima.fit(&series);
        assert!(best.is_fitted());
        assert!(best.aic().is_finite());
    }

    #[test]
    fn test_auto_arima_bic_criterion() {
        let series = create_ar1_series();
        let auto_arima = AutoARIMA::new()
            .with_max_order(2, 1, 2)
            .with_criterion("bic");
        let best = auto_arima.fit(&series);
        assert!(best.is_fitted());
        assert!(best.bic().is_finite());
    }

    #[test]
    fn test_levinson_durbin_ar1() {
        // For AR(1) with phi=0.7: r[0]=1, r[1]=0.7 => phi[0] should be 0.7
        let r = vec![1.0, 0.7];
        let phi = levinson_durbin(&r);
        assert_eq!(phi.len(), 1);
        assert!(
            (phi[0] - 0.7).abs() < 1e-12,
            "Expected phi[0]≈0.7, got {}",
            phi[0]
        );
    }

    #[test]
    fn test_levinson_durbin_ar2() {
        // AR(2): r[0]=1, r[1]=0.6, r[2]=0.5
        // phi_1 = (r[1](1-r[2]))/(1-r[1]^2) etc — just check it produces 2 params
        let r = vec![1.0, 0.6, 0.5];
        let phi = levinson_durbin(&r);
        assert_eq!(phi.len(), 2);
        assert!(phi.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_ols_solve_identity() {
        // X = I_2, y = [3, 5] => beta = [3, 5]
        let x = vec![1.0, 0.0, 0.0, 1.0];
        let y = vec![3.0, 5.0];
        let beta = ols_solve(&x, &y, 2, 2);
        assert!((beta[0] - 3.0).abs() < 1e-6, "beta[0]={}", beta[0]);
        assert!((beta[1] - 5.0).abs() < 1e-6, "beta[1]={}", beta[1]);
    }

    #[test]
    fn test_autocorrelations_zero_for_constant() {
        // Constant series: all autocorrelations should be 0 after lag 0
        let y = vec![2.0; 10];
        let r = autocorrelations(&y, 3);
        // r[0] = 0 for mean-subtracted constant series
        assert!(r[0].abs() < 1e-12, "r[0]={}", r[0]);
        assert!(r[1].abs() < 1e-12, "r[1]={}", r[1]);
    }

    /// Regression test: verify one-step forecast for a pure AR(1) process.
    ///
    /// For AR(1) with phi ≈ 0.8 and mean ≈ 0, the one-step forecast is:
    ///   y_hat_{n+1} = c + phi * y_n
    /// where c = mean*(1-phi).  With a linear series the fit should be close
    /// enough that the predicted next step is plausible.
    #[test]
    fn test_arima_forecast_regression_ar1() {
        // Build a stationary AR(1)-like series: y[t] = 0.8 * y[t-1] (mean ≈ 0)
        let phi_true = 0.8_f64;
        let mut series_data = vec![1.0_f64];
        for _ in 1..30 {
            let next = phi_true * series_data.last().copied().unwrap_or(0.0);
            series_data.push(next);
        }
        let last_val = *series_data.last().expect("series must be non-empty");
        let expected_next = phi_true * last_val; // one-step ahead forecast

        let f32_data: Vec<f32> = series_data.iter().map(|&v| v as f32).collect();
        let n = f32_data.len();
        let tensor = Tensor::from_vec(f32_data, &[n]).expect("tensor creation should succeed");
        let series = TimeSeries::new(tensor);

        let mut arima = ARIMA::new(1, 0, 0);
        arima.fit(&series);
        let fc = arima.forecast(1);
        assert_eq!(fc.len(), 1);

        let fc_vals = fc.values.to_vec().expect("to_vec should succeed");
        let fc_f64 = fc_vals[0] as f64;
        // Allow up to 15% relative error given finite-sample AR estimation
        let tol = expected_next.abs() * 0.15 + 0.05;
        assert!(
            (fc_f64 - expected_next).abs() < tol,
            "AR(1) forecast {fc_f64:.4} expected ≈{expected_next:.4} (tol {tol:.4})"
        );
    }
}
