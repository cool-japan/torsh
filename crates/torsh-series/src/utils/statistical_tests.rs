//! Statistical tests for time series analysis
//!
//! This module provides various statistical hypothesis tests for time series,
//! including stationarity tests, autocorrelation tests, and normality tests.
//!
//! NOTE: These implementations use placeholder calculations until scirs2-stats
//! provides the full hypothesis testing API.

use crate::TimeSeries;
use torsh_core::error::{Result, TorshError};

// Placeholder result types for statistical tests
// These will be replaced with scirs2-stats types when available

/// Result of Augmented Dickey-Fuller test
#[derive(Debug, Clone)]
pub struct ADFResult {
    pub test_statistic: f64,
    pub p_value: f64,
    pub n_lags: usize,
    pub n_obs: usize,
    pub critical_values: Vec<(String, f64)>,
}

/// Result of Ljung-Box test
#[derive(Debug, Clone)]
pub struct LjungBoxResult {
    pub test_statistic: f64,
    pub p_value: f64,
    pub df: usize,
}

/// Result of Jarque-Bera test
#[derive(Debug, Clone)]
pub struct JarqueBeraResult {
    pub test_statistic: f64,
    pub p_value: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

/// Result of KPSS test
#[derive(Debug, Clone)]
pub struct KPSSResult {
    pub test_statistic: f64,
    pub p_value: f64,
    pub n_lags: usize,
    pub critical_values: Vec<(String, f64)>,
}

/// Result of Phillips-Perron test
#[derive(Debug, Clone)]
pub struct PPResult {
    pub test_statistic: f64,
    pub p_value: f64,
    pub n_lags: usize,
    pub critical_values: Vec<(String, f64)>,
}

/// Approximate chi-squared survival function P(X > x) with `df` degrees of freedom.
///
/// Uses the regularised incomplete gamma function approximated via a continued-fraction
/// expansion (Lentz's algorithm).  Accurate to ~4 decimal places for typical df/x ranges.
fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    // P(X > x) = 1 - gamma_lower(df/2, x/2) / Gamma(df/2)
    //          = Gamma_upper(df/2, x/2) / Gamma(df/2)
    //          = regularised_upper_incomplete_gamma(df/2, x/2)
    regularised_gamma_upper(df / 2.0, x / 2.0)
}

/// Regularised upper incomplete gamma: Q(a, x) = Gamma(a, x) / Gamma(a).
///
/// Uses the series expansion for small x (x < a + 1) and continued-fraction for
/// large x.
fn regularised_gamma_upper(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 1.0;
    }
    if x == 0.0 {
        return 1.0;
    }
    // For large x relative to a, use continued-fraction representation
    if x > a + 1.0 {
        gamma_cf(a, x)
    } else {
        1.0 - gamma_series(a, x)
    }
}

/// Series expansion for regularised lower incomplete gamma P(a, x).
fn gamma_series(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 3e-10;
    let ln_gamma_a = ln_gamma(a);
    let mut ap = a;
    let mut del = 1.0 / a;
    let mut sum = del;
    for _ in 0..max_iter {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * eps {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma_a).exp()
}

/// Continued-fraction expansion for regularised upper incomplete gamma Q(a, x).
fn gamma_cf(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 3e-10;
    let fpmin = f64::MIN_POSITIVE / eps;
    let ln_gamma_a = ln_gamma(a);
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / fpmin;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=max_iter {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < fpmin {
            d = fpmin;
        }
        c = b + an / c;
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
    h * (-x + a * x.ln() - ln_gamma_a).exp()
}

/// Natural log of Gamma(x) via Lanczos approximation (g=7, n=9).
///
/// Valid for all x > 0.  For 0 < x < 1 the recurrence relation
/// `ln Γ(x) = ln Γ(x+1) − ln(x)` is applied repeatedly until the
/// argument reaches ≥ 1, where the Lanczos kernel is stable.
fn ln_gamma(x: f64) -> f64 {
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
    // Shift x up until it is ≥ 1 using the recurrence Γ(x+1) = x·Γ(x)
    // i.e. ln Γ(x) = ln Γ(x+k) − sum_{j=0}^{k-1} ln(x+j)
    let mut shift = 0.0;
    let mut xr = x;
    while xr < 1.0 {
        shift += xr.ln();
        xr += 1.0;
    }
    // Now xr ≥ 1 — apply Lanczos on z = xr − 1
    let z = xr - 1.0;
    let mut ser = C[0];
    for (k, &ck) in C[1..].iter().enumerate() {
        ser += ck / (z + (k + 1) as f64);
    }
    let t = z + G + 0.5;
    let lanczos = (2.0 * std::f64::consts::PI).sqrt().ln() + (z + 0.5) * t.ln() - t + ser.ln();
    // Undo the shift: ln Γ(x) = ln Γ(xr) − shift
    lanczos - shift
}



/// Augmented Dickey-Fuller test for stationarity
///
/// Tests the null hypothesis that a unit root is present in the time series.
/// If rejected (p-value < significance level), the series is likely stationary.
///
/// # Arguments
/// * `series` - The time series to test
/// * `regression` - Type of regression ("c": constant, "ct": constant and trend, "nc": no constant)
/// * `max_lags` - Maximum number of lags to include (None for automatic selection)
///
/// # Returns
/// Test statistic, p-value, number of lags used, and critical values
pub fn augmented_dickey_fuller_test(
    series: &TimeSeries,
    regression: &str,
    max_lags: Option<usize>,
) -> Result<ADFResult> {
    // Convert TimeSeries to vec
    let data: Vec<f64> = series.values.to_vec().map_err(|e| {
        TorshError::InvalidArgument(format!("Failed to convert tensor to vec: {}", e))
    })?.iter().map(|&v| v as f64).collect();

    let n = data.len();
    if n < 4 {
        return Err(TorshError::InvalidArgument(
            "ADF test requires at least 4 observations".to_string(),
        ));
    }

    // Automatic lag selection by Schwert rule, capped to ensure enough observations
    // for the OLS regression (need at least k+3 observations where k = n_regressors)
    let max_feasible_lags = if n > 6 { n / 3 } else { 0 };
    let lags = max_lags.unwrap_or_else(|| {
        ((12.0 * (n as f64 / 100.0).powf(0.25)) as usize).min(max_feasible_lags)
    });

    // Build regression: Δy_t = α + β*y_{t-1} + Σ γ_k*Δy_{t-k} + ε_t
    // Design matrix columns: intercept (maybe trend), y_{t-1}, Δy_{t-1}, ..., Δy_{t-lags}
    let use_trend = regression == "ct";
    let use_const = regression != "nc";
    let t_start = lags + 1;
    let t_end = n;
    let n_obs = t_end - t_start;

    if n_obs < 3 {
        return Err(TorshError::InvalidArgument(
            "Not enough observations after lag selection".to_string(),
        ));
    }

    // Build differences
    let mut dy = vec![0.0f64; n];
    for i in 1..n {
        dy[i] = data[i] - data[i - 1];
    }

    // Number of regressors: [const?] + [trend?] + y_{t-1} + lags differences
    let n_regressors = (if use_const { 1 } else { 0 })
        + (if use_trend { 1 } else { 0 })
        + 1
        + lags;

    // Build X matrix and y vector
    let mut x_mat = vec![0.0f64; n_obs * n_regressors];
    let mut y_vec = vec![0.0f64; n_obs];

    for (row, t) in (t_start..t_end).enumerate() {
        y_vec[row] = dy[t];
        let mut col = 0;
        if use_const {
            x_mat[row * n_regressors + col] = 1.0;
            col += 1;
        }
        if use_trend {
            x_mat[row * n_regressors + col] = t as f64;
            col += 1;
        }
        // y_{t-1}
        x_mat[row * n_regressors + col] = data[t - 1];
        col += 1;
        // Δy_{t-k} for k=1..lags
        for k in 1..=lags {
            x_mat[row * n_regressors + col] = dy[t - k];
            col += 1;
        }
    }

    // OLS via normal equations using Cholesky on (X'X)
    let k = n_regressors;
    // Compute X'X
    let mut xtx = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0;
            for row in 0..n_obs {
                s += x_mat[row * k + i] * x_mat[row * k + j];
            }
            xtx[i * k + j] = s;
        }
    }
    // Compute X'y
    let mut xty = vec![0.0f64; k];
    for i in 0..k {
        let mut s = 0.0;
        for row in 0..n_obs {
            s += x_mat[row * k + i] * y_vec[row];
        }
        xty[i] = s;
    }

    // Solve (X'X) beta = X'y via Cholesky
    let beta = cholesky_solve_f64(&xtx, &xty, k).unwrap_or_else(|| {
        // Fallback: use simple diagonal regression (ridge with large lambda)
        let lambda = 1e-4;
        let mut b = vec![0.0f64; k];
        for i in 0..k {
            let denom = xtx[i * k + i] + lambda;
            if denom.abs() > 1e-15 {
                b[i] = xty[i] / denom;
            }
        }
        b
    });

    // Residuals and variance
    let mut rss = 0.0f64;
    for row in 0..n_obs {
        let mut yhat = 0.0;
        for i in 0..k {
            yhat += x_mat[row * k + i] * beta[i];
        }
        let e = y_vec[row] - yhat;
        rss += e * e;
    }
    let df_resid = n_obs - k;
    let s2 = rss / df_resid.max(1) as f64;

    // Index of y_{t-1} coefficient in beta
    let beta_idx = if use_const { 1 } else { 0 } + if use_trend { 1 } else { 0 };

    // Variance of beta[beta_idx] = s2 * [(X'X)^{-1}]_{beta_idx, beta_idx}
    let xtx_inv = cholesky_invert_f64(&xtx, k)
        .unwrap_or_else(|| vec![1.0f64 / s2; k * k]);
    let se_beta = (s2 * xtx_inv[beta_idx * k + beta_idx]).sqrt();
    let test_stat = if se_beta > 1e-15 { beta[beta_idx] / se_beta } else { 0.0 };

    // Approximate p-value via MacKinnon (1994) response surface for the
    // no-constant case; use critical value table otherwise.
    // We use an approximation: map test stat to a normal tail (conservative).
    // More accurate: use MacKinnon's response-surface coefficients.
    let p_value = adf_p_value_approx(test_stat, n_obs, regression);

    Ok(ADFResult {
        test_statistic: test_stat,
        p_value,
        n_lags: lags,
        n_obs: n_obs - lags,
        critical_values: vec![
            ("1%".to_string(), -3.43),
            ("5%".to_string(), -2.86),
            ("10%".to_string(), -2.57),
        ],
    })
}

/// Approximate ADF p-value from test statistic using MacKinnon (1994) tables.
fn adf_p_value_approx(stat: f64, n: usize, regression: &str) -> f64 {
    // MacKinnon (1994) response-surface coefficients for p-value computation
    // Polynomial fit: τ_p = a0 + a1/T + a2/T^2  (T = sample size)
    // For regression = "c" (constant only)
    let t = n as f64;
    // Approximate quantiles at 0.01, 0.05, 0.10, 0.90 for "c" regression
    let crit = if regression == "ct" {
        // constant + trend
        [(-4.15 - 49.37/t), (-3.45 - 26.28/t), (-3.13 - 17.83/t)]
    } else if regression == "nc" {
        [(-2.60 - 13.50/t), (-1.95 - 7.24/t), (-1.61 - 4.56/t)]
    } else {
        // "c" default
        [(-3.43 - 6.50/t), (-2.86 - 2.86/t), (-2.57 - 1.77/t)]
    };
    // Linear interpolation between known p-value levels
    if stat <= crit[0] {
        // More extreme than 1%
        let slope = (crit[1] - crit[0]) / (0.05_f64.ln() - 0.01_f64.ln());
        let lp = 0.01_f64.ln() + (stat - crit[0]) / slope;
        lp.exp().max(1e-6).min(1.0)
    } else if stat <= crit[1] {
        let slope = (crit[2] - crit[1]) / (0.10_f64.ln() - 0.05_f64.ln());
        let lp = 0.05_f64.ln() + (stat - crit[1]) / slope;
        lp.exp().max(0.001).min(0.999)
    } else if stat <= crit[2] {
        let slope = (crit[2] - crit[1]) / (0.10_f64.ln() - 0.05_f64.ln());
        let lp = 0.10_f64.ln() + (stat - crit[2]) / slope;
        lp.exp().max(0.001).min(0.999)
    } else {
        // Not significant; use normal approximation for the tail
        let z = (stat - (-2.57)).max(0.0) * 0.5;
        0.10 + 0.90 * (1.0 - (-z * z / 2.0).exp())
    }
}

/// Cholesky-based linear system solve for symmetric positive-definite matrix.
/// Returns `None` if the matrix is singular/not PD.
fn cholesky_solve_f64(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let l = cholesky_lower_f64(a, n)?;
    // Forward substitution L y = b
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * y[j];
        }
        let diag = l[i * n + i];
        if diag.abs() < 1e-15 {
            return None;
        }
        y[i] = s / diag;
    }
    // Back substitution L^T x = y
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= l[j * n + i] * x[j];
        }
        let diag = l[i * n + i];
        if diag.abs() < 1e-15 {
            return None;
        }
        x[i] = s / diag;
    }
    Some(x)
}

/// Invert symmetric PD matrix using Cholesky.
fn cholesky_invert_f64(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let l = cholesky_lower_f64(a, n)?;
    let mut inv = vec![0.0f64; n * n];
    for col in 0..n {
        let mut e = vec![0.0f64; n];
        e[col] = 1.0;
        let x = {
            let mut y = vec![0.0f64; n];
            for i in 0..n {
                let mut s = e[i];
                for j in 0..i {
                    s -= l[i * n + j] * y[j];
                }
                y[i] = s / l[i * n + i];
            }
            let mut x = vec![0.0f64; n];
            for i in (0..n).rev() {
                let mut s = y[i];
                for j in (i + 1)..n {
                    s -= l[j * n + i] * x[j];
                }
                x[i] = s / l[i * n + i];
            }
            x
        };
        for row in 0..n {
            inv[row * n + col] = x[row];
        }
    }
    Some(inv)
}

/// Cholesky-Banachiewicz decomposition (f64).
fn cholesky_lower_f64(a: &[f64], n: usize) -> Option<Vec<f64>> {
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

/// Ljung-Box test for autocorrelation in residuals
///
/// Tests the null hypothesis that residuals are independently distributed
/// (no autocorrelation). Used to check model adequacy.
///
/// # Arguments
/// * `series` - The time series (typically residuals from a model)
/// * `lags` - Number of lags to test
///
/// # Returns
/// Test statistic, p-value, and degrees of freedom
pub fn ljung_box(series: &TimeSeries, lags: usize) -> Result<LjungBoxResult> {
    // Convert TimeSeries to vec
    let data = series.values.to_vec().map_err(|e| {
        TorshError::InvalidArgument(format!("Failed to convert tensor to vec: {}", e))
    })?;

    let n = data.len();
    if n < 2 {
        return Err(TorshError::InvalidArgument(
            "Ljung-Box test requires at least 2 observations".to_string(),
        ));
    }
    if lags == 0 || lags >= n {
        return Err(TorshError::InvalidArgument(
            "lags must be between 1 and n-1".to_string(),
        ));
    }

    let nf = n as f64;
    let mean = data.iter().map(|&v| v as f64).sum::<f64>() / nf;

    // Variance (denominator for autocorrelations)
    let variance: f64 = data.iter().map(|&v| {
        let d = v as f64 - mean;
        d * d
    }).sum::<f64>();

    let mut q_stat = 0.0;
    for k in 1..=lags {
        let mut cross = 0.0;
        for t in k..n {
            cross += (data[t] as f64 - mean) * (data[t - k] as f64 - mean);
        }
        if variance > 1e-15 {
            let rk = cross / variance;
            q_stat += (rk * rk) / (nf - k as f64);
        }
    }
    q_stat *= nf * (nf + 2.0);

    // p-value from chi-squared distribution with `lags` degrees of freedom
    let p_value = chi2_sf(q_stat, lags as f64);

    Ok(LjungBoxResult {
        test_statistic: q_stat,
        p_value,
        df: lags,
    })
}

/// Jarque-Bera test for normality
///
/// Tests the null hypothesis that the data is normally distributed.
/// Based on skewness and kurtosis of the sample.
///
/// # Arguments
/// * `series` - The time series to test
///
/// # Returns
/// Test statistic, p-value, skewness, and kurtosis
pub fn jarque_bera(series: &TimeSeries) -> Result<JarqueBeraResult> {
    // Convert TimeSeries to vec
    let data = series.values.to_vec().map_err(|e| {
        TorshError::InvalidArgument(format!("Failed to convert tensor to vec: {}", e))
    })?;

    let n = data.len();
    if n < 3 {
        return Err(TorshError::InvalidArgument(
            "Jarque-Bera test requires at least 3 observations".to_string(),
        ));
    }
    let nf = n as f64;
    let mean = data.iter().map(|&v| v as f64).sum::<f64>() / nf;

    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    let mut m4 = 0.0f64;

    for &x in &data {
        let dev = x as f64 - mean;
        m2 += dev * dev;
        m3 += dev * dev * dev;
        m4 += dev * dev * dev * dev;
    }

    m2 /= nf;
    m3 /= nf;
    m4 /= nf;

    let skewness = if m2 > 1e-15 { m3 / m2.powf(1.5) } else { 0.0 };
    let kurtosis = if m2 > 1e-15 { m4 / (m2 * m2) - 3.0 } else { 0.0 };

    let jb_stat = (nf / 6.0) * (skewness * skewness + (kurtosis * kurtosis) / 4.0);

    // JB ~ chi2(2) asymptotically
    let p_value = chi2_sf(jb_stat, 2.0);

    Ok(JarqueBeraResult {
        test_statistic: jb_stat,
        p_value,
        skewness,
        kurtosis,
    })
}

/// KPSS test for stationarity
///
/// Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.
/// Null hypothesis: The series is stationary (opposite of ADF test).
///
/// # Arguments
/// * `series` - The time series to test
/// * `regression` - Type of regression ("c": constant, "ct": constant and trend)
/// * `lags` - Number of lags to use (None for automatic selection)
pub fn kpss_test(
    series: &TimeSeries,
    regression: &str,
    lags: Option<usize>,
) -> Result<KPSSResult> {
    // Convert TimeSeries to vec
    let data: Vec<f64> = series.values.to_vec().map_err(|e| {
        TorshError::InvalidArgument(format!("Failed to convert tensor to vec: {}", e))
    })?.iter().map(|&v| v as f64).collect();

    let n = data.len();
    if n < 3 {
        return Err(TorshError::InvalidArgument(
            "KPSS test requires at least 3 observations".to_string(),
        ));
    }

    let lags_used = lags.unwrap_or_else(|| {
        (4.0 * (n as f64 / 100.0).powf(0.25)) as usize
    });

    // Detrend: subtract mean (or linear trend for "ct")
    let residuals: Vec<f64> = if regression == "ct" {
        // OLS detrending: e_t = y_t - (a + b*t)
        let nf = n as f64;
        let t_bar = (nf - 1.0) / 2.0;
        let stt: f64 = (0..n).map(|i| (i as f64 - t_bar).powi(2)).sum();
        let sy: f64 = data.iter().sum();
        let sty: f64 = data.iter().enumerate().map(|(i, &v)| (i as f64 - t_bar) * v).sum();
        let b = sty / stt;
        let a = sy / nf - b * t_bar;
        data.iter().enumerate().map(|(i, &v)| v - a - b * i as f64).collect()
    } else {
        // Demean
        let mean = data.iter().sum::<f64>() / n as f64;
        data.iter().map(|&v| v - mean).collect()
    };

    // Partial sums S_t = Σ_{i=1}^{t} e_i
    let mut partial_sums = Vec::with_capacity(n);
    let mut running = 0.0f64;
    for &e in &residuals {
        running += e;
        partial_sums.push(running);
    }

    // Long-run variance estimate (Newey-West)
    let sigma2: f64 = {
        // Variance
        let var = residuals.iter().map(|&e| e * e).sum::<f64>() / n as f64;
        let mut cov_sum = 0.0f64;
        for k in 1..=lags_used {
            let bart_weight = 1.0 - k as f64 / (lags_used as f64 + 1.0);
            let cov_k: f64 = (k..n).map(|t| residuals[t] * residuals[t - k]).sum::<f64>() / n as f64;
            cov_sum += 2.0 * bart_weight * cov_k;
        }
        var + cov_sum
    };

    // KPSS statistic
    let nf = n as f64;
    let kpss_stat = if sigma2 > 1e-15 {
        partial_sums.iter().map(|&s| s * s).sum::<f64>() / (nf * nf * sigma2)
    } else {
        0.0
    };

    // Approximate p-value by interpolating critical value tables
    // KPSS critical values (from KPSS 1992, regression="c")
    let (cvs_c, cvs_ct) = (
        [(0.10, 0.347), (0.05, 0.463), (0.025, 0.574), (0.01, 0.739)],
        [(0.10, 0.119), (0.05, 0.146), (0.025, 0.176), (0.01, 0.216)],
    );
    let cvs = if regression == "ct" { cvs_ct } else { cvs_c };

    let p_value = kpss_p_value_interpolate(kpss_stat, &cvs);

    Ok(KPSSResult {
        test_statistic: kpss_stat,
        p_value,
        n_lags: lags_used,
        critical_values: vec![
            ("1%".to_string(), cvs[3].1),
            ("5%".to_string(), cvs[1].1),
            ("10%".to_string(), cvs[0].1),
        ],
    })
}

/// Interpolate KPSS p-value from critical values table.
fn kpss_p_value_interpolate(stat: f64, cvs: &[(f64, f64)]) -> f64 {
    // cvs sorted from largest p to smallest p (10%, 5%, 2.5%, 1%)
    if stat < cvs[0].1 {
        return 0.90; // p > 0.10
    }
    if stat >= cvs[cvs.len() - 1].1 {
        return 0.01; // p ≤ 0.01
    }
    for i in 0..cvs.len() - 1 {
        if stat >= cvs[i].1 && stat < cvs[i + 1].1 {
            let t = (stat - cvs[i].1) / (cvs[i + 1].1 - cvs[i].1);
            return cvs[i].0 + t * (cvs[i + 1].0 - cvs[i].0);
        }
    }
    0.05
}

/// Phillips-Perron test for unit root
///
/// An alternative to the ADF test that is robust to heteroskedasticity
/// and autocorrelation in the error term.
pub fn phillips_perron_test(series: &TimeSeries, regression: &str) -> Result<PPResult> {
    // Convert TimeSeries to vec
    let data: Vec<f64> = series.values.to_vec().map_err(|e| {
        TorshError::InvalidArgument(format!("Failed to convert tensor to vec: {}", e))
    })?.iter().map(|&v| v as f64).collect();

    let n = data.len();
    if n < 4 {
        return Err(TorshError::InvalidArgument(
            "Phillips-Perron test requires at least 4 observations".to_string(),
        ));
    }

    let lags = (4.0 * (n as f64 / 100.0).powf(0.25)) as usize;

    // Build regression: Δy_t = α + β*y_{t-1} + ε_t
    let use_const = regression != "nc";
    let use_trend = regression == "ct";
    let t_start = 1usize;
    let n_obs = n - t_start;

    let mut dy = vec![0.0f64; n];
    for i in 1..n {
        dy[i] = data[i] - data[i - 1];
    }

    let n_reg = (if use_const { 1 } else { 0 }) + (if use_trend { 1 } else { 0 }) + 1;
    let mut x_mat = vec![0.0f64; n_obs * n_reg];
    let mut y_vec = vec![0.0f64; n_obs];

    for (row, t) in (t_start..n).enumerate() {
        y_vec[row] = dy[t];
        let mut col = 0;
        if use_const {
            x_mat[row * n_reg + col] = 1.0;
            col += 1;
        }
        if use_trend {
            x_mat[row * n_reg + col] = t as f64;
            col += 1;
        }
        x_mat[row * n_reg + col] = data[t - 1];
    }

    let k = n_reg;
    let mut xtx = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0;
            for row in 0..n_obs {
                s += x_mat[row * k + i] * x_mat[row * k + j];
            }
            xtx[i * k + j] = s;
        }
    }
    let mut xty = vec![0.0f64; k];
    for i in 0..k {
        let mut s = 0.0;
        for row in 0..n_obs {
            s += x_mat[row * k + i] * y_vec[row];
        }
        xty[i] = s;
    }

    let beta = cholesky_solve_f64(&xtx, &xty, k).unwrap_or_else(|| {
        let lambda = 1e-4;
        let mut b = vec![0.0f64; k];
        for i in 0..k {
            let denom = xtx[i * k + i] + lambda;
            if denom.abs() > 1e-15 {
                b[i] = xty[i] / denom;
            }
        }
        b
    });

    // Residuals
    let mut residuals = vec![0.0f64; n_obs];
    for row in 0..n_obs {
        let mut yhat = 0.0;
        for i in 0..k {
            yhat += x_mat[row * k + i] * beta[i];
        }
        residuals[row] = y_vec[row] - yhat;
    }

    let s2 = residuals.iter().map(|&e| e * e).sum::<f64>() / (n_obs - k).max(1) as f64;

    // Long-run variance via Newey-West
    let s2_lrv: f64 = {
        let var = residuals.iter().map(|&e| e * e).sum::<f64>() / n_obs as f64;
        let mut cov_sum = 0.0f64;
        for lag in 1..=lags {
            let w = 1.0 - lag as f64 / (lags as f64 + 1.0);
            let cov_k: f64 = (lag..n_obs).map(|t| residuals[t] * residuals[t - lag]).sum::<f64>() / n_obs as f64;
            cov_sum += 2.0 * w * cov_k;
        }
        var + cov_sum
    };

    let beta_idx = (if use_const { 1 } else { 0 }) + (if use_trend { 1 } else { 0 });
    let xtx_inv = cholesky_invert_f64(&xtx, k)
        .unwrap_or_else(|| { let mut v = vec![0.0f64; k*k]; for i in 0..k { v[i*k+i] = 1.0; } v });
    let se_ols = (s2 * xtx_inv[beta_idx * k + beta_idx]).sqrt();

    // PP correction: Z_tau = (s2 / s2_lrv)^0.5 * t_stat - 0.5 * (s2_lrv - s2) / ...
    // Simplified PP Z_alpha correction
    let t_stat_ols = if se_ols > 1e-15 { beta[beta_idx] / se_ols } else { 0.0 };
    // PP test stat: Z_t = sqrt(s2/s2_lrv) * t_stat_ols (simplified)
    let pp_stat = if s2_lrv > 1e-15 {
        (s2 / s2_lrv).sqrt() * t_stat_ols
    } else {
        t_stat_ols
    };

    let p_value = adf_p_value_approx(pp_stat, n_obs, regression);

    Ok(PPResult {
        test_statistic: pp_stat,
        p_value,
        n_lags: lags,
        critical_values: vec![
            ("1%".to_string(), -3.43),
            ("5%".to_string(), -2.86),
            ("10%".to_string(), -2.57),
        ],
    })
}

/// Run a comprehensive stationarity test suite
///
/// Runs multiple stationarity tests (ADF, KPSS, PP) and returns a summary.
pub struct StationarityTestSuite {
    pub adf_result: ADFResult,
    pub kpss_result: KPSSResult,
    pub pp_result: PPResult,
}

impl StationarityTestSuite {
    /// Run all stationarity tests
    pub fn run(series: &TimeSeries) -> Result<Self> {
        let adf_result = augmented_dickey_fuller_test(series, "c", None)?;
        let kpss_result = kpss_test(series, "c", None)?;
        let pp_result = phillips_perron_test(series, "c")?;

        Ok(Self {
            adf_result,
            kpss_result,
            pp_result,
        })
    }

    /// Check if series is likely stationary based on all tests
    pub fn is_stationary(&self, significance_level: f64) -> bool {
        // ADF: null hypothesis is non-stationary (reject if p < alpha)
        let adf_stationary = self.adf_result.p_value < significance_level;

        // KPSS: null hypothesis is stationary (accept if p >= alpha)
        let kpss_stationary = self.kpss_result.p_value >= significance_level;

        // PP: null hypothesis is non-stationary (reject if p < alpha)
        let pp_stationary = self.pp_result.p_value < significance_level;

        // Consider stationary if majority of tests agree
        let stationary_count = [adf_stationary, kpss_stationary, pp_stationary]
            .iter()
            .filter(|&&x| x)
            .count();

        stationary_count >= 2
    }
}

/// Run diagnostics on model residuals
///
/// Comprehensive residual diagnostics including autocorrelation,
/// normality, and heteroskedasticity tests.
pub struct ResidualDiagnostics {
    pub ljung_box: LjungBoxResult,
    pub jarque_bera: JarqueBeraResult,
    pub durbin_watson: f64,
}

impl ResidualDiagnostics {
    /// Perform residual diagnostics
    pub fn run(residuals: &TimeSeries, lags: usize) -> Result<Self> {
        let ljung_box = ljung_box(residuals, lags)?;
        let jarque_bera = jarque_bera(residuals)?;
        let durbin_watson = calculate_durbin_watson(residuals)?;

        Ok(Self {
            ljung_box,
            jarque_bera,
            durbin_watson,
        })
    }

    /// Check if residuals pass all diagnostic tests
    pub fn is_well_specified(&self, significance_level: f64) -> bool {
        // Ljung-Box: null is no autocorrelation (accept if p >= alpha)
        let no_autocorr = self.ljung_box.p_value >= significance_level;

        // Jarque-Bera: null is normality (accept if p >= alpha)
        let is_normal = self.jarque_bera.p_value >= significance_level;

        // Durbin-Watson: 1.5 to 2.5 is acceptable range
        let dw_ok = self.durbin_watson >= 1.5 && self.durbin_watson <= 2.5;

        no_autocorr && is_normal && dw_ok
    }
}

/// Calculate Durbin-Watson statistic for autocorrelation
///
/// DW statistic ranges from 0 to 4:
/// - ~2: No autocorrelation
/// - <2: Positive autocorrelation
/// - >2: Negative autocorrelation
fn calculate_durbin_watson(series: &TimeSeries) -> Result<f64> {
    let n = series.len();
    if n < 2 {
        return Err(TorshError::InvalidArgument(
            "Series too short for Durbin-Watson test".to_string(),
        ));
    }

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    // Calculate sum of squared differences
    for i in 1..n {
        let curr = series.values.get_item_flat(i)? as f64;
        let prev = series.values.get_item_flat(i - 1)? as f64;
        let diff = curr - prev;
        numerator += diff * diff;
    }

    // Calculate sum of squared residuals
    for i in 0..n {
        let val = series.values.get_item_flat(i)? as f64;
        denominator += val * val;
    }

    if denominator.abs() < 1e-10 {
        return Err(TorshError::InvalidArgument(
            "Zero variance in residuals".to_string(),
        ));
    }

    Ok(numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::Tensor;

    fn create_test_series() -> TimeSeries {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let tensor = Tensor::from_vec(data, &[10]).expect("Tensor should succeed");
        TimeSeries::new(tensor)
    }

    fn create_stationary_series() -> TimeSeries {
        // Create deterministic "random-like" stationary series for testing
        // Using a simple combination of sine waves to simulate white noise
        let data: Vec<f32> = (0..100)
            .map(|i| {
                let t = i as f32 * 0.1;
                // Combine multiple frequencies to create pseudo-random stationary data
                (t * 3.7).sin() * 0.5 + (t * 7.3).cos() * 0.3 + (t * 13.1).sin() * 0.2
            })
            .collect();
        let tensor = Tensor::from_vec(data, &[100]).expect("Tensor should succeed");
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_augmented_dickey_fuller() {
        let series = create_test_series();
        let result = augmented_dickey_fuller_test(&series, "c", None)
            .expect("augmented dickey fuller test should succeed");

        assert!(result.test_statistic.is_finite());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.n_lags > 0);
    }

    #[test]
    fn test_ljung_box() {
        let series = create_test_series();
        let result = ljung_box(&series, 3).expect("ljung box should succeed");

        assert!(result.test_statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.df, 3);
    }

    #[test]
    fn test_jarque_bera() {
        let series = create_stationary_series();
        let result = jarque_bera(&series).expect("jarque bera should succeed");

        assert!(result.test_statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.skewness.is_finite());
        assert!(result.kurtosis.is_finite());
    }

    #[test]
    fn test_kpss() {
        let series = create_test_series();
        let result = kpss_test(&series, "c", None).expect("kpss test should succeed");

        assert!(result.test_statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_phillips_perron() {
        let series = create_test_series();
        let result =
            phillips_perron_test(&series, "c").expect("phillips perron test should succeed");

        assert!(result.test_statistic.is_finite());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_stationarity_suite() {
        let series = create_stationary_series();
        let suite =
            StationarityTestSuite::run(&series).expect("Stationarity Test Suite should succeed");

        assert!(suite.adf_result.test_statistic.is_finite());
        assert!(suite.kpss_result.test_statistic >= 0.0);
        assert!(suite.pp_result.test_statistic.is_finite());
    }

    #[test]
    fn test_residual_diagnostics() {
        let series = create_stationary_series();
        let diagnostics =
            ResidualDiagnostics::run(&series, 5).expect("Residual Diagnostics should succeed");

        assert!(diagnostics.ljung_box.test_statistic >= 0.0);
        assert!(diagnostics.jarque_bera.test_statistic >= 0.0);
        assert!(diagnostics.durbin_watson >= 0.0 && diagnostics.durbin_watson <= 4.0);
    }

    #[test]
    fn test_durbin_watson() {
        let series = create_test_series();
        let dw = calculate_durbin_watson(&series).expect("calculate durbin watson should succeed");

        assert!(dw >= 0.0 && dw <= 4.0);
    }

    /// Verify chi2_sf against well-known chi-squared quantiles.
    ///
    /// These reference values come from standard statistical tables and are
    /// independent of any particular software implementation.
    #[test]
    fn test_chi2_sf_known_quantiles() {
        // P(chi²(1) > 3.841) ≈ 0.05  (95th percentile of chi²(1))
        let p1 = chi2_sf(3.841, 1.0);
        assert!(
            (p1 - 0.05).abs() < 0.005,
            "chi2_sf(3.841, 1) = {p1}, expected ≈ 0.05"
        );

        // P(chi²(2) > 5.991) ≈ 0.05  (95th percentile of chi²(2))
        let p2 = chi2_sf(5.991, 2.0);
        assert!(
            (p2 - 0.05).abs() < 0.005,
            "chi2_sf(5.991, 2) = {p2}, expected ≈ 0.05"
        );

        // P(chi²(1) > 6.635) ≈ 0.01  (99th percentile of chi²(1))
        let p3 = chi2_sf(6.635, 1.0);
        assert!(
            (p3 - 0.01).abs() < 0.002,
            "chi2_sf(6.635, 1) = {p3}, expected ≈ 0.01"
        );

        // P(chi²(4) > 9.488) ≈ 0.05  (95th percentile of chi²(4))
        let p4 = chi2_sf(9.488, 4.0);
        assert!(
            (p4 - 0.05).abs() < 0.005,
            "chi2_sf(9.488, 4) = {p4}, expected ≈ 0.05"
        );

        // Near-zero x must return nearly 1.0
        let p_near_zero = chi2_sf(1e-10, 2.0);
        assert!(p_near_zero > 0.999, "chi2_sf near 0 should be ≈ 1.0, got {p_near_zero}");

        // Large x must return nearly 0.0
        let p_large = chi2_sf(100.0, 2.0);
        assert!(p_large < 1e-20, "chi2_sf(100, 2) should be ≈ 0.0, got {p_large}");
    }
}
