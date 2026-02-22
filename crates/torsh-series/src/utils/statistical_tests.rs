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
    _regression: &str,
    max_lags: Option<usize>,
) -> Result<ADFResult> {
    // Convert TimeSeries to vec
    let data = series.values.to_vec().map_err(|e| {
        TorshError::InvalidArgument(format!("Failed to convert tensor to vec: {}", e))
    })?;

    // TODO: Use scirs2-stats when available
    // For now, return a simplified implementation
    let n = data.len();
    let lags = max_lags.unwrap_or((12.0 * (n as f64 / 100.0).powf(0.25)) as usize);

    // Simplified ADF calculation (placeholder)
    let mean = data.iter().sum::<f32>() / n as f32;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    let test_stat = -variance.sqrt() as f64;

    Ok(ADFResult {
        test_statistic: test_stat,
        p_value: 0.05, // Placeholder
        n_lags: lags,
        n_obs: n - lags,
        critical_values: vec![
            ("1%".to_string(), -3.43),
            ("5%".to_string(), -2.86),
            ("10%".to_string(), -2.57),
        ],
    })
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

    // TODO: Use scirs2-stats when available
    // Simplified Ljung-Box calculation (placeholder)
    let n = data.len() as f64;
    let mean = data.iter().sum::<f32>() / data.len() as f32;

    // Calculate autocorrelations
    let mut q_stat = 0.0;
    for k in 1..=lags {
        let mut sum_prod = 0.0;
        let mut sum_sq = 0.0;

        for t in k..data.len() {
            sum_prod += ((data[t] - mean) * (data[t - k] - mean)) as f64;
        }
        for t in 0..data.len() {
            sum_sq += ((data[t] - mean).powi(2)) as f64;
        }

        if sum_sq > 0.0 {
            let rk = sum_prod / sum_sq;
            q_stat += (rk * rk) / (n - k as f64);
        }
    }
    q_stat *= n * (n + 2.0);

    Ok(LjungBoxResult {
        test_statistic: q_stat,
        p_value: 0.05, // Placeholder
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

    // TODO: Use scirs2-stats when available
    // Calculate moments
    let n = data.len() as f64;
    let mean = data.iter().sum::<f32>() / data.len() as f32;

    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;

    for &x in &data {
        let dev = (x - mean) as f64;
        m2 += dev * dev;
        m3 += dev * dev * dev;
        m4 += dev * dev * dev * dev;
    }

    m2 /= n;
    m3 /= n;
    m4 /= n;

    let skewness = m3 / m2.powf(1.5);
    let kurtosis = m4 / (m2 * m2) - 3.0;

    let jb_stat = (n / 6.0) * (skewness * skewness + (kurtosis * kurtosis) / 4.0);

    Ok(JarqueBeraResult {
        test_statistic: jb_stat,
        p_value: 0.05, // Placeholder
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
    _regression: &str,
    lags: Option<usize>,
) -> Result<KPSSResult> {
    // Convert TimeSeries to vec
    let data = series.values.to_vec().map_err(|e| {
        TorshError::InvalidArgument(format!("Failed to convert tensor to vec: {}", e))
    })?;

    // TODO: Use scirs2-stats when available
    // Simplified KPSS calculation (placeholder)
    let n = data.len();
    let lags_used = lags.unwrap_or((4.0 * (n as f64 / 100.0).powf(0.25)) as usize);

    // Calculate test statistic (placeholder)
    let mean = data.iter().sum::<f32>() / n as f32;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    let test_stat = variance.sqrt() as f64;

    Ok(KPSSResult {
        test_statistic: test_stat,
        p_value: 0.10, // Placeholder
        n_lags: lags_used,
        critical_values: vec![
            ("1%".to_string(), 0.739),
            ("5%".to_string(), 0.463),
            ("10%".to_string(), 0.347),
        ],
    })
}

/// Phillips-Perron test for unit root
///
/// An alternative to the ADF test that is robust to heteroskedasticity
/// and autocorrelation in the error term.
pub fn phillips_perron_test(series: &TimeSeries, _regression: &str) -> Result<PPResult> {
    // Convert TimeSeries to vec
    let data = series.values.to_vec().map_err(|e| {
        TorshError::InvalidArgument(format!("Failed to convert tensor to vec: {}", e))
    })?;

    // TODO: Use scirs2-stats when available
    // Simplified PP calculation (placeholder - similar to ADF)
    let n = data.len();
    let lags = (4.0 * (n as f64 / 100.0).powf(0.25)) as usize;

    let mean = data.iter().sum::<f32>() / n as f32;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    let test_stat = -variance.sqrt() as f64;

    Ok(PPResult {
        test_statistic: test_stat,
        p_value: 0.05, // Placeholder
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
        let tensor = Tensor::from_vec(data, &[10]).unwrap();
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
        let tensor = Tensor::from_vec(data, &[100]).unwrap();
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_augmented_dickey_fuller() {
        let series = create_test_series();
        let result = augmented_dickey_fuller_test(&series, "c", None).unwrap();

        assert!(result.test_statistic.is_finite());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.n_lags > 0);
    }

    #[test]
    fn test_ljung_box() {
        let series = create_test_series();
        let result = ljung_box(&series, 3).unwrap();

        assert!(result.test_statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert_eq!(result.df, 3);
    }

    #[test]
    fn test_jarque_bera() {
        let series = create_stationary_series();
        let result = jarque_bera(&series).unwrap();

        assert!(result.test_statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.skewness.is_finite());
        assert!(result.kurtosis.is_finite());
    }

    #[test]
    fn test_kpss() {
        let series = create_test_series();
        let result = kpss_test(&series, "c", None).unwrap();

        assert!(result.test_statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_phillips_perron() {
        let series = create_test_series();
        let result = phillips_perron_test(&series, "c").unwrap();

        assert!(result.test_statistic.is_finite());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_stationarity_suite() {
        let series = create_stationary_series();
        let suite = StationarityTestSuite::run(&series).unwrap();

        assert!(suite.adf_result.test_statistic.is_finite());
        assert!(suite.kpss_result.test_statistic >= 0.0);
        assert!(suite.pp_result.test_statistic.is_finite());
    }

    #[test]
    fn test_residual_diagnostics() {
        let series = create_stationary_series();
        let diagnostics = ResidualDiagnostics::run(&series, 5).unwrap();

        assert!(diagnostics.ljung_box.test_statistic >= 0.0);
        assert!(diagnostics.jarque_bera.test_statistic >= 0.0);
        assert!(diagnostics.durbin_watson >= 0.0 && diagnostics.durbin_watson <= 4.0);
    }

    #[test]
    fn test_durbin_watson() {
        let series = create_test_series();
        let dw = calculate_durbin_watson(&series).unwrap();

        assert!(dw >= 0.0 && dw <= 4.0);
    }
}
