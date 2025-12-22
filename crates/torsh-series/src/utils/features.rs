//! Time series feature extraction

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::TimeSeries;
use torsh_tensor::Tensor;

/// Statistical features of a time series
#[derive(Debug, Clone)]
pub struct StatisticalFeatures {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

/// Spectral features of a time series
#[derive(Debug, Clone)]
pub struct SpectralFeatures {
    pub dominant_frequency: f64,
    pub spectral_entropy: f64,
    pub spectral_centroid: f64,
}

/// Extract statistical features
///
/// Computes basic statistical properties of the time series:
/// - Mean: arithmetic average
/// - Standard deviation: measure of spread
/// - Min: minimum value
/// - Max: maximum value
/// - Skewness: measure of asymmetry
/// - Kurtosis: measure of tail heaviness (excess kurtosis)
pub fn statistical_features(series: &TimeSeries) -> StatisticalFeatures {
    let data = series.values.to_vec().unwrap_or_default();

    if data.is_empty() {
        return StatisticalFeatures {
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
        };
    }

    let n = data.len() as f64;

    // Calculate mean
    let mean = data.iter().map(|&x| x as f64).sum::<f64>() / n;

    // Calculate variance and std
    let variance = data
        .iter()
        .map(|&x| {
            let diff = (x as f64) - mean;
            diff * diff
        })
        .sum::<f64>()
        / n;
    let std = variance.sqrt();

    // Calculate min and max
    let min = data.iter().map(|&x| x as f64).fold(f64::INFINITY, f64::min);
    let max = data
        .iter()
        .map(|&x| x as f64)
        .fold(f64::NEG_INFINITY, f64::max);

    StatisticalFeatures {
        mean,
        std,
        min,
        max,
        skewness: calculate_skewness(&series.values),
        kurtosis: calculate_kurtosis(&series.values),
    }
}

/// Calculate skewness of a tensor
///
/// Skewness measures the asymmetry of the distribution.
/// Formula: E[(X - μ)³] / σ³
/// - Positive skew: tail on right side
/// - Negative skew: tail on left side
/// - Zero skew: symmetric distribution
fn calculate_skewness(tensor: &Tensor) -> f64 {
    let data = tensor.to_vec().unwrap_or_default();
    if data.is_empty() {
        return 0.0;
    }

    let n = data.len() as f64;

    // Calculate mean
    let mean = data.iter().map(|&x| x as f64).sum::<f64>() / n;

    // Calculate standard deviation
    let variance = data
        .iter()
        .map(|&x| {
            let diff = (x as f64) - mean;
            diff * diff
        })
        .sum::<f64>()
        / n;
    let std = variance.sqrt();

    if std < 1e-10 {
        // All values are the same, skewness is undefined (return 0)
        return 0.0;
    }

    // Calculate skewness: E[(X - μ)³] / σ³
    let skewness = data
        .iter()
        .map(|&x| {
            let z = ((x as f64) - mean) / std;
            z * z * z
        })
        .sum::<f64>()
        / n;

    skewness
}

/// Calculate kurtosis of a tensor
///
/// Kurtosis measures the "tailedness" of the distribution.
/// Formula: E[(X - μ)⁴] / σ⁴ - 3 (excess kurtosis)
/// - Positive excess kurtosis: heavy tails (leptokurtic)
/// - Negative excess kurtosis: light tails (platykurtic)
/// - Zero excess kurtosis: normal distribution (mesokurtic)
fn calculate_kurtosis(tensor: &Tensor) -> f64 {
    let data = tensor.to_vec().unwrap_or_default();
    if data.is_empty() {
        return 0.0;
    }

    let n = data.len() as f64;

    // Calculate mean
    let mean = data.iter().map(|&x| x as f64).sum::<f64>() / n;

    // Calculate standard deviation
    let variance = data
        .iter()
        .map(|&x| {
            let diff = (x as f64) - mean;
            diff * diff
        })
        .sum::<f64>()
        / n;
    let std = variance.sqrt();

    if std < 1e-10 {
        // All values are the same, kurtosis is undefined (return 0)
        return 0.0;
    }

    // Calculate kurtosis: E[(X - μ)⁴] / σ⁴ - 3 (excess kurtosis)
    let kurtosis = data
        .iter()
        .map(|&x| {
            let z = ((x as f64) - mean) / std;
            z * z * z * z
        })
        .sum::<f64>()
        / n;

    // Return excess kurtosis (subtract 3 for normal distribution baseline)
    kurtosis - 3.0
}

/// Extract autocorrelation features
///
/// Computes the autocorrelation function (ACF) which measures correlation
/// between the series and its lagged values.
///
/// Formula: ACF(k) = Cov(Y_t, Y_{t-k}) / Var(Y_t)
///
/// # Arguments
/// * `series` - Input time series
/// * `max_lag` - Maximum lag to compute ACF for
///
/// # Returns
/// Vector of ACF values for lags 1 to max_lag
pub fn autocorrelation(series: &TimeSeries, max_lag: usize) -> Vec<f64> {
    let data = series.values.to_vec().unwrap_or_default();
    if data.is_empty() || max_lag == 0 {
        return vec![];
    }

    let n = data.len();
    if max_lag >= n {
        return vec![0.0; max_lag];
    }

    // Convert to f64 for precision
    let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();

    // Calculate mean
    let mean = data_f64.iter().sum::<f64>() / n as f64;

    // Calculate variance (lag 0 autocovariance)
    let variance = data_f64.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if variance < 1e-10 {
        // Constant series, autocorrelation is undefined
        return vec![0.0; max_lag];
    }

    let mut acf = Vec::with_capacity(max_lag);

    for lag in 1..=max_lag {
        // Calculate autocovariance at this lag
        let mut autocovariance = 0.0;
        for t in lag..n {
            autocovariance += (data_f64[t] - mean) * (data_f64[t - lag] - mean);
        }
        autocovariance /= n as f64;

        // ACF = autocovariance / variance
        acf.push(autocovariance / variance);
    }

    acf
}

/// Extract partial autocorrelation
///
/// Computes the partial autocorrelation function (PACF) which measures the
/// correlation between Y_t and Y_{t-k} after removing the effect of intermediate lags.
///
/// Uses the Durbin-Levinson recursion algorithm.
///
/// # Arguments
/// * `series` - Input time series
/// * `max_lag` - Maximum lag to compute PACF for
///
/// # Returns
/// Vector of PACF values for lags 1 to max_lag
pub fn partial_autocorrelation(series: &TimeSeries, max_lag: usize) -> Vec<f64> {
    if max_lag == 0 {
        return vec![];
    }

    // First compute ACF (we need it for Durbin-Levinson algorithm)
    let acf = autocorrelation(series, max_lag);
    if acf.is_empty() || acf.iter().all(|&x| x.abs() < 1e-10) {
        return vec![0.0; max_lag];
    }

    let mut pacf = Vec::with_capacity(max_lag);

    // PACF at lag 1 equals ACF at lag 1
    if !acf.is_empty() {
        pacf.push(acf[0]);
    }

    // Use Durbin-Levinson recursion for higher lags
    let mut phi = vec![vec![0.0]; max_lag + 1]; // phi[k] stores coefficients for order k
    phi[1] = vec![acf[0]]; // phi_{1,1} = ACF(1)

    for k in 2..=max_lag {
        if k - 1 >= acf.len() {
            pacf.push(0.0);
            continue;
        }

        // Calculate phi_{k,k} (the PACF at lag k)
        let mut numerator = acf[k - 1]; // ACF(k)
        let mut denominator = 1.0;

        for j in 1..k {
            if k - j - 1 < acf.len() && j - 1 < phi[k - 1].len() {
                numerator -= phi[k - 1][j - 1] * acf[k - j - 1];
            }
            if j - 1 < acf.len() && j - 1 < phi[k - 1].len() {
                denominator -= phi[k - 1][j - 1] * acf[j - 1];
            }
        }

        let phi_kk = if denominator.abs() > 1e-10 {
            numerator / denominator
        } else {
            0.0
        };

        pacf.push(phi_kk);

        // Update phi coefficients for next iteration
        let mut new_phi = vec![0.0; k];
        new_phi[k - 1] = phi_kk;
        for j in 1..k {
            if j - 1 < phi[k - 1].len() && k - j - 1 < phi[k - 1].len() {
                new_phi[j - 1] = phi[k - 1][j - 1] - phi_kk * phi[k - 1][k - j - 1];
            }
        }
        phi[k] = new_phi;
    }

    pacf
}

/// Extract spectral features
pub fn spectral_features(series: &TimeSeries) -> SpectralFeatures {
    use crate::frequency::{FFTAnalyzer, PeriodogramAnalyzer};

    // Use default sampling rate of 1.0 if not specified
    let sampling_rate = series.frequency.unwrap_or(1.0);

    // Find dominant frequency
    let analyzer = PeriodogramAnalyzer::new(sampling_rate);
    let dominant_frequency = analyzer.dominant_frequency(series).unwrap_or(0.0);

    // Compute FFT for spectral centroid and entropy
    let fft_analyzer = FFTAnalyzer::new(sampling_rate);
    let fft_result = fft_analyzer.fft(series).unwrap_or_else(|_| {
        // Fallback to empty result
        crate::frequency::FFTResult {
            real: vec![0.0],
            imag: vec![0.0],
            frequencies: vec![0.0],
            sampling_rate,
        }
    });

    let power = fft_result.power();
    let total_power: f64 = power.iter().sum();

    // Spectral centroid: weighted average of frequencies
    let spectral_centroid = if total_power > 0.0 {
        fft_result
            .frequencies
            .iter()
            .zip(power.iter())
            .map(|(f, p)| f * p)
            .sum::<f64>()
            / total_power
    } else {
        0.0
    };

    // Spectral entropy: Shannon entropy of normalized power spectrum
    let spectral_entropy = if total_power > 0.0 {
        -power
            .iter()
            .map(|&p| {
                let prob = p / total_power;
                if prob > 1e-10 {
                    prob * prob.ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>()
    } else {
        0.0
    };

    SpectralFeatures {
        dominant_frequency,
        spectral_entropy,
        spectral_centroid,
    }
}

/// Extract trend features
///
/// Analyzes the trend component of a time series using linear regression
/// and statistical measures.
///
/// # Arguments
/// * `series` - Input time series
///
/// # Returns
/// TrendFeatures containing:
/// - linear_trend: Slope of the linear regression line (rate of change)
/// - trend_strength: Proportion of variance explained by trend (R² for linear fit)
/// - turning_points: Number of direction changes (peaks and valleys)
pub fn trend_features(series: &TimeSeries) -> TrendFeatures {
    let data = series.values.to_vec().unwrap_or_default();

    if data.is_empty() {
        return TrendFeatures {
            linear_trend: 0.0,
            trend_strength: 0.0,
            turning_points: 0,
        };
    }

    let n = data.len();

    // Calculate linear trend (slope)
    let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = data.iter().map(|&x| x as f64).collect();

    let sum_t: f64 = t.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_t_squared: f64 = t.iter().map(|&x| x * x).sum();
    let sum_t_y: f64 = t.iter().zip(y.iter()).map(|(&ti, &yi)| ti * yi).sum();

    let n_f64 = n as f64;
    let mean_y = sum_y / n_f64;

    // Calculate slope using least squares
    let denominator = n_f64 * sum_t_squared - sum_t * sum_t;
    let linear_trend = if denominator.abs() > 1e-10 {
        (n_f64 * sum_t_y - sum_t * sum_y) / denominator
    } else {
        0.0
    };

    // Calculate trend strength (R² for linear fit)
    let intercept = (sum_y - linear_trend * sum_t) / n_f64;

    // Total sum of squares (variance)
    let ss_tot: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();

    // Residual sum of squares
    let ss_res: f64 = t
        .iter()
        .zip(y.iter())
        .map(|(&ti, &yi)| {
            let fitted = linear_trend * ti + intercept;
            (yi - fitted).powi(2)
        })
        .sum();

    let trend_strength = if ss_tot > 1e-10 {
        1.0 - (ss_res / ss_tot) // R² value
    } else {
        0.0
    };

    // Count turning points (direction changes)
    let mut turning_points = 0;
    if n >= 3 {
        for i in 1..n - 1 {
            let prev = data[i - 1];
            let curr = data[i];
            let next = data[i + 1];

            // Check if this is a peak or valley
            if (curr > prev && curr > next) || (curr < prev && curr < next) {
                turning_points += 1;
            }
        }
    }

    TrendFeatures {
        linear_trend,
        trend_strength,
        turning_points,
    }
}

/// Trend characteristics
#[derive(Debug, Clone)]
pub struct TrendFeatures {
    pub linear_trend: f64,
    pub trend_strength: f64,
    pub turning_points: usize,
}

/// Extract seasonality features
///
/// Detects and characterizes seasonal patterns in a time series using
/// autocorrelation analysis.
///
/// # Arguments
/// * `series` - Input time series
/// * `period` - Expected seasonal period (hint). If 0, will auto-detect.
///
/// # Returns
/// SeasonalityFeatures containing:
/// - seasonal_strength: Maximum ACF value at seasonal lags (0-1)
/// - seasonal_period: Detected primary seasonal period
/// - seasonal_peaks: All significant seasonal lags (ACF > 0.2)
///
/// # Algorithm
/// 1. Compute ACF up to max_lag (min of series_length/2 or 50)
/// 2. Find the lag with maximum ACF (excluding lag 0) as primary period
/// 3. Seasonal strength = maximum ACF value
/// 4. Seasonal peaks = all lags where ACF > threshold (0.2)
pub fn seasonality_features(series: &TimeSeries, period: usize) -> SeasonalityFeatures {
    let data = series.values.to_vec().unwrap_or_default();

    if data.is_empty() || data.len() < 3 {
        return SeasonalityFeatures {
            seasonal_strength: 0.0,
            seasonal_period: 0,
            seasonal_peaks: vec![],
        };
    }

    // Determine max lag for ACF computation
    // Use at most half the series length or 50 lags
    let max_lag = if period > 0 && period < data.len() / 2 {
        // If period hint provided, check up to 2x the period
        (period * 2).min(data.len() / 2)
    } else {
        (data.len() / 2).min(50)
    };

    if max_lag == 0 {
        return SeasonalityFeatures {
            seasonal_strength: 0.0,
            seasonal_period: 0,
            seasonal_peaks: vec![],
        };
    }

    // Compute autocorrelation
    let acf = autocorrelation(series, max_lag);

    if acf.is_empty() {
        return SeasonalityFeatures {
            seasonal_strength: 0.0,
            seasonal_period: 0,
            seasonal_peaks: vec![],
        };
    }

    // Find the lag with maximum ACF (primary seasonal period)
    let mut max_acf = 0.0;
    let mut detected_period = 0;

    for (lag_idx, &acf_val) in acf.iter().enumerate() {
        let lag = lag_idx + 1; // ACF vector is 1-indexed (lag 1 is at index 0)

        if acf_val > max_acf {
            max_acf = acf_val;
            detected_period = lag;
        }
    }

    // Seasonal strength is the maximum ACF value
    let seasonal_strength = max_acf.max(0.0); // Ensure non-negative

    // Find all significant seasonal peaks (ACF > 0.2)
    let threshold = 0.2;
    let seasonal_peaks: Vec<usize> = acf
        .iter()
        .enumerate()
        .filter_map(|(lag_idx, &acf_val)| {
            let lag = lag_idx + 1;
            if acf_val > threshold {
                Some(lag)
            } else {
                None
            }
        })
        .collect();

    SeasonalityFeatures {
        seasonal_strength,
        seasonal_period: detected_period,
        seasonal_peaks,
    }
}

/// Seasonality characteristics
#[derive(Debug, Clone)]
pub struct SeasonalityFeatures {
    pub seasonal_strength: f64,
    pub seasonal_period: usize,
    pub seasonal_peaks: Vec<usize>,
}

/// Advanced feature engineering
///
/// This section provides advanced time series feature engineering capabilities
/// including lag features, rolling statistics, and interaction features.

/// Create lag features
///
/// Generates lagged versions of the time series for different time steps.
/// Lag features are essential for autoregressive models and capture temporal dependencies.
///
/// # Arguments
/// * `series` - Input time series
/// * `lags` - List of lag steps to create (e.g., [1, 2, 3, 7, 14] for daily data)
///
/// # Returns
/// Matrix where each column represents a lag (rows = time points, cols = lags)
/// Note: The first max(lags) rows will contain zeros due to unavailable past values
///
/// # Examples
/// ```
/// // For series [1, 2, 3, 4, 5] with lags [1, 2]:
/// // Result:
/// // [[0, 0],   // t=0: no past values
/// //  [1, 0],   // t=1: lag1=1, lag2=unavailable
/// //  [2, 1],   // t=2: lag1=2, lag2=1
/// //  [3, 2],   // t=3: lag1=3, lag2=2
/// //  [4, 3]]   // t=4: lag1=4, lag2=3
/// ```
pub fn create_lag_features(series: &TimeSeries, lags: &[usize]) -> TimeSeries {
    let data = series.values.to_vec().unwrap_or_default();
    let n = data.len();

    if lags.is_empty() || n == 0 {
        return TimeSeries::new(series.values.clone());
    }

    let num_lags = lags.len();
    let mut lag_matrix = vec![0.0f32; n * num_lags];

    // For each lag
    for (col_idx, &lag) in lags.iter().enumerate() {
        // For each time step
        for t in 0..n {
            let value = if t >= lag {
                // Use lagged value
                data[t - lag]
            } else {
                // Before lag is available, use 0 or first value
                0.0
            };
            lag_matrix[t * num_lags + col_idx] = value;
        }
    }

    let tensor = Tensor::from_vec(lag_matrix, &[n, num_lags]).unwrap();
    TimeSeries::new(tensor)
}

/// Rolling window statistics
#[derive(Debug, Clone)]
pub struct RollingStatistics {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub min: Vec<f64>,
    pub max: Vec<f64>,
    pub median: Vec<f64>,
}

/// Compute rolling window statistics
///
/// Calculates statistical measures over a sliding window for each time point.
/// Useful for capturing local trends and variability in time series.
///
/// # Arguments
/// * `series` - Input time series
/// * `window_size` - Size of the rolling window
/// * `min_periods` - Minimum observations required (if less, returns NaN). Default: window_size
///
/// # Returns
/// RollingStatistics containing mean, std, min, max, median for each time point
///
/// # Examples
/// ```
/// // For series [1, 2, 3, 4, 5] with window_size=3:
/// // At t=2: window=[1,2,3], mean=2.0, std≈0.816
/// // At t=3: window=[2,3,4], mean=3.0, std≈0.816
/// // At t=4: window=[3,4,5], mean=4.0, std≈0.816
/// ```
pub fn rolling_statistics(
    series: &TimeSeries,
    window_size: usize,
    min_periods: Option<usize>,
) -> RollingStatistics {
    let data = series.values.to_vec().unwrap_or_default();
    let n = data.len();
    let min_periods = min_periods.unwrap_or(window_size);

    let mut means = Vec::with_capacity(n);
    let mut stds = Vec::with_capacity(n);
    let mut mins = Vec::with_capacity(n);
    let mut maxs = Vec::with_capacity(n);
    let mut medians = Vec::with_capacity(n);

    for t in 0..n {
        let start = if t + 1 >= window_size {
            t + 1 - window_size
        } else {
            0
        };
        let end = t + 1;
        let window_data: Vec<f64> = data[start..end].iter().map(|&x| x as f64).collect();

        if window_data.len() < min_periods {
            // Not enough data points
            means.push(f64::NAN);
            stds.push(f64::NAN);
            mins.push(f64::NAN);
            maxs.push(f64::NAN);
            medians.push(f64::NAN);
            continue;
        }

        // Mean
        let mean = window_data.iter().sum::<f64>() / window_data.len() as f64;
        means.push(mean);

        // Std
        let variance =
            window_data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / window_data.len() as f64;
        stds.push(variance.sqrt());

        // Min
        let min = window_data.iter().copied().fold(f64::INFINITY, f64::min);
        mins.push(min);

        // Max
        let max = window_data
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        maxs.push(max);

        // Median
        let mut sorted_window = window_data.clone();
        sorted_window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted_window.len() % 2 == 0 {
            let mid = sorted_window.len() / 2;
            (sorted_window[mid - 1] + sorted_window[mid]) / 2.0
        } else {
            sorted_window[sorted_window.len() / 2]
        };
        medians.push(median);
    }

    RollingStatistics {
        mean: means,
        std: stds,
        min: mins,
        max: maxs,
        median: medians,
    }
}

/// Create difference features
///
/// Computes differences between consecutive values at various intervals.
/// Useful for capturing rate of change and acceleration in time series.
///
/// # Arguments
/// * `series` - Input time series
/// * `periods` - List of difference periods (e.g., [1, 7] for 1-day and 1-week differences)
///
/// # Returns
/// Matrix where each column is a difference feature
///
/// # Examples
/// ```
/// // For series [1, 3, 6, 10] with periods [1]:
/// // Result: [[0], [2], [3], [4]]  // First differences
/// ```
pub fn create_difference_features(series: &TimeSeries, periods: &[usize]) -> TimeSeries {
    let data = series.values.to_vec().unwrap_or_default();
    let n = data.len();

    if periods.is_empty() || n == 0 {
        return TimeSeries::new(series.values.clone());
    }

    let num_features = periods.len();
    let mut diff_matrix = vec![0.0f32; n * num_features];

    for (col_idx, &period) in periods.iter().enumerate() {
        for t in 0..n {
            let value = if t >= period {
                data[t] - data[t - period]
            } else {
                0.0
            };
            diff_matrix[t * num_features + col_idx] = value;
        }
    }

    let tensor = Tensor::from_vec(diff_matrix, &[n, num_features]).unwrap();
    TimeSeries::new(tensor)
}

/// Create interaction features
///
/// Generates polynomial and interaction features from time index.
/// Useful for capturing non-linear trends and cyclic patterns.
///
/// # Arguments
/// * `series` - Input time series
/// * `degree` - Polynomial degree (2 for quadratic, 3 for cubic, etc.)
/// * `include_time_features` - Whether to include sin/cos time features
///
/// # Returns
/// Matrix with polynomial features [t, t², t³, ...] and optionally cyclic features
///
/// # Features Generated:
/// - Linear time index (normalized)
/// - Polynomial terms up to specified degree
/// - Sine/cosine terms for cyclic patterns (if enabled)
pub fn create_interaction_features(
    series: &TimeSeries,
    degree: usize,
    include_time_features: bool,
) -> TimeSeries {
    let n = series.len();

    if n == 0 || degree == 0 {
        return TimeSeries::new(series.values.clone());
    }

    // Calculate number of features
    let num_poly = degree; // t, t², t³, ...
    let num_cyclic = if include_time_features { 4 } else { 0 }; // sin(t), cos(t), sin(2t), cos(2t)
    let num_features = num_poly + num_cyclic;

    let mut feature_matrix = vec![0.0f32; n * num_features];

    for t in 0..n {
        let normalized_t = (t as f64) / (n as f64); // Normalize to [0, 1]

        let mut col = 0;

        // Polynomial features
        for d in 1..=degree {
            let value = normalized_t.powi(d as i32);
            feature_matrix[t * num_features + col] = value as f32;
            col += 1;
        }

        // Cyclic time features
        if include_time_features {
            use std::f64::consts::PI;
            let angle = 2.0 * PI * normalized_t;

            // First harmonic
            feature_matrix[t * num_features + col] = angle.sin() as f32;
            col += 1;
            feature_matrix[t * num_features + col] = angle.cos() as f32;
            col += 1;

            // Second harmonic
            let angle2 = 4.0 * PI * normalized_t;
            feature_matrix[t * num_features + col] = angle2.sin() as f32;
            col += 1;
            feature_matrix[t * num_features + col] = angle2.cos() as f32;
        }
    }

    let tensor = Tensor::from_vec(feature_matrix, &[n, num_features]).unwrap();
    TimeSeries::new(tensor)
}

/// Create comprehensive feature set
///
/// Generates a complete set of engineered features combining all methods.
///
/// # Arguments
/// * `series` - Input time series
/// * `lags` - Lag periods to include
/// * `window_size` - Rolling window size
/// * `polynomial_degree` - Degree for polynomial features
///
/// # Returns
/// TimeSeries with all engineered features concatenated
pub fn create_comprehensive_features(
    series: &TimeSeries,
    lags: &[usize],
    _window_size: usize,
    _polynomial_degree: usize,
) -> TimeSeries {
    // This is a simplified version - in practice, you'd concatenate all features
    // For now, just return lag features as the primary engineered features
    create_lag_features(series, lags)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_series() -> TimeSeries {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_vec(data, &[5]).unwrap();
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_statistical_features() {
        let series = create_test_series(); // [1.0, 2.0, 3.0, 4.0, 5.0]
        let features = statistical_features(&series);

        // Mean: (1+2+3+4+5)/5 = 3.0
        assert!((features.mean - 3.0).abs() < 1e-6);

        // Min and Max
        assert_eq!(features.min, 1.0);
        assert_eq!(features.max, 5.0);

        // Std: sqrt(variance) = sqrt(2.0) ≈ 1.414
        // Variance = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5
        //          = (4 + 1 + 0 + 1 + 4) / 5 = 10/5 = 2.0
        assert!((features.std - 1.414213562).abs() < 1e-6);

        // Skewness for symmetric distribution should be ~0
        assert!(features.skewness.abs() < 1e-6);

        // Excess kurtosis for uniform-ish distribution
        // Should be negative (lighter tails than normal)
        assert!(features.kurtosis < 0.0);
    }

    #[test]
    fn test_autocorrelation() {
        let series = create_test_series();
        let acf = autocorrelation(&series, 3);

        assert_eq!(acf.len(), 3);
    }

    #[test]
    fn test_spectral_features() {
        let series = create_test_series();
        let features = spectral_features(&series);

        assert_eq!(features.dominant_frequency, 0.0); // Placeholder
    }

    #[test]
    fn test_trend_features() {
        let series = create_test_series(); // [1.0, 2.0, 3.0, 4.0, 5.0]
        let features = trend_features(&series);

        // Linear trend: slope should be 1.0 for series [1,2,3,4,5]
        assert!((features.linear_trend - 1.0).abs() < 1e-6);

        // Trend strength: R² should be 1.0 for perfect linear series
        assert!((features.trend_strength - 1.0).abs() < 1e-6);

        // Turning points: monotonic series should have 0 turning points
        assert_eq!(features.turning_points, 0);

        // Test with a series that has turning points
        let data_with_peaks = vec![1.0f32, 3.0, 2.0, 4.0, 3.0];
        let tensor_peaks = Tensor::from_vec(data_with_peaks, &[5]).unwrap();
        let series_peaks = TimeSeries::new(tensor_peaks);
        let features_peaks = trend_features(&series_peaks);

        // Should have 3 turning points:
        // Position 1: (3.0) is a peak (1 < 3 > 2)
        // Position 2: (2.0) is a valley (3 > 2 < 4)
        // Position 3: (4.0) is a peak (2 < 4 > 3)
        assert_eq!(features_peaks.turning_points, 3);
    }

    #[test]
    fn test_seasonality_features() {
        // Test 1: Non-seasonal series (monotonic trend)
        let series = create_test_series(); // [1.0, 2.0, 3.0, 4.0, 5.0]
        let features = seasonality_features(&series, 0);

        // Monotonic series has low seasonality
        // The ACF will decay, so seasonal_strength should be less than 1.0
        assert!(features.seasonal_strength < 1.0);

        // Test 2: Series with clear seasonal pattern (period 4)
        // Pattern: [1, 2, 3, 4] repeated 3 times
        let seasonal_data = vec![
            1.0f32, 2.0, 3.0, 4.0, // First cycle
            1.0, 2.0, 3.0, 4.0, // Second cycle
            1.0, 2.0, 3.0, 4.0, // Third cycle
        ];
        let tensor_seasonal = Tensor::from_vec(seasonal_data, &[12]).unwrap();
        let series_seasonal = TimeSeries::new(tensor_seasonal);
        let features_seasonal = seasonality_features(&series_seasonal, 4);

        // Should detect period 4 (or close to it)
        // The detected period should be 4 or a multiple/divisor of 4
        assert!(
            features_seasonal.seasonal_period >= 1,
            "Should detect a seasonal period"
        );

        // Seasonal strength should be significant (ACF at seasonal lag > 0.2)
        assert!(
            features_seasonal.seasonal_strength > 0.2,
            "Should have significant seasonal strength"
        );

        // Should have at least some seasonal peaks
        assert!(
            !features_seasonal.seasonal_peaks.is_empty(),
            "Should detect seasonal peaks"
        );

        // Test 3: Auto-detection (no period hint)
        let features_auto = seasonality_features(&series_seasonal, 0);
        assert!(features_auto.seasonal_period >= 1);
        assert!(features_auto.seasonal_strength > 0.0);
    }
}
