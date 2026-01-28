//! Time series preprocessing utilities

use crate::TimeSeries;
use torsh_tensor::Tensor;

/// Difference time series
///
/// Computes the nth-order difference of a time series.
/// Differencing is used to make time series stationary by removing trends.
///
/// # Arguments
/// * `series` - Input time series
/// * `order` - Order of differencing (1 = first difference, 2 = second difference, etc.)
///
/// # Returns
/// Differenced time series with length reduced by `order`
///
/// # Formula
/// - Order 1: `diff[t]` = `y[t]` - `y[t-1]`
/// - Order 2: Apply first difference twice
/// - Order n: Apply first difference n times
///
/// # Examples
/// ```text
/// // First difference of [1, 3, 6, 10]:
/// // Result: [2, 3, 4] (differences between consecutive values)
/// ```text
pub fn diff(series: &TimeSeries, order: usize) -> TimeSeries {
    if order == 0 {
        return TimeSeries::new(series.values.clone());
    }

    let mut data = series.values.to_vec().unwrap_or_default();

    // Apply differencing 'order' times
    for _ in 0..order {
        if data.len() <= 1 {
            // Can't difference a series with 1 or fewer elements
            break;
        }

        // Compute first-order difference: `y[t]` - `y[t-1]`
        let mut diff_data = Vec::with_capacity(data.len() - 1);
        for i in 1..data.len() {
            diff_data.push(data[i] - data[i - 1]);
        }

        data = diff_data;
    }

    let n = data.len();
    let tensor = if n > 0 {
        Tensor::from_vec(data, &[n]).expect("tensor creation should succeed")
    } else {
        // Return empty tensor if all data was consumed
        Tensor::from_vec(vec![0.0f32], &[1]).expect("tensor creation should succeed")
    };

    TimeSeries::new(tensor)
}

/// Detrend time series
///
/// Removes trend component from time series using the specified method.
/// Currently supports linear detrending.
///
/// # Arguments
/// * `series` - Input time series
/// * `method` - Detrending method: "linear" (default) or "mean"
///
/// # Returns
/// Detrended time series with trend component removed
///
/// # Methods
/// - "linear": Fits and subtracts a linear trend (y = a*t + b)
/// - "mean": Subtracts the mean (constant detrending)
/// - Any other value defaults to linear
///
/// # Algorithm (Linear)
/// 1. Fit linear regression: y = slope*t + intercept
/// 2. Compute slope using least squares
/// 3. Subtract fitted trend from original series
pub fn detrend(series: &TimeSeries, method: &str) -> TimeSeries {
    let data = series.values.to_vec().unwrap_or_default();

    if data.is_empty() {
        return TimeSeries::new(series.values.clone());
    }

    let n = data.len();

    match method {
        "mean" => {
            // Mean detrending: subtract the mean from all values
            let mean = data.iter().sum::<f32>() / n as f32;
            let detrended: Vec<f32> = data.iter().map(|&x| x - mean).collect();
            let tensor = Tensor::from_vec(detrended, &[n]).expect("tensor creation should succeed");
            TimeSeries::new(tensor)
        }
        _ => {
            // Linear detrending (default)
            // Fit: y = slope*t + intercept

            // Create time indices: t = 0, 1, 2, ..., n-1
            let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let y: Vec<f64> = data.iter().map(|&x| x as f64).collect();

            // Compute sums for least squares
            let sum_t: f64 = t.iter().sum();
            let sum_y: f64 = y.iter().sum();
            let sum_t_squared: f64 = t.iter().map(|&x| x * x).sum();
            let sum_t_y: f64 = t.iter().zip(y.iter()).map(|(&ti, &yi)| ti * yi).sum();

            let n_f64 = n as f64;

            // Calculate slope and intercept using least squares formulas
            // slope = (n*Σ(t*y) - Σt*Σy) / (n*Σ(t²) - (Σt)²)
            let denominator = n_f64 * sum_t_squared - sum_t * sum_t;

            let (slope, intercept) = if denominator.abs() > 1e-10 {
                let slope = (n_f64 * sum_t_y - sum_t * sum_y) / denominator;
                let intercept = (sum_y - slope * sum_t) / n_f64;
                (slope, intercept)
            } else {
                // Constant series or numerical issue, use mean detrending
                (0.0, sum_y / n_f64)
            };

            // Subtract fitted trend from original data
            let detrended: Vec<f32> = t
                .iter()
                .zip(y.iter())
                .map(|(&ti, &yi)| {
                    let fitted = slope * ti + intercept;
                    (yi - fitted) as f32
                })
                .collect();

            let tensor = Tensor::from_vec(detrended, &[n]).expect("tensor creation should succeed");
            TimeSeries::new(tensor)
        }
    }
}

/// Normalize time series
///
/// Performs z-score normalization (standardization) to transform data
/// to have mean ≈ 0 and standard deviation ≈ 1.
///
/// This is an alias for `standard_scale()` provided for API convenience.
/// Formula: z = (x - μ) / σ
///
/// # Arguments
/// * `series` - Input time series
///
/// # Returns
/// Normalized time series with mean ≈ 0 and std ≈ 1
///
/// # See Also
/// - `standard_scale()` - Direct implementation
/// - `min_max_scale()` - Alternative scaling to [0, 1]
pub fn normalize(series: &TimeSeries) -> TimeSeries {
    standard_scale(series)
}

/// Apply moving average
///
/// Computes a trailing moving average with the specified window size.
/// For each position i, calculates the mean of values from max(0, i-window+1) to i.
///
/// # Arguments
/// * `series` - Input time series
/// * `window` - Window size for averaging (must be >= 1)
///
/// # Returns
/// Smoothed time series with same length as input.
/// Early values (where full window isn't available) use smaller windows.
///
/// # Examples
/// ```text
/// // For series [1, 2, 3, 4, 5] with window=3:
/// // Position 0: mean([1]) = 1.0
/// // Position 1: mean([1, 2]) = 1.5
/// // Position 2: mean([1, 2, 3]) = 2.0
/// // Position 3: mean([2, 3, 4]) = 3.0
/// // Position 4: mean([3, 4, 5]) = 4.0
/// ```text
pub fn moving_average(series: &TimeSeries, window: usize) -> TimeSeries {
    if window == 0 {
        return TimeSeries::new(series.values.clone());
    }

    let data = series.values.to_vec().unwrap_or_default();
    if data.is_empty() {
        return TimeSeries::new(series.values.clone());
    }

    let n = data.len();
    let mut result = Vec::with_capacity(n);

    // Compute trailing moving average
    for i in 0..n {
        // Window starts at max(0, i - window + 1) and ends at i (inclusive)
        let start = if i + 1 >= window { i + 1 - window } else { 0 };
        let window_size = i - start + 1;

        // Calculate mean of values in window
        let sum: f32 = data[start..=i].iter().sum();
        let mean = sum / window_size as f32;
        result.push(mean);
    }

    let tensor = Tensor::from_vec(result, &[n]).expect("tensor creation should succeed");
    TimeSeries::new(tensor)
}

/// Apply exponential moving average
pub fn ema(series: &TimeSeries, alpha: f64) -> TimeSeries {
    let mut result = vec![0.0f32; series.len()];
    let values = series.values.to_vec().expect("conversion should succeed");

    result[0] = values[0];
    for i in 1..values.len() {
        result[i] = (alpha as f32) * values[i] + ((1.0 - alpha) as f32) * result[i - 1];
    }

    let tensor = Tensor::from_vec(result, &[series.len()]).expect("tensor creation should succeed");
    TimeSeries::new(tensor)
}

/// Box-Cox transformation
///
/// A power transformation that stabilizes variance and makes data more normal-distributed.
///
/// Formula:
/// - If λ = 0: y = ln(x)
/// - If λ ≠ 0: y = (x^λ - 1) / λ
///
/// # Arguments
/// * `series` - Input time series (all values must be positive)
/// * `lambda` - Transformation parameter
///
/// # Returns
/// Transformed time series
pub fn box_cox(series: &TimeSeries, lambda: f32) -> TimeSeries {
    let data = series.values.to_vec().unwrap_or_default();

    let transformed: Vec<f32> = data
        .iter()
        .map(|&x| {
            if x <= 0.0 {
                // Box-Cox requires positive values
                // Return a sentinel or handle error - here we'll use a small positive value
                if lambda.abs() < 1e-6 {
                    // lambda ≈ 0: use log transformation
                    (1e-10f32).ln()
                } else {
                    // lambda ≠ 0: use power transformation
                    ((1e-10f32).powf(lambda) - 1.0) / lambda
                }
            } else if lambda.abs() < 1e-6 {
                // lambda ≈ 0: use log transformation
                x.ln()
            } else {
                // lambda ≠ 0: use power transformation
                (x.powf(lambda) - 1.0) / lambda
            }
        })
        .collect();

    let tensor =
        Tensor::from_vec(transformed, &[series.len()]).expect("tensor creation should succeed");
    TimeSeries::new(tensor)
}

/// Inverse Box-Cox transformation
///
/// Reverses the Box-Cox transformation to recover original scale.
///
/// Formula:
/// - If λ = 0: x = exp(y)
/// - If λ ≠ 0: x = (λ * y + 1)^(1/λ)
///
/// # Arguments
/// * `series` - Transformed time series
/// * `lambda` - Same λ parameter used in forward transformation
///
/// # Returns
/// Original-scale time series
pub fn inv_box_cox(series: &TimeSeries, lambda: f32) -> TimeSeries {
    let data = series.values.to_vec().unwrap_or_default();

    let inv_transformed: Vec<f32> = data
        .iter()
        .map(|&y| {
            if lambda.abs() < 1e-6 {
                // lambda ≈ 0: inverse of log is exp
                y.exp()
            } else {
                // lambda ≠ 0: inverse power transformation
                let base = lambda * y + 1.0;
                if base > 0.0 {
                    base.powf(1.0 / lambda)
                } else {
                    // Handle numerical issues
                    1e-10
                }
            }
        })
        .collect();

    let tensor =
        Tensor::from_vec(inv_transformed, &[series.len()]).expect("tensor creation should succeed");
    TimeSeries::new(tensor)
}

/// Standard scaling (zero mean, unit variance)
///
/// Also known as z-score normalization. Transforms data to have mean=0 and std=1.
///
/// Formula: z = (x - μ) / σ
/// where μ is the mean and σ is the standard deviation
///
/// # Arguments
/// * `series` - Input time series
///
/// # Returns
/// Standardized time series with mean ≈ 0 and std ≈ 1
pub fn standard_scale(series: &TimeSeries) -> TimeSeries {
    let data = series.values.to_vec().unwrap_or_default();

    if data.is_empty() {
        return TimeSeries::new(series.values.clone());
    }

    // Compute mean
    let mean = data.iter().sum::<f32>() / data.len() as f32;

    // Compute standard deviation
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    let std = variance.sqrt();

    // Avoid division by zero
    if std < 1e-10 {
        // All values are the same, return zeros
        let zeros_vec = vec![0.0f32; data.len()];
        let tensor =
            Tensor::from_vec(zeros_vec, &[series.len()]).expect("tensor creation should succeed");
        return TimeSeries::new(tensor);
    }

    // Standardize: z = (x - mean) / std
    let standardized: Vec<f32> = data.iter().map(|&x| (x - mean) / std).collect();

    let tensor =
        Tensor::from_vec(standardized, &[series.len()]).expect("tensor creation should succeed");
    TimeSeries::new(tensor)
}

/// Min-max scaling to [0, 1]
///
/// Scales data linearly to the range [0, 1].
///
/// Formula: x_scaled = (x - min) / (max - min)
///
/// # Arguments
/// * `series` - Input time series
///
/// # Returns
/// Scaled time series with values in [0, 1]
pub fn min_max_scale(series: &TimeSeries) -> TimeSeries {
    let data = series.values.to_vec().unwrap_or_default();

    if data.is_empty() {
        return TimeSeries::new(series.values.clone());
    }

    // Find min and max
    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Avoid division by zero
    let range = max_val - min_val;
    if range < 1e-10 {
        // All values are the same, return zeros (or could return 0.5)
        let zeros_vec = vec![0.0f32; data.len()];
        let tensor =
            Tensor::from_vec(zeros_vec, &[series.len()]).expect("tensor creation should succeed");
        return TimeSeries::new(tensor);
    }

    // Scale to [0, 1]: x_scaled = (x - min) / (max - min)
    let scaled: Vec<f32> = data.iter().map(|&x| (x - min_val) / range).collect();

    let tensor = Tensor::from_vec(scaled, &[series.len()]).expect("tensor creation should succeed");
    TimeSeries::new(tensor)
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
    fn test_diff() {
        let series = create_test_series(); // [1.0, 2.0, 3.0, 4.0, 5.0]

        // Test first-order differencing
        let diffed = diff(&series, 1);
        assert_eq!(diffed.len(), series.len() - 1); // Length reduced by 1

        let result = diffed.values.to_vec().unwrap();
        // First differences: [2-1, 3-2, 4-3, 5-4] = [1, 1, 1, 1]
        assert_eq!(result.len(), 4);
        for &val in &result {
            assert!((val - 1.0).abs() < 1e-6, "First difference should be 1.0");
        }

        // Test second-order differencing
        let diffed2 = diff(&series, 2);
        assert_eq!(diffed2.len(), series.len() - 2); // Length reduced by 2

        let result2 = diffed2.values.to_vec().unwrap();
        // Second differences: diff of [1, 1, 1, 1] = [0, 0, 0]
        assert_eq!(result2.len(), 3);
        for &val in &result2 {
            assert!(val.abs() < 1e-6, "Second difference should be 0.0");
        }

        // Test zero-order (no differencing)
        let diffed0 = diff(&series, 0);
        assert_eq!(diffed0.len(), series.len());
    }

    #[test]
    fn test_detrend() {
        let series = create_test_series(); // [1.0, 2.0, 3.0, 4.0, 5.0]

        // Test linear detrending
        let detrended = detrend(&series, "linear");
        assert_eq!(detrended.len(), series.len());

        let result = detrended.values.to_vec().unwrap();

        // Original series has perfect linear trend: y = 1.0*t + 1.0
        // After removing this trend, residuals should be ~0
        // The fitted line through [1,2,3,4,5] at t=[0,1,2,3,4] should match exactly
        for &val in &result {
            assert!(val.abs() < 1e-5, "Detrended value {} should be near 0", val);
        }

        // Test mean detrending
        let mean_detrended = detrend(&series, "mean");
        let mean_result = mean_detrended.values.to_vec().unwrap();

        // After mean detrending, the mean should be ~0
        let mean_after: f32 = mean_result.iter().sum::<f32>() / mean_result.len() as f32;
        assert!(
            mean_after.abs() < 1e-5,
            "Mean after detrending should be ~0"
        );
    }

    #[test]
    fn test_normalize() {
        let series = create_test_series(); // [1.0, 2.0, 3.0, 4.0, 5.0]
        let normalized = normalize(&series);
        assert_eq!(normalized.len(), series.len());

        let result = normalized.values.to_vec().unwrap();

        // After normalization, mean should be ~0
        let mean = result.iter().sum::<f32>() / result.len() as f32;
        assert!(mean.abs() < 1e-5, "Mean after normalization should be ~0");

        // After normalization, std should be ~1
        let variance =
            result.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / result.len() as f32;
        let std = variance.sqrt();
        assert!(
            (std - 1.0).abs() < 1e-5,
            "Std after normalization should be ~1"
        );
    }

    #[test]
    fn test_ema() {
        let series = create_test_series();
        let smoothed = ema(&series, 0.3);
        assert_eq!(smoothed.len(), series.len());
    }

    #[test]
    fn test_moving_average() {
        let series = create_test_series(); // [1.0, 2.0, 3.0, 4.0, 5.0]
        let smoothed = moving_average(&series, 3);
        assert_eq!(smoothed.len(), series.len());

        let result = smoothed.values.to_vec().unwrap();
        // Position 0: mean([1]) = 1.0
        assert!((result[0] - 1.0).abs() < 1e-6);
        // Position 1: mean([1, 2]) = 1.5
        assert!((result[1] - 1.5).abs() < 1e-6);
        // Position 2: mean([1, 2, 3]) = 2.0
        assert!((result[2] - 2.0).abs() < 1e-6);
        // Position 3: mean([2, 3, 4]) = 3.0
        assert!((result[3] - 3.0).abs() < 1e-6);
        // Position 4: mean([3, 4, 5]) = 4.0
        assert!((result[4] - 4.0).abs() < 1e-6);
    }
}
