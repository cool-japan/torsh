//! Time series forecasting metrics
//!
//! This module provides metrics specifically designed for evaluating time series forecasts.
//! All metrics follow SciRS2 POLICY - using scirs2-core abstractions.

use torsh_core::error::TorshError;
use torsh_tensor::Tensor;

/// Mean Absolute Scaled Error (MASE)
///
/// Scale-independent measure of forecast accuracy. Values < 1 indicate the forecast
/// is better than the naive seasonal forecast, > 1 means worse.
///
/// MASE = mean(|y_t - ŷ_t|) / mean(|y_t - y_{t-m}|)
/// where m is the seasonal period (default 1 for non-seasonal data)
pub fn mase(
    y_true: &Tensor,
    y_pred: &Tensor,
    y_train: Option<&Tensor>,
    seasonal_period: usize,
) -> Result<f64, TorshError> {
    let true_vec = y_true.to_vec()?;
    let pred_vec = y_pred.to_vec()?;

    if true_vec.len() != pred_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    // Calculate MAE of forecast
    let mae: f64 = true_vec
        .iter()
        .zip(pred_vec.iter())
        .map(|(&t, &p)| (t as f64 - p as f64).abs())
        .sum::<f64>()
        / true_vec.len() as f64;

    // Calculate scaling factor from training data
    let scale = if let Some(train) = y_train {
        let train_vec = train.to_vec()?;
        if train_vec.len() <= seasonal_period {
            return Err(TorshError::InvalidArgument(
                "Training data too short for seasonal period".to_string(),
            ));
        }

        let naive_errors: f64 = train_vec[seasonal_period..]
            .iter()
            .zip(train_vec[..train_vec.len() - seasonal_period].iter())
            .map(|(&t, &t_prev)| (t as f64 - t_prev as f64).abs())
            .sum();

        naive_errors / (train_vec.len() - seasonal_period) as f64
    } else {
        // If no training data, use naive forecast on test data
        if true_vec.len() <= seasonal_period {
            return Err(TorshError::InvalidArgument(
                "Test data too short for seasonal period".to_string(),
            ));
        }

        let naive_errors: f64 = true_vec[seasonal_period..]
            .iter()
            .zip(true_vec[..true_vec.len() - seasonal_period].iter())
            .map(|(&t, &t_prev)| (t as f64 - t_prev as f64).abs())
            .sum();

        naive_errors / (true_vec.len() - seasonal_period) as f64
    };

    if scale == 0.0 {
        return Err(TorshError::InvalidArgument(
            "Scaling factor is zero (constant series)".to_string(),
        ));
    }

    Ok(mae / scale)
}

/// Symmetric Mean Absolute Percentage Error (SMAPE)
///
/// Symmetric version of MAPE that addresses issues with asymmetry and undefined values.
/// Returns percentage error in range [0, 200].
///
/// SMAPE = (100/n) * Σ|y_t - ŷ_t| / ((|y_t| + |ŷ_t|) / 2)
pub fn smape(y_true: &Tensor, y_pred: &Tensor) -> Result<f64, TorshError> {
    let true_vec = y_true.to_vec()?;
    let pred_vec = y_pred.to_vec()?;

    if true_vec.len() != pred_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    let mut sum = 0.0;
    let mut count = 0;

    for (&t, &p) in true_vec.iter().zip(pred_vec.iter()) {
        let t = t as f64;
        let p = p as f64;
        let denominator = (t.abs() + p.abs()) / 2.0;

        if denominator > 1e-10 {
            sum += (t - p).abs() / denominator;
            count += 1;
        }
    }

    if count == 0 {
        return Err(TorshError::InvalidArgument(
            "All values too close to zero".to_string(),
        ));
    }

    Ok(100.0 * sum / count as f64)
}

/// Mean Absolute Percentage Error (MAPE)
///
/// Standard percentage error metric. Undefined when true values are zero.
/// Returns percentage error.
///
/// MAPE = (100/n) * Σ|y_t - ŷ_t| / |y_t|
pub fn mape(y_true: &Tensor, y_pred: &Tensor) -> Result<f64, TorshError> {
    let true_vec = y_true.to_vec()?;
    let pred_vec = y_pred.to_vec()?;

    if true_vec.len() != pred_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    let mut sum = 0.0;
    let mut count = 0;

    for (&t, &p) in true_vec.iter().zip(pred_vec.iter()) {
        let t = t as f64;
        let p = p as f64;

        if t.abs() > 1e-10 {
            sum += ((t - p).abs() / t.abs()) * 100.0;
            count += 1;
        }
    }

    if count == 0 {
        return Err(TorshError::InvalidArgument(
            "All true values are zero".to_string(),
        ));
    }

    Ok(sum / count as f64)
}

/// Mean Scaled Interval Score (MSIS)
///
/// Evaluates prediction intervals for time series forecasts.
/// Lower is better.
///
/// # Arguments
/// * `y_true` - Actual values
/// * `lower_bound` - Lower bound of prediction interval
/// * `upper_bound` - Upper bound of prediction interval
/// * `alpha` - Coverage probability (e.g., 0.05 for 95% interval)
/// * `y_train` - Training data for scaling (optional)
/// * `seasonal_period` - Seasonal period for scaling
pub fn msis(
    y_true: &Tensor,
    lower_bound: &Tensor,
    upper_bound: &Tensor,
    alpha: f64,
    y_train: Option<&Tensor>,
    seasonal_period: usize,
) -> Result<f64, TorshError> {
    let true_vec = y_true.to_vec()?;
    let lower_vec = lower_bound.to_vec()?;
    let upper_vec = upper_bound.to_vec()?;

    if true_vec.len() != lower_vec.len() || true_vec.len() != upper_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    // Calculate interval score
    let mut interval_score = 0.0;
    for i in 0..true_vec.len() {
        let y = true_vec[i] as f64;
        let l = lower_vec[i] as f64;
        let u = upper_vec[i] as f64;

        let width = u - l;
        let lower_penalty = if y < l { 2.0 / alpha * (l - y) } else { 0.0 };
        let upper_penalty = if y > u { 2.0 / alpha * (y - u) } else { 0.0 };

        interval_score += width + lower_penalty + upper_penalty;
    }
    interval_score /= true_vec.len() as f64;

    // Calculate scaling factor
    let scale = if let Some(train) = y_train {
        let train_vec = train.to_vec()?;
        if train_vec.len() <= seasonal_period {
            return Err(TorshError::InvalidArgument(
                "Training data too short for seasonal period".to_string(),
            ));
        }

        let naive_errors: f64 = train_vec[seasonal_period..]
            .iter()
            .zip(train_vec[..train_vec.len() - seasonal_period].iter())
            .map(|(&t, &t_prev)| (t as f64 - t_prev as f64).abs())
            .sum();

        naive_errors / (train_vec.len() - seasonal_period) as f64
    } else {
        1.0 // No scaling if training data not provided
    };

    if scale == 0.0 {
        return Err(TorshError::InvalidArgument(
            "Scaling factor is zero".to_string(),
        ));
    }

    Ok(interval_score / scale)
}

/// Theil's U Statistic
///
/// Measures forecast accuracy relative to a naive no-change forecast.
/// Values < 1 indicate better than naive, = 1 equal to naive, > 1 worse than naive.
///
/// U = sqrt(Σ(y_t - ŷ_t)²) / sqrt(Σ(y_t - y_{t-1})²)
pub fn theil_u(y_true: &Tensor, y_pred: &Tensor) -> Result<f64, TorshError> {
    let true_vec = y_true.to_vec()?;
    let pred_vec = y_pred.to_vec()?;

    if true_vec.len() != pred_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    if true_vec.len() < 2 {
        return Err(TorshError::InvalidArgument(
            "Need at least 2 points for Theil's U".to_string(),
        ));
    }

    // Forecast error
    let forecast_mse: f64 = true_vec
        .iter()
        .zip(pred_vec.iter())
        .map(|(&t, &p)| {
            let diff = t as f64 - p as f64;
            diff * diff
        })
        .sum::<f64>();

    // Naive forecast error (no-change forecast)
    let naive_mse: f64 = true_vec[1..]
        .iter()
        .zip(true_vec[..true_vec.len() - 1].iter())
        .map(|(&t, &t_prev)| {
            let diff = t as f64 - t_prev as f64;
            diff * diff
        })
        .sum::<f64>();

    if naive_mse == 0.0 {
        return Err(TorshError::InvalidArgument(
            "Naive forecast error is zero (constant series)".to_string(),
        ));
    }

    Ok((forecast_mse / naive_mse).sqrt())
}

/// Mean Directional Accuracy (MDA)
///
/// Percentage of times the forecast correctly predicts the direction of change.
/// Returns value in [0, 100].
pub fn mean_directional_accuracy(y_true: &Tensor, y_pred: &Tensor) -> Result<f64, TorshError> {
    let true_vec = y_true.to_vec()?;
    let pred_vec = y_pred.to_vec()?;

    if true_vec.len() != pred_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    if true_vec.len() < 2 {
        return Err(TorshError::InvalidArgument(
            "Need at least 2 points for MDA".to_string(),
        ));
    }

    let mut correct = 0;
    for i in 1..true_vec.len() {
        let true_direction = true_vec[i] as f64 - true_vec[i - 1] as f64;
        let pred_direction = pred_vec[i] as f64 - pred_vec[i - 1] as f64;

        if true_direction * pred_direction >= 0.0 {
            correct += 1;
        }
    }

    Ok(100.0 * correct as f64 / (true_vec.len() - 1) as f64)
}

/// Tracking Signal
///
/// Monitors forecast bias by tracking cumulative forecast error.
/// Large absolute values indicate systematic bias.
///
/// TS = Σ(y_t - ŷ_t) / MAD where MAD = mean absolute deviation
pub fn tracking_signal(y_true: &Tensor, y_pred: &Tensor) -> Result<f64, TorshError> {
    let true_vec = y_true.to_vec()?;
    let pred_vec = y_pred.to_vec()?;

    if true_vec.len() != pred_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    let mut cumulative_error = 0.0;
    let mut absolute_errors = 0.0;

    for (&t, &p) in true_vec.iter().zip(pred_vec.iter()) {
        let error = t as f64 - p as f64;
        cumulative_error += error;
        absolute_errors += error.abs();
    }

    let mad = absolute_errors / true_vec.len() as f64;

    if mad < 1e-10 {
        return Err(TorshError::InvalidArgument(
            "Mean absolute deviation too small".to_string(),
        ));
    }

    Ok(cumulative_error / mad)
}

/// Dynamic Time Warping (DTW) Distance
///
/// Measures similarity between two time series by finding optimal alignment.
/// Lower values indicate more similar series.
///
/// Note: This is a basic implementation without optimizations like window constraints.
pub fn dtw_distance(series1: &Tensor, series2: &Tensor) -> Result<f64, TorshError> {
    let s1 = series1.to_vec()?;
    let s2 = series2.to_vec()?;

    let n = s1.len();
    let m = s2.len();

    if n == 0 || m == 0 {
        return Err(TorshError::InvalidArgument("Empty series".to_string()));
    }

    // DTW matrix
    let mut dtw = vec![vec![f64::INFINITY; m + 1]; n + 1];
    dtw[0][0] = 0.0;

    for i in 1..=n {
        for j in 1..=m {
            let cost = (s1[i - 1] as f64 - s2[j - 1] as f64).abs();
            dtw[i][j] = cost + dtw[i - 1][j].min(dtw[i][j - 1]).min(dtw[i - 1][j - 1]);
        }
    }

    Ok(dtw[n][m])
}

/// Autocorrelation of forecast errors
///
/// Measures whether forecast errors are correlated with previous errors.
/// Values near 0 indicate independent errors (desirable).
/// Lag specifies which previous error to correlate with.
pub fn error_autocorrelation(
    y_true: &Tensor,
    y_pred: &Tensor,
    lag: usize,
) -> Result<f64, TorshError> {
    let true_vec = y_true.to_vec()?;
    let pred_vec = y_pred.to_vec()?;

    if true_vec.len() != pred_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    let errors: Vec<f64> = true_vec
        .iter()
        .zip(pred_vec.iter())
        .map(|(&t, &p)| t as f64 - p as f64)
        .collect();

    if errors.len() <= lag {
        return Err(TorshError::InvalidArgument(
            "Not enough data for specified lag".to_string(),
        ));
    }

    // Calculate mean
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;

    // Calculate autocorrelation
    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in lag..errors.len() {
        numerator += (errors[i] - mean) * (errors[i - lag] - mean);
    }

    for error in &errors {
        denominator += (error - mean).powi(2);
    }

    if denominator < 1e-10 {
        return Err(TorshError::InvalidArgument(
            "Variance too small for autocorrelation".to_string(),
        ));
    }

    Ok(numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_mase_simple() {
        let y_true = from_vec(vec![3.0, 4.0, 5.0, 6.0], &[4], torsh_core::DeviceType::Cpu).unwrap();
        let y_pred = from_vec(vec![2.5, 4.2, 5.1, 5.8], &[4], torsh_core::DeviceType::Cpu).unwrap();
        let y_train = from_vec(vec![1.0, 2.0, 3.0], &[3], torsh_core::DeviceType::Cpu).unwrap();

        let result = mase(&y_true, &y_pred, Some(&y_train), 1).unwrap();
        assert!(result > 0.0);
    }

    #[test]
    fn test_smape_symmetric() {
        let y_true = from_vec(vec![100.0, 200.0], &[2], torsh_core::DeviceType::Cpu).unwrap();
        let y_pred = from_vec(vec![110.0, 180.0], &[2], torsh_core::DeviceType::Cpu).unwrap();

        let result = smape(&y_true, &y_pred).unwrap();
        assert!(result > 0.0 && result <= 200.0);
    }

    #[test]
    fn test_smape_perfect_forecast() {
        let y_true = from_vec(vec![1.0, 2.0, 3.0], &[3], torsh_core::DeviceType::Cpu).unwrap();
        let y_pred = from_vec(vec![1.0, 2.0, 3.0], &[3], torsh_core::DeviceType::Cpu).unwrap();

        let result = smape(&y_true, &y_pred).unwrap();
        assert_relative_eq!(result, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mape_calculation() {
        let y_true =
            from_vec(vec![100.0, 200.0, 150.0], &[3], torsh_core::DeviceType::Cpu).unwrap();
        let y_pred =
            from_vec(vec![110.0, 190.0, 160.0], &[3], torsh_core::DeviceType::Cpu).unwrap();

        let result = mape(&y_true, &y_pred).unwrap();
        assert!(result > 0.0 && result < 100.0);
    }

    #[test]
    fn test_theil_u_perfect_forecast() {
        let y_true = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], torsh_core::DeviceType::Cpu).unwrap();
        let y_pred = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], torsh_core::DeviceType::Cpu).unwrap();

        let result = theil_u(&y_true, &y_pred).unwrap();
        assert_relative_eq!(result, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mean_directional_accuracy_perfect() {
        let y_true = from_vec(
            vec![1.0, 2.0, 3.0, 2.5, 3.5],
            &[5],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();
        let y_pred = from_vec(
            vec![1.0, 2.1, 3.2, 2.6, 3.6],
            &[5],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();

        let result = mean_directional_accuracy(&y_true, &y_pred).unwrap();
        assert_relative_eq!(result, 100.0, epsilon = 1e-6);
    }

    #[test]
    fn test_tracking_signal() {
        let y_true = from_vec(
            vec![100.0, 110.0, 105.0, 115.0],
            &[4],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();
        let y_pred = from_vec(
            vec![98.0, 108.0, 103.0, 113.0],
            &[4],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();

        let result = tracking_signal(&y_true, &y_pred).unwrap();
        assert!(result.is_finite());
    }

    #[test]
    fn test_dtw_distance_identical() {
        let s1 = from_vec(vec![1.0, 2.0, 3.0], &[3], torsh_core::DeviceType::Cpu).unwrap();
        let s2 = from_vec(vec![1.0, 2.0, 3.0], &[3], torsh_core::DeviceType::Cpu).unwrap();

        let result = dtw_distance(&s1, &s2).unwrap();
        assert_relative_eq!(result, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dtw_distance_different() {
        let s1 = from_vec(vec![1.0, 2.0, 3.0], &[3], torsh_core::DeviceType::Cpu).unwrap();
        let s2 = from_vec(vec![2.0, 3.0, 4.0], &[3], torsh_core::DeviceType::Cpu).unwrap();

        let result = dtw_distance(&s1, &s2).unwrap();
        assert!(result > 0.0);
    }

    #[test]
    fn test_error_autocorrelation() {
        let y_true = from_vec(
            vec![1.0, 2.0, 1.5, 2.5, 2.0, 3.0],
            &[6],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();
        let y_pred = from_vec(
            vec![1.1, 1.9, 1.6, 2.4, 2.1, 2.9],
            &[6],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();

        let result = error_autocorrelation(&y_true, &y_pred, 1).unwrap();
        assert!(result.abs() <= 1.0); // Autocorrelation bounded by [-1, 1]
    }

    #[test]
    fn test_msis_calculation() {
        let y_true =
            from_vec(vec![100.0, 110.0, 105.0], &[3], torsh_core::DeviceType::Cpu).unwrap();
        let lower = from_vec(vec![95.0, 105.0, 100.0], &[3], torsh_core::DeviceType::Cpu).unwrap();
        let upper = from_vec(vec![105.0, 115.0, 110.0], &[3], torsh_core::DeviceType::Cpu).unwrap();

        let result = msis(&y_true, &lower, &upper, 0.05, None, 1).unwrap();
        assert!(result > 0.0);
    }
}
