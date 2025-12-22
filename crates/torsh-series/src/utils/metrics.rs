//! Time series evaluation metrics

use crate::TimeSeries;

/// Mean Absolute Error
///
/// Calculates MAE = (1/n) Σ |y_i - ŷ_i|
///
/// # Arguments
/// * `actual` - The actual observed values
/// * `predicted` - The predicted values
///
/// # Returns
/// Mean absolute error, or 0.0 if lengths don't match or series is empty
pub fn mae(actual: &TimeSeries, predicted: &TimeSeries) -> f64 {
    let actual_data = actual.values.to_vec().unwrap_or_default();
    let predicted_data = predicted.values.to_vec().unwrap_or_default();

    if actual_data.len() != predicted_data.len() || actual_data.is_empty() {
        return 0.0;
    }

    let n = actual_data.len() as f64;
    let sum_abs_errors: f64 = actual_data
        .iter()
        .zip(predicted_data.iter())
        .map(|(&a, &p)| ((a - p) as f64).abs())
        .sum();

    sum_abs_errors / n
}

/// Mean Squared Error
///
/// Calculates MSE = (1/n) Σ (y_i - ŷ_i)²
///
/// # Arguments
/// * `actual` - The actual observed values
/// * `predicted` - The predicted values
///
/// # Returns
/// Mean squared error, or 0.0 if lengths don't match or series is empty
pub fn mse(actual: &TimeSeries, predicted: &TimeSeries) -> f64 {
    let actual_data = actual.values.to_vec().unwrap_or_default();
    let predicted_data = predicted.values.to_vec().unwrap_or_default();

    if actual_data.len() != predicted_data.len() || actual_data.is_empty() {
        return 0.0;
    }

    let n = actual_data.len() as f64;
    let sum_sq_errors: f64 = actual_data
        .iter()
        .zip(predicted_data.iter())
        .map(|(&a, &p)| {
            let err = (a - p) as f64;
            err * err
        })
        .sum();

    sum_sq_errors / n
}

/// Root Mean Squared Error
pub fn rmse(actual: &TimeSeries, predicted: &TimeSeries) -> f64 {
    mse(actual, predicted).sqrt()
}

/// Mean Absolute Percentage Error
///
/// Calculates MAPE = (100/n) Σ |y_i - ŷ_i| / |y_i|
///
/// Note: Returns 0.0 if any actual value is zero (to avoid division by zero).
/// Consider using sMAPE for series with zeros.
///
/// # Arguments
/// * `actual` - The actual observed values
/// * `predicted` - The predicted values
///
/// # Returns
/// Mean absolute percentage error (0-100 scale), or 0.0 if invalid
pub fn mape(actual: &TimeSeries, predicted: &TimeSeries) -> f64 {
    let actual_data = actual.values.to_vec().unwrap_or_default();
    let predicted_data = predicted.values.to_vec().unwrap_or_default();

    if actual_data.len() != predicted_data.len() || actual_data.is_empty() {
        return 0.0;
    }

    // Check for zeros in actual data
    if actual_data.iter().any(|&a| a.abs() < 1e-10) {
        return 0.0;
    }

    let n = actual_data.len() as f64;
    let sum_pct_errors: f64 = actual_data
        .iter()
        .zip(predicted_data.iter())
        .map(|(&a, &p)| {
            let a_f64 = a as f64;
            let p_f64 = p as f64;
            ((a_f64 - p_f64).abs() / a_f64.abs()) * 100.0
        })
        .sum();

    sum_pct_errors / n
}

/// Symmetric Mean Absolute Percentage Error
///
/// Calculates sMAPE = (100/n) Σ 2|y_i - ŷ_i| / (|y_i| + |ŷ_i|)
///
/// This metric is symmetric and handles zeros better than MAPE.
/// Returns values in [0, 100] range.
///
/// # Arguments
/// * `actual` - The actual observed values
/// * `predicted` - The predicted values
///
/// # Returns
/// Symmetric mean absolute percentage error (0-100 scale)
pub fn smape(actual: &TimeSeries, predicted: &TimeSeries) -> f64 {
    let actual_data = actual.values.to_vec().unwrap_or_default();
    let predicted_data = predicted.values.to_vec().unwrap_or_default();

    if actual_data.len() != predicted_data.len() || actual_data.is_empty() {
        return 0.0;
    }

    let n = actual_data.len() as f64;
    let sum_smape: f64 = actual_data
        .iter()
        .zip(predicted_data.iter())
        .map(|(&a, &p)| {
            let a_f64 = a as f64;
            let p_f64 = p as f64;
            let numerator = (a_f64 - p_f64).abs();
            let denominator = a_f64.abs() + p_f64.abs();

            if denominator < 1e-10 {
                0.0 // Both actual and predicted are zero
            } else {
                (2.0 * numerator / denominator) * 100.0
            }
        })
        .sum();

    sum_smape / n
}

/// R-squared (Coefficient of Determination)
///
/// Calculates R² = 1 - (SS_res / SS_tot)
/// where SS_res = Σ(y_i - ŷ_i)² and SS_tot = Σ(y_i - ȳ)²
///
/// R² = 1 indicates perfect prediction, R² = 0 indicates predictions
/// equal to the mean, and R² < 0 indicates predictions worse than mean.
///
/// # Arguments
/// * `actual` - The actual observed values
/// * `predicted` - The predicted values
///
/// # Returns
/// R-squared value, typically in range (-∞, 1]
pub fn r2(actual: &TimeSeries, predicted: &TimeSeries) -> f64 {
    let actual_data = actual.values.to_vec().unwrap_or_default();
    let predicted_data = predicted.values.to_vec().unwrap_or_default();

    if actual_data.len() != predicted_data.len() || actual_data.is_empty() {
        return 0.0;
    }

    let n = actual_data.len() as f64;

    // Calculate mean of actual values
    let mean_actual: f64 = actual_data.iter().map(|&a| a as f64).sum::<f64>() / n;

    // Calculate SS_res (residual sum of squares)
    let ss_res: f64 = actual_data
        .iter()
        .zip(predicted_data.iter())
        .map(|(&a, &p)| {
            let err = (a as f64) - (p as f64);
            err * err
        })
        .sum();

    // Calculate SS_tot (total sum of squares)
    let ss_tot: f64 = actual_data
        .iter()
        .map(|&a| {
            let diff = (a as f64) - mean_actual;
            diff * diff
        })
        .sum();

    if ss_tot < 1e-10 {
        // All actual values are equal (constant series)
        return 0.0;
    }

    1.0 - (ss_res / ss_tot)
}

/// Mean Absolute Scaled Error
///
/// Calculates MASE = MAE / MAE_naive
/// where MAE_naive is the MAE of a naive seasonal forecast.
///
/// MASE is scale-independent and useful for comparing forecast accuracy
/// across different datasets. MASE < 1 indicates the forecast is better
/// than the naive seasonal method.
///
/// # Arguments
/// * `actual` - The actual observed values
/// * `predicted` - The predicted values
/// * `seasonal_period` - The seasonal period (1 for non-seasonal, 12 for monthly, etc.)
///
/// # Returns
/// Mean absolute scaled error
pub fn mase(actual: &TimeSeries, predicted: &TimeSeries, seasonal_period: usize) -> f64 {
    let actual_data = actual.values.to_vec().unwrap_or_default();
    let predicted_data = predicted.values.to_vec().unwrap_or_default();

    if actual_data.len() != predicted_data.len() || actual_data.len() <= seasonal_period {
        return 0.0;
    }

    let n = actual_data.len();

    // Calculate MAE of forecast
    let forecast_mae = mae(actual, predicted);

    // Calculate MAE of naive seasonal forecast
    // Naive forecast: y_hat[t] = y[t - seasonal_period]
    let naive_errors_sum: f64 = (seasonal_period..n)
        .map(|i| {
            let actual_val = actual_data[i] as f64;
            let naive_pred = actual_data[i - seasonal_period] as f64;
            (actual_val - naive_pred).abs()
        })
        .sum();

    let naive_mae = naive_errors_sum / ((n - seasonal_period) as f64);

    if naive_mae < 1e-10 {
        return 0.0; // Avoid division by zero
    }

    forecast_mae / naive_mae
}

/// Theil's U Statistic (Uncertainty Coefficient)
///
/// Calculates Theil's U = √(MSE) / (√(Σy²/n) + √(Σŷ²/n))
///
/// This metric is scale-independent and ranges from 0 to 1.
/// U = 0 indicates perfect forecast, U = 1 indicates forecast
/// is no better than a naive no-change forecast.
///
/// # Arguments
/// * `actual` - The actual observed values
/// * `predicted` - The predicted values
///
/// # Returns
/// Theil's U statistic, typically in range [0, 1]
pub fn theil_u(actual: &TimeSeries, predicted: &TimeSeries) -> f64 {
    let actual_data = actual.values.to_vec().unwrap_or_default();
    let predicted_data = predicted.values.to_vec().unwrap_or_default();

    if actual_data.len() != predicted_data.len() || actual_data.is_empty() {
        return 0.0;
    }

    let n = actual_data.len() as f64;

    // Calculate RMSE
    let mse_val = mse(actual, predicted);
    let rmse_val = mse_val.sqrt();

    // Calculate RMS of actual values
    let rms_actual = (actual_data
        .iter()
        .map(|&a| {
            let val = a as f64;
            val * val
        })
        .sum::<f64>()
        / n)
        .sqrt();

    // Calculate RMS of predicted values
    let rms_predicted = (predicted_data
        .iter()
        .map(|&p| {
            let val = p as f64;
            val * val
        })
        .sum::<f64>()
        / n)
        .sqrt();

    let denominator = rms_actual + rms_predicted;

    if denominator < 1e-10 {
        return 0.0; // Both series are zero
    }

    rmse_val / denominator
}

/// Directional Accuracy
///
/// Calculates the percentage of times the forecast correctly predicts
/// the direction of change (increase/decrease) compared to the previous value.
///
/// This metric is useful for applications where the direction of change
/// is more important than the magnitude (e.g., stock price movement).
///
/// # Arguments
/// * `actual` - The actual observed values
/// * `predicted` - The predicted values
///
/// # Returns
/// Percentage of correct directional predictions (0-100 scale)
pub fn directional_accuracy(actual: &TimeSeries, predicted: &TimeSeries) -> f64 {
    let actual_data = actual.values.to_vec().unwrap_or_default();
    let predicted_data = predicted.values.to_vec().unwrap_or_default();

    if actual_data.len() != predicted_data.len() || actual_data.len() < 2 {
        return 0.0;
    }

    let n = actual_data.len();
    let mut correct_directions = 0;
    let mut total_comparisons = 0;

    for i in 1..n {
        let actual_change = (actual_data[i] - actual_data[i - 1]) as f64;
        let predicted_change = (predicted_data[i] - predicted_data[i - 1]) as f64;

        // Check if both changes have the same sign (both positive or both negative)
        if actual_change.abs() > 1e-10 || predicted_change.abs() > 1e-10 {
            total_comparisons += 1;
            if actual_change * predicted_change > 0.0 {
                correct_directions += 1;
            } else if actual_change.abs() < 1e-10 && predicted_change.abs() < 1e-10 {
                // Both no change - count as correct
                correct_directions += 1;
            }
        }
    }

    if total_comparisons == 0 {
        return 0.0;
    }

    (correct_directions as f64 / total_comparisons as f64) * 100.0
}

/// Maximum Error
///
/// Calculates the maximum absolute difference between actual and predicted values.
/// max|y_i - ŷ_i|
///
/// This metric is useful for identifying worst-case forecast errors.
///
/// # Arguments
/// * `actual` - The actual observed values
/// * `predicted` - The predicted values
///
/// # Returns
/// Maximum absolute error
pub fn max_error(actual: &TimeSeries, predicted: &TimeSeries) -> f64 {
    let actual_data = actual.values.to_vec().unwrap_or_default();
    let predicted_data = predicted.values.to_vec().unwrap_or_default();

    if actual_data.len() != predicted_data.len() || actual_data.is_empty() {
        return 0.0;
    }

    actual_data
        .iter()
        .zip(predicted_data.iter())
        .map(|(&a, &p)| ((a - p) as f64).abs())
        .fold(0.0f64, f64::max)
}

/// Forecast evaluation metrics bundle
#[derive(Debug, Clone)]
pub struct ForecastMetrics {
    pub mae: f64,
    pub mse: f64,
    pub rmse: f64,
    pub mape: f64,
    pub smape: f64,
    pub r2: f64,
    pub mase: f64,
    pub theil_u: f64,
    pub directional_accuracy: f64,
    pub max_error: f64,
}

/// Calculate all standard forecast evaluation metrics
pub fn evaluate_forecast(
    actual: &TimeSeries,
    predicted: &TimeSeries,
    seasonal_period: Option<usize>,
) -> ForecastMetrics {
    let seasonal_period = seasonal_period.unwrap_or(1);

    ForecastMetrics {
        mae: mae(actual, predicted),
        mse: mse(actual, predicted),
        rmse: rmse(actual, predicted),
        mape: mape(actual, predicted),
        smape: smape(actual, predicted),
        r2: r2(actual, predicted),
        mase: mase(actual, predicted, seasonal_period),
        theil_u: theil_u(actual, predicted),
        directional_accuracy: directional_accuracy(actual, predicted),
        max_error: max_error(actual, predicted),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::Tensor;

    fn create_test_series(data: Vec<f32>) -> TimeSeries {
        let len = data.len();
        let tensor = Tensor::from_vec(data, &[len]).unwrap();
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_mae() {
        let actual = create_test_series(vec![1.0, 2.0, 3.0]);
        let predicted = create_test_series(vec![1.1, 2.1, 2.9]);

        let error = mae(&actual, &predicted);
        // Expected: (|1.0-1.1| + |2.0-2.1| + |3.0-2.9|) / 3 = (0.1 + 0.1 + 0.1) / 3 = 0.1
        assert!((error - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_mse() {
        let actual = create_test_series(vec![1.0, 2.0, 3.0]);
        let predicted = create_test_series(vec![1.1, 2.1, 2.9]);

        let error = mse(&actual, &predicted);
        // Expected: ((1.0-1.1)² + (2.0-2.1)² + (3.0-2.9)²) / 3 = (0.01 + 0.01 + 0.01) / 3 = 0.01
        assert!((error - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_rmse() {
        let actual = create_test_series(vec![1.0, 2.0, 3.0]);
        let predicted = create_test_series(vec![1.1, 2.1, 2.9]);

        let error = rmse(&actual, &predicted);
        // Expected: √0.01 = 0.1
        assert!((error - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_r2() {
        let actual = create_test_series(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let predicted = create_test_series(vec![1.1, 2.0, 2.9, 4.1, 4.9]);

        let r2_val = r2(&actual, &predicted);
        // Should be close to 1.0 for good predictions
        assert!(r2_val > 0.95);
    }

    #[test]
    fn test_max_error() {
        let actual = create_test_series(vec![1.0, 2.0, 3.0, 4.0]);
        let predicted = create_test_series(vec![1.5, 2.1, 2.9, 3.8]);

        let max_err = max_error(&actual, &predicted);
        // Maximum error is |1.0 - 1.5| = 0.5
        assert!((max_err - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_directional_accuracy() {
        // Both series increasing
        let actual = create_test_series(vec![1.0, 2.0, 3.0, 4.0]);
        let predicted = create_test_series(vec![1.0, 2.1, 3.2, 4.3]);

        let da = directional_accuracy(&actual, &predicted);
        // All directions correct (3 out of 3 changes)
        assert_eq!(da, 100.0);
    }

    #[test]
    fn test_smape() {
        let actual = create_test_series(vec![100.0, 200.0, 300.0]);
        let predicted = create_test_series(vec![110.0, 190.0, 310.0]);

        let smape_val = smape(&actual, &predicted);
        // Should be small percentage for close predictions
        assert!(smape_val > 0.0 && smape_val < 10.0);
    }

    #[test]
    fn test_evaluate_forecast() {
        let actual = create_test_series(vec![1.0, 2.0, 3.0, 4.0]);
        let predicted = create_test_series(vec![1.1, 2.1, 2.9, 3.8]);

        let metrics = evaluate_forecast(&actual, &predicted, Some(2));

        // Verify all metrics are computed
        assert!(metrics.mae > 0.0);
        assert!(metrics.mse > 0.0);
        assert!(metrics.rmse > 0.0);
        assert!(metrics.r2 > 0.9); // Good R² for close predictions
        assert!(metrics.max_error > 0.0);
    }
}
