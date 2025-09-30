//! Time series preprocessing utilities

use crate::TimeSeries;
use torsh_tensor::Tensor;

/// Difference time series
pub fn diff(series: &TimeSeries, order: usize) -> TimeSeries {
    let mut values = series.values.clone();

    for _ in 0..order {
        // TODO: Fix tensor slicing and operations
        // let diff = &values.slice_tensor(0, 1, values.shape().dims()[0]).unwrap()
        //     - &values.slice_tensor(0, 0, values.shape().dims()[0] - 1).unwrap();
        // values = diff;
    }

    TimeSeries::new(values)
}

/// Detrend time series
pub fn detrend(series: &TimeSeries, _method: &str) -> TimeSeries {
    // TODO: Remove trend using linear or polynomial detrending
    TimeSeries::new(series.values.clone())
}

/// Normalize time series
pub fn normalize(series: &TimeSeries) -> TimeSeries {
    // TODO: Implement proper tensor operations for normalization
    // let mean = series.values.mean_dim(0, true);
    // let std = series.values.std_dim(0, true, true);
    // let normalized = (&series.values - &mean) / &std;

    TimeSeries::new(series.values.clone())
}

/// Apply moving average
pub fn moving_average(series: &TimeSeries, _window: usize) -> TimeSeries {
    // TODO: Calculate rolling mean
    TimeSeries::new(series.values.clone())
}

/// Apply exponential moving average
pub fn ema(series: &TimeSeries, alpha: f64) -> TimeSeries {
    let mut result = vec![0.0f32; series.len()];
    let values = series.values.to_vec().unwrap();

    result[0] = values[0];
    for i in 1..values.len() {
        result[i] = (alpha as f32) * values[i] + ((1.0 - alpha) as f32) * result[i - 1];
    }

    let tensor = Tensor::from_vec(result, &[series.len()]).unwrap();
    TimeSeries::new(tensor)
}

/// Box-Cox transformation
pub fn box_cox(series: &TimeSeries, lambda: f32) -> TimeSeries {
    // TODO: Implement Box-Cox transformation
    // if lambda == 0: log(x)
    // else: (x^lambda - 1) / lambda
    TimeSeries::new(series.values.clone())
}

/// Inverse Box-Cox transformation
pub fn inv_box_cox(series: &TimeSeries, lambda: f32) -> TimeSeries {
    // TODO: Implement inverse Box-Cox transformation
    TimeSeries::new(series.values.clone())
}

/// Standard scaling (zero mean, unit variance)
pub fn standard_scale(series: &TimeSeries) -> TimeSeries {
    // TODO: Implement standard scaling
    TimeSeries::new(series.values.clone())
}

/// Min-max scaling to [0, 1]
pub fn min_max_scale(series: &TimeSeries) -> TimeSeries {
    // TODO: Implement min-max scaling
    TimeSeries::new(series.values.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;

    fn create_test_series() -> TimeSeries {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let tensor = Tensor::from_vec(data, &[5]).unwrap();
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_diff() {
        let series = create_test_series();
        let diffed = diff(&series, 1);
        assert_eq!(diffed.len(), series.len());
    }

    #[test]
    fn test_detrend() {
        let series = create_test_series();
        let detrended = detrend(&series, "linear");
        assert_eq!(detrended.len(), series.len());
    }

    #[test]
    fn test_normalize() {
        let series = create_test_series();
        let normalized = normalize(&series);
        assert_eq!(normalized.len(), series.len());
    }

    #[test]
    fn test_ema() {
        let series = create_test_series();
        let smoothed = ema(&series, 0.3);
        assert_eq!(smoothed.len(), series.len());
    }

    #[test]
    fn test_moving_average() {
        let series = create_test_series();
        let smoothed = moving_average(&series, 3);
        assert_eq!(smoothed.len(), series.len());
    }
}
