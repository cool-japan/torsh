//! Time series validation and cross-validation methods

use crate::TimeSeries;

/// Time series cross-validation
pub struct TimeSeriesCV {
    n_splits: usize,
    max_train_size: Option<usize>,
    gap: usize,
}

impl TimeSeriesCV {
    /// Create a new time series cross-validator
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            max_train_size: None,
            gap: 0,
        }
    }

    /// Set maximum training size
    pub fn with_max_train_size(mut self, max_train_size: usize) -> Self {
        self.max_train_size = Some(max_train_size);
        self
    }

    /// Set gap between train and test
    pub fn with_gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }

    /// Generate train/test splits
    pub fn split(
        &self,
        series: &TimeSeries,
    ) -> Result<Vec<(TimeSeries, TimeSeries)>, torsh_core::error::TorshError> {
        let mut splits = Vec::new();
        let n = series.len();
        let test_size = n / (self.n_splits + 1);

        for i in 1..=self.n_splits {
            let train_end = test_size * i;
            let test_start = train_end + self.gap;
            let test_end = test_start + test_size;

            if test_end <= n {
                // Apply max_train_size if set
                let train_start = if let Some(max_size) = self.max_train_size {
                    (train_end).saturating_sub(max_size)
                } else {
                    0
                };

                let train = series.slice(train_start, train_end)?;
                let test = series.slice(test_start, test_end)?;
                splits.push((train, test));
            }
        }

        Ok(splits)
    }
}

/// Walk-forward validation
pub fn walk_forward_validation<F>(
    series: &TimeSeries,
    window_size: usize,
    step_size: usize,
    predict_fn: F,
) -> Result<Vec<f64>, torsh_core::error::TorshError>
where
    F: Fn(&TimeSeries) -> f64,
{
    let mut errors = Vec::new();
    let n = series.len();

    for i in (window_size..n).step_by(step_size) {
        let train = series.slice(i - window_size, i)?;
        let pred = predict_fn(&train);
        let actual = series.values.get_item_flat(i)? as f64;
        errors.push((pred - actual).abs());
    }

    Ok(errors)
}

/// Expanding window validation
pub fn expanding_window_validation<F>(
    series: &TimeSeries,
    min_train_size: usize,
    test_size: usize,
    predict_fn: F,
) -> Result<Vec<f64>, torsh_core::error::TorshError>
where
    F: Fn(&TimeSeries) -> f64,
{
    let mut errors = Vec::new();
    let n = series.len();

    if min_train_size + test_size > n {
        return Ok(errors);
    }

    for i in (min_train_size..=(n - test_size)).step_by(test_size) {
        let train = series.slice(0, i)?;
        let test_start = i;
        let test_end = (i + test_size).min(n);

        for j in test_start..test_end {
            let pred = predict_fn(&train);
            let actual = series.values.get_item_flat(j)? as f64;
            errors.push((pred - actual).abs());
        }
    }

    Ok(errors)
}

/// Rolling window validation with fixed window size
pub fn rolling_window_validation<F>(
    series: &TimeSeries,
    window_size: usize,
    test_size: usize,
    predict_fn: F,
) -> Result<Vec<f64>, torsh_core::error::TorshError>
where
    F: Fn(&TimeSeries) -> f64,
{
    let mut errors = Vec::new();
    let n = series.len();

    if window_size + test_size > n {
        return Ok(errors);
    }

    for i in window_size..(n - test_size + 1) {
        let train = series.slice(i - window_size, i)?;
        let test_start = i;
        let test_end = (i + test_size).min(n);

        for j in test_start..test_end {
            let pred = predict_fn(&train);
            let actual = series.values.get_item_flat(j)? as f64;
            errors.push((pred - actual).abs());
        }
    }

    Ok(errors)
}

/// Blocked cross-validation for time series
pub struct BlockedTimeSeriesCV {
    n_splits: usize,
    block_size: usize,
    gap: usize,
}

impl BlockedTimeSeriesCV {
    /// Create a new blocked cross-validator
    pub fn new(n_splits: usize, block_size: usize) -> Self {
        Self {
            n_splits,
            block_size,
            gap: 0,
        }
    }

    /// Set gap between blocks
    pub fn with_gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }

    /// Generate blocked splits
    pub fn split(
        &self,
        series: &TimeSeries,
    ) -> Result<Vec<(TimeSeries, TimeSeries)>, torsh_core::error::TorshError> {
        let mut splits = Vec::new();
        let n = series.len();
        let total_block_size = self.block_size + self.gap;

        for i in 0..self.n_splits {
            let test_start = i * total_block_size;
            let test_end = (test_start + self.block_size).min(n);

            if test_end <= n {
                // Create train set excluding the test block and gap
                let mut _train_indices = Vec::new();

                // Add indices before the test block (with gap)
                let train_end_before = test_start.saturating_sub(self.gap);
                _train_indices.extend(0..train_end_before);

                // Add indices after the test block (with gap)
                let train_start_after = (test_end + self.gap).min(n);
                if train_start_after < n {
                    _train_indices.extend(train_start_after..n);
                }

                if !_train_indices.is_empty() {
                    // For simplicity, use the first contiguous segment as training
                    // In practice, you might want to combine non-contiguous segments
                    let train = if train_end_before > 0 {
                        series.slice(0, train_end_before)?
                    } else {
                        series.slice(train_start_after, n)?
                    };
                    let test = series.slice(test_start, test_end)?;
                    splits.push((train, test));
                }
            }
        }

        Ok(splits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::{creation::*, Tensor};

    fn create_test_series() -> TimeSeries {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let tensor = Tensor::from_vec(data, &[10]).unwrap();
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_timeseries_cv() {
        let series = create_test_series();
        let cv = TimeSeriesCV::new(3);
        let splits = cv.split(&series).unwrap();

        assert!(!splits.is_empty());
        // Each split should have train and test sets
        for (train, test) in splits {
            assert!(train.len() > 0);
            assert!(test.len() > 0);
        }
    }

    #[test]
    fn test_timeseries_cv_with_gap() {
        let series = create_test_series();
        let cv = TimeSeriesCV::new(2).with_gap(1);
        let splits = cv.split(&series).unwrap();

        assert!(!splits.is_empty());
    }

    #[test]
    fn test_walk_forward_validation() {
        let series = create_test_series();
        let errors = walk_forward_validation(&series, 3, 1, |_| 5.0).unwrap();

        assert!(!errors.is_empty());
    }

    #[test]
    fn test_blocked_timeseries_cv() {
        let series = create_test_series();
        let cv = BlockedTimeSeriesCV::new(2, 3);
        let splits = cv.split(&series).unwrap();

        assert!(!splits.is_empty());
    }

    #[test]
    fn test_expanding_window_validation() {
        let series = create_test_series();
        let errors = expanding_window_validation(&series, 3, 2, |_| 5.0).unwrap();

        assert!(!errors.is_empty());
    }
}
