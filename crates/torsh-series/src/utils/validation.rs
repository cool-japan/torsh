//! Time series validation and cross-validation methods
//!
//! This module provides comprehensive cross-validation strategies for time series:
//! - Time Series CV: Standard expanding window approach
//! - Blocked CV: Non-overlapping blocks for dependent data
//! - Purged CV: Financial time series with embargo periods
//! - Nested CV: Hyperparameter tuning with proper time series splits
//! - Combinatorial Purged CV (CPCV): Advanced method for overlapping labels

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

/// Purged cross-validation for financial time series
///
/// Implements the purging and embargo techniques from "Advances in Financial Machine Learning"
/// by Marcos López de Prado to prevent lookahead bias and information leakage.
///
/// # Features
/// - Purging: Remove samples from training set that overlap with test set
/// - Embargo: Add buffer period after test set to account for serial correlation
pub struct PurgedTimeSeriesCV {
    n_splits: usize,
    test_size: usize,
    purge_window: usize,
    embargo_window: usize,
}

impl PurgedTimeSeriesCV {
    /// Create a new purged cross-validator
    ///
    /// # Arguments
    /// * `n_splits` - Number of train/test splits
    /// * `test_size` - Size of each test fold
    /// * `purge_window` - Number of samples to purge before test set
    /// * `embargo_window` - Number of samples to embargo after test set
    pub fn new(
        n_splits: usize,
        test_size: usize,
        purge_window: usize,
        embargo_window: usize,
    ) -> Self {
        Self {
            n_splits,
            test_size,
            purge_window,
            embargo_window,
        }
    }

    /// Generate purged splits
    pub fn split(
        &self,
        series: &TimeSeries,
    ) -> Result<Vec<(TimeSeries, TimeSeries)>, torsh_core::error::TorshError> {
        let mut splits = Vec::new();
        let n = series.len();
        let split_size = n / self.n_splits;

        for i in 0..self.n_splits {
            let test_start = i * split_size;
            let test_end = (test_start + self.test_size).min(n);

            if test_end >= n {
                break;
            }

            // Purge samples before test set
            let purge_start = test_start.saturating_sub(self.purge_window);

            // Embargo samples after test set
            let embargo_end = (test_end + self.embargo_window).min(n);

            // Create training set excluding purged and embargoed regions
            let train_before_end = purge_start;
            let train_after_start = embargo_end;

            if train_before_end > 0 {
                let train = series.slice(0, train_before_end)?;
                let test = series.slice(test_start, test_end)?;
                splits.push((train, test));
            } else if train_after_start < n {
                let train = series.slice(train_after_start, n)?;
                let test = series.slice(test_start, test_end)?;
                splits.push((train, test));
            }
        }

        Ok(splits)
    }
}

/// Combinatorial Purged Cross-Validation (CPCV)
///
/// An advanced technique for financial time series that generates multiple
/// paths through the data, each satisfying the purging and embargo constraints.
/// This provides better coverage than standard purged CV.
///
/// Reference: "Advances in Financial Machine Learning" Chapter 12
pub struct CombinatorialPurgedCV {
    n_paths: usize,
    test_size: usize,
    purge_window: usize,
    embargo_window: usize,
}

impl CombinatorialPurgedCV {
    /// Create a new combinatorial purged cross-validator
    pub fn new(
        n_paths: usize,
        test_size: usize,
        purge_window: usize,
        embargo_window: usize,
    ) -> Self {
        Self {
            n_paths,
            test_size,
            purge_window,
            embargo_window,
        }
    }

    /// Generate combinatorial purged splits
    pub fn split(
        &self,
        series: &TimeSeries,
    ) -> Result<Vec<Vec<(TimeSeries, TimeSeries)>>, torsh_core::error::TorshError> {
        let n = series.len();
        let mut all_paths = Vec::new();

        // Generate different paths by varying the starting point
        for path_idx in 0..self.n_paths {
            let mut path_splits = Vec::new();
            let offset = (path_idx * n) / (self.n_paths * 3); // Stagger starting points

            let mut current_pos = offset;
            while current_pos + self.test_size < n {
                let test_start = current_pos;
                let test_end = current_pos + self.test_size;

                // Apply purging and embargo
                let purge_start = test_start.saturating_sub(self.purge_window);
                let embargo_end = (test_end + self.embargo_window).min(n);

                // Training data before test
                if purge_start > offset {
                    let train = series.slice(offset, purge_start)?;
                    let test = series.slice(test_start, test_end)?;
                    path_splits.push((train, test));
                }

                // Move to next position
                current_pos = embargo_end;
            }

            if !path_splits.is_empty() {
                all_paths.push(path_splits);
            }
        }

        Ok(all_paths)
    }
}

/// Nested cross-validation for hyperparameter tuning
///
/// Implements nested CV where:
/// - Outer loop: Model evaluation
/// - Inner loop: Hyperparameter selection
///
/// This prevents overfitting to the validation set during hyperparameter tuning.
pub struct NestedTimeSeriesCV {
    n_outer_splits: usize,
    n_inner_splits: usize,
    gap: usize,
}

impl NestedTimeSeriesCV {
    /// Create a new nested cross-validator
    pub fn new(n_outer_splits: usize, n_inner_splits: usize) -> Self {
        Self {
            n_outer_splits,
            n_inner_splits,
            gap: 0,
        }
    }

    /// Set gap between train and test
    pub fn with_gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }

    /// Generate nested splits
    ///
    /// Returns: Vec<(outer_train, outer_test, Vec<(inner_train, inner_val)>)>
    pub fn split(
        &self,
        series: &TimeSeries,
    ) -> Result<
        Vec<(TimeSeries, TimeSeries, Vec<(TimeSeries, TimeSeries)>)>,
        torsh_core::error::TorshError,
    > {
        let mut nested_splits = Vec::new();
        let n = series.len();
        let outer_test_size = n / (self.n_outer_splits + 1);

        for i in 1..=self.n_outer_splits {
            let outer_train_end = outer_test_size * i;
            let outer_test_start = outer_train_end + self.gap;
            let outer_test_end = outer_test_start + outer_test_size;

            if outer_test_end <= n {
                let outer_train = series.slice(0, outer_train_end)?;
                let outer_test = series.slice(outer_test_start, outer_test_end)?;

                // Generate inner splits for hyperparameter tuning
                let inner_cv = TimeSeriesCV::new(self.n_inner_splits).with_gap(self.gap);
                let inner_splits = inner_cv.split(&outer_train)?;

                nested_splits.push((outer_train, outer_test, inner_splits));
            }
        }

        Ok(nested_splits)
    }
}

/// Cross-validation with custom scoring function
pub struct ScoredTimeSeriesCV<F>
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    cv: TimeSeriesCV,
    scorer: F,
}

impl<F> ScoredTimeSeriesCV<F>
where
    F: Fn(&[f64], &[f64]) -> f64,
{
    /// Create a new scored cross-validator
    pub fn new(n_splits: usize, scorer: F) -> Self {
        Self {
            cv: TimeSeriesCV::new(n_splits),
            scorer,
        }
    }

    /// Evaluate model using cross-validation with scoring
    pub fn evaluate<M>(
        &self,
        series: &TimeSeries,
        mut model: M,
    ) -> Result<Vec<f64>, torsh_core::error::TorshError>
    where
        M: FnMut(
            &TimeSeries,
            &TimeSeries,
        ) -> Result<(Vec<f64>, Vec<f64>), torsh_core::error::TorshError>,
    {
        let splits = self.cv.split(series)?;
        let mut scores = Vec::with_capacity(splits.len());

        for (train, test) in splits {
            let (predictions, actuals) = model(&train, &test)?;
            let score = (self.scorer)(&predictions, &actuals);
            scores.push(score);
        }

        Ok(scores)
    }
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
    use torsh_tensor::Tensor;

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

    #[test]
    fn test_purged_cv() {
        let series = create_test_series();
        let cv = PurgedTimeSeriesCV::new(2, 2, 1, 1);
        let splits = cv.split(&series).unwrap();

        // Should have at least one split
        assert!(!splits.is_empty());

        for (train, test) in splits {
            assert!(train.len() > 0);
            assert_eq!(test.len(), 2);
        }
    }

    #[test]
    fn test_combinatorial_purged_cv() {
        let series = create_test_series();
        let cv = CombinatorialPurgedCV::new(2, 2, 1, 1);
        let all_paths = cv.split(&series).unwrap();

        // Should generate at least one path
        assert!(!all_paths.is_empty());

        for path in all_paths {
            assert!(!path.is_empty());
        }
    }

    #[test]
    fn test_nested_cv() {
        let series = create_test_series();
        let cv = NestedTimeSeriesCV::new(2, 2);
        let nested_splits = cv.split(&series).unwrap();

        assert!(!nested_splits.is_empty());

        for (outer_train, outer_test, inner_splits) in nested_splits {
            assert!(outer_train.len() > 0);
            assert!(outer_test.len() > 0);
            assert!(!inner_splits.is_empty());

            for (inner_train, inner_val) in inner_splits {
                assert!(inner_train.len() > 0);
                assert!(inner_val.len() > 0);
            }
        }
    }

    #[test]
    fn test_scored_cv() {
        let series = create_test_series();
        let scorer = |predictions: &[f64], actuals: &[f64]| {
            // Simple MSE scorer
            predictions
                .iter()
                .zip(actuals.iter())
                .map(|(p, a)| (p - a).powi(2))
                .sum::<f64>()
                / predictions.len() as f64
        };

        let cv = ScoredTimeSeriesCV::new(2, scorer);

        let model = |_train: &TimeSeries, test: &TimeSeries| {
            let predictions = vec![5.0; test.len()];
            let mut actuals = Vec::new();
            for i in 0..test.len() {
                actuals.push(test.values.get_item_flat(i).unwrap() as f64);
            }
            Ok((predictions, actuals))
        };

        let scores = cv.evaluate(&series, model).unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores.iter().all(|&s| s >= 0.0));
    }
}
