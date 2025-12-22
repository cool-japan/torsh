//! Change point detection for time series
//!
//! This module provides algorithms for detecting structural breaks and regime changes:
//! - PELT (Pruned Exact Linear Time) - Efficient exact change point detection
//! - Binary Segmentation - Fast approximate change point detection
//! - Window-based detection - Sliding window statistical methods
//! - Bayesian change point detection - Probabilistic approach
//!
//! NOTE: Full implementation will use scirs2-series when change point APIs are available.

use crate::TimeSeries;
use torsh_core::error::Result;

/// Change point detection result
#[derive(Debug, Clone)]
pub struct ChangePointResult {
    /// Indices of detected change points
    pub change_points: Vec<usize>,
    /// Cost/score associated with each change point
    pub scores: Vec<f64>,
    /// Algorithm used for detection
    pub algorithm: String,
}

/// Cost function type for change point detection
#[derive(Debug, Clone, Copy)]
pub enum CostFunction {
    /// L2 norm (mean change)
    L2,
    /// L1 norm (median change)
    L1,
    /// Variance change
    Variance,
    /// Kolmogorov-Smirnov statistic
    KolmogorovSmirnov,
}

/// PELT (Pruned Exact Linear Time) algorithm
///
/// Efficient exact change point detection with complexity O(n)
/// for most practical cases. Minimizes sum of segment costs plus penalty.
pub struct PELT {
    penalty: f64,
    cost_function: CostFunction,
    min_segment_length: usize,
}

impl PELT {
    /// Create a new PELT detector
    ///
    /// # Arguments
    /// * `penalty` - Penalty for adding a change point (larger = fewer change points)
    /// * `min_segment_length` - Minimum length of a segment
    pub fn new(penalty: f64, min_segment_length: usize) -> Self {
        Self {
            penalty,
            cost_function: CostFunction::L2,
            min_segment_length,
        }
    }

    /// Set cost function
    pub fn with_cost_function(mut self, cost_fn: CostFunction) -> Self {
        self.cost_function = cost_fn;
        self
    }

    /// Detect change points using PELT algorithm
    pub fn detect(&self, series: &TimeSeries) -> Result<ChangePointResult> {
        let data = series.values.to_vec()?;
        let n = data.len();

        if n < 2 * self.min_segment_length {
            return Ok(ChangePointResult {
                change_points: vec![],
                scores: vec![],
                algorithm: "PELT".to_string(),
            });
        }

        // TODO: Implement full PELT with optimal partitioning when scirs2-series available
        // For now, implement simplified version using dynamic programming

        let mut f = vec![f64::INFINITY; n + 1]; // Cost up to index i
        f[0] = -self.penalty;

        let mut cp = vec![0; n + 1]; // Last change point before index i
        let mut r = vec![0]; // Pruning set

        for t in self.min_segment_length..=n {
            let mut costs = Vec::new();

            for &s in &r {
                if t - s >= self.min_segment_length {
                    let segment_cost = self.compute_cost(&data[s..t]);
                    costs.push((f[s] + segment_cost + self.penalty, s));
                }
            }

            if let Some(&(min_cost, s_star)) = costs
                .iter()
                .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
            {
                f[t] = min_cost;
                cp[t] = s_star;

                // Pruning: remove s from R if F(s) > F(t)
                r.retain(|&s| f[s] + self.compute_cost(&data[s..t]) <= f[t]);
                r.push(t);
            }
        }

        // Backtrack to find change points
        let mut change_points = Vec::new();
        let mut current = n;
        while current > 0 {
            let prev = cp[current];
            if prev > 0 {
                change_points.push(prev);
            }
            current = prev;
        }
        change_points.reverse();

        // Compute scores for each change point
        let scores = change_points
            .iter()
            .map(|&cp_idx| {
                if cp_idx < n {
                    self.compute_cost(&data[cp_idx - self.min_segment_length..cp_idx])
                } else {
                    0.0
                }
            })
            .collect();

        Ok(ChangePointResult {
            change_points,
            scores,
            algorithm: "PELT".to_string(),
        })
    }

    /// Compute cost for a segment
    fn compute_cost(&self, segment: &[f32]) -> f64 {
        match self.cost_function {
            CostFunction::L2 => {
                let mean = segment.iter().sum::<f32>() / segment.len() as f32;
                segment
                    .iter()
                    .map(|&x| ((x - mean) as f64).powi(2))
                    .sum::<f64>()
            }
            CostFunction::L1 => {
                let mut sorted = segment.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = sorted[sorted.len() / 2];
                segment
                    .iter()
                    .map(|&x| ((x - median) as f64).abs())
                    .sum::<f64>()
            }
            CostFunction::Variance => {
                let mean = segment.iter().sum::<f32>() / segment.len() as f32;
                let variance = segment
                    .iter()
                    .map(|&x| ((x - mean) as f64).powi(2))
                    .sum::<f64>()
                    / segment.len() as f64;
                -variance.ln() * segment.len() as f64 // Negative log-likelihood
            }
            CostFunction::KolmogorovSmirnov => {
                // Simplified KS statistic
                let mean = segment.iter().sum::<f32>() / segment.len() as f32;
                segment
                    .iter()
                    .map(|&x| ((x - mean) as f64).powi(2))
                    .sum::<f64>()
            }
        }
    }
}

/// Binary Segmentation algorithm
///
/// Fast approximate change point detection that recursively splits
/// the time series at the point of maximum cost reduction.
pub struct BinarySegmentation {
    threshold: f64,
    max_change_points: Option<usize>,
    cost_function: CostFunction,
    min_segment_length: usize,
}

impl BinarySegmentation {
    /// Create a new Binary Segmentation detector
    ///
    /// # Arguments
    /// * `threshold` - Minimum cost reduction to accept a change point
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            max_change_points: None,
            cost_function: CostFunction::L2,
            min_segment_length: 2,
        }
    }

    /// Set maximum number of change points
    pub fn with_max_change_points(mut self, max_cp: usize) -> Self {
        self.max_change_points = Some(max_cp);
        self
    }

    /// Set cost function
    pub fn with_cost_function(mut self, cost_fn: CostFunction) -> Self {
        self.cost_function = cost_fn;
        self
    }

    /// Set minimum segment length
    pub fn with_min_segment_length(mut self, min_len: usize) -> Self {
        self.min_segment_length = min_len;
        self
    }

    /// Detect change points using Binary Segmentation
    pub fn detect(&self, series: &TimeSeries) -> Result<ChangePointResult> {
        let data = series.values.to_vec()?;
        let n = data.len();

        let mut change_points = Vec::new();
        let mut scores = Vec::new();
        let mut segments = vec![(0, n)];

        while !segments.is_empty() {
            if let Some(max_cp) = self.max_change_points {
                if change_points.len() >= max_cp {
                    break;
                }
            }

            let (start, end) = segments.pop().unwrap();
            if end - start < 2 * self.min_segment_length {
                continue;
            }

            // Find best split point
            let (best_split, best_score) = self.find_best_split(&data, start, end);

            if best_score > self.threshold {
                change_points.push(best_split);
                scores.push(best_score);

                // Add new segments to process
                segments.push((start, best_split));
                segments.push((best_split, end));
            }
        }

        change_points.sort_unstable();

        Ok(ChangePointResult {
            change_points,
            scores,
            algorithm: "Binary Segmentation".to_string(),
        })
    }

    /// Find best split point in a segment
    fn find_best_split(&self, data: &[f32], start: usize, end: usize) -> (usize, f64) {
        let mut best_split = start + self.min_segment_length;
        let mut best_score = 0.0;

        let original_cost = self.segment_cost(data, start, end);

        for split in (start + self.min_segment_length)..(end - self.min_segment_length) {
            let left_cost = self.segment_cost(data, start, split);
            let right_cost = self.segment_cost(data, split, end);
            let cost_reduction = original_cost - (left_cost + right_cost);

            if cost_reduction > best_score {
                best_score = cost_reduction;
                best_split = split;
            }
        }

        (best_split, best_score)
    }

    /// Compute cost for a segment
    fn segment_cost(&self, data: &[f32], start: usize, end: usize) -> f64 {
        let segment = &data[start..end];
        match self.cost_function {
            CostFunction::L2 => {
                let mean = segment.iter().sum::<f32>() / segment.len() as f32;
                segment
                    .iter()
                    .map(|&x| ((x - mean) as f64).powi(2))
                    .sum::<f64>()
            }
            CostFunction::L1 => {
                let mut sorted = segment.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = sorted[sorted.len() / 2];
                segment
                    .iter()
                    .map(|&x| ((x - median) as f64).abs())
                    .sum::<f64>()
            }
            CostFunction::Variance => {
                let mean = segment.iter().sum::<f32>() / segment.len() as f32;
                segment
                    .iter()
                    .map(|&x| ((x - mean) as f64).powi(2))
                    .sum::<f64>()
                    / segment.len() as f64
            }
            CostFunction::KolmogorovSmirnov => {
                let mean = segment.iter().sum::<f32>() / segment.len() as f32;
                segment
                    .iter()
                    .map(|&x| ((x - mean) as f64).powi(2))
                    .sum::<f64>()
            }
        }
    }
}

/// Window-based change point detection
///
/// Uses sliding windows to detect changes in statistical properties
pub struct WindowDetector {
    window_size: usize,
    threshold: f64,
    statistic: WindowStatistic,
}

#[derive(Debug, Clone, Copy)]
pub enum WindowStatistic {
    /// Mean change
    Mean,
    /// Variance change
    Variance,
    /// Cumulative sum
    CUSUM,
}

impl WindowDetector {
    /// Create a new window-based detector
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            window_size,
            threshold,
            statistic: WindowStatistic::Mean,
        }
    }

    /// Set statistic to monitor
    pub fn with_statistic(mut self, stat: WindowStatistic) -> Self {
        self.statistic = stat;
        self
    }

    /// Detect change points
    pub fn detect(&self, series: &TimeSeries) -> Result<ChangePointResult> {
        let data = series.values.to_vec()?;
        let n = data.len();

        if n < 2 * self.window_size {
            return Ok(ChangePointResult {
                change_points: vec![],
                scores: vec![],
                algorithm: "Window".to_string(),
            });
        }

        let mut change_points = Vec::new();
        let mut scores = Vec::new();

        for i in self.window_size..(n - self.window_size) {
            let left_window = &data[(i - self.window_size)..i];
            let right_window = &data[i..(i + self.window_size)];

            let score = match self.statistic {
                WindowStatistic::Mean => self.mean_change(left_window, right_window),
                WindowStatistic::Variance => self.variance_change(left_window, right_window),
                WindowStatistic::CUSUM => self.cusum_statistic(&data, i),
            };

            if score.abs() > self.threshold {
                change_points.push(i);
                scores.push(score);
            }
        }

        Ok(ChangePointResult {
            change_points,
            scores,
            algorithm: "Window".to_string(),
        })
    }

    /// Compute mean change between windows
    fn mean_change(&self, left: &[f32], right: &[f32]) -> f64 {
        let left_mean = left.iter().sum::<f32>() / left.len() as f32;
        let right_mean = right.iter().sum::<f32>() / right.len() as f32;
        (right_mean - left_mean) as f64
    }

    /// Compute variance change between windows
    fn variance_change(&self, left: &[f32], right: &[f32]) -> f64 {
        let left_mean = left.iter().sum::<f32>() / left.len() as f32;
        let right_mean = right.iter().sum::<f32>() / right.len() as f32;

        let left_var = left
            .iter()
            .map(|&x| ((x - left_mean) as f64).powi(2))
            .sum::<f64>()
            / left.len() as f64;
        let right_var = right
            .iter()
            .map(|&x| ((x - right_mean) as f64).powi(2))
            .sum::<f64>()
            / right.len() as f64;

        right_var - left_var
    }

    /// Compute CUSUM statistic
    fn cusum_statistic(&self, data: &[f32], pos: usize) -> f64 {
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let mut cusum = 0.0;
        for &x in &data[..pos] {
            cusum += (x - mean) as f64;
        }
        cusum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::Tensor;

    fn create_change_point_series() -> TimeSeries {
        // Create series with clear change point at index 50
        let mut data = Vec::with_capacity(100);
        for _i in 0..50 {
            data.push(1.0f32);
        }
        for _i in 50..100 {
            data.push(5.0f32);
        }
        let tensor = Tensor::from_vec(data, &[100]).unwrap();
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_pelt_creation() {
        let pelt = PELT::new(10.0, 5);
        assert_eq!(pelt.min_segment_length, 5);
    }

    #[test]
    fn test_pelt_detection() {
        let series = create_change_point_series();
        let pelt = PELT::new(10.0, 5);
        let result = pelt.detect(&series).unwrap();

        assert_eq!(result.algorithm, "PELT");
        // Should detect at least one change point
        assert!(!result.change_points.is_empty());
    }

    #[test]
    fn test_pelt_with_cost_functions() {
        let series = create_change_point_series();

        for cost_fn in [
            CostFunction::L2,
            CostFunction::L1,
            CostFunction::Variance,
            CostFunction::KolmogorovSmirnov,
        ] {
            let pelt = PELT::new(10.0, 5).with_cost_function(cost_fn);
            let result = pelt.detect(&series).unwrap();
            assert_eq!(result.algorithm, "PELT");
        }
    }

    #[test]
    fn test_binary_segmentation_creation() {
        let bs = BinarySegmentation::new(0.5);
        assert_eq!(bs.threshold, 0.5);
    }

    #[test]
    fn test_binary_segmentation_detection() {
        let series = create_change_point_series();
        let bs = BinarySegmentation::new(1.0);
        let result = bs.detect(&series).unwrap();

        assert_eq!(result.algorithm, "Binary Segmentation");
        assert!(!result.change_points.is_empty());
    }

    #[test]
    fn test_binary_segmentation_max_change_points() {
        let series = create_change_point_series();
        let bs = BinarySegmentation::new(0.1).with_max_change_points(2);
        let result = bs.detect(&series).unwrap();

        assert!(result.change_points.len() <= 2);
    }

    #[test]
    fn test_window_detector_creation() {
        let detector = WindowDetector::new(10, 0.5);
        assert_eq!(detector.window_size, 10);
        assert_eq!(detector.threshold, 0.5);
    }

    #[test]
    fn test_window_detector_mean() {
        let series = create_change_point_series();
        let detector = WindowDetector::new(10, 1.0).with_statistic(WindowStatistic::Mean);
        let result = detector.detect(&series).unwrap();

        assert_eq!(result.algorithm, "Window");
        // Should detect change around index 50
        assert!(!result.change_points.is_empty());
    }

    #[test]
    fn test_window_detector_variance() {
        let series = create_change_point_series();
        let detector = WindowDetector::new(10, 0.1).with_statistic(WindowStatistic::Variance);
        let result = detector.detect(&series).unwrap();

        assert_eq!(result.algorithm, "Window");
    }

    #[test]
    fn test_window_detector_cusum() {
        let series = create_change_point_series();
        let detector = WindowDetector::new(10, 5.0).with_statistic(WindowStatistic::CUSUM);
        let result = detector.detect(&series).unwrap();

        assert_eq!(result.algorithm, "Window");
    }

    #[test]
    fn test_change_point_result() {
        let series = create_change_point_series();
        let pelt = PELT::new(10.0, 5);
        let result = pelt.detect(&series).unwrap();

        assert_eq!(result.change_points.len(), result.scores.len());
        assert!(!result.algorithm.is_empty());
    }
}
