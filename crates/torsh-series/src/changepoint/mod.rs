//! Change point detection for time series
//!
//! This module provides algorithms for detecting structural breaks and regime changes:
//! - PELT (Pruned Exact Linear Time) - Efficient exact change point detection
//! - Binary Segmentation - Fast approximate change point detection
//! - Window-based detection - Sliding window statistical methods
//! - Bayesian change point detection - Probabilistic approach

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

/// Cost function type for change point detection.
///
/// Each variant is a genuine additive segment cost suitable for optimal
/// partitioning. All are super-additive (splitting a segment never increases the
/// total cost), so PELT pruning with constant `K = 0` is exact.
#[derive(Debug, Clone, Copy)]
pub enum CostFunction {
    /// Gaussian mean-change cost: the sum of squared deviations about the segment
    /// mean (`Σ(xᵢ − x̄)²`). This is the negative log-likelihood for a change in
    /// mean with fixed variance, up to additive constants.
    L2,
    /// Laplace / median-change cost: the sum of absolute deviations about the
    /// segment median (`Σ|xᵢ − median|`), robust to outliers.
    L1,
    /// Gaussian variance-change cost: the `n·ln(σ̂²)` negative log-likelihood term
    /// where `σ̂²` is the MLE variance of the segment.
    Variance,
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

    /// Detect change points using the PELT algorithm.
    ///
    /// Implements exact optimal partitioning (Jackson et al. 2005) with the pruning
    /// step of Killick, Fearnhead & Eckley (2012). The dynamic program
    ///
    /// ```text
    /// F(s) = min_{t in R} [ F(t) + C(y_{t+1..s}) + beta ]
    /// ```
    ///
    /// is solved left-to-right where `C` is the selected segment cost, `beta` the
    /// penalty and `R` the set of admissible last-changepoints. After `F(s)` is
    /// computed, every candidate `t` with `F(t) + C(y_{t+1..s}) + K > F(s)` is pruned
    /// (`K = 0`, valid for the super-additive costs used here), which yields the same
    /// optimum as the un-pruned recursion at near-linear expected cost.
    pub fn detect(&self, series: &TimeSeries) -> Result<ChangePointResult> {
        let data = series.values.to_vec()?;
        Ok(self.run(&data, true))
    }

    /// Suggested BIC / SIC penalty for a mean-change model.
    ///
    /// On the raw sum-of-squared-errors cost scale, the Bayesian Information
    /// Criterion contributes `2·variance·ln(n)` per added changepoint (one extra
    /// mean parameter per new segment), where `variance` is the (assumed known)
    /// noise variance.
    pub fn bic_penalty(n: usize, variance: f64) -> f64 {
        2.0 * variance * (n.max(1) as f64).ln()
    }

    /// Core optimal-partitioning solver.
    ///
    /// When `prune` is `true` the PELT pruning step is applied (near-linear); when
    /// `false` every past candidate is considered (reference `O(n²)` optimal
    /// partitioning). For the super-additive costs in [`CostFunction`] both return
    /// the identical global optimum.
    fn run(&self, data: &[f32], prune: bool) -> ChangePointResult {
        let n = data.len();
        if n == 0 {
            return ChangePointResult {
                change_points: vec![],
                scores: vec![],
                algorithm: "PELT".to_string(),
            };
        }

        // Prefix sums (f64) give O(1) Gaussian segment costs.
        let mut prefix1 = vec![0.0f64; n + 1];
        let mut prefix2 = vec![0.0f64; n + 1];
        for i in 0..n {
            let x = data[i] as f64;
            prefix1[i + 1] = prefix1[i] + x;
            prefix2[i + 1] = prefix2[i] + x * x;
        }

        let beta = self.penalty;
        let min_len = self.min_segment_length.max(1);

        // F(s): optimal cost of segmenting data[0..s]; cp[s]: best last changepoint.
        let mut f = vec![f64::INFINITY; n + 1];
        f[0] = -beta;
        let mut cp = vec![0usize; n + 1];
        let mut candidates: Vec<usize> = vec![0];

        for s in 1..=n {
            let mut best = f64::INFINITY;
            let mut best_t = 0usize;

            if prune {
                for &t in &candidates {
                    if s - t < min_len || !f[t].is_finite() {
                        continue;
                    }
                    let total = f[t] + self.segment_cost(data, &prefix1, &prefix2, t, s) + beta;
                    if total < best {
                        best = total;
                        best_t = t;
                    }
                }
            } else {
                for t in 0..s {
                    if s - t < min_len || !f[t].is_finite() {
                        continue;
                    }
                    let total = f[t] + self.segment_cost(data, &prefix1, &prefix2, t, s) + beta;
                    if total < best {
                        best = total;
                        best_t = t;
                    }
                }
            }

            if best.is_finite() {
                f[s] = best;
                cp[s] = best_t;
            }

            if prune {
                if f[s].is_finite() {
                    // Keep `t` only if it can still be optimal for a future endpoint:
                    //   F(t) + C(y_{t+1..s}) <= F(s)   (pruning constant K = 0).
                    let f_s = f[s];
                    candidates.retain(|&t| {
                        f[t].is_finite()
                            && f[t] + self.segment_cost(data, &prefix1, &prefix2, t, s) <= f_s
                    });
                }
                candidates.push(s);
            }
        }

        // Backtrack the optimal segmentation, recording the cost of each segment.
        let mut change_points = Vec::new();
        let mut scores = Vec::new();
        let mut cur = n;
        while cur > 0 {
            let prev = cp[cur];
            if prev >= cur {
                break; // defensive: cp[cur] is always < cur for a valid path
            }
            if prev > 0 {
                change_points.push(prev);
                scores.push(self.segment_cost(data, &prefix1, &prefix2, prev, cur));
            }
            cur = prev;
        }
        change_points.reverse();
        scores.reverse();

        ChangePointResult {
            change_points,
            scores,
            algorithm: "PELT".to_string(),
        }
    }

    /// Cost of the segment `data[a..b]` under the configured cost function.
    ///
    /// Gaussian costs (`L2`, `Variance`) are evaluated in O(1) from the prefix
    /// sums; the robust `L1` cost is computed directly from the slice.
    fn segment_cost(
        &self,
        data: &[f32],
        prefix1: &[f64],
        prefix2: &[f64],
        a: usize,
        b: usize,
    ) -> f64 {
        let n_seg = (b - a) as f64;
        match self.cost_function {
            CostFunction::L2 => {
                let sum1 = prefix1[b] - prefix1[a];
                let sum2 = prefix2[b] - prefix2[a];
                // Sum of squared errors about the segment mean: Σx² − (Σx)²/n.
                (sum2 - sum1 * sum1 / n_seg).max(0.0)
            }
            CostFunction::Variance => {
                let sum1 = prefix1[b] - prefix1[a];
                let sum2 = prefix2[b] - prefix2[a];
                let sse = (sum2 - sum1 * sum1 / n_seg).max(0.0);
                // Gaussian variance-change NLL term n·ln(σ̂²); floor σ̂² to stay finite.
                let var = (sse / n_seg).max(1e-12);
                n_seg * var.ln()
            }
            CostFunction::L1 => {
                let seg = &data[a..b];
                let mut sorted: Vec<f32> = seg.to_vec();
                sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
                let median = sorted[sorted.len() / 2];
                seg.iter().map(|&x| ((x - median) as f64).abs()).sum()
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

            let (start, end) = segments.pop().expect("segments was checked non-empty");
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
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
        let tensor = Tensor::from_vec(data, &[100]).expect("Tensor should succeed");
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
        let result = pelt
            .detect(&series)
            .expect("detection operation should succeed");

        assert_eq!(result.algorithm, "PELT");
        // Should detect at least one change point
        assert!(!result.change_points.is_empty());
    }

    #[test]
    fn test_pelt_with_cost_functions() {
        let series = create_change_point_series();

        for cost_fn in [CostFunction::L2, CostFunction::L1, CostFunction::Variance] {
            let pelt = PELT::new(10.0, 5).with_cost_function(cost_fn);
            let result = pelt
                .detect(&series)
                .expect("detection operation should succeed");
            assert_eq!(result.algorithm, "PELT");
        }
    }

    /// Build a three-regime series (means 0, 10, -5 over [0,50), [50,100), [100,150))
    /// with additive Gaussian noise, deterministically seeded.
    fn create_three_regime_series(seed: u64, noise_std: f64) -> Vec<f32> {
        use scirs2_core::random::{Distribution, Normal, Random};
        let mut rng = Random::seed(seed);
        let noise = Normal::new(0.0f64, noise_std).expect("normal distribution should succeed");
        let means = [0.0f64, 10.0, -5.0];
        let mut data = Vec::with_capacity(150);
        for &mu in means.iter() {
            for _ in 0..50 {
                let v = mu + noise.sample(&mut rng);
                data.push(v as f32);
            }
        }
        data
    }

    #[test]
    fn test_bic_penalty_formula() {
        // 2 * variance * ln(n)
        let p = PELT::bic_penalty(150, 1.0);
        assert!((p - 2.0 * (150f64).ln()).abs() < 1e-9, "got {}", p);
    }

    /// KNOWN-ANSWER: PELT must recover exactly the two true changepoints {50, 100}
    /// of a synthetic mean-shift series (means 0, 10, -5 with unit Gaussian noise).
    #[test]
    fn test_pelt_known_changepoints() {
        let data = create_three_regime_series(12345, 1.0);
        let n = data.len();
        let tensor = Tensor::from_vec(data, &[n]).expect("Tensor should succeed");
        let series = TimeSeries::new(tensor);

        // BIC-family penalty: with mean gaps of 10 and 15 against unit noise the two
        // true changepoints each cut the SSE by O(10^3), while a spurious split saves
        // only O(noise variance); any penalty in (~tens, ~10^3) recovers exactly
        // {50, 100}. We use 5x the textbook BIC value to stay comfortably inside it.
        let penalty = 5.0 * PELT::bic_penalty(n, 1.0);
        let pelt = PELT::new(penalty, 10).with_cost_function(CostFunction::L2);
        let result = pelt
            .detect(&series)
            .expect("detection operation should succeed");

        assert_eq!(
            result.change_points.len(),
            2,
            "expected exactly two changepoints, got {:?}",
            result.change_points
        );
        assert!(
            (result.change_points[0] as i64 - 50).abs() <= 2,
            "first changepoint should be ~50, got {}",
            result.change_points[0]
        );
        assert!(
            (result.change_points[1] as i64 - 100).abs() <= 2,
            "second changepoint should be ~100, got {}",
            result.change_points[1]
        );
        assert_eq!(result.change_points.len(), result.scores.len());
    }

    /// PELT pruning must not change the optimum: the pruned run and the un-pruned
    /// optimal-partitioning run must produce identical changepoints and scores.
    #[test]
    fn test_pelt_pruning_matches_optimal_partitioning() {
        let data = create_three_regime_series(2024, 1.0);
        let n = data.len();
        let penalty = 5.0 * PELT::bic_penalty(n, 1.0);
        let pelt = PELT::new(penalty, 10).with_cost_function(CostFunction::L2);

        let pruned = pelt.run(&data, true);
        let optimal = pelt.run(&data, false);

        assert_eq!(
            pruned.change_points, optimal.change_points,
            "pruning changed the optimal partition"
        );
        assert_eq!(pruned.scores.len(), optimal.scores.len());
        for (a, b) in pruned.scores.iter().zip(optimal.scores.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "segment scores differ: {} vs {}",
                a,
                b
            );
        }
    }

    /// Pruning equivalence must also hold for a harder, noisier multi-regime series
    /// at a smaller penalty (more candidate changepoints survive).
    #[test]
    fn test_pelt_pruning_equivalence_dense() {
        let data = create_three_regime_series(7, 2.0);
        let n = data.len();
        let penalty = PELT::bic_penalty(n, 4.0);
        let pelt = PELT::new(penalty, 5).with_cost_function(CostFunction::L2);

        let pruned = pelt.run(&data, true);
        let optimal = pelt.run(&data, false);
        assert_eq!(
            pruned.change_points, optimal.change_points,
            "pruning changed the optimal partition (dense case)"
        );
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
        let result = bs
            .detect(&series)
            .expect("detection operation should succeed");

        assert_eq!(result.algorithm, "Binary Segmentation");
        assert!(!result.change_points.is_empty());
    }

    #[test]
    fn test_binary_segmentation_max_change_points() {
        let series = create_change_point_series();
        let bs = BinarySegmentation::new(0.1).with_max_change_points(2);
        let result = bs
            .detect(&series)
            .expect("detection operation should succeed");

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
        let result = detector
            .detect(&series)
            .expect("detection operation should succeed");

        assert_eq!(result.algorithm, "Window");
        // Should detect change around index 50
        assert!(!result.change_points.is_empty());
    }

    #[test]
    fn test_window_detector_variance() {
        let series = create_change_point_series();
        let detector = WindowDetector::new(10, 0.1).with_statistic(WindowStatistic::Variance);
        let result = detector
            .detect(&series)
            .expect("detection operation should succeed");

        assert_eq!(result.algorithm, "Window");
    }

    #[test]
    fn test_window_detector_cusum() {
        let series = create_change_point_series();
        let detector = WindowDetector::new(10, 5.0).with_statistic(WindowStatistic::CUSUM);
        let result = detector
            .detect(&series)
            .expect("detection operation should succeed");

        assert_eq!(result.algorithm, "Window");
    }

    #[test]
    fn test_change_point_result() {
        let series = create_change_point_series();
        let pelt = PELT::new(10.0, 5);
        let result = pelt
            .detect(&series)
            .expect("detection operation should succeed");

        assert_eq!(result.change_points.len(), result.scores.len());
        assert!(!result.algorithm.is_empty());
    }
}
