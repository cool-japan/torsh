//! Advanced statistical hypothesis testing for model comparison
//!
//! This module provides statistical tests for comparing models and validating results.

use serde::{Deserialize, Serialize};

/// Paired t-test for comparing two models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedTTest {
    /// Test statistic
    pub t_statistic: f64,
    /// Degrees of freedom
    pub df: usize,
    /// P-value (two-tailed)
    pub p_value: f64,
    /// Mean difference
    pub mean_diff: f64,
    /// Standard error of difference
    pub std_error: f64,
    /// 95% confidence interval for mean difference
    pub confidence_interval_95: (f64, f64),
}

impl PairedTTest {
    /// Perform a paired t-test
    ///
    /// Tests whether the mean difference between paired samples is zero
    pub fn compute(scores_a: &[f64], scores_b: &[f64]) -> Self {
        assert_eq!(
            scores_a.len(),
            scores_b.len(),
            "Paired samples must have same length"
        );
        let n = scores_a.len();
        assert!(n >= 2, "Need at least 2 samples");

        // Compute differences
        let diffs: Vec<f64> = scores_a
            .iter()
            .zip(scores_b.iter())
            .map(|(a, b)| a - b)
            .collect();

        let mean_diff = diffs.iter().sum::<f64>() / n as f64;
        let variance = diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std_error = variance.sqrt() / (n as f64).sqrt();

        let t_statistic = mean_diff / std_error;
        let df = n - 1;

        // Approximate p-value using normal distribution for large n
        // For small n, this is less accurate but gives a reasonable approximation
        let p_value = if n >= 30 {
            2.0 * (1.0 - approx_normal_cdf(t_statistic.abs()))
        } else {
            // Use Student's t-distribution approximation
            2.0 * (1.0 - approx_t_cdf(t_statistic.abs(), df))
        };

        // 95% CI using t-distribution
        let t_critical = approx_t_quantile(0.975, df);
        let margin = t_critical * std_error;
        let confidence_interval_95 = (mean_diff - margin, mean_diff + margin);

        Self {
            t_statistic,
            df,
            p_value,
            mean_diff,
            std_error,
            confidence_interval_95,
        }
    }

    /// Check if the difference is statistically significant at given alpha level
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
}

/// McNemar's test for comparing classifier performance on same test set
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McNemarTest {
    /// Test statistic (chi-squared)
    pub chi_squared: f64,
    /// P-value
    pub p_value: f64,
    /// Number of samples where A correct, B wrong
    pub n_01: usize,
    /// Number of samples where A wrong, B correct
    pub n_10: usize,
    /// Total disagreements
    pub total_disagreements: usize,
}

impl McNemarTest {
    /// Perform McNemar's test
    ///
    /// # Arguments
    /// * `y_true` - True labels
    /// * `y_pred_a` - Predictions from model A
    /// * `y_pred_b` - Predictions from model B
    pub fn compute(y_true: &[usize], y_pred_a: &[usize], y_pred_b: &[usize]) -> Self {
        assert_eq!(y_true.len(), y_pred_a.len());
        assert_eq!(y_true.len(), y_pred_b.len());

        let mut n_01 = 0; // A correct, B wrong
        let mut n_10 = 0; // A wrong, B correct

        for i in 0..y_true.len() {
            let a_correct = y_pred_a[i] == y_true[i];
            let b_correct = y_pred_b[i] == y_true[i];

            match (a_correct, b_correct) {
                (true, false) => n_01 += 1,
                (false, true) => n_10 += 1,
                _ => {} // Both correct or both wrong - not counted
            }
        }

        let total_disagreements = n_01 + n_10;

        // McNemar's test statistic with continuity correction
        let chi_squared = if total_disagreements == 0 {
            0.0
        } else {
            let diff = (n_01 as f64 - n_10 as f64).abs() - 1.0; // Continuity correction
            (diff * diff) / (n_01 + n_10) as f64
        };

        // P-value from chi-squared distribution with 1 df
        let p_value = 1.0 - approx_chi_squared_cdf(chi_squared, 1);

        Self {
            chi_squared,
            p_value,
            n_01,
            n_10,
            total_disagreements,
        }
    }

    /// Check if the difference is statistically significant
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
}

/// 5x2 cv paired t-test for comparing two models
///
/// More reliable than simple paired t-test for small datasets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiveByTwoCVTest {
    /// Test statistic
    pub t_statistic: f64,
    /// Degrees of freedom (always 5 for 5x2 cv)
    pub df: usize,
    /// P-value
    pub p_value: f64,
    /// Mean of differences
    pub mean_diff: f64,
}

impl FiveByTwoCVTest {
    /// Perform 5x2 cv paired t-test
    ///
    /// # Arguments
    /// * `fold_diffs` - Differences in performance for each of 10 folds (5 repeats x 2 folds)
    pub fn compute(fold_diffs: &[f64]) -> Self {
        assert_eq!(
            fold_diffs.len(),
            10,
            "5x2 cv test requires exactly 10 fold differences"
        );

        // Organize as 5 repeats of 2 folds
        let mut s_squared = 0.0;
        for repeat in 0..5 {
            let d1 = fold_diffs[repeat * 2];
            let d2 = fold_diffs[repeat * 2 + 1];
            let mean_repeat = (d1 + d2) / 2.0;
            s_squared += (d1 - mean_repeat).powi(2) + (d2 - mean_repeat).powi(2);
        }

        let mean_diff = fold_diffs.iter().sum::<f64>() / 10.0;
        let variance = s_squared / 10.0;

        let t_statistic = mean_diff / variance.sqrt();
        let df = 5;
        let p_value = 2.0 * (1.0 - approx_t_cdf(t_statistic.abs(), df));

        Self {
            t_statistic,
            df,
            p_value,
            mean_diff,
        }
    }

    /// Check if the difference is statistically significant
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
}

/// Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WilcoxonTest {
    /// Test statistic (W)
    pub w_statistic: f64,
    /// P-value (approximate)
    pub p_value: f64,
    /// Number of non-zero differences
    pub n_valid: usize,
}

impl WilcoxonTest {
    /// Perform Wilcoxon signed-rank test
    pub fn compute(scores_a: &[f64], scores_b: &[f64]) -> Self {
        assert_eq!(scores_a.len(), scores_b.len());

        // Compute differences and their absolute values, excluding zeros
        let mut diffs_with_ranks: Vec<(f64, f64)> = scores_a
            .iter()
            .zip(scores_b.iter())
            .map(|(a, b)| a - b)
            .filter(|&d| d.abs() > 1e-10) // Exclude near-zero differences
            .map(|d| (d, d.abs()))
            .collect();

        let n_valid = diffs_with_ranks.len();
        if n_valid == 0 {
            return Self {
                w_statistic: 0.0,
                p_value: 1.0,
                n_valid: 0,
            };
        }

        // Rank by absolute value
        diffs_with_ranks.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .expect("absolute differences should be comparable")
        });

        // Sum of ranks for positive differences
        let w_statistic: f64 = diffs_with_ranks
            .iter()
            .enumerate()
            .filter(|(_, (diff, _))| *diff > 0.0)
            .map(|(i, _)| (i + 1) as f64)
            .sum();

        // Normal approximation for large n
        let n = n_valid as f64;
        let expected = n * (n + 1.0) / 4.0;
        let variance = n * (n + 1.0) * (2.0 * n + 1.0) / 24.0;
        let z_statistic = (w_statistic - expected) / variance.sqrt();
        let p_value = 2.0 * (1.0 - approx_normal_cdf(z_statistic.abs()));

        Self {
            w_statistic,
            p_value,
            n_valid,
        }
    }

    /// Check if the difference is statistically significant
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
}

// Statistical distribution approximations

/// Approximate normal CDF using error function approximation
fn approx_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz and Stegun)
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Approximate t-distribution CDF
fn approx_t_cdf(t: f64, df: usize) -> f64 {
    if df >= 30 {
        // Use normal approximation for large df
        approx_normal_cdf(t)
    } else {
        // Simple approximation for small df
        let x = df as f64 / (df as f64 + t * t);
        0.5 + 0.5 * x.sqrt() * (1.0 - x).powf(0.5)
    }
}

/// Approximate t-distribution quantile
fn approx_t_quantile(p: f64, df: usize) -> f64 {
    if df >= 30 {
        // Use normal quantile for large df
        approx_normal_quantile(p)
    } else {
        // Simple approximation for common quantiles
        match (df, p) {
            (1, 0.975) => 12.706,
            (2, 0.975) => 4.303,
            (3, 0.975) => 3.182,
            (4, 0.975) => 2.776,
            (5, 0.975) => 2.571,
            (10, 0.975) => 2.228,
            (20, 0.975) => 2.086,
            _ => 1.96 + 1.0 / df as f64, // Rough approximation
        }
    }
}

/// Approximate normal quantile (inverse CDF)
fn approx_normal_quantile(p: f64) -> f64 {
    // Beasley-Springer-Moro algorithm approximation
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];

    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];

    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];

    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }

    if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }

    let q = (-2.0 * (1.0 - p).ln()).sqrt();
    -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
}

/// Approximate chi-squared CDF
fn approx_chi_squared_cdf(x: f64, df: usize) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Wilson-Hilferty approximation
    let k = df as f64;
    let z = ((x / k).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / (2.0 / (9.0 * k)).sqrt();
    approx_normal_cdf(z)
}

/// Friedman test for comparing multiple models across multiple datasets
///
/// Non-parametric test for repeated measures (alternative to repeated measures ANOVA)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FriedmanTest {
    /// Test statistic (chi-squared distributed)
    pub chi_squared: f64,
    /// Degrees of freedom
    pub df: usize,
    /// P-value
    pub p_value: f64,
    /// Number of models (treatments)
    pub n_models: usize,
    /// Number of datasets (blocks)
    pub n_datasets: usize,
}

impl FriedmanTest {
    /// Perform Friedman test
    ///
    /// # Arguments
    /// * `scores` - Matrix of scores: `scores[dataset_idx][model_idx]`
    pub fn compute(scores: &[Vec<f64>]) -> Self {
        let n_datasets = scores.len();
        let n_models = scores[0].len();

        assert!(n_datasets >= 2, "Need at least 2 datasets");
        assert!(n_models >= 3, "Need at least 3 models for Friedman test");
        assert!(
            scores.iter().all(|s| s.len() == n_models),
            "All datasets must have same number of models"
        );

        // Rank models within each dataset
        let mut rank_sums = vec![0.0; n_models];

        for dataset_scores in scores {
            let ranks = Self::rank_values(dataset_scores);
            for (model_idx, &rank) in ranks.iter().enumerate() {
                rank_sums[model_idx] += rank;
            }
        }

        // Compute Friedman statistic
        let k = n_models as f64;
        let n = n_datasets as f64;

        let sum_of_squares: f64 = rank_sums.iter().map(|&r| r * r).sum();

        let chi_squared = (12.0 / (n * k * (k + 1.0))) * sum_of_squares - 3.0 * n * (k + 1.0);

        let df = n_models - 1;
        let p_value = 1.0 - approx_chi_squared_cdf(chi_squared, df);

        Self {
            chi_squared,
            df,
            p_value,
            n_models,
            n_datasets,
        }
    }

    /// Assign ranks (average rank for ties)
    fn rank_values(values: &[f64]) -> Vec<f64> {
        let n = values.len();
        let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .expect("values should be comparable for ranking")
        }); // Descending order

        let mut ranks = vec![0.0; n];
        let mut i = 0;

        while i < n {
            let mut j = i;
            while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-10 {
                j += 1;
            }

            let avg_rank = ((i + 1) + j) as f64 / 2.0;
            for k in i..j {
                ranks[indexed[k].0] = avg_rank;
            }

            i = j;
        }

        ranks
    }

    /// Check if difference is statistically significant at given alpha level
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
}

/// Nemenyi post-hoc test after Friedman test
///
/// Pairwise comparison of models after rejecting the Friedman null hypothesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NemenyiTest {
    /// Critical difference for significance
    pub critical_difference: f64,
    /// Pairwise rank differences
    pub rank_differences: Vec<Vec<f64>>,
    /// Model names/indices
    pub n_models: usize,
}

impl NemenyiTest {
    /// Perform Nemenyi post-hoc test
    ///
    /// # Arguments
    /// * `scores` - Matrix of scores: `scores[dataset_idx][model_idx]`
    /// * `alpha` - Significance level (default 0.05)
    pub fn compute(scores: &[Vec<f64>], alpha: f64) -> Self {
        let n_datasets = scores.len();
        let n_models = scores[0].len();

        // Compute average ranks
        let mut rank_sums = vec![0.0; n_models];

        for dataset_scores in scores {
            let ranks = FriedmanTest::rank_values(dataset_scores);
            for (model_idx, &rank) in ranks.iter().enumerate() {
                rank_sums[model_idx] += rank;
            }
        }

        let avg_ranks: Vec<f64> = rank_sums.iter().map(|&r| r / n_datasets as f64).collect();

        // Compute pairwise differences
        let mut rank_differences = vec![vec![0.0; n_models]; n_models];
        for i in 0..n_models {
            for j in 0..n_models {
                rank_differences[i][j] = (avg_ranks[i] - avg_ranks[j]).abs();
            }
        }

        // Critical difference (simplified Nemenyi critical value)
        // CD = q_alpha * sqrt(k(k+1) / (6N))
        // where q_alpha is the studentized range statistic
        let k = n_models as f64;
        let n = n_datasets as f64;

        // Approximate q_alpha for alpha=0.05
        let q_alpha = if alpha <= 0.05 {
            2.344 + (k - 3.0) * 0.124 // Rough approximation
        } else {
            2.0
        };

        let critical_difference = q_alpha * (k * (k + 1.0) / (6.0 * n)).sqrt();

        Self {
            critical_difference,
            rank_differences,
            n_models,
        }
    }

    /// Check if two models are significantly different
    pub fn is_different(&self, model_i: usize, model_j: usize) -> bool {
        self.rank_differences[model_i][model_j] > self.critical_difference
    }
}

/// Mann-Whitney U test (Wilcoxon rank-sum test)
///
/// Non-parametric test for comparing two independent groups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MannWhitneyTest {
    /// U statistic
    pub u_statistic: f64,
    /// P-value (two-tailed)
    pub p_value: f64,
    /// Sample size group A
    pub n_a: usize,
    /// Sample size group B
    pub n_b: usize,
}

impl MannWhitneyTest {
    /// Perform Mann-Whitney U test
    pub fn compute(group_a: &[f64], group_b: &[f64]) -> Self {
        let n_a = group_a.len();
        let n_b = group_b.len();

        assert!(n_a >= 1 && n_b >= 1, "Need at least 1 sample in each group");

        // Combine and rank
        let mut combined: Vec<(f64, usize)> = Vec::with_capacity(n_a + n_b);
        for &val in group_a {
            combined.push((val, 0)); // 0 for group A
        }
        for &val in group_b {
            combined.push((val, 1)); // 1 for group B
        }

        combined.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .expect("combined values should be comparable")
        });

        // Assign ranks (average for ties)
        let mut ranks = vec![0.0; n_a + n_b];
        let mut i = 0;

        while i < combined.len() {
            let mut j = i;
            while j < combined.len() && (combined[j].0 - combined[i].0).abs() < 1e-10 {
                j += 1;
            }

            let avg_rank = ((i + 1) + j) as f64 / 2.0;
            for k in i..j {
                ranks[k] = avg_rank;
            }

            i = j;
        }

        // Sum ranks for group A
        let mut rank_sum_a = 0.0;
        for (idx, &(_, group)) in combined.iter().enumerate() {
            if group == 0 {
                rank_sum_a += ranks[idx];
            }
        }

        // Calculate U statistics
        let u_a = rank_sum_a - (n_a * (n_a + 1)) as f64 / 2.0;
        let u_b = (n_a * n_b) as f64 - u_a;
        let u_statistic = u_a.min(u_b);

        // Normal approximation for p-value
        let mean_u = (n_a * n_b) as f64 / 2.0;
        let std_u = ((n_a * n_b * (n_a + n_b + 1)) as f64 / 12.0).sqrt();
        let z = (u_statistic - mean_u) / std_u;
        let p_value = 2.0 * (1.0 - approx_normal_cdf(z.abs()));

        Self {
            u_statistic,
            p_value,
            n_a,
            n_b,
        }
    }

    /// Check if difference is statistically significant at given alpha level
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
}

/// Kruskal-Wallis H test
///
/// Non-parametric alternative to one-way ANOVA for comparing multiple independent groups
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KruskalWallisTest {
    /// H statistic (chi-squared distributed)
    pub h_statistic: f64,
    /// Degrees of freedom
    pub df: usize,
    /// P-value
    pub p_value: f64,
    /// Number of groups
    pub n_groups: usize,
}

impl KruskalWallisTest {
    /// Perform Kruskal-Wallis test
    ///
    /// # Arguments
    /// * `groups` - Vector of groups, each containing samples
    pub fn compute(groups: &[Vec<f64>]) -> Self {
        let n_groups = groups.len();
        assert!(n_groups >= 2, "Need at least 2 groups");

        // Combine all samples
        let mut combined: Vec<(f64, usize)> = Vec::new();
        for (group_idx, group) in groups.iter().enumerate() {
            for &val in group {
                combined.push((val, group_idx));
            }
        }

        let n_total = combined.len();
        combined.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .expect("combined values should be comparable")
        });

        // Assign ranks
        let mut ranks = vec![0.0; n_total];
        let mut i = 0;

        while i < n_total {
            let mut j = i;
            while j < n_total && (combined[j].0 - combined[i].0).abs() < 1e-10 {
                j += 1;
            }

            let avg_rank = ((i + 1) + j) as f64 / 2.0;
            for k in i..j {
                ranks[k] = avg_rank;
            }

            i = j;
        }

        // Sum ranks for each group
        let mut rank_sums = vec![0.0; n_groups];
        let mut group_sizes = vec![0; n_groups];

        for (idx, &(_, group_idx)) in combined.iter().enumerate() {
            rank_sums[group_idx] += ranks[idx];
            group_sizes[group_idx] += 1;
        }

        // Compute H statistic
        let n = n_total as f64;
        let mut h_statistic = 0.0;

        for i in 0..n_groups {
            let n_i = group_sizes[i] as f64;
            let r_i = rank_sums[i];
            h_statistic += (r_i * r_i) / n_i;
        }

        h_statistic = (12.0 / (n * (n + 1.0))) * h_statistic - 3.0 * (n + 1.0);

        let df = n_groups - 1;
        let p_value = 1.0 - approx_chi_squared_cdf(h_statistic, df);

        Self {
            h_statistic,
            df,
            p_value,
            n_groups,
        }
    }

    /// Check if difference is statistically significant at given alpha level
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paired_t_test() {
        let scores_a = vec![0.85, 0.87, 0.84, 0.86, 0.88];
        let scores_b = vec![0.80, 0.82, 0.79, 0.81, 0.83];

        let test = PairedTTest::compute(&scores_a, &scores_b);

        assert!(test.t_statistic > 0.0); // A is better than B
        assert!(test.mean_diff > 0.0);
        assert!(test.std_error > 0.0);
        assert!(test.df == 4);
    }

    #[test]
    fn test_mcnemar_test() {
        let y_true = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let y_pred_a = vec![0, 1, 0, 1, 1, 0, 0, 1]; // 75% accuracy, disagrees on 2
        let y_pred_b = vec![0, 1, 1, 0, 0, 1, 0, 1]; // 75% accuracy, disagrees on 2

        let test = McNemarTest::compute(&y_true, &y_pred_a, &y_pred_b);

        assert_eq!(test.total_disagreements, test.n_01 + test.n_10);
        assert!(test.p_value >= 0.0 && test.p_value <= 1.0);
    }

    #[test]
    fn test_5x2_cv_test() {
        // 10 fold differences (5 repeats x 2 folds)
        let fold_diffs = vec![0.05, 0.03, 0.04, 0.02, 0.06, 0.04, 0.03, 0.05, 0.04, 0.03];

        let test = FiveByTwoCVTest::compute(&fold_diffs);

        assert_eq!(test.df, 5);
        assert!(test.mean_diff > 0.0);
        assert!(test.p_value >= 0.0 && test.p_value <= 1.0);
    }

    #[test]
    fn test_wilcoxon_test() {
        let scores_a = vec![0.85, 0.87, 0.84, 0.86, 0.88];
        let scores_b = vec![0.80, 0.82, 0.79, 0.81, 0.83];

        let test = WilcoxonTest::compute(&scores_a, &scores_b);

        assert_eq!(test.n_valid, 5);
        assert!(test.p_value >= 0.0 && test.p_value <= 1.0);
    }

    #[test]
    fn test_friedman_test() {
        // 3 models tested on 4 datasets
        let scores = vec![
            vec![0.85, 0.80, 0.78], // Dataset 1
            vec![0.88, 0.82, 0.80], // Dataset 2
            vec![0.84, 0.79, 0.77], // Dataset 3
            vec![0.86, 0.81, 0.79], // Dataset 4
        ];

        let test = FriedmanTest::compute(&scores);

        assert_eq!(test.n_models, 3);
        assert_eq!(test.n_datasets, 4);
        assert_eq!(test.df, 2);
        assert!(test.chi_squared > 0.0);
        assert!(test.p_value >= 0.0 && test.p_value <= 1.0);
    }

    #[test]
    fn test_nemenyi_test() {
        let scores = vec![
            vec![0.85, 0.80, 0.78],
            vec![0.88, 0.82, 0.80],
            vec![0.84, 0.79, 0.77],
        ];

        let test = NemenyiTest::compute(&scores, 0.05);

        assert_eq!(test.n_models, 3);
        assert!(test.critical_difference > 0.0);
        assert_eq!(test.rank_differences.len(), 3);
    }

    #[test]
    fn test_mann_whitney_test() {
        let group_a = vec![0.85, 0.87, 0.84, 0.86, 0.88];
        let group_b = vec![0.80, 0.82, 0.79, 0.81, 0.83];

        let test = MannWhitneyTest::compute(&group_a, &group_b);

        assert_eq!(test.n_a, 5);
        assert_eq!(test.n_b, 5);
        assert!(test.u_statistic >= 0.0);
        assert!(test.p_value >= 0.0 && test.p_value <= 1.0);
    }

    #[test]
    fn test_kruskal_wallis_test() {
        let groups = vec![
            vec![0.85, 0.87, 0.84],
            vec![0.80, 0.82, 0.79],
            vec![0.88, 0.89, 0.90],
        ];

        let test = KruskalWallisTest::compute(&groups);

        assert_eq!(test.n_groups, 3);
        assert_eq!(test.df, 2);
        assert!(test.h_statistic > 0.0);
        assert!(test.p_value >= 0.0 && test.p_value <= 1.0);
    }

    #[test]
    fn test_normal_cdf() {
        assert!((approx_normal_cdf(0.0) - 0.5).abs() < 0.001);
        assert!(approx_normal_cdf(-1.96) < 0.05);
        assert!(approx_normal_cdf(1.96) > 0.95);
    }
}
