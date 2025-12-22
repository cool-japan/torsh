//! Statistical validation and bootstrap confidence intervals
//!
//! This module provides comprehensive statistical analysis for machine learning metrics,
//! including bootstrap confidence intervals, hypothesis testing, and cross-validation.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::Metric;
use scirs2_core::random::Random;
use std::collections::HashMap;
use torsh_tensor::Tensor;

/// Bootstrap confidence interval result
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    /// The metric value
    pub metric_value: f64,
    /// Confidence interval bounds (lower, upper)
    pub confidence_interval: (f64, f64),
    /// Bootstrap standard error
    pub standard_error: f64,
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
    /// Confidence level used
    pub confidence_level: f64,
    /// Bootstrap distribution percentiles
    pub percentiles: Vec<(f64, f64)>,
}

/// Hypothesis test result
#[derive(Debug, Clone)]
pub struct HypothesisTestResult {
    /// Test statistic value
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Whether the null hypothesis is rejected at alpha level
    pub is_significant: bool,
    /// Alpha level used
    pub alpha: f64,
    /// Test type
    pub test_type: String,
    /// Effect size (if applicable)
    pub effect_size: Option<f64>,
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    /// CV scores for each fold
    pub cv_scores: Vec<f64>,
    /// Mean score across folds
    pub mean_score: f64,
    /// Standard deviation across folds
    pub std_score: f64,
    /// 95% confidence interval of the mean
    pub confidence_interval: (f64, f64),
    /// Fold-wise detailed metrics
    pub fold_metrics: Vec<HashMap<String, f64>>,
}

/// Bootstrap confidence interval calculator
pub struct BootstrapCI {
    n_bootstrap: usize,
    confidence_level: f64,
    random_seed: Option<u64>,
    stratify: bool,
}

impl BootstrapCI {
    /// Create a new bootstrap CI calculator
    pub fn new(n_bootstrap: usize, confidence_level: f64) -> Self {
        Self {
            n_bootstrap,
            confidence_level,
            random_seed: None,
            stratify: false,
        }
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Enable stratified bootstrap (for classification problems)
    pub fn with_stratification(mut self, stratify: bool) -> Self {
        self.stratify = stratify;
        self
    }

    /// Compute bootstrap confidence interval for a metric
    pub fn compute_ci<M: Metric + Clone>(
        &self,
        metric: &M,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> BootstrapResult {
        let mut rng = Random::seed(self.random_seed.unwrap_or(42));
        let original_score = metric.compute(predictions, targets);

        let pred_vec = predictions.to_vec().unwrap_or_default();
        let target_vec = targets.to_vec().unwrap_or_default();

        if pred_vec.is_empty() || target_vec.is_empty() || pred_vec.len() != target_vec.len() {
            return BootstrapResult {
                metric_value: original_score,
                confidence_interval: (original_score, original_score),
                standard_error: 0.0,
                n_bootstrap: 0,
                confidence_level: self.confidence_level,
                percentiles: vec![],
            };
        }

        let n_samples = pred_vec.len();
        let mut bootstrap_scores = Vec::with_capacity(self.n_bootstrap);

        // Perform bootstrap sampling
        for _ in 0..self.n_bootstrap {
            let (boot_pred, boot_target) = if self.stratify {
                self.stratified_bootstrap_sample(&pred_vec, &target_vec, &mut rng)
            } else {
                self.simple_bootstrap_sample(&pred_vec, &target_vec, &mut rng)
            };

            // Create tensors for this bootstrap sample
            if let (Ok(pred_tensor), Ok(target_tensor)) = (
                torsh_tensor::creation::from_vec(
                    boot_pred,
                    &[n_samples],
                    torsh_core::device::DeviceType::Cpu,
                ),
                torsh_tensor::creation::from_vec(
                    boot_target,
                    &[n_samples],
                    torsh_core::device::DeviceType::Cpu,
                ),
            ) {
                let score = metric.compute(&pred_tensor, &target_tensor);
                if score.is_finite() {
                    bootstrap_scores.push(score);
                }
            }
        }

        if bootstrap_scores.is_empty() {
            return BootstrapResult {
                metric_value: original_score,
                confidence_interval: (original_score, original_score),
                standard_error: 0.0,
                n_bootstrap: 0,
                confidence_level: self.confidence_level,
                percentiles: vec![],
            };
        }

        bootstrap_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Calculate confidence interval
        let alpha = 1.0 - self.confidence_level;
        let lower_percentile = (alpha / 2.0) * 100.0;
        let upper_percentile = (1.0 - alpha / 2.0) * 100.0;

        let ci_lower = self.percentile(&bootstrap_scores, lower_percentile);
        let ci_upper = self.percentile(&bootstrap_scores, upper_percentile);

        // Calculate standard error
        let mean_bootstrap: f64 =
            bootstrap_scores.iter().sum::<f64>() / bootstrap_scores.len() as f64;
        let variance: f64 = bootstrap_scores
            .iter()
            .map(|x| (x - mean_bootstrap).powi(2))
            .sum::<f64>()
            / (bootstrap_scores.len() - 1) as f64;
        let std_error = variance.sqrt();

        // Calculate various percentiles
        let percentiles = vec![
            (5.0, self.percentile(&bootstrap_scores, 5.0)),
            (10.0, self.percentile(&bootstrap_scores, 10.0)),
            (25.0, self.percentile(&bootstrap_scores, 25.0)),
            (50.0, self.percentile(&bootstrap_scores, 50.0)),
            (75.0, self.percentile(&bootstrap_scores, 75.0)),
            (90.0, self.percentile(&bootstrap_scores, 90.0)),
            (95.0, self.percentile(&bootstrap_scores, 95.0)),
        ];

        BootstrapResult {
            metric_value: original_score,
            confidence_interval: (ci_lower, ci_upper),
            standard_error: std_error,
            n_bootstrap: bootstrap_scores.len(),
            confidence_level: self.confidence_level,
            percentiles,
        }
    }

    fn simple_bootstrap_sample<R: scirs2_core::random::Rng>(
        &self,
        predictions: &[f32],
        targets: &[f32],
        rng: &mut R,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = predictions.len();
        let mut boot_pred = Vec::with_capacity(n);
        let mut boot_target = Vec::with_capacity(n);

        for _ in 0..n {
            let idx = rng.random_range(0..n);
            boot_pred.push(predictions[idx]);
            boot_target.push(targets[idx]);
        }

        (boot_pred, boot_target)
    }

    fn stratified_bootstrap_sample<R: scirs2_core::random::Rng>(
        &self,
        predictions: &[f32],
        targets: &[f32],
        rng: &mut R,
    ) -> (Vec<f32>, Vec<f32>) {
        // Group by target classes for stratification
        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &target) in targets.iter().enumerate() {
            class_indices.entry(target as i32).or_default().push(i);
        }

        let n = predictions.len();
        let mut boot_pred = Vec::with_capacity(n);
        let mut boot_target = Vec::with_capacity(n);

        // Sample proportionally from each class
        for (_class, indices) in &class_indices {
            let class_size = indices.len();
            let n_samples =
                (class_size as f64 * n as f64 / predictions.len() as f64).round() as usize;

            for _ in 0..n_samples {
                let idx = rng.random_range(0..indices.len());
                let sample_idx = indices[idx];
                boot_pred.push(predictions[sample_idx]);
                boot_target.push(targets[sample_idx]);
            }
        }

        // Fill remaining samples if needed
        while boot_pred.len() < n {
            let idx = rng.random_range(0..predictions.len());
            boot_pred.push(predictions[idx]);
            boot_target.push(targets[idx]);
        }

        (boot_pred, boot_target)
    }

    fn percentile(&self, sorted_values: &[f64], percentile: f64) -> f64 {
        if sorted_values.is_empty() {
            return 0.0;
        }

        let index = (percentile / 100.0) * (sorted_values.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper || upper >= sorted_values.len() {
            sorted_values[lower.min(sorted_values.len() - 1)]
        } else {
            let weight = index - lower as f64;
            sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
        }
    }
}

/// Permutation test for comparing two metrics
pub struct PermutationTest {
    n_permutations: usize,
    random_seed: Option<u64>,
}

impl PermutationTest {
    /// Create a new permutation test
    pub fn new(n_permutations: usize) -> Self {
        Self {
            n_permutations,
            random_seed: None,
        }
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Test if two metrics are significantly different
    pub fn compare_metrics<M: Metric + Clone>(
        &self,
        metric: &M,
        predictions1: &Tensor,
        targets1: &Tensor,
        predictions2: &Tensor,
        targets2: &Tensor,
        alpha: f64,
    ) -> HypothesisTestResult {
        let score1 = metric.compute(predictions1, targets1);
        let score2 = metric.compute(predictions2, targets2);
        let observed_diff = (score1 - score2).abs();

        let pred1_vec = predictions1.to_vec().unwrap_or_default();
        let target1_vec = targets1.to_vec().unwrap_or_default();
        let pred2_vec = predictions2.to_vec().unwrap_or_default();
        let target2_vec = targets2.to_vec().unwrap_or_default();

        if pred1_vec.is_empty()
            || pred2_vec.is_empty()
            || pred1_vec.len() != target1_vec.len()
            || pred2_vec.len() != target2_vec.len()
        {
            return HypothesisTestResult {
                statistic: observed_diff,
                p_value: 1.0,
                is_significant: false,
                alpha,
                test_type: "permutation_test".to_string(),
                effect_size: None,
            };
        }

        let mut rng = Random::seed(self.random_seed.unwrap_or(42));
        let mut extreme_count = 0;

        // Combine all data
        let mut all_pred = pred1_vec.clone();
        all_pred.extend(pred2_vec.iter());
        let mut all_target = target1_vec.clone();
        all_target.extend(target2_vec.iter());

        let n1 = pred1_vec.len();
        let n_total = all_pred.len();

        // Perform permutation test
        for _ in 0..self.n_permutations {
            // Shuffle combined data
            for i in (1..n_total).rev() {
                let j = rng.gen_range(0..=i);
                all_pred.swap(i, j);
                all_target.swap(i, j);
            }

            // Split into two groups
            let perm_pred1 = &all_pred[..n1];
            let perm_target1 = &all_target[..n1];
            let perm_pred2 = &all_pred[n1..];
            let perm_target2 = &all_target[n1..];

            // Compute scores for permuted groups
            if let (
                Ok(perm_tensor1_pred),
                Ok(perm_tensor1_target),
                Ok(perm_tensor2_pred),
                Ok(perm_tensor2_target),
            ) = (
                torsh_tensor::creation::from_vec(
                    perm_pred1.to_vec(),
                    &[n1],
                    torsh_core::device::DeviceType::Cpu,
                ),
                torsh_tensor::creation::from_vec(
                    perm_target1.to_vec(),
                    &[n1],
                    torsh_core::device::DeviceType::Cpu,
                ),
                torsh_tensor::creation::from_vec(
                    perm_pred2.to_vec(),
                    &[perm_pred2.len()],
                    torsh_core::device::DeviceType::Cpu,
                ),
                torsh_tensor::creation::from_vec(
                    perm_target2.to_vec(),
                    &[perm_target2.len()],
                    torsh_core::device::DeviceType::Cpu,
                ),
            ) {
                let perm_score1 = metric.compute(&perm_tensor1_pred, &perm_tensor1_target);
                let perm_score2 = metric.compute(&perm_tensor2_pred, &perm_tensor2_target);
                let perm_diff = (perm_score1 - perm_score2).abs();

                if perm_diff >= observed_diff {
                    extreme_count += 1;
                }
            }
        }

        let p_value = extreme_count as f64 / self.n_permutations as f64;
        let effect_size = Some(observed_diff); // Simple effect size

        HypothesisTestResult {
            statistic: observed_diff,
            p_value,
            is_significant: p_value < alpha,
            alpha,
            test_type: "permutation_test".to_string(),
            effect_size,
        }
    }
}

/// McNemar's test for comparing two classifiers
pub struct McNemarTest;

impl McNemarTest {
    /// Perform McNemar's test
    pub fn test(
        predictions1: &Tensor,
        predictions2: &Tensor,
        targets: &Tensor,
        alpha: f64,
    ) -> HypothesisTestResult {
        match (
            predictions1.to_vec(),
            predictions2.to_vec(),
            targets.to_vec(),
        ) {
            (Ok(pred1_vec), Ok(pred2_vec), Ok(target_vec)) => {
                if pred1_vec.len() != pred2_vec.len()
                    || pred1_vec.len() != target_vec.len()
                    || pred1_vec.is_empty()
                {
                    return HypothesisTestResult {
                        statistic: 0.0,
                        p_value: 1.0,
                        is_significant: false,
                        alpha,
                        test_type: "mcnemar_test".to_string(),
                        effect_size: None,
                    };
                }

                // Count disagreement cases
                let mut b = 0; // Model 1 correct, Model 2 wrong
                let mut c = 0; // Model 1 wrong, Model 2 correct

                for i in 0..pred1_vec.len() {
                    let pred1_correct = (pred1_vec[i] - target_vec[i]).abs() < 0.5;
                    let pred2_correct = (pred2_vec[i] - target_vec[i]).abs() < 0.5;

                    match (pred1_correct, pred2_correct) {
                        (true, false) => b += 1,
                        (false, true) => c += 1,
                        _ => {} // Both correct or both wrong
                    }
                }

                // McNemar's test statistic with continuity correction
                let statistic = if b + c > 0 {
                    ((b as f64 - c as f64).abs() - 0.5).powi(2) / (b + c) as f64
                } else {
                    0.0
                };

                // Chi-square distribution with 1 degree of freedom
                // Approximate p-value using simple approximation
                let p_value = if statistic > 3.841 {
                    0.05
                } else if statistic > 6.635 {
                    0.01
                } else {
                    1.0
                };

                HypothesisTestResult {
                    statistic,
                    p_value,
                    is_significant: p_value < alpha,
                    alpha,
                    test_type: "mcnemar_test".to_string(),
                    effect_size: Some((b as f64 - c as f64) / (b + c) as f64),
                }
            }
            _ => HypothesisTestResult {
                statistic: 0.0,
                p_value: 1.0,
                is_significant: false,
                alpha,
                test_type: "mcnemar_test".to_string(),
                effect_size: None,
            },
        }
    }
}

/// Cross-validation evaluator
pub struct CrossValidator {
    n_folds: usize,
    shuffle: bool,
    random_seed: Option<u64>,
    stratify: bool,
}

impl CrossValidator {
    /// Create a new cross-validator
    pub fn new(n_folds: usize) -> Self {
        Self {
            n_folds,
            shuffle: true,
            random_seed: None,
            stratify: false,
        }
    }

    /// Set whether to shuffle data before folding
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Enable stratified cross-validation
    pub fn with_stratification(mut self, stratify: bool) -> Self {
        self.stratify = stratify;
        self
    }

    /// Perform cross-validation evaluation
    pub fn evaluate<M: Metric + Clone>(
        &self,
        metric: &M,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> CrossValidationResult {
        let pred_vec = predictions.to_vec().unwrap_or_default();
        let target_vec = targets.to_vec().unwrap_or_default();

        if pred_vec.is_empty() || target_vec.is_empty() || pred_vec.len() != target_vec.len() {
            return CrossValidationResult {
                cv_scores: vec![],
                mean_score: 0.0,
                std_score: 0.0,
                confidence_interval: (0.0, 0.0),
                fold_metrics: vec![],
            };
        }

        let indices = self.create_folds(&target_vec);
        let mut cv_scores = Vec::new();
        let mut fold_metrics = Vec::new();

        for fold_indices in indices {
            // Create test fold
            let test_pred: Vec<f32> = fold_indices.iter().map(|&i| pred_vec[i]).collect();
            let test_target: Vec<f32> = fold_indices.iter().map(|&i| target_vec[i]).collect();

            if let (Ok(test_pred_tensor), Ok(test_target_tensor)) = (
                torsh_tensor::creation::from_vec(
                    test_pred,
                    &[fold_indices.len()],
                    torsh_core::device::DeviceType::Cpu,
                ),
                torsh_tensor::creation::from_vec(
                    test_target,
                    &[fold_indices.len()],
                    torsh_core::device::DeviceType::Cpu,
                ),
            ) {
                let score = metric.compute(&test_pred_tensor, &test_target_tensor);
                if score.is_finite() {
                    cv_scores.push(score);

                    // Store detailed metrics for this fold
                    let mut fold_result = HashMap::new();
                    fold_result.insert(metric.name().to_string(), score);
                    fold_metrics.push(fold_result);
                }
            }
        }

        let mean_score = if cv_scores.is_empty() {
            0.0
        } else {
            cv_scores.iter().sum::<f64>() / cv_scores.len() as f64
        };

        let std_score = if cv_scores.len() > 1 {
            let variance: f64 = cv_scores
                .iter()
                .map(|x| (x - mean_score).powi(2))
                .sum::<f64>()
                / (cv_scores.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // 95% confidence interval using t-distribution approximation
        let ci_margin = if cv_scores.len() > 1 {
            1.96 * std_score / (cv_scores.len() as f64).sqrt()
        } else {
            0.0
        };

        CrossValidationResult {
            cv_scores,
            mean_score,
            std_score,
            confidence_interval: (mean_score - ci_margin, mean_score + ci_margin),
            fold_metrics,
        }
    }

    fn create_folds(&self, targets: &[f32]) -> Vec<Vec<usize>> {
        let n = targets.len();
        let mut indices: Vec<usize> = (0..n).collect();

        if self.shuffle {
            let mut rng = Random::seed(self.random_seed.unwrap_or(42));
            for i in (1..n).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }
        }

        let fold_size = n / self.n_folds;
        let remainder = n % self.n_folds;

        let mut folds = Vec::new();
        let mut start = 0;

        for i in 0..self.n_folds {
            let current_fold_size = fold_size + if i < remainder { 1 } else { 0 };
            let end = start + current_fold_size;

            if end <= n {
                folds.push(indices[start..end].to_vec());
            }

            start = end;
        }

        folds
    }
}

/// Statistical significance tester for multiple metrics
pub struct MultipleComparisonCorrection;

impl MultipleComparisonCorrection {
    /// Apply Bonferroni correction to p-values
    pub fn bonferroni_correction(p_values: &[f64]) -> Vec<f64> {
        let n = p_values.len() as f64;
        p_values.iter().map(|p| (p * n).min(1.0)).collect()
    }

    /// Apply Benjamini-Hochberg (FDR) correction
    pub fn benjamini_hochberg_correction(p_values: &[f64]) -> Vec<f64> {
        if p_values.is_empty() {
            return vec![];
        }

        let n = p_values.len();
        let mut indexed_p: Vec<(usize, f64)> =
            p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();

        // Sort by p-value
        indexed_p.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut corrected = vec![0.0; n];

        // Apply BH correction
        let mut min_val = 1.0;
        for i in (0..n).rev() {
            let (original_idx, p_val) = indexed_p[i];
            let corrected_p = (p_val * n as f64 / (i + 1) as f64).min(min_val);
            corrected[original_idx] = corrected_p;
            min_val = corrected_p;
        }

        corrected
    }
}

/// Effect size calculator
pub struct EffectSize;

impl EffectSize {
    /// Calculate Cohen's d for continuous metrics
    pub fn cohens_d(group1: &[f64], group2: &[f64]) -> f64 {
        if group1.is_empty() || group2.is_empty() {
            return 0.0;
        }

        let mean1: f64 = group1.iter().sum::<f64>() / group1.len() as f64;
        let mean2: f64 = group2.iter().sum::<f64>() / group2.len() as f64;

        let var1: f64 =
            group1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (group1.len() - 1) as f64;
        let var2: f64 =
            group2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (group2.len() - 1) as f64;

        let pooled_std = ((var1 * (group1.len() - 1) as f64 + var2 * (group2.len() - 1) as f64)
            / (group1.len() + group2.len() - 2) as f64)
            .sqrt();

        if pooled_std > 1e-10 {
            (mean1 - mean2) / pooled_std
        } else {
            0.0
        }
    }

    /// Calculate Cliff's delta (robust effect size)
    pub fn cliffs_delta(group1: &[f64], group2: &[f64]) -> f64 {
        if group1.is_empty() || group2.is_empty() {
            return 0.0;
        }

        let mut dominance = 0;
        let total_pairs = group1.len() * group2.len();

        for &x1 in group1 {
            for &x2 in group2 {
                if x1 > x2 {
                    dominance += 1;
                } else if x1 < x2 {
                    dominance -= 1;
                }
                // Ties contribute 0
            }
        }

        dominance as f64 / total_pairs as f64
    }
}

/// Metric stability analyzer
pub struct StabilityAnalyzer {
    n_trials: usize,
    subsample_ratio: f64,
    random_seed: Option<u64>,
}

impl StabilityAnalyzer {
    /// Create a new stability analyzer
    pub fn new(n_trials: usize, subsample_ratio: f64) -> Self {
        Self {
            n_trials,
            subsample_ratio,
            random_seed: None,
        }
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Analyze metric stability across different subsamples
    pub fn analyze_stability<M: Metric + Clone>(
        &self,
        metric: &M,
        predictions: &Tensor,
        targets: &Tensor,
    ) -> (f64, f64, Vec<f64>) {
        let pred_vec = predictions.to_vec().unwrap_or_default();
        let target_vec = targets.to_vec().unwrap_or_default();

        if pred_vec.is_empty() || target_vec.is_empty() || pred_vec.len() != target_vec.len() {
            return (0.0, 0.0, vec![]);
        }

        let mut rng = Random::seed(self.random_seed.unwrap_or(42));
        let n_samples = pred_vec.len();
        let subsample_size = (n_samples as f64 * self.subsample_ratio) as usize;

        let mut trial_scores = Vec::new();

        for _ in 0..self.n_trials {
            // Create random subsample
            let mut indices: Vec<usize> = (0..n_samples).collect();
            for i in (1..n_samples).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }

            let subsample_pred: Vec<f32> = indices[..subsample_size]
                .iter()
                .map(|&i| pred_vec[i])
                .collect();
            let subsample_target: Vec<f32> = indices[..subsample_size]
                .iter()
                .map(|&i| target_vec[i])
                .collect();

            if let (Ok(pred_tensor), Ok(target_tensor)) = (
                torsh_tensor::creation::from_vec(
                    subsample_pred,
                    &[subsample_size],
                    torsh_core::device::DeviceType::Cpu,
                ),
                torsh_tensor::creation::from_vec(
                    subsample_target,
                    &[subsample_size],
                    torsh_core::device::DeviceType::Cpu,
                ),
            ) {
                let score = metric.compute(&pred_tensor, &target_tensor);
                if score.is_finite() {
                    trial_scores.push(score);
                }
            }
        }

        if trial_scores.is_empty() {
            return (0.0, 0.0, vec![]);
        }

        let mean_score: f64 = trial_scores.iter().sum::<f64>() / trial_scores.len() as f64;
        let std_score: f64 = if trial_scores.len() > 1 {
            let variance: f64 = trial_scores
                .iter()
                .map(|x| (x - mean_score).powi(2))
                .sum::<f64>()
                / (trial_scores.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        (mean_score, std_score, trial_scores)
    }
}
