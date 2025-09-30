//! Fairness and bias detection metrics

use crate::Metric;
use scirs2_core::random::{Random, Rng};
use std::collections::HashMap;
use torsh_tensor::Tensor;

/// Demographic Parity metric
/// Measures if positive prediction rates are similar across groups
pub struct DemographicParity {
    threshold: f64,
}

impl DemographicParity {
    /// Create a new Demographic Parity metric
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Compute demographic parity difference
    /// Returns |P(Y=1|A=0) - P(Y=1|A=1)| where A is sensitive attribute
    pub fn compute_difference(&self, predictions: &Tensor, sensitive_attributes: &Tensor) -> f64 {
        match (predictions.to_vec(), sensitive_attributes.to_vec()) {
            (Ok(pred_vec), Ok(attr_vec)) => {
                if pred_vec.len() != attr_vec.len() || pred_vec.is_empty() {
                    return 0.0;
                }

                // Group by sensitive attribute
                let mut group_stats = HashMap::new();

                for i in 0..pred_vec.len() {
                    let group = attr_vec[i] as i32;
                    let prediction = pred_vec[i] >= self.threshold as f32;

                    let stats = group_stats.entry(group).or_insert((0, 0)); // (positive, total)
                    if prediction {
                        stats.0 += 1;
                    }
                    stats.1 += 1;
                }

                // Calculate positive rates for each group
                let mut rates = Vec::new();
                for (_, (positive, total)) in group_stats {
                    if total > 0 {
                        rates.push(positive as f64 / total as f64);
                    }
                }

                if rates.len() < 2 {
                    return 0.0;
                }

                // Return maximum difference between any two groups
                let min_rate = rates.iter().copied().fold(f64::INFINITY, f64::min);
                let max_rate = rates.iter().copied().fold(f64::NEG_INFINITY, f64::max);

                max_rate - min_rate
            }
            _ => 0.0,
        }
    }

    /// Compute demographic parity ratio
    /// Returns min(P(Y=1|A=0), P(Y=1|A=1)) / max(P(Y=1|A=0), P(Y=1|A=1))
    pub fn compute_ratio(&self, predictions: &Tensor, sensitive_attributes: &Tensor) -> f64 {
        match (predictions.to_vec(), sensitive_attributes.to_vec()) {
            (Ok(pred_vec), Ok(attr_vec)) => {
                if pred_vec.len() != attr_vec.len() || pred_vec.is_empty() {
                    return 1.0;
                }

                let mut group_stats = HashMap::new();

                for i in 0..pred_vec.len() {
                    let group = attr_vec[i] as i32;
                    let prediction = pred_vec[i] >= self.threshold as f32;

                    let stats = group_stats.entry(group).or_insert((0, 0));
                    if prediction {
                        stats.0 += 1;
                    }
                    stats.1 += 1;
                }

                let mut rates = Vec::new();
                for (_, (positive, total)) in group_stats {
                    if total > 0 {
                        rates.push(positive as f64 / total as f64);
                    }
                }

                if rates.len() < 2 {
                    return 1.0;
                }

                let min_rate = rates.iter().copied().fold(f64::INFINITY, f64::min);
                let max_rate = rates.iter().copied().fold(f64::NEG_INFINITY, f64::max);

                if max_rate > 0.0 {
                    min_rate / max_rate
                } else {
                    1.0
                }
            }
            _ => 1.0,
        }
    }
}

impl Metric for DemographicParity {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        // Assumes targets contains sensitive attributes for this metric
        self.compute_difference(predictions, targets)
    }

    fn name(&self) -> &str {
        "demographic_parity"
    }
}

/// Equalized Odds metric
/// Measures if true positive and false positive rates are similar across groups
pub struct EqualizedOdds {
    threshold: f64,
}

impl EqualizedOdds {
    /// Create a new Equalized Odds metric
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Compute equalized odds difference
    pub fn compute_difference(
        &self,
        predictions: &Tensor,
        true_labels: &Tensor,
        sensitive_attributes: &Tensor,
    ) -> (f64, f64) {
        // Returns (TPR difference, FPR difference)
        match (
            predictions.to_vec(),
            true_labels.to_vec(),
            sensitive_attributes.to_vec(),
        ) {
            (Ok(pred_vec), Ok(label_vec), Ok(attr_vec)) => {
                if pred_vec.len() != label_vec.len()
                    || pred_vec.len() != attr_vec.len()
                    || pred_vec.is_empty()
                {
                    return (0.0, 0.0);
                }

                // Group by sensitive attribute and compute confusion matrix for each
                let mut group_stats = HashMap::new();

                for i in 0..pred_vec.len() {
                    let group = attr_vec[i] as i32;
                    let prediction = pred_vec[i] >= self.threshold as f32;
                    let true_label = label_vec[i] >= 0.5;

                    let stats = group_stats.entry(group).or_insert((0, 0, 0, 0)); // (TP, TN, FP, FN)

                    match (prediction, true_label) {
                        (true, true) => stats.0 += 1,   // TP
                        (false, false) => stats.1 += 1, // TN
                        (true, false) => stats.2 += 1,  // FP
                        (false, true) => stats.3 += 1,  // FN
                    }
                }

                // Calculate TPR and FPR for each group
                let mut tprs = Vec::new();
                let mut fprs = Vec::new();

                for (_, (tp, tn, fp, fn_count)) in group_stats {
                    let tpr = if tp + fn_count > 0 {
                        tp as f64 / (tp + fn_count) as f64
                    } else {
                        0.0
                    };

                    let fpr = if fp + tn > 0 {
                        fp as f64 / (fp + tn) as f64
                    } else {
                        0.0
                    };

                    tprs.push(tpr);
                    fprs.push(fpr);
                }

                let tpr_diff = if tprs.len() >= 2 {
                    let min_tpr = tprs.iter().copied().fold(f64::INFINITY, f64::min);
                    let max_tpr = tprs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    max_tpr - min_tpr
                } else {
                    0.0
                };

                let fpr_diff = if fprs.len() >= 2 {
                    let min_fpr = fprs.iter().copied().fold(f64::INFINITY, f64::min);
                    let max_fpr = fprs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    max_fpr - min_fpr
                } else {
                    0.0
                };

                (tpr_diff, fpr_diff)
            }
            _ => (0.0, 0.0),
        }
    }
}

impl Metric for EqualizedOdds {
    fn compute(&self, _predictions: &Tensor, _targets: &Tensor) -> f64 {
        // This metric requires three inputs, so this is a placeholder
        0.0
    }

    fn name(&self) -> &str {
        "equalized_odds"
    }
}

/// Calibration metric
/// Measures if predicted probabilities match actual outcome frequencies
pub struct Calibration {
    n_bins: usize,
}

impl Calibration {
    /// Create a new Calibration metric
    pub fn new(n_bins: usize) -> Self {
        Self { n_bins }
    }

    /// Compute Expected Calibration Error (ECE)
    pub fn compute_ece(&self, predictions: &Tensor, true_labels: &Tensor) -> f64 {
        match (predictions.to_vec(), true_labels.to_vec()) {
            (Ok(pred_vec), Ok(label_vec)) => {
                if pred_vec.len() != label_vec.len() || pred_vec.is_empty() {
                    return 0.0;
                }

                let n = pred_vec.len();
                let mut bins = vec![(0.0, 0, 0); self.n_bins]; // (sum_predictions, num_correct, num_total)

                // Assign predictions to bins
                for i in 0..n {
                    let pred = pred_vec[i] as f64;
                    let label = label_vec[i] >= 0.5;

                    let bin_idx =
                        ((pred * self.n_bins as f64).floor() as usize).min(self.n_bins - 1);

                    bins[bin_idx].0 += pred;
                    if label {
                        bins[bin_idx].1 += 1;
                    }
                    bins[bin_idx].2 += 1;
                }

                // Compute ECE
                let mut ece = 0.0;

                for (sum_pred, num_correct, num_total) in bins {
                    if num_total > 0 {
                        let avg_pred = sum_pred / num_total as f64;
                        let avg_accuracy = num_correct as f64 / num_total as f64;
                        let weight = num_total as f64 / n as f64;

                        ece += weight * (avg_pred - avg_accuracy).abs();
                    }
                }

                ece
            }
            _ => 0.0,
        }
    }

    /// Compute Maximum Calibration Error (MCE)
    pub fn compute_mce(&self, predictions: &Tensor, true_labels: &Tensor) -> f64 {
        match (predictions.to_vec(), true_labels.to_vec()) {
            (Ok(pred_vec), Ok(label_vec)) => {
                if pred_vec.len() != label_vec.len() || pred_vec.is_empty() {
                    return 0.0;
                }

                let n = pred_vec.len();
                let mut bins = vec![(0.0, 0, 0); self.n_bins];

                for i in 0..n {
                    let pred = pred_vec[i] as f64;
                    let label = label_vec[i] >= 0.5;

                    let bin_idx =
                        ((pred * self.n_bins as f64).floor() as usize).min(self.n_bins - 1);

                    bins[bin_idx].0 += pred;
                    if label {
                        bins[bin_idx].1 += 1;
                    }
                    bins[bin_idx].2 += 1;
                }

                let mut max_error: f64 = 0.0;

                for (sum_pred, num_correct, num_total) in bins {
                    if num_total > 0 {
                        let avg_pred = sum_pred / num_total as f64;
                        let avg_accuracy = num_correct as f64 / num_total as f64;
                        let error = (avg_pred - avg_accuracy).abs();

                        max_error = max_error.max(error);
                    }
                }

                max_error
            }
            _ => 0.0,
        }
    }
}

impl Metric for Calibration {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_ece(predictions, targets)
    }

    fn name(&self) -> &str {
        "calibration_ece"
    }
}

/// Individual Fairness metric
/// Measures if similar individuals receive similar outcomes
pub struct IndividualFairness {
    distance_threshold: f64,
    outcome_threshold: f64,
}

impl IndividualFairness {
    /// Create a new Individual Fairness metric
    pub fn new(distance_threshold: f64, outcome_threshold: f64) -> Self {
        Self {
            distance_threshold,
            outcome_threshold,
        }
    }

    /// Compute individual fairness violation rate
    /// Features should be normalized
    pub fn compute_violation_rate(&self, features: &Tensor, predictions: &Tensor) -> f64 {
        match (features.to_vec(), predictions.to_vec()) {
            (Ok(feat_vec), Ok(pred_vec)) => {
                let feat_shape = features.shape();
                let feat_dims = feat_shape.dims();

                if feat_dims.len() != 2 || feat_dims[0] != pred_vec.len() {
                    return 0.0;
                }

                let n_samples = feat_dims[0];
                let n_features = feat_dims[1];

                if n_samples == 0 || n_features == 0 {
                    return 0.0;
                }

                let mut violations = 0;
                let mut total_pairs = 0;

                // Check all pairs of individuals
                for i in 0..n_samples {
                    for j in (i + 1)..n_samples {
                        // Compute feature distance
                        let mut distance = 0.0;
                        for k in 0..n_features {
                            let diff = feat_vec[i * n_features + k] - feat_vec[j * n_features + k];
                            distance += (diff * diff) as f64;
                        }
                        distance = distance.sqrt();

                        // If individuals are similar (small distance)
                        if distance <= self.distance_threshold {
                            // Check if outcomes are dissimilar
                            let outcome_diff = (pred_vec[i] - pred_vec[j]).abs() as f64;
                            if outcome_diff > self.outcome_threshold {
                                violations += 1;
                            }
                            total_pairs += 1;
                        }
                    }
                }

                if total_pairs > 0 {
                    violations as f64 / total_pairs as f64
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }
}

impl Metric for IndividualFairness {
    fn compute(&self, predictions: &Tensor, features: &Tensor) -> f64 {
        self.compute_violation_rate(features, predictions)
    }

    fn name(&self) -> &str {
        "individual_fairness"
    }
}

/// Bias Amplification metric
/// Measures if the model amplifies existing biases in the data
pub struct BiasAmplification;

impl BiasAmplification {
    /// Compute bias amplification ratio
    /// Returns ratio of bias in predictions vs bias in ground truth
    pub fn compute_amplification(
        &self,
        predictions: &Tensor,
        true_labels: &Tensor,
        sensitive_attributes: &Tensor,
        threshold: f64,
    ) -> f64 {
        match (
            predictions.to_vec(),
            true_labels.to_vec(),
            sensitive_attributes.to_vec(),
        ) {
            (Ok(pred_vec), Ok(label_vec), Ok(attr_vec)) => {
                if pred_vec.len() != label_vec.len()
                    || pred_vec.len() != attr_vec.len()
                    || pred_vec.is_empty()
                {
                    return 1.0;
                }

                // Calculate bias in ground truth labels
                let mut gt_group_stats = HashMap::new();
                let mut pred_group_stats = HashMap::new();

                for i in 0..pred_vec.len() {
                    let group = attr_vec[i] as i32;
                    let true_positive = label_vec[i] >= 0.5;
                    let pred_positive = pred_vec[i] >= threshold as f32;

                    // Ground truth stats
                    let gt_stats = gt_group_stats.entry(group).or_insert((0, 0));
                    if true_positive {
                        gt_stats.0 += 1;
                    }
                    gt_stats.1 += 1;

                    // Prediction stats
                    let pred_stats = pred_group_stats.entry(group).or_insert((0, 0));
                    if pred_positive {
                        pred_stats.0 += 1;
                    }
                    pred_stats.1 += 1;
                }

                // Calculate positive rates for each group
                let gt_rates: Vec<f64> = gt_group_stats
                    .values()
                    .map(|(pos, total)| {
                        if *total > 0 {
                            *pos as f64 / *total as f64
                        } else {
                            0.0
                        }
                    })
                    .collect();

                let pred_rates: Vec<f64> = pred_group_stats
                    .values()
                    .map(|(pos, total)| {
                        if *total > 0 {
                            *pos as f64 / *total as f64
                        } else {
                            0.0
                        }
                    })
                    .collect();

                if gt_rates.len() < 2 || pred_rates.len() < 2 {
                    return 1.0;
                }

                // Calculate bias as difference between max and min rates
                let gt_bias = gt_rates.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                    - gt_rates.iter().copied().fold(f64::INFINITY, f64::min);

                let pred_bias = pred_rates.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                    - pred_rates.iter().copied().fold(f64::INFINITY, f64::min);

                if gt_bias > 0.0 {
                    pred_bias / gt_bias
                } else if pred_bias > 0.0 {
                    f64::INFINITY // Bias introduced where none existed
                } else {
                    1.0
                }
            }
            _ => 1.0,
        }
    }
}

impl Metric for BiasAmplification {
    fn compute(&self, _predictions: &Tensor, _targets: &Tensor) -> f64 {
        // This metric requires special handling
        1.0
    }

    fn name(&self) -> &str {
        "bias_amplification"
    }
}
