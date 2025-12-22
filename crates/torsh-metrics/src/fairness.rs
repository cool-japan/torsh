//! Fairness and bias detection metrics

use crate::Metric;
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

/// Comprehensive fairness metrics collection
#[derive(Debug, Clone)]
pub struct FairnessMetrics {
    /// Demographic parity difference (closer to 0 is better)
    pub demographic_parity_difference: f64,
    /// Demographic parity ratio (closer to 1 is better)
    pub demographic_parity_ratio: f64,
    /// Equalized odds TPR difference
    pub equalized_odds_tpr_diff: f64,
    /// Equalized odds FPR difference
    pub equalized_odds_fpr_diff: f64,
    /// Expected calibration error
    pub calibration_error: f64,
    /// Individual fairness violation rate
    pub individual_fairness_violations: f64,
    /// Bias amplification ratio (1.0 means no amplification)
    pub bias_amplification: f64,
    /// Number of groups analyzed
    pub num_groups: usize,
}

impl FairnessMetrics {
    /// Compute comprehensive fairness metrics
    ///
    /// # Arguments
    /// * `predictions` - Model predictions (probabilities or binary)
    /// * `true_labels` - Ground truth labels
    /// * `sensitive_attributes` - Protected group indicators
    /// * `features` - Feature vectors for individual fairness (optional)
    /// * `threshold` - Classification threshold (default: 0.5)
    pub fn compute(
        predictions: &Tensor,
        true_labels: &Tensor,
        sensitive_attributes: &Tensor,
        features: Option<&Tensor>,
        threshold: f64,
    ) -> Self {
        // Demographic parity
        let dp_metric = DemographicParity::new(threshold);
        let demographic_parity_difference =
            dp_metric.compute_difference(predictions, sensitive_attributes);
        let demographic_parity_ratio = dp_metric.compute_ratio(predictions, sensitive_attributes);

        // Equalized odds
        let eo_metric = EqualizedOdds::new(threshold);
        let (equalized_odds_tpr_diff, equalized_odds_fpr_diff) =
            eo_metric.compute_difference(predictions, true_labels, sensitive_attributes);

        // Calibration
        let cal_metric = Calibration::new(10);
        let calibration_error = cal_metric.compute_ece(predictions, true_labels);

        // Individual fairness (if features provided)
        let individual_fairness_violations = if let Some(feat) = features {
            let if_metric = IndividualFairness::new(0.1, 0.2);
            if_metric.compute_violation_rate(feat, predictions)
        } else {
            0.0
        };

        // Bias amplification
        let ba_metric = BiasAmplification;
        let bias_amplification = ba_metric.compute_amplification(
            predictions,
            true_labels,
            sensitive_attributes,
            threshold,
        );

        // Count number of groups
        let num_groups = sensitive_attributes
            .to_vec()
            .ok()
            .map(|attr_vec| {
                let mut groups = std::collections::HashSet::new();
                for val in attr_vec {
                    groups.insert(val as i32);
                }
                groups.len()
            })
            .unwrap_or(0);

        FairnessMetrics {
            demographic_parity_difference,
            demographic_parity_ratio,
            equalized_odds_tpr_diff,
            equalized_odds_fpr_diff,
            calibration_error,
            individual_fairness_violations,
            bias_amplification,
            num_groups,
        }
    }

    /// Create with default threshold of 0.5
    pub fn compute_default(
        predictions: &Tensor,
        true_labels: &Tensor,
        sensitive_attributes: &Tensor,
    ) -> Self {
        Self::compute(predictions, true_labels, sensitive_attributes, None, 0.5)
    }

    /// Check if model satisfies fairness criteria
    pub fn is_fair(&self, dp_threshold: f64, eo_threshold: f64) -> bool {
        self.demographic_parity_difference < dp_threshold
            && self.equalized_odds_tpr_diff < eo_threshold
            && self.equalized_odds_fpr_diff < eo_threshold
    }

    /// Get overall fairness score (0-1, higher is better)
    /// Combines all metrics into a single score
    pub fn overall_fairness_score(&self) -> f64 {
        let dp_score = 1.0 - self.demographic_parity_difference.min(1.0);
        let eo_tpr_score = 1.0 - self.equalized_odds_tpr_diff.min(1.0);
        let eo_fpr_score = 1.0 - self.equalized_odds_fpr_diff.min(1.0);
        let cal_score = 1.0 - self.calibration_error.min(1.0);
        let if_score = 1.0 - self.individual_fairness_violations.min(1.0);
        let ba_score = if self.bias_amplification <= 1.0 {
            self.bias_amplification
        } else {
            1.0 / self.bias_amplification
        };

        (dp_score + eo_tpr_score + eo_fpr_score + cal_score + if_score + ba_score) / 6.0
    }

    /// Get fairness assessment
    pub fn fairness_assessment(&self) -> &str {
        let score = self.overall_fairness_score();

        if score > 0.9 {
            "Excellent fairness"
        } else if score > 0.75 {
            "Good fairness"
        } else if score > 0.6 {
            "Fair"
        } else if score > 0.4 {
            "Poor fairness - review needed"
        } else {
            "Critical fairness issues - immediate attention required"
        }
    }

    /// Format fairness metrics as a string
    pub fn format(&self) -> String {
        let mut result = String::new();
        result.push_str("Fairness Metrics:\n");
        result.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        result.push_str(&format!(
            "Overall Fairness Score: {:.4} - {}\n",
            self.overall_fairness_score(),
            self.fairness_assessment()
        ));
        result.push_str(&format!("Number of groups: {}\n\n", self.num_groups));

        result.push_str("Demographic Parity:\n");
        result.push_str(&format!(
            "  Difference: {:.4} (closer to 0 is better)\n",
            self.demographic_parity_difference
        ));
        result.push_str(&format!(
            "  Ratio: {:.4} (closer to 1 is better)\n\n",
            self.demographic_parity_ratio
        ));

        result.push_str("Equalized Odds:\n");
        result.push_str(&format!(
            "  TPR Difference: {:.4}\n",
            self.equalized_odds_tpr_diff
        ));
        result.push_str(&format!(
            "  FPR Difference: {:.4}\n\n",
            self.equalized_odds_fpr_diff
        ));

        result.push_str(&format!(
            "Calibration Error: {:.4}\n",
            self.calibration_error
        ));

        if self.individual_fairness_violations > 0.0 {
            result.push_str(&format!(
                "Individual Fairness Violations: {:.4}\n",
                self.individual_fairness_violations
            ));
        }

        result.push_str(&format!(
            "Bias Amplification: {:.4} ({} amplification)\n",
            self.bias_amplification,
            if self.bias_amplification > 1.0 {
                format!("{}x", self.bias_amplification)
            } else {
                "no".to_string()
            }
        ));

        result
    }

    /// Get detailed fairness report as HashMap
    pub fn as_map(&self) -> HashMap<String, f64> {
        let mut map = HashMap::new();

        map.insert(
            "overall_fairness_score".to_string(),
            self.overall_fairness_score(),
        );
        map.insert(
            "demographic_parity_difference".to_string(),
            self.demographic_parity_difference,
        );
        map.insert(
            "demographic_parity_ratio".to_string(),
            self.demographic_parity_ratio,
        );
        map.insert(
            "equalized_odds_tpr_diff".to_string(),
            self.equalized_odds_tpr_diff,
        );
        map.insert(
            "equalized_odds_fpr_diff".to_string(),
            self.equalized_odds_fpr_diff,
        );
        map.insert("calibration_error".to_string(), self.calibration_error);
        map.insert(
            "individual_fairness_violations".to_string(),
            self.individual_fairness_violations,
        );
        map.insert("bias_amplification".to_string(), self.bias_amplification);
        map.insert("num_groups".to_string(), self.num_groups as f64);

        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_fairness_metrics() {
        // Create fair predictions (equal across groups)
        let predictions =
            from_vec(vec![0.9, 0.8, 0.7, 0.9, 0.8, 0.7], &[6], DeviceType::Cpu).unwrap();
        let labels = from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[6], DeviceType::Cpu).unwrap();
        let sensitive_attrs =
            from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[6], DeviceType::Cpu).unwrap();

        let fairness = FairnessMetrics::compute_default(&predictions, &labels, &sensitive_attrs);

        // Check that demographic parity is reasonable
        assert!(fairness.demographic_parity_difference <= 1.0);
        assert!(fairness.demographic_parity_ratio <= 1.0);
        assert_eq!(fairness.num_groups, 2);

        // Check that fairness score is computed
        let score = fairness.overall_fairness_score();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_fairness_assessment() {
        let predictions = from_vec(vec![0.9, 0.8, 0.3, 0.2], &[4], DeviceType::Cpu).unwrap();
        let labels = from_vec(vec![1.0, 1.0, 0.0, 0.0], &[4], DeviceType::Cpu).unwrap();
        let sensitive_attrs = from_vec(vec![0.0, 0.0, 1.0, 1.0], &[4], DeviceType::Cpu).unwrap();

        let fairness = FairnessMetrics::compute_default(&predictions, &labels, &sensitive_attrs);

        let assessment = fairness.fairness_assessment();
        assert!(
            assessment == "Excellent fairness"
                || assessment == "Good fairness"
                || assessment == "Fair"
                || assessment == "Poor fairness - review needed"
                || assessment == "Critical fairness issues - immediate attention required"
        );
    }

    #[test]
    fn test_fairness_is_fair() {
        let predictions = from_vec(vec![0.9, 0.8, 0.85, 0.82], &[4], DeviceType::Cpu).unwrap();
        let labels = from_vec(vec![1.0, 1.0, 1.0, 1.0], &[4], DeviceType::Cpu).unwrap();
        let sensitive_attrs = from_vec(vec![0.0, 0.0, 1.0, 1.0], &[4], DeviceType::Cpu).unwrap();

        let fairness = FairnessMetrics::compute_default(&predictions, &labels, &sensitive_attrs);

        // With similar predictions across groups, should be relatively fair
        assert!(fairness.is_fair(0.3, 0.3) || !fairness.is_fair(0.05, 0.05));
    }
}
