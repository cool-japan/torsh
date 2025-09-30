//! Uncertainty quantification metrics

use crate::Metric;
use scirs2_core::random::{Random, Rng};
use torsh_tensor::Tensor;

/// Expected Calibration Error for uncertainty calibration
pub struct ExpectedCalibrationError {
    n_bins: usize,
    norm: String,
}

impl ExpectedCalibrationError {
    /// Create a new ECE metric
    pub fn new(n_bins: usize) -> Self {
        Self {
            n_bins,
            norm: "l1".to_string(),
        }
    }

    /// Set norm type (l1 or l2)
    pub fn with_norm(mut self, norm: &str) -> Self {
        self.norm = norm.to_string();
        self
    }

    /// Compute ECE from predicted probabilities and true labels
    pub fn compute_ece(&self, confidences: &Tensor, predictions: &Tensor, labels: &Tensor) -> f64 {
        match (confidences.to_vec(), predictions.to_vec(), labels.to_vec()) {
            (Ok(conf_vec), Ok(pred_vec), Ok(label_vec)) => {
                if conf_vec.len() != pred_vec.len()
                    || conf_vec.len() != label_vec.len()
                    || conf_vec.is_empty()
                {
                    return 0.0;
                }

                let n = conf_vec.len();
                let mut bins = vec![(0.0, 0, 0); self.n_bins]; // (sum_confidence, num_correct, num_total)

                // Assign predictions to bins based on confidence
                for i in 0..n {
                    let confidence = conf_vec[i] as f64;
                    let predicted_class = pred_vec[i] as usize;
                    let true_class = label_vec[i] as usize;

                    let bin_idx =
                        ((confidence * self.n_bins as f64).floor() as usize).min(self.n_bins - 1);

                    bins[bin_idx].0 += confidence;
                    if predicted_class == true_class {
                        bins[bin_idx].1 += 1;
                    }
                    bins[bin_idx].2 += 1;
                }

                // Compute ECE
                let mut ece = 0.0;

                for (sum_conf, num_correct, num_total) in bins {
                    if num_total > 0 {
                        let avg_confidence = sum_conf / num_total as f64;
                        let avg_accuracy = num_correct as f64 / num_total as f64;
                        let weight = num_total as f64 / n as f64;

                        let error = (avg_confidence - avg_accuracy).abs();
                        ece += weight
                            * if self.norm == "l2" {
                                error * error
                            } else {
                                error
                            };
                    }
                }

                if self.norm == "l2" {
                    ece.sqrt()
                } else {
                    ece
                }
            }
            _ => 0.0,
        }
    }
}

impl Metric for ExpectedCalibrationError {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        // Assume predictions contains confidences and targets contains both preds and labels
        // This is a simplified interface - real usage would need separate confidences
        self.compute_ece(predictions, predictions, targets)
    }

    fn name(&self) -> &str {
        "expected_calibration_error"
    }
}

/// Maximum Calibration Error
pub struct MaximumCalibrationError {
    n_bins: usize,
}

impl MaximumCalibrationError {
    /// Create a new MCE metric
    pub fn new(n_bins: usize) -> Self {
        Self { n_bins }
    }

    /// Compute MCE from predicted probabilities and true labels
    pub fn compute_mce(&self, confidences: &Tensor, predictions: &Tensor, labels: &Tensor) -> f64 {
        match (confidences.to_vec(), predictions.to_vec(), labels.to_vec()) {
            (Ok(conf_vec), Ok(pred_vec), Ok(label_vec)) => {
                if conf_vec.len() != pred_vec.len()
                    || conf_vec.len() != label_vec.len()
                    || conf_vec.is_empty()
                {
                    return 0.0;
                }

                let n = conf_vec.len();
                let mut bins = vec![(0.0, 0, 0); self.n_bins];

                for i in 0..n {
                    let confidence = conf_vec[i] as f64;
                    let predicted_class = pred_vec[i] as usize;
                    let true_class = label_vec[i] as usize;

                    let bin_idx =
                        ((confidence * self.n_bins as f64).floor() as usize).min(self.n_bins - 1);

                    bins[bin_idx].0 += confidence;
                    if predicted_class == true_class {
                        bins[bin_idx].1 += 1;
                    }
                    bins[bin_idx].2 += 1;
                }

                let mut max_error: f64 = 0.0;

                for (sum_conf, num_correct, num_total) in bins {
                    if num_total > 0 {
                        let avg_confidence = sum_conf / num_total as f64;
                        let avg_accuracy = num_correct as f64 / num_total as f64;
                        let error = (avg_confidence - avg_accuracy).abs();

                        max_error = max_error.max(error);
                    }
                }

                max_error
            }
            _ => 0.0,
        }
    }
}

impl Metric for MaximumCalibrationError {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_mce(predictions, predictions, targets)
    }

    fn name(&self) -> &str {
        "maximum_calibration_error"
    }
}

/// Prediction Interval Coverage Probability (PICP)
/// Measures what fraction of true values fall within predicted intervals
pub struct PredictionIntervalCoverage {
    confidence_level: f64,
}

impl PredictionIntervalCoverage {
    /// Create a new PICP metric
    pub fn new(confidence_level: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&confidence_level),
            "Confidence level must be between 0 and 1"
        );
        Self { confidence_level }
    }

    /// Compute PICP from predicted intervals and true values
    pub fn compute_picp(
        &self,
        lower_bounds: &Tensor,
        upper_bounds: &Tensor,
        true_values: &Tensor,
    ) -> f64 {
        match (
            lower_bounds.to_vec(),
            upper_bounds.to_vec(),
            true_values.to_vec(),
        ) {
            (Ok(lower_vec), Ok(upper_vec), Ok(true_vec)) => {
                if lower_vec.len() != upper_vec.len()
                    || lower_vec.len() != true_vec.len()
                    || lower_vec.is_empty()
                {
                    return 0.0;
                }

                let mut covered = 0;
                let n = true_vec.len();

                for i in 0..n {
                    let lower = lower_vec[i] as f64;
                    let upper = upper_vec[i] as f64;
                    let true_val = true_vec[i] as f64;

                    if lower <= true_val && true_val <= upper {
                        covered += 1;
                    }
                }

                covered as f64 / n as f64
            }
            _ => 0.0,
        }
    }

    /// Compute Mean Prediction Interval Width (MPIW)
    pub fn compute_mpiw(&self, lower_bounds: &Tensor, upper_bounds: &Tensor) -> f64 {
        match (lower_bounds.to_vec(), upper_bounds.to_vec()) {
            (Ok(lower_vec), Ok(upper_vec)) => {
                if lower_vec.len() != upper_vec.len() || lower_vec.is_empty() {
                    return 0.0;
                }

                let total_width: f64 = lower_vec
                    .iter()
                    .zip(upper_vec.iter())
                    .map(|(&lower, &upper)| (upper - lower).abs() as f64)
                    .sum();

                total_width / lower_vec.len() as f64
            }
            _ => 0.0,
        }
    }
}

impl Metric for PredictionIntervalCoverage {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        // Assume predictions contains both lower and upper bounds (half each)
        let pred_vec = predictions.to_vec().unwrap_or_default();
        if pred_vec.len() % 2 != 0 {
            return 0.0;
        }

        let mid = pred_vec.len() / 2;
        let lower_vec = &pred_vec[..mid];
        let upper_vec = &pred_vec[mid..];

        match (
            torsh_tensor::creation::from_vec(
                lower_vec.to_vec(),
                &[lower_vec.len()],
                torsh_core::device::DeviceType::Cpu,
            ),
            torsh_tensor::creation::from_vec(
                upper_vec.to_vec(),
                &[upper_vec.len()],
                torsh_core::device::DeviceType::Cpu,
            ),
        ) {
            (Ok(lower_tensor), Ok(upper_tensor)) => {
                self.compute_picp(&lower_tensor, &upper_tensor, targets)
            }
            _ => 0.0,
        }
    }

    fn name(&self) -> &str {
        "prediction_interval_coverage"
    }
}

/// Brier Score for probabilistic predictions
pub struct BrierScore;

impl BrierScore {
    /// Compute Brier score
    pub fn compute_brier(&self, probabilities: &Tensor, labels: &Tensor) -> f64 {
        match (probabilities.to_vec(), labels.to_vec()) {
            (Ok(prob_vec), Ok(label_vec)) => {
                if prob_vec.len() != label_vec.len() || prob_vec.is_empty() {
                    return 0.0;
                }

                let mut score = 0.0;
                let n = label_vec.len();

                for i in 0..n {
                    let prob = prob_vec[i] as f64;
                    let label = if label_vec[i] >= 0.5 { 1.0 } else { 0.0 };
                    let diff = prob - label;
                    score += diff * diff;
                }

                score / n as f64
            }
            _ => 0.0,
        }
    }

    /// Compute multi-class Brier score
    pub fn compute_multiclass_brier(&self, probabilities: &Tensor, labels: &Tensor) -> f64 {
        match (probabilities.to_vec(), labels.to_vec()) {
            (Ok(prob_vec), Ok(label_vec)) => {
                let shape = probabilities.shape();
                let dims = shape.dims();

                if dims.len() != 2 || dims[0] != label_vec.len() {
                    return 0.0;
                }

                let n_samples = dims[0];
                let n_classes = dims[1];

                if n_samples == 0 || n_classes == 0 {
                    return 0.0;
                }

                let mut score = 0.0;

                for i in 0..n_samples {
                    let true_class = label_vec[i] as usize;

                    for j in 0..n_classes {
                        let prob = prob_vec[i * n_classes + j] as f64;
                        let target = if j == true_class { 1.0 } else { 0.0 };
                        let diff = prob - target;
                        score += diff * diff;
                    }
                }

                score / n_samples as f64
            }
            _ => 0.0,
        }
    }
}

impl Metric for BrierScore {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_brier(predictions, targets)
    }

    fn name(&self) -> &str {
        "brier_score"
    }
}

/// Entropy-based uncertainty measures
pub struct EntropyMeasures;

impl EntropyMeasures {
    /// Compute predictive entropy (epistemic + aleatoric uncertainty)
    pub fn predictive_entropy(&self, probabilities: &Tensor) -> f64 {
        match probabilities.to_vec() {
            Ok(prob_vec) => {
                let shape = probabilities.shape();
                let dims = shape.dims();

                if dims.len() != 2 {
                    return 0.0;
                }

                let n_samples = dims[0];
                let n_classes = dims[1];

                if n_samples == 0 || n_classes == 0 {
                    return 0.0;
                }

                let mut total_entropy = 0.0;

                for i in 0..n_samples {
                    let mut entropy = 0.0;

                    for j in 0..n_classes {
                        let prob = prob_vec[i * n_classes + j] as f64;
                        if prob > 1e-10 {
                            entropy -= prob * prob.ln();
                        }
                    }

                    total_entropy += entropy;
                }

                total_entropy / n_samples as f64
            }
            _ => 0.0,
        }
    }

    /// Compute mutual information (epistemic uncertainty estimate)
    /// Requires multiple Monte Carlo samples
    pub fn mutual_information(&self, mc_predictions: &[Tensor]) -> f64 {
        if mc_predictions.is_empty() {
            return 0.0;
        }

        // Get dimensions from first prediction
        let shape = mc_predictions[0].shape();
        let dims = shape.dims();

        if dims.len() != 2 {
            return 0.0;
        }

        let n_samples = dims[0];
        let n_classes = dims[1];
        let n_mc_samples = mc_predictions.len();

        // Compute average predictive entropy and entropy of average predictions
        let mut avg_predictive_entropy = 0.0;
        let mut avg_predictions = vec![vec![0.0; n_classes]; n_samples];

        // Collect all MC predictions
        for mc_pred in mc_predictions {
            if let Ok(pred_vec) = mc_pred.to_vec() {
                for i in 0..n_samples {
                    let mut sample_entropy = 0.0;

                    for j in 0..n_classes {
                        let prob = pred_vec[i * n_classes + j] as f64;
                        avg_predictions[i][j] += prob / n_mc_samples as f64;

                        if prob > 1e-10 {
                            sample_entropy -= prob * prob.ln();
                        }
                    }

                    avg_predictive_entropy += sample_entropy / (n_samples * n_mc_samples) as f64;
                }
            }
        }

        // Compute entropy of average predictions
        let mut entropy_of_avg = 0.0;
        for i in 0..n_samples {
            for j in 0..n_classes {
                let avg_prob = avg_predictions[i][j];
                if avg_prob > 1e-10 {
                    entropy_of_avg -= avg_prob * avg_prob.ln();
                }
            }
        }
        entropy_of_avg /= n_samples as f64;

        // Mutual information = H[E[p]] - E[H[p]]
        entropy_of_avg - avg_predictive_entropy
    }
}

impl Metric for EntropyMeasures {
    fn compute(&self, predictions: &Tensor, _targets: &Tensor) -> f64 {
        self.predictive_entropy(predictions)
    }

    fn name(&self) -> &str {
        "predictive_entropy"
    }
}

/// Reliability diagram for visualizing calibration
pub struct ReliabilityDiagram {
    n_bins: usize,
}

impl ReliabilityDiagram {
    /// Create a new reliability diagram
    pub fn new(n_bins: usize) -> Self {
        Self { n_bins }
    }

    /// Compute reliability diagram data
    /// Returns (bin_boundaries, bin_accuracies, bin_confidences, bin_counts)
    pub fn compute_diagram_data(
        &self,
        confidences: &Tensor,
        predictions: &Tensor,
        labels: &Tensor,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>) {
        match (confidences.to_vec(), predictions.to_vec(), labels.to_vec()) {
            (Ok(conf_vec), Ok(pred_vec), Ok(label_vec)) => {
                if conf_vec.len() != pred_vec.len()
                    || conf_vec.len() != label_vec.len()
                    || conf_vec.is_empty()
                {
                    return (vec![], vec![], vec![], vec![]);
                }

                let n = conf_vec.len();
                let mut bins = vec![(0.0, 0, 0); self.n_bins]; // (sum_confidence, num_correct, num_total)

                for i in 0..n {
                    let confidence = conf_vec[i] as f64;
                    let predicted_class = pred_vec[i] as usize;
                    let true_class = label_vec[i] as usize;

                    let bin_idx =
                        ((confidence * self.n_bins as f64).floor() as usize).min(self.n_bins - 1);

                    bins[bin_idx].0 += confidence;
                    if predicted_class == true_class {
                        bins[bin_idx].1 += 1;
                    }
                    bins[bin_idx].2 += 1;
                }

                let mut bin_boundaries = Vec::new();
                let mut bin_accuracies = Vec::new();
                let mut bin_confidences = Vec::new();
                let mut bin_counts = Vec::new();

                for (i, (sum_conf, num_correct, num_total)) in bins.into_iter().enumerate() {
                    let bin_lower = i as f64 / self.n_bins as f64;
                    let bin_upper = (i + 1) as f64 / self.n_bins as f64;

                    bin_boundaries.push((bin_lower + bin_upper) / 2.0);

                    if num_total > 0 {
                        bin_accuracies.push(num_correct as f64 / num_total as f64);
                        bin_confidences.push(sum_conf / num_total as f64);
                    } else {
                        bin_accuracies.push(0.0);
                        bin_confidences.push(0.0);
                    }

                    bin_counts.push(num_total);
                }

                (bin_boundaries, bin_accuracies, bin_confidences, bin_counts)
            }
            _ => (vec![], vec![], vec![], vec![]),
        }
    }
}

impl Metric for ReliabilityDiagram {
    fn compute(&self, _predictions: &Tensor, _targets: &Tensor) -> f64 {
        // This metric produces diagram data, not a single score
        0.0
    }

    fn name(&self) -> &str {
        "reliability_diagram"
    }
}
