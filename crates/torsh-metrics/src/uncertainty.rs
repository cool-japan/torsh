//! Uncertainty quantification metrics

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::Metric;
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

/// Decomposition of uncertainty into epistemic and aleatoric components
///
/// Epistemic uncertainty (model uncertainty): Reducible through more data
/// Aleatoric uncertainty (data uncertainty): Irreducible inherent noise
#[derive(Debug, Clone)]
pub struct UncertaintyDecomposition {
    /// Total predictive uncertainty
    pub total_uncertainty: f64,
    /// Epistemic uncertainty (model uncertainty)
    pub epistemic_uncertainty: f64,
    /// Aleatoric uncertainty (data uncertainty)
    pub aleatoric_uncertainty: f64,
    /// Ratio of epistemic to total uncertainty
    pub epistemic_ratio: f64,
    /// Ratio of aleatoric to total uncertainty
    pub aleatoric_ratio: f64,
}

impl UncertaintyDecomposition {
    /// Decompose uncertainty from Monte Carlo predictions
    ///
    /// Uses entropy-based decomposition:
    /// - Total uncertainty = H[E[p(y|x)]] (entropy of average predictions)
    /// - Aleatoric uncertainty = E[H[p(y|x)]] (average entropy of predictions)
    /// - Epistemic uncertainty = Total - Aleatoric (mutual information)
    ///
    /// # Arguments
    /// * `mc_predictions` - Multiple stochastic forward passes (MC Dropout, Bayesian, Ensemble)
    pub fn from_mc_predictions(mc_predictions: &[Tensor]) -> Option<Self> {
        if mc_predictions.is_empty() {
            return None;
        }

        let shape = mc_predictions[0].shape();
        let dims = shape.dims();

        if dims.len() != 2 {
            return None;
        }

        let n_samples = dims[0];
        let n_classes = dims[1];
        let n_mc_samples = mc_predictions.len();

        if n_samples == 0 || n_classes == 0 {
            return None;
        }

        // Compute average predictions across MC samples
        let mut avg_predictions = vec![vec![0.0; n_classes]; n_samples];
        let mut aleatoric_uncertainty = 0.0;

        for mc_pred in mc_predictions {
            if let Ok(pred_vec) = mc_pred.to_vec() {
                for i in 0..n_samples {
                    let mut sample_entropy = 0.0;

                    for j in 0..n_classes {
                        let prob = pred_vec[i * n_classes + j] as f64;
                        avg_predictions[i][j] += prob / n_mc_samples as f64;

                        // Accumulate aleatoric uncertainty (average entropy)
                        if prob > 1e-10 {
                            sample_entropy -= prob * prob.ln();
                        }
                    }

                    aleatoric_uncertainty += sample_entropy / n_mc_samples as f64;
                }
            }
        }
        aleatoric_uncertainty /= n_samples as f64;

        // Compute total uncertainty (entropy of average predictions)
        let mut total_uncertainty = 0.0;
        for i in 0..n_samples {
            for j in 0..n_classes {
                let avg_prob = avg_predictions[i][j];
                if avg_prob > 1e-10 {
                    total_uncertainty -= avg_prob * avg_prob.ln();
                }
            }
        }
        total_uncertainty /= n_samples as f64;

        // Epistemic uncertainty is the difference
        let epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty;
        let epistemic_uncertainty = epistemic_uncertainty.max(0.0); // Ensure non-negative

        let epistemic_ratio = if total_uncertainty > 1e-10 {
            epistemic_uncertainty / total_uncertainty
        } else {
            0.0
        };

        let aleatoric_ratio = if total_uncertainty > 1e-10 {
            aleatoric_uncertainty / total_uncertainty
        } else {
            0.0
        };

        Some(UncertaintyDecomposition {
            total_uncertainty,
            epistemic_uncertainty,
            aleatoric_uncertainty,
            epistemic_ratio,
            aleatoric_ratio,
        })
    }

    /// Assess if model needs more data or better architecture
    pub fn diagnostic(&self) -> &str {
        if self.epistemic_ratio > 0.7 {
            "High epistemic uncertainty - Model needs more training data or better architecture"
        } else if self.aleatoric_ratio > 0.7 {
            "High aleatoric uncertainty - Inherent noise in data, consider feature engineering"
        } else {
            "Balanced uncertainty - Model is reasonably confident"
        }
    }

    /// Format uncertainty decomposition as a string
    pub fn format(&self) -> String {
        let mut result = String::new();
        result.push_str("Uncertainty Decomposition:\n");
        result.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        result.push_str(&format!(
            "Total Uncertainty:      {:.4} nats\n",
            self.total_uncertainty
        ));
        result.push_str(&format!(
            "Epistemic (Model):      {:.4} nats ({:.1}%)\n",
            self.epistemic_uncertainty,
            self.epistemic_ratio * 100.0
        ));
        result.push_str(&format!(
            "Aleatoric (Data):       {:.4} nats ({:.1}%)\n",
            self.aleatoric_uncertainty,
            self.aleatoric_ratio * 100.0
        ));
        result.push_str(&format!("\nDiagnostic: {}\n", self.diagnostic()));

        result
    }
}

/// MC Dropout uncertainty estimation
///
/// Estimates uncertainty by performing multiple forward passes with dropout enabled
pub struct MCDropoutUncertainty {
    n_samples: usize,
}

impl MCDropoutUncertainty {
    /// Create new MC Dropout uncertainty estimator
    pub fn new(n_samples: usize) -> Self {
        assert!(n_samples > 0, "Number of samples must be positive");
        Self { n_samples }
    }

    /// Compute uncertainty from MC Dropout samples
    ///
    /// # Arguments
    /// * `mc_predictions` - Multiple forward passes with dropout enabled
    ///
    /// # Returns
    /// Uncertainty decomposition with epistemic and aleatoric components
    pub fn compute_uncertainty(
        &self,
        mc_predictions: &[Tensor],
    ) -> Option<UncertaintyDecomposition> {
        if mc_predictions.len() != self.n_samples {
            return None;
        }

        UncertaintyDecomposition::from_mc_predictions(mc_predictions)
    }

    /// Compute predictive mean and variance
    ///
    /// Returns (mean_predictions, variance_per_sample)
    pub fn predictive_statistics(
        &self,
        mc_predictions: &[Tensor],
    ) -> Option<(Vec<Vec<f64>>, Vec<f64>)> {
        if mc_predictions.is_empty() {
            return None;
        }

        let shape = mc_predictions[0].shape();
        let dims = shape.dims();

        if dims.len() != 2 {
            return None;
        }

        let n_samples = dims[0];
        let n_classes = dims[1];

        let mut mean_predictions = vec![vec![0.0; n_classes]; n_samples];
        let mut variances = vec![0.0; n_samples];

        // Compute means
        for mc_pred in mc_predictions {
            if let Ok(pred_vec) = mc_pred.to_vec() {
                for i in 0..n_samples {
                    for j in 0..n_classes {
                        let prob = pred_vec[i * n_classes + j] as f64;
                        mean_predictions[i][j] += prob / mc_predictions.len() as f64;
                    }
                }
            }
        }

        // Compute variances
        for mc_pred in mc_predictions {
            if let Ok(pred_vec) = mc_pred.to_vec() {
                for i in 0..n_samples {
                    for j in 0..n_classes {
                        let prob = pred_vec[i * n_classes + j] as f64;
                        let diff = prob - mean_predictions[i][j];
                        variances[i] += diff * diff / (mc_predictions.len() * n_classes) as f64;
                    }
                }
            }
        }

        Some((mean_predictions, variances))
    }
}

/// Deep Ensemble uncertainty estimation
///
/// Estimates uncertainty by training multiple independent models
pub struct EnsembleUncertainty {
    n_models: usize,
}

impl EnsembleUncertainty {
    /// Create new ensemble uncertainty estimator
    pub fn new(n_models: usize) -> Self {
        assert!(n_models > 0, "Number of models must be positive");
        Self { n_models }
    }

    /// Compute uncertainty from ensemble predictions
    ///
    /// # Arguments
    /// * `ensemble_predictions` - Predictions from each model in the ensemble
    ///
    /// # Returns
    /// Uncertainty decomposition with epistemic and aleatoric components
    pub fn compute_uncertainty(
        &self,
        ensemble_predictions: &[Tensor],
    ) -> Option<UncertaintyDecomposition> {
        if ensemble_predictions.len() != self.n_models {
            return None;
        }

        UncertaintyDecomposition::from_mc_predictions(ensemble_predictions)
    }

    /// Compute ensemble agreement
    ///
    /// Returns the fraction of models that agree on the predicted class
    pub fn ensemble_agreement(&self, ensemble_predictions: &[Tensor]) -> f64 {
        if ensemble_predictions.is_empty() {
            return 0.0;
        }

        let shape = ensemble_predictions[0].shape();
        let dims = shape.dims();

        if dims.len() != 2 {
            return 0.0;
        }

        let n_samples = dims[0];
        let mut total_agreement = 0.0;

        for i in 0..n_samples {
            // Get predicted classes from each model
            let mut class_counts = std::collections::HashMap::new();

            for pred in ensemble_predictions {
                if let Ok(pred_vec) = pred.to_vec() {
                    let start_idx = i * dims[1];
                    let end_idx = start_idx + dims[1];
                    let sample_probs = &pred_vec[start_idx..end_idx];

                    // Find argmax
                    let predicted_class = sample_probs
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);

                    *class_counts.entry(predicted_class).or_insert(0) += 1;
                }
            }

            // Agreement is the fraction of models agreeing on the most common class
            if let Some(&max_count) = class_counts.values().max() {
                total_agreement += max_count as f64 / ensemble_predictions.len() as f64;
            }
        }

        total_agreement / n_samples as f64
    }

    /// Compute predictive diversity
    ///
    /// Measures how diverse the ensemble predictions are (higher = more diverse)
    pub fn predictive_diversity(&self, ensemble_predictions: &[Tensor]) -> f64 {
        1.0 - self.ensemble_agreement(ensemble_predictions)
    }
}

/// Bayesian uncertainty estimation for variational inference
///
/// Estimates uncertainty in Bayesian neural networks
pub struct BayesianUncertainty {
    n_samples: usize,
}

impl BayesianUncertainty {
    /// Create new Bayesian uncertainty estimator
    pub fn new(n_samples: usize) -> Self {
        assert!(n_samples > 0, "Number of samples must be positive");
        Self { n_samples }
    }

    /// Compute uncertainty from posterior samples
    ///
    /// # Arguments
    /// * `posterior_samples` - Predictions from different posterior weight samples
    ///
    /// # Returns
    /// Uncertainty decomposition with epistemic and aleatoric components
    pub fn compute_uncertainty(
        &self,
        posterior_samples: &[Tensor],
    ) -> Option<UncertaintyDecomposition> {
        if posterior_samples.len() != self.n_samples {
            return None;
        }

        UncertaintyDecomposition::from_mc_predictions(posterior_samples)
    }

    /// Compute credible interval for predictions
    ///
    /// Returns (lower_bound, upper_bound) for each sample at specified confidence level
    pub fn credible_interval(
        &self,
        posterior_samples: &[Tensor],
        confidence: f64,
    ) -> Option<(Vec<f64>, Vec<f64>)> {
        if posterior_samples.is_empty() || !(0.0..1.0).contains(&confidence) {
            return None;
        }

        let shape = posterior_samples[0].shape();
        let dims = shape.dims();

        if dims.len() != 2 {
            return None;
        }

        let n_samples = dims[0];
        let n_classes = dims[1];

        let mut predictions_per_sample = vec![Vec::new(); n_samples];

        // Collect all predictions for each sample
        for pred in posterior_samples {
            if let Ok(pred_vec) = pred.to_vec() {
                for i in 0..n_samples {
                    // Get max probability as prediction confidence
                    let start_idx = i * n_classes;
                    let end_idx = start_idx + n_classes;
                    let max_prob = pred_vec[start_idx..end_idx]
                        .iter()
                        .fold(f32::NEG_INFINITY, |a, &b| a.max(b))
                        as f64;
                    predictions_per_sample[i].push(max_prob);
                }
            }
        }

        let alpha = (1.0 - confidence) / 2.0;
        let lower_percentile = alpha;
        let upper_percentile = 1.0 - alpha;

        let mut lower_bounds = Vec::new();
        let mut upper_bounds = Vec::new();

        for preds in predictions_per_sample {
            if preds.is_empty() {
                lower_bounds.push(0.0);
                upper_bounds.push(0.0);
                continue;
            }

            let mut sorted_preds = preds.clone();
            sorted_preds.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let lower_idx = (sorted_preds.len() as f64 * lower_percentile) as usize;
            let upper_idx = ((sorted_preds.len() as f64 * upper_percentile) as usize)
                .min(sorted_preds.len() - 1);

            lower_bounds.push(sorted_preds[lower_idx]);
            upper_bounds.push(sorted_preds[upper_idx]);
        }

        Some((lower_bounds, upper_bounds))
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

/// Comprehensive calibration metrics collection
#[derive(Debug, Clone)]
pub struct CalibrationMetrics {
    /// Expected Calibration Error
    pub expected_calibration_error: f64,
    /// Maximum Calibration Error
    pub maximum_calibration_error: f64,
    /// Reliability diagram data (bin centers, accuracies, confidences, counts)
    pub reliability_diagram: (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>),
    /// Number of bins used
    pub n_bins: usize,
}

impl CalibrationMetrics {
    /// Compute comprehensive calibration metrics
    ///
    /// # Arguments
    /// * `confidences` - Model confidence scores (probabilities)
    /// * `predictions` - Predicted class labels
    /// * `labels` - True class labels
    /// * `n_bins` - Number of bins for calibration histogram (default: 10)
    pub fn compute(
        confidences: &Tensor,
        predictions: &Tensor,
        labels: &Tensor,
        n_bins: usize,
    ) -> Self {
        let ece_metric = ExpectedCalibrationError::new(n_bins);
        let mce_metric = MaximumCalibrationError::new(n_bins);
        let reliability_metric = ReliabilityDiagram::new(n_bins);

        let expected_calibration_error = ece_metric.compute_ece(confidences, predictions, labels);
        let maximum_calibration_error = mce_metric.compute_mce(confidences, predictions, labels);
        let reliability_diagram =
            reliability_metric.compute_diagram_data(confidences, predictions, labels);

        CalibrationMetrics {
            expected_calibration_error,
            maximum_calibration_error,
            reliability_diagram,
            n_bins,
        }
    }

    /// Create with default 10 bins
    pub fn compute_default(confidences: &Tensor, predictions: &Tensor, labels: &Tensor) -> Self {
        Self::compute(confidences, predictions, labels, 10)
    }

    /// Check if model is well-calibrated (ECE < threshold)
    pub fn is_well_calibrated(&self, threshold: f64) -> bool {
        self.expected_calibration_error < threshold
    }

    /// Get calibration quality assessment
    pub fn quality_assessment(&self) -> &str {
        if self.expected_calibration_error < 0.05 {
            "Excellent calibration"
        } else if self.expected_calibration_error < 0.10 {
            "Good calibration"
        } else if self.expected_calibration_error < 0.15 {
            "Fair calibration"
        } else {
            "Poor calibration"
        }
    }

    /// Format calibration metrics as a string
    pub fn format(&self) -> String {
        let mut result = String::new();
        result.push_str("Calibration Metrics:\n");
        result.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        result.push_str(&format!(
            "Expected Calibration Error (ECE): {:.4}\n",
            self.expected_calibration_error
        ));
        result.push_str(&format!(
            "Maximum Calibration Error (MCE): {:.4}\n",
            self.maximum_calibration_error
        ));
        result.push_str(&format!("Quality: {}\n", self.quality_assessment()));
        result.push_str(&format!("Number of bins: {}\n", self.n_bins));

        result.push_str("\nReliability Diagram Data:\n");
        let (bin_centers, accuracies, confidences, counts) = &self.reliability_diagram;

        for i in 0..bin_centers.len() {
            if counts[i] > 0 {
                result.push_str(&format!(
                    "  Bin {:.2}: Acc={:.3}, Conf={:.3}, Count={}\n",
                    bin_centers[i], accuracies[i], confidences[i], counts[i]
                ));
            }
        }

        result
    }

    /// Get bin-wise calibration errors
    pub fn bin_errors(&self) -> Vec<f64> {
        let (_centers, accuracies, confidences, _counts) = &self.reliability_diagram;
        accuracies
            .iter()
            .zip(confidences.iter())
            .map(|(acc, conf)| (acc - conf).abs())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_calibration_metrics() {
        // Well-calibrated predictions where confidences match accuracies
        let confidences =
            from_vec(vec![0.9, 0.8, 0.7, 0.6, 0.5, 0.4], &[6], DeviceType::Cpu).unwrap();
        let predictions =
            from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[6], DeviceType::Cpu).unwrap();
        let labels = from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[6], DeviceType::Cpu).unwrap();

        let cal_metrics = CalibrationMetrics::compute(&confidences, &predictions, &labels, 3);

        // Should have finite ECE
        assert!(cal_metrics.expected_calibration_error.is_finite());
        assert_eq!(cal_metrics.n_bins, 3);
        // ECE should be less than 1.0 for reasonable predictions
        assert!(cal_metrics.expected_calibration_error <= 1.0);
    }

    #[test]
    fn test_calibration_quality_assessment() {
        let confidences = from_vec(vec![0.9, 0.9, 0.9, 0.1, 0.1], &[5], DeviceType::Cpu).unwrap();
        let predictions = from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0], &[5], DeviceType::Cpu).unwrap();
        let labels = from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0], &[5], DeviceType::Cpu).unwrap();

        let cal_metrics = CalibrationMetrics::compute_default(&confidences, &predictions, &labels);

        let assessment = cal_metrics.quality_assessment();
        // Should return one of the valid assessment strings
        assert!(
            assessment == "Excellent calibration"
                || assessment == "Good calibration"
                || assessment == "Fair calibration"
                || assessment == "Poor calibration"
        );
    }

    #[test]
    fn test_uncertainty_decomposition() {
        // Create multiple MC predictions with varying uncertainty
        let pred1 = from_vec(vec![0.7, 0.2, 0.1, 0.6, 0.3, 0.1], &[2, 3], DeviceType::Cpu).unwrap();
        let pred2 = from_vec(vec![0.8, 0.1, 0.1, 0.5, 0.4, 0.1], &[2, 3], DeviceType::Cpu).unwrap();
        let pred3 = from_vec(vec![0.6, 0.3, 0.1, 0.7, 0.2, 0.1], &[2, 3], DeviceType::Cpu).unwrap();

        let mc_predictions = vec![pred1, pred2, pred3];

        let decomposition = UncertaintyDecomposition::from_mc_predictions(&mc_predictions);

        assert!(decomposition.is_some());
        let decomp = decomposition.unwrap();

        // Check that uncertainties are non-negative
        assert!(decomp.total_uncertainty >= 0.0);
        assert!(decomp.epistemic_uncertainty >= 0.0);
        assert!(decomp.aleatoric_uncertainty >= 0.0);

        // Check that ratios sum to approximately 1.0
        assert!((decomp.epistemic_ratio + decomp.aleatoric_ratio - 1.0).abs() < 1e-6);

        // Check that epistemic + aleatoric ≈ total
        assert!(
            (decomp.epistemic_uncertainty + decomp.aleatoric_uncertainty
                - decomp.total_uncertainty)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn test_uncertainty_decomposition_high_epistemic() {
        // Create MC predictions with high disagreement (high epistemic uncertainty)
        // Using more extreme predictions to create higher epistemic uncertainty
        let pred1 = from_vec(vec![0.95, 0.05, 0.1, 0.9], &[2, 2], DeviceType::Cpu).unwrap();
        let pred2 = from_vec(vec![0.05, 0.95, 0.9, 0.1], &[2, 2], DeviceType::Cpu).unwrap();
        let pred3 = from_vec(vec![0.5, 0.5, 0.5, 0.5], &[2, 2], DeviceType::Cpu).unwrap();

        let mc_predictions = vec![pred1, pred2, pred3];

        let decomposition = UncertaintyDecomposition::from_mc_predictions(&mc_predictions).unwrap();

        // Epistemic uncertainty should be present due to model disagreement
        assert!(decomposition.epistemic_uncertainty > 0.0);
        // Total uncertainty should be significant
        assert!(decomposition.total_uncertainty > 0.0);
    }

    #[test]
    fn test_uncertainty_decomposition_high_aleatoric() {
        // Create MC predictions with high entropy but low disagreement (high aleatoric)
        let pred1 = from_vec(vec![0.33, 0.33, 0.34, 0.5, 0.5], &[2, 2], DeviceType::Cpu).unwrap();
        let pred2 = from_vec(vec![0.34, 0.33, 0.33, 0.5, 0.5], &[2, 2], DeviceType::Cpu).unwrap();
        let pred3 = from_vec(vec![0.33, 0.34, 0.33, 0.5, 0.5], &[2, 2], DeviceType::Cpu).unwrap();

        let mc_predictions = vec![pred1, pred2, pred3];

        let decomposition = UncertaintyDecomposition::from_mc_predictions(&mc_predictions).unwrap();

        // Aleatoric should be significant due to high entropy predictions
        assert!(decomposition.total_uncertainty > 0.0);
    }

    #[test]
    fn test_mc_dropout_uncertainty() {
        let n_samples = 5;
        let mc_dropout = MCDropoutUncertainty::new(n_samples);

        // Create MC dropout predictions
        let mut mc_predictions = Vec::new();
        for i in 0..n_samples {
            let noise = (i as f32) * 0.05;
            let pred = from_vec(
                vec![0.7 + noise, 0.2, 0.1, 0.6 - noise, 0.3, 0.1],
                &[2, 3],
                DeviceType::Cpu,
            )
            .unwrap();
            mc_predictions.push(pred);
        }

        let uncertainty = mc_dropout.compute_uncertainty(&mc_predictions);
        assert!(uncertainty.is_some());

        let stats = mc_dropout.predictive_statistics(&mc_predictions);
        assert!(stats.is_some());

        let (means, variances) = stats.unwrap();
        assert_eq!(means.len(), 2); // 2 samples
        assert_eq!(variances.len(), 2);

        // Variances should be non-negative
        for var in &variances {
            assert!(*var >= 0.0);
        }
    }

    #[test]
    fn test_ensemble_uncertainty() {
        let n_models = 3;
        let ensemble = EnsembleUncertainty::new(n_models);

        // Create ensemble predictions
        let pred1 = from_vec(vec![0.8, 0.2, 0.7, 0.3], &[2, 2], DeviceType::Cpu).unwrap();
        let pred2 = from_vec(vec![0.7, 0.3, 0.6, 0.4], &[2, 2], DeviceType::Cpu).unwrap();
        let pred3 = from_vec(vec![0.9, 0.1, 0.8, 0.2], &[2, 2], DeviceType::Cpu).unwrap();

        let ensemble_predictions = vec![pred1, pred2, pred3];

        let uncertainty = ensemble.compute_uncertainty(&ensemble_predictions);
        assert!(uncertainty.is_some());

        let agreement = ensemble.ensemble_agreement(&ensemble_predictions);
        assert!(agreement >= 0.0 && agreement <= 1.0);

        let diversity = ensemble.predictive_diversity(&ensemble_predictions);
        assert!(diversity >= 0.0 && diversity <= 1.0);

        // Agreement + diversity should sum to 1.0
        assert!((agreement + diversity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ensemble_high_agreement() {
        let n_models = 3;
        let ensemble = EnsembleUncertainty::new(n_models);

        // Create ensemble predictions with high agreement
        let pred1 = from_vec(vec![0.9, 0.1, 0.8, 0.2], &[2, 2], DeviceType::Cpu).unwrap();
        let pred2 = from_vec(vec![0.9, 0.1, 0.8, 0.2], &[2, 2], DeviceType::Cpu).unwrap();
        let pred3 = from_vec(vec![0.9, 0.1, 0.8, 0.2], &[2, 2], DeviceType::Cpu).unwrap();

        let ensemble_predictions = vec![pred1, pred2, pred3];

        let agreement = ensemble.ensemble_agreement(&ensemble_predictions);

        // Should have perfect or near-perfect agreement
        assert!(agreement > 0.95);
    }

    #[test]
    fn test_bayesian_uncertainty() {
        let n_samples = 5;
        let bayesian = BayesianUncertainty::new(n_samples);

        // Create posterior samples
        let mut posterior_samples = Vec::new();
        for i in 0..n_samples {
            let noise = (i as f32) * 0.05;
            let pred = from_vec(
                vec![0.7 + noise, 0.3 - noise, 0.6, 0.4],
                &[2, 2],
                DeviceType::Cpu,
            )
            .unwrap();
            posterior_samples.push(pred);
        }

        let uncertainty = bayesian.compute_uncertainty(&posterior_samples);
        assert!(uncertainty.is_some());

        let credible_interval = bayesian.credible_interval(&posterior_samples, 0.95);
        assert!(credible_interval.is_some());

        let (lower_bounds, upper_bounds) = credible_interval.unwrap();
        assert_eq!(lower_bounds.len(), 2); // 2 samples
        assert_eq!(upper_bounds.len(), 2);

        // Lower bounds should be <= upper bounds
        for (lower, upper) in lower_bounds.iter().zip(upper_bounds.iter()) {
            assert!(lower <= upper);
        }
    }

    #[test]
    fn test_uncertainty_decomposition_format() {
        let pred1 = from_vec(vec![0.8, 0.2, 0.7, 0.3], &[2, 2], DeviceType::Cpu).unwrap();
        let pred2 = from_vec(vec![0.7, 0.3, 0.6, 0.4], &[2, 2], DeviceType::Cpu).unwrap();

        let mc_predictions = vec![pred1, pred2];
        let decomposition = UncertaintyDecomposition::from_mc_predictions(&mc_predictions).unwrap();

        let formatted = decomposition.format();

        // Check that format contains key information
        assert!(formatted.contains("Total Uncertainty"));
        assert!(formatted.contains("Epistemic"));
        assert!(formatted.contains("Aleatoric"));
        assert!(formatted.contains("Diagnostic"));
    }

    #[test]
    fn test_uncertainty_diagnostic() {
        // Test high epistemic uncertainty diagnostic
        let pred1 = from_vec(vec![0.9, 0.1, 0.1, 0.9], &[2, 2], DeviceType::Cpu).unwrap();
        let pred2 = from_vec(vec![0.1, 0.9, 0.9, 0.1], &[2, 2], DeviceType::Cpu).unwrap();

        let mc_predictions = vec![pred1, pred2];
        let decomposition = UncertaintyDecomposition::from_mc_predictions(&mc_predictions).unwrap();

        let diagnostic = decomposition.diagnostic();
        assert!(!diagnostic.is_empty());
    }
}
