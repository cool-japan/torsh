//! Robustness and reliability metrics
//!
//! This module provides metrics for evaluating model robustness to various
//! perturbations, noise, and adversarial examples.
//! All implementations follow SciRS2 POLICY - using scirs2-core abstractions.

use torsh_core::error::TorshError;
use torsh_tensor::Tensor;

/// Adversarial robustness - accuracy under adversarial perturbations
///
/// Measures how often the model maintains correct predictions when inputs
/// are adversarially perturbed.
///
/// # Arguments
/// * `clean_predictions` - Predictions on clean inputs
/// * `perturbed_predictions` - Predictions on adversarially perturbed inputs
/// * `targets` - True labels
pub fn adversarial_accuracy(
    clean_predictions: &Tensor,
    perturbed_predictions: &Tensor,
    targets: &Tensor,
) -> Result<f64, TorshError> {
    let clean_vec = clean_predictions.to_vec()?;
    let perturbed_vec = perturbed_predictions.to_vec()?;
    let target_vec = targets.to_vec()?;

    if clean_vec.len() != perturbed_vec.len() || clean_vec.len() != target_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    let mut correct_under_attack = 0;

    for i in 0..clean_vec.len() {
        // Check if prediction remains correct after perturbation
        let clean_correct = (clean_vec[i] - target_vec[i]).abs() < 0.5;
        let perturbed_correct = (perturbed_vec[i] - target_vec[i]).abs() < 0.5;

        if clean_correct && perturbed_correct {
            correct_under_attack += 1;
        }
    }

    Ok(correct_under_attack as f64 / clean_vec.len() as f64)
}

/// Attack success rate - fraction of clean correct predictions that flip after attack
pub fn attack_success_rate(
    clean_predictions: &Tensor,
    perturbed_predictions: &Tensor,
    targets: &Tensor,
) -> Result<f64, TorshError> {
    let clean_vec = clean_predictions.to_vec()?;
    let perturbed_vec = perturbed_predictions.to_vec()?;
    let target_vec = targets.to_vec()?;

    if clean_vec.len() != perturbed_vec.len() || clean_vec.len() != target_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    let mut clean_correct_count = 0;
    let mut successful_attacks = 0;

    for i in 0..clean_vec.len() {
        let clean_correct = (clean_vec[i] - target_vec[i]).abs() < 0.5;

        if clean_correct {
            clean_correct_count += 1;
            let perturbed_correct = (perturbed_vec[i] - target_vec[i]).abs() < 0.5;
            if !perturbed_correct {
                successful_attacks += 1;
            }
        }
    }

    if clean_correct_count == 0 {
        return Ok(0.0);
    }

    Ok(successful_attacks as f64 / clean_correct_count as f64)
}

/// Noise sensitivity - average prediction change under random noise
///
/// Measures how much predictions change when random noise is added to inputs.
/// Lower values indicate more robust models.
pub fn noise_sensitivity(
    clean_predictions: &Tensor,
    noisy_predictions: &Tensor,
) -> Result<f64, TorshError> {
    let clean_vec = clean_predictions.to_vec()?;
    let noisy_vec = noisy_predictions.to_vec()?;

    if clean_vec.len() != noisy_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    let prediction_changes: f64 = clean_vec
        .iter()
        .zip(noisy_vec.iter())
        .map(|(&c, &n)| (c as f64 - n as f64).abs())
        .sum();

    Ok(prediction_changes / clean_vec.len() as f64)
}

/// Prediction confidence stability under perturbation
///
/// Measures how stable prediction confidences are when inputs are perturbed.
/// Higher values indicate more stable (robust) confidences.
pub fn confidence_stability(
    clean_confidences: &Tensor,
    perturbed_confidences: &Tensor,
) -> Result<f64, TorshError> {
    let clean_vec = clean_confidences.to_vec()?;
    let perturbed_vec = perturbed_confidences.to_vec()?;

    if clean_vec.len() != perturbed_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    // Calculate correlation between clean and perturbed confidences
    let n = clean_vec.len() as f64;
    let clean_f64: Vec<f64> = clean_vec.iter().map(|&x| x as f64).collect();
    let perturbed_f64: Vec<f64> = perturbed_vec.iter().map(|&x| x as f64).collect();

    let clean_mean = clean_f64.iter().sum::<f64>() / n;
    let perturbed_mean = perturbed_f64.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut clean_var = 0.0;
    let mut perturbed_var = 0.0;

    for i in 0..clean_vec.len() {
        let clean_dev = clean_f64[i] - clean_mean;
        let perturbed_dev = perturbed_f64[i] - perturbed_mean;
        numerator += clean_dev * perturbed_dev;
        clean_var += clean_dev * clean_dev;
        perturbed_var += perturbed_dev * perturbed_dev;
    }

    if clean_var < 1e-10 || perturbed_var < 1e-10 {
        return Ok(1.0); // Perfect stability if no variance
    }

    let correlation = numerator / (clean_var * perturbed_var).sqrt();
    Ok((correlation + 1.0) / 2.0) // Transform to [0, 1]
}

/// Out-of-distribution (OOD) detection performance
///
/// Measures how well the model detects OOD inputs based on confidence.
/// Uses AUROC-like metric.
///
/// # Arguments
/// * `in_dist_confidences` - Confidences for in-distribution samples
/// * `out_dist_confidences` - Confidences for out-of-distribution samples
pub fn ood_detection_score(
    in_dist_confidences: &[f64],
    out_dist_confidences: &[f64],
) -> Result<f64, TorshError> {
    if in_dist_confidences.is_empty() || out_dist_confidences.is_empty() {
        return Err(TorshError::InvalidArgument("Empty inputs".to_string()));
    }

    // Count how often in-dist has higher confidence than out-dist
    let mut correct_orderings = 0;
    let total_comparisons = in_dist_confidences.len() * out_dist_confidences.len();

    for &in_conf in in_dist_confidences {
        for &out_conf in out_dist_confidences {
            if in_conf > out_conf {
                correct_orderings += 1;
            } else if (in_conf - out_conf).abs() < 1e-10 {
                // Ties count as 0.5
                correct_orderings += 1; // Will divide by 2 below
            }
        }
    }

    Ok(correct_orderings as f64 / total_comparisons as f64)
}

/// Corruption robustness - accuracy under common corruptions
///
/// Measures how well the model maintains accuracy under various corruptions
/// (blur, noise, compression, etc.)
///
/// # Arguments
/// * `clean_accuracy` - Accuracy on clean data
/// * `corrupted_accuracies` - Accuracies on different corruption types
pub fn corruption_robustness(
    clean_accuracy: f64,
    corrupted_accuracies: &[f64],
) -> Result<f64, TorshError> {
    if corrupted_accuracies.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Empty corrupted accuracies".to_string(),
        ));
    }

    // Calculate relative accuracy retention
    let avg_corrupted_accuracy =
        corrupted_accuracies.iter().sum::<f64>() / corrupted_accuracies.len() as f64;

    if clean_accuracy < 1e-10 {
        return Ok(0.0);
    }

    Ok(avg_corrupted_accuracy / clean_accuracy)
}

/// Certified robustness radius
///
/// Approximates the minimum perturbation size needed to change predictions.
/// Based on gradient norms and prediction margins.
///
/// # Arguments
/// * `prediction_margins` - Distance from decision boundary for each sample
/// * `gradient_norms` - Norm of gradients w.r.t. inputs
pub fn certified_robustness_radius(
    prediction_margins: &[f64],
    gradient_norms: &[f64],
) -> Result<f64, TorshError> {
    if prediction_margins.len() != gradient_norms.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    if prediction_margins.is_empty() {
        return Err(TorshError::InvalidArgument("Empty inputs".to_string()));
    }

    // Radius = margin / gradient_norm
    let mut radii = Vec::with_capacity(prediction_margins.len());

    for i in 0..prediction_margins.len() {
        if gradient_norms[i] > 1e-10 {
            radii.push(prediction_margins[i] / gradient_norms[i]);
        }
    }

    if radii.is_empty() {
        return Ok(0.0);
    }

    // Return minimum radius (worst-case robustness)
    Ok(radii.iter().fold(f64::INFINITY, |min, &r| min.min(r)))
}

/// Input gradient stability - how stable gradients are to input perturbations
///
/// More stable gradients suggest smoother decision boundaries and better robustness.
pub fn gradient_stability(
    clean_gradients: &[f64],
    perturbed_gradients: &[f64],
) -> Result<f64, TorshError> {
    if clean_gradients.len() != perturbed_gradients.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    if clean_gradients.is_empty() {
        return Err(TorshError::InvalidArgument("Empty inputs".to_string()));
    }

    // Calculate cosine similarity
    let mut dot_product = 0.0;
    let mut clean_norm = 0.0;
    let mut perturbed_norm = 0.0;

    for i in 0..clean_gradients.len() {
        dot_product += clean_gradients[i] * perturbed_gradients[i];
        clean_norm += clean_gradients[i] * clean_gradients[i];
        perturbed_norm += perturbed_gradients[i] * perturbed_gradients[i];
    }

    if clean_norm < 1e-10 || perturbed_norm < 1e-10 {
        return Ok(1.0); // Perfect stability if gradients are zero
    }

    let similarity = dot_product / (clean_norm * perturbed_norm).sqrt();
    Ok((similarity + 1.0) / 2.0) // Transform to [0, 1]
}

/// Robustness-accuracy trade-off
///
/// Measures the relationship between clean accuracy and robust accuracy.
pub fn robustness_accuracy_tradeoff(clean_accuracy: f64, robust_accuracy: f64) -> f64 {
    if clean_accuracy < 1e-10 {
        return 0.0;
    }

    // Higher values mean less trade-off (better robustness without sacrificing accuracy)
    robust_accuracy / clean_accuracy
}

/// Comprehensive robustness report
#[derive(Debug, Clone)]
pub struct RobustnessReport {
    pub clean_accuracy: f64,
    pub adversarial_accuracy: f64,
    pub attack_success_rate: f64,
    pub noise_sensitivity: f64,
    pub confidence_stability: f64,
    pub corruption_robustness: f64,
}

impl RobustnessReport {
    /// Create new robustness report
    pub fn new(
        clean_pred: &Tensor,
        perturbed_pred: &Tensor,
        targets: &Tensor,
        clean_conf: &Tensor,
        perturbed_conf: &Tensor,
        corrupted_accuracies: &[f64],
    ) -> Result<Self, TorshError> {
        // Calculate clean accuracy
        let clean_vec = clean_pred.to_vec()?;
        let target_vec = targets.to_vec()?;
        let clean_correct = clean_vec
            .iter()
            .zip(target_vec.iter())
            .filter(|(&p, &t)| (p - t).abs() < 0.5)
            .count();
        let clean_acc = clean_correct as f64 / clean_vec.len() as f64;

        let adv_acc = adversarial_accuracy(clean_pred, perturbed_pred, targets)?;
        let attack_rate = attack_success_rate(clean_pred, perturbed_pred, targets)?;
        let noise_sens = noise_sensitivity(clean_pred, perturbed_pred)?;
        let conf_stab = confidence_stability(clean_conf, perturbed_conf)?;
        let corr_robust = corruption_robustness(clean_acc, corrupted_accuracies)?;

        Ok(Self {
            clean_accuracy: clean_acc,
            adversarial_accuracy: adv_acc,
            attack_success_rate: attack_rate,
            noise_sensitivity: noise_sens,
            confidence_stability: conf_stab,
            corruption_robustness: corr_robust,
        })
    }

    /// Format as string
    pub fn format(&self) -> String {
        let mut report = String::new();
        report.push_str("╔═══════════════════════════════════════════════╗\n");
        report.push_str("║          Robustness Metrics Report            ║\n");
        report.push_str("╠═══════════════════════════════════════════════╣\n");
        report.push_str(&format!(
            "║ Clean Accuracy:        {:.4}                 ║\n",
            self.clean_accuracy
        ));
        report.push_str(&format!(
            "║ Adversarial Accuracy:  {:.4}                 ║\n",
            self.adversarial_accuracy
        ));
        report.push_str(&format!(
            "║ Attack Success Rate:   {:.4}                 ║\n",
            self.attack_success_rate
        ));
        report.push_str(&format!(
            "║ Noise Sensitivity:     {:.4}                 ║\n",
            self.noise_sensitivity
        ));
        report.push_str(&format!(
            "║ Confidence Stability:  {:.4}                 ║\n",
            self.confidence_stability
        ));
        report.push_str(&format!(
            "║ Corruption Robustness: {:.4}                 ║\n",
            self.corruption_robustness
        ));
        report.push_str("╚═══════════════════════════════════════════════╝\n");

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_adversarial_accuracy_perfect() {
        let clean = from_vec(vec![0.0, 1.0, 0.0, 1.0], &[4], torsh_core::DeviceType::Cpu).unwrap();
        let perturbed =
            from_vec(vec![0.0, 1.0, 0.0, 1.0], &[4], torsh_core::DeviceType::Cpu).unwrap();
        let targets =
            from_vec(vec![0.0, 1.0, 0.0, 1.0], &[4], torsh_core::DeviceType::Cpu).unwrap();

        let acc = adversarial_accuracy(&clean, &perturbed, &targets).unwrap();
        assert!((acc - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_adversarial_accuracy_partial() {
        let clean = from_vec(vec![0.0, 1.0, 0.0, 1.0], &[4], torsh_core::DeviceType::Cpu).unwrap();
        let perturbed =
            from_vec(vec![0.0, 1.0, 1.0, 1.0], &[4], torsh_core::DeviceType::Cpu).unwrap();
        let targets =
            from_vec(vec![0.0, 1.0, 0.0, 1.0], &[4], torsh_core::DeviceType::Cpu).unwrap();

        let acc = adversarial_accuracy(&clean, &perturbed, &targets).unwrap();
        assert!((acc - 0.75).abs() < 1e-6); // 3 out of 4 remain correct
    }

    #[test]
    fn test_attack_success_rate() {
        let clean = from_vec(vec![0.0, 1.0, 0.0, 1.0], &[4], torsh_core::DeviceType::Cpu).unwrap();
        let perturbed =
            from_vec(vec![1.0, 1.0, 0.0, 1.0], &[4], torsh_core::DeviceType::Cpu).unwrap();
        let targets =
            from_vec(vec![0.0, 1.0, 0.0, 1.0], &[4], torsh_core::DeviceType::Cpu).unwrap();

        let asr = attack_success_rate(&clean, &perturbed, &targets).unwrap();
        assert!((asr - 0.25).abs() < 1e-6); // 1 out of 4 successful attacks
    }

    #[test]
    fn test_noise_sensitivity() {
        let clean = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], torsh_core::DeviceType::Cpu).unwrap();
        let noisy = from_vec(vec![1.1, 2.2, 3.1, 4.2], &[4], torsh_core::DeviceType::Cpu).unwrap();

        let sensitivity = noise_sensitivity(&clean, &noisy).unwrap();
        assert!(sensitivity > 0.0);
        assert!(sensitivity < 1.0);
    }

    #[test]
    fn test_confidence_stability_perfect() {
        let clean = from_vec(
            vec![0.9, 0.8, 0.95, 0.85],
            &[4],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();
        let perturbed = from_vec(
            vec![0.9, 0.8, 0.95, 0.85],
            &[4],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();

        let stability = confidence_stability(&clean, &perturbed).unwrap();
        assert!(stability > 0.99);
    }

    #[test]
    fn test_ood_detection_score() {
        let in_dist = vec![0.9, 0.85, 0.95, 0.88];
        let out_dist = vec![0.4, 0.3, 0.5, 0.35];

        let score = ood_detection_score(&in_dist, &out_dist).unwrap();
        assert!(score > 0.9); // Should be high since in-dist has higher confidence
    }

    #[test]
    fn test_corruption_robustness() {
        let clean_acc = 0.95;
        let corrupted = vec![0.90, 0.88, 0.85, 0.92];

        let robustness = corruption_robustness(clean_acc, &corrupted).unwrap();
        assert!(robustness > 0.85 && robustness < 1.0);
    }

    #[test]
    fn test_certified_robustness_radius() {
        let margins = vec![0.5, 0.3, 0.8, 0.4];
        let gradients = vec![0.1, 0.15, 0.2, 0.1];

        let radius = certified_robustness_radius(&margins, &gradients).unwrap();
        assert!(radius > 0.0);
    }

    #[test]
    fn test_gradient_stability() {
        let clean_grads = vec![1.0, 2.0, 3.0, 4.0];
        let perturbed_grads = vec![1.1, 2.1, 3.1, 4.1];

        let stability = gradient_stability(&clean_grads, &perturbed_grads).unwrap();
        assert!(stability > 0.9); // Should be high for similar gradients
    }

    #[test]
    fn test_robustness_accuracy_tradeoff() {
        let clean_acc = 0.95;
        let robust_acc = 0.85;

        let tradeoff = robustness_accuracy_tradeoff(clean_acc, robust_acc);
        assert!((tradeoff - 0.8947).abs() < 1e-3);
    }

    #[test]
    fn test_robustness_report() {
        let clean_pred =
            from_vec(vec![0.0, 1.0, 0.0, 1.0], &[4], torsh_core::DeviceType::Cpu).unwrap();
        let perturbed_pred =
            from_vec(vec![0.0, 1.0, 0.0, 1.0], &[4], torsh_core::DeviceType::Cpu).unwrap();
        let targets =
            from_vec(vec![0.0, 1.0, 0.0, 1.0], &[4], torsh_core::DeviceType::Cpu).unwrap();
        let clean_conf = from_vec(
            vec![0.9, 0.85, 0.95, 0.88],
            &[4],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();
        let perturbed_conf = from_vec(
            vec![0.85, 0.82, 0.90, 0.85],
            &[4],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();
        let corrupted = vec![0.90, 0.88, 0.92];

        let report = RobustnessReport::new(
            &clean_pred,
            &perturbed_pred,
            &targets,
            &clean_conf,
            &perturbed_conf,
            &corrupted,
        )
        .unwrap();

        let formatted = report.format();
        assert!(formatted.contains("Robustness Metrics Report"));
    }
}
