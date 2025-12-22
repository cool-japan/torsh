//! Utility functions for metrics

use torsh_core::{device::DeviceType, error::TorshError};
use torsh_tensor::{creation::from_vec, Tensor};

/// Convert probabilities to class predictions
pub fn probs_to_preds(probs: &Tensor, threshold: f64) -> Result<Tensor, TorshError> {
    let shape = probs.shape();
    let dims = shape.dims();

    if dims.len() == 1 || (dims.len() == 2 && dims[1] == 1) {
        // Binary classification - use threshold
        match probs.to_vec() {
            Ok(probs_vec) => {
                let preds: Vec<f32> = probs_vec
                    .iter()
                    .map(|&p| if p >= threshold as f32 { 1.0 } else { 0.0 })
                    .collect();

                from_vec(
                    preds,
                    &[probs_vec.len()], // Always output as 1D for binary classification
                    torsh_core::device::DeviceType::Cpu,
                )
            }
            Err(e) => Err(e),
        }
    } else if dims.len() == 2 {
        // Multi-class classification - use manual argmax
        match probs.to_vec() {
            Ok(probs_vec) => {
                let rows = dims[0];
                let cols = dims[1];

                if rows == 0 || cols == 0 {
                    return Err(TorshError::InvalidArgument("Empty tensor".to_string()));
                }

                let mut preds = Vec::with_capacity(rows);

                // Manually compute argmax for each row
                for i in 0..rows {
                    let mut max_idx = 0;
                    let mut max_val = probs_vec[i * cols];

                    for j in 1..cols {
                        let val = probs_vec[i * cols + j];
                        if val > max_val {
                            max_val = val;
                            max_idx = j;
                        }
                    }
                    preds.push(max_idx as f32);
                }

                from_vec(preds, &[rows], torsh_core::device::DeviceType::Cpu)
            }
            Err(e) => Err(e),
        }
    } else {
        Err(TorshError::InvalidArgument(
            "Input tensor must be 1D or 2D".to_string(),
        ))
    }
}

/// One-hot encode labels
pub fn one_hot(labels: &Tensor, num_classes: usize) -> Result<Tensor, TorshError> {
    let labels_vec = labels.to_vec()?;
    let batch_size = labels_vec.len();

    // Create one-hot encoded data
    let mut one_hot_data = vec![0.0f32; batch_size * num_classes];

    for (i, &label) in labels_vec.iter().enumerate() {
        let label_idx = label as usize;
        if label_idx < num_classes {
            one_hot_data[i * num_classes + label_idx] = 1.0;
        } else {
            return Err(TorshError::InvalidArgument(format!(
                "Label {} exceeds num_classes {}",
                label_idx, num_classes
            )));
        }
    }

    from_vec(one_hot_data, &[batch_size, num_classes], DeviceType::Cpu)
}

/// Calculate class weights for imbalanced datasets
pub fn compute_class_weights(labels: &Tensor, num_classes: usize) -> Result<Tensor, TorshError> {
    let mut counts = vec![0.0; num_classes];
    let labels_vec = labels.to_vec()?;

    for label in &labels_vec {
        let label_idx = *label as usize;
        if label_idx < num_classes {
            counts[label_idx] += 1.0;
        }
    }

    let total = labels_vec.len() as f64;
    let weights: Vec<f64> = counts
        .iter()
        .map(|&count| {
            if count > 0.0 {
                total / (num_classes as f64 * count)
            } else {
                0.0
            }
        })
        .collect();

    let weights_f32: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
    Ok(from_vec(weights_f32, &[num_classes], DeviceType::Cpu)?)
}

/// Bootstrap confidence intervals
pub fn bootstrap_ci(
    scores: &[f64],
    confidence: f64,
    n_bootstrap: usize,
    seed: Option<u64>,
) -> (f64, f64) {
    use scirs2_core::random::Random;

    let mut rng = Random::seed(seed.unwrap_or_else(|| {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }));

    let mut bootstrap_scores = Vec::with_capacity(n_bootstrap);
    let n = scores.len();

    for _ in 0..n_bootstrap {
        let mut sample_sum = 0.0;
        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            sample_sum += scores[idx];
        }
        bootstrap_scores.push(sample_sum / n as f64);
    }

    bootstrap_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let alpha = (1.0 - confidence) / 2.0;
    let lower_idx = ((alpha * n_bootstrap as f64) as usize).max(0);
    let upper_idx = (((1.0 - alpha) * n_bootstrap as f64) as usize).min(n_bootstrap - 1);

    (bootstrap_scores[lower_idx], bootstrap_scores[upper_idx])
}

/// Calculate optimal threshold for binary classification
pub fn find_optimal_threshold(
    probs: &Tensor,
    targets: &Tensor,
    metric: OptimizeMetric,
) -> Result<f64, TorshError> {
    let probs_vec = probs.to_vec()?;
    let targets_vec = targets.to_vec()?;

    let mut thresholds: Vec<f32> = probs_vec.clone();
    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    thresholds.dedup();

    let mut best_threshold = 0.5_f32;
    let mut best_score = f64::NEG_INFINITY;

    for &threshold in &thresholds {
        let preds: Vec<i64> = probs_vec
            .iter()
            .map(|&p| if p >= threshold { 1 } else { 0 })
            .collect();

        let targets_i64: Vec<i64> = targets_vec.iter().map(|&x| x as i64).collect();

        let score = match metric {
            OptimizeMetric::F1 => calculate_f1(&preds, &targets_i64),
            OptimizeMetric::Accuracy => calculate_accuracy(&preds, &targets_i64),
            OptimizeMetric::Balanced => calculate_balanced_accuracy(&preds, &targets_i64),
        };

        if score > best_score {
            best_score = score;
            best_threshold = threshold;
        }
    }

    Ok(best_threshold as f64)
}

#[derive(Debug, Clone, Copy)]
pub enum OptimizeMetric {
    F1,
    Accuracy,
    Balanced,
}

fn calculate_f1(preds: &[i64], targets: &[i64]) -> f64 {
    let (tp, fp, fn_count) = calculate_confusion_elements(preds, targets);
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_count > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        0.0
    };

    if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

fn calculate_accuracy(preds: &[i64], targets: &[i64]) -> f64 {
    let correct: usize = preds
        .iter()
        .zip(targets.iter())
        .filter(|(p, t)| p == t)
        .count();
    correct as f64 / preds.len() as f64
}

fn calculate_balanced_accuracy(preds: &[i64], targets: &[i64]) -> f64 {
    let (tp, tn, fp, fn_count) = calculate_confusion_matrix(preds, targets);
    let sensitivity = if tp + fn_count > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        0.0
    };
    let specificity = if tn + fp > 0 {
        tn as f64 / (tn + fp) as f64
    } else {
        0.0
    };
    (sensitivity + specificity) / 2.0
}

fn calculate_confusion_elements(preds: &[i64], targets: &[i64]) -> (usize, usize, usize) {
    let mut tp = 0;
    let mut fp = 0;
    let mut fn_count = 0;

    for (&pred, &target) in preds.iter().zip(targets.iter()) {
        if pred == 1 && target == 1 {
            tp += 1;
        } else if pred == 1 && target == 0 {
            fp += 1;
        } else if pred == 0 && target == 1 {
            fn_count += 1;
        }
    }

    (tp, fp, fn_count)
}

fn calculate_confusion_matrix(preds: &[i64], targets: &[i64]) -> (usize, usize, usize, usize) {
    let mut tp = 0;
    let mut tn = 0;
    let mut fp = 0;
    let mut fn_count = 0;

    for (&pred, &target) in preds.iter().zip(targets.iter()) {
        match (pred, target) {
            (1, 1) => tp += 1,
            (0, 0) => tn += 1,
            (1, 0) => fp += 1,
            (0, 1) => fn_count += 1,
            _ => {}
        }
    }

    (tp, tn, fp, fn_count)
}

/// Comprehensive metric evaluation result
#[derive(Debug, Clone)]
pub struct MetricEvaluationResult {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub confusion_matrix: Option<Vec<Vec<usize>>>,
}

/// Evaluate multiple classification metrics at once
pub fn evaluate_classification(
    predictions: &Tensor,
    targets: &Tensor,
) -> Result<MetricEvaluationResult, TorshError> {
    let pred_vec = predictions.to_vec()?;
    let target_vec = targets.to_vec()?;

    if pred_vec.len() != target_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    let preds: Vec<i64> = pred_vec.iter().map(|&x| x as i64).collect();
    let targets_i64: Vec<i64> = target_vec.iter().map(|&x| x as i64).collect();

    let accuracy = calculate_accuracy(&preds, &targets_i64);
    let f1_score = calculate_f1(&preds, &targets_i64);

    let (tp, fp, fn_count) = calculate_confusion_elements(&preds, &targets_i64);
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_count > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        0.0
    };

    Ok(MetricEvaluationResult {
        accuracy,
        precision,
        recall,
        f1_score,
        confusion_matrix: None,
    })
}

/// Quick evaluation helper - compute common metrics
pub fn quick_eval(predictions: &Tensor, targets: &Tensor) -> String {
    match evaluate_classification(predictions, targets) {
        Ok(result) => format!(
            "Accuracy: {:.4} | Precision: {:.4} | Recall: {:.4} | F1: {:.4}",
            result.accuracy, result.precision, result.recall, result.f1_score
        ),
        Err(e) => format!("Error: {:?}", e),
    }
}

/// Compute prediction confidence statistics
pub fn confidence_statistics(probabilities: &Tensor) -> Result<(f64, f64, f64, f64), TorshError> {
    let prob_vec = probabilities.to_vec()?;

    if prob_vec.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Empty probabilities".to_string(),
        ));
    }

    let prob_f64: Vec<f64> = prob_vec.iter().map(|&x| x as f64).collect();

    // Mean confidence
    let mean = prob_f64.iter().sum::<f64>() / prob_f64.len() as f64;

    // Std deviation
    let variance =
        prob_f64.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / prob_f64.len() as f64;
    let std_dev = variance.sqrt();

    // Min and max
    let min = prob_f64.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = prob_f64.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    Ok((mean, std_dev, min, max))
}

/// Check if predictions are well-calibrated (confidence matches accuracy)
pub fn calibration_check(
    probabilities: &Tensor,
    predictions: &Tensor,
    targets: &Tensor,
) -> Result<bool, TorshError> {
    let prob_vec = probabilities.to_vec()?;
    let pred_vec = predictions.to_vec()?;
    let target_vec = targets.to_vec()?;

    if prob_vec.len() != pred_vec.len() || prob_vec.len() != target_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    // Compute average confidence
    let avg_confidence = prob_vec.iter().map(|&x| x as f64).sum::<f64>() / prob_vec.len() as f64;

    // Compute accuracy
    let correct: usize = pred_vec
        .iter()
        .zip(target_vec.iter())
        .filter(|(p, t)| (**p - **t).abs() < 1e-6)
        .count();
    let accuracy = correct as f64 / pred_vec.len() as f64;

    // Well-calibrated if confidence is within 5% of accuracy
    Ok((avg_confidence - accuracy).abs() < 0.05)
}

/// Detect if model is overconfident or underconfident
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibrationStatus {
    WellCalibrated,
    Overconfident,
    Underconfident,
}

pub fn detect_calibration_status(
    probabilities: &Tensor,
    predictions: &Tensor,
    targets: &Tensor,
) -> Result<CalibrationStatus, TorshError> {
    let prob_vec = probabilities.to_vec()?;
    let pred_vec = predictions.to_vec()?;
    let target_vec = targets.to_vec()?;

    if prob_vec.len() != pred_vec.len() || prob_vec.len() != target_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    let avg_confidence = prob_vec.iter().map(|&x| x as f64).sum::<f64>() / prob_vec.len() as f64;

    let correct: usize = pred_vec
        .iter()
        .zip(target_vec.iter())
        .filter(|(p, t)| (**p - **t).abs() < 1e-6)
        .count();
    let accuracy = correct as f64 / pred_vec.len() as f64;

    let diff = avg_confidence - accuracy;

    if diff.abs() < 0.05 {
        Ok(CalibrationStatus::WellCalibrated)
    } else if diff > 0.0 {
        Ok(CalibrationStatus::Overconfident)
    } else {
        Ok(CalibrationStatus::Underconfident)
    }
}

/// Format a comprehensive metric report
pub fn format_metric_report(
    accuracy: f64,
    precision: f64,
    recall: f64,
    f1: f64,
    additional_info: Option<&str>,
) -> String {
    let mut report = String::new();
    report.push_str("╔═══════════════════════════════════════════════╗\n");
    report.push_str("║         Classification Metrics Report         ║\n");
    report.push_str("╠═══════════════════════════════════════════════╣\n");
    report.push_str(&format!(
        "║ Accuracy:  {:.4} ({:.2}%)                    ║\n",
        accuracy,
        accuracy * 100.0
    ));
    report.push_str(&format!(
        "║ Precision: {:.4} ({:.2}%)                    ║\n",
        precision,
        precision * 100.0
    ));
    report.push_str(&format!(
        "║ Recall:    {:.4} ({:.2}%)                    ║\n",
        recall,
        recall * 100.0
    ));
    report.push_str(&format!(
        "║ F1-Score:  {:.4} ({:.2}%)                    ║\n",
        f1,
        f1 * 100.0
    ));

    if let Some(info) = additional_info {
        report.push_str("╠═══════════════════════════════════════════════╣\n");
        report.push_str(&format!("║ {:<45} ║\n", info));
    }

    report.push_str("╚═══════════════════════════════════════════════╝\n");

    report
}

/// Helper to create visualization-ready data from predictions
pub fn prepare_confusion_matrix_data(
    predictions: &Tensor,
    targets: &Tensor,
    num_classes: usize,
) -> Result<Vec<Vec<usize>>, TorshError> {
    let pred_vec = predictions.to_vec()?;
    let target_vec = targets.to_vec()?;

    if pred_vec.len() != target_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    let mut matrix = vec![vec![0; num_classes]; num_classes];

    for (pred, target) in pred_vec.iter().zip(target_vec.iter()) {
        let pred_class = (*pred as usize).min(num_classes - 1);
        let target_class = (*target as usize).min(num_classes - 1);
        matrix[target_class][pred_class] += 1;
    }

    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_one_hot_encoding_basic() {
        let labels = from_vec(vec![0.0, 1.0, 2.0], &[3], DeviceType::Cpu).unwrap();
        let one_hot_result = one_hot(&labels, 3).unwrap();
        let result_vec = one_hot_result.to_vec().unwrap();

        // Expected: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        let expected = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(result_vec, expected);
    }

    #[test]
    fn test_one_hot_encoding_with_duplicates() {
        let labels = from_vec(vec![0.0, 0.0, 1.0, 2.0, 1.0], &[5], DeviceType::Cpu).unwrap();
        let one_hot_result = one_hot(&labels, 3).unwrap();
        let result_vec = one_hot_result.to_vec().unwrap();

        // Expected: [[1,0,0], [1,0,0], [0,1,0], [0,0,1], [0,1,0]]
        let expected = vec![
            1.0, 0.0, 0.0, // Label 0
            1.0, 0.0, 0.0, // Label 0
            0.0, 1.0, 0.0, // Label 1
            0.0, 0.0, 1.0, // Label 2
            0.0, 1.0, 0.0, // Label 1
        ];
        assert_eq!(result_vec, expected);
    }

    #[test]
    fn test_one_hot_encoding_invalid_label() {
        let labels = from_vec(vec![0.0, 1.0, 3.0], &[3], DeviceType::Cpu).unwrap();
        let result = one_hot(&labels, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_one_hot_encoding_single_class() {
        let labels = from_vec(vec![0.0], &[1], DeviceType::Cpu).unwrap();
        let one_hot_result = one_hot(&labels, 1).unwrap();
        let result_vec = one_hot_result.to_vec().unwrap();

        assert_eq!(result_vec, vec![1.0]);
    }
}
