//! Utility functions for metrics

use scirs2_core::legacy::rng;
use scirs2_core::random::{Random, Rng};
use torsh_core::{device::DeviceType, error::TorshError};
use torsh_tensor::{
    creation::{from_vec, zeros},
    Tensor,
};

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
    let batch_size = labels.shape().dims()[0];
    let mut one_hot = zeros(&[batch_size, num_classes])?;

    for i in 0..batch_size {
        let label = labels
            .slice_tensor(0, i, i + 1)?
            .to_vec()?
            .get(0)
            .copied()
            .unwrap_or(0.0) as usize;
        if label < num_classes {
            // TODO: Implement proper one-hot encoding when tensor indexing is available
            // For now, this is a placeholder implementation
        }
    }

    Ok(one_hot)
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
    use scirs2_core::random::{Random, Rng};

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
