//! Integration tests for torsh-metrics

use approx::assert_relative_eq;
use torsh_core::device::DeviceType;
use torsh_metrics::{
    classification::{Accuracy, F1Score, Precision, Recall},
    regression::{ExplainedVariance, HuberLoss, R2Score, MAE, MAPE, MSE, RMSE},
    utils::{bootstrap_ci, compute_class_weights, probs_to_preds},
    Metric, MetricCollection,
};
use torsh_tensor::{creation::from_vec, Tensor};

/// Helper function to create tensor from array
fn tensor_from_slice(data: &[f32]) -> Tensor {
    from_vec(data.to_vec(), &[data.len()], DeviceType::Cpu).unwrap()
}

/// Helper function to create tensor from 2D array
fn tensor_from_2d(data: &[&[f32]]) -> Tensor {
    let flat: Vec<f32> = data.iter().flat_map(|row| row.iter()).copied().collect();
    let rows = data.len();
    let cols = data[0].len();
    from_vec(flat, &[rows, cols], DeviceType::Cpu).unwrap()
}

#[cfg(test)]
mod classification_tests {
    use super::*;

    #[test]
    fn test_accuracy_perfect_prediction() {
        // Perfect binary classification
        let predictions = tensor_from_2d(&[
            &[0.1, 0.9], // Class 1
            &[0.8, 0.2], // Class 0
            &[0.3, 0.7], // Class 1
            &[0.9, 0.1], // Class 0
        ]);
        let targets = tensor_from_slice(&[1.0, 0.0, 1.0, 0.0]);

        let accuracy = Accuracy::new();
        let result = accuracy.compute(&predictions, &targets);
        assert_relative_eq!(result, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_accuracy_imperfect_prediction() {
        // Imperfect binary classification
        let predictions = tensor_from_2d(&[
            &[0.1, 0.9], // Correct: Class 1
            &[0.2, 0.8], // Incorrect: predicted 1, actual 0
            &[0.7, 0.3], // Incorrect: predicted 0, actual 1
            &[0.9, 0.1], // Correct: Class 0
        ]);
        let targets = tensor_from_slice(&[1.0, 0.0, 1.0, 0.0]);

        let accuracy = Accuracy::new();
        let result = accuracy.compute(&predictions, &targets);
        assert_relative_eq!(result, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_top_k_accuracy() {
        // Multi-class with top-2 accuracy
        let predictions = tensor_from_2d(&[
            &[0.1, 0.2, 0.7], // Top-2: [2, 1], target: 2 ✓
            &[0.3, 0.6, 0.1], // Top-2: [1, 0], target: 0 ✓
            &[0.8, 0.1, 0.1], // Top-2: [0, 1], target: 2 ✗
        ]);
        let targets = tensor_from_slice(&[2.0, 0.0, 2.0]);

        let accuracy = Accuracy::top_k(2);
        let result = accuracy.compute(&predictions, &targets);
        assert_relative_eq!(result, 2.0 / 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_precision_binary() {
        let predictions = tensor_from_2d(&[
            &[0.1, 0.9], // Predicted: 1, Actual: 1 (TP)
            &[0.7, 0.3], // Predicted: 0, Actual: 1 (FN)
            &[0.2, 0.8], // Predicted: 1, Actual: 0 (FP)
            &[0.9, 0.1], // Predicted: 0, Actual: 0 (TN)
        ]);
        let targets = tensor_from_slice(&[1.0, 1.0, 0.0, 0.0]);

        // TP=1, FP=1, so precision = 1/(1+1) = 0.5
        let precision = Precision::micro();
        let result = precision.compute(&predictions, &targets);
        assert_relative_eq!(result, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_recall_binary() {
        let predictions = tensor_from_2d(&[
            &[0.1, 0.9], // Predicted: 1, Actual: 1 (TP)
            &[0.7, 0.3], // Predicted: 0, Actual: 1 (FN)
            &[0.2, 0.8], // Predicted: 1, Actual: 0 (FP)
            &[0.9, 0.1], // Predicted: 0, Actual: 0 (TN)
        ]);
        let targets = tensor_from_slice(&[1.0, 1.0, 0.0, 0.0]);

        // TP=1, FN=1, so recall = 1/(1+1) = 0.5
        let recall = Recall::micro();
        let result = recall.compute(&predictions, &targets);
        assert_relative_eq!(result, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_f1_score_binary() {
        let predictions = tensor_from_2d(&[
            &[0.1, 0.9], // Predicted: 1, Actual: 1 (TP)
            &[0.7, 0.3], // Predicted: 0, Actual: 1 (FN)
            &[0.2, 0.8], // Predicted: 1, Actual: 0 (FP)
            &[0.9, 0.1], // Predicted: 0, Actual: 0 (TN)
        ]);
        let targets = tensor_from_slice(&[1.0, 1.0, 0.0, 0.0]);

        // Precision = 0.5, Recall = 0.5, F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        let f1 = F1Score::micro();
        let result = f1.compute(&predictions, &targets);
        assert_relative_eq!(result, 0.5, epsilon = 1e-6);
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    #[test]
    fn test_mse_perfect_prediction() {
        let predictions = tensor_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let targets = tensor_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let mse = MSE;
        let result = mse.compute(&predictions, &targets);
        assert_relative_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mse_calculation() {
        let predictions = tensor_from_slice(&[1.0, 2.0, 3.0]);
        let targets = tensor_from_slice(&[1.5, 2.5, 2.5]);

        // Errors: [-0.5, -0.5, 0.5]
        // Squared errors: [0.25, 0.25, 0.25]
        // MSE = (0.25 + 0.25 + 0.25) / 3 = 0.25
        let mse = MSE;
        let result = mse.compute(&predictions, &targets);
        assert_relative_eq!(result, 0.25, epsilon = 1e-6);
    }

    #[test]
    fn test_rmse_calculation() {
        let predictions = tensor_from_slice(&[1.0, 2.0, 3.0]);
        let targets = tensor_from_slice(&[1.5, 2.5, 2.5]);

        // MSE = 0.25, RMSE = sqrt(0.25) = 0.5
        let rmse = RMSE;
        let result = rmse.compute(&predictions, &targets);
        assert_relative_eq!(result, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_mae_calculation() {
        let predictions = tensor_from_slice(&[1.0, 2.0, 3.0]);
        let targets = tensor_from_slice(&[1.5, 2.5, 2.5]);

        // Errors: [0.5, 0.5, 0.5]
        // MAE = (0.5 + 0.5 + 0.5) / 3 = 0.5
        let mae = MAE;
        let result = mae.compute(&predictions, &targets);
        assert_relative_eq!(result, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_r2_score_perfect() {
        let predictions = tensor_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let targets = tensor_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let r2 = R2Score::new();
        let result = r2.compute(&predictions, &targets);
        assert_relative_eq!(result, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_r2_score_calculation() {
        let predictions = tensor_from_slice(&[2.5, 0.0, 2.0, 8.0]);
        let targets = tensor_from_slice(&[3.0, -0.5, 2.0, 7.0]);

        // Mean of targets = (3.0 - 0.5 + 2.0 + 7.0) / 4 = 2.875
        // TSS = (3-2.875)^2 + (-0.5-2.875)^2 + (2-2.875)^2 + (7-2.875)^2
        //     = 0.015625 + 11.390625 + 0.765625 + 17.015625 = 29.1875
        // RSS = (3-2.5)^2 + (-0.5-0)^2 + (2-2)^2 + (7-8)^2
        //     = 0.25 + 0.25 + 0 + 1 = 1.5
        // R² = 1 - 1.5/29.1875 ≈ 0.949

        let r2 = R2Score::new();
        let result = r2.compute(&predictions, &targets);
        assert_relative_eq!(result, 0.9486081, epsilon = 1e-5);
    }

    #[test]
    fn test_mape_calculation() {
        let predictions = tensor_from_slice(&[90.0, 110.0, 95.0]);
        let targets = tensor_from_slice(&[100.0, 100.0, 100.0]);

        // Percentage errors: |90-100|/100, |110-100|/100, |95-100|/100
        //                  = 0.1, 0.1, 0.05
        // MAPE = (0.1 + 0.1 + 0.05) / 3 * 100 = 8.333%
        let mape = MAPE::new();
        let result = mape.compute(&predictions, &targets);
        assert_relative_eq!(result, 8.333333, epsilon = 1e-5);
    }

    #[test]
    fn test_huber_loss_small_errors() {
        let predictions = tensor_from_slice(&[1.0, 2.0, 3.0]);
        let targets = tensor_from_slice(&[1.1, 2.1, 2.9]);
        let delta = 0.5;

        // All errors (0.1, 0.1, 0.1) are <= delta (0.5)
        // Use quadratic: 0.5 * error^2
        // Loss = (0.5 * 0.01 + 0.5 * 0.01 + 0.5 * 0.01) / 3 = 0.005
        let huber = HuberLoss::new(delta);
        let result = huber.compute(&predictions, &targets);
        assert_relative_eq!(result, 0.005, epsilon = 1e-6);
    }

    #[test]
    fn test_huber_loss_large_errors() {
        let predictions = tensor_from_slice(&[1.0, 2.0]);
        let targets = tensor_from_slice(&[2.0, 4.0]);
        let delta = 0.5;

        // Errors (1.0, 2.0) are > delta (0.5)
        // Use linear: delta * |error| - 0.5 * delta^2
        // Error 1: 0.5 * 1.0 - 0.5 * 0.25 = 0.375
        // Error 2: 0.5 * 2.0 - 0.5 * 0.25 = 0.875
        // Loss = (0.375 + 0.875) / 2 = 0.625
        let huber = HuberLoss::new(delta);
        let result = huber.compute(&predictions, &targets);
        assert_relative_eq!(result, 0.625, epsilon = 1e-6);
    }

    #[test]
    fn test_explained_variance() {
        let predictions = tensor_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let targets = tensor_from_slice(&[1.1, 1.9, 3.1, 3.9]);

        let ev = ExplainedVariance::new();
        let result = ev.compute(&predictions, &targets);
        // Should be close to 1.0 for good predictions
        assert!(result > 0.9);
    }
}

#[cfg(test)]
mod utility_tests {
    use super::*;

    #[test]
    fn test_probs_to_preds_binary() {
        let probs = tensor_from_2d(&[
            &[0.3, 0.7], // Should predict class 1
            &[0.8, 0.2], // Should predict class 0
            &[0.4, 0.6], // Should predict class 1
        ]);

        let preds = probs_to_preds(&probs, 0.5).unwrap();
        let preds_vec = preds.to_vec().unwrap();
        assert_eq!(preds_vec, &[1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_bootstrap_ci() {
        let scores = &[0.1, 0.2, 0.15, 0.18, 0.22, 0.16, 0.19, 0.21, 0.17, 0.20];
        let (lower, upper) = bootstrap_ci(scores, 0.95, 1000, Some(42));

        // Should be reasonable confidence interval around the mean (0.178)
        assert!(lower < upper);
        assert!(lower > 0.1);
        assert!(upper < 0.25);
    }

    #[test]
    fn test_compute_class_weights() {
        // Imbalanced dataset: 1 sample of class 0, 3 samples of class 1
        let labels = tensor_from_slice(&[0.0, 1.0, 1.0, 1.0]);
        let weights = compute_class_weights(&labels, 2).unwrap();
        let weights_vec = weights.to_vec().unwrap();

        // Weight for class 0: 4 / (2 * 1) = 2.0
        // Weight for class 1: 4 / (2 * 3) = 0.667
        assert_relative_eq!(weights_vec[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(weights_vec[1], 0.666667, epsilon = 1e-5);
    }
}

#[cfg(test)]
mod collection_tests {
    use super::*;

    #[test]
    fn test_metric_collection() {
        let predictions = tensor_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let targets = tensor_from_slice(&[1.1, 1.9, 3.1, 3.9]);

        let mut collection = MetricCollection::new()
            .add(MSE)
            .add(MAE)
            .add(R2Score::new());

        let results = collection.compute(&predictions, &targets);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, "mse");
        assert_eq!(results[1].0, "mae");
        assert_eq!(results[2].0, "r2_score");

        // All metrics should return reasonable values
        assert!(results[0].1 > 0.0); // MSE should be positive
        assert!(results[1].1 > 0.0); // MAE should be positive
        assert!(results[2].1 > 0.5); // R2 should be decent for this data
    }

    #[test]
    fn test_metric_collection_format() {
        let predictions = tensor_from_slice(&[1.0, 2.0, 3.0]);
        let targets = tensor_from_slice(&[1.0, 2.0, 3.0]);

        let mut collection = MetricCollection::new().add(MSE);

        collection.compute(&predictions, &targets);
        let formatted = collection.format_results();

        assert!(formatted.contains("mse"));
        assert!(formatted.contains("0.0000"));
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn test_empty_tensors() {
        let empty_tensor = tensor_from_slice(&[]);
        let accuracy = Accuracy::new();
        let result = accuracy.compute(&empty_tensor, &empty_tensor);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_mismatched_sizes() {
        let predictions = tensor_from_slice(&[1.0, 2.0]);
        let targets = tensor_from_slice(&[1.0]);

        let mse = MSE;
        let result = mse.compute(&predictions, &targets);
        assert_eq!(result, 0.0); // Should handle gracefully
    }

    #[test]
    fn test_mape_with_zero_targets() {
        let predictions = tensor_from_slice(&[1.0, 2.0]);
        let targets = tensor_from_slice(&[0.0, 2.0]); // First target is zero

        let mape = MAPE::new();
        let result = mape.compute(&predictions, &targets);
        // Should skip zero targets and compute on remaining ones
        assert_relative_eq!(result, 0.0, epsilon = 1e-6); // Only second prediction is perfect
    }
}
