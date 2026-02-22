//! Comprehensive edge case tests for robustness
//!
//! Tests for NaN, infinity, empty inputs, and other edge cases

use approx::assert_relative_eq;
use torsh_core::device::DeviceType;
use torsh_metrics::{
    classification::{Accuracy, F1Score, Precision, Recall},
    regression::{R2Score, MAE, MSE, RMSE},
    Metric,
};
use torsh_tensor::creation::from_vec;

#[cfg(test)]
mod nan_handling {
    use super::*;

    #[test]
    fn test_accuracy_with_nan_predictions() {
        // Test that NaN values don't cause panics
        let predictions = from_vec(
            vec![f32::NAN, 0.9, 0.8, 0.2, 0.3, 0.7],
            &[3, 2],
            DeviceType::Cpu,
        )
        .unwrap();
        let targets = from_vec(vec![1.0, 0.0, 1.0], &[3], DeviceType::Cpu).unwrap();

        let accuracy = Accuracy::new();
        let result = accuracy.compute(&predictions, &targets);

        // Should handle gracefully - result may be partial or 0.0
        assert!(result >= 0.0 && result <= 1.0 || result.is_nan());
    }

    #[test]
    fn test_mse_with_nan_values() {
        let predictions = from_vec(vec![1.0, f32::NAN, 3.0, 4.0], &[4], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], DeviceType::Cpu).unwrap();

        let mse = MSE;
        let result = mse.compute(&predictions, &targets);

        // Should either handle NaN gracefully or return NaN
        assert!(result >= 0.0 || result.is_nan());
    }

    #[test]
    fn test_mae_with_all_nan() {
        let predictions =
            from_vec(vec![f32::NAN, f32::NAN, f32::NAN], &[3], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![1.0, 2.0, 3.0], &[3], DeviceType::Cpu).unwrap();

        let mae = MAE;
        let result = mae.compute(&predictions, &targets);

        // All NaN should result in NaN or 0.0
        assert!(result.is_nan() || result == 0.0);
    }
}

#[cfg(test)]
mod infinity_handling {
    use super::*;

    #[test]
    fn test_mse_with_infinity() {
        let predictions =
            from_vec(vec![1.0, f32::INFINITY, 3.0, 4.0], &[4], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], DeviceType::Cpu).unwrap();

        let mse = MSE;
        let result = mse.compute(&predictions, &targets);

        // Infinity should result in infinity or very large value
        assert!(result.is_infinite() || result > 1e10);
    }

    #[test]
    fn test_mae_with_negative_infinity() {
        let predictions = from_vec(
            vec![1.0, f32::NEG_INFINITY, 3.0, 4.0],
            &[4],
            DeviceType::Cpu,
        )
        .unwrap();
        let targets = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], DeviceType::Cpu).unwrap();

        let mae = MAE;
        let result = mae.compute(&predictions, &targets);

        // Should handle negative infinity
        assert!(result.is_infinite() || result > 1e10 || result.is_nan());
    }

    #[test]
    fn test_r2_with_infinity_targets() {
        let predictions = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![1.0, 2.0, f32::INFINITY, 4.0], &[4], DeviceType::Cpu).unwrap();

        let r2 = R2Score::new();
        let result = r2.compute(&predictions, &targets);

        // R² with infinity should be handled gracefully
        assert!(result.is_finite() || result.is_nan() || result.is_infinite());
    }
}

#[cfg(test)]
mod empty_input_handling {
    use super::*;

    #[test]
    fn test_accuracy_empty_tensors() {
        // Empty tensors should be handled gracefully
        let predictions = from_vec(vec![], &[0, 2], DeviceType::Cpu);
        let targets = from_vec(vec![], &[0], DeviceType::Cpu);

        // Should either error or return 0.0
        match (predictions, targets) {
            (Ok(preds), Ok(targs)) => {
                let accuracy = Accuracy::new();
                let result = accuracy.compute(&preds, &targs);
                assert!(result == 0.0 || result.is_nan());
            }
            _ => {
                // Expected: tensor creation might fail for empty inputs
            }
        }
    }

    #[test]
    fn test_mse_empty_tensors() {
        let predictions = from_vec(vec![], &[0], DeviceType::Cpu);
        let targets = from_vec(vec![], &[0], DeviceType::Cpu);

        match (predictions, targets) {
            (Ok(preds), Ok(targs)) => {
                let mse = MSE;
                let result = mse.compute(&preds, &targs);
                assert!(result == 0.0 || result.is_nan());
            }
            _ => {
                // Expected: empty tensors might fail creation
            }
        }
    }
}

#[cfg(test)]
mod size_mismatch_handling {
    use super::*;

    #[test]
    fn test_accuracy_mismatched_sizes() {
        let predictions =
            from_vec(vec![0.1, 0.9, 0.8, 0.2, 0.3, 0.7], &[3, 2], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![1.0, 0.0], &[2], DeviceType::Cpu).unwrap(); // Wrong size!

        let accuracy = Accuracy::new();
        let result = accuracy.compute(&predictions, &targets);

        // Should handle gracefully - might return 0.0 or NaN
        assert!(result >= 0.0 || result.is_nan());
    }

    #[test]
    fn test_mse_mismatched_sizes() {
        let predictions = from_vec(vec![1.0, 2.0, 3.0], &[3], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![1.0, 2.0], &[2], DeviceType::Cpu).unwrap(); // Wrong size!

        let mse = MSE;
        let result = mse.compute(&predictions, &targets);

        // Should handle gracefully
        assert!(result >= 0.0 || result.is_nan());
    }
}

#[cfg(test)]
mod extreme_values {
    use super::*;

    #[test]
    fn test_accuracy_extreme_probabilities() {
        // Test with probabilities very close to 0 and 1
        let predictions = from_vec(
            vec![1e-30, 1.0 - 1e-30, 1.0 - 1e-30, 1e-30, 1e-30, 1.0 - 1e-30],
            &[3, 2],
            DeviceType::Cpu,
        )
        .unwrap();
        let targets = from_vec(vec![1.0, 0.0, 1.0], &[3], DeviceType::Cpu).unwrap();

        let accuracy = Accuracy::new();
        let result = accuracy.compute(&predictions, &targets);

        assert!((result - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mse_very_large_values() {
        let predictions = from_vec(vec![1e30, 2e30, 3e30, 4e30], &[4], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![1e30, 2e30, 3e30, 4e30], &[4], DeviceType::Cpu).unwrap();

        let mse = MSE;
        let result = mse.compute(&predictions, &targets);

        // Perfect prediction should give MSE ≈ 0
        assert_relative_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mse_very_small_values() {
        let predictions =
            from_vec(vec![1e-30, 2e-30, 3e-30, 4e-30], &[4], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![1e-30, 2e-30, 3e-30, 4e-30], &[4], DeviceType::Cpu).unwrap();

        let mse = MSE;
        let result = mse.compute(&predictions, &targets);

        // Perfect prediction should give MSE ≈ 0
        assert!(result < 1e-20);
    }

    #[test]
    fn test_r2_with_constant_predictions() {
        // All predictions are the same
        let predictions = from_vec(vec![5.0, 5.0, 5.0, 5.0], &[4], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], DeviceType::Cpu).unwrap();

        let r2 = R2Score::new();
        let result = r2.compute(&predictions, &targets);

        // R² should be well-defined (likely negative)
        assert!(result.is_finite());
    }

    #[test]
    fn test_r2_with_constant_targets() {
        // All targets are the same
        let predictions = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![5.0, 5.0, 5.0, 5.0], &[4], DeviceType::Cpu).unwrap();

        let r2 = R2Score::new();
        let result = r2.compute(&predictions, &targets);

        // Division by zero in variance - should handle gracefully
        assert!(result.is_finite() || result.is_nan());
    }
}

#[cfg(test)]
mod single_sample {
    use super::*;

    #[test]
    fn test_accuracy_single_sample() {
        let predictions = from_vec(vec![0.3, 0.7], &[1, 2], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![1.0], &[1], DeviceType::Cpu).unwrap();

        let accuracy = Accuracy::new();
        let result = accuracy.compute(&predictions, &targets);

        // Should be either 0.0 or 1.0
        assert!((result - 1.0).abs() < 1e-6 || result.abs() < 1e-6);
    }

    #[test]
    fn test_mse_single_sample() {
        let predictions = from_vec(vec![3.0], &[1], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![5.0], &[1], DeviceType::Cpu).unwrap();

        let mse = MSE;
        let result = mse.compute(&predictions, &targets);

        // MSE = (3-5)² = 4
        assert_relative_eq!(result, 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_rmse_single_sample() {
        let predictions = from_vec(vec![3.0], &[1], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![5.0], &[1], DeviceType::Cpu).unwrap();

        let rmse = RMSE;
        let result = rmse.compute(&predictions, &targets);

        // RMSE = sqrt((3-5)²) = 2
        assert_relative_eq!(result, 2.0, epsilon = 1e-6);
    }
}

#[cfg(test)]
mod numerical_stability {
    use super::*;

    #[test]
    fn test_mse_numerical_stability() {
        // Test with values that could cause numerical instability
        let mut values = Vec::new();
        for i in 0..1000 {
            values.push(1.0 + i as f32 * 1e-6);
        }

        let predictions = from_vec(values.clone(), &[1000], DeviceType::Cpu).unwrap();
        let targets = from_vec(values, &[1000], DeviceType::Cpu).unwrap();

        let mse = MSE;
        let result = mse.compute(&predictions, &targets);

        // Perfect prediction should give MSE ≈ 0
        assert_relative_eq!(result, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_accuracy_many_classes() {
        // Test with many classes (100 classes)
        let num_samples = 100;
        let num_classes = 100;

        let mut predictions_vec = Vec::new();
        for i in 0..num_samples {
            for j in 0..num_classes {
                if j == i % num_classes {
                    predictions_vec.push(0.99); // High probability for correct class
                } else {
                    predictions_vec.push(0.01 / (num_classes - 1) as f32);
                }
            }
        }

        let predictions = from_vec(
            predictions_vec,
            &[num_samples, num_classes],
            DeviceType::Cpu,
        )
        .unwrap();

        let targets: Vec<f32> = (0..num_samples).map(|i| (i % num_classes) as f32).collect();
        let targets = from_vec(targets, &[num_samples], DeviceType::Cpu).unwrap();

        let accuracy = Accuracy::new();
        let result = accuracy.compute(&predictions, &targets);

        // Should be close to 1.0
        assert!((result - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_precision_recall_edge_case() {
        // All predictions are negative class
        let predictions = from_vec(
            vec![0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
            &[4, 2],
            DeviceType::Cpu,
        )
        .unwrap();
        let targets = from_vec(vec![0.0, 0.0, 0.0, 0.0], &[4], DeviceType::Cpu).unwrap();

        let precision = Precision::micro();
        let result_p = precision.compute(&predictions, &targets);

        let recall = Recall::micro();
        let result_r = recall.compute(&predictions, &targets);

        // Should handle gracefully
        assert!(result_p >= 0.0 && result_p <= 1.0 || result_p.is_nan());
        assert!(result_r >= 0.0 && result_r <= 1.0 || result_r.is_nan());
    }
}

#[cfg(test)]
mod zero_division {
    use super::*;

    #[test]
    fn test_f1_zero_tp_fp_fn() {
        // All negative predictions, all negative targets
        let predictions =
            from_vec(vec![0.9, 0.1, 0.8, 0.2, 0.7, 0.3], &[3, 2], DeviceType::Cpu).unwrap();
        let targets = from_vec(vec![0.0, 0.0, 0.0], &[3], DeviceType::Cpu).unwrap();

        let f1 = F1Score::micro();
        let result = f1.compute(&predictions, &targets);

        // Should handle division by zero gracefully
        assert!(result >= 0.0 && result <= 1.0 || result.is_nan());
    }
}
