//! Reference validation tests comparing torsh-metrics with scikit-learn expected values
//!
//! These tests validate that our implementations produce the same results as
//! scikit-learn on known datasets and edge cases.

use torsh_metrics::{
    SklearnAccuracy, SklearnF1Score, SklearnMeanAbsoluteError, SklearnMeanSquaredError,
    SklearnPrecision, SklearnR2Score, SklearnRecall,
};

const TOLERANCE: f64 = 1e-6;

mod classification_reference {
    use super::*;

    #[test]
    fn test_accuracy_perfect_prediction() {
        // sklearn.metrics.accuracy_score([0, 1, 2, 3], [0, 1, 2, 3]) = 1.0
        let y_true = vec![0, 1, 2, 3];
        let y_pred = vec![0, 1, 2, 3];
        let accuracy = SklearnAccuracy::new().compute(&y_true, &y_pred);
        assert!((accuracy - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_accuracy_zero_prediction() {
        // sklearn.metrics.accuracy_score([0, 1, 2, 3], [1, 2, 3, 0]) = 0.0
        let y_true = vec![0, 1, 2, 3];
        let y_pred = vec![1, 2, 3, 0];
        let accuracy = SklearnAccuracy::new().compute(&y_true, &y_pred);
        assert!((accuracy - 0.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_accuracy_half_correct() {
        // sklearn.metrics.accuracy_score([0, 0, 1, 1], [0, 1, 0, 1]) = 0.5
        let y_true = vec![0, 0, 1, 1];
        let y_pred = vec![0, 1, 0, 1];
        let accuracy = SklearnAccuracy::new().compute(&y_true, &y_pred);
        assert!((accuracy - 0.5).abs() < TOLERANCE);
    }

    #[test]
    fn test_precision_binary() {
        // sklearn.metrics.precision_score([0, 1, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0], average='binary') = 1.0
        let y_true = vec![0, 1, 1, 0, 1, 1];
        let y_pred = vec![0, 1, 0, 0, 1, 0];
        let precision = SklearnPrecision::new().compute(&y_true, &y_pred);
        assert!((precision - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_recall_binary() {
        // sklearn.metrics.recall_score([0, 1, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0], average='binary') = 0.5
        let y_true = vec![0, 1, 1, 0, 1, 1];
        let y_pred = vec![0, 1, 0, 0, 1, 0];
        let recall = SklearnRecall::new().compute(&y_true, &y_pred);
        assert!((recall - 0.5).abs() < TOLERANCE);
    }

    #[test]
    fn test_f1_binary() {
        // sklearn.metrics.f1_score([0, 1, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0], average='binary')
        // precision = 1.0, recall = 0.5, f1 = 2 * 1.0 * 0.5 / (1.0 + 0.5) = 0.666...
        let y_true = vec![0, 1, 1, 0, 1, 1];
        let y_pred = vec![0, 1, 0, 0, 1, 0];
        let f1 = SklearnF1Score::new().compute(&y_true, &y_pred);
        assert!((f1 - 0.6666666666666666).abs() < TOLERANCE);
    }

    #[test]
    fn test_precision_macro() {
        // sklearn.metrics.precision_score([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1], average='macro')
        // Class 0: TP=2, FP=1, precision=2/3
        // Class 1: TP=0, FP=2, precision=0
        // Class 2: TP=0, FP=0, precision=0 (with zero_division=0)
        // Macro avg = (2/3 + 0 + 0) / 3 = 0.2222...
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 2, 1, 0, 0, 1];
        let precision = SklearnPrecision::new()
            .with_average("macro")
            .compute(&y_true, &y_pred);
        assert!((precision - 0.2222222222222222).abs() < TOLERANCE);
    }

    #[test]
    fn test_recall_macro() {
        // sklearn.metrics.recall_score([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1], average='macro')
        // Class 0: TP=2, FN=0, recall=1.0
        // Class 1: TP=0, FN=2, recall=0.0
        // Class 2: TP=0, FN=2, recall=0.0
        // Macro avg = (1.0 + 0.0 + 0.0) / 3 = 0.3333...
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 2, 1, 0, 0, 1];
        let recall = SklearnRecall::new()
            .with_average("macro")
            .compute(&y_true, &y_pred);
        assert!((recall - 0.3333333333333333).abs() < TOLERANCE);
    }

    #[test]
    fn test_f1_macro() {
        // sklearn.metrics.f1_score([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1], average='macro')
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 2, 1, 0, 0, 1];
        let f1 = SklearnF1Score::new()
            .with_average("macro")
            .compute(&y_true, &y_pred);

        // Class 0: precision=2/3, recall=1.0, f1=0.8
        // Class 1: precision=0, recall=0, f1=0
        // Class 2: precision=0, recall=0, f1=0
        // Macro avg = (0.8 + 0 + 0) / 3 = 0.2666...
        assert!((f1 - 0.26666666666666666).abs() < TOLERANCE);
    }

    #[test]
    fn test_precision_weighted() {
        // sklearn.metrics.precision_score([0, 1, 2, 0, 1, 2], [0, 2, 1, 0, 0, 1], average='weighted')
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 2, 1, 0, 0, 1];
        let precision = SklearnPrecision::new()
            .with_average("weighted")
            .compute(&y_true, &y_pred);

        // Class 0: precision=2/3, support=2, weight=2/6
        // Class 1: precision=0, support=2, weight=2/6
        // Class 2: precision=0, support=2, weight=2/6
        // Weighted avg = (2/3 * 2/6 + 0 * 2/6 + 0 * 2/6) = 0.2222...
        assert!((precision - 0.2222222222222222).abs() < TOLERANCE);
    }
}

mod regression_reference {
    use super::*;

    #[test]
    fn test_mse_perfect_prediction() {
        // sklearn.metrics.mean_squared_error([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) = 0.0
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.0, 2.0, 3.0];
        let mse = SklearnMeanSquaredError::new().compute(&y_true, &y_pred);
        assert!((mse - 0.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_mse_simple() {
        // sklearn.metrics.mean_squared_error([3.0, -0.5, 2.0, 7.0], [2.5, 0.0, 2.0, 8.0])
        // = ((3-2.5)^2 + (-0.5-0)^2 + (2-2)^2 + (7-8)^2) / 4
        // = (0.25 + 0.25 + 0 + 1) / 4 = 0.375
        let y_true = vec![3.0, -0.5, 2.0, 7.0];
        let y_pred = vec![2.5, 0.0, 2.0, 8.0];
        let mse = SklearnMeanSquaredError::new().compute(&y_true, &y_pred);
        assert!((mse - 0.375).abs() < TOLERANCE);
    }

    #[test]
    fn test_rmse() {
        // sklearn.metrics.mean_squared_error([3.0, -0.5, 2.0, 7.0], [2.5, 0.0, 2.0, 8.0], squared=False)
        // = sqrt(0.375) = 0.6123724356957945
        let y_true = vec![3.0, -0.5, 2.0, 7.0];
        let y_pred = vec![2.5, 0.0, 2.0, 8.0];
        let rmse = SklearnMeanSquaredError::new()
            .with_squared(false)
            .compute(&y_true, &y_pred);
        assert!((rmse - 0.6123724356957945).abs() < TOLERANCE);
    }

    #[test]
    fn test_mae_simple() {
        // sklearn.metrics.mean_absolute_error([3.0, -0.5, 2.0, 7.0], [2.5, 0.0, 2.0, 8.0])
        // = (|3-2.5| + |-0.5-0| + |2-2| + |7-8|) / 4
        // = (0.5 + 0.5 + 0 + 1) / 4 = 0.5
        let y_true = vec![3.0, -0.5, 2.0, 7.0];
        let y_pred = vec![2.5, 0.0, 2.0, 8.0];
        let mae = SklearnMeanAbsoluteError::new().compute(&y_true, &y_pred);
        assert!((mae - 0.5).abs() < TOLERANCE);
    }

    #[test]
    fn test_r2_perfect() {
        // sklearn.metrics.r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) = 1.0
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.0, 2.0, 3.0];
        let r2 = SklearnR2Score::new().compute(&y_true, &y_pred);
        assert!((r2 - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_r2_simple() {
        // sklearn.metrics.r2_score([3.0, -0.5, 2.0, 7.0], [2.5, 0.0, 2.0, 8.0])
        let y_true = vec![3.0, -0.5, 2.0, 7.0];
        let y_pred = vec![2.5, 0.0, 2.0, 8.0];
        let r2 = SklearnR2Score::new().compute(&y_true, &y_pred);

        // Mean of y_true = (3.0 - 0.5 + 2.0 + 7.0) / 4 = 2.875
        // SS_tot = sum((y_true - mean)^2) = 29.1875
        // SS_res = sum((y_true - y_pred)^2) = 1.5
        // R2 = 1 - SS_res/SS_tot = 1 - 1.5/29.1875 ≈ 0.9485897
        assert!((r2 - 0.9486081370449679).abs() < TOLERANCE);
    }

    #[test]
    fn test_r2_constant_prediction() {
        // When predictions are constant (equal to mean of y_true), R^2 = 0
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let mean = 2.5;
        let y_pred = vec![mean, mean, mean, mean];
        let r2 = SklearnR2Score::new().compute(&y_true, &y_pred);
        assert!((r2 - 0.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_r2_worse_than_mean() {
        // When predictions are worse than mean, R^2 < 0
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![4.0, 3.0, 2.0, 1.0]; // Reversed
        let r2 = SklearnR2Score::new().compute(&y_true, &y_pred);
        assert!(r2 < 0.0);
    }
}

mod edge_cases {
    use super::*;

    #[test]
    fn test_single_sample_accuracy() {
        let y_true = vec![1];
        let y_pred = vec![1];
        let accuracy = SklearnAccuracy::new().compute(&y_true, &y_pred);
        assert!((accuracy - 1.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_binary_all_zeros() {
        // When all predictions and targets are 0
        let y_true = vec![0, 0, 0, 0];
        let y_pred = vec![0, 0, 0, 0];
        let precision = SklearnPrecision::new()
            .with_pos_label(1)
            .with_zero_division(0.0)
            .compute(&y_true, &y_pred);

        // No positive predictions, so precision is 0 (by zero_division policy)
        assert!((precision - 0.0).abs() < TOLERANCE);
    }

    #[test]
    fn test_zero_variance_r2() {
        // When y_true has zero variance (all same value), R^2 is undefined
        let y_true = vec![2.0, 2.0, 2.0, 2.0];
        let y_pred = vec![2.0, 2.0, 2.0, 2.0];
        let r2 = SklearnR2Score::new().compute(&y_true, &y_pred);
        // Our implementation returns 0.0 for this edge case (sklearn behavior)
        assert!((r2 - 0.0).abs() < TOLERANCE);
    }
}

mod multiclass_reference {
    use super::*;

    #[test]
    fn test_multiclass_accuracy() {
        // 5-class classification
        let y_true = vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
        let y_pred = vec![0, 1, 2, 3, 4, 1, 2, 3, 4, 0]; // 50% accuracy
        let accuracy = SklearnAccuracy::new().compute(&y_true, &y_pred);
        assert!((accuracy - 0.5).abs() < TOLERANCE);
    }

    #[test]
    fn test_multiclass_micro_precision() {
        // Micro averaging: same as accuracy for multi-class
        let y_true = vec![0, 1, 2, 0, 1, 2];
        let y_pred = vec![0, 2, 1, 0, 0, 1];
        let precision = SklearnPrecision::new()
            .with_average("micro")
            .compute(&y_true, &y_pred);
        let accuracy = SklearnAccuracy::new().compute(&y_true, &y_pred);
        assert!((precision - accuracy).abs() < TOLERANCE);
    }
}
