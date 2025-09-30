//! Advanced metrics demonstration
//!
//! This example showcases the comprehensive evaluation capabilities of torsh-metrics,
//! including classification, regression, fairness, uncertainty quantification, and
//! deep learning specific metrics.

use torsh_core::device::DeviceType;
use torsh_metrics::{
    // Core metrics
    classification::{Accuracy, AverageMethod, F1Score, Precision, Recall},
    // Advanced metrics
    deep_learning::{InceptionScore, Perplexity, SemanticSimilarity},
    fairness::{Calibration, DemographicParity},
    regression::{HuberLoss, R2Score, MAE, MSE},

    uncertainty::{BrierScore, EntropyMeasures, ExpectedCalibrationError},

    // Utilities
    utils::{bootstrap_ci, compute_class_weights, probs_to_preds},
    Metric,
    MetricCollection,
};
use torsh_tensor::{creation::from_vec, Tensor};

/// Create example tensor from 2D array
fn tensor_2d(data: &[&[f32]]) -> Tensor {
    let flat: Vec<f32> = data.iter().flat_map(|row| row.iter()).copied().collect();
    let rows = data.len();
    let cols = data[0].len();
    from_vec(flat, &[rows, cols], DeviceType::Cpu).unwrap()
}

/// Create example tensor from 1D array
fn tensor_1d(data: &[f32]) -> Tensor {
    from_vec(data.to_vec(), &[data.len()], DeviceType::Cpu).unwrap()
}

fn main() {
    println!("🚀 ToRSh Metrics - Advanced Evaluation Demo");
    println!("{}", "=".repeat(50));

    // ============================================================================
    // 1. CLASSIFICATION METRICS DEMONSTRATION
    // ============================================================================
    println!("\n📊 1. Classification Metrics");
    println!("{}", "-".repeat(30));

    // Create synthetic classification data
    let classification_predictions = tensor_2d(&[
        &[0.1, 0.9], // Predicted class 1, True class 1 ✓
        &[0.8, 0.2], // Predicted class 0, True class 0 ✓
        &[0.3, 0.7], // Predicted class 1, True class 1 ✓
        &[0.6, 0.4], // Predicted class 0, True class 1 ✗
        &[0.9, 0.1], // Predicted class 0, True class 0 ✓
    ]);
    let classification_targets = tensor_1d(&[1.0, 0.0, 1.0, 1.0, 0.0]);

    // Single metrics
    let accuracy = Accuracy::new();
    let precision = Precision::micro();
    let recall = Recall::micro();
    let f1 = F1Score::micro();

    println!("Individual Classification Metrics:");
    println!(
        "  Accuracy: {:.4}",
        accuracy.compute(&classification_predictions, &classification_targets)
    );
    println!(
        "  Precision: {:.4}",
        precision.compute(&classification_predictions, &classification_targets)
    );
    println!(
        "  Recall: {:.4}",
        recall.compute(&classification_predictions, &classification_targets)
    );
    println!(
        "  F1 Score: {:.4}",
        f1.compute(&classification_predictions, &classification_targets)
    );

    // Metric collection for batch evaluation
    let mut classification_metrics = MetricCollection::new()
        .add(Accuracy::new())
        .add(Precision::micro())
        .add(Recall::micro())
        .add(F1Score::micro())
        .add(Accuracy::top_k(2));

    let results =
        classification_metrics.compute(&classification_predictions, &classification_targets);
    println!("\nMetric Collection Results:");
    for (name, value) in results {
        println!("  {}: {:.4}", name, value);
    }

    // ============================================================================
    // 2. REGRESSION METRICS DEMONSTRATION
    // ============================================================================
    println!("\n📈 2. Regression Metrics");
    println!("{}", "-".repeat(30));

    // Create synthetic regression data
    let regression_predictions = tensor_1d(&[2.1, 3.9, 5.8, 8.2, 10.1]);
    let regression_targets = tensor_1d(&[2.0, 4.0, 6.0, 8.0, 10.0]);

    let mse = MSE;
    let mae = MAE;
    let r2 = R2Score::new();
    let huber = HuberLoss::new(1.0);

    println!("Regression Metrics:");
    println!(
        "  MSE: {:.6}",
        mse.compute(&regression_predictions, &regression_targets)
    );
    println!(
        "  MAE: {:.6}",
        mae.compute(&regression_predictions, &regression_targets)
    );
    println!(
        "  R² Score: {:.6}",
        r2.compute(&regression_predictions, &regression_targets)
    );
    println!(
        "  Huber Loss: {:.6}",
        huber.compute(&regression_predictions, &regression_targets)
    );

    // ============================================================================
    // 3. DEEP LEARNING SPECIFIC METRICS
    // ============================================================================
    println!("\n🧠 3. Deep Learning Metrics");
    println!("{}", "-".repeat(30));

    // Language model evaluation (Perplexity)
    let language_logits = tensor_2d(&[
        &[2.0, 1.0, 0.5], // Target class 0
        &[0.5, 3.0, 1.2], // Target class 1
        &[1.1, 0.8, 2.5], // Target class 2
        &[2.2, 0.9, 1.1], // Target class 0
    ]);
    let language_targets = tensor_1d(&[0.0, 1.0, 2.0, 0.0]);

    let perplexity = Perplexity::new();
    println!("Language Model Metrics:");
    println!(
        "  Perplexity: {:.4}",
        perplexity.compute(&language_logits, &language_targets)
    );

    // Generative model evaluation (Inception Score)
    let generated_predictions = tensor_2d(&[
        &[0.8, 0.15, 0.05], // High confidence
        &[0.4, 0.35, 0.25], // Lower confidence
        &[0.9, 0.05, 0.05], // Very high confidence
        &[0.6, 0.3, 0.1],   // Medium confidence
    ]);

    let inception_score = InceptionScore::new(1);
    println!(
        "  Inception Score: {:.4}",
        inception_score.compute(&generated_predictions, &language_targets)
    );

    // Semantic similarity
    let embeddings1 = tensor_1d(&[1.0, 2.0, 3.0, 4.0]);
    let embeddings2 = tensor_1d(&[1.1, 2.1, 2.9, 4.1]);

    let semantic_sim = SemanticSimilarity;
    println!(
        "  Semantic Similarity: {:.4}",
        semantic_sim.compute(&embeddings1, &embeddings2)
    );

    // ============================================================================
    // 4. FAIRNESS AND BIAS METRICS
    // ============================================================================
    println!("\n⚖️  4. Fairness Metrics");
    println!("{}", "-".repeat(30));

    // Create synthetic fairness evaluation data
    let fair_predictions = tensor_1d(&[0.8, 0.3, 0.9, 0.2, 0.7, 0.4]); // Predictions
    let sensitive_attrs = tensor_1d(&[0.0, 0.0, 1.0, 1.0, 0.0, 1.0]); // Group membership
    let fair_labels = tensor_1d(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]); // True labels

    let demographic_parity = DemographicParity::new(0.5);
    println!("Fairness Metrics:");
    println!(
        "  Demographic Parity Diff: {:.4}",
        demographic_parity.compute_difference(&fair_predictions, &sensitive_attrs)
    );
    println!(
        "  Demographic Parity Ratio: {:.4}",
        demographic_parity.compute_ratio(&fair_predictions, &sensitive_attrs)
    );

    // Calibration analysis
    let calibration = Calibration::new(5);
    println!(
        "  Expected Calibration Error: {:.4}",
        calibration.compute_ece(&fair_predictions, &fair_labels)
    );
    println!(
        "  Maximum Calibration Error: {:.4}",
        calibration.compute_mce(&fair_predictions, &fair_labels)
    );

    // ============================================================================
    // 5. UNCERTAINTY QUANTIFICATION
    // ============================================================================
    println!("\n🎯 5. Uncertainty Quantification");
    println!("{}", "-".repeat(30));

    // Probabilistic predictions for uncertainty analysis
    let prob_predictions = tensor_1d(&[0.9, 0.7, 0.6, 0.8, 0.55]);
    let binary_labels = tensor_1d(&[1.0, 1.0, 0.0, 1.0, 1.0]);

    let ece = ExpectedCalibrationError::new(3);
    let brier = BrierScore;
    let entropy = EntropyMeasures;

    // Multi-class predictions for entropy calculation
    let multiclass_probs = tensor_2d(&[
        &[0.7, 0.2, 0.1],
        &[0.3, 0.6, 0.1],
        &[0.1, 0.1, 0.8],
        &[0.5, 0.3, 0.2],
    ]);

    println!("Uncertainty Metrics:");
    println!(
        "  Expected Calibration Error: {:.6}",
        ece.compute(&prob_predictions, &binary_labels)
    );
    println!(
        "  Brier Score: {:.6}",
        brier.compute(&prob_predictions, &binary_labels)
    );
    println!(
        "  Predictive Entropy: {:.6}",
        entropy.compute(&multiclass_probs, &binary_labels)
    );

    // ============================================================================
    // 6. UTILITY FUNCTIONS DEMONSTRATION
    // ============================================================================
    println!("\n🛠️  6. Utility Functions");
    println!("{}", "-".repeat(30));

    // Convert probabilities to predictions
    let prob_tensor = tensor_2d(&[&[0.3, 0.7], &[0.8, 0.2], &[0.45, 0.55]]);

    match probs_to_preds(&prob_tensor, 0.5) {
        Ok(preds) => {
            println!(
                "Converted Predictions: {:?}",
                preds.to_vec().unwrap_or_default()
            );
        }
        Err(e) => println!("Error converting predictions: {:?}", e),
    }

    // Bootstrap confidence intervals
    let sample_scores = vec![0.85, 0.87, 0.82, 0.89, 0.84, 0.86, 0.88, 0.83, 0.85, 0.87];
    let (ci_lower, ci_upper) = bootstrap_ci(&sample_scores, 0.95, 1000, Some(42));
    println!("Bootstrap 95% CI: [{:.4}, {:.4}]", ci_lower, ci_upper);

    // Class weight computation for imbalanced data
    let imbalanced_labels = tensor_1d(&[0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    match compute_class_weights(&imbalanced_labels, 2) {
        Ok(weights) => {
            let weights_vec = weights.to_vec().unwrap_or_default();
            println!(
                "Class Weights - Class 0: {:.4}, Class 1: {:.4}",
                weights_vec.get(0).unwrap_or(&0.0),
                weights_vec.get(1).unwrap_or(&0.0)
            );
        }
        Err(e) => println!("Error computing class weights: {:?}", e),
    }

    // ============================================================================
    // 7. COMPREHENSIVE MODEL EVALUATION
    // ============================================================================
    println!("\n🔍 7. Comprehensive Model Evaluation");
    println!("{}", "-".repeat(30));

    // Create a realistic evaluation scenario
    let model_predictions = tensor_2d(&[
        &[0.05, 0.9, 0.05], // Class 1, High confidence
        &[0.8, 0.15, 0.05], // Class 0, High confidence
        &[0.1, 0.2, 0.7],   // Class 2, High confidence
        &[0.6, 0.35, 0.05], // Class 0, Medium confidence
        &[0.3, 0.4, 0.3],   // Class 1, Low confidence
        &[0.1, 0.1, 0.8],   // Class 2, High confidence
    ]);
    let true_labels = tensor_1d(&[1.0, 0.0, 2.0, 0.0, 1.0, 2.0]);

    // Comprehensive metric collection
    let mut comprehensive_metrics = MetricCollection::new()
        .add(Accuracy::new())
        .add(Accuracy::top_k(2))
        .add(Precision::macro_averaged())
        .add(Recall::macro_averaged())
        .add(F1Score::macro_averaged());

    let comprehensive_results = comprehensive_metrics.compute(&model_predictions, &true_labels);

    println!("Comprehensive Model Evaluation:");
    println!("{}", comprehensive_metrics.format_results());

    // Additional analysis
    println!("\nDetailed Analysis:");
    for (name, value) in comprehensive_results {
        let performance_level = match value {
            v if v >= 0.9 => "Excellent",
            v if v >= 0.8 => "Good",
            v if v >= 0.7 => "Fair",
            _ => "Needs Improvement",
        };
        println!("  {} = {:.4} ({})", name, value, performance_level);
    }

    println!();
    println!("{}", "=".repeat(50));
    println!("✅ Demo completed! ToRSh Metrics provides comprehensive evaluation capabilities");
    println!("   for classification, regression, fairness, uncertainty, and deep learning.");
    println!("{}", "=".repeat(50));
}
