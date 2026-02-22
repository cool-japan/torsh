//! Simple demonstration of advanced ML metrics and experiment tracking

use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use torsh_metrics::{
    // Reporting
    ComparisonReport,
    // Advanced ML metrics
    ContinualLearningMetrics,
    DomainAdaptationMetrics,
    FewShotMetrics,
    // Experiment tracking
    MLflowClient,
    MetaLearningMetrics,
    MetricReport,
    // Scikit-learn compatibility
    SklearnAccuracy,
    SklearnF1Score,
    SklearnMeanSquaredError,
    WandbClient,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh Metrics Advanced Features Demo ===\n");

    demo_meta_learning()?;
    demo_few_shot_learning()?;
    demo_domain_adaptation()?;
    demo_continual_learning()?;
    demo_experiment_tracking()?;
    demo_sklearn_compat()?;
    demo_reporting()?;

    println!("\n=== Demo Complete ===");
    Ok(())
}

fn demo_meta_learning() -> Result<(), Box<dyn std::error::Error>> {
    println!("1. Meta-Learning Metrics");
    println!("-----------------------");

    let task_train_scores = vec![0.92, 0.88, 0.90, 0.85, 0.87];
    let task_test_scores = vec![0.87, 0.82, 0.85, 0.80, 0.83];
    let adaptation_steps = vec![10, 15, 12, 18, 14];
    let baseline_scores = vec![0.60, 0.58, 0.62, 0.55, 0.59];

    let metrics = MetaLearningMetrics::compute(
        &task_train_scores,
        &task_test_scores,
        &adaptation_steps,
        Some(&baseline_scores),
    );

    println!(
        "Task Adaptation Speed: {:.2} steps",
        metrics.task_adaptation_speed
    );
    println!(
        "Few-Shot Generalization Gap: {:.4}",
        metrics.few_shot_generalization_gap
    );
    println!(
        "Average Task Performance: {:.4}\n",
        metrics.average_task_performance
    );

    Ok(())
}

fn demo_few_shot_learning() -> Result<(), Box<dyn std::error::Error>> {
    println!("2. Few-Shot Learning Metrics");
    println!("---------------------------");

    let episode_accuracies = vec![0.85, 0.87, 0.84, 0.88, 0.86];

    let metrics = FewShotMetrics::compute(&episode_accuracies, None, None, None, None);

    println!(
        "N-Way K-Shot Accuracy: {:.4}",
        metrics.n_way_k_shot_accuracy
    );
    println!(
        "Mean Episode Performance: {:.4}\n",
        metrics.mean_episode_performance
    );

    Ok(())
}

fn demo_domain_adaptation() -> Result<(), Box<dyn std::error::Error>> {
    println!("3. Domain Adaptation Metrics");
    println!("---------------------------");

    let source_features = Array2::from_shape_fn((100, 50), |(i, j)| ((i + j) as f64 * 0.1).sin());
    let target_features =
        Array2::from_shape_fn((100, 50), |(i, j)| ((i + j) as f64 * 0.1 + 1.0).cos());

    let metrics = DomainAdaptationMetrics::compute(
        &source_features,
        &target_features,
        0.92,
        0.78,
        None,
        None,
    );

    println!("MMD Distance: {:.4}", metrics.mmd_distance);
    println!("Adaptation Gap: {:.4}\n", metrics.adaptation_gap);

    Ok(())
}

fn demo_continual_learning() -> Result<(), Box<dyn std::error::Error>> {
    println!("4. Continual Learning Metrics");
    println!("----------------------------");

    let task_accuracies = Array2::from_shape_vec(
        (3, 3),
        vec![0.92, 0.88, 0.85, 0.00, 0.90, 0.87, 0.00, 0.00, 0.91],
    )?;

    let metrics = ContinualLearningMetrics::compute(&task_accuracies);

    println!("Backward Transfer: {:.4}", metrics.backward_transfer);
    println!("Forward Transfer: {:.4}", metrics.forward_transfer);
    println!("Forgetting Measure: {:.4}\n", metrics.forgetting_measure);

    Ok(())
}

fn demo_experiment_tracking() -> Result<(), Box<dyn std::error::Error>> {
    println!("5. Experiment Tracking");
    println!("---------------------");

    // W&B tracking
    println!("W&B Tracking:");
    let mut wandb = WandbClient::new("demo_project");
    wandb.init(Some("demo_run".to_string()), None, None, None)?;

    let mut metrics = HashMap::new();
    metrics.insert("train_acc".to_string(), 0.95);
    metrics.insert("val_acc".to_string(), 0.92);
    wandb.log(metrics, Some(0))?;
    println!("  - Logged metrics to W&B");
    wandb.finish()?;

    // MLflow tracking
    println!("MLflow Tracking:");
    let mut mlflow = MLflowClient::new("http://localhost:5000", "demo_experiment");
    mlflow.start_run(Some("demo_mlflow_run".to_string()))?;
    mlflow.log_param("learning_rate", "0.001")?;
    mlflow.log_metric("accuracy", 0.95, Some(0), None)?;
    println!("  - Logged metrics to MLflow");
    mlflow.end_run()?;

    println!();
    Ok(())
}

fn demo_sklearn_compat() -> Result<(), Box<dyn std::error::Error>> {
    println!("6. Scikit-learn Compatibility");
    println!("----------------------------");

    let y_true = vec![0, 1, 2, 0, 1, 2];
    let y_pred = vec![0, 2, 1, 0, 0, 1];

    let accuracy = SklearnAccuracy::new().compute(&y_true, &y_pred);
    println!("Accuracy: {:.4}", accuracy);

    let f1_macro = SklearnF1Score::new()
        .with_average("macro")
        .compute(&y_true, &y_pred);
    println!("F1 Score (macro): {:.4}", f1_macro);

    let y_true_reg = vec![3.0, -0.5, 2.0, 7.0];
    let y_pred_reg = vec![2.5, 0.0, 2.0, 8.0];

    let mse = SklearnMeanSquaredError::new().compute(&y_true_reg, &y_pred_reg);
    println!("MSE: {:.4}\n", mse);

    Ok(())
}

fn demo_reporting() -> Result<(), Box<dyn std::error::Error>> {
    println!("7. Reporting");
    println!("------------");

    // Create metric reports
    let mut baseline_report = MetricReport::new("Baseline Model".to_string());
    baseline_report.add_metric("accuracy".to_string(), 0.85);
    baseline_report.add_metric("f1_score".to_string(), 0.825);

    let mut new_report = MetricReport::new("New Model".to_string());
    new_report.add_metric("accuracy".to_string(), 0.95);
    new_report.add_metric("f1_score".to_string(), 0.925);

    println!("Baseline Report:\n{}", baseline_report.to_markdown());

    // Create a comparison report
    let mut comparison = ComparisonReport::new("Model Comparison".to_string());
    comparison.add_experiment(baseline_report);
    comparison.add_experiment(new_report);

    println!("Comparison:\n{}", comparison.to_markdown());

    Ok(())
}
