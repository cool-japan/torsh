# torsh-metrics

Comprehensive evaluation metrics for ToRSh - powered by SciRS2.

## Overview

This crate provides extensive evaluation metrics for machine learning models with a scikit-learn compatible API. It leverages `scirs2-metrics` for high-performance implementations while maintaining full integration with ToRSh's tensor operations.

## Features

- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC
- **Regression Metrics**: MSE, RMSE, MAE, R², MAPE, Huber loss
- **Ranking Metrics**: NDCG, MAP, MRR, Precision@K, Recall@K
- **Clustering Metrics**: Silhouette, Davies-Bouldin, Calinski-Harabasz, NMI, ARI
- **Deep Learning Metrics**: Perplexity, BLEU, ROUGE, accuracy@k
- **Fairness Metrics**: Demographic parity, Equal opportunity, Disparate impact
- **Uncertainty Metrics**: Expected calibration error, Brier score
- **Statistical Tests**: t-test, Mann-Whitney U, Wilcoxon, permutation tests
- **Custom Metrics**: Define your own metrics with automatic batching
- **GPU Acceleration**: Compute metrics on GPU for large datasets

## Usage

### Classification Metrics

#### Binary Classification

```rust
use torsh_metrics::classification::*;
use torsh_tensor::prelude::*;

// Predictions and ground truth
let y_true = tensor![0, 1, 1, 0, 1, 0, 1, 1];
let y_pred = tensor![0, 1, 1, 0, 0, 1, 1, 1];

// Accuracy
let acc = accuracy(&y_true, &y_pred)?;
println!("Accuracy: {:.4}", acc);

// Precision, Recall, F1-Score
let precision = precision_score(&y_true, &y_pred, None, None, None, None)?;
let recall = recall_score(&y_true, &y_pred, None, None, None, None)?;
let f1 = f1_score(&y_true, &y_pred, None, None, None, None)?;

println!("Precision: {:.4}, Recall: {:.4}, F1: {:.4}", precision, recall, f1);

// Confusion matrix
let cm = confusion_matrix(&y_true, &y_pred, None)?;
println!("Confusion Matrix:\n{:?}", cm);

// ROC-AUC (requires probability scores)
let y_scores = tensor![0.1, 0.9, 0.8, 0.2, 0.4, 0.7, 0.95, 0.85];
let roc_auc = roc_auc_score(&y_true, &y_scores, None, None, None)?;
println!("ROC-AUC: {:.4}", roc_auc);

// Precision-Recall curve
let (precision_curve, recall_curve, thresholds) = precision_recall_curve(&y_true, &y_scores, None, None)?;

// Average Precision (AP) / PR-AUC
let ap = average_precision_score(&y_true, &y_scores, None, None)?;
println!("Average Precision: {:.4}", ap);
```

#### Multi-class Classification

```rust
use torsh_metrics::classification::*;

let y_true = tensor![0, 1, 2, 0, 1, 2, 1, 2];
let y_pred = tensor![0, 2, 1, 0, 1, 1, 1, 2];

// Accuracy
let acc = accuracy(&y_true, &y_pred)?;

// Macro-averaged metrics (unweighted mean)
let f1_macro = f1_score(&y_true, &y_pred, None, Some("macro"), None, None)?;

// Micro-averaged metrics (global average)
let f1_micro = f1_score(&y_true, &y_pred, None, Some("micro"), None, None)?;

// Weighted-averaged metrics (weighted by support)
let f1_weighted = f1_score(&y_true, &y_pred, None, Some("weighted"), None, None)?;

// Per-class metrics
let f1_per_class = f1_score(&y_true, &y_pred, None, None, None, None)?;
println!("Per-class F1: {:?}", f1_per_class);

// Classification report
let report = classification_report(&y_true, &y_pred, None, None, None)?;
println!("{}", report);

// Multi-class ROC-AUC
let y_scores = randn(&[8, 3])?;  // Probability scores for 3 classes
let roc_auc_ovr = roc_auc_score(&y_true, &y_scores, Some("ovr"), Some("macro"), None)?;  // One-vs-Rest
let roc_auc_ovo = roc_auc_score(&y_true, &y_scores, Some("ovo"), Some("macro"), None)?;  // One-vs-One
```

#### Multi-label Classification

```rust
use torsh_metrics::classification::*;

// Each sample can have multiple labels
let y_true = tensor![[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]];
let y_pred = tensor![[1, 0, 0], [0, 1, 0], [1, 1, 1], [0, 0, 1]];

// Hamming loss (fraction of wrong labels)
let hamming = hamming_loss(&y_true, &y_pred)?;

// Jaccard similarity (IoU)
let jaccard = jaccard_score(&y_true, &y_pred, None)?;

// Subset accuracy (exact match ratio)
let subset_acc = accuracy(&y_true, &y_pred)?;

// Label-based metrics
let f1_samples = f1_score(&y_true, &y_pred, None, Some("samples"), None, None)?;
let f1_macro = f1_score(&y_true, &y_pred, None, Some("macro"), None, None)?;
```

### Regression Metrics

```rust
use torsh_metrics::regression::*;

let y_true = tensor![3.0, -0.5, 2.0, 7.0];
let y_pred = tensor![2.5, 0.0, 2.0, 8.0];

// Mean Squared Error (MSE)
let mse = mean_squared_error(&y_true, &y_pred, None, None)?;
println!("MSE: {:.4}", mse);

// Root Mean Squared Error (RMSE)
let rmse = mse.sqrt();
println!("RMSE: {:.4}", rmse);

// Mean Absolute Error (MAE)
let mae = mean_absolute_error(&y_true, &y_pred, None, None)?;
println!("MAE: {:.4}", mae);

// R² Score (coefficient of determination)
let r2 = r2_score(&y_true, &y_pred, None, None, None)?;
println!("R²: {:.4}", r2);

// Mean Absolute Percentage Error (MAPE)
let mape = mean_absolute_percentage_error(&y_true, &y_pred, None, None)?;
println!("MAPE: {:.4}%", mape * 100.0);

// Median Absolute Error
let med_ae = median_absolute_error(&y_true, &y_pred)?;

// Explained Variance Score
let explained_var = explained_variance_score(&y_true, &y_pred, None, None, None)?;

// Max Error
let max_err = max_error(&y_true, &y_pred)?;

// Huber loss (robust to outliers)
let huber = huber_loss(&y_true, &y_pred, 1.35)?;
```

### Ranking Metrics

```rust
use torsh_metrics::ranking::*;

// Search results: relevance scores
let relevance = tensor![[3, 2, 3, 0, 1, 2]];  // Graded relevance (0-3)
let scores = tensor![[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]];  // Model scores

// Normalized Discounted Cumulative Gain (NDCG)
let ndcg_5 = ndcg_score(&relevance, &scores, Some(5), None, None)?;
let ndcg_10 = ndcg_score(&relevance, &scores, Some(10), None, None)?;
println!("NDCG@5: {:.4}, NDCG@10: {:.4}", ndcg_5, ndcg_10);

// Mean Average Precision (MAP)
let binary_relevance = tensor![[1, 1, 1, 0, 0, 1]];
let map = mean_average_precision(&binary_relevance, &scores)?;
println!("MAP: {:.4}", map);

// Mean Reciprocal Rank (MRR)
let mrr = mean_reciprocal_rank(&binary_relevance, &scores)?;
println!("MRR: {:.4}", mrr);

// Precision@K and Recall@K
let precision_at_5 = precision_at_k(&binary_relevance, &scores, 5)?;
let recall_at_5 = recall_at_k(&binary_relevance, &scores, 5)?;

// Hit Rate (at least one relevant item in top-k)
let hit_rate = hit_rate_at_k(&binary_relevance, &scores, 10)?;
```

### Clustering Metrics

```rust
use torsh_metrics::clustering::*;

let features = randn(&[100, 10])?;
let labels = tensor![/* cluster assignments */];

// Internal metrics (no ground truth needed)

// Silhouette score (-1 to 1, higher is better)
let silhouette = silhouette_score(&features, &labels, "euclidean")?;
println!("Silhouette Score: {:.4}", silhouette);

// Davies-Bouldin index (lower is better)
let db = davies_bouldin_score(&features, &labels)?;
println!("Davies-Bouldin Index: {:.4}", db);

// Calinski-Harabasz score (higher is better)
let ch = calinski_harabasz_score(&features, &labels)?;
println!("Calinski-Harabasz Score: {:.4}", ch);

// External metrics (with ground truth)
let true_labels = tensor![/* ground truth */];

// Adjusted Rand Index (-1 to 1, 1 is perfect)
let ari = adjusted_rand_score(&true_labels, &labels)?;
println!("Adjusted Rand Index: {:.4}", ari);

// Normalized Mutual Information (0 to 1, 1 is perfect)
let nmi = normalized_mutual_info_score(&true_labels, &labels, Some("arithmetic"))?;

// Fowlkes-Mallows Index
let fmi = fowlkes_mallows_score(&true_labels, &labels)?;

// V-measure (harmonic mean of homogeneity and completeness)
let v_measure = v_measure_score(&true_labels, &labels, None)?;

// Homogeneity and Completeness
let (homogeneity, completeness, v) = homogeneity_completeness_v_measure(&true_labels, &labels, None)?;
```

### Deep Learning Metrics

#### Perplexity (Language Models)

```rust
use torsh_metrics::deep_learning::*;

let logits = randn(&[32, 100, 10000])?;  // [batch, seq_len, vocab_size]
let targets = randint(0, 10000, &[32, 100])?;

let perplexity = perplexity_score(&logits, &targets, None, None)?;
println!("Perplexity: {:.4}", perplexity);
```

#### Accuracy@k (Top-k Accuracy)

```rust
use torsh_metrics::deep_learning::*;

let logits = randn(&[128, 1000])?;  // ImageNet predictions
let targets = randint(0, 1000, &[128])?;

let top1_acc = top_k_accuracy(&logits, &targets, 1)?;
let top5_acc = top_k_accuracy(&logits, &targets, 5)?;
println!("Top-1: {:.4}, Top-5: {:.4}", top1_acc, top5_acc);
```

#### BLEU Score (Machine Translation)

```rust
use torsh_metrics::deep_learning::*;

let references = vec![
    "the cat is on the mat".to_string(),
    "there is a cat on the mat".to_string(),
];
let hypothesis = "the cat is on the mat".to_string();

let bleu = bleu_score(&references, &hypothesis, Some(4), None)?;
println!("BLEU-4: {:.4}", bleu);
```

#### ROUGE Score (Text Summarization)

```rust
use torsh_metrics::deep_learning::*;

let reference = "the quick brown fox jumps over the lazy dog".to_string();
let hypothesis = "the fast brown fox jumps over the dog".to_string();

let rouge_1 = rouge_n(&reference, &hypothesis, 1)?;
let rouge_2 = rouge_n(&reference, &hypothesis, 2)?;
let rouge_l = rouge_l(&reference, &hypothesis)?;

println!("ROUGE-1: {:.4}, ROUGE-2: {:.4}, ROUGE-L: {:.4}", rouge_1, rouge_2, rouge_l);
```

### Fairness Metrics

```rust
use torsh_metrics::fairness::*;

let y_true = tensor![1, 0, 1, 1, 0, 1, 0, 0];
let y_pred = tensor![1, 0, 1, 0, 0, 1, 1, 0];
let sensitive_attr = tensor![0, 0, 1, 1, 0, 1, 0, 1];  // e.g., gender

// Demographic Parity Difference (should be close to 0)
let dp_diff = demographic_parity_difference(&y_pred, &sensitive_attr)?;
println!("Demographic Parity Difference: {:.4}", dp_diff);

// Equal Opportunity Difference (for positive class)
let eo_diff = equal_opportunity_difference(&y_true, &y_pred, &sensitive_attr)?;
println!("Equal Opportunity Difference: {:.4}", eo_diff);

// Disparate Impact Ratio (should be close to 1)
let di_ratio = disparate_impact_ratio(&y_pred, &sensitive_attr)?;
println!("Disparate Impact Ratio: {:.4}", di_ratio);

// Statistical Parity
let stat_parity = statistical_parity(&y_pred, &sensitive_attr)?;
```

### Uncertainty Metrics

#### Calibration

```rust
use torsh_metrics::uncertainty::*;

let y_true = tensor![0, 1, 1, 0, 1];
let y_prob = tensor![0.2, 0.8, 0.9, 0.3, 0.6];  // Predicted probabilities

// Expected Calibration Error (ECE)
let ece = expected_calibration_error(&y_true, &y_prob, 10)?;  // 10 bins
println!("ECE: {:.4}", ece);

// Maximum Calibration Error (MCE)
let mce = maximum_calibration_error(&y_true, &y_prob, 10)?;

// Brier Score (lower is better, 0 is perfect)
let brier = brier_score(&y_true, &y_prob)?;
println!("Brier Score: {:.4}", brier);

// Reliability diagram
let (bin_accs, bin_confs, bin_counts) = calibration_curve(&y_true, &y_prob, 10, "uniform")?;
```

#### Prediction Intervals

```rust
use torsh_metrics::uncertainty::*;

let y_true = tensor![3.0, 5.0, 2.0, 8.0];
let y_lower = tensor![2.5, 4.5, 1.5, 7.0];  // Lower bound
let y_upper = tensor![3.5, 5.5, 2.5, 9.0];  // Upper bound

// Prediction Interval Coverage Probability (PICP)
let picp = prediction_interval_coverage(&y_true, &y_lower, &y_upper)?;
println!("Coverage: {:.2}%", picp * 100.0);

// Mean Prediction Interval Width (MPIW)
let mpiw = mean_prediction_interval_width(&y_lower, &y_upper)?;
println!("Mean Interval Width: {:.4}", mpiw);
```

### Statistical Tests

```rust
use torsh_metrics::statistical_tests::*;

let model1_scores = tensor![0.85, 0.87, 0.83, 0.89, 0.84];
let model2_scores = tensor![0.82, 0.84, 0.81, 0.86, 0.83];

// Paired t-test
let (t_stat, p_value) = paired_t_test(&model1_scores, &model2_scores)?;
println!("t-statistic: {:.4}, p-value: {:.4}", t_stat, p_value);

if p_value < 0.05 {
    println!("Significant difference at α=0.05");
}

// Wilcoxon signed-rank test (non-parametric)
let (w_stat, p_value) = wilcoxon_test(&model1_scores, &model2_scores)?;

// Mann-Whitney U test (independent samples)
let group1 = tensor![0.85, 0.87, 0.83];
let group2 = tensor![0.82, 0.84, 0.81];
let (u_stat, p_value) = mann_whitney_u_test(&group1, &group2)?;

// Permutation test
let (p_value, null_distribution) = permutation_test(&model1_scores, &model2_scores, 10000)?;
```

### Model Selection

```rust
use torsh_metrics::model_selection::*;

let y_true = tensor![0, 1, 2, 0, 1, 2];
let y_pred_model1 = tensor![0, 1, 1, 0, 1, 2];
let y_pred_model2 = tensor![0, 2, 2, 0, 1, 1];

// Compare multiple models
let models = vec![y_pred_model1, y_pred_model2];
let scores = compare_models(&y_true, &models, "accuracy")?;

println!("Model scores: {:?}", scores);

// Cross-validation scores analysis
let cv_scores = tensor![0.85, 0.87, 0.83, 0.89, 0.84];

let mean_score = cv_scores.mean()?;
let std_score = cv_scores.std(Some(1))?;

println!("CV Score: {:.4} ± {:.4}", mean_score, std_score);
```

### Streaming Metrics

```rust
use torsh_metrics::streaming::*;

// For large datasets that don't fit in memory
let mut metric = StreamingAccuracy::new();

for batch in data_loader {
    let (y_true, y_pred) = batch;
    metric.update(&y_true, &y_pred)?;
}

let final_accuracy = metric.compute()?;
println!("Accuracy: {:.4}", final_accuracy);

// Streaming confusion matrix
let mut cm = StreamingConfusionMatrix::new(num_classes);

for batch in data_loader {
    cm.update(&y_true, &y_pred)?;
}

let confusion_matrix = cm.compute()?;
```

### GPU Acceleration

```rust
use torsh_metrics::prelude::*;

// Move tensors to GPU
let y_true = tensor![...].to_device("cuda:0")?;
let y_pred = tensor![...].to_device("cuda:0")?;

// Metrics are automatically computed on GPU
let acc = accuracy(&y_true, &y_pred)?;
let f1 = f1_score(&y_true, &y_pred, None, Some("macro"), None, None)?;

// For very large evaluations
let large_y_true = randint(0, 1000, &[10_000_000])?.to_device("cuda:0")?;
let large_y_pred = randint(0, 1000, &[10_000_000])?.to_device("cuda:0")?;

let acc = accuracy(&large_y_true, &large_y_pred)?;  // Computed on GPU
```

### Custom Metrics

```rust
use torsh_metrics::prelude::*;

// Define custom metric
struct CustomMetric;

impl Metric for CustomMetric {
    fn compute(&self, y_true: &Tensor, y_pred: &Tensor) -> Result<f32> {
        // Your custom logic
        let diff = (y_true - y_pred)?.abs()?;
        let custom_score = diff.mean()?.item();
        Ok(custom_score)
    }
}

// Use custom metric
let metric = CustomMetric;
let score = metric.compute(&y_true, &y_pred)?;

// Register for use with metric name
register_metric("custom", Box::new(CustomMetric))?;
let score = compute_metric("custom", &y_true, &y_pred)?;
```

## Integration with SciRS2

This crate leverages the SciRS2 ecosystem for:

- High-performance metric computations through `scirs2-metrics`
- Optimized tensor operations via `scirs2-core`
- Statistical functions for hypothesis testing

All implementations follow the [SciRS2 POLICY](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md) for consistent APIs and optimal performance.

## Performance Tips

1. **Use GPU acceleration** for large-scale evaluations (>1M samples)
2. **Use streaming metrics** for datasets that don't fit in memory
3. **Batch predictions** before computing metrics when possible
4. **Cache metric objects** when computing multiple metrics on the same data
5. **Use parallel features** with `features = ["parallel"]` in Cargo.toml

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.
