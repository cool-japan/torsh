//! Advanced ML Metrics for Meta-Learning, Few-Shot Learning, Domain Adaptation, and Continual Learning
//!
//! This module provides specialized metrics for evaluating advanced machine learning paradigms
//! including meta-learning, few-shot learning, domain adaptation, and continual learning.

use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Meta-learning evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningMetrics {
    /// Task adaptation speed (average steps to converge per task)
    pub task_adaptation_speed: f64,
    /// Few-shot generalization gap (train vs test performance)
    pub few_shot_generalization_gap: f64,
    /// Meta-overfitting score (meta-train vs meta-test gap)
    pub meta_overfitting_score: f64,
    /// Cross-task transfer efficiency
    pub cross_task_transfer_efficiency: f64,
    /// Average task performance
    pub average_task_performance: f64,
    /// Task variance (consistency across tasks)
    pub task_variance: f64,
}

impl MetaLearningMetrics {
    /// Compute meta-learning metrics from task-specific results
    ///
    /// # Arguments
    /// * `task_train_scores` - Training scores for each task
    /// * `task_test_scores` - Test scores for each task
    /// * `adaptation_steps` - Number of gradient steps needed for each task
    /// * `baseline_scores` - Optional baseline (no adaptation) scores
    pub fn compute(
        task_train_scores: &[f64],
        task_test_scores: &[f64],
        adaptation_steps: &[usize],
        baseline_scores: Option<&[f64]>,
    ) -> Self {
        assert_eq!(
            task_train_scores.len(),
            task_test_scores.len(),
            "Train and test scores must have same length"
        );
        assert_eq!(
            task_train_scores.len(),
            adaptation_steps.len(),
            "Scores and adaptation steps must have same length"
        );

        let n_tasks = task_train_scores.len() as f64;

        // Task adaptation speed (average steps needed)
        let task_adaptation_speed = adaptation_steps.iter().sum::<usize>() as f64 / n_tasks;

        // Few-shot generalization gap (average train-test difference)
        let few_shot_generalization_gap = task_train_scores
            .iter()
            .zip(task_test_scores.iter())
            .map(|(train, test)| (train - test).abs())
            .sum::<f64>()
            / n_tasks;

        // Meta-overfitting score (variance in test scores)
        let mean_test_score = task_test_scores.iter().sum::<f64>() / n_tasks;
        let meta_overfitting_score = task_test_scores
            .iter()
            .map(|score| (score - mean_test_score).powi(2))
            .sum::<f64>()
            / n_tasks;

        // Cross-task transfer efficiency
        let cross_task_transfer_efficiency = if let Some(baseline) = baseline_scores {
            let improvement = task_test_scores
                .iter()
                .zip(baseline.iter())
                .map(|(adapted, base)| (adapted - base).max(0.0))
                .sum::<f64>()
                / n_tasks;
            improvement
        } else {
            0.0
        };

        // Average task performance
        let average_task_performance = mean_test_score;

        // Task variance
        let task_variance = meta_overfitting_score.sqrt();

        Self {
            task_adaptation_speed,
            few_shot_generalization_gap,
            meta_overfitting_score,
            cross_task_transfer_efficiency,
            average_task_performance,
            task_variance,
        }
    }
}

/// Few-shot learning evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FewShotMetrics {
    /// N-way K-shot accuracy
    pub n_way_k_shot_accuracy: f64,
    /// Support-query similarity score
    pub support_query_similarity: f64,
    /// Prototype quality score
    pub prototype_quality: f64,
    /// Episode-wise performance metrics
    pub episode_performances: Vec<f64>,
    /// Mean episode performance
    pub mean_episode_performance: f64,
    /// Episode performance variance
    pub episode_variance: f64,
}

impl FewShotMetrics {
    /// Compute few-shot learning metrics
    ///
    /// # Arguments
    /// * `episode_accuracies` - Accuracy for each evaluation episode
    /// * `support_embeddings` - Support set embeddings (n_way * k_shot, embedding_dim)
    /// * `query_embeddings` - Query set embeddings (n_queries, embedding_dim)
    /// * `support_labels` - Labels for support set
    /// * `query_labels` - Labels for query set
    pub fn compute(
        episode_accuracies: &[f64],
        support_embeddings: Option<&Array2<f64>>,
        query_embeddings: Option<&Array2<f64>>,
        support_labels: Option<&[usize]>,
        _query_labels: Option<&[usize]>,
    ) -> Self {
        let n_episodes = episode_accuracies.len() as f64;

        // N-way K-shot accuracy (mean episode accuracy)
        let n_way_k_shot_accuracy = episode_accuracies.iter().sum::<f64>() / n_episodes;

        // Episode variance
        let episode_variance = episode_accuracies
            .iter()
            .map(|acc| (acc - n_way_k_shot_accuracy).powi(2))
            .sum::<f64>()
            / n_episodes;

        // Support-query similarity (if embeddings provided)
        let support_query_similarity =
            if let (Some(support), Some(query)) = (support_embeddings, query_embeddings) {
                compute_support_query_similarity(support, query)
            } else {
                0.0
            };

        // Prototype quality (if labels and embeddings provided)
        let prototype_quality =
            if let (Some(support), Some(labels)) = (support_embeddings, support_labels) {
                compute_prototype_quality(support, labels)
            } else {
                0.0
            };

        Self {
            n_way_k_shot_accuracy,
            support_query_similarity,
            prototype_quality,
            episode_performances: episode_accuracies.to_vec(),
            mean_episode_performance: n_way_k_shot_accuracy,
            episode_variance,
        }
    }
}

/// Domain adaptation evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainAdaptationMetrics {
    /// Maximum Mean Discrepancy (MMD) between source and target
    pub mmd_distance: f64,
    /// CORAL distance (covariance alignment)
    pub coral_distance: f64,
    /// Adaptation gap (source accuracy - target accuracy)
    pub adaptation_gap: f64,
    /// Source domain accuracy
    pub source_accuracy: f64,
    /// Target domain accuracy
    pub target_accuracy: f64,
    /// Domain confusion score (0-1, higher is better)
    pub domain_confusion_score: f64,
}

impl DomainAdaptationMetrics {
    /// Compute domain adaptation metrics
    ///
    /// # Arguments
    /// * `source_features` - Features from source domain (n_source, feature_dim)
    /// * `target_features` - Features from target domain (n_target, feature_dim)
    /// * `source_accuracy` - Accuracy on source domain
    /// * `target_accuracy` - Accuracy on target domain
    /// * `domain_predictions` - Domain classifier predictions (0=source, 1=target)
    /// * `domain_labels` - True domain labels
    pub fn compute(
        source_features: &Array2<f64>,
        target_features: &Array2<f64>,
        source_accuracy: f64,
        target_accuracy: f64,
        domain_predictions: Option<&[usize]>,
        domain_labels: Option<&[usize]>,
    ) -> Self {
        // MMD distance
        let mmd_distance = compute_mmd(source_features, target_features);

        // CORAL distance
        let coral_distance = compute_coral(source_features, target_features);

        // Adaptation gap
        let adaptation_gap = (source_accuracy - target_accuracy).abs();

        // Domain confusion score (how well domain classifier is confused)
        let domain_confusion_score =
            if let (Some(preds), Some(labels)) = (domain_predictions, domain_labels) {
                let correct = preds
                    .iter()
                    .zip(labels.iter())
                    .filter(|(p, l)| p == l)
                    .count();
                // For adaptation, we want domain classifier to fail (be confused)
                1.0 - (correct as f64 / preds.len() as f64)
            } else {
                0.0
            };

        Self {
            mmd_distance,
            coral_distance,
            adaptation_gap,
            source_accuracy,
            target_accuracy,
            domain_confusion_score,
        }
    }
}

/// Continual learning evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinualLearningMetrics {
    /// Backward transfer (how learning new tasks affects old tasks)
    pub backward_transfer: f64,
    /// Forward transfer (how old knowledge helps new tasks)
    pub forward_transfer: f64,
    /// Forgetting measure (average forgetting across tasks)
    pub forgetting_measure: f64,
    /// Average accuracy across all tasks after all training
    pub average_accuracy: f64,
    /// Learning curve area (integral under accuracy curve)
    pub learning_curve_area: f64,
    /// Per-task forgetting
    pub per_task_forgetting: Vec<f64>,
}

impl ContinualLearningMetrics {
    /// Compute continual learning metrics
    ///
    /// # Arguments
    /// * `task_accuracies` - Matrix of accuracies (n_tasks, n_tasks) where entry (i, j)
    ///                       is accuracy on task i after training on task j
    pub fn compute(task_accuracies: &Array2<f64>) -> Self {
        let n_tasks = task_accuracies.nrows();
        assert_eq!(
            task_accuracies.ncols(),
            n_tasks,
            "Task accuracy matrix must be square"
        );

        // Average accuracy (diagonal average after all training)
        let average_accuracy = (0..n_tasks)
            .map(|i| task_accuracies[[i, n_tasks - 1]])
            .sum::<f64>()
            / n_tasks as f64;

        // Backward transfer: BWT = 1/(T-1) * sum_{i<T} (a_{i,T} - a_{i,i})
        let backward_transfer = if n_tasks > 1 {
            (0..n_tasks - 1)
                .map(|i| task_accuracies[[i, n_tasks - 1]] - task_accuracies[[i, i]])
                .sum::<f64>()
                / (n_tasks - 1) as f64
        } else {
            0.0
        };

        // Forward transfer: FWT = 1/(T-1) * sum_{i>1} (a_{i,i-1} - a_{i,i}^random)
        // We approximate random accuracy as 0.0 or use first-task baseline
        let forward_transfer = if n_tasks > 1 {
            (1..n_tasks)
                .map(|i| task_accuracies[[i, i - 1]] - 0.0) // Assumes random baseline is 0
                .sum::<f64>()
                / (n_tasks - 1) as f64
        } else {
            0.0
        };

        // Forgetting measure: F = 1/T * sum_i max_{j<T} (a_{i,j} - a_{i,T})
        let per_task_forgetting: Vec<f64> = (0..n_tasks)
            .map(|i| {
                let max_prev_acc = (0..n_tasks)
                    .map(|j| task_accuracies[[i, j]])
                    .fold(f64::NEG_INFINITY, f64::max);
                (max_prev_acc - task_accuracies[[i, n_tasks - 1]]).max(0.0)
            })
            .collect();

        let forgetting_measure = per_task_forgetting.iter().sum::<f64>() / n_tasks as f64;

        // Learning curve area (sum of accuracies across training)
        let learning_curve_area = (0..n_tasks)
            .map(|j| (0..=j).map(|i| task_accuracies[[i, j]]).sum::<f64>() / (j + 1) as f64)
            .sum::<f64>()
            / n_tasks as f64;

        Self {
            backward_transfer,
            forward_transfer,
            forgetting_measure,
            average_accuracy,
            learning_curve_area,
            per_task_forgetting,
        }
    }

    /// Compute plasticity (ability to learn new tasks)
    pub fn plasticity(&self) -> f64 {
        self.forward_transfer.max(0.0)
    }

    /// Compute stability (resistance to forgetting)
    pub fn stability(&self) -> f64 {
        1.0 - self.forgetting_measure
    }

    /// Compute plasticity-stability balance
    pub fn plasticity_stability_balance(&self) -> f64 {
        2.0 * self.plasticity() * self.stability() / (self.plasticity() + self.stability() + 1e-8)
    }
}

// Helper functions

/// Compute support-query similarity score
fn compute_support_query_similarity(
    support_embeddings: &Array2<f64>,
    query_embeddings: &Array2<f64>,
) -> f64 {
    let n_support = support_embeddings.nrows();
    let n_query = query_embeddings.nrows();

    let mut total_similarity = 0.0;

    for i in 0..n_query {
        let query = query_embeddings.row(i);
        let mut max_similarity = f64::NEG_INFINITY;

        for j in 0..n_support {
            let support = support_embeddings.row(j);
            let similarity = cosine_similarity(&query.to_vec(), &support.to_vec());
            max_similarity = max_similarity.max(similarity);
        }

        total_similarity += max_similarity;
    }

    total_similarity / n_query as f64
}

/// Compute prototype quality score
fn compute_prototype_quality(support_embeddings: &Array2<f64>, support_labels: &[usize]) -> f64 {
    let unique_labels: Vec<usize> = {
        let mut labels = support_labels.to_vec();
        labels.sort_unstable();
        labels.dedup();
        labels
    };

    let n_classes = unique_labels.len();
    let embedding_dim = support_embeddings.ncols();

    // Compute prototypes (class centroids)
    let mut prototypes = Vec::with_capacity(n_classes);
    for &label in &unique_labels {
        let class_embeddings: Vec<_> = support_labels
            .iter()
            .enumerate()
            .filter(|(_, l)| **l == label)
            .map(|(i, _)| support_embeddings.row(i))
            .collect();

        let n_samples = class_embeddings.len();
        let mut prototype = vec![0.0; embedding_dim];
        for emb in class_embeddings {
            for (j, val) in emb.iter().enumerate() {
                prototype[j] += val;
            }
        }
        for val in &mut prototype {
            *val /= n_samples as f64;
        }
        prototypes.push(prototype);
    }

    // Compute inter-class distance (higher is better)
    let mut inter_class_distances = Vec::new();
    for i in 0..n_classes {
        for j in (i + 1)..n_classes {
            let dist = euclidean_distance(&prototypes[i], &prototypes[j]);
            inter_class_distances.push(dist);
        }
    }

    // Compute intra-class variance (lower is better)
    let mut intra_class_variances = Vec::new();
    for (idx, &label) in unique_labels.iter().enumerate() {
        let class_embeddings: Vec<_> = support_labels
            .iter()
            .enumerate()
            .filter(|(_, l)| **l == label)
            .map(|(i, _)| support_embeddings.row(i).to_vec())
            .collect();

        let prototype = &prototypes[idx];
        let variance = class_embeddings
            .iter()
            .map(|emb| euclidean_distance(emb, prototype).powi(2))
            .sum::<f64>()
            / class_embeddings.len() as f64;
        intra_class_variances.push(variance);
    }

    let mean_inter_class_dist =
        inter_class_distances.iter().sum::<f64>() / inter_class_distances.len().max(1) as f64;
    let mean_intra_class_var =
        intra_class_variances.iter().sum::<f64>() / intra_class_variances.len().max(1) as f64;

    // Quality = inter-class distance / (intra-class variance + epsilon)
    mean_inter_class_dist / (mean_intra_class_var + 1e-8)
}

/// Compute Maximum Mean Discrepancy (MMD) between two distributions
fn compute_mmd(source_features: &Array2<f64>, target_features: &Array2<f64>) -> f64 {
    // Compute mean embeddings
    let source_mean = compute_mean_embedding(source_features);
    let target_mean = compute_mean_embedding(target_features);

    // MMD = ||mean(source) - mean(target)||^2
    euclidean_distance(&source_mean, &target_mean).powi(2)
}

/// Compute CORAL distance (covariance alignment)
fn compute_coral(source_features: &Array2<f64>, target_features: &Array2<f64>) -> f64 {
    let source_cov = compute_covariance(source_features);
    let target_cov = compute_covariance(target_features);

    // Frobenius norm of difference
    let mut distance = 0.0;
    for i in 0..source_cov.nrows() {
        for j in 0..source_cov.ncols() {
            distance += (source_cov[[i, j]] - target_cov[[i, j]]).powi(2);
        }
    }

    distance.sqrt()
}

/// Compute mean embedding
fn compute_mean_embedding(features: &Array2<f64>) -> Vec<f64> {
    let n_samples = features.nrows();
    let n_features = features.ncols();

    let mut mean = vec![0.0; n_features];
    for i in 0..n_samples {
        for j in 0..n_features {
            mean[j] += features[[i, j]];
        }
    }

    for val in &mut mean {
        *val /= n_samples as f64;
    }

    mean
}

/// Compute covariance matrix
fn compute_covariance(features: &Array2<f64>) -> Array2<f64> {
    let n_samples = features.nrows();
    let n_features = features.ncols();

    // Center the data
    let mean = compute_mean_embedding(features);
    let mut centered = features.clone();
    for i in 0..n_samples {
        for j in 0..n_features {
            centered[[i, j]] -= mean[j];
        }
    }

    // Compute covariance: C = (1/n) * X^T * X
    let mut cov = Array2::<f64>::zeros((n_features, n_features));
    for i in 0..n_features {
        for j in 0..n_features {
            let mut sum = 0.0;
            for k in 0..n_samples {
                sum += centered[[k, i]] * centered[[k, j]];
            }
            cov[[i, j]] = sum / n_samples as f64;
        }
    }

    cov
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    dot_product / (norm_a * norm_b + 1e-8)
}

/// Compute Euclidean distance between two vectors
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_meta_learning_metrics() {
        let task_train_scores = vec![0.9, 0.85, 0.88, 0.92];
        let task_test_scores = vec![0.85, 0.80, 0.82, 0.87];
        let adaptation_steps = vec![10, 15, 12, 8];
        let baseline_scores = vec![0.6, 0.55, 0.58, 0.62];

        let metrics = MetaLearningMetrics::compute(
            &task_train_scores,
            &task_test_scores,
            &adaptation_steps,
            Some(&baseline_scores),
        );

        assert!(metrics.task_adaptation_speed > 0.0);
        assert!(metrics.few_shot_generalization_gap >= 0.0);
        assert!(metrics.average_task_performance > 0.0);
        assert!(metrics.cross_task_transfer_efficiency > 0.0);
    }

    #[test]
    fn test_few_shot_metrics() {
        let episode_accuracies = vec![0.85, 0.82, 0.88, 0.86, 0.84];

        let metrics = FewShotMetrics::compute(&episode_accuracies, None, None, None, None);

        assert!((metrics.n_way_k_shot_accuracy - 0.85).abs() < 0.01);
        assert!(metrics.episode_variance >= 0.0);
        assert_eq!(metrics.episode_performances.len(), 5);
    }

    #[test]
    fn test_domain_adaptation_metrics() {
        let source_features = Array2::from_shape_fn((100, 10), |(i, j)| (i + j) as f64 * 0.1);
        let target_features = Array2::from_shape_fn((100, 10), |(i, j)| (i + j) as f64 * 0.1 + 1.0);

        let metrics = DomainAdaptationMetrics::compute(
            &source_features,
            &target_features,
            0.90,
            0.75,
            None,
            None,
        );

        assert!(metrics.mmd_distance >= 0.0);
        assert!(metrics.coral_distance >= 0.0);
        assert!((metrics.adaptation_gap - 0.15).abs() < 0.01);
        assert_eq!(metrics.source_accuracy, 0.90);
        assert_eq!(metrics.target_accuracy, 0.75);
    }

    #[test]
    fn test_continual_learning_metrics() {
        // Task accuracy matrix: task_accuracies[i][j] = accuracy on task i after training on task j
        let task_accuracies = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.9, 0.85, 0.80, // Task 0: degrades as we learn new tasks
                0.0, 0.88, 0.85, // Task 1: improves then degrades slightly
                0.0, 0.0, 0.92, // Task 2: only learned at the end
            ],
        )
        .unwrap();

        let metrics = ContinualLearningMetrics::compute(&task_accuracies);

        assert!(metrics.average_accuracy > 0.0);
        assert!(metrics.forgetting_measure >= 0.0);
        assert_eq!(metrics.per_task_forgetting.len(), 3);
        assert!(metrics.stability() >= 0.0 && metrics.stability() <= 1.0);
        assert!(metrics.plasticity() >= 0.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((euclidean_distance(&a, &b) - 1.0).abs() < 1e-6);

        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
    }
}
