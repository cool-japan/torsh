//! Comprehensive tests for advanced metrics modules
//!
//! Tests for streaming, clustering, ranking, and statistical validation metrics.

use approx::assert_relative_eq;
use torsh_core::device::DeviceType;
use torsh_metrics::{
    classification::Accuracy,
    clustering::{
        AdjustedRandIndex, CalinskiHarabasz, DaviesBouldin, DistanceMetric, DunnIndex, Inertia,
        NormalizedMutualInfo, SilhouetteScore, VMeasure,
    },
    ranking::{
        Coverage, F1AtK, HitRateAtK, LearningToRankMetrics, MeanAveragePrecision,
        MeanReciprocalRank, PrecisionAtK, RankingAUC, RecallAtK, NDCG,
    },
    statistics::{
        BootstrapCI, CrossValidator, EffectSize, MultipleComparisonCorrection, PermutationTest,
    },
    streaming::{StreamingAUROC, StreamingAccuracy, StreamingConfusionMatrix, StreamingStats},
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
mod streaming_tests {
    use super::*;

    #[test]
    fn test_streaming_accuracy() {
        let mut streaming_accuracy = StreamingAccuracy::new();

        // First batch
        let predictions1 = tensor_from_2d(&[
            &[0.1, 0.9], // Correct
            &[0.8, 0.2], // Correct
        ]);
        let targets1 = tensor_from_slice(&[1.0, 0.0]);

        streaming_accuracy.update(&predictions1, &targets1);
        assert_relative_eq!(streaming_accuracy.compute(), 1.0, epsilon = 1e-6);

        // Second batch with some errors
        let predictions2 = tensor_from_2d(&[
            &[0.3, 0.7], // Correct
            &[0.2, 0.8], // Incorrect (predicted 1, actual 0)
        ]);
        let targets2 = tensor_from_slice(&[1.0, 0.0]);

        streaming_accuracy.update(&predictions2, &targets2);
        // Total: 3 correct out of 4 predictions
        assert_relative_eq!(streaming_accuracy.compute(), 0.75, epsilon = 1e-6);
    }

    #[test]
    fn test_streaming_top_k_accuracy() {
        let mut top_k_accuracy = StreamingAccuracy::top_k(2);

        let predictions = tensor_from_2d(&[
            &[0.1, 0.2, 0.7], // Top-2: [2, 1], target: 2 ✓
            &[0.3, 0.6, 0.1], // Top-2: [1, 0], target: 0 ✓
            &[0.8, 0.1, 0.1], // Top-2: [0, 1], target: 2 ✗
        ]);
        let targets = tensor_from_slice(&[2.0, 0.0, 2.0]);

        top_k_accuracy.update(&predictions, &targets);
        assert_relative_eq!(top_k_accuracy.compute(), 2.0 / 3.0, epsilon = 1e-5);
    }

    #[test]
    fn test_streaming_confusion_matrix() {
        let mut confusion_matrix = StreamingConfusionMatrix::new(3);

        let predictions = tensor_from_2d(&[
            &[0.7, 0.2, 0.1], // Predicted: 0, Actual: 0
            &[0.1, 0.8, 0.1], // Predicted: 1, Actual: 1
            &[0.2, 0.3, 0.5], // Predicted: 2, Actual: 0 (error)
        ]);
        let targets = tensor_from_slice(&[0.0, 1.0, 0.0]);

        confusion_matrix.update(&predictions, &targets);

        let matrix = confusion_matrix.matrix();
        assert_eq!(matrix[0][0], 1); // True class 0, predicted 0
        assert_eq!(matrix[1][1], 1); // True class 1, predicted 1
        assert_eq!(matrix[0][2], 1); // True class 0, predicted 2 (error)

        let precision_per_class = confusion_matrix.precision_per_class();
        let recall_per_class = confusion_matrix.recall_per_class();
        let f1_per_class = confusion_matrix.f1_per_class();

        assert_eq!(precision_per_class.len(), 3);
        assert_eq!(recall_per_class.len(), 3);
        assert_eq!(f1_per_class.len(), 3);
    }

    #[test]
    fn test_streaming_auroc() {
        let mut auroc = StreamingAUROC::new(1000);

        let predictions = tensor_from_slice(&[0.9, 0.8, 0.3, 0.2]);
        let targets = tensor_from_slice(&[1.0, 1.0, 0.0, 0.0]);

        auroc.update(&predictions, &targets);
        let score = auroc.compute();

        // Perfect separation should give AUROC = 1.0
        assert_relative_eq!(score, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_streaming_stats() {
        let mut stats = StreamingStats::new();

        let values = &[1.0, 2.0, 3.0, 4.0, 5.0];
        for &value in values {
            stats.update(value);
        }

        assert_relative_eq!(stats.mean(), 3.0, epsilon = 1e-6);
        assert_relative_eq!(stats.variance(), 2.0, epsilon = 1e-6);
        assert_eq!(stats.count(), 5);

        let (min_val, max_val) = stats.min_max();
        assert_eq!(min_val, 1.0);
        assert_eq!(max_val, 5.0);
    }

    #[test]
    fn test_streaming_stats_merge() {
        let mut stats1 = StreamingStats::new();
        stats1.update(1.0);
        stats1.update(2.0);

        let mut stats2 = StreamingStats::new();
        stats2.update(3.0);
        stats2.update(4.0);

        stats1.merge(&stats2);

        assert_relative_eq!(stats1.mean(), 2.5, epsilon = 1e-6);
        assert_eq!(stats1.count(), 4);
    }
}

#[cfg(test)]
mod clustering_tests {
    use super::*;

    fn create_simple_clustering_data() -> (Tensor, Tensor) {
        // Create simple 2D clustering data: two clear clusters
        let data = tensor_from_2d(&[
            // Cluster 1 (around origin)
            &[0.0, 0.0],
            &[0.1, 0.1],
            &[0.0, 0.2],
            &[0.2, 0.0],
            // Cluster 2 (around (10, 10))
            &[10.0, 10.0],
            &[10.1, 10.1],
            &[10.0, 10.2],
            &[10.2, 10.0],
        ]);

        let labels = tensor_from_slice(&[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        (data, labels)
    }

    #[test]
    fn test_silhouette_score() {
        let (data, labels) = create_simple_clustering_data();
        let silhouette = SilhouetteScore::new(DistanceMetric::Euclidean);

        let score = silhouette.compute_score(&data, &labels);
        // Should be very high for well-separated clusters
        assert!(score > 0.8);
    }

    #[test]
    fn test_davies_bouldin_index() {
        let (data, labels) = create_simple_clustering_data();
        let davies_bouldin = DaviesBouldin;

        let score = davies_bouldin.compute_score(&data, &labels);
        // Should be low for well-separated clusters (lower is better)
        assert!(score < 1.0);
    }

    #[test]
    fn test_calinski_harabasz_index() {
        let (data, labels) = create_simple_clustering_data();
        let calinski_harabasz = CalinskiHarabasz;

        let score = calinski_harabasz.compute_score(&data, &labels);
        // Should be high for well-separated clusters (higher is better)
        assert!(score > 10.0);
    }

    #[test]
    fn test_dunn_index() {
        let (data, labels) = create_simple_clustering_data();
        let dunn = DunnIndex;

        let score = dunn.compute_score(&data, &labels);
        // Should be high for well-separated clusters (higher is better)
        assert!(score > 1.0);
    }

    #[test]
    fn test_adjusted_rand_index_perfect() {
        // Perfect clustering match
        let true_labels = tensor_from_slice(&[0.0, 0.0, 1.0, 1.0]);
        let pred_labels = tensor_from_slice(&[0.0, 0.0, 1.0, 1.0]);

        let ari = AdjustedRandIndex;
        let score = ari.compute_score(&true_labels, &pred_labels);
        assert_relative_eq!(score, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_adjusted_rand_index_random() {
        // This specific clustering has ARI = -0.5 (worse than random)
        let true_labels = tensor_from_slice(&[0.0, 0.0, 1.0, 1.0]);
        let pred_labels = tensor_from_slice(&[0.0, 1.0, 0.0, 1.0]);

        let ari = AdjustedRandIndex;
        let score = ari.compute_score(&true_labels, &pred_labels);
        assert_relative_eq!(score, -0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_normalized_mutual_info() {
        let true_labels = tensor_from_slice(&[0.0, 0.0, 1.0, 1.0]);
        let pred_labels = tensor_from_slice(&[0.0, 0.0, 1.0, 1.0]);

        let nmi = NormalizedMutualInfo::new("arithmetic");
        let score = nmi.compute_score(&true_labels, &pred_labels);
        assert_relative_eq!(score, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_v_measure() {
        let true_labels = tensor_from_slice(&[0.0, 0.0, 1.0, 1.0]);
        let pred_labels = tensor_from_slice(&[0.0, 0.0, 1.0, 1.0]);

        let v_measure = VMeasure::new(1.0);
        let score = v_measure.compute_score(&true_labels, &pred_labels);
        assert_relative_eq!(score, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_inertia() {
        let (data, labels) = create_simple_clustering_data();
        let inertia = Inertia;

        let score = inertia.compute_score(&data, &labels, None);
        // Should be low for tight clusters
        assert!(score >= 0.0);
        assert!(score < 10.0); // Reasonable upper bound for our test data
    }
}

#[cfg(test)]
mod ranking_tests {
    use super::*;

    fn create_ranking_data() -> (Tensor, Tensor) {
        // Create binary relevance labels and scores
        let relevance = tensor_from_slice(&[1.0, 0.0, 1.0, 0.0, 1.0]);
        let scores = tensor_from_slice(&[0.9, 0.3, 0.8, 0.2, 0.7]);
        (relevance, scores)
    }

    #[test]
    fn test_ndcg() {
        let (relevance, scores) = create_ranking_data();
        let ndcg = NDCG::new();

        let score = ndcg.compute_score(&relevance, &scores);
        // Should be high for good ranking
        assert!(score > 0.8);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_ndcg_at_k() {
        let (relevance, scores) = create_ranking_data();
        let ndcg_at_3 = NDCG::new().at_k(3);

        let score = ndcg_at_3.compute_score(&relevance, &scores);
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_mean_average_precision() {
        let (relevance, scores) = create_ranking_data();
        let map = MeanAveragePrecision::new();

        let score = map.compute_score(&relevance, &scores);
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_mean_reciprocal_rank() {
        let (relevance, scores) = create_ranking_data();
        let mrr = MeanReciprocalRank::new();

        let score = mrr.compute_score(&relevance, &scores);
        // First relevant item should be ranked first (score 0.9)
        assert_relative_eq!(score, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_precision_at_k() {
        let (relevance, scores) = create_ranking_data();
        let precision_at_3 = PrecisionAtK::new(3);

        let score = precision_at_3.compute_score(&relevance, &scores);
        // Top-3 scores are [0.9, 0.8, 0.7] with relevance [1, 1, 1]
        assert_relative_eq!(score, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_recall_at_k() {
        let (relevance, scores) = create_ranking_data();
        let recall_at_3 = RecallAtK::new(3);

        let score = recall_at_3.compute_score(&relevance, &scores);
        // All 3 relevant items are in top-3
        assert_relative_eq!(score, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_f1_at_k() {
        let (relevance, scores) = create_ranking_data();
        let f1_at_3 = F1AtK::new(3);

        let score = f1_at_3.compute_score(&relevance, &scores);
        // Perfect precision and recall should give F1 = 1.0
        assert_relative_eq!(score, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_hit_rate_at_k() {
        let (relevance, scores) = create_ranking_data();
        let hit_rate_at_1 = HitRateAtK::new(1);

        let score = hit_rate_at_1.compute_score(&relevance, &scores);
        // Top-1 item is relevant
        assert_relative_eq!(score, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_ranking_auc() {
        let (relevance, scores) = create_ranking_data();
        let ranking_auc = RankingAUC;

        let score = ranking_auc.compute_score(&relevance, &scores);
        // Should be 1.0 since all positive scores > all negative scores
        assert_relative_eq!(score, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_coverage() {
        let coverage = Coverage::new(100); // Catalog size of 100

        // Simulate batch recommendations
        let scores_batch = tensor_from_2d(&[
            &[0.9, 0.8, 0.1, 0.2], // User 1 recommendations: items 0,1
            &[0.1, 0.2, 0.9, 0.8], // User 2 recommendations: items 2,3
        ]);

        let coverage_score = coverage.compute_coverage_at_k(&scores_batch, 2);
        // Should recommend 4 unique items out of 100
        assert_relative_eq!(coverage_score, 4.0 / 100.0, epsilon = 1e-6);
    }

    #[test]
    fn test_learning_to_rank_metrics() {
        let (relevance, scores) = create_ranking_data();
        let ltr_metrics = LearningToRankMetrics::new(&[1, 3, 5]);

        let results = ltr_metrics.compute_all(&relevance, &scores);

        assert!(results.contains_key("ndcg"));
        assert!(results.contains_key("map"));
        assert!(results.contains_key("mrr"));
        assert!(results.contains_key("precision@1"));
        assert!(results.contains_key("recall@3"));

        // All scores should be reasonable
        for (_, score) in &results {
            assert!(*score >= 0.0);
            assert!(*score <= 1.0);
        }
    }
}

#[cfg(test)]
mod statistics_tests {
    use super::*;
    use torsh_metrics::classification::Accuracy;

    #[test]
    fn test_bootstrap_ci() {
        // Use 1D predictions for binary classification
        let predictions = tensor_from_slice(&[0.9, 0.2, 0.8, 0.1]);
        let targets = tensor_from_slice(&[1.0, 0.0, 1.0, 0.0]);

        let bootstrap = BootstrapCI::new(100, 0.95).with_seed(42);
        let metric = Accuracy::new();

        let result = bootstrap.compute_ci(&metric, &predictions, &targets);

        assert_eq!(result.metric_value, 1.0); // Perfect accuracy
        assert!(result.confidence_interval.0 <= result.metric_value);
        assert!(result.confidence_interval.1 >= result.metric_value);
        assert!(result.standard_error >= 0.0);
        assert_eq!(result.n_bootstrap, 100);
    }

    #[test]
    fn test_permutation_test() {
        let predictions1 = tensor_from_2d(&[&[0.1, 0.9], &[0.8, 0.2]]);
        let targets1 = tensor_from_slice(&[1.0, 0.0]);

        let predictions2 = tensor_from_2d(&[&[0.2, 0.8], &[0.7, 0.3]]);
        let targets2 = tensor_from_slice(&[1.0, 0.0]);

        let perm_test = PermutationTest::new(100).with_seed(42);
        let metric = Accuracy::new();

        let result = perm_test.compare_metrics(
            &metric,
            &predictions1,
            &targets1,
            &predictions2,
            &targets2,
            0.05,
        );

        assert!(result.p_value >= 0.0);
        assert!(result.p_value <= 1.0);
        assert!(result.statistic >= 0.0);
        assert_eq!(result.test_type, "permutation_test");
    }

    #[test]
    fn test_cross_validator() {
        // Use 1D predictions for binary classification
        let predictions = tensor_from_slice(&[0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8, 0.1]);
        let targets = tensor_from_slice(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);

        let cv = CrossValidator::new(4).with_seed(42);
        let metric = Accuracy::new();

        let result = cv.evaluate(&metric, &predictions, &targets);

        assert_eq!(result.cv_scores.len(), 4);
        assert!(result.mean_score >= 0.0);
        assert!(result.mean_score <= 1.0);
        assert!(result.std_score >= 0.0);
        assert!(result.confidence_interval.0 <= result.confidence_interval.1);
    }

    #[test]
    fn test_bonferroni_correction() {
        let p_values = &[0.01, 0.02, 0.05, 0.10];
        let corrected = MultipleComparisonCorrection::bonferroni_correction(p_values);

        assert_eq!(corrected.len(), 4);
        assert_relative_eq!(corrected[0], 0.04, epsilon = 1e-6); // 0.01 * 4
        assert_relative_eq!(corrected[1], 0.08, epsilon = 1e-6); // 0.02 * 4
        assert_relative_eq!(corrected[2], 0.20, epsilon = 1e-6); // 0.05 * 4
        assert_relative_eq!(corrected[3], 0.40, epsilon = 1e-6); // 0.10 * 4
    }

    #[test]
    fn test_benjamini_hochberg_correction() {
        let p_values = &[0.01, 0.02, 0.05, 0.10];
        let corrected = MultipleComparisonCorrection::benjamini_hochberg_correction(p_values);

        assert_eq!(corrected.len(), 4);
        // BH correction should be less conservative than Bonferroni
        for (original, corrected_p) in p_values.iter().zip(corrected.iter()) {
            assert!(*corrected_p >= *original);
        }
    }

    #[test]
    fn test_cohens_d() {
        let group1 = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let group2 = &[3.0, 4.0, 5.0, 6.0, 7.0];

        let effect_size = EffectSize::cohens_d(group1, group2);

        // Should be negative since group2 has higher mean
        assert!(effect_size < 0.0);
        assert!(effect_size.abs() > 1.0); // Large effect size
    }

    #[test]
    fn test_cliffs_delta() {
        let group1 = &[1.0, 2.0, 3.0];
        let group2 = &[4.0, 5.0, 6.0];

        let delta = EffectSize::cliffs_delta(group1, group2);

        // All values in group2 > all values in group1
        assert_relative_eq!(delta, -1.0, epsilon = 1e-6);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use torsh_metrics::streaming::StreamingMetricCollection;

    #[test]
    fn test_streaming_metric_collection() {
        let mut collection = StreamingMetricCollection::new()
            .add_metric(StreamingAccuracy::new())
            .add_metric(StreamingAUROC::new(1000));

        let predictions = tensor_from_2d(&[&[0.1, 0.9], &[0.8, 0.2]]);
        let targets = tensor_from_slice(&[1.0, 0.0]);

        collection.update(&predictions, &targets);
        let results = collection.compute();

        assert!(results.contains_key("streaming_accuracy"));
        assert!(results.contains_key("streaming_auroc"));

        for (_, score) in &results {
            assert!(*score >= 0.0);
            assert!(*score <= 1.0);
        }
    }

    #[test]
    fn test_clustering_pipeline() {
        let (data, labels) = {
            // Create simple 2D clustering data
            let data = tensor_from_2d(&[&[0.0, 0.0], &[0.1, 0.1], &[10.0, 10.0], &[10.1, 10.1]]);
            let labels = tensor_from_slice(&[0.0, 0.0, 1.0, 1.0]);
            (data, labels)
        };

        // Test multiple clustering metrics
        let silhouette = SilhouetteScore::new(DistanceMetric::Euclidean);
        let davies_bouldin = DaviesBouldin;
        let calinski_harabasz = CalinskiHarabasz;

        let silhouette_score = silhouette.compute_score(&data, &labels);
        let davies_bouldin_score = davies_bouldin.compute_score(&data, &labels);
        let calinski_harabasz_score = calinski_harabasz.compute_score(&data, &labels);

        // Well-separated clusters should have good scores
        assert!(silhouette_score > 0.5);
        assert!(davies_bouldin_score < 2.0);
        assert!(calinski_harabasz_score > 1.0);
    }

    #[test]
    fn test_ranking_pipeline() {
        // Test complete ranking evaluation pipeline
        let relevance = tensor_from_slice(&[1.0, 0.0, 1.0, 0.0, 1.0]);
        let scores = tensor_from_slice(&[0.9, 0.3, 0.8, 0.2, 0.7]);

        let metrics = LearningToRankMetrics::new(&[1, 3, 5]);
        let results = metrics.compute_all(&relevance, &scores);

        // Verify we got all expected metrics
        let expected_metrics = &[
            "ndcg",
            "map",
            "mrr",
            "precision@1",
            "precision@3",
            "precision@5",
            "recall@1",
            "recall@3",
            "recall@5",
        ];

        for metric_name in expected_metrics {
            assert!(results.contains_key(*metric_name));
            let score = results[*metric_name];
            assert!(score >= 0.0);
            assert!(score <= 1.0);
        }
    }
}

#[cfg(test)]
mod edge_cases_advanced {
    use super::*;

    #[test]
    fn test_empty_data_clustering() {
        let empty_data = tensor_from_slice(&[]);
        let empty_labels = tensor_from_slice(&[]);

        let silhouette = SilhouetteScore::new(DistanceMetric::Euclidean);
        let score = silhouette.compute_score(&empty_data, &empty_labels);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_single_cluster_metrics() {
        // All points in same cluster
        let data = tensor_from_2d(&[&[1.0, 1.0], &[1.1, 1.1], &[0.9, 0.9]]);
        let labels = tensor_from_slice(&[0.0, 0.0, 0.0]);

        let silhouette = SilhouetteScore::new(DistanceMetric::Euclidean);
        let score = silhouette.compute_score(&data, &labels);
        assert_eq!(score, 0.0); // Single cluster should give 0
    }

    #[test]
    fn test_ranking_no_relevant_items() {
        let relevance = tensor_from_slice(&[0.0, 0.0, 0.0]);
        let scores = tensor_from_slice(&[0.9, 0.5, 0.1]);

        let map = MeanAveragePrecision::new();
        let score = map.compute_score(&relevance, &scores);
        assert_eq!(score, 0.0); // No relevant items
    }

    #[test]
    fn test_streaming_reset() {
        let mut streaming_accuracy = StreamingAccuracy::new();

        let predictions = tensor_from_2d(&[&[0.1, 0.9], &[0.8, 0.2]]);
        let targets = tensor_from_slice(&[1.0, 0.0]);

        streaming_accuracy.update(&predictions, &targets);
        assert_eq!(streaming_accuracy.count(), 2);

        streaming_accuracy.reset();
        assert_eq!(streaming_accuracy.count(), 0);
        assert_eq!(streaming_accuracy.compute(), 0.0);
    }

    #[test]
    fn test_bootstrap_insufficient_data() {
        let predictions = tensor_from_slice(&[]);
        let targets = tensor_from_slice(&[]);

        let bootstrap = BootstrapCI::new(100, 0.95);
        let metric = Accuracy::new();

        let result = bootstrap.compute_ci(&metric, &predictions, &targets);
        assert_eq!(result.n_bootstrap, 0);
        assert_eq!(result.confidence_interval, (0.0, 0.0));
    }
}
