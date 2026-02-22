//! Clustering evaluation metrics
//!
//! This module provides comprehensive evaluation metrics for clustering algorithms,
//! organized into logical categories:
//!
//! - **Distance-based metrics**: Metrics that rely on distance calculations between
//!   data points and cluster centers (silhouette score, Calinski-Harabasz, Davies-Bouldin)
//! - **Information-theoretic metrics**: Metrics based on information theory and entropy
//!   (NMI, AMI, homogeneity, completeness, V-measure)
//! - **Set-based metrics**: Metrics that evaluate clustering based on set comparisons
//!   and combinatorial analysis (ARI, Fowlkes-Mallows)
//!
//! All metrics are implemented using SciRS2 as the computational backend for
//! efficient array operations and scientific computing.

pub mod distance_based;
pub mod gap_statistic;
pub mod information_theoretic;
pub mod set_based;
pub mod utils;

// Re-export distance-based metrics
pub use distance_based::{
    calinski_harabasz_score, davies_bouldin_score, dunn_index, silhouette_score, xie_beni_index,
};

// Re-export information-theoretic metrics
pub use information_theoretic::{
    adjusted_mutual_info_score, completeness_score, homogeneity_score,
    normalized_mutual_info_score, v_measure_score,
};

// Re-export set-based metrics
pub use set_based::{adjusted_rand_score, fowlkes_mallows_score};

// Re-export utility functions that may be useful externally
pub use utils::{combinations, compute_entropy};

// Re-export gap statistic for optimal k selection
pub use gap_statistic::{GapStatistic, GapStatisticConfig, GapStatisticResult};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::Tensor;

    #[test]
    fn test_all_metrics_integration() -> Result<(), Box<dyn std::error::Error>> {
        // Create test data with two well-separated clusters
        let data = Tensor::from_vec(
            vec![
                // Cluster 0
                0.0, 0.0, 0.1, 0.1, // Cluster 1
                5.0, 5.0, 5.1, 5.1,
            ],
            &[4, 2],
        )?;

        let labels_true = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[4])?;
        let labels_pred = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[4])?; // Perfect match

        // Test distance-based metrics
        let silhouette = silhouette_score(&data, &labels_pred)?;
        assert!(
            silhouette > 0.5,
            "Silhouette score should be high for well-separated clusters"
        );

        let ch_score = calinski_harabasz_score(&data, &labels_pred)?;
        assert!(
            ch_score > 1.0,
            "CH score should be > 1 for well-separated clusters"
        );

        let db_score = davies_bouldin_score(&data, &labels_pred)?;
        assert!(db_score >= 0.0, "DB score should be non-negative");

        // Test information-theoretic metrics
        let nmi = normalized_mutual_info_score(&labels_true, &labels_pred)?;
        assert_relative_eq!(nmi, 1.0, epsilon = 1e-6);

        let ami = adjusted_mutual_info_score(&labels_true, &labels_pred)?;
        assert_relative_eq!(ami, 1.0, epsilon = 1e-6);

        let homogeneity = homogeneity_score(&labels_true, &labels_pred)?;
        assert_relative_eq!(homogeneity, 1.0, epsilon = 1e-6);

        let completeness = completeness_score(&labels_true, &labels_pred)?;
        assert_relative_eq!(completeness, 1.0, epsilon = 1e-6);

        let v_measure = v_measure_score(&labels_true, &labels_pred)?;
        assert_relative_eq!(v_measure, 1.0, epsilon = 1e-6);

        // Test set-based metrics
        let ari = adjusted_rand_score(&labels_true, &labels_pred)?;
        assert_relative_eq!(ari, 1.0, epsilon = 1e-6);

        let fm = fowlkes_mallows_score(&labels_true, &labels_pred)?;
        assert_relative_eq!(fm, 1.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_metrics_consistency() -> Result<(), Box<dyn std::error::Error>> {
        // Test with random clustering - should give reasonable scores
        let data = Tensor::from_vec(
            vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1, 9.0, 9.0, 9.1, 9.1],
            &[6, 2],
        )?;

        let labels_true = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0], &[6])?;
        let labels_pred = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0], &[6])?;

        // All perfect match metrics should be 1.0
        let perfect_metrics = vec![
            normalized_mutual_info_score(&labels_true, &labels_pred)?,
            adjusted_mutual_info_score(&labels_true, &labels_pred)?,
            homogeneity_score(&labels_true, &labels_pred)?,
            completeness_score(&labels_true, &labels_pred)?,
            v_measure_score(&labels_true, &labels_pred)?,
            adjusted_rand_score(&labels_true, &labels_pred)?,
            fowlkes_mallows_score(&labels_true, &labels_pred)?,
        ];

        for score in perfect_metrics {
            assert_relative_eq!(score, 1.0, epsilon = 1e-6);
        }

        // Distance-based metrics should be reasonable
        let silhouette = silhouette_score(&data, &labels_pred)?;
        assert!(silhouette >= 0.0 && silhouette <= 1.0);

        let ch_score = calinski_harabasz_score(&data, &labels_pred)?;
        assert!(ch_score > 0.0);

        let db_score = davies_bouldin_score(&data, &labels_pred)?;
        assert!(db_score >= 0.0);

        Ok(())
    }
}
