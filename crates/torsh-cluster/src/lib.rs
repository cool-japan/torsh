//! Unsupervised learning and clustering algorithms for ToRSh
//!
//! This crate provides PyTorch-compatible clustering algorithms built on top of
//! the SciRS2 ecosystem, offering high-performance implementations of popular
//! clustering methods with extensive customization options.
//!
//! # Key Features
//!
//! - **K-Means Clustering**: Classic centroid-based clustering with multiple initialization strategies
//! - **Hierarchical Clustering**: Agglomerative and divisive clustering methods
//! - **DBSCAN**: Density-based clustering for arbitrary-shaped clusters
//! - **Gaussian Mixture Models**: Probabilistic clustering with EM algorithm
//! - **Spectral Clustering**: Graph-based clustering using eigendecomposition
//! - **Evaluation Metrics**: Silhouette score, adjusted rand index, and more
//!
//! # SciRS2 Integration
//!
//! All clustering algorithms are built on `scirs2-cluster` foundation:
//! - Leverages scirs2-core for random number generation and array operations
//! - Uses scirs2-stats for statistical computations
//! - Integrates with scirs2-metrics for clustering evaluation
//! - Employs scirs2-linalg for linear algebra operations
//!
//! # Example Usage
//!
//! ```rust
//! use torsh_cluster::prelude::*;
//! use torsh_tensor::Tensor;
//!
//! // Create sample data
//! let data = Tensor::randn(&[100, 2])?;
//!
//! // Perform K-means clustering
//! let kmeans = KMeans::new(3)
//!     .max_iters(100)
//!     .tolerance(1e-4);
//!
//! let result = kmeans.fit(&data)?;
//! println!("Cluster centers: {:?}", result.centroids);
//! println!("Labels: {:?}", result.labels);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod algorithms;
pub mod error;
pub mod evaluation;
pub mod initialization;
pub mod traits;
pub mod utils;

// Re-export core clustering algorithms
pub use algorithms::{
    dbscan::{DBSCANConfig, DBSCANResult, DBSCAN},
    gaussian_mixture::{GMConfig, GMResult, GaussianMixture},
    hierarchical::{AgglomerativeClustering, HierarchicalResult, Linkage},
    kmeans::{InitMethod, KMeans, KMeansConfig, KMeansResult},
    spectral::{SpectralClustering, SpectralConfig, SpectralResult},
};

// Re-export evaluation metrics
pub use evaluation::{
    metrics::{
        adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score,
        davies_bouldin_score, fowlkes_mallows_score, homogeneity_score,
        normalized_mutual_info_score, silhouette_score, v_measure_score,
    },
    ClusteringMetric, EvaluationResult,
};

// Re-export initialization methods
pub use initialization::{
    forgy::Forgy, kmeans_plus_plus::KMeansPlusPlus, random_partition::RandomPartition,
    InitializationStrategy,
};

// Re-export traits
pub use traits::{ClusteringAlgorithm, ClusteringResult, Fit, FitPredict, Transform};

// Re-export utilities
pub use utils::{
    distance::{cosine_distance, euclidean_distance, manhattan_distance, DistanceMetric},
    preprocessing::{normalize_features, standardize_features, PreprocessingMethod},
    validation::{validate_cluster_input, validate_n_clusters, ClusterValidation},
};

// Re-export error types
pub use error::{ClusterError, ClusterResult};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

/// Prelude module for convenient imports
pub mod prelude {

    pub use crate::algorithms::{
        dbscan::{DBSCANConfig, DBSCAN},
        gaussian_mixture::{GMConfig, GaussianMixture},
        hierarchical::{AgglomerativeClustering, Linkage},
        kmeans::{InitMethod, KMeans, KMeansConfig},
        spectral::{SpectralClustering, SpectralConfig},
    };

    pub use crate::evaluation::{
        metrics::{adjusted_rand_score, silhouette_score},
        ClusteringMetric,
    };

    pub use crate::initialization::{
        Forgy, InitializationStrategy, KMeansPlusPlus, RandomPartition,
    };

    pub use crate::traits::{ClusteringAlgorithm, ClusteringResult, Fit, FitPredict, Transform};

    pub use crate::utils::{
        distance::{cosine_distance, euclidean_distance, DistanceMetric},
        preprocessing::{normalize_features, standardize_features},
    };

    pub use crate::error::{ClusterError, ClusterResult};
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::Tensor;

    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert_eq!(VERSION_MAJOR, 0);
        assert_eq!(VERSION_MINOR, 1);
        assert_eq!(VERSION_PATCH, 0);
    }

    #[test]
    fn test_dbscan_basic() -> Result<(), Box<dyn std::error::Error>> {
        // Simple 2D dataset with clear clusters
        let data = Tensor::from_vec(
            vec![
                // Cluster 1 (around origin)
                0.0, 0.0, 0.1, 0.1, 0.0, 0.2, 0.2, 0.0, // Cluster 2 (around (5,5))
                5.0, 5.0, 5.1, 5.1, 5.0, 5.2, 5.2, 5.0,
                // Noise point (far from both clusters)
                10.0, 10.0,
            ],
            &[9, 2],
        )?;

        let dbscan = DBSCAN::new(0.5, 2);
        let result = dbscan.fit(&data)?;

        // Should find 2 clusters and 1 noise point
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.noise_points.len(), 1);
        assert!(!result.core_sample_indices.is_empty());

        Ok(())
    }

    #[test]
    fn test_hierarchical_basic() -> Result<(), Box<dyn std::error::Error>> {
        // Simple 2D dataset
        let data = Tensor::from_vec(vec![0.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1], &[4, 2])?;

        let hierarchical = AgglomerativeClustering::new(2);
        let result = hierarchical.fit(&data)?;

        // Should find exactly 2 clusters
        assert_eq!(result.n_clusters, 2);

        // Check labels are valid (should be 0 and 1)
        let labels_vec = result.labels.to_vec()?;
        let unique_labels: std::collections::HashSet<i32> =
            labels_vec.iter().map(|&x| x as i32).collect();
        assert_eq!(unique_labels.len(), 2);

        Ok(())
    }

    #[test]
    fn test_silhouette_score_basic() -> Result<(), Box<dyn std::error::Error>> {
        // Simple well-separated clusters
        let data = Tensor::from_vec(
            vec![
                // Cluster 0
                0.0, 0.0, 0.1, 0.1, // Cluster 1
                10.0, 10.0, 10.1, 10.1,
            ],
            &[4, 2],
        )?;

        let labels = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[4])?;

        let score = silhouette_score(&data, &labels)?;

        // Well-separated clusters should have high silhouette score (close to 1.0)
        assert!(
            score > 0.5,
            "Silhouette score should be positive for well-separated clusters"
        );

        Ok(())
    }

    #[test]
    fn test_adjusted_rand_score_perfect() -> Result<(), Box<dyn std::error::Error>> {
        // Perfect clustering match
        let labels_true = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[4])?;
        let labels_pred = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[4])?;

        let score = adjusted_rand_score(&labels_true, &labels_pred)?;

        // Perfect match should give score of 1.0
        assert_relative_eq!(score, 1.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_calinski_harabasz_score() -> Result<(), Box<dyn std::error::Error>> {
        // Well-separated clusters should have high CH score
        let data = Tensor::from_vec(
            vec![
                // Cluster 0
                0.0, 0.0, 0.1, 0.1, // Cluster 1
                5.0, 5.0, 5.1, 5.1,
            ],
            &[4, 2],
        )?;

        let labels = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[4])?;

        let score = calinski_harabasz_score(&data, &labels)?;

        // Well-separated clusters should have high CH score
        assert!(
            score > 1.0,
            "Calinski-Harabasz score should be > 1 for well-separated clusters"
        );

        Ok(())
    }

    #[test]
    fn test_davies_bouldin_score() -> Result<(), Box<dyn std::error::Error>> {
        // Well-separated clusters should have low DB score
        let data = Tensor::from_vec(
            vec![
                // Cluster 0
                0.0, 0.0, 0.1, 0.1, // Cluster 1
                10.0, 10.0, 10.1, 10.1,
            ],
            &[4, 2],
        )?;

        let labels = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[4])?;

        let score = davies_bouldin_score(&data, &labels)?;

        // Well-separated clusters should have low DB score (closer to 0 is better)
        assert!(score >= 0.0, "Davies-Bouldin score should be non-negative");
        assert!(
            score < 5.0,
            "Davies-Bouldin score should be reasonable for well-separated clusters"
        );

        Ok(())
    }

    #[test]
    fn test_gmm_basic() -> Result<(), Box<dyn std::error::Error>> {
        use algorithms::gaussian_mixture::{CovarianceType, GaussianMixture};

        // Simple 2D dataset with clear clusters
        let data = Tensor::from_vec(
            vec![
                // Cluster 0 (around origin)
                0.0, 0.0, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, // Cluster 1 (around (5,5))
                5.0, 5.0, 5.1, 5.1, 4.9, 5.1, 5.1, 4.9,
            ],
            &[8, 2],
        )?;

        let gmm = GaussianMixture::new(2)
            .covariance_type(CovarianceType::Diag)
            .max_iters(50)
            .tolerance(1e-3);

        let result = gmm.fit(&data)?;

        // Should find exactly 2 clusters
        assert_eq!(result.n_clusters(), 2);

        // Should have proper shapes
        assert_eq!(result.means.shape().dims(), &[2, 2]); // 2 components, 2 features
        assert_eq!(result.weights.shape().dims(), &[2]); // 2 components
        assert_eq!(result.labels.shape().dims(), &[8]); // 8 samples
        assert_eq!(result.responsibilities.shape().dims(), &[8, 2]); // 8 samples, 2 components

        // Log-likelihood should be reasonable (not negative infinity)
        assert!(result.log_likelihood > f64::NEG_INFINITY);

        // AIC and BIC should be finite
        assert!(result.aic.is_finite());
        assert!(result.bic.is_finite());

        // Number of iterations should be reasonable
        assert!(result.n_iter <= 50);

        println!(
            "GMM test passed - log_likelihood: {}, AIC: {}, BIC: {}, converged: {}",
            result.log_likelihood, result.aic, result.bic, result.converged
        );

        Ok(())
    }

    #[test]
    fn test_spectral_clustering_basic() -> Result<(), Box<dyn std::error::Error>> {
        use algorithms::spectral::{AffinityType, SpectralClustering};

        // Simple 2D dataset with clear clusters
        let data = Tensor::from_vec(
            vec![
                // Cluster 0 (around origin)
                0.0, 0.0, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, // Cluster 1 (around (3,3))
                3.0, 3.0, 3.1, 3.1, 2.9, 3.1, 3.1, 2.9,
            ],
            &[8, 2],
        )?;

        let spectral = SpectralClustering::new(2)
            .affinity(AffinityType::Rbf)
            .gamma(1.0);

        let result = spectral.fit(&data)?;

        // Should find exactly 2 clusters
        assert_eq!(result.n_clusters(), 2);

        // Should have proper shapes
        assert_eq!(result.labels.shape().dims(), &[8]); // 8 samples
        assert_eq!(result.affinity_matrix.shape().dims(), &[8, 8]); // 8x8 affinity matrix
        assert_eq!(result.embedding.shape().dims(), &[8, 2]); // 8 samples, 2 clusters embedding
        assert_eq!(result.eigenvalues.shape().dims(), &[8]); // 8 eigenvalues

        // Should have successfully created embedding
        assert!(result.embedding_success);

        // K-means iterations should be reasonable
        assert!(result.kmeans_iterations <= 100);

        println!(
            "Spectral clustering test passed - embedding_success: {}, kmeans_iterations: {}",
            result.embedding_success, result.kmeans_iterations
        );

        Ok(())
    }
}
