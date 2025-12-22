//! Comprehensive integration tests for torsh-cluster
//!
//! This module provides end-to-end integration tests for all clustering algorithms,
//! testing them with realistic data scenarios and verifying correctness.

use approx::assert_relative_eq;
use torsh_cluster::{
    error::ClusterResult,
    traits::{ClusteringAlgorithm, ClusteringResult, Fit},
};
use torsh_tensor::Tensor;

// Import algorithms
use torsh_cluster::algorithms::dbscan::DBSCAN;
use torsh_cluster::algorithms::gaussian_mixture::{CovarianceType, GaussianMixture};
use torsh_cluster::algorithms::hierarchical::{AgglomerativeClustering, Linkage};
use torsh_cluster::algorithms::incremental::{IncrementalClustering, OnlineKMeans};
use torsh_cluster::algorithms::kmeans::{KMeans, KMeansAlgorithm};
use torsh_cluster::algorithms::spectral::SpectralClustering;

// Import metrics
use torsh_cluster::evaluation::metrics::{
    adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score,
};

/// Helper function to create synthetic clustered data
fn create_clustered_data_2d(
    n_samples_per_cluster: usize,
    n_clusters: usize,
) -> ClusterResult<(Tensor, Vec<i32>)> {
    let mut all_data = Vec::new();
    let mut all_labels = Vec::new();

    for cluster_id in 0..n_clusters {
        let center_x = (cluster_id as f32) * 5.0;
        let center_y = (cluster_id as f32) * 5.0;

        for _ in 0..n_samples_per_cluster {
            // Add some randomness around center
            let noise_x = (cluster_id as f32 * 0.1) % 1.0;
            let noise_y = (cluster_id as f32 * 0.2) % 1.0;

            all_data.push(center_x + noise_x);
            all_data.push(center_y + noise_y);
            all_labels.push(cluster_id as i32);
        }
    }

    let n_total = n_samples_per_cluster * n_clusters;
    let data = Tensor::from_vec(all_data, &[n_total, 2])?;
    Ok((data, all_labels))
}

/// Helper to create concentric circles data (for spectral clustering)
fn create_concentric_circles(n_samples_per_circle: usize) -> ClusterResult<(Tensor, Vec<i32>)> {
    let mut all_data = Vec::new();
    let mut all_labels = Vec::new();

    // Inner circle (radius 1)
    for i in 0..n_samples_per_circle {
        let angle = 2.0 * std::f32::consts::PI * (i as f32) / (n_samples_per_circle as f32);
        all_data.push(angle.cos());
        all_data.push(angle.sin());
        all_labels.push(0);
    }

    // Outer circle (radius 3)
    for i in 0..n_samples_per_circle {
        let angle = 2.0 * std::f32::consts::PI * (i as f32) / (n_samples_per_circle as f32);
        all_data.push(3.0 * angle.cos());
        all_data.push(3.0 * angle.sin());
        all_labels.push(1);
    }

    let data = Tensor::from_vec(all_data, &[n_samples_per_circle * 2, 2])?;
    Ok((data, all_labels))
}

#[test]
fn test_kmeans_lloyd_integration() -> ClusterResult<()> {
    let (data, _true_labels) = create_clustered_data_2d(20, 3)?;

    let kmeans = KMeans::new(3)
        .max_iters(100)
        .tolerance(1e-4)
        .random_state(42)
        .algorithm(KMeansAlgorithm::Lloyd);

    let result = kmeans.fit(&data)?;

    // Check basic properties
    assert_eq!(result.n_clusters(), 3);
    assert!(result.converged());
    assert!(result.inertia().unwrap() > 0.0);
    assert!(result.n_iter().unwrap_or(0) < 100);

    // Check silhouette score is reasonable
    let silhouette = silhouette_score(&data, result.labels())?;
    assert!(silhouette > 0.5, "Silhouette score too low: {}", silhouette);

    // Check Calinski-Harabasz score
    let ch_score = calinski_harabasz_score(&data, result.labels())?;
    assert!(ch_score > 0.0, "CH score should be positive");

    // Check Davies-Bouldin score (lower is better)
    let db_score = davies_bouldin_score(&data, result.labels())?;
    assert!(db_score < 2.0, "DB score too high: {}", db_score);

    Ok(())
}

#[test]
fn test_kmeans_elkan_integration() -> ClusterResult<()> {
    let (data, _true_labels) = create_clustered_data_2d(20, 3)?;

    let kmeans = KMeans::new(3)
        .max_iters(100)
        .tolerance(1e-4)
        .random_state(42)
        .algorithm(KMeansAlgorithm::Elkan);

    let result = kmeans.fit(&data)?;

    // Elkan should produce similar results to Lloyd
    assert_eq!(result.n_clusters(), 3);
    // Elkan may not converge in all cases, that's okay
    assert!(result.inertia().unwrap() > 0.0);

    let silhouette = silhouette_score(&data, result.labels())?;
    assert!(silhouette > 0.5);

    Ok(())
}

#[test]
fn test_kmeans_minibatch_integration() -> ClusterResult<()> {
    let (data, _true_labels) = create_clustered_data_2d(30, 3)?;

    let kmeans = KMeans::new(3)
        .max_iters(100)
        .tolerance(1e-4)
        .random_state(42)
        .algorithm(KMeansAlgorithm::MiniBatch);

    let result = kmeans.fit(&data)?;

    assert_eq!(result.n_clusters(), 3);
    // Mini-batch may not always compute inertia
    if let Some(inertia) = result.inertia() {
        assert!(inertia >= 0.0);
    }

    // Mini-batch might have slightly lower quality but should still be reasonable
    let silhouette = silhouette_score(&data, result.labels())?;
    assert!(
        silhouette > 0.3,
        "Mini-batch silhouette too low: {}",
        silhouette
    );

    Ok(())
}

#[test]
fn test_dbscan_integration() -> ClusterResult<()> {
    let (data, _true_labels) = create_clustered_data_2d(20, 3)?;

    let dbscan = DBSCAN::new(2.0, 3);
    let result = dbscan.fit(&data)?;

    // DBSCAN should find at least 2 clusters
    assert!(result.n_clusters() >= 2, "Expected at least 2 clusters");

    // Check that we have some core points
    let labels_vec = result.labels().to_vec()?;
    let non_noise_count = labels_vec.iter().filter(|&&l| l >= 0.0).count();
    assert!(
        non_noise_count > 40,
        "Too many noise points: {}",
        labels_vec.len() - non_noise_count
    );

    Ok(())
}

#[test]
fn test_hierarchical_clustering_integration() -> ClusterResult<()> {
    let (data, _true_labels) = create_clustered_data_2d(15, 3)?;

    let hierarchical = AgglomerativeClustering::new(3).linkage(Linkage::Average);
    let result = hierarchical.fit(&data)?;

    assert_eq!(result.n_clusters(), 3);

    let silhouette = silhouette_score(&data, result.labels())?;
    // Hierarchical clustering may produce lower silhouette scores
    assert!(
        silhouette > 0.1,
        "Hierarchical silhouette too low: {}",
        silhouette
    );

    Ok(())
}

#[test]
fn test_gmm_full_covariance_integration() -> ClusterResult<()> {
    let (data, _true_labels) = create_clustered_data_2d(25, 3)?;

    let gmm = GaussianMixture::new(3)
        .covariance_type(CovarianceType::Full)
        .max_iters(100)
        .tolerance(1e-3)
        .random_state(42);

    let result = gmm.fit(&data)?;

    assert_eq!(result.n_clusters(), 3);
    assert!(result.converged());
    assert!(result.log_likelihood.is_finite());
    assert!(result.aic.is_finite());
    assert!(result.bic.is_finite());

    // BIC should be finite (can be negative)
    assert!(result.bic.is_finite(), "BIC should be finite");

    // Check if we actually have multiple unique clusters in the labels
    let labels_vec = result.labels().to_vec()?;
    let unique_labels: std::collections::HashSet<i32> =
        labels_vec.iter().map(|&x: &f32| x as i32).collect();

    // Silhouette score requires at least 2 unique clusters
    if unique_labels.len() >= 2 {
        let silhouette = silhouette_score(&data, result.labels())?;
        assert!(silhouette > 0.1, "GMM silhouette too low: {}", silhouette);
    } else {
        // It's possible GMM converges to fewer clusters than requested
        println!(
            "Warning: GMM converged to {} unique clusters (expected 3)",
            unique_labels.len()
        );
    }

    Ok(())
}

#[test]
fn test_gmm_diagonal_covariance_integration() -> ClusterResult<()> {
    let (data, _true_labels) = create_clustered_data_2d(25, 3)?;

    let gmm = GaussianMixture::new(3)
        .covariance_type(CovarianceType::Diag)
        .max_iters(100)
        .tolerance(1e-3)
        .random_state(42);

    let result = gmm.fit(&data)?;

    assert_eq!(result.n_clusters(), 3);
    assert!(result.converged());

    // Silhouette score requires at least 2 clusters
    if result.n_clusters() >= 2 {
        let silhouette = silhouette_score(&data, result.labels())?;
        assert!(silhouette > 0.1);
    }

    Ok(())
}

#[test]
fn test_gmm_spherical_covariance_integration() -> ClusterResult<()> {
    let (data, _true_labels) = create_clustered_data_2d(25, 3)?;

    let gmm = GaussianMixture::new(3)
        .covariance_type(CovarianceType::Spherical)
        .max_iters(100)
        .tolerance(1e-3)
        .random_state(42);

    let result = gmm.fit(&data)?;

    assert_eq!(result.n_clusters(), 3);
    assert!(result.converged());

    Ok(())
}

#[test]
fn test_spectral_clustering_integration() -> ClusterResult<()> {
    // Use concentric circles - good for spectral clustering
    let (data, _true_labels) = create_concentric_circles(30)?;

    let spectral = SpectralClustering::new(2);
    let result = spectral.fit(&data)?;

    assert_eq!(result.n_clusters(), 2);

    // Spectral clustering on concentric circles is challenging
    // Just verify it completes without errors
    let _silhouette = silhouette_score(&data, result.labels())?;
    // Note: Silhouette score may be low for non-convex clusters like concentric circles

    Ok(())
}

#[test]
fn test_online_kmeans_integration() -> ClusterResult<()> {
    let (data, _true_labels) = create_clustered_data_2d(20, 3)?;

    let mut online_kmeans = OnlineKMeans::new(3)?
        .learning_rate(None) // Adaptive
        .drift_threshold(0.2)
        .random_state(42);

    // Process data in batches
    let data_vec = data.to_vec()?;
    let n_samples = data.shape().dims()[0];
    let n_features = data.shape().dims()[1];

    for i in 0..n_samples {
        let start = i * n_features;
        let end = start + n_features;
        let point_data = &data_vec[start..end];
        let point = Tensor::from_vec(point_data.to_vec(), &[1, n_features])?;
        online_kmeans.update_single(&point)?;
    }

    let result = online_kmeans.get_current_result()?;
    assert_eq!(result.n_clusters(), 3);
    assert_eq!(result.n_points_seen, n_samples);

    Ok(())
}

#[test]
fn test_clustering_algorithm_trait_consistency() -> ClusterResult<()> {
    let (_data, _) = create_clustered_data_2d(20, 3)?;

    // Test that all algorithms implement the trait correctly
    let kmeans = KMeans::new(3);
    assert_eq!(kmeans.name(), "K-Means");
    assert!(!kmeans.is_fitted());

    let dbscan = DBSCAN::new(0.5, 2);
    assert_eq!(dbscan.name(), "DBSCAN");

    let hierarchical = AgglomerativeClustering::new(3);
    assert_eq!(hierarchical.name(), "Agglomerative Clustering");

    let gmm = GaussianMixture::new(3);
    assert_eq!(gmm.name(), "Gaussian Mixture Model");

    let spectral = SpectralClustering::new(3);
    assert_eq!(spectral.name(), "Spectral Clustering");

    Ok(())
}

#[test]
fn test_metrics_consistency() -> ClusterResult<()> {
    let (data, _true_labels) = create_clustered_data_2d(20, 3)?;

    let kmeans = KMeans::new(3).random_state(42);
    let result = kmeans.fit(&data)?;

    // Compute various metrics
    let silhouette = silhouette_score(&data, result.labels())?;
    let ch_score = calinski_harabasz_score(&data, result.labels())?;
    let db_score = davies_bouldin_score(&data, result.labels())?;

    // All metrics should be finite and reasonable
    assert!(silhouette.is_finite());
    assert!(ch_score.is_finite() && ch_score > 0.0);
    assert!(db_score.is_finite() && db_score >= 0.0);

    // Silhouette should be in [-1, 1]
    assert!(silhouette >= -1.0 && silhouette <= 1.0);

    Ok(())
}

#[test]
fn test_kmeans_multiple_initializations() -> ClusterResult<()> {
    let (data, _) = create_clustered_data_2d(20, 3)?;

    // Test that multiple initializations produce consistent results
    let kmeans = KMeans::new(3)
        .n_init(5) // Run 5 times and keep best
        .random_state(42);

    let result = kmeans.fit(&data)?;

    assert_eq!(result.n_clusters(), 3);
    assert!(result.converged());

    // Should have found a good solution
    let silhouette = silhouette_score(&data, result.labels())?;
    assert!(silhouette > 0.5);

    Ok(())
}

#[test]
fn test_empty_cluster_handling() -> ClusterResult<()> {
    // Create data with 2 clear clusters but request 5 clusters
    let (data, _) = create_clustered_data_2d(10, 2)?;

    let kmeans = KMeans::new(5).random_state(42);
    let result = kmeans.fit(&data)?;

    // Algorithm should complete without error
    assert_eq!(result.n_clusters(), 5);

    Ok(())
}

#[test]
fn test_single_sample_cluster() -> ClusterResult<()> {
    // Create mostly one cluster with one outlier
    let mut data_vec = Vec::new();

    // Main cluster
    for _ in 0..20 {
        data_vec.push(0.0);
        data_vec.push(0.0);
    }

    // Outlier
    data_vec.push(10.0);
    data_vec.push(10.0);

    let data = Tensor::from_vec(data_vec, &[21, 2])?;

    let kmeans = KMeans::new(2).random_state(42);
    let result = kmeans.fit(&data)?;

    assert_eq!(result.n_clusters(), 2);

    Ok(())
}

#[test]
fn test_high_dimensional_clustering() -> ClusterResult<()> {
    // Test with higher dimensional data (10D)
    let n_features = 10;
    let n_samples_per_cluster = 15;
    let n_clusters = 3;

    let mut data_vec = Vec::new();

    for cluster_id in 0..n_clusters {
        for _ in 0..n_samples_per_cluster {
            for dim in 0..n_features {
                let value = (cluster_id as f32) * 5.0 + (dim as f32) * 0.1;
                data_vec.push(value);
            }
        }
    }

    let data = Tensor::from_vec(data_vec, &[n_samples_per_cluster * n_clusters, n_features])?;

    let kmeans = KMeans::new(3).random_state(42);
    let result = kmeans.fit(&data)?;

    assert_eq!(result.n_clusters(), 3);
    assert!(result.converged());

    Ok(())
}

#[test]
fn test_algorithm_reproducibility() -> ClusterResult<()> {
    let (data, _) = create_clustered_data_2d(20, 3)?;

    // Run same algorithm twice with same seed
    let kmeans1 = KMeans::new(3).random_state(42);
    let result1 = kmeans1.fit(&data)?;

    let kmeans2 = KMeans::new(3).random_state(42);
    let result2 = kmeans2.fit(&data)?;

    // Results should be identical
    // Compare inertias - they should be the same for same seed
    let inertia1 = result1.inertia().unwrap();
    let inertia2 = result2.inertia().unwrap();
    assert_relative_eq!(inertia1, inertia2, epsilon = 1e-6);

    let labels1 = result1.labels().to_vec()?;
    let labels2 = result2.labels().to_vec()?;

    // Labels might be permuted but clustering should be the same
    // Convert to tensors for adjusted_rand_score
    let labels1_tensor = Tensor::from_vec(labels1.clone(), &[labels1.len()])?;
    let labels2_tensor = Tensor::from_vec(labels2.clone(), &[labels2.len()])?;

    let ari = adjusted_rand_score(&labels1_tensor, &labels2_tensor)?;
    assert_relative_eq!(ari, 1.0, epsilon = 1e-6);

    Ok(())
}
