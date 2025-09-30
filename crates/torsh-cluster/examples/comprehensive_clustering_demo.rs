//! Comprehensive Clustering Algorithm Demonstration
//!
//! This example demonstrates all clustering algorithms available in torsh-cluster,
//! along with evaluation metrics and best practices for real-world usage.

use scirs2_core::random::prelude::*;
use std::collections::HashMap;
use torsh_cluster::{
    algorithms::{
        gaussian_mixture::CovarianceType,
        hierarchical::Linkage,
        incremental::{IncrementalClustering, OnlineKMeans},
        kmeans::KMeansAlgorithm,
        *,
    },
    evaluation::metrics::*,
    traits::*,
};
use torsh_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 ToRSh Comprehensive Clustering Algorithm Demonstration");
    println!("=========================================================");

    // Create synthetic datasets for demonstration
    let (blob_data, blob_labels) = create_blob_dataset()?;
    let (circle_data, circle_labels) = create_circle_dataset()?;
    let streaming_data = create_streaming_dataset()?;

    println!("\n📊 Dataset Information:");
    println!(
        "  • Blob dataset: {} samples, {} features",
        blob_data.shape().dims()[0],
        blob_data.shape().dims()[1]
    );
    println!(
        "  • Circle dataset: {} samples, {} features",
        circle_data.shape().dims()[0],
        circle_data.shape().dims()[1]
    );
    println!("  • Streaming dataset: {} batches", streaming_data.len());

    // Demonstrate all clustering algorithms
    demonstrate_kmeans(&blob_data, &blob_labels)?;
    demonstrate_gaussian_mixture(&blob_data, &blob_labels)?;
    demonstrate_spectral_clustering(&circle_data, &circle_labels)?;
    demonstrate_dbscan(&circle_data, &circle_labels)?;
    demonstrate_hierarchical_clustering(&blob_data, &blob_labels)?;
    demonstrate_online_clustering(&streaming_data)?;

    // Demonstrate evaluation metrics
    demonstrate_evaluation_metrics(&blob_data, &blob_labels)?;

    // Demonstrate optimal k selection
    demonstrate_optimal_k_selection(&blob_data)?;

    println!("\n✅ Comprehensive demonstration completed successfully!");
    println!("🔍 Check the individual algorithm outputs above for detailed results.");

    Ok(())
}

/// Demonstrate K-Means clustering with different variants
fn demonstrate_kmeans(
    data: &Tensor,
    true_labels: &Tensor,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 K-Means Clustering Demonstration");
    println!("==================================");

    let algorithms = [
        ("Lloyd's Algorithm", KMeansAlgorithm::Lloyd),
        ("Elkan's Algorithm", KMeansAlgorithm::Elkan),
        ("Mini-batch K-Means", KMeansAlgorithm::MiniBatch),
    ];

    for (name, algorithm) in &algorithms {
        println!("\n📈 Testing {}", name);

        let kmeans = KMeans::new(3)
            .algorithm(*algorithm)
            .max_iters(100)
            .tolerance(1e-4)
            .n_init(5)
            .random_state(42);

        let start = std::time::Instant::now();
        let result = kmeans.fit(data)?;
        let duration = start.elapsed();

        let silhouette = silhouette_score(data, &result.labels)?;
        let ari = adjusted_rand_score(true_labels, &result.labels)?;

        println!("  ✓ Inertia: {:.4}", result.inertia);
        println!("  ✓ Iterations: {}", result.n_iter);
        println!("  ✓ Converged: {}", result.converged());
        println!("  ✓ Silhouette Score: {:.4}", silhouette);
        println!("  ✓ Adjusted Rand Index: {:.4}", ari);
        println!("  ⏱️  Execution Time: {:?}", duration);
    }

    Ok(())
}

/// Demonstrate Gaussian Mixture Model clustering
fn demonstrate_gaussian_mixture(
    data: &Tensor,
    true_labels: &Tensor,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Gaussian Mixture Model Demonstration");
    println!("======================================");

    let covariance_types = [
        ("Diagonal", CovarianceType::Diag),
        ("Full", CovarianceType::Full),
        ("Spherical", CovarianceType::Spherical),
    ];

    for (name, cov_type) in &covariance_types {
        println!("\n📈 Testing {} Covariance", name);

        let gmm = GaussianMixture::new(3)
            .covariance_type(*cov_type)
            .max_iters(100)
            .tolerance(1e-6)
            .random_state(42);

        let start = std::time::Instant::now();
        let result = gmm.fit(data)?;
        let duration = start.elapsed();

        let silhouette = silhouette_score(data, &result.labels)?;
        let ari = adjusted_rand_score(true_labels, &result.labels)?;

        println!("  ✓ Log-likelihood: {:.4}", result.log_likelihood);
        println!("  ✓ AIC: {:.4}", result.aic);
        println!("  ✓ BIC: {:.4}", result.bic);
        println!("  ✓ Converged: {}", result.converged());
        println!("  ✓ Silhouette Score: {:.4}", silhouette);
        println!("  ✓ Adjusted Rand Index: {:.4}", ari);
        println!("  ⏱️  Execution Time: {:?}", duration);
    }

    Ok(())
}

/// Demonstrate Spectral Clustering
fn demonstrate_spectral_clustering(
    data: &Tensor,
    true_labels: &Tensor,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Spectral Clustering Demonstration");
    println!("===================================");

    let spectral = SpectralClustering::new(2).gamma(1.0).random_state(42);

    let start = std::time::Instant::now();
    let result = spectral.fit(data)?;
    let duration = start.elapsed();

    let silhouette = silhouette_score(data, &result.labels)?;
    let ari = adjusted_rand_score(true_labels, &result.labels)?;

    println!("  ✓ Converged: {}", result.converged());
    println!("  ✓ Eigenvalues computed: true");
    println!("  ✓ Silhouette Score: {:.4}", silhouette);
    println!("  ✓ Adjusted Rand Index: {:.4}", ari);
    println!("  ⏱️  Execution Time: {:?}", duration);

    Ok(())
}

/// Demonstrate DBSCAN clustering
fn demonstrate_dbscan(
    data: &Tensor,
    true_labels: &Tensor,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 DBSCAN Clustering Demonstration");
    println!("=================================");

    let dbscan = DBSCAN::new(0.5, 5).metric("euclidean");

    let start = std::time::Instant::now();
    let result = dbscan.fit(data)?;
    let duration = start.elapsed();

    let silhouette = silhouette_score(data, &result.labels)?;
    let ari = adjusted_rand_score(true_labels, &result.labels)?;

    println!("  ✓ Clusters found: {}", result.n_clusters);
    println!("  ✓ Core samples: {}", result.core_sample_indices.len());
    println!("  ✓ Noise points: {}", result.noise_points.len());
    println!("  ✓ Silhouette Score: {:.4}", silhouette);
    println!("  ✓ Adjusted Rand Index: {:.4}", ari);
    println!("  ⏱️  Execution Time: {:?}", duration);

    Ok(())
}

/// Demonstrate Hierarchical Clustering
fn demonstrate_hierarchical_clustering(
    data: &Tensor,
    true_labels: &Tensor,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Hierarchical Clustering Demonstration");
    println!("=======================================");

    let hierarchical = AgglomerativeClustering::new(3).linkage(Linkage::Ward);

    let start = std::time::Instant::now();
    let result = hierarchical.fit(data)?;
    let duration = start.elapsed();

    let silhouette = silhouette_score(data, &result.labels)?;
    let ari = adjusted_rand_score(true_labels, &result.labels)?;

    println!("  ✓ Clusters formed: {}", result.n_clusters());
    println!("  ✓ Silhouette Score: {:.4}", silhouette);
    println!("  ✓ Adjusted Rand Index: {:.4}", ari);
    println!("  ⏱️  Execution Time: {:?}", duration);

    Ok(())
}

/// Demonstrate Online/Incremental Clustering
fn demonstrate_online_clustering(
    streaming_data: &[Tensor],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Online K-Means Clustering Demonstration");
    println!("==========================================");

    let mut online_kmeans = OnlineKMeans::new(3)?
        .learning_rate(None) // Adaptive
        .drift_threshold(0.1)
        .random_state(42);

    let start = std::time::Instant::now();

    for (batch_idx, batch) in streaming_data.iter().enumerate() {
        online_kmeans.update_batch(batch)?;

        if batch_idx % 10 == 0 {
            let result = online_kmeans.get_current_result()?;
            let drift_detected = online_kmeans.detect_drift();

            println!(
                "  📊 Batch {}: {} points processed, drift: {}",
                batch_idx, result.n_points_seen, drift_detected
            );
        }
    }

    let duration = start.elapsed();
    let final_result = online_kmeans.get_current_result()?;

    println!("  ✓ Total points processed: {}", final_result.n_points_seen);
    println!(
        "  ✓ Final learning rate: {:.6}",
        final_result.current_learning_rate
    );
    println!("  ✓ Drift detected: {}", final_result.drift_detected);
    println!(
        "  ✓ Average intra-cluster distance: {:.4}",
        final_result.avg_intra_cluster_distance
    );
    println!("  ⏱️  Execution Time: {:?}", duration);

    Ok(())
}

/// Demonstrate comprehensive evaluation metrics
fn demonstrate_evaluation_metrics(
    data: &Tensor,
    true_labels: &Tensor,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Comprehensive Evaluation Metrics Demonstration");
    println!("================================================");

    // Generate predictions using K-means
    let kmeans = KMeans::new(3).random_state(42);
    let result = kmeans.fit(data)?;
    let pred_labels = &result.labels;

    println!("\n📊 Distance-based Metrics:");
    println!(
        "  • Silhouette Score: {:.4}",
        silhouette_score(data, pred_labels)?
    );
    println!(
        "  • Calinski-Harabasz Score: {:.4}",
        calinski_harabasz_score(data, pred_labels)?
    );
    println!(
        "  • Davies-Bouldin Score: {:.4}",
        davies_bouldin_score(data, pred_labels)?
    );
    println!("  • Dunn Index: {:.4}", dunn_index(data, pred_labels)?);
    println!(
        "  • Xie-Beni Index: {:.4}",
        xie_beni_index(data, pred_labels)?
    );

    println!("\n📊 Information-theoretic Metrics:");
    println!(
        "  • Normalized Mutual Information: {:.4}",
        normalized_mutual_info_score(true_labels, pred_labels)?
    );
    println!(
        "  • Adjusted Mutual Information: {:.4}",
        adjusted_mutual_info_score(true_labels, pred_labels)?
    );
    println!(
        "  • Homogeneity Score: {:.4}",
        homogeneity_score(true_labels, pred_labels)?
    );
    println!(
        "  • Completeness Score: {:.4}",
        completeness_score(true_labels, pred_labels)?
    );
    println!(
        "  • V-Measure Score: {:.4}",
        v_measure_score(true_labels, pred_labels)?
    );

    println!("\n📊 Set-based Metrics:");
    println!(
        "  • Adjusted Rand Index: {:.4}",
        adjusted_rand_score(true_labels, pred_labels)?
    );
    println!(
        "  • Fowlkes-Mallows Score: {:.4}",
        fowlkes_mallows_score(true_labels, pred_labels)?
    );

    Ok(())
}

/// Demonstrate optimal k selection using Gap Statistic
fn demonstrate_optimal_k_selection(data: &Tensor) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Optimal K Selection using Gap Statistic");
    println!("==========================================");

    let config = GapStatisticConfig {
        max_k: 8,
        n_refs: 10,
        random_state: Some(42),
        ..Default::default()
    };

    let mut gap_stat = GapStatistic::new(config);

    let start = std::time::Instant::now();
    let result = gap_stat.compute(data)?;
    let duration = start.elapsed();

    println!("  🎯 Optimal number of clusters: {}", result.optimal_k);
    println!("  📊 Gap values:");
    for (i, &gap) in result.gap_values.iter().enumerate() {
        let k = i + 1;
        let marker = if k == result.optimal_k {
            " ← Optimal"
        } else {
            ""
        };
        println!("     k={}: Gap={:.4}{}", k, gap, marker);
    }

    let summary = result.summary();
    println!("  📈 Summary: {:?}", summary);
    println!("  ⏱️  Execution Time: {:?}", duration);

    Ok(())
}

/// Create a synthetic blob dataset with known clusters
fn create_blob_dataset() -> Result<(Tensor, Tensor), Box<dyn std::error::Error>> {
    let mut data = Vec::new();
    let mut labels = Vec::new();

    // Cluster 0: around (0, 0)
    for _ in 0..50 {
        data.extend_from_slice(&[
            thread_rng().gen::<f32>() * 2.0 - 1.0,
            thread_rng().gen::<f32>() * 2.0 - 1.0,
        ]);
        labels.push(0.0);
    }

    // Cluster 1: around (5, 5)
    for _ in 0..50 {
        data.extend_from_slice(&[
            thread_rng().gen::<f32>() * 2.0 + 4.0,
            thread_rng().gen::<f32>() * 2.0 + 4.0,
        ]);
        labels.push(1.0);
    }

    // Cluster 2: around (-5, 5)
    for _ in 0..50 {
        data.extend_from_slice(&[
            thread_rng().gen::<f32>() * 2.0 - 6.0,
            thread_rng().gen::<f32>() * 2.0 + 4.0,
        ]);
        labels.push(2.0);
    }

    let data_tensor = Tensor::from_vec(data, &[150, 2])?;
    let labels_tensor = Tensor::from_vec(labels, &[150])?;

    Ok((data_tensor, labels_tensor))
}

/// Create a synthetic circle dataset
fn create_circle_dataset() -> Result<(Tensor, Tensor), Box<dyn std::error::Error>> {
    let mut data = Vec::new();
    let mut labels = Vec::new();

    // Inner circle
    for i in 0..50 {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / 50.0;
        let radius = 2.0 + thread_rng().gen::<f32>() * 0.5;
        data.extend_from_slice(&[radius * angle.cos(), radius * angle.sin()]);
        labels.push(0.0);
    }

    // Outer circle
    for i in 0..50 {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / 50.0;
        let radius = 5.0 + thread_rng().gen::<f32>() * 0.5;
        data.extend_from_slice(&[radius * angle.cos(), radius * angle.sin()]);
        labels.push(1.0);
    }

    let data_tensor = Tensor::from_vec(data, &[100, 2])?;
    let labels_tensor = Tensor::from_vec(labels, &[100])?;

    Ok((data_tensor, labels_tensor))
}

/// Create streaming data batches
fn create_streaming_dataset() -> Result<Vec<Tensor>, Box<dyn std::error::Error>> {
    let mut batches = Vec::new();

    for batch_idx in 0..50 {
        let mut batch_data = Vec::new();

        // Create 20 points per batch with some drift
        for _ in 0..20 {
            let drift = batch_idx as f32 * 0.1;
            batch_data.extend_from_slice(&[
                thread_rng().gen::<f32>() * 4.0 - 2.0 + drift,
                thread_rng().gen::<f32>() * 4.0 - 2.0 + drift,
            ]);
        }

        batches.push(Tensor::from_vec(batch_data, &[20, 2])?);
    }

    Ok(batches)
}
