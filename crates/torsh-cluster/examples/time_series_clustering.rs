//! Time-Series Clustering Example
//!
//! This example demonstrates how to cluster time-series data using various algorithms.
//! We'll show different approaches:
//! 1. Feature-based clustering (extract features from time series)
//! 2. Direct clustering on time series windows
//! 3. Online clustering for streaming time series data
//!
//! Run with: cargo run --example time_series_clustering

use torsh_cluster::{
    algorithms::{
        hierarchical::{AgglomerativeClustering, Linkage},
        incremental::{IncrementalClustering, OnlineKMeans},
        kmeans::KMeans,
    },
    evaluation::metrics::silhouette_score,
    traits::{ClusteringResult, Fit},
    utils::preprocessing::standardize_features,
};
use torsh_tensor::Tensor;

/// Generate synthetic time series data with different patterns
fn generate_time_series_data() -> (Vec<Vec<f32>>, Vec<usize>) {
    let n_series = 150;
    let series_length = 100;
    let mut all_series = Vec::new();
    let mut labels = Vec::new();

    println!("Generating synthetic time series data...");

    // Pattern 1: Sine wave
    for i in 0..50 {
        let mut series = Vec::new();
        let freq = 0.1 + (i as f32 * 0.01);
        let phase = (i as f32) * 0.1;
        for t in 0..series_length {
            let value = (2.0 * std::f32::consts::PI * freq * t as f32 + phase).sin();
            series.push(value);
        }
        all_series.push(series);
        labels.push(0);
    }

    // Pattern 2: Exponential growth/decay
    for i in 0..50 {
        let mut series = Vec::new();
        let rate = if i % 2 == 0 { 0.02 } else { -0.02 };
        for t in 0..series_length {
            let value = (rate * t as f32).exp();
            series.push(value);
        }
        all_series.push(series);
        labels.push(1);
    }

    // Pattern 3: Random walk
    for i in 0..50 {
        let mut series = Vec::new();
        let mut value = 0.0;
        let volatility = 0.1 + (i as f32 * 0.01);
        for t in 0..series_length {
            let noise = ((t * 7 + i * 13) % 100) as f32 / 50.0 - 1.0; // Deterministic "random"
            value += noise * volatility;
            series.push(value);
        }
        all_series.push(series);
        labels.push(2);
    }

    println!(
        "Generated {} time series with {} points each",
        n_series, series_length
    );
    println!("  - 50 sine waves (cluster 0)");
    println!("  - 50 exponential patterns (cluster 1)");
    println!("  - 50 random walks (cluster 2)");

    (all_series, labels)
}

/// Extract statistical features from time series
fn extract_features(series: &[f32]) -> Vec<f32> {
    let n = series.len() as f32;

    // Mean
    let mean = series.iter().sum::<f32>() / n;

    // Standard deviation
    let variance = series.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();

    // Min and max
    let min = series.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = series.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Trend (linear regression slope approximation)
    let mut sum_xy = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_x2 = 0.0;
    for (i, &y) in series.iter().enumerate() {
        let x = i as f32;
        sum_xy += x * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
    }
    let trend = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

    // First difference statistics
    let mut diffs = Vec::new();
    for i in 1..series.len() {
        diffs.push(series[i] - series[i - 1]);
    }
    let diff_mean = diffs.iter().sum::<f32>() / (n - 1.0);
    let diff_std = (diffs.iter().map(|x| (x - diff_mean).powi(2)).sum::<f32>() / (n - 1.0)).sqrt();

    // Autocorrelation at lag 1
    let mut autocorr = 0.0;
    for i in 1..series.len() {
        autocorr += (series[i] - mean) * (series[i - 1] - mean);
    }
    autocorr /= (n - 1.0) * variance;

    vec![mean, std, min, max, trend, diff_mean, diff_std, autocorr]
}

/// Example 1: Feature-based clustering
fn example_feature_based_clustering(series_data: &[Vec<f32>], true_labels: &[usize]) {
    println!("\n{}", "=".repeat(60));
    println!("Example 1: Feature-Based Clustering");
    println!("{}", "=".repeat(60));

    // Extract features from all time series
    println!("\nExtracting features from time series...");
    let mut feature_data = Vec::new();
    for series in series_data {
        let features = extract_features(series);
        feature_data.extend(features);
    }

    let n_series = series_data.len();
    let n_features = 8;

    // Create tensor and standardize
    let features_tensor =
        Tensor::from_vec(feature_data, &[n_series, n_features]).expect("Failed to create tensor");

    println!("Standardizing features...");
    let standardized = standardize_features(&features_tensor).expect("Failed to standardize");

    // K-Means clustering
    println!("\nApplying K-Means clustering (k=3)...");
    let kmeans = KMeans::new(3).max_iters(100).random_state(42);

    let result = kmeans.fit(&standardized).expect("K-Means failed");

    println!("K-Means Results:");
    println!("  - Converged: {}", result.converged);
    println!("  - Iterations: {}", result.n_iter);
    println!("  - Inertia: {:.4}", result.inertia);

    // Evaluate clustering quality
    let silhouette = silhouette_score(&standardized, result.labels())
        .expect("Failed to compute silhouette score");
    println!("  - Silhouette Score: {:.4}", silhouette);

    // Compare with true labels
    let predicted_labels = result.labels.to_vec().expect("Failed to get labels");
    let mut confusion_matrix = vec![vec![0; 3]; 3];
    for (i, &pred) in predicted_labels.iter().enumerate() {
        let true_label = true_labels[i];
        confusion_matrix[true_label][pred as usize] += 1;
    }

    println!("\nConfusion Matrix (rows=true, cols=predicted):");
    for (i, row) in confusion_matrix.iter().enumerate() {
        println!("  Cluster {}: {:?}", i, row);
    }
}

/// Example 2: Hierarchical clustering for dendrogram
fn example_hierarchical_clustering(series_data: &[Vec<f32>]) {
    println!("\n{}", "=".repeat(60));
    println!("Example 2: Hierarchical Clustering");
    println!("{}", "=".repeat(60));

    // Extract features
    let mut feature_data = Vec::new();
    for series in series_data {
        let features = extract_features(series);
        feature_data.extend(features);
    }

    let n_series = series_data.len();
    let n_features = 8;

    let features_tensor =
        Tensor::from_vec(feature_data, &[n_series, n_features]).expect("Failed to create tensor");
    let standardized = standardize_features(&features_tensor).expect("Failed to standardize");

    // Try different linkage methods
    let linkages = [
        (Linkage::Average, "Average"),
        (Linkage::Complete, "Complete"),
        (Linkage::Ward, "Ward"),
    ];

    println!("\nComparing different linkage methods:");
    for (linkage, name) in linkages.iter() {
        let hierarchical = AgglomerativeClustering::new(3).linkage(*linkage);

        let result = hierarchical
            .fit(&standardized)
            .expect("Hierarchical clustering failed");

        let silhouette = silhouette_score(&standardized, result.labels())
            .expect("Failed to compute silhouette score");

        println!("  - {}: Silhouette = {:.4}", name, silhouette);
    }
}

/// Example 3: Online clustering for streaming time series
fn example_online_clustering() {
    println!("\n{}", "=".repeat(60));
    println!("Example 3: Online Clustering for Streaming Data");
    println!("{}", "=".repeat(60));

    println!("\nSimulating streaming time series data...");

    // Initialize online K-means
    let mut online_kmeans = OnlineKMeans::new(3)
        .expect("Failed to create online K-means")
        .drift_threshold(0.15);

    // Simulate streaming data
    let window_size = 20; // Use last 20 points as features
    let mut stream_count = 0;

    // Generate a continuous stream
    for batch_id in 0..10 {
        println!("\nProcessing batch {}...", batch_id + 1);

        // Generate batch of time series windows
        let mut batch_data = Vec::new();
        for _ in 0..10 {
            // Generate a short time series segment
            let pattern = (batch_id % 3) as f32;
            for t in 0..window_size {
                let value = match pattern as usize {
                    0 => (0.2 * t as f32).sin(),
                    1 => (0.02 * t as f32).exp(),
                    _ => t as f32 * 0.1 - 1.0,
                };
                batch_data.push(value);
            }
            stream_count += 1;
        }

        let batch_tensor =
            Tensor::from_vec(batch_data, &[10, window_size]).expect("Failed to create tensor");

        // Update online clustering
        online_kmeans
            .update_batch(&batch_tensor)
            .expect("Failed to update online K-means");

        // Check for concept drift
        if online_kmeans.detect_drift() {
            println!("  ⚠️  Concept drift detected!");
        }

        let result = online_kmeans
            .get_current_result()
            .expect("Failed to get result");
        println!("  - Points processed: {}", result.n_points_seen);
        println!("  - Learning rate: {:.6}", result.current_learning_rate);
        println!(
            "  - Avg intra-cluster distance: {:.4}",
            result.avg_intra_cluster_distance
        );
    }

    println!("\nOnline clustering complete!");
    println!("  - Total points processed: {}", stream_count);
}

/// Example 4: Sliding window approach
fn example_sliding_window_clustering(series_data: &[Vec<f32>]) {
    println!("\n{}", "=".repeat(60));
    println!("Example 4: Sliding Window Clustering");
    println!("{}", "=".repeat(60));

    println!("\nExtracting sliding windows from time series...");

    let window_size = 20;
    let stride = 5;
    let mut windows = Vec::new();

    // Extract windows from first few series
    for series in series_data.iter().take(30) {
        for i in (0..series.len() - window_size).step_by(stride) {
            windows.extend_from_slice(&series[i..i + window_size]);
        }
    }

    let n_windows = windows.len() / window_size;
    println!("Extracted {} windows of size {}", n_windows, window_size);

    let windows_tensor =
        Tensor::from_vec(windows, &[n_windows, window_size]).expect("Failed to create tensor");
    let standardized = standardize_features(&windows_tensor).expect("Failed to standardize");

    // Cluster windows
    println!("\nClustering windows with K-Means (k=5)...");
    let kmeans = KMeans::new(5).max_iters(50).random_state(42);

    let result = kmeans.fit(&standardized).expect("K-Means failed");

    let silhouette = silhouette_score(&standardized, result.labels())
        .expect("Failed to compute silhouette score");

    println!("Results:");
    println!("  - Converged: {}", result.converged);
    println!("  - Silhouette Score: {:.4}", silhouette);

    // Count windows per cluster
    let labels = result.labels.to_vec().expect("Failed to get labels");
    let mut cluster_counts = vec![0; 5];
    for label in labels {
        cluster_counts[label as usize] += 1;
    }
    println!("  - Cluster distribution: {:?}", cluster_counts);
}

fn main() {
    println!("\n{}", "#".repeat(60));
    println!("# Time-Series Clustering Example");
    println!("{}", "#".repeat(60));
    println!("\nThis example demonstrates various approaches to clustering time-series data:");
    println!("  1. Feature-based clustering (extract statistical features)");
    println!("  2. Hierarchical clustering with different linkage methods");
    println!("  3. Online clustering for streaming time series");
    println!("  4. Sliding window approach for subsequence clustering");

    // Generate synthetic data
    let (series_data, true_labels) = generate_time_series_data();

    // Run examples
    example_feature_based_clustering(&series_data, &true_labels);
    example_hierarchical_clustering(&series_data);
    example_online_clustering();
    example_sliding_window_clustering(&series_data);

    println!("\n{}", "=".repeat(60));
    println!("All examples completed successfully!");
    println!("{}\n", "=".repeat(60));

    println!("\n💡 Key Takeaways:");
    println!("  • Feature-based clustering works well for overall pattern identification");
    println!("  • Hierarchical clustering reveals relationships between time series");
    println!("  • Online clustering is ideal for streaming data with concept drift");
    println!("  • Sliding windows enable subsequence-level pattern discovery");
    println!("\n📚 For more information, see ALGORITHM_COMPARISON.md");
}
