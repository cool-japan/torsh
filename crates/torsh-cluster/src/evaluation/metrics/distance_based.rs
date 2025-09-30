//! Distance-based clustering evaluation metrics
//!
//! This module contains metrics that primarily rely on distance calculations
//! between data points and cluster centers.

use crate::error::{ClusterError, ClusterResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};
use torsh_tensor::Tensor;

use super::utils::compute_pairwise_distances;

/// Compute silhouette score for clustering quality assessment
///
/// The silhouette score measures how similar a point is to its own cluster
/// compared to other clusters. Values range from -1 to 1, where:
/// - 1 indicates the point is well-clustered
/// - 0 indicates the point is on the border between clusters
/// - -1 indicates the point might be assigned to the wrong cluster
pub fn silhouette_score(data: &Tensor, labels: &Tensor) -> ClusterResult<f64> {
    // Convert tensors to ndarray format for SciRS2 processing
    let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;
    let labels_vec = labels.to_vec().map_err(ClusterError::TensorError)?;

    let shape = data.shape();
    let data_shape = shape.dims();
    if data_shape.len() != 2 {
        return Err(ClusterError::InvalidInput(
            "Data tensor must be 2-dimensional".to_string(),
        ));
    }

    let n_samples = data_shape[0];
    let n_features = data_shape[1];

    if labels_vec.len() != n_samples {
        return Err(ClusterError::InvalidInput(
            "Number of labels must match number of samples".to_string(),
        ));
    }

    // Convert to Array2 for efficient computation
    let data_array = Array2::from_shape_vec((n_samples, n_features), data_vec)
        .map_err(|e| ClusterError::InvalidInput(format!("Failed to reshape data array: {}", e)))?;

    // Convert labels to integers
    let labels_int: Vec<i32> = labels_vec.iter().map(|&x| x as i32).collect();

    // Get unique cluster labels
    let unique_labels: HashSet<i32> = labels_int.iter().cloned().collect();
    let n_clusters = unique_labels.len();

    if n_clusters < 2 {
        return Err(ClusterError::InvalidInput(
            "Need at least 2 clusters for silhouette analysis".to_string(),
        ));
    }

    if n_clusters == n_samples {
        return Ok(0.0); // Each sample is its own cluster
    }

    // Compute pairwise distances using SciRS2
    let distances = compute_pairwise_distances(&data_array)?;

    // Calculate silhouette coefficient for each sample
    let mut silhouette_scores = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let current_label = labels_int[i];

        // Find samples in same cluster (excluding current sample)
        let same_cluster: Vec<usize> = labels_int
            .iter()
            .enumerate()
            .filter_map(|(idx, &label)| {
                if label == current_label && idx != i {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();

        // If only one sample in cluster, silhouette is 0
        if same_cluster.is_empty() {
            silhouette_scores.push(0.0);
            continue;
        }

        // Calculate mean intra-cluster distance (a)
        let intra_cluster_dist: f64 = same_cluster
            .iter()
            .map(|&j| distances[[i, j]] as f64)
            .sum::<f64>()
            / same_cluster.len() as f64;

        // Calculate mean distance to nearest neighboring cluster (b)
        let mut min_inter_cluster_dist = f64::INFINITY;

        for &other_label in &unique_labels {
            if other_label == current_label {
                continue;
            }

            let other_cluster: Vec<usize> = labels_int
                .iter()
                .enumerate()
                .filter_map(|(idx, &label)| {
                    if label == other_label {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect();

            if !other_cluster.is_empty() {
                let inter_cluster_dist: f64 = other_cluster
                    .iter()
                    .map(|&j| distances[[i, j]] as f64)
                    .sum::<f64>()
                    / other_cluster.len() as f64;

                min_inter_cluster_dist = min_inter_cluster_dist.min(inter_cluster_dist);
            }
        }

        // Calculate silhouette coefficient: (b - a) / max(a, b)
        let silhouette_coeff = if min_inter_cluster_dist.is_infinite() {
            0.0
        } else {
            let max_dist = intra_cluster_dist.max(min_inter_cluster_dist);
            if max_dist == 0.0 {
                0.0
            } else {
                (min_inter_cluster_dist - intra_cluster_dist) / max_dist
            }
        };

        silhouette_scores.push(silhouette_coeff);
    }

    // Return mean silhouette score
    let mean_score = silhouette_scores.iter().sum::<f64>() / n_samples as f64;
    Ok(mean_score)
}

/// Compute Calinski-Harabasz score (variance ratio criterion)
///
/// This metric measures the ratio of between-cluster variance to within-cluster variance.
/// Higher values indicate better-defined clusters. The score is unbounded above.
pub fn calinski_harabasz_score(data: &Tensor, labels: &Tensor) -> ClusterResult<f64> {
    let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;
    let labels_vec = labels.to_vec().map_err(ClusterError::TensorError)?;

    let shape = data.shape();
    let data_shape = shape.dims();
    if data_shape.len() != 2 {
        return Err(ClusterError::InvalidInput(
            "Data tensor must be 2-dimensional".to_string(),
        ));
    }

    let n_samples = data_shape[0];
    let n_features = data_shape[1];

    if labels_vec.len() != n_samples {
        return Err(ClusterError::InvalidInput(
            "Number of labels must match number of samples".to_string(),
        ));
    }

    // Convert to Array2 for efficient computation
    let data_array = Array2::from_shape_vec((n_samples, n_features), data_vec)
        .map_err(|e| ClusterError::InvalidInput(format!("Failed to reshape data array: {}", e)))?;

    // Convert labels to integers and get unique clusters
    let labels_int: Vec<i32> = labels_vec.iter().map(|&x| x as i32).collect();
    let unique_labels: HashSet<i32> = labels_int.iter().cloned().collect();
    let k = unique_labels.len();

    if k < 2 {
        return Err(ClusterError::InvalidInput(
            "Need at least 2 clusters for Calinski-Harabasz score".to_string(),
        ));
    }

    if k >= n_samples {
        return Ok(0.0); // Each sample is its own cluster
    }

    // Calculate overall mean (centroid of all data)
    let mut overall_mean = Array1::zeros(n_features);
    for i in 0..n_samples {
        for j in 0..n_features {
            overall_mean[j] += data_array[[i, j]];
        }
    }
    overall_mean /= n_samples as f32;

    // Calculate cluster centroids and sizes
    let mut cluster_centroids: HashMap<i32, Array1<f32>> = HashMap::new();
    let mut cluster_sizes: HashMap<i32, usize> = HashMap::new();

    for &label in &unique_labels {
        let mut centroid = Array1::zeros(n_features);
        let mut count = 0;

        for (i, &sample_label) in labels_int.iter().enumerate() {
            if sample_label == label {
                for j in 0..n_features {
                    centroid[j] += data_array[[i, j]];
                }
                count += 1;
            }
        }

        if count > 0 {
            centroid /= count as f32;
            cluster_centroids.insert(label, centroid);
            cluster_sizes.insert(label, count);
        }
    }

    // Calculate between-cluster sum of squares (BCSS)
    let mut bcss = 0.0_f64;
    for (&label, centroid) in &cluster_centroids {
        let cluster_size = cluster_sizes[&label] as f64;
        let diff = centroid.to_owned() - &overall_mean;
        let squared_distance: f64 = diff.iter().map(|&x| (x as f64) * (x as f64)).sum();
        bcss += cluster_size * squared_distance;
    }

    // Calculate within-cluster sum of squares (WCSS)
    let mut wcss = 0.0_f64;
    for (i, &sample_label) in labels_int.iter().enumerate() {
        if let Some(centroid) = cluster_centroids.get(&sample_label) {
            let sample = data_array.row(i);
            let mut squared_distance = 0.0_f64;

            for j in 0..n_features {
                let diff = sample[j] as f64 - centroid[j] as f64;
                squared_distance += diff * diff;
            }

            wcss += squared_distance;
        }
    }

    // Calculate Calinski-Harabasz score
    if wcss == 0.0 {
        // All points are at their cluster centroids
        Ok(f64::INFINITY)
    } else {
        let ch_score = (bcss / (k - 1) as f64) / (wcss / (n_samples - k) as f64);
        Ok(ch_score)
    }
}

/// Compute Davies-Bouldin score
///
/// This metric measures the average similarity between each cluster and its most similar cluster.
/// Lower values indicate better clustering, with 0 being the best possible score.
pub fn davies_bouldin_score(data: &Tensor, labels: &Tensor) -> ClusterResult<f64> {
    let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;
    let labels_vec = labels.to_vec().map_err(ClusterError::TensorError)?;

    let shape = data.shape();
    let data_shape = shape.dims();
    if data_shape.len() != 2 {
        return Err(ClusterError::InvalidInput(
            "Data tensor must be 2-dimensional".to_string(),
        ));
    }

    let n_samples = data_shape[0];
    let n_features = data_shape[1];

    if labels_vec.len() != n_samples {
        return Err(ClusterError::InvalidInput(
            "Number of labels must match number of samples".to_string(),
        ));
    }

    // Convert to Array2 for efficient computation
    let data_array = Array2::from_shape_vec((n_samples, n_features), data_vec)
        .map_err(|e| ClusterError::InvalidInput(format!("Failed to reshape data array: {}", e)))?;

    // Convert labels to integers and get unique clusters
    let labels_int: Vec<i32> = labels_vec.iter().map(|&x| x as i32).collect();
    let unique_labels: HashSet<i32> = labels_int.iter().cloned().collect();
    let k = unique_labels.len();

    if k < 2 {
        return Err(ClusterError::InvalidInput(
            "Need at least 2 clusters for Davies-Bouldin score".to_string(),
        ));
    }

    if k >= n_samples {
        return Ok(0.0); // Each sample is its own cluster - perfect separation
    }

    // Calculate cluster centroids
    let mut cluster_centroids: HashMap<i32, Array1<f32>> = HashMap::new();

    for &label in &unique_labels {
        let mut centroid = Array1::zeros(n_features);
        let mut count = 0;

        for (i, &sample_label) in labels_int.iter().enumerate() {
            if sample_label == label {
                for j in 0..n_features {
                    centroid[j] += data_array[[i, j]];
                }
                count += 1;
            }
        }

        if count > 0 {
            centroid /= count as f32;
            cluster_centroids.insert(label, centroid);
        }
    }

    // Calculate within-cluster scatter (average distance from centroid) for each cluster
    let mut within_cluster_scatter: HashMap<i32, f64> = HashMap::new();

    for &label in &unique_labels {
        if let Some(centroid) = cluster_centroids.get(&label) {
            let mut total_distance = 0.0_f64;
            let mut count = 0;

            for (i, &sample_label) in labels_int.iter().enumerate() {
                if sample_label == label {
                    let sample = data_array.row(i);
                    let mut distance = 0.0_f64;

                    for j in 0..n_features {
                        let diff = sample[j] as f64 - centroid[j] as f64;
                        distance += diff * diff;
                    }

                    total_distance += distance.sqrt();
                    count += 1;
                }
            }

            if count > 0 {
                within_cluster_scatter.insert(label, total_distance / count as f64);
            }
        }
    }

    // Calculate Davies-Bouldin index
    let mut db_sum = 0.0_f64;

    for &label_i in &unique_labels {
        let mut max_similarity = 0.0_f64;

        for &label_j in &unique_labels {
            if label_i != label_j {
                if let (Some(centroid_i), Some(centroid_j)) = (
                    cluster_centroids.get(&label_i),
                    cluster_centroids.get(&label_j),
                ) {
                    // Calculate distance between cluster centroids
                    let mut centroid_distance = 0.0_f64;
                    for k in 0..n_features {
                        let diff = centroid_i[k] as f64 - centroid_j[k] as f64;
                        centroid_distance += diff * diff;
                    }
                    centroid_distance = centroid_distance.sqrt();

                    // Calculate similarity measure
                    if centroid_distance > 0.0 {
                        let scatter_i = within_cluster_scatter.get(&label_i).unwrap_or(&0.0);
                        let scatter_j = within_cluster_scatter.get(&label_j).unwrap_or(&0.0);
                        let similarity = (scatter_i + scatter_j) / centroid_distance;
                        max_similarity = max_similarity.max(similarity);
                    }
                }
            }
        }

        db_sum += max_similarity;
    }

    // Davies-Bouldin score is the average maximum similarity
    let db_score = db_sum / k as f64;
    Ok(db_score)
}

/// Compute Dunn Index for clustering validation
///
/// The Dunn Index measures clustering quality by calculating the ratio of the minimum
/// inter-cluster distance to the maximum intra-cluster distance. Higher values indicate
/// better clustering with well-separated, compact clusters.
///
/// # Formula
/// Dunn Index = min(inter-cluster distance) / max(intra-cluster distance)
///
/// # Arguments
/// * `data` - Data points as a 2D tensor (samples × features)
/// * `labels` - Cluster labels for each data point
///
/// # Returns
/// * `Ok(dunn_index)` - The Dunn index (higher is better)
/// * `Err(ClusterError)` - If inputs are invalid
pub fn dunn_index(data: &Tensor, labels: &Tensor) -> ClusterResult<f64> {
    let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;
    let labels_vec = labels.to_vec().map_err(ClusterError::TensorError)?;

    let shape = data.shape();
    let data_shape = shape.dims();
    if data_shape.len() != 2 {
        return Err(ClusterError::InvalidInput(
            "Data tensor must be 2-dimensional".to_string(),
        ));
    }

    let n_samples = data_shape[0];
    let n_features = data_shape[1];

    if labels_vec.len() != n_samples {
        return Err(ClusterError::InvalidInput(
            "Number of labels must match number of samples".to_string(),
        ));
    }

    // Convert to Array2 for efficient computation
    let data_array = Array2::from_shape_vec((n_samples, n_features), data_vec)
        .map_err(|e| ClusterError::InvalidInput(format!("Failed to reshape data array: {}", e)))?;

    // Convert labels to integers and get unique clusters
    let labels_int: Vec<i32> = labels_vec.iter().map(|&x| x as i32).collect();
    let unique_labels: HashSet<i32> = labels_int.iter().cloned().collect();
    let k = unique_labels.len();

    if k < 2 {
        return Err(ClusterError::InvalidInput(
            "Need at least 2 clusters for Dunn Index".to_string(),
        ));
    }

    // Group data points by cluster
    let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
    for (i, &label) in labels_int.iter().enumerate() {
        clusters.entry(label).or_default().push(i);
    }

    // Calculate minimum inter-cluster distance
    let mut min_inter_cluster_distance = f64::INFINITY;

    for (&label_i, points_i) in &clusters {
        for (&label_j, points_j) in &clusters {
            if label_i >= label_j {
                continue; // Only compute each pair once
            }

            // Find minimum distance between any two points in different clusters
            for &i in points_i {
                for &j in points_j {
                    let mut distance = 0.0_f64;
                    for k in 0..n_features {
                        let diff = data_array[[i, k]] as f64 - data_array[[j, k]] as f64;
                        distance += diff * diff;
                    }
                    distance = distance.sqrt();
                    min_inter_cluster_distance = min_inter_cluster_distance.min(distance);
                }
            }
        }
    }

    // Calculate maximum intra-cluster distance
    let mut max_intra_cluster_distance = 0.0_f64;

    for points in clusters.values() {
        if points.len() < 2 {
            continue; // Skip clusters with only one point
        }

        for i in 0..points.len() {
            for j in i + 1..points.len() {
                let idx_i = points[i];
                let idx_j = points[j];

                let mut distance = 0.0_f64;
                for k in 0..n_features {
                    let diff = data_array[[idx_i, k]] as f64 - data_array[[idx_j, k]] as f64;
                    distance += diff * diff;
                }
                distance = distance.sqrt();
                max_intra_cluster_distance = max_intra_cluster_distance.max(distance);
            }
        }
    }

    // Dunn Index = min_inter / max_intra
    if max_intra_cluster_distance == 0.0 {
        // All points in each cluster are identical
        Ok(f64::INFINITY)
    } else {
        Ok(min_inter_cluster_distance / max_intra_cluster_distance)
    }
}

/// Compute Xie-Beni Index for clustering validation
///
/// The Xie-Beni Index is a fuzzy clustering validity measure that evaluates both
/// compactness and separation of clusters. Lower values indicate better clustering.
///
/// # Formula
/// XB Index = Σ(||xi - cj||²) / (N × min(||ca - cb||²))
/// where xi is a data point, cj is the nearest cluster center, and ca, cb are cluster centers.
///
/// # Arguments
/// * `data` - Data points as a 2D tensor (samples × features)
/// * `labels` - Cluster labels for each data point
///
/// # Returns
/// * `Ok(xb_index)` - The Xie-Beni index (lower is better)
/// * `Err(ClusterError)` - If inputs are invalid
pub fn xie_beni_index(data: &Tensor, labels: &Tensor) -> ClusterResult<f64> {
    let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;
    let labels_vec = labels.to_vec().map_err(ClusterError::TensorError)?;

    let shape = data.shape();
    let data_shape = shape.dims();
    if data_shape.len() != 2 {
        return Err(ClusterError::InvalidInput(
            "Data tensor must be 2-dimensional".to_string(),
        ));
    }

    let n_samples = data_shape[0];
    let n_features = data_shape[1];

    if labels_vec.len() != n_samples {
        return Err(ClusterError::InvalidInput(
            "Number of labels must match number of samples".to_string(),
        ));
    }

    // Convert to Array2 for efficient computation
    let data_array = Array2::from_shape_vec((n_samples, n_features), data_vec)
        .map_err(|e| ClusterError::InvalidInput(format!("Failed to reshape data array: {}", e)))?;

    // Convert labels to integers and get unique clusters
    let labels_int: Vec<i32> = labels_vec.iter().map(|&x| x as i32).collect();
    let unique_labels: HashSet<i32> = labels_int.iter().cloned().collect();
    let k = unique_labels.len();

    if k < 2 {
        return Err(ClusterError::InvalidInput(
            "Need at least 2 clusters for Xie-Beni Index".to_string(),
        ));
    }

    // Calculate cluster centroids
    let mut cluster_centroids: HashMap<i32, Array1<f32>> = HashMap::new();

    for &label in &unique_labels {
        let mut centroid = Array1::zeros(n_features);
        let mut count = 0;

        for (i, &sample_label) in labels_int.iter().enumerate() {
            if sample_label == label {
                for j in 0..n_features {
                    centroid[j] += data_array[[i, j]];
                }
                count += 1;
            }
        }

        if count > 0 {
            centroid /= count as f32;
            cluster_centroids.insert(label, centroid);
        }
    }

    // Calculate numerator: sum of squared distances from points to their cluster centers
    let mut numerator = 0.0_f64;
    for (i, &sample_label) in labels_int.iter().enumerate() {
        if let Some(centroid) = cluster_centroids.get(&sample_label) {
            let sample = data_array.row(i);
            let mut squared_distance = 0.0_f64;

            for j in 0..n_features {
                let diff = sample[j] as f64 - centroid[j] as f64;
                squared_distance += diff * diff;
            }

            numerator += squared_distance;
        }
    }

    // Calculate denominator: N × minimum squared distance between cluster centers
    let mut min_centroid_distance_squared = f64::INFINITY;

    let centroids_vec: Vec<_> = cluster_centroids.values().collect();
    for i in 0..centroids_vec.len() {
        for j in i + 1..centroids_vec.len() {
            let centroid_i = &centroids_vec[i];
            let centroid_j = &centroids_vec[j];

            let mut distance_squared = 0.0_f64;
            for k in 0..n_features {
                let diff = centroid_i[k] as f64 - centroid_j[k] as f64;
                distance_squared += diff * diff;
            }

            min_centroid_distance_squared = min_centroid_distance_squared.min(distance_squared);
        }
    }

    let denominator = n_samples as f64 * min_centroid_distance_squared;

    // Xie-Beni Index = numerator / denominator
    if denominator == 0.0 {
        // All cluster centers are identical
        Ok(f64::INFINITY)
    } else {
        Ok(numerator / denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dunn_index_basic() -> Result<(), Box<dyn std::error::Error>> {
        // Create well-separated clusters
        let data = Tensor::from_vec(
            vec![
                // Cluster 0 (around origin)
                0.0, 0.0, 0.1, 0.1, -0.1, 0.1, // Cluster 1 (around (5,5))
                5.0, 5.0, 5.1, 5.1, 4.9, 5.1,
            ],
            &[6, 2],
        )?;

        let labels = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[6])?;

        let dunn = dunn_index(&data, &labels)?;

        // Dunn index should be positive for well-separated clusters
        assert!(dunn > 0.0, "Dunn index should be positive: {}", dunn);

        Ok(())
    }

    #[test]
    fn test_dunn_index_perfect_separation() -> Result<(), Box<dyn std::error::Error>> {
        // Create perfectly separated point clusters
        let data = Tensor::from_vec(
            vec![
                0.0, 0.0, // Cluster 0
                10.0, 10.0, // Cluster 1
            ],
            &[2, 2],
        )?;

        let labels = Tensor::from_vec(vec![0.0, 1.0], &[2])?;

        let dunn = dunn_index(&data, &labels)?;

        // Should have high Dunn index for perfect separation
        assert!(
            dunn > 1.0,
            "Dunn index should be high for perfect separation: {}",
            dunn
        );

        Ok(())
    }

    #[test]
    fn test_xie_beni_index_basic() -> Result<(), Box<dyn std::error::Error>> {
        // Create well-separated clusters
        let data = Tensor::from_vec(
            vec![
                // Cluster 0
                0.0, 0.0, 0.1, 0.1, -0.1, 0.1, // Cluster 1
                5.0, 5.0, 5.1, 5.1, 4.9, 5.1,
            ],
            &[6, 2],
        )?;

        let labels = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[6])?;

        let xb = xie_beni_index(&data, &labels)?;

        // Xie-Beni index should be positive and finite
        assert!(
            xb > 0.0 && xb.is_finite(),
            "Xie-Beni index should be positive and finite: {}",
            xb
        );

        Ok(())
    }

    #[test]
    fn test_xie_beni_index_perfect_clusters() -> Result<(), Box<dyn std::error::Error>> {
        // Create identical points in each cluster (perfect compactness)
        let data = Tensor::from_vec(
            vec![
                0.0, 0.0, 0.0, 0.0, // Cluster 0 (identical points)
                5.0, 5.0, 5.0, 5.0, // Cluster 1 (identical points)
            ],
            &[4, 2],
        )?;

        let labels = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[4])?;

        let xb = xie_beni_index(&data, &labels)?;

        // Should be 0 for perfect clusters (numerator = 0)
        assert_relative_eq!(xb, 0.0, epsilon = 1e-10);

        Ok(())
    }

    #[test]
    fn test_metrics_error_cases() -> Result<(), Box<dyn std::error::Error>> {
        // Test with insufficient clusters
        let data = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[2, 2])?;
        let single_cluster_labels = Tensor::from_vec(vec![0.0, 0.0], &[2])?;

        // Should error with single cluster
        assert!(dunn_index(&data, &single_cluster_labels).is_err());
        assert!(xie_beni_index(&data, &single_cluster_labels).is_err());

        // Test with mismatched dimensions
        let mismatched_labels = Tensor::from_vec(vec![0.0], &[1])?;
        assert!(dunn_index(&data, &mismatched_labels).is_err());
        assert!(xie_beni_index(&data, &mismatched_labels).is_err());

        Ok(())
    }
}
