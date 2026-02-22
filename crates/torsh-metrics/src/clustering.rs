//! Clustering metrics with comprehensive evaluation algorithms
//!
//! This module provides robust clustering evaluation metrics that work with
//! both known ground truth labels and unsupervised clustering results.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::Metric;
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;
use torsh_tensor::Tensor;

#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
    Chebyshev,
}

/// Silhouette Score - measures how similar an object is to its own cluster
/// compared to other clusters. Range: [-1, 1], higher is better.
pub struct SilhouetteScore {
    metric: DistanceMetric,
}

impl SilhouetteScore {
    /// Create a new silhouette score metric
    pub fn new(metric: DistanceMetric) -> Self {
        Self { metric }
    }

    /// Compute silhouette score
    pub fn compute_score(&self, data: &Tensor, labels: &Tensor) -> f64 {
        match (data.to_vec(), labels.to_vec()) {
            (Ok(data_vec), Ok(labels_vec)) => {
                let shape = data.shape();
                let dims = shape.dims();

                if dims.len() != 2 || dims[0] != labels_vec.len() || dims[0] < 2 {
                    return 0.0;
                }

                let n_samples = dims[0];
                let n_features = dims[1];

                // Convert to more efficient format
                let data_array = Array2::from_shape_vec(
                    (n_samples, n_features),
                    data_vec.iter().map(|&x| x as f64).collect(),
                )
                .unwrap_or_else(|_| Array2::zeros((n_samples, n_features)));

                let labels_array = Array1::from_vec(labels_vec.iter().map(|&x| x as i32).collect());

                self.compute_silhouette_vectorized(&data_array, &labels_array)
            }
            _ => 0.0,
        }
    }

    fn compute_silhouette_vectorized(&self, data: &Array2<f64>, labels: &Array1<i32>) -> f64 {
        let n_samples = data.nrows();

        // Get unique labels and their indices
        let mut label_to_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            label_to_indices.entry(label).or_default().push(i);
        }

        // Remove clusters with only one sample
        label_to_indices.retain(|_, indices| indices.len() > 1);

        if label_to_indices.len() < 2 {
            return 0.0; // Need at least 2 clusters
        }

        let mut silhouette_scores = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let label = labels[i];

            // Skip if this sample's cluster has only one member
            if label_to_indices
                .get(&label)
                .map_or(true, |indices| indices.len() <= 1)
            {
                continue;
            }

            // Calculate a(i): mean distance to other points in same cluster
            let same_cluster = &label_to_indices[&label];
            let a = if same_cluster.len() > 1 {
                let distances: f64 = same_cluster
                    .iter()
                    .filter(|&&j| j != i)
                    .map(|&j| self.compute_distance(data.row(i), data.row(j)))
                    .sum();
                distances / (same_cluster.len() - 1) as f64
            } else {
                0.0
            };

            // Calculate b(i): minimum mean distance to points in other clusters
            let mut min_b = f64::INFINITY;
            for (other_label, other_indices) in &label_to_indices {
                if *other_label != label {
                    let distances: f64 = other_indices
                        .iter()
                        .map(|&j| self.compute_distance(data.row(i), data.row(j)))
                        .sum();
                    let mean_distance = distances / other_indices.len() as f64;
                    min_b = min_b.min(mean_distance);
                }
            }

            if min_b.is_finite() {
                let silhouette = (min_b - a) / a.max(min_b);
                silhouette_scores.push(silhouette);
            }
        }

        if silhouette_scores.is_empty() {
            0.0
        } else {
            silhouette_scores.iter().sum::<f64>() / silhouette_scores.len() as f64
        }
    }

    fn compute_distance(
        &self,
        point1: scirs2_core::ndarray::ArrayView1<f64>,
        point2: scirs2_core::ndarray::ArrayView1<f64>,
    ) -> f64 {
        match self.metric {
            DistanceMetric::Euclidean => point1
                .iter()
                .zip(point2.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt(),
            DistanceMetric::Manhattan => point1
                .iter()
                .zip(point2.iter())
                .map(|(&a, &b)| (a - b).abs())
                .sum::<f64>(),
            DistanceMetric::Cosine => {
                let dot_product: f64 = point1.iter().zip(point2.iter()).map(|(&a, &b)| a * b).sum();
                let norm1: f64 = point1.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();
                let norm2: f64 = point2.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();

                if norm1 > 1e-10 && norm2 > 1e-10 {
                    1.0 - (dot_product / (norm1 * norm2))
                } else {
                    1.0
                }
            }
            DistanceMetric::Chebyshev => point1
                .iter()
                .zip(point2.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0, f64::max),
        }
    }
}

impl Metric for SilhouetteScore {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        // For clustering metrics, predictions=data, targets=labels
        self.compute_score(predictions, targets)
    }

    fn name(&self) -> &str {
        "silhouette_score"
    }
}

/// Davies-Bouldin Index - lower values indicate better clustering
pub struct DaviesBouldin;

impl DaviesBouldin {
    /// Compute Davies-Bouldin index
    pub fn compute_score(&self, data: &Tensor, labels: &Tensor) -> f64 {
        match (data.to_vec(), labels.to_vec()) {
            (Ok(data_vec), Ok(labels_vec)) => {
                let shape = data.shape();
                let dims = shape.dims();

                if dims.len() != 2 || dims[0] != labels_vec.len() || dims[0] < 2 {
                    return f64::INFINITY;
                }

                let n_samples = dims[0];
                let n_features = dims[1];

                let data_array = Array2::from_shape_vec(
                    (n_samples, n_features),
                    data_vec.iter().map(|&x| x as f64).collect(),
                )
                .unwrap_or_else(|_| Array2::zeros((n_samples, n_features)));

                let labels_array = Array1::from_vec(labels_vec.iter().map(|&x| x as i32).collect());

                self.compute_davies_bouldin(&data_array, &labels_array)
            }
            _ => f64::INFINITY,
        }
    }

    fn compute_davies_bouldin(&self, data: &Array2<f64>, labels: &Array1<i32>) -> f64 {
        // Group data by clusters
        let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            clusters.entry(label).or_default().push(i);
        }

        if clusters.len() < 2 {
            return f64::INFINITY;
        }

        // Calculate centroids and within-cluster distances
        let mut centroids: HashMap<i32, Array1<f64>> = HashMap::new();
        let mut within_distances: HashMap<i32, f64> = HashMap::new();

        for (label, indices) in &clusters {
            let n_points = indices.len();
            if n_points == 0 {
                continue;
            }

            // Calculate centroid
            let mut centroid = Array1::<f64>::zeros(data.ncols());
            for &i in indices {
                let row = data.row(i);
                for (j, &val) in row.iter().enumerate() {
                    centroid[j] += val;
                }
            }
            for val in centroid.iter_mut() {
                *val /= n_points as f64;
            }

            // Calculate within-cluster distance (average distance to centroid)
            let mut sum_distance = 0.0;
            for &i in indices {
                let distance = euclidean_distance(data.row(i), centroid.view());
                sum_distance += distance;
            }

            centroids.insert(*label, centroid);
            within_distances.insert(*label, sum_distance / n_points as f64);
        }

        // Calculate Davies-Bouldin index
        let mut sum_db = 0.0;
        let cluster_labels: Vec<i32> = clusters.keys().cloned().collect();

        for &label_i in &cluster_labels {
            let mut max_ratio: f64 = 0.0;

            for &label_j in &cluster_labels {
                if label_i != label_j {
                    let si = within_distances[&label_i];
                    let sj = within_distances[&label_j];
                    let mij =
                        euclidean_distance(centroids[&label_i].view(), centroids[&label_j].view());

                    if mij > 1e-10 {
                        let ratio = (si + sj) / mij;
                        max_ratio = max_ratio.max(ratio);
                    }
                }
            }

            sum_db += max_ratio;
        }

        sum_db / cluster_labels.len() as f64
    }
}

impl Metric for DaviesBouldin {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_score(predictions, targets)
    }

    fn name(&self) -> &str {
        "davies_bouldin"
    }
}

/// Calinski-Harabasz Index (Variance Ratio Criterion) - higher is better
pub struct CalinskiHarabasz;

impl CalinskiHarabasz {
    /// Compute Calinski-Harabasz index
    pub fn compute_score(&self, data: &Tensor, labels: &Tensor) -> f64 {
        match (data.to_vec(), labels.to_vec()) {
            (Ok(data_vec), Ok(labels_vec)) => {
                let shape = data.shape();
                let dims = shape.dims();

                if dims.len() != 2 || dims[0] != labels_vec.len() || dims[0] < 2 {
                    return 0.0;
                }

                let n_samples = dims[0];
                let n_features = dims[1];

                let data_array = Array2::from_shape_vec(
                    (n_samples, n_features),
                    data_vec.iter().map(|&x| x as f64).collect(),
                )
                .unwrap_or_else(|_| Array2::zeros((n_samples, n_features)));

                let labels_array = Array1::from_vec(labels_vec.iter().map(|&x| x as i32).collect());

                self.compute_calinski_harabasz(&data_array, &labels_array)
            }
            _ => 0.0,
        }
    }

    fn compute_calinski_harabasz(&self, data: &Array2<f64>, labels: &Array1<i32>) -> f64 {
        let n_samples = data.nrows();
        let n_features = data.ncols();

        // Group data by clusters
        let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            clusters.entry(label).or_default().push(i);
        }

        let n_clusters = clusters.len();
        if n_clusters < 2 || n_samples <= n_clusters {
            return 0.0;
        }

        // Calculate overall mean
        let mut overall_mean = Array1::<f64>::zeros(n_features);
        for i in 0..n_samples {
            for j in 0..n_features {
                overall_mean[j] += data[[i, j]];
            }
        }
        for val in overall_mean.iter_mut() {
            *val /= n_samples as f64;
        }

        // Calculate between-cluster sum of squares (BCSS)
        let mut bcss = 0.0;
        for (_, indices) in &clusters {
            let cluster_size = indices.len();
            if cluster_size == 0 {
                continue;
            }

            // Calculate cluster centroid
            let mut centroid = Array1::<f64>::zeros(n_features);
            for &i in indices {
                for j in 0..n_features {
                    centroid[j] += data[[i, j]];
                }
            }
            for val in centroid.iter_mut() {
                *val /= cluster_size as f64;
            }

            // Add to BCSS
            for j in 0..n_features {
                let diff = centroid[j] - overall_mean[j];
                bcss += cluster_size as f64 * diff * diff;
            }
        }

        // Calculate within-cluster sum of squares (WCSS)
        let mut wcss = 0.0;
        for (_, indices) in &clusters {
            let cluster_size = indices.len();
            if cluster_size == 0 {
                continue;
            }

            // Calculate cluster centroid
            let mut centroid = Array1::<f64>::zeros(n_features);
            for &i in indices {
                for j in 0..n_features {
                    centroid[j] += data[[i, j]];
                }
            }
            for val in centroid.iter_mut() {
                *val /= cluster_size as f64;
            }

            // Add to WCSS
            for &i in indices {
                for j in 0..n_features {
                    let diff = data[[i, j]] - centroid[j];
                    wcss += diff * diff;
                }
            }
        }

        if wcss > 1e-10 {
            (bcss / (n_clusters - 1) as f64) / (wcss / (n_samples - n_clusters) as f64)
        } else {
            0.0
        }
    }
}

impl Metric for CalinskiHarabasz {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_score(predictions, targets)
    }

    fn name(&self) -> &str {
        "calinski_harabasz"
    }
}

/// Adjusted Rand Index - measures similarity between true and predicted clustering
/// Range: [-1, 1], 1 is perfect match, 0 is random
pub struct AdjustedRandIndex;

impl AdjustedRandIndex {
    /// Compute adjusted rand index
    pub fn compute_score(&self, true_labels: &Tensor, pred_labels: &Tensor) -> f64 {
        match (true_labels.to_vec(), pred_labels.to_vec()) {
            (Ok(true_vec), Ok(pred_vec)) => {
                if true_vec.len() != pred_vec.len() || true_vec.is_empty() {
                    return 0.0;
                }

                let true_labels = true_vec.iter().map(|&x| x as i32).collect::<Vec<_>>();
                let pred_labels = pred_vec.iter().map(|&x| x as i32).collect::<Vec<_>>();

                self.compute_ari(&true_labels, &pred_labels)
            }
            _ => 0.0,
        }
    }

    fn compute_ari(&self, true_labels: &[i32], pred_labels: &[i32]) -> f64 {
        let n = true_labels.len();
        if n == 0 {
            return 0.0;
        }

        // Build contingency table
        let mut contingency: HashMap<(i32, i32), usize> = HashMap::new();
        let mut true_counts: HashMap<i32, usize> = HashMap::new();
        let mut pred_counts: HashMap<i32, usize> = HashMap::new();

        for (&true_label, &pred_label) in true_labels.iter().zip(pred_labels.iter()) {
            *contingency.entry((true_label, pred_label)).or_insert(0) += 1;
            *true_counts.entry(true_label).or_insert(0) += 1;
            *pred_counts.entry(pred_label).or_insert(0) += 1;
        }

        // Calculate index
        let mut sum_nij_choose_2 = 0.0;
        for &count in contingency.values() {
            if count >= 2 {
                sum_nij_choose_2 += (count * (count - 1)) as f64 / 2.0;
            }
        }

        let mut sum_ai_choose_2 = 0.0;
        for &count in true_counts.values() {
            if count >= 2 {
                sum_ai_choose_2 += (count * (count - 1)) as f64 / 2.0;
            }
        }

        let mut sum_bj_choose_2 = 0.0;
        for &count in pred_counts.values() {
            if count >= 2 {
                sum_bj_choose_2 += (count * (count - 1)) as f64 / 2.0;
            }
        }

        let expected_index = (sum_ai_choose_2 * sum_bj_choose_2) / ((n * (n - 1)) as f64 / 2.0);
        let max_index = (sum_ai_choose_2 + sum_bj_choose_2) / 2.0;

        if max_index - expected_index > 1e-10 {
            (sum_nij_choose_2 - expected_index) / (max_index - expected_index)
        } else {
            1.0 // Perfect clustering when max_index equals expected_index
        }
    }
}

impl Metric for AdjustedRandIndex {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_score(targets, predictions) // Note: swapped order for consistency
    }

    fn name(&self) -> &str {
        "adjusted_rand_index"
    }
}

/// Normalized Mutual Information
pub struct NormalizedMutualInfo {
    average_method: String,
}

impl NormalizedMutualInfo {
    /// Create a new NMI metric
    pub fn new(average_method: &str) -> Self {
        Self {
            average_method: average_method.to_string(),
        }
    }

    /// Compute NMI score
    pub fn compute_score(&self, true_labels: &Tensor, pred_labels: &Tensor) -> f64 {
        match (true_labels.to_vec(), pred_labels.to_vec()) {
            (Ok(true_vec), Ok(pred_vec)) => {
                if true_vec.len() != pred_vec.len() || true_vec.is_empty() {
                    return 0.0;
                }

                let true_labels = true_vec.iter().map(|&x| x as i32).collect::<Vec<_>>();
                let pred_labels = pred_vec.iter().map(|&x| x as i32).collect::<Vec<_>>();

                self.compute_nmi(&true_labels, &pred_labels)
            }
            _ => 0.0,
        }
    }

    fn compute_nmi(&self, true_labels: &[i32], pred_labels: &[i32]) -> f64 {
        let n = true_labels.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        // Build contingency table
        let mut contingency: HashMap<(i32, i32), f64> = HashMap::new();
        let mut true_counts: HashMap<i32, f64> = HashMap::new();
        let mut pred_counts: HashMap<i32, f64> = HashMap::new();

        for (&true_label, &pred_label) in true_labels.iter().zip(pred_labels.iter()) {
            *contingency.entry((true_label, pred_label)).or_insert(0.0) += 1.0;
            *true_counts.entry(true_label).or_insert(0.0) += 1.0;
            *pred_counts.entry(pred_label).or_insert(0.0) += 1.0;
        }

        // Calculate mutual information
        let mut mi = 0.0;
        for (&(true_label, pred_label), &nij) in &contingency {
            if nij > 0.0 {
                let pi = true_counts[&true_label] / n;
                let pj = pred_counts[&pred_label] / n;
                let pij = nij / n;

                if pi > 0.0 && pj > 0.0 {
                    mi += pij * (pij / (pi * pj)).ln();
                }
            }
        }

        // Calculate entropies
        let mut h_true = 0.0;
        for &count in true_counts.values() {
            if count > 0.0 {
                let p = count / n;
                h_true -= p * p.ln();
            }
        }

        let mut h_pred = 0.0;
        for &count in pred_counts.values() {
            if count > 0.0 {
                let p = count / n;
                h_pred -= p * p.ln();
            }
        }

        // Compute NMI based on averaging method
        match self.average_method.as_str() {
            "arithmetic" => {
                let avg = (h_true + h_pred) / 2.0;
                if avg > 1e-10 {
                    mi / avg
                } else {
                    0.0
                }
            }
            "geometric" => {
                if h_true > 1e-10 && h_pred > 1e-10 {
                    mi / (h_true * h_pred).sqrt()
                } else {
                    0.0
                }
            }
            "max" => {
                let max_h = h_true.max(h_pred);
                if max_h > 1e-10 {
                    mi / max_h
                } else {
                    0.0
                }
            }
            "min" => {
                let min_h = h_true.min(h_pred);
                if min_h > 1e-10 {
                    mi / min_h
                } else {
                    0.0
                }
            }
            _ => {
                // Default: arithmetic mean
                let avg = (h_true + h_pred) / 2.0;
                if avg > 1e-10 {
                    mi / avg
                } else {
                    0.0
                }
            }
        }
    }
}

impl Metric for NormalizedMutualInfo {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_score(targets, predictions)
    }

    fn name(&self) -> &str {
        "normalized_mutual_info"
    }
}

// Helper function for Euclidean distance
fn euclidean_distance(
    point1: scirs2_core::ndarray::ArrayView1<f64>,
    point2: scirs2_core::ndarray::ArrayView1<f64>,
) -> f64 {
    point1
        .iter()
        .zip(point2.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Dunn Index - ratio of minimum inter-cluster distance to maximum intra-cluster distance
/// Higher values indicate better clustering
pub struct DunnIndex;

impl DunnIndex {
    /// Compute Dunn index
    pub fn compute_score(&self, data: &Tensor, labels: &Tensor) -> f64 {
        match (data.to_vec(), labels.to_vec()) {
            (Ok(data_vec), Ok(labels_vec)) => {
                let shape = data.shape();
                let dims = shape.dims();

                if dims.len() != 2 || dims[0] != labels_vec.len() || dims[0] < 2 {
                    return 0.0;
                }

                let n_samples = dims[0];
                let n_features = dims[1];

                let data_array = Array2::from_shape_vec(
                    (n_samples, n_features),
                    data_vec.iter().map(|&x| x as f64).collect(),
                )
                .unwrap_or_else(|_| Array2::zeros((n_samples, n_features)));

                let labels_array = Array1::from_vec(labels_vec.iter().map(|&x| x as i32).collect());

                self.compute_dunn(&data_array, &labels_array)
            }
            _ => 0.0,
        }
    }

    fn compute_dunn(&self, data: &Array2<f64>, labels: &Array1<i32>) -> f64 {
        // Group data by clusters
        let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            clusters.entry(label).or_default().push(i);
        }

        if clusters.len() < 2 {
            return 0.0;
        }

        // Calculate minimum inter-cluster distance
        let cluster_labels: Vec<i32> = clusters.keys().cloned().collect();
        let mut min_inter_distance = f64::INFINITY;

        for i in 0..cluster_labels.len() {
            for j in (i + 1)..cluster_labels.len() {
                let cluster_i = &clusters[&cluster_labels[i]];
                let cluster_j = &clusters[&cluster_labels[j]];

                for &idx_i in cluster_i {
                    for &idx_j in cluster_j {
                        let distance = euclidean_distance(data.row(idx_i), data.row(idx_j));
                        min_inter_distance = min_inter_distance.min(distance);
                    }
                }
            }
        }

        // Calculate maximum intra-cluster distance
        let mut max_intra_distance: f64 = 0.0;

        for (_, indices) in &clusters {
            for i in 0..indices.len() {
                for j in (i + 1)..indices.len() {
                    let distance = euclidean_distance(data.row(indices[i]), data.row(indices[j]));
                    max_intra_distance = max_intra_distance.max(distance);
                }
            }
        }

        if max_intra_distance > 1e-10 {
            min_inter_distance / max_intra_distance
        } else {
            f64::INFINITY // Perfect clustering
        }
    }
}

impl Metric for DunnIndex {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_score(predictions, targets)
    }

    fn name(&self) -> &str {
        "dunn_index"
    }
}

/// V-measure Score - harmonic mean between homogeneity and completeness
pub struct VMeasure {
    beta: f64,
}

impl VMeasure {
    /// Create a new V-measure metric
    pub fn new(beta: f64) -> Self {
        Self { beta }
    }

    /// Compute V-measure score
    pub fn compute_score(&self, true_labels: &Tensor, pred_labels: &Tensor) -> f64 {
        match (true_labels.to_vec(), pred_labels.to_vec()) {
            (Ok(true_vec), Ok(pred_vec)) => {
                if true_vec.len() != pred_vec.len() || true_vec.is_empty() {
                    return 0.0;
                }

                let true_labels = true_vec.iter().map(|&x| x as i32).collect::<Vec<_>>();
                let pred_labels = pred_vec.iter().map(|&x| x as i32).collect::<Vec<_>>();

                let (homogeneity, completeness) =
                    self.compute_homogeneity_completeness(&true_labels, &pred_labels);

                if homogeneity + completeness > 1e-10 {
                    (1.0 + self.beta * self.beta) * homogeneity * completeness
                        / (self.beta * self.beta * homogeneity + completeness)
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    fn compute_homogeneity_completeness(
        &self,
        true_labels: &[i32],
        pred_labels: &[i32],
    ) -> (f64, f64) {
        let n = true_labels.len() as f64;
        if n == 0.0 {
            return (0.0, 0.0);
        }

        // Build contingency table
        let mut contingency: HashMap<(i32, i32), f64> = HashMap::new();
        let mut true_counts: HashMap<i32, f64> = HashMap::new();
        let mut pred_counts: HashMap<i32, f64> = HashMap::new();

        for (&true_label, &pred_label) in true_labels.iter().zip(pred_labels.iter()) {
            *contingency.entry((true_label, pred_label)).or_insert(0.0) += 1.0;
            *true_counts.entry(true_label).or_insert(0.0) += 1.0;
            *pred_counts.entry(pred_label).or_insert(0.0) += 1.0;
        }

        // Calculate conditional entropies
        let mut h_c_k = 0.0; // H(C|K) - entropy of true labels given predicted labels
        for &nk in pred_counts.values() {
            if nk > 0.0 {
                let mut h_cluster = 0.0;
                for (&(_true_label, pred_label), &nij) in &contingency {
                    if pred_counts.get(&pred_label) == Some(&nk) && nij > 0.0 {
                        let p = nij / nk;
                        h_cluster -= p * p.ln();
                    }
                }
                h_c_k += (nk / n) * h_cluster;
            }
        }

        let mut h_k_c = 0.0; // H(K|C) - entropy of predicted labels given true labels
        for &nc in true_counts.values() {
            if nc > 0.0 {
                let mut h_cluster = 0.0;
                for (&(true_label, _pred_label), &nij) in &contingency {
                    if true_counts.get(&true_label) == Some(&nc) && nij > 0.0 {
                        let p = nij / nc;
                        h_cluster -= p * p.ln();
                    }
                }
                h_k_c += (nc / n) * h_cluster;
            }
        }

        // Calculate entropies
        let mut h_c = 0.0; // H(C) - entropy of true labels
        for &count in true_counts.values() {
            if count > 0.0 {
                let p = count / n;
                h_c -= p * p.ln();
            }
        }

        let mut h_k = 0.0; // H(K) - entropy of predicted labels
        for &count in pred_counts.values() {
            if count > 0.0 {
                let p = count / n;
                h_k -= p * p.ln();
            }
        }

        let homogeneity = if h_c > 1e-10 { 1.0 - h_c_k / h_c } else { 1.0 };
        let completeness = if h_k > 1e-10 { 1.0 - h_k_c / h_k } else { 1.0 };

        (homogeneity, completeness)
    }
}

impl Metric for VMeasure {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_score(targets, predictions)
    }

    fn name(&self) -> &str {
        "v_measure"
    }
}

/// Inertia (Within-cluster sum of squares) - lower is better
pub struct Inertia;

impl Inertia {
    /// Compute inertia
    pub fn compute_score(&self, data: &Tensor, labels: &Tensor, centroids: Option<&Tensor>) -> f64 {
        match (data.to_vec(), labels.to_vec()) {
            (Ok(data_vec), Ok(labels_vec)) => {
                let shape = data.shape();
                let dims = shape.dims();

                if dims.len() != 2 || dims[0] != labels_vec.len() {
                    return f64::INFINITY;
                }

                let n_samples = dims[0];
                let n_features = dims[1];

                let data_array = Array2::from_shape_vec(
                    (n_samples, n_features),
                    data_vec.iter().map(|&x| x as f64).collect(),
                )
                .unwrap_or_else(|_| Array2::zeros((n_samples, n_features)));

                let labels_array = Array1::from_vec(labels_vec.iter().map(|&x| x as i32).collect());

                if let Some(centroids_tensor) = centroids {
                    if let Ok(centroids_vec) = centroids_tensor.to_vec() {
                        let centroids_shape = centroids_tensor.shape();
                        let centroids_dims = centroids_shape.dims();

                        if centroids_dims.len() == 2 && centroids_dims[1] == n_features {
                            let centroids_array = Array2::from_shape_vec(
                                (centroids_dims[0], centroids_dims[1]),
                                centroids_vec.iter().map(|&x| x as f64).collect(),
                            )
                            .expect("centroids array should have valid shape");
                            return self.compute_inertia_with_centroids(
                                &data_array,
                                &labels_array,
                                &centroids_array,
                            );
                        }
                    }
                }

                self.compute_inertia_without_centroids(&data_array, &labels_array)
            }
            _ => f64::INFINITY,
        }
    }

    fn compute_inertia_without_centroids(&self, data: &Array2<f64>, labels: &Array1<i32>) -> f64 {
        // Group data by clusters and compute centroids
        let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            clusters.entry(label).or_default().push(i);
        }

        let mut total_inertia = 0.0;

        for (_, indices) in &clusters {
            if indices.is_empty() {
                continue;
            }

            // Calculate cluster centroid
            let mut centroid = Array1::<f64>::zeros(data.ncols());
            for &i in indices {
                for j in 0..data.ncols() {
                    centroid[j] += data[[i, j]];
                }
            }
            for val in centroid.iter_mut() {
                *val /= indices.len() as f64;
            }

            // Calculate sum of squared distances to centroid
            for &i in indices {
                let distance = euclidean_distance(data.row(i), centroid.view());
                total_inertia += distance * distance;
            }
        }

        total_inertia
    }

    fn compute_inertia_with_centroids(
        &self,
        data: &Array2<f64>,
        labels: &Array1<i32>,
        centroids: &Array2<f64>,
    ) -> f64 {
        let mut total_inertia = 0.0;

        for (i, &label) in labels.iter().enumerate() {
            if (label as usize) < centroids.nrows() {
                let centroid = centroids.row(label as usize);
                let distance = euclidean_distance(data.row(i), centroid.view());
                total_inertia += distance * distance;
            }
        }

        total_inertia
    }
}

impl Metric for Inertia {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        self.compute_score(predictions, targets, None)
    }

    fn name(&self) -> &str {
        "inertia"
    }
}
