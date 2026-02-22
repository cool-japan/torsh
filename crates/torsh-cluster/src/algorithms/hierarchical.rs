//! Hierarchical clustering implementation

use crate::error::{ClusterError, ClusterResult};
use crate::traits::{
    ClusteringAlgorithm, ClusteringResult, Fit, FitPredict, HierarchicalClustering,
};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_tensor::Tensor;

/// Linkage criteria for hierarchical clustering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Linkage {
    Ward,
    Complete,
    Average,
    Single,
}

/// Hierarchical clustering result
#[derive(Debug, Clone)]
pub struct HierarchicalResult {
    pub labels: Tensor,
    pub n_clusters: usize,
    pub linkage_matrix: Option<Tensor>,
}

impl ClusteringResult for HierarchicalResult {
    fn labels(&self) -> &Tensor {
        &self.labels
    }

    fn n_clusters(&self) -> usize {
        self.n_clusters
    }
}

/// Agglomerative clustering
#[derive(Debug, Clone)]
pub struct AgglomerativeClustering {
    n_clusters: usize,
    linkage: Linkage,
    fitted: bool,
}

impl AgglomerativeClustering {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            linkage: Linkage::Ward,
            fitted: false,
        }
    }

    pub fn linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }
}

impl ClusteringAlgorithm for AgglomerativeClustering {
    fn name(&self) -> &str {
        "Agglomerative Clustering"
    }

    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("n_clusters".to_string(), self.n_clusters.to_string());
        params.insert("linkage".to_string(), format!("{:?}", self.linkage));
        params
    }

    fn set_params(&mut self, _params: HashMap<String, String>) -> ClusterResult<()> {
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl Fit for AgglomerativeClustering {
    type Result = HierarchicalResult;

    fn fit(&self, data: &Tensor) -> ClusterResult<Self::Result> {
        self.validate_input(data)?;

        // Convert tensor to array for processing
        let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;
        let shape = data.shape();
        let data_shape = shape.dims();

        if data_shape.len() != 2 {
            return Err(ClusterError::InvalidInput(
                "Data tensor must be 2-dimensional".to_string(),
            ));
        }

        let n_samples = data_shape[0];
        let n_features = data_shape[1];

        if self.n_clusters > n_samples {
            return Err(ClusterError::InvalidInput(
                "Number of clusters cannot exceed number of samples".to_string(),
            ));
        }

        let data_array =
            Array2::from_shape_vec((n_samples, n_features), data_vec).map_err(|e| {
                ClusterError::InvalidInput(format!("Failed to reshape data array: {}", e))
            })?;

        // Perform agglomerative clustering
        let labels = self.perform_agglomerative_clustering(&data_array)?;

        // Convert labels to tensor
        let labels_vec: Vec<f32> = labels.iter().map(|&x| x as f32).collect();
        let labels_tensor = Tensor::from_vec(labels_vec, &[n_samples])?;

        Ok(HierarchicalResult {
            labels: labels_tensor,
            n_clusters: self.n_clusters,
            linkage_matrix: None, // TODO: Implement linkage matrix if needed
        })
    }
}

impl AgglomerativeClustering {
    /// Perform agglomerative hierarchical clustering
    fn perform_agglomerative_clustering(&self, data: &Array2<f32>) -> ClusterResult<Vec<usize>> {
        let n_samples = data.nrows();

        // Initialize clusters - each point is its own cluster
        let mut clusters: Vec<Vec<usize>> = (0..n_samples).map(|i| vec![i]).collect();
        let mut cluster_labels = (0..n_samples).collect::<Vec<usize>>();

        // Compute initial distance matrix between all pairs of points
        let mut distance_matrix = self.compute_initial_distances(data)?;

        // Agglomerate until we have the desired number of clusters
        let mut current_n_clusters = n_samples;

        while current_n_clusters > self.n_clusters {
            // Find the two closest clusters
            let (cluster1_idx, cluster2_idx) = self.find_closest_clusters(&distance_matrix)?;

            // Merge the two closest clusters
            let merged_cluster = self.merge_clusters(&clusters, cluster1_idx, cluster2_idx);

            // Update cluster labels
            for &point_idx in &merged_cluster {
                cluster_labels[point_idx] = cluster1_idx;
            }

            // Update clusters list
            clusters[cluster1_idx] = merged_cluster;
            clusters.remove(cluster2_idx);

            // Update distance matrix
            self.update_distance_matrix(
                data,
                &mut distance_matrix,
                &clusters,
                cluster1_idx,
                cluster2_idx,
            )?;

            current_n_clusters -= 1;
        }

        // Assign final cluster labels (0 to n_clusters-1)
        let mut final_labels = vec![0; n_samples];
        for (cluster_id, cluster) in clusters.iter().enumerate() {
            for &point_idx in cluster {
                final_labels[point_idx] = cluster_id;
            }
        }

        Ok(final_labels)
    }

    /// Compute initial pairwise distances between all points
    fn compute_initial_distances(&self, data: &Array2<f32>) -> ClusterResult<Vec<Vec<f64>>> {
        let n_samples = data.nrows();
        let mut distances = vec![vec![0.0; n_samples]; n_samples];

        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let dist = self.euclidean_distance(&data.row(i), &data.row(j));
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }

        Ok(distances)
    }

    /// Compute Euclidean distance between two points
    fn euclidean_distance(&self, point1: &ArrayView1<f32>, point2: &ArrayView1<f32>) -> f64 {
        let mut sum_sq = 0.0_f64;
        for (&a, &b) in point1.iter().zip(point2.iter()) {
            let diff = a as f64 - b as f64;
            sum_sq += diff * diff;
        }
        sum_sq.sqrt()
    }

    /// Find the two closest clusters based on the distance matrix
    fn find_closest_clusters(&self, distance_matrix: &[Vec<f64>]) -> ClusterResult<(usize, usize)> {
        let n_clusters = distance_matrix.len();
        let mut min_distance = f64::INFINITY;
        let mut closest_pair = (0, 1);

        #[allow(clippy::needless_range_loop)]
        for i in 0..n_clusters {
            for j in (i + 1)..n_clusters {
                if distance_matrix[i][j] < min_distance {
                    min_distance = distance_matrix[i][j];
                    closest_pair = (i, j);
                }
            }
        }

        if min_distance == f64::INFINITY {
            return Err(ClusterError::InvalidInput(
                "Could not find closest clusters".to_string(),
            ));
        }

        Ok(closest_pair)
    }

    /// Merge two clusters
    fn merge_clusters(&self, clusters: &[Vec<usize>], idx1: usize, idx2: usize) -> Vec<usize> {
        let mut merged = clusters[idx1].clone();
        merged.extend_from_slice(&clusters[idx2]);
        merged
    }

    /// Update distance matrix after merging two clusters
    fn update_distance_matrix(
        &self,
        data: &Array2<f32>,
        distance_matrix: &mut Vec<Vec<f64>>,
        clusters: &[Vec<usize>],
        merged_idx: usize,
        removed_idx: usize,
    ) -> ClusterResult<()> {
        let n_clusters = clusters.len();

        // Update distances from the merged cluster to all other clusters
        for i in 0..n_clusters {
            if i != merged_idx {
                let new_distance =
                    self.compute_cluster_distance(data, &clusters[merged_idx], &clusters[i])?;

                distance_matrix[merged_idx][i] = new_distance;
                distance_matrix[i][merged_idx] = new_distance;
            }
        }

        // Remove the merged cluster row/column from distance matrix
        distance_matrix.remove(removed_idx);
        for row in distance_matrix.iter_mut() {
            row.remove(removed_idx);
        }

        Ok(())
    }

    /// Compute distance between two clusters based on linkage criterion
    fn compute_cluster_distance(
        &self,
        data: &Array2<f32>,
        cluster1: &[usize],
        cluster2: &[usize],
    ) -> ClusterResult<f64> {
        match self.linkage {
            Linkage::Single => {
                // Single linkage: minimum distance
                let mut min_dist = f64::INFINITY;
                for &i in cluster1 {
                    for &j in cluster2 {
                        let dist = self.euclidean_distance(&data.row(i), &data.row(j));
                        min_dist = min_dist.min(dist);
                    }
                }
                Ok(min_dist)
            }
            Linkage::Complete => {
                // Complete linkage: maximum distance
                let mut max_dist = 0.0_f64;
                for &i in cluster1 {
                    for &j in cluster2 {
                        let dist = self.euclidean_distance(&data.row(i), &data.row(j));
                        max_dist = max_dist.max(dist);
                    }
                }
                Ok(max_dist)
            }
            Linkage::Average => {
                // Average linkage: average distance
                let mut total_dist = 0.0;
                let mut count = 0;
                for &i in cluster1 {
                    for &j in cluster2 {
                        let dist = self.euclidean_distance(&data.row(i), &data.row(j));
                        total_dist += dist;
                        count += 1;
                    }
                }
                Ok(total_dist / count as f64)
            }
            Linkage::Ward => {
                // Ward linkage: increase in within-cluster sum of squares
                // For simplicity, we'll use centroid distance weighted by cluster sizes
                let centroid1 = self.compute_centroid(data, cluster1);
                let centroid2 = self.compute_centroid(data, cluster2);
                let centroid_dist = self.euclidean_distance_arrays(&centroid1, &centroid2);

                // Weight by cluster sizes (Ward criterion approximation)
                let n1 = cluster1.len() as f64;
                let n2 = cluster2.len() as f64;
                let weight = (n1 * n2) / (n1 + n2);

                Ok(weight * centroid_dist * centroid_dist)
            }
        }
    }

    /// Compute centroid of a cluster
    fn compute_centroid(&self, data: &Array2<f32>, cluster: &[usize]) -> Array1<f64> {
        let n_features = data.ncols();
        let mut centroid = Array1::zeros(n_features);

        for &point_idx in cluster {
            let point = data.row(point_idx);
            for (i, &value) in point.iter().enumerate() {
                centroid[i] += value as f64;
            }
        }

        let cluster_size = cluster.len() as f64;
        for value in centroid.iter_mut() {
            *value /= cluster_size;
        }

        centroid
    }

    /// Compute Euclidean distance between two Array1<f64>
    fn euclidean_distance_arrays(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        let mut sum_sq = 0.0;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let diff = x - y;
            sum_sq += diff * diff;
        }
        sum_sq.sqrt()
    }
}

impl FitPredict for AgglomerativeClustering {
    type Result = HierarchicalResult;

    fn fit_predict(&self, data: &Tensor) -> ClusterResult<Self::Result> {
        self.fit(data)
    }
}

impl HierarchicalClustering for AgglomerativeClustering {
    type Tree = Tensor;

    fn extract_flat_clustering(&self, _n_clusters: usize) -> ClusterResult<Tensor> {
        Err(ClusterError::NotImplemented(
            "extract_flat_clustering not yet implemented".to_string(),
        ))
    }

    fn extract_clustering_by_distance(&self, _threshold: f64) -> ClusterResult<Tensor> {
        Err(ClusterError::NotImplemented(
            "extract_clustering_by_distance not yet implemented".to_string(),
        ))
    }
}
