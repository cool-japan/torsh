//! Hierarchical clustering implementation

use crate::error::{ClusterError, ClusterResult};
use crate::traits::{
    ClusteringAlgorithm, ClusteringResult, Fit, FitPredict, HierarchicalClustering,
};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
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

/// A single merge step in the agglomerative clustering dendrogram.
///
/// Each row describes one merge: the two cluster indices that were merged,
/// the distance at which they were merged, and the size of the resulting
/// cluster.  This is the same layout used by scipy's `linkage` matrix.
#[derive(Debug, Clone)]
pub struct LinkageStep {
    /// Index of the first cluster being merged
    pub cluster_a: usize,
    /// Index of the second cluster being merged
    pub cluster_b: usize,
    /// Distance between the two clusters at the time of merge
    pub distance: f64,
    /// Total number of original observations in the merged cluster
    pub size: usize,
}

/// Hierarchical clustering result
#[derive(Debug, Clone)]
pub struct HierarchicalResult {
    pub labels: Tensor,
    pub n_clusters: usize,
    /// Linkage matrix in tensor form: shape [n_merges, 4] where columns are
    /// [cluster_a, cluster_b, distance, size]
    pub linkage_matrix: Option<Tensor>,
    /// Full merge history for dendrogram extraction
    pub(crate) merge_history: Vec<LinkageStep>,
    /// Total number of original data samples
    pub(crate) n_samples: usize,
}

impl ClusteringResult for HierarchicalResult {
    fn labels(&self) -> &Tensor {
        &self.labels
    }

    fn n_clusters(&self) -> usize {
        self.n_clusters
    }
}

impl HierarchicalResult {
    /// Extract flat cluster labels for a target number of clusters.
    ///
    /// Replays the merge history in reverse to cut the dendrogram at the
    /// right level.  The returned tensor has the same length as the original
    /// data and contains cluster IDs in `0..target_n_clusters`.
    pub fn flat_clusters(&self, target_n_clusters: usize) -> ClusterResult<Tensor> {
        if target_n_clusters == 0 {
            return Err(ClusterError::InvalidInput(
                "target_n_clusters must be at least 1".to_string(),
            ));
        }
        if target_n_clusters > self.n_samples {
            return Err(ClusterError::InvalidInput(format!(
                "target_n_clusters ({}) exceeds number of samples ({})",
                target_n_clusters, self.n_samples
            )));
        }

        // Walk the merge history bottom-up, stopping when we have
        // target_n_clusters groups.
        let n_merges_to_apply = self.n_samples.saturating_sub(target_n_clusters);
        let steps = &self.merge_history[..n_merges_to_apply.min(self.merge_history.len())];

        // union-find for sample → cluster mapping
        let mut parent: Vec<usize> = (0..self.n_samples).collect();

        // Counter so that each new merged cluster gets a fresh synthetic ID.
        // We use n_samples + i as the ID for the i-th merge (matches scipy convention).
        for (i, step) in steps.iter().enumerate() {
            let new_id = self.n_samples + i;
            // Re-root all members of cluster_a and cluster_b to new_id
            let a_root = find_root(&mut parent, step.cluster_a);
            let b_root = find_root(&mut parent, step.cluster_b);
            parent[a_root] = new_id;
            parent[b_root] = new_id;
            // The new node points to itself
            if new_id >= parent.len() {
                parent.resize(new_id + 1, new_id);
            } else {
                parent[new_id] = new_id;
            }
        }

        // Collect the root of each original sample
        let mut roots: Vec<usize> = (0..self.n_samples)
            .map(|i| find_root(&mut parent, i))
            .collect();

        // Re-label roots to consecutive integers 0..k
        let mut root_map: HashMap<usize, usize> = HashMap::new();
        let mut next_label = 0usize;
        for r in &mut roots {
            let label = *root_map.entry(*r).or_insert_with(|| {
                let l = next_label;
                next_label += 1;
                l
            });
            *r = label;
        }

        let data: Vec<f32> = roots.iter().map(|&l| l as f32).collect();
        Tensor::from_vec(data, &[self.n_samples]).map_err(ClusterError::TensorError)
    }

    /// Extract flat cluster labels by cutting the dendrogram at a distance threshold.
    ///
    /// All merges whose distance is **less than** `threshold` are applied;
    /// merges at or above the threshold are not.
    pub fn clusters_by_distance(&self, threshold: f64) -> ClusterResult<Tensor> {
        if threshold < 0.0 {
            return Err(ClusterError::InvalidInput(
                "distance threshold must be non-negative".to_string(),
            ));
        }

        // Apply only merges whose distance is strictly below the threshold
        let mut parent: Vec<usize> = (0..self.n_samples).collect();
        let mut next_synthetic = self.n_samples;

        for step in &self.merge_history {
            if step.distance >= threshold {
                break; // merge_history is in ascending distance order
            }
            let a_root = find_root(&mut parent, step.cluster_a);
            let b_root = find_root(&mut parent, step.cluster_b);
            let new_id = next_synthetic;
            next_synthetic += 1;
            if new_id >= parent.len() {
                parent.resize(new_id + 1, new_id);
            }
            parent[a_root] = new_id;
            parent[b_root] = new_id;
            parent[new_id] = new_id;
        }

        // Collect roots
        let mut roots: Vec<usize> = (0..self.n_samples)
            .map(|i| find_root(&mut parent, i))
            .collect();

        // Compact labels
        let mut root_map: HashMap<usize, usize> = HashMap::new();
        let mut next_label = 0usize;
        for r in &mut roots {
            let label = *root_map.entry(*r).or_insert_with(|| {
                let l = next_label;
                next_label += 1;
                l
            });
            *r = label;
        }

        let data: Vec<f32> = roots.iter().map(|&l| l as f32).collect();
        Tensor::from_vec(data, &[self.n_samples]).map_err(ClusterError::TensorError)
    }
}

/// Path-compressing union-find helper
fn find_root(parent: &mut Vec<usize>, mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]]; // path compression
        x = parent[x];
    }
    x
}

/// Agglomerative clustering
#[derive(Debug, Clone)]
pub struct AgglomerativeClustering {
    n_clusters: usize,
    linkage: Linkage,
    /// Interior-mutable cache of the last `fit` result so that the
    /// `HierarchicalClustering` trait methods (which take `&self`) can
    /// access the stored dendrogram.
    last_result: RefCell<Option<HierarchicalResult>>,
}

impl AgglomerativeClustering {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            linkage: Linkage::Ward,
            last_result: RefCell::new(None),
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
        self.last_result.borrow().is_some()
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

        // Perform agglomerative clustering, collecting the full merge history
        let (labels, merge_history) = self.perform_agglomerative_clustering(&data_array)?;

        // Convert labels to tensor
        let labels_vec: Vec<f32> = labels.iter().map(|&x| x as f32).collect();
        let labels_tensor = Tensor::from_vec(labels_vec, &[n_samples])?;

        // Build the linkage matrix tensor: shape [n_merges, 4]
        // Columns: [cluster_a, cluster_b, distance, size]
        let n_merges = merge_history.len();
        let linkage_matrix = if n_merges > 0 {
            let mut lm_data: Vec<f32> = Vec::with_capacity(n_merges * 4);
            for step in &merge_history {
                lm_data.push(step.cluster_a as f32);
                lm_data.push(step.cluster_b as f32);
                lm_data.push(step.distance as f32);
                lm_data.push(step.size as f32);
            }
            Some(Tensor::from_vec(lm_data, &[n_merges, 4])?)
        } else {
            None
        };

        let result = HierarchicalResult {
            labels: labels_tensor,
            n_clusters: self.n_clusters,
            linkage_matrix,
            merge_history,
            n_samples,
        };

        // Cache the result for `HierarchicalClustering` trait methods
        *self.last_result.borrow_mut() = Some(result.clone());

        Ok(result)
    }
}

impl AgglomerativeClustering {
    /// Perform agglomerative hierarchical clustering.
    ///
    /// Returns the final cluster labels for `n_clusters` clusters together with
    /// the **complete** merge history (all `n_samples - 1` merge steps in
    /// ascending distance order), which is needed to implement dendrogram
    /// extraction later.
    fn perform_agglomerative_clustering(
        &self,
        data: &Array2<f32>,
    ) -> ClusterResult<(Vec<usize>, Vec<LinkageStep>)> {
        let n_samples = data.nrows();

        // Initialize clusters - each point is its own cluster
        // We use virtual cluster IDs: originals are 0..n_samples; each merge
        // creates a new synthetic node starting at n_samples.
        let mut clusters: Vec<Vec<usize>> = (0..n_samples).map(|i| vec![i]).collect();
        // Maps cluster index in `clusters` vec → virtual node ID
        let mut cluster_ids: Vec<usize> = (0..n_samples).collect();

        // Compute initial distance matrix between all pairs of points
        let mut distance_matrix = self.compute_initial_distances(data)?;

        // Record every merge, including those beyond n_clusters, so we have the
        // full dendrogram for extraction.
        let mut merge_history: Vec<LinkageStep> = Vec::with_capacity(n_samples.saturating_sub(1));
        let mut next_synthetic_id = n_samples;

        // We must run until 1 cluster to build the full dendrogram.
        // We record the index where we reach n_clusters so we can derive labels.
        let mut labels_at_n_clusters: Option<Vec<usize>> = None;

        let mut current_n_clusters = n_samples;

        while current_n_clusters > 1 {
            // Find the two closest clusters.  By convention, cluster1_idx < cluster2_idx.
            let (raw_a, raw_b) = self.find_closest_clusters(&distance_matrix)?;
            // Ensure cluster1_idx always refers to the lower-indexed slot so that
            // removal of cluster2_idx never invalidates cluster1_idx.
            let (cluster1_idx, cluster2_idx) = if raw_a < raw_b {
                (raw_a, raw_b)
            } else {
                (raw_b, raw_a)
            };

            let merge_distance = distance_matrix[cluster1_idx][cluster2_idx];

            // Record the merge step using virtual IDs (before any modification).
            let merged_size = clusters[cluster1_idx].len() + clusters[cluster2_idx].len();
            merge_history.push(LinkageStep {
                cluster_a: cluster_ids[cluster1_idx],
                cluster_b: cluster_ids[cluster2_idx],
                distance: merge_distance,
                size: merged_size,
            });

            // Build the new merged cluster contents.
            let merged_cluster = self.merge_clusters(&clusters, cluster1_idx, cluster2_idx);

            // --- Step 1: update distance matrix BEFORE modifying clusters/ids ---
            // Recompute the distances from cluster1_idx (future merged cluster) to
            // every other current cluster.  cluster2_idx is being absorbed so we
            // skip it; the row/col will be deleted in the next step.
            let n_current = clusters.len(); // before removal
            for i in 0..n_current {
                if i == cluster1_idx || i == cluster2_idx {
                    continue;
                }
                // Compute distance between the merged content and cluster[i].
                // We use `merged_cluster` for the first argument because it
                // already holds the union of cluster1 and cluster2 contents.
                let new_dist =
                    self.compute_cluster_distance(data, &merged_cluster, &clusters[i])?;
                distance_matrix[cluster1_idx][i] = new_dist;
                distance_matrix[i][cluster1_idx] = new_dist;
            }

            // --- Step 2: Remove the cluster2 row/col from the distance matrix ---
            distance_matrix.remove(cluster2_idx);
            for row in distance_matrix.iter_mut() {
                row.remove(cluster2_idx);
            }

            // --- Step 3: Update cluster bookkeeping ---
            clusters[cluster1_idx] = merged_cluster;
            cluster_ids[cluster1_idx] = next_synthetic_id;
            next_synthetic_id += 1;

            clusters.remove(cluster2_idx);
            cluster_ids.remove(cluster2_idx);

            current_n_clusters -= 1;

            // Snapshot labels when we first reach the desired number of clusters.
            if current_n_clusters == self.n_clusters && labels_at_n_clusters.is_none() {
                let mut snapshot = vec![0usize; n_samples];
                for (cluster_id, cluster) in clusters.iter().enumerate() {
                    for &point_idx in cluster {
                        snapshot[point_idx] = cluster_id;
                    }
                }
                labels_at_n_clusters = Some(snapshot);
            }
        }

        // If n_clusters == 1 we need to snapshot now
        if labels_at_n_clusters.is_none() {
            let mut snapshot = vec![0usize; n_samples];
            for (cluster_id, cluster) in clusters.iter().enumerate() {
                for &point_idx in cluster {
                    snapshot[point_idx] = cluster_id;
                }
            }
            labels_at_n_clusters = Some(snapshot);
        }

        let final_labels = labels_at_n_clusters.unwrap_or_else(|| vec![0; n_samples]);

        Ok((final_labels, merge_history))
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

    fn get_tree(&self) -> Option<&Self::Tree> {
        // We cannot return a reference to the interior of a RefCell safely.
        // Callers that need the linkage matrix should use the `HierarchicalResult`
        // returned from `fit()` directly.
        None
    }

    /// Extract flat cluster labels for `n_clusters` clusters.
    ///
    /// Requires `fit()` to have been called first; returns an error otherwise.
    fn extract_flat_clustering(&self, n_clusters: usize) -> ClusterResult<Tensor> {
        let borrow = self.last_result.borrow();
        let result = borrow.as_ref().ok_or_else(|| {
            ClusterError::InvalidInput(
                "extract_flat_clustering requires calling fit() first".to_string(),
            )
        })?;
        result.flat_clusters(n_clusters)
    }

    /// Extract flat cluster labels by cutting the dendrogram at `threshold`.
    ///
    /// Requires `fit()` to have been called first; returns an error otherwise.
    fn extract_clustering_by_distance(&self, threshold: f64) -> ClusterResult<Tensor> {
        let borrow = self.last_result.borrow();
        let result = borrow.as_ref().ok_or_else(|| {
            ClusterError::InvalidInput(
                "extract_clustering_by_distance requires calling fit() first".to_string(),
            )
        })?;
        result.clusters_by_distance(threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{Fit, HierarchicalClustering};

    fn make_data() -> Tensor {
        // Two clearly separated clusters: points near (0,0) and near (10,10)
        Tensor::from_vec(
            vec![
                0.0_f32, 0.0, 0.1, 0.1, 0.2, 0.0, // cluster 0
                10.0, 10.0, 10.1, 10.1, 10.2, 10.0, // cluster 1
            ],
            &[6, 2],
        )
        .expect("test data should be valid")
    }

    #[test]
    fn test_fit_produces_two_clusters() {
        let data = make_data();
        let model = AgglomerativeClustering::new(2);
        let result = model.fit(&data).expect("fit should succeed");
        assert_eq!(result.n_clusters, 2);
        let labels = result
            .labels
            .to_vec()
            .expect("labels tensor should be readable");
        // First 3 points belong to one cluster, last 3 to the other
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_fit_builds_linkage_matrix() {
        let data = make_data();
        // Use Single linkage which is guaranteed to be monotone
        let model = AgglomerativeClustering::new(2).linkage(Linkage::Single);
        let result = model.fit(&data).expect("fit should succeed");
        // 6 samples → 5 merges total
        assert_eq!(result.merge_history.len(), 5);
        // Linkage tensor shape [5, 4]
        let lm = result
            .linkage_matrix
            .expect("linkage matrix should be Some");
        assert_eq!(lm.shape().dims(), &[5, 4]);
        // Single linkage distances are monotone non-decreasing
        let lm_vec = lm.to_vec().expect("lm to_vec should work");
        let distances: Vec<f32> = lm_vec.chunks(4).map(|c| c[2]).collect();
        for w in distances.windows(2) {
            assert!(
                w[0] <= w[1] + 1e-4,
                "Single linkage distances should be non-decreasing: {} > {}",
                w[0],
                w[1]
            );
        }
        // Sizes should be positive and the last merge should cover all 6 samples
        let sizes: Vec<f32> = lm_vec.chunks(4).map(|c| c[3]).collect();
        assert_eq!(
            sizes[4] as usize, 6,
            "final merge should cover all 6 samples"
        );
    }

    #[test]
    fn test_extract_flat_clustering_without_fit_errors() {
        let model = AgglomerativeClustering::new(2);
        let err = model.extract_flat_clustering(2);
        assert!(err.is_err(), "should error when not fitted");
    }

    #[test]
    fn test_extract_flat_clustering_after_fit() {
        let data = make_data();
        let model = AgglomerativeClustering::new(1); // fit with 1 cluster
        model.fit(&data).expect("fit should succeed");
        // Re-cut the same dendrogram to get 2 clusters
        let labels_tensor = model
            .extract_flat_clustering(2)
            .expect("extract should succeed");
        let labels = labels_tensor.to_vec().expect("labels to_vec should work");
        assert_eq!(labels.len(), 6);
        // Same cluster membership expectations as the 2-cluster fit
        assert_eq!(labels[0], labels[1]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_extract_clustering_by_distance() {
        let data = make_data();
        let model = AgglomerativeClustering::new(1);
        model.fit(&data).expect("fit should succeed");
        // A threshold of 5.0 should separate the two far-apart clusters
        let labels_tensor = model
            .extract_clustering_by_distance(5.0)
            .expect("extract by distance should succeed");
        let labels = labels_tensor.to_vec().expect("to_vec should work");
        assert_eq!(labels.len(), 6);
        assert_ne!(
            labels[0], labels[5],
            "two clusters expected at threshold 5.0"
        );
    }

    #[test]
    fn test_linkage_single() {
        let data = make_data();
        let model = AgglomerativeClustering::new(2).linkage(Linkage::Single);
        let result = model.fit(&data).expect("single linkage fit should succeed");
        assert_eq!(result.n_clusters, 2);
    }

    #[test]
    fn test_linkage_complete() {
        let data = make_data();
        let model = AgglomerativeClustering::new(2).linkage(Linkage::Complete);
        let result = model
            .fit(&data)
            .expect("complete linkage fit should succeed");
        assert_eq!(result.n_clusters, 2);
    }
}
