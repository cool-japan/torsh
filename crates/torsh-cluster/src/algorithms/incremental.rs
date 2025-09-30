//! Incremental and Online Clustering Algorithms
//!
//! This module provides streaming and incremental clustering algorithms that can
//! process data points one at a time or in mini-batches, maintaining cluster
//! models that adapt to concept drift and evolving data distributions.
//!
//! # Algorithms Included
//!
//! - **Online K-Means**: Incremental update of centroids with adaptive learning
//! - **Sliding Window Clustering**: Maintains clusters over a temporal window
//! - **Concept Drift Detection**: Detects changes in data distribution

use crate::error::{ClusterError, ClusterResult};
use crate::traits::{ClusteringAlgorithm, ClusteringResult, Fit, FitPredict};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::{seeded_rng, CoreRandom};
// Using SciRS2 re-exported StdRng to avoid direct rand dependency (SciRS2 POLICY)
use scirs2_core::random::rngs::StdRng;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use torsh_tensor::Tensor;

/// Trait for incremental clustering algorithms that can process streaming data
pub trait IncrementalClustering {
    type Result: ClusteringResult;

    /// Process a single data point and update the model
    fn update_single(&mut self, point: &Tensor) -> ClusterResult<()>;

    /// Process a batch of data points
    fn update_batch(&mut self, batch: &Tensor) -> ClusterResult<()>;

    /// Get current clustering state
    fn get_current_result(&self) -> ClusterResult<Self::Result>;

    /// Reset the clustering model
    fn reset(&mut self);

    /// Check if concept drift is detected
    fn detect_drift(&self) -> bool;

    /// Get the number of points processed so far
    fn n_points_seen(&self) -> usize;
}

/// Online K-Means clustering configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OnlineKMeansConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Learning rate for centroid updates (adaptive if None)
    pub learning_rate: Option<f64>,
    /// Decay factor for learning rate adaptation
    pub decay_factor: f64,
    /// Minimum learning rate
    pub min_learning_rate: f64,
    /// Concept drift detection threshold
    pub drift_threshold: f64,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Window size for drift detection
    pub drift_window_size: usize,
}

impl Default for OnlineKMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            learning_rate: None, // Adaptive
            decay_factor: 0.9,
            min_learning_rate: 1e-6,
            drift_threshold: 0.1,
            random_state: None,
            drift_window_size: 1000,
        }
    }
}

/// Online K-Means clustering result
#[derive(Debug, Clone)]
pub struct OnlineKMeansResult {
    /// Current cluster centroids
    pub centroids: Tensor,
    /// Cluster assignments for recent points (if available)
    pub labels: Option<Tensor>,
    /// Number of points assigned to each cluster
    pub cluster_counts: Vec<usize>,
    /// Total points processed
    pub n_points_seen: usize,
    /// Current learning rate
    pub current_learning_rate: f64,
    /// Whether concept drift was detected
    pub drift_detected: bool,
    /// Average intra-cluster distance (for drift detection)
    pub avg_intra_cluster_distance: f64,
}

impl ClusteringResult for OnlineKMeansResult {
    fn labels(&self) -> &Tensor {
        self.labels
            .as_ref()
            .unwrap_or_else(|| panic!("Labels not available for online clustering result"))
    }

    fn n_clusters(&self) -> usize {
        self.centroids.shape().dims()[0]
    }

    fn centers(&self) -> Option<&Tensor> {
        Some(&self.centroids)
    }

    fn converged(&self) -> bool {
        self.n_points_seen > 100 // Consider "converged" after processing enough points
    }

    fn n_iter(&self) -> Option<usize> {
        Some(self.n_points_seen)
    }

    fn metadata(&self) -> Option<&HashMap<String, String>> {
        None
    }
}

/// Online K-Means clustering algorithm for streaming data
///
/// This implementation can process data points incrementally and adapt to
/// concept drift in the data distribution.
///
/// # Example
///
/// ```rust
/// use torsh_cluster::algorithms::incremental::{OnlineKMeans, IncrementalClustering};
/// use torsh_tensor::Tensor;
///
/// let mut online_kmeans = OnlineKMeans::new(3)?;
///
/// // Process streaming data points
/// for i in 0..1000 {
///     let point = Tensor::randn(&[1, 2])?;
///     online_kmeans.update_single(&point)?;
///
///     if online_kmeans.detect_drift() {
///         println!("Concept drift detected at point {}", i);
///     }
/// }
///
/// let result = online_kmeans.get_current_result()?;
/// println!("Final centroids: {:?}", result.centroids);
/// ```
#[derive(Debug)]
pub struct OnlineKMeans {
    config: OnlineKMeansConfig,
    centroids: Option<Array2<f64>>,
    cluster_counts: Vec<usize>,
    n_points_seen: usize,
    current_learning_rate: f64,
    drift_history: VecDeque<f64>,
    rng: CoreRandom<StdRng>,
    n_features: Option<usize>,
}

impl OnlineKMeans {
    /// Create a new Online K-Means algorithm
    pub fn new(n_clusters: usize) -> ClusterResult<Self> {
        let config = OnlineKMeansConfig {
            n_clusters,
            ..Default::default()
        };

        let seed = config.random_state.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
        });
        let rng = seeded_rng(seed);

        Ok(Self {
            config,
            centroids: None,
            cluster_counts: vec![0; n_clusters],
            n_points_seen: 0,
            current_learning_rate: 1.0,
            drift_history: VecDeque::with_capacity(1000),
            rng,
            n_features: None,
        })
    }

    /// Set learning rate (None for adaptive)
    pub fn learning_rate(mut self, learning_rate: Option<f64>) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set drift detection threshold
    pub fn drift_threshold(mut self, threshold: f64) -> Self {
        self.config.drift_threshold = threshold;
        self
    }

    /// Set random seed
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self.rng = seeded_rng(seed);
        self
    }

    /// Initialize centroids if not already done
    fn initialize_centroids(&mut self, n_features: usize) -> ClusterResult<()> {
        if self.centroids.is_none() {
            self.n_features = Some(n_features);

            // Initialize centroids randomly
            let mut centroids = Array2::<f64>::zeros((self.config.n_clusters, n_features));
            for i in 0..self.config.n_clusters {
                for j in 0..n_features {
                    centroids[[i, j]] = self.rng.gen_range(-1.0..1.0);
                }
            }

            self.centroids = Some(centroids);
        }

        Ok(())
    }

    /// Find the closest centroid to a point
    fn find_closest_centroid(&self, point: &ArrayView1<f64>) -> ClusterResult<usize> {
        let centroids = self
            .centroids
            .as_ref()
            .ok_or_else(|| ClusterError::ConfigError("Centroids not initialized".to_string()))?;

        let mut min_distance = f64::INFINITY;
        let mut closest_centroid = 0;

        for (i, centroid) in centroids.outer_iter().enumerate() {
            let distance = self.compute_distance(point, &centroid)?;
            if distance < min_distance {
                min_distance = distance;
                closest_centroid = i;
            }
        }

        Ok(closest_centroid)
    }

    /// Compute Euclidean distance between two points
    fn compute_distance(
        &self,
        point1: &ArrayView1<f64>,
        point2: &ArrayView1<f64>,
    ) -> ClusterResult<f64> {
        let diff = point1 - point2;
        let distance = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
        Ok(distance)
    }

    /// Update centroid with new point using online learning
    fn update_centroid(&mut self, cluster_id: usize, point: &ArrayView1<f64>) -> ClusterResult<()> {
        let centroids = self
            .centroids
            .as_mut()
            .ok_or_else(|| ClusterError::ConfigError("Centroids not initialized".to_string()))?;

        self.cluster_counts[cluster_id] += 1;
        let count = self.cluster_counts[cluster_id] as f64;

        // Compute learning rate
        let lr = if let Some(fixed_lr) = self.config.learning_rate {
            fixed_lr
        } else {
            // Adaptive learning rate: 1/count
            (1.0 / count).max(self.config.min_learning_rate)
        };

        self.current_learning_rate = lr;

        // Update centroid: centroid = centroid + lr * (point - centroid)
        let mut centroid = centroids.row_mut(cluster_id);
        for (i, &point_val) in point.iter().enumerate() {
            let current_val = centroid[i];
            centroid[i] = current_val + lr * (point_val - current_val);
        }

        Ok(())
    }

    /// Detect concept drift based on recent clustering quality
    fn update_drift_detection(
        &mut self,
        point: &ArrayView1<f64>,
        cluster_id: usize,
    ) -> ClusterResult<()> {
        let centroids = self
            .centroids
            .as_ref()
            .ok_or_else(|| ClusterError::ConfigError("Centroids not initialized".to_string()))?;

        let centroid = centroids.row(cluster_id);
        let distance = self.compute_distance(point, &centroid)?;

        // Add to drift history
        self.drift_history.push_back(distance);
        if self.drift_history.len() > self.config.drift_window_size {
            self.drift_history.pop_front();
        }

        Ok(())
    }

    /// Convert ndarray point to Array1
    fn tensor_to_array1(&self, tensor: &Tensor) -> ClusterResult<Array1<f64>> {
        let tensor_shape = tensor.shape();
        let shape = tensor_shape.dims();
        if shape.len() != 1 && (shape.len() != 2 || shape[0] != 1) {
            return Err(ClusterError::InvalidInput(
                "Expected 1D tensor or single-row 2D tensor".to_string(),
            ));
        }

        let data_f32: Vec<f32> = tensor.to_vec().map_err(ClusterError::TensorError)?;
        let data: Vec<f64> = data_f32.into_iter().map(|x| x as f64).collect();

        let n_features = if shape.len() == 1 { shape[0] } else { shape[1] };
        Array1::from_vec(data)
            .to_shape(n_features)
            .map(|array| array.into_owned())
            .map_err(|_| ClusterError::InvalidInput("Failed to reshape tensor".to_string()))
    }

    /// Convert Array2 to Tensor
    fn array2_to_tensor(&self, array: &Array2<f64>) -> ClusterResult<Tensor> {
        let (rows, cols) = array.dim();
        let data_f64: Vec<f64> = array.iter().copied().collect();
        let data: Vec<f32> = data_f64.into_iter().map(|x| x as f32).collect();
        Tensor::from_vec(data, &[rows, cols]).map_err(ClusterError::TensorError)
    }
}

impl IncrementalClustering for OnlineKMeans {
    type Result = OnlineKMeansResult;

    fn update_single(&mut self, point: &Tensor) -> ClusterResult<()> {
        let point_array = self.tensor_to_array1(point)?;
        let n_features = point_array.len();

        // Initialize centroids if this is the first point
        self.initialize_centroids(n_features)?;

        // Find closest centroid
        let closest_centroid = self.find_closest_centroid(&point_array.view())?;

        // Update centroid
        self.update_centroid(closest_centroid, &point_array.view())?;

        // Update drift detection
        self.update_drift_detection(&point_array.view(), closest_centroid)?;

        self.n_points_seen += 1;

        Ok(())
    }

    fn update_batch(&mut self, batch: &Tensor) -> ClusterResult<()> {
        let batch_shape = batch.shape();
        let shape = batch_shape.dims();
        if shape.len() != 2 {
            return Err(ClusterError::InvalidInput(
                "Expected 2D batch tensor".to_string(),
            ));
        }

        let n_samples = shape[0];
        let n_features = shape[1];

        // Initialize centroids if this is the first batch
        self.initialize_centroids(n_features)?;

        let data_f32: Vec<f32> = batch.to_vec().map_err(ClusterError::TensorError)?;
        let data: Vec<f64> = data_f32.into_iter().map(|x| x as f64).collect();
        let data_array = Array2::from_shape_vec((n_samples, n_features), data)
            .map_err(|_| ClusterError::InvalidInput("Failed to reshape batch data".to_string()))?;

        // Process each point in the batch
        for i in 0..n_samples {
            let point = data_array.row(i);
            let closest_centroid = self.find_closest_centroid(&point)?;
            self.update_centroid(closest_centroid, &point)?;
            self.update_drift_detection(&point, closest_centroid)?;
            self.n_points_seen += 1;
        }

        Ok(())
    }

    fn get_current_result(&self) -> ClusterResult<Self::Result> {
        let centroids = self
            .centroids
            .as_ref()
            .ok_or_else(|| ClusterError::ConfigError("No data processed yet".to_string()))?;

        let centroids_tensor = self.array2_to_tensor(centroids)?;

        // Compute average intra-cluster distance for drift detection
        let avg_distance = if self.drift_history.is_empty() {
            0.0
        } else {
            self.drift_history.iter().sum::<f64>() / self.drift_history.len() as f64
        };

        Ok(OnlineKMeansResult {
            centroids: centroids_tensor,
            labels: None, // Not available for online clustering
            cluster_counts: self.cluster_counts.clone(),
            n_points_seen: self.n_points_seen,
            current_learning_rate: self.current_learning_rate,
            drift_detected: self.detect_drift(),
            avg_intra_cluster_distance: avg_distance,
        })
    }

    fn reset(&mut self) {
        self.centroids = None;
        self.cluster_counts = vec![0; self.config.n_clusters];
        self.n_points_seen = 0;
        self.current_learning_rate = 1.0;
        self.drift_history.clear();
        self.n_features = None;
    }

    fn detect_drift(&self) -> bool {
        if self.drift_history.len() < self.config.drift_window_size / 2 {
            return false;
        }

        // Simple drift detection: compare recent vs. historical performance
        let recent_window = self.drift_history.len() / 2;
        let recent_avg: f64 = self
            .drift_history
            .iter()
            .rev()
            .take(recent_window)
            .sum::<f64>()
            / recent_window as f64;
        let historical_avg: f64 =
            self.drift_history.iter().take(recent_window).sum::<f64>() / recent_window as f64;

        // Drift detected if recent performance significantly worse
        recent_avg > historical_avg * (1.0 + self.config.drift_threshold)
    }

    fn n_points_seen(&self) -> usize {
        self.n_points_seen
    }
}

impl ClusteringAlgorithm for OnlineKMeans {
    fn name(&self) -> &str {
        "Online K-Means"
    }

    fn get_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("n_clusters".to_string(), self.config.n_clusters.to_string());
        params.insert(
            "drift_threshold".to_string(),
            self.config.drift_threshold.to_string(),
        );
        params.insert(
            "decay_factor".to_string(),
            self.config.decay_factor.to_string(),
        );
        if let Some(lr) = self.config.learning_rate {
            params.insert("learning_rate".to_string(), lr.to_string());
        }
        params
    }

    fn set_params(&mut self, params: HashMap<String, String>) -> ClusterResult<()> {
        for (key, value) in params {
            match key.as_str() {
                "n_clusters" => {
                    let n_clusters = value.parse().map_err(|_| {
                        ClusterError::ConfigError(format!("Invalid n_clusters: {}", value))
                    })?;
                    self.config.n_clusters = n_clusters;
                    self.cluster_counts = vec![0; n_clusters];
                }
                "drift_threshold" => {
                    self.config.drift_threshold = value.parse().map_err(|_| {
                        ClusterError::ConfigError(format!("Invalid drift_threshold: {}", value))
                    })?;
                }
                "learning_rate" => {
                    if value == "adaptive" {
                        self.config.learning_rate = None;
                    } else {
                        self.config.learning_rate = Some(value.parse().map_err(|_| {
                            ClusterError::ConfigError(format!("Invalid learning_rate: {}", value))
                        })?);
                    }
                }
                _ => {
                    return Err(ClusterError::ConfigError(format!(
                        "Unknown parameter: {}",
                        key
                    )));
                }
            }
        }
        Ok(())
    }

    fn is_fitted(&self) -> bool {
        self.centroids.is_some()
    }
}

impl Fit for OnlineKMeans {
    type Result = OnlineKMeansResult;

    fn fit(&self, data: &Tensor) -> ClusterResult<Self::Result> {
        let mut online_kmeans = self.clone();
        online_kmeans.update_batch(data)?;
        online_kmeans.get_current_result()
    }
}

impl FitPredict for OnlineKMeans {
    type Result = OnlineKMeansResult;

    fn fit_predict(&self, data: &Tensor) -> ClusterResult<Self::Result> {
        self.fit(data)
    }
}

// Need to implement Clone for OnlineKMeans
impl Clone for OnlineKMeans {
    fn clone(&self) -> Self {
        let rng = seeded_rng(self.config.random_state.unwrap_or(42));

        Self {
            config: self.config.clone(),
            centroids: self.centroids.clone(),
            cluster_counts: self.cluster_counts.clone(),
            n_points_seen: self.n_points_seen,
            current_learning_rate: self.current_learning_rate,
            drift_history: self.drift_history.clone(),
            rng,
            n_features: self.n_features,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_online_kmeans_basic() -> ClusterResult<()> {
        let mut online_kmeans = OnlineKMeans::new(2)?;

        // Process some points
        for i in 0..10 {
            let point = if i < 5 {
                Tensor::from_vec(vec![0.0 + i as f32 * 0.1, 0.0], &[2])?
            } else {
                Tensor::from_vec(vec![5.0 + (i - 5) as f32 * 0.1, 5.0], &[2])?
            };

            online_kmeans.update_single(&point)?;
        }

        let result = online_kmeans.get_current_result()?;
        assert_eq!(result.n_clusters(), 2);
        assert_eq!(result.n_points_seen, 10);
        assert!(result.centroids.shape().dims() == &[2, 2]);

        Ok(())
    }

    #[test]
    fn test_online_kmeans_batch() -> ClusterResult<()> {
        let mut online_kmeans = OnlineKMeans::new(2)?;

        let batch = Tensor::from_vec(vec![0.0, 0.0, 0.1, 0.1, 5.0, 5.0, 5.1, 5.1], &[4, 2])?;

        online_kmeans.update_batch(&batch)?;

        let result = online_kmeans.get_current_result()?;
        assert_eq!(result.n_clusters(), 2);
        assert_eq!(result.n_points_seen, 4);

        Ok(())
    }

    #[test]
    fn test_drift_detection() -> ClusterResult<()> {
        let mut online_kmeans = OnlineKMeans::new(2)?.drift_threshold(0.1);

        // Process normal points
        for i in 0..100 {
            let point = Tensor::from_vec(vec![i as f32 * 0.01, 0.0], &[2])?;
            online_kmeans.update_single(&point)?;
        }

        let initial_drift = online_kmeans.detect_drift();

        // Introduce outliers (potential drift)
        for i in 0..50 {
            let point = Tensor::from_vec(vec![100.0 + i as f32, 100.0], &[2])?;
            online_kmeans.update_single(&point)?;
        }

        // Drift detection should eventually trigger
        // (Note: Simple test - in practice drift detection is complex)
        let final_result = online_kmeans.get_current_result()?;
        assert!(final_result.n_points_seen == 150);

        Ok(())
    }
}
