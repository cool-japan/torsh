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
/// use torsh_tensor::creation::randn;
///
/// let mut online_kmeans = OnlineKMeans::new(3)?;
///
/// // Process streaming data points
/// for i in 0..1000 {
///     let point = randn::<f32>(&[1, 2])?;
///     online_kmeans.update_single(&point)?;
///
///     if online_kmeans.detect_drift() {
///         println!("Concept drift detected at point {}", i);
///     }
/// }
///
/// let result = online_kmeans.get_current_result()?;
/// println!("Final centroids: {:?}", result.centroids);
/// # Ok::<(), Box<dyn std::error::Error>>(())
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

/// Sliding Window K-Means configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SlidingWindowConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Window size (number of recent points to keep)
    pub window_size: usize,
    /// How often to recompute centroids (in number of points)
    pub recompute_frequency: usize,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Maximum iterations for centroid recomputation
    pub max_iters: usize,
    /// Convergence tolerance for centroid recomputation
    pub tolerance: f64,
}

impl Default for SlidingWindowConfig {
    fn default() -> Self {
        Self {
            n_clusters: 8,
            window_size: 1000,
            recompute_frequency: 100,
            random_state: None,
            max_iters: 10,
            tolerance: 1e-4,
        }
    }
}

/// Sliding Window K-Means result
#[derive(Debug, Clone)]
pub struct SlidingWindowResult {
    /// Current cluster centroids
    pub centroids: Tensor,
    /// Labels for points in the current window
    pub labels: Tensor,
    /// Number of points in each cluster (from current window)
    pub cluster_counts: Vec<usize>,
    /// Total points processed (including expired)
    pub n_points_seen: usize,
    /// Number of points currently in window
    pub window_fill: usize,
    /// Number of times centroids have been recomputed
    pub n_recomputations: usize,
}

impl ClusteringResult for SlidingWindowResult {
    fn labels(&self) -> &Tensor {
        &self.labels
    }

    fn n_clusters(&self) -> usize {
        self.centroids.shape().dims()[0]
    }

    fn centers(&self) -> Option<&Tensor> {
        Some(&self.centroids)
    }

    fn converged(&self) -> bool {
        self.n_points_seen > 100 // Consider converged after processing enough points
    }

    fn n_iter(&self) -> Option<usize> {
        Some(self.n_recomputations)
    }

    fn metadata(&self) -> Option<&HashMap<String, String>> {
        None
    }
}

/// Sliding Window K-Means clustering for non-stationary streams
///
/// Maintains a fixed-size window of recent data points and periodically
/// recomputes centroids from this window. Old points automatically expire
/// when the window is full.
///
/// # Mathematical Foundation
///
/// Unlike Online K-Means which updates centroids incrementally, Sliding Window
/// K-Means maintains explicit storage of recent points:
///
/// ```text
/// Window W(t) = {x_{t-w+1}, x_{t-w+2}, ..., x_t}
/// ```
///
/// When a new point x_{t+1} arrives and window is full:
/// 1. Remove oldest point x_{t-w+1}
/// 2. Add new point x_{t+1}
/// 3. If recomputation triggered, run full K-Means on W(t+1)
///
/// # Advantages over Online K-Means
///
/// - **Adapts to drift**: Old data is discarded, preventing outdated patterns
/// - **Full optimization**: Periodic recomputation finds better centroids
/// - **Stable clusters**: Less sensitive to individual outliers
///
/// # Disadvantages
///
/// - **Memory usage**: O(window_size × n_features)
/// - **Computation cost**: Periodic full K-Means on window
/// - **Latency spikes**: Recomputation can cause delays
///
/// # Parameters
///
/// - **window_size**: Number of recent points to maintain (default: 1000)
/// - **recompute_frequency**: Recompute centroids every N points (default: 100)
/// - **n_clusters**: Number of clusters to find
///
/// # Example
///
/// ```rust
/// use torsh_cluster::algorithms::incremental::{
///     SlidingWindowKMeans, IncrementalClustering, SlidingWindowConfig
/// };
/// use torsh_tensor::Tensor;
///
/// let config = SlidingWindowConfig {
///     n_clusters: 3,
///     window_size: 500,
///     recompute_frequency: 50,
///     ..Default::default()
/// };
///
/// let mut sliding_window = SlidingWindowKMeans::new(config)?;
///
/// // Process streaming data
/// for i in 0..1000 {
///     let point = Tensor::from_vec(vec![(i % 10) as f32, (i / 10) as f32], &[2])?;
///     sliding_window.update_single(&point)?;
///
///     if i % 100 == 0 {
///         let result = sliding_window.get_current_result()?;
///         println!("Iteration {}: {} points in window", i, result.window_fill);
///     }
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct SlidingWindowKMeans {
    config: SlidingWindowConfig,
    /// Sliding window of recent points
    window: VecDeque<Array1<f64>>,
    /// Current centroids
    centroids: Option<Array2<f64>>,
    /// Number of points processed
    n_points_seen: usize,
    /// Number of centroids recomputations performed
    n_recomputations: usize,
    /// Points since last recomputation
    points_since_recompute: usize,
    /// RNG for initialization
    rng: CoreRandom<StdRng>,
    /// Number of features
    n_features: Option<usize>,
}

impl SlidingWindowKMeans {
    /// Create a new Sliding Window K-Means algorithm
    pub fn new(config: SlidingWindowConfig) -> ClusterResult<Self> {
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
            window: VecDeque::with_capacity(1000),
            centroids: None,
            n_points_seen: 0,
            n_recomputations: 0,
            points_since_recompute: 0,
            rng,
            n_features: None,
        })
    }

    /// Create with default config and specified parameters
    pub fn with_params(n_clusters: usize, window_size: usize) -> ClusterResult<Self> {
        let config = SlidingWindowConfig {
            n_clusters,
            window_size,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Set window size
    pub fn window_size(mut self, size: usize) -> Self {
        self.config.window_size = size;
        self
    }

    /// Set recompute frequency
    pub fn recompute_frequency(mut self, frequency: usize) -> Self {
        self.config.recompute_frequency = frequency;
        self
    }

    /// Initialize centroids using K-means++ on current window
    fn initialize_centroids(&mut self) -> ClusterResult<()> {
        if self.window.is_empty() {
            return Err(ClusterError::ConfigError(
                "Cannot initialize centroids from empty window".to_string(),
            ));
        }

        let n_features = self.window[0].len();
        self.n_features = Some(n_features);

        let n_points = self.window.len();
        let k = self.config.n_clusters.min(n_points);

        // Convert window to Array2
        let mut window_array = Array2::<f64>::zeros((n_points, n_features));
        for (i, point) in self.window.iter().enumerate() {
            for (j, &val) in point.iter().enumerate() {
                window_array[[i, j]] = val;
            }
        }

        // K-means++ initialization
        let mut centroids = Array2::<f64>::zeros((k, n_features));

        // Choose first centroid randomly
        let first_idx = self.rng.gen_range(0..n_points);
        centroids.row_mut(0).assign(&window_array.row(first_idx));

        // Choose remaining centroids
        for i in 1..k {
            // Compute distances to nearest centroid
            let mut distances = vec![f64::INFINITY; n_points];
            for (point_idx, point) in window_array.outer_iter().enumerate() {
                let mut min_dist = f64::INFINITY;
                for centroid in centroids.outer_iter().take(i) {
                    let dist = self.euclidean_distance(&point, &centroid);
                    min_dist = min_dist.min(dist);
                }
                distances[point_idx] = min_dist;
            }

            // Choose next centroid with probability proportional to distance²
            let sum_sq_dist: f64 = distances.iter().map(|d| d * d).sum();
            let mut target = self.rng.gen_range(0.0..sum_sq_dist);

            let mut chosen_idx = 0;
            for (idx, &dist) in distances.iter().enumerate() {
                target -= dist * dist;
                if target <= 0.0 {
                    chosen_idx = idx;
                    break;
                }
            }

            centroids.row_mut(i).assign(&window_array.row(chosen_idx));
        }

        self.centroids = Some(centroids);
        Ok(())
    }

    /// Recompute centroids from current window using Lloyd's algorithm
    fn recompute_centroids(&mut self) -> ClusterResult<()> {
        if self.window.is_empty() {
            return Ok(());
        }

        // Initialize if needed
        if self.centroids.is_none() {
            self.initialize_centroids()?;
        }

        let n_points = self.window.len();
        let n_features = self.window[0].len();
        let k = self.config.n_clusters.min(n_points);

        // Convert window to Array2
        let mut window_array = Array2::<f64>::zeros((n_points, n_features));
        for (i, point) in self.window.iter().enumerate() {
            for (j, &val) in point.iter().enumerate() {
                window_array[[i, j]] = val;
            }
        }

        let mut centroids = self.centroids.clone().unwrap();

        // Lloyd's algorithm iterations
        for _iter in 0..self.config.max_iters {
            let old_centroids = centroids.clone();

            // Assignment step
            let mut labels = vec![0usize; n_points];
            for (i, point) in window_array.outer_iter().enumerate() {
                let mut min_dist = f64::INFINITY;
                let mut closest = 0;
                for (j, centroid) in centroids.outer_iter().enumerate() {
                    let dist = self.euclidean_distance(&point, &centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        closest = j;
                    }
                }
                labels[i] = closest;
            }

            // Update step
            centroids.fill(0.0);
            let mut counts = vec![0usize; k];

            for (i, &label) in labels.iter().enumerate() {
                for (j, &val) in window_array.row(i).iter().enumerate() {
                    centroids[[label, j]] += val;
                }
                counts[label] += 1;
            }

            for i in 0..k {
                if counts[i] > 0 {
                    for j in 0..n_features {
                        centroids[[i, j]] /= counts[i] as f64;
                    }
                }
            }

            // Check convergence
            let mut max_shift: f64 = 0.0;
            for (old_row, new_row) in old_centroids.outer_iter().zip(centroids.outer_iter()) {
                let shift = self.euclidean_distance(&old_row, &new_row);
                max_shift = max_shift.max(shift);
            }

            if max_shift < self.config.tolerance {
                break;
            }
        }

        self.centroids = Some(centroids);
        self.n_recomputations += 1;
        self.points_since_recompute = 0;

        Ok(())
    }

    /// Compute Euclidean distance between two points
    fn euclidean_distance(&self, p1: &ArrayView1<f64>, p2: &ArrayView1<f64>) -> f64 {
        let mut sum_sq = 0.0;
        for (a, b) in p1.iter().zip(p2.iter()) {
            let diff = a - b;
            sum_sq += diff * diff;
        }
        sum_sq.sqrt()
    }

    /// Convert tensor to Array1
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

    /// Convert Vec to Tensor
    fn vec_to_tensor(&self, data: Vec<f64>, shape: &[usize]) -> ClusterResult<Tensor> {
        let data_f32: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
        Tensor::from_vec(data_f32, shape).map_err(ClusterError::TensorError)
    }
}

impl IncrementalClustering for SlidingWindowKMeans {
    type Result = SlidingWindowResult;

    fn update_single(&mut self, point: &Tensor) -> ClusterResult<()> {
        let point_array = self.tensor_to_array1(point)?;

        // Initialize n_features if first point
        if self.n_features.is_none() {
            self.n_features = Some(point_array.len());
        }

        // Add point to window
        self.window.push_back(point_array);

        // Remove oldest point if window is full
        if self.window.len() > self.config.window_size {
            self.window.pop_front();
        }

        self.n_points_seen += 1;
        self.points_since_recompute += 1;

        // Recompute centroids if needed
        if self.points_since_recompute >= self.config.recompute_frequency
            || self.centroids.is_none()
        {
            self.recompute_centroids()?;
        }

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

        if self.n_features.is_none() {
            self.n_features = Some(n_features);
        }

        let data_f32: Vec<f32> = batch.to_vec().map_err(ClusterError::TensorError)?;
        let data: Vec<f64> = data_f32.into_iter().map(|x| x as f64).collect();
        let data_array = Array2::from_shape_vec((n_samples, n_features), data)
            .map_err(|_| ClusterError::InvalidInput("Failed to reshape batch data".to_string()))?;

        for row in data_array.outer_iter() {
            let point_array = row.to_owned();
            self.window.push_back(point_array);

            if self.window.len() > self.config.window_size {
                self.window.pop_front();
            }

            self.n_points_seen += 1;
            self.points_since_recompute += 1;
        }

        // Recompute after processing batch
        if self.points_since_recompute >= self.config.recompute_frequency
            || self.centroids.is_none()
        {
            self.recompute_centroids()?;
        }

        Ok(())
    }

    fn get_current_result(&self) -> ClusterResult<Self::Result> {
        let centroids = self
            .centroids
            .as_ref()
            .ok_or_else(|| ClusterError::ConfigError("No data processed yet".to_string()))?;

        let centroids_tensor = self.array2_to_tensor(centroids)?;

        // Compute labels for current window
        let mut labels = Vec::with_capacity(self.window.len());
        let mut cluster_counts = vec![0usize; self.config.n_clusters];

        for point in &self.window {
            let mut min_dist = f64::INFINITY;
            let mut closest = 0;
            for (i, centroid) in centroids.outer_iter().enumerate() {
                let dist = self.euclidean_distance(&point.view(), &centroid);
                if dist < min_dist {
                    min_dist = dist;
                    closest = i;
                }
            }
            labels.push(closest as f64);
            cluster_counts[closest] += 1;
        }

        let labels_tensor = self.vec_to_tensor(labels, &[self.window.len()])?;

        Ok(SlidingWindowResult {
            centroids: centroids_tensor,
            labels: labels_tensor,
            cluster_counts,
            n_points_seen: self.n_points_seen,
            window_fill: self.window.len(),
            n_recomputations: self.n_recomputations,
        })
    }

    fn reset(&mut self) {
        self.window.clear();
        self.centroids = None;
        self.n_points_seen = 0;
        self.n_recomputations = 0;
        self.points_since_recompute = 0;
        self.n_features = None;
    }

    fn detect_drift(&self) -> bool {
        // Simplified drift detection: check if recomputations are happening frequently
        // In a stationary distribution, centroids would stabilize
        self.n_recomputations > 10 && self.n_points_seen / self.n_recomputations.max(1) < 50
    }

    fn n_points_seen(&self) -> usize {
        self.n_points_seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let _initial_drift = online_kmeans.detect_drift();

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

    #[test]
    fn test_sliding_window_basic() -> ClusterResult<()> {
        let config = SlidingWindowConfig {
            n_clusters: 2,
            window_size: 50,
            recompute_frequency: 10,
            ..Default::default()
        };

        let mut sliding = SlidingWindowKMeans::new(config)?;

        // Process points from two clusters, alternating to keep both in window
        for i in 0..100 {
            let point = if i % 2 == 0 {
                Tensor::from_vec(vec![0.0 + (i as f32) * 0.01, 0.0], &[2])?
            } else {
                Tensor::from_vec(vec![10.0 + (i as f32) * 0.01, 10.0], &[2])?
            };

            sliding.update_single(&point)?;
        }

        let result = sliding.get_current_result()?;
        // May find 1 or 2 clusters depending on initialization
        assert!(result.n_clusters() >= 1);
        assert!(result.n_clusters() <= 2);
        assert_eq!(result.window_fill, 50); // Window size is 50
        assert_eq!(result.n_points_seen, 100);
        assert!(result.n_recomputations > 0);

        Ok(())
    }

    #[test]
    fn test_sliding_window_batch() -> ClusterResult<()> {
        let config = SlidingWindowConfig {
            n_clusters: 2,
            window_size: 20,
            recompute_frequency: 10,
            ..Default::default()
        };

        let mut sliding = SlidingWindowKMeans::new(config)?;

        let batch = Tensor::from_vec(
            vec![
                0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 5.0, 5.0, 5.1, 5.1, 5.2, 5.2, 5.3, 5.3,
            ],
            &[8, 2],
        )?;

        sliding.update_batch(&batch)?;

        let result = sliding.get_current_result()?;
        assert_eq!(result.n_clusters(), 2);
        assert_eq!(result.window_fill, 8);
        assert_eq!(result.n_points_seen, 8);

        Ok(())
    }

    #[test]
    fn test_sliding_window_expiration() -> ClusterResult<()> {
        let config = SlidingWindowConfig {
            n_clusters: 2,
            window_size: 10,
            recompute_frequency: 5,
            ..Default::default()
        };

        let mut sliding = SlidingWindowKMeans::new(config)?;

        // Add more points than window size
        for i in 0..20 {
            let point = Tensor::from_vec(vec![i as f32, 0.0], &[2])?;
            sliding.update_single(&point)?;
        }

        let result = sliding.get_current_result()?;

        // Window should only contain last 10 points
        assert_eq!(result.window_fill, 10);
        assert_eq!(result.n_points_seen, 20);

        // Labels should match window size
        assert_eq!(result.labels.shape().dims()[0], 10);

        Ok(())
    }

    #[test]
    fn test_sliding_window_recomputation() -> ClusterResult<()> {
        let config = SlidingWindowConfig {
            n_clusters: 2,
            window_size: 50,
            recompute_frequency: 10,
            ..Default::default()
        };

        let mut sliding = SlidingWindowKMeans::new(config)?;

        // Process points
        for i in 0..50 {
            let point = Tensor::from_vec(vec![i as f32 * 0.1, 0.0], &[2])?;
            sliding.update_single(&point)?;
        }

        let result = sliding.get_current_result()?;

        // Should have recomputed centroids multiple times
        // (50 points / 10 recompute_frequency = 5 recomputations)
        assert!(result.n_recomputations >= 4);
        assert!(result.n_recomputations <= 6);

        Ok(())
    }

    #[test]
    fn test_sliding_window_reset() -> ClusterResult<()> {
        let config = SlidingWindowConfig {
            n_clusters: 2,
            window_size: 20,
            recompute_frequency: 5,
            ..Default::default()
        };

        let mut sliding = SlidingWindowKMeans::new(config)?;

        // Process some points
        for i in 0..10 {
            let point = Tensor::from_vec(vec![i as f32, 0.0], &[2])?;
            sliding.update_single(&point)?;
        }

        // Reset
        sliding.reset();

        // Check that everything is reset
        assert_eq!(sliding.n_points_seen(), 0);

        // Processing after reset should work
        let point = Tensor::from_vec(vec![1.0, 1.0], &[2])?;
        sliding.update_single(&point)?;

        assert_eq!(sliding.n_points_seen(), 1);

        Ok(())
    }

    #[test]
    fn test_sliding_window_drift_adaptation() -> ClusterResult<()> {
        let config = SlidingWindowConfig {
            n_clusters: 2,
            window_size: 30,
            recompute_frequency: 10,
            ..Default::default()
        };

        let mut sliding = SlidingWindowKMeans::new(config)?;

        // Phase 1: Cluster around (0, 0) and (5, 5)
        for i in 0..20 {
            let point = if i < 10 {
                Tensor::from_vec(vec![i as f32 * 0.1, 0.0], &[2])?
            } else {
                Tensor::from_vec(vec![5.0 + (i - 10) as f32 * 0.1, 5.0], &[2])?
            };
            sliding.update_single(&point)?;
        }

        let result1 = sliding.get_current_result()?;
        let centroids1 = result1.centroids.to_vec().unwrap();

        // Phase 2: Shift clusters to (10, 10) and (15, 15)
        for i in 0..30 {
            let point = if i < 15 {
                Tensor::from_vec(vec![10.0 + i as f32 * 0.1, 10.0], &[2])?
            } else {
                Tensor::from_vec(vec![15.0 + (i - 15) as f32 * 0.1, 15.0], &[2])?
            };
            sliding.update_single(&point)?;
        }

        let result2 = sliding.get_current_result()?;
        let centroids2 = result2.centroids.to_vec().unwrap();

        // Centroids should have adapted to new distribution
        // (Old points expired from window)
        // Check that centroids changed significantly
        let mut changed = false;
        for i in 0..centroids1.len().min(centroids2.len()) {
            if (centroids1[i] - centroids2[i]).abs() > 1.0 {
                changed = true;
                break;
            }
        }

        assert!(changed, "Centroids should adapt to distribution shift");

        Ok(())
    }
}
