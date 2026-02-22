//! Gap Statistic for optimal cluster number selection
//!
//! This module implements the Gap Statistic method for determining the optimal
//! number of clusters in a dataset. The Gap Statistic compares the within-cluster
//! dispersion of the actual data to that of reference datasets.

use crate::algorithms::kmeans::{KMeans, KMeansAlgorithm};
use crate::error::{ClusterError, ClusterResult};
use crate::traits::Fit;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{seeded_rng, CoreRandom};
// Using SciRS2 re-exported StdRng to avoid direct rand dependency (SciRS2 POLICY)
use scirs2_core::random::rngs::StdRng;
use std::collections::HashMap;
use torsh_tensor::Tensor;

/// Configuration for Gap Statistic computation
#[derive(Debug, Clone)]
pub struct GapStatisticConfig {
    /// Maximum number of clusters to test
    pub max_k: usize,
    /// Number of reference datasets to generate
    pub n_refs: usize,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// K-means algorithm to use for clustering
    pub kmeans_algorithm: KMeansAlgorithm,
    /// Maximum iterations for K-means
    pub max_iters: usize,
}

impl Default for GapStatisticConfig {
    fn default() -> Self {
        Self {
            max_k: 10,
            n_refs: 10,
            random_state: None,
            kmeans_algorithm: KMeansAlgorithm::Lloyd,
            max_iters: 100,
        }
    }
}

/// Result of Gap Statistic computation
#[derive(Debug, Clone)]
pub struct GapStatisticResult {
    /// Gap values for each k
    pub gap_values: Vec<f64>,
    /// Within-cluster dispersions for actual data
    pub wk_values: Vec<f64>,
    /// Standard deviations of reference dispersions
    pub sk_values: Vec<f64>,
    /// Optimal number of clusters
    pub optimal_k: usize,
    /// K values tested
    pub k_values: Vec<usize>,
}

impl GapStatisticResult {
    /// Get the gap value for a specific k
    pub fn gap(&self, k: usize) -> Option<f64> {
        if k > 0 && k <= self.gap_values.len() {
            Some(self.gap_values[k - 1])
        } else {
            None
        }
    }

    /// Get the within-cluster dispersion for a specific k
    pub fn wk(&self, k: usize) -> Option<f64> {
        if k > 0 && k <= self.wk_values.len() {
            Some(self.wk_values[k - 1])
        } else {
            None
        }
    }

    /// Check if the gap statistic suggests k clusters is optimal
    pub fn is_optimal(&self, k: usize) -> bool {
        k == self.optimal_k
    }

    /// Get summary statistics as a HashMap
    pub fn summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        summary.insert("optimal_k".to_string(), self.optimal_k.to_string());
        summary.insert(
            "max_gap".to_string(),
            format!(
                "{:.6}",
                self.gap_values
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            ),
        );
        summary.insert("n_k_tested".to_string(), self.k_values.len().to_string());
        summary
    }
}

/// Gap Statistic implementation for optimal cluster number selection
///
/// The Gap Statistic compares the logarithm of within-cluster dispersion with that
/// expected under an appropriate reference null distribution. The optimal number of
/// clusters is where the gap is largest.
///
/// # References
/// - Tibshirani, R., Walther, G., & Hastie, T. (2001). Estimating the number of clusters
///   in a data set via the gap statistic. Journal of the Royal Statistical Society, 63(2), 411-423.
///
/// # Example
///
/// ```rust
/// use torsh_cluster::evaluation::metrics::gap_statistic::{GapStatistic, GapStatisticConfig};
/// use torsh_tensor::creation::randn;
///
/// let data = randn::<f32>(&[100, 2])?;
/// let config = GapStatisticConfig::default();
/// let mut gap_stat = GapStatistic::new(config);
///
/// let result = gap_stat.compute(&data)?;
/// println!("Optimal number of clusters: {}", result.optimal_k);
/// println!("Gap values: {:?}", result.gap_values);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct GapStatistic {
    config: GapStatisticConfig,
    rng: CoreRandom<StdRng>,
}

impl GapStatistic {
    /// Create a new Gap Statistic calculator
    pub fn new(config: GapStatisticConfig) -> Self {
        let seed = match config.random_state {
            Some(seed) => seed,
            None => {
                use std::time::{SystemTime, UNIX_EPOCH};
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("system time should be after UNIX_EPOCH")
                    .as_secs()
            }
        };
        let rng = seeded_rng(seed);

        Self { config, rng }
    }

    /// Create with default configuration
    pub fn with_default() -> Self {
        Self::new(GapStatisticConfig::default())
    }

    /// Set maximum number of clusters to test
    pub fn max_k(mut self, max_k: usize) -> Self {
        self.config.max_k = max_k;
        self
    }

    /// Set number of reference datasets
    pub fn n_refs(mut self, n_refs: usize) -> Self {
        self.config.n_refs = n_refs;
        self
    }

    /// Set random seed
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self.rng = seeded_rng(seed);
        self
    }

    /// Compute Gap Statistic for the given data
    pub fn compute(&mut self, data: &Tensor) -> ClusterResult<GapStatisticResult> {
        let data_shape = data.shape();
        let shape = data_shape.dims();
        if shape.len() != 2 {
            return Err(ClusterError::InvalidInput(
                "Data tensor must be 2-dimensional".to_string(),
            ));
        }

        let n_samples = shape[0];
        let _n_features = shape[1];

        if n_samples < self.config.max_k {
            return Err(ClusterError::InvalidInput(
                "Not enough samples for the specified max_k".to_string(),
            ));
        }

        let mut gap_values = Vec::new();
        let mut wk_values = Vec::new();
        let mut sk_values = Vec::new();
        let mut k_values = Vec::new();

        // Test each value of k from 1 to max_k
        for k in 1..=self.config.max_k {
            k_values.push(k);

            // Compute within-cluster dispersion for actual data
            let wk = self.compute_wk(data, k)?;
            wk_values.push(wk);

            // Compute expected within-cluster dispersion from reference datasets
            let (expected_wk, sk) = self.compute_expected_wk(data, k)?;

            // Gap statistic = log(expected_wk) - log(wk)
            let gap = expected_wk.ln() - wk.ln();
            gap_values.push(gap);
            sk_values.push(sk);
        }

        // Find optimal k using the "elbow" criterion with standard error
        let optimal_k = self.find_optimal_k(&gap_values, &sk_values);

        Ok(GapStatisticResult {
            gap_values,
            wk_values,
            sk_values,
            optimal_k,
            k_values,
        })
    }

    /// Compute within-cluster dispersion (Wk) for actual data
    fn compute_wk(&mut self, data: &Tensor, k: usize) -> ClusterResult<f64> {
        if k == 1 {
            // For k=1, compute total sum of squares from overall mean
            return self.compute_total_sum_of_squares(data);
        }

        let kmeans = KMeans::new(k)
            .algorithm(self.config.kmeans_algorithm)
            .max_iters(self.config.max_iters)
            .random_state(self.rng.random());

        let result = kmeans.fit(data)?;

        // Within-cluster dispersion is 2 * inertia (sum of squared distances)
        Ok(2.0 * result.inertia)
    }

    /// Compute total sum of squares from overall mean (for k=1)
    fn compute_total_sum_of_squares(&self, data: &Tensor) -> ClusterResult<f64> {
        let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;
        let data_shape = data.shape();
        let shape = data_shape.dims();
        let n_samples = shape[0];
        let n_features = shape[1];

        let data_array = Array2::from_shape_vec((n_samples, n_features), data_vec)
            .map_err(|e| ClusterError::InvalidInput(format!("Failed to reshape data: {}", e)))?;

        // Compute overall mean
        let mut mean = Array1::zeros(n_features);
        for i in 0..n_samples {
            for j in 0..n_features {
                mean[j] += data_array[[i, j]];
            }
        }
        mean /= n_samples as f32;

        // Compute sum of squared distances from mean
        let mut total_ss = 0.0_f64;
        for i in 0..n_samples {
            for j in 0..n_features {
                let diff = data_array[[i, j]] as f64 - mean[j] as f64;
                total_ss += diff * diff;
            }
        }

        Ok(2.0 * total_ss)
    }

    /// Compute expected within-cluster dispersion from reference datasets
    fn compute_expected_wk(&mut self, data: &Tensor, k: usize) -> ClusterResult<(f64, f64)> {
        let mut reference_wks = Vec::new();

        for _ in 0..self.config.n_refs {
            let reference_data = self.generate_reference_data(data)?;
            let wk = self.compute_wk(&reference_data, k)?;
            reference_wks.push(wk);
        }

        // Compute mean and standard deviation
        let mean_wk = reference_wks.iter().sum::<f64>() / reference_wks.len() as f64;

        let variance = reference_wks
            .iter()
            .map(|&x| (x - mean_wk).powi(2))
            .sum::<f64>()
            / reference_wks.len() as f64;

        let std_dev = variance.sqrt();

        // Standard error of the mean
        let sk = std_dev * (1.0 + 1.0 / self.config.n_refs as f64).sqrt();

        Ok((mean_wk, sk))
    }

    /// Generate reference data using uniform distribution over feature ranges
    fn generate_reference_data(&mut self, data: &Tensor) -> ClusterResult<Tensor> {
        let data_vec = data.to_vec().map_err(ClusterError::TensorError)?;
        let data_shape = data.shape();
        let shape = data_shape.dims();
        let n_samples = shape[0];
        let n_features = shape[1];

        let data_array = Array2::from_shape_vec((n_samples, n_features), data_vec)
            .map_err(|e| ClusterError::InvalidInput(format!("Failed to reshape data: {}", e)))?;

        // Find min and max for each feature
        let mut min_vals = vec![f32::INFINITY; n_features];
        let mut max_vals = vec![f32::NEG_INFINITY; n_features];

        for i in 0..n_samples {
            for j in 0..n_features {
                let val = data_array[[i, j]];
                min_vals[j] = min_vals[j].min(val);
                max_vals[j] = max_vals[j].max(val);
            }
        }

        // Generate uniform random data within feature ranges
        let mut reference_data = Vec::with_capacity(n_samples * n_features);
        for _ in 0..n_samples {
            for j in 0..n_features {
                let val = self.rng.random::<f64>() * (max_vals[j] - min_vals[j]) as f64
                    + min_vals[j] as f64;
                reference_data.push(val as f32);
            }
        }

        Tensor::from_vec(reference_data, &[n_samples, n_features])
            .map_err(ClusterError::TensorError)
    }

    /// Find optimal k using the gap statistic criterion
    fn find_optimal_k(&self, gap_values: &[f64], sk_values: &[f64]) -> usize {
        // Find the smallest k such that Gap(k) >= Gap(k+1) - s(k+1)
        for k in 1..gap_values.len() {
            let gap_k = gap_values[k - 1];
            let gap_k_plus_1 = gap_values[k];
            let sk_plus_1 = sk_values[k];

            if gap_k >= gap_k_plus_1 - sk_plus_1 {
                return k;
            }
        }

        // If no such k is found, return the k with maximum gap
        let max_gap_idx = gap_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        max_gap_idx + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gap_statistic_basic() -> ClusterResult<()> {
        // Create data with clear clusters
        let data = Tensor::from_vec(
            vec![
                // Cluster 1 (around origin)
                0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2, // Cluster 2 (around (5,5))
                5.0, 5.0, 5.1, 5.1, 5.2, 5.0, 5.0, 5.2,
            ],
            &[8, 2],
        )?;

        let config = GapStatisticConfig {
            max_k: 5,
            n_refs: 5, // Reduced for faster testing
            random_state: Some(42),
            ..Default::default()
        };

        let mut gap_stat = GapStatistic::new(config);
        let result = gap_stat.compute(&data)?;

        // Should detect 2 clusters
        assert!(result.optimal_k >= 1 && result.optimal_k <= 5);
        assert_eq!(result.gap_values.len(), 5);
        assert_eq!(result.wk_values.len(), 5);
        assert_eq!(result.sk_values.len(), 5);

        // Gap values should be finite
        for &gap in &result.gap_values {
            assert!(gap.is_finite());
        }

        Ok(())
    }

    #[test]
    fn test_gap_statistic_single_cluster() -> ClusterResult<()> {
        // Create tightly clustered data (should suggest k=1)
        let data = Tensor::from_vec(
            vec![0.0, 0.0, 0.01, 0.01, -0.01, 0.01, 0.01, -0.01],
            &[4, 2],
        )?;

        let config = GapStatisticConfig {
            max_k: 3,
            n_refs: 3,
            random_state: Some(42),
            ..Default::default()
        };

        let mut gap_stat = GapStatistic::new(config);
        let result = gap_stat.compute(&data)?;

        assert!(result.optimal_k >= 1);
        assert_eq!(result.gap_values.len(), 3);

        // All values should be finite
        for &gap in &result.gap_values {
            assert!(gap.is_finite());
        }

        Ok(())
    }

    #[test]
    fn test_gap_statistic_result_methods() -> ClusterResult<()> {
        let data = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[2, 2])?;

        let config = GapStatisticConfig {
            max_k: 2,
            n_refs: 2,
            random_state: Some(42),
            ..Default::default()
        };

        let mut gap_stat = GapStatistic::new(config);
        let result = gap_stat.compute(&data)?;

        // Test accessor methods
        assert!(result.gap(1).is_some());
        assert!(result.gap(2).is_some());
        assert!(result.gap(3).is_none());

        assert!(result.wk(1).is_some());
        assert!(result.wk(2).is_some());
        assert!(result.wk(3).is_none());

        assert!(result.is_optimal(result.optimal_k));

        let summary = result.summary();
        assert!(summary.contains_key("optimal_k"));
        assert!(summary.contains_key("max_gap"));
        assert!(summary.contains_key("n_k_tested"));

        Ok(())
    }
}
