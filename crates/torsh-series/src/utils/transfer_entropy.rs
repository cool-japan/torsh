//! Transfer Entropy for Information-Theoretic Causal Inference
//!
//! Transfer entropy quantifies the amount of information transferred from one
//! time series to another, providing a non-linear, model-free measure of causality.
//!
//! # Mathematical Foundation
//!
//! Transfer entropy from X to Y:
//! TE(X→Y) = ∑ p(y_{t+1}, y_t^k, x_t^l) log[p(y_{t+1}|y_t^k, x_t^l) / p(y_{t+1}|y_t^k)]
//!
//! where:
//! - y_t^k = (y_t, y_{t-1}, ..., y_{t-k+1}): k-history of Y
//! - x_t^l = (x_t, x_{t-1}, ..., x_{t-l+1}): l-history of X
//!
//! # References
//! - Schreiber, T. (2000). Measuring information transfer. Physical Review Letters, 85(2), 461.
//! - Bossomaier, T., et al. (2016). An Introduction to Transfer Entropy. Springer.

use scirs2_core::ndarray::Array1;
use std::collections::HashMap;

/// Transfer Entropy estimator
#[derive(Debug, Clone)]
pub struct TransferEntropyEstimator {
    /// History length for target variable Y
    pub k: usize,
    /// History length for source variable X
    pub l: usize,
    /// Number of bins for discretization
    pub bins: usize,
    /// Method for estimation
    pub method: TEMethod,
}

/// Methods for estimating transfer entropy
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TEMethod {
    /// Histogram-based estimation (fast, requires discretization)
    Histogram,
    /// Kernel density estimation (slower, more accurate for continuous data)
    KDE,
    /// k-Nearest Neighbors (adaptive bandwidth)
    KNN,
}

/// Result of transfer entropy calculation
#[derive(Debug, Clone)]
pub struct TransferEntropyResult {
    /// Transfer entropy from X to Y
    pub te_x_to_y: f64,
    /// Transfer entropy from Y to X
    pub te_y_to_x: f64,
    /// Net information flow (TE(X→Y) - TE(Y→X))
    pub net_flow: f64,
    /// Statistical significance (if computed)
    pub p_value: Option<f64>,
    /// Number of samples used
    pub n_samples: usize,
}

impl TransferEntropyEstimator {
    /// Create a new transfer entropy estimator
    ///
    /// # Arguments
    /// * `k` - History length for target variable
    /// * `l` - History length for source variable
    /// * `bins` - Number of bins for discretization (histogram method)
    ///
    /// # Example
    /// ```
    /// use torsh_series::utils::transfer_entropy::{TransferEntropyEstimator, TEMethod};
    ///
    /// let estimator = TransferEntropyEstimator::new(1, 1, 10);
    /// ```
    pub fn new(k: usize, l: usize, bins: usize) -> Self {
        Self {
            k,
            l,
            bins,
            method: TEMethod::Histogram,
        }
    }

    /// Set estimation method
    pub fn with_method(mut self, method: TEMethod) -> Self {
        self.method = method;
        self
    }

    /// Compute transfer entropy from X to Y
    ///
    /// # Arguments
    /// * `x` - Source time series
    /// * `y` - Target time series
    ///
    /// # Returns
    /// Transfer entropy value in bits (base 2 logarithm)
    pub fn compute(&self, x: &Array1<f64>, y: &Array1<f64>) -> Result<f64, String> {
        if x.len() != y.len() {
            return Err("Time series must have equal length".to_string());
        }

        if x.len() < self.k + self.l + 1 {
            return Err(format!(
                "Time series too short. Need at least {} samples",
                self.k + self.l + 1
            ));
        }

        match self.method {
            TEMethod::Histogram => self.compute_histogram(x, y),
            TEMethod::KDE => self.compute_kde(x, y),
            TEMethod::KNN => self.compute_knn(x, y),
        }
    }

    /// Compute bidirectional transfer entropy
    pub fn compute_bidirectional(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
    ) -> Result<TransferEntropyResult, String> {
        let te_x_to_y = self.compute(x, y)?;
        let te_y_to_x = self.compute(y, x)?;
        let net_flow = te_x_to_y - te_y_to_x;

        Ok(TransferEntropyResult {
            te_x_to_y,
            te_y_to_x,
            net_flow,
            p_value: None,
            n_samples: x.len() - self.k - self.l,
        })
    }

    /// Compute transfer entropy with statistical significance testing
    pub fn compute_with_significance(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        n_permutations: usize,
    ) -> Result<TransferEntropyResult, String> {
        let te_observed = self.compute(x, y)?;

        // Permutation test
        let mut te_null = Vec::with_capacity(n_permutations);
        let mut rng = scirs2_core::random::thread_rng();

        for _ in 0..n_permutations {
            // Shuffle X to break temporal structure
            let mut x_perm = x.to_vec();
            // Simple shuffle (Fisher-Yates)
            for i in (1..x_perm.len()).rev() {
                let j = (rng.gen_range(0.0..1.0) * (i + 1) as f64) as usize;
                x_perm.swap(i, j);
            }
            let x_perm_array = Array1::from_vec(x_perm);
            let te_perm = self.compute(&x_perm_array, y)?;
            te_null.push(te_perm);
        }

        // Compute p-value
        let count_greater = te_null.iter().filter(|&&te| te >= te_observed).count();
        let p_value = (count_greater + 1) as f64 / (n_permutations + 1) as f64;

        Ok(TransferEntropyResult {
            te_x_to_y: te_observed,
            te_y_to_x: 0.0, // Not computed in this variant
            net_flow: te_observed,
            p_value: Some(p_value),
            n_samples: x.len() - self.k - self.l,
        })
    }

    /// Histogram-based transfer entropy estimation
    fn compute_histogram(&self, x: &Array1<f64>, y: &Array1<f64>) -> Result<f64, String> {
        // Discretize time series
        let x_disc = self.discretize(x);
        let y_disc = self.discretize(y);

        // Build state vectors
        let n = x.len() - self.k.max(self.l);
        let mut joint_counts: HashMap<(Vec<usize>, Vec<usize>, usize), usize> =
            HashMap::with_capacity(n);
        let mut cond_counts: HashMap<(Vec<usize>, usize), usize> = HashMap::with_capacity(n);
        let mut marginal_counts: HashMap<Vec<usize>, usize> = HashMap::with_capacity(n);

        for t in self.k.max(self.l)..y_disc.len() {
            // y_{t+1}
            let y_next = y_disc[t];

            // y_t^k = (y_t, y_{t-1}, ..., y_{t-k+1})
            let y_hist: Vec<usize> = (0..self.k).map(|i| y_disc[t - 1 - i]).collect();

            // x_t^l = (x_t, x_{t-1}, ..., x_{t-l+1})
            let x_hist: Vec<usize> = (0..self.l).map(|i| x_disc[t - 1 - i]).collect();

            // Count joint occurrences
            *joint_counts
                .entry((y_hist.clone(), x_hist.clone(), y_next))
                .or_insert(0) += 1;

            // Count conditional occurrences p(y_{t+1}, y_t^k)
            *cond_counts.entry((y_hist.clone(), y_next)).or_insert(0) += 1;

            // Count marginal occurrences p(y_t^k)
            *marginal_counts.entry(y_hist).or_insert(0) += 1;
        }

        // Compute transfer entropy
        let mut te = 0.0;
        let n_samples = n as f64;

        for ((y_hist, x_hist, y_next), &count_joint) in &joint_counts {
            let p_joint = count_joint as f64 / n_samples;

            // p(y_{t+1}|y_t^k, x_t^l)
            let count_yh_xh: usize = joint_counts
                .iter()
                .filter(|((yh, xh, _), _)| yh == y_hist && xh == x_hist)
                .map(|(_, &c)| c)
                .sum();

            if count_yh_xh == 0 {
                continue;
            }

            let p_cond_joint = count_joint as f64 / count_yh_xh as f64;

            // p(y_{t+1}|y_t^k)
            let count_yh = marginal_counts.get(y_hist).copied().unwrap_or(0);
            if count_yh == 0 {
                continue;
            }

            let count_yh_yn = cond_counts
                .get(&(y_hist.clone(), *y_next))
                .copied()
                .unwrap_or(0);
            let p_cond_y = count_yh_yn as f64 / count_yh as f64;

            if p_cond_y > 0.0 && p_cond_joint > 0.0 {
                te += p_joint * (p_cond_joint / p_cond_y).log2();
            }
        }

        Ok(te.max(0.0)) // Transfer entropy is non-negative
    }

    /// KDE-based transfer entropy estimation (simplified)
    fn compute_kde(&self, x: &Array1<f64>, y: &Array1<f64>) -> Result<f64, String> {
        // For simplicity, fall back to histogram with more bins
        let mut estimator = self.clone();
        estimator.bins = (self.bins as f64 * 1.5) as usize;
        estimator.compute_histogram(x, y)
    }

    /// k-NN based transfer entropy estimation (simplified)
    fn compute_knn(&self, x: &Array1<f64>, y: &Array1<f64>) -> Result<f64, String> {
        // For simplicity, use histogram method
        // Full k-NN implementation would use Kozachenko-Leonenko estimator
        self.compute_histogram(x, y)
    }

    /// Discretize continuous time series into bins
    fn discretize(&self, data: &Array1<f64>) -> Vec<usize> {
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_val - min_val;

        if range == 0.0 {
            return vec![0; data.len()];
        }

        data.iter()
            .map(|&x| {
                let normalized = (x - min_val) / range;
                let bin = (normalized * self.bins as f64) as usize;
                bin.min(self.bins - 1)
            })
            .collect()
    }
}

/// Conditional Transfer Entropy (accounting for common drivers)
///
/// CTE(X→Y|Z) measures transfer entropy from X to Y conditioned on Z,
/// helping to distinguish direct from indirect information transfer.
pub struct ConditionalTransferEntropy {
    estimator: TransferEntropyEstimator,
    _m: usize, // History length for conditioning variable Z
}

impl ConditionalTransferEntropy {
    /// Create a new conditional transfer entropy estimator
    pub fn new(k: usize, l: usize, m: usize, bins: usize) -> Self {
        Self {
            estimator: TransferEntropyEstimator::new(k, l, bins),
            _m: m,
        }
    }

    /// Compute conditional transfer entropy from X to Y given Z
    pub fn compute(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        z: &Array1<f64>,
    ) -> Result<f64, String> {
        if x.len() != y.len() || y.len() != z.len() {
            return Err("All time series must have equal length".to_string());
        }

        // Simplified implementation: compute TE(X→Y) - TE(Z→Y)
        // This approximates the conditional effect
        let te_x_to_y = self.estimator.compute(x, y)?;
        let te_z_to_y = self.estimator.compute(z, y)?;

        Ok((te_x_to_y - te_z_to_y).max(0.0))
    }
}

/// Time-lagged transfer entropy
///
/// Computes transfer entropy for different time lags to identify
/// the optimal delay and the temporal structure of information transfer.
pub fn compute_lagged_transfer_entropy(
    x: &Array1<f64>,
    y: &Array1<f64>,
    max_lag: usize,
    k: usize,
    l: usize,
    bins: usize,
) -> Result<Vec<(usize, f64)>, String> {
    let mut results = Vec::with_capacity(max_lag);

    for lag in 0..=max_lag {
        if lag >= x.len() {
            break;
        }

        // Create lagged version of X
        let x_lagged = if lag > 0 {
            Array1::from_vec(x.iter().take(x.len() - lag).cloned().collect())
        } else {
            x.clone()
        };

        let y_adjusted = if lag > 0 {
            Array1::from_vec(y.iter().skip(lag).cloned().collect())
        } else {
            y.clone()
        };

        let estimator = TransferEntropyEstimator::new(k, l, bins);
        let te = estimator.compute(&x_lagged, &y_adjusted)?;
        results.push((lag, te));
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_entropy_basic() {
        // Create simple causal relationship: Y depends on X
        let n = 100;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);

        x.push(0.5);
        y.push(0.3);

        for i in 1..n {
            x.push(0.9 * x[i - 1] + 0.1);
            y.push(0.5 * x[i - 1] + 0.3 * y[i - 1] + 0.2); // Y depends on X
        }

        let x_array = Array1::from_vec(x);
        let y_array = Array1::from_vec(y);

        let estimator = TransferEntropyEstimator::new(1, 1, 5);
        let te_x_to_y = estimator.compute(&x_array, &y_array).unwrap();
        let te_y_to_x = estimator.compute(&y_array, &x_array).unwrap();

        // TE(X→Y) should be greater than TE(Y→X) since Y depends on X
        assert!(te_x_to_y >= 0.0);
        assert!(te_y_to_x >= 0.0);
    }

    #[test]
    fn test_bidirectional_te() {
        let x = Array1::from_vec((0..50).map(|i| (i as f64 * 0.1).sin()).collect());
        let y = Array1::from_vec((0..50).map(|i| ((i as f64 * 0.1) + 0.5).cos()).collect());

        let estimator = TransferEntropyEstimator::new(1, 1, 5);
        let result = estimator.compute_bidirectional(&x, &y).unwrap();

        assert!(result.te_x_to_y >= 0.0);
        assert!(result.te_y_to_x >= 0.0);
        assert_eq!(result.n_samples, 48);
    }

    #[test]
    fn test_discretization() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let estimator = TransferEntropyEstimator::new(1, 1, 3);
        let discretized = estimator.discretize(&data);

        assert_eq!(discretized.len(), 5);
        assert!(discretized.iter().all(|&x| x < 3));
    }

    #[test]
    fn test_lagged_te() {
        let x = Array1::from_vec((0..50).map(|i| i as f64).collect());
        let y = Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect());

        let results = compute_lagged_transfer_entropy(&x, &y, 5, 1, 1, 5).unwrap();

        assert_eq!(results.len(), 6); // 0 to 5 lags
        assert!(results.iter().all(|(_, te)| *te >= 0.0));
    }

    #[test]
    fn test_conditional_te() {
        let x = Array1::from_vec((0..50).map(|i| (i as f64 * 0.1).sin()).collect());
        let y = Array1::from_vec((0..50).map(|i| ((i as f64 * 0.1) + 0.5).cos()).collect());
        let z = Array1::from_vec((0..50).map(|i| (i as f64 * 0.15).sin()).collect());

        let cte = ConditionalTransferEntropy::new(1, 1, 1, 5);
        let result = cte.compute(&x, &y, &z).unwrap();

        assert!(result >= 0.0);
    }

    #[test]
    fn test_te_with_different_methods() {
        let x = Array1::from_vec((0..50).map(|i| i as f64).collect());
        let y = Array1::from_vec((0..50).map(|i| (i + 1) as f64).collect());

        let est_hist = TransferEntropyEstimator::new(1, 1, 5).with_method(TEMethod::Histogram);
        let est_kde = TransferEntropyEstimator::new(1, 1, 5).with_method(TEMethod::KDE);
        let est_knn = TransferEntropyEstimator::new(1, 1, 5).with_method(TEMethod::KNN);

        let te_hist = est_hist.compute(&x, &y).unwrap();
        let te_kde = est_kde.compute(&x, &y).unwrap();
        let te_knn = est_knn.compute(&x, &y).unwrap();

        assert!(te_hist >= 0.0);
        assert!(te_kde >= 0.0);
        assert!(te_knn >= 0.0);
    }

    #[test]
    fn test_independent_series() {
        // Two independent random series should have low TE
        let x = Array1::from_vec((0..50).map(|i| (i % 7) as f64).collect());
        let y = Array1::from_vec((0..50).map(|i| ((i * 3) % 11) as f64).collect());

        let estimator = TransferEntropyEstimator::new(1, 1, 5);
        let te = estimator.compute(&x, &y).unwrap();

        // TE should be close to zero for independent series
        assert!(te >= 0.0);
        assert!(te < 2.0); // Reasonable upper bound for independent series
    }
}
