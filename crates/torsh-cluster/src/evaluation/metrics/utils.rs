//! Utility functions for clustering evaluation metrics
//!
//! This module contains shared utility functions used across different
//! clustering evaluation metrics.

use crate::error::ClusterResult;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

/// Compute pairwise distances between all samples using SciRS2
///
/// This function calculates the Euclidean distance between every pair of samples
/// in the dataset and returns a symmetric distance matrix.
pub fn compute_pairwise_distances(data: &Array2<f32>) -> ClusterResult<Array2<f32>> {
    let n_samples = data.nrows();
    let mut distances = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        for j in i..n_samples {
            if i == j {
                distances[[i, j]] = 0.0;
            } else {
                let sample_i = data.row(i);
                let sample_j = data.row(j);
                let dist = euclidean_distance_manual(&sample_i, &sample_j);
                distances[[i, j]] = dist;
                distances[[j, i]] = dist; // Symmetric
            }
        }
    }

    Ok(distances)
}

/// Manual euclidean distance calculation using SciRS2 array operations
///
/// Computes the Euclidean distance between two array views representing
/// data points in n-dimensional space.
pub fn euclidean_distance_manual(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    let diff: Array1<f32> = a.to_owned() - b.to_owned();
    let squared_diff: Array1<f32> = diff.mapv(|x| x * x);
    squared_diff.sum().sqrt()
}

/// Calculate combinations C(n, k) = n! / (k! * (n-k)!)
///
/// This is used in various clustering metrics that involve combinatorial
/// calculations, particularly for adjusted indices.
pub fn combinations(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    // Use the more efficient formula: C(n,k) = C(n,n-k)
    let k = if k > n - k { n - k } else { k };

    let mut result = 1_u64;
    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }
    result
}

/// Compute entropy from label counts
///
/// This function calculates the Shannon entropy for a given distribution
/// of cluster labels, which is used in information-theoretic metrics.
pub fn compute_entropy(counts: &std::collections::HashMap<i32, usize>, total: usize) -> f64 {
    let mut entropy = 0.0;
    for &count in counts.values() {
        if count > 0 {
            let p = count as f64 / total as f64;
            entropy -= p * p.ln();
        }
    }
    entropy
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_euclidean_distance_manual() {
        let a = Array1::from_vec(vec![0.0_f32, 0.0_f32]);
        let b = Array1::from_vec(vec![3.0_f32, 4.0_f32]);

        let distance = euclidean_distance_manual(&a.view(), &b.view());
        assert_relative_eq!(distance, 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_combinations() {
        assert_eq!(combinations(5, 2), 10);
        assert_eq!(combinations(4, 0), 1);
        assert_eq!(combinations(4, 4), 1);
        assert_eq!(combinations(3, 5), 0); // k > n
    }

    #[test]
    fn test_compute_entropy() {
        use std::collections::HashMap;

        let mut counts = HashMap::new();
        counts.insert(0, 2);
        counts.insert(1, 2);

        let entropy = compute_entropy(&counts, 4);
        // Entropy should be ln(2) ≈ 0.693 for uniform distribution
        assert_relative_eq!(entropy, 2.0_f64.ln(), epsilon = 1e-6);
    }
}
