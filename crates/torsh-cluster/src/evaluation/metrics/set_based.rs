//! Set-based clustering evaluation metrics
//!
//! This module contains metrics that evaluate clustering quality based on
//! set comparisons and combinatorial analysis of cluster assignments.

use crate::error::{ClusterError, ClusterResult};
use std::collections::{HashMap, HashSet};
use torsh_tensor::Tensor;

use super::utils::combinations;

/// Compute adjusted rand index comparing two clusterings
///
/// ARI measures the similarity between two clusterings, adjusting for the chance
/// grouping of elements. Values range from -1 to 1, where 1 indicates perfect
/// agreement, 0 indicates random assignment, and negative values indicate
/// agreement worse than random.
pub fn adjusted_rand_score(labels_true: &Tensor, labels_pred: &Tensor) -> ClusterResult<f64> {
    let true_vec = labels_true.to_vec().map_err(ClusterError::TensorError)?;
    let pred_vec = labels_pred.to_vec().map_err(ClusterError::TensorError)?;

    if true_vec.len() != pred_vec.len() {
        return Err(ClusterError::InvalidInput(
            "Labels must have the same length".to_string(),
        ));
    }

    let n = true_vec.len();
    if n <= 1 {
        return Ok(1.0); // Perfect agreement for trivial cases
    }

    // Convert to integer labels for easier processing
    let true_labels: Vec<i32> = true_vec.iter().map(|&x| x as i32).collect();
    let pred_labels: Vec<i32> = pred_vec.iter().map(|&x| x as i32).collect();

    // Get unique labels for both clusterings
    let _true_unique: HashSet<i32> = true_labels.iter().cloned().collect();
    let _pred_unique: HashSet<i32> = pred_labels.iter().cloned().collect();

    // Build contingency table
    let mut contingency_table: HashMap<(i32, i32), usize> = HashMap::new();

    for (t, p) in true_labels.iter().zip(pred_labels.iter()) {
        *contingency_table.entry((*t, *p)).or_insert(0) += 1;
    }

    // Calculate sums for ARI formula
    let mut sum_comb_c = 0_f64; // Sum of C(n_ij, 2)
    let mut a_sum = HashMap::new(); // Sum for each true cluster
    let mut b_sum = HashMap::new(); // Sum for each predicted cluster

    for (&(true_label, pred_label), &count) in &contingency_table {
        // Add to contingency combinations
        if count >= 2 {
            sum_comb_c += combinations(count as u64, 2) as f64;
        }

        // Track row sums (true clusters)
        *a_sum.entry(true_label).or_insert(0) += count;
        // Track column sums (predicted clusters)
        *b_sum.entry(pred_label).or_insert(0) += count;
    }

    // Calculate sum of C(a_i, 2) and sum of C(b_j, 2)
    let sum_comb_a: f64 = a_sum
        .values()
        .map(|&count| {
            if count >= 2 {
                combinations(count as u64, 2) as f64
            } else {
                0.0
            }
        })
        .sum();

    let sum_comb_b: f64 = b_sum
        .values()
        .map(|&count| {
            if count >= 2 {
                combinations(count as u64, 2) as f64
            } else {
                0.0
            }
        })
        .sum();

    // Total possible pairs
    let total_pairs = if n >= 2 {
        combinations(n as u64, 2) as f64
    } else {
        1.0
    };

    // Expected value (for adjustment)
    let expected_value = (sum_comb_a * sum_comb_b) / total_pairs;

    // Compute ARI
    let numerator = sum_comb_c - expected_value;
    let denominator = 0.5 * (sum_comb_a + sum_comb_b) - expected_value;

    if denominator.abs() < f64::EPSILON {
        Ok(0.0) // No clustering structure
    } else {
        Ok(numerator / denominator)
    }
}

/// Compute Fowlkes-Mallows score
///
/// The Fowlkes-Mallows index measures the similarity between two clusterings
/// based on the geometric mean of pairwise precision and recall. Values range
/// from 0 to 1, where 1 indicates perfect agreement.
pub fn fowlkes_mallows_score(labels_true: &Tensor, labels_pred: &Tensor) -> ClusterResult<f64> {
    let true_vec = labels_true.to_vec().map_err(ClusterError::TensorError)?;
    let pred_vec = labels_pred.to_vec().map_err(ClusterError::TensorError)?;

    if true_vec.len() != pred_vec.len() {
        return Err(ClusterError::InvalidInput(
            "Labels must have the same length".to_string(),
        ));
    }

    let n = true_vec.len();
    if n <= 1 {
        return Ok(1.0); // Perfect score for trivial cases
    }

    // Convert to integer labels for easier processing
    let true_labels: Vec<i32> = true_vec.iter().map(|&x| x as i32).collect();
    let pred_labels: Vec<i32> = pred_vec.iter().map(|&x| x as i32).collect();

    // Count pairwise statistics
    let mut tp = 0_u64; // True positives: pairs in same true class and same predicted cluster
    let mut fp = 0_u64; // False positives: pairs in different true classes but same predicted cluster
    let mut fn_count = 0_u64; // False negatives: pairs in same true class but different predicted clusters

    // Compute pairwise statistics efficiently
    for i in 0..n {
        for j in (i + 1)..n {
            let same_true = true_labels[i] == true_labels[j];
            let same_pred = pred_labels[i] == pred_labels[j];

            match (same_true, same_pred) {
                (true, true) => tp += 1,        // Same true class, same predicted cluster
                (false, true) => fp += 1,       // Different true classes, same predicted cluster
                (true, false) => fn_count += 1, // Same true class, different predicted clusters
                (false, false) => {} // Different true classes, different predicted clusters (true negative)
            }
        }
    }

    // Calculate precision and recall
    let precision = if tp + fp == 0 {
        1.0 // No predicted positive pairs
    } else {
        tp as f64 / (tp + fp) as f64
    };

    let recall = if tp + fn_count == 0 {
        1.0 // No true positive pairs
    } else {
        tp as f64 / (tp + fn_count) as f64
    };

    // Fowlkes-Mallows score is the geometric mean of precision and recall
    let fm_score = (precision * recall).sqrt();
    Ok(fm_score.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_ari_perfect_match() -> Result<(), Box<dyn std::error::Error>> {
        let labels_true = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[4])?;
        let labels_pred = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[4])?;

        let ari = adjusted_rand_score(&labels_true, &labels_pred)?;
        assert_relative_eq!(ari, 1.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_fm_perfect_match() -> Result<(), Box<dyn std::error::Error>> {
        let labels_true = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[4])?;
        let labels_pred = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[4])?;

        let fm = fowlkes_mallows_score(&labels_true, &labels_pred)?;
        assert_relative_eq!(fm, 1.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_ari_random_assignment() -> Result<(), Box<dyn std::error::Error>> {
        // Test with truly random assignment that breaks clustering structure
        let labels_true = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0], &[6])?;
        let labels_pred = Tensor::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0], &[6])?; // Alternating pattern

        let ari = adjusted_rand_score(&labels_true, &labels_pred)?;
        // This should give a low (close to 0 or negative) ARI score
        assert!(
            ari < 0.5,
            "ARI should be low for random assignment: got {}",
            ari
        );
        Ok(())
    }
}
