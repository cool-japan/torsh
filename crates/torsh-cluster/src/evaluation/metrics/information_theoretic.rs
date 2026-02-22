//! Information-theoretic clustering evaluation metrics
//!
//! This module contains metrics based on information theory, including
//! mutual information, entropy, and related measures for evaluating
//! the similarity between clusterings.

use crate::error::{ClusterError, ClusterResult};
use std::collections::HashMap;
use torsh_tensor::Tensor;

use super::utils::compute_entropy;

/// Compute normalized mutual information
///
/// NMI measures the amount of information shared between two clusterings,
/// normalized by their individual entropies. Values range from 0 to 1,
/// where 1 indicates perfect agreement.
pub fn normalized_mutual_info_score(
    labels_true: &Tensor,
    labels_pred: &Tensor,
) -> ClusterResult<f64> {
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

    // Convert to integer labels
    let true_labels: Vec<i32> = true_vec.iter().map(|&x| x as i32).collect();
    let pred_labels: Vec<i32> = pred_vec.iter().map(|&x| x as i32).collect();

    // Build contingency table and compute frequencies
    let mut contingency_table: HashMap<(i32, i32), usize> = HashMap::new();
    let mut true_counts: HashMap<i32, usize> = HashMap::new();
    let mut pred_counts: HashMap<i32, usize> = HashMap::new();

    for (t, p) in true_labels.iter().zip(pred_labels.iter()) {
        *contingency_table.entry((*t, *p)).or_insert(0) += 1;
        *true_counts.entry(*t).or_insert(0) += 1;
        *pred_counts.entry(*p).or_insert(0) += 1;
    }

    // Compute mutual information
    let mut mi = 0.0;
    for (&(true_label, pred_label), &joint_count) in &contingency_table {
        if joint_count > 0 {
            let p_joint = joint_count as f64 / n as f64;
            let p_true = true_counts[&true_label] as f64 / n as f64;
            let p_pred = pred_counts[&pred_label] as f64 / n as f64;

            mi += p_joint * (p_joint / (p_true * p_pred)).ln();
        }
    }

    // Compute entropy for true labels
    let entropy_true = compute_entropy(&true_counts, n);

    // Compute entropy for predicted labels
    let entropy_pred = compute_entropy(&pred_counts, n);

    // Compute normalized mutual information (geometric mean normalization)
    if entropy_true == 0.0 || entropy_pred == 0.0 {
        if entropy_true == entropy_pred {
            Ok(1.0) // Both have no entropy (constant labels)
        } else {
            Ok(0.0) // One has entropy, one doesn't
        }
    } else {
        let nmi = mi / (entropy_true * entropy_pred).sqrt();
        Ok(nmi.clamp(0.0, 1.0)) // Clamp to [0, 1] due to numerical precision
    }
}

/// Compute adjusted mutual information
///
/// AMI adjusts the mutual information score for chance, similar to how
/// the adjusted rand index adjusts the rand index. Values range around 0-1,
/// with 1 indicating perfect agreement and 0 indicating agreement at chance level.
pub fn adjusted_mutual_info_score(
    labels_true: &Tensor,
    labels_pred: &Tensor,
) -> ClusterResult<f64> {
    let true_vec = labels_true.to_vec().map_err(ClusterError::TensorError)?;
    let pred_vec = labels_pred.to_vec().map_err(ClusterError::TensorError)?;

    if true_vec.len() != pred_vec.len() {
        return Err(ClusterError::InvalidInput(
            "Labels must have the same length".to_string(),
        ));
    }

    let n = true_vec.len() as f64;
    if n <= 1.0 {
        return Ok(1.0); // Perfect agreement for trivial cases
    }

    // Convert to integer labels
    let true_labels: Vec<i32> = true_vec.iter().map(|&x| x as i32).collect();
    let pred_labels: Vec<i32> = pred_vec.iter().map(|&x| x as i32).collect();

    // Build contingency table and compute frequencies
    let mut contingency_table: HashMap<(i32, i32), usize> = HashMap::new();
    let mut true_counts: HashMap<i32, usize> = HashMap::new();
    let mut pred_counts: HashMap<i32, usize> = HashMap::new();

    for (t, p) in true_labels.iter().zip(pred_labels.iter()) {
        *contingency_table.entry((*t, *p)).or_insert(0) += 1;
        *true_counts.entry(*t).or_insert(0) += 1;
        *pred_counts.entry(*p).or_insert(0) += 1;
    }

    // Compute mutual information
    let mut mi = 0.0;
    for (&(true_label, pred_label), &joint_count) in &contingency_table {
        if joint_count > 0 {
            let p_joint = joint_count as f64 / n;
            let p_true = true_counts[&true_label] as f64 / n;
            let p_pred = pred_counts[&pred_label] as f64 / n;

            mi += p_joint * (p_joint / (p_true * p_pred)).ln();
        }
    }

    // Compute entropies
    let entropy_true = compute_entropy(&true_counts, n as usize);
    let entropy_pred = compute_entropy(&pred_counts, n as usize);

    // Compute expected mutual information (EMI) for adjustment
    let emi = compute_expected_mutual_info(&true_counts, &pred_counts, n as usize);

    // Compute adjusted mutual information
    let normalizer = 0.5 * (entropy_true + entropy_pred);

    if normalizer.abs() < f64::EPSILON {
        Ok(0.0) // No information in either clustering
    } else {
        let ami = (mi - emi) / (normalizer - emi);
        Ok(ami.clamp(0.0, 1.0)) // Clamp to [0, 1] due to numerical precision
    }
}

/// Compute expected mutual information for AMI calculation
///
/// This function calculates the expected mutual information under the null
/// hypothesis of independence between the two clusterings.
fn compute_expected_mutual_info(
    true_counts: &HashMap<i32, usize>,
    pred_counts: &HashMap<i32, usize>,
    n: usize,
) -> f64 {
    let mut emi = 0.0;

    // Simplified EMI calculation using hypergeometric distribution
    // This is an approximation suitable for most practical cases
    for &a_i in true_counts.values() {
        for &b_j in pred_counts.values() {
            let a_i = a_i as f64;
            let b_j = b_j as f64;
            let n = n as f64;

            // Expected value of the hypergeometric distribution
            let expected_n_ij = (a_i * b_j) / n;

            if expected_n_ij > 0.0 {
                let p_true = a_i / n;
                let p_pred = b_j / n;
                let p_joint = expected_n_ij / n;

                if p_joint > 0.0 {
                    emi += p_joint * (p_joint / (p_true * p_pred)).ln();
                }
            }
        }
    }

    emi
}

/// Compute homogeneity score
///
/// Homogeneity measures whether each cluster contains only members of a single class.
/// A clustering result satisfies homogeneity if all of its clusters contain only
/// data points which are members of a single class. Values range from 0 to 1,
/// where 1 indicates perfect homogeneity.
pub fn homogeneity_score(labels_true: &Tensor, labels_pred: &Tensor) -> ClusterResult<f64> {
    let true_vec = labels_true.to_vec().map_err(ClusterError::TensorError)?;
    let pred_vec = labels_pred.to_vec().map_err(ClusterError::TensorError)?;

    if true_vec.len() != pred_vec.len() {
        return Err(ClusterError::InvalidInput(
            "Labels must have the same length".to_string(),
        ));
    }

    let n = true_vec.len();
    if n <= 1 {
        return Ok(1.0); // Perfect homogeneity for trivial cases
    }

    // Convert to integer labels
    let true_labels: Vec<i32> = true_vec.iter().map(|&x| x as i32).collect();
    let pred_labels: Vec<i32> = pred_vec.iter().map(|&x| x as i32).collect();

    // Build contingency table and count frequencies
    let mut contingency_table: HashMap<(i32, i32), usize> = HashMap::new();
    let mut true_counts: HashMap<i32, usize> = HashMap::new();
    let mut pred_counts: HashMap<i32, usize> = HashMap::new();

    for (t, p) in true_labels.iter().zip(pred_labels.iter()) {
        *contingency_table.entry((*t, *p)).or_insert(0) += 1;
        *true_counts.entry(*t).or_insert(0) += 1;
        *pred_counts.entry(*p).or_insert(0) += 1;
    }

    // Compute entropy H(Y) - entropy of true classes
    let entropy_true = compute_entropy(&true_counts, n);

    if entropy_true == 0.0 {
        return Ok(1.0); // Perfect homogeneity when only one true class
    }

    // Compute conditional entropy H(Y|C) - entropy of true classes given predicted clusters
    let mut conditional_entropy = 0.0;

    for (&pred_label, &cluster_size) in &pred_counts {
        if cluster_size > 0 {
            let cluster_prob = cluster_size as f64 / n as f64;

            // Find true class distribution within this predicted cluster
            let mut true_in_cluster: HashMap<i32, usize> = HashMap::new();

            for (&(true_label, predicted_label), &count) in &contingency_table {
                if predicted_label == pred_label {
                    *true_in_cluster.entry(true_label).or_insert(0) += count;
                }
            }

            // Compute entropy of true classes within this cluster
            let cluster_entropy = compute_entropy(&true_in_cluster, cluster_size);
            conditional_entropy += cluster_prob * cluster_entropy;
        }
    }

    // Homogeneity = 1 - H(Y|C) / H(Y)
    let homogeneity = 1.0 - (conditional_entropy / entropy_true);
    Ok(homogeneity.clamp(0.0, 1.0))
}

/// Compute completeness score
///
/// Completeness measures whether all members of a given class are assigned to the same cluster.
/// A clustering result satisfies completeness if all the data points that are members
/// of a given class are elements of the same cluster. Values range from 0 to 1,
/// where 1 indicates perfect completeness.
pub fn completeness_score(labels_true: &Tensor, labels_pred: &Tensor) -> ClusterResult<f64> {
    let true_vec = labels_true.to_vec().map_err(ClusterError::TensorError)?;
    let pred_vec = labels_pred.to_vec().map_err(ClusterError::TensorError)?;

    if true_vec.len() != pred_vec.len() {
        return Err(ClusterError::InvalidInput(
            "Labels must have the same length".to_string(),
        ));
    }

    let n = true_vec.len();
    if n <= 1 {
        return Ok(1.0); // Perfect completeness for trivial cases
    }

    // Convert to integer labels
    let true_labels: Vec<i32> = true_vec.iter().map(|&x| x as i32).collect();
    let pred_labels: Vec<i32> = pred_vec.iter().map(|&x| x as i32).collect();

    // Build contingency table and count frequencies
    let mut contingency_table: HashMap<(i32, i32), usize> = HashMap::new();
    let mut true_counts: HashMap<i32, usize> = HashMap::new();
    let mut pred_counts: HashMap<i32, usize> = HashMap::new();

    for (t, p) in true_labels.iter().zip(pred_labels.iter()) {
        *contingency_table.entry((*t, *p)).or_insert(0) += 1;
        *true_counts.entry(*t).or_insert(0) += 1;
        *pred_counts.entry(*p).or_insert(0) += 1;
    }

    // Compute entropy H(C) - entropy of predicted clusters
    let entropy_pred = compute_entropy(&pred_counts, n);

    if entropy_pred == 0.0 {
        return Ok(1.0); // Perfect completeness when only one predicted cluster
    }

    // Compute conditional entropy H(C|Y) - entropy of predicted clusters given true classes
    let mut conditional_entropy = 0.0;

    for (&true_label, &class_size) in &true_counts {
        if class_size > 0 {
            let class_prob = class_size as f64 / n as f64;

            // Find predicted cluster distribution within this true class
            let mut pred_in_class: HashMap<i32, usize> = HashMap::new();

            for (&(true_lbl, predicted_label), &count) in &contingency_table {
                if true_lbl == true_label {
                    *pred_in_class.entry(predicted_label).or_insert(0) += count;
                }
            }

            // Compute entropy of predicted clusters within this true class
            let class_entropy = compute_entropy(&pred_in_class, class_size);
            conditional_entropy += class_prob * class_entropy;
        }
    }

    // Completeness = 1 - H(C|Y) / H(C)
    let completeness = 1.0 - (conditional_entropy / entropy_pred);
    Ok(completeness.clamp(0.0, 1.0))
}

/// Compute V-measure score
///
/// The V-measure is the harmonic mean of homogeneity and completeness.
/// It provides a single measure that captures both aspects of clustering quality.
/// Values range from 0 to 1, where 1 indicates perfect clustering.
pub fn v_measure_score(labels_true: &Tensor, labels_pred: &Tensor) -> ClusterResult<f64> {
    let homogeneity = homogeneity_score(labels_true, labels_pred)?;
    let completeness = completeness_score(labels_true, labels_pred)?;

    if homogeneity + completeness == 0.0 {
        Ok(0.0)
    } else {
        Ok(2.0 * homogeneity * completeness / (homogeneity + completeness))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_nmi_perfect_match() -> Result<(), Box<dyn std::error::Error>> {
        let labels_true = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[4])?;
        let labels_pred = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[4])?;

        let nmi = normalized_mutual_info_score(&labels_true, &labels_pred)?;
        assert_relative_eq!(nmi, 1.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_v_measure_perfect_match() -> Result<(), Box<dyn std::error::Error>> {
        let labels_true = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[4])?;
        let labels_pred = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0], &[4])?;

        let v_measure = v_measure_score(&labels_true, &labels_pred)?;
        assert_relative_eq!(v_measure, 1.0, epsilon = 1e-6);

        Ok(())
    }
}
