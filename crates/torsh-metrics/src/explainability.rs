//! Explainability and interpretability metrics
//!
//! This module provides metrics for evaluating model explanations and interpretability.
//! All implementations follow SciRS2 POLICY - using scirs2-core abstractions.

use torsh_core::error::TorshError;

/// Feature importance stability across multiple runs or data subsets
///
/// Measures consistency of feature importance rankings.
/// Higher values indicate more stable/reliable feature rankings.
///
/// # Arguments
/// * `importances` - Multiple feature importance vectors (n_runs × n_features)
///
/// # Returns
/// Stability score in [0, 1], where 1 = perfectly stable
pub fn feature_importance_stability(importances: &[Vec<f64>]) -> Result<f64, TorshError> {
    if importances.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Empty importance vectors".to_string(),
        ));
    }

    let n_features = importances[0].len();
    if n_features == 0 {
        return Err(TorshError::InvalidArgument("Empty feature set".to_string()));
    }

    // Check all vectors have same length
    if !importances.iter().all(|v| v.len() == n_features) {
        return Err(TorshError::InvalidArgument(
            "Inconsistent feature counts".to_string(),
        ));
    }

    // Calculate pairwise rank correlations (Spearman's rho)
    let mut total_correlation = 0.0;
    let mut count = 0;

    for i in 0..importances.len() {
        for j in (i + 1)..importances.len() {
            let correlation = rank_correlation(&importances[i], &importances[j]);
            total_correlation += correlation;
            count += 1;
        }
    }

    if count == 0 {
        return Ok(1.0); // Single run is perfectly stable with itself
    }

    // Average correlation (transformed to [0, 1])
    let avg_correlation = total_correlation / count as f64;
    Ok((avg_correlation + 1.0) / 2.0)
}

/// Spearman's rank correlation coefficient
fn rank_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n != y.len() || n == 0 {
        return 0.0;
    }

    // Create rankings
    let rank_x = assign_ranks(x);
    let rank_y = assign_ranks(y);

    // Calculate correlation
    let mean_rank = (n + 1) as f64 / 2.0;
    let mut numerator = 0.0;
    let mut denom_x = 0.0;
    let mut denom_y = 0.0;

    for i in 0..n {
        let dx = rank_x[i] - mean_rank;
        let dy = rank_y[i] - mean_rank;
        numerator += dx * dy;
        denom_x += dx * dx;
        denom_y += dy * dy;
    }

    if denom_x < 1e-10 || denom_y < 1e-10 {
        return 0.0;
    }

    numerator / (denom_x * denom_y).sqrt()
}

/// Assign ranks to values (average rank for ties)
fn assign_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0; n];
    let mut i = 0;

    while i < n {
        let mut j = i;
        // Find ties
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-10 {
            j += 1;
        }

        // Average rank for ties
        let avg_rank = ((i + 1) + j) as f64 / 2.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }

        i = j;
    }

    ranks
}

/// Feature attribution agreement between methods
///
/// Measures how well different explanation methods agree on feature attributions.
///
/// # Arguments
/// * `attributions` - Multiple attribution vectors from different methods
///
/// # Returns
/// Agreement score in [0, 1], where 1 = perfect agreement
pub fn attribution_agreement(attributions: &[Vec<f64>]) -> Result<f64, TorshError> {
    if attributions.len() < 2 {
        return Err(TorshError::InvalidArgument(
            "Need at least 2 attribution methods".to_string(),
        ));
    }

    let n_features = attributions[0].len();
    if !attributions.iter().all(|v| v.len() == n_features) {
        return Err(TorshError::InvalidArgument(
            "Inconsistent feature counts".to_string(),
        ));
    }

    // Calculate pairwise cosine similarity
    let mut total_similarity = 0.0;
    let mut count = 0;

    for i in 0..attributions.len() {
        for j in (i + 1)..attributions.len() {
            let similarity = cosine_similarity(&attributions[i], &attributions[j]);
            total_similarity += similarity;
            count += 1;
        }
    }

    Ok(total_similarity / count as f64)
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    (dot_product / (norm_a * norm_b)).max(-1.0).min(1.0)
}

/// Faithfulness - how well explanation matches model predictions
///
/// Measures correlation between attribution magnitudes and prediction changes
/// when features are perturbed.
///
/// # Arguments
/// * `attributions` - Feature attributions
/// * `perturbation_effects` - Actual effect on predictions when features are removed
pub fn explanation_faithfulness(
    attributions: &[f64],
    perturbation_effects: &[f64],
) -> Result<f64, TorshError> {
    if attributions.len() != perturbation_effects.len() {
        return Err(TorshError::InvalidArgument(
            "Mismatched lengths".to_string(),
        ));
    }

    if attributions.is_empty() {
        return Err(TorshError::InvalidArgument("Empty inputs".to_string()));
    }

    // Calculate absolute values for comparison
    let abs_attributions: Vec<f64> = attributions.iter().map(|x| x.abs()).collect();

    // Calculate correlation
    let corr = correlation(&abs_attributions, perturbation_effects);

    // Transform to [0, 1] where 1 is perfect faithfulness
    Ok((corr + 1.0) / 2.0)
}

/// Pearson correlation coefficient
fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut denom_x = 0.0;
    let mut denom_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        numerator += dx * dy;
        denom_x += dx * dx;
        denom_y += dy * dy;
    }

    if denom_x < 1e-10 || denom_y < 1e-10 {
        return 0.0;
    }

    numerator / (denom_x * denom_y).sqrt()
}

/// Explanation completeness - coverage of important features
///
/// Measures what fraction of truly important features are identified by explanations.
///
/// # Arguments
/// * `attributions` - Feature attributions from explanation method
/// * `true_importances` - Ground truth feature importances (if known)
/// * `top_k` - Number of top features to consider
pub fn explanation_completeness(
    attributions: &[f64],
    true_importances: &[f64],
    top_k: usize,
) -> Result<f64, TorshError> {
    if attributions.len() != true_importances.len() {
        return Err(TorshError::InvalidArgument(
            "Mismatched lengths".to_string(),
        ));
    }

    let n = attributions.len();
    if top_k == 0 || top_k > n {
        return Err(TorshError::InvalidArgument(
            "Invalid top_k value".to_string(),
        ));
    }

    // Get top k features by attributions
    let mut attr_indexed: Vec<(usize, f64)> = attributions
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v.abs()))
        .collect();
    attr_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_attr: std::collections::HashSet<usize> =
        attr_indexed.iter().take(top_k).map(|(i, _)| *i).collect();

    // Get top k features by true importance
    let mut true_indexed: Vec<(usize, f64)> = true_importances
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v.abs()))
        .collect();
    true_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_true: std::collections::HashSet<usize> =
        true_indexed.iter().take(top_k).map(|(i, _)| *i).collect();

    // Calculate overlap (Jaccard similarity)
    let intersection = top_attr.intersection(&top_true).count();
    let union = top_attr.union(&top_true).count();

    if union == 0 {
        return Ok(0.0);
    }

    Ok(intersection as f64 / union as f64)
}

/// Monotonicity - how monotonic the relationship is between feature values and predictions
///
/// Higher monotonicity suggests more interpretable feature effects.
///
/// # Arguments
/// * `feature_values` - Sorted feature values
/// * `predictions` - Corresponding predictions
pub fn feature_monotonicity(
    feature_values: &[f64],
    predictions: &[f64],
) -> Result<f64, TorshError> {
    if feature_values.len() != predictions.len() {
        return Err(TorshError::InvalidArgument(
            "Mismatched lengths".to_string(),
        ));
    }

    if feature_values.len() < 2 {
        return Err(TorshError::InvalidArgument(
            "Need at least 2 points".to_string(),
        ));
    }

    // Count monotonic pairs
    let mut increasing = 0;
    let mut decreasing = 0;

    for i in 1..feature_values.len() {
        if predictions[i] > predictions[i - 1] {
            increasing += 1;
        } else if predictions[i] < predictions[i - 1] {
            decreasing += 1;
        }
    }

    let total_pairs = feature_values.len() - 1;
    let max_monotonic = increasing.max(decreasing);

    Ok(max_monotonic as f64 / total_pairs as f64)
}

/// Feature interaction strength
///
/// Measures how much features interact (non-additive effects).
/// Lower values suggest more interpretable additive relationships.
///
/// # Arguments
/// * `individual_effects` - Sum of individual feature effects
/// * `joint_effect` - Actual effect when features are used together
pub fn interaction_strength(individual_effects: f64, joint_effect: f64) -> f64 {
    let denominator = individual_effects.abs().max(joint_effect.abs());
    if denominator < 1e-10 {
        return 0.0;
    }

    ((joint_effect - individual_effects).abs() / denominator).min(1.0)
}

/// Counterfactual validity
///
/// Measures how realistic counterfactual explanations are.
/// Compares distance to nearest training sample.
///
/// # Arguments
/// * `counterfactual` - Generated counterfactual example
/// * `training_data` - Training dataset for reference
pub fn counterfactual_validity(
    counterfactual: &[f64],
    training_data: &[Vec<f64>],
) -> Result<f64, TorshError> {
    if training_data.is_empty() {
        return Err(TorshError::InvalidArgument(
            "Empty training data".to_string(),
        ));
    }

    let n_features = counterfactual.len();
    if !training_data.iter().all(|x| x.len() == n_features) {
        return Err(TorshError::InvalidArgument(
            "Inconsistent feature counts".to_string(),
        ));
    }

    // Find minimum distance to training examples
    let mut min_distance = f64::INFINITY;

    for training_point in training_data {
        let distance = euclidean_distance(counterfactual, training_point);
        min_distance = min_distance.min(distance);
    }

    // Normalize by feature space dimension
    let normalized_distance = min_distance / (n_features as f64).sqrt();

    // Transform to validity score (closer = more valid)
    // Using exp(-d) transformation
    Ok((-normalized_distance).exp())
}

/// Euclidean distance
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Comprehensive explainability metrics
#[derive(Debug, Clone)]
pub struct ExplainabilityMetrics {
    pub feature_stability: f64,
    pub attribution_agreement: f64,
    pub faithfulness: f64,
    pub completeness: f64,
    pub average_monotonicity: f64,
}

impl ExplainabilityMetrics {
    /// Create new explainability metrics report
    pub fn new(
        importances: &[Vec<f64>],
        attributions: &[Vec<f64>],
        perturbation_effects: &[f64],
        true_importances: &[f64],
        top_k: usize,
    ) -> Result<Self, TorshError> {
        let feature_stability = feature_importance_stability(importances)?;
        let attribution_agreement_score = attribution_agreement(attributions)?;

        // Use first attribution method for faithfulness
        let faithfulness = if !attributions.is_empty() {
            explanation_faithfulness(&attributions[0], perturbation_effects)?
        } else {
            0.0
        };

        // Use first attribution method for completeness
        let completeness = if !attributions.is_empty() {
            explanation_completeness(&attributions[0], true_importances, top_k)?
        } else {
            0.0
        };

        Ok(Self {
            feature_stability,
            attribution_agreement: attribution_agreement_score,
            faithfulness,
            completeness,
            average_monotonicity: 0.0, // Would need additional data to compute
        })
    }

    /// Format as string
    pub fn format(&self) -> String {
        let mut report = String::new();
        report.push_str("╔═══════════════════════════════════════════════╗\n");
        report.push_str("║       Explainability Metrics Report           ║\n");
        report.push_str("╠═══════════════════════════════════════════════╣\n");
        report.push_str(&format!(
            "║ Feature Stability:     {:.4}                 ║\n",
            self.feature_stability
        ));
        report.push_str(&format!(
            "║ Attribution Agreement: {:.4}                 ║\n",
            self.attribution_agreement
        ));
        report.push_str(&format!(
            "║ Faithfulness:          {:.4}                 ║\n",
            self.faithfulness
        ));
        report.push_str(&format!(
            "║ Completeness:          {:.4}                 ║\n",
            self.completeness
        ));
        report.push_str("╚═══════════════════════════════════════════════╝\n");

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_importance_stability_perfect() {
        let importances = vec![
            vec![0.5, 0.3, 0.2],
            vec![0.5, 0.3, 0.2],
            vec![0.5, 0.3, 0.2],
        ];

        let stability = feature_importance_stability(&importances).unwrap();
        assert!(stability > 0.99); // Should be very close to 1.0
    }

    #[test]
    fn test_feature_importance_stability_variable() {
        let importances = vec![
            vec![0.5, 0.3, 0.2],
            vec![0.3, 0.5, 0.2],
            vec![0.2, 0.3, 0.5],
        ];

        let stability = feature_importance_stability(&importances).unwrap();
        assert!(stability >= 0.0 && stability <= 1.0);
        assert!(stability < 0.99); // Should not be perfect
    }

    #[test]
    fn test_attribution_agreement_perfect() {
        let attributions = vec![vec![1.0, 0.5, 0.3], vec![1.0, 0.5, 0.3]];

        let agreement = attribution_agreement(&attributions).unwrap();
        assert!((agreement - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_attribution_agreement_opposite() {
        let attributions = vec![vec![1.0, 0.5, 0.3], vec![-1.0, -0.5, -0.3]];

        let agreement = attribution_agreement(&attributions).unwrap();
        assert!(agreement < 0.0); // Negative correlation
    }

    #[test]
    fn test_explanation_faithfulness() {
        let attributions = vec![0.8, 0.5, 0.2, 0.1];
        let perturbations = vec![0.9, 0.6, 0.3, 0.15];

        let faithfulness = explanation_faithfulness(&attributions, &perturbations).unwrap();
        assert!(faithfulness > 0.5); // Should be fairly faithful
    }

    #[test]
    fn test_explanation_completeness() {
        let attributions = vec![0.8, 0.6, 0.3, 0.1, 0.05];
        let true_importances = vec![0.9, 0.7, 0.2, 0.15, 0.05];

        let completeness = explanation_completeness(&attributions, &true_importances, 3).unwrap();
        assert!(completeness > 0.5); // Should identify most important features
    }

    #[test]
    fn test_feature_monotonicity_perfect() {
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let monotonicity = feature_monotonicity(&features, &predictions).unwrap();
        assert!((monotonicity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_feature_monotonicity_non_monotonic() {
        let features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let predictions = vec![1.0, 3.0, 2.0, 5.0, 4.0];

        let monotonicity = feature_monotonicity(&features, &predictions).unwrap();
        assert!(monotonicity < 1.0);
    }

    #[test]
    fn test_interaction_strength_no_interaction() {
        let individual = 5.0;
        let joint = 5.0;

        let strength = interaction_strength(individual, joint);
        assert!(strength.abs() < 1e-6);
    }

    #[test]
    fn test_interaction_strength_high_interaction() {
        let individual = 5.0;
        let joint = 10.0;

        let strength = interaction_strength(individual, joint);
        assert!(strength > 0.4);
    }

    #[test]
    fn test_counterfactual_validity() {
        let counterfactual = vec![1.0, 2.0, 3.0];
        let training_data = vec![
            vec![1.1, 2.1, 3.1],
            vec![5.0, 6.0, 7.0],
            vec![10.0, 11.0, 12.0],
        ];

        let validity = counterfactual_validity(&counterfactual, &training_data).unwrap();
        assert!(validity > 0.5); // Close to training data
    }

    #[test]
    fn test_explainability_metrics_report() {
        let importances = vec![vec![0.5, 0.3, 0.2], vec![0.5, 0.3, 0.2]];
        let attributions = vec![vec![0.8, 0.5, 0.2], vec![0.8, 0.5, 0.2]];
        let perturbations = vec![0.9, 0.6, 0.3];
        let true_importances = vec![0.9, 0.6, 0.3];

        let metrics = ExplainabilityMetrics::new(
            &importances,
            &attributions,
            &perturbations,
            &true_importances,
            2,
        )
        .unwrap();

        let formatted = metrics.format();
        assert!(formatted.contains("Explainability Metrics Report"));
    }
}
