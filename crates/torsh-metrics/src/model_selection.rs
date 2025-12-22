//! Model selection metrics for choosing optimal models
//!
//! This module provides information criteria and model comparison metrics
//! for selecting the best model among competing alternatives.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Akaike Information Criterion (AIC)
///
/// AIC = 2k - 2ln(L)
/// where k is the number of parameters and L is the likelihood
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIC {
    /// Number of parameters in the model
    pub n_parameters: usize,
    /// Log-likelihood of the model
    pub log_likelihood: f64,
    /// AIC value
    pub value: f64,
}

impl AIC {
    /// Compute AIC given the number of parameters and log-likelihood
    pub fn compute(n_parameters: usize, log_likelihood: f64) -> Self {
        let value = 2.0 * n_parameters as f64 - 2.0 * log_likelihood;
        Self {
            n_parameters,
            log_likelihood,
            value,
        }
    }

    /// Compute AIC from residuals assuming normal distribution
    ///
    /// For linear regression: AIC = n*ln(RSS/n) + 2k
    /// where RSS is residual sum of squares
    pub fn from_residuals(residuals: &[f64], n_parameters: usize) -> Self {
        let n = residuals.len() as f64;
        let rss: f64 = residuals.iter().map(|r| r * r).sum();
        let log_likelihood =
            -(n / 2.0) * (1.0 + (2.0 * std::f64::consts::PI).ln()) - (n / 2.0) * (rss / n).ln();
        Self::compute(n_parameters, log_likelihood)
    }
}

/// Corrected Akaike Information Criterion (AICc)
///
/// AICc = AIC + 2k(k+1)/(n-k-1)
/// Used for small sample sizes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AICc {
    /// Underlying AIC
    pub aic: AIC,
    /// Number of samples
    pub n_samples: usize,
    /// AICc value
    pub value: f64,
}

impl AICc {
    /// Compute AICc given AIC and sample size
    pub fn compute(aic: AIC, n_samples: usize) -> Self {
        let k = aic.n_parameters as f64;
        let n = n_samples as f64;
        let correction = if n > k + 1.0 {
            2.0 * k * (k + 1.0) / (n - k - 1.0)
        } else {
            f64::INFINITY // Not enough samples
        };
        let value = aic.value + correction;
        Self {
            aic,
            n_samples,
            value,
        }
    }

    /// Compute AICc from residuals
    pub fn from_residuals(residuals: &[f64], n_parameters: usize) -> Self {
        let aic = AIC::from_residuals(residuals, n_parameters);
        Self::compute(aic, residuals.len())
    }
}

/// Bayesian Information Criterion (BIC)
///
/// BIC = k*ln(n) - 2ln(L)
/// where k is parameters, n is samples, L is likelihood
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BIC {
    /// Number of parameters
    pub n_parameters: usize,
    /// Number of samples
    pub n_samples: usize,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// BIC value
    pub value: f64,
}

impl BIC {
    /// Compute BIC
    pub fn compute(n_parameters: usize, n_samples: usize, log_likelihood: f64) -> Self {
        let value = (n_parameters as f64) * (n_samples as f64).ln() - 2.0 * log_likelihood;
        Self {
            n_parameters,
            n_samples,
            log_likelihood,
            value,
        }
    }

    /// Compute BIC from residuals assuming normal distribution
    pub fn from_residuals(residuals: &[f64], n_parameters: usize) -> Self {
        let n = residuals.len() as f64;
        let rss: f64 = residuals.iter().map(|r| r * r).sum();
        let log_likelihood =
            -(n / 2.0) * (1.0 + (2.0 * std::f64::consts::PI).ln()) - (n / 2.0) * (rss / n).ln();
        Self::compute(n_parameters, residuals.len(), log_likelihood)
    }
}

/// Hannan-Quinn Information Criterion (HQIC)
///
/// HQIC = -2ln(L) + 2k*ln(ln(n))
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HQIC {
    /// Number of parameters
    pub n_parameters: usize,
    /// Number of samples
    pub n_samples: usize,
    /// Log-likelihood
    pub log_likelihood: f64,
    /// HQIC value
    pub value: f64,
}

impl HQIC {
    /// Compute HQIC
    pub fn compute(n_parameters: usize, n_samples: usize, log_likelihood: f64) -> Self {
        let n = n_samples as f64;
        let k = n_parameters as f64;
        let value = -2.0 * log_likelihood + 2.0 * k * n.ln().ln();
        Self {
            n_parameters,
            n_samples,
            log_likelihood,
            value,
        }
    }

    /// Compute HQIC from residuals
    pub fn from_residuals(residuals: &[f64], n_parameters: usize) -> Self {
        let n = residuals.len() as f64;
        let rss: f64 = residuals.iter().map(|r| r * r).sum();
        let log_likelihood =
            -(n / 2.0) * (1.0 + (2.0 * std::f64::consts::PI).ln()) - (n / 2.0) * (rss / n).ln();
        Self::compute(n_parameters, residuals.len(), log_likelihood)
    }
}

/// Model comparison report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparisonReport {
    /// Model name
    pub model_name: String,
    /// AIC value
    pub aic: f64,
    /// AICc value
    pub aicc: f64,
    /// BIC value
    pub bic: f64,
    /// HQIC value
    pub hqic: f64,
    /// Number of parameters
    pub n_parameters: usize,
    /// Number of samples
    pub n_samples: usize,
    /// Log-likelihood
    pub log_likelihood: f64,
}

impl ModelComparisonReport {
    /// Create a comparison report for a model
    pub fn new(
        model_name: String,
        n_parameters: usize,
        n_samples: usize,
        log_likelihood: f64,
    ) -> Self {
        let aic_obj = AIC::compute(n_parameters, log_likelihood);
        let aicc_obj = AICc::compute(aic_obj.clone(), n_samples);
        let bic_obj = BIC::compute(n_parameters, n_samples, log_likelihood);
        let hqic_obj = HQIC::compute(n_parameters, n_samples, log_likelihood);

        Self {
            model_name,
            aic: aic_obj.value,
            aicc: aicc_obj.value,
            bic: bic_obj.value,
            hqic: hqic_obj.value,
            n_parameters,
            n_samples,
            log_likelihood,
        }
    }

    /// Create from residuals
    pub fn from_residuals(model_name: String, residuals: &[f64], n_parameters: usize) -> Self {
        let n = residuals.len() as f64;
        let rss: f64 = residuals.iter().map(|r| r * r).sum();
        let log_likelihood =
            -(n / 2.0) * (1.0 + (2.0 * std::f64::consts::PI).ln()) - (n / 2.0) * (rss / n).ln();
        Self::new(model_name, n_parameters, residuals.len(), log_likelihood)
    }
}

/// Multi-model comparison table
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModelComparison {
    /// Individual model reports
    pub models: Vec<ModelComparisonReport>,
    /// Best model by AIC
    pub best_by_aic: Option<String>,
    /// Best model by BIC
    pub best_by_bic: Option<String>,
    /// AIC weights (probability each model is the best)
    pub aic_weights: HashMap<String, f64>,
}

impl MultiModelComparison {
    /// Create a multi-model comparison
    pub fn new(models: Vec<ModelComparisonReport>) -> Self {
        if models.is_empty() {
            return Self {
                models,
                best_by_aic: None,
                best_by_bic: None,
                aic_weights: HashMap::new(),
            };
        }

        // Find best models
        let best_by_aic = models
            .iter()
            .min_by(|a, b| a.aic.partial_cmp(&b.aic).unwrap())
            .map(|m| m.model_name.clone());

        let best_by_bic = models
            .iter()
            .min_by(|a, b| a.bic.partial_cmp(&b.bic).unwrap())
            .map(|m| m.model_name.clone());

        // Compute AIC weights
        let min_aic = models.iter().map(|m| m.aic).fold(f64::INFINITY, f64::min);
        let delta_aics: Vec<_> = models.iter().map(|m| m.aic - min_aic).collect();
        let likelihood_sum: f64 = delta_aics.iter().map(|d| (-0.5 * d).exp()).sum();

        let mut aic_weights = HashMap::new();
        for (model, delta) in models.iter().zip(delta_aics.iter()) {
            let weight = (-0.5 * delta).exp() / likelihood_sum;
            aic_weights.insert(model.model_name.clone(), weight);
        }

        Self {
            models,
            best_by_aic,
            best_by_bic,
            aic_weights,
        }
    }

    /// Generate a formatted comparison table
    pub fn to_table(&self) -> String {
        let mut table = String::new();
        table.push_str("| Model | AIC | AICc | BIC | HQIC | Params | AIC Weight |\n");
        table.push_str("|-------|-----|------|-----|------|--------|------------|\n");

        for model in &self.models {
            let weight = self.aic_weights.get(&model.model_name).unwrap_or(&0.0);
            table.push_str(&format!(
                "| {} | {:.2} | {:.2} | {:.2} | {:.2} | {} | {:.4} |\n",
                model.model_name,
                model.aic,
                model.aicc,
                model.bic,
                model.hqic,
                model.n_parameters,
                weight
            ));
        }

        table.push_str(&format!(
            "\nBest by AIC: {}\n",
            self.best_by_aic.as_ref().unwrap_or(&"N/A".to_string())
        ));
        table.push_str(&format!(
            "Best by BIC: {}\n",
            self.best_by_bic.as_ref().unwrap_or(&"N/A".to_string())
        ));

        table
    }
}

/// Cross-validation score types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CVScoreType {
    /// Mean Absolute Error
    MAE,
    /// Mean Squared Error
    MSE,
    /// Root Mean Squared Error
    RMSE,
    /// R-squared
    R2,
    /// Accuracy (for classification)
    Accuracy,
    /// F1 Score (for classification)
    F1,
}

/// Cross-validation results for model selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVModelSelection {
    /// Model name
    pub model_name: String,
    /// CV scores for each fold
    pub fold_scores: Vec<f64>,
    /// Mean CV score
    pub mean_score: f64,
    /// Standard deviation of CV scores
    pub std_score: f64,
    /// Standard error
    pub std_error: f64,
    /// 95% confidence interval
    pub confidence_interval_95: (f64, f64),
    /// Score type
    pub score_type: CVScoreType,
}

impl CVModelSelection {
    /// Create CV model selection results
    pub fn new(model_name: String, fold_scores: Vec<f64>, score_type: CVScoreType) -> Self {
        let n = fold_scores.len() as f64;
        let mean_score = fold_scores.iter().sum::<f64>() / n;
        let variance = fold_scores
            .iter()
            .map(|s| (s - mean_score).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        let std_score = variance.sqrt();
        let std_error = std_score / n.sqrt();

        // 95% CI using t-distribution approximation (z=1.96 for large n)
        let margin = 1.96 * std_error;
        let confidence_interval_95 = (mean_score - margin, mean_score + margin);

        Self {
            model_name,
            fold_scores,
            mean_score,
            std_score,
            std_error,
            confidence_interval_95,
            score_type,
        }
    }
}

/// Compare multiple models using cross-validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVModelComparison {
    /// Individual model CV results
    pub models: Vec<CVModelSelection>,
    /// Best model by mean score
    pub best_model: Option<String>,
}

impl CVModelComparison {
    /// Create a CV model comparison
    ///
    /// # Arguments
    /// * `models` - CV results for each model
    /// * `higher_is_better` - True if higher scores are better (e.g., accuracy, R2)
    pub fn new(models: Vec<CVModelSelection>, higher_is_better: bool) -> Self {
        if models.is_empty() {
            return Self {
                models,
                best_model: None,
            };
        }

        let best_model = if higher_is_better {
            models
                .iter()
                .max_by(|a, b| a.mean_score.partial_cmp(&b.mean_score).unwrap())
                .map(|m| m.model_name.clone())
        } else {
            models
                .iter()
                .min_by(|a, b| a.mean_score.partial_cmp(&b.mean_score).unwrap())
                .map(|m| m.model_name.clone())
        };

        Self { models, best_model }
    }

    /// Generate a formatted comparison table
    pub fn to_table(&self) -> String {
        let mut table = String::new();
        table.push_str("| Model | Mean Score | Std Dev | Std Error | 95% CI |\n");
        table.push_str("|-------|------------|---------|-----------|--------|\n");

        for model in &self.models {
            table.push_str(&format!(
                "| {} | {:.4} | {:.4} | {:.4} | ({:.4}, {:.4}) |\n",
                model.model_name,
                model.mean_score,
                model.std_score,
                model.std_error,
                model.confidence_interval_95.0,
                model.confidence_interval_95.1
            ));
        }

        table.push_str(&format!(
            "\nBest Model: {}\n",
            self.best_model.as_ref().unwrap_or(&"N/A".to_string())
        ));

        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aic() {
        // Example: linear regression with 3 parameters
        let residuals = vec![0.1, -0.2, 0.15, -0.1, 0.05, 0.0, -0.05];
        let aic = AIC::from_residuals(&residuals, 3);

        assert_eq!(aic.n_parameters, 3);
        assert!(aic.value.is_finite());
        assert!(aic.log_likelihood.is_finite());
    }

    #[test]
    fn test_aicc() {
        let residuals = vec![0.1, -0.2, 0.15, -0.1, 0.05];
        let aicc = AICc::from_residuals(&residuals, 3);

        assert!(aicc.value > aicc.aic.value); // Correction should increase AIC
        assert_eq!(aicc.n_samples, 5);
    }

    #[test]
    fn test_bic() {
        let residuals = vec![0.1, -0.2, 0.15, -0.1, 0.05, 0.0, -0.05, 0.1, -0.1, 0.05];
        let bic = BIC::from_residuals(&residuals, 3);

        assert_eq!(bic.n_parameters, 3);
        assert_eq!(bic.n_samples, 10);
        assert!(bic.value.is_finite());
    }

    #[test]
    fn test_model_comparison() {
        let model1 = ModelComparisonReport::from_residuals(
            "Model1".to_string(),
            &vec![0.1, -0.2, 0.15, -0.1, 0.05],
            2,
        );

        let model2 = ModelComparisonReport::from_residuals(
            "Model2".to_string(),
            &vec![0.1, -0.2, 0.15, -0.1, 0.05],
            4,
        );

        let comparison = MultiModelComparison::new(vec![model1, model2]);

        assert!(comparison.best_by_aic.is_some());
        assert!(comparison.best_by_bic.is_some());
        assert_eq!(comparison.aic_weights.len(), 2);

        let total_weight: f64 = comparison.aic_weights.values().sum();
        assert!((total_weight - 1.0).abs() < 1e-6); // Weights should sum to 1
    }

    #[test]
    fn test_cv_model_selection() {
        let fold_scores = vec![0.85, 0.87, 0.84, 0.86, 0.88];
        let cv = CVModelSelection::new(
            "TestModel".to_string(),
            fold_scores.clone(),
            CVScoreType::Accuracy,
        );

        assert_eq!(cv.fold_scores.len(), 5);
        assert!((cv.mean_score - 0.86).abs() < 1e-6);
        assert!(cv.std_score > 0.0);
        assert!(cv.confidence_interval_95.0 < cv.mean_score);
        assert!(cv.confidence_interval_95.1 > cv.mean_score);
    }

    #[test]
    fn test_cv_model_comparison() {
        let model1 = CVModelSelection::new(
            "Model1".to_string(),
            vec![0.85, 0.87, 0.84],
            CVScoreType::Accuracy,
        );

        let model2 = CVModelSelection::new(
            "Model2".to_string(),
            vec![0.80, 0.82, 0.79],
            CVScoreType::Accuracy,
        );

        let comparison = CVModelComparison::new(vec![model1, model2], true);

        assert_eq!(comparison.best_model, Some("Model1".to_string()));
    }

    #[test]
    fn test_aic_weights() {
        // Create models with different AIC values
        let mut models = vec![];
        for i in 0..3 {
            let residuals = vec![0.1 * (i as f64), -0.2, 0.15, -0.1, 0.05];
            let model =
                ModelComparisonReport::from_residuals(format!("Model{}", i), &residuals, 2 + i);
            models.push(model);
        }

        let comparison = MultiModelComparison::new(models);
        let weights: Vec<f64> = comparison.aic_weights.values().copied().collect();

        // Best model (lowest AIC) should have highest weight
        assert!(weights.iter().any(|&w| w > 0.0));

        // All weights should be between 0 and 1
        assert!(weights.iter().all(|&w| w >= 0.0 && w <= 1.0));
    }
}
