//! Advanced statistical analysis for quantization

use crate::QScheme;
use std::collections::HashMap;

/// Advanced statistical analysis for quantization
pub struct AdvancedStatisticalAnalyzer {
    /// Sample size for statistical tests
    pub sample_size: usize,
    /// Confidence level for statistical tests (default: 0.95)
    pub confidence_level: f32,
}

impl Default for AdvancedStatisticalAnalyzer {
    fn default() -> Self {
        Self {
            sample_size: 1000,
            confidence_level: 0.95,
        }
    }
}

impl AdvancedStatisticalAnalyzer {
    /// Create a new statistical analyzer
    pub fn new(sample_size: usize, confidence_level: f32) -> Self {
        Self {
            sample_size,
            confidence_level,
        }
    }

    /// Perform statistical significance test
    pub fn test_significance(
        &self,
        baseline: &[f32],
        quantized: &[f32],
    ) -> StatisticalSignificance {
        let baseline_mean = Self::calculate_mean(baseline);
        let quantized_mean = Self::calculate_mean(quantized);
        let baseline_std = Self::calculate_std_dev(baseline, baseline_mean);
        let quantized_std = Self::calculate_std_dev(quantized, quantized_mean);

        // Simplified t-test calculation
        let pooled_std = ((baseline_std.powi(2) + quantized_std.powi(2)) / 2.0).sqrt();
        let t_statistic = (baseline_mean - quantized_mean)
            / (pooled_std
                * ((1.0 / baseline.len() as f32) + (1.0 / quantized.len() as f32)).sqrt());

        let p_value = Self::calculate_p_value(t_statistic.abs());
        let is_significant = p_value < (1.0 - self.confidence_level);

        StatisticalSignificance {
            t_statistic,
            p_value,
            is_significant,
            confidence_level: self.confidence_level,
            effect_size: (baseline_mean - quantized_mean).abs() / pooled_std,
        }
    }

    /// Generate comprehensive statistical report
    pub fn generate_comprehensive_report(
        &self,
        baseline_accuracy: &[f32],
        quantized_accuracy: &[f32],
        schemes: &[QScheme],
    ) -> ComprehensiveStatisticalReport {
        let mut scheme_analysis = HashMap::new();

        for &scheme in schemes {
            let significance = self.test_significance(baseline_accuracy, quantized_accuracy);
            let risk_level = self.assess_risk_level(&significance);

            scheme_analysis.insert(scheme, (significance, risk_level));
        }

        let overall_mean_baseline = Self::calculate_mean(baseline_accuracy);
        let overall_mean_quantized = Self::calculate_mean(quantized_accuracy);
        let overall_variance_baseline = Self::calculate_variance(baseline_accuracy);
        let overall_variance_quantized = Self::calculate_variance(quantized_accuracy);

        ComprehensiveStatisticalReport {
            overall_mean_baseline,
            overall_mean_quantized,
            overall_variance_baseline,
            overall_variance_quantized,
            scheme_analysis,
            sample_size: self.sample_size,
            confidence_level: self.confidence_level,
        }
    }

    /// Assess risk level based on statistical significance
    pub fn assess_risk_level(&self, significance: &StatisticalSignificance) -> RiskLevel {
        if !significance.is_significant {
            RiskLevel::Low
        } else if significance.effect_size < 0.2 {
            RiskLevel::Low
        } else if significance.effect_size < 0.5 {
            RiskLevel::Medium
        } else if significance.effect_size < 0.8 {
            RiskLevel::High
        } else {
            RiskLevel::Critical
        }
    }

    // Helper methods
    fn calculate_mean(values: &[f32]) -> f32 {
        values.iter().sum::<f32>() / values.len() as f32
    }

    fn calculate_std_dev(values: &[f32], mean: f32) -> f32 {
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        variance.sqrt()
    }

    fn calculate_variance(values: &[f32]) -> f32 {
        let mean = Self::calculate_mean(values);
        values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32
    }

    fn calculate_p_value(t_stat: f32) -> f32 {
        // Simplified p-value calculation (approximation)
        if t_stat < 1.96 {
            0.05
        } else if t_stat < 2.58 {
            0.01
        } else {
            0.001
        }
    }
}

/// Statistical significance test results
#[derive(Debug, Clone)]
pub struct StatisticalSignificance {
    /// T-statistic value
    pub t_statistic: f32,
    /// P-value of the test
    pub p_value: f32,
    /// Whether the difference is statistically significant
    pub is_significant: bool,
    /// Confidence level used for the test
    pub confidence_level: f32,
    /// Effect size (Cohen's d)
    pub effect_size: f32,
}

/// Comprehensive statistical report
#[derive(Debug, Clone)]
pub struct ComprehensiveStatisticalReport {
    /// Overall mean baseline accuracy
    pub overall_mean_baseline: f32,
    /// Overall mean quantized accuracy
    pub overall_mean_quantized: f32,
    /// Overall variance in baseline accuracy
    pub overall_variance_baseline: f32,
    /// Overall variance in quantized accuracy
    pub overall_variance_quantized: f32,
    /// Analysis for each quantization scheme
    pub scheme_analysis: HashMap<QScheme, (StatisticalSignificance, RiskLevel)>,
    /// Sample size used for analysis
    pub sample_size: usize,
    /// Confidence level used
    pub confidence_level: f32,
}

/// Risk level assessment for quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    /// Low risk - minimal impact on accuracy
    Low,
    /// Medium risk - moderate impact on accuracy
    Medium,
    /// High risk - significant impact on accuracy
    High,
    /// Critical risk - severe impact on accuracy
    Critical,
}
