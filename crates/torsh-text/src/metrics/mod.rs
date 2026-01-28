//! Comprehensive text metrics and evaluation suite
//!
//! This module provides a unified interface to various text evaluation metrics
//! commonly used in natural language processing, machine translation, text
//! summarization, and other text generation tasks.
//!
//! # Features
//!
//! - **BLEU scores**: Machine translation evaluation with various smoothing techniques
//! - **ROUGE scores**: Text summarization evaluation with multiple variants
//! - **Perplexity**: Language model evaluation with confidence intervals
//! - **Edit distances**: String similarity with multiple algorithms
//! - **BERTScore**: Contextual embedding-based evaluation
//! - **Semantic similarity**: Advanced semantic analysis with domain awareness
//! - **Word overlap**: Multiple overlap metrics with semantic extensions
//! - **Coherence analysis**: Multi-dimensional text coherence evaluation
//! - **Fluency assessment**: Comprehensive fluency analysis across multiple dimensions
//!
//! # Quick Start
//!
//! ```rust
//! use torsh_text::metrics::*;
//!
//! // Basic usage - single metrics
//! let bleu_score = calculate_bleu_simple("The cat sat on the mat", "A cat was sitting on the mat");
//! let rouge_score = calculate_rouge_simple("The quick brown fox", "The fast brown fox", RougeType::Rouge1);
//! let perplexity = calculate_perplexity_simple(&[0.1, 0.2, 0.7]);
//!
//! // Advanced usage - comprehensive evaluation
//! let evaluator = TextEvaluator::with_default_config();
//! let report = evaluator.comprehensive_evaluation("reference text", "candidate text");
//! println!("Overall quality: {:.3}", report.overall_score);
//! ```
//!
//! # Modules
//!
//! Each metric type has its own specialized module with advanced configuration
//! options and detailed analysis capabilities.

pub mod bert_score;
pub mod bleu;
pub mod coherence;
pub mod edit_distance;
pub mod fluency;
pub mod overlap;
pub mod perplexity;
pub mod rouge;
pub mod semantic;

// Re-export key types and functions for convenience
pub use bert_score::{BertScore, BertScoreConfig, BertScoreResult};
pub use bleu::{BleuMetrics, BleuScore};
pub use edit_distance::{DistanceAlgorithm, EditDistance, EditDistanceConfig};
pub use perplexity::{
    PerplexityCalculator, PerplexityConfig, SequencePerplexityMetrics, SmoothingMethod,
};
pub use rouge::{RougeMetrics, RougeScore, RougeType};
pub use semantic::{SemanticAnalysisConfig, SemanticAnalysisResult, SemanticAnalyzer};
pub use overlap::{OverlapConfig, OverlapResult, WordOverlapCalculator};
pub use coherence::{CoherenceAnalyzer, CoherenceConfig, CoherenceResult};
pub use fluency::{FluentTextAnalyzer as FluencyAnalyzer, AnalysisConfig as FluencyConfig, ComprehensiveFluencyAnalysis as FluencyResult};

use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_core::random::{rng, Random};
use std::collections::HashMap;

/// Comprehensive evaluation configuration for all metrics
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    pub bleu_config: BleuConfig,
    pub rouge_config: RougeConfig,
    pub perplexity_config: PerplexityConfig,
    pub edit_distance_config: EditDistanceConfig,
    pub bert_score_config: BertScoreConfig,
    pub semantic_config: SemanticAnalysisConfig,
    pub overlap_config: OverlapConfig,
    pub coherence_config: CoherenceConfig,
    pub fluency_config: FluencyConfig,
    pub weights: EvaluationWeights,
    pub enable_confidence_intervals: bool,
    pub enable_statistical_analysis: bool,
}

/// Configuration for BLEU score calculation
#[derive(Debug, Clone)]
pub struct BleuConfig {
    pub max_ngram: usize,
    pub smooth: bool,
}

impl Default for BleuConfig {
    fn default() -> Self {
        Self {
            max_ngram: 4,
            smooth: true,
        }
    }
}

/// Configuration for ROUGE score calculation
#[derive(Debug, Clone)]
pub struct RougeConfig {
    pub rouge_types: Vec<RougeType>,
    pub use_stemming: bool,
}

impl Default for RougeConfig {
    fn default() -> Self {
        Self {
            rouge_types: vec![RougeType::Rouge1, RougeType::Rouge2, RougeType::RougeL],
            use_stemming: false,
        }
    }
}

/// Weights for combining different metric scores
#[derive(Debug, Clone)]
pub struct EvaluationWeights {
    pub bleu_weight: f64,
    pub rouge_weight: f64,
    pub perplexity_weight: f64,
    pub edit_distance_weight: f64,
    pub bert_score_weight: f64,
    pub semantic_weight: f64,
    pub overlap_weight: f64,
    pub coherence_weight: f64,
    pub fluency_weight: f64,
}

impl Default for EvaluationWeights {
    fn default() -> Self {
        Self {
            bleu_weight: 0.15,
            rouge_weight: 0.15,
            perplexity_weight: 0.10,
            edit_distance_weight: 0.10,
            bert_score_weight: 0.15,
            semantic_weight: 0.10,
            overlap_weight: 0.05,
            coherence_weight: 0.10,
            fluency_weight: 0.10,
        }
    }
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            bleu_config: BleuConfig::default(),
            rouge_config: RougeConfig::default(),
            perplexity_config: PerplexityConfig::default(),
            edit_distance_config: EditDistanceConfig::default(),
            bert_score_config: BertScoreConfig::default(),
            semantic_config: SemanticAnalysisConfig::default(),
            overlap_config: OverlapConfig::default(),
            coherence_config: CoherenceConfig::default(),
            fluency_config: FluencyConfig::default(),
            weights: EvaluationWeights::default(),
            enable_confidence_intervals: true,
            enable_statistical_analysis: true,
        }
    }
}

/// Comprehensive evaluation result containing all metric scores
#[derive(Debug, Clone)]
pub struct ComprehensiveEvaluationResult {
    pub overall_score: f64,
    pub bleu_result: BleuResult,
    pub rouge_result: RougeResult,
    pub perplexity_result: Option<SequencePerplexityMetrics>,
    pub edit_distance_result: EditDistanceResult,
    pub bert_score_result: BertScoreResult,
    pub semantic_result: SemanticAnalysisResult,
    pub overlap_result: OverlapResult,
    pub coherence_result: CoherenceResult,
    pub fluency_result: FluencyResult,
    pub metric_breakdown: HashMap<String, f64>,
    pub confidence_intervals: Option<ConfidenceIntervals>,
    pub statistical_summary: Option<StatisticalSummary>,
    pub quality_assessment: QualityAssessment,
}

/// BLEU evaluation result
#[derive(Debug, Clone)]
pub struct BleuResult {
    pub bleu_score: f64,
    pub precision_scores: Vec<f64>,
}

/// ROUGE evaluation result
#[derive(Debug, Clone)]
pub struct RougeResult {
    pub rouge_score: f64,
    pub rouge1: f64,
    pub rouge2: f64,
    pub rougeL: f64,
}

/// Edit distance evaluation result
#[derive(Debug, Clone)]
pub struct EditDistanceResult {
    pub distance: usize,
    pub normalized_distance: f64,
    pub similarity: f64,
}

/// Confidence intervals for metric reliability
#[derive(Debug, Clone)]
pub struct ConfidenceIntervals {
    pub overall_score_ci: (f64, f64),
    pub bleu_ci: (f64, f64),
    pub rouge_ci: (f64, f64),
    pub bert_score_ci: (f64, f64),
    pub semantic_ci: (f64, f64),
    pub coherence_ci: (f64, f64),
    pub fluency_ci: (f64, f64),
    pub confidence_level: f64,
}

/// Statistical summary of evaluation results
#[derive(Debug, Clone)]
pub struct StatisticalSummary {
    pub metric_correlations: HashMap<(String, String), f64>,
    pub metric_reliability: HashMap<String, f64>,
    pub outlier_detection: Vec<String>,
    pub consistency_score: f64,
    pub variance_analysis: HashMap<String, f64>,
}

/// Overall quality assessment
#[derive(Debug, Clone)]
pub struct QualityAssessment {
    pub quality_level: QualityLevel,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub recommendations: Vec<String>,
    pub overall_grade: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum QualityLevel {
    Excellent,
    Good,
    Fair,
    Poor,
    VeryPoor,
}

/// Main text evaluator with unified interface
pub struct TextEvaluator {
    config: EvaluationConfig,
    bleu_calculator: BleuScore,
    rouge_calculator: RougeScore,
    perplexity_calculator: PerplexityCalculator,
    edit_distance_calculator: EditDistance,
    bert_score_calculator: BertScore,
    semantic_calculator: SemanticAnalyzer,
    overlap_calculator: WordOverlapCalculator,
    coherence_analyzer: CoherenceAnalyzer,
    fluency_analyzer: FluencyAnalyzer,
}

impl TextEvaluator {
    /// Create a new text evaluator with custom configuration
    pub fn new(config: EvaluationConfig) -> Self {
        let bleu_calculator = BleuScore::default();
        let rouge_calculator = RougeScore::default();
        let perplexity_calculator = PerplexityCalculator::new(config.perplexity_config.clone());
        let edit_distance_calculator = EditDistance::new(config.edit_distance_config.clone());
        let bert_score_calculator = BertScore::new(config.bert_score_config.clone());
        let semantic_calculator = SemanticAnalyzer::new(config.semantic_config.clone());
        let overlap_calculator = WordOverlapCalculator::new(config.overlap_config.clone());
        let coherence_analyzer = CoherenceAnalyzer::new(config.coherence_config.clone());
        let fluency_analyzer = FluencyAnalyzer::with_config(config.fluency_config.clone());

        Self {
            config,
            bleu_calculator,
            rouge_calculator,
            perplexity_calculator,
            edit_distance_calculator,
            bert_score_calculator,
            semantic_calculator,
            overlap_calculator,
            coherence_analyzer,
            fluency_analyzer,
        }
    }

    /// Create a new text evaluator with default configuration
    pub fn with_default_config() -> Self {
        Self::new(EvaluationConfig::default())
    }

    /// Perform comprehensive evaluation of candidate text against reference
    pub fn comprehensive_evaluation(
        &self,
        reference: &str,
        candidate: &str,
    ) -> ComprehensiveEvaluationResult {
        // Calculate individual metrics
        let bleu_result = self.bleu_calculator.calculate_bleu(reference, candidate);
        let rouge_result =
            self.rouge_calculator
                .calculate_rouge(reference, candidate, RougeType::Rouge1);
        let edit_distance_result = self
            .edit_distance_calculator
            .calculate_distance(reference, candidate);
        let bert_score_result = self
            .bert_score_calculator
            .calculate_bert_score(reference, candidate);
        let semantic_result = self
            .semantic_calculator
            .calculate_similarity(reference, candidate);
        let overlap_result = self
            .overlap_calculator
            .calculate_comprehensive_overlap(reference, candidate);
        let coherence_result = self.coherence_analyzer.analyze_coherence(candidate);
        let fluency_result = self.fluency_analyzer.analyze_fluency(candidate);

        // Perplexity calculation (optional, requires probability distributions)
        let perplexity_result = None; // Would require actual language model probabilities

        // Calculate overall score
        let overall_score = self.calculate_overall_score(
            &bleu_result,
            &rouge_result,
            &edit_distance_result,
            &bert_score_result,
            &semantic_result,
            &overlap_result,
            &coherence_result,
            &fluency_result,
        );

        // Create metric breakdown
        let metric_breakdown = self.create_metric_breakdown(
            &bleu_result,
            &rouge_result,
            &edit_distance_result,
            &bert_score_result,
            &semantic_result,
            &overlap_result,
            &coherence_result,
            &fluency_result,
        );

        // Calculate confidence intervals if enabled
        let confidence_intervals = if self.config.enable_confidence_intervals {
            Some(self.calculate_confidence_intervals(
                &bleu_result,
                &rouge_result,
                &bert_score_result,
                &semantic_result,
                &coherence_result,
                &fluency_result,
            ))
        } else {
            None
        };

        // Perform statistical analysis if enabled
        let statistical_summary = if self.config.enable_statistical_analysis {
            Some(self.calculate_statistical_summary(&metric_breakdown))
        } else {
            None
        };

        // Generate quality assessment
        let quality_assessment = self.assess_quality(overall_score, &metric_breakdown);

        ComprehensiveEvaluationResult {
            overall_score,
            bleu_result,
            rouge_result,
            perplexity_result,
            edit_distance_result,
            bert_score_result,
            semantic_result,
            overlap_result,
            coherence_result,
            fluency_result,
            metric_breakdown,
            confidence_intervals,
            statistical_summary,
            quality_assessment,
        }
    }

    /// Evaluate multiple candidate texts against references
    pub fn batch_evaluation(
        &self,
        references: &[String],
        candidates: &[String],
    ) -> Vec<ComprehensiveEvaluationResult> {
        assert_eq!(references.len(), candidates.len());

        references
            .iter()
            .zip(candidates.iter())
            .map(|(ref_text, cand_text)| self.comprehensive_evaluation(ref_text, cand_text))
            .collect()
    }

    /// Compare multiple candidate texts and rank them
    pub fn comparative_evaluation(
        &self,
        reference: &str,
        candidates: &[String],
    ) -> ComparativeEvaluationResult {
        let evaluations: Vec<ComprehensiveEvaluationResult> = candidates
            .iter()
            .map(|candidate| self.comprehensive_evaluation(reference, candidate))
            .collect();

        // Rank candidates by overall score
        let mut ranked_indices: Vec<usize> = (0..candidates.len()).collect();
        ranked_indices.sort_by(|&a, &b| {
            evaluations[b]
                .overall_score
                .partial_cmp(&evaluations[a].overall_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best_candidate = ranked_indices[0];
        let worst_candidate = *ranked_indices.last().unwrap();

        let score_differences = self.calculate_score_differences(&evaluations);
        let statistical_significance = self.test_statistical_significance(&evaluations);

        ComparativeEvaluationResult {
            evaluations,
            ranked_indices,
            best_candidate,
            worst_candidate,
            score_differences,
            statistical_significance,
        }
    }

    /// Analyze text quality across multiple dimensions
    pub fn quality_analysis(&self, text: &str) -> QualityAnalysisResult {
        let coherence_result = self.coherence_analyzer.analyze_coherence(text);
        let fluency_result = self.fluency_analyzer.analyze_fluency(text);

        let readability_score = fluency_result.prosodic_fluency.reading_ease;
        let complexity_score = fluency_result.syntactic_fluency.syntactic_complexity;
        let naturalness_score = fluency_result.quality_indicators.naturalness_score;
        let coherence_score = coherence_result.overall_coherence;

        let overall_quality = (readability_score + naturalness_score + coherence_score) / 3.0;

        let quality_dimensions =
            self.analyze_quality_dimensions(&coherence_result, &fluency_result);

        QualityAnalysisResult {
            overall_quality,
            readability_score,
            complexity_score,
            naturalness_score,
            coherence_score,
            quality_dimensions,
            coherence_result,
            fluency_result,
        }
    }

    fn calculate_overall_score(
        &self,
        bleu: &BleuResult,
        rouge: &RougeResult,
        edit_distance: &EditDistanceResult,
        bert_score: &BertScoreResult,
        semantic: &SemanticResult,
        overlap: &OverlapResult,
        coherence: &CoherenceResult,
        fluency: &FluencyResult,
    ) -> f64 {
        let weights = &self.config.weights;

        let bleu_score = bleu.bleu_score;
        let rouge_score = rouge.rouge_score;
        let edit_distance_score = 1.0 - (edit_distance.normalized_distance / 100.0).min(1.0);
        let bert_score_score = bert_score.f1_score;
        let semantic_score = semantic.overall_similarity;
        let overlap_score = overlap.jaccard;
        let coherence_score = coherence.overall_coherence;
        let fluency_score = fluency.overall_fluency;

        (bleu_score * weights.bleu_weight)
            + (rouge_score * weights.rouge_weight)
            + (edit_distance_score * weights.edit_distance_weight)
            + (bert_score_score * weights.bert_score_weight)
            + (semantic_score * weights.semantic_weight)
            + (overlap_score * weights.overlap_weight)
            + (coherence_score * weights.coherence_weight)
            + (fluency_score * weights.fluency_weight)
    }

    fn create_metric_breakdown(
        &self,
        bleu: &BleuResult,
        rouge: &RougeResult,
        edit_distance: &EditDistanceResult,
        bert_score: &BertScoreResult,
        semantic: &SemanticResult,
        overlap: &OverlapResult,
        coherence: &CoherenceResult,
        fluency: &FluencyResult,
    ) -> HashMap<String, f64> {
        let mut breakdown = HashMap::new();

        breakdown.insert("bleu_score".to_string(), bleu.bleu_score);
        breakdown.insert("rouge_score".to_string(), rouge.rouge_score);
        breakdown.insert(
            "edit_distance_similarity".to_string(),
            1.0 - (edit_distance.normalized_distance / 100.0).min(1.0),
        );
        breakdown.insert("bert_f1_score".to_string(), bert_score.f1_score);
        breakdown.insert("bert_precision".to_string(), bert_score.precision);
        breakdown.insert("bert_recall".to_string(), bert_score.recall);
        breakdown.insert(
            "semantic_similarity".to_string(),
            semantic.overall_similarity,
        );
        breakdown.insert("word_overlap_jaccard".to_string(), overlap.jaccard);
        breakdown.insert("word_overlap_dice".to_string(), overlap.dice);
        breakdown.insert("coherence_overall".to_string(), coherence.overall_coherence);
        breakdown.insert("coherence_local".to_string(), coherence.local_coherence);
        breakdown.insert("coherence_global".to_string(), coherence.global_coherence);
        breakdown.insert("fluency_overall".to_string(), fluency.overall_fluency);
        breakdown.insert(
            "fluency_grammaticality".to_string(),
            fluency.syntactic_fluency.grammaticality_score,
        );
        breakdown.insert(
            "fluency_readability".to_string(),
            fluency.prosodic_fluency.reading_ease,
        );

        breakdown
    }

    fn calculate_confidence_intervals(
        &self,
        bleu: &BleuResult,
        rouge: &RougeResult,
        bert_score: &BertScoreResult,
        semantic: &SemanticResult,
        coherence: &CoherenceResult,
        fluency: &FluencyResult,
    ) -> ConfidenceIntervals {
        // Simplified confidence interval calculation
        // In practice, this would use bootstrap sampling or other statistical methods
        let confidence_level = 0.95;
        let margin_of_error = 0.05;

        ConfidenceIntervals {
            overall_score_ci: (0.0, 1.0), // Would be calculated from actual data
            bleu_ci: (
                (bleu.bleu_score - margin_of_error).max(0.0),
                (bleu.bleu_score + margin_of_error).min(1.0),
            ),
            rouge_ci: (
                (rouge.rouge_score - margin_of_error).max(0.0),
                (rouge.rouge_score + margin_of_error).min(1.0),
            ),
            bert_score_ci: (
                (bert_score.f1_score - margin_of_error).max(0.0),
                (bert_score.f1_score + margin_of_error).min(1.0),
            ),
            semantic_ci: (
                (semantic.overall_similarity - margin_of_error).max(0.0),
                (semantic.overall_similarity + margin_of_error).min(1.0),
            ),
            coherence_ci: (
                (coherence.overall_coherence - margin_of_error).max(0.0),
                (coherence.overall_coherence + margin_of_error).min(1.0),
            ),
            fluency_ci: (
                (fluency.overall_fluency - margin_of_error).max(0.0),
                (fluency.overall_fluency + margin_of_error).min(1.0),
            ),
            confidence_level,
        }
    }

    fn calculate_statistical_summary(
        &self,
        metric_breakdown: &HashMap<String, f64>,
    ) -> StatisticalSummary {
        let metric_names: Vec<String> = metric_breakdown.keys().cloned().collect();
        let mut correlations = HashMap::new();
        let mut reliability = HashMap::new();
        let mut variance_analysis = HashMap::new();

        // Calculate pairwise correlations (simplified)
        for i in 0..metric_names.len() {
            for j in (i + 1)..metric_names.len() {
                let metric1 = &metric_names[i];
                let metric2 = &metric_names[j];
                let score1 = metric_breakdown.get(metric1).unwrap_or(&0.0);
                let score2 = metric_breakdown.get(metric2).unwrap_or(&0.0);

                // Simplified correlation (would use proper statistical correlation in practice)
                let correlation = 1.0 - (score1 - score2).abs();
                correlations.insert((metric1.clone(), metric2.clone()), correlation);
            }
        }

        // Calculate reliability scores (simplified)
        for metric_name in &metric_names {
            let score = metric_breakdown.get(metric_name).unwrap_or(&0.0);
            let reliability_score = if *score > 0.8 {
                0.9
            } else if *score > 0.6 {
                0.8
            } else if *score > 0.4 {
                0.7
            } else {
                0.6
            };
            reliability.insert(metric_name.clone(), reliability_score);
        }

        // Calculate variance analysis
        let mean_score = metric_breakdown.values().sum::<f64>() / metric_breakdown.len() as f64;
        for (metric_name, score) in metric_breakdown {
            let variance = (score - mean_score).powi(2);
            variance_analysis.insert(metric_name.clone(), variance);
        }

        // Detect outliers
        let outlier_detection = metric_breakdown
            .iter()
            .filter(|(_, &score)| (score - mean_score).abs() > 0.3)
            .map(|(name, _)| name.clone())
            .collect();

        // Calculate consistency score
        let total_variance: f64 = variance_analysis.values().sum();
        let consistency_score = 1.0 / (1.0 + total_variance);

        StatisticalSummary {
            metric_correlations: correlations,
            metric_reliability: reliability,
            outlier_detection,
            consistency_score,
            variance_analysis,
        }
    }

    fn assess_quality(
        &self,
        overall_score: f64,
        metric_breakdown: &HashMap<String, f64>,
    ) -> QualityAssessment {
        let quality_level = if overall_score >= 0.9 {
            QualityLevel::Excellent
        } else if overall_score >= 0.75 {
            QualityLevel::Good
        } else if overall_score >= 0.6 {
            QualityLevel::Fair
        } else if overall_score >= 0.4 {
            QualityLevel::Poor
        } else {
            QualityLevel::VeryPoor
        };

        let mut strengths = Vec::new();
        let mut weaknesses = Vec::new();
        let mut recommendations = Vec::new();

        // Analyze strengths and weaknesses
        for (metric, &score) in metric_breakdown {
            if score >= 0.8 {
                strengths.push(format!("Strong {}: {:.3}", metric, score));
            } else if score <= 0.4 {
                weaknesses.push(format!("Weak {}: {:.3}", metric, score));

                // Generate recommendations based on weakness
                match metric.as_str() {
                    s if s.contains("bleu") => {
                        recommendations.push("Improve lexical similarity to reference".to_string())
                    }
                    s if s.contains("rouge") => {
                        recommendations.push("Enhance content overlap with reference".to_string())
                    }
                    s if s.contains("coherence") => {
                        recommendations.push("Strengthen text organization and flow".to_string())
                    }
                    s if s.contains("fluency") => {
                        recommendations.push("Improve grammaticality and naturalness".to_string())
                    }
                    s if s.contains("semantic") => {
                        recommendations.push("Enhance semantic relatedness".to_string())
                    }
                    _ => recommendations
                        .push("Consider overall text quality improvements".to_string()),
                }
            }
        }

        if strengths.is_empty() {
            strengths.push("Consider focusing on fundamental text quality".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Maintain current quality standards".to_string());
        }

        let overall_grade = overall_score * 100.0;

        QualityAssessment {
            quality_level,
            strengths,
            weaknesses,
            recommendations,
            overall_grade,
        }
    }

    fn calculate_score_differences(
        &self,
        evaluations: &[ComprehensiveEvaluationResult],
    ) -> Vec<f64> {
        let mut differences = Vec::new();

        for i in 0..evaluations.len() {
            for j in (i + 1)..evaluations.len() {
                let diff = (evaluations[i].overall_score - evaluations[j].overall_score).abs();
                differences.push(diff);
            }
        }

        differences
    }

    fn test_statistical_significance(&self, evaluations: &[ComprehensiveEvaluationResult]) -> bool {
        if evaluations.len() < 2 {
            return false;
        }

        let scores: Vec<f64> = evaluations.iter().map(|e| e.overall_score).collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores
            .iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f64>()
            / scores.len() as f64;
        let std_dev = variance.sqrt();

        // Simple significance test - would use proper statistical tests in practice
        std_dev > 0.05
    }

    fn analyze_quality_dimensions(
        &self,
        coherence: &CoherenceResult,
        fluency: &FluencyResult,
    ) -> QualityDimensions {
        QualityDimensions {
            entity_coherence: coherence.entity_coherence.entity_grid_coherence,
            lexical_coherence: coherence.lexical_coherence.lexical_chain_coherence,
            discourse_coherence: coherence.discourse_coherence.discourse_marker_coherence,
            syntactic_fluency: fluency.syntactic_fluency.grammaticality_score,
            lexical_fluency: fluency.lexical_fluency.lexical_diversity,
            semantic_fluency: fluency.semantic_fluency.semantic_coherence,
            prosodic_fluency: fluency.prosodic_fluency.reading_ease,
            pragmatic_fluency: fluency.pragmatic_fluency.communicative_effectiveness,
        }
    }

    /// Calculate system-level evaluation metrics for multiple test cases
    pub fn system_evaluation(&self, test_cases: &[(String, String)]) -> SystemEvaluationResult {
        let evaluations: Vec<ComprehensiveEvaluationResult> = test_cases
            .iter()
            .map(|(reference, candidate)| self.comprehensive_evaluation(reference, candidate))
            .collect();

        let overall_scores: Vec<f64> = evaluations.iter().map(|e| e.overall_score).collect();

        let mean_score = overall_scores.iter().sum::<f64>() / overall_scores.len() as f64;
        let variance = overall_scores
            .iter()
            .map(|score| (score - mean_score).powi(2))
            .sum::<f64>()
            / overall_scores.len() as f64;
        let std_dev = variance.sqrt();

        let min_score = overall_scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_score = overall_scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Calculate percentiles
        let mut sorted_scores = overall_scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let percentile_25 = sorted_scores[sorted_scores.len() / 4];
        let median = sorted_scores[sorted_scores.len() / 2];
        let percentile_75 = sorted_scores[3 * sorted_scores.len() / 4];

        let metric_averages = self.calculate_metric_averages(&evaluations);
        let consistency_analysis = self.analyze_consistency(&evaluations);

        SystemEvaluationResult {
            mean_score,
            std_dev,
            min_score,
            max_score,
            median,
            percentile_25,
            percentile_75,
            metric_averages,
            consistency_analysis,
            evaluations,
        }
    }

    fn calculate_metric_averages(
        &self,
        evaluations: &[ComprehensiveEvaluationResult],
    ) -> HashMap<String, f64> {
        let mut averages = HashMap::new();
        let count = evaluations.len() as f64;

        if count == 0.0 {
            return averages;
        }

        let bleu_sum: f64 = evaluations.iter().map(|e| e.bleu_result.bleu_score).sum();
        let rouge_sum: f64 = evaluations.iter().map(|e| e.rouge_result.rouge_score).sum();
        let bert_f1_sum: f64 = evaluations
            .iter()
            .map(|e| e.bert_score_result.f1_score)
            .sum();
        let semantic_sum: f64 = evaluations
            .iter()
            .map(|e| e.semantic_result.overall_similarity)
            .sum();
        let coherence_sum: f64 = evaluations
            .iter()
            .map(|e| e.coherence_result.overall_coherence)
            .sum();
        let fluency_sum: f64 = evaluations
            .iter()
            .map(|e| e.fluency_result.overall_fluency)
            .sum();

        averages.insert("bleu_average".to_string(), bleu_sum / count);
        averages.insert("rouge_average".to_string(), rouge_sum / count);
        averages.insert("bert_f1_average".to_string(), bert_f1_sum / count);
        averages.insert("semantic_average".to_string(), semantic_sum / count);
        averages.insert("coherence_average".to_string(), coherence_sum / count);
        averages.insert("fluency_average".to_string(), fluency_sum / count);

        averages
    }

    fn analyze_consistency(
        &self,
        evaluations: &[ComprehensiveEvaluationResult],
    ) -> ConsistencyAnalysis {
        let overall_scores: Vec<f64> = evaluations.iter().map(|e| e.overall_score).collect();
        let mean = overall_scores.iter().sum::<f64>() / overall_scores.len() as f64;
        let variance = overall_scores
            .iter()
            .map(|score| (score - mean).powi(2))
            .sum::<f64>()
            / overall_scores.len() as f64;

        let coefficient_of_variation = if mean > 0.0 {
            variance.sqrt() / mean
        } else {
            0.0
        };
        let consistency_score = 1.0 - coefficient_of_variation.min(1.0);

        let outliers = evaluations
            .iter()
            .enumerate()
            .filter(|(_, e)| (e.overall_score - mean).abs() > 2.0 * variance.sqrt())
            .map(|(idx, _)| idx)
            .collect();

        ConsistencyAnalysis {
            consistency_score,
            coefficient_of_variation,
            outlier_indices: outliers,
            variance,
        }
    }
}

/// Result of comparative evaluation
#[derive(Debug)]
pub struct ComparativeEvaluationResult {
    pub evaluations: Vec<ComprehensiveEvaluationResult>,
    pub ranked_indices: Vec<usize>,
    pub best_candidate: usize,
    pub worst_candidate: usize,
    pub score_differences: Vec<f64>,
    pub statistical_significance: bool,
}

/// Result of quality analysis
#[derive(Debug)]
pub struct QualityAnalysisResult {
    pub overall_quality: f64,
    pub readability_score: f64,
    pub complexity_score: f64,
    pub naturalness_score: f64,
    pub coherence_score: f64,
    pub quality_dimensions: QualityDimensions,
    pub coherence_result: CoherenceResult,
    pub fluency_result: FluencyResult,
}

/// Quality dimensions breakdown
#[derive(Debug)]
pub struct QualityDimensions {
    pub entity_coherence: f64,
    pub lexical_coherence: f64,
    pub discourse_coherence: f64,
    pub syntactic_fluency: f64,
    pub lexical_fluency: f64,
    pub semantic_fluency: f64,
    pub prosodic_fluency: f64,
    pub pragmatic_fluency: f64,
}

/// System-level evaluation result
#[derive(Debug)]
pub struct SystemEvaluationResult {
    pub mean_score: f64,
    pub std_dev: f64,
    pub min_score: f64,
    pub max_score: f64,
    pub median: f64,
    pub percentile_25: f64,
    pub percentile_75: f64,
    pub metric_averages: HashMap<String, f64>,
    pub consistency_analysis: ConsistencyAnalysis,
    pub evaluations: Vec<ComprehensiveEvaluationResult>,
}

/// Consistency analysis result
#[derive(Debug)]
pub struct ConsistencyAnalysis {
    pub consistency_score: f64,
    pub coefficient_of_variation: f64,
    pub outlier_indices: Vec<usize>,
    pub variance: f64,
}

// Convenience functions for quick evaluation
/// Quick BLEU score calculation
pub fn quick_bleu(reference: &str, candidate: &str) -> f64 {
    match BleuScore::default().calculate(candidate, &[reference]) {
        Ok(result) => result,
        Err(_) => 0.0,
    }
}

/// Quick ROUGE score calculation
pub fn quick_rouge(reference: &str, candidate: &str) -> f64 {
    match RougeScore::default().calculate(candidate, reference) {
        Ok(result) => result.f1_score,
        Err(_) => 0.0,
    }
}

/// Quick semantic similarity calculation
pub fn quick_semantic_similarity(text1: &str, text2: &str) -> f64 {
    // Simple implementation using word overlap as a proxy
    let words1: std::collections::HashSet<&str> = text1.split_whitespace().collect();
    let words2: std::collections::HashSet<&str> = text2.split_whitespace().collect();
    let intersection = words1.intersection(&words2).count();
    let union = words1.union(&words2).count();
    if union > 0 {
        intersection as f64 / union as f64
    } else {
        0.0
    }
}

/// Quick coherence score calculation
pub fn quick_coherence(text: &str) -> f64 {
    // Simple coherence heuristic based on sentence count and average sentence length
    let sentences: Vec<&str> = text.split('.').filter(|s| !s.trim().is_empty()).collect();
    if sentences.is_empty() {
        return 0.0;
    }

    let avg_sentence_length = sentences
        .iter()
        .map(|s| s.split_whitespace().count())
        .sum::<usize>() as f64
        / sentences.len() as f64;
    // Score based on reasonable sentence length (10-30 words is considered good)
    if avg_sentence_length >= 10.0 && avg_sentence_length <= 30.0 {
        0.8 + (20.0 - (avg_sentence_length - 20.0).abs()) / 100.0
    } else {
        0.5
    }
}

/// Quick fluency score calculation
pub fn quick_fluency(text: &str) -> f64 {
    // Simple fluency score based on basic text properties
    let word_count = text.split_whitespace().count();
    let char_count = text.chars().count();

    if word_count == 0 {
        return 0.0;
    }

    let avg_word_length = char_count as f64 / word_count as f64;
    // Score based on reasonable average word length (4-6 characters is typical for English)
    if avg_word_length >= 4.0 && avg_word_length <= 6.0 {
        0.8
    } else {
        0.6
    }
}

/// Quick overall quality score (combines multiple metrics)
pub fn quick_quality_score(reference: &str, candidate: &str) -> f64 {
    let evaluator = TextEvaluator::with_default_config();
    let result = evaluator.comprehensive_evaluation(reference, candidate);
    result.overall_score
}

/// Batch evaluation for multiple text pairs
pub fn batch_evaluate(pairs: &[(String, String)]) -> Vec<f64> {
    let evaluator = TextEvaluator::with_default_config();
    pairs
        .iter()
        .map(|(reference, candidate)| {
            let result = evaluator.comprehensive_evaluation(reference, candidate);
            result.overall_score
        })
        .collect()
}

/// Create evaluation report in human-readable format
pub fn create_evaluation_report(result: &ComprehensiveEvaluationResult) -> String {
    let mut report = String::new();

    report.push_str("=== TEXT EVALUATION REPORT ===\n\n");

    report.push_str(&format!("Overall Score: {:.3}\n", result.overall_score));
    report.push_str(&format!(
        "Quality Level: {:?}\n",
        result.quality_assessment.quality_level
    ));
    report.push_str(&format!(
        "Overall Grade: {:.1}%\n\n",
        result.quality_assessment.overall_grade
    ));

    report.push_str("=== METRIC BREAKDOWN ===\n");
    // Use metric_breakdown for scores that aren't in direct fields
    let bleu_score = result.metric_breakdown.get("bleu").unwrap_or(&0.0);
    let rouge_score = result.metric_breakdown.get("rouge").unwrap_or(&0.0);
    let overlap_score = result.metric_breakdown.get("overlap").unwrap_or(&0.0);
    let coherence_score = result.metric_breakdown.get("coherence").unwrap_or(&0.0);
    let fluency_score = result.metric_breakdown.get("fluency").unwrap_or(&0.0);

    report.push_str(&format!("BLEU Score: {:.3}\n", bleu_score));
    report.push_str(&format!("ROUGE Score: {:.3}\n", rouge_score));
    report.push_str(&format!(
        "BERTScore F1: {:.3}\n",
        result.bert_score_result.f1_score
    ));
    report.push_str(&format!(
        "Semantic Similarity: {:.3}\n",
        result.semantic_result.overall_similarity
    ));
    report.push_str(&format!("Word Overlap (Jaccard): {:.3}\n", overlap_score));
    report.push_str(&format!("Coherence: {:.3}\n", coherence_score));
    report.push_str(&format!("Fluency: {:.3}\n", fluency_score));

    report.push_str("\n=== QUALITY ASSESSMENT ===\n");
    report.push_str("Strengths:\n");
    for strength in &result.quality_assessment.strengths {
        report.push_str(&format!("  • {}\n", strength));
    }

    if !result.quality_assessment.weaknesses.is_empty() {
        report.push_str("\nWeaknesses:\n");
        for weakness in &result.quality_assessment.weaknesses {
            report.push_str(&format!("  • {}\n", weakness));
        }
    }

    report.push_str("\nRecommendations:\n");
    for recommendation in &result.quality_assessment.recommendations {
        report.push_str(&format!("  • {}\n", recommendation));
    }

    if let Some(confidence_intervals) = &result.confidence_intervals {
        report.push_str(&format!(
            "\n=== CONFIDENCE INTERVALS ({:.0}%) ===\n",
            confidence_intervals.confidence_level * 100.0
        ));
        report.push_str(&format!(
            "BLEU: [{:.3}, {:.3}]\n",
            confidence_intervals.bleu_ci.0, confidence_intervals.bleu_ci.1
        ));
        report.push_str(&format!(
            "ROUGE: [{:.3}, {:.3}]\n",
            confidence_intervals.rouge_ci.0, confidence_intervals.rouge_ci.1
        ));
        report.push_str(&format!(
            "BERTScore: [{:.3}, {:.3}]\n",
            confidence_intervals.bert_score_ci.0, confidence_intervals.bert_score_ci.1
        ));
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comprehensive_evaluation() {
        let evaluator = TextEvaluator::with_default_config();
        let reference = "The cat sat on the mat and looked very comfortable.";
        let candidate = "A cat was sitting on the mat and appeared quite comfortable.";

        let result = evaluator.comprehensive_evaluation(reference, candidate);

        assert!(result.overall_score > 0.0);
        assert!(result.overall_score <= 1.0);
        assert!(!result.metric_breakdown.is_empty());
        assert!(matches!(
            result.quality_assessment.quality_level,
            QualityLevel::Good | QualityLevel::Fair | QualityLevel::Excellent
        ));
    }

    #[test]
    fn test_batch_evaluation() {
        let evaluator = TextEvaluator::with_default_config();
        let references = vec![
            "The cat sat on the mat.".to_string(),
            "The dog ran in the park.".to_string(),
        ];
        let candidates = vec![
            "A cat was sitting on the mat.".to_string(),
            "The dog was running in the park.".to_string(),
        ];

        let results = evaluator.batch_evaluation(&references, &candidates);

        assert_eq!(results.len(), 2);
        assert!(results[0].overall_score > 0.0);
        assert!(results[1].overall_score > 0.0);
    }

    #[test]
    fn test_comparative_evaluation() {
        let evaluator = TextEvaluator::with_default_config();
        let reference = "The cat sat on the mat.";
        let candidates = vec![
            "A cat was sitting on the mat.".to_string(),
            "The dog ran in the park.".to_string(),
            "A feline was resting on the rug.".to_string(),
        ];

        let result = evaluator.comparative_evaluation(reference, &candidates);

        assert_eq!(result.evaluations.len(), 3);
        assert_eq!(result.ranked_indices.len(), 3);
        assert!(
            result.evaluations[result.best_candidate].overall_score
                >= result.evaluations[result.worst_candidate].overall_score
        );
    }

    #[test]
    fn test_quality_analysis() {
        let evaluator = TextEvaluator::with_default_config();
        let text = "This is a well-written text with good structure. It demonstrates clear organization and maintains coherence throughout. The sentences flow naturally and the vocabulary is appropriate.";

        let result = evaluator.quality_analysis(text);

        assert!(result.overall_quality > 0.0);
        assert!(result.overall_quality <= 1.0);
        assert!(result.readability_score >= 0.0);
        assert!(result.coherence_score >= 0.0);
        assert!(result.naturalness_score >= 0.0);
    }

    #[test]
    fn test_system_evaluation() {
        let evaluator = TextEvaluator::with_default_config();
        let test_cases = vec![
            (
                "The cat sat on the mat.".to_string(),
                "A cat was sitting on the mat.".to_string(),
            ),
            (
                "The dog ran quickly.".to_string(),
                "The dog was running fast.".to_string(),
            ),
            (
                "It was raining heavily.".to_string(),
                "There was heavy rain.".to_string(),
            ),
        ];

        let result = evaluator.system_evaluation(&test_cases);

        assert!(result.mean_score > 0.0);
        assert!(result.std_dev >= 0.0);
        assert!(result.min_score <= result.max_score);
        assert!(result.percentile_25 <= result.median);
        assert!(result.median <= result.percentile_75);
        assert_eq!(result.evaluations.len(), 3);
    }

    #[test]
    fn test_convenience_functions() {
        let reference = "The cat sat on the mat.";
        let candidate = "A cat was sitting on the mat.";

        let bleu_score = quick_bleu(reference, candidate);
        let rouge_score = quick_rouge(reference, candidate);
        let semantic_score = quick_semantic_similarity(reference, candidate);
        let coherence_score = quick_coherence(candidate);
        let fluency_score = quick_fluency(candidate);
        let quality_score = quick_quality_score(reference, candidate);

        assert!(bleu_score >= 0.0 && bleu_score <= 1.0);
        assert!(rouge_score >= 0.0 && rouge_score <= 1.0);
        assert!(semantic_score >= 0.0 && semantic_score <= 1.0);
        assert!(coherence_score >= 0.0 && coherence_score <= 1.0);
        assert!(fluency_score >= 0.0 && fluency_score <= 1.0);
        assert!(quality_score >= 0.0 && quality_score <= 1.0);
    }

    #[test]
    fn test_batch_evaluate() {
        let pairs = vec![
            ("Hello world".to_string(), "Hi world".to_string()),
            ("Good morning".to_string(), "Good evening".to_string()),
        ];

        let scores = batch_evaluate(&pairs);

        assert_eq!(scores.len(), 2);
        assert!(scores[0] >= 0.0 && scores[0] <= 1.0);
        assert!(scores[1] >= 0.0 && scores[1] <= 1.0);
    }

    #[test]
    fn test_evaluation_report() {
        let evaluator = TextEvaluator::with_default_config();
        let reference = "The cat sat on the mat.";
        let candidate = "A cat was sitting on the mat.";

        let result = evaluator.comprehensive_evaluation(reference, candidate);
        let report = create_evaluation_report(&result);

        assert!(!report.is_empty());
        assert!(report.contains("TEXT EVALUATION REPORT"));
        assert!(report.contains("Overall Score"));
        assert!(report.contains("METRIC BREAKDOWN"));
        assert!(report.contains("QUALITY ASSESSMENT"));
    }

    #[test]
    fn test_configuration_customization() {
        let mut config = EvaluationConfig::default();
        config.weights.bleu_weight = 0.3;
        config.weights.semantic_weight = 0.3;
        config.enable_confidence_intervals = false;

        let evaluator = TextEvaluator::new(config);
        let reference = "The cat sat on the mat.";
        let candidate = "A cat was sitting on the mat.";

        let result = evaluator.comprehensive_evaluation(reference, candidate);

        assert!(result.overall_score > 0.0);
        assert!(result.confidence_intervals.is_none());
    }

    #[test]
    fn test_edge_cases() {
        let evaluator = TextEvaluator::with_default_config();

        // Empty strings
        let result1 = evaluator.comprehensive_evaluation("", "");
        assert!(result1.overall_score >= 0.0);

        // Very different texts
        let result2 = evaluator.comprehensive_evaluation(
            "The cat sat on the mat.",
            "Machine learning algorithms are complex.",
        );
        assert!(result2.overall_score >= 0.0);
        assert!(result2.overall_score < 0.5); // Should be low similarity

        // Identical texts
        let result3 = evaluator.comprehensive_evaluation("Hello world", "Hello world");
        assert!(result3.overall_score > 0.8); // Should be high similarity
    }

    #[test]
    fn test_quality_levels() {
        let evaluator = TextEvaluator::with_default_config();

        // Test with different quality texts
        let high_quality_ref = "The sophisticated algorithm demonstrated remarkable performance across multiple evaluation metrics.";
        let high_quality_cand =
            "The advanced algorithm showed excellent performance on various evaluation measures.";

        let low_quality_ref = "The cat.";
        let low_quality_cand = "Dog run fast very.";

        let high_result = evaluator.comprehensive_evaluation(high_quality_ref, high_quality_cand);
        let low_result = evaluator.comprehensive_evaluation(low_quality_ref, low_quality_cand);

        assert!(high_result.overall_score > low_result.overall_score);

        match high_result.quality_assessment.quality_level {
            QualityLevel::Good | QualityLevel::Fair | QualityLevel::Excellent => assert!(true),
            _ => assert!(false, "Expected higher quality level"),
        }

        match low_result.quality_assessment.quality_level {
            QualityLevel::Poor | QualityLevel::VeryPoor => assert!(true),
            _ => assert!(false, "Expected lower quality level"),
        }
    }

    #[test]
    fn test_metric_weights() {
        let mut config1 = EvaluationConfig::default();
        config1.weights.bleu_weight = 1.0;
        config1.weights.rouge_weight = 0.0;
        config1.weights.semantic_weight = 0.0;
        config1.weights.coherence_weight = 0.0;
        config1.weights.fluency_weight = 0.0;
        config1.weights.bert_score_weight = 0.0;
        config1.weights.overlap_weight = 0.0;
        config1.weights.edit_distance_weight = 0.0;

        let mut config2 = EvaluationConfig::default();
        config2.weights.bleu_weight = 0.0;
        config2.weights.semantic_weight = 1.0;
        config2.weights.rouge_weight = 0.0;
        config2.weights.coherence_weight = 0.0;
        config2.weights.fluency_weight = 0.0;
        config2.weights.bert_score_weight = 0.0;
        config2.weights.overlap_weight = 0.0;
        config2.weights.edit_distance_weight = 0.0;

        let evaluator1 = TextEvaluator::new(config1);
        let evaluator2 = TextEvaluator::new(config2);

        let reference = "The cat sat on the mat.";
        let candidate = "A cat was sitting on the mat.";

        let result1 = evaluator1.comprehensive_evaluation(reference, candidate);
        let result2 = evaluator2.comprehensive_evaluation(reference, candidate);

        // Results should be different due to different weightings
        assert!((result1.overall_score - result2.overall_score).abs() > 0.01);
    }
}
