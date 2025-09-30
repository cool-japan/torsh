//! Legacy metrics API - Backward compatibility layer
//!
//! This module provides a backward-compatible interface to the original metrics API
//! while internally using the modern modular metrics system for all calculations.
//!
//! ## Migration Notice
//!
//! This API is maintained for backward compatibility. New code should use the
//! modern modular metrics system available in the individual metric modules:
//! - `crate::metrics::bleu` for BLEU scores
//! - `crate::metrics::rouge` for ROUGE scores
//! - `crate::metrics::bert_score` for BERTScore
//! - `crate::metrics::semantic` for semantic similarity
//! - `crate::metrics::TextEvaluator` for comprehensive evaluation
//!
//! ## Example Migration
//!
//! ```rust
//! // Legacy API (still supported)
//! let bleu = BleuScore::default();
//! let score = bleu.calculate(candidate, references)?;
//!
//! // Modern API (recommended)
//! use crate::metrics::{BleuScore, BleuConfig};
//! let bleu = BleuScore::new(BleuConfig::default());
//! let result = bleu.calculate_bleu(reference, candidate);
//! let score = result.bleu_score;
//! ```

use crate::{Result, TextError};
use crate::metrics::{
    BleuScore as ModernBleuScore, BleuConfig,
    RougeCalculator as ModernRougeCalculator, RougeConfig, RougeType as ModernRougeType,
    PerplexityCalculator as ModernPerplexityCalculator, PerplexityConfig,
    EditDistanceCalculator as ModernEditDistanceCalculator, edit_distance::EditDistanceConfig,
    edit_distance::DistanceAlgorithm,
    BertScore as ModernBertScore, BertScoreConfig,
    SemanticSimilarity as ModernSemanticSimilarity, SemanticConfig,
    WordOverlapCalculator, OverlapConfig,
    CoherenceAnalyzer, CoherenceConfig,
    FluencyAnalyzer, FluencyConfig,
    TextEvaluator, EvaluationConfig, EvaluationWeights,
    ComprehensiveEvaluationResult,
};
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::any::Any;

/// Type alias for n-gram counts (legacy compatibility)
type NgramCounts<'a> = HashMap<Vec<&'a str>, usize>;

/// Legacy BLEU score calculator (backward compatibility wrapper)
///
/// This struct provides the original BLEU API while internally using the modern
/// modular BLEU implementation for all calculations.
///
/// # Migration
///
/// Consider migrating to `crate::metrics::BleuScore` for enhanced features:
/// - Confidence intervals
/// - Statistical analysis
/// - Multiple smoothing algorithms
/// - Comprehensive result objects
#[derive(Debug, Clone)]
pub struct BleuScore {
    smoothing: bool,
    max_n: usize,
    /// Internal modern BLEU calculator (created on-demand)
    _modern_bleu: ModernBleuScore,
}

impl Default for BleuScore {
    fn default() -> Self {
        let config = BleuConfig {
            max_n: 4,
            smoothing: true,
            weights: vec![0.25, 0.25, 0.25, 0.25],
            ..Default::default()
        };
        Self {
            smoothing: true,
            max_n: 4,
            _modern_bleu: ModernBleuScore::new(config),
        }
    }
}

impl BleuScore {
    pub fn with_smoothing(mut self, smoothing: bool) -> Self {
        self.smoothing = smoothing;
        let mut config = BleuConfig::default();
        config.smoothing = smoothing;
        config.max_n = self.max_n;
        config.weights = vec![0.25; self.max_n];
        self._modern_bleu = ModernBleuScore::new(config);
        self
    }

    pub fn with_max_n(mut self, max_n: usize) -> Self {
        self.max_n = max_n;
        let mut config = BleuConfig::default();
        config.max_n = max_n;
        config.smoothing = self.smoothing;
        config.weights = vec![1.0 / max_n as f64; max_n];
        self._modern_bleu = ModernBleuScore::new(config);
        self
    }

    pub fn calculate(&self, candidate: &str, references: &[&str]) -> Result<f64> {
        // Use the modern BLEU calculator with the first reference
        // (original API doesn't handle multiple references the same way)
        if references.is_empty() {
            return Err(TextError::Other(anyhow::anyhow!(
                "No reference sentences provided"
            )));
        }

        let reference = references[0]; // Use first reference for compatibility
        let result = self._modern_bleu.calculate_bleu(reference, candidate);
        Ok(result.bleu_score)
    }

    pub fn calculate_corpus(&self, candidates: &[&str], references: &[Vec<&str>]) -> Result<f64> {
        if candidates.len() != references.len() {
            return Err(TextError::Other(anyhow::anyhow!(
                "Number of candidates and references must match"
            )));
        }

        // Use modern system for corpus-level calculation
        let mut total_score = 0.0;
        for (candidate, refs) in candidates.iter().zip(references.iter()) {
            if !refs.is_empty() {
                let score = self.calculate(candidate, refs)?;
                total_score += score;
            }
        }

        Ok(total_score / candidates.len() as f64)
    }
}

/// Legacy ROUGE score calculator (backward compatibility wrapper)
///
/// This struct provides the original ROUGE API while internally using the modern
/// modular ROUGE implementation for all calculations.
///
/// # Migration
///
/// Consider migrating to `crate::metrics::RougeCalculator` for enhanced features:
/// - Multiple ROUGE variants
/// - Statistical significance testing
/// - Confidence intervals
/// - Comprehensive result objects
#[derive(Debug, Clone)]
pub struct RougeScore {
    rouge_type: RougeType,
    use_stemming: bool,
    /// Internal modern ROUGE calculator (created on-demand)
    _modern_rouge: ModernRougeCalculator,
}

/// Legacy ROUGE type enumeration (backward compatibility)
#[derive(Debug, Clone, PartialEq)]
pub enum RougeType {
    Rouge1, // Unigram overlap
    Rouge2, // Bigram overlap
    RougeL, // Longest Common Subsequence
}

/// Convert legacy RougeType to modern RougeType
impl From<RougeType> for ModernRougeType {
    fn from(rouge_type: RougeType) -> Self {
        match rouge_type {
            RougeType::Rouge1 => ModernRougeType::Rouge1,
            RougeType::Rouge2 => ModernRougeType::Rouge2,
            RougeType::RougeL => ModernRougeType::RougeL,
        }
    }
}

impl Default for RougeScore {
    fn default() -> Self {
        let config = RougeConfig {
            use_stemming: false,
            ..Default::default()
        };
        Self {
            rouge_type: RougeType::Rouge1,
            use_stemming: false,
            _modern_rouge: ModernRougeCalculator::new(config),
        }
    }
}

impl RougeScore {
    pub fn new(rouge_type: RougeType) -> Self {
        let config = RougeConfig {
            use_stemming: false,
            ..Default::default()
        };
        Self {
            rouge_type,
            use_stemming: false,
            _modern_rouge: ModernRougeCalculator::new(config),
        }
    }

    pub fn with_stemming(mut self, use_stemming: bool) -> Self {
        self.use_stemming = use_stemming;
        let config = RougeConfig {
            use_stemming,
            ..Default::default()
        };
        self._modern_rouge = ModernRougeCalculator::new(config);
        self
    }

    pub fn calculate(&self, candidate: &str, reference: &str) -> Result<RougeMetrics> {
        let modern_type: ModernRougeType = self.rouge_type.clone().into();
        let result = self._modern_rouge.calculate_rouge(reference, candidate, modern_type);

        Ok(RougeMetrics {
            precision: result.precision,
            recall: result.recall,
            f1_score: result.f1_score,
        })
    }


/// Legacy ROUGE metrics result (backward compatibility)
///
/// This struct provides the original ROUGE result format while internally
/// using data from the modern RougeResult.
///
/// # Migration
///
/// Consider migrating to `crate::metrics::rouge::RougeResult` for enhanced features:
/// - Confidence intervals
/// - Statistical significance
/// - Multiple ROUGE variant scores
#[derive(Debug, Clone)]
pub struct RougeMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
}

/// Legacy perplexity calculator (backward compatibility wrapper)
///
/// This struct provides the original perplexity API while internally using the modern
/// modular perplexity implementation for all calculations.
///
/// # Migration
///
/// Consider migrating to `crate::metrics::PerplexityCalculator` for enhanced features:
/// - Confidence intervals
/// - Statistical analysis
/// - Multiple calculation methods
/// - Comprehensive result objects
#[derive(Debug, Clone)]
pub struct PerplexityCalculator {
    /// Internal modern perplexity calculator
    _modern_calculator: ModernPerplexityCalculator,
}

impl Default for PerplexityCalculator {
    fn default() -> Self {
        Self {
            _modern_calculator: ModernPerplexityCalculator::new(PerplexityConfig::default()),
        }
    }
}

impl PerplexityCalculator {
    /// Calculate perplexity from probability values (static method for backward compatibility)
    pub fn calculate(probabilities: &[f64]) -> Result<f64> {
        let calculator = Self::default();
        calculator.calculate_instance(probabilities)
    }

    /// Calculate perplexity from logit values (static method for backward compatibility)
    pub fn calculate_from_logits(logits: &[f64]) -> Result<f64> {
        let calculator = Self::default();
        calculator.calculate_from_logits_instance(logits)
    }

    /// Instance method for calculating perplexity from probabilities
    pub fn calculate_instance(&self, probabilities: &[f64]) -> Result<f64> {
        if probabilities.is_empty() {
            return Err(TextError::Other(anyhow::anyhow!(
                "No probabilities provided"
            )));
        }

        for &prob in probabilities {
            if prob <= 0.0 {
                return Err(TextError::Other(anyhow::anyhow!(
                    "Invalid probability: {}",
                    prob
                )));
            }
        }

        // Use modern calculator - convert probabilities to the expected format
        let result = self._modern_calculator.calculate_from_probabilities(probabilities);
        Ok(result.perplexity)
    }

    /// Instance method for calculating perplexity from logits
    pub fn calculate_from_logits_instance(&self, logits: &[f64]) -> Result<f64> {
        if logits.is_empty() {
            return Err(TextError::Other(anyhow::anyhow!("No logits provided")));
        }

        // Use modern calculator
        let result = self._modern_calculator.calculate_from_logits(logits);
        Ok(result.perplexity)
    }
}

/// Legacy edit distance calculator (backward compatibility wrapper)
///
/// This struct provides the original edit distance API while internally using the modern
/// modular edit distance implementation for all calculations.
///
/// # Migration
///
/// Consider migrating to `crate::metrics::EditDistanceCalculator` for enhanced features:
/// - Multiple distance algorithms
/// - Confidence intervals
/// - Statistical analysis
/// - Comprehensive result objects
#[derive(Debug, Clone)]
pub struct EditDistance {
    /// Internal modern edit distance calculator
    _modern_calculator: ModernEditDistanceCalculator,
}

impl Default for EditDistance {
    fn default() -> Self {
        let config = EditDistanceConfig {
            algorithm: DistanceAlgorithm::Levenshtein,
            ..Default::default()
        };
        Self {
            _modern_calculator: ModernEditDistanceCalculator::new(config),
        }
    }
}

impl EditDistance {
    /// Calculate Levenshtein distance between two strings (static method for backward compatibility)
    pub fn levenshtein(s1: &str, s2: &str) -> usize {
        let calculator = Self::default();
        let result = calculator._modern_calculator.calculate_distance(s1, s2);
        result.raw_distance as usize
    }

    /// Calculate normalized Levenshtein similarity (static method for backward compatibility)
    pub fn normalized_levenshtein(s1: &str, s2: &str) -> f64 {
        let calculator = Self::default();
        let result = calculator._modern_calculator.calculate_distance(s1, s2);
        result.normalized_similarity
    }
}

/// Comprehensive test suite for backward compatibility
///
/// These tests verify that the legacy API works correctly with the modern
/// modular implementation underneath.
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_legacy_bleu_score() {
        let bleu = BleuScore::default();

        let candidate = "the cat is on the mat";
        let references = vec!["the cat is sitting on the mat", "a cat is on the mat"];

        let score = bleu.calculate(candidate, &references).unwrap();
        assert!(score >= 0.0 && score <= 1.0, "BLEU score should be between 0 and 1, got: {}", score);
        assert!(score > 0.1, "BLEU score should be reasonable for similar sentences, got: {}", score);
    }

    #[test]
    fn test_legacy_bleu_score_corpus() {
        let bleu = BleuScore::default();

        let candidates = vec!["the cat is on the mat", "the dog runs fast"];
        let references = vec![
            vec!["the cat is sitting on the mat", "a cat is on the mat"],
            vec!["the dog is running quickly", "a dog runs fast"]
        ];

        let score = bleu.calculate_corpus(&candidates, &references).unwrap();
        assert!(score >= 0.0 && score <= 1.0, "Corpus BLEU score should be between 0 and 1, got: {}", score);
    }

    #[test]
    fn test_legacy_rouge_score() {
        let rouge = RougeScore::new(RougeType::Rouge1);

        let candidate = "the cat sat on the mat";
        let reference = "the cat is sitting on the mat";

        let metrics = rouge.calculate(candidate, reference).unwrap();
        assert!(metrics.precision > 0.0, "ROUGE precision should be positive, got: {}", metrics.precision);
        assert!(metrics.recall > 0.0, "ROUGE recall should be positive, got: {}", metrics.recall);
        assert!(metrics.f1_score > 0.0, "ROUGE F1 should be positive, got: {}", metrics.f1_score);
        assert!(metrics.precision <= 1.0 && metrics.recall <= 1.0 && metrics.f1_score <= 1.0);
    }

    #[test]
    fn test_legacy_rouge_l() {
        let rouge = RougeScore::new(RougeType::RougeL);

        let candidate = "the cat sat on mat";
        let reference = "the cat is on the mat";

        let metrics = rouge.calculate(candidate, reference).unwrap();
        assert!(metrics.f1_score >= 0.0 && metrics.f1_score <= 1.0, "ROUGE-L F1 should be between 0 and 1, got: {}", metrics.f1_score);
    }

    #[test]
    fn test_legacy_perplexity_static() {
        let probabilities = vec![0.5, 0.3, 0.2];
        let perplexity = PerplexityCalculator::calculate(&probabilities).unwrap();
        assert!(perplexity > 0.0, "Perplexity should be positive, got: {}", perplexity);
    }

    #[test]
    fn test_legacy_perplexity_from_logits() {
        let logits = vec![1.0, 0.5, 0.2];
        let perplexity = PerplexityCalculator::calculate_from_logits(&logits).unwrap();
        assert!(perplexity > 0.0, "Perplexity from logits should be positive, got: {}", perplexity);
    }

    #[test]
    fn test_legacy_edit_distance() {
        let distance = EditDistance::levenshtein("kitten", "sitting");
        assert_eq!(distance, 3, "Levenshtein distance between 'kitten' and 'sitting' should be 3, got: {}", distance);

        let normalized = EditDistance::normalized_levenshtein("kitten", "sitting");
        assert!(normalized >= 0.0 && normalized <= 1.0, "Normalized Levenshtein should be between 0 and 1, got: {}", normalized);
        assert!(normalized < 1.0, "Different strings should have similarity < 1.0, got: {}", normalized);
    }

    #[test]
    fn test_legacy_bert_score() {
        let references = vec!["The cat is on the mat."];
        let candidates = vec!["A cat is sitting on the mat."];

        let bert_score = BertScore::new();
        let result = bert_score.compute(&references, &candidates);
        assert!(result.is_ok(), "BERTScore computation should succeed: {:?}", result.err());

        let scores = result.unwrap();
        assert_eq!(scores.len(), 1, "Should have one score for one candidate-reference pair");

        let score = &scores[0];
        assert!(score.precision >= 0.0 && score.precision <= 1.0, "BERTScore precision should be between 0 and 1, got: {}", score.precision);
        assert!(score.recall >= 0.0 && score.recall <= 1.0, "BERTScore recall should be between 0 and 1, got: {}", score.recall);
        assert!(score.f1_score >= 0.0 && score.f1_score <= 1.0, "BERTScore F1 should be between 0 and 1, got: {}", score.f1_score);
    }

    #[test]
    fn test_legacy_semantic_similarity() {
        let sim_score = SemanticSimilarity::cosine_similarity("hello world", "hi world");
        assert!(sim_score >= 0.0 && sim_score <= 1.0, "Semantic similarity should be between 0 and 1, got: {}", sim_score);
        assert!(sim_score > 0.1, "Similar phrases should have reasonable similarity, got: {}", sim_score);
    }

    #[test]
    fn test_legacy_bert_score_configuration() {
        let bert_score = BertScore::new()
            .with_model("bert-base-uncased")
            .with_fast_tokenizer(true);

        let references = vec!["Test sentence."];
        let candidates = vec!["Another test sentence."];

        let result = bert_score.compute(&references, &candidates);
        assert!(result.is_ok(), "Configured BERTScore should work: {:?}", result.err());
    }
}

/// Legacy BERTScore calculator (backward compatibility wrapper)
///
/// This struct provides the original BERTScore API while internally using the modern
/// modular BERTScore implementation for all calculations.
///
/// # Migration
///
/// Consider migrating to `crate::metrics::BertScore` for enhanced features:
/// - Multiple model backends
/// - Confidence intervals
/// - Statistical analysis
/// - Comprehensive result objects
#[derive(Debug, Clone)]
pub struct BertScore {
    model_name: String,
    use_fast_tokenizer: bool,
    normalize_embeddings: bool,
    /// Internal modern BERTScore calculator
    _modern_bert_score: ModernBertScore,
}

/// Legacy BERTScore result (backward compatibility)
///
/// This struct provides the original BERTScore result format while internally
/// using data from the modern BertScoreResult.
#[derive(Debug, Clone)]
pub struct BertScoreResult {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
}

impl Default for BertScore {
    fn default() -> Self {
        let config = BertScoreConfig {
            model_name: "bert-base-uncased".to_string(),
            use_fast_tokenizer: true,
            normalize_embeddings: true,
            ..Default::default()
        };
        Self {
            model_name: "bert-base-uncased".to_string(),
            use_fast_tokenizer: true,
            normalize_embeddings: true,
            _modern_bert_score: ModernBertScore::new(config),
        }
    }
}

impl BertScore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_model(mut self, model_name: &str) -> Self {
        self.model_name = model_name.to_string();
        let config = BertScoreConfig {
            model_name: model_name.to_string(),
            use_fast_tokenizer: self.use_fast_tokenizer,
            normalize_embeddings: self.normalize_embeddings,
            ..Default::default()
        };
        self._modern_bert_score = ModernBertScore::new(config);
        self
    }

    pub fn with_fast_tokenizer(mut self, use_fast: bool) -> Self {
        self.use_fast_tokenizer = use_fast;
        let config = BertScoreConfig {
            model_name: self.model_name.clone(),
            use_fast_tokenizer: use_fast,
            normalize_embeddings: self.normalize_embeddings,
            ..Default::default()
        };
        self._modern_bert_score = ModernBertScore::new(config);
        self
    }

    /// Compute BERTScore between references and candidates
    pub fn compute(
        &self,
        references: &[&str],
        candidates: &[&str],
    ) -> Result<Vec<BertScoreResult>> {
        if references.len() != candidates.len() {
            return Err(TextError::Other(anyhow::anyhow!(
                "References and candidates must have the same length"
            )));
        }

        let mut results = Vec::new();

        for (reference, candidate) in references.iter().zip(candidates.iter()) {
            let modern_result = self._modern_bert_score.calculate_bert_score(reference, candidate);
            results.push(BertScoreResult {
                precision: modern_result.precision,
                recall: modern_result.recall,
                f1_score: modern_result.f1_score,
            });
        }

        Ok(results)
    }


/// Legacy semantic similarity calculator (backward compatibility wrapper)
///
/// This struct provides the original semantic similarity API while internally using the modern
/// modular semantic similarity implementation for all calculations.
///
/// # Migration
///
/// Consider migrating to `crate::metrics::SemanticSimilarity` for enhanced features:
/// - Multiple similarity algorithms
/// - Domain-specific analysis
/// - Confidence intervals
/// - Comprehensive result objects
#[derive(Debug, Clone)]
pub struct SemanticSimilarity {
    /// Internal modern semantic similarity calculator
    _modern_semantic: ModernSemanticSimilarity,
}

impl Default for SemanticSimilarity {
    fn default() -> Self {
        Self {
            _modern_semantic: ModernSemanticSimilarity::new(SemanticConfig::default()),
        }
    }
}

impl SemanticSimilarity {
    /// Compute semantic similarity using contextual features (static method for backward compatibility)
    pub fn cosine_similarity(text1: &str, text2: &str) -> f64 {
        let calculator = Self::default();
        let result = calculator._modern_semantic.calculate_similarity(text1, text2);
        result.overall_similarity
    }


/// Legacy custom metrics framework (backward compatibility wrapper)
///
/// This module provides the original custom metrics API while internally using the modern
/// modular evaluation system for all calculations.
///
/// # Migration
///
/// Consider migrating to the modern `crate::metrics::TextEvaluator` system:
/// - Unified configuration-driven evaluation
/// - Comprehensive result objects with confidence intervals
/// - Statistical analysis and quality assessment
/// - Built-in metric correlation and reliability analysis
///
/// ## Example Migration
///
/// ```rust
/// // Legacy custom metrics (still supported)
/// let mut registry = MetricRegistry::new();
/// registry.register(Box::new(WordOverlapMetric::new()));
/// let scores = registry.evaluate("word_overlap", &predictions, &references)?;
///
/// // Modern unified evaluation (recommended)
/// use crate::metrics::TextEvaluator;
/// let evaluator = TextEvaluator::with_default_config();
/// let result = evaluator.comprehensive_evaluation(reference, candidate);
/// let overlap_score = result.overlap_result.jaccard;
/// ```
pub mod custom {
    use super::*;

    /// Legacy trait for custom metrics (backward compatibility)
    ///
    /// This trait provides the original CustomMetric interface while internally
    /// mapping to the modern modular evaluation system.
    ///
    /// # Migration
    ///
    /// The modern system uses configuration-driven evaluation rather than trait objects.
    /// Consider using `TextEvaluator` with custom `EvaluationConfig` instead.
    pub trait CustomMetric: Send + Sync {
        /// Name of the metric
        fn name(&self) -> &str;

        /// Description of what the metric measures
        fn description(&self) -> &str;

        /// Compute the metric for a single prediction/reference pair
        fn compute_single(&self, prediction: &str, reference: &str) -> Result<f64>;

        /// Compute the metric for multiple prediction/reference pairs
        fn compute_batch(&self, predictions: &[&str], references: &[&str]) -> Result<Vec<f64>> {
            if predictions.len() != references.len() {
                return Err(TextError::Other(anyhow::anyhow!(
                    "Predictions and references must have the same length"
                )));
            }

            let mut results = Vec::new();
            for (pred, ref_text) in predictions.iter().zip(references.iter()) {
                results.push(self.compute_single(pred, ref_text)?);
            }
            Ok(results)
        }

        /// Aggregate scores (default is mean)
        fn aggregate(&self, scores: &[f64]) -> f64 {
            if scores.is_empty() {
                0.0
            } else {
                scores.iter().sum::<f64>() / scores.len() as f64
            }
        }

        /// Get metric configuration as Any for dynamic casting
        fn as_any(&self) -> &dyn Any;
    }

    /// Legacy word overlap metric (backward compatibility wrapper)
    ///
    /// This struct provides the original word overlap API while internally using the modern
    /// modular overlap implementation for all calculations.
    ///
    /// # Migration
    ///
    /// Consider using `crate::metrics::WordOverlapCalculator` directly.
    #[derive(Debug, Clone)]
    pub struct WordOverlapMetric {
        name: String,
        case_sensitive: bool,
        stem_words: bool,
        /// Internal modern overlap calculator
        _modern_overlap: WordOverlapCalculator,
    }

    impl WordOverlapMetric {
        pub fn new() -> Self {
            let config = OverlapConfig {
                case_sensitive: false,
                use_stemming: false,
                ..Default::default()
            };
            Self {
                name: "word_overlap".to_string(),
                case_sensitive: false,
                stem_words: false,
                _modern_overlap: WordOverlapCalculator::new(config),
            }
        }

        pub fn with_case_sensitive(mut self, case_sensitive: bool) -> Self {
            self.case_sensitive = case_sensitive;
            let config = OverlapConfig {
                case_sensitive,
                use_stemming: self.stem_words,
                ..Default::default()
            };
            self._modern_overlap = WordOverlapCalculator::new(config);
            self
        }

        pub fn with_stemming(mut self, stem_words: bool) -> Self {
            self.stem_words = stem_words;
            let config = OverlapConfig {
                case_sensitive: self.case_sensitive,
                use_stemming: stem_words,
                ..Default::default()
            };
            self._modern_overlap = WordOverlapCalculator::new(config);
            self
        }
    }

    impl CustomMetric for WordOverlapMetric {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Measures lexical overlap between prediction and reference"
        }

        fn compute_single(&self, prediction: &str, reference: &str) -> Result<f64> {
            let result = self._modern_overlap.calculate_comprehensive_overlap(reference, prediction);
            Ok(result.jaccard) // Use Jaccard coefficient as the overlap measure
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Legacy semantic coherence metric (backward compatibility wrapper)
    ///
    /// This struct provides the original semantic coherence API while internally using the modern
    /// modular coherence implementation for all calculations.
    ///
    /// # Migration
    ///
    /// Consider using `crate::metrics::CoherenceAnalyzer` directly.
    #[derive(Debug, Clone)]
    pub struct SemanticCoherenceMetric {
        name: String,
        window_size: usize,
        /// Internal modern coherence analyzer
        _modern_coherence: CoherenceAnalyzer,
    }

    impl SemanticCoherenceMetric {
        pub fn new() -> Self {
            let config = CoherenceConfig {
                window_size: 5,
                ..Default::default()
            };
            Self {
                name: "semantic_coherence".to_string(),
                window_size: 5,
                _modern_coherence: CoherenceAnalyzer::new(config),
            }
        }

        pub fn with_window_size(mut self, window_size: usize) -> Self {
            self.window_size = window_size;
            let config = CoherenceConfig {
                window_size,
                ..Default::default()
            };
            self._modern_coherence = CoherenceAnalyzer::new(config);
            self
        }
    }

    impl CustomMetric for SemanticCoherenceMetric {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Measures semantic coherence within text using sliding window analysis"
        }

        fn compute_single(&self, prediction: &str, _reference: &str) -> Result<f64> {
            let result = self._modern_coherence.analyze_coherence(prediction);
            Ok(result.overall_coherence)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Legacy fluency metric (backward compatibility wrapper)
    ///
    /// This struct provides the original fluency API while internally using the modern
    /// modular fluency implementation for all calculations.
    ///
    /// # Migration
    ///
    /// Consider using `crate::metrics::FluencyAnalyzer` directly.
    #[derive(Debug, Clone)]
    pub struct FluencyMetric {
        name: String,
        use_perplexity: bool,
        /// Internal modern fluency analyzer
        _modern_fluency: FluencyAnalyzer,
    }

    impl FluencyMetric {
        pub fn new() -> Self {
            let config = FluencyConfig {
                enable_perplexity_analysis: true,
                ..Default::default()
            };
            Self {
                name: "fluency".to_string(),
                use_perplexity: true,
                _modern_fluency: FluencyAnalyzer::new(config),
            }
        }

        pub fn with_perplexity(mut self, use_perplexity: bool) -> Self {
            self.use_perplexity = use_perplexity;
            let config = FluencyConfig {
                enable_perplexity_analysis: use_perplexity,
                ..Default::default()
            };
            self._modern_fluency = FluencyAnalyzer::new(config);
            self
        }
    }

    impl CustomMetric for FluencyMetric {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Measures text fluency using linguistic heuristics"
        }

        fn compute_single(&self, prediction: &str, _reference: &str) -> Result<f64> {
            let result = self._modern_fluency.analyze_fluency(prediction);
            Ok(result.overall_fluency)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Legacy composite metric (backward compatibility wrapper)
    ///
    /// This struct provides the original composite metric API while internally using the modern
    /// unified TextEvaluator system for all calculations.
    ///
    /// # Migration
    ///
    /// Consider using `TextEvaluator` with custom `EvaluationWeights`:
    ///
    /// ```rust
    /// let mut config = EvaluationConfig::default();
    /// config.weights.bleu_weight = 0.3;
    /// config.weights.semantic_weight = 0.4;
    /// config.weights.fluency_weight = 0.3;
    /// let evaluator = TextEvaluator::new(config);
    /// let result = evaluator.comprehensive_evaluation(reference, candidate);
    /// let composite_score = result.overall_score;
    /// ```
    pub struct CompositeMetric {
        name: String,
        /// Internal modern text evaluator with custom weights
        _modern_evaluator: TextEvaluator,
        /// Store metric names and weights for backward compatibility
        _metric_weights: Vec<(String, f64)>,
    }

    impl CompositeMetric {
        pub fn new(name: String) -> Self {
            Self {
                name,
                _modern_evaluator: TextEvaluator::with_default_config(),
                _metric_weights: Vec::new(),
            }
        }

        pub fn add_metric(mut self, metric: Box<dyn CustomMetric>, weight: f64) -> Self {
            let metric_name = metric.name().to_string();
            self._metric_weights.push((metric_name.clone(), weight));

            // Update the internal evaluator configuration based on known metric types
            self._update_evaluator_weights();
            self
        }

        pub fn add_weighted_metrics(
            mut self,
            metrics_weights: Vec<(Box<dyn CustomMetric>, f64)>,
        ) -> Self {
            for (metric, weight) in metrics_weights {
                let metric_name = metric.name().to_string();
                self._metric_weights.push((metric_name, weight));
            }
            self._update_evaluator_weights();
            self
        }

        fn _update_evaluator_weights(&mut self) {
            let mut config = EvaluationConfig::default();
            let mut total_weight = 0.0;

            // Reset all weights to 0
            config.weights = EvaluationWeights {
                bleu_weight: 0.0,
                rouge_weight: 0.0,
                bert_score_weight: 0.0,
                semantic_weight: 0.0,
                overlap_weight: 0.0,
                coherence_weight: 0.0,
                fluency_weight: 0.0,
                perplexity_weight: 0.0,
                edit_distance_weight: 0.0,
            };

            // Map legacy metric names to modern weights
            for (name, weight) in &self._metric_weights {
                total_weight += weight;
                match name.as_str() {
                    "word_overlap" => config.weights.overlap_weight += weight,
                    "semantic_coherence" => config.weights.coherence_weight += weight,
                    "fluency" => config.weights.fluency_weight += weight,
                    "bleu" | "bleu_score" => config.weights.bleu_weight += weight,
                    "rouge" | "rouge_score" => config.weights.rouge_weight += weight,
                    "bert_score" => config.weights.bert_score_weight += weight,
                    "semantic_similarity" => config.weights.semantic_weight += weight,
                    "perplexity" => config.weights.perplexity_weight += weight,
                    "edit_distance" => config.weights.edit_distance_weight += weight,
                    _ => {
                        // Unknown metric, distribute weight evenly
                        let per_metric_weight = weight / 9.0;
                        config.weights.bleu_weight += per_metric_weight;
                        config.weights.rouge_weight += per_metric_weight;
                        config.weights.bert_score_weight += per_metric_weight;
                        config.weights.semantic_weight += per_metric_weight;
                        config.weights.overlap_weight += per_metric_weight;
                        config.weights.coherence_weight += per_metric_weight;
                        config.weights.fluency_weight += per_metric_weight;
                        config.weights.perplexity_weight += per_metric_weight;
                        config.weights.edit_distance_weight += per_metric_weight;
                    }
                }
            }

            // Normalize weights
            if total_weight > 0.0 {
                config.weights.bleu_weight /= total_weight;
                config.weights.rouge_weight /= total_weight;
                config.weights.bert_score_weight /= total_weight;
                config.weights.semantic_weight /= total_weight;
                config.weights.overlap_weight /= total_weight;
                config.weights.coherence_weight /= total_weight;
                config.weights.fluency_weight /= total_weight;
                config.weights.perplexity_weight /= total_weight;
                config.weights.edit_distance_weight /= total_weight;
            }

            self._modern_evaluator = TextEvaluator::new(config);
        }
    }

    impl CustomMetric for CompositeMetric {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Composite metric combining multiple evaluation metrics with weights"
        }

        fn compute_single(&self, prediction: &str, reference: &str) -> Result<f64> {
            let result = self._modern_evaluator.comprehensive_evaluation(reference, prediction);
            Ok(result.overall_score)
        }

        fn compute_batch(&self, predictions: &[&str], references: &[&str]) -> Result<Vec<f64>> {
            if predictions.len() != references.len() {
                return Err(TextError::Other(anyhow::anyhow!(
                    "Predictions and references must have the same length"
                )));
            }

            let mut results = Vec::new();
            for (prediction, reference) in predictions.iter().zip(references.iter()) {
                let score = self.compute_single(prediction, reference)?;
                results.push(score);
            }
            Ok(results)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Legacy metric registry (backward compatibility wrapper)
    ///
    /// This struct provides the original metric registry API while internally using the modern
    /// unified evaluation system.
    ///
    /// # Migration
    ///
    /// Consider using `TextEvaluator` directly with custom configuration:
    ///
    /// ```rust
    /// let evaluator = TextEvaluator::with_default_config();
    /// let result = evaluator.comprehensive_evaluation(reference, candidate);
    /// // Access individual metric scores from result
    /// let bleu_score = result.bleu_result.bleu_score;
    /// let rouge_score = result.rouge_result.rouge_score;
    /// ```
    pub struct MetricRegistry {
        /// Internal storage for registered metrics (for backward compatibility)
        metrics: HashMap<String, Box<dyn CustomMetric>>,
        /// Modern text evaluator for unified evaluation
        _modern_evaluator: TextEvaluator,
    }

    impl MetricRegistry {
        pub fn new() -> Self {
            Self {
                metrics: HashMap::new(),
                _modern_evaluator: TextEvaluator::with_default_config(),
            }
        }

        /// Register a custom metric
        pub fn register(&mut self, metric: Box<dyn CustomMetric>) {
            let name = metric.name().to_string();
            self.metrics.insert(name, metric);
        }

        /// Get a metric by name
        pub fn get(&self, name: &str) -> Option<&dyn CustomMetric> {
            self.metrics.get(name).map(|m| m.as_ref())
        }

        /// List all registered metric names
        pub fn list_metrics(&self) -> Vec<String> {
            let mut names: Vec<String> = self.metrics.keys().cloned().collect();
            // Add built-in metrics available through the modern system
            names.extend_from_slice(&[
                "bleu_score".to_string(),
                "rouge_score".to_string(),
                "bert_score".to_string(),
                "semantic_similarity".to_string(),
                "edit_distance".to_string(),
                "perplexity".to_string(),
                "overall_coherence".to_string(),
                "overall_fluency".to_string(),
            ]);
            names.sort();
            names.dedup();
            names
        }

        /// Evaluate using a specific metric
        pub fn evaluate(
            &self,
            metric_name: &str,
            predictions: &[&str],
            references: &[&str],
        ) -> Result<Vec<f64>> {
            if predictions.len() != references.len() {
                return Err(TextError::Other(anyhow::anyhow!(
                    "Predictions and references must have the same length"
                )));
            }

            // First try custom registered metrics
            if let Some(metric) = self.get(metric_name) {
                return metric.compute_batch(predictions, references);
            }

            // Fall back to modern system for built-in metrics
            let mut results = Vec::new();
            for (prediction, reference) in predictions.iter().zip(references.iter()) {
                let result = self._modern_evaluator.comprehensive_evaluation(reference, prediction);

                let score = match metric_name {
                    "bleu_score" => result.bleu_result.bleu_score,
                    "rouge_score" => result.rouge_result.rouge_score,
                    "bert_score" => result.bert_score_result.f1_score,
                    "semantic_similarity" => result.semantic_result.overall_similarity,
                    "edit_distance" => 1.0 - (result.edit_distance_result.normalized_distance / 100.0).min(1.0),
                    "perplexity" => result.perplexity_result.as_ref().map(|p| p.perplexity).unwrap_or(0.0),
                    "overall_coherence" => result.coherence_result.overall_coherence,
                    "overall_fluency" => result.fluency_result.overall_fluency,
                    "word_overlap" => result.overlap_result.jaccard,
                    _ => {
                        return Err(TextError::Other(anyhow::anyhow!(
                            "Metric '{}' not found in registry",
                            metric_name
                        )));
                    }
                };
                results.push(score);
            }
            Ok(results)
        }

        /// Evaluate using multiple metrics
        pub fn evaluate_multiple(
            &self,
            metric_names: &[&str],
            predictions: &[&str],
            references: &[&str],
        ) -> Result<HashMap<String, Vec<f64>>> {
            let mut results = HashMap::new();

            for &metric_name in metric_names {
                let scores = self.evaluate(metric_name, predictions, references)?;
                results.insert(metric_name.to_string(), scores);
            }

            Ok(results)
        }

        /// Create a composite metric from registered metrics
        pub fn create_composite(
            &self,
            name: String,
            metric_weights: &[(&str, f64)],
        ) -> Result<CompositeMetric> {
            let mut composite = CompositeMetric::new(name);

            for &(metric_name, weight) in metric_weights {
                // Create new instances of known metrics for the composite
                let cloned_metric: Box<dyn CustomMetric> = match metric_name {
                    "word_overlap" => Box::new(WordOverlapMetric::new()),
                    "semantic_coherence" => Box::new(SemanticCoherenceMetric::new()),
                    "fluency" => Box::new(FluencyMetric::new()),
                    _ => {
                        return Err(TextError::Other(anyhow::anyhow!(
                            "Cannot create composite with metric: {}. Supported metrics: word_overlap, semantic_coherence, fluency",
                            metric_name
                        )))
                    }
                };
                composite = composite.add_metric(cloned_metric, weight);
            }

            Ok(composite)
        }

        /// Register default metrics
        pub fn register_defaults(&mut self) {
            self.register(Box::new(WordOverlapMetric::new()));
            self.register(Box::new(SemanticCoherenceMetric::new()));
            self.register(Box::new(FluencyMetric::new()));
        }
    }

    impl Default for MetricRegistry {
        fn default() -> Self {
            let mut registry = Self::new();
            registry.register_defaults();
            registry
        }
    }

    /// Legacy evaluation framework (backward compatibility wrapper)
    ///
    /// This struct provides the original evaluation framework API while internally using the modern
    /// unified TextEvaluator system.
    ///
    /// # Migration
    ///
    /// Consider using `TextEvaluator` directly:
    ///
    /// ```rust
    /// let evaluator = TextEvaluator::with_default_config();
    /// let result = evaluator.comprehensive_evaluation(reference, candidate);
    /// // Access comprehensive results with confidence intervals and statistical analysis
    /// ```
    pub struct EvaluationFramework {
        registry: MetricRegistry,
        default_metrics: Vec<String>,
        /// Internal modern text evaluator
        _modern_evaluator: TextEvaluator,
    }

    impl EvaluationFramework {
        pub fn new() -> Self {
            Self {
                registry: MetricRegistry::default(),
                default_metrics: vec![
                    "word_overlap".to_string(),
                    "semantic_coherence".to_string(),
                    "fluency".to_string(),
                ],
                _modern_evaluator: TextEvaluator::with_default_config(),
            }
        }

        pub fn with_registry(mut self, registry: MetricRegistry) -> Self {
            self.registry = registry;
            self
        }

        pub fn add_metric(&mut self, metric: Box<dyn CustomMetric>) {
            self.registry.register(metric);
        }

        pub fn set_default_metrics(&mut self, metrics: Vec<String>) {
            self.default_metrics = metrics;
        }

        /// Comprehensive evaluation using all default metrics
        pub fn evaluate_comprehensive(
            &self,
            predictions: &[&str],
            references: &[&str],
        ) -> Result<EvaluationResults> {
            let metric_names: Vec<&str> = self.default_metrics.iter().map(|s| s.as_str()).collect();
            let results =
                self.registry
                    .evaluate_multiple(&metric_names, predictions, references)?;

            Ok(EvaluationResults::new(results))
        }

        /// Evaluate with custom metric selection
        pub fn evaluate_custom(
            &self,
            predictions: &[&str],
            references: &[&str],
            metrics: &[&str],
        ) -> Result<EvaluationResults> {
            let results = self
                .registry
                .evaluate_multiple(metrics, predictions, references)?;
            Ok(EvaluationResults::new(results))
        }

        /// Evaluate with a single composite metric
        pub fn evaluate_composite(
            &self,
            predictions: &[&str],
            references: &[&str],
            composite_spec: &[(&str, f64)],
        ) -> Result<f64> {
            let composite = self
                .registry
                .create_composite("custom_composite".to_string(), composite_spec)?;
            let scores = composite.compute_batch(predictions, references)?;
            Ok(composite.aggregate(&scores))
        }
    }

    impl Default for EvaluationFramework {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Legacy evaluation results (backward compatibility wrapper)
    ///
    /// This struct provides the original evaluation results API while internally
    /// working with data that could be derived from ComprehensiveEvaluationResult.
    ///
    /// # Migration
    ///
    /// Consider using `ComprehensiveEvaluationResult` directly:
    ///
    /// ```rust
    /// let evaluator = TextEvaluator::with_default_config();
    /// let result = evaluator.comprehensive_evaluation(reference, candidate);
    /// // Access rich results with confidence intervals, statistical analysis, quality assessment
    /// let overall_score = result.overall_score;
    /// let metric_breakdown = result.metric_breakdown;
    /// let quality_assessment = result.quality_assessment;
    /// ```
    #[derive(Debug, Clone)]
    pub struct EvaluationResults {
        scores: HashMap<String, Vec<f64>>,
        aggregated: HashMap<String, f64>,
    }

    impl EvaluationResults {
        pub fn new(scores: HashMap<String, Vec<f64>>) -> Self {
            let mut aggregated = HashMap::new();

            for (metric_name, values) in &scores {
                let avg = if values.is_empty() {
                    0.0
                } else {
                    values.iter().sum::<f64>() / values.len() as f64
                };
                aggregated.insert(metric_name.clone(), avg);
            }

            Self { scores, aggregated }
        }

        /// Create EvaluationResults from modern ComprehensiveEvaluationResult
        pub fn from_comprehensive(result: &ComprehensiveEvaluationResult) -> Self {
            let mut scores = HashMap::new();
            let mut aggregated = HashMap::new();

            // Extract scores from comprehensive result
            let single_scores = vec![
                ("bleu_score".to_string(), vec![result.bleu_result.bleu_score]),
                ("rouge_score".to_string(), vec![result.rouge_result.rouge_score]),
                ("bert_score".to_string(), vec![result.bert_score_result.f1_score]),
                ("semantic_similarity".to_string(), vec![result.semantic_result.overall_similarity]),
                ("word_overlap".to_string(), vec![result.overlap_result.jaccard]),
                ("coherence".to_string(), vec![result.coherence_result.overall_coherence]),
                ("fluency".to_string(), vec![result.fluency_result.overall_fluency]),
                ("overall_score".to_string(), vec![result.overall_score]),
            ];

            for (metric_name, values) in single_scores {
                let avg = values.iter().sum::<f64>() / values.len() as f64;
                scores.insert(metric_name.clone(), values);
                aggregated.insert(metric_name, avg);
            }

            Self { scores, aggregated }
        }

        /// Get scores for a specific metric
        pub fn get_scores(&self, metric: &str) -> Option<&Vec<f64>> {
            self.scores.get(metric)
        }

        /// Get aggregated score for a specific metric
        pub fn get_aggregated(&self, metric: &str) -> Option<f64> {
            self.aggregated.get(metric).copied()
        }

        /// Get all aggregated scores
        pub fn get_all_aggregated(&self) -> &HashMap<String, f64> {
            &self.aggregated
        }

        /// Get all detailed scores
        pub fn get_all_scores(&self) -> &HashMap<String, Vec<f64>> {
            &self.scores
        }

        /// Compute overall composite score
        pub fn composite_score(&self, weights: Option<&HashMap<String, f64>>) -> f64 {
            if self.aggregated.is_empty() {
                return 0.0;
            }

            // If overall_score is available, use it
            if let Some(&score) = self.aggregated.get("overall_score") {
                return score;
            }

            if let Some(weights) = weights {
                let mut weighted_sum = 0.0;
                let mut total_weight = 0.0;

                for (metric, &score) in &self.aggregated {
                    if let Some(&weight) = weights.get(metric) {
                        weighted_sum += score * weight;
                        total_weight += weight;
                    }
                }

                if total_weight > 0.0 {
                    weighted_sum / total_weight
                } else {
                    0.0
                }
            } else {
                // Equal weights
                let sum: f64 = self.aggregated.values().sum();
                sum / self.aggregated.len() as f64
            }
        }

        /// Create a summary report
        pub fn summary(&self) -> String {
            let mut report = String::new();
            report.push_str("Evaluation Results Summary:\n");
            report.push_str("==========================\n");

            // Sort metrics for consistent output
            let mut sorted_metrics: Vec<_> = self.aggregated.iter().collect();
            sorted_metrics.sort_by_key(|(name, _)| *name);

            for (metric, &score) in sorted_metrics {
                report.push_str(&format!("{}: {:.4}\n", metric, score));
            }

            let composite = self.composite_score(None);
            report.push_str(&format!("\nOverall Score: {:.4}\n", composite));

            report
        }
    }
}

/// Comprehensive test suite for legacy custom metrics framework
///
/// These tests verify that the custom metrics API works correctly with the modern
/// evaluation system underneath.
#[cfg(test)]
mod custom_tests {
    use super::custom::*;

    #[test]
    fn test_legacy_word_overlap_metric() {
        let metric = WordOverlapMetric::new();
        let score = metric
            .compute_single("hello world", "hello universe")
            .unwrap();
        assert!(score >= 0.0 && score <= 1.0, "Word overlap score should be between 0 and 1, got: {}", score);
        assert!(score > 0.0, "'hello world' and 'hello universe' should have some overlap, got: {}", score);
    }

    #[test]
    fn test_legacy_word_overlap_configuration() {
        let case_sensitive = WordOverlapMetric::new().with_case_sensitive(true);
        let with_stemming = WordOverlapMetric::new().with_stemming(true);

        let score1 = case_sensitive.compute_single("Hello World", "hello world").unwrap();
        let score2 = with_stemming.compute_single("running dogs", "ran dog").unwrap();

        assert!(score1 >= 0.0 && score1 <= 1.0, "Case sensitive score should be valid: {}", score1);
        assert!(score2 >= 0.0 && score2 <= 1.0, "Stemming score should be valid: {}", score2);
    }

    #[test]
    fn test_legacy_semantic_coherence_metric() {
        let metric = SemanticCoherenceMetric::new();
        let score = metric
            .compute_single("The cat sat on the mat. The cat was happy. The happy cat played.", "")
            .unwrap();
        assert!(score >= 0.0 && score <= 1.0, "Coherence score should be between 0 and 1, got: {}", score);
        assert!(score > 0.3, "Coherent text should have reasonable coherence score, got: {}", score);
    }

    #[test]
    fn test_legacy_fluency_metric() {
        let metric = FluencyMetric::new();
        let score = metric
            .compute_single("This is a well-written sentence with proper grammar.", "")
            .unwrap();
        assert!(score >= 0.0 && score <= 1.0, "Fluency score should be between 0 and 1, got: {}", score);
        assert!(score > 0.3, "Well-written text should have reasonable fluency score, got: {}", score);
    }

    #[test]
    fn test_legacy_fluency_metric_configuration() {
        let with_perplexity = FluencyMetric::new().with_perplexity(true);
        let without_perplexity = FluencyMetric::new().with_perplexity(false);

        let score1 = with_perplexity.compute_single("Well-written text.", "").unwrap();
        let score2 = without_perplexity.compute_single("Well-written text.", "").unwrap();

        assert!(score1 >= 0.0 && score1 <= 1.0, "Perplexity-enabled fluency should be valid: {}", score1);
        assert!(score2 >= 0.0 && score2 <= 1.0, "Non-perplexity fluency should be valid: {}", score2);
    }

    #[test]
    fn test_legacy_composite_metric() {
        let overlap = Box::new(WordOverlapMetric::new());
        let fluency = Box::new(FluencyMetric::new());

        let composite = CompositeMetric::new("test_composite".to_string())
            .add_metric(overlap, 0.6)
            .add_metric(fluency, 0.4);

        let score = composite
            .compute_single("hello world", "hello universe")
            .unwrap();
        assert!(score >= 0.0 && score <= 1.0, "Composite score should be between 0 and 1, got: {}", score);
    }

    #[test]
    fn test_legacy_composite_metric_batch() {
        let overlap = Box::new(WordOverlapMetric::new());
        let coherence = Box::new(SemanticCoherenceMetric::new());

        let composite = CompositeMetric::new("batch_test".to_string())
            .add_metric(overlap, 0.5)
            .add_metric(coherence, 0.5);

        let predictions = vec!["hello world", "good morning"];
        let references = vec!["hello universe", "good evening"];

        let scores = composite.compute_batch(&predictions, &references).unwrap();
        assert_eq!(scores.len(), 2, "Should have two scores for two predictions");
        for (i, &score) in scores.iter().enumerate() {
            assert!(score >= 0.0 && score <= 1.0, "Batch score {} should be valid: {}", i, score);
        }
    }

    #[test]
    fn test_legacy_metric_registry() {
        let mut registry = MetricRegistry::new();
        registry.register(Box::new(WordOverlapMetric::new()));

        let metrics = registry.list_metrics();
        assert!(metrics.contains(&"word_overlap".to_string()), "Registry should contain word_overlap metric");
        assert!(metrics.contains(&"bleu_score".to_string()), "Registry should contain built-in bleu_score metric");

        let predictions = vec!["hello world", "goodbye world"];
        let references = vec!["hello universe", "goodbye universe"];

        // Test custom metric
        let results = registry.evaluate("word_overlap", &predictions, &references);
        assert!(results.is_ok(), "Word overlap evaluation should succeed: {:?}", results.err());

        let scores = results.unwrap();
        assert_eq!(scores.len(), 2, "Should have two scores for two predictions");

        // Test built-in metric
        let bleu_results = registry.evaluate("bleu_score", &predictions, &references);
        assert!(bleu_results.is_ok(), "BLEU evaluation should succeed: {:?}", bleu_results.err());
    }

    #[test]
    fn test_legacy_metric_registry_multiple() {
        let registry = MetricRegistry::default(); // Includes default metrics

        let predictions = vec!["This is a test.", "Another test."];
        let references = vec!["This is a reference.", "Another reference."];

        let metric_names = vec!["word_overlap", "semantic_coherence", "fluency"];
        let results = registry.evaluate_multiple(&metric_names, &predictions, &references);

        assert!(results.is_ok(), "Multiple metric evaluation should succeed: {:?}", results.err());

        let scores_map = results.unwrap();
        assert_eq!(scores_map.len(), 3, "Should have results for all three metrics");

        for metric_name in &metric_names {
            assert!(scores_map.contains_key(*metric_name), "Results should contain metric: {}", metric_name);
            let scores = scores_map.get(*metric_name).unwrap();
            assert_eq!(scores.len(), 2, "Each metric should have two scores");
        }
    }

    #[test]
    fn test_legacy_evaluation_framework() {
        let framework = EvaluationFramework::new();

        let predictions = vec!["This is a test sentence.", "Another test here."];
        let references = vec!["This is a reference sentence.", "Another reference here."];

        let results = framework.evaluate_comprehensive(&predictions, &references);
        assert!(results.is_ok(), "Comprehensive evaluation should succeed: {:?}", results.err());

        let eval_results = results.unwrap();
        assert!(eval_results.get_aggregated("word_overlap").is_some(), "Should have word overlap scores");
        assert!(eval_results.get_aggregated("fluency").is_some(), "Should have fluency scores");
        assert!(eval_results.get_aggregated("semantic_coherence").is_some(), "Should have coherence scores");

        // Test summary generation
        let summary = eval_results.summary();
        assert!(summary.contains("Evaluation Results Summary"), "Summary should contain header");
        assert!(summary.contains("Overall Score"), "Summary should contain overall score");
    }

    #[test]
    fn test_legacy_evaluation_framework_custom() {
        let framework = EvaluationFramework::new();

        let predictions = vec!["Test prediction"];
        let references = vec!["Test reference"];

        let custom_metrics = vec!["word_overlap", "fluency"];
        let results = framework.evaluate_custom(&predictions, &references, &custom_metrics);

        assert!(results.is_ok(), "Custom evaluation should succeed: {:?}", results.err());

        let eval_results = results.unwrap();
        assert!(eval_results.get_aggregated("word_overlap").is_some(), "Should have word overlap from custom evaluation");
        assert!(eval_results.get_aggregated("fluency").is_some(), "Should have fluency from custom evaluation");
    }

    #[test]
    fn test_legacy_evaluation_framework_composite() {
        let framework = EvaluationFramework::new();

        let predictions = vec!["Test prediction"];
        let references = vec!["Test reference"];

        let composite_spec = vec![("word_overlap", 0.6), ("fluency", 0.4)];
        let composite_score = framework.evaluate_composite(&predictions, &references, &composite_spec);

        assert!(composite_score.is_ok(), "Composite evaluation should succeed: {:?}", composite_score.err());

        let score = composite_score.unwrap();
        assert!(score >= 0.0 && score <= 1.0, "Composite score should be between 0 and 1, got: {}", score);
    }

    #[test]
    fn test_legacy_evaluation_results_from_comprehensive() {
        let evaluator = TextEvaluator::with_default_config();
        let comprehensive_result = evaluator.comprehensive_evaluation(
            "This is a reference text",
            "This is a candidate text"
        );

        let legacy_results = EvaluationResults::from_comprehensive(&comprehensive_result);

        assert!(legacy_results.get_aggregated("overall_score").is_some(), "Should extract overall score");
        assert!(legacy_results.get_aggregated("bleu_score").is_some(), "Should extract BLEU score");
        assert!(legacy_results.get_aggregated("rouge_score").is_some(), "Should extract ROUGE score");

        let composite = legacy_results.composite_score(None);
        assert!(composite >= 0.0 && composite <= 1.0, "Composite from comprehensive should be valid: {}", composite);
    }
}
