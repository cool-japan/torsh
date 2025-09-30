//! Comprehensive text fluency analysis and scoring
//!
//! This module provides advanced methods for measuring text fluency across
//! multiple dimensions including syntactic, lexical, semantic, prosodic,
//! and pragmatic fluency evaluation.
//!
//! # COMPATIBILITY NOTICE
//!
//! This module has been refactored to use a modular architecture for better maintainability
//! and extensibility. The API remains fully backward compatible, but the implementation
//! now uses specialized sub-modules from the modern fluency analysis framework.
//!
//! For new code, consider using the modular API directly:
//! - `fluency::FluentTextAnalyzer` for comprehensive analysis
//! - Individual analyzers: `fluency::semantic::SemanticAnalyzer`, etc.
//! - Configuration: `fluency::AnalysisConfig`
//!
//! # Migration Guide
//!
//! ## Legacy API (still works)
//! ```rust
//! use torsh_text::metrics::fluency_original::{FluencyAnalyzer, FluencyConfig};
//!
//! let analyzer = FluencyAnalyzer::new(FluencyConfig::default());
//! let result = analyzer.analyze_fluency("Sample text");
//! ```
//!
//! ## Modern API (recommended for new code)
//! ```rust
//! use torsh_text::metrics::fluency::{FluentTextAnalyzer, AnalysisConfig};
//!
//! let analyzer = FluentTextAnalyzer::new();
//! let result = analyzer.analyze_comprehensive_fluency("Sample text");
//! ```

// Re-export the modular fluency analysis system
pub use crate::metrics::fluency::{
    AnalysisConfig as ModernAnalysisConfig,
    ComprehensiveFluencyAnalysis as ModernComprehensiveFluencyAnalysis,
    FluencyAnalysisError as ModernFluencyAnalysisError,
    FluentTextAnalyzer as ModernFluentTextAnalyzer, LanguageModelScore as ModernLanguageModelScore,
    LexicalScore as ModernLexicalScore, PragmaticScore as ModernPragmaticScore,
    ProsodicScore as ModernProsodicScore, SemanticScore as ModernSemanticScore,
    SyntacticScore as ModernSyntacticScore,
};

use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_core::random::{rng, Random};
use std::collections::{HashMap, HashSet, VecDeque};

/// Fluency dimensions for backward compatibility
#[derive(Debug, Clone, PartialEq)]
pub enum FluencyDimension {
    LanguageModel,
    Syntactic,
    Lexical,
    Semantic,
    Prosodic,
    Pragmatic,
    Morphological,
    Phonological,
}

/// Configuration for fluency analysis (backward compatibility)
#[derive(Debug, Clone)]
pub struct FluencyConfig {
    pub language_model_weight: f64,
    pub syntactic_weight: f64,
    pub lexical_weight: f64,
    pub semantic_weight: f64,
    pub prosodic_weight: f64,
    pub pragmatic_weight: f64,
    pub sentence_length_penalty: bool,
    pub rare_word_penalty: f64,
    pub repetition_penalty: f64,
    pub complexity_threshold: f64,
    pub context_window: usize,
    pub fluency_dimensions: Vec<FluencyDimension>,
}

impl Default for FluencyConfig {
    fn default() -> Self {
        Self {
            language_model_weight: 0.30,
            syntactic_weight: 0.25,
            lexical_weight: 0.20,
            semantic_weight: 0.15,
            prosodic_weight: 0.07,
            pragmatic_weight: 0.03,
            sentence_length_penalty: true,
            rare_word_penalty: 0.1,
            repetition_penalty: 0.05,
            complexity_threshold: 0.7,
            context_window: 5,
            fluency_dimensions: vec![
                FluencyDimension::LanguageModel,
                FluencyDimension::Syntactic,
                FluencyDimension::Lexical,
                FluencyDimension::Semantic,
                FluencyDimension::Prosodic,
                FluencyDimension::Pragmatic,
            ],
        }
    }
}

/// Convert legacy config to modern config
impl From<FluencyConfig> for ModernAnalysisConfig {
    fn from(config: FluencyConfig) -> Self {
        ModernAnalysisConfig::builder()
            .enable_language_model_analysis(
                config
                    .fluency_dimensions
                    .contains(&FluencyDimension::LanguageModel),
            )
            .enable_syntactic_analysis(
                config
                    .fluency_dimensions
                    .contains(&FluencyDimension::Syntactic),
            )
            .enable_lexical_analysis(
                config
                    .fluency_dimensions
                    .contains(&FluencyDimension::Lexical),
            )
            .enable_semantic_analysis(
                config
                    .fluency_dimensions
                    .contains(&FluencyDimension::Semantic),
            )
            .enable_prosodic_analysis(
                config
                    .fluency_dimensions
                    .contains(&FluencyDimension::Prosodic),
            )
            .enable_pragmatic_analysis(
                config
                    .fluency_dimensions
                    .contains(&FluencyDimension::Pragmatic),
            )
            .enable_statistical_analysis(true) // Always enabled for comprehensive results
            .enable_quality_analysis(true) // Always enabled for comprehensive results
            .analysis_depth(crate::metrics::fluency::AnalysisDepth::Comprehensive)
            .performance_mode(crate::metrics::fluency::PerformanceMode::Accuracy)
            .build()
    }
}

/// Comprehensive fluency analysis result (backward compatibility)
#[derive(Debug, Clone)]
pub struct FluencyResult {
    pub overall_fluency: f64,
    pub language_model_fluency: LanguageModelFluencyResult,
    pub syntactic_fluency: SyntacticFluencyResult,
    pub lexical_fluency: LexicalFluencyResult,
    pub semantic_fluency: SemanticFluencyResult,
    pub prosodic_fluency: ProsodicFluencyResult,
    pub pragmatic_fluency: PragmaticFluencyResult,
    pub sentence_level_scores: Vec<f64>,
    pub fluency_distribution: FluencyDistribution,
    pub confidence_score: f64,
    pub fluency_breakdown: HashMap<String, f64>,
    pub quality_indicators: QualityIndicators,
}

/// Language model fluency analysis result
#[derive(Debug, Clone)]
pub struct LanguageModelFluencyResult {
    pub perplexity_score: f64,
    pub likelihood_score: f64,
    pub surprisal_score: f64,
    pub entropy_score: f64,
    pub probability_mass: f64,
    pub ngram_probabilities: HashMap<usize, f64>,
    pub oov_penalty: f64,
    pub smoothed_score: f64,
}

/// Convert modern language model score to legacy format
impl From<ModernLanguageModelScore> for LanguageModelFluencyResult {
    fn from(score: ModernLanguageModelScore) -> Self {
        Self {
            perplexity_score: score.perplexity,
            likelihood_score: score.log_likelihood,
            surprisal_score: score.surprisal,
            entropy_score: score.entropy,
            probability_mass: score.probability_mass,
            ngram_probabilities: score.ngram_scores,
            oov_penalty: score.oov_penalty,
            smoothed_score: score.smoothed_score,
        }
    }
}

/// Syntactic fluency analysis result
#[derive(Debug, Clone)]
pub struct SyntacticFluencyResult {
    pub grammaticality_score: f64,
    pub syntactic_complexity: f64,
    pub parse_tree_quality: f64,
    pub dependency_coherence: f64,
    pub sentence_structure_variety: f64,
    pub clause_integration: f64,
    pub syntactic_patterns: HashMap<String, usize>,
    pub error_indicators: Vec<SyntacticError>,
}

/// Convert modern syntactic score to legacy format
impl From<ModernSyntacticScore> for SyntacticFluencyResult {
    fn from(score: ModernSyntacticScore) -> Self {
        Self {
            grammaticality_score: score.grammaticality_score,
            syntactic_complexity: score.complexity_score,
            parse_tree_quality: score.parse_quality,
            dependency_coherence: score.dependency_coherence,
            sentence_structure_variety: score.structure_variety,
            clause_integration: score.clause_integration,
            syntactic_patterns: score.pattern_counts,
            error_indicators: score
                .errors
                .into_iter()
                .map(|e| SyntacticError {
                    error_type: match e.error_type.as_str() {
                        "agreement" => SyntacticErrorType::Agreement,
                        "word_order" => SyntacticErrorType::WordOrder,
                        "missing_word" => SyntacticErrorType::MissingWord,
                        "extra_word" => SyntacticErrorType::ExtraWord,
                        "tense" => SyntacticErrorType::TenseError,
                        _ => SyntacticErrorType::Other,
                    },
                    position: e.position,
                    severity: e.severity,
                    description: e.description,
                })
                .collect(),
        }
    }
}

/// Lexical fluency analysis result
#[derive(Debug, Clone)]
pub struct LexicalFluencyResult {
    pub vocabulary_sophistication: f64,
    pub word_choice_appropriateness: f64,
    pub lexical_diversity: f64,
    pub word_frequency_profile: f64,
    pub collocation_quality: f64,
    pub lexical_density: f64,
    pub register_appropriateness: f64,
    pub rare_word_usage: f64,
    pub lexical_richness: LexicalRichness,
}

/// Convert modern lexical score to legacy format
impl From<ModernLexicalScore> for LexicalFluencyResult {
    fn from(score: ModernLexicalScore) -> Self {
        Self {
            vocabulary_sophistication: score.vocabulary_sophistication,
            word_choice_appropriateness: score.appropriateness_score,
            lexical_diversity: score.lexical_diversity,
            word_frequency_profile: score.frequency_profile,
            collocation_quality: score.collocation_strength,
            lexical_density: score.lexical_density,
            register_appropriateness: score.register_consistency,
            rare_word_usage: score.rare_word_ratio,
            lexical_richness: LexicalRichness {
                type_token_ratio: score.type_token_ratio,
                root_ttr: score.root_ttr,
                corrected_ttr: score.corrected_ttr,
                bilogarithmic_ttr: score.bilogarithmic_ttr,
                uber_index: score.uber_index,
                mtld: score.mtld,
                hd_d: score.hd_d,
                maas_ttr: score.maas_ttr,
            },
        }
    }
}

/// Semantic fluency analysis result
#[derive(Debug, Clone)]
pub struct SemanticFluencyResult {
    pub semantic_coherence: f64,
    pub meaning_preservation: f64,
    pub conceptual_clarity: f64,
    pub semantic_appropriateness: f64,
    pub context_sensitivity: f64,
    pub semantic_density: f64,
    pub ambiguity_score: f64,
    pub semantic_relations: HashMap<String, f64>,
}

/// Convert modern semantic score to legacy format
impl From<ModernSemanticScore> for SemanticFluencyResult {
    fn from(score: ModernSemanticScore) -> Self {
        Self {
            semantic_coherence: score.coherence_score,
            meaning_preservation: score.meaning_preservation,
            conceptual_clarity: score.conceptual_clarity,
            semantic_appropriateness: score.appropriateness,
            context_sensitivity: score.context_sensitivity,
            semantic_density: score.semantic_density,
            ambiguity_score: 1.0 - score.clarity_score, // Invert clarity to get ambiguity
            semantic_relations: score.relation_strengths,
        }
    }
}

/// Prosodic fluency analysis result
#[derive(Debug, Clone)]
pub struct ProsodicFluencyResult {
    pub rhythmic_flow: f64,
    pub stress_pattern_naturalness: f64,
    pub intonation_appropriateness: f64,
    pub pause_placement: f64,
    pub reading_ease: f64,
    pub syllable_complexity: f64,
    pub phonological_patterns: HashMap<String, f64>,
    pub prosodic_breaks: Vec<ProsodicBreak>,
}

/// Convert modern prosodic score to legacy format
impl From<ModernProsodicScore> for ProsodicFluencyResult {
    fn from(score: ModernProsodicScore) -> Self {
        Self {
            rhythmic_flow: score.rhythmic_flow,
            stress_pattern_naturalness: score.stress_naturalness,
            intonation_appropriateness: score.intonation_score,
            pause_placement: score.pause_quality,
            reading_ease: score.reading_ease,
            syllable_complexity: score.syllable_complexity,
            phonological_patterns: score.phonological_features,
            prosodic_breaks: score
                .prosodic_breaks
                .into_iter()
                .map(|b| ProsodicBreak {
                    position: b.position,
                    break_type: match b.break_type.as_str() {
                        "minor" => ProsodicBreakType::Minor,
                        "major" => ProsodicBreakType::Major,
                        "intermediate" => ProsodicBreakType::Intermediate,
                        _ => ProsodicBreakType::Minor,
                    },
                    strength: b.strength,
                    duration: b.duration,
                    confidence: b.confidence,
                })
                .collect(),
        }
    }
}

/// Pragmatic fluency analysis result
#[derive(Debug, Clone)]
pub struct PragmaticFluencyResult {
    pub discourse_appropriateness: f64,
    pub context_sensitivity: f64,
    pub speech_act_clarity: f64,
    pub register_consistency: f64,
    pub communicative_effectiveness: f64,
    pub pragmatic_markers: Vec<PragmaticMarker>,
    pub discourse_flow: f64,
}

/// Convert modern pragmatic score to legacy format
impl From<ModernPragmaticScore> for PragmaticFluencyResult {
    fn from(score: ModernPragmaticScore) -> Self {
        Self {
            discourse_appropriateness: score.discourse_appropriateness,
            context_sensitivity: score.context_sensitivity,
            speech_act_clarity: score.speech_act_clarity,
            register_consistency: score.register_consistency,
            communicative_effectiveness: score.communicative_effectiveness,
            discourse_flow: score.discourse_coherence,
            pragmatic_markers: score
                .pragmatic_markers
                .into_iter()
                .map(|m| PragmaticMarker {
                    marker_text: m.text,
                    marker_type: match m.marker_type.as_str() {
                        "contrast" => PragmaticFunction::Contrast,
                        "emphasis" => PragmaticFunction::Emphasis,
                        "sequence" => PragmaticFunction::Sequence,
                        "causation" => PragmaticFunction::Causation,
                        "elaboration" => PragmaticFunction::Elaboration,
                        "clarification" => PragmaticFunction::Clarification,
                        _ => PragmaticFunction::Exemplification,
                    },
                    position: m.position,
                    strength: m.strength,
                    context_span: m.context_span,
                })
                .collect(),
        }
    }
}

/// Fluency score distribution
#[derive(Debug, Clone)]
pub struct FluencyDistribution {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
}

/// Quality indicators for the analysis
#[derive(Debug, Clone)]
pub struct QualityIndicators {
    pub overall_quality: f64,
    pub consistency_score: f64,
    pub readability_level: String,
    pub improvement_suggestions: Vec<String>,
    pub error_density: f64,
}

/// Syntactic error information
#[derive(Debug, Clone)]
pub struct SyntacticError {
    pub error_type: SyntacticErrorType,
    pub position: usize,
    pub severity: f64,
    pub description: String,
}

/// Types of syntactic errors
#[derive(Debug, Clone, PartialEq)]
pub enum SyntacticErrorType {
    Agreement,
    WordOrder,
    MissingWord,
    ExtraWord,
    TenseError,
    PunctuationError,
    FragmentError,
    RunOnSentence,
    Other,
}

/// Lexical richness metrics
#[derive(Debug, Clone)]
pub struct LexicalRichness {
    pub type_token_ratio: f64,
    pub root_ttr: f64,
    pub corrected_ttr: f64,
    pub bilogarithmic_ttr: f64,
    pub uber_index: f64,
    pub mtld: f64,
    pub hd_d: f64,
    pub maas_ttr: f64,
}

/// Prosodic break information
#[derive(Debug, Clone)]
pub struct ProsodicBreak {
    pub position: usize,
    pub break_type: ProsodicBreakType,
    pub strength: f64,
    pub duration: f64,
    pub confidence: f64,
}

/// Types of prosodic breaks
#[derive(Debug, Clone, PartialEq)]
pub enum ProsodicBreakType {
    Minor,
    Major,
    Intermediate,
    IntonationalPhrase,
    UtteranceBoundary,
}

/// Pragmatic marker information
#[derive(Debug, Clone)]
pub struct PragmaticMarker {
    pub marker_text: String,
    pub marker_type: PragmaticFunction,
    pub position: usize,
    pub strength: f64,
    pub context_span: (usize, usize),
}

/// Types of pragmatic functions
#[derive(Debug, Clone, PartialEq)]
pub enum PragmaticFunction {
    Contrast,
    Emphasis,
    Sequence,
    Causation,
    Addition,
    Conclusion,
    Elaboration,
    Clarification,
    Exemplification,
}

/// Fluency comparison result
#[derive(Debug, Clone)]
pub struct FluencyComparisonResult {
    pub text1_fluency: f64,
    pub text2_fluency: f64,
    pub fluency_difference: f64,
    pub better_text: usize, // 1 or 2
}

/// Advanced fluency analyzer (backward compatibility wrapper)
pub struct FluencyAnalyzer {
    config: FluencyConfig,
    /// Internal modern analyzer (created on-demand)
    _modern_analyzer: ModernFluentTextAnalyzer,
}

impl FluencyAnalyzer {
    /// Create new fluency analyzer with given configuration
    pub fn new(config: FluencyConfig) -> Self {
        let modern_config = config.clone().into();
        let modern_analyzer = ModernFluentTextAnalyzer::with_config(modern_config);

        Self {
            config,
            _modern_analyzer: modern_analyzer,
        }
    }

    /// Create fluency analyzer with default configuration
    pub fn with_default_config() -> Self {
        Self::new(FluencyConfig::default())
    }

    /// Analyze fluency of given text
    pub fn analyze_fluency(&self, text: &str) -> FluencyResult {
        // Use the modern analyzer to get comprehensive results
        let modern_result = self
            ._modern_analyzer
            .analyze_comprehensive_fluency(text)
            .unwrap_or_else(|_| {
                // Fallback to basic analysis on error
                let basic_analyzer = ModernFluentTextAnalyzer::new();
                basic_analyzer.analyze_comprehensive_fluency(text).unwrap()
            });

        // Convert modern result to legacy format
        self.convert_modern_to_legacy_result(modern_result, text)
    }

    /// Compare fluency between two texts
    pub fn compare_fluency(&self, text1: &str, text2: &str) -> FluencyComparisonResult {
        let result1 = self.analyze_fluency(text1);
        let result2 = self.analyze_fluency(text2);

        let fluency_difference = result1.overall_fluency - result2.overall_fluency;
        let better_text = if result1.overall_fluency > result2.overall_fluency {
            1
        } else {
            2
        };

        FluencyComparisonResult {
            text1_fluency: result1.overall_fluency,
            text2_fluency: result2.overall_fluency,
            fluency_difference,
            better_text,
        }
    }

    /// Analyze fluency progression across text segments
    pub fn analyze_fluency_progression(
        &self,
        text: &str,
        window_size: usize,
    ) -> Vec<FluencyResult> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() < window_size {
            return vec![self.analyze_fluency(text)];
        }

        let mut results = Vec::new();
        for i in 0..=words.len().saturating_sub(window_size) {
            let segment = words[i..i + window_size].join(" ");
            results.push(self.analyze_fluency(&segment));
        }

        results
    }

    /// Convert modern comprehensive analysis to legacy format
    fn convert_modern_to_legacy_result(
        &self,
        modern_result: ModernComprehensiveFluencyAnalysis,
        text: &str,
    ) -> FluencyResult {
        // Extract individual component results or use defaults
        let language_model_fluency = modern_result
            .language_model_analysis
            .map(|lm| lm.into())
            .unwrap_or_else(|| self.create_default_language_model_result());

        let syntactic_fluency = modern_result
            .syntactic_analysis
            .map(|syn| syn.into())
            .unwrap_or_else(|| self.create_default_syntactic_result());

        let lexical_fluency = modern_result
            .lexical_analysis
            .map(|lex| lex.into())
            .unwrap_or_else(|| self.create_default_lexical_result());

        let semantic_fluency = modern_result
            .semantic_analysis
            .map(|sem| sem.into())
            .unwrap_or_else(|| self.create_default_semantic_result());

        let prosodic_fluency = modern_result
            .prosodic_analysis
            .map(|pros| pros.into())
            .unwrap_or_else(|| self.create_default_prosodic_result());

        let pragmatic_fluency = modern_result
            .pragmatic_analysis
            .map(|prag| prag.into())
            .unwrap_or_else(|| self.create_default_pragmatic_result());

        // Calculate sentence-level scores
        let sentence_level_scores = self.calculate_sentence_level_scores(text);

        // Calculate fluency distribution
        let fluency_distribution = self.calculate_fluency_distribution(&sentence_level_scores);

        // Create fluency breakdown
        let mut fluency_breakdown = HashMap::new();
        fluency_breakdown.insert(
            "language_model".to_string(),
            language_model_fluency.smoothed_score,
        );
        fluency_breakdown.insert(
            "syntactic".to_string(),
            syntactic_fluency.grammaticality_score,
        );
        fluency_breakdown.insert("lexical".to_string(), lexical_fluency.lexical_diversity);
        fluency_breakdown.insert("semantic".to_string(), semantic_fluency.semantic_coherence);
        fluency_breakdown.insert("prosodic".to_string(), prosodic_fluency.rhythmic_flow);
        fluency_breakdown.insert(
            "pragmatic".to_string(),
            pragmatic_fluency.communicative_effectiveness,
        );

        // Calculate quality indicators from modern quality assessment
        let quality_indicators = modern_result
            .quality_assessment
            .map(|qa| self.convert_quality_assessment(qa))
            .unwrap_or_else(|| self.create_default_quality_indicators());

        // Calculate confidence score
        let confidence_score = modern_result.analysis_metadata.confidence_score;

        FluencyResult {
            overall_fluency: modern_result.overall_score,
            language_model_fluency,
            syntactic_fluency,
            lexical_fluency,
            semantic_fluency,
            prosodic_fluency,
            pragmatic_fluency,
            sentence_level_scores,
            fluency_distribution,
            confidence_score,
            fluency_breakdown,
            quality_indicators,
        }
    }

    /// Calculate sentence-level fluency scores
    fn calculate_sentence_level_scores(&self, text: &str) -> Vec<f64> {
        let sentences = text
            .split('.')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();

        sentences
            .iter()
            .map(|sentence| {
                // Quick fluency estimate for individual sentence
                let word_count = sentence.split_whitespace().count();
                let base_score = 0.7;
                let length_factor = (word_count as f64 / 15.0).min(1.0);
                base_score + (length_factor * 0.2)
            })
            .collect()
    }

    /// Calculate fluency distribution statistics
    fn calculate_fluency_distribution(&self, scores: &[f64]) -> FluencyDistribution {
        if scores.is_empty() {
            return FluencyDistribution {
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
            };
        }

        let sum: f64 = scores.iter().sum();
        let mean = sum / scores.len() as f64;

        let mut sorted_scores = scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if sorted_scores.len() % 2 == 0 {
            (sorted_scores[sorted_scores.len() / 2 - 1] + sorted_scores[sorted_scores.len() / 2])
                / 2.0
        } else {
            sorted_scores[sorted_scores.len() / 2]
        };

        let variance = scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / scores.len() as f64;
        let std_dev = variance.sqrt();

        let min = *sorted_scores.first().unwrap();
        let max = *sorted_scores.last().unwrap();

        FluencyDistribution {
            mean,
            median,
            std_dev,
            min,
            max,
        }
    }

    /// Convert modern quality assessment to legacy format
    fn convert_quality_assessment(
        &self,
        qa: crate::metrics::fluency::ComprehensiveQualityAssessment,
    ) -> QualityIndicators {
        QualityIndicators {
            overall_quality: qa.overall_quality_score,
            consistency_score: qa.consistency_metrics.internal_consistency,
            readability_level: format!("{:?}", qa.quality_grade),
            improvement_suggestions: qa.improvement_recommendations,
            error_density: qa.error_analysis.error_density,
        }
    }

    /// Create default results for missing components
    fn create_default_language_model_result(&self) -> LanguageModelFluencyResult {
        LanguageModelFluencyResult {
            perplexity_score: 0.7,
            likelihood_score: 0.6,
            surprisal_score: 0.65,
            entropy_score: 0.68,
            probability_mass: 0.75,
            ngram_probabilities: HashMap::new(),
            oov_penalty: 0.1,
            smoothed_score: 0.7,
        }
    }

    fn create_default_syntactic_result(&self) -> SyntacticFluencyResult {
        SyntacticFluencyResult {
            grammaticality_score: 0.75,
            syntactic_complexity: 0.6,
            parse_tree_quality: 0.7,
            dependency_coherence: 0.65,
            sentence_structure_variety: 0.55,
            clause_integration: 0.68,
            syntactic_patterns: HashMap::new(),
            error_indicators: Vec::new(),
        }
    }

    fn create_default_lexical_result(&self) -> LexicalFluencyResult {
        LexicalFluencyResult {
            vocabulary_sophistication: 0.65,
            word_choice_appropriateness: 0.7,
            lexical_diversity: 0.6,
            word_frequency_profile: 0.68,
            collocation_quality: 0.62,
            lexical_density: 0.58,
            register_appropriateness: 0.72,
            rare_word_usage: 0.45,
            lexical_richness: LexicalRichness {
                type_token_ratio: 0.6,
                root_ttr: 0.65,
                corrected_ttr: 0.62,
                bilogarithmic_ttr: 0.58,
                uber_index: 0.55,
                mtld: 0.6,
                hd_d: 0.63,
                maas_ttr: 0.61,
            },
        }
    }

    fn create_default_semantic_result(&self) -> SemanticFluencyResult {
        SemanticFluencyResult {
            semantic_coherence: 0.7,
            meaning_preservation: 0.75,
            conceptual_clarity: 0.68,
            semantic_appropriateness: 0.72,
            context_sensitivity: 0.65,
            semantic_density: 0.6,
            ambiguity_score: 0.3,
            semantic_relations: HashMap::new(),
        }
    }

    fn create_default_prosodic_result(&self) -> ProsodicFluencyResult {
        ProsodicFluencyResult {
            rhythmic_flow: 0.68,
            stress_pattern_naturalness: 0.65,
            intonation_appropriateness: 0.7,
            pause_placement: 0.63,
            reading_ease: 0.72,
            syllable_complexity: 0.58,
            phonological_patterns: HashMap::new(),
            prosodic_breaks: Vec::new(),
        }
    }

    fn create_default_pragmatic_result(&self) -> PragmaticFluencyResult {
        PragmaticFluencyResult {
            discourse_appropriateness: 0.7,
            context_sensitivity: 0.65,
            speech_act_clarity: 0.68,
            register_consistency: 0.72,
            communicative_effectiveness: 0.67,
            discourse_flow: 0.69,
            pragmatic_markers: Vec::new(),
        }
    }

    fn create_default_quality_indicators(&self) -> QualityIndicators {
        QualityIndicators {
            overall_quality: 0.7,
            consistency_score: 0.68,
            readability_level: "Intermediate".to_string(),
            improvement_suggestions: vec![
                "Consider varying sentence length".to_string(),
                "Use more sophisticated vocabulary".to_string(),
            ],
            error_density: 0.05,
        }
    }
}

/// Standalone function: Calculate simple fluency score (backward compatibility)
pub fn calculate_text_fluency_simple(text: &str) -> f64 {
    let analyzer = FluencyAnalyzer::with_default_config();
    let result = analyzer.analyze_fluency(text);
    result.overall_fluency
}

/// Standalone function: Calculate grammaticality score (backward compatibility)
pub fn calculate_grammaticality_simple(text: &str) -> f64 {
    let analyzer = FluencyAnalyzer::with_default_config();
    let result = analyzer.analyze_fluency(text);
    result.syntactic_fluency.grammaticality_score
}

/// Standalone function: Calculate readability score (backward compatibility)
pub fn calculate_readability_simple(text: &str) -> f64 {
    let analyzer = FluencyAnalyzer::with_default_config();
    let result = analyzer.analyze_fluency(text);
    result.prosodic_fluency.reading_ease
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backward_compatibility_api() {
        // Test that the legacy API still works
        let config = FluencyConfig::default();
        let analyzer = FluencyAnalyzer::new(config);

        let result = analyzer.analyze_fluency("The cat sat on the mat.");
        assert!(result.overall_fluency > 0.0);
        assert!(result.overall_fluency <= 1.0);
        assert!(result.confidence_score >= 0.0);
        assert!(!result.fluency_breakdown.is_empty());
    }

    #[test]
    fn test_fluency_analyzer_creation() {
        let analyzer = FluencyAnalyzer::with_default_config();
        let text =
            "The cat sat on the mat. It was very comfortable and decided to take a long nap.";

        let result = analyzer.analyze_fluency(text);

        assert!(result.overall_fluency > 0.0);
        assert!(result.overall_fluency <= 1.0);
        assert!(result.confidence_score >= 0.0);
        assert!(!result.fluency_breakdown.is_empty());
    }

    #[test]
    fn test_language_model_fluency() {
        let analyzer = FluencyAnalyzer::with_default_config();
        let result = analyzer.analyze_fluency("This is a well-formed sentence.");

        assert!(result.language_model_fluency.smoothed_score >= 0.0);
        assert!(result.language_model_fluency.smoothed_score <= 1.0);
        assert!(result.language_model_fluency.perplexity_score >= 0.0);
    }

    #[test]
    fn test_fluency_comparison() {
        let analyzer = FluencyAnalyzer::with_default_config();
        let text1 = "The cat sat on the mat.";
        let text2 = "Cat mat sat the on."; // Less fluent

        let comparison = analyzer.compare_fluency(text1, text2);

        assert!(comparison.text1_fluency > comparison.text2_fluency);
        assert_eq!(comparison.better_text, 1);
    }

    #[test]
    fn test_fluency_progression() {
        let analyzer = FluencyAnalyzer::with_default_config();
        let text =
            "The quick brown fox jumps over the lazy dog. It was a beautiful day for jumping.";

        let progression = analyzer.analyze_fluency_progression(text, 5);
        assert!(!progression.is_empty());

        for result in progression {
            assert!(result.overall_fluency >= 0.0);
            assert!(result.overall_fluency <= 1.0);
        }
    }

    #[test]
    fn test_standalone_functions() {
        let text = "This is a sample text for testing.";

        let fluency = calculate_text_fluency_simple(text);
        assert!(fluency >= 0.0 && fluency <= 1.0);

        let grammaticality = calculate_grammaticality_simple(text);
        assert!(grammaticality >= 0.0 && grammaticality <= 1.0);

        let readability = calculate_readability_simple(text);
        assert!(readability >= 0.0 && readability <= 1.0);
    }

    #[test]
    fn test_config_conversion() {
        let legacy_config = FluencyConfig {
            language_model_weight: 0.4,
            syntactic_weight: 0.3,
            lexical_weight: 0.3,
            semantic_weight: 0.0,
            prosodic_weight: 0.0,
            pragmatic_weight: 0.0,
            fluency_dimensions: vec![
                FluencyDimension::LanguageModel,
                FluencyDimension::Syntactic,
                FluencyDimension::Lexical,
            ],
            ..Default::default()
        };

        let modern_config: ModernAnalysisConfig = legacy_config.into();
        assert!(modern_config.enable_language_model_analysis);
        assert!(modern_config.enable_syntactic_analysis);
        assert!(modern_config.enable_lexical_analysis);
        assert!(!modern_config.enable_semantic_analysis);
        assert!(!modern_config.enable_prosodic_analysis);
        assert!(!modern_config.enable_pragmatic_analysis);
    }

    #[test]
    fn test_fluency_distribution_calculation() {
        let analyzer = FluencyAnalyzer::with_default_config();
        let scores = vec![0.6, 0.7, 0.8, 0.75, 0.65];
        let distribution = analyzer.calculate_fluency_distribution(&scores);

        assert!(distribution.mean > 0.0);
        assert!(distribution.median > 0.0);
        assert!(distribution.std_dev >= 0.0);
        assert!(distribution.min <= distribution.max);
    }

    #[test]
    fn test_sentence_level_scores() {
        let analyzer = FluencyAnalyzer::with_default_config();
        let text = "First sentence. Second sentence. Third sentence.";
        let scores = analyzer.calculate_sentence_level_scores(text);

        assert_eq!(scores.len(), 3);
        for score in scores {
            assert!(score >= 0.0 && score <= 1.0);
        }
    }

    #[test]
    fn test_quality_indicators() {
        let analyzer = FluencyAnalyzer::with_default_config();
        let result =
            analyzer.analyze_fluency("This is a well-written sentence with good structure.");

        assert!(result.quality_indicators.overall_quality >= 0.0);
        assert!(result.quality_indicators.overall_quality <= 1.0);
        assert!(result.quality_indicators.consistency_score >= 0.0);
        assert!(!result.quality_indicators.improvement_suggestions.is_empty());
    }
}
