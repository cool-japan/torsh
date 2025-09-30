//! Comprehensive fluency analysis framework for text quality assessment.
//!
//! This module provides a modular, extensible system for analyzing text fluency across
//! multiple dimensions including language model scoring, syntactic analysis, lexical
//! diversity, semantic coherence, prosodic patterns, pragmatic appropriateness,
//! statistical characteristics, and overall quality assessment.
//!
//! # Architecture
//!
//! The fluency analysis system is built with a modular architecture where each dimension
//! of analysis is handled by a specialized module:
//!
//! - **Language Model Analysis**: Perplexity, surprisal, n-gram analysis
//! - **Syntactic Analysis**: Grammar, complexity, parse quality
//! - **Lexical Analysis**: Vocabulary sophistication, diversity, richness
//! - **Semantic Analysis**: Coherence, meaning preservation, conceptual clarity
//! - **Prosodic Analysis**: Rhythm, stress patterns, intonation
//! - **Pragmatic Analysis**: Context appropriateness, communicative effectiveness
//! - **Statistical Analysis**: Comprehensive statistical calculations and measures
//! - **Quality Analysis**: Integrated quality assessment and improvement recommendations
//!
//! # Usage
//!
//! ## Basic Analysis
//!
//! ```rust
//! use torsh_text::metrics::fluency::{FluentTextAnalyzer, AnalysisConfig};
//!
//! let analyzer = FluentTextAnalyzer::new();
//! let text = "This is a well-written example of fluent text with good structure.";
//! let analysis = analyzer.analyze_comprehensive_fluency(text).unwrap();
//!
//! println!("Overall fluency score: {:.3}", analysis.overall_score);
//! println!("Quality grade: {:?}", analysis.quality_assessment.quality_grade);
//! ```
//!
//! ## Selective Analysis
//!
//! ```rust
//! use torsh_text::metrics::fluency::{FluentTextAnalyzer, AnalysisConfig};
//!
//! let config = AnalysisConfig::builder()
//!     .enable_language_model_analysis(true)
//!     .enable_semantic_analysis(true)
//!     .enable_quality_analysis(true)
//!     .disable_prosodic_analysis() // Skip prosodic analysis for speed
//!     .build();
//!
//! let analyzer = FluentTextAnalyzer::with_config(config);
//! let analysis = analyzer.analyze_text("Sample text here").unwrap();
//! ```
//!
//! ## Individual Module Usage
//!
//! ```rust
//! use torsh_text::metrics::fluency::semantic::SemanticAnalyzer;
//!
//! let semantic_analyzer = SemanticAnalyzer::new();
//! let semantic_score = semantic_analyzer.analyze_semantic_fluency("Sample text").unwrap();
//! println!("Semantic coherence: {:.3}", semantic_score.coherence_score);
//! ```

pub mod language_model;
pub mod lexical;
pub mod pragmatic;
pub mod prosodic;
pub mod quality;
pub mod semantic;
pub mod statistical;
pub mod syntactic;

// Re-export key types for convenience
pub use language_model::{LanguageModelAnalyzer, LanguageModelScore};
pub use lexical::{LexicalAnalyzer, LexicalScore};
pub use pragmatic::{PragmaticAnalyzer, PragmaticScore};
pub use prosodic::{ProsodicAnalyzer, ProsodicScore};
pub use quality::{ComprehensiveQualityAssessment, QualityAnalyzer, QualityGrade};
pub use semantic::{SemanticAnalyzer, SemanticScore};
pub use statistical::{DescriptiveStatistics, StatisticalAnalyzer};
pub use syntactic::{SyntacticAnalyzer, SyntacticScore};

use scirs2_core::ndarray::Array1;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;

/// Configuration for fluency analysis, allowing selective enabling/disabling of analysis modules.
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    pub enable_language_model_analysis: bool,
    pub enable_syntactic_analysis: bool,
    pub enable_lexical_analysis: bool,
    pub enable_semantic_analysis: bool,
    pub enable_prosodic_analysis: bool,
    pub enable_pragmatic_analysis: bool,
    pub enable_statistical_analysis: bool,
    pub enable_quality_analysis: bool,
    pub quality_weights: Option<quality::QualityWeights>,
    pub quality_thresholds: Option<quality::QualityThresholds>,
    pub analysis_depth: AnalysisDepth,
    pub performance_mode: PerformanceMode,
}

/// Builder for AnalysisConfig to provide a fluent configuration API.
#[derive(Debug)]
pub struct AnalysisConfigBuilder {
    config: AnalysisConfig,
}

/// Depth of analysis to perform.
#[derive(Debug, Clone, PartialEq)]
pub enum AnalysisDepth {
    /// Fast analysis with basic metrics
    Basic,
    /// Standard analysis with most metrics (default)
    Standard,
    /// Comprehensive analysis with all metrics
    Comprehensive,
    /// Deep analysis with advanced statistical measures
    Deep,
}

/// Performance optimization mode.
#[derive(Debug, Clone, PartialEq)]
pub enum PerformanceMode {
    /// Optimize for accuracy (default)
    Accuracy,
    /// Balance speed and accuracy
    Balanced,
    /// Optimize for speed
    Speed,
}

/// Comprehensive fluency analysis result containing scores from all enabled dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct ComprehensiveFluencyAnalysis {
    pub overall_score: f64,
    pub dimensional_scores: DimensionalScores,
    pub language_model_analysis: Option<LanguageModelScore>,
    pub syntactic_analysis: Option<SyntacticScore>,
    pub lexical_analysis: Option<LexicalScore>,
    pub semantic_analysis: Option<SemanticScore>,
    pub prosodic_analysis: Option<ProsodicScore>,
    pub pragmatic_analysis: Option<PragmaticScore>,
    pub statistical_summary: Option<DescriptiveStatistics>,
    pub quality_assessment: Option<ComprehensiveQualityAssessment>,
    pub analysis_metadata: AnalysisMetadata,
}

/// Dimensional scores organized by category.
#[derive(Debug, Clone, PartialEq)]
pub struct DimensionalScores {
    pub language_model_score: f64,
    pub syntactic_score: f64,
    pub lexical_score: f64,
    pub semantic_score: f64,
    pub prosodic_score: f64,
    pub pragmatic_score: f64,
    pub statistical_score: f64,
    pub quality_score: f64,
}

/// Metadata about the analysis performed.
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisMetadata {
    pub text_length: usize,
    pub word_count: usize,
    pub sentence_count: usize,
    pub analysis_duration_ms: u64,
    pub enabled_modules: Vec<String>,
    pub analysis_config: String,
    pub warnings: Vec<String>,
}

/// Main analyzer that coordinates all fluency analysis modules.
#[derive(Debug, Clone)]
pub struct FluentTextAnalyzer {
    config: AnalysisConfig,
    language_model_analyzer: LanguageModelAnalyzer,
    syntactic_analyzer: SyntacticAnalyzer,
    lexical_analyzer: LexicalAnalyzer,
    semantic_analyzer: SemanticAnalyzer,
    prosodic_analyzer: ProsodicAnalyzer,
    pragmatic_analyzer: PragmaticAnalyzer,
    statistical_analyzer: StatisticalAnalyzer,
    quality_analyzer: QualityAnalyzer,
}

/// Errors that can occur during fluency analysis.
#[derive(Debug)]
pub enum FluencyAnalysisError {
    LanguageModelError(language_model::LanguageModelError),
    SyntacticError(syntactic::SyntacticError),
    LexicalError(lexical::LexicalError),
    SemanticError(semantic::SemanticError),
    ProsodicError(prosodic::ProsodicError),
    PragmaticError(pragmatic::PragmaticError),
    StatisticalError(statistical::StatisticalError),
    QualityError(quality::QualityError),
    ConfigurationError(String),
    TextProcessingError(String),
    AnalysisError(String),
}

impl fmt::Display for FluencyAnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FluencyAnalysisError::LanguageModelError(e) => write!(f, "Language model error: {}", e),
            FluencyAnalysisError::SyntacticError(e) => write!(f, "Syntactic analysis error: {}", e),
            FluencyAnalysisError::LexicalError(e) => write!(f, "Lexical analysis error: {}", e),
            FluencyAnalysisError::SemanticError(e) => write!(f, "Semantic analysis error: {}", e),
            FluencyAnalysisError::ProsodicError(e) => write!(f, "Prosodic analysis error: {}", e),
            FluencyAnalysisError::PragmaticError(e) => write!(f, "Pragmatic analysis error: {}", e),
            FluencyAnalysisError::StatisticalError(e) => {
                write!(f, "Statistical analysis error: {}", e)
            }
            FluencyAnalysisError::QualityError(e) => write!(f, "Quality analysis error: {}", e),
            FluencyAnalysisError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {}", msg)
            }
            FluencyAnalysisError::TextProcessingError(msg) => {
                write!(f, "Text processing error: {}", msg)
            }
            FluencyAnalysisError::AnalysisError(msg) => write!(f, "Analysis error: {}", msg),
        }
    }
}

impl Error for FluencyAnalysisError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            FluencyAnalysisError::LanguageModelError(e) => Some(e),
            FluencyAnalysisError::SyntacticError(e) => Some(e),
            FluencyAnalysisError::LexicalError(e) => Some(e),
            FluencyAnalysisError::SemanticError(e) => Some(e),
            FluencyAnalysisError::ProsodicError(e) => Some(e),
            FluencyAnalysisError::PragmaticError(e) => Some(e),
            FluencyAnalysisError::StatisticalError(e) => Some(e),
            FluencyAnalysisError::QualityError(e) => Some(e),
            _ => None,
        }
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            enable_language_model_analysis: true,
            enable_syntactic_analysis: true,
            enable_lexical_analysis: true,
            enable_semantic_analysis: true,
            enable_prosodic_analysis: true,
            enable_pragmatic_analysis: true,
            enable_statistical_analysis: true,
            enable_quality_analysis: true,
            quality_weights: None,
            quality_thresholds: None,
            analysis_depth: AnalysisDepth::Standard,
            performance_mode: PerformanceMode::Accuracy,
        }
    }
}

impl AnalysisConfig {
    /// Create a new builder for configuring analysis options.
    pub fn builder() -> AnalysisConfigBuilder {
        AnalysisConfigBuilder {
            config: Self::default(),
        }
    }

    /// Create a fast configuration for basic analysis.
    pub fn fast() -> Self {
        Self {
            enable_language_model_analysis: true,
            enable_syntactic_analysis: true,
            enable_lexical_analysis: true,
            enable_semantic_analysis: true,
            enable_prosodic_analysis: false,    // Skip for speed
            enable_pragmatic_analysis: false,   // Skip for speed
            enable_statistical_analysis: false, // Skip for speed
            enable_quality_analysis: true,
            quality_weights: None,
            quality_thresholds: None,
            analysis_depth: AnalysisDepth::Basic,
            performance_mode: PerformanceMode::Speed,
        }
    }

    /// Create a comprehensive configuration for detailed analysis.
    pub fn comprehensive() -> Self {
        Self {
            enable_language_model_analysis: true,
            enable_syntactic_analysis: true,
            enable_lexical_analysis: true,
            enable_semantic_analysis: true,
            enable_prosodic_analysis: true,
            enable_pragmatic_analysis: true,
            enable_statistical_analysis: true,
            enable_quality_analysis: true,
            quality_weights: None,
            quality_thresholds: None,
            analysis_depth: AnalysisDepth::Comprehensive,
            performance_mode: PerformanceMode::Accuracy,
        }
    }
}

impl AnalysisConfigBuilder {
    /// Enable language model analysis.
    pub fn enable_language_model_analysis(mut self, enable: bool) -> Self {
        self.config.enable_language_model_analysis = enable;
        self
    }

    /// Enable syntactic analysis.
    pub fn enable_syntactic_analysis(mut self, enable: bool) -> Self {
        self.config.enable_syntactic_analysis = enable;
        self
    }

    /// Enable lexical analysis.
    pub fn enable_lexical_analysis(mut self, enable: bool) -> Self {
        self.config.enable_lexical_analysis = enable;
        self
    }

    /// Enable semantic analysis.
    pub fn enable_semantic_analysis(mut self, enable: bool) -> Self {
        self.config.enable_semantic_analysis = enable;
        self
    }

    /// Enable prosodic analysis.
    pub fn enable_prosodic_analysis(mut self, enable: bool) -> Self {
        self.config.enable_prosodic_analysis = enable;
        self
    }

    /// Disable prosodic analysis for faster processing.
    pub fn disable_prosodic_analysis(mut self) -> Self {
        self.config.enable_prosodic_analysis = false;
        self
    }

    /// Enable pragmatic analysis.
    pub fn enable_pragmatic_analysis(mut self, enable: bool) -> Self {
        self.config.enable_pragmatic_analysis = enable;
        self
    }

    /// Disable pragmatic analysis for faster processing.
    pub fn disable_pragmatic_analysis(mut self) -> Self {
        self.config.enable_pragmatic_analysis = false;
        self
    }

    /// Enable statistical analysis.
    pub fn enable_statistical_analysis(mut self, enable: bool) -> Self {
        self.config.enable_statistical_analysis = enable;
        self
    }

    /// Enable quality analysis.
    pub fn enable_quality_analysis(mut self, enable: bool) -> Self {
        self.config.enable_quality_analysis = enable;
        self
    }

    /// Set custom quality weights.
    pub fn with_quality_weights(mut self, weights: quality::QualityWeights) -> Self {
        self.config.quality_weights = Some(weights);
        self
    }

    /// Set custom quality thresholds.
    pub fn with_quality_thresholds(mut self, thresholds: quality::QualityThresholds) -> Self {
        self.config.quality_thresholds = Some(thresholds);
        self
    }

    /// Set analysis depth.
    pub fn with_analysis_depth(mut self, depth: AnalysisDepth) -> Self {
        self.config.analysis_depth = depth;
        self
    }

    /// Set performance mode.
    pub fn with_performance_mode(mut self, mode: PerformanceMode) -> Self {
        self.config.performance_mode = mode;
        self
    }

    /// Build the final configuration.
    pub fn build(self) -> AnalysisConfig {
        self.config
    }
}

impl Default for FluentTextAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl FluentTextAnalyzer {
    /// Create a new fluency analyzer with default configuration.
    pub fn new() -> Self {
        Self::with_config(AnalysisConfig::default())
    }

    /// Create a new fluency analyzer with custom configuration.
    pub fn with_config(config: AnalysisConfig) -> Self {
        let mut quality_analyzer = QualityAnalyzer::new();

        if let Some(weights) = &config.quality_weights {
            quality_analyzer = quality_analyzer.with_weights(weights.clone());
        }

        if let Some(thresholds) = &config.quality_thresholds {
            quality_analyzer = quality_analyzer.with_thresholds(thresholds.clone());
        }

        Self {
            config,
            language_model_analyzer: LanguageModelAnalyzer::new(),
            syntactic_analyzer: SyntacticAnalyzer::new(),
            lexical_analyzer: LexicalAnalyzer::new(),
            semantic_analyzer: SemanticAnalyzer::new(),
            prosodic_analyzer: ProsodicAnalyzer::new(),
            pragmatic_analyzer: PragmaticAnalyzer::new(),
            statistical_analyzer: StatisticalAnalyzer::new(),
            quality_analyzer,
        }
    }

    /// Create a fast analyzer optimized for speed over comprehensive analysis.
    pub fn fast() -> Self {
        Self::with_config(AnalysisConfig::fast())
    }

    /// Create a comprehensive analyzer for detailed analysis.
    pub fn comprehensive() -> Self {
        Self::with_config(AnalysisConfig::comprehensive())
    }

    /// Perform comprehensive fluency analysis on the given text.
    pub fn analyze_comprehensive_fluency(
        &self,
        text: &str,
    ) -> Result<ComprehensiveFluencyAnalysis, FluencyAnalysisError> {
        let start_time = std::time::Instant::now();

        // Validate input
        if text.trim().is_empty() {
            return Err(FluencyAnalysisError::TextProcessingError(
                "Empty text provided".to_string(),
            ));
        }

        // Perform individual analyses based on configuration
        let language_model_analysis = if self.config.enable_language_model_analysis {
            Some(
                self.language_model_analyzer
                    .analyze_language_model_fluency(text)
                    .map_err(FluencyAnalysisError::LanguageModelError)?,
            )
        } else {
            None
        };

        let syntactic_analysis = if self.config.enable_syntactic_analysis {
            Some(
                self.syntactic_analyzer
                    .analyze_syntactic_fluency(text)
                    .map_err(FluencyAnalysisError::SyntacticError)?,
            )
        } else {
            None
        };

        let lexical_analysis = if self.config.enable_lexical_analysis {
            Some(
                self.lexical_analyzer
                    .analyze_lexical_fluency(text)
                    .map_err(FluencyAnalysisError::LexicalError)?,
            )
        } else {
            None
        };

        let semantic_analysis = if self.config.enable_semantic_analysis {
            Some(
                self.semantic_analyzer
                    .analyze_semantic_fluency(text)
                    .map_err(FluencyAnalysisError::SemanticError)?,
            )
        } else {
            None
        };

        let prosodic_analysis = if self.config.enable_prosodic_analysis {
            Some(
                self.prosodic_analyzer
                    .analyze_prosodic_fluency(text)
                    .map_err(FluencyAnalysisError::ProsodicError)?,
            )
        } else {
            None
        };

        let pragmatic_analysis = if self.config.enable_pragmatic_analysis {
            Some(
                self.pragmatic_analyzer
                    .analyze_pragmatic_fluency(text)
                    .map_err(FluencyAnalysisError::PragmaticError)?,
            )
        } else {
            None
        };

        // Calculate statistical summary
        let statistical_summary = if self.config.enable_statistical_analysis {
            let scores: Vec<f64> = vec![
                language_model_analysis
                    .as_ref()
                    .map(|a| a.overall_score)
                    .unwrap_or(0.0),
                syntactic_analysis
                    .as_ref()
                    .map(|a| a.overall_score)
                    .unwrap_or(0.0),
                lexical_analysis
                    .as_ref()
                    .map(|a| a.overall_score)
                    .unwrap_or(0.0),
                semantic_analysis
                    .as_ref()
                    .map(|a| a.overall_score)
                    .unwrap_or(0.0),
                prosodic_analysis
                    .as_ref()
                    .map(|a| a.overall_score)
                    .unwrap_or(0.0),
                pragmatic_analysis
                    .as_ref()
                    .map(|a| a.overall_score)
                    .unwrap_or(0.0),
            ];

            let score_array = Array1::from_vec(scores);
            Some(
                self.statistical_analyzer
                    .descriptive_statistics(&score_array)
                    .map_err(FluencyAnalysisError::StatisticalError)?,
            )
        } else {
            None
        };

        // Perform quality analysis
        let quality_assessment = if self.config.enable_quality_analysis {
            Some(
                self.quality_analyzer
                    .analyze_comprehensive_quality(
                        text,
                        language_model_analysis.clone(),
                        syntactic_analysis.clone(),
                        lexical_analysis.clone(),
                        semantic_analysis.clone(),
                        prosodic_analysis.clone(),
                        pragmatic_analysis.clone(),
                    )
                    .map_err(FluencyAnalysisError::QualityError)?,
            )
        } else {
            None
        };

        // Calculate dimensional scores
        let dimensional_scores = self.calculate_dimensional_scores(
            &language_model_analysis,
            &syntactic_analysis,
            &lexical_analysis,
            &semantic_analysis,
            &prosodic_analysis,
            &pragmatic_analysis,
            &quality_assessment,
        );

        // Calculate overall score
        let overall_score = self.calculate_overall_score(&dimensional_scores);

        // Prepare analysis metadata
        let analysis_duration = start_time.elapsed().as_millis() as u64;
        let analysis_metadata = self.create_analysis_metadata(text, analysis_duration);

        Ok(ComprehensiveFluencyAnalysis {
            overall_score,
            dimensional_scores,
            language_model_analysis,
            syntactic_analysis,
            lexical_analysis,
            semantic_analysis,
            prosodic_analysis,
            pragmatic_analysis,
            statistical_summary,
            quality_assessment,
            analysis_metadata,
        })
    }

    /// Analyze only specific dimensions of fluency.
    pub fn analyze_selective_fluency(
        &self,
        text: &str,
        dimensions: &[&str],
    ) -> Result<ComprehensiveFluencyAnalysis, FluencyAnalysisError> {
        // Create a custom config based on requested dimensions
        let mut custom_config = self.config.clone();

        // Disable all analyses first
        custom_config.enable_language_model_analysis = false;
        custom_config.enable_syntactic_analysis = false;
        custom_config.enable_lexical_analysis = false;
        custom_config.enable_semantic_analysis = false;
        custom_config.enable_prosodic_analysis = false;
        custom_config.enable_pragmatic_analysis = false;
        custom_config.enable_statistical_analysis = false;
        custom_config.enable_quality_analysis = false;

        // Enable requested dimensions
        for dimension in dimensions {
            match *dimension {
                "language_model" => custom_config.enable_language_model_analysis = true,
                "syntactic" => custom_config.enable_syntactic_analysis = true,
                "lexical" => custom_config.enable_lexical_analysis = true,
                "semantic" => custom_config.enable_semantic_analysis = true,
                "prosodic" => custom_config.enable_prosodic_analysis = true,
                "pragmatic" => custom_config.enable_pragmatic_analysis = true,
                "statistical" => custom_config.enable_statistical_analysis = true,
                "quality" => custom_config.enable_quality_analysis = true,
                _ => {
                    return Err(FluencyAnalysisError::ConfigurationError(format!(
                        "Unknown dimension: {}",
                        dimension
                    )))
                }
            }
        }

        let custom_analyzer = Self::with_config(custom_config);
        custom_analyzer.analyze_comprehensive_fluency(text)
    }

    /// Quick analysis with minimal computational overhead.
    pub fn analyze_quick_fluency(&self, text: &str) -> Result<f64, FluencyAnalysisError> {
        let quick_config = AnalysisConfig::fast();
        let quick_analyzer = Self::with_config(quick_config);
        let analysis = quick_analyzer.analyze_comprehensive_fluency(text)?;
        Ok(analysis.overall_score)
    }

    /// Analyze text fluency with specific focus on readability.
    pub fn analyze_readability(
        &self,
        text: &str,
    ) -> Result<ReadabilityAnalysis, FluencyAnalysisError> {
        let dimensions = vec!["syntactic", "lexical", "semantic"];
        let analysis = self.analyze_selective_fluency(text, &dimensions)?;

        Ok(ReadabilityAnalysis {
            overall_readability: analysis.overall_score,
            syntactic_complexity: analysis.dimensional_scores.syntactic_score,
            lexical_difficulty: 1.0 - analysis.dimensional_scores.lexical_score, // Invert for difficulty
            semantic_clarity: analysis.dimensional_scores.semantic_score,
            reading_level: self.estimate_reading_level(analysis.overall_score),
            recommendations: self.generate_readability_recommendations(&analysis),
        })
    }

    // Helper methods

    fn calculate_dimensional_scores(
        &self,
        language_model_analysis: &Option<LanguageModelScore>,
        syntactic_analysis: &Option<SyntacticScore>,
        lexical_analysis: &Option<LexicalScore>,
        semantic_analysis: &Option<SemanticScore>,
        prosodic_analysis: &Option<ProsodicScore>,
        pragmatic_analysis: &Option<PragmaticScore>,
        quality_assessment: &Option<ComprehensiveQualityAssessment>,
    ) -> DimensionalScores {
        DimensionalScores {
            language_model_score: language_model_analysis
                .as_ref()
                .map(|a| a.overall_score)
                .unwrap_or(0.0),
            syntactic_score: syntactic_analysis
                .as_ref()
                .map(|a| a.overall_score)
                .unwrap_or(0.0),
            lexical_score: lexical_analysis
                .as_ref()
                .map(|a| a.overall_score)
                .unwrap_or(0.0),
            semantic_score: semantic_analysis
                .as_ref()
                .map(|a| a.overall_score)
                .unwrap_or(0.0),
            prosodic_score: prosodic_analysis
                .as_ref()
                .map(|a| a.overall_score)
                .unwrap_or(0.0),
            pragmatic_score: pragmatic_analysis
                .as_ref()
                .map(|a| a.overall_score)
                .unwrap_or(0.0),
            statistical_score: 0.0, // Calculated separately
            quality_score: quality_assessment
                .as_ref()
                .map(|a| a.overall_quality_score)
                .unwrap_or(0.0),
        }
    }

    fn calculate_overall_score(&self, dimensional_scores: &DimensionalScores) -> f64 {
        let scores = vec![
            dimensional_scores.language_model_score,
            dimensional_scores.syntactic_score,
            dimensional_scores.lexical_score,
            dimensional_scores.semantic_score,
            dimensional_scores.prosodic_score,
            dimensional_scores.pragmatic_score,
        ];

        let valid_scores: Vec<f64> = scores.into_iter().filter(|&s| s > 0.0).collect();

        if valid_scores.is_empty() {
            0.0
        } else {
            valid_scores.iter().sum::<f64>() / valid_scores.len() as f64
        }
    }

    fn create_analysis_metadata(&self, text: &str, duration_ms: u64) -> AnalysisMetadata {
        let word_count = text.split_whitespace().count();
        let sentence_count = text.split('.').filter(|s| !s.trim().is_empty()).count();

        let mut enabled_modules = Vec::new();
        if self.config.enable_language_model_analysis {
            enabled_modules.push("language_model".to_string());
        }
        if self.config.enable_syntactic_analysis {
            enabled_modules.push("syntactic".to_string());
        }
        if self.config.enable_lexical_analysis {
            enabled_modules.push("lexical".to_string());
        }
        if self.config.enable_semantic_analysis {
            enabled_modules.push("semantic".to_string());
        }
        if self.config.enable_prosodic_analysis {
            enabled_modules.push("prosodic".to_string());
        }
        if self.config.enable_pragmatic_analysis {
            enabled_modules.push("pragmatic".to_string());
        }
        if self.config.enable_statistical_analysis {
            enabled_modules.push("statistical".to_string());
        }
        if self.config.enable_quality_analysis {
            enabled_modules.push("quality".to_string());
        }

        let analysis_config = format!(
            "depth:{:?}, mode:{:?}",
            self.config.analysis_depth, self.config.performance_mode
        );

        let mut warnings = Vec::new();
        if word_count < 10 {
            warnings.push("Text is very short for comprehensive analysis".to_string());
        }
        if sentence_count < 2 {
            warnings.push("Text contains very few sentences".to_string());
        }

        AnalysisMetadata {
            text_length: text.len(),
            word_count,
            sentence_count,
            analysis_duration_ms: duration_ms,
            enabled_modules,
            analysis_config,
            warnings,
        }
    }

    fn estimate_reading_level(&self, overall_score: f64) -> ReadingLevel {
        if overall_score >= 0.9 {
            ReadingLevel::Graduate
        } else if overall_score >= 0.8 {
            ReadingLevel::College
        } else if overall_score >= 0.7 {
            ReadingLevel::HighSchool
        } else if overall_score >= 0.6 {
            ReadingLevel::MiddleSchool
        } else {
            ReadingLevel::Elementary
        }
    }

    fn generate_readability_recommendations(
        &self,
        analysis: &ComprehensiveFluencyAnalysis,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if analysis.dimensional_scores.syntactic_score < 0.7 {
            recommendations.push("Simplify sentence structure for better readability".to_string());
        }

        if analysis.dimensional_scores.lexical_score < 0.7 {
            recommendations.push("Use more common vocabulary to improve accessibility".to_string());
        }

        if analysis.dimensional_scores.semantic_score < 0.7 {
            recommendations.push("Improve logical flow and coherence between ideas".to_string());
        }

        if recommendations.is_empty() {
            recommendations
                .push("Text demonstrates good readability across all dimensions".to_string());
        }

        recommendations
    }
}

/// Specialized readability analysis result.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadabilityAnalysis {
    pub overall_readability: f64,
    pub syntactic_complexity: f64,
    pub lexical_difficulty: f64,
    pub semantic_clarity: f64,
    pub reading_level: ReadingLevel,
    pub recommendations: Vec<String>,
}

/// Estimated reading level based on analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum ReadingLevel {
    Elementary,
    MiddleSchool,
    HighSchool,
    College,
    Graduate,
}

/// Legacy compatibility interface for backward compatibility with existing code.
pub struct FluentText;

impl FluentText {
    /// Legacy method for backward compatibility.
    pub fn analyze_fluency(text: &str) -> Result<f64, FluencyAnalysisError> {
        let analyzer = FluentTextAnalyzer::new();
        analyzer.analyze_quick_fluency(text)
    }

    /// Legacy method for comprehensive analysis.
    pub fn analyze_comprehensive(
        text: &str,
    ) -> Result<ComprehensiveFluencyAnalysis, FluencyAnalysisError> {
        let analyzer = FluentTextAnalyzer::new();
        analyzer.analyze_comprehensive_fluency(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fluent_text_analyzer_creation() {
        let analyzer = FluentTextAnalyzer::new();
        assert!(analyzer.config.enable_language_model_analysis);
        assert!(analyzer.config.enable_semantic_analysis);
        assert!(analyzer.config.enable_quality_analysis);
    }

    #[test]
    fn test_analysis_config_builder() {
        let config = AnalysisConfig::builder()
            .enable_language_model_analysis(true)
            .enable_semantic_analysis(true)
            .disable_prosodic_analysis()
            .disable_pragmatic_analysis()
            .with_analysis_depth(AnalysisDepth::Basic)
            .with_performance_mode(PerformanceMode::Speed)
            .build();

        assert!(config.enable_language_model_analysis);
        assert!(config.enable_semantic_analysis);
        assert!(!config.enable_prosodic_analysis);
        assert!(!config.enable_pragmatic_analysis);
        assert_eq!(config.analysis_depth, AnalysisDepth::Basic);
        assert_eq!(config.performance_mode, PerformanceMode::Speed);
    }

    #[test]
    fn test_fast_configuration() {
        let config = AnalysisConfig::fast();
        assert!(config.enable_language_model_analysis);
        assert!(!config.enable_prosodic_analysis); // Disabled for speed
        assert!(!config.enable_pragmatic_analysis); // Disabled for speed
        assert_eq!(config.performance_mode, PerformanceMode::Speed);
    }

    #[test]
    fn test_comprehensive_configuration() {
        let config = AnalysisConfig::comprehensive();
        assert!(config.enable_language_model_analysis);
        assert!(config.enable_prosodic_analysis);
        assert!(config.enable_pragmatic_analysis);
        assert!(config.enable_statistical_analysis);
        assert_eq!(config.analysis_depth, AnalysisDepth::Comprehensive);
    }

    #[test]
    fn test_comprehensive_fluency_analysis() {
        let analyzer = FluentTextAnalyzer::new();
        let text = "This is a comprehensive test of the fluency analysis system. It includes multiple sentences with varying complexity levels. The text demonstrates good structure, appropriate vocabulary, and clear communication patterns.";

        let result = analyzer.analyze_comprehensive_fluency(text);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(analysis.overall_score >= 0.0);
        assert!(analysis.overall_score <= 1.0);
        assert!(analysis.dimensional_scores.language_model_score >= 0.0);
        assert!(analysis.dimensional_scores.semantic_score >= 0.0);
        assert!(analysis.language_model_analysis.is_some());
        assert!(analysis.quality_assessment.is_some());

        // Check metadata
        assert!(analysis.analysis_metadata.word_count > 0);
        assert!(analysis.analysis_metadata.sentence_count > 0);
        assert!(!analysis.analysis_metadata.enabled_modules.is_empty());
    }

    #[test]
    fn test_selective_fluency_analysis() {
        let analyzer = FluentTextAnalyzer::new();
        let text = "This text will be analyzed for specific dimensions only.";

        let dimensions = vec!["semantic", "lexical"];
        let result = analyzer.analyze_selective_fluency(text, &dimensions);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(analysis.semantic_analysis.is_some());
        assert!(analysis.lexical_analysis.is_some());
        assert!(analysis.prosodic_analysis.is_none()); // Should be None since not requested
    }

    #[test]
    fn test_quick_fluency_analysis() {
        let analyzer = FluentTextAnalyzer::new();
        let text = "Quick analysis test sentence with moderate complexity.";

        let result = analyzer.analyze_quick_fluency(text);
        assert!(result.is_ok());

        let score = result.unwrap();
        assert!(score >= 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_readability_analysis() {
        let analyzer = FluentTextAnalyzer::new();
        let text = "This is a simple sentence. It is easy to read and understand.";

        let result = analyzer.analyze_readability(text);
        assert!(result.is_ok());

        let readability = result.unwrap();
        assert!(readability.overall_readability >= 0.0);
        assert!(readability.overall_readability <= 1.0);
        assert!(!readability.recommendations.is_empty());
        assert!(matches!(
            readability.reading_level,
            ReadingLevel::Elementary
                | ReadingLevel::MiddleSchool
                | ReadingLevel::HighSchool
                | ReadingLevel::College
                | ReadingLevel::Graduate
        ));
    }

    #[test]
    fn test_fast_analyzer() {
        let analyzer = FluentTextAnalyzer::fast();
        let text = "Fast analysis test with efficient processing.";

        let result = analyzer.analyze_comprehensive_fluency(text);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        // Prosodic and pragmatic should be None in fast mode
        assert!(analysis.prosodic_analysis.is_none());
        assert!(analysis.pragmatic_analysis.is_none());
        // But language model and semantic should still be available
        assert!(analysis.language_model_analysis.is_some());
        assert!(analysis.semantic_analysis.is_some());
    }

    #[test]
    fn test_comprehensive_analyzer() {
        let analyzer = FluentTextAnalyzer::comprehensive();
        let text =
            "Comprehensive analysis includes all available modules and deep statistical analysis.";

        let result = analyzer.analyze_comprehensive_fluency(text);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        // All analyses should be available in comprehensive mode
        assert!(analysis.language_model_analysis.is_some());
        assert!(analysis.syntactic_analysis.is_some());
        assert!(analysis.lexical_analysis.is_some());
        assert!(analysis.semantic_analysis.is_some());
        assert!(analysis.prosodic_analysis.is_some());
        assert!(analysis.pragmatic_analysis.is_some());
        assert!(analysis.statistical_summary.is_some());
        assert!(analysis.quality_assessment.is_some());
    }

    #[test]
    fn test_empty_text_handling() {
        let analyzer = FluentTextAnalyzer::new();

        let result = analyzer.analyze_comprehensive_fluency("");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            FluencyAnalysisError::TextProcessingError(_)
        ));

        let result2 = analyzer.analyze_comprehensive_fluency("   ");
        assert!(result2.is_err());
    }

    #[test]
    fn test_invalid_dimension_handling() {
        let analyzer = FluentTextAnalyzer::new();
        let text = "Test text for invalid dimension handling.";

        let invalid_dimensions = vec!["invalid_dimension"];
        let result = analyzer.analyze_selective_fluency(text, &invalid_dimensions);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            FluencyAnalysisError::ConfigurationError(_)
        ));
    }

    #[test]
    fn test_dimensional_score_calculation() {
        let analyzer = FluentTextAnalyzer::new();

        // Create some mock analyses
        let lm_analysis = Some(LanguageModelScore {
            overall_score: 0.8,
            perplexity: 15.0,
            surprisal_score: 0.7,
            fluency_score: 0.85,
            coherence_score: 0.75,
            n_gram_score: 0.8,
            detailed_metrics: Default::default(),
        });

        let semantic_analysis = Some(SemanticScore {
            overall_score: 0.9,
            coherence_score: 0.85,
            consistency_score: 0.88,
            clarity_score: 0.92,
            relevance_score: 0.87,
            depth_score: 0.9,
            detailed_metrics: Default::default(),
        });

        let dimensional_scores = analyzer.calculate_dimensional_scores(
            &lm_analysis,
            &None, // syntactic
            &None, // lexical
            &semantic_analysis,
            &None, // prosodic
            &None, // pragmatic
            &None, // quality
        );

        assert_eq!(dimensional_scores.language_model_score, 0.8);
        assert_eq!(dimensional_scores.semantic_score, 0.9);
        assert_eq!(dimensional_scores.syntactic_score, 0.0); // None = 0.0
    }

    #[test]
    fn test_overall_score_calculation() {
        let analyzer = FluentTextAnalyzer::new();

        let dimensional_scores = DimensionalScores {
            language_model_score: 0.8,
            syntactic_score: 0.7,
            lexical_score: 0.9,
            semantic_score: 0.85,
            prosodic_score: 0.75,
            pragmatic_score: 0.8,
            statistical_score: 0.0, // Not used in calculation
            quality_score: 0.0,     // Not used in calculation
        };

        let overall_score = analyzer.calculate_overall_score(&dimensional_scores);
        let expected = (0.8 + 0.7 + 0.9 + 0.85 + 0.75 + 0.8) / 6.0;
        assert!((overall_score - expected).abs() < 0.001);
    }

    #[test]
    fn test_analysis_metadata_creation() {
        let analyzer = FluentTextAnalyzer::new();
        let text = "This is a test. It has two sentences.";

        let metadata = analyzer.create_analysis_metadata(text, 100);

        assert_eq!(metadata.text_length, text.len());
        assert_eq!(metadata.word_count, 8);
        assert_eq!(metadata.sentence_count, 2);
        assert_eq!(metadata.analysis_duration_ms, 100);
        assert!(!metadata.enabled_modules.is_empty());
        assert!(!metadata.analysis_config.is_empty());
    }

    #[test]
    fn test_reading_level_estimation() {
        let analyzer = FluentTextAnalyzer::new();

        assert!(matches!(
            analyzer.estimate_reading_level(0.95),
            ReadingLevel::Graduate
        ));
        assert!(matches!(
            analyzer.estimate_reading_level(0.85),
            ReadingLevel::College
        ));
        assert!(matches!(
            analyzer.estimate_reading_level(0.75),
            ReadingLevel::HighSchool
        ));
        assert!(matches!(
            analyzer.estimate_reading_level(0.65),
            ReadingLevel::MiddleSchool
        ));
        assert!(matches!(
            analyzer.estimate_reading_level(0.55),
            ReadingLevel::Elementary
        ));
    }

    #[test]
    fn test_legacy_compatibility() {
        let text = "Legacy compatibility test sentence.";

        let result = FluentText::analyze_fluency(text);
        assert!(result.is_ok());
        let score = result.unwrap();
        assert!(score >= 0.0);
        assert!(score <= 1.0);

        let result2 = FluentText::analyze_comprehensive(text);
        assert!(result2.is_ok());
        let analysis = result2.unwrap();
        assert!(analysis.overall_score >= 0.0);
    }

    #[test]
    fn test_custom_quality_weights() {
        let custom_weights = quality::QualityWeights {
            language_model_weight: 0.4,
            syntactic_weight: 0.2,
            lexical_weight: 0.2,
            semantic_weight: 0.1,
            prosodic_weight: 0.05,
            pragmatic_weight: 0.03,
            statistical_weight: 0.02,
        };

        let config = AnalysisConfig::builder()
            .with_quality_weights(custom_weights)
            .build();

        let analyzer = FluentTextAnalyzer::with_config(config);
        let text = "Testing custom quality weights configuration.";

        let result = analyzer.analyze_comprehensive_fluency(text);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(analysis.quality_assessment.is_some());
    }

    #[test]
    fn test_performance_modes() {
        let text = "Performance mode testing text with reasonable length for analysis.";

        // Speed mode
        let speed_config = AnalysisConfig::builder()
            .with_performance_mode(PerformanceMode::Speed)
            .build();
        let speed_analyzer = FluentTextAnalyzer::with_config(speed_config);
        let speed_result = speed_analyzer.analyze_comprehensive_fluency(text);
        assert!(speed_result.is_ok());

        // Balanced mode
        let balanced_config = AnalysisConfig::builder()
            .with_performance_mode(PerformanceMode::Balanced)
            .build();
        let balanced_analyzer = FluentTextAnalyzer::with_config(balanced_config);
        let balanced_result = balanced_analyzer.analyze_comprehensive_fluency(text);
        assert!(balanced_result.is_ok());

        // Accuracy mode
        let accuracy_config = AnalysisConfig::builder()
            .with_performance_mode(PerformanceMode::Accuracy)
            .build();
        let accuracy_analyzer = FluentTextAnalyzer::with_config(accuracy_config);
        let accuracy_result = accuracy_analyzer.analyze_comprehensive_fluency(text);
        assert!(accuracy_result.is_ok());
    }

    #[test]
    fn test_analysis_depth_modes() {
        let text =
            "Analysis depth testing with comprehensive evaluation of different depth levels.";

        // Basic depth
        let basic_config = AnalysisConfig::builder()
            .with_analysis_depth(AnalysisDepth::Basic)
            .build();
        let basic_analyzer = FluentTextAnalyzer::with_config(basic_config);
        let basic_result = basic_analyzer.analyze_comprehensive_fluency(text);
        assert!(basic_result.is_ok());

        // Standard depth
        let standard_config = AnalysisConfig::builder()
            .with_analysis_depth(AnalysisDepth::Standard)
            .build();
        let standard_analyzer = FluentTextAnalyzer::with_config(standard_config);
        let standard_result = standard_analyzer.analyze_comprehensive_fluency(text);
        assert!(standard_result.is_ok());

        // Comprehensive depth
        let comprehensive_config = AnalysisConfig::builder()
            .with_analysis_depth(AnalysisDepth::Comprehensive)
            .build();
        let comprehensive_analyzer = FluentTextAnalyzer::with_config(comprehensive_config);
        let comprehensive_result = comprehensive_analyzer.analyze_comprehensive_fluency(text);
        assert!(comprehensive_result.is_ok());

        // Deep depth
        let deep_config = AnalysisConfig::builder()
            .with_analysis_depth(AnalysisDepth::Deep)
            .build();
        let deep_analyzer = FluentTextAnalyzer::with_config(deep_config);
        let deep_result = deep_analyzer.analyze_comprehensive_fluency(text);
        assert!(deep_result.is_ok());
    }
}
