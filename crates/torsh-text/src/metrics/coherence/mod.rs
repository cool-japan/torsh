//! Coherence Analysis Module
//!
//! This module provides comprehensive coherence analysis capabilities for natural language text.
//! It combines multiple specialized coherence analysis approaches including entity-based,
//! lexical, discourse, topic, and structural coherence analysis with advanced statistical
//! measurement and comparison capabilities.
//!
//! # Overview
//!
//! Coherence analysis helps assess how well-connected and logically organized a piece of text is.
//! This module provides both granular analysis through specialized submodules and unified
//! analysis through the main `CoherenceAnalyzer` interface.
//!
//! # Architecture
//!
//! The module is organized into specialized analyzers:
//!
//! - **Entity Coherence**: Entity chains, coreference resolution, entity grids
//! - **Lexical Coherence**: Lexical chains, semantic fields, vocabulary consistency
//! - **Discourse Coherence**: Discourse markers, rhetorical structure, cohesion devices
//! - **Topic Coherence**: Topic modeling, thematic progression, topic transitions
//! - **Structural Coherence**: Paragraph organization, hierarchical structure, document flow
//! - **Coherence Metrics**: Statistical analysis, confidence scoring, comparative analysis
//!
//! # Usage Examples
//!
//! ## Basic Analysis
//! ```rust
//! use torsh_text::metrics::coherence::{CoherenceAnalyzer, CoherenceConfig};
//!
//! let analyzer = CoherenceAnalyzer::with_default_config();
//! let result = analyzer.analyze_coherence("Your text here...");
//! println!("Overall coherence: {}", result.overall_coherence);
//! ```
//!
//! ## Advanced Configuration
//! ```rust
//! use torsh_text::metrics::coherence::{CoherenceAnalyzer, CoherenceConfig, CoherenceType};
//!
//! let config = CoherenceConfig {
//!     window_size: 5,
//!     min_entity_mentions: 3,
//!     topic_threshold: 0.7,
//!     coherence_types: vec![CoherenceType::Entity, CoherenceType::Lexical],
//!     ..Default::default()
//! };
//!
//! let analyzer = CoherenceAnalyzer::new(config);
//! let result = analyzer.analyze_coherence("Your text here...");
//! ```
//!
//! ## Specialized Analysis
//! ```rust
//! use torsh_text::metrics::coherence::entity_coherence::{EntityCoherenceAnalyzer, EntityCoherenceConfig};
//!
//! let entity_analyzer = EntityCoherenceAnalyzer::with_default_config();
//! let entity_result = entity_analyzer.analyze_entity_coherence("Your text here...");
//! ```

pub mod coherence_metrics;
pub mod discourse_coherence;
pub mod entity_coherence;
pub mod lexical_coherence;
pub mod structural_coherence;
pub mod topic_coherence;

use std::collections::{HashMap, HashSet};
use thiserror::Error;

// Re-export key types from submodules
pub use coherence_metrics::{
    CoherenceMetricsCalculator, CoherenceMetricsConfig, CoherenceMetricsError,
    CoherenceMetricsResult,
};
pub use discourse_coherence::{
    DiscourseCoherenceAnalyzer, DiscourseCoherenceConfig, DiscourseCoherenceError,
    DiscourseCoherenceResult,
};
pub use entity_coherence::{
    EntityCoherenceAnalyzer, EntityCoherenceConfig, EntityCoherenceError, EntityCoherenceResult,
};
pub use lexical_coherence::{
    LexicalCoherenceAnalyzer, LexicalCoherenceConfig, LexicalCoherenceError, LexicalCoherenceResult,
};
pub use structural_coherence::{
    StructuralCoherenceAnalyzer, StructuralCoherenceConfig, StructuralCoherenceError,
    StructuralCoherenceResult,
};
pub use topic_coherence::{
    TopicCoherenceAnalyzer, TopicCoherenceConfig, TopicCoherenceError, TopicCoherenceResult,
};

/// Errors that can occur during coherence analysis
#[derive(Error, Debug, Clone, PartialEq)]
pub enum CoherenceError {
    #[error("Entity coherence analysis failed: {0}")]
    EntityAnalysis(#[from] EntityCoherenceError),
    #[error("Lexical coherence analysis failed: {0}")]
    LexicalAnalysis(#[from] LexicalCoherenceError),
    #[error("Discourse coherence analysis failed: {0}")]
    DiscourseAnalysis(#[from] DiscourseCoherenceError),
    #[error("Topic coherence analysis failed: {0}")]
    TopicAnalysis(#[from] TopicCoherenceError),
    #[error("Structural coherence analysis failed: {0}")]
    StructuralAnalysis(#[from] StructuralCoherenceError),
    #[error("Coherence metrics calculation failed: {0}")]
    MetricsCalculation(#[from] CoherenceMetricsError),
    #[error("Invalid input: {message}")]
    InvalidInput { message: String },
    #[error("Configuration error: {parameter} = {value}")]
    ConfigurationError { parameter: String, value: String },
    #[error("Analysis failed: {operation}")]
    AnalysisError { operation: String },
}

/// Types of coherence analysis available
#[derive(Debug, Clone, PartialEq)]
pub enum CoherenceType {
    Entity,
    Lexical,
    Discourse,
    Topic,
    Structural,
    All,
}

/// Configuration for comprehensive coherence analysis
#[derive(Debug, Clone)]
pub struct CoherenceConfig {
    pub window_size: usize,
    pub min_entity_mentions: usize,
    pub topic_threshold: f64,
    pub semantic_similarity_threshold: f64,
    pub use_coreference_resolution: bool,
    pub use_semantic_similarity: bool,
    pub coherence_types: Vec<CoherenceType>,
    pub entity_config: EntityCoherenceConfig,
    pub lexical_config: LexicalCoherenceConfig,
    pub discourse_config: DiscourseCoherenceConfig,
    pub topic_config: TopicCoherenceConfig,
    pub structural_config: StructuralCoherenceConfig,
    pub metrics_config: CoherenceMetricsConfig,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            window_size: 3,
            min_entity_mentions: 2,
            topic_threshold: 0.6,
            semantic_similarity_threshold: 0.5,
            use_coreference_resolution: true,
            use_semantic_similarity: true,
            coherence_types: vec![CoherenceType::All],
            entity_config: EntityCoherenceConfig::default(),
            lexical_config: LexicalCoherenceConfig::default(),
            discourse_config: DiscourseCoherenceConfig::default(),
            topic_config: TopicCoherenceConfig::default(),
            structural_config: StructuralCoherenceConfig::default(),
            metrics_config: CoherenceMetricsConfig::default(),
        }
    }
}

/// Comprehensive coherence analysis result
#[derive(Debug, Clone, PartialEq)]
pub struct CoherenceResult {
    pub overall_coherence: f64,
    pub entity_coherence: EntityCoherenceResult,
    pub lexical_coherence: LexicalCoherenceResult,
    pub discourse_coherence: DiscourseCoherenceResult,
    pub topic_coherence: TopicCoherenceResult,
    pub structural_coherence: StructuralCoherenceResult,
    pub metrics_result: CoherenceMetricsResult,
    pub analysis_metadata: AnalysisMetadata,
}

/// Metadata about the coherence analysis
#[derive(Debug, Clone, PartialEq)]
pub struct AnalysisMetadata {
    pub text_length: usize,
    pub sentence_count: usize,
    pub paragraph_count: usize,
    pub word_count: usize,
    pub analysis_duration_ms: u64,
    pub enabled_analyses: Vec<CoherenceType>,
    pub confidence_level: f64,
}

/// Result of coherence comparison between two texts
#[derive(Debug, Clone, PartialEq)]
pub struct CoherenceComparisonResult {
    pub text1_coherence: f64,
    pub text2_coherence: f64,
    pub coherence_difference: f64,
    pub better_text: u8,
    pub detailed_comparison: HashMap<String, f64>,
}

/// Main coherence analyzer that coordinates all specialized analyzers
pub struct CoherenceAnalyzer {
    config: CoherenceConfig,
    entity_analyzer: EntityCoherenceAnalyzer,
    lexical_analyzer: LexicalCoherenceAnalyzer,
    discourse_analyzer: DiscourseCoherenceAnalyzer,
    topic_analyzer: TopicCoherenceAnalyzer,
    structural_analyzer: StructuralCoherenceAnalyzer,
    metrics_calculator: CoherenceMetricsCalculator,
}

impl CoherenceAnalyzer {
    /// Create new coherence analyzer with custom configuration
    pub fn new(config: CoherenceConfig) -> Self {
        let entity_analyzer = EntityCoherenceAnalyzer::new(config.entity_config.clone());
        let lexical_analyzer = LexicalCoherenceAnalyzer::new(config.lexical_config.clone());
        let discourse_analyzer = DiscourseCoherenceAnalyzer::new(config.discourse_config.clone());
        let topic_analyzer = TopicCoherenceAnalyzer::new(config.topic_config.clone());
        let structural_analyzer =
            StructuralCoherenceAnalyzer::new(config.structural_config.clone());
        let metrics_calculator = CoherenceMetricsCalculator::new(config.metrics_config.clone());

        Self {
            config,
            entity_analyzer,
            lexical_analyzer,
            discourse_analyzer,
            topic_analyzer,
            structural_analyzer,
            metrics_calculator,
        }
    }

    /// Create analyzer with default configuration
    pub fn with_default_config() -> Self {
        Self::new(CoherenceConfig::default())
    }

    /// Perform comprehensive coherence analysis on text
    pub fn analyze_coherence(&self, text: &str) -> Result<CoherenceResult, CoherenceError> {
        let start_time = std::time::Instant::now();

        if text.trim().is_empty() {
            return Err(CoherenceError::InvalidInput {
                message: "Input text is empty".to_string(),
            });
        }

        // Text preprocessing
        let sentences = self.split_into_sentences(text);
        let paragraphs = self.split_into_paragraphs(text);
        let words = self.extract_words(text);

        // Perform individual analyses based on configuration
        let entity_coherence = if self.should_analyze(&CoherenceType::Entity) {
            self.entity_analyzer.analyze_entity_coherence(text)?
        } else {
            EntityCoherenceResult::default()
        };

        let lexical_coherence = if self.should_analyze(&CoherenceType::Lexical) {
            self.lexical_analyzer.analyze_lexical_coherence(text)?
        } else {
            LexicalCoherenceResult::default()
        };

        let discourse_coherence = if self.should_analyze(&CoherenceType::Discourse) {
            self.discourse_analyzer.analyze_discourse_coherence(text)?
        } else {
            DiscourseCoherenceResult::default()
        };

        let topic_coherence = if self.should_analyze(&CoherenceType::Topic) {
            self.topic_analyzer.analyze_topic_coherence(text)?
        } else {
            TopicCoherenceResult::default()
        };

        let structural_coherence = if self.should_analyze(&CoherenceType::Structural) {
            self.structural_analyzer
                .analyze_structural_coherence(text)?
        } else {
            StructuralCoherenceResult::default()
        };

        // Calculate comprehensive metrics
        let metrics_result = self.metrics_calculator.calculate_comprehensive_metrics(
            &entity_coherence,
            &lexical_coherence,
            &discourse_coherence,
            &topic_coherence,
            &structural_coherence,
            &sentences,
            &paragraphs,
        )?;

        // Create analysis metadata
        let analysis_duration = start_time.elapsed().as_millis() as u64;
        let analysis_metadata = AnalysisMetadata {
            text_length: text.len(),
            sentence_count: sentences.len(),
            paragraph_count: paragraphs.len(),
            word_count: words.len(),
            analysis_duration_ms: analysis_duration,
            enabled_analyses: self.get_enabled_analyses(),
            confidence_level: metrics_result.confidence_score,
        };

        Ok(CoherenceResult {
            overall_coherence: metrics_result.overall_coherence,
            entity_coherence,
            lexical_coherence,
            discourse_coherence,
            topic_coherence,
            structural_coherence,
            metrics_result,
            analysis_metadata,
        })
    }

    /// Compare coherence between two texts
    pub fn compare_coherence(
        &self,
        text1: &str,
        text2: &str,
    ) -> Result<CoherenceComparisonResult, CoherenceError> {
        let result1 = self.analyze_coherence(text1)?;
        let result2 = self.analyze_coherence(text2)?;

        let comparison = self
            .metrics_calculator
            .compare_coherence(&result1.metrics_result, &result2.metrics_result)?;

        Ok(CoherenceComparisonResult {
            text1_coherence: result1.overall_coherence,
            text2_coherence: result2.overall_coherence,
            coherence_difference: comparison.coherence_difference,
            better_text: comparison.better_text,
            detailed_comparison: comparison.detailed_comparison,
        })
    }

    /// Analyze only specific type of coherence
    pub fn analyze_specific_coherence(
        &self,
        text: &str,
        coherence_type: CoherenceType,
    ) -> Result<CoherenceResult, CoherenceError> {
        let mut config = self.config.clone();
        config.coherence_types = vec![coherence_type];

        let analyzer = CoherenceAnalyzer::new(config);
        analyzer.analyze_coherence(text)
    }

    /// Get coherence summary as string
    pub fn get_coherence_summary(&self, result: &CoherenceResult) -> String {
        let mut summary = String::new();

        summary.push_str(&format!("=== COHERENCE ANALYSIS SUMMARY ===\n\n"));
        summary.push_str(&format!(
            "Overall Coherence Score: {:.3}\n",
            result.overall_coherence
        ));
        summary.push_str(&format!(
            "Confidence Level: {:.3}\n\n",
            result.analysis_metadata.confidence_level
        ));

        summary.push_str("Component Scores:\n");
        summary.push_str(&format!(
            "  Entity Coherence: {:.3}\n",
            result.entity_coherence.entity_grid_coherence
        ));
        summary.push_str(&format!(
            "  Lexical Coherence: {:.3}\n",
            result.lexical_coherence.lexical_chain_coherence
        ));
        summary.push_str(&format!(
            "  Discourse Coherence: {:.3}\n",
            result.discourse_coherence.discourse_marker_coherence
        ));
        summary.push_str(&format!(
            "  Topic Coherence: {:.3}\n",
            result.topic_coherence.topic_consistency
        ));
        summary.push_str(&format!(
            "  Structural Coherence: {:.3}\n\n",
            result.structural_coherence.paragraph_coherence
        ));

        summary.push_str("Text Statistics:\n");
        summary.push_str(&format!(
            "  Length: {} characters\n",
            result.analysis_metadata.text_length
        ));
        summary.push_str(&format!(
            "  Sentences: {}\n",
            result.analysis_metadata.sentence_count
        ));
        summary.push_str(&format!(
            "  Paragraphs: {}\n",
            result.analysis_metadata.paragraph_count
        ));
        summary.push_str(&format!(
            "  Words: {}\n",
            result.analysis_metadata.word_count
        ));
        summary.push_str(&format!(
            "  Analysis Duration: {} ms\n",
            result.analysis_metadata.analysis_duration_ms
        ));

        if let Some(ref stats) = result.metrics_result.statistical_analysis {
            summary.push_str(&format!("\nStatistical Analysis:\n"));
            summary.push_str(&format!("  Mean: {:.3}\n", stats.mean));
            summary.push_str(&format!("  Standard Deviation: {:.3}\n", stats.std_dev));
            summary.push_str(&format!("  Median: {:.3}\n", stats.median));
        }

        summary
    }

    // Private helper methods

    fn should_analyze(&self, coherence_type: &CoherenceType) -> bool {
        self.config.coherence_types.contains(coherence_type)
            || self.config.coherence_types.contains(&CoherenceType::All)
    }

    fn get_enabled_analyses(&self) -> Vec<CoherenceType> {
        if self.config.coherence_types.contains(&CoherenceType::All) {
            vec![
                CoherenceType::Entity,
                CoherenceType::Lexical,
                CoherenceType::Discourse,
                CoherenceType::Topic,
                CoherenceType::Structural,
            ]
        } else {
            self.config.coherence_types.clone()
        }
    }

    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        text.split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.len() > 3)
            .collect()
    }

    fn split_into_paragraphs(&self, text: &str) -> Vec<String> {
        text.split("\n\n")
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect()
    }

    fn extract_words(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()).to_lowercase())
            .filter(|w| !w.is_empty() && w.len() > 2)
            .collect()
    }
}

// Provide default implementations for result types to handle cases where analysis is disabled
impl Default for EntityCoherenceResult {
    fn default() -> Self {
        Self {
            entity_grid_coherence: 0.0,
            entity_transition_coherence: 0.0,
            coreference_coherence: 0.0,
            entity_density: 0.0,
            salience_score: 0.0,
            entity_chains: Vec::new(),
            dominant_entities: Vec::new(),
            entity_distribution: HashMap::new(),
        }
    }
}

impl Default for LexicalCoherenceResult {
    fn default() -> Self {
        Self {
            lexical_chain_coherence: 0.0,
            semantic_field_coherence: 0.0,
            lexical_repetition_score: 0.0,
            vocabulary_consistency: 0.0,
            lexical_density: 0.0,
            word_relatedness: 0.0,
            lexical_chains: Vec::new(),
            semantic_profiles: Vec::new(),
            topic_evolution: None,
            chain_statistics: HashMap::new(),
        }
    }
}

impl Default for DiscourseCoherenceResult {
    fn default() -> Self {
        Self {
            discourse_marker_coherence: 0.0,
            transition_coherence: 0.0,
            rhetorical_structure_coherence: 0.0,
            cohesion_score: 0.0,
            discourse_markers: Vec::new(),
            transition_quality: Vec::new(),
            rhetorical_relations: HashMap::new(),
            cohesion_devices: HashMap::new(),
        }
    }
}

impl Default for TopicCoherenceResult {
    fn default() -> Self {
        Self {
            topic_consistency: 0.0,
            topic_shift_coherence: 0.0,
            thematic_unity: 0.0,
            topics: Vec::new(),
            topic_transitions: Vec::new(),
            topic_distribution: HashMap::new(),
            coherence_per_topic: HashMap::new(),
        }
    }
}

impl Default for StructuralCoherenceResult {
    fn default() -> Self {
        Self {
            paragraph_coherence: 0.0,
            section_coherence: 0.0,
            organizational_coherence: 0.0,
            hierarchical_coherence: 0.0,
            structural_consistency: 0.0,
            paragraph_transitions: Vec::new(),
            structural_markers: Vec::new(),
            coherence_patterns: Vec::new(),
        }
    }
}

/// Convenience functions for simple coherence analysis

/// Calculate overall coherence score for text (simplified interface)
pub fn calculate_overall_coherence_simple(text: &str) -> Result<f64, CoherenceError> {
    let analyzer = CoherenceAnalyzer::with_default_config();
    let result = analyzer.analyze_coherence(text)?;
    Ok(result.overall_coherence)
}

/// Calculate entity coherence score for text
pub fn calculate_entity_coherence_simple(text: &str) -> Result<f64, CoherenceError> {
    let analyzer = CoherenceAnalyzer::with_default_config();
    let result = analyzer.analyze_specific_coherence(text, CoherenceType::Entity)?;
    Ok(result.entity_coherence.entity_grid_coherence)
}

/// Calculate lexical coherence score for text
pub fn calculate_lexical_coherence_simple(text: &str) -> Result<f64, CoherenceError> {
    let analyzer = CoherenceAnalyzer::with_default_config();
    let result = analyzer.analyze_specific_coherence(text, CoherenceType::Lexical)?;
    Ok(result.lexical_coherence.lexical_chain_coherence)
}

/// Calculate discourse coherence score for text
pub fn calculate_discourse_coherence_simple(text: &str) -> Result<f64, CoherenceError> {
    let analyzer = CoherenceAnalyzer::with_default_config();
    let result = analyzer.analyze_specific_coherence(text, CoherenceType::Discourse)?;
    Ok(result.discourse_coherence.discourse_marker_coherence)
}

/// Calculate topic coherence score for text
pub fn calculate_topic_coherence_simple(text: &str) -> Result<f64, CoherenceError> {
    let analyzer = CoherenceAnalyzer::with_default_config();
    let result = analyzer.analyze_specific_coherence(text, CoherenceType::Topic)?;
    Ok(result.topic_coherence.topic_consistency)
}

/// Calculate structural coherence score for text
pub fn calculate_structural_coherence_simple(text: &str) -> Result<f64, CoherenceError> {
    let analyzer = CoherenceAnalyzer::with_default_config();
    let result = analyzer.analyze_specific_coherence(text, CoherenceType::Structural)?;
    Ok(result.structural_coherence.paragraph_coherence)
}

/// Compare coherence between two texts (simplified interface)
pub fn compare_text_coherence_simple(
    text1: &str,
    text2: &str,
) -> Result<CoherenceComparisonResult, CoherenceError> {
    let analyzer = CoherenceAnalyzer::with_default_config();
    analyzer.compare_coherence(text1, text2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coherence_analyzer_creation() {
        let analyzer = CoherenceAnalyzer::with_default_config();
        assert_eq!(analyzer.config.window_size, 3);
        assert_eq!(analyzer.config.min_entity_mentions, 2);
    }

    #[test]
    fn test_coherence_analysis() -> Result<(), CoherenceError> {
        let analyzer = CoherenceAnalyzer::with_default_config();
        let text = "The cat sat on the mat. The cat was very comfortable. It decided to take a nap. The weather was perfect for sleeping.";

        let result = analyzer.analyze_coherence(text)?;

        assert!(result.overall_coherence >= 0.0 && result.overall_coherence <= 1.0);
        assert!(result.entity_coherence.entity_grid_coherence >= 0.0);
        assert!(result.lexical_coherence.lexical_chain_coherence >= 0.0);
        assert!(result.discourse_coherence.discourse_marker_coherence >= 0.0);
        assert!(result.topic_coherence.topic_consistency >= 0.0);
        assert!(result.structural_coherence.paragraph_coherence >= 0.0);
        assert!(result.analysis_metadata.text_length > 0);
        assert!(result.analysis_metadata.sentence_count > 0);

        Ok(())
    }

    #[test]
    fn test_specific_coherence_analysis() -> Result<(), CoherenceError> {
        let analyzer = CoherenceAnalyzer::with_default_config();
        let text = "The dog ran quickly. The quick dog jumped high. Dogs are energetic animals.";

        let entity_result = analyzer.analyze_specific_coherence(text, CoherenceType::Entity)?;
        assert!(entity_result.entity_coherence.entity_grid_coherence >= 0.0);

        let lexical_result = analyzer.analyze_specific_coherence(text, CoherenceType::Lexical)?;
        assert!(lexical_result.lexical_coherence.lexical_chain_coherence >= 0.0);

        Ok(())
    }

    #[test]
    fn test_coherence_comparison() -> Result<(), CoherenceError> {
        let analyzer = CoherenceAnalyzer::with_default_config();

        let text1 = "The cat sat on the mat. The cat was comfortable. It decided to sleep.";
        let text2 = "Random words. Unrelated sentence. No coherence here.";

        let comparison = analyzer.compare_coherence(text1, text2)?;

        assert!(comparison.text1_coherence >= 0.0 && comparison.text1_coherence <= 1.0);
        assert!(comparison.text2_coherence >= 0.0 && comparison.text2_coherence <= 1.0);
        assert!(comparison.better_text == 1 || comparison.better_text == 2);
        assert!(!comparison.detailed_comparison.is_empty());

        Ok(())
    }

    #[test]
    fn test_simple_calculation_functions() -> Result<(), CoherenceError> {
        let text = "The cat sat on the mat. The cat was comfortable. It decided to sleep.";

        let overall_coherence = calculate_overall_coherence_simple(text)?;
        assert!(overall_coherence >= 0.0 && overall_coherence <= 1.0);

        let entity_coherence = calculate_entity_coherence_simple(text)?;
        assert!(entity_coherence >= 0.0 && entity_coherence <= 1.0);

        let lexical_coherence = calculate_lexical_coherence_simple(text)?;
        assert!(lexical_coherence >= 0.0 && lexical_coherence <= 1.0);

        let discourse_coherence = calculate_discourse_coherence_simple(text)?;
        assert!(discourse_coherence >= 0.0 && discourse_coherence <= 1.0);

        let topic_coherence = calculate_topic_coherence_simple(text)?;
        assert!(topic_coherence >= 0.0 && topic_coherence <= 1.0);

        let structural_coherence = calculate_structural_coherence_simple(text)?;
        assert!(structural_coherence >= 0.0 && structural_coherence <= 1.0);

        Ok(())
    }

    #[test]
    fn test_comparison_simple() -> Result<(), CoherenceError> {
        let text1 = "The cat sat on the mat. The cat was comfortable.";
        let text2 = "Random sentence. Another unrelated thought.";

        let comparison = compare_text_coherence_simple(text1, text2)?;
        assert!(comparison.coherence_difference != 0.0);
        assert!(comparison.better_text == 1 || comparison.better_text == 2);

        Ok(())
    }

    #[test]
    fn test_coherence_summary() -> Result<(), CoherenceError> {
        let analyzer = CoherenceAnalyzer::with_default_config();
        let text =
            "The cat sat on the mat. The cat was very comfortable. It decided to take a nap.";

        let result = analyzer.analyze_coherence(text)?;
        let summary = analyzer.get_coherence_summary(&result);

        assert!(summary.contains("COHERENCE ANALYSIS SUMMARY"));
        assert!(summary.contains("Overall Coherence Score"));
        assert!(summary.contains("Component Scores"));
        assert!(summary.contains("Text Statistics"));

        Ok(())
    }

    #[test]
    fn test_custom_configuration() -> Result<(), CoherenceError> {
        let config = CoherenceConfig {
            window_size: 5,
            min_entity_mentions: 3,
            topic_threshold: 0.8,
            coherence_types: vec![CoherenceType::Entity, CoherenceType::Lexical],
            ..Default::default()
        };

        let analyzer = CoherenceAnalyzer::new(config);
        let text = "The dog ran quickly. The dog jumped high. The dog was energetic.";

        let result = analyzer.analyze_coherence(text)?;
        assert!(result.overall_coherence >= 0.0);

        Ok(())
    }

    #[test]
    fn test_error_handling() {
        let analyzer = CoherenceAnalyzer::with_default_config();

        // Test empty text
        let result = analyzer.analyze_coherence("");
        assert!(matches!(result, Err(CoherenceError::InvalidInput { .. })));

        // Test whitespace-only text
        let result = analyzer.analyze_coherence("   \n\t  ");
        assert!(matches!(result, Err(CoherenceError::InvalidInput { .. })));
    }

    #[test]
    fn test_text_preprocessing() {
        let analyzer = CoherenceAnalyzer::with_default_config();

        let sentences =
            analyzer.split_into_sentences("First sentence. Second sentence! Third sentence?");
        assert_eq!(sentences.len(), 3);

        let paragraphs = analyzer
            .split_into_paragraphs("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.");
        assert_eq!(paragraphs.len(), 3);

        let words = analyzer.extract_words("The quick brown fox jumps over the lazy dog.");
        assert!(words.len() > 5);
        assert!(words.iter().all(|w| w.chars().all(|c| c.is_alphabetic())));
    }

    #[test]
    fn test_analysis_metadata() -> Result<(), CoherenceError> {
        let analyzer = CoherenceAnalyzer::with_default_config();
        let text = "The cat sat on the mat. The cat was comfortable. It decided to sleep.";

        let result = analyzer.analyze_coherence(text)?;
        let metadata = &result.analysis_metadata;

        assert!(metadata.text_length > 0);
        assert!(metadata.sentence_count > 0);
        assert!(metadata.word_count > 0);
        assert!(metadata.analysis_duration_ms > 0);
        assert!(!metadata.enabled_analyses.is_empty());
        assert!(metadata.confidence_level >= 0.0 && metadata.confidence_level <= 1.0);

        Ok(())
    }

    #[test]
    fn test_coherence_types() {
        let entity_type = CoherenceType::Entity;
        let lexical_type = CoherenceType::Lexical;
        let all_type = CoherenceType::All;

        assert_ne!(entity_type, lexical_type);
        assert_ne!(entity_type, all_type);

        // Test in configuration
        let config = CoherenceConfig {
            coherence_types: vec![entity_type.clone(), lexical_type.clone()],
            ..Default::default()
        };

        assert!(config.coherence_types.contains(&CoherenceType::Entity));
        assert!(config.coherence_types.contains(&CoherenceType::Lexical));
        assert!(!config.coherence_types.contains(&CoherenceType::All));
    }

    #[test]
    fn test_default_implementations() {
        let entity_result = EntityCoherenceResult::default();
        assert_eq!(entity_result.entity_grid_coherence, 0.0);
        assert!(entity_result.entity_chains.is_empty());

        let lexical_result = LexicalCoherenceResult::default();
        assert_eq!(lexical_result.lexical_chain_coherence, 0.0);
        assert!(lexical_result.lexical_chains.is_empty());

        let discourse_result = DiscourseCoherenceResult::default();
        assert_eq!(discourse_result.discourse_marker_coherence, 0.0);
        assert!(discourse_result.discourse_markers.is_empty());

        let topic_result = TopicCoherenceResult::default();
        assert_eq!(topic_result.topic_consistency, 0.0);
        assert!(topic_result.topics.is_empty());

        let structural_result = StructuralCoherenceResult::default();
        assert_eq!(structural_result.paragraph_coherence, 0.0);
        assert!(structural_result.paragraph_transitions.is_empty());
    }

    #[test]
    fn test_selective_analysis() -> Result<(), CoherenceError> {
        let config = CoherenceConfig {
            coherence_types: vec![CoherenceType::Entity],
            ..Default::default()
        };

        let analyzer = CoherenceAnalyzer::new(config);
        let text = "The cat sat on the mat. The cat was comfortable.";

        let result = analyzer.analyze_coherence(text)?;

        // Entity analysis should have run
        assert!(result.entity_coherence.entity_grid_coherence >= 0.0);

        // Other analyses should be default (not run)
        assert_eq!(result.lexical_coherence.lexical_chain_coherence, 0.0);
        assert_eq!(result.discourse_coherence.discourse_marker_coherence, 0.0);
        assert_eq!(result.topic_coherence.topic_consistency, 0.0);
        assert_eq!(result.structural_coherence.paragraph_coherence, 0.0);

        Ok(())
    }

    #[test]
    fn test_result_debugging() -> Result<(), CoherenceError> {
        let analyzer = CoherenceAnalyzer::with_default_config();
        let text = "The cat sat on the mat. The cat was comfortable.";

        let result = analyzer.analyze_coherence(text)?;

        // Test that the result can be formatted for debugging
        let debug_string = format!("{:?}", result);
        assert!(debug_string.contains("CoherenceResult"));
        assert!(debug_string.contains("overall_coherence"));

        Ok(())
    }
}
