//! Modern modular topic coherence analysis system
//!
//! This module provides a comprehensive, modular approach to topic coherence analysis.
//! It replaces the monolithic implementation with a clean, extensible architecture
//! that separates concerns while providing enhanced functionality.
//!
//! # Architecture
//!
//! The modular system consists of:
//!
//! - **Configuration Management**: Centralized, typed configuration with builder patterns
//! - **Similarity Calculations**: Multiple similarity algorithms with configurable weights
//! - **Topic Extraction**: Pluggable extraction algorithms (TF-IDF, clustering, LSA, etc.)
//! - **Metrics Calculation**: Comprehensive metrics with statistical analysis
//! - **Results**: Structured, serializable results with detailed analysis
//!
//! # Usage
//!
//! ## Basic Usage
//!
//! ```rust
//! use crate::metrics::coherence::topic_coherence::{
//!     TopicCoherenceAnalyzer, TopicCoherenceConfig
//! };
//!
//! let analyzer = TopicCoherenceAnalyzer::new(TopicCoherenceConfig::default());
//! let result = analyzer.analyze_topic_coherence("Your text here...")?;
//!
//! println!("Topic consistency: {:.3}", result.topic_consistency);
//! println!("Found {} topics", result.topics.len());
//! ```
//!
//! ## Advanced Configuration
//!
//! ```rust
//! use crate::metrics::coherence::topic_coherence::{
//!     TopicCoherenceAnalyzer, TopicCoherenceConfigBuilder,
//!     config::TopicModelingApproach
//! };
//!
//! let config = TopicCoherenceConfigBuilder::new()
//!     .extraction_approach(TopicModelingApproach::TfIdf)
//!     .max_topics(15)
//!     .topic_threshold(0.7)
//!     .enable_advanced_analysis(true)
//!     .build();
//!
//! let analyzer = TopicCoherenceAnalyzer::new(config);
//! let result = analyzer.analyze_topic_coherence(text)?;
//! ```
//!
//! # Migration from Legacy API
//!
//! The legacy `TopicCoherenceAnalyzer` is still supported through backward compatibility,
//! but new code should use the modern modular API for enhanced features and performance.

pub mod config;
pub mod extraction;
pub mod metrics;
pub mod results;
pub mod similarity;

// Re-export key types for convenience
pub use config::{
    AdvancedAnalysisConfig, AnalysisConfig, MetricsConfig, SimilarityConfig, TopicCoherenceConfig,
    TopicCoherenceConfigBuilder, TopicExtractionConfig, TopicModelingApproach,
};
pub use extraction::{ExtractionError, TopicExtractionFactory, TopicExtractor};
pub use metrics::TopicCoherenceMetricsCalculator;
pub use results::{
    AdvancedTopicAnalysis, DetailedTopicMetrics, SemanticProfile, ThematicProgressionPattern,
    Topic, TopicCoherenceResult, TopicEvolution, TopicQualityMetrics, TopicRelationshipAnalysis,
    TopicTransition, TopicTransitionType,
};
pub use similarity::SimilarityCalculator;

use chrono::Utc;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Error types for topic coherence analysis
#[derive(Debug, thiserror::Error)]
pub enum TopicCoherenceError {
    #[error("Empty text provided for topic analysis")]
    EmptyText,
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
    #[error("Topic extraction failed: {0}")]
    TopicExtractionError(#[from] ExtractionError),
    #[error("Analysis error: {0}")]
    AnalysisError(String),
    #[error("Processing error: {0}")]
    ProcessingError(String),
}

/// Modern modular topic coherence analyzer
///
/// This analyzer provides a clean, modular interface for topic coherence analysis
/// with pluggable extraction algorithms, configurable similarity calculations,
/// and comprehensive metrics analysis.
pub struct TopicCoherenceAnalyzer {
    config: TopicCoherenceConfig,
    similarity_calculator: SimilarityCalculator,
    metrics_calculator: TopicCoherenceMetricsCalculator,
}

impl TopicCoherenceAnalyzer {
    /// Create a new analyzer with default configuration
    pub fn new(config: TopicCoherenceConfig) -> Self {
        let similarity_calculator = SimilarityCalculator::new(config.similarity.clone());
        let metrics_calculator = TopicCoherenceMetricsCalculator::new(
            config.metrics.clone(),
            similarity_calculator.clone(),
        );

        Self {
            config,
            similarity_calculator,
            metrics_calculator,
        }
    }

    /// Create a new analyzer with default settings
    pub fn with_default_config() -> Self {
        Self::new(TopicCoherenceConfig::default())
    }

    /// Analyze topic coherence of the given text
    pub fn analyze_topic_coherence(
        &self,
        text: &str,
    ) -> Result<TopicCoherenceResult, TopicCoherenceError> {
        let start_time = Instant::now();

        if text.trim().is_empty() {
            return Err(TopicCoherenceError::EmptyText);
        }

        // Split text into sentences
        let sentences = self.split_into_sentences(text)?;

        // Extract topics using configured approach
        let topics = self.extract_topics(&sentences)?;

        // Calculate topic transitions
        let topic_transitions = self
            .metrics_calculator
            .calculate_topic_transitions(&topics, &sentences);

        // Calculate core metrics
        let topic_consistency = self.metrics_calculator.calculate_topic_consistency(&topics);
        let topic_shift_coherence = self
            .metrics_calculator
            .calculate_topic_shift_coherence(&topic_transitions);
        let topic_development = self
            .metrics_calculator
            .calculate_topic_development(&topics, &sentences);
        let thematic_unity = self.metrics_calculator.calculate_thematic_unity(&topics);

        // Calculate additional metrics
        let topic_distribution = self
            .metrics_calculator
            .calculate_topic_distribution(&topics);
        let coherence_per_topic = self
            .metrics_calculator
            .calculate_coherence_per_topic(&topics);

        // Generate detailed metrics
        let detailed_metrics = self.metrics_calculator.generate_detailed_metrics(
            &topics,
            &sentences,
            &topic_transitions,
        );

        // Analyze topic relationships
        let topic_relationships = self.metrics_calculator.analyze_topic_relationships(&topics);

        // Perform advanced analysis if enabled
        let advanced_analysis = if self.config.use_comprehensive_analysis {
            Some(self.perform_advanced_analysis(&topics, &sentences, &topic_transitions)?)
        } else {
            None
        };

        // Create analysis metadata
        let analysis_duration = start_time.elapsed().as_millis() as u64;
        let analysis_metadata = results::AnalysisMetadata {
            config_summary: format!(
                "Approach: {:?}, Max Topics: {}, Advanced: {}",
                self.config.extraction.approach,
                self.config.extraction.max_topics,
                self.config.use_comprehensive_analysis
            ),
            text_length: text.len(),
            sentences_processed: sentences.len(),
            analysis_duration_ms: analysis_duration,
            timestamp: Utc::now().to_rfc3339(),
            analysis_version: "2.0.0-modular".to_string(),
        };

        Ok(TopicCoherenceResult {
            topic_consistency,
            topic_shift_coherence,
            topic_development,
            thematic_unity,
            topics,
            topic_transitions,
            topic_distribution,
            coherence_per_topic,
            detailed_metrics,
            topic_relationships,
            advanced_analysis,
            analysis_metadata,
        })
    }

    /// Analyze topic coherence for multiple texts (batch processing)
    pub fn analyze_batch(
        &self,
        texts: &[String],
    ) -> Result<Vec<TopicCoherenceResult>, TopicCoherenceError> {
        let mut results = Vec::new();

        for text in texts {
            let result = self.analyze_topic_coherence(text)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Compare topic coherence between two texts
    pub fn compare_texts(
        &self,
        text1: &str,
        text2: &str,
    ) -> Result<TopicCoherenceComparison, TopicCoherenceError> {
        let result1 = self.analyze_topic_coherence(text1)?;
        let result2 = self.analyze_topic_coherence(text2)?;

        let consistency_difference = result2.topic_consistency - result1.topic_consistency;
        let unity_difference = result2.thematic_unity - result1.thematic_unity;
        let development_difference = result2.topic_development - result1.topic_development;

        let shared_topics = self.find_shared_topics(&result1.topics, &result2.topics);
        let unique_topics_1 = result1.topics.len() - shared_topics.len();
        let unique_topics_2 = result2.topics.len() - shared_topics.len();

        Ok(TopicCoherenceComparison {
            result1,
            result2,
            consistency_difference,
            unity_difference,
            development_difference,
            shared_topics,
            unique_topics_1,
            unique_topics_2,
        })
    }

    /// Get analyzer configuration
    pub fn get_config(&self) -> &TopicCoherenceConfig {
        &self.config
    }

    /// Update analyzer configuration
    pub fn update_config(&mut self, config: TopicCoherenceConfig) {
        self.config = config;
        self.similarity_calculator = SimilarityCalculator::new(self.config.similarity.clone());
        self.metrics_calculator = TopicCoherenceMetricsCalculator::new(
            self.config.metrics.clone(),
            self.similarity_calculator.clone(),
        );
    }

    // Private implementation methods

    fn split_into_sentences(&self, text: &str) -> Result<Vec<String>, TopicCoherenceError> {
        let sentences: Vec<String> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.len() > 10)
            .collect();

        if sentences.is_empty() {
            return Err(TopicCoherenceError::ProcessingError(
                "No valid sentences found in text".to_string(),
            ));
        }

        Ok(sentences)
    }

    fn extract_topics(&self, sentences: &[String]) -> Result<Vec<Topic>, TopicCoherenceError> {
        let extractor = TopicExtractionFactory::create_extractor(
            &self.config.extraction,
            self.similarity_calculator.clone(),
        );

        let topics = extractor.extract_topics(sentences)?;
        Ok(topics)
    }

    fn perform_advanced_analysis(
        &self,
        topics: &[Topic],
        sentences: &[String],
        transitions: &[TopicTransition],
    ) -> Result<AdvancedTopicAnalysis, TopicCoherenceError> {
        let hierarchical_structure = self.build_hierarchical_structure(topics);
        let dynamic_patterns = self.identify_dynamic_patterns(topics, transitions);
        let network_characteristics = self.analyze_network_characteristics(topics);
        let progression_pattern = self.identify_progression_pattern(topics, transitions);
        let temporal_dynamics = self.calculate_temporal_dynamics(topics, sentences);
        let cross_topic_influences = self.calculate_cross_topic_influences(topics);

        Ok(AdvancedTopicAnalysis {
            hierarchical_structure,
            dynamic_patterns,
            network_characteristics,
            progression_pattern,
            temporal_dynamics,
            cross_topic_influences,
        })
    }

    fn build_hierarchical_structure(&self, topics: &[Topic]) -> HashMap<String, Vec<String>> {
        let mut structure = HashMap::new();

        // Group topics by hierarchical level
        for topic in topics {
            if topic.hierarchical_level == 0 {
                // Main topics
                let subtopics: Vec<String> = topics
                    .iter()
                    .filter(|t| {
                        t.hierarchical_level > 0
                            && self
                                .similarity_calculator
                                .keyword_set_similarity(&topic.keywords, &t.keywords)
                                > 0.3
                    })
                    .map(|t| t.topic_id.clone())
                    .collect();

                structure.insert(topic.topic_id.clone(), subtopics);
            }
        }

        structure
    }

    fn identify_dynamic_patterns(
        &self,
        topics: &[Topic],
        transitions: &[TopicTransition],
    ) -> Vec<String> {
        let mut patterns = Vec::new();

        // Analyze evolution patterns
        let evolution_patterns: HashSet<String> = topics
            .iter()
            .map(|t| t.evolution.evolution_pattern.clone())
            .collect();

        patterns.extend(evolution_patterns);

        // Analyze transition patterns
        let smooth_transitions = transitions
            .iter()
            .filter(|t| matches!(t.transition_type, TopicTransitionType::Smooth))
            .count();

        let total_transitions = transitions.len();

        if total_transitions > 0 {
            let smooth_ratio = smooth_transitions as f64 / total_transitions as f64;
            if smooth_ratio > 0.7 {
                patterns.push("smooth_progression".to_string());
            } else if smooth_ratio < 0.3 {
                patterns.push("abrupt_shifts".to_string());
            } else {
                patterns.push("mixed_transitions".to_string());
            }
        }

        patterns
    }

    fn analyze_network_characteristics(&self, topics: &[Topic]) -> HashMap<String, f64> {
        let mut characteristics = HashMap::new();

        let total_relationships: usize = topics.iter().map(|t| t.relationships.len()).sum();

        let average_connections = if !topics.is_empty() {
            total_relationships as f64 / topics.len() as f64
        } else {
            0.0
        };

        characteristics.insert("average_connections".to_string(), average_connections);

        // Calculate network density
        let possible_connections = if topics.len() > 1 {
            topics.len() * (topics.len() - 1)
        } else {
            1
        };

        let density = total_relationships as f64 / possible_connections as f64;
        characteristics.insert("network_density".to_string(), density);

        // Find central topics (highest connectivity)
        if let Some(max_connections) = topics.iter().map(|t| t.relationships.len()).max() {
            characteristics.insert("max_connections".to_string(), max_connections as f64);
        }

        characteristics
    }

    fn identify_progression_pattern(
        &self,
        topics: &[Topic],
        transitions: &[TopicTransition],
    ) -> ThematicProgressionPattern {
        if topics.len() <= 1 {
            return ThematicProgressionPattern::Linear;
        }

        // Analyze return patterns
        let return_transitions = transitions
            .iter()
            .filter(|t| matches!(t.transition_type, TopicTransitionType::Return))
            .count();

        if return_transitions > transitions.len() / 3 {
            return ThematicProgressionPattern::Spiral;
        }

        // Analyze hierarchical structure
        let hierarchical_levels: HashSet<usize> =
            topics.iter().map(|t| t.hierarchical_level).collect();

        if hierarchical_levels.len() > 2 {
            return ThematicProgressionPattern::Hierarchical;
        }

        // Analyze network complexity
        let high_connectivity_topics = topics.iter().filter(|t| t.relationships.len() > 2).count();

        if high_connectivity_topics > topics.len() / 2 {
            return ThematicProgressionPattern::Network;
        }

        // Default to linear
        ThematicProgressionPattern::Linear
    }

    fn calculate_temporal_dynamics(&self, topics: &[Topic], sentences: &[String]) -> Vec<f64> {
        let windows = 10;
        let window_size = sentences.len() / windows;
        let mut dynamics = Vec::new();

        for i in 0..windows {
            let window_start = i * window_size;
            let window_end = ((i + 1) * window_size).min(sentences.len());

            if window_start < sentences.len() {
                let window_sentences = &sentences[window_start..window_end];
                let window_intensity =
                    self.calculate_window_topic_intensity(topics, window_sentences);
                dynamics.push(window_intensity);
            }
        }

        dynamics
    }

    fn calculate_window_topic_intensity(
        &self,
        topics: &[Topic],
        window_sentences: &[String],
    ) -> f64 {
        let window_text = window_sentences.join(" ").to_lowercase();
        let mut total_intensity = 0.0;

        for topic in topics {
            let topic_intensity = topic
                .keywords
                .iter()
                .map(|keyword| window_text.matches(&keyword.to_lowercase()).count())
                .sum::<usize>() as f64;

            total_intensity += topic_intensity;
        }

        let total_words = window_sentences
            .iter()
            .map(|s| s.split_whitespace().count())
            .sum::<usize>();

        if total_words > 0 {
            total_intensity / total_words as f64
        } else {
            0.0
        }
    }

    fn calculate_cross_topic_influences(
        &self,
        topics: &[Topic],
    ) -> HashMap<String, HashMap<String, f64>> {
        let mut influences = HashMap::new();

        for topic in topics {
            let mut topic_influences = HashMap::new();

            for other_topic in topics {
                if topic.topic_id != other_topic.topic_id {
                    let influence = self
                        .similarity_calculator
                        .keyword_set_similarity(&topic.keywords, &other_topic.keywords);

                    topic_influences.insert(other_topic.topic_id.clone(), influence);
                }
            }

            influences.insert(topic.topic_id.clone(), topic_influences);
        }

        influences
    }

    fn find_shared_topics(&self, topics1: &[Topic], topics2: &[Topic]) -> Vec<(String, String)> {
        let mut shared = Vec::new();

        for topic1 in topics1 {
            for topic2 in topics2 {
                let similarity = self
                    .similarity_calculator
                    .keyword_set_similarity(&topic1.keywords, &topic2.keywords);

                if similarity > 0.5 {
                    shared.push((topic1.topic_id.clone(), topic2.topic_id.clone()));
                }
            }
        }

        shared
    }
}

impl Default for TopicCoherenceAnalyzer {
    fn default() -> Self {
        Self::with_default_config()
    }
}

/// Result of comparing topic coherence between two texts
#[derive(Debug)]
pub struct TopicCoherenceComparison {
    pub result1: TopicCoherenceResult,
    pub result2: TopicCoherenceResult,
    pub consistency_difference: f64,
    pub unity_difference: f64,
    pub development_difference: f64,
    pub shared_topics: Vec<(String, String)>,
    pub unique_topics_1: usize,
    pub unique_topics_2: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyzer_creation() {
        let analyzer = TopicCoherenceAnalyzer::with_default_config();
        let config = analyzer.get_config();
        assert!(config.use_comprehensive_analysis);
    }

    #[test]
    fn test_basic_analysis() {
        let analyzer = TopicCoherenceAnalyzer::with_default_config();
        let text = "Computer software development is important. Programming requires skill and practice. Software engineering involves many disciplines. Code quality matters in development.";

        let result = analyzer.analyze_topic_coherence(text);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert!(analysis.topic_consistency >= 0.0);
        assert!(analysis.thematic_unity >= 0.0);
        assert!(!analysis.topics.is_empty());
    }

    #[test]
    fn test_empty_text_error() {
        let analyzer = TopicCoherenceAnalyzer::with_default_config();
        let result = analyzer.analyze_topic_coherence("");

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            TopicCoherenceError::EmptyText
        ));
    }

    #[test]
    fn test_custom_configuration() {
        let config = TopicCoherenceConfigBuilder::new()
            .max_topics(5)
            .topic_threshold(0.8)
            .enable_advanced_analysis(false)
            .build();

        let analyzer = TopicCoherenceAnalyzer::new(config);
        assert_eq!(analyzer.get_config().extraction.max_topics, 5);
        assert!(!analyzer.get_config().use_comprehensive_analysis);
    }

    #[test]
    fn test_batch_analysis() {
        let analyzer = TopicCoherenceAnalyzer::with_default_config();
        let texts = vec![
            "Technology is advancing rapidly.".to_string(),
            "Science helps us understand the world.".to_string(),
        ];

        let results = analyzer.analyze_batch(&texts);
        assert!(results.is_ok());

        let analysis_results = results.unwrap();
        assert_eq!(analysis_results.len(), 2);
    }

    #[test]
    fn test_text_comparison() {
        let analyzer = TopicCoherenceAnalyzer::with_default_config();
        let text1 = "Computer programming is a valuable skill for software development.";
        let text2 =
            "Software engineering requires good programming abilities and technical knowledge.";

        let comparison = analyzer.compare_texts(text1, text2);
        assert!(comparison.is_ok());

        let comp_result = comparison.unwrap();
        assert!(!comp_result.shared_topics.is_empty());
    }

    #[test]
    fn test_config_update() {
        let mut analyzer = TopicCoherenceAnalyzer::with_default_config();
        let original_max = analyzer.get_config().extraction.max_topics;

        let new_config = TopicCoherenceConfigBuilder::new().max_topics(20).build();

        analyzer.update_config(new_config);
        assert_eq!(analyzer.get_config().extraction.max_topics, 20);
        assert_ne!(analyzer.get_config().extraction.max_topics, original_max);
    }
}
