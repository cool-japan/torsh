//! Topic coherence analysis configuration
//!
//! This module provides centralized configuration management for all topic coherence
//! analysis components. It separates configuration concerns from implementation
//! details and provides a clean, extensible configuration interface.

use serde::{Deserialize, Serialize};

/// Topic modeling approaches available in the system
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TopicModelingApproach {
    /// Simple keyword clustering based on similarity
    KeywordClustering,
    /// TF-IDF based topic extraction with term frequency analysis
    TfIdf,
    /// Latent Semantic Analysis (simplified implementation)
    LatentSemantic,
    /// Co-occurrence based modeling using word relationships
    CoOccurrence,
    /// Hierarchical topic modeling with topic trees
    Hierarchical,
    /// Dynamic topic modeling with evolution tracking
    Dynamic,
}

impl Default for TopicModelingApproach {
    fn default() -> Self {
        Self::KeywordClustering
    }
}

/// Configuration for topic extraction algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicExtractionConfig {
    /// Modeling approach to use
    pub approach: TopicModelingApproach,
    /// Minimum topic size (number of keywords)
    pub min_topic_size: usize,
    /// Maximum number of topics to extract
    pub max_topics: usize,
    /// Topic similarity threshold for clustering
    pub topic_threshold: f64,
    /// Keyword extraction sensitivity
    pub keyword_sensitivity: f64,
    /// Minimum topic prominence threshold
    pub min_topic_prominence: f64,
}

impl Default for TopicExtractionConfig {
    fn default() -> Self {
        Self {
            approach: TopicModelingApproach::default(),
            min_topic_size: 2,
            max_topics: 10,
            topic_threshold: 0.6,
            keyword_sensitivity: 0.7,
            min_topic_prominence: 0.1,
        }
    }
}

/// Configuration for similarity calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityConfig {
    /// Enable character-level similarity calculation
    pub enable_character_similarity: bool,
    /// Enable semantic similarity calculation
    pub enable_semantic_similarity: bool,
    /// Enable co-occurrence similarity calculation
    pub enable_cooccurrence_similarity: bool,
    /// Weight for character similarity in combined calculations
    pub character_similarity_weight: f64,
    /// Weight for semantic similarity in combined calculations
    pub semantic_similarity_weight: f64,
    /// Weight for co-occurrence similarity in combined calculations
    pub cooccurrence_similarity_weight: f64,
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            enable_character_similarity: true,
            enable_semantic_similarity: true,
            enable_cooccurrence_similarity: true,
            character_similarity_weight: 0.3,
            semantic_similarity_weight: 0.5,
            cooccurrence_similarity_weight: 0.2,
        }
    }
}

/// Configuration for topic analysis components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Enable topic evolution tracking
    pub track_topic_evolution: bool,
    /// Enable semantic profile analysis
    pub enable_semantic_profiling: bool,
    /// Enable quality metrics calculation
    pub calculate_quality_metrics: bool,
    /// Enable relationship analysis between topics
    pub analyze_relationships: bool,
    /// Topic overlap threshold for relationship detection
    pub topic_overlap_threshold: f64,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            track_topic_evolution: true,
            enable_semantic_profiling: true,
            calculate_quality_metrics: true,
            analyze_relationships: true,
            topic_overlap_threshold: 0.3,
        }
    }
}

/// Configuration for metrics calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Topic coherence threshold for quality assessment
    pub coherence_threshold: f64,
    /// Enable detailed metrics calculation
    pub enable_detailed_metrics: bool,
    /// Enable statistical significance testing
    pub enable_statistical_testing: bool,
    /// Confidence level for statistical tests
    pub confidence_level: f64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            coherence_threshold: 0.5,
            enable_detailed_metrics: true,
            enable_statistical_testing: true,
            confidence_level: 0.95,
        }
    }
}

/// Configuration for advanced analysis features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedAnalysisConfig {
    /// Enable hierarchical topic analysis
    pub enable_hierarchical_analysis: bool,
    /// Enable dynamic topic modeling
    pub enable_dynamic_modeling: bool,
    /// Enable topic network analysis
    pub enable_network_analysis: bool,
    /// Maximum topic depth for hierarchical analysis
    pub max_topic_depth: usize,
    /// Enable temporal analysis of topic evolution
    pub enable_temporal_analysis: bool,
}

impl Default for AdvancedAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_hierarchical_analysis: true,
            enable_dynamic_modeling: true,
            enable_network_analysis: true,
            max_topic_depth: 4,
            enable_temporal_analysis: true,
        }
    }
}

/// Comprehensive topic coherence analysis configuration
///
/// This configuration struct provides a centralized way to configure all aspects
/// of topic coherence analysis while maintaining modularity between components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicCoherenceConfig {
    /// Topic extraction configuration
    pub extraction: TopicExtractionConfig,
    /// Similarity calculation configuration
    pub similarity: SimilarityConfig,
    /// Analysis components configuration
    pub analysis: AnalysisConfig,
    /// Metrics calculation configuration
    pub metrics: MetricsConfig,
    /// Advanced analysis configuration
    pub advanced: AdvancedAnalysisConfig,
    /// Enable comprehensive analysis mode
    pub use_comprehensive_analysis: bool,
}

impl Default for TopicCoherenceConfig {
    fn default() -> Self {
        Self {
            extraction: TopicExtractionConfig::default(),
            similarity: SimilarityConfig::default(),
            analysis: AnalysisConfig::default(),
            metrics: MetricsConfig::default(),
            advanced: AdvancedAnalysisConfig::default(),
            use_comprehensive_analysis: true,
        }
    }
}

/// Builder pattern for TopicCoherenceConfig
pub struct TopicCoherenceConfigBuilder {
    config: TopicCoherenceConfig,
}

impl TopicCoherenceConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: TopicCoherenceConfig::default(),
        }
    }

    pub fn extraction_approach(mut self, approach: TopicModelingApproach) -> Self {
        self.config.extraction.approach = approach;
        self
    }

    pub fn max_topics(mut self, max_topics: usize) -> Self {
        self.config.extraction.max_topics = max_topics;
        self
    }

    pub fn topic_threshold(mut self, threshold: f64) -> Self {
        self.config.extraction.topic_threshold = threshold;
        self
    }

    pub fn enable_advanced_analysis(mut self, enable: bool) -> Self {
        self.config.use_comprehensive_analysis = enable;
        self
    }

    pub fn semantic_similarity_weight(mut self, weight: f64) -> Self {
        self.config.similarity.semantic_similarity_weight = weight;
        self
    }

    pub fn build(self) -> TopicCoherenceConfig {
        self.config
    }
}

impl Default for TopicCoherenceConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TopicCoherenceConfig::default();
        assert_eq!(config.extraction.max_topics, 10);
        assert_eq!(
            config.extraction.approach,
            TopicModelingApproach::KeywordClustering
        );
        assert!(config.use_comprehensive_analysis);
    }

    #[test]
    fn test_config_builder() {
        let config = TopicCoherenceConfigBuilder::new()
            .extraction_approach(TopicModelingApproach::TfIdf)
            .max_topics(15)
            .topic_threshold(0.8)
            .enable_advanced_analysis(false)
            .build();

        assert_eq!(config.extraction.approach, TopicModelingApproach::TfIdf);
        assert_eq!(config.extraction.max_topics, 15);
        assert_eq!(config.extraction.topic_threshold, 0.8);
        assert!(!config.use_comprehensive_analysis);
    }

    #[test]
    fn test_config_serialization() {
        let config = TopicCoherenceConfig::default();
        let serialized = serde_json::to_string(&config).expect("Should serialize");
        let deserialized: TopicCoherenceConfig =
            serde_json::from_str(&serialized).expect("Should deserialize");

        assert_eq!(
            config.extraction.max_topics,
            deserialized.extraction.max_topics
        );
        assert_eq!(config.extraction.approach, deserialized.extraction.approach);
    }
}
