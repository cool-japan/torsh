//! Configuration Management for Semantic Fluency Analysis
//!
//! This module provides comprehensive configuration management for semantic fluency
//! analysis, including hierarchical configuration structures, preset configurations,
//! and validation support for different analysis modes and use cases.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during semantic configuration
#[derive(Error, Debug)]
pub enum SemanticConfigError {
    #[error("Invalid weight configuration: {0}")]
    InvalidWeight(String),
    #[error("Invalid threshold value: {0}")]
    InvalidThreshold(String),
    #[error("Configuration validation failed: {0}")]
    ValidationError(String),
    #[error("Preset configuration not found: {0}")]
    PresetNotFound(String),
}

/// Main configuration for semantic fluency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticConfig {
    /// General semantic analysis settings
    pub general: GeneralSemanticConfig,
    /// Coherence analysis configuration
    pub coherence: CoherenceAnalysisConfig,
    /// Meaning analysis configuration
    pub meaning: MeaningAnalysisConfig,
    /// Context analysis configuration
    pub context: ContextAnalysisConfig,
    /// Relations analysis configuration
    pub relations: RelationsAnalysisConfig,
    /// Advanced analysis configuration
    pub advanced: AdvancedSemanticConfig,
}

/// General semantic analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralSemanticConfig {
    /// Enable case sensitive analysis
    pub case_sensitive: bool,
    /// Minimum word length for analysis
    pub min_word_length: usize,
    /// Maximum word length for analysis
    pub max_word_length: usize,
    /// Enable stemming/lemmatization
    pub enable_lemmatization: bool,
    /// Enable stopword filtering
    pub filter_stopwords: bool,
    /// Language code for analysis
    pub language: String,
    /// Processing timeout in seconds
    pub timeout_seconds: u64,
}

/// Coherence analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceAnalysisConfig {
    /// Weight for semantic coherence in overall score
    pub coherence_weight: f64,
    /// Minimum semantic similarity threshold
    pub similarity_threshold: f64,
    /// Context window size for coherence analysis
    pub context_window: usize,
    /// Enable field coherence analysis
    pub enable_field_coherence: bool,
    /// Enable semantic overlap calculation
    pub enable_overlap_analysis: bool,
    /// Coherence calculation method
    pub calculation_method: CoherenceMethod,
}

/// Meaning analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeaningAnalysisConfig {
    /// Weight for meaning preservation
    pub preservation_weight: f64,
    /// Weight for conceptual clarity
    pub clarity_weight: f64,
    /// Minimum conceptual consistency threshold
    pub consistency_threshold: f64,
    /// Enable conceptual depth analysis
    pub enable_depth_analysis: bool,
    /// Enable meaning drift detection
    pub enable_drift_detection: bool,
    /// Ambiguity detection sensitivity
    pub ambiguity_sensitivity: f64,
}

/// Context analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAnalysisConfig {
    /// Weight for context sensitivity
    pub context_weight: f64,
    /// Context window size
    pub window_size: usize,
    /// Enable transition analysis
    pub enable_transitions: bool,
    /// Transition smoothness threshold
    pub smoothness_threshold: f64,
    /// Enable context adaptation analysis
    pub enable_adaptation: bool,
    /// Context change detection sensitivity
    pub change_sensitivity: f64,
}

/// Relations analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationsAnalysisConfig {
    /// Enable synonymy analysis
    pub enable_synonymy: bool,
    /// Enable antonymy analysis
    pub enable_antonymy: bool,
    /// Enable hyponymy analysis
    pub enable_hyponymy: bool,
    /// Enable meronymy analysis
    pub enable_meronymy: bool,
    /// Semantic relation confidence threshold
    pub relation_threshold: f64,
    /// Enable relation network building
    pub build_networks: bool,
    /// Maximum network depth
    pub max_network_depth: usize,
}

/// Advanced semantic analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedSemanticConfig {
    /// Weight for semantic appropriateness
    pub appropriateness_weight: f64,
    /// Enable advanced semantic relation analysis
    pub enable_advanced_relations: bool,
    /// Enable discourse coherence analysis
    pub enable_discourse_analysis: bool,
    /// Enable topic coherence analysis
    pub enable_topic_analysis: bool,
    /// Enable figurative language analysis
    pub enable_figurative_analysis: bool,
    /// Enable information structure analysis
    pub enable_information_structure: bool,
    /// Semantic field coverage requirement
    pub min_semantic_coverage: f64,
    /// Advanced analysis depth level
    pub analysis_depth: AnalysisDepth,
}

/// Coherence calculation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoherenceMethod {
    /// Simple overlap-based calculation
    Overlap,
    /// Weighted semantic similarity
    WeightedSimilarity,
    /// Vector-based similarity
    VectorSimilarity,
    /// Graph-based coherence
    GraphBased,
    /// Hybrid approach combining multiple methods
    Hybrid,
}

/// Analysis depth levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisDepth {
    /// Basic semantic analysis
    Basic,
    /// Standard analysis with core features
    Standard,
    /// Advanced analysis with all features
    Advanced,
    /// Comprehensive analysis with experimental features
    Comprehensive,
}

/// Inconsistency types for semantic analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum InconsistencyType {
    /// Contradictory statements
    ContradictoryStatements,
    /// Inconsistent terminology
    InconsistentTerminology,
    /// Semantic field mixing
    SemanticFieldMixing,
    /// Conceptual drift
    ConceptualDrift,
    /// Reference inconsistency
    ReferenceInconsistency,
}

/// Coherence relation types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CoherenceRelationType {
    /// Causal relationship
    Causal,
    /// Temporal relationship
    Temporal,
    /// Contrastive relationship
    Contrastive,
    /// Additive relationship
    Additive,
    /// Elaboration relationship
    Elaboration,
    /// Background information
    Background,
}

/// Focus types for information structure analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FocusType {
    /// New information focus
    NewInformation,
    /// Contrastive focus
    Contrastive,
    /// Corrective focus
    Corrective,
    /// Selective focus
    Selective,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            general: GeneralSemanticConfig::default(),
            coherence: CoherenceAnalysisConfig::default(),
            meaning: MeaningAnalysisConfig::default(),
            context: ContextAnalysisConfig::default(),
            relations: RelationsAnalysisConfig::default(),
            advanced: AdvancedSemanticConfig::default(),
        }
    }
}

impl Default for GeneralSemanticConfig {
    fn default() -> Self {
        Self {
            case_sensitive: false,
            min_word_length: 2,
            max_word_length: 50,
            enable_lemmatization: true,
            filter_stopwords: true,
            language: "en".to_string(),
            timeout_seconds: 300,
        }
    }
}

impl Default for CoherenceAnalysisConfig {
    fn default() -> Self {
        Self {
            coherence_weight: 0.25,
            similarity_threshold: 0.5,
            context_window: 3,
            enable_field_coherence: true,
            enable_overlap_analysis: true,
            calculation_method: CoherenceMethod::WeightedSimilarity,
        }
    }
}

impl Default for MeaningAnalysisConfig {
    fn default() -> Self {
        Self {
            preservation_weight: 0.20,
            clarity_weight: 0.20,
            consistency_threshold: 0.6,
            enable_depth_analysis: true,
            enable_drift_detection: true,
            ambiguity_sensitivity: 0.7,
        }
    }
}

impl Default for ContextAnalysisConfig {
    fn default() -> Self {
        Self {
            context_weight: 0.20,
            window_size: 3,
            enable_transitions: true,
            smoothness_threshold: 0.6,
            enable_adaptation: true,
            change_sensitivity: 0.5,
        }
    }
}

impl Default for RelationsAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_synonymy: true,
            enable_antonymy: true,
            enable_hyponymy: true,
            enable_meronymy: true,
            relation_threshold: 0.5,
            build_networks: true,
            max_network_depth: 4,
        }
    }
}

impl Default for AdvancedSemanticConfig {
    fn default() -> Self {
        Self {
            appropriateness_weight: 0.15,
            enable_advanced_relations: true,
            enable_discourse_analysis: true,
            enable_topic_analysis: true,
            enable_figurative_analysis: false,
            enable_information_structure: false,
            min_semantic_coverage: 0.6,
            analysis_depth: AnalysisDepth::Standard,
        }
    }
}

impl SemanticConfig {
    /// Create a minimal configuration for basic semantic analysis
    pub fn minimal() -> Self {
        Self {
            general: GeneralSemanticConfig {
                case_sensitive: false,
                min_word_length: 3,
                max_word_length: 30,
                enable_lemmatization: false,
                filter_stopwords: false,
                language: "en".to_string(),
                timeout_seconds: 60,
            },
            coherence: CoherenceAnalysisConfig {
                coherence_weight: 0.5,
                similarity_threshold: 0.6,
                context_window: 2,
                enable_field_coherence: false,
                enable_overlap_analysis: true,
                calculation_method: CoherenceMethod::Overlap,
            },
            meaning: MeaningAnalysisConfig {
                preservation_weight: 0.3,
                clarity_weight: 0.2,
                consistency_threshold: 0.5,
                enable_depth_analysis: false,
                enable_drift_detection: false,
                ambiguity_sensitivity: 0.5,
            },
            context: ContextAnalysisConfig {
                context_weight: 0.2,
                window_size: 2,
                enable_transitions: false,
                smoothness_threshold: 0.5,
                enable_adaptation: false,
                change_sensitivity: 0.3,
            },
            relations: RelationsAnalysisConfig {
                enable_synonymy: true,
                enable_antonymy: false,
                enable_hyponymy: false,
                enable_meronymy: false,
                relation_threshold: 0.6,
                build_networks: false,
                max_network_depth: 2,
            },
            advanced: AdvancedSemanticConfig {
                appropriateness_weight: 0.0,
                enable_advanced_relations: false,
                enable_discourse_analysis: false,
                enable_topic_analysis: false,
                enable_figurative_analysis: false,
                enable_information_structure: false,
                min_semantic_coverage: 0.3,
                analysis_depth: AnalysisDepth::Basic,
            },
        }
    }

    /// Create a comprehensive configuration for advanced semantic analysis
    pub fn comprehensive() -> Self {
        Self {
            general: GeneralSemanticConfig {
                case_sensitive: true,
                min_word_length: 1,
                max_word_length: 100,
                enable_lemmatization: true,
                filter_stopwords: true,
                language: "en".to_string(),
                timeout_seconds: 600,
            },
            coherence: CoherenceAnalysisConfig {
                coherence_weight: 0.3,
                similarity_threshold: 0.4,
                context_window: 5,
                enable_field_coherence: true,
                enable_overlap_analysis: true,
                calculation_method: CoherenceMethod::Hybrid,
            },
            meaning: MeaningAnalysisConfig {
                preservation_weight: 0.25,
                clarity_weight: 0.25,
                consistency_threshold: 0.7,
                enable_depth_analysis: true,
                enable_drift_detection: true,
                ambiguity_sensitivity: 0.8,
            },
            context: ContextAnalysisConfig {
                context_weight: 0.25,
                window_size: 5,
                enable_transitions: true,
                smoothness_threshold: 0.7,
                enable_adaptation: true,
                change_sensitivity: 0.7,
            },
            relations: RelationsAnalysisConfig {
                enable_synonymy: true,
                enable_antonymy: true,
                enable_hyponymy: true,
                enable_meronymy: true,
                relation_threshold: 0.4,
                build_networks: true,
                max_network_depth: 6,
            },
            advanced: AdvancedSemanticConfig {
                appropriateness_weight: 0.2,
                enable_advanced_relations: true,
                enable_discourse_analysis: true,
                enable_topic_analysis: true,
                enable_figurative_analysis: true,
                enable_information_structure: true,
                min_semantic_coverage: 0.8,
                analysis_depth: AnalysisDepth::Comprehensive,
            },
        }
    }

    /// Create configuration optimized for academic text analysis
    pub fn for_academic_text() -> Self {
        Self {
            general: GeneralSemanticConfig {
                case_sensitive: true,
                min_word_length: 3,
                max_word_length: 50,
                enable_lemmatization: true,
                filter_stopwords: false,
                language: "en".to_string(),
                timeout_seconds: 300,
            },
            coherence: CoherenceAnalysisConfig {
                coherence_weight: 0.35,
                similarity_threshold: 0.6,
                context_window: 4,
                enable_field_coherence: true,
                enable_overlap_analysis: true,
                calculation_method: CoherenceMethod::GraphBased,
            },
            meaning: MeaningAnalysisConfig {
                preservation_weight: 0.3,
                clarity_weight: 0.25,
                consistency_threshold: 0.75,
                enable_depth_analysis: true,
                enable_drift_detection: true,
                ambiguity_sensitivity: 0.8,
            },
            context: ContextAnalysisConfig {
                context_weight: 0.2,
                window_size: 4,
                enable_transitions: true,
                smoothness_threshold: 0.7,
                enable_adaptation: true,
                change_sensitivity: 0.6,
            },
            relations: RelationsAnalysisConfig {
                enable_synonymy: true,
                enable_antonymy: true,
                enable_hyponymy: true,
                enable_meronymy: true,
                relation_threshold: 0.5,
                build_networks: true,
                max_network_depth: 5,
            },
            advanced: AdvancedSemanticConfig {
                appropriateness_weight: 0.15,
                enable_advanced_relations: true,
                enable_discourse_analysis: true,
                enable_topic_analysis: true,
                enable_figurative_analysis: false,
                enable_information_structure: true,
                min_semantic_coverage: 0.7,
                analysis_depth: AnalysisDepth::Advanced,
            },
        }
    }

    /// Create configuration optimized for creative writing analysis
    pub fn for_creative_writing() -> Self {
        Self {
            general: GeneralSemanticConfig {
                case_sensitive: false,
                min_word_length: 1,
                max_word_length: 100,
                enable_lemmatization: false,
                filter_stopwords: false,
                language: "en".to_string(),
                timeout_seconds: 300,
            },
            coherence: CoherenceAnalysisConfig {
                coherence_weight: 0.2,
                similarity_threshold: 0.4,
                context_window: 5,
                enable_field_coherence: true,
                enable_overlap_analysis: true,
                calculation_method: CoherenceMethod::VectorSimilarity,
            },
            meaning: MeaningAnalysisConfig {
                preservation_weight: 0.15,
                clarity_weight: 0.15,
                consistency_threshold: 0.5,
                enable_depth_analysis: true,
                enable_drift_detection: true,
                ambiguity_sensitivity: 0.6,
            },
            context: ContextAnalysisConfig {
                context_weight: 0.3,
                window_size: 5,
                enable_transitions: true,
                smoothness_threshold: 0.5,
                enable_adaptation: true,
                change_sensitivity: 0.8,
            },
            relations: RelationsAnalysisConfig {
                enable_synonymy: true,
                enable_antonymy: true,
                enable_hyponymy: true,
                enable_meronymy: true,
                relation_threshold: 0.4,
                build_networks: true,
                max_network_depth: 4,
            },
            advanced: AdvancedSemanticConfig {
                appropriateness_weight: 0.35,
                enable_advanced_relations: true,
                enable_discourse_analysis: true,
                enable_topic_analysis: true,
                enable_figurative_analysis: true,
                enable_information_structure: true,
                min_semantic_coverage: 0.5,
                analysis_depth: AnalysisDepth::Comprehensive,
            },
        }
    }

    /// Validate configuration values
    pub fn validate(&self) -> Result<(), SemanticConfigError> {
        // Validate weights sum to reasonable total
        let total_weight = self.coherence.coherence_weight
            + self.meaning.preservation_weight
            + self.meaning.clarity_weight
            + self.context.context_weight
            + self.advanced.appropriateness_weight;

        if total_weight > 1.2 || total_weight < 0.8 {
            return Err(SemanticConfigError::ValidationError(format!(
                "Total weights ({:.2}) should be approximately 1.0",
                total_weight
            )));
        }

        // Validate thresholds are in valid ranges
        if self.coherence.similarity_threshold < 0.0 || self.coherence.similarity_threshold > 1.0 {
            return Err(SemanticConfigError::InvalidThreshold(
                "Similarity threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        if self.meaning.consistency_threshold < 0.0 || self.meaning.consistency_threshold > 1.0 {
            return Err(SemanticConfigError::InvalidThreshold(
                "Consistency threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        // Validate window sizes
        if self.coherence.context_window == 0 {
            return Err(SemanticConfigError::ValidationError(
                "Context window must be greater than 0".to_string(),
            ));
        }

        if self.context.window_size == 0 {
            return Err(SemanticConfigError::ValidationError(
                "Context window size must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }

    /// Generate a cache key for this configuration
    pub fn cache_key(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash key configuration parameters that affect analysis results
        self.coherence.coherence_weight.to_bits().hash(&mut hasher);
        self.meaning.preservation_weight.to_bits().hash(&mut hasher);
        self.context.context_weight.to_bits().hash(&mut hasher);
        self.coherence
            .similarity_threshold
            .to_bits()
            .hash(&mut hasher);
        self.coherence.context_window.hash(&mut hasher);
        self.relations.enable_synonymy.hash(&mut hasher);
        self.relations.enable_antonymy.hash(&mut hasher);
        self.advanced.enable_advanced_relations.hash(&mut hasher);

        format!("{:x}", hasher.finish())
    }
}

/// Builder for semantic configuration
pub struct SemanticConfigBuilder {
    config: SemanticConfig,
}

impl SemanticConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: SemanticConfig::default(),
        }
    }

    /// Set coherence weight
    pub fn coherence_weight(mut self, weight: f64) -> Self {
        self.config.coherence.coherence_weight = weight;
        self
    }

    /// Set meaning preservation weight
    pub fn preservation_weight(mut self, weight: f64) -> Self {
        self.config.meaning.preservation_weight = weight;
        self
    }

    /// Set context sensitivity weight
    pub fn context_weight(mut self, weight: f64) -> Self {
        self.config.context.context_weight = weight;
        self
    }

    /// Set similarity threshold
    pub fn similarity_threshold(mut self, threshold: f64) -> Self {
        self.config.coherence.similarity_threshold = threshold;
        self
    }

    /// Enable advanced relations analysis
    pub fn enable_advanced_relations(mut self, enable: bool) -> Self {
        self.config.advanced.enable_advanced_relations = enable;
        self
    }

    /// Set analysis depth
    pub fn analysis_depth(mut self, depth: AnalysisDepth) -> Self {
        self.config.advanced.analysis_depth = depth;
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<SemanticConfig, SemanticConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for SemanticConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SemanticConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_minimal_config() {
        let config = SemanticConfig::minimal();
        assert!(config.validate().is_ok());
        assert_eq!(config.advanced.analysis_depth, AnalysisDepth::Basic);
    }

    #[test]
    fn test_comprehensive_config() {
        let config = SemanticConfig::comprehensive();
        assert!(config.validate().is_ok());
        assert_eq!(config.advanced.analysis_depth, AnalysisDepth::Comprehensive);
        assert!(config.advanced.enable_figurative_analysis);
    }

    #[test]
    fn test_academic_config() {
        let config = SemanticConfig::for_academic_text();
        assert!(config.validate().is_ok());
        assert!(config.general.case_sensitive);
        assert!(config.meaning.enable_depth_analysis);
    }

    #[test]
    fn test_creative_config() {
        let config = SemanticConfig::for_creative_writing();
        assert!(config.validate().is_ok());
        assert!(config.advanced.enable_figurative_analysis);
        assert_eq!(config.context.context_weight, 0.3);
    }

    #[test]
    fn test_config_builder() {
        let config = SemanticConfigBuilder::new()
            .coherence_weight(0.4)
            .similarity_threshold(0.7)
            .enable_advanced_relations(true)
            .analysis_depth(AnalysisDepth::Advanced)
            .build();

        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.coherence.coherence_weight, 0.4);
        assert_eq!(config.coherence.similarity_threshold, 0.7);
        assert!(config.advanced.enable_advanced_relations);
    }

    #[test]
    fn test_invalid_weights() {
        let mut config = SemanticConfig::default();
        config.coherence.coherence_weight = 2.0; // Invalid weight
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_threshold() {
        let mut config = SemanticConfig::default();
        config.coherence.similarity_threshold = 1.5; // Invalid threshold
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cache_key_generation() {
        let config1 = SemanticConfig::default();
        let config2 = SemanticConfig::default();
        let config3 = SemanticConfig::minimal();

        assert_eq!(config1.cache_key(), config2.cache_key());
        assert_ne!(config1.cache_key(), config3.cache_key());
    }

    #[test]
    fn test_enum_serialization() {
        let config = SemanticConfig::comprehensive();
        let serialized = serde_json::to_string(&config);
        assert!(serialized.is_ok());

        let deserialized: Result<SemanticConfig, _> = serde_json::from_str(&serialized.unwrap());
        assert!(deserialized.is_ok());
    }
}
