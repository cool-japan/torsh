//! Configuration system for lexical coherence analysis
//!
//! This module provides comprehensive configuration management for lexical coherence analysis,
//! including chain building parameters, semantic analysis settings, and various analysis
//! thresholds and weights.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Errors that can occur during lexical coherence analysis
#[derive(Debug, Error)]
pub enum LexicalCoherenceError {
    #[error("Empty text provided for lexical analysis")]
    EmptyText,
    #[error("Invalid lexical chain configuration: {0}")]
    InvalidConfiguration(String),
    #[error("Semantic lexicon error: {0}")]
    SemanticLexiconError(String),
    #[error("Word processing error: {0}")]
    WordProcessingError(String),
    #[error("Chain building failed: {0}")]
    ChainBuildingError(String),
    #[error("Coherence calculation failed: {0}")]
    CalculationError(String),
}

/// Semantic relationships between words in lexical chains
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Hash, Eq)]
pub enum SemanticRelationship {
    /// Words with similar meanings
    Synonymy,
    /// Hierarchical relationship (specific to general)
    Hyponymy,
    /// Part-whole relationship
    Meronymy,
    /// Opposite meanings
    Antonymy,
    /// Associated or related concepts
    Association,
    /// Sequential relationship
    Sequential,
    /// Causal relationship
    Causal,
    /// Temporal relationship
    Temporal,
    /// Morphological relationship (same root)
    Morphological,
    /// Collocational relationship
    Collocation,
}

/// Lexical chain types for different coherence patterns
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LexicalChainType {
    /// Simple lexical repetition
    Repetition,
    /// Synonymous words
    Synonymous,
    /// Superordinate-subordinate relationships
    Hierarchical,
    /// Part-whole relationships
    Meronymic,
    /// Thematic grouping
    Thematic,
    /// Morphological variants
    Morphological,
    /// Collocational patterns
    Collocational,
    /// Mixed relationship types
    Mixed,
}

/// Lexical cohesion device types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CohesionDeviceType {
    /// Exact repetition
    Repetition,
    /// Synonyms and near-synonyms
    Synonymy,
    /// General-specific relationships
    Hyponymy,
    /// Part-whole relationships
    Meronymy,
    /// Associated words
    Collocation,
    /// Opposite meanings
    Antonymy,
    /// Morphological variants
    Morphological,
    /// Bridging references
    Bridging,
}

/// Configuration for lexical coherence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalCoherenceConfig {
    /// General analysis settings
    pub general: GeneralLexicalConfig,
    /// Chain building configuration
    pub chains: ChainBuildingConfig,
    /// Semantic analysis configuration
    pub semantic: SemanticAnalysisConfig,
    /// Cohesion analysis configuration
    pub cohesion: CohesionAnalysisConfig,
    /// Advanced analysis configuration
    pub advanced: AdvancedLexicalConfig,
}

/// General configuration for lexical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralLexicalConfig {
    /// Enable caching of analysis results
    pub enable_caching: bool,
    /// Use parallel processing where possible
    pub parallel_processing: bool,
    /// Include detailed metrics in results
    pub detailed_metrics: bool,
    /// Minimum word length for analysis
    pub min_word_length: usize,
    /// Maximum analysis depth
    pub max_analysis_depth: usize,
    /// Enable debug output
    pub debug_output: bool,
}

/// Configuration for lexical chain building
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainBuildingConfig {
    /// Enable chain building
    pub enabled: bool,
    /// Minimum chain length
    pub min_chain_length: usize,
    /// Maximum chain length
    pub max_chain_length: usize,
    /// Maximum distance between chain elements
    pub max_distance: usize,
    /// Similarity threshold for chain extension
    pub similarity_threshold: f64,
    /// Use semantic relationships in chains
    pub use_semantic_relations: bool,
    /// Use morphological relationships
    pub use_morphological_relations: bool,
    /// Enable position-aware chain building
    pub position_aware: bool,
    /// Chain type weights
    pub chain_type_weights: HashMap<LexicalChainType, f64>,
}

/// Configuration for semantic analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAnalysisConfig {
    /// Enable semantic field analysis
    pub enabled: bool,
    /// Use built-in semantic lexicon
    pub use_builtin_lexicon: bool,
    /// Semantic similarity threshold
    pub similarity_threshold: f64,
    /// Maximum semantic field size
    pub max_field_size: usize,
    /// Enable morphological analysis
    pub morphological_analysis: bool,
    /// Character similarity weight
    pub character_similarity_weight: f64,
    /// Semantic similarity weight
    pub semantic_similarity_weight: f64,
    /// Morphological similarity weight
    pub morphological_similarity_weight: f64,
    /// Relationship type weights
    pub relationship_weights: HashMap<SemanticRelationship, f64>,
}

/// Configuration for cohesion analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionAnalysisConfig {
    /// Enable cohesion device analysis
    pub enabled: bool,
    /// Analyze repetition devices
    pub analyze_repetition: bool,
    /// Analyze synonymy devices
    pub analyze_synonymy: bool,
    /// Analyze hyponymy devices
    pub analyze_hyponymy: bool,
    /// Analyze meronymy devices
    pub analyze_meronymy: bool,
    /// Analyze collocation devices
    pub analyze_collocation: bool,
    /// Analyze antonymy devices
    pub analyze_antonymy: bool,
    /// Analyze morphological devices
    pub analyze_morphological: bool,
    /// Analyze bridging devices
    pub analyze_bridging: bool,
    /// Device type weights
    pub device_weights: HashMap<CohesionDeviceType, f64>,
}

/// Configuration for advanced analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedLexicalConfig {
    /// Enable advanced analysis features
    pub enabled: bool,
    /// Perform network analysis
    pub network_analysis: bool,
    /// Calculate information measures
    pub information_measures: bool,
    /// Estimate cognitive load
    pub cognitive_load_estimation: bool,
    /// Analyze discourse alignment
    pub discourse_alignment: bool,
    /// Perform temporal analysis
    pub temporal_analysis: bool,
    /// Calculate lexical diversity
    pub lexical_diversity: bool,
    /// Analyze pattern statistics
    pub pattern_statistics: bool,
}

impl Default for LexicalCoherenceConfig {
    fn default() -> Self {
        Self {
            general: GeneralLexicalConfig::default(),
            chains: ChainBuildingConfig::default(),
            semantic: SemanticAnalysisConfig::default(),
            cohesion: CohesionAnalysisConfig::default(),
            advanced: AdvancedLexicalConfig::default(),
        }
    }
}

impl Default for GeneralLexicalConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            parallel_processing: true,
            detailed_metrics: false,
            min_word_length: 3,
            max_analysis_depth: 10,
            debug_output: false,
        }
    }
}

impl Default for ChainBuildingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_chain_length: 2,
            max_chain_length: 50,
            max_distance: 10,
            similarity_threshold: 0.5,
            use_semantic_relations: true,
            use_morphological_relations: true,
            position_aware: true,
            chain_type_weights: Self::default_chain_type_weights(),
        }
    }
}

impl ChainBuildingConfig {
    fn default_chain_type_weights() -> HashMap<LexicalChainType, f64> {
        let mut weights = HashMap::new();
        weights.insert(LexicalChainType::Repetition, 1.0);
        weights.insert(LexicalChainType::Synonymous, 0.9);
        weights.insert(LexicalChainType::Hierarchical, 0.8);
        weights.insert(LexicalChainType::Meronymic, 0.7);
        weights.insert(LexicalChainType::Thematic, 0.8);
        weights.insert(LexicalChainType::Morphological, 0.6);
        weights.insert(LexicalChainType::Collocational, 0.7);
        weights.insert(LexicalChainType::Mixed, 0.5);
        weights
    }
}

impl Default for SemanticAnalysisConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            use_builtin_lexicon: true,
            similarity_threshold: 0.6,
            max_field_size: 20,
            morphological_analysis: true,
            character_similarity_weight: 0.3,
            semantic_similarity_weight: 0.5,
            morphological_similarity_weight: 0.2,
            relationship_weights: Self::default_relationship_weights(),
        }
    }
}

impl SemanticAnalysisConfig {
    fn default_relationship_weights() -> HashMap<SemanticRelationship, f64> {
        let mut weights = HashMap::new();
        weights.insert(SemanticRelationship::Synonymy, 1.0);
        weights.insert(SemanticRelationship::Hyponymy, 0.8);
        weights.insert(SemanticRelationship::Meronymy, 0.7);
        weights.insert(SemanticRelationship::Antonymy, 0.6);
        weights.insert(SemanticRelationship::Association, 0.5);
        weights.insert(SemanticRelationship::Sequential, 0.4);
        weights.insert(SemanticRelationship::Causal, 0.6);
        weights.insert(SemanticRelationship::Temporal, 0.5);
        weights.insert(SemanticRelationship::Morphological, 0.7);
        weights.insert(SemanticRelationship::Collocation, 0.6);
        weights
    }
}

impl Default for CohesionAnalysisConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            analyze_repetition: true,
            analyze_synonymy: true,
            analyze_hyponymy: true,
            analyze_meronymy: true,
            analyze_collocation: true,
            analyze_antonymy: false,
            analyze_morphological: true,
            analyze_bridging: false,
            device_weights: Self::default_device_weights(),
        }
    }
}

impl CohesionAnalysisConfig {
    fn default_device_weights() -> HashMap<CohesionDeviceType, f64> {
        let mut weights = HashMap::new();
        weights.insert(CohesionDeviceType::Repetition, 1.0);
        weights.insert(CohesionDeviceType::Synonymy, 0.9);
        weights.insert(CohesionDeviceType::Hyponymy, 0.8);
        weights.insert(CohesionDeviceType::Meronymy, 0.7);
        weights.insert(CohesionDeviceType::Collocation, 0.6);
        weights.insert(CohesionDeviceType::Antonymy, 0.5);
        weights.insert(CohesionDeviceType::Morphological, 0.7);
        weights.insert(CohesionDeviceType::Bridging, 0.4);
        weights
    }
}

impl Default for AdvancedLexicalConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            network_analysis: false,
            information_measures: false,
            cognitive_load_estimation: false,
            discourse_alignment: false,
            temporal_analysis: false,
            lexical_diversity: false,
            pattern_statistics: false,
        }
    }
}

impl LexicalCoherenceConfig {
    /// Create a minimal configuration for basic analysis
    pub fn minimal() -> Self {
        Self {
            general: GeneralLexicalConfig {
                enable_caching: false,
                parallel_processing: false,
                detailed_metrics: false,
                min_word_length: 3,
                max_analysis_depth: 3,
                debug_output: false,
            },
            chains: ChainBuildingConfig {
                enabled: true,
                min_chain_length: 2,
                max_chain_length: 10,
                max_distance: 5,
                similarity_threshold: 0.7,
                use_semantic_relations: false,
                use_morphological_relations: false,
                position_aware: false,
                chain_type_weights: ChainBuildingConfig::default_chain_type_weights(),
            },
            semantic: SemanticAnalysisConfig {
                enabled: false,
                use_builtin_lexicon: false,
                similarity_threshold: 0.8,
                max_field_size: 5,
                morphological_analysis: false,
                character_similarity_weight: 1.0,
                semantic_similarity_weight: 0.0,
                morphological_similarity_weight: 0.0,
                relationship_weights: SemanticAnalysisConfig::default_relationship_weights(),
            },
            cohesion: CohesionAnalysisConfig {
                enabled: true,
                analyze_repetition: true,
                analyze_synonymy: false,
                analyze_hyponymy: false,
                analyze_meronymy: false,
                analyze_collocation: false,
                analyze_antonymy: false,
                analyze_morphological: false,
                analyze_bridging: false,
                device_weights: CohesionAnalysisConfig::default_device_weights(),
            },
            advanced: AdvancedLexicalConfig {
                enabled: false,
                network_analysis: false,
                information_measures: false,
                cognitive_load_estimation: false,
                discourse_alignment: false,
                temporal_analysis: false,
                lexical_diversity: false,
                pattern_statistics: false,
            },
        }
    }

    /// Create a comprehensive configuration for in-depth analysis
    pub fn comprehensive() -> Self {
        Self {
            general: GeneralLexicalConfig {
                enable_caching: true,
                parallel_processing: true,
                detailed_metrics: true,
                min_word_length: 2,
                max_analysis_depth: 15,
                debug_output: false,
            },
            chains: ChainBuildingConfig {
                enabled: true,
                min_chain_length: 2,
                max_chain_length: 100,
                max_distance: 20,
                similarity_threshold: 0.3,
                use_semantic_relations: true,
                use_morphological_relations: true,
                position_aware: true,
                chain_type_weights: ChainBuildingConfig::default_chain_type_weights(),
            },
            semantic: SemanticAnalysisConfig {
                enabled: true,
                use_builtin_lexicon: true,
                similarity_threshold: 0.4,
                max_field_size: 50,
                morphological_analysis: true,
                character_similarity_weight: 0.3,
                semantic_similarity_weight: 0.5,
                morphological_similarity_weight: 0.2,
                relationship_weights: SemanticAnalysisConfig::default_relationship_weights(),
            },
            cohesion: CohesionAnalysisConfig {
                enabled: true,
                analyze_repetition: true,
                analyze_synonymy: true,
                analyze_hyponymy: true,
                analyze_meronymy: true,
                analyze_collocation: true,
                analyze_antonymy: true,
                analyze_morphological: true,
                analyze_bridging: true,
                device_weights: CohesionAnalysisConfig::default_device_weights(),
            },
            advanced: AdvancedLexicalConfig {
                enabled: true,
                network_analysis: true,
                information_measures: true,
                cognitive_load_estimation: true,
                discourse_alignment: true,
                temporal_analysis: true,
                lexical_diversity: true,
                pattern_statistics: true,
            },
        }
    }

    /// Create configuration optimized for academic papers
    pub fn for_academic_papers() -> Self {
        let mut config = Self::comprehensive();

        // Academic papers have rich lexical chains
        config.chains.min_chain_length = 3;
        config.chains.similarity_threshold = 0.4;

        // Strong semantic analysis for academic vocabulary
        config.semantic.similarity_threshold = 0.5;
        config.semantic.max_field_size = 30;

        // Enable advanced analysis for academic discourse
        config.advanced.enabled = true;
        config.advanced.discourse_alignment = true;

        config
    }

    /// Create configuration optimized for creative writing
    pub fn for_creative_writing() -> Self {
        let mut config = Self::default();

        // Creative writing has diverse vocabulary
        config.chains.similarity_threshold = 0.4;
        config.chains.max_distance = 15;

        // Enable semantic analysis for literary devices
        config.semantic.enabled = true;
        config.semantic.morphological_analysis = true;

        // Enable temporal analysis for narrative flow
        config.advanced.temporal_analysis = true;
        config.advanced.lexical_diversity = true;

        config
    }

    /// Create configuration optimized for technical documentation
    pub fn for_technical_docs() -> Self {
        let mut config = Self::default();

        // Technical docs have repetitive terminology
        config.chains.similarity_threshold = 0.6;
        config.cohesion.analyze_repetition = true;
        config.cohesion.analyze_morphological = true;

        // Less semantic variation
        config.semantic.similarity_threshold = 0.7;

        config
    }

    /// Builder pattern for custom configurations
    pub fn builder() -> LexicalCoherenceConfigBuilder {
        LexicalCoherenceConfigBuilder::new()
    }
}

/// Builder for creating custom lexical coherence configurations
pub struct LexicalCoherenceConfigBuilder {
    config: LexicalCoherenceConfig,
}

impl LexicalCoherenceConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: LexicalCoherenceConfig::default(),
        }
    }

    pub fn general_config(mut self, general: GeneralLexicalConfig) -> Self {
        self.config.general = general;
        self
    }

    pub fn chain_config(mut self, chains: ChainBuildingConfig) -> Self {
        self.config.chains = chains;
        self
    }

    pub fn semantic_config(mut self, semantic: SemanticAnalysisConfig) -> Self {
        self.config.semantic = semantic;
        self
    }

    pub fn cohesion_config(mut self, cohesion: CohesionAnalysisConfig) -> Self {
        self.config.cohesion = cohesion;
        self
    }

    pub fn advanced_config(mut self, advanced: AdvancedLexicalConfig) -> Self {
        self.config.advanced = advanced;
        self
    }

    pub fn build(self) -> LexicalCoherenceConfig {
        self.config
    }
}
