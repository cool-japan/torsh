//! Configuration system for discourse coherence analysis
//!
//! This module provides comprehensive configuration management for discourse coherence analysis,
//! including analysis parameters, marker type definitions, rhetorical relation types, and
//! preset configurations for different document types.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Errors that can occur during discourse coherence analysis
#[derive(Debug, Error)]
pub enum DiscourseCoherenceError {
    #[error("Empty text provided for discourse analysis")]
    EmptyText,
    #[error("Invalid discourse configuration: {0}")]
    InvalidConfiguration(String),
    #[error("Discourse marker analysis error: {0}")]
    DiscourseMarkerError(String),
    #[error("Rhetorical structure analysis error: {0}")]
    RhetoricalStructureError(String),
    #[error("Cohesion analysis error: {0}")]
    CohesionAnalysisError(String),
    #[error("Transition analysis failed: {0}")]
    TransitionAnalysisError(String),
    #[error("Discourse processing error: {0}")]
    ProcessingError(String),
}

/// Types of discourse markers for different rhetorical functions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Hash, Eq)]
pub enum DiscourseMarkerType {
    /// Addition and continuation markers
    Addition,
    /// Contrast and opposition markers
    Contrast,
    /// Cause and effect markers
    Cause,
    /// Temporal and sequential markers
    Temporal,
    /// Conditional markers
    Conditional,
    /// Concession markers
    Concession,
    /// Elaboration and explanation markers
    Elaboration,
    /// Exemplification markers
    Exemplification,
    /// Summary and conclusion markers
    Summary,
    /// Emphasis markers
    Emphasis,
    /// Comparison markers
    Comparison,
    /// Alternative markers
    Alternative,
    /// Reformulation markers
    Reformulation,
}

/// Rhetorical relation types for discourse structure analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RhetoricalRelationType {
    /// Elaboration relationship
    Elaboration,
    /// Background information
    Background,
    /// Circumstance description
    Circumstance,
    /// Solutionhood relationship
    Solutionhood,
    /// Enablement relationship
    Enablement,
    /// Motivation relationship
    Motivation,
    /// Evidence relationship
    Evidence,
    /// Justify relationship
    Justify,
    /// Antithesis relationship
    Antithesis,
    /// Concession relationship
    Concession,
    /// Sequence relationship
    Sequence,
    /// Contrast relationship
    Contrast,
    /// Joint relationship (coordination)
    Joint,
    /// List relationship
    List,
    /// Restatement relationship
    Restatement,
    /// Summary relationship
    Summary,
}

/// Types of cohesive devices for text analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CohesiveDeviceType {
    /// Personal pronouns (he, she, it, they)
    PersonalPronoun,
    /// Demonstrative pronouns and determiners (this, that, these, those)
    Demonstrative,
    /// Comparative references (such, same, other, another)
    Comparative,
    /// Substitution (one, ones, so, not)
    Substitution,
    /// Ellipsis (omitted elements)
    Ellipsis,
    /// Conjunction (and, but, or, so, because)
    Conjunction,
    /// Lexical repetition (same word)
    Repetition,
    /// Synonymy (different words, similar meaning)
    Synonymy,
    /// Antonymy (opposite meanings)
    Antonymy,
    /// Hyponymy (superordinate/subordinate relationship)
    Hyponymy,
    /// Meronymy (part-whole relationship)
    Meronymy,
    /// Collocation (words that commonly occur together)
    Collocation,
}

/// Quality levels for discourse transitions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TransitionQuality {
    /// Smooth and natural transition
    Excellent,
    /// Good transition with minor issues
    Good,
    /// Acceptable transition
    Fair,
    /// Poor transition with noticeable disruption
    Poor,
    /// Very poor or absent transition
    VeryPoor,
}

/// Comprehensive configuration for discourse coherence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseCoherenceConfig {
    /// General analysis parameters
    pub general: GeneralDiscourseConfig,
    /// Discourse marker analysis configuration
    pub markers: DiscourseMarkerConfig,
    /// Rhetorical structure analysis configuration
    pub rhetorical: RhetoricalStructureConfig,
    /// Cohesion analysis configuration
    pub cohesion: CohesionAnalysisConfig,
    /// Transition analysis configuration
    pub transitions: TransitionAnalysisConfig,
    /// Information structure configuration
    pub information: InformationStructureConfig,
    /// Advanced analysis configuration
    pub advanced: AdvancedAnalysisConfig,
}

/// General configuration parameters for discourse analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralDiscourseConfig {
    /// Enable caching of analysis results
    pub enable_caching: bool,
    /// Use parallel processing where possible
    pub parallel_processing: bool,
    /// Include detailed metrics in results
    pub detailed_metrics: bool,
    /// Minimum sentence length for analysis
    pub min_sentence_length: usize,
    /// Maximum analysis depth
    pub max_analysis_depth: usize,
    /// Enable debug output
    pub debug_output: bool,
}

/// Configuration for discourse marker analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseMarkerConfig {
    /// Enable discourse marker analysis
    pub enabled: bool,
    /// Analyze multiword markers
    pub analyze_multiword_markers: bool,
    /// Include context analysis for markers
    pub include_context_analysis: bool,
    /// Minimum marker confidence threshold
    pub min_confidence_threshold: f64,
    /// Maximum context window size
    pub context_window_size: usize,
    /// Marker type weights
    pub marker_weights: HashMap<DiscourseMarkerType, f64>,
    /// Custom marker patterns
    pub custom_markers: HashMap<String, DiscourseMarkerType>,
}

/// Configuration for rhetorical structure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhetoricalStructureConfig {
    /// Enable rhetorical structure analysis
    pub enabled: bool,
    /// Build discourse tree structure
    pub build_discourse_tree: bool,
    /// Maximum tree depth
    pub max_tree_depth: usize,
    /// Minimum relation confidence
    pub min_relation_confidence: f64,
    /// Relation type weights
    pub relation_weights: HashMap<RhetoricalRelationType, f64>,
    /// Enable nucleus-satellite detection
    pub nucleus_satellite_detection: bool,
}

/// Configuration for cohesion analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionAnalysisConfig {
    /// Enable cohesion analysis
    pub enabled: bool,
    /// Analyze reference cohesion
    pub analyze_reference_cohesion: bool,
    /// Analyze lexical cohesion
    pub analyze_lexical_cohesion: bool,
    /// Analyze conjunctive cohesion
    pub analyze_conjunctive_cohesion: bool,
    /// Maximum reference distance
    pub max_reference_distance: usize,
    /// Cohesive device weights
    pub device_weights: HashMap<CohesiveDeviceType, f64>,
    /// Enable ambiguous reference detection
    pub detect_ambiguous_references: bool,
}

/// Configuration for transition analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionAnalysisConfig {
    /// Enable transition quality analysis
    pub enabled: bool,
    /// Analyze lexical overlap
    pub analyze_lexical_overlap: bool,
    /// Analyze semantic continuity
    pub analyze_semantic_continuity: bool,
    /// Analyze structural continuity
    pub analyze_structural_continuity: bool,
    /// Lexical overlap weight
    pub lexical_overlap_weight: f64,
    /// Semantic continuity weight
    pub semantic_continuity_weight: f64,
    /// Structural continuity weight
    pub structural_continuity_weight: f64,
    /// Transition marker weight
    pub transition_marker_weight: f64,
}

/// Configuration for information structure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationStructureConfig {
    /// Enable information structure analysis
    pub enabled: bool,
    /// Analyze given-new information balance
    pub analyze_given_new_balance: bool,
    /// Analyze topic-focus articulation
    pub analyze_topic_focus: bool,
    /// Analyze thematic progression
    pub analyze_thematic_progression: bool,
    /// Calculate information density
    pub calculate_information_density: bool,
    /// Information packaging analysis
    pub analyze_information_packaging: bool,
}

/// Configuration for advanced discourse analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedAnalysisConfig {
    /// Enable advanced analysis features
    pub enabled: bool,
    /// Perform cognitive processing analysis
    pub cognitive_analysis: bool,
    /// Perform genre analysis
    pub genre_analysis: bool,
    /// Calculate information-theoretic measures
    pub information_theoretic_measures: bool,
    /// Build coherence networks
    pub build_coherence_networks: bool,
    /// Perform temporal coherence analysis
    pub temporal_coherence_analysis: bool,
}

impl Default for DiscourseCoherenceConfig {
    fn default() -> Self {
        Self {
            general: GeneralDiscourseConfig::default(),
            markers: DiscourseMarkerConfig::default(),
            rhetorical: RhetoricalStructureConfig::default(),
            cohesion: CohesionAnalysisConfig::default(),
            transitions: TransitionAnalysisConfig::default(),
            information: InformationStructureConfig::default(),
            advanced: AdvancedAnalysisConfig::default(),
        }
    }
}

impl Default for GeneralDiscourseConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            parallel_processing: true,
            detailed_metrics: false,
            min_sentence_length: 3,
            max_analysis_depth: 10,
            debug_output: false,
        }
    }
}

impl Default for DiscourseMarkerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            analyze_multiword_markers: true,
            include_context_analysis: true,
            min_confidence_threshold: 0.5,
            context_window_size: 3,
            marker_weights: Self::default_marker_weights(),
            custom_markers: HashMap::new(),
        }
    }
}

impl DiscourseMarkerConfig {
    fn default_marker_weights() -> HashMap<DiscourseMarkerType, f64> {
        let mut weights = HashMap::new();
        weights.insert(DiscourseMarkerType::Addition, 1.0);
        weights.insert(DiscourseMarkerType::Contrast, 1.2);
        weights.insert(DiscourseMarkerType::Cause, 1.3);
        weights.insert(DiscourseMarkerType::Temporal, 1.1);
        weights.insert(DiscourseMarkerType::Conditional, 1.2);
        weights.insert(DiscourseMarkerType::Concession, 1.4);
        weights.insert(DiscourseMarkerType::Elaboration, 1.0);
        weights.insert(DiscourseMarkerType::Exemplification, 1.1);
        weights.insert(DiscourseMarkerType::Summary, 1.3);
        weights.insert(DiscourseMarkerType::Emphasis, 1.1);
        weights.insert(DiscourseMarkerType::Comparison, 1.2);
        weights.insert(DiscourseMarkerType::Alternative, 1.1);
        weights.insert(DiscourseMarkerType::Reformulation, 1.2);
        weights
    }
}

impl Default for RhetoricalStructureConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            build_discourse_tree: false,
            max_tree_depth: 5,
            min_relation_confidence: 0.6,
            relation_weights: Self::default_relation_weights(),
            nucleus_satellite_detection: false,
        }
    }
}

impl RhetoricalStructureConfig {
    fn default_relation_weights() -> HashMap<RhetoricalRelationType, f64> {
        let mut weights = HashMap::new();
        weights.insert(RhetoricalRelationType::Elaboration, 1.0);
        weights.insert(RhetoricalRelationType::Background, 1.1);
        weights.insert(RhetoricalRelationType::Circumstance, 1.0);
        weights.insert(RhetoricalRelationType::Solutionhood, 1.3);
        weights.insert(RhetoricalRelationType::Enablement, 1.2);
        weights.insert(RhetoricalRelationType::Motivation, 1.2);
        weights.insert(RhetoricalRelationType::Evidence, 1.4);
        weights.insert(RhetoricalRelationType::Justify, 1.3);
        weights.insert(RhetoricalRelationType::Antithesis, 1.4);
        weights.insert(RhetoricalRelationType::Concession, 1.3);
        weights.insert(RhetoricalRelationType::Sequence, 1.1);
        weights.insert(RhetoricalRelationType::Contrast, 1.2);
        weights.insert(RhetoricalRelationType::Joint, 0.9);
        weights.insert(RhetoricalRelationType::List, 0.8);
        weights.insert(RhetoricalRelationType::Restatement, 1.0);
        weights.insert(RhetoricalRelationType::Summary, 1.2);
        weights
    }
}

impl Default for CohesionAnalysisConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            analyze_reference_cohesion: true,
            analyze_lexical_cohesion: true,
            analyze_conjunctive_cohesion: true,
            max_reference_distance: 5,
            device_weights: Self::default_device_weights(),
            detect_ambiguous_references: true,
        }
    }
}

impl CohesionAnalysisConfig {
    fn default_device_weights() -> HashMap<CohesiveDeviceType, f64> {
        let mut weights = HashMap::new();
        weights.insert(CohesiveDeviceType::PersonalPronoun, 1.2);
        weights.insert(CohesiveDeviceType::Demonstrative, 1.3);
        weights.insert(CohesiveDeviceType::Comparative, 1.1);
        weights.insert(CohesiveDeviceType::Substitution, 1.0);
        weights.insert(CohesiveDeviceType::Ellipsis, 0.9);
        weights.insert(CohesiveDeviceType::Conjunction, 1.4);
        weights.insert(CohesiveDeviceType::Repetition, 0.8);
        weights.insert(CohesiveDeviceType::Synonymy, 1.2);
        weights.insert(CohesiveDeviceType::Antonymy, 1.1);
        weights.insert(CohesiveDeviceType::Hyponymy, 1.3);
        weights.insert(CohesiveDeviceType::Meronymy, 1.2);
        weights.insert(CohesiveDeviceType::Collocation, 1.0);
        weights
    }
}

impl Default for TransitionAnalysisConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            analyze_lexical_overlap: true,
            analyze_semantic_continuity: true,
            analyze_structural_continuity: true,
            lexical_overlap_weight: 0.3,
            semantic_continuity_weight: 0.4,
            structural_continuity_weight: 0.2,
            transition_marker_weight: 0.1,
        }
    }
}

impl Default for InformationStructureConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            analyze_given_new_balance: true,
            analyze_topic_focus: true,
            analyze_thematic_progression: true,
            calculate_information_density: true,
            analyze_information_packaging: true,
        }
    }
}

impl Default for AdvancedAnalysisConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cognitive_analysis: false,
            genre_analysis: false,
            information_theoretic_measures: false,
            build_coherence_networks: false,
            temporal_coherence_analysis: false,
        }
    }
}

impl DiscourseCoherenceConfig {
    /// Create a minimal configuration for basic analysis
    pub fn minimal() -> Self {
        Self {
            general: GeneralDiscourseConfig {
                enable_caching: false,
                parallel_processing: false,
                detailed_metrics: false,
                min_sentence_length: 1,
                max_analysis_depth: 3,
                debug_output: false,
            },
            markers: DiscourseMarkerConfig {
                enabled: true,
                analyze_multiword_markers: false,
                include_context_analysis: false,
                min_confidence_threshold: 0.7,
                context_window_size: 1,
                marker_weights: DiscourseMarkerConfig::default_marker_weights(),
                custom_markers: HashMap::new(),
            },
            rhetorical: RhetoricalStructureConfig {
                enabled: false,
                build_discourse_tree: false,
                max_tree_depth: 2,
                min_relation_confidence: 0.8,
                relation_weights: RhetoricalStructureConfig::default_relation_weights(),
                nucleus_satellite_detection: false,
            },
            cohesion: CohesionAnalysisConfig {
                enabled: true,
                analyze_reference_cohesion: true,
                analyze_lexical_cohesion: false,
                analyze_conjunctive_cohesion: false,
                max_reference_distance: 3,
                device_weights: CohesionAnalysisConfig::default_device_weights(),
                detect_ambiguous_references: false,
            },
            transitions: TransitionAnalysisConfig {
                enabled: true,
                analyze_lexical_overlap: true,
                analyze_semantic_continuity: false,
                analyze_structural_continuity: false,
                lexical_overlap_weight: 1.0,
                semantic_continuity_weight: 0.0,
                structural_continuity_weight: 0.0,
                transition_marker_weight: 0.0,
            },
            information: InformationStructureConfig {
                enabled: false,
                analyze_given_new_balance: false,
                analyze_topic_focus: false,
                analyze_thematic_progression: false,
                calculate_information_density: false,
                analyze_information_packaging: false,
            },
            advanced: AdvancedAnalysisConfig {
                enabled: false,
                cognitive_analysis: false,
                genre_analysis: false,
                information_theoretic_measures: false,
                build_coherence_networks: false,
                temporal_coherence_analysis: false,
            },
        }
    }

    /// Create a comprehensive configuration for in-depth analysis
    pub fn comprehensive() -> Self {
        Self {
            general: GeneralDiscourseConfig {
                enable_caching: true,
                parallel_processing: true,
                detailed_metrics: true,
                min_sentence_length: 1,
                max_analysis_depth: 15,
                debug_output: false,
            },
            markers: DiscourseMarkerConfig {
                enabled: true,
                analyze_multiword_markers: true,
                include_context_analysis: true,
                min_confidence_threshold: 0.3,
                context_window_size: 5,
                marker_weights: DiscourseMarkerConfig::default_marker_weights(),
                custom_markers: HashMap::new(),
            },
            rhetorical: RhetoricalStructureConfig {
                enabled: true,
                build_discourse_tree: true,
                max_tree_depth: 10,
                min_relation_confidence: 0.4,
                relation_weights: RhetoricalStructureConfig::default_relation_weights(),
                nucleus_satellite_detection: true,
            },
            cohesion: CohesionAnalysisConfig {
                enabled: true,
                analyze_reference_cohesion: true,
                analyze_lexical_cohesion: true,
                analyze_conjunctive_cohesion: true,
                max_reference_distance: 10,
                device_weights: CohesionAnalysisConfig::default_device_weights(),
                detect_ambiguous_references: true,
            },
            transitions: TransitionAnalysisConfig {
                enabled: true,
                analyze_lexical_overlap: true,
                analyze_semantic_continuity: true,
                analyze_structural_continuity: true,
                lexical_overlap_weight: 0.3,
                semantic_continuity_weight: 0.4,
                structural_continuity_weight: 0.2,
                transition_marker_weight: 0.1,
            },
            information: InformationStructureConfig {
                enabled: true,
                analyze_given_new_balance: true,
                analyze_topic_focus: true,
                analyze_thematic_progression: true,
                calculate_information_density: true,
                analyze_information_packaging: true,
            },
            advanced: AdvancedAnalysisConfig {
                enabled: true,
                cognitive_analysis: true,
                genre_analysis: true,
                information_theoretic_measures: true,
                build_coherence_networks: true,
                temporal_coherence_analysis: true,
            },
        }
    }

    /// Create configuration optimized for academic papers
    pub fn for_academic_papers() -> Self {
        let mut config = Self::comprehensive();

        // Academic papers have strong rhetorical structure
        config.rhetorical.enabled = true;
        config.rhetorical.build_discourse_tree = true;
        config.rhetorical.min_relation_confidence = 0.5;

        // Academic papers use many discourse markers
        config.markers.analyze_multiword_markers = true;
        config.markers.min_confidence_threshold = 0.4;

        // Strong cohesion expectations
        config.cohesion.analyze_lexical_cohesion = true;
        config.cohesion.max_reference_distance = 8;

        // Advanced analysis useful for academic writing
        config.advanced.enabled = true;
        config.advanced.genre_analysis = true;

        config
    }

    /// Create configuration optimized for narrative texts
    pub fn for_narratives() -> Self {
        let mut config = Self::default();

        // Narratives emphasize temporal coherence
        config.advanced.temporal_coherence_analysis = true;

        // Strong reference cohesion in narratives
        config.cohesion.analyze_reference_cohesion = true;
        config.cohesion.max_reference_distance = 10;

        // Temporal markers are important
        config
            .markers
            .marker_weights
            .insert(DiscourseMarkerType::Temporal, 1.5);

        // Information structure analysis less critical
        config.information.enabled = false;

        config
    }

    /// Create configuration optimized for technical documentation
    pub fn for_technical_docs() -> Self {
        let mut config = Self::default();

        // Technical docs have clear structure
        config.rhetorical.enabled = true;
        config.rhetorical.build_discourse_tree = false; // Often explicit structure

        // Clear transitions important
        config.transitions.analyze_structural_continuity = true;

        // Lexical cohesion through technical terms
        config.cohesion.analyze_lexical_cohesion = true;

        // Information packaging important
        config.information.analyze_information_packaging = true;

        config
    }

    /// Builder pattern for custom configurations
    pub fn builder() -> DiscourseCoherenceConfigBuilder {
        DiscourseCoherenceConfigBuilder::new()
    }
}

/// Builder for creating custom discourse coherence configurations
pub struct DiscourseCoherenceConfigBuilder {
    config: DiscourseCoherenceConfig,
}

impl DiscourseCoherenceConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: DiscourseCoherenceConfig::default(),
        }
    }

    pub fn general_config(mut self, general: GeneralDiscourseConfig) -> Self {
        self.config.general = general;
        self
    }

    pub fn marker_config(mut self, markers: DiscourseMarkerConfig) -> Self {
        self.config.markers = markers;
        self
    }

    pub fn rhetorical_config(mut self, rhetorical: RhetoricalStructureConfig) -> Self {
        self.config.rhetorical = rhetorical;
        self
    }

    pub fn cohesion_config(mut self, cohesion: CohesionAnalysisConfig) -> Self {
        self.config.cohesion = cohesion;
        self
    }

    pub fn transition_config(mut self, transitions: TransitionAnalysisConfig) -> Self {
        self.config.transitions = transitions;
        self
    }

    pub fn information_config(mut self, information: InformationStructureConfig) -> Self {
        self.config.information = information;
        self
    }

    pub fn advanced_config(mut self, advanced: AdvancedAnalysisConfig) -> Self {
        self.config.advanced = advanced;
        self
    }

    pub fn build(self) -> DiscourseCoherenceConfig {
        self.config
    }
}
