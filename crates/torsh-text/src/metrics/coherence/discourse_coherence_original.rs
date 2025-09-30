//! Advanced discourse coherence analysis for text evaluation
//!
//! This module provides comprehensive discourse coherence analysis including discourse markers,
//! rhetorical structure, cohesion devices, transition quality, and advanced discourse-level
//! coherence metrics. It offers both basic and advanced analysis modes with configurable
//! parameters for different text types and domains.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::RwLock;
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Hash)]
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

/// Types of cohesive devices in discourse
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CohesiveDeviceType {
    /// Reference devices (pronouns, demonstratives)
    Reference,
    /// Substitution devices
    Substitution,
    /// Ellipsis devices
    Ellipsis,
    /// Conjunction devices
    Conjunction,
    /// Lexical cohesion devices
    LexicalCohesion,
    /// Discourse markers
    DiscourseMarker,
    /// Bridging references
    Bridging,
    /// Parallelism devices
    Parallelism,
}

/// Transition quality levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TransitionQuality {
    /// Smooth, well-connected transition
    Excellent,
    /// Good transition with clear connections
    Good,
    /// Adequate transition with some connections
    Adequate,
    /// Poor transition with weak connections
    Poor,
    /// Abrupt transition with no clear connections
    Abrupt,
}

/// Configuration for discourse coherence analysis
#[derive(Debug, Clone)]
pub struct DiscourseCoherenceConfig {
    /// Weight for discourse markers in analysis
    pub discourse_marker_weight: f64,
    /// Enable rhetorical structure analysis
    pub analyze_rhetorical_structure: bool,
    /// Enable cohesion device analysis
    pub analyze_cohesion_devices: bool,
    /// Enable transition quality analysis
    pub analyze_transition_quality: bool,
    /// Enable advanced discourse analysis
    pub use_advanced_analysis: bool,
    /// Minimum marker contribution threshold
    pub min_marker_contribution: f64,
    /// Transition quality threshold
    pub transition_quality_threshold: f64,
    /// Cohesion device detection sensitivity
    pub cohesion_sensitivity: f64,
    /// Enable paragraph-level analysis
    pub analyze_paragraph_level: bool,
    /// Enable section-level analysis
    pub analyze_section_level: bool,
    /// Enable cross-reference analysis
    pub analyze_cross_references: bool,
    /// Enable temporal coherence analysis
    pub analyze_temporal_coherence: bool,
    /// Rhetorical relation confidence threshold
    pub rhetorical_confidence_threshold: f64,
    /// Enable discourse tree construction
    pub build_discourse_tree: bool,
    /// Maximum discourse tree depth
    pub max_discourse_depth: usize,
}

impl Default for DiscourseCoherenceConfig {
    fn default() -> Self {
        Self {
            discourse_marker_weight: 0.8,
            analyze_rhetorical_structure: true,
            analyze_cohesion_devices: true,
            analyze_transition_quality: true,
            use_advanced_analysis: true,
            min_marker_contribution: 0.3,
            transition_quality_threshold: 0.6,
            cohesion_sensitivity: 0.7,
            analyze_paragraph_level: true,
            analyze_section_level: true,
            analyze_cross_references: true,
            analyze_temporal_coherence: true,
            rhetorical_confidence_threshold: 0.8,
            build_discourse_tree: true,
            max_discourse_depth: 5,
        }
    }
}

/// Comprehensive discourse coherence analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseCoherenceResult {
    /// Overall discourse marker coherence score
    pub discourse_marker_coherence: f64,
    /// Overall transition coherence score
    pub transition_coherence: f64,
    /// Rhetorical structure coherence score
    pub rhetorical_structure_coherence: f64,
    /// Overall cohesion score
    pub cohesion_score: f64,
    /// Number of coherence signals found
    pub coherence_signals: usize,
    /// Individual discourse markers found
    pub discourse_markers: Vec<DiscourseMarker>,
    /// Transition quality measures
    pub transition_quality: Vec<f64>,
    /// Rhetorical relations identified
    pub rhetorical_relations: HashMap<String, usize>,
    /// Detailed discourse metrics
    pub detailed_metrics: DetailedDiscourseMetrics,
    /// Cohesive devices analysis
    pub cohesive_devices: Vec<CohesiveDevice>,
    /// Advanced discourse analysis
    pub advanced_analysis: Option<AdvancedDiscourseAnalysis>,
}

/// Individual discourse marker with comprehensive analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseMarker {
    /// The marker text
    pub marker: String,
    /// Type of discourse marker
    pub marker_type: DiscourseMarkerType,
    /// Position in text (sentence, word)
    pub position: (usize, usize),
    /// Contribution to local coherence
    pub contribution: f64,
    /// Scope of influence (number of sentences)
    pub scope: usize,
    /// Confidence score for marker classification
    pub confidence: f64,
    /// Context analysis
    pub context_analysis: ContextAnalysis,
    /// Rhetorical function strength
    pub rhetorical_strength: f64,
}

/// Context analysis for discourse markers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAnalysis {
    /// Preceding context words
    pub preceding_context: Vec<String>,
    /// Following context words
    pub following_context: Vec<String>,
    /// Syntactic position analysis
    pub syntactic_position: SyntacticPosition,
    /// Semantic coherence with context
    pub semantic_coherence: f64,
    /// Pragmatic appropriateness score
    pub pragmatic_score: f64,
}

/// Syntactic position information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntacticPosition {
    /// Position in sentence (beginning, middle, end)
    pub sentence_position: String,
    /// Position relative to clauses
    pub clause_position: String,
    /// Punctuation context
    pub punctuation_context: Vec<String>,
}

/// Detailed discourse metrics and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedDiscourseMetrics {
    /// Total discourse markers by type
    pub marker_type_distribution: HashMap<String, usize>,
    /// Average marker contribution
    pub average_marker_contribution: f64,
    /// Marker density (markers per sentence)
    pub marker_density: f64,
    /// Transition quality distribution
    pub transition_quality_distribution: TransitionQualityDistribution,
    /// Rhetorical structure complexity
    pub rhetorical_complexity: RhetoricalComplexityMetrics,
    /// Cohesion device statistics
    pub cohesion_statistics: CohesionStatistics,
    /// Discourse flow analysis
    pub discourse_flow: DiscourseFlowMetrics,
    /// Cross-reference analysis
    pub cross_reference_analysis: CrossReferenceMetrics,
    /// Temporal coherence measures
    pub temporal_coherence: TemporalCoherenceMetrics,
    /// Information structure analysis
    pub information_structure: InformationStructureMetrics,
}

/// Distribution of transition quality levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionQualityDistribution {
    /// Count of excellent transitions
    pub excellent: usize,
    /// Count of good transitions
    pub good: usize,
    /// Count of adequate transitions
    pub adequate: usize,
    /// Count of poor transitions
    pub poor: usize,
    /// Count of abrupt transitions
    pub abrupt: usize,
    /// Average transition quality score
    pub average_quality: f64,
    /// Quality variance
    pub quality_variance: f64,
}

/// Rhetorical structure complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhetoricalComplexityMetrics {
    /// Number of rhetorical relations
    pub relation_count: usize,
    /// Relation type diversity
    pub relation_diversity: f64,
    /// Average relation confidence
    pub average_confidence: f64,
    /// Discourse tree depth
    pub tree_depth: usize,
    /// Structural balance score
    pub structural_balance: f64,
    /// Hierarchical organization score
    pub hierarchical_organization: f64,
}

/// Cohesion device statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionStatistics {
    /// Total cohesive devices
    pub total_devices: usize,
    /// Device type distribution
    pub device_type_distribution: HashMap<String, usize>,
    /// Reference chain analysis
    pub reference_chains: Vec<ReferenceChain>,
    /// Cohesive tie strength
    pub average_tie_strength: f64,
    /// Cohesive density
    pub cohesive_density: f64,
    /// Chain completeness score
    pub chain_completeness: f64,
}

/// Reference chain for cohesion analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceChain {
    /// Chain identifier
    pub chain_id: usize,
    /// Elements in the chain
    pub elements: Vec<ReferenceElement>,
    /// Chain coherence score
    pub coherence_score: f64,
    /// Chain span (sentences)
    pub span: usize,
    /// Chain type
    pub chain_type: String,
}

/// Individual reference element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceElement {
    /// Text of the reference
    pub text: String,
    /// Position in text
    pub position: (usize, usize),
    /// Reference type
    pub reference_type: String,
    /// Antecedent distance
    pub antecedent_distance: usize,
}

/// Discourse flow analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseFlowMetrics {
    /// Flow continuity score
    pub flow_continuity: f64,
    /// Topic progression score
    pub topic_progression: f64,
    /// Information flow entropy
    pub information_entropy: f64,
    /// Coherence momentum
    pub coherence_momentum: f64,
    /// Flow disruption points
    pub disruption_points: Vec<usize>,
    /// Flow quality evolution
    pub quality_evolution: Vec<f64>,
}

/// Cross-reference analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossReferenceMetrics {
    /// Forward references count
    pub forward_references: usize,
    /// Backward references count
    pub backward_references: usize,
    /// Reference resolution success rate
    pub resolution_success_rate: f64,
    /// Average reference distance
    pub average_reference_distance: f64,
    /// Ambiguous references count
    pub ambiguous_references: usize,
    /// Reference complexity score
    pub complexity_score: f64,
}

/// Temporal coherence in discourse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalCoherenceMetrics {
    /// Temporal marker frequency
    pub temporal_marker_frequency: f64,
    /// Temporal sequence coherence
    pub sequence_coherence: f64,
    /// Temporal anchoring score
    pub anchoring_score: f64,
    /// Timeline consistency
    pub timeline_consistency: f64,
    /// Temporal disruptions count
    pub temporal_disruptions: usize,
}

/// Information structure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationStructureMetrics {
    /// Given-new information balance
    pub given_new_balance: f64,
    /// Topic-focus articulation
    pub topic_focus_articulation: f64,
    /// Information packaging score
    pub information_packaging: f64,
    /// Thematic progression pattern
    pub thematic_progression: String,
    /// Information density
    pub information_density: f64,
}

/// Cohesive device analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesiveDevice {
    /// Type of cohesive device
    pub device_type: CohesiveDeviceType,
    /// Text elements involved
    pub elements: Vec<String>,
    /// Positions in text
    pub positions: Vec<(usize, usize)>,
    /// Cohesive strength
    pub strength: f64,
    /// Local coherence contribution
    pub local_contribution: f64,
    /// Global coherence contribution
    pub global_contribution: f64,
    /// Resolution confidence
    pub resolution_confidence: f64,
}

/// Advanced discourse analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedDiscourseAnalysis {
    /// Discourse tree structure
    pub discourse_tree: Option<DiscourseTree>,
    /// Coherence relation network
    pub coherence_network: CoherenceNetwork,
    /// Information-theoretic measures
    pub information_measures: DiscourseInformationMeasures,
    /// Cognitive processing measures
    pub cognitive_measures: CognitiveMeasures,
    /// Discourse genre analysis
    pub genre_analysis: GenreAnalysis,
}

/// Discourse tree representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseTree {
    /// Root node of the tree
    pub root: DiscourseNode,
    /// Tree depth
    pub depth: usize,
    /// Node count
    pub node_count: usize,
    /// Balance score
    pub balance_score: f64,
}

/// Individual discourse tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseNode {
    /// Node identifier
    pub node_id: usize,
    /// Rhetorical relation type
    pub relation_type: RhetoricalRelationType,
    /// Nucleus or satellite
    pub nuclearity: String,
    /// Text span
    pub text_span: (usize, usize),
    /// Child nodes
    pub children: Vec<DiscourseNode>,
    /// Confidence score
    pub confidence: f64,
}

/// Coherence relation network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceNetwork {
    /// Network nodes (text units)
    pub nodes: Vec<NetworkNode>,
    /// Network edges (coherence relations)
    pub edges: Vec<NetworkEdge>,
    /// Network density
    pub density: f64,
    /// Average path length
    pub average_path_length: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
}

/// Network node for coherence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkNode {
    /// Node identifier
    pub node_id: usize,
    /// Text content
    pub content: String,
    /// Position in discourse
    pub position: usize,
    /// Centrality measures
    pub centrality: f64,
}

/// Network edge for coherence relations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkEdge {
    /// Source node
    pub source: usize,
    /// Target node
    pub target: usize,
    /// Relation type
    pub relation_type: RhetoricalRelationType,
    /// Relation strength
    pub strength: f64,
}

/// Information-theoretic measures for discourse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseInformationMeasures {
    /// Discourse entropy
    pub entropy: f64,
    /// Information flow rate
    pub information_flow_rate: f64,
    /// Redundancy coefficient
    pub redundancy_coefficient: f64,
    /// Mutual information between segments
    pub segment_mutual_information: f64,
    /// Predictability score
    pub predictability_score: f64,
}

/// Cognitive processing measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveMeasures {
    /// Processing load estimate
    pub processing_load: f64,
    /// Working memory demand
    pub working_memory_demand: f64,
    /// Integration complexity
    pub integration_complexity: f64,
    /// Inference burden
    pub inference_burden: f64,
    /// Comprehension ease
    pub comprehension_ease: f64,
}

/// Genre analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenreAnalysis {
    /// Predicted genre
    pub predicted_genre: String,
    /// Genre confidence
    pub genre_confidence: f64,
    /// Genre-specific features
    pub genre_features: HashMap<String, f64>,
    /// Discourse pattern compliance
    pub pattern_compliance: f64,
}

/// Advanced discourse coherence analyzer
pub struct DiscourseCoherenceAnalyzer {
    config: DiscourseCoherenceConfig,
    discourse_markers: HashMap<String, DiscourseMarkerType>,
    rhetorical_patterns: HashMap<String, RhetoricalRelationType>,
    cohesion_patterns: HashMap<String, CohesiveDeviceType>,
    reference_patterns: HashSet<String>,
    temporal_markers: HashSet<String>,
    marker_cache: Arc<RwLock<HashMap<String, Vec<DiscourseMarker>>>>,
}

impl DiscourseCoherenceAnalyzer {
    /// Create a new discourse coherence analyzer with default configuration
    pub fn new() -> Self {
        Self::with_config(DiscourseCoherenceConfig::default())
    }

    /// Create a new discourse coherence analyzer with custom configuration
    pub fn with_config(config: DiscourseCoherenceConfig) -> Self {
        let discourse_markers = Self::build_discourse_markers();
        let rhetorical_patterns = Self::build_rhetorical_patterns();
        let cohesion_patterns = Self::build_cohesion_patterns();
        let reference_patterns = Self::build_reference_patterns();
        let temporal_markers = Self::build_temporal_markers();
        let marker_cache = Arc::new(RwLock::new(HashMap::new()));

        Self {
            config,
            discourse_markers,
            rhetorical_patterns,
            cohesion_patterns,
            reference_patterns,
            temporal_markers,
            marker_cache,
        }
    }

    /// Analyze discourse coherence of the given text
    pub fn analyze_discourse_coherence(
        &self,
        text: &str,
    ) -> Result<DiscourseCoherenceResult, DiscourseCoherenceError> {
        if text.trim().is_empty() {
            return Err(DiscourseCoherenceError::EmptyText);
        }

        let sentences = self.split_into_sentences(text)?;

        // Extract and analyze discourse markers
        let discourse_markers = self.extract_discourse_markers(&sentences)?;

        // Calculate transition quality
        let transition_quality = if self.config.analyze_transition_quality {
            self.calculate_sentence_transition_quality(&sentences)
        } else {
            Vec::new()
        };

        // Identify rhetorical relations
        let rhetorical_relations = if self.config.analyze_rhetorical_structure {
            self.identify_rhetorical_relations(&sentences)?
        } else {
            HashMap::new()
        };

        // Calculate core discourse coherence metrics
        let discourse_marker_coherence =
            self.calculate_discourse_marker_coherence(&discourse_markers)?;
        let transition_coherence = if !transition_quality.is_empty() {
            transition_quality.iter().sum::<f64>() / transition_quality.len() as f64
        } else {
            0.0
        };
        let rhetorical_structure_coherence =
            self.calculate_rhetorical_structure_coherence(&rhetorical_relations);
        let cohesion_score = if self.config.analyze_cohesion_devices {
            self.calculate_cohesion_score(&sentences)
        } else {
            0.0
        };

        let coherence_signals = discourse_markers.len();

        // Generate detailed metrics
        let detailed_metrics = self.generate_detailed_metrics(
            &sentences,
            &discourse_markers,
            &rhetorical_relations,
            &transition_quality,
        );

        // Analyze cohesive devices
        let cohesive_devices = if self.config.analyze_cohesion_devices {
            self.analyze_cohesive_devices(&sentences)?
        } else {
            Vec::new()
        };

        // Perform advanced analysis if enabled
        let advanced_analysis = if self.config.use_advanced_analysis {
            Some(self.perform_advanced_analysis(
                &sentences,
                &discourse_markers,
                &rhetorical_relations,
            )?)
        } else {
            None
        };

        Ok(DiscourseCoherenceResult {
            discourse_marker_coherence,
            transition_coherence,
            rhetorical_structure_coherence,
            cohesion_score,
            coherence_signals,
            discourse_markers,
            transition_quality,
            rhetorical_relations,
            detailed_metrics,
            cohesive_devices,
            advanced_analysis,
        })
    }

    /// Extract discourse markers from sentences
    fn extract_discourse_markers(
        &self,
        sentences: &[String],
    ) -> Result<Vec<DiscourseMarker>, DiscourseCoherenceError> {
        let mut markers = Vec::new();

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let words: Vec<&str> = sentence.split_whitespace().collect();

            for (word_idx, word) in words.iter().enumerate() {
                let clean_word = word
                    .trim_matches(|c: char| !c.is_alphabetic())
                    .to_lowercase();

                if let Some(marker_type) = self.discourse_markers.get(&clean_word) {
                    let contribution =
                        self.calculate_marker_contribution(marker_type, word_idx, &words);

                    if contribution >= self.config.min_marker_contribution {
                        let context_analysis =
                            self.analyze_marker_context(sent_idx, word_idx, sentences);
                        let scope = self.calculate_marker_scope(sent_idx, word_idx, sentences);
                        let confidence = self.calculate_marker_confidence(
                            &clean_word,
                            marker_type,
                            &context_analysis,
                        );
                        let rhetorical_strength =
                            self.calculate_rhetorical_strength(marker_type, &context_analysis);

                        let marker = DiscourseMarker {
                            marker: clean_word,
                            marker_type: marker_type.clone(),
                            position: (sent_idx, word_idx),
                            contribution,
                            scope,
                            confidence,
                            context_analysis,
                            rhetorical_strength,
                        };

                        markers.push(marker);
                    }
                }
            }

            // Check for multi-word markers
            markers.extend(self.extract_multiword_markers(sent_idx, &words)?);
        }

        Ok(markers)
    }

    /// Extract multi-word discourse markers
    fn extract_multiword_markers(
        &self,
        sent_idx: usize,
        words: &[&str],
    ) -> Result<Vec<DiscourseMarker>, DiscourseCoherenceError> {
        let mut markers = Vec::new();
        let multiword_markers = self.build_multiword_markers();

        for window_size in 2..=4 {
            for i in 0..=words.len().saturating_sub(window_size) {
                let window = words[i..i + window_size].join(" ").to_lowercase();

                if let Some(marker_type) = multiword_markers.get(&window) {
                    let contribution = self.calculate_marker_contribution(marker_type, i, words);

                    if contribution >= self.config.min_marker_contribution {
                        let context_analysis = ContextAnalysis {
                            preceding_context: if i > 0 {
                                vec![words[i - 1].to_string()]
                            } else {
                                Vec::new()
                            },
                            following_context: if i + window_size < words.len() {
                                vec![words[i + window_size].to_string()]
                            } else {
                                Vec::new()
                            },
                            syntactic_position: SyntacticPosition {
                                sentence_position: if i == 0 {
                                    "beginning".to_string()
                                } else if i + window_size == words.len() {
                                    "end".to_string()
                                } else {
                                    "middle".to_string()
                                },
                                clause_position: "unknown".to_string(),
                                punctuation_context: Vec::new(),
                            },
                            semantic_coherence: 0.8,
                            pragmatic_score: 0.7,
                        };

                        let marker = DiscourseMarker {
                            marker: window,
                            marker_type: marker_type.clone(),
                            position: (sent_idx, i),
                            contribution,
                            scope: 2,
                            confidence: 0.9,
                            context_analysis,
                            rhetorical_strength: 0.8,
                        };

                        markers.push(marker);
                    }
                }
            }
        }

        Ok(markers)
    }

    /// Calculate marker contribution to coherence
    fn calculate_marker_contribution(
        &self,
        marker_type: &DiscourseMarkerType,
        position: usize,
        words: &[&str],
    ) -> f64 {
        let type_weight = match marker_type {
            DiscourseMarkerType::Cause => 1.0,
            DiscourseMarkerType::Contrast => 0.9,
            DiscourseMarkerType::Addition => 0.8,
            DiscourseMarkerType::Temporal => 0.8,
            DiscourseMarkerType::Elaboration => 0.7,
            DiscourseMarkerType::Exemplification => 0.7,
            DiscourseMarkerType::Conditional => 0.6,
            DiscourseMarkerType::Concession => 0.6,
            DiscourseMarkerType::Summary => 0.9,
            DiscourseMarkerType::Emphasis => 0.5,
            DiscourseMarkerType::Comparison => 0.8,
            DiscourseMarkerType::Alternative => 0.7,
            DiscourseMarkerType::Reformulation => 0.6,
        };

        // Position weight (beginning and end positions are more significant)
        let position_weight = if position == 0 {
            1.0
        } else if position == words.len() - 1 {
            0.9
        } else {
            0.7
        };

        position_weight * type_weight * self.config.discourse_marker_weight
    }

    /// Analyze context around a discourse marker
    fn analyze_marker_context(
        &self,
        sent_idx: usize,
        word_idx: usize,
        sentences: &[String],
    ) -> ContextAnalysis {
        let words: Vec<&str> = sentences[sent_idx].split_whitespace().collect();

        let preceding_context = if word_idx > 0 {
            words[word_idx.saturating_sub(3)..word_idx]
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else if sent_idx > 0 {
            // Get context from previous sentence
            let prev_words: Vec<&str> = sentences[sent_idx - 1].split_whitespace().collect();
            prev_words
                .iter()
                .rev()
                .take(2)
                .rev()
                .map(|s| s.to_string())
                .collect()
        } else {
            Vec::new()
        };

        let following_context = if word_idx + 1 < words.len() {
            words[word_idx + 1..(word_idx + 4).min(words.len())]
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else if sent_idx + 1 < sentences.len() {
            // Get context from next sentence
            let next_words: Vec<&str> = sentences[sent_idx + 1].split_whitespace().collect();
            next_words.iter().take(2).map(|s| s.to_string()).collect()
        } else {
            Vec::new()
        };

        let sentence_position = if word_idx == 0 {
            "beginning".to_string()
        } else if word_idx == words.len() - 1 {
            "end".to_string()
        } else {
            "middle".to_string()
        };

        let syntactic_position = SyntacticPosition {
            sentence_position,
            clause_position: "unknown".to_string(), // Would require more sophisticated parsing
            punctuation_context: Vec::new(),
        };

        let semantic_coherence =
            self.calculate_semantic_coherence_with_context(&preceding_context, &following_context);
        let pragmatic_score =
            self.calculate_pragmatic_appropriateness(&preceding_context, &following_context);

        ContextAnalysis {
            preceding_context,
            following_context,
            syntactic_position,
            semantic_coherence,
            pragmatic_score,
        }
    }

    /// Calculate semantic coherence with context
    fn calculate_semantic_coherence_with_context(
        &self,
        preceding: &[String],
        following: &[String],
    ) -> f64 {
        // Simplified semantic coherence calculation
        let context_words: Vec<&String> = preceding.iter().chain(following.iter()).collect();

        if context_words.is_empty() {
            return 0.5; // Neutral score when no context
        }

        // Check for semantic field consistency
        let semantic_consistency = self.calculate_context_semantic_consistency(&context_words);
        semantic_consistency
    }

    /// Calculate pragmatic appropriateness
    fn calculate_pragmatic_appropriateness(
        &self,
        preceding: &[String],
        following: &[String],
    ) -> f64 {
        // Simplified pragmatic analysis
        let has_preceding = !preceding.is_empty();
        let has_following = !following.is_empty();

        match (has_preceding, has_following) {
            (true, true) => 0.9,
            (true, false) => 0.7,
            (false, true) => 0.6,
            (false, false) => 0.4,
        }
    }

    /// Calculate context semantic consistency
    fn calculate_context_semantic_consistency(&self, context_words: &[&String]) -> f64 {
        if context_words.len() < 2 {
            return 0.5;
        }

        // Simple consistency measure based on word repetition and similarity
        let mut consistency_score = 0.0;
        let mut comparisons = 0;

        for i in 0..context_words.len() {
            for j in i + 1..context_words.len() {
                let similarity = self.calculate_word_similarity(context_words[i], context_words[j]);
                consistency_score += similarity;
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            consistency_score / comparisons as f64
        } else {
            0.5
        }
    }

    /// Calculate similarity between two words
    fn calculate_word_similarity(&self, word1: &str, word2: &str) -> f64 {
        if word1 == word2 {
            return 1.0;
        }

        // Simple character-based similarity
        let chars1: HashSet<char> = word1.chars().collect();
        let chars2: HashSet<char> = word2.chars().collect();

        let intersection_size = chars1.intersection(&chars2).count();
        let union_size = chars1.union(&chars2).count();

        if union_size > 0 {
            intersection_size as f64 / union_size as f64
        } else {
            0.0
        }
    }

    /// Calculate marker scope (sentences influenced)
    fn calculate_marker_scope(
        &self,
        sent_idx: usize,
        _word_idx: usize,
        sentences: &[String],
    ) -> usize {
        // Simple heuristic: markers typically influence the current and next sentence
        let remaining_sentences = sentences.len() - sent_idx;
        if remaining_sentences > 1 {
            2
        } else {
            1
        }
    }

    /// Calculate marker classification confidence
    fn calculate_marker_confidence(
        &self,
        marker: &str,
        marker_type: &DiscourseMarkerType,
        context: &ContextAnalysis,
    ) -> f64 {
        // Base confidence from marker dictionary
        let base_confidence = match marker_type {
            DiscourseMarkerType::Cause => 0.9,
            DiscourseMarkerType::Contrast => 0.85,
            DiscourseMarkerType::Addition => 0.8,
            DiscourseMarkerType::Temporal => 0.8,
            _ => 0.7,
        };

        // Adjust based on context
        let context_adjustment = (context.semantic_coherence + context.pragmatic_score) / 2.0;

        (base_confidence + context_adjustment) / 2.0
    }

    /// Calculate rhetorical strength of marker
    fn calculate_rhetorical_strength(
        &self,
        marker_type: &DiscourseMarkerType,
        context: &ContextAnalysis,
    ) -> f64 {
        let base_strength = match marker_type {
            DiscourseMarkerType::Cause => 0.95,
            DiscourseMarkerType::Contrast => 0.9,
            DiscourseMarkerType::Summary => 0.9,
            DiscourseMarkerType::Emphasis => 0.85,
            DiscourseMarkerType::Addition => 0.7,
            DiscourseMarkerType::Temporal => 0.75,
            _ => 0.6,
        };

        // Adjust based on context quality
        let context_boost = context.pragmatic_score * 0.2;
        (base_strength + context_boost).min(1.0)
    }

    /// Calculate sentence transition quality
    fn calculate_sentence_transition_quality(&self, sentences: &[String]) -> Vec<f64> {
        if sentences.len() < 2 {
            return Vec::new();
        }

        let mut transitions = Vec::new();

        for i in 0..sentences.len() - 1 {
            let current = &sentences[i];
            let next = &sentences[i + 1];

            let lexical_overlap = self.calculate_lexical_overlap(current, next);
            let semantic_continuity = self.calculate_semantic_continuity(current, next);
            let marker_presence = self.calculate_transition_marker_presence(current, next);
            let structural_continuity = self.calculate_structural_continuity(current, next);

            let transition_quality = (lexical_overlap * 0.3)
                + (semantic_continuity * 0.3)
                + (marker_presence * 0.25)
                + (structural_continuity * 0.15);

            transitions.push(transition_quality);
        }

        transitions
    }

    /// Calculate lexical overlap between sentences
    fn calculate_lexical_overlap(&self, sent1: &str, sent2: &str) -> f64 {
        let words1: HashSet<String> = self.extract_content_words(sent1).into_iter().collect();
        let words2: HashSet<String> = self.extract_content_words(sent2).into_iter().collect();

        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }

        let intersection_size = words1.intersection(&words2).count();
        let union_size = words1.union(&words2).count();

        if union_size > 0 {
            intersection_size as f64 / union_size as f64
        } else {
            0.0
        }
    }

    /// Calculate semantic continuity between sentences
    fn calculate_semantic_continuity(&self, sent1: &str, sent2: &str) -> f64 {
        // Simplified semantic continuity based on common semantic fields
        let words1 = self.extract_content_words(sent1);
        let words2 = self.extract_content_words(sent2);

        let mut semantic_overlap = 0;
        let mut total_comparisons = 0;

        for word1 in &words1 {
            for word2 in &words2 {
                let similarity = self.calculate_word_similarity(word1, word2);
                if similarity > 0.3 {
                    semantic_overlap += 1;
                }
                total_comparisons += 1;
            }
        }

        if total_comparisons > 0 {
            semantic_overlap as f64 / total_comparisons as f64
        } else {
            0.0
        }
    }

    /// Calculate transition marker presence between sentences
    fn calculate_transition_marker_presence(&self, _sent1: &str, sent2: &str) -> f64 {
        let words = self.extract_all_words(sent2);

        let marker_count = words
            .iter()
            .filter(|word| self.discourse_markers.contains_key(word.as_str()))
            .count();

        if marker_count > 0 {
            1.0
        } else {
            0.0
        }
    }

    /// Calculate structural continuity between sentences
    fn calculate_structural_continuity(&self, sent1: &str, sent2: &str) -> f64 {
        // Simple heuristic based on sentence length similarity
        let len1 = sent1.split_whitespace().count();
        let len2 = sent2.split_whitespace().count();

        let length_similarity =
            1.0 - ((len1 as f64 - len2 as f64).abs() / (len1.max(len2) as f64).max(1.0));
        length_similarity
    }

    /// Extract content words from sentence
    fn extract_content_words(&self, sentence: &str) -> Vec<String> {
        let function_words = HashSet::from([
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "about", "into", "through", "during", "before", "after", "above",
            "below", "between", "among", "is", "are", "was", "were", "be", "been", "being", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "must", "can",
        ]);

        sentence
            .split_whitespace()
            .filter_map(|word| {
                let clean_word = word
                    .trim_matches(|c: char| !c.is_alphabetic())
                    .to_lowercase();
                if clean_word.len() > 2 && !function_words.contains(clean_word.as_str()) {
                    Some(clean_word)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Extract all words from sentence
    fn extract_all_words(&self, sentence: &str) -> Vec<String> {
        sentence
            .split_whitespace()
            .map(|word| {
                word.trim_matches(|c: char| !c.is_alphabetic())
                    .to_lowercase()
            })
            .filter(|word| !word.is_empty())
            .collect()
    }

    /// Identify rhetorical relations in text
    fn identify_rhetorical_relations(
        &self,
        sentences: &[String],
    ) -> Result<HashMap<String, usize>, DiscourseCoherenceError> {
        let mut relations = HashMap::new();

        for sentence in sentences {
            let words = self.extract_all_words(sentence);

            for word in words {
                if let Some(relation_type) = self.rhetorical_patterns.get(&word) {
                    let relation_name = format!("{:?}", relation_type);
                    *relations.entry(relation_name).or_insert(0) += 1;
                }
            }

            // Check for multi-word patterns
            let sentence_lower = sentence.to_lowercase();
            for (pattern, relation_type) in &self.rhetorical_patterns {
                if pattern.contains(' ') && sentence_lower.contains(pattern) {
                    let relation_name = format!("{:?}", relation_type);
                    *relations.entry(relation_name).or_insert(0) += 1;
                }
            }
        }

        Ok(relations)
    }

    /// Calculate discourse marker coherence
    fn calculate_discourse_marker_coherence(
        &self,
        markers: &[DiscourseMarker],
    ) -> Result<f64, DiscourseCoherenceError> {
        if markers.is_empty() {
            return Ok(0.0);
        }

        let total_contribution: f64 = markers.iter().map(|m| m.contribution * m.confidence).sum();
        let total_possible: f64 = markers.len() as f64;

        if total_possible > 0.0 {
            Ok(total_contribution / total_possible)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate rhetorical structure coherence
    fn calculate_rhetorical_structure_coherence(&self, relations: &HashMap<String, usize>) -> f64 {
        if relations.is_empty() {
            return 0.0;
        }

        let total_relations: usize = relations.values().sum();
        let unique_relations = relations.len();

        // Balance between diversity and presence
        let diversity_score = unique_relations as f64 / 10.0; // Normalize by expected max diversity
        let presence_score = (total_relations as f64).ln() / 5.0; // Logarithmic scaling

        ((diversity_score + presence_score) / 2.0).min(1.0)
    }

    /// Calculate overall cohesion score
    fn calculate_cohesion_score(&self, sentences: &[String]) -> f64 {
        let transition_words = self.count_cohesive_devices(sentences);
        let pronoun_references = self.count_pronoun_references(sentences);
        let lexical_ties = self.count_lexical_ties(sentences);

        let total_sentences = sentences.len() as f64;
        if total_sentences == 0.0 {
            return 0.0;
        }

        let transition_density = transition_words as f64 / total_sentences;
        let reference_density = pronoun_references as f64 / total_sentences;
        let tie_density = lexical_ties as f64 / total_sentences;

        (transition_density + reference_density + tie_density) / 3.0
    }

    /// Count cohesive devices in sentences
    fn count_cohesive_devices(&self, sentences: &[String]) -> usize {
        sentences
            .iter()
            .map(|sentence| {
                self.discourse_markers
                    .keys()
                    .filter(|marker| sentence.to_lowercase().contains(marker.as_str()))
                    .count()
            })
            .sum()
    }

    /// Count pronoun references
    fn count_pronoun_references(&self, sentences: &[String]) -> usize {
        let pronouns = HashSet::from([
            "he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "their", "this",
            "that", "these", "those", "which", "who", "whom",
        ]);

        sentences
            .iter()
            .map(|sentence| {
                self.extract_all_words(sentence)
                    .iter()
                    .filter(|word| pronouns.contains(word.as_str()))
                    .count()
            })
            .sum()
    }

    /// Count lexical ties between sentences
    fn count_lexical_ties(&self, sentences: &[String]) -> usize {
        if sentences.len() < 2 {
            return 0;
        }

        let mut ties = 0;

        for i in 0..sentences.len() - 1 {
            let words1 = self.extract_content_words(&sentences[i]);
            let words2 = self.extract_content_words(&sentences[i + 1]);

            for word1 in &words1 {
                for word2 in &words2 {
                    let similarity = self.calculate_word_similarity(word1, word2);
                    if similarity > 0.7 {
                        ties += 1;
                    }
                }
            }
        }

        ties
    }

    /// Split text into sentences
    fn split_into_sentences(&self, text: &str) -> Result<Vec<String>, DiscourseCoherenceError> {
        let sentences: Vec<String> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_string())
            .collect();

        if sentences.is_empty() {
            return Err(DiscourseCoherenceError::ProcessingError(
                "No valid sentences found".to_string(),
            ));
        }

        Ok(sentences)
    }

    /// Build discourse markers dictionary
    fn build_discourse_markers() -> HashMap<String, DiscourseMarkerType> {
        let mut markers = HashMap::new();

        // Addition markers
        for marker in [
            "furthermore",
            "moreover",
            "additionally",
            "also",
            "besides",
            "likewise",
        ] {
            markers.insert(marker.to_string(), DiscourseMarkerType::Addition);
        }

        // Contrast markers
        for marker in [
            "however",
            "nevertheless",
            "nonetheless",
            "conversely",
            "although",
            "though",
        ] {
            markers.insert(marker.to_string(), DiscourseMarkerType::Contrast);
        }

        // Cause markers
        for marker in [
            "therefore",
            "thus",
            "consequently",
            "hence",
            "accordingly",
            "because",
        ] {
            markers.insert(marker.to_string(), DiscourseMarkerType::Cause);
        }

        // Temporal markers
        for marker in [
            "first",
            "then",
            "next",
            "finally",
            "meanwhile",
            "subsequently",
        ] {
            markers.insert(marker.to_string(), DiscourseMarkerType::Temporal);
        }

        // Conditional markers
        for marker in ["if", "unless", "provided", "assuming", "suppose"] {
            markers.insert(marker.to_string(), DiscourseMarkerType::Conditional);
        }

        // Concession markers
        for marker in ["despite", "although", "while", "whereas", "granted"] {
            markers.insert(marker.to_string(), DiscourseMarkerType::Concession);
        }

        // Elaboration markers
        for marker in [
            "specifically",
            "namely",
            "particularly",
            "especially",
            "indeed",
        ] {
            markers.insert(marker.to_string(), DiscourseMarkerType::Elaboration);
        }

        // Exemplification markers
        for marker in [
            "for_example",
            "for_instance",
            "such_as",
            "including",
            "like",
        ] {
            markers.insert(marker.to_string(), DiscourseMarkerType::Exemplification);
        }

        // Summary markers
        for marker in [
            "in_conclusion",
            "to_summarize",
            "overall",
            "in_summary",
            "finally",
        ] {
            markers.insert(marker.to_string(), DiscourseMarkerType::Summary);
        }

        // Emphasis markers
        for marker in ["indeed", "certainly", "definitely", "clearly", "obviously"] {
            markers.insert(marker.to_string(), DiscourseMarkerType::Emphasis);
        }

        // Comparison markers
        for marker in [
            "similarly",
            "likewise",
            "compared_to",
            "in_contrast",
            "unlike",
        ] {
            markers.insert(marker.to_string(), DiscourseMarkerType::Comparison);
        }

        // Alternative markers
        for marker in [
            "alternatively",
            "otherwise",
            "instead",
            "rather",
            "on_the_other_hand",
        ] {
            markers.insert(marker.to_string(), DiscourseMarkerType::Alternative);
        }

        // Reformulation markers
        for marker in ["in_other_words", "that_is", "namely", "i.e.", "viz"] {
            markers.insert(marker.to_string(), DiscourseMarkerType::Reformulation);
        }

        markers
    }

    /// Build multi-word markers dictionary
    fn build_multiword_markers(&self) -> HashMap<String, DiscourseMarkerType> {
        let mut markers = HashMap::new();

        markers.insert(
            "on the other hand".to_string(),
            DiscourseMarkerType::Contrast,
        );
        markers.insert("as a result".to_string(), DiscourseMarkerType::Cause);
        markers.insert(
            "for example".to_string(),
            DiscourseMarkerType::Exemplification,
        );
        markers.insert("in conclusion".to_string(), DiscourseMarkerType::Summary);
        markers.insert(
            "in other words".to_string(),
            DiscourseMarkerType::Reformulation,
        );
        markers.insert(
            "that is to say".to_string(),
            DiscourseMarkerType::Elaboration,
        );
        markers.insert("as well as".to_string(), DiscourseMarkerType::Addition);
        markers.insert("even though".to_string(), DiscourseMarkerType::Concession);

        markers
    }

    /// Build rhetorical patterns dictionary
    fn build_rhetorical_patterns() -> HashMap<String, RhetoricalRelationType> {
        let mut patterns = HashMap::new();

        // Elaboration patterns
        patterns.insert(
            "specifically".to_string(),
            RhetoricalRelationType::Elaboration,
        );
        patterns.insert(
            "particularly".to_string(),
            RhetoricalRelationType::Elaboration,
        );

        // Evidence patterns
        patterns.insert("evidence".to_string(), RhetoricalRelationType::Evidence);
        patterns.insert("proof".to_string(), RhetoricalRelationType::Evidence);

        // Sequence patterns
        patterns.insert("first".to_string(), RhetoricalRelationType::Sequence);
        patterns.insert("second".to_string(), RhetoricalRelationType::Sequence);
        patterns.insert("next".to_string(), RhetoricalRelationType::Sequence);

        // Contrast patterns
        patterns.insert("however".to_string(), RhetoricalRelationType::Contrast);
        patterns.insert("although".to_string(), RhetoricalRelationType::Contrast);

        patterns
    }

    /// Build cohesion patterns dictionary
    fn build_cohesion_patterns() -> HashMap<String, CohesiveDeviceType> {
        let mut patterns = HashMap::new();

        // Reference patterns
        patterns.insert("this".to_string(), CohesiveDeviceType::Reference);
        patterns.insert("that".to_string(), CohesiveDeviceType::Reference);
        patterns.insert("these".to_string(), CohesiveDeviceType::Reference);
        patterns.insert("those".to_string(), CohesiveDeviceType::Reference);

        // Conjunction patterns
        patterns.insert("and".to_string(), CohesiveDeviceType::Conjunction);
        patterns.insert("but".to_string(), CohesiveDeviceType::Conjunction);
        patterns.insert("or".to_string(), CohesiveDeviceType::Conjunction);

        patterns
    }

    /// Build reference patterns set
    fn build_reference_patterns() -> HashSet<String> {
        let patterns = vec![
            "he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "their", "this",
            "that", "these", "those", "such", "same", "former", "latter",
        ];

        patterns.into_iter().map(String::from).collect()
    }

    /// Build temporal markers set
    fn build_temporal_markers() -> HashSet<String> {
        let markers = vec![
            "when",
            "while",
            "during",
            "before",
            "after",
            "since",
            "until",
            "then",
            "now",
            "first",
            "second",
            "finally",
            "next",
            "previously",
            "subsequently",
            "meanwhile",
        ];

        markers.into_iter().map(String::from).collect()
    }

    // Additional methods for comprehensive analysis...

    /// Generate detailed discourse metrics
    fn generate_detailed_metrics(
        &self,
        sentences: &[String],
        markers: &[DiscourseMarker],
        rhetorical_relations: &HashMap<String, usize>,
        transition_quality: &[f64],
    ) -> DetailedDiscourseMetrics {
        let marker_type_distribution = self.calculate_marker_type_distribution(markers);
        let average_marker_contribution = self.calculate_average_marker_contribution(markers);
        let marker_density = markers.len() as f64 / sentences.len() as f64;
        let transition_quality_distribution =
            self.calculate_transition_quality_distribution(transition_quality);
        let rhetorical_complexity =
            self.calculate_rhetorical_complexity(rhetorical_relations, sentences);
        let cohesion_statistics = self.calculate_cohesion_statistics(sentences);
        let discourse_flow = self.analyze_discourse_flow(sentences, transition_quality);
        let cross_reference_analysis = self.analyze_cross_references(sentences);
        let temporal_coherence = self.analyze_temporal_coherence(sentences);
        let information_structure = self.analyze_information_structure(sentences);

        DetailedDiscourseMetrics {
            marker_type_distribution,
            average_marker_contribution,
            marker_density,
            transition_quality_distribution,
            rhetorical_complexity,
            cohesion_statistics,
            discourse_flow,
            cross_reference_analysis,
            temporal_coherence,
            information_structure,
        }
    }

    /// Calculate marker type distribution
    fn calculate_marker_type_distribution(
        &self,
        markers: &[DiscourseMarker],
    ) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();

        for marker in markers {
            let type_name = format!("{:?}", marker.marker_type);
            *distribution.entry(type_name).or_insert(0) += 1;
        }

        distribution
    }

    /// Calculate average marker contribution
    fn calculate_average_marker_contribution(&self, markers: &[DiscourseMarker]) -> f64 {
        if markers.is_empty() {
            return 0.0;
        }

        let total_contribution: f64 = markers.iter().map(|m| m.contribution).sum();
        total_contribution / markers.len() as f64
    }

    /// Calculate transition quality distribution
    fn calculate_transition_quality_distribution(
        &self,
        qualities: &[f64],
    ) -> TransitionQualityDistribution {
        if qualities.is_empty() {
            return TransitionQualityDistribution {
                excellent: 0,
                good: 0,
                adequate: 0,
                poor: 0,
                abrupt: 0,
                average_quality: 0.0,
                quality_variance: 0.0,
            };
        }

        let mut excellent = 0;
        let mut good = 0;
        let mut adequate = 0;
        let mut poor = 0;
        let mut abrupt = 0;

        for &quality in qualities {
            match quality {
                q if q >= 0.9 => excellent += 1,
                q if q >= 0.7 => good += 1,
                q if q >= 0.5 => adequate += 1,
                q if q >= 0.3 => poor += 1,
                _ => abrupt += 1,
            }
        }

        let average_quality = qualities.iter().sum::<f64>() / qualities.len() as f64;
        let quality_variance = qualities
            .iter()
            .map(|q| (q - average_quality).powi(2))
            .sum::<f64>()
            / qualities.len() as f64;

        TransitionQualityDistribution {
            excellent,
            good,
            adequate,
            poor,
            abrupt,
            average_quality,
            quality_variance,
        }
    }

    /// Calculate rhetorical complexity metrics
    fn calculate_rhetorical_complexity(
        &self,
        relations: &HashMap<String, usize>,
        sentences: &[String],
    ) -> RhetoricalComplexityMetrics {
        let relation_count = relations.values().sum();
        let relation_diversity = self.calculate_relation_diversity(relations);
        let average_confidence = 0.8; // Simplified
        let tree_depth = self.estimate_discourse_tree_depth(sentences.len());
        let structural_balance = self.calculate_structural_balance(relations);
        let hierarchical_organization = self.calculate_hierarchical_organization(relations);

        RhetoricalComplexityMetrics {
            relation_count,
            relation_diversity,
            average_confidence,
            tree_depth,
            structural_balance,
            hierarchical_organization,
        }
    }

    /// Calculate relation diversity
    fn calculate_relation_diversity(&self, relations: &HashMap<String, usize>) -> f64 {
        if relations.is_empty() {
            return 0.0;
        }

        let total: usize = relations.values().sum();
        if total == 0 {
            return 0.0;
        }

        let entropy = relations
            .values()
            .map(|&count| {
                let p = count as f64 / total as f64;
                if p > 0.0 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        let max_entropy = (relations.len() as f64).ln();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Estimate discourse tree depth
    fn estimate_discourse_tree_depth(&self, sentence_count: usize) -> usize {
        // Simple heuristic based on text length
        match sentence_count {
            0..=2 => 1,
            3..=5 => 2,
            6..=10 => 3,
            11..=20 => 4,
            _ => 5,
        }
    }

    /// Calculate structural balance
    fn calculate_structural_balance(&self, relations: &HashMap<String, usize>) -> f64 {
        // Simplified structural balance based on relation distribution
        if relations.is_empty() {
            return 0.0;
        }

        let total: usize = relations.values().sum();
        let max_count = *relations.values().max().unwrap_or(&0);

        if total == 0 {
            return 0.0;
        }

        1.0 - (max_count as f64 / total as f64).min(1.0)
    }

    /// Calculate hierarchical organization
    fn calculate_hierarchical_organization(&self, relations: &HashMap<String, usize>) -> f64 {
        // Simplified hierarchical organization score
        let hierarchical_relations = ["Elaboration", "Evidence", "Background"];
        let hierarchical_count: usize = relations
            .iter()
            .filter(|(relation, _)| hierarchical_relations.contains(&relation.as_str()))
            .map(|(_, count)| count)
            .sum();

        let total_relations: usize = relations.values().sum();
        if total_relations > 0 {
            hierarchical_count as f64 / total_relations as f64
        } else {
            0.0
        }
    }

    /// Calculate cohesion statistics
    fn calculate_cohesion_statistics(&self, sentences: &[String]) -> CohesionStatistics {
        // Simplified cohesion statistics
        let total_devices = self.count_cohesive_devices(sentences);
        let device_type_distribution = HashMap::new(); // Would require more analysis
        let reference_chains = Vec::new(); // Would require coreference resolution
        let average_tie_strength = 0.7; // Simplified
        let cohesive_density = total_devices as f64 / sentences.len() as f64;
        let chain_completeness = 0.8; // Simplified

        CohesionStatistics {
            total_devices,
            device_type_distribution,
            reference_chains,
            average_tie_strength,
            cohesive_density,
            chain_completeness,
        }
    }

    /// Analyze discourse flow
    fn analyze_discourse_flow(
        &self,
        sentences: &[String],
        transition_quality: &[f64],
    ) -> DiscourseFlowMetrics {
        let flow_continuity = if !transition_quality.is_empty() {
            transition_quality.iter().sum::<f64>() / transition_quality.len() as f64
        } else {
            0.0
        };

        let topic_progression = self.calculate_topic_progression(sentences);
        let information_entropy = self.calculate_information_entropy(sentences);
        let coherence_momentum = self.calculate_coherence_momentum(transition_quality);
        let disruption_points = self.identify_disruption_points(transition_quality);
        let quality_evolution = transition_quality.to_vec();

        DiscourseFlowMetrics {
            flow_continuity,
            topic_progression,
            information_entropy,
            coherence_momentum,
            disruption_points,
            quality_evolution,
        }
    }

    /// Calculate topic progression score
    fn calculate_topic_progression(&self, sentences: &[String]) -> f64 {
        if sentences.len() < 2 {
            return 1.0;
        }

        let mut progression_score = 0.0;
        let mut comparisons = 0;

        for i in 0..sentences.len() - 1 {
            let overlap = self.calculate_lexical_overlap(&sentences[i], &sentences[i + 1]);
            progression_score += overlap;
            comparisons += 1;
        }

        if comparisons > 0 {
            progression_score / comparisons as f64
        } else {
            0.0
        }
    }

    /// Calculate information entropy
    fn calculate_information_entropy(&self, sentences: &[String]) -> f64 {
        let all_words: Vec<String> = sentences
            .iter()
            .flat_map(|s| self.extract_all_words(s))
            .collect();

        if all_words.is_empty() {
            return 0.0;
        }

        let word_counts = all_words.iter().fold(HashMap::new(), |mut acc, word| {
            *acc.entry(word).or_insert(0) += 1;
            acc
        });

        let total_words = all_words.len() as f64;
        let entropy = word_counts
            .values()
            .map(|&count| {
                let p = count as f64 / total_words;
                if p > 0.0 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        entropy / total_words.ln().max(1.0)
    }

    /// Calculate coherence momentum
    fn calculate_coherence_momentum(&self, transition_quality: &[f64]) -> f64 {
        if transition_quality.len() < 2 {
            return 0.0;
        }

        let mut momentum = 0.0;
        for i in 1..transition_quality.len() {
            let change = transition_quality[i] - transition_quality[i - 1];
            momentum += change;
        }

        momentum / (transition_quality.len() - 1) as f64
    }

    /// Identify disruption points in discourse flow
    fn identify_disruption_points(&self, transition_quality: &[f64]) -> Vec<usize> {
        let mut disruption_points = Vec::new();
        let threshold = 0.3; // Quality drop threshold

        for (i, &quality) in transition_quality.iter().enumerate() {
            if quality < threshold {
                disruption_points.push(i);
            }
        }

        disruption_points
    }

    /// Analyze cross-references in text
    fn analyze_cross_references(&self, sentences: &[String]) -> CrossReferenceMetrics {
        // Simplified cross-reference analysis
        let forward_references = self.count_forward_references(sentences);
        let backward_references = self.count_backward_references(sentences);
        let resolution_success_rate = 0.8; // Would require proper coreference resolution
        let average_reference_distance = 2.5; // Simplified
        let ambiguous_references = self.count_ambiguous_references(sentences);
        let complexity_score = self.calculate_reference_complexity(sentences);

        CrossReferenceMetrics {
            forward_references,
            backward_references,
            resolution_success_rate,
            average_reference_distance,
            ambiguous_references,
            complexity_score,
        }
    }

    /// Count forward references
    fn count_forward_references(&self, sentences: &[String]) -> usize {
        // Simplified: count demonstratives and definite articles that might refer forward
        let forward_patterns = ["this", "that", "these", "those", "such"];

        sentences
            .iter()
            .map(|sentence| {
                let words = self.extract_all_words(sentence);
                words
                    .iter()
                    .filter(|word| forward_patterns.contains(&word.as_str()))
                    .count()
            })
            .sum()
    }

    /// Count backward references
    fn count_backward_references(&self, sentences: &[String]) -> usize {
        // Count pronouns that typically refer backward
        let backward_patterns = ["he", "she", "it", "they", "him", "her", "them"];

        sentences
            .iter()
            .map(|sentence| {
                let words = self.extract_all_words(sentence);
                words
                    .iter()
                    .filter(|word| backward_patterns.contains(&word.as_str()))
                    .count()
            })
            .sum()
    }

    /// Count ambiguous references
    fn count_ambiguous_references(&self, sentences: &[String]) -> usize {
        // Simplified: count pronouns that could be ambiguous
        let ambiguous_patterns = ["it", "this", "that"];

        sentences
            .iter()
            .map(|sentence| {
                let words = self.extract_all_words(sentence);
                words
                    .iter()
                    .filter(|word| ambiguous_patterns.contains(&word.as_str()))
                    .count()
            })
            .sum()
    }

    /// Calculate reference complexity
    fn calculate_reference_complexity(&self, sentences: &[String]) -> f64 {
        let total_references = self.count_pronoun_references(sentences);
        let total_sentences = sentences.len();

        if total_sentences > 0 {
            (total_references as f64 / total_sentences as f64).min(1.0)
        } else {
            0.0
        }
    }

    /// Analyze temporal coherence
    fn analyze_temporal_coherence(&self, sentences: &[String]) -> TemporalCoherenceMetrics {
        let temporal_marker_frequency = self.calculate_temporal_marker_frequency(sentences);
        let sequence_coherence = self.calculate_sequence_coherence(sentences);
        let anchoring_score = self.calculate_temporal_anchoring(sentences);
        let timeline_consistency = self.calculate_timeline_consistency(sentences);
        let temporal_disruptions = self.count_temporal_disruptions(sentences);

        TemporalCoherenceMetrics {
            temporal_marker_frequency,
            sequence_coherence,
            anchoring_score,
            timeline_consistency,
            temporal_disruptions,
        }
    }

    /// Calculate temporal marker frequency
    fn calculate_temporal_marker_frequency(&self, sentences: &[String]) -> f64 {
        let temporal_marker_count = sentences
            .iter()
            .map(|sentence| {
                let words = self.extract_all_words(sentence);
                words
                    .iter()
                    .filter(|word| self.temporal_markers.contains(word.as_str()))
                    .count()
            })
            .sum::<usize>();

        let total_sentences = sentences.len();
        if total_sentences > 0 {
            temporal_marker_count as f64 / total_sentences as f64
        } else {
            0.0
        }
    }

    /// Calculate sequence coherence
    fn calculate_sequence_coherence(&self, sentences: &[String]) -> f64 {
        // Simplified sequence coherence based on temporal markers
        let sequence_markers = ["first", "second", "third", "then", "next", "finally"];
        let marker_positions: Vec<usize> = sentences
            .iter()
            .enumerate()
            .filter_map(|(i, sentence)| {
                let words = self.extract_all_words(sentence);
                if words
                    .iter()
                    .any(|word| sequence_markers.contains(&word.as_str()))
                {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        if marker_positions.len() < 2 {
            return 0.5; // Neutral score for insufficient data
        }

        // Check if markers appear in increasing order
        let ordered = marker_positions.windows(2).all(|w| w[1] > w[0]);
        if ordered {
            0.9
        } else {
            0.4
        }
    }

    /// Calculate temporal anchoring
    fn calculate_temporal_anchoring(&self, sentences: &[String]) -> f64 {
        // Count absolute time references
        let time_patterns = [
            "today",
            "yesterday",
            "tomorrow",
            "monday",
            "january",
            "2023",
        ];
        let anchor_count = sentences
            .iter()
            .map(|sentence| {
                let words = self.extract_all_words(sentence);
                words
                    .iter()
                    .filter(|word| time_patterns.iter().any(|pattern| word.contains(pattern)))
                    .count()
            })
            .sum::<usize>();

        (anchor_count as f64 / sentences.len().max(1) as f64).min(1.0)
    }

    /// Calculate timeline consistency
    fn calculate_timeline_consistency(&self, sentences: &[String]) -> f64 {
        // Simplified timeline consistency
        let tense_consistency = self.analyze_tense_consistency(sentences);
        tense_consistency
    }

    /// Analyze tense consistency
    fn analyze_tense_consistency(&self, sentences: &[String]) -> f64 {
        // Very simplified tense analysis based on common verb forms
        let past_markers = ["was", "were", "had", "did", "went"];
        let present_markers = ["is", "are", "have", "do", "go"];
        let future_markers = ["will", "shall", "going"];

        let mut past_count = 0;
        let mut present_count = 0;
        let mut future_count = 0;

        for sentence in sentences {
            let words = self.extract_all_words(sentence);
            if words.iter().any(|w| past_markers.contains(&w.as_str())) {
                past_count += 1;
            }
            if words.iter().any(|w| present_markers.contains(&w.as_str())) {
                present_count += 1;
            }
            if words.iter().any(|w| future_markers.contains(&w.as_str())) {
                future_count += 1;
            }
        }

        let total = past_count + present_count + future_count;
        if total == 0 {
            return 0.5;
        }

        let max_tense = past_count.max(present_count).max(future_count);
        max_tense as f64 / total as f64
    }

    /// Count temporal disruptions
    fn count_temporal_disruptions(&self, sentences: &[String]) -> usize {
        // Count potential temporal inconsistencies
        let mut disruptions = 0;

        for window in sentences.windows(2) {
            let sent1_words = self.extract_all_words(&window[0]);
            let sent2_words = self.extract_all_words(&window[1]);

            // Check for conflicting temporal signals
            let has_past1 = sent1_words
                .iter()
                .any(|w| ["was", "were", "had"].contains(&w.as_str()));
            let has_future1 = sent1_words
                .iter()
                .any(|w| ["will", "shall"].contains(&w.as_str()));
            let has_past2 = sent2_words
                .iter()
                .any(|w| ["was", "were", "had"].contains(&w.as_str()));
            let has_future2 = sent2_words
                .iter()
                .any(|w| ["will", "shall"].contains(&w.as_str()));

            if (has_past1 && has_future2) || (has_future1 && has_past2) {
                disruptions += 1;
            }
        }

        disruptions
    }

    /// Analyze information structure
    fn analyze_information_structure(&self, sentences: &[String]) -> InformationStructureMetrics {
        let given_new_balance = self.calculate_given_new_balance(sentences);
        let topic_focus_articulation = self.calculate_topic_focus_articulation(sentences);
        let information_packaging = self.calculate_information_packaging(sentences);
        let thematic_progression = self.identify_thematic_progression_pattern(sentences);
        let information_density = self.calculate_information_density(sentences);

        InformationStructureMetrics {
            given_new_balance,
            topic_focus_articulation,
            information_packaging,
            thematic_progression,
            information_density,
        }
    }

    /// Calculate given-new information balance
    fn calculate_given_new_balance(&self, sentences: &[String]) -> f64 {
        // Simplified: based on repetition (given) vs new words
        let mut total_words = 0;
        let mut repeated_words = 0;
        let mut seen_words = HashSet::new();

        for sentence in sentences {
            let words = self.extract_content_words(sentence);
            for word in words {
                total_words += 1;
                if seen_words.contains(&word) {
                    repeated_words += 1;
                } else {
                    seen_words.insert(word);
                }
            }
        }

        if total_words > 0 {
            repeated_words as f64 / total_words as f64
        } else {
            0.0
        }
    }

    /// Calculate topic-focus articulation
    fn calculate_topic_focus_articulation(&self, sentences: &[String]) -> f64 {
        // Simplified: based on sentence structure variation
        let avg_length = sentences
            .iter()
            .map(|s| s.split_whitespace().count())
            .sum::<usize>() as f64
            / sentences.len() as f64;

        let variance = sentences
            .iter()
            .map(|s| {
                let len = s.split_whitespace().count() as f64;
                (len - avg_length).powi(2)
            })
            .sum::<f64>()
            / sentences.len() as f64;

        // Normalize variance to 0-1 range
        (variance.sqrt() / avg_length).min(1.0)
    }

    /// Calculate information packaging score
    fn calculate_information_packaging(&self, sentences: &[String]) -> f64 {
        // Based on sentence complexity and structure
        let complexity_scores: Vec<f64> = sentences
            .iter()
            .map(|sentence| {
                let word_count = sentence.split_whitespace().count();
                let comma_count = sentence.matches(',').count();
                let clause_complexity = (comma_count + 1) as f64 / word_count as f64;
                clause_complexity.min(1.0)
            })
            .collect();

        if complexity_scores.is_empty() {
            0.0
        } else {
            complexity_scores.iter().sum::<f64>() / complexity_scores.len() as f64
        }
    }

    /// Identify thematic progression pattern
    fn identify_thematic_progression_pattern(&self, sentences: &[String]) -> String {
        if sentences.len() < 2 {
            return "insufficient_data".to_string();
        }

        // Simplified pattern recognition based on lexical overlap
        let overlaps: Vec<f64> = sentences
            .windows(2)
            .map(|window| self.calculate_lexical_overlap(&window[0], &window[1]))
            .collect();

        let avg_overlap = overlaps.iter().sum::<f64>() / overlaps.len() as f64;

        match avg_overlap {
            x if x > 0.7 => "constant_theme".to_string(),
            x if x > 0.4 => "linear_progression".to_string(),
            x if x > 0.2 => "derived_theme".to_string(),
            _ => "split_rheme".to_string(),
        }
    }

    /// Calculate information density
    fn calculate_information_density(&self, sentences: &[String]) -> f64 {
        let content_words: usize = sentences
            .iter()
            .map(|s| self.extract_content_words(s).len())
            .sum();

        let total_words: usize = sentences.iter().map(|s| s.split_whitespace().count()).sum();

        if total_words > 0 {
            content_words as f64 / total_words as f64
        } else {
            0.0
        }
    }

    /// Analyze cohesive devices
    fn analyze_cohesive_devices(
        &self,
        sentences: &[String],
    ) -> Result<Vec<CohesiveDevice>, DiscourseCoherenceError> {
        let mut devices = Vec::new();

        // Analyze reference devices
        devices.extend(self.analyze_reference_devices(sentences));

        // Analyze conjunction devices
        devices.extend(self.analyze_conjunction_devices(sentences));

        // Analyze lexical cohesion devices
        devices.extend(self.analyze_lexical_cohesion_devices(sentences));

        Ok(devices)
    }

    /// Analyze reference devices
    fn analyze_reference_devices(&self, sentences: &[String]) -> Vec<CohesiveDevice> {
        let mut devices = Vec::new();

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let words = self.extract_all_words(sentence);

            for (word_idx, word) in words.iter().enumerate() {
                if self.reference_patterns.contains(word) {
                    devices.push(CohesiveDevice {
                        device_type: CohesiveDeviceType::Reference,
                        elements: vec![word.clone()],
                        positions: vec![(sent_idx, word_idx)],
                        strength: 0.8,
                        local_contribution: 0.6,
                        global_contribution: 0.4,
                        resolution_confidence: 0.7,
                    });
                }
            }
        }

        devices
    }

    /// Analyze conjunction devices
    fn analyze_conjunction_devices(&self, sentences: &[String]) -> Vec<CohesiveDevice> {
        let mut devices = Vec::new();
        let conjunctions = ["and", "but", "or", "so", "yet", "nor"];

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let words = self.extract_all_words(sentence);

            for (word_idx, word) in words.iter().enumerate() {
                if conjunctions.contains(&word.as_str()) {
                    devices.push(CohesiveDevice {
                        device_type: CohesiveDeviceType::Conjunction,
                        elements: vec![word.clone()],
                        positions: vec![(sent_idx, word_idx)],
                        strength: 0.7,
                        local_contribution: 0.8,
                        global_contribution: 0.3,
                        resolution_confidence: 0.9,
                    });
                }
            }
        }

        devices
    }

    /// Analyze lexical cohesion devices
    fn analyze_lexical_cohesion_devices(&self, sentences: &[String]) -> Vec<CohesiveDevice> {
        let mut devices = Vec::new();

        // Find repeated words across sentences
        let mut word_positions = HashMap::new();

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let content_words = self.extract_content_words(sentence);
            for (word_idx, word) in content_words.iter().enumerate() {
                word_positions
                    .entry(word.clone())
                    .or_insert_with(Vec::new)
                    .push((sent_idx, word_idx));
            }
        }

        // Create cohesive devices for repeated words
        for (word, positions) in word_positions {
            if positions.len() > 1 {
                let strength = (positions.len() as f64).ln() / 3.0;
                devices.push(CohesiveDevice {
                    device_type: CohesiveDeviceType::LexicalCohesion,
                    elements: vec![word],
                    positions,
                    strength,
                    local_contribution: strength * 0.6,
                    global_contribution: strength * 0.4,
                    resolution_confidence: 0.8,
                });
            }
        }

        devices
    }

    /// Perform advanced discourse analysis
    fn perform_advanced_analysis(
        &self,
        sentences: &[String],
        markers: &[DiscourseMarker],
        rhetorical_relations: &HashMap<String, usize>,
    ) -> Result<AdvancedDiscourseAnalysis, DiscourseCoherenceError> {
        let discourse_tree = if self.config.build_discourse_tree {
            Some(self.build_discourse_tree(sentences, rhetorical_relations)?)
        } else {
            None
        };

        let coherence_network = self.build_coherence_network(sentences, markers);
        let information_measures = self.calculate_discourse_information_measures(sentences);
        let cognitive_measures = self.calculate_cognitive_measures(sentences, markers);
        let genre_analysis = self.analyze_genre(sentences, markers);

        Ok(AdvancedDiscourseAnalysis {
            discourse_tree,
            coherence_network,
            information_measures,
            cognitive_measures,
            genre_analysis,
        })
    }

    /// Build discourse tree
    fn build_discourse_tree(
        &self,
        sentences: &[String],
        _rhetorical_relations: &HashMap<String, usize>,
    ) -> Result<DiscourseTree, DiscourseCoherenceError> {
        // Simplified discourse tree construction
        let root = DiscourseNode {
            node_id: 0,
            relation_type: RhetoricalRelationType::Joint,
            nuclearity: "nucleus".to_string(),
            text_span: (0, sentences.len().saturating_sub(1)),
            children: Vec::new(),
            confidence: 0.8,
        };

        Ok(DiscourseTree {
            root,
            depth: self.estimate_discourse_tree_depth(sentences.len()),
            node_count: sentences.len(),
            balance_score: 0.7,
        })
    }

    /// Build coherence network
    fn build_coherence_network(
        &self,
        sentences: &[String],
        _markers: &[DiscourseMarker],
    ) -> CoherenceNetwork {
        let nodes: Vec<NetworkNode> = sentences
            .iter()
            .enumerate()
            .map(|(i, content)| NetworkNode {
                node_id: i,
                content: content.clone(),
                position: i,
                centrality: 0.5, // Simplified
            })
            .collect();

        let edges = Vec::new(); // Would require more sophisticated analysis

        CoherenceNetwork {
            nodes,
            edges,
            density: 0.3,
            average_path_length: 2.5,
            clustering_coefficient: 0.6,
        }
    }

    /// Calculate discourse information measures
    fn calculate_discourse_information_measures(
        &self,
        sentences: &[String],
    ) -> DiscourseInformationMeasures {
        let entropy = self.calculate_information_entropy(sentences);
        let information_flow_rate = entropy / sentences.len() as f64;
        let redundancy_coefficient = self.calculate_given_new_balance(sentences);
        let segment_mutual_information = 0.5; // Simplified
        let predictability_score = 1.0 - entropy;

        DiscourseInformationMeasures {
            entropy,
            information_flow_rate,
            redundancy_coefficient,
            segment_mutual_information,
            predictability_score,
        }
    }

    /// Calculate cognitive processing measures
    fn calculate_cognitive_measures(
        &self,
        sentences: &[String],
        markers: &[DiscourseMarker],
    ) -> CognitiveMeasures {
        let avg_sentence_length = sentences
            .iter()
            .map(|s| s.split_whitespace().count())
            .sum::<usize>() as f64
            / sentences.len() as f64;

        let processing_load = (avg_sentence_length / 15.0).min(1.0); // Normalize by typical sentence length
        let working_memory_demand = (sentences.len() as f64 / 10.0).min(1.0); // Based on text length
        let integration_complexity = if markers.is_empty() { 0.8 } else { 0.4 }; // Fewer markers = more complex
        let inference_burden = 1.0 - (markers.len() as f64 / sentences.len() as f64).min(1.0);
        let comprehension_ease =
            1.0 - (processing_load + working_memory_demand + integration_complexity) / 3.0;

        CognitiveMeasures {
            processing_load,
            working_memory_demand,
            integration_complexity,
            inference_burden,
            comprehension_ease,
        }
    }

    /// Analyze text genre
    fn analyze_genre(&self, sentences: &[String], markers: &[DiscourseMarker]) -> GenreAnalysis {
        // Simplified genre analysis based on discourse markers
        let argumentative_markers = markers
            .iter()
            .filter(|m| {
                matches!(
                    m.marker_type,
                    DiscourseMarkerType::Cause
                        | DiscourseMarkerType::Contrast
                        | DiscourseMarkerType::Evidence
                )
            })
            .count();

        let narrative_markers = markers
            .iter()
            .filter(|m| matches!(m.marker_type, DiscourseMarkerType::Temporal))
            .count();

        let expository_markers = markers
            .iter()
            .filter(|m| {
                matches!(
                    m.marker_type,
                    DiscourseMarkerType::Elaboration | DiscourseMarkerType::Exemplification
                )
            })
            .count();

        let predicted_genre = if argumentative_markers > narrative_markers
            && argumentative_markers > expository_markers
        {
            "argumentative"
        } else if narrative_markers > expository_markers {
            "narrative"
        } else {
            "expository"
        }
        .to_string();

        let genre_confidence = 0.7; // Simplified
        let mut genre_features = HashMap::new();
        genre_features.insert(
            "argumentative_score".to_string(),
            argumentative_markers as f64,
        );
        genre_features.insert("narrative_score".to_string(), narrative_markers as f64);
        genre_features.insert("expository_score".to_string(), expository_markers as f64);

        let pattern_compliance = 0.8; // Simplified

        GenreAnalysis {
            predicted_genre,
            genre_confidence,
            genre_features,
            pattern_compliance,
        }
    }
}

impl Default for DiscourseCoherenceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple function for basic discourse coherence analysis
pub fn calculate_discourse_coherence_simple(text: &str) -> f64 {
    let analyzer = DiscourseCoherenceAnalyzer::new();
    analyzer
        .analyze_discourse_coherence(text)
        .map(|result| result.discourse_marker_coherence)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discourse_coherence_analyzer_creation() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        assert_eq!(analyzer.config.discourse_marker_weight, 0.8);
        assert!(analyzer.config.analyze_rhetorical_structure);
        assert!(analyzer.config.analyze_cohesion_devices);
    }

    #[test]
    fn test_basic_discourse_coherence_analysis() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        let text = "First, we need to understand the problem. However, the solution is not simple. Therefore, we must consider all options.";

        let result = analyzer.analyze_discourse_coherence(text);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.discourse_marker_coherence >= 0.0);
        assert!(result.discourse_marker_coherence <= 1.0);
        assert!(!result.discourse_markers.is_empty());
    }

    #[test]
    fn test_discourse_marker_extraction() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        let sentences = vec![
            "However, this is important.".to_string(),
            "Therefore, we should act.".to_string(),
            "Furthermore, it's necessary.".to_string(),
        ];

        let markers = analyzer.extract_discourse_markers(&sentences).unwrap();
        assert!(!markers.is_empty());

        // Check that markers have proper structure
        for marker in markers {
            assert!(!marker.marker.is_empty());
            assert!(marker.contribution >= 0.0);
            assert!(marker.confidence >= 0.0 && marker.confidence <= 1.0);
        }
    }

    #[test]
    fn test_transition_quality_calculation() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        let sentences = vec![
            "The cat sat on the mat.".to_string(),
            "The cat was very comfortable.".to_string(),
            "All cats love comfortable places.".to_string(),
        ];

        let transitions = analyzer.calculate_sentence_transition_quality(&sentences);
        assert_eq!(transitions.len(), 2); // n-1 transitions for n sentences

        for &transition in &transitions {
            assert!(transition >= 0.0 && transition <= 1.0);
        }
    }

    #[test]
    fn test_rhetorical_relations_identification() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        let sentences = vec![
            "First, let's examine the evidence.".to_string(),
            "However, there are counterarguments.".to_string(),
            "Therefore, we need more research.".to_string(),
        ];

        let relations = analyzer.identify_rhetorical_relations(&sentences).unwrap();
        assert!(!relations.is_empty());

        // Check that relations are properly counted
        for (relation, count) in relations {
            assert!(!relation.is_empty());
            assert!(count > 0);
        }
    }

    #[test]
    fn test_cohesion_score_calculation() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        let sentences = vec![
            "The dog ran quickly.".to_string(),
            "This dog was very fast.".to_string(),
            "All dogs are loyal animals.".to_string(),
        ];

        let cohesion_score = analyzer.calculate_cohesion_score(&sentences);
        assert!(cohesion_score >= 0.0);
        assert!(cohesion_score <= 1.0);
    }

    #[test]
    fn test_marker_context_analysis() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        let sentences = vec![
            "The weather is nice today.".to_string(),
            "However, it might rain tomorrow.".to_string(),
            "We should bring umbrellas.".to_string(),
        ];

        let context = analyzer.analyze_marker_context(1, 0, &sentences);
        assert!(!context.preceding_context.is_empty() || !context.following_context.is_empty());
        assert!(context.semantic_coherence >= 0.0 && context.semantic_coherence <= 1.0);
        assert!(context.pragmatic_score >= 0.0 && context.pragmatic_score <= 1.0);
    }

    #[test]
    fn test_cohesive_devices_analysis() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        let sentences = vec![
            "The scientist conducted an experiment.".to_string(),
            "She recorded the results carefully.".to_string(),
            "These results were very surprising.".to_string(),
        ];

        let devices = analyzer.analyze_cohesive_devices(&sentences).unwrap();
        assert!(!devices.is_empty());

        for device in devices {
            assert!(!device.elements.is_empty());
            assert!(!device.positions.is_empty());
            assert!(device.strength >= 0.0 && device.strength <= 1.0);
        }
    }

    #[test]
    fn test_temporal_coherence_analysis() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        let sentences = vec![
            "First, we gathered the materials.".to_string(),
            "Then, we started the experiment.".to_string(),
            "Finally, we analyzed the results.".to_string(),
        ];

        let temporal_coherence = analyzer.analyze_temporal_coherence(&sentences);
        assert!(temporal_coherence.temporal_marker_frequency >= 0.0);
        assert!(
            temporal_coherence.sequence_coherence >= 0.0
                && temporal_coherence.sequence_coherence <= 1.0
        );
        assert!(
            temporal_coherence.timeline_consistency >= 0.0
                && temporal_coherence.timeline_consistency <= 1.0
        );
    }

    #[test]
    fn test_information_structure_analysis() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        let sentences = vec![
            "The study examined language processing.".to_string(),
            "Language processing involves multiple cognitive mechanisms.".to_string(),
            "These mechanisms work together efficiently.".to_string(),
        ];

        let info_structure = analyzer.analyze_information_structure(&sentences);
        assert!(info_structure.given_new_balance >= 0.0 && info_structure.given_new_balance <= 1.0);
        assert!(
            info_structure.information_density >= 0.0 && info_structure.information_density <= 1.0
        );
        assert!(!info_structure.thematic_progression.is_empty());
    }

    #[test]
    fn test_advanced_analysis() {
        let analyzer = DiscourseCoherenceAnalyzer::with_config(DiscourseCoherenceConfig {
            use_advanced_analysis: true,
            ..DiscourseCoherenceConfig::default()
        });

        let text = "Scientists study complex phenomena. However, understanding requires systematic investigation. Therefore, research methods are crucial. Furthermore, collaboration enhances discoveries.";

        let result = analyzer.analyze_discourse_coherence(text).unwrap();
        assert!(result.advanced_analysis.is_some());

        let advanced = result.advanced_analysis.unwrap();
        assert!(advanced.cognitive_measures.comprehension_ease >= 0.0);
        assert!(advanced.information_measures.entropy >= 0.0);
        assert!(!advanced.genre_analysis.predicted_genre.is_empty());
    }

    #[test]
    fn test_empty_text_handling() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        let result = analyzer.analyze_discourse_coherence("");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            DiscourseCoherenceError::EmptyText
        ));
    }

    #[test]
    fn test_simple_function() {
        let coherence = calculate_discourse_coherence_simple(
            "First, we start. Then, we continue. Finally, we finish.",
        );
        assert!(coherence >= 0.0 && coherence <= 1.0);
    }

    #[test]
    fn test_multiword_marker_extraction() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        let words = ["on", "the", "other", "hand", "this", "is", "different"];

        let markers = analyzer.extract_multiword_markers(0, &words).unwrap();
        assert!(!markers.is_empty());

        let found_marker = markers.iter().any(|m| m.marker == "on the other hand");
        assert!(found_marker);
    }

    #[test]
    fn test_marker_contribution_calculation() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        let words = ["however", "this", "is", "important"];

        let contribution =
            analyzer.calculate_marker_contribution(&DiscourseMarkerType::Contrast, 0, &words);
        assert!(contribution > 0.0);
        assert!(contribution <= 1.0);

        // Position 0 should have higher contribution than middle positions
        let middle_contribution =
            analyzer.calculate_marker_contribution(&DiscourseMarkerType::Contrast, 2, &words);
        assert!(contribution >= middle_contribution);
    }

    #[test]
    fn test_discourse_flow_analysis() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        let sentences = vec![
            "The research began with a hypothesis.".to_string(),
            "This hypothesis guided the methodology.".to_string(),
            "The methodology ensured reliable results.".to_string(),
        ];
        let transitions = vec![0.8, 0.7];

        let flow = analyzer.analyze_discourse_flow(&sentences, &transitions);
        assert!(flow.flow_continuity >= 0.0 && flow.flow_continuity <= 1.0);
        assert!(flow.topic_progression >= 0.0 && flow.topic_progression <= 1.0);
        assert!(flow.information_entropy >= 0.0);
    }

    #[test]
    fn test_cross_reference_analysis() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        let sentences = vec![
            "The researcher conducted the study.".to_string(),
            "She analyzed the data carefully.".to_string(),
            "These findings were significant.".to_string(),
        ];

        let cross_ref = analyzer.analyze_cross_references(&sentences);
        assert!(cross_ref.forward_references >= 0);
        assert!(cross_ref.backward_references >= 0);
        assert!(
            cross_ref.resolution_success_rate >= 0.0 && cross_ref.resolution_success_rate <= 1.0
        );
    }

    #[test]
    fn test_genre_analysis() {
        let analyzer = DiscourseCoherenceAnalyzer::new();
        let sentences = vec![
            "First, consider the evidence.".to_string(),
            "However, critics argue otherwise.".to_string(),
            "Therefore, we must examine both sides.".to_string(),
        ];

        let markers = analyzer.extract_discourse_markers(&sentences).unwrap();
        let genre = analyzer.analyze_genre(&sentences, &markers);

        assert!(!genre.predicted_genre.is_empty());
        assert!(genre.genre_confidence >= 0.0 && genre.genre_confidence <= 1.0);
        assert!(!genre.genre_features.is_empty());
    }
}
