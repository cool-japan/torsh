//! Result structures and data types for discourse coherence analysis
//!
//! This module contains all result structures, metrics, and supporting data types
//! used in discourse coherence analysis, providing comprehensive serialization
//! support and detailed analysis results.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

use super::config::{
    CohesiveDeviceType, DiscourseMarkerType, RhetoricalRelationType, TransitionQuality,
};

/// Comprehensive result of discourse coherence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseCoherenceResult {
    /// Overall discourse coherence score (0.0 to 1.0)
    pub overall_coherence_score: f64,
    /// Discourse marker coherence contribution
    pub marker_coherence_score: f64,
    /// Rhetorical structure coherence contribution
    pub rhetorical_structure_score: f64,
    /// Cohesion quality score
    pub cohesion_score: f64,
    /// Average transition quality scores
    pub transition_scores: Vec<f64>,
    /// Identified discourse markers
    pub discourse_markers: Vec<DiscourseMarker>,
    /// Identified rhetorical relations
    pub rhetorical_relations: HashMap<String, usize>,
    /// Detailed metrics (optional for comprehensive analysis)
    pub detailed_metrics: Option<DetailedDiscourseMetrics>,
}

/// Detailed metrics for comprehensive discourse analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedDiscourseMetrics {
    /// Discourse marker analysis details
    pub marker_analysis: Option<DiscourseMarkerAnalysis>,
    /// Rhetorical structure analysis details
    pub rhetorical_analysis: Option<RhetoricalStructureAnalysis>,
    /// Cohesion analysis details
    pub cohesion_analysis: Option<CohesionAnalysis>,
    /// Transition analysis details
    pub transition_analysis: Option<TransitionAnalysis>,
    /// Information structure analysis
    pub information_structure: Option<InformationStructureMetrics>,
    /// Advanced analysis results
    pub advanced_analysis: Option<AdvancedDiscourseAnalysis>,
}

/// Individual discourse marker with context and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseMarker {
    /// Type of discourse marker
    pub marker_type: DiscourseMarkerType,
    /// Text content of the marker
    pub text: String,
    /// Position in original text
    pub position: (usize, usize),
    /// Sentence index
    pub sentence_index: usize,
    /// Word index within sentence
    pub word_index: usize,
    /// Context analysis
    pub context: ContextAnalysis,
    /// Marker confidence score
    pub confidence: f64,
    /// Rhetorical strength
    pub rhetorical_strength: f64,
    /// Scope of influence (number of sentences affected)
    pub scope: usize,
    /// Syntactic position information
    pub syntactic_position: SyntacticPosition,
}

/// Context analysis for discourse markers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextAnalysis {
    /// Preceding context words
    pub preceding_context: Vec<String>,
    /// Following context words
    pub following_context: Vec<String>,
    /// Semantic coherence with preceding context
    pub semantic_coherence_preceding: f64,
    /// Semantic coherence with following context
    pub semantic_coherence_following: f64,
    /// Pragmatic appropriateness score
    pub pragmatic_appropriateness: f64,
}

/// Syntactic position information for markers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntacticPosition {
    /// Position within sentence (beginning, middle, end)
    pub sentence_position: String,
    /// Clause boundary information
    pub clause_boundary: bool,
    /// Part of speech tag
    pub pos_tag: String,
    /// Dependency relation
    pub dependency_relation: String,
}

/// Detailed discourse marker analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseMarkerAnalysis {
    /// Total marker count
    pub total_markers: usize,
    /// Marker density per 100 words
    pub marker_density: f64,
    /// Distribution by marker type
    pub type_distribution: HashMap<DiscourseMarkerType, usize>,
    /// Average confidence score
    pub average_confidence: f64,
    /// Marker effectiveness score
    pub effectiveness_score: f64,
    /// Context integration score
    pub context_integration: f64,
    /// Multiword marker statistics
    pub multiword_statistics: MultiwordMarkerStats,
}

/// Statistics for multiword discourse markers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiwordMarkerStats {
    /// Count of multiword markers
    pub multiword_count: usize,
    /// Average length of multiword markers
    pub average_length: f64,
    /// Most common multiword patterns
    pub common_patterns: Vec<(String, usize)>,
}

/// Rhetorical structure analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhetoricalStructureAnalysis {
    /// Identified rhetorical relations
    pub relations: HashMap<RhetoricalRelationType, usize>,
    /// Relation distribution score
    pub relation_distribution_score: f64,
    /// Structural complexity measure
    pub structural_complexity: f64,
    /// Discourse tree (if built)
    pub discourse_tree: Option<DiscourseTree>,
    /// Nucleus-satellite analysis
    pub nucleus_satellite_analysis: Option<NucleusSatelliteAnalysis>,
    /// Relation confidence scores
    pub relation_confidences: HashMap<RhetoricalRelationType, f64>,
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
    /// Tree complexity measure
    pub complexity_score: f64,
}

/// Individual discourse tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseNode {
    /// Node identifier
    pub node_id: usize,
    /// Rhetorical relation type
    pub relation_type: RhetoricalRelationType,
    /// Nucleus or satellite designation
    pub nuclearity: String,
    /// Text span covered by this node
    pub text_span: (usize, usize),
    /// Child nodes
    pub children: Vec<DiscourseNode>,
    /// Confidence score for this relation
    pub confidence: f64,
    /// Salience score
    pub salience: f64,
}

/// Analysis of nucleus-satellite structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NucleusSatelliteAnalysis {
    /// Nucleus identification accuracy
    pub nucleus_accuracy: f64,
    /// Satellite attachment patterns
    pub satellite_patterns: HashMap<String, usize>,
    /// Nuclear chain analysis
    pub nuclear_chains: Vec<NuclearChain>,
    /// Embedding depth statistics
    pub embedding_statistics: EmbeddingStats,
}

/// Representation of a nuclear chain in discourse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NuclearChain {
    /// Chain identifier
    pub chain_id: usize,
    /// Nuclear elements in order
    pub nuclear_elements: Vec<usize>,
    /// Attached satellites
    pub satellites: Vec<usize>,
    /// Chain coherence score
    pub coherence_score: f64,
}

/// Statistics about discourse structure embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingStats {
    /// Maximum embedding depth
    pub max_depth: usize,
    /// Average embedding depth
    pub average_depth: f64,
    /// Depth distribution
    pub depth_distribution: HashMap<usize, usize>,
}

/// Cohesion analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionAnalysis {
    /// Overall cohesion score
    pub overall_cohesion_score: f64,
    /// Identified cohesive devices
    pub cohesive_devices: Vec<CohesiveDevice>,
    /// Reference cohesion metrics
    pub reference_cohesion: ReferenceCohesionMetrics,
    /// Lexical cohesion metrics
    pub lexical_cohesion: LexicalCohesionMetrics,
    /// Conjunctive cohesion metrics
    pub conjunctive_cohesion: ConjunctiveCohesionMetrics,
    /// Temporal coherence metrics
    pub temporal_coherence: TemporalCoherenceMetrics,
}

/// Individual cohesive device analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesiveDevice {
    /// Type of cohesive device
    pub device_type: CohesiveDeviceType,
    /// Text elements involved in cohesion
    pub elements: Vec<String>,
    /// Positions of elements in text
    pub positions: Vec<(usize, usize)>,
    /// Cohesive strength score
    pub strength: f64,
    /// Local coherence contribution
    pub local_contribution: f64,
    /// Global coherence contribution
    pub global_contribution: f64,
    /// Resolution confidence
    pub resolution_confidence: f64,
    /// Distance between cohesive elements
    pub distance: usize,
}

/// Reference cohesion analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceCohesionMetrics {
    /// Total reference links
    pub total_references: usize,
    /// Reference density per sentence
    pub reference_density: f64,
    /// Successful reference resolution rate
    pub resolution_success_rate: f64,
    /// Average reference distance
    pub average_reference_distance: f64,
    /// Ambiguous references count
    pub ambiguous_references: usize,
    /// Reference complexity score
    pub complexity_score: f64,
    /// Chain analysis
    pub reference_chains: Vec<ReferenceChain>,
}

/// Analysis of reference chains in text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceChain {
    /// Chain identifier
    pub chain_id: usize,
    /// Referenced entity or concept
    pub entity: String,
    /// Referring expressions in order
    pub referring_expressions: Vec<String>,
    /// Positions of expressions
    pub positions: Vec<(usize, usize)>,
    /// Chain coherence score
    pub coherence_score: f64,
    /// Chain completeness score
    pub completeness_score: f64,
}

/// Lexical cohesion analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalCohesionMetrics {
    /// Lexical tie count
    pub lexical_ties: usize,
    /// Lexical density score
    pub lexical_density: f64,
    /// Repetition analysis
    pub repetition_analysis: RepetitionAnalysis,
    /// Synonym network metrics
    pub synonym_networks: SynonymNetworkMetrics,
    /// Semantic field coherence
    pub semantic_field_coherence: f64,
    /// Lexical sophistication score
    pub sophistication_score: f64,
}

/// Analysis of lexical repetition patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepetitionAnalysis {
    /// Exact repetitions count
    pub exact_repetitions: usize,
    /// Morphological variations count
    pub morphological_variations: usize,
    /// Most repeated terms
    pub frequent_terms: Vec<(String, usize)>,
    /// Repetition distribution score
    pub distribution_score: f64,
}

/// Metrics for synonym networks in text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynonymNetworkMetrics {
    /// Number of synonym clusters
    pub cluster_count: usize,
    /// Average cluster size
    pub average_cluster_size: f64,
    /// Network connectivity score
    pub connectivity_score: f64,
    /// Largest clusters
    pub major_clusters: Vec<SynonymCluster>,
}

/// Individual synonym cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynonymCluster {
    /// Cluster identifier
    pub cluster_id: usize,
    /// Words in the cluster
    pub words: Vec<String>,
    /// Cluster coherence score
    pub coherence_score: f64,
    /// Semantic similarity threshold
    pub similarity_threshold: f64,
}

/// Conjunctive cohesion analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConjunctiveCohesionMetrics {
    /// Conjunction count by type
    pub conjunction_counts: HashMap<String, usize>,
    /// Conjunctive density
    pub conjunctive_density: f64,
    /// Logical flow score
    pub logical_flow_score: f64,
    /// Conjunction effectiveness
    pub conjunction_effectiveness: f64,
    /// Complex conjunction analysis
    pub complex_conjunctions: Vec<ComplexConjunction>,
}

/// Analysis of complex conjunctive structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexConjunction {
    /// Conjunction text
    pub text: String,
    /// Logical relation expressed
    pub logical_relation: String,
    /// Position in text
    pub position: (usize, usize),
    /// Effectiveness score
    pub effectiveness: f64,
    /// Scope of conjunction
    pub scope: usize,
}

/// Temporal coherence analysis metrics
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
    /// Temporal reference chains
    pub temporal_chains: Vec<TemporalChain>,
}

/// Analysis of temporal reference chains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalChain {
    /// Chain identifier
    pub chain_id: usize,
    /// Temporal expressions
    pub expressions: Vec<String>,
    /// Temporal ordering
    pub ordering: Vec<usize>,
    /// Consistency score
    pub consistency_score: f64,
}

/// Transition analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionAnalysis {
    /// Individual transition quality scores
    pub transition_quality_scores: Vec<TransitionQualityScore>,
    /// Overall transition quality
    pub overall_transition_quality: f64,
    /// Lexical overlap statistics
    pub lexical_overlap_stats: LexicalOverlapStats,
    /// Semantic continuity metrics
    pub semantic_continuity: SemanticContinuityMetrics,
    /// Structural continuity analysis
    pub structural_continuity: StructuralContinuityMetrics,
}

/// Quality score for individual transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionQualityScore {
    /// Transition position (between sentences i and i+1)
    pub position: (usize, usize),
    /// Quality classification
    pub quality: TransitionQuality,
    /// Numerical quality score
    pub score: f64,
    /// Contributing factors
    pub factors: TransitionFactors,
}

/// Factors contributing to transition quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionFactors {
    /// Lexical overlap contribution
    pub lexical_overlap: f64,
    /// Semantic continuity contribution
    pub semantic_continuity: f64,
    /// Structural continuity contribution
    pub structural_continuity: f64,
    /// Transition marker presence
    pub marker_presence: f64,
}

/// Statistics for lexical overlap between sentences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalOverlapStats {
    /// Average lexical overlap score
    pub average_overlap: f64,
    /// Distribution of overlap scores
    pub overlap_distribution: HashMap<String, usize>,
    /// Content word vs function word overlap
    pub content_vs_function: ContentFunctionOverlap,
}

/// Comparison of content vs function word overlap
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentFunctionOverlap {
    /// Content word overlap average
    pub content_word_overlap: f64,
    /// Function word overlap average
    pub function_word_overlap: f64,
    /// Ratio of content to function overlap
    pub content_function_ratio: f64,
}

/// Semantic continuity analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticContinuityMetrics {
    /// Average semantic similarity scores
    pub average_semantic_similarity: f64,
    /// Semantic coherence distribution
    pub coherence_distribution: Vec<f64>,
    /// Topic continuity analysis
    pub topic_continuity: TopicContinuityMetrics,
    /// Conceptual overlap metrics
    pub conceptual_overlap: ConceptualOverlapMetrics,
}

/// Analysis of topic continuity across sentences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicContinuityMetrics {
    /// Topic shift frequency
    pub topic_shift_frequency: f64,
    /// Average topic coherence
    pub average_topic_coherence: f64,
    /// Major topic transitions
    pub major_transitions: Vec<TopicTransition>,
}

/// Individual topic transition analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicTransition {
    /// Position of transition
    pub position: usize,
    /// Source topic indicators
    pub source_topic: Vec<String>,
    /// Target topic indicators
    pub target_topic: Vec<String>,
    /// Transition smoothness score
    pub smoothness: f64,
}

/// Analysis of conceptual overlap between sentences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptualOverlapMetrics {
    /// Concept extraction statistics
    pub concept_stats: ConceptStats,
    /// Conceptual similarity network
    pub similarity_network: ConceptualSimilarityNetwork,
    /// Abstract concept tracking
    pub abstract_concepts: Vec<AbstractConcept>,
}

/// Statistics about concept extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptStats {
    /// Total concepts identified
    pub total_concepts: usize,
    /// Average concepts per sentence
    pub concepts_per_sentence: f64,
    /// Concept diversity score
    pub diversity_score: f64,
}

/// Network of conceptual similarities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptualSimilarityNetwork {
    /// Network nodes (concepts)
    pub nodes: Vec<String>,
    /// Network edges (similarities)
    pub edges: Vec<(usize, usize, f64)>,
    /// Network density
    pub density: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
}

/// Abstract concept tracking across discourse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractConcept {
    /// Concept identifier
    pub concept_id: String,
    /// Manifestations in text
    pub manifestations: Vec<String>,
    /// Positions of manifestations
    pub positions: Vec<usize>,
    /// Conceptual stability score
    pub stability_score: f64,
}

/// Structural continuity analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralContinuityMetrics {
    /// Syntactic pattern continuity
    pub syntactic_continuity: f64,
    /// Sentence length variation
    pub length_variation: SentenceLengthVariation,
    /// Structural parallelism
    pub parallelism_analysis: ParallelismAnalysis,
}

/// Analysis of sentence length variation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentenceLengthVariation {
    /// Average sentence length
    pub average_length: f64,
    /// Length standard deviation
    pub length_std_dev: f64,
    /// Length variation score
    pub variation_score: f64,
    /// Length distribution
    pub length_distribution: HashMap<String, usize>,
}

/// Analysis of structural parallelism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelismAnalysis {
    /// Parallel structure count
    pub parallel_structures: usize,
    /// Parallelism strength score
    pub parallelism_strength: f64,
    /// Identified parallel patterns
    pub parallel_patterns: Vec<ParallelPattern>,
}

/// Individual parallel structure pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelPattern {
    /// Pattern identifier
    pub pattern_id: usize,
    /// Parallel elements
    pub elements: Vec<String>,
    /// Element positions
    pub positions: Vec<usize>,
    /// Pattern strength
    pub strength: f64,
}

/// Information structure analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationStructureMetrics {
    /// Given-new information balance
    pub given_new_balance: f64,
    /// Topic-focus articulation score
    pub topic_focus_articulation: f64,
    /// Information packaging score
    pub information_packaging: f64,
    /// Thematic progression pattern
    pub thematic_progression: String,
    /// Information density metrics
    pub information_density: f64,
    /// Theme analysis
    pub theme_analysis: ThemeAnalysis,
    /// Focus tracking
    pub focus_tracking: FocusTracking,
}

/// Analysis of thematic elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeAnalysis {
    /// Identified themes
    pub themes: Vec<Theme>,
    /// Theme continuity score
    pub continuity_score: f64,
    /// Theme development patterns
    pub development_patterns: Vec<String>,
}

/// Individual theme in discourse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Theme {
    /// Theme identifier
    pub theme_id: String,
    /// Theme manifestations
    pub manifestations: Vec<String>,
    /// Positions in text
    pub positions: Vec<usize>,
    /// Theme prominence score
    pub prominence: f64,
}

/// Focus tracking across discourse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusTracking {
    /// Focus shifts count
    pub focus_shifts: usize,
    /// Focus continuity score
    pub continuity_score: f64,
    /// Major focus elements
    pub major_foci: Vec<FocusElement>,
}

/// Individual focus element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusElement {
    /// Focus content
    pub content: String,
    /// Position in text
    pub position: usize,
    /// Focus strength
    pub strength: f64,
}

/// Advanced discourse analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedDiscourseAnalysis {
    /// Discourse tree structure (if built)
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

/// Network representation of coherence relations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceNetwork {
    /// Network nodes (discourse units)
    pub nodes: Vec<CoherenceNode>,
    /// Network edges (coherence relations)
    pub edges: Vec<CoherenceEdge>,
    /// Network density measure
    pub density: f64,
    /// Network connectivity score
    pub connectivity: f64,
    /// Central nodes analysis
    pub central_nodes: Vec<usize>,
}

/// Node in coherence network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceNode {
    /// Node identifier
    pub node_id: usize,
    /// Text content
    pub content: String,
    /// Centrality measures
    pub centrality: NodeCentrality,
    /// Node importance score
    pub importance: f64,
}

/// Centrality measures for network nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCentrality {
    /// Degree centrality
    pub degree: f64,
    /// Betweenness centrality
    pub betweenness: f64,
    /// Closeness centrality
    pub closeness: f64,
    /// Eigenvector centrality
    pub eigenvector: f64,
}

/// Edge in coherence network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceEdge {
    /// Source node
    pub source: usize,
    /// Target node
    pub target: usize,
    /// Relation type
    pub relation_type: RhetoricalRelationType,
    /// Edge weight (strength)
    pub weight: f64,
    /// Confidence score
    pub confidence: f64,
}

/// Information-theoretic measures for discourse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseInformationMeasures {
    /// Entropy measures
    pub entropy: EntropyMeasures,
    /// Information flow metrics
    pub information_flow: InformationFlowMetrics,
    /// Redundancy analysis
    pub redundancy: RedundancyAnalysis,
    /// Compression analysis
    pub compression: CompressionAnalysis,
}

/// Entropy-based information measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyMeasures {
    /// Lexical entropy
    pub lexical_entropy: f64,
    /// Syntactic entropy
    pub syntactic_entropy: f64,
    /// Semantic entropy
    pub semantic_entropy: f64,
    /// Overall discourse entropy
    pub discourse_entropy: f64,
}

/// Information flow analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationFlowMetrics {
    /// Information rate
    pub information_rate: f64,
    /// Flow consistency
    pub flow_consistency: f64,
    /// Information bottlenecks
    pub bottlenecks: Vec<InformationBottleneck>,
}

/// Information bottleneck in discourse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationBottleneck {
    /// Position of bottleneck
    pub position: usize,
    /// Bottleneck severity
    pub severity: f64,
    /// Contributing factors
    pub factors: Vec<String>,
}

/// Redundancy analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundancyAnalysis {
    /// Overall redundancy score
    pub overall_redundancy: f64,
    /// Beneficial redundancy
    pub beneficial_redundancy: f64,
    /// Excessive redundancy
    pub excessive_redundancy: f64,
    /// Redundancy patterns
    pub patterns: Vec<RedundancyPattern>,
}

/// Individual redundancy pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundancyPattern {
    /// Pattern type
    pub pattern_type: String,
    /// Instances of pattern
    pub instances: Vec<(usize, usize)>,
    /// Pattern strength
    pub strength: f64,
}

/// Text compression analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionAnalysis {
    /// Compression ratio
    pub compression_ratio: f64,
    /// Information density
    pub information_density: f64,
    /// Compressibility score
    pub compressibility: f64,
}

/// Cognitive processing measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveMeasures {
    /// Processing load estimate
    pub processing_load: f64,
    /// Memory load estimate
    pub memory_load: f64,
    /// Attention tracking metrics
    pub attention_metrics: AttentionMetrics,
    /// Comprehension difficulty score
    pub comprehension_difficulty: f64,
    /// Cognitive accessibility score
    pub cognitive_accessibility: f64,
}

/// Attention-based discourse metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionMetrics {
    /// Attention switching frequency
    pub attention_switches: usize,
    /// Focus maintenance score
    pub focus_maintenance: f64,
    /// Attention distribution
    pub attention_distribution: Vec<AttentionFocus>,
}

/// Individual attention focus point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionFocus {
    /// Position in text
    pub position: usize,
    /// Focus strength
    pub strength: f64,
    /// Duration estimate
    pub duration: f64,
}

/// Genre analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenreAnalysis {
    /// Predicted genre
    pub predicted_genre: String,
    /// Genre confidence score
    pub confidence: f64,
    /// Genre-specific features
    pub genre_features: GenreFeatures,
    /// Register analysis
    pub register_analysis: RegisterAnalysis,
}

/// Features characteristic of discourse genres
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenreFeatures {
    /// Argumentative features
    pub argumentative: ArgumentativeFeatures,
    /// Narrative features
    pub narrative: NarrativeFeatures,
    /// Expository features
    pub expository: ExpositoryFeatures,
    /// Descriptive features
    pub descriptive: DescriptiveFeatures,
}

/// Features of argumentative discourse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgumentativeFeatures {
    /// Claim-evidence patterns
    pub claim_evidence_patterns: usize,
    /// Logical connectors frequency
    pub logical_connectors: f64,
    /// Counter-argument presence
    pub counter_arguments: usize,
    /// Persuasive strength
    pub persuasive_strength: f64,
}

/// Features of narrative discourse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeFeatures {
    /// Temporal progression markers
    pub temporal_markers: usize,
    /// Character reference frequency
    pub character_references: f64,
    /// Action verb density
    pub action_verb_density: f64,
    /// Narrative coherence score
    pub narrative_coherence: f64,
}

/// Features of expository discourse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpositoryFeatures {
    /// Definition patterns
    pub definition_patterns: usize,
    /// Classification markers
    pub classification_markers: usize,
    /// Explanation structures
    pub explanation_structures: usize,
    /// Technical vocabulary density
    pub technical_vocabulary: f64,
}

/// Features of descriptive discourse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptiveFeatures {
    /// Descriptive adjective density
    pub adjective_density: f64,
    /// Spatial markers frequency
    pub spatial_markers: f64,
    /// Sensory detail indicators
    pub sensory_details: usize,
    /// Descriptive vividness score
    pub vividness_score: f64,
}

/// Register analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterAnalysis {
    /// Formality level
    pub formality_level: String,
    /// Formality score (0.0 = informal, 1.0 = very formal)
    pub formality_score: f64,
    /// Register features
    pub register_features: RegisterFeatures,
    /// Audience adaptation score
    pub audience_adaptation: f64,
}

/// Features indicating discourse register
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterFeatures {
    /// Lexical sophistication
    pub lexical_sophistication: f64,
    /// Syntactic complexity
    pub syntactic_complexity: f64,
    /// Discourse marker formality
    pub marker_formality: f64,
    /// Technical language usage
    pub technical_usage: f64,
}
