//! Result structures and data types for lexical coherence analysis
//!
//! This module contains all result structures, metrics, and supporting data types
//! used in lexical coherence analysis, providing comprehensive serialization
//! support and detailed analysis results.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

use super::config::{CohesionDeviceType, LexicalChainType, SemanticRelationship};

/// Comprehensive result of lexical coherence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalCoherenceResult {
    /// Overall lexical coherence score (0.0 to 1.0)
    pub overall_coherence_score: f64,
    /// Lexical chain coherence contribution
    pub chain_coherence_score: f64,
    /// Semantic field coherence contribution
    pub semantic_field_score: f64,
    /// Vocabulary consistency score
    pub vocabulary_consistency: f64,
    /// Identified lexical chains
    pub lexical_chains: Vec<LexicalChain>,
    /// Semantic field mappings
    pub semantic_fields: HashMap<String, Vec<String>>,
    /// Detailed metrics (optional for comprehensive analysis)
    pub detailed_metrics: Option<DetailedLexicalMetrics>,
}

/// Individual lexical chain with detailed analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalChain {
    /// Unique chain identifier
    pub chain_id: usize,
    /// Words in the chain with their positions
    pub words: Vec<(String, Vec<(usize, usize)>)>,
    /// Type of lexical chain
    pub chain_type: LexicalChainType,
    /// Semantic relationship between chain elements
    pub semantic_relationship: SemanticRelationship,
    /// Chain coherence score
    pub coherence_score: f64,
    /// Chain strength (based on frequency and distribution)
    pub strength: f64,
    /// Average distance between chain elements
    pub average_distance: f64,
    /// Maximum distance between chain elements
    pub max_distance: f64,
    /// Chain coverage (sentences spanned)
    pub coverage: f64,
    /// Chain density (words per sentence in coverage)
    pub density: f64,
}

/// Detailed metrics for comprehensive lexical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedLexicalMetrics {
    /// Lexical diversity analysis
    pub lexical_diversity: Option<LexicalDiversityMetrics>,
    /// Pattern statistics
    pub pattern_statistics: Option<PatternStatistics>,
    /// Temporal coherence metrics
    pub temporal_coherence: Option<TemporalCoherenceMetrics>,
    /// Chain connectivity analysis
    pub chain_connectivity: Option<ChainConnectivity>,
    /// Cohesion device analysis
    pub cohesion_devices: Option<Vec<CohesionDevice>>,
    /// Advanced chain analysis
    pub advanced_analysis: Option<AdvancedChainAnalysis>,
}

/// Lexical diversity and richness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalDiversityMetrics {
    /// Type-token ratio
    pub type_token_ratio: f64,
    /// Moving average type-token ratio
    pub mattr: f64,
    /// Measure of textual lexical diversity
    pub mtld: f64,
    /// Honore's statistic
    pub honore_h: f64,
    /// Brunet's index
    pub brunet_w: f64,
    /// Unique word percentage
    pub unique_word_percentage: f64,
    /// Lexical sophistication score
    pub sophistication_score: f64,
    /// Vocabulary range
    pub vocabulary_range: VocabularyRange,
}

/// Vocabulary range analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabularyRange {
    /// High frequency words percentage
    pub high_frequency_percentage: f64,
    /// Mid frequency words percentage
    pub mid_frequency_percentage: f64,
    /// Low frequency words percentage
    pub low_frequency_percentage: f64,
    /// Academic vocabulary percentage
    pub academic_vocabulary_percentage: f64,
    /// Technical vocabulary percentage
    pub technical_vocabulary_percentage: f64,
}

/// Pattern statistics in lexical usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternStatistics {
    /// Repetition patterns
    pub repetition_patterns: HashMap<String, usize>,
    /// Morphological patterns
    pub morphological_patterns: HashMap<String, Vec<String>>,
    /// Collocation patterns
    pub collocation_patterns: HashMap<String, Vec<String>>,
    /// Semantic field patterns
    pub semantic_patterns: HashMap<String, usize>,
    /// Chain type distribution
    pub chain_type_distribution: HashMap<LexicalChainType, usize>,
    /// Relationship type distribution
    pub relationship_distribution: HashMap<SemanticRelationship, usize>,
}

/// Temporal coherence in lexical usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalCoherenceMetrics {
    /// Lexical consistency across text segments
    pub consistency_across_segments: f64,
    /// Vocabulary shift patterns
    pub vocabulary_shifts: Vec<VocabularyShift>,
    /// Periodic patterns in word usage
    pub periodic_patterns: Vec<PeriodicPattern>,
    /// Temporal clustering coefficient
    pub temporal_clustering: f64,
    /// Lexical flow smoothness
    pub flow_smoothness: f64,
}

/// Vocabulary shift at specific positions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabularyShift {
    /// Position in text (sentence index)
    pub position: usize,
    /// Shift magnitude
    pub magnitude: f64,
    /// Shift type description
    pub shift_type: String,
    /// Words involved in the shift
    pub shift_words: Vec<String>,
}

/// Periodic pattern in lexical usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodicPattern {
    /// Pattern identifier
    pub pattern_id: String,
    /// Pattern period (in sentences)
    pub period: usize,
    /// Pattern strength
    pub strength: f64,
    /// Words involved in the pattern
    pub pattern_words: Vec<String>,
    /// Pattern positions
    pub positions: Vec<usize>,
}

/// Positional distribution of lexical elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionDistribution {
    /// Distribution across text segments
    pub segment_distribution: HashMap<String, f64>,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Dispersion index
    pub dispersion_index: f64,
    /// Position variance
    pub position_variance: f64,
    /// Coverage uniformity
    pub coverage_uniformity: f64,
}

/// Chain connectivity and network properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainConnectivity {
    /// Average chain overlap
    pub average_overlap: f64,
    /// Chain interconnectedness
    pub interconnectedness: f64,
    /// Network density
    pub network_density: f64,
    /// Connected components count
    pub connected_components: usize,
    /// Largest component size
    pub largest_component_size: usize,
    /// Average path length
    pub average_path_length: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
}

/// Individual cohesion device analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionDevice {
    /// Type of cohesion device
    pub device_type: CohesionDeviceType,
    /// Source word or phrase
    pub source: String,
    /// Target word or phrase
    pub target: String,
    /// Position of source
    pub source_position: (usize, usize),
    /// Position of target
    pub target_position: (usize, usize),
    /// Distance between elements
    pub distance: usize,
    /// Cohesive strength
    pub strength: f64,
    /// Confidence in the relationship
    pub confidence: f64,
    /// Context relevance
    pub context_relevance: f64,
}

/// Advanced chain analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedChainAnalysis {
    /// Chain network analysis
    pub network_analysis: Option<ChainNetworkAnalysis>,
    /// Information-theoretic measures
    pub information_measures: Option<InformationMeasures>,
    /// Cognitive load metrics
    pub cognitive_metrics: Option<CognitiveMeasures>,
    /// Discourse alignment metrics
    pub discourse_alignment: Option<DiscourseAlignmentMetrics>,
}

/// Network analysis of lexical chains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainNetworkAnalysis {
    /// Number of nodes (unique words)
    pub node_count: usize,
    /// Number of edges (relationships)
    pub edge_count: usize,
    /// Network density
    pub density: f64,
    /// Average degree
    pub average_degree: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Average path length
    pub average_path_length: f64,
    /// Degree distribution
    pub degree_distribution: HashMap<usize, usize>,
    /// Central nodes (high degree/betweenness)
    pub central_nodes: Vec<CentralNode>,
}

/// Central node in the chain network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentralNode {
    /// Word or phrase
    pub word: String,
    /// Degree centrality
    pub degree_centrality: f64,
    /// Betweenness centrality
    pub betweenness_centrality: f64,
    /// Closeness centrality
    pub closeness_centrality: f64,
    /// PageRank score
    pub pagerank: f64,
}

/// Information-theoretic measures for lexical analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationMeasures {
    /// Lexical entropy
    pub lexical_entropy: f64,
    /// Chain entropy
    pub chain_entropy: f64,
    /// Mutual information between chains
    pub chain_mutual_information: f64,
    /// Information redundancy
    pub information_redundancy: f64,
    /// Conditional entropy
    pub conditional_entropy: f64,
    /// Information flow metrics
    pub information_flow: InformationFlowMetrics,
}

/// Information flow analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationFlowMetrics {
    /// Forward information flow
    pub forward_flow: f64,
    /// Backward information flow
    pub backward_flow: f64,
    /// Flow consistency
    pub flow_consistency: f64,
    /// Information accumulation
    pub information_accumulation: f64,
    /// Flow bottlenecks
    pub bottlenecks: Vec<InformationBottleneck>,
}

/// Information bottleneck in the text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationBottleneck {
    /// Position of bottleneck
    pub position: usize,
    /// Bottleneck severity
    pub severity: f64,
    /// Affected chains
    pub affected_chains: Vec<usize>,
}

/// Cognitive processing load metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveMeasures {
    /// Processing load estimate
    pub processing_load: f64,
    /// Memory load estimate
    pub memory_load: f64,
    /// Integration effort
    pub integration_effort: f64,
    /// Disambiguation load
    pub disambiguation_load: f64,
    /// Cognitive accessibility score
    pub accessibility_score: f64,
    /// Working memory demand
    pub working_memory_demand: f64,
    /// Cognitive complexity factors
    pub complexity_factors: CognitiveComplexityFactors,
}

/// Factors contributing to cognitive complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveComplexityFactors {
    /// Lexical complexity
    pub lexical_complexity: f64,
    /// Chain complexity
    pub chain_complexity: f64,
    /// Semantic complexity
    pub semantic_complexity: f64,
    /// Structural complexity
    pub structural_complexity: f64,
    /// Integration complexity
    pub integration_complexity: f64,
}

/// Discourse alignment metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseAlignmentMetrics {
    /// Topic-lexicon alignment
    pub topic_alignment: f64,
    /// Register consistency
    pub register_consistency: f64,
    /// Functional alignment
    pub functional_alignment: f64,
    /// Stylistic coherence
    pub stylistic_coherence: f64,
    /// Genre appropriateness
    pub genre_appropriateness: f64,
    /// Alignment analysis details
    pub alignment_details: AlignmentDetails,
}

/// Detailed alignment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentDetails {
    /// Topic-specific chains
    pub topic_chains: HashMap<String, Vec<usize>>,
    /// Register markers
    pub register_markers: Vec<RegisterMarker>,
    /// Functional categories
    pub functional_categories: HashMap<String, f64>,
    /// Stylistic features
    pub stylistic_features: HashMap<String, f64>,
}

/// Register marker analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisterMarker {
    /// Marker text
    pub marker: String,
    /// Register category
    pub category: String,
    /// Position in text
    pub position: (usize, usize),
    /// Marker strength
    pub strength: f64,
}

/// Lexical coherence statistics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalCoherenceStatistics {
    /// Total word count
    pub total_words: usize,
    /// Unique word count
    pub unique_words: usize,
    /// Total chains found
    pub total_chains: usize,
    /// Average chain length
    pub average_chain_length: f64,
    /// Lexical density
    pub lexical_density: f64,
    /// Coherence distribution
    pub coherence_distribution: CoherenceDistribution,
}

/// Distribution of coherence scores
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceDistribution {
    /// Score ranges and their frequencies
    pub score_ranges: BTreeMap<String, usize>,
    /// Mean coherence score
    pub mean_score: f64,
    /// Standard deviation
    pub standard_deviation: f64,
    /// Skewness
    pub skewness: f64,
    /// Kurtosis
    pub kurtosis: f64,
}

impl LexicalCoherenceResult {
    /// Create a basic result with minimal information
    pub fn basic(overall_score: f64, chains: Vec<LexicalChain>) -> Self {
        Self {
            overall_coherence_score: overall_score,
            chain_coherence_score: overall_score * 0.6,
            semantic_field_score: overall_score * 0.3,
            vocabulary_consistency: overall_score * 0.1,
            lexical_chains: chains,
            semantic_fields: HashMap::new(),
            detailed_metrics: None,
        }
    }

    /// Check if detailed metrics are available
    pub fn has_detailed_metrics(&self) -> bool {
        self.detailed_metrics.is_some()
    }

    /// Get chain statistics
    pub fn get_chain_statistics(&self) -> LexicalCoherenceStatistics {
        let total_chains = self.lexical_chains.len();
        let total_words: usize = self
            .lexical_chains
            .iter()
            .map(|chain| chain.words.len())
            .sum();
        let unique_words: usize = self
            .lexical_chains
            .iter()
            .flat_map(|chain| chain.words.iter().map(|(word, _)| word))
            .collect::<std::collections::HashSet<_>>()
            .len();

        let average_chain_length = if total_chains > 0 {
            self.lexical_chains
                .iter()
                .map(|chain| chain.words.len())
                .sum::<usize>() as f64
                / total_chains as f64
        } else {
            0.0
        };

        let scores: Vec<f64> = self
            .lexical_chains
            .iter()
            .map(|chain| chain.coherence_score)
            .collect();
        let mean_score = if !scores.is_empty() {
            scores.iter().sum::<f64>() / scores.len() as f64
        } else {
            0.0
        };

        let variance = if !scores.is_empty() {
            scores
                .iter()
                .map(|score| (score - mean_score).powi(2))
                .sum::<f64>()
                / scores.len() as f64
        } else {
            0.0
        };

        LexicalCoherenceStatistics {
            total_words,
            unique_words,
            total_chains,
            average_chain_length,
            lexical_density: if total_words > 0 {
                unique_words as f64 / total_words as f64
            } else {
                0.0
            },
            coherence_distribution: CoherenceDistribution {
                score_ranges: BTreeMap::new(), // Would be calculated in full implementation
                mean_score,
                standard_deviation: variance.sqrt(),
                skewness: 0.0, // Would be calculated
                kurtosis: 0.0, // Would be calculated
            },
        }
    }
}

impl LexicalChain {
    /// Get total word count in chain
    pub fn total_word_count(&self) -> usize {
        self.words
            .iter()
            .map(|(_, positions)| positions.len())
            .sum()
    }

    /// Get sentence span of the chain
    pub fn sentence_span(&self) -> (usize, usize) {
        let all_positions: Vec<usize> = self
            .words
            .iter()
            .flat_map(|(_, positions)| positions.iter().map(|(sent, _)| *sent))
            .collect();

        if all_positions.is_empty() {
            (0, 0)
        } else {
            (
                *all_positions.iter().min().unwrap(),
                *all_positions.iter().max().unwrap(),
            )
        }
    }

    /// Calculate chain complexity
    pub fn complexity(&self) -> f64 {
        let unique_words = self.words.len() as f64;
        let total_positions = self.total_word_count() as f64;
        let span = self.sentence_span();
        let sentence_span = (span.1 - span.0 + 1) as f64;

        // Complexity based on diversity, distribution, and span
        let diversity_factor = unique_words / total_positions.max(1.0);
        let distribution_factor = total_positions / sentence_span.max(1.0);
        let relationship_factor = match self.semantic_relationship {
            SemanticRelationship::Synonymy => 0.9,
            SemanticRelationship::Hyponymy => 0.8,
            SemanticRelationship::Meronymy => 0.7,
            _ => 0.5,
        };

        (diversity_factor + distribution_factor + relationship_factor) / 3.0
    }
}

impl DetailedLexicalMetrics {
    /// Check if advanced analysis is available
    pub fn has_advanced_analysis(&self) -> bool {
        self.advanced_analysis.is_some()
    }

    /// Get complexity summary
    pub fn complexity_summary(&self) -> f64 {
        let mut complexity = 0.0;
        let mut factors = 0;

        if let Some(diversity) = &self.lexical_diversity {
            complexity += 1.0 - diversity.type_token_ratio; // Lower TTR = higher complexity
            factors += 1;
        }

        if let Some(connectivity) = &self.chain_connectivity {
            complexity += connectivity.network_density;
            factors += 1;
        }

        if let Some(advanced) = &self.advanced_analysis {
            if let Some(cognitive) = &advanced.cognitive_metrics {
                complexity += cognitive.processing_load;
                factors += 1;
            }
        }

        if factors > 0 {
            complexity / factors as f64
        } else {
            0.0
        }
    }
}
