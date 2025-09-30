//! Result Structures for Semantic Fluency Analysis
//!
//! This module provides comprehensive result structures for semantic fluency analysis,
//! including detailed metrics, analysis results, and supporting data types with full
//! serialization support.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::config::{CoherenceRelationType, FocusType, InconsistencyType};

/// Main result structure for semantic fluency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFluencyResult {
    /// Overall semantic fluency score (0.0 to 1.0)
    pub overall_score: f64,
    /// Semantic coherence score
    pub semantic_coherence: f64,
    /// Meaning preservation score
    pub meaning_preservation: f64,
    /// Conceptual clarity score
    pub conceptual_clarity: f64,
    /// Semantic appropriateness score
    pub semantic_appropriateness: f64,
    /// Context sensitivity score
    pub context_sensitivity: f64,
    /// Semantic density measure
    pub semantic_density: f64,
    /// Ambiguity score (lower is better)
    pub ambiguity_score: f64,
    /// Semantic relations analysis
    pub semantic_relations: HashMap<String, f64>,
    /// Advanced metrics (if enabled)
    pub advanced_metrics: Option<AdvancedSemanticMetrics>,
    /// Analysis insights and recommendations
    pub insights: Vec<String>,
    /// Detailed breakdown by component
    pub component_scores: HashMap<String, f64>,
    /// Analysis metadata
    pub metadata: AnalysisMetadata,
}

/// Advanced semantic metrics for detailed analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedSemanticMetrics {
    /// Semantic field coverage analysis
    pub field_coverage: SemanticFieldCoverage,
    /// Conceptual complexity analysis
    pub conceptual_complexity: ConceptualComplexity,
    /// Consistency analysis results
    pub consistency_analysis: ConsistencyAnalysis,
    /// Topic coherence analysis
    pub topic_coherence: Option<TopicCoherence>,
    /// Semantic role analysis
    pub semantic_roles: Option<SemanticRoleAnalysis>,
    /// Figurative language analysis
    pub figurative_language: Option<FigurativeLanguageAnalysis>,
    /// Discourse analysis results
    pub discourse_analysis: Option<DiscourseAnalysis>,
    /// Information structure analysis
    pub information_structure: Option<InformationStructure>,
}

/// Semantic field coverage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFieldCoverage {
    /// Overall coverage score
    pub overall_coverage: f64,
    /// Identified semantic fields
    pub semantic_fields: HashMap<String, f64>,
    /// Field distribution evenness
    pub distribution_evenness: f64,
    /// Field transition coherence
    pub transition_coherence: f64,
    /// Dominant field
    pub dominant_field: Option<String>,
    /// Field diversity index
    pub diversity_index: f64,
}

/// Conceptual complexity analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptualComplexity {
    /// Overall complexity score
    pub overall_complexity: f64,
    /// Conceptual depth measure
    pub conceptual_depth: f64,
    /// Conceptual density
    pub conceptual_density: f64,
    /// Abstract concept ratio
    pub abstract_concept_ratio: f64,
    /// Complexity distribution
    pub complexity_distribution: ComplexityDistribution,
    /// Conceptual hierarchy depth
    pub hierarchy_depth: usize,
}

/// Distribution of complexity across text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityDistribution {
    /// Complexity variance across segments
    pub variance: f64,
    /// Peak complexity points
    pub peaks: Vec<usize>,
    /// Complexity progression pattern
    pub progression_pattern: String,
    /// Complexity clustering coefficient
    pub clustering_coefficient: f64,
}

/// Consistency analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyAnalysis {
    /// Overall consistency score
    pub overall_consistency: f64,
    /// Terminological consistency
    pub terminological_consistency: f64,
    /// Conceptual consistency
    pub conceptual_consistency: f64,
    /// Semantic field consistency
    pub field_consistency: f64,
    /// Identified inconsistencies
    pub inconsistencies: Vec<InconsistencyPattern>,
    /// Consistency trend over text
    pub consistency_trend: Vec<f64>,
}

/// Pattern of inconsistency found in text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InconsistencyPattern {
    /// Type of inconsistency
    pub inconsistency_type: InconsistencyType,
    /// Severity score (0.0 to 1.0)
    pub severity: f64,
    /// Locations in text where inconsistency occurs
    pub locations: Vec<(usize, usize)>,
    /// Description of the inconsistency
    pub description: String,
    /// Suggested resolution
    pub suggested_resolution: Option<String>,
    /// Confidence in detection
    pub confidence: f64,
    /// Related terms or concepts
    pub related_terms: Vec<String>,
    /// Context snippets showing the inconsistency
    pub context_examples: Vec<String>,
    /// Impact on overall coherence
    pub coherence_impact: f64,
}

/// Topic coherence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicCoherence {
    /// Overall topic coherence score
    pub overall_coherence: f64,
    /// Topic consistency across segments
    pub topic_consistency: f64,
    /// Topic transition smoothness
    pub transition_smoothness: f64,
    /// Main topics identified
    pub main_topics: Vec<Topic>,
    /// Topic distribution
    pub topic_distribution: HashMap<String, f64>,
    /// Topic evolution pattern
    pub evolution_pattern: Vec<String>,
    /// Inter-topic relationships
    pub topic_relationships: HashMap<(String, String), f64>,
}

/// Topic representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topic {
    /// Topic identifier
    pub id: String,
    /// Topic label or description
    pub label: String,
    /// Topic keywords
    pub keywords: Vec<String>,
    /// Topic strength/relevance
    pub strength: f64,
    /// Coverage in text (as proportion)
    pub coverage: f64,
    /// Topic coherence score
    pub coherence_score: f64,
}

/// Semantic role analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRoleAnalysis {
    /// Agent-patient relationship clarity
    pub role_clarity: f64,
    /// Thematic role consistency
    pub role_consistency: f64,
    /// Role assignment ambiguity
    pub role_ambiguity: f64,
    /// Complex role structures found
    pub complex_structures: Vec<ComplexRoleStructure>,
    /// Role distribution analysis
    pub role_distribution: HashMap<String, f64>,
}

/// Complex semantic role structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexRoleStructure {
    pub structure_type: String,
    pub complexity_score: f64,
    pub participants: Vec<String>,
    pub relationships: HashMap<String, String>,
    pub clarity_score: f64,
    pub examples: Vec<String>,
}

/// Figurative language analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FigurativeLanguageAnalysis {
    /// Overall figurative language density
    pub figurative_density: f64,
    /// Metaphor usage analysis
    pub metaphor_analysis: MetaphorAnalysis,
    /// Metonymy instances
    pub metonymy_count: usize,
    /// Synecdoche instances
    pub synecdoche_count: usize,
    /// Irony/sarcasm detection
    pub irony_score: f64,
    /// Figurative coherence
    pub figurative_coherence: f64,
}

/// Metaphor analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaphorAnalysis {
    /// Number of metaphors identified
    pub metaphor_count: usize,
    /// Metaphor consistency score
    pub consistency: f64,
    /// Conceptual metaphor coherence
    pub conceptual_coherence: f64,
    /// Individual metaphor instances
    pub metaphors: Vec<MetaphorInstance>,
    /// Dominant metaphorical themes
    pub dominant_themes: Vec<String>,
}

/// Individual metaphor instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaphorInstance {
    /// Source domain
    pub source_domain: String,
    /// Target domain
    pub target_domain: String,
    /// Metaphor text
    pub text: String,
    /// Position in text
    pub position: (usize, usize),
    /// Novelty score
    pub novelty: f64,
    /// Coherence with context
    pub coherence: f64,
    /// Conceptual mapping strength
    pub mapping_strength: f64,
}

/// Discourse analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseAnalysis {
    /// Discourse coherence relations
    pub coherence_relations: Vec<CoherenceRelation>,
    /// Discourse marker analysis
    pub marker_analysis: DiscourseMarkerAnalysis,
    /// Rhetorical structure coherence
    pub rhetorical_coherence: f64,
    /// Information flow coherence
    pub information_flow: f64,
    /// Global discourse structure
    pub discourse_structure: Vec<String>,
}

/// Coherence relation between text segments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceRelation {
    /// Relation type
    pub relation_type: CoherenceRelationType,
    /// Source segment
    pub source_segment: (usize, usize),
    /// Target segment
    pub target_segment: (usize, usize),
    /// Relation strength
    pub strength: f64,
    /// Confidence in relation
    pub confidence: f64,
    /// Explicit markers (if any)
    pub explicit_markers: Vec<String>,
    /// Relation description
    pub description: String,
}

/// Discourse marker analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseMarkerAnalysis {
    /// Marker usage frequency
    pub marker_frequency: HashMap<String, usize>,
    /// Marker appropriateness score
    pub appropriateness: f64,
    /// Marker diversity
    pub diversity: f64,
    /// Missing markers (where needed)
    pub missing_markers: Vec<String>,
    /// Overused markers
    pub overused_markers: Vec<String>,
    /// Marker placement accuracy
    pub placement_accuracy: f64,
}

/// Information structure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationStructure {
    /// Given-new information balance
    pub given_new_balance: f64,
    /// Topic-comment structure clarity
    pub topic_comment_clarity: f64,
    /// Focus structure analysis
    pub focus_structure: FocusStructure,
    /// Information density distribution
    pub density_distribution: Vec<f64>,
    /// Information progression coherence
    pub progression_coherence: f64,
}

/// Focus structure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusStructure {
    /// Overall focus clarity
    pub focus_clarity: f64,
    /// Focus points identified
    pub focus_points: Vec<FocusPoint>,
    /// Focus transition coherence
    pub transition_coherence: f64,
    /// Focus distribution evenness
    pub distribution_evenness: f64,
}

/// Individual focus point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FocusPoint {
    /// Focus type
    pub focus_type: FocusType,
    /// Position in text
    pub position: (usize, usize),
    /// Focus strength
    pub strength: f64,
    /// Focused element
    pub focused_element: String,
    /// Context appropriateness
    pub appropriateness: f64,
}

/// Analysis metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// Analysis timestamp
    pub timestamp: String,
    /// Analysis duration in seconds
    pub duration_seconds: f64,
    /// Configuration hash used
    pub config_hash: String,
    /// Text statistics
    pub text_stats: TextStatistics,
    /// Processing statistics
    pub processing_stats: ProcessingStatistics,
    /// Analysis version
    pub version: String,
}

/// Basic text statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextStatistics {
    /// Total character count
    pub character_count: usize,
    /// Word count
    pub word_count: usize,
    /// Sentence count
    pub sentence_count: usize,
    /// Average sentence length
    pub avg_sentence_length: f64,
    /// Vocabulary size (unique words)
    pub vocabulary_size: usize,
    /// Type-token ratio
    pub type_token_ratio: f64,
}

/// Processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStatistics {
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Processing stages completed
    pub stages_completed: Vec<String>,
    /// Error count
    pub error_count: usize,
    /// Warning count
    pub warning_count: usize,
}

/// Semantic relation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRelation {
    /// Source word or concept
    pub source: String,
    /// Target word or concept
    pub target: String,
    /// Relation type
    pub relation_type: String,
    /// Relation strength
    pub strength: f64,
    /// Confidence in relation
    pub confidence: f64,
    /// Context where relation occurs
    pub context: String,
}

/// Semantic network node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticNode {
    /// Node identifier (word or concept)
    pub id: String,
    /// Node label for display
    pub label: String,
    /// Node importance/centrality
    pub centrality: f64,
    /// Associated semantic field
    pub semantic_field: Option<String>,
    /// Node type (word, concept, etc.)
    pub node_type: String,
    /// Frequency in text
    pub frequency: usize,
}

/// Semantic network edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEdge {
    /// Source node ID
    pub source: String,
    /// Target node ID
    pub target: String,
    /// Edge weight/strength
    pub weight: f64,
    /// Edge type
    pub edge_type: String,
    /// Confidence in edge
    pub confidence: f64,
}

/// Complete semantic network representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticNetwork {
    /// Network nodes
    pub nodes: Vec<SemanticNode>,
    /// Network edges
    pub edges: Vec<SemanticEdge>,
    /// Network density
    pub density: f64,
    /// Network diameter
    pub diameter: usize,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Connected components count
    pub connected_components: usize,
}

impl SemanticFluencyResult {
    /// Create a new empty result with basic structure
    pub fn new() -> Self {
        Self {
            overall_score: 0.0,
            semantic_coherence: 0.0,
            meaning_preservation: 0.0,
            conceptual_clarity: 0.0,
            semantic_appropriateness: 0.0,
            context_sensitivity: 0.0,
            semantic_density: 0.0,
            ambiguity_score: 0.0,
            semantic_relations: HashMap::new(),
            advanced_metrics: None,
            insights: Vec::new(),
            component_scores: HashMap::new(),
            metadata: AnalysisMetadata::default(),
        }
    }

    /// Calculate overall score from component scores
    pub fn calculate_overall_score(&mut self, weights: &HashMap<String, f64>) {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (component, score) in &self.component_scores {
            if let Some(&weight) = weights.get(component) {
                weighted_sum += score * weight;
                total_weight += weight;
            }
        }

        self.overall_score = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        };
    }

    /// Add an insight to the results
    pub fn add_insight(&mut self, insight: String) {
        self.insights.push(insight);
    }

    /// Get the top N insights by relevance
    pub fn get_top_insights(&self, n: usize) -> Vec<&String> {
        self.insights.iter().take(n).collect()
    }

    /// Check if analysis includes advanced metrics
    pub fn has_advanced_metrics(&self) -> bool {
        self.advanced_metrics.is_some()
    }

    /// Get semantic relation score by type
    pub fn get_relation_score(&self, relation_type: &str) -> Option<f64> {
        self.semantic_relations.get(relation_type).copied()
    }
}

impl Default for AnalysisMetadata {
    fn default() -> Self {
        Self {
            timestamp: chrono::Utc::now().to_rfc3339(),
            duration_seconds: 0.0,
            config_hash: String::new(),
            text_stats: TextStatistics::default(),
            processing_stats: ProcessingStatistics::default(),
            version: "1.0.0".to_string(),
        }
    }
}

impl Default for TextStatistics {
    fn default() -> Self {
        Self {
            character_count: 0,
            word_count: 0,
            sentence_count: 0,
            avg_sentence_length: 0.0,
            vocabulary_size: 0,
            type_token_ratio: 0.0,
        }
    }
}

impl Default for ProcessingStatistics {
    fn default() -> Self {
        Self {
            memory_usage_mb: 0.0,
            cache_hit_rate: 0.0,
            cache_misses: 0,
            stages_completed: Vec::new(),
            error_count: 0,
            warning_count: 0,
        }
    }
}

impl TextStatistics {
    /// Calculate text statistics from sentences
    pub fn from_sentences(sentences: &[String]) -> Self {
        let character_count = sentences.iter().map(|s| s.len()).sum();
        let sentence_count = sentences.len();

        let all_words: Vec<String> = sentences
            .iter()
            .flat_map(|s| s.split_whitespace())
            .map(|w| w.to_lowercase())
            .collect();

        let word_count = all_words.len();
        let vocabulary: std::collections::HashSet<&String> = all_words.iter().collect();
        let vocabulary_size = vocabulary.len();

        let avg_sentence_length = if sentence_count > 0 {
            word_count as f64 / sentence_count as f64
        } else {
            0.0
        };

        let type_token_ratio = if word_count > 0 {
            vocabulary_size as f64 / word_count as f64
        } else {
            0.0
        };

        Self {
            character_count,
            word_count,
            sentence_count,
            avg_sentence_length,
            vocabulary_size,
            type_token_ratio,
        }
    }
}

// Add chrono dependency for timestamps
extern crate chrono;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_creation() {
        let result = SemanticFluencyResult::new();
        assert_eq!(result.overall_score, 0.0);
        assert!(result.insights.is_empty());
        assert!(!result.has_advanced_metrics());
    }

    #[test]
    fn test_overall_score_calculation() {
        let mut result = SemanticFluencyResult::new();
        result.component_scores.insert("coherence".to_string(), 0.8);
        result.component_scores.insert("meaning".to_string(), 0.7);
        result.component_scores.insert("context".to_string(), 0.9);

        let mut weights = HashMap::new();
        weights.insert("coherence".to_string(), 0.4);
        weights.insert("meaning".to_string(), 0.3);
        weights.insert("context".to_string(), 0.3);

        result.calculate_overall_score(&weights);

        let expected = (0.8 * 0.4 + 0.7 * 0.3 + 0.9 * 0.3) / 1.0;
        assert!((result.overall_score - expected).abs() < 0.001);
    }

    #[test]
    fn test_insights_management() {
        let mut result = SemanticFluencyResult::new();

        result.add_insight("Good semantic coherence detected".to_string());
        result.add_insight("Consider improving meaning clarity".to_string());
        result.add_insight("Context sensitivity could be enhanced".to_string());

        assert_eq!(result.insights.len(), 3);

        let top_insights = result.get_top_insights(2);
        assert_eq!(top_insights.len(), 2);
    }

    #[test]
    fn test_text_statistics_calculation() {
        let sentences = vec![
            "This is a test sentence.".to_string(),
            "Another sentence for testing.".to_string(),
            "Final test sentence here.".to_string(),
        ];

        let stats = TextStatistics::from_sentences(&sentences);

        assert_eq!(stats.sentence_count, 3);
        assert!(stats.word_count > 0);
        assert!(stats.vocabulary_size > 0);
        assert!(stats.type_token_ratio > 0.0 && stats.type_token_ratio <= 1.0);
        assert!(stats.avg_sentence_length > 0.0);
    }

    #[test]
    fn test_relation_score_retrieval() {
        let mut result = SemanticFluencyResult::new();
        result
            .semantic_relations
            .insert("synonymy".to_string(), 0.8);
        result
            .semantic_relations
            .insert("antonymy".to_string(), 0.3);

        assert_eq!(result.get_relation_score("synonymy"), Some(0.8));
        assert_eq!(result.get_relation_score("antonymy"), Some(0.3));
        assert_eq!(result.get_relation_score("hyponymy"), None);
    }

    #[test]
    fn test_serialization() {
        let result = SemanticFluencyResult::new();
        let serialized = serde_json::to_string(&result);
        assert!(serialized.is_ok());

        let deserialized: Result<SemanticFluencyResult, _> =
            serde_json::from_str(&serialized.unwrap());
        assert!(deserialized.is_ok());
    }

    #[test]
    fn test_advanced_metrics_detection() {
        let mut result = SemanticFluencyResult::new();
        assert!(!result.has_advanced_metrics());

        result.advanced_metrics = Some(AdvancedSemanticMetrics {
            field_coverage: SemanticFieldCoverage {
                overall_coverage: 0.8,
                semantic_fields: HashMap::new(),
                distribution_evenness: 0.7,
                transition_coherence: 0.9,
                dominant_field: None,
                diversity_index: 0.6,
            },
            conceptual_complexity: ConceptualComplexity {
                overall_complexity: 0.5,
                conceptual_depth: 0.6,
                conceptual_density: 0.4,
                abstract_concept_ratio: 0.3,
                complexity_distribution: ComplexityDistribution {
                    variance: 0.2,
                    peaks: vec![1, 3, 5],
                    progression_pattern: "increasing".to_string(),
                    clustering_coefficient: 0.7,
                },
                hierarchy_depth: 3,
            },
            consistency_analysis: ConsistencyAnalysis {
                overall_consistency: 0.8,
                terminological_consistency: 0.9,
                conceptual_consistency: 0.7,
                field_consistency: 0.8,
                inconsistencies: Vec::new(),
                consistency_trend: vec![0.8, 0.7, 0.9],
            },
            topic_coherence: None,
            semantic_roles: None,
            figurative_language: None,
            discourse_analysis: None,
            information_structure: None,
        });

        assert!(result.has_advanced_metrics());
    }
}
