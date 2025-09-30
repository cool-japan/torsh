//! Topic coherence analysis results
//!
//! This module provides comprehensive result structures for topic coherence analysis.
//! It organizes all result types in a clean hierarchy that reflects the modular
//! analysis architecture.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Topic transition types for detailed transition analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TopicTransitionType {
    /// Smooth transition with clear semantic connections
    Smooth,
    /// Gradual shift with some thematic overlap
    Gradual,
    /// Abrupt topic change without clear connection
    Abrupt,
    /// Topic return (circular reference to previous topic)
    Return,
    /// Topic branching into multiple related subtopics
    Branching,
    /// Multiple topics merging into unified theme
    Merging,
    /// Topic elaboration with deeper detail
    Elaboration,
    /// Topic digression away from main theme
    Digression,
}

/// Thematic progression patterns in text structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ThematicProgressionPattern {
    /// Linear progression through sequential topics
    Linear,
    /// Spiral development returning to themes with deeper analysis
    Spiral,
    /// Hierarchical organization with main topics and subtopics
    Hierarchical,
    /// Circular structure returning to initial themes
    Circular,
    /// Tree-like branching from central themes
    TreeBranching,
    /// Network-like interconnections between topics
    Network,
    /// Fragmented structure with disconnected topics
    Fragmented,
}

/// Development stage in topic evolution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevelopmentStage {
    /// Stage identifier name
    pub stage_name: String,
    /// Text span covered by this stage
    pub span: (usize, usize),
    /// Key characteristics of this development stage
    pub characteristics: Vec<String>,
    /// Intensity of topic presence in this stage
    pub intensity: f64,
    /// Stage duration relative to total text
    pub duration_ratio: f64,
}

/// Topic evolution analysis over the course of text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicEvolution {
    /// Overall evolution pattern classification
    pub evolution_pattern: String,
    /// Intensity trajectory showing topic strength over text positions
    pub intensity_trajectory: Vec<f64>,
    /// Distinct development stages in topic evolution
    pub development_stages: Vec<DevelopmentStage>,
    /// Position where topic reaches peak prominence
    pub peak_position: usize,
    /// Consistency score of evolution (smooth vs erratic)
    pub consistency_score: f64,
    /// Topic lifespan as ratio of total text
    pub lifespan_ratio: f64,
}

/// Conceptual cluster within a topic's semantic profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptualCluster {
    /// Cluster identifier name
    pub cluster_name: String,
    /// Words grouped in this conceptual cluster
    pub words: Vec<String>,
    /// Internal coherence of the cluster
    pub coherence: f64,
    /// Centrality score within the topic
    pub centrality: f64,
    /// Semantic weight of cluster within topic
    pub semantic_weight: f64,
}

/// Semantic profile describing the conceptual characteristics of a topic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticProfile {
    /// Identified semantic fields for the topic
    pub semantic_fields: Vec<String>,
    /// Conceptual clusters within the topic
    pub conceptual_clusters: Vec<ConceptualCluster>,
    /// Overall semantic coherence score
    pub semantic_coherence: f64,
    /// Level of abstraction (concrete vs abstract concepts)
    pub abstractness_level: f64,
    /// Semantic diversity (breadth of concepts)
    pub semantic_diversity: f64,
    /// Conceptual density (concept concentration)
    pub conceptual_density: f64,
}

/// Quality metrics for individual topics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicQualityMetrics {
    /// Internal coherence (how well keywords relate)
    pub internal_coherence: f64,
    /// External distinctiveness (how unique vs other topics)
    pub distinctiveness: f64,
    /// Topic focus (concentration of keywords)
    pub focus: f64,
    /// Topic coverage (proportion of text covered)
    pub coverage: f64,
    /// Topic stability (consistency across text)
    pub stability: f64,
    /// Interpretability score (how clear the topic is)
    pub interpretability: f64,
    /// Topic complexity (number of distinct concepts)
    pub complexity: f64,
}

/// Relationship between topics with detailed characterization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicRelationship {
    /// ID of the related topic
    pub related_topic_id: String,
    /// Type of relationship (semantic, temporal, hierarchical)
    pub relationship_type: String,
    /// Strength of the relationship
    pub strength: f64,
    /// Confidence in the relationship detection
    pub confidence: f64,
    /// Direction of relationship (if applicable)
    pub directionality: Option<String>,
}

/// Comprehensive topic with detailed analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topic {
    /// Unique topic identifier
    pub topic_id: String,
    /// Primary keywords defining the topic
    pub keywords: Vec<String>,
    /// Overall coherence score for the topic
    pub coherence_score: f64,
    /// Text span covered by the topic (start, end positions)
    pub span: (usize, usize),
    /// Prominence score (importance within text)
    pub prominence: f64,
    /// Topic density (keyword concentration)
    pub density: f64,
    /// Evolution analysis of topic over text
    pub evolution: TopicEvolution,
    /// Semantic characteristics profile
    pub semantic_profile: SemanticProfile,
    /// Quality assessment metrics
    pub quality_metrics: TopicQualityMetrics,
    /// Hierarchical level (0 = main topic, higher = subtopic)
    pub hierarchical_level: usize,
    /// Relationships to other topics
    pub relationships: Vec<TopicRelationship>,
}

/// Topic transition with detailed analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicTransition {
    /// Source topic identifier
    pub from_topic: String,
    /// Target topic identifier
    pub to_topic: String,
    /// Position in text where transition occurs
    pub position: usize,
    /// Quality score of the transition
    pub transition_quality: f64,
    /// Classification of transition type
    pub transition_type: TopicTransitionType,
    /// Smoothness of transition (0 = abrupt, 1 = seamless)
    pub smoothness: f64,
    /// Bridging words or phrases facilitating transition
    pub bridging_elements: Vec<String>,
}

/// Detailed metrics for comprehensive topic analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedTopicMetrics {
    /// Average topic coherence across all topics
    pub average_topic_coherence: f64,
    /// Topic coherence variance (consistency measure)
    pub topic_coherence_variance: f64,
    /// Average transition quality score
    pub average_transition_quality: f64,
    /// Number of high-quality transitions
    pub high_quality_transitions: usize,
    /// Topic coverage ratio (text covered by topics)
    pub topic_coverage_ratio: f64,
    /// Average topic lifespan
    pub average_topic_lifespan: f64,
    /// Topic overlap ratio
    pub topic_overlap_ratio: f64,
    /// Semantic diversity score
    pub semantic_diversity: f64,
}

/// Topic relationship network analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicRelationshipAnalysis {
    /// Network density (proportion of possible connections)
    pub network_density: f64,
    /// Central topics (highest connectivity)
    pub central_topics: Vec<String>,
    /// Topic clusters (groups of related topics)
    pub topic_clusters: Vec<Vec<String>>,
    /// Average relationship strength
    pub average_relationship_strength: f64,
    /// Number of relationship types identified
    pub relationship_types_count: usize,
}

/// Advanced topic analysis with sophisticated features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedTopicAnalysis {
    /// Hierarchical topic structure
    pub hierarchical_structure: HashMap<String, Vec<String>>,
    /// Dynamic topic evolution patterns
    pub dynamic_patterns: Vec<String>,
    /// Topic network characteristics
    pub network_characteristics: HashMap<String, f64>,
    /// Thematic progression pattern
    pub progression_pattern: ThematicProgressionPattern,
    /// Temporal dynamics of topic development
    pub temporal_dynamics: Vec<f64>,
    /// Cross-topic influence scores
    pub cross_topic_influences: HashMap<String, HashMap<String, f64>>,
}

/// Comprehensive topic coherence analysis results
///
/// This is the main result structure that contains all analysis outputs
/// from the modular topic coherence analysis system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicCoherenceResult {
    /// Overall topic consistency score (0.0 - 1.0)
    pub topic_consistency: f64,
    /// Topic shift coherence score (smoothness of transitions)
    pub topic_shift_coherence: f64,
    /// Topic development quality score
    pub topic_development: f64,
    /// Thematic unity score (overall cohesion)
    pub thematic_unity: f64,
    /// All identified topics with comprehensive analysis
    pub topics: Vec<Topic>,
    /// All topic transitions with detailed characterization
    pub topic_transitions: Vec<TopicTransition>,
    /// Distribution of topics across the text
    pub topic_distribution: HashMap<String, f64>,
    /// Coherence score for each individual topic
    pub coherence_per_topic: HashMap<String, f64>,
    /// Detailed metrics and statistical analysis
    pub detailed_metrics: DetailedTopicMetrics,
    /// Topic relationship network analysis
    pub topic_relationships: TopicRelationshipAnalysis,
    /// Advanced analysis features (optional)
    pub advanced_analysis: Option<AdvancedTopicAnalysis>,
    /// Analysis metadata
    pub analysis_metadata: AnalysisMetadata,
}

/// Metadata about the analysis process and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    /// Configuration used for analysis
    pub config_summary: String,
    /// Text length analyzed
    pub text_length: usize,
    /// Number of sentences processed
    pub sentences_processed: usize,
    /// Analysis duration in milliseconds
    pub analysis_duration_ms: u64,
    /// Timestamp of analysis
    pub timestamp: String,
    /// Analysis version
    pub analysis_version: String,
}

impl TopicCoherenceResult {
    /// Create a new result with default values
    pub fn new() -> Self {
        Self {
            topic_consistency: 0.0,
            topic_shift_coherence: 0.0,
            topic_development: 0.0,
            thematic_unity: 0.0,
            topics: Vec::new(),
            topic_transitions: Vec::new(),
            topic_distribution: HashMap::new(),
            coherence_per_topic: HashMap::new(),
            detailed_metrics: DetailedTopicMetrics {
                average_topic_coherence: 0.0,
                topic_coherence_variance: 0.0,
                average_transition_quality: 0.0,
                high_quality_transitions: 0,
                topic_coverage_ratio: 0.0,
                average_topic_lifespan: 0.0,
                topic_overlap_ratio: 0.0,
                semantic_diversity: 0.0,
            },
            topic_relationships: TopicRelationshipAnalysis {
                network_density: 0.0,
                central_topics: Vec::new(),
                topic_clusters: Vec::new(),
                average_relationship_strength: 0.0,
                relationship_types_count: 0,
            },
            advanced_analysis: None,
            analysis_metadata: AnalysisMetadata {
                config_summary: String::new(),
                text_length: 0,
                sentences_processed: 0,
                analysis_duration_ms: 0,
                timestamp: String::new(),
                analysis_version: "2.0.0".to_string(),
            },
        }
    }

    /// Calculate overall quality score combining all metrics
    pub fn overall_quality_score(&self) -> f64 {
        (self.topic_consistency
            + self.topic_shift_coherence
            + self.topic_development
            + self.thematic_unity)
            / 4.0
    }

    /// Get summary statistics for topics
    pub fn topic_summary(&self) -> HashMap<String, f64> {
        let mut summary = HashMap::new();

        summary.insert("total_topics".to_string(), self.topics.len() as f64);
        summary.insert(
            "total_transitions".to_string(),
            self.topic_transitions.len() as f64,
        );
        summary.insert(
            "average_coherence".to_string(),
            self.coherence_per_topic.values().sum::<f64>() / self.coherence_per_topic.len() as f64,
        );

        if !self.topics.is_empty() {
            let avg_prominence =
                self.topics.iter().map(|t| t.prominence).sum::<f64>() / self.topics.len() as f64;
            summary.insert("average_prominence".to_string(), avg_prominence);
        }

        summary
    }
}

impl Default for TopicCoherenceResult {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_creation() {
        let result = TopicCoherenceResult::new();
        assert_eq!(result.topics.len(), 0);
        assert_eq!(result.topic_transitions.len(), 0);
        assert_eq!(result.overall_quality_score(), 0.0);
    }

    #[test]
    fn test_overall_quality_calculation() {
        let mut result = TopicCoherenceResult::new();
        result.topic_consistency = 0.8;
        result.topic_shift_coherence = 0.7;
        result.topic_development = 0.6;
        result.thematic_unity = 0.9;

        assert_eq!(result.overall_quality_score(), 0.75);
    }

    #[test]
    fn test_result_serialization() {
        let result = TopicCoherenceResult::new();
        let serialized = serde_json::to_string(&result).expect("Should serialize");
        let deserialized: TopicCoherenceResult =
            serde_json::from_str(&serialized).expect("Should deserialize");

        assert_eq!(result.topics.len(), deserialized.topics.len());
        assert_eq!(
            result.overall_quality_score(),
            deserialized.overall_quality_score()
        );
    }
}
