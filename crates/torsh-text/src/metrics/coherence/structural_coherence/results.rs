//! Result structures for structural coherence analysis
//!
//! This module defines comprehensive result structures that capture all aspects
//! of structural coherence analysis, including hierarchical structure, discourse
//! patterns, structural markers, and detailed metrics.

use crate::metrics::coherence::structural_coherence::config::{
    DiscoursePatternType, DocumentStructureType, HierarchicalLevel, StructuralMarkerType,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive structural coherence analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralCoherenceResult {
    /// Overall paragraph coherence score
    pub paragraph_coherence: f64,
    /// Section-level coherence score
    pub section_coherence: f64,
    /// Organizational coherence score
    pub organizational_coherence: f64,
    /// Hierarchical structure coherence score
    pub hierarchical_coherence: f64,
    /// Paragraph transition quality scores
    pub paragraph_transitions: Vec<f64>,
    /// Structural markers found in text
    pub structural_markers: Vec<String>,
    /// Coherence patterns analysis
    pub coherence_patterns: HashMap<String, f64>,
    /// Overall structural consistency score
    pub structural_consistency: f64,
    /// Detailed structural metrics
    pub detailed_metrics: DetailedStructuralMetrics,
    /// Document structure analysis
    pub document_structure: DocumentStructureAnalysis,
    /// Advanced structural analysis
    pub advanced_analysis: Option<AdvancedStructuralAnalysis>,
}

/// Detailed structural metrics and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedStructuralMetrics {
    /// Total number of paragraphs
    pub total_paragraphs: usize,
    /// Average paragraph length (words)
    pub average_paragraph_length: f64,
    /// Paragraph length distribution
    pub paragraph_length_distribution: Vec<usize>,
    /// Section boundary detection results
    pub section_boundaries: Vec<SectionBoundary>,
    /// Hierarchical structure analysis
    pub hierarchical_structure: HierarchicalStructureAnalysis,
    /// Discourse pattern analysis
    pub discourse_patterns: DiscoursePatternAnalysis,
    /// Structural marker analysis
    pub marker_analysis: StructuralMarkerAnalysis,
    /// Global coherence measures
    pub global_coherence: GlobalCoherenceMetrics,
    /// Document completeness analysis
    pub document_completeness: DocumentCompletenessMetrics,
    /// Structural complexity measures
    pub complexity_measures: StructuralComplexityMetrics,
}

/// Document structure analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentStructureAnalysis {
    /// Detected document structure type
    pub detected_structure_type: DocumentStructureType,
    /// Structure detection confidence
    pub detection_confidence: f64,
    /// Genre template compliance score
    pub genre_compliance: f64,
    /// Missing structural elements
    pub missing_elements: Vec<String>,
    /// Structural violations
    pub violations: Vec<StructuralViolation>,
    /// Overall structure quality score
    pub structure_quality: f64,
}

/// Structural violation details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralViolation {
    /// Violation type
    pub violation_type: String,
    /// Position in text
    pub position: usize,
    /// Severity (0.0-1.0)
    pub severity: f64,
    /// Description of the violation
    pub description: String,
    /// Suggested correction
    pub suggested_correction: Option<String>,
}

/// Section boundary identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionBoundary {
    /// Position in text (paragraph index)
    pub position: usize,
    /// Boundary strength (confidence)
    pub strength: f64,
    /// Boundary type
    pub boundary_type: String,
    /// Section title (if detected)
    pub section_title: Option<String>,
    /// Hierarchical level
    pub hierarchical_level: HierarchicalLevel,
}

/// Hierarchical structure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalStructureAnalysis {
    /// Detected hierarchical levels
    pub detected_levels: Vec<HierarchicalLevel>,
    /// Level transitions analysis
    pub level_transitions: Vec<LevelTransition>,
    /// Hierarchical balance score
    pub balance_score: f64,
    /// Depth distribution
    pub depth_distribution: HashMap<String, usize>,
    /// Structural tree representation
    pub structural_tree: Option<StructuralTree>,
    /// Hierarchical consistency
    pub consistency_score: f64,
}

/// Level transition in hierarchical structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelTransition {
    /// From level
    pub from_level: HierarchicalLevel,
    /// To level
    pub to_level: HierarchicalLevel,
    /// Transition position
    pub position: usize,
    /// Transition quality
    pub quality: f64,
    /// Transition appropriateness
    pub appropriateness: f64,
}

/// Structural tree representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralTree {
    /// Root node
    pub root: StructuralNode,
    /// Tree depth
    pub depth: usize,
    /// Node count
    pub node_count: usize,
    /// Tree balance score
    pub balance_score: f64,
}

/// Structural tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralNode {
    /// Node identifier
    pub node_id: usize,
    /// Hierarchical level
    pub level: HierarchicalLevel,
    /// Content (text span)
    pub content_span: (usize, usize),
    /// Node title
    pub title: Option<String>,
    /// Child nodes
    pub children: Vec<StructuralNode>,
    /// Node-specific metrics
    pub metrics: NodeMetrics,
}

/// Node-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetrics {
    /// Content length
    pub content_length: usize,
    /// Internal coherence score
    pub internal_coherence: f64,
    /// Connection strength to parent
    pub parent_connection: f64,
    /// Connection strength to children
    pub children_connections: Vec<f64>,
}

/// Discourse pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoursePatternAnalysis {
    /// Detected patterns
    pub detected_patterns: Vec<DetectedPattern>,
    /// Pattern distribution across document
    pub pattern_distribution: HashMap<String, f64>,
    /// Pattern coherence scores
    pub pattern_coherence_scores: HashMap<String, f64>,
    /// Pattern transitions
    pub pattern_transitions: Vec<PatternTransition>,
    /// Overall pattern consistency
    pub pattern_consistency: f64,
}

/// Detected discourse pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    /// Pattern type
    pub pattern_type: DiscoursePatternType,
    /// Pattern strength
    pub strength: f64,
    /// Text span where pattern occurs
    pub span: (usize, usize),
    /// Pattern completeness
    pub completeness: f64,
    /// Supporting evidence
    pub evidence: Vec<String>,
    /// Pattern quality score
    pub quality_score: f64,
}

/// Pattern transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternTransition {
    /// From pattern
    pub from_pattern: DiscoursePatternType,
    /// To pattern
    pub to_pattern: DiscoursePatternType,
    /// Transition position
    pub position: usize,
    /// Transition smoothness
    pub smoothness: f64,
    /// Transition appropriateness
    pub appropriateness: f64,
}

/// Structural marker analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralMarkerAnalysis {
    /// Detected markers by type
    pub markers_by_type: HashMap<String, Vec<StructuralMarker>>,
    /// Marker density throughout document
    pub marker_density: f64,
    /// Marker distribution analysis
    pub distribution_analysis: MarkerDistributionAnalysis,
    /// Marker effectiveness scores
    pub effectiveness_scores: HashMap<String, f64>,
    /// Missing markers analysis
    pub missing_markers: Vec<String>,
}

/// Individual structural marker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralMarker {
    /// Marker text
    pub text: String,
    /// Marker type
    pub marker_type: StructuralMarkerType,
    /// Position in text
    pub position: usize,
    /// Marker strength/confidence
    pub strength: f64,
    /// Context information
    pub context: String,
    /// Effectiveness score
    pub effectiveness: f64,
}

/// Marker distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkerDistributionAnalysis {
    /// Distribution evenness score
    pub evenness_score: f64,
    /// Markers per section
    pub markers_per_section: Vec<usize>,
    /// Marker clustering analysis
    pub clustering_analysis: HashMap<String, f64>,
    /// Distribution quality
    pub distribution_quality: f64,
}

/// Global coherence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalCoherenceMetrics {
    /// Overall document flow
    pub document_flow: f64,
    /// Information progression quality
    pub information_progression: f64,
    /// Thematic consistency
    pub thematic_consistency: f64,
    /// Structural integration
    pub structural_integration: f64,
    /// Reader guidance effectiveness
    pub reader_guidance: f64,
}

/// Document completeness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentCompletenessMetrics {
    /// Introduction quality
    pub introduction_quality: f64,
    /// Body development quality
    pub body_development_quality: f64,
    /// Conclusion quality
    pub conclusion_quality: f64,
    /// Supporting elements presence
    pub supporting_elements: HashMap<String, bool>,
    /// Overall completeness score
    pub completeness_score: f64,
}

/// Structural complexity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralComplexityMetrics {
    /// Hierarchical complexity
    pub hierarchical_complexity: f64,
    /// Pattern complexity
    pub pattern_complexity: f64,
    /// Marker complexity
    pub marker_complexity: f64,
    /// Overall structural complexity
    pub overall_complexity: f64,
    /// Complexity appropriateness
    pub complexity_appropriateness: f64,
}

/// Advanced structural analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedStructuralAnalysis {
    /// Rhetorical structure analysis
    pub rhetorical_structure: RhetoricalStructureAnalysis,
    /// Genre-specific analysis
    pub genre_analysis: GenreAnalysis,
    /// Reader experience metrics
    pub reader_experience: ReaderExperienceMetrics,
    /// Structural optimization suggestions
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
    /// Cross-reference analysis
    pub cross_reference_analysis: CrossReferenceAnalysis,
}

/// Rhetorical structure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhetoricalStructureAnalysis {
    /// Rhetorical moves detected
    pub rhetorical_moves: Vec<RhetoricalMove>,
    /// Move sequencing quality
    pub move_sequencing: f64,
    /// Rhetorical effectiveness
    pub rhetorical_effectiveness: f64,
    /// Argument structure quality
    pub argument_structure: f64,
}

/// Rhetorical move
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhetoricalMove {
    /// Move type
    pub move_type: String,
    /// Text span
    pub span: (usize, usize),
    /// Move strength
    pub strength: f64,
    /// Supporting linguistic markers
    pub linguistic_markers: Vec<String>,
    /// Move effectiveness
    pub effectiveness: f64,
}

/// Genre-specific analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenreAnalysis {
    /// Genre conventions adherence
    pub conventions_adherence: f64,
    /// Expected vs. actual structure alignment
    pub structure_alignment: f64,
    /// Genre-specific quality metrics
    pub genre_quality_metrics: HashMap<String, f64>,
    /// Deviation analysis
    pub deviation_analysis: Vec<String>,
}

/// Reader experience metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReaderExperienceMetrics {
    /// Navigation ease
    pub navigation_ease: f64,
    /// Information accessibility
    pub information_accessibility: f64,
    /// Cognitive load estimation
    pub cognitive_load: f64,
    /// Reading flow quality
    pub reading_flow: f64,
    /// Comprehension support
    pub comprehension_support: f64,
}

/// Optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    /// Suggestion type
    pub suggestion_type: String,
    /// Target position
    pub position: usize,
    /// Priority (1-10)
    pub priority: usize,
    /// Description
    pub description: String,
    /// Expected improvement
    pub expected_improvement: f64,
    /// Implementation difficulty
    pub implementation_difficulty: String,
}

/// Cross-reference analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossReferenceAnalysis {
    /// Internal references quality
    pub internal_references: f64,
    /// Reference consistency
    pub reference_consistency: f64,
    /// Navigation support
    pub navigation_support: f64,
    /// Reference effectiveness
    pub reference_effectiveness: HashMap<String, f64>,
}

impl Default for StructuralCoherenceResult {
    fn default() -> Self {
        Self {
            paragraph_coherence: 0.0,
            section_coherence: 0.0,
            organizational_coherence: 0.0,
            hierarchical_coherence: 0.0,
            paragraph_transitions: Vec::new(),
            structural_markers: Vec::new(),
            coherence_patterns: HashMap::new(),
            structural_consistency: 0.0,
            detailed_metrics: DetailedStructuralMetrics::default(),
            document_structure: DocumentStructureAnalysis::default(),
            advanced_analysis: None,
        }
    }
}

impl Default for DetailedStructuralMetrics {
    fn default() -> Self {
        Self {
            total_paragraphs: 0,
            average_paragraph_length: 0.0,
            paragraph_length_distribution: Vec::new(),
            section_boundaries: Vec::new(),
            hierarchical_structure: HierarchicalStructureAnalysis::default(),
            discourse_patterns: DiscoursePatternAnalysis::default(),
            marker_analysis: StructuralMarkerAnalysis::default(),
            global_coherence: GlobalCoherenceMetrics::default(),
            document_completeness: DocumentCompletenessMetrics::default(),
            complexity_measures: StructuralComplexityMetrics::default(),
        }
    }
}

impl Default for DocumentStructureAnalysis {
    fn default() -> Self {
        Self {
            detected_structure_type: DocumentStructureType::Unknown,
            detection_confidence: 0.0,
            genre_compliance: 0.0,
            missing_elements: Vec::new(),
            violations: Vec::new(),
            structure_quality: 0.0,
        }
    }
}

// Additional default implementations for other structs
impl Default for HierarchicalStructureAnalysis {
    fn default() -> Self {
        Self {
            detected_levels: Vec::new(),
            level_transitions: Vec::new(),
            balance_score: 0.0,
            depth_distribution: HashMap::new(),
            structural_tree: None,
            consistency_score: 0.0,
        }
    }
}

impl Default for DiscoursePatternAnalysis {
    fn default() -> Self {
        Self {
            detected_patterns: Vec::new(),
            pattern_distribution: HashMap::new(),
            pattern_coherence_scores: HashMap::new(),
            pattern_transitions: Vec::new(),
            pattern_consistency: 0.0,
        }
    }
}

impl Default for StructuralMarkerAnalysis {
    fn default() -> Self {
        Self {
            markers_by_type: HashMap::new(),
            marker_density: 0.0,
            distribution_analysis: MarkerDistributionAnalysis::default(),
            effectiveness_scores: HashMap::new(),
            missing_markers: Vec::new(),
        }
    }
}

impl Default for MarkerDistributionAnalysis {
    fn default() -> Self {
        Self {
            evenness_score: 0.0,
            markers_per_section: Vec::new(),
            clustering_analysis: HashMap::new(),
            distribution_quality: 0.0,
        }
    }
}

impl Default for GlobalCoherenceMetrics {
    fn default() -> Self {
        Self {
            document_flow: 0.0,
            information_progression: 0.0,
            thematic_consistency: 0.0,
            structural_integration: 0.0,
            reader_guidance: 0.0,
        }
    }
}

impl Default for DocumentCompletenessMetrics {
    fn default() -> Self {
        Self {
            introduction_quality: 0.0,
            body_development_quality: 0.0,
            conclusion_quality: 0.0,
            supporting_elements: HashMap::new(),
            completeness_score: 0.0,
        }
    }
}

impl Default for StructuralComplexityMetrics {
    fn default() -> Self {
        Self {
            hierarchical_complexity: 0.0,
            pattern_complexity: 0.0,
            marker_complexity: 0.0,
            overall_complexity: 0.0,
            complexity_appropriateness: 0.0,
        }
    }
}
