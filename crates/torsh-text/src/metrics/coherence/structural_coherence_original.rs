//! Advanced structural coherence analysis for text evaluation
//!
//! This module provides comprehensive structural coherence analysis including paragraph
//! coherence, organizational structure, hierarchical relationships, discourse patterns,
//! and advanced structural metrics. It offers both basic and advanced analysis modes
//! with configurable parameters for different document types and genres.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::RwLock;
use thiserror::Error;

/// Errors that can occur during structural coherence analysis
#[derive(Debug, Error)]
pub enum StructuralCoherenceError {
    #[error("Empty text provided for structural analysis")]
    EmptyText,
    #[error("Invalid structural configuration: {0}")]
    InvalidConfiguration(String),
    #[error("Document parsing error: {0}")]
    DocumentParsingError(String),
    #[error("Structural analysis error: {0}")]
    StructuralAnalysisError(String),
    #[error("Hierarchical analysis failed: {0}")]
    HierarchicalAnalysisError(String),
    #[error("Pattern recognition error: {0}")]
    PatternRecognitionError(String),
    #[error("Organizational analysis error: {0}")]
    OrganizationalAnalysisError(String),
}

/// Document structure types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DocumentStructureType {
    /// Academic paper structure
    Academic,
    /// Technical document structure
    Technical,
    /// Narrative structure
    Narrative,
    /// Argumentative structure
    Argumentative,
    /// Expository structure
    Expository,
    /// Descriptive structure
    Descriptive,
    /// Report structure
    Report,
    /// Blog post structure
    BlogPost,
    /// Unknown structure
    Unknown,
}

/// Hierarchical levels in document structure
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HierarchicalLevel {
    /// Document level
    Document,
    /// Chapter/major section level
    Chapter,
    /// Section level
    Section,
    /// Subsection level
    Subsection,
    /// Paragraph level
    Paragraph,
    /// Sentence level
    Sentence,
    /// Phrase level
    Phrase,
}

/// Discourse pattern types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DiscoursePatternType {
    /// Problem-solution pattern
    ProblemSolution,
    /// Cause-effect pattern
    CauseEffect,
    /// Compare-contrast pattern
    CompareContrast,
    /// Chronological pattern
    Chronological,
    /// Spatial pattern
    Spatial,
    /// Classification pattern
    Classification,
    /// Definition pattern
    Definition,
    /// Process pattern
    Process,
    /// Mixed pattern
    Mixed,
}

/// Structural marker categories
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StructuralMarkerType {
    /// Introduction markers
    Introduction,
    /// Transition markers
    Transition,
    /// Conclusion markers
    Conclusion,
    /// Enumeration markers
    Enumeration,
    /// Example markers
    Example,
    /// Emphasis markers
    Emphasis,
    /// Reference markers
    Reference,
    /// Section markers
    Section,
}

/// Configuration for structural coherence analysis
#[derive(Debug, Clone)]
pub struct StructuralCoherenceConfig {
    /// Weight for structural analysis
    pub structural_weight: f64,
    /// Enable hierarchical analysis
    pub enable_hierarchical_analysis: bool,
    /// Enable organizational pattern analysis
    pub enable_organizational_analysis: bool,
    /// Enable advanced structural analysis
    pub use_advanced_analysis: bool,
    /// Minimum paragraph length for analysis
    pub min_paragraph_length: usize,
    /// Paragraph coherence threshold
    pub paragraph_coherence_threshold: f64,
    /// Enable discourse pattern detection
    pub detect_discourse_patterns: bool,
    /// Enable structural marker analysis
    pub analyze_structural_markers: bool,
    /// Document structure type (if known)
    pub expected_structure_type: Option<DocumentStructureType>,
    /// Maximum hierarchical depth to analyze
    pub max_hierarchical_depth: usize,
    /// Enable section boundary detection
    pub detect_section_boundaries: bool,
    /// Enable global coherence analysis
    pub analyze_global_coherence: bool,
    /// Structural consistency sensitivity
    pub consistency_sensitivity: f64,
    /// Enable rhetorical structure analysis
    pub analyze_rhetorical_structure: bool,
    /// Transition quality threshold
    pub transition_quality_threshold: f64,
}

impl Default for StructuralCoherenceConfig {
    fn default() -> Self {
        Self {
            structural_weight: 0.3,
            enable_hierarchical_analysis: true,
            enable_organizational_analysis: true,
            use_advanced_analysis: true,
            min_paragraph_length: 50,
            paragraph_coherence_threshold: 0.5,
            detect_discourse_patterns: true,
            analyze_structural_markers: true,
            expected_structure_type: None,
            max_hierarchical_depth: 6,
            detect_section_boundaries: true,
            analyze_global_coherence: true,
            consistency_sensitivity: 0.7,
            analyze_rhetorical_structure: true,
            transition_quality_threshold: 0.4,
        }
    }
}

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
    /// Node coherence score
    pub coherence_score: f64,
}

/// Discourse pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoursePatternAnalysis {
    /// Detected patterns
    pub detected_patterns: Vec<DetectedPattern>,
    /// Pattern distribution
    pub pattern_distribution: HashMap<String, f64>,
    /// Pattern coherence scores
    pub pattern_coherence: HashMap<String, f64>,
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
    /// Pattern span (paragraph range)
    pub span: (usize, usize),
    /// Pattern strength
    pub strength: f64,
    /// Pattern completeness
    pub completeness: f64,
    /// Supporting evidence
    pub evidence: Vec<String>,
}

/// Pattern transition between different discourse patterns
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
}

/// Structural marker analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralMarkerAnalysis {
    /// Markers by type
    pub markers_by_type: HashMap<String, Vec<StructuralMarker>>,
    /// Marker density (markers per paragraph)
    pub marker_density: f64,
    /// Marker distribution analysis
    pub distribution_analysis: MarkerDistributionAnalysis,
    /// Marker effectiveness scores
    pub effectiveness_scores: HashMap<String, f64>,
    /// Missing marker analysis
    pub missing_markers: Vec<String>,
}

/// Individual structural marker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralMarker {
    /// Marker text
    pub marker_text: String,
    /// Marker type
    pub marker_type: StructuralMarkerType,
    /// Position in document
    pub position: (usize, usize), // (paragraph, word)
    /// Local effectiveness
    pub effectiveness: f64,
    /// Context appropriateness
    pub appropriateness: f64,
}

/// Marker distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkerDistributionAnalysis {
    /// Distribution evenness
    pub evenness: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Expected vs actual distribution
    pub distribution_alignment: f64,
    /// Marker spacing analysis
    pub spacing_analysis: SpacingAnalysis,
}

/// Spacing analysis for markers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpacingAnalysis {
    /// Average spacing between markers
    pub average_spacing: f64,
    /// Spacing variance
    pub spacing_variance: f64,
    /// Optimal spacing alignment
    pub optimal_alignment: f64,
    /// Spacing regularity
    pub regularity: f64,
}

/// Global coherence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalCoherenceMetrics {
    /// Beginning-end coherence
    pub beginning_end_coherence: f64,
    /// Document-level thematic consistency
    pub thematic_consistency: f64,
    /// Global information flow
    pub information_flow: f64,
    /// Structural completeness
    pub structural_completeness: f64,
    /// Overall document unity
    pub document_unity: f64,
    /// Cross-paragraph coherence
    pub cross_paragraph_coherence: f64,
}

/// Document completeness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentCompletenessMetrics {
    /// Has introduction
    pub has_introduction: bool,
    /// Has development/body
    pub has_development: bool,
    /// Has conclusion
    pub has_conclusion: bool,
    /// Completeness score
    pub completeness_score: f64,
    /// Missing components
    pub missing_components: Vec<String>,
    /// Component quality scores
    pub component_quality: HashMap<String, f64>,
}

/// Structural complexity measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralComplexityMetrics {
    /// Hierarchical complexity
    pub hierarchical_complexity: f64,
    /// Organizational complexity
    pub organizational_complexity: f64,
    /// Pattern complexity
    pub pattern_complexity: f64,
    /// Structural diversity
    pub structural_diversity: f64,
    /// Integration complexity
    pub integration_complexity: f64,
    /// Cognitive processing load
    pub cognitive_load: f64,
}

/// Document structure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentStructureAnalysis {
    /// Detected structure type
    pub detected_structure_type: DocumentStructureType,
    /// Structure confidence
    pub structure_confidence: f64,
    /// Structure features
    pub structure_features: HashMap<String, f64>,
    /// Structure adherence score
    pub adherence_score: f64,
    /// Alternative structure suggestions
    pub alternative_structures: Vec<StructureSuggestion>,
}

/// Structure suggestion for document improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureSuggestion {
    /// Suggested structure type
    pub suggested_type: DocumentStructureType,
    /// Suggestion confidence
    pub confidence: f64,
    /// Improvement potential
    pub improvement_potential: f64,
    /// Specific recommendations
    pub recommendations: Vec<String>,
}

/// Advanced structural analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedStructuralAnalysis {
    /// Rhetorical structure analysis
    pub rhetorical_structure: RhetoricalStructureAnalysis,
    /// Information architecture analysis
    pub information_architecture: InformationArchitectureAnalysis,
    /// Cognitive structure analysis
    pub cognitive_structure: CognitiveStructureAnalysis,
    /// Genre-specific analysis
    pub genre_analysis: GenreSpecificAnalysis,
    /// Quality assessment
    pub quality_assessment: StructuralQualityAssessment,
}

/// Rhetorical structure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhetoricalStructureAnalysis {
    /// Rhetorical moves identified
    pub rhetorical_moves: Vec<RhetoricalMove>,
    /// Move sequences
    pub move_sequences: Vec<MoveSequence>,
    /// Rhetorical effectiveness
    pub effectiveness_score: f64,
    /// Genre appropriateness
    pub genre_appropriateness: f64,
}

/// Individual rhetorical move
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RhetoricalMove {
    /// Move type
    pub move_type: String,
    /// Move span
    pub span: (usize, usize),
    /// Move strength
    pub strength: f64,
    /// Move function
    pub function: String,
    /// Linguistic markers
    pub linguistic_markers: Vec<String>,
}

/// Sequence of rhetorical moves
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveSequence {
    /// Sequence identifier
    pub sequence_id: usize,
    /// Moves in sequence
    pub moves: Vec<String>,
    /// Sequence coherence
    pub coherence: f64,
    /// Sequence appropriateness
    pub appropriateness: f64,
}

/// Information architecture analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationArchitectureAnalysis {
    /// Information hierarchy
    pub information_hierarchy: InformationHierarchy,
    /// Information flow analysis
    pub flow_analysis: InformationFlowAnalysis,
    /// Information density distribution
    pub density_distribution: Vec<f64>,
    /// Architecture effectiveness
    pub effectiveness: f64,
}

/// Information hierarchy structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationHierarchy {
    /// Hierarchy levels
    pub levels: Vec<InformationLevel>,
    /// Level relationships
    pub relationships: Vec<LevelRelationship>,
    /// Hierarchy balance
    pub balance_score: f64,
}

/// Information level in hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationLevel {
    /// Level identifier
    pub level_id: usize,
    /// Level importance
    pub importance: f64,
    /// Information density
    pub density: f64,
    /// Content categories
    pub categories: Vec<String>,
}

/// Relationship between information levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelRelationship {
    /// Parent level
    pub parent_level: usize,
    /// Child level
    pub child_level: usize,
    /// Relationship strength
    pub strength: f64,
    /// Relationship type
    pub relationship_type: String,
}

/// Information flow analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationFlowAnalysis {
    /// Flow direction analysis
    pub flow_direction: String,
    /// Flow continuity score
    pub continuity_score: f64,
    /// Information bottlenecks
    pub bottlenecks: Vec<usize>,
    /// Flow efficiency
    pub efficiency: f64,
}

/// Cognitive structure analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveStructureAnalysis {
    /// Cognitive load distribution
    pub load_distribution: Vec<f64>,
    /// Processing complexity
    pub processing_complexity: f64,
    /// Working memory demands
    pub working_memory_demands: f64,
    /// Comprehension ease
    pub comprehension_ease: f64,
    /// Mental model coherence
    pub mental_model_coherence: f64,
}

/// Genre-specific analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenreSpecificAnalysis {
    /// Genre conventions adherence
    pub conventions_adherence: f64,
    /// Genre-specific features
    pub genre_features: HashMap<String, f64>,
    /// Expected vs actual structure
    pub structure_alignment: f64,
    /// Genre recommendations
    pub recommendations: Vec<String>,
}

/// Structural quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralQualityAssessment {
    /// Overall structural quality
    pub overall_quality: f64,
    /// Quality dimensions
    pub quality_dimensions: HashMap<String, f64>,
    /// Improvement recommendations
    pub improvement_areas: Vec<String>,
    /// Quality benchmarking
    pub benchmark_comparison: f64,
}

/// Advanced structural coherence analyzer
pub struct StructuralCoherenceAnalyzer {
    config: StructuralCoherenceConfig,
    structural_markers: Arc<RwLock<HashMap<String, StructuralMarkerType>>>,
    discourse_patterns: HashMap<String, DiscoursePatternType>,
    rhetorical_moves: HashMap<String, String>,
    genre_templates: HashMap<DocumentStructureType, GenreTemplate>,
    analysis_cache: Arc<RwLock<HashMap<String, StructuralCoherenceResult>>>,
}

/// Genre template for structure analysis
#[derive(Debug, Clone)]
pub struct GenreTemplate {
    /// Expected sections
    pub expected_sections: Vec<String>,
    /// Section order requirements
    pub section_order: Vec<String>,
    /// Required components
    pub required_components: Vec<String>,
    /// Typical patterns
    pub typical_patterns: Vec<DiscoursePatternType>,
}

impl StructuralCoherenceAnalyzer {
    /// Create a new structural coherence analyzer with default configuration
    pub fn new() -> Self {
        Self::with_config(StructuralCoherenceConfig::default())
    }

    /// Create a new structural coherence analyzer with custom configuration
    pub fn with_config(config: StructuralCoherenceConfig) -> Self {
        let structural_markers = Arc::new(RwLock::new(Self::build_structural_markers()));
        let discourse_patterns = Self::build_discourse_patterns();
        let rhetorical_moves = Self::build_rhetorical_moves();
        let genre_templates = Self::build_genre_templates();
        let analysis_cache = Arc::new(RwLock::new(HashMap::new()));

        Self {
            config,
            structural_markers,
            discourse_patterns,
            rhetorical_moves,
            genre_templates,
            analysis_cache,
        }
    }

    /// Analyze structural coherence of the given text
    pub fn analyze_structural_coherence(
        &self,
        text: &str,
    ) -> Result<StructuralCoherenceResult, StructuralCoherenceError> {
        if text.trim().is_empty() {
            return Err(StructuralCoherenceError::EmptyText);
        }

        let paragraphs = self.split_into_paragraphs(text)?;
        let sentences = self.split_into_sentences(text)?;

        // Core structural coherence analysis
        let paragraph_coherence = self.calculate_paragraph_coherence(&paragraphs);
        let section_coherence = self.calculate_section_coherence(&paragraphs);
        let organizational_coherence = self.calculate_organizational_coherence(&paragraphs);
        let hierarchical_coherence = if self.config.enable_hierarchical_analysis {
            self.calculate_hierarchical_coherence(&paragraphs)?
        } else {
            0.0
        };

        // Additional analysis components
        let paragraph_transitions = self.calculate_paragraph_transitions(&paragraphs);
        let structural_markers = self.extract_structural_markers(&paragraphs);
        let coherence_patterns = self.identify_coherence_patterns(&paragraphs);
        let structural_consistency = self.calculate_structural_consistency(&paragraphs);

        // Generate detailed metrics
        let detailed_metrics = self.generate_detailed_metrics(&paragraphs, &sentences)?;

        // Document structure analysis
        let document_structure = self.analyze_document_structure(&paragraphs, &sentences)?;

        // Advanced analysis if enabled
        let advanced_analysis = if self.config.use_advanced_analysis {
            Some(self.perform_advanced_analysis(&paragraphs, &sentences)?)
        } else {
            None
        };

        Ok(StructuralCoherenceResult {
            paragraph_coherence,
            section_coherence,
            organizational_coherence,
            hierarchical_coherence,
            paragraph_transitions,
            structural_markers,
            coherence_patterns,
            structural_consistency,
            detailed_metrics,
            document_structure,
            advanced_analysis,
        })
    }

    /// Split text into paragraphs
    fn split_into_paragraphs(&self, text: &str) -> Result<Vec<String>, StructuralCoherenceError> {
        let paragraphs: Vec<String> = text
            .split("\n\n")
            .filter(|p| !p.trim().is_empty() && p.trim().len() >= self.config.min_paragraph_length)
            .map(|p| p.trim().to_string())
            .collect();

        if paragraphs.is_empty() {
            return Err(StructuralCoherenceError::DocumentParsingError(
                "No valid paragraphs found".to_string(),
            ));
        }

        Ok(paragraphs)
    }

    /// Split text into sentences
    fn split_into_sentences(&self, text: &str) -> Result<Vec<String>, StructuralCoherenceError> {
        let sentences: Vec<String> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_string())
            .collect();

        if sentences.is_empty() {
            return Err(StructuralCoherenceError::DocumentParsingError(
                "No valid sentences found".to_string(),
            ));
        }

        Ok(sentences)
    }

    /// Calculate paragraph coherence
    fn calculate_paragraph_coherence(&self, paragraphs: &[String]) -> f64 {
        if paragraphs.is_empty() {
            return 0.0;
        }

        let coherence_scores: Vec<f64> = paragraphs
            .iter()
            .map(|paragraph| {
                let sentences = self.split_paragraph_into_sentences(paragraph);
                self.calculate_intra_paragraph_coherence(&sentences)
            })
            .collect();

        coherence_scores.iter().sum::<f64>() / coherence_scores.len() as f64
    }

    /// Split paragraph into sentences
    fn split_paragraph_into_sentences(&self, paragraph: &str) -> Vec<String> {
        paragraph
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_string())
            .collect()
    }

    /// Calculate intra-paragraph coherence
    fn calculate_intra_paragraph_coherence(&self, sentences: &[String]) -> f64 {
        if sentences.len() < 2 {
            return 1.0;
        }

        let mut total_coherence = 0.0;
        let mut comparisons = 0;

        for i in 0..sentences.len() - 1 {
            let coherence =
                self.calculate_sentence_pair_coherence(&sentences[i], &sentences[i + 1]);
            total_coherence += coherence;
            comparisons += 1;
        }

        if comparisons > 0 {
            total_coherence / comparisons as f64
        } else {
            0.0
        }
    }

    /// Calculate coherence between two sentences
    fn calculate_sentence_pair_coherence(&self, sent1: &str, sent2: &str) -> f64 {
        let lexical_overlap = self.calculate_lexical_overlap(sent1, sent2);
        let structural_continuity = self.calculate_structural_continuity(sent1, sent2);
        let semantic_continuity = self.calculate_semantic_continuity(sent1, sent2);

        (lexical_overlap * 0.4) + (structural_continuity * 0.3) + (semantic_continuity * 0.3)
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

    /// Calculate structural continuity between sentences
    fn calculate_structural_continuity(&self, sent1: &str, sent2: &str) -> f64 {
        let markers1 = self.count_structural_elements(sent1);
        let markers2 = self.count_structural_elements(sent2);

        let marker_consistency = if markers1 > 0 || markers2 > 0 {
            1.0 - ((markers1 as f64 - markers2 as f64).abs() / (markers1 + markers2) as f64)
        } else {
            1.0
        };

        // Length similarity as a proxy for structural similarity
        let len1 = sent1.split_whitespace().count() as f64;
        let len2 = sent2.split_whitespace().count() as f64;
        let length_similarity = 1.0 - ((len1 - len2).abs() / len1.max(len2));

        (marker_consistency * 0.6) + (length_similarity * 0.4)
    }

    /// Count structural elements in sentence
    fn count_structural_elements(&self, sentence: &str) -> usize {
        if let Ok(markers) = self.structural_markers.read() {
            markers
                .keys()
                .filter(|marker| sentence.to_lowercase().contains(marker.as_str()))
                .count()
        } else {
            0
        }
    }

    /// Calculate semantic continuity between sentences
    fn calculate_semantic_continuity(&self, sent1: &str, sent2: &str) -> f64 {
        // Simplified semantic continuity based on word similarity
        let words1 = self.extract_content_words(sent1);
        let words2 = self.extract_content_words(sent2);

        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let mut similarity_sum = 0.0;
        let mut comparisons = 0;

        for word1 in &words1 {
            for word2 in &words2 {
                let similarity = self.calculate_word_similarity(word1, word2);
                similarity_sum += similarity;
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            similarity_sum / comparisons as f64
        } else {
            0.0
        }
    }

    /// Calculate word similarity
    fn calculate_word_similarity(&self, word1: &str, word2: &str) -> f64 {
        if word1 == word2 {
            return 1.0;
        }

        // Character-based similarity
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

    /// Calculate section coherence
    fn calculate_section_coherence(&self, paragraphs: &[String]) -> f64 {
        if paragraphs.len() < 2 {
            return 1.0;
        }

        let mut total_coherence = 0.0;
        let mut comparisons = 0;

        // Calculate coherence between adjacent paragraphs
        for i in 0..paragraphs.len() - 1 {
            let coherence =
                self.calculate_paragraph_pair_coherence(&paragraphs[i], &paragraphs[i + 1]);
            total_coherence += coherence;
            comparisons += 1;
        }

        if comparisons > 0 {
            total_coherence / comparisons as f64
        } else {
            0.0
        }
    }

    /// Calculate coherence between paragraph pairs
    fn calculate_paragraph_pair_coherence(&self, para1: &str, para2: &str) -> f64 {
        // Get the last sentence of first paragraph and first sentence of second paragraph
        let para1_sentences = self.split_paragraph_into_sentences(para1);
        let para2_sentences = self.split_paragraph_into_sentences(para2);

        if let (Some(last_sentence), Some(first_sentence)) =
            (para1_sentences.last(), para2_sentences.first())
        {
            let lexical_overlap = self.calculate_lexical_overlap(last_sentence, first_sentence);
            let semantic_continuity =
                self.calculate_semantic_continuity(last_sentence, first_sentence);
            let structural_continuity =
                self.calculate_structural_continuity(last_sentence, first_sentence);

            (lexical_overlap * 0.4) + (semantic_continuity * 0.4) + (structural_continuity * 0.2)
        } else {
            0.0
        }
    }

    /// Calculate organizational coherence
    fn calculate_organizational_coherence(&self, paragraphs: &[String]) -> f64 {
        let mut organization_score = 0.0;
        let total_paragraphs = paragraphs.len() as f64;

        if total_paragraphs == 0.0 {
            return 0.0;
        }

        for (idx, paragraph) in paragraphs.iter().enumerate() {
            // Weight based on position in document
            let position_weight = match idx {
                0 => 1.3,                              // Introduction
                i if i == paragraphs.len() - 1 => 1.2, // Conclusion
                _ => 1.0,                              // Body paragraphs
            };

            let structural_markers = self.count_structural_elements(paragraph);
            let paragraph_score = (structural_markers as f64).min(1.0) * position_weight;
            organization_score += paragraph_score;
        }

        organization_score / total_paragraphs
    }

    /// Calculate hierarchical coherence
    fn calculate_hierarchical_coherence(
        &self,
        paragraphs: &[String],
    ) -> Result<f64, StructuralCoherenceError> {
        let mut hierarchy_score = 0.0;
        let mut level_changes = 0;

        for i in 0..paragraphs.len().saturating_sub(1) {
            let current_level = self.infer_structural_level(&paragraphs[i]);
            let next_level = self.infer_structural_level(&paragraphs[i + 1]);

            let level_change = (current_level as i32 - next_level as i32).abs();

            // Smooth transitions get higher scores
            let transition_quality = match level_change {
                0 => 1.0, // Same level
                1 => 0.8, // One level change
                2 => 0.6, // Two level change
                _ => 0.4, // Major level change
            };

            hierarchy_score += transition_quality;
            level_changes += 1;
        }

        if level_changes > 0 {
            Ok(hierarchy_score / level_changes as f64)
        } else {
            Ok(1.0)
        }
    }

    /// Infer structural level of paragraph
    fn infer_structural_level(&self, paragraph: &str) -> usize {
        let structural_indicators = vec![
            ("introduction", 1),
            ("conclusion", 1),
            ("chapter", 2),
            ("section", 3),
            ("subsection", 4),
        ];

        for (indicator, level) in structural_indicators {
            if paragraph.to_lowercase().contains(indicator) {
                return level;
            }
        }

        5 // Default paragraph level
    }

    /// Calculate paragraph transitions
    fn calculate_paragraph_transitions(&self, paragraphs: &[String]) -> Vec<f64> {
        let mut transitions = Vec::new();

        for i in 0..paragraphs.len().saturating_sub(1) {
            let transition_quality =
                self.calculate_paragraph_pair_coherence(&paragraphs[i], &paragraphs[i + 1]);
            transitions.push(transition_quality);
        }

        transitions
    }

    /// Extract structural markers from paragraphs
    fn extract_structural_markers(&self, paragraphs: &[String]) -> Vec<String> {
        let mut found_markers = Vec::new();

        if let Ok(markers) = self.structural_markers.read() {
            for paragraph in paragraphs {
                for marker in markers.keys() {
                    if paragraph.to_lowercase().contains(marker) {
                        found_markers.push(marker.clone());
                    }
                }
            }
        }

        found_markers
    }

    /// Identify coherence patterns
    fn identify_coherence_patterns(&self, paragraphs: &[String]) -> HashMap<String, f64> {
        let mut patterns = HashMap::new();

        let introduction_pattern = if !paragraphs.is_empty() {
            self.calculate_introduction_quality(&paragraphs[0])
        } else {
            0.0
        };
        patterns.insert("introduction".to_string(), introduction_pattern);

        let conclusion_pattern = if !paragraphs.is_empty() {
            self.calculate_conclusion_quality(paragraphs.last().unwrap())
        } else {
            0.0
        };
        patterns.insert("conclusion".to_string(), conclusion_pattern);

        let development_pattern = if paragraphs.len() > 2 {
            self.calculate_development_quality(&paragraphs[1..paragraphs.len() - 1])
        } else {
            0.0
        };
        patterns.insert("development".to_string(), development_pattern);

        patterns
    }

    /// Calculate introduction quality
    fn calculate_introduction_quality(&self, paragraph: &str) -> f64 {
        let introduction_indicators = vec![
            "introduce",
            "overview",
            "begin",
            "start",
            "first",
            "initially",
            "this paper",
            "this study",
            "this research",
            "purpose",
            "objective",
        ];

        let indicator_count = introduction_indicators
            .iter()
            .filter(|indicator| paragraph.to_lowercase().contains(indicator))
            .count();

        (indicator_count as f64 / introduction_indicators.len() as f64).min(1.0)
    }

    /// Calculate conclusion quality
    fn calculate_conclusion_quality(&self, paragraph: &str) -> f64 {
        let conclusion_indicators = vec![
            "conclude",
            "summary",
            "finally",
            "in conclusion",
            "therefore",
            "thus",
            "to summarize",
            "in summary",
            "overall",
            "lastly",
            "end",
        ];

        let indicator_count = conclusion_indicators
            .iter()
            .filter(|indicator| paragraph.to_lowercase().contains(indicator))
            .count();

        (indicator_count as f64 / conclusion_indicators.len() as f64).min(1.0)
    }

    /// Calculate development quality
    fn calculate_development_quality(&self, paragraphs: &[String]) -> f64 {
        if paragraphs.is_empty() {
            return 0.0;
        }

        let coherence_scores: Vec<f64> = paragraphs
            .iter()
            .map(|paragraph| {
                let sentences = self.split_paragraph_into_sentences(paragraph);
                self.calculate_intra_paragraph_coherence(&sentences)
            })
            .collect();

        coherence_scores.iter().sum::<f64>() / coherence_scores.len() as f64
    }

    /// Calculate structural consistency
    fn calculate_structural_consistency(&self, paragraphs: &[String]) -> f64 {
        if paragraphs.len() < 2 {
            return 1.0;
        }

        // Calculate paragraph length consistency
        let paragraph_lengths: Vec<usize> = paragraphs
            .iter()
            .map(|p| p.split_whitespace().count())
            .collect();

        let mean_length =
            paragraph_lengths.iter().sum::<usize>() as f64 / paragraph_lengths.len() as f64;
        let variance = paragraph_lengths
            .iter()
            .map(|&len| (len as f64 - mean_length).powi(2))
            .sum::<f64>()
            / paragraph_lengths.len() as f64;

        // Consistency score (higher variance = lower consistency)
        1.0 / (1.0 + variance.sqrt() / mean_length)
    }

    /// Generate detailed structural metrics
    fn generate_detailed_metrics(
        &self,
        paragraphs: &[String],
        sentences: &[String],
    ) -> Result<DetailedStructuralMetrics, StructuralCoherenceError> {
        let total_paragraphs = paragraphs.len();
        let paragraph_lengths: Vec<usize> = paragraphs
            .iter()
            .map(|p| p.split_whitespace().count())
            .collect();
        let average_paragraph_length =
            paragraph_lengths.iter().sum::<usize>() as f64 / paragraphs.len() as f64;

        let section_boundaries = self.detect_section_boundaries(paragraphs);
        let hierarchical_structure = self.analyze_hierarchical_structure(paragraphs)?;
        let discourse_patterns = self.analyze_discourse_patterns(paragraphs)?;
        let marker_analysis = self.analyze_structural_markers_detailed(paragraphs);
        let global_coherence = self.calculate_global_coherence_metrics(sentences, paragraphs);
        let document_completeness = self.analyze_document_completeness(paragraphs);
        let complexity_measures = self.calculate_complexity_measures(paragraphs, sentences);

        Ok(DetailedStructuralMetrics {
            total_paragraphs,
            average_paragraph_length,
            paragraph_length_distribution: paragraph_lengths,
            section_boundaries,
            hierarchical_structure,
            discourse_patterns,
            marker_analysis,
            global_coherence,
            document_completeness,
            complexity_measures,
        })
    }

    /// Detect section boundaries
    fn detect_section_boundaries(&self, paragraphs: &[String]) -> Vec<SectionBoundary> {
        let mut boundaries = Vec::new();

        for (idx, paragraph) in paragraphs.iter().enumerate() {
            let boundary_strength = self.calculate_boundary_strength(paragraph);

            if boundary_strength > 0.5 {
                let boundary_type = self.classify_boundary_type(paragraph);
                let section_title = self.extract_section_title(paragraph);
                let hierarchical_level = self.determine_hierarchical_level(paragraph);

                boundaries.push(SectionBoundary {
                    position: idx,
                    strength: boundary_strength,
                    boundary_type,
                    section_title,
                    hierarchical_level,
                });
            }
        }

        boundaries
    }

    /// Calculate boundary strength
    fn calculate_boundary_strength(&self, paragraph: &str) -> f64 {
        let boundary_indicators = vec![
            "chapter",
            "section",
            "introduction",
            "conclusion",
            "background",
            "methodology",
            "results",
            "discussion",
            "overview",
            "summary",
        ];

        let indicator_count = boundary_indicators
            .iter()
            .filter(|indicator| paragraph.to_lowercase().contains(indicator))
            .count();

        let structural_markers = self.count_structural_elements(paragraph);

        ((indicator_count + structural_markers) as f64 / 5.0).min(1.0)
    }

    /// Classify boundary type
    fn classify_boundary_type(&self, paragraph: &str) -> String {
        let paragraph_lower = paragraph.to_lowercase();

        if paragraph_lower.contains("chapter") {
            "chapter".to_string()
        } else if paragraph_lower.contains("section") {
            "section".to_string()
        } else if paragraph_lower.contains("introduction") {
            "introduction".to_string()
        } else if paragraph_lower.contains("conclusion") {
            "conclusion".to_string()
        } else {
            "paragraph".to_string()
        }
    }

    /// Extract section title
    fn extract_section_title(&self, paragraph: &str) -> Option<String> {
        // Simple heuristic: if paragraph is short and contains structural markers, treat as title
        let word_count = paragraph.split_whitespace().count();
        let has_structural_marker = self.count_structural_elements(paragraph) > 0;

        if word_count < 10 && has_structural_marker {
            Some(paragraph.trim().to_string())
        } else {
            None
        }
    }

    /// Determine hierarchical level
    fn determine_hierarchical_level(&self, paragraph: &str) -> HierarchicalLevel {
        let paragraph_lower = paragraph.to_lowercase();

        if paragraph_lower.contains("chapter") {
            HierarchicalLevel::Chapter
        } else if paragraph_lower.contains("section") {
            HierarchicalLevel::Section
        } else if paragraph_lower.contains("subsection") {
            HierarchicalLevel::Subsection
        } else {
            HierarchicalLevel::Paragraph
        }
    }

    /// Analyze hierarchical structure
    fn analyze_hierarchical_structure(
        &self,
        paragraphs: &[String],
    ) -> Result<HierarchicalStructureAnalysis, StructuralCoherenceError> {
        let detected_levels = self.detect_hierarchical_levels(paragraphs);
        let level_transitions = self.analyze_level_transitions(paragraphs);
        let balance_score = self.calculate_hierarchical_balance(&detected_levels);
        let depth_distribution = self.calculate_depth_distribution(&detected_levels);
        let structural_tree = if self.config.max_hierarchical_depth > 3 {
            Some(self.build_structural_tree(paragraphs)?)
        } else {
            None
        };
        let consistency_score = self.calculate_hierarchical_consistency(&level_transitions);

        Ok(HierarchicalStructureAnalysis {
            detected_levels,
            level_transitions,
            balance_score,
            depth_distribution,
            structural_tree,
            consistency_score,
        })
    }

    /// Detect hierarchical levels in paragraphs
    fn detect_hierarchical_levels(&self, paragraphs: &[String]) -> Vec<HierarchicalLevel> {
        paragraphs
            .iter()
            .map(|paragraph| self.determine_hierarchical_level(paragraph))
            .collect()
    }

    /// Analyze level transitions
    fn analyze_level_transitions(&self, paragraphs: &[String]) -> Vec<LevelTransition> {
        let mut transitions = Vec::new();

        for i in 0..paragraphs.len().saturating_sub(1) {
            let from_level = self.determine_hierarchical_level(&paragraphs[i]);
            let to_level = self.determine_hierarchical_level(&paragraphs[i + 1]);

            if from_level != to_level {
                let quality = self.calculate_transition_quality(&from_level, &to_level);
                let appropriateness =
                    self.calculate_transition_appropriateness(&from_level, &to_level);

                transitions.push(LevelTransition {
                    from_level,
                    to_level,
                    position: i + 1,
                    quality,
                    appropriateness,
                });
            }
        }

        transitions
    }

    /// Calculate transition quality between levels
    fn calculate_transition_quality(
        &self,
        from_level: &HierarchicalLevel,
        to_level: &HierarchicalLevel,
    ) -> f64 {
        let level_distance = self.calculate_level_distance(from_level, to_level);

        match level_distance {
            0 => 1.0,
            1 => 0.9,
            2 => 0.7,
            3 => 0.5,
            _ => 0.3,
        }
    }

    /// Calculate distance between hierarchical levels
    fn calculate_level_distance(
        &self,
        level1: &HierarchicalLevel,
        level2: &HierarchicalLevel,
    ) -> usize {
        let level1_num = self.hierarchical_level_to_number(level1);
        let level2_num = self.hierarchical_level_to_number(level2);

        (level1_num as i32 - level2_num as i32).unsigned_abs() as usize
    }

    /// Convert hierarchical level to number for comparison
    fn hierarchical_level_to_number(&self, level: &HierarchicalLevel) -> usize {
        match level {
            HierarchicalLevel::Document => 0,
            HierarchicalLevel::Chapter => 1,
            HierarchicalLevel::Section => 2,
            HierarchicalLevel::Subsection => 3,
            HierarchicalLevel::Paragraph => 4,
            HierarchicalLevel::Sentence => 5,
            HierarchicalLevel::Phrase => 6,
        }
    }

    /// Calculate transition appropriateness
    fn calculate_transition_appropriateness(
        &self,
        from_level: &HierarchicalLevel,
        to_level: &HierarchicalLevel,
    ) -> f64 {
        // Simplified appropriateness based on typical document structure
        match (from_level, to_level) {
            (HierarchicalLevel::Chapter, HierarchicalLevel::Section) => 0.9,
            (HierarchicalLevel::Section, HierarchicalLevel::Subsection) => 0.9,
            (HierarchicalLevel::Section, HierarchicalLevel::Paragraph) => 0.8,
            (HierarchicalLevel::Paragraph, HierarchicalLevel::Paragraph) => 0.7,
            _ => 0.5,
        }
    }

    /// Calculate hierarchical balance
    fn calculate_hierarchical_balance(&self, levels: &[HierarchicalLevel]) -> f64 {
        if levels.is_empty() {
            return 0.0;
        }

        // Count occurrences of each level
        let mut level_counts = HashMap::new();
        for level in levels {
            *level_counts.entry(format!("{:?}", level)).or_insert(0) += 1;
        }

        // Calculate entropy (higher entropy = better balance)
        let total = levels.len() as f64;
        let entropy = level_counts
            .values()
            .map(|&count| {
                let p = count as f64 / total;
                if p > 0.0 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();

        let max_entropy = (level_counts.len() as f64).ln();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Calculate depth distribution
    fn calculate_depth_distribution(&self, levels: &[HierarchicalLevel]) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();

        for level in levels {
            *distribution.entry(format!("{:?}", level)).or_insert(0) += 1;
        }

        distribution
    }

    /// Build structural tree
    fn build_structural_tree(
        &self,
        paragraphs: &[String],
    ) -> Result<StructuralTree, StructuralCoherenceError> {
        let root = StructuralNode {
            node_id: 0,
            level: HierarchicalLevel::Document,
            content_span: (0, paragraphs.len().saturating_sub(1)),
            title: Some("Document".to_string()),
            children: Vec::new(),
            coherence_score: 0.8,
        };

        Ok(StructuralTree {
            root,
            depth: self.config.max_hierarchical_depth,
            node_count: paragraphs.len(),
            balance_score: 0.7,
        })
    }

    /// Calculate hierarchical consistency
    fn calculate_hierarchical_consistency(&self, transitions: &[LevelTransition]) -> f64 {
        if transitions.is_empty() {
            return 1.0;
        }

        let total_quality: f64 = transitions.iter().map(|t| t.quality).sum();
        total_quality / transitions.len() as f64
    }

    /// Analyze discourse patterns
    fn analyze_discourse_patterns(
        &self,
        paragraphs: &[String],
    ) -> Result<DiscoursePatternAnalysis, StructuralCoherenceError> {
        let detected_patterns = self.detect_discourse_patterns(paragraphs);
        let pattern_distribution = self.calculate_pattern_distribution(&detected_patterns);
        let pattern_coherence = self.calculate_pattern_coherence_scores(&detected_patterns);
        let pattern_transitions = self.analyze_pattern_transitions(&detected_patterns);
        let pattern_consistency = self.calculate_pattern_consistency(&detected_patterns);

        Ok(DiscoursePatternAnalysis {
            detected_patterns,
            pattern_distribution,
            pattern_coherence,
            pattern_transitions,
            pattern_consistency,
        })
    }

    /// Detect discourse patterns in text
    fn detect_discourse_patterns(&self, paragraphs: &[String]) -> Vec<DetectedPattern> {
        let mut patterns = Vec::new();

        for (idx, paragraph) in paragraphs.iter().enumerate() {
            let pattern_type = self.classify_discourse_pattern(paragraph);
            let strength = self.calculate_pattern_strength(paragraph, &pattern_type);
            let completeness = self.calculate_pattern_completeness(paragraph, &pattern_type);
            let evidence = self.collect_pattern_evidence(paragraph, &pattern_type);

            patterns.push(DetectedPattern {
                pattern_type,
                span: (idx, idx),
                strength,
                completeness,
                evidence,
            });
        }

        patterns
    }

    /// Classify discourse pattern for paragraph
    fn classify_discourse_pattern(&self, paragraph: &str) -> DiscoursePatternType {
        let paragraph_lower = paragraph.to_lowercase();

        // Simple pattern detection based on keywords
        if paragraph_lower.contains("problem") && paragraph_lower.contains("solution") {
            DiscoursePatternType::ProblemSolution
        } else if paragraph_lower.contains("cause") || paragraph_lower.contains("effect") {
            DiscoursePatternType::CauseEffect
        } else if paragraph_lower.contains("compare") || paragraph_lower.contains("contrast") {
            DiscoursePatternType::CompareContrast
        } else if paragraph_lower.contains("first")
            || paragraph_lower.contains("then")
            || paragraph_lower.contains("finally")
        {
            DiscoursePatternType::Chronological
        } else if paragraph_lower.contains("definition") || paragraph_lower.contains("define") {
            DiscoursePatternType::Definition
        } else {
            DiscoursePatternType::Mixed
        }
    }

    /// Calculate pattern strength
    fn calculate_pattern_strength(
        &self,
        paragraph: &str,
        pattern_type: &DiscoursePatternType,
    ) -> f64 {
        let keywords = self.get_pattern_keywords(pattern_type);
        let paragraph_lower = paragraph.to_lowercase();

        let keyword_count = keywords
            .iter()
            .filter(|keyword| paragraph_lower.contains(keyword.as_str()))
            .count();

        (keyword_count as f64 / keywords.len() as f64).min(1.0)
    }

    /// Get keywords for discourse pattern
    fn get_pattern_keywords(&self, pattern_type: &DiscoursePatternType) -> Vec<String> {
        match pattern_type {
            DiscoursePatternType::ProblemSolution => vec![
                "problem".to_string(),
                "solution".to_string(),
                "issue".to_string(),
                "resolve".to_string(),
                "solve".to_string(),
                "address".to_string(),
            ],
            DiscoursePatternType::CauseEffect => vec![
                "cause".to_string(),
                "effect".to_string(),
                "result".to_string(),
                "because".to_string(),
                "therefore".to_string(),
                "consequently".to_string(),
            ],
            DiscoursePatternType::CompareContrast => vec![
                "compare".to_string(),
                "contrast".to_string(),
                "similar".to_string(),
                "different".to_string(),
                "however".to_string(),
                "whereas".to_string(),
            ],
            DiscoursePatternType::Chronological => vec![
                "first".to_string(),
                "second".to_string(),
                "then".to_string(),
                "next".to_string(),
                "finally".to_string(),
                "after".to_string(),
            ],
            _ => Vec::new(),
        }
    }

    /// Calculate pattern completeness
    fn calculate_pattern_completeness(
        &self,
        paragraph: &str,
        pattern_type: &DiscoursePatternType,
    ) -> f64 {
        // Simplified completeness based on paragraph length and keyword density
        let word_count = paragraph.split_whitespace().count();
        let keywords = self.get_pattern_keywords(pattern_type);
        let keyword_density = keywords
            .iter()
            .filter(|keyword| paragraph.to_lowercase().contains(keyword.as_str()))
            .count() as f64
            / word_count as f64;

        (keyword_density * 10.0).min(1.0)
    }

    /// Collect pattern evidence
    fn collect_pattern_evidence(
        &self,
        paragraph: &str,
        pattern_type: &DiscoursePatternType,
    ) -> Vec<String> {
        let keywords = self.get_pattern_keywords(pattern_type);
        keywords
            .iter()
            .filter(|keyword| paragraph.to_lowercase().contains(keyword.as_str()))
            .cloned()
            .collect()
    }

    /// Calculate pattern distribution
    fn calculate_pattern_distribution(&self, patterns: &[DetectedPattern]) -> HashMap<String, f64> {
        let mut distribution = HashMap::new();
        let total_patterns = patterns.len() as f64;

        for pattern in patterns {
            let pattern_name = format!("{:?}", pattern.pattern_type);
            *distribution.entry(pattern_name).or_insert(0.0) += 1.0;
        }

        // Normalize to get proportions
        for value in distribution.values_mut() {
            *value /= total_patterns;
        }

        distribution
    }

    /// Calculate pattern coherence scores
    fn calculate_pattern_coherence_scores(
        &self,
        patterns: &[DetectedPattern],
    ) -> HashMap<String, f64> {
        let mut coherence_scores = HashMap::new();

        // Group patterns by type
        let mut pattern_groups: HashMap<String, Vec<&DetectedPattern>> = HashMap::new();
        for pattern in patterns {
            let pattern_name = format!("{:?}", pattern.pattern_type);
            pattern_groups
                .entry(pattern_name)
                .or_insert_with(Vec::new)
                .push(pattern);
        }

        // Calculate average coherence for each pattern type
        for (pattern_name, group) in pattern_groups {
            let average_strength: f64 =
                group.iter().map(|p| p.strength).sum::<f64>() / group.len() as f64;
            coherence_scores.insert(pattern_name, average_strength);
        }

        coherence_scores
    }

    /// Analyze pattern transitions
    fn analyze_pattern_transitions(&self, patterns: &[DetectedPattern]) -> Vec<PatternTransition> {
        let mut transitions = Vec::new();

        for i in 0..patterns.len().saturating_sub(1) {
            let from_pattern = patterns[i].pattern_type.clone();
            let to_pattern = patterns[i + 1].pattern_type.clone();

            if from_pattern != to_pattern {
                let smoothness =
                    self.calculate_pattern_transition_smoothness(&from_pattern, &to_pattern);

                transitions.push(PatternTransition {
                    from_pattern,
                    to_pattern,
                    position: i + 1,
                    smoothness,
                });
            }
        }

        transitions
    }

    /// Calculate pattern transition smoothness
    fn calculate_pattern_transition_smoothness(
        &self,
        _from: &DiscoursePatternType,
        _to: &DiscoursePatternType,
    ) -> f64 {
        // Simplified transition smoothness
        0.6
    }

    /// Calculate pattern consistency
    fn calculate_pattern_consistency(&self, patterns: &[DetectedPattern]) -> f64 {
        if patterns.is_empty() {
            return 0.0;
        }

        let average_strength: f64 =
            patterns.iter().map(|p| p.strength).sum::<f64>() / patterns.len() as f64;
        let average_completeness: f64 =
            patterns.iter().map(|p| p.completeness).sum::<f64>() / patterns.len() as f64;

        (average_strength + average_completeness) / 2.0
    }

    /// Analyze structural markers in detail
    fn analyze_structural_markers_detailed(
        &self,
        paragraphs: &[String],
    ) -> StructuralMarkerAnalysis {
        let markers_by_type = self.categorize_markers_by_type(paragraphs);
        let marker_density = self.calculate_marker_density(paragraphs);
        let distribution_analysis = self.analyze_marker_distribution(paragraphs);
        let effectiveness_scores = self.calculate_marker_effectiveness(paragraphs);
        let missing_markers = self.identify_missing_markers(paragraphs);

        StructuralMarkerAnalysis {
            markers_by_type,
            marker_density,
            distribution_analysis,
            effectiveness_scores,
            missing_markers,
        }
    }

    /// Categorize markers by type
    fn categorize_markers_by_type(
        &self,
        paragraphs: &[String],
    ) -> HashMap<String, Vec<StructuralMarker>> {
        let mut markers_by_type = HashMap::new();

        if let Ok(markers_map) = self.structural_markers.read() {
            for (para_idx, paragraph) in paragraphs.iter().enumerate() {
                let words: Vec<&str> = paragraph.split_whitespace().collect();

                for (word_idx, word) in words.iter().enumerate() {
                    let clean_word = word
                        .trim_matches(|c: char| !c.is_alphabetic())
                        .to_lowercase();

                    if let Some(marker_type) = markers_map.get(&clean_word) {
                        let marker = StructuralMarker {
                            marker_text: clean_word.clone(),
                            marker_type: marker_type.clone(),
                            position: (para_idx, word_idx),
                            effectiveness: 0.8,   // Simplified
                            appropriateness: 0.7, // Simplified
                        };

                        let type_name = format!("{:?}", marker_type);
                        markers_by_type
                            .entry(type_name)
                            .or_insert_with(Vec::new)
                            .push(marker);
                    }
                }
            }
        }

        markers_by_type
    }

    /// Calculate marker density
    fn calculate_marker_density(&self, paragraphs: &[String]) -> f64 {
        let total_markers = self.extract_structural_markers(paragraphs).len();
        let total_paragraphs = paragraphs.len();

        if total_paragraphs > 0 {
            total_markers as f64 / total_paragraphs as f64
        } else {
            0.0
        }
    }

    /// Analyze marker distribution
    fn analyze_marker_distribution(&self, paragraphs: &[String]) -> MarkerDistributionAnalysis {
        // Simplified marker distribution analysis
        MarkerDistributionAnalysis {
            evenness: 0.7,
            clustering_coefficient: 0.5,
            distribution_alignment: 0.6,
            spacing_analysis: SpacingAnalysis {
                average_spacing: 2.5,
                spacing_variance: 1.2,
                optimal_alignment: 0.8,
                regularity: 0.6,
            },
        }
    }

    /// Calculate marker effectiveness
    fn calculate_marker_effectiveness(&self, _paragraphs: &[String]) -> HashMap<String, f64> {
        let mut effectiveness = HashMap::new();

        effectiveness.insert("Introduction".to_string(), 0.9);
        effectiveness.insert("Transition".to_string(), 0.8);
        effectiveness.insert("Conclusion".to_string(), 0.9);
        effectiveness.insert("Enumeration".to_string(), 0.7);

        effectiveness
    }

    /// Identify missing markers
    fn identify_missing_markers(&self, paragraphs: &[String]) -> Vec<String> {
        let mut missing = Vec::new();

        // Check for missing introduction markers
        if !paragraphs.is_empty() {
            let first_paragraph = &paragraphs[0];
            if self.calculate_introduction_quality(first_paragraph) < 0.3 {
                missing.push("introduction_marker".to_string());
            }
        }

        // Check for missing conclusion markers
        if !paragraphs.is_empty() {
            let last_paragraph = paragraphs.last().unwrap();
            if self.calculate_conclusion_quality(last_paragraph) < 0.3 {
                missing.push("conclusion_marker".to_string());
            }
        }

        missing
    }

    /// Calculate global coherence metrics
    fn calculate_global_coherence_metrics(
        &self,
        sentences: &[String],
        paragraphs: &[String],
    ) -> GlobalCoherenceMetrics {
        let beginning_end_coherence = self.calculate_beginning_end_coherence(paragraphs);
        let thematic_consistency = self.calculate_thematic_consistency(paragraphs);
        let information_flow = self.calculate_information_flow(sentences);
        let structural_completeness = self.calculate_structural_completeness(paragraphs);
        let document_unity = self.calculate_document_unity(paragraphs);
        let cross_paragraph_coherence = self.calculate_cross_paragraph_coherence(paragraphs);

        GlobalCoherenceMetrics {
            beginning_end_coherence,
            thematic_consistency,
            information_flow,
            structural_completeness,
            document_unity,
            cross_paragraph_coherence,
        }
    }

    /// Calculate beginning-end coherence
    fn calculate_beginning_end_coherence(&self, paragraphs: &[String]) -> f64 {
        if paragraphs.len() < 2 {
            return 1.0;
        }

        let first_paragraph = &paragraphs[0];
        let last_paragraph = paragraphs.last().unwrap();

        let lexical_overlap = self.calculate_lexical_overlap(first_paragraph, last_paragraph);
        let semantic_continuity =
            self.calculate_semantic_continuity(first_paragraph, last_paragraph);

        (lexical_overlap * 0.6) + (semantic_continuity * 0.4)
    }

    /// Calculate thematic consistency
    fn calculate_thematic_consistency(&self, paragraphs: &[String]) -> f64 {
        if paragraphs.len() < 2 {
            return 1.0;
        }

        let mut consistency_scores = Vec::new();

        for i in 0..paragraphs.len() {
            for j in i + 1..paragraphs.len() {
                let consistency =
                    self.calculate_semantic_continuity(&paragraphs[i], &paragraphs[j]);
                consistency_scores.push(consistency);
            }
        }

        consistency_scores.iter().sum::<f64>() / consistency_scores.len() as f64
    }

    /// Calculate information flow
    fn calculate_information_flow(&self, sentences: &[String]) -> f64 {
        if sentences.len() < 2 {
            return 1.0;
        }

        let mut flow_scores = Vec::new();

        for i in 0..sentences.len() - 1 {
            let flow_score =
                self.calculate_sentence_pair_coherence(&sentences[i], &sentences[i + 1]);
            flow_scores.push(flow_score);
        }

        flow_scores.iter().sum::<f64>() / flow_scores.len() as f64
    }

    /// Calculate structural completeness
    fn calculate_structural_completeness(&self, paragraphs: &[String]) -> f64 {
        let has_introduction = paragraphs
            .first()
            .map(|p| self.calculate_introduction_quality(p) > 0.3)
            .unwrap_or(false);

        let has_conclusion = paragraphs
            .last()
            .map(|p| self.calculate_conclusion_quality(p) > 0.3)
            .unwrap_or(false);

        let has_development = paragraphs.len() > 2;

        let completeness_score = [has_introduction, has_conclusion, has_development]
            .iter()
            .filter(|&&x| x)
            .count() as f64
            / 3.0;

        completeness_score
    }

    /// Calculate document unity
    fn calculate_document_unity(&self, paragraphs: &[String]) -> f64 {
        // Simplified unity calculation based on average paragraph coherence
        self.calculate_paragraph_coherence(paragraphs)
    }

    /// Calculate cross-paragraph coherence
    fn calculate_cross_paragraph_coherence(&self, paragraphs: &[String]) -> f64 {
        if paragraphs.len() < 2 {
            return 1.0;
        }

        let mut total_coherence = 0.0;
        let mut comparisons = 0;

        for i in 0..paragraphs.len() {
            for j in i + 1..paragraphs.len() {
                let coherence =
                    self.calculate_paragraph_pair_coherence(&paragraphs[i], &paragraphs[j]);
                total_coherence += coherence;
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            total_coherence / comparisons as f64
        } else {
            0.0
        }
    }

    /// Analyze document completeness
    fn analyze_document_completeness(&self, paragraphs: &[String]) -> DocumentCompletenessMetrics {
        let has_introduction = paragraphs
            .first()
            .map(|p| self.calculate_introduction_quality(p) > 0.3)
            .unwrap_or(false);

        let has_development = paragraphs.len() > 2;

        let has_conclusion = paragraphs
            .last()
            .map(|p| self.calculate_conclusion_quality(p) > 0.3)
            .unwrap_or(false);

        let completeness_score = [has_introduction, has_development, has_conclusion]
            .iter()
            .filter(|&&x| x)
            .count() as f64
            / 3.0;

        let mut missing_components = Vec::new();
        if !has_introduction {
            missing_components.push("introduction".to_string());
        }
        if !has_development {
            missing_components.push("development".to_string());
        }
        if !has_conclusion {
            missing_components.push("conclusion".to_string());
        }

        let mut component_quality = HashMap::new();
        if let Some(first_para) = paragraphs.first() {
            component_quality.insert(
                "introduction".to_string(),
                self.calculate_introduction_quality(first_para),
            );
        }
        if paragraphs.len() > 2 {
            let development_quality =
                self.calculate_development_quality(&paragraphs[1..paragraphs.len() - 1]);
            component_quality.insert("development".to_string(), development_quality);
        }
        if let Some(last_para) = paragraphs.last() {
            component_quality.insert(
                "conclusion".to_string(),
                self.calculate_conclusion_quality(last_para),
            );
        }

        DocumentCompletenessMetrics {
            has_introduction,
            has_development,
            has_conclusion,
            completeness_score,
            missing_components,
            component_quality,
        }
    }

    /// Calculate complexity measures
    fn calculate_complexity_measures(
        &self,
        paragraphs: &[String],
        sentences: &[String],
    ) -> StructuralComplexityMetrics {
        let hierarchical_complexity = self.calculate_hierarchical_complexity(paragraphs);
        let organizational_complexity = self.calculate_organizational_complexity(paragraphs);
        let pattern_complexity = self.calculate_pattern_complexity(paragraphs);
        let structural_diversity = self.calculate_structural_diversity(paragraphs);
        let integration_complexity = self.calculate_integration_complexity(paragraphs, sentences);
        let cognitive_load = self.calculate_cognitive_load(paragraphs, sentences);

        StructuralComplexityMetrics {
            hierarchical_complexity,
            organizational_complexity,
            pattern_complexity,
            structural_diversity,
            integration_complexity,
            cognitive_load,
        }
    }

    /// Calculate hierarchical complexity
    fn calculate_hierarchical_complexity(&self, paragraphs: &[String]) -> f64 {
        let levels = self.detect_hierarchical_levels(paragraphs);
        let unique_levels: HashSet<_> = levels.iter().collect();

        // Complexity increases with number of levels and transitions
        let level_diversity = unique_levels.len() as f64;
        let transition_count = paragraphs.len().saturating_sub(1) as f64;

        (level_diversity.ln() + transition_count.ln()) / 5.0
    }

    /// Calculate organizational complexity
    fn calculate_organizational_complexity(&self, paragraphs: &[String]) -> f64 {
        let marker_count = self.extract_structural_markers(paragraphs).len() as f64;
        let paragraph_count = paragraphs.len() as f64;

        if paragraph_count > 0.0 {
            (marker_count / paragraph_count).min(1.0)
        } else {
            0.0
        }
    }

    /// Calculate pattern complexity
    fn calculate_pattern_complexity(&self, paragraphs: &[String]) -> f64 {
        let patterns = self.detect_discourse_patterns(paragraphs);
        let pattern_types: HashSet<_> = patterns.iter().map(|p| &p.pattern_type).collect();

        pattern_types.len() as f64 / 5.0 // Normalize by expected max patterns
    }

    /// Calculate structural diversity
    fn calculate_structural_diversity(&self, paragraphs: &[String]) -> f64 {
        // Diversity based on paragraph length variance
        if paragraphs.is_empty() {
            return 0.0;
        }

        let lengths: Vec<f64> = paragraphs
            .iter()
            .map(|p| p.split_whitespace().count() as f64)
            .collect();

        let mean_length = lengths.iter().sum::<f64>() / lengths.len() as f64;
        let variance = lengths
            .iter()
            .map(|&len| (len - mean_length).powi(2))
            .sum::<f64>()
            / lengths.len() as f64;

        (variance.sqrt() / mean_length).min(1.0)
    }

    /// Calculate integration complexity
    fn calculate_integration_complexity(&self, paragraphs: &[String], sentences: &[String]) -> f64 {
        let paragraph_count = paragraphs.len() as f64;
        let sentence_count = sentences.len() as f64;
        let avg_sentences_per_paragraph = sentence_count / paragraph_count.max(1.0);

        (avg_sentences_per_paragraph / 10.0).min(1.0)
    }

    /// Calculate cognitive load
    fn calculate_cognitive_load(&self, paragraphs: &[String], sentences: &[String]) -> f64 {
        let paragraph_complexity = self.calculate_hierarchical_complexity(paragraphs);
        let sentence_complexity = sentences.len() as f64 / 50.0; // Normalize by typical document size
        let marker_complexity =
            self.extract_structural_markers(paragraphs).len() as f64 / paragraphs.len() as f64;

        ((paragraph_complexity + sentence_complexity + marker_complexity) / 3.0).min(1.0)
    }

    /// Analyze document structure
    fn analyze_document_structure(
        &self,
        paragraphs: &[String],
        _sentences: &[String],
    ) -> Result<DocumentStructureAnalysis, StructuralCoherenceError> {
        let detected_structure_type = self.detect_structure_type(paragraphs);
        let structure_confidence =
            self.calculate_structure_confidence(paragraphs, &detected_structure_type);
        let structure_features = self.extract_structure_features(paragraphs);
        let adherence_score =
            self.calculate_structure_adherence(paragraphs, &detected_structure_type);
        let alternative_structures = self.suggest_alternative_structures(paragraphs);

        Ok(DocumentStructureAnalysis {
            detected_structure_type,
            structure_confidence,
            structure_features,
            adherence_score,
            alternative_structures,
        })
    }

    /// Detect document structure type
    fn detect_structure_type(&self, paragraphs: &[String]) -> DocumentStructureType {
        if let Some(expected) = &self.config.expected_structure_type {
            return expected.clone();
        }

        // Simple heuristic-based detection
        let text = paragraphs.join(" ").to_lowercase();

        if text.contains("abstract") || text.contains("methodology") || text.contains("results") {
            DocumentStructureType::Academic
        } else if text.contains("introduction") && text.contains("conclusion") {
            DocumentStructureType::Technical
        } else if text.contains("once upon") || text.contains("story") {
            DocumentStructureType::Narrative
        } else if text.contains("argue") || text.contains("claim") || text.contains("evidence") {
            DocumentStructureType::Argumentative
        } else {
            DocumentStructureType::Unknown
        }
    }

    /// Calculate structure confidence
    fn calculate_structure_confidence(
        &self,
        paragraphs: &[String],
        structure_type: &DocumentStructureType,
    ) -> f64 {
        let expected_features = self.get_expected_structure_features(structure_type);
        let text = paragraphs.join(" ").to_lowercase();

        let feature_count = expected_features
            .iter()
            .filter(|feature| text.contains(feature.as_str()))
            .count();

        feature_count as f64 / expected_features.len() as f64
    }

    /// Get expected features for structure type
    fn get_expected_structure_features(
        &self,
        structure_type: &DocumentStructureType,
    ) -> Vec<String> {
        match structure_type {
            DocumentStructureType::Academic => vec![
                "abstract".to_string(),
                "introduction".to_string(),
                "methodology".to_string(),
                "results".to_string(),
                "discussion".to_string(),
                "conclusion".to_string(),
            ],
            DocumentStructureType::Technical => vec![
                "overview".to_string(),
                "specifications".to_string(),
                "implementation".to_string(),
                "examples".to_string(),
                "summary".to_string(),
            ],
            DocumentStructureType::Argumentative => vec![
                "thesis".to_string(),
                "argument".to_string(),
                "evidence".to_string(),
                "counterargument".to_string(),
                "conclusion".to_string(),
            ],
            _ => Vec::new(),
        }
    }

    /// Extract structure features
    fn extract_structure_features(&self, paragraphs: &[String]) -> HashMap<String, f64> {
        let mut features = HashMap::new();

        let text = paragraphs.join(" ").to_lowercase();

        features.insert(
            "has_introduction".to_string(),
            if text.contains("introduction") {
                1.0
            } else {
                0.0
            },
        );
        features.insert(
            "has_conclusion".to_string(),
            if text.contains("conclusion") {
                1.0
            } else {
                0.0
            },
        );
        features.insert(
            "has_structure_markers".to_string(),
            self.extract_structural_markers(paragraphs).len() as f64 / paragraphs.len() as f64,
        );

        features
    }

    /// Calculate structure adherence
    fn calculate_structure_adherence(
        &self,
        paragraphs: &[String],
        structure_type: &DocumentStructureType,
    ) -> f64 {
        self.calculate_structure_confidence(paragraphs, structure_type)
    }

    /// Suggest alternative structures
    fn suggest_alternative_structures(&self, _paragraphs: &[String]) -> Vec<StructureSuggestion> {
        vec![StructureSuggestion {
            suggested_type: DocumentStructureType::Technical,
            confidence: 0.7,
            improvement_potential: 0.3,
            recommendations: vec![
                "Add section headers".to_string(),
                "Use clearer transitions".to_string(),
            ],
        }]
    }

    /// Perform advanced structural analysis
    fn perform_advanced_analysis(
        &self,
        paragraphs: &[String],
        sentences: &[String],
    ) -> Result<AdvancedStructuralAnalysis, StructuralCoherenceError> {
        let rhetorical_structure = self.analyze_rhetorical_structure(paragraphs)?;
        let information_architecture =
            self.analyze_information_architecture(paragraphs, sentences)?;
        let cognitive_structure = self.analyze_cognitive_structure(paragraphs, sentences)?;
        let genre_analysis = self.analyze_genre_specific_features(paragraphs)?;
        let quality_assessment = self.assess_structural_quality(paragraphs, sentences)?;

        Ok(AdvancedStructuralAnalysis {
            rhetorical_structure,
            information_architecture,
            cognitive_structure,
            genre_analysis,
            quality_assessment,
        })
    }

    /// Analyze rhetorical structure
    fn analyze_rhetorical_structure(
        &self,
        paragraphs: &[String],
    ) -> Result<RhetoricalStructureAnalysis, StructuralCoherenceError> {
        let rhetorical_moves = self.identify_rhetorical_moves(paragraphs);
        let move_sequences = self.analyze_move_sequences(&rhetorical_moves);
        let effectiveness_score = self.calculate_rhetorical_effectiveness(&rhetorical_moves);
        let genre_appropriateness = self.calculate_genre_appropriateness(&rhetorical_moves);

        Ok(RhetoricalStructureAnalysis {
            rhetorical_moves,
            move_sequences,
            effectiveness_score,
            genre_appropriateness,
        })
    }

    /// Identify rhetorical moves
    fn identify_rhetorical_moves(&self, paragraphs: &[String]) -> Vec<RhetoricalMove> {
        paragraphs
            .iter()
            .enumerate()
            .map(|(idx, paragraph)| {
                let move_type = self.classify_rhetorical_move(paragraph);
                let strength = self.calculate_move_strength(paragraph, &move_type);
                let function = self.determine_move_function(&move_type);
                let linguistic_markers = self.extract_linguistic_markers(paragraph, &move_type);

                RhetoricalMove {
                    move_type,
                    span: (idx, idx),
                    strength,
                    function,
                    linguistic_markers,
                }
            })
            .collect()
    }

    /// Classify rhetorical move
    fn classify_rhetorical_move(&self, paragraph: &str) -> String {
        let paragraph_lower = paragraph.to_lowercase();

        if paragraph_lower.contains("introduction") || paragraph_lower.contains("begin") {
            "Introduction".to_string()
        } else if paragraph_lower.contains("evidence") || paragraph_lower.contains("support") {
            "Evidence".to_string()
        } else if paragraph_lower.contains("conclusion") || paragraph_lower.contains("therefore") {
            "Conclusion".to_string()
        } else {
            "Development".to_string()
        }
    }

    /// Calculate move strength
    fn calculate_move_strength(&self, paragraph: &str, move_type: &str) -> f64 {
        let word_count = paragraph.split_whitespace().count();
        let marker_count = self.count_structural_elements(paragraph);

        ((marker_count as f64 / 5.0) + (word_count as f64 / 100.0)).min(1.0)
    }

    /// Determine move function
    fn determine_move_function(&self, move_type: &str) -> String {
        match move_type {
            "Introduction" => "Establish context and purpose".to_string(),
            "Evidence" => "Support claims with data".to_string(),
            "Conclusion" => "Summarize and synthesize".to_string(),
            _ => "Develop argument".to_string(),
        }
    }

    /// Extract linguistic markers
    fn extract_linguistic_markers(&self, paragraph: &str, _move_type: &str) -> Vec<String> {
        self.extract_structural_markers(&[paragraph.to_string()])
    }

    /// Analyze move sequences
    fn analyze_move_sequences(&self, moves: &[RhetoricalMove]) -> Vec<MoveSequence> {
        vec![MoveSequence {
            sequence_id: 0,
            moves: moves.iter().map(|m| m.move_type.clone()).collect(),
            coherence: 0.8,
            appropriateness: 0.7,
        }]
    }

    /// Calculate rhetorical effectiveness
    fn calculate_rhetorical_effectiveness(&self, moves: &[RhetoricalMove]) -> f64 {
        if moves.is_empty() {
            return 0.0;
        }

        let average_strength: f64 =
            moves.iter().map(|m| m.strength).sum::<f64>() / moves.len() as f64;
        average_strength
    }

    /// Calculate genre appropriateness
    fn calculate_genre_appropriateness(&self, _moves: &[RhetoricalMove]) -> f64 {
        0.7 // Simplified
    }

    /// Analyze information architecture
    fn analyze_information_architecture(
        &self,
        paragraphs: &[String],
        sentences: &[String],
    ) -> Result<InformationArchitectureAnalysis, StructuralCoherenceError> {
        let information_hierarchy = self.build_information_hierarchy(paragraphs);
        let flow_analysis = self.analyze_information_flow(sentences);
        let density_distribution = self.calculate_information_density_distribution(paragraphs);
        let effectiveness =
            self.calculate_architecture_effectiveness(&information_hierarchy, &flow_analysis);

        Ok(InformationArchitectureAnalysis {
            information_hierarchy,
            flow_analysis,
            density_distribution,
            effectiveness,
        })
    }

    /// Build information hierarchy
    fn build_information_hierarchy(&self, paragraphs: &[String]) -> InformationHierarchy {
        let levels = paragraphs
            .iter()
            .enumerate()
            .map(|(idx, paragraph)| InformationLevel {
                level_id: idx,
                importance: self.calculate_paragraph_importance(paragraph),
                density: self.calculate_information_density(paragraph),
                categories: self.categorize_paragraph_content(paragraph),
            })
            .collect();

        let relationships = Vec::new(); // Simplified

        InformationHierarchy {
            levels,
            relationships,
            balance_score: 0.7,
        }
    }

    /// Calculate paragraph importance
    fn calculate_paragraph_importance(&self, paragraph: &str) -> f64 {
        let word_count = paragraph.split_whitespace().count() as f64;
        let marker_count = self.count_structural_elements(paragraph) as f64;

        ((word_count / 100.0) + (marker_count / 5.0)).min(1.0)
    }

    /// Calculate information density
    fn calculate_information_density(&self, paragraph: &str) -> f64 {
        let content_words = self.extract_content_words(paragraph);
        let total_words = paragraph.split_whitespace().count();

        if total_words > 0 {
            content_words.len() as f64 / total_words as f64
        } else {
            0.0
        }
    }

    /// Categorize paragraph content
    fn categorize_paragraph_content(&self, paragraph: &str) -> Vec<String> {
        let mut categories = Vec::new();

        let paragraph_lower = paragraph.to_lowercase();
        if paragraph_lower.contains("method") || paragraph_lower.contains("approach") {
            categories.push("methodology".to_string());
        }
        if paragraph_lower.contains("result") || paragraph_lower.contains("finding") {
            categories.push("results".to_string());
        }
        if paragraph_lower.contains("discuss") || paragraph_lower.contains("analyze") {
            categories.push("discussion".to_string());
        }

        if categories.is_empty() {
            categories.push("general".to_string());
        }

        categories
    }

    /// Analyze information flow
    fn analyze_information_flow(&self, sentences: &[String]) -> InformationFlowAnalysis {
        InformationFlowAnalysis {
            flow_direction: "forward".to_string(),
            continuity_score: self.calculate_information_flow(sentences),
            bottlenecks: Vec::new(),
            efficiency: 0.7,
        }
    }

    /// Calculate information density distribution
    fn calculate_information_density_distribution(&self, paragraphs: &[String]) -> Vec<f64> {
        paragraphs
            .iter()
            .map(|paragraph| self.calculate_information_density(paragraph))
            .collect()
    }

    /// Calculate architecture effectiveness
    fn calculate_architecture_effectiveness(
        &self,
        hierarchy: &InformationHierarchy,
        flow: &InformationFlowAnalysis,
    ) -> f64 {
        (hierarchy.balance_score + flow.continuity_score + flow.efficiency) / 3.0
    }

    /// Analyze cognitive structure
    fn analyze_cognitive_structure(
        &self,
        paragraphs: &[String],
        sentences: &[String],
    ) -> Result<CognitiveStructureAnalysis, StructuralCoherenceError> {
        let load_distribution = self.calculate_cognitive_load_distribution(paragraphs);
        let processing_complexity = self.calculate_processing_complexity(sentences);
        let working_memory_demands = self.calculate_working_memory_demands(paragraphs);
        let comprehension_ease = self.calculate_comprehension_ease(paragraphs, sentences);
        let mental_model_coherence = self.calculate_mental_model_coherence(paragraphs);

        Ok(CognitiveStructureAnalysis {
            load_distribution,
            processing_complexity,
            working_memory_demands,
            comprehension_ease,
            mental_model_coherence,
        })
    }

    /// Calculate cognitive load distribution
    fn calculate_cognitive_load_distribution(&self, paragraphs: &[String]) -> Vec<f64> {
        paragraphs
            .iter()
            .map(|paragraph| {
                let word_count = paragraph.split_whitespace().count() as f64;
                let complexity = self.count_structural_elements(paragraph) as f64;
                (word_count / 50.0 + complexity / 3.0).min(1.0)
            })
            .collect()
    }

    /// Calculate processing complexity
    fn calculate_processing_complexity(&self, sentences: &[String]) -> f64 {
        let avg_sentence_length = sentences
            .iter()
            .map(|s| s.split_whitespace().count())
            .sum::<usize>() as f64
            / sentences.len() as f64;

        (avg_sentence_length / 25.0).min(1.0)
    }

    /// Calculate working memory demands
    fn calculate_working_memory_demands(&self, paragraphs: &[String]) -> f64 {
        let avg_paragraph_length = paragraphs
            .iter()
            .map(|p| p.split_whitespace().count())
            .sum::<usize>() as f64
            / paragraphs.len() as f64;

        (avg_paragraph_length / 100.0).min(1.0)
    }

    /// Calculate comprehension ease
    fn calculate_comprehension_ease(&self, paragraphs: &[String], sentences: &[String]) -> f64 {
        let structural_clarity = self.calculate_structural_consistency(paragraphs);
        let flow_quality = self.calculate_information_flow(sentences);
        let marker_support = self.calculate_marker_density(paragraphs);

        (structural_clarity + flow_quality + marker_support) / 3.0
    }

    /// Calculate mental model coherence
    fn calculate_mental_model_coherence(&self, paragraphs: &[String]) -> f64 {
        self.calculate_paragraph_coherence(paragraphs)
    }

    /// Analyze genre-specific features
    fn analyze_genre_specific_features(
        &self,
        paragraphs: &[String],
    ) -> Result<GenreSpecificAnalysis, StructuralCoherenceError> {
        let structure_type = self.detect_structure_type(paragraphs);
        let conventions_adherence =
            self.calculate_structure_confidence(paragraphs, &structure_type);
        let genre_features = self.extract_structure_features(paragraphs);
        let structure_alignment = conventions_adherence;
        let recommendations = vec!["Enhance structural markers".to_string()];

        Ok(GenreSpecificAnalysis {
            conventions_adherence,
            genre_features,
            structure_alignment,
            recommendations,
        })
    }

    /// Assess structural quality
    fn assess_structural_quality(
        &self,
        paragraphs: &[String],
        sentences: &[String],
    ) -> Result<StructuralQualityAssessment, StructuralCoherenceError> {
        let overall_quality = self.calculate_overall_structural_quality(paragraphs, sentences);
        let quality_dimensions = self.calculate_quality_dimensions(paragraphs, sentences);
        let improvement_areas = self.identify_improvement_areas(paragraphs);
        let benchmark_comparison = 0.7; // Simplified benchmark

        Ok(StructuralQualityAssessment {
            overall_quality,
            quality_dimensions,
            improvement_areas,
            benchmark_comparison,
        })
    }

    /// Calculate overall structural quality
    fn calculate_overall_structural_quality(
        &self,
        paragraphs: &[String],
        sentences: &[String],
    ) -> f64 {
        let paragraph_coherence = self.calculate_paragraph_coherence(paragraphs);
        let organizational_coherence = self.calculate_organizational_coherence(paragraphs);
        let flow_quality = self.calculate_information_flow(sentences);

        (paragraph_coherence + organizational_coherence + flow_quality) / 3.0
    }

    /// Calculate quality dimensions
    fn calculate_quality_dimensions(
        &self,
        paragraphs: &[String],
        sentences: &[String],
    ) -> HashMap<String, f64> {
        let mut dimensions = HashMap::new();

        dimensions.insert(
            "coherence".to_string(),
            self.calculate_paragraph_coherence(paragraphs),
        );
        dimensions.insert(
            "organization".to_string(),
            self.calculate_organizational_coherence(paragraphs),
        );
        dimensions.insert(
            "flow".to_string(),
            self.calculate_information_flow(sentences),
        );
        dimensions.insert(
            "completeness".to_string(),
            self.calculate_structural_completeness(paragraphs),
        );

        dimensions
    }

    /// Identify improvement areas
    fn identify_improvement_areas(&self, paragraphs: &[String]) -> Vec<String> {
        let mut areas = Vec::new();

        if self.calculate_paragraph_coherence(paragraphs) < 0.6 {
            areas.push("paragraph_coherence".to_string());
        }

        if self.calculate_organizational_coherence(paragraphs) < 0.5 {
            areas.push("organizational_structure".to_string());
        }

        if self.extract_structural_markers(paragraphs).len() < paragraphs.len() / 2 {
            areas.push("structural_markers".to_string());
        }

        areas
    }

    /// Build structural markers
    fn build_structural_markers() -> HashMap<String, StructuralMarkerType> {
        let mut markers = HashMap::new();

        // Introduction markers
        for marker in [
            "introduction",
            "overview",
            "begin",
            "start",
            "purpose",
            "objective",
        ] {
            markers.insert(marker.to_string(), StructuralMarkerType::Introduction);
        }

        // Transition markers
        for marker in [
            "however",
            "furthermore",
            "moreover",
            "therefore",
            "consequently",
            "meanwhile",
        ] {
            markers.insert(marker.to_string(), StructuralMarkerType::Transition);
        }

        // Conclusion markers
        for marker in ["conclusion", "summary", "finally", "overall", "lastly"] {
            markers.insert(marker.to_string(), StructuralMarkerType::Conclusion);
        }

        // Enumeration markers
        for marker in ["first", "second", "third", "firstly", "secondly", "thirdly"] {
            markers.insert(marker.to_string(), StructuralMarkerType::Enumeration);
        }

        // Example markers
        for marker in ["example", "instance", "case", "illustration"] {
            markers.insert(marker.to_string(), StructuralMarkerType::Example);
        }

        // Section markers
        for marker in ["chapter", "section", "subsection", "part"] {
            markers.insert(marker.to_string(), StructuralMarkerType::Section);
        }

        markers
    }

    /// Build discourse patterns
    fn build_discourse_patterns() -> HashMap<String, DiscoursePatternType> {
        let mut patterns = HashMap::new();

        patterns.insert("problem".to_string(), DiscoursePatternType::ProblemSolution);
        patterns.insert("cause".to_string(), DiscoursePatternType::CauseEffect);
        patterns.insert("compare".to_string(), DiscoursePatternType::CompareContrast);
        patterns.insert("first".to_string(), DiscoursePatternType::Chronological);
        patterns.insert("definition".to_string(), DiscoursePatternType::Definition);

        patterns
    }

    /// Build rhetorical moves
    fn build_rhetorical_moves() -> HashMap<String, String> {
        let mut moves = HashMap::new();

        moves.insert("introduction".to_string(), "Establish context".to_string());
        moves.insert("evidence".to_string(), "Support claims".to_string());
        moves.insert("conclusion".to_string(), "Synthesize findings".to_string());

        moves
    }

    /// Build genre templates
    fn build_genre_templates() -> HashMap<DocumentStructureType, GenreTemplate> {
        let mut templates = HashMap::new();

        templates.insert(
            DocumentStructureType::Academic,
            GenreTemplate {
                expected_sections: vec![
                    "abstract".to_string(),
                    "introduction".to_string(),
                    "methodology".to_string(),
                    "results".to_string(),
                    "discussion".to_string(),
                    "conclusion".to_string(),
                ],
                section_order: vec![
                    "abstract".to_string(),
                    "introduction".to_string(),
                    "methodology".to_string(),
                    "results".to_string(),
                    "discussion".to_string(),
                    "conclusion".to_string(),
                ],
                required_components: vec!["introduction".to_string(), "conclusion".to_string()],
                typical_patterns: vec![
                    DiscoursePatternType::ProblemSolution,
                    DiscoursePatternType::CauseEffect,
                ],
            },
        );

        templates
    }
}

impl Default for StructuralCoherenceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple function for basic structural coherence analysis
pub fn calculate_structural_coherence_simple(text: &str) -> f64 {
    let analyzer = StructuralCoherenceAnalyzer::new();
    analyzer
        .analyze_structural_coherence(text)
        .map(|result| result.paragraph_coherence)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structural_coherence_analyzer_creation() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        assert_eq!(analyzer.config.structural_weight, 0.3);
        assert!(analyzer.config.enable_hierarchical_analysis);
        assert!(analyzer.config.enable_organizational_analysis);
    }

    #[test]
    fn test_basic_structural_coherence_analysis() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        let text = "Introduction: This paper discusses structural analysis.\n\nThe methodology involves systematic evaluation of text structure. We examine paragraph coherence and organizational patterns.\n\nResults show that structured texts have better coherence. The analysis reveals clear patterns in well-organized documents.\n\nConclusion: Structural coherence is essential for document quality.";

        let result = analyzer.analyze_structural_coherence(text);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.paragraph_coherence >= 0.0 && result.paragraph_coherence <= 1.0);
        assert!(result.organizational_coherence >= 0.0 && result.organizational_coherence <= 1.0);
        assert!(!result.structural_markers.is_empty());
    }

    #[test]
    fn test_paragraph_splitting() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        let text = "This is the first paragraph with sufficient length for analysis.\n\nThis is the second paragraph that also meets the minimum length requirement.";

        let paragraphs = analyzer.split_into_paragraphs(text).unwrap();
        assert_eq!(paragraphs.len(), 2);
        assert!(paragraphs[0].contains("first paragraph"));
        assert!(paragraphs[1].contains("second paragraph"));
    }

    #[test]
    fn test_paragraph_coherence_calculation() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        let paragraphs = vec![
            "This is a coherent paragraph about machine learning. Machine learning algorithms process data efficiently. Data processing improves with better algorithms.".to_string(),
            "Another paragraph discusses research methods. Research methods help validate findings. Good methods ensure reliable results.".to_string(),
        ];

        let coherence = analyzer.calculate_paragraph_coherence(&paragraphs);
        assert!(coherence >= 0.0 && coherence <= 1.0);
        assert!(coherence > 0.0); // Should be positive for coherent text
    }

    #[test]
    fn test_sentence_pair_coherence() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        let sent1 = "Machine learning algorithms analyze data patterns.";
        let sent2 = "Data analysis reveals important machine learning insights.";

        let coherence = analyzer.calculate_sentence_pair_coherence(sent1, sent2);
        assert!(coherence >= 0.0 && coherence <= 1.0);
        assert!(coherence > 0.3); // Should show some coherence due to shared terms
    }

    #[test]
    fn test_lexical_overlap_calculation() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        let sent1 = "The research method examines data patterns.";
        let sent2 = "Data patterns reveal research insights.";

        let overlap = analyzer.calculate_lexical_overlap(sent1, sent2);
        assert!(overlap >= 0.0 && overlap <= 1.0);
        assert!(overlap > 0.0); // Should have overlap due to shared words
    }

    #[test]
    fn test_content_word_extraction() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        let sentence = "The quick brown fox jumps over the lazy dog.";

        let content_words = analyzer.extract_content_words(sentence);
        assert!(content_words.contains(&"quick".to_string()));
        assert!(content_words.contains(&"brown".to_string()));
        assert!(content_words.contains(&"fox".to_string()));
        assert!(!content_words.contains(&"the".to_string())); // Function word should be filtered
    }

    #[test]
    fn test_structural_continuity_calculation() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        let sent1 = "First, we examine the data structure.";
        let sent2 = "Second, we analyze the results.";

        let continuity = analyzer.calculate_structural_continuity(sent1, sent2);
        assert!(continuity >= 0.0 && continuity <= 1.0);
    }

    #[test]
    fn test_organizational_coherence_calculation() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        let paragraphs = vec![
            "Introduction: This paper presents a new approach to text analysis.".to_string(),
            "The methodology involves computational analysis of document structure.".to_string(),
            "Results show significant improvements in coherence detection.".to_string(),
            "Conclusion: The proposed method effectively measures text coherence.".to_string(),
        ];

        let organizational_coherence = analyzer.calculate_organizational_coherence(&paragraphs);
        assert!(organizational_coherence >= 0.0 && organizational_coherence <= 1.0);
    }

    #[test]
    fn test_hierarchical_coherence_analysis() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        let paragraphs = vec![
            "Introduction: This section introduces the topic.".to_string(),
            "Chapter 1: Background information about the subject.".to_string(),
            "Section 1.1: Detailed analysis of the first aspect.".to_string(),
            "Conclusion: Summary of the main findings.".to_string(),
        ];

        let hierarchical_coherence = analyzer
            .calculate_hierarchical_coherence(&paragraphs)
            .unwrap();
        assert!(hierarchical_coherence >= 0.0 && hierarchical_coherence <= 1.0);
    }

    #[test]
    fn test_structural_level_inference() {
        let analyzer = StructuralCoherenceAnalyzer::new();

        let intro_level = analyzer.infer_structural_level("Introduction: This paper discusses...");
        let chapter_level = analyzer.infer_structural_level("Chapter 1: Background information...");
        let section_level = analyzer.infer_structural_level("Section 2.1: Detailed analysis...");

        assert_eq!(intro_level, 1);
        assert_eq!(chapter_level, 2);
        assert_eq!(section_level, 3);
    }

    #[test]
    fn test_structural_marker_extraction() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        let paragraphs = vec![
            "Introduction: This paper presents new findings.".to_string(),
            "Furthermore, the research shows important results.".to_string(),
            "Finally, we conclude with recommendations.".to_string(),
        ];

        let markers = analyzer.extract_structural_markers(&paragraphs);
        assert!(!markers.is_empty());
        // Should find markers like "furthermore", "finally"
    }

    #[test]
    fn test_coherence_pattern_identification() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        let paragraphs = vec![
            "Introduction: This study examines text coherence patterns.".to_string(),
            "The analysis reveals several important patterns in document structure.".to_string(),
            "Finally, we conclude that structural patterns improve readability.".to_string(),
        ];

        let patterns = analyzer.identify_coherence_patterns(&paragraphs);
        assert!(patterns.contains_key("introduction"));
        assert!(patterns.contains_key("conclusion"));
        assert!(patterns.contains_key("development"));

        // Introduction and conclusion should have positive scores
        assert!(patterns["introduction"] > 0.0);
        assert!(patterns["conclusion"] > 0.0);
    }

    #[test]
    fn test_introduction_quality_calculation() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        let intro_paragraph =
            "Introduction: This paper introduces a novel approach to text analysis.";
        let regular_paragraph = "The data shows interesting patterns in the results.";

        let intro_quality = analyzer.calculate_introduction_quality(intro_paragraph);
        let regular_quality = analyzer.calculate_introduction_quality(regular_paragraph);

        assert!(intro_quality > regular_quality);
        assert!(intro_quality > 0.0);
    }

    #[test]
    fn test_conclusion_quality_calculation() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        let conclusion_paragraph =
            "In conclusion, this study demonstrates the effectiveness of the proposed method.";
        let regular_paragraph = "The analysis shows various patterns in the data.";

        let conclusion_quality = analyzer.calculate_conclusion_quality(conclusion_paragraph);
        let regular_quality = analyzer.calculate_conclusion_quality(regular_paragraph);

        assert!(conclusion_quality > regular_quality);
        assert!(conclusion_quality > 0.0);
    }

    #[test]
    fn test_structural_consistency_calculation() {
        let analyzer = StructuralCoherenceAnalyzer::new();

        // Consistent paragraph lengths
        let consistent_paragraphs = vec![
            "This is a paragraph with approximately the same length as others in this test."
                .to_string(),
            "Another paragraph that maintains similar length to ensure structural consistency."
                .to_string(),
            "The third paragraph also follows the same pattern of consistent length and structure."
                .to_string(),
        ];

        // Inconsistent paragraph lengths
        let inconsistent_paragraphs = vec![
            "Short.".to_string(),
            "This is a much longer paragraph that contains significantly more words and content than the previous paragraph, creating inconsistency.".to_string(),
            "Medium length paragraph here.".to_string(),
        ];

        let consistent_score = analyzer.calculate_structural_consistency(&consistent_paragraphs);
        let inconsistent_score =
            analyzer.calculate_structural_consistency(&inconsistent_paragraphs);

        assert!(consistent_score > inconsistent_score);
        assert!(consistent_score >= 0.0 && consistent_score <= 1.0);
        assert!(inconsistent_score >= 0.0 && inconsistent_score <= 1.0);
    }

    #[test]
    fn test_section_boundary_detection() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        let paragraphs = vec![
            "Introduction: This paper presents new research.".to_string(),
            "Background information about the subject matter.".to_string(),
            "Chapter 1: Methodology and approach.".to_string(),
            "Section 1.1: Data collection procedures.".to_string(),
            "Conclusion: Summary of findings.".to_string(),
        ];

        let boundaries = analyzer.detect_section_boundaries(&paragraphs);
        assert!(!boundaries.is_empty());

        // Should detect boundaries at structural markers
        assert!(boundaries
            .iter()
            .any(|b| b.boundary_type.contains("introduction")));
    }

    #[test]
    fn test_discourse_pattern_detection() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        let paragraphs = vec![
            "The problem with current methods is their limited accuracy.".to_string(),
            "Our solution addresses this problem through novel algorithms.".to_string(),
            "First, we collect data. Then, we process it. Finally, we analyze results.".to_string(),
        ];

        let patterns = analyzer.detect_discourse_patterns(&paragraphs);
        assert!(!patterns.is_empty());

        // Should detect problem-solution and chronological patterns
        let pattern_types: HashSet<_> = patterns.iter().map(|p| &p.pattern_type).collect();
        assert!(
            pattern_types.contains(&DiscoursePatternType::ProblemSolution)
                || pattern_types.contains(&DiscoursePatternType::Chronological)
        );
    }

    #[test]
    fn test_document_structure_detection() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        let paragraphs = vec![
            "Abstract: This paper presents novel research findings.".to_string(),
            "Introduction: The research addresses important questions.".to_string(),
            "Methodology: We used computational analysis methods.".to_string(),
            "Results: The analysis revealed significant patterns.".to_string(),
            "Conclusion: The findings have important implications.".to_string(),
        ];

        let structure_type = analyzer.detect_structure_type(&paragraphs);
        assert_eq!(structure_type, DocumentStructureType::Academic);
    }

    #[test]
    fn test_advanced_analysis() {
        let analyzer = StructuralCoherenceAnalyzer::with_config(StructuralCoherenceConfig {
            use_advanced_analysis: true,
            enable_hierarchical_analysis: true,
            analyze_rhetorical_structure: true,
            ..StructuralCoherenceConfig::default()
        });

        let text = "Introduction: This research examines document structure.\n\nMethodology: We analyze structural patterns systematically.\n\nResults: The analysis reveals clear organizational patterns.\n\nConclusion: Structural analysis improves document quality.";

        let result = analyzer.analyze_structural_coherence(text).unwrap();
        assert!(result.advanced_analysis.is_some());

        let advanced = result.advanced_analysis.unwrap();
        assert!(advanced.quality_assessment.overall_quality >= 0.0);
        assert!(!advanced.rhetorical_structure.rhetorical_moves.is_empty());
    }

    #[test]
    fn test_empty_text_handling() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        let result = analyzer.analyze_structural_coherence("");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            StructuralCoherenceError::EmptyText
        ));
    }

    #[test]
    fn test_simple_function() {
        let coherence = calculate_structural_coherence_simple("Introduction paragraph here.\n\nMain content with analysis.\n\nConclusion paragraph here.");
        assert!(coherence >= 0.0 && coherence <= 1.0);
    }

    #[test]
    fn test_hierarchical_level_determination() {
        let analyzer = StructuralCoherenceAnalyzer::new();

        let chapter_level =
            analyzer.determine_hierarchical_level("Chapter 1: Introduction to the subject");
        let section_level = analyzer.determine_hierarchical_level("Section 2.1: Detailed analysis");
        let paragraph_level = analyzer.determine_hierarchical_level(
            "This is a regular paragraph without structural markers.",
        );

        assert_eq!(chapter_level, HierarchicalLevel::Chapter);
        assert_eq!(section_level, HierarchicalLevel::Section);
        assert_eq!(paragraph_level, HierarchicalLevel::Paragraph);
    }

    #[test]
    fn test_word_similarity_calculation() {
        let analyzer = StructuralCoherenceAnalyzer::new();

        // Test identical words
        assert_eq!(analyzer.calculate_word_similarity("test", "test"), 1.0);

        // Test completely different words
        let similarity = analyzer.calculate_word_similarity("cat", "dog");
        assert!(similarity >= 0.0 && similarity <= 1.0);

        // Test similar words
        let similarity = analyzer.calculate_word_similarity("analyze", "analysis");
        assert!(similarity > 0.0);
    }

    #[test]
    fn test_global_coherence_metrics() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        let sentences = vec![
            "The research focuses on text analysis.".to_string(),
            "Text analysis involves examining document structure.".to_string(),
            "Document structure affects readability and comprehension.".to_string(),
        ];
        let paragraphs = vec![
            "Introduction: This research examines text analysis methods.".to_string(),
            "The main focus is on structural coherence in documents.".to_string(),
            "Conclusion: Structural analysis improves document quality.".to_string(),
        ];

        let global_metrics = analyzer.calculate_global_coherence_metrics(&sentences, &paragraphs);

        assert!(
            global_metrics.beginning_end_coherence >= 0.0
                && global_metrics.beginning_end_coherence <= 1.0
        );
        assert!(
            global_metrics.thematic_consistency >= 0.0
                && global_metrics.thematic_consistency <= 1.0
        );
        assert!(global_metrics.information_flow >= 0.0 && global_metrics.information_flow <= 1.0);
        assert!(global_metrics.document_unity >= 0.0 && global_metrics.document_unity <= 1.0);
    }

    #[test]
    fn test_document_completeness_analysis() {
        let analyzer = StructuralCoherenceAnalyzer::new();

        // Complete document
        let complete_paragraphs = vec![
            "Introduction: This paper introduces the research topic.".to_string(),
            "The main analysis covers several important aspects.".to_string(),
            "In conclusion, the research demonstrates significant findings.".to_string(),
        ];

        let completeness = analyzer.analyze_document_completeness(&complete_paragraphs);
        assert!(completeness.has_introduction);
        assert!(completeness.has_development);
        assert!(completeness.has_conclusion);
        assert!(completeness.completeness_score > 0.8);

        // Incomplete document (missing conclusion)
        let incomplete_paragraphs = vec![
            "This is the opening paragraph.".to_string(),
            "Here is some analysis content.".to_string(),
        ];

        let incomplete_completeness =
            analyzer.analyze_document_completeness(&incomplete_paragraphs);
        assert!(incomplete_completeness.completeness_score < 1.0);
        assert!(incomplete_completeness
            .missing_components
            .contains(&"conclusion".to_string()));
    }

    #[test]
    fn test_complexity_measures() {
        let analyzer = StructuralCoherenceAnalyzer::new();
        let paragraphs = vec![
            "Introduction: This chapter introduces complex analysis methods.".to_string(),
            "Section 1: Basic concepts and definitions.".to_string(),
            "Subsection 1.1: Detailed technical specifications.".to_string(),
            "Furthermore, advanced techniques require careful consideration.".to_string(),
            "Finally, we conclude with comprehensive recommendations.".to_string(),
        ];
        let sentences = vec![
            "This is a sentence.".to_string(),
            "Another sentence follows.".to_string(),
            "Complex sentences require more processing effort.".to_string(),
        ];

        let complexity = analyzer.calculate_complexity_measures(&paragraphs, &sentences);

        assert!(complexity.hierarchical_complexity >= 0.0);
        assert!(complexity.organizational_complexity >= 0.0);
        assert!(complexity.cognitive_load >= 0.0 && complexity.cognitive_load <= 1.0);
        assert!(complexity.structural_diversity >= 0.0);
    }
}
