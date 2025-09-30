//! Configuration management for structural coherence analysis
//!
//! This module provides comprehensive configuration options for all aspects of
//! structural coherence analysis, including hierarchical analysis, discourse patterns,
//! structural markers, and advanced analysis features.

use serde::{Deserialize, Serialize};

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

impl Default for DocumentStructureType {
    fn default() -> Self {
        DocumentStructureType::Unknown
    }
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

impl Default for HierarchicalLevel {
    fn default() -> Self {
        HierarchicalLevel::Paragraph
    }
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

impl Default for DiscoursePatternType {
    fn default() -> Self {
        DiscoursePatternType::Mixed
    }
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

impl Default for StructuralMarkerType {
    fn default() -> Self {
        StructuralMarkerType::Transition
    }
}

/// Configuration for structural coherence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralCoherenceConfig {
    /// General analysis configuration
    pub general: GeneralAnalysisConfig,
    /// Hierarchical analysis configuration
    pub hierarchical: HierarchicalAnalysisConfig,
    /// Discourse pattern analysis configuration
    pub discourse: DiscoursePatternConfig,
    /// Structural marker analysis configuration
    pub markers: StructuralMarkerConfig,
    /// Boundary detection configuration
    pub boundaries: BoundaryDetectionConfig,
    /// Coherence calculation configuration
    pub coherence: CoherenceCalculationConfig,
    /// Advanced analysis configuration
    pub advanced: AdvancedAnalysisConfig,
}

/// General analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralAnalysisConfig {
    /// Weight for structural analysis
    pub structural_weight: f64,
    /// Minimum paragraph length for analysis
    pub min_paragraph_length: usize,
    /// Document structure type (if known)
    pub expected_structure_type: Option<DocumentStructureType>,
    /// Structural consistency sensitivity
    pub consistency_sensitivity: f64,
    /// Enable comprehensive analysis
    pub use_comprehensive_analysis: bool,
}

/// Hierarchical analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalAnalysisConfig {
    /// Enable hierarchical analysis
    pub enable_analysis: bool,
    /// Maximum hierarchical depth to analyze
    pub max_depth: usize,
    /// Hierarchical balance threshold
    pub balance_threshold: f64,
    /// Level transition sensitivity
    pub transition_sensitivity: f64,
    /// Enable structural tree generation
    pub generate_structural_tree: bool,
    /// Minimum confidence for level detection
    pub min_level_confidence: f64,
}

/// Discourse pattern analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoursePatternConfig {
    /// Enable discourse pattern detection
    pub enable_detection: bool,
    /// Pattern detection sensitivity
    pub detection_sensitivity: f64,
    /// Minimum pattern strength threshold
    pub min_pattern_strength: f64,
    /// Enable pattern transition analysis
    pub analyze_transitions: bool,
    /// Pattern completeness threshold
    pub completeness_threshold: f64,
    /// Maximum patterns per document
    pub max_patterns_per_document: usize,
}

/// Structural marker analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralMarkerConfig {
    /// Enable structural marker analysis
    pub enable_analysis: bool,
    /// Marker detection sensitivity
    pub detection_sensitivity: f64,
    /// Enable marker effectiveness analysis
    pub analyze_effectiveness: bool,
    /// Marker density threshold
    pub density_threshold: f64,
    /// Enable missing marker detection
    pub detect_missing_markers: bool,
    /// Custom marker patterns
    pub custom_markers: Vec<String>,
}

/// Boundary detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryDetectionConfig {
    /// Enable section boundary detection
    pub enable_detection: bool,
    /// Boundary strength threshold
    pub strength_threshold: f64,
    /// Enable automatic section title extraction
    pub extract_titles: bool,
    /// Title extraction confidence threshold
    pub title_confidence_threshold: f64,
    /// Maximum boundary detection depth
    pub max_detection_depth: usize,
}

/// Coherence calculation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceCalculationConfig {
    /// Paragraph coherence threshold
    pub paragraph_threshold: f64,
    /// Section coherence threshold
    pub section_threshold: f64,
    /// Transition quality threshold
    pub transition_threshold: f64,
    /// Enable semantic continuity analysis
    pub enable_semantic_continuity: bool,
    /// Lexical overlap weight
    pub lexical_overlap_weight: f64,
    /// Structural continuity weight
    pub structural_continuity_weight: f64,
    /// Semantic continuity weight
    pub semantic_continuity_weight: f64,
}

/// Advanced analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedAnalysisConfig {
    /// Enable advanced analysis features
    pub enable_advanced_analysis: bool,
    /// Enable global coherence analysis
    pub analyze_global_coherence: bool,
    /// Enable rhetorical structure analysis
    pub analyze_rhetorical_structure: bool,
    /// Enable document completeness analysis
    pub analyze_completeness: bool,
    /// Enable complexity analysis
    pub analyze_complexity: bool,
    /// Advanced analysis depth
    pub analysis_depth: usize,
}

impl Default for StructuralCoherenceConfig {
    fn default() -> Self {
        Self {
            general: GeneralAnalysisConfig::default(),
            hierarchical: HierarchicalAnalysisConfig::default(),
            discourse: DiscoursePatternConfig::default(),
            markers: StructuralMarkerConfig::default(),
            boundaries: BoundaryDetectionConfig::default(),
            coherence: CoherenceCalculationConfig::default(),
            advanced: AdvancedAnalysisConfig::default(),
        }
    }
}

impl Default for GeneralAnalysisConfig {
    fn default() -> Self {
        Self {
            structural_weight: 0.3,
            min_paragraph_length: 50,
            expected_structure_type: None,
            consistency_sensitivity: 0.7,
            use_comprehensive_analysis: true,
        }
    }
}

impl Default for HierarchicalAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_analysis: true,
            max_depth: 6,
            balance_threshold: 0.5,
            transition_sensitivity: 0.7,
            generate_structural_tree: true,
            min_level_confidence: 0.6,
        }
    }
}

impl Default for DiscoursePatternConfig {
    fn default() -> Self {
        Self {
            enable_detection: true,
            detection_sensitivity: 0.7,
            min_pattern_strength: 0.5,
            analyze_transitions: true,
            completeness_threshold: 0.6,
            max_patterns_per_document: 10,
        }
    }
}

impl Default for StructuralMarkerConfig {
    fn default() -> Self {
        Self {
            enable_analysis: true,
            detection_sensitivity: 0.6,
            analyze_effectiveness: true,
            density_threshold: 0.1,
            detect_missing_markers: true,
            custom_markers: Vec::new(),
        }
    }
}

impl Default for BoundaryDetectionConfig {
    fn default() -> Self {
        Self {
            enable_detection: true,
            strength_threshold: 0.5,
            extract_titles: true,
            title_confidence_threshold: 0.7,
            max_detection_depth: 4,
        }
    }
}

impl Default for CoherenceCalculationConfig {
    fn default() -> Self {
        Self {
            paragraph_threshold: 0.5,
            section_threshold: 0.6,
            transition_threshold: 0.4,
            enable_semantic_continuity: true,
            lexical_overlap_weight: 0.4,
            structural_continuity_weight: 0.3,
            semantic_continuity_weight: 0.3,
        }
    }
}

impl Default for AdvancedAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_advanced_analysis: true,
            analyze_global_coherence: true,
            analyze_rhetorical_structure: true,
            analyze_completeness: true,
            analyze_complexity: true,
            analysis_depth: 3,
        }
    }
}

/// Configuration builder for easier setup
pub struct StructuralCoherenceConfigBuilder {
    config: StructuralCoherenceConfig,
}

impl StructuralCoherenceConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: StructuralCoherenceConfig::default(),
        }
    }

    /// Set structural weight
    pub fn structural_weight(mut self, weight: f64) -> Self {
        self.config.general.structural_weight = weight;
        self
    }

    /// Set minimum paragraph length
    pub fn min_paragraph_length(mut self, length: usize) -> Self {
        self.config.general.min_paragraph_length = length;
        self
    }

    /// Set expected document structure type
    pub fn expected_structure_type(mut self, structure_type: DocumentStructureType) -> Self {
        self.config.general.expected_structure_type = Some(structure_type);
        self
    }

    /// Enable/disable hierarchical analysis
    pub fn enable_hierarchical_analysis(mut self, enable: bool) -> Self {
        self.config.hierarchical.enable_analysis = enable;
        self
    }

    /// Set hierarchical analysis depth
    pub fn hierarchical_depth(mut self, depth: usize) -> Self {
        self.config.hierarchical.max_depth = depth;
        self
    }

    /// Enable/disable discourse pattern detection
    pub fn enable_discourse_patterns(mut self, enable: bool) -> Self {
        self.config.discourse.enable_detection = enable;
        self
    }

    /// Set discourse pattern sensitivity
    pub fn discourse_sensitivity(mut self, sensitivity: f64) -> Self {
        self.config.discourse.detection_sensitivity = sensitivity;
        self
    }

    /// Enable/disable structural marker analysis
    pub fn enable_structural_markers(mut self, enable: bool) -> Self {
        self.config.markers.enable_analysis = enable;
        self
    }

    /// Enable/disable boundary detection
    pub fn enable_boundary_detection(mut self, enable: bool) -> Self {
        self.config.boundaries.enable_detection = enable;
        self
    }

    /// Set paragraph coherence threshold
    pub fn paragraph_threshold(mut self, threshold: f64) -> Self {
        self.config.coherence.paragraph_threshold = threshold;
        self
    }

    /// Enable/disable advanced analysis
    pub fn enable_advanced_analysis(mut self, enable: bool) -> Self {
        self.config.advanced.enable_advanced_analysis = enable;
        self
    }

    /// Enable comprehensive analysis
    pub fn use_comprehensive_analysis(mut self, use_comprehensive: bool) -> Self {
        self.config.general.use_comprehensive_analysis = use_comprehensive;
        self
    }

    /// Build the configuration
    pub fn build(self) -> StructuralCoherenceConfig {
        self.config
    }
}

impl Default for StructuralCoherenceConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience methods for common configurations
impl StructuralCoherenceConfig {
    /// Create a builder for this configuration
    pub fn builder() -> StructuralCoherenceConfigBuilder {
        StructuralCoherenceConfigBuilder::new()
    }

    /// Create configuration optimized for academic papers
    pub fn for_academic_paper() -> Self {
        Self::builder()
            .expected_structure_type(DocumentStructureType::Academic)
            .enable_hierarchical_analysis(true)
            .hierarchical_depth(6)
            .enable_discourse_patterns(true)
            .discourse_sensitivity(0.8)
            .enable_structural_markers(true)
            .enable_boundary_detection(true)
            .enable_advanced_analysis(true)
            .build()
    }

    /// Create configuration optimized for technical documents
    pub fn for_technical_document() -> Self {
        Self::builder()
            .expected_structure_type(DocumentStructureType::Technical)
            .enable_hierarchical_analysis(true)
            .hierarchical_depth(8)
            .enable_discourse_patterns(true)
            .enable_structural_markers(true)
            .enable_boundary_detection(true)
            .paragraph_threshold(0.6)
            .build()
    }

    /// Create configuration optimized for narratives
    pub fn for_narrative() -> Self {
        Self::builder()
            .expected_structure_type(DocumentStructureType::Narrative)
            .enable_hierarchical_analysis(false)
            .enable_discourse_patterns(true)
            .discourse_sensitivity(0.6)
            .enable_structural_markers(false)
            .paragraph_threshold(0.4)
            .build()
    }

    /// Create lightweight configuration for basic analysis
    pub fn lightweight() -> Self {
        Self::builder()
            .enable_hierarchical_analysis(false)
            .enable_discourse_patterns(false)
            .enable_structural_markers(false)
            .enable_boundary_detection(false)
            .enable_advanced_analysis(false)
            .use_comprehensive_analysis(false)
            .build()
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.general.structural_weight < 0.0 || self.general.structural_weight > 1.0 {
            return Err("Structural weight must be between 0.0 and 1.0".to_string());
        }

        if self.general.min_paragraph_length == 0 {
            return Err("Minimum paragraph length must be greater than 0".to_string());
        }

        if self.hierarchical.max_depth == 0 {
            return Err("Maximum hierarchical depth must be greater than 0".to_string());
        }

        if self.coherence.paragraph_threshold < 0.0 || self.coherence.paragraph_threshold > 1.0 {
            return Err("Paragraph threshold must be between 0.0 and 1.0".to_string());
        }

        // Validate weight sum for coherence calculation
        let weight_sum = self.coherence.lexical_overlap_weight
            + self.coherence.structural_continuity_weight
            + self.coherence.semantic_continuity_weight;

        if (weight_sum - 1.0).abs() > 0.01 {
            return Err("Coherence calculation weights should sum to 1.0".to_string());
        }

        Ok(())
    }
}
