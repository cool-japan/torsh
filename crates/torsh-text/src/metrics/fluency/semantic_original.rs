//! Semantic Fluency Analysis
//!
//! This module provides comprehensive semantic fluency evaluation including
//! semantic coherence, meaning preservation, conceptual clarity, context sensitivity,
//! and semantic relation analysis.

use scirs2_core::ndarray::{array, Array1, Array2};
use std::collections::{HashMap, HashSet};

/// Configuration for semantic analysis
#[derive(Debug, Clone)]
pub struct SemanticConfig {
    /// Weight for semantic coherence
    pub coherence_weight: f64,
    /// Weight for meaning preservation
    pub preservation_weight: f64,
    /// Weight for conceptual clarity
    pub clarity_weight: f64,
    /// Weight for semantic appropriateness
    pub appropriateness_weight: f64,
    /// Weight for context sensitivity
    pub context_weight: f64,
    /// Threshold for semantic similarity
    pub similarity_threshold: f64,
    /// Enable advanced semantic relation analysis
    pub enable_advanced_relations: bool,
    /// Context window size for semantic analysis
    pub context_window: usize,
    /// Enable discourse coherence analysis
    pub enable_discourse_analysis: bool,
    /// Semantic field coverage requirement
    pub min_semantic_coverage: f64,
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            coherence_weight: 0.25,
            preservation_weight: 0.20,
            clarity_weight: 0.20,
            appropriateness_weight: 0.15,
            context_weight: 0.20,
            similarity_threshold: 0.5,
            enable_advanced_relations: true,
            context_window: 3,
            enable_discourse_analysis: true,
            min_semantic_coverage: 0.6,
        }
    }
}

/// Results of semantic fluency analysis
#[derive(Debug, Clone)]
pub struct SemanticFluencyResult {
    /// Overall semantic coherence score
    pub semantic_coherence: f64,
    /// Meaning preservation across text
    pub meaning_preservation: f64,
    /// Conceptual clarity measure
    pub conceptual_clarity: f64,
    /// Semantic appropriateness score
    pub semantic_appropriateness: f64,
    /// Context sensitivity measure
    pub context_sensitivity: f64,
    /// Semantic density (concepts per unit)
    pub semantic_density: f64,
    /// Ambiguity score (clarity of meaning)
    pub ambiguity_score: f64,
    /// Semantic relation analysis
    pub semantic_relations: HashMap<String, f64>,
    /// Advanced semantic metrics
    pub advanced_metrics: AdvancedSemanticMetrics,
    /// Discourse-level analysis
    pub discourse_analysis: DiscourseAnalysis,
}

/// Advanced semantic metrics
#[derive(Debug, Clone)]
pub struct AdvancedSemanticMetrics {
    /// Semantic field coverage analysis
    pub field_coverage: SemanticFieldCoverage,
    /// Conceptual complexity analysis
    pub conceptual_complexity: ConceptualComplexity,
    /// Semantic consistency across text
    pub consistency_analysis: ConsistencyAnalysis,
    /// Topic coherence measures
    pub topic_coherence: TopicCoherence,
    /// Semantic role analysis
    pub semantic_roles: SemanticRoleAnalysis,
    /// Metaphor and figurative language detection
    pub figurative_language: FigurativeLanguageAnalysis,
}

/// Semantic field coverage analysis
#[derive(Debug, Clone)]
pub struct SemanticFieldCoverage {
    /// Coverage percentage for each semantic field
    pub field_coverage_map: HashMap<String, f64>,
    /// Overall field diversity
    pub field_diversity: f64,
    /// Dominant semantic fields
    pub dominant_fields: Vec<String>,
    /// Field transition patterns
    pub field_transitions: HashMap<(String, String), usize>,
    /// Coverage consistency score
    pub coverage_consistency: f64,
}

/// Conceptual complexity analysis
#[derive(Debug, Clone)]
pub struct ConceptualComplexity {
    /// Abstract vs concrete concept ratio
    pub abstraction_ratio: f64,
    /// Conceptual depth (hierarchical levels)
    pub conceptual_depth: f64,
    /// Concept interconnectedness
    pub interconnectedness: f64,
    /// Conceptual novelty measure
    pub novelty_score: f64,
    /// Complexity distribution
    pub complexity_distribution: ComplexityDistribution,
}

/// Complexity distribution metrics
#[derive(Debug, Clone)]
pub struct ComplexityDistribution {
    /// Simple concept percentage
    pub simple_concepts: f64,
    /// Moderate complexity percentage
    pub moderate_concepts: f64,
    /// Complex concept percentage
    pub complex_concepts: f64,
    /// Complexity variance across text
    pub complexity_variance: f64,
}

/// Consistency analysis results
#[derive(Debug, Clone)]
pub struct ConsistencyAnalysis {
    /// Terminological consistency
    pub terminological_consistency: f64,
    /// Conceptual consistency
    pub conceptual_consistency: f64,
    /// Semantic frame consistency
    pub frame_consistency: f64,
    /// Inconsistency patterns
    pub inconsistency_patterns: Vec<InconsistencyPattern>,
}

/// Inconsistency pattern detection
#[derive(Debug, Clone)]
pub struct InconsistencyPattern {
    /// Pattern type
    pub pattern_type: InconsistencyType,
    /// Pattern description
    pub description: String,
    /// Severity level
    pub severity: f64,
    /// Locations in text
    pub locations: Vec<usize>,
    /// Suggested resolution
    pub resolution_suggestion: String,
}

/// Types of semantic inconsistencies
#[derive(Debug, Clone)]
pub enum InconsistencyType {
    /// Contradictory statements
    ContradictoryStatements,
    /// Inconsistent terminology
    InconsistentTerminology,
    /// Semantic field mixing
    SemanticFieldMixing,
    /// Conceptual drift
    ConceptualDrift,
    /// Reference inconsistency
    ReferenceInconsistency,
}

/// Topic coherence analysis
#[derive(Debug, Clone)]
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
}

/// Topic representation
#[derive(Debug, Clone)]
pub struct Topic {
    /// Topic name/identifier
    pub name: String,
    /// Topic keywords
    pub keywords: Vec<String>,
    /// Topic weight in text
    pub weight: f64,
    /// Topic coherence score
    pub coherence_score: f64,
}

/// Semantic role analysis
#[derive(Debug, Clone)]
pub struct SemanticRoleAnalysis {
    /// Agent-action-patient patterns
    pub role_patterns: HashMap<String, usize>,
    /// Semantic role diversity
    pub role_diversity: f64,
    /// Role assignment accuracy
    pub assignment_accuracy: f64,
    /// Complex role structures
    pub complex_structures: Vec<ComplexRoleStructure>,
}

/// Complex semantic role structure
#[derive(Debug, Clone)]
pub struct ComplexRoleStructure {
    /// Structure type
    pub structure_type: String,
    /// Components involved
    pub components: Vec<String>,
    /// Complexity score
    pub complexity_score: f64,
    /// Semantic clarity
    pub clarity_score: f64,
}

/// Figurative language analysis
#[derive(Debug, Clone)]
pub struct FigurativeLanguageAnalysis {
    /// Metaphor detection and analysis
    pub metaphors: Vec<MetaphorInstance>,
    /// Figurative language density
    pub figurative_density: f64,
    /// Conceptual metaphor patterns
    pub conceptual_patterns: HashMap<String, f64>,
    /// Figurative language appropriateness
    pub appropriateness_score: f64,
}

/// Metaphor instance
#[derive(Debug, Clone)]
pub struct MetaphorInstance {
    /// Source domain
    pub source_domain: String,
    /// Target domain
    pub target_domain: String,
    /// Metaphor text
    pub text: String,
    /// Conceptual mapping strength
    pub mapping_strength: f64,
    /// Novelty score
    pub novelty: f64,
}

/// Discourse-level semantic analysis
#[derive(Debug, Clone)]
pub struct DiscourseAnalysis {
    /// Global coherence score
    pub global_coherence: f64,
    /// Local coherence measures
    pub local_coherence: Vec<f64>,
    /// Coherence relations between sentences
    pub coherence_relations: Vec<CoherenceRelation>,
    /// Discourse markers effectiveness
    pub discourse_markers: DiscourseMarkerAnalysis,
    /// Information structure analysis
    pub information_structure: InformationStructure,
}

/// Coherence relation between text segments
#[derive(Debug, Clone)]
pub struct CoherenceRelation {
    /// Relation type
    pub relation_type: CoherenceRelationType,
    /// Source segment index
    pub source_segment: usize,
    /// Target segment index
    pub target_segment: usize,
    /// Relation strength
    pub strength: f64,
    /// Explicit markers present
    pub explicit_markers: Vec<String>,
}

/// Types of coherence relations
#[derive(Debug, Clone)]
pub enum CoherenceRelationType {
    /// Causal relationship
    Causal,
    /// Temporal relationship
    Temporal,
    /// Contrastive relationship
    Contrastive,
    /// Additive relationship
    Additive,
    /// Elaboration relationship
    Elaboration,
    /// Background information
    Background,
}

/// Discourse marker analysis
#[derive(Debug, Clone)]
pub struct DiscourseMarkerAnalysis {
    /// Marker usage frequency
    pub marker_frequency: HashMap<String, usize>,
    /// Marker appropriateness score
    pub appropriateness: f64,
    /// Marker diversity
    pub diversity: f64,
    /// Missing markers (where needed)
    pub missing_markers: Vec<String>,
}

/// Information structure analysis
#[derive(Debug, Clone)]
pub struct InformationStructure {
    /// Given-new information balance
    pub given_new_balance: f64,
    /// Topic continuity score
    pub topic_continuity: f64,
    /// Information density per segment
    pub information_density: Vec<f64>,
    /// Focus structure analysis
    pub focus_structure: FocusStructure,
}

/// Focus structure in discourse
#[derive(Debug, Clone)]
pub struct FocusStructure {
    /// Main focus points
    pub focus_points: Vec<FocusPoint>,
    /// Focus transition smoothness
    pub transition_smoothness: f64,
    /// Focus hierarchy depth
    pub hierarchy_depth: usize,
}

/// Individual focus point
#[derive(Debug, Clone)]
pub struct FocusPoint {
    /// Focus content
    pub content: String,
    /// Focus strength
    pub strength: f64,
    /// Segment location
    pub location: usize,
    /// Focus type
    pub focus_type: FocusType,
}

/// Types of discourse focus
#[derive(Debug, Clone)]
pub enum FocusType {
    /// New information introduction
    NewInformation,
    /// Contrastive focus
    Contrastive,
    /// Emphatic focus
    Emphatic,
    /// Topic shift
    TopicShift,
}

/// Semantic analyzer for fluency evaluation
pub struct SemanticAnalyzer {
    config: SemanticConfig,
    semantic_networks: HashMap<String, HashSet<String>>,
    conceptual_hierarchies: HashMap<String, Vec<String>>,
    semantic_relations: SemanticRelationDatabase,
    discourse_markers: HashMap<String, CoherenceRelationType>,
    figurative_patterns: HashMap<String, Vec<String>>,
}

/// Database of semantic relations
#[derive(Debug, Clone)]
pub struct SemanticRelationDatabase {
    /// Synonymy relations
    pub synonyms: HashMap<String, HashSet<String>>,
    /// Antonymy relations
    pub antonyms: HashMap<String, HashSet<String>>,
    /// Hyponymy relations (is-a)
    pub hyponyms: HashMap<String, HashSet<String>>,
    /// Meronymy relations (part-of)
    pub meronyms: HashMap<String, HashSet<String>>,
    /// Semantic similarity scores
    pub similarity_matrix: HashMap<(String, String), f64>,
}

impl SemanticAnalyzer {
    /// Create a new semantic analyzer
    pub fn new(config: SemanticConfig) -> Self {
        let semantic_networks = Self::build_semantic_networks();
        let conceptual_hierarchies = Self::build_conceptual_hierarchies();
        let semantic_relations = Self::build_semantic_relations();
        let discourse_markers = Self::build_discourse_markers();
        let figurative_patterns = Self::build_figurative_patterns();

        Self {
            config,
            semantic_networks,
            conceptual_hierarchies,
            semantic_relations,
            discourse_markers,
            figurative_patterns,
        }
    }

    /// Create analyzer with default configuration
    pub fn with_default_config() -> Self {
        Self::new(SemanticConfig::default())
    }

    /// Build semantic networks
    fn build_semantic_networks() -> HashMap<String, HashSet<String>> {
        let mut networks = HashMap::new();

        // Animals semantic field
        networks.insert(
            "animals".to_string(),
            [
                "dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep", "goat", "chicken",
                "duck", "rabbit", "mouse", "elephant", "lion", "tiger",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );

        // Emotions semantic field
        networks.insert(
            "emotions".to_string(),
            [
                "happy",
                "sad",
                "angry",
                "excited",
                "nervous",
                "calm",
                "worried",
                "joyful",
                "fearful",
                "surprised",
                "disgusted",
                "proud",
                "ashamed",
                "guilty",
                "content",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );

        // Colors semantic field
        networks.insert(
            "colors".to_string(),
            [
                "red",
                "blue",
                "green",
                "yellow",
                "orange",
                "purple",
                "pink",
                "brown",
                "black",
                "white",
                "gray",
                "violet",
                "turquoise",
                "maroon",
                "navy",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );

        // Transportation semantic field
        networks.insert(
            "transportation".to_string(),
            [
                "car",
                "bus",
                "train",
                "plane",
                "bicycle",
                "motorcycle",
                "truck",
                "boat",
                "ship",
                "helicopter",
                "subway",
                "taxi",
                "scooter",
                "van",
                "limousine",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );

        // Technology semantic field
        networks.insert(
            "technology".to_string(),
            [
                "computer",
                "phone",
                "internet",
                "software",
                "hardware",
                "database",
                "algorithm",
                "programming",
                "digital",
                "artificial",
                "intelligence",
                "robot",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );

        networks
    }

    /// Build conceptual hierarchies
    fn build_conceptual_hierarchies() -> HashMap<String, Vec<String>> {
        let mut hierarchies = HashMap::new();

        // Animal hierarchy
        hierarchies.insert(
            "animal".to_string(),
            vec![
                "mammal".to_string(),
                "bird".to_string(),
                "fish".to_string(),
                "reptile".to_string(),
            ],
        );
        hierarchies.insert(
            "mammal".to_string(),
            vec![
                "dog".to_string(),
                "cat".to_string(),
                "horse".to_string(),
                "cow".to_string(),
            ],
        );

        // Vehicle hierarchy
        hierarchies.insert(
            "vehicle".to_string(),
            vec![
                "car".to_string(),
                "truck".to_string(),
                "motorcycle".to_string(),
                "bicycle".to_string(),
            ],
        );

        // Abstract concepts
        hierarchies.insert(
            "concept".to_string(),
            vec![
                "idea".to_string(),
                "theory".to_string(),
                "principle".to_string(),
                "belief".to_string(),
            ],
        );

        hierarchies
    }

    /// Build semantic relation database
    fn build_semantic_relations() -> SemanticRelationDatabase {
        let mut synonyms = HashMap::new();
        synonyms.insert(
            "happy".to_string(),
            ["joyful", "glad", "cheerful", "elated"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        synonyms.insert(
            "big".to_string(),
            ["large", "huge", "enormous", "massive"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        synonyms.insert(
            "small".to_string(),
            ["tiny", "little", "minute", "miniature"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );

        let mut antonyms = HashMap::new();
        antonyms.insert(
            "happy".to_string(),
            ["sad", "miserable", "depressed"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        antonyms.insert(
            "big".to_string(),
            ["small", "tiny", "little"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        antonyms.insert(
            "hot".to_string(),
            ["cold", "freezing", "frigid"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );

        let mut hyponyms = HashMap::new();
        hyponyms.insert(
            "animal".to_string(),
            ["dog", "cat", "bird", "fish"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        hyponyms.insert(
            "vehicle".to_string(),
            ["car", "truck", "bicycle", "motorcycle"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        hyponyms.insert(
            "fruit".to_string(),
            ["apple", "orange", "banana", "grape"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );

        let mut meronyms = HashMap::new();
        meronyms.insert(
            "car".to_string(),
            ["wheel", "engine", "door", "window"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        meronyms.insert(
            "house".to_string(),
            ["roof", "wall", "floor", "room"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        meronyms.insert(
            "body".to_string(),
            ["head", "arm", "leg", "hand"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );

        SemanticRelationDatabase {
            synonyms,
            antonyms,
            hyponyms,
            meronyms,
            similarity_matrix: HashMap::new(),
        }
    }

    /// Build discourse marker mappings
    fn build_discourse_markers() -> HashMap<String, CoherenceRelationType> {
        let mut markers = HashMap::new();

        // Causal markers
        markers.insert("because".to_string(), CoherenceRelationType::Causal);
        markers.insert("since".to_string(), CoherenceRelationType::Causal);
        markers.insert("therefore".to_string(), CoherenceRelationType::Causal);
        markers.insert("consequently".to_string(), CoherenceRelationType::Causal);

        // Temporal markers
        markers.insert("then".to_string(), CoherenceRelationType::Temporal);
        markers.insert("next".to_string(), CoherenceRelationType::Temporal);
        markers.insert("finally".to_string(), CoherenceRelationType::Temporal);
        markers.insert("meanwhile".to_string(), CoherenceRelationType::Temporal);

        // Contrastive markers
        markers.insert("however".to_string(), CoherenceRelationType::Contrastive);
        markers.insert("but".to_string(), CoherenceRelationType::Contrastive);
        markers.insert(
            "nevertheless".to_string(),
            CoherenceRelationType::Contrastive,
        );
        markers.insert(
            "on the other hand".to_string(),
            CoherenceRelationType::Contrastive,
        );

        // Additive markers
        markers.insert("also".to_string(), CoherenceRelationType::Additive);
        markers.insert("furthermore".to_string(), CoherenceRelationType::Additive);
        markers.insert("moreover".to_string(), CoherenceRelationType::Additive);
        markers.insert("in addition".to_string(), CoherenceRelationType::Additive);

        markers
    }

    /// Build figurative language patterns
    fn build_figurative_patterns() -> HashMap<String, Vec<String>> {
        let mut patterns = HashMap::new();

        patterns.insert(
            "time_is_money".to_string(),
            vec![
                "spend time".to_string(),
                "save time".to_string(),
                "waste time".to_string(),
                "invest time".to_string(),
                "time is valuable".to_string(),
            ],
        );

        patterns.insert(
            "argument_is_war".to_string(),
            vec![
                "attack argument".to_string(),
                "defend position".to_string(),
                "win debate".to_string(),
                "shoot down idea".to_string(),
                "target weakness".to_string(),
            ],
        );

        patterns.insert(
            "life_is_journey".to_string(),
            vec![
                "life path".to_string(),
                "crossroads".to_string(),
                "dead end".to_string(),
                "moving forward".to_string(),
                "journey of life".to_string(),
            ],
        );

        patterns
    }

    /// Analyze semantic fluency of sentences
    pub fn analyze_semantic_fluency(&self, sentences: &[String]) -> SemanticFluencyResult {
        let semantic_coherence = self.calculate_semantic_coherence(sentences);
        let meaning_preservation = self.calculate_meaning_preservation(sentences);
        let conceptual_clarity = self.calculate_conceptual_clarity(sentences);
        let semantic_appropriateness = self.calculate_semantic_appropriateness(sentences);
        let context_sensitivity = self.calculate_context_sensitivity(sentences);
        let semantic_density = self.calculate_semantic_density(sentences);
        let ambiguity_score = self.calculate_ambiguity_score(sentences);
        let semantic_relations = self.analyze_semantic_relations(sentences);

        let advanced_metrics = if self.config.enable_advanced_relations {
            self.calculate_advanced_metrics(sentences)
        } else {
            self.create_default_advanced_metrics()
        };

        let discourse_analysis = if self.config.enable_discourse_analysis {
            self.analyze_discourse_structure(sentences)
        } else {
            self.create_default_discourse_analysis()
        };

        SemanticFluencyResult {
            semantic_coherence,
            meaning_preservation,
            conceptual_clarity,
            semantic_appropriateness,
            context_sensitivity,
            semantic_density,
            ambiguity_score,
            semantic_relations,
            advanced_metrics,
            discourse_analysis,
        }
    }

    /// Calculate semantic coherence across sentences
    pub fn calculate_semantic_coherence(&self, sentences: &[String]) -> f64 {
        if sentences.len() < 2 {
            return 1.0;
        }

        let mut coherence_sum = 0.0;
        let mut comparisons = 0;

        for i in 0..sentences.len() - 1 {
            let current_words = self.tokenize_sentence(&sentences[i]);
            let next_words = self.tokenize_sentence(&sentences[i + 1]);

            let semantic_overlap = self.calculate_semantic_overlap(&current_words, &next_words);
            coherence_sum += semantic_overlap;
            comparisons += 1;
        }

        if comparisons > 0 {
            coherence_sum / comparisons as f64
        } else {
            0.0
        }
    }

    /// Calculate semantic overlap between word lists
    fn calculate_semantic_overlap(&self, words1: &[String], words2: &[String]) -> f64 {
        if words1.is_empty() || words2.is_empty() {
            return 0.0;
        }

        let mut overlap_score = 0.0;
        let mut total_comparisons = 0;

        for word1 in words1 {
            for word2 in words2 {
                let similarity = self.calculate_word_semantic_similarity(word1, word2);
                overlap_score += similarity;
                total_comparisons += 1;
            }
        }

        if total_comparisons > 0 {
            overlap_score / total_comparisons as f64
        } else {
            0.0
        }
    }

    /// Calculate semantic similarity between two words
    fn calculate_word_semantic_similarity(&self, word1: &str, word2: &str) -> f64 {
        // Check if words are identical
        if word1 == word2 {
            return 1.0;
        }

        // Check cached similarity
        if let Some(&similarity) = self
            .semantic_relations
            .similarity_matrix
            .get(&(word1.to_string(), word2.to_string()))
        {
            return similarity;
        }

        // Check synonymy
        if let Some(synonyms) = self.semantic_relations.synonyms.get(word1) {
            if synonyms.contains(word2) {
                return 0.9;
            }
        }

        // Check if words are in the same semantic field
        for field_words in self.semantic_networks.values() {
            if field_words.contains(word1) && field_words.contains(word2) {
                return 0.7;
            }
        }

        // Check hyponymy relations
        for (hypernym, hyponyms) in &self.semantic_relations.hyponyms {
            if (word1 == hypernym && hyponyms.contains(word2))
                || (word2 == hypernym && hyponyms.contains(word1))
            {
                return 0.6;
            }
        }

        // Default similarity based on string similarity (simplified)
        let common_chars = word1.chars().filter(|c| word2.contains(*c)).count();
        let max_length = word1.len().max(word2.len());

        if max_length > 0 {
            (common_chars as f64 / max_length as f64) * 0.3
        } else {
            0.0
        }
    }

    /// Calculate meaning preservation
    pub fn calculate_meaning_preservation(&self, sentences: &[String]) -> f64 {
        let mut preservation_sum = 0.0;

        for sentence in sentences {
            let words = self.tokenize_sentence(sentence);
            let content_words = self.filter_content_words(&words);

            let preservation = if !content_words.is_empty() {
                let semantic_density = content_words.len() as f64 / words.len().max(1) as f64;
                let conceptual_weight = self.calculate_conceptual_weight(&content_words);
                (semantic_density + conceptual_weight) / 2.0
            } else {
                0.0
            };

            preservation_sum += preservation;
        }

        if !sentences.is_empty() {
            preservation_sum / sentences.len() as f64
        } else {
            0.0
        }
    }

    /// Filter content words from word list
    fn filter_content_words(&self, words: &[String]) -> Vec<String> {
        let function_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "as", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did",
        ];

        words
            .iter()
            .filter(|word| !function_words.contains(&word.as_str()))
            .cloned()
            .collect()
    }

    /// Calculate conceptual weight of words
    fn calculate_conceptual_weight(&self, words: &[String]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }

        let mut weight_sum = 0.0;
        for word in words {
            let weight = if self.is_abstract_concept(word) {
                1.0 // Abstract concepts have high weight
            } else if self.is_concrete_concept(word) {
                0.8 // Concrete concepts have medium-high weight
            } else {
                0.5 // Other words have medium weight
            };
            weight_sum += weight;
        }

        weight_sum / words.len() as f64
    }

    /// Check if word represents abstract concept
    fn is_abstract_concept(&self, word: &str) -> bool {
        let abstract_indicators = [
            "concept",
            "idea",
            "thought",
            "belief",
            "theory",
            "principle",
            "philosophy",
            "emotion",
            "feeling",
        ];
        abstract_indicators
            .iter()
            .any(|indicator| word.contains(indicator))
            || word.ends_with("tion")
            || word.ends_with("ness")
            || word.ends_with("ity")
    }

    /// Check if word represents concrete concept
    fn is_concrete_concept(&self, word: &str) -> bool {
        // Check if word is in concrete semantic fields
        let concrete_fields = ["animals", "colors", "transportation"];
        concrete_fields.iter().any(|field| {
            if let Some(field_words) = self.semantic_networks.get(*field) {
                field_words.contains(word)
            } else {
                false
            }
        })
    }

    /// Calculate conceptual clarity
    pub fn calculate_conceptual_clarity(&self, sentences: &[String]) -> f64 {
        let mut clarity_sum = 0.0;

        for sentence in sentences {
            let words = self.tokenize_sentence(sentence);
            let abstract_words = self.count_abstract_words(&words);
            let concrete_words = self.count_concrete_words(&words);
            let ambiguous_words = self.count_ambiguous_words(&words);

            let total_content = abstract_words + concrete_words;
            let clarity = if total_content > 0 {
                let concrete_ratio = concrete_words as f64 / total_content as f64;
                let ambiguity_penalty = (ambiguous_words as f64 / words.len().max(1) as f64) * 0.3;
                (concrete_ratio * 0.7 + 0.3 - ambiguity_penalty)
                    .max(0.0)
                    .min(1.0)
            } else {
                0.5
            };

            clarity_sum += clarity;
        }

        if !sentences.is_empty() {
            clarity_sum / sentences.len() as f64
        } else {
            0.0
        }
    }

    /// Count abstract words in text
    fn count_abstract_words(&self, words: &[String]) -> usize {
        words
            .iter()
            .filter(|word| self.is_abstract_concept(word))
            .count()
    }

    /// Count concrete words in text
    fn count_concrete_words(&self, words: &[String]) -> usize {
        words
            .iter()
            .filter(|word| self.is_concrete_concept(word))
            .count()
    }

    /// Count ambiguous words
    fn count_ambiguous_words(&self, words: &[String]) -> usize {
        let ambiguous_words = [
            "bank", "bark", "bat", "fair", "light", "right", "left", "match", "mean", "run", "play",
        ];
        words
            .iter()
            .filter(|word| ambiguous_words.contains(&word.as_str()))
            .count()
    }

    /// Calculate semantic appropriateness
    pub fn calculate_semantic_appropriateness(&self, sentences: &[String]) -> f64 {
        let mut appropriateness_sum = 0.0;

        for sentence in sentences {
            let words = self.tokenize_sentence(sentence);
            let semantic_consistency = self.calculate_semantic_consistency(&words);
            let field_coherence = self.calculate_field_coherence(&words);

            let appropriateness = (semantic_consistency + field_coherence) / 2.0;
            appropriateness_sum += appropriateness;
        }

        if !sentences.is_empty() {
            appropriateness_sum / sentences.len() as f64
        } else {
            0.0
        }
    }

    /// Calculate semantic consistency within a sentence
    fn calculate_semantic_consistency(&self, words: &[String]) -> f64 {
        if words.len() < 2 {
            return 1.0;
        }

        let mut consistency_sum = 0.0;
        let mut comparisons = 0;

        for i in 0..words.len() {
            for j in i + 1..words.len() {
                let similarity = self.calculate_word_semantic_similarity(&words[i], &words[j]);
                consistency_sum += similarity;
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            consistency_sum / comparisons as f64
        } else {
            0.0
        }
    }

    /// Calculate semantic field coherence
    fn calculate_field_coherence(&self, words: &[String]) -> f64 {
        let mut field_counts = HashMap::new();
        let mut total_field_words = 0;

        for word in words {
            for (field_name, field_words) in &self.semantic_networks {
                if field_words.contains(word) {
                    *field_counts.entry(field_name.clone()).or_insert(0) += 1;
                    total_field_words += 1;
                    break;
                }
            }
        }

        if total_field_words == 0 {
            return 0.5; // Neutral coherence for words not in semantic fields
        }

        // Calculate field distribution entropy (lower entropy = more coherent)
        let mut entropy = 0.0;
        for &count in field_counts.values() {
            if count > 0 {
                let p = count as f64 / total_field_words as f64;
                entropy -= p * p.log2();
            }
        }

        // Normalize and invert entropy (higher coherence = lower entropy)
        let max_entropy = (field_counts.len() as f64).log2().max(1.0);
        1.0 - (entropy / max_entropy)
    }

    /// Calculate context sensitivity
    pub fn calculate_context_sensitivity(&self, sentences: &[String]) -> f64 {
        if sentences.len() < 2 {
            return 1.0;
        }

        let mut sensitivity_sum = 0.0;
        let mut comparisons = 0;

        for i in 0..sentences.len().saturating_sub(self.config.context_window) {
            let window_end = (i + self.config.context_window).min(sentences.len());
            let context_coherence =
                self.calculate_context_window_coherence(&sentences[i..window_end]);
            sensitivity_sum += context_coherence;
            comparisons += 1;
        }

        if comparisons > 0 {
            sensitivity_sum / comparisons as f64
        } else {
            0.0
        }
    }

    /// Calculate coherence within a context window
    fn calculate_context_window_coherence(&self, sentences: &[String]) -> f64 {
        if sentences.len() < 2 {
            return 1.0;
        }

        let all_words: Vec<String> = sentences
            .iter()
            .flat_map(|s| self.tokenize_sentence(s))
            .collect();

        let semantic_fields = self.identify_semantic_fields(&all_words);
        let field_transitions = self.analyze_field_transitions(sentences);

        let field_consistency = if !semantic_fields.is_empty() {
            1.0 / semantic_fields.len() as f64 // Fewer fields = more consistent
        } else {
            0.5
        };

        let transition_smoothness = self.calculate_transition_smoothness(&field_transitions);

        (field_consistency + transition_smoothness) / 2.0
    }

    /// Identify semantic fields in word list
    fn identify_semantic_fields(&self, words: &[String]) -> HashSet<String> {
        let mut fields = HashSet::new();

        for word in words {
            for (field_name, field_words) in &self.semantic_networks {
                if field_words.contains(word) {
                    fields.insert(field_name.clone());
                }
            }
        }

        fields
    }

    /// Analyze field transitions between sentences
    fn analyze_field_transitions(&self, sentences: &[String]) -> Vec<(String, String)> {
        let mut transitions = Vec::new();

        for i in 0..sentences.len().saturating_sub(1) {
            let words1 = self.tokenize_sentence(&sentences[i]);
            let words2 = self.tokenize_sentence(&sentences[i + 1]);

            let fields1 = self.identify_semantic_fields(&words1);
            let fields2 = self.identify_semantic_fields(&words2);

            for field1 in &fields1 {
                for field2 in &fields2 {
                    if field1 != field2 {
                        transitions.push((field1.clone(), field2.clone()));
                    }
                }
            }
        }

        transitions
    }

    /// Calculate transition smoothness
    fn calculate_transition_smoothness(&self, transitions: &[(String, String)]) -> f64 {
        if transitions.is_empty() {
            return 1.0;
        }

        // Count transition frequency
        let mut transition_counts = HashMap::new();
        for transition in transitions {
            *transition_counts.entry(transition.clone()).or_insert(0) += 1;
        }

        // Calculate smoothness based on transition diversity
        let unique_transitions = transition_counts.len();
        let total_transitions = transitions.len();

        if total_transitions > 0 {
            1.0 - (unique_transitions as f64 / total_transitions as f64).min(1.0)
        } else {
            1.0
        }
    }

    /// Calculate semantic density
    pub fn calculate_semantic_density(&self, sentences: &[String]) -> f64 {
        let mut density_sum = 0.0;

        for sentence in sentences {
            let words = self.tokenize_sentence(sentence);
            let content_words = self.filter_content_words(&words);

            let unique_concepts = self.count_unique_semantic_concepts(&content_words);
            let density = if !content_words.is_empty() {
                unique_concepts as f64 / content_words.len() as f64
            } else {
                0.0
            };

            density_sum += density;
        }

        if !sentences.is_empty() {
            density_sum / sentences.len() as f64
        } else {
            0.0
        }
    }

    /// Count unique semantic concepts
    fn count_unique_semantic_concepts(&self, words: &[String]) -> usize {
        let mut concepts = HashSet::new();

        for word in words {
            // Check semantic networks
            for (field_name, field_words) in &self.semantic_networks {
                if field_words.contains(word) {
                    concepts.insert(field_name.clone());
                    break;
                }
            }

            // Check conceptual hierarchies
            for (concept, _) in &self.conceptual_hierarchies {
                if word == concept {
                    concepts.insert(concept.clone());
                    break;
                }
            }

            // Default: each content word is a concept
            concepts.insert(word.clone());
        }

        concepts.len()
    }

    /// Calculate ambiguity score
    pub fn calculate_ambiguity_score(&self, sentences: &[String]) -> f64 {
        let mut ambiguity_sum = 0.0;

        for sentence in sentences {
            let words = self.tokenize_sentence(sentence);
            let ambiguous_count = self.count_ambiguous_words(&words);
            let polysemous_count = self.count_polysemous_words(&words);

            let total_ambiguity = ambiguous_count + polysemous_count;
            let clarity_score = if !words.is_empty() {
                1.0 - (total_ambiguity as f64 / words.len() as f64).min(1.0)
            } else {
                1.0
            };

            ambiguity_sum += clarity_score;
        }

        if !sentences.is_empty() {
            ambiguity_sum / sentences.len() as f64
        } else {
            1.0
        }
    }

    /// Count polysemous words (words with multiple meanings)
    fn count_polysemous_words(&self, words: &[String]) -> usize {
        // Simplified polysemy detection based on word appearing in multiple semantic fields
        let mut polysemous_count = 0;

        for word in words {
            let mut field_count = 0;
            for field_words in self.semantic_networks.values() {
                if field_words.contains(word) {
                    field_count += 1;
                }
            }

            if field_count > 1 {
                polysemous_count += 1;
            }
        }

        polysemous_count
    }

    /// Analyze semantic relations in text
    pub fn analyze_semantic_relations(&self, sentences: &[String]) -> HashMap<String, f64> {
        let all_words = sentences
            .iter()
            .flat_map(|s| self.tokenize_sentence(s))
            .collect::<Vec<_>>();

        let mut relations = HashMap::new();
        relations.insert(
            "synonymy".to_string(),
            self.calculate_synonymy_relations(&all_words),
        );
        relations.insert(
            "antonymy".to_string(),
            self.calculate_antonymy_relations(&all_words),
        );
        relations.insert(
            "hyponymy".to_string(),
            self.calculate_hyponymy_relations(&all_words),
        );
        relations.insert(
            "meronymy".to_string(),
            self.calculate_meronymy_relations(&all_words),
        );

        relations
    }

    /// Calculate synonymy relations
    fn calculate_synonymy_relations(&self, words: &[String]) -> f64 {
        let mut synonym_pairs = 0;
        let mut total_pairs = 0;

        for i in 0..words.len() {
            for j in i + 1..words.len() {
                total_pairs += 1;

                if let Some(synonyms) = self.semantic_relations.synonyms.get(&words[i]) {
                    if synonyms.contains(&words[j]) {
                        synonym_pairs += 1;
                    }
                }
            }
        }

        if total_pairs > 0 {
            synonym_pairs as f64 / total_pairs as f64
        } else {
            0.0
        }
    }

    /// Calculate antonymy relations
    fn calculate_antonymy_relations(&self, words: &[String]) -> f64 {
        let mut antonym_pairs = 0;
        let mut total_pairs = 0;

        for i in 0..words.len() {
            for j in i + 1..words.len() {
                total_pairs += 1;

                if let Some(antonyms) = self.semantic_relations.antonyms.get(&words[i]) {
                    if antonyms.contains(&words[j]) {
                        antonym_pairs += 1;
                    }
                }
            }
        }

        if total_pairs > 0 {
            antonym_pairs as f64 / total_pairs as f64
        } else {
            0.0
        }
    }

    /// Calculate hyponymy relations
    fn calculate_hyponymy_relations(&self, words: &[String]) -> f64 {
        let mut hyponym_score = 0.0;
        let mut total_checks = 0;

        for (hypernym, hyponyms) in &self.semantic_relations.hyponyms {
            total_checks += 1;

            let has_hypernym = words.contains(hypernym);
            let hyponym_count = hyponyms.iter().filter(|h| words.contains(h)).count();

            if has_hypernym && hyponym_count > 0 {
                hyponym_score += 1.0;
            } else if hyponym_count > 1 {
                hyponym_score += 0.5;
            }
        }

        if total_checks > 0 {
            hyponym_score / total_checks as f64
        } else {
            0.0
        }
    }

    /// Calculate meronymy relations
    fn calculate_meronymy_relations(&self, words: &[String]) -> f64 {
        let mut meronym_score = 0.0;
        let mut total_checks = 0;

        for (whole, parts) in &self.semantic_relations.meronyms {
            total_checks += 1;

            let has_whole = words.contains(whole);
            let part_count = parts.iter().filter(|p| words.contains(p)).count();

            if has_whole && part_count > 0 {
                meronym_score += 1.0;
            } else if part_count > 1 {
                meronym_score += 0.5;
            }
        }

        if total_checks > 0 {
            meronym_score / total_checks as f64
        } else {
            0.0
        }
    }

    /// Calculate advanced semantic metrics
    fn calculate_advanced_metrics(&self, sentences: &[String]) -> AdvancedSemanticMetrics {
        let field_coverage = self.analyze_semantic_field_coverage(sentences);
        let conceptual_complexity = self.analyze_conceptual_complexity(sentences);
        let consistency_analysis = self.analyze_consistency(sentences);
        let topic_coherence = self.analyze_topic_coherence(sentences);
        let semantic_roles = self.analyze_semantic_roles(sentences);
        let figurative_language = self.analyze_figurative_language(sentences);

        AdvancedSemanticMetrics {
            field_coverage,
            conceptual_complexity,
            consistency_analysis,
            topic_coherence,
            semantic_roles,
            figurative_language,
        }
    }

    /// Create default advanced metrics when disabled
    fn create_default_advanced_metrics(&self) -> AdvancedSemanticMetrics {
        AdvancedSemanticMetrics {
            field_coverage: SemanticFieldCoverage {
                field_coverage_map: HashMap::new(),
                field_diversity: 0.5,
                dominant_fields: vec![],
                field_transitions: HashMap::new(),
                coverage_consistency: 0.5,
            },
            conceptual_complexity: ConceptualComplexity {
                abstraction_ratio: 0.5,
                conceptual_depth: 2.0,
                interconnectedness: 0.5,
                novelty_score: 0.0,
                complexity_distribution: ComplexityDistribution {
                    simple_concepts: 0.6,
                    moderate_concepts: 0.3,
                    complex_concepts: 0.1,
                    complexity_variance: 0.2,
                },
            },
            consistency_analysis: ConsistencyAnalysis {
                terminological_consistency: 0.8,
                conceptual_consistency: 0.7,
                frame_consistency: 0.7,
                inconsistency_patterns: vec![],
            },
            topic_coherence: TopicCoherence {
                overall_coherence: 0.6,
                topic_consistency: 0.7,
                transition_smoothness: 0.6,
                main_topics: vec![],
                topic_distribution: HashMap::new(),
            },
            semantic_roles: SemanticRoleAnalysis {
                role_patterns: HashMap::new(),
                role_diversity: 0.5,
                assignment_accuracy: 0.6,
                complex_structures: vec![],
            },
            figurative_language: FigurativeLanguageAnalysis {
                metaphors: vec![],
                figurative_density: 0.0,
                conceptual_patterns: HashMap::new(),
                appropriateness_score: 0.8,
            },
        }
    }

    /// Analyze semantic field coverage
    fn analyze_semantic_field_coverage(&self, sentences: &[String]) -> SemanticFieldCoverage {
        let all_words: Vec<String> = sentences
            .iter()
            .flat_map(|s| self.tokenize_sentence(s))
            .collect();

        let mut field_coverage_map = HashMap::new();
        let mut total_words = 0;

        for (field_name, field_words) in &self.semantic_networks {
            let coverage_count = all_words
                .iter()
                .filter(|word| field_words.contains(word))
                .count();

            field_coverage_map.insert(
                field_name.clone(),
                coverage_count as f64 / field_words.len() as f64,
            );
            total_words += coverage_count;
        }

        let field_diversity = field_coverage_map
            .values()
            .filter(|&&coverage| coverage > 0.0)
            .count() as f64
            / field_coverage_map.len() as f64;

        let mut dominant_fields: Vec<(String, f64)> = field_coverage_map
            .iter()
            .map(|(field, &coverage)| (field.clone(), coverage))
            .collect();
        dominant_fields.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let dominant_fields: Vec<String> = dominant_fields
            .into_iter()
            .take(3)
            .map(|(field, _)| field)
            .collect();

        let field_transitions = self.analyze_field_transitions(sentences).into_iter().fold(
            HashMap::new(),
            |mut acc, transition| {
                *acc.entry(transition).or_insert(0) += 1;
                acc
            },
        );

        let coverage_consistency = if !field_coverage_map.is_empty() {
            let values: Vec<f64> = field_coverage_map.values().copied().collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
            1.0 - variance.sqrt().min(1.0)
        } else {
            0.0
        };

        SemanticFieldCoverage {
            field_coverage_map,
            field_diversity,
            dominant_fields,
            field_transitions,
            coverage_consistency,
        }
    }

    /// Analyze conceptual complexity
    fn analyze_conceptual_complexity(&self, sentences: &[String]) -> ConceptualComplexity {
        let all_words: Vec<String> = sentences
            .iter()
            .flat_map(|s| self.tokenize_sentence(s))
            .collect();

        let abstract_count = self.count_abstract_words(&all_words);
        let concrete_count = self.count_concrete_words(&all_words);
        let total_concepts = abstract_count + concrete_count;

        let abstraction_ratio = if total_concepts > 0 {
            abstract_count as f64 / total_concepts as f64
        } else {
            0.5
        };

        let conceptual_depth = self.calculate_conceptual_depth(&all_words);
        let interconnectedness = self.calculate_concept_interconnectedness(&all_words);
        let novelty_score = self.calculate_conceptual_novelty(&all_words);

        let complexity_distribution = self.analyze_complexity_distribution(&all_words);

        ConceptualComplexity {
            abstraction_ratio,
            conceptual_depth,
            interconnectedness,
            novelty_score,
            complexity_distribution,
        }
    }

    /// Calculate conceptual depth
    fn calculate_conceptual_depth(&self, words: &[String]) -> f64 {
        let mut max_depth = 0;
        let mut total_depth = 0;
        let mut concept_count = 0;

        for word in words {
            let depth = self.get_concept_hierarchy_depth(word);
            if depth > 0 {
                max_depth = max_depth.max(depth);
                total_depth += depth;
                concept_count += 1;
            }
        }

        if concept_count > 0 {
            total_depth as f64 / concept_count as f64
        } else {
            1.0
        }
    }

    /// Get hierarchy depth for a concept
    fn get_concept_hierarchy_depth(&self, word: &str) -> usize {
        // Check how deep the word is in conceptual hierarchies
        for (level, hierarchy) in self.conceptual_hierarchies.iter().enumerate() {
            if hierarchy.1.contains(&word.to_string()) {
                return level + 1;
            }
        }
        0
    }

    /// Calculate concept interconnectedness
    fn calculate_concept_interconnectedness(&self, words: &[String]) -> f64 {
        if words.len() < 2 {
            return 0.0;
        }

        let mut connection_count = 0;
        let mut total_pairs = 0;

        for i in 0..words.len() {
            for j in i + 1..words.len() {
                total_pairs += 1;
                if self.are_concepts_connected(&words[i], &words[j]) {
                    connection_count += 1;
                }
            }
        }

        if total_pairs > 0 {
            connection_count as f64 / total_pairs as f64
        } else {
            0.0
        }
    }

    /// Check if two concepts are connected
    fn are_concepts_connected(&self, word1: &str, word2: &str) -> bool {
        // Check semantic networks
        for field_words in self.semantic_networks.values() {
            if field_words.contains(word1) && field_words.contains(word2) {
                return true;
            }
        }

        // Check semantic relations
        if let Some(synonyms) = self.semantic_relations.synonyms.get(word1) {
            if synonyms.contains(word2) {
                return true;
            }
        }

        // Check hyponymy relations
        for hyponyms in self.semantic_relations.hyponyms.values() {
            if hyponyms.contains(word1) && hyponyms.contains(word2) {
                return true;
            }
        }

        false
    }

    /// Calculate conceptual novelty
    fn calculate_conceptual_novelty(&self, words: &[String]) -> f64 {
        let mut novel_concepts = 0;
        let mut total_concepts = 0;

        for word in words {
            total_concepts += 1;

            // Check if word is not in any of our known semantic networks
            let is_novel = !self
                .semantic_networks
                .values()
                .any(|field_words| field_words.contains(word));

            if is_novel && word.len() > 6 {
                // Longer words might be more novel
                novel_concepts += 1;
            }
        }

        if total_concepts > 0 {
            novel_concepts as f64 / total_concepts as f64
        } else {
            0.0
        }
    }

    /// Analyze complexity distribution
    fn analyze_complexity_distribution(&self, words: &[String]) -> ComplexityDistribution {
        let mut simple_count = 0;
        let mut moderate_count = 0;
        let mut complex_count = 0;

        for word in words {
            let complexity = self.assess_word_complexity(word);
            match complexity {
                0 => simple_count += 1,
                1 => moderate_count += 1,
                2 => complex_count += 1,
                _ => {}
            }
        }

        let total = words.len() as f64;
        let simple_concepts = simple_count as f64 / total;
        let moderate_concepts = moderate_count as f64 / total;
        let complex_concepts = complex_count as f64 / total;

        let complexity_variance = self.calculate_complexity_variance(words);

        ComplexityDistribution {
            simple_concepts,
            moderate_concepts,
            complex_concepts,
            complexity_variance,
        }
    }

    /// Assess word complexity (0=simple, 1=moderate, 2=complex)
    fn assess_word_complexity(&self, word: &str) -> usize {
        if word.len() <= 4 {
            0 // Simple
        } else if word.len() <= 8 {
            1 // Moderate
        } else {
            2 // Complex
        }
    }

    /// Calculate complexity variance
    fn calculate_complexity_variance(&self, words: &[String]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }

        let complexities: Vec<f64> = words
            .iter()
            .map(|word| self.assess_word_complexity(word) as f64)
            .collect();

        let mean = complexities.iter().sum::<f64>() / complexities.len() as f64;
        let variance = complexities
            .iter()
            .map(|&c| (c - mean).powi(2))
            .sum::<f64>()
            / complexities.len() as f64;

        variance
    }

    /// Analyze consistency across text
    fn analyze_consistency(&self, sentences: &[String]) -> ConsistencyAnalysis {
        let terminological_consistency = self.calculate_terminological_consistency(sentences);
        let conceptual_consistency = self.calculate_conceptual_consistency(sentences);
        let frame_consistency = self.calculate_frame_consistency(sentences);
        let inconsistency_patterns = self.detect_inconsistency_patterns(sentences);

        ConsistencyAnalysis {
            terminological_consistency,
            conceptual_consistency,
            frame_consistency,
            inconsistency_patterns,
        }
    }

    /// Calculate terminological consistency
    fn calculate_terminological_consistency(&self, sentences: &[String]) -> f64 {
        // Simplified: check for consistent use of synonyms
        let all_words: Vec<String> = sentences
            .iter()
            .flat_map(|s| self.tokenize_sentence(s))
            .collect();

        let mut consistency_score = 0.0;
        let mut checks = 0;

        for (word, synonyms) in &self.semantic_relations.synonyms {
            if all_words.contains(word) {
                checks += 1;
                let synonym_count = synonyms
                    .iter()
                    .filter(|syn| all_words.contains(syn))
                    .count();

                // Consistency is better when fewer synonyms are mixed
                consistency_score += if synonym_count == 0 {
                    1.0
                } else {
                    1.0 / (synonym_count + 1) as f64
                };
            }
        }

        if checks > 0 {
            consistency_score / checks as f64
        } else {
            0.8 // Default to reasonably consistent
        }
    }

    /// Calculate conceptual consistency
    fn calculate_conceptual_consistency(&self, sentences: &[String]) -> f64 {
        if sentences.len() < 2 {
            return 1.0;
        }

        let mut consistency_scores = Vec::new();

        for i in 0..sentences.len() - 1 {
            let words1 = self.tokenize_sentence(&sentences[i]);
            let words2 = self.tokenize_sentence(&sentences[i + 1]);

            let consistency = self.calculate_semantic_consistency(&words1) * 0.5
                + self.calculate_semantic_consistency(&words2) * 0.5;
            consistency_scores.push(consistency);
        }

        if !consistency_scores.is_empty() {
            consistency_scores.iter().sum::<f64>() / consistency_scores.len() as f64
        } else {
            0.7
        }
    }

    /// Calculate semantic frame consistency
    fn calculate_frame_consistency(&self, sentences: &[String]) -> f64 {
        // Simplified frame consistency based on semantic field coherence
        let mut frame_scores = Vec::new();

        for sentence in sentences {
            let words = self.tokenize_sentence(sentence);
            let frame_score = self.calculate_field_coherence(&words);
            frame_scores.push(frame_score);
        }

        if !frame_scores.is_empty() {
            frame_scores.iter().sum::<f64>() / frame_scores.len() as f64
        } else {
            0.6
        }
    }

    /// Detect inconsistency patterns
    fn detect_inconsistency_patterns(&self, sentences: &[String]) -> Vec<InconsistencyPattern> {
        let mut patterns = Vec::new();

        // Check for contradictory statements (simplified)
        for (i, sentence1) in sentences.iter().enumerate() {
            for (j, sentence2) in sentences.iter().enumerate().skip(i + 1) {
                if self.are_sentences_contradictory(sentence1, sentence2) {
                    patterns.push(InconsistencyPattern {
                        pattern_type: InconsistencyType::ContradictoryStatements,
                        description: "Contradictory statements detected".to_string(),
                        severity: 0.8,
                        locations: vec![i, j],
                        resolution_suggestion: "Review and resolve contradictory information"
                            .to_string(),
                    });
                }
            }
        }

        patterns
    }

    /// Check if sentences are contradictory (simplified)
    fn are_sentences_contradictory(&self, sentence1: &str, sentence2: &str) -> bool {
        let words1: HashSet<String> = self.tokenize_sentence(sentence1).into_iter().collect();
        let words2: HashSet<String> = self.tokenize_sentence(sentence2).into_iter().collect();

        // Check for antonym pairs
        for word1 in &words1 {
            if let Some(antonyms) = self.semantic_relations.antonyms.get(word1) {
                for word2 in &words2 {
                    if antonyms.contains(word2) {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Analyze topic coherence
    fn analyze_topic_coherence(&self, sentences: &[String]) -> TopicCoherence {
        let main_topics = self.extract_main_topics(sentences);
        let topic_distribution = self.calculate_topic_distribution(&main_topics);
        let overall_coherence = self.calculate_overall_topic_coherence(&main_topics);
        let topic_consistency = self.calculate_topic_consistency(sentences, &main_topics);
        let transition_smoothness =
            self.calculate_topic_transition_smoothness(sentences, &main_topics);

        TopicCoherence {
            overall_coherence,
            topic_consistency,
            transition_smoothness,
            main_topics,
            topic_distribution,
        }
    }

    /// Extract main topics from text
    fn extract_main_topics(&self, sentences: &[String]) -> Vec<Topic> {
        let mut topic_words = HashMap::new();

        // Use semantic fields as proxy for topics
        for sentence in sentences {
            let words = self.tokenize_sentence(sentence);
            let fields = self.identify_semantic_fields(&words);

            for field in fields {
                let counter = topic_words.entry(field.clone()).or_insert(0);
                *counter += 1;
            }
        }

        topic_words
            .into_iter()
            .map(|(field, count)| Topic {
                name: field.clone(),
                keywords: self.get_field_keywords(&field),
                weight: count as f64 / sentences.len() as f64,
                coherence_score: 0.7, // Simplified
            })
            .collect()
    }

    /// Get keywords for a semantic field
    fn get_field_keywords(&self, field: &str) -> Vec<String> {
        if let Some(field_words) = self.semantic_networks.get(field) {
            field_words.iter().cloned().take(5).collect()
        } else {
            vec![]
        }
    }

    /// Calculate topic distribution
    fn calculate_topic_distribution(&self, topics: &[Topic]) -> HashMap<String, f64> {
        topics
            .iter()
            .map(|topic| (topic.name.clone(), topic.weight))
            .collect()
    }

    /// Calculate overall topic coherence
    fn calculate_overall_topic_coherence(&self, topics: &[Topic]) -> f64 {
        if topics.is_empty() {
            return 0.5;
        }

        let mean_coherence = topics
            .iter()
            .map(|topic| topic.coherence_score)
            .sum::<f64>()
            / topics.len() as f64;

        mean_coherence
    }

    /// Calculate topic consistency
    fn calculate_topic_consistency(&self, sentences: &[String], topics: &[Topic]) -> f64 {
        if sentences.is_empty() || topics.is_empty() {
            return 0.5;
        }

        // Simplified consistency based on topic distribution uniformity
        let total_weight: f64 = topics.iter().map(|t| t.weight).sum();

        if total_weight > 0.0 {
            let entropy = topics
                .iter()
                .map(|topic| {
                    let p = topic.weight / total_weight;
                    if p > 0.0 {
                        -p * p.log2()
                    } else {
                        0.0
                    }
                })
                .sum::<f64>();

            let max_entropy = (topics.len() as f64).log2();
            if max_entropy > 0.0 {
                1.0 - (entropy / max_entropy)
            } else {
                1.0
            }
        } else {
            0.5
        }
    }

    /// Calculate topic transition smoothness
    fn calculate_topic_transition_smoothness(&self, sentences: &[String], topics: &[Topic]) -> f64 {
        if sentences.len() < 2 {
            return 1.0;
        }

        let mut smooth_transitions = 0;
        let mut total_transitions = 0;

        for i in 0..sentences.len() - 1 {
            let fields1 = self.identify_semantic_fields(&self.tokenize_sentence(&sentences[i]));
            let fields2 = self.identify_semantic_fields(&self.tokenize_sentence(&sentences[i + 1]));

            total_transitions += 1;

            // Smooth if there's overlap in semantic fields
            if !fields1.is_disjoint(&fields2) || fields1.is_empty() || fields2.is_empty() {
                smooth_transitions += 1;
            }
        }

        if total_transitions > 0 {
            smooth_transitions as f64 / total_transitions as f64
        } else {
            1.0
        }
    }

    /// Analyze semantic roles
    fn analyze_semantic_roles(&self, sentences: &[String]) -> SemanticRoleAnalysis {
        let mut role_patterns = HashMap::new();
        let mut complex_structures = Vec::new();

        for sentence in sentences {
            let patterns = self.extract_role_patterns(sentence);
            for (pattern, count) in patterns {
                *role_patterns.entry(pattern).or_insert(0) += count;
            }

            let complex_struct = self.identify_complex_structures(sentence);
            complex_structures.extend(complex_struct);
        }

        let role_diversity = if !role_patterns.is_empty() {
            role_patterns.len() as f64 / sentences.len().max(1) as f64
        } else {
            0.0
        };

        let assignment_accuracy = 0.7; // Simplified

        SemanticRoleAnalysis {
            role_patterns,
            role_diversity,
            assignment_accuracy,
            complex_structures,
        }
    }

    /// Extract semantic role patterns from sentence
    fn extract_role_patterns(&self, sentence: &str) -> HashMap<String, usize> {
        let mut patterns = HashMap::new();
        let words = self.tokenize_sentence(sentence);

        // Simplified pattern recognition
        if words.len() >= 3 {
            // Look for Agent-Action-Patient patterns
            for i in 0..words.len().saturating_sub(2) {
                if self.is_likely_agent(&words[i])
                    && self.is_likely_action(&words[i + 1])
                    && self.is_likely_patient(&words[i + 2])
                {
                    *patterns
                        .entry("agent-action-patient".to_string())
                        .or_insert(0) += 1;
                }
            }
        }

        patterns
    }

    /// Check if word is likely an agent
    fn is_likely_agent(&self, word: &str) -> bool {
        // Simplified: check if word is in animals or human-related semantic fields
        if let Some(animals) = self.semantic_networks.get("animals") {
            animals.contains(word)
        } else {
            false
        }
    }

    /// Check if word is likely an action
    fn is_likely_action(&self, word: &str) -> bool {
        // Simplified: check for verb patterns
        word.ends_with("ing") || word.ends_with("ed") || word.ends_with("s")
    }

    /// Check if word is likely a patient
    fn is_likely_patient(&self, word: &str) -> bool {
        // Simplified: assume concrete nouns can be patients
        self.is_concrete_concept(word)
    }

    /// Identify complex semantic role structures
    fn identify_complex_structures(&self, sentence: &str) -> Vec<ComplexRoleStructure> {
        let words = self.tokenize_sentence(sentence);
        let mut structures = Vec::new();

        // Look for complex structures (simplified)
        if words.len() > 8 {
            structures.push(ComplexRoleStructure {
                structure_type: "complex_sentence".to_string(),
                components: words.clone(),
                complexity_score: words.len() as f64 / 10.0,
                clarity_score: 0.6,
            });
        }

        structures
    }

    /// Analyze figurative language
    fn analyze_figurative_language(&self, sentences: &[String]) -> FigurativeLanguageAnalysis {
        let mut metaphors = Vec::new();
        let mut pattern_matches = HashMap::new();

        for sentence in sentences {
            // Check for metaphor patterns
            for (pattern_name, pattern_phrases) in &self.figurative_patterns {
                for phrase in pattern_phrases {
                    if sentence.to_lowercase().contains(phrase) {
                        *pattern_matches.entry(pattern_name.clone()).or_insert(0.0) += 1.0;

                        metaphors.push(MetaphorInstance {
                            source_domain: self.extract_source_domain(pattern_name),
                            target_domain: self.extract_target_domain(pattern_name),
                            text: phrase.clone(),
                            mapping_strength: 0.7,
                            novelty: 0.3,
                        });
                    }
                }
            }
        }

        let total_words = sentences
            .iter()
            .map(|s| self.tokenize_sentence(s).len())
            .sum::<usize>();

        let figurative_density = if total_words > 0 {
            metaphors.len() as f64 / total_words as f64
        } else {
            0.0
        };

        let appropriateness_score = if figurative_density > 0.1 {
            0.6 // Too many metaphors might reduce clarity
        } else {
            0.8
        };

        FigurativeLanguageAnalysis {
            metaphors,
            figurative_density,
            conceptual_patterns: pattern_matches,
            appropriateness_score,
        }
    }

    /// Extract source domain from pattern name
    fn extract_source_domain(&self, pattern_name: &str) -> String {
        match pattern_name {
            "time_is_money" => "money".to_string(),
            "argument_is_war" => "war".to_string(),
            "life_is_journey" => "journey".to_string(),
            _ => "abstract".to_string(),
        }
    }

    /// Extract target domain from pattern name
    fn extract_target_domain(&self, pattern_name: &str) -> String {
        match pattern_name {
            "time_is_money" => "time".to_string(),
            "argument_is_war" => "argument".to_string(),
            "life_is_journey" => "life".to_string(),
            _ => "abstract".to_string(),
        }
    }

    /// Analyze discourse structure
    fn analyze_discourse_structure(&self, sentences: &[String]) -> DiscourseAnalysis {
        let global_coherence = self.calculate_global_coherence(sentences);
        let local_coherence = self.calculate_local_coherence(sentences);
        let coherence_relations = self.identify_coherence_relations(sentences);
        let discourse_markers = self.analyze_discourse_markers(sentences);
        let information_structure = self.analyze_information_structure(sentences);

        DiscourseAnalysis {
            global_coherence,
            local_coherence,
            coherence_relations,
            discourse_markers,
            information_structure,
        }
    }

    /// Create default discourse analysis
    fn create_default_discourse_analysis(&self) -> DiscourseAnalysis {
        DiscourseAnalysis {
            global_coherence: 0.6,
            local_coherence: vec![],
            coherence_relations: vec![],
            discourse_markers: DiscourseMarkerAnalysis {
                marker_frequency: HashMap::new(),
                appropriateness: 0.7,
                diversity: 0.5,
                missing_markers: vec![],
            },
            information_structure: InformationStructure {
                given_new_balance: 0.6,
                topic_continuity: 0.7,
                information_density: vec![],
                focus_structure: FocusStructure {
                    focus_points: vec![],
                    transition_smoothness: 0.6,
                    hierarchy_depth: 2,
                },
            },
        }
    }

    /// Calculate global coherence
    fn calculate_global_coherence(&self, sentences: &[String]) -> f64 {
        if sentences.len() < 2 {
            return 1.0;
        }

        // Global coherence based on semantic field consistency across all sentences
        let all_words: Vec<String> = sentences
            .iter()
            .flat_map(|s| self.tokenize_sentence(s))
            .collect();

        let semantic_fields = self.identify_semantic_fields(&all_words);
        let field_coherence = self.calculate_field_coherence(&all_words);

        field_coherence
    }

    /// Calculate local coherence for each sentence
    fn calculate_local_coherence(&self, sentences: &[String]) -> Vec<f64> {
        sentences
            .iter()
            .map(|sentence| {
                let words = self.tokenize_sentence(sentence);
                self.calculate_semantic_consistency(&words)
            })
            .collect()
    }

    /// Identify coherence relations between sentences
    fn identify_coherence_relations(&self, sentences: &[String]) -> Vec<CoherenceRelation> {
        let mut relations = Vec::new();

        for i in 0..sentences.len().saturating_sub(1) {
            let relation_type = self.determine_coherence_relation(&sentences[i], &sentences[i + 1]);
            let strength =
                self.calculate_relation_strength(&sentences[i], &sentences[i + 1], &relation_type);
            let explicit_markers = self.find_explicit_markers(&sentences[i + 1], &relation_type);

            relations.push(CoherenceRelation {
                relation_type,
                source_segment: i,
                target_segment: i + 1,
                strength,
                explicit_markers,
            });
        }

        relations
    }

    /// Determine coherence relation type between sentences
    fn determine_coherence_relation(
        &self,
        sentence1: &str,
        sentence2: &str,
    ) -> CoherenceRelationType {
        // Check for explicit discourse markers
        for (marker, relation_type) in &self.discourse_markers {
            if sentence2.to_lowercase().contains(marker) {
                return relation_type.clone();
            }
        }

        // Default to elaboration if no explicit markers
        CoherenceRelationType::Elaboration
    }

    /// Calculate relation strength
    fn calculate_relation_strength(
        &self,
        sentence1: &str,
        sentence2: &str,
        relation_type: &CoherenceRelationType,
    ) -> f64 {
        let words1 = self.tokenize_sentence(sentence1);
        let words2 = self.tokenize_sentence(sentence2);

        let semantic_overlap = self.calculate_semantic_overlap(&words1, &words2);

        // Adjust strength based on relation type
        match relation_type {
            CoherenceRelationType::Causal => semantic_overlap * 0.8,
            CoherenceRelationType::Temporal => semantic_overlap * 0.9,
            CoherenceRelationType::Contrastive => semantic_overlap * 0.6,
            CoherenceRelationType::Additive => semantic_overlap * 0.9,
            CoherenceRelationType::Elaboration => semantic_overlap,
            CoherenceRelationType::Background => semantic_overlap * 0.7,
        }
    }

    /// Find explicit discourse markers
    fn find_explicit_markers(
        &self,
        sentence: &str,
        relation_type: &CoherenceRelationType,
    ) -> Vec<String> {
        self.discourse_markers
            .iter()
            .filter(|(_, marker_type)| {
                std::mem::discriminant(*marker_type) == std::mem::discriminant(relation_type)
            })
            .filter(|(marker, _)| sentence.to_lowercase().contains(marker))
            .map(|(marker, _)| marker.clone())
            .collect()
    }

    /// Analyze discourse markers
    fn analyze_discourse_markers(&self, sentences: &[String]) -> DiscourseMarkerAnalysis {
        let mut marker_frequency = HashMap::new();
        let mut found_markers = Vec::new();

        for sentence in sentences {
            for (marker, _) in &self.discourse_markers {
                if sentence.to_lowercase().contains(marker) {
                    *marker_frequency.entry(marker.clone()).or_insert(0) += 1;
                    found_markers.push(marker.clone());
                }
            }
        }

        let diversity = if !marker_frequency.is_empty() {
            marker_frequency.len() as f64 / self.discourse_markers.len() as f64
        } else {
            0.0
        };

        let appropriateness = if !found_markers.is_empty() {
            0.8 // Simplified appropriateness score
        } else {
            0.5
        };

        let missing_markers = self
            .discourse_markers
            .keys()
            .filter(|marker| !marker_frequency.contains_key(*marker))
            .cloned()
            .collect();

        DiscourseMarkerAnalysis {
            marker_frequency,
            appropriateness,
            diversity,
            missing_markers,
        }
    }

    /// Analyze information structure
    fn analyze_information_structure(&self, sentences: &[String]) -> InformationStructure {
        let given_new_balance = self.calculate_given_new_balance(sentences);
        let topic_continuity = self.calculate_topic_continuity(sentences);
        let information_density = self.calculate_information_density(sentences);
        let focus_structure = self.analyze_focus_structure(sentences);

        InformationStructure {
            given_new_balance,
            topic_continuity,
            information_density,
            focus_structure,
        }
    }

    /// Calculate given-new information balance
    fn calculate_given_new_balance(&self, sentences: &[String]) -> f64 {
        if sentences.len() < 2 {
            return 0.5;
        }

        let mut balance_scores = Vec::new();
        let mut known_concepts = HashSet::new();

        for sentence in sentences {
            let words = self.tokenize_sentence(sentence);
            let content_words = self.filter_content_words(&words);

            let given_count = content_words
                .iter()
                .filter(|word| known_concepts.contains(word))
                .count();

            let new_count = content_words.len() - given_count;

            let balance = if content_words.is_empty() {
                0.5
            } else {
                given_count as f64 / content_words.len() as f64
            };

            balance_scores.push(balance);

            // Add new concepts to known set
            for word in content_words {
                known_concepts.insert(word);
            }
        }

        if !balance_scores.is_empty() {
            balance_scores.iter().sum::<f64>() / balance_scores.len() as f64
        } else {
            0.5
        }
    }

    /// Calculate topic continuity
    fn calculate_topic_continuity(&self, sentences: &[String]) -> f64 {
        if sentences.len() < 2 {
            return 1.0;
        }

        let mut continuity_scores = Vec::new();

        for i in 0..sentences.len() - 1 {
            let fields1 = self.identify_semantic_fields(&self.tokenize_sentence(&sentences[i]));
            let fields2 = self.identify_semantic_fields(&self.tokenize_sentence(&sentences[i + 1]));

            let continuity = if fields1.is_empty() || fields2.is_empty() {
                0.5
            } else {
                let overlap = fields1.intersection(&fields2).count();
                let total = fields1.union(&fields2).count();

                if total > 0 {
                    overlap as f64 / total as f64
                } else {
                    0.5
                }
            };

            continuity_scores.push(continuity);
        }

        if !continuity_scores.is_empty() {
            continuity_scores.iter().sum::<f64>() / continuity_scores.len() as f64
        } else {
            0.5
        }
    }

    /// Calculate information density per sentence
    fn calculate_information_density(&self, sentences: &[String]) -> Vec<f64> {
        sentences
            .iter()
            .map(|sentence| {
                let words = self.tokenize_sentence(sentence);
                let content_words = self.filter_content_words(&words);
                let unique_concepts = self.count_unique_semantic_concepts(&content_words);

                if !words.is_empty() {
                    unique_concepts as f64 / words.len() as f64
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Analyze focus structure
    fn analyze_focus_structure(&self, sentences: &[String]) -> FocusStructure {
        let mut focus_points = Vec::new();

        for (i, sentence) in sentences.iter().enumerate() {
            let words = self.tokenize_sentence(sentence);
            let content_words = self.filter_content_words(&words);

            // Simplified focus identification
            if !content_words.is_empty() {
                let focus_point = FocusPoint {
                    content: content_words.join(" "),
                    strength: content_words.len() as f64 / words.len().max(1) as f64,
                    location: i,
                    focus_type: FocusType::NewInformation, // Simplified
                };
                focus_points.push(focus_point);
            }
        }

        let transition_smoothness = if focus_points.len() > 1 {
            let strength_differences: Vec<f64> = focus_points
                .windows(2)
                .map(|pair| (pair[1].strength - pair[0].strength).abs())
                .collect();

            let avg_difference =
                strength_differences.iter().sum::<f64>() / strength_differences.len() as f64;
            1.0 - avg_difference.min(1.0)
        } else {
            1.0
        };

        FocusStructure {
            focus_points,
            transition_smoothness,
            hierarchy_depth: 2, // Simplified
        }
    }

    /// Tokenize sentence into words
    pub fn tokenize_sentence(&self, sentence: &str) -> Vec<String> {
        sentence
            .split_whitespace()
            .map(|word| {
                word.trim_matches(|c: char| !c.is_alphabetic())
                    .to_lowercase()
            })
            .filter(|word| !word.is_empty())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_analyzer_creation() {
        let config = SemanticConfig::default();
        let analyzer = SemanticAnalyzer::new(config);

        assert_eq!(analyzer.config.coherence_weight, 0.25);
        assert!(!analyzer.semantic_networks.is_empty());
    }

    #[test]
    fn test_semantic_coherence() {
        let analyzer = SemanticAnalyzer::with_default_config();
        let sentences = vec![
            "The dog runs in the park.".to_string(),
            "The animal moves quickly through the green space.".to_string(),
        ];

        let coherence = analyzer.calculate_semantic_coherence(&sentences);
        assert!(coherence >= 0.0);
        assert!(coherence <= 1.0);
    }

    #[test]
    fn test_semantic_overlap() {
        let analyzer = SemanticAnalyzer::with_default_config();
        let words1 = vec!["dog".to_string(), "cat".to_string()];
        let words2 = vec!["animal".to_string(), "pet".to_string()];

        let overlap = analyzer.calculate_semantic_overlap(&words1, &words2);
        assert!(overlap >= 0.0);
        assert!(overlap <= 1.0);
    }

    #[test]
    fn test_meaning_preservation() {
        let analyzer = SemanticAnalyzer::with_default_config();
        let sentences = vec!["The quick brown fox jumps.".to_string()];

        let preservation = analyzer.calculate_meaning_preservation(&sentences);
        assert!(preservation >= 0.0);
        assert!(preservation <= 1.0);
    }

    #[test]
    fn test_conceptual_clarity() {
        let analyzer = SemanticAnalyzer::with_default_config();
        let sentences = vec![
            "The concrete table is sturdy.".to_string(),
            "The abstract concept is complex.".to_string(),
        ];

        let clarity = analyzer.calculate_conceptual_clarity(&sentences);
        assert!(clarity >= 0.0);
        assert!(clarity <= 1.0);
    }

    #[test]
    fn test_semantic_relations() {
        let analyzer = SemanticAnalyzer::with_default_config();
        let sentences = vec!["The happy joyful dog runs quickly.".to_string()];

        let relations = analyzer.analyze_semantic_relations(&sentences);
        assert!(relations.contains_key("synonymy"));
        assert!(relations.contains_key("antonymy"));
        assert!(relations.contains_key("hyponymy"));
        assert!(relations.contains_key("meronymy"));
    }

    #[test]
    fn test_semantic_fluency_analysis() {
        let analyzer = SemanticAnalyzer::with_default_config();
        let sentences = vec![
            "The dog runs happily in the park.".to_string(),
            "The animal enjoys the green outdoor space.".to_string(),
        ];

        let result = analyzer.analyze_semantic_fluency(&sentences);
        assert!(result.semantic_coherence >= 0.0);
        assert!(result.meaning_preservation >= 0.0);
        assert!(result.conceptual_clarity >= 0.0);
        assert!(result.semantic_appropriateness >= 0.0);
        assert!(!result.semantic_relations.is_empty());
    }

    #[test]
    fn test_ambiguity_scoring() {
        let analyzer = SemanticAnalyzer::with_default_config();
        let sentences = vec![
            "The bank is near the river.".to_string(), // Contains ambiguous word "bank"
            "The clear message is understood.".to_string(),
        ];

        let ambiguity_score = analyzer.calculate_ambiguity_score(&sentences);
        assert!(ambiguity_score >= 0.0);
        assert!(ambiguity_score <= 1.0);
    }

    #[test]
    fn test_context_sensitivity() {
        let analyzer = SemanticAnalyzer::with_default_config();
        let sentences = vec![
            "The dog barks loudly.".to_string(),
            "The animal makes noise.".to_string(),
            "The sound echoes.".to_string(),
        ];

        let sensitivity = analyzer.calculate_context_sensitivity(&sentences);
        assert!(sensitivity >= 0.0);
        assert!(sensitivity <= 1.0);
    }

    #[test]
    fn test_semantic_density() {
        let analyzer = SemanticAnalyzer::with_default_config();
        let sentences = vec!["The red car drives fast.".to_string()];

        let density = analyzer.calculate_semantic_density(&sentences);
        assert!(density >= 0.0);
        assert!(density <= 1.0);
    }
}
