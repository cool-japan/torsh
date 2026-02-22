use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rng, Random};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone)]
pub struct EntityCoherenceAnalyzer {
    entity_recognition_threshold: f64,
    coreference_threshold: f64,
    salience_decay_factor: f64,
    max_chain_distance: usize,
    entity_patterns: HashMap<String, f64>,
    grammatical_role_weights: HashMap<GrammaticalRole, f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntityCoherenceResult {
    pub overall_coherence: f64,
    pub entity_grid_coherence: f64,
    pub entity_transition_coherence: f64,
    pub coreference_coherence: f64,
    pub entity_density: f64,
    pub entity_chains: Vec<EntityChain>,
    pub dominant_entities: Vec<String>,
    pub entity_distribution: HashMap<String, usize>,
    pub salience_score: f64,
    pub detailed_metrics: DetailedEntityMetrics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DetailedEntityMetrics {
    pub entity_grid_analysis: EntityGridAnalysis,
    pub coreference_analysis: CoreferenceAnalysis,
    pub salience_analysis: SalienceAnalysis,
    pub coherence_patterns: CoherencePatterns,
    pub entity_interactions: EntityInteractions,
    pub temporal_coherence: TemporalEntityCoherence,
    pub advanced_metrics: AdvancedEntityMetrics,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntityGridAnalysis {
    pub grid_matrix: Array2<i32>,
    pub entity_names: Vec<String>,
    pub sentence_indices: Vec<usize>,
    pub transition_patterns: HashMap<String, f64>,
    pub syntactic_role_distributions: HashMap<GrammaticalRole, f64>,
    pub coherence_violations: Vec<CoherenceViolation>,
    pub grid_density: f64,
    pub role_consistency: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoreferenceAnalysis {
    pub coreference_chains: Vec<CoreferenceChain>,
    pub chain_coherence_scores: Vec<f64>,
    pub resolution_accuracy: f64,
    pub antecedent_distances: Array1<f64>,
    pub pronoun_resolution_quality: f64,
    pub bridging_anaphora_analysis: BridgingAnaphoraAnalysis,
    pub coreference_density: f64,
    pub ambiguity_resolution: AmbiguityResolution,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SalienceAnalysis {
    pub entity_salience_scores: HashMap<String, f64>,
    pub salience_distribution: Array1<f64>,
    pub topic_continuity: f64,
    pub focus_tracking: FocusTracking,
    pub salience_transitions: Vec<SalienceTransition>,
    pub global_vs_local_salience: HashMap<String, (f64, f64)>,
    pub salience_hierarchy: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoherencePatterns {
    pub centering_patterns: CenteringPatterns,
    pub entity_persistence: EntityPersistence,
    pub referential_patterns: ReferentialPatterns,
    pub discourse_new_vs_given: DiscourseNewGiven,
    pub information_status: InformationStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntityInteractions {
    pub entity_relationships: HashMap<String, HashMap<String, RelationshipType>>,
    pub interaction_matrix: Array2<f64>,
    pub collaborative_coherence: f64,
    pub entity_conflict_resolution: f64,
    pub semantic_role_consistency: f64,
    pub interaction_patterns: Vec<InteractionPattern>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TemporalEntityCoherence {
    pub temporal_entity_tracking: Vec<TemporalEntityState>,
    pub temporal_consistency: f64,
    pub timeline_coherence: f64,
    pub temporal_reference_resolution: f64,
    pub event_participant_coherence: f64,
    pub temporal_salience_changes: Array1<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdvancedEntityMetrics {
    pub entity_coherence_entropy: f64,
    pub information_theoretic_coherence: f64,
    pub graph_coherence_metrics: GraphCoherenceMetrics,
    pub cognitive_load_assessment: CognitiveLoadAssessment,
    pub predictive_coherence: PredictiveCoherence,
    pub multi_dimensional_coherence: MultiDimensionalCoherence,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntityChain {
    pub entity: String,
    pub mentions: Vec<EntityMention>,
    pub coherence_score: f64,
    pub salience: f64,
    pub role_consistency: f64,
    pub chain_type: EntityChainType,
    pub coherence_trajectory: Array1<f64>,
    pub semantic_consistency: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntityMention {
    pub sentence_index: usize,
    pub position: usize,
    pub mention_type: EntityMentionType,
    pub grammatical_role: GrammaticalRole,
    pub mention_text: String,
    pub confidence: f64,
    pub accessibility_score: f64,
    pub context: MentionContext,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EntityMentionType {
    Proper,
    Definite,
    Indefinite,
    Pronoun,
    Demonstrative,
    Zero, // Zero anaphora
    Epithet,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GrammaticalRole {
    Subject,
    Object,
    IndirectObject,
    PrepositionalObject,
    Possessor,
    Complement,
    Adjunct,
    Other,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EntityChainType {
    Protagonist,
    Supporting,
    Background,
    Temporary,
    Thematic,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MentionContext {
    pub surrounding_words: Vec<String>,
    pub semantic_context: String,
    pub discourse_context: String,
    pub syntactic_context: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoherenceViolation {
    pub violation_type: ViolationType,
    pub severity: f64,
    pub location: (usize, usize), // (sentence_index, position)
    pub description: String,
    pub suggested_fix: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ViolationType {
    UnexpectedRoleShift,
    MissingAntecedent,
    AmbiguousReference,
    InconsistentReference,
    SalienceViolation,
    AccessibilityViolation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoreferenceChain {
    pub chain_id: usize,
    pub mentions: Vec<EntityMention>,
    pub head_mention: Option<usize>, // Index into mentions
    pub chain_coherence: f64,
    pub resolution_confidence: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BridgingAnaphoraAnalysis {
    pub bridging_references: Vec<BridgingReference>,
    pub bridging_coherence: f64,
    pub inference_complexity: f64,
    pub resolution_success_rate: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BridgingReference {
    pub anaphor: EntityMention,
    pub anchor: EntityMention,
    pub relationship: BridgingRelationType,
    pub inference_strength: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BridgingRelationType {
    PartWhole,
    SetMember,
    Causal,
    Spatial,
    Temporal,
    Conceptual,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AmbiguityResolution {
    pub ambiguous_references: Vec<AmbiguousReference>,
    pub resolution_strategies: Vec<ResolutionStrategy>,
    pub ambiguity_impact_on_coherence: f64,
    pub resolution_success_rate: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AmbiguousReference {
    pub mention: EntityMention,
    pub possible_antecedents: Vec<EntityMention>,
    pub disambiguation_features: Vec<DisambiguationFeature>,
    pub resolution_confidence: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DisambiguationFeature {
    GrammaticalAgreement,
    SemanticCompatibility,
    Recency,
    Salience,
    SyntacticParallelism,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResolutionStrategy {
    RecencyPreference,
    SaliencePreference,
    SyntacticParallelism,
    SemanticCompatibility,
    HeuristicCombination,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FocusTracking {
    pub focus_entities: Vec<FocusEntity>,
    pub focus_transitions: Vec<FocusTransition>,
    pub focus_coherence: f64,
    pub attention_management: AttentionManagement,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FocusEntity {
    pub entity: String,
    pub focus_level: FocusLevel,
    pub focus_duration: usize,
    pub focus_strength: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FocusLevel {
    Primary,
    Secondary,
    Background,
    Peripheral,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FocusTransition {
    pub from_entity: String,
    pub to_entity: String,
    pub transition_type: FocusTransitionType,
    pub transition_smoothness: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FocusTransitionType {
    Continue,
    Shift,
    Return,
    Establish,
    Maintain,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionManagement {
    pub attention_distribution: HashMap<String, f64>,
    pub cognitive_load_score: f64,
    pub attention_coherence: f64,
    pub focus_management_quality: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SalienceTransition {
    pub entity: String,
    pub from_salience: f64,
    pub to_salience: f64,
    pub transition_cause: SalienceTransitionCause,
    pub transition_smoothness: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SalienceTransitionCause {
    NewMention,
    RoleChange,
    TopicShift,
    DiscourseMarker,
    SemanticContent,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CenteringPatterns {
    pub centering_transitions: Vec<CenteringTransition>,
    pub preferred_center_tracking: Vec<String>,
    pub backward_looking_center: Vec<Option<String>>,
    pub forward_looking_centers: Vec<Vec<String>>,
    pub centering_coherence: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CenteringTransition {
    pub transition_type: CenteringTransitionType,
    pub coherence_cost: f64,
    pub sentence_pair: (usize, usize),
}

#[derive(Debug, Clone, PartialEq)]
pub enum CenteringTransitionType {
    Continue,
    Retain,
    SmoothShift,
    RoughShift,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntityPersistence {
    pub persistence_patterns: HashMap<String, PersistencePattern>,
    pub average_persistence: f64,
    pub persistence_coherence: f64,
    pub dropout_patterns: Vec<DropoutPattern>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PersistencePattern {
    pub entity: String,
    pub appearances: Vec<usize>, // Sentence indices
    pub persistence_score: f64,
    pub gap_analysis: Vec<usize>, // Gaps between appearances
}

#[derive(Debug, Clone, PartialEq)]
pub struct DropoutPattern {
    pub entity: String,
    pub last_mention: usize,
    pub predicted_relevance: f64,
    pub dropout_impact: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReferentialPatterns {
    pub reference_chains: Vec<ReferenceChain>,
    pub referential_density: f64,
    pub reference_complexity: f64,
    pub anaphoric_coherence: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceChain {
    pub chain_elements: Vec<ReferenceElement>,
    pub chain_coherence: f64,
    pub reference_types: HashMap<EntityMentionType, usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReferenceElement {
    pub mention: EntityMention,
    pub reference_distance: usize,
    pub accessibility_cost: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiscourseNewGiven {
    pub new_entities: Vec<String>,
    pub given_entities: Vec<String>,
    pub new_given_balance: f64,
    pub information_flow_coherence: f64,
    pub given_new_transitions: Vec<GivenNewTransition>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GivenNewTransition {
    pub from_status: InformationStatus,
    pub to_status: InformationStatus,
    pub transition_appropriateness: f64,
    pub entity: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InformationStatus {
    New,
    Given,
    Accessible,
    Inferrable,
    Anchored,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RelationshipType {
    Coreference,
    PartOf,
    Causation,
    Temporal,
    Spatial,
    Social,
    Possession,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InteractionPattern {
    pub entities: Vec<String>,
    pub interaction_type: InteractionType,
    pub coherence_contribution: f64,
    pub temporal_span: (usize, usize),
}

#[derive(Debug, Clone, PartialEq)]
pub enum InteractionType {
    Collaboration,
    Conflict,
    Dependency,
    Parallel,
    Sequential,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TemporalEntityState {
    pub entity: String,
    pub temporal_location: usize,
    pub state_properties: HashMap<String, String>,
    pub coherence_with_previous: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphCoherenceMetrics {
    pub entity_graph_connectivity: f64,
    pub graph_clustering_coefficient: f64,
    pub centrality_measures: HashMap<String, f64>,
    pub graph_coherence_score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CognitiveLoadAssessment {
    pub working_memory_load: f64,
    pub processing_complexity: f64,
    pub reference_resolution_cost: f64,
    pub overall_cognitive_load: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PredictiveCoherence {
    pub predictability_scores: Array1<f64>,
    pub surprise_measures: Array1<f64>,
    pub expectation_violations: Vec<ExpectationViolation>,
    pub predictive_accuracy: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpectationViolation {
    pub location: usize,
    pub violation_type: String,
    pub severity: f64,
    pub impact_on_coherence: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiDimensionalCoherence {
    pub syntactic_coherence: f64,
    pub semantic_coherence: f64,
    pub pragmatic_coherence: f64,
    pub discourse_coherence: f64,
    pub integrated_coherence: f64,
}

#[derive(Debug)]
pub enum EntityCoherenceError {
    InsufficientEntities(String),
    ParsingError(String),
    ChainConstructionError(String),
    CoherenceCalculationError(String),
    InvalidConfiguration(String),
}

impl fmt::Display for EntityCoherenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EntityCoherenceError::InsufficientEntities(msg) => {
                write!(f, "Insufficient entities: {}", msg)
            }
            EntityCoherenceError::ParsingError(msg) => write!(f, "Parsing error: {}", msg),
            EntityCoherenceError::ChainConstructionError(msg) => {
                write!(f, "Chain construction error: {}", msg)
            }
            EntityCoherenceError::CoherenceCalculationError(msg) => {
                write!(f, "Coherence calculation error: {}", msg)
            }
            EntityCoherenceError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
        }
    }
}

impl Error for EntityCoherenceError {}

impl Default for EntityCoherenceAnalyzer {
    fn default() -> Self {
        let mut entity_patterns = HashMap::new();
        entity_patterns.insert("person".to_string(), 1.0);
        entity_patterns.insert("organization".to_string(), 0.8);
        entity_patterns.insert("location".to_string(), 0.7);
        entity_patterns.insert("event".to_string(), 0.6);

        let mut grammatical_role_weights = HashMap::new();
        grammatical_role_weights.insert(GrammaticalRole::Subject, 1.0);
        grammatical_role_weights.insert(GrammaticalRole::Object, 0.8);
        grammatical_role_weights.insert(GrammaticalRole::IndirectObject, 0.6);
        grammatical_role_weights.insert(GrammaticalRole::PrepositionalObject, 0.5);
        grammatical_role_weights.insert(GrammaticalRole::Possessor, 0.4);
        grammatical_role_weights.insert(GrammaticalRole::Complement, 0.3);
        grammatical_role_weights.insert(GrammaticalRole::Adjunct, 0.2);
        grammatical_role_weights.insert(GrammaticalRole::Other, 0.1);

        Self {
            entity_recognition_threshold: 0.7,
            coreference_threshold: 0.8,
            salience_decay_factor: 0.9,
            max_chain_distance: 10,
            entity_patterns,
            grammatical_role_weights,
        }
    }
}

impl EntityCoherenceAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_thresholds(
        mut self,
        recognition_threshold: f64,
        coreference_threshold: f64,
    ) -> Self {
        self.entity_recognition_threshold = recognition_threshold;
        self.coreference_threshold = coreference_threshold;
        self
    }

    pub fn with_salience_decay(mut self, decay_factor: f64) -> Self {
        self.salience_decay_factor = decay_factor;
        self
    }

    pub fn analyze_entity_coherence(
        &self,
        text: &str,
    ) -> Result<EntityCoherenceResult, EntityCoherenceError> {
        let sentences = self.split_into_sentences(text);
        if sentences.len() < 2 {
            return Err(EntityCoherenceError::InsufficientEntities(
                "Need at least 2 sentences for coherence analysis".to_string(),
            ));
        }

        // Extract entities and build chains
        let entity_chains = self.build_entity_chains(&sentences)?;
        if entity_chains.is_empty() {
            return Err(EntityCoherenceError::InsufficientEntities(
                "No entity chains found".to_string(),
            ));
        }

        // Perform comprehensive analysis
        let entity_grid_analysis = self.analyze_entity_grid(&sentences, &entity_chains)?;
        let coreference_analysis = self.analyze_coreference(&entity_chains)?;
        let salience_analysis = self.analyze_salience(&entity_chains, sentences.len())?;
        let coherence_patterns = self.analyze_coherence_patterns(&entity_chains, &sentences)?;
        let entity_interactions = self.analyze_entity_interactions(&entity_chains)?;
        let temporal_coherence =
            self.analyze_temporal_entity_coherence(&entity_chains, &sentences)?;
        let advanced_metrics =
            self.calculate_advanced_entity_metrics(&entity_chains, &sentences)?;

        // Calculate overall coherence scores
        let entity_grid_coherence = entity_grid_analysis.role_consistency;
        let entity_transition_coherence = self.calculate_transition_coherence(&entity_chains)?;
        let coreference_coherence = coreference_analysis.resolution_accuracy;
        let entity_density = self.calculate_entity_density(&entity_chains, sentences.len());
        let salience_score = salience_analysis.topic_continuity;

        let overall_coherence = self.calculate_overall_entity_coherence(
            entity_grid_coherence,
            entity_transition_coherence,
            coreference_coherence,
            salience_score,
        );

        // Build result statistics
        let dominant_entities = self.identify_dominant_entities(&entity_chains);
        let entity_distribution = self.calculate_entity_distribution(&entity_chains);

        let detailed_metrics = DetailedEntityMetrics {
            entity_grid_analysis,
            coreference_analysis,
            salience_analysis,
            coherence_patterns,
            entity_interactions,
            temporal_coherence,
            advanced_metrics,
        };

        Ok(EntityCoherenceResult {
            overall_coherence,
            entity_grid_coherence,
            entity_transition_coherence,
            coreference_coherence,
            entity_density,
            entity_chains,
            dominant_entities,
            entity_distribution,
            salience_score,
            detailed_metrics,
        })
    }

    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        text.split('.')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_string())
            .collect()
    }

    fn build_entity_chains(
        &self,
        sentences: &[String],
    ) -> Result<Vec<EntityChain>, EntityCoherenceError> {
        let mut entity_chains = Vec::new();
        let mut entity_mentions_map: HashMap<String, Vec<EntityMention>> = HashMap::new();

        // Extract mentions from each sentence
        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let mentions = self.extract_entity_mentions(sentence, sent_idx)?;
            for mention in mentions {
                let key = self.normalize_entity(&mention.mention_text);
                entity_mentions_map
                    .entry(key)
                    .or_insert_with(Vec::new)
                    .push(mention);
            }
        }

        // Build chains from mentions
        for (entity, mentions) in entity_mentions_map {
            if mentions.len() >= 2 {
                // Only consider entities with multiple mentions
                let chain = self.construct_entity_chain(entity, mentions)?;
                entity_chains.push(chain);
            }
        }

        // Sort chains by salience
        entity_chains.sort_by(|a, b| {
            b.salience
                .partial_cmp(&a.salience)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(entity_chains)
    }

    fn extract_entity_mentions(
        &self,
        sentence: &str,
        sentence_index: usize,
    ) -> Result<Vec<EntityMention>, EntityCoherenceError> {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        let mut mentions = Vec::new();

        for (pos, word) in words.iter().enumerate() {
            if self.is_potential_entity(word) {
                let mention_type = self.classify_mention_type(word, &words, pos);
                let grammatical_role = self.determine_grammatical_role(word, &words, pos);
                let confidence =
                    self.calculate_mention_confidence(word, &mention_type, &grammatical_role);

                if confidence >= self.entity_recognition_threshold {
                    let context = self.extract_mention_context(&words, pos);
                    let accessibility_score =
                        self.calculate_accessibility_score(&mention_type, &grammatical_role, pos);

                    mentions.push(EntityMention {
                        sentence_index,
                        position: pos,
                        mention_type,
                        grammatical_role,
                        mention_text: word.to_string(),
                        confidence,
                        accessibility_score,
                        context,
                    });
                }
            }
        }

        Ok(mentions)
    }

    fn is_potential_entity(&self, word: &str) -> bool {
        // Simple heuristics - in practice would use NER
        let first_char = word.chars().next().unwrap_or(' ');
        first_char.is_uppercase()
            || word.contains("he")
            || word.contains("she")
            || word.contains("it")
            || word.contains("they")
            || word.contains("this")
            || word.contains("that")
    }

    fn classify_mention_type(&self, word: &str, words: &[&str], pos: usize) -> EntityMentionType {
        if word.chars().next().unwrap_or(' ').is_uppercase() {
            EntityMentionType::Proper
        } else if word == "he" || word == "she" || word == "it" || word == "they" {
            EntityMentionType::Pronoun
        } else if word == "this" || word == "that" || word == "these" || word == "those" {
            EntityMentionType::Demonstrative
        } else if pos > 0 && words[pos - 1] == "the" {
            EntityMentionType::Definite
        } else if pos > 0 && (words[pos - 1] == "a" || words[pos - 1] == "an") {
            EntityMentionType::Indefinite
        } else {
            EntityMentionType::Definite // Default
        }
    }

    fn determine_grammatical_role(
        &self,
        word: &str,
        words: &[&str],
        pos: usize,
    ) -> GrammaticalRole {
        // Simplified grammatical role detection
        if pos == 0 || (pos > 0 && words[pos - 1] == ".") {
            GrammaticalRole::Subject
        } else if pos > 0 && self.is_verb(words[pos - 1]) {
            GrammaticalRole::Object
        } else if pos > 0 && self.is_preposition(words[pos - 1]) {
            GrammaticalRole::PrepositionalObject
        } else if pos > 0 && words[pos - 1] == "'s" {
            GrammaticalRole::Possessor
        } else {
            GrammaticalRole::Other
        }
    }

    fn is_verb(&self, word: &str) -> bool {
        // Simplified verb detection
        word.ends_with("ed")
            || word.ends_with("ing")
            || [
                "is", "are", "was", "were", "have", "has", "had", "do", "does", "did",
            ]
            .contains(&word)
    }

    fn is_preposition(&self, word: &str) -> bool {
        [
            "of", "in", "on", "at", "by", "for", "with", "to", "from", "about",
        ]
        .contains(&word)
    }

    fn calculate_mention_confidence(
        &self,
        word: &str,
        mention_type: &EntityMentionType,
        grammatical_role: &GrammaticalRole,
    ) -> f64 {
        let mut confidence = 0.5;

        // Boost confidence for proper nouns
        if matches!(mention_type, EntityMentionType::Proper) {
            confidence += 0.3;
        }

        // Boost confidence for subject position
        if matches!(grammatical_role, GrammaticalRole::Subject) {
            confidence += 0.2;
        }

        // Apply role weights
        if let Some(weight) = self.grammatical_role_weights.get(grammatical_role) {
            confidence *= weight;
        }

        confidence.min(1.0)
    }

    fn extract_mention_context(&self, words: &[&str], pos: usize) -> MentionContext {
        let start = pos.saturating_sub(2);
        let end = (pos + 3).min(words.len());
        let surrounding_words: Vec<String> =
            words[start..end].iter().map(|s| s.to_string()).collect();

        MentionContext {
            surrounding_words,
            semantic_context: "general".to_string(), // Placeholder
            discourse_context: "narrative".to_string(), // Placeholder
            syntactic_context: "sentence".to_string(), // Placeholder
        }
    }

    fn calculate_accessibility_score(
        &self,
        mention_type: &EntityMentionType,
        grammatical_role: &GrammaticalRole,
        position: usize,
    ) -> f64 {
        let mut score = 0.5;

        // Different mention types have different accessibility
        score += match mention_type {
            EntityMentionType::Pronoun => 0.3,
            EntityMentionType::Definite => 0.2,
            EntityMentionType::Demonstrative => 0.2,
            EntityMentionType::Proper => 0.1,
            EntityMentionType::Indefinite => -0.1,
            _ => 0.0,
        };

        // Subject position is more accessible
        if matches!(grammatical_role, GrammaticalRole::Subject) {
            score += 0.2;
        }

        // Earlier positions are more accessible (recency effect)
        score += 0.1 / (position as f64 + 1.0);

        score.max(0.0).min(1.0)
    }

    fn normalize_entity(&self, entity: &str) -> String {
        entity.to_lowercase().trim().to_string()
    }

    fn construct_entity_chain(
        &self,
        entity: String,
        mut mentions: Vec<EntityMention>,
    ) -> Result<EntityChain, EntityCoherenceError> {
        // Sort mentions by sentence index
        mentions.sort_by_key(|m| m.sentence_index);

        // Calculate chain coherence
        let coherence_score = self.calculate_chain_coherence(&mentions)?;

        // Calculate salience based on mention frequency and positions
        let salience = self.calculate_entity_salience(&mentions);

        // Calculate role consistency
        let role_consistency = self.calculate_role_consistency(&mentions);

        // Determine chain type
        let chain_type = self.classify_chain_type(&mentions, salience);

        // Calculate coherence trajectory
        let coherence_trajectory = self.calculate_coherence_trajectory(&mentions)?;

        // Calculate semantic consistency
        let semantic_consistency = self.calculate_semantic_consistency(&mentions);

        Ok(EntityChain {
            entity,
            mentions,
            coherence_score,
            salience,
            role_consistency,
            chain_type,
            coherence_trajectory,
            semantic_consistency,
        })
    }

    fn calculate_chain_coherence(
        &self,
        mentions: &[EntityMention],
    ) -> Result<f64, EntityCoherenceError> {
        if mentions.len() < 2 {
            return Ok(0.0);
        }

        let mut coherence_sum = 0.0;
        let mut pair_count = 0;

        for i in 1..mentions.len() {
            let distance = mentions[i].sentence_index - mentions[i - 1].sentence_index;
            let distance_penalty = (-0.1 * distance as f64).exp();

            let role_consistency =
                if mentions[i].grammatical_role == mentions[i - 1].grammatical_role {
                    1.0
                } else {
                    0.5
                };

            let type_coherence = self.calculate_mention_type_coherence(
                &mentions[i - 1].mention_type,
                &mentions[i].mention_type,
            );

            coherence_sum += distance_penalty * role_consistency * type_coherence;
            pair_count += 1;
        }

        Ok(coherence_sum / pair_count as f64)
    }

    fn calculate_mention_type_coherence(
        &self,
        type1: &EntityMentionType,
        type2: &EntityMentionType,
    ) -> f64 {
        match (type1, type2) {
            (EntityMentionType::Proper, EntityMentionType::Pronoun) => 0.9,
            (EntityMentionType::Definite, EntityMentionType::Pronoun) => 0.8,
            (EntityMentionType::Pronoun, EntityMentionType::Pronoun) => 0.7,
            (a, b) if a == b => 1.0,
            _ => 0.5,
        }
    }

    fn calculate_entity_salience(&self, mentions: &[EntityMention]) -> f64 {
        let mut salience = 0.0;
        let total_mentions = mentions.len() as f64;

        for mention in mentions {
            // Subject mentions are more salient
            let role_weight = self
                .grammatical_role_weights
                .get(&mention.grammatical_role)
                .unwrap_or(&0.1);

            // First mentions are more salient
            let position_weight = 1.0 / (mention.sentence_index as f64 + 1.0);

            salience += role_weight * position_weight * mention.confidence;
        }

        // Normalize by number of mentions
        (salience / total_mentions).min(1.0)
    }

    fn calculate_role_consistency(&self, mentions: &[EntityMention]) -> f64 {
        if mentions.len() < 2 {
            return 1.0;
        }

        let primary_role = &mentions[0].grammatical_role;
        let consistent_roles = mentions
            .iter()
            .filter(|m| &m.grammatical_role == primary_role)
            .count();
        consistent_roles as f64 / mentions.len() as f64
    }

    fn classify_chain_type(&self, mentions: &[EntityMention], salience: f64) -> EntityChainType {
        if salience > 0.8 && mentions.len() > 5 {
            EntityChainType::Protagonist
        } else if salience > 0.6 {
            EntityChainType::Supporting
        } else if mentions.len() < 3 {
            EntityChainType::Temporary
        } else {
            EntityChainType::Background
        }
    }

    fn calculate_coherence_trajectory(
        &self,
        mentions: &[EntityMention],
    ) -> Result<Array1<f64>, EntityCoherenceError> {
        let mut trajectory = Array1::<f64>::zeros(mentions.len());

        for i in 0..mentions.len() {
            let local_coherence = if i == 0 {
                mentions[i].accessibility_score
            } else {
                let distance = mentions[i].sentence_index - mentions[i - 1].sentence_index;
                let distance_factor = (-0.1 * distance as f64).exp();
                distance_factor * mentions[i].accessibility_score
            };
            trajectory[i] = local_coherence;
        }

        Ok(trajectory)
    }

    fn calculate_semantic_consistency(&self, mentions: &[EntityMention]) -> f64 {
        // Simplified semantic consistency based on context similarity
        if mentions.len() < 2 {
            return 1.0;
        }

        let mut consistency_sum = 0.0;
        let mut pair_count = 0;

        for i in 1..mentions.len() {
            let context_similarity =
                self.calculate_context_similarity(&mentions[i - 1].context, &mentions[i].context);
            consistency_sum += context_similarity;
            pair_count += 1;
        }

        consistency_sum / pair_count as f64
    }

    fn calculate_context_similarity(
        &self,
        context1: &MentionContext,
        context2: &MentionContext,
    ) -> f64 {
        // Simple word overlap measure
        let words1: HashSet<String> = context1.surrounding_words.iter().cloned().collect();
        let words2: HashSet<String> = context2.surrounding_words.iter().cloned().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    // Additional analysis methods (simplified implementations)

    fn analyze_entity_grid(
        &self,
        sentences: &[String],
        entity_chains: &[EntityChain],
    ) -> Result<EntityGridAnalysis, EntityCoherenceError> {
        let n_sentences = sentences.len();
        let n_entities = entity_chains.len().min(10); // Limit for performance

        let mut grid_matrix = Array2::<i32>::zeros((n_entities, n_sentences));
        let mut entity_names = Vec::new();

        // Build the grid
        for (entity_idx, chain) in entity_chains.iter().take(n_entities).enumerate() {
            entity_names.push(chain.entity.clone());
            for mention in &chain.mentions {
                if mention.sentence_index < n_sentences {
                    grid_matrix[[entity_idx, mention.sentence_index]] =
                        self.encode_grammatical_role(&mention.grammatical_role);
                }
            }
        }

        let sentence_indices: Vec<usize> = (0..n_sentences).collect();
        let transition_patterns = self.calculate_grid_transition_patterns(&grid_matrix)?;
        let syntactic_role_distributions = self.calculate_role_distributions(entity_chains);
        let coherence_violations = self.detect_coherence_violations(&grid_matrix, &entity_names)?;
        let grid_density = self.calculate_grid_density(&grid_matrix);
        let role_consistency = self.calculate_grid_role_consistency(&grid_matrix);

        Ok(EntityGridAnalysis {
            grid_matrix,
            entity_names,
            sentence_indices,
            transition_patterns,
            syntactic_role_distributions,
            coherence_violations,
            grid_density,
            role_consistency,
        })
    }

    fn encode_grammatical_role(&self, role: &GrammaticalRole) -> i32 {
        match role {
            GrammaticalRole::Subject => 3,
            GrammaticalRole::Object => 2,
            GrammaticalRole::IndirectObject => 1,
            GrammaticalRole::PrepositionalObject => 1,
            _ => 0,
        }
    }

    fn calculate_grid_transition_patterns(
        &self,
        grid: &Array2<i32>,
    ) -> Result<HashMap<String, f64>, EntityCoherenceError> {
        let mut patterns = HashMap::new();
        let (n_entities, n_sentences) = grid.dim();

        for entity_idx in 0..n_entities {
            let mut transitions = Vec::new();
            for sent_idx in 1..n_sentences {
                let prev_role = grid[[entity_idx, sent_idx - 1]];
                let curr_role = grid[[entity_idx, sent_idx]];
                transitions.push((prev_role, curr_role));
            }

            // Count transition types
            let continue_transitions = transitions
                .iter()
                .filter(|(p, c)| *p > 0 && *c > 0 && p == c)
                .count();
            let shift_transitions = transitions
                .iter()
                .filter(|(p, c)| *p > 0 && *c > 0 && p != c)
                .count();

            if !transitions.is_empty() {
                patterns.insert(
                    format!("entity_{}_continue", entity_idx),
                    continue_transitions as f64 / transitions.len() as f64,
                );
                patterns.insert(
                    format!("entity_{}_shift", entity_idx),
                    shift_transitions as f64 / transitions.len() as f64,
                );
            }
        }

        Ok(patterns)
    }

    fn calculate_role_distributions(
        &self,
        entity_chains: &[EntityChain],
    ) -> HashMap<GrammaticalRole, f64> {
        let mut role_counts = HashMap::new();
        let mut total_mentions = 0;

        for chain in entity_chains {
            for mention in &chain.mentions {
                *role_counts
                    .entry(mention.grammatical_role.clone())
                    .or_insert(0) += 1;
                total_mentions += 1;
            }
        }

        role_counts
            .into_iter()
            .map(|(role, count)| (role, count as f64 / total_mentions as f64))
            .collect()
    }

    fn detect_coherence_violations(
        &self,
        grid: &Array2<i32>,
        entity_names: &[String],
    ) -> Result<Vec<CoherenceViolation>, EntityCoherenceError> {
        let mut violations = Vec::new();
        let (n_entities, n_sentences) = grid.dim();

        for entity_idx in 0..n_entities {
            for sent_idx in 1..n_sentences {
                let prev_role = grid[[entity_idx, sent_idx - 1]];
                let curr_role = grid[[entity_idx, sent_idx]];

                // Check for unexpected role shifts
                if prev_role == 3 && curr_role == 0 && sent_idx < n_sentences - 1 {
                    // Subject dropped unexpectedly
                    violations.push(CoherenceViolation {
                        violation_type: ViolationType::UnexpectedRoleShift,
                        severity: 0.7,
                        location: (sent_idx, entity_idx),
                        description: format!(
                            "Entity '{}' dropped as subject unexpectedly",
                            entity_names[entity_idx]
                        ),
                        suggested_fix: "Consider adding a pronoun reference".to_string(),
                    });
                }
            }
        }

        Ok(violations)
    }

    fn calculate_grid_density(&self, grid: &Array2<i32>) -> f64 {
        let total_cells = grid.len();
        let filled_cells = grid.iter().filter(|&&cell| cell > 0).count();
        filled_cells as f64 / total_cells as f64
    }

    fn calculate_grid_role_consistency(&self, grid: &Array2<i32>) -> f64 {
        let (n_entities, n_sentences) = grid.dim();
        let mut consistency_sum = 0.0;
        let mut consistency_count = 0;

        for entity_idx in 0..n_entities {
            let roles: Vec<i32> = (0..n_sentences)
                .map(|s| grid[[entity_idx, s]])
                .filter(|&r| r > 0)
                .collect();

            if roles.len() > 1 {
                let most_common_role = self.find_most_common_role(&roles);
                let consistent_roles = roles.iter().filter(|&&r| r == most_common_role).count();
                consistency_sum += consistent_roles as f64 / roles.len() as f64;
                consistency_count += 1;
            }
        }

        if consistency_count > 0 {
            consistency_sum / consistency_count as f64
        } else {
            1.0
        }
    }

    fn find_most_common_role(&self, roles: &[i32]) -> i32 {
        let mut counts = HashMap::new();
        for &role in roles {
            *counts.entry(role).or_insert(0) += 1;
        }
        *counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(role, _)| role)
            .unwrap_or(&0)
    }

    // Placeholder implementations for remaining complex methods
    fn analyze_coreference(
        &self,
        entity_chains: &[EntityChain],
    ) -> Result<CoreferenceAnalysis, EntityCoherenceError> {
        let mut coreference_chains = Vec::new();
        let mut chain_coherence_scores = Vec::new();

        for (i, chain) in entity_chains.iter().enumerate() {
            let coref_chain = CoreferenceChain {
                chain_id: i,
                mentions: chain.mentions.clone(),
                head_mention: Some(0),
                chain_coherence: chain.coherence_score,
                resolution_confidence: 0.8,
            };
            chain_coherence_scores.push(chain.coherence_score);
            coreference_chains.push(coref_chain);
        }

        let resolution_accuracy =
            chain_coherence_scores.iter().sum::<f64>() / chain_coherence_scores.len() as f64;
        let antecedent_distances = Array1::<f64>::from_vec(vec![2.0, 3.0, 1.0, 4.0]); // Placeholder

        Ok(CoreferenceAnalysis {
            coreference_chains,
            chain_coherence_scores,
            resolution_accuracy,
            antecedent_distances,
            pronoun_resolution_quality: 0.75,
            bridging_anaphora_analysis: BridgingAnaphoraAnalysis {
                bridging_references: Vec::new(),
                bridging_coherence: 0.6,
                inference_complexity: 0.4,
                resolution_success_rate: 0.7,
            },
            coreference_density: 0.3,
            ambiguity_resolution: AmbiguityResolution {
                ambiguous_references: Vec::new(),
                resolution_strategies: vec![ResolutionStrategy::SaliencePreference],
                ambiguity_impact_on_coherence: 0.1,
                resolution_success_rate: 0.8,
            },
        })
    }

    // Additional method implementations would continue with similar patterns...
    // For brevity, providing simplified placeholder implementations

    fn analyze_salience(
        &self,
        entity_chains: &[EntityChain],
        num_sentences: usize,
    ) -> Result<SalienceAnalysis, EntityCoherenceError> {
        let mut entity_salience_scores = HashMap::new();
        for chain in entity_chains {
            entity_salience_scores.insert(chain.entity.clone(), chain.salience);
        }

        let salience_values: Vec<f64> = entity_salience_scores.values().cloned().collect();
        let salience_distribution = Array1::from_vec(salience_values);

        Ok(SalienceAnalysis {
            entity_salience_scores,
            salience_distribution,
            topic_continuity: 0.8,
            focus_tracking: FocusTracking {
                focus_entities: Vec::new(),
                focus_transitions: Vec::new(),
                focus_coherence: 0.7,
                attention_management: AttentionManagement {
                    attention_distribution: HashMap::new(),
                    cognitive_load_score: 0.5,
                    attention_coherence: 0.6,
                    focus_management_quality: 0.7,
                },
            },
            salience_transitions: Vec::new(),
            global_vs_local_salience: HashMap::new(),
            salience_hierarchy: entity_chains.iter().map(|c| c.entity.clone()).collect(),
        })
    }

    fn analyze_coherence_patterns(
        &self,
        entity_chains: &[EntityChain],
        sentences: &[String],
    ) -> Result<CoherencePatterns, EntityCoherenceError> {
        Ok(CoherencePatterns {
            centering_patterns: CenteringPatterns {
                centering_transitions: Vec::new(),
                preferred_center_tracking: Vec::new(),
                backward_looking_center: Vec::new(),
                forward_looking_centers: Vec::new(),
                centering_coherence: 0.7,
            },
            entity_persistence: EntityPersistence {
                persistence_patterns: HashMap::new(),
                average_persistence: 0.6,
                persistence_coherence: 0.7,
                dropout_patterns: Vec::new(),
            },
            referential_patterns: ReferentialPatterns {
                reference_chains: Vec::new(),
                referential_density: 0.4,
                reference_complexity: 0.5,
                anaphoric_coherence: 0.8,
            },
            discourse_new_vs_given: DiscourseNewGiven {
                new_entities: Vec::new(),
                given_entities: Vec::new(),
                new_given_balance: 0.6,
                information_flow_coherence: 0.7,
                given_new_transitions: Vec::new(),
            },
            information_status: InformationStatus::Given,
        })
    }

    fn analyze_entity_interactions(
        &self,
        entity_chains: &[EntityChain],
    ) -> Result<EntityInteractions, EntityCoherenceError> {
        Ok(EntityInteractions {
            entity_relationships: HashMap::new(),
            interaction_matrix: Array2::<f64>::zeros((entity_chains.len(), entity_chains.len())),
            collaborative_coherence: 0.6,
            entity_conflict_resolution: 0.8,
            semantic_role_consistency: 0.7,
            interaction_patterns: Vec::new(),
        })
    }

    fn analyze_temporal_entity_coherence(
        &self,
        entity_chains: &[EntityChain],
        sentences: &[String],
    ) -> Result<TemporalEntityCoherence, EntityCoherenceError> {
        Ok(TemporalEntityCoherence {
            temporal_entity_tracking: Vec::new(),
            temporal_consistency: 0.8,
            timeline_coherence: 0.7,
            temporal_reference_resolution: 0.9,
            event_participant_coherence: 0.8,
            temporal_salience_changes: Array1::<f64>::zeros(sentences.len()),
        })
    }

    fn calculate_advanced_entity_metrics(
        &self,
        entity_chains: &[EntityChain],
        sentences: &[String],
    ) -> Result<AdvancedEntityMetrics, EntityCoherenceError> {
        Ok(AdvancedEntityMetrics {
            entity_coherence_entropy: 2.3,
            information_theoretic_coherence: 0.7,
            graph_coherence_metrics: GraphCoherenceMetrics {
                entity_graph_connectivity: 0.6,
                graph_clustering_coefficient: 0.4,
                centrality_measures: HashMap::new(),
                graph_coherence_score: 0.7,
            },
            cognitive_load_assessment: CognitiveLoadAssessment {
                working_memory_load: 0.5,
                processing_complexity: 0.6,
                reference_resolution_cost: 0.4,
                overall_cognitive_load: 0.5,
            },
            predictive_coherence: PredictiveCoherence {
                predictability_scores: Array1::<f64>::zeros(sentences.len()),
                surprise_measures: Array1::<f64>::zeros(sentences.len()),
                expectation_violations: Vec::new(),
                predictive_accuracy: 0.8,
            },
            multi_dimensional_coherence: MultiDimensionalCoherence {
                syntactic_coherence: 0.8,
                semantic_coherence: 0.7,
                pragmatic_coherence: 0.6,
                discourse_coherence: 0.8,
                integrated_coherence: 0.7,
            },
        })
    }

    fn calculate_transition_coherence(
        &self,
        entity_chains: &[EntityChain],
    ) -> Result<f64, EntityCoherenceError> {
        if entity_chains.is_empty() {
            return Ok(0.0);
        }

        let mut transition_sum = 0.0;
        let mut transition_count = 0;

        for chain in entity_chains {
            for i in 1..chain.mentions.len() {
                let distance =
                    chain.mentions[i].sentence_index - chain.mentions[i - 1].sentence_index;
                let transition_quality = (-0.1 * distance as f64).exp();
                transition_sum += transition_quality;
                transition_count += 1;
            }
        }

        Ok(if transition_count > 0 {
            transition_sum / transition_count as f64
        } else {
            0.0
        })
    }

    fn calculate_entity_density(&self, entity_chains: &[EntityChain], num_sentences: usize) -> f64 {
        let total_mentions: usize = entity_chains.iter().map(|c| c.mentions.len()).sum();
        total_mentions as f64 / num_sentences as f64
    }

    fn calculate_overall_entity_coherence(
        &self,
        grid_coherence: f64,
        transition_coherence: f64,
        coreference_coherence: f64,
        salience_score: f64,
    ) -> f64 {
        (grid_coherence * 0.3
            + transition_coherence * 0.25
            + coreference_coherence * 0.25
            + salience_score * 0.2)
            .min(1.0)
    }

    fn identify_dominant_entities(&self, entity_chains: &[EntityChain]) -> Vec<String> {
        let mut chains = entity_chains.to_vec();
        chains.sort_by(|a, b| {
            b.salience
                .partial_cmp(&a.salience)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        chains.into_iter().take(5).map(|c| c.entity).collect()
    }

    fn calculate_entity_distribution(
        &self,
        entity_chains: &[EntityChain],
    ) -> HashMap<String, usize> {
        entity_chains
            .iter()
            .map(|c| (c.entity.clone(), c.mentions.len()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_coherence_analyzer_creation() {
        let analyzer = EntityCoherenceAnalyzer::new();
        assert_eq!(analyzer.entity_recognition_threshold, 0.7);
        assert_eq!(analyzer.coreference_threshold, 0.8);
        assert!(!analyzer.entity_patterns.is_empty());
    }

    #[test]
    fn test_entity_coherence_analysis() {
        let analyzer = EntityCoherenceAnalyzer::new();
        let text = "John went to the store. He bought some milk. John returned home and gave the milk to Mary.";

        let result = analyzer.analyze_entity_coherence(text);
        assert!(result.is_ok());

        let coherence = result.unwrap();
        assert!(coherence.overall_coherence >= 0.0);
        assert!(coherence.overall_coherence <= 1.0);
        assert!(!coherence.entity_chains.is_empty());
        assert!(!coherence.dominant_entities.is_empty());
    }

    #[test]
    fn test_entity_mention_extraction() {
        let analyzer = EntityCoherenceAnalyzer::new();
        let sentence = "John bought a book from the store";

        let result = analyzer.extract_entity_mentions(sentence, 0);
        assert!(result.is_ok());

        let mentions = result.unwrap();
        assert!(!mentions.is_empty());
        assert!(mentions.iter().any(|m| m.mention_text == "John"));
    }

    #[test]
    fn test_mention_type_classification() {
        let analyzer = EntityCoherenceAnalyzer::new();
        let words = vec!["John", "went", "to", "the", "store"];

        let mention_type = analyzer.classify_mention_type("John", &words, 0);
        assert_eq!(mention_type, EntityMentionType::Proper);

        let words2 = vec!["He", "went", "home"];
        let mention_type2 = analyzer.classify_mention_type("He", &words2, 0);
        assert_eq!(mention_type2, EntityMentionType::Pronoun);
    }

    #[test]
    fn test_grammatical_role_determination() {
        let analyzer = EntityCoherenceAnalyzer::new();
        let words = vec!["John", "bought", "milk"];

        let role1 = analyzer.determine_grammatical_role("John", &words, 0);
        assert_eq!(role1, GrammaticalRole::Subject);

        let role2 = analyzer.determine_grammatical_role("milk", &words, 2);
        assert_eq!(role2, GrammaticalRole::Object);
    }

    #[test]
    fn test_entity_chain_construction() {
        let analyzer = EntityCoherenceAnalyzer::new();

        let mention1 = EntityMention {
            sentence_index: 0,
            position: 0,
            mention_type: EntityMentionType::Proper,
            grammatical_role: GrammaticalRole::Subject,
            mention_text: "John".to_string(),
            confidence: 0.9,
            accessibility_score: 0.8,
            context: MentionContext {
                surrounding_words: vec!["John".to_string(), "went".to_string()],
                semantic_context: "action".to_string(),
                discourse_context: "narrative".to_string(),
                syntactic_context: "main_clause".to_string(),
            },
        };

        let mention2 = EntityMention {
            sentence_index: 1,
            position: 0,
            mention_type: EntityMentionType::Pronoun,
            grammatical_role: GrammaticalRole::Subject,
            mention_text: "He".to_string(),
            confidence: 0.8,
            accessibility_score: 0.9,
            context: MentionContext {
                surrounding_words: vec!["He".to_string(), "bought".to_string()],
                semantic_context: "action".to_string(),
                discourse_context: "narrative".to_string(),
                syntactic_context: "main_clause".to_string(),
            },
        };

        let mentions = vec![mention1, mention2];
        let result = analyzer.construct_entity_chain("john".to_string(), mentions);

        assert!(result.is_ok());
        let chain = result.unwrap();
        assert_eq!(chain.entity, "john");
        assert_eq!(chain.mentions.len(), 2);
        assert!(chain.coherence_score > 0.0);
    }

    #[test]
    fn test_chain_coherence_calculation() {
        let analyzer = EntityCoherenceAnalyzer::new();

        let mentions = vec![
            EntityMention {
                sentence_index: 0,
                position: 0,
                mention_type: EntityMentionType::Proper,
                grammatical_role: GrammaticalRole::Subject,
                mention_text: "John".to_string(),
                confidence: 0.9,
                accessibility_score: 0.8,
                context: MentionContext {
                    surrounding_words: vec![],
                    semantic_context: "".to_string(),
                    discourse_context: "".to_string(),
                    syntactic_context: "".to_string(),
                },
            },
            EntityMention {
                sentence_index: 1,
                position: 0,
                mention_type: EntityMentionType::Pronoun,
                grammatical_role: GrammaticalRole::Subject,
                mention_text: "He".to_string(),
                confidence: 0.8,
                accessibility_score: 0.9,
                context: MentionContext {
                    surrounding_words: vec![],
                    semantic_context: "".to_string(),
                    discourse_context: "".to_string(),
                    syntactic_context: "".to_string(),
                },
            },
        ];

        let result = analyzer.calculate_chain_coherence(&mentions);
        assert!(result.is_ok());

        let coherence = result.unwrap();
        assert!(coherence > 0.0);
        assert!(coherence <= 1.0);
    }

    #[test]
    fn test_entity_salience_calculation() {
        let analyzer = EntityCoherenceAnalyzer::new();

        let mentions = vec![EntityMention {
            sentence_index: 0,
            position: 0,
            mention_type: EntityMentionType::Proper,
            grammatical_role: GrammaticalRole::Subject,
            mention_text: "John".to_string(),
            confidence: 0.9,
            accessibility_score: 0.8,
            context: MentionContext {
                surrounding_words: vec![],
                semantic_context: "".to_string(),
                discourse_context: "".to_string(),
                syntactic_context: "".to_string(),
            },
        }];

        let salience = analyzer.calculate_entity_salience(&mentions);
        assert!(salience > 0.0);
        assert!(salience <= 1.0);
    }

    #[test]
    fn test_mention_type_coherence() {
        let analyzer = EntityCoherenceAnalyzer::new();

        let coherence1 = analyzer.calculate_mention_type_coherence(
            &EntityMentionType::Proper,
            &EntityMentionType::Pronoun,
        );
        assert_eq!(coherence1, 0.9);

        let coherence2 = analyzer.calculate_mention_type_coherence(
            &EntityMentionType::Proper,
            &EntityMentionType::Proper,
        );
        assert_eq!(coherence2, 1.0);
    }

    #[test]
    fn test_insufficient_entities_error() {
        let analyzer = EntityCoherenceAnalyzer::new();
        let text = "Short.";

        let result = analyzer.analyze_entity_coherence(text);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            EntityCoherenceError::InsufficientEntities(_)
        ));
    }

    #[test]
    fn test_entity_distribution_calculation() {
        let analyzer = EntityCoherenceAnalyzer::new();

        let chains = vec![EntityChain {
            entity: "john".to_string(),
            mentions: vec![EntityMention {
                sentence_index: 0,
                position: 0,
                mention_type: EntityMentionType::Proper,
                grammatical_role: GrammaticalRole::Subject,
                mention_text: "John".to_string(),
                confidence: 0.9,
                accessibility_score: 0.8,
                context: MentionContext {
                    surrounding_words: vec![],
                    semantic_context: "".to_string(),
                    discourse_context: "".to_string(),
                    syntactic_context: "".to_string(),
                },
            }],
            coherence_score: 0.8,
            salience: 0.9,
            role_consistency: 1.0,
            chain_type: EntityChainType::Protagonist,
            coherence_trajectory: Array1::<f64>::zeros(1),
            semantic_consistency: 0.8,
        }];

        let distribution = analyzer.calculate_entity_distribution(&chains);
        assert!(distribution.contains_key("john"));
        assert_eq!(distribution["john"], 1);
    }

    #[test]
    fn test_grammatical_role_encoding() {
        let analyzer = EntityCoherenceAnalyzer::new();

        assert_eq!(
            analyzer.encode_grammatical_role(&GrammaticalRole::Subject),
            3
        );
        assert_eq!(
            analyzer.encode_grammatical_role(&GrammaticalRole::Object),
            2
        );
        assert_eq!(
            analyzer.encode_grammatical_role(&GrammaticalRole::IndirectObject),
            1
        );
        assert_eq!(analyzer.encode_grammatical_role(&GrammaticalRole::Other), 0);
    }
}
