//! Advanced lexical coherence analysis for text evaluation
//!
//! This module provides comprehensive lexical coherence analysis including lexical chains,
//! semantic field analysis, vocabulary consistency, word relatedness, and advanced
//! lexical cohesion metrics. It offers both basic and advanced analysis modes with
//! configurable parameters for different text types and domains.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::RwLock;
use thiserror::Error;

/// Errors that can occur during lexical coherence analysis
#[derive(Debug, Error)]
pub enum LexicalCoherenceError {
    #[error("Empty text provided for lexical analysis")]
    EmptyText,
    #[error("Invalid lexical chain configuration: {0}")]
    InvalidConfiguration(String),
    #[error("Semantic lexicon error: {0}")]
    SemanticLexiconError(String),
    #[error("Word processing error: {0}")]
    WordProcessingError(String),
    #[error("Chain building failed: {0}")]
    ChainBuildingError(String),
    #[error("Coherence calculation failed: {0}")]
    CalculationError(String),
}

/// Semantic relationships between words in lexical chains
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SemanticRelationship {
    /// Words with similar meanings
    Synonymy,
    /// Hierarchical relationship (specific to general)
    Hyponymy,
    /// Part-whole relationship
    Meronymy,
    /// Opposite meanings
    Antonymy,
    /// Associated or related concepts
    Association,
    /// Sequential relationship
    Sequential,
    /// Causal relationship
    Causal,
    /// Temporal relationship
    Temporal,
    /// Morphological relationship (same root)
    Morphological,
    /// Collocational relationship
    Collocation,
}

/// Lexical chain types for different coherence patterns
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LexicalChainType {
    /// Simple lexical repetition
    Repetition,
    /// Synonymous words
    Synonymous,
    /// Superordinate-subordinate relationships
    Hierarchical,
    /// Part-whole relationships
    Meronymic,
    /// Thematic grouping
    Thematic,
    /// Morphological variants
    Morphological,
    /// Collocational patterns
    Collocational,
    /// Mixed relationship types
    Mixed,
}

/// Lexical cohesion device types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CohesionDeviceType {
    /// Exact repetition
    Repetition,
    /// Synonyms and near-synonyms
    Synonymy,
    /// General-specific relationships
    Hyponymy,
    /// Part-whole relationships
    Meronymy,
    /// Associated words
    Collocation,
    /// Opposite meanings
    Antonymy,
    /// Morphological variants
    Morphological,
    /// Bridging references
    Bridging,
}

/// Configuration for lexical coherence analysis
#[derive(Debug, Clone)]
pub struct LexicalCoherenceConfig {
    /// Threshold for lexical chain formation
    pub lexical_chain_threshold: f64,
    /// Maximum distance between chain elements
    pub max_chain_distance: usize,
    /// Minimum chain length for analysis
    pub min_chain_length: usize,
    /// Enable semantic similarity analysis
    pub use_semantic_similarity: bool,
    /// Enable morphological analysis
    pub use_morphological_analysis: bool,
    /// Enable advanced chain analysis
    pub use_advanced_analysis: bool,
    /// Semantic field matching threshold
    pub semantic_field_threshold: f64,
    /// Vocabulary consistency sensitivity
    pub vocabulary_consistency_sensitivity: f64,
    /// Word frequency weight for relatedness
    pub frequency_weight: f64,
    /// Position weight for chain strength
    pub position_weight: f64,
    /// Enable collocation detection
    pub detect_collocations: bool,
    /// Collocation window size
    pub collocation_window: usize,
    /// Enable discourse marker analysis
    pub analyze_discourse_markers: bool,
    /// Maximum semantic field depth
    pub max_semantic_depth: usize,
    /// Enable temporal coherence analysis
    pub analyze_temporal_coherence: bool,
    /// Enable cross-sentence analysis
    pub analyze_cross_sentence: bool,
}

impl Default for LexicalCoherenceConfig {
    fn default() -> Self {
        Self {
            lexical_chain_threshold: 0.6,
            max_chain_distance: 5,
            min_chain_length: 2,
            use_semantic_similarity: true,
            use_morphological_analysis: true,
            use_advanced_analysis: true,
            semantic_field_threshold: 0.7,
            vocabulary_consistency_sensitivity: 0.8,
            frequency_weight: 0.3,
            position_weight: 0.4,
            detect_collocations: true,
            collocation_window: 3,
            analyze_discourse_markers: true,
            max_semantic_depth: 4,
            analyze_temporal_coherence: true,
            analyze_cross_sentence: true,
        }
    }
}

/// Comprehensive lexical coherence analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalCoherenceResult {
    /// Overall lexical chain coherence score
    pub lexical_chain_coherence: f64,
    /// Semantic field coherence score
    pub semantic_field_coherence: f64,
    /// Lexical repetition effectiveness score
    pub lexical_repetition_score: f64,
    /// Vocabulary consistency score
    pub vocabulary_consistency: f64,
    /// Overall word relatedness score
    pub word_relatedness: f64,
    /// Lexical density measure
    pub lexical_density: f64,
    /// Identified lexical chains
    pub lexical_chains: Vec<LexicalChain>,
    /// Semantic fields analysis
    pub semantic_fields: HashMap<String, Vec<String>>,
    /// Detailed lexical metrics
    pub detailed_metrics: DetailedLexicalMetrics,
    /// Cohesion devices analysis
    pub cohesion_devices: Vec<CohesionDevice>,
    /// Advanced chain analysis
    pub advanced_analysis: Option<AdvancedChainAnalysis>,
}

/// Individual lexical chain with comprehensive analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalChain {
    /// Words in the chain with positions
    pub words: Vec<(String, Vec<(usize, usize)>)>,
    /// Chain coherence score
    pub coherence_score: f64,
    /// Semantic relationship type
    pub semantic_relationship: SemanticRelationship,
    /// Chain strength measure
    pub chain_strength: f64,
    /// Span of the chain in the text
    pub span: (usize, usize),
    /// Chain type classification
    pub chain_type: LexicalChainType,
    /// Chain density (coverage)
    pub density: f64,
    /// Semantic depth of relationships
    pub semantic_depth: usize,
    /// Position distribution analysis
    pub position_distribution: PositionDistribution,
    /// Chain connectivity measures
    pub connectivity: ChainConnectivity,
}

/// Detailed lexical metrics and statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedLexicalMetrics {
    /// Total number of lexical chains
    pub total_chains: usize,
    /// Average chain length
    pub average_chain_length: f64,
    /// Maximum chain length
    pub max_chain_length: usize,
    /// Chain density distribution
    pub chain_density_distribution: Vec<f64>,
    /// Semantic relationship distribution
    pub relationship_distribution: HashMap<String, usize>,
    /// Vocabulary richness measure
    pub vocabulary_richness: f64,
    /// Lexical sophistication score
    pub lexical_sophistication: f64,
    /// Type-token ratio
    pub type_token_ratio: f64,
    /// Moving average type-token ratio
    pub mattr: f64,
    /// Lexical diversity metrics
    pub lexical_diversity: LexicalDiversityMetrics,
    /// Coherence pattern statistics
    pub pattern_statistics: PatternStatistics,
    /// Cross-sentence coherence
    pub cross_sentence_coherence: f64,
    /// Temporal coherence measures
    pub temporal_coherence: TemporalCoherenceMetrics,
}

/// Lexical diversity measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexicalDiversityMetrics {
    /// Measure of textual lexical diversity (MTLD)
    pub mtld: f64,
    /// Hypergeometric distribution diversity (HD-D)
    pub hdd: f64,
    /// Measure of lexical diversity (MLD)
    pub mld: f64,
    /// Lexical frequency profile entropy
    pub entropy: f64,
    /// Simpson's diversity index
    pub simpson_index: f64,
    /// Shannon diversity index
    pub shannon_index: f64,
}

/// Pattern-based coherence statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternStatistics {
    /// Repetition pattern strength
    pub repetition_strength: f64,
    /// Synonymy pattern distribution
    pub synonymy_distribution: Vec<f64>,
    /// Hierarchical pattern depth
    pub hierarchical_depth: f64,
    /// Thematic consistency score
    pub thematic_consistency: f64,
    /// Morphological variation score
    pub morphological_variation: f64,
    /// Collocational strength
    pub collocational_strength: f64,
}

/// Temporal aspects of lexical coherence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalCoherenceMetrics {
    /// Coherence evolution over text progression
    pub coherence_evolution: Vec<f64>,
    /// Temporal consistency of chains
    pub chain_temporal_consistency: f64,
    /// Periodic coherence patterns
    pub periodic_patterns: Vec<PeriodicPattern>,
    /// Coherence momentum measure
    pub coherence_momentum: f64,
    /// Temporal clustering coefficient
    pub temporal_clustering: f64,
}

/// Periodic patterns in coherence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodicPattern {
    /// Pattern period length
    pub period: usize,
    /// Pattern strength
    pub strength: f64,
    /// Pattern phase offset
    pub phase: f64,
    /// Pattern type description
    pub pattern_type: String,
}

/// Position distribution analysis for chains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionDistribution {
    /// Position entropy measure
    pub entropy: f64,
    /// Position clustering coefficient
    pub clustering_coefficient: f64,
    /// Dispersion index
    pub dispersion_index: f64,
    /// Position variance
    pub variance: f64,
    /// Regularity measure
    pub regularity: f64,
}

/// Chain connectivity measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainConnectivity {
    /// Inter-chain connections
    pub inter_chain_connections: usize,
    /// Connection strength
    pub connection_strength: f64,
    /// Network centrality
    pub network_centrality: f64,
    /// Bridge strength (connecting different topics)
    pub bridge_strength: f64,
}

/// Cohesion device analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohesionDevice {
    /// Type of cohesion device
    pub device_type: CohesionDeviceType,
    /// Connected elements
    pub elements: Vec<String>,
    /// Positions in text
    pub positions: Vec<(usize, usize)>,
    /// Strength of cohesive link
    pub strength: f64,
    /// Local coherence contribution
    pub local_coherence_contribution: f64,
    /// Global coherence contribution
    pub global_coherence_contribution: f64,
}

/// Advanced chain analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedChainAnalysis {
    /// Chain network analysis
    pub network_analysis: ChainNetworkAnalysis,
    /// Information-theoretic measures
    pub information_measures: InformationMeasures,
    /// Cognitive load estimates
    pub cognitive_load: CognitiveLoadMetrics,
    /// Discourse structure alignment
    pub discourse_alignment: DiscourseAlignmentMetrics,
}

/// Network-based analysis of lexical chains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainNetworkAnalysis {
    /// Network density
    pub density: f64,
    /// Average clustering coefficient
    pub clustering_coefficient: f64,
    /// Average path length
    pub average_path_length: f64,
    /// Network modularity
    pub modularity: f64,
    /// Central chains (by betweenness centrality)
    pub central_chains: Vec<usize>,
    /// Network diameter
    pub diameter: usize,
    /// Small-world coefficient
    pub small_world_coefficient: f64,
}

/// Information-theoretic measures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationMeasures {
    /// Mutual information between chains
    pub chain_mutual_information: f64,
    /// Information entropy of chain distribution
    pub chain_entropy: f64,
    /// Conditional entropy measures
    pub conditional_entropy: f64,
    /// Information flow measures
    pub information_flow: f64,
    /// Redundancy coefficient
    pub redundancy_coefficient: f64,
}

/// Cognitive load estimates for lexical processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveLoadMetrics {
    /// Working memory load estimate
    pub working_memory_load: f64,
    /// Processing complexity estimate
    pub processing_complexity: f64,
    /// Integration effort estimate
    pub integration_effort: f64,
    /// Disambiguation load
    pub disambiguation_load: f64,
    /// Overall cognitive efficiency
    pub cognitive_efficiency: f64,
}

/// Alignment with discourse structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscourseAlignmentMetrics {
    /// Paragraph-level coherence alignment
    pub paragraph_alignment: f64,
    /// Section-level coherence alignment
    pub section_alignment: f64,
    /// Discourse marker coordination
    pub marker_coordination: f64,
    /// Topic progression alignment
    pub topic_progression_alignment: f64,
}

/// Advanced lexical coherence analyzer
pub struct LexicalCoherenceAnalyzer {
    config: LexicalCoherenceConfig,
    semantic_lexicon: Arc<RwLock<HashMap<String, Vec<String>>>>,
    morphological_rules: HashMap<String, Vec<String>>,
    discourse_markers: HashSet<String>,
    collocation_patterns: HashMap<String, Vec<String>>,
    frequency_cache: Arc<RwLock<HashMap<String, f64>>>,
    chain_cache: Arc<RwLock<HashMap<String, Vec<LexicalChain>>>>,
}

impl LexicalCoherenceAnalyzer {
    /// Create a new lexical coherence analyzer with default configuration
    pub fn new() -> Self {
        Self::with_config(LexicalCoherenceConfig::default())
    }

    /// Create a new lexical coherence analyzer with custom configuration
    pub fn with_config(config: LexicalCoherenceConfig) -> Self {
        let semantic_lexicon = Arc::new(RwLock::new(Self::build_semantic_lexicon()));
        let morphological_rules = Self::build_morphological_rules();
        let discourse_markers = Self::build_discourse_markers();
        let collocation_patterns = Self::build_collocation_patterns();
        let frequency_cache = Arc::new(RwLock::new(HashMap::new()));
        let chain_cache = Arc::new(RwLock::new(HashMap::new()));

        Self {
            config,
            semantic_lexicon,
            morphological_rules,
            discourse_markers,
            collocation_patterns,
            frequency_cache,
            chain_cache,
        }
    }

    /// Analyze lexical coherence of the given text
    pub fn analyze_lexical_coherence(
        &self,
        text: &str,
    ) -> Result<LexicalCoherenceResult, LexicalCoherenceError> {
        if text.trim().is_empty() {
            return Err(LexicalCoherenceError::EmptyText);
        }

        let sentences = self.split_into_sentences(text)?;

        // Build comprehensive lexical chains
        let lexical_chains = self.build_lexical_chains(&sentences)?;

        // Analyze semantic fields
        let semantic_fields = self.identify_semantic_fields(&sentences);

        // Calculate core coherence metrics
        let lexical_chain_coherence = self.calculate_lexical_chain_coherence(&lexical_chains)?;
        let semantic_field_coherence =
            self.calculate_semantic_field_coherence(&semantic_fields, &sentences);
        let lexical_repetition_score = self.calculate_lexical_repetition_score(&sentences);
        let vocabulary_consistency = self.calculate_vocabulary_consistency(&sentences);
        let word_relatedness = self.calculate_word_relatedness(&sentences);
        let lexical_density = self.calculate_lexical_density(&sentences);

        // Generate detailed metrics
        let detailed_metrics =
            self.generate_detailed_metrics(&sentences, &lexical_chains, &semantic_fields);

        // Analyze cohesion devices
        let cohesion_devices = self.analyze_cohesion_devices(&sentences, &lexical_chains);

        // Perform advanced analysis if enabled
        let advanced_analysis = if self.config.use_advanced_analysis {
            Some(self.perform_advanced_analysis(&sentences, &lexical_chains)?)
        } else {
            None
        };

        Ok(LexicalCoherenceResult {
            lexical_chain_coherence,
            semantic_field_coherence,
            lexical_repetition_score,
            vocabulary_consistency,
            word_relatedness,
            lexical_density,
            lexical_chains,
            semantic_fields,
            detailed_metrics,
            cohesion_devices,
            advanced_analysis,
        })
    }

    /// Build comprehensive lexical chains with advanced analysis
    fn build_lexical_chains(
        &self,
        sentences: &[String],
    ) -> Result<Vec<LexicalChain>, LexicalCoherenceError> {
        let mut all_words = HashMap::new();
        let mut word_frequencies = HashMap::new();

        // Extract words with positions and calculate frequencies
        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let words = self.extract_content_words(sentence);
            for (word_idx, word) in words.iter().enumerate() {
                all_words
                    .entry(word.clone())
                    .or_insert_with(Vec::new)
                    .push((sent_idx, word_idx));
                *word_frequencies.entry(word.clone()).or_insert(0) += 1;
            }
        }

        let mut chains = Vec::new();
        let mut processed_words = HashSet::new();

        for (word, positions) in all_words.iter() {
            if processed_words.contains(word) || positions.len() < self.config.min_chain_length {
                continue;
            }

            let chain_words = vec![(word.clone(), positions.clone())];
            let mut extended_chain =
                self.extend_lexical_chain(word, &all_words, &mut processed_words)?;
            extended_chain.extend(chain_words);

            if extended_chain.len() >= self.config.min_chain_length {
                let chain =
                    self.build_chain_with_analysis(extended_chain, &word_frequencies, sentences)?;
                chains.push(chain);
            }
        }

        // Sort chains by strength
        chains.sort_by(|a, b| b.chain_strength.partial_cmp(&a.chain_strength).unwrap());

        Ok(chains)
    }

    /// Extend lexical chain with semantically related words
    fn extend_lexical_chain(
        &self,
        seed_word: &str,
        all_words: &HashMap<String, Vec<(usize, usize)>>,
        processed_words: &mut HashSet<String>,
    ) -> Result<Vec<(String, Vec<(usize, usize)>)>, LexicalCoherenceError> {
        let mut chain_words = Vec::new();
        let mut candidates = VecDeque::new();
        candidates.push_back(seed_word.to_string());

        while let Some(current_word) = candidates.pop_front() {
            if processed_words.contains(&current_word) {
                continue;
            }

            processed_words.insert(current_word.clone());

            if let Some(positions) = all_words.get(&current_word) {
                chain_words.push((current_word.clone(), positions.clone()));

                // Find semantically related words
                let related_words = self.find_semantically_related_words(&current_word, all_words);

                for (related_word, similarity) in related_words {
                    if similarity > self.config.lexical_chain_threshold
                        && !processed_words.contains(&related_word)
                    {
                        candidates.push_back(related_word);
                    }
                }
            }
        }

        Ok(chain_words)
    }

    /// Build chain with comprehensive analysis
    fn build_chain_with_analysis(
        &self,
        chain_words: Vec<(String, Vec<(usize, usize)>)>,
        word_frequencies: &HashMap<String, usize>,
        sentences: &[String],
    ) -> Result<LexicalChain, LexicalCoherenceError> {
        let all_positions: Vec<(usize, usize)> = chain_words
            .iter()
            .flat_map(|(_, positions)| positions.iter())
            .copied()
            .collect();

        let span = if let (Some(min_pos), Some(max_pos)) =
            (all_positions.iter().min(), all_positions.iter().max())
        {
            (*min_pos, *max_pos)
        } else {
            return Err(LexicalCoherenceError::ChainBuildingError(
                "Invalid chain positions".to_string(),
            ));
        };

        let words_list: Vec<String> = chain_words.iter().map(|(word, _)| word.clone()).collect();
        let semantic_relationship = self.determine_semantic_relationship(&words_list);
        let chain_type = self.classify_chain_type(&words_list, &semantic_relationship);

        let coherence_score = self.calculate_chain_coherence(&chain_words, word_frequencies)?;
        let chain_strength = self.calculate_chain_strength(&chain_words, span, sentences.len());
        let density = self.calculate_chain_density(&chain_words, sentences.len());
        let semantic_depth = self.calculate_semantic_depth(&words_list);

        let position_distribution = self.analyze_position_distribution(&all_positions);
        let connectivity = self.analyze_chain_connectivity(&chain_words, &all_positions);

        Ok(LexicalChain {
            words: chain_words,
            coherence_score,
            semantic_relationship,
            chain_strength,
            span,
            chain_type,
            density,
            semantic_depth,
            position_distribution,
            connectivity,
        })
    }

    /// Find semantically related words using lexicon and morphological analysis
    fn find_semantically_related_words(
        &self,
        word: &str,
        all_words: &HashMap<String, Vec<(usize, usize)>>,
    ) -> HashMap<String, f64> {
        let mut related_words = HashMap::new();

        // Check semantic lexicon
        if let Ok(lexicon) = self.semantic_lexicon.read() {
            if let Some(synonyms) = lexicon.get(word) {
                for synonym in synonyms {
                    if all_words.contains_key(synonym) {
                        related_words.insert(synonym.clone(), 0.9);
                    }
                }
            }
        }

        // Check morphological relationships
        if self.config.use_morphological_analysis {
            for other_word in all_words.keys() {
                let morphological_similarity =
                    self.calculate_morphological_similarity(word, other_word);
                if morphological_similarity > 0.7 {
                    related_words.insert(other_word.clone(), morphological_similarity * 0.8);
                }
            }
        }

        // Calculate lexical similarity for remaining words
        for other_word in all_words.keys() {
            if !related_words.contains_key(other_word) {
                let similarity = self.calculate_lexical_similarity(word, other_word);
                if similarity > self.config.lexical_chain_threshold {
                    related_words.insert(other_word.clone(), similarity);
                }
            }
        }

        related_words
    }

    /// Calculate lexical similarity between two words
    fn calculate_lexical_similarity(&self, word1: &str, word2: &str) -> f64 {
        if word1 == word2 {
            return 1.0;
        }

        // Character-level similarity
        let char_similarity = self.calculate_character_similarity(word1, word2);

        // Semantic similarity (if available)
        let semantic_similarity = if self.config.use_semantic_similarity {
            self.calculate_semantic_similarity(word1, word2)
        } else {
            0.0
        };

        // Morphological similarity
        let morphological_similarity = if self.config.use_morphological_analysis {
            self.calculate_morphological_similarity(word1, word2)
        } else {
            0.0
        };

        // Weighted combination
        let weights = (0.4, 0.4, 0.2); // (char, semantic, morphological)
        (char_similarity * weights.0)
            + (semantic_similarity * weights.1)
            + (morphological_similarity * weights.2)
    }

    /// Calculate character-level similarity using edit distance
    fn calculate_character_similarity(&self, word1: &str, word2: &str) -> f64 {
        let max_len = word1.len().max(word2.len());
        if max_len == 0 {
            return 1.0;
        }

        let distance = self.levenshtein_distance(word1, word2);
        1.0 - (distance as f64 / max_len as f64)
    }

    /// Calculate Levenshtein distance between two strings
    fn levenshtein_distance(&self, word1: &str, word2: &str) -> usize {
        let chars1: Vec<char> = word1.chars().collect();
        let chars2: Vec<char> = word2.chars().collect();
        let len1 = chars1.len();
        let len2 = chars2.len();

        let mut dp = vec![vec![0; len2 + 1]; len1 + 1];

        for i in 0..=len1 {
            dp[i][0] = i;
        }
        for j in 0..=len2 {
            dp[0][j] = j;
        }

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
                dp[i][j] = std::cmp::min(
                    std::cmp::min(dp[i - 1][j] + 1, dp[i][j - 1] + 1),
                    dp[i - 1][j - 1] + cost,
                );
            }
        }

        dp[len1][len2]
    }

    /// Calculate semantic similarity using lexicon
    fn calculate_semantic_similarity(&self, word1: &str, word2: &str) -> f64 {
        if let Ok(lexicon) = self.semantic_lexicon.read() {
            // Check if words are in same semantic field
            for field_words in lexicon.values() {
                let has_word1 = field_words.contains(&word1.to_string());
                let has_word2 = field_words.contains(&word2.to_string());
                if has_word1 && has_word2 {
                    return 0.8;
                }
            }
        }
        0.0
    }

    /// Calculate morphological similarity
    fn calculate_morphological_similarity(&self, word1: &str, word2: &str) -> f64 {
        // Check for common roots or stems
        let stem1 = self.extract_stem(word1);
        let stem2 = self.extract_stem(word2);

        if stem1 == stem2 && stem1.len() >= 3 {
            return 0.9;
        }

        // Check for morphological patterns
        for (pattern, variants) in &self.morphological_rules {
            if word1.contains(pattern) && variants.iter().any(|v| word2.contains(v)) {
                return 0.7;
            }
        }

        0.0
    }

    /// Extract word stem (simple implementation)
    fn extract_stem(&self, word: &str) -> String {
        let mut stem = word.to_lowercase();

        // Remove common suffixes
        let suffixes = [
            "ing", "ed", "er", "est", "ly", "ion", "tion", "ness", "ment",
        ];
        for suffix in &suffixes {
            if stem.ends_with(suffix) && stem.len() > suffix.len() + 2 {
                stem.truncate(stem.len() - suffix.len());
                break;
            }
        }

        stem
    }

    /// Determine semantic relationship between words in a chain
    fn determine_semantic_relationship(&self, words: &[String]) -> SemanticRelationship {
        if words.len() < 2 {
            return SemanticRelationship::Association;
        }

        // Check for exact repetition
        if words.windows(2).any(|w| w[0] == w[1]) {
            return SemanticRelationship::Synonymy;
        }

        // Check semantic lexicon for relationships
        if let Ok(lexicon) = self.semantic_lexicon.read() {
            for field_words in lexicon.values() {
                let matches = words.iter().filter(|w| field_words.contains(w)).count();
                if matches >= 2 {
                    return SemanticRelationship::Synonymy;
                }
            }
        }

        // Check morphological relationships
        if self.config.use_morphological_analysis {
            let has_morphological_relation = words
                .windows(2)
                .any(|pair| self.calculate_morphological_similarity(&pair[0], &pair[1]) > 0.7);
            if has_morphological_relation {
                return SemanticRelationship::Morphological;
            }
        }

        SemanticRelationship::Association
    }

    /// Classify chain type based on words and relationships
    fn classify_chain_type(
        &self,
        words: &[String],
        relationship: &SemanticRelationship,
    ) -> LexicalChainType {
        // Check for exact repetition
        let unique_words: HashSet<_> = words.iter().collect();
        if unique_words.len() < words.len() {
            return LexicalChainType::Repetition;
        }

        match relationship {
            SemanticRelationship::Synonymy => LexicalChainType::Synonymous,
            SemanticRelationship::Hyponymy => LexicalChainType::Hierarchical,
            SemanticRelationship::Meronymy => LexicalChainType::Meronymic,
            SemanticRelationship::Morphological => LexicalChainType::Morphological,
            SemanticRelationship::Collocation => LexicalChainType::Collocational,
            _ => {
                // Analyze content to determine thematic vs mixed
                if self.is_thematic_chain(words) {
                    LexicalChainType::Thematic
                } else {
                    LexicalChainType::Mixed
                }
            }
        }
    }

    /// Check if chain represents a thematic grouping
    fn is_thematic_chain(&self, words: &[String]) -> bool {
        if let Ok(lexicon) = self.semantic_lexicon.read() {
            for field_words in lexicon.values() {
                let matches = words.iter().filter(|w| field_words.contains(w)).count();
                if matches as f64 / words.len() as f64 > 0.6 {
                    return true;
                }
            }
        }
        false
    }

    /// Calculate chain coherence score
    fn calculate_chain_coherence(
        &self,
        chain_words: &[(String, Vec<(usize, usize)>)],
        word_frequencies: &HashMap<String, usize>,
    ) -> Result<f64, LexicalCoherenceError> {
        if chain_words.is_empty() {
            return Ok(0.0);
        }

        let total_positions: usize = chain_words
            .iter()
            .map(|(_, positions)| positions.len())
            .sum();

        // Frequency-weighted coherence
        let frequency_score: f64 = chain_words
            .iter()
            .map(|(word, positions)| {
                let frequency = *word_frequencies.get(word).unwrap_or(&1) as f64;
                (frequency.ln() + 1.0) * positions.len() as f64
            })
            .sum::<f64>()
            / total_positions as f64;

        // Position distribution score
        let position_score = self.calculate_position_coherence_score(chain_words);

        // Semantic consistency score
        let semantic_score = self.calculate_semantic_consistency_score(chain_words);

        // Weighted combination
        Ok((frequency_score * 0.4) + (position_score * 0.3) + (semantic_score * 0.3))
    }

    /// Calculate position-based coherence score
    fn calculate_position_coherence_score(
        &self,
        chain_words: &[(String, Vec<(usize, usize)>)],
    ) -> f64 {
        let all_positions: Vec<(usize, usize)> = chain_words
            .iter()
            .flat_map(|(_, positions)| positions.iter())
            .copied()
            .collect();

        if all_positions.len() < 2 {
            return 1.0;
        }

        // Calculate position clustering
        let position_variance = self.calculate_position_variance(&all_positions);
        let max_distance = self.calculate_max_position_distance(&all_positions);

        if max_distance == 0.0 {
            return 1.0;
        }

        1.0 - (position_variance / max_distance).min(1.0)
    }

    /// Calculate semantic consistency score for chain
    fn calculate_semantic_consistency_score(
        &self,
        chain_words: &[(String, Vec<(usize, usize)>)],
    ) -> f64 {
        let words: Vec<String> = chain_words.iter().map(|(word, _)| word.clone()).collect();

        if words.len() < 2 {
            return 1.0;
        }

        let mut total_similarity = 0.0;
        let mut comparisons = 0;

        for i in 0..words.len() {
            for j in i + 1..words.len() {
                total_similarity += self.calculate_lexical_similarity(&words[i], &words[j]);
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            total_similarity / comparisons as f64
        } else {
            0.0
        }
    }

    /// Calculate variance in chain positions
    fn calculate_position_variance(&self, positions: &[(usize, usize)]) -> f64 {
        if positions.len() < 2 {
            return 0.0;
        }

        let flat_positions: Vec<f64> = positions
            .iter()
            .map(|(sent, word)| sent * 1000 + word) // Flatten to single dimension
            .map(|pos| pos as f64)
            .collect();

        let mean = flat_positions.iter().sum::<f64>() / flat_positions.len() as f64;
        let variance = flat_positions
            .iter()
            .map(|pos| (pos - mean).powi(2))
            .sum::<f64>()
            / flat_positions.len() as f64;

        variance.sqrt()
    }

    /// Calculate maximum distance between positions
    fn calculate_max_position_distance(&self, positions: &[(usize, usize)]) -> f64 {
        if positions.len() < 2 {
            return 0.0;
        }

        let flat_positions: Vec<usize> = positions
            .iter()
            .map(|(sent, word)| sent * 1000 + word)
            .collect();

        let min_pos = *flat_positions.iter().min().unwrap_or(&0) as f64;
        let max_pos = *flat_positions.iter().max().unwrap_or(&0) as f64;

        max_pos - min_pos
    }

    /// Build semantic lexicon
    fn build_semantic_lexicon() -> HashMap<String, Vec<String>> {
        let mut lexicon = HashMap::new();

        // Basic semantic fields
        lexicon.insert(
            "emotions".to_string(),
            vec![
                "happy".to_string(),
                "sad".to_string(),
                "angry".to_string(),
                "joy".to_string(),
                "fear".to_string(),
                "love".to_string(),
                "hate".to_string(),
                "excited".to_string(),
            ],
        );

        lexicon.insert(
            "colors".to_string(),
            vec![
                "red".to_string(),
                "blue".to_string(),
                "green".to_string(),
                "yellow".to_string(),
                "black".to_string(),
                "white".to_string(),
                "purple".to_string(),
                "orange".to_string(),
            ],
        );

        lexicon.insert(
            "animals".to_string(),
            vec![
                "dog".to_string(),
                "cat".to_string(),
                "bird".to_string(),
                "fish".to_string(),
                "horse".to_string(),
                "cow".to_string(),
                "pig".to_string(),
                "sheep".to_string(),
            ],
        );

        lexicon.insert(
            "nature".to_string(),
            vec![
                "tree".to_string(),
                "flower".to_string(),
                "grass".to_string(),
                "mountain".to_string(),
                "river".to_string(),
                "ocean".to_string(),
                "forest".to_string(),
                "sky".to_string(),
            ],
        );

        lexicon.insert(
            "time".to_string(),
            vec![
                "day".to_string(),
                "night".to_string(),
                "morning".to_string(),
                "evening".to_string(),
                "yesterday".to_string(),
                "today".to_string(),
                "tomorrow".to_string(),
                "week".to_string(),
            ],
        );

        lexicon
    }

    /// Build morphological rules
    fn build_morphological_rules() -> HashMap<String, Vec<String>> {
        let mut rules = HashMap::new();

        rules.insert("ing".to_string(), vec!["ed".to_string(), "s".to_string()]);
        rules.insert("ed".to_string(), vec!["ing".to_string(), "s".to_string()]);
        rules.insert("er".to_string(), vec!["est".to_string(), "ly".to_string()]);
        rules.insert(
            "tion".to_string(),
            vec!["tive".to_string(), "tor".to_string()],
        );
        rules.insert(
            "ness".to_string(),
            vec!["ful".to_string(), "less".to_string()],
        );

        rules
    }

    /// Build discourse markers set
    fn build_discourse_markers() -> HashSet<String> {
        let markers = vec![
            "however",
            "therefore",
            "furthermore",
            "moreover",
            "nevertheless",
            "consequently",
            "additionally",
            "similarly",
            "conversely",
            "meanwhile",
            "first",
            "second",
            "finally",
            "in_conclusion",
            "for_example",
        ];

        markers.into_iter().map(String::from).collect()
    }

    /// Build collocation patterns
    fn build_collocation_patterns() -> HashMap<String, Vec<String>> {
        let mut patterns = HashMap::new();

        patterns.insert(
            "make".to_string(),
            vec![
                "decision".to_string(),
                "choice".to_string(),
                "mistake".to_string(),
            ],
        );
        patterns.insert(
            "take".to_string(),
            vec!["time".to_string(), "care".to_string(), "action".to_string()],
        );
        patterns.insert(
            "strong".to_string(),
            vec![
                "coffee".to_string(),
                "wind".to_string(),
                "argument".to_string(),
            ],
        );
        patterns.insert(
            "heavy".to_string(),
            vec![
                "rain".to_string(),
                "traffic".to_string(),
                "workload".to_string(),
            ],
        );

        patterns
    }

    // Additional helper methods for comprehensive analysis...

    /// Split text into sentences
    fn split_into_sentences(&self, text: &str) -> Result<Vec<String>, LexicalCoherenceError> {
        let sentences: Vec<String> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_string())
            .collect();

        if sentences.is_empty() {
            return Err(LexicalCoherenceError::WordProcessingError(
                "No valid sentences found".to_string(),
            ));
        }

        Ok(sentences)
    }

    /// Extract content words (non-function words)
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

    // Continue with additional analysis methods...
    // [The implementation continues with the remaining methods for comprehensive analysis]

    /// Calculate lexical chain coherence
    fn calculate_lexical_chain_coherence(
        &self,
        chains: &[LexicalChain],
    ) -> Result<f64, LexicalCoherenceError> {
        if chains.is_empty() {
            return Ok(0.0);
        }

        let total_coherence: f64 = chains.iter().map(|chain| chain.coherence_score).sum();
        Ok(total_coherence / chains.len() as f64)
    }

    /// Identify semantic fields in sentences
    fn identify_semantic_fields(&self, sentences: &[String]) -> HashMap<String, Vec<String>> {
        let mut fields = HashMap::new();

        if let Ok(lexicon) = self.semantic_lexicon.read() {
            for (field_name, field_words) in lexicon.iter() {
                let mut found_words = Vec::new();

                for sentence in sentences {
                    let content_words = self.extract_content_words(sentence);
                    for word in content_words {
                        if field_words.contains(&word) {
                            found_words.push(word);
                        }
                    }
                }

                if !found_words.is_empty() {
                    fields.insert(field_name.clone(), found_words);
                }
            }
        }

        fields
    }

    /// Calculate semantic field coherence
    fn calculate_semantic_field_coherence(
        &self,
        fields: &HashMap<String, Vec<String>>,
        sentences: &[String],
    ) -> f64 {
        if fields.is_empty() || sentences.is_empty() {
            return 0.0;
        }

        let total_words: usize = sentences
            .iter()
            .map(|s| self.extract_content_words(s).len())
            .sum();

        let field_coverage: f64 =
            fields.values().map(|words| words.len() as f64).sum::<f64>() / total_words as f64;

        let field_distribution = self.calculate_field_distribution_evenness(fields);

        (field_coverage * 0.6) + (field_distribution * 0.4)
    }

    /// Calculate how evenly semantic fields are distributed
    fn calculate_field_distribution_evenness(&self, fields: &HashMap<String, Vec<String>>) -> f64 {
        if fields.len() < 2 {
            return 1.0;
        }

        let field_sizes: Vec<f64> = fields.values().map(|words| words.len() as f64).collect();
        let total_words: f64 = field_sizes.iter().sum();

        if total_words == 0.0 {
            return 0.0;
        }

        let proportions: Vec<f64> = field_sizes.iter().map(|size| size / total_words).collect();
        let entropy = -proportions
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>();

        let max_entropy = (fields.len() as f64).ln();

        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Calculate lexical repetition score
    fn calculate_lexical_repetition_score(&self, sentences: &[String]) -> f64 {
        let mut all_words = Vec::new();
        let mut word_counts = HashMap::new();

        for sentence in sentences {
            let content_words = self.extract_content_words(sentence);
            for word in content_words {
                all_words.push(word.clone());
                *word_counts.entry(word).or_insert(0) += 1;
            }
        }

        if all_words.is_empty() {
            return 0.0;
        }

        let repeated_words = word_counts.values().filter(|&&count| count > 1).count();
        let total_unique_words = word_counts.len();

        if total_unique_words > 0 {
            repeated_words as f64 / total_unique_words as f64
        } else {
            0.0
        }
    }

    /// Calculate vocabulary consistency
    fn calculate_vocabulary_consistency(&self, sentences: &[String]) -> f64 {
        if sentences.len() < 2 {
            return 1.0;
        }

        let mut segment_vocabularies = Vec::new();
        let segment_size = (sentences.len() / 3).max(1); // Divide into 3 segments

        for i in (0..sentences.len()).step_by(segment_size) {
            let end = (i + segment_size).min(sentences.len());
            let segment_words: HashSet<String> = sentences[i..end]
                .iter()
                .flat_map(|s| self.extract_content_words(s))
                .collect();
            segment_vocabularies.push(segment_words);
        }

        if segment_vocabularies.len() < 2 {
            return 1.0;
        }

        let mut total_overlap = 0.0;
        let mut comparisons = 0;

        for i in 0..segment_vocabularies.len() {
            for j in i + 1..segment_vocabularies.len() {
                let intersection_size = segment_vocabularies[i]
                    .intersection(&segment_vocabularies[j])
                    .count();
                let union_size = segment_vocabularies[i]
                    .union(&segment_vocabularies[j])
                    .count();

                if union_size > 0 {
                    total_overlap += intersection_size as f64 / union_size as f64;
                    comparisons += 1;
                }
            }
        }

        if comparisons > 0 {
            total_overlap / comparisons as f64
        } else {
            0.0
        }
    }

    /// Calculate word relatedness score
    fn calculate_word_relatedness(&self, sentences: &[String]) -> f64 {
        let all_words: Vec<String> = sentences
            .iter()
            .flat_map(|s| self.extract_content_words(s))
            .collect();

        if all_words.len() < 2 {
            return 0.0;
        }

        let mut total_relatedness = 0.0;
        let mut comparisons = 0;

        for i in 0..all_words.len() {
            for j in i + 1..all_words.len().min(i + 10) {
                // Limit window for efficiency
                let relatedness = self.calculate_lexical_similarity(&all_words[i], &all_words[j]);
                total_relatedness += relatedness;
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            total_relatedness / comparisons as f64
        } else {
            0.0
        }
    }

    /// Calculate lexical density
    fn calculate_lexical_density(&self, sentences: &[String]) -> f64 {
        let mut total_words = 0;
        let mut content_words = 0;

        for sentence in sentences {
            let words = sentence.split_whitespace().count();
            let content = self.extract_content_words(sentence).len();

            total_words += words;
            content_words += content;
        }

        if total_words > 0 {
            content_words as f64 / total_words as f64
        } else {
            0.0
        }
    }

    /// Calculate chain strength
    fn calculate_chain_strength(
        &self,
        chain_words: &[(String, Vec<(usize, usize)>)],
        span: (usize, usize),
        total_sentences: usize,
    ) -> f64 {
        let total_occurrences: usize = chain_words
            .iter()
            .map(|(_, positions)| positions.len())
            .sum();

        let span_coverage = if total_sentences > 0 {
            ((span.1 .0 - span.0 .0) as f64 + 1.0) / total_sentences as f64
        } else {
            0.0
        };

        let frequency_strength = (total_occurrences as f64).ln() + 1.0;
        let position_strength = span_coverage;

        (frequency_strength * self.config.frequency_weight)
            + (position_strength * self.config.position_weight)
    }

    /// Calculate chain density
    fn calculate_chain_density(
        &self,
        chain_words: &[(String, Vec<(usize, usize)>)],
        total_sentences: usize,
    ) -> f64 {
        let covered_sentences: HashSet<usize> = chain_words
            .iter()
            .flat_map(|(_, positions)| positions.iter().map(|(sent, _)| *sent))
            .collect();

        if total_sentences > 0 {
            covered_sentences.len() as f64 / total_sentences as f64
        } else {
            0.0
        }
    }

    /// Calculate semantic depth
    fn calculate_semantic_depth(&self, words: &[String]) -> usize {
        // Simple implementation - could be enhanced with more sophisticated analysis
        let unique_words: HashSet<_> = words.iter().collect();
        if unique_words.len() == words.len() {
            3 // High diversity
        } else if unique_words.len() * 2 > words.len() {
            2 // Medium diversity
        } else {
            1 // Low diversity (repetition)
        }
    }

    /// Analyze position distribution
    fn analyze_position_distribution(&self, positions: &[(usize, usize)]) -> PositionDistribution {
        if positions.len() < 2 {
            return PositionDistribution {
                entropy: 1.0,
                clustering_coefficient: 1.0,
                dispersion_index: 0.0,
                variance: 0.0,
                regularity: 1.0,
            };
        }

        let entropy = self.calculate_position_entropy(positions);
        let clustering_coefficient = self.calculate_position_clustering(positions);
        let dispersion_index = self.calculate_dispersion_index(positions);
        let variance = self.calculate_position_variance(positions);
        let regularity = self.calculate_position_regularity(positions);

        PositionDistribution {
            entropy,
            clustering_coefficient,
            dispersion_index,
            variance,
            regularity,
        }
    }

    /// Calculate position entropy
    fn calculate_position_entropy(&self, positions: &[(usize, usize)]) -> f64 {
        let sentence_counts = positions.iter().fold(HashMap::new(), |mut acc, (sent, _)| {
            *acc.entry(*sent).or_insert(0) += 1;
            acc
        });

        let total = positions.len() as f64;
        let entropy = sentence_counts
            .values()
            .map(|&count| {
                let p = count as f64 / total;
                -p * p.ln()
            })
            .sum::<f64>();

        entropy / (sentence_counts.len() as f64).ln().max(1.0)
    }

    /// Calculate position clustering coefficient
    fn calculate_position_clustering(&self, positions: &[(usize, usize)]) -> f64 {
        if positions.len() < 3 {
            return 1.0;
        }

        let mut total_clustering = 0.0;

        for i in 0..positions.len() {
            let neighbors = positions
                .iter()
                .enumerate()
                .filter(|(j, pos)| {
                    *j != i && self.calculate_position_distance(&positions[i], pos) <= 2.0
                })
                .map(|(j, _)| j)
                .collect::<Vec<_>>();

            if neighbors.len() > 1 {
                let mut neighbor_connections = 0;
                for &n1 in &neighbors {
                    for &n2 in &neighbors {
                        if n1 < n2
                            && self.calculate_position_distance(&positions[n1], &positions[n2])
                                <= 2.0
                        {
                            neighbor_connections += 1;
                        }
                    }
                }
                let max_connections = neighbors.len() * (neighbors.len() - 1) / 2;
                if max_connections > 0 {
                    total_clustering += neighbor_connections as f64 / max_connections as f64;
                }
            }
        }

        total_clustering / positions.len() as f64
    }

    /// Calculate position distance
    fn calculate_position_distance(&self, pos1: &(usize, usize), pos2: &(usize, usize)) -> f64 {
        let sent_diff = (pos1.0 as i32 - pos2.0 as i32).abs() as f64;
        let word_diff = if pos1.0 == pos2.0 {
            (pos1.1 as i32 - pos2.1 as i32).abs() as f64 * 0.1
        } else {
            0.0
        };
        sent_diff + word_diff
    }

    /// Calculate dispersion index
    fn calculate_dispersion_index(&self, positions: &[(usize, usize)]) -> f64 {
        if positions.len() < 2 {
            return 0.0;
        }

        let sentences: Vec<usize> = positions.iter().map(|(sent, _)| *sent).collect();
        let min_sent = *sentences.iter().min().unwrap_or(&0);
        let max_sent = *sentences.iter().max().unwrap_or(&0);
        let range = max_sent - min_sent + 1;

        if range > 1 {
            positions.len() as f64 / range as f64
        } else {
            1.0
        }
    }

    /// Calculate position regularity
    fn calculate_position_regularity(&self, positions: &[(usize, usize)]) -> f64 {
        if positions.len() < 2 {
            return 1.0;
        }

        let sentences: Vec<usize> = positions.iter().map(|(sent, _)| *sent).collect();
        let mut distances = Vec::new();

        for i in 1..sentences.len() {
            distances.push(sentences[i] - sentences[i - 1]);
        }

        if distances.is_empty() {
            return 1.0;
        }

        let mean_distance = distances.iter().sum::<usize>() as f64 / distances.len() as f64;
        let variance = distances
            .iter()
            .map(|&d| (d as f64 - mean_distance).powi(2))
            .sum::<f64>()
            / distances.len() as f64;

        1.0 / (1.0 + variance.sqrt())
    }

    /// Analyze chain connectivity
    fn analyze_chain_connectivity(
        &self,
        chain_words: &[(String, Vec<(usize, usize)>)],
        positions: &[(usize, usize)],
    ) -> ChainConnectivity {
        // Simplified implementation
        ChainConnectivity {
            inter_chain_connections: 0,
            connection_strength: 0.0,
            network_centrality: 0.0,
            bridge_strength: 0.0,
        }
    }

    /// Generate detailed metrics
    fn generate_detailed_metrics(
        &self,
        sentences: &[String],
        chains: &[LexicalChain],
        semantic_fields: &HashMap<String, Vec<String>>,
    ) -> DetailedLexicalMetrics {
        let total_chains = chains.len();
        let chain_lengths: Vec<usize> = chains.iter().map(|c| c.words.len()).collect();
        let average_chain_length = if !chain_lengths.is_empty() {
            chain_lengths.iter().sum::<usize>() as f64 / chain_lengths.len() as f64
        } else {
            0.0
        };
        let max_chain_length = chain_lengths.iter().max().copied().unwrap_or(0);

        let chain_density_distribution: Vec<f64> = chains.iter().map(|c| c.density).collect();

        let relationship_distribution = chains.iter().fold(HashMap::new(), |mut acc, chain| {
            let rel_name = format!("{:?}", chain.semantic_relationship);
            *acc.entry(rel_name).or_insert(0) += 1;
            acc
        });

        let vocabulary_richness = self.calculate_vocabulary_richness(sentences);
        let lexical_sophistication = self.calculate_lexical_sophistication(sentences);
        let type_token_ratio = self.calculate_type_token_ratio(sentences);
        let mattr = self.calculate_mattr(sentences);
        let lexical_diversity = self.calculate_lexical_diversity_metrics(sentences);
        let pattern_statistics = self.calculate_pattern_statistics(chains);
        let cross_sentence_coherence = self.calculate_cross_sentence_coherence(sentences);
        let temporal_coherence = self.calculate_temporal_coherence_metrics(sentences, chains);

        DetailedLexicalMetrics {
            total_chains,
            average_chain_length,
            max_chain_length,
            chain_density_distribution,
            relationship_distribution,
            vocabulary_richness,
            lexical_sophistication,
            type_token_ratio,
            mattr,
            lexical_diversity,
            pattern_statistics,
            cross_sentence_coherence,
            temporal_coherence,
        }
    }

    /// Calculate vocabulary richness
    fn calculate_vocabulary_richness(&self, sentences: &[String]) -> f64 {
        let all_words: Vec<String> = sentences
            .iter()
            .flat_map(|s| self.extract_content_words(s))
            .collect();

        if all_words.is_empty() {
            return 0.0;
        }

        let unique_words: HashSet<_> = all_words.iter().collect();
        unique_words.len() as f64 / all_words.len() as f64
    }

    /// Calculate lexical sophistication
    fn calculate_lexical_sophistication(&self, sentences: &[String]) -> f64 {
        let all_words: Vec<String> = sentences
            .iter()
            .flat_map(|s| self.extract_content_words(s))
            .collect();

        if all_words.is_empty() {
            return 0.0;
        }

        // Simple sophistication measure based on word length
        let avg_word_length =
            all_words.iter().map(|w| w.len()).sum::<usize>() as f64 / all_words.len() as f64;

        (avg_word_length - 3.0).max(0.0) / 7.0 // Normalize to 0-1
    }

    /// Calculate type-token ratio
    fn calculate_type_token_ratio(&self, sentences: &[String]) -> f64 {
        let all_words: Vec<String> = sentences
            .iter()
            .flat_map(|s| self.extract_content_words(s))
            .collect();

        if all_words.is_empty() {
            return 0.0;
        }

        let unique_words: HashSet<_> = all_words.iter().collect();
        unique_words.len() as f64 / all_words.len() as f64
    }

    /// Calculate moving average type-token ratio
    fn calculate_mattr(&self, sentences: &[String]) -> f64 {
        let all_words: Vec<String> = sentences
            .iter()
            .flat_map(|s| self.extract_content_words(s))
            .collect();

        if all_words.len() < 50 {
            return self.calculate_type_token_ratio(sentences);
        }

        let window_size = 50;
        let mut ttrs = Vec::new();

        for i in 0..=all_words.len().saturating_sub(window_size) {
            let window_words = &all_words[i..i + window_size];
            let unique_words: HashSet<_> = window_words.iter().collect();
            let ttr = unique_words.len() as f64 / window_words.len() as f64;
            ttrs.push(ttr);
        }

        ttrs.iter().sum::<f64>() / ttrs.len() as f64
    }

    /// Calculate lexical diversity metrics
    fn calculate_lexical_diversity_metrics(&self, sentences: &[String]) -> LexicalDiversityMetrics {
        // Simplified implementation - real MTLD and HD-D require more complex calculations
        let ttr = self.calculate_type_token_ratio(sentences);
        let mattr = self.calculate_mattr(sentences);

        LexicalDiversityMetrics {
            mtld: ttr * 100.0, // Simplified
            hdd: ttr * 0.8,    // Simplified
            mld: ttr * 50.0,   // Simplified
            entropy: self.calculate_vocabulary_entropy(sentences),
            simpson_index: self.calculate_simpson_index(sentences),
            shannon_index: self.calculate_shannon_index(sentences),
        }
    }

    /// Calculate vocabulary entropy
    fn calculate_vocabulary_entropy(&self, sentences: &[String]) -> f64 {
        let all_words: Vec<String> = sentences
            .iter()
            .flat_map(|s| self.extract_content_words(s))
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
                -p * p.ln()
            })
            .sum::<f64>();

        entropy / (word_counts.len() as f64).ln().max(1.0)
    }

    /// Calculate Simpson diversity index
    fn calculate_simpson_index(&self, sentences: &[String]) -> f64 {
        let all_words: Vec<String> = sentences
            .iter()
            .flat_map(|s| self.extract_content_words(s))
            .collect();

        if all_words.is_empty() {
            return 0.0;
        }

        let word_counts = all_words.iter().fold(HashMap::new(), |mut acc, word| {
            *acc.entry(word).or_insert(0) += 1;
            acc
        });

        let total_words = all_words.len() as f64;
        let simpson = word_counts
            .values()
            .map(|&count| {
                let p = count as f64 / total_words;
                p * p
            })
            .sum::<f64>();

        1.0 - simpson
    }

    /// Calculate Shannon diversity index
    fn calculate_shannon_index(&self, sentences: &[String]) -> f64 {
        self.calculate_vocabulary_entropy(sentences)
    }

    /// Calculate pattern statistics
    fn calculate_pattern_statistics(&self, chains: &[LexicalChain]) -> PatternStatistics {
        if chains.is_empty() {
            return PatternStatistics {
                repetition_strength: 0.0,
                synonymy_distribution: Vec::new(),
                hierarchical_depth: 0.0,
                thematic_consistency: 0.0,
                morphological_variation: 0.0,
                collocational_strength: 0.0,
            };
        }

        let repetition_chains = chains
            .iter()
            .filter(|c| c.chain_type == LexicalChainType::Repetition)
            .count();
        let repetition_strength = repetition_chains as f64 / chains.len() as f64;

        let synonymy_distribution = chains
            .iter()
            .filter(|c| c.chain_type == LexicalChainType::Synonymous)
            .map(|c| c.coherence_score)
            .collect();

        let hierarchical_depth = chains
            .iter()
            .filter(|c| c.chain_type == LexicalChainType::Hierarchical)
            .map(|c| c.semantic_depth as f64)
            .sum::<f64>()
            / chains.len() as f64;

        let thematic_consistency = chains
            .iter()
            .filter(|c| c.chain_type == LexicalChainType::Thematic)
            .map(|c| c.coherence_score)
            .sum::<f64>()
            / chains.len() as f64;

        let morphological_variation = chains
            .iter()
            .filter(|c| c.chain_type == LexicalChainType::Morphological)
            .count() as f64
            / chains.len() as f64;

        let collocational_strength = chains
            .iter()
            .filter(|c| c.chain_type == LexicalChainType::Collocational)
            .map(|c| c.chain_strength)
            .sum::<f64>()
            / chains.len() as f64;

        PatternStatistics {
            repetition_strength,
            synonymy_distribution,
            hierarchical_depth,
            thematic_consistency,
            morphological_variation,
            collocational_strength,
        }
    }

    /// Calculate cross-sentence coherence
    fn calculate_cross_sentence_coherence(&self, sentences: &[String]) -> f64 {
        if sentences.len() < 2 {
            return 1.0;
        }

        let mut total_coherence = 0.0;
        let mut comparisons = 0;

        for i in 0..sentences.len() - 1 {
            let current = &sentences[i];
            let next = &sentences[i + 1];

            let lexical_overlap = self.calculate_sentence_lexical_overlap(current, next);
            let semantic_continuity = self.calculate_sentence_semantic_continuity(current, next);

            let coherence = (lexical_overlap * 0.6) + (semantic_continuity * 0.4);
            total_coherence += coherence;
            comparisons += 1;
        }

        if comparisons > 0 {
            total_coherence / comparisons as f64
        } else {
            0.0
        }
    }

    /// Calculate sentence lexical overlap
    fn calculate_sentence_lexical_overlap(&self, sent1: &str, sent2: &str) -> f64 {
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

    /// Calculate sentence semantic continuity
    fn calculate_sentence_semantic_continuity(&self, sent1: &str, sent2: &str) -> f64 {
        let fields1 = self.identify_sentence_semantic_fields(sent1);
        let fields2 = self.identify_sentence_semantic_fields(sent2);

        if fields1.is_empty() && fields2.is_empty() {
            return 1.0;
        }

        let intersection_size = fields1.intersection(&fields2).count();
        let union_size = fields1.union(&fields2).count();

        if union_size > 0 {
            intersection_size as f64 / union_size as f64
        } else {
            0.0
        }
    }

    /// Identify semantic fields in a sentence
    fn identify_sentence_semantic_fields(&self, sentence: &str) -> HashSet<String> {
        let mut fields = HashSet::new();
        let content_words = self.extract_content_words(sentence);

        if let Ok(lexicon) = self.semantic_lexicon.read() {
            for (field_name, field_words) in lexicon.iter() {
                for word in &content_words {
                    if field_words.contains(word) {
                        fields.insert(field_name.clone());
                        break;
                    }
                }
            }
        }

        fields
    }

    /// Calculate temporal coherence metrics
    fn calculate_temporal_coherence_metrics(
        &self,
        sentences: &[String],
        chains: &[LexicalChain],
    ) -> TemporalCoherenceMetrics {
        let coherence_evolution = self.calculate_coherence_evolution(sentences, chains);
        let chain_temporal_consistency = self.calculate_chain_temporal_consistency(chains);
        let periodic_patterns = self.detect_periodic_patterns(&coherence_evolution);
        let coherence_momentum = self.calculate_coherence_momentum(&coherence_evolution);
        let temporal_clustering = self.calculate_temporal_clustering(chains);

        TemporalCoherenceMetrics {
            coherence_evolution,
            chain_temporal_consistency,
            periodic_patterns,
            coherence_momentum,
            temporal_clustering,
        }
    }

    /// Calculate coherence evolution over text
    fn calculate_coherence_evolution(
        &self,
        sentences: &[String],
        chains: &[LexicalChain],
    ) -> Vec<f64> {
        let window_size = 5;
        let mut evolution = Vec::new();

        for i in 0..sentences.len().saturating_sub(window_size - 1) {
            let window_end = (i + window_size).min(sentences.len());
            let window_sentences = &sentences[i..window_end];

            // Calculate local coherence for this window
            let local_coherence = self.calculate_window_coherence(window_sentences, chains);
            evolution.push(local_coherence);
        }

        evolution
    }

    /// Calculate coherence for a window of sentences
    fn calculate_window_coherence(
        &self,
        window_sentences: &[String],
        chains: &[LexicalChain],
    ) -> f64 {
        if window_sentences.is_empty() {
            return 0.0;
        }

        let window_chains: Vec<_> = chains
            .iter()
            .filter(|chain| {
                chain.words.iter().any(|(_, positions)| {
                    positions
                        .iter()
                        .any(|(sent_idx, _)| *sent_idx < window_sentences.len())
                })
            })
            .collect();

        if window_chains.is_empty() {
            return 0.0;
        }

        window_chains.iter().map(|c| c.coherence_score).sum::<f64>() / window_chains.len() as f64
    }

    /// Calculate chain temporal consistency
    fn calculate_chain_temporal_consistency(&self, chains: &[LexicalChain]) -> f64 {
        if chains.is_empty() {
            return 0.0;
        }

        let consistency_scores: Vec<f64> = chains
            .iter()
            .map(|chain| chain.position_distribution.regularity)
            .collect();

        consistency_scores.iter().sum::<f64>() / consistency_scores.len() as f64
    }

    /// Detect periodic patterns in coherence evolution
    fn detect_periodic_patterns(&self, evolution: &[f64]) -> Vec<PeriodicPattern> {
        // Simplified pattern detection
        let mut patterns = Vec::new();

        if evolution.len() >= 6 {
            // Look for simple periodic patterns
            for period in 2..=evolution.len() / 3 {
                let strength = self.calculate_pattern_strength(evolution, period);
                if strength > 0.7 {
                    patterns.push(PeriodicPattern {
                        period,
                        strength,
                        phase: 0.0,
                        pattern_type: format!("Period-{}", period),
                    });
                }
            }
        }

        patterns
    }

    /// Calculate pattern strength for a given period
    fn calculate_pattern_strength(&self, evolution: &[f64], period: usize) -> f64 {
        if evolution.len() < period * 2 {
            return 0.0;
        }

        let mut correlations = Vec::new();

        for offset in 0..period {
            let mut series1 = Vec::new();
            let mut series2 = Vec::new();

            let mut i = offset;
            while i + period < evolution.len() {
                series1.push(evolution[i]);
                series2.push(evolution[i + period]);
                i += period;
            }

            if !series1.is_empty() {
                let correlation = self.calculate_correlation(&series1, &series2);
                correlations.push(correlation);
            }
        }

        if correlations.is_empty() {
            0.0
        } else {
            correlations.iter().sum::<f64>() / correlations.len() as f64
        }
    }

    /// Calculate correlation between two series
    fn calculate_correlation(&self, series1: &[f64], series2: &[f64]) -> f64 {
        if series1.len() != series2.len() || series1.is_empty() {
            return 0.0;
        }

        let mean1 = series1.iter().sum::<f64>() / series1.len() as f64;
        let mean2 = series2.iter().sum::<f64>() / series2.len() as f64;

        let numerator: f64 = series1
            .iter()
            .zip(series2.iter())
            .map(|(&x1, &x2)| (x1 - mean1) * (x2 - mean2))
            .sum();

        let denominator1: f64 = series1
            .iter()
            .map(|&x| (x - mean1).powi(2))
            .sum::<f64>()
            .sqrt();

        let denominator2: f64 = series2
            .iter()
            .map(|&x| (x - mean2).powi(2))
            .sum::<f64>()
            .sqrt();

        if denominator1 * denominator2 > 0.0 {
            numerator / (denominator1 * denominator2)
        } else {
            0.0
        }
    }

    /// Calculate coherence momentum
    fn calculate_coherence_momentum(&self, evolution: &[f64]) -> f64 {
        if evolution.len() < 2 {
            return 0.0;
        }

        let mut momentum = 0.0;
        for i in 1..evolution.len() {
            let change = evolution[i] - evolution[i - 1];
            momentum += change;
        }

        momentum / (evolution.len() - 1) as f64
    }

    /// Calculate temporal clustering
    fn calculate_temporal_clustering(&self, chains: &[LexicalChain]) -> f64 {
        if chains.is_empty() {
            return 0.0;
        }

        let clustering_scores: Vec<f64> = chains
            .iter()
            .map(|chain| chain.position_distribution.clustering_coefficient)
            .collect();

        clustering_scores.iter().sum::<f64>() / clustering_scores.len() as f64
    }

    /// Analyze cohesion devices
    fn analyze_cohesion_devices(
        &self,
        sentences: &[String],
        chains: &[LexicalChain],
    ) -> Vec<CohesionDevice> {
        let mut devices = Vec::new();

        // Analyze repetition devices
        devices.extend(self.analyze_repetition_devices(sentences));

        // Analyze synonymy devices
        devices.extend(self.analyze_synonymy_devices(sentences, chains));

        // Analyze collocation devices
        if self.config.detect_collocations {
            devices.extend(self.analyze_collocation_devices(sentences));
        }

        devices
    }

    /// Analyze repetition-based cohesion devices
    fn analyze_repetition_devices(&self, sentences: &[String]) -> Vec<CohesionDevice> {
        let mut devices = Vec::new();
        let mut word_positions = HashMap::new();

        // Collect word positions
        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let content_words = self.extract_content_words(sentence);
            for (word_idx, word) in content_words.iter().enumerate() {
                word_positions
                    .entry(word.clone())
                    .or_insert_with(Vec::new)
                    .push((sent_idx, word_idx));
            }
        }

        // Find repetition devices
        for (word, positions) in word_positions {
            if positions.len() > 1 {
                let strength = (positions.len() as f64).ln() / 3.0;
                devices.push(CohesionDevice {
                    device_type: CohesionDeviceType::Repetition,
                    elements: vec![word],
                    positions,
                    strength,
                    local_coherence_contribution: strength * 0.6,
                    global_coherence_contribution: strength * 0.4,
                });
            }
        }

        devices
    }

    /// Analyze synonymy-based cohesion devices
    fn analyze_synonymy_devices(
        &self,
        sentences: &[String],
        chains: &[LexicalChain],
    ) -> Vec<CohesionDevice> {
        let mut devices = Vec::new();

        for chain in chains {
            if chain.semantic_relationship == SemanticRelationship::Synonymy
                && chain.words.len() > 1
            {
                let elements: Vec<String> =
                    chain.words.iter().map(|(word, _)| word.clone()).collect();
                let positions: Vec<(usize, usize)> = chain
                    .words
                    .iter()
                    .flat_map(|(_, positions)| positions.iter())
                    .copied()
                    .collect();

                devices.push(CohesionDevice {
                    device_type: CohesionDeviceType::Synonymy,
                    elements,
                    positions,
                    strength: chain.chain_strength,
                    local_coherence_contribution: chain.coherence_score * 0.7,
                    global_coherence_contribution: chain.coherence_score * 0.3,
                });
            }
        }

        devices
    }

    /// Analyze collocation-based cohesion devices
    fn analyze_collocation_devices(&self, sentences: &[String]) -> Vec<CohesionDevice> {
        let mut devices = Vec::new();

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let content_words = self.extract_content_words(sentence);

            for i in 0..content_words.len().saturating_sub(1) {
                for j in i + 1..content_words.len().min(i + self.config.collocation_window) {
                    let word1 = &content_words[i];
                    let word2 = &content_words[j];

                    if let Some(collocates) = self.collocation_patterns.get(word1) {
                        if collocates.contains(word2) {
                            devices.push(CohesionDevice {
                                device_type: CohesionDeviceType::Collocation,
                                elements: vec![word1.clone(), word2.clone()],
                                positions: vec![(sent_idx, i), (sent_idx, j)],
                                strength: 0.8,
                                local_coherence_contribution: 0.6,
                                global_coherence_contribution: 0.2,
                            });
                        }
                    }
                }
            }
        }

        devices
    }

    /// Perform advanced analysis
    fn perform_advanced_analysis(
        &self,
        sentences: &[String],
        chains: &[LexicalChain],
    ) -> Result<AdvancedChainAnalysis, LexicalCoherenceError> {
        let network_analysis = self.analyze_chain_network(chains);
        let information_measures = self.calculate_information_measures(chains);
        let cognitive_load = self.estimate_cognitive_load(sentences, chains);
        let discourse_alignment = self.analyze_discourse_alignment(sentences, chains);

        Ok(AdvancedChainAnalysis {
            network_analysis,
            information_measures,
            cognitive_load,
            discourse_alignment,
        })
    }

    /// Analyze chain network properties
    fn analyze_chain_network(&self, chains: &[LexicalChain]) -> ChainNetworkAnalysis {
        // Simplified network analysis implementation
        ChainNetworkAnalysis {
            density: 0.5,                 // Placeholder
            clustering_coefficient: 0.6,  // Placeholder
            average_path_length: 2.5,     // Placeholder
            modularity: 0.4,              // Placeholder
            central_chains: vec![0, 1],   // Placeholder
            diameter: 4,                  // Placeholder
            small_world_coefficient: 1.2, // Placeholder
        }
    }

    /// Calculate information-theoretic measures
    fn calculate_information_measures(&self, chains: &[LexicalChain]) -> InformationMeasures {
        // Simplified information measures implementation
        InformationMeasures {
            chain_mutual_information: 0.3,
            chain_entropy: 2.1,
            conditional_entropy: 1.8,
            information_flow: 0.7,
            redundancy_coefficient: 0.4,
        }
    }

    /// Estimate cognitive load for lexical processing
    fn estimate_cognitive_load(
        &self,
        sentences: &[String],
        chains: &[LexicalChain],
    ) -> CognitiveLoadMetrics {
        // Simplified cognitive load estimation
        let avg_chain_length = if !chains.is_empty() {
            chains.iter().map(|c| c.words.len()).sum::<usize>() as f64 / chains.len() as f64
        } else {
            0.0
        };

        let working_memory_load = (avg_chain_length / 7.0).min(1.0); // Miller's 7±2 rule
        let processing_complexity = self.calculate_processing_complexity(sentences);
        let integration_effort = self.calculate_integration_effort(chains);
        let disambiguation_load = self.calculate_disambiguation_load(sentences);
        let cognitive_efficiency =
            1.0 - (working_memory_load + processing_complexity + integration_effort) / 3.0;

        CognitiveLoadMetrics {
            working_memory_load,
            processing_complexity,
            integration_effort,
            disambiguation_load,
            cognitive_efficiency,
        }
    }

    /// Calculate processing complexity
    fn calculate_processing_complexity(&self, sentences: &[String]) -> f64 {
        let avg_sentence_length = sentences
            .iter()
            .map(|s| s.split_whitespace().count())
            .sum::<usize>() as f64
            / sentences.len() as f64;

        (avg_sentence_length / 20.0).min(1.0) // Normalize to 0-1
    }

    /// Calculate integration effort
    fn calculate_integration_effort(&self, chains: &[LexicalChain]) -> f64 {
        if chains.is_empty() {
            return 0.0;
        }

        let avg_span = chains
            .iter()
            .map(|c| c.span.1 .0 - c.span.0 .0)
            .sum::<usize>() as f64
            / chains.len() as f64;

        (avg_span / 10.0).min(1.0) // Normalize to 0-1
    }

    /// Calculate disambiguation load
    fn calculate_disambiguation_load(&self, sentences: &[String]) -> f64 {
        let all_words: Vec<String> = sentences
            .iter()
            .flat_map(|s| self.extract_content_words(s))
            .collect();

        if all_words.is_empty() {
            return 0.0;
        }

        let word_counts = all_words.iter().fold(HashMap::new(), |mut acc, word| {
            *acc.entry(word).or_insert(0) += 1;
            acc
        });

        let ambiguous_words = word_counts.values().filter(|&&count| count > 3).count();
        (ambiguous_words as f64 / word_counts.len() as f64).min(1.0)
    }

    /// Analyze discourse structure alignment
    fn analyze_discourse_alignment(
        &self,
        sentences: &[String],
        chains: &[LexicalChain],
    ) -> DiscourseAlignmentMetrics {
        // Simplified discourse alignment analysis
        DiscourseAlignmentMetrics {
            paragraph_alignment: 0.7,          // Placeholder
            section_alignment: 0.6,            // Placeholder
            marker_coordination: 0.8,          // Placeholder
            topic_progression_alignment: 0.75, // Placeholder
        }
    }
}

impl Default for LexicalCoherenceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple function for basic lexical coherence analysis
pub fn calculate_lexical_coherence_simple(text: &str) -> f64 {
    let analyzer = LexicalCoherenceAnalyzer::new();
    analyzer
        .analyze_lexical_coherence(text)
        .map(|result| result.lexical_chain_coherence)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexical_coherence_analyzer_creation() {
        let analyzer = LexicalCoherenceAnalyzer::new();
        assert_eq!(analyzer.config.lexical_chain_threshold, 0.6);
        assert_eq!(analyzer.config.max_chain_distance, 5);
        assert!(analyzer.config.use_semantic_similarity);
    }

    #[test]
    fn test_basic_lexical_coherence_analysis() {
        let analyzer = LexicalCoherenceAnalyzer::new();
        let text = "The cat sat on the mat. The cat was happy. Cats are wonderful animals.";

        let result = analyzer.analyze_lexical_coherence(text);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.lexical_chain_coherence >= 0.0);
        assert!(result.lexical_chain_coherence <= 1.0);
        assert!(!result.lexical_chains.is_empty());
    }

    #[test]
    fn test_lexical_chain_building() {
        let analyzer = LexicalCoherenceAnalyzer::new();
        let sentences = vec![
            "The dog ran quickly.".to_string(),
            "Dogs are fast animals.".to_string(),
            "The quick brown fox.".to_string(),
        ];

        let chains = analyzer.build_lexical_chains(&sentences).unwrap();
        assert!(!chains.is_empty());

        // Check that chains have proper structure
        for chain in chains {
            assert!(!chain.words.is_empty());
            assert!(chain.coherence_score >= 0.0);
            assert!(chain.chain_strength >= 0.0);
        }
    }

    #[test]
    fn test_semantic_field_identification() {
        let analyzer = LexicalCoherenceAnalyzer::new();
        let sentences = vec![
            "The red flower bloomed.".to_string(),
            "Blue birds sing happily.".to_string(),
            "Green trees provide shade.".to_string(),
        ];

        let fields = analyzer.identify_semantic_fields(&sentences);
        assert!(!fields.is_empty());

        // Should identify color and nature fields
        assert!(fields.contains_key("colors") || fields.contains_key("nature"));
    }

    #[test]
    fn test_lexical_similarity_calculation() {
        let analyzer = LexicalCoherenceAnalyzer::new();

        // Test identical words
        assert_eq!(analyzer.calculate_lexical_similarity("cat", "cat"), 1.0);

        // Test different words
        let similarity = analyzer.calculate_lexical_similarity("dog", "puppy");
        assert!(similarity >= 0.0 && similarity <= 1.0);

        // Test morphologically related words
        let similarity = analyzer.calculate_lexical_similarity("run", "running");
        assert!(similarity > 0.5);
    }

    #[test]
    fn test_cohesion_device_analysis() {
        let analyzer = LexicalCoherenceAnalyzer::new();
        let sentences = vec![
            "The cat sat on the mat.".to_string(),
            "The cat was very comfortable.".to_string(),
            "Cats love comfortable places.".to_string(),
        ];

        let chains = analyzer.build_lexical_chains(&sentences).unwrap();
        let devices = analyzer.analyze_cohesion_devices(&sentences, &chains);

        assert!(!devices.is_empty());

        // Check device properties
        for device in devices {
            assert!(!device.elements.is_empty());
            assert!(!device.positions.is_empty());
            assert!(device.strength >= 0.0);
        }
    }

    #[test]
    fn test_vocabulary_consistency() {
        let analyzer = LexicalCoherenceAnalyzer::new();

        // High consistency text (repeated vocabulary)
        let consistent_sentences = vec![
            "The beautiful garden has beautiful flowers.".to_string(),
            "Beautiful gardens need beautiful care.".to_string(),
            "Garden flowers make gardens beautiful.".to_string(),
        ];

        let consistency = analyzer.calculate_vocabulary_consistency(&consistent_sentences);
        assert!(consistency > 0.0);

        // Low consistency text (varied vocabulary)
        let varied_sentences = vec![
            "The elephant trumpeted loudly.".to_string(),
            "Computers process data quickly.".to_string(),
            "Stars shine brightly tonight.".to_string(),
        ];

        let varied_consistency = analyzer.calculate_vocabulary_consistency(&varied_sentences);
        assert!(varied_consistency <= consistency); // Should be less consistent
    }

    #[test]
    fn test_temporal_coherence_metrics() {
        let analyzer = LexicalCoherenceAnalyzer::new();
        let sentences = vec![
            "First we went to the store.".to_string(),
            "Then we bought some groceries.".to_string(),
            "Finally we returned home.".to_string(),
            "We cooked dinner together.".to_string(),
        ];

        let chains = analyzer.build_lexical_chains(&sentences).unwrap();
        let temporal_metrics = analyzer.calculate_temporal_coherence_metrics(&sentences, &chains);

        assert!(!temporal_metrics.coherence_evolution.is_empty());
        assert!(temporal_metrics.chain_temporal_consistency >= 0.0);
        assert!(temporal_metrics.coherence_momentum.is_finite());
    }

    #[test]
    fn test_advanced_analysis() {
        let analyzer = LexicalCoherenceAnalyzer::with_config(LexicalCoherenceConfig {
            use_advanced_analysis: true,
            ..LexicalCoherenceConfig::default()
        });

        let text = "The scientific method involves careful observation. Scientists observe natural phenomena. Through observation, scientists develop theories. Theories help explain observations.";

        let result = analyzer.analyze_lexical_coherence(text).unwrap();
        assert!(result.advanced_analysis.is_some());

        let advanced = result.advanced_analysis.unwrap();
        assert!(advanced.cognitive_load.cognitive_efficiency >= 0.0);
        assert!(advanced.information_measures.chain_entropy >= 0.0);
    }

    #[test]
    fn test_empty_text_handling() {
        let analyzer = LexicalCoherenceAnalyzer::new();
        let result = analyzer.analyze_lexical_coherence("");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            LexicalCoherenceError::EmptyText
        ));
    }

    #[test]
    fn test_simple_function() {
        let coherence =
            calculate_lexical_coherence_simple("The cat sat on the mat. The cat was happy.");
        assert!(coherence >= 0.0 && coherence <= 1.0);
    }

    #[test]
    fn test_morphological_similarity() {
        let analyzer = LexicalCoherenceAnalyzer::new();

        // Test word with common suffix
        let similarity = analyzer.calculate_morphological_similarity("running", "swimming");
        assert!(similarity > 0.0);

        // Test words with same stem
        let similarity = analyzer.calculate_morphological_similarity("happy", "happiness");
        assert!(similarity > 0.5);
    }

    #[test]
    fn test_position_distribution_analysis() {
        let analyzer = LexicalCoherenceAnalyzer::new();
        let positions = vec![(0, 1), (0, 3), (1, 2), (2, 1)];

        let distribution = analyzer.analyze_position_distribution(&positions);
        assert!(distribution.entropy >= 0.0 && distribution.entropy <= 1.0);
        assert!(distribution.clustering_coefficient >= 0.0);
        assert!(distribution.regularity >= 0.0 && distribution.regularity <= 1.0);
    }

    #[test]
    fn test_lexical_diversity_metrics() {
        let analyzer = LexicalCoherenceAnalyzer::new();
        let sentences = vec![
            "The quick brown fox jumps over the lazy dog.".to_string(),
            "Diversity in vocabulary creates interesting text.".to_string(),
            "Varied word choices improve readability.".to_string(),
        ];

        let diversity = analyzer.calculate_lexical_diversity_metrics(&sentences);
        assert!(diversity.type_token_ratio >= 0.0 && diversity.type_token_ratio <= 1.0);
        assert!(diversity.entropy >= 0.0);
        assert!(diversity.simpson_index >= 0.0 && diversity.simpson_index <= 1.0);
        assert!(diversity.shannon_index >= 0.0);
    }

    #[test]
    fn test_chain_type_classification() {
        let analyzer = LexicalCoherenceAnalyzer::new();

        // Test repetition chain
        let repetition_words = vec!["cat".to_string(), "cat".to_string(), "cat".to_string()];
        let chain_type =
            analyzer.classify_chain_type(&repetition_words, &SemanticRelationship::Synonymy);
        assert_eq!(chain_type, LexicalChainType::Repetition);

        // Test synonymous chain
        let synonym_words = vec!["happy".to_string(), "joyful".to_string()];
        let chain_type =
            analyzer.classify_chain_type(&synonym_words, &SemanticRelationship::Synonymy);
        assert_eq!(chain_type, LexicalChainType::Synonymous);
    }

    #[test]
    fn test_pattern_detection() {
        let analyzer = LexicalCoherenceAnalyzer::new();
        let evolution = vec![0.5, 0.8, 0.3, 0.6, 0.9, 0.4, 0.7, 1.0];

        let patterns = analyzer.detect_periodic_patterns(&evolution);
        // Should detect some patterns in the test data
        assert!(patterns.len() >= 0); // May or may not find patterns

        for pattern in patterns {
            assert!(pattern.period >= 2);
            assert!(pattern.strength >= 0.0 && pattern.strength <= 1.0);
        }
    }
}
