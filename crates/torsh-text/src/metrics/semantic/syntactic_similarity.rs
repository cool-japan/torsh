//! Advanced syntactic similarity analysis and pattern matching system
//!
//! This module provides comprehensive syntactic pattern analysis capabilities for text comparison,
//! supporting multiple syntactic similarity approaches, structural analysis, and grammar pattern matching.
//! Designed for production use with extensive configuration options and robust error handling.

use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_core::random::{rng, Random};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use thiserror::Error;

/// Errors that can occur during syntactic similarity analysis
#[derive(Error, Debug)]
pub enum SyntacticSimilarityError {
    #[error("Invalid syntactic configuration: {message}")]
    InvalidConfiguration { message: String },

    #[error("Syntactic analysis failed: {reason}")]
    AnalysisFailed { reason: String },

    #[error("Pattern extraction failed: {error}")]
    PatternExtractionFailed { error: String },

    #[error("POS tagging failed: {reason}")]
    PosTaggingFailed { reason: String },

    #[error("Structural analysis failed: {error}")]
    StructuralAnalysisFailed { error: String },

    #[error("Dependency parsing failed: {reason}")]
    DependencyParsingFailed { reason: String },

    #[error("Similarity computation failed: {error}")]
    SimilarityComputationFailed { error: String },

    #[error("Tree comparison failed: {reason}")]
    TreeComparisonFailed { reason: String },
}

/// Syntactic similarity approaches
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyntacticApproach {
    /// Part-of-speech sequence similarity
    PosSequence,
    /// Sentence structure complexity comparison
    StructuralComplexity,
    /// Grammar pattern matching
    GrammarPattern,
    /// Phrase structure analysis
    PhraseStructure,
    /// Dependency relationship comparison
    DependencyStructure,
    /// Syntactic tree similarity
    TreeSimilarity,
    /// Combined multi-dimensional syntactic analysis
    Comprehensive,
}

/// Part-of-speech tags (simplified set)
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PosTag {
    Noun,
    Verb,
    Adjective,
    Adverb,
    Preposition,
    Pronoun,
    Determiner,
    Conjunction,
    Interjection,
    Punctuation,
    Unknown,
}

impl PosTag {
    /// Simple heuristic POS tagging based on word patterns and context
    pub fn tag_word(word: &str, context: Option<&str>) -> PosTag {
        let word_lower = word.to_lowercase();

        // Punctuation
        if word.chars().all(|c| c.is_ascii_punctuation()) {
            return PosTag::Punctuation;
        }

        // Common determiners
        if matches!(
            word_lower.as_str(),
            "the"
                | "a"
                | "an"
                | "this"
                | "that"
                | "these"
                | "those"
                | "my"
                | "your"
                | "his"
                | "her"
                | "its"
                | "our"
                | "their"
        ) {
            return PosTag::Determiner;
        }

        // Common prepositions
        if matches!(
            word_lower.as_str(),
            "in" | "on"
                | "at"
                | "by"
                | "for"
                | "with"
                | "to"
                | "of"
                | "from"
                | "into"
                | "onto"
                | "upon"
        ) {
            return PosTag::Preposition;
        }

        // Common pronouns
        if matches!(
            word_lower.as_str(),
            "i" | "you"
                | "he"
                | "she"
                | "it"
                | "we"
                | "they"
                | "me"
                | "him"
                | "her"
                | "us"
                | "them"
        ) {
            return PosTag::Pronoun;
        }

        // Common conjunctions
        if matches!(
            word_lower.as_str(),
            "and"
                | "or"
                | "but"
                | "so"
                | "yet"
                | "nor"
                | "for"
                | "because"
                | "although"
                | "while"
                | "since"
        ) {
            return PosTag::Conjunction;
        }

        // Verb patterns (simplified)
        if word_lower.ends_with("ing")
            || word_lower.ends_with("ed")
            || word_lower.ends_with("s")
            || matches!(
                word_lower.as_str(),
                "is" | "are"
                    | "was"
                    | "were"
                    | "be"
                    | "been"
                    | "have"
                    | "has"
                    | "had"
                    | "do"
                    | "does"
                    | "did"
            )
        {
            return PosTag::Verb;
        }

        // Adverb patterns
        if word_lower.ends_with("ly") {
            return PosTag::Adverb;
        }

        // Adjective patterns (basic heuristics)
        if word_lower.ends_with("ous")
            || word_lower.ends_with("ful")
            || word_lower.ends_with("less")
            || word_lower.ends_with("able")
            || word_lower.ends_with("ible")
            || word_lower.ends_with("al")
        {
            return PosTag::Adjective;
        }

        // Default to noun for other words
        PosTag::Noun
    }

    /// Get all possible POS tags
    pub fn all_tags() -> Vec<PosTag> {
        vec![
            PosTag::Noun,
            PosTag::Verb,
            PosTag::Adjective,
            PosTag::Adverb,
            PosTag::Preposition,
            PosTag::Pronoun,
            PosTag::Determiner,
            PosTag::Conjunction,
            PosTag::Interjection,
            PosTag::Punctuation,
        ]
    }
}

impl std::fmt::Display for PosTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Dependency relation types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DependencyRelation {
    Subject,
    Object,
    Modifier,
    Complement,
    Determiner,
    Preposition,
    Conjunction,
    Root,
    Unknown,
}

/// Dependency arc representing relationships between words
#[derive(Debug, Clone)]
pub struct DependencyArc {
    pub head_index: usize,
    pub dependent_index: usize,
    pub relation: DependencyRelation,
    pub head_word: String,
    pub dependent_word: String,
}

/// Syntactic tree node for structural analysis
#[derive(Debug, Clone)]
pub struct SyntacticTreeNode {
    pub word: String,
    pub pos_tag: PosTag,
    pub index: usize,
    pub children: Vec<usize>,
    pub parent: Option<usize>,
    pub depth: usize,
}

/// Syntactic pattern representing common structural patterns
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SyntacticPattern {
    pub pattern_type: String,
    pub pos_sequence: Vec<PosTag>,
    pub pattern_weight: f64,
    pub complexity_score: f64,
}

/// Result of syntactic similarity analysis
#[derive(Debug, Clone)]
pub struct SyntacticSimilarityResult {
    /// Overall syntactic similarity score (0.0-1.0)
    pub similarity_score: f64,
    /// Similarity scores by approach
    pub approach_scores: HashMap<SyntacticApproach, f64>,
    /// POS sequence similarity
    pub pos_similarity: f64,
    /// Structural complexity similarity
    pub structure_similarity: f64,
    /// Grammar pattern similarity
    pub pattern_similarity: f64,
    /// Phrase structure similarity
    pub phrase_similarity: f64,
    /// Dependency structure similarity
    pub dependency_similarity: f64,
    /// Tree structure similarity
    pub tree_similarity: f64,
    /// Identified common patterns
    pub common_patterns: Vec<SyntacticPattern>,
    /// Structural differences
    pub structural_differences: Vec<String>,
    /// Analysis metadata
    pub metadata: SyntacticAnalysisMetadata,
}

/// Metadata about syntactic analysis
#[derive(Debug, Clone)]
pub struct SyntacticAnalysisMetadata {
    /// Analysis approach used
    pub approach: SyntacticApproach,
    /// Analysis timestamp
    pub timestamp: std::time::SystemTime,
    /// Processing time in milliseconds
    pub processing_time_ms: u128,
    /// Number of syntactic patterns analyzed
    pub pattern_count: usize,
    /// Structural complexity of text1
    pub text1_complexity: f64,
    /// Structural complexity of text2
    pub text2_complexity: f64,
    /// Quality score of analysis (0.0-1.0)
    pub quality_score: f64,
    /// Tree depth analyzed
    pub tree_depth: usize,
    /// Number of dependency relations found
    pub dependency_count: usize,
}

/// Configuration for syntactic similarity analysis
#[derive(Debug, Clone)]
pub struct SyntacticSimilarityConfig {
    /// Primary analysis approach
    pub approach: SyntacticApproach,
    /// Weight for POS sequence similarity
    pub pos_weight: f64,
    /// Weight for structural complexity similarity
    pub structure_weight: f64,
    /// Weight for grammar pattern similarity
    pub pattern_weight: f64,
    /// Weight for phrase structure similarity
    pub phrase_weight: f64,
    /// Weight for dependency similarity
    pub dependency_weight: f64,
    /// Weight for tree similarity
    pub tree_weight: f64,
    /// Minimum pattern length to consider
    pub min_pattern_length: usize,
    /// Maximum tree depth to analyze
    pub max_tree_depth: usize,
    /// Enable detailed pattern analysis
    pub detailed_patterns: bool,
    /// Enable dependency parsing
    pub enable_dependencies: bool,
    /// Enable tree construction
    pub enable_tree_analysis: bool,
}

impl Default for SyntacticSimilarityConfig {
    fn default() -> Self {
        Self {
            approach: SyntacticApproach::Comprehensive,
            pos_weight: 0.2,
            structure_weight: 0.2,
            pattern_weight: 0.15,
            phrase_weight: 0.15,
            dependency_weight: 0.15,
            tree_weight: 0.15,
            min_pattern_length: 2,
            max_tree_depth: 10,
            detailed_patterns: true,
            enable_dependencies: true,
            enable_tree_analysis: true,
        }
    }
}

impl SyntacticSimilarityConfig {
    /// Create a new configuration builder
    pub fn builder() -> SyntacticSimilarityConfigBuilder {
        SyntacticSimilarityConfigBuilder::new()
    }

    /// Validate configuration weights sum to approximately 1.0
    pub fn validate_weights(&self) -> Result<(), SyntacticSimilarityError> {
        let total_weight = self.pos_weight
            + self.structure_weight
            + self.pattern_weight
            + self.phrase_weight
            + self.dependency_weight
            + self.tree_weight;

        if (total_weight - 1.0).abs() > 0.1 {
            return Err(SyntacticSimilarityError::InvalidConfiguration {
                message: format!(
                    "Weights sum to {:.2}, should sum to approximately 1.0",
                    total_weight
                ),
            });
        }

        Ok(())
    }
}

/// Builder for syntactic similarity configuration
pub struct SyntacticSimilarityConfigBuilder {
    config: SyntacticSimilarityConfig,
}

impl SyntacticSimilarityConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: SyntacticSimilarityConfig::default(),
        }
    }

    pub fn approach(mut self, approach: SyntacticApproach) -> Self {
        self.config.approach = approach;
        self
    }

    pub fn pos_weight(mut self, weight: f64) -> Self {
        self.config.pos_weight = weight.clamp(0.0, 1.0);
        self
    }

    pub fn structure_weight(mut self, weight: f64) -> Self {
        self.config.structure_weight = weight.clamp(0.0, 1.0);
        self
    }

    pub fn pattern_weight(mut self, weight: f64) -> Self {
        self.config.pattern_weight = weight.clamp(0.0, 1.0);
        self
    }

    pub fn phrase_weight(mut self, weight: f64) -> Self {
        self.config.phrase_weight = weight.clamp(0.0, 1.0);
        self
    }

    pub fn dependency_weight(mut self, weight: f64) -> Self {
        self.config.dependency_weight = weight.clamp(0.0, 1.0);
        self
    }

    pub fn tree_weight(mut self, weight: f64) -> Self {
        self.config.tree_weight = weight.clamp(0.0, 1.0);
        self
    }

    pub fn min_pattern_length(mut self, length: usize) -> Self {
        self.config.min_pattern_length = length.max(1);
        self
    }

    pub fn max_tree_depth(mut self, depth: usize) -> Self {
        self.config.max_tree_depth = depth.max(1);
        self
    }

    pub fn detailed_patterns(mut self, enable: bool) -> Self {
        self.config.detailed_patterns = enable;
        self
    }

    pub fn enable_dependencies(mut self, enable: bool) -> Self {
        self.config.enable_dependencies = enable;
        self
    }

    pub fn enable_tree_analysis(mut self, enable: bool) -> Self {
        self.config.enable_tree_analysis = enable;
        self
    }

    pub fn build(self) -> Result<SyntacticSimilarityConfig, SyntacticSimilarityError> {
        self.config.validate_weights()?;
        Ok(self.config)
    }
}

/// Advanced syntactic similarity analyzer
pub struct SyntacticSimilarityAnalyzer {
    config: SyntacticSimilarityConfig,
    pattern_cache: HashMap<String, Vec<SyntacticPattern>>,
    dependency_rules: HashMap<String, Vec<DependencyRelation>>,
}

impl SyntacticSimilarityAnalyzer {
    /// Create a new syntactic similarity analyzer with the given configuration
    pub fn new(config: SyntacticSimilarityConfig) -> Result<Self, SyntacticSimilarityError> {
        config.validate_weights()?;

        let mut analyzer = Self {
            config,
            pattern_cache: HashMap::new(),
            dependency_rules: HashMap::new(),
        };

        analyzer.initialize_dependency_rules();

        Ok(analyzer)
    }

    /// Create a syntactic similarity analyzer with default configuration
    pub fn default() -> Result<Self, SyntacticSimilarityError> {
        Self::new(SyntacticSimilarityConfig::default())
    }

    /// Analyze syntactic similarity between two texts
    pub fn analyze_similarity(
        &mut self,
        text1: &str,
        text2: &str,
    ) -> Result<SyntacticSimilarityResult, SyntacticSimilarityError> {
        if text1.trim().is_empty() || text2.trim().is_empty() {
            return Err(SyntacticSimilarityError::AnalysisFailed {
                reason: "Empty text provided".to_string(),
            });
        }

        let start_time = std::time::Instant::now();

        // Tokenize and POS tag both texts
        let tokens1 = self.tokenize_and_tag(text1)?;
        let tokens2 = self.tokenize_and_tag(text2)?;

        // Calculate individual similarity components based on approach
        let mut approach_scores = HashMap::new();

        let pos_similarity = self.calculate_pos_similarity(&tokens1, &tokens2)?;
        approach_scores.insert(SyntacticApproach::PosSequence, pos_similarity);

        let structure_similarity = self.calculate_structure_similarity(&tokens1, &tokens2)?;
        approach_scores.insert(
            SyntacticApproach::StructuralComplexity,
            structure_similarity,
        );

        let pattern_similarity = self.calculate_pattern_similarity(&tokens1, &tokens2)?;
        approach_scores.insert(SyntacticApproach::GrammarPattern, pattern_similarity);

        let phrase_similarity = self.calculate_phrase_similarity(&tokens1, &tokens2)?;
        approach_scores.insert(SyntacticApproach::PhraseStructure, phrase_similarity);

        let dependency_similarity = if self.config.enable_dependencies {
            self.calculate_dependency_similarity(&tokens1, &tokens2)?
        } else {
            0.0
        };
        approach_scores.insert(
            SyntacticApproach::DependencyStructure,
            dependency_similarity,
        );

        let tree_similarity = if self.config.enable_tree_analysis {
            self.calculate_tree_similarity(&tokens1, &tokens2)?
        } else {
            0.0
        };
        approach_scores.insert(SyntacticApproach::TreeSimilarity, tree_similarity);

        // Calculate weighted overall similarity
        let overall_similarity = self.calculate_weighted_similarity(
            pos_similarity,
            structure_similarity,
            pattern_similarity,
            phrase_similarity,
            dependency_similarity,
            tree_similarity,
        );

        // Find common patterns
        let common_patterns = self.find_common_patterns(&tokens1, &tokens2)?;

        // Identify structural differences
        let structural_differences = self.identify_structural_differences(&tokens1, &tokens2);

        // Calculate complexity scores
        let text1_complexity = self.calculate_structural_complexity(&tokens1);
        let text2_complexity = self.calculate_structural_complexity(&tokens2);

        // Calculate quality score
        let quality_score = self.calculate_quality_score(&tokens1, &tokens2, &common_patterns);

        let processing_time = start_time.elapsed();

        let metadata = SyntacticAnalysisMetadata {
            approach: self.config.approach,
            timestamp: std::time::SystemTime::now(),
            processing_time_ms: processing_time.as_millis(),
            pattern_count: common_patterns.len(),
            text1_complexity,
            text2_complexity,
            quality_score,
            tree_depth: self.config.max_tree_depth,
            dependency_count: tokens1.len() + tokens2.len(),
        };

        Ok(SyntacticSimilarityResult {
            similarity_score: overall_similarity,
            approach_scores,
            pos_similarity,
            structure_similarity,
            pattern_similarity,
            phrase_similarity,
            dependency_similarity,
            tree_similarity,
            common_patterns,
            structural_differences,
            metadata,
        })
    }

    /// Tokenize text and assign POS tags
    fn tokenize_and_tag(
        &self,
        text: &str,
    ) -> Result<Vec<(String, PosTag)>, SyntacticSimilarityError> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tagged_words = Vec::new();

        for (i, word) in words.iter().enumerate() {
            let context = if i > 0 { Some(words[i - 1]) } else { None };
            let pos_tag = PosTag::tag_word(word, context);
            tagged_words.push((word.to_string(), pos_tag));
        }

        Ok(tagged_words)
    }

    /// Calculate POS sequence similarity
    fn calculate_pos_similarity(
        &self,
        tokens1: &[(String, PosTag)],
        tokens2: &[(String, PosTag)],
    ) -> Result<f64, SyntacticSimilarityError> {
        let pos_seq1: Vec<PosTag> = tokens1.iter().map(|(_, pos)| pos.clone()).collect();
        let pos_seq2: Vec<PosTag> = tokens2.iter().map(|(_, pos)| pos.clone()).collect();

        // Use longest common subsequence for POS sequences
        let lcs_length = self.longest_common_subsequence(&pos_seq1, &pos_seq2);
        let max_length = pos_seq1.len().max(pos_seq2.len()) as f64;

        if max_length == 0.0 {
            return Ok(0.0);
        }

        Ok(lcs_length as f64 / max_length)
    }

    /// Calculate structural complexity similarity
    fn calculate_structure_similarity(
        &self,
        tokens1: &[(String, PosTag)],
        tokens2: &[(String, PosTag)],
    ) -> Result<f64, SyntacticSimilarityError> {
        let complexity1 = self.calculate_structural_complexity(tokens1);
        let complexity2 = self.calculate_structural_complexity(tokens2);

        // Similarity based on complexity difference
        let max_complexity = complexity1.max(complexity2);
        if max_complexity == 0.0 {
            return Ok(1.0);
        }

        let similarity = 1.0 - (complexity1 - complexity2).abs() / max_complexity;
        Ok(similarity.clamp(0.0, 1.0))
    }

    /// Calculate structural complexity of a token sequence
    fn calculate_structural_complexity(&self, tokens: &[(String, PosTag)]) -> f64 {
        if tokens.is_empty() {
            return 0.0;
        }

        let mut complexity = 0.0;

        // Sentence length factor
        complexity += (tokens.len() as f64).log2();

        // POS diversity factor
        let unique_pos: HashSet<_> = tokens.iter().map(|(_, pos)| pos).collect();
        complexity += unique_pos.len() as f64;

        // Nested structure factor (based on punctuation and conjunctions)
        let punctuation_count = tokens
            .iter()
            .filter(|(_, pos)| *pos == PosTag::Punctuation)
            .count();
        let conjunction_count = tokens
            .iter()
            .filter(|(_, pos)| *pos == PosTag::Conjunction)
            .count();
        complexity += (punctuation_count + conjunction_count) as f64 * 0.5;

        complexity
    }

    /// Calculate grammar pattern similarity
    fn calculate_pattern_similarity(
        &mut self,
        tokens1: &[(String, PosTag)],
        tokens2: &[(String, PosTag)],
    ) -> Result<f64, SyntacticSimilarityError> {
        let patterns1 = self.extract_syntactic_patterns(tokens1)?;
        let patterns2 = self.extract_syntactic_patterns(tokens2)?;

        if patterns1.is_empty() && patterns2.is_empty() {
            return Ok(1.0);
        }

        if patterns1.is_empty() || patterns2.is_empty() {
            return Ok(0.0);
        }

        // Find common patterns
        let mut common_score = 0.0;
        let mut total_patterns = HashSet::new();

        for pattern1 in &patterns1 {
            total_patterns.insert(&pattern1.pattern_type);
            for pattern2 in &patterns2 {
                total_patterns.insert(&pattern2.pattern_type);
                if pattern1.pos_sequence == pattern2.pos_sequence {
                    common_score += (pattern1.pattern_weight + pattern2.pattern_weight) / 2.0;
                }
            }
        }

        let max_possible_score = total_patterns.len() as f64;
        if max_possible_score == 0.0 {
            return Ok(0.0);
        }

        Ok((common_score / max_possible_score).clamp(0.0, 1.0))
    }

    /// Extract syntactic patterns from tokens
    fn extract_syntactic_patterns(
        &mut self,
        tokens: &[(String, PosTag)],
    ) -> Result<Vec<SyntacticPattern>, SyntacticSimilarityError> {
        let cache_key = format!(
            "{:?}",
            tokens.iter().map(|(_, pos)| pos).collect::<Vec<_>>()
        );

        if let Some(cached_patterns) = self.pattern_cache.get(&cache_key) {
            return Ok(cached_patterns.clone());
        }

        let mut patterns = Vec::new();

        // Extract n-gram patterns of POS sequences
        for n in self.config.min_pattern_length..=(tokens.len().min(5)) {
            for window in tokens.windows(n) {
                let pos_sequence: Vec<PosTag> = window.iter().map(|(_, pos)| pos.clone()).collect();
                let pattern_type = format!("pos_ngram_{}", n);

                let complexity_score = self.calculate_pattern_complexity(&pos_sequence);
                let pattern_weight = 1.0 / (n as f64).sqrt(); // Longer patterns get less weight

                patterns.push(SyntacticPattern {
                    pattern_type,
                    pos_sequence,
                    pattern_weight,
                    complexity_score,
                });
            }
        }

        // Extract specific grammatical patterns
        self.extract_specific_patterns(tokens, &mut patterns);

        self.pattern_cache.insert(cache_key, patterns.clone());
        Ok(patterns)
    }

    /// Extract specific grammatical patterns
    fn extract_specific_patterns(
        &self,
        tokens: &[(String, PosTag)],
        patterns: &mut Vec<SyntacticPattern>,
    ) {
        // Noun phrase patterns
        self.extract_noun_phrase_patterns(tokens, patterns);

        // Verb phrase patterns
        self.extract_verb_phrase_patterns(tokens, patterns);

        // Prepositional phrase patterns
        self.extract_prepositional_phrase_patterns(tokens, patterns);

        // Question patterns
        self.extract_question_patterns(tokens, patterns);
    }

    /// Extract noun phrase patterns
    fn extract_noun_phrase_patterns(
        &self,
        tokens: &[(String, PosTag)],
        patterns: &mut Vec<SyntacticPattern>,
    ) {
        // Look for patterns like: Det + Adj + Noun, Det + Noun, Adj + Noun
        for window in tokens.windows(3) {
            match (&window[0].1, &window[1].1, &window[2].1) {
                (PosTag::Determiner, PosTag::Adjective, PosTag::Noun) => {
                    patterns.push(SyntacticPattern {
                        pattern_type: "noun_phrase_det_adj_noun".to_string(),
                        pos_sequence: vec![PosTag::Determiner, PosTag::Adjective, PosTag::Noun],
                        pattern_weight: 0.8,
                        complexity_score: 2.5,
                    });
                }
                _ => {}
            }
        }

        for window in tokens.windows(2) {
            match (&window[0].1, &window[1].1) {
                (PosTag::Determiner, PosTag::Noun) => {
                    patterns.push(SyntacticPattern {
                        pattern_type: "noun_phrase_det_noun".to_string(),
                        pos_sequence: vec![PosTag::Determiner, PosTag::Noun],
                        pattern_weight: 0.6,
                        complexity_score: 1.5,
                    });
                }
                (PosTag::Adjective, PosTag::Noun) => {
                    patterns.push(SyntacticPattern {
                        pattern_type: "noun_phrase_adj_noun".to_string(),
                        pos_sequence: vec![PosTag::Adjective, PosTag::Noun],
                        pattern_weight: 0.7,
                        complexity_score: 1.8,
                    });
                }
                _ => {}
            }
        }
    }

    /// Extract verb phrase patterns
    fn extract_verb_phrase_patterns(
        &self,
        tokens: &[(String, PosTag)],
        patterns: &mut Vec<SyntacticPattern>,
    ) {
        for window in tokens.windows(2) {
            match (&window[0].1, &window[1].1) {
                (PosTag::Verb, PosTag::Adverb) => {
                    patterns.push(SyntacticPattern {
                        pattern_type: "verb_phrase_verb_adv".to_string(),
                        pos_sequence: vec![PosTag::Verb, PosTag::Adverb],
                        pattern_weight: 0.6,
                        complexity_score: 1.4,
                    });
                }
                (PosTag::Adverb, PosTag::Verb) => {
                    patterns.push(SyntacticPattern {
                        pattern_type: "verb_phrase_adv_verb".to_string(),
                        pos_sequence: vec![PosTag::Adverb, PosTag::Verb],
                        pattern_weight: 0.7,
                        complexity_score: 1.6,
                    });
                }
                _ => {}
            }
        }
    }

    /// Extract prepositional phrase patterns
    fn extract_prepositional_phrase_patterns(
        &self,
        tokens: &[(String, PosTag)],
        patterns: &mut Vec<SyntacticPattern>,
    ) {
        for window in tokens.windows(3) {
            match (&window[0].1, &window[1].1, &window[2].1) {
                (PosTag::Preposition, PosTag::Determiner, PosTag::Noun) => {
                    patterns.push(SyntacticPattern {
                        pattern_type: "prep_phrase_prep_det_noun".to_string(),
                        pos_sequence: vec![PosTag::Preposition, PosTag::Determiner, PosTag::Noun],
                        pattern_weight: 0.8,
                        complexity_score: 2.2,
                    });
                }
                _ => {}
            }
        }

        for window in tokens.windows(2) {
            match (&window[0].1, &window[1].1) {
                (PosTag::Preposition, PosTag::Noun) => {
                    patterns.push(SyntacticPattern {
                        pattern_type: "prep_phrase_prep_noun".to_string(),
                        pos_sequence: vec![PosTag::Preposition, PosTag::Noun],
                        pattern_weight: 0.6,
                        complexity_score: 1.3,
                    });
                }
                _ => {}
            }
        }
    }

    /// Extract question patterns
    fn extract_question_patterns(
        &self,
        tokens: &[(String, PosTag)],
        patterns: &mut Vec<SyntacticPattern>,
    ) {
        // Look for question words at the beginning
        if !tokens.is_empty() {
            let first_word = &tokens[0].0.to_lowercase();
            if matches!(
                first_word.as_str(),
                "what" | "where" | "when" | "why" | "how" | "who" | "which"
            ) {
                patterns.push(SyntacticPattern {
                    pattern_type: "question_wh".to_string(),
                    pos_sequence: vec![PosTag::Pronoun], // Simplified
                    pattern_weight: 0.9,
                    complexity_score: 2.0,
                });
            }
        }

        // Look for yes/no questions (Verb + Pronoun patterns)
        for window in tokens.windows(2) {
            match (&window[0].1, &window[1].1) {
                (PosTag::Verb, PosTag::Pronoun) => {
                    if matches!(
                        window[0].0.to_lowercase().as_str(),
                        "is" | "are" | "was" | "were" | "do" | "does" | "did"
                    ) {
                        patterns.push(SyntacticPattern {
                            pattern_type: "question_yesno".to_string(),
                            pos_sequence: vec![PosTag::Verb, PosTag::Pronoun],
                            pattern_weight: 0.8,
                            complexity_score: 1.7,
                        });
                    }
                }
                _ => {}
            }
        }
    }

    /// Calculate phrase structure similarity
    fn calculate_phrase_similarity(
        &self,
        tokens1: &[(String, PosTag)],
        tokens2: &[(String, PosTag)],
    ) -> Result<f64, SyntacticSimilarityError> {
        let phrases1 = self.extract_phrase_structures(tokens1);
        let phrases2 = self.extract_phrase_structures(tokens2);

        if phrases1.is_empty() && phrases2.is_empty() {
            return Ok(1.0);
        }

        let common_phrases = phrases1
            .iter()
            .filter(|phrase1| phrases2.contains(phrase1))
            .count();

        let total_phrases = phrases1.len() + phrases2.len();
        if total_phrases == 0 {
            return Ok(0.0);
        }

        Ok((2.0 * common_phrases as f64) / total_phrases as f64)
    }

    /// Extract phrase structures from tokens
    fn extract_phrase_structures(&self, tokens: &[(String, PosTag)]) -> Vec<String> {
        let mut phrases = Vec::new();

        // Extract noun phrases
        let mut i = 0;
        while i < tokens.len() {
            let mut phrase = String::new();
            let mut phrase_length = 0;

            // Look for determiner
            if i < tokens.len() && tokens[i].1 == PosTag::Determiner {
                phrase.push_str(&format!("{} ", tokens[i].1));
                phrase_length += 1;
                i += 1;
            }

            // Look for adjectives
            while i < tokens.len() && tokens[i].1 == PosTag::Adjective {
                phrase.push_str(&format!("{} ", tokens[i].1));
                phrase_length += 1;
                i += 1;
            }

            // Look for noun
            if i < tokens.len() && tokens[i].1 == PosTag::Noun {
                phrase.push_str(&tokens[i].1.to_string());
                phrase_length += 1;
                i += 1;

                if phrase_length > 1 {
                    phrases.push(format!("NP: {}", phrase.trim()));
                }
            } else {
                i += 1;
            }
        }

        phrases
    }

    /// Calculate dependency structure similarity
    fn calculate_dependency_similarity(
        &self,
        tokens1: &[(String, PosTag)],
        tokens2: &[(String, PosTag)],
    ) -> Result<f64, SyntacticSimilarityError> {
        let deps1 = self.extract_dependency_relations(tokens1)?;
        let deps2 = self.extract_dependency_relations(tokens2)?;

        if deps1.is_empty() && deps2.is_empty() {
            return Ok(1.0);
        }

        // Compare dependency structures
        let common_relations = self.count_common_dependencies(&deps1, &deps2);
        let total_relations = deps1.len() + deps2.len();

        if total_relations == 0 {
            return Ok(0.0);
        }

        Ok((2.0 * common_relations as f64) / total_relations as f64)
    }

    /// Extract dependency relations (simplified heuristic approach)
    fn extract_dependency_relations(
        &self,
        tokens: &[(String, PosTag)],
    ) -> Result<Vec<DependencyArc>, SyntacticSimilarityError> {
        let mut dependencies = Vec::new();

        // Simple heuristic dependency parsing
        for i in 0..tokens.len() {
            match &tokens[i].1 {
                PosTag::Determiner => {
                    // Determiner modifies the next noun
                    if i + 1 < tokens.len() && tokens[i + 1].1 == PosTag::Noun {
                        dependencies.push(DependencyArc {
                            head_index: i + 1,
                            dependent_index: i,
                            relation: DependencyRelation::Determiner,
                            head_word: tokens[i + 1].0.clone(),
                            dependent_word: tokens[i].0.clone(),
                        });
                    }
                }
                PosTag::Adjective => {
                    // Adjective modifies the next noun
                    if i + 1 < tokens.len() && tokens[i + 1].1 == PosTag::Noun {
                        dependencies.push(DependencyArc {
                            head_index: i + 1,
                            dependent_index: i,
                            relation: DependencyRelation::Modifier,
                            head_word: tokens[i + 1].0.clone(),
                            dependent_word: tokens[i].0.clone(),
                        });
                    }
                }
                PosTag::Adverb => {
                    // Adverb modifies the previous verb
                    if i > 0 && tokens[i - 1].1 == PosTag::Verb {
                        dependencies.push(DependencyArc {
                            head_index: i - 1,
                            dependent_index: i,
                            relation: DependencyRelation::Modifier,
                            head_word: tokens[i - 1].0.clone(),
                            dependent_word: tokens[i].0.clone(),
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(dependencies)
    }

    /// Count common dependency relations
    fn count_common_dependencies(&self, deps1: &[DependencyArc], deps2: &[DependencyArc]) -> usize {
        let mut common_count = 0;

        for dep1 in deps1 {
            for dep2 in deps2 {
                if dep1.relation == dep2.relation
                    && dep1.head_word == dep2.head_word
                    && dep1.dependent_word == dep2.dependent_word
                {
                    common_count += 1;
                }
            }
        }

        common_count
    }

    /// Calculate tree structure similarity
    fn calculate_tree_similarity(
        &self,
        tokens1: &[(String, PosTag)],
        tokens2: &[(String, PosTag)],
    ) -> Result<f64, SyntacticSimilarityError> {
        let tree1 = self.build_syntactic_tree(tokens1)?;
        let tree2 = self.build_syntactic_tree(tokens2)?;

        // Compare tree structures using tree edit distance
        let tree_distance = self.calculate_tree_edit_distance(&tree1, &tree2)?;
        let max_nodes = tree1.len().max(tree2.len()) as f64;

        if max_nodes == 0.0 {
            return Ok(1.0);
        }

        let similarity = 1.0 - (tree_distance as f64 / max_nodes);
        Ok(similarity.clamp(0.0, 1.0))
    }

    /// Build a simplified syntactic tree
    fn build_syntactic_tree(
        &self,
        tokens: &[(String, PosTag)],
    ) -> Result<Vec<SyntacticTreeNode>, SyntacticSimilarityError> {
        let mut nodes = Vec::new();

        // Create nodes for each token
        for (i, (word, pos_tag)) in tokens.iter().enumerate() {
            nodes.push(SyntacticTreeNode {
                word: word.clone(),
                pos_tag: pos_tag.clone(),
                index: i,
                children: Vec::new(),
                parent: None,
                depth: 0,
            });
        }

        // Build tree structure (simplified heuristic approach)
        for i in 0..nodes.len() {
            match &nodes[i].pos_tag {
                PosTag::Determiner => {
                    // Attach to next noun
                    if i + 1 < nodes.len() && nodes[i + 1].pos_tag == PosTag::Noun {
                        nodes[i + 1].children.push(i);
                        nodes[i].parent = Some(i + 1);
                        nodes[i].depth = 1;
                    }
                }
                PosTag::Adjective => {
                    // Attach to next noun
                    if i + 1 < nodes.len() && nodes[i + 1].pos_tag == PosTag::Noun {
                        nodes[i + 1].children.push(i);
                        nodes[i].parent = Some(i + 1);
                        nodes[i].depth = 1;
                    }
                }
                _ => {}
            }
        }

        Ok(nodes)
    }

    /// Calculate tree edit distance (simplified version)
    fn calculate_tree_edit_distance(
        &self,
        tree1: &[SyntacticTreeNode],
        tree2: &[SyntacticTreeNode],
    ) -> Result<usize, SyntacticSimilarityError> {
        // Simplified tree edit distance based on node differences
        let mut distance = 0;

        let size_diff = (tree1.len() as i32 - tree2.len() as i32).abs() as usize;
        distance += size_diff;

        // Compare node types for common positions
        let common_size = tree1.len().min(tree2.len());
        for i in 0..common_size {
            if tree1[i].pos_tag != tree2[i].pos_tag {
                distance += 1;
            }
            if tree1[i].children.len() != tree2[i].children.len() {
                distance += 1;
            }
        }

        Ok(distance)
    }

    /// Calculate pattern complexity
    fn calculate_pattern_complexity(&self, pos_sequence: &[PosTag]) -> f64 {
        let mut complexity = pos_sequence.len() as f64;

        // Add complexity based on POS diversity
        let unique_pos: HashSet<_> = pos_sequence.iter().collect();
        complexity += unique_pos.len() as f64 * 0.5;

        // Add complexity based on specific patterns
        if pos_sequence.contains(&PosTag::Verb) && pos_sequence.contains(&PosTag::Noun) {
            complexity += 1.0;
        }
        if pos_sequence.contains(&PosTag::Adjective) {
            complexity += 0.5;
        }

        complexity
    }

    /// Calculate weighted overall similarity
    fn calculate_weighted_similarity(
        &self,
        pos_sim: f64,
        struct_sim: f64,
        pattern_sim: f64,
        phrase_sim: f64,
        dep_sim: f64,
        tree_sim: f64,
    ) -> f64 {
        self.config.pos_weight * pos_sim
            + self.config.structure_weight * struct_sim
            + self.config.pattern_weight * pattern_sim
            + self.config.phrase_weight * phrase_sim
            + self.config.dependency_weight * dep_sim
            + self.config.tree_weight * tree_sim
    }

    /// Find common syntactic patterns between texts
    fn find_common_patterns(
        &mut self,
        tokens1: &[(String, PosTag)],
        tokens2: &[(String, PosTag)],
    ) -> Result<Vec<SyntacticPattern>, SyntacticSimilarityError> {
        let patterns1 = self.extract_syntactic_patterns(tokens1)?;
        let patterns2 = self.extract_syntactic_patterns(tokens2)?;

        let mut common_patterns = Vec::new();

        for pattern1 in &patterns1 {
            for pattern2 in &patterns2 {
                if pattern1.pos_sequence == pattern2.pos_sequence
                    && pattern1.pattern_type == pattern2.pattern_type
                {
                    let avg_weight = (pattern1.pattern_weight + pattern2.pattern_weight) / 2.0;
                    let avg_complexity =
                        (pattern1.complexity_score + pattern2.complexity_score) / 2.0;

                    common_patterns.push(SyntacticPattern {
                        pattern_type: pattern1.pattern_type.clone(),
                        pos_sequence: pattern1.pos_sequence.clone(),
                        pattern_weight: avg_weight,
                        complexity_score: avg_complexity,
                    });
                }
            }
        }

        // Remove duplicates
        common_patterns.sort_by(|a, b| a.pattern_type.cmp(&b.pattern_type));
        common_patterns
            .dedup_by(|a, b| a.pattern_type == b.pattern_type && a.pos_sequence == b.pos_sequence);

        Ok(common_patterns)
    }

    /// Identify structural differences between texts
    fn identify_structural_differences(
        &self,
        tokens1: &[(String, PosTag)],
        tokens2: &[(String, PosTag)],
    ) -> Vec<String> {
        let mut differences = Vec::new();

        // Length difference
        let length_diff = (tokens1.len() as i32 - tokens2.len() as i32).abs();
        if length_diff > 5 {
            differences.push(format!(
                "Significant length difference: {} words",
                length_diff
            ));
        }

        // POS distribution differences
        let pos_dist1 = self.calculate_pos_distribution(tokens1);
        let pos_dist2 = self.calculate_pos_distribution(tokens2);

        for pos_tag in PosTag::all_tags() {
            let freq1 = pos_dist1.get(&pos_tag).unwrap_or(&0.0);
            let freq2 = pos_dist2.get(&pos_tag).unwrap_or(&0.0);
            let diff = (freq1 - freq2).abs();

            if diff > 0.2 {
                differences.push(format!("{} frequency differs by {:.2}", pos_tag, diff));
            }
        }

        // Complexity difference
        let complexity1 = self.calculate_structural_complexity(tokens1);
        let complexity2 = self.calculate_structural_complexity(tokens2);
        let complexity_diff = (complexity1 - complexity2).abs();

        if complexity_diff > 2.0 {
            differences.push(format!(
                "Structural complexity differs by {:.2}",
                complexity_diff
            ));
        }

        differences
    }

    /// Calculate POS tag distribution
    fn calculate_pos_distribution(&self, tokens: &[(String, PosTag)]) -> HashMap<PosTag, f64> {
        let mut distribution = HashMap::new();
        let total_tokens = tokens.len() as f64;

        if total_tokens == 0.0 {
            return distribution;
        }

        for (_, pos_tag) in tokens {
            *distribution.entry(pos_tag.clone()).or_insert(0.0) += 1.0;
        }

        // Normalize to frequencies
        for freq in distribution.values_mut() {
            *freq /= total_tokens;
        }

        distribution
    }

    /// Calculate quality score for the analysis
    fn calculate_quality_score(
        &self,
        tokens1: &[(String, PosTag)],
        tokens2: &[(String, PosTag)],
        common_patterns: &[SyntacticPattern],
    ) -> f64 {
        let length_factor = (tokens1.len().min(tokens2.len()) as f64 / 100.0).min(1.0);
        let pattern_factor = (common_patterns.len() as f64 / 10.0).min(1.0);
        let complexity_factor = {
            let c1 = self.calculate_structural_complexity(tokens1);
            let c2 = self.calculate_structural_complexity(tokens2);
            ((c1 + c2) / 10.0).min(1.0)
        };

        (length_factor * 0.4 + pattern_factor * 0.4 + complexity_factor * 0.2).clamp(0.0, 1.0)
    }

    /// Longest common subsequence for POS sequences
    fn longest_common_subsequence(&self, seq1: &[PosTag], seq2: &[PosTag]) -> usize {
        let m = seq1.len();
        let n = seq2.len();

        if m == 0 || n == 0 {
            return 0;
        }

        let mut dp = vec![vec![0; n + 1]; m + 1];

        for i in 1..=m {
            for j in 1..=n {
                if seq1[i - 1] == seq2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }

        dp[m][n]
    }

    /// Initialize dependency parsing rules
    fn initialize_dependency_rules(&mut self) {
        // Simple dependency rules for common patterns
        self.dependency_rules
            .insert("det_noun".to_string(), vec![DependencyRelation::Determiner]);
        self.dependency_rules
            .insert("adj_noun".to_string(), vec![DependencyRelation::Modifier]);
        self.dependency_rules
            .insert("adv_verb".to_string(), vec![DependencyRelation::Modifier]);
        self.dependency_rules
            .insert("verb_noun".to_string(), vec![DependencyRelation::Object]);
        self.dependency_rules
            .insert("noun_verb".to_string(), vec![DependencyRelation::Subject]);
    }

    /// Update configuration
    pub fn update_config(
        &mut self,
        config: SyntacticSimilarityConfig,
    ) -> Result<(), SyntacticSimilarityError> {
        config.validate_weights()?;
        self.config = config;
        Ok(())
    }

    /// Get current configuration
    pub fn get_config(&self) -> &SyntacticSimilarityConfig {
        &self.config
    }

    /// Clear pattern cache
    pub fn clear_cache(&mut self) {
        self.pattern_cache.clear();
    }
}

/// Convenience function for simple syntactic similarity analysis
pub fn analyze_syntactic_similarity(
    text1: &str,
    text2: &str,
) -> Result<SyntacticSimilarityResult, SyntacticSimilarityError> {
    let mut analyzer = SyntacticSimilarityAnalyzer::default()?;
    analyzer.analyze_similarity(text1, text2)
}

/// Convenience function for syntactic similarity with custom config
pub fn analyze_syntactic_similarity_with_config(
    text1: &str,
    text2: &str,
    config: SyntacticSimilarityConfig,
) -> Result<SyntacticSimilarityResult, SyntacticSimilarityError> {
    let mut analyzer = SyntacticSimilarityAnalyzer::new(config)?;
    analyzer.analyze_similarity(text1, text2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syntactic_analyzer_creation() {
        let analyzer = SyntacticSimilarityAnalyzer::default();
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_pos_tagging() {
        let analyzer = SyntacticSimilarityAnalyzer::default().unwrap();
        let tokens = analyzer
            .tokenize_and_tag("The quick brown fox jumps")
            .unwrap();

        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0].1, PosTag::Determiner); // "The"
        assert_eq!(tokens[1].1, PosTag::Adjective); // "quick" (heuristic may vary)
        assert_eq!(tokens[4].1, PosTag::Verb); // "jumps"
    }

    #[test]
    fn test_similar_sentences() {
        let mut analyzer = SyntacticSimilarityAnalyzer::default().unwrap();
        let text1 = "The quick brown fox jumps over the lazy dog.";
        let text2 = "The fast brown fox leaps over the sleepy dog.";

        let result = analyzer.analyze_similarity(text1, text2).unwrap();
        assert!(result.similarity_score > 0.5);
        assert!(result.pos_similarity > 0.7); // Same POS structure
    }

    #[test]
    fn test_different_structures() {
        let mut analyzer = SyntacticSimilarityAnalyzer::default().unwrap();
        let text1 = "The cat sits on the mat.";
        let text2 = "Running quickly through the forest, the deer escaped.";

        let result = analyzer.analyze_similarity(text1, text2).unwrap();
        assert!(result.similarity_score < 0.5);
        assert!(!result.structural_differences.is_empty());
    }

    #[test]
    fn test_empty_text_error() {
        let mut analyzer = SyntacticSimilarityAnalyzer::default().unwrap();
        let result = analyzer.analyze_similarity("", "Some text");
        assert!(result.is_err());
        match result.unwrap_err() {
            SyntacticSimilarityError::AnalysisFailed { .. } => {}
            _ => panic!("Expected AnalysisFailed error"),
        }
    }

    #[test]
    fn test_configuration_builder() {
        let config = SyntacticSimilarityConfig::builder()
            .approach(SyntacticApproach::PosSequence)
            .pos_weight(0.4)
            .structure_weight(0.3)
            .pattern_weight(0.2)
            .phrase_weight(0.1)
            .dependency_weight(0.0)
            .tree_weight(0.0)
            .min_pattern_length(3)
            .max_tree_depth(5)
            .detailed_patterns(false)
            .enable_dependencies(false)
            .enable_tree_analysis(false)
            .build();

        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.approach, SyntacticApproach::PosSequence);
        assert_eq!(config.pos_weight, 0.4);
        assert_eq!(config.min_pattern_length, 3);
        assert_eq!(config.detailed_patterns, false);
    }

    #[test]
    fn test_invalid_weight_configuration() {
        let config = SyntacticSimilarityConfig::builder()
            .pos_weight(0.8)
            .structure_weight(0.8) // Total > 1.0
            .pattern_weight(0.0)
            .phrase_weight(0.0)
            .dependency_weight(0.0)
            .tree_weight(0.0)
            .build();

        assert!(config.is_err());
        match config.unwrap_err() {
            SyntacticSimilarityError::InvalidConfiguration { .. } => {}
            _ => panic!("Expected InvalidConfiguration error"),
        }
    }

    #[test]
    fn test_pattern_extraction() {
        let mut analyzer = SyntacticSimilarityAnalyzer::default().unwrap();
        let tokens = analyzer.tokenize_and_tag("The quick brown fox").unwrap();
        let patterns = analyzer.extract_syntactic_patterns(&tokens).unwrap();

        assert!(!patterns.is_empty());
        // Should find various n-gram patterns
        assert!(patterns
            .iter()
            .any(|p| p.pattern_type.contains("pos_ngram")));
    }

    #[test]
    fn test_structural_complexity() {
        let analyzer = SyntacticSimilarityAnalyzer::default().unwrap();
        let simple_tokens = analyzer.tokenize_and_tag("The cat sat.").unwrap();
        let complex_tokens = analyzer
            .tokenize_and_tag(
                "The extremely intelligent cat sat very comfortably on the soft, warm cushion.",
            )
            .unwrap();

        let simple_complexity = analyzer.calculate_structural_complexity(&simple_tokens);
        let complex_complexity = analyzer.calculate_structural_complexity(&complex_tokens);

        assert!(complex_complexity > simple_complexity);
    }

    #[test]
    fn test_question_patterns() {
        let mut analyzer = SyntacticSimilarityAnalyzer::default().unwrap();
        let question_tokens = analyzer.tokenize_and_tag("What is the answer?").unwrap();
        let patterns = analyzer
            .extract_syntactic_patterns(&question_tokens)
            .unwrap();

        // Should detect question patterns
        assert!(patterns.iter().any(|p| p.pattern_type.contains("question")));
    }

    #[test]
    fn test_phrase_extraction() {
        let analyzer = SyntacticSimilarityAnalyzer::default().unwrap();
        let tokens = analyzer.tokenize_and_tag("The big red car").unwrap();
        let phrases = analyzer.extract_phrase_structures(&tokens);

        // Should extract noun phrase
        assert!(!phrases.is_empty());
        assert!(phrases.iter().any(|p| p.starts_with("NP:")));
    }

    #[test]
    fn test_convenience_functions() {
        let text1 = "The dog barks loudly.";
        let text2 = "The cat meows quietly.";

        let result = analyze_syntactic_similarity(text1, text2);
        assert!(result.is_ok());

        let config = SyntacticSimilarityConfig::builder()
            .approach(SyntacticApproach::PosSequence)
            .pos_weight(0.6)
            .structure_weight(0.4)
            .pattern_weight(0.0)
            .phrase_weight(0.0)
            .dependency_weight(0.0)
            .tree_weight(0.0)
            .build()
            .unwrap();

        let result = analyze_syntactic_similarity_with_config(text1, text2, config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_dependency_extraction() {
        let analyzer = SyntacticSimilarityAnalyzer::default().unwrap();
        let tokens = analyzer.tokenize_and_tag("The big cat").unwrap();
        let deps = analyzer.extract_dependency_relations(&tokens).unwrap();

        // Should find dependency relations
        assert!(!deps.is_empty());
        assert!(deps
            .iter()
            .any(|d| d.relation == DependencyRelation::Determiner));
    }

    #[test]
    fn test_tree_building() {
        let analyzer = SyntacticSimilarityAnalyzer::default().unwrap();
        let tokens = analyzer.tokenize_and_tag("The quick fox").unwrap();
        let tree = analyzer.build_syntactic_tree(&tokens).unwrap();

        assert_eq!(tree.len(), 3);
        // Check that some nodes have children (dependencies)
        assert!(tree.iter().any(|node| !node.children.is_empty()));
    }

    #[test]
    fn test_cache_functionality() {
        let mut analyzer = SyntacticSimilarityAnalyzer::default().unwrap();
        let tokens = analyzer.tokenize_and_tag("Test sentence").unwrap();

        // First extraction should populate cache
        let _patterns1 = analyzer.extract_syntactic_patterns(&tokens).unwrap();
        assert!(!analyzer.pattern_cache.is_empty());

        // Clear cache
        analyzer.clear_cache();
        assert!(analyzer.pattern_cache.is_empty());
    }

    #[test]
    fn test_quality_score_calculation() {
        let mut analyzer = SyntacticSimilarityAnalyzer::default().unwrap();
        let complex_text = "The sophisticated algorithm efficiently processes complex data structures with remarkable precision and accuracy.";
        let simple_text = "Cat runs.";

        let complex_result = analyzer
            .analyze_similarity(complex_text, complex_text)
            .unwrap();
        let simple_result = analyzer
            .analyze_similarity(simple_text, simple_text)
            .unwrap();

        // Complex text should have higher quality score due to more patterns
        assert!(complex_result.metadata.quality_score >= simple_result.metadata.quality_score);
    }
}
