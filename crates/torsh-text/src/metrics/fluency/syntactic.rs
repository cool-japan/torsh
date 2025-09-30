//! Syntactic Fluency Analysis
//!
//! This module provides comprehensive syntactic fluency evaluation including
//! grammaticality assessment, complexity analysis, parse tree quality,
//! and dependency coherence analysis.

use scirs2_core::ndarray::{array, Array1, Array2};
use std::collections::HashMap;

/// Configuration for syntactic analysis
#[derive(Debug, Clone)]
pub struct SyntacticConfig {
    /// Weight for grammaticality score
    pub grammaticality_weight: f64,
    /// Weight for syntactic complexity
    pub complexity_weight: f64,
    /// Weight for parse tree quality
    pub parse_quality_weight: f64,
    /// Weight for dependency coherence
    pub dependency_weight: f64,
    /// Enable sentence length penalty
    pub sentence_length_penalty: bool,
    /// Maximum acceptable sentence length
    pub max_sentence_length: usize,
    /// Minimum acceptable sentence length
    pub min_sentence_length: usize,
    /// Severity threshold for error reporting
    pub error_severity_threshold: f64,
    /// Enable advanced syntactic pattern analysis
    pub enable_advanced_patterns: bool,
}

impl Default for SyntacticConfig {
    fn default() -> Self {
        Self {
            grammaticality_weight: 0.30,
            complexity_weight: 0.25,
            parse_quality_weight: 0.25,
            dependency_weight: 0.20,
            sentence_length_penalty: true,
            max_sentence_length: 50,
            min_sentence_length: 5,
            error_severity_threshold: 0.5,
            enable_advanced_patterns: true,
        }
    }
}

/// Results of syntactic fluency analysis
#[derive(Debug, Clone)]
pub struct SyntacticFluencyResult {
    /// Overall grammaticality score (0.0 to 1.0)
    pub grammaticality_score: f64,
    /// Syntactic complexity measure
    pub syntactic_complexity: f64,
    /// Parse tree quality assessment
    pub parse_tree_quality: f64,
    /// Dependency coherence score
    pub dependency_coherence: f64,
    /// Sentence structure variety measure
    pub sentence_structure_variety: f64,
    /// Clause integration quality
    pub clause_integration: f64,
    /// Syntactic patterns detected
    pub syntactic_patterns: HashMap<String, usize>,
    /// Syntactic errors detected
    pub error_indicators: Vec<SyntacticError>,
    /// Detailed syntactic metrics
    pub detailed_metrics: DetailedSyntacticMetrics,
    /// Parse complexity analysis
    pub parse_complexity: ParseComplexityAnalysis,
}

/// Detailed syntactic metrics
#[derive(Debug, Clone)]
pub struct DetailedSyntacticMetrics {
    /// Subject-verb agreement score
    pub subject_verb_agreement: f64,
    /// Article usage accuracy
    pub article_usage_accuracy: f64,
    /// Word order correctness
    pub word_order_score: f64,
    /// Clause coordination quality
    pub clause_coordination: f64,
    /// Subordination quality
    pub subordination_quality: f64,
    /// Sentence completeness score
    pub completeness_score: f64,
    /// Structural balance measure
    pub structural_balance: f64,
}

/// Parse complexity analysis
#[derive(Debug, Clone)]
pub struct ParseComplexityAnalysis {
    /// Average dependency depth
    pub average_dependency_depth: f64,
    /// Maximum dependency depth
    pub maximum_dependency_depth: f64,
    /// Parse tree height statistics
    pub parse_tree_height: TreeHeightStats,
    /// Phrase structure complexity
    pub phrase_complexity: f64,
    /// Coordination complexity
    pub coordination_complexity: f64,
    /// Embedding depth analysis
    pub embedding_depth: EmbeddingDepthAnalysis,
}

/// Tree height statistics
#[derive(Debug, Clone)]
pub struct TreeHeightStats {
    /// Mean tree height
    pub mean_height: f64,
    /// Maximum tree height
    pub max_height: f64,
    /// Height variance across sentences
    pub height_variance: f64,
    /// Height distribution
    pub height_distribution: HashMap<usize, usize>,
}

/// Embedding depth analysis
#[derive(Debug, Clone)]
pub struct EmbeddingDepthAnalysis {
    /// Maximum embedding depth
    pub max_embedding_depth: usize,
    /// Average embedding depth
    pub average_embedding_depth: f64,
    /// Embedding distribution by type
    pub embedding_by_type: HashMap<String, usize>,
    /// Complex embedding patterns
    pub complex_patterns: Vec<String>,
}

/// Syntactic error information
#[derive(Debug, Clone)]
pub struct SyntacticError {
    /// Type of syntactic error
    pub error_type: SyntacticErrorType,
    /// Position in sentence (word index)
    pub position: usize,
    /// Error severity (0.0 to 1.0)
    pub severity: f64,
    /// Error description
    pub description: String,
    /// Correction suggestion if available
    pub correction_suggestion: Option<String>,
    /// Context information
    pub context: SyntacticErrorContext,
}

/// Context information for syntactic errors
#[derive(Debug, Clone)]
pub struct SyntacticErrorContext {
    /// Sentence index where error occurs
    pub sentence_index: usize,
    /// Surrounding words for context
    pub surrounding_context: Vec<String>,
    /// Grammatical rule violated
    pub rule_violated: String,
    /// Confidence in error detection
    pub confidence: f64,
}

/// Types of syntactic errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntacticErrorType {
    /// Subject-verb disagreement
    SubjectVerbDisagreement,
    /// Missing or incorrect article
    MissingArticle,
    /// Word order violation
    WordOrderViolation,
    /// Missing subject
    MissingSubject,
    /// Missing predicate
    MissingPredicate,
    /// Unbalanced structures (parentheses, brackets)
    UnbalancedStructures,
    /// Dangling modifier
    DanglingModifier,
    /// Fragment sentence
    SentenceFragment,
    /// Run-on sentence
    RunOnSentence,
    /// Comma splice
    CommaSplice,
    /// Agreement error (other than subject-verb)
    AgreementError,
    /// Tense inconsistency
    TenseInconsistency,
}

/// Syntactic pattern types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyntacticPattern {
    /// Simple sentence pattern
    SimpleSentence,
    /// Compound sentence with coordination
    CompoundSentence,
    /// Complex sentence with subordination
    ComplexSentence,
    /// Compound-complex sentence
    CompoundComplexSentence,
    /// Relative clause pattern
    RelativeClause,
    /// Adverbial clause pattern
    AdverbialClause,
    /// Participial phrase
    ParticipalPhrase,
    /// Prepositional phrase
    PrepositionalPhrase,
    /// Infinitive phrase
    InfinitivePhrase,
}

/// Syntactic analyzer
pub struct SyntacticAnalyzer {
    config: SyntacticConfig,
    grammar_rules: GrammarRuleSet,
    pattern_recognizers: Vec<PatternRecognizer>,
}

/// Grammar rule set for syntactic analysis
#[derive(Debug, Clone)]
pub struct GrammarRuleSet {
    /// Subject-verb agreement rules
    pub agreement_rules: HashMap<String, Vec<String>>,
    /// Article usage rules
    pub article_rules: ArticleRules,
    /// Word order patterns
    pub word_order_patterns: Vec<WordOrderPattern>,
    /// Clause structure rules
    pub clause_rules: Vec<ClauseRule>,
}

/// Article usage rules
#[derive(Debug, Clone)]
pub struct ArticleRules {
    /// Words that require "a"
    pub definite_a: HashSet<String>,
    /// Words that require "an"
    pub definite_an: HashSet<String>,
    /// Vowel sounds for article determination
    pub vowel_sounds: HashSet<char>,
    /// Exception words
    pub exceptions: HashMap<String, String>,
}

/// Word order pattern
#[derive(Debug, Clone)]
pub struct WordOrderPattern {
    /// Pattern name
    pub name: String,
    /// Expected word order
    pub order: Vec<String>,
    /// Pattern weight/importance
    pub weight: f64,
    /// Context requirements
    pub context: Vec<String>,
}

/// Clause structure rule
#[derive(Debug, Clone)]
pub struct ClauseRule {
    /// Rule name
    pub name: String,
    /// Required elements
    pub required_elements: Vec<String>,
    /// Optional elements
    pub optional_elements: Vec<String>,
    /// Order constraints
    pub order_constraints: Vec<String>,
}

/// Pattern recognizer for syntactic structures
#[derive(Debug, Clone)]
pub struct PatternRecognizer {
    /// Pattern type
    pub pattern_type: SyntacticPattern,
    /// Recognition rules
    pub rules: Vec<RecognitionRule>,
    /// Confidence threshold
    pub confidence_threshold: f64,
}

/// Recognition rule for patterns
#[derive(Debug, Clone)]
pub struct RecognitionRule {
    /// Rule description
    pub description: String,
    /// Matching criteria
    pub criteria: Vec<String>,
    /// Rule weight
    pub weight: f64,
}

use std::collections::HashSet;

impl SyntacticAnalyzer {
    /// Create a new syntactic analyzer
    pub fn new(config: SyntacticConfig) -> Self {
        let grammar_rules = Self::build_default_grammar_rules();
        let pattern_recognizers = Self::build_pattern_recognizers();

        Self {
            config,
            grammar_rules,
            pattern_recognizers,
        }
    }

    /// Create analyzer with default configuration
    pub fn with_default_config() -> Self {
        Self::new(SyntacticConfig::default())
    }

    /// Build default grammar rules
    fn build_default_grammar_rules() -> GrammarRuleSet {
        let mut agreement_rules = HashMap::new();
        agreement_rules.insert(
            "singular".to_string(),
            vec![
                "is".to_string(),
                "was".to_string(),
                "has".to_string(),
                "does".to_string(),
            ],
        );
        agreement_rules.insert(
            "plural".to_string(),
            vec![
                "are".to_string(),
                "were".to_string(),
                "have".to_string(),
                "do".to_string(),
            ],
        );

        let vowel_sounds = ['a', 'e', 'i', 'o', 'u'].iter().cloned().collect();
        let article_rules = ArticleRules {
            definite_a: HashSet::new(),
            definite_an: HashSet::new(),
            vowel_sounds,
            exceptions: HashMap::new(),
        };

        GrammarRuleSet {
            agreement_rules,
            article_rules,
            word_order_patterns: vec![],
            clause_rules: vec![],
        }
    }

    /// Build pattern recognizers
    fn build_pattern_recognizers() -> Vec<PatternRecognizer> {
        vec![
            PatternRecognizer {
                pattern_type: SyntacticPattern::SimpleSentence,
                rules: vec![],
                confidence_threshold: 0.7,
            },
            PatternRecognizer {
                pattern_type: SyntacticPattern::CompoundSentence,
                rules: vec![],
                confidence_threshold: 0.6,
            },
        ]
    }

    /// Analyze syntactic fluency of sentences
    pub fn analyze_syntactic_fluency(&self, sentences: &[String]) -> SyntacticFluencyResult {
        let mut grammaticality_scores = Vec::new();
        let mut complexity_scores = Vec::new();
        let mut parse_quality_scores = Vec::new();
        let mut syntactic_patterns = HashMap::new();
        let mut all_errors = Vec::new();
        let mut detailed_metrics_vec = Vec::new();

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let grammar_score = self.evaluate_grammaticality(sentence);
            let complexity_score = self.calculate_syntactic_complexity(sentence);
            let parse_score = self.evaluate_parse_quality(sentence);

            grammaticality_scores.push(grammar_score);
            complexity_scores.push(complexity_score);
            parse_quality_scores.push(parse_score);

            // Extract syntactic patterns
            let patterns = self.extract_syntactic_patterns(sentence);
            for (pattern, count) in patterns {
                *syntactic_patterns.entry(pattern).or_insert(0) += count;
            }

            // Detect syntactic errors
            let errors = self.detect_syntactic_errors(sentence, sent_idx);
            all_errors.extend(errors);

            // Collect detailed metrics
            let detailed = self.calculate_detailed_metrics(sentence);
            detailed_metrics_vec.push(detailed);
        }

        // Calculate aggregate scores
        let grammaticality_score = Self::calculate_mean(&grammaticality_scores);
        let syntactic_complexity = Self::calculate_mean(&complexity_scores);
        let parse_tree_quality = Self::calculate_mean(&parse_quality_scores);

        let dependency_coherence = self.calculate_dependency_coherence(sentences);
        let sentence_structure_variety = self.calculate_structure_variety(&syntactic_patterns);
        let clause_integration = self.calculate_clause_integration(sentences);

        // Aggregate detailed metrics
        let detailed_metrics = self.aggregate_detailed_metrics(&detailed_metrics_vec);

        // Calculate parse complexity analysis
        let parse_complexity = self.analyze_parse_complexity(sentences);

        SyntacticFluencyResult {
            grammaticality_score,
            syntactic_complexity,
            parse_tree_quality,
            dependency_coherence,
            sentence_structure_variety,
            clause_integration,
            syntactic_patterns,
            error_indicators: all_errors,
            detailed_metrics,
            parse_complexity,
        }
    }

    /// Evaluate grammaticality of a sentence
    pub fn evaluate_grammaticality(&self, sentence: &str) -> f64 {
        let words = self.tokenize_sentence(sentence);
        let mut score = 1.0;

        if words.is_empty() {
            return 0.0;
        }

        // Apply sentence length penalty
        if self.config.sentence_length_penalty {
            if words.len() > self.config.max_sentence_length {
                score *= 0.8;
            } else if words.len() < self.config.min_sentence_length {
                score *= 0.9;
            }
        }

        // Check subject-verb agreement
        let subject_verb_agreement = self.check_subject_verb_agreement(&words);
        score *= subject_verb_agreement;

        // Check article usage
        let article_usage = self.check_article_usage(&words);
        score *= article_usage;

        // Check word order
        let word_order = self.check_word_order(&words);
        score *= word_order;

        score.max(0.0).min(1.0)
    }

    /// Check subject-verb agreement
    fn check_subject_verb_agreement(&self, words: &[String]) -> f64 {
        let mut score = 1.0;
        let subjects = vec!["he", "she", "it"];
        let plural_subjects = vec!["they", "we", "you"];
        let singular_verbs = vec!["is", "was", "has", "does"];
        let plural_verbs = vec!["are", "were", "have", "do"];

        for i in 0..words.len().saturating_sub(1) {
            let word = &words[i];
            let next_word = &words[i + 1];

            if subjects.contains(&word.as_str()) && plural_verbs.contains(&next_word.as_str()) {
                score *= 0.7;
            } else if plural_subjects.contains(&word.as_str())
                && singular_verbs.contains(&next_word.as_str())
            {
                score *= 0.7;
            }
        }

        score
    }

    /// Check article usage
    fn check_article_usage(&self, words: &[String]) -> f64 {
        let mut score = 1.0;
        let vowels = vec!['a', 'e', 'i', 'o', 'u'];

        for i in 0..words.len().saturating_sub(1) {
            let word = &words[i];
            let next_word = &words[i + 1];

            if word == "a"
                && next_word
                    .chars()
                    .next()
                    .map_or(false, |c| vowels.contains(&c))
            {
                score *= 0.8;
            } else if word == "an"
                && next_word
                    .chars()
                    .next()
                    .map_or(true, |c| !vowels.contains(&c))
            {
                score *= 0.8;
            }
        }

        score
    }

    /// Check word order
    fn check_word_order(&self, words: &[String]) -> f64 {
        let mut score = 1.0;
        let prepositions = vec!["in", "on", "at", "by", "with", "from", "to", "for"];

        for (i, word) in words.iter().enumerate() {
            if prepositions.contains(&word.as_str()) {
                if i == 0 {
                    score *= 0.9; // Preposition at start is unusual
                }
                if i == words.len() - 1 {
                    score *= 0.8; // Preposition at end is problematic
                }
            }
        }

        score
    }

    /// Calculate syntactic complexity
    pub fn calculate_syntactic_complexity(&self, sentence: &str) -> f64 {
        let words = self.tokenize_sentence(sentence);

        let clause_count = self.count_clauses(sentence);
        let phrase_count = self.count_phrases(sentence);
        let dependency_depth = self.calculate_dependency_depth(&words);

        let base_complexity = (words.len() as f64).ln() / 5.0;
        let structure_complexity = (clause_count as f64 * 0.3) + (phrase_count as f64 * 0.2);
        let depth_complexity = dependency_depth / 10.0;

        (base_complexity + structure_complexity + depth_complexity).min(1.0)
    }

    /// Count clauses in sentence
    fn count_clauses(&self, sentence: &str) -> usize {
        let clause_indicators = vec![
            "that", "which", "who", "whom", "whose", "when", "where", "while", "because", "since",
            "although", "if",
        ];
        let mut count = 1; // At least one main clause

        for indicator in clause_indicators {
            count += sentence.to_lowercase().matches(indicator).count();
        }

        count
    }

    /// Count phrases in sentence
    fn count_phrases(&self, sentence: &str) -> usize {
        let phrase_indicators = vec![
            " of ", " in ", " on ", " at ", " by ", " with ", " from ", " to ", " for ",
        ];
        let mut count = 0;

        for indicator in phrase_indicators {
            count += sentence.to_lowercase().matches(indicator).count();
        }

        count
    }

    /// Calculate dependency depth
    pub fn calculate_dependency_depth(&self, words: &[String]) -> f64 {
        let mut max_depth = 0;
        let mut current_depth = 0;
        let open_markers = vec!["(", "[", "{"];
        let close_markers = vec![")", "]", "}"];

        for word in words {
            for marker in &open_markers {
                current_depth += word.matches(marker).count();
            }
            max_depth = max_depth.max(current_depth);

            for marker in &close_markers {
                current_depth = current_depth.saturating_sub(word.matches(marker).count());
            }
        }

        max_depth as f64
    }

    /// Evaluate parse quality
    pub fn evaluate_parse_quality(&self, sentence: &str) -> f64 {
        let words = self.tokenize_sentence(sentence);

        if words.is_empty() {
            return 0.0;
        }

        let has_subject = self.has_subject(&words);
        let has_predicate = self.has_predicate(&words);
        let balanced_structures = self.check_balanced_structures(sentence);

        let completeness = if has_subject && has_predicate {
            1.0
        } else {
            0.6
        };
        let balance = if balanced_structures { 1.0 } else { 0.8 };

        (completeness + balance) / 2.0
    }

    /// Check if sentence has subject
    fn has_subject(&self, words: &[String]) -> bool {
        let subjects = vec!["i", "you", "he", "she", "it", "we", "they", "this", "that"];
        words.iter().any(|word| subjects.contains(&word.as_str()))
    }

    /// Check if sentence has predicate
    fn has_predicate(&self, words: &[String]) -> bool {
        let verbs = vec![
            "is", "are", "was", "were", "have", "has", "had", "do", "does", "did", "will", "would",
            "can", "could", "should", "must",
        ];
        words.iter().any(|word| {
            verbs.contains(&word.as_str()) || word.ends_with("ing") || word.ends_with("ed")
        })
    }

    /// Check balanced structures
    fn check_balanced_structures(&self, sentence: &str) -> bool {
        let open_count = sentence.matches('(').count()
            + sentence.matches('[').count()
            + sentence.matches('{').count();
        let close_count = sentence.matches(')').count()
            + sentence.matches(']').count()
            + sentence.matches('}').count();
        open_count == close_count
    }

    /// Extract syntactic patterns
    pub fn extract_syntactic_patterns(&self, sentence: &str) -> HashMap<String, usize> {
        let mut patterns = HashMap::new();

        if sentence.contains(" that ") {
            *patterns
                .entry("subordinate_clause".to_string())
                .or_insert(0) += 1;
        }

        if sentence.contains(" and ") || sentence.contains(" or ") || sentence.contains(" but ") {
            *patterns.entry("coordination".to_string()).or_insert(0) += 1;
        }

        if sentence.contains(" which ") || sentence.contains(" who ") {
            *patterns.entry("relative_clause".to_string()).or_insert(0) += 1;
        }

        if sentence.contains(" because ")
            || sentence.contains(" since ")
            || sentence.contains(" if ")
        {
            *patterns.entry("adverbial_clause".to_string()).or_insert(0) += 1;
        }

        if self.config.enable_advanced_patterns {
            self.extract_advanced_patterns(sentence, &mut patterns);
        }

        patterns
    }

    /// Extract advanced syntactic patterns
    fn extract_advanced_patterns(&self, sentence: &str, patterns: &mut HashMap<String, usize>) {
        // Participial phrases
        if sentence.contains("ing ") && !sentence.starts_with("ing") {
            *patterns
                .entry("participial_phrase".to_string())
                .or_insert(0) += 1;
        }

        // Infinitive phrases
        if sentence.contains(" to ") {
            *patterns.entry("infinitive_phrase".to_string()).or_insert(0) += 1;
        }

        // Prepositional phrases (more sophisticated detection)
        let prep_count = sentence.matches(" in ").count()
            + sentence.matches(" on ").count()
            + sentence.matches(" at ").count()
            + sentence.matches(" by ").count();
        if prep_count > 0 {
            *patterns
                .entry("prepositional_phrase".to_string())
                .or_insert(0) += prep_count;
        }
    }

    /// Detect syntactic errors
    pub fn detect_syntactic_errors(&self, sentence: &str, sent_idx: usize) -> Vec<SyntacticError> {
        let mut errors = Vec::new();
        let words = self.tokenize_sentence(sentence);

        // Check article errors
        for i in 0..words.len().saturating_sub(1) {
            let word = &words[i];
            let next_word = &words[i + 1];

            if (word == "a" && next_word.starts_with(|c: char| "aeiou".contains(c)))
                || (word == "an" && !next_word.starts_with(|c: char| "aeiou".contains(c)))
            {
                let context = SyntacticErrorContext {
                    sentence_index: sent_idx,
                    surrounding_context: self.get_surrounding_context(&words, i, 2),
                    rule_violated: "Article-vowel agreement".to_string(),
                    confidence: 0.8,
                };

                errors.push(SyntacticError {
                    error_type: SyntacticErrorType::MissingArticle,
                    position: i,
                    severity: 0.3,
                    description: "Article usage error".to_string(),
                    correction_suggestion: Some(if word == "a" {
                        "an".to_string()
                    } else {
                        "a".to_string()
                    }),
                    context,
                });
            }
        }

        // Check subject-verb agreement errors
        self.detect_agreement_errors(&words, sent_idx, &mut errors);

        // Check structural balance
        if !self.check_balanced_structures(sentence) {
            let context = SyntacticErrorContext {
                sentence_index: sent_idx,
                surrounding_context: vec![sentence.to_string()],
                rule_violated: "Structural balance".to_string(),
                confidence: 0.9,
            };

            errors.push(SyntacticError {
                error_type: SyntacticErrorType::UnbalancedStructures,
                position: 0,
                severity: 0.6,
                description: "Unbalanced parentheses, brackets, or braces".to_string(),
                correction_suggestion: None,
                context,
            });
        }

        errors
    }

    /// Detect subject-verb agreement errors
    fn detect_agreement_errors(
        &self,
        words: &[String],
        sent_idx: usize,
        errors: &mut Vec<SyntacticError>,
    ) {
        let subjects = vec!["he", "she", "it"];
        let plural_verbs = vec!["are", "were", "have", "do"];

        for i in 0..words.len().saturating_sub(1) {
            let word = &words[i];
            let next_word = &words[i + 1];

            if subjects.contains(&word.as_str()) && plural_verbs.contains(&next_word.as_str()) {
                let context = SyntacticErrorContext {
                    sentence_index: sent_idx,
                    surrounding_context: self.get_surrounding_context(words, i, 3),
                    rule_violated: "Subject-verb agreement".to_string(),
                    confidence: 0.85,
                };

                errors.push(SyntacticError {
                    error_type: SyntacticErrorType::SubjectVerbDisagreement,
                    position: i,
                    severity: 0.7,
                    description: format!("Subject '{}' disagrees with verb '{}'", word, next_word),
                    correction_suggestion: Self::suggest_verb_correction(next_word),
                    context,
                });
            }
        }
    }

    /// Get surrounding context for error
    fn get_surrounding_context(
        &self,
        words: &[String],
        position: usize,
        window: usize,
    ) -> Vec<String> {
        let start = position.saturating_sub(window);
        let end = (position + window + 1).min(words.len());
        words[start..end].to_vec()
    }

    /// Suggest verb correction
    fn suggest_verb_correction(verb: &str) -> Option<String> {
        match verb {
            "are" => Some("is".to_string()),
            "were" => Some("was".to_string()),
            "have" => Some("has".to_string()),
            "do" => Some("does".to_string()),
            _ => None,
        }
    }

    /// Calculate dependency coherence
    pub fn calculate_dependency_coherence(&self, sentences: &[String]) -> f64 {
        if sentences.is_empty() {
            return 0.0;
        }

        let mut total_coherence = 0.0;
        for sentence in sentences {
            let words = self.tokenize_sentence(sentence);
            let coherence = self.calculate_sentence_dependency_coherence(&words);
            total_coherence += coherence;
        }

        total_coherence / sentences.len() as f64
    }

    /// Calculate sentence-level dependency coherence
    fn calculate_sentence_dependency_coherence(&self, words: &[String]) -> f64 {
        if words.len() < 2 {
            return 1.0;
        }

        let mut coherence_sum = 0.0;
        for i in 0..words.len() - 1 {
            let word1 = &words[i];
            let word2 = &words[i + 1];
            let semantic_coherence = self.calculate_word_semantic_coherence(word1, word2);
            let syntactic_coherence = self.calculate_word_syntactic_coherence(word1, word2);
            coherence_sum += (semantic_coherence + syntactic_coherence) / 2.0;
        }

        coherence_sum / (words.len() - 1) as f64
    }

    /// Calculate semantic coherence between words
    fn calculate_word_semantic_coherence(&self, word1: &str, word2: &str) -> f64 {
        // Simplified semantic coherence based on word types and patterns
        let function_words = vec![
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "from", "with",
        ];

        if function_words.contains(&word1) || function_words.contains(&word2) {
            0.8 // Function words generally have good coherence
        } else if word1.len() > 3 && word2.len() > 3 {
            0.6 // Content words
        } else {
            0.4 // Short words might have less semantic content
        }
    }

    /// Calculate syntactic coherence between words
    fn calculate_word_syntactic_coherence(&self, word1: &str, word2: &str) -> f64 {
        // Check for common syntactic patterns
        let verbs = vec!["is", "are", "was", "were", "have", "has", "had"];
        let nouns_indicators = vec!["the", "a", "an", "this", "that", "these", "those"];

        if nouns_indicators.contains(&word1) {
            0.9 // Determiner-noun pattern
        } else if verbs.contains(&word2) {
            0.8 // Subject-verb pattern
        } else {
            0.6 // General coherence
        }
    }

    /// Calculate structure variety
    pub fn calculate_structure_variety(&self, patterns: &HashMap<String, usize>) -> f64 {
        if patterns.is_empty() {
            return 0.0;
        }

        let total_patterns: usize = patterns.values().sum();
        if total_patterns == 0 {
            return 0.0;
        }

        self.calculate_pattern_entropy(patterns, total_patterns)
    }

    /// Calculate pattern entropy for variety measure
    fn calculate_pattern_entropy(&self, patterns: &HashMap<String, usize>, total: usize) -> f64 {
        let mut entropy = 0.0;
        for &count in patterns.values() {
            if count > 0 {
                let p = count as f64 / total as f64;
                entropy -= p * p.log2();
            }
        }
        entropy / (patterns.len() as f64).log2().max(1.0)
    }

    /// Calculate clause integration
    pub fn calculate_clause_integration(&self, sentences: &[String]) -> f64 {
        if sentences.is_empty() {
            return 0.0;
        }

        let mut total_integration = 0.0;
        for sentence in sentences {
            let integration = self.calculate_sentence_clause_integration(sentence);
            total_integration += integration;
        }

        total_integration / sentences.len() as f64
    }

    /// Calculate sentence-level clause integration
    fn calculate_sentence_clause_integration(&self, sentence: &str) -> f64 {
        let clause_count = self.count_clauses(sentence);
        let coordination_markers = sentence.matches(" and ").count()
            + sentence.matches(" but ").count()
            + sentence.matches(" or ").count();
        let subordination_markers = sentence.matches(" that ").count()
            + sentence.matches(" which ").count()
            + sentence.matches(" because ").count();

        if clause_count <= 1 {
            return 1.0; // Simple sentence has perfect integration
        }

        let integration_score =
            (coordination_markers + subordination_markers) as f64 / (clause_count - 1) as f64;
        integration_score.min(1.0)
    }

    /// Calculate detailed metrics for a sentence
    fn calculate_detailed_metrics(&self, sentence: &str) -> DetailedSyntacticMetrics {
        let words = self.tokenize_sentence(sentence);

        DetailedSyntacticMetrics {
            subject_verb_agreement: self.check_subject_verb_agreement(&words),
            article_usage_accuracy: self.check_article_usage(&words),
            word_order_score: self.check_word_order(&words),
            clause_coordination: self.calculate_clause_coordination(sentence),
            subordination_quality: self.calculate_subordination_quality(sentence),
            completeness_score: if self.has_subject(&words) && self.has_predicate(&words) {
                1.0
            } else {
                0.5
            },
            structural_balance: if self.check_balanced_structures(sentence) {
                1.0
            } else {
                0.0
            },
        }
    }

    /// Calculate clause coordination quality
    fn calculate_clause_coordination(&self, sentence: &str) -> f64 {
        let coordination_markers = sentence.matches(" and ").count()
            + sentence.matches(" but ").count()
            + sentence.matches(" or ").count();
        let clause_count = self.count_clauses(sentence);

        if clause_count <= 1 {
            return 1.0;
        }

        if coordination_markers > 0 {
            0.8 // Good coordination
        } else {
            0.4 // Poor coordination
        }
    }

    /// Calculate subordination quality
    fn calculate_subordination_quality(&self, sentence: &str) -> f64 {
        let subordination_markers = sentence.matches(" that ").count()
            + sentence.matches(" which ").count()
            + sentence.matches(" because ").count()
            + sentence.matches(" since ").count();
        let clause_count = self.count_clauses(sentence);

        if clause_count <= 1 {
            return 1.0;
        }

        if subordination_markers > 0 {
            0.9 // Excellent subordination
        } else {
            0.3 // Poor subordination
        }
    }

    /// Aggregate detailed metrics across sentences
    fn aggregate_detailed_metrics(
        &self,
        metrics_vec: &[DetailedSyntacticMetrics],
    ) -> DetailedSyntacticMetrics {
        if metrics_vec.is_empty() {
            return DetailedSyntacticMetrics {
                subject_verb_agreement: 0.0,
                article_usage_accuracy: 0.0,
                word_order_score: 0.0,
                clause_coordination: 0.0,
                subordination_quality: 0.0,
                completeness_score: 0.0,
                structural_balance: 0.0,
            };
        }

        let len = metrics_vec.len() as f64;
        DetailedSyntacticMetrics {
            subject_verb_agreement: metrics_vec
                .iter()
                .map(|m| m.subject_verb_agreement)
                .sum::<f64>()
                / len,
            article_usage_accuracy: metrics_vec
                .iter()
                .map(|m| m.article_usage_accuracy)
                .sum::<f64>()
                / len,
            word_order_score: metrics_vec.iter().map(|m| m.word_order_score).sum::<f64>() / len,
            clause_coordination: metrics_vec
                .iter()
                .map(|m| m.clause_coordination)
                .sum::<f64>()
                / len,
            subordination_quality: metrics_vec
                .iter()
                .map(|m| m.subordination_quality)
                .sum::<f64>()
                / len,
            completeness_score: metrics_vec
                .iter()
                .map(|m| m.completeness_score)
                .sum::<f64>()
                / len,
            structural_balance: metrics_vec
                .iter()
                .map(|m| m.structural_balance)
                .sum::<f64>()
                / len,
        }
    }

    /// Analyze parse complexity
    fn analyze_parse_complexity(&self, sentences: &[String]) -> ParseComplexityAnalysis {
        let mut depths = Vec::new();
        let mut height_counts = HashMap::new();

        for sentence in sentences {
            let words = self.tokenize_sentence(sentence);
            let depth = self.calculate_dependency_depth(&words);
            depths.push(depth);

            let height = self.estimate_parse_tree_height(sentence);
            *height_counts.entry(height).or_insert(0) += 1;
        }

        let average_dependency_depth = if !depths.is_empty() {
            depths.iter().sum::<f64>() / depths.len() as f64
        } else {
            0.0
        };

        let maximum_dependency_depth = depths.iter().cloned().fold(0.0, f64::max);

        let mean_height = height_counts
            .iter()
            .map(|(&height, &count)| height as f64 * count as f64)
            .sum::<f64>()
            / sentences.len().max(1) as f64;

        let max_height = height_counts.keys().cloned().max().unwrap_or(0);

        let height_variance = if sentences.len() > 1 {
            height_counts
                .iter()
                .map(|(&height, &count)| {
                    let diff = height as f64 - mean_height;
                    diff * diff * count as f64
                })
                .sum::<f64>()
                / (sentences.len() - 1) as f64
        } else {
            0.0
        };

        let tree_height = TreeHeightStats {
            mean_height,
            max_height: max_height as f64,
            height_variance,
            height_distribution: height_counts,
        };

        let phrase_complexity = self.calculate_phrase_complexity(sentences);
        let coordination_complexity = self.calculate_coordination_complexity(sentences);
        let embedding_depth = self.analyze_embedding_depth(sentences);

        ParseComplexityAnalysis {
            average_dependency_depth,
            maximum_dependency_depth,
            parse_tree_height: tree_height,
            phrase_complexity,
            coordination_complexity,
            embedding_depth,
        }
    }

    /// Estimate parse tree height for a sentence
    fn estimate_parse_tree_height(&self, sentence: &str) -> usize {
        let clause_count = self.count_clauses(sentence);
        let phrase_count = self.count_phrases(sentence);

        // Simplified estimation
        3 + clause_count + phrase_count / 2
    }

    /// Calculate phrase complexity
    fn calculate_phrase_complexity(&self, sentences: &[String]) -> f64 {
        if sentences.is_empty() {
            return 0.0;
        }

        let mut total_complexity = 0.0;
        for sentence in sentences {
            let phrase_count = self.count_phrases(sentence);
            let words = self.tokenize_sentence(sentence);
            let complexity = if !words.is_empty() {
                phrase_count as f64 / words.len() as f64
            } else {
                0.0
            };
            total_complexity += complexity;
        }

        total_complexity / sentences.len() as f64
    }

    /// Calculate coordination complexity
    fn calculate_coordination_complexity(&self, sentences: &[String]) -> f64 {
        if sentences.is_empty() {
            return 0.0;
        }

        let mut total_complexity = 0.0;
        for sentence in sentences {
            let coordination_count = sentence.matches(" and ").count()
                + sentence.matches(" or ").count()
                + sentence.matches(" but ").count();
            let clause_count = self.count_clauses(sentence);

            let complexity = if clause_count > 1 {
                coordination_count as f64 / (clause_count - 1) as f64
            } else {
                0.0
            };
            total_complexity += complexity;
        }

        total_complexity / sentences.len() as f64
    }

    /// Analyze embedding depth
    fn analyze_embedding_depth(&self, sentences: &[String]) -> EmbeddingDepthAnalysis {
        let mut max_embedding = 0;
        let mut total_embedding = 0.0;
        let mut embedding_by_type = HashMap::new();
        let mut complex_patterns = Vec::new();

        for sentence in sentences {
            let embedding = self.calculate_embedding_depth(sentence);
            max_embedding = max_embedding.max(embedding);
            total_embedding += embedding as f64;

            if embedding > 2 {
                complex_patterns.push(format!(
                    "Deep embedding (depth {}): {}",
                    embedding, sentence
                ));
            }

            // Count different embedding types
            if sentence.contains(" that ") {
                *embedding_by_type
                    .entry("complement_clause".to_string())
                    .or_insert(0) += 1;
            }
            if sentence.contains(" which ") {
                *embedding_by_type
                    .entry("relative_clause".to_string())
                    .or_insert(0) += 1;
            }
        }

        let average_embedding = if !sentences.is_empty() {
            total_embedding / sentences.len() as f64
        } else {
            0.0
        };

        EmbeddingDepthAnalysis {
            max_embedding_depth: max_embedding,
            average_embedding_depth: average_embedding,
            embedding_by_type,
            complex_patterns,
        }
    }

    /// Calculate embedding depth for a sentence
    fn calculate_embedding_depth(&self, sentence: &str) -> usize {
        let mut depth = 0;
        let mut max_depth = 0;

        // Count nested structures
        for char in sentence.chars() {
            match char {
                '(' | '[' | '{' => {
                    depth += 1;
                    max_depth = max_depth.max(depth);
                }
                ')' | ']' | '}' => {
                    depth = depth.saturating_sub(1);
                }
                _ => {}
            }
        }

        // Also consider clause embedding
        let clause_depth = sentence.matches(" that ").count() + sentence.matches(" which ").count();
        max_depth + clause_depth
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

    /// Calculate mean of a vector of values
    fn calculate_mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syntactic_analyzer_creation() {
        let config = SyntacticConfig::default();
        let analyzer = SyntacticAnalyzer::new(config);

        assert_eq!(analyzer.config.grammaticality_weight, 0.30);
    }

    #[test]
    fn test_grammaticality_evaluation() {
        let analyzer = SyntacticAnalyzer::with_default_config();

        let good_sentence = "The cat sits on the mat.";
        let bad_sentence = "Cat the on sits mat the.";

        let good_score = analyzer.evaluate_grammaticality(good_sentence);
        let bad_score = analyzer.evaluate_grammaticality(bad_sentence);

        assert!(good_score > bad_score);
        assert!(good_score >= 0.0 && good_score <= 1.0);
    }

    #[test]
    fn test_syntactic_complexity() {
        let analyzer = SyntacticAnalyzer::with_default_config();

        let simple = "The cat sits.";
        let complex = "The cat that lives in the house sits on the mat because it is comfortable.";

        let simple_complexity = analyzer.calculate_syntactic_complexity(simple);
        let complex_complexity = analyzer.calculate_syntactic_complexity(complex);

        assert!(complex_complexity > simple_complexity);
    }

    #[test]
    fn test_pattern_extraction() {
        let analyzer = SyntacticAnalyzer::with_default_config();
        let sentence = "The cat that is black and the dog which is white play together.";

        let patterns = analyzer.extract_syntactic_patterns(sentence);

        assert!(patterns.contains_key("relative_clause"));
        assert!(patterns.contains_key("coordination"));
    }

    #[test]
    fn test_error_detection() {
        let analyzer = SyntacticAnalyzer::with_default_config();
        let sentence = "He are going to the store with a apple."; // Multiple errors

        let errors = analyzer.detect_syntactic_errors(sentence, 0);

        assert!(!errors.is_empty());
        assert!(errors
            .iter()
            .any(|e| matches!(e.error_type, SyntacticErrorType::SubjectVerbDisagreement)));
        assert!(errors
            .iter()
            .any(|e| matches!(e.error_type, SyntacticErrorType::MissingArticle)));
    }

    #[test]
    fn test_fluency_analysis() {
        let analyzer = SyntacticAnalyzer::with_default_config();
        let sentences = vec![
            "The quick brown fox jumps over the lazy dog.".to_string(),
            "She reads books that are interesting.".to_string(),
        ];

        let result = analyzer.analyze_syntactic_fluency(&sentences);

        assert!(result.grammaticality_score >= 0.0);
        assert!(result.grammaticality_score <= 1.0);
        assert!(result.syntactic_complexity >= 0.0);
        assert!(!result.syntactic_patterns.is_empty());
    }

    #[test]
    fn test_dependency_coherence() {
        let analyzer = SyntacticAnalyzer::with_default_config();
        let sentences = vec!["The cat sits on the comfortable mat.".to_string()];

        let coherence = analyzer.calculate_dependency_coherence(&sentences);

        assert!(coherence >= 0.0);
        assert!(coherence <= 1.0);
    }
}
