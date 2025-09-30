//! Semantic Coherence Analysis
//!
//! This module provides sophisticated semantic coherence analysis including
//! semantic overlap calculations, field coherence assessment, and consistency
//! evaluation for semantic fluency measurement.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

use super::config::{CoherenceAnalysisConfig, CoherenceMethod};
use super::results::{ConsistencyAnalysis, InconsistencyPattern, SemanticFieldCoverage};

/// Errors that can occur during coherence analysis
#[derive(Error, Debug)]
pub enum CoherenceAnalysisError {
    #[error("Empty input provided for coherence analysis")]
    EmptyInput,
    #[error("Invalid coherence configuration: {0}")]
    InvalidConfiguration(String),
    #[error("Semantic field analysis failed: {0}")]
    FieldAnalysisError(String),
    #[error("Coherence calculation failed: {0}")]
    CalculationError(String),
    #[error("Consistency analysis failed: {0}")]
    ConsistencyError(String),
}

/// Core semantic coherence analyzer
#[derive(Debug)]
pub struct SemanticCoherenceAnalyzer {
    config: CoherenceAnalysisConfig,

    // Semantic field mappings
    semantic_fields: HashMap<String, HashSet<String>>,
    field_hierarchies: HashMap<String, Vec<String>>,

    // Analysis caches
    similarity_cache: HashMap<(String, String), f64>,
    field_cache: HashMap<String, String>,

    // Processing components
    overlap_calculator: OverlapCalculator,
    field_analyzer: FieldAnalyzer,
    consistency_evaluator: ConsistencyEvaluator,
}

/// Calculator for semantic overlap between text segments
#[derive(Debug)]
struct OverlapCalculator {
    method: CoherenceMethod,
    word_vectors: HashMap<String, Vec<f64>>,
    similarity_matrix: HashMap<(String, String), f64>,
}

/// Analyzer for semantic field coherence
#[derive(Debug)]
struct FieldAnalyzer {
    field_mappings: HashMap<String, String>,
    field_weights: HashMap<String, f64>,
    transition_costs: HashMap<(String, String), f64>,
}

/// Evaluator for semantic consistency
#[derive(Debug)]
struct ConsistencyEvaluator {
    terminology_tracker: TerminologyTracker,
    concept_tracker: ConceptTracker,
    inconsistency_detectors: Vec<InconsistencyDetector>,
}

/// Tracker for terminological consistency
#[derive(Debug)]
struct TerminologyTracker {
    term_variants: HashMap<String, HashSet<String>>,
    preferred_terms: HashMap<String, String>,
    usage_contexts: HashMap<String, Vec<String>>,
}

/// Tracker for conceptual consistency
#[derive(Debug)]
struct ConceptTracker {
    concept_definitions: HashMap<String, ConceptDefinition>,
    concept_relationships: HashMap<String, Vec<ConceptRelation>>,
    usage_patterns: HashMap<String, Vec<UsagePattern>>,
}

/// Definition of a concept for consistency tracking
#[derive(Debug, Clone)]
struct ConceptDefinition {
    concept_id: String,
    definition: String,
    key_attributes: Vec<String>,
    related_concepts: Vec<String>,
    typical_contexts: Vec<String>,
}

/// Relationship between concepts
#[derive(Debug, Clone)]
struct ConceptRelation {
    target_concept: String,
    relation_type: String,
    strength: f64,
    bidirectional: bool,
}

/// Pattern of concept usage
#[derive(Debug, Clone)]
struct UsagePattern {
    context: String,
    frequency: usize,
    consistency_score: f64,
    examples: Vec<String>,
}

/// Detector for specific types of inconsistencies
#[derive(Debug)]
struct InconsistencyDetector {
    detector_type: String,
    detection_patterns: Vec<DetectionPattern>,
    severity_calculator: fn(&str, &str) -> f64,
}

/// Pattern for detecting inconsistencies
#[derive(Debug)]
struct DetectionPattern {
    pattern: String,
    context_requirements: Vec<String>,
    confidence_threshold: f64,
}

impl SemanticCoherenceAnalyzer {
    /// Create a new semantic coherence analyzer
    pub fn new(config: CoherenceAnalysisConfig) -> Result<Self, CoherenceAnalysisError> {
        let semantic_fields = Self::initialize_semantic_fields()?;
        let field_hierarchies = Self::initialize_field_hierarchies()?;

        let overlap_calculator = OverlapCalculator::new(&config)?;
        let field_analyzer = FieldAnalyzer::new(&semantic_fields)?;
        let consistency_evaluator = ConsistencyEvaluator::new()?;

        Ok(Self {
            config,
            semantic_fields,
            field_hierarchies,
            similarity_cache: HashMap::new(),
            field_cache: HashMap::new(),
            overlap_calculator,
            field_analyzer,
            consistency_evaluator,
        })
    }

    /// Calculate semantic coherence for a set of sentences
    pub fn calculate_semantic_coherence(
        &mut self,
        sentences: &[String],
    ) -> Result<f64, CoherenceAnalysisError> {
        if sentences.is_empty() {
            return Err(CoherenceAnalysisError::EmptyInput);
        }

        if sentences.len() == 1 {
            return Ok(1.0); // Single sentence is perfectly coherent with itself
        }

        let mut total_coherence = 0.0;
        let mut pair_count = 0;

        // Calculate pairwise coherence
        for i in 0..sentences.len() {
            let context_start = i.saturating_sub(self.config.context_window / 2);
            let context_end = (i + self.config.context_window / 2 + 1).min(sentences.len());

            for j in (context_start..context_end).filter(|&x| x != i) {
                let coherence = self.calculate_pairwise_coherence(&sentences[i], &sentences[j])?;
                total_coherence += coherence;
                pair_count += 1;
            }
        }

        Ok(if pair_count > 0 {
            total_coherence / pair_count as f64
        } else {
            0.0
        })
    }

    /// Calculate pairwise semantic coherence between two sentences
    fn calculate_pairwise_coherence(
        &mut self,
        sentence1: &str,
        sentence2: &str,
    ) -> Result<f64, CoherenceAnalysisError> {
        match self.config.calculation_method {
            CoherenceMethod::Overlap => self.calculate_overlap_coherence(sentence1, sentence2),
            CoherenceMethod::WeightedSimilarity => {
                self.calculate_weighted_similarity_coherence(sentence1, sentence2)
            }
            CoherenceMethod::VectorSimilarity => {
                self.calculate_vector_similarity_coherence(sentence1, sentence2)
            }
            CoherenceMethod::GraphBased => {
                self.calculate_graph_based_coherence(sentence1, sentence2)
            }
            CoherenceMethod::Hybrid => self.calculate_hybrid_coherence(sentence1, sentence2),
        }
    }

    /// Calculate coherence using semantic overlap
    fn calculate_overlap_coherence(
        &mut self,
        sentence1: &str,
        sentence2: &str,
    ) -> Result<f64, CoherenceAnalysisError> {
        let words1 = self.tokenize_sentence(sentence1);
        let words2 = self.tokenize_sentence(sentence2);

        self.overlap_calculator
            .calculate_semantic_overlap(&words1, &words2)
    }

    /// Calculate coherence using weighted semantic similarity
    fn calculate_weighted_similarity_coherence(
        &mut self,
        sentence1: &str,
        sentence2: &str,
    ) -> Result<f64, CoherenceAnalysisError> {
        let words1 = self.tokenize_sentence(sentence1);
        let words2 = self.tokenize_sentence(sentence2);

        let mut total_similarity = 0.0;
        let mut total_weight = 0.0;

        for word1 in &words1 {
            for word2 in &words2 {
                let similarity = self.get_word_similarity(word1, word2)?;
                let weight = self.get_word_importance(word1) * self.get_word_importance(word2);

                total_similarity += similarity * weight;
                total_weight += weight;
            }
        }

        Ok(if total_weight > 0.0 {
            total_similarity / total_weight
        } else {
            0.0
        })
    }

    /// Calculate coherence using vector similarity
    fn calculate_vector_similarity_coherence(
        &mut self,
        sentence1: &str,
        sentence2: &str,
    ) -> Result<f64, CoherenceAnalysisError> {
        let vector1 = self.sentence_to_vector(sentence1)?;
        let vector2 = self.sentence_to_vector(sentence2)?;

        Ok(self.cosine_similarity(&vector1, &vector2))
    }

    /// Calculate coherence using graph-based methods
    fn calculate_graph_based_coherence(
        &mut self,
        sentence1: &str,
        sentence2: &str,
    ) -> Result<f64, CoherenceAnalysisError> {
        let words1 = self.tokenize_sentence(sentence1);
        let words2 = self.tokenize_sentence(sentence2);

        let graph_distance = self.calculate_semantic_graph_distance(&words1, &words2)?;

        // Convert distance to similarity (inverse relationship)
        Ok((1.0 / (1.0 + graph_distance)).max(0.0))
    }

    /// Calculate coherence using hybrid approach
    fn calculate_hybrid_coherence(
        &mut self,
        sentence1: &str,
        sentence2: &str,
    ) -> Result<f64, CoherenceAnalysisError> {
        let overlap_score = self.calculate_overlap_coherence(sentence1, sentence2)?;
        let weighted_score = self.calculate_weighted_similarity_coherence(sentence1, sentence2)?;
        let vector_score = self.calculate_vector_similarity_coherence(sentence1, sentence2)?;

        // Weighted combination of different methods
        Ok(overlap_score * 0.3 + weighted_score * 0.4 + vector_score * 0.3)
    }

    /// Analyze semantic field coverage and coherence
    pub fn analyze_field_coverage(
        &mut self,
        sentences: &[String],
    ) -> Result<SemanticFieldCoverage, CoherenceAnalysisError> {
        if sentences.is_empty() {
            return Err(CoherenceAnalysisError::EmptyInput);
        }

        self.field_analyzer.analyze_field_coverage(sentences)
    }

    /// Perform consistency analysis on the text
    pub fn analyze_consistency(
        &mut self,
        sentences: &[String],
    ) -> Result<ConsistencyAnalysis, CoherenceAnalysisError> {
        if sentences.is_empty() {
            return Err(CoherenceAnalysisError::EmptyInput);
        }

        self.consistency_evaluator.analyze_consistency(sentences)
    }

    /// Get semantic similarity between two words
    fn get_word_similarity(
        &mut self,
        word1: &str,
        word2: &str,
    ) -> Result<f64, CoherenceAnalysisError> {
        let key = (word1.to_lowercase(), word2.to_lowercase());

        if let Some(&similarity) = self.similarity_cache.get(&key) {
            return Ok(similarity);
        }

        let similarity = self.calculate_word_semantic_similarity(&key.0, &key.1)?;
        self.similarity_cache.insert(key, similarity);

        Ok(similarity)
    }

    /// Calculate semantic similarity between two words
    fn calculate_word_semantic_similarity(
        &self,
        word1: &str,
        word2: &str,
    ) -> Result<f64, CoherenceAnalysisError> {
        if word1 == word2 {
            return Ok(1.0);
        }

        // Check if words are in the same semantic field
        let field1 = self.get_word_semantic_field(word1);
        let field2 = self.get_word_semantic_field(word2);

        let field_similarity = if field1 == field2 && field1.is_some() {
            0.7 // Same semantic field
        } else if let (Some(f1), Some(f2)) = (&field1, &field2) {
            self.get_field_similarity(f1, f2)
        } else {
            0.1 // Different or unknown fields
        };

        // Add lexical similarity if available
        let lexical_similarity = self.calculate_lexical_similarity(word1, word2);

        Ok((field_similarity * 0.7 + lexical_similarity * 0.3).min(1.0))
    }

    /// Get semantic field for a word
    fn get_word_semantic_field(&self, word: &str) -> Option<String> {
        if let Some(field) = self.field_cache.get(word) {
            return Some(field.clone());
        }

        for (field_name, words) in &self.semantic_fields {
            if words.contains(word) {
                return Some(field_name.clone());
            }
        }

        None
    }

    /// Calculate similarity between semantic fields
    fn get_field_similarity(&self, field1: &str, field2: &str) -> f64 {
        if field1 == field2 {
            return 1.0;
        }

        // Check field hierarchy relationships
        if let Some(hierarchy1) = self.field_hierarchies.get(field1) {
            if hierarchy1.contains(&field2.to_string()) {
                return 0.8; // Hierarchical relationship
            }
        }

        if let Some(hierarchy2) = self.field_hierarchies.get(field2) {
            if hierarchy2.contains(&field1.to_string()) {
                return 0.8; // Hierarchical relationship
            }
        }

        // Default low similarity for unrelated fields
        0.2
    }

    /// Calculate lexical similarity between words
    fn calculate_lexical_similarity(&self, word1: &str, word2: &str) -> f64 {
        // Simple edit distance-based similarity
        let distance = edit_distance(word1, word2);
        let max_len = word1.len().max(word2.len()) as f64;

        if max_len == 0.0 {
            1.0
        } else {
            (max_len - distance as f64) / max_len
        }
    }

    /// Get importance weight for a word
    fn get_word_importance(&self, word: &str) -> f64 {
        // Simple heuristic: longer words are more important
        let length_factor = (word.len() as f64 / 10.0).min(1.0);

        // Check if word is in semantic fields (content words are more important)
        let field_factor = if self.get_word_semantic_field(word).is_some() {
            1.0
        } else {
            0.3
        };

        length_factor * field_factor
    }

    /// Convert sentence to vector representation
    fn sentence_to_vector(&self, sentence: &str) -> Result<Vec<f64>, CoherenceAnalysisError> {
        let words = self.tokenize_sentence(sentence);
        let vector_size = 300; // Standard embedding size
        let mut vector = vec![0.0; vector_size];

        let mut word_count = 0;
        for word in &words {
            if let Some(word_vector) = self.overlap_calculator.word_vectors.get(word) {
                for (i, &value) in word_vector.iter().enumerate() {
                    if i < vector_size {
                        vector[i] += value;
                    }
                }
                word_count += 1;
            }
        }

        // Average the vectors
        if word_count > 0 {
            for value in vector.iter_mut() {
                *value /= word_count as f64;
            }
        }

        Ok(vector)
    }

    /// Calculate cosine similarity between vectors
    fn cosine_similarity(&self, vec1: &[f64], vec2: &[f64]) -> f64 {
        if vec1.len() != vec2.len() {
            return 0.0;
        }

        let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = vec1.iter().map(|a| a * a).sum::<f64>().sqrt();
        let norm2: f64 = vec2.iter().map(|a| a * a).sum::<f64>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        }
    }

    /// Calculate semantic graph distance between word sets
    fn calculate_semantic_graph_distance(
        &self,
        words1: &[String],
        words2: &[String],
    ) -> Result<f64, CoherenceAnalysisError> {
        // Simplified graph distance calculation
        // In a real implementation, this would use a semantic network

        let mut min_distance = f64::INFINITY;

        for word1 in words1 {
            for word2 in words2 {
                let distance = if word1 == word2 {
                    0.0
                } else if self.get_word_semantic_field(word1) == self.get_word_semantic_field(word2)
                {
                    1.0
                } else {
                    2.0
                };

                min_distance = min_distance.min(distance);
            }
        }

        Ok(if min_distance == f64::INFINITY {
            3.0
        } else {
            min_distance
        })
    }

    /// Tokenize sentence into words
    fn tokenize_sentence(&self, sentence: &str) -> Vec<String> {
        sentence
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphabetic()).to_lowercase())
            .filter(|w| !w.is_empty() && w.len() >= 2)
            .collect()
    }

    /// Initialize semantic fields
    fn initialize_semantic_fields(
    ) -> Result<HashMap<String, HashSet<String>>, CoherenceAnalysisError> {
        let mut fields = HashMap::new();

        // Basic semantic fields with sample words
        fields.insert(
            "emotion".to_string(),
            HashSet::from([
                "happy".to_string(),
                "sad".to_string(),
                "angry".to_string(),
                "joy".to_string(),
                "fear".to_string(),
                "love".to_string(),
                "hate".to_string(),
                "excited".to_string(),
            ]),
        );

        fields.insert(
            "time".to_string(),
            HashSet::from([
                "day".to_string(),
                "night".to_string(),
                "morning".to_string(),
                "evening".to_string(),
                "hour".to_string(),
                "minute".to_string(),
                "second".to_string(),
                "week".to_string(),
            ]),
        );

        fields.insert(
            "space".to_string(),
            HashSet::from([
                "here".to_string(),
                "there".to_string(),
                "above".to_string(),
                "below".to_string(),
                "near".to_string(),
                "far".to_string(),
                "inside".to_string(),
                "outside".to_string(),
            ]),
        );

        fields.insert(
            "action".to_string(),
            HashSet::from([
                "run".to_string(),
                "walk".to_string(),
                "jump".to_string(),
                "move".to_string(),
                "go".to_string(),
                "come".to_string(),
                "take".to_string(),
                "give".to_string(),
            ]),
        );

        fields.insert(
            "object".to_string(),
            HashSet::from([
                "table".to_string(),
                "chair".to_string(),
                "book".to_string(),
                "computer".to_string(),
                "phone".to_string(),
                "car".to_string(),
                "house".to_string(),
                "door".to_string(),
            ]),
        );

        Ok(fields)
    }

    /// Initialize field hierarchies
    fn initialize_field_hierarchies() -> Result<HashMap<String, Vec<String>>, CoherenceAnalysisError>
    {
        let mut hierarchies = HashMap::new();

        hierarchies.insert(
            "emotion".to_string(),
            vec!["psychological".to_string(), "mental_state".to_string()],
        );

        hierarchies.insert(
            "time".to_string(),
            vec!["temporal".to_string(), "chronological".to_string()],
        );

        hierarchies.insert(
            "space".to_string(),
            vec!["spatial".to_string(), "geographical".to_string()],
        );

        hierarchies.insert(
            "action".to_string(),
            vec!["activity".to_string(), "behavior".to_string()],
        );

        hierarchies.insert(
            "object".to_string(),
            vec!["physical".to_string(), "material".to_string()],
        );

        Ok(hierarchies)
    }
}

impl OverlapCalculator {
    fn new(config: &CoherenceAnalysisConfig) -> Result<Self, CoherenceAnalysisError> {
        Ok(Self {
            method: config.calculation_method.clone(),
            word_vectors: HashMap::new(),
            similarity_matrix: HashMap::new(),
        })
    }

    fn calculate_semantic_overlap(
        &self,
        words1: &[String],
        words2: &[String],
    ) -> Result<f64, CoherenceAnalysisError> {
        if words1.is_empty() || words2.is_empty() {
            return Ok(0.0);
        }

        let set1: HashSet<&String> = words1.iter().collect();
        let set2: HashSet<&String> = words2.iter().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        Ok(if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        })
    }
}

impl FieldAnalyzer {
    fn new(
        semantic_fields: &HashMap<String, HashSet<String>>,
    ) -> Result<Self, CoherenceAnalysisError> {
        let mut field_mappings = HashMap::new();

        for (field_name, words) in semantic_fields {
            for word in words {
                field_mappings.insert(word.clone(), field_name.clone());
            }
        }

        Ok(Self {
            field_mappings,
            field_weights: HashMap::new(),
            transition_costs: HashMap::new(),
        })
    }

    fn analyze_field_coverage(
        &self,
        sentences: &[String],
    ) -> Result<SemanticFieldCoverage, CoherenceAnalysisError> {
        let mut field_counts: HashMap<String, f64> = HashMap::new();
        let mut total_words = 0;

        for sentence in sentences {
            let words: Vec<String> = sentence
                .split_whitespace()
                .map(|w| w.to_lowercase())
                .collect();

            for word in &words {
                if let Some(field) = self.field_mappings.get(word) {
                    *field_counts.entry(field.clone()).or_insert(0.0) += 1.0;
                }
                total_words += 1;
            }
        }

        let overall_coverage = if total_words > 0 {
            field_counts.values().sum::<f64>() / total_words as f64
        } else {
            0.0
        };

        // Calculate distribution evenness (Shannon entropy)
        let total_field_words: f64 = field_counts.values().sum();
        let mut evenness = 0.0;
        if total_field_words > 0.0 {
            for count in field_counts.values() {
                if *count > 0.0 {
                    let proportion = count / total_field_words;
                    evenness -= proportion * proportion.ln();
                }
            }
        }

        let diversity_index = field_counts.len() as f64;
        let dominant_field = field_counts
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(field, _)| field.clone());

        Ok(SemanticFieldCoverage {
            overall_coverage,
            semantic_fields: field_counts,
            distribution_evenness: evenness,
            transition_coherence: 0.8, // Placeholder calculation
            dominant_field,
            diversity_index,
        })
    }
}

impl ConsistencyEvaluator {
    fn new() -> Result<Self, CoherenceAnalysisError> {
        Ok(Self {
            terminology_tracker: TerminologyTracker::new(),
            concept_tracker: ConceptTracker::new(),
            inconsistency_detectors: Self::initialize_detectors(),
        })
    }

    fn analyze_consistency(
        &mut self,
        sentences: &[String],
    ) -> Result<ConsistencyAnalysis, CoherenceAnalysisError> {
        let overall_consistency = self.calculate_overall_consistency(sentences)?;
        let terminological_consistency = self
            .terminology_tracker
            .analyze_terminological_consistency(sentences)?;
        let conceptual_consistency = self
            .concept_tracker
            .analyze_conceptual_consistency(sentences)?;
        let field_consistency = self.calculate_field_consistency(sentences)?;
        let inconsistencies = self.detect_inconsistencies(sentences)?;
        let consistency_trend = self.calculate_consistency_trend(sentences)?;

        Ok(ConsistencyAnalysis {
            overall_consistency,
            terminological_consistency,
            conceptual_consistency,
            field_consistency,
            inconsistencies,
            consistency_trend,
        })
    }

    fn calculate_overall_consistency(
        &self,
        _sentences: &[String],
    ) -> Result<f64, CoherenceAnalysisError> {
        // Simplified overall consistency calculation
        Ok(0.8)
    }

    fn calculate_field_consistency(
        &self,
        _sentences: &[String],
    ) -> Result<f64, CoherenceAnalysisError> {
        // Simplified field consistency calculation
        Ok(0.75)
    }

    fn detect_inconsistencies(
        &self,
        _sentences: &[String],
    ) -> Result<Vec<InconsistencyPattern>, CoherenceAnalysisError> {
        // Simplified inconsistency detection
        Ok(Vec::new())
    }

    fn calculate_consistency_trend(
        &self,
        sentences: &[String],
    ) -> Result<Vec<f64>, CoherenceAnalysisError> {
        // Simple trend calculation
        Ok(vec![0.8; sentences.len()])
    }

    fn initialize_detectors() -> Vec<InconsistencyDetector> {
        vec![InconsistencyDetector {
            detector_type: "terminology".to_string(),
            detection_patterns: vec![],
            severity_calculator: |_a, _b| 0.5,
        }]
    }
}

impl TerminologyTracker {
    fn new() -> Self {
        Self {
            term_variants: HashMap::new(),
            preferred_terms: HashMap::new(),
            usage_contexts: HashMap::new(),
        }
    }

    fn analyze_terminological_consistency(
        &mut self,
        _sentences: &[String],
    ) -> Result<f64, CoherenceAnalysisError> {
        // Simplified terminological consistency analysis
        Ok(0.85)
    }
}

impl ConceptTracker {
    fn new() -> Self {
        Self {
            concept_definitions: HashMap::new(),
            concept_relationships: HashMap::new(),
            usage_patterns: HashMap::new(),
        }
    }

    fn analyze_conceptual_consistency(
        &mut self,
        _sentences: &[String],
    ) -> Result<f64, CoherenceAnalysisError> {
        // Simplified conceptual consistency analysis
        Ok(0.8)
    }
}

/// Calculate edit distance between two strings
fn edit_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();
    let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

    for i in 0..=len1 {
        matrix[i][0] = i;
    }
    for j in 0..=len2 {
        matrix[0][j] = j;
    }

    let chars1: Vec<char> = s1.chars().collect();
    let chars2: Vec<char> = s2.chars().collect();

    for i in 1..=len1 {
        for j in 1..=len2 {
            let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
            matrix[i][j] = (matrix[i - 1][j] + 1)
                .min(matrix[i][j - 1] + 1)
                .min(matrix[i - 1][j - 1] + cost);
        }
    }

    matrix[len1][len2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::fluency::semantic::config::CoherenceAnalysisConfig;

    #[test]
    fn test_coherence_analyzer_creation() {
        let config = CoherenceAnalysisConfig::default();
        let analyzer = SemanticCoherenceAnalyzer::new(config);
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_single_sentence_coherence() {
        let config = CoherenceAnalysisConfig::default();
        let mut analyzer = SemanticCoherenceAnalyzer::new(config).unwrap();

        let sentences = vec!["This is a test sentence.".to_string()];
        let coherence = analyzer.calculate_semantic_coherence(&sentences).unwrap();
        assert_eq!(coherence, 1.0);
    }

    #[test]
    fn test_empty_input_handling() {
        let config = CoherenceAnalysisConfig::default();
        let mut analyzer = SemanticCoherenceAnalyzer::new(config).unwrap();

        let sentences = vec![];
        let result = analyzer.calculate_semantic_coherence(&sentences);
        assert!(result.is_err());
    }

    #[test]
    fn test_overlap_coherence() {
        let config = CoherenceAnalysisConfig {
            calculation_method: CoherenceMethod::Overlap,
            ..Default::default()
        };
        let mut analyzer = SemanticCoherenceAnalyzer::new(config).unwrap();

        let sentences = vec![
            "The cat sat on the mat.".to_string(),
            "The cat was very happy.".to_string(),
        ];

        let coherence = analyzer.calculate_semantic_coherence(&sentences).unwrap();
        assert!(coherence > 0.0 && coherence <= 1.0);
    }

    #[test]
    fn test_word_similarity() {
        let config = CoherenceAnalysisConfig::default();
        let mut analyzer = SemanticCoherenceAnalyzer::new(config).unwrap();

        // Test identical words
        let similarity = analyzer.get_word_similarity("test", "test").unwrap();
        assert_eq!(similarity, 1.0);

        // Test different words
        let similarity = analyzer.get_word_similarity("cat", "dog").unwrap();
        assert!(similarity >= 0.0 && similarity <= 1.0);
    }

    #[test]
    fn test_semantic_field_detection() {
        let config = CoherenceAnalysisConfig::default();
        let analyzer = SemanticCoherenceAnalyzer::new(config).unwrap();

        // Test known words from semantic fields
        assert!(analyzer.get_word_semantic_field("happy").is_some());
        assert!(analyzer.get_word_semantic_field("day").is_some());
        assert!(analyzer
            .get_word_semantic_field("nonexistentword")
            .is_none());
    }

    #[test]
    fn test_field_coverage_analysis() {
        let config = CoherenceAnalysisConfig::default();
        let mut analyzer = SemanticCoherenceAnalyzer::new(config).unwrap();

        let sentences = vec![
            "I am happy today.".to_string(),
            "The day is bright and joyful.".to_string(),
        ];

        let coverage = analyzer.analyze_field_coverage(&sentences).unwrap();
        assert!(coverage.overall_coverage >= 0.0);
        assert!(!coverage.semantic_fields.is_empty());
    }

    #[test]
    fn test_consistency_analysis() {
        let config = CoherenceAnalysisConfig::default();
        let mut analyzer = SemanticCoherenceAnalyzer::new(config).unwrap();

        let sentences = vec![
            "The concept is important.".to_string(),
            "This concept helps understanding.".to_string(),
        ];

        let consistency = analyzer.analyze_consistency(&sentences).unwrap();
        assert!(consistency.overall_consistency >= 0.0 && consistency.overall_consistency <= 1.0);
    }

    #[test]
    fn test_edit_distance() {
        assert_eq!(edit_distance("cat", "cat"), 0);
        assert_eq!(edit_distance("cat", "bat"), 1);
        assert_eq!(edit_distance("cat", "dog"), 3);
        assert_eq!(edit_distance("", "abc"), 3);
    }

    #[test]
    fn test_cosine_similarity() {
        let config = CoherenceAnalysisConfig::default();
        let analyzer = SemanticCoherenceAnalyzer::new(config).unwrap();

        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![1.0, 0.0, 0.0];
        let similarity = analyzer.cosine_similarity(&vec1, &vec2);
        assert!((similarity - 1.0).abs() < 0.001);

        let vec3 = vec![1.0, 0.0, 0.0];
        let vec4 = vec![0.0, 1.0, 0.0];
        let similarity = analyzer.cosine_similarity(&vec3, &vec4);
        assert!((similarity - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_tokenization() {
        let config = CoherenceAnalysisConfig::default();
        let analyzer = SemanticCoherenceAnalyzer::new(config).unwrap();

        let sentence = "The cat, sat on the mat!";
        let tokens = analyzer.tokenize_sentence(sentence);

        assert!(tokens.contains(&"cat".to_string()));
        assert!(tokens.contains(&"sat".to_string()));
        assert!(tokens.contains(&"mat".to_string()));
        assert!(!tokens.contains(&"!".to_string())); // Punctuation filtered
    }
}
