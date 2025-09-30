//! Lexical chain building and analysis
//!
//! This module provides comprehensive lexical chain building including semantic relationship
//! detection, chain extension algorithms, and chain quality analysis for measuring
//! lexical coherence through word relationships and distributions.

use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

use super::config::{
    ChainBuildingConfig, LexicalChainType, LexicalCoherenceError, SemanticRelationship,
};
use super::results::{ChainConnectivity, LexicalChain, PositionDistribution};

/// Errors specific to chain building
#[derive(Debug, Error)]
pub enum ChainBuildingError {
    #[error("Failed to build lexical chains: {0}")]
    ChainBuildingFailed(String),
    #[error("Chain extension error: {0}")]
    ChainExtensionError(String),
    #[error("Semantic analysis error: {0}")]
    SemanticAnalysisError(String),
    #[error("Invalid chain configuration: {0}")]
    InvalidConfiguration(String),
}

/// Specialized analyzer for lexical chain building
pub struct LexicalChainBuilder {
    config: ChainBuildingConfig,
    semantic_lexicon: HashMap<String, Vec<String>>,
    morphological_rules: HashMap<String, Vec<String>>,
    word_frequency_cache: HashMap<String, f64>,
}

impl LexicalChainBuilder {
    /// Create a new lexical chain builder
    pub fn new(config: ChainBuildingConfig) -> Self {
        let semantic_lexicon = Self::build_semantic_lexicon();
        let morphological_rules = Self::build_morphological_rules();
        let word_frequency_cache = HashMap::new();

        Self {
            config,
            semantic_lexicon,
            morphological_rules,
            word_frequency_cache,
        }
    }

    /// Build lexical chains from sentences
    pub fn build_chains(
        &mut self,
        sentences: &[String],
    ) -> Result<Vec<LexicalChain>, ChainBuildingError> {
        if sentences.is_empty() {
            return Ok(Vec::new());
        }

        // Extract content words with positions
        let words_with_positions = self.extract_words_with_positions(sentences);

        // Build initial chains
        let mut chains = Vec::new();
        let mut processed_words = HashSet::new();

        for (word, positions) in &words_with_positions {
            if processed_words.contains(word) {
                continue;
            }

            if positions.len() >= self.config.min_chain_length {
                let chain = self.build_chain_starting_from(word, &words_with_positions)?;
                if chain.words.len() >= self.config.min_chain_length {
                    for (chain_word, _) in &chain.words {
                        processed_words.insert(chain_word.clone());
                    }
                    chains.push(chain);
                }
            }
        }

        // Extend chains with semantically related words
        if self.config.use_semantic_relations {
            chains = self.extend_chains_semantically(chains, &words_with_positions)?;
        }

        // Filter chains by configuration constraints
        chains = self.filter_chains(chains);

        Ok(chains)
    }

    /// Extract words with their positions from sentences
    fn extract_words_with_positions(
        &self,
        sentences: &[String],
    ) -> HashMap<String, Vec<(usize, usize)>> {
        let mut word_positions = HashMap::new();

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            for (word_idx, word) in sentence.split_whitespace().enumerate() {
                let clean_word = self.clean_word(word);
                if self.is_content_word(&clean_word) {
                    word_positions
                        .entry(clean_word)
                        .or_insert_with(Vec::new)
                        .push((sent_idx, word_idx));
                }
            }
        }

        word_positions
    }

    /// Build a chain starting from a specific word
    fn build_chain_starting_from(
        &self,
        start_word: &str,
        all_words: &HashMap<String, Vec<(usize, usize)>>,
    ) -> Result<LexicalChain, ChainBuildingError> {
        let mut chain_words = vec![(start_word.to_string(), all_words[start_word].clone())];
        let mut candidates = self.find_related_words(start_word, all_words);

        // Sort candidates by relatedness and position
        candidates.sort_by(|a, b| {
            let relatedness_cmp = b.1.partial_cmp(&a.1).unwrap();
            if relatedness_cmp == std::cmp::Ordering::Equal {
                // If relatedness is equal, prefer closer positions
                let pos_a = self.calculate_average_position(&all_words[&a.0]);
                let pos_b = self.calculate_average_position(&all_words[&b.0]);
                pos_a.partial_cmp(&pos_b).unwrap()
            } else {
                relatedness_cmp
            }
        });

        // Add related words to chain
        for (candidate_word, relatedness) in candidates {
            if chain_words.len() >= self.config.max_chain_length {
                break;
            }

            if relatedness >= self.config.similarity_threshold {
                let positions = all_words[&candidate_word].clone();
                if self.should_add_to_chain(&chain_words, &candidate_word, &positions) {
                    chain_words.push((candidate_word, positions));
                }
            }
        }

        // Determine chain properties
        let semantic_relationship = self.determine_semantic_relationship(&chain_words);
        let chain_type = self.classify_chain_type(&chain_words, &semantic_relationship);
        let coherence_score = self.calculate_chain_coherence(&chain_words);
        let (strength, avg_distance, max_distance, coverage, density) =
            self.calculate_chain_metrics(&chain_words);

        Ok(LexicalChain {
            chain_id: 0, // Will be set by caller
            words: chain_words,
            chain_type,
            semantic_relationship,
            coherence_score,
            strength,
            average_distance: avg_distance,
            max_distance,
            coverage,
            density,
        })
    }

    /// Find words related to the given word
    fn find_related_words(
        &self,
        target_word: &str,
        all_words: &HashMap<String, Vec<(usize, usize)>>,
    ) -> Vec<(String, f64)> {
        let mut related = Vec::new();

        for (word, _) in all_words {
            if word != target_word {
                let similarity = self.calculate_word_similarity(target_word, word);
                if similarity > 0.0 {
                    related.push((word.clone(), similarity));
                }
            }
        }

        related
    }

    /// Calculate similarity between two words
    fn calculate_word_similarity(&self, word1: &str, word2: &str) -> f64 {
        let mut similarity = 0.0;
        let mut weight_sum = 0.0;

        // Exact match
        if word1 == word2 {
            return 1.0;
        }

        // Character-based similarity
        let char_similarity = self.calculate_character_similarity(word1, word2);
        similarity += char_similarity * 0.3;
        weight_sum += 0.3;

        // Morphological similarity
        if self.config.use_morphological_relations {
            let morph_similarity = self.calculate_morphological_similarity(word1, word2);
            similarity += morph_similarity * 0.3;
            weight_sum += 0.3;
        }

        // Semantic similarity
        if self.config.use_semantic_relations {
            let semantic_similarity = self.calculate_semantic_similarity(word1, word2);
            similarity += semantic_similarity * 0.4;
            weight_sum += 0.4;
        }

        if weight_sum > 0.0 {
            similarity / weight_sum
        } else {
            char_similarity
        }
    }

    /// Calculate character-based similarity using Levenshtein distance
    fn calculate_character_similarity(&self, word1: &str, word2: &str) -> f64 {
        let distance = self.levenshtein_distance(word1, word2);
        let max_len = word1.len().max(word2.len());

        if max_len == 0 {
            1.0
        } else {
            1.0 - (distance as f64 / max_len as f64)
        }
    }

    /// Calculate Levenshtein distance between two strings
    fn levenshtein_distance(&self, word1: &str, word2: &str) -> usize {
        let len1 = word1.len();
        let len2 = word2.len();

        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        let chars1: Vec<char> = word1.chars().collect();
        let chars2: Vec<char> = word2.chars().collect();

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
                matrix[i][j] = std::cmp::min(
                    std::cmp::min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1),
                    matrix[i - 1][j - 1] + cost,
                );
            }
        }

        matrix[len1][len2]
    }

    /// Calculate morphological similarity
    fn calculate_morphological_similarity(&self, word1: &str, word2: &str) -> f64 {
        let stem1 = self.extract_stem(word1);
        let stem2 = self.extract_stem(word2);

        if stem1 == stem2 && stem1.len() > 2 {
            0.9 // High similarity for same stem
        } else {
            // Check morphological rules
            if let Some(variants) = self.morphological_rules.get(&stem1) {
                if variants.contains(&word2.to_string()) {
                    return 0.8;
                }
            }
            if let Some(variants) = self.morphological_rules.get(&stem2) {
                if variants.contains(&word1.to_string()) {
                    return 0.8;
                }
            }
            0.0
        }
    }

    /// Extract word stem (simplified)
    fn extract_stem(&self, word: &str) -> String {
        let word = word.to_lowercase();

        // Common English suffixes
        let suffixes = [
            ("tion", ""),
            ("sion", ""),
            ("ing", ""),
            ("ed", ""),
            ("er", ""),
            ("est", ""),
            ("ly", ""),
            ("ness", ""),
            ("ment", ""),
            ("ful", ""),
            ("less", ""),
            ("able", ""),
            ("ible", ""),
            ("ous", ""),
            ("ious", ""),
            ("al", ""),
            ("ary", ""),
            ("ory", ""),
            ("s", ""),
            ("es", ""),
            ("ies", "y"),
        ];

        for (suffix, replacement) in &suffixes {
            if word.len() > suffix.len() + 2 && word.ends_with(suffix) {
                let stem = &word[..word.len() - suffix.len()];
                return stem.to_string() + replacement;
            }
        }

        word
    }

    /// Calculate semantic similarity using built-in lexicon
    fn calculate_semantic_similarity(&self, word1: &str, word2: &str) -> f64 {
        // Check if words are synonyms
        if let Some(synonyms) = self.semantic_lexicon.get(word1) {
            if synonyms.contains(&word2.to_string()) {
                return 0.9;
            }
        }
        if let Some(synonyms) = self.semantic_lexicon.get(word2) {
            if synonyms.contains(&word1.to_string()) {
                return 0.9;
            }
        }

        // Check for shared semantic field
        let word1_fields = self.get_semantic_fields(word1);
        let word2_fields = self.get_semantic_fields(word2);

        let intersection: HashSet<_> = word1_fields.intersection(&word2_fields).collect();
        let union: HashSet<_> = word1_fields.union(&word2_fields).collect();

        if union.is_empty() {
            0.0
        } else {
            (intersection.len() as f64) / (union.len() as f64) * 0.7
        }
    }

    /// Get semantic fields for a word
    fn get_semantic_fields(&self, word: &str) -> HashSet<String> {
        let mut fields = HashSet::new();

        // This is simplified - in a real implementation, you'd use a comprehensive semantic lexicon
        let semantic_categories = [
            (
                "emotion",
                vec![
                    "happy",
                    "sad",
                    "angry",
                    "excited",
                    "pleased",
                    "disappointed",
                ],
            ),
            (
                "time",
                vec![
                    "before", "after", "during", "then", "now", "later", "early", "late",
                ],
            ),
            (
                "space",
                vec![
                    "above", "below", "near", "far", "inside", "outside", "here", "there",
                ],
            ),
            (
                "causation",
                vec![
                    "because",
                    "cause",
                    "effect",
                    "result",
                    "reason",
                    "therefore",
                ],
            ),
            (
                "evaluation",
                vec!["good", "bad", "excellent", "poor", "better", "worse"],
            ),
        ];

        for (category, words) in &semantic_categories {
            if words.contains(&word) {
                fields.insert(category.to_string());
            }
        }

        fields
    }

    /// Check if a word should be added to the current chain
    fn should_add_to_chain(
        &self,
        current_chain: &[(String, Vec<(usize, usize)>)],
        candidate_word: &str,
        candidate_positions: &[(usize, usize)],
    ) -> bool {
        // Check distance constraints
        if self.config.max_distance > 0 {
            let chain_positions: Vec<(usize, usize)> = current_chain
                .iter()
                .flat_map(|(_, positions)| positions.iter())
                .cloned()
                .collect();

            for candidate_pos in candidate_positions {
                let min_distance = chain_positions
                    .iter()
                    .map(|chain_pos| self.calculate_position_distance(candidate_pos, chain_pos))
                    .min();

                if let Some(distance) = min_distance {
                    if distance <= self.config.max_distance {
                        return true;
                    }
                }
            }
            return false;
        }

        // Check if word is already in chain
        !current_chain.iter().any(|(word, _)| word == candidate_word)
    }

    /// Calculate distance between two positions
    fn calculate_position_distance(&self, pos1: &(usize, usize), pos2: &(usize, usize)) -> usize {
        if pos1.0 == pos2.0 {
            // Same sentence
            (pos1.1 as i32 - pos2.1 as i32).abs() as usize
        } else {
            // Different sentences
            (pos1.0 as i32 - pos2.0 as i32).abs() as usize * 10
                + (pos1.1 as i32 - pos2.1 as i32).abs() as usize
        }
    }

    /// Extend chains with semantically related words
    fn extend_chains_semantically(
        &self,
        mut chains: Vec<LexicalChain>,
        all_words: &HashMap<String, Vec<(usize, usize)>>,
    ) -> Result<Vec<LexicalChain>, ChainBuildingError> {
        for chain in &mut chains {
            let mut added_words = Vec::new();

            // Look for semantically related words not already in chain
            for (word, positions) in all_words {
                if chain.words.iter().any(|(chain_word, _)| chain_word == word) {
                    continue;
                }

                // Check semantic relatedness to any word in chain
                let max_relatedness = chain
                    .words
                    .iter()
                    .map(|(chain_word, _)| self.calculate_word_similarity(chain_word, word))
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0);

                if max_relatedness >= self.config.similarity_threshold {
                    added_words.push((word.clone(), positions.clone()));
                }

                if chain.words.len() + added_words.len() >= self.config.max_chain_length {
                    break;
                }
            }

            // Add the new words to the chain
            chain.words.extend(added_words);

            // Recalculate chain metrics
            let (strength, avg_distance, max_distance, coverage, density) =
                self.calculate_chain_metrics(&chain.words);
            chain.strength = strength;
            chain.average_distance = avg_distance;
            chain.max_distance = max_distance;
            chain.coverage = coverage;
            chain.density = density;
        }

        Ok(chains)
    }

    /// Filter chains based on configuration constraints
    fn filter_chains(&self, chains: Vec<LexicalChain>) -> Vec<LexicalChain> {
        chains
            .into_iter()
            .filter(|chain| {
                chain.words.len() >= self.config.min_chain_length
                    && chain.words.len() <= self.config.max_chain_length
            })
            .collect()
    }

    /// Determine semantic relationship for a chain
    fn determine_semantic_relationship(
        &self,
        chain_words: &[(String, Vec<(usize, usize)>)],
    ) -> SemanticRelationship {
        if chain_words.len() < 2 {
            return SemanticRelationship::Morphological;
        }

        // Check for repetition
        let unique_words: HashSet<_> = chain_words.iter().map(|(word, _)| word).collect();
        if unique_words.len() == 1 {
            return SemanticRelationship::Morphological; // Exact repetition
        }

        // Check for synonymy
        for i in 0..chain_words.len() - 1 {
            for j in i + 1..chain_words.len() {
                let word1 = &chain_words[i].0;
                let word2 = &chain_words[j].0;

                if let Some(synonyms) = self.semantic_lexicon.get(word1) {
                    if synonyms.contains(word2) {
                        return SemanticRelationship::Synonymy;
                    }
                }
            }
        }

        // Check for morphological relationship
        let stems: HashSet<String> = chain_words
            .iter()
            .map(|(word, _)| self.extract_stem(word))
            .collect();
        if stems.len() == 1 {
            return SemanticRelationship::Morphological;
        }

        // Default to association
        SemanticRelationship::Association
    }

    /// Classify the type of lexical chain
    fn classify_chain_type(
        &self,
        chain_words: &[(String, Vec<(usize, usize)>)],
        relationship: &SemanticRelationship,
    ) -> LexicalChainType {
        match relationship {
            SemanticRelationship::Morphological => {
                let unique_words: HashSet<_> = chain_words.iter().map(|(word, _)| word).collect();
                if unique_words.len() == 1 {
                    LexicalChainType::Repetition
                } else {
                    LexicalChainType::Morphological
                }
            }
            SemanticRelationship::Synonymy => LexicalChainType::Synonymous,
            SemanticRelationship::Hyponymy => LexicalChainType::Hierarchical,
            SemanticRelationship::Meronymy => LexicalChainType::Meronymic,
            SemanticRelationship::Collocation => LexicalChainType::Collocational,
            _ => {
                if self.is_thematic_chain(chain_words) {
                    LexicalChainType::Thematic
                } else {
                    LexicalChainType::Mixed
                }
            }
        }
    }

    /// Check if chain represents a thematic grouping
    fn is_thematic_chain(&self, chain_words: &[(String, Vec<(usize, usize)>)]) -> bool {
        let words: Vec<&str> = chain_words.iter().map(|(word, _)| word.as_str()).collect();

        // Simple thematic detection based on semantic fields
        let shared_fields = self.find_shared_semantic_fields(&words);
        shared_fields.len() > 0
            && shared_fields
                .iter()
                .any(|field| field != "general" && field != "common")
    }

    /// Find shared semantic fields among words
    fn find_shared_semantic_fields(&self, words: &[&str]) -> Vec<String> {
        if words.is_empty() {
            return Vec::new();
        }

        let mut field_intersection = self.get_semantic_fields(words[0]);

        for word in &words[1..] {
            let word_fields = self.get_semantic_fields(word);
            field_intersection = field_intersection
                .intersection(&word_fields)
                .cloned()
                .collect();
        }

        field_intersection.into_iter().collect()
    }

    /// Calculate chain coherence score
    fn calculate_chain_coherence(&self, chain_words: &[(String, Vec<(usize, usize)>)]) -> f64 {
        if chain_words.len() < 2 {
            return 1.0;
        }

        let position_coherence = self.calculate_position_coherence_score(chain_words);
        let semantic_coherence = self.calculate_semantic_coherence_score(chain_words);

        (position_coherence + semantic_coherence) / 2.0
    }

    /// Calculate position-based coherence
    fn calculate_position_coherence_score(
        &self,
        chain_words: &[(String, Vec<(usize, usize)>)],
    ) -> f64 {
        let all_positions: Vec<(usize, usize)> = chain_words
            .iter()
            .flat_map(|(_, positions)| positions.iter())
            .cloned()
            .collect();

        if all_positions.len() < 2 {
            return 1.0;
        }

        // Calculate position variance
        let variance = self.calculate_position_variance(&all_positions);

        // Lower variance = higher coherence
        1.0 / (1.0 + variance)
    }

    /// Calculate semantic coherence score
    fn calculate_semantic_coherence_score(
        &self,
        chain_words: &[(String, Vec<(usize, usize)>)],
    ) -> f64 {
        if chain_words.len() < 2 {
            return 1.0;
        }

        let words: Vec<&str> = chain_words.iter().map(|(word, _)| word.as_str()).collect();
        let mut total_similarity = 0.0;
        let mut pair_count = 0;

        for i in 0..words.len() - 1 {
            for j in i + 1..words.len() {
                total_similarity += self.calculate_word_similarity(words[i], words[j]);
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            total_similarity / pair_count as f64
        } else {
            0.0
        }
    }

    /// Calculate position variance
    fn calculate_position_variance(&self, positions: &[(usize, usize)]) -> f64 {
        if positions.len() < 2 {
            return 0.0;
        }

        // Convert positions to single dimension (sentence * 100 + word)
        let linear_positions: Vec<f64> = positions
            .iter()
            .map(|(sent, word)| *sent as f64 * 100.0 + *word as f64)
            .collect();

        let mean = linear_positions.iter().sum::<f64>() / linear_positions.len() as f64;
        let variance = linear_positions
            .iter()
            .map(|pos| (pos - mean).powi(2))
            .sum::<f64>()
            / linear_positions.len() as f64;

        variance
    }

    /// Calculate various chain metrics
    fn calculate_chain_metrics(
        &self,
        chain_words: &[(String, Vec<(usize, usize)>)],
    ) -> (f64, f64, f64, f64, f64) {
        if chain_words.is_empty() {
            return (0.0, 0.0, 0.0, 0.0, 0.0);
        }

        // Calculate strength based on frequency and semantic coherence
        let total_occurrences: usize = chain_words
            .iter()
            .map(|(_, positions)| positions.len())
            .sum();
        let unique_words = chain_words.len();
        let strength = (total_occurrences as f64) / (unique_words as f64).max(1.0);

        // Calculate distances
        let all_positions: Vec<(usize, usize)> = chain_words
            .iter()
            .flat_map(|(_, positions)| positions.iter())
            .cloned()
            .collect();

        let (avg_distance, max_distance) = if all_positions.len() > 1 {
            let mut distances = Vec::new();
            for i in 0..all_positions.len() - 1 {
                for j in i + 1..all_positions.len() {
                    distances.push(
                        self.calculate_position_distance(&all_positions[i], &all_positions[j]),
                    );
                }
            }
            let avg = distances.iter().sum::<usize>() as f64 / distances.len() as f64;
            let max = *distances.iter().max().unwrap_or(&0) as f64;
            (avg, max)
        } else {
            (0.0, 0.0)
        };

        // Calculate coverage (sentences spanned)
        let sentence_indices: HashSet<usize> =
            all_positions.iter().map(|(sent, _)| *sent).collect();
        let coverage = sentence_indices.len() as f64;

        // Calculate density (words per sentence in coverage)
        let density = if coverage > 0.0 {
            total_occurrences as f64 / coverage
        } else {
            0.0
        };

        (strength, avg_distance, max_distance, coverage, density)
    }

    /// Calculate average position for a set of positions
    fn calculate_average_position(&self, positions: &[(usize, usize)]) -> f64 {
        if positions.is_empty() {
            return 0.0;
        }

        let sum: f64 = positions
            .iter()
            .map(|(sent, word)| *sent as f64 * 100.0 + *word as f64)
            .sum();

        sum / positions.len() as f64
    }

    /// Clean and normalize a word
    fn clean_word(&self, word: &str) -> String {
        word.trim_matches(|c: char| !c.is_alphabetic())
            .to_lowercase()
    }

    /// Check if a word is a content word
    fn is_content_word(&self, word: &str) -> bool {
        if word.len() < self.config.min_chain_length {
            return false;
        }

        let stop_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "about", "into", "through", "during", "before", "after", "above",
            "below", "over", "under", "between", "is", "am", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "can", "this", "that", "these", "those", "i", "you", "he", "she", "it",
            "we", "they",
        ];

        !stop_words.contains(&word)
    }

    /// Build semantic lexicon (simplified)
    fn build_semantic_lexicon() -> HashMap<String, Vec<String>> {
        let mut lexicon = HashMap::new();

        // Basic synonym groups
        lexicon.insert(
            "big".to_string(),
            vec![
                "large".to_string(),
                "huge".to_string(),
                "enormous".to_string(),
            ],
        );
        lexicon.insert(
            "small".to_string(),
            vec![
                "little".to_string(),
                "tiny".to_string(),
                "minute".to_string(),
            ],
        );
        lexicon.insert(
            "good".to_string(),
            vec![
                "great".to_string(),
                "excellent".to_string(),
                "wonderful".to_string(),
            ],
        );
        lexicon.insert(
            "bad".to_string(),
            vec![
                "poor".to_string(),
                "terrible".to_string(),
                "awful".to_string(),
            ],
        );
        lexicon.insert(
            "happy".to_string(),
            vec![
                "joyful".to_string(),
                "pleased".to_string(),
                "delighted".to_string(),
            ],
        );
        lexicon.insert(
            "sad".to_string(),
            vec![
                "unhappy".to_string(),
                "depressed".to_string(),
                "melancholy".to_string(),
            ],
        );

        // Add reverse mappings
        let mut reverse_mappings = HashMap::new();
        for (key, values) in &lexicon {
            for value in values {
                reverse_mappings
                    .entry(value.clone())
                    .or_insert_with(Vec::new)
                    .push(key.clone());
            }
        }
        lexicon.extend(reverse_mappings);

        lexicon
    }

    /// Build morphological rules (simplified)
    fn build_morphological_rules() -> HashMap<String, Vec<String>> {
        let mut rules = HashMap::new();

        // Common morphological patterns
        let patterns = [
            ("run", vec!["runs", "running", "ran"]),
            ("walk", vec!["walks", "walking", "walked"]),
            ("think", vec!["thinks", "thinking", "thought"]),
            ("write", vec!["writes", "writing", "wrote", "written"]),
            ("read", vec!["reads", "reading"]),
        ];

        for (stem, variants) in &patterns {
            rules.insert(
                stem.to_string(),
                variants.iter().map(|s| s.to_string()).collect(),
            );
        }

        rules
    }

    /// Analyze chain connectivity
    pub fn analyze_chain_connectivity(&self, chains: &[LexicalChain]) -> ChainConnectivity {
        if chains.is_empty() {
            return ChainConnectivity {
                average_overlap: 0.0,
                interconnectedness: 0.0,
                network_density: 0.0,
                connected_components: 0,
                largest_component_size: 0,
                average_path_length: 0.0,
                clustering_coefficient: 0.0,
            };
        }

        // Calculate chain overlaps
        let mut total_overlap = 0.0;
        let mut overlap_count = 0;

        for i in 0..chains.len() - 1 {
            for j in i + 1..chains.len() {
                let overlap = self.calculate_chain_overlap(&chains[i], &chains[j]);
                total_overlap += overlap;
                overlap_count += 1;
            }
        }

        let average_overlap = if overlap_count > 0 {
            total_overlap / overlap_count as f64
        } else {
            0.0
        };

        // Calculate network properties
        let total_words: HashSet<String> = chains
            .iter()
            .flat_map(|chain| chain.words.iter().map(|(word, _)| word.clone()))
            .collect();

        let possible_edges = chains.len() * (chains.len() - 1) / 2;
        let actual_edges = overlap_count;

        let network_density = if possible_edges > 0 {
            actual_edges as f64 / possible_edges as f64
        } else {
            0.0
        };

        ChainConnectivity {
            average_overlap,
            interconnectedness: average_overlap,
            network_density,
            connected_components: chains.len(), // Simplified
            largest_component_size: total_words.len(),
            average_path_length: 2.0, // Simplified
            clustering_coefficient: network_density,
        }
    }

    /// Calculate overlap between two chains
    fn calculate_chain_overlap(&self, chain1: &LexicalChain, chain2: &LexicalChain) -> f64 {
        let words1: HashSet<&String> = chain1.words.iter().map(|(word, _)| word).collect();
        let words2: HashSet<&String> = chain2.words.iter().map(|(word, _)| word).collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }
}
