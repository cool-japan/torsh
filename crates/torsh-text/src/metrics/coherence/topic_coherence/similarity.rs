//! Similarity calculation module for topic coherence analysis
//!
//! This module provides various similarity calculation algorithms that are used
//! across different topic extraction and analysis components. It centralizes
//! all similarity logic in a clean, reusable interface.

use crate::metrics::coherence::topic_coherence::config::SimilarityConfig;
use std::collections::{HashMap, HashSet};

/// Comprehensive similarity calculator with multiple algorithms
pub struct SimilarityCalculator {
    config: SimilarityConfig,
    semantic_lexicon: HashMap<String, Vec<String>>,
    cooccurrence_cache: HashMap<(String, String), f64>,
}

impl SimilarityCalculator {
    /// Create a new similarity calculator with configuration
    pub fn new(config: SimilarityConfig) -> Self {
        Self {
            config,
            semantic_lexicon: Self::build_semantic_lexicon(),
            cooccurrence_cache: HashMap::new(),
        }
    }

    /// Calculate combined similarity using all enabled methods
    pub fn calculate_similarity(&self, word1: &str, word2: &str) -> f64 {
        let mut total_similarity = 0.0;
        let mut total_weight = 0.0;

        if self.config.enable_character_similarity {
            let char_sim = self.character_similarity(word1, word2);
            total_similarity += char_sim * self.config.character_similarity_weight;
            total_weight += self.config.character_similarity_weight;
        }

        if self.config.enable_semantic_similarity {
            let semantic_sim = self.semantic_similarity(word1, word2);
            total_similarity += semantic_sim * self.config.semantic_similarity_weight;
            total_weight += self.config.semantic_similarity_weight;
        }

        if self.config.enable_cooccurrence_similarity {
            let cooc_sim = self.cooccurrence_similarity(word1, word2);
            total_similarity += cooc_sim * self.config.cooccurrence_similarity_weight;
            total_weight += self.config.cooccurrence_similarity_weight;
        }

        if total_weight > 0.0 {
            total_similarity / total_weight
        } else {
            0.0
        }
    }

    /// Calculate character-level similarity between two words
    pub fn character_similarity(&self, word1: &str, word2: &str) -> f64 {
        if word1.is_empty() || word2.is_empty() {
            return 0.0;
        }

        if word1 == word2 {
            return 1.0;
        }

        // Levenshtein distance based similarity
        let distance = self.levenshtein_distance(word1, word2);
        let max_len = word1.chars().count().max(word2.chars().count());

        if max_len == 0 {
            1.0
        } else {
            1.0 - (distance as f64 / max_len as f64)
        }
    }

    /// Calculate semantic similarity based on word relationships
    pub fn semantic_similarity(&self, word1: &str, word2: &str) -> f64 {
        if word1 == word2 {
            return 1.0;
        }

        // Check if words are in the same semantic field
        let word1_lower = word1.to_lowercase();
        let word2_lower = word2.to_lowercase();

        // Direct semantic relationship check
        if let Some(related_words) = self.semantic_lexicon.get(&word1_lower) {
            if related_words.contains(&word2_lower) {
                return 0.8; // High similarity for direct semantic relationship
            }
        }

        // Reverse relationship check
        if let Some(related_words) = self.semantic_lexicon.get(&word2_lower) {
            if related_words.contains(&word1_lower) {
                return 0.8;
            }
        }

        // Shared semantic field check
        let mut shared_fields = 0;
        let mut total_fields = 0;

        for (field_word, related_words) in &self.semantic_lexicon {
            if related_words.contains(&word1_lower) || field_word == &word1_lower {
                total_fields += 1;
                if related_words.contains(&word2_lower) || field_word == &word2_lower {
                    shared_fields += 1;
                }
            }
        }

        if total_fields > 0 {
            (shared_fields as f64 / total_fields as f64) * 0.6
        } else {
            // Fallback to word similarity heuristics
            self.semantic_heuristics(&word1_lower, &word2_lower)
        }
    }

    /// Calculate co-occurrence based similarity
    pub fn cooccurrence_similarity(&self, word1: &str, word2: &str) -> f64 {
        if word1 == word2 {
            return 1.0;
        }

        let key = if word1 < word2 {
            (word1.to_string(), word2.to_string())
        } else {
            (word2.to_string(), word1.to_string())
        };

        // For now, return a basic similarity based on word patterns
        // In a full implementation, this would use corpus statistics
        self.cooccurrence_cache
            .get(&key)
            .copied()
            .unwrap_or_else(|| self.estimate_cooccurrence_similarity(word1, word2))
    }

    /// Calculate topic coherence based on keyword similarities
    pub fn topic_coherence(&self, keywords: &[String]) -> f64 {
        if keywords.len() < 2 {
            return 1.0;
        }

        let mut total_similarity = 0.0;
        let mut pair_count = 0;

        for i in 0..keywords.len() {
            for j in (i + 1)..keywords.len() {
                let similarity = self.calculate_similarity(&keywords[i], &keywords[j]);
                total_similarity += similarity;
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            total_similarity / pair_count as f64
        } else {
            0.0
        }
    }

    /// Calculate similarity between two sets of keywords
    pub fn keyword_set_similarity(&self, set1: &[String], set2: &[String]) -> f64 {
        if set1.is_empty() || set2.is_empty() {
            return 0.0;
        }

        let mut max_similarities = Vec::new();

        for word1 in set1 {
            let mut max_sim = 0.0;
            for word2 in set2 {
                let sim = self.calculate_similarity(word1, word2);
                max_sim = max_sim.max(sim);
            }
            max_similarities.push(max_sim);
        }

        if max_similarities.is_empty() {
            0.0
        } else {
            max_similarities.iter().sum::<f64>() / max_similarities.len() as f64
        }
    }

    // Private helper methods

    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();

        let len1 = chars1.len();
        let len2 = chars2.len();

        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }

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
                dp[i][j] = (dp[i - 1][j] + 1)
                    .min(dp[i][j - 1] + 1)
                    .min(dp[i - 1][j - 1] + cost);
            }
        }

        dp[len1][len2]
    }

    fn semantic_heuristics(&self, word1: &str, word2: &str) -> f64 {
        // Basic semantic similarity heuristics
        let mut similarity = 0.0;

        // Suffix similarity (for related word forms)
        if word1.len() > 3 && word2.len() > 3 {
            let suffix1 = &word1[word1.len() - 3..];
            let suffix2 = &word2[word2.len() - 3..];
            if suffix1 == suffix2 {
                similarity += 0.3;
            }
        }

        // Prefix similarity
        if word1.len() > 3 && word2.len() > 3 {
            let prefix1 = &word1[..3];
            let prefix2 = &word2[..3];
            if prefix1 == prefix2 {
                similarity += 0.2;
            }
        }

        // Root similarity (basic stemming)
        let root1 = self.simple_stem(word1);
        let root2 = self.simple_stem(word2);
        if root1 == root2 && root1.len() > 2 {
            similarity += 0.4;
        }

        similarity.min(1.0)
    }

    fn simple_stem(&self, word: &str) -> String {
        // Very basic stemming for semantic similarity
        if word.ends_with("ing") && word.len() > 4 {
            word[..word.len() - 3].to_string()
        } else if word.ends_with("ed") && word.len() > 3 {
            word[..word.len() - 2].to_string()
        } else if word.ends_with("s") && word.len() > 2 {
            word[..word.len() - 1].to_string()
        } else if word.ends_with("ly") && word.len() > 3 {
            word[..word.len() - 2].to_string()
        } else {
            word.to_string()
        }
    }

    fn estimate_cooccurrence_similarity(&self, word1: &str, word2: &str) -> f64 {
        // Simplified co-occurrence estimation based on word patterns
        let word1_lower = word1.to_lowercase();
        let word2_lower = word2.to_lowercase();

        // Check if words are likely to co-occur based on patterns
        let mut cooc_score = 0.0;

        // Same semantic category indicators
        if self.same_semantic_category(&word1_lower, &word2_lower) {
            cooc_score += 0.4;
        }

        // Syntactic compatibility
        if self.syntactically_compatible(&word1_lower, &word2_lower) {
            cooc_score += 0.3;
        }

        // Length compatibility (words of similar length often co-occur)
        let length_diff = (word1.len() as i32 - word2.len() as i32).abs();
        if length_diff <= 2 {
            cooc_score += 0.2;
        }

        cooc_score.min(1.0)
    }

    fn same_semantic_category(&self, word1: &str, word2: &str) -> bool {
        // Check for semantic category indicators
        let categories = vec![
            vec!["good", "great", "excellent", "wonderful", "amazing"], // positive
            vec!["bad", "terrible", "awful", "horrible", "poor"],       // negative
            vec!["big", "large", "huge", "enormous", "giant"],          // size
            vec!["small", "tiny", "little", "mini", "compact"],         // size
            vec!["fast", "quick", "rapid", "swift", "speedy"],          // speed
            vec!["slow", "gradual", "leisurely", "sluggish"],           // speed
        ];

        for category in &categories {
            if category.contains(&word1) && category.contains(&word2) {
                return true;
            }
        }

        false
    }

    fn syntactically_compatible(&self, word1: &str, word2: &str) -> bool {
        // Basic syntactic compatibility check
        let noun_endings = vec!["tion", "sion", "ment", "ness", "ity"];
        let verb_endings = vec!["ing", "ed", "er"];
        let adj_endings = vec!["ly", "ful", "ous", "ive"];

        let word1_is_noun = noun_endings.iter().any(|ending| word1.ends_with(ending));
        let word2_is_noun = noun_endings.iter().any(|ending| word2.ends_with(ending));

        let word1_is_verb = verb_endings.iter().any(|ending| word1.ends_with(ending));
        let word2_is_verb = verb_endings.iter().any(|ending| word2.ends_with(ending));

        let word1_is_adj = adj_endings.iter().any(|ending| word1.ends_with(ending));
        let word2_is_adj = adj_endings.iter().any(|ending| word2.ends_with(ending));

        // Nouns and adjectives often co-occur
        (word1_is_noun && word2_is_adj) || (word1_is_adj && word2_is_noun) ||
        // Verbs and nouns often co-occur
        (word1_is_verb && word2_is_noun) || (word1_is_noun && word2_is_verb)
    }

    fn build_semantic_lexicon() -> HashMap<String, Vec<String>> {
        let mut lexicon = HashMap::new();

        // Technology domain
        lexicon.insert(
            "computer".to_string(),
            vec![
                "software".to_string(),
                "hardware".to_string(),
                "programming".to_string(),
                "algorithm".to_string(),
                "data".to_string(),
                "system".to_string(),
            ],
        );

        // Science domain
        lexicon.insert(
            "science".to_string(),
            vec![
                "research".to_string(),
                "experiment".to_string(),
                "theory".to_string(),
                "hypothesis".to_string(),
                "analysis".to_string(),
                "discovery".to_string(),
            ],
        );

        // Education domain
        lexicon.insert(
            "education".to_string(),
            vec![
                "school".to_string(),
                "student".to_string(),
                "teacher".to_string(),
                "learning".to_string(),
                "knowledge".to_string(),
                "study".to_string(),
            ],
        );

        // Business domain
        lexicon.insert(
            "business".to_string(),
            vec![
                "company".to_string(),
                "market".to_string(),
                "profit".to_string(),
                "management".to_string(),
                "strategy".to_string(),
                "customer".to_string(),
            ],
        );

        // Health domain
        lexicon.insert(
            "health".to_string(),
            vec![
                "medical".to_string(),
                "doctor".to_string(),
                "patient".to_string(),
                "treatment".to_string(),
                "medicine".to_string(),
                "hospital".to_string(),
            ],
        );

        lexicon
    }
}

impl Default for SimilarityCalculator {
    fn default() -> Self {
        Self::new(SimilarityConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_character_similarity() {
        let calculator = SimilarityCalculator::default();

        assert_eq!(calculator.character_similarity("hello", "hello"), 1.0);
        assert_eq!(calculator.character_similarity("", ""), 1.0);
        assert_eq!(calculator.character_similarity("", "hello"), 0.0);

        let sim = calculator.character_similarity("hello", "hallo");
        assert!(sim > 0.5 && sim < 1.0);
    }

    #[test]
    fn test_semantic_similarity() {
        let calculator = SimilarityCalculator::default();

        assert_eq!(calculator.semantic_similarity("computer", "computer"), 1.0);

        let sim = calculator.semantic_similarity("computer", "software");
        assert!(sim > 0.5);
    }

    #[test]
    fn test_topic_coherence() {
        let calculator = SimilarityCalculator::default();

        let keywords = vec![
            "computer".to_string(),
            "software".to_string(),
            "programming".to_string(),
        ];

        let coherence = calculator.topic_coherence(&keywords);
        assert!(coherence > 0.0 && coherence <= 1.0);
    }

    #[test]
    fn test_keyword_set_similarity() {
        let calculator = SimilarityCalculator::default();

        let set1 = vec!["computer".to_string(), "software".to_string()];
        let set2 = vec!["hardware".to_string(), "programming".to_string()];

        let sim = calculator.keyword_set_similarity(&set1, &set2);
        assert!(sim >= 0.0 && sim <= 1.0);
    }

    #[test]
    fn test_levenshtein_distance() {
        let calculator = SimilarityCalculator::default();

        assert_eq!(calculator.levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(calculator.levenshtein_distance("hello", "hello"), 0);
        assert_eq!(calculator.levenshtein_distance("", "hello"), 5);
    }
}
