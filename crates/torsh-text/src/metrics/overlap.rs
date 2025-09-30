//! Comprehensive word overlap metrics for text analysis
//!
//! This module provides various methods for measuring overlap between texts,
//! including traditional set-based metrics, n-gram overlaps, and advanced
//! positional and weighted overlap calculations.

use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_core::random::{rng, Random};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

#[derive(Debug, Clone, PartialEq)]
pub enum OverlapMetric {
    Jaccard,
    Dice,
    Cosine,
    Overlap,
    SimpleRatio,
    WeightedRatio,
    PositionalOverlap,
    NGramOverlap(usize),
    SemanticOverlap,
    HierarchicalOverlap,
}

#[derive(Debug, Clone)]
pub struct OverlapConfig {
    pub case_sensitive: bool,
    pub remove_punctuation: bool,
    pub remove_stopwords: bool,
    pub min_word_length: usize,
    pub use_stemming: bool,
    pub position_weight: f64,
    pub frequency_weight: f64,
    pub semantic_threshold: f64,
}

impl Default for OverlapConfig {
    fn default() -> Self {
        Self {
            case_sensitive: false,
            remove_punctuation: true,
            remove_stopwords: true,
            min_word_length: 2,
            use_stemming: false,
            position_weight: 0.1,
            frequency_weight: 0.5,
            semantic_threshold: 0.7,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OverlapResult {
    pub jaccard: f64,
    pub dice: f64,
    pub cosine: f64,
    pub overlap_coefficient: f64,
    pub simple_ratio: f64,
    pub weighted_ratio: f64,
    pub positional_overlap: f64,
    pub ngram_overlap: HashMap<usize, f64>,
    pub semantic_overlap: f64,
    pub intersection_size: usize,
    pub union_size: usize,
    pub text1_unique: usize,
    pub text2_unique: usize,
    pub common_words: Vec<String>,
    pub confidence_score: f64,
}

#[derive(Debug, Clone)]
pub struct NGramOverlapResult {
    pub ngram_size: usize,
    pub total_ngrams_text1: usize,
    pub total_ngrams_text2: usize,
    pub common_ngrams: usize,
    pub unique_text1: usize,
    pub unique_text2: usize,
    pub jaccard: f64,
    pub dice: f64,
    pub overlap_ratio: f64,
    pub coverage_text1: f64,
    pub coverage_text2: f64,
}

#[derive(Debug, Clone)]
pub struct PositionalOverlapResult {
    pub exact_position_matches: usize,
    pub near_position_matches: usize,
    pub position_similarity: f64,
    pub order_preservation: f64,
    pub distance_penalty: f64,
    pub weighted_overlap: f64,
}

#[derive(Debug, Clone)]
pub struct SemanticOverlapResult {
    pub semantic_matches: usize,
    pub semantic_similarity: f64,
    pub concept_overlap: f64,
    pub domain_alignment: f64,
    pub contextual_overlap: f64,
    pub weighted_semantic_score: f64,
}

pub struct WordOverlapCalculator {
    config: OverlapConfig,
    stopwords: HashSet<String>,
    semantic_lexicon: HashMap<String, Vec<String>>,
}

impl WordOverlapCalculator {
    pub fn new(config: OverlapConfig) -> Self {
        let stopwords = Self::load_stopwords();
        let semantic_lexicon = Self::build_semantic_lexicon();

        Self {
            config,
            stopwords,
            semantic_lexicon,
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(OverlapConfig::default())
    }

    fn load_stopwords() -> HashSet<String> {
        let words = vec![
            "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from", "has", "he",
            "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will",
            "with", "would", "could", "should", "shall", "may", "might", "can", "must", "do",
            "does", "did", "have", "had", "having", "this", "these", "they", "them", "their",
            "there", "then", "than", "when", "where", "who", "what", "why", "how", "which",
            "while", "we", "us", "our", "you", "your", "i", "my", "me", "mine", "his", "her",
            "hers", "him", "she", "if", "or", "but", "nor", "so", "yet", "because", "since",
            "unless", "until", "before", "after", "above", "below", "up", "down", "out", "off",
            "over", "under", "again", "further", "once", "here", "any", "both", "each", "few",
            "more", "most", "other", "some", "such", "no", "not", "only", "own", "same", "so",
            "too", "very",
        ];
        words.into_iter().map(String::from).collect()
    }

    fn build_semantic_lexicon() -> HashMap<String, Vec<String>> {
        let mut lexicon = HashMap::new();

        lexicon.insert(
            "good".to_string(),
            vec![
                "excellent",
                "great",
                "wonderful",
                "amazing",
                "fantastic",
                "superb",
                "outstanding",
                "brilliant",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );
        lexicon.insert(
            "bad".to_string(),
            vec![
                "terrible",
                "awful",
                "horrible",
                "dreadful",
                "poor",
                "disappointing",
                "inadequate",
                "inferior",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );
        lexicon.insert(
            "big".to_string(),
            vec![
                "large",
                "huge",
                "enormous",
                "massive",
                "gigantic",
                "immense",
                "vast",
                "substantial",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );
        lexicon.insert(
            "small".to_string(),
            vec![
                "tiny",
                "little",
                "minute",
                "compact",
                "petite",
                "miniature",
                "microscopic",
                "diminutive",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );
        lexicon.insert(
            "fast".to_string(),
            vec![
                "quick",
                "rapid",
                "speedy",
                "swift",
                "hasty",
                "brisk",
                "prompt",
                "expeditious",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );
        lexicon.insert(
            "slow".to_string(),
            vec![
                "sluggish",
                "leisurely",
                "gradual",
                "unhurried",
                "delayed",
                "tardy",
                "plodding",
                "dawdling",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );
        lexicon.insert(
            "happy".to_string(),
            vec![
                "joyful",
                "cheerful",
                "delighted",
                "elated",
                "ecstatic",
                "jubilant",
                "euphoric",
                "blissful",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );
        lexicon.insert(
            "sad".to_string(),
            vec![
                "melancholy",
                "sorrowful",
                "dejected",
                "despondent",
                "gloomy",
                "mournful",
                "disheartened",
                "downcast",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        );

        lexicon
    }

    fn preprocess_text(&self, text: &str) -> Vec<String> {
        let mut processed = text.to_string();

        if self.config.remove_punctuation {
            processed = processed
                .chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                .collect();
        }

        if !self.config.case_sensitive {
            processed = processed.to_lowercase();
        }

        let words: Vec<String> = processed
            .split_whitespace()
            .filter(|word| {
                if word.len() < self.config.min_word_length {
                    return false;
                }
                if self.config.remove_stopwords && self.stopwords.contains(*word) {
                    return false;
                }
                true
            })
            .map(|word| {
                if self.config.use_stemming {
                    self.simple_stem(word)
                } else {
                    word.to_string()
                }
            })
            .collect();

        words
    }

    fn simple_stem(&self, word: &str) -> String {
        let suffixes = vec![
            "ing", "ed", "er", "est", "ly", "tion", "sion", "ness", "ment", "able", "ible",
        ];

        for suffix in suffixes {
            if word.ends_with(suffix) && word.len() > suffix.len() + 2 {
                return word[..word.len() - suffix.len()].to_string();
            }
        }

        word.to_string()
    }

    pub fn calculate_comprehensive_overlap(&self, text1: &str, text2: &str) -> OverlapResult {
        let words1 = self.preprocess_text(text1);
        let words2 = self.preprocess_text(text2);

        let set1: HashSet<String> = words1.iter().cloned().collect();
        let set2: HashSet<String> = words2.iter().cloned().collect();

        let intersection: HashSet<_> = set1.intersection(&set2).collect();
        let union: HashSet<_> = set1.union(&set2).collect();

        let intersection_size = intersection.len();
        let union_size = union.len();
        let text1_unique = set1.len() - intersection_size;
        let text2_unique = set2.len() - intersection_size;

        let jaccard = if union_size > 0 {
            intersection_size as f64 / union_size as f64
        } else {
            0.0
        };

        let dice = if (set1.len() + set2.len()) > 0 {
            2.0 * intersection_size as f64 / (set1.len() + set2.len()) as f64
        } else {
            0.0
        };

        let cosine = self.calculate_cosine_similarity(&words1, &words2);
        let overlap_coefficient = self.calculate_overlap_coefficient(&set1, &set2);
        let simple_ratio = self.calculate_simple_ratio(&words1, &words2);
        let weighted_ratio = self.calculate_weighted_ratio(&words1, &words2);
        let positional_overlap = self
            .calculate_positional_overlap(&words1, &words2)
            .weighted_overlap;

        let mut ngram_overlap = HashMap::new();
        for n in 1..=3 {
            let ngram_result = self.calculate_ngram_overlap(&words1, &words2, n);
            ngram_overlap.insert(n, ngram_result.jaccard);
        }

        let semantic_overlap = self
            .calculate_semantic_overlap(&words1, &words2)
            .weighted_semantic_score;

        let common_words: Vec<String> = intersection.into_iter().cloned().collect();
        let confidence_score = self.calculate_confidence_score(jaccard, dice, cosine);

        OverlapResult {
            jaccard,
            dice,
            cosine,
            overlap_coefficient,
            simple_ratio,
            weighted_ratio,
            positional_overlap,
            ngram_overlap,
            semantic_overlap,
            intersection_size,
            union_size,
            text1_unique,
            text2_unique,
            common_words,
            confidence_score,
        }
    }

    fn calculate_cosine_similarity(&self, words1: &[String], words2: &[String]) -> f64 {
        let mut vocab = HashSet::new();
        words1.iter().for_each(|w| {
            vocab.insert(w.clone());
        });
        words2.iter().for_each(|w| {
            vocab.insert(w.clone());
        });

        let vocab_vec: Vec<String> = vocab.into_iter().collect();
        let mut vec1 = vec![0.0; vocab_vec.len()];
        let mut vec2 = vec![0.0; vocab_vec.len()];

        for (i, word) in vocab_vec.iter().enumerate() {
            vec1[i] = words1.iter().filter(|&w| w == word).count() as f64;
            vec2[i] = words2.iter().filter(|&w| w == word).count() as f64;
        }

        let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = vec1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = vec2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        }
    }

    fn calculate_overlap_coefficient(&self, set1: &HashSet<String>, set2: &HashSet<String>) -> f64 {
        let intersection_size = set1.intersection(set2).count();
        let min_size = set1.len().min(set2.len());

        if min_size > 0 {
            intersection_size as f64 / min_size as f64
        } else {
            0.0
        }
    }

    fn calculate_simple_ratio(&self, words1: &[String], words2: &[String]) -> f64 {
        let common_words = words1.iter().filter(|word| words2.contains(word)).count();

        let total_words = (words1.len() + words2.len()) as f64;
        if total_words > 0.0 {
            2.0 * common_words as f64 / total_words
        } else {
            0.0
        }
    }

    fn calculate_weighted_ratio(&self, words1: &[String], words2: &[String]) -> f64 {
        let freq1 = self.calculate_word_frequencies(words1);
        let freq2 = self.calculate_word_frequencies(words2);

        let mut weighted_common = 0.0;
        let mut total_weight1 = 0.0;
        let mut total_weight2 = 0.0;

        for (word, count1) in &freq1 {
            let weight1 = (*count1 as f64).log2() + 1.0;
            total_weight1 += weight1;

            if let Some(count2) = freq2.get(word) {
                let weight2 = (*count2 as f64).log2() + 1.0;
                weighted_common += weight1.min(weight2);
            }
        }

        for (word, count2) in &freq2 {
            let weight2 = (*count2 as f64).log2() + 1.0;
            total_weight2 += weight2;
        }

        let total_weight = total_weight1 + total_weight2;
        if total_weight > 0.0 {
            2.0 * weighted_common / total_weight
        } else {
            0.0
        }
    }

    fn calculate_word_frequencies(&self, words: &[String]) -> HashMap<String, usize> {
        let mut freq = HashMap::new();
        for word in words {
            *freq.entry(word.clone()).or_insert(0) += 1;
        }
        freq
    }

    pub fn calculate_ngram_overlap(
        &self,
        words1: &[String],
        words2: &[String],
        n: usize,
    ) -> NGramOverlapResult {
        let ngrams1 = self.generate_ngrams(words1, n);
        let ngrams2 = self.generate_ngrams(words2, n);

        let set1: HashSet<Vec<String>> = ngrams1.into_iter().collect();
        let set2: HashSet<Vec<String>> = ngrams2.into_iter().collect();

        let intersection: HashSet<_> = set1.intersection(&set2).collect();
        let union: HashSet<_> = set1.union(&set2).collect();

        let common_ngrams = intersection.len();
        let total_ngrams_text1 = set1.len();
        let total_ngrams_text2 = set2.len();
        let unique_text1 = set1.len() - common_ngrams;
        let unique_text2 = set2.len() - common_ngrams;

        let jaccard = if union.len() > 0 {
            common_ngrams as f64 / union.len() as f64
        } else {
            0.0
        };

        let dice = if (set1.len() + set2.len()) > 0 {
            2.0 * common_ngrams as f64 / (set1.len() + set2.len()) as f64
        } else {
            0.0
        };

        let overlap_ratio = if set1.len().max(set2.len()) > 0 {
            common_ngrams as f64 / set1.len().max(set2.len()) as f64
        } else {
            0.0
        };

        let coverage_text1 = if set1.len() > 0 {
            common_ngrams as f64 / set1.len() as f64
        } else {
            0.0
        };

        let coverage_text2 = if set2.len() > 0 {
            common_ngrams as f64 / set2.len() as f64
        } else {
            0.0
        };

        NGramOverlapResult {
            ngram_size: n,
            total_ngrams_text1,
            total_ngrams_text2,
            common_ngrams,
            unique_text1,
            unique_text2,
            jaccard,
            dice,
            overlap_ratio,
            coverage_text1,
            coverage_text2,
        }
    }

    fn generate_ngrams(&self, words: &[String], n: usize) -> Vec<Vec<String>> {
        if words.len() < n {
            return vec![];
        }

        words.windows(n).map(|window| window.to_vec()).collect()
    }

    pub fn calculate_positional_overlap(
        &self,
        words1: &[String],
        words2: &[String],
    ) -> PositionalOverlapResult {
        let mut exact_matches = 0;
        let mut near_matches = 0;
        let mut position_similarity_sum = 0.0;
        let mut total_comparisons = 0;
        let mut distance_penalty_sum = 0.0;

        let min_len = words1.len().min(words2.len());
        let max_len = words1.len().max(words2.len());

        for i in 0..min_len {
            if words1[i] == words2[i] {
                exact_matches += 1;
                position_similarity_sum += 1.0;
            } else {
                if let Some(pos) = words2.iter().position(|w| w == &words1[i]) {
                    let distance = (i as isize - pos as isize).abs() as f64;
                    let max_distance = max_len as f64;
                    let similarity = 1.0 - (distance / max_distance);

                    if similarity > 0.5 {
                        near_matches += 1;
                        position_similarity_sum += similarity * 0.5;
                    }

                    distance_penalty_sum += distance / max_distance;
                }
            }
            total_comparisons += 1;
        }

        let position_similarity = if total_comparisons > 0 {
            position_similarity_sum / total_comparisons as f64
        } else {
            0.0
        };

        let order_preservation = self.calculate_order_preservation(words1, words2);

        let distance_penalty = if total_comparisons > 0 {
            distance_penalty_sum / total_comparisons as f64
        } else {
            0.0
        };

        let weighted_overlap = (position_similarity * (1.0 - self.config.position_weight))
            + (order_preservation * self.config.position_weight);

        PositionalOverlapResult {
            exact_position_matches: exact_matches,
            near_position_matches: near_matches,
            position_similarity,
            order_preservation,
            distance_penalty,
            weighted_overlap,
        }
    }

    fn calculate_order_preservation(&self, words1: &[String], words2: &[String]) -> f64 {
        let common_words: Vec<String> = words1
            .iter()
            .filter(|word| words2.contains(word))
            .cloned()
            .collect();

        if common_words.len() < 2 {
            return 1.0;
        }

        let mut preserved_pairs = 0;
        let mut total_pairs = 0;

        for i in 0..common_words.len() - 1 {
            for j in i + 1..common_words.len() {
                let word1 = &common_words[i];
                let word2 = &common_words[j];

                if let (Some(pos1_1), Some(pos1_2)) = (
                    words1.iter().position(|w| w == word1),
                    words1.iter().position(|w| w == word2),
                ) {
                    if let (Some(pos2_1), Some(pos2_2)) = (
                        words2.iter().position(|w| w == word1),
                        words2.iter().position(|w| w == word2),
                    ) {
                        if (pos1_1 < pos1_2 && pos2_1 < pos2_2)
                            || (pos1_1 > pos1_2 && pos2_1 > pos2_2)
                        {
                            preserved_pairs += 1;
                        }
                        total_pairs += 1;
                    }
                }
            }
        }

        if total_pairs > 0 {
            preserved_pairs as f64 / total_pairs as f64
        } else {
            1.0
        }
    }

    pub fn calculate_semantic_overlap(
        &self,
        words1: &[String],
        words2: &[String],
    ) -> SemanticOverlapResult {
        let mut semantic_matches = 0;
        let mut semantic_similarity_sum = 0.0;
        let mut total_comparisons = 0;

        for word1 in words1 {
            let mut best_similarity = 0.0;
            let mut found_match = false;

            for word2 in words2 {
                if word1 == word2 {
                    semantic_matches += 1;
                    best_similarity = 1.0;
                    found_match = true;
                    break;
                } else {
                    let sim = self.calculate_word_semantic_similarity(word1, word2);
                    if sim > self.config.semantic_threshold {
                        best_similarity = best_similarity.max(sim);
                        found_match = true;
                    }
                }
            }

            if found_match {
                semantic_similarity_sum += best_similarity;
            }
            total_comparisons += 1;
        }

        let semantic_similarity = if total_comparisons > 0 {
            semantic_similarity_sum / total_comparisons as f64
        } else {
            0.0
        };

        let concept_overlap = self.calculate_concept_overlap(words1, words2);
        let domain_alignment = self.calculate_domain_alignment(words1, words2);
        let contextual_overlap = self.calculate_contextual_overlap(words1, words2);

        let weighted_semantic_score = (semantic_similarity * 0.4)
            + (concept_overlap * 0.3)
            + (domain_alignment * 0.2)
            + (contextual_overlap * 0.1);

        SemanticOverlapResult {
            semantic_matches,
            semantic_similarity,
            concept_overlap,
            domain_alignment,
            contextual_overlap,
            weighted_semantic_score,
        }
    }

    fn calculate_word_semantic_similarity(&self, word1: &str, word2: &str) -> f64 {
        if let Some(synonyms) = self.semantic_lexicon.get(word1) {
            if synonyms.contains(&word2.to_string()) {
                return 0.8;
            }
        }

        if let Some(synonyms) = self.semantic_lexicon.get(word2) {
            if synonyms.contains(&word1.to_string()) {
                return 0.8;
            }
        }

        let edit_distance = self.calculate_edit_distance(word1, word2);
        let max_len = word1.len().max(word2.len());

        if max_len > 0 {
            1.0 - (edit_distance as f64 / max_len as f64)
        } else {
            0.0
        }
    }

    fn calculate_edit_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();

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

    fn calculate_concept_overlap(&self, words1: &[String], words2: &[String]) -> f64 {
        let concepts1 = self.extract_concepts(words1);
        let concepts2 = self.extract_concepts(words2);

        let common_concepts = concepts1.intersection(&concepts2).count();
        let total_concepts = concepts1.union(&concepts2).count();

        if total_concepts > 0 {
            common_concepts as f64 / total_concepts as f64
        } else {
            0.0
        }
    }

    fn extract_concepts(&self, words: &[String]) -> HashSet<String> {
        let mut concepts = HashSet::new();

        for word in words {
            if let Some(synonyms) = self.semantic_lexicon.get(word) {
                concepts.insert(word.clone());
                for synonym in synonyms {
                    concepts.insert(synonym.clone());
                }
            } else {
                concepts.insert(word.clone());
            }
        }

        concepts
    }

    fn calculate_domain_alignment(&self, words1: &[String], words2: &[String]) -> f64 {
        let domain1 = self.identify_dominant_domain(words1);
        let domain2 = self.identify_dominant_domain(words2);

        if domain1 == domain2 {
            1.0
        } else {
            self.calculate_domain_similarity(&domain1, &domain2)
        }
    }

    fn identify_dominant_domain(&self, words: &[String]) -> String {
        let mut domain_scores = HashMap::new();

        for word in words {
            if word.len() > 3 {
                let domain =
                    if word.contains("tech") || word.contains("data") || word.contains("system") {
                        "technology"
                    } else if word.contains("business")
                        || word.contains("market")
                        || word.contains("sales")
                    {
                        "business"
                    } else if word.contains("health")
                        || word.contains("medical")
                        || word.contains("patient")
                    {
                        "healthcare"
                    } else if word.contains("learn")
                        || word.contains("study")
                        || word.contains("research")
                    {
                        "education"
                    } else {
                        "general"
                    };

                *domain_scores.entry(domain.to_string()).or_insert(0) += 1;
            }
        }

        domain_scores
            .into_iter()
            .max_by_key(|(_, score)| *score)
            .map(|(domain, _)| domain)
            .unwrap_or_else(|| "general".to_string())
    }

    fn calculate_domain_similarity(&self, domain1: &str, domain2: &str) -> f64 {
        match (domain1, domain2) {
            ("technology", "business") | ("business", "technology") => 0.6,
            ("healthcare", "technology") | ("technology", "healthcare") => 0.5,
            ("education", "technology") | ("technology", "education") => 0.7,
            ("business", "healthcare") | ("healthcare", "business") => 0.4,
            ("education", "business") | ("business", "education") => 0.5,
            ("education", "healthcare") | ("healthcare", "education") => 0.6,
            _ => 0.3,
        }
    }

    fn calculate_contextual_overlap(&self, words1: &[String], words2: &[String]) -> f64 {
        let context1 = self.extract_context_features(words1);
        let context2 = self.extract_context_features(words2);

        let mut common_features = 0;
        let mut total_features = 0;

        for feature in &context1 {
            if context2.contains(feature) {
                common_features += 1;
            }
            total_features += 1;
        }

        for feature in &context2 {
            if !context1.contains(feature) {
                total_features += 1;
            }
        }

        if total_features > 0 {
            common_features as f64 / total_features as f64
        } else {
            0.0
        }
    }

    fn extract_context_features(&self, words: &[String]) -> HashSet<String> {
        let mut features = HashSet::new();

        for window in words.windows(2) {
            if window.len() == 2 {
                let bigram = format!("{}_{}", window[0], window[1]);
                features.insert(bigram);
            }
        }

        for window in words.windows(3) {
            if window.len() == 3 {
                let trigram = format!("{}_{}_{}", window[0], window[1], window[2]);
                features.insert(trigram);
            }
        }

        features
    }

    fn calculate_confidence_score(&self, jaccard: f64, dice: f64, cosine: f64) -> f64 {
        let metrics = vec![jaccard, dice, cosine];
        let mean = metrics.iter().sum::<f64>() / metrics.len() as f64;
        let variance =
            metrics.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / metrics.len() as f64;
        let std_dev = variance.sqrt();

        let consistency = 1.0 - std_dev;
        let strength = mean;

        (consistency * 0.3) + (strength * 0.7)
    }

    pub fn calculate_overlap_matrix(&self, texts: &[String]) -> Array2<f64> {
        let n = texts.len();
        let mut matrix = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    matrix[[i, j]] = 1.0;
                } else {
                    let result = self.calculate_comprehensive_overlap(&texts[i], &texts[j]);
                    matrix[[i, j]] = result.jaccard;
                }
            }
        }

        matrix
    }

    pub fn find_most_similar_pairs(
        &self,
        texts: &[String],
        threshold: f64,
    ) -> Vec<(usize, usize, f64)> {
        let mut pairs = Vec::new();

        for i in 0..texts.len() {
            for j in (i + 1)..texts.len() {
                let result = self.calculate_comprehensive_overlap(&texts[i], &texts[j]);
                if result.jaccard >= threshold {
                    pairs.push((i, j, result.jaccard));
                }
            }
        }

        pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        pairs
    }

    pub fn cluster_by_overlap(&self, texts: &[String], threshold: f64) -> Vec<Vec<usize>> {
        let mut clusters = Vec::new();
        let mut assigned = vec![false; texts.len()];

        for i in 0..texts.len() {
            if assigned[i] {
                continue;
            }

            let mut cluster = vec![i];
            assigned[i] = true;

            for j in (i + 1)..texts.len() {
                if assigned[j] {
                    continue;
                }

                let result = self.calculate_comprehensive_overlap(&texts[i], &texts[j]);
                if result.jaccard >= threshold {
                    cluster.push(j);
                    assigned[j] = true;
                }
            }

            clusters.push(cluster);
        }

        clusters
    }

    pub fn analyze_overlap_distribution(&self, overlaps: &[f64]) -> OverlapDistributionAnalysis {
        if overlaps.is_empty() {
            return OverlapDistributionAnalysis::default();
        }

        let mut sorted_overlaps = overlaps.to_vec();
        sorted_overlaps.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = overlaps.iter().sum::<f64>() / overlaps.len() as f64;
        let variance =
            overlaps.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / overlaps.len() as f64;
        let std_dev = variance.sqrt();

        let min = sorted_overlaps[0];
        let max = sorted_overlaps[sorted_overlaps.len() - 1];
        let median = if sorted_overlaps.len() % 2 == 0 {
            (sorted_overlaps[sorted_overlaps.len() / 2 - 1]
                + sorted_overlaps[sorted_overlaps.len() / 2])
                / 2.0
        } else {
            sorted_overlaps[sorted_overlaps.len() / 2]
        };

        let q1_idx = sorted_overlaps.len() / 4;
        let q3_idx = 3 * sorted_overlaps.len() / 4;
        let q1 = sorted_overlaps[q1_idx];
        let q3 = sorted_overlaps[q3_idx];

        OverlapDistributionAnalysis {
            mean,
            std_dev,
            min,
            max,
            median,
            q1,
            q3,
            count: overlaps.len(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct OverlapDistributionAnalysis {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub q1: f64,
    pub q3: f64,
    pub count: usize,
}

pub fn jaccard_similarity<T: Clone + Eq + Hash>(set1: &HashSet<T>, set2: &HashSet<T>) -> f64 {
    let intersection_size = set1.intersection(set2).count();
    let union_size = set1.union(set2).count();

    if union_size > 0 {
        intersection_size as f64 / union_size as f64
    } else {
        0.0
    }
}

pub fn dice_coefficient<T: Clone + Eq + Hash>(set1: &HashSet<T>, set2: &HashSet<T>) -> f64 {
    let intersection_size = set1.intersection(set2).count();

    if (set1.len() + set2.len()) > 0 {
        2.0 * intersection_size as f64 / (set1.len() + set2.len()) as f64
    } else {
        0.0
    }
}

pub fn overlap_coefficient<T: Clone + Eq + Hash>(set1: &HashSet<T>, set2: &HashSet<T>) -> f64 {
    let intersection_size = set1.intersection(set2).count();
    let min_size = set1.len().min(set2.len());

    if min_size > 0 {
        intersection_size as f64 / min_size as f64
    } else {
        0.0
    }
}

pub fn word_overlap_ratio(text1: &str, text2: &str) -> f64 {
    let words1: HashSet<String> = text1.split_whitespace().map(|s| s.to_lowercase()).collect();
    let words2: HashSet<String> = text2.split_whitespace().map(|s| s.to_lowercase()).collect();

    jaccard_similarity(&words1, &words2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_overlap_calculation() {
        let calculator = WordOverlapCalculator::with_default_config();
        let text1 = "The quick brown fox jumps over the lazy dog";
        let text2 = "A quick brown fox leaps over a lazy cat";

        let result = calculator.calculate_comprehensive_overlap(text1, text2);

        assert!(result.jaccard > 0.0);
        assert!(result.dice > 0.0);
        assert!(result.cosine > 0.0);
        assert!(result.overlap_coefficient > 0.0);
        assert!(result.confidence_score > 0.0);
    }

    #[test]
    fn test_identical_texts() {
        let calculator = WordOverlapCalculator::with_default_config();
        let text = "Hello world this is a test";

        let result = calculator.calculate_comprehensive_overlap(text, text);

        assert!((result.jaccard - 1.0).abs() < 1e-10);
        assert!((result.dice - 1.0).abs() < 1e-10);
        assert!((result.cosine - 1.0).abs() < 1e-10);
        assert!((result.overlap_coefficient - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_no_overlap() {
        let calculator = WordOverlapCalculator::with_default_config();
        let text1 = "apple banana cherry";
        let text2 = "dog elephant fox";

        let result = calculator.calculate_comprehensive_overlap(text1, text2);

        assert!((result.jaccard - 0.0).abs() < 1e-10);
        assert!((result.dice - 0.0).abs() < 1e-10);
        assert!(result.cosine >= 0.0);
    }

    #[test]
    fn test_ngram_overlap() {
        let calculator = WordOverlapCalculator::with_default_config();
        let words1 = vec![
            "the".to_string(),
            "quick".to_string(),
            "brown".to_string(),
            "fox".to_string(),
        ];
        let words2 = vec![
            "a".to_string(),
            "quick".to_string(),
            "brown".to_string(),
            "cat".to_string(),
        ];

        let result = calculator.calculate_ngram_overlap(&words1, &words2, 2);

        assert!(result.common_ngrams > 0);
        assert!(result.jaccard > 0.0);
        assert!(result.dice > 0.0);
    }

    #[test]
    fn test_positional_overlap() {
        let calculator = WordOverlapCalculator::with_default_config();
        let words1 = vec!["hello".to_string(), "world".to_string(), "test".to_string()];
        let words2 = vec![
            "hello".to_string(),
            "world".to_string(),
            "example".to_string(),
        ];

        let result = calculator.calculate_positional_overlap(&words1, &words2);

        assert!(result.exact_position_matches > 0);
        assert!(result.position_similarity > 0.0);
        assert!(result.order_preservation > 0.0);
    }

    #[test]
    fn test_semantic_overlap() {
        let calculator = WordOverlapCalculator::with_default_config();
        let words1 = vec!["good".to_string(), "excellent".to_string()];
        let words2 = vec!["great".to_string(), "wonderful".to_string()];

        let result = calculator.calculate_semantic_overlap(&words1, &words2);

        assert!(result.weighted_semantic_score > 0.0);
    }

    #[test]
    fn test_utility_functions() {
        let set1: HashSet<i32> = vec![1, 2, 3, 4].into_iter().collect();
        let set2: HashSet<i32> = vec![3, 4, 5, 6].into_iter().collect();

        let jaccard = jaccard_similarity(&set1, &set2);
        let dice = dice_coefficient(&set1, &set2);
        let overlap = overlap_coefficient(&set1, &set2);

        assert!((jaccard - 1.0 / 3.0).abs() < 1e-10);
        assert!((dice - 0.5).abs() < 1e-10);
        assert!((overlap - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_word_overlap_ratio() {
        let ratio = word_overlap_ratio("hello world test", "hello world example");
        assert!(ratio > 0.0);
        assert!(ratio < 1.0);
    }

    #[test]
    fn test_overlap_matrix() {
        let calculator = WordOverlapCalculator::with_default_config();
        let texts = vec![
            "hello world".to_string(),
            "hello universe".to_string(),
            "goodbye world".to_string(),
        ];

        let matrix = calculator.calculate_overlap_matrix(&texts);

        assert_eq!(matrix.shape(), [3, 3]);
        assert!((matrix[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((matrix[[1, 1]] - 1.0).abs() < 1e-10);
        assert!((matrix[[2, 2]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_clustering() {
        let calculator = WordOverlapCalculator::with_default_config();
        let texts = vec![
            "apple banana cherry".to_string(),
            "apple banana grape".to_string(),
            "dog cat mouse".to_string(),
            "dog cat bird".to_string(),
        ];

        let clusters = calculator.cluster_by_overlap(&texts, 0.3);

        assert!(clusters.len() <= texts.len());
        assert!(!clusters.is_empty());
    }

    #[test]
    fn test_overlap_distribution_analysis() {
        let calculator = WordOverlapCalculator::with_default_config();
        let overlaps = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

        let analysis = calculator.analyze_overlap_distribution(&overlaps);

        assert!((analysis.mean - 0.5).abs() < 1e-10);
        assert!(analysis.std_dev > 0.0);
        assert!((analysis.median - 0.5).abs() < 1e-10);
        assert_eq!(analysis.count, 9);
    }

    #[test]
    fn test_config_variations() {
        let mut config = OverlapConfig::default();
        config.case_sensitive = true;
        config.remove_stopwords = false;

        let calculator = WordOverlapCalculator::new(config);
        let result = calculator.calculate_comprehensive_overlap("The CAT", "the cat");

        assert!(result.jaccard < 1.0);
    }

    #[test]
    fn test_edge_cases() {
        let calculator = WordOverlapCalculator::with_default_config();

        let result = calculator.calculate_comprehensive_overlap("", "");
        assert!((result.jaccard - 0.0).abs() < 1e-10);

        let result = calculator.calculate_comprehensive_overlap("hello", "");
        assert!((result.jaccard - 0.0).abs() < 1e-10);

        let result = calculator.calculate_comprehensive_overlap("", "world");
        assert!((result.jaccard - 0.0).abs() < 1e-10);
    }
}
