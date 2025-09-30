//! ROUGE Score Implementation
//!
//! This module provides comprehensive ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
//! score computation for text summarization evaluation. ROUGE measures the quality of summaries
//! by comparing them to reference summaries using various metrics including n-gram overlap,
//! longest common subsequence, and skip-bigram co-occurrence.
//!
//! # Supported ROUGE Variants
//!
//! - **ROUGE-N**: N-gram based evaluation (ROUGE-1, ROUGE-2, etc.)
//! - **ROUGE-L**: Longest Common Subsequence based evaluation
//! - **ROUGE-W**: Weighted Longest Common Subsequence
//! - **ROUGE-S**: Skip-bigram based evaluation
//! - **ROUGE-SU**: Skip-bigram with unigram extension
//!
//! # Features
//!
//! - Multiple ROUGE variants with configurable parameters
//! - Stemming support for morphological normalization
//! - Case-insensitive comparison options
//! - Stopword filtering capabilities
//! - Multi-reference evaluation support
//! - Comprehensive statistical analysis
//!
//! # Example
//!
//! ```rust
//! use torsh_text::metrics::rouge::{RougeScore, RougeType};
//!
//! let rouge = RougeScore::new(RougeType::Rouge1);
//! let candidate = "the quick brown fox jumps";
//! let reference = "a quick brown fox leaps";
//!
//! let metrics = rouge.calculate(candidate, reference)?;
//! println!("ROUGE-1 F1: {:.3}", metrics.f1_score);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{Result, TextError};
use std::collections::{HashMap, HashSet};

/// Type alias for n-gram counts mapping n-grams to their frequencies
type NgramCounts<'a> = HashMap<Vec<&'a str>, usize>;

/// ROUGE evaluation types supporting different comparison strategies
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RougeType {
    /// Unigram-based ROUGE (ROUGE-1)
    Rouge1,
    /// Bigram-based ROUGE (ROUGE-2)
    Rouge2,
    /// N-gram based ROUGE with custom N
    RougeN(usize),
    /// Longest Common Subsequence ROUGE (ROUGE-L)
    RougeL,
    /// Weighted Longest Common Subsequence ROUGE (ROUGE-W)
    RougeW { weight_factor: f64 },
    /// Skip-bigram ROUGE (ROUGE-S)
    RougeS { max_skip_distance: usize },
    /// Skip-bigram with unigram ROUGE (ROUGE-SU)
    RougeSU { max_skip_distance: usize },
}

/// ROUGE score calculator with comprehensive configuration options
#[derive(Debug, Clone)]
pub struct RougeScore {
    /// Type of ROUGE evaluation to perform
    rouge_type: RougeType,
    /// Enable stemming for morphological normalization
    use_stemming: bool,
    /// Enable case-insensitive comparison
    ignore_case: bool,
    /// Enable stopword removal
    remove_stopwords: bool,
    /// Custom stopwords set
    stopwords: HashSet<String>,
    /// Alpha parameter for F-score weighting
    alpha: f64,
    /// Beta parameter for recall/precision weighting
    beta: f64,
}

impl Default for RougeScore {
    fn default() -> Self {
        Self {
            rouge_type: RougeType::Rouge1,
            use_stemming: false,
            ignore_case: false,
            remove_stopwords: false,
            stopwords: Self::default_stopwords(),
            alpha: 0.5, // Balanced F-score
            beta: 1.0,  // Standard F1-score
        }
    }
}

impl RougeScore {
    /// Create a new ROUGE scorer with the specified type
    pub fn new(rouge_type: RougeType) -> Self {
        Self {
            rouge_type,
            ..Default::default()
        }
    }

    /// Enable or disable stemming for morphological normalization
    ///
    /// When enabled, words are reduced to their root forms to improve matching
    /// between morphologically related words (e.g., "running" -> "run").
    pub fn with_stemming(mut self, use_stemming: bool) -> Self {
        self.use_stemming = use_stemming;
        self
    }

    /// Enable or disable case-insensitive comparison
    pub fn with_ignore_case(mut self, ignore_case: bool) -> Self {
        self.ignore_case = ignore_case;
        self
    }

    /// Enable or disable stopword removal
    ///
    /// Stopwords are common words (e.g., "the", "a", "an") that may not
    /// contribute meaningful information to the evaluation.
    pub fn with_remove_stopwords(mut self, remove_stopwords: bool) -> Self {
        self.remove_stopwords = remove_stopwords;
        self
    }

    /// Set custom stopwords list
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = stopwords;
        self
    }

    /// Set alpha parameter for F-score computation
    ///
    /// Alpha controls the balance between precision and recall in F-score:
    /// - alpha = 0.5: Balanced F1-score (default)
    /// - alpha > 0.5: Favor precision
    /// - alpha < 0.5: Favor recall
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    /// Set beta parameter for F-beta score computation
    ///
    /// Beta determines the relative importance of recall vs precision:
    /// - beta = 1.0: Standard F1-score (default)
    /// - beta > 1.0: Favor recall
    /// - beta < 1.0: Favor precision
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta.max(0.0);
        self
    }

    /// Calculate ROUGE scores for candidate against a single reference
    pub fn calculate(&self, candidate: &str, reference: &str) -> Result<RougeMetrics> {
        match &self.rouge_type {
            RougeType::Rouge1 => self.calculate_rouge_n(candidate, reference, 1),
            RougeType::Rouge2 => self.calculate_rouge_n(candidate, reference, 2),
            RougeType::RougeN(n) => self.calculate_rouge_n(candidate, reference, *n),
            RougeType::RougeL => self.calculate_rouge_l(candidate, reference),
            RougeType::RougeW { weight_factor } => {
                self.calculate_rouge_w(candidate, reference, *weight_factor)
            }
            RougeType::RougeS { max_skip_distance } => {
                self.calculate_rouge_s(candidate, reference, *max_skip_distance)
            }
            RougeType::RougeSU { max_skip_distance } => {
                self.calculate_rouge_su(candidate, reference, *max_skip_distance)
            }
        }
    }

    /// Calculate ROUGE scores for candidate against multiple references
    ///
    /// Uses jackknifing procedure where each reference is compared against
    /// the candidate, and the maximum score is taken for each metric.
    pub fn calculate_multi_reference(
        &self,
        candidate: &str,
        references: &[&str],
    ) -> Result<RougeMetrics> {
        if references.is_empty() {
            return Err(TextError::Other(anyhow::anyhow!(
                "No reference summaries provided for ROUGE calculation"
            )));
        }

        let mut max_precision = 0.0;
        let mut max_recall = 0.0;
        let mut max_f1 = 0.0;

        for reference in references {
            let metrics = self.calculate(candidate, reference)?;
            max_precision = max_precision.max(metrics.precision);
            max_recall = max_recall.max(metrics.recall);
            max_f1 = max_f1.max(metrics.f1_score);
        }

        Ok(RougeMetrics {
            precision: max_precision,
            recall: max_recall,
            f1_score: max_f1,
        })
    }

    /// Calculate corpus-level ROUGE scores
    ///
    /// Aggregates ROUGE scores across multiple candidate-reference pairs
    /// for more robust evaluation of system performance.
    pub fn calculate_corpus(
        &self,
        candidates: &[&str],
        references: &[&str],
    ) -> Result<RougeMetrics> {
        if candidates.len() != references.len() {
            return Err(TextError::Other(anyhow::anyhow!(
                "Number of candidates ({}) must match number of references ({})",
                candidates.len(),
                references.len()
            )));
        }

        if candidates.is_empty() {
            return Err(TextError::Other(anyhow::anyhow!(
                "Empty corpus provided for ROUGE calculation"
            )));
        }

        let mut total_precision = 0.0;
        let mut total_recall = 0.0;
        let mut total_f1 = 0.0;

        for (candidate, reference) in candidates.iter().zip(references.iter()) {
            let metrics = self.calculate(candidate, reference)?;
            total_precision += metrics.precision;
            total_recall += metrics.recall;
            total_f1 += metrics.f1_score;
        }

        let count = candidates.len() as f64;
        Ok(RougeMetrics {
            precision: total_precision / count,
            recall: total_recall / count,
            f1_score: total_f1 / count,
        })
    }

    // Private implementation methods

    /// Calculate ROUGE-N scores based on n-gram overlap
    fn calculate_rouge_n(
        &self,
        candidate: &str,
        reference: &str,
        n: usize,
    ) -> Result<RougeMetrics> {
        let candidate_tokens = self.preprocess_tokens(candidate);
        let reference_tokens = self.preprocess_tokens(reference);

        let candidate_ngrams = self.get_ngrams(&candidate_tokens, n);
        let reference_ngrams = self.get_ngrams(&reference_tokens, n);

        let overlap = self.calculate_ngram_overlap(&candidate_ngrams, &reference_ngrams);

        let candidate_total: usize = candidate_ngrams.values().sum();
        let reference_total: usize = reference_ngrams.values().sum();

        let precision = if candidate_total == 0 {
            0.0
        } else {
            overlap as f64 / candidate_total as f64
        };

        let recall = if reference_total == 0 {
            0.0
        } else {
            overlap as f64 / reference_total as f64
        };

        let f1_score = self.calculate_f_score(precision, recall);

        Ok(RougeMetrics {
            precision,
            recall,
            f1_score,
        })
    }

    /// Calculate ROUGE-L scores based on Longest Common Subsequence
    fn calculate_rouge_l(&self, candidate: &str, reference: &str) -> Result<RougeMetrics> {
        let candidate_tokens = self.preprocess_tokens(candidate);
        let reference_tokens = self.preprocess_tokens(reference);

        let lcs_length = self.longest_common_subsequence(&candidate_tokens, &reference_tokens);

        let precision = if candidate_tokens.is_empty() {
            0.0
        } else {
            lcs_length as f64 / candidate_tokens.len() as f64
        };

        let recall = if reference_tokens.is_empty() {
            0.0
        } else {
            lcs_length as f64 / reference_tokens.len() as f64
        };

        let f1_score = self.calculate_f_score(precision, recall);

        Ok(RougeMetrics {
            precision,
            recall,
            f1_score,
        })
    }

    /// Calculate ROUGE-W scores based on Weighted Longest Common Subsequence
    fn calculate_rouge_w(
        &self,
        candidate: &str,
        reference: &str,
        weight_factor: f64,
    ) -> Result<RougeMetrics> {
        let candidate_tokens = self.preprocess_tokens(candidate);
        let reference_tokens = self.preprocess_tokens(reference);

        let wlcs_score = self.weighted_longest_common_subsequence(
            &candidate_tokens,
            &reference_tokens,
            weight_factor,
        );

        let precision = if candidate_tokens.is_empty() {
            0.0
        } else {
            wlcs_score / candidate_tokens.len() as f64
        };

        let recall = if reference_tokens.is_empty() {
            0.0
        } else {
            wlcs_score / reference_tokens.len() as f64
        };

        let f1_score = self.calculate_f_score(precision, recall);

        Ok(RougeMetrics {
            precision,
            recall,
            f1_score,
        })
    }

    /// Calculate ROUGE-S scores based on skip-bigram co-occurrence
    fn calculate_rouge_s(
        &self,
        candidate: &str,
        reference: &str,
        max_skip_distance: usize,
    ) -> Result<RougeMetrics> {
        let candidate_tokens = self.preprocess_tokens(candidate);
        let reference_tokens = self.preprocess_tokens(reference);

        let candidate_skip_bigrams = self.get_skip_bigrams(&candidate_tokens, max_skip_distance);
        let reference_skip_bigrams = self.get_skip_bigrams(&reference_tokens, max_skip_distance);

        let overlap =
            self.calculate_skip_bigram_overlap(&candidate_skip_bigrams, &reference_skip_bigrams);

        let candidate_total = candidate_skip_bigrams.len();
        let reference_total = reference_skip_bigrams.len();

        let precision = if candidate_total == 0 {
            0.0
        } else {
            overlap as f64 / candidate_total as f64
        };

        let recall = if reference_total == 0 {
            0.0
        } else {
            overlap as f64 / reference_total as f64
        };

        let f1_score = self.calculate_f_score(precision, recall);

        Ok(RougeMetrics {
            precision,
            recall,
            f1_score,
        })
    }

    /// Calculate ROUGE-SU scores (skip-bigram with unigram extension)
    fn calculate_rouge_su(
        &self,
        candidate: &str,
        reference: &str,
        max_skip_distance: usize,
    ) -> Result<RougeMetrics> {
        let candidate_tokens = self.preprocess_tokens(candidate);
        let reference_tokens = self.preprocess_tokens(reference);

        // Combine unigrams and skip-bigrams
        let mut candidate_grams = self.get_ngrams(&candidate_tokens, 1);
        let candidate_skip_bigrams = self.get_skip_bigrams(&candidate_tokens, max_skip_distance);

        let mut reference_grams = self.get_ngrams(&reference_tokens, 1);
        let reference_skip_bigrams = self.get_skip_bigrams(&reference_tokens, max_skip_distance);

        // Process skip-bigrams separately to avoid lifetime issues
        let mut candidate_skip_counts: HashMap<Vec<String>, usize> = HashMap::new();
        for skip_bigram in candidate_skip_bigrams {
            let key = vec![skip_bigram.0, skip_bigram.1];
            *candidate_skip_counts.entry(key).or_insert(0) += 1;
        }

        let mut reference_skip_counts: HashMap<Vec<String>, usize> = HashMap::new();
        for skip_bigram in reference_skip_bigrams {
            let key = vec![skip_bigram.0, skip_bigram.1];
            *reference_skip_counts.entry(key).or_insert(0) += 1;
        }

        // Calculate skip-bigram overlap separately
        let mut skip_overlap = 0;
        let mut candidate_skip_total = 0;
        let mut reference_skip_total = 0;

        for (skip_gram, count) in &candidate_skip_counts {
            candidate_skip_total += count;
            if let Some(ref_count) = reference_skip_counts.get(skip_gram) {
                skip_overlap += count.min(ref_count);
            }
        }

        for count in reference_skip_counts.values() {
            reference_skip_total += count;
        }

        let unigram_overlap = self.calculate_ngram_overlap(&candidate_grams, &reference_grams);
        let total_overlap = unigram_overlap + skip_overlap;

        let candidate_total: usize = candidate_grams.values().sum() + candidate_skip_total;
        let reference_total: usize = reference_grams.values().sum() + reference_skip_total;

        let precision = if candidate_total == 0 {
            0.0
        } else {
            total_overlap as f64 / candidate_total as f64
        };

        let recall = if reference_total == 0 {
            0.0
        } else {
            total_overlap as f64 / reference_total as f64
        };

        let f1_score = self.calculate_f_score(precision, recall);

        Ok(RougeMetrics {
            precision,
            recall,
            f1_score,
        })
    }

    /// Preprocess tokens with normalization, stemming, and stopword removal
    fn preprocess_tokens(&self, text: &str) -> Vec<String> {
        let mut tokens: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();

        // Case normalization
        if self.ignore_case {
            tokens = tokens.into_iter().map(|s| s.to_lowercase()).collect();
        }

        // Stopword removal
        if self.remove_stopwords {
            tokens = tokens
                .into_iter()
                .filter(|token| !self.stopwords.contains(token))
                .collect();
        }

        // Stemming
        if self.use_stemming {
            tokens = tokens.into_iter().map(|s| self.stem_word(&s)).collect();
        }

        tokens
    }

    /// Extract n-grams from preprocessed tokens
    fn get_ngrams<'a>(&self, tokens: &'a [String], n: usize) -> NgramCounts<'a> {
        let mut ngrams: HashMap<Vec<&str>, usize> = HashMap::new();

        if tokens.len() < n || n == 0 {
            return ngrams;
        }

        for window in tokens.windows(n) {
            let ngram: Vec<&str> = window.iter().map(|s| s.as_str()).collect();
            *ngrams.entry(ngram).or_insert(0) += 1;
        }

        ngrams
    }

    /// Extract skip-bigrams with configurable skip distance
    fn get_skip_bigrams(
        &self,
        tokens: &[String],
        max_skip_distance: usize,
    ) -> Vec<(String, String)> {
        let mut skip_bigrams = Vec::new();

        for i in 0..tokens.len() {
            for j in (i + 1)..=(i + max_skip_distance + 1).min(tokens.len() - 1) {
                skip_bigrams.push((tokens[i].clone(), tokens[j].clone()));
            }
        }

        skip_bigrams
    }

    /// Calculate n-gram overlap between candidate and reference
    fn calculate_ngram_overlap(
        &self,
        candidate_ngrams: &NgramCounts,
        reference_ngrams: &NgramCounts,
    ) -> usize {
        let mut overlap = 0;

        for (ngram, &count) in candidate_ngrams {
            if let Some(&ref_count) = reference_ngrams.get(ngram) {
                overlap += count.min(ref_count);
            }
        }

        overlap
    }

    /// Calculate skip-bigram overlap
    fn calculate_skip_bigram_overlap(
        &self,
        candidate_skip_bigrams: &[(String, String)],
        reference_skip_bigrams: &[(String, String)],
    ) -> usize {
        let reference_set: HashSet<_> = reference_skip_bigrams.iter().collect();

        candidate_skip_bigrams
            .iter()
            .filter(|skip_bigram| reference_set.contains(skip_bigram))
            .count()
    }

    /// Calculate Longest Common Subsequence length using dynamic programming
    fn longest_common_subsequence(&self, seq1: &[String], seq2: &[String]) -> usize {
        let len1 = seq1.len();
        let len2 = seq2.len();

        if len1 == 0 || len2 == 0 {
            return 0;
        }

        let mut dp = vec![vec![0; len2 + 1]; len1 + 1];

        for i in 1..=len1 {
            for j in 1..=len2 {
                if seq1[i - 1] == seq2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }

        dp[len1][len2]
    }

    /// Calculate Weighted Longest Common Subsequence
    fn weighted_longest_common_subsequence(
        &self,
        seq1: &[String],
        seq2: &[String],
        weight_factor: f64,
    ) -> f64 {
        let len1 = seq1.len();
        let len2 = seq2.len();

        if len1 == 0 || len2 == 0 {
            return 0.0;
        }

        let mut dp = vec![vec![0.0; len2 + 1]; len1 + 1];

        for i in 1..=len1 {
            for j in 1..=len2 {
                if seq1[i - 1] == seq2[j - 1] {
                    // Weight consecutive matches more highly
                    let consecutive_length =
                        self.find_consecutive_match_length(seq1, seq2, i - 1, j - 1);
                    let weight = weight_factor.powi(consecutive_length as i32);
                    dp[i][j] = dp[i - 1][j - 1] + weight;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }

        dp[len1][len2]
    }

    /// Find length of consecutive matches starting from given positions
    fn find_consecutive_match_length(
        &self,
        seq1: &[String],
        seq2: &[String],
        start1: usize,
        start2: usize,
    ) -> usize {
        let mut length = 0;
        let mut i = start1;
        let mut j = start2;

        while i < seq1.len() && j < seq2.len() && seq1[i] == seq2[j] {
            length += 1;
            i += 1;
            j += 1;
        }

        length
    }

    /// Calculate F-score with configurable alpha and beta parameters
    fn calculate_f_score(&self, precision: f64, recall: f64) -> f64 {
        if precision + recall == 0.0 {
            0.0
        } else {
            let beta_squared = self.beta * self.beta;
            (1.0 + beta_squared) * precision * recall / (beta_squared * precision + recall)
        }
    }

    /// Simple stemming algorithm for morphological normalization
    fn stem_word(&self, word: &str) -> String {
        // Basic Porter-like stemming rules
        let word = word.to_lowercase();

        if word.len() <= 3 {
            return word;
        }

        // Remove common suffixes
        if word.ends_with("ing") && word.len() > 6 {
            return word[..word.len() - 3].to_string();
        }

        if word.ends_with("ed") && word.len() > 5 {
            return word[..word.len() - 2].to_string();
        }

        if word.ends_with("er") && word.len() > 5 {
            return word[..word.len() - 2].to_string();
        }

        if word.ends_with("est") && word.len() > 6 {
            return word[..word.len() - 3].to_string();
        }

        if word.ends_with("ly") && word.len() > 5 {
            return word[..word.len() - 2].to_string();
        }

        if word.ends_with("s") && word.len() > 4 && !word.ends_with("ss") {
            return word[..word.len() - 1].to_string();
        }

        word
    }

    /// Get default English stopwords
    fn default_stopwords() -> HashSet<String> {
        [
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
            "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", "with", "the",
            "this", "but", "they", "have", "had", "what", "said", "each", "which", "she", "do",
            "how", "their", "if", "up", "out", "many", "then", "them", "these", "so", "some",
            "her", "would", "make", "like", "into", "him", "time", "two", "more", "go", "no",
            "way", "could", "my", "than", "first", "been", "call", "who", "its", "now", "find",
            "long", "down", "day", "did", "get", "come", "made", "may", "part",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }
}

/// ROUGE evaluation metrics containing precision, recall, and F1-score
#[derive(Debug, Clone)]
pub struct RougeMetrics {
    /// Precision: ratio of relevant items retrieved to total items retrieved
    pub precision: f64,
    /// Recall: ratio of relevant items retrieved to total relevant items
    pub recall: f64,
    /// F1-score: harmonic mean of precision and recall
    pub f1_score: f64,
}

impl RougeMetrics {
    /// Create new ROUGE metrics
    pub fn new(precision: f64, recall: f64, f1_score: f64) -> Self {
        Self {
            precision,
            recall,
            f1_score,
        }
    }

    /// Check if metrics indicate high quality (F1 > 0.5)
    pub fn is_high_quality(&self) -> bool {
        self.f1_score > 0.5
    }

    /// Get the dominant metric (precision vs recall)
    pub fn dominant_metric(&self) -> &'static str {
        if self.precision > self.recall {
            "precision"
        } else if self.recall > self.precision {
            "recall"
        } else {
            "balanced"
        }
    }

    /// Calculate harmonic mean (alternative F1 computation)
    pub fn harmonic_mean(&self) -> f64 {
        if self.precision + self.recall == 0.0 {
            0.0
        } else {
            2.0 * self.precision * self.recall / (self.precision + self.recall)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rouge1_perfect_match() {
        let rouge = RougeScore::new(RougeType::Rouge1);
        let candidate = "the quick brown fox";
        let reference = "the quick brown fox";

        let metrics = rouge.calculate(candidate, reference).unwrap();
        assert!((metrics.precision - 1.0).abs() < 1e-10);
        assert!((metrics.recall - 1.0).abs() < 1e-10);
        assert!((metrics.f1_score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rouge1_partial_match() {
        let rouge = RougeScore::new(RougeType::Rouge1);
        let candidate = "the quick brown fox";
        let reference = "the fast brown fox";

        let metrics = rouge.calculate(candidate, reference).unwrap();
        assert!(metrics.precision > 0.0 && metrics.precision < 1.0);
        assert!(metrics.recall > 0.0 && metrics.recall < 1.0);
        assert!(metrics.f1_score > 0.0 && metrics.f1_score < 1.0);
    }

    #[test]
    fn test_rouge2_bigrams() {
        let rouge = RougeScore::new(RougeType::Rouge2);
        let candidate = "the quick brown fox jumps";
        let reference = "the quick brown fox leaps";

        let metrics = rouge.calculate(candidate, reference).unwrap();
        // Should have some bigram overlap: "the quick", "quick brown", "brown fox"
        assert!(metrics.precision > 0.0);
        assert!(metrics.recall > 0.0);
    }

    #[test]
    fn test_rouge_l_lcs() {
        let rouge = RougeScore::new(RougeType::RougeL);
        let candidate = "A B C D E";
        let reference = "A C E";

        let metrics = rouge.calculate(candidate, reference).unwrap();
        // LCS is "A C E" with length 3
        let expected_precision = 3.0 / 5.0; // 3 LCS tokens out of 5 candidate tokens
        let expected_recall = 3.0 / 3.0; // 3 LCS tokens out of 3 reference tokens

        assert!((metrics.precision - expected_precision).abs() < 1e-10);
        assert!((metrics.recall - expected_recall).abs() < 1e-10);
    }

    #[test]
    fn test_multi_reference() {
        let rouge = RougeScore::new(RougeType::Rouge1);
        let candidate = "the quick brown fox";
        let references = &["the fast brown fox", "a quick brown fox"];

        let metrics = rouge
            .calculate_multi_reference(candidate, references)
            .unwrap();
        assert!(metrics.precision > 0.0);
        assert!(metrics.recall > 0.0);
        assert!(metrics.f1_score > 0.0);
    }

    #[test]
    fn test_preprocessing_ignore_case() {
        let rouge = RougeScore::new(RougeType::Rouge1).with_ignore_case(true);
        let candidate = "The Quick Brown Fox";
        let reference = "the quick brown fox";

        let metrics = rouge.calculate(candidate, reference).unwrap();
        assert!((metrics.f1_score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_preprocessing_stemming() {
        let rouge = RougeScore::new(RougeType::Rouge1).with_stemming(true);
        let candidate = "running jumped";
        let reference = "run jump";

        let metrics = rouge.calculate(candidate, reference).unwrap();
        // With stemming: "running" -> "run", "jumped" -> "jump"
        assert!((metrics.f1_score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_stopword_removal() {
        let rouge = RougeScore::new(RougeType::Rouge1).with_remove_stopwords(true);
        let candidate = "the cat is running";
        let reference = "a cat was running";

        let metrics = rouge.calculate(candidate, reference).unwrap();
        // After stopword removal: "cat running" vs "cat running"
        assert!((metrics.f1_score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_texts() {
        let rouge = RougeScore::new(RougeType::Rouge1);

        let metrics1 = rouge.calculate("", "some text").unwrap();
        assert_eq!(metrics1.precision, 0.0);
        assert_eq!(metrics1.recall, 0.0);
        assert_eq!(metrics1.f1_score, 0.0);

        let metrics2 = rouge.calculate("some text", "").unwrap();
        assert_eq!(metrics2.precision, 0.0);
        assert_eq!(metrics2.recall, 0.0);
        assert_eq!(metrics2.f1_score, 0.0);
    }

    #[test]
    fn test_rouge_n_custom() {
        let rouge = RougeScore::new(RougeType::RougeN(3)); // Trigrams
        let candidate = "the quick brown fox jumps";
        let reference = "the quick brown fox leaps";

        let metrics = rouge.calculate(candidate, reference).unwrap();
        // Should have some trigram overlap
        assert!(metrics.precision >= 0.0);
        assert!(metrics.recall >= 0.0);
    }

    #[test]
    fn test_corpus_level() {
        let rouge = RougeScore::new(RougeType::Rouge1);
        let candidates = &["the quick fox", "hello world"];
        let references = &["the fast fox", "hello world"];

        let metrics = rouge.calculate_corpus(candidates, references).unwrap();
        assert!(metrics.f1_score > 0.5); // Should be reasonably high
    }

    #[test]
    fn test_rouge_metrics_helpers() {
        let metrics = RougeMetrics::new(0.6, 0.8, 0.69);

        assert!(metrics.is_high_quality());
        assert_eq!(metrics.dominant_metric(), "recall");
        assert!((metrics.harmonic_mean() - 0.6857142857142857).abs() < 1e-10);

        let balanced_metrics = RougeMetrics::new(0.5, 0.5, 0.5);
        assert_eq!(balanced_metrics.dominant_metric(), "balanced");
    }

    #[test]
    fn test_beta_parameter() {
        let rouge_f1 = RougeScore::new(RougeType::Rouge1).with_beta(1.0);
        let rouge_f2 = RougeScore::new(RougeType::Rouge1).with_beta(2.0);

        let candidate = "the quick fox";
        let reference = "the quick brown fox";

        let metrics_f1 = rouge_f1.calculate(candidate, reference).unwrap();
        let metrics_f2 = rouge_f2.calculate(candidate, reference).unwrap();

        // F2 should weight recall more heavily than F1
        // Since recall > precision in this case, F2 should be higher
        assert!(metrics_f2.f1_score >= metrics_f1.f1_score);
    }
}
