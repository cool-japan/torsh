//! BLEU Score Implementation
//!
//! This module provides comprehensive BLEU (Bilingual Evaluation Understudy) score computation
//! for machine translation evaluation. BLEU measures the quality of machine-translated text by
//! comparing it to human reference translations using n-gram precision and brevity penalty.
//!
//! # Features
//!
//! - Standard BLEU-4 scoring with configurable n-gram order
//! - Smoothing techniques for handling zero precision
//! - Corpus-level BLEU computation
//! - Individual sentence BLEU scoring
//! - Comprehensive statistical analysis
//!
//! # Example
//!
//! ```rust
//! use torsh_text::metrics::bleu::BleuScore;
//!
//! let bleu = BleuScore::default();
//! let candidate = "the quick brown fox";
//! let references = &["the fast brown fox", "a quick brown fox"];
//!
//! let score = bleu.calculate(candidate, references)?;
//! println!("BLEU score: {:.3}", score);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{Result, TextError};
use std::collections::HashMap;

/// Type alias for n-gram counts mapping n-grams to their frequencies
type NgramCounts<'a> = HashMap<Vec<&'a str>, usize>;

/// BLEU score calculator with configurable parameters
///
/// BLEU (Bilingual Evaluation Understudy) is a metric for evaluating machine translation
/// quality by measuring n-gram precision between candidate and reference translations.
#[derive(Debug, Clone)]
pub struct BleuScore {
    /// Enable smoothing for zero precision scores
    smoothing: bool,
    /// Maximum n-gram order (typically 4 for BLEU-4)
    max_n: usize,
    /// Minimum sentence length for meaningful evaluation
    min_length: usize,
    /// Use geometric mean for precision aggregation
    use_geometric_mean: bool,
}

impl Default for BleuScore {
    fn default() -> Self {
        Self {
            smoothing: true,
            max_n: 4,
            min_length: 1,
            use_geometric_mean: true,
        }
    }
}

impl BleuScore {
    /// Create a new BLEU scorer with default parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure smoothing for handling zero precision scores
    ///
    /// When enabled, adds a small epsilon to zero precision scores to prevent
    /// the geometric mean from becoming zero.
    pub fn with_smoothing(mut self, smoothing: bool) -> Self {
        self.smoothing = smoothing;
        self
    }

    /// Set the maximum n-gram order
    ///
    /// Standard BLEU uses n=4, but this can be adjusted based on the application.
    /// Higher values capture longer phrase matches but may be more sparse.
    pub fn with_max_n(mut self, max_n: usize) -> Self {
        self.max_n = max_n.max(1); // Ensure at least unigrams
        self
    }

    /// Set minimum sentence length for evaluation
    ///
    /// Sentences shorter than this threshold will receive special handling
    pub fn with_min_length(mut self, min_length: usize) -> Self {
        self.min_length = min_length;
        self
    }

    /// Configure precision aggregation method
    ///
    /// When true (default), uses geometric mean. When false, uses arithmetic mean.
    pub fn with_geometric_mean(mut self, use_geometric_mean: bool) -> Self {
        self.use_geometric_mean = use_geometric_mean;
        self
    }

    /// Calculate BLEU score for a single candidate against multiple references
    ///
    /// # Arguments
    ///
    /// * `candidate` - The machine translation to evaluate
    /// * `references` - Array of human reference translations
    ///
    /// # Returns
    ///
    /// BLEU score between 0.0 and 1.0, where 1.0 indicates perfect match
    pub fn calculate(&self, candidate: &str, references: &[&str]) -> Result<f64> {
        if references.is_empty() {
            return Err(TextError::Other(anyhow::anyhow!(
                "No reference sentences provided for BLEU calculation"
            )));
        }

        let candidate_tokens: Vec<&str> = self.tokenize(candidate);
        let reference_tokens: Vec<Vec<&str>> =
            references.iter().map(|r| self.tokenize(r)).collect();

        if candidate_tokens.is_empty() {
            return Ok(0.0);
        }

        // Calculate precision scores for each n-gram order
        let mut precision_scores = Vec::new();

        for n in 1..=self.max_n {
            let precision =
                self.calculate_ngram_precision(&candidate_tokens, &reference_tokens, n)?;
            precision_scores.push(precision);
        }

        // Calculate brevity penalty
        let brevity_penalty = self.calculate_brevity_penalty(&candidate_tokens, &reference_tokens);

        // Aggregate precision scores
        let aggregated_precision = if self.use_geometric_mean {
            self.geometric_mean(&precision_scores)
        } else {
            self.arithmetic_mean(&precision_scores)
        };

        Ok(brevity_penalty * aggregated_precision)
    }

    /// Calculate corpus-level BLEU score
    ///
    /// Computes BLEU score across multiple candidate-reference pairs,
    /// which is more robust than averaging individual sentence scores.
    pub fn calculate_corpus(&self, candidates: &[&str], references: &[Vec<&str>]) -> Result<f64> {
        if candidates.len() != references.len() {
            return Err(TextError::Other(anyhow::anyhow!(
                "Number of candidates ({}) must match number of reference sets ({})",
                candidates.len(),
                references.len()
            )));
        }

        if candidates.is_empty() {
            return Err(TextError::Other(anyhow::anyhow!(
                "Empty corpus provided for BLEU calculation"
            )));
        }

        // Tokenize all candidates and references
        let candidate_tokens: Vec<Vec<&str>> =
            candidates.iter().map(|c| self.tokenize(c)).collect();

        let reference_tokens: Vec<Vec<Vec<&str>>> = references
            .iter()
            .map(|refs| refs.iter().map(|r| self.tokenize(r)).collect())
            .collect();

        // Calculate corpus-level precision for each n-gram order
        let mut precision_scores = Vec::new();

        for n in 1..=self.max_n {
            let precision =
                self.calculate_corpus_ngram_precision(&candidate_tokens, &reference_tokens, n)?;
            precision_scores.push(precision);
        }

        // Calculate corpus-level brevity penalty
        let brevity_penalty =
            self.calculate_corpus_brevity_penalty(&candidate_tokens, &reference_tokens);

        // Aggregate precision scores
        let aggregated_precision = if self.use_geometric_mean {
            self.geometric_mean(&precision_scores)
        } else {
            self.arithmetic_mean(&precision_scores)
        };

        Ok(brevity_penalty * aggregated_precision)
    }

    /// Get detailed BLEU metrics including per-order precision scores
    pub fn calculate_detailed(&self, candidate: &str, references: &[&str]) -> Result<BleuMetrics> {
        if references.is_empty() {
            return Err(TextError::Other(anyhow::anyhow!(
                "No reference sentences provided for detailed BLEU calculation"
            )));
        }

        let candidate_tokens: Vec<&str> = self.tokenize(candidate);
        let reference_tokens: Vec<Vec<&str>> =
            references.iter().map(|r| self.tokenize(r)).collect();

        let mut precision_scores = Vec::new();
        let mut ngram_matches = Vec::new();
        let mut ngram_totals = Vec::new();

        for n in 1..=self.max_n {
            let (precision, matches, total) =
                self.calculate_ngram_precision_detailed(&candidate_tokens, &reference_tokens, n)?;
            precision_scores.push(precision);
            ngram_matches.push(matches);
            ngram_totals.push(total);
        }

        let brevity_penalty = self.calculate_brevity_penalty(&candidate_tokens, &reference_tokens);

        let aggregated_precision = if self.use_geometric_mean {
            self.geometric_mean(&precision_scores)
        } else {
            self.arithmetic_mean(&precision_scores)
        };

        let bleu_score = brevity_penalty * aggregated_precision;

        Ok(BleuMetrics {
            bleu_score,
            precision_scores,
            brevity_penalty,
            ngram_matches,
            ngram_totals,
            candidate_length: candidate_tokens.len(),
            reference_length: self.effective_reference_length(&reference_tokens),
        })
    }

    // Private implementation methods

    /// Tokenize text into whitespace-separated tokens
    fn tokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        text.split_whitespace().collect()
    }

    /// Calculate n-gram precision for a single sentence
    fn calculate_ngram_precision(
        &self,
        candidate_tokens: &[&str],
        reference_tokens: &[Vec<&str>],
        n: usize,
    ) -> Result<f64> {
        let (precision, _, _) =
            self.calculate_ngram_precision_detailed(candidate_tokens, reference_tokens, n)?;
        Ok(precision)
    }

    /// Calculate n-gram precision with detailed match information
    fn calculate_ngram_precision_detailed(
        &self,
        candidate_tokens: &[&str],
        reference_tokens: &[Vec<&str>],
        n: usize,
    ) -> Result<(f64, usize, usize)> {
        let candidate_ngrams = self.get_ngrams(candidate_tokens, n);
        let mut reference_ngrams_counts: HashMap<Vec<&str>, usize> = HashMap::new();

        // Build reference n-gram counts (taking maximum count across references)
        for ref_tokens in reference_tokens {
            let ref_ngrams = self.get_ngrams(ref_tokens, n);
            for (ngram, count) in ref_ngrams {
                let entry = reference_ngrams_counts.entry(ngram).or_insert(0);
                *entry = (*entry).max(count);
            }
        }

        let mut matched = 0;
        let mut total = 0;

        // Count matches between candidate and reference n-grams
        for (ngram, count) in candidate_ngrams {
            total += count;
            if let Some(&ref_count) = reference_ngrams_counts.get(&ngram) {
                matched += count.min(ref_count);
            }
        }

        let precision = if total == 0 {
            0.0
        } else if self.smoothing && matched == 0 && total > 0 {
            // Add-one smoothing for zero precision
            1.0 / (total + 1) as f64
        } else {
            matched as f64 / total as f64
        };

        Ok((precision, matched, total))
    }

    /// Calculate corpus-level n-gram precision
    fn calculate_corpus_ngram_precision(
        &self,
        candidate_tokens: &[Vec<&str>],
        reference_tokens: &[Vec<Vec<&str>>],
        n: usize,
    ) -> Result<f64> {
        let mut total_matched = 0;
        let mut total_count = 0;

        for (candidate, references) in candidate_tokens.iter().zip(reference_tokens.iter()) {
            let (_, matched, count) =
                self.calculate_ngram_precision_detailed(candidate, references, n)?;
            total_matched += matched;
            total_count += count;
        }

        if total_count == 0 {
            Ok(0.0)
        } else if self.smoothing && total_matched == 0 && total_count > 0 {
            Ok(1.0 / (total_count + 1) as f64)
        } else {
            Ok(total_matched as f64 / total_count as f64)
        }
    }

    /// Extract n-grams from tokenized text
    fn get_ngrams<'a>(&self, tokens: &[&'a str], n: usize) -> NgramCounts<'a> {
        let mut ngrams: HashMap<Vec<&str>, usize> = HashMap::new();

        if tokens.len() < n || n == 0 {
            return ngrams;
        }

        for window in tokens.windows(n) {
            let ngram = window.to_vec();
            *ngrams.entry(ngram).or_insert(0) += 1;
        }

        ngrams
    }

    /// Calculate brevity penalty to penalize short translations
    fn calculate_brevity_penalty(
        &self,
        candidate_tokens: &[&str],
        reference_tokens: &[Vec<&str>],
    ) -> f64 {
        let candidate_length = candidate_tokens.len();
        let reference_length = self.effective_reference_length(reference_tokens);

        if candidate_length == 0 {
            return 0.0;
        }

        if candidate_length >= reference_length {
            1.0
        } else {
            // BP = exp(1 - r/c) where r is reference length, c is candidate length
            (1.0 - reference_length as f64 / candidate_length as f64).exp()
        }
    }

    /// Calculate corpus-level brevity penalty
    fn calculate_corpus_brevity_penalty(
        &self,
        candidate_tokens: &[Vec<&str>],
        reference_tokens: &[Vec<Vec<&str>>],
    ) -> f64 {
        let total_candidate_length: usize =
            candidate_tokens.iter().map(|tokens| tokens.len()).sum();
        let total_reference_length: usize = reference_tokens
            .iter()
            .map(|refs| self.effective_reference_length(refs))
            .sum();

        if total_candidate_length == 0 {
            return 0.0;
        }

        if total_candidate_length >= total_reference_length {
            1.0
        } else {
            (1.0 - total_reference_length as f64 / total_candidate_length as f64).exp()
        }
    }

    /// Find the reference length closest to the candidate length
    fn effective_reference_length(&self, reference_tokens: &[Vec<&str>]) -> usize {
        if reference_tokens.is_empty() {
            return 0;
        }

        // Return the length of the reference that is closest in length to some hypothetical candidate
        // For simplicity, we use the shortest reference (most conservative)
        reference_tokens
            .iter()
            .map(|tokens| tokens.len())
            .min()
            .unwrap_or(0)
    }

    /// Calculate geometric mean of precision scores
    fn geometric_mean(&self, scores: &[f64]) -> f64 {
        if scores.is_empty() {
            return 0.0;
        }

        // Handle zero precision scores
        if scores.iter().any(|&score| score == 0.0) {
            if self.smoothing {
                // Use add-epsilon smoothing
                let epsilon = 1e-7;
                let smoothed_scores: Vec<f64> = scores
                    .iter()
                    .map(|&score| if score == 0.0 { epsilon } else { score })
                    .collect();
                let log_sum: f64 = smoothed_scores.iter().map(|score| score.ln()).sum();
                (log_sum / smoothed_scores.len() as f64).exp()
            } else {
                0.0
            }
        } else {
            let log_sum: f64 = scores.iter().map(|score| score.ln()).sum();
            (log_sum / scores.len() as f64).exp()
        }
    }

    /// Calculate arithmetic mean of precision scores
    fn arithmetic_mean(&self, scores: &[f64]) -> f64 {
        if scores.is_empty() {
            return 0.0;
        }

        scores.iter().sum::<f64>() / scores.len() as f64
    }
}

/// Detailed BLEU metrics including component scores
#[derive(Debug, Clone)]
pub struct BleuMetrics {
    /// Overall BLEU score
    pub bleu_score: f64,
    /// Precision scores for each n-gram order (1-gram, 2-gram, etc.)
    pub precision_scores: Vec<f64>,
    /// Brevity penalty factor
    pub brevity_penalty: f64,
    /// Number of n-gram matches for each order
    pub ngram_matches: Vec<usize>,
    /// Total number of n-grams for each order
    pub ngram_totals: Vec<usize>,
    /// Length of candidate translation
    pub candidate_length: usize,
    /// Effective reference translation length
    pub reference_length: usize,
}

impl BleuMetrics {
    /// Get precision score for a specific n-gram order (1-indexed)
    pub fn precision(&self, n: usize) -> Option<f64> {
        if n == 0 || n > self.precision_scores.len() {
            None
        } else {
            Some(self.precision_scores[n - 1])
        }
    }

    /// Get match ratio for a specific n-gram order (1-indexed)
    pub fn match_ratio(&self, n: usize) -> Option<f64> {
        if n == 0 || n > self.ngram_matches.len() || n > self.ngram_totals.len() {
            None
        } else {
            let matches = self.ngram_matches[n - 1];
            let total = self.ngram_totals[n - 1];
            if total == 0 {
                Some(0.0)
            } else {
                Some(matches as f64 / total as f64)
            }
        }
    }

    /// Check if the translation is longer than the reference (no brevity penalty)
    pub fn is_adequate_length(&self) -> bool {
        self.brevity_penalty >= 1.0
    }

    /// Get length ratio (candidate/reference)
    pub fn length_ratio(&self) -> f64 {
        if self.reference_length == 0 {
            if self.candidate_length == 0 {
                1.0
            } else {
                f64::INFINITY
            }
        } else {
            self.candidate_length as f64 / self.reference_length as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_match() {
        let bleu = BleuScore::default();
        let candidate = "the quick brown fox";
        let references = &["the quick brown fox"];

        let score = bleu.calculate(candidate, references).unwrap();
        assert!(
            (score - 1.0).abs() < 1e-10,
            "Perfect match should give score of 1.0, got {}",
            score
        );
    }

    #[test]
    fn test_empty_candidate() {
        let bleu = BleuScore::default();
        let candidate = "";
        let references = &["the quick brown fox"];

        let score = bleu.calculate(candidate, references).unwrap();
        assert_eq!(score, 0.0, "Empty candidate should give score of 0.0");
    }

    #[test]
    fn test_no_references() {
        let bleu = BleuScore::default();
        let candidate = "the quick brown fox";
        let references: &[&str] = &[];

        let result = bleu.calculate(candidate, references);
        assert!(result.is_err(), "No references should return error");
    }

    #[test]
    fn test_partial_match() {
        let bleu = BleuScore::default();
        let candidate = "the quick brown fox";
        let references = &["the fast brown fox"];

        let score = bleu.calculate(candidate, references).unwrap();
        assert!(
            score > 0.0 && score < 1.0,
            "Partial match should give score between 0 and 1, got {}",
            score
        );
    }

    #[test]
    fn test_multiple_references() {
        let bleu = BleuScore::default();
        let candidate = "the quick brown fox";
        let references = &["the fast brown fox", "a quick brown fox"];

        let score = bleu.calculate(candidate, references).unwrap();
        assert!(
            score > 0.0,
            "Multiple references should give positive score, got {}",
            score
        );
    }

    #[test]
    fn test_corpus_level() {
        let bleu = BleuScore::default();
        let candidates = &["the quick brown fox", "hello world"];
        let references = &[
            vec!["the fast brown fox", "a quick brown fox"],
            vec!["hello world", "hi world"],
        ];

        let score = bleu.calculate_corpus(candidates, references).unwrap();
        assert!(
            score > 0.0,
            "Corpus BLEU should give positive score, got {}",
            score
        );
    }

    #[test]
    fn test_detailed_metrics() {
        let bleu = BleuScore::default();
        let candidate = "the quick brown fox";
        let references = &["the fast brown fox"];

        let metrics = bleu.calculate_detailed(candidate, references).unwrap();

        assert!(metrics.bleu_score > 0.0);
        assert_eq!(metrics.precision_scores.len(), 4); // Default max_n = 4
        assert!(metrics.brevity_penalty > 0.0);
        assert_eq!(metrics.candidate_length, 4);
    }

    #[test]
    fn test_smoothing() {
        let bleu_with_smoothing = BleuScore::default().with_smoothing(true);
        let bleu_without_smoothing = BleuScore::default().with_smoothing(false);

        let candidate = "completely different words";
        let references = &["the quick brown fox"];

        let score_with = bleu_with_smoothing
            .calculate(candidate, references)
            .unwrap();
        let score_without = bleu_without_smoothing
            .calculate(candidate, references)
            .unwrap();

        // With no matches, smoothing should give a small positive score
        // while without smoothing should give 0
        assert!(score_with > 0.0);
        assert_eq!(score_without, 0.0);
    }

    #[test]
    fn test_ngram_extraction() {
        let bleu = BleuScore::default();
        let tokens = vec!["the", "quick", "brown", "fox"];

        let unigrams = bleu.get_ngrams(&tokens, 1);
        assert_eq!(unigrams.len(), 4);
        assert_eq!(unigrams[&vec!["the"]], 1);

        let bigrams = bleu.get_ngrams(&tokens, 2);
        assert_eq!(bigrams.len(), 3);
        assert_eq!(bigrams[&vec!["the", "quick"]], 1);

        let trigrams = bleu.get_ngrams(&tokens, 3);
        assert_eq!(trigrams.len(), 2);
        assert_eq!(trigrams[&vec!["the", "quick", "brown"]], 1);
    }

    #[test]
    fn test_brevity_penalty() {
        let bleu = BleuScore::default();

        // Shorter candidate should have brevity penalty < 1
        let short_candidate = vec!["fox"];
        let references = vec![vec!["the", "quick", "brown", "fox"]];
        let bp_short = bleu.calculate_brevity_penalty(&short_candidate, &references);
        assert!(
            bp_short < 1.0,
            "Short candidate should have brevity penalty < 1.0, got {}",
            bp_short
        );

        // Equal length should have brevity penalty = 1
        let equal_candidate = vec!["the", "quick", "brown", "fox"];
        let bp_equal = bleu.calculate_brevity_penalty(&equal_candidate, &references);
        assert!(
            (bp_equal - 1.0).abs() < 1e-10,
            "Equal length should have brevity penalty = 1.0, got {}",
            bp_equal
        );

        // Longer candidate should have brevity penalty = 1
        let long_candidate = vec!["the", "quick", "brown", "fox", "jumps"];
        let bp_long = bleu.calculate_brevity_penalty(&long_candidate, &references);
        assert!(
            (bp_long - 1.0).abs() < 1e-10,
            "Long candidate should have brevity penalty = 1.0, got {}",
            bp_long
        );
    }

    #[test]
    fn test_configuration() {
        let bleu = BleuScore::new()
            .with_max_n(2)
            .with_smoothing(false)
            .with_min_length(2)
            .with_geometric_mean(false);

        let candidate = "quick fox";
        let references = &["the quick brown fox"];

        let metrics = bleu.calculate_detailed(candidate, references).unwrap();
        assert_eq!(metrics.precision_scores.len(), 2); // max_n = 2
    }
}
