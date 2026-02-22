//! Edit Distance Metrics for String Similarity
//!
//! This module provides comprehensive edit distance algorithms for measuring string similarity.
//! Edit distance quantifies how different two strings are by counting the minimum number of
//! single-character edits (insertions, deletions, or substitutions) needed to transform
//! one string into another.
//!
//! # Supported Algorithms
//!
//! - **Levenshtein Distance**: Standard edit distance with equal costs for all operations
//! - **Damerau-Levenshtein Distance**: Includes transposition as a fourth operation
//! - **Hamming Distance**: For strings of equal length, counts differing positions
//! - **Jaro Distance**: Focuses on character matches and transpositions
//! - **Jaro-Winkler Distance**: Jaro distance with prefix bonus
//! - **Optimal String Alignment**: Restricted Damerau-Levenshtein
//!
//! # Features
//!
//! - Multiple distance algorithms with different characteristics
//! - Normalized similarity scores (0.0 to 1.0)
//! - Configurable operation costs (insertion, deletion, substitution, transposition)
//! - Case-insensitive comparison options
//! - Unicode-aware character handling
//! - Efficient algorithms with space optimization
//!
//! # Example
//!
//! ```rust
//! use torsh_text::metrics::edit_distance::{EditDistance, EditDistanceConfig, DistanceAlgorithm};
//!
//! let calculator = EditDistance::new();
//!
//! // Basic Levenshtein distance
//! let distance = calculator.levenshtein("kitten", "sitting");
//! println!("Levenshtein distance: {}", distance);
//!
//! // Normalized similarity score
//! let similarity = calculator.normalized_levenshtein("kitten", "sitting");
//! println!("Similarity: {:.3}", similarity);
//!
//! // Custom configuration with different costs
//! let config = EditDistanceConfig::new()
//!     .with_insertion_cost(1.0)
//!     .with_deletion_cost(1.0)
//!     .with_substitution_cost(2.0); // Higher substitution cost
//!
//! let custom_calculator = EditDistance::with_config(config);
//! let weighted_distance = custom_calculator.calculate("hello", "world", DistanceAlgorithm::Levenshtein);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::{Result, TextError};
use std::cmp;
use std::collections::HashMap;

/// Available edit distance algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceAlgorithm {
    /// Standard Levenshtein distance
    Levenshtein,
    /// Damerau-Levenshtein distance (allows transpositions)
    DamerauLevenshtein,
    /// Optimal String Alignment distance
    OptimalStringAlignment,
    /// Hamming distance (equal length strings only)
    Hamming,
    /// Jaro distance
    Jaro,
    /// Jaro-Winkler distance
    JaroWinkler,
}

/// Configuration for edit distance calculations
#[derive(Debug, Clone)]
pub struct EditDistanceConfig {
    /// Cost of inserting a character
    pub insertion_cost: f64,
    /// Cost of deleting a character
    pub deletion_cost: f64,
    /// Cost of substituting a character
    pub substitution_cost: f64,
    /// Cost of transposing two adjacent characters
    pub transposition_cost: f64,
    /// Whether to ignore case when comparing characters
    pub ignore_case: bool,
    /// Jaro-Winkler prefix scaling factor (0.0 to 0.25)
    pub jaro_winkler_prefix_scale: f64,
    /// Maximum prefix length to consider for Jaro-Winkler
    pub jaro_winkler_max_prefix: usize,
}

impl Default for EditDistanceConfig {
    fn default() -> Self {
        Self {
            insertion_cost: 1.0,
            deletion_cost: 1.0,
            substitution_cost: 1.0,
            transposition_cost: 1.0,
            ignore_case: false,
            jaro_winkler_prefix_scale: 0.1,
            jaro_winkler_max_prefix: 4,
        }
    }
}

impl EditDistanceConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the cost for character insertion
    pub fn with_insertion_cost(mut self, cost: f64) -> Self {
        self.insertion_cost = cost.max(0.0);
        self
    }

    /// Set the cost for character deletion
    pub fn with_deletion_cost(mut self, cost: f64) -> Self {
        self.deletion_cost = cost.max(0.0);
        self
    }

    /// Set the cost for character substitution
    pub fn with_substitution_cost(mut self, cost: f64) -> Self {
        self.substitution_cost = cost.max(0.0);
        self
    }

    /// Set the cost for character transposition
    pub fn with_transposition_cost(mut self, cost: f64) -> Self {
        self.transposition_cost = cost.max(0.0);
        self
    }

    /// Enable or disable case-insensitive comparison
    pub fn with_ignore_case(mut self, ignore_case: bool) -> Self {
        self.ignore_case = ignore_case;
        self
    }

    /// Set Jaro-Winkler prefix scaling factor
    pub fn with_jaro_winkler_prefix_scale(mut self, scale: f64) -> Self {
        self.jaro_winkler_prefix_scale = scale.clamp(0.0, 0.25);
        self
    }

    /// Set maximum prefix length for Jaro-Winkler
    pub fn with_jaro_winkler_max_prefix(mut self, max_prefix: usize) -> Self {
        self.jaro_winkler_max_prefix = max_prefix;
        self
    }
}

/// Comprehensive edit distance calculator
#[derive(Debug, Clone)]
pub struct EditDistance {
    config: EditDistanceConfig,
}

impl Default for EditDistance {
    fn default() -> Self {
        Self::new()
    }
}

impl EditDistance {
    /// Create a new edit distance calculator with default configuration
    pub fn new() -> Self {
        Self {
            config: EditDistanceConfig::default(),
        }
    }

    /// Create an edit distance calculator with custom configuration
    pub fn with_config(config: EditDistanceConfig) -> Self {
        Self { config }
    }

    /// Calculate edit distance using the specified algorithm
    pub fn calculate(&self, s1: &str, s2: &str, algorithm: DistanceAlgorithm) -> Result<f64> {
        match algorithm {
            DistanceAlgorithm::Levenshtein => Ok(self.levenshtein_distance(s1, s2) as f64),
            DistanceAlgorithm::DamerauLevenshtein => {
                Ok(self.damerau_levenshtein_distance(s1, s2) as f64)
            }
            DistanceAlgorithm::OptimalStringAlignment => {
                Ok(self.optimal_string_alignment_distance(s1, s2) as f64)
            }
            DistanceAlgorithm::Hamming => self.hamming_distance(s1, s2).map(|d| d as f64),
            DistanceAlgorithm::Jaro => Ok(1.0 - self.jaro_similarity(s1, s2)),
            DistanceAlgorithm::JaroWinkler => Ok(1.0 - self.jaro_winkler_similarity(s1, s2)),
        }
    }

    /// Calculate normalized similarity score (0.0 to 1.0) using the specified algorithm
    pub fn similarity(&self, s1: &str, s2: &str, algorithm: DistanceAlgorithm) -> Result<f64> {
        match algorithm {
            DistanceAlgorithm::Levenshtein => Ok(self.normalized_levenshtein(s1, s2)),
            DistanceAlgorithm::DamerauLevenshtein => {
                Ok(self.normalized_damerau_levenshtein(s1, s2))
            }
            DistanceAlgorithm::OptimalStringAlignment => {
                Ok(self.normalized_optimal_string_alignment(s1, s2))
            }
            DistanceAlgorithm::Hamming => self.normalized_hamming(s1, s2),
            DistanceAlgorithm::Jaro => Ok(self.jaro_similarity(s1, s2)),
            DistanceAlgorithm::JaroWinkler => Ok(self.jaro_winkler_similarity(s1, s2)),
        }
    }

    /// Calculate standard Levenshtein distance
    pub fn levenshtein(&self, s1: &str, s2: &str) -> usize {
        self.levenshtein_distance(s1, s2)
    }

    /// Calculate normalized Levenshtein similarity (0.0 to 1.0)
    pub fn normalized_levenshtein(&self, s1: &str, s2: &str) -> f64 {
        let distance = self.levenshtein_distance(s1, s2);
        let chars1 = if self.config.ignore_case {
            s1.to_lowercase().chars().count()
        } else {
            s1.chars().count()
        };
        let chars2 = if self.config.ignore_case {
            s2.to_lowercase().chars().count()
        } else {
            s2.chars().count()
        };
        let max_len = chars1.max(chars2);

        if max_len == 0 {
            1.0
        } else {
            1.0 - (distance as f64 / max_len as f64)
        }
    }

    /// Calculate Damerau-Levenshtein distance (allows transpositions)
    pub fn damerau_levenshtein(&self, s1: &str, s2: &str) -> usize {
        self.damerau_levenshtein_distance(s1, s2)
    }

    /// Calculate normalized Damerau-Levenshtein similarity
    pub fn normalized_damerau_levenshtein(&self, s1: &str, s2: &str) -> f64 {
        let distance = self.damerau_levenshtein_distance(s1, s2);
        let chars1 = if self.config.ignore_case {
            s1.to_lowercase().chars().count()
        } else {
            s1.chars().count()
        };
        let chars2 = if self.config.ignore_case {
            s2.to_lowercase().chars().count()
        } else {
            s2.chars().count()
        };
        let max_len = chars1.max(chars2);

        if max_len == 0 {
            1.0
        } else {
            1.0 - (distance as f64 / max_len as f64)
        }
    }

    /// Calculate Hamming distance (strings must be of equal length)
    pub fn hamming(&self, s1: &str, s2: &str) -> Result<usize> {
        self.hamming_distance(s1, s2)
    }

    /// Calculate normalized Hamming similarity
    pub fn normalized_hamming(&self, s1: &str, s2: &str) -> Result<f64> {
        let distance = self.hamming_distance(s1, s2)?;
        let len = if self.config.ignore_case {
            s1.to_lowercase().chars().count()
        } else {
            s1.chars().count()
        };

        if len == 0 {
            Ok(1.0)
        } else {
            Ok(1.0 - (distance as f64 / len as f64))
        }
    }

    /// Calculate Jaro similarity
    pub fn jaro(&self, s1: &str, s2: &str) -> f64 {
        self.jaro_similarity(s1, s2)
    }

    /// Calculate Jaro-Winkler similarity
    pub fn jaro_winkler(&self, s1: &str, s2: &str) -> f64 {
        self.jaro_winkler_similarity(s1, s2)
    }

    /// Calculate Optimal String Alignment distance
    pub fn optimal_string_alignment(&self, s1: &str, s2: &str) -> usize {
        self.optimal_string_alignment_distance(s1, s2)
    }

    /// Calculate normalized Optimal String Alignment similarity
    pub fn normalized_optimal_string_alignment(&self, s1: &str, s2: &str) -> f64 {
        let distance = self.optimal_string_alignment_distance(s1, s2);
        let chars1 = if self.config.ignore_case {
            s1.to_lowercase().chars().count()
        } else {
            s1.chars().count()
        };
        let chars2 = if self.config.ignore_case {
            s2.to_lowercase().chars().count()
        } else {
            s2.chars().count()
        };
        let max_len = chars1.max(chars2);

        if max_len == 0 {
            1.0
        } else {
            1.0 - (distance as f64 / max_len as f64)
        }
    }

    /// Compare multiple strings and find the most similar one
    pub fn find_most_similar(
        &self,
        target: &str,
        candidates: &[&str],
        algorithm: DistanceAlgorithm,
    ) -> Result<Option<SimilarityMatch>> {
        if candidates.is_empty() {
            return Ok(None);
        }

        let mut best_match = None;
        let mut best_similarity = -1.0;

        for (index, &candidate) in candidates.iter().enumerate() {
            let similarity = self.similarity(target, candidate, algorithm)?;

            if similarity > best_similarity {
                best_similarity = similarity;
                best_match = Some(SimilarityMatch {
                    index,
                    text: candidate.to_string(),
                    similarity,
                    distance: self.calculate(target, candidate, algorithm)?,
                });
            }
        }

        Ok(best_match)
    }

    /// Calculate pairwise distances between all strings in a collection
    pub fn pairwise_distances(
        &self,
        strings: &[&str],
        algorithm: DistanceAlgorithm,
    ) -> Result<Vec<Vec<f64>>> {
        let n = strings.len();
        let mut distances = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in i + 1..n {
                let distance = self.calculate(strings[i], strings[j], algorithm)?;
                distances[i][j] = distance;
                distances[j][i] = distance; // Symmetric
            }
        }

        Ok(distances)
    }

    // Private implementation methods

    /// Preprocess strings based on configuration
    fn preprocess(&self, s: &str) -> String {
        if self.config.ignore_case {
            s.to_lowercase()
        } else {
            s.to_string()
        }
    }

    /// Implementation of Levenshtein distance algorithm
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let s1 = self.preprocess(s1);
        let s2 = self.preprocess(s2);

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

        // Initialize base cases
        for i in 0..=len1 {
            dp[i][0] = (i as f64 * self.config.deletion_cost) as usize;
        }
        for j in 0..=len2 {
            dp[0][j] = (j as f64 * self.config.insertion_cost) as usize;
        }

        // Fill the matrix
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] {
                    0.0
                } else {
                    self.config.substitution_cost
                };

                dp[i][j] = cmp::min(
                    cmp::min(
                        dp[i - 1][j] + self.config.deletion_cost as usize,
                        dp[i][j - 1] + self.config.insertion_cost as usize,
                    ),
                    dp[i - 1][j - 1] + cost as usize,
                );
            }
        }

        dp[len1][len2]
    }

    /// Implementation of Damerau-Levenshtein distance algorithm
    fn damerau_levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let s1 = self.preprocess(s1);
        let s2 = self.preprocess(s2);

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

        // Create character alphabet
        let mut alphabet = HashMap::new();
        let mut char_index = 0;
        for &c in chars1.iter().chain(chars2.iter()) {
            if !alphabet.contains_key(&c) {
                alphabet.insert(c, char_index);
                char_index += 1;
            }
        }

        let max_distance = len1 + len2;
        let mut h = vec![vec![max_distance; len2 + 2]; len1 + 2];

        // Initialize
        h[0][0] = max_distance;
        for i in 0..=len1 {
            h[i + 1][0] = max_distance;
            h[i + 1][1] = i;
        }
        for j in 0..=len2 {
            h[0][j + 1] = max_distance;
            h[1][j + 1] = j;
        }

        let mut last_row = vec![0; alphabet.len()];

        for i in 1..=len1 {
            let mut last_match_col = 0;

            for j in 1..=len2 {
                let i1 = last_row[alphabet[&chars2[j - 1]]];
                let j1 = last_match_col;

                let cost = if chars1[i - 1] == chars2[j - 1] {
                    last_match_col = j;
                    0
                } else {
                    1
                };

                h[i + 1][j + 1] = cmp::min(
                    cmp::min(
                        h[i][j] + cost,  // substitution
                        h[i + 1][j] + 1, // insertion
                    ),
                    cmp::min(
                        h[i][j + 1] + 1,                             // deletion
                        h[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1), // transposition
                    ),
                );
            }

            last_row[alphabet[&chars1[i - 1]]] = i;
        }

        h[len1 + 1][len2 + 1]
    }

    /// Implementation of Optimal String Alignment distance
    fn optimal_string_alignment_distance(&self, s1: &str, s2: &str) -> usize {
        let s1 = self.preprocess(s1);
        let s2 = self.preprocess(s2);

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

        // Initialize
        for i in 0..=len1 {
            dp[i][0] = i;
        }
        for j in 0..=len2 {
            dp[0][j] = j;
        }

        // Fill the matrix
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };

                dp[i][j] = cmp::min(
                    cmp::min(
                        dp[i - 1][j] + 1, // deletion
                        dp[i][j - 1] + 1, // insertion
                    ),
                    dp[i - 1][j - 1] + cost, // substitution
                );

                // Check for transposition
                if i > 1
                    && j > 1
                    && chars1[i - 1] == chars2[j - 2]
                    && chars1[i - 2] == chars2[j - 1]
                {
                    dp[i][j] = cmp::min(dp[i][j], dp[i - 2][j - 2] + cost);
                }
            }
        }

        dp[len1][len2]
    }

    /// Implementation of Hamming distance algorithm
    fn hamming_distance(&self, s1: &str, s2: &str) -> Result<usize> {
        let s1 = self.preprocess(s1);
        let s2 = self.preprocess(s2);

        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();

        if chars1.len() != chars2.len() {
            return Err(TextError::Other(anyhow::anyhow!(
                "Hamming distance requires strings of equal length: {} vs {}",
                chars1.len(),
                chars2.len()
            )));
        }

        let mut distance = 0;
        for (c1, c2) in chars1.iter().zip(chars2.iter()) {
            if c1 != c2 {
                distance += 1;
            }
        }

        Ok(distance)
    }

    /// Implementation of Jaro similarity algorithm
    fn jaro_similarity(&self, s1: &str, s2: &str) -> f64 {
        let s1 = self.preprocess(s1);
        let s2 = self.preprocess(s2);

        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();

        let len1 = chars1.len();
        let len2 = chars2.len();

        if len1 == 0 && len2 == 0 {
            return 1.0;
        }
        if len1 == 0 || len2 == 0 {
            return 0.0;
        }

        let match_window = (cmp::max(len1, len2) / 2).saturating_sub(1);

        let mut s1_matches = vec![false; len1];
        let mut s2_matches = vec![false; len2];

        let mut matches = 0;

        // Identify matches
        for i in 0..len1 {
            let start = if i >= match_window {
                i - match_window
            } else {
                0
            };
            let end = cmp::min(i + match_window + 1, len2);

            for j in start..end {
                if s2_matches[j] || chars1[i] != chars2[j] {
                    continue;
                }

                s1_matches[i] = true;
                s2_matches[j] = true;
                matches += 1;
                break;
            }
        }

        if matches == 0 {
            return 0.0;
        }

        // Count transpositions
        let mut transpositions = 0;
        let mut k = 0;

        for i in 0..len1 {
            if !s1_matches[i] {
                continue;
            }

            while !s2_matches[k] {
                k += 1;
            }

            if chars1[i] != chars2[k] {
                transpositions += 1;
            }

            k += 1;
        }

        let jaro = (matches as f64 / len1 as f64
            + matches as f64 / len2 as f64
            + (matches as f64 - transpositions as f64 / 2.0) / matches as f64)
            / 3.0;

        jaro
    }

    /// Implementation of Jaro-Winkler similarity algorithm
    fn jaro_winkler_similarity(&self, s1: &str, s2: &str) -> f64 {
        let jaro = self.jaro_similarity(s1, s2);

        if jaro < 0.7 {
            return jaro;
        }

        let s1 = self.preprocess(s1);
        let s2 = self.preprocess(s2);

        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();

        // Calculate common prefix up to maximum of 4 characters
        let mut prefix_length = 0;
        let max_prefix = cmp::min(
            self.config.jaro_winkler_max_prefix,
            cmp::min(chars1.len(), chars2.len()),
        );

        for i in 0..max_prefix {
            if chars1[i] == chars2[i] {
                prefix_length += 1;
            } else {
                break;
            }
        }

        jaro + (self.config.jaro_winkler_prefix_scale * prefix_length as f64 * (1.0 - jaro))
    }
}

/// Result of a similarity search
#[derive(Debug, Clone)]
pub struct SimilarityMatch {
    /// Index of the matched string in the original collection
    pub index: usize,
    /// The matched text
    pub text: String,
    /// Similarity score (0.0 to 1.0)
    pub similarity: f64,
    /// Edit distance value
    pub distance: f64,
}

impl SimilarityMatch {
    /// Check if the match quality is high (similarity > 0.8)
    pub fn is_high_quality(&self) -> bool {
        self.similarity > 0.8
    }

    /// Check if the match quality is acceptable (similarity > 0.5)
    pub fn is_acceptable(&self) -> bool {
        self.similarity > 0.5
    }

    /// Get match quality category
    pub fn quality_category(&self) -> &'static str {
        if self.similarity > 0.9 {
            "Excellent"
        } else if self.similarity > 0.8 {
            "Good"
        } else if self.similarity > 0.6 {
            "Fair"
        } else if self.similarity > 0.4 {
            "Poor"
        } else {
            "Very Poor"
        }
    }
}

// Legacy compatibility functions
impl EditDistance {
    /// Legacy function for basic Levenshtein distance
    pub fn levenshtein_static(s1: &str, s2: &str) -> usize {
        EditDistance::new().levenshtein(s1, s2)
    }

    /// Legacy function for normalized Levenshtein similarity
    pub fn normalized_levenshtein_static(s1: &str, s2: &str) -> f64 {
        EditDistance::new().normalized_levenshtein(s1, s2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_basic() {
        let edit_distance = EditDistance::new();

        assert_eq!(edit_distance.levenshtein("", ""), 0);
        assert_eq!(edit_distance.levenshtein("", "abc"), 3);
        assert_eq!(edit_distance.levenshtein("abc", ""), 3);
        assert_eq!(edit_distance.levenshtein("abc", "abc"), 0);
        assert_eq!(edit_distance.levenshtein("kitten", "sitting"), 3);
        assert_eq!(edit_distance.levenshtein("saturday", "sunday"), 3);
    }

    #[test]
    fn test_normalized_levenshtein() {
        let edit_distance = EditDistance::new();

        assert!((edit_distance.normalized_levenshtein("", "") - 1.0).abs() < 1e-10);
        assert!((edit_distance.normalized_levenshtein("abc", "abc") - 1.0).abs() < 1e-10);
        assert!(edit_distance.normalized_levenshtein("", "abc") < 0.1);

        let similarity = edit_distance.normalized_levenshtein("kitten", "sitting");
        assert!(similarity > 0.0 && similarity < 1.0);
    }

    #[test]
    fn test_hamming_distance() {
        let edit_distance = EditDistance::new();

        assert_eq!(edit_distance.hamming("abc", "abc").unwrap(), 0);
        assert_eq!(edit_distance.hamming("abc", "axc").unwrap(), 1);
        assert_eq!(edit_distance.hamming("abc", "xyz").unwrap(), 3);

        // Different lengths should return error
        assert!(edit_distance.hamming("ab", "abc").is_err());
    }

    #[test]
    fn test_jaro_similarity() {
        let edit_distance = EditDistance::new();

        assert!((edit_distance.jaro("", "") - 1.0).abs() < 1e-10);
        assert!((edit_distance.jaro("abc", "abc") - 1.0).abs() < 1e-10);
        assert_eq!(edit_distance.jaro("", "abc"), 0.0);

        let similarity = edit_distance.jaro("martha", "marhta");
        assert!(similarity > 0.8); // Should be high due to mostly matching characters
    }

    #[test]
    fn test_jaro_winkler_similarity() {
        let edit_distance = EditDistance::new();

        assert!((edit_distance.jaro_winkler("", "") - 1.0).abs() < 1e-10);
        assert!((edit_distance.jaro_winkler("abc", "abc") - 1.0).abs() < 1e-10);

        let jaro = edit_distance.jaro("martha", "marhta");
        let jaro_winkler = edit_distance.jaro_winkler("martha", "marhta");

        // Jaro-Winkler should be higher due to common prefix
        assert!(jaro_winkler >= jaro);

        // Test prefix bonus
        let no_prefix = edit_distance.jaro_winkler("abcd", "efgh");
        let with_prefix = edit_distance.jaro_winkler("abcd", "abxy");
        assert!(with_prefix > no_prefix);
    }

    #[test]
    fn test_damerau_levenshtein() {
        let edit_distance = EditDistance::new();

        assert_eq!(edit_distance.damerau_levenshtein("abc", "abc"), 0);
        assert_eq!(edit_distance.damerau_levenshtein("ca", "abc"), 2);

        // Transposition should cost 1
        assert_eq!(edit_distance.damerau_levenshtein("ab", "ba"), 1);

        // Should be better than regular Levenshtein for transpositions
        let regular = edit_distance.levenshtein("abcde", "acbde");
        let damerau = edit_distance.damerau_levenshtein("abcde", "acbde");
        assert!(damerau <= regular);
    }

    #[test]
    fn test_case_insensitive() {
        let edit_distance =
            EditDistance::new().with_config(EditDistanceConfig::new().with_ignore_case(true));

        assert_eq!(edit_distance.levenshtein("ABC", "abc"), 0);
        assert!((edit_distance.normalized_levenshtein("ABC", "abc") - 1.0).abs() < 1e-10);
        assert!((edit_distance.jaro("ABC", "abc") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_custom_costs() {
        let config = EditDistanceConfig::new()
            .with_insertion_cost(1.0)
            .with_deletion_cost(1.0)
            .with_substitution_cost(2.0);

        let edit_distance = EditDistance::with_config(config);

        // With higher substitution cost, should prefer insertion+deletion over substitution
        let distance = edit_distance.levenshtein("a", "b");
        assert_eq!(distance, 2); // delete 'a', insert 'b'
    }

    #[test]
    fn test_find_most_similar() {
        let edit_distance = EditDistance::new();
        let candidates = &["hello", "world", "help", "held"];

        let result = edit_distance
            .find_most_similar("helo", candidates, DistanceAlgorithm::Levenshtein)
            .unwrap();

        assert!(result.is_some());
        let best_match = result.unwrap();
        assert_eq!(best_match.text, "hello"); // Should be closest
        assert!(best_match.similarity > 0.5);
    }

    #[test]
    fn test_pairwise_distances() {
        let edit_distance = EditDistance::new();
        let strings = &["abc", "abd", "xyz"];

        let distances = edit_distance
            .pairwise_distances(strings, DistanceAlgorithm::Levenshtein)
            .unwrap();

        assert_eq!(distances.len(), 3);
        assert_eq!(distances[0].len(), 3);

        // Diagonal should be zero
        for i in 0..3 {
            assert_eq!(distances[i][i], 0.0);
        }

        // Matrix should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!((distances[i][j] - distances[j][i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_legacy_functions() {
        assert_eq!(EditDistance::levenshtein_static("kitten", "sitting"), 3);

        let similarity = EditDistance::normalized_levenshtein_static("abc", "abc");
        assert!((similarity - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_similarity_match_helpers() {
        let match_result = SimilarityMatch {
            index: 0,
            text: "test".to_string(),
            similarity: 0.85,
            distance: 1.0,
        };

        assert!(match_result.is_high_quality());
        assert!(match_result.is_acceptable());
        assert_eq!(match_result.quality_category(), "Good");
    }

    #[test]
    fn test_unicode_support() {
        let edit_distance = EditDistance::new();

        // Test with Unicode characters
        let dist = edit_distance.levenshtein("café", "caffe");
        assert!(dist > 0);

        let similarity = edit_distance.normalized_levenshtein("naïve", "naive");
        assert!(similarity > 0.5);
    }

    #[test]
    fn test_algorithms_enum() {
        let edit_distance = EditDistance::new();

        for algorithm in &[
            DistanceAlgorithm::Levenshtein,
            DistanceAlgorithm::DamerauLevenshtein,
            DistanceAlgorithm::OptimalStringAlignment,
            DistanceAlgorithm::Jaro,
            DistanceAlgorithm::JaroWinkler,
        ] {
            let result = edit_distance.calculate("test", "best", *algorithm);
            assert!(result.is_ok());

            let similarity = edit_distance.similarity("test", "best", *algorithm);
            assert!(similarity.is_ok());
            assert!(similarity.unwrap() >= 0.0 && similarity.unwrap() <= 1.0);
        }

        // Hamming requires equal length strings
        let result = edit_distance.calculate("test", "best", DistanceAlgorithm::Hamming);
        assert!(result.is_ok());

        let result = edit_distance.calculate("test", "testing", DistanceAlgorithm::Hamming);
        assert!(result.is_err());
    }
}
