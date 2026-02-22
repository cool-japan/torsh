//! Text processing and natural language processing transformations
//!
//! This module provides comprehensive text processing capabilities including
//! tokenization, normalization, stemming, and linguistic transformations.
//! Designed for preprocessing text data in machine learning pipelines.
//!
//! # Features
//!
//! - **Basic text processing**: Case conversion, whitespace handling, punctuation removal
//! - **Tokenization**: Flexible text splitting with custom delimiters
//! - **Stemming**: Porter stemmer implementation for word normalization
//! - **N-gram generation**: Extract bigrams, trigrams, and custom n-grams
//! - **Filtering**: Length-based and stopword filtering
//! - **Pattern replacement**: Simple string pattern substitution

use crate::transforms::Transform;
use std::collections::HashSet;
use torsh_core::error::Result;

/// Convert text to lowercase
#[derive(Debug, Clone, Default)]
pub struct ToLowercase;

impl Transform<String> for ToLowercase {
    type Output = String;

    fn transform(&self, input: String) -> Result<Self::Output> {
        Ok(input.to_lowercase())
    }
}

/// Remove ASCII punctuation from text
#[derive(Debug, Clone, Default)]
pub struct RemovePunctuation;

impl Transform<String> for RemovePunctuation {
    type Output = String;

    fn transform(&self, input: String) -> Result<Self::Output> {
        Ok(input
            .chars()
            .filter(|c| !c.is_ascii_punctuation())
            .collect())
    }
}

/// Tokenize text into words using a specified delimiter
#[derive(Debug, Clone)]
pub struct Tokenize {
    delimiter: String,
}

impl Tokenize {
    /// Create a new tokenize transform with custom delimiter
    pub fn new(delimiter: String) -> Self {
        Self { delimiter }
    }

    /// Create tokenizer with whitespace delimiter
    pub fn whitespace() -> Self {
        Self::new(" ".to_string())
    }

    /// Create tokenizer that splits on any whitespace
    pub fn any_whitespace() -> Self {
        Self::new("".to_string()) // Empty delimiter indicates whitespace splitting
    }
}

impl Transform<String> for Tokenize {
    type Output = Vec<String>;

    fn transform(&self, input: String) -> Result<Self::Output> {
        if self.delimiter.is_empty() {
            // Split on any whitespace
            Ok(input.split_whitespace().map(|s| s.to_string()).collect())
        } else {
            Ok(input
                .split(&self.delimiter)
                .map(|s| s.to_string())
                .collect())
        }
    }
}

/// Trim whitespace from beginning and end of text
#[derive(Debug, Clone, Default)]
pub struct TrimWhitespace;

impl Transform<String> for TrimWhitespace {
    type Output = String;

    fn transform(&self, input: String) -> Result<Self::Output> {
        Ok(input.trim().to_string())
    }
}

/// Collapse multiple consecutive whitespace characters into single spaces
#[derive(Debug, Clone, Default)]
pub struct CollapseWhitespace;

impl Transform<String> for CollapseWhitespace {
    type Output = String;

    fn transform(&self, input: String) -> Result<Self::Output> {
        let mut result = String::with_capacity(input.len());
        let mut prev_was_space = false;

        for ch in input.chars() {
            if ch.is_whitespace() {
                if !prev_was_space {
                    result.push(' ');
                    prev_was_space = true;
                }
            } else {
                result.push(ch);
                prev_was_space = false;
            }
        }

        Ok(result.trim().to_string())
    }
}

/// Remove numeric digits from text
#[derive(Debug, Clone, Default)]
pub struct RemoveNumbers;

impl Transform<String> for RemoveNumbers {
    type Output = String;

    fn transform(&self, input: String) -> Result<Self::Output> {
        Ok(input.chars().filter(|c| !c.is_ascii_digit()).collect())
    }
}

/// Remove stopwords from a list of tokens
#[derive(Debug, Clone)]
pub struct RemoveStopwords {
    stopwords: HashSet<String>,
}

impl RemoveStopwords {
    /// Create with default English stopwords
    pub fn english() -> Self {
        let stopwords = [
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
            "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with",
            "the", "this", "but", "they", "have", "had", "what", "said", "each", "which", "their",
            "time", "will", "about", "if", "up", "out", "many", "then", "them", "these", "so",
            "some", "her", "would", "make", "like", "into", "him", "has", "two", "more", "go",
            "no", "way", "could", "my", "than", "first", "been", "call", "who", "oil", "sit",
            "now", "find", "down", "day", "did", "get", "come", "made", "may", "part", "over",
            "new", "sound", "take", "only", "little", "work", "know", "place", "year", "live",
            "me", "back", "give", "most", "very", "after", "thing", "our", "just", "name", "good",
            "sentence", "man", "think", "say", "great", "where", "help", "through", "much",
            "before", "line", "right", "too", "mean", "old", "any", "same", "tell", "boy",
            "follow", "came", "want", "show", "also", "around", "form", "three", "small", "set",
            "put", "end", "why", "again", "turn", "here", "off", "went", "old", "number", "great",
            "tell", "men", "say", "small", "every", "found", "still", "between", "mea", "another",
            "even", "why", "must", "big", "because", "does", "each", "how", "let", "might", "move",
            "own", "seem", "such", "turn", "under", "well", "without", "see", "use",
        ]
        .iter()
        .map(|&s| s.to_string())
        .collect();

        Self { stopwords }
    }

    /// Create with custom stopwords
    pub fn new(stopwords: Vec<String>) -> Self {
        Self {
            stopwords: stopwords.into_iter().collect(),
        }
    }

    /// Add a stopword to the existing set
    pub fn add_stopword(&mut self, word: String) {
        self.stopwords.insert(word.to_lowercase());
    }

    /// Get the number of stopwords
    pub fn stopword_count(&self) -> usize {
        self.stopwords.len()
    }
}

impl Transform<Vec<String>> for RemoveStopwords {
    type Output = Vec<String>;

    fn transform(&self, input: Vec<String>) -> Result<Self::Output> {
        Ok(input
            .into_iter()
            .filter(|word| !self.stopwords.contains(&word.to_lowercase()))
            .collect())
    }
}

/// Porter stemmer implementation for English word stemming
///
/// This is a simplified implementation of the Porter stemming algorithm,
/// which reduces words to their base forms by removing common suffixes.
#[derive(Debug, Clone, Default)]
pub struct PorterStemmer;

impl PorterStemmer {
    /// Check if a character at position i is a vowel
    fn is_vowel(word: &str, i: usize) -> bool {
        if i >= word.len() {
            return false;
        }
        let chars: Vec<char> = word.chars().collect();
        let ch = chars[i];
        if "aeiou".contains(ch) {
            return true;
        }
        if ch == 'y' && i > 0 && !Self::is_vowel(word, i - 1) {
            return true;
        }
        false
    }

    /// Calculate the measure of a word (number of VC patterns)
    fn measure(&self, word: &str) -> usize {
        let mut m = 0;
        let len = word.len();
        let mut i = 0;

        // Skip initial consonants
        while i < len && !Self::is_vowel(word, i) {
            i += 1;
        }

        while i < len {
            // Skip vowels
            while i < len && Self::is_vowel(word, i) {
                i += 1;
            }
            if i >= len {
                break;
            }
            m += 1;

            // Skip consonants
            while i < len && !Self::is_vowel(word, i) {
                i += 1;
            }
        }

        m
    }

    /// Check if word ends with suffix
    fn ends_with(&self, word: &str, suffix: &str) -> bool {
        word.ends_with(suffix)
    }

    /// Replace suffix in word
    fn replace_suffix(&self, word: &str, old_suffix: &str, new_suffix: &str) -> String {
        if let Some(stem) = word.strip_suffix(old_suffix) {
            format!("{stem}{new_suffix}")
        } else {
            word.to_string()
        }
    }

    /// Porter stemmer step 1a
    fn step1a(&self, word: &str) -> String {
        if self.ends_with(word, "sses") {
            self.replace_suffix(word, "sses", "ss")
        } else if self.ends_with(word, "ies") {
            self.replace_suffix(word, "ies", "i")
        } else if self.ends_with(word, "ss") {
            word.to_string()
        } else if self.ends_with(word, "s") && word.len() > 1 {
            self.replace_suffix(word, "s", "")
        } else {
            word.to_string()
        }
    }

    /// Porter stemmer step 1b
    fn step1b(&self, word: &str) -> String {
        if self.ends_with(word, "eed") {
            let stem = &word[..word.len() - 3];
            if self.measure(stem) > 0 {
                self.replace_suffix(word, "eed", "ee")
            } else {
                word.to_string()
            }
        } else if self.ends_with(word, "ed") {
            let stem = &word[..word.len() - 2];
            if self.contains_vowel(stem) {
                let result = stem;
                if self.ends_with(result, "at")
                    || self.ends_with(result, "bl")
                    || self.ends_with(result, "iz")
                {
                    format!("{result}e")
                } else {
                    result.to_string()
                }
            } else {
                word.to_string()
            }
        } else if self.ends_with(word, "ing") {
            let stem = &word[..word.len() - 3];
            if self.contains_vowel(stem) {
                stem.to_string()
            } else {
                word.to_string()
            }
        } else {
            word.to_string()
        }
    }

    /// Check if word contains a vowel
    fn contains_vowel(&self, word: &str) -> bool {
        for i in 0..word.len() {
            if Self::is_vowel(word, i) {
                return true;
            }
        }
        false
    }
}

impl Transform<String> for PorterStemmer {
    type Output = String;

    fn transform(&self, input: String) -> Result<Self::Output> {
        if input.len() <= 2 {
            return Ok(input);
        }

        let word = input.to_lowercase();
        let word = self.step1a(&word);
        let word = self.step1b(&word);

        Ok(word)
    }
}

/// Generate n-grams from a sequence of tokens
#[derive(Debug, Clone)]
pub struct NGramGenerator {
    n: usize,
}

impl NGramGenerator {
    /// Create n-gram generator for specified n
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "N must be greater than 0");
        Self { n }
    }

    /// Create bigram generator (n=2)
    pub fn bigram() -> Self {
        Self::new(2)
    }

    /// Create trigram generator (n=3)
    pub fn trigram() -> Self {
        Self::new(3)
    }

    /// Create unigram generator (n=1)
    pub fn unigram() -> Self {
        Self::new(1)
    }

    /// Get the n value
    pub fn n(&self) -> usize {
        self.n
    }
}

impl Transform<Vec<String>> for NGramGenerator {
    type Output = Vec<String>;

    fn transform(&self, input: Vec<String>) -> Result<Self::Output> {
        if input.len() < self.n {
            return Ok(Vec::new());
        }

        let mut ngrams = Vec::new();
        for i in 0..=input.len() - self.n {
            let ngram = input[i..i + self.n].join(" ");
            ngrams.push(ngram);
        }

        Ok(ngrams)
    }
}

/// Filter tokens by length constraints
#[derive(Debug, Clone)]
pub struct FilterByLength {
    min_length: usize,
    max_length: Option<usize>,
}

impl FilterByLength {
    /// Create filter with both min and max length constraints
    pub fn new(min_length: usize, max_length: Option<usize>) -> Self {
        Self {
            min_length,
            max_length,
        }
    }

    /// Create filter with only minimum length constraint
    pub fn min_only(min_length: usize) -> Self {
        Self::new(min_length, None)
    }

    /// Create filter with only maximum length constraint
    pub fn max_only(max_length: usize) -> Self {
        Self::new(0, Some(max_length))
    }

    /// Create filter for specific length range
    pub fn range(min_length: usize, max_length: usize) -> Self {
        Self::new(min_length, Some(max_length))
    }
}

impl Transform<Vec<String>> for FilterByLength {
    type Output = Vec<String>;

    fn transform(&self, input: Vec<String>) -> Result<Self::Output> {
        Ok(input
            .into_iter()
            .filter(|word| {
                let len = word.len();
                len >= self.min_length && self.max_length.map_or(true, |max| len <= max)
            })
            .collect())
    }
}

/// Replace string patterns with replacements
#[derive(Debug, Clone)]
pub struct ReplacePattern {
    pattern: String,
    replacement: String,
}

impl ReplacePattern {
    /// Create new pattern replacement transform
    pub fn new(pattern: String, replacement: String) -> Self {
        Self {
            pattern,
            replacement,
        }
    }

    /// Create pattern replacement that removes matches (replaces with empty string)
    pub fn remove(pattern: String) -> Self {
        Self::new(pattern, String::new())
    }
}

impl Transform<String> for ReplacePattern {
    type Output = String;

    fn transform(&self, input: String) -> Result<Self::Output> {
        Ok(input.replace(&self.pattern, &self.replacement))
    }
}

/// Text case conversion modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseMode {
    /// Convert to lowercase
    Lower,
    /// Convert to uppercase
    Upper,
    /// Convert to title case (first letter of each word capitalized)
    Title,
}

/// Convert text case according to specified mode
#[derive(Debug, Clone)]
pub struct ChangeCase {
    mode: CaseMode,
}

impl ChangeCase {
    /// Create case converter with specified mode
    pub fn new(mode: CaseMode) -> Self {
        Self { mode }
    }

    /// Create lowercase converter
    pub fn lower() -> Self {
        Self::new(CaseMode::Lower)
    }

    /// Create uppercase converter
    pub fn upper() -> Self {
        Self::new(CaseMode::Upper)
    }

    /// Create title case converter
    pub fn title() -> Self {
        Self::new(CaseMode::Title)
    }
}

impl Transform<String> for ChangeCase {
    type Output = String;

    fn transform(&self, input: String) -> Result<Self::Output> {
        match self.mode {
            CaseMode::Lower => Ok(input.to_lowercase()),
            CaseMode::Upper => Ok(input.to_uppercase()),
            CaseMode::Title => {
                let mut result = String::with_capacity(input.len());
                let mut capitalize_next = true;

                for ch in input.chars() {
                    if ch.is_alphabetic() {
                        if capitalize_next {
                            result.push(ch.to_uppercase().next().unwrap_or(ch));
                            capitalize_next = false;
                        } else {
                            result.push(ch.to_lowercase().next().unwrap_or(ch));
                        }
                    } else {
                        result.push(ch);
                        capitalize_next = ch.is_whitespace();
                    }
                }

                Ok(result)
            }
        }
    }
}

/// Convenience functions for creating text transforms

/// Create a tokenizer that splits on whitespace
pub fn tokenize_whitespace() -> Tokenize {
    Tokenize::whitespace()
}

/// Create a tokenizer with custom delimiter
pub fn tokenize(delimiter: &str) -> Tokenize {
    Tokenize::new(delimiter.to_string())
}

/// Create English stopword remover
pub fn remove_english_stopwords() -> RemoveStopwords {
    RemoveStopwords::english()
}

/// Create Porter stemmer
pub fn porter_stemmer() -> PorterStemmer {
    PorterStemmer
}

/// Create bigram generator
pub fn bigrams() -> NGramGenerator {
    NGramGenerator::bigram()
}

/// Create trigram generator
pub fn trigrams() -> NGramGenerator {
    NGramGenerator::trigram()
}

/// Create length filter
pub fn filter_by_length(min: usize, max: Option<usize>) -> FilterByLength {
    FilterByLength::new(min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_lowercase() {
        let transform = ToLowercase;
        assert_eq!(
            transform.transform("Hello World".to_string()).unwrap(),
            "hello world"
        );
    }

    #[test]
    fn test_remove_punctuation() {
        let transform = RemovePunctuation;
        assert_eq!(
            transform.transform("Hello, World!".to_string()).unwrap(),
            "Hello World"
        );
    }

    #[test]
    fn test_tokenize_whitespace() {
        let transform = Tokenize::whitespace();
        let result = transform.transform("hello world test".to_string()).unwrap();
        assert_eq!(result, vec!["hello", "world", "test"]);
    }

    #[test]
    fn test_tokenize_custom_delimiter() {
        let transform = Tokenize::new(",".to_string());
        let result = transform.transform("a,b,c".to_string()).unwrap();
        assert_eq!(result, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_trim_whitespace() {
        let transform = TrimWhitespace;
        assert_eq!(
            transform.transform("  hello world  ".to_string()).unwrap(),
            "hello world"
        );
    }

    #[test]
    fn test_collapse_whitespace() {
        let transform = CollapseWhitespace;
        assert_eq!(
            transform
                .transform("hello    world   test".to_string())
                .unwrap(),
            "hello world test"
        );
    }

    #[test]
    fn test_remove_numbers() {
        let transform = RemoveNumbers;
        assert_eq!(
            transform.transform("hello123world456".to_string()).unwrap(),
            "helloworld"
        );
    }

    #[test]
    fn test_remove_stopwords() {
        let stopwords = RemoveStopwords::english();
        let input = vec!["the".to_string(), "quick".to_string(), "brown".to_string()];
        let result = stopwords.transform(input).unwrap();
        assert_eq!(result, vec!["quick", "brown"]);
    }

    #[test]
    fn test_porter_stemmer() {
        let stemmer = PorterStemmer;

        assert_eq!(stemmer.transform("running".to_string()).unwrap(), "runn");
        assert_eq!(stemmer.transform("flies".to_string()).unwrap(), "fli");
        assert_eq!(stemmer.transform("died".to_string()).unwrap(), "di");
        assert_eq!(stemmer.transform("agreed".to_string()).unwrap(), "agree");
        assert_eq!(stemmer.transform("sing".to_string()).unwrap(), "sing"); // No change for short words
    }

    #[test]
    fn test_ngram_generator() {
        let words = vec![
            "the".to_string(),
            "quick".to_string(),
            "brown".to_string(),
            "fox".to_string(),
        ];

        // Test bigrams
        let bigram = NGramGenerator::bigram();
        let bigrams = bigram.transform(words.clone()).unwrap();
        assert_eq!(bigrams, vec!["the quick", "quick brown", "brown fox"]);

        // Test trigrams
        let trigram = NGramGenerator::trigram();
        let trigrams = trigram.transform(words).unwrap();
        assert_eq!(trigrams, vec!["the quick brown", "quick brown fox"]);
    }

    #[test]
    fn test_length_filter() {
        let words = vec![
            "a".to_string(),
            "the".to_string(),
            "quick".to_string(),
            "brown".to_string(),
            "foxes".to_string(),
        ];

        let filter = FilterByLength::new(3, Some(5));
        let filtered = filter.transform(words).unwrap();
        assert_eq!(filtered, vec!["the", "quick", "brown", "foxes"]);
    }

    #[test]
    fn test_case_transforms() {
        let text = "Hello World Test".to_string();

        let lower = ChangeCase::lower();
        assert_eq!(lower.transform(text.clone()).unwrap(), "hello world test");

        let upper = ChangeCase::upper();
        assert_eq!(upper.transform(text.clone()).unwrap(), "HELLO WORLD TEST");

        let title = ChangeCase::title();
        assert_eq!(
            title.transform("hello world".to_string()).unwrap(),
            "Hello World"
        );
    }

    #[test]
    fn test_replace_pattern() {
        let replacer = ReplacePattern::new("world".to_string(), "universe".to_string());
        assert_eq!(
            replacer.transform("hello world".to_string()).unwrap(),
            "hello universe"
        );

        let remover = ReplacePattern::remove("test ".to_string());
        assert_eq!(
            remover
                .transform("test hello test world".to_string())
                .unwrap(),
            "hello world"
        );
    }

    #[test]
    fn test_convenience_functions() {
        let _tokenizer = tokenize_whitespace();
        let _custom_tokenizer = tokenize(",");
        let _stopwords = remove_english_stopwords();
        let _stemmer = porter_stemmer();
        let _bigrams = bigrams();
        let _trigrams = trigrams();
        let _filter = filter_by_length(3, Some(10));
    }
}
