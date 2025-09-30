//! Convenience utilities for common text processing tasks
//!
//! This module provides high-level, easy-to-use functions for common text processing
//! operations that combine multiple lower-level components for maximum convenience.

use crate::analysis::{TextAnalyzer, TextStatistics};
// use crate::metrics::BleuScore;  // Temporarily disabled due to module issues
use crate::scirs2_ops::advanced_analytics::{
    compute_advanced_stats, AdvancedTextSampler, ComplexityAnalyzer,
};
use crate::scirs2_ops::performance::PerformanceMonitor;
use crate::tokenization::{BPETokenizer, Tokenizer, WhitespaceTokenizer};
use crate::utils::{TextCleaner, TextNormalizer};
use crate::{Result, TextError};
use rayon::prelude::*;
use std::collections::HashMap;

/// One-stop text processing pipeline for quick text analysis
pub struct QuickTextProcessor {
    normalizer: TextNormalizer,
    cleaner: TextCleaner,
    tokenizer: Box<dyn Tokenizer>,
}

impl Default for QuickTextProcessor {
    fn default() -> Self {
        Self {
            normalizer: TextNormalizer::new(),
            cleaner: TextCleaner::new(),
            tokenizer: Box::new(WhitespaceTokenizer::new()),
        }
    }
}

impl QuickTextProcessor {
    /// Create a new quick text processor with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set custom normalizer
    pub fn with_normalizer(mut self, normalizer: TextNormalizer) -> Self {
        self.normalizer = normalizer;
        self
    }

    /// Set custom cleaner
    pub fn with_cleaner(mut self, cleaner: TextCleaner) -> Self {
        self.cleaner = cleaner;
        self
    }

    /// Set custom tokenizer
    pub fn with_tokenizer(mut self, tokenizer: Box<dyn Tokenizer>) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Process text through the full pipeline: clean -> normalize -> tokenize
    pub fn process(&self, text: &str) -> Result<Vec<String>> {
        let cleaned = self.cleaner.clean(text);
        let normalized = self.normalizer.normalize(&cleaned);
        self.tokenizer.tokenize(&normalized)
    }

    /// Get quick statistics for text
    pub fn quick_stats(&self, text: &str) -> Result<TextStatistics> {
        let analyzer = TextAnalyzer::default();
        analyzer.analyze(text)
    }

    /// Compare two texts and return similarity score (0.0 to 1.0)
    pub fn similarity(&self, text1: &str, text2: &str) -> Result<f64> {
        let tokens1 = self.process(text1)?;
        let tokens2 = self.process(text2)?;

        // Simple Jaccard similarity
        let set1: std::collections::HashSet<_> = tokens1.iter().collect();
        let set2: std::collections::HashSet<_> = tokens2.iter().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        if union == 0 {
            Ok(0.0)
        } else {
            Ok(intersection as f64 / union as f64)
        }
    }
}

/// Batch text processing for efficient handling of multiple documents
pub struct BatchTextProcessor {
    processor: QuickTextProcessor,
    batch_size: usize,
}

impl BatchTextProcessor {
    /// Create a new batch processor
    pub fn new(processor: QuickTextProcessor, batch_size: usize) -> Self {
        Self {
            processor,
            batch_size,
        }
    }

    /// Process multiple texts efficiently in batches
    pub fn process_batch(&self, texts: &[String]) -> Result<Vec<Vec<String>>> {
        texts
            .chunks(self.batch_size)
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|text| self.processor.process(text))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()
            .map(|batches| batches.into_iter().flatten().collect())
    }

    /// Get statistics for multiple texts
    pub fn batch_stats(&self, texts: &[String]) -> Result<Vec<TextStatistics>> {
        texts
            .iter()
            .map(|text| self.processor.quick_stats(text))
            .collect()
    }

    /// Get statistics for multiple texts using parallel processing
    pub fn batch_stats_parallel(&self, texts: &[String]) -> Result<Vec<TextStatistics>> {
        texts
            .par_iter()
            .map(|text| self.processor.quick_stats(text))
            .collect()
    }

    /// Create similarity matrix for multiple texts
    /// Optimized to only calculate upper triangle since similarity is symmetric
    pub fn similarity_matrix(&self, texts: &[String]) -> Result<Vec<Vec<f64>>> {
        let n = texts.len();
        let mut matrix = vec![vec![0.0; n]; n];

        // Set diagonal to 1.0 (self-similarity)
        for i in 0..n {
            matrix[i][i] = 1.0;
        }

        // Calculate upper triangle and mirror to lower triangle
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = self.processor.similarity(&texts[i], &texts[j])?;
                matrix[i][j] = sim;
                matrix[j][i] = sim; // Mirror to lower triangle
            }
        }

        Ok(matrix)
    }

    /// Create similarity matrix for multiple texts using parallel processing
    /// Optimized to only calculate upper triangle since similarity is symmetric
    pub fn similarity_matrix_parallel(&self, texts: &[String]) -> Result<Vec<Vec<f64>>> {
        let n = texts.len();
        let mut matrix = vec![vec![0.0; n]; n];

        // Set diagonal to 1.0 (self-similarity)
        for i in 0..n {
            matrix[i][i] = 1.0;
        }

        // Calculate upper triangle pairs in parallel
        let pairs: Vec<(usize, usize)> = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .collect();

        let similarities: Result<Vec<f64>> = pairs
            .par_iter()
            .map(|&(i, j)| self.processor.similarity(&texts[i], &texts[j]))
            .collect();

        let similarities = similarities?;

        // Fill matrix with calculated similarities
        for (idx, &(i, j)) in pairs.iter().enumerate() {
            let sim = similarities[idx];
            matrix[i][j] = sim;
            matrix[j][i] = sim; // Mirror to lower triangle
        }

        Ok(matrix)
    }
}

/// Language detection utilities
pub struct LanguageDetector {
    /// Language-specific character frequency patterns
    patterns: HashMap<String, HashMap<char, f64>>,
}

impl LanguageDetector {
    /// Create a new language detector with basic patterns
    pub fn new() -> Self {
        let mut patterns = HashMap::new();

        // English pattern (simplified)
        let mut english = HashMap::new();
        english.insert('e', 0.127);
        english.insert('t', 0.091);
        english.insert('a', 0.082);
        english.insert('o', 0.075);
        english.insert('i', 0.070);
        english.insert('n', 0.067);
        english.insert('s', 0.063);
        english.insert('h', 0.061);
        english.insert('r', 0.060);
        patterns.insert("en".to_string(), english);

        // Spanish pattern (simplified)
        let mut spanish = HashMap::new();
        spanish.insert('e', 0.137);
        spanish.insert('a', 0.125);
        spanish.insert('o', 0.088);
        spanish.insert('s', 0.080);
        spanish.insert('r', 0.069);
        spanish.insert('n', 0.067);
        spanish.insert('i', 0.063);
        spanish.insert('d', 0.058);
        spanish.insert('l', 0.050);
        patterns.insert("es".to_string(), spanish);

        Self { patterns }
    }

    /// Detect the most likely language for the given text
    pub fn detect(&self, text: &str) -> Option<String> {
        if text.is_empty() {
            return None;
        }

        let text_chars: Vec<char> = text
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphabetic())
            .collect();
        if text_chars.is_empty() {
            return None;
        }

        let mut char_freq = HashMap::new();
        for &ch in &text_chars {
            *char_freq.entry(ch).or_insert(0) += 1;
        }

        let total_chars = text_chars.len() as f64;
        let mut text_pattern = HashMap::new();
        for (ch, count) in char_freq {
            text_pattern.insert(ch, count as f64 / total_chars);
        }

        let mut best_lang = None;
        let mut best_score = f64::INFINITY;

        for (lang, pattern) in &self.patterns {
            let mut score = 0.0;

            // Calculate chi-squared-like distance
            for (&ch, &expected_freq) in pattern {
                let observed_freq = text_pattern.get(&ch).unwrap_or(&0.0);
                score += (observed_freq - expected_freq).powi(2) / expected_freq;
            }

            if score < best_score {
                best_score = score;
                best_lang = Some(lang.clone());
            }
        }

        best_lang
    }

    /// Add a new language pattern
    pub fn add_language(&mut self, lang: String, pattern: HashMap<char, f64>) {
        self.patterns.insert(lang, pattern);
    }
}

impl Default for LanguageDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Additional text processing utilities for common patterns
pub struct TextUtilities;

impl TextUtilities {
    /// Extract sentences from text with improved sentence boundary detection
    pub fn extract_sentences(text: &str) -> Vec<String> {
        // Handle common abbreviations that shouldn't be sentence boundaries
        let abbreviations = [
            "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Ph.D.", "M.D.", "etc.", "vs.", "e.g.", "i.e.",
        ];
        let mut processed_text = text.to_string();

        // Temporarily replace abbreviations to avoid false sentence breaks
        for (i, abbrev) in abbreviations.iter().enumerate() {
            let placeholder = format!("__ABBREV_{}__", i);
            processed_text = processed_text.replace(abbrev, &placeholder);
        }

        // Split on sentence endings
        let sentences: Vec<String> = processed_text
            .split(&['.', '!', '?'])
            .filter_map(|s| {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            })
            .collect();

        // Restore abbreviations
        sentences
            .into_iter()
            .map(|mut sentence| {
                for (i, abbrev) in abbreviations.iter().enumerate() {
                    let placeholder = format!("__ABBREV_{}__", i);
                    sentence = sentence.replace(&placeholder, abbrev);
                }
                sentence
            })
            .collect()
    }

    /// Extract keywords from text using simple TF-IDF-like approach
    pub fn extract_keywords(text: &str, max_keywords: usize) -> Vec<(String, f64)> {
        let stop_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
            "does", "did", "will", "would", "could", "should", "may", "might", "can", "this",
            "that", "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him",
            "her", "us", "them", "my", "your", "his", "her", "its", "our", "their",
        ];

        let words: Vec<String> = text
            .to_lowercase()
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| w.len() > 2 && !stop_words.contains(w))
            .map(|w| w.to_string())
            .collect();

        let mut word_freq = HashMap::new();
        let total_words = words.len() as f64;

        for word in words {
            *word_freq.entry(word).or_insert(0) += 1;
        }

        let mut keywords: Vec<(String, f64)> = word_freq
            .into_iter()
            .map(|(word, freq)| {
                let tf = freq as f64 / total_words;
                // Simple scoring: frequency weighted by word length
                let score = tf * (word.len() as f64).ln();
                (word, score)
            })
            .collect();

        keywords.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        keywords.truncate(max_keywords);
        keywords
    }

    /// Clean and normalize text for analysis
    pub fn quick_clean(text: &str) -> String {
        // Remove common Unicode issues
        let cleaned = text
            .replace('\u{200B}', "") // Zero-width space
            .replace('\u{FEFF}', "") // Byte order mark
            .replace('\u{00A0}', " ") // Non-breaking space
            .replace('\r', ""); // Carriage returns

        // Normalize multiple whitespace to single space
        let re = regex::Regex::new(r"\s+").unwrap_or_else(|_| panic!("Invalid regex"));
        re.replace_all(&cleaned, " ").trim().to_string()
    }

    /// Detect if text is likely to be in a specific encoding
    pub fn detect_encoding_issues(text: &str) -> Vec<String> {
        let mut issues = Vec::new();

        // Check for common encoding problems
        if text.contains('\u{FFFD}') {
            issues.push("Contains replacement characters (encoding corruption)".to_string());
        }

        if text.chars().any(|c| c as u32 > 0x10FFFF) {
            issues.push("Contains invalid Unicode code points".to_string());
        }

        // Check for suspicious byte sequences that might indicate encoding issues
        let suspicious_patterns = ["Ã¡", "Ã©", "Ã\u{AD}", "Ã³", "Ãº", "Ã±"];
        for pattern in &suspicious_patterns {
            if text.contains(pattern) {
                issues.push("Possible UTF-8 encoding interpreted as Latin-1".to_string());
                break;
            }
        }

        issues
    }

    /// Measure text complexity using multiple metrics
    pub fn text_complexity(text: &str) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();

        let sentences = Self::extract_sentences(text);
        let words: Vec<&str> = text.split_whitespace().collect();
        let chars: Vec<char> = text.chars().collect();

        // Basic metrics
        metrics.insert("sentence_count".to_string(), sentences.len() as f64);
        metrics.insert("word_count".to_string(), words.len() as f64);
        metrics.insert("char_count".to_string(), chars.len() as f64);

        if !sentences.is_empty() && !words.is_empty() {
            // Average words per sentence
            metrics.insert(
                "avg_words_per_sentence".to_string(),
                words.len() as f64 / sentences.len() as f64,
            );

            // Average characters per word
            let total_word_chars: usize = words.iter().map(|w| w.len()).sum();
            metrics.insert(
                "avg_chars_per_word".to_string(),
                total_word_chars as f64 / words.len() as f64,
            );

            // Lexical diversity (unique words / total words)
            let unique_words: std::collections::HashSet<String> =
                words.iter().map(|w| w.to_lowercase()).collect();
            metrics.insert(
                "lexical_diversity".to_string(),
                unique_words.len() as f64 / words.len() as f64,
            );

            // Punctuation density
            let punctuation_count = chars.iter().filter(|c| c.is_ascii_punctuation()).count();
            metrics.insert(
                "punctuation_density".to_string(),
                punctuation_count as f64 / chars.len() as f64,
            );
        }

        metrics
    }
}

/// Text quality assessment utilities
pub struct TextQualityAssessor;

impl TextQualityAssessor {
    /// Assess readability using a simplified Flesch-like score
    pub fn readability_score(text: &str) -> f64 {
        let sentences = text
            .split(&['.', '!', '?'])
            .filter(|s| !s.trim().is_empty())
            .count();
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len();

        if sentences == 0 || word_count == 0 {
            return 0.0;
        }

        let syllables: usize = words.iter().map(|word| count_syllables(word)).sum();

        let avg_sentence_length = word_count as f64 / sentences as f64;
        let avg_syllables_per_word = syllables as f64 / word_count as f64;

        // Simplified Flesch formula
        206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    }

    /// Calculate text diversity (unique words / total words)
    pub fn lexical_diversity(text: &str) -> f64 {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }

        let unique_words: std::collections::HashSet<_> =
            words.iter().map(|w| w.to_lowercase()).collect();
        unique_words.len() as f64 / words.len() as f64
    }

    /// Detect potential spam indicators
    pub fn spam_indicators(text: &str) -> HashMap<String, f64> {
        let mut indicators = HashMap::new();

        let text_lower = text.to_lowercase();

        // Excessive caps ratio
        let caps_count = text.chars().filter(|c| c.is_uppercase()).count();
        let letter_count = text.chars().filter(|c| c.is_alphabetic()).count();
        let caps_ratio = if letter_count > 0 {
            caps_count as f64 / letter_count as f64
        } else {
            0.0
        };
        indicators.insert("caps_ratio".to_string(), caps_ratio);

        // Exclamation marks
        let exclamation_count = text.chars().filter(|&c| c == '!').count();
        let exclamation_ratio = exclamation_count as f64 / text.len() as f64;
        indicators.insert("exclamation_ratio".to_string(), exclamation_ratio);

        // Common spam keywords
        let spam_keywords = ["free", "win", "offer", "click", "buy", "sale", "urgent"];
        let spam_score = spam_keywords
            .iter()
            .filter(|&&keyword| text_lower.contains(keyword))
            .count() as f64
            / spam_keywords.len() as f64;
        indicators.insert("spam_keywords".to_string(), spam_score);

        indicators
    }
}

/// Count approximate syllables in a word (simplified heuristic)
fn count_syllables(word: &str) -> usize {
    let word = word.to_lowercase();
    let vowels = "aeiouy";
    let mut count = 0;
    let mut prev_was_vowel = false;

    for ch in word.chars() {
        let is_vowel = vowels.contains(ch);
        if is_vowel && !prev_was_vowel {
            count += 1;
        }
        prev_was_vowel = is_vowel;
    }

    // Handle silent 'e'
    if word.ends_with('e') && count > 1 {
        count -= 1;
    }

    // Minimum of 1 syllable
    count.max(1)
}

/// Comprehensive text analysis report
#[derive(Debug, Clone)]
pub struct ComprehensiveTextReport {
    pub basic_stats: TextStatistics,
    pub advanced_stats: crate::scirs2_ops::advanced_analytics::AdvancedTextStats,
    pub complexity_metrics: crate::scirs2_ops::advanced_analytics::ComplexityMetrics,
    pub performance_metrics: crate::scirs2_ops::performance::PerformanceMetrics,
    pub sample_words: Vec<String>,
    pub sample_sentences: Vec<String>,
}

/// Enhanced text analyzer with comprehensive reporting
pub struct EnhancedTextAnalyzer {
    sampler: AdvancedTextSampler,
    monitor: PerformanceMonitor,
}

impl EnhancedTextAnalyzer {
    /// Create a new enhanced text analyzer
    pub fn new() -> Self {
        Self {
            sampler: AdvancedTextSampler::with_seed(42),
            monitor: PerformanceMonitor::new("enhanced_analysis"),
        }
    }

    /// Create a new enhanced text analyzer with custom seed
    pub fn with_seed(seed: u64) -> Self {
        Self {
            sampler: AdvancedTextSampler::with_seed(seed),
            monitor: PerformanceMonitor::new("enhanced_analysis"),
        }
    }

    /// Perform comprehensive text analysis
    pub fn analyze_comprehensive(&mut self, text: &str) -> Result<ComprehensiveTextReport> {
        let (result, performance_metrics) = self.monitor.time_operation(text, || -> Result<_> {
            // Basic statistics
            let analyzer = TextAnalyzer::default();
            let basic_stats = analyzer.analyze(text)?;

            // Advanced statistics
            let advanced_stats = compute_advanced_stats(text)?;

            // Complexity analysis
            let complexity_metrics = ComplexityAnalyzer::analyze_complexity(text)?;

            // Sample words and sentences
            let sample_words = self.sampler.sample_words(text, 5);
            let sample_sentences = self.sampler.sample_sentences(text, 3);

            Ok((
                basic_stats,
                advanced_stats,
                complexity_metrics,
                sample_words,
                sample_sentences,
            ))
        });

        let (basic_stats, advanced_stats, complexity_metrics, sample_words, sample_sentences) =
            result?;

        Ok(ComprehensiveTextReport {
            basic_stats,
            advanced_stats,
            complexity_metrics,
            performance_metrics,
            sample_words,
            sample_sentences,
        })
    }

    /// Quick analysis with just key metrics
    pub fn analyze_quick(&mut self, text: &str) -> Result<HashMap<String, f64>> {
        let mut metrics = HashMap::new();

        let (result, performance_metrics) = self.monitor.time_operation(text, || -> Result<_> {
            let advanced_stats = compute_advanced_stats(text)?;
            let complexity = ComplexityAnalyzer::analyze_complexity(text)?;
            Ok((advanced_stats, complexity))
        });

        let (advanced_stats, complexity) = result?;

        metrics.insert("word_count".to_string(), advanced_stats.word_count as f64);
        metrics.insert(
            "sentence_count".to_string(),
            advanced_stats.sentence_count as f64,
        );
        metrics.insert(
            "lexical_diversity".to_string(),
            advanced_stats.lexical_diversity,
        );
        metrics.insert(
            "readability_score".to_string(),
            advanced_stats.readability_score,
        );
        metrics.insert(
            "overall_complexity".to_string(),
            complexity.overall_complexity,
        );
        metrics.insert(
            "chars_per_second".to_string(),
            performance_metrics.throughput_chars_per_sec,
        );

        Ok(metrics)
    }

    /// Generate summary text based on analysis
    pub fn generate_summary(&mut self, text: &str, max_words: usize) -> Result<String> {
        // Use Markov chain generation based on the input text
        let summary = self.sampler.generate_markov_text(text, max_words, 2);
        Ok(summary)
    }
}

impl Default for EnhancedTextAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quick_text_processor() {
        let processor = QuickTextProcessor::new();
        let result = processor.process("Hello, world! This is a test.").unwrap();
        assert!(!result.is_empty());
    }

    #[test]
    fn test_similarity() {
        let processor = QuickTextProcessor::new();
        let sim = processor.similarity("hello world", "hello world").unwrap();
        assert_eq!(sim, 1.0);

        let sim = processor.similarity("hello world", "goodbye moon").unwrap();
        assert!(sim < 1.0);
    }

    #[test]
    fn test_language_detection() {
        let detector = LanguageDetector::new();
        let lang = detector.detect("The quick brown fox jumps over the lazy dog");
        assert_eq!(lang, Some("en".to_string()));
    }

    #[test]
    fn test_readability_score() {
        let score = TextQualityAssessor::readability_score(
            "This is a simple sentence. It is easy to read.",
        );
        assert!(score > 0.0);
    }

    #[test]
    fn test_lexical_diversity() {
        let diversity = TextQualityAssessor::lexical_diversity("the the the the");
        assert_eq!(diversity, 0.25); // 1 unique word out of 4 total
    }

    #[test]
    fn test_syllable_count() {
        assert_eq!(count_syllables("hello"), 2);
        assert_eq!(count_syllables("world"), 1);
        assert_eq!(count_syllables("beautiful"), 3);
    }

    #[test]
    fn test_similarity_matrix_optimization() {
        let processor = QuickTextProcessor::new();
        let batch_processor = BatchTextProcessor::new(processor, 2);
        let texts = vec![
            "hello world".to_string(),
            "hello world".to_string(),
            "goodbye moon".to_string(),
        ];

        let matrix = batch_processor.similarity_matrix(&texts).unwrap();
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);
        assert_eq!(matrix[0][0], 1.0); // Self-similarity
        assert_eq!(matrix[0][1], 1.0); // Identical texts
        assert!(matrix[0][2] < 1.0); // Different texts
    }

    #[test]
    fn test_extract_sentences() {
        let text =
            "Dr. Smith went to the store. He bought milk! Did he remember eggs? Yes, he did.";
        let sentences = TextUtilities::extract_sentences(text);
        assert_eq!(sentences.len(), 4);
        assert!(sentences[0].contains("Dr. Smith"));
    }

    #[test]
    fn test_extract_keywords() {
        let text =
            "machine learning is a powerful artificial intelligence technique for data analysis";
        let keywords = TextUtilities::extract_keywords(text, 3);
        assert!(keywords.len() <= 3);
        assert!(!keywords.is_empty());
    }

    #[test]
    fn test_quick_clean() {
        let messy_text = "Hello\u{200B}world\u{00A0}with\u{FEFF}issues\r\n  multiple   spaces";
        let cleaned = TextUtilities::quick_clean(messy_text);
        assert!(!cleaned.contains('\u{200B}'));
        assert!(!cleaned.contains('\u{FEFF}'));
        assert!(!cleaned.contains('\r'));
        assert!(!cleaned.contains("  ")); // No double spaces
    }

    #[test]
    fn test_text_complexity() {
        let text = "This is a simple sentence. This is another sentence with more words.";
        let metrics = TextUtilities::text_complexity(text);
        assert!(metrics.contains_key("sentence_count"));
        assert!(metrics.contains_key("word_count"));
        assert!(metrics.contains_key("lexical_diversity"));
        assert_eq!(metrics["sentence_count"], 2.0);
    }
}
