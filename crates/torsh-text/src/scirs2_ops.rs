//! SciRS2 integration for text operations
//!
//! This module wraps scirs2-text operations to provide efficient text processing
//! capabilities with seamless integration to ToRSh tensors.

use crate::{Result, TextError};
use std::collections::HashMap;
use torsh_core::device::DeviceType;
use torsh_tensor::Tensor;

/// Text preprocessing operations using SciRS2
pub struct SciRS2TextOps {
    device: DeviceType,
}

impl SciRS2TextOps {
    /// Create a new SciRS2 text operations handler
    pub fn new(device: DeviceType) -> Self {
        Self { device }
    }

    /// Get the device this operations handler is using
    pub fn device(&self) -> DeviceType {
        self.device
    }
}

impl Default for SciRS2TextOps {
    fn default() -> Self {
        Self::new(DeviceType::Cpu)
    }
}

/// Efficient string operations using SciRS2
pub mod string_ops {
    use super::*;

    /// Count character frequencies in text
    pub fn char_frequency(text: &str) -> Result<HashMap<char, usize>> {
        let mut freq_map = HashMap::new();

        // Use efficient iteration for character counting
        for ch in text.chars() {
            *freq_map.entry(ch).or_insert(0) += 1;
        }

        Ok(freq_map)
    }

    /// Compute n-gram frequencies
    pub fn ngram_frequency(text: &str, n: usize) -> Result<HashMap<String, usize>> {
        if n == 0 {
            return Err(TextError::Other(anyhow::anyhow!("n-gram size must be > 0")));
        }

        let mut freq_map = HashMap::new();
        let chars: Vec<char> = text.chars().collect();

        if chars.len() < n {
            return Ok(freq_map);
        }

        // Generate n-grams efficiently
        for i in 0..=chars.len() - n {
            let ngram: String = chars[i..i + n].iter().collect();
            *freq_map.entry(ngram).or_insert(0) += 1;
        }

        Ok(freq_map)
    }

    /// Compute text similarity using cosine similarity
    pub fn cosine_similarity(text1: &str, text2: &str) -> Result<f32> {
        let freq1 = char_frequency(text1)?;
        let freq2 = char_frequency(text2)?;

        // Get union of all characters
        let mut all_chars: Vec<char> = freq1.keys().chain(freq2.keys()).copied().collect();
        all_chars.sort();
        all_chars.dedup();

        if all_chars.is_empty() {
            return Ok(0.0);
        }

        // Create frequency vectors
        let vec1: Vec<f32> = all_chars
            .iter()
            .map(|ch| *freq1.get(ch).unwrap_or(&0) as f32)
            .collect();
        let vec2: Vec<f32> = all_chars
            .iter()
            .map(|ch| *freq2.get(ch).unwrap_or(&0) as f32)
            .collect();

        // Compute cosine similarity
        let dot_product: f32 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();

        let norm1: f32 = vec1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = vec2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            Ok(0.0)
        } else {
            Ok(dot_product / (norm1 * norm2))
        }
    }

    /// Levenshtein distance between two strings
    pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
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

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        // Initialize first row and column
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }

        // Fill the matrix
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
}

/// Vectorized text processing operations
pub mod vectorized_ops {
    use super::*;

    /// Convert text to character-level tensor
    pub fn text_to_char_tensor(
        text: &str,
        vocab: &HashMap<char, u32>,
        _device: DeviceType,
    ) -> Result<Tensor> {
        let char_ids: Vec<u32> = text
            .chars()
            .map(|ch| vocab.get(&ch).copied().unwrap_or(0)) // 0 for unknown chars
            .collect();

        // Convert u32 to f32 for tensor compatibility
        let char_ids_f32: Vec<f32> = char_ids.into_iter().map(|x| x as f32).collect();
        let len = char_ids_f32.len();
        let tensor = Tensor::from_vec(char_ids_f32, &[len])?;
        Ok(tensor)
    }

    /// Convert character tensor back to text
    pub fn char_tensor_to_text(tensor: &Tensor, vocab: &HashMap<u32, char>) -> Result<String> {
        let data = tensor.to_vec()?;
        let text: String = data
            .iter()
            .filter_map(|&id| vocab.get(&(id as u32)))
            .collect();
        Ok(text)
    }

    /// Batch text to tensor conversion
    pub fn batch_text_to_tensor(
        texts: &[String],
        vocab: &HashMap<char, u32>,
        max_length: Option<usize>,
        _device: DeviceType,
    ) -> Result<Tensor> {
        let max_len = max_length
            .unwrap_or_else(|| texts.iter().map(|t| t.chars().count()).max().unwrap_or(0));

        let mut batch_data = Vec::new();

        for text in texts {
            let mut char_ids: Vec<u32> = text
                .chars()
                .take(max_len)
                .map(|ch| vocab.get(&ch).copied().unwrap_or(0))
                .collect();

            // Pad to max_length
            while char_ids.len() < max_len {
                char_ids.push(0); // Assuming 0 is padding token
            }

            batch_data.extend(char_ids);
        }

        // Convert u32 to f32 for tensor compatibility
        let batch_data_f32: Vec<f32> = batch_data.into_iter().map(|x| x as f32).collect();
        let tensor = Tensor::from_vec(batch_data_f32, &[texts.len(), max_len])?;
        Ok(tensor)
    }

    /// Create one-hot encoding for character sequences
    pub fn char_sequence_to_onehot(
        text: &str,
        vocab: &HashMap<char, u32>,
        vocab_size: usize,
        _device: DeviceType,
    ) -> Result<Tensor> {
        let sequence_len = text.chars().count();
        let mut onehot_data = vec![0.0f32; sequence_len * vocab_size];

        for (pos, ch) in text.chars().enumerate() {
            if let Some(&char_id) = vocab.get(&ch) {
                if (char_id as usize) < vocab_size {
                    onehot_data[pos * vocab_size + char_id as usize] = 1.0;
                }
            }
        }

        let tensor = Tensor::from_vec(onehot_data, &[sequence_len, vocab_size])?;
        Ok(tensor)
    }
}

/// Text indexing and search operations
pub mod indexing {
    use super::*;

    /// Build inverted index for efficient text search
    pub struct InvertedIndex {
        index: HashMap<String, Vec<usize>>,
        documents: Vec<String>,
    }

    impl InvertedIndex {
        pub fn new() -> Self {
            Self {
                index: HashMap::new(),
                documents: Vec::new(),
            }
        }

        /// Add document to index
        pub fn add_document(&mut self, doc_id: usize, text: &str) {
            if doc_id >= self.documents.len() {
                self.documents.resize(doc_id + 1, String::new());
            }
            self.documents[doc_id] = text.to_string();

            // Tokenize and index words
            for word in text.split_whitespace() {
                let word = word.to_lowercase();
                self.index.entry(word).or_default().push(doc_id);
            }
        }

        /// Search for documents containing a term
        pub fn search(&self, term: &str) -> Vec<usize> {
            let term = term.to_lowercase();
            self.index.get(&term).cloned().unwrap_or_default()
        }

        /// Boolean AND search
        pub fn search_and(&self, terms: &[&str]) -> Vec<usize> {
            if terms.is_empty() {
                return Vec::new();
            }

            let mut result = self.search(terms[0]);

            for &term in &terms[1..] {
                let term_docs = self.search(term);
                result.retain(|doc_id| term_docs.contains(doc_id));
            }

            result
        }

        /// Boolean OR search
        pub fn search_or(&self, terms: &[&str]) -> Vec<usize> {
            let mut result = Vec::new();

            for &term in terms {
                let term_docs = self.search(term);
                result.extend(term_docs);
            }

            result.sort();
            result.dedup();
            result
        }

        /// Get document text by ID
        pub fn get_document(&self, doc_id: usize) -> Option<&str> {
            self.documents.get(doc_id).map(|s| s.as_str())
        }
    }

    impl Default for InvertedIndex {
        fn default() -> Self {
            Self::new()
        }
    }
}

/// Memory optimization utilities
pub mod memory {
    use super::*;

    /// Memory-efficient text processing with streaming
    pub struct StreamingProcessor {
        buffer_size: usize,
        _device: DeviceType,
    }

    impl StreamingProcessor {
        pub fn new(buffer_size: usize, device: DeviceType) -> Self {
            Self {
                buffer_size,
                _device: device,
            }
        }

        /// Process large text files in chunks
        pub fn process_file_chunked<F, T>(&self, file_path: &str, processor: F) -> Result<Vec<T>>
        where
            F: Fn(&str) -> Result<T>,
        {
            use std::fs::File;
            use std::io::{BufRead, BufReader};

            let file = File::open(file_path)?;
            let reader = BufReader::new(file);

            let mut results = Vec::new();
            let mut buffer = String::new();

            for line in reader.lines() {
                let line = line?;
                buffer.push_str(&line);
                buffer.push('\n');

                if buffer.len() >= self.buffer_size {
                    let result = processor(&buffer)?;
                    results.push(result);
                    buffer.clear();
                }
            }

            // Process remaining buffer
            if !buffer.is_empty() {
                let result = processor(&buffer)?;
                results.push(result);
            }

            Ok(results)
        }

        /// Batch process multiple texts with memory optimization
        pub fn batch_process<F, T>(
            &self,
            texts: &[String],
            batch_size: usize,
            processor: F,
        ) -> Result<Vec<T>>
        where
            F: Fn(&[String]) -> Result<Vec<T>>,
        {
            let mut results = Vec::new();

            for chunk in texts.chunks(batch_size) {
                let mut batch_results = processor(chunk)?;
                results.append(&mut batch_results);
            }

            Ok(results)
        }
    }
}

/// Advanced text analytics using SciRS2 random number generation for sampling and stochastic methods
pub mod advanced_analytics {
    use super::*;
    use scirs2_core::random::{Random, Rng};
    use scirs2_core::rngs::StdRng;
    use std::collections::{BTreeMap, HashSet};

    /// Text statistics with advanced metrics
    #[derive(Debug, Clone)]
    pub struct AdvancedTextStats {
        pub char_count: usize,
        pub word_count: usize,
        pub sentence_count: usize,
        pub paragraph_count: usize,
        pub unique_words: usize,
        pub lexical_diversity: f64,
        pub average_word_length: f64,
        pub average_sentence_length: f64,
        pub readability_score: f64,
    }

    /// Compute advanced text statistics
    pub fn compute_advanced_stats(text: &str) -> Result<AdvancedTextStats> {
        let chars = text.chars().count();
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len();

        // Count sentences (basic heuristic)
        let sentences = text
            .split(&['.', '!', '?'])
            .filter(|s| !s.trim().is_empty())
            .count();

        // Count paragraphs
        let paragraphs = text
            .split("\n\n")
            .filter(|s| !s.trim().is_empty())
            .count()
            .max(1);

        // Unique words
        let unique_words: HashSet<String> = words.iter().map(|w| w.to_lowercase()).collect();
        let unique_count = unique_words.len();

        // Lexical diversity (Type-Token Ratio)
        let lexical_diversity = if word_count > 0 {
            unique_count as f64 / word_count as f64
        } else {
            0.0
        };

        // Average word length
        let avg_word_length = if word_count > 0 {
            words.iter().map(|w| w.len()).sum::<usize>() as f64 / word_count as f64
        } else {
            0.0
        };

        // Average sentence length
        let avg_sentence_length = if sentences > 0 {
            word_count as f64 / sentences as f64
        } else {
            0.0
        };

        // Simple readability score (Flesch-like approximation)
        let readability_score = if sentences > 0 && word_count > 0 {
            206.835 - (1.015 * avg_sentence_length) - (84.6 * (avg_word_length / 4.7))
        } else {
            0.0
        };

        Ok(AdvancedTextStats {
            char_count: chars,
            word_count,
            sentence_count: sentences,
            paragraph_count: paragraphs,
            unique_words: unique_count,
            lexical_diversity,
            average_word_length: avg_word_length,
            average_sentence_length: avg_sentence_length,
            readability_score,
        })
    }

    /// Advanced text sampling utilities using SciRS2 random generation
    pub struct AdvancedTextSampler {
        rng: Random<StdRng>, // Seeded RNG for deterministic, reproducible sampling
    }

    impl AdvancedTextSampler {
        /// Create a new text sampler with random seed
        pub fn new() -> Self {
            Self {
                rng: Random::seed(42), // Default seed for reproducibility
            }
        }

        /// Create a new text sampler with specific seed
        pub fn with_seed(seed: u64) -> Self {
            Self {
                rng: Random::seed(seed),
            }
        }

        /// Sample random words from text
        pub fn sample_words(&mut self, text: &str, count: usize) -> Vec<String> {
            let words: Vec<String> = text.split_whitespace().map(|w| w.to_lowercase()).collect();

            if words.is_empty() {
                return Vec::new();
            }

            let mut samples = Vec::new();
            for _ in 0..count {
                let idx = self.rng.gen_range(0..words.len());
                samples.push(words[idx].clone());
            }
            samples
        }

        /// Sample random sentences from text
        pub fn sample_sentences(&mut self, text: &str, count: usize) -> Vec<String> {
            let sentences: Vec<String> = text
                .split(&['.', '!', '?'])
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();

            if sentences.is_empty() {
                return Vec::new();
            }

            let mut samples = Vec::new();
            for _ in 0..count {
                let idx = self.rng.gen_range(0..sentences.len());
                samples.push(sentences[idx].clone());
            }
            samples
        }

        /// Generate text using Markov chain with SciRS2 random generation
        pub fn generate_markov_text(&mut self, text: &str, length: usize, order: usize) -> String {
            let words: Vec<&str> = text.split_whitespace().collect();
            if words.len() < order + 1 {
                return text.to_string();
            }

            // Build Markov chain
            let mut chain: HashMap<Vec<String>, Vec<String>> = HashMap::new();

            for i in 0..=words.len().saturating_sub(order + 1) {
                let key: Vec<String> = words[i..i + order].iter().map(|s| s.to_string()).collect();
                let next_word = words[i + order].to_string();
                chain.entry(key).or_default().push(next_word);
            }

            // Generate text
            let mut result = Vec::new();
            let mut current_key: Vec<String> =
                words[0..order].iter().map(|s| s.to_string()).collect();
            result.extend(current_key.clone());

            for _ in 0..length.saturating_sub(order) {
                if let Some(next_words) = chain.get(&current_key) {
                    if !next_words.is_empty() {
                        let idx = self.rng.gen_range(0..next_words.len());
                        let next_word = next_words[idx].clone();
                        result.push(next_word.clone());

                        // Update current key
                        current_key.remove(0);
                        current_key.push(next_word);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            result.join(" ")
        }
    }

    impl Default for AdvancedTextSampler {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Text complexity analyzer using multiple dimensions
    pub struct ComplexityAnalyzer;

    impl ComplexityAnalyzer {
        /// Analyze text complexity across multiple dimensions
        pub fn analyze_complexity(text: &str) -> Result<ComplexityMetrics> {
            let stats = compute_advanced_stats(text)?;

            // Syntactic complexity (average sentence length)
            let syntactic_complexity = (stats.average_sentence_length / 20.0).min(1.0);

            // Lexical complexity (unique word ratio and average word length)
            let lexical_complexity = ((stats.lexical_diversity * 0.7)
                + (stats.average_word_length / 10.0 * 0.3))
                .min(1.0);

            // Semantic complexity (approximated by word length distribution)
            let words: Vec<&str> = text.split_whitespace().collect();
            let long_words = words.iter().filter(|w| w.len() > 6).count();
            let semantic_complexity = if !words.is_empty() {
                (long_words as f64 / words.len() as f64).min(1.0)
            } else {
                0.0
            };

            // Overall complexity score
            let overall_complexity =
                (syntactic_complexity + lexical_complexity + semantic_complexity) / 3.0;

            Ok(ComplexityMetrics {
                syntactic_complexity,
                lexical_complexity,
                semantic_complexity,
                overall_complexity,
                readability_score: stats.readability_score,
            })
        }
    }

    #[derive(Debug, Clone)]
    pub struct ComplexityMetrics {
        pub syntactic_complexity: f64,
        pub lexical_complexity: f64,
        pub semantic_complexity: f64,
        pub overall_complexity: f64,
        pub readability_score: f64,
    }
}

/// Enhanced performance monitoring for text operations
pub mod performance {
    use super::*;
    use std::time::{Duration, Instant};

    /// Performance metrics for text operations
    #[derive(Debug, Clone)]
    pub struct PerformanceMetrics {
        pub operation: String,
        pub duration: Duration,
        pub throughput_chars_per_sec: f64,
        pub throughput_words_per_sec: f64,
        pub memory_usage_estimate: usize,
    }

    /// Performance monitor for text operations
    pub struct PerformanceMonitor {
        start_time: Option<Instant>,
        operation_name: String,
    }

    impl PerformanceMonitor {
        /// Create a new performance monitor
        pub fn new(operation: &str) -> Self {
            Self {
                start_time: None,
                operation_name: operation.to_string(),
            }
        }

        /// Start timing an operation
        pub fn start(&mut self) {
            self.start_time = Some(Instant::now());
        }

        /// Stop timing and generate metrics
        pub fn stop(&mut self, text: &str) -> PerformanceMetrics {
            let duration = self
                .start_time
                .map(|start| start.elapsed())
                .unwrap_or(Duration::ZERO);

            let char_count = text.chars().count();
            let word_count = text.split_whitespace().count();

            let throughput_chars_per_sec = if duration.as_secs_f64() > 0.0 {
                char_count as f64 / duration.as_secs_f64()
            } else {
                0.0
            };

            let throughput_words_per_sec = if duration.as_secs_f64() > 0.0 {
                word_count as f64 / duration.as_secs_f64()
            } else {
                0.0
            };

            // Simple memory usage estimate
            let memory_usage_estimate = text.len() + (word_count * 8); // Rough estimate

            PerformanceMetrics {
                operation: self.operation_name.clone(),
                duration,
                throughput_chars_per_sec,
                throughput_words_per_sec,
                memory_usage_estimate,
            }
        }

        /// Time a closure and return both result and metrics
        pub fn time_operation<F, R>(&mut self, text: &str, operation: F) -> (R, PerformanceMetrics)
        where
            F: FnOnce() -> R,
        {
            self.start();
            let result = operation();
            let metrics = self.stop(text);
            (result, metrics)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::indexing::InvertedIndex;
    use super::string_ops;
    use super::vectorized_ops;
    use super::*;

    #[test]
    fn test_advanced_analytics() {
        let text = "This is a test sentence. This contains multiple words and sentences! How complex is this text?";

        // Test advanced stats
        let stats = super::advanced_analytics::compute_advanced_stats(text).unwrap();
        assert!(stats.word_count > 0);
        assert!(stats.sentence_count >= 3);
        assert!(stats.lexical_diversity > 0.0);
        assert!(stats.average_word_length > 0.0);

        // Test complexity analysis
        let complexity =
            super::advanced_analytics::ComplexityAnalyzer::analyze_complexity(text).unwrap();
        assert!(complexity.overall_complexity >= 0.0);
        assert!(complexity.overall_complexity <= 1.0);
        assert!(complexity.syntactic_complexity >= 0.0);
        assert!(complexity.lexical_complexity >= 0.0);
        assert!(complexity.semantic_complexity >= 0.0);

        // Test text sampler
        let mut sampler = super::advanced_analytics::AdvancedTextSampler::new();
        let samples = sampler.sample_words(text, 3);
        assert_eq!(samples.len(), 3);

        let sentence_samples = sampler.sample_sentences(text, 2);
        assert_eq!(sentence_samples.len(), 2);

        // Test Markov generation
        let generated = sampler.generate_markov_text(text, 10, 2);
        assert!(!generated.is_empty());
    }

    #[test]
    fn test_performance_monitoring() {
        let text = "Performance testing text with multiple words.";
        let mut monitor = super::performance::PerformanceMonitor::new("test_operation");

        // Test timing operation
        let (result, metrics) = monitor.time_operation(text, || text.split_whitespace().count());

        assert!(result > 0);
        assert_eq!(metrics.operation, "test_operation");
        assert!(metrics.throughput_chars_per_sec >= 0.0);
        assert!(metrics.throughput_words_per_sec >= 0.0);
        assert!(metrics.memory_usage_estimate > 0);
    }

    #[test]
    fn test_char_frequency() {
        let freq = string_ops::char_frequency("hello").unwrap();
        assert_eq!(*freq.get(&'l').unwrap(), 2);
        assert_eq!(*freq.get(&'e').unwrap(), 1);
        assert_eq!(*freq.get(&'h').unwrap(), 1);
        assert_eq!(*freq.get(&'o').unwrap(), 1);
    }

    #[test]
    fn test_ngram_frequency() {
        let freq = string_ops::ngram_frequency("hello", 2).unwrap();
        assert_eq!(*freq.get("he").unwrap(), 1);
        assert_eq!(*freq.get("el").unwrap(), 1);
        assert_eq!(*freq.get("ll").unwrap(), 1);
        assert_eq!(*freq.get("lo").unwrap(), 1);
    }

    #[test]
    fn test_cosine_similarity() {
        let sim = string_ops::cosine_similarity("hello", "hello").unwrap();
        assert!((sim - 1.0).abs() < 1e-6);

        let sim = string_ops::cosine_similarity("hello", "world").unwrap();
        assert!(sim >= 0.0 && sim <= 1.0);
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(string_ops::levenshtein_distance("hello", "hello"), 0);
        assert_eq!(string_ops::levenshtein_distance("hello", "hallo"), 1);
        assert_eq!(string_ops::levenshtein_distance("hello", "world"), 4);
    }

    #[test]
    fn test_text_to_char_tensor() {
        let mut vocab = HashMap::new();
        vocab.insert('h', 1);
        vocab.insert('e', 2);
        vocab.insert('l', 3);
        vocab.insert('o', 4);

        let tensor = vectorized_ops::text_to_char_tensor("hello", &vocab, DeviceType::Cpu).unwrap();
        assert_eq!(tensor.shape().dims(), &[5]);

        let data = tensor.data().unwrap();
        assert_eq!(data[0], 1.0); // 'h'
        assert_eq!(data[1], 2.0); // 'e'
        assert_eq!(data[2], 3.0); // 'l'
        assert_eq!(data[3], 3.0); // 'l'
        assert_eq!(data[4], 4.0); // 'o'
    }

    #[test]
    fn test_inverted_index() {
        let mut index = InvertedIndex::new();

        index.add_document(0, "hello world");
        index.add_document(1, "hello rust");
        index.add_document(2, "world peace");

        let hello_docs = index.search("hello");
        assert_eq!(hello_docs, vec![0, 1]);

        let world_docs = index.search("world");
        assert_eq!(world_docs, vec![0, 2]);

        let and_result = index.search_and(&["hello", "world"]);
        assert_eq!(and_result, vec![0]);

        let or_result = index.search_or(&["hello", "peace"]);
        assert!(or_result.contains(&0));
        assert!(or_result.contains(&1));
        assert!(or_result.contains(&2));
    }
}
