//! Language Model-based Fluency Analysis
//!
//! This module provides comprehensive language model-based fluency evaluation
//! including perplexity, surprisal, entropy, and n-gram probability analysis.

use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_core::random::{rng, Random};
use std::collections::HashMap;

/// Configuration for language model analysis
#[derive(Debug, Clone)]
pub struct LanguageModelConfig {
    /// Weight for perplexity score
    pub perplexity_weight: f64,
    /// Weight for likelihood score
    pub likelihood_weight: f64,
    /// Weight for surprisal score
    pub surprisal_weight: f64,
    /// Weight for entropy score
    pub entropy_weight: f64,
    /// Penalty for out-of-vocabulary words
    pub oov_penalty_factor: f64,
    /// Maximum n-gram size to consider
    pub max_ngram_size: usize,
    /// Smoothing parameter for probability calculations
    pub smoothing_factor: f64,
    /// Minimum frequency threshold for vocabulary
    pub min_word_frequency: f64,
    /// Enable advanced smoothing techniques
    pub enable_advanced_smoothing: bool,
    /// Context window size for language modeling
    pub context_window: usize,
}

impl Default for LanguageModelConfig {
    fn default() -> Self {
        Self {
            perplexity_weight: 0.30,
            likelihood_weight: 0.25,
            surprisal_weight: 0.20,
            entropy_weight: 0.15,
            oov_penalty_factor: 0.1,
            max_ngram_size: 3,
            smoothing_factor: 0.001,
            min_word_frequency: 0.01,
            enable_advanced_smoothing: true,
            context_window: 5,
        }
    }
}

/// Results of language model fluency analysis
#[derive(Debug, Clone)]
pub struct LanguageModelFluencyResult {
    /// Perplexity-based fluency score (normalized)
    pub perplexity_score: f64,
    /// Likelihood-based fluency score
    pub likelihood_score: f64,
    /// Surprisal-based fluency score
    pub surprisal_score: f64,
    /// Entropy-based fluency score
    pub entropy_score: f64,
    /// Probability mass of the text
    pub probability_mass: f64,
    /// N-gram probability scores by n-gram size
    pub ngram_probabilities: HashMap<usize, f64>,
    /// Out-of-vocabulary penalty applied
    pub oov_penalty: f64,
    /// Final smoothed language model score
    pub smoothed_score: f64,
    /// Detailed probability distribution
    pub probability_distribution: ProbabilityDistribution,
    /// Language model confidence metrics
    pub confidence_metrics: ConfidenceMetrics,
}

/// Probability distribution analysis
#[derive(Debug, Clone)]
pub struct ProbabilityDistribution {
    /// Word-level probability scores
    pub word_probabilities: Vec<f64>,
    /// Sentence-level probability scores
    pub sentence_probabilities: Vec<f64>,
    /// Probability variance across text
    pub probability_variance: f64,
    /// Probability skewness measure
    pub probability_skewness: f64,
    /// Cross-entropy with reference model
    pub cross_entropy: f64,
    /// Kullback-Leibler divergence from expected distribution
    pub kl_divergence: f64,
}

/// Language model confidence metrics
#[derive(Debug, Clone)]
pub struct ConfidenceMetrics {
    /// Overall confidence in language model predictions
    pub overall_confidence: f64,
    /// Confidence distribution across sentences
    pub sentence_confidence: Vec<f64>,
    /// Uncertainty estimation
    pub uncertainty_score: f64,
    /// Calibration score of probability estimates
    pub calibration_score: f64,
    /// Reliability measure
    pub reliability_score: f64,
}

/// Language model token information
#[derive(Debug, Clone)]
pub struct TokenInfo {
    /// Token text
    pub token: String,
    /// Token probability
    pub probability: f64,
    /// Token surprisal
    pub surprisal: f64,
    /// Token entropy contribution
    pub entropy_contribution: f64,
    /// Context relevance score
    pub context_relevance: f64,
}

/// N-gram analysis results
#[derive(Debug, Clone)]
pub struct NgramAnalysis {
    /// Unigram probabilities and frequencies
    pub unigrams: HashMap<String, f64>,
    /// Bigram probabilities and frequencies
    pub bigrams: HashMap<String, f64>,
    /// Trigram probabilities and frequencies
    pub trigrams: HashMap<String, f64>,
    /// Higher-order n-gram probabilities
    pub higher_ngrams: HashMap<usize, HashMap<String, f64>>,
    /// N-gram coverage statistics
    pub coverage_stats: NgramCoverageStats,
}

/// N-gram coverage statistics
#[derive(Debug, Clone)]
pub struct NgramCoverageStats {
    /// Percentage of text covered by known n-grams
    pub coverage_percentage: f64,
    /// Out-of-vocabulary n-gram count
    pub oov_ngram_count: usize,
    /// Total n-gram count
    pub total_ngram_count: usize,
    /// Average n-gram frequency
    pub average_frequency: f64,
    /// Frequency distribution statistics
    pub frequency_distribution: HashMap<String, usize>,
}

/// Language model analyzer
pub struct LanguageModelAnalyzer {
    config: LanguageModelConfig,
    word_frequencies: HashMap<String, f64>,
    ngram_models: HashMap<usize, HashMap<String, f64>>,
    vocabulary_size: usize,
    token_info_cache: HashMap<String, TokenInfo>,
}

impl LanguageModelAnalyzer {
    /// Create a new language model analyzer
    pub fn new(config: LanguageModelConfig) -> Self {
        Self {
            config,
            word_frequencies: HashMap::new(),
            ngram_models: HashMap::new(),
            vocabulary_size: 0,
            token_info_cache: HashMap::new(),
        }
    }

    /// Create analyzer with default configuration
    pub fn with_default_config() -> Self {
        Self::new(LanguageModelConfig::default())
    }

    /// Initialize with word frequency data
    pub fn with_word_frequencies(mut self, frequencies: HashMap<String, f64>) -> Self {
        self.vocabulary_size = frequencies.len();
        self.word_frequencies = frequencies;
        self.build_ngram_models();
        self
    }

    /// Build n-gram language models from word frequencies
    fn build_ngram_models(&mut self) {
        // Initialize n-gram models up to max_ngram_size
        for n in 1..=self.config.max_ngram_size {
            self.ngram_models.insert(n, HashMap::new());
        }

        // Build simple n-gram models based on word frequencies
        // In a real implementation, this would use proper training data
        for (word, freq) in &self.word_frequencies {
            self.ngram_models
                .get_mut(&1)
                .unwrap()
                .insert(word.clone(), *freq);
        }
    }

    /// Analyze language model fluency for a text
    pub fn analyze_language_model_fluency(
        &self,
        sentences: &[String],
    ) -> LanguageModelFluencyResult {
        let mut total_log_prob = 0.0;
        let mut total_surprisal = 0.0;
        let mut total_entropy = 0.0;
        let mut total_words = 0;
        let mut oov_count = 0;
        let mut word_probabilities = Vec::new();
        let mut sentence_probabilities = Vec::new();

        let mut ngram_probabilities = HashMap::new();
        for n in 1..=self.config.max_ngram_size {
            ngram_probabilities.insert(n, 0.0);
        }

        for sentence in sentences {
            let words = self.tokenize_sentence(sentence);
            total_words += words.len();
            let mut sentence_log_prob = 0.0;

            for word in &words {
                let freq = self
                    .word_frequencies
                    .get(word)
                    .unwrap_or(&self.config.smoothing_factor);

                if *freq < self.config.min_word_frequency {
                    oov_count += 1;
                }

                let prob = freq.max(1e-10);
                let log_prob = prob.ln();
                let surprisal = -prob.log2();
                let entropy = -prob * prob.log2();

                total_log_prob += log_prob;
                total_surprisal += surprisal;
                total_entropy += entropy;
                sentence_log_prob += log_prob;

                word_probabilities.push(prob);
            }

            sentence_probabilities.push(sentence_log_prob.exp());

            // Calculate n-gram probabilities
            for n in 1..=self.config.max_ngram_size {
                let ngram_prob = self.calculate_ngram_probability(&words, n);
                *ngram_probabilities.get_mut(&n).unwrap() += ngram_prob;
            }
        }

        // Normalize n-gram probabilities
        for (_, prob) in &mut ngram_probabilities {
            if !sentences.is_empty() {
                *prob /= sentences.len() as f64;
            }
        }

        // Calculate final scores
        let perplexity_score = if total_words > 0 {
            let perplexity = (-total_log_prob / total_words as f64).exp();
            1.0 / perplexity.min(1000.0)
        } else {
            0.0
        };

        let likelihood_score =
            1.0 / (1.0 + (-total_log_prob / total_words.max(1) as f64).exp() / 100.0);

        let surprisal_score = if total_words > 0 {
            1.0 / (1.0 + total_surprisal / total_words as f64)
        } else {
            0.0
        };

        let entropy_score = if total_words > 0 {
            let avg_entropy = total_entropy / total_words as f64;
            1.0 / (1.0 + avg_entropy)
        } else {
            0.0
        };

        let probability_mass = total_log_prob.exp();

        let oov_penalty = if total_words > 0 {
            1.0 - (oov_count as f64 / total_words as f64) * self.config.oov_penalty_factor
        } else {
            1.0
        };

        let smoothed_score = self.calculate_smoothed_score(
            likelihood_score,
            surprisal_score,
            entropy_score,
            oov_penalty,
        );

        // Calculate probability distribution metrics
        let probability_distribution =
            self.calculate_probability_distribution(&word_probabilities, &sentence_probabilities);

        // Calculate confidence metrics
        let confidence_metrics = self.calculate_confidence_metrics(
            &word_probabilities,
            &sentence_probabilities,
            total_words,
        );

        LanguageModelFluencyResult {
            perplexity_score,
            likelihood_score,
            surprisal_score,
            entropy_score,
            probability_mass,
            ngram_probabilities,
            oov_penalty,
            smoothed_score,
            probability_distribution,
            confidence_metrics,
        }
    }

    /// Tokenize sentence into words
    pub fn tokenize_sentence(&self, sentence: &str) -> Vec<String> {
        sentence
            .split_whitespace()
            .map(|word| {
                word.trim_matches(|c: char| !c.is_alphabetic())
                    .to_lowercase()
            })
            .filter(|word| !word.is_empty())
            .collect()
    }

    /// Calculate n-gram probability for a sequence of words
    pub fn calculate_ngram_probability(&self, words: &[String], n: usize) -> f64 {
        if words.len() < n || n == 0 {
            return 0.0;
        }

        let mut total_prob = 0.0;
        let mut ngram_count = 0;

        for i in 0..=words.len().saturating_sub(n) {
            let ngram = &words[i..i + n];

            // Calculate n-gram probability
            let ngram_prob = if n == 1 {
                // Unigram probability
                self.word_frequencies
                    .get(&ngram[0])
                    .unwrap_or(&self.config.smoothing_factor)
                    .clone()
            } else {
                // Higher-order n-gram probability (simplified)
                ngram
                    .iter()
                    .map(|word| {
                        self.word_frequencies
                            .get(word)
                            .unwrap_or(&self.config.smoothing_factor)
                    })
                    .product::<f64>()
                    .powf(1.0 / n as f64)
            };

            total_prob += ngram_prob;
            ngram_count += 1;
        }

        if ngram_count > 0 {
            total_prob / ngram_count as f64
        } else {
            0.0
        }
    }

    /// Calculate smoothed language model score
    fn calculate_smoothed_score(
        &self,
        likelihood: f64,
        surprisal: f64,
        entropy: f64,
        oov_penalty: f64,
    ) -> f64 {
        if self.config.enable_advanced_smoothing {
            // Advanced smoothing with multiple components
            let weighted_score = likelihood * self.config.likelihood_weight
                + surprisal * self.config.surprisal_weight
                + entropy * self.config.entropy_weight;
            weighted_score * oov_penalty
        } else {
            // Simple smoothing
            likelihood * oov_penalty * 0.8 + surprisal * 0.2
        }
    }

    /// Calculate probability distribution metrics
    fn calculate_probability_distribution(
        &self,
        word_probs: &[f64],
        sentence_probs: &[f64],
    ) -> ProbabilityDistribution {
        let mean_word_prob = if !word_probs.is_empty() {
            word_probs.iter().sum::<f64>() / word_probs.len() as f64
        } else {
            0.0
        };

        let probability_variance = if word_probs.len() > 1 {
            let variance = word_probs
                .iter()
                .map(|&p| (p - mean_word_prob).powi(2))
                .sum::<f64>()
                / (word_probs.len() - 1) as f64;
            variance
        } else {
            0.0
        };

        let probability_skewness = self.calculate_skewness(word_probs, mean_word_prob);

        let cross_entropy = if !word_probs.is_empty() {
            -word_probs.iter().map(|&p| p * p.log2()).sum::<f64>() / word_probs.len() as f64
        } else {
            0.0
        };

        let kl_divergence = self.calculate_kl_divergence(word_probs);

        ProbabilityDistribution {
            word_probabilities: word_probs.to_vec(),
            sentence_probabilities: sentence_probs.to_vec(),
            probability_variance,
            probability_skewness,
            cross_entropy,
            kl_divergence,
        }
    }

    /// Calculate confidence metrics
    fn calculate_confidence_metrics(
        &self,
        word_probs: &[f64],
        sentence_probs: &[f64],
        total_words: usize,
    ) -> ConfidenceMetrics {
        let overall_confidence = if !word_probs.is_empty() {
            let mean_prob = word_probs.iter().sum::<f64>() / word_probs.len() as f64;
            (mean_prob * 2.0).min(1.0)
        } else {
            0.0
        };

        let sentence_confidence: Vec<f64> = sentence_probs
            .iter()
            .map(|&p| (p * 10.0).min(1.0))
            .collect();

        let uncertainty_score = if !word_probs.is_empty() {
            let entropy = -word_probs
                .iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| p * p.log2())
                .sum::<f64>();
            1.0 - (entropy / (total_words as f64).log2()).min(1.0)
        } else {
            0.0
        };

        let calibration_score = self.calculate_calibration_score(word_probs);
        let reliability_score = self.calculate_reliability_score(word_probs, sentence_probs);

        ConfidenceMetrics {
            overall_confidence,
            sentence_confidence,
            uncertainty_score,
            calibration_score,
            reliability_score,
        }
    }

    /// Calculate skewness of probability distribution
    fn calculate_skewness(&self, values: &[f64], mean: f64) -> f64 {
        if values.len() < 3 {
            return 0.0;
        }

        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

        if variance == 0.0 {
            return 0.0;
        }

        let std_dev = variance.sqrt();
        let skewness = values
            .iter()
            .map(|&x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>()
            / values.len() as f64;

        skewness
    }

    /// Calculate KL divergence from uniform distribution
    fn calculate_kl_divergence(&self, probabilities: &[f64]) -> f64 {
        if probabilities.is_empty() {
            return 0.0;
        }

        let uniform_prob = 1.0 / probabilities.len() as f64;
        probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * (p / uniform_prob).log2())
            .sum()
    }

    /// Calculate calibration score
    fn calculate_calibration_score(&self, probabilities: &[f64]) -> f64 {
        // Simplified calibration score based on probability consistency
        if probabilities.is_empty() {
            return 0.0;
        }

        let mean_prob = probabilities.iter().sum::<f64>() / probabilities.len() as f64;
        let variance = probabilities
            .iter()
            .map(|&p| (p - mean_prob).powi(2))
            .sum::<f64>()
            / probabilities.len() as f64;

        // Well-calibrated models have consistent probabilities
        1.0 / (1.0 + variance * 10.0)
    }

    /// Calculate reliability score
    fn calculate_reliability_score(&self, word_probs: &[f64], sentence_probs: &[f64]) -> f64 {
        if word_probs.is_empty() || sentence_probs.is_empty() {
            return 0.0;
        }

        // Reliability based on consistency between word and sentence level probabilities
        let word_mean = word_probs.iter().sum::<f64>() / word_probs.len() as f64;
        let sentence_mean = sentence_probs.iter().sum::<f64>() / sentence_probs.len() as f64;

        let consistency = 1.0 - (word_mean - sentence_mean).abs();
        consistency.max(0.0)
    }

    /// Analyze detailed n-gram patterns
    pub fn analyze_ngram_patterns(&self, sentences: &[String]) -> NgramAnalysis {
        let mut unigrams = HashMap::new();
        let mut bigrams = HashMap::new();
        let mut trigrams = HashMap::new();
        let mut higher_ngrams = HashMap::new();

        let mut total_ngrams = 0;
        let mut oov_ngrams = 0;

        for sentence in sentences {
            let words = self.tokenize_sentence(sentence);

            // Collect unigrams
            for word in &words {
                let count = unigrams.entry(word.clone()).or_insert(0.0);
                *count += 1.0;
                total_ngrams += 1;

                if !self.word_frequencies.contains_key(word) {
                    oov_ngrams += 1;
                }
            }

            // Collect bigrams
            for window in words.windows(2) {
                let bigram = format!("{} {}", window[0], window[1]);
                let count = bigrams.entry(bigram).or_insert(0.0);
                *count += 1.0;
            }

            // Collect trigrams
            for window in words.windows(3) {
                let trigram = format!("{} {} {}", window[0], window[1], window[2]);
                let count = trigrams.entry(trigram).or_insert(0.0);
                *count += 1.0;
            }

            // Collect higher-order n-grams
            for n in 4..=self.config.max_ngram_size {
                if words.len() >= n {
                    let ngram_map = higher_ngrams.entry(n).or_insert_with(HashMap::new);
                    for window in words.windows(n) {
                        let ngram = window.join(" ");
                        let count = ngram_map.entry(ngram).or_insert(0.0);
                        *count += 1.0;
                    }
                }
            }
        }

        // Calculate coverage statistics
        let coverage_percentage = if total_ngrams > 0 {
            ((total_ngrams - oov_ngrams) as f64 / total_ngrams as f64) * 100.0
        } else {
            0.0
        };

        let average_frequency = if !unigrams.is_empty() {
            unigrams.values().sum::<f64>() / unigrams.len() as f64
        } else {
            0.0
        };

        let mut frequency_distribution = HashMap::new();
        for (ngram, freq) in &unigrams {
            let freq_bucket = format!("freq_{}", (freq.log10().max(0.0) as usize));
            *frequency_distribution.entry(freq_bucket).or_insert(0) += 1;
        }

        let coverage_stats = NgramCoverageStats {
            coverage_percentage,
            oov_ngram_count: oov_ngrams,
            total_ngram_count: total_ngrams,
            average_frequency,
            frequency_distribution,
        };

        NgramAnalysis {
            unigrams,
            bigrams,
            trigrams,
            higher_ngrams,
            coverage_stats,
        }
    }

    /// Get token information for a word
    pub fn get_token_info(&self, token: &str) -> TokenInfo {
        if let Some(cached) = self.token_info_cache.get(token) {
            return cached.clone();
        }

        let probability = self
            .word_frequencies
            .get(token)
            .unwrap_or(&self.config.smoothing_factor)
            .clone();

        let surprisal = -probability.log2();
        let entropy_contribution = -probability * probability.log2();
        let context_relevance = self.calculate_context_relevance(token);

        TokenInfo {
            token: token.to_string(),
            probability,
            surprisal,
            entropy_contribution,
            context_relevance,
        }
    }

    /// Calculate context relevance for a token
    fn calculate_context_relevance(&self, token: &str) -> f64 {
        // Simplified context relevance based on frequency and vocabulary size
        let freq = self
            .word_frequencies
            .get(token)
            .unwrap_or(&self.config.smoothing_factor);
        let relative_freq = freq / self.vocabulary_size.max(1) as f64;
        (relative_freq * 100.0).min(1.0)
    }

    /// Update word frequencies with new data
    pub fn update_frequencies(&mut self, new_frequencies: HashMap<String, f64>) {
        for (word, freq) in new_frequencies {
            self.word_frequencies.insert(word, freq);
        }
        self.vocabulary_size = self.word_frequencies.len();
        self.build_ngram_models();
        self.token_info_cache.clear(); // Clear cache after update
    }

    /// Get current vocabulary statistics
    pub fn get_vocabulary_stats(&self) -> VocabularyStats {
        let total_words = self.word_frequencies.len();
        let total_frequency: f64 = self.word_frequencies.values().sum();
        let average_frequency = if total_words > 0 {
            total_frequency / total_words as f64
        } else {
            0.0
        };

        let high_freq_words = self
            .word_frequencies
            .values()
            .filter(|&&freq| freq > self.config.min_word_frequency * 10.0)
            .count();

        let low_freq_words = self
            .word_frequencies
            .values()
            .filter(|&&freq| freq <= self.config.min_word_frequency)
            .count();

        VocabularyStats {
            total_words,
            average_frequency,
            high_frequency_words: high_freq_words,
            low_frequency_words: low_freq_words,
            vocabulary_coverage: (high_freq_words as f64 / total_words.max(1) as f64) * 100.0,
        }
    }
}

/// Vocabulary statistics
#[derive(Debug, Clone)]
pub struct VocabularyStats {
    /// Total number of unique words in vocabulary
    pub total_words: usize,
    /// Average word frequency
    pub average_frequency: f64,
    /// Number of high-frequency words
    pub high_frequency_words: usize,
    /// Number of low-frequency words
    pub low_frequency_words: usize,
    /// Percentage of vocabulary that is high-frequency
    pub vocabulary_coverage: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_model_analyzer_creation() {
        let config = LanguageModelConfig::default();
        let analyzer = LanguageModelAnalyzer::new(config);

        assert_eq!(analyzer.config.max_ngram_size, 3);
        assert_eq!(analyzer.vocabulary_size, 0);
    }

    #[test]
    fn test_tokenization() {
        let analyzer = LanguageModelAnalyzer::with_default_config();
        let tokens = analyzer.tokenize_sentence("Hello, world! This is a test.");

        assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test"]);
    }

    #[test]
    fn test_ngram_probability_calculation() {
        let mut frequencies = HashMap::new();
        frequencies.insert("hello".to_string(), 0.1);
        frequencies.insert("world".to_string(), 0.05);
        frequencies.insert("test".to_string(), 0.02);

        let analyzer =
            LanguageModelAnalyzer::with_default_config().with_word_frequencies(frequencies);

        let words = vec!["hello".to_string(), "world".to_string()];
        let unigram_prob = analyzer.calculate_ngram_probability(&words, 1);
        let bigram_prob = analyzer.calculate_ngram_probability(&words, 2);

        assert!(unigram_prob > 0.0);
        assert!(bigram_prob > 0.0);
        assert!(bigram_prob < unigram_prob); // Bigrams typically have lower probability
    }

    #[test]
    fn test_fluency_analysis() {
        let mut frequencies = HashMap::new();
        frequencies.insert("the".to_string(), 0.1);
        frequencies.insert("quick".to_string(), 0.05);
        frequencies.insert("brown".to_string(), 0.03);
        frequencies.insert("fox".to_string(), 0.02);

        let analyzer =
            LanguageModelAnalyzer::with_default_config().with_word_frequencies(frequencies);

        let sentences = vec!["The quick brown fox".to_string()];
        let result = analyzer.analyze_language_model_fluency(&sentences);

        assert!(result.overall_confidence >= 0.0);
        assert!(result.overall_confidence <= 1.0);
        assert!(result.likelihood_score >= 0.0);
        assert!(result.perplexity_score >= 0.0);
        assert!(!result.ngram_probabilities.is_empty());
    }

    #[test]
    fn test_vocabulary_stats() {
        let mut frequencies = HashMap::new();
        frequencies.insert("common".to_string(), 0.5);
        frequencies.insert("rare".to_string(), 0.005);
        frequencies.insert("medium".to_string(), 0.1);

        let analyzer =
            LanguageModelAnalyzer::with_default_config().with_word_frequencies(frequencies);

        let stats = analyzer.get_vocabulary_stats();
        assert_eq!(stats.total_words, 3);
        assert!(stats.average_frequency > 0.0);
        assert!(stats.high_frequency_words >= 1);
    }

    #[test]
    fn test_token_info() {
        let mut frequencies = HashMap::new();
        frequencies.insert("example".to_string(), 0.08);

        let analyzer =
            LanguageModelAnalyzer::with_default_config().with_word_frequencies(frequencies);

        let token_info = analyzer.get_token_info("example");
        assert_eq!(token_info.token, "example");
        assert!(token_info.probability > 0.0);
        assert!(token_info.surprisal > 0.0);
    }

    #[test]
    fn test_ngram_analysis() {
        let mut frequencies = HashMap::new();
        frequencies.insert("hello".to_string(), 0.1);
        frequencies.insert("world".to_string(), 0.05);

        let analyzer =
            LanguageModelAnalyzer::with_default_config().with_word_frequencies(frequencies);

        let sentences = vec!["hello world hello world".to_string()];
        let analysis = analyzer.analyze_ngram_patterns(&sentences);

        assert!(!analysis.unigrams.is_empty());
        assert!(!analysis.bigrams.is_empty());
        assert!(analysis.coverage_stats.total_ngram_count > 0);
    }
}
