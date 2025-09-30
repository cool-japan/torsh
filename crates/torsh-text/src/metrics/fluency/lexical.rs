//! Lexical Fluency Analysis
//!
//! This module provides comprehensive lexical fluency evaluation including
//! vocabulary sophistication, lexical diversity, word choice appropriateness,
//! and lexical richness analysis.

use scirs2_core::ndarray::{array, Array1, Array2};
use std::collections::{HashMap, HashSet};

/// Configuration for lexical analysis
#[derive(Debug, Clone)]
pub struct LexicalConfig {
    /// Weight for vocabulary sophistication
    pub sophistication_weight: f64,
    /// Weight for word choice appropriateness
    pub appropriateness_weight: f64,
    /// Weight for lexical diversity
    pub diversity_weight: f64,
    /// Weight for lexical density
    pub density_weight: f64,
    /// Weight for register appropriateness
    pub register_weight: f64,
    /// Threshold for rare word detection
    pub rare_word_threshold: f64,
    /// Threshold for common word detection
    pub common_word_threshold: f64,
    /// Enable advanced lexical metrics
    pub enable_advanced_metrics: bool,
    /// Window size for moving average TTR
    pub ttr_window_size: usize,
    /// Enable collocation analysis
    pub enable_collocation_analysis: bool,
}

impl Default for LexicalConfig {
    fn default() -> Self {
        Self {
            sophistication_weight: 0.25,
            appropriateness_weight: 0.20,
            diversity_weight: 0.20,
            density_weight: 0.15,
            register_weight: 0.20,
            rare_word_threshold: 0.001,
            common_word_threshold: 0.1,
            enable_advanced_metrics: true,
            ttr_window_size: 50,
            enable_collocation_analysis: true,
        }
    }
}

/// Results of lexical fluency analysis
#[derive(Debug, Clone)]
pub struct LexicalFluencyResult {
    /// Vocabulary sophistication measure
    pub vocabulary_sophistication: f64,
    /// Word choice appropriateness score
    pub word_choice_appropriateness: f64,
    /// Lexical diversity measure (TTR-based)
    pub lexical_diversity: f64,
    /// Word frequency profile quality
    pub word_frequency_profile: f64,
    /// Collocation quality measure
    pub collocation_quality: f64,
    /// Lexical density (content vs function words)
    pub lexical_density: f64,
    /// Register appropriateness score
    pub register_appropriateness: f64,
    /// Rare word usage measure
    pub rare_word_usage: f64,
    /// Detailed lexical richness analysis
    pub lexical_richness: LexicalRichness,
    /// Advanced lexical metrics
    pub advanced_metrics: AdvancedLexicalMetrics,
}

/// Lexical richness analysis
#[derive(Debug, Clone)]
pub struct LexicalRichness {
    /// Type-Token Ratio
    pub type_token_ratio: f64,
    /// Moving Average Type-Token Ratio (MATTR)
    pub moving_average_ttr: f64,
    /// Measure of Textual Lexical Diversity (MTLD)
    pub mtld: f64,
    /// Hapax Legomena ratio (words appearing once)
    pub hapax_legomena_ratio: f64,
    /// Lexical sophistication index
    pub lexical_sophistication_index: f64,
    /// Word length diversity
    pub word_length_diversity: f64,
    /// Semantic field diversity
    pub semantic_diversity: f64,
    /// Vocabulary growth curve metrics
    pub vocabulary_growth: VocabularyGrowthMetrics,
}

/// Advanced lexical metrics
#[derive(Debug, Clone)]
pub struct AdvancedLexicalMetrics {
    /// Word frequency distribution analysis
    pub frequency_distribution: FrequencyDistributionAnalysis,
    /// Collocation strength analysis
    pub collocation_analysis: CollocationAnalysis,
    /// Register consistency analysis
    pub register_analysis: RegisterAnalysis,
    /// Lexical sophistication by category
    pub sophistication_by_category: HashMap<String, f64>,
    /// Word class distribution
    pub word_class_distribution: WordClassDistribution,
    /// Lexical innovation measure
    pub lexical_innovation: f64,
}

/// Vocabulary growth curve metrics
#[derive(Debug, Clone)]
pub struct VocabularyGrowthMetrics {
    /// Growth rate (new words per segment)
    pub growth_rate: f64,
    /// Growth consistency across text
    pub growth_consistency: f64,
    /// Vocabulary plateau point
    pub plateau_point: Option<usize>,
    /// Growth curve parameters
    pub curve_parameters: GrowthCurveParams,
}

/// Growth curve parameters
#[derive(Debug, Clone)]
pub struct GrowthCurveParams {
    /// Initial growth rate
    pub initial_rate: f64,
    /// Asymptotic vocabulary size
    pub asymptotic_size: f64,
    /// Growth decay parameter
    pub decay_parameter: f64,
}

/// Frequency distribution analysis
#[derive(Debug, Clone)]
pub struct FrequencyDistributionAnalysis {
    /// Zipf's law fit quality
    pub zipf_fit_quality: f64,
    /// High-frequency word concentration
    pub high_freq_concentration: f64,
    /// Mid-frequency word balance
    pub mid_freq_balance: f64,
    /// Low-frequency word richness
    pub low_freq_richness: f64,
    /// Frequency distribution entropy
    pub distribution_entropy: f64,
}

/// Collocation analysis results
#[derive(Debug, Clone)]
pub struct CollocationAnalysis {
    /// Overall collocation strength
    pub overall_strength: f64,
    /// Bigram quality scores
    pub bigram_quality: HashMap<String, f64>,
    /// Trigram quality scores
    pub trigram_quality: HashMap<String, f64>,
    /// Semantic coherence of collocations
    pub semantic_coherence: f64,
    /// Novel collocation usage
    pub novel_usage: f64,
}

/// Register analysis results
#[derive(Debug, Clone)]
pub struct RegisterAnalysis {
    /// Overall register consistency
    pub consistency_score: f64,
    /// Formal register usage
    pub formal_usage: f64,
    /// Informal register usage
    pub informal_usage: f64,
    /// Technical register usage
    pub technical_usage: f64,
    /// Register mixing penalties
    pub mixing_penalties: f64,
    /// Contextual appropriateness
    pub contextual_appropriateness: f64,
}

/// Word class distribution
#[derive(Debug, Clone)]
pub struct WordClassDistribution {
    /// Noun usage percentage
    pub nouns: f64,
    /// Verb usage percentage
    pub verbs: f64,
    /// Adjective usage percentage
    pub adjectives: f64,
    /// Adverb usage percentage
    pub adverbs: f64,
    /// Function word usage percentage
    pub function_words: f64,
    /// Word class variety entropy
    pub variety_entropy: f64,
}

/// Lexical analyzer
pub struct LexicalAnalyzer {
    config: LexicalConfig,
    word_frequencies: HashMap<String, f64>,
    collocations: HashMap<(String, String), f64>,
    semantic_fields: HashMap<String, String>,
    register_categories: RegisterCategories,
    word_classes: HashMap<String, WordClass>,
}

/// Word class categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WordClass {
    Noun,
    Verb,
    Adjective,
    Adverb,
    Preposition,
    Conjunction,
    Article,
    Pronoun,
    Function,
    Content,
}

/// Register categories for analysis
#[derive(Debug, Clone)]
pub struct RegisterCategories {
    /// Formal vocabulary items
    pub formal_words: HashSet<String>,
    /// Informal vocabulary items
    pub informal_words: HashSet<String>,
    /// Technical vocabulary items
    pub technical_words: HashSet<String>,
    /// Slang and colloquial items
    pub slang_words: HashSet<String>,
    /// Academic vocabulary
    pub academic_words: HashSet<String>,
}

impl LexicalAnalyzer {
    /// Create a new lexical analyzer
    pub fn new(config: LexicalConfig) -> Self {
        Self {
            config,
            word_frequencies: HashMap::new(),
            collocations: HashMap::new(),
            semantic_fields: HashMap::new(),
            register_categories: Self::build_register_categories(),
            word_classes: HashMap::new(),
        }
    }

    /// Create analyzer with default configuration
    pub fn with_default_config() -> Self {
        Self::new(LexicalConfig::default())
    }

    /// Initialize with word frequency data
    pub fn with_word_frequencies(mut self, frequencies: HashMap<String, f64>) -> Self {
        self.word_frequencies = frequencies;
        self
    }

    /// Initialize with collocation data
    pub fn with_collocations(mut self, collocations: HashMap<(String, String), f64>) -> Self {
        self.collocations = collocations;
        self
    }

    /// Build default register categories
    fn build_register_categories() -> RegisterCategories {
        let formal_words = [
            "therefore",
            "however",
            "nevertheless",
            "consequently",
            "furthermore",
            "moreover",
            "utilize",
            "implement",
            "demonstrate",
            "establish",
            "indicate",
            "analyze",
            "examine",
            "investigate",
            "construct",
            "facilitate",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let informal_words = [
            "yeah", "ok", "gonna", "wanna", "kinda", "sorta", "stuff", "things", "guys", "folks",
            "pretty", "really", "super", "way", "like",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let technical_words = [
            "algorithm",
            "implementation",
            "optimization",
            "configuration",
            "parameter",
            "interface",
            "architecture",
            "infrastructure",
            "methodology",
            "framework",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let slang_words = [
            "cool", "awesome", "dude", "bro", "sick", "lit", "epic", "rad", "tight",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        let academic_words = [
            "hypothesis",
            "empirical",
            "theoretical",
            "methodology",
            "paradigm",
            "correlation",
            "causation",
            "significant",
            "comprehensive",
            "systematic",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        RegisterCategories {
            formal_words,
            informal_words,
            technical_words,
            slang_words,
            academic_words,
        }
    }

    /// Analyze lexical fluency of text
    pub fn analyze_lexical_fluency(
        &self,
        sentences: &[String],
        full_text: &str,
    ) -> LexicalFluencyResult {
        let all_words = self.extract_all_words(sentences);

        let vocabulary_sophistication = self.calculate_vocabulary_sophistication(&all_words);
        let word_choice_appropriateness = self.calculate_word_choice_appropriateness(&all_words);
        let lexical_diversity = self.calculate_lexical_diversity(&all_words);
        let word_frequency_profile = self.calculate_word_frequency_profile(&all_words);
        let collocation_quality = self.calculate_collocation_quality(sentences);
        let lexical_density = self.calculate_lexical_density(&all_words);
        let register_appropriateness = self.calculate_register_appropriateness(&all_words);
        let rare_word_usage = self.calculate_rare_word_usage(&all_words);

        let lexical_richness = self.calculate_lexical_richness(&all_words, full_text);

        let advanced_metrics = if self.config.enable_advanced_metrics {
            self.calculate_advanced_metrics(&all_words, sentences)
        } else {
            self.create_default_advanced_metrics()
        };

        LexicalFluencyResult {
            vocabulary_sophistication,
            word_choice_appropriateness,
            lexical_diversity,
            word_frequency_profile,
            collocation_quality,
            lexical_density,
            register_appropriateness,
            rare_word_usage,
            lexical_richness,
            advanced_metrics,
        }
    }

    /// Extract all words from sentences
    fn extract_all_words(&self, sentences: &[String]) -> Vec<String> {
        sentences
            .iter()
            .flat_map(|sentence| self.tokenize_sentence(sentence))
            .collect()
    }

    /// Calculate vocabulary sophistication
    pub fn calculate_vocabulary_sophistication(&self, words: &[String]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }

        let mut sophistication_sum = 0.0;
        for word in words {
            let frequency = self.word_frequencies.get(word).unwrap_or(&0.01);
            let sophistication = 1.0 - frequency;
            sophistication_sum += sophistication;
        }

        sophistication_sum / words.len() as f64
    }

    /// Calculate word choice appropriateness
    pub fn calculate_word_choice_appropriateness(&self, words: &[String]) -> f64 {
        if words.is_empty() {
            return 1.0;
        }

        let mut appropriateness_sum = 0.0;
        for word in words {
            let frequency = self.word_frequencies.get(word).unwrap_or(&0.01);
            let appropriateness = if *frequency < self.config.rare_word_threshold {
                0.7 // Very rare words might be inappropriate
            } else if *frequency > self.config.common_word_threshold {
                0.9 // Common words are generally appropriate
            } else {
                1.0 // Mid-frequency words are most appropriate
            };

            appropriateness_sum += appropriateness;
        }

        appropriateness_sum / words.len() as f64
    }

    /// Calculate lexical diversity (Type-Token Ratio)
    pub fn calculate_lexical_diversity(&self, words: &[String]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }

        let unique_words: HashSet<String> = words.iter().cloned().collect();
        unique_words.len() as f64 / words.len() as f64
    }

    /// Calculate word frequency profile
    pub fn calculate_word_frequency_profile(&self, words: &[String]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }

        let mut high_freq_count = 0;
        let mut mid_freq_count = 0;
        let mut low_freq_count = 0;

        for word in words {
            let frequency = self.word_frequencies.get(word).unwrap_or(&0.01);

            if *frequency > self.config.common_word_threshold {
                high_freq_count += 1;
            } else if *frequency > 0.01 {
                mid_freq_count += 1;
            } else {
                low_freq_count += 1;
            }
        }

        let total_words = words.len() as f64;
        let high_freq_ratio = high_freq_count as f64 / total_words;
        let mid_freq_ratio = mid_freq_count as f64 / total_words;
        let low_freq_ratio = low_freq_count as f64 / total_words;

        // Ideal distribution: 60% high, 30% mid, 10% low
        let ideal_score = 1.0
            - ((high_freq_ratio - 0.6).abs()
                + (mid_freq_ratio - 0.3).abs()
                + (low_freq_ratio - 0.1).abs());

        ideal_score.max(0.0)
    }

    /// Calculate collocation quality
    pub fn calculate_collocation_quality(&self, sentences: &[String]) -> f64 {
        if !self.config.enable_collocation_analysis {
            return 0.5;
        }

        let mut quality_sum = 0.0;
        let mut collocation_count = 0;

        for sentence in sentences {
            let words = self.tokenize_sentence(sentence);

            // Analyze bigrams
            for i in 0..words.len().saturating_sub(1) {
                let word1 = &words[i];
                let word2 = &words[i + 1];

                if let Some(strength) = self.collocations.get(&(word1.clone(), word2.clone())) {
                    quality_sum += strength;
                } else {
                    // Estimate collocation strength based on semantic coherence
                    let semantic_strength = self.calculate_semantic_coherence(word1, word2);
                    quality_sum += semantic_strength;
                }
                collocation_count += 1;
            }
        }

        if collocation_count > 0 {
            quality_sum / collocation_count as f64
        } else {
            0.5
        }
    }

    /// Calculate semantic coherence between two words
    fn calculate_semantic_coherence(&self, word1: &str, word2: &str) -> f64 {
        // Simplified semantic coherence based on word types and patterns
        let function_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "from", "with",
        ];

        if function_words.contains(&word1) || function_words.contains(&word2) {
            0.8 // Function words generally have good coherence
        } else if word1.len() > 3 && word2.len() > 3 {
            0.6 // Content words
        } else {
            0.4 // Short words might have less semantic content
        }
    }

    /// Calculate lexical density
    pub fn calculate_lexical_density(&self, words: &[String]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }

        let function_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "as", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can",
            "must", "shall",
        ];

        let content_word_count = words
            .iter()
            .filter(|word| !function_words.contains(&word.as_str()))
            .count();

        content_word_count as f64 / words.len() as f64
    }

    /// Calculate register appropriateness
    pub fn calculate_register_appropriateness(&self, words: &[String]) -> f64 {
        if words.is_empty() {
            return 1.0;
        }

        let mut formal_count = 0;
        let mut informal_count = 0;
        let mut slang_count = 0;
        let mut technical_count = 0;
        let mut academic_count = 0;

        for word in words {
            if self.register_categories.formal_words.contains(word) {
                formal_count += 1;
            } else if self.register_categories.informal_words.contains(word) {
                informal_count += 1;
            } else if self.register_categories.slang_words.contains(word) {
                slang_count += 1;
            } else if self.register_categories.technical_words.contains(word) {
                technical_count += 1;
            } else if self.register_categories.academic_words.contains(word) {
                academic_count += 1;
            }
        }

        let total_marked_words =
            formal_count + informal_count + slang_count + technical_count + academic_count;

        if total_marked_words == 0 {
            return 0.8; // Neutral register
        }

        // Penalize excessive mixing of registers
        let register_diversity = [
            formal_count,
            informal_count,
            slang_count,
            technical_count,
            academic_count,
        ]
        .iter()
        .filter(|&&count| count > 0)
        .count();

        let consistency_penalty = if register_diversity > 2 {
            0.1 * (register_diversity - 2) as f64
        } else {
            0.0
        };
        let slang_penalty = slang_count as f64 / words.len() as f64 * 0.2;

        (1.0 - consistency_penalty - slang_penalty).max(0.0)
    }

    /// Calculate rare word usage
    pub fn calculate_rare_word_usage(&self, words: &[String]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }

        let rare_word_count = words
            .iter()
            .filter(|word| {
                let freq = self.word_frequencies.get(*word).unwrap_or(&0.01);
                *freq < self.config.rare_word_threshold
            })
            .count();

        let rare_ratio = rare_word_count as f64 / words.len() as f64;

        // Optimal rare word usage is around 5-10%
        if rare_ratio <= 0.05 {
            rare_ratio * 20.0 // Reward up to 5%
        } else if rare_ratio <= 0.10 {
            1.0 // Optimal range
        } else {
            1.0 - (rare_ratio - 0.10) * 2.0 // Penalize excessive rare words
        }
        .max(0.0)
        .min(1.0)
    }

    /// Calculate comprehensive lexical richness
    pub fn calculate_lexical_richness(&self, words: &[String], full_text: &str) -> LexicalRichness {
        let type_token_ratio = self.calculate_lexical_diversity(words);
        let moving_average_ttr = self.calculate_moving_average_ttr(words);
        let mtld = self.calculate_mtld(words);
        let hapax_legomena_ratio = self.calculate_hapax_legomena_ratio(words);
        let lexical_sophistication_index = self.calculate_lexical_sophistication_index(words);
        let word_length_diversity = self.calculate_word_length_diversity(words);
        let semantic_diversity = self.calculate_semantic_diversity(words);
        let vocabulary_growth = self.calculate_vocabulary_growth(words);

        LexicalRichness {
            type_token_ratio,
            moving_average_ttr,
            mtld,
            hapax_legomena_ratio,
            lexical_sophistication_index,
            word_length_diversity,
            semantic_diversity,
            vocabulary_growth,
        }
    }

    /// Calculate Moving Average Type-Token Ratio (MATTR)
    fn calculate_moving_average_ttr(&self, words: &[String]) -> f64 {
        if words.len() < self.config.ttr_window_size {
            return self.calculate_lexical_diversity(words);
        }

        let mut ttr_sum = 0.0;
        let mut window_count = 0;

        for i in 0..=words.len().saturating_sub(self.config.ttr_window_size) {
            let window = &words[i..i + self.config.ttr_window_size];
            let unique_words: HashSet<String> = window.iter().cloned().collect();
            let ttr = unique_words.len() as f64 / self.config.ttr_window_size as f64;
            ttr_sum += ttr;
            window_count += 1;
        }

        if window_count > 0 {
            ttr_sum / window_count as f64
        } else {
            self.calculate_lexical_diversity(words)
        }
    }

    /// Calculate Measure of Textual Lexical Diversity (MTLD)
    fn calculate_mtld(&self, words: &[String]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }

        let threshold = 0.72; // Standard MTLD threshold
        let mut segment_lengths = Vec::new();
        let mut current_segment = Vec::new();

        for word in words {
            current_segment.push(word.clone());
            let unique_words: HashSet<String> = current_segment.iter().cloned().collect();
            let ttr = unique_words.len() as f64 / current_segment.len() as f64;

            if ttr < threshold || current_segment.len() >= 100 {
                segment_lengths.push(current_segment.len());
                current_segment.clear();
            }
        }

        // Handle remaining segment
        if !current_segment.is_empty() {
            segment_lengths.push(current_segment.len());
        }

        if !segment_lengths.is_empty() {
            let mean_length =
                segment_lengths.iter().sum::<usize>() as f64 / segment_lengths.len() as f64;
            mean_length
        } else {
            words.len() as f64
        }
    }

    /// Calculate hapax legomena ratio (words appearing exactly once)
    fn calculate_hapax_legomena_ratio(&self, words: &[String]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }

        let mut word_counts = HashMap::new();
        for word in words {
            *word_counts.entry(word.clone()).or_insert(0) += 1;
        }

        let hapax_count = word_counts.values().filter(|&&count| count == 1).count();
        hapax_count as f64 / word_counts.len() as f64
    }

    /// Calculate lexical sophistication index
    fn calculate_lexical_sophistication_index(&self, words: &[String]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }

        let mut sophistication_scores = Vec::new();
        for word in words {
            let frequency = self.word_frequencies.get(word).unwrap_or(&0.01);
            let length_score = (word.len() as f64 / 10.0).min(1.0);
            let frequency_score = (1.0 - frequency).max(0.0);
            let sophistication = (length_score + frequency_score) / 2.0;
            sophistication_scores.push(sophistication);
        }

        sophistication_scores.iter().sum::<f64>() / sophistication_scores.len() as f64
    }

    /// Calculate word length diversity
    fn calculate_word_length_diversity(&self, words: &[String]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }

        let mut length_counts = HashMap::new();
        for word in words {
            *length_counts.entry(word.len()).or_insert(0) += 1;
        }

        let total_words = words.len();
        self.calculate_length_entropy(&length_counts, total_words)
    }

    /// Calculate entropy for length distribution
    fn calculate_length_entropy(&self, length_counts: &HashMap<usize, usize>, total: usize) -> f64 {
        let mut entropy = 0.0;
        for &count in length_counts.values() {
            if count > 0 {
                let p = count as f64 / total as f64;
                entropy -= p * p.log2();
            }
        }

        // Normalize by maximum possible entropy
        let max_entropy = (length_counts.len() as f64).log2();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Calculate semantic diversity
    fn calculate_semantic_diversity(&self, words: &[String]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }

        // Simplified semantic field categorization
        let mut field_counts = HashMap::new();
        for word in words {
            let field = self.get_semantic_field(word);
            *field_counts.entry(field).or_insert(0) += 1;
        }

        let total_words = words.len();
        self.calculate_semantic_entropy(&field_counts, total_words)
    }

    /// Get semantic field for a word (simplified categorization)
    fn get_semantic_field(&self, word: &str) -> String {
        if let Some(field) = self.semantic_fields.get(word) {
            field.clone()
        } else {
            // Simple heuristic categorization
            match word.chars().next() {
                Some(c) if c >= 'a' && c <= 'f' => "abstract".to_string(),
                Some(c) if c >= 'g' && c <= 'l' => "concrete".to_string(),
                Some(c) if c >= 'm' && c <= 'r' => "action".to_string(),
                Some(c) if c >= 's' && c <= 'z' => "descriptive".to_string(),
                _ => "other".to_string(),
            }
        }
    }

    /// Calculate entropy for semantic field distribution
    fn calculate_semantic_entropy(
        &self,
        field_counts: &HashMap<String, usize>,
        total: usize,
    ) -> f64 {
        let mut entropy = 0.0;
        for &count in field_counts.values() {
            if count > 0 {
                let p = count as f64 / total as f64;
                entropy -= p * p.log2();
            }
        }

        // Normalize by maximum possible entropy
        let max_entropy = (field_counts.len() as f64).log2();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Calculate vocabulary growth metrics
    fn calculate_vocabulary_growth(&self, words: &[String]) -> VocabularyGrowthMetrics {
        if words.is_empty() {
            return VocabularyGrowthMetrics {
                growth_rate: 0.0,
                growth_consistency: 0.0,
                plateau_point: None,
                curve_parameters: GrowthCurveParams {
                    initial_rate: 0.0,
                    asymptotic_size: 0.0,
                    decay_parameter: 0.0,
                },
            };
        }

        let segment_size = (words.len() / 10).max(10);
        let mut vocabulary_sizes = Vec::new();
        let mut seen_words = HashSet::new();

        for i in (0..words.len()).step_by(segment_size) {
            let end = (i + segment_size).min(words.len());
            for word in &words[i..end] {
                seen_words.insert(word.clone());
            }
            vocabulary_sizes.push(seen_words.len());
        }

        let growth_rate = if vocabulary_sizes.len() > 1 {
            let total_growth = vocabulary_sizes.last().unwrap() - vocabulary_sizes.first().unwrap();
            total_growth as f64 / (vocabulary_sizes.len() - 1) as f64
        } else {
            0.0
        };

        let growth_consistency = self.calculate_growth_consistency(&vocabulary_sizes);
        let plateau_point = self.find_plateau_point(&vocabulary_sizes);
        let curve_parameters = self.estimate_growth_curve_parameters(&vocabulary_sizes);

        VocabularyGrowthMetrics {
            growth_rate,
            growth_consistency,
            plateau_point,
            curve_parameters,
        }
    }

    /// Calculate growth consistency
    fn calculate_growth_consistency(&self, sizes: &[usize]) -> f64 {
        if sizes.len() < 3 {
            return 1.0;
        }

        let mut growth_rates = Vec::new();
        for i in 1..sizes.len() {
            let rate = (sizes[i] as f64 - sizes[i - 1] as f64).max(0.0);
            growth_rates.push(rate);
        }

        if growth_rates.is_empty() {
            return 1.0;
        }

        let mean_rate = growth_rates.iter().sum::<f64>() / growth_rates.len() as f64;
        let variance = growth_rates
            .iter()
            .map(|&rate| (rate - mean_rate).powi(2))
            .sum::<f64>()
            / growth_rates.len() as f64;

        let coefficient_of_variation = if mean_rate > 0.0 {
            variance.sqrt() / mean_rate
        } else {
            1.0
        };

        (1.0 - coefficient_of_variation).max(0.0)
    }

    /// Find plateau point in vocabulary growth
    fn find_plateau_point(&self, sizes: &[usize]) -> Option<usize> {
        if sizes.len() < 3 {
            return None;
        }

        let threshold = 0.1; // Growth rate threshold for plateau detection
        for i in 2..sizes.len() {
            let recent_growth = (sizes[i] as f64 - sizes[i - 2] as f64) / 2.0;
            if recent_growth < threshold {
                return Some(i);
            }
        }

        None
    }

    /// Estimate growth curve parameters
    fn estimate_growth_curve_parameters(&self, sizes: &[usize]) -> GrowthCurveParams {
        if sizes.len() < 3 {
            return GrowthCurveParams {
                initial_rate: 0.0,
                asymptotic_size: sizes.last().copied().unwrap_or(0) as f64,
                decay_parameter: 0.0,
            };
        }

        let initial_rate = if sizes.len() > 1 {
            (sizes[1] as f64 - sizes[0] as f64).max(0.0)
        } else {
            0.0
        };

        let asymptotic_size = sizes.last().copied().unwrap_or(0) as f64 * 1.2; // Estimate

        let decay_parameter = if initial_rate > 0.0 {
            0.1 // Simplified estimate
        } else {
            0.0
        };

        GrowthCurveParams {
            initial_rate,
            asymptotic_size,
            decay_parameter,
        }
    }

    /// Calculate advanced lexical metrics
    fn calculate_advanced_metrics(
        &self,
        words: &[String],
        sentences: &[String],
    ) -> AdvancedLexicalMetrics {
        let frequency_distribution = self.analyze_frequency_distribution(words);
        let collocation_analysis = self.analyze_collocations(sentences);
        let register_analysis = self.analyze_register_usage(words);
        let sophistication_by_category = self.calculate_sophistication_by_category(words);
        let word_class_distribution = self.analyze_word_class_distribution(words);
        let lexical_innovation = self.calculate_lexical_innovation(words);

        AdvancedLexicalMetrics {
            frequency_distribution,
            collocation_analysis,
            register_analysis,
            sophistication_by_category,
            word_class_distribution,
            lexical_innovation,
        }
    }

    /// Create default advanced metrics when disabled
    fn create_default_advanced_metrics(&self) -> AdvancedLexicalMetrics {
        AdvancedLexicalMetrics {
            frequency_distribution: FrequencyDistributionAnalysis {
                zipf_fit_quality: 0.0,
                high_freq_concentration: 0.0,
                mid_freq_balance: 0.0,
                low_freq_richness: 0.0,
                distribution_entropy: 0.0,
            },
            collocation_analysis: CollocationAnalysis {
                overall_strength: 0.5,
                bigram_quality: HashMap::new(),
                trigram_quality: HashMap::new(),
                semantic_coherence: 0.5,
                novel_usage: 0.0,
            },
            register_analysis: RegisterAnalysis {
                consistency_score: 0.8,
                formal_usage: 0.0,
                informal_usage: 0.0,
                technical_usage: 0.0,
                mixing_penalties: 0.0,
                contextual_appropriateness: 0.8,
            },
            sophistication_by_category: HashMap::new(),
            word_class_distribution: WordClassDistribution {
                nouns: 0.3,
                verbs: 0.2,
                adjectives: 0.15,
                adverbs: 0.1,
                function_words: 0.25,
                variety_entropy: 0.5,
            },
            lexical_innovation: 0.0,
        }
    }

    /// Analyze frequency distribution patterns
    fn analyze_frequency_distribution(&self, words: &[String]) -> FrequencyDistributionAnalysis {
        let mut word_counts = HashMap::new();
        for word in words {
            *word_counts.entry(word.clone()).or_insert(0) += 1;
        }

        let mut frequencies: Vec<usize> = word_counts.values().copied().collect();
        frequencies.sort_by(|a, b| b.cmp(a)); // Sort descending

        let zipf_fit_quality = self.calculate_zipf_fit(&frequencies);

        let total_words = words.len();
        let high_freq_words = frequencies.iter().take(total_words / 10).sum::<usize>();
        let high_freq_concentration = high_freq_words as f64 / total_words as f64;

        let mid_range_start = total_words / 10;
        let mid_range_end = total_words / 2;
        let mid_freq_words = frequencies
            .iter()
            .skip(mid_range_start)
            .take(mid_range_end - mid_range_start)
            .sum::<usize>();
        let mid_freq_balance = mid_freq_words as f64 / total_words as f64;

        let hapax_count = word_counts.values().filter(|&&count| count == 1).count();
        let low_freq_richness = hapax_count as f64 / word_counts.len() as f64;

        let distribution_entropy = self.calculate_frequency_entropy(&word_counts, total_words);

        FrequencyDistributionAnalysis {
            zipf_fit_quality,
            high_freq_concentration,
            mid_freq_balance,
            low_freq_richness,
            distribution_entropy,
        }
    }

    /// Calculate Zipf's law fit quality
    fn calculate_zipf_fit(&self, frequencies: &[usize]) -> f64 {
        if frequencies.len() < 2 {
            return 0.0;
        }

        // Simplified Zipf fit calculation
        let mut error_sum = 0.0;
        for (i, &freq) in frequencies.iter().enumerate() {
            let expected = frequencies[0] as f64 / (i + 1) as f64;
            let actual = freq as f64;
            error_sum += (expected - actual).abs() / expected.max(1.0);
        }

        let mean_error = error_sum / frequencies.len() as f64;
        (1.0 - mean_error).max(0.0)
    }

    /// Calculate entropy for frequency distribution
    fn calculate_frequency_entropy(
        &self,
        word_counts: &HashMap<String, usize>,
        total: usize,
    ) -> f64 {
        let mut entropy = 0.0;
        for &count in word_counts.values() {
            if count > 0 {
                let p = count as f64 / total as f64;
                entropy -= p * p.log2();
            }
        }

        // Normalize by vocabulary size
        let max_entropy = (word_counts.len() as f64).log2();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Analyze collocations in detail
    fn analyze_collocations(&self, sentences: &[String]) -> CollocationAnalysis {
        let mut bigram_quality = HashMap::new();
        let mut trigram_quality = HashMap::new();
        let mut coherence_scores = Vec::new();

        for sentence in sentences {
            let words = self.tokenize_sentence(sentence);

            // Analyze bigrams
            for i in 0..words.len().saturating_sub(1) {
                let bigram = format!("{} {}", words[i], words[i + 1]);
                let quality = self.calculate_semantic_coherence(&words[i], &words[i + 1]);
                bigram_quality.insert(bigram, quality);
                coherence_scores.push(quality);
            }

            // Analyze trigrams
            for i in 0..words.len().saturating_sub(2) {
                let trigram = format!("{} {} {}", words[i], words[i + 1], words[i + 2]);
                let quality1 = self.calculate_semantic_coherence(&words[i], &words[i + 1]);
                let quality2 = self.calculate_semantic_coherence(&words[i + 1], &words[i + 2]);
                let avg_quality = (quality1 + quality2) / 2.0;
                trigram_quality.insert(trigram, avg_quality);
            }
        }

        let overall_strength = if !coherence_scores.is_empty() {
            coherence_scores.iter().sum::<f64>() / coherence_scores.len() as f64
        } else {
            0.5
        };

        let semantic_coherence = overall_strength;
        let novel_usage = self.calculate_novel_collocation_usage(&bigram_quality);

        CollocationAnalysis {
            overall_strength,
            bigram_quality,
            trigram_quality,
            semantic_coherence,
            novel_usage,
        }
    }

    /// Calculate novel collocation usage
    fn calculate_novel_collocation_usage(&self, bigram_quality: &HashMap<String, f64>) -> f64 {
        let known_collocations = bigram_quality.len().min(self.collocations.len());
        let novel_count = bigram_quality.len().saturating_sub(known_collocations);

        if bigram_quality.is_empty() {
            0.0
        } else {
            novel_count as f64 / bigram_quality.len() as f64
        }
    }

    /// Analyze register usage in detail
    fn analyze_register_usage(&self, words: &[String]) -> RegisterAnalysis {
        let mut formal_count = 0;
        let mut informal_count = 0;
        let mut technical_count = 0;
        let mut mixing_penalties = 0.0;

        for word in words {
            if self.register_categories.formal_words.contains(word) {
                formal_count += 1;
            } else if self.register_categories.informal_words.contains(word) {
                informal_count += 1;
            } else if self.register_categories.technical_words.contains(word) {
                technical_count += 1;
            }
        }

        let total_marked = formal_count + informal_count + technical_count;
        let formal_usage = if total_marked > 0 {
            formal_count as f64 / total_marked as f64
        } else {
            0.0
        };
        let informal_usage = if total_marked > 0 {
            informal_count as f64 / total_marked as f64
        } else {
            0.0
        };
        let technical_usage = if total_marked > 0 {
            technical_count as f64 / total_marked as f64
        } else {
            0.0
        };

        // Calculate mixing penalty based on register diversity
        let register_types = [formal_usage, informal_usage, technical_usage]
            .iter()
            .filter(|&&usage| usage > 0.1)
            .count();

        if register_types > 1 {
            mixing_penalties = 0.1 * (register_types - 1) as f64;
        }

        let consistency_score = 1.0 - mixing_penalties;
        let contextual_appropriateness = consistency_score * 0.9; // Simplified

        RegisterAnalysis {
            consistency_score,
            formal_usage,
            informal_usage,
            technical_usage,
            mixing_penalties,
            contextual_appropriateness,
        }
    }

    /// Calculate sophistication by word category
    fn calculate_sophistication_by_category(&self, words: &[String]) -> HashMap<String, f64> {
        let mut category_words: HashMap<String, Vec<String>> = HashMap::new();

        for word in words {
            let category = self.categorize_word(word);
            category_words
                .entry(category)
                .or_insert_with(Vec::new)
                .push(word.clone());
        }

        let mut sophistication_by_category = HashMap::new();
        for (category, category_word_list) in category_words {
            let sophistication = self.calculate_vocabulary_sophistication(&category_word_list);
            sophistication_by_category.insert(category, sophistication);
        }

        sophistication_by_category
    }

    /// Categorize word by type (simplified)
    fn categorize_word(&self, word: &str) -> String {
        if let Some(word_class) = self.word_classes.get(word) {
            format!("{:?}", word_class)
        } else {
            // Simple heuristic categorization
            if word.ends_with("ing") || word.ends_with("ed") {
                "Verb".to_string()
            } else if word.ends_with("ly") {
                "Adverb".to_string()
            } else if word.ends_with("tion") || word.ends_with("ness") {
                "Noun".to_string()
            } else {
                "Other".to_string()
            }
        }
    }

    /// Analyze word class distribution
    fn analyze_word_class_distribution(&self, words: &[String]) -> WordClassDistribution {
        let mut noun_count = 0;
        let mut verb_count = 0;
        let mut adjective_count = 0;
        let mut adverb_count = 0;
        let mut function_count = 0;

        let function_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "as", "is", "are", "was", "were",
        ];

        for word in words {
            if function_words.contains(&word.as_str()) {
                function_count += 1;
            } else if word.ends_with("ing") || word.ends_with("ed") {
                verb_count += 1;
            } else if word.ends_with("ly") {
                adverb_count += 1;
            } else if word.ends_with("tion") || word.ends_with("ness") {
                noun_count += 1;
            } else {
                // Default to adjective for remaining content words
                adjective_count += 1;
            }
        }

        let total = words.len() as f64;
        let nouns = noun_count as f64 / total;
        let verbs = verb_count as f64 / total;
        let adjectives = adjective_count as f64 / total;
        let adverbs = adverb_count as f64 / total;
        let function_words_ratio = function_count as f64 / total;

        // Calculate variety entropy
        let distributions = [nouns, verbs, adjectives, adverbs, function_words_ratio];
        let variety_entropy = self.calculate_distribution_entropy(&distributions);

        WordClassDistribution {
            nouns,
            verbs,
            adjectives,
            adverbs,
            function_words: function_words_ratio,
            variety_entropy,
        }
    }

    /// Calculate entropy for distribution
    fn calculate_distribution_entropy(&self, distribution: &[f64]) -> f64 {
        let mut entropy = 0.0;
        for &p in distribution {
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        // Normalize by maximum possible entropy
        let max_entropy = (distribution.len() as f64).log2();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Calculate lexical innovation measure
    fn calculate_lexical_innovation(&self, words: &[String]) -> f64 {
        if words.is_empty() {
            return 0.0;
        }

        let mut innovation_score = 0.0;
        let mut innovation_count = 0;

        for word in words {
            let frequency = self.word_frequencies.get(word).unwrap_or(&0.0);
            if *frequency == 0.0 {
                // Completely novel word
                innovation_score += 1.0;
                innovation_count += 1;
            } else if *frequency < 0.0001 {
                // Very rare word usage
                innovation_score += 0.8;
                innovation_count += 1;
            } else if word.len() > 10 {
                // Long words might indicate sophisticated usage
                innovation_score += 0.3;
                innovation_count += 1;
            }
        }

        if innovation_count > 0 {
            (innovation_score / innovation_count as f64).min(1.0)
        } else {
            0.0
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexical_analyzer_creation() {
        let config = LexicalConfig::default();
        let analyzer = LexicalAnalyzer::new(config);

        assert_eq!(analyzer.config.sophistication_weight, 0.25);
    }

    #[test]
    fn test_vocabulary_sophistication() {
        let mut frequencies = HashMap::new();
        frequencies.insert("common".to_string(), 0.1);
        frequencies.insert("rare".to_string(), 0.001);

        let analyzer = LexicalAnalyzer::with_default_config().with_word_frequencies(frequencies);

        let words = vec!["common".to_string(), "rare".to_string()];
        let sophistication = analyzer.calculate_vocabulary_sophistication(&words);

        assert!(sophistication > 0.0);
        assert!(sophistication < 1.0);
    }

    #[test]
    fn test_lexical_diversity() {
        let analyzer = LexicalAnalyzer::with_default_config();
        let words = vec!["cat".to_string(), "dog".to_string(), "cat".to_string()];

        let diversity = analyzer.calculate_lexical_diversity(&words);
        assert_eq!(diversity, 2.0 / 3.0); // 2 unique words out of 3 total
    }

    #[test]
    fn test_lexical_density() {
        let analyzer = LexicalAnalyzer::with_default_config();
        let words = vec!["the".to_string(), "cat".to_string(), "runs".to_string()];

        let density = analyzer.calculate_lexical_density(&words);
        assert!(density > 0.0); // Should have some content words
        assert!(density < 1.0); // Should have some function words
    }

    #[test]
    fn test_moving_average_ttr() {
        let analyzer = LexicalAnalyzer::with_default_config();
        let words: Vec<String> = (0..100)
            .map(|i| format!("word{}", i % 20)) // 20 unique words repeated
            .collect();

        let mattr = analyzer.calculate_moving_average_ttr(&words);
        assert!(mattr > 0.0);
        assert!(mattr <= 1.0);
    }

    #[test]
    fn test_hapax_legomena_ratio() {
        let analyzer = LexicalAnalyzer::with_default_config();
        let words = vec![
            "unique1".to_string(),
            "unique2".to_string(),
            "repeated".to_string(),
            "repeated".to_string(),
        ];

        let hapax_ratio = analyzer.calculate_hapax_legomena_ratio(&words);
        assert_eq!(hapax_ratio, 2.0 / 3.0); // 2 hapax words out of 3 unique
    }

    #[test]
    fn test_register_appropriateness() {
        let analyzer = LexicalAnalyzer::with_default_config();

        let formal_words = vec!["therefore".to_string(), "however".to_string()];
        let formal_score = analyzer.calculate_register_appropriateness(&formal_words);

        let mixed_words = vec!["therefore".to_string(), "yeah".to_string()];
        let mixed_score = analyzer.calculate_register_appropriateness(&mixed_words);

        assert!(formal_score > mixed_score); // Consistent register should score higher
    }

    #[test]
    fn test_lexical_fluency_analysis() {
        let mut frequencies = HashMap::new();
        frequencies.insert("the".to_string(), 0.1);
        frequencies.insert("cat".to_string(), 0.05);
        frequencies.insert("sophisticated".to_string(), 0.001);

        let analyzer = LexicalAnalyzer::with_default_config().with_word_frequencies(frequencies);

        let sentences = vec!["The sophisticated cat runs quickly.".to_string()];
        let full_text = "The sophisticated cat runs quickly.";

        let result = analyzer.analyze_lexical_fluency(&sentences, full_text);

        assert!(result.vocabulary_sophistication >= 0.0);
        assert!(result.lexical_diversity >= 0.0);
        assert!(result.lexical_richness.type_token_ratio >= 0.0);
        assert!(result.advanced_metrics.lexical_innovation >= 0.0);
    }
}
