//! Feature Extraction Module
//!
//! This module provides comprehensive semantic feature extraction capabilities for text analysis.
//! It extracts various types of features including linguistic, semantic domain, word embedding,
//! and contextual features that capture different aspects of text meaning and structure.
//!
//! # Feature Categories
//!
//! ## Linguistic Features
//! - **Length Features**: Word count, character count, average word length
//! - **POS Features**: Simplified part-of-speech patterns and distributions
//! - **Syntactic Features**: Bigrams, lexical diversity, complexity measures
//!
//! ## Semantic Features
//! - **Domain Features**: Coverage across predefined semantic domains
//! - **Word Embeddings**: Simulated word embedding representations
//! - **Contextual Features**: Context-aware semantic representations
//!
//! ## Advanced Features
//! - **Topic Distributions**: Topic modeling-based feature vectors
//! - **Sentiment Features**: Sentiment-aware semantic representations
//! - **Word Importance**: TF-IDF and frequency-based word weighting
//!
//! # Usage Examples
//!
//! ```rust
//! use torsh_text::metrics::semantic::feature_extraction::{FeatureExtractor, FeatureExtractionConfig};
//!
//! let config = FeatureExtractionConfig::new()
//!     .with_linguistic_features(true)
//!     .with_semantic_domains(true)
//!     .with_word_embeddings(true);
//!
//! let extractor = FeatureExtractor::with_config(config);
//! let features = extractor.extract_features("Your text here...")?;
//!
//! println!("Feature vector dimension: {}", features.features.len());
//! ```

use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Errors that can occur during feature extraction
#[derive(Error, Debug, Clone, PartialEq)]
pub enum FeatureExtractionError {
    #[error("Invalid text input: {message}")]
    InvalidInput { message: String },
    #[error("Feature extraction failed: {operation} - {reason}")]
    ExtractionError { operation: String, reason: String },
    #[error("Configuration error: {parameter} = {value}")]
    ConfigurationError { parameter: String, value: String },
    #[error(
        "Insufficient text for analysis: minimum {min_words} words required, got {actual_words}"
    )]
    InsufficientText {
        min_words: usize,
        actual_words: usize,
    },
}

/// Feature extraction strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureExtraction {
    /// Basic linguistic features (POS, length, etc.)
    Linguistic,
    /// Extended features with semantic domains
    Extended,
    /// Comprehensive features with embeddings
    Comprehensive,
    /// All available features
    Complete,
}

/// Configuration for feature extraction
#[derive(Debug, Clone)]
pub struct FeatureExtractionConfig {
    pub extraction_strategy: FeatureExtraction,
    pub feature_dimension: usize,
    pub use_semantic_domains: bool,
    pub use_word_embeddings: bool,
    pub use_topic_modeling: bool,
    pub use_sentiment_analysis: bool,
    pub ignore_case: bool,
    pub remove_stopwords: bool,
    pub use_stemming: bool,
    pub min_word_length: usize,
    pub embedding_dimension: usize,
    pub normalize_features: bool,
}

impl Default for FeatureExtractionConfig {
    fn default() -> Self {
        Self {
            extraction_strategy: FeatureExtraction::Extended,
            feature_dimension: 128,
            use_semantic_domains: true,
            use_word_embeddings: true,
            use_topic_modeling: false,
            use_sentiment_analysis: false,
            ignore_case: true,
            remove_stopwords: true,
            use_stemming: false,
            min_word_length: 2,
            embedding_dimension: 32,
            normalize_features: true,
        }
    }
}

impl FeatureExtractionConfig {
    /// Create new configuration with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set feature extraction strategy
    pub fn with_strategy(mut self, strategy: FeatureExtraction) -> Self {
        self.extraction_strategy = strategy;
        self
    }

    /// Set feature dimension
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.feature_dimension = dimension;
        self
    }

    /// Enable linguistic features
    pub fn with_linguistic_features(mut self, enable: bool) -> Self {
        // Linguistic features are always enabled as base features
        self
    }

    /// Enable semantic domain features
    pub fn with_semantic_domains(mut self, enable: bool) -> Self {
        self.use_semantic_domains = enable;
        self
    }

    /// Enable word embedding features
    pub fn with_word_embeddings(mut self, enable: bool) -> Self {
        self.use_word_embeddings = enable;
        self
    }

    /// Enable topic modeling features
    pub fn with_topic_modeling(mut self, enable: bool) -> Self {
        self.use_topic_modeling = enable;
        self
    }

    /// Enable sentiment analysis features
    pub fn with_sentiment_analysis(mut self, enable: bool) -> Self {
        self.use_sentiment_analysis = enable;
        self
    }

    /// Enable text preprocessing options
    pub fn with_preprocessing(
        mut self,
        ignore_case: bool,
        remove_stopwords: bool,
        use_stemming: bool,
    ) -> Self {
        self.ignore_case = ignore_case;
        self.remove_stopwords = remove_stopwords;
        self.use_stemming = use_stemming;
        self
    }

    /// Enable feature normalization
    pub fn with_normalization(mut self, enable: bool) -> Self {
        self.normalize_features = enable;
        self
    }
}

/// Semantic feature vector with comprehensive metadata
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticFeatureVector {
    /// Main feature vector
    pub features: Vec<f64>,
    /// Semantic domain scores
    pub domain_scores: HashMap<String, f64>,
    /// Word importance weights
    pub word_weights: HashMap<String, f64>,
    /// Sentiment scores (if enabled)
    pub sentiment_scores: Option<HashMap<String, f64>>,
    /// Topic distribution (if enabled)
    pub topic_distribution: Option<Vec<f64>>,
    /// Feature extraction metadata
    pub metadata: FeatureMetadata,
}

/// Metadata about feature extraction process
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureMetadata {
    pub text_length: usize,
    pub word_count: usize,
    pub unique_words: usize,
    pub extraction_time_ms: u64,
    pub features_used: Vec<String>,
    pub normalization_applied: bool,
}

/// Processed text representation for feature extraction
#[derive(Debug, Clone)]
pub struct ProcessedText {
    pub original: String,
    pub processed: String,
    pub tokens: Vec<String>,
    pub word_frequencies: HashMap<String, usize>,
    pub bigrams: Vec<String>,
}

/// Main feature extractor
pub struct FeatureExtractor {
    config: FeatureExtractionConfig,
    stopwords: HashSet<String>,
    domain_vocabularies: HashMap<String, HashSet<String>>,
    embedding_cache: HashMap<String, Vec<f64>>,
}

impl FeatureExtractor {
    /// Create new feature extractor with default configuration
    pub fn new() -> Self {
        Self::with_config(FeatureExtractionConfig::default())
    }

    /// Create feature extractor with custom configuration
    pub fn with_config(config: FeatureExtractionConfig) -> Self {
        let mut extractor = Self {
            config,
            stopwords: HashSet::new(),
            domain_vocabularies: HashMap::new(),
            embedding_cache: HashMap::new(),
        };

        extractor.initialize_resources();
        extractor
    }

    /// Extract comprehensive semantic features from text
    pub fn extract_features(
        &self,
        text: &str,
    ) -> Result<SemanticFeatureVector, FeatureExtractionError> {
        let start_time = std::time::Instant::now();

        // Validate input
        if text.trim().is_empty() {
            return Err(FeatureExtractionError::InvalidInput {
                message: "Input text is empty".to_string(),
            });
        }

        // Process text
        let processed_text = self.process_text(text)?;

        // Check minimum word requirement
        if processed_text.tokens.len() < 1 {
            return Err(FeatureExtractionError::InsufficientText {
                min_words: 1,
                actual_words: processed_text.tokens.len(),
            });
        }

        // Initialize feature vector
        let mut features = vec![0.0; self.config.feature_dimension];
        let mut features_used = Vec::new();

        // Extract features based on configuration
        match self.config.extraction_strategy {
            FeatureExtraction::Linguistic => {
                self.add_linguistic_features(&mut features, &processed_text)?;
                features_used.push("linguistic".to_string());
            }
            FeatureExtraction::Extended => {
                self.add_linguistic_features(&mut features, &processed_text)?;
                features_used.push("linguistic".to_string());

                if self.config.use_semantic_domains {
                    self.add_semantic_domain_features(&mut features, &processed_text)?;
                    features_used.push("semantic_domains".to_string());
                }
            }
            FeatureExtraction::Comprehensive => {
                self.add_linguistic_features(&mut features, &processed_text)?;
                features_used.push("linguistic".to_string());

                if self.config.use_semantic_domains {
                    self.add_semantic_domain_features(&mut features, &processed_text)?;
                    features_used.push("semantic_domains".to_string());
                }

                if self.config.use_word_embeddings {
                    self.add_word_embedding_features(&mut features, &processed_text)?;
                    features_used.push("word_embeddings".to_string());
                }
            }
            FeatureExtraction::Complete => {
                self.add_linguistic_features(&mut features, &processed_text)?;
                self.add_semantic_domain_features(&mut features, &processed_text)?;
                self.add_word_embedding_features(&mut features, &processed_text)?;
                self.add_contextual_features(&mut features, &processed_text)?;
                features_used.extend(vec![
                    "linguistic".to_string(),
                    "semantic_domains".to_string(),
                    "word_embeddings".to_string(),
                    "contextual".to_string(),
                ]);
            }
        }

        // Normalize features if configured
        if self.config.normalize_features {
            self.normalize_feature_vector(&mut features);
        }

        // Extract domain scores
        let domain_scores = self.compute_domain_scores(&processed_text.tokens)?;

        // Calculate word importance weights
        let word_weights = self.compute_word_weights(&processed_text)?;

        // Extract sentiment scores if enabled
        let sentiment_scores = if self.config.use_sentiment_analysis {
            Some(self.extract_sentiment_features(&processed_text.tokens)?)
        } else {
            None
        };

        // Extract topic distribution if enabled
        let topic_distribution = if self.config.use_topic_modeling {
            Some(self.compute_topic_distribution(&processed_text.tokens)?)
        } else {
            None
        };

        let extraction_time = start_time.elapsed().as_millis() as u64;

        // Create metadata
        let metadata = FeatureMetadata {
            text_length: text.len(),
            word_count: processed_text.tokens.len(),
            unique_words: processed_text.word_frequencies.len(),
            extraction_time_ms: extraction_time,
            features_used,
            normalization_applied: self.config.normalize_features,
        };

        Ok(SemanticFeatureVector {
            features,
            domain_scores,
            word_weights,
            sentiment_scores,
            topic_distribution,
            metadata,
        })
    }

    /// Process text for feature extraction
    pub fn process_text(&self, text: &str) -> Result<ProcessedText, FeatureExtractionError> {
        let original = text.to_string();

        // Preprocessing
        let processed = self.preprocess_text(text);

        // Tokenization
        let tokens = self.tokenize(&processed);

        // Frequency analysis
        let word_frequencies = self.compute_word_frequencies(&tokens);

        // Extract bigrams
        let bigrams = self.extract_bigrams(&tokens);

        Ok(ProcessedText {
            original,
            processed,
            tokens,
            word_frequencies,
            bigrams,
        })
    }

    /// Get feature importance scores for debugging
    pub fn get_feature_importance(&self, features: &SemanticFeatureVector) -> HashMap<String, f64> {
        let mut importance = HashMap::new();

        // Calculate importance based on feature magnitudes
        let total_magnitude: f64 = features.features.iter().map(|x| x.abs()).sum();

        if total_magnitude > 0.0 {
            for (i, &value) in features.features.iter().enumerate() {
                let relative_importance = value.abs() / total_magnitude;
                importance.insert(format!("feature_{}", i), relative_importance);
            }
        }

        // Add domain scores
        for (domain, &score) in &features.domain_scores {
            importance.insert(format!("domain_{}", domain), score);
        }

        importance
    }

    // Private helper methods

    fn initialize_resources(&mut self) {
        self.initialize_stopwords();
        self.initialize_domain_vocabularies();
        self.initialize_embedding_cache();
    }

    fn initialize_stopwords(&mut self) {
        let stopwords = vec![
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
            "is", "it", "its", "of", "on", "that", "the", "to", "was", "will", "with", "would",
            "could", "should", "may", "might", "can", "must", "shall", "this", "these", "those",
            "they", "them", "their", "we", "us", "our", "i", "me", "my", "you", "your", "she",
            "her", "him", "his",
        ];

        self.stopwords = stopwords.into_iter().map(String::from).collect();
    }

    fn initialize_domain_vocabularies(&mut self) {
        // Technology domain
        let tech_words: HashSet<String> = vec![
            "computer",
            "software",
            "algorithm",
            "data",
            "network",
            "internet",
            "programming",
            "code",
            "system",
            "application",
            "digital",
            "technology",
            "database",
            "server",
            "cloud",
            "artificial",
            "intelligence",
            "machine",
            "learning",
            "neural",
            "processing",
            "optimization",
            "automation",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        self.domain_vocabularies
            .insert("Technology".to_string(), tech_words);

        // Science domain
        let science_words: HashSet<String> = vec![
            "research",
            "study",
            "experiment",
            "analysis",
            "hypothesis",
            "theory",
            "method",
            "result",
            "conclusion",
            "evidence",
            "observation",
            "measurement",
            "scientific",
            "laboratory",
            "testing",
            "validation",
            "discovery",
            "innovation",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        self.domain_vocabularies
            .insert("Science".to_string(), science_words);

        // Business domain
        let business_words: HashSet<String> = vec![
            "company",
            "market",
            "profit",
            "customer",
            "product",
            "service",
            "strategy",
            "business",
            "management",
            "revenue",
            "growth",
            "investment",
            "marketing",
            "sales",
            "finance",
            "economics",
            "corporate",
            "commercial",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        self.domain_vocabularies
            .insert("Business".to_string(), business_words);

        // Education domain
        let education_words: HashSet<String> = vec![
            "education",
            "learning",
            "teaching",
            "student",
            "teacher",
            "school",
            "university",
            "course",
            "curriculum",
            "knowledge",
            "skill",
            "training",
            "academic",
            "pedagogical",
            "instruction",
            "assessment",
            "development",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        self.domain_vocabularies
            .insert("Education".to_string(), education_words);

        // Healthcare domain
        let healthcare_words: HashSet<String> = vec![
            "health",
            "medical",
            "treatment",
            "patient",
            "doctor",
            "diagnosis",
            "therapy",
            "medicine",
            "clinical",
            "healthcare",
            "pharmaceutical",
            "wellness",
            "prevention",
            "recovery",
            "rehabilitation",
            "symptom",
            "disease",
            "cure",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        self.domain_vocabularies
            .insert("Healthcare".to_string(), healthcare_words);
    }

    fn initialize_embedding_cache(&mut self) {
        // Pre-cache common word embeddings for performance
        let common_words = vec![
            "the",
            "and",
            "or",
            "but",
            "because",
            "good",
            "bad",
            "fast",
            "slow",
            "big",
            "small",
            "hot",
            "cold",
            "new",
            "old",
            "computer",
            "data",
            "analysis",
            "research",
            "business",
            "market",
            "education",
            "learning",
        ];

        for word in common_words {
            let embedding = self.generate_word_embedding(word);
            self.embedding_cache.insert(word.to_string(), embedding);
        }
    }

    fn preprocess_text(&self, text: &str) -> String {
        let mut processed = if self.config.ignore_case {
            text.to_lowercase()
        } else {
            text.to_string()
        };

        // Remove punctuation for better word matching
        processed = processed
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c.is_whitespace() {
                    c
                } else {
                    ' '
                }
            })
            .collect();

        // Normalize whitespace
        processed = processed.split_whitespace().collect::<Vec<_>>().join(" ");

        processed
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut words: Vec<String> = text
            .split_whitespace()
            .map(|s| s.to_string())
            .filter(|w| w.len() >= self.config.min_word_length)
            .collect();

        // Remove stopwords if configured
        if self.config.remove_stopwords {
            words = words
                .into_iter()
                .filter(|word| !self.stopwords.contains(word))
                .collect();
        }

        // Apply stemming if configured
        if self.config.use_stemming {
            words = words
                .into_iter()
                .map(|word| self.stem_word(&word))
                .collect();
        }

        words
    }

    fn stem_word(&self, word: &str) -> String {
        if word.len() <= 3 {
            return word.to_string();
        }

        // Basic suffix removal
        if word.ends_with("ing") && word.len() > 6 {
            return word[..word.len() - 3].to_string();
        }
        if word.ends_with("ed") && word.len() > 5 {
            return word[..word.len() - 2].to_string();
        }
        if word.ends_with("er") && word.len() > 5 {
            return word[..word.len() - 2].to_string();
        }
        if word.ends_with("ly") && word.len() > 5 {
            return word[..word.len() - 2].to_string();
        }
        if word.ends_with("s") && word.len() > 4 && !word.ends_with("ss") {
            return word[..word.len() - 1].to_string();
        }

        word.to_string()
    }

    fn compute_word_frequencies(&self, tokens: &[String]) -> HashMap<String, usize> {
        let mut frequencies = HashMap::new();
        for token in tokens {
            *frequencies.entry(token.clone()).or_insert(0) += 1;
        }
        frequencies
    }

    fn extract_bigrams(&self, words: &[String]) -> Vec<String> {
        words
            .windows(2)
            .map(|window| format!("{}_{}", window[0], window[1]))
            .collect()
    }

    // Feature extraction methods

    fn add_linguistic_features(
        &self,
        features: &mut [f64],
        processed_text: &ProcessedText,
    ) -> Result<(), FeatureExtractionError> {
        let words = &processed_text.tokens;
        let text = &processed_text.processed;

        if words.is_empty() {
            return Ok(());
        }

        let base_idx = 0;
        let max_features = 16; // Reserve 16 features for linguistic analysis

        if features.len() < base_idx + max_features {
            return Err(FeatureExtractionError::ExtractionError {
                operation: "linguistic_features".to_string(),
                reason: "Insufficient feature vector size".to_string(),
            });
        }

        // Length features
        features[base_idx] = (words.len() as f64 / 100.0).min(1.0);
        features[base_idx + 1] = (text.chars().count() as f64 / 1000.0).min(1.0);

        // Average word length
        let avg_word_len: f64 =
            words.iter().map(|w| w.len()).sum::<usize>() as f64 / words.len() as f64;
        features[base_idx + 2] = (avg_word_len / 20.0).min(1.0);

        // POS-like features (simplified)
        let mut noun_count = 0;
        let mut verb_count = 0;
        let mut adj_count = 0;
        let mut adv_count = 0;

        for word in words {
            if word.ends_with("ing") || word.ends_with("ed") || word.ends_with("s") {
                verb_count += 1;
            } else if word.ends_with("ly") {
                adv_count += 1;
            } else if word.len() > 5 {
                noun_count += 1;
            } else {
                adj_count += 1;
            }
        }

        features[base_idx + 3] = noun_count as f64 / words.len() as f64;
        features[base_idx + 4] = verb_count as f64 / words.len() as f64;
        features[base_idx + 5] = adj_count as f64 / words.len() as f64;
        features[base_idx + 6] = adv_count as f64 / words.len() as f64;

        // Complexity features
        let unique_words = processed_text.word_frequencies.len();
        features[base_idx + 7] = unique_words as f64 / words.len() as f64; // Lexical diversity

        // Syntactic patterns
        features[base_idx + 8] = (processed_text.bigrams.len() as f64 / 100.0).min(1.0);

        // Additional linguistic features
        features[base_idx + 9] = (processed_text.word_frequencies.values().max().unwrap_or(&1)
            / words.len())
        .min(1.0) as f64; // Max word frequency
        features[base_idx + 10] =
            (words.iter().map(|w| w.len()).sum::<usize>() as f64 / text.len() as f64).min(1.0); // Character density

        Ok(())
    }

    fn add_semantic_domain_features(
        &self,
        features: &mut [f64],
        processed_text: &ProcessedText,
    ) -> Result<(), FeatureExtractionError> {
        let base_idx = 16; // Start after linguistic features
        let max_domains = 16; // Reserve 16 features for domain analysis

        if features.len() < base_idx + max_domains {
            return Err(FeatureExtractionError::ExtractionError {
                operation: "semantic_domain_features".to_string(),
                reason: "Insufficient feature vector size".to_string(),
            });
        }

        let words = &processed_text.tokens;
        let mut domain_idx = 0;

        for (domain_name, domain_vocab) in &self.domain_vocabularies {
            if domain_idx >= max_domains {
                break;
            }

            let domain_score = self.compute_domain_coverage(words, domain_vocab);
            features[base_idx + domain_idx] = domain_score;
            domain_idx += 1;
        }

        Ok(())
    }

    fn add_word_embedding_features(
        &self,
        features: &mut [f64],
        processed_text: &ProcessedText,
    ) -> Result<(), FeatureExtractionError> {
        let base_idx = 32; // Start after domain features
        let embedding_dim = self.config.embedding_dimension.min(32); // Limit to 32 dimensions

        if features.len() < base_idx + embedding_dim {
            return Err(FeatureExtractionError::ExtractionError {
                operation: "word_embedding_features".to_string(),
                reason: "Insufficient feature vector size for embeddings".to_string(),
            });
        }

        let words = &processed_text.tokens;
        if words.is_empty() {
            return Ok(());
        }

        // Average word embeddings
        for word in words {
            let embedding = self.get_word_embedding(word);
            for (j, &emb_val) in embedding.iter().enumerate().take(embedding_dim) {
                features[base_idx + j] += emb_val / words.len() as f64;
            }
        }

        Ok(())
    }

    fn add_contextual_features(
        &self,
        features: &mut [f64],
        processed_text: &ProcessedText,
    ) -> Result<(), FeatureExtractionError> {
        let base_idx = 64; // Start after embedding features
        let contextual_dim = 16; // Reserve 16 features for contextual analysis

        if features.len() < base_idx + contextual_dim {
            return Err(FeatureExtractionError::ExtractionError {
                operation: "contextual_features".to_string(),
                reason: "Insufficient feature vector size for contextual features".to_string(),
            });
        }

        let words = &processed_text.tokens;

        // Contextual diversity - how varied the word usage is
        let mut context_scores = Vec::new();
        for i in 0..words.len() {
            let start = i.saturating_sub(2);
            let end = (i + 3).min(words.len());
            let context_words: HashSet<_> = words[start..end].iter().collect();
            context_scores.push(context_words.len() as f64);
        }

        if !context_scores.is_empty() {
            let avg_context = context_scores.iter().sum::<f64>() / context_scores.len() as f64;
            features[base_idx] = (avg_context / 5.0).min(1.0); // Normalize context diversity
        }

        // Semantic coherence within windows
        let window_size = 5;
        let mut coherence_scores = Vec::new();

        for i in 0..=words.len().saturating_sub(window_size) {
            let window = &words[i..i + window_size];
            let coherence = self.compute_window_coherence(window);
            coherence_scores.push(coherence);
        }

        if !coherence_scores.is_empty() {
            let avg_coherence =
                coherence_scores.iter().sum::<f64>() / coherence_scores.len() as f64;
            features[base_idx + 1] = avg_coherence;
        }

        Ok(())
    }

    fn normalize_feature_vector(&self, features: &mut [f64]) {
        let magnitude: f64 = features.iter().map(|x| x * x).sum::<f64>().sqrt();
        if magnitude > 0.0 {
            for feature in features.iter_mut() {
                *feature /= magnitude;
            }
        }
    }

    fn compute_domain_scores(
        &self,
        words: &[String],
    ) -> Result<HashMap<String, f64>, FeatureExtractionError> {
        let mut domain_scores = HashMap::new();

        for (domain_name, domain_vocab) in &self.domain_vocabularies {
            let score = self.compute_domain_coverage(words, domain_vocab);
            domain_scores.insert(domain_name.clone(), score);
        }

        Ok(domain_scores)
    }

    fn compute_domain_coverage(&self, words: &[String], domain_vocab: &HashSet<String>) -> f64 {
        if words.is_empty() || domain_vocab.is_empty() {
            return 0.0;
        }

        let matches = words
            .iter()
            .filter(|word| domain_vocab.contains(*word))
            .count();

        matches as f64 / words.len() as f64
    }

    fn compute_word_weights(
        &self,
        processed_text: &ProcessedText,
    ) -> Result<HashMap<String, f64>, FeatureExtractionError> {
        let mut weights = HashMap::new();
        let total_words = processed_text.tokens.len() as f64;

        if total_words == 0.0 {
            return Ok(weights);
        }

        // TF-IDF-like weighting
        for (word, &frequency) in &processed_text.word_frequencies {
            let tf = frequency as f64 / total_words;

            // Simplified IDF calculation (assume corpus of common words)
            let is_common = self.stopwords.contains(word);
            let idf = if is_common { 1.0 } else { 2.0 };

            let weight = tf * idf;
            weights.insert(word.clone(), weight);
        }

        Ok(weights)
    }

    fn extract_sentiment_features(
        &self,
        words: &[String],
    ) -> Result<HashMap<String, f64>, FeatureExtractionError> {
        // Simplified sentiment analysis
        let positive_words = [
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "love",
            "best",
            "perfect",
            "outstanding",
        ];
        let negative_words = [
            "bad",
            "terrible",
            "awful",
            "horrible",
            "worst",
            "hate",
            "disappointing",
            "poor",
            "fail",
            "wrong",
        ];

        let mut positive_count = 0;
        let mut negative_count = 0;

        for word in words {
            if positive_words.contains(&word.as_str()) {
                positive_count += 1;
            } else if negative_words.contains(&word.as_str()) {
                negative_count += 1;
            }
        }

        let total_sentiment_words = positive_count + negative_count;
        let mut sentiment_scores = HashMap::new();

        if total_sentiment_words > 0 {
            sentiment_scores.insert(
                "positive".to_string(),
                positive_count as f64 / total_sentiment_words as f64,
            );
            sentiment_scores.insert(
                "negative".to_string(),
                negative_count as f64 / total_sentiment_words as f64,
            );
        } else {
            sentiment_scores.insert("positive".to_string(), 0.0);
            sentiment_scores.insert("negative".to_string(), 0.0);
        }

        sentiment_scores.insert(
            "neutral".to_string(),
            1.0 - sentiment_scores.get("positive").unwrap_or(&0.0)
                - sentiment_scores.get("negative").unwrap_or(&0.0),
        );

        Ok(sentiment_scores)
    }

    fn compute_topic_distribution(
        &self,
        words: &[String],
    ) -> Result<Vec<f64>, FeatureExtractionError> {
        // Simplified topic modeling using domain vocabularies
        let num_topics = 5; // Fixed number of topics
        let mut topic_scores = vec![0.0; num_topics];

        let topics = [
            "Technology",
            "Science",
            "Business",
            "Education",
            "Healthcare",
        ];

        for (i, topic_name) in topics.iter().enumerate() {
            if let Some(topic_vocab) = self.domain_vocabularies.get(*topic_name) {
                topic_scores[i] = self.compute_domain_coverage(words, topic_vocab);
            }
        }

        // Normalize topic distribution
        let total: f64 = topic_scores.iter().sum();
        if total > 0.0 {
            for score in &mut topic_scores {
                *score /= total;
            }
        }

        Ok(topic_scores)
    }

    fn get_word_embedding(&self, word: &str) -> Vec<f64> {
        if let Some(embedding) = self.embedding_cache.get(word) {
            embedding.clone()
        } else {
            self.generate_word_embedding(word)
        }
    }

    fn generate_word_embedding(&self, word: &str) -> Vec<f64> {
        // Simplified word embedding generation based on character patterns
        let mut embedding = vec![0.0; self.config.embedding_dimension];

        // Hash-based pseudo-random embedding
        let mut hash = 0u64;
        for (i, ch) in word.chars().enumerate() {
            hash = hash
                .wrapping_mul(31)
                .wrapping_add(ch as u64)
                .wrapping_add(i as u64);
        }

        for i in 0..self.config.embedding_dimension {
            let val = ((hash.wrapping_mul(i as u64 + 1)) % 1000) as f64 / 1000.0;
            embedding[i] = (val - 0.5) * 2.0; // Center around 0 with range [-1, 1]
        }

        embedding
    }

    fn compute_window_coherence(&self, words: &[String]) -> f64 {
        if words.len() < 2 {
            return 1.0;
        }

        let mut total_similarity = 0.0;
        let mut comparisons = 0;

        // Compare all pairs in the window
        for i in 0..words.len() {
            for j in (i + 1)..words.len() {
                let similarity = self.compute_word_similarity(&words[i], &words[j]);
                total_similarity += similarity;
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            total_similarity / comparisons as f64
        } else {
            0.0
        }
    }

    fn compute_word_similarity(&self, word1: &str, word2: &str) -> f64 {
        if word1 == word2 {
            return 1.0;
        }

        // Simple character-based similarity
        let chars1: HashSet<char> = word1.chars().collect();
        let chars2: HashSet<char> = word2.chars().collect();

        let intersection = chars1.intersection(&chars2).count();
        let union = chars1.union(&chars2).count();

        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for simple feature extraction

/// Extract basic semantic features from text
pub fn extract_basic_features(text: &str) -> Result<Vec<f64>, FeatureExtractionError> {
    let extractor = FeatureExtractor::new();
    let features = extractor.extract_features(text)?;
    Ok(features.features)
}

/// Extract features with custom dimension
pub fn extract_features_with_dimension(
    text: &str,
    dimension: usize,
) -> Result<Vec<f64>, FeatureExtractionError> {
    let config = FeatureExtractionConfig::new().with_dimension(dimension);
    let extractor = FeatureExtractor::with_config(config);
    let features = extractor.extract_features(text)?;
    Ok(features.features)
}

/// Extract comprehensive features with all components enabled
pub fn extract_comprehensive_features(
    text: &str,
) -> Result<SemanticFeatureVector, FeatureExtractionError> {
    let config = FeatureExtractionConfig::new()
        .with_strategy(FeatureExtraction::Complete)
        .with_semantic_domains(true)
        .with_word_embeddings(true)
        .with_topic_modeling(true)
        .with_sentiment_analysis(true);

    let extractor = FeatureExtractor::with_config(config);
    extractor.extract_features(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_extractor_creation() {
        let extractor = FeatureExtractor::new();
        assert_eq!(extractor.config.feature_dimension, 128);
        assert!(extractor.config.use_semantic_domains);
    }

    #[test]
    fn test_basic_feature_extraction() -> Result<(), FeatureExtractionError> {
        let extractor = FeatureExtractor::new();
        let text = "The computer processes data using advanced algorithms.";

        let features = extractor.extract_features(text)?;

        assert_eq!(features.features.len(), 128);
        assert!(features.metadata.word_count > 0);
        assert!(features.metadata.text_length > 0);
        assert!(!features.domain_scores.is_empty());
        assert!(!features.word_weights.is_empty());

        Ok(())
    }

    #[test]
    fn test_text_preprocessing() -> Result<(), FeatureExtractionError> {
        let extractor = FeatureExtractor::new();
        let text = "Hello, World! This is a TEST.";

        let processed = extractor.process_text(text)?;

        assert!(!processed.tokens.is_empty());
        assert!(processed.processed.to_lowercase().contains("hello"));
        assert!(processed.processed.to_lowercase().contains("world"));
        assert!(!processed.word_frequencies.is_empty());

        Ok(())
    }

    #[test]
    fn test_different_extraction_strategies() -> Result<(), FeatureExtractionError> {
        let text = "Artificial intelligence and machine learning are transforming technology.";

        // Test linguistic strategy
        let config_linguistic =
            FeatureExtractionConfig::new().with_strategy(FeatureExtraction::Linguistic);
        let extractor_linguistic = FeatureExtractor::with_config(config_linguistic);
        let features_linguistic = extractor_linguistic.extract_features(text)?;

        // Test comprehensive strategy
        let config_comprehensive =
            FeatureExtractionConfig::new().with_strategy(FeatureExtraction::Comprehensive);
        let extractor_comprehensive = FeatureExtractor::with_config(config_comprehensive);
        let features_comprehensive = extractor_comprehensive.extract_features(text)?;

        // Comprehensive should have more feature categories
        assert!(
            features_comprehensive.metadata.features_used.len()
                >= features_linguistic.metadata.features_used.len()
        );

        Ok(())
    }

    #[test]
    fn test_domain_scoring() -> Result<(), FeatureExtractionError> {
        let extractor = FeatureExtractor::new();
        let tech_text = "Programming algorithms with neural networks and databases.";
        let business_text = "Company revenue growth and market investment strategies.";

        let tech_features = extractor.extract_features(tech_text)?;
        let business_features = extractor.extract_features(business_text)?;

        // Technology text should score higher on Technology domain
        let tech_tech_score = tech_features
            .domain_scores
            .get("Technology")
            .unwrap_or(&0.0);
        let business_tech_score = business_features
            .domain_scores
            .get("Technology")
            .unwrap_or(&0.0);
        assert!(tech_tech_score > business_tech_score);

        // Business text should score higher on Business domain
        let tech_business_score = tech_features.domain_scores.get("Business").unwrap_or(&0.0);
        let business_business_score = business_features
            .domain_scores
            .get("Business")
            .unwrap_or(&0.0);
        assert!(business_business_score > tech_business_score);

        Ok(())
    }

    #[test]
    fn test_feature_importance() -> Result<(), FeatureExtractionError> {
        let extractor = FeatureExtractor::new();
        let text = "Advanced machine learning algorithms process large datasets efficiently.";

        let features = extractor.extract_features(text)?;
        let importance = extractor.get_feature_importance(&features);

        assert!(!importance.is_empty());

        // Check that importance scores are between 0 and 1
        for (_, &score) in &importance {
            assert!(score >= 0.0 && score <= 1.0);
        }

        Ok(())
    }

    #[test]
    fn test_configuration_options() -> Result<(), FeatureExtractionError> {
        let text = "Testing configuration options for feature extraction.";

        // Test with normalization enabled
        let config_norm = FeatureExtractionConfig::new().with_normalization(true);
        let extractor_norm = FeatureExtractor::with_config(config_norm);
        let features_norm = extractor_norm.extract_features(text)?;

        // Test with normalization disabled
        let config_no_norm = FeatureExtractionConfig::new().with_normalization(false);
        let extractor_no_norm = FeatureExtractor::with_config(config_no_norm);
        let features_no_norm = extractor_no_norm.extract_features(text)?;

        // Check metadata reflects configuration
        assert!(features_norm.metadata.normalization_applied);
        assert!(!features_no_norm.metadata.normalization_applied);

        Ok(())
    }

    #[test]
    fn test_convenience_functions() -> Result<(), FeatureExtractionError> {
        let text = "Simple test for convenience functions.";

        let basic_features = extract_basic_features(text)?;
        assert_eq!(basic_features.len(), 128);

        let custom_dim_features = extract_features_with_dimension(text, 64)?;
        assert_eq!(custom_dim_features.len(), 64);

        let comprehensive_features = extract_comprehensive_features(text)?;
        assert_eq!(comprehensive_features.features.len(), 128);
        assert!(comprehensive_features.sentiment_scores.is_some());

        Ok(())
    }

    #[test]
    fn test_error_handling() {
        let extractor = FeatureExtractor::new();

        // Test empty text
        let result = extractor.extract_features("");
        assert!(matches!(
            result,
            Err(FeatureExtractionError::InvalidInput { .. })
        ));

        // Test whitespace-only text
        let result = extractor.extract_features("   \n\t   ");
        assert!(matches!(
            result,
            Err(FeatureExtractionError::InvalidInput { .. })
        ));
    }

    #[test]
    fn test_stemming() {
        let config = FeatureExtractionConfig::new().with_preprocessing(true, false, true); // Enable stemming

        let extractor = FeatureExtractor::with_config(config);

        assert_eq!(extractor.stem_word("running"), "run");
        assert_eq!(extractor.stem_word("played"), "play");
        assert_eq!(extractor.stem_word("quickly"), "quick");
        assert_eq!(extractor.stem_word("books"), "book");
        assert_eq!(extractor.stem_word("cat"), "cat"); // Short words unchanged
    }

    #[test]
    fn test_word_embeddings() -> Result<(), FeatureExtractionError> {
        let extractor = FeatureExtractor::new();

        let embedding1 = extractor.get_word_embedding("computer");
        let embedding2 = extractor.get_word_embedding("technology");
        let embedding3 = extractor.get_word_embedding("computer"); // Same as embedding1

        assert_eq!(embedding1.len(), extractor.config.embedding_dimension);
        assert_eq!(embedding2.len(), extractor.config.embedding_dimension);
        assert_eq!(embedding1, embedding3); // Consistent embeddings

        Ok(())
    }
}
