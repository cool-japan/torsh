//! Sentiment Analysis Module
//!
//! This module provides comprehensive sentiment analysis capabilities for semantic similarity
//! computation. It analyzes emotional tone, polarity, and affective content in text to
//! enhance semantic understanding and enable sentiment-aware similarity calculations.
//!
//! # Key Features
//!
//! ## Sentiment Detection
//! - **Polarity Analysis**: Positive, negative, and neutral sentiment classification
//! - **Intensity Scoring**: Magnitude of sentiment expression
//! - **Compound Scoring**: Overall sentiment with direction and strength
//!
//! ## Advanced Analysis
//! - **Emotion Detection**: Multi-dimensional emotional analysis
//! - **Context-Aware Sentiment**: Context-sensitive sentiment interpretation
//! - **Sentiment Progression**: Sentiment flow and transitions within text
//!
//! ## Similarity Features
//! - **Sentiment Alignment**: Measure similarity between sentiment profiles
//! - **Emotional Resonance**: Detect emotional compatibility between texts
//! - **Affective Distance**: Quantify emotional dissimilarity
//!
//! # Usage Examples
//!
//! ```rust
//! use torsh_text::metrics::semantic::sentiment_analysis::{SentimentAnalyzer, SentimentConfig};
//!
//! let config = SentimentConfig::new()
//!     .with_emotion_detection(true)
//!     .with_context_analysis(true)
//!     .with_intensity_weighting(true);
//!
//! let analyzer = SentimentAnalyzer::with_config(config);
//!
//! let sentiment = analyzer.analyze_sentiment("I absolutely love this amazing product!")?;
//! println!("Sentiment: positive={:.3}, compound={:.3}",
//!          sentiment.positive, sentiment.compound);
//!
//! let similarity = analyzer.compute_sentiment_similarity(&sentiment1, &sentiment2)?;
//! println!("Sentiment similarity: {:.3}", similarity);
//! ```

use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Errors that can occur during sentiment analysis
#[derive(Error, Debug, Clone, PartialEq)]
pub enum SentimentAnalysisError {
    #[error("Invalid text input: {message}")]
    InvalidInput { message: String },
    #[error("Sentiment analysis failed: {operation} - {reason}")]
    AnalysisError { operation: String, reason: String },
    #[error("Configuration error: {parameter} = {value}")]
    ConfigurationError { parameter: String, value: String },
    #[error("Insufficient sentiment data for comparison")]
    InsufficientData,
}

/// Sentiment analysis configuration
#[derive(Debug, Clone)]
pub struct SentimentConfig {
    pub enable_emotion_detection: bool,
    pub enable_context_analysis: bool,
    pub enable_intensity_weighting: bool,
    pub enable_negation_handling: bool,
    pub use_custom_lexicon: bool,
    pub sentiment_threshold: f64,
    pub emotion_threshold: f64,
    pub context_window_size: usize,
    pub intensity_amplifier: f64,
}

impl Default for SentimentConfig {
    fn default() -> Self {
        Self {
            enable_emotion_detection: true,
            enable_context_analysis: false,
            enable_intensity_weighting: true,
            enable_negation_handling: true,
            use_custom_lexicon: false,
            sentiment_threshold: 0.1,
            emotion_threshold: 0.1,
            context_window_size: 3,
            intensity_amplifier: 1.5,
        }
    }
}

impl SentimentConfig {
    /// Create new configuration with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable emotion detection
    pub fn with_emotion_detection(mut self, enable: bool) -> Self {
        self.enable_emotion_detection = enable;
        self
    }

    /// Enable context-aware analysis
    pub fn with_context_analysis(mut self, enable: bool) -> Self {
        self.enable_context_analysis = enable;
        self
    }

    /// Enable intensity weighting
    pub fn with_intensity_weighting(mut self, enable: bool) -> Self {
        self.enable_intensity_weighting = enable;
        self
    }

    /// Enable negation handling
    pub fn with_negation_handling(mut self, enable: bool) -> Self {
        self.enable_negation_handling = enable;
        self
    }

    /// Set sentiment detection threshold
    pub fn with_sentiment_threshold(mut self, threshold: f64) -> Self {
        self.sentiment_threshold = threshold;
        self
    }

    /// Set context window size
    pub fn with_context_window(mut self, window_size: usize) -> Self {
        self.context_window_size = window_size;
        self
    }
}

/// Comprehensive sentiment analysis scores
#[derive(Debug, Clone, PartialEq)]
pub struct SentimentScores {
    /// Positive sentiment score (0.0 - 1.0)
    pub positive: f64,
    /// Negative sentiment score (0.0 - 1.0)
    pub negative: f64,
    /// Neutral sentiment score (0.0 - 1.0)
    pub neutral: f64,
    /// Compound sentiment score (-1.0 to 1.0)
    pub compound: f64,
    /// Sentiment intensity/confidence
    pub intensity: f64,
    /// Individual emotion scores
    pub emotions: Option<EmotionScores>,
    /// Context-aware sentiment adjustments
    pub context_adjustments: Option<ContextualSentiment>,
}

/// Multi-dimensional emotion analysis
#[derive(Debug, Clone, PartialEq)]
pub struct EmotionScores {
    pub joy: f64,
    pub anger: f64,
    pub fear: f64,
    pub sadness: f64,
    pub surprise: f64,
    pub disgust: f64,
    pub anticipation: f64,
    pub trust: f64,
}

/// Context-aware sentiment adjustments
#[derive(Debug, Clone, PartialEq)]
pub struct ContextualSentiment {
    /// Raw sentiment before context analysis
    pub raw_sentiment: f64,
    /// Context-adjusted sentiment
    pub adjusted_sentiment: f64,
    /// Confidence in context adjustment
    pub context_confidence: f64,
    /// Identified contextual modifiers
    pub modifiers: Vec<SentimentModifier>,
}

/// Sentiment modification patterns
#[derive(Debug, Clone, PartialEq)]
pub struct SentimentModifier {
    pub modifier_type: ModifierType,
    pub position: usize,
    pub strength: f64,
    pub scope: (usize, usize), // Start and end word positions
}

/// Types of sentiment modifiers
#[derive(Debug, Clone, PartialEq)]
pub enum ModifierType {
    Negation,    // "not", "never"
    Intensifier, // "very", "extremely"
    Diminisher,  // "slightly", "somewhat"
    Conditional, // "if", "perhaps"
    Contrast,    // "but", "however"
}

/// Sentiment progression analysis
#[derive(Debug, Clone, PartialEq)]
pub struct SentimentProgression {
    pub sentence_sentiments: Vec<SentimentScores>,
    pub overall_trend: SentimentTrend,
    pub volatility: f64,
    pub peak_sentiment: (usize, f64), // Position and strength of strongest sentiment
    pub sentiment_transitions: Vec<SentimentTransition>,
}

/// Sentiment trend over text
#[derive(Debug, Clone, PartialEq)]
pub enum SentimentTrend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
    Mixed,
}

/// Sentiment transition between text segments
#[derive(Debug, Clone, PartialEq)]
pub struct SentimentTransition {
    pub from_sentence: usize,
    pub to_sentence: usize,
    pub sentiment_change: f64,
    pub transition_type: TransitionType,
}

/// Types of sentiment transitions
#[derive(Debug, Clone, PartialEq)]
pub enum TransitionType {
    Smooth,   // Gradual change
    Sharp,    // Sudden change
    Reversal, // Complete flip
    Neutral,  // Little change
}

/// Sentiment similarity comparison result
#[derive(Debug, Clone, PartialEq)]
pub struct SentimentSimilarityResult {
    pub overall_similarity: f64,
    pub polarity_similarity: f64,
    pub intensity_similarity: f64,
    pub emotion_similarity: Option<f64>,
    pub context_similarity: Option<f64>,
    pub alignment_score: f64,
}

/// Main sentiment analyzer
pub struct SentimentAnalyzer {
    config: SentimentConfig,
    positive_lexicon: HashSet<String>,
    negative_lexicon: HashSet<String>,
    emotion_lexicon: HashMap<String, EmotionScores>,
    intensifier_lexicon: HashMap<String, f64>,
    negation_lexicon: HashSet<String>,
    modifier_patterns: Vec<ModifierPattern>,
}

/// Sentiment modifier patterns
#[derive(Debug, Clone)]
struct ModifierPattern {
    pattern: Vec<String>,
    modifier_type: ModifierType,
    strength: f64,
}

impl SentimentAnalyzer {
    /// Create new sentiment analyzer with default configuration
    pub fn new() -> Self {
        Self::with_config(SentimentConfig::default())
    }

    /// Create sentiment analyzer with custom configuration
    pub fn with_config(config: SentimentConfig) -> Self {
        let mut analyzer = Self {
            config,
            positive_lexicon: HashSet::new(),
            negative_lexicon: HashSet::new(),
            emotion_lexicon: HashMap::new(),
            intensifier_lexicon: HashMap::new(),
            negation_lexicon: HashSet::new(),
            modifier_patterns: Vec::new(),
        };

        analyzer.initialize_lexicons();
        analyzer.initialize_patterns();
        analyzer
    }

    /// Analyze sentiment of text
    pub fn analyze_sentiment(&self, text: &str) -> Result<SentimentScores, SentimentAnalysisError> {
        if text.trim().is_empty() {
            return Err(SentimentAnalysisError::InvalidInput {
                message: "Input text is empty".to_string(),
            });
        }

        let words = self.tokenize_and_normalize(text);

        // Basic sentiment analysis
        let (positive_score, negative_score, intensity) = self.compute_basic_sentiment(&words);

        // Apply context analysis if enabled
        let (adjusted_positive, adjusted_negative, context_adjustments) =
            if self.config.enable_context_analysis {
                let adjustments = self.analyze_context(&words, positive_score, negative_score)?;
                (
                    adjustments.adjusted_sentiment.max(0.0),
                    adjustments.adjusted_sentiment.min(0.0).abs(),
                    Some(adjustments),
                )
            } else {
                (positive_score, negative_score, None)
            };

        // Normalize scores
        let total = adjusted_positive + adjusted_negative;
        let (norm_positive, norm_negative) = if total > 0.0 {
            (adjusted_positive / total, adjusted_negative / total)
        } else {
            (0.0, 0.0)
        };

        let neutral = 1.0 - norm_positive - norm_negative;
        let compound = norm_positive - norm_negative;

        // Emotion analysis if enabled
        let emotions = if self.config.enable_emotion_detection {
            Some(self.analyze_emotions(&words)?)
        } else {
            None
        };

        Ok(SentimentScores {
            positive: norm_positive,
            negative: norm_negative,
            neutral: neutral.max(0.0),
            compound,
            intensity,
            emotions,
            context_adjustments,
        })
    }

    /// Analyze sentiment progression across text
    pub fn analyze_sentiment_progression(
        &self,
        text: &str,
    ) -> Result<SentimentProgression, SentimentAnalysisError> {
        let sentences = self.split_into_sentences(text);

        if sentences.is_empty() {
            return Err(SentimentAnalysisError::InvalidInput {
                message: "No sentences found in text".to_string(),
            });
        }

        let mut sentence_sentiments = Vec::new();
        for sentence in &sentences {
            sentence_sentiments.push(self.analyze_sentiment(sentence)?);
        }

        let overall_trend = self.determine_sentiment_trend(&sentence_sentiments);
        let volatility = self.calculate_sentiment_volatility(&sentence_sentiments);
        let peak_sentiment = self.find_peak_sentiment(&sentence_sentiments);
        let transitions = self.analyze_sentiment_transitions(&sentence_sentiments);

        Ok(SentimentProgression {
            sentence_sentiments,
            overall_trend,
            volatility,
            peak_sentiment,
            sentiment_transitions: transitions,
        })
    }

    /// Compute similarity between two sentiment profiles
    pub fn compute_sentiment_similarity(
        &self,
        sentiment1: &SentimentScores,
        sentiment2: &SentimentScores,
    ) -> Result<f64, SentimentAnalysisError> {
        let result = self.analyze_sentiment_similarity(sentiment1, sentiment2)?;
        Ok(result.overall_similarity)
    }

    /// Analyze detailed sentiment similarity
    pub fn analyze_sentiment_similarity(
        &self,
        sentiment1: &SentimentScores,
        sentiment2: &SentimentScores,
    ) -> Result<SentimentSimilarityResult, SentimentAnalysisError> {
        // Polarity similarity (positive/negative alignment)
        let polarity_sim = self.compute_polarity_similarity(sentiment1, sentiment2);

        // Intensity similarity
        let intensity_sim = 1.0 - (sentiment1.intensity - sentiment2.intensity).abs();

        // Emotion similarity if available
        let emotion_sim =
            if let (Some(emo1), Some(emo2)) = (&sentiment1.emotions, &sentiment2.emotions) {
                Some(self.compute_emotion_similarity(emo1, emo2))
            } else {
                None
            };

        // Context similarity if available
        let context_sim = if let (Some(ctx1), Some(ctx2)) = (
            &sentiment1.context_adjustments,
            &sentiment2.context_adjustments,
        ) {
            Some(self.compute_context_similarity(ctx1, ctx2))
        } else {
            None
        };

        // Overall alignment score
        let alignment_score = self.compute_alignment_score(sentiment1, sentiment2);

        // Weighted overall similarity
        let mut overall = polarity_sim * 0.4 + intensity_sim * 0.3 + alignment_score * 0.3;

        if let Some(emo_sim) = emotion_sim {
            overall = overall * 0.7 + emo_sim * 0.3;
        }

        if let Some(ctx_sim) = context_sim {
            overall = overall * 0.8 + ctx_sim * 0.2;
        }

        Ok(SentimentSimilarityResult {
            overall_similarity: overall,
            polarity_similarity: polarity_sim,
            intensity_similarity: intensity_sim,
            emotion_similarity: emotion_sim,
            context_similarity: context_sim,
            alignment_score,
        })
    }

    /// Compare sentiment profiles of multiple texts
    pub fn compare_multiple_sentiments(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Vec<f64>>, SentimentAnalysisError> {
        let sentiments: Result<Vec<_>, _> = texts
            .iter()
            .map(|text| self.analyze_sentiment(text))
            .collect();
        let sentiments = sentiments?;

        let n = sentiments.len();
        let mut similarity_matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    similarity_matrix[i][j] = 1.0;
                } else {
                    similarity_matrix[i][j] =
                        self.compute_sentiment_similarity(&sentiments[i], &sentiments[j])?;
                }
            }
        }

        Ok(similarity_matrix)
    }

    // Private helper methods

    fn initialize_lexicons(&mut self) {
        // Initialize positive sentiment lexicon
        let positive_words = vec![
            "amazing",
            "awesome",
            "beautiful",
            "best",
            "better",
            "brilliant",
            "excellent",
            "fantastic",
            "good",
            "great",
            "happy",
            "incredible",
            "love",
            "lovely",
            "nice",
            "outstanding",
            "perfect",
            "pleased",
            "positive",
            "satisfied",
            "superb",
            "terrific",
            "wonderful",
            "marvelous",
            "delightful",
            "enjoyable",
            "pleasant",
            "remarkable",
            "spectacular",
            "stunning",
            "fabulous",
            "magnificent",
            "glorious",
            "exceptional",
            "impressive",
            "admirable",
            "charming",
            "elegant",
            "graceful",
            "radiant",
            "vibrant",
        ];
        self.positive_lexicon = positive_words.into_iter().map(String::from).collect();

        // Initialize negative sentiment lexicon
        let negative_words = vec![
            "awful",
            "bad",
            "terrible",
            "horrible",
            "worst",
            "disappointing",
            "disgusting",
            "hate",
            "nasty",
            "pathetic",
            "poor",
            "sad",
            "angry",
            "annoyed",
            "frustrated",
            "upset",
            "miserable",
            "depressing",
            "dreadful",
            "shocking",
            "appalling",
            "disastrous",
            "catastrophic",
            "abysmal",
            "atrocious",
            "ghastly",
            "hideous",
            "loathsome",
            "repulsive",
            "revolting",
            "sickening",
            "vile",
            "wretched",
            "deplorable",
            "despicable",
            "detestable",
            "heinous",
            "horrendous",
            "monstrous",
        ];
        self.negative_lexicon = negative_words.into_iter().map(String::from).collect();

        // Initialize intensifier lexicon
        let intensifiers = vec![
            ("very", 1.5),
            ("extremely", 2.0),
            ("incredibly", 2.0),
            ("absolutely", 2.0),
            ("totally", 1.8),
            ("completely", 1.8),
            ("utterly", 2.0),
            ("highly", 1.5),
            ("quite", 1.3),
            ("rather", 1.2),
            ("really", 1.5),
            ("truly", 1.6),
            ("exceptionally", 1.9),
            ("remarkably", 1.7),
            ("extraordinarily", 2.1),
            ("tremendously", 1.8),
            ("immensely", 1.8),
            ("profoundly", 1.7),
        ];
        self.intensifier_lexicon = intensifiers
            .into_iter()
            .map(|(word, weight)| (word.to_string(), weight))
            .collect();

        // Initialize negation lexicon
        let negations = vec![
            "not",
            "never",
            "no",
            "none",
            "nothing",
            "neither",
            "nowhere",
            "nobody",
            "can't",
            "won't",
            "don't",
            "doesn't",
            "didn't",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
            "haven't",
            "hasn't",
            "hadn't",
            "shouldn't",
            "wouldn't",
            "couldn't",
        ];
        self.negation_lexicon = negations.into_iter().map(String::from).collect();

        // Initialize emotion lexicon (simplified)
        self.initialize_emotion_lexicon();
    }

    fn initialize_emotion_lexicon(&mut self) {
        let emotion_words = vec![
            (
                "joy",
                EmotionScores {
                    joy: 1.0,
                    anger: 0.0,
                    fear: 0.0,
                    sadness: 0.0,
                    surprise: 0.0,
                    disgust: 0.0,
                    anticipation: 0.3,
                    trust: 0.2,
                },
            ),
            (
                "happy",
                EmotionScores {
                    joy: 0.9,
                    anger: 0.0,
                    fear: 0.0,
                    sadness: 0.0,
                    surprise: 0.0,
                    disgust: 0.0,
                    anticipation: 0.2,
                    trust: 0.1,
                },
            ),
            (
                "angry",
                EmotionScores {
                    joy: 0.0,
                    anger: 1.0,
                    fear: 0.0,
                    sadness: 0.0,
                    surprise: 0.0,
                    disgust: 0.3,
                    anticipation: 0.0,
                    trust: 0.0,
                },
            ),
            (
                "afraid",
                EmotionScores {
                    joy: 0.0,
                    anger: 0.0,
                    fear: 1.0,
                    sadness: 0.2,
                    surprise: 0.0,
                    disgust: 0.0,
                    anticipation: 0.0,
                    trust: 0.0,
                },
            ),
            (
                "sad",
                EmotionScores {
                    joy: 0.0,
                    anger: 0.0,
                    fear: 0.1,
                    sadness: 1.0,
                    surprise: 0.0,
                    disgust: 0.0,
                    anticipation: 0.0,
                    trust: 0.0,
                },
            ),
            (
                "surprised",
                EmotionScores {
                    joy: 0.2,
                    anger: 0.0,
                    fear: 0.1,
                    sadness: 0.0,
                    surprise: 1.0,
                    disgust: 0.0,
                    anticipation: 0.0,
                    trust: 0.0,
                },
            ),
            (
                "disgusted",
                EmotionScores {
                    joy: 0.0,
                    anger: 0.4,
                    fear: 0.1,
                    sadness: 0.1,
                    surprise: 0.0,
                    disgust: 1.0,
                    anticipation: 0.0,
                    trust: 0.0,
                },
            ),
            (
                "excited",
                EmotionScores {
                    joy: 0.8,
                    anger: 0.0,
                    fear: 0.0,
                    sadness: 0.0,
                    surprise: 0.3,
                    disgust: 0.0,
                    anticipation: 0.9,
                    trust: 0.0,
                },
            ),
            (
                "trust",
                EmotionScores {
                    joy: 0.3,
                    anger: 0.0,
                    fear: 0.0,
                    sadness: 0.0,
                    surprise: 0.0,
                    disgust: 0.0,
                    anticipation: 0.2,
                    trust: 1.0,
                },
            ),
        ];

        for (word, emotions) in emotion_words {
            self.emotion_lexicon.insert(word.to_string(), emotions);
        }
    }

    fn initialize_patterns(&mut self) {
        // Initialize modifier patterns
        self.modifier_patterns = vec![
            ModifierPattern {
                pattern: vec!["not".to_string()],
                modifier_type: ModifierType::Negation,
                strength: -1.0,
            },
            ModifierPattern {
                pattern: vec!["very".to_string()],
                modifier_type: ModifierType::Intensifier,
                strength: 1.5,
            },
            ModifierPattern {
                pattern: vec!["extremely".to_string()],
                modifier_type: ModifierType::Intensifier,
                strength: 2.0,
            },
            ModifierPattern {
                pattern: vec!["slightly".to_string()],
                modifier_type: ModifierType::Diminisher,
                strength: 0.5,
            },
            ModifierPattern {
                pattern: vec!["but".to_string()],
                modifier_type: ModifierType::Contrast,
                strength: -0.3,
            },
        ];
    }

    fn tokenize_and_normalize(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|word| word.trim_matches(|c: char| !c.is_alphabetic()))
            .filter(|word| !word.is_empty())
            .map(String::from)
            .collect()
    }

    fn split_into_sentences(&self, text: &str) -> Vec<String> {
        text.split(|c| c == '.' || c == '!' || c == '?')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn compute_basic_sentiment(&self, words: &[String]) -> (f64, f64, f64) {
        let mut positive_score = 0.0;
        let mut negative_score = 0.0;
        let mut intensity_sum = 0.0;
        let mut word_count = 0;

        for (i, word) in words.iter().enumerate() {
            let mut base_score = 0.0;
            let mut is_sentiment_word = false;

            // Check for sentiment words
            if self.positive_lexicon.contains(word) {
                base_score = 1.0;
                is_sentiment_word = true;
            } else if self.negative_lexicon.contains(word) {
                base_score = -1.0;
                is_sentiment_word = true;
            }

            if is_sentiment_word {
                let mut final_score = base_score;
                let mut intensity = 1.0;

                // Apply intensity modifications
                if self.config.enable_intensity_weighting {
                    if let Some(intensifier_strength) = self.check_intensifiers(words, i) {
                        intensity *= intensifier_strength;
                        final_score *= intensifier_strength;
                    }
                }

                // Apply negation if enabled
                if self.config.enable_negation_handling {
                    if self.check_negation(words, i) {
                        final_score *= -1.0;
                    }
                }

                if final_score > 0.0 {
                    positive_score += final_score;
                } else {
                    negative_score += final_score.abs();
                }

                intensity_sum += intensity;
                word_count += 1;
            }
        }

        let avg_intensity = if word_count > 0 {
            intensity_sum / word_count as f64
        } else {
            0.0
        };

        (positive_score, negative_score, avg_intensity)
    }

    fn check_intensifiers(&self, words: &[String], position: usize) -> Option<f64> {
        // Check previous words for intensifiers
        for i in position.saturating_sub(2)..position {
            if let Some(&strength) = self.intensifier_lexicon.get(&words[i]) {
                return Some(strength);
            }
        }
        None
    }

    fn check_negation(&self, words: &[String], position: usize) -> bool {
        // Check previous words for negation
        for i in position.saturating_sub(3)..position {
            if self.negation_lexicon.contains(&words[i]) {
                return true;
            }
        }
        false
    }

    fn analyze_context(
        &self,
        words: &[String],
        positive: f64,
        negative: f64,
    ) -> Result<ContextualSentiment, SentimentAnalysisError> {
        let raw_sentiment = positive - negative;
        let mut modifiers = Vec::new();

        // Detect sentiment modifiers in context
        for (i, word) in words.iter().enumerate() {
            for pattern in &self.modifier_patterns {
                if pattern.pattern.len() == 1 && &pattern.pattern[0] == word {
                    let start = i.saturating_sub(1);
                    let end = (i + 2).min(words.len());

                    modifiers.push(SentimentModifier {
                        modifier_type: pattern.modifier_type.clone(),
                        position: i,
                        strength: pattern.strength,
                        scope: (start, end),
                    });
                }
            }
        }

        // Apply context adjustments
        let mut adjusted_sentiment = raw_sentiment;
        let mut context_confidence = 1.0;

        for modifier in &modifiers {
            match modifier.modifier_type {
                ModifierType::Negation => {
                    adjusted_sentiment *= -0.8; // Partial reversal
                    context_confidence *= 0.9;
                }
                ModifierType::Intensifier => {
                    adjusted_sentiment *= modifier.strength;
                    context_confidence *= 1.1;
                }
                ModifierType::Diminisher => {
                    adjusted_sentiment *= modifier.strength;
                    context_confidence *= 0.95;
                }
                ModifierType::Contrast => {
                    adjusted_sentiment *= (1.0 + modifier.strength);
                    context_confidence *= 0.8;
                }
                ModifierType::Conditional => {
                    adjusted_sentiment *= 0.7; // Reduce certainty
                    context_confidence *= 0.7;
                }
            }
        }

        context_confidence = context_confidence.clamp(0.0, 1.0);

        Ok(ContextualSentiment {
            raw_sentiment,
            adjusted_sentiment,
            context_confidence,
            modifiers,
        })
    }

    fn analyze_emotions(&self, words: &[String]) -> Result<EmotionScores, SentimentAnalysisError> {
        let mut emotion_totals = EmotionScores {
            joy: 0.0,
            anger: 0.0,
            fear: 0.0,
            sadness: 0.0,
            surprise: 0.0,
            disgust: 0.0,
            anticipation: 0.0,
            trust: 0.0,
        };

        let mut emotion_count = 0;

        for word in words {
            if let Some(emotions) = self.emotion_lexicon.get(word) {
                emotion_totals.joy += emotions.joy;
                emotion_totals.anger += emotions.anger;
                emotion_totals.fear += emotions.fear;
                emotion_totals.sadness += emotions.sadness;
                emotion_totals.surprise += emotions.surprise;
                emotion_totals.disgust += emotions.disgust;
                emotion_totals.anticipation += emotions.anticipation;
                emotion_totals.trust += emotions.trust;
                emotion_count += 1;
            }
        }

        if emotion_count > 0 {
            let count = emotion_count as f64;
            emotion_totals.joy /= count;
            emotion_totals.anger /= count;
            emotion_totals.fear /= count;
            emotion_totals.sadness /= count;
            emotion_totals.surprise /= count;
            emotion_totals.disgust /= count;
            emotion_totals.anticipation /= count;
            emotion_totals.trust /= count;
        }

        Ok(emotion_totals)
    }

    fn compute_polarity_similarity(
        &self,
        sentiment1: &SentimentScores,
        sentiment2: &SentimentScores,
    ) -> f64 {
        let vec1 = vec![sentiment1.positive, sentiment1.negative, sentiment1.neutral];
        let vec2 = vec![sentiment2.positive, sentiment2.negative, sentiment2.neutral];
        self.cosine_similarity(&vec1, &vec2)
    }

    fn compute_emotion_similarity(
        &self,
        emotions1: &EmotionScores,
        emotions2: &EmotionScores,
    ) -> f64 {
        let vec1 = vec![
            emotions1.joy,
            emotions1.anger,
            emotions1.fear,
            emotions1.sadness,
            emotions1.surprise,
            emotions1.disgust,
            emotions1.anticipation,
            emotions1.trust,
        ];
        let vec2 = vec![
            emotions2.joy,
            emotions2.anger,
            emotions2.fear,
            emotions2.sadness,
            emotions2.surprise,
            emotions2.disgust,
            emotions2.anticipation,
            emotions2.trust,
        ];
        self.cosine_similarity(&vec1, &vec2)
    }

    fn compute_context_similarity(
        &self,
        context1: &ContextualSentiment,
        context2: &ContextualSentiment,
    ) -> f64 {
        let confidence_sim =
            1.0 - (context1.context_confidence - context2.context_confidence).abs();
        let adjustment_sim =
            1.0 - (context1.adjusted_sentiment - context2.adjusted_sentiment).abs() / 2.0;
        (confidence_sim + adjustment_sim) / 2.0
    }

    fn compute_alignment_score(
        &self,
        sentiment1: &SentimentScores,
        sentiment2: &SentimentScores,
    ) -> f64 {
        // Check if sentiments are aligned (same polarity) or opposed
        let polarity1 = if sentiment1.compound > 0.1 {
            1.0
        } else if sentiment1.compound < -0.1 {
            -1.0
        } else {
            0.0
        };
        let polarity2 = if sentiment2.compound > 0.1 {
            1.0
        } else if sentiment2.compound < -0.1 {
            -1.0
        } else {
            0.0
        };

        if polarity1 * polarity2 > 0.0 {
            // Same polarity - compute strength alignment
            1.0 - (sentiment1.intensity - sentiment2.intensity).abs() / 2.0
        } else if polarity1 * polarity2 < 0.0 {
            // Opposite polarity - very low similarity
            0.1
        } else {
            // One or both neutral
            0.5
        }
    }

    fn cosine_similarity(&self, vec1: &[f64], vec2: &[f64]) -> f64 {
        if vec1.len() != vec2.len() {
            return 0.0;
        }

        let dot_product: f64 = vec1.iter().zip(vec2).map(|(a, b)| a * b).sum();
        let norm1: f64 = vec1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = vec2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            (dot_product / (norm1 * norm2)).max(0.0).min(1.0)
        }
    }

    fn determine_sentiment_trend(&self, sentiments: &[SentimentScores]) -> SentimentTrend {
        if sentiments.len() < 2 {
            return SentimentTrend::Stable;
        }

        let compounds: Vec<f64> = sentiments.iter().map(|s| s.compound).collect();
        let mut increases = 0;
        let mut decreases = 0;
        let mut total_change = 0.0;

        for i in 1..compounds.len() {
            let change = compounds[i] - compounds[i - 1];
            total_change += change.abs();

            if change > 0.1 {
                increases += 1;
            } else if change < -0.1 {
                decreases += 1;
            }
        }

        let avg_change = total_change / (compounds.len() - 1) as f64;

        if avg_change > 0.3 {
            SentimentTrend::Volatile
        } else if increases > decreases * 2 {
            SentimentTrend::Increasing
        } else if decreases > increases * 2 {
            SentimentTrend::Decreasing
        } else if increases > 0 && decreases > 0 {
            SentimentTrend::Mixed
        } else {
            SentimentTrend::Stable
        }
    }

    fn calculate_sentiment_volatility(&self, sentiments: &[SentimentScores]) -> f64 {
        if sentiments.len() < 2 {
            return 0.0;
        }

        let compounds: Vec<f64> = sentiments.iter().map(|s| s.compound).collect();
        let mean = compounds.iter().sum::<f64>() / compounds.len() as f64;
        let variance =
            compounds.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / compounds.len() as f64;

        variance.sqrt()
    }

    fn find_peak_sentiment(&self, sentiments: &[SentimentScores]) -> (usize, f64) {
        sentiments
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.compound.abs().partial_cmp(&b.compound.abs()).unwrap())
            .map(|(i, s)| (i, s.compound))
            .unwrap_or((0, 0.0))
    }

    fn analyze_sentiment_transitions(
        &self,
        sentiments: &[SentimentScores],
    ) -> Vec<SentimentTransition> {
        let mut transitions = Vec::new();

        for i in 1..sentiments.len() {
            let change = sentiments[i].compound - sentiments[i - 1].compound;
            let transition_type = match change.abs() {
                x if x > 0.5 => {
                    if change * sentiments[i - 1].compound < 0.0 {
                        TransitionType::Reversal
                    } else {
                        TransitionType::Sharp
                    }
                }
                x if x > 0.2 => TransitionType::Smooth,
                _ => TransitionType::Neutral,
            };

            transitions.push(SentimentTransition {
                from_sentence: i - 1,
                to_sentence: i,
                sentiment_change: change,
                transition_type,
            });
        }

        transitions
    }
}

impl Default for SentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for simple sentiment analysis

/// Analyze basic sentiment of text
pub fn analyze_basic_sentiment(text: &str) -> Result<SentimentScores, SentimentAnalysisError> {
    let analyzer = SentimentAnalyzer::new();
    analyzer.analyze_sentiment(text)
}

/// Compute sentiment similarity between two texts
pub fn compute_sentiment_similarity_simple(
    text1: &str,
    text2: &str,
) -> Result<f64, SentimentAnalysisError> {
    let analyzer = SentimentAnalyzer::new();
    let sentiment1 = analyzer.analyze_sentiment(text1)?;
    let sentiment2 = analyzer.analyze_sentiment(text2)?;
    analyzer.compute_sentiment_similarity(&sentiment1, &sentiment2)
}

/// Analyze sentiment with emotion detection
pub fn analyze_sentiment_with_emotions(
    text: &str,
) -> Result<SentimentScores, SentimentAnalysisError> {
    let config = SentimentConfig::new().with_emotion_detection(true);
    let analyzer = SentimentAnalyzer::with_config(config);
    analyzer.analyze_sentiment(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentiment_analyzer_creation() {
        let analyzer = SentimentAnalyzer::new();
        assert!(!analyzer.positive_lexicon.is_empty());
        assert!(!analyzer.negative_lexicon.is_empty());
    }

    #[test]
    fn test_positive_sentiment_analysis() -> Result<(), SentimentAnalysisError> {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze_sentiment("This is absolutely amazing and wonderful!")?;

        assert!(result.positive > 0.5);
        assert!(result.compound > 0.0);
        assert!(result.negative < result.positive);

        Ok(())
    }

    #[test]
    fn test_negative_sentiment_analysis() -> Result<(), SentimentAnalysisError> {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze_sentiment("This is terrible and awful!")?;

        assert!(result.negative > 0.5);
        assert!(result.compound < 0.0);
        assert!(result.positive < result.negative);

        Ok(())
    }

    #[test]
    fn test_neutral_sentiment_analysis() -> Result<(), SentimentAnalysisError> {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze_sentiment("The weather is cloudy today.")?;

        assert!(result.neutral > 0.5);
        assert!(result.compound.abs() < 0.3);

        Ok(())
    }

    #[test]
    fn test_negation_handling() -> Result<(), SentimentAnalysisError> {
        let config = SentimentConfig::new().with_negation_handling(true);
        let analyzer = SentimentAnalyzer::with_config(config);

        let positive_result = analyzer.analyze_sentiment("This is good.")?;
        let negated_result = analyzer.analyze_sentiment("This is not good.")?;

        assert!(positive_result.compound > 0.0);
        assert!(negated_result.compound < 0.0);

        Ok(())
    }

    #[test]
    fn test_intensity_weighting() -> Result<(), SentimentAnalysisError> {
        let config = SentimentConfig::new().with_intensity_weighting(true);
        let analyzer = SentimentAnalyzer::with_config(config);

        let normal_result = analyzer.analyze_sentiment("This is good.")?;
        let intense_result = analyzer.analyze_sentiment("This is extremely good.")?;

        assert!(intense_result.intensity > normal_result.intensity);

        Ok(())
    }

    #[test]
    fn test_emotion_analysis() -> Result<(), SentimentAnalysisError> {
        let config = SentimentConfig::new().with_emotion_detection(true);
        let analyzer = SentimentAnalyzer::with_config(config);

        let result = analyzer.analyze_sentiment("I am so happy and excited!")?;

        assert!(result.emotions.is_some());
        if let Some(emotions) = result.emotions {
            assert!(emotions.joy > 0.0);
        }

        Ok(())
    }

    #[test]
    fn test_sentiment_similarity() -> Result<(), SentimentAnalysisError> {
        let analyzer = SentimentAnalyzer::new();

        let sentiment1 = analyzer.analyze_sentiment("This is great and amazing!")?;
        let sentiment2 = analyzer.analyze_sentiment("This is wonderful and fantastic!")?;
        let sentiment3 = analyzer.analyze_sentiment("This is terrible and awful!")?;

        let similarity_positive =
            analyzer.compute_sentiment_similarity(&sentiment1, &sentiment2)?;
        let similarity_mixed = analyzer.compute_sentiment_similarity(&sentiment1, &sentiment3)?;

        assert!(similarity_positive > similarity_mixed);
        assert!(similarity_positive > 0.5);

        Ok(())
    }

    #[test]
    fn test_sentiment_progression() -> Result<(), SentimentAnalysisError> {
        let analyzer = SentimentAnalyzer::new();
        let text = "I love this product! It works perfectly. However, it broke after a week. I'm very disappointed.";

        let progression = analyzer.analyze_sentiment_progression(text)?;

        assert!(!progression.sentence_sentiments.is_empty());
        assert!(!progression.sentiment_transitions.is_empty());
        assert!(progression.volatility > 0.0);

        Ok(())
    }

    #[test]
    fn test_context_analysis() -> Result<(), SentimentAnalysisError> {
        let config = SentimentConfig::new().with_context_analysis(true);
        let analyzer = SentimentAnalyzer::with_config(config);

        let result =
            analyzer.analyze_sentiment("This movie is good, but the ending was disappointing.")?;

        assert!(result.context_adjustments.is_some());

        Ok(())
    }

    #[test]
    fn test_multiple_text_comparison() -> Result<(), SentimentAnalysisError> {
        let analyzer = SentimentAnalyzer::new();
        let texts = vec![
            "I love this!",
            "This is amazing!",
            "This is terrible.",
            "This is awful.",
        ];

        let similarities = analyzer.compare_multiple_sentiments(&texts)?;

        assert_eq!(similarities.len(), 4);
        assert_eq!(similarities[0].len(), 4);

        // Similar sentiments should have high similarity
        assert!(similarities[0][1] > 0.7); // Both positive
        assert!(similarities[2][3] > 0.7); // Both negative

        // Opposite sentiments should have low similarity
        assert!(similarities[0][2] < 0.3); // Positive vs negative

        Ok(())
    }

    #[test]
    fn test_convenience_functions() -> Result<(), SentimentAnalysisError> {
        let result = analyze_basic_sentiment("This is a great product!")?;
        assert!(result.positive > 0.5);

        let similarity = compute_sentiment_similarity_simple("Great product!", "Amazing item!")?;
        assert!(similarity > 0.5);

        let emotion_result = analyze_sentiment_with_emotions("I am so happy!")?;
        assert!(emotion_result.emotions.is_some());

        Ok(())
    }

    #[test]
    fn test_error_handling() {
        let analyzer = SentimentAnalyzer::new();

        // Test empty text
        let result = analyzer.analyze_sentiment("");
        assert!(matches!(
            result,
            Err(SentimentAnalysisError::InvalidInput { .. })
        ));

        // Test whitespace-only text
        let result = analyzer.analyze_sentiment("   \n\t   ");
        assert!(matches!(
            result,
            Err(SentimentAnalysisError::InvalidInput { .. })
        ));
    }
}
