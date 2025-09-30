//! Topic Modeling Module
//!
//! This module provides advanced topic modeling capabilities for semantic analysis.
//! It supports multiple topic modeling approaches including keyword-based clustering,
//! TF-IDF topic extraction, latent semantic analysis, and dynamic topic modeling
//! to enhance semantic similarity computation through topical understanding.
//!
//! # Key Features
//!
//! ## Topic Modeling Approaches
//! - **Keyword Clustering**: Topic identification through predefined keyword sets
//! - **TF-IDF Topics**: Topic extraction using term frequency-inverse document frequency
//! - **Latent Semantic**: Dimensionality reduction-based topic discovery
//! - **Dynamic Modeling**: Time-aware topic evolution and trends
//! - **Hierarchical Topics**: Multi-level topic organization and relationships
//!
//! ## Advanced Analysis
//! - **Topic Evolution**: Track topic changes over document sections
//! - **Topic Coherence**: Measure internal topic consistency and quality
//! - **Cross-Topic Similarity**: Analyze relationships between different topics
//! - **Topic Diversity**: Measure topical breadth and coverage
//!
//! ## Similarity Features
//! - **Topic Distribution Similarity**: Compare topic probability distributions
//! - **Dominant Topic Matching**: Focus on primary topic alignments
//! - **Topic Overlap Analysis**: Quantify shared topical content
//! - **Thematic Progression**: Analyze topic flow and transitions
//!
//! # Usage Examples
//!
//! ```rust
//! use torsh_text::metrics::semantic::topic_modeling::{TopicModeler, TopicModelingConfig, TopicModelingApproach};
//!
//! let config = TopicModelingConfig::new()
//!     .with_approach(TopicModelingApproach::TfIdf)
//!     .with_num_topics(10)
//!     .with_dynamic_modeling(true);
//!
//! let modeler = TopicModeler::with_config(config);
//!
//! let topics = modeler.extract_topics("Your text here...")?;
//! println!("Dominant topic: {:?}", topics.dominant_topic);
//!
//! let similarity = modeler.compute_topic_similarity(&topics1, &topics2)?;
//! println!("Topic similarity: {:.3}", similarity);
//! ```

use std::collections::{HashMap, HashSet};
use thiserror::Error;

/// Errors that can occur during topic modeling
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TopicModelingError {
    #[error("Invalid input text: {message}")]
    InvalidInput { message: String },
    #[error("Topic modeling failed: {operation} - {reason}")]
    ModelingError { operation: String, reason: String },
    #[error("Configuration error: {parameter} = {value}")]
    ConfigurationError { parameter: String, value: String },
    #[error("Insufficient vocabulary for topic modeling: {word_count} words found")]
    InsufficientVocabulary { word_count: usize },
    #[error("Topic analysis failed: {topic_id}")]
    TopicAnalysisError { topic_id: String },
}

/// Topic modeling approaches
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TopicModelingApproach {
    /// Keyword-based topic clustering
    KeywordClustering,
    /// TF-IDF based topic extraction
    TfIdf,
    /// Latent semantic analysis
    LatentSemantic,
    /// Co-occurrence based topics
    CoOccurrence,
    /// Hierarchical topic modeling
    Hierarchical,
    /// Dynamic topic modeling with evolution
    Dynamic,
}

/// Configuration for topic modeling
#[derive(Debug, Clone)]
pub struct TopicModelingConfig {
    pub approach: TopicModelingApproach,
    pub num_topics: usize,
    pub min_topic_coherence: f64,
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub use_custom_topics: bool,
    pub enable_topic_evolution: bool,
    pub enable_hierarchical_analysis: bool,
    pub vocabulary_threshold: usize,
    pub topic_diversity_weight: f64,
    pub coherence_weight: f64,
}

impl Default for TopicModelingConfig {
    fn default() -> Self {
        Self {
            approach: TopicModelingApproach::KeywordClustering,
            num_topics: 8,
            min_topic_coherence: 0.3,
            max_iterations: 100,
            convergence_threshold: 0.001,
            use_custom_topics: false,
            enable_topic_evolution: false,
            enable_hierarchical_analysis: false,
            vocabulary_threshold: 10,
            topic_diversity_weight: 0.3,
            coherence_weight: 0.7,
        }
    }
}

impl TopicModelingConfig {
    /// Create new configuration with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set topic modeling approach
    pub fn with_approach(mut self, approach: TopicModelingApproach) -> Self {
        self.approach = approach;
        self
    }

    /// Set number of topics to extract
    pub fn with_num_topics(mut self, num_topics: usize) -> Self {
        self.num_topics = num_topics;
        self
    }

    /// Set minimum topic coherence threshold
    pub fn with_min_coherence(mut self, coherence: f64) -> Self {
        self.min_topic_coherence = coherence;
        self
    }

    /// Enable dynamic topic modeling
    pub fn with_dynamic_modeling(mut self, enable: bool) -> Self {
        self.enable_topic_evolution = enable;
        self
    }

    /// Enable hierarchical topic analysis
    pub fn with_hierarchical_analysis(mut self, enable: bool) -> Self {
        self.enable_hierarchical_analysis = enable;
        self
    }

    /// Set vocabulary threshold
    pub fn with_vocabulary_threshold(mut self, threshold: usize) -> Self {
        self.vocabulary_threshold = threshold;
        self
    }
}

/// Comprehensive topic modeling result
#[derive(Debug, Clone, PartialEq)]
pub struct TopicModelingResult {
    /// Topic probability distribution
    pub topic_distribution: Vec<f64>,
    /// Individual topic information
    pub topics: Vec<Topic>,
    /// Dominant topic
    pub dominant_topic: Option<Topic>,
    /// Topic diversity score
    pub diversity_score: f64,
    /// Overall topic coherence
    pub coherence_score: f64,
    /// Topic evolution (if enabled)
    pub topic_evolution: Option<TopicEvolution>,
    /// Hierarchical structure (if enabled)
    pub hierarchical_structure: Option<TopicHierarchy>,
    /// Modeling metadata
    pub metadata: TopicModelingMetadata,
}

/// Individual topic information
#[derive(Debug, Clone, PartialEq)]
pub struct Topic {
    pub id: String,
    pub name: String,
    pub probability: f64,
    pub keywords: Vec<TopicKeyword>,
    pub coherence: f64,
    pub representative_words: Vec<String>,
    pub topic_type: TopicType,
}

/// Topic keyword with relevance score
#[derive(Debug, Clone, PartialEq)]
pub struct TopicKeyword {
    pub word: String,
    pub weight: f64,
    pub frequency: usize,
    pub distinctiveness: f64,
}

/// Types of topics
#[derive(Debug, Clone, PartialEq)]
pub enum TopicType {
    Thematic,   // Content-based topic
    Functional, // Structure/style-based topic
    Sentiment,  // Emotion-based topic
    Domain,     // Domain-specific topic
    Mixed,      // Combination of types
}

/// Topic evolution analysis
#[derive(Debug, Clone, PartialEq)]
pub struct TopicEvolution {
    /// Topic distributions across text segments
    pub segment_distributions: Vec<Vec<f64>>,
    /// Topic transitions between segments
    pub transitions: Vec<TopicTransition>,
    /// Evolution patterns
    pub evolution_patterns: Vec<EvolutionPattern>,
    /// Topic stability scores
    pub stability_scores: Vec<f64>,
}

/// Topic transition information
#[derive(Debug, Clone, PartialEq)]
pub struct TopicTransition {
    pub from_segment: usize,
    pub to_segment: usize,
    pub topic_changes: HashMap<String, f64>,
    pub transition_strength: f64,
    pub transition_type: TransitionType,
}

/// Types of topic transitions
#[derive(Debug, Clone, PartialEq)]
pub enum TransitionType {
    Smooth,     // Gradual topic shift
    Sharp,      // Sudden topic change
    Cyclical,   // Return to previous topic
    Divergent,  // Multiple new topics emerge
    Convergent, // Multiple topics merge
}

/// Topic evolution patterns
#[derive(Debug, Clone, PartialEq)]
pub struct EvolutionPattern {
    pub pattern_type: PatternType,
    pub affected_topics: Vec<String>,
    pub strength: f64,
    pub segments: (usize, usize), // Start and end segments
}

/// Types of evolution patterns
#[derive(Debug, Clone, PartialEq)]
pub enum PatternType {
    Emergence,       // New topic appears
    Decay,           // Topic fades away
    Persistence,     // Topic remains stable
    Oscillation,     // Topic comes and goes
    Intensification, // Topic becomes stronger
}

/// Hierarchical topic structure
#[derive(Debug, Clone, PartialEq)]
pub struct TopicHierarchy {
    /// Root topics (highest level)
    pub root_topics: Vec<Topic>,
    /// Topic relationships
    pub relationships: Vec<TopicRelationship>,
    /// Hierarchy levels
    pub levels: Vec<TopicLevel>,
    /// Topic clustering information
    pub clusters: Vec<TopicCluster>,
}

/// Relationship between topics
#[derive(Debug, Clone, PartialEq)]
pub struct TopicRelationship {
    pub parent_topic: String,
    pub child_topic: String,
    pub relationship_type: RelationshipType,
    pub strength: f64,
}

/// Types of topic relationships
#[derive(Debug, Clone, PartialEq)]
pub enum RelationshipType {
    SubTopic,   // Child is subtopic of parent
    Related,    // Topics are thematically related
    Opposite,   // Topics are contrasting
    Sequential, // Topics follow in sequence
}

/// Level in topic hierarchy
#[derive(Debug, Clone, PartialEq)]
pub struct TopicLevel {
    pub level: usize,
    pub topics: Vec<String>,
    pub coherence: f64,
}

/// Topic cluster information
#[derive(Debug, Clone, PartialEq)]
pub struct TopicCluster {
    pub cluster_id: String,
    pub topics: Vec<String>,
    pub centroid: Vec<f64>,
    pub coherence: f64,
    pub diversity: f64,
}

/// Topic modeling metadata
#[derive(Debug, Clone, PartialEq)]
pub struct TopicModelingMetadata {
    pub approach_used: TopicModelingApproach,
    pub vocabulary_size: usize,
    pub iterations_performed: usize,
    pub convergence_achieved: bool,
    pub modeling_time_ms: u64,
    pub quality_metrics: TopicQualityMetrics,
}

/// Quality metrics for topic modeling
#[derive(Debug, Clone, PartialEq)]
pub struct TopicQualityMetrics {
    pub perplexity: f64,
    pub coherence: f64,
    pub diversity: f64,
    pub interpretability: f64,
    pub stability: f64,
}

/// Topic similarity comparison result
#[derive(Debug, Clone, PartialEq)]
pub struct TopicSimilarityResult {
    pub overall_similarity: f64,
    pub distribution_similarity: f64,
    pub dominant_topic_similarity: f64,
    pub keyword_overlap_similarity: f64,
    pub evolution_similarity: Option<f64>,
    pub hierarchical_similarity: Option<f64>,
}

/// Main topic modeler
pub struct TopicModeler {
    config: TopicModelingConfig,
    predefined_topics: HashMap<String, HashSet<String>>,
    topic_hierarchies: HashMap<String, Vec<String>>,
    vocabulary: HashSet<String>,
    word_cooccurrence: HashMap<String, HashMap<String, usize>>,
}

impl TopicModeler {
    /// Create new topic modeler with default configuration
    pub fn new() -> Self {
        Self::with_config(TopicModelingConfig::default())
    }

    /// Create topic modeler with custom configuration
    pub fn with_config(config: TopicModelingConfig) -> Self {
        let mut modeler = Self {
            config,
            predefined_topics: HashMap::new(),
            topic_hierarchies: HashMap::new(),
            vocabulary: HashSet::new(),
            word_cooccurrence: HashMap::new(),
        };

        modeler.initialize_predefined_topics();
        modeler.initialize_hierarchies();
        modeler
    }

    /// Extract topics from text using configured approach
    pub fn extract_topics(
        &mut self,
        text: &str,
    ) -> Result<TopicModelingResult, TopicModelingError> {
        let start_time = std::time::Instant::now();

        if text.trim().is_empty() {
            return Err(TopicModelingError::InvalidInput {
                message: "Input text is empty".to_string(),
            });
        }

        // Preprocess and build vocabulary
        let words = self.preprocess_text(text)?;
        self.build_vocabulary(&words);

        if self.vocabulary.len() < self.config.vocabulary_threshold {
            return Err(TopicModelingError::InsufficientVocabulary {
                word_count: self.vocabulary.len(),
            });
        }

        // Extract topics based on configured approach
        let (topic_distribution, topics) = match self.config.approach {
            TopicModelingApproach::KeywordClustering => self.extract_keyword_topics(&words)?,
            TopicModelingApproach::TfIdf => self.extract_tfidf_topics(&words)?,
            TopicModelingApproach::LatentSemantic => self.extract_latent_semantic_topics(&words)?,
            TopicModelingApproach::CoOccurrence => self.extract_cooccurrence_topics(&words)?,
            TopicModelingApproach::Hierarchical => self.extract_hierarchical_topics(&words)?,
            TopicModelingApproach::Dynamic => self.extract_dynamic_topics(text)?,
        };

        // Find dominant topic
        let dominant_topic = self.find_dominant_topic(&topics);

        // Calculate diversity and coherence scores
        let diversity_score = self.calculate_topic_diversity(&topic_distribution);
        let coherence_score = self.calculate_overall_coherence(&topics);

        // Extract topic evolution if enabled
        let topic_evolution = if self.config.enable_topic_evolution {
            Some(self.analyze_topic_evolution(text, &topics)?)
        } else {
            None
        };

        // Extract hierarchical structure if enabled
        let hierarchical_structure = if self.config.enable_hierarchical_analysis {
            Some(self.build_topic_hierarchy(&topics)?)
        } else {
            None
        };

        let modeling_time = start_time.elapsed().as_millis() as u64;

        // Create quality metrics
        let quality_metrics = self.calculate_quality_metrics(&topics, &topic_distribution);

        let metadata = TopicModelingMetadata {
            approach_used: self.config.approach,
            vocabulary_size: self.vocabulary.len(),
            iterations_performed: 1, // Simplified for now
            convergence_achieved: true,
            modeling_time_ms: modeling_time,
            quality_metrics,
        };

        Ok(TopicModelingResult {
            topic_distribution,
            topics,
            dominant_topic,
            diversity_score,
            coherence_score,
            topic_evolution,
            hierarchical_structure,
            metadata,
        })
    }

    /// Compute similarity between two topic modeling results
    pub fn compute_topic_similarity(
        &self,
        result1: &TopicModelingResult,
        result2: &TopicModelingResult,
    ) -> Result<f64, TopicModelingError> {
        let similarity_result = self.analyze_topic_similarity(result1, result2)?;
        Ok(similarity_result.overall_similarity)
    }

    /// Analyze detailed topic similarity
    pub fn analyze_topic_similarity(
        &self,
        result1: &TopicModelingResult,
        result2: &TopicModelingResult,
    ) -> Result<TopicSimilarityResult, TopicModelingError> {
        // Distribution similarity using cosine similarity
        let distribution_similarity =
            self.cosine_similarity(&result1.topic_distribution, &result2.topic_distribution);

        // Dominant topic similarity
        let dominant_topic_similarity = match (&result1.dominant_topic, &result2.dominant_topic) {
            (Some(topic1), Some(topic2)) => {
                self.compute_individual_topic_similarity(topic1, topic2)
            }
            (None, None) => 1.0,
            _ => 0.0,
        };

        // Keyword overlap similarity
        let keyword_overlap_similarity =
            self.compute_keyword_overlap_similarity(&result1.topics, &result2.topics);

        // Evolution similarity if both have evolution data
        let evolution_similarity = match (&result1.topic_evolution, &result2.topic_evolution) {
            (Some(evo1), Some(evo2)) => Some(self.compute_evolution_similarity(evo1, evo2)),
            _ => None,
        };

        // Hierarchical similarity if both have hierarchical data
        let hierarchical_similarity = match (
            &result1.hierarchical_structure,
            &result2.hierarchical_structure,
        ) {
            (Some(hier1), Some(hier2)) => Some(self.compute_hierarchical_similarity(hier1, hier2)),
            _ => None,
        };

        // Weighted overall similarity
        let mut overall_similarity = distribution_similarity * 0.4
            + dominant_topic_similarity * 0.3
            + keyword_overlap_similarity * 0.3;

        if let Some(evo_sim) = evolution_similarity {
            overall_similarity = overall_similarity * 0.8 + evo_sim * 0.2;
        }

        if let Some(hier_sim) = hierarchical_similarity {
            overall_similarity = overall_similarity * 0.9 + hier_sim * 0.1;
        }

        Ok(TopicSimilarityResult {
            overall_similarity,
            distribution_similarity,
            dominant_topic_similarity,
            keyword_overlap_similarity,
            evolution_similarity,
            hierarchical_similarity,
        })
    }

    /// Compare topics across multiple texts
    pub fn compare_multiple_topics(
        &mut self,
        texts: &[&str],
    ) -> Result<Vec<Vec<f64>>, TopicModelingError> {
        let results: Result<Vec<_>, _> =
            texts.iter().map(|text| self.extract_topics(text)).collect();
        let results = results?;

        let n = results.len();
        let mut similarity_matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    similarity_matrix[i][j] = 1.0;
                } else {
                    similarity_matrix[i][j] =
                        self.compute_topic_similarity(&results[i], &results[j])?;
                }
            }
        }

        Ok(similarity_matrix)
    }

    // Private helper methods

    fn initialize_predefined_topics(&mut self) {
        // Technology topics
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
            "artificial",
            "intelligence",
            "machine",
            "learning",
            "database",
            "server",
            "cloud",
            "cybersecurity",
            "blockchain",
            "automation",
            "robotics",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        self.predefined_topics
            .insert("Technology".to_string(), tech_words);

        // Science topics
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
            "biology",
            "chemistry",
            "physics",
            "mathematics",
            "genetics",
            "ecology",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        self.predefined_topics
            .insert("Science".to_string(), science_words);

        // Business topics
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
            "entrepreneur",
            "startup",
            "innovation",
            "competition",
            "industry",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        self.predefined_topics
            .insert("Business".to_string(), business_words);

        // Health topics
        let health_words: HashSet<String> = vec![
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
            "hospital",
            "surgery",
            "mental",
            "nutrition",
            "fitness",
            "lifestyle",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        self.predefined_topics
            .insert("Health".to_string(), health_words);

        // Education topics
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
            "classroom",
            "lecture",
            "study",
            "research",
            "degree",
            "certification",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        self.predefined_topics
            .insert("Education".to_string(), education_words);

        // Arts & Culture topics
        let arts_words: HashSet<String> = vec![
            "art",
            "culture",
            "music",
            "literature",
            "painting",
            "sculpture",
            "theater",
            "film",
            "dance",
            "creative",
            "aesthetic",
            "artistic",
            "gallery",
            "museum",
            "exhibition",
            "performance",
            "entertainment",
            "design",
            "architecture",
            "photography",
            "poetry",
            "novel",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        self.predefined_topics
            .insert("Arts".to_string(), arts_words);

        // Social topics
        let social_words: HashSet<String> = vec![
            "society",
            "community",
            "social",
            "people",
            "culture",
            "tradition",
            "family",
            "relationship",
            "friendship",
            "communication",
            "interaction",
            "collaboration",
            "cooperation",
            "conflict",
            "diversity",
            "inclusion",
            "equality",
            "justice",
            "rights",
            "responsibility",
            "citizenship",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        self.predefined_topics
            .insert("Social".to_string(), social_words);

        // Environmental topics
        let environment_words: HashSet<String> = vec![
            "environment",
            "climate",
            "ecology",
            "sustainability",
            "green",
            "renewable",
            "conservation",
            "pollution",
            "biodiversity",
            "ecosystem",
            "wildlife",
            "natural",
            "earth",
            "planet",
            "carbon",
            "emission",
            "recycling",
            "energy",
            "solar",
            "wind",
            "forest",
            "ocean",
            "atmosphere",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        self.predefined_topics
            .insert("Environment".to_string(), environment_words);
    }

    fn initialize_hierarchies(&mut self) {
        // Technology hierarchy
        self.topic_hierarchies.insert(
            "Technology".to_string(),
            vec![
                "Artificial Intelligence".to_string(),
                "Software Development".to_string(),
                "Data Science".to_string(),
                "Cybersecurity".to_string(),
            ],
        );

        // Science hierarchy
        self.topic_hierarchies.insert(
            "Science".to_string(),
            vec![
                "Life Sciences".to_string(),
                "Physical Sciences".to_string(),
                "Mathematical Sciences".to_string(),
                "Applied Sciences".to_string(),
            ],
        );
    }

    fn preprocess_text(&self, text: &str) -> Result<Vec<String>, TopicModelingError> {
        let words: Vec<String> = text
            .to_lowercase()
            .split_whitespace()
            .map(|word| word.trim_matches(|c: char| !c.is_alphabetic()))
            .filter(|word| !word.is_empty() && word.len() > 2)
            .map(String::from)
            .collect();

        if words.is_empty() {
            return Err(TopicModelingError::InvalidInput {
                message: "No valid words found in text".to_string(),
            });
        }

        Ok(words)
    }

    fn build_vocabulary(&mut self, words: &[String]) {
        self.vocabulary.clear();
        for word in words {
            self.vocabulary.insert(word.clone());
        }

        // Build co-occurrence matrix
        self.word_cooccurrence.clear();
        let window_size = 5;

        for i in 0..words.len() {
            let word = &words[i];
            for j in (i.saturating_sub(window_size))..=(i + window_size).min(words.len() - 1) {
                if i != j {
                    let coword = &words[j];
                    *self
                        .word_cooccurrence
                        .entry(word.clone())
                        .or_insert_with(HashMap::new)
                        .entry(coword.clone())
                        .or_insert(0) += 1;
                }
            }
        }
    }

    fn extract_keyword_topics(
        &self,
        words: &[String],
    ) -> Result<(Vec<f64>, Vec<Topic>), TopicModelingError> {
        let mut topic_scores = vec![0.0; self.predefined_topics.len()];
        let mut topics = Vec::new();

        for (topic_idx, (topic_name, topic_keywords)) in self.predefined_topics.iter().enumerate() {
            let mut topic_score = 0.0;
            let mut matched_keywords = Vec::new();

            for word in words {
                if topic_keywords.contains(word) {
                    topic_score += 1.0;
                    matched_keywords.push(TopicKeyword {
                        word: word.clone(),
                        weight: 1.0,
                        frequency: words.iter().filter(|&w| w == word).count(),
                        distinctiveness: 1.0,
                    });
                }
            }

            topic_score /= words.len() as f64;
            topic_scores[topic_idx] = topic_score;

            if topic_score > 0.0 {
                topics.push(Topic {
                    id: format!("topic_{}", topic_idx),
                    name: topic_name.clone(),
                    probability: topic_score,
                    keywords: matched_keywords,
                    coherence: self.calculate_topic_coherence_simple(&topic_keywords, words),
                    representative_words: topic_keywords.iter().take(5).cloned().collect(),
                    topic_type: TopicType::Thematic,
                });
            }
        }

        // Normalize topic distribution
        let sum: f64 = topic_scores.iter().sum();
        if sum > 0.0 {
            for score in &mut topic_scores {
                *score /= sum;
            }
        }

        Ok((topic_scores, topics))
    }

    fn extract_tfidf_topics(
        &self,
        words: &[String],
    ) -> Result<(Vec<f64>, Vec<Topic>), TopicModelingError> {
        // Calculate TF-IDF scores
        let mut word_frequencies = HashMap::new();
        for word in words {
            *word_frequencies.entry(word.clone()).or_insert(0) += 1;
        }

        let total_words = words.len() as f64;
        let mut tfidf_scores = HashMap::new();

        for (word, frequency) in &word_frequencies {
            let tf = *frequency as f64 / total_words;
            // Simplified IDF (assume corpus knowledge)
            let idf = if self.is_common_word(word) { 1.0 } else { 2.0 };
            tfidf_scores.insert(word.clone(), tf * idf);
        }

        // Cluster words into topics based on TF-IDF scores
        let mut sorted_words: Vec<_> = tfidf_scores.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let num_topics = self.config.num_topics.min(sorted_words.len() / 3);
        let words_per_topic = sorted_words.len() / num_topics.max(1);

        let mut topics = Vec::new();
        let mut topic_distribution = vec![0.0; num_topics];

        for topic_idx in 0..num_topics {
            let start_idx = topic_idx * words_per_topic;
            let end_idx = ((topic_idx + 1) * words_per_topic).min(sorted_words.len());

            let topic_words = &sorted_words[start_idx..end_idx];
            let topic_probability =
                topic_words.iter().map(|(_, score)| score).sum::<f64>() / sorted_words.len() as f64;

            let keywords = topic_words
                .iter()
                .map(|(word, score)| TopicKeyword {
                    word: word.clone(),
                    weight: *score,
                    frequency: word_frequencies.get(*word).unwrap_or(&0),
                    distinctiveness: *score,
                })
                .collect();

            let representative_words = topic_words
                .iter()
                .take(5)
                .map(|(word, _)| word.clone())
                .collect();

            topics.push(Topic {
                id: format!("tfidf_topic_{}", topic_idx),
                name: format!("TF-IDF Topic {}", topic_idx + 1),
                probability: topic_probability,
                keywords,
                coherence: 0.8, // Simplified coherence
                representative_words,
                topic_type: TopicType::Thematic,
            });

            topic_distribution[topic_idx] = topic_probability;
        }

        // Normalize distribution
        let sum: f64 = topic_distribution.iter().sum();
        if sum > 0.0 {
            for score in &mut topic_distribution {
                *score /= sum;
            }
        }

        Ok((topic_distribution, topics))
    }

    fn extract_latent_semantic_topics(
        &self,
        words: &[String],
    ) -> Result<(Vec<f64>, Vec<Topic>), TopicModelingError> {
        // Simplified latent semantic analysis using word co-occurrence
        let vocabulary: Vec<String> = self.vocabulary.iter().cloned().collect();
        let vocab_size = vocabulary.len();

        // Create co-occurrence matrix
        let mut cooccurrence_matrix = vec![vec![0.0; vocab_size]; vocab_size];

        for (i, word1) in vocabulary.iter().enumerate() {
            for (j, word2) in vocabulary.iter().enumerate() {
                if let Some(cooccurrences) = self.word_cooccurrence.get(word1) {
                    if let Some(&count) = cooccurrences.get(word2) {
                        cooccurrence_matrix[i][j] = count as f64;
                    }
                }
            }
        }

        // Simplified dimensionality reduction (simulate SVD)
        let num_topics = self.config.num_topics.min(vocab_size);
        let mut topics = Vec::new();
        let mut topic_distribution = vec![0.0; num_topics];

        for topic_idx in 0..num_topics {
            let topic_words =
                self.extract_topic_words_from_matrix(&cooccurrence_matrix, &vocabulary, topic_idx);
            let topic_probability = 1.0 / num_topics as f64; // Equal distribution for simplicity

            let keywords = topic_words
                .iter()
                .enumerate()
                .map(|(idx, word)| TopicKeyword {
                    word: word.clone(),
                    weight: 1.0 - (idx as f64 / topic_words.len() as f64),
                    frequency: words.iter().filter(|&w| w == word).count(),
                    distinctiveness: 0.8,
                })
                .collect();

            topics.push(Topic {
                id: format!("lsa_topic_{}", topic_idx),
                name: format!("LSA Topic {}", topic_idx + 1),
                probability: topic_probability,
                keywords,
                coherence: 0.7,
                representative_words: topic_words.into_iter().take(5).collect(),
                topic_type: TopicType::Thematic,
            });

            topic_distribution[topic_idx] = topic_probability;
        }

        Ok((topic_distribution, topics))
    }

    fn extract_cooccurrence_topics(
        &self,
        words: &[String],
    ) -> Result<(Vec<f64>, Vec<Topic>), TopicModelingError> {
        // Find strongly co-occurring word groups
        let mut topic_clusters = Vec::new();
        let mut used_words = HashSet::new();

        for word in &self.vocabulary {
            if used_words.contains(word) {
                continue;
            }

            if let Some(cooccurrences) = self.word_cooccurrence.get(word) {
                let mut cluster = vec![word.clone()];
                used_words.insert(word.clone());

                // Find strongly associated words
                for (coword, &count) in cooccurrences {
                    if count > 2 && !used_words.contains(coword) {
                        cluster.push(coword.clone());
                        used_words.insert(coword.clone());
                    }
                }

                if cluster.len() > 1 {
                    topic_clusters.push(cluster);
                }
            }
        }

        // Convert clusters to topics
        let num_topics = topic_clusters.len().min(self.config.num_topics);
        let mut topics = Vec::new();
        let mut topic_distribution = vec![0.0; num_topics];

        for (topic_idx, cluster) in topic_clusters.into_iter().take(num_topics).enumerate() {
            let cluster_score = cluster
                .iter()
                .map(|word| words.iter().filter(|&w| w == word).count())
                .sum::<usize>() as f64
                / words.len() as f64;

            let keywords = cluster
                .iter()
                .map(|word| TopicKeyword {
                    word: word.clone(),
                    weight: 1.0,
                    frequency: words.iter().filter(|&w| w == word).count(),
                    distinctiveness: 0.9,
                })
                .collect();

            topics.push(Topic {
                id: format!("cooccur_topic_{}", topic_idx),
                name: format!("Co-occurrence Topic {}", topic_idx + 1),
                probability: cluster_score,
                keywords,
                coherence: 0.8,
                representative_words: cluster.into_iter().take(5).collect(),
                topic_type: TopicType::Thematic,
            });

            topic_distribution[topic_idx] = cluster_score;
        }

        // Normalize distribution
        let sum: f64 = topic_distribution.iter().sum();
        if sum > 0.0 {
            for score in &mut topic_distribution {
                *score /= sum;
            }
        }

        Ok((topic_distribution, topics))
    }

    fn extract_hierarchical_topics(
        &self,
        words: &[String],
    ) -> Result<(Vec<f64>, Vec<Topic>), TopicModelingError> {
        // First extract base topics using keyword approach
        let (mut base_distribution, base_topics) = self.extract_keyword_topics(words)?;

        // Create hierarchical structure
        let mut hierarchical_topics = Vec::new();

        for (parent_topic, subtopics) in &self.topic_hierarchies {
            if let Some(parent_topic_obj) = base_topics.iter().find(|t| &t.name == parent_topic) {
                // Create parent topic
                let mut parent = parent_topic_obj.clone();
                parent.topic_type = TopicType::Domain;

                // Create subtopics with reduced probability
                for (subtopic_idx, subtopic_name) in subtopics.iter().enumerate() {
                    let subtopic_probability = parent.probability / subtopics.len() as f64;

                    let subtopic = Topic {
                        id: format!("hier_{}_{}", parent.id, subtopic_idx),
                        name: subtopic_name.clone(),
                        probability: subtopic_probability,
                        keywords: parent.keywords.clone(),
                        coherence: parent.coherence * 0.9,
                        representative_words: parent.representative_words.clone(),
                        topic_type: TopicType::Thematic,
                    };

                    hierarchical_topics.push(subtopic);
                }

                hierarchical_topics.push(parent);
            }
        }

        // Update distribution for hierarchical topics
        let mut hierarchical_distribution = vec![0.0; hierarchical_topics.len()];
        for (idx, topic) in hierarchical_topics.iter().enumerate() {
            hierarchical_distribution[idx] = topic.probability;
        }

        // Normalize
        let sum: f64 = hierarchical_distribution.iter().sum();
        if sum > 0.0 {
            for score in &mut hierarchical_distribution {
                *score /= sum;
            }
        }

        Ok((hierarchical_distribution, hierarchical_topics))
    }

    fn extract_dynamic_topics(
        &self,
        text: &str,
    ) -> Result<(Vec<f64>, Vec<Topic>), TopicModelingError> {
        // Split text into segments for dynamic analysis
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        if sentences.len() < 2 {
            // Fall back to static analysis
            let words = self.preprocess_text(text)?;
            return self.extract_keyword_topics(&words);
        }

        let segment_size = sentences.len().max(3) / 3; // Divide into 3 segments
        let mut segment_topics = Vec::new();

        for segment_start in (0..sentences.len()).step_by(segment_size) {
            let segment_end = (segment_start + segment_size).min(sentences.len());
            let segment_text = sentences[segment_start..segment_end].join(". ");
            let segment_words = self.preprocess_text(&segment_text)?;

            let (_, topics) = self.extract_keyword_topics(&segment_words)?;
            segment_topics.push(topics);
        }

        // Merge and evolve topics
        let mut evolved_topics = Vec::new();
        let mut topic_evolution_tracker = HashMap::new();

        for (segment_idx, topics) in segment_topics.iter().enumerate() {
            for topic in topics {
                let evolution_key = topic.name.clone();
                let entry = topic_evolution_tracker
                    .entry(evolution_key.clone())
                    .or_insert_with(Vec::new);
                entry.push((segment_idx, topic.probability));

                if segment_idx == segment_topics.len() - 1 {
                    // Create evolved topic based on all segments
                    let avg_probability =
                        entry.iter().map(|(_, prob)| prob).sum::<f64>() / entry.len() as f64;

                    let evolved_topic = Topic {
                        id: format!("dynamic_{}", evolution_key.replace(' ', "_")),
                        name: evolution_key,
                        probability: avg_probability,
                        keywords: topic.keywords.clone(),
                        coherence: topic.coherence,
                        representative_words: topic.representative_words.clone(),
                        topic_type: TopicType::Mixed,
                    };

                    evolved_topics.push(evolved_topic);
                }
            }
        }

        // Create distribution
        let mut topic_distribution = vec![0.0; evolved_topics.len()];
        for (idx, topic) in evolved_topics.iter().enumerate() {
            topic_distribution[idx] = topic.probability;
        }

        // Normalize
        let sum: f64 = topic_distribution.iter().sum();
        if sum > 0.0 {
            for score in &mut topic_distribution {
                *score /= sum;
            }
        }

        Ok((topic_distribution, evolved_topics))
    }

    // Helper methods continue...

    fn find_dominant_topic(&self, topics: &[Topic]) -> Option<Topic> {
        topics
            .iter()
            .max_by(|a, b| a.probability.partial_cmp(&b.probability).unwrap())
            .cloned()
    }

    fn calculate_topic_diversity(&self, distribution: &[f64]) -> f64 {
        // Calculate Shannon entropy for diversity
        let mut entropy = 0.0;
        for &prob in distribution {
            if prob > 0.0 {
                entropy -= prob * prob.ln();
            }
        }
        entropy / (distribution.len() as f64).ln()
    }

    fn calculate_overall_coherence(&self, topics: &[Topic]) -> f64 {
        if topics.is_empty() {
            return 0.0;
        }

        topics.iter().map(|topic| topic.coherence).sum::<f64>() / topics.len() as f64
    }

    fn calculate_topic_coherence_simple(
        &self,
        topic_words: &HashSet<String>,
        text_words: &[String],
    ) -> f64 {
        let topic_word_count = text_words
            .iter()
            .filter(|word| topic_words.contains(*word))
            .count();

        if topic_word_count == 0 {
            0.0
        } else {
            topic_word_count as f64 / text_words.len() as f64
        }
    }

    fn is_common_word(&self, word: &str) -> bool {
        let common_words = [
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        ];
        common_words.contains(&word)
    }

    fn extract_topic_words_from_matrix(
        &self,
        matrix: &[Vec<f64>],
        vocabulary: &[String],
        topic_idx: usize,
    ) -> Vec<String> {
        // Simplified topic extraction from co-occurrence matrix
        let mut word_scores: Vec<(String, f64)> = vocabulary
            .iter()
            .enumerate()
            .map(|(word_idx, word)| {
                let score = matrix[word_idx].iter().sum::<f64>() / vocabulary.len() as f64;
                (word.clone(), score)
            })
            .collect();

        word_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let words_per_topic = vocabulary.len() / self.config.num_topics.max(1);
        let start_idx = topic_idx * words_per_topic;
        let end_idx = ((topic_idx + 1) * words_per_topic).min(word_scores.len());

        word_scores[start_idx..end_idx]
            .iter()
            .map(|(word, _)| word.clone())
            .collect()
    }

    fn analyze_topic_evolution(
        &self,
        text: &str,
        topics: &[Topic],
    ) -> Result<TopicEvolution, TopicModelingError> {
        // Split text into segments and analyze topic changes
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        let segment_size = sentences.len().max(3) / 3;
        let mut segment_distributions = Vec::new();

        for segment_start in (0..sentences.len()).step_by(segment_size) {
            let segment_end = (segment_start + segment_size).min(sentences.len());
            let segment_text = sentences[segment_start..segment_end].join(". ");

            // Simplified: assign topic probabilities based on keyword matching
            let mut segment_dist = vec![0.0; topics.len()];
            for (topic_idx, topic) in topics.iter().enumerate() {
                let topic_score = topic
                    .keywords
                    .iter()
                    .filter(|keyword| segment_text.to_lowercase().contains(&keyword.word))
                    .count() as f64
                    / topic.keywords.len().max(1) as f64;
                segment_dist[topic_idx] = topic_score;
            }

            // Normalize
            let sum: f64 = segment_dist.iter().sum();
            if sum > 0.0 {
                for score in &mut segment_dist {
                    *score /= sum;
                }
            }

            segment_distributions.push(segment_dist);
        }

        // Analyze transitions
        let mut transitions = Vec::new();
        for i in 1..segment_distributions.len() {
            let mut topic_changes = HashMap::new();
            let mut transition_strength = 0.0;

            for (topic_idx, topic) in topics.iter().enumerate() {
                let change =
                    segment_distributions[i][topic_idx] - segment_distributions[i - 1][topic_idx];
                topic_changes.insert(topic.id.clone(), change);
                transition_strength += change.abs();
            }

            let transition_type = if transition_strength > 0.5 {
                TransitionType::Sharp
            } else if transition_strength > 0.2 {
                TransitionType::Smooth
            } else {
                TransitionType::Smooth
            };

            transitions.push(TopicTransition {
                from_segment: i - 1,
                to_segment: i,
                topic_changes,
                transition_strength,
                transition_type,
            });
        }

        // Calculate stability scores
        let stability_scores = topics
            .iter()
            .enumerate()
            .map(|(topic_idx, _)| {
                let values: Vec<f64> = segment_distributions
                    .iter()
                    .map(|dist| dist[topic_idx])
                    .collect();
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance =
                    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
                1.0 - variance.sqrt() // Higher stability = lower variance
            })
            .collect();

        // Identify evolution patterns
        let evolution_patterns = self.identify_evolution_patterns(&segment_distributions, topics);

        Ok(TopicEvolution {
            segment_distributions,
            transitions,
            evolution_patterns,
            stability_scores,
        })
    }

    fn build_topic_hierarchy(
        &self,
        topics: &[Topic],
    ) -> Result<TopicHierarchy, TopicModelingError> {
        // Group topics into clusters based on similarity
        let mut clusters = Vec::new();
        let mut relationships = Vec::new();
        let mut levels = Vec::new();

        // Simple clustering based on keyword overlap
        let mut remaining_topics: Vec<_> = topics.iter().collect();
        let mut cluster_id = 0;

        while !remaining_topics.is_empty() {
            let seed_topic = remaining_topics.remove(0);
            let mut cluster_topics = vec![seed_topic.id.clone()];

            // Find similar topics
            let mut to_remove = Vec::new();
            for (idx, topic) in remaining_topics.iter().enumerate() {
                let similarity = self.compute_individual_topic_similarity(seed_topic, topic);
                if similarity > 0.5 {
                    cluster_topics.push(topic.id.clone());
                    to_remove.push(idx);

                    relationships.push(TopicRelationship {
                        parent_topic: seed_topic.id.clone(),
                        child_topic: topic.id.clone(),
                        relationship_type: RelationshipType::Related,
                        strength: similarity,
                    });
                }
            }

            // Remove clustered topics from remaining
            for &idx in to_remove.iter().rev() {
                remaining_topics.remove(idx);
            }

            if cluster_topics.len() > 1 {
                clusters.push(TopicCluster {
                    cluster_id: format!("cluster_{}", cluster_id),
                    topics: cluster_topics,
                    centroid: vec![0.5; topics.len()], // Simplified centroid
                    coherence: seed_topic.coherence,
                    diversity: 0.7,
                });
                cluster_id += 1;
            }
        }

        // Create levels (simplified: 2-level hierarchy)
        let root_topics = topics
            .iter()
            .filter(|topic| topic.probability > 0.1)
            .cloned()
            .collect();

        levels.push(TopicLevel {
            level: 0,
            topics: root_topics.iter().map(|t| t.id.clone()).collect(),
            coherence: root_topics.iter().map(|t| t.coherence).sum::<f64>()
                / root_topics.len().max(1) as f64,
        });

        Ok(TopicHierarchy {
            root_topics,
            relationships,
            levels,
            clusters,
        })
    }

    fn calculate_quality_metrics(
        &self,
        topics: &[Topic],
        distribution: &[f64],
    ) -> TopicQualityMetrics {
        let coherence =
            topics.iter().map(|t| t.coherence).sum::<f64>() / topics.len().max(1) as f64;
        let diversity = self.calculate_topic_diversity(distribution);

        TopicQualityMetrics {
            perplexity: 10.0, // Simplified
            coherence,
            diversity,
            interpretability: 0.8, // Simplified
            stability: 0.7,        // Simplified
        }
    }

    fn compute_individual_topic_similarity(&self, topic1: &Topic, topic2: &Topic) -> f64 {
        // Keyword overlap similarity
        let keywords1: HashSet<_> = topic1.keywords.iter().map(|k| &k.word).collect();
        let keywords2: HashSet<_> = topic2.keywords.iter().map(|k| &k.word).collect();

        let intersection = keywords1.intersection(&keywords2).count();
        let union = keywords1.union(&keywords2).count();

        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }

    fn compute_keyword_overlap_similarity(&self, topics1: &[Topic], topics2: &[Topic]) -> f64 {
        let mut total_similarity = 0.0;
        let mut comparisons = 0;

        for topic1 in topics1 {
            for topic2 in topics2 {
                total_similarity += self.compute_individual_topic_similarity(topic1, topic2);
                comparisons += 1;
            }
        }

        if comparisons > 0 {
            total_similarity / comparisons as f64
        } else {
            0.0
        }
    }

    fn compute_evolution_similarity(&self, evo1: &TopicEvolution, evo2: &TopicEvolution) -> f64 {
        // Compare evolution patterns
        let stability_sim = if evo1.stability_scores.len() == evo2.stability_scores.len() {
            self.cosine_similarity(&evo1.stability_scores, &evo2.stability_scores)
        } else {
            0.5
        };

        let pattern_sim = if evo1.evolution_patterns.len() == evo2.evolution_patterns.len() {
            0.7 // Simplified pattern similarity
        } else {
            0.5
        };

        (stability_sim + pattern_sim) / 2.0
    }

    fn compute_hierarchical_similarity(
        &self,
        hier1: &TopicHierarchy,
        hier2: &TopicHierarchy,
    ) -> f64 {
        // Compare hierarchical structures
        let level_sim = if hier1.levels.len() == hier2.levels.len() {
            0.8
        } else {
            0.5
        };

        let cluster_sim = if hier1.clusters.len() == hier2.clusters.len() {
            0.7
        } else {
            0.4
        };

        (level_sim + cluster_sim) / 2.0
    }

    fn identify_evolution_patterns(
        &self,
        segment_distributions: &[Vec<f64>],
        topics: &[Topic],
    ) -> Vec<EvolutionPattern> {
        let mut patterns = Vec::new();

        for (topic_idx, topic) in topics.iter().enumerate() {
            let topic_progression: Vec<f64> = segment_distributions
                .iter()
                .map(|dist| dist[topic_idx])
                .collect();

            if topic_progression.len() < 2 {
                continue;
            }

            // Detect patterns
            let start_value = topic_progression[0];
            let end_value = *topic_progression.last().unwrap();
            let max_value = topic_progression.iter().fold(0.0f64, |acc, &x| acc.max(x));

            let pattern_type = if end_value > start_value + 0.2 {
                PatternType::Emergence
            } else if start_value > end_value + 0.2 {
                PatternType::Decay
            } else if max_value > start_value + 0.3 && end_value < max_value - 0.1 {
                PatternType::Oscillation
            } else if (end_value - start_value).abs() < 0.1 {
                PatternType::Persistence
            } else {
                PatternType::Intensification
            };

            patterns.push(EvolutionPattern {
                pattern_type,
                affected_topics: vec![topic.id.clone()],
                strength: (end_value - start_value).abs(),
                segments: (0, segment_distributions.len() - 1),
            });
        }

        patterns
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
}

impl Default for TopicModeler {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for simple topic modeling

/// Extract basic topics from text using keyword clustering
pub fn extract_basic_topics(text: &str) -> Result<TopicModelingResult, TopicModelingError> {
    let mut modeler = TopicModeler::new();
    modeler.extract_topics(text)
}

/// Extract topics using TF-IDF approach
pub fn extract_tfidf_topics(
    text: &str,
    num_topics: usize,
) -> Result<TopicModelingResult, TopicModelingError> {
    let config = TopicModelingConfig::new()
        .with_approach(TopicModelingApproach::TfIdf)
        .with_num_topics(num_topics);
    let mut modeler = TopicModeler::with_config(config);
    modeler.extract_topics(text)
}

/// Compare topic similarity between two texts
pub fn compare_topic_similarity_simple(
    text1: &str,
    text2: &str,
) -> Result<f64, TopicModelingError> {
    let mut modeler = TopicModeler::new();
    let topics1 = modeler.extract_topics(text1)?;
    let topics2 = modeler.extract_topics(text2)?;
    modeler.compute_topic_similarity(&topics1, &topics2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_modeler_creation() {
        let modeler = TopicModeler::new();
        assert!(!modeler.predefined_topics.is_empty());
        assert_eq!(modeler.config.num_topics, 8);
    }

    #[test]
    fn test_keyword_topic_extraction() -> Result<(), TopicModelingError> {
        let mut modeler = TopicModeler::new();
        let text = "Computer software development with artificial intelligence and machine learning algorithms";

        let result = modeler.extract_topics(text)?;

        assert!(!result.topics.is_empty());
        assert!(!result.topic_distribution.is_empty());
        assert!(result.diversity_score >= 0.0 && result.diversity_score <= 1.0);

        // Should find technology-related topics
        let has_tech_topic = result.topics.iter().any(|topic| {
            topic.name.contains("Technology")
                || topic
                    .keywords
                    .iter()
                    .any(|kw| kw.word == "computer" || kw.word == "software")
        });
        assert!(has_tech_topic);

        Ok(())
    }

    #[test]
    fn test_tfidf_topic_extraction() -> Result<(), TopicModelingError> {
        let config = TopicModelingConfig::new().with_approach(TopicModelingApproach::TfIdf);
        let mut modeler = TopicModeler::with_config(config);

        let text = "Business market analysis shows financial growth in technology sector with innovation driving investment";

        let result = modeler.extract_topics(text)?;

        assert!(!result.topics.is_empty());
        assert_eq!(result.metadata.approach_used, TopicModelingApproach::TfIdf);

        Ok(())
    }

    #[test]
    fn test_dynamic_topic_modeling() -> Result<(), TopicModelingError> {
        let config = TopicModelingConfig::new()
            .with_approach(TopicModelingApproach::Dynamic)
            .with_dynamic_modeling(true);
        let mut modeler = TopicModeler::with_config(config);

        let text = "Technology advances rapidly. Innovation drives market changes. Business adapts to new trends. Future looks promising for development.";

        let result = modeler.extract_topics(text)?;

        assert!(!result.topics.is_empty());
        assert!(result.topic_evolution.is_some());

        if let Some(evolution) = &result.topic_evolution {
            assert!(!evolution.segment_distributions.is_empty());
        }

        Ok(())
    }

    #[test]
    fn test_topic_similarity() -> Result<(), TopicModelingError> {
        let mut modeler = TopicModeler::new();

        let text1 = "Computer technology software development programming artificial intelligence";
        let text2 = "Technology innovation software engineering machine learning algorithms";
        let text3 = "Medical healthcare treatment patient doctor diagnosis therapy";

        let result1 = modeler.extract_topics(text1)?;
        let result2 = modeler.extract_topics(text2)?;
        let result3 = modeler.extract_topics(text3)?;

        let similarity_tech = modeler.compute_topic_similarity(&result1, &result2)?;
        let similarity_mixed = modeler.compute_topic_similarity(&result1, &result3)?;

        assert!(similarity_tech > similarity_mixed);
        assert!(similarity_tech > 0.3); // Should have decent similarity

        Ok(())
    }

    #[test]
    fn test_hierarchical_topic_modeling() -> Result<(), TopicModelingError> {
        let config = TopicModelingConfig::new()
            .with_approach(TopicModelingApproach::Hierarchical)
            .with_hierarchical_analysis(true);
        let mut modeler = TopicModeler::with_config(config);

        let text = "Science research study analysis with scientific method and laboratory testing";

        let result = modeler.extract_topics(text)?;

        assert!(!result.topics.is_empty());
        assert!(result.hierarchical_structure.is_some());

        Ok(())
    }

    #[test]
    fn test_multiple_topic_comparison() -> Result<(), TopicModelingError> {
        let mut modeler = TopicModeler::new();

        let texts = vec![
            "Technology computer software",
            "Technology innovation development",
            "Medical healthcare treatment",
            "Business market finance",
        ];

        let similarity_matrix = modeler.compare_multiple_topics(&texts)?;

        assert_eq!(similarity_matrix.len(), 4);
        assert_eq!(similarity_matrix[0].len(), 4);

        // Diagonal should be 1.0
        for i in 0..4 {
            assert_eq!(similarity_matrix[i][i], 1.0);
        }

        // Technology texts should be more similar to each other
        assert!(similarity_matrix[0][1] > similarity_matrix[0][2]);

        Ok(())
    }

    #[test]
    fn test_topic_quality_metrics() -> Result<(), TopicModelingError> {
        let mut modeler = TopicModeler::new();
        let text =
            "Research study analysis experiment scientific method laboratory testing validation";

        let result = modeler.extract_topics(text)?;

        let quality = &result.metadata.quality_metrics;
        assert!(quality.coherence >= 0.0 && quality.coherence <= 1.0);
        assert!(quality.diversity >= 0.0 && quality.diversity <= 1.0);
        assert!(quality.interpretability >= 0.0 && quality.interpretability <= 1.0);

        Ok(())
    }

    #[test]
    fn test_convenience_functions() -> Result<(), TopicModelingError> {
        let text = "Technology software development programming algorithms";

        let basic_result = extract_basic_topics(text)?;
        assert!(!basic_result.topics.is_empty());

        let tfidf_result = extract_tfidf_topics(text, 5)?;
        assert!(!tfidf_result.topics.is_empty());
        assert_eq!(
            tfidf_result.metadata.approach_used,
            TopicModelingApproach::TfIdf
        );

        let similarity = compare_topic_similarity_simple(
            "technology software programming",
            "computer development algorithms",
        )?;
        assert!(similarity >= 0.0 && similarity <= 1.0);

        Ok(())
    }

    #[test]
    fn test_error_handling() {
        let mut modeler = TopicModeler::new();

        // Test empty text
        let result = modeler.extract_topics("");
        assert!(matches!(
            result,
            Err(TopicModelingError::InvalidInput { .. })
        ));

        // Test text with insufficient vocabulary
        let result = modeler.extract_topics("a b c");
        assert!(matches!(
            result,
            Err(TopicModelingError::InsufficientVocabulary { .. })
        ));
    }

    #[test]
    fn test_topic_evolution_analysis() -> Result<(), TopicModelingError> {
        let config = TopicModelingConfig::new().with_dynamic_modeling(true);
        let mut modeler = TopicModeler::with_config(config);

        let text = "Technology starts simple. Then innovation accelerates progress. Finally systems become complex. The future holds more advancement.";

        let result = modeler.extract_topics(text)?;

        if let Some(evolution) = &result.topic_evolution {
            assert!(!evolution.segment_distributions.is_empty());
            assert!(!evolution.stability_scores.is_empty());
            assert!(!evolution.evolution_patterns.is_empty());
        }

        Ok(())
    }
}
