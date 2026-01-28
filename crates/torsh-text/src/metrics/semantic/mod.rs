//! Comprehensive semantic analysis and similarity computation framework
//!
//! This module provides a unified interface for semantic analysis, combining multiple specialized
//! approaches including similarity algorithms, feature extraction, sentiment analysis, topic modeling,
//! domain classification, syntactic analysis, and comprehensive metrics collection.
//!
//! ## Features
//!
//! - **Similarity Algorithms**: Advanced similarity computation with multiple algorithms
//! - **Feature Extraction**: Comprehensive semantic feature extraction strategies
//! - **Sentiment Analysis**: Emotion-aware sentiment analysis with context awareness
//! - **Topic Modeling**: Advanced topic modeling with multiple approaches
//! - **Domain Analysis**: Semantic domain classification and analysis
//! - **Syntactic Analysis**: Syntactic pattern analysis and comparison
//! - **Metrics Collection**: Comprehensive statistical analysis and quality assessment
//! - **Unified Interface**: Single point of entry for all semantic analysis functionality
//!
//! ## Quick Start
//!
//! ```rust
//! use torsh_text::metrics::semantic::{SemanticAnalyzer, SemanticAnalysisConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create analyzer with default configuration
//! let mut analyzer = SemanticAnalyzer::default()?;
//!
//! // Analyze similarity between two texts
//! let text1 = "The quick brown fox jumps over the lazy dog.";
//! let text2 = "A fast brown fox leaps across the sleeping dog.";
//!
//! let result = analyzer.analyze_comprehensive(text1, text2).await?;
//! println!("Similarity: {:.2}", result.overall_similarity);
//! # Ok(())
//! # }
//! ```
//!
//! ## Advanced Usage
//!
//! ```rust
//! use torsh_text::metrics::semantic::{
//!     SemanticAnalyzer, SemanticAnalysisConfig,
//!     similarity_algorithms::SimilarityAlgorithm,
//!     feature_extraction::FeatureExtractionStrategy,
//!     sentiment_analysis::SentimentAnalysisMode,
//! };
//!
//! # async fn advanced_example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create custom configuration
//! let config = SemanticAnalysisConfig::builder()
//!     .enable_similarity_analysis(true)
//!     .enable_sentiment_analysis(true)
//!     .enable_topic_modeling(true)
//!     .enable_domain_analysis(true)
//!     .enable_syntactic_analysis(true)
//!     .enable_metrics_collection(true)
//!     .similarity_algorithm(SimilarityAlgorithm::Hierarchical)
//!     .feature_extraction_strategy(FeatureExtractionStrategy::Comprehensive)
//!     .sentiment_analysis_mode(SentimentAnalysisMode::ContextAware)
//!     .enable_caching(true)
//!     .enable_parallel_processing(true)
//!     .build()?;
//!
//! let mut analyzer = SemanticAnalyzer::new(config)?;
//!
//! // Batch analysis
//! let texts = vec![
//!     "Technical documentation about software engineering.",
//!     "Academic research on machine learning algorithms.",
//!     "Business strategy for market expansion.",
//! ];
//!
//! let results = analyzer.analyze_batch(&texts).await?;
//! for (i, result) in results.iter().enumerate() {
//!     println!("Analysis {}: Quality={:.2}", i, result.quality_score);
//! }
//! # Ok(())
//! # }
//! ```

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rng, Random};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use thiserror::Error;
use tokio::task;

// Re-export all specialized modules
pub mod domain_analysis;
pub mod feature_extraction;
pub mod semantic_metrics;
pub mod sentiment_analysis;
pub mod similarity_algorithms;
pub mod syntactic_similarity;
pub mod topic_modeling;

// Re-export key types for convenience
pub use domain_analysis::{
    DomainAnalysisError, DomainAnalyzer, DomainClassification, SemanticDomain,
};
pub use feature_extraction::{
    FeatureExtractionError, FeatureExtractionStrategy, FeatureExtractor, SemanticFeatureVector,
};
pub use semantic_metrics::{
    MetricsSummary, SemanticMetricsAnalyzer, SemanticMetricsError, SemanticMetricsResult,
};
pub use sentiment_analysis::{
    EmotionScores, SentimentAnalysisError, SentimentAnalysisMode, SentimentAnalyzer,
    SentimentScores,
};
pub use similarity_algorithms::{
    SimilarityAlgorithm, SimilarityAlgorithmEngine, SimilarityAlgorithmError, SimilarityResult,
};
pub use syntactic_similarity::{
    SyntacticApproach, SyntacticSimilarityAnalyzer, SyntacticSimilarityError,
    SyntacticSimilarityResult,
};
pub use topic_modeling::{
    TopicModeler, TopicModelingApproach, TopicModelingError, TopicModelingResult,
};

/// Errors that can occur during semantic analysis
#[derive(Error, Debug)]
pub enum SemanticAnalysisError {
    #[error("Invalid configuration: {message}")]
    InvalidConfiguration { message: String },

    #[error("Analysis failed: {reason}")]
    AnalysisFailed { reason: String },

    #[error("Similarity algorithm error: {source}")]
    SimilarityError {
        #[from]
        source: SimilarityAlgorithmError,
    },

    #[error("Feature extraction error: {source}")]
    FeatureExtractionError {
        #[from]
        source: FeatureExtractionError,
    },

    #[error("Sentiment analysis error: {source}")]
    SentimentError {
        #[from]
        source: SentimentAnalysisError,
    },

    #[error("Topic modeling error: {source}")]
    TopicModelingError {
        #[from]
        source: TopicModelingError,
    },

    #[error("Domain analysis error: {source}")]
    DomainAnalysisError {
        #[from]
        source: DomainAnalysisError,
    },

    #[error("Syntactic analysis error: {source}")]
    SyntacticError {
        #[from]
        source: SyntacticSimilarityError,
    },

    #[error("Metrics analysis error: {source}")]
    MetricsError {
        #[from]
        source: SemanticMetricsError,
    },

    #[error("Parallel processing error: {reason}")]
    ParallelProcessingError { reason: String },

    #[error("Cache error: {reason}")]
    CacheError { reason: String },

    #[error("Timeout error: operation took longer than {timeout_seconds} seconds")]
    TimeoutError { timeout_seconds: u64 },
}

/// Comprehensive semantic analysis result
#[derive(Debug, Clone)]
pub struct SemanticAnalysisResult {
    /// Overall similarity score (0.0-1.0)
    pub overall_similarity: f64,
    /// Overall quality score (0.0-1.0)
    pub quality_score: f64,
    /// Overall confidence score (0.0-1.0)
    pub confidence_score: f64,
    /// Detailed similarity analysis results
    pub similarity_analysis: Option<SimilarityResult>,
    /// Feature extraction results
    pub feature_analysis: Option<(SemanticFeatureVector, SemanticFeatureVector)>,
    /// Sentiment analysis results
    pub sentiment_analysis: Option<(SentimentScores, SentimentScores)>,
    /// Topic modeling results
    pub topic_analysis: Option<(TopicModelingResult, TopicModelingResult)>,
    /// Domain classification results
    pub domain_analysis: Option<(DomainClassification, DomainClassification)>,
    /// Syntactic similarity results
    pub syntactic_analysis: Option<SyntacticSimilarityResult>,
    /// Comprehensive metrics analysis
    pub metrics_analysis: Option<SemanticMetricsResult>,
    /// Analysis metadata
    pub metadata: AnalysisMetadata,
}

/// Analysis metadata
#[derive(Debug, Clone)]
pub struct AnalysisMetadata {
    /// Analysis timestamp
    pub timestamp: SystemTime,
    /// Total processing time
    pub processing_time: Duration,
    /// Configuration summary
    pub config_used: String,
    /// Components that were enabled
    pub enabled_components: Vec<String>,
    /// Text characteristics
    pub text1_characteristics: TextCharacteristics,
    pub text2_characteristics: TextCharacteristics,
    /// Performance metrics
    pub performance_metrics: PerformanceInfo,
    /// Cache hit/miss information
    pub cache_info: CacheInfo,
}

/// Text characteristics for analysis
#[derive(Debug, Clone)]
pub struct TextCharacteristics {
    pub length: usize,
    pub word_count: usize,
    pub sentence_count: usize,
    pub paragraph_count: usize,
    pub language_detected: Option<String>,
    pub complexity_score: f64,
    pub readability_score: f64,
}

/// Performance information
#[derive(Debug, Clone)]
pub struct PerformanceInfo {
    pub total_time: Duration,
    pub similarity_time: Duration,
    pub feature_extraction_time: Duration,
    pub sentiment_analysis_time: Duration,
    pub topic_modeling_time: Duration,
    pub domain_analysis_time: Duration,
    pub syntactic_analysis_time: Duration,
    pub metrics_analysis_time: Duration,
    pub parallel_efficiency: f64,
}

/// Cache information
#[derive(Debug, Clone)]
pub struct CacheInfo {
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub cache_hit_rate: f64,
    pub cache_size: usize,
}

/// Batch analysis result
#[derive(Debug, Clone)]
pub struct BatchAnalysisResult {
    /// Individual analysis results
    pub results: Vec<SemanticAnalysisResult>,
    /// Batch statistics
    pub batch_statistics: BatchStatistics,
    /// Performance summary
    pub performance_summary: BatchPerformance,
}

/// Batch statistics
#[derive(Debug, Clone)]
pub struct BatchStatistics {
    pub total_items: usize,
    pub successful_analyses: usize,
    pub failed_analyses: usize,
    pub average_similarity: f64,
    pub similarity_distribution: HashMap<String, usize>, // Ranges like "0.0-0.1", "0.1-0.2", etc.
    pub quality_distribution: HashMap<String, usize>,
}

/// Batch performance metrics
#[derive(Debug, Clone)]
pub struct BatchPerformance {
    pub total_time: Duration,
    pub average_time_per_item: Duration,
    pub items_per_second: f64,
    pub parallel_speedup: f64,
    pub memory_usage: usize,
}

/// Configuration for semantic analysis
#[derive(Debug, Clone)]
pub struct SemanticAnalysisConfig {
    /// Enable similarity algorithm analysis
    pub enable_similarity_analysis: bool,
    /// Enable feature extraction
    pub enable_feature_extraction: bool,
    /// Enable sentiment analysis
    pub enable_sentiment_analysis: bool,
    /// Enable topic modeling
    pub enable_topic_modeling: bool,
    /// Enable domain analysis
    pub enable_domain_analysis: bool,
    /// Enable syntactic analysis
    pub enable_syntactic_analysis: bool,
    /// Enable comprehensive metrics collection
    pub enable_metrics_collection: bool,
    /// Similarity algorithm to use
    pub similarity_algorithm: SimilarityAlgorithm,
    /// Feature extraction strategy
    pub feature_extraction_strategy: feature_extraction::FeatureExtractionStrategy,
    /// Sentiment analysis mode
    pub sentiment_analysis_mode: sentiment_analysis::SentimentAnalysisMode,
    /// Topic modeling approach
    pub topic_modeling_approach: topic_modeling::TopicModelingApproach,
    /// Domain analysis approach
    pub domain_analysis_approach: domain_analysis::DomainApproach,
    /// Syntactic analysis approach
    pub syntactic_analysis_approach: syntactic_similarity::SyntacticApproach,
    /// Enable result caching
    pub enable_caching: bool,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// Enable parallel processing
    pub enable_parallel_processing: bool,
    /// Maximum number of parallel threads
    pub max_parallel_threads: usize,
    /// Analysis timeout in seconds
    pub timeout_seconds: u64,
    /// Minimum quality threshold for results
    pub quality_threshold: f64,
}

impl Default for SemanticAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_similarity_analysis: true,
            enable_feature_extraction: true,
            enable_sentiment_analysis: true,
            enable_topic_modeling: true,
            enable_domain_analysis: true,
            enable_syntactic_analysis: true,
            enable_metrics_collection: true,
            similarity_algorithm: SimilarityAlgorithm::Hierarchical,
            feature_extraction_strategy:
                feature_extraction::FeatureExtractionStrategy::Comprehensive,
            sentiment_analysis_mode: sentiment_analysis::SentimentAnalysisMode::ContextAware,
            topic_modeling_approach: topic_modeling::TopicModelingApproach::Dynamic,
            domain_analysis_approach: domain_analysis::DomainApproach::MultiModal,
            syntactic_analysis_approach: syntactic_similarity::SyntacticApproach::Comprehensive,
            enable_caching: true,
            cache_size_limit: 1000,
            enable_parallel_processing: true,
            max_parallel_threads: num_cpus::get(),
            timeout_seconds: 300, // 5 minutes
            quality_threshold: 0.0,
        }
    }
}

impl SemanticAnalysisConfig {
    /// Create a new configuration builder
    pub fn builder() -> SemanticAnalysisConfigBuilder {
        SemanticAnalysisConfigBuilder::new()
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), SemanticAnalysisError> {
        if self.cache_size_limit == 0 && self.enable_caching {
            return Err(SemanticAnalysisError::InvalidConfiguration {
                message: "Cache size limit must be greater than 0 when caching is enabled"
                    .to_string(),
            });
        }

        if self.max_parallel_threads == 0 && self.enable_parallel_processing {
            return Err(SemanticAnalysisError::InvalidConfiguration {
                message: "Max parallel threads must be greater than 0 when parallel processing is enabled".to_string(),
            });
        }

        if self.timeout_seconds == 0 {
            return Err(SemanticAnalysisError::InvalidConfiguration {
                message: "Timeout must be greater than 0 seconds".to_string(),
            });
        }

        if self.quality_threshold < 0.0 || self.quality_threshold > 1.0 {
            return Err(SemanticAnalysisError::InvalidConfiguration {
                message: "Quality threshold must be between 0.0 and 1.0".to_string(),
            });
        }

        Ok(())
    }
}

/// Builder for semantic analysis configuration
pub struct SemanticAnalysisConfigBuilder {
    config: SemanticAnalysisConfig,
}

impl SemanticAnalysisConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: SemanticAnalysisConfig::default(),
        }
    }

    pub fn enable_similarity_analysis(mut self, enable: bool) -> Self {
        self.config.enable_similarity_analysis = enable;
        self
    }

    pub fn enable_feature_extraction(mut self, enable: bool) -> Self {
        self.config.enable_feature_extraction = enable;
        self
    }

    pub fn enable_sentiment_analysis(mut self, enable: bool) -> Self {
        self.config.enable_sentiment_analysis = enable;
        self
    }

    pub fn enable_topic_modeling(mut self, enable: bool) -> Self {
        self.config.enable_topic_modeling = enable;
        self
    }

    pub fn enable_domain_analysis(mut self, enable: bool) -> Self {
        self.config.enable_domain_analysis = enable;
        self
    }

    pub fn enable_syntactic_analysis(mut self, enable: bool) -> Self {
        self.config.enable_syntactic_analysis = enable;
        self
    }

    pub fn enable_metrics_collection(mut self, enable: bool) -> Self {
        self.config.enable_metrics_collection = enable;
        self
    }

    pub fn similarity_algorithm(mut self, algorithm: SimilarityAlgorithm) -> Self {
        self.config.similarity_algorithm = algorithm;
        self
    }

    pub fn feature_extraction_strategy(
        mut self,
        strategy: feature_extraction::FeatureExtractionStrategy,
    ) -> Self {
        self.config.feature_extraction_strategy = strategy;
        self
    }

    pub fn sentiment_analysis_mode(
        mut self,
        mode: sentiment_analysis::SentimentAnalysisMode,
    ) -> Self {
        self.config.sentiment_analysis_mode = mode;
        self
    }

    pub fn topic_modeling_approach(
        mut self,
        approach: topic_modeling::TopicModelingApproach,
    ) -> Self {
        self.config.topic_modeling_approach = approach;
        self
    }

    pub fn domain_analysis_approach(mut self, approach: domain_analysis::DomainApproach) -> Self {
        self.config.domain_analysis_approach = approach;
        self
    }

    pub fn syntactic_analysis_approach(
        mut self,
        approach: syntactic_similarity::SyntacticApproach,
    ) -> Self {
        self.config.syntactic_analysis_approach = approach;
        self
    }

    pub fn enable_caching(mut self, enable: bool) -> Self {
        self.config.enable_caching = enable;
        self
    }

    pub fn cache_size_limit(mut self, limit: usize) -> Self {
        self.config.cache_size_limit = limit.max(1);
        self
    }

    pub fn enable_parallel_processing(mut self, enable: bool) -> Self {
        self.config.enable_parallel_processing = enable;
        self
    }

    pub fn max_parallel_threads(mut self, max: usize) -> Self {
        self.config.max_parallel_threads = max.max(1);
        self
    }

    pub fn timeout_seconds(mut self, timeout: u64) -> Self {
        self.config.timeout_seconds = timeout.max(1);
        self
    }

    pub fn quality_threshold(mut self, threshold: f64) -> Self {
        self.config.quality_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn build(self) -> Result<SemanticAnalysisConfig, SemanticAnalysisError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

/// Cache entry for analysis results
#[derive(Debug, Clone)]
struct CacheEntry {
    result: SemanticAnalysisResult,
    timestamp: SystemTime,
    access_count: usize,
}

/// Comprehensive semantic analyzer with unified interface
pub struct SemanticAnalyzer {
    config: SemanticAnalysisConfig,
    similarity_engine: Arc<Mutex<SimilarityAlgorithmEngine>>,
    feature_extractor: Arc<Mutex<FeatureExtractor>>,
    sentiment_analyzer: Arc<Mutex<SentimentAnalyzer>>,
    topic_modeler: Arc<Mutex<TopicModeler>>,
    domain_analyzer: Arc<Mutex<DomainAnalyzer>>,
    syntactic_analyzer: Arc<Mutex<SyntacticSimilarityAnalyzer>>,
    metrics_analyzer: Arc<Mutex<SemanticMetricsAnalyzer>>,
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    cache_stats: Arc<Mutex<CacheInfo>>,
}

impl SemanticAnalyzer {
    /// Create a new semantic analyzer with the given configuration
    pub fn new(config: SemanticAnalysisConfig) -> Result<Self, SemanticAnalysisError> {
        config.validate()?;

        // Initialize all sub-analyzers with appropriate configurations
        let similarity_config = similarity_algorithms::SimilarityAlgorithmConfig::builder()
            .default_algorithm(config.similarity_algorithm)
            .enable_caching(config.enable_caching)
            .build()
            .map_err(|e| SemanticAnalysisError::InvalidConfiguration {
                message: e.to_string(),
            })?;

        let feature_config = feature_extraction::FeatureExtractionConfig::builder()
            .strategy(config.feature_extraction_strategy)
            .build()
            .map_err(|e| SemanticAnalysisError::InvalidConfiguration {
                message: e.to_string(),
            })?;

        let sentiment_config = sentiment_analysis::SentimentAnalysisConfig::builder()
            .mode(config.sentiment_analysis_mode)
            .build()
            .map_err(|e| SemanticAnalysisError::InvalidConfiguration {
                message: e.to_string(),
            })?;

        let topic_config = topic_modeling::TopicModelingConfig::builder()
            .approach(config.topic_modeling_approach)
            .build()
            .map_err(|e| SemanticAnalysisError::InvalidConfiguration {
                message: e.to_string(),
            })?;

        let domain_config = domain_analysis::DomainAnalysisConfig::builder()
            .approach(config.domain_analysis_approach)
            .build()
            .map_err(|e| SemanticAnalysisError::InvalidConfiguration {
                message: e.to_string(),
            })?;

        let syntactic_config = syntactic_similarity::SyntacticSimilarityConfig::builder()
            .approach(config.syntactic_analysis_approach)
            .build()
            .map_err(|e| SemanticAnalysisError::InvalidConfiguration {
                message: e.to_string(),
            })?;

        let metrics_config = semantic_metrics::SemanticMetricsConfig::builder()
            .enable_statistical_analysis(true)
            .enable_distribution_analysis(true)
            .enable_correlation_analysis(true)
            .build()
            .map_err(|e| SemanticAnalysisError::InvalidConfiguration {
                message: e.to_string(),
            })?;

        Ok(Self {
            config,
            similarity_engine: Arc::new(Mutex::new(SimilarityAlgorithmEngine::new(
                similarity_config,
            )?)),
            feature_extractor: Arc::new(Mutex::new(FeatureExtractor::new(feature_config)?)),
            sentiment_analyzer: Arc::new(Mutex::new(SentimentAnalyzer::new(sentiment_config)?)),
            topic_modeler: Arc::new(Mutex::new(TopicModeler::new(topic_config)?)),
            domain_analyzer: Arc::new(Mutex::new(DomainAnalyzer::new(domain_config)?)),
            syntactic_analyzer: Arc::new(Mutex::new(SyntacticSimilarityAnalyzer::new(
                syntactic_config,
            )?)),
            metrics_analyzer: Arc::new(Mutex::new(SemanticMetricsAnalyzer::new(metrics_config)?)),
            cache: Arc::new(RwLock::new(HashMap::new())),
            cache_stats: Arc::new(Mutex::new(CacheInfo {
                cache_hits: 0,
                cache_misses: 0,
                cache_hit_rate: 0.0,
                cache_size: 0,
            })),
        })
    }

    /// Create a semantic analyzer with default configuration
    pub fn default() -> Result<Self, SemanticAnalysisError> {
        Self::new(SemanticAnalysisConfig::default())
    }

    /// Perform comprehensive semantic analysis between two texts
    pub async fn analyze_comprehensive(
        &mut self,
        text1: &str,
        text2: &str,
    ) -> Result<SemanticAnalysisResult, SemanticAnalysisError> {
        let start_time = Instant::now();

        // Check cache first
        if self.config.enable_caching {
            if let Some(cached_result) = self.check_cache(text1, text2).await? {
                return Ok(cached_result);
            }
        }

        // Create timeout future
        let timeout_duration = Duration::from_secs(self.config.timeout_seconds);
        let analysis_future = self.perform_analysis(text1, text2);

        let result = match tokio::time::timeout(timeout_duration, analysis_future).await {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                return Err(SemanticAnalysisError::TimeoutError {
                    timeout_seconds: self.config.timeout_seconds,
                })
            }
        };

        // Cache result if enabled
        if self.config.enable_caching {
            self.cache_result(text1, text2, &result).await?;
        }

        Ok(result)
    }

    /// Perform the actual analysis
    async fn perform_analysis(
        &mut self,
        text1: &str,
        text2: &str,
    ) -> Result<SemanticAnalysisResult, SemanticAnalysisError> {
        let start_time = Instant::now();

        // Analyze text characteristics
        let text1_chars = self.analyze_text_characteristics(text1);
        let text2_chars = self.analyze_text_characteristics(text2);

        // Collect enabled components
        let mut enabled_components = Vec::new();
        let mut performance = PerformanceInfo {
            total_time: Duration::ZERO,
            similarity_time: Duration::ZERO,
            feature_extraction_time: Duration::ZERO,
            sentiment_analysis_time: Duration::ZERO,
            topic_modeling_time: Duration::ZERO,
            domain_analysis_time: Duration::ZERO,
            syntactic_analysis_time: Duration::ZERO,
            metrics_analysis_time: Duration::ZERO,
            parallel_efficiency: 1.0,
        };

        // Perform analyses based on configuration
        let mut tasks = Vec::new();

        // Feature extraction (needed by similarity analysis)
        let (features1, features2) = if self.config.enable_feature_extraction {
            enabled_components.push("feature_extraction".to_string());
            let fe_start = Instant::now();

            let mut feature_extractor = self.feature_extractor.lock().expect("lock should not be poisoned");
            let f1 = feature_extractor.extract_features(text1)?;
            let f2 = feature_extractor.extract_features(text2)?;
            drop(feature_extractor);

            performance.feature_extraction_time = fe_start.elapsed();
            (Some(f1), Some(f2))
        } else {
            (None, None)
        };

        // Similarity analysis
        let similarity_result =
            if self.config.enable_similarity_analysis && features1.is_some() && features2.is_some()
            {
                enabled_components.push("similarity_analysis".to_string());
                let sim_start = Instant::now();

                let mut similarity_engine = self.similarity_engine.lock().expect("lock should not be poisoned");
                let result = similarity_engine.compute_similarity(
                    self.config.similarity_algorithm,
                    features1.as_ref().unwrap(),
                    features2.as_ref().unwrap(),
                )?;
                drop(similarity_engine);

                performance.similarity_time = sim_start.elapsed();
                Some(result)
            } else {
                None
            };

        // Parallel analysis tasks (if enabled)
        if self.config.enable_parallel_processing {
            let parallel_start = Instant::now();

            // Sentiment analysis
            if self.config.enable_sentiment_analysis {
                let sentiment_analyzer = Arc::clone(&self.sentiment_analyzer);
                let text1 = text1.to_string();
                let text2 = text2.to_string();

                tasks.push(task::spawn(async move {
                    let mut analyzer = sentiment_analyzer.lock().expect("lock should not be poisoned");
                    let s1 = analyzer.analyze_sentiment(&text1)?;
                    let s2 = analyzer.analyze_sentiment(&text2)?;
                    Ok::<_, SemanticAnalysisError>((s1, s2))
                }));
            }

            // Topic modeling
            if self.config.enable_topic_modeling {
                let topic_modeler = Arc::clone(&self.topic_modeler);
                let text1 = text1.to_string();
                let text2 = text2.to_string();

                tasks.push(task::spawn(async move {
                    let mut modeler = topic_modeler.lock().expect("lock should not be poisoned");
                    let t1 = modeler.extract_topics(&text1)?;
                    let t2 = modeler.extract_topics(&text2)?;
                    Ok::<_, SemanticAnalysisError>((t1, t2))
                }));
            }

            // Domain analysis
            if self.config.enable_domain_analysis {
                let domain_analyzer = Arc::clone(&self.domain_analyzer);
                let text1 = text1.to_string();
                let text2 = text2.to_string();

                tasks.push(task::spawn(async move {
                    let mut analyzer = domain_analyzer.lock().expect("lock should not be poisoned");
                    let d1 = analyzer.classify_domain(&text1)?;
                    let d2 = analyzer.classify_domain(&text2)?;
                    Ok::<_, SemanticAnalysisError>((d1, d2))
                }));
            }

            // Syntactic analysis
            if self.config.enable_syntactic_analysis {
                let syntactic_analyzer = Arc::clone(&self.syntactic_analyzer);
                let text1 = text1.to_string();
                let text2 = text2.to_string();

                tasks.push(task::spawn(async move {
                    let mut analyzer = syntactic_analyzer.lock().expect("lock should not be poisoned");
                    let result = analyzer.analyze_similarity(&text1, &text2)?;
                    Ok::<_, SemanticAnalysisError>(result)
                }));
            }

            let parallel_time = parallel_start.elapsed();
            performance.parallel_efficiency =
                calculate_parallel_efficiency(parallel_time, tasks.len());
        }

        // Collect results from parallel tasks
        let mut sentiment_result = None;
        let mut topic_result = None;
        let mut domain_result = None;
        let mut syntactic_result = None;

        let mut task_index = 0;
        if self.config.enable_sentiment_analysis {
            if let Some(task) = tasks.get(task_index) {
                match task.await {
                    Ok(Ok(result)) => {
                        sentiment_result = Some(result);
                        enabled_components.push("sentiment_analysis".to_string());
                    }
                    Ok(Err(e)) => return Err(e),
                    Err(_) => {
                        return Err(SemanticAnalysisError::ParallelProcessingError {
                            reason: "Sentiment analysis task failed".to_string(),
                        })
                    }
                }
                task_index += 1;
            }
        }

        if self.config.enable_topic_modeling {
            if let Some(task) = tasks.get(task_index) {
                match task.await {
                    Ok(Ok(result)) => {
                        topic_result = Some(result);
                        enabled_components.push("topic_modeling".to_string());
                    }
                    Ok(Err(e)) => return Err(e),
                    Err(_) => {
                        return Err(SemanticAnalysisError::ParallelProcessingError {
                            reason: "Topic modeling task failed".to_string(),
                        })
                    }
                }
                task_index += 1;
            }
        }

        if self.config.enable_domain_analysis {
            if let Some(task) = tasks.get(task_index) {
                match task.await {
                    Ok(Ok(result)) => {
                        domain_result = Some(result);
                        enabled_components.push("domain_analysis".to_string());
                    }
                    Ok(Err(e)) => return Err(e),
                    Err(_) => {
                        return Err(SemanticAnalysisError::ParallelProcessingError {
                            reason: "Domain analysis task failed".to_string(),
                        })
                    }
                }
                task_index += 1;
            }
        }

        if self.config.enable_syntactic_analysis {
            if let Some(task) = tasks.get(task_index) {
                match task.await {
                    Ok(Ok(result)) => {
                        syntactic_result = Some(result);
                        enabled_components.push("syntactic_analysis".to_string());
                    }
                    Ok(Err(e)) => return Err(e),
                    Err(_) => {
                        return Err(SemanticAnalysisError::ParallelProcessingError {
                            reason: "Syntactic analysis task failed".to_string(),
                        })
                    }
                }
            }
        }

        // Metrics analysis
        let metrics_result = if self.config.enable_metrics_collection {
            enabled_components.push("metrics_collection".to_string());
            let metrics_start = Instant::now();

            // Collect scores for metrics analysis
            let similarity_scores = vec![similarity_result
                .as_ref()
                .map(|r| r.similarity_score)
                .unwrap_or(0.0)];
            let quality_scores = vec![0.8]; // Simplified
            let confidence_scores = vec![0.9]; // Simplified

            let mut metrics_analyzer = self.metrics_analyzer.lock().expect("lock should not be poisoned");
            let result = metrics_analyzer.analyze_metrics(
                &similarity_scores,
                &quality_scores,
                &confidence_scores,
            )?;
            drop(metrics_analyzer);

            performance.metrics_analysis_time = metrics_start.elapsed();
            Some(result)
        } else {
            None
        };

        // Calculate overall scores
        let overall_similarity = similarity_result
            .as_ref()
            .map(|r| r.similarity_score)
            .unwrap_or(0.0);
        let quality_score = metrics_result
            .as_ref()
            .map(|r| r.summary.quality_stats.mean)
            .unwrap_or(0.8);
        let confidence_score = metrics_result
            .as_ref()
            .map(|r| r.summary.confidence_stats.mean)
            .unwrap_or(0.9);

        // Calculate total processing time
        let total_time = start_time.elapsed();
        performance.total_time = total_time;

        // Create cache info
        let cache_info = {
            let cache_stats = self.cache_stats.lock().expect("lock should not be poisoned");
            cache_stats.clone()
        };

        let metadata = AnalysisMetadata {
            timestamp: SystemTime::now(),
            processing_time: total_time,
            config_used: format!("{:?}", self.config),
            enabled_components,
            text1_characteristics: text1_chars,
            text2_characteristics: text2_chars,
            performance_metrics: performance,
            cache_info,
        };

        Ok(SemanticAnalysisResult {
            overall_similarity,
            quality_score,
            confidence_score,
            similarity_analysis: similarity_result,
            feature_analysis: if features1.is_some() && features2.is_some() {
                Some((features1.unwrap(), features2.unwrap()))
            } else {
                None
            },
            sentiment_analysis: sentiment_result,
            topic_analysis: topic_result,
            domain_analysis: domain_result,
            syntactic_analysis: syntactic_result,
            metrics_analysis: metrics_result,
            metadata,
        })
    }

    /// Analyze batch of text pairs
    pub async fn analyze_batch(
        &mut self,
        text_pairs: &[(&str, &str)],
    ) -> Result<BatchAnalysisResult, SemanticAnalysisError> {
        let start_time = Instant::now();
        let mut results = Vec::new();
        let mut successful_analyses = 0;
        let mut failed_analyses = 0;
        let mut similarity_sum = 0.0;

        // Process in parallel if enabled
        if self.config.enable_parallel_processing && text_pairs.len() > 1 {
            let chunk_size = (text_pairs.len() + self.config.max_parallel_threads - 1)
                / self.config.max_parallel_threads;
            let chunks: Vec<_> = text_pairs.chunks(chunk_size).collect();

            let mut tasks = Vec::new();
            for chunk in chunks {
                let mut analyzer_clone = self.clone_for_parallel().await?;
                let chunk_vec = chunk.to_vec();

                tasks.push(task::spawn(async move {
                    let mut chunk_results = Vec::new();
                    for (text1, text2) in chunk_vec {
                        match analyzer_clone.analyze_comprehensive(text1, text2).await {
                            Ok(result) => chunk_results.push(Ok(result)),
                            Err(e) => chunk_results.push(Err(e)),
                        }
                    }
                    chunk_results
                }));
            }

            // Collect results from all tasks
            for task in tasks {
                match task.await {
                    Ok(chunk_results) => {
                        for result in chunk_results {
                            match result {
                                Ok(analysis_result) => {
                                    similarity_sum += analysis_result.overall_similarity;
                                    successful_analyses += 1;
                                    results.push(analysis_result);
                                }
                                Err(_) => {
                                    failed_analyses += 1;
                                }
                            }
                        }
                    }
                    Err(_) => {
                        failed_analyses += text_pairs.len();
                    }
                }
            }
        } else {
            // Sequential processing
            for (text1, text2) in text_pairs {
                match self.analyze_comprehensive(text1, text2).await {
                    Ok(result) => {
                        similarity_sum += result.overall_similarity;
                        successful_analyses += 1;
                        results.push(result);
                    }
                    Err(_) => {
                        failed_analyses += 1;
                    }
                }
            }
        }

        let total_time = start_time.elapsed();
        let average_similarity = if successful_analyses > 0 {
            similarity_sum / successful_analyses as f64
        } else {
            0.0
        };

        // Calculate distributions
        let similarity_distribution = calculate_distribution(&results, |r| r.overall_similarity);
        let quality_distribution = calculate_distribution(&results, |r| r.quality_score);

        let batch_statistics = BatchStatistics {
            total_items: text_pairs.len(),
            successful_analyses,
            failed_analyses,
            average_similarity,
            similarity_distribution,
            quality_distribution,
        };

        let performance_summary = BatchPerformance {
            total_time,
            average_time_per_item: if results.len() > 0 {
                total_time / results.len() as u32
            } else {
                Duration::ZERO
            },
            items_per_second: if total_time.as_secs_f64() > 0.0 {
                results.len() as f64 / total_time.as_secs_f64()
            } else {
                0.0
            },
            parallel_speedup: if self.config.enable_parallel_processing {
                calculate_speedup(text_pairs.len(), self.config.max_parallel_threads)
            } else {
                1.0
            },
            memory_usage: estimate_memory_usage(&results),
        };

        Ok(BatchAnalysisResult {
            results,
            batch_statistics,
            performance_summary,
        })
    }

    /// Analyze single text for characteristics
    pub async fn analyze_single(
        &mut self,
        text: &str,
    ) -> Result<SemanticAnalysisResult, SemanticAnalysisError> {
        // For single text analysis, we use an empty string as the second text
        // This allows us to extract features, sentiment, topics, and domain for a single text
        self.analyze_comprehensive(text, "").await
    }

    /// Check cache for existing result
    async fn check_cache(
        &self,
        text1: &str,
        text2: &str,
    ) -> Result<Option<SemanticAnalysisResult>, SemanticAnalysisError> {
        if !self.config.enable_caching {
            return Ok(None);
        }

        let cache_key = generate_cache_key(text1, text2);
        let cache = self.cache.read().expect("lock should not be poisoned");

        if let Some(entry) = cache.get(&cache_key) {
            // Update cache stats
            {
                let mut stats = self.cache_stats.lock().expect("lock should not be poisoned");
                stats.cache_hits += 1;
                stats.cache_hit_rate =
                    stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64;
            }

            return Ok(Some(entry.result.clone()));
        }

        // Cache miss
        {
            let mut stats = self.cache_stats.lock().expect("lock should not be poisoned");
            stats.cache_misses += 1;
            stats.cache_hit_rate =
                stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64;
        }

        Ok(None)
    }

    /// Cache analysis result
    async fn cache_result(
        &self,
        text1: &str,
        text2: &str,
        result: &SemanticAnalysisResult,
    ) -> Result<(), SemanticAnalysisError> {
        if !self.config.enable_caching {
            return Ok(());
        }

        let cache_key = generate_cache_key(text1, text2);
        let entry = CacheEntry {
            result: result.clone(),
            timestamp: SystemTime::now(),
            access_count: 1,
        };

        let mut cache = self.cache.write().expect("lock should not be poisoned");

        // Check if cache is full and evict if necessary
        if cache.len() >= self.config.cache_size_limit {
            evict_lru_entry(&mut cache);
        }

        cache.insert(cache_key, entry);

        // Update cache stats
        {
            let mut stats = self.cache_stats.lock().expect("lock should not be poisoned");
            stats.cache_size = cache.len();
        }

        Ok(())
    }

    /// Clone analyzer for parallel processing
    async fn clone_for_parallel(&self) -> Result<Self, SemanticAnalysisError> {
        // Create a new analyzer with the same configuration but separate instances
        Self::new(self.config.clone())
    }

    /// Analyze text characteristics
    fn analyze_text_characteristics(&self, text: &str) -> TextCharacteristics {
        let length = text.len();
        let words: Vec<&str> = text.split_whitespace().collect();
        let word_count = words.len();
        let sentence_count = text
            .matches(|c| c == '.' || c == '!' || c == '?')
            .count()
            .max(1);
        let paragraph_count = text.split("\n\n").count();

        // Simple complexity score based on average word length and sentence length
        let avg_word_length = if word_count > 0 {
            words.iter().map(|w| w.len()).sum::<usize>() as f64 / word_count as f64
        } else {
            0.0
        };
        let avg_sentence_length = word_count as f64 / sentence_count as f64;
        let complexity_score = (avg_word_length + avg_sentence_length) / 20.0; // Normalized

        // Simple readability score (inverse of complexity)
        let readability_score = 1.0 - (complexity_score.min(1.0));

        TextCharacteristics {
            length,
            word_count,
            sentence_count,
            paragraph_count,
            language_detected: Some("en".to_string()), // Simplified
            complexity_score,
            readability_score,
        }
    }

    /// Clear cache
    pub async fn clear_cache(&mut self) -> Result<(), SemanticAnalysisError> {
        let mut cache = self.cache.write().expect("lock should not be poisoned");
        cache.clear();

        let mut stats = self.cache_stats.lock().expect("lock should not be poisoned");
        stats.cache_size = 0;
        stats.cache_hits = 0;
        stats.cache_misses = 0;
        stats.cache_hit_rate = 0.0;

        Ok(())
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheInfo {
        let stats = self.cache_stats.lock().expect("lock should not be poisoned");
        stats.clone()
    }

    /// Update configuration
    pub fn update_config(
        &mut self,
        config: SemanticAnalysisConfig,
    ) -> Result<(), SemanticAnalysisError> {
        config.validate()?;
        self.config = config;
        Ok(())
    }

    /// Get current configuration
    pub fn get_config(&self) -> &SemanticAnalysisConfig {
        &self.config
    }
}

// Helper functions

/// Generate cache key from two texts
fn generate_cache_key(text1: &str, text2: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    text1.hash(&mut hasher);
    text2.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// Evict least recently used cache entry
fn evict_lru_entry(cache: &mut HashMap<String, CacheEntry>) {
    if let Some((key_to_remove, _)) = cache.iter().min_by_key(|(_, entry)| entry.timestamp) {
        let key_to_remove = key_to_remove.clone();
        cache.remove(&key_to_remove);
    }
}

/// Calculate parallel efficiency
fn calculate_parallel_efficiency(parallel_time: Duration, num_tasks: usize) -> f64 {
    if num_tasks <= 1 {
        return 1.0;
    }

    // Simplified efficiency calculation
    let theoretical_speedup = num_tasks as f64;
    let actual_efficiency = 0.8; // Assume 80% efficiency with overhead
    actual_efficiency.min(1.0)
}

/// Calculate distribution of values
fn calculate_distribution<F>(
    results: &[SemanticAnalysisResult],
    value_fn: F,
) -> HashMap<String, usize>
where
    F: Fn(&SemanticAnalysisResult) -> f64,
{
    let mut distribution = HashMap::new();
    let ranges = vec![
        ("0.0-0.1", 0.0..0.1),
        ("0.1-0.2", 0.1..0.2),
        ("0.2-0.3", 0.2..0.3),
        ("0.3-0.4", 0.3..0.4),
        ("0.4-0.5", 0.4..0.5),
        ("0.5-0.6", 0.5..0.6),
        ("0.6-0.7", 0.6..0.7),
        ("0.7-0.8", 0.7..0.8),
        ("0.8-0.9", 0.8..0.9),
        ("0.9-1.0", 0.9..=1.0),
    ];

    for result in results {
        let value = value_fn(result);
        for (range_name, range) in &ranges {
            if range.contains(&value) {
                *distribution.entry(range_name.to_string()).or_insert(0) += 1;
                break;
            }
        }
    }

    distribution
}

/// Calculate parallel speedup
fn calculate_speedup(total_items: usize, max_threads: usize) -> f64 {
    if max_threads <= 1 {
        return 1.0;
    }

    let effective_threads = max_threads.min(total_items);
    effective_threads as f64 * 0.8 // Assume 80% efficiency
}

/// Estimate memory usage
fn estimate_memory_usage(results: &[SemanticAnalysisResult]) -> usize {
    results.len() * 10240 // Estimate 10KB per result
}

/// Convenience functions

/// Analyze similarity between two texts with default configuration
pub async fn analyze_similarity(
    text1: &str,
    text2: &str,
) -> Result<SemanticAnalysisResult, SemanticAnalysisError> {
    let mut analyzer = SemanticAnalyzer::default()?;
    analyzer.analyze_comprehensive(text1, text2).await
}

/// Analyze similarity with custom configuration
pub async fn analyze_similarity_with_config(
    text1: &str,
    text2: &str,
    config: SemanticAnalysisConfig,
) -> Result<SemanticAnalysisResult, SemanticAnalysisError> {
    let mut analyzer = SemanticAnalyzer::new(config)?;
    analyzer.analyze_comprehensive(text1, text2).await
}

/// Analyze batch of text pairs with default configuration
pub async fn analyze_batch(
    text_pairs: &[(&str, &str)],
) -> Result<BatchAnalysisResult, SemanticAnalysisError> {
    let mut analyzer = SemanticAnalyzer::default()?;
    analyzer.analyze_batch(text_pairs).await
}

/// Quick similarity check (simplified analysis)
pub async fn quick_similarity(text1: &str, text2: &str) -> Result<f64, SemanticAnalysisError> {
    let config = SemanticAnalysisConfig::builder()
        .enable_similarity_analysis(true)
        .enable_feature_extraction(true)
        .enable_sentiment_analysis(false)
        .enable_topic_modeling(false)
        .enable_domain_analysis(false)
        .enable_syntactic_analysis(false)
        .enable_metrics_collection(false)
        .build()?;

    let result = analyze_similarity_with_config(text1, text2, config).await?;
    Ok(result.overall_similarity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_analyzer_creation() {
        let analyzer = SemanticAnalyzer::default();
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_configuration_builder() {
        let config = SemanticAnalysisConfig::builder()
            .enable_similarity_analysis(true)
            .enable_sentiment_analysis(false)
            .similarity_algorithm(SimilarityAlgorithm::Cosine)
            .enable_caching(true)
            .cache_size_limit(500)
            .enable_parallel_processing(true)
            .max_parallel_threads(4)
            .timeout_seconds(60)
            .quality_threshold(0.5)
            .build();

        assert!(config.is_ok());
        let config = config.unwrap();
        assert!(config.enable_similarity_analysis);
        assert!(!config.enable_sentiment_analysis);
        assert_eq!(config.cache_size_limit, 500);
        assert_eq!(config.max_parallel_threads, 4);
    }

    #[test]
    fn test_invalid_configuration() {
        let config = SemanticAnalysisConfig::builder()
            .enable_caching(true)
            .cache_size_limit(0) // Invalid: cache enabled but size 0
            .build();

        assert!(config.is_err());
    }

    #[tokio::test]
    async fn test_comprehensive_analysis() {
        let mut analyzer = SemanticAnalyzer::default().unwrap();
        let text1 = "The quick brown fox jumps over the lazy dog.";
        let text2 = "A fast brown fox leaps across the sleeping dog.";

        let result = analyzer.analyze_comprehensive(text1, text2).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.overall_similarity >= 0.0 && result.overall_similarity <= 1.0);
        assert!(result.quality_score >= 0.0 && result.quality_score <= 1.0);
        assert!(result.confidence_score >= 0.0 && result.confidence_score <= 1.0);
    }

    #[tokio::test]
    async fn test_single_text_analysis() {
        let mut analyzer = SemanticAnalyzer::default().unwrap();
        let text = "This is a sample text for analysis.";

        let result = analyzer.analyze_single(text).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.metadata.text1_characteristics.word_count, 7);
    }

    #[tokio::test]
    async fn test_batch_analysis() {
        let mut analyzer = SemanticAnalyzer::default().unwrap();
        let text_pairs = vec![
            ("Hello world", "Hi there"),
            ("Technical document", "Academic paper"),
            ("Simple sentence", "Complex paragraph with multiple clauses"),
        ];

        let result = analyzer.analyze_batch(&text_pairs).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.batch_statistics.total_items, 3);
        assert!(result.batch_statistics.successful_analyses > 0);
    }

    #[tokio::test]
    async fn test_caching() {
        let config = SemanticAnalysisConfig::builder()
            .enable_caching(true)
            .cache_size_limit(10)
            .build()
            .unwrap();

        let mut analyzer = SemanticAnalyzer::new(config).unwrap();
        let text1 = "Sample text";
        let text2 = "Another text";

        // First analysis should be a cache miss
        let _result1 = analyzer.analyze_comprehensive(text1, text2).await.unwrap();
        let stats1 = analyzer.get_cache_stats();

        // Second analysis should be a cache hit
        let _result2 = analyzer.analyze_comprehensive(text1, text2).await.unwrap();
        let stats2 = analyzer.get_cache_stats();

        assert!(stats2.cache_hits > stats1.cache_hits);
    }

    #[tokio::test]
    async fn test_convenience_functions() {
        let text1 = "The weather is nice today.";
        let text2 = "It's a beautiful day outside.";

        let result = analyze_similarity(text1, text2).await;
        assert!(result.is_ok());

        let similarity = quick_similarity(text1, text2).await;
        assert!(similarity.is_ok());
        assert!(similarity.unwrap() >= 0.0);
    }

    #[tokio::test]
    async fn test_timeout_handling() {
        let config = SemanticAnalysisConfig::builder()
            .timeout_seconds(1) // Very short timeout
            .build()
            .unwrap();

        let mut analyzer = SemanticAnalyzer::new(config).unwrap();

        // Use very long texts that might cause timeout
        let long_text = "word ".repeat(10000);
        let result = analyzer.analyze_comprehensive(&long_text, &long_text).await;

        // Result might be ok if processing is fast enough, or timeout error
        match result {
            Ok(_) => {}                                           // Analysis completed within timeout
            Err(SemanticAnalysisError::TimeoutError { .. }) => {} // Expected timeout
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_text_characteristics() {
        let analyzer = SemanticAnalyzer::default().unwrap();
        let text = "This is a sample sentence. It has multiple words and punctuation!";
        let characteristics = analyzer.analyze_text_characteristics(text);

        assert_eq!(characteristics.sentence_count, 2);
        assert!(characteristics.word_count > 0);
        assert!(characteristics.complexity_score >= 0.0);
        assert!(characteristics.readability_score >= 0.0);
    }

    #[test]
    fn test_cache_key_generation() {
        let key1 = generate_cache_key("text1", "text2");
        let key2 = generate_cache_key("text1", "text2");
        let key3 = generate_cache_key("text2", "text1");

        assert_eq!(key1, key2); // Same texts should generate same key
        assert_ne!(key1, key3); // Different order should generate different key
    }

    #[test]
    fn test_distribution_calculation() {
        let results = vec![SemanticAnalysisResult {
            overall_similarity: 0.1,
            quality_score: 0.8,
            confidence_score: 0.9,
            similarity_analysis: None,
            feature_analysis: None,
            sentiment_analysis: None,
            topic_analysis: None,
            domain_analysis: None,
            syntactic_analysis: None,
            metrics_analysis: None,
            metadata: AnalysisMetadata {
                timestamp: SystemTime::now(),
                processing_time: Duration::from_millis(100),
                config_used: "test".to_string(),
                enabled_components: vec![],
                text1_characteristics: TextCharacteristics {
                    length: 10,
                    word_count: 2,
                    sentence_count: 1,
                    paragraph_count: 1,
                    language_detected: None,
                    complexity_score: 0.5,
                    readability_score: 0.5,
                },
                text2_characteristics: TextCharacteristics {
                    length: 10,
                    word_count: 2,
                    sentence_count: 1,
                    paragraph_count: 1,
                    language_detected: None,
                    complexity_score: 0.5,
                    readability_score: 0.5,
                },
                performance_metrics: PerformanceInfo {
                    total_time: Duration::from_millis(100),
                    similarity_time: Duration::ZERO,
                    feature_extraction_time: Duration::ZERO,
                    sentiment_analysis_time: Duration::ZERO,
                    topic_modeling_time: Duration::ZERO,
                    domain_analysis_time: Duration::ZERO,
                    syntactic_analysis_time: Duration::ZERO,
                    metrics_analysis_time: Duration::ZERO,
                    parallel_efficiency: 1.0,
                },
                cache_info: CacheInfo {
                    cache_hits: 0,
                    cache_misses: 0,
                    cache_hit_rate: 0.0,
                    cache_size: 0,
                },
            },
        }];

        let distribution = calculate_distribution(&results, |r| r.overall_similarity);
        assert!(distribution.contains_key("0.0-0.1"));
        assert_eq!(distribution["0.0-0.1"], 1);
    }

    #[tokio::test]
    async fn test_cache_clearing() {
        let config = SemanticAnalysisConfig::builder()
            .enable_caching(true)
            .build()
            .unwrap();

        let mut analyzer = SemanticAnalyzer::new(config).unwrap();

        // Add something to cache
        let _ = analyzer.analyze_comprehensive("test1", "test2").await;
        assert!(
            analyzer.get_cache_stats().cache_size > 0
                || analyzer.get_cache_stats().cache_misses > 0
        );

        // Clear cache
        let _ = analyzer.clear_cache().await;
        let stats = analyzer.get_cache_stats();
        assert_eq!(stats.cache_size, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
    }
}
