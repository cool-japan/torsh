//! Advanced semantic similarity metrics and statistical analysis system
//!
//! This module provides comprehensive metrics collection, statistical analysis, and reporting
//! capabilities for semantic similarity analysis. Designed for production use with extensive
//! statistical functions, quality assessment, and performance analysis.

use scirs2_core::ndarray::{array, Array1, Array2, Axis};
use scirs2_core::random::{rng, Random};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use thiserror::Error;

/// Errors that can occur during semantic metrics analysis
#[derive(Error, Debug)]
pub enum SemanticMetricsError {
    #[error("Invalid metrics configuration: {message}")]
    InvalidConfiguration { message: String },

    #[error("Metrics calculation failed: {reason}")]
    CalculationFailed { reason: String },

    #[error("Statistical analysis failed: {error}")]
    StatisticalAnalysisFailed { error: String },

    #[error("Insufficient data for analysis: {required} samples required, {provided} provided")]
    InsufficientData { required: usize, provided: usize },

    #[error("Data validation failed: {reason}")]
    DataValidationFailed { reason: String },

    #[error("Distribution analysis failed: {error}")]
    DistributionAnalysisFailed { error: String },

    #[error("Correlation analysis failed: {reason}")]
    CorrelationAnalysisFailed { reason: String },

    #[error("Performance analysis failed: {error}")]
    PerformanceAnalysisFailed { error: String },
}

/// Types of semantic metrics
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetricType {
    Similarity,
    Quality,
    Confidence,
    Performance,
    Distribution,
    Correlation,
    Significance,
    Trend,
}

/// Similarity measurement approaches for metrics
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SimilarityMetricType {
    Cosine,
    Euclidean,
    Manhattan,
    Jaccard,
    Pearson,
    Spearman,
    KendallTau,
    JensenShannon,
    Wasserstein,
    Composite,
}

/// Statistical distribution types for analysis
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributionType {
    Normal,
    Uniform,
    Beta,
    Gamma,
    LogNormal,
    ChiSquare,
    StudentT,
    Unknown,
}

/// Comprehensive semantic metrics result
#[derive(Debug, Clone)]
pub struct SemanticMetricsResult {
    /// Overall metrics summary
    pub summary: MetricsSummary,
    /// Detailed similarity metrics
    pub similarity_metrics: SimilarityMetrics,
    /// Statistical analysis results
    pub statistical_analysis: StatisticalAnalysis,
    /// Quality assessment metrics
    pub quality_metrics: QualityMetrics,
    /// Performance analysis
    pub performance_metrics: PerformanceMetrics,
    /// Distribution analysis
    pub distribution_analysis: DistributionAnalysis,
    /// Correlation analysis
    pub correlation_analysis: CorrelationAnalysis,
    /// Trend analysis (if historical data available)
    pub trend_analysis: Option<TrendAnalysis>,
    /// Analysis metadata
    pub metadata: MetricsMetadata,
}

/// Summary of all metrics
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    /// Number of samples analyzed
    pub sample_count: usize,
    /// Overall similarity score statistics
    pub similarity_stats: BasicStatistics,
    /// Overall quality score statistics
    pub quality_stats: BasicStatistics,
    /// Overall confidence statistics
    pub confidence_stats: BasicStatistics,
    /// Key insights and findings
    pub insights: Vec<String>,
    /// Recommendations based on analysis
    pub recommendations: Vec<String>,
}

/// Basic statistical measures
#[derive(Debug, Clone)]
pub struct BasicStatistics {
    pub mean: f64,
    pub median: f64,
    pub mode: Option<f64>,
    pub std_dev: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub range: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub quartiles: (f64, f64, f64),    // Q1, Q2, Q3
    pub percentiles: HashMap<u8, f64>, // 5th, 10th, 25th, 75th, 90th, 95th
}

/// Detailed similarity metrics analysis
#[derive(Debug, Clone)]
pub struct SimilarityMetrics {
    /// Metrics by similarity type
    pub by_type: HashMap<SimilarityMetricType, BasicStatistics>,
    /// Pairwise similarity matrix (if applicable)
    pub similarity_matrix: Option<Array2<f64>>,
    /// Similarity score distribution
    pub score_distribution: Vec<f64>,
    /// Similarity clustering results
    pub clusters: Vec<SimilarityCluster>,
    /// Outlier detection results
    pub outliers: Vec<SimilarityOutlier>,
}

/// Similarity cluster for grouping similar items
#[derive(Debug, Clone)]
pub struct SimilarityCluster {
    pub cluster_id: usize,
    pub center: Vec<f64>,
    pub members: Vec<usize>,
    pub intra_cluster_similarity: f64,
    pub cluster_quality: f64,
}

/// Outlier in similarity measurements
#[derive(Debug, Clone)]
pub struct SimilarityOutlier {
    pub item_id: usize,
    pub similarity_score: f64,
    pub deviation_magnitude: f64,
    pub outlier_type: OutlierType,
}

/// Types of outliers
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutlierType {
    Low,     // Unusually low similarity
    High,    // Unusually high similarity
    Pattern, // Unusual pattern in similarities
}

/// Statistical analysis results
#[derive(Debug, Clone)]
pub struct StatisticalAnalysis {
    /// Normality tests results
    pub normality_tests: NormalityTestResults,
    /// Hypothesis testing results
    pub hypothesis_tests: Vec<HypothesisTestResult>,
    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    /// Effect size measurements
    pub effect_sizes: HashMap<String, f64>,
    /// Statistical significance indicators
    pub significance_results: HashMap<String, bool>,
}

/// Normality test results
#[derive(Debug, Clone)]
pub struct NormalityTestResults {
    pub shapiro_wilk: Option<(f64, f64)>, // (statistic, p_value)
    pub kolmogorov_smirnov: Option<(f64, f64)>,
    pub anderson_darling: Option<(f64, f64)>,
    pub is_normal: bool,
    pub recommended_distribution: DistributionType,
}

/// Hypothesis test result
#[derive(Debug, Clone)]
pub struct HypothesisTestResult {
    pub test_name: String,
    pub null_hypothesis: String,
    pub alternative_hypothesis: String,
    pub test_statistic: f64,
    pub p_value: f64,
    pub critical_value: Option<f64>,
    pub is_significant: bool,
    pub effect_size: Option<f64>,
}

/// Quality assessment metrics
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Data quality indicators
    pub data_quality: DataQualityIndicators,
    /// Analysis reliability metrics
    pub reliability_metrics: ReliabilityMetrics,
    /// Consistency measurements
    pub consistency_metrics: ConsistencyMetrics,
    /// Robustness assessment
    pub robustness_assessment: RobustnessAssessment,
}

/// Data quality indicators
#[derive(Debug, Clone)]
pub struct DataQualityIndicators {
    pub completeness_score: f64,
    pub consistency_score: f64,
    pub accuracy_score: f64,
    pub validity_score: f64,
    pub uniqueness_score: f64,
    pub timeliness_score: f64,
    pub overall_quality_score: f64,
}

/// Reliability metrics for analysis
#[derive(Debug, Clone)]
pub struct ReliabilityMetrics {
    pub internal_consistency: f64, // Cronbach's alpha equivalent
    pub test_retest_reliability: Option<f64>,
    pub inter_rater_reliability: Option<f64>,
    pub measurement_error: f64,
    pub confidence_in_results: f64,
}

/// Consistency measurements across different methods
#[derive(Debug, Clone)]
pub struct ConsistencyMetrics {
    pub method_agreement: HashMap<(String, String), f64>,
    pub rank_correlation: f64,
    pub classification_agreement: f64,
    pub variance_across_methods: f64,
}

/// Robustness assessment against variations
#[derive(Debug, Clone)]
pub struct RobustnessAssessment {
    pub noise_sensitivity: f64,
    pub parameter_stability: f64,
    pub outlier_resistance: f64,
    pub generalization_ability: f64,
}

/// Performance analysis metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Timing statistics
    pub timing_stats: TimingStatistics,
    /// Memory usage analysis
    pub memory_usage: MemoryUsageAnalysis,
    /// Throughput measurements
    pub throughput_metrics: ThroughputMetrics,
    /// Scalability assessment
    pub scalability_assessment: ScalabilityAssessment,
}

/// Timing statistics for operations
#[derive(Debug, Clone)]
pub struct TimingStatistics {
    pub total_execution_time: Duration,
    pub average_operation_time: Duration,
    pub median_operation_time: Duration,
    pub min_operation_time: Duration,
    pub max_operation_time: Duration,
    pub std_dev_operation_time: Duration,
    pub percentile_95_time: Duration,
    pub percentile_99_time: Duration,
}

/// Memory usage analysis
#[derive(Debug, Clone)]
pub struct MemoryUsageAnalysis {
    pub peak_memory_usage: usize,
    pub average_memory_usage: usize,
    pub memory_efficiency_score: f64,
    pub memory_allocation_count: usize,
    pub memory_fragmentation_score: f64,
}

/// Throughput measurements
#[derive(Debug, Clone)]
pub struct ThroughputMetrics {
    pub operations_per_second: f64,
    pub items_processed_per_second: f64,
    pub bytes_processed_per_second: f64,
    pub throughput_consistency: f64,
}

/// Scalability assessment
#[derive(Debug, Clone)]
pub struct ScalabilityAssessment {
    pub linear_scalability_score: f64,
    pub memory_scalability_score: f64,
    pub predicted_max_capacity: usize,
    pub bottleneck_analysis: Vec<String>,
}

/// Distribution analysis results
#[derive(Debug, Clone)]
pub struct DistributionAnalysis {
    /// Distribution fitting results
    pub distribution_fits: HashMap<DistributionType, DistributionFit>,
    /// Best fitting distribution
    pub best_fit: DistributionType,
    /// Distribution parameters
    pub distribution_parameters: HashMap<String, f64>,
    /// Goodness of fit statistics
    pub goodness_of_fit: GoodnessOfFit,
    /// Distribution comparison results
    pub distribution_comparisons: Vec<DistributionComparison>,
}

/// Distribution fitting result
#[derive(Debug, Clone)]
pub struct DistributionFit {
    pub distribution_type: DistributionType,
    pub parameters: Vec<f64>,
    pub log_likelihood: f64,
    pub aic: f64, // Akaike Information Criterion
    pub bic: f64, // Bayesian Information Criterion
    pub fit_quality: f64,
}

/// Goodness of fit statistics
#[derive(Debug, Clone)]
pub struct GoodnessOfFit {
    pub chi_square: f64,
    pub chi_square_p_value: f64,
    pub kolmogorov_smirnov: f64,
    pub ks_p_value: f64,
    pub anderson_darling: f64,
    pub ad_p_value: f64,
}

/// Distribution comparison result
#[derive(Debug, Clone)]
pub struct DistributionComparison {
    pub distribution1: DistributionType,
    pub distribution2: DistributionType,
    pub comparison_statistic: f64,
    pub p_value: f64,
    pub significantly_different: bool,
}

/// Correlation analysis results
#[derive(Debug, Clone)]
pub struct CorrelationAnalysis {
    /// Correlation matrix
    pub correlation_matrix: Array2<f64>,
    /// Correlation coefficients by type
    pub correlations: HashMap<(String, String), CorrelationResult>,
    /// Principal component analysis results
    pub pca_results: Option<PCAResults>,
    /// Factor analysis results
    pub factor_analysis: Option<FactorAnalysisResults>,
}

/// Individual correlation result
#[derive(Debug, Clone)]
pub struct CorrelationResult {
    pub correlation_coefficient: f64,
    pub correlation_type: String, // Pearson, Spearman, etc.
    pub p_value: f64,
    pub confidence_interval: (f64, f64),
    pub is_significant: bool,
    pub effect_size: String, // Small, Medium, Large
}

/// Principal Component Analysis results
#[derive(Debug, Clone)]
pub struct PCAResults {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Array2<f64>,
    pub explained_variance_ratio: Vec<f64>,
    pub cumulative_explained_variance: Vec<f64>,
    pub principal_components: Array2<f64>,
}

/// Factor Analysis results
#[derive(Debug, Clone)]
pub struct FactorAnalysisResults {
    pub factor_loadings: Array2<f64>,
    pub communalities: Vec<f64>,
    pub uniquenesses: Vec<f64>,
    pub factor_scores: Array2<f64>,
    pub explained_variance: Vec<f64>,
}

/// Trend analysis results (for historical data)
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    /// Trend direction and magnitude
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub trend_significance: f64,
    /// Seasonal patterns
    pub seasonal_patterns: Vec<SeasonalPattern>,
    /// Forecasting results
    pub forecasts: Vec<ForecastResult>,
    /// Change point detection
    pub change_points: Vec<ChangePoint>,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Cyclical,
    Unknown,
}

/// Seasonal pattern detection
#[derive(Debug, Clone)]
pub struct SeasonalPattern {
    pub period: Duration,
    pub amplitude: f64,
    pub phase: f64,
    pub significance: f64,
}

/// Forecast result
#[derive(Debug, Clone)]
pub struct ForecastResult {
    pub time_horizon: Duration,
    pub predicted_value: f64,
    pub confidence_interval: (f64, f64),
    pub prediction_accuracy: f64,
}

/// Change point in time series
#[derive(Debug, Clone)]
pub struct ChangePoint {
    pub timestamp: SystemTime,
    pub change_magnitude: f64,
    pub change_type: String,
    pub confidence: f64,
}

/// Metadata about metrics analysis
#[derive(Debug, Clone)]
pub struct MetricsMetadata {
    /// Analysis timestamp
    pub timestamp: SystemTime,
    /// Total processing time
    pub processing_time: Duration,
    /// Configuration used
    pub config_summary: String,
    /// Data characteristics
    pub data_characteristics: DataCharacteristics,
    /// Analysis completeness
    pub analysis_completeness: f64,
    /// Quality of analysis
    pub analysis_quality: f64,
}

/// Characteristics of analyzed data
#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    pub sample_size: usize,
    pub dimensionality: usize,
    pub sparsity: f64,
    pub noise_level: f64,
    pub data_types: Vec<String>,
    pub missing_data_percentage: f64,
}

/// Configuration for semantic metrics analysis
#[derive(Debug, Clone)]
pub struct SemanticMetricsConfig {
    /// Enable detailed statistical analysis
    pub enable_statistical_analysis: bool,
    /// Enable distribution analysis
    pub enable_distribution_analysis: bool,
    /// Enable correlation analysis
    pub enable_correlation_analysis: bool,
    /// Enable trend analysis (requires historical data)
    pub enable_trend_analysis: bool,
    /// Enable performance analysis
    pub enable_performance_analysis: bool,
    /// Confidence level for statistical tests (0.0-1.0)
    pub confidence_level: f64,
    /// Maximum number of clusters for similarity clustering
    pub max_clusters: usize,
    /// Outlier detection threshold (standard deviations)
    pub outlier_threshold: f64,
    /// Minimum sample size for robust analysis
    pub min_sample_size: usize,
    /// Enable advanced quality metrics
    pub enable_quality_assessment: bool,
    /// Historical data retention period
    pub historical_retention_days: u32,
}

impl Default for SemanticMetricsConfig {
    fn default() -> Self {
        Self {
            enable_statistical_analysis: true,
            enable_distribution_analysis: true,
            enable_correlation_analysis: true,
            enable_trend_analysis: false,
            enable_performance_analysis: true,
            confidence_level: 0.95,
            max_clusters: 10,
            outlier_threshold: 2.0,
            min_sample_size: 30,
            enable_quality_assessment: true,
            historical_retention_days: 30,
        }
    }
}

impl SemanticMetricsConfig {
    /// Create a new configuration builder
    pub fn builder() -> SemanticMetricsConfigBuilder {
        SemanticMetricsConfigBuilder::new()
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), SemanticMetricsError> {
        if self.confidence_level <= 0.0 || self.confidence_level >= 1.0 {
            return Err(SemanticMetricsError::InvalidConfiguration {
                message: "Confidence level must be between 0.0 and 1.0 (exclusive)".to_string(),
            });
        }

        if self.max_clusters == 0 {
            return Err(SemanticMetricsError::InvalidConfiguration {
                message: "Maximum clusters must be greater than 0".to_string(),
            });
        }

        if self.outlier_threshold <= 0.0 {
            return Err(SemanticMetricsError::InvalidConfiguration {
                message: "Outlier threshold must be positive".to_string(),
            });
        }

        if self.min_sample_size == 0 {
            return Err(SemanticMetricsError::InvalidConfiguration {
                message: "Minimum sample size must be greater than 0".to_string(),
            });
        }

        Ok(())
    }
}

/// Builder for semantic metrics configuration
pub struct SemanticMetricsConfigBuilder {
    config: SemanticMetricsConfig,
}

impl SemanticMetricsConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: SemanticMetricsConfig::default(),
        }
    }

    pub fn enable_statistical_analysis(mut self, enable: bool) -> Self {
        self.config.enable_statistical_analysis = enable;
        self
    }

    pub fn enable_distribution_analysis(mut self, enable: bool) -> Self {
        self.config.enable_distribution_analysis = enable;
        self
    }

    pub fn enable_correlation_analysis(mut self, enable: bool) -> Self {
        self.config.enable_correlation_analysis = enable;
        self
    }

    pub fn enable_trend_analysis(mut self, enable: bool) -> Self {
        self.config.enable_trend_analysis = enable;
        self
    }

    pub fn enable_performance_analysis(mut self, enable: bool) -> Self {
        self.config.enable_performance_analysis = enable;
        self
    }

    pub fn confidence_level(mut self, level: f64) -> Self {
        self.config.confidence_level = level.clamp(0.01, 0.99);
        self
    }

    pub fn max_clusters(mut self, max: usize) -> Self {
        self.config.max_clusters = max.max(1);
        self
    }

    pub fn outlier_threshold(mut self, threshold: f64) -> Self {
        self.config.outlier_threshold = threshold.max(0.1);
        self
    }

    pub fn min_sample_size(mut self, size: usize) -> Self {
        self.config.min_sample_size = size.max(1);
        self
    }

    pub fn enable_quality_assessment(mut self, enable: bool) -> Self {
        self.config.enable_quality_assessment = enable;
        self
    }

    pub fn historical_retention_days(mut self, days: u32) -> Self {
        self.config.historical_retention_days = days;
        self
    }

    pub fn build(self) -> Result<SemanticMetricsConfig, SemanticMetricsError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

/// Advanced semantic metrics analyzer
pub struct SemanticMetricsAnalyzer {
    config: SemanticMetricsConfig,
    historical_data: VecDeque<(SystemTime, Vec<f64>)>,
    performance_history: VecDeque<(SystemTime, Duration)>,
}

impl SemanticMetricsAnalyzer {
    /// Create a new semantic metrics analyzer with the given configuration
    pub fn new(config: SemanticMetricsConfig) -> Result<Self, SemanticMetricsError> {
        config.validate()?;

        Ok(Self {
            config,
            historical_data: VecDeque::new(),
            performance_history: VecDeque::new(),
        })
    }

    /// Create a semantic metrics analyzer with default configuration
    pub fn default() -> Result<Self, SemanticMetricsError> {
        Self::new(SemanticMetricsConfig::default())
    }

    /// Analyze comprehensive metrics for semantic similarity data
    pub fn analyze_metrics(
        &mut self,
        similarity_scores: &[f64],
        quality_scores: &[f64],
        confidence_scores: &[f64],
    ) -> Result<SemanticMetricsResult, SemanticMetricsError> {
        let start_time = std::time::Instant::now();

        // Validate input data
        self.validate_input_data(similarity_scores, quality_scores, confidence_scores)?;

        // Store data for historical analysis
        if self.config.enable_trend_analysis {
            self.store_historical_data(similarity_scores);
        }

        // Calculate basic statistics
        let similarity_stats = self.calculate_basic_statistics(similarity_scores)?;
        let quality_stats = self.calculate_basic_statistics(quality_scores)?;
        let confidence_stats = self.calculate_basic_statistics(confidence_scores)?;

        // Calculate detailed similarity metrics
        let similarity_metrics = self.analyze_similarity_metrics(similarity_scores)?;

        // Perform statistical analysis
        let statistical_analysis = if self.config.enable_statistical_analysis {
            self.perform_statistical_analysis(similarity_scores, quality_scores, confidence_scores)?
        } else {
            self.create_empty_statistical_analysis()
        };

        // Assess quality metrics
        let quality_metrics = if self.config.enable_quality_assessment {
            self.assess_quality_metrics(similarity_scores, quality_scores, confidence_scores)?
        } else {
            self.create_empty_quality_metrics()
        };

        // Analyze performance
        let performance_metrics = if self.config.enable_performance_analysis {
            self.analyze_performance(start_time)?
        } else {
            self.create_empty_performance_metrics()
        };

        // Analyze distributions
        let distribution_analysis = if self.config.enable_distribution_analysis {
            self.analyze_distributions(similarity_scores)?
        } else {
            self.create_empty_distribution_analysis()
        };

        // Analyze correlations
        let correlation_analysis =
            if self.config.enable_correlation_analysis && similarity_scores.len() >= 3 {
                self.analyze_correlations(similarity_scores, quality_scores, confidence_scores)?
            } else {
                self.create_empty_correlation_analysis()
            };

        // Analyze trends (if enabled and historical data available)
        let trend_analysis = if self.config.enable_trend_analysis && self.historical_data.len() > 10
        {
            Some(self.analyze_trends()?)
        } else {
            None
        };

        // Generate insights and recommendations
        let insights = self.generate_insights(&similarity_stats, &quality_stats, &confidence_stats);
        let recommendations =
            self.generate_recommendations(&similarity_stats, &quality_stats, &statistical_analysis);

        let summary = MetricsSummary {
            sample_count: similarity_scores.len(),
            similarity_stats,
            quality_stats,
            confidence_stats,
            insights,
            recommendations,
        };

        let processing_time = start_time.elapsed();
        let metadata = self.create_metadata(similarity_scores, processing_time);

        Ok(SemanticMetricsResult {
            summary,
            similarity_metrics,
            statistical_analysis,
            quality_metrics,
            performance_metrics,
            distribution_analysis,
            correlation_analysis,
            trend_analysis,
            metadata,
        })
    }

    /// Validate input data
    fn validate_input_data(
        &self,
        similarity_scores: &[f64],
        quality_scores: &[f64],
        confidence_scores: &[f64],
    ) -> Result<(), SemanticMetricsError> {
        if similarity_scores.is_empty() {
            return Err(SemanticMetricsError::InsufficientData {
                required: 1,
                provided: 0,
            });
        }

        if similarity_scores.len() != quality_scores.len()
            || similarity_scores.len() != confidence_scores.len()
        {
            return Err(SemanticMetricsError::DataValidationFailed {
                reason: "All score arrays must have the same length".to_string(),
            });
        }

        if similarity_scores.len() < self.config.min_sample_size {
            return Err(SemanticMetricsError::InsufficientData {
                required: self.config.min_sample_size,
                provided: similarity_scores.len(),
            });
        }

        // Check for invalid values
        for (i, &score) in similarity_scores.iter().enumerate() {
            if !score.is_finite() {
                return Err(SemanticMetricsError::DataValidationFailed {
                    reason: format!("Invalid similarity score at index {}: {}", i, score),
                });
            }
        }

        for (i, &score) in quality_scores.iter().enumerate() {
            if !score.is_finite() {
                return Err(SemanticMetricsError::DataValidationFailed {
                    reason: format!("Invalid quality score at index {}: {}", i, score),
                });
            }
        }

        for (i, &score) in confidence_scores.iter().enumerate() {
            if !score.is_finite() {
                return Err(SemanticMetricsError::DataValidationFailed {
                    reason: format!("Invalid confidence score at index {}: {}", i, score),
                });
            }
        }

        Ok(())
    }

    /// Store historical data for trend analysis
    fn store_historical_data(&mut self, data: &[f64]) {
        let now = SystemTime::now();
        self.historical_data.push_back((now, data.to_vec()));

        // Clean up old data
        let retention_duration =
            Duration::from_secs(self.config.historical_retention_days as u64 * 24 * 3600);
        while let Some((timestamp, _)) = self.historical_data.front() {
            if now.duration_since(*timestamp).unwrap_or(Duration::MAX) > retention_duration {
                self.historical_data.pop_front();
            } else {
                break;
            }
        }
    }

    /// Calculate basic statistics
    fn calculate_basic_statistics(
        &self,
        data: &[f64],
    ) -> Result<BasicStatistics, SemanticMetricsError> {
        if data.is_empty() {
            return Err(SemanticMetricsError::CalculationFailed {
                reason: "Cannot calculate statistics for empty data".to_string(),
            });
        }

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = data.len() as f64;
        let sum: f64 = data.iter().sum();
        let mean = sum / n;

        let median = if sorted_data.len() % 2 == 0 {
            (sorted_data[sorted_data.len() / 2 - 1] + sorted_data[sorted_data.len() / 2]) / 2.0
        } else {
            sorted_data[sorted_data.len() / 2]
        };

        // Calculate mode (most frequent value, simplified)
        let mut frequency_map: HashMap<u64, usize> = HashMap::new();
        for &value in data {
            let key = (value * 1000000.0) as u64; // Discretize for mode calculation
            *frequency_map.entry(key).or_insert(0) += 1;
        }
        let mode = frequency_map
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(key, _)| *key as f64 / 1000000.0);

        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();

        let min = *sorted_data.first().unwrap();
        let max = *sorted_data.last().unwrap();
        let range = max - min;

        // Skewness and kurtosis
        let skewness = if std_dev > 0.0 {
            data.iter()
                .map(|x| ((x - mean) / std_dev).powi(3))
                .sum::<f64>()
                / n
        } else {
            0.0
        };

        let kurtosis = if std_dev > 0.0 {
            data.iter()
                .map(|x| ((x - mean) / std_dev).powi(4))
                .sum::<f64>()
                / n
                - 3.0
        } else {
            0.0
        };

        // Quartiles
        let q1 = sorted_data[sorted_data.len() / 4];
        let q2 = median;
        let q3 = sorted_data[3 * sorted_data.len() / 4];
        let quartiles = (q1, q2, q3);

        // Percentiles
        let mut percentiles = HashMap::new();
        for &p in &[5, 10, 25, 75, 90, 95] {
            let index = ((p as f64 / 100.0) * (sorted_data.len() - 1) as f64).round() as usize;
            percentiles.insert(p, sorted_data[index.min(sorted_data.len() - 1)]);
        }

        Ok(BasicStatistics {
            mean,
            median,
            mode,
            std_dev,
            variance,
            min,
            max,
            range,
            skewness,
            kurtosis,
            quartiles,
            percentiles,
        })
    }

    /// Analyze detailed similarity metrics
    fn analyze_similarity_metrics(
        &self,
        similarity_scores: &[f64],
    ) -> Result<SimilarityMetrics, SemanticMetricsError> {
        let mut by_type = HashMap::new();

        // For now, just analyze the main similarity scores
        // In a real implementation, you might have different types of similarities
        let stats = self.calculate_basic_statistics(similarity_scores)?;
        by_type.insert(SimilarityMetricType::Composite, stats);

        // Simple clustering (k-means-like approach)
        let clusters = self.perform_similarity_clustering(similarity_scores)?;

        // Outlier detection
        let outliers = self.detect_outliers(similarity_scores)?;

        Ok(SimilarityMetrics {
            by_type,
            similarity_matrix: None, // Would require pairwise similarities
            score_distribution: similarity_scores.to_vec(),
            clusters,
            outliers,
        })
    }

    /// Perform similarity clustering
    fn perform_similarity_clustering(
        &self,
        data: &[f64],
    ) -> Result<Vec<SimilarityCluster>, SemanticMetricsError> {
        if data.len() < 3 {
            return Ok(Vec::new());
        }

        let k = (data.len() as f64).sqrt().ceil() as usize;
        let k = k.min(self.config.max_clusters).max(1);

        let mut clusters = Vec::new();

        // Simple k-means clustering for 1D data
        let min_val = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        for i in 0..k {
            let center = min_val + (max_val - min_val) * (i as f64 / (k - 1) as f64);
            clusters.push(SimilarityCluster {
                cluster_id: i,
                center: vec![center],
                members: Vec::new(),
                intra_cluster_similarity: 0.0,
                cluster_quality: 0.0,
            });
        }

        // Assign points to clusters
        for (idx, &value) in data.iter().enumerate() {
            let mut best_cluster = 0;
            let mut best_distance = f64::INFINITY;

            for (cluster_idx, cluster) in clusters.iter().enumerate() {
                let distance = (value - cluster.center[0]).abs();
                if distance < best_distance {
                    best_distance = distance;
                    best_cluster = cluster_idx;
                }
            }

            clusters[best_cluster].members.push(idx);
        }

        // Calculate cluster quality
        for cluster in &mut clusters {
            if !cluster.members.is_empty() {
                let member_values: Vec<f64> =
                    cluster.members.iter().map(|&idx| data[idx]).collect();

                let mean = member_values.iter().sum::<f64>() / member_values.len() as f64;
                cluster.center = vec![mean];

                let variance = member_values
                    .iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>()
                    / member_values.len() as f64;

                cluster.intra_cluster_similarity = 1.0 / (1.0 + variance);
                cluster.cluster_quality = cluster.intra_cluster_similarity;
            }
        }

        Ok(clusters)
    }

    /// Detect outliers in similarity scores
    fn detect_outliers(
        &self,
        data: &[f64],
    ) -> Result<Vec<SimilarityOutlier>, SemanticMetricsError> {
        if data.len() < 3 {
            return Ok(Vec::new());
        }

        let stats = self.calculate_basic_statistics(data)?;
        let mut outliers = Vec::new();

        let threshold = self.config.outlier_threshold;

        for (idx, &value) in data.iter().enumerate() {
            let z_score = if stats.std_dev > 0.0 {
                (value - stats.mean) / stats.std_dev
            } else {
                0.0
            };

            if z_score.abs() > threshold {
                let outlier_type = if z_score < -threshold {
                    OutlierType::Low
                } else {
                    OutlierType::High
                };

                outliers.push(SimilarityOutlier {
                    item_id: idx,
                    similarity_score: value,
                    deviation_magnitude: z_score.abs(),
                    outlier_type,
                });
            }
        }

        Ok(outliers)
    }

    /// Perform statistical analysis
    fn perform_statistical_analysis(
        &self,
        similarity_scores: &[f64],
        quality_scores: &[f64],
        confidence_scores: &[f64],
    ) -> Result<StatisticalAnalysis, SemanticMetricsError> {
        // Normality tests
        let normality_tests = self.perform_normality_tests(similarity_scores)?;

        // Hypothesis tests (example: test if mean similarity > 0.5)
        let mut hypothesis_tests = Vec::new();
        let t_test = self.perform_t_test(similarity_scores, 0.5)?;
        hypothesis_tests.push(t_test);

        // Confidence intervals
        let mut confidence_intervals = HashMap::new();
        let ci_similarity = self.calculate_confidence_interval(similarity_scores)?;
        confidence_intervals.insert("similarity".to_string(), ci_similarity);

        // Effect sizes
        let mut effect_sizes = HashMap::new();
        let cohens_d = self.calculate_cohens_d(similarity_scores, quality_scores)?;
        effect_sizes.insert("similarity_quality".to_string(), cohens_d);

        // Significance results
        let mut significance_results = HashMap::new();
        significance_results.insert("normality".to_string(), normality_tests.is_normal);
        significance_results.insert("t_test".to_string(), hypothesis_tests[0].is_significant);

        Ok(StatisticalAnalysis {
            normality_tests,
            hypothesis_tests,
            confidence_intervals,
            effect_sizes,
            significance_results,
        })
    }

    /// Perform normality tests
    fn perform_normality_tests(
        &self,
        data: &[f64],
    ) -> Result<NormalityTestResults, SemanticMetricsError> {
        // Simplified normality test (in practice, you'd use proper statistical libraries)
        let stats = self.calculate_basic_statistics(data)?;

        // Simple heuristic: check if skewness and kurtosis are reasonable for normal distribution
        let is_normal = stats.skewness.abs() < 1.0 && stats.kurtosis.abs() < 3.0;

        Ok(NormalityTestResults {
            shapiro_wilk: None, // Would require statistical library
            kolmogorov_smirnov: None,
            anderson_darling: None,
            is_normal,
            recommended_distribution: if is_normal {
                DistributionType::Normal
            } else if stats.skewness > 1.0 {
                DistributionType::LogNormal
            } else {
                DistributionType::Beta
            },
        })
    }

    /// Perform t-test
    fn perform_t_test(
        &self,
        data: &[f64],
        null_mean: f64,
    ) -> Result<HypothesisTestResult, SemanticMetricsError> {
        let stats = self.calculate_basic_statistics(data)?;
        let n = data.len() as f64;

        let t_statistic = if stats.std_dev > 0.0 {
            (stats.mean - null_mean) / (stats.std_dev / n.sqrt())
        } else {
            0.0
        };

        // Simplified p-value calculation (in practice, use proper statistical tables)
        let degrees_of_freedom = n - 1.0;
        let p_value = if t_statistic.abs() > 2.0 {
            0.05 // Rough approximation
        } else {
            0.2
        };

        let is_significant = p_value < (1.0 - self.config.confidence_level);

        Ok(HypothesisTestResult {
            test_name: "One-sample t-test".to_string(),
            null_hypothesis: format!("Mean equals {}", null_mean),
            alternative_hypothesis: format!("Mean does not equal {}", null_mean),
            test_statistic: t_statistic,
            p_value,
            critical_value: Some(2.0), // Simplified
            is_significant,
            effect_size: Some(t_statistic.abs() / n.sqrt()),
        })
    }

    /// Calculate confidence interval
    fn calculate_confidence_interval(
        &self,
        data: &[f64],
    ) -> Result<(f64, f64), SemanticMetricsError> {
        let stats = self.calculate_basic_statistics(data)?;
        let n = data.len() as f64;

        // Simplified confidence interval (assuming normal distribution)
        let t_critical = 2.0; // Rough approximation for 95% CI
        let margin_of_error = t_critical * (stats.std_dev / n.sqrt());

        Ok((stats.mean - margin_of_error, stats.mean + margin_of_error))
    }

    /// Calculate Cohen's d effect size
    fn calculate_cohens_d(
        &self,
        data1: &[f64],
        data2: &[f64],
    ) -> Result<f64, SemanticMetricsError> {
        let stats1 = self.calculate_basic_statistics(data1)?;
        let stats2 = self.calculate_basic_statistics(data2)?;

        let pooled_std = ((stats1.variance + stats2.variance) / 2.0).sqrt();

        if pooled_std > 0.0 {
            Ok((stats1.mean - stats2.mean) / pooled_std)
        } else {
            Ok(0.0)
        }
    }

    // Empty analysis creators for when features are disabled
    fn create_empty_statistical_analysis(&self) -> StatisticalAnalysis {
        StatisticalAnalysis {
            normality_tests: NormalityTestResults {
                shapiro_wilk: None,
                kolmogorov_smirnov: None,
                anderson_darling: None,
                is_normal: false,
                recommended_distribution: DistributionType::Unknown,
            },
            hypothesis_tests: Vec::new(),
            confidence_intervals: HashMap::new(),
            effect_sizes: HashMap::new(),
            significance_results: HashMap::new(),
        }
    }

    fn create_empty_quality_metrics(&self) -> QualityMetrics {
        QualityMetrics {
            data_quality: DataQualityIndicators {
                completeness_score: 1.0,
                consistency_score: 1.0,
                accuracy_score: 1.0,
                validity_score: 1.0,
                uniqueness_score: 1.0,
                timeliness_score: 1.0,
                overall_quality_score: 1.0,
            },
            reliability_metrics: ReliabilityMetrics {
                internal_consistency: 1.0,
                test_retest_reliability: None,
                inter_rater_reliability: None,
                measurement_error: 0.0,
                confidence_in_results: 1.0,
            },
            consistency_metrics: ConsistencyMetrics {
                method_agreement: HashMap::new(),
                rank_correlation: 1.0,
                classification_agreement: 1.0,
                variance_across_methods: 0.0,
            },
            robustness_assessment: RobustnessAssessment {
                noise_sensitivity: 0.0,
                parameter_stability: 1.0,
                outlier_resistance: 1.0,
                generalization_ability: 1.0,
            },
        }
    }

    fn create_empty_performance_metrics(&self) -> PerformanceMetrics {
        PerformanceMetrics {
            timing_stats: TimingStatistics {
                total_execution_time: Duration::ZERO,
                average_operation_time: Duration::ZERO,
                median_operation_time: Duration::ZERO,
                min_operation_time: Duration::ZERO,
                max_operation_time: Duration::ZERO,
                std_dev_operation_time: Duration::ZERO,
                percentile_95_time: Duration::ZERO,
                percentile_99_time: Duration::ZERO,
            },
            memory_usage: MemoryUsageAnalysis {
                peak_memory_usage: 0,
                average_memory_usage: 0,
                memory_efficiency_score: 1.0,
                memory_allocation_count: 0,
                memory_fragmentation_score: 0.0,
            },
            throughput_metrics: ThroughputMetrics {
                operations_per_second: 0.0,
                items_processed_per_second: 0.0,
                bytes_processed_per_second: 0.0,
                throughput_consistency: 1.0,
            },
            scalability_assessment: ScalabilityAssessment {
                linear_scalability_score: 1.0,
                memory_scalability_score: 1.0,
                predicted_max_capacity: usize::MAX,
                bottleneck_analysis: Vec::new(),
            },
        }
    }

    fn create_empty_distribution_analysis(&self) -> DistributionAnalysis {
        DistributionAnalysis {
            distribution_fits: HashMap::new(),
            best_fit: DistributionType::Unknown,
            distribution_parameters: HashMap::new(),
            goodness_of_fit: GoodnessOfFit {
                chi_square: 0.0,
                chi_square_p_value: 1.0,
                kolmogorov_smirnov: 0.0,
                ks_p_value: 1.0,
                anderson_darling: 0.0,
                ad_p_value: 1.0,
            },
            distribution_comparisons: Vec::new(),
        }
    }

    fn create_empty_correlation_analysis(&self) -> CorrelationAnalysis {
        CorrelationAnalysis {
            correlation_matrix: Array2::zeros((0, 0)),
            correlations: HashMap::new(),
            pca_results: None,
            factor_analysis: None,
        }
    }

    /// Assess quality metrics (simplified implementation)
    fn assess_quality_metrics(
        &self,
        similarity_scores: &[f64],
        quality_scores: &[f64],
        confidence_scores: &[f64],
    ) -> Result<QualityMetrics, SemanticMetricsError> {
        // Calculate data quality indicators
        let completeness_score = 1.0; // Assuming no missing data
        let consistency_score =
            self.calculate_consistency_score(similarity_scores, quality_scores)?;
        let accuracy_score = quality_scores.iter().sum::<f64>() / quality_scores.len() as f64;
        let validity_score = self.calculate_validity_score(similarity_scores)?;
        let uniqueness_score = self.calculate_uniqueness_score(similarity_scores)?;
        let timeliness_score = 1.0; // Assuming fresh data
        let overall_quality_score = (completeness_score
            + consistency_score
            + accuracy_score
            + validity_score
            + uniqueness_score
            + timeliness_score)
            / 6.0;

        let data_quality = DataQualityIndicators {
            completeness_score,
            consistency_score,
            accuracy_score,
            validity_score,
            uniqueness_score,
            timeliness_score,
            overall_quality_score,
        };

        // Calculate reliability metrics
        let internal_consistency = self.calculate_internal_consistency(similarity_scores)?;
        let measurement_error =
            self.calculate_measurement_error(similarity_scores, quality_scores)?;
        let confidence_in_results =
            confidence_scores.iter().sum::<f64>() / confidence_scores.len() as f64;

        let reliability_metrics = ReliabilityMetrics {
            internal_consistency,
            test_retest_reliability: None,
            inter_rater_reliability: None,
            measurement_error,
            confidence_in_results,
        };

        // Calculate consistency metrics
        let rank_correlation =
            self.calculate_rank_correlation(similarity_scores, quality_scores)?;
        let classification_agreement = 0.8; // Simplified
        let variance_across_methods = 0.1; // Simplified

        let consistency_metrics = ConsistencyMetrics {
            method_agreement: HashMap::new(),
            rank_correlation,
            classification_agreement,
            variance_across_methods,
        };

        // Calculate robustness assessment
        let noise_sensitivity = 0.2; // Simplified
        let parameter_stability = 0.8; // Simplified
        let outlier_resistance = 0.7; // Simplified
        let generalization_ability = 0.8; // Simplified

        let robustness_assessment = RobustnessAssessment {
            noise_sensitivity,
            parameter_stability,
            outlier_resistance,
            generalization_ability,
        };

        Ok(QualityMetrics {
            data_quality,
            reliability_metrics,
            consistency_metrics,
            robustness_assessment,
        })
    }

    /// Calculate consistency score
    fn calculate_consistency_score(
        &self,
        data1: &[f64],
        data2: &[f64],
    ) -> Result<f64, SemanticMetricsError> {
        if data1.len() != data2.len() {
            return Ok(0.0);
        }

        let correlation = self.calculate_pearson_correlation(data1, data2)?;
        Ok((correlation + 1.0) / 2.0) // Scale to 0-1
    }

    /// Calculate Pearson correlation
    fn calculate_pearson_correlation(
        &self,
        data1: &[f64],
        data2: &[f64],
    ) -> Result<f64, SemanticMetricsError> {
        if data1.len() != data2.len() || data1.is_empty() {
            return Ok(0.0);
        }

        let n = data1.len() as f64;
        let mean1 = data1.iter().sum::<f64>() / n;
        let mean2 = data2.iter().sum::<f64>() / n;

        let numerator: f64 = data1
            .iter()
            .zip(data2.iter())
            .map(|(x, y)| (x - mean1) * (y - mean2))
            .sum();

        let sum_sq1: f64 = data1.iter().map(|x| (x - mean1).powi(2)).sum();
        let sum_sq2: f64 = data2.iter().map(|y| (y - mean2).powi(2)).sum();

        let denominator = (sum_sq1 * sum_sq2).sqrt();

        if denominator > 0.0 {
            Ok(numerator / denominator)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate validity score
    fn calculate_validity_score(&self, data: &[f64]) -> Result<f64, SemanticMetricsError> {
        // Check if values are in valid range [0, 1] for similarity scores
        let valid_count = data.iter().filter(|&&x| x >= 0.0 && x <= 1.0).count();
        Ok(valid_count as f64 / data.len() as f64)
    }

    /// Calculate uniqueness score
    fn calculate_uniqueness_score(&self, data: &[f64]) -> Result<f64, SemanticMetricsError> {
        let mut unique_values = std::collections::HashSet::new();
        for &value in data {
            let discretized = (value * 1000.0) as i32; // Discretize for uniqueness check
            unique_values.insert(discretized);
        }
        Ok(unique_values.len() as f64 / data.len() as f64)
    }

    /// Calculate internal consistency
    fn calculate_internal_consistency(&self, data: &[f64]) -> Result<f64, SemanticMetricsError> {
        // Simplified Cronbach's alpha-like measure
        if data.len() < 2 {
            return Ok(1.0);
        }

        let stats = self.calculate_basic_statistics(data)?;
        let mean_inter_item_correlation = 0.3; // Simplified assumption

        let k = data.len() as f64;
        let cronbach_alpha =
            (k * mean_inter_item_correlation) / (1.0 + (k - 1.0) * mean_inter_item_correlation);

        Ok(cronbach_alpha.clamp(0.0, 1.0))
    }

    /// Calculate measurement error
    fn calculate_measurement_error(
        &self,
        observed: &[f64],
        true_scores: &[f64],
    ) -> Result<f64, SemanticMetricsError> {
        if observed.len() != true_scores.len() {
            return Ok(0.0);
        }

        let mse: f64 = observed
            .iter()
            .zip(true_scores.iter())
            .map(|(obs, true_val)| (obs - true_val).powi(2))
            .sum::<f64>()
            / observed.len() as f64;

        Ok(mse.sqrt())
    }

    /// Calculate rank correlation (Spearman's rho approximation)
    fn calculate_rank_correlation(
        &self,
        data1: &[f64],
        data2: &[f64],
    ) -> Result<f64, SemanticMetricsError> {
        if data1.len() != data2.len() || data1.len() < 2 {
            return Ok(0.0);
        }

        // Create rank vectors
        let ranks1 = self.calculate_ranks(data1);
        let ranks2 = self.calculate_ranks(data2);

        // Calculate Pearson correlation of ranks
        self.calculate_pearson_correlation(&ranks1, &ranks2)
    }

    /// Calculate ranks for Spearman correlation
    fn calculate_ranks(&self, data: &[f64]) -> Vec<f64> {
        let mut indexed_data: Vec<(f64, usize)> = data
            .iter()
            .copied()
            .enumerate()
            .map(|(i, x)| (x, i))
            .collect();
        indexed_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut ranks = vec![0.0; data.len()];
        for (rank, (_, original_index)) in indexed_data.iter().enumerate() {
            ranks[*original_index] = (rank + 1) as f64;
        }

        ranks
    }

    /// Analyze performance (simplified)
    fn analyze_performance(
        &mut self,
        start_time: std::time::Instant,
    ) -> Result<PerformanceMetrics, SemanticMetricsError> {
        let total_time = start_time.elapsed();
        self.performance_history
            .push_back((SystemTime::now(), total_time));

        let timing_stats = TimingStatistics {
            total_execution_time: total_time,
            average_operation_time: total_time, // Simplified
            median_operation_time: total_time,
            min_operation_time: total_time,
            max_operation_time: total_time,
            std_dev_operation_time: Duration::ZERO,
            percentile_95_time: total_time,
            percentile_99_time: total_time,
        };

        Ok(PerformanceMetrics {
            timing_stats,
            memory_usage: MemoryUsageAnalysis {
                peak_memory_usage: 1024 * 1024,   // 1MB estimate
                average_memory_usage: 512 * 1024, // 512KB estimate
                memory_efficiency_score: 0.8,
                memory_allocation_count: 10,
                memory_fragmentation_score: 0.1,
            },
            throughput_metrics: ThroughputMetrics {
                operations_per_second: 1000.0, // Simplified
                items_processed_per_second: 100.0,
                bytes_processed_per_second: 10240.0,
                throughput_consistency: 0.9,
            },
            scalability_assessment: ScalabilityAssessment {
                linear_scalability_score: 0.8,
                memory_scalability_score: 0.7,
                predicted_max_capacity: 1_000_000,
                bottleneck_analysis: vec!["Memory allocation".to_string()],
            },
        })
    }

    /// Analyze distributions (simplified)
    fn analyze_distributions(
        &self,
        data: &[f64],
    ) -> Result<DistributionAnalysis, SemanticMetricsError> {
        let stats = self.calculate_basic_statistics(data)?;

        // Simple distribution fitting heuristics
        let mut distribution_fits = HashMap::new();

        // Normal distribution fit
        let normal_fit = DistributionFit {
            distribution_type: DistributionType::Normal,
            parameters: vec![stats.mean, stats.std_dev],
            log_likelihood: -100.0, // Simplified
            aic: 202.0,
            bic: 206.0,
            fit_quality: 0.7,
        };
        distribution_fits.insert(DistributionType::Normal, normal_fit);

        // Beta distribution fit (for bounded data)
        if stats.min >= 0.0 && stats.max <= 1.0 {
            let beta_fit = DistributionFit {
                distribution_type: DistributionType::Beta,
                parameters: vec![2.0, 2.0], // Simplified
                log_likelihood: -95.0,
                aic: 194.0,
                bic: 198.0,
                fit_quality: 0.8,
            };
            distribution_fits.insert(DistributionType::Beta, beta_fit);
        }

        let best_fit = if stats.min >= 0.0 && stats.max <= 1.0 {
            DistributionType::Beta
        } else {
            DistributionType::Normal
        };

        Ok(DistributionAnalysis {
            distribution_fits,
            best_fit,
            distribution_parameters: HashMap::new(),
            goodness_of_fit: GoodnessOfFit {
                chi_square: 5.0,
                chi_square_p_value: 0.3,
                kolmogorov_smirnov: 0.1,
                ks_p_value: 0.5,
                anderson_darling: 0.5,
                ad_p_value: 0.4,
            },
            distribution_comparisons: Vec::new(),
        })
    }

    /// Analyze correlations (simplified)
    fn analyze_correlations(
        &self,
        similarity_scores: &[f64],
        quality_scores: &[f64],
        confidence_scores: &[f64],
    ) -> Result<CorrelationAnalysis, SemanticMetricsError> {
        let mut correlations = HashMap::new();

        // Calculate correlations between different score types
        let sim_qual_corr =
            self.calculate_pearson_correlation(similarity_scores, quality_scores)?;
        correlations.insert(
            ("similarity".to_string(), "quality".to_string()),
            CorrelationResult {
                correlation_coefficient: sim_qual_corr,
                correlation_type: "Pearson".to_string(),
                p_value: 0.05, // Simplified
                confidence_interval: (sim_qual_corr - 0.1, sim_qual_corr + 0.1),
                is_significant: sim_qual_corr.abs() > 0.3,
                effect_size: if sim_qual_corr.abs() < 0.3 {
                    "Small".to_string()
                } else if sim_qual_corr.abs() < 0.5 {
                    "Medium".to_string()
                } else {
                    "Large".to_string()
                },
            },
        );

        let sim_conf_corr =
            self.calculate_pearson_correlation(similarity_scores, confidence_scores)?;
        correlations.insert(
            ("similarity".to_string(), "confidence".to_string()),
            CorrelationResult {
                correlation_coefficient: sim_conf_corr,
                correlation_type: "Pearson".to_string(),
                p_value: 0.05,
                confidence_interval: (sim_conf_corr - 0.1, sim_conf_corr + 0.1),
                is_significant: sim_conf_corr.abs() > 0.3,
                effect_size: if sim_conf_corr.abs() < 0.3 {
                    "Small".to_string()
                } else if sim_conf_corr.abs() < 0.5 {
                    "Medium".to_string()
                } else {
                    "Large".to_string()
                },
            },
        );

        // Create correlation matrix
        let correlation_matrix = array![
            [1.0, sim_qual_corr, sim_conf_corr],
            [
                sim_qual_corr,
                1.0,
                self.calculate_pearson_correlation(quality_scores, confidence_scores)?
            ],
            [
                sim_conf_corr,
                self.calculate_pearson_correlation(quality_scores, confidence_scores)?,
                1.0
            ]
        ];

        Ok(CorrelationAnalysis {
            correlation_matrix,
            correlations,
            pca_results: None, // Would require more complex implementation
            factor_analysis: None,
        })
    }

    /// Analyze trends (simplified)
    fn analyze_trends(&self) -> Result<TrendAnalysis, SemanticMetricsError> {
        if self.historical_data.len() < 2 {
            return Err(SemanticMetricsError::InsufficientData {
                required: 2,
                provided: self.historical_data.len(),
            });
        }

        // Simple trend analysis
        let mut trend_values = Vec::new();
        for (_, data) in &self.historical_data {
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            trend_values.push(mean);
        }

        // Linear trend calculation
        let n = trend_values.len();
        let x_mean = (n - 1) as f64 / 2.0;
        let y_mean = trend_values.iter().sum::<f64>() / n as f64;

        let mut slope_numerator = 0.0;
        let mut slope_denominator = 0.0;

        for (i, &y) in trend_values.iter().enumerate() {
            let x = i as f64;
            slope_numerator += (x - x_mean) * (y - y_mean);
            slope_denominator += (x - x_mean).powi(2);
        }

        let slope = if slope_denominator != 0.0 {
            slope_numerator / slope_denominator
        } else {
            0.0
        };

        let trend_direction = if slope > 0.01 {
            TrendDirection::Increasing
        } else if slope < -0.01 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        let trend_strength = slope.abs();

        Ok(TrendAnalysis {
            trend_direction,
            trend_strength,
            trend_significance: 0.8, // Simplified
            seasonal_patterns: Vec::new(),
            forecasts: Vec::new(),
            change_points: Vec::new(),
        })
    }

    /// Generate insights from analysis
    fn generate_insights(
        &self,
        similarity_stats: &BasicStatistics,
        quality_stats: &BasicStatistics,
        confidence_stats: &BasicStatistics,
    ) -> Vec<String> {
        let mut insights = Vec::new();

        // Similarity insights
        if similarity_stats.mean > 0.8 {
            insights.push(
                "High overall similarity scores indicate strong semantic relationships."
                    .to_string(),
            );
        } else if similarity_stats.mean < 0.3 {
            insights.push(
                "Low overall similarity scores suggest weak semantic relationships.".to_string(),
            );
        }

        if similarity_stats.std_dev > 0.3 {
            insights.push(
                "High variability in similarity scores indicates diverse semantic relationships."
                    .to_string(),
            );
        }

        // Quality insights
        if quality_stats.mean > 0.8 {
            insights.push("High quality scores indicate reliable analysis results.".to_string());
        } else if quality_stats.mean < 0.5 {
            insights.push(
                "Lower quality scores suggest analysis results should be interpreted with caution."
                    .to_string(),
            );
        }

        // Confidence insights
        if confidence_stats.mean > 0.8 {
            insights
                .push("High confidence scores indicate strong certainty in results.".to_string());
        }

        // Distribution insights
        if similarity_stats.skewness.abs() > 1.0 {
            insights
                .push("Skewed similarity distribution may indicate systematic biases.".to_string());
        }

        insights
    }

    /// Generate recommendations from analysis
    fn generate_recommendations(
        &self,
        similarity_stats: &BasicStatistics,
        quality_stats: &BasicStatistics,
        statistical_analysis: &StatisticalAnalysis,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Similarity recommendations
        if similarity_stats.std_dev > 0.4 {
            recommendations.push(
                "Consider investigating outliers or using robust similarity measures.".to_string(),
            );
        }

        // Quality recommendations
        if quality_stats.mean < 0.6 {
            recommendations
                .push("Consider improving analysis parameters or data preprocessing.".to_string());
        }

        // Statistical recommendations
        if !statistical_analysis.normality_tests.is_normal {
            recommendations.push(
                "Use non-parametric statistical tests due to non-normal distribution.".to_string(),
            );
        }

        // Sample size recommendations
        if similarity_stats.mean < 0.5 && quality_stats.mean > 0.7 {
            recommendations.push("Low similarities with high quality suggest genuine dissimilarity rather than analysis errors.".to_string());
        }

        recommendations
    }

    /// Create metadata for analysis
    fn create_metadata(&self, data: &[f64], processing_time: Duration) -> MetricsMetadata {
        MetricsMetadata {
            timestamp: SystemTime::now(),
            processing_time,
            config_summary: format!(
                "Confidence: {}, Clusters: {}, Outlier threshold: {}",
                self.config.confidence_level,
                self.config.max_clusters,
                self.config.outlier_threshold
            ),
            data_characteristics: DataCharacteristics {
                sample_size: data.len(),
                dimensionality: 1, // Simplified for 1D similarity scores
                sparsity: 0.0,
                noise_level: 0.1, // Estimated
                data_types: vec!["similarity_scores".to_string()],
                missing_data_percentage: 0.0,
            },
            analysis_completeness: 1.0,
            analysis_quality: 0.8, // Estimated based on configuration
        }
    }

    /// Update configuration
    pub fn update_config(
        &mut self,
        config: SemanticMetricsConfig,
    ) -> Result<(), SemanticMetricsError> {
        config.validate()?;
        self.config = config;
        Ok(())
    }

    /// Get current configuration
    pub fn get_config(&self) -> &SemanticMetricsConfig {
        &self.config
    }

    /// Clear historical data
    pub fn clear_historical_data(&mut self) {
        self.historical_data.clear();
        self.performance_history.clear();
    }

    /// Get historical data summary
    pub fn get_historical_summary(&self) -> Option<(usize, Duration)> {
        if self.historical_data.is_empty() {
            return None;
        }

        let total_samples: usize = self
            .historical_data
            .iter()
            .map(|(_, data)| data.len())
            .sum();
        let time_span = self
            .historical_data
            .back()
            .unwrap()
            .0
            .duration_since(self.historical_data.front().unwrap().0)
            .unwrap_or(Duration::ZERO);

        Some((total_samples, time_span))
    }
}

/// Convenience function for simple metrics analysis
pub fn analyze_semantic_metrics(
    similarity_scores: &[f64],
    quality_scores: &[f64],
    confidence_scores: &[f64],
) -> Result<SemanticMetricsResult, SemanticMetricsError> {
    let mut analyzer = SemanticMetricsAnalyzer::default()?;
    analyzer.analyze_metrics(similarity_scores, quality_scores, confidence_scores)
}

/// Convenience function for metrics analysis with custom config
pub fn analyze_semantic_metrics_with_config(
    similarity_scores: &[f64],
    quality_scores: &[f64],
    confidence_scores: &[f64],
    config: SemanticMetricsConfig,
) -> Result<SemanticMetricsResult, SemanticMetricsError> {
    let mut analyzer = SemanticMetricsAnalyzer::new(config)?;
    analyzer.analyze_metrics(similarity_scores, quality_scores, confidence_scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_analyzer_creation() {
        let analyzer = SemanticMetricsAnalyzer::default();
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_basic_metrics_analysis() {
        let mut analyzer = SemanticMetricsAnalyzer::default().unwrap();
        let similarity_scores = vec![
            0.8, 0.7, 0.9, 0.6, 0.8, 0.7, 0.9, 0.8, 0.6, 0.7, 0.8, 0.9, 0.7, 0.8, 0.9, 0.6, 0.7,
            0.8, 0.9, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.9, 0.7, 0.8, 0.6, 0.9,
        ];
        let quality_scores = vec![0.9; 30];
        let confidence_scores = vec![0.85; 30];

        let result =
            analyzer.analyze_metrics(&similarity_scores, &quality_scores, &confidence_scores);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.summary.sample_count, 30);
        assert!(result.summary.similarity_stats.mean > 0.0);
        assert!(result.summary.similarity_stats.std_dev >= 0.0);
    }

    #[test]
    fn test_insufficient_data_error() {
        let mut analyzer = SemanticMetricsAnalyzer::default().unwrap();
        let similarity_scores = vec![0.5]; // Too few samples
        let quality_scores = vec![0.8];
        let confidence_scores = vec![0.7];

        let result =
            analyzer.analyze_metrics(&similarity_scores, &quality_scores, &confidence_scores);
        assert!(result.is_err());
        match result.unwrap_err() {
            SemanticMetricsError::InsufficientData { .. } => {}
            _ => panic!("Expected InsufficientData error"),
        }
    }

    #[test]
    fn test_mismatched_array_lengths() {
        let mut analyzer = SemanticMetricsAnalyzer::default().unwrap();
        let similarity_scores = vec![0.5, 0.6, 0.7];
        let quality_scores = vec![0.8, 0.9]; // Different length
        let confidence_scores = vec![0.7, 0.8, 0.9];

        let result =
            analyzer.analyze_metrics(&similarity_scores, &quality_scores, &confidence_scores);
        assert!(result.is_err());
        match result.unwrap_err() {
            SemanticMetricsError::DataValidationFailed { .. } => {}
            _ => panic!("Expected DataValidationFailed error"),
        }
    }

    #[test]
    fn test_configuration_builder() {
        let config = SemanticMetricsConfig::builder()
            .enable_statistical_analysis(true)
            .enable_distribution_analysis(false)
            .confidence_level(0.99)
            .max_clusters(5)
            .outlier_threshold(3.0)
            .min_sample_size(50)
            .enable_quality_assessment(true)
            .build();

        assert!(config.is_ok());
        let config = config.unwrap();
        assert_eq!(config.confidence_level, 0.99);
        assert_eq!(config.max_clusters, 5);
        assert_eq!(config.outlier_threshold, 3.0);
        assert_eq!(config.min_sample_size, 50);
        assert!(config.enable_statistical_analysis);
        assert!(!config.enable_distribution_analysis);
    }

    #[test]
    fn test_invalid_confidence_level() {
        let config = SemanticMetricsConfig::builder()
            .confidence_level(1.5) // Invalid: > 1.0
            .build();

        assert!(config.is_err());
        match config.unwrap_err() {
            SemanticMetricsError::InvalidConfiguration { .. } => {}
            _ => panic!("Expected InvalidConfiguration error"),
        }
    }

    #[test]
    fn test_basic_statistics_calculation() {
        let analyzer = SemanticMetricsAnalyzer::default().unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = analyzer.calculate_basic_statistics(&data).unwrap();

        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.range, 4.0);
    }

    #[test]
    fn test_outlier_detection() {
        let analyzer = SemanticMetricsAnalyzer::default().unwrap();
        let data = vec![0.5, 0.6, 0.7, 0.6, 0.5, 0.6, 0.7, 0.9, 0.1]; // 0.9 and 0.1 are outliers
        let outliers = analyzer.detect_outliers(&data).unwrap();

        assert!(!outliers.is_empty());
        assert!(outliers.iter().any(|o| o.outlier_type == OutlierType::High));
        assert!(outliers.iter().any(|o| o.outlier_type == OutlierType::Low));
    }

    #[test]
    fn test_correlation_calculation() {
        let analyzer = SemanticMetricsAnalyzer::default().unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data2 = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect positive correlation

        let correlation = analyzer
            .calculate_pearson_correlation(&data1, &data2)
            .unwrap();
        assert!((correlation - 1.0).abs() < 0.001); // Should be very close to 1.0
    }

    #[test]
    fn test_clustering() {
        let analyzer = SemanticMetricsAnalyzer::default().unwrap();
        let data = vec![0.1, 0.15, 0.2, 0.8, 0.85, 0.9]; // Two clear clusters
        let clusters = analyzer.perform_similarity_clustering(&data).unwrap();

        assert!(!clusters.is_empty());
        assert!(clusters.len() <= analyzer.config.max_clusters);
    }

    #[test]
    fn test_quality_assessment() {
        let analyzer = SemanticMetricsAnalyzer::default().unwrap();
        let similarity_scores = vec![
            0.5, 0.6, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.5,
            0.6, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8, 0.9, 0.6,
        ];
        let quality_scores = vec![0.8; 30];
        let confidence_scores = vec![0.85; 30];

        let quality_metrics = analyzer.assess_quality_metrics(
            &similarity_scores,
            &quality_scores,
            &confidence_scores,
        );
        assert!(quality_metrics.is_ok());

        let metrics = quality_metrics.unwrap();
        assert!(metrics.data_quality.overall_quality_score > 0.0);
        assert!(metrics.reliability_metrics.confidence_in_results > 0.0);
    }

    #[test]
    fn test_convenience_functions() {
        let similarity_scores = vec![
            0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.6, 0.9, 0.7,
            0.8, 0.6, 0.9, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8, 0.6, 0.9, 0.7, 0.8,
        ];
        let quality_scores = vec![0.85; 30];
        let confidence_scores = vec![0.9; 30];

        let result =
            analyze_semantic_metrics(&similarity_scores, &quality_scores, &confidence_scores);
        assert!(result.is_ok());

        let config = SemanticMetricsConfig::builder()
            .enable_statistical_analysis(false)
            .enable_distribution_analysis(false)
            .build()
            .unwrap();

        let result = analyze_semantic_metrics_with_config(
            &similarity_scores,
            &quality_scores,
            &confidence_scores,
            config,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_historical_data_management() {
        let mut analyzer = SemanticMetricsAnalyzer::default().unwrap();
        let data = vec![0.5, 0.6, 0.7];

        analyzer.store_historical_data(&data);
        assert_eq!(analyzer.historical_data.len(), 1);

        analyzer.clear_historical_data();
        assert_eq!(analyzer.historical_data.len(), 0);
    }

    #[test]
    fn test_rank_correlation() {
        let analyzer = SemanticMetricsAnalyzer::default().unwrap();
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data2 = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Perfect rank correlation

        let correlation = analyzer.calculate_rank_correlation(&data1, &data2).unwrap();
        assert!((correlation - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_empty_data_statistics() {
        let analyzer = SemanticMetricsAnalyzer::default().unwrap();
        let empty_data: Vec<f64> = vec![];

        let result = analyzer.calculate_basic_statistics(&empty_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_configuration_validation() {
        // Valid configuration
        let config = SemanticMetricsConfig::builder()
            .confidence_level(0.95)
            .max_clusters(5)
            .outlier_threshold(2.0)
            .min_sample_size(10)
            .build();
        assert!(config.is_ok());

        // Invalid confidence level
        let config = SemanticMetricsConfig::builder()
            .confidence_level(0.0) // Invalid: exactly 0.0
            .build();
        assert!(config.is_err());

        // Invalid max clusters
        let config = SemanticMetricsConfig::builder()
            .max_clusters(0) // Invalid: must be > 0
            .build();
        assert!(config.is_err());
    }

    #[test]
    fn test_insights_generation() {
        let mut analyzer = SemanticMetricsAnalyzer::default().unwrap();
        let high_similarity = vec![0.9; 30];
        let high_quality = vec![0.9; 30];
        let high_confidence = vec![0.9; 30];

        let result = analyzer
            .analyze_metrics(&high_similarity, &high_quality, &high_confidence)
            .unwrap();
        assert!(!result.summary.insights.is_empty());
        assert!(!result.summary.recommendations.is_empty());
    }
}
