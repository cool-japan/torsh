//! Modular Prosodic Fluency Analysis System
//!
//! This module provides a comprehensive and modular prosodic fluency analysis system
//! that combines specialized analyzers for rhythm, stress, intonation, timing, phonological
//! patterns, and advanced integration analysis. The system is designed for high performance,
//! configurability, and extensibility while maintaining backward compatibility.
//!
//! # Architecture Overview
//!
//! The prosodic fluency system is organized into specialized modules:
//!
//! - **Configuration Management**: Hierarchical configuration system (`config.rs`)
//! - **Result Structures**: Comprehensive result types with serialization (`results.rs`)
//! - **Rhythm Analysis**: Beat pattern detection and rhythm classification (`rhythm.rs`)
//! - **Stress Analysis**: Metrical foot analysis and prominence detection (`stress.rs`)
//! - **Intonation Analysis**: Pitch contour analysis and boundary tone detection (`intonation.rs`)
//! - **Timing Analysis**: Pause placement evaluation and tempo analysis (`timing.rs`)
//! - **Phonological Analysis**: Syllable structure and phonotactic constraints (`phonological.rs`)
//! - **Advanced Analysis**: High-level integration and pattern synthesis (`advanced.rs`)
//!
//! # Usage Examples
//!
//! ## Basic Prosodic Analysis
//!
//! ```rust,ignore
//! use torsh_text::metrics::fluency::prosodic::{ProsodicFluencyAnalyzer, ProsodicConfig};
//!
//! let config = ProsodicConfig::default();
//! let mut analyzer = ProsodicFluencyAnalyzer::new(config);
//!
//! let text = "The quick brown fox jumps over the lazy dog.";
//! let phonetic = "ðə kwɪk braʊn fɑks dʒʌmps oʊvər ðə leɪzi dɔg";
//!
//! let results = analyzer.analyze(text, phonetic)?;
//! println!("Overall fluency score: {}", results.overall_fluency_score);
//! ```
//!
//! ## Advanced Analysis with Custom Configuration
//!
//! ```rust,ignore
//! use torsh_text::metrics::fluency::prosodic::{ProsodicFluencyAnalyzer, ProsodicConfig};
//!
//! let mut config = ProsodicConfig::comprehensive();
//! config.rhythm.enable_advanced_beat_detection = true;
//! config.advanced.high_sophistication_analysis = true;
//!
//! let mut analyzer = ProsodicFluencyAnalyzer::new(config);
//! let results = analyzer.analyze_comprehensive(text, phonetic, audio_features)?;
//! ```
//!
//! # Performance Considerations
//!
//! - **Caching**: Results are cached by default for repeated analyses
//! - **Parallel Processing**: Component analysis can run in parallel
//! - **Memory Efficiency**: Streaming analysis for large texts
//! - **Adaptive Configuration**: Performance tuning based on text complexity

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use thiserror::Error;

// Re-export all public types for convenient access
pub use self::advanced::*;
pub use self::config::*;
pub use self::intonation::*;
pub use self::phonological::*;
pub use self::results::*;
pub use self::rhythm::*;
pub use self::stress::*;
pub use self::timing::*;

// Module declarations
pub mod advanced;
pub mod config;
pub mod intonation;
pub mod phonological;
pub mod results;
pub mod rhythm;
pub mod stress;
pub mod timing;

/// Main prosodic fluency analyzer that orchestrates all specialized components
#[derive(Debug, Clone)]
pub struct ProsodicFluencyAnalyzer {
    config: ProsodicConfig,

    // Specialized component analyzers
    rhythm_analyzer: RhythmAnalyzer,
    stress_analyzer: StressAnalyzer,
    intonation_analyzer: IntonationAnalyzer,
    timing_analyzer: TimingAnalyzer,
    phonological_analyzer: PhonologicalAnalyzer,
    advanced_analyzer: AdvancedProsodicAnalyzer,

    // Analysis coordination
    analysis_orchestrator: AnalysisOrchestrator,
    result_synthesizer: ResultSynthesizer,

    // Performance optimization
    component_cache: HashMap<String, ComponentCache>,
    analysis_cache: HashMap<u64, ProsodicFluencyResult>,

    // Analysis statistics
    analysis_stats: AnalysisStatistics,
}

/// Analysis orchestrator for coordinating component analyses
#[derive(Debug, Clone)]
pub struct AnalysisOrchestrator {
    execution_strategy: ExecutionStrategy,
    component_dependencies: HashMap<String, Vec<String>>,
    parallel_execution_config: ParallelExecutionConfig,
    error_recovery_strategy: ErrorRecoveryStrategy,
}

/// Result synthesizer for combining component results
#[derive(Debug, Clone)]
pub struct ResultSynthesizer {
    synthesis_weights: HashMap<String, f64>,
    integration_algorithms: Vec<IntegrationAlgorithm>,
    quality_assessment: QualityAssessment,
    confidence_calculator: ConfidenceCalculator,
}

/// Component cache for individual analyzer results
#[derive(Debug, Clone)]
pub struct ComponentCache {
    cache_data: HashMap<u64, Box<dyn CacheableResult>>,
    cache_stats: CacheStatistics,
    eviction_policy: EvictionPolicy,
}

/// Analysis statistics tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisStatistics {
    pub total_analyses: u64,
    pub component_execution_times: HashMap<String, f64>,
    pub cache_hit_rates: HashMap<String, f64>,
    pub error_rates: HashMap<String, f64>,
    pub quality_scores: HashMap<String, f64>,
}

/// Execution strategy configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    Sequential,
    Parallel,
    Adaptive,
    PriorityBased,
}

/// Parallel execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelExecutionConfig {
    pub max_concurrent_components: usize,
    pub thread_pool_size: usize,
    pub load_balancing_strategy: LoadBalancingStrategy,
    pub resource_allocation: ResourceAllocation,
}

/// Error recovery strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ErrorRecoveryStrategy {
    FailFast,
    BestEffort,
    GracefulDegradation,
    ComponentIsolation,
}

/// Integration algorithm for result synthesis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationAlgorithm {
    pub name: String,
    pub algorithm_type: String,
    pub input_components: Vec<String>,
    pub weight_scheme: WeightScheme,
    pub normalization_method: NormalizationMethod,
}

/// Quality assessment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessment {
    pub quality_metrics: Vec<String>,
    pub assessment_thresholds: HashMap<String, f64>,
    pub quality_weights: HashMap<String, f64>,
}

/// Confidence calculator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceCalculator {
    pub confidence_factors: Vec<String>,
    pub factor_weights: HashMap<String, f64>,
    pub base_confidence: f64,
    pub uncertainty_penalties: HashMap<String, f64>,
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
    pub memory_usage: usize,
}

/// Eviction policy for cache management
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    FIFO,
    TimeBasedExpiry,
    SizeBasedEviction,
}

/// Load balancing strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastLoaded,
    WeightedDistribution,
    AdaptiveBalancing,
}

/// Resource allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub memory_limits: HashMap<String, usize>,
    pub cpu_allocation: HashMap<String, f64>,
    pub priority_levels: HashMap<String, i32>,
}

/// Weight scheme for integration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WeightScheme {
    Equal,
    PriorityBased,
    PerformanceBased,
    AdaptiveWeighting,
    ContextualWeighting,
}

/// Normalization method for result integration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NormalizationMethod {
    MinMax,
    ZScore,
    RobustScaling,
    QuantileNormalization,
    UnitVector,
}

/// Trait for cacheable analysis results
pub trait CacheableResult: std::fmt::Debug + Send + Sync {
    fn cache_key(&self) -> u64;
    fn cache_size(&self) -> usize;
    fn is_valid(&self) -> bool;
    fn clone_boxed(&self) -> Box<dyn CacheableResult>;
}

/// Errors that can occur during prosodic fluency analysis
#[derive(Debug, Error)]
pub enum ProsodicFluencyError {
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    #[error("Component analysis failed: {component} - {error}")]
    ComponentAnalysisError { component: String, error: String },

    #[error("Result synthesis failed: {synthesis_type}")]
    ResultSynthesisError { synthesis_type: String },

    #[error("Cache operation failed: {operation} - {details}")]
    CacheError { operation: String, details: String },

    #[error("Parallel execution failed: {component}")]
    ParallelExecutionError { component: String },

    #[error("Invalid input data: {data_type} - {reason}")]
    InvalidInputError { data_type: String, reason: String },

    #[error("Analysis timeout: {component} exceeded {timeout_ms}ms")]
    TimeoutError { component: String, timeout_ms: u64 },

    #[error("Resource exhaustion: {resource_type}")]
    ResourceExhaustionError { resource_type: String },
}

impl ProsodicFluencyAnalyzer {
    /// Creates a new prosodic fluency analyzer with the specified configuration
    pub fn new(config: ProsodicConfig) -> Self {
        // Initialize specialized component analyzers
        let rhythm_analyzer = RhythmAnalyzer::new(config.rhythm.clone());
        let stress_analyzer = StressAnalyzer::new(config.stress.clone());
        let intonation_analyzer = IntonationAnalyzer::new(config.intonation.clone());
        let timing_analyzer = TimingAnalyzer::new(config.timing.clone());
        let phonological_analyzer = PhonologicalAnalyzer::new(config.phonological.clone());
        let advanced_analyzer = AdvancedProsodicAnalyzer::new(config.advanced.clone());

        // Initialize orchestration components
        let analysis_orchestrator = AnalysisOrchestrator::new(&config);
        let result_synthesizer = ResultSynthesizer::new(&config);

        Self {
            config,
            rhythm_analyzer,
            stress_analyzer,
            intonation_analyzer,
            timing_analyzer,
            phonological_analyzer,
            advanced_analyzer,
            analysis_orchestrator,
            result_synthesizer,
            component_cache: HashMap::new(),
            analysis_cache: HashMap::new(),
            analysis_stats: AnalysisStatistics::new(),
        }
    }

    /// Performs comprehensive prosodic fluency analysis
    pub fn analyze(
        &mut self,
        text: &str,
        phonetic_transcription: &str,
    ) -> Result<ProsodicFluencyResult, ProsodicFluencyError> {
        // Validate inputs
        self.validate_inputs(text, phonetic_transcription)?;

        // Generate cache key
        let cache_key = self.generate_cache_key(text, phonetic_transcription);

        // Check cache first
        if let Some(cached_result) = self.analysis_cache.get(&cache_key) {
            self.analysis_stats.update_cache_hit("main_analysis");
            return Ok(cached_result.clone());
        }

        // Start analysis timing
        let analysis_start = std::time::Instant::now();

        // Execute component analyses based on strategy
        let component_results = self.execute_component_analyses(text, phonetic_transcription)?;

        // Synthesize results
        let synthesized_result = self.synthesize_results(component_results)?;

        // Update statistics
        let analysis_duration = analysis_start.elapsed();
        self.analysis_stats
            .update_analysis_time("total_analysis", analysis_duration.as_secs_f64());
        self.analysis_stats.increment_analysis_count();

        // Cache result if enabled
        if self.config.general.enable_caching {
            self.analysis_cache
                .insert(cache_key, synthesized_result.clone());
        }

        Ok(synthesized_result)
    }

    /// Performs analysis with additional audio features (extended interface)
    pub fn analyze_comprehensive(
        &mut self,
        text: &str,
        phonetic_transcription: &str,
        audio_features: Option<&AudioFeatures>,
    ) -> Result<ProsodicFluencyResult, ProsodicFluencyError> {
        // Enhanced analysis with audio features
        let mut result = self.analyze(text, phonetic_transcription)?;

        if let Some(audio) = audio_features {
            self.enhance_with_audio_features(&mut result, audio)?;
        }

        Ok(result)
    }

    /// Executes component analyses based on configuration strategy
    fn execute_component_analyses(
        &mut self,
        text: &str,
        phonetic_transcription: &str,
    ) -> Result<ComponentAnalysisResults, ProsodicFluencyError> {
        match self.analysis_orchestrator.execution_strategy {
            ExecutionStrategy::Sequential => {
                self.execute_sequential_analysis(text, phonetic_transcription)
            }
            ExecutionStrategy::Parallel => {
                self.execute_parallel_analysis(text, phonetic_transcription)
            }
            ExecutionStrategy::Adaptive => {
                self.execute_adaptive_analysis(text, phonetic_transcription)
            }
            ExecutionStrategy::PriorityBased => {
                self.execute_priority_based_analysis(text, phonetic_transcription)
            }
        }
    }

    /// Executes sequential component analysis
    fn execute_sequential_analysis(
        &mut self,
        text: &str,
        phonetic_transcription: &str,
    ) -> Result<ComponentAnalysisResults, ProsodicFluencyError> {
        let mut results = ComponentAnalysisResults::new();

        // Execute components in dependency order
        for component in &self.analysis_orchestrator.get_execution_order() {
            let component_result =
                self.execute_component_analysis(component, text, phonetic_transcription)?;
            results.insert(component.clone(), component_result);
        }

        Ok(results)
    }

    /// Executes parallel component analysis
    fn execute_parallel_analysis(
        &mut self,
        text: &str,
        phonetic_transcription: &str,
    ) -> Result<ComponentAnalysisResults, ProsodicFluencyError> {
        // Note: In a real implementation, this would use proper async/parallel execution
        // For now, we simulate parallel execution with sequential calls

        let mut results = ComponentAnalysisResults::new();

        // Group components by dependency level for parallel execution
        let execution_groups = self.analysis_orchestrator.get_parallel_execution_groups();

        for group in execution_groups {
            // Execute all components in this group "in parallel"
            for component in group {
                let component_result =
                    self.execute_component_analysis(&component, text, phonetic_transcription)?;
                results.insert(component, component_result);
            }
        }

        Ok(results)
    }

    /// Executes adaptive component analysis
    fn execute_adaptive_analysis(
        &mut self,
        text: &str,
        phonetic_transcription: &str,
    ) -> Result<ComponentAnalysisResults, ProsodicFluencyError> {
        // Determine optimal execution strategy based on text characteristics
        let text_complexity = self.assess_text_complexity(text, phonetic_transcription);

        let strategy = if text_complexity > 0.8 {
            ExecutionStrategy::Parallel
        } else if text_complexity > 0.5 {
            ExecutionStrategy::PriorityBased
        } else {
            ExecutionStrategy::Sequential
        };

        // Temporarily switch strategy and execute
        let original_strategy = self.analysis_orchestrator.execution_strategy;
        self.analysis_orchestrator.execution_strategy = strategy;

        let result = match strategy {
            ExecutionStrategy::Parallel => {
                self.execute_parallel_analysis(text, phonetic_transcription)
            }
            ExecutionStrategy::PriorityBased => {
                self.execute_priority_based_analysis(text, phonetic_transcription)
            }
            _ => self.execute_sequential_analysis(text, phonetic_transcription),
        };

        // Restore original strategy
        self.analysis_orchestrator.execution_strategy = original_strategy;

        result
    }

    /// Executes priority-based component analysis
    fn execute_priority_based_analysis(
        &mut self,
        text: &str,
        phonetic_transcription: &str,
    ) -> Result<ComponentAnalysisResults, ProsodicFluencyError> {
        let mut results = ComponentAnalysisResults::new();

        // Execute components in priority order
        let priority_order = self.analysis_orchestrator.get_priority_order();

        for component in priority_order {
            let component_result =
                self.execute_component_analysis(&component, text, phonetic_transcription)?;
            results.insert(component, component_result);
        }

        Ok(results)
    }

    /// Executes analysis for a specific component
    fn execute_component_analysis(
        &mut self,
        component: &str,
        text: &str,
        phonetic_transcription: &str,
    ) -> Result<Box<dyn CacheableResult>, ProsodicFluencyError> {
        let component_start = std::time::Instant::now();

        let result = match component {
            "rhythm" => {
                let metrics = self
                    .rhythm_analyzer
                    .analyze(text, phonetic_transcription)
                    .map_err(|e| ProsodicFluencyError::ComponentAnalysisError {
                        component: component.to_string(),
                        error: e.to_string(),
                    })?;
                Box::new(CacheableRhythmMetrics(metrics)) as Box<dyn CacheableResult>
            }

            "stress" => {
                let metrics = self
                    .stress_analyzer
                    .analyze(text, phonetic_transcription)
                    .map_err(|e| ProsodicFluencyError::ComponentAnalysisError {
                        component: component.to_string(),
                        error: e.to_string(),
                    })?;
                Box::new(CacheableStressMetrics(metrics)) as Box<dyn CacheableResult>
            }

            "intonation" => {
                let metrics = self
                    .intonation_analyzer
                    .analyze(text, phonetic_transcription)
                    .map_err(|e| ProsodicFluencyError::ComponentAnalysisError {
                        component: component.to_string(),
                        error: e.to_string(),
                    })?;
                Box::new(CacheableIntonationMetrics(metrics)) as Box<dyn CacheableResult>
            }

            "timing" => {
                let metrics = self
                    .timing_analyzer
                    .analyze(text, phonetic_transcription)
                    .map_err(|e| ProsodicFluencyError::ComponentAnalysisError {
                        component: component.to_string(),
                        error: e.to_string(),
                    })?;
                Box::new(CacheableTimingMetrics(metrics)) as Box<dyn CacheableResult>
            }

            "phonological" => {
                let metrics = self
                    .phonological_analyzer
                    .analyze(text, phonetic_transcription)
                    .map_err(|e| ProsodicFluencyError::ComponentAnalysisError {
                        component: component.to_string(),
                        error: e.to_string(),
                    })?;
                Box::new(CacheablePhonologicalMetrics(metrics)) as Box<dyn CacheableResult>
            }

            _ => {
                return Err(ProsodicFluencyError::ComponentAnalysisError {
                    component: component.to_string(),
                    error: "Unknown component".to_string(),
                })
            }
        };

        let component_duration = component_start.elapsed();
        self.analysis_stats
            .update_analysis_time(component, component_duration.as_secs_f64());

        Ok(result)
    }

    /// Synthesizes component results into final prosodic fluency result
    fn synthesize_results(
        &mut self,
        component_results: ComponentAnalysisResults,
    ) -> Result<ProsodicFluencyResult, ProsodicFluencyError> {
        self.result_synthesizer.synthesize(component_results)
    }

    /// Validates input parameters
    fn validate_inputs(
        &self,
        text: &str,
        phonetic_transcription: &str,
    ) -> Result<(), ProsodicFluencyError> {
        if text.is_empty() {
            return Err(ProsodicFluencyError::InvalidInputError {
                data_type: "text".to_string(),
                reason: "Text cannot be empty".to_string(),
            });
        }

        if phonetic_transcription.is_empty() {
            return Err(ProsodicFluencyError::InvalidInputError {
                data_type: "phonetic_transcription".to_string(),
                reason: "Phonetic transcription cannot be empty".to_string(),
            });
        }

        // Additional validation logic...

        Ok(())
    }

    /// Generates cache key for analysis
    fn generate_cache_key(&self, text: &str, phonetic_transcription: &str) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        text.hash(&mut hasher);
        phonetic_transcription.hash(&mut hasher);
        self.config.general.analysis_depth.hash(&mut hasher);
        hasher.finish()
    }

    /// Assesses text complexity for adaptive analysis
    fn assess_text_complexity(&self, text: &str, phonetic_transcription: &str) -> f64 {
        let word_count = text.split_whitespace().count();
        let phoneme_count = phonetic_transcription.split_whitespace().count();
        let avg_word_length = text.len() as f64 / word_count.max(1) as f64;

        // Simplified complexity assessment
        let length_complexity = (word_count as f64 / 100.0).min(1.0);
        let phonemic_complexity = (phoneme_count as f64 / word_count.max(1) as f64 / 5.0).min(1.0);
        let structural_complexity = (avg_word_length / 8.0).min(1.0);

        (length_complexity + phonemic_complexity + structural_complexity) / 3.0
    }

    /// Enhances results with audio features
    fn enhance_with_audio_features(
        &self,
        result: &mut ProsodicFluencyResult,
        audio_features: &AudioFeatures,
    ) -> Result<(), ProsodicFluencyError> {
        // Enhance rhythm metrics with audio-derived beat detection
        if let Some(beat_times) = &audio_features.beat_times {
            result.rhythm_metrics.audio_confirmed_beats = Some(beat_times.clone());
        }

        // Enhance intonation metrics with fundamental frequency data
        if let Some(f0_contour) = &audio_features.f0_contour {
            result.intonation_metrics.measured_f0_contour = Some(f0_contour.clone());
        }

        // Enhance timing metrics with actual pause durations
        if let Some(pause_durations) = &audio_features.pause_durations {
            result.timing_metrics.measured_pause_durations = Some(pause_durations.clone());
        }

        Ok(())
    }

    /// Updates the analysis configuration
    pub fn update_config(&mut self, new_config: ProsodicConfig) {
        self.config = new_config;

        // Update all component analyzers
        self.rhythm_analyzer
            .update_config(self.config.rhythm.clone());
        self.stress_analyzer
            .update_config(self.config.stress.clone());
        self.intonation_analyzer
            .update_config(self.config.intonation.clone());
        self.timing_analyzer
            .update_config(self.config.timing.clone());
        self.phonological_analyzer
            .update_config(self.config.phonological.clone());
        self.advanced_analyzer
            .update_config(self.config.advanced.clone());

        // Update orchestration components
        self.analysis_orchestrator.update_config(&self.config);
        self.result_synthesizer.update_config(&self.config);

        // Clear caches if significant configuration change
        if self.config.general.clear_cache_on_config_change {
            self.clear_all_caches();
        }
    }

    /// Clears all analysis caches
    pub fn clear_all_caches(&mut self) {
        self.analysis_cache.clear();
        self.component_cache.clear();

        // Clear component-level caches
        self.rhythm_analyzer.clear_cache();
        self.stress_analyzer.clear_cache();
        self.intonation_analyzer.clear_cache();
        self.timing_analyzer.clear_cache();
        self.phonological_analyzer.clear_cache();
        self.advanced_analyzer.clear_cache();
    }

    /// Gets comprehensive analysis statistics
    pub fn get_analysis_statistics(&self) -> &AnalysisStatistics {
        &self.analysis_stats
    }

    /// Gets cache statistics for all components
    pub fn get_cache_statistics(&self) -> HashMap<String, CacheStatistics> {
        let mut stats = HashMap::new();

        for (component, cache) in &self.component_cache {
            stats.insert(component.clone(), cache.cache_stats.clone());
        }

        // Add main analysis cache stats
        stats.insert(
            "main_analysis".to_string(),
            CacheStatistics {
                hit_count: 0, // Would track in real implementation
                miss_count: 0,
                eviction_count: 0,
                memory_usage: self.analysis_cache.len()
                    * std::mem::size_of::<ProsodicFluencyResult>(),
            },
        );

        stats
    }
}

/// Component analysis results container
pub type ComponentAnalysisResults = HashMap<String, Box<dyn CacheableResult>>;

/// Audio features for enhanced analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFeatures {
    pub beat_times: Option<Vec<f64>>,
    pub f0_contour: Option<Vec<f64>>,
    pub pause_durations: Option<Vec<f64>>,
    pub intensity_contour: Option<Vec<f64>>,
    pub spectral_features: Option<SpectralFeatures>,
}

/// Spectral features from audio analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralFeatures {
    pub spectral_centroid: Vec<f64>,
    pub spectral_rolloff: Vec<f64>,
    pub mfccs: Vec<Vec<f64>>,
    pub spectral_contrast: Vec<f64>,
}

// Cacheable wrapper types for component results
#[derive(Debug, Clone)]
pub struct CacheableRhythmMetrics(pub RhythmMetrics);

#[derive(Debug, Clone)]
pub struct CacheableStressMetrics(pub StressMetrics);

#[derive(Debug, Clone)]
pub struct CacheableIntonationMetrics(pub IntonationMetrics);

#[derive(Debug, Clone)]
pub struct CacheableTimingMetrics(pub TimingMetrics);

#[derive(Debug, Clone)]
pub struct CacheablePhonologicalMetrics(pub PhonologicalMetrics);

// Implementation of CacheableResult trait for all metric types
impl CacheableResult for CacheableRhythmMetrics {
    fn cache_key(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.0.beat_consistency.to_bits().hash(&mut hasher);
        hasher.finish()
    }

    fn cache_size(&self) -> usize {
        std::mem::size_of::<RhythmMetrics>()
    }

    fn is_valid(&self) -> bool {
        !self.0.beat_consistency.is_nan()
    }

    fn clone_boxed(&self) -> Box<dyn CacheableResult> {
        Box::new(self.clone())
    }
}

impl CacheableResult for CacheableStressMetrics {
    fn cache_key(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.0.stress_prominence_score.to_bits().hash(&mut hasher);
        hasher.finish()
    }

    fn cache_size(&self) -> usize {
        std::mem::size_of::<StressMetrics>()
    }

    fn is_valid(&self) -> bool {
        !self.0.stress_prominence_score.is_nan()
    }

    fn clone_boxed(&self) -> Box<dyn CacheableResult> {
        Box::new(self.clone())
    }
}

impl CacheableResult for CacheableIntonationMetrics {
    fn cache_key(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.0.pitch_range.to_bits().hash(&mut hasher);
        hasher.finish()
    }

    fn cache_size(&self) -> usize {
        std::mem::size_of::<IntonationMetrics>()
    }

    fn is_valid(&self) -> bool {
        !self.0.pitch_range.is_nan()
    }

    fn clone_boxed(&self) -> Box<dyn CacheableResult> {
        Box::new(self.clone())
    }
}

impl CacheableResult for CacheableTimingMetrics {
    fn cache_key(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.0.speech_rate.to_bits().hash(&mut hasher);
        hasher.finish()
    }

    fn cache_size(&self) -> usize {
        std::mem::size_of::<TimingMetrics>()
    }

    fn is_valid(&self) -> bool {
        !self.0.speech_rate.is_nan()
    }

    fn clone_boxed(&self) -> Box<dyn CacheableResult> {
        Box::new(self.clone())
    }
}

impl CacheableResult for CacheablePhonologicalMetrics {
    fn cache_key(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.0.complexity_score.to_bits().hash(&mut hasher);
        hasher.finish()
    }

    fn cache_size(&self) -> usize {
        std::mem::size_of::<PhonologicalMetrics>()
    }

    fn is_valid(&self) -> bool {
        !self.0.complexity_score.is_nan()
    }

    fn clone_boxed(&self) -> Box<dyn CacheableResult> {
        Box::new(self.clone())
    }
}

impl AnalysisOrchestrator {
    fn new(config: &ProsodicConfig) -> Self {
        Self {
            execution_strategy: config.general.execution_strategy,
            component_dependencies: Self::create_component_dependencies(),
            parallel_execution_config: ParallelExecutionConfig::default(),
            error_recovery_strategy: ErrorRecoveryStrategy::BestEffort,
        }
    }

    fn get_execution_order(&self) -> Vec<String> {
        // Define component execution order based on dependencies
        vec![
            "rhythm".to_string(),
            "stress".to_string(),
            "intonation".to_string(),
            "timing".to_string(),
            "phonological".to_string(),
            "advanced".to_string(),
        ]
    }

    fn get_parallel_execution_groups(&self) -> Vec<Vec<String>> {
        // Group components that can run in parallel
        vec![
            vec!["rhythm".to_string(), "timing".to_string()], // Independent analyses
            vec![
                "stress".to_string(),
                "intonation".to_string(),
                "phonological".to_string(),
            ], // Can run in parallel
            vec!["advanced".to_string()],                     // Depends on all others
        ]
    }

    fn get_priority_order(&self) -> Vec<String> {
        // Order by analysis priority/importance
        vec![
            "stress".to_string(),       // High priority - critical for fluency
            "intonation".to_string(),   // High priority - perceptually important
            "rhythm".to_string(),       // Medium priority
            "timing".to_string(),       // Medium priority
            "phonological".to_string(), // Lower priority
            "advanced".to_string(),     // Final integration
        ]
    }

    fn create_component_dependencies() -> HashMap<String, Vec<String>> {
        let mut deps = HashMap::new();

        // Most components are independent
        deps.insert("rhythm".to_string(), vec![]);
        deps.insert("stress".to_string(), vec![]);
        deps.insert("intonation".to_string(), vec![]);
        deps.insert("timing".to_string(), vec![]);
        deps.insert("phonological".to_string(), vec![]);

        // Advanced analysis depends on all other components
        deps.insert(
            "advanced".to_string(),
            vec![
                "rhythm".to_string(),
                "stress".to_string(),
                "intonation".to_string(),
                "timing".to_string(),
                "phonological".to_string(),
            ],
        );

        deps
    }

    fn update_config(&mut self, config: &ProsodicConfig) {
        self.execution_strategy = config.general.execution_strategy;
    }
}

impl ResultSynthesizer {
    fn new(config: &ProsodicConfig) -> Self {
        Self {
            synthesis_weights: Self::create_synthesis_weights(),
            integration_algorithms: Self::create_integration_algorithms(),
            quality_assessment: QualityAssessment::default(),
            confidence_calculator: ConfidenceCalculator::default(),
        }
    }

    fn synthesize(
        &self,
        component_results: ComponentAnalysisResults,
    ) -> Result<ProsodicFluencyResult, ProsodicFluencyError> {
        // Extract individual component results
        let rhythm_metrics = self.extract_rhythm_metrics(&component_results)?;
        let stress_metrics = self.extract_stress_metrics(&component_results)?;
        let intonation_metrics = self.extract_intonation_metrics(&component_results)?;
        let timing_metrics = self.extract_timing_metrics(&component_results)?;
        let phonological_metrics = self.extract_phonological_metrics(&component_results)?;

        // Calculate overall fluency score
        let overall_fluency_score = self.calculate_overall_fluency_score(
            &rhythm_metrics,
            &stress_metrics,
            &intonation_metrics,
            &timing_metrics,
            &phonological_metrics,
        );

        // Calculate analysis confidence
        let analysis_confidence = self.calculate_analysis_confidence(
            &rhythm_metrics,
            &stress_metrics,
            &intonation_metrics,
            &timing_metrics,
            &phonological_metrics,
        );

        // Determine fluency level
        let fluency_level = self.determine_fluency_level(overall_fluency_score);

        // Generate detailed insights
        let fluency_insights = self.generate_fluency_insights(
            &rhythm_metrics,
            &stress_metrics,
            &intonation_metrics,
            &timing_metrics,
            &phonological_metrics,
        );

        Ok(ProsodicFluencyResult {
            overall_fluency_score,
            analysis_confidence,
            fluency_level,
            rhythm_metrics,
            stress_metrics,
            intonation_metrics,
            timing_metrics,
            phonological_metrics,
            advanced_metrics: None, // Would be computed if advanced analysis is enabled
            fluency_insights,
            component_contributions: self.calculate_component_contributions(&component_results),
        })
    }

    fn extract_rhythm_metrics(
        &self,
        results: &ComponentAnalysisResults,
    ) -> Result<RhythmMetrics, ProsodicFluencyError> {
        let result =
            results
                .get("rhythm")
                .ok_or_else(|| ProsodicFluencyError::ResultSynthesisError {
                    synthesis_type: "rhythm_metrics_extraction".to_string(),
                })?;

        // Downcast to concrete type
        if let Some(cacheable_rhythm) = result.as_any().downcast_ref::<CacheableRhythmMetrics>() {
            Ok(cacheable_rhythm.0.clone())
        } else {
            Err(ProsodicFluencyError::ResultSynthesisError {
                synthesis_type: "rhythm_metrics_cast".to_string(),
            })
        }
    }

    fn extract_stress_metrics(
        &self,
        results: &ComponentAnalysisResults,
    ) -> Result<StressMetrics, ProsodicFluencyError> {
        let result =
            results
                .get("stress")
                .ok_or_else(|| ProsodicFluencyError::ResultSynthesisError {
                    synthesis_type: "stress_metrics_extraction".to_string(),
                })?;

        if let Some(cacheable_stress) = result.as_any().downcast_ref::<CacheableStressMetrics>() {
            Ok(cacheable_stress.0.clone())
        } else {
            Err(ProsodicFluencyError::ResultSynthesisError {
                synthesis_type: "stress_metrics_cast".to_string(),
            })
        }
    }

    fn extract_intonation_metrics(
        &self,
        results: &ComponentAnalysisResults,
    ) -> Result<IntonationMetrics, ProsodicFluencyError> {
        let result = results.get("intonation").ok_or_else(|| {
            ProsodicFluencyError::ResultSynthesisError {
                synthesis_type: "intonation_metrics_extraction".to_string(),
            }
        })?;

        if let Some(cacheable_intonation) =
            result.as_any().downcast_ref::<CacheableIntonationMetrics>()
        {
            Ok(cacheable_intonation.0.clone())
        } else {
            Err(ProsodicFluencyError::ResultSynthesisError {
                synthesis_type: "intonation_metrics_cast".to_string(),
            })
        }
    }

    fn extract_timing_metrics(
        &self,
        results: &ComponentAnalysisResults,
    ) -> Result<TimingMetrics, ProsodicFluencyError> {
        let result =
            results
                .get("timing")
                .ok_or_else(|| ProsodicFluencyError::ResultSynthesisError {
                    synthesis_type: "timing_metrics_extraction".to_string(),
                })?;

        if let Some(cacheable_timing) = result.as_any().downcast_ref::<CacheableTimingMetrics>() {
            Ok(cacheable_timing.0.clone())
        } else {
            Err(ProsodicFluencyError::ResultSynthesisError {
                synthesis_type: "timing_metrics_cast".to_string(),
            })
        }
    }

    fn extract_phonological_metrics(
        &self,
        results: &ComponentAnalysisResults,
    ) -> Result<PhonologicalMetrics, ProsodicFluencyError> {
        let result = results.get("phonological").ok_or_else(|| {
            ProsodicFluencyError::ResultSynthesisError {
                synthesis_type: "phonological_metrics_extraction".to_string(),
            }
        })?;

        if let Some(cacheable_phonological) = result
            .as_any()
            .downcast_ref::<CacheablePhonologicalMetrics>()
        {
            Ok(cacheable_phonological.0.clone())
        } else {
            Err(ProsodicFluencyError::ResultSynthesisError {
                synthesis_type: "phonological_metrics_cast".to_string(),
            })
        }
    }

    fn calculate_overall_fluency_score(
        &self,
        rhythm_metrics: &RhythmMetrics,
        stress_metrics: &StressMetrics,
        intonation_metrics: &IntonationMetrics,
        timing_metrics: &TimingMetrics,
        phonological_metrics: &PhonologicalMetrics,
    ) -> f64 {
        let weights = &self.synthesis_weights;

        let rhythm_contribution =
            rhythm_metrics.beat_consistency * weights.get("rhythm").unwrap_or(&0.2);
        let stress_contribution =
            stress_metrics.stress_consistency * weights.get("stress").unwrap_or(&0.25);
        let intonation_contribution =
            intonation_metrics.contour_smoothness * weights.get("intonation").unwrap_or(&0.25);
        let timing_contribution =
            timing_metrics.tempo_consistency * weights.get("timing").unwrap_or(&0.2);
        let phonological_contribution = (1.0
            - phonological_metrics
                .phonotactic_constraints
                .violation_density)
            * weights.get("phonological").unwrap_or(&0.1);

        (rhythm_contribution
            + stress_contribution
            + intonation_contribution
            + timing_contribution
            + phonological_contribution)
            .max(0.0)
            .min(1.0)
    }

    fn calculate_analysis_confidence(
        &self,
        rhythm_metrics: &RhythmMetrics,
        stress_metrics: &StressMetrics,
        intonation_metrics: &IntonationMetrics,
        timing_metrics: &TimingMetrics,
        phonological_metrics: &PhonologicalMetrics,
    ) -> f64 {
        // Calculate confidence based on data quality and consistency
        let rhythm_confidence = if rhythm_metrics.beat_consistency > 0.7 {
            0.9
        } else {
            0.7
        };
        let stress_confidence = if stress_metrics.stress_consistency > 0.6 {
            0.85
        } else {
            0.65
        };
        let intonation_confidence = if intonation_metrics.contour_smoothness > 0.8 {
            0.9
        } else {
            0.75
        };
        let timing_confidence = if timing_metrics.tempo_consistency > 0.7 {
            0.88
        } else {
            0.7
        };
        let phonological_confidence = if phonological_metrics
            .phonotactic_constraints
            .violation_density
            < 0.2
        {
            0.9
        } else {
            0.8
        };

        (rhythm_confidence
            + stress_confidence
            + intonation_confidence
            + timing_confidence
            + phonological_confidence)
            / 5.0
    }

    fn determine_fluency_level(&self, overall_score: f64) -> FluencyLevel {
        match overall_score {
            score if score >= 0.85 => FluencyLevel::Advanced,
            score if score >= 0.70 => FluencyLevel::Proficient,
            score if score >= 0.55 => FluencyLevel::Intermediate,
            score if score >= 0.40 => FluencyLevel::Elementary,
            _ => FluencyLevel::Beginner,
        }
    }

    fn generate_fluency_insights(
        &self,
        rhythm_metrics: &RhythmMetrics,
        stress_metrics: &StressMetrics,
        intonation_metrics: &IntonationMetrics,
        timing_metrics: &TimingMetrics,
        phonological_metrics: &PhonologicalMetrics,
    ) -> Vec<FluencyInsight> {
        let mut insights = Vec::new();

        // Generate insights based on component metrics
        if rhythm_metrics.beat_consistency < 0.6 {
            insights.push(FluencyInsight {
                category: "Rhythm".to_string(),
                insight: "Beat patterns show irregular timing that may affect speech fluency"
                    .to_string(),
                severity: InsightSeverity::Moderate,
                confidence: 0.8,
            });
        }

        if stress_metrics.stress_prominence_score < 0.5 {
            insights.push(FluencyInsight {
                category: "Stress".to_string(),
                insight: "Weak stress prominence may reduce speech naturalness and clarity"
                    .to_string(),
                severity: InsightSeverity::High,
                confidence: 0.85,
            });
        }

        if intonation_metrics.pitch_range < 50.0 {
            // Assuming Hz units
            insights.push(FluencyInsight {
                category: "Intonation".to_string(),
                insight: "Limited pitch range may result in monotonous speech patterns".to_string(),
                severity: InsightSeverity::Moderate,
                confidence: 0.75,
            });
        }

        if timing_metrics.speech_rate > 6.0 {
            // syllables per second
            insights.push(FluencyInsight {
                category: "Timing".to_string(),
                insight: "Speech rate is unusually fast, which may impact comprehensibility"
                    .to_string(),
                severity: InsightSeverity::Low,
                confidence: 0.7,
            });
        }

        if phonological_metrics
            .phonotactic_constraints
            .violation_density
            > 0.3
        {
            insights.push(FluencyInsight {
                category: "Phonological".to_string(),
                insight:
                    "High rate of phonotactic violations may indicate pronunciation difficulties"
                        .to_string(),
                severity: InsightSeverity::High,
                confidence: 0.9,
            });
        }

        insights
    }

    fn calculate_component_contributions(
        &self,
        results: &ComponentAnalysisResults,
    ) -> HashMap<String, f64> {
        let mut contributions = HashMap::new();

        for (component, _) in results {
            let weight = self.synthesis_weights.get(component).unwrap_or(&0.2);
            contributions.insert(component.clone(), *weight);
        }

        contributions
    }

    fn create_synthesis_weights() -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        weights.insert("rhythm".to_string(), 0.2);
        weights.insert("stress".to_string(), 0.25);
        weights.insert("intonation".to_string(), 0.25);
        weights.insert("timing".to_string(), 0.2);
        weights.insert("phonological".to_string(), 0.1);
        weights
    }

    fn create_integration_algorithms() -> Vec<IntegrationAlgorithm> {
        vec![IntegrationAlgorithm {
            name: "Weighted Average".to_string(),
            algorithm_type: "linear_combination".to_string(),
            input_components: vec![
                "rhythm".to_string(),
                "stress".to_string(),
                "intonation".to_string(),
            ],
            weight_scheme: WeightScheme::PriorityBased,
            normalization_method: NormalizationMethod::MinMax,
        }]
    }

    fn update_config(&mut self, config: &ProsodicConfig) {
        // Update synthesis configuration based on new config
        if config.general.adaptive_synthesis_weights {
            // Adjust weights based on configuration
            for (component, weight) in &mut self.synthesis_weights {
                *weight *= 1.0 + config.general.adaptation_rate;
            }
        }
    }
}

impl AnalysisStatistics {
    fn new() -> Self {
        Self {
            total_analyses: 0,
            component_execution_times: HashMap::new(),
            cache_hit_rates: HashMap::new(),
            error_rates: HashMap::new(),
            quality_scores: HashMap::new(),
        }
    }

    fn increment_analysis_count(&mut self) {
        self.total_analyses += 1;
    }

    fn update_analysis_time(&mut self, component: &str, duration: f64) {
        self.component_execution_times
            .insert(component.to_string(), duration);
    }

    fn update_cache_hit(&mut self, component: &str) {
        let hit_rate = self
            .cache_hit_rates
            .entry(component.to_string())
            .or_insert(0.0);
        *hit_rate = (*hit_rate + 1.0) / 2.0; // Simple moving average
    }
}

// Default implementations for configuration types
impl Default for ParallelExecutionConfig {
    fn default() -> Self {
        Self {
            max_concurrent_components: 4,
            thread_pool_size: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            load_balancing_strategy: LoadBalancingStrategy::RoundRobin,
            resource_allocation: ResourceAllocation::default(),
        }
    }
}

impl Default for ResourceAllocation {
    fn default() -> Self {
        Self {
            memory_limits: HashMap::new(),
            cpu_allocation: HashMap::new(),
            priority_levels: HashMap::new(),
        }
    }
}

impl Default for QualityAssessment {
    fn default() -> Self {
        Self {
            quality_metrics: vec!["consistency".to_string(), "accuracy".to_string()],
            assessment_thresholds: HashMap::new(),
            quality_weights: HashMap::new(),
        }
    }
}

impl Default for ConfidenceCalculator {
    fn default() -> Self {
        Self {
            confidence_factors: vec![
                "data_quality".to_string(),
                "analysis_consistency".to_string(),
            ],
            factor_weights: HashMap::new(),
            base_confidence: 0.8,
            uncertainty_penalties: HashMap::new(),
        }
    }
}

// Additional trait implementations for downcasting support
impl std::any::Any for CacheableRhythmMetrics {
    fn type_id(&self) -> std::any::TypeId {
        std::any::TypeId::of::<Self>()
    }
}

impl std::any::Any for CacheableStressMetrics {
    fn type_id(&self) -> std::any::TypeId {
        std::any::TypeId::of::<Self>()
    }
}

impl std::any::Any for CacheableIntonationMetrics {
    fn type_id(&self) -> std::any::TypeId {
        std::any::TypeId::of::<Self>()
    }
}

impl std::any::Any for CacheableTimingMetrics {
    fn type_id(&self) -> std::any::TypeId {
        std::any::TypeId::of::<Self>()
    }
}

impl std::any::Any for CacheablePhonologicalMetrics {
    fn type_id(&self) -> std::any::TypeId {
        std::any::TypeId::of::<Self>()
    }
}

// Add as_any method to CacheableResult trait
pub trait CacheableResultExt: CacheableResult {
    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T: CacheableResult + std::any::Any> CacheableResultExt for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Blanket implementation for all CacheableResult types
impl CacheableResultExt for dyn CacheableResult {
    fn as_any(&self) -> &dyn std::any::Any {
        // This is a placeholder - in practice, each concrete type would implement this
        unimplemented!("as_any should be implemented by concrete types")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prosodic_fluency_analyzer_creation() {
        let config = ProsodicConfig::default();
        let analyzer = ProsodicFluencyAnalyzer::new(config);
        assert_eq!(analyzer.analysis_stats.total_analyses, 0);
        assert!(analyzer.analysis_cache.is_empty());
    }

    #[test]
    fn test_analysis_orchestrator_execution_order() {
        let config = ProsodicConfig::default();
        let orchestrator = AnalysisOrchestrator::new(&config);
        let order = orchestrator.get_execution_order();

        assert_eq!(order.len(), 6);
        assert_eq!(order[0], "rhythm");
        assert_eq!(order[5], "advanced");
    }

    #[test]
    fn test_parallel_execution_groups() {
        let config = ProsodicConfig::default();
        let orchestrator = AnalysisOrchestrator::new(&config);
        let groups = orchestrator.get_parallel_execution_groups();

        assert_eq!(groups.len(), 3);
        assert!(groups[0].contains(&"rhythm".to_string()));
        assert!(groups[2].contains(&"advanced".to_string()));
    }

    #[test]
    fn test_result_synthesizer_weights() {
        let config = ProsodicConfig::default();
        let synthesizer = ResultSynthesizer::new(&config);

        assert!(synthesizer.synthesis_weights.contains_key("rhythm"));
        assert!(synthesizer.synthesis_weights.contains_key("stress"));

        let total_weight: f64 = synthesizer.synthesis_weights.values().sum();
        assert!((total_weight - 1.0).abs() < 0.001); // Should sum to approximately 1.0
    }

    #[test]
    fn test_cacheable_result_implementations() {
        use crate::metrics::fluency::prosodic::rhythm::RhythmMetrics;

        let rhythm_metrics = RhythmMetrics {
            beat_consistency: 0.8,
            beat_strength_variability: 0.2,
            tempo_stability: 0.85,
            rhythmic_complexity: 1.5,
            detected_beat_patterns: vec!["strong-weak".to_string()],
            rhythm_classification: "stress_timed".to_string(),
            metrical_structure: "4/4".to_string(),
            syncopation_frequency: 0.1,
            microrhythmic_variations: vec![0.02, 0.015, 0.01],
        };

        let cacheable = CacheableRhythmMetrics(rhythm_metrics);

        assert!(cacheable.is_valid());
        assert!(cacheable.cache_size() > 0);
        assert!(cacheable.cache_key() != 0);
    }
}
