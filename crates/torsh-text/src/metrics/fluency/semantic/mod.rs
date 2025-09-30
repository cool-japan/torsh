//! Modular Semantic Fluency Analysis System
//!
//! This module provides a comprehensive, modular approach to semantic fluency analysis,
//! orchestrating specialized analyzers for coherence, meaning, context, relations, and
//! advanced semantic features while maintaining high performance and extensibility.

pub mod advanced;
pub mod coherence;
pub mod config;
pub mod context;
pub mod meaning;
pub mod relations;
pub mod results;

// Re-export key types for convenience
pub use advanced::{AdvancedResult, AdvancedSemanticAnalyzer, SemanticVector};
pub use coherence::{CoherenceResult, SemanticCoherenceAnalyzer};
pub use config::*;
pub use context::{ContextResult, SemanticContextAnalyzer};
pub use meaning::{MeaningResult, SemanticMeaningAnalyzer};
pub use relations::{RelationsResult, SemanticRelationsAnalyzer};
pub use results::*;

use crate::error::TextAnalysisError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModularSemanticError {
    #[error("Configuration error: {0}")]
    ConfigError(String),
    #[error("Analysis error: {0}")]
    AnalysisError(String),
    #[error("Integration error: {0}")]
    IntegrationError(String),
    #[error("Cache error: {0}")]
    CacheError(String),
    #[error("Coherence analysis error: {0}")]
    CoherenceError(#[from] coherence::CoherenceAnalysisError),
    #[error("Meaning analysis error: {0}")]
    MeaningError(#[from] meaning::MeaningAnalysisError),
    #[error("Context analysis error: {0}")]
    ContextError(#[from] context::ContextAnalysisError),
    #[error("Relations analysis error: {0}")]
    RelationsError(#[from] relations::RelationsAnalysisError),
    #[error("Advanced analysis error: {0}")]
    AdvancedError(#[from] advanced::AdvancedAnalysisError),
}

pub type ModularSemanticResult<T> = Result<T, ModularSemanticError>;

/// Predefined analysis presets for different use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticAnalysisPreset {
    /// Minimal analysis for basic semantic understanding
    Minimal,
    /// Comprehensive analysis with all features enabled
    Comprehensive,
    /// Academic-focused analysis with scholarly features
    Academic,
    /// Creative analysis optimized for creative content
    Creative,
}

/// Performance metrics for the modular semantic analysis system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModularPerformanceMetrics {
    pub total_analysis_time: Duration,
    pub coherence_analysis_time: Duration,
    pub meaning_analysis_time: Duration,
    pub context_analysis_time: Duration,
    pub relations_analysis_time: Duration,
    pub advanced_analysis_time: Duration,
    pub integration_time: Duration,
    pub cache_hit_rate: f64,
    pub memory_usage: usize,
}

/// Analysis session tracking for comprehensive insights
#[derive(Debug, Clone)]
pub struct AnalysisSession {
    pub session_id: String,
    pub start_time: Instant,
    pub text_length: usize,
    pub sentence_count: usize,
    pub enabled_modules: Vec<String>,
    pub performance_metrics: ModularPerformanceMetrics,
}

/// Intelligent caching system for semantic analysis results
#[derive(Debug, Clone)]
pub struct SemanticAnalysisCache {
    coherence_cache: HashMap<u64, SemanticCoherenceMetrics>,
    meaning_cache: HashMap<u64, MeaningPreservationMetrics>,
    context_cache: HashMap<u64, ContextualSemanticMetrics>,
    relations_cache: HashMap<u64, SemanticRelationsMetrics>,
    advanced_cache: HashMap<u64, AdvancedSemanticMetrics>,
    integrated_cache: HashMap<u64, SemanticFluencyResult>,
    cache_statistics: CacheStatistics,
}

#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    pub total_requests: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub evictions: usize,
}

impl SemanticAnalysisCache {
    pub fn new() -> Self {
        Self {
            coherence_cache: HashMap::new(),
            meaning_cache: HashMap::new(),
            context_cache: HashMap::new(),
            relations_cache: HashMap::new(),
            advanced_cache: HashMap::new(),
            integrated_cache: HashMap::new(),
            cache_statistics: CacheStatistics::default(),
        }
    }

    pub fn hit_rate(&self) -> f64 {
        if self.cache_statistics.total_requests == 0 {
            0.0
        } else {
            self.cache_statistics.cache_hits as f64 / self.cache_statistics.total_requests as f64
        }
    }

    pub fn clear(&mut self) {
        self.coherence_cache.clear();
        self.meaning_cache.clear();
        self.context_cache.clear();
        self.relations_cache.clear();
        self.advanced_cache.clear();
        self.integrated_cache.clear();
        self.cache_statistics = CacheStatistics::default();
    }
}

/// Incremental processing manager for large document analysis
#[derive(Debug, Clone)]
pub struct IncrementalProcessor {
    pub chunk_size: usize,
    pub overlap_size: usize,
    pub processing_strategy: ProcessingStrategy,
    pub results_buffer: Vec<SemanticFluencyResult>,
}

#[derive(Debug, Clone)]
pub enum ProcessingStrategy {
    Sequential,
    Parallel,
    Adaptive,
}

impl IncrementalProcessor {
    pub fn new(chunk_size: usize, overlap_size: usize) -> Self {
        Self {
            chunk_size,
            overlap_size,
            processing_strategy: ProcessingStrategy::Adaptive,
            results_buffer: Vec::new(),
        }
    }
}

/// Advanced integration manager for coordinating modular components
#[derive(Debug, Clone)]
pub struct IntegrationManager {
    pub integration_strategy: IntegrationStrategy,
    pub component_weights: ComponentWeights,
    pub conflict_resolution: ConflictResolutionStrategy,
    pub quality_thresholds: QualityThresholds,
}

#[derive(Debug, Clone)]
pub enum IntegrationStrategy {
    WeightedAverage,
    EnsembleMethod,
    HierarchicalIntegration,
    AdaptiveIntegration,
}

#[derive(Debug, Clone)]
pub struct ComponentWeights {
    pub coherence_weight: f64,
    pub meaning_weight: f64,
    pub context_weight: f64,
    pub relations_weight: f64,
    pub advanced_weight: f64,
}

impl Default for ComponentWeights {
    fn default() -> Self {
        Self {
            coherence_weight: 0.25,
            meaning_weight: 0.25,
            context_weight: 0.2,
            relations_weight: 0.2,
            advanced_weight: 0.1,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ConflictResolutionStrategy {
    TakeMaximum,
    TakeMinimum,
    TakeAverage,
    UseConfidenceWeighting,
    CustomStrategy(fn(f64, f64) -> f64),
}

#[derive(Debug, Clone)]
pub struct QualityThresholds {
    pub min_coherence_score: f64,
    pub min_meaning_score: f64,
    pub min_context_score: f64,
    pub min_relations_score: f64,
    pub min_advanced_score: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_coherence_score: 0.3,
            min_meaning_score: 0.3,
            min_context_score: 0.3,
            min_relations_score: 0.3,
            min_advanced_score: 0.3,
        }
    }
}

/// Main modular semantic fluency analyzer orchestrating all components
#[derive(Debug)]
pub struct ModularSemanticFluencyAnalyzer {
    config: SemanticConfig,
    coherence_analyzer: SemanticCoherenceAnalyzer,
    meaning_analyzer: SemanticMeaningAnalyzer,
    context_analyzer: SemanticContextAnalyzer,
    relations_analyzer: SemanticRelationsAnalyzer,
    advanced_analyzer: AdvancedSemanticAnalyzer,
    analysis_cache: SemanticAnalysisCache,
    incremental_processor: IncrementalProcessor,
    integration_manager: IntegrationManager,
    current_session: Option<AnalysisSession>,
}

impl ModularSemanticFluencyAnalyzer {
    /// Create new modular semantic analyzer with comprehensive configuration
    pub fn new(config: SemanticConfig) -> ModularSemanticResult<Self> {
        let coherence_analyzer = SemanticCoherenceAnalyzer::new(config.coherence.clone())
            .map_err(|e| ModularSemanticError::CoherenceError(e))?;

        let meaning_analyzer = SemanticMeaningAnalyzer::new(config.meaning.clone())
            .map_err(|e| ModularSemanticError::MeaningError(e))?;

        let context_analyzer = SemanticContextAnalyzer::new(config.context.clone())
            .map_err(|e| ModularSemanticError::ContextError(e))?;

        let relations_analyzer = SemanticRelationsAnalyzer::new(config.relations.clone())
            .map_err(|e| ModularSemanticError::RelationsError(e))?;

        let advanced_analyzer = AdvancedSemanticAnalyzer::new(config.advanced.clone())
            .map_err(|e| ModularSemanticError::AdvancedError(e))?;

        Ok(Self {
            config,
            coherence_analyzer,
            meaning_analyzer,
            context_analyzer,
            relations_analyzer,
            advanced_analyzer,
            analysis_cache: SemanticAnalysisCache::new(),
            incremental_processor: IncrementalProcessor::new(1000, 100), // Default chunk size
            integration_manager: IntegrationManager {
                integration_strategy: IntegrationStrategy::AdaptiveIntegration,
                component_weights: ComponentWeights::default(),
                conflict_resolution: ConflictResolutionStrategy::UseConfidenceWeighting,
                quality_thresholds: QualityThresholds::default(),
            },
            current_session: None,
        })
    }

    /// Create analyzer with preset configuration for common use cases
    pub fn with_preset(preset: SemanticAnalysisPreset) -> ModularSemanticResult<Self> {
        let config = match preset {
            SemanticAnalysisPreset::Minimal => SemanticConfig::minimal(),
            SemanticAnalysisPreset::Comprehensive => SemanticConfig::comprehensive(),
            SemanticAnalysisPreset::Academic => SemanticConfig::academic(),
            SemanticAnalysisPreset::Creative => SemanticConfig::creative(),
        };

        Self::new(config)
    }

    /// Perform comprehensive modular semantic fluency analysis
    pub fn analyze_semantic_fluency(
        &mut self,
        text: &str,
    ) -> ModularSemanticResult<SemanticFluencyResult> {
        self.start_analysis_session(text);

        // Check integrated cache first
        let cache_key = self.generate_cache_key(text);
        if let Some(cached_result) = self.analysis_cache.integrated_cache.get(&cache_key) {
            self.analysis_cache.cache_statistics.total_requests += 1;
            self.analysis_cache.cache_statistics.cache_hits += 1;
            return Ok(cached_result.clone());
        }

        self.analysis_cache.cache_statistics.total_requests += 1;
        self.analysis_cache.cache_statistics.cache_misses += 1;

        // Handle large documents with incremental processing
        if text.len() > self.config.general.max_text_length {
            return self.analyze_with_incremental_processing(text);
        }

        // Perform modular analysis with all enabled components
        let integration_start = Instant::now();

        let result = self.perform_integrated_analysis(text)?;

        // Update performance metrics
        if let Some(ref mut session) = self.current_session {
            session.performance_metrics.integration_time = integration_start.elapsed();
            session.performance_metrics.cache_hit_rate = self.analysis_cache.hit_rate();
        }

        // Cache the integrated result
        self.analysis_cache
            .integrated_cache
            .insert(cache_key, result.clone());

        // Clean up session
        self.finalize_analysis_session();

        Ok(result)
    }

    /// Perform analysis with confidence scoring
    pub fn analyze_with_confidence(
        &mut self,
        text: &str,
    ) -> ModularSemanticResult<(SemanticFluencyResult, f64)> {
        let result = self.analyze_semantic_fluency(text)?;
        let confidence = self.calculate_analysis_confidence(&result)?;
        Ok((result, confidence))
    }

    /// Analyze semantic fluency for multiple texts in batch
    pub fn analyze_batch(
        &mut self,
        texts: &[&str],
    ) -> ModularSemanticResult<Vec<SemanticFluencyResult>> {
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let result = self.analyze_semantic_fluency(text)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Compare semantic fluency between two texts
    pub fn compare_semantic_fluency(
        &mut self,
        text1: &str,
        text2: &str,
    ) -> ModularSemanticResult<SemanticComparisonResult> {
        let result1 = self.analyze_semantic_fluency(text1)?;
        let result2 = self.analyze_semantic_fluency(text2)?;

        let comparison = SemanticComparisonResult {
            text1_result: result1.clone(),
            text2_result: result2.clone(),
            overall_difference: (result1.overall_score - result2.overall_score).abs(),
            coherence_difference: (result1.semantic_coherence - result2.semantic_coherence).abs(),
            meaning_difference: self.calculate_meaning_difference(&result1, &result2)?,
            context_difference: self.calculate_context_difference(&result1, &result2)?,
            relations_difference: self.calculate_relations_difference(&result1, &result2)?,
            advanced_difference: self.calculate_advanced_difference(&result1, &result2)?,
            similarity_score: self.calculate_semantic_similarity(&result1, &result2)?,
            recommendation: self.generate_improvement_recommendation(&result1, &result2)?,
        };

        Ok(comparison)
    }

    /// Get comprehensive performance and cache statistics
    pub fn get_statistics(&self) -> AnalysisStatistics {
        AnalysisStatistics {
            cache_statistics: self.analysis_cache.cache_statistics.clone(),
            performance_metrics: self
                .current_session
                .as_ref()
                .map(|s| s.performance_metrics.clone())
                .unwrap_or_default(),
            cache_sizes: CacheSizes {
                coherence_cache_size: self.analysis_cache.coherence_cache.len(),
                meaning_cache_size: self.analysis_cache.meaning_cache.len(),
                context_cache_size: self.analysis_cache.context_cache.len(),
                relations_cache_size: self.analysis_cache.relations_cache.len(),
                advanced_cache_size: self.analysis_cache.advanced_cache.len(),
                integrated_cache_size: self.analysis_cache.integrated_cache.len(),
            },
        }
    }

    /// Clear all caches and reset performance counters
    pub fn clear_caches(&mut self) {
        self.analysis_cache.clear();
    }

    /// Configure integration weights for component balance
    pub fn configure_integration_weights(
        &mut self,
        weights: ComponentWeights,
    ) -> ModularSemanticResult<()> {
        // Validate weights sum to reasonable value
        let total_weight = weights.coherence_weight
            + weights.meaning_weight
            + weights.context_weight
            + weights.relations_weight
            + weights.advanced_weight;

        if total_weight <= 0.0 {
            return Err(ModularSemanticError::ConfigError(
                "Integration weights must sum to positive value".to_string(),
            ));
        }

        self.integration_manager.component_weights = weights;
        Ok(())
    }

    // Private implementation methods

    fn start_analysis_session(&mut self, text: &str) {
        let sentences = self.extract_sentences(text);

        let session = AnalysisSession {
            session_id: format!("session_{}", chrono::Utc::now().timestamp()),
            start_time: Instant::now(),
            text_length: text.len(),
            sentence_count: sentences.len(),
            enabled_modules: self.get_enabled_modules(),
            performance_metrics: ModularPerformanceMetrics::default(),
        };

        self.current_session = Some(session);
    }

    fn perform_integrated_analysis(
        &mut self,
        text: &str,
    ) -> ModularSemanticResult<SemanticFluencyResult> {
        let mut result = SemanticFluencyResult::default();

        // Coherence analysis
        if self.config.coherence.enabled {
            let start_time = Instant::now();
            let coherence_metrics = self.coherence_analyzer.analyze_semantic_coherence(text)?;
            result.semantic_coherence = coherence_metrics.overall_coherence_score;
            result.coherence_breakdown = Some(coherence_metrics);

            if let Some(ref mut session) = self.current_session {
                session.performance_metrics.coherence_analysis_time = start_time.elapsed();
            }
        }

        // Meaning analysis
        if self.config.meaning.enabled {
            let start_time = Instant::now();
            let meaning_metrics = self
                .meaning_analyzer
                .analyze_meaning_preservation(text, None)?;
            result.meaning_preservation = meaning_metrics.overall_preservation;
            result.meaning_breakdown = Some(meaning_metrics);

            if let Some(ref mut session) = self.current_session {
                session.performance_metrics.meaning_analysis_time = start_time.elapsed();
            }
        }

        // Context analysis
        if self.config.context.enabled {
            let start_time = Instant::now();
            let context_metrics = self
                .context_analyzer
                .analyze_contextual_semantics(text, None)?;
            result.contextual_clarity = context_metrics.overall_contextual_score;
            result.context_breakdown = Some(context_metrics);

            if let Some(ref mut session) = self.current_session {
                session.performance_metrics.context_analysis_time = start_time.elapsed();
            }
        }

        // Relations analysis
        if self.config.relations.enabled {
            let start_time = Instant::now();
            let relations_metrics = self
                .relations_analyzer
                .analyze_semantic_relations(text, None)?;
            result.semantic_connectivity = relations_metrics.overall_relation_score;
            result.relations_breakdown = Some(relations_metrics);

            if let Some(ref mut session) = self.current_session {
                session.performance_metrics.relations_analysis_time = start_time.elapsed();
            }
        }

        // Advanced analysis
        if self.config.advanced.enabled {
            let start_time = Instant::now();
            let advanced_metrics = self
                .advanced_analyzer
                .analyze_advanced_semantics(text, None)?;
            result.conceptual_clarity = advanced_metrics.overall_advanced_score;
            result.advanced_breakdown = Some(advanced_metrics);

            if let Some(ref mut session) = self.current_session {
                session.performance_metrics.advanced_analysis_time = start_time.elapsed();
            }
        }

        // Calculate integrated overall score
        result.overall_score = self.calculate_integrated_score(&result)?;

        // Generate insights and recommendations
        result.insights = self.generate_analysis_insights(&result)?;
        result.recommendations = self.generate_improvement_recommendations(&result)?;

        Ok(result)
    }

    fn calculate_integrated_score(
        &self,
        result: &SemanticFluencyResult,
    ) -> ModularSemanticResult<f64> {
        let weights = &self.integration_manager.component_weights;
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        if self.config.coherence.enabled {
            weighted_sum += result.semantic_coherence * weights.coherence_weight;
            total_weight += weights.coherence_weight;
        }

        if self.config.meaning.enabled {
            weighted_sum += result.meaning_preservation * weights.meaning_weight;
            total_weight += weights.meaning_weight;
        }

        if self.config.context.enabled {
            weighted_sum += result.contextual_clarity * weights.context_weight;
            total_weight += weights.context_weight;
        }

        if self.config.relations.enabled {
            weighted_sum += result.semantic_connectivity * weights.relations_weight;
            total_weight += weights.relations_weight;
        }

        if self.config.advanced.enabled {
            weighted_sum += result.conceptual_clarity * weights.advanced_weight;
            total_weight += weights.advanced_weight;
        }

        if total_weight == 0.0 {
            return Ok(0.0);
        }

        let integrated_score = weighted_sum / total_weight;
        Ok(integrated_score.max(0.0).min(1.0))
    }

    fn extract_sentences(&self, text: &str) -> Vec<String> {
        text.split(&self.config.general.sentence_delimiters)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn generate_cache_key(&self, text: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        self.config.hash(&mut hasher);
        hasher.finish()
    }

    fn finalize_analysis_session(&mut self) {
        if let Some(ref mut session) = self.current_session {
            session.performance_metrics.total_analysis_time = session.start_time.elapsed();
        }
    }

    fn get_enabled_modules(&self) -> Vec<String> {
        let mut modules = Vec::new();

        if self.config.coherence.enabled {
            modules.push("coherence".to_string());
        }
        if self.config.meaning.enabled {
            modules.push("meaning".to_string());
        }
        if self.config.context.enabled {
            modules.push("context".to_string());
        }
        if self.config.relations.enabled {
            modules.push("relations".to_string());
        }
        if self.config.advanced.enabled {
            modules.push("advanced".to_string());
        }

        modules
    }
}

impl Default for ModularSemanticFluencyAnalyzer {
    fn default() -> Self {
        Self::new(SemanticConfig::default()).unwrap()
    }
}

// Additional supporting types and implementations would continue here...
// This represents the core orchestration system for the modular semantic fluency analyzer

impl Default for ModularPerformanceMetrics {
    fn default() -> Self {
        Self {
            total_analysis_time: Duration::from_secs(0),
            coherence_analysis_time: Duration::from_secs(0),
            meaning_analysis_time: Duration::from_secs(0),
            context_analysis_time: Duration::from_secs(0),
            relations_analysis_time: Duration::from_secs(0),
            advanced_analysis_time: Duration::from_secs(0),
            integration_time: Duration::from_secs(0),
            cache_hit_rate: 0.0,
            memory_usage: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnalysisStatistics {
    pub cache_statistics: CacheStatistics,
    pub performance_metrics: ModularPerformanceMetrics,
    pub cache_sizes: CacheSizes,
}

#[derive(Debug, Clone)]
pub struct CacheSizes {
    pub coherence_cache_size: usize,
    pub meaning_cache_size: usize,
    pub context_cache_size: usize,
    pub relations_cache_size: usize,
    pub advanced_cache_size: usize,
    pub integrated_cache_size: usize,
}
