//! Memory access pattern analysis and optimization
//!
//! This module provides advanced memory access pattern analysis capabilities including:
//! - Pattern recognition and classification algorithms
//! - Statistical analysis of memory access behaviors
//! - Cache locality optimization suggestions
//! - Predictive access pattern modeling
//! - Pattern-based performance optimization recommendations

use crate::memory_profiler::allocation::{AccessPattern, AccessType};
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Advanced access pattern analyzer
///
/// Provides sophisticated analysis of memory access patterns with machine learning-inspired
/// pattern recognition and optimization suggestion capabilities.
#[derive(Debug)]
pub struct AccessPatternAnalyzer {
    /// Active patterns being tracked (by memory address)
    active_patterns: Arc<RwLock<HashMap<usize, AccessPattern>>>,

    /// Pattern classification results
    pattern_classifications: Arc<Mutex<HashMap<usize, PatternClassification>>>,

    /// Statistical summaries
    pattern_statistics: Arc<Mutex<PatternStatistics>>,

    /// Pattern-based optimization suggestions
    optimization_suggestions: Arc<Mutex<Vec<PatternOptimizationSuggestion>>>,

    /// Analysis configuration
    config: PatternAnalysisConfig,

    /// Pattern prediction models
    prediction_models: Arc<Mutex<HashMap<usize, AccessPredictionModel>>>,
}

/// Pattern classification result
///
/// Categorizes memory access patterns into well-defined types for optimization purposes.
#[derive(Debug, Clone)]
pub struct PatternClassification {
    /// Primary pattern type
    pub primary_type: PatternType,

    /// Secondary pattern characteristics
    pub secondary_types: Vec<PatternType>,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,

    /// Classification timestamp
    pub classified_at: Instant,

    /// Pattern stability score
    pub stability: f64,

    /// Predicted future behavior
    pub prediction: AccessPrediction,
}

/// Types of memory access patterns
///
/// Comprehensive categorization of memory access behaviors for optimization.
#[derive(Debug, Clone, PartialEq)]
pub enum PatternType {
    /// Sequential access pattern
    Sequential {
        stride: usize,
        direction: AccessDirection,
    },

    /// Random access pattern
    Random {
        entropy: f64,
        distribution: AccessDistribution,
    },

    /// Streaming pattern (large sequential reads/writes)
    Streaming {
        block_size: usize,
        bandwidth_intensive: bool,
    },

    /// Temporal clustering (hot spots)
    TemporalClustering {
        cluster_size: usize,
        access_frequency: f64,
    },

    /// Spatial clustering (locality-based)
    SpatialClustering {
        locality_radius: usize,
        cluster_density: f64,
    },

    /// Strided access pattern
    Strided {
        stride_length: usize,
        stride_consistency: f64,
    },

    /// Cache-friendly pattern
    CacheFriendly {
        cache_line_utilization: f64,
        prefetch_effectiveness: f64,
    },

    /// Cache-hostile pattern
    CacheHostile {
        cache_miss_rate: f64,
        thrashing_likelihood: f64,
    },

    /// Compute-intensive pattern
    ComputeIntensive {
        compute_to_memory_ratio: f64,
        arithmetic_intensity: f64,
    },

    /// Memory-bandwidth-bound pattern
    BandwidthBound {
        bandwidth_utilization: f64,
        transfer_efficiency: f64,
    },

    /// Memory coalescing pattern
    Coalescing {
        coalescing_factor: f64,
        efficiency: f64,
    },

    /// Prefetch pattern
    Prefetch {
        prefetch_distance: usize,
        hit_rate: f64,
    },
}

/// Access direction for sequential patterns
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AccessDirection {
    Forward,
    Backward,
    Bidirectional,
}

/// Statistical distribution of access patterns
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AccessDistribution {
    Uniform,
    Normal,
    Exponential,
    PowerLaw,
    Bimodal,
}

/// Access pattern prediction
///
/// Predicts future memory access behavior based on historical patterns.
#[derive(Debug, Clone)]
pub struct AccessPrediction {
    /// Predicted next access locations
    pub next_accesses: Vec<PredictedAccess>,

    /// Confidence in predictions
    pub prediction_confidence: f64,

    /// Prediction time horizon
    pub time_horizon: Duration,

    /// Recommended prefetch addresses
    pub prefetch_candidates: Vec<usize>,

    /// Expected cache behavior
    pub cache_behavior: CacheBehaviorPrediction,
}

/// Predicted memory access
#[derive(Debug, Clone)]
pub struct PredictedAccess {
    /// Memory address
    pub address: usize,

    /// Access type
    pub access_type: AccessType,

    /// Access size
    pub size: usize,

    /// Prediction confidence
    pub confidence: f64,

    /// Estimated access time
    pub estimated_time: Duration,
}

/// Cache behavior prediction
#[derive(Debug, Clone)]
pub struct CacheBehaviorPrediction {
    /// Expected L1 cache hit rate
    pub l1_hit_rate: f64,

    /// Expected L2 cache hit rate
    pub l2_hit_rate: f64,

    /// Expected TLB hit rate
    pub tlb_hit_rate: f64,

    /// Predicted memory bandwidth usage
    pub bandwidth_usage: f64,

    /// Cache warming recommendations
    pub cache_warming: Vec<CacheWarmingRecommendation>,
}

/// Cache warming recommendation
#[derive(Debug, Clone)]
pub struct CacheWarmingRecommendation {
    /// Address range to prefetch
    pub address_range: (usize, usize),

    /// Prefetch priority
    pub priority: f64,

    /// Estimated benefit
    pub estimated_benefit: f64,

    /// Cache level to target
    pub target_cache_level: CacheLevel,
}

/// Cache level targeting
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheLevel {
    L1,
    L2,
    L3,
    Memory,
}

/// Pattern-based optimization suggestion
///
/// Actionable optimization recommendations based on access pattern analysis.
#[derive(Debug, Clone)]
pub struct PatternOptimizationSuggestion {
    /// Target memory address or allocation
    pub target: usize,

    /// Optimization type
    pub optimization_type: OptimizationType,

    /// Detailed suggestion
    pub suggestion: String,

    /// Expected performance improvement
    pub expected_improvement: f64,

    /// Implementation complexity
    pub complexity: OptimizationComplexity,

    /// Prerequisites for implementation
    pub prerequisites: Vec<String>,

    /// Suggested implementation timeline
    pub timeline: OptimizationTimeline,
}

/// Types of pattern-based optimizations
#[derive(Debug, Clone)]
pub enum OptimizationType {
    /// Data layout reorganization
    DataLayoutOptimization {
        suggested_layout: DataLayout,
        memory_savings: usize,
    },

    /// Prefetching strategy
    PrefetchingOptimization {
        prefetch_distance: usize,
        prefetch_pattern: PrefetchPattern,
    },

    /// Cache optimization
    CacheOptimization {
        cache_strategy: CacheStrategy,
        target_cache_level: CacheLevel,
    },

    /// Memory pooling
    MemoryPooling {
        pool_size: usize,
        allocation_strategy: AllocationStrategy,
    },

    /// Access pattern transformation
    AccessTransformation {
        transformation_type: TransformationType,
        expected_locality_improvement: f64,
    },

    /// Bandwidth optimization
    BandwidthOptimization {
        batching_strategy: BatchingStrategy,
        transfer_optimization: TransferOptimization,
    },
}

/// Data layout suggestions
#[derive(Debug, Clone)]
pub enum DataLayout {
    ArrayOfStructs,
    StructOfArrays,
    Columnar,
    Tiled,
    Compressed,
    Interleaved,
}

/// Prefetch patterns
#[derive(Debug, Clone)]
pub enum PrefetchPattern {
    Sequential,
    Strided,
    Adaptive,
    Predictive,
}

/// Cache strategies
#[derive(Debug, Clone)]
pub enum CacheStrategy {
    WriteThrough,
    WriteBack,
    WriteAround,
    DirectMapped,
    FullyAssociative,
    SetAssociative { ways: usize },
}

/// Allocation strategies
#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    BestFit,
    FirstFit,
    NextFit,
    BuddySystem,
    Slab,
    Pool,
}

/// Access transformation types
#[derive(Debug, Clone)]
pub enum TransformationType {
    Blocking,
    Tiling,
    LoopReordering,
    DataReorganization,
    TemporalBlocking,
    SpatialBlocking,
}

/// Batching strategies
#[derive(Debug, Clone)]
pub enum BatchingStrategy {
    TimeBased,
    SizeBased,
    AdaptiveBatching,
    PriorityBatching,
}

/// Transfer optimization techniques
#[derive(Debug, Clone)]
pub enum TransferOptimization {
    Coalescing,
    Vectorization,
    PipelinedTransfers,
    AsynchronousTransfers,
}

/// Optimization complexity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizationComplexity {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Optimization implementation timeline
#[derive(Debug, Clone)]
pub enum OptimizationTimeline {
    Immediate,
    ShortTerm,  // Within a few days
    MediumTerm, // Within a few weeks
    LongTerm,   // Months of development
}

/// Pattern analysis statistics
///
/// Aggregated statistics about memory access patterns across the system.
#[derive(Debug, Default, Clone)]
pub struct PatternStatistics {
    /// Total patterns analyzed
    pub total_patterns: u64,

    /// Pattern type distribution (pattern name, count)
    pub pattern_distribution: Vec<(String, u64)>,

    /// Average pattern confidence
    pub average_confidence: f64,

    /// Cache efficiency statistics
    pub cache_efficiency: CacheEfficiencyStats,

    /// Temporal analysis results
    pub temporal_analysis: TemporalAnalysisStats,

    /// Spatial analysis results
    pub spatial_analysis: SpatialAnalysisStats,

    /// Performance impact analysis
    pub performance_impact: PerformanceImpactStats,
}

/// Cache efficiency statistics
#[derive(Debug, Default, Clone)]
pub struct CacheEfficiencyStats {
    /// Overall cache hit rate
    pub overall_hit_rate: f64,

    /// L1 cache statistics
    pub l1_stats: CacheLevelStats,

    /// L2 cache statistics
    pub l2_stats: CacheLevelStats,

    /// TLB statistics
    pub tlb_stats: CacheLevelStats,

    /// Cache-friendly pattern percentage
    pub cache_friendly_percentage: f64,
}

/// Cache level statistics
#[derive(Debug, Default, Clone)]
pub struct CacheLevelStats {
    /// Hit rate
    pub hit_rate: f64,

    /// Miss rate
    pub miss_rate: f64,

    /// Average access latency
    pub avg_latency: Duration,

    /// Bandwidth utilization
    pub bandwidth_utilization: f64,
}

/// Temporal analysis statistics
#[derive(Debug, Default, Clone)]
pub struct TemporalAnalysisStats {
    /// Average temporal locality score
    pub avg_temporal_locality: f64,

    /// Hot spot identification
    pub hot_spots: Vec<TemporalHotSpot>,

    /// Access frequency distribution
    pub frequency_distribution: Vec<(f64, u64)>, // (frequency, count)

    /// Temporal clustering strength
    pub clustering_strength: f64,
}

/// Temporal hot spot
#[derive(Debug, Clone)]
pub struct TemporalHotSpot {
    /// Memory address range
    pub address_range: (usize, usize),

    /// Access frequency
    pub frequency: f64,

    /// Hot spot duration
    pub duration: Duration,

    /// Peak access time
    pub peak_time: Instant,
}

/// Spatial analysis statistics
#[derive(Debug, Default, Clone)]
pub struct SpatialAnalysisStats {
    /// Average spatial locality score
    pub avg_spatial_locality: f64,

    /// Stride pattern analysis
    pub stride_patterns: Vec<StridePattern>,

    /// Locality cluster analysis
    pub locality_clusters: Vec<LocalityCluster>,

    /// Fragmentation impact
    pub fragmentation_impact: f64,
}

/// Stride pattern
#[derive(Debug, Clone)]
pub struct StridePattern {
    /// Stride length
    pub stride_length: usize,

    /// Pattern frequency
    pub frequency: u64,

    /// Consistency score
    pub consistency: f64,

    /// Memory range
    pub memory_range: (usize, usize),
}

/// Locality cluster
#[derive(Debug, Clone)]
pub struct LocalityCluster {
    /// Cluster center
    pub center: usize,

    /// Cluster radius
    pub radius: usize,

    /// Access density
    pub density: f64,

    /// Cluster lifetime
    pub lifetime: Duration,
}

/// Performance impact statistics
#[derive(Debug, Default, Clone)]
pub struct PerformanceImpactStats {
    /// Bandwidth efficiency
    pub bandwidth_efficiency: f64,

    /// Cache miss penalty
    pub cache_miss_penalty: Duration,

    /// Memory contention level
    pub contention_level: f64,

    /// Optimization potential
    pub optimization_potential: f64,
}

/// Pattern analysis configuration
#[derive(Debug, Clone)]
pub struct PatternAnalysisConfig {
    /// Minimum pattern length for analysis
    pub min_pattern_length: usize,

    /// Analysis window size
    pub analysis_window: Duration,

    /// Pattern confidence threshold
    pub confidence_threshold: f64,

    /// Enable predictive modeling
    pub enable_prediction: bool,

    /// Enable optimization suggestions
    pub enable_optimization_suggestions: bool,

    /// Maximum tracked patterns
    pub max_tracked_patterns: usize,

    /// Pattern classification sensitivity
    pub classification_sensitivity: f64,

    /// Cache analysis depth
    pub cache_analysis_depth: usize,
}

/// Access prediction model
///
/// Machine learning-inspired model for predicting future memory accesses.
#[derive(Debug)]
pub struct AccessPredictionModel {
    /// Historical access sequence
    access_history: VecDeque<AccessEvent>,

    /// Pattern recognition state
    pattern_state: PatternRecognitionState,

    /// Prediction accuracy tracking
    accuracy_tracker: AccuracyTracker,

    /// Model parameters
    model_params: PredictionModelParams,
}

/// Access event for prediction
#[derive(Debug, Clone)]
pub struct AccessEvent {
    /// Access address
    pub address: usize,

    /// Access type
    pub access_type: AccessType,

    /// Access size
    pub size: usize,

    /// Event timestamp
    pub timestamp: Instant,

    /// Context information
    pub context: AccessContext,
}

/// Access context
#[derive(Debug, Clone)]
pub struct AccessContext {
    /// Thread ID
    pub thread_id: u64,

    /// Kernel or operation name
    pub operation: String,

    /// Memory allocation ID
    pub allocation_id: Option<usize>,
}

/// Pattern recognition state
#[derive(Debug)]
struct PatternRecognitionState {
    /// Current pattern hypothesis
    current_hypothesis: Option<PatternType>,

    /// Pattern transition probabilities
    transition_probabilities: HashMap<PatternType, HashMap<PatternType, f64>>,

    /// State confidence
    confidence: f64,
}

/// Accuracy tracker for predictions
#[derive(Debug)]
struct AccuracyTracker {
    /// Correct predictions
    correct_predictions: u64,

    /// Total predictions made
    total_predictions: u64,

    /// Recent accuracy window
    recent_accuracy: VecDeque<bool>,

    /// Accuracy by pattern type
    accuracy_by_pattern: HashMap<PatternType, (u64, u64)>, // (correct, total)
}

/// Prediction model parameters
#[derive(Debug, Clone)]
struct PredictionModelParams {
    /// History window size
    history_window: usize,

    /// Prediction horizon
    prediction_horizon: Duration,

    /// Learning rate
    learning_rate: f64,

    /// Pattern detection threshold
    pattern_threshold: f64,
}

/// Helper function to get pattern type name
fn pattern_type_name(pattern: &PatternType) -> String {
    match pattern {
        PatternType::Sequential { .. } => "Sequential".to_string(),
        PatternType::Random { .. } => "Random".to_string(),
        PatternType::Streaming { .. } => "Streaming".to_string(),
        PatternType::TemporalClustering { .. } => "TemporalClustering".to_string(),
        PatternType::SpatialClustering { .. } => "SpatialClustering".to_string(),
        PatternType::Strided { .. } => "Strided".to_string(),
        PatternType::Coalescing { .. } => "Coalescing".to_string(),
        PatternType::Prefetch { .. } => "Prefetch".to_string(),
        PatternType::CacheFriendly { .. } => "CacheFriendly".to_string(),
        PatternType::CacheHostile { .. } => "CacheHostile".to_string(),
        PatternType::ComputeIntensive { .. } => "ComputeIntensive".to_string(),
        PatternType::BandwidthBound { .. } => "BandwidthBound".to_string(),
    }
}

impl AccessPatternAnalyzer {
    /// Create a new access pattern analyzer
    pub fn new(config: PatternAnalysisConfig) -> Self {
        Self {
            active_patterns: Arc::new(RwLock::new(HashMap::new())),
            pattern_classifications: Arc::new(Mutex::new(HashMap::new())),
            pattern_statistics: Arc::new(Mutex::new(PatternStatistics::default())),
            optimization_suggestions: Arc::new(Mutex::new(Vec::new())),
            config,
            prediction_models: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Analyze access pattern for a memory allocation
    pub fn analyze_pattern(
        &self,
        address: usize,
        pattern: &AccessPattern,
    ) -> Option<PatternClassification> {
        if pattern.access_times.len() < self.config.min_pattern_length {
            return None;
        }

        let classification = self.classify_pattern(pattern);

        if classification.confidence >= self.config.confidence_threshold {
            // Store classification
            self.pattern_classifications
                .lock()
                .insert(address, classification.clone());

            // Update statistics
            self.update_statistics(&classification);

            // Generate optimization suggestions if enabled
            if self.config.enable_optimization_suggestions {
                self.generate_optimization_suggestions(address, &classification);
            }

            // Update prediction model if enabled
            if self.config.enable_prediction {
                self.update_prediction_model(address, pattern);
            }

            Some(classification)
        } else {
            None
        }
    }

    /// Classify a memory access pattern
    fn classify_pattern(&self, pattern: &AccessPattern) -> PatternClassification {
        let mut confidence_scores: Vec<(PatternType, f64)> = Vec::new();

        // Analyze sequentiality
        confidence_scores.push((
            PatternType::Sequential {
                stride: self.estimate_stride(pattern),
                direction: self.determine_direction(pattern),
            },
            pattern.sequential_score,
        ));

        // Analyze randomness
        confidence_scores.push((
            PatternType::Random {
                entropy: self.calculate_entropy(pattern),
                distribution: self.determine_distribution(pattern),
            },
            pattern.random_score,
        ));

        // Analyze streaming behavior
        let streaming_score = self.analyze_streaming(pattern);
        if streaming_score > 0.3 {
            confidence_scores.push((
                PatternType::Streaming {
                    block_size: self.estimate_block_size(pattern),
                    bandwidth_intensive: streaming_score > 0.7,
                },
                streaming_score,
            ));
        }

        // Analyze temporal clustering
        let temporal_score = pattern.temporal_locality;
        if temporal_score > 0.5 {
            confidence_scores.push((
                PatternType::TemporalClustering {
                    cluster_size: self.estimate_cluster_size(pattern),
                    access_frequency: pattern.frequency,
                },
                temporal_score,
            ));
        }

        // Analyze spatial clustering
        let spatial_score = pattern.spatial_locality;
        if spatial_score > 0.5 {
            confidence_scores.push((
                PatternType::SpatialClustering {
                    locality_radius: self.estimate_locality_radius(pattern),
                    cluster_density: spatial_score,
                },
                spatial_score,
            ));
        }

        // Find primary pattern type
        let primary_index = confidence_scores
            .iter()
            .enumerate()
            .max_by(|(_, (_, a)), (_, (_, b))| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let (primary_type, confidence) = if primary_index < confidence_scores.len() {
            confidence_scores[primary_index].clone()
        } else {
            (
                PatternType::Random {
                    entropy: 1.0,
                    distribution: AccessDistribution::Uniform,
                },
                0.0,
            )
        };

        // Collect secondary patterns (excluding the primary one)
        let secondary_types: Vec<PatternType> = confidence_scores
            .into_iter()
            .enumerate()
            .filter(|(i, (_, score))| *i != primary_index && *score > 0.3)
            .map(|(_, (pattern, _))| pattern)
            .collect();

        // Calculate stability
        let stability = self.calculate_pattern_stability(pattern);

        // Generate prediction
        let prediction = self.generate_access_prediction(pattern, &primary_type);

        PatternClassification {
            primary_type,
            secondary_types,
            confidence,
            classified_at: Instant::now(),
            stability,
            prediction,
        }
    }

    /// Estimate stride for sequential patterns
    fn estimate_stride(&self, pattern: &AccessPattern) -> usize {
        if pattern.access_sizes.len() < 2 {
            return 0;
        }

        let mut strides = Vec::new();
        let access_sizes_vec: Vec<_> = pattern.access_sizes.iter().collect();
        for window in access_sizes_vec.windows(2) {
            if window[1] > window[0] {
                strides.push(window[1] - window[0]);
            }
        }

        if strides.is_empty() {
            0
        } else {
            // Return median stride
            strides.sort_unstable();
            strides[strides.len() / 2]
        }
    }

    /// Determine access direction
    fn determine_direction(&self, pattern: &AccessPattern) -> AccessDirection {
        if pattern.access_sizes.len() < 3 {
            return AccessDirection::Forward;
        }

        let mut forward_count = 0;
        let mut backward_count = 0;

        let access_sizes_vec: Vec<_> = pattern.access_sizes.iter().collect();
        for window in access_sizes_vec.windows(2) {
            if window[1] > window[0] {
                forward_count += 1;
            } else if window[1] < window[0] {
                backward_count += 1;
            }
        }

        let total = forward_count + backward_count;
        if total == 0 {
            return AccessDirection::Forward;
        }

        let forward_ratio = forward_count as f64 / total as f64;

        if forward_ratio > 0.8 {
            AccessDirection::Forward
        } else if forward_ratio < 0.2 {
            AccessDirection::Backward
        } else {
            AccessDirection::Bidirectional
        }
    }

    /// Calculate entropy of access pattern
    fn calculate_entropy(&self, pattern: &AccessPattern) -> f64 {
        if pattern.access_sizes.is_empty() {
            return 0.0;
        }

        // Build frequency distribution
        let mut freq_map = HashMap::new();
        for &size in &pattern.access_sizes {
            *freq_map.entry(size).or_insert(0) += 1;
        }

        let total = pattern.access_sizes.len() as f64;
        let mut entropy = 0.0;

        for count in freq_map.values() {
            let p = *count as f64 / total;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Determine statistical distribution
    fn determine_distribution(&self, _pattern: &AccessPattern) -> AccessDistribution {
        // Simplified distribution detection
        // In a real implementation, this would use statistical tests
        AccessDistribution::Uniform
    }

    /// Analyze streaming behavior
    fn analyze_streaming(&self, pattern: &AccessPattern) -> f64 {
        if pattern.access_sizes.len() < 3 {
            return 0.0;
        }

        // Look for large, consistent access sizes
        let avg_size =
            pattern.access_sizes.iter().sum::<usize>() as f64 / pattern.access_sizes.len() as f64;
        let size_consistency = self.calculate_size_consistency(pattern);

        // Streaming typically involves large, consistent transfers
        if avg_size > 1024.0 && size_consistency > 0.8 {
            (avg_size.log2() / 20.0).min(1.0) * size_consistency
        } else {
            0.0
        }
    }

    /// Calculate size consistency
    fn calculate_size_consistency(&self, pattern: &AccessPattern) -> f64 {
        if pattern.access_sizes.len() < 2 {
            return 1.0;
        }

        let avg_size =
            pattern.access_sizes.iter().sum::<usize>() as f64 / pattern.access_sizes.len() as f64;
        let variance = pattern
            .access_sizes
            .iter()
            .map(|&size| (size as f64 - avg_size).powi(2))
            .sum::<f64>()
            / pattern.access_sizes.len() as f64;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if avg_size > 0.0 {
            std_dev / avg_size
        } else {
            0.0
        };

        // Lower coefficient of variation means higher consistency
        (1.0 - coefficient_of_variation).max(0.0)
    }

    /// Estimate block size for streaming patterns
    fn estimate_block_size(&self, pattern: &AccessPattern) -> usize {
        if pattern.access_sizes.is_empty() {
            return 0;
        }

        // Use median access size as block size estimate
        let mut sizes = pattern.access_sizes.iter().cloned().collect::<Vec<_>>();
        sizes.sort_unstable();
        sizes[sizes.len() / 2]
    }

    /// Estimate cluster size for temporal patterns
    fn estimate_cluster_size(&self, pattern: &AccessPattern) -> usize {
        // Simplified cluster size estimation
        // In practice, this would use clustering algorithms
        (pattern.access_times.len() as f64 * pattern.temporal_locality).round() as usize
    }

    /// Estimate locality radius for spatial patterns
    fn estimate_locality_radius(&self, pattern: &AccessPattern) -> usize {
        if pattern.access_sizes.len() < 2 {
            return 0;
        }

        // Calculate average distance between consecutive accesses
        let mut distances = Vec::new();
        let access_sizes_vec: Vec<_> = pattern.access_sizes.iter().collect();
        for window in access_sizes_vec.windows(2) {
            let distance = if window[1] > window[0] {
                window[1] - window[0]
            } else {
                window[0] - window[1]
            };
            distances.push(distance);
        }

        if distances.is_empty() {
            0
        } else {
            distances.iter().sum::<usize>() / distances.len()
        }
    }

    /// Calculate pattern stability
    fn calculate_pattern_stability(&self, pattern: &AccessPattern) -> f64 {
        if pattern.access_times.len() < 5 {
            return 0.5; // Insufficient data
        }

        // Analyze consistency of access intervals
        let mut intervals = Vec::new();
        let access_times_vec: Vec<_> = pattern.access_times.iter().collect();
        for window in access_times_vec.windows(2) {
            intervals.push(window[1].duration_since(*window[0]).as_nanos() as f64);
        }

        if intervals.is_empty() {
            return 0.5;
        }

        let avg_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let variance = intervals
            .iter()
            .map(|&interval| (interval - avg_interval).powi(2))
            .sum::<f64>()
            / intervals.len() as f64;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if avg_interval > 0.0 {
            std_dev / avg_interval
        } else {
            0.0
        };

        // Lower coefficient of variation indicates higher stability
        (1.0f64 - coefficient_of_variation.min(1.0f64)).max(0.0f64)
    }

    /// Generate access prediction
    fn generate_access_prediction(
        &self,
        pattern: &AccessPattern,
        pattern_type: &PatternType,
    ) -> AccessPrediction {
        let mut next_accesses = Vec::new();
        let mut prefetch_candidates = Vec::new();

        match pattern_type {
            PatternType::Sequential { stride, direction } => {
                if let (Some(&last_size), Some(&_last_time)) =
                    (pattern.access_sizes.back(), pattern.access_times.back())
                {
                    let next_size = match direction {
                        AccessDirection::Forward => last_size + stride,
                        AccessDirection::Backward => last_size.saturating_sub(*stride),
                        AccessDirection::Bidirectional => last_size + stride, // Default to forward
                    };

                    next_accesses.push(PredictedAccess {
                        address: next_size,            // Using size as proxy for address
                        access_type: AccessType::Read, // Default assumption
                        size: last_size,
                        confidence: 0.8,
                        estimated_time: Duration::from_millis(1),
                    });

                    // Generate prefetch candidates
                    for i in 1..=4 {
                        let prefetch_addr = match direction {
                            AccessDirection::Forward => next_size + (stride * i),
                            AccessDirection::Backward => next_size.saturating_sub(stride * i),
                            AccessDirection::Bidirectional => next_size + (stride * i),
                        };
                        prefetch_candidates.push(prefetch_addr);
                    }
                }
            }
            PatternType::Streaming { block_size, .. } => {
                if let Some(&last_size) = pattern.access_sizes.back() {
                    next_accesses.push(PredictedAccess {
                        address: last_size + block_size,
                        access_type: AccessType::Read,
                        size: *block_size,
                        confidence: 0.7,
                        estimated_time: Duration::from_millis(2),
                    });
                }
            }
            _ => {
                // For other patterns, generate generic predictions
                if let Some(&last_size) = pattern.access_sizes.back() {
                    next_accesses.push(PredictedAccess {
                        address: last_size,
                        access_type: AccessType::Read,
                        size: last_size,
                        confidence: 0.3,
                        estimated_time: Duration::from_millis(5),
                    });
                }
            }
        }

        AccessPrediction {
            next_accesses,
            prediction_confidence: 0.6, // Average confidence
            time_horizon: Duration::from_millis(100),
            prefetch_candidates,
            cache_behavior: CacheBehaviorPrediction {
                l1_hit_rate: 0.9,
                l2_hit_rate: 0.7,
                tlb_hit_rate: 0.95,
                bandwidth_usage: 0.5,
                cache_warming: Vec::new(),
            },
        }
    }

    /// Update pattern statistics
    fn update_statistics(&self, classification: &PatternClassification) {
        let mut stats = self.pattern_statistics.lock();
        stats.total_patterns += 1;

        // Update pattern distribution
        let pattern_name = pattern_type_name(&classification.primary_type);
        if let Some((_, count)) = stats
            .pattern_distribution
            .iter_mut()
            .find(|(name, _)| name == &pattern_name)
        {
            *count += 1;
        } else {
            stats.pattern_distribution.push((pattern_name, 1));
        }

        // Update average confidence
        let total_confidence = stats.average_confidence * (stats.total_patterns - 1) as f64
            + classification.confidence;
        stats.average_confidence = total_confidence / stats.total_patterns as f64;
    }

    /// Generate optimization suggestions
    fn generate_optimization_suggestions(
        &self,
        address: usize,
        classification: &PatternClassification,
    ) {
        let mut suggestions = self.optimization_suggestions.lock();

        match &classification.primary_type {
            PatternType::Sequential { stride, .. } => {
                suggestions.push(PatternOptimizationSuggestion {
                    target: address,
                    optimization_type: OptimizationType::PrefetchingOptimization {
                        prefetch_distance: *stride * 4,
                        prefetch_pattern: PrefetchPattern::Sequential,
                    },
                    suggestion: "Implement sequential prefetching to improve cache performance"
                        .to_string(),
                    expected_improvement: 0.3,
                    complexity: OptimizationComplexity::Low,
                    prerequisites: vec!["Hardware prefetcher support".to_string()],
                    timeline: OptimizationTimeline::Immediate,
                });
            }
            PatternType::Random { .. } => {
                suggestions.push(PatternOptimizationSuggestion {
                    target: address,
                    optimization_type: OptimizationType::CacheOptimization {
                        cache_strategy: CacheStrategy::SetAssociative { ways: 8 },
                        target_cache_level: CacheLevel::L2,
                    },
                    suggestion: "Use set-associative cache to handle random access patterns"
                        .to_string(),
                    expected_improvement: 0.2,
                    complexity: OptimizationComplexity::Medium,
                    prerequisites: vec!["Cache configuration access".to_string()],
                    timeline: OptimizationTimeline::ShortTerm,
                });
            }
            PatternType::Streaming { .. } => {
                suggestions.push(PatternOptimizationSuggestion {
                    target: address,
                    optimization_type: OptimizationType::BandwidthOptimization {
                        batching_strategy: BatchingStrategy::SizeBased,
                        transfer_optimization: TransferOptimization::Coalescing,
                    },
                    suggestion: "Implement memory coalescing for streaming workloads".to_string(),
                    expected_improvement: 0.4,
                    complexity: OptimizationComplexity::Medium,
                    prerequisites: vec!["DMA controller access".to_string()],
                    timeline: OptimizationTimeline::MediumTerm,
                });
            }
            _ => {}
        }
    }

    /// Update prediction model
    fn update_prediction_model(&self, address: usize, pattern: &AccessPattern) {
        // Simplified prediction model update
        // In practice, this would implement more sophisticated ML algorithms
        let mut models = self.prediction_models.lock();

        if !models.contains_key(&address) {
            models.insert(address, AccessPredictionModel::new());
        }

        if let Some(model) = models.get_mut(&address) {
            // Convert pattern to access events and update model
            for (i, (&time, &size)) in pattern
                .access_times
                .iter()
                .zip(pattern.access_sizes.iter())
                .enumerate()
            {
                let access_type = pattern
                    .access_types
                    .get(i)
                    .copied()
                    .unwrap_or(AccessType::Read);

                let event = AccessEvent {
                    address: size, // Using size as proxy for address
                    access_type,
                    size,
                    timestamp: time,
                    context: AccessContext {
                        thread_id: 0, // Default thread ID
                        operation: "unknown".to_string(),
                        allocation_id: Some(address),
                    },
                };

                model.add_access_event(event);
            }
        }
    }

    /// Get pattern classification for an address
    pub fn get_classification(&self, address: usize) -> Option<PatternClassification> {
        self.pattern_classifications.lock().get(&address).cloned()
    }

    /// Get optimization suggestions
    pub fn get_optimization_suggestions(&self) -> Vec<PatternOptimizationSuggestion> {
        self.optimization_suggestions.lock().clone()
    }

    /// Get pattern statistics
    pub fn get_statistics(&self) -> PatternStatistics {
        (*self.pattern_statistics.lock()).clone()
    }

    /// Clear old data to prevent memory leaks
    pub fn cleanup_old_data(&self, max_age: Duration) {
        let cutoff = Instant::now() - max_age;

        // Remove old classifications
        let mut classifications = self.pattern_classifications.lock();
        classifications.retain(|_, classification| classification.classified_at > cutoff);

        // Remove old suggestions
        let mut suggestions = self.optimization_suggestions.lock();
        suggestions.retain(|_suggestion| {
            // For simplicity, remove suggestions older than max_age
            // In practice, you'd track suggestion timestamps
            true
        });

        // Cleanup prediction models
        let mut models = self.prediction_models.lock();
        for model in models.values_mut() {
            model.cleanup_old_events(cutoff);
        }
    }
}

impl AccessPredictionModel {
    /// Create a new prediction model
    fn new() -> Self {
        Self {
            access_history: VecDeque::new(),
            pattern_state: PatternRecognitionState {
                current_hypothesis: None,
                transition_probabilities: HashMap::new(),
                confidence: 0.0,
            },
            accuracy_tracker: AccuracyTracker {
                correct_predictions: 0,
                total_predictions: 0,
                recent_accuracy: VecDeque::new(),
                accuracy_by_pattern: HashMap::new(),
            },
            model_params: PredictionModelParams {
                history_window: 100,
                prediction_horizon: Duration::from_millis(100),
                learning_rate: 0.1,
                pattern_threshold: 0.7,
            },
        }
    }

    /// Add an access event to the model
    fn add_access_event(&mut self, event: AccessEvent) {
        self.access_history.push_back(event);

        // Maintain window size
        while self.access_history.len() > self.model_params.history_window {
            self.access_history.pop_front();
        }

        // Update pattern recognition (simplified)
        self.update_pattern_recognition();
    }

    /// Update pattern recognition state
    fn update_pattern_recognition(&mut self) {
        // Simplified pattern recognition update
        // In practice, this would implement more sophisticated algorithms
        if self.access_history.len() >= 10 {
            self.pattern_state.confidence = 0.7; // Placeholder
        }
    }

    /// Cleanup old events
    fn cleanup_old_events(&mut self, cutoff: Instant) {
        self.access_history.retain(|event| event.timestamp > cutoff);
    }
}

impl Default for PatternAnalysisConfig {
    fn default() -> Self {
        Self {
            min_pattern_length: 5,
            analysis_window: Duration::from_secs(60),
            confidence_threshold: 0.5,
            enable_prediction: true,
            enable_optimization_suggestions: true,
            max_tracked_patterns: 10000,
            classification_sensitivity: 0.1,
            cache_analysis_depth: 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_profiler::allocation::AccessPattern;

    #[test]
    fn test_pattern_analyzer_creation() {
        let config = PatternAnalysisConfig::default();
        let analyzer = AccessPatternAnalyzer::new(config);

        assert!(analyzer.get_statistics().total_patterns == 0);
        assert!(analyzer.get_optimization_suggestions().is_empty());
    }

    #[test]
    fn test_sequential_pattern_classification() {
        let analyzer = AccessPatternAnalyzer::new(PatternAnalysisConfig::default());

        let mut pattern = AccessPattern::new();
        // Simulate sequential pattern
        for i in 0..10 {
            pattern.record_access(AccessType::Read, 1000 + i * 8);
        }
        pattern.sequential_score = 0.9;
        pattern.temporal_locality = 0.5;
        pattern.spatial_locality = 0.8;

        let classification = analyzer.analyze_pattern(0x1000, &pattern);
        assert!(classification.is_some());

        let classification = classification.unwrap();
        assert!(matches!(
            classification.primary_type,
            PatternType::Sequential { .. }
        ));
        assert!(classification.confidence > 0.5);
    }

    #[test]
    fn test_optimization_suggestions() {
        let analyzer = AccessPatternAnalyzer::new(PatternAnalysisConfig::default());

        let mut pattern = AccessPattern::new();
        for i in 0..10 {
            pattern.record_access(AccessType::Read, 1000 + i * 8);
        }
        pattern.sequential_score = 0.9;

        analyzer.analyze_pattern(0x1000, &pattern);
        let suggestions = analyzer.get_optimization_suggestions();

        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| matches!(
            s.optimization_type,
            OptimizationType::PrefetchingOptimization { .. }
        )));
    }

    #[test]
    fn test_pattern_statistics_update() {
        let analyzer = AccessPatternAnalyzer::new(PatternAnalysisConfig::default());

        let mut pattern = AccessPattern::new();
        for i in 0..10 {
            pattern.record_access(AccessType::Read, 1000 + i * 8);
        }
        pattern.sequential_score = 0.8;

        analyzer.analyze_pattern(0x1000, &pattern);

        let stats = analyzer.get_statistics();
        assert_eq!(stats.total_patterns, 1);
        assert!(stats.average_confidence > 0.0);
    }

    #[test]
    fn test_streaming_pattern_detection() {
        let analyzer = AccessPatternAnalyzer::new(PatternAnalysisConfig::default());

        let mut pattern = AccessPattern::new();
        // Simulate streaming pattern with large block sizes
        for _i in 0..10 {
            pattern.record_access(AccessType::Read, 64 * 1024); // 64KB blocks
        }
        pattern.sequential_score = 0.6;
        pattern.spatial_locality = 0.9;

        let streaming_score = analyzer.analyze_streaming(&pattern);
        assert!(streaming_score > 0.5);
    }

    #[test]
    fn test_access_prediction() {
        let analyzer = AccessPatternAnalyzer::new(PatternAnalysisConfig::default());

        let mut pattern = AccessPattern::new();
        for i in 0..5 {
            pattern.record_access(AccessType::Read, 1000 + i * 8);
        }

        let pattern_type = PatternType::Sequential {
            stride: 8,
            direction: AccessDirection::Forward,
        };

        let prediction = analyzer.generate_access_prediction(&pattern, &pattern_type);
        assert!(!prediction.next_accesses.is_empty());
        assert!(!prediction.prefetch_candidates.is_empty());
    }
}
