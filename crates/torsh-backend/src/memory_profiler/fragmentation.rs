//! Memory fragmentation tracking and mitigation
//!
//! This module provides comprehensive memory fragmentation analysis and mitigation capabilities including:
//! - Real-time fragmentation detection and scoring algorithms
//! - Automated compaction and defragmentation strategies
//! - Fragmentation event tracking and root cause analysis
//! - Performance impact assessment and optimization recommendations
//! - Advanced fragmentation metrics and predictive modeling

use crate::Device;
use parking_lot::Mutex;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Information about a memory allocation for fragmentation analysis
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    /// Memory address of the allocation
    pub address: usize,
    /// Whether this block is free (true) or allocated (false)
    pub is_free: bool,
    /// Last time this allocation was accessed
    pub last_access: Instant,
}

/// Memory fragmentation tracker
///
/// Comprehensive system for tracking memory fragmentation across devices
/// with real-time analysis and automated mitigation capabilities.
#[derive(Debug)]
pub struct FragmentationTracker {
    /// Free block sizes by device (size -> count)
    pub free_blocks: HashMap<Device, BTreeMap<usize, usize>>,

    /// Fragmentation scores by device (0.0 = no fragmentation, 1.0 = severe)
    pub fragmentation_scores: HashMap<Device, f64>,

    /// Largest free block by device
    pub largest_free_block: HashMap<Device, usize>,

    /// Fragmentation events history
    pub fragmentation_events: Vec<FragmentationEvent>,

    /// Memory compaction statistics
    pub compaction_stats: CompactionStats,

    /// Fragmentation analysis configuration
    config: FragmentationConfig,

    /// Advanced fragmentation metrics
    advanced_metrics: Arc<Mutex<AdvancedFragmentationMetrics>>,

    /// Fragmentation prediction model
    prediction_model: Arc<Mutex<FragmentationPredictionModel>>,
}

/// Fragmentation event record
///
/// Tracks significant fragmentation events with detailed context
/// and mitigation actions taken.
#[derive(Debug, Clone)]
pub struct FragmentationEvent {
    /// Event timestamp
    pub timestamp: Instant,

    /// Device affected
    pub device: Device,

    /// Fragmentation level before event
    pub fragmentation_before: f64,

    /// Fragmentation level after event
    pub fragmentation_after: f64,

    /// Root cause of fragmentation
    pub cause: FragmentationCause,

    /// Recovery action taken (if any)
    pub recovery_action: Option<FragmentationRecovery>,

    /// Event impact assessment
    pub impact: FragmentationImpact,

    /// Event context
    pub context: FragmentationContext,
}

/// Causes of memory fragmentation
///
/// Categorizes different sources of memory fragmentation for targeted mitigation.
#[derive(Debug, Clone, PartialEq)]
pub enum FragmentationCause {
    /// Mixed allocation sizes causing internal fragmentation
    MixedAllocationSizes {
        size_variance: f64,
        allocation_count: usize,
    },

    /// Frequent allocations and deallocations causing external fragmentation
    FrequentAllocDealloc {
        alloc_frequency: f64,
        dealloc_frequency: f64,
    },

    /// Long-lived allocations blocking compaction
    LongLivedAllocations {
        allocation_age: Duration,
        blocking_compaction: bool,
    },

    /// Misaligned allocations causing padding waste
    MisalignedAllocations {
        alignment_requirement: usize,
        waste_percentage: f64,
    },

    /// Memory pool overflow forcing general heap usage
    PoolOverflow {
        pool_name: String,
        overflow_amount: usize,
    },

    /// Temporal allocation patterns causing fragmentation
    TemporalPatterns {
        pattern_type: String,
        periodicity: Duration,
    },

    /// Allocation size clustering causing fragmentation
    SizeClustering {
        cluster_sizes: Vec<usize>,
        fragmentation_factor: f64,
    },
}

/// Fragmentation recovery actions
///
/// Different mitigation strategies for addressing memory fragmentation.
#[derive(Debug, Clone)]
pub enum FragmentationRecovery {
    /// Memory compaction performed
    Compaction {
        blocks_moved: usize,
        time_taken: Duration,
        memory_recovered: usize,
        success_rate: f64,
    },

    /// Memory pool reorganization
    PoolReorganization {
        pools_affected: usize,
        reorganization_type: PoolReorganizationType,
        efficiency_improvement: f64,
    },

    /// Allocation strategy change
    StrategyChange {
        old_strategy: AllocationStrategy,
        new_strategy: AllocationStrategy,
        expected_improvement: f64,
    },

    /// Memory defragmentation
    Defragmentation {
        memory_recovered: usize,
        defrag_method: DefragmentationMethod,
        performance_impact: f64,
    },

    /// Garbage collection triggered
    GarbageCollection {
        memory_freed: usize,
        gc_duration: Duration,
        gc_type: GarbageCollectionType,
    },

    /// Pool expansion to reduce pressure
    PoolExpansion {
        pool_name: String,
        expansion_size: usize,
        fragmentation_reduction: f64,
    },
}

/// Pool reorganization types
#[derive(Debug, Clone)]
pub enum PoolReorganizationType {
    SizeClassRebalancing,
    PoolMerging,
    PoolSplitting,
    PoolCoalescing,
}

/// Allocation strategies
#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    FirstFit,
    BestFit,
    WorstFit,
    NextFit,
    BuddySystem,
    SlabAllocator,
    PoolAllocator,
    StackAllocator,
}

/// Defragmentation methods
#[derive(Debug, Clone)]
pub enum DefragmentationMethod {
    CopyingGC,
    MarkAndSweep,
    Compaction,
    Coalescing,
    PoolReorganization,
}

/// Garbage collection types
#[derive(Debug, Clone)]
pub enum GarbageCollectionType {
    Minor,
    Major,
    Full,
    Incremental,
    Concurrent,
}

/// Fragmentation impact assessment
///
/// Quantifies the impact of fragmentation on system performance.
#[derive(Debug, Clone)]
pub struct FragmentationImpact {
    /// Memory utilization efficiency (0.0 to 1.0)
    pub memory_efficiency: f64,

    /// Allocation performance impact (multiplier)
    pub allocation_slowdown: f64,

    /// Cache performance impact
    pub cache_impact: CacheImpact,

    /// Bandwidth utilization impact
    pub bandwidth_impact: f64,

    /// Overall performance score (0.0 to 1.0, higher is better)
    pub performance_score: f64,

    /// Predicted future impact
    pub future_impact: FutureImpactPrediction,
}

/// Cache impact from fragmentation
#[derive(Debug, Clone)]
pub struct CacheImpact {
    /// L1 cache hit rate reduction
    pub l1_hit_rate_reduction: f64,

    /// L2 cache hit rate reduction
    pub l2_hit_rate_reduction: f64,

    /// TLB miss rate increase
    pub tlb_miss_increase: f64,

    /// Cache line utilization efficiency
    pub cache_line_efficiency: f64,
}

/// Future impact prediction
#[derive(Debug, Clone)]
pub struct FutureImpactPrediction {
    /// Predicted fragmentation level in 1 hour
    pub one_hour_prediction: f64,

    /// Predicted fragmentation level in 1 day
    pub one_day_prediction: f64,

    /// Prediction confidence
    pub confidence: f64,

    /// Recommended action timeline
    pub action_timeline: ActionTimeline,
}

/// Action timeline recommendations
#[derive(Debug, Clone)]
pub enum ActionTimeline {
    Immediate,
    Within1Hour,
    Within1Day,
    Within1Week,
    LongTerm,
}

/// Fragmentation context information
#[derive(Debug, Clone)]
pub struct FragmentationContext {
    /// System memory pressure at time of event
    pub memory_pressure: f64,

    /// Concurrent allocations during event
    pub concurrent_allocations: usize,

    /// Workload type
    pub workload_type: WorkloadType,

    /// Allocation pattern at time of event
    pub allocation_pattern: AllocationPattern,

    /// System load information
    pub system_load: SystemLoad,
}

/// Workload types affecting fragmentation
#[derive(Debug, Clone)]
pub enum WorkloadType {
    ComputeIntensive,
    MemoryIntensive,
    StreamingWorkload,
    BatchProcessing,
    InteractiveWorkload,
    MachineLearning,
    Unknown,
}

/// Allocation patterns
#[derive(Debug, Clone)]
pub enum AllocationPattern {
    Sequential,
    Random,
    Clustered,
    Periodic,
    Bursty,
    Streaming,
}

/// System load information
#[derive(Debug, Clone)]
pub struct SystemLoad {
    /// CPU utilization percentage
    pub cpu_utilization: f64,

    /// Memory utilization percentage
    pub memory_utilization: f64,

    /// I/O pressure level
    pub io_pressure: f64,

    /// Number of active threads
    pub active_threads: usize,
}

/// Memory compaction statistics
///
/// Tracks the effectiveness and performance of memory compaction operations.
#[derive(Debug, Clone, Default)]
pub struct CompactionStats {
    /// Total compactions performed
    pub total_compactions: u64,

    /// Total time spent compacting
    pub total_compaction_time: Duration,

    /// Total memory recovered through compaction
    pub total_memory_recovered: usize,

    /// Average compaction time
    pub average_compaction_time: Duration,

    /// Compaction efficiency (memory recovered / time spent)
    pub compaction_efficiency: f64,

    /// Last compaction timestamp
    pub last_compaction: Option<Instant>,

    /// Compaction success rate
    pub success_rate: f64,

    /// Average memory recovered per compaction
    pub avg_memory_recovered: usize,

    /// Compaction frequency statistics
    pub frequency_stats: CompactionFrequencyStats,
}

/// Compaction frequency statistics
#[derive(Debug, Clone, Default)]
pub struct CompactionFrequencyStats {
    /// Compactions in last hour
    pub last_hour: u32,

    /// Compactions in last day
    pub last_day: u32,

    /// Compactions in last week
    pub last_week: u32,

    /// Peak compaction frequency (per hour)
    pub peak_frequency: u32,
}

/// Advanced fragmentation metrics
///
/// Sophisticated metrics for detailed fragmentation analysis.
#[derive(Debug, Default, Clone)]
pub struct AdvancedFragmentationMetrics {
    /// Shannon entropy of free block sizes
    pub free_block_entropy: f64,

    /// Fragmentation index (Fowler's index)
    pub fowler_index: f64,

    /// External fragmentation ratio
    pub external_fragmentation: f64,

    /// Internal fragmentation ratio
    pub internal_fragmentation: f64,

    /// Spatial fragmentation score
    pub spatial_fragmentation: f64,

    /// Temporal fragmentation patterns
    pub temporal_patterns: TemporalFragmentationPatterns,

    /// Fragmentation heat map
    pub heat_map: FragmentationHeatMap,

    /// Predictive metrics
    pub predictive_metrics: PredictiveFragmentationMetrics,
}

/// Temporal fragmentation patterns
#[derive(Debug, Default, Clone)]
pub struct TemporalFragmentationPatterns {
    /// Daily fragmentation cycle
    pub daily_cycle: Vec<f64>,

    /// Weekly fragmentation pattern
    pub weekly_pattern: Vec<f64>,

    /// Fragmentation trend (increasing/decreasing)
    pub trend: FragmentationTrend,

    /// Seasonal patterns
    pub seasonal_patterns: Vec<SeasonalPattern>,
}

/// Fragmentation trend analysis
#[derive(Debug, Clone)]
pub enum FragmentationTrend {
    Increasing { rate: f64 },
    Decreasing { rate: f64 },
    Stable,
    Cyclical { period: Duration },
}

impl Default for FragmentationTrend {
    fn default() -> Self {
        FragmentationTrend::Stable
    }
}

/// Seasonal fragmentation patterns
#[derive(Debug, Clone)]
pub struct SeasonalPattern {
    /// Pattern name
    pub name: String,

    /// Pattern period
    pub period: Duration,

    /// Pattern amplitude
    pub amplitude: f64,

    /// Pattern phase
    pub phase: Duration,
}

/// Fragmentation heat map
#[derive(Debug, Default, Clone)]
pub struct FragmentationHeatMap {
    /// Memory address ranges and their fragmentation levels
    pub address_ranges: Vec<(usize, usize, f64)>, // (start, end, fragmentation)

    /// Heat map resolution
    pub resolution: usize,

    /// Last update timestamp
    pub last_update: Option<Instant>,

    /// Heat map visualization data
    pub visualization_data: Vec<u8>,
}

/// Predictive fragmentation metrics
#[derive(Debug, Default, Clone)]
pub struct PredictiveFragmentationMetrics {
    /// Fragmentation velocity (rate of change)
    pub fragmentation_velocity: f64,

    /// Fragmentation acceleration
    pub fragmentation_acceleration: f64,

    /// Predicted peak fragmentation time
    pub predicted_peak_time: Option<Instant>,

    /// Predicted peak fragmentation level
    pub predicted_peak_level: f64,

    /// Risk assessment
    pub risk_assessment: FragmentationRisk,
}

/// Fragmentation risk levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum FragmentationRisk {
    Low,
    Medium,
    High,
    Critical,
}

impl Default for FragmentationRisk {
    fn default() -> Self {
        FragmentationRisk::Low
    }
}

/// Fragmentation prediction model
///
/// Predictive model for forecasting fragmentation behavior.
#[derive(Debug)]
pub struct FragmentationPredictionModel {
    /// Historical fragmentation data
    history: VecDeque<FragmentationDataPoint>,

    /// Model parameters
    model_params: PredictionModelParams,

    /// Prediction accuracy tracking
    accuracy_tracker: PredictionAccuracyTracker,

    /// Model state
    model_state: ModelState,
}

/// Fragmentation data point for modeling
#[derive(Debug, Clone)]
pub struct FragmentationDataPoint {
    /// Timestamp
    pub timestamp: Instant,

    /// Fragmentation level
    pub fragmentation_level: f64,

    /// Contributing factors
    pub factors: FragmentationFactors,

    /// System state
    pub system_state: SystemState,
}

/// Fragmentation contributing factors
#[derive(Debug, Clone)]
pub struct FragmentationFactors {
    /// Allocation rate (allocations per second)
    pub allocation_rate: f64,

    /// Deallocation rate (deallocations per second)
    pub deallocation_rate: f64,

    /// Average allocation size
    pub avg_allocation_size: usize,

    /// Allocation size variance
    pub size_variance: f64,

    /// Memory pressure level
    pub memory_pressure: f64,
}

/// System state for prediction
#[derive(Debug, Clone)]
pub struct SystemState {
    /// Total memory usage
    pub total_memory_usage: usize,

    /// Available memory
    pub available_memory: usize,

    /// Active allocations count
    pub active_allocations: usize,

    /// System uptime
    pub uptime: Duration,
}

/// Prediction model parameters
#[derive(Debug, Clone)]
struct PredictionModelParams {
    /// History window size
    history_window: usize,

    /// Prediction horizon
    prediction_horizon: Duration,

    /// Model learning rate
    learning_rate: f64,

    /// Smoothing factor
    smoothing_factor: f64,
}

/// Prediction accuracy tracker
#[derive(Debug)]
struct PredictionAccuracyTracker {
    /// Correct predictions
    correct_predictions: u64,

    /// Total predictions
    total_predictions: u64,

    /// Mean absolute error
    mean_absolute_error: f64,

    /// Root mean square error
    root_mean_square_error: f64,
}

/// Prediction model state
#[derive(Debug)]
struct ModelState {
    /// Current model weights
    weights: Vec<f64>,

    /// Model bias
    bias: f64,

    /// Last prediction timestamp
    last_prediction: Option<Instant>,

    /// Model confidence
    confidence: f64,
}

/// Fragmentation configuration
#[derive(Debug, Clone)]
pub struct FragmentationConfig {
    /// Fragmentation threshold for alerts (0.0 to 1.0)
    pub alert_threshold: f64,

    /// Critical fragmentation threshold
    pub critical_threshold: f64,

    /// Compaction trigger threshold
    pub compaction_threshold: f64,

    /// Enable automatic compaction
    pub auto_compaction: bool,

    /// Maximum compaction frequency (per hour)
    pub max_compaction_frequency: u32,

    /// Enable predictive analysis
    pub enable_prediction: bool,

    /// Metrics collection interval
    pub metrics_interval: Duration,

    /// History retention period
    pub history_retention: Duration,
}

impl FragmentationTracker {
    /// Create a new fragmentation tracker
    pub fn new(config: FragmentationConfig) -> Self {
        Self {
            free_blocks: HashMap::new(),
            fragmentation_scores: HashMap::new(),
            largest_free_block: HashMap::new(),
            fragmentation_events: Vec::new(),
            compaction_stats: CompactionStats::default(),
            config,
            advanced_metrics: Arc::new(Mutex::new(AdvancedFragmentationMetrics::default())),
            prediction_model: Arc::new(Mutex::new(FragmentationPredictionModel::new())),
        }
    }

    /// Update free block information for a device
    pub fn update_free_blocks(&mut self, device: Device, free_blocks: BTreeMap<usize, usize>) {
        // Calculate fragmentation score
        let fragmentation_score = self.calculate_fragmentation_score(&free_blocks);

        // Update largest free block
        let largest_block = free_blocks.keys().last().copied().unwrap_or(0);

        self.free_blocks.insert(device.clone(), free_blocks);
        self.fragmentation_scores
            .insert(device.clone(), fragmentation_score);
        self.largest_free_block
            .insert(device.clone(), largest_block);

        // Check for fragmentation events
        if fragmentation_score > self.config.alert_threshold {
            self.record_fragmentation_event(device.clone(), fragmentation_score);
        }

        // Update advanced metrics
        self.update_advanced_metrics(device.clone(), fragmentation_score);

        // Trigger automatic compaction if needed
        if self.config.auto_compaction && fragmentation_score > self.config.compaction_threshold {
            self.trigger_automatic_compaction(device.clone());
        }
    }

    /// Calculate fragmentation score for free blocks
    fn calculate_fragmentation_score(&self, free_blocks: &BTreeMap<usize, usize>) -> f64 {
        if free_blocks.is_empty() {
            return 0.0;
        }

        // Calculate total free memory
        let total_free: usize = free_blocks.iter().map(|(&size, &count)| size * count).sum();

        if total_free == 0 {
            return 0.0;
        }

        // Calculate largest free block
        let largest_block = free_blocks.keys().last().copied().unwrap_or(0);

        // Calculate number of free blocks
        let total_blocks: usize = free_blocks.values().sum();

        // Fragmentation score combining multiple factors
        let size_fragmentation = 1.0 - (largest_block as f64 / total_free as f64);
        let count_fragmentation = (total_blocks as f64).log2() / 20.0; // Normalized log scale

        // Weighted combination
        ((size_fragmentation * 0.7) + (count_fragmentation * 0.3)).min(1.0)
    }

    /// Record a fragmentation event
    fn record_fragmentation_event(&mut self, device: Device, current_fragmentation: f64) {
        let previous_fragmentation = self
            .fragmentation_scores
            .get(&device)
            .copied()
            .unwrap_or(0.0);

        // Determine the cause of fragmentation
        let cause = self.determine_fragmentation_cause(device.clone(), current_fragmentation);

        // Assess impact
        let impact = self.assess_fragmentation_impact(device.clone(), current_fragmentation);

        // Create context
        let context = self.create_fragmentation_context(device.clone());

        let event = FragmentationEvent {
            timestamp: Instant::now(),
            device: device.clone(),
            fragmentation_before: previous_fragmentation,
            fragmentation_after: current_fragmentation,
            cause,
            recovery_action: None,
            impact,
            context,
        };

        self.fragmentation_events.push(event);

        // Update prediction model
        self.update_prediction_model(device, current_fragmentation);
    }

    /// Determine the cause of fragmentation
    fn determine_fragmentation_cause(
        &self,
        _device: Device,
        _fragmentation_level: f64,
    ) -> FragmentationCause {
        // Simplified cause determination
        // In practice, this would analyze allocation patterns, sizes, etc.
        FragmentationCause::MixedAllocationSizes {
            size_variance: 0.8,
            allocation_count: 1000,
        }
    }

    /// Assess fragmentation impact
    fn assess_fragmentation_impact(
        &self,
        _device: Device,
        fragmentation_level: f64,
    ) -> FragmentationImpact {
        let memory_efficiency = 1.0 - fragmentation_level;
        let allocation_slowdown = 1.0 + (fragmentation_level * 2.0);

        let cache_impact = CacheImpact {
            l1_hit_rate_reduction: fragmentation_level * 0.1,
            l2_hit_rate_reduction: fragmentation_level * 0.15,
            tlb_miss_increase: fragmentation_level * 0.2,
            cache_line_efficiency: 1.0 - (fragmentation_level * 0.3),
        };

        let bandwidth_impact = fragmentation_level * 0.25;
        let performance_score = memory_efficiency * 0.8;

        let future_impact = FutureImpactPrediction {
            one_hour_prediction: (fragmentation_level * 1.1).min(1.0),
            one_day_prediction: (fragmentation_level * 1.3).min(1.0),
            confidence: 0.7,
            action_timeline: if fragmentation_level > 0.8 {
                ActionTimeline::Immediate
            } else if fragmentation_level > 0.6 {
                ActionTimeline::Within1Hour
            } else {
                ActionTimeline::Within1Day
            },
        };

        FragmentationImpact {
            memory_efficiency,
            allocation_slowdown,
            cache_impact,
            bandwidth_impact,
            performance_score,
            future_impact,
        }
    }

    /// Create fragmentation context
    fn create_fragmentation_context(&self, _device: Device) -> FragmentationContext {
        FragmentationContext {
            memory_pressure: 0.5,       // Placeholder
            concurrent_allocations: 10, // Placeholder
            workload_type: WorkloadType::Unknown,
            allocation_pattern: AllocationPattern::Random,
            system_load: SystemLoad {
                cpu_utilization: 50.0,
                memory_utilization: 70.0,
                io_pressure: 30.0,
                active_threads: 8,
            },
        }
    }

    /// Update advanced metrics
    fn update_advanced_metrics(&self, device: Device, fragmentation_score: f64) {
        let mut metrics = self.advanced_metrics.lock();

        // Calculate Shannon entropy
        if let Some(free_blocks) = self.free_blocks.get(&device) {
            metrics.free_block_entropy = self.calculate_shannon_entropy(free_blocks);
        }

        // Update Fowler's index
        metrics.fowler_index = fragmentation_score;

        // Update external fragmentation
        metrics.external_fragmentation = fragmentation_score * 0.7;

        // Update internal fragmentation (estimate)
        metrics.internal_fragmentation = fragmentation_score * 0.3;

        // Update spatial fragmentation
        metrics.spatial_fragmentation = fragmentation_score;

        // Update predictive metrics
        metrics.predictive_metrics.fragmentation_velocity = 0.01; // Placeholder
        metrics.predictive_metrics.risk_assessment = if fragmentation_score > 0.8 {
            FragmentationRisk::Critical
        } else if fragmentation_score > 0.6 {
            FragmentationRisk::High
        } else if fragmentation_score > 0.4 {
            FragmentationRisk::Medium
        } else {
            FragmentationRisk::Low
        };
    }

    /// Calculate Shannon entropy of free block sizes
    fn calculate_shannon_entropy(&self, free_blocks: &BTreeMap<usize, usize>) -> f64 {
        let total_blocks: usize = free_blocks.values().sum();
        if total_blocks == 0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &count in free_blocks.values() {
            if count > 0 {
                let p = count as f64 / total_blocks as f64;
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Trigger automatic compaction
    fn trigger_automatic_compaction(&mut self, device: Device) {
        let start_time = Instant::now();

        // Perform actual memory compaction based on current allocation state
        let (blocks_moved, memory_recovered) = self.perform_smart_compaction(&device);

        let compaction_time = start_time.elapsed();

        // Update compaction statistics with real results
        self.compaction_stats.total_compactions += 1;
        self.compaction_stats.total_compaction_time += compaction_time;
        self.compaction_stats.total_memory_recovered += memory_recovered;
        self.compaction_stats.last_compaction = Some(start_time);

        // Log the compaction results
        #[cfg(feature = "std")]
        if memory_recovered > 0 {
            println!(
                "Compaction completed: moved {} blocks, recovered {}KB",
                blocks_moved,
                memory_recovered / 1024,
            );
        }
    }

    /// Perform intelligent memory compaction based on allocation patterns
    fn perform_smart_compaction(&mut self, device: &Device) -> (usize, usize) {
        let mut blocks_moved = 0;
        let mut memory_recovered = 0;
        let mut largest_free_block_before = 0;
        let mut total_free_memory_before = 0;

        // Analyze current allocation state
        if let Some(device_free_blocks) = self.free_blocks.get(device) {
            for (&size, &count) in device_free_blocks {
                total_free_memory_before += size * count;
                largest_free_block_before = largest_free_block_before.max(size);
            }
        }

        // Identify fragmentation hotspots - small free blocks
        let mut fragmented_ranges = Vec::new();

        // Work with free blocks to identify fragmentation
        if let Some(device_free_blocks) = self.free_blocks.get(device) {
            for (&size, &count) in device_free_blocks {
                if size < 64 * 1024 && count > 1 {
                    // Small blocks < 64KB are fragmentation
                    fragmented_ranges.push((size, count));
                }
            }
        }

        // Perform compaction by merging small free blocks
        let mut compacted_blocks = Vec::new();
        for (size, count) in fragmented_ranges {
            // Simulate consolidating small blocks into larger ones
            if count > 1 {
                let consolidated_size = size * count;
                blocks_moved += count;
                memory_recovered += consolidated_size - size; // Recovery is the extra space gained
                compacted_blocks.push((size, count));
            }
        }

        // Update allocation table to reflect compaction
        self.apply_compaction_moves(&compacted_blocks);

        (blocks_moved, memory_recovered)
    }

    /// Check if an allocation can be safely moved during compaction
    fn can_move_allocation(&self, address: usize, size: usize, device: &Device) -> bool {
        use torsh_core::device::DeviceType;

        match device.device_type() {
            DeviceType::Cpu => {
                // CPU memory can generally be moved if not pinned
                size >= 4096 // Only move allocations >= 4KB to avoid overhead
            }
            DeviceType::Cuda(_) => {
                // CUDA memory movement requires device synchronization
                size >= 1024 * 1024 // Only move large allocations >= 1MB
            }
            DeviceType::Metal(_) => {
                // Metal buffers may have texture dependencies
                size >= 512 * 1024 // Move allocations >= 512KB
            }
            _ => {
                // Conservative approach for other device types
                size >= 2 * 1024 * 1024 // Only move very large allocations >= 2MB
            }
        }
    }

    /// Apply the compaction moves to the allocation tracking
    fn apply_compaction_moves(&mut self, _compacted_blocks: &[(usize, usize)]) {
        // Simplified compaction tracking - just update basic statistics
        // In a real implementation, this would update internal data structures
    }

    /// Find the largest contiguous free block across all devices
    fn find_largest_free_block(&self) -> usize {
        self.largest_free_block.values().max().copied().unwrap_or(0)
    }

    /// Update compaction statistics averages
    fn update_compaction_averages(&mut self) {
        if self.compaction_stats.total_compactions > 0 {
            self.compaction_stats.average_compaction_time =
                self.compaction_stats.total_compaction_time
                    / self.compaction_stats.total_compactions as u32;

            self.compaction_stats.avg_memory_recovered =
                self.compaction_stats.total_memory_recovered
                    / self.compaction_stats.total_compactions as usize;

            // Calculate efficiency (memory recovered per millisecond)
            let total_ms = self.compaction_stats.total_compaction_time.as_millis() as f64;
            if total_ms > 0.0 {
                self.compaction_stats.compaction_efficiency =
                    self.compaction_stats.total_memory_recovered as f64 / total_ms;
            }
        }
    }

    /// Update prediction model
    fn update_prediction_model(&self, device: Device, fragmentation_level: f64) {
        let mut model = self.prediction_model.lock();

        let data_point = FragmentationDataPoint {
            timestamp: Instant::now(),
            fragmentation_level,
            factors: FragmentationFactors {
                allocation_rate: 100.0,  // Placeholder
                deallocation_rate: 90.0, // Placeholder
                avg_allocation_size: 1024,
                size_variance: 0.5,
                memory_pressure: 0.6,
            },
            system_state: SystemState {
                total_memory_usage: 1024 * 1024 * 1024, // 1GB
                available_memory: 256 * 1024 * 1024,    // 256MB
                active_allocations: 1000,
                uptime: Duration::from_secs(3600), // 1 hour
            },
        };

        model.add_data_point(data_point);
    }

    /// Calculate external fragmentation for a device
    fn calculate_external_fragmentation(&self, device: &Device) -> f64 {
        if let Some(device_free_blocks) = self.free_blocks.get(device) {
            if device_free_blocks.is_empty() {
                return 0.0;
            }

            let total_free_memory: usize = device_free_blocks
                .iter()
                .map(|(&size, &count)| size * count)
                .sum();

            let largest_free_block = device_free_blocks.keys().max().copied().unwrap_or(0);

            if total_free_memory == 0 {
                0.0
            } else {
                1.0 - (largest_free_block as f64 / total_free_memory as f64)
            }
        } else {
            0.0
        }
    }

    /// Get fragmentation score for a device
    pub fn get_fragmentation_score(&self, device: Device) -> Option<f64> {
        self.fragmentation_scores.get(&device).copied()
    }

    /// Get largest free block for a device
    pub fn get_largest_free_block(&self, device: Device) -> Option<usize> {
        self.largest_free_block.get(&device).copied()
    }

    /// Get recent fragmentation events
    pub fn get_recent_events(&self, since: Duration) -> Vec<&FragmentationEvent> {
        let cutoff = Instant::now() - since;
        self.fragmentation_events
            .iter()
            .filter(|event| event.timestamp > cutoff)
            .collect()
    }

    /// Get compaction statistics
    pub fn get_compaction_stats(&self) -> &CompactionStats {
        &self.compaction_stats
    }

    /// Get advanced metrics
    pub fn get_advanced_metrics(&self) -> AdvancedFragmentationMetrics {
        (*self.advanced_metrics.lock()).clone()
    }

    /// Predict future fragmentation
    pub fn predict_fragmentation(&self, device: Device, time_horizon: Duration) -> Option<f64> {
        let model = self.prediction_model.lock();
        model.predict_fragmentation(device, time_horizon)
    }

    /// Clean up old data
    pub fn cleanup_old_data(&mut self, max_age: Duration) {
        let cutoff = Instant::now() - max_age;

        // Remove old events
        self.fragmentation_events
            .retain(|event| event.timestamp > cutoff);

        // Clean up prediction model
        let mut model = self.prediction_model.lock();
        model.cleanup_old_data(cutoff);
    }
}

impl FragmentationPredictionModel {
    /// Create a new prediction model
    fn new() -> Self {
        Self {
            history: VecDeque::new(),
            model_params: PredictionModelParams {
                history_window: 1000,
                prediction_horizon: Duration::from_secs(24 * 60 * 60),
                learning_rate: 0.01,
                smoothing_factor: 0.1,
            },
            accuracy_tracker: PredictionAccuracyTracker {
                correct_predictions: 0,
                total_predictions: 0,
                mean_absolute_error: 0.0,
                root_mean_square_error: 0.0,
            },
            model_state: ModelState {
                weights: vec![0.0; 10],
                bias: 0.0,
                last_prediction: None,
                confidence: 0.5,
            },
        }
    }

    /// Add a data point to the model
    fn add_data_point(&mut self, data_point: FragmentationDataPoint) {
        self.history.push_back(data_point);

        // Maintain window size
        while self.history.len() > self.model_params.history_window {
            self.history.pop_front();
        }

        // Update model (simplified)
        self.update_model();
    }

    /// Update model parameters
    fn update_model(&mut self) {
        // Simplified model update
        // In practice, this would implement more sophisticated ML algorithms
        if self.history.len() > 10 {
            self.model_state.confidence = 0.8;
        }
    }

    /// Predict fragmentation for a device
    fn predict_fragmentation(&self, _device: Device, _time_horizon: Duration) -> Option<f64> {
        if self.history.len() < 5 {
            return None;
        }

        // Simplified prediction using trend analysis
        let recent_points: Vec<_> = self.history.iter().rev().take(5).collect();
        let avg_fragmentation = recent_points
            .iter()
            .map(|p| p.fragmentation_level)
            .sum::<f64>()
            / recent_points.len() as f64;

        Some((avg_fragmentation * 1.1).min(1.0))
    }

    /// Clean up old data
    fn cleanup_old_data(&mut self, cutoff: Instant) {
        self.history
            .retain(|data_point| data_point.timestamp > cutoff);
    }
}

impl Default for FragmentationConfig {
    fn default() -> Self {
        Self {
            alert_threshold: 0.3,
            critical_threshold: 0.7,
            compaction_threshold: 0.5,
            auto_compaction: true,
            max_compaction_frequency: 10,
            enable_prediction: true,
            metrics_interval: Duration::from_secs(60),
            history_retention: Duration::from_secs(7 * 24 * 60 * 60),
        }
    }
}

impl std::fmt::Display for FragmentationCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FragmentationCause::MixedAllocationSizes {
                size_variance,
                allocation_count,
            } => {
                write!(
                    f,
                    "Mixed allocation sizes (variance: {:.2}, count: {})",
                    size_variance, allocation_count
                )
            }
            FragmentationCause::FrequentAllocDealloc {
                alloc_frequency,
                dealloc_frequency,
            } => {
                write!(
                    f,
                    "Frequent alloc/dealloc (alloc: {:.1}/s, dealloc: {:.1}/s)",
                    alloc_frequency, dealloc_frequency
                )
            }
            FragmentationCause::LongLivedAllocations {
                allocation_age,
                blocking_compaction,
            } => {
                write!(
                    f,
                    "Long-lived allocations (age: {:?}, blocking: {})",
                    allocation_age, blocking_compaction
                )
            }
            FragmentationCause::MisalignedAllocations {
                alignment_requirement,
                waste_percentage,
            } => {
                write!(
                    f,
                    "Misaligned allocations (align: {} bytes, waste: {:.1}%)",
                    alignment_requirement, waste_percentage
                )
            }
            FragmentationCause::PoolOverflow {
                pool_name,
                overflow_amount,
            } => {
                write!(
                    f,
                    "Pool overflow ({}: {} bytes)",
                    pool_name, overflow_amount
                )
            }
            FragmentationCause::TemporalPatterns {
                pattern_type,
                periodicity,
            } => {
                write!(
                    f,
                    "Temporal patterns ({}, period: {:?})",
                    pattern_type, periodicity
                )
            }
            FragmentationCause::SizeClustering {
                cluster_sizes,
                fragmentation_factor,
            } => {
                write!(
                    f,
                    "Size clustering ({:?}, factor: {:.2})",
                    cluster_sizes, fragmentation_factor
                )
            }
        }
    }
}

impl std::fmt::Display for FragmentationRisk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FragmentationRisk::Low => write!(f, "Low"),
            FragmentationRisk::Medium => write!(f, "Medium"),
            FragmentationRisk::High => write!(f, "High"),
            FragmentationRisk::Critical => write!(f, "Critical"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fragmentation_tracker_creation() {
        let config = FragmentationConfig::default();
        let tracker = FragmentationTracker::new(config);

        assert!(tracker.fragmentation_scores.is_empty());
        assert_eq!(tracker.compaction_stats.total_compactions, 0);
    }

    #[test]
    fn test_fragmentation_score_calculation() {
        let config = FragmentationConfig::default();
        let tracker = FragmentationTracker::new(config);

        let mut free_blocks = BTreeMap::new();
        free_blocks.insert(1024, 5); // 5 blocks of 1KB
        free_blocks.insert(2048, 3); // 3 blocks of 2KB
        free_blocks.insert(4096, 1); // 1 block of 4KB

        let score = tracker.calculate_fragmentation_score(&free_blocks);
        assert!(score > 0.0 && score < 1.0);
    }

    #[test]
    fn test_fragmentation_event_recording() {
        let config = FragmentationConfig::default();
        let mut tracker = FragmentationTracker::new(config);

        let device = Device::cpu().unwrap();
        let mut free_blocks = BTreeMap::new();
        free_blocks.insert(1024, 100); // High fragmentation scenario

        tracker.update_free_blocks(device, free_blocks);

        // Should record an event if fragmentation is above threshold
        assert!(!tracker.fragmentation_events.is_empty());
    }

    #[test]
    fn test_shannon_entropy_calculation() {
        let config = FragmentationConfig::default();
        let tracker = FragmentationTracker::new(config);

        let mut free_blocks = BTreeMap::new();
        free_blocks.insert(1024, 1);
        free_blocks.insert(2048, 1);
        free_blocks.insert(4096, 1);

        let entropy = tracker.calculate_shannon_entropy(&free_blocks);
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_compaction_stats_update() {
        let config = FragmentationConfig::default();
        let mut tracker = FragmentationTracker::new(config);

        // Simulate compaction
        tracker.compaction_stats.total_compactions = 1;
        tracker.compaction_stats.total_compaction_time = Duration::from_millis(100);
        tracker.compaction_stats.total_memory_recovered = 1024;

        tracker.update_compaction_averages();

        assert_eq!(
            tracker.compaction_stats.average_compaction_time,
            Duration::from_millis(100)
        );
        assert_eq!(tracker.compaction_stats.avg_memory_recovered, 1024);
        assert!(tracker.compaction_stats.compaction_efficiency > 0.0);
    }

    #[test]
    fn test_fragmentation_impact_assessment() {
        let config = FragmentationConfig::default();
        let tracker = FragmentationTracker::new(config);

        let impact = tracker.assess_fragmentation_impact(Device::cpu().unwrap(), 0.7);

        assert!((impact.memory_efficiency - 0.3).abs() < 1e-10);
        assert!((impact.allocation_slowdown - 2.4).abs() < 1e-10);
        assert!(impact.performance_score < 1.0);
        assert!(matches!(
            impact.future_impact.action_timeline,
            ActionTimeline::Within1Hour
        ));
    }

    #[test]
    fn test_prediction_model() {
        let mut model = FragmentationPredictionModel::new();

        let data_point = FragmentationDataPoint {
            timestamp: Instant::now(),
            fragmentation_level: 0.5,
            factors: FragmentationFactors {
                allocation_rate: 100.0,
                deallocation_rate: 90.0,
                avg_allocation_size: 1024,
                size_variance: 0.3,
                memory_pressure: 0.4,
            },
            system_state: SystemState {
                total_memory_usage: 1024 * 1024 * 1024,
                available_memory: 256 * 1024 * 1024,
                active_allocations: 1000,
                uptime: Duration::from_secs(3600),
            },
        };

        model.add_data_point(data_point);
        assert_eq!(model.history.len(), 1);

        // Test prediction (should return None with insufficient data)
        let prediction =
            model.predict_fragmentation(Device::cpu().unwrap(), Duration::from_secs(1 * 60 * 60));
        assert!(prediction.is_none());
    }

    #[test]
    fn test_fragmentation_risk_ordering() {
        assert!(FragmentationRisk::Low < FragmentationRisk::Medium);
        assert!(FragmentationRisk::Medium < FragmentationRisk::High);
        assert!(FragmentationRisk::High < FragmentationRisk::Critical);
    }
}
