//! Allocator statistics and performance metrics for SciRS2 integration
//!
//! This module contains comprehensive statistics tracking for SciRS2 allocators,
//! including performance metrics, advanced analytics, and profiling data.

use super::config::{
    ConfigRecommendation, FragmentationTrend, OptimizationOpportunity, PerformanceTier,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// SciRS2 allocator statistics
///
/// Comprehensive statistics for individual SciRS2 allocators with performance
/// metrics and optimization insights.
#[derive(Debug, Clone)]
pub struct ScirS2AllocatorStats {
    /// Allocator name
    pub name: String,

    /// Total allocations performed
    pub total_allocations: u64,

    /// Total deallocations performed
    pub total_deallocations: u64,

    /// Current allocated bytes
    pub current_allocated: usize,

    /// Peak allocated bytes
    pub peak_allocated: usize,

    /// Allocation failures
    pub allocation_failures: u64,

    /// Average allocation time
    pub avg_allocation_time: Duration,

    /// Memory efficiency (0.0 to 1.0)
    pub memory_efficiency: f64,

    /// Advanced allocator metrics
    pub advanced_metrics: AllocatorAdvancedMetrics,

    /// Performance characteristics
    pub performance_profile: AllocatorPerformanceProfile,

    /// Last update timestamp
    pub last_update: Instant,
}

/// Advanced allocator metrics
#[derive(Debug, Clone)]
pub struct AllocatorAdvancedMetrics {
    /// Allocation size distribution
    pub size_distribution: SizeDistribution,

    /// Allocation latency percentiles
    pub latency_percentiles: LatencyPercentiles,

    /// Fragmentation metrics
    pub fragmentation_metrics: AllocatorFragmentationMetrics,

    /// Cache performance
    pub cache_performance: AllocatorCachePerformance,

    /// Thread contention metrics
    pub contention_metrics: ContentionMetrics,
}

/// Size distribution statistics
#[derive(Debug, Clone)]
pub struct SizeDistribution {
    /// Small allocations (< 1KB)
    pub small_count: u64,

    /// Medium allocations (1KB - 1MB)
    pub medium_count: u64,

    /// Large allocations (> 1MB)
    pub large_count: u64,

    /// Average allocation size
    pub average_size: usize,

    /// Size variance
    pub size_variance: f64,

    /// Most common allocation sizes
    pub common_sizes: Vec<(usize, u64)>,
}

/// Allocation latency percentiles
#[derive(Debug, Clone)]
pub struct LatencyPercentiles {
    /// 50th percentile (median)
    pub p50: Duration,

    /// 90th percentile
    pub p90: Duration,

    /// 95th percentile
    pub p95: Duration,

    /// 99th percentile
    pub p99: Duration,

    /// 99.9th percentile
    pub p999: Duration,

    /// Maximum latency observed
    pub max: Duration,
}

/// Allocator-specific fragmentation metrics
#[derive(Debug, Clone)]
pub struct AllocatorFragmentationMetrics {
    /// Internal fragmentation ratio
    pub internal_fragmentation: f64,

    /// External fragmentation ratio
    pub external_fragmentation: f64,

    /// Free block count
    pub free_block_count: usize,

    /// Largest free block size
    pub largest_free_block: usize,

    /// Fragmentation trend
    pub fragmentation_trend: FragmentationTrend,
}

/// Allocator cache performance
#[derive(Debug, Clone)]
pub struct AllocatorCachePerformance {
    /// Allocation cache hit rate
    pub allocation_cache_hit_rate: f64,

    /// Free list efficiency
    pub free_list_efficiency: f64,

    /// Cache warming effectiveness
    pub cache_warming_effectiveness: f64,

    /// Cache pollution level
    pub cache_pollution_level: f64,
}

/// Thread contention metrics
#[derive(Debug, Clone)]
pub struct ContentionMetrics {
    /// Lock contention time (total)
    pub lock_contention_time: Duration,

    /// Average contention per allocation
    pub avg_contention_per_alloc: Duration,

    /// Peak concurrent allocations
    pub peak_concurrent_allocations: usize,

    /// Contention hotspots
    pub contention_hotspots: Vec<String>,
}

/// Allocator performance profile
#[derive(Debug, Clone)]
pub struct AllocatorPerformanceProfile {
    /// Performance tier (High/Medium/Low)
    pub performance_tier: PerformanceTier,

    /// Optimization opportunities
    pub optimization_opportunities: Vec<OptimizationOpportunity>,

    /// Recommended configuration changes
    pub recommended_config: Vec<ConfigRecommendation>,

    /// Performance score (0.0 to 1.0)
    pub performance_score: f64,
}

/// Performance snapshot
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Timestamp
    pub timestamp: Instant,

    /// Allocation latency
    pub allocation_latency: Duration,

    /// Memory bandwidth utilization
    pub bandwidth_utilization: f64,

    /// Cache hit rate
    pub cache_hit_rate: f64,

    /// CPU utilization
    pub cpu_utilization: f64,
}

/// Memory state snapshot
#[derive(Debug, Clone)]
pub struct MemoryStateSnapshot {
    /// Total allocated memory
    pub total_allocated: usize,

    /// Available memory
    pub available_memory: usize,

    /// Fragmentation level
    pub fragmentation_level: f64,

    /// Active allocations count
    pub active_allocations: usize,
}

/// Allocation usage statistics
#[derive(Debug, Clone)]
pub struct AllocationUsageStats {
    /// Total accesses
    pub total_accesses: u64,

    /// Read/write ratio
    pub read_write_ratio: f64,

    /// Access pattern
    pub access_pattern: String,

    /// Cache efficiency
    pub cache_efficiency: f64,
}

impl ScirS2AllocatorStats {
    /// Create new allocator statistics
    pub fn new(name: String) -> Self {
        Self {
            name,
            total_allocations: 0,
            total_deallocations: 0,
            current_allocated: 0,
            peak_allocated: 0,
            allocation_failures: 0,
            avg_allocation_time: Duration::from_secs(0),
            memory_efficiency: 1.0,
            advanced_metrics: AllocatorAdvancedMetrics::default(),
            performance_profile: AllocatorPerformanceProfile::default(),
            last_update: Instant::now(),
        }
    }

    /// Update allocation statistics
    pub fn record_allocation(&mut self, size: usize, duration: Duration) {
        self.total_allocations += 1;
        self.current_allocated += size;
        self.peak_allocated = self.peak_allocated.max(self.current_allocated);

        // Update average allocation time (simple moving average)
        let new_avg = if self.total_allocations == 1 {
            duration
        } else {
            let total_time =
                self.avg_allocation_time * (self.total_allocations - 1) as u32 + duration;
            total_time / self.total_allocations as u32
        };
        self.avg_allocation_time = new_avg;

        // Update size distribution
        self.advanced_metrics
            .size_distribution
            .record_allocation(size);

        // Update latency percentiles
        self.advanced_metrics
            .latency_percentiles
            .record_latency(duration);

        self.last_update = Instant::now();
    }

    /// Record deallocation
    pub fn record_deallocation(&mut self, size: usize) {
        self.total_deallocations += 1;
        self.current_allocated = self.current_allocated.saturating_sub(size);

        // Update memory efficiency
        self.update_memory_efficiency();

        self.last_update = Instant::now();
    }

    /// Record allocation failure
    pub fn record_allocation_failure(&mut self) {
        self.allocation_failures += 1;
        self.last_update = Instant::now();
    }

    /// Update memory efficiency based on current state
    pub fn update_memory_efficiency(&mut self) {
        if self.peak_allocated > 0 {
            self.memory_efficiency = self.current_allocated as f64 / self.peak_allocated as f64;
        }
    }

    /// Get allocation rate (allocations per second)
    pub fn allocation_rate(&self) -> f64 {
        let elapsed = self.last_update.duration_since(
            Instant::now() - Duration::from_secs(60), // Assume tracking started 60 seconds ago
        );
        if elapsed.as_secs() > 0 {
            self.total_allocations as f64 / elapsed.as_secs() as f64
        } else {
            0.0
        }
    }

    /// Get deallocation rate (deallocations per second)
    pub fn deallocation_rate(&self) -> f64 {
        let elapsed = self.last_update.duration_since(
            Instant::now() - Duration::from_secs(60), // Assume tracking started 60 seconds ago
        );
        if elapsed.as_secs() > 0 {
            self.total_deallocations as f64 / elapsed.as_secs() as f64
        } else {
            0.0
        }
    }

    /// Calculate failure rate
    pub fn failure_rate(&self) -> f64 {
        if self.total_allocations > 0 {
            self.allocation_failures as f64 / self.total_allocations as f64
        } else {
            0.0
        }
    }

    /// Check if allocator is healthy
    pub fn is_healthy(&self) -> bool {
        self.failure_rate() < 0.01 && // Less than 1% failure rate
        self.memory_efficiency > 0.7 && // At least 70% memory efficiency
        self.avg_allocation_time < Duration::from_millis(1) // Less than 1ms average allocation time
    }
}

impl Default for AllocatorAdvancedMetrics {
    fn default() -> Self {
        Self {
            size_distribution: SizeDistribution::default(),
            latency_percentiles: LatencyPercentiles::default(),
            fragmentation_metrics: AllocatorFragmentationMetrics::default(),
            cache_performance: AllocatorCachePerformance::default(),
            contention_metrics: ContentionMetrics::default(),
        }
    }
}

impl Default for SizeDistribution {
    fn default() -> Self {
        Self {
            small_count: 0,
            medium_count: 0,
            large_count: 0,
            average_size: 0,
            size_variance: 0.0,
            common_sizes: Vec::new(),
        }
    }
}

impl SizeDistribution {
    /// Record a new allocation size
    pub fn record_allocation(&mut self, size: usize) {
        match size {
            s if s < 1024 => self.small_count += 1,
            s if s <= 1024 * 1024 => self.medium_count += 1,
            _ => self.large_count += 1,
        }

        // Update average size
        let total_allocations = self.small_count + self.medium_count + self.large_count;
        if total_allocations == 1 {
            self.average_size = size;
        } else {
            self.average_size = (self.average_size * (total_allocations - 1) as usize + size)
                / total_allocations as usize;
        }
    }
}

impl Default for LatencyPercentiles {
    fn default() -> Self {
        Self {
            p50: Duration::from_secs(0),
            p90: Duration::from_secs(0),
            p95: Duration::from_secs(0),
            p99: Duration::from_secs(0),
            p999: Duration::from_secs(0),
            max: Duration::from_secs(0),
        }
    }
}

impl LatencyPercentiles {
    /// Record a new latency measurement
    pub fn record_latency(&mut self, latency: Duration) {
        // Simplified percentile update - in a real implementation,
        // you'd use a proper percentile data structure like t-digest
        self.max = self.max.max(latency);

        // Update percentiles (simplified)
        if self.p50 == Duration::from_secs(0) || latency < self.p50 {
            self.p50 = latency;
        }
    }
}

impl Default for AllocatorFragmentationMetrics {
    fn default() -> Self {
        Self {
            internal_fragmentation: 0.0,
            external_fragmentation: 0.0,
            free_block_count: 0,
            largest_free_block: 0,
            fragmentation_trend: FragmentationTrend::Unknown,
        }
    }
}

impl Default for AllocatorCachePerformance {
    fn default() -> Self {
        Self {
            allocation_cache_hit_rate: 0.0,
            free_list_efficiency: 0.0,
            cache_warming_effectiveness: 0.0,
            cache_pollution_level: 0.0,
        }
    }
}

impl Default for ContentionMetrics {
    fn default() -> Self {
        Self {
            lock_contention_time: Duration::from_secs(0),
            avg_contention_per_alloc: Duration::from_secs(0),
            peak_concurrent_allocations: 0,
            contention_hotspots: Vec::new(),
        }
    }
}

impl Default for AllocatorPerformanceProfile {
    fn default() -> Self {
        Self {
            performance_tier: PerformanceTier::Medium,
            optimization_opportunities: Vec::new(),
            recommended_config: Vec::new(),
            performance_score: 0.5,
        }
    }
}

impl PerformanceSnapshot {
    /// Create a new performance snapshot
    pub fn new() -> Self {
        Self {
            timestamp: Instant::now(),
            allocation_latency: Duration::from_secs(0),
            bandwidth_utilization: 0.0,
            cache_hit_rate: 0.0,
            cpu_utilization: 0.0,
        }
    }

    /// Update snapshot with new measurements
    pub fn update(&mut self, latency: Duration, bandwidth: f64, cache_rate: f64, cpu: f64) {
        self.timestamp = Instant::now();
        self.allocation_latency = latency;
        self.bandwidth_utilization = bandwidth;
        self.cache_hit_rate = cache_rate;
        self.cpu_utilization = cpu;
    }
}

impl MemoryStateSnapshot {
    /// Create a new memory state snapshot
    pub fn new() -> Self {
        Self {
            total_allocated: 0,
            available_memory: 0,
            fragmentation_level: 0.0,
            active_allocations: 0,
        }
    }

    /// Update snapshot with current memory state
    pub fn update(
        &mut self,
        allocated: usize,
        available: usize,
        fragmentation: f64,
        active: usize,
    ) {
        self.total_allocated = allocated;
        self.available_memory = available;
        self.fragmentation_level = fragmentation;
        self.active_allocations = active;
    }

    /// Calculate memory pressure (0.0 to 1.0)
    pub fn memory_pressure(&self) -> f64 {
        if self.total_allocated + self.available_memory == 0 {
            0.0
        } else {
            self.total_allocated as f64 / (self.total_allocated + self.available_memory) as f64
        }
    }
}

/// Statistics aggregator for multiple allocators
pub struct AllocatorStatsAggregator {
    stats: HashMap<String, ScirS2AllocatorStats>,
}

impl AllocatorStatsAggregator {
    /// Create new aggregator
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
        }
    }

    /// Add or update allocator statistics
    pub fn update_stats(&mut self, stats: ScirS2AllocatorStats) {
        self.stats.insert(stats.name.clone(), stats);
    }

    /// Get statistics for a specific allocator
    pub fn get_stats(&self, name: &str) -> Option<&ScirS2AllocatorStats> {
        self.stats.get(name)
    }

    /// Get all statistics
    pub fn get_all_stats(&self) -> &HashMap<String, ScirS2AllocatorStats> {
        &self.stats
    }

    /// Calculate aggregate metrics
    pub fn calculate_aggregate_metrics(&self) -> AggregateMetrics {
        let mut total_allocations = 0;
        let mut total_deallocations = 0;
        let mut total_allocated = 0;
        let mut total_failures = 0;
        let mut efficiency_sum = 0.0;

        for stats in self.stats.values() {
            total_allocations += stats.total_allocations;
            total_deallocations += stats.total_deallocations;
            total_allocated += stats.current_allocated;
            total_failures += stats.allocation_failures;
            efficiency_sum += stats.memory_efficiency;
        }

        let allocator_count = self.stats.len();
        let avg_efficiency = if allocator_count > 0 {
            efficiency_sum / allocator_count as f64
        } else {
            0.0
        };

        AggregateMetrics {
            total_allocations,
            total_deallocations,
            total_allocated,
            total_failures,
            average_efficiency: avg_efficiency,
            allocator_count,
        }
    }
}

/// Aggregate metrics across all allocators
#[derive(Debug, Clone)]
pub struct AggregateMetrics {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub total_allocated: usize,
    pub total_failures: u64,
    pub average_efficiency: f64,
    pub allocator_count: usize,
}
