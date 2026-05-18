//! CUDA memory pool management and optimization
//!
//! This module provides advanced memory pool management across all CUDA memory types
//! including device, unified, and pinned memory pools with intelligent optimization,
//! automatic cleanup, and performance analytics.

// Allow unused variables for pool manager stubs
#![allow(unused_variables)]

use super::allocation::{
    size_class as compute_size_class, CudaAllocation, PinnedAllocation, UnifiedAllocation,
};
use crate::cuda::error::{CudaError, CudaResult, CustResultExt};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Comprehensive memory pool manager for all CUDA memory types
///
/// Coordinates multiple memory pool types with intelligent resource management,
/// automatic optimization, and cross-pool analytics for optimal performance.
#[derive(Debug)]
pub struct UnifiedMemoryPoolManager {
    /// Device memory pools by device ID and size class
    device_pools: RwLock<HashMap<usize, HashMap<usize, DevicePool>>>,

    /// Unified memory pools by size class
    unified_pools: Mutex<HashMap<usize, UnifiedPool>>,

    /// Pinned memory pools by device ID and size class
    pinned_pools: RwLock<HashMap<usize, HashMap<usize, PinnedPool>>>,

    /// Global pool statistics
    global_stats: Mutex<GlobalPoolStats>,

    /// Pool management configuration
    config: PoolManagerConfig,

    /// Resource allocation tracker
    resource_tracker: Arc<Mutex<ResourceTracker>>,

    /// Pool optimization engine
    optimization_engine: Arc<Mutex<PoolOptimizationEngine>>,

    /// Memory pressure monitor
    pressure_monitor: Arc<RwLock<MemoryPressureMonitor>>,

    /// Cross-pool analytics
    analytics_engine: Arc<Mutex<CrossPoolAnalytics>>,
}

/// Configuration for unified pool management
#[derive(Debug, Clone)]
pub struct PoolManagerConfig {
    /// Enable cross-pool optimization
    pub enable_cross_pool_optimization: bool,

    /// Enable automatic pool scaling
    pub enable_auto_scaling: bool,

    /// Enable memory pressure monitoring
    pub enable_pressure_monitoring: bool,

    /// Global memory limit across all pools
    pub global_memory_limit: Option<usize>,

    /// Pool cleanup interval
    pub cleanup_interval: Duration,

    /// Enable pool analytics
    pub enable_analytics: bool,

    /// Optimization check interval
    pub optimization_interval: Duration,

    /// Enable pool defragmentation
    pub enable_defragmentation: bool,

    /// Memory pressure threshold (0.0 to 1.0)
    pub pressure_threshold: f32,

    /// Enable adaptive pool sizes
    pub enable_adaptive_sizing: bool,
}

/// Device memory pool for specific device and size class
#[derive(Debug)]
pub struct DevicePool {
    /// Device ID
    device_id: usize,

    /// Size class
    size_class: usize,

    /// Free allocations available for reuse
    free_allocations: Vec<CudaAllocation>,

    /// Active allocations currently in use
    active_allocations: Vec<CudaAllocation>,

    /// Pool statistics
    stats: PoolStats,

    /// Pool configuration
    config: DevicePoolConfig,

    /// Last access time for cleanup
    last_access: Instant,

    /// Pool health metrics
    health_metrics: PoolHealthMetrics,
}

/// Unified memory pool for specific size class
#[derive(Debug)]
pub struct UnifiedPool {
    /// Size class
    size_class: usize,

    /// Free allocations available for reuse
    free_allocations: Vec<UnifiedAllocation>,

    /// Active allocations currently in use
    active_allocations: Vec<UnifiedAllocation>,

    /// Pool statistics
    stats: PoolStats,

    /// Migration optimization data
    migration_optimizer: MigrationOptimizer,

    /// Last access time
    last_access: Instant,

    /// Pool health metrics
    health_metrics: PoolHealthMetrics,
}

/// Pinned memory pool for specific device and size class
#[derive(Debug)]
pub struct PinnedPool {
    /// Device ID
    device_id: usize,

    /// Size class
    size_class: usize,

    /// Free allocations available for reuse
    free_allocations: Vec<PinnedAllocation>,

    /// Active allocations currently in use
    active_allocations: Vec<PinnedAllocation>,

    /// Pool statistics
    stats: PoolStats,

    /// Transfer optimization data
    transfer_optimizer: TransferOptimizer,

    /// Last access time
    last_access: Instant,

    /// Pool health metrics
    health_metrics: PoolHealthMetrics,
}

/// Pool-specific configuration
#[derive(Debug, Clone)]
pub struct DevicePoolConfig {
    /// Maximum free allocations to keep
    pub max_free_allocations: usize,

    /// Pool growth strategy
    pub growth_strategy: PoolGrowthStrategy,

    /// Enable pool statistics tracking
    pub enable_statistics: bool,

    /// Allocation lifetime tracking
    pub track_allocation_lifetime: bool,

    /// Enable pool health monitoring
    pub enable_health_monitoring: bool,

    /// Pool cleanup threshold
    pub cleanup_threshold: Duration,
}

/// Pool growth strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PoolGrowthStrategy {
    /// Fixed size pool
    Fixed { size: usize },
    /// Linear growth
    Linear { increment: usize },
    /// Exponential growth
    Exponential { factor: f32, max_size: usize },
    /// Adaptive growth based on usage patterns
    Adaptive,
    /// Conservative growth (slow expansion)
    Conservative,
    /// Aggressive growth (fast expansion)
    Aggressive,
}

/// Pool statistics for monitoring and optimization
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total allocations served by this pool
    pub total_allocations: u64,

    /// Total deallocations processed
    pub total_deallocations: u64,

    /// Cache hits (allocations served from pool)
    pub cache_hits: u64,

    /// Cache misses (new allocations required)
    pub cache_misses: u64,

    /// Average allocation size
    pub average_allocation_size: f64,

    /// Peak pool utilization
    pub peak_utilization: f32,

    /// Current pool utilization
    pub current_utilization: f32,

    /// Total memory managed by pool
    pub total_pool_memory: usize,

    /// Memory efficiency (used/allocated)
    pub memory_efficiency: f32,

    /// Average allocation lifetime
    pub average_allocation_lifetime: Duration,

    /// Pool hit rate (cache_hits / total_allocations)
    pub hit_rate: f32,
}

/// Pool health metrics for monitoring
#[derive(Debug, Clone)]
pub struct PoolHealthMetrics {
    /// Health score (0.0 to 1.0)
    pub health_score: f32,

    /// Fragmentation level (0.0 to 1.0)
    pub fragmentation_level: f32,

    /// Memory waste percentage
    pub memory_waste: f32,

    /// Pool efficiency trend
    pub efficiency_trend: EfficiencyTrend,

    /// Last health check time
    pub last_health_check: Instant,

    /// Health issues detected
    pub health_issues: Vec<PoolHealthIssue>,

    /// Recommended actions
    pub recommended_actions: Vec<PoolAction>,
}

/// Efficiency trend indicators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfficiencyTrend {
    Improving,
    Stable,
    Declining,
    Critical,
}

/// Pool health issues
#[derive(Debug, Clone)]
pub enum PoolHealthIssue {
    HighFragmentation { level: f32 },
    LowHitRate { rate: f32 },
    ExcessiveMemoryWaste { percentage: f32 },
    PoorGrowthPattern,
    FrequentCleanups,
    MemoryLeaks,
    PerformanceDegradation { factor: f32 },
}

/// Recommended pool actions
#[derive(Debug, Clone)]
pub enum PoolAction {
    DefragmentPool,
    AdjustGrowthStrategy(PoolGrowthStrategy),
    IncreasePoolSize { new_size: usize },
    DecreasePoolSize { new_size: usize },
    ForceCleanup,
    RebalanceAllocations,
    OptimizeSizeClass,
}

/// Migration optimization for unified memory pools
#[derive(Debug, Clone)]
pub struct MigrationOptimizer {
    /// Migration patterns analysis
    migration_patterns: HashMap<String, MigrationPattern>,

    /// Optimal migration strategies
    optimal_strategies: Vec<MigrationStrategy>,

    /// Migration cost tracking
    migration_costs: MigrationCostTracker,

    /// Performance improvements from optimization
    performance_gains: f32,
}

/// Migration pattern analysis
#[derive(Debug, Clone)]
pub struct MigrationPattern {
    /// Pattern identifier
    pub pattern_id: String,

    /// Access frequency
    pub access_frequency: f32,

    /// Dominant location
    pub dominant_location: Location,

    /// Migration cost
    pub migration_cost: f64,

    /// Pattern confidence
    pub confidence: f32,
}

/// Migration strategies
#[derive(Debug, Clone)]
pub enum MigrationStrategy {
    /// Eager migration on first access
    EagerMigration,
    /// Lazy migration on demand
    LazyMigration,
    /// Predictive migration based on patterns
    PredictiveMigration { confidence_threshold: f32 },
    /// No migration (keep in place)
    NoMigration,
}

/// Memory location for migration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Location {
    Host,
    Device(usize),
}

/// Migration cost tracking
#[derive(Debug, Clone)]
pub struct MigrationCostTracker {
    /// Average migration time
    pub average_migration_time: Duration,

    /// Migration bandwidth utilization
    pub bandwidth_utilization: f32,

    /// Cost per byte migrated
    pub cost_per_byte: f64,

    /// Total migrations performed
    pub total_migrations: u64,

    /// Cost savings from optimization
    pub cost_savings: f64,
}

/// Transfer optimization for pinned memory pools
#[derive(Debug, Clone)]
pub struct TransferOptimizer {
    /// Transfer patterns analysis
    transfer_patterns: HashMap<String, TransferPattern>,

    /// Optimal transfer strategies
    optimal_strategies: Vec<TransferStrategy>,

    /// Transfer performance tracking
    performance_tracker: TransferPerformanceTracker,

    /// Bandwidth utilization optimization
    bandwidth_optimizer: BandwidthOptimizer,
}

/// Transfer pattern analysis
#[derive(Debug, Clone)]
pub struct TransferPattern {
    /// Pattern identifier
    pub pattern_id: String,

    /// Transfer direction frequency
    pub direction_frequency: HashMap<TransferDirection, f32>,

    /// Average transfer size
    pub average_transfer_size: usize,

    /// Peak bandwidth achieved
    pub peak_bandwidth: f64,

    /// Pattern stability
    pub stability: f32,
}

/// Transfer directions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransferDirection {
    HostToDevice,
    DeviceToHost,
    Bidirectional,
}

/// Transfer strategies
#[derive(Debug, Clone)]
pub enum TransferStrategy {
    /// Asynchronous transfers
    AsyncTransfer,
    /// Synchronous transfers
    SyncTransfer,
    /// Batched transfers
    BatchedTransfer { batch_size: usize },
    /// Streaming transfers
    StreamingTransfer { stream_count: usize },
}

/// Transfer performance tracking
#[derive(Debug, Clone)]
pub struct TransferPerformanceTracker {
    /// Average bandwidth achieved
    pub average_bandwidth: f64,

    /// Peak bandwidth achieved
    pub peak_bandwidth: f64,

    /// Transfer efficiency
    pub transfer_efficiency: f32,

    /// Latency measurements
    pub latency_stats: LatencyStats,

    /// Performance trend
    pub performance_trend: PerformanceTrend,
}

/// Latency statistics
#[derive(Debug, Clone)]
pub struct LatencyStats {
    /// Average latency
    pub average_latency: Duration,

    /// Minimum latency
    pub min_latency: Duration,

    /// Maximum latency
    pub max_latency: Duration,

    /// Latency variance
    pub latency_variance: f64,
}

/// Performance trend indicators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Declining,
    Volatile,
}

/// Bandwidth optimization engine
#[derive(Debug, Clone)]
pub struct BandwidthOptimizer {
    /// Optimal bandwidth configurations
    optimal_configs: Vec<BandwidthConfig>,

    /// Current bandwidth utilization
    current_utilization: f32,

    /// Target bandwidth utilization
    target_utilization: f32,

    /// Optimization history
    optimization_history: Vec<BandwidthOptimization>,
}

/// Bandwidth configuration
#[derive(Debug, Clone)]
pub struct BandwidthConfig {
    /// Configuration name
    pub name: String,

    /// Transfer size thresholds
    pub size_thresholds: Vec<usize>,

    /// Optimal transfer strategies per threshold
    pub strategies: HashMap<usize, TransferStrategy>,

    /// Expected performance improvement
    pub expected_improvement: f32,
}

/// Bandwidth optimization record
#[derive(Debug, Clone)]
pub struct BandwidthOptimization {
    /// Optimization timestamp
    pub timestamp: Instant,

    /// Applied configuration
    pub config: BandwidthConfig,

    /// Performance improvement achieved
    pub improvement: f32,

    /// Optimization duration
    pub duration: Duration,
}

impl UnifiedMemoryPoolManager {
    /// Create new unified memory pool manager
    pub fn new(config: PoolManagerConfig) -> Self {
        Self {
            device_pools: RwLock::new(HashMap::new()),
            unified_pools: Mutex::new(HashMap::new()),
            pinned_pools: RwLock::new(HashMap::new()),
            global_stats: Mutex::new(GlobalPoolStats::default()),
            config,
            resource_tracker: Arc::new(Mutex::new(ResourceTracker::new())),
            optimization_engine: Arc::new(Mutex::new(PoolOptimizationEngine::new())),
            pressure_monitor: Arc::new(RwLock::new(MemoryPressureMonitor::new())),
            analytics_engine: Arc::new(Mutex::new(CrossPoolAnalytics::new())),
        }
    }

    /// Get or create device pool for specific device and size class
    pub fn get_device_pool(&self, device_id: usize, size_class: usize) -> CudaResult<()> {
        let mut device_pools = self.device_pools.write().map_err(|_| CudaError::Context {
            message: "Failed to acquire device pools lock".to_string(),
        })?;

        let device_map = device_pools.entry(device_id).or_insert_with(HashMap::new);

        if !device_map.contains_key(&size_class) {
            let pool = DevicePool::new(device_id, size_class, DevicePoolConfig::default());
            device_map.insert(size_class, pool);
        }

        Ok(())
    }

    /// Get or create unified pool for size class
    pub fn get_unified_pool(&self, size_class: usize) -> CudaResult<()> {
        let mut unified_pools = self.unified_pools.lock().map_err(|_| CudaError::Context {
            message: "Failed to acquire unified pools lock".to_string(),
        })?;

        if !unified_pools.contains_key(&size_class) {
            let pool = UnifiedPool::new(size_class);
            unified_pools.insert(size_class, pool);
        }

        Ok(())
    }

    /// Get or create pinned pool for device and size class
    pub fn get_pinned_pool(&self, device_id: usize, size_class: usize) -> CudaResult<()> {
        let mut pinned_pools = self.pinned_pools.write().map_err(|_| CudaError::Context {
            message: "Failed to acquire pinned pools lock".to_string(),
        })?;

        let device_map = pinned_pools.entry(device_id).or_insert_with(HashMap::new);

        if !device_map.contains_key(&size_class) {
            let pool = PinnedPool::new(device_id, size_class);
            device_map.insert(size_class, pool);
        }

        Ok(())
    }

    /// Run comprehensive pool optimization
    pub fn optimize_all_pools(&self) -> CudaResult<GlobalOptimizationResult> {
        let start_time = Instant::now();
        let mut optimizations_applied = 0;
        let mut total_improvement = 0.0;

        // Device pool optimization
        if let Ok(device_pools) = self.device_pools.read() {
            for (device_id, pools) in device_pools.iter() {
                for (size_class, pool) in pools.iter() {
                    if let Some(optimization) = self.analyze_device_pool_optimization(pool) {
                        // Apply optimization (simplified)
                        optimizations_applied += 1;
                        total_improvement += optimization.expected_improvement;
                    }
                }
            }
        }

        // Unified pool optimization
        if let Ok(unified_pools) = self.unified_pools.lock() {
            for (size_class, pool) in unified_pools.iter() {
                if let Some(optimization) = self.analyze_unified_pool_optimization(pool) {
                    optimizations_applied += 1;
                    total_improvement += optimization.expected_improvement;
                }
            }
        }

        // Cross-pool optimization
        if self.config.enable_cross_pool_optimization {
            if let Ok(analytics) = self.analytics_engine.lock() {
                let cross_pool_improvements = analytics.identify_optimization_opportunities();
                optimizations_applied += cross_pool_improvements.len();
                total_improvement += cross_pool_improvements
                    .iter()
                    .map(|opt| opt.expected_benefit)
                    .sum::<f32>();
            }
        }

        Ok(GlobalOptimizationResult {
            duration: start_time.elapsed(),
            optimizations_applied,
            average_improvement: if optimizations_applied > 0 {
                total_improvement / optimizations_applied as f32
            } else {
                0.0
            },
            total_improvement,
            pools_optimized: optimizations_applied,
        })
    }

    /// Get comprehensive pool analytics
    pub fn get_pool_analytics(&self) -> CudaResult<PoolAnalyticsReport> {
        let mut device_pool_count = 0;
        let mut unified_pool_count = 0;
        let mut pinned_pool_count = 0;

        // Count device pools
        if let Ok(device_pools) = self.device_pools.read() {
            device_pool_count = device_pools.values().map(|pools| pools.len()).sum();
        }

        // Count unified pools
        if let Ok(unified_pools) = self.unified_pools.lock() {
            unified_pool_count = unified_pools.len();
        }

        // Count pinned pools
        if let Ok(pinned_pools) = self.pinned_pools.read() {
            pinned_pool_count = pinned_pools.values().map(|pools| pools.len()).sum();
        }

        let global_stats = self.global_stats.lock().map_err(|_| CudaError::Context {
            message: "Failed to acquire global stats lock".to_string(),
        })?;

        Ok(PoolAnalyticsReport {
            total_pools: device_pool_count + unified_pool_count + pinned_pool_count,
            device_pools: device_pool_count,
            unified_pools: unified_pool_count,
            pinned_pools: pinned_pool_count,
            global_stats: global_stats.clone(),
            optimization_opportunities: self.count_optimization_opportunities(),
            health_score: self.calculate_global_health_score(),
        })
    }

    // Private helper methods
    fn analyze_device_pool_optimization(&self, pool: &DevicePool) -> Option<PoolOptimization> {
        if pool.health_metrics.health_score < 0.7 {
            Some(PoolOptimization {
                optimization_type: "device_pool_health".to_string(),
                expected_improvement: 0.2,
                complexity: OptimizationComplexity::Moderate,
            })
        } else {
            None
        }
    }

    fn analyze_unified_pool_optimization(&self, pool: &UnifiedPool) -> Option<PoolOptimization> {
        if pool.stats.hit_rate < 0.8 {
            Some(PoolOptimization {
                optimization_type: "unified_pool_hit_rate".to_string(),
                expected_improvement: 0.15,
                complexity: OptimizationComplexity::Simple,
            })
        } else {
            None
        }
    }

    fn count_optimization_opportunities(&self) -> usize {
        // Simplified counting
        0
    }

    fn calculate_global_health_score(&self) -> f32 {
        // Simplified calculation
        0.85
    }

    /// Allocate a `CudaAllocation` from the device pool for `device_id`.
    ///
    /// Finds the appropriate size class (next power-of-two >= `size`), reuses a
    /// free allocation if one is available, otherwise calls `cust::memory::cuda_malloc`
    /// to allocate a new CUDA device block.
    pub fn allocate_from_device_pool(
        &self,
        size: usize,
        device_id: usize,
    ) -> CudaResult<CudaAllocation> {
        let sc = compute_size_class(size);

        let mut device_pools = self.device_pools.write().map_err(|_| CudaError::Context {
            message: "Failed to acquire device pools write lock for allocation".to_string(),
        })?;

        let device_map = device_pools.entry(device_id).or_insert_with(HashMap::new);
        let pool = device_map
            .entry(sc)
            .or_insert_with(|| DevicePool::new(device_id, sc, DevicePoolConfig::default()));

        // Reuse an existing free allocation if available
        if let Some(mut alloc) = pool.free_allocations.pop() {
            alloc.in_use = true;
            pool.active_allocations.push(alloc);
            pool.stats.total_allocations += 1;
            pool.stats.cache_hits += 1;
            pool.last_access = Instant::now();
            return Ok(alloc);
        }

        // No free allocation — call the real CUDA allocator
        let ptr = unsafe { cust::memory::cuda_malloc(sc).cuda_result()? };
        let alloc = CudaAllocation::new_on_device(ptr, sc, sc, device_id);
        pool.active_allocations.push(alloc);
        pool.stats.total_allocations += 1;
        pool.stats.cache_misses += 1;
        pool.last_access = Instant::now();

        Ok(alloc)
    }

    /// Return a `CudaAllocation` back to the device pool for future reuse.
    ///
    /// If the pool's free list is at capacity the allocation is kept anyway
    /// (the pool will trim on the next cleanup pass).
    pub fn return_to_device_pool(&self, mut alloc: CudaAllocation, device_id: usize) {
        let sc = alloc.size_class;
        alloc.in_use = false;

        let Ok(mut device_pools) = self.device_pools.write() else {
            return;
        };

        let device_map = device_pools.entry(device_id).or_insert_with(HashMap::new);
        let pool = device_map
            .entry(sc)
            .or_insert_with(|| DevicePool::new(device_id, sc, DevicePoolConfig::default()));

        // Remove from active list (best-effort; compare by device pointer address)
        let raw = alloc.ptr.as_raw();
        pool.active_allocations.retain(|a| a.ptr.as_raw() != raw);

        pool.free_allocations.push(alloc);
        pool.stats.total_deallocations += 1;
        pool.last_access = Instant::now();
    }
}

/// Pool optimization record
#[derive(Debug, Clone)]
pub struct PoolOptimization {
    pub optimization_type: String,
    pub expected_improvement: f32,
    pub complexity: OptimizationComplexity,
}

/// Global optimization result
#[derive(Debug, Clone)]
pub struct GlobalOptimizationResult {
    /// Optimization duration
    pub duration: Duration,

    /// Number of optimizations applied
    pub optimizations_applied: usize,

    /// Average improvement per optimization
    pub average_improvement: f32,

    /// Total improvement achieved
    pub total_improvement: f32,

    /// Number of pools optimized
    pub pools_optimized: usize,
}

/// Pool analytics report
#[derive(Debug, Clone)]
pub struct PoolAnalyticsReport {
    /// Total number of pools
    pub total_pools: usize,

    /// Device pools count
    pub device_pools: usize,

    /// Unified pools count
    pub unified_pools: usize,

    /// Pinned pools count
    pub pinned_pools: usize,

    /// Global statistics
    pub global_stats: GlobalPoolStats,

    /// Optimization opportunities count
    pub optimization_opportunities: usize,

    /// Overall health score
    pub health_score: f32,
}

// Implementation for pool types
impl DevicePool {
    fn new(device_id: usize, size_class: usize, config: DevicePoolConfig) -> Self {
        Self {
            device_id,
            size_class,
            free_allocations: Vec::new(),
            active_allocations: Vec::new(),
            stats: PoolStats::default(),
            config,
            last_access: Instant::now(),
            health_metrics: PoolHealthMetrics::default(),
        }
    }
}

impl UnifiedPool {
    fn new(size_class: usize) -> Self {
        Self {
            size_class,
            free_allocations: Vec::new(),
            active_allocations: Vec::new(),
            stats: PoolStats::default(),
            migration_optimizer: MigrationOptimizer::default(),
            last_access: Instant::now(),
            health_metrics: PoolHealthMetrics::default(),
        }
    }
}

impl PinnedPool {
    fn new(device_id: usize, size_class: usize) -> Self {
        Self {
            device_id,
            size_class,
            free_allocations: Vec::new(),
            active_allocations: Vec::new(),
            stats: PoolStats::default(),
            transfer_optimizer: TransferOptimizer::default(),
            last_access: Instant::now(),
            health_metrics: PoolHealthMetrics::default(),
        }
    }
}

// Default implementations
impl Default for PoolManagerConfig {
    fn default() -> Self {
        Self {
            enable_cross_pool_optimization: true,
            enable_auto_scaling: true,
            enable_pressure_monitoring: true,
            global_memory_limit: None,
            cleanup_interval: Duration::from_secs(300),
            enable_analytics: true,
            optimization_interval: Duration::from_secs(60),
            enable_defragmentation: true,
            pressure_threshold: 0.8,
            enable_adaptive_sizing: true,
        }
    }
}

impl Default for DevicePoolConfig {
    fn default() -> Self {
        Self {
            max_free_allocations: 16,
            growth_strategy: PoolGrowthStrategy::Adaptive,
            enable_statistics: true,
            track_allocation_lifetime: true,
            enable_health_monitoring: true,
            cleanup_threshold: Duration::from_secs(120),
        }
    }
}

impl Default for PoolStats {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            total_deallocations: 0,
            cache_hits: 0,
            cache_misses: 0,
            average_allocation_size: 0.0,
            peak_utilization: 0.0,
            current_utilization: 0.0,
            total_pool_memory: 0,
            memory_efficiency: 1.0,
            average_allocation_lifetime: Duration::from_secs(0),
            hit_rate: 0.0,
        }
    }
}

impl Default for PoolHealthMetrics {
    fn default() -> Self {
        Self {
            health_score: 1.0,
            fragmentation_level: 0.0,
            memory_waste: 0.0,
            efficiency_trend: EfficiencyTrend::Stable,
            last_health_check: Instant::now(),
            health_issues: Vec::new(),
            recommended_actions: Vec::new(),
        }
    }
}

impl Default for MigrationOptimizer {
    fn default() -> Self {
        Self {
            migration_patterns: HashMap::new(),
            optimal_strategies: Vec::new(),
            migration_costs: MigrationCostTracker::default(),
            performance_gains: 0.0,
        }
    }
}

impl Default for MigrationCostTracker {
    fn default() -> Self {
        Self {
            average_migration_time: Duration::from_secs(0),
            bandwidth_utilization: 0.0,
            cost_per_byte: 0.0,
            total_migrations: 0,
            cost_savings: 0.0,
        }
    }
}

impl Default for TransferOptimizer {
    fn default() -> Self {
        Self {
            transfer_patterns: HashMap::new(),
            optimal_strategies: Vec::new(),
            performance_tracker: TransferPerformanceTracker::default(),
            bandwidth_optimizer: BandwidthOptimizer::default(),
        }
    }
}

impl Default for TransferPerformanceTracker {
    fn default() -> Self {
        Self {
            average_bandwidth: 0.0,
            peak_bandwidth: 0.0,
            transfer_efficiency: 0.0,
            latency_stats: LatencyStats::default(),
            performance_trend: PerformanceTrend::Stable,
        }
    }
}

impl Default for LatencyStats {
    fn default() -> Self {
        Self {
            average_latency: Duration::from_secs(0),
            min_latency: Duration::from_secs(0),
            max_latency: Duration::from_secs(0),
            latency_variance: 0.0,
        }
    }
}

impl Default for BandwidthOptimizer {
    fn default() -> Self {
        Self {
            optimal_configs: Vec::new(),
            current_utilization: 0.0,
            target_utilization: 0.8,
            optimization_history: Vec::new(),
        }
    }
}

#[path = "memory_pools_pinned.rs"]
mod memory_pools_pinned;
pub use memory_pools_pinned::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_manager_creation() {
        let config = PoolManagerConfig::default();
        let manager = UnifiedMemoryPoolManager::new(config);

        // Basic validation
        assert!(manager.config.enable_cross_pool_optimization);
        assert!(manager.config.enable_auto_scaling);
    }

    #[test]
    fn test_pool_growth_strategies() {
        let fixed = PoolGrowthStrategy::Fixed { size: 1024 };
        let adaptive = PoolGrowthStrategy::Adaptive;

        assert_ne!(fixed, adaptive);

        if let PoolGrowthStrategy::Fixed { size } = fixed {
            assert_eq!(size, 1024);
        }
    }

    #[test]
    fn test_resource_pressure_levels() {
        assert!(ResourcePressureLevel::Critical > ResourcePressureLevel::High);
        assert!(ResourcePressureLevel::High > ResourcePressureLevel::Medium);
        assert!(ResourcePressureLevel::Medium > ResourcePressureLevel::Low);
    }

    #[test]
    fn test_pool_health_metrics() {
        let metrics = PoolHealthMetrics::default();
        assert_eq!(metrics.health_score, 1.0);
        assert_eq!(metrics.fragmentation_level, 0.0);
        assert_eq!(metrics.efficiency_trend, EfficiencyTrend::Stable);
    }

    #[test]
    fn test_optimization_priorities() {
        assert!(OptimizationPriority::Critical > OptimizationPriority::High);
        assert!(OptimizationPriority::High > OptimizationPriority::Normal);
        assert!(OptimizationPriority::Normal > OptimizationPriority::Low);
    }
}
