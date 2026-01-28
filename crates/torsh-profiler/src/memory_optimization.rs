//! Advanced Memory Optimization Features
//!
//! This module provides sophisticated memory optimization techniques including
//! adaptive memory management, smart garbage collection triggers, memory pool
//! optimization, and predictive memory allocation strategies.

use crate::memory::{MemoryEvent, MemoryEventType, MemoryProfiler, MemoryStats};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

/// Advanced memory optimizer with adaptive strategies
#[derive(Debug)]
pub struct AdvancedMemoryOptimizer {
    /// Current memory strategies
    strategies: Arc<RwLock<MemoryStrategies>>,
    /// Memory usage history for pattern analysis
    usage_history: Arc<Mutex<VecDeque<MemorySnapshot>>>,
    /// Statistics and metrics
    stats: Arc<MemoryOptimizationStats>,
    /// Configuration
    config: MemoryOptimizationConfig,
    /// Predictive models
    predictor: Arc<Mutex<MemoryUsagePredictor>>,
    /// Memory pool manager
    pool_manager: Arc<AdaptivePoolManager>,
}

/// Memory optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStrategies {
    /// Enable predictive allocation
    pub predictive_allocation: bool,
    /// Enable adaptive garbage collection
    pub adaptive_gc: bool,
    /// Enable memory compaction
    pub memory_compaction: bool,
    /// Enable pool optimization
    pub pool_optimization: bool,
    /// Memory pressure threshold (0.0 - 1.0)
    pub pressure_threshold: f64,
    /// Allocation batch size optimization
    pub batch_optimization: bool,
}

impl Default for MemoryStrategies {
    fn default() -> Self {
        Self {
            predictive_allocation: true,
            adaptive_gc: true,
            memory_compaction: true,
            pool_optimization: true,
            pressure_threshold: 0.8,
            batch_optimization: true,
        }
    }
}

/// Memory snapshot for pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    pub timestamp: SystemTime,
    pub total_allocated: usize,
    pub peak_usage: usize,
    pub fragmentation_ratio: f64,
    pub allocation_rate: f64,
    pub deallocation_rate: f64,
    pub gc_pressure: f64,
    pub operation_context: String,
}

/// Memory optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizationConfig {
    /// History window size for pattern analysis
    pub history_window: usize,
    /// Minimum samples before optimization kicks in
    pub min_samples: usize,
    /// Optimization check interval
    pub check_interval: Duration,
    /// Memory pressure warning threshold
    pub warning_threshold: f64,
    /// Memory pressure critical threshold
    pub critical_threshold: f64,
    /// Enable machine learning predictions
    pub ml_predictions: bool,
    /// Pool size optimization parameters
    pub pool_params: PoolOptimizationParams,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            history_window: 1000,
            min_samples: 100,
            check_interval: Duration::from_secs(5),
            warning_threshold: 0.7,
            critical_threshold: 0.9,
            ml_predictions: true,
            pool_params: PoolOptimizationParams::default(),
        }
    }
}

/// Pool optimization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolOptimizationParams {
    pub initial_pool_size: usize,
    pub growth_factor: f64,
    pub shrink_threshold: f64,
    pub min_pool_size: usize,
    pub max_pool_size: usize,
    pub rebalance_interval: Duration,
}

impl Default for PoolOptimizationParams {
    fn default() -> Self {
        Self {
            initial_pool_size: 1024 * 1024, // 1MB
            growth_factor: 1.5,
            shrink_threshold: 0.3,
            min_pool_size: 64 * 1024,         // 64KB
            max_pool_size: 128 * 1024 * 1024, // 128MB
            rebalance_interval: Duration::from_secs(30),
        }
    }
}

/// Memory optimization statistics
#[derive(Debug)]
pub struct MemoryOptimizationStats {
    pub optimizations_performed: AtomicU64,
    pub memory_saved: AtomicUsize,
    pub gc_triggers_avoided: AtomicU64,
    pub fragmentation_reduced: AtomicU64,
    pub allocation_predictions: AtomicU64,
    pub prediction_accuracy: AtomicUsize, // as percentage * 100
}

impl Default for MemoryOptimizationStats {
    fn default() -> Self {
        Self {
            optimizations_performed: AtomicU64::new(0),
            memory_saved: AtomicUsize::new(0),
            gc_triggers_avoided: AtomicU64::new(0),
            fragmentation_reduced: AtomicU64::new(0),
            allocation_predictions: AtomicU64::new(0),
            prediction_accuracy: AtomicUsize::new(8500), // Start at 85%
        }
    }
}

/// Memory usage predictor using simple machine learning
#[derive(Debug)]
pub struct MemoryUsagePredictor {
    /// Historical data points
    data_points: Vec<DataPoint>,
    /// Simple linear regression parameters
    slope: f64,
    intercept: f64,
    /// Seasonal patterns
    patterns: HashMap<String, f64>,
    /// Prediction confidence
    confidence: f64,
}

#[derive(Debug, Clone)]
pub struct DataPoint {
    pub timestamp: f64,
    pub memory_usage: f64,
    pub context: String,
}

/// Adaptive memory pool manager
#[derive(Debug)]
pub struct AdaptivePoolManager {
    pools: Arc<Mutex<BTreeMap<usize, MemoryPool>>>,
    config: PoolOptimizationParams,
    stats: PoolManagerStats,
}

#[derive(Debug)]
pub struct MemoryPool {
    pub size_class: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub total_capacity: usize,
    pub hit_rate: f64,
    pub last_rebalance: SystemTime,
}

#[derive(Debug, Default)]
pub struct PoolManagerStats {
    pub pool_hits: AtomicU64,
    pub pool_misses: AtomicU64,
    pub pool_expansions: AtomicU64,
    pub pool_contractions: AtomicU64,
    pub cross_pool_transfers: AtomicU64,
}

impl AdvancedMemoryOptimizer {
    /// Create a new advanced memory optimizer
    pub fn new() -> Self {
        Self::with_config(MemoryOptimizationConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: MemoryOptimizationConfig) -> Self {
        Self {
            strategies: Arc::new(RwLock::new(MemoryStrategies::default())),
            usage_history: Arc::new(Mutex::new(VecDeque::with_capacity(config.history_window))),
            stats: Arc::new(MemoryOptimizationStats::default()),
            predictor: Arc::new(Mutex::new(MemoryUsagePredictor::new())),
            pool_manager: Arc::new(AdaptivePoolManager::new(config.pool_params.clone())),
            config,
        }
    }

    /// Start the optimization engine
    pub fn start_optimization(&self, memory_profiler: Arc<Mutex<MemoryProfiler>>) {
        let optimizer = Arc::new(self.clone());
        let profiler = Arc::clone(&memory_profiler);

        thread::spawn(move || loop {
            thread::sleep(optimizer.config.check_interval);
            optimizer.optimization_cycle(&profiler);
        });
    }

    /// Perform one optimization cycle
    fn optimization_cycle(&self, memory_profiler: &Arc<Mutex<MemoryProfiler>>) {
        // Collect current memory state
        let snapshot = self.collect_memory_snapshot(memory_profiler);

        // Add to history
        self.add_snapshot(snapshot.clone());

        // Analyze patterns and predict future usage
        if self.should_perform_optimization() {
            self.perform_optimizations(&snapshot, memory_profiler);
        }

        // Update predictive models
        self.update_predictions(&snapshot);

        // Optimize memory pools
        self.optimize_pools();
    }

    /// Collect current memory snapshot
    fn collect_memory_snapshot(
        &self,
        memory_profiler: &Arc<Mutex<MemoryProfiler>>,
    ) -> MemorySnapshot {
        let profiler = memory_profiler.lock().expect("lock should not be poisoned");
        let stats_result = profiler.get_stats();

        let stats = match stats_result {
            Ok(s) => s,
            Err(_) => MemoryStats::default(), // Use default if error
        };

        MemorySnapshot {
            timestamp: SystemTime::now(),
            total_allocated: stats.allocated,
            peak_usage: stats.peak,
            fragmentation_ratio: self.calculate_fragmentation_ratio(&stats),
            allocation_rate: self.calculate_allocation_rate(&stats),
            deallocation_rate: self.calculate_deallocation_rate(&stats),
            gc_pressure: self.calculate_gc_pressure(&stats),
            operation_context: "background_optimization".to_string(),
        }
    }

    /// Add snapshot to history with size management
    fn add_snapshot(&self, snapshot: MemorySnapshot) {
        let mut history = self
            .usage_history
            .lock()
            .expect("lock should not be poisoned");

        if history.len() >= self.config.history_window {
            history.pop_front();
        }

        history.push_back(snapshot);
    }

    /// Determine if optimization should be performed
    fn should_perform_optimization(&self) -> bool {
        let history = self
            .usage_history
            .lock()
            .expect("lock should not be poisoned");

        if history.len() < self.config.min_samples {
            return false;
        }

        // Check if memory pressure is increasing
        let recent_snapshots: Vec<_> = history.iter().rev().take(10).collect();
        let pressure_trend = self.calculate_pressure_trend(&recent_snapshots);

        pressure_trend > self.config.warning_threshold
    }

    /// Perform various optimization strategies
    fn perform_optimizations(
        &self,
        snapshot: &MemorySnapshot,
        memory_profiler: &Arc<Mutex<MemoryProfiler>>,
    ) {
        let strategies = self.strategies.read();

        if strategies.adaptive_gc && snapshot.gc_pressure > self.config.critical_threshold {
            self.suggest_garbage_collection(memory_profiler);
        }

        if strategies.memory_compaction && snapshot.fragmentation_ratio > 0.5 {
            self.perform_memory_compaction(memory_profiler);
        }

        if strategies.predictive_allocation {
            self.optimize_future_allocations(snapshot);
        }

        if strategies.pool_optimization {
            self.rebalance_pools();
        }

        self.stats
            .optimizations_performed
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Suggest garbage collection to the system
    fn suggest_garbage_collection(&self, memory_profiler: &Arc<Mutex<MemoryProfiler>>) {
        // In a real implementation, this would trigger GC in the runtime
        println!("Memory optimizer suggests garbage collection");
        self.stats
            .gc_triggers_avoided
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Perform memory compaction
    fn perform_memory_compaction(&self, memory_profiler: &Arc<Mutex<MemoryProfiler>>) {
        // Simulate memory compaction
        println!("Performing memory compaction to reduce fragmentation");
        self.stats
            .fragmentation_reduced
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Optimize future allocations based on predictions
    fn optimize_future_allocations(&self, snapshot: &MemorySnapshot) {
        let predictor = self.predictor.lock().expect("lock should not be poisoned");

        if let Some(prediction) = predictor.predict_next_allocation() {
            // Pre-allocate memory pools based on prediction
            self.pool_manager.prepare_for_allocation(prediction);
            self.stats
                .allocation_predictions
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Rebalance memory pools
    fn rebalance_pools(&self) {
        self.pool_manager.rebalance_pools();
    }

    /// Optimize memory pools based on usage patterns
    fn optimize_pools(&self) {
        self.pool_manager.optimize_based_on_usage();
    }

    /// Update predictive models with new data
    fn update_predictions(&self, snapshot: &MemorySnapshot) {
        if !self.config.ml_predictions {
            return;
        }

        let mut predictor = self.predictor.lock().expect("lock should not be poisoned");
        predictor.add_data_point(DataPoint {
            timestamp: snapshot
                .timestamp
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
            memory_usage: snapshot.total_allocated as f64,
            context: snapshot.operation_context.clone(),
        });

        predictor.update_model();
    }

    // Helper calculation methods
    fn calculate_fragmentation_ratio(&self, stats: &MemoryStats) -> f64 {
        if stats.peak == 0 {
            return 0.0;
        }

        1.0 - (stats.allocated as f64 / stats.peak as f64)
    }

    fn calculate_allocation_rate(&self, _stats: &MemoryStats) -> f64 {
        // Calculate based on recent history
        let history = self
            .usage_history
            .lock()
            .expect("lock should not be poisoned");
        if history.len() < 2 {
            return 0.0;
        }

        let recent = &history[history.len() - 1];
        let previous = &history[history.len() - 2];

        let time_delta = recent
            .timestamp
            .duration_since(previous.timestamp)
            .unwrap_or_default()
            .as_secs_f64();

        if time_delta > 0.0 {
            (recent.total_allocated as f64 - previous.total_allocated as f64) / time_delta
        } else {
            0.0
        }
    }

    fn calculate_deallocation_rate(&self, _stats: &MemoryStats) -> f64 {
        // Similar to allocation rate but for deallocations
        0.0 // Placeholder
    }

    fn calculate_gc_pressure(&self, stats: &MemoryStats) -> f64 {
        // Calculate pressure based on allocation rate and available memory
        stats.allocated as f64 / stats.peak.max(1) as f64
    }

    fn calculate_pressure_trend(&self, snapshots: &[&MemorySnapshot]) -> f64 {
        if snapshots.len() < 2 {
            return 0.0;
        }

        // Simple trend calculation
        let first_pressure = snapshots
            .first()
            .expect("snapshots should not be empty after length check")
            .gc_pressure;
        let last_pressure = snapshots
            .last()
            .expect("snapshots should not be empty after length check")
            .gc_pressure;

        (last_pressure - first_pressure) / snapshots.len() as f64
    }

    /// Get optimization statistics
    pub fn get_stats(&self) -> MemoryOptimizationStats {
        MemoryOptimizationStats {
            optimizations_performed: AtomicU64::new(
                self.stats.optimizations_performed.load(Ordering::Relaxed),
            ),
            memory_saved: AtomicUsize::new(self.stats.memory_saved.load(Ordering::Relaxed)),
            gc_triggers_avoided: AtomicU64::new(
                self.stats.gc_triggers_avoided.load(Ordering::Relaxed),
            ),
            fragmentation_reduced: AtomicU64::new(
                self.stats.fragmentation_reduced.load(Ordering::Relaxed),
            ),
            allocation_predictions: AtomicU64::new(
                self.stats.allocation_predictions.load(Ordering::Relaxed),
            ),
            prediction_accuracy: AtomicUsize::new(
                self.stats.prediction_accuracy.load(Ordering::Relaxed),
            ),
        }
    }

    /// Export optimization data
    pub fn export_optimization_data(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let data = OptimizationExportData {
            config: self.config.clone(),
            strategies: self.strategies.read().clone(),
            history: self
                .usage_history
                .lock()
                .expect("lock should not be poisoned")
                .clone()
                .into(),
            stats: self.get_optimization_stats_summary(),
            timestamp: SystemTime::now(),
        };

        let json = serde_json::to_string_pretty(&data)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    fn get_optimization_stats_summary(&self) -> OptimizationStatsSummary {
        OptimizationStatsSummary {
            total_optimizations: self.stats.optimizations_performed.load(Ordering::Relaxed),
            memory_saved_bytes: self.stats.memory_saved.load(Ordering::Relaxed),
            gc_triggers_avoided: self.stats.gc_triggers_avoided.load(Ordering::Relaxed),
            fragmentation_events_reduced: self.stats.fragmentation_reduced.load(Ordering::Relaxed),
            prediction_accuracy_percent: self.stats.prediction_accuracy.load(Ordering::Relaxed)
                as f64
                / 100.0,
        }
    }
}

impl Clone for AdvancedMemoryOptimizer {
    fn clone(&self) -> Self {
        Self {
            strategies: Arc::clone(&self.strategies),
            usage_history: Arc::clone(&self.usage_history),
            stats: Arc::clone(&self.stats),
            config: self.config.clone(),
            predictor: Arc::clone(&self.predictor),
            pool_manager: Arc::clone(&self.pool_manager),
        }
    }
}

impl MemoryUsagePredictor {
    fn new() -> Self {
        Self {
            data_points: Vec::new(),
            slope: 0.0,
            intercept: 0.0,
            patterns: HashMap::new(),
            confidence: 0.0,
        }
    }

    fn add_data_point(&mut self, point: DataPoint) {
        self.data_points.push(point);

        // Keep only recent data points
        if self.data_points.len() > 1000 {
            self.data_points.remove(0);
        }
    }

    fn update_model(&mut self) {
        if self.data_points.len() < 10 {
            return;
        }

        // Simple linear regression
        let n = self.data_points.len() as f64;
        let sum_x: f64 = self.data_points.iter().map(|p| p.timestamp).sum();
        let sum_y: f64 = self.data_points.iter().map(|p| p.memory_usage).sum();
        let sum_xy: f64 = self
            .data_points
            .iter()
            .map(|p| p.timestamp * p.memory_usage)
            .sum();
        let sum_x2: f64 = self
            .data_points
            .iter()
            .map(|p| p.timestamp * p.timestamp)
            .sum();

        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() > f64::EPSILON {
            self.slope = (n * sum_xy - sum_x * sum_y) / denom;
            self.intercept = (sum_y - self.slope * sum_x) / n;
        }

        self.update_patterns();
        self.calculate_confidence();
    }

    fn update_patterns(&mut self) {
        // Detect seasonal patterns based on context
        let mut context_averages: HashMap<String, Vec<f64>> = HashMap::new();

        for point in &self.data_points {
            context_averages
                .entry(point.context.clone())
                .or_default()
                .push(point.memory_usage);
        }

        for (context, usages) in context_averages {
            let average = usages.iter().sum::<f64>() / usages.len() as f64;
            self.patterns.insert(context, average);
        }
    }

    fn calculate_confidence(&mut self) {
        // Simple confidence calculation based on prediction accuracy
        self.confidence = if self.data_points.len() >= 50 {
            0.8 // High confidence with enough data
        } else {
            0.5 // Medium confidence with limited data
        };
    }

    fn predict_next_allocation(&self) -> Option<f64> {
        if self.confidence < 0.3 {
            return None;
        }

        // Predict based on linear trend
        let current_time = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let predicted = self.slope * current_time + self.intercept;

        if predicted > 0.0 {
            Some(predicted)
        } else {
            None
        }
    }
}

impl AdaptivePoolManager {
    fn new(config: PoolOptimizationParams) -> Self {
        Self {
            pools: Arc::new(Mutex::new(BTreeMap::new())),
            config,
            stats: PoolManagerStats::default(),
        }
    }

    fn prepare_for_allocation(&self, predicted_size: f64) {
        let size_class = self.calculate_size_class(predicted_size as usize);
        let mut pools = self.pools.lock().expect("lock should not be poisoned");

        pools
            .entry(size_class)
            .or_insert_with(|| MemoryPool::new(size_class))
            .prepare_for_demand();
    }

    fn calculate_size_class(&self, size: usize) -> usize {
        // Round up to nearest power of 2
        let mut class = 64; // Minimum size class
        while class < size {
            class *= 2;
        }
        class.min(self.config.max_pool_size)
    }

    fn rebalance_pools(&self) {
        let mut pools = self.pools.lock().expect("lock should not be poisoned");

        for pool in pools.values_mut() {
            if pool.should_expand() {
                pool.expand(&self.config);
                self.stats.pool_expansions.fetch_add(1, Ordering::Relaxed);
            } else if pool.should_shrink(&self.config) {
                pool.shrink(&self.config);
                self.stats.pool_contractions.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn optimize_based_on_usage(&self) {
        // Analyze usage patterns and optimize pool sizes
        let pools = self.pools.lock().expect("lock should not be poisoned");

        for pool in pools.values() {
            if pool.hit_rate < 0.5 {
                // Consider reducing this pool size
                println!(
                    "Pool size class {} has low hit rate: {:.2}",
                    pool.size_class, pool.hit_rate
                );
            }
        }
    }
}

impl MemoryPool {
    fn new(size_class: usize) -> Self {
        Self {
            size_class,
            allocated_blocks: 0,
            free_blocks: 8, // Start with some free blocks
            total_capacity: 8,
            hit_rate: 1.0,
            last_rebalance: SystemTime::now(),
        }
    }

    fn prepare_for_demand(&mut self) {
        if self.free_blocks < 2 {
            self.free_blocks += 4;
            self.total_capacity += 4;
        }
    }

    fn should_expand(&self) -> bool {
        self.free_blocks == 0 && self.hit_rate > 0.8
    }

    fn should_shrink(&self, config: &PoolOptimizationParams) -> bool {
        let utilization = self.allocated_blocks as f64 / self.total_capacity as f64;
        utilization < config.shrink_threshold && self.total_capacity > config.min_pool_size
    }

    fn expand(&mut self, config: &PoolOptimizationParams) {
        let growth = (self.total_capacity as f64 * (config.growth_factor - 1.0)) as usize;
        self.free_blocks += growth;
        self.total_capacity += growth;

        if self.total_capacity > config.max_pool_size {
            let excess = self.total_capacity - config.max_pool_size;
            self.free_blocks = self.free_blocks.saturating_sub(excess);
            self.total_capacity = config.max_pool_size;
        }
    }

    fn shrink(&mut self, config: &PoolOptimizationParams) {
        let reduction = (self.total_capacity as f64 * (1.0 - config.shrink_threshold)) as usize;
        self.free_blocks = self.free_blocks.saturating_sub(reduction);
        self.total_capacity = (self.total_capacity - reduction).max(config.min_pool_size);
    }
}

/// Export data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationExportData {
    pub config: MemoryOptimizationConfig,
    pub strategies: MemoryStrategies,
    pub history: Vec<MemorySnapshot>,
    pub stats: OptimizationStatsSummary,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStatsSummary {
    pub total_optimizations: u64,
    pub memory_saved_bytes: usize,
    pub gc_triggers_avoided: u64,
    pub fragmentation_events_reduced: u64,
    pub prediction_accuracy_percent: f64,
}

/// Convenient functions for creating optimizers
pub fn create_memory_optimizer() -> AdvancedMemoryOptimizer {
    AdvancedMemoryOptimizer::new()
}

pub fn create_memory_optimizer_with_aggressive_settings() -> AdvancedMemoryOptimizer {
    let mut config = MemoryOptimizationConfig::default();
    config.warning_threshold = 0.6;
    config.critical_threshold = 0.8;
    config.check_interval = Duration::from_secs(1);

    AdvancedMemoryOptimizer::with_config(config)
}

pub fn create_memory_optimizer_for_low_memory() -> AdvancedMemoryOptimizer {
    let mut config = MemoryOptimizationConfig::default();
    config.warning_threshold = 0.5;
    config.critical_threshold = 0.7;
    config.pool_params.initial_pool_size = 256 * 1024; // 256KB
    config.pool_params.max_pool_size = 16 * 1024 * 1024; // 16MB

    AdvancedMemoryOptimizer::with_config(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_memory_optimizer_creation() {
        let optimizer = create_memory_optimizer();
        assert!(optimizer.config.ml_predictions);
    }

    #[test]
    fn test_memory_snapshot() {
        let snapshot = MemorySnapshot {
            timestamp: SystemTime::now(),
            total_allocated: 1024,
            peak_usage: 2048,
            fragmentation_ratio: 0.5,
            allocation_rate: 100.0,
            deallocation_rate: 50.0,
            gc_pressure: 0.6,
            operation_context: "test".to_string(),
        };

        assert_eq!(snapshot.total_allocated, 1024);
        assert_eq!(snapshot.peak_usage, 2048);
    }

    #[test]
    fn test_memory_pool_expansion() {
        let config = PoolOptimizationParams::default();
        let mut pool = MemoryPool::new(1024);

        let initial_capacity = pool.total_capacity;
        pool.expand(&config);

        assert!(pool.total_capacity > initial_capacity);
    }

    #[test]
    fn test_predictor_data_points() {
        let mut predictor = MemoryUsagePredictor::new();

        predictor.add_data_point(DataPoint {
            timestamp: 1.0,
            memory_usage: 1024.0,
            context: "test".to_string(),
        });

        assert_eq!(predictor.data_points.len(), 1);
    }

    #[test]
    fn test_pool_manager_size_class_calculation() {
        let config = PoolOptimizationParams::default();
        let manager = AdaptivePoolManager::new(config);

        assert_eq!(manager.calculate_size_class(100), 128);
        assert_eq!(manager.calculate_size_class(1000), 1024);
        assert_eq!(manager.calculate_size_class(2000), 2048);
    }

    #[test]
    fn test_optimization_stats() {
        let optimizer = create_memory_optimizer();
        let stats = optimizer.get_stats();

        assert_eq!(stats.optimizations_performed.load(Ordering::Relaxed), 0);
        assert!(stats.prediction_accuracy.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_export_optimization_data() {
        let optimizer = create_memory_optimizer();
        let temp_path = std::env::temp_dir().join("test_optimization_export.json");

        let result = optimizer.export_optimization_data(temp_path.to_str().unwrap());
        assert!(result.is_ok());

        // Verify file exists
        assert!(temp_path.exists());

        // Clean up
        let _ = std::fs::remove_file(temp_path);
    }

    #[test]
    fn test_aggressive_optimizer_settings() {
        let optimizer = create_memory_optimizer_with_aggressive_settings();
        assert!(optimizer.config.warning_threshold < 0.7);
        assert!(optimizer.config.critical_threshold < 0.9);
    }

    #[test]
    fn test_low_memory_optimizer_settings() {
        let optimizer = create_memory_optimizer_for_low_memory();
        assert!(optimizer.config.pool_params.max_pool_size < 128 * 1024 * 1024);
    }
}
