//! Memory Management for ZeRO-3 CPU Offloading
//!
//! This module implements intelligent memory management strategies for ZeRO-3
//! (Zero Redundancy Optimizer Stage 3) with CPU offloading. It provides
//! automatic memory optimization, garbage collection, and dynamic allocation
//! strategies to maximize performance while staying within memory budgets.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::TorshResult;
use log::info;
use std::sync::{Arc, Mutex};

use super::config::{AutoMemoryStrategy, CpuCompressionMethod, Zero3CpuOffloadConfig};

/// Memory manager for ZeRO-3 optimizations
///
/// Implements intelligent memory management strategies including:
/// - Automatic garbage collection of unused tensors
/// - Dynamic CPU/GPU offloading based on memory pressure
/// - Adaptive prefetch buffer sizing
/// - Dynamic compression level adjustment
/// - Memory pressure monitoring and optimization
pub struct Zero3MemoryManager {
    /// Configuration for memory management
    config: Zero3CpuOffloadConfig,
    /// Current memory usage statistics
    memory_stats: Arc<Mutex<Zero3MemoryStats>>,
    /// Memory pressure history for trend analysis
    pressure_history: Arc<Mutex<Vec<f32>>>,
    /// Optimization strategy state
    strategy_state: Arc<Mutex<MemoryStrategyState>>,
    /// Performance metrics for optimization decisions
    perf_metrics: Arc<Mutex<MemoryPerformanceMetrics>>,
}

impl Zero3MemoryManager {
    /// Create a new memory manager
    pub fn new(config: &Zero3CpuOffloadConfig) -> Self {
        Self {
            config: config.clone(),
            memory_stats: Arc::new(Mutex::new(Zero3MemoryStats::new())),
            pressure_history: Arc::new(Mutex::new(Vec::with_capacity(100))),
            strategy_state: Arc::new(Mutex::new(MemoryStrategyState::new())),
            perf_metrics: Arc::new(Mutex::new(MemoryPerformanceMetrics::new())),
        }
    }

    /// Check and optimize memory usage
    ///
    /// This is the main entry point for memory optimization. It analyzes current
    /// memory usage, calculates memory pressure, and applies appropriate optimization
    /// strategies based on the configured memory management approach.
    pub async fn check_and_optimize_memory(&self) -> TorshResult<()> {
        let start_time = std::time::Instant::now();

        // Update current memory statistics
        self.update_memory_statistics().await?;

        let current_stats = self.memory_stats.lock().unwrap().clone();
        let memory_pressure = self.calculate_memory_pressure(&current_stats);

        // Record pressure history for trend analysis
        {
            let mut history = self.pressure_history.lock().unwrap();
            history.push(memory_pressure);
            if history.len() > 100 {
                history.remove(0); // Keep only last 100 measurements
            }
        }

        info!(
            "   🧹 Memory optimization check - Pressure: {:.2}% (GPU: {:.1}MB, CPU: {:.1}MB)",
            memory_pressure * 100.0,
            current_stats.gpu_memory_used as f64 / (1024.0 * 1024.0),
            current_stats.cpu_memory_used as f64 / (1024.0 * 1024.0)
        );

        // Apply optimization strategies based on memory pressure and configuration
        match self.config.auto_memory_management {
            AutoMemoryStrategy::Conservative => {
                self.apply_conservative_strategy(memory_pressure).await?
            }
            AutoMemoryStrategy::Balanced => self.apply_balanced_strategy(memory_pressure).await?,
            AutoMemoryStrategy::Aggressive => {
                self.apply_aggressive_strategy(memory_pressure).await?
            }
            AutoMemoryStrategy::Extreme => self.apply_extreme_strategy(memory_pressure).await?,
        }

        // Record optimization performance
        let optimization_time = start_time.elapsed();
        {
            let mut metrics = self.perf_metrics.lock().unwrap();
            metrics.record_optimization_cycle(optimization_time, memory_pressure);
        }

        Ok(())
    }

    /// Apply conservative memory management strategy
    async fn apply_conservative_strategy(&self, memory_pressure: f32) -> TorshResult<()> {
        // Conservative approach: Only act when memory pressure is high
        if memory_pressure > 0.85 {
            info!("   🚨 Conservative strategy: High memory pressure detected");
            self.garbage_collect_unused_tensors().await?;
            self.selective_offload_to_cpu(0.3).await?; // Offload 30%
        } else if memory_pressure > 0.75 {
            info!("     Conservative strategy: Medium memory pressure");
            self.garbage_collect_unused_tensors().await?;
        }

        // Reduce prefetch only when absolutely necessary
        if memory_pressure > 0.9 {
            self.reduce_prefetch_buffers(0.8).await?; // Reduce to 80%
        }

        Ok(())
    }

    /// Apply balanced memory management strategy
    async fn apply_balanced_strategy(&self, memory_pressure: f32) -> TorshResult<()> {
        // Regular garbage collection
        if memory_pressure > 0.6 {
            self.garbage_collect_unused_tensors().await?;
        }

        // Progressive offloading based on pressure
        if memory_pressure > 0.8 {
            info!("   🚨 Balanced strategy: Aggressive CPU offloading");
            self.aggressive_offload_to_cpu(0.7).await?; // Offload 70%
        } else if memory_pressure > 0.65 {
            info!("     Balanced strategy: Selective CPU offloading");
            self.selective_offload_to_cpu(0.4).await?; // Offload 40%
        }

        // Adaptive prefetch management
        if memory_pressure > 0.75 {
            self.reduce_prefetch_buffers(0.6).await?;
        } else if memory_pressure < 0.4 {
            self.optimize_prefetch_strategy().await?;
        }

        // Dynamic compression when memory is tight
        if memory_pressure > 0.7 {
            self.enable_dynamic_compression().await?;
        }

        Ok(())
    }

    /// Apply aggressive memory management strategy
    async fn apply_aggressive_strategy(&self, memory_pressure: f32) -> TorshResult<()> {
        // Frequent garbage collection
        if memory_pressure > 0.5 {
            self.garbage_collect_unused_tensors().await?;
        }

        // Aggressive offloading at lower thresholds
        if memory_pressure > 0.7 {
            info!("   🚨 Aggressive strategy: Maximum CPU offloading");
            self.aggressive_offload_to_cpu(0.9).await?; // Offload 90%
            self.enable_dynamic_compression().await?;
        } else if memory_pressure > 0.5 {
            info!("     Aggressive strategy: Preemptive CPU offloading");
            self.selective_offload_to_cpu(0.6).await?; // Offload 60%
        }

        // Dynamic prefetch adjustment
        if memory_pressure > 0.6 {
            self.reduce_prefetch_buffers(0.5).await?;
        } else if memory_pressure < 0.3 {
            self.optimize_prefetch_strategy().await?;
        }

        // Proactive compression
        if memory_pressure > 0.6 {
            self.enable_dynamic_compression().await?;
        }

        Ok(())
    }

    /// Apply extreme memory management strategy
    async fn apply_extreme_strategy(&self, memory_pressure: f32) -> TorshResult<()> {
        // Continuous garbage collection
        if memory_pressure > 0.3 {
            self.garbage_collect_unused_tensors().await?;
        }

        // Maximum offloading at any sign of pressure
        if memory_pressure > 0.5 {
            info!("   🚨 Extreme strategy: Maximum CPU offloading and compression");
            self.aggressive_offload_to_cpu(0.95).await?; // Offload 95%
            self.enable_dynamic_compression().await?;
            self.reduce_prefetch_buffers(0.25).await?; // Minimal prefetch
        } else if memory_pressure > 0.3 {
            info!("     Extreme strategy: Preemptive optimization");
            self.selective_offload_to_cpu(0.8).await?; // Offload 80%
            self.enable_dynamic_compression().await?;
        }

        // Always use compression in extreme mode
        if memory_pressure > 0.4 {
            self.enable_dynamic_compression().await?;
        }

        // Memory defragmentation
        if memory_pressure > 0.6 {
            self.defragment_memory().await?;
        }

        Ok(())
    }

    /// Calculate memory pressure based on current memory usage
    pub fn calculate_memory_pressure(&self, stats: &Zero3MemoryStats) -> f32 {
        // Use configuration limits or reasonable defaults for total memory
        let gpu_memory_total = self.config.max_gpu_memory_mb * 1024 * 1024;
        let cpu_memory_total = self.config.max_cpu_memory_mb * 1024 * 1024;

        let gpu_pressure = if gpu_memory_total > 0 {
            stats.gpu_memory_used as f32 / gpu_memory_total as f32
        } else {
            0.0
        };

        let cpu_pressure = if cpu_memory_total > 0 {
            stats.cpu_memory_used as f32 / cpu_memory_total as f32
        } else {
            0.0
        };

        // Weight GPU memory pressure more heavily as it's typically the bottleneck
        // Also consider the rate of memory usage increase
        let base_pressure = 0.8 * gpu_pressure + 0.2 * cpu_pressure;

        // Add pressure from memory usage trend
        let trend_pressure = self.calculate_pressure_trend();

        (base_pressure + 0.1 * trend_pressure).min(1.0)
    }

    /// Calculate memory pressure trend from history
    fn calculate_pressure_trend(&self) -> f32 {
        let history = self.pressure_history.lock().unwrap();
        if history.len() < 5 {
            return 0.0;
        }

        // Calculate trend over last 5 measurements
        let recent: Vec<f32> = history.iter().rev().take(5).cloned().collect();
        let trend = (recent[0] - recent[4]) / 4.0; // Average change per measurement

        // Convert trend to pressure contribution (positive trend adds pressure)
        (trend * 5.0).clamp(0.0, 1.0)
    }

    /// Update memory statistics from system and component usage
    async fn update_memory_statistics(&self) -> TorshResult<()> {
        // In a real implementation, this would query:
        // - GPU memory usage from CUDA runtime
        // - CPU memory usage from system calls
        // - Parameter counts from storage managers
        // - Compression ratios from compression statistics

        let mut stats = self.memory_stats.lock().unwrap();

        // Mock update - in practice would gather real statistics
        stats.gpu_memory_used = self.estimate_gpu_memory_usage();
        stats.cpu_memory_used = self.estimate_cpu_memory_usage();
        stats.total_parameters = self.estimate_total_parameters();
        stats.parameters_on_cpu = self.estimate_parameters_on_cpu();
        stats.parameters_on_gpu = self.estimate_parameters_on_gpu();
        stats.compression_ratio = self.calculate_compression_ratio();

        Ok(())
    }

    /// Garbage collect unused tensors to free memory
    #[allow(unused_assignments)]
    async fn garbage_collect_unused_tensors(&self) -> TorshResult<()> {
        let start_time = std::time::Instant::now();
        let mut freed_bytes = 0;

        // In a real implementation, this would:
        // 1. Scan for tensors that haven't been accessed recently
        // 2. Check reference counts to identify unused tensors
        // 3. Free tensors that are no longer needed
        // 4. Consolidate fragmented memory
        // 5. Update GPU memory allocator state

        // Mock implementation: estimate garbage collection effectiveness
        let current_stats = self.memory_stats.lock().unwrap().clone();
        let estimated_unused = (current_stats.gpu_memory_used as f32 * 0.1) as usize; // Assume 10% is garbage
        freed_bytes = estimated_unused;

        // Simulate garbage collection work
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;

        if freed_bytes > 0 {
            info!(
                "   🗑️  Garbage collected {} MB of unused tensors in {:?}",
                freed_bytes / (1024 * 1024),
                start_time.elapsed()
            );

            // Update statistics
            let mut stats = self.memory_stats.lock().unwrap();
            stats.gpu_memory_used = stats.gpu_memory_used.saturating_sub(freed_bytes);
        }

        Ok(())
    }

    /// Aggressively offload parameters to CPU when memory pressure is high
    async fn aggressive_offload_to_cpu(&self, offload_ratio: f32) -> TorshResult<()> {
        info!(
            "   🚨 Aggressive CPU offloading: {:.0}% of parameters",
            offload_ratio * 100.0
        );

        let start_time = std::time::Instant::now();

        // In a real implementation, this would:
        // 1. Identify all parameters currently on GPU
        // 2. Rank parameters by access recency and importance
        // 3. Offload the specified percentage to CPU
        // 4. Use compression for CPU storage to maximize capacity
        // 5. Update GPU cache to reflect changes

        // Mock aggressive offloading
        let current_stats = self.memory_stats.lock().unwrap().clone();
        let target_offload_bytes = (current_stats.gpu_memory_used as f32 * offload_ratio) as usize;

        // Simulate offloading work
        let offload_time_ms = (target_offload_bytes as f64 / (1024.0 * 1024.0) * 10.0) as u64; // 10ms per MB
        tokio::time::sleep(tokio::time::Duration::from_millis(offload_time_ms)).await;

        // Update statistics
        {
            let mut stats = self.memory_stats.lock().unwrap();
            stats.gpu_memory_used = stats.gpu_memory_used.saturating_sub(target_offload_bytes);
            stats.cpu_memory_used += target_offload_bytes;
            stats.parameters_on_gpu =
                (stats.parameters_on_gpu as f32 * (1.0 - offload_ratio)) as usize;
            stats.parameters_on_cpu += (stats.total_parameters as f32 * offload_ratio) as usize;
        }

        info!(
            "   ⬇️  Offloaded {} MB to CPU in {:?}",
            target_offload_bytes / (1024 * 1024),
            start_time.elapsed()
        );

        Ok(())
    }

    /// Selectively offload parameters based on usage patterns
    async fn selective_offload_to_cpu(&self, offload_ratio: f32) -> TorshResult<()> {
        info!(
            "     Selective CPU offloading: {:.0}% of parameters",
            offload_ratio * 100.0
        );

        let start_time = std::time::Instant::now();

        // In a real implementation, this would:
        // 1. Analyze parameter access patterns
        // 2. Offload least recently used parameters
        // 3. Keep frequently accessed parameters on GPU
        // 4. Use historical data to predict future access
        // 5. Implement smart caching strategies

        let current_stats = self.memory_stats.lock().unwrap().clone();
        let target_offload_bytes = (current_stats.gpu_memory_used as f32 * offload_ratio) as usize;

        // Simulate selective offloading (more efficient than aggressive)
        let offload_time_ms = (target_offload_bytes as f64 / (1024.0 * 1024.0) * 5.0) as u64; // 5ms per MB
        tokio::time::sleep(tokio::time::Duration::from_millis(offload_time_ms)).await;

        // Update statistics
        {
            let mut stats = self.memory_stats.lock().unwrap();
            stats.gpu_memory_used = stats.gpu_memory_used.saturating_sub(target_offload_bytes);
            stats.cpu_memory_used += target_offload_bytes;
            stats.parameters_on_gpu =
                (stats.parameters_on_gpu as f32 * (1.0 - offload_ratio)) as usize;
            stats.parameters_on_cpu += (stats.total_parameters as f32 * offload_ratio) as usize;
        }

        info!(
            "   ⬇️  Selectively offloaded {} MB to CPU in {:?}",
            target_offload_bytes / (1024 * 1024),
            start_time.elapsed()
        );

        Ok(())
    }

    /// Optimize prefetch strategy based on available memory
    async fn optimize_prefetch_strategy(&self) -> TorshResult<()> {
        info!("    Low memory pressure - Optimizing prefetch strategy");

        // In a real implementation, this would:
        // 1. Increase prefetch buffer size
        // 2. Enable more aggressive prefetching
        // 3. Prefetch multiple layers ahead
        // 4. Use parallel prefetch streams
        // 5. Optimize prefetch scheduling

        let optimal_buffer_size = self.config.prefetch_buffer_size * 2;

        {
            let mut strategy_state = self.strategy_state.lock().unwrap();
            strategy_state.current_prefetch_multiplier = 2.0;
            strategy_state.prefetch_optimization_active = true;
        }

        info!(
            "   ⚡ Increasing prefetch buffer size to {} ({}x multiplier)",
            optimal_buffer_size, 2.0
        );

        Ok(())
    }

    /// Reduce prefetch buffers to conserve memory
    async fn reduce_prefetch_buffers(&self, reduction_factor: f32) -> TorshResult<()> {
        info!(
            "   📉 High memory pressure - Reducing prefetch buffers by {:.0}%",
            (1.0 - reduction_factor) * 100.0
        );

        // In a real implementation, this would:
        // 1. Reduce prefetch buffer size
        // 2. Limit number of prefetched parameters
        // 3. Use just-in-time loading instead of prefetching
        // 4. Prioritize immediate needs over speculative loading
        // 5. Disable parallel prefetch streams

        let minimal_buffer_size =
            ((self.config.prefetch_buffer_size as f32 * reduction_factor) as usize).max(1);

        {
            let mut strategy_state = self.strategy_state.lock().unwrap();
            strategy_state.current_prefetch_multiplier = reduction_factor;
            strategy_state.prefetch_optimization_active = false;
        }

        info!(
            "   🔻 Reduced prefetch buffer size to {} ({:.1}x multiplier)",
            minimal_buffer_size, reduction_factor
        );

        Ok(())
    }

    /// Enable dynamic compression based on memory pressure
    async fn enable_dynamic_compression(&self) -> TorshResult<()> {
        let current_compression = self.config.cpu_compression;

        let target_compression = match current_compression {
            CpuCompressionMethod::None => {
                info!("   🗜️  Enabling FP16 compression for CPU storage");
                CpuCompressionMethod::FP16
            }
            CpuCompressionMethod::FP16 => {
                info!("   🗜️  Upgrading to BF16 compression for CPU storage");
                CpuCompressionMethod::BF16
            }
            CpuCompressionMethod::BF16 => {
                info!("   🗜️  Upgrading to INT8 compression for CPU storage");
                CpuCompressionMethod::INT8
            }
            CpuCompressionMethod::INT8 => {
                info!("   🗜️  Upgrading to Quantization compression for CPU storage");
                CpuCompressionMethod::Quantization
            }
            _ => {
                info!("   🗜️  Maximum compression already enabled");
                current_compression
            }
        };

        {
            let mut strategy_state = self.strategy_state.lock().unwrap();
            strategy_state.dynamic_compression_level = target_compression;
            strategy_state.compression_upgrade_active = true;
        }

        Ok(())
    }

    /// Defragment memory to reduce fragmentation
    async fn defragment_memory(&self) -> TorshResult<()> {
        info!("   🔧 Defragmenting memory to reduce fragmentation");

        let start_time = std::time::Instant::now();

        // In a real implementation, this would:
        // 1. Analyze memory fragmentation patterns
        // 2. Consolidate small allocations
        // 3. Rearrange memory layout for better efficiency
        // 4. Update allocator metadata
        // 5. Optimize memory alignment

        // Simulate defragmentation work
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let current_stats = self.memory_stats.lock().unwrap().clone();
        let estimated_savings = (current_stats.gpu_memory_used as f32 * 0.05) as usize; // 5% savings

        if estimated_savings > 0 {
            // Update statistics to reflect defragmentation savings
            let mut stats = self.memory_stats.lock().unwrap();
            stats.gpu_memory_used = stats.gpu_memory_used.saturating_sub(estimated_savings);
        }

        info!(
            "   ✨ Memory defragmentation completed: {} MB saved in {:?}",
            estimated_savings / (1024 * 1024),
            start_time.elapsed()
        );

        Ok(())
    }

    /// Get current memory statistics
    pub fn get_memory_stats(&self) -> Zero3MemoryStats {
        self.memory_stats.lock().unwrap().clone()
    }

    /// Get memory management performance metrics
    pub fn get_performance_metrics(&self) -> MemoryPerformanceMetrics {
        self.perf_metrics.lock().unwrap().clone()
    }

    /// Get current strategy state
    pub fn get_strategy_state(&self) -> MemoryStrategyState {
        self.strategy_state.lock().unwrap().clone()
    }

    /// Force immediate memory optimization regardless of pressure
    pub async fn force_memory_optimization(&self) -> TorshResult<()> {
        info!("   🚨 Forced memory optimization requested");

        self.garbage_collect_unused_tensors().await?;
        self.aggressive_offload_to_cpu(0.8).await?;
        self.enable_dynamic_compression().await?;
        self.defragment_memory().await?;

        info!("    Forced memory optimization completed");
        Ok(())
    }

    // Helper methods for estimation (in practice, these would query real systems)

    fn estimate_gpu_memory_usage(&self) -> usize {
        // Mock GPU memory usage - in practice would query CUDA runtime
        1024 * 1024 * 1024 // 1GB
    }

    fn estimate_cpu_memory_usage(&self) -> usize {
        // Mock CPU memory usage - in practice would query system memory
        2 * 1024 * 1024 * 1024 // 2GB
    }

    fn estimate_total_parameters(&self) -> usize {
        1000000 // 1M parameters
    }

    fn estimate_parameters_on_cpu(&self) -> usize {
        700000 // 700K parameters on CPU
    }

    fn estimate_parameters_on_gpu(&self) -> usize {
        300000 // 300K parameters on GPU
    }

    fn calculate_compression_ratio(&self) -> f32 {
        match self.config.cpu_compression {
            CpuCompressionMethod::None => 1.0,
            CpuCompressionMethod::FP16 => 2.0,
            CpuCompressionMethod::BF16 => 2.0,
            CpuCompressionMethod::INT8 => 4.0,
            CpuCompressionMethod::Quantization => 8.0,
            CpuCompressionMethod::LosslessCompression => 1.5,
        }
    }
}

/// Memory statistics for ZeRO-3
#[derive(Debug, Clone)]
pub struct Zero3MemoryStats {
    /// Memory used on CPU in bytes
    pub cpu_memory_used: usize,
    /// Memory used on GPU in bytes
    pub gpu_memory_used: usize,
    /// Total number of parameters in the model
    pub total_parameters: usize,
    /// Number of parameters currently stored on CPU
    pub parameters_on_cpu: usize,
    /// Number of parameters currently cached on GPU
    pub parameters_on_gpu: usize,
    /// Compression ratio achieved (original_size / compressed_size)
    pub compression_ratio: f32,
}

impl Zero3MemoryStats {
    pub fn new() -> Self {
        Self {
            cpu_memory_used: 0,
            gpu_memory_used: 0,
            total_parameters: 0,
            parameters_on_cpu: 0,
            parameters_on_gpu: 0,
            compression_ratio: 1.0,
        }
    }

    /// Get total memory usage across CPU and GPU
    pub fn total_memory_used(&self) -> usize {
        self.cpu_memory_used + self.gpu_memory_used
    }

    /// Get memory usage efficiency (parameters per byte)
    pub fn memory_efficiency(&self) -> f32 {
        if self.total_memory_used() > 0 {
            self.total_parameters as f32 / self.total_memory_used() as f32
        } else {
            0.0
        }
    }

    /// Get CPU memory usage as percentage of total
    pub fn cpu_memory_percentage(&self) -> f32 {
        let total = self.total_memory_used();
        if total > 0 {
            self.cpu_memory_used as f32 / total as f32
        } else {
            0.0
        }
    }

    /// Get GPU memory usage as percentage of total
    pub fn gpu_memory_percentage(&self) -> f32 {
        let total = self.total_memory_used();
        if total > 0 {
            self.gpu_memory_used as f32 / total as f32
        } else {
            0.0
        }
    }
}

impl Default for Zero3MemoryStats {
    fn default() -> Self {
        Self::new()
    }
}

/// State tracking for memory management strategies
#[derive(Debug, Clone)]
pub struct MemoryStrategyState {
    /// Current prefetch buffer multiplier
    pub current_prefetch_multiplier: f32,
    /// Whether prefetch optimization is currently active
    pub prefetch_optimization_active: bool,
    /// Current dynamic compression level
    pub dynamic_compression_level: CpuCompressionMethod,
    /// Whether compression upgrade is active
    pub compression_upgrade_active: bool,
    /// Last memory pressure reading
    pub last_memory_pressure: f32,
    /// Number of optimization cycles performed
    pub optimization_cycles: u64,
    /// Timestamp of last optimization
    pub last_optimization_time: std::time::Instant,
}

impl MemoryStrategyState {
    pub fn new() -> Self {
        Self {
            current_prefetch_multiplier: 1.0,
            prefetch_optimization_active: false,
            dynamic_compression_level: CpuCompressionMethod::None,
            compression_upgrade_active: false,
            last_memory_pressure: 0.0,
            optimization_cycles: 0,
            last_optimization_time: std::time::Instant::now(),
        }
    }
}

impl Default for MemoryStrategyState {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance metrics for memory management operations
#[derive(Debug, Clone)]
pub struct MemoryPerformanceMetrics {
    /// Total number of optimization cycles
    pub total_optimization_cycles: u64,
    /// Total time spent in optimization
    pub total_optimization_time: std::time::Duration,
    /// Average optimization time per cycle
    pub average_optimization_time: std::time::Duration,
    /// Peak memory pressure observed
    pub peak_memory_pressure: f32,
    /// Average memory pressure
    pub average_memory_pressure: f32,
    /// Number of garbage collection operations
    pub garbage_collection_count: u64,
    /// Total memory freed by garbage collection
    pub total_memory_freed: usize,
    /// Number of CPU offload operations
    pub cpu_offload_count: u64,
    /// Total data offloaded to CPU
    pub total_data_offloaded: usize,
}

impl MemoryPerformanceMetrics {
    pub fn new() -> Self {
        Self {
            total_optimization_cycles: 0,
            total_optimization_time: std::time::Duration::ZERO,
            average_optimization_time: std::time::Duration::ZERO,
            peak_memory_pressure: 0.0,
            average_memory_pressure: 0.0,
            garbage_collection_count: 0,
            total_memory_freed: 0,
            cpu_offload_count: 0,
            total_data_offloaded: 0,
        }
    }

    /// Record a completed optimization cycle
    pub fn record_optimization_cycle(&mut self, duration: std::time::Duration, pressure: f32) {
        self.total_optimization_cycles += 1;
        self.total_optimization_time += duration;
        self.average_optimization_time =
            self.total_optimization_time / self.total_optimization_cycles as u32;

        if pressure > self.peak_memory_pressure {
            self.peak_memory_pressure = pressure;
        }

        // Update running average of memory pressure
        self.average_memory_pressure =
            (self.average_memory_pressure * (self.total_optimization_cycles - 1) as f32 + pressure)
                / self.total_optimization_cycles as f32;
    }

    /// Record garbage collection operation
    pub fn record_garbage_collection(&mut self, memory_freed: usize) {
        self.garbage_collection_count += 1;
        self.total_memory_freed += memory_freed;
    }

    /// Record CPU offload operation
    pub fn record_cpu_offload(&mut self, data_offloaded: usize) {
        self.cpu_offload_count += 1;
        self.total_data_offloaded += data_offloaded;
    }

    /// Get optimization efficiency (cycles per second)
    pub fn optimization_efficiency(&self) -> f64 {
        if !self.total_optimization_time.is_zero() {
            self.total_optimization_cycles as f64 / self.total_optimization_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Get memory management effectiveness (memory freed per operation)
    pub fn memory_management_effectiveness(&self) -> f64 {
        if self.garbage_collection_count > 0 {
            self.total_memory_freed as f64 / self.garbage_collection_count as f64
        } else {
            0.0
        }
    }
}

impl Default for MemoryPerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats_creation() {
        let stats = Zero3MemoryStats::new();
        assert_eq!(stats.cpu_memory_used, 0);
        assert_eq!(stats.gpu_memory_used, 0);
        assert_eq!(stats.total_memory_used(), 0);
        assert_eq!(stats.memory_efficiency(), 0.0);
    }

    #[test]
    fn test_memory_stats_calculations() {
        let mut stats = Zero3MemoryStats::new();
        stats.cpu_memory_used = 1000;
        stats.gpu_memory_used = 2000;
        stats.total_parameters = 100;

        assert_eq!(stats.total_memory_used(), 3000);
        assert!((stats.memory_efficiency() - 100.0 / 3000.0).abs() < 1e-6);
        assert!((stats.cpu_memory_percentage() - 1.0 / 3.0).abs() < 1e-6);
        assert!((stats.gpu_memory_percentage() - 2.0 / 3.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_memory_manager_creation() {
        let config = Zero3CpuOffloadConfig::default();
        let manager = Zero3MemoryManager::new(&config);

        let stats = manager.get_memory_stats();
        assert_eq!(stats.cpu_memory_used, 0);

        let metrics = manager.get_performance_metrics();
        assert_eq!(metrics.total_optimization_cycles, 0);
    }

    #[tokio::test]
    async fn test_memory_optimization() {
        let config = Zero3CpuOffloadConfig::default();
        let manager = Zero3MemoryManager::new(&config);

        // Test optimization doesn't crash
        manager.check_and_optimize_memory().await.unwrap();

        let metrics = manager.get_performance_metrics();
        assert_eq!(metrics.total_optimization_cycles, 1);
    }

    #[test]
    fn test_strategy_state() {
        let state = MemoryStrategyState::new();
        assert_eq!(state.current_prefetch_multiplier, 1.0);
        assert!(!state.prefetch_optimization_active);
        assert!(!state.compression_upgrade_active);
    }

    #[test]
    fn test_performance_metrics() {
        let mut metrics = MemoryPerformanceMetrics::new();

        metrics.record_optimization_cycle(std::time::Duration::from_millis(100), 0.5);
        assert_eq!(metrics.total_optimization_cycles, 1);
        assert_eq!(metrics.peak_memory_pressure, 0.5);

        metrics.record_garbage_collection(1000);
        assert_eq!(metrics.garbage_collection_count, 1);
        assert_eq!(metrics.total_memory_freed, 1000);
    }

    #[test]
    fn test_memory_pressure_calculation() {
        let config = Zero3CpuOffloadConfig {
            max_gpu_memory_mb: 1024, // 1GB
            max_cpu_memory_mb: 2048, // 2GB
            ..Zero3CpuOffloadConfig::default()
        };
        let manager = Zero3MemoryManager::new(&config);

        let stats = Zero3MemoryStats {
            gpu_memory_used: 512 * 1024 * 1024,  // 512MB
            cpu_memory_used: 1024 * 1024 * 1024, // 1GB
            ..Zero3MemoryStats::new()
        };

        let pressure = manager.calculate_memory_pressure(&stats);
        // GPU pressure: 0.5, CPU pressure: 0.5
        // Weighted: 0.8 * 0.5 + 0.2 * 0.5 = 0.5
        assert!((pressure - 0.5).abs() < 0.1);
    }
}
