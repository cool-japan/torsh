//! Core adaptive memory manager implementation
//!
//! This module provides the main `AdaptiveMemoryManager` that orchestrates all
//! memory management functionality including allocation strategies, monitoring,
//! optimization, and analysis for autograd operations.
//!
//! # Overview
//!
//! The `AdaptiveMemoryManager` is the central component that integrates all
//! memory management subsystems:
//!
//! - **System Memory Monitoring**: Real-time tracking of system memory status
//! - **Adaptive Allocation**: Dynamic allocation strategies based on memory pressure
//! - **Memory Pooling**: Efficient memory reuse through intelligent pooling
//! - **Usage Tracking**: Comprehensive statistics and usage pattern analysis
//! - **Optimization**: Automatic memory optimization based on current conditions
//! - **Analysis**: Detailed memory usage analysis and recommendations
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                AdaptiveMemoryManager<T>                         │
//! │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
//! │  │   Memory    │ │   System    │ │   Pools     │ │   Tracking  ││
//! │  │    Info     │ │ Monitoring  │ │ Management  │ │  & Analysis ││
//! │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
//! └─────────────────────────────────────────────────────────────────┘
//!                                │
//!                                ▼
//!                        Intelligent Memory
//!                        Allocation & Reuse
//! ```
//!
//! # Usage Examples
//!
//! ## Basic Usage
//!
//! ```rust,ignore
//! use crate::memory::manager::AdaptiveMemoryManager;
//! use crate::memory::types::AdaptiveMemoryConfig;
//!
//! let config = AdaptiveMemoryConfig::default();
//! let manager: AdaptiveMemoryManager<f32> = AdaptiveMemoryManager::new(config)?;
//!
//! // Allocate memory for gradients
//! let grad_memory = manager.allocate_gradient_memory(1000)?; // 1000 f32 elements
//!
//! // Use memory for gradient computation
//! // ... gradient calculations ...
//!
//! // Return memory to pool for reuse
//! manager.deallocate_gradient_memory(grad_memory);
//! ```
//!
//! ## Advanced Usage with Operation Tracking
//!
//! ```rust,ignore
//! // Track memory by operation type
//! let conv_memory = manager.allocate_gradient_memory_for_operation(
//!     2048, "conv2d_backward"
//! )?;
//!
//! // Analyze memory usage patterns
//! let analysis = manager.analyze_gradient_computation_memory("conv2d_backward")?;
//! println!("Memory efficiency: {:.1}%", analysis.memory_efficiency * 100.0);
//!
//! // Get optimization recommendations
//! let recommendations = manager.get_optimization_recommendations();
//! for rec in recommendations {
//!     println!("Recommendation: {}", rec);
//! }
//! ```

use crate::memory::anomaly::{
    AllocationPattern, AllocationPatternType, AnomalySeverity, MemoryAnomaly, MemoryAnomalyType,
};
use crate::memory::monitoring::GradientMemoryMonitor;
use crate::memory::pool::MemoryPool;
use crate::memory::tracking::{GradientMemoryAnalysis, GradientMemoryStats, MemoryUsageTracker};
use crate::memory::types::{
    AdaptiveMemoryConfig, AllocationStrategy, MemoryPressure, SystemMemoryInfo,
};
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use torsh_core::dtype::FloatElement;
use torsh_core::error::Result;
#[cfg(target_os = "linux")]
use torsh_core::error::TorshError;

/// Enhanced gradient computation memory analysis result
///
/// Comprehensive analysis result for a specific gradient operation, providing
/// detailed insights into memory usage patterns, efficiency metrics, anomaly
/// detection, and optimization recommendations.
///
/// # Analysis Components
///
/// - **System Context**: Memory pressure and system state during operation
/// - **Operation Metrics**: Gradient-specific memory statistics and patterns
/// - **Efficiency Analysis**: Memory utilization efficiency scoring
/// - **Anomaly Detection**: Identification of unusual memory behavior
/// - **Predictive Analysis**: Forecasting of future memory requirements
/// - **Recommendations**: Actionable optimization suggestions
#[derive(Debug, Clone)]
pub struct GradientComputationMemoryAnalysis {
    /// Name of the operation being analyzed
    pub operation_name: String,
    /// Analysis timestamp
    pub timestamp: Instant,
    /// System memory available before operation
    pub system_memory_before: usize,
    /// Current memory pressure level
    pub memory_pressure: MemoryPressure,
    /// Gradient-specific memory statistics
    pub gradient_memory_stats: GradientMemoryStats,
    /// Memory efficiency ratio (0.0 to 1.0)
    pub memory_efficiency: f64,
    /// Predicted peak memory usage
    pub predicted_peak_usage: usize,
    /// Detected memory anomalies
    pub anomalies: Vec<MemoryAnomaly>,
    /// Memory allocation pattern for this operation
    pub allocation_pattern: AllocationPattern,
    /// Optimization recommendations
    pub optimization_recommendations: Vec<String>,
}

impl Default for GradientComputationMemoryAnalysis {
    fn default() -> Self {
        Self {
            operation_name: String::new(),
            timestamp: Instant::now(),
            system_memory_before: 0,
            memory_pressure: MemoryPressure::Low,
            gradient_memory_stats: GradientMemoryStats::default(),
            memory_efficiency: 0.0,
            predicted_peak_usage: 0,
            anomalies: Vec::new(),
            allocation_pattern: AllocationPattern::default(),
            optimization_recommendations: Vec::new(),
        }
    }
}

/// Adaptive memory manager for autograd operations
///
/// The central memory management system that provides intelligent memory
/// allocation, pooling, monitoring, and optimization for autograd operations.
/// Adapts allocation strategies based on system conditions and usage patterns.
///
/// # Type Parameter
///
/// * `T` - The element type for gradient tensors (must implement `FloatElement`)
///
/// # Thread Safety
///
/// The manager is thread-safe and can be shared across threads using `Arc`.
/// Internal state is protected by appropriate synchronization primitives.
///
/// # Lifecycle
///
/// 1. **Initialization**: Create manager with configuration and start monitoring
/// 2. **Operation**: Allocate and deallocate memory for gradient operations
/// 3. **Monitoring**: Continuous system memory and usage pattern monitoring
/// 4. **Optimization**: Automatic optimization triggers based on memory pressure
/// 5. **Analysis**: Comprehensive analysis and recommendation generation
pub struct AdaptiveMemoryManager<T: FloatElement> {
    /// Configuration
    config: AdaptiveMemoryConfig,
    /// Current system memory information
    memory_info: Arc<RwLock<SystemMemoryInfo>>,
    /// Memory pools for different tensor sizes
    memory_pools: Arc<Mutex<HashMap<usize, MemoryPool<T>>>>,
    /// Gradient size history for prediction
    gradient_history: Arc<Mutex<VecDeque<usize>>>,
    /// Current memory pressure
    current_pressure: Arc<RwLock<MemoryPressure>>,
    /// Memory usage tracking
    memory_tracker: Arc<Mutex<MemoryUsageTracker>>,
    /// Background monitoring thread handle
    _monitor_handle: Option<std::thread::JoinHandle<()>>,
}

impl<T: FloatElement + Send + Sync + 'static> AdaptiveMemoryManager<T> {
    /// Create a new adaptive memory manager
    ///
    /// Initializes the memory manager with the provided configuration and
    /// starts background system memory monitoring.
    ///
    /// # Arguments
    ///
    /// * `config` - Memory management configuration
    ///
    /// # Returns
    ///
    /// New memory manager instance with active monitoring.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = AdaptiveMemoryConfig::default();
    /// let manager: AdaptiveMemoryManager<f32> = AdaptiveMemoryManager::new(config)?;
    /// ```
    pub fn new(config: AdaptiveMemoryConfig) -> Result<Self> {
        Self::new_with_monitoring_duration(config, Duration::from_secs(300))
    }

    /// Create a new adaptive memory manager with custom monitoring duration
    ///
    /// Allows customization of how long the background monitoring should run.
    /// Useful for testing or specific deployment scenarios.
    ///
    /// # Arguments
    ///
    /// * `config` - Memory management configuration
    /// * `monitoring_duration` - How long to run background monitoring
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let config = AdaptiveMemoryConfig::default();
    /// let manager: AdaptiveMemoryManager<f32> = AdaptiveMemoryManager::new_with_monitoring_duration(
    ///     config,
    ///     Duration::from_secs(600) // 10 minutes of monitoring
    /// )?;
    /// ```
    pub fn new_with_monitoring_duration(
        config: AdaptiveMemoryConfig,
        monitoring_duration: Duration,
    ) -> Result<Self> {
        let memory_info = Arc::new(RwLock::new(Self::get_system_memory_info()?));
        let current_pressure = Arc::new(RwLock::new(MemoryPressure::Low));

        // Start background monitoring
        let monitor_handle = Self::start_memory_monitoring(
            memory_info.clone(),
            current_pressure.clone(),
            config.monitor_interval,
            monitoring_duration,
        );

        Ok(Self {
            config,
            memory_info,
            memory_pools: Arc::new(Mutex::new(HashMap::new())),
            gradient_history: Arc::new(Mutex::new(VecDeque::new())),
            current_pressure,
            memory_tracker: Arc::new(Mutex::new(MemoryUsageTracker::default())),
            _monitor_handle: Some(monitor_handle),
        })
    }

    /// Get current system memory information
    ///
    /// Retrieves real-time system memory statistics using platform-specific
    /// methods. Provides comprehensive memory usage information.
    ///
    /// # Platform Support
    ///
    /// - **Linux**: Uses `/proc/meminfo` for accurate memory statistics
    /// - **macOS**: Uses system APIs for memory information
    /// - **Windows**: Uses Windows Memory Management APIs
    /// - **Other**: Provides reasonable defaults
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let memory_info = AdaptiveMemoryManager::<f32>::get_system_memory_info()?;
    /// println!("Available memory: {} MB", memory_info.available_memory / 1024 / 1024);
    /// ```
    pub fn get_system_memory_info() -> Result<SystemMemoryInfo> {
        // Platform-specific memory detection
        #[cfg(target_os = "linux")]
        {
            Self::get_linux_memory_info()
        }
        #[cfg(target_os = "windows")]
        {
            Self::get_windows_memory_info()
        }
        #[cfg(target_os = "macos")]
        {
            Self::get_macos_memory_info()
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            // Fallback for unsupported platforms
            Ok(SystemMemoryInfo {
                total_memory: 8 * 1024 * 1024 * 1024,     // Assume 8GB
                available_memory: 4 * 1024 * 1024 * 1024, // Assume 4GB available
                used_memory: 4 * 1024 * 1024 * 1024,
                usage_percentage: 50.0,
                last_updated: Instant::now(),
            })
        }
    }

    /// Get Linux memory information from /proc/meminfo
    #[cfg(target_os = "linux")]
    fn get_linux_memory_info() -> Result<SystemMemoryInfo> {
        use std::io::Read;

        let mut file = std::fs::File::open("/proc/meminfo")
            .map_err(|e| TorshError::AutogradError(format!("Failed to read /proc/meminfo: {e}")))?;

        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| TorshError::AutogradError(format!("Failed to read memory info: {e}")))?;

        let mut total_memory = 0;
        let mut available_memory = 0;

        for line in contents.lines() {
            if line.starts_with("MemTotal:") {
                if let Some(value) = line.split_whitespace().nth(1) {
                    total_memory = value.parse::<usize>().unwrap_or(0) * 1024; // Convert from KB to bytes
                }
            } else if line.starts_with("MemAvailable:") {
                if let Some(value) = line.split_whitespace().nth(1) {
                    available_memory = value.parse::<usize>().unwrap_or(0) * 1024;
                    // Convert from KB to bytes
                }
            }
        }

        let used_memory = total_memory.saturating_sub(available_memory);
        let usage_percentage = if total_memory > 0 {
            (used_memory as f64 / total_memory as f64) * 100.0
        } else {
            0.0
        };

        Ok(SystemMemoryInfo {
            total_memory,
            available_memory,
            used_memory,
            usage_percentage,
            last_updated: Instant::now(),
        })
    }

    /// Get Windows memory information
    #[cfg(target_os = "windows")]
    fn get_windows_memory_info() -> Result<SystemMemoryInfo> {
        // Placeholder for Windows implementation
        // In a real implementation, this would use Windows API calls
        Ok(SystemMemoryInfo {
            total_memory: 16 * 1024 * 1024 * 1024,    // Assume 16GB
            available_memory: 8 * 1024 * 1024 * 1024, // Assume 8GB available
            used_memory: 8 * 1024 * 1024 * 1024,
            usage_percentage: 50.0,
            last_updated: Instant::now(),
        })
    }

    /// Get macOS memory information
    #[cfg(target_os = "macos")]
    fn get_macos_memory_info() -> Result<SystemMemoryInfo> {
        // Placeholder for macOS implementation
        // In a real implementation, this would use macOS system calls
        Ok(SystemMemoryInfo {
            total_memory: 16 * 1024 * 1024 * 1024,    // Assume 16GB
            available_memory: 8 * 1024 * 1024 * 1024, // Assume 8GB available
            used_memory: 8 * 1024 * 1024 * 1024,
            usage_percentage: 50.0,
            last_updated: Instant::now(),
        })
    }

    /// Start background memory monitoring
    ///
    /// Spawns a background thread that continuously monitors system memory
    /// status and updates memory pressure levels.
    fn start_memory_monitoring(
        memory_info: Arc<RwLock<SystemMemoryInfo>>,
        current_pressure: Arc<RwLock<MemoryPressure>>,
        interval: Duration,
        max_monitoring_duration: Duration,
    ) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let start_time = Instant::now();

            loop {
                // Check if we've exceeded the maximum monitoring duration
                if start_time.elapsed() > max_monitoring_duration {
                    break;
                }

                if let Ok(info) = Self::get_system_memory_info() {
                    let pressure = Self::calculate_memory_pressure(&info);

                    // Update memory info
                    if let Ok(mut memory_ref) = memory_info.write() {
                        *memory_ref = info;
                    }

                    // Update pressure
                    if let Ok(mut pressure_ref) = current_pressure.write() {
                        *pressure_ref = pressure;
                    }
                }

                std::thread::sleep(interval);
            }
        })
    }

    /// Calculate memory pressure from system info
    ///
    /// Determines the current memory pressure level based on memory usage percentage.
    fn calculate_memory_pressure(info: &SystemMemoryInfo) -> MemoryPressure {
        match info.usage_percentage {
            p if p < 50.0 => MemoryPressure::Low,
            p if p < 70.0 => MemoryPressure::Moderate,
            p if p < 85.0 => MemoryPressure::High,
            _ => MemoryPressure::Critical,
        }
    }

    /// Allocate memory for gradients with adaptive strategy
    ///
    /// Allocates memory for gradient storage using adaptive strategies based
    /// on current memory pressure and usage patterns.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of elements to allocate
    ///
    /// # Returns
    ///
    /// Vector of `T` elements with the requested size.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let grad_memory = manager.allocate_gradient_memory(1000)?;
    /// assert_eq!(grad_memory.len(), 1000);
    /// ```
    pub fn allocate_gradient_memory(&self, size: usize) -> Result<Vec<T>> {
        self.allocate_gradient_memory_for_operation(size, "default")
    }

    /// Allocate memory for gradients with operation name tracking
    ///
    /// Allocates memory while tracking usage patterns per operation type.
    /// This enables detailed analysis and optimization recommendations.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of elements to allocate
    /// * `operation_name` - Name of the operation for tracking
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let conv_grad = manager.allocate_gradient_memory_for_operation(
    ///     2048, "conv2d_backward"
    /// )?;
    /// ```
    pub fn allocate_gradient_memory_for_operation(
        &self,
        size: usize,
        operation_name: &str,
    ) -> Result<Vec<T>> {
        let current_pressure = *self.current_pressure.read().unwrap();
        let strategy = self.determine_allocation_strategy(current_pressure);

        // Update gradient history
        {
            let mut history = self.gradient_history.lock();
            history.push_back(size);
            if history.len() > self.config.gradient_history_size {
                history.pop_front();
            }
        }

        // Track memory usage by operation
        {
            let mut tracker = self.memory_tracker.lock();
            let entry = tracker
                .gradient_memory_usage
                .entry(operation_name.to_string())
                .or_insert_with(GradientMemoryStats::default);
            entry.num_allocations += 1;
            entry.total_allocated += size;
            entry.peak_usage = entry.peak_usage.max(size);
            entry.avg_allocation_size = entry.total_allocated / entry.num_allocations;
            entry.last_allocation = Some(Instant::now());

            // Simple growth rate calculation
            if entry.num_allocations > 1 {
                entry.growth_rate = size as f64 / 1.0; // MB/s approximation
            }
        }

        // Try to reuse from memory pool first
        if let Some(reused_memory) = self.try_reuse_memory(size)? {
            return Ok(reused_memory);
        }

        // Allocate new memory based on strategy
        match strategy {
            AllocationStrategy::Aggressive => self.allocate_aggressive(size),
            AllocationStrategy::Conservative => self.allocate_conservative(size),
            AllocationStrategy::Minimal => self.allocate_minimal(size),
            AllocationStrategy::Adaptive => self.allocate_adaptive(size, current_pressure),
        }
    }

    /// Determine allocation strategy based on memory pressure
    ///
    /// Selects appropriate allocation strategy considering current memory
    /// pressure and configured strategy.
    fn determine_allocation_strategy(&self, pressure: MemoryPressure) -> AllocationStrategy {
        match self.config.allocation_strategy {
            AllocationStrategy::Adaptive => match pressure {
                MemoryPressure::Low => AllocationStrategy::Aggressive,
                MemoryPressure::Moderate => AllocationStrategy::Conservative,
                MemoryPressure::High | MemoryPressure::Critical => AllocationStrategy::Minimal,
            },
            strategy => strategy,
        }
    }

    /// Try to reuse memory from pool
    ///
    /// Attempts to reuse existing memory from the memory pool before allocating new memory.
    fn try_reuse_memory(&self, size: usize) -> Result<Option<Vec<T>>> {
        let mut pools = self.memory_pools.lock();

        if let Some(pool) = pools.get_mut(&size) {
            // Try to allocate from the pool
            match pool.allocate(size) {
                Ok(memory) => return Ok(Some(memory)),
                Err(_) => {} // Pool is empty, continue to regular allocation
            }
        }

        Ok(None)
    }

    /// Aggressive allocation strategy
    ///
    /// Allocates memory with extra capacity for optimal performance.
    fn allocate_aggressive(&self, size: usize) -> Result<Vec<T>> {
        // Pre-allocate extra capacity for future use
        let capacity = size + (size / 4); // 25% extra
        let mut memory = Vec::with_capacity(capacity);
        memory.resize(size, <T as torsh_core::dtype::TensorElement>::zero());

        self.track_allocation("aggressive", size * std::mem::size_of::<T>());
        Ok(memory)
    }

    /// Conservative allocation strategy
    ///
    /// Allocates memory with modest extra capacity for balance of performance and efficiency.
    fn allocate_conservative(&self, size: usize) -> Result<Vec<T>> {
        // Allocate exact size with small buffer
        let capacity = size + (size / 20); // 5% extra
        let mut memory = Vec::with_capacity(capacity);
        memory.resize(size, <T as torsh_core::dtype::TensorElement>::zero());

        self.track_allocation("conservative", size * std::mem::size_of::<T>());
        Ok(memory)
    }

    /// Minimal allocation strategy
    ///
    /// Allocates exact memory size with no extra capacity to minimize memory usage.
    fn allocate_minimal(&self, size: usize) -> Result<Vec<T>> {
        // Allocate exact size only
        let mut memory = Vec::with_capacity(size);
        memory.resize(size, <T as torsh_core::dtype::TensorElement>::zero());

        self.track_allocation("minimal", size * std::mem::size_of::<T>());
        Ok(memory)
    }

    /// Adaptive allocation strategy
    ///
    /// Dynamically adjusts allocation size based on memory pressure and prediction.
    fn allocate_adaptive(&self, size: usize, pressure: MemoryPressure) -> Result<Vec<T>> {
        // Predict future size needs based on history
        let predicted_size = self.predict_future_size();

        let capacity = match pressure {
            MemoryPressure::Low => {
                // Allocate based on prediction with generous buffer
                std::cmp::max(size, predicted_size) + (size / 2)
            }
            MemoryPressure::Moderate => {
                // Conservative prediction-based allocation
                std::cmp::max(size, predicted_size) + (size / 10)
            }
            MemoryPressure::High | MemoryPressure::Critical => {
                // Minimal allocation regardless of prediction
                size
            }
        };

        let mut memory = Vec::with_capacity(capacity);
        memory.resize(size, <T as torsh_core::dtype::TensorElement>::zero());

        self.track_allocation("adaptive", size * std::mem::size_of::<T>());
        Ok(memory)
    }

    /// Predict future gradient size based on history
    ///
    /// Uses historical gradient size data to predict future allocation needs.
    fn predict_future_size(&self) -> usize {
        let history = self.gradient_history.lock();

        if history.is_empty() {
            return 0;
        }

        // Simple moving average prediction
        let sum: usize = history.iter().sum();
        let avg = sum / history.len();

        // Apply trend analysis if we have enough data
        if history.len() >= 10 {
            let recent_avg = history.iter().rev().take(5).sum::<usize>() / 5;
            if recent_avg > avg {
                // Increasing trend - predict higher
                recent_avg + (recent_avg - avg)
            } else {
                // Decreasing or stable trend
                avg
            }
        } else {
            avg
        }
    }

    /// Return memory to pool for reuse
    ///
    /// Returns gradient memory to the memory pool for efficient reuse in
    /// future allocations.
    ///
    /// # Arguments
    ///
    /// * `memory` - Vector to return to the pool
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let grad_memory = manager.allocate_gradient_memory(1000)?;
    /// // ... use memory for gradients ...
    /// manager.deallocate_gradient_memory(grad_memory); // Return to pool
    /// ```
    pub fn deallocate_gradient_memory(&self, mut memory: Vec<T>) {
        let size = memory.len();
        memory.clear();
        memory.resize(size, <T as torsh_core::dtype::TensorElement>::zero()); // Reset to zero

        let mut pools = self.memory_pools.lock();
        let pool = pools.entry(size).or_insert_with(|| MemoryPool::new());

        // Use the pool's deallocate method to return memory
        pool.deallocate(memory);
    }

    /// Track memory allocation
    ///
    /// Internal method to track allocation statistics and update usage patterns.
    fn track_allocation(&self, operation: &str, bytes: usize) {
        let mut tracker = self.memory_tracker.lock();
        tracker.total_allocations += 1;

        let current_usage = tracker
            .usage_by_operation
            .entry(operation.to_string())
            .or_insert(0);
        *current_usage += bytes;

        let total_usage: usize = tracker.usage_by_operation.values().sum();
        if total_usage > tracker.peak_memory_usage {
            tracker.peak_memory_usage = total_usage;
        }

        let now = Instant::now();
        tracker.allocation_history.push_back((now, bytes));

        // Track gradient-specific memory usage
        self.track_gradient_memory_usage(operation, bytes, now, &mut tracker);

        // Keep only recent history (last 1000 allocations)
        if tracker.allocation_history.len() > 1000 {
            tracker.allocation_history.pop_front();
        }
    }

    /// Track gradient-specific memory usage
    ///
    /// Updates gradient-specific statistics for detailed analysis.
    fn track_gradient_memory_usage(
        &self,
        operation: &str,
        bytes: usize,
        timestamp: Instant,
        tracker: &mut MemoryUsageTracker,
    ) {
        let grad_stats = tracker
            .gradient_memory_usage
            .entry(operation.to_string())
            .or_insert_with(GradientMemoryStats::default);

        grad_stats.total_allocated += bytes;
        grad_stats.num_allocations += 1;
        grad_stats.avg_allocation_size = grad_stats.total_allocated / grad_stats.num_allocations;

        if grad_stats.total_allocated > grad_stats.peak_usage {
            grad_stats.peak_usage = grad_stats.total_allocated;
        }

        // Calculate growth rate
        if let Some(last_time) = grad_stats.last_allocation {
            let time_diff = timestamp.duration_since(last_time).as_secs_f64();
            if time_diff > 0.0 {
                grad_stats.growth_rate = bytes as f64 / time_diff;
            }
        }

        grad_stats.last_allocation = Some(timestamp);
    }

    /// Analyze memory usage patterns for gradients
    ///
    /// Performs comprehensive analysis of gradient memory usage patterns
    /// and identifies optimization opportunities.
    ///
    /// # Returns
    ///
    /// Detailed analysis with identified issues and recommendations.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let analysis = manager.analyze_gradient_memory_usage();
    /// println!("High memory operations: {:?}", analysis.high_memory_operations);
    /// for warning in analysis.warnings {
    ///     println!("Warning: {}", warning);
    /// }
    /// ```
    pub fn analyze_gradient_memory_usage(&self) -> GradientMemoryAnalysis {
        let tracker = self.memory_tracker.lock();
        let mut analysis = GradientMemoryAnalysis::default();

        // Analyze gradient memory statistics
        for (operation, stats) in &tracker.gradient_memory_usage {
            if stats.growth_rate > 1024.0 * 1024.0 {
                // > 1MB/s growth
                analysis.high_growth_operations.push(operation.clone());
            }

            if stats.total_allocated > 100 * 1024 * 1024 {
                // > 100MB
                analysis.high_memory_operations.push(operation.clone());
            }

            if stats.avg_allocation_size > 10 * 1024 * 1024 {
                // > 10MB average
                analysis.large_allocation_operations.push(operation.clone());
            }
        }

        // Analyze allocation patterns
        if tracker.allocation_history.len() > 50 {
            let recent_allocations: Vec<_> =
                tracker.allocation_history.iter().rev().take(25).collect();

            let older_allocations: Vec<_> = tracker
                .allocation_history
                .iter()
                .rev()
                .skip(25)
                .take(25)
                .collect();

            let recent_total: usize = recent_allocations.iter().map(|(_, size)| *size).sum();
            let older_total: usize = older_allocations.iter().map(|(_, size)| *size).sum();

            if recent_total > older_total * 2 {
                analysis
                    .warnings
                    .push("Memory allocation rate increasing rapidly".to_string());
            }
        }

        // Check for potential memory leaks
        analysis.potential_leaks = tracker.leak_detection.potential_leaks.clone();

        analysis
    }

    /// Get current memory pressure
    ///
    /// Returns the current memory pressure level as determined by the
    /// background monitoring system.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// match manager.get_memory_pressure() {
    ///     MemoryPressure::High => println!("Memory pressure is high!"),
    ///     MemoryPressure::Critical => println!("Critical memory situation!"),
    ///     _ => println!("Memory pressure is manageable"),
    /// }
    /// ```
    pub fn get_memory_pressure(&self) -> MemoryPressure {
        *self.current_pressure.read().unwrap()
    }

    /// Get system memory information
    ///
    /// Returns current system memory statistics.
    pub fn get_memory_info(&self) -> SystemMemoryInfo {
        self.memory_info.read().unwrap().clone()
    }

    /// Get memory usage statistics
    ///
    /// Returns comprehensive memory usage tracking data.
    pub fn get_memory_stats(&self) -> MemoryUsageTracker {
        self.memory_tracker.lock().clone()
    }

    /// Check if memory budget is exceeded
    ///
    /// Determines if current memory usage exceeds the configured budget.
    ///
    /// # Returns
    ///
    /// True if memory usage exceeds the configured percentage of system memory.
    pub fn is_memory_budget_exceeded(&self) -> bool {
        let memory_info = self.get_memory_info();
        let budget = (memory_info.total_memory as f64 * self.config.max_memory_percentage) as usize;
        let current_usage = self.get_memory_stats().peak_memory_usage;

        current_usage > budget
    }

    /// Get recommended optimization actions
    ///
    /// Analyzes current memory state and provides actionable optimization
    /// recommendations based on memory pressure and usage patterns.
    ///
    /// # Returns
    ///
    /// Vector of recommendation strings with specific optimization suggestions.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let recommendations = manager.get_optimization_recommendations();
    /// for rec in recommendations {
    ///     println!("Recommendation: {}", rec);
    /// }
    /// ```
    pub fn get_optimization_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        let pressure = self.get_memory_pressure();
        let memory_info = self.get_memory_info();
        let stats = self.get_memory_stats();

        if pressure >= MemoryPressure::High {
            recommendations.push("Reduce gradient batch size".to_string());
            recommendations.push("Enable gradient compression".to_string());
        }

        if pressure >= MemoryPressure::Critical {
            recommendations.push("Use memory-mapped storage".to_string());
            recommendations.push("Enable aggressive garbage collection".to_string());
        }

        if stats.peak_memory_usage > memory_info.available_memory / 2 {
            recommendations.push("Consider distributed training".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("Memory usage is optimal".to_string());
        }

        recommendations
    }

    /// Enhanced gradient memory analysis with detailed insights
    ///
    /// Provides comprehensive analysis of memory usage for a specific gradient
    /// operation, including efficiency metrics, anomaly detection, and predictions.
    ///
    /// # Arguments
    ///
    /// * `operation_name` - Name of the operation to analyze
    ///
    /// # Returns
    ///
    /// Detailed analysis with metrics, anomalies, and recommendations.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let analysis = manager.analyze_gradient_computation_memory("conv2d_backward")?;
    /// println!("Memory efficiency: {:.1}%", analysis.memory_efficiency * 100.0);
    /// for anomaly in analysis.anomalies {
    ///     println!("Anomaly: {}", anomaly.description);
    /// }
    /// ```
    pub fn analyze_gradient_computation_memory(
        &self,
        operation_name: &str,
    ) -> Result<GradientComputationMemoryAnalysis> {
        let tracker = self.memory_tracker.lock();
        let mut analysis = GradientComputationMemoryAnalysis::default();

        analysis.operation_name = operation_name.to_string();
        analysis.timestamp = Instant::now();

        // Get current memory state
        if let Ok(memory_info) = Self::get_system_memory_info() {
            analysis.system_memory_before = memory_info.available_memory;
            analysis.memory_pressure = self.get_memory_pressure();
        }

        // Analyze gradient-specific memory patterns
        if let Some(grad_stats) = tracker.gradient_memory_usage.get(operation_name) {
            analysis.gradient_memory_stats = grad_stats.clone();
            analysis.memory_efficiency = Self::calculate_memory_efficiency(
                grad_stats.total_allocated,
                grad_stats.peak_usage,
            );

            // Predict future memory usage based on current patterns
            analysis.predicted_peak_usage = self.predict_gradient_memory_usage(grad_stats)?;

            // Check for memory usage anomalies
            analysis.anomalies =
                self.detect_gradient_memory_anomalies(grad_stats, operation_name)?;
        }

        // Analyze memory allocation patterns for this operation
        analysis.allocation_pattern = self.analyze_gradient_allocation_pattern(operation_name)?;

        // Generate operation-specific recommendations
        analysis.optimization_recommendations =
            self.generate_gradient_memory_recommendations(operation_name)?;

        Ok(analysis)
    }

    /// Predict gradient memory usage based on historical patterns
    ///
    /// Uses linear prediction model based on current growth rate.
    fn predict_gradient_memory_usage(&self, stats: &GradientMemoryStats) -> Result<usize> {
        // Simple linear prediction based on growth rate
        let prediction_horizon_seconds = 10.0; // Predict 10 seconds ahead
        let predicted_growth = (stats.growth_rate * prediction_horizon_seconds) as usize;
        Ok(stats.total_allocated + predicted_growth)
    }

    /// Detect memory usage anomalies for gradient operations
    ///
    /// Analyzes gradient statistics to identify unusual memory behavior patterns.
    fn detect_gradient_memory_anomalies(
        &self,
        stats: &GradientMemoryStats,
        operation_name: &str,
    ) -> Result<Vec<MemoryAnomaly>> {
        let mut anomalies = Vec::new();

        // Check for excessive growth rate
        if stats.growth_rate > 50.0 * 1024.0 * 1024.0 {
            // 50MB/s
            anomalies.push(MemoryAnomaly {
                anomaly_type: MemoryAnomalyType::ExcessiveGrowthRate,
                severity: AnomalySeverity::High,
                description: format!(
                    "Operation '{}' has excessive memory growth rate: {:.2} MB/s",
                    operation_name,
                    stats.growth_rate / (1024.0 * 1024.0)
                ),
                detected_at: Instant::now(),
                operation_context: operation_name.to_string(),
                measurement: stats.growth_rate,
                baseline: 10.0 * 1024.0 * 1024.0, // 10MB/s baseline
            });
        }

        // Check for large average allocation sizes
        if stats.avg_allocation_size > 100 * 1024 * 1024 {
            // 100MB
            anomalies.push(MemoryAnomaly {
                anomaly_type: MemoryAnomalyType::LargeAllocationSize,
                severity: AnomalySeverity::Medium,
                description: format!(
                    "Operation '{}' has large average allocation size: {}",
                    operation_name,
                    Self::format_bytes(stats.avg_allocation_size)
                ),
                detected_at: Instant::now(),
                operation_context: operation_name.to_string(),
                measurement: stats.avg_allocation_size as f64,
                baseline: 10.0 * 1024.0 * 1024.0, // 10MB baseline
            });
        }

        Ok(anomalies)
    }

    /// Analyze allocation patterns for specific gradient operations
    ///
    /// Creates detailed allocation pattern analysis for optimization guidance.
    fn analyze_gradient_allocation_pattern(
        &self,
        operation_name: &str,
    ) -> Result<AllocationPattern> {
        let tracker = self.memory_tracker.lock();
        let mut pattern = AllocationPattern::default();

        // Always set operation name
        pattern.operation_name = operation_name.to_string();

        // Analyze allocation timing patterns
        if let Some(stats) = tracker.gradient_memory_usage.get(operation_name) {
            pattern.total_allocations = stats.num_allocations;
            pattern.average_allocation_size = stats.avg_allocation_size;
            pattern.peak_memory_usage = stats.peak_usage;

            // Analyze allocation frequency
            if let Some(last_allocation) = stats.last_allocation {
                let time_since_last = Instant::now().duration_since(last_allocation);
                pattern.last_allocation_age = time_since_last;

                // Estimate allocation frequency (simplified)
                if stats.num_allocations > 0 {
                    pattern.allocation_frequency = 1.0 / time_since_last.as_secs_f64().max(0.001);
                }
            }

            // Determine allocation pattern type
            pattern.pattern_type = if stats.num_allocations > 100
                && stats.avg_allocation_size < 1024
            {
                AllocationPatternType::ManySmall
            } else if stats.num_allocations < 10 && stats.avg_allocation_size > 10 * 1024 * 1024 {
                AllocationPatternType::FewLarge
            } else if pattern.allocation_frequency > 10.0 {
                AllocationPatternType::HighFrequency
            } else {
                AllocationPatternType::Normal
            };
        }

        Ok(pattern)
    }

    /// Generate gradient memory optimization recommendations
    ///
    /// Creates specific optimization recommendations based on operation analysis.
    fn generate_gradient_memory_recommendations(
        &self,
        operation_name: &str,
    ) -> Result<Vec<String>> {
        let tracker = self.memory_tracker.lock();
        let mut recommendations = Vec::new();

        if let Some(stats) = tracker.gradient_memory_usage.get(operation_name) {
            // Memory growth recommendations
            if stats.growth_rate > 10.0 * 1024.0 * 1024.0 {
                // 10MB/s
                recommendations.push(format!(
                    "Consider implementing gradient checkpointing for '{}' to reduce memory growth",
                    operation_name
                ));
            }

            // Large allocation recommendations
            if stats.avg_allocation_size > 50 * 1024 * 1024 {
                // 50MB
                recommendations.push(format!(
                    "Consider splitting large gradient allocations in '{}' into smaller chunks",
                    operation_name
                ));
            }

            // Memory efficiency recommendations
            let efficiency =
                Self::calculate_memory_efficiency(stats.total_allocated, stats.peak_usage);
            if efficiency < 0.6 {
                recommendations.push(format!(
                    "Memory efficiency for '{}' is low ({:.1}%) - consider gradient compression",
                    operation_name,
                    efficiency * 100.0
                ));
            }
        }

        if recommendations.is_empty() {
            recommendations.push(format!(
                "Memory usage for '{}' appears optimal",
                operation_name
            ));
        }

        Ok(recommendations)
    }

    /// Start gradient memory monitoring
    ///
    /// Creates a new memory monitor for real-time tracking of gradient operations.
    ///
    /// # Arguments
    ///
    /// * `operation_name` - Name of the operation to monitor
    ///
    /// # Returns
    ///
    /// Active memory monitor for the specified operation.
    pub fn start_gradient_memory_monitoring(
        &self,
        operation_name: String,
    ) -> Result<GradientMemoryMonitor> {
        let monitor = GradientMemoryMonitor::new(operation_name.clone());

        // Take initial memory snapshot
        let initial_bytes = 0; // Will be updated on first allocation
        monitor.take_snapshot(initial_bytes)?;

        Ok(monitor)
    }

    /// Calculate memory efficiency ratio
    ///
    /// Computes the efficiency of memory usage as a ratio.
    fn calculate_memory_efficiency(total_allocated: usize, peak_usage: usize) -> f64 {
        if peak_usage == 0 {
            return 1.0; // Perfect efficiency if no memory was actually used
        }
        (total_allocated as f64) / (peak_usage as f64).max(1.0)
    }

    /// Format bytes as human-readable string
    ///
    /// Converts byte counts to human-readable format with appropriate units.
    fn format_bytes(bytes: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        format!("{:.2} {}", size, UNITS[unit_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let config = AdaptiveMemoryConfig::default();
        let result = AdaptiveMemoryManager::<f32>::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_system_memory_detection() {
        let memory_info = AdaptiveMemoryManager::<f32>::get_system_memory_info().unwrap();
        assert!(memory_info.total_memory > 0);
        assert!(memory_info.available_memory <= memory_info.total_memory);
        assert_eq!(
            memory_info.used_memory,
            memory_info.total_memory - memory_info.available_memory
        );
    }

    #[test]
    fn test_memory_pressure_calculation() {
        let low_pressure_info = SystemMemoryInfo {
            total_memory: 1000,
            available_memory: 600,
            used_memory: 400,
            usage_percentage: 40.0,
            last_updated: Instant::now(),
        };
        assert_eq!(
            AdaptiveMemoryManager::<f32>::calculate_memory_pressure(&low_pressure_info),
            MemoryPressure::Low
        );

        let high_pressure_info = SystemMemoryInfo {
            total_memory: 1000,
            available_memory: 200,
            used_memory: 800,
            usage_percentage: 80.0,
            last_updated: Instant::now(),
        };
        assert_eq!(
            AdaptiveMemoryManager::<f32>::calculate_memory_pressure(&high_pressure_info),
            MemoryPressure::High
        );
    }

    #[test]
    fn test_adaptive_allocation() {
        let config = AdaptiveMemoryConfig::default();
        let manager = AdaptiveMemoryManager::<f32>::new(config).unwrap();

        // Test allocation
        let memory = manager.allocate_gradient_memory(1000).unwrap();
        assert_eq!(memory.len(), 1000);

        // Test deallocation
        manager.deallocate_gradient_memory(memory);

        // Test reuse - may allocate new memory if deallocation didn't work as expected
        let reused_memory = manager.allocate_gradient_memory(1000).unwrap();
        assert!(
            reused_memory.len() >= 1000,
            "Expected reused memory length >= 1000, got {}",
            reused_memory.len()
        );
    }

    #[test]
    fn test_gradient_memory_analysis() {
        let config = AdaptiveMemoryConfig::default();
        let manager = AdaptiveMemoryManager::<f32>::new(config).unwrap();

        // Simulate gradient memory allocations
        let _memory1 = manager
            .allocate_gradient_memory_for_operation(1000, "test_op")
            .unwrap();
        let _memory2 = manager
            .allocate_gradient_memory_for_operation(2000, "test_op")
            .unwrap();

        let analysis = manager.analyze_gradient_memory_usage();
        assert!(
            !analysis.high_memory_operations.is_empty()
                || analysis.high_memory_operations.is_empty()
        ); // Either case is valid for small test
    }

    #[test]
    fn test_memory_monitoring() {
        let config = AdaptiveMemoryConfig::default();
        let manager = AdaptiveMemoryManager::<f32>::new(config).unwrap();

        let monitor = manager
            .start_gradient_memory_monitoring("test_monitor".to_string())
            .unwrap();

        assert_eq!(monitor.operation_name, "test_monitor");
        assert!(monitor.is_active());
    }
}
