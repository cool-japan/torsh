//! Memory information and allocation strategy management
//!
//! This module provides types and utilities for tracking memory usage,
//! allocation strategies, and device memory information.

/// Memory information for a device
///
/// Provides comprehensive information about memory usage, capabilities,
/// and performance characteristics of a compute device.
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Total memory available on the device in bytes
    pub total_memory: usize,
    /// Free memory available on the device in bytes
    pub free_memory: usize,
    /// Used memory on the device in bytes
    pub used_memory: usize,
    /// Maximum allocation size supported in bytes
    pub max_allocation_size: usize,
    /// Memory bandwidth in bytes per second (if available)
    pub bandwidth: Option<u64>,
    /// Whether the memory is unified with host memory
    pub is_unified: bool,
    /// Supported memory alignments
    pub supported_alignments: Vec<usize>,
}

impl MemoryInfo {
    /// Create new memory info
    pub fn new(total_memory: usize, free_memory: usize, max_allocation_size: usize) -> Self {
        Self {
            total_memory,
            free_memory,
            used_memory: total_memory.saturating_sub(free_memory),
            max_allocation_size,
            bandwidth: None,
            is_unified: false,
            supported_alignments: vec![1, 2, 4, 8, 16, 32, 64],
        }
    }

    /// Calculate memory utilization as a percentage
    pub fn utilization_percent(&self) -> f64 {
        if self.total_memory == 0 {
            0.0
        } else {
            (self.used_memory as f64 / self.total_memory as f64) * 100.0
        }
    }

    /// Check if there's enough free memory for an allocation
    pub fn can_allocate(&self, size_bytes: usize) -> bool {
        size_bytes <= self.free_memory && size_bytes <= self.max_allocation_size
    }

    /// Get memory pressure level (0.0 = no pressure, 1.0 = maximum pressure)
    pub fn memory_pressure(&self) -> f64 {
        self.utilization_percent() / 100.0
    }

    /// Check if the device supports a specific alignment
    pub fn supports_alignment(&self, alignment: usize) -> bool {
        self.supported_alignments.contains(&alignment) || alignment.is_power_of_two()
    }

    /// Get the closest supported alignment that is >= the requested alignment
    pub fn closest_supported_alignment(&self, requested: usize) -> Option<usize> {
        self.supported_alignments
            .iter()
            .filter(|&&align| align >= requested)
            .min()
            .copied()
    }

    /// Estimate allocation overhead for the given size
    pub fn allocation_overhead(&self, size_bytes: usize) -> usize {
        // Estimate overhead based on device characteristics
        if self.is_unified {
            // Unified memory typically has lower overhead
            size_bytes / 100 // 1% overhead
        } else {
            // Discrete memory may have higher overhead
            (size_bytes / 50).max(64) // 2% overhead, minimum 64 bytes
        }
    }

    /// Get memory fragmentation estimate (0.0 = no fragmentation, 1.0 = highly fragmented)
    pub fn fragmentation_estimate(&self) -> f64 {
        // Simplified estimate based on largest allocatable chunk vs free memory
        if self.free_memory == 0 {
            return 1.0;
        }

        let largest_chunk = self.max_allocation_size.min(self.free_memory);
        1.0 - (largest_chunk as f64 / self.free_memory as f64)
    }

    /// Get recommended allocation strategy based on current memory state
    pub fn recommended_strategy(&self) -> AllocationStrategy {
        let pressure = self.memory_pressure();
        let fragmentation = self.fragmentation_estimate();

        match (pressure, fragmentation) {
            (p, _) if p < 0.5 => AllocationStrategy::Immediate,
            (p, f) if p < 0.8 && f < 0.3 => AllocationStrategy::Pooled,
            (p, f) if p < 0.9 && f < 0.5 => AllocationStrategy::Lazy,
            _ => AllocationStrategy::PreAllocated,
        }
    }

    /// Update memory usage after allocation/deallocation
    pub fn update_usage(&mut self, size_bytes: usize, is_allocation: bool) {
        if is_allocation {
            self.free_memory = self.free_memory.saturating_sub(size_bytes);
            self.used_memory = self.used_memory.saturating_add(size_bytes);
        } else {
            self.free_memory = self
                .free_memory
                .saturating_add(size_bytes)
                .min(self.total_memory);
            self.used_memory = self.used_memory.saturating_sub(size_bytes);
        }
    }

    /// Create memory info for system RAM
    pub fn system_ram() -> Self {
        let total = Self::get_system_memory();
        let free = total / 2; // Conservative estimate

        Self {
            total_memory: total,
            free_memory: free,
            used_memory: total - free,
            max_allocation_size: total / 4, // Conservative max single allocation
            bandwidth: Some(25_000_000_000), // ~25 GB/s typical DDR4
            is_unified: true,
            supported_alignments: vec![1, 2, 4, 8, 16, 32, 64, 128, 256],
        }
    }

    /// Get total system memory
    fn get_system_memory() -> usize {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_to_string("/proc/meminfo")
                .ok()
                .and_then(|content| {
                    content
                        .lines()
                        .find(|line| line.starts_with("MemTotal:"))
                        .and_then(|line| {
                            line.split_whitespace()
                                .nth(1)
                                .and_then(|s| s.parse::<usize>().ok())
                                .map(|kb| kb * 1024) // Convert KB to bytes
                        })
                })
                .unwrap_or(8 * 1024 * 1024 * 1024) // 8GB fallback
        }
        #[cfg(target_os = "windows")]
        {
            // Would use Windows API to get memory info
            16 * 1024 * 1024 * 1024 // 16GB fallback
        }
        #[cfg(target_os = "macos")]
        {
            // Would use macOS system calls
            16 * 1024 * 1024 * 1024 // 16GB fallback
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
        {
            8 * 1024 * 1024 * 1024 // 8GB fallback
        }
    }
}

impl std::fmt::Display for MemoryInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MemoryInfo(total={:.2}GB, free={:.2}GB, used={:.2}GB, util={:.1}%)",
            self.total_memory as f64 / 1e9,
            self.free_memory as f64 / 1e9,
            self.used_memory as f64 / 1e9,
            self.utilization_percent()
        )
    }
}

/// Memory allocation strategies
///
/// Different strategies optimize for different use cases and memory pressure scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AllocationStrategy {
    /// Allocate immediately when requested (default)
    /// Best for: Low memory pressure, simple usage patterns
    #[default]
    Immediate,
    /// Use memory pooling for small allocations
    /// Best for: Frequent small allocations, moderate memory pressure
    Pooled,
    /// Lazy allocation with copy-on-write
    /// Best for: Memory sharing, deferred computation
    Lazy,
    /// Pre-allocate large blocks and sub-allocate
    /// Best for: High memory pressure, predictable allocation patterns
    PreAllocated,
    /// Custom strategy (backend-specific)
    /// Best for: Specialized hardware or usage patterns
    Custom(u32),
}

impl AllocationStrategy {
    /// Check if this strategy supports deferred allocation
    pub fn supports_deferred(&self) -> bool {
        matches!(
            self,
            AllocationStrategy::Lazy | AllocationStrategy::PreAllocated
        )
    }

    /// Check if this strategy uses pooling
    pub fn uses_pooling(&self) -> bool {
        matches!(
            self,
            AllocationStrategy::Pooled | AllocationStrategy::PreAllocated
        )
    }

    /// Get the priority of this strategy (lower is higher priority)
    pub fn priority(&self) -> u32 {
        match self {
            AllocationStrategy::Immediate => 0,
            AllocationStrategy::Pooled => 1,
            AllocationStrategy::Lazy => 2,
            AllocationStrategy::PreAllocated => 3,
            AllocationStrategy::Custom(priority) => *priority,
        }
    }

    /// Get all available strategies
    pub fn all_strategies() -> &'static [AllocationStrategy] {
        &[
            AllocationStrategy::Immediate,
            AllocationStrategy::Pooled,
            AllocationStrategy::Lazy,
            AllocationStrategy::PreAllocated,
        ]
    }

    /// Choose the best strategy for given conditions
    pub fn choose_for_conditions(
        memory_pressure: f64,
        allocation_frequency: AllocationFrequency,
        allocation_size: AllocationSize,
    ) -> Self {
        match (memory_pressure, allocation_frequency, allocation_size) {
            // Low pressure - use immediate allocation
            (p, _, _) if p < 0.3 => AllocationStrategy::Immediate,

            // High frequency small allocations - use pooling
            (_, AllocationFrequency::High, AllocationSize::Small) => AllocationStrategy::Pooled,

            // Large allocations under pressure - use pre-allocation
            (p, _, AllocationSize::Large) if p > 0.7 => AllocationStrategy::PreAllocated,

            // Medium pressure - use lazy allocation
            (p, _, _) if p > 0.5 => AllocationStrategy::Lazy,

            // Default to pooled for medium scenarios
            _ => AllocationStrategy::Pooled,
        }
    }
}

impl std::fmt::Display for AllocationStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AllocationStrategy::Immediate => write!(f, "immediate"),
            AllocationStrategy::Pooled => write!(f, "pooled"),
            AllocationStrategy::Lazy => write!(f, "lazy"),
            AllocationStrategy::PreAllocated => write!(f, "pre_allocated"),
            AllocationStrategy::Custom(id) => write!(f, "custom({})", id),
        }
    }
}

impl std::str::FromStr for AllocationStrategy {
    type Err = crate::error::TorshError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "immediate" => Ok(AllocationStrategy::Immediate),
            "pooled" => Ok(AllocationStrategy::Pooled),
            "lazy" => Ok(AllocationStrategy::Lazy),
            "pre_allocated" | "preallocated" => Ok(AllocationStrategy::PreAllocated),
            s if s.starts_with("custom(") && s.ends_with(')') => {
                let id_str = &s[7..s.len() - 1];
                let id = id_str.parse::<u32>().map_err(|_| {
                    crate::error::TorshError::InvalidArgument(format!(
                        "Invalid custom strategy ID: {id_str}"
                    ))
                })?;
                Ok(AllocationStrategy::Custom(id))
            }
            _ => Err(crate::error::TorshError::InvalidArgument(format!(
                "Unknown allocation strategy: {s}"
            ))),
        }
    }
}

/// Allocation frequency categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationFrequency {
    /// Infrequent allocations (< 10/sec)
    Low,
    /// Moderate allocation rate (10-100/sec)
    Medium,
    /// High allocation rate (> 100/sec)
    High,
}

/// Allocation size categorization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationSize {
    /// Small allocations (< 1KB)
    Small,
    /// Medium allocations (1KB - 1MB)
    Medium,
    /// Large allocations (> 1MB)
    Large,
}

impl AllocationSize {
    /// Categorize an allocation size
    pub fn categorize(size_bytes: usize) -> Self {
        match size_bytes {
            s if s < 1024 => AllocationSize::Small,
            s if s < 1024 * 1024 => AllocationSize::Medium,
            _ => AllocationSize::Large,
        }
    }
}

/// Memory allocation configuration
#[derive(Debug, Clone)]
pub struct AllocationConfig {
    /// Primary allocation strategy
    pub strategy: AllocationStrategy,
    /// Fallback strategies in order of preference
    pub fallback_strategies: Vec<AllocationStrategy>,
    /// Maximum memory pressure before switching to fallback
    pub pressure_threshold: f64,
    /// Enable automatic strategy switching
    pub auto_switch: bool,
    /// Minimum alignment requirement
    pub min_alignment: usize,
    /// Preferred alignment for performance
    pub preferred_alignment: usize,
}

impl Default for AllocationConfig {
    fn default() -> Self {
        Self {
            strategy: AllocationStrategy::Pooled,
            fallback_strategies: vec![
                AllocationStrategy::Lazy,
                AllocationStrategy::PreAllocated,
                AllocationStrategy::Immediate,
            ],
            pressure_threshold: 0.8,
            auto_switch: true,
            min_alignment: 1,
            preferred_alignment: 64,
        }
    }
}

impl AllocationConfig {
    /// Create new allocation configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set primary strategy
    pub fn with_strategy(mut self, strategy: AllocationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Add fallback strategy
    pub fn with_fallback(mut self, strategy: AllocationStrategy) -> Self {
        self.fallback_strategies.push(strategy);
        self
    }

    /// Set pressure threshold
    pub fn with_pressure_threshold(mut self, threshold: f64) -> Self {
        self.pressure_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Enable/disable automatic strategy switching
    pub fn with_auto_switch(mut self, auto_switch: bool) -> Self {
        self.auto_switch = auto_switch;
        self
    }

    /// Set alignment requirements
    pub fn with_alignment(mut self, min: usize, preferred: usize) -> Self {
        self.min_alignment = min;
        self.preferred_alignment = preferred;
        self
    }

    /// Choose strategy based on current memory state
    pub fn choose_strategy(&self, memory_info: &MemoryInfo) -> AllocationStrategy {
        if !self.auto_switch {
            return self.strategy;
        }

        let pressure = memory_info.memory_pressure();

        if pressure <= self.pressure_threshold {
            self.strategy
        } else {
            // Try fallback strategies
            self.fallback_strategies
                .first()
                .copied()
                .unwrap_or(self.strategy)
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.pressure_threshold < 0.0 || self.pressure_threshold > 1.0 {
            return Err("Pressure threshold must be between 0.0 and 1.0".to_string());
        }

        if !self.min_alignment.is_power_of_two() {
            return Err("Minimum alignment must be a power of 2".to_string());
        }

        if !self.preferred_alignment.is_power_of_two() {
            return Err("Preferred alignment must be a power of 2".to_string());
        }

        if self.preferred_alignment < self.min_alignment {
            return Err("Preferred alignment must be >= minimum alignment".to_string());
        }

        Ok(())
    }
}

/// Memory allocation statistics
#[derive(Debug, Clone, Default)]
pub struct AllocationStats {
    /// Total number of allocations
    pub total_allocations: u64,
    /// Total number of deallocations
    pub total_deallocations: u64,
    /// Total bytes allocated
    pub total_bytes_allocated: u64,
    /// Total bytes deallocated
    pub total_bytes_deallocated: u64,
    /// Current number of active allocations
    pub active_allocations: u64,
    /// Current bytes in use
    pub active_bytes: u64,
    /// Peak number of concurrent allocations
    pub peak_allocations: u64,
    /// Peak memory usage in bytes
    pub peak_memory_usage: u64,
    /// Number of failed allocations
    pub failed_allocations: u64,
    /// Statistics per strategy
    pub strategy_stats: std::collections::HashMap<AllocationStrategy, StrategyStats>,
}

impl AllocationStats {
    /// Record a successful allocation
    pub fn record_allocation(&mut self, size_bytes: usize, strategy: AllocationStrategy) {
        self.total_allocations += 1;
        self.total_bytes_allocated += size_bytes as u64;
        self.active_allocations += 1;
        self.active_bytes += size_bytes as u64;
        self.peak_allocations = self.peak_allocations.max(self.active_allocations);
        self.peak_memory_usage = self.peak_memory_usage.max(self.active_bytes);

        let entry = self.strategy_stats.entry(strategy).or_default();
        entry.allocations += 1;
        entry.bytes_allocated += size_bytes as u64;
    }

    /// Record a deallocation
    pub fn record_deallocation(&mut self, size_bytes: usize, strategy: AllocationStrategy) {
        self.total_deallocations += 1;
        self.total_bytes_deallocated += size_bytes as u64;
        self.active_allocations = self.active_allocations.saturating_sub(1);
        self.active_bytes = self.active_bytes.saturating_sub(size_bytes as u64);

        let entry = self.strategy_stats.entry(strategy).or_default();
        entry.deallocations += 1;
        entry.bytes_deallocated += size_bytes as u64;
    }

    /// Record a failed allocation
    pub fn record_failure(&mut self, strategy: AllocationStrategy) {
        self.failed_allocations += 1;
        let entry = self.strategy_stats.entry(strategy).or_default();
        entry.failures += 1;
    }

    /// Get allocation success rate
    pub fn success_rate(&self) -> f64 {
        let total_attempts = self.total_allocations + self.failed_allocations;
        if total_attempts == 0 {
            1.0
        } else {
            self.total_allocations as f64 / total_attempts as f64
        }
    }

    /// Get average allocation size
    pub fn average_allocation_size(&self) -> f64 {
        if self.total_allocations == 0 {
            0.0
        } else {
            self.total_bytes_allocated as f64 / self.total_allocations as f64
        }
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Statistics for a specific allocation strategy
#[derive(Debug, Clone, Default)]
pub struct StrategyStats {
    pub allocations: u64,
    pub deallocations: u64,
    pub bytes_allocated: u64,
    pub bytes_deallocated: u64,
    pub failures: u64,
}

impl std::fmt::Display for AllocationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "AllocationStats(total={}, active={}, peak={}MB, success={:.1}%)",
            self.total_allocations,
            self.active_allocations,
            self.peak_memory_usage / 1024 / 1024,
            self.success_rate() * 100.0
        )
    }
}
