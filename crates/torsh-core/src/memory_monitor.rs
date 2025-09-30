//! Advanced system memory monitoring with platform-specific APIs

use crate::error::{Result, TorshError};
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

/// Memory pressure levels for adaptive allocation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    /// Normal memory conditions - no restrictions
    Normal,
    /// Moderate pressure - prefer smaller allocations
    Moderate,
    /// High pressure - minimize new allocations
    High,
    /// Critical pressure - emergency cleanup needed
    Critical,
}

/// Detailed system memory statistics
#[derive(Debug, Clone)]
pub struct SystemMemoryStats {
    /// Total physical memory in bytes
    pub total_physical: u64,
    /// Available physical memory in bytes
    pub available_physical: u64,
    /// Used physical memory in bytes
    pub used_physical: u64,
    /// Total virtual memory in bytes (if applicable)
    pub total_virtual: Option<u64>,
    /// Available virtual memory in bytes (if applicable)
    pub available_virtual: Option<u64>,
    /// Memory cached by the system in bytes
    pub cached: Option<u64>,
    /// Buffer memory in bytes (Linux)
    pub buffers: Option<u64>,
    /// Swap total in bytes
    pub swap_total: Option<u64>,
    /// Swap free in bytes
    pub swap_free: Option<u64>,
    /// Current memory pressure level
    pub pressure: MemoryPressure,
    /// Timestamp when stats were collected
    pub timestamp: SystemTime,
}

/// Memory monitoring configuration
#[derive(Debug, Clone)]
pub struct MemoryMonitorConfig {
    /// How often to update memory statistics
    pub update_interval: Duration,
    /// Number of historical samples to keep
    pub history_size: usize,
    /// Memory pressure thresholds (as percentage of total memory)
    pub pressure_thresholds: MemoryPressureThresholds,
    /// Enable detailed platform-specific monitoring
    pub enable_detailed_monitoring: bool,
}

/// Memory pressure detection thresholds
#[derive(Debug, Clone)]
pub struct MemoryPressureThresholds {
    /// Moderate pressure threshold (% of total memory used)
    pub moderate_threshold: f64,
    /// High pressure threshold (% of total memory used)
    pub high_threshold: f64,
    /// Critical pressure threshold (% of total memory used)
    pub critical_threshold: f64,
}

impl Default for MemoryPressureThresholds {
    fn default() -> Self {
        Self {
            moderate_threshold: 70.0, // 70% memory usage
            high_threshold: 85.0,     // 85% memory usage
            critical_threshold: 95.0, // 95% memory usage
        }
    }
}

impl Default for MemoryMonitorConfig {
    fn default() -> Self {
        Self {
            update_interval: Duration::from_secs(1),
            history_size: 300, // 5 minutes at 1 second intervals
            pressure_thresholds: MemoryPressureThresholds::default(),
            enable_detailed_monitoring: true,
        }
    }
}

/// Advanced system memory monitor with platform-specific optimizations
pub struct SystemMemoryMonitor {
    /// Current memory statistics
    current_stats: RwLock<SystemMemoryStats>,
    /// Historical memory statistics
    history: RwLock<VecDeque<SystemMemoryStats>>,
    /// Configuration
    config: MemoryMonitorConfig,
    /// Last update timestamp
    last_update: AtomicU64,
    /// Number of pressure events detected
    pressure_events: AtomicU64,
}

impl SystemMemoryMonitor {
    /// Create a new memory monitor with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(MemoryMonitorConfig::default())
    }

    /// Create a new memory monitor with custom configuration
    pub fn with_config(config: MemoryMonitorConfig) -> Result<Self> {
        let initial_stats = Self::collect_memory_stats_impl(&config)?;

        Ok(Self {
            current_stats: RwLock::new(initial_stats.clone()),
            history: RwLock::new({
                let mut history = VecDeque::with_capacity(config.history_size);
                history.push_back(initial_stats);
                history
            }),
            config,
            last_update: AtomicU64::new(0),
            pressure_events: AtomicU64::new(0),
        })
    }

    /// Get current memory statistics
    pub fn current_stats(&self) -> SystemMemoryStats {
        self.current_stats.read().clone()
    }

    /// Update memory statistics if enough time has passed
    pub fn update_if_needed(&self) -> Result<bool> {
        let now = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let last_update = self.last_update.load(Ordering::Relaxed);

        if now.saturating_sub(last_update) >= self.config.update_interval.as_secs() {
            self.force_update()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Force an immediate update of memory statistics
    pub fn force_update(&self) -> Result<()> {
        let new_stats = Self::collect_memory_stats_impl(&self.config)?;

        // Update current stats
        {
            let mut current = self.current_stats.write();
            let old_pressure = current.pressure;
            *current = new_stats.clone();

            // Track pressure level changes
            if new_stats.pressure != old_pressure && new_stats.pressure != MemoryPressure::Normal {
                self.pressure_events.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Update history
        {
            let mut history = self.history.write();
            history.push_back(new_stats);
            if history.len() > self.config.history_size {
                history.pop_front();
            }
        }

        self.last_update.store(
            SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            Ordering::Relaxed,
        );

        Ok(())
    }

    /// Get memory statistics history
    pub fn get_history(&self) -> Vec<SystemMemoryStats> {
        self.history.read().iter().cloned().collect()
    }

    /// Get memory utilization trend (positive = increasing, negative = decreasing)
    pub fn get_memory_trend(&self) -> Option<f64> {
        let history = self.history.read();
        if history.len() < 2 {
            return None;
        }

        let recent = &history[history.len() - 1];
        let older = &history[history.len() - 2];

        let recent_util = recent.used_physical as f64 / recent.total_physical as f64;
        let older_util = older.used_physical as f64 / older.total_physical as f64;

        Some(recent_util - older_util)
    }

    /// Get average memory utilization over the history window
    pub fn get_average_utilization(&self) -> f64 {
        let history = self.history.read();
        if history.is_empty() {
            return 0.0;
        }

        let sum: f64 = history
            .iter()
            .map(|stats| stats.used_physical as f64 / stats.total_physical as f64)
            .sum();

        sum / history.len() as f64
    }

    /// Check if there's enough memory for a requested allocation
    pub fn can_allocate(&self, size: usize) -> bool {
        let stats = self.current_stats.read();

        match stats.pressure {
            MemoryPressure::Normal => stats.available_physical >= size as u64,
            MemoryPressure::Moderate => {
                // Be more conservative - require 2x the requested size available
                stats.available_physical >= (size as u64).saturating_mul(2)
            }
            MemoryPressure::High => {
                // Very conservative - require 4x the requested size available
                stats.available_physical >= (size as u64).saturating_mul(4)
            }
            MemoryPressure::Critical => false, // Don't allow new allocations
        }
    }

    /// Get recommended allocation strategy based on current memory pressure
    pub fn get_allocation_strategy(&self) -> AllocationStrategy {
        match self.current_stats().pressure {
            MemoryPressure::Normal => AllocationStrategy::Normal,
            MemoryPressure::Moderate => AllocationStrategy::Conservative,
            MemoryPressure::High => AllocationStrategy::Minimal,
            MemoryPressure::Critical => AllocationStrategy::Emergency,
        }
    }

    /// Get the number of memory pressure events detected
    pub fn pressure_event_count(&self) -> u64 {
        self.pressure_events.load(Ordering::Relaxed)
    }

    /// Platform-specific memory statistics collection
    fn collect_memory_stats_impl(config: &MemoryMonitorConfig) -> Result<SystemMemoryStats> {
        #[cfg(target_os = "linux")]
        {
            Self::collect_linux_memory_stats(config)
        }
        #[cfg(target_os = "macos")]
        {
            Self::collect_macos_memory_stats(config)
        }
        #[cfg(target_os = "windows")]
        {
            Self::collect_windows_memory_stats(config)
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            Self::collect_fallback_memory_stats(config)
        }
    }

    /// Linux-specific memory statistics using /proc/meminfo
    #[cfg(target_os = "linux")]
    fn collect_linux_memory_stats(config: &MemoryMonitorConfig) -> Result<SystemMemoryStats> {
        use std::fs;

        let meminfo = fs::read_to_string("/proc/meminfo")
            .map_err(|e| TorshError::IoError(format!("Failed to read /proc/meminfo: {e}")))?;

        let mut total_physical = 0u64;
        let mut available_physical = 0u64;
        let mut cached = 0u64;
        let mut buffers = 0u64;
        let mut swap_total = 0u64;
        let mut swap_free = 0u64;

        for line in meminfo.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }

            let value = parts[1].parse::<u64>().map_err(|e| {
                TorshError::ConversionError(format!("Failed to parse memory value: {e}"))
            })?;

            match parts[0] {
                "MemTotal:" => total_physical = value * 1024, // Convert KB to bytes
                "MemAvailable:" => available_physical = value * 1024,
                "Cached:" => cached = value * 1024,
                "Buffers:" => buffers = value * 1024,
                "SwapTotal:" => swap_total = value * 1024,
                "SwapFree:" => swap_free = value * 1024,
                _ => {}
            }
        }

        let used_physical = total_physical - available_physical;
        let pressure =
            Self::calculate_pressure(used_physical, total_physical, &config.pressure_thresholds);

        Ok(SystemMemoryStats {
            total_physical,
            available_physical,
            used_physical,
            total_virtual: None,
            available_virtual: None,
            cached: Some(cached),
            buffers: Some(buffers),
            swap_total: Some(swap_total),
            swap_free: Some(swap_free),
            pressure,
            timestamp: SystemTime::now(),
        })
    }

    /// macOS-specific memory statistics using vm_statistics64
    #[cfg(target_os = "macos")]
    fn collect_macos_memory_stats(config: &MemoryMonitorConfig) -> Result<SystemMemoryStats> {
        use std::process::Command;

        // Get total physical memory
        let total_output = Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .map_err(|e| TorshError::IoError(format!("Failed to get total memory: {e}")))?;

        let total_physical = String::from_utf8_lossy(&total_output.stdout)
            .trim()
            .parse::<u64>()
            .map_err(|e| {
                TorshError::ConversionError(format!("Failed to parse total memory: {e}"))
            })?;

        // Get memory statistics using vm_stat
        let vm_output = Command::new("vm_stat")
            .output()
            .map_err(|e| TorshError::IoError(format!("Failed to get vm_stat: {e}")))?;

        let vm_str = String::from_utf8_lossy(&vm_output.stdout);

        let mut page_size = 4096u64;
        let mut free_pages = 0u64;
        let mut inactive_pages = 0u64;
        let mut speculative_pages = 0u64;

        for line in vm_str.lines() {
            if line.contains("page size of") {
                if let Some(size_str) = line
                    .split_whitespace()
                    .find(|s| s.chars().all(|c| c.is_ascii_digit()))
                {
                    page_size = size_str.parse().map_err(|e| {
                        TorshError::ConversionError(format!("Failed to parse page size: {e}"))
                    })?;
                }
            } else if line.starts_with("Pages free:") {
                free_pages = Self::parse_pages_line(line)?;
            } else if line.starts_with("Pages inactive:") {
                inactive_pages = Self::parse_pages_line(line)?;
            } else if line.starts_with("Pages speculative:") {
                speculative_pages = Self::parse_pages_line(line)?;
            }
        }

        // Available memory includes free, inactive, and speculative pages
        let available_physical = (free_pages + inactive_pages + speculative_pages) * page_size;
        let used_physical = total_physical - available_physical;
        let pressure =
            Self::calculate_pressure(used_physical, total_physical, &config.pressure_thresholds);

        Ok(SystemMemoryStats {
            total_physical,
            available_physical,
            used_physical,
            total_virtual: None,
            available_virtual: None,
            cached: None,
            buffers: None,
            swap_total: None,
            swap_free: None,
            pressure,
            timestamp: SystemTime::now(),
        })
    }

    #[cfg(target_os = "macos")]
    fn parse_pages_line(line: &str) -> Result<u64> {
        line.split_whitespace()
            .nth(2)
            .ok_or_else(|| TorshError::ConversionError("Invalid pages line format".to_string()))?
            .trim_end_matches('.')
            .parse::<u64>()
            .map_err(|e| TorshError::ConversionError(format!("Failed to parse pages: {e}")))
    }

    /// Windows-specific memory statistics using GlobalMemoryStatusEx
    #[cfg(target_os = "windows")]
    fn collect_windows_memory_stats(config: &MemoryMonitorConfig) -> Result<SystemMemoryStats> {
        use std::process::Command;

        // Use PowerShell for more reliable memory information
        let output = Command::new("powershell")
            .args(&[
                "-Command",
                r#"
                $memory = Get-WmiObject -Class Win32_OperatingSystem;
                $cs = Get-WmiObject -Class Win32_ComputerSystem;
                Write-Host "TotalPhysical:$($cs.TotalPhysicalMemory)";
                Write-Host "FreePhysical:$($memory.FreePhysicalMemory * 1024)";
                Write-Host "TotalVirtual:$($memory.TotalVirtualMemorySize * 1024)";
                Write-Host "FreeVirtual:$($memory.FreeVirtualMemory * 1024)";
            "#,
            ])
            .output()
            .map_err(|e| TorshError::IoError(format!("Failed to get memory info: {e}")))?;

        if !output.status.success() {
            return Err(TorshError::IoError("PowerShell command failed".to_string()));
        }

        let output_str = String::from_utf8_lossy(&output.stdout);

        let mut total_physical = 0u64;
        let mut available_physical = 0u64;
        let mut total_virtual = 0u64;
        let mut available_virtual = 0u64;

        for line in output_str.lines() {
            if let Some((key, value)) = line.split_once(':') {
                let parsed_value = value.parse::<u64>().map_err(|e| {
                    TorshError::ConversionError(format!("Failed to parse {key}: {e}"))
                })?;

                match key {
                    "TotalPhysical" => total_physical = parsed_value,
                    "FreePhysical" => available_physical = parsed_value,
                    "TotalVirtual" => total_virtual = parsed_value,
                    "FreeVirtual" => available_virtual = parsed_value,
                    _ => {}
                }
            }
        }

        let used_physical = total_physical - available_physical;
        let pressure =
            Self::calculate_pressure(used_physical, total_physical, &config.pressure_thresholds);

        Ok(SystemMemoryStats {
            total_physical,
            available_physical,
            used_physical,
            total_virtual: Some(total_virtual),
            available_virtual: Some(available_virtual),
            cached: None,
            buffers: None,
            swap_total: None,
            swap_free: None,
            pressure,
            timestamp: SystemTime::now(),
        })
    }

    /// Fallback memory statistics for unsupported platforms
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    fn collect_fallback_memory_stats(config: &MemoryMonitorConfig) -> Result<SystemMemoryStats> {
        // Estimate based on common system configurations
        let total_physical = 16_000_000_000u64; // 16GB
        let available_physical = 8_000_000_000u64; // 8GB
        let used_physical = total_physical - available_physical;
        let pressure =
            Self::calculate_pressure(used_physical, total_physical, &config.pressure_thresholds);

        Ok(SystemMemoryStats {
            total_physical,
            available_physical,
            used_physical,
            total_virtual: None,
            available_virtual: None,
            cached: None,
            buffers: None,
            swap_total: None,
            swap_free: None,
            pressure,
            timestamp: SystemTime::now(),
        })
    }

    /// Calculate memory pressure based on usage and thresholds
    fn calculate_pressure(
        used: u64,
        total: u64,
        thresholds: &MemoryPressureThresholds,
    ) -> MemoryPressure {
        if total == 0 {
            return MemoryPressure::Critical;
        }

        let usage_percent = (used as f64 / total as f64) * 100.0;

        if usage_percent >= thresholds.critical_threshold {
            MemoryPressure::Critical
        } else if usage_percent >= thresholds.high_threshold {
            MemoryPressure::High
        } else if usage_percent >= thresholds.moderate_threshold {
            MemoryPressure::Moderate
        } else {
            MemoryPressure::Normal
        }
    }
}

impl Default for SystemMemoryMonitor {
    fn default() -> Self {
        Self::new().expect("Failed to create default memory monitor")
    }
}

/// Allocation strategies based on memory pressure
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Normal allocation - no restrictions
    Normal,
    /// Conservative allocation - prefer smaller chunks
    Conservative,
    /// Minimal allocation - only essential allocations
    Minimal,
    /// Emergency mode - defer all non-critical allocations
    Emergency,
}

impl AllocationStrategy {
    /// Get the maximum recommended allocation size for this strategy
    pub fn max_allocation_size(&self, available_memory: u64) -> u64 {
        match self {
            AllocationStrategy::Normal => available_memory / 2,
            AllocationStrategy::Conservative => available_memory / 4,
            AllocationStrategy::Minimal => available_memory / 8,
            AllocationStrategy::Emergency => 0,
        }
    }

    /// Check if an allocation of the given size should be allowed
    pub fn should_allow_allocation(&self, size: u64, available_memory: u64) -> bool {
        size <= self.max_allocation_size(available_memory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_monitor_creation() {
        let monitor = SystemMemoryMonitor::new();
        assert!(monitor.is_ok());

        let monitor = monitor.unwrap();
        let stats = monitor.current_stats();
        assert!(stats.total_physical > 0);
    }

    #[test]
    fn test_pressure_calculation() {
        let thresholds = MemoryPressureThresholds::default();

        assert_eq!(
            SystemMemoryMonitor::calculate_pressure(1000, 2000, &thresholds),
            MemoryPressure::Normal
        );

        assert_eq!(
            SystemMemoryMonitor::calculate_pressure(1500, 2000, &thresholds),
            MemoryPressure::Moderate
        );

        assert_eq!(
            SystemMemoryMonitor::calculate_pressure(1800, 2000, &thresholds),
            MemoryPressure::High
        );

        assert_eq!(
            SystemMemoryMonitor::calculate_pressure(1950, 2000, &thresholds),
            MemoryPressure::Critical
        );
    }

    #[test]
    fn test_allocation_strategy() {
        let available = 1000u64;

        assert_eq!(
            AllocationStrategy::Normal.max_allocation_size(available),
            500
        );
        assert_eq!(
            AllocationStrategy::Conservative.max_allocation_size(available),
            250
        );
        assert_eq!(
            AllocationStrategy::Minimal.max_allocation_size(available),
            125
        );
        assert_eq!(
            AllocationStrategy::Emergency.max_allocation_size(available),
            0
        );

        assert!(AllocationStrategy::Normal.should_allow_allocation(400, available));
        assert!(!AllocationStrategy::Normal.should_allow_allocation(600, available));
        assert!(!AllocationStrategy::Emergency.should_allow_allocation(1, available));
    }

    #[test]
    fn test_memory_monitor_updates() {
        let monitor = SystemMemoryMonitor::new().unwrap();

        // Test that we can update multiple times
        assert!(monitor.force_update().is_ok());
        assert!(monitor.force_update().is_ok());

        let history = monitor.get_history();
        assert!(history.len() >= 2);
    }

    #[test]
    fn test_can_allocate() {
        let config = MemoryMonitorConfig {
            pressure_thresholds: MemoryPressureThresholds {
                moderate_threshold: 50.0,
                high_threshold: 75.0,
                critical_threshold: 90.0,
            },
            ..Default::default()
        };

        let monitor = SystemMemoryMonitor::with_config(config).unwrap();

        // Test allocation decisions based on current memory pressure
        let stats = monitor.current_stats();
        let small_allocation = (stats.total_physical / 1000) as usize;

        // Small allocations should generally be allowed unless in critical state
        if stats.pressure != MemoryPressure::Critical {
            assert!(monitor.can_allocate(small_allocation));
        }
    }
}
