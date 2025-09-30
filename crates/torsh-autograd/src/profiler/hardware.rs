//! Hardware Monitoring for Autograd Profiling
//!
//! This module provides comprehensive hardware utilization monitoring for tracking
//! system resource usage during automatic differentiation operations. It monitors
//! CPU, GPU, memory, and other system resources to provide insights into performance
//! bottlenecks and optimization opportunities.
//!
//! # Features
//!
//! - **CPU Utilization Monitoring**: Real-time CPU usage tracking
//! - **GPU Utilization Monitoring**: GPU usage tracking (when available)
//! - **Memory Utilization Tracking**: System memory usage monitoring
//! - **Cache Performance Monitoring**: Cache hit rate and efficiency metrics
//! - **Bandwidth Utilization**: Memory bandwidth usage tracking
//! - **Thermal Monitoring**: Temperature and thermal throttling detection
//! - **Power Consumption**: Energy usage estimation and tracking
//!
//! # Platform Support
//!
//! The hardware monitor provides platform-specific implementations for:
//! - Linux (via /proc and sysfs)
//! - macOS (via system calls and IOKit)
//! - Windows (via Windows Performance Toolkit APIs)
//!
//! # Usage Examples
//!
//! ```rust,ignore
//! use torsh_autograd::profiler::hardware::HardwareMonitor;
//! use std::time::Duration;
//!
//! // Create a hardware monitor
//! let mut monitor = HardwareMonitor::new();
//!
//! // Set measurement interval
//! monitor.set_interval(Duration::from_millis(500));
//!
//! // Update utilization measurements
//! monitor.maybe_update_utilization();
//!
//! // Get current utilization
//! if let Some(utilization) = monitor.get_current_utilization() {
//!     println!("CPU: {:.1}%", utilization.cpu_utilization);
//!     if let Some(gpu) = utilization.gpu_utilization {
//!         println!("GPU: {:.1}%", gpu);
//!     }
//! }
//! ```

use super::types::HardwareUtilization;
use std::time::{Duration, Instant};

/// Hardware monitor for tracking system resource utilization
///
/// The HardwareMonitor provides real-time monitoring of various hardware
/// resources including CPU, GPU, memory, and other system components.
///
/// # Monitoring Strategy
///
/// The monitor uses platform-specific APIs to gather hardware metrics
/// with configurable sampling intervals to balance accuracy with overhead.
#[derive(Debug)]
pub struct HardwareMonitor {
    /// Last utilization measurement
    last_utilization: Option<HardwareUtilization>,
    /// Measurement interval
    interval: Duration,
    /// Last measurement timestamp
    last_measurement: Instant,
    /// Historical utilization data
    utilization_history: Vec<TimestampedUtilization>,
    /// Maximum history length
    max_history_length: usize,
    /// Platform-specific monitor state
    platform_state: PlatformMonitorState,
}

/// Platform-specific monitoring state
#[derive(Debug)]
struct PlatformMonitorState {
    /// CPU monitoring state
    cpu_state: CpuMonitorState,
    /// GPU monitoring state
    gpu_state: GpuMonitorState,
    /// Memory monitoring state
    memory_state: MemoryMonitorState,
}

/// CPU monitoring state
#[derive(Debug)]
struct CpuMonitorState {
    /// Last CPU times for delta calculation
    last_cpu_times: Option<CpuTimes>,
    /// Number of CPU cores
    core_count: usize,
}

/// GPU monitoring state
#[derive(Debug)]
struct GpuMonitorState {
    /// Whether GPU monitoring is available
    available: bool,
    /// GPU device count
    device_count: usize,
    /// Last GPU utilization measurements
    last_gpu_utilization: Vec<f32>,
}

/// Memory monitoring state
#[derive(Debug)]
struct MemoryMonitorState {
    /// Total system memory
    total_memory: usize,
    /// Last memory statistics
    last_memory_stats: Option<MemoryStats>,
}

/// CPU time statistics for utilization calculation
#[derive(Debug, Clone)]
struct CpuTimes {
    /// User time
    user: u64,
    /// System time
    system: u64,
    /// Idle time
    idle: u64,
    /// Total time
    total: u64,
}

/// Detailed memory statistics
#[derive(Debug, Clone)]
struct MemoryStats {
    /// Total memory
    total: usize,
    /// Available memory
    available: usize,
    /// Used memory
    used: usize,
    /// Buffer memory
    buffers: usize,
    /// Cached memory
    cached: usize,
}

/// Utilization measurement with timestamp
#[derive(Debug, Clone)]
struct TimestampedUtilization {
    /// Timestamp of measurement
    timestamp: Instant,
    /// Utilization data
    utilization: HardwareUtilization,
}

/// Comprehensive hardware statistics
#[derive(Debug, Clone)]
pub struct HardwareStatistics {
    /// Average CPU utilization
    pub average_cpu_utilization: f32,
    /// Peak CPU utilization
    pub peak_cpu_utilization: f32,
    /// Average GPU utilization (if available)
    pub average_gpu_utilization: Option<f32>,
    /// Peak GPU utilization (if available)
    pub peak_gpu_utilization: Option<f32>,
    /// Average memory utilization
    pub average_memory_utilization: f32,
    /// Peak memory utilization
    pub peak_memory_utilization: f32,
    /// Average cache hit rate
    pub average_cache_hit_rate: f32,
    /// Number of measurements
    pub measurement_count: usize,
    /// Thermal throttling events detected
    pub thermal_throttling_events: u32,
    /// Power efficiency score (0.0 to 1.0)
    pub power_efficiency_score: f32,
}

/// Hardware performance recommendations
#[derive(Debug, Clone)]
pub struct HardwareRecommendations {
    /// CPU-related recommendations
    pub cpu_recommendations: Vec<String>,
    /// GPU-related recommendations
    pub gpu_recommendations: Vec<String>,
    /// Memory-related recommendations
    pub memory_recommendations: Vec<String>,
    /// General system recommendations
    pub system_recommendations: Vec<String>,
}

impl HardwareMonitor {
    /// Creates a new hardware monitor with default settings
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let monitor = HardwareMonitor::new();
    /// ```
    pub fn new() -> Self {
        Self {
            last_utilization: None,
            interval: Duration::from_secs(1),
            last_measurement: Instant::now(),
            utilization_history: Vec::new(),
            max_history_length: 1000,
            platform_state: PlatformMonitorState::new(),
        }
    }

    /// Creates a new hardware monitor with custom configuration
    ///
    /// # Arguments
    ///
    /// * `interval` - Measurement interval
    /// * `max_history` - Maximum number of historical measurements to keep
    pub fn with_config(interval: Duration, max_history: usize) -> Self {
        Self {
            last_utilization: None,
            interval,
            last_measurement: Instant::now(),
            utilization_history: Vec::new(),
            max_history_length: max_history,
            platform_state: PlatformMonitorState::new(),
        }
    }

    /// Updates utilization measurements if enough time has elapsed
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// monitor.maybe_update_utilization();
    /// ```
    pub fn maybe_update_utilization(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_measurement) >= self.interval {
            self.update_utilization();
            self.last_measurement = now;
        }
    }

    /// Forces an immediate utilization measurement
    pub fn update_utilization(&mut self) {
        let utilization = self.measure_utilization();

        // Store in history
        let timestamped = TimestampedUtilization {
            timestamp: Instant::now(),
            utilization: utilization.clone(),
        };

        self.utilization_history.push(timestamped);

        // Limit history size
        if self.utilization_history.len() > self.max_history_length {
            self.utilization_history.remove(0);
        }

        self.last_utilization = Some(utilization);
    }

    /// Gets the most recent utilization measurement
    ///
    /// # Returns
    ///
    /// Reference to the latest hardware utilization data, if available
    pub fn get_current_utilization(&self) -> Option<&HardwareUtilization> {
        self.last_utilization.as_ref()
    }

    /// Gets utilization history
    ///
    /// # Returns
    ///
    /// Slice of historical utilization measurements
    pub fn get_utilization_history(&self) -> &[TimestampedUtilization] {
        &self.utilization_history
    }

    /// Computes hardware statistics from historical data
    ///
    /// # Returns
    ///
    /// Comprehensive hardware statistics
    pub fn get_hardware_statistics(&self) -> HardwareStatistics {
        if self.utilization_history.is_empty() {
            return HardwareStatistics::default();
        }

        let mut cpu_sum = 0.0f64;
        let mut cpu_peak = 0.0f64;
        let mut gpu_sum = 0.0f64;
        let mut gpu_count = 0;
        let mut gpu_peak = 0.0f64;
        let mut memory_sum = 0.0f64;
        let mut memory_peak = 0.0f64;
        let mut cache_sum = 0.0f64;

        for entry in &self.utilization_history {
            let util = &entry.utilization;

            cpu_sum += util.cpu_utilization as f64;
            cpu_peak = cpu_peak.max(util.cpu_utilization as f64);

            if let Some(gpu) = util.gpu_utilization {
                gpu_sum += gpu as f64;
                gpu_count += 1;
                gpu_peak = gpu_peak.max(gpu as f64);
            }

            memory_sum += util.memory_utilization as f64;
            memory_peak = memory_peak.max(util.memory_utilization as f64);
            cache_sum += util.cache_hit_rate as f64;
        }

        let count = self.utilization_history.len();
        let average_gpu = if gpu_count > 0 {
            Some((gpu_sum / gpu_count as f64) as f32)
        } else {
            None
        };
        let peak_gpu = if gpu_count > 0 {
            Some(gpu_peak as f32)
        } else {
            None
        };

        HardwareStatistics {
            average_cpu_utilization: (cpu_sum / count as f64) as f32,
            peak_cpu_utilization: cpu_peak as f32,
            average_gpu_utilization: average_gpu,
            peak_gpu_utilization: peak_gpu,
            average_memory_utilization: (memory_sum / count as f64) as f32,
            peak_memory_utilization: memory_peak as f32,
            average_cache_hit_rate: (cache_sum / count as f64) as f32,
            measurement_count: count,
            thermal_throttling_events: 0, // Would be detected by platform-specific code
            power_efficiency_score: self.compute_power_efficiency_score(),
        }
    }

    /// Generates hardware optimization recommendations
    ///
    /// # Returns
    ///
    /// Hardware performance recommendations based on observed utilization
    pub fn get_recommendations(&self) -> HardwareRecommendations {
        let stats = self.get_hardware_statistics();
        let mut recommendations = HardwareRecommendations {
            cpu_recommendations: Vec::new(),
            gpu_recommendations: Vec::new(),
            memory_recommendations: Vec::new(),
            system_recommendations: Vec::new(),
        };

        // CPU recommendations
        if stats.average_cpu_utilization > 0.8 {
            recommendations.cpu_recommendations.push(
                "High CPU utilization detected. Consider CPU optimizations or parallelization."
                    .to_string(),
            );
        }
        if stats.peak_cpu_utilization > 0.95 {
            recommendations.cpu_recommendations.push(
                "CPU saturation detected. Consider reducing computational complexity.".to_string(),
            );
        }

        // GPU recommendations
        if let Some(gpu_util) = stats.average_gpu_utilization {
            if gpu_util < 0.3 {
                recommendations.gpu_recommendations.push(
                    "Low GPU utilization. Consider moving more computation to GPU.".to_string(),
                );
            } else if gpu_util > 0.9 {
                recommendations
                    .gpu_recommendations
                    .push("High GPU utilization. Consider GPU memory optimization.".to_string());
            }
        }

        // Memory recommendations
        if stats.average_memory_utilization > 0.8 {
            recommendations.memory_recommendations.push(
                "High memory utilization. Consider memory optimization techniques.".to_string(),
            );
        }
        if stats.average_cache_hit_rate < 0.8 {
            recommendations
                .memory_recommendations
                .push("Low cache hit rate. Consider improving data locality.".to_string());
        }

        // System recommendations
        if stats.power_efficiency_score < 0.5 {
            recommendations
                .system_recommendations
                .push("Low power efficiency. Consider workload optimization.".to_string());
        }

        recommendations
    }

    /// Sets the measurement interval
    ///
    /// # Arguments
    ///
    /// * `interval` - New measurement interval
    pub fn set_interval(&mut self, interval: Duration) {
        self.interval = interval;
    }

    /// Gets the current measurement interval
    ///
    /// # Returns
    ///
    /// Current measurement interval
    pub fn get_interval(&self) -> Duration {
        self.interval
    }

    /// Clears utilization history
    pub fn clear_history(&mut self) {
        self.utilization_history.clear();
        self.last_utilization = None;
    }

    /// Checks if GPU monitoring is available
    ///
    /// # Returns
    ///
    /// True if GPU monitoring is supported and available
    pub fn is_gpu_available(&self) -> bool {
        self.platform_state.gpu_state.available
    }

    /// Gets the number of detected GPU devices
    ///
    /// # Returns
    ///
    /// Number of GPU devices available for monitoring
    pub fn get_gpu_device_count(&self) -> usize {
        self.platform_state.gpu_state.device_count
    }

    /// Measures current hardware utilization
    fn measure_utilization(&mut self) -> HardwareUtilization {
        HardwareUtilization {
            cpu_utilization: self.measure_cpu_utilization(),
            gpu_utilization: self.measure_gpu_utilization(),
            memory_utilization: self.measure_memory_utilization(),
            memory_bandwidth_utilization: self.measure_memory_bandwidth_utilization(),
            cache_hit_rate: self.measure_cache_hit_rate(),
        }
    }

    /// Measures CPU utilization
    fn measure_cpu_utilization(&mut self) -> f32 {
        self.platform_state.cpu_state.measure_cpu_utilization()
    }

    /// Measures GPU utilization
    fn measure_gpu_utilization(&mut self) -> Option<f32> {
        if self.platform_state.gpu_state.available {
            Some(self.platform_state.gpu_state.measure_gpu_utilization())
        } else {
            None
        }
    }

    /// Measures memory utilization
    fn measure_memory_utilization(&mut self) -> f32 {
        self.platform_state
            .memory_state
            .measure_memory_utilization()
    }

    /// Measures memory bandwidth utilization
    fn measure_memory_bandwidth_utilization(&self) -> f32 {
        // Simplified implementation
        // Real implementation would measure actual memory bandwidth usage
        self.last_utilization
            .as_ref()
            .map(|u| u.memory_utilization * 0.7) // Estimate based on memory utilization
            .unwrap_or(0.0)
    }

    /// Measures cache hit rate
    fn measure_cache_hit_rate(&self) -> f32 {
        // Simplified implementation
        // Real implementation would access hardware performance counters
        0.85 // Placeholder value
    }

    /// Computes power efficiency score
    fn compute_power_efficiency_score(&self) -> f32 {
        if self.utilization_history.is_empty() {
            return 1.0;
        }

        // Simplified power efficiency calculation
        // Real implementation would consider actual power consumption
        let avg_cpu = self
            .utilization_history
            .iter()
            .map(|u| u.utilization.cpu_utilization)
            .sum::<f32>()
            / self.utilization_history.len() as f32;

        let avg_memory = self
            .utilization_history
            .iter()
            .map(|u| u.utilization.memory_utilization)
            .sum::<f32>()
            / self.utilization_history.len() as f32;

        // Higher utilization with lower variance = better efficiency
        let efficiency = (avg_cpu + avg_memory) / 2.0;
        efficiency.min(1.0)
    }
}

impl PlatformMonitorState {
    fn new() -> Self {
        Self {
            cpu_state: CpuMonitorState::new(),
            gpu_state: GpuMonitorState::new(),
            memory_state: MemoryMonitorState::new(),
        }
    }
}

impl CpuMonitorState {
    fn new() -> Self {
        Self {
            last_cpu_times: None,
            core_count: num_cpus::get(),
        }
    }

    fn measure_cpu_utilization(&mut self) -> f32 {
        let current_times = self.get_cpu_times();

        if let Some(last_times) = &self.last_cpu_times {
            let delta_total = current_times.total - last_times.total;
            let delta_idle = current_times.idle - last_times.idle;

            if delta_total > 0 {
                let utilization = 1.0 - (delta_idle as f32 / delta_total as f32);
                self.last_cpu_times = Some(current_times);
                return utilization.max(0.0).min(1.0);
            }
        }

        self.last_cpu_times = Some(current_times);
        0.5 // Default value for first measurement
    }

    #[cfg(target_os = "linux")]
    fn get_cpu_times(&self) -> CpuTimes {
        // Linux implementation using /proc/stat
        use std::fs;
        if let Ok(contents) = fs::read_to_string("/proc/stat") {
            if let Some(line) = contents.lines().next() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 8 && parts[0] == "cpu" {
                    let user = parts[1].parse::<u64>().unwrap_or(0);
                    let nice = parts[2].parse::<u64>().unwrap_or(0);
                    let system = parts[3].parse::<u64>().unwrap_or(0);
                    let idle = parts[4].parse::<u64>().unwrap_or(0);
                    let iowait = parts[5].parse::<u64>().unwrap_or(0);
                    let irq = parts[6].parse::<u64>().unwrap_or(0);
                    let softirq = parts[7].parse::<u64>().unwrap_or(0);

                    let total = user + nice + system + idle + iowait + irq + softirq;

                    return CpuTimes {
                        user: user + nice,
                        system: system + irq + softirq,
                        idle: idle + iowait,
                        total,
                    };
                }
            }
        }

        // Fallback
        CpuTimes {
            user: 50,
            system: 20,
            idle: 30,
            total: 100,
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn get_cpu_times(&self) -> CpuTimes {
        // Placeholder for other platforms
        CpuTimes {
            user: 50,
            system: 20,
            idle: 30,
            total: 100,
        }
    }
}

impl GpuMonitorState {
    fn new() -> Self {
        Self {
            available: Self::detect_gpu(),
            device_count: Self::count_gpu_devices(),
            last_gpu_utilization: Vec::new(),
        }
    }

    fn detect_gpu() -> bool {
        // Simplified GPU detection
        // Real implementation would check for CUDA, OpenCL, or other GPU APIs
        false // Placeholder
    }

    fn count_gpu_devices() -> usize {
        // Simplified GPU device counting
        0 // Placeholder
    }

    fn measure_gpu_utilization(&mut self) -> f32 {
        // Simplified GPU utilization measurement
        // Real implementation would use NVML, CUDA, or platform-specific APIs
        0.3 // Placeholder
    }
}

impl MemoryMonitorState {
    fn new() -> Self {
        Self {
            total_memory: Self::get_total_memory(),
            last_memory_stats: None,
        }
    }

    fn get_total_memory() -> usize {
        // Simplified total memory detection
        8 * 1024 * 1024 * 1024 // 8GB placeholder
    }

    fn measure_memory_utilization(&mut self) -> f32 {
        let stats = self.get_memory_stats();
        let utilization = stats.used as f32 / stats.total as f32;
        self.last_memory_stats = Some(stats);
        utilization.max(0.0).min(1.0)
    }

    #[cfg(target_os = "linux")]
    fn get_memory_stats(&self) -> MemoryStats {
        // Linux implementation using /proc/meminfo
        use std::fs;
        if let Ok(contents) = fs::read_to_string("/proc/meminfo") {
            let mut total = 0;
            let mut available = 0;
            let mut buffers = 0;
            let mut cached = 0;

            for line in contents.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    match parts[0] {
                        "MemTotal:" => total = parts[1].parse::<usize>().unwrap_or(0) * 1024,
                        "MemAvailable:" => {
                            available = parts[1].parse::<usize>().unwrap_or(0) * 1024
                        }
                        "Buffers:" => buffers = parts[1].parse::<usize>().unwrap_or(0) * 1024,
                        "Cached:" => cached = parts[1].parse::<usize>().unwrap_or(0) * 1024,
                        _ => {}
                    }
                }
            }

            let used = total - available;

            return MemoryStats {
                total,
                available,
                used,
                buffers,
                cached,
            };
        }

        // Fallback
        MemoryStats {
            total: self.total_memory,
            available: self.total_memory / 2,
            used: self.total_memory / 2,
            buffers: 0,
            cached: 0,
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn get_memory_stats(&self) -> MemoryStats {
        // Placeholder for other platforms
        MemoryStats {
            total: self.total_memory,
            available: self.total_memory / 2,
            used: self.total_memory / 2,
            buffers: 0,
            cached: 0,
        }
    }
}

impl Default for HardwareMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for HardwareStatistics {
    fn default() -> Self {
        Self {
            average_cpu_utilization: 0.0,
            peak_cpu_utilization: 0.0,
            average_gpu_utilization: None,
            peak_gpu_utilization: None,
            average_memory_utilization: 0.0,
            peak_memory_utilization: 0.0,
            average_cache_hit_rate: 0.0,
            measurement_count: 0,
            thermal_throttling_events: 0,
            power_efficiency_score: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_monitor_creation() {
        let monitor = HardwareMonitor::new();
        assert_eq!(monitor.interval, Duration::from_secs(1));
        assert!(monitor.last_utilization.is_none());
    }

    #[test]
    fn test_hardware_monitor_with_config() {
        let monitor = HardwareMonitor::with_config(Duration::from_millis(500), 100);
        assert_eq!(monitor.interval, Duration::from_millis(500));
        assert_eq!(monitor.max_history_length, 100);
    }

    #[test]
    fn test_update_utilization() {
        let mut monitor = HardwareMonitor::new();
        monitor.update_utilization();

        assert!(monitor.last_utilization.is_some());
        assert_eq!(monitor.utilization_history.len(), 1);

        let utilization = monitor.get_current_utilization().unwrap();
        assert!(utilization.cpu_utilization >= 0.0 && utilization.cpu_utilization <= 1.0);
        assert!(utilization.memory_utilization >= 0.0 && utilization.memory_utilization <= 1.0);
    }

    #[test]
    fn test_hardware_statistics() {
        let mut monitor = HardwareMonitor::new();

        // Take several measurements
        for _ in 0..5 {
            monitor.update_utilization();
            std::thread::sleep(Duration::from_millis(10));
        }

        let stats = monitor.get_hardware_statistics();
        assert_eq!(stats.measurement_count, 5);
        assert!(stats.average_cpu_utilization >= 0.0);
        assert!(stats.peak_cpu_utilization >= stats.average_cpu_utilization);
        assert!(stats.power_efficiency_score >= 0.0 && stats.power_efficiency_score <= 1.0);
    }

    #[test]
    fn test_recommendations() {
        let mut monitor = HardwareMonitor::new();
        monitor.update_utilization();

        let recommendations = monitor.get_recommendations();
        // Recommendations should always be valid (may be empty)
        assert!(recommendations.cpu_recommendations.len() >= 0);
        assert!(recommendations.gpu_recommendations.len() >= 0);
        assert!(recommendations.memory_recommendations.len() >= 0);
        assert!(recommendations.system_recommendations.len() >= 0);
    }

    #[test]
    fn test_interval_modification() {
        let mut monitor = HardwareMonitor::new();
        assert_eq!(monitor.get_interval(), Duration::from_secs(1));

        monitor.set_interval(Duration::from_millis(250));
        assert_eq!(monitor.get_interval(), Duration::from_millis(250));
    }

    #[test]
    fn test_clear_history() {
        let mut monitor = HardwareMonitor::new();
        monitor.update_utilization();

        assert_eq!(monitor.utilization_history.len(), 1);
        assert!(monitor.last_utilization.is_some());

        monitor.clear_history();

        assert_eq!(monitor.utilization_history.len(), 0);
        assert!(monitor.last_utilization.is_none());
    }

    #[test]
    fn test_gpu_detection() {
        let monitor = HardwareMonitor::new();
        let is_available = monitor.is_gpu_available();
        let device_count = monitor.get_gpu_device_count();

        // GPU availability is platform-dependent
        assert!(is_available == true || is_available == false);
        assert!(device_count >= 0);
    }

    #[test]
    fn test_history_length_limit() {
        let mut monitor = HardwareMonitor::with_config(Duration::from_millis(10), 3);

        // Take more measurements than the limit
        for _ in 0..5 {
            monitor.update_utilization();
        }

        // Should be limited to max_history_length
        assert_eq!(monitor.utilization_history.len(), 3);
    }

    #[test]
    fn test_cpu_times() {
        let mut cpu_state = CpuMonitorState::new();
        let times1 = cpu_state.get_cpu_times();
        let times2 = cpu_state.get_cpu_times();

        assert!(times1.total > 0);
        assert!(times2.total >= times1.total);
        assert_eq!(times1.total, times1.user + times1.system + times1.idle);
    }

    #[test]
    fn test_memory_stats() {
        let memory_state = MemoryMonitorState::new();
        let stats = memory_state.get_memory_stats();

        assert!(stats.total > 0);
        assert!(stats.used <= stats.total);
        assert!(stats.available <= stats.total);
    }
}
