//! Communication profiling and performance monitoring for distributed training
//!
//! This module provides comprehensive profiling capabilities for distributed communication
//! operations, including timing measurements, bandwidth analysis, and performance statistics.

use crate::{TorshDistributedError, TorshResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Type of communication operation being profiled
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommunicationOpType {
    AllReduce,
    AllGather,
    ReduceScatter,
    Broadcast,
    Reduce,
    Scatter,
    Gather,
    Send,
    Recv,
    Barrier,
    AllToAll,
    Custom(u32),
}

impl std::fmt::Display for CommunicationOpType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CommunicationOpType::AllReduce => write!(f, "AllReduce"),
            CommunicationOpType::AllGather => write!(f, "AllGather"),
            CommunicationOpType::ReduceScatter => write!(f, "ReduceScatter"),
            CommunicationOpType::Broadcast => write!(f, "Broadcast"),
            CommunicationOpType::Reduce => write!(f, "Reduce"),
            CommunicationOpType::Scatter => write!(f, "Scatter"),
            CommunicationOpType::Gather => write!(f, "Gather"),
            CommunicationOpType::Send => write!(f, "Send"),
            CommunicationOpType::Recv => write!(f, "Recv"),
            CommunicationOpType::Barrier => write!(f, "Barrier"),
            CommunicationOpType::AllToAll => write!(f, "AllToAll"),
            CommunicationOpType::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

/// Individual communication event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationEvent {
    /// Unique event identifier
    pub event_id: u64,
    /// Type of communication operation
    pub op_type: CommunicationOpType,
    /// Rank of the process that initiated the operation
    pub rank: u32,
    /// World size at the time of operation
    pub world_size: u32,
    /// Size of data transferred in bytes
    pub data_size_bytes: usize,
    /// Start timestamp
    pub start_time: SystemTime,
    /// Duration of the operation
    pub duration: Duration,
    /// Bandwidth achieved (bytes per second)
    pub bandwidth_bps: f64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl CommunicationEvent {
    /// Create a new communication event
    pub fn new(
        event_id: u64,
        op_type: CommunicationOpType,
        rank: u32,
        world_size: u32,
        data_size_bytes: usize,
        start_time: SystemTime,
        duration: Duration,
    ) -> Self {
        let bandwidth_bps = if duration.as_secs_f64() > 0.0 {
            data_size_bytes as f64 / duration.as_secs_f64()
        } else {
            0.0
        };

        Self {
            event_id,
            op_type,
            rank,
            world_size,
            data_size_bytes,
            start_time,
            duration,
            bandwidth_bps,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the event
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get latency in milliseconds
    pub fn latency_ms(&self) -> f64 {
        self.duration.as_secs_f64() * 1000.0
    }

    /// Get bandwidth in MB/s
    pub fn bandwidth_mbps(&self) -> f64 {
        self.bandwidth_bps / (1024.0 * 1024.0)
    }
}

/// Statistics for a specific communication operation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStats {
    /// Total number of operations
    pub count: u64,
    /// Total data transferred (bytes)
    pub total_bytes: u64,
    /// Total time spent (duration)
    pub total_duration: Duration,
    /// Minimum latency observed
    pub min_latency: Duration,
    /// Maximum latency observed
    pub max_latency: Duration,
    /// Average latency
    pub avg_latency: Duration,
    /// Average bandwidth (bytes per second)
    pub avg_bandwidth_bps: f64,
    /// 95th percentile latency
    pub p95_latency: Duration,
    /// 99th percentile latency
    pub p99_latency: Duration,
}

impl Default for OperationStats {
    fn default() -> Self {
        Self {
            count: 0,
            total_bytes: 0,
            total_duration: Duration::ZERO,
            min_latency: Duration::MAX,
            max_latency: Duration::ZERO,
            avg_latency: Duration::ZERO,
            avg_bandwidth_bps: 0.0,
            p95_latency: Duration::ZERO,
            p99_latency: Duration::ZERO,
        }
    }
}

impl OperationStats {
    /// Add a new event to the statistics
    pub fn add_event(&mut self, event: &CommunicationEvent) {
        self.count += 1;
        self.total_bytes += event.data_size_bytes as u64;
        self.total_duration += event.duration;

        if event.duration < self.min_latency {
            self.min_latency = event.duration;
        }
        if event.duration > self.max_latency {
            self.max_latency = event.duration;
        }

        // Recalculate averages
        self.avg_latency = self.total_duration / self.count as u32;
        if self.total_duration.as_secs_f64() > 0.0 {
            self.avg_bandwidth_bps = self.total_bytes as f64 / self.total_duration.as_secs_f64();
        }
    }

    /// Calculate percentiles from a list of durations
    pub fn calculate_percentiles(&mut self, durations: &mut [Duration]) {
        if durations.is_empty() {
            return;
        }

        durations.sort();
        let len = durations.len();

        let p95_idx = (len as f64 * 0.95).ceil() as usize - 1;
        let p99_idx = (len as f64 * 0.99).ceil() as usize - 1;

        self.p95_latency = durations[p95_idx.min(len - 1)];
        self.p99_latency = durations[p99_idx.min(len - 1)];
    }
}

/// Profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    /// Whether profiling is enabled
    pub enabled: bool,
    /// Maximum number of events to keep in memory
    pub max_events: usize,
    /// Whether to track detailed per-operation statistics
    pub track_per_operation_stats: bool,
    /// Whether to track per-rank statistics
    pub track_per_rank_stats: bool,
    /// Sampling rate (0.0 to 1.0, 1.0 means profile all operations)
    pub sampling_rate: f64,
    /// Minimum operation duration to record (microseconds)
    pub min_duration_us: u64,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_events: 10000,
            track_per_operation_stats: true,
            track_per_rank_stats: true,
            sampling_rate: 1.0,
            min_duration_us: 0,
        }
    }
}

/// Thread-safe communication profiler
pub struct CommunicationProfiler {
    /// Configuration
    config: RwLock<ProfilingConfig>,
    /// Event counter for unique IDs
    event_counter: Mutex<u64>,
    /// Circular buffer of recent events
    events: Mutex<Vec<CommunicationEvent>>,
    /// Statistics per operation type
    operation_stats: RwLock<HashMap<CommunicationOpType, OperationStats>>,
    /// Statistics per rank
    rank_stats: RwLock<HashMap<u32, HashMap<CommunicationOpType, OperationStats>>>,
    /// Global start time for relative timestamps
    start_time: SystemTime,
}

impl CommunicationProfiler {
    /// Create a new profiler with default configuration
    pub fn new() -> Self {
        Self::with_config(ProfilingConfig::default())
    }

    /// Create a new profiler with custom configuration
    pub fn with_config(config: ProfilingConfig) -> Self {
        Self {
            config: RwLock::new(config),
            event_counter: Mutex::new(0),
            events: Mutex::new(Vec::new()),
            operation_stats: RwLock::new(HashMap::new()),
            rank_stats: RwLock::new(HashMap::new()),
            start_time: SystemTime::now(),
        }
    }

    /// Start timing a communication operation
    pub fn start_timing(&self) -> ProfilingTimer {
        ProfilingTimer::new()
    }

    /// Record a communication event
    pub fn record_event(
        &self,
        op_type: CommunicationOpType,
        rank: u32,
        world_size: u32,
        data_size_bytes: usize,
        timer: ProfilingTimer,
    ) -> TorshResult<()> {
        let config = self
            .config
            .read()
            .map_err(|_| TorshDistributedError::backend_error("profiling", "Lock poisoned"))?;

        if !config.enabled {
            return Ok(());
        }

        let duration = timer.elapsed();

        // Skip if duration is below threshold
        if duration.as_micros() < config.min_duration_us as u128 {
            return Ok(());
        }

        // Apply sampling
        if config.sampling_rate < 1.0 {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            (
                rank,
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos(),
            )
                .hash(&mut hasher);
            let hash_val = hasher.finish();
            let sample_threshold = (u64::MAX as f64 * config.sampling_rate) as u64;

            if hash_val > sample_threshold {
                return Ok(());
            }
        }

        // Generate unique event ID
        let event_id = {
            let mut counter = self
                .event_counter
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("profiling", "Lock poisoned"))?;
            *counter += 1;
            *counter
        };

        // Create event
        let event = CommunicationEvent::new(
            event_id,
            op_type,
            rank,
            world_size,
            data_size_bytes,
            timer.start_time,
            duration,
        );

        // Store event
        {
            let mut events = self
                .events
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("profiling", "Lock poisoned"))?;
            events.push(event.clone());

            // Maintain circular buffer
            if events.len() > config.max_events {
                events.remove(0);
            }
        }

        // Update statistics
        if config.track_per_operation_stats {
            let mut stats = self
                .operation_stats
                .write()
                .map_err(|_| TorshDistributedError::backend_error("profiling", "Lock poisoned"))?;
            stats.entry(op_type).or_default().add_event(&event);
        }

        if config.track_per_rank_stats {
            let mut rank_stats = self
                .rank_stats
                .write()
                .map_err(|_| TorshDistributedError::backend_error("profiling", "Lock poisoned"))?;
            rank_stats
                .entry(rank)
                .or_default()
                .entry(op_type)
                .or_default()
                .add_event(&event);
        }

        Ok(())
    }

    /// Get statistics for a specific operation type
    pub fn get_operation_stats(
        &self,
        op_type: CommunicationOpType,
    ) -> TorshResult<Option<OperationStats>> {
        let stats = self
            .operation_stats
            .read()
            .map_err(|_| TorshDistributedError::backend_error("profiling", "Lock poisoned"))?;
        Ok(stats.get(&op_type).cloned())
    }

    /// Get all operation statistics
    pub fn get_all_operation_stats(
        &self,
    ) -> TorshResult<HashMap<CommunicationOpType, OperationStats>> {
        let stats = self
            .operation_stats
            .read()
            .map_err(|_| TorshDistributedError::backend_error("profiling", "Lock poisoned"))?;
        Ok(stats.clone())
    }

    /// Get statistics for a specific rank
    pub fn get_rank_stats(
        &self,
        rank: u32,
    ) -> TorshResult<Option<HashMap<CommunicationOpType, OperationStats>>> {
        let rank_stats = self
            .rank_stats
            .read()
            .map_err(|_| TorshDistributedError::backend_error("profiling", "Lock poisoned"))?;
        Ok(rank_stats.get(&rank).cloned())
    }

    /// Get recent events (last N events)
    pub fn get_recent_events(&self, count: usize) -> TorshResult<Vec<CommunicationEvent>> {
        let events = self
            .events
            .lock()
            .map_err(|_| TorshDistributedError::backend_error("profiling", "Lock poisoned"))?;
        let start_idx = events.len().saturating_sub(count);
        Ok(events[start_idx..].to_vec())
    }

    /// Get all events
    pub fn get_all_events(&self) -> TorshResult<Vec<CommunicationEvent>> {
        let events = self
            .events
            .lock()
            .map_err(|_| TorshDistributedError::backend_error("profiling", "Lock poisoned"))?;
        Ok(events.clone())
    }

    /// Get the count of failed operations across all ranks and operation types
    pub fn get_failed_operations_count(&self) -> u64 {
        let events = match self.events.lock() {
            Ok(events) => events,
            Err(_) => return 0, // Return 0 if lock is poisoned
        };

        // Count events that indicate failures (placeholder implementation)
        // In a real implementation, you would track operation success/failure explicitly
        events
            .iter()
            .filter(|event| {
                // Consider events with very high latency as potential failures
                // This is a heuristic approach for demonstration
                event.duration.as_millis() > 10000 || event.metadata.contains_key("error")
            })
            .count() as u64
    }

    /// Clear all profiling data
    pub fn clear(&self) -> TorshResult<()> {
        {
            let mut events = self
                .events
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("profiling", "Lock poisoned"))?;
            events.clear();
        }

        {
            let mut stats = self
                .operation_stats
                .write()
                .map_err(|_| TorshDistributedError::backend_error("profiling", "Lock poisoned"))?;
            stats.clear();
        }

        {
            let mut rank_stats = self
                .rank_stats
                .write()
                .map_err(|_| TorshDistributedError::backend_error("profiling", "Lock poisoned"))?;
            rank_stats.clear();
        }

        {
            let mut counter = self
                .event_counter
                .lock()
                .map_err(|_| TorshDistributedError::backend_error("profiling", "Lock poisoned"))?;
            *counter = 0;
        }

        Ok(())
    }

    /// Update configuration
    pub fn update_config(&self, config: ProfilingConfig) -> TorshResult<()> {
        let mut current_config = self
            .config
            .write()
            .map_err(|_| TorshDistributedError::backend_error("profiling", "Lock poisoned"))?;
        *current_config = config;
        Ok(())
    }

    /// Export profiling data to JSON
    pub fn export_json(&self) -> TorshResult<String> {
        #[derive(Serialize)]
        struct ExportData {
            config: ProfilingConfig,
            events: Vec<CommunicationEvent>,
            operation_stats: HashMap<CommunicationOpType, OperationStats>,
            rank_stats: HashMap<u32, HashMap<CommunicationOpType, OperationStats>>,
        }

        let config = self
            .config
            .read()
            .map_err(|_| TorshDistributedError::backend_error("profiling", "Lock poisoned"))?
            .clone();
        let events = self.get_all_events()?;
        let operation_stats = self.get_all_operation_stats()?;
        let rank_stats = self
            .rank_stats
            .read()
            .map_err(|_| TorshDistributedError::backend_error("profiling", "Lock poisoned"))?
            .clone();

        let export_data = ExportData {
            config,
            events,
            operation_stats,
            rank_stats,
        };

        serde_json::to_string_pretty(&export_data).map_err(|e| {
            TorshDistributedError::backend_error(
                "profiling",
                format!("JSON serialization failed: {}", e),
            )
        })
    }

    /// Generate a summary report
    pub fn generate_summary(&self) -> TorshResult<String> {
        let mut report = String::new();
        report.push_str("=== Communication Profiling Summary ===\n\n");

        let events = self.get_all_events()?;
        let operation_stats = self.get_all_operation_stats()?;

        report.push_str(&format!("Total Events: {}\n", events.len()));
        report.push_str(&format!(
            "Profiling Duration: {:.2}s\n\n",
            SystemTime::now()
                .duration_since(self.start_time)
                .unwrap_or_default()
                .as_secs_f64()
        ));

        report.push_str("=== Per-Operation Statistics ===\n");
        for (op_type, stats) in operation_stats.iter() {
            report.push_str(&format!("\n{} Operations:\n", op_type));
            report.push_str(&format!("  Count: {}\n", stats.count));
            report.push_str(&format!(
                "  Total Data: {:.2} MB\n",
                stats.total_bytes as f64 / (1024.0 * 1024.0)
            ));
            report.push_str(&format!(
                "  Avg Latency: {:.2} ms\n",
                stats.avg_latency.as_secs_f64() * 1000.0
            ));
            report.push_str(&format!(
                "  Min Latency: {:.2} ms\n",
                stats.min_latency.as_secs_f64() * 1000.0
            ));
            report.push_str(&format!(
                "  Max Latency: {:.2} ms\n",
                stats.max_latency.as_secs_f64() * 1000.0
            ));
            report.push_str(&format!(
                "  Avg Bandwidth: {:.2} MB/s\n",
                stats.avg_bandwidth_bps / (1024.0 * 1024.0)
            ));
        }

        Ok(report)
    }
}

impl Default for CommunicationProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Timer for measuring communication operation duration
pub struct ProfilingTimer {
    start_time: SystemTime,
    start_instant: Instant,
}

impl ProfilingTimer {
    /// Create a new timer and start timing
    pub fn new() -> Self {
        Self {
            start_time: SystemTime::now(),
            start_instant: Instant::now(),
        }
    }

    /// Get elapsed duration
    pub fn elapsed(&self) -> Duration {
        self.start_instant.elapsed()
    }

    /// Get start time
    pub fn start_time(&self) -> SystemTime {
        self.start_time
    }
}

/// Global profiler instance
static GLOBAL_PROFILER: std::sync::OnceLock<Arc<CommunicationProfiler>> =
    std::sync::OnceLock::new();

/// Get the global profiler instance
pub fn get_global_profiler() -> &'static Arc<CommunicationProfiler> {
    GLOBAL_PROFILER.get_or_init(|| Arc::new(CommunicationProfiler::new()))
}

/// Initialize the global profiler with custom configuration
pub fn init_global_profiler(config: ProfilingConfig) -> TorshResult<()> {
    let profiler = Arc::new(CommunicationProfiler::with_config(config));
    GLOBAL_PROFILER.set(profiler).map_err(|_| {
        TorshDistributedError::backend_error("profiling", "Global profiler already initialized")
    })?;
    Ok(())
}

/// Convenience macro for profiling communication operations
#[macro_export]
macro_rules! profile_communication {
    ($op_type:expr, $rank:expr, $world_size:expr, $data_size:expr, $code:block) => {{
        let profiler = $crate::profiling::get_global_profiler();
        let timer = profiler.start_timing();
        let result = $code;
        let _ = profiler.record_event($op_type, $rank, $world_size, $data_size, timer);
        result
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = CommunicationProfiler::new();
        let stats = profiler.get_all_operation_stats().unwrap();
        assert!(stats.is_empty());
    }

    #[test]
    fn test_event_recording() {
        let profiler = CommunicationProfiler::new();
        let timer = profiler.start_timing();
        std::thread::sleep(Duration::from_millis(10));

        profiler
            .record_event(CommunicationOpType::AllReduce, 0, 4, 1024, timer)
            .unwrap();

        let events = profiler.get_all_events().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].op_type, CommunicationOpType::AllReduce);
        assert_eq!(events[0].data_size_bytes, 1024);
    }

    #[test]
    fn test_operation_stats() {
        let profiler = CommunicationProfiler::new();

        // Record multiple events
        for i in 0..5 {
            let timer = profiler.start_timing();
            std::thread::sleep(Duration::from_millis(1));
            profiler
                .record_event(CommunicationOpType::AllReduce, 0, 4, 1024 * (i + 1), timer)
                .unwrap();
        }

        let stats = profiler
            .get_operation_stats(CommunicationOpType::AllReduce)
            .unwrap();
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert_eq!(stats.count, 5);
        assert_eq!(stats.total_bytes, 1024 + 2048 + 3072 + 4096 + 5120);
    }

    #[test]
    fn test_profiler_macro() {
        let result = profile_communication!(CommunicationOpType::Broadcast, 0, 4, 2048, {
            std::thread::sleep(Duration::from_millis(5));
            42
        });

        assert_eq!(result, 42);

        let profiler = get_global_profiler();
        let events = profiler.get_all_events().unwrap();
        assert!(!events.is_empty());
    }

    #[test]
    fn test_json_export() {
        let profiler = CommunicationProfiler::new();
        let timer = profiler.start_timing();
        std::thread::sleep(Duration::from_millis(1));

        profiler
            .record_event(CommunicationOpType::AllGather, 0, 4, 512, timer)
            .unwrap();

        let json = profiler.export_json().unwrap();
        assert!(json.contains("AllGather"));
        assert!(json.contains("512"));
    }

    #[test]
    fn test_summary_generation() {
        let profiler = CommunicationProfiler::new();
        let timer = profiler.start_timing();
        std::thread::sleep(Duration::from_millis(1));

        profiler
            .record_event(CommunicationOpType::Reduce, 0, 4, 256, timer)
            .unwrap();

        let summary = profiler.generate_summary().unwrap();
        assert!(summary.contains("Communication Profiling Summary"));
        assert!(summary.contains("Reduce Operations"));
    }
}
