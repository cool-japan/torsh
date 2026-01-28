//! Performance profiling and monitoring utilities
//!
//! This module provides comprehensive profiling tools for tracking
//! package operations, measuring performance, and identifying bottlenecks.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// Operation profiler that tracks performance metrics
#[derive(Debug, Clone)]
pub struct OperationProfiler {
    entries: Arc<Mutex<Vec<ProfileEntry>>>,
    active_operations: Arc<Mutex<HashMap<String, Instant>>>,
}

/// A single profiling entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileEntry {
    /// Operation name
    pub name: String,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Start timestamp
    pub start_time: std::time::SystemTime,
    /// Memory delta (bytes allocated)
    pub memory_delta: i64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Profiling statistics for an operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileStats {
    /// Operation name
    pub name: String,
    /// Number of calls
    pub count: usize,
    /// Total duration
    pub total_duration_ms: u64,
    /// Average duration
    pub avg_duration_ms: f64,
    /// Minimum duration
    pub min_duration_ms: u64,
    /// Maximum duration
    pub max_duration_ms: u64,
    /// Standard deviation
    pub std_dev_ms: f64,
    /// 50th percentile (median)
    pub p50_ms: u64,
    /// 95th percentile
    pub p95_ms: u64,
    /// 99th percentile
    pub p99_ms: u64,
}

/// Package operation types for profiling
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PackageOperation {
    /// Package creation
    Create,
    /// Package loading
    Load,
    /// Package saving
    Save,
    /// Resource addition
    AddResource,
    /// Resource retrieval
    GetResource,
    /// Compression operation
    Compress,
    /// Decompression operation
    Decompress,
    /// Signing operation
    Sign,
    /// Verification operation
    Verify,
    /// Encryption operation
    Encrypt,
    /// Decryption operation
    Decrypt,
    /// Dependency resolution
    ResolveDependencies,
    /// Installation operation
    Install,
    /// Custom operation
    Custom(String),
}

impl PackageOperation {
    /// Get operation name as string
    pub fn as_str(&self) -> &str {
        match self {
            Self::Create => "create",
            Self::Load => "load",
            Self::Save => "save",
            Self::AddResource => "add_resource",
            Self::GetResource => "get_resource",
            Self::Compress => "compress",
            Self::Decompress => "decompress",
            Self::Sign => "sign",
            Self::Verify => "verify",
            Self::Encrypt => "encrypt",
            Self::Decrypt => "decrypt",
            Self::ResolveDependencies => "resolve_dependencies",
            Self::Install => "install",
            Self::Custom(name) => name,
        }
    }
}

impl OperationProfiler {
    /// Create a new operation profiler
    pub fn new() -> Self {
        Self {
            entries: Arc::new(Mutex::new(Vec::new())),
            active_operations: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Start profiling an operation
    pub fn start(&self, operation: &str) {
        let mut active = self
            .active_operations
            .lock()
            .expect("lock should not be poisoned");
        active.insert(operation.to_string(), Instant::now());
    }

    /// End profiling an operation
    pub fn end(&self, operation: &str) {
        self.end_with_metadata(operation, HashMap::new());
    }

    /// End profiling with additional metadata
    pub fn end_with_metadata(&self, operation: &str, metadata: HashMap<String, String>) {
        let mut active = self
            .active_operations
            .lock()
            .expect("lock should not be poisoned");
        if let Some(start) = active.remove(operation) {
            let duration = start.elapsed();
            let entry = ProfileEntry {
                name: operation.to_string(),
                duration_ms: duration.as_millis() as u64,
                start_time: std::time::SystemTime::now() - duration,
                memory_delta: 0, // Would require memory tracking integration
                metadata,
            };

            let mut entries = self.entries.lock().expect("lock should not be poisoned");
            entries.push(entry);
        }
    }

    /// Get all profile entries
    pub fn entries(&self) -> Vec<ProfileEntry> {
        self.entries
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Clear all entries
    pub fn clear(&self) {
        self.entries
            .lock()
            .expect("lock should not be poisoned")
            .clear();
        self.active_operations
            .lock()
            .expect("lock should not be poisoned")
            .clear();
    }

    /// Get statistics for a specific operation
    pub fn stats_for(&self, operation: &str) -> Option<ProfileStats> {
        let entries = self.entries.lock().expect("lock should not be poisoned");
        let operation_entries: Vec<_> = entries.iter().filter(|e| e.name == operation).collect();

        if operation_entries.is_empty() {
            return None;
        }

        let count = operation_entries.len();
        let durations: Vec<u64> = operation_entries.iter().map(|e| e.duration_ms).collect();

        let total: u64 = durations.iter().sum();
        let avg = total as f64 / count as f64;
        let min = *durations.iter().min().expect("reduction should succeed");
        let max = *durations.iter().max().expect("reduction should succeed");

        // Calculate standard deviation
        let variance = durations
            .iter()
            .map(|d| {
                let diff = *d as f64 - avg;
                diff * diff
            })
            .sum::<f64>()
            / count as f64;
        let std_dev = variance.sqrt();

        // Calculate percentiles
        let mut sorted_durations = durations.clone();
        sorted_durations.sort_unstable();
        let p50 = percentile(&sorted_durations, 50.0);
        let p95 = percentile(&sorted_durations, 95.0);
        let p99 = percentile(&sorted_durations, 99.0);

        Some(ProfileStats {
            name: operation.to_string(),
            count,
            total_duration_ms: total,
            avg_duration_ms: avg,
            min_duration_ms: min,
            max_duration_ms: max,
            std_dev_ms: std_dev,
            p50_ms: p50,
            p95_ms: p95,
            p99_ms: p99,
        })
    }

    /// Get statistics for all operations
    pub fn all_stats(&self) -> Vec<ProfileStats> {
        let entries = self.entries.lock().expect("lock should not be poisoned");
        let mut operation_names: Vec<String> = entries.iter().map(|e| e.name.clone()).collect();
        operation_names.sort();
        operation_names.dedup();

        drop(entries); // Release lock before calling stats_for

        operation_names
            .into_iter()
            .filter_map(|name| self.stats_for(&name))
            .collect()
    }

    /// Export profile report as JSON
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        let stats = self.all_stats();
        serde_json::to_string_pretty(&stats)
    }

    /// Get total profiling time
    pub fn total_time_ms(&self) -> u64 {
        self.entries
            .lock()
            .expect("lock should not be poisoned")
            .iter()
            .map(|e| e.duration_ms)
            .sum()
    }

    /// Get operation count
    pub fn operation_count(&self) -> usize {
        self.entries
            .lock()
            .expect("lock should not be poisoned")
            .len()
    }
}

impl Default for OperationProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate percentile from sorted data
fn percentile(sorted_data: &[u64], p: f64) -> u64 {
    if sorted_data.is_empty() {
        return 0;
    }
    let index = ((p / 100.0) * (sorted_data.len() - 1) as f64).round() as usize;
    sorted_data[index.min(sorted_data.len() - 1)]
}

/// RAII guard for automatic profiling
pub struct ProfileGuard<'a> {
    profiler: &'a OperationProfiler,
    operation: String,
    metadata: HashMap<String, String>,
}

impl<'a> ProfileGuard<'a> {
    /// Create a new profile guard
    pub fn new(profiler: &'a OperationProfiler, operation: impl Into<String>) -> Self {
        let operation = operation.into();
        profiler.start(&operation);
        Self {
            profiler,
            operation,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the profile entry
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

impl<'a> Drop for ProfileGuard<'a> {
    fn drop(&mut self) {
        self.profiler
            .end_with_metadata(&self.operation, self.metadata.clone());
    }
}

/// Global profiler instance
static GLOBAL_PROFILER: once_cell::sync::Lazy<OperationProfiler> =
    once_cell::sync::Lazy::new(OperationProfiler::new);

/// Get the global profiler instance
pub fn global_profiler() -> &'static OperationProfiler {
    &GLOBAL_PROFILER
}

/// Profile a function execution
pub fn profile<F, R>(operation: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = ProfileGuard::new(&GLOBAL_PROFILER, operation);
    f()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_profiler_basic() {
        let profiler = OperationProfiler::new();

        profiler.start("test_op");
        std::thread::sleep(Duration::from_millis(10));
        profiler.end("test_op");

        let entries = profiler.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "test_op");
        assert!(entries[0].duration_ms >= 10);
    }

    #[test]
    fn test_profiler_multiple_operations() {
        let profiler = OperationProfiler::new();

        for _ in 0..5 {
            profiler.start("op");
            std::thread::sleep(Duration::from_millis(5));
            profiler.end("op");
        }

        let stats = profiler.stats_for("op").unwrap();
        assert_eq!(stats.count, 5);
        assert!(stats.avg_duration_ms >= 5.0);
        assert!(stats.total_duration_ms >= 25);
    }

    #[test]
    fn test_profiler_stats() {
        let profiler = OperationProfiler::new();

        // Add entries with known durations
        profiler.start("test");
        std::thread::sleep(Duration::from_millis(10));
        profiler.end("test");

        profiler.start("test");
        std::thread::sleep(Duration::from_millis(20));
        profiler.end("test");

        let stats = profiler.stats_for("test").unwrap();
        assert_eq!(stats.count, 2);
        assert!(stats.min_duration_ms >= 10);
        assert!(stats.max_duration_ms >= 20);
        assert!(stats.avg_duration_ms >= 15.0);
    }

    #[test]
    fn test_profile_guard() {
        let profiler = OperationProfiler::new();

        {
            let _guard = ProfileGuard::new(&profiler, "guarded_op");
            std::thread::sleep(Duration::from_millis(10));
        } // Guard drops here, automatically recording the profile

        let entries = profiler.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "guarded_op");
    }

    #[test]
    fn test_profile_guard_with_metadata() {
        let profiler = OperationProfiler::new();

        {
            let mut guard = ProfileGuard::new(&profiler, "meta_op");
            guard.add_metadata("key", "value");
            std::thread::sleep(Duration::from_millis(5));
        }

        let entries = profiler.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].metadata.get("key").unwrap(), "value");
    }

    #[test]
    fn test_profiler_clear() {
        let profiler = OperationProfiler::new();

        profiler.start("op");
        profiler.end("op");
        assert_eq!(profiler.entries().len(), 1);

        profiler.clear();
        assert_eq!(profiler.entries().len(), 0);
    }

    #[test]
    fn test_profiler_export_json() {
        let profiler = OperationProfiler::new();

        profiler.start("test");
        std::thread::sleep(Duration::from_millis(5));
        profiler.end("test");

        let json = profiler.export_json().unwrap();
        assert!(json.contains("test"));
        assert!(json.contains("count"));
        // Verify it's valid JSON
        let _parsed: Vec<ProfileStats> = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_package_operation_names() {
        assert_eq!(PackageOperation::Create.as_str(), "create");
        assert_eq!(PackageOperation::Load.as_str(), "load");
        assert_eq!(PackageOperation::Compress.as_str(), "compress");
        assert_eq!(
            PackageOperation::Custom("custom_op".to_string()).as_str(),
            "custom_op"
        );
    }

    #[test]
    fn test_global_profiler() {
        global_profiler().clear(); // Clear any previous entries

        let result = profile("global_test", || {
            std::thread::sleep(Duration::from_millis(5));
            42
        });

        assert_eq!(result, 42);
        let entries = global_profiler().entries();
        assert!(!entries.is_empty());
    }

    #[test]
    fn test_percentile_calculation() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(percentile(&data, 0.0), 1);
        // 50th percentile of 10 items: index = 0.5 * 9 = 4.5, rounded = 5, value = 6
        assert_eq!(percentile(&data, 50.0), 6);
        assert_eq!(percentile(&data, 100.0), 10);

        // Test with smaller dataset
        let small = vec![1, 2, 3];
        assert_eq!(percentile(&small, 0.0), 1);
        assert_eq!(percentile(&small, 50.0), 2);
        assert_eq!(percentile(&small, 100.0), 3);
    }
}
