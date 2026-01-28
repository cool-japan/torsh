//! Memory Profiling System with Allocation Tracking
//!
//! This module provides comprehensive memory profiling capabilities for tracking tensor
//! memory usage, detecting leaks, and analyzing allocation patterns.
//!
//! # Features
//!
//! - **Allocation Tracking**: Track all tensor allocations and deallocations
//! - **Memory Leak Detection**: Identify memory leaks and unreleased tensors
//! - **Peak Usage Monitoring**: Track peak memory usage over time
//! - **Allocation Patterns**: Analyze allocation size distributions and patterns
//! - **Stack Traces**: Optional stack trace capture for allocation sites
//! - **Per-Device Tracking**: Separate tracking for CPU and GPU memory
//! - **Timeline Analysis**: Track memory usage over time
//!
//! # Examples
//!
//! ```rust,ignore
//! use torsh_tensor::memory_profiler::{MemoryProfiler, ProfilerConfig};
//!
//! // Create profiler
//! let profiler = MemoryProfiler::new(ProfilerConfig::default());
//!
//! // Track allocations
//! profiler.track_allocation(1024, "tensor_data", DeviceType::Cpu);
//!
//! // Get memory report
//! let report = profiler.generate_report();
//! println!("Peak memory: {} bytes", report.peak_memory_bytes);
//! println!("Current allocations: {}", report.active_allocations);
//!
//! // Detect leaks
//! let leaks = profiler.detect_leaks();
//! if !leaks.is_empty() {
//!     println!("Warning: {} memory leaks detected!", leaks.len());
//! }
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use torsh_core::device::DeviceType;

/// Configuration for memory profiler
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Whether to capture stack traces (high overhead)
    pub capture_stack_traces: bool,
    /// Maximum number of timeline entries to keep
    pub max_timeline_entries: usize,
    /// Whether to track allocation patterns
    pub track_patterns: bool,
    /// Sampling rate (1.0 = all allocations, 0.5 = half, etc.)
    pub sampling_rate: f64,
    /// Size threshold for tracking (bytes, smaller allocations ignored)
    pub size_threshold: usize,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            capture_stack_traces: false,
            max_timeline_entries: 10000,
            track_patterns: true,
            sampling_rate: 1.0,
            size_threshold: 0,
        }
    }
}

/// Information about a memory allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationInfo {
    /// Unique allocation ID
    pub id: u64,
    /// Size in bytes
    pub size: usize,
    /// Allocation type/tag
    pub tag: String,
    /// Device where memory was allocated (as string)
    pub device: String,
    /// Timestamp of allocation
    pub timestamp: SystemTime,
    /// Stack trace (if enabled)
    pub stack_trace: Option<Vec<String>>,
    /// Whether this allocation has been freed
    pub freed: bool,
    /// Timestamp of deallocation (if freed)
    pub freed_at: Option<SystemTime>,
}

/// Memory usage snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Timestamp of snapshot
    pub timestamp: SystemTime,
    /// Total allocated bytes
    pub total_bytes: usize,
    /// Number of active allocations
    pub active_count: usize,
    /// Allocated bytes by device
    pub by_device: HashMap<String, usize>,
    /// Allocated bytes by tag
    pub by_tag: HashMap<String, usize>,
}

/// Memory leak information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLeak {
    /// Allocation that leaked
    pub allocation: AllocationInfo,
    /// How long it's been since allocation (seconds)
    pub age_seconds: f64,
}

/// Memory usage patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationPattern {
    /// Size distribution (size ranges -> count)
    pub size_distribution: HashMap<String, usize>,
    /// Most common allocation sizes
    pub common_sizes: Vec<(usize, usize)>, // (size, count)
    /// Average allocation size
    pub avg_size: usize,
    /// Median allocation size
    pub median_size: usize,
    /// Total number of allocations
    pub total_allocations: usize,
    /// Total number of deallocations
    pub total_deallocations: usize,
}

/// Comprehensive memory usage report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReport {
    /// Report generation timestamp
    pub timestamp: SystemTime,
    /// Current memory usage (bytes)
    pub current_memory_bytes: usize,
    /// Peak memory usage (bytes)
    pub peak_memory_bytes: usize,
    /// Number of active allocations
    pub active_allocations: usize,
    /// Total allocations ever made
    pub total_allocations: u64,
    /// Total deallocations ever made
    pub total_deallocations: u64,
    /// Memory by device
    pub memory_by_device: HashMap<String, usize>,
    /// Memory by tag
    pub memory_by_tag: HashMap<String, usize>,
    /// Allocation patterns (if enabled)
    pub patterns: Option<AllocationPattern>,
    /// Detected memory leaks
    pub potential_leaks: Vec<MemoryLeak>,
}

/// Memory profiler implementation
pub struct MemoryProfiler {
    /// Configuration
    config: ProfilerConfig,
    /// Next allocation ID
    next_id: Arc<Mutex<u64>>,
    /// Active allocations
    allocations: Arc<RwLock<HashMap<u64, AllocationInfo>>>,
    /// Allocation history (freed allocations)
    history: Arc<RwLock<VecDeque<AllocationInfo>>>,
    /// Memory timeline
    timeline: Arc<RwLock<VecDeque<MemorySnapshot>>>,
    /// Statistics
    stats: Arc<RwLock<ProfilerStats>>,
}

/// Internal statistics
#[derive(Debug, Clone, Default)]
struct ProfilerStats {
    /// Total bytes ever allocated
    total_allocated: u64,
    /// Total bytes ever deallocated
    total_deallocated: u64,
    /// Peak memory usage
    peak_memory: usize,
    /// Peak allocation count
    peak_allocations: usize,
    /// Allocation count by size bucket
    size_buckets: HashMap<String, usize>,
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            config,
            next_id: Arc::new(Mutex::new(1)), // Start at 1 so IDs are > 0
            allocations: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(VecDeque::new())),
            timeline: Arc::new(RwLock::new(VecDeque::new())),
            stats: Arc::new(RwLock::new(ProfilerStats::default())),
        }
    }

    /// Track a new allocation
    pub fn track_allocation(&self, size: usize, tag: impl Into<String>, device: DeviceType) -> u64 {
        // Apply sampling (only if sampling_rate < 1.0)
        if self.config.sampling_rate < 1.0 && self.config.sampling_rate > 0.0 {
            use scirs2_core::random::*;
            let mut rng = thread_rng();
            let sample = rng.random::<f64>();
            if sample >= self.config.sampling_rate {
                return 0; // Skip this allocation
            }
        }

        // Apply size threshold
        if size < self.config.size_threshold {
            return 0;
        }

        let id = {
            let mut next_id = self.next_id.lock().expect("lock should not be poisoned");
            let id = *next_id;
            *next_id += 1;
            id
        };

        let allocation = AllocationInfo {
            id,
            size,
            tag: tag.into(),
            device: format!("{:?}", device),
            timestamp: SystemTime::now(),
            stack_trace: if self.config.capture_stack_traces {
                Some(Self::capture_stack_trace())
            } else {
                None
            },
            freed: false,
            freed_at: None,
        };

        // Store allocation
        {
            let mut allocations = self
                .allocations
                .write()
                .expect("lock should not be poisoned");
            allocations.insert(id, allocation);
        }

        // Update statistics
        {
            let mut stats = self.stats.write().expect("lock should not be poisoned");
            stats.total_allocated += size as u64;

            let current_memory = self.get_current_memory();
            if current_memory > stats.peak_memory {
                stats.peak_memory = current_memory;
            }

            let current_count = self
                .allocations
                .read()
                .expect("lock should not be poisoned")
                .len();
            if current_count > stats.peak_allocations {
                stats.peak_allocations = current_count;
            }

            if self.config.track_patterns {
                let bucket = Self::size_to_bucket(size);
                *stats.size_buckets.entry(bucket).or_insert(0) += 1;
            }
        }

        // Take snapshot periodically
        if id % 100 == 0 {
            self.take_snapshot();
        }

        id
    }

    /// Track deallocation
    pub fn track_deallocation(&self, id: u64) {
        let allocation = {
            let mut allocations = self
                .allocations
                .write()
                .expect("lock should not be poisoned");
            allocations.remove(&id)
        };

        if let Some(mut alloc) = allocation {
            alloc.freed = true;
            alloc.freed_at = Some(SystemTime::now());

            // Update statistics
            {
                let mut stats = self.stats.write().expect("lock should not be poisoned");
                stats.total_deallocated += alloc.size as u64;
            }

            // Add to history
            {
                let mut history = self.history.write().expect("lock should not be poisoned");
                history.push_back(alloc);

                // Limit history size
                if history.len() > self.config.max_timeline_entries {
                    history.pop_front();
                }
            }
        }
    }

    /// Get current memory usage
    pub fn get_current_memory(&self) -> usize {
        self.allocations
            .read()
            .unwrap()
            .values()
            .map(|a| a.size)
            .sum()
    }

    /// Get peak memory usage
    pub fn get_peak_memory(&self) -> usize {
        self.stats
            .read()
            .expect("lock should not be poisoned")
            .peak_memory
    }

    /// Get number of active allocations
    pub fn get_active_count(&self) -> usize {
        self.allocations
            .read()
            .expect("lock should not be poisoned")
            .len()
    }

    /// Take a memory snapshot
    pub fn take_snapshot(&self) {
        let allocations = self
            .allocations
            .read()
            .expect("lock should not be poisoned");

        let mut by_device = HashMap::new();
        let mut by_tag = HashMap::new();
        let mut total_bytes = 0;

        for alloc in allocations.values() {
            total_bytes += alloc.size;
            *by_device.entry(format!("{:?}", alloc.device)).or_insert(0) += alloc.size;
            *by_tag.entry(alloc.tag.clone()).or_insert(0) += alloc.size;
        }

        let snapshot = MemorySnapshot {
            timestamp: SystemTime::now(),
            total_bytes,
            active_count: allocations.len(),
            by_device,
            by_tag,
        };

        let mut timeline = self.timeline.write().expect("lock should not be poisoned");
        timeline.push_back(snapshot);

        // Limit timeline size
        if timeline.len() > self.config.max_timeline_entries {
            timeline.pop_front();
        }
    }

    /// Detect potential memory leaks
    pub fn detect_leaks(&self) -> Vec<MemoryLeak> {
        let now = SystemTime::now();
        let allocations = self
            .allocations
            .read()
            .expect("lock should not be poisoned");

        allocations
            .values()
            .filter_map(|alloc| {
                let age = now
                    .duration_since(alloc.timestamp)
                    .unwrap_or(Duration::from_secs(0));

                // Consider leaks if allocation has been alive for more than 60 seconds
                if age.as_secs() > 60 {
                    Some(MemoryLeak {
                        allocation: alloc.clone(),
                        age_seconds: age.as_secs_f64(),
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Generate allocation patterns
    pub fn generate_patterns(&self) -> AllocationPattern {
        let history = self.history.read().expect("lock should not be poisoned");
        let allocations = self
            .allocations
            .read()
            .expect("lock should not be poisoned");

        let mut all_sizes: Vec<usize> = history.iter().map(|a| a.size).collect();
        all_sizes.extend(allocations.values().map(|a| a.size));

        if all_sizes.is_empty() {
            return AllocationPattern {
                size_distribution: HashMap::new(),
                common_sizes: Vec::new(),
                avg_size: 0,
                median_size: 0,
                total_allocations: 0,
                total_deallocations: 0,
            };
        }

        // Calculate distribution
        let mut size_distribution = HashMap::new();
        for &size in &all_sizes {
            let bucket = Self::size_to_bucket(size);
            *size_distribution.entry(bucket).or_insert(0) += 1;
        }

        // Find most common sizes
        let mut size_counts: HashMap<usize, usize> = HashMap::new();
        for &size in &all_sizes {
            *size_counts.entry(size).or_insert(0) += 1;
        }

        let mut common_sizes: Vec<(usize, usize)> = size_counts.into_iter().collect();
        common_sizes.sort_by(|a, b| b.1.cmp(&a.1));
        common_sizes.truncate(10);

        // Calculate statistics
        let avg_size = all_sizes.iter().sum::<usize>() / all_sizes.len();

        all_sizes.sort_unstable();
        let median_size = all_sizes[all_sizes.len() / 2];

        AllocationPattern {
            size_distribution,
            common_sizes,
            avg_size,
            median_size,
            total_allocations: history.len() + allocations.len(),
            total_deallocations: history.len(),
        }
    }

    /// Generate comprehensive memory report
    pub fn generate_report(&self) -> MemoryReport {
        let allocations = self
            .allocations
            .read()
            .expect("lock should not be poisoned");
        let stats = self.stats.read().expect("lock should not be poisoned");

        let mut memory_by_device = HashMap::new();
        let mut memory_by_tag = HashMap::new();

        for alloc in allocations.values() {
            *memory_by_device
                .entry(format!("{:?}", alloc.device))
                .or_insert(0) += alloc.size;
            *memory_by_tag.entry(alloc.tag.clone()).or_insert(0) += alloc.size;
        }

        let current_memory = self.get_current_memory();
        let potential_leaks = self.detect_leaks();

        MemoryReport {
            timestamp: SystemTime::now(),
            current_memory_bytes: current_memory,
            peak_memory_bytes: stats.peak_memory,
            active_allocations: allocations.len(),
            total_allocations: stats.total_allocated
                / if current_memory > 0 {
                    current_memory
                } else {
                    1
                } as u64,
            total_deallocations: stats.total_deallocated
                / if current_memory > 0 {
                    current_memory
                } else {
                    1
                } as u64,
            memory_by_device,
            memory_by_tag,
            patterns: if self.config.track_patterns {
                Some(self.generate_patterns())
            } else {
                None
            },
            potential_leaks,
        }
    }

    /// Get timeline snapshots
    pub fn get_timeline(&self) -> Vec<MemorySnapshot> {
        self.timeline
            .read()
            .expect("lock should not be poisoned")
            .iter()
            .cloned()
            .collect()
    }

    /// Clear all tracking data
    pub fn clear(&self) {
        self.allocations
            .write()
            .expect("lock should not be poisoned")
            .clear();
        self.history
            .write()
            .expect("lock should not be poisoned")
            .clear();
        self.timeline
            .write()
            .expect("lock should not be poisoned")
            .clear();
        *self.stats.write().expect("lock should not be poisoned") = ProfilerStats::default();
    }

    /// Export report to JSON file
    pub fn export_report(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let report = self.generate_report();
        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Capture stack trace (placeholder)
    fn capture_stack_trace() -> Vec<String> {
        // In a real implementation, use the backtrace crate
        vec!["Stack trace not implemented".to_string()]
    }

    /// Convert size to bucket string for distribution
    fn size_to_bucket(size: usize) -> String {
        match size {
            0..=1024 => "0-1KB".to_string(),
            1025..=10240 => "1-10KB".to_string(),
            10241..=102400 => "10-100KB".to_string(),
            102401..=1048576 => "100KB-1MB".to_string(),
            1048577..=10485760 => "1-10MB".to_string(),
            10485761..=104857600 => "10-100MB".to_string(),
            _ => "100MB+".to_string(),
        }
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new(ProfilerConfig::default())
    }
}

/// Global memory profiler (singleton)
static GLOBAL_PROFILER: once_cell::sync::Lazy<Mutex<Option<MemoryProfiler>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(None));

/// Initialize global memory profiler
pub fn init_global_profiler(config: ProfilerConfig) {
    let mut global = GLOBAL_PROFILER.lock().expect("lock should not be poisoned");
    *global = Some(MemoryProfiler::new(config));
}

/// Get global memory profiler
pub fn global_profiler() -> Option<MemoryProfiler> {
    GLOBAL_PROFILER
        .lock()
        .unwrap()
        .as_ref()
        .map(|p| MemoryProfiler {
            config: p.config.clone(),
            next_id: Arc::clone(&p.next_id),
            allocations: Arc::clone(&p.allocations),
            history: Arc::clone(&p.history),
            timeline: Arc::clone(&p.timeline),
            stats: Arc::clone(&p.stats),
        })
}

/// Track allocation using global profiler
pub fn track_allocation(size: usize, tag: impl Into<String>, device: DeviceType) -> Option<u64> {
    global_profiler().map(|p| p.track_allocation(size, tag, device))
}

/// Track deallocation using global profiler
pub fn track_deallocation(id: u64) {
    if let Some(profiler) = global_profiler() {
        profiler.track_deallocation(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = MemoryProfiler::new(ProfilerConfig::default());
        assert_eq!(profiler.get_current_memory(), 0);
        assert_eq!(profiler.get_active_count(), 0);
    }

    #[test]
    fn test_track_allocation() {
        let profiler = MemoryProfiler::new(ProfilerConfig::default());
        let id = profiler.track_allocation(1024, "test", DeviceType::Cpu);

        assert_eq!(profiler.get_current_memory(), 1024);
        assert_eq!(profiler.get_active_count(), 1);
        assert!(id > 0);
    }

    #[test]
    fn test_track_deallocation() {
        let profiler = MemoryProfiler::new(ProfilerConfig::default());
        let id = profiler.track_allocation(1024, "test", DeviceType::Cpu);
        profiler.track_deallocation(id);

        assert_eq!(profiler.get_current_memory(), 0);
        assert_eq!(profiler.get_active_count(), 0);
    }

    #[test]
    fn test_peak_memory() {
        let profiler = MemoryProfiler::new(ProfilerConfig::default());

        let id1 = profiler.track_allocation(1024, "test1", DeviceType::Cpu);
        let id2 = profiler.track_allocation(2048, "test2", DeviceType::Cpu);

        assert_eq!(profiler.get_peak_memory(), 3072);

        profiler.track_deallocation(id1);
        profiler.track_deallocation(id2);

        // Peak should still be 3072
        assert_eq!(profiler.get_peak_memory(), 3072);
    }

    #[test]
    fn test_memory_report() {
        let profiler = MemoryProfiler::new(ProfilerConfig::default());

        profiler.track_allocation(1024, "tensor1", DeviceType::Cpu);
        profiler.track_allocation(2048, "tensor2", DeviceType::Cpu);

        let report = profiler.generate_report();
        assert_eq!(report.current_memory_bytes, 3072);
        assert_eq!(report.active_allocations, 2);
        assert!(report.patterns.is_some());
    }

    #[test]
    fn test_allocation_patterns() {
        let profiler = MemoryProfiler::new(ProfilerConfig {
            track_patterns: true,
            ..Default::default()
        });

        // Create various sized allocations
        profiler.track_allocation(512, "small", DeviceType::Cpu);
        profiler.track_allocation(1024, "medium", DeviceType::Cpu);
        profiler.track_allocation(1024, "medium", DeviceType::Cpu);
        profiler.track_allocation(10240, "large", DeviceType::Cpu);

        let patterns = profiler.generate_patterns();
        assert_eq!(patterns.total_allocations, 4);
        assert!(patterns.common_sizes.len() > 0);
    }

    #[test]
    fn test_size_threshold() {
        let profiler = MemoryProfiler::new(ProfilerConfig {
            size_threshold: 1000,
            ..Default::default()
        });

        // This should be tracked
        let id1 = profiler.track_allocation(2000, "large", DeviceType::Cpu);
        assert!(id1 > 0);
        assert_eq!(profiler.get_active_count(), 1);

        // This should be ignored
        let id2 = profiler.track_allocation(500, "small", DeviceType::Cpu);
        assert_eq!(id2, 0);
        assert_eq!(profiler.get_active_count(), 1);
    }

    #[test]
    fn test_snapshot() {
        let profiler = MemoryProfiler::new(ProfilerConfig::default());

        profiler.track_allocation(1024, "test", DeviceType::Cpu);
        profiler.take_snapshot();

        let timeline = profiler.get_timeline();
        assert!(!timeline.is_empty());
        assert_eq!(timeline[0].total_bytes, 1024);
        assert_eq!(timeline[0].active_count, 1);
    }

    #[test]
    fn test_clear() {
        let profiler = MemoryProfiler::new(ProfilerConfig::default());

        profiler.track_allocation(1024, "test", DeviceType::Cpu);
        assert_eq!(profiler.get_active_count(), 1);

        profiler.clear();
        assert_eq!(profiler.get_active_count(), 0);
        assert_eq!(profiler.get_current_memory(), 0);
    }
}
