//! Memory profiling

use crate::TorshResult;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use torsh_core::TorshError;

// Type alias for complex allocation trace type
type AllocationTrace = (usize, std::time::Instant, Option<String>);
type AllocationTracesMap = HashMap<usize, AllocationTrace>;

/// Memory statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub allocated: usize,
    pub reserved: usize,
    pub peak: usize,
    pub allocations: usize,
    pub deallocations: usize,
}

/// Memory leak information
#[derive(Debug, Clone)]
pub struct MemoryLeak {
    pub ptr: usize,
    pub size: usize,
    pub stack_trace: Option<String>,
    pub allocation_time: std::time::Instant,
}

/// Memory leak detection results
#[derive(Debug, Clone, Default)]
pub struct LeakDetectionResults {
    pub potential_leaks: Vec<MemoryLeak>,
    pub total_leaked_bytes: usize,
    pub leak_count: usize,
}

/// Memory allocation event for timeline tracking
#[derive(Debug, Clone)]
pub struct MemoryEvent {
    pub event_type: MemoryEventType,
    pub ptr: usize,
    pub size: usize,
    pub timestamp: std::time::Instant,
    pub thread_id: usize,
}

/// Type of memory event
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryEventType {
    Allocation,
    Deallocation,
}

/// Memory fragmentation analysis results
#[derive(Debug, Clone)]
pub struct FragmentationAnalysis {
    pub total_allocated: usize,
    pub largest_free_block: usize,
    pub fragmentation_ratio: f64,
    pub free_blocks: Vec<MemoryBlock>,
    pub allocated_blocks: Vec<MemoryBlock>,
    pub external_fragmentation: f64,
    pub internal_fragmentation: f64,
}

/// Memory block representation
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    pub start_address: usize,
    pub size: usize,
    pub end_address: usize,
}

/// Memory timeline analysis
#[derive(Debug, Clone)]
pub struct MemoryTimeline {
    pub events: Vec<MemoryEvent>,
    pub peak_usage_time: std::time::Instant,
    pub peak_usage_bytes: usize,
    pub allocation_rate: f64,   // allocations per second
    pub deallocation_rate: f64, // deallocations per second
    pub average_allocation_size: f64,
    pub memory_usage_over_time: Vec<(std::time::Duration, usize)>,
}

/// Memory profiler for tracking allocations and deallocations
pub struct MemoryProfiler {
    stats: Arc<Mutex<MemoryStats>>,
    allocations: Arc<Mutex<HashMap<usize, usize>>>, // ptr -> size
    enabled: bool,
    leak_detection_enabled: bool,
    allocation_traces: Arc<Mutex<AllocationTracesMap>>, // ptr -> (size, time, stack_trace)
    timeline_enabled: bool,
    timeline_events: Arc<Mutex<Vec<MemoryEvent>>>,
    fragmentation_enabled: bool,
    memory_pool_size: usize, // Simulated memory pool for fragmentation analysis
    start_time: std::time::Instant,
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new() -> Self {
        Self {
            stats: Arc::new(Mutex::new(MemoryStats::default())),
            allocations: Arc::new(Mutex::new(HashMap::new())),
            enabled: false,
            leak_detection_enabled: false,
            allocation_traces: Arc::new(Mutex::new(HashMap::new())),
            timeline_enabled: false,
            timeline_events: Arc::new(Mutex::new(Vec::new())),
            fragmentation_enabled: false,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB default pool size
            start_time: std::time::Instant::now(),
        }
    }

    /// Enable memory profiling
    pub fn enable(&mut self) {
        self.enabled = true;
        self.start_time = std::time::Instant::now();
        if let Ok(mut stats) = self.stats.lock() {
            *stats = MemoryStats::default();
        }
        if let Ok(mut allocations) = self.allocations.lock() {
            allocations.clear();
        }
        if let Ok(mut traces) = self.allocation_traces.lock() {
            traces.clear();
        }
        if let Ok(mut timeline) = self.timeline_events.lock() {
            timeline.clear();
        }
    }

    /// Disable memory profiling
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Record memory allocation
    pub fn record_allocation(&self, ptr: usize, size: usize) -> TorshResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let mut stats = self.stats.lock().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire lock on stats".to_string())
        })?;

        let mut allocations = self.allocations.lock().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire lock on allocations".to_string())
        })?;

        allocations.insert(ptr, size);
        stats.allocated += size;
        stats.allocations += 1;

        if stats.allocated > stats.peak {
            stats.peak = stats.allocated;
        }

        Ok(())
    }

    /// Record memory deallocation
    pub fn record_deallocation(&self, ptr: usize) -> TorshResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let timestamp = std::time::Instant::now();
        let thread_id = std::thread::current().id();
        let thread_id_num = format!("{thread_id:?}")
            .chars()
            .filter(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse::<usize>()
            .unwrap_or(0);

        let mut stats = self.stats.lock().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire lock on stats".to_string())
        })?;

        let mut allocations = self.allocations.lock().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire lock on allocations".to_string())
        })?;

        if let Some(size) = allocations.remove(&ptr) {
            stats.allocated = stats.allocated.saturating_sub(size);
            stats.deallocations += 1;

            // Remove from leak detection tracking if enabled
            if self.leak_detection_enabled {
                let mut traces = self.allocation_traces.lock().map_err(|_| {
                    TorshError::InvalidArgument(
                        "Failed to acquire lock on allocation traces".to_string(),
                    )
                })?;
                traces.remove(&ptr);
            }

            // Track for timeline if enabled
            if self.timeline_enabled {
                let mut timeline = self.timeline_events.lock().map_err(|_| {
                    TorshError::InvalidArgument(
                        "Failed to acquire lock on timeline events".to_string(),
                    )
                })?;
                timeline.push(MemoryEvent {
                    event_type: MemoryEventType::Deallocation,
                    ptr,
                    size,
                    timestamp,
                    thread_id: thread_id_num,
                });
            }
        }

        Ok(())
    }

    /// Get current memory statistics
    pub fn get_stats(&self) -> TorshResult<MemoryStats> {
        let stats = self.stats.lock().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire lock on stats".to_string())
        })?;
        Ok(stats.clone())
    }

    /// Reset memory statistics
    pub fn reset(&self) -> TorshResult<()> {
        let mut stats = self.stats.lock().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire lock on stats".to_string())
        })?;

        let mut allocations = self.allocations.lock().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire lock on allocations".to_string())
        })?;

        let mut traces = self.allocation_traces.lock().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire lock on allocation traces".to_string())
        })?;

        *stats = MemoryStats::default();
        allocations.clear();
        traces.clear();

        Ok(())
    }

    /// Enable or disable leak detection
    pub fn set_leak_detection_enabled(&mut self, enabled: bool) {
        self.leak_detection_enabled = enabled;
        if enabled {
            // Clear existing traces when enabling
            if let Ok(mut traces) = self.allocation_traces.lock() {
                traces.clear();
            }
        }
    }

    /// Check if leak detection is enabled
    pub fn is_leak_detection_enabled(&self) -> bool {
        self.leak_detection_enabled
    }

    /// Enable or disable timeline tracking
    pub fn set_timeline_enabled(&mut self, enabled: bool) {
        self.timeline_enabled = enabled;
        if enabled {
            if let Ok(mut timeline) = self.timeline_events.lock() {
                timeline.clear();
            }
        }
    }

    /// Check if timeline tracking is enabled
    pub fn is_timeline_enabled(&self) -> bool {
        self.timeline_enabled
    }

    /// Enable or disable fragmentation analysis
    pub fn set_fragmentation_enabled(&mut self, enabled: bool) {
        self.fragmentation_enabled = enabled;
    }

    /// Check if fragmentation analysis is enabled
    pub fn is_fragmentation_enabled(&self) -> bool {
        self.fragmentation_enabled
    }

    /// Set the memory pool size for fragmentation analysis
    pub fn set_memory_pool_size(&mut self, size: usize) {
        self.memory_pool_size = size;
    }

    /// Record memory allocation with optional stack trace
    pub fn record_allocation_with_trace(
        &self,
        ptr: usize,
        size: usize,
        stack_trace: Option<String>,
    ) -> TorshResult<()> {
        if !self.enabled {
            return Ok(());
        }

        let timestamp = std::time::Instant::now();
        let thread_id = std::thread::current().id();
        let thread_id_num = format!("{thread_id:?}")
            .chars()
            .filter(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse::<usize>()
            .unwrap_or(0);

        let mut stats = self.stats.lock().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire lock on stats".to_string())
        })?;

        let mut allocations = self.allocations.lock().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire lock on allocations".to_string())
        })?;

        allocations.insert(ptr, size);
        stats.allocated += size;
        stats.allocations += 1;

        if stats.allocated > stats.peak {
            stats.peak = stats.allocated;
        }

        // Track for leak detection if enabled
        if self.leak_detection_enabled {
            let mut traces = self.allocation_traces.lock().map_err(|_| {
                TorshError::InvalidArgument(
                    "Failed to acquire lock on allocation traces".to_string(),
                )
            })?;
            traces.insert(ptr, (size, timestamp, stack_trace));
        }

        // Track for timeline if enabled
        if self.timeline_enabled {
            let mut timeline = self.timeline_events.lock().map_err(|_| {
                TorshError::InvalidArgument("Failed to acquire lock on timeline events".to_string())
            })?;
            timeline.push(MemoryEvent {
                event_type: MemoryEventType::Allocation,
                ptr,
                size,
                timestamp,
                thread_id: thread_id_num,
            });
        }

        Ok(())
    }

    /// Detect memory leaks
    pub fn detect_leaks(&self) -> TorshResult<LeakDetectionResults> {
        if !self.leak_detection_enabled {
            return Ok(LeakDetectionResults::default());
        }

        let traces = self.allocation_traces.lock().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire lock on allocation traces".to_string())
        })?;

        let mut potential_leaks = Vec::new();
        let mut total_leaked_bytes = 0;

        for (&ptr, &(size, allocation_time, ref stack_trace)) in traces.iter() {
            potential_leaks.push(MemoryLeak {
                ptr,
                size,
                stack_trace: stack_trace.clone(),
                allocation_time,
            });
            total_leaked_bytes += size;
        }

        Ok(LeakDetectionResults {
            leak_count: potential_leaks.len(),
            total_leaked_bytes,
            potential_leaks,
        })
    }

    /// Get leaks older than a specific duration
    pub fn get_leaks_older_than(
        &self,
        duration: std::time::Duration,
    ) -> TorshResult<LeakDetectionResults> {
        if !self.leak_detection_enabled {
            return Ok(LeakDetectionResults::default());
        }

        let traces = self.allocation_traces.lock().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire lock on allocation traces".to_string())
        })?;

        let now = std::time::Instant::now();
        let mut potential_leaks = Vec::new();
        let mut total_leaked_bytes = 0;

        for (&ptr, &(size, allocation_time, ref stack_trace)) in traces.iter() {
            if now.duration_since(allocation_time) > duration {
                potential_leaks.push(MemoryLeak {
                    ptr,
                    size,
                    stack_trace: stack_trace.clone(),
                    allocation_time,
                });
                total_leaked_bytes += size;
            }
        }

        Ok(LeakDetectionResults {
            leak_count: potential_leaks.len(),
            total_leaked_bytes,
            potential_leaks,
        })
    }

    /// Get the largest potential leaks
    pub fn get_largest_leaks(&self, count: usize) -> TorshResult<LeakDetectionResults> {
        if !self.leak_detection_enabled {
            return Ok(LeakDetectionResults::default());
        }

        let traces = self.allocation_traces.lock().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire lock on allocation traces".to_string())
        })?;

        let mut leaks: Vec<MemoryLeak> = traces
            .iter()
            .map(
                |(&ptr, &(size, allocation_time, ref stack_trace))| MemoryLeak {
                    ptr,
                    size,
                    stack_trace: stack_trace.clone(),
                    allocation_time,
                },
            )
            .collect();

        leaks.sort_by(|a, b| b.size.cmp(&a.size));
        leaks.truncate(count);

        let total_leaked_bytes = leaks.iter().map(|leak| leak.size).sum();

        Ok(LeakDetectionResults {
            leak_count: leaks.len(),
            total_leaked_bytes,
            potential_leaks: leaks,
        })
    }

    /// Analyze memory fragmentation
    pub fn analyze_fragmentation(&self) -> TorshResult<FragmentationAnalysis> {
        if !self.fragmentation_enabled {
            return Err(TorshError::InvalidArgument(
                "Fragmentation analysis is not enabled".to_string(),
            ));
        }

        let allocations = self.allocations.lock().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire lock on allocations".to_string())
        })?;

        let mut allocated_blocks: Vec<MemoryBlock> = allocations
            .iter()
            .map(|(&ptr, &size)| MemoryBlock {
                start_address: ptr,
                size,
                end_address: ptr + size,
            })
            .collect();

        // Sort blocks by start address
        allocated_blocks.sort_by_key(|block| block.start_address);

        // Calculate free blocks (simplified simulation)
        let mut free_blocks = Vec::new();
        let mut current_addr = 0;

        for block in &allocated_blocks {
            if current_addr < block.start_address {
                free_blocks.push(MemoryBlock {
                    start_address: current_addr,
                    size: block.start_address - current_addr,
                    end_address: block.start_address,
                });
            }
            current_addr = block.end_address;
        }

        // Add remaining free space
        if current_addr < self.memory_pool_size {
            free_blocks.push(MemoryBlock {
                start_address: current_addr,
                size: self.memory_pool_size - current_addr,
                end_address: self.memory_pool_size,
            });
        }

        let total_allocated: usize = allocated_blocks.iter().map(|b| b.size).sum();
        let total_free: usize = free_blocks.iter().map(|b| b.size).sum();
        let largest_free_block = free_blocks.iter().map(|b| b.size).max().unwrap_or(0);

        // Calculate fragmentation metrics
        let fragmentation_ratio = if total_free > 0 {
            1.0 - (largest_free_block as f64 / total_free as f64)
        } else {
            0.0
        };

        let external_fragmentation = if total_free > 0 {
            (total_free - largest_free_block) as f64 / total_free as f64
        } else {
            0.0
        };

        // Internal fragmentation is harder to calculate without knowing actual allocation patterns
        // For now, we'll estimate it as a percentage based on allocation sizes
        let avg_allocation_size = if allocated_blocks.is_empty() {
            0.0
        } else {
            total_allocated as f64 / allocated_blocks.len() as f64
        };

        let internal_fragmentation = if avg_allocation_size > 0.0 {
            // Estimate 5% internal fragmentation for small allocations
            (avg_allocation_size / 1024.0).min(1.0) * 0.05
        } else {
            0.0
        };

        Ok(FragmentationAnalysis {
            total_allocated,
            largest_free_block,
            fragmentation_ratio,
            free_blocks,
            allocated_blocks,
            external_fragmentation,
            internal_fragmentation,
        })
    }

    /// Get memory timeline analysis
    pub fn get_timeline_analysis(&self) -> TorshResult<MemoryTimeline> {
        if !self.timeline_enabled {
            return Err(TorshError::InvalidArgument(
                "Timeline tracking is not enabled".to_string(),
            ));
        }

        let timeline = self.timeline_events.lock().map_err(|_| {
            TorshError::InvalidArgument("Failed to acquire lock on timeline events".to_string())
        })?;

        if timeline.is_empty() {
            return Ok(MemoryTimeline {
                events: Vec::new(),
                peak_usage_time: self.start_time,
                peak_usage_bytes: 0,
                allocation_rate: 0.0,
                deallocation_rate: 0.0,
                average_allocation_size: 0.0,
                memory_usage_over_time: Vec::new(),
            });
        }

        let events = timeline.clone();
        let total_duration = timeline
            .last()
            .expect("timeline should not be empty after early return check")
            .timestamp
            .duration_since(self.start_time)
            .as_secs_f64();

        // Calculate allocation and deallocation rates
        let allocation_count = events
            .iter()
            .filter(|e| e.event_type == MemoryEventType::Allocation)
            .count();
        let deallocation_count = events
            .iter()
            .filter(|e| e.event_type == MemoryEventType::Deallocation)
            .count();

        let allocation_rate = if total_duration > 0.0 {
            allocation_count as f64 / total_duration
        } else {
            0.0
        };

        let deallocation_rate = if total_duration > 0.0 {
            deallocation_count as f64 / total_duration
        } else {
            0.0
        };

        // Calculate average allocation size
        let allocation_sizes: Vec<usize> = events
            .iter()
            .filter(|e| e.event_type == MemoryEventType::Allocation)
            .map(|e| e.size)
            .collect();

        let average_allocation_size = if allocation_sizes.is_empty() {
            0.0
        } else {
            allocation_sizes.iter().sum::<usize>() as f64 / allocation_sizes.len() as f64
        };

        // Build memory usage over time
        let mut memory_usage_over_time = Vec::new();
        let mut current_usage = 0usize;
        let mut peak_usage_bytes = 0usize;
        let mut peak_usage_time = self.start_time;

        for event in &events {
            match event.event_type {
                MemoryEventType::Allocation => {
                    current_usage += event.size;
                    if current_usage > peak_usage_bytes {
                        peak_usage_bytes = current_usage;
                        peak_usage_time = event.timestamp;
                    }
                }
                MemoryEventType::Deallocation => {
                    current_usage = current_usage.saturating_sub(event.size);
                }
            }

            memory_usage_over_time.push((
                event.timestamp.duration_since(self.start_time),
                current_usage,
            ));
        }

        Ok(MemoryTimeline {
            events,
            peak_usage_time,
            peak_usage_bytes,
            allocation_rate,
            deallocation_rate,
            average_allocation_size,
            memory_usage_over_time,
        })
    }

    /// Export timeline to CSV format
    pub fn export_timeline_csv(&self, path: &str) -> TorshResult<()> {
        let timeline = self.get_timeline_analysis()?;

        use std::fs::File;
        use std::io::{BufWriter, Write};

        let file = File::create(path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create file {path}: {e}"))
        })?;

        let mut writer = BufWriter::new(file);

        // Write CSV header
        writeln!(
            writer,
            "timestamp_ms,event_type,ptr,size,thread_id,cumulative_usage"
        )
        .map_err(|e| TorshError::InvalidArgument(format!("Failed to write CSV header: {e}")))?;

        let mut current_usage = 0usize;
        for event in &timeline.events {
            match event.event_type {
                MemoryEventType::Allocation => current_usage += event.size,
                MemoryEventType::Deallocation => {
                    current_usage = current_usage.saturating_sub(event.size)
                }
            }

            let timestamp_ms = event.timestamp.duration_since(self.start_time).as_millis();
            let event_type_str = match event.event_type {
                MemoryEventType::Allocation => "allocation",
                MemoryEventType::Deallocation => "deallocation",
            };

            writeln!(
                writer,
                "{},{},{:#x},{},{},{}",
                timestamp_ms, event_type_str, event.ptr, event.size, event.thread_id, current_usage
            )
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to write CSV row: {e}")))?;
        }

        writer
            .flush()
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to flush CSV writer: {e}")))
    }
}

/// Get system memory information
pub fn get_system_memory() -> TorshResult<SystemMemoryInfo> {
    // This is a simplified implementation
    // In a real implementation, you would use platform-specific APIs
    Ok(SystemMemoryInfo {
        total: 8 * 1024 * 1024 * 1024,     // 8 GB
        available: 4 * 1024 * 1024 * 1024, // 4 GB
        used: 4 * 1024 * 1024 * 1024,      // 4 GB
    })
}

/// System memory information
#[derive(Debug, Clone)]
pub struct SystemMemoryInfo {
    pub total: usize,
    pub available: usize,
    pub used: usize,
}

/// Profile memory usage
pub fn profile_memory() -> TorshResult<MemoryStats> {
    let mut profiler = MemoryProfiler::new();
    profiler.enable();

    // Simulate some allocations
    let ptr1 = 0x1000;
    let ptr2 = 0x2000;

    profiler.record_allocation(ptr1, 1024)?;
    profiler.record_allocation(ptr2, 2048)?;
    profiler.record_deallocation(ptr1)?;

    profiler.get_stats()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_profiler_creation() {
        let profiler = MemoryProfiler::new();
        assert!(!profiler.enabled);
    }

    #[test]
    fn test_memory_profiler_enable_disable() {
        let mut profiler = MemoryProfiler::new();
        profiler.enable();
        assert!(profiler.enabled);

        profiler.disable();
        assert!(!profiler.enabled);
    }

    #[test]
    fn test_memory_allocation_tracking() {
        let mut profiler = MemoryProfiler::new();
        profiler.enable();

        profiler.record_allocation(0x1000, 1024).unwrap();
        profiler.record_allocation(0x2000, 2048).unwrap();

        let stats = profiler.get_stats().unwrap();
        assert_eq!(stats.allocated, 3072);
        assert_eq!(stats.allocations, 2);
        assert_eq!(stats.peak, 3072);
    }

    #[test]
    fn test_memory_deallocation_tracking() {
        let mut profiler = MemoryProfiler::new();
        profiler.enable();

        profiler.record_allocation(0x1000, 1024).unwrap();
        profiler.record_allocation(0x2000, 2048).unwrap();
        profiler.record_deallocation(0x1000).unwrap();

        let stats = profiler.get_stats().unwrap();
        assert_eq!(stats.allocated, 2048);
        assert_eq!(stats.allocations, 2);
        assert_eq!(stats.deallocations, 1);
        assert_eq!(stats.peak, 3072); // Peak should remain
    }

    #[test]
    fn test_memory_stats_reset() {
        let mut profiler = MemoryProfiler::new();
        profiler.enable();

        profiler.record_allocation(0x1000, 1024).unwrap();
        profiler.reset().unwrap();

        let stats = profiler.get_stats().unwrap();
        assert_eq!(stats.allocated, 0);
        assert_eq!(stats.allocations, 0);
        assert_eq!(stats.peak, 0);
    }

    #[test]
    fn test_leak_detection_basic() {
        let mut profiler = MemoryProfiler::new();
        profiler.enable();
        profiler.set_leak_detection_enabled(true);

        assert!(profiler.is_leak_detection_enabled());

        // Allocate some memory without deallocating
        profiler
            .record_allocation_with_trace(0x1000, 1024, Some("test_trace".to_string()))
            .unwrap();
        profiler
            .record_allocation_with_trace(0x2000, 2048, None)
            .unwrap();

        // Detect leaks
        let leaks = profiler.detect_leaks().unwrap();
        assert_eq!(leaks.leak_count, 2);
        assert_eq!(leaks.total_leaked_bytes, 3072);
        assert_eq!(leaks.potential_leaks.len(), 2);

        // Check leak details (order may vary due to HashMap iteration)
        let leak_0x1000 = leaks
            .potential_leaks
            .iter()
            .find(|l| l.ptr == 0x1000)
            .unwrap();
        assert_eq!(leak_0x1000.size, 1024);
        assert!(leak_0x1000.stack_trace.is_some());

        let leak_0x2000 = leaks
            .potential_leaks
            .iter()
            .find(|l| l.ptr == 0x2000)
            .unwrap();
        assert_eq!(leak_0x2000.size, 2048);
        assert!(leak_0x2000.stack_trace.is_none());
    }

    #[test]
    fn test_leak_detection_with_deallocation() {
        let mut profiler = MemoryProfiler::new();
        profiler.enable();
        profiler.set_leak_detection_enabled(true);

        // Allocate memory
        profiler
            .record_allocation_with_trace(0x1000, 1024, None)
            .unwrap();
        profiler
            .record_allocation_with_trace(0x2000, 2048, None)
            .unwrap();

        // Deallocate one
        profiler.record_deallocation(0x1000).unwrap();

        // Check leaks
        let leaks = profiler.detect_leaks().unwrap();
        assert_eq!(leaks.leak_count, 1);
        assert_eq!(leaks.total_leaked_bytes, 2048);
        assert_eq!(leaks.potential_leaks[0].ptr, 0x2000);
    }

    #[test]
    fn test_leak_detection_disabled() {
        let mut profiler = MemoryProfiler::new();
        profiler.enable();
        // Don't enable leak detection

        assert!(!profiler.is_leak_detection_enabled());

        profiler
            .record_allocation_with_trace(0x1000, 1024, None)
            .unwrap();

        let leaks = profiler.detect_leaks().unwrap();
        assert_eq!(leaks.leak_count, 0);
        assert_eq!(leaks.total_leaked_bytes, 0);
    }

    #[test]
    fn test_get_largest_leaks() {
        let mut profiler = MemoryProfiler::new();
        profiler.enable();
        profiler.set_leak_detection_enabled(true);

        // Allocate different sizes
        profiler
            .record_allocation_with_trace(0x1000, 512, None)
            .unwrap();
        profiler
            .record_allocation_with_trace(0x2000, 2048, None)
            .unwrap();
        profiler
            .record_allocation_with_trace(0x3000, 1024, None)
            .unwrap();

        // Get largest 2 leaks
        let largest_leaks = profiler.get_largest_leaks(2).unwrap();
        assert_eq!(largest_leaks.leak_count, 2);
        assert_eq!(largest_leaks.potential_leaks[0].size, 2048); // Largest first
        assert_eq!(largest_leaks.potential_leaks[1].size, 1024); // Second largest
    }

    #[test]
    fn test_get_leaks_older_than() {
        let mut profiler = MemoryProfiler::new();
        profiler.enable();
        profiler.set_leak_detection_enabled(true);

        // Allocate memory
        profiler
            .record_allocation_with_trace(0x1000, 1024, None)
            .unwrap();

        // Wait a bit
        std::thread::sleep(std::time::Duration::from_millis(10));

        profiler
            .record_allocation_with_trace(0x2000, 2048, None)
            .unwrap();

        // Check for leaks older than 5ms
        let old_leaks = profiler
            .get_leaks_older_than(std::time::Duration::from_millis(5))
            .unwrap();
        assert_eq!(old_leaks.leak_count, 1);
        assert_eq!(old_leaks.potential_leaks[0].ptr, 0x1000);

        // Check for leaks older than 20ms (should be none)
        let very_old_leaks = profiler
            .get_leaks_older_than(std::time::Duration::from_millis(20))
            .unwrap();
        assert_eq!(very_old_leaks.leak_count, 0);
    }

    #[test]
    fn test_fragmentation_analysis() {
        let mut profiler = MemoryProfiler::new();
        profiler.enable();
        profiler.set_fragmentation_enabled(true);
        profiler.set_memory_pool_size(1024 * 1024); // 1MB pool

        // Allocate some memory with gaps
        profiler.record_allocation(0x1000, 1024).unwrap();
        profiler.record_allocation(0x3000, 2048).unwrap(); // Gap between 0x1400 and 0x3000
        profiler.record_allocation(0x4000, 512).unwrap();

        let analysis = profiler.analyze_fragmentation().unwrap();

        assert!(analysis.total_allocated > 0);
        assert!(analysis.fragmentation_ratio >= 0.0);
        assert!(!analysis.allocated_blocks.is_empty());
        assert!(!analysis.free_blocks.is_empty());
        assert!(analysis.external_fragmentation >= 0.0);
        assert!(analysis.internal_fragmentation >= 0.0);
    }

    #[test]
    fn test_fragmentation_analysis_disabled() {
        let mut profiler = MemoryProfiler::new();
        profiler.enable();
        // Don't enable fragmentation analysis

        let result = profiler.analyze_fragmentation();
        assert!(result.is_err());
    }

    #[test]
    fn test_timeline_tracking() {
        let mut profiler = MemoryProfiler::new();
        profiler.enable();
        profiler.set_timeline_enabled(true);

        // Record some memory operations
        profiler
            .record_allocation_with_trace(0x1000, 1024, None)
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));

        profiler
            .record_allocation_with_trace(0x2000, 2048, None)
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1));

        profiler.record_deallocation(0x1000).unwrap();

        let timeline = profiler.get_timeline_analysis().unwrap();

        assert_eq!(timeline.events.len(), 3);
        assert!(timeline.peak_usage_bytes > 0);
        assert!(timeline.allocation_rate >= 0.0);
        assert!(timeline.deallocation_rate >= 0.0);
        assert!(timeline.average_allocation_size > 0.0);
        assert!(!timeline.memory_usage_over_time.is_empty());
    }

    #[test]
    fn test_timeline_tracking_disabled() {
        let mut profiler = MemoryProfiler::new();
        profiler.enable();
        // Don't enable timeline tracking

        let result = profiler.get_timeline_analysis();
        assert!(result.is_err());
    }

    #[test]
    fn test_timeline_csv_export() {
        let mut profiler = MemoryProfiler::new();
        profiler.enable();
        profiler.set_timeline_enabled(true);

        // Record some memory operations
        profiler
            .record_allocation_with_trace(0x1000, 1024, None)
            .unwrap();
        profiler
            .record_allocation_with_trace(0x2000, 2048, None)
            .unwrap();
        profiler.record_deallocation(0x1000).unwrap();

        let result = profiler.export_timeline_csv("/tmp/test_timeline.csv");
        assert!(result.is_ok());

        // Clean up
        let _ = std::fs::remove_file("/tmp/test_timeline.csv");
    }

    #[test]
    fn test_memory_event_types() {
        use super::MemoryEventType;

        assert_eq!(MemoryEventType::Allocation, MemoryEventType::Allocation);
        assert_eq!(MemoryEventType::Deallocation, MemoryEventType::Deallocation);
        assert_ne!(MemoryEventType::Allocation, MemoryEventType::Deallocation);
    }

    #[test]
    fn test_timeline_configuration() {
        let mut profiler = MemoryProfiler::new();

        assert!(!profiler.is_timeline_enabled());
        assert!(!profiler.is_fragmentation_enabled());

        profiler.set_timeline_enabled(true);
        assert!(profiler.is_timeline_enabled());

        profiler.set_fragmentation_enabled(true);
        assert!(profiler.is_fragmentation_enabled());

        profiler.set_memory_pool_size(2 * 1024 * 1024); // 2MB
                                                        // Memory pool size is set (internal field, no getter for now)
    }
}
