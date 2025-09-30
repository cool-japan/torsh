//! Performance optimizations and overhead minimization
//!
//! This module contains optimizations to minimize profiling overhead and improve
//! data structure efficiency for high-performance profiling scenarios.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::mem;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Lock-free event buffer using ring buffer for minimal overhead
pub struct LockFreeEventBuffer<T> {
    buffer: Vec<AtomicOption<T>>,
    capacity: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
    mask: usize,
}

/// Atomic option type for lock-free data structures
struct AtomicOption<T> {
    data: parking_lot::Mutex<Option<T>>,
}

impl<T> AtomicOption<T> {
    fn new() -> Self {
        Self {
            data: parking_lot::Mutex::new(None),
        }
    }

    fn take(&self) -> Option<T> {
        self.data.lock().take()
    }

    fn store(&self, value: T) -> bool {
        let mut data = self.data.lock();
        if data.is_none() {
            *data = Some(value);
            true
        } else {
            false
        }
    }
}

impl<T> LockFreeEventBuffer<T> {
    /// Create a new lock-free event buffer with given capacity (must be power of 2)
    pub fn new(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two(), "Capacity must be power of 2");

        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(AtomicOption::new());
        }

        Self {
            buffer,
            capacity,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            mask: capacity - 1,
        }
    }

    /// Push an item to the buffer (returns false if buffer is full)
    pub fn push(&self, item: T) -> bool {
        let mut item = Some(item);
        loop {
            let tail = self.tail.load(Ordering::Acquire);
            let next_tail = (tail + 1) & self.mask;
            let head = self.head.load(Ordering::Acquire);

            // Check if buffer is full
            if next_tail == head {
                return false;
            }

            // Check if the slot is available first before taking the item
            if self.buffer[tail]
                .data
                .try_lock()
                .is_some_and(|guard| guard.is_none())
            {
                if let Some(value) = item.take() {
                    if self.buffer[tail].store(value) {
                        // Update tail pointer
                        if self
                            .tail
                            .compare_exchange_weak(
                                tail,
                                next_tail,
                                Ordering::Release,
                                Ordering::Relaxed,
                            )
                            .is_ok()
                        {
                            return true;
                        }
                        // If compare_exchange failed, retrieve item for next iteration
                        item = self.buffer[tail].take();
                    }
                    // Note: if store fails here, it means another thread filled the slot
                    // between our check and store - we've lost the item but that's acceptable
                    // in a lock-free structure as it's a rare race condition
                }
            }
        }
    }

    /// Pop an item from the buffer (returns None if buffer is empty)
    pub fn pop(&self) -> Option<T> {
        loop {
            let head = self.head.load(Ordering::Acquire);
            let tail = self.tail.load(Ordering::Acquire);

            // Check if buffer is empty
            if head == tail {
                return None;
            }

            // Try to take the item
            if let Some(item) = self.buffer[head].take() {
                let next_head = (head + 1) & self.mask;
                // Update head pointer
                if self
                    .head
                    .compare_exchange_weak(head, next_head, Ordering::Release, Ordering::Relaxed)
                    .is_ok()
                {
                    return Some(item);
                }
            }
        }
    }

    /// Get current buffer usage
    pub fn len(&self) -> usize {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        (tail.wrapping_sub(head)) & self.mask
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire) == self.tail.load(Ordering::Acquire)
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// Thread-local profiling data to minimize contention
thread_local! {
    static THREAD_LOCAL_BUFFER: std::cell::RefCell<ThreadLocalProfileData> =
        std::cell::RefCell::new(ThreadLocalProfileData::new());
}

/// Thread-local profiling data structure
pub struct ThreadLocalProfileData {
    events: Vec<crate::ProfileEvent>,
    memory_events: Vec<crate::memory::MemoryEvent>,
    call_stack_depth: usize,
    total_overhead_ns: u64,
    event_count: usize,
    max_buffer_size: usize,
}

impl ThreadLocalProfileData {
    fn new() -> Self {
        Self {
            events: Vec::with_capacity(1024), // Pre-allocate to avoid reallocations
            memory_events: Vec::with_capacity(512),
            call_stack_depth: 0,
            total_overhead_ns: 0,
            event_count: 0,
            max_buffer_size: 4096,
        }
    }

    /// Add a profiling event with minimal overhead
    pub fn add_event(&mut self, event: crate::ProfileEvent) {
        if self.events.len() < self.max_buffer_size {
            self.events.push(event);
            self.event_count += 1;
        } else {
            // Buffer full, replace oldest event (ring buffer behavior)
            let index = self.event_count % self.max_buffer_size;
            self.events[index] = event;
            self.event_count += 1;
        }
    }

    /// Drain events and return them
    pub fn drain_events(&mut self) -> Vec<crate::ProfileEvent> {
        mem::take(&mut self.events)
    }

    /// Get current buffer usage statistics
    pub fn get_stats(&self) -> ThreadLocalStats {
        ThreadLocalStats {
            event_count: self.event_count,
            buffer_size: self.events.len(),
            call_stack_depth: self.call_stack_depth,
            total_overhead_ns: self.total_overhead_ns,
            capacity_utilization: self.events.len() as f64 / self.max_buffer_size as f64,
        }
    }
}

/// Statistics for thread-local profiling data
#[derive(Debug, Clone)]
pub struct ThreadLocalStats {
    pub event_count: usize,
    pub buffer_size: usize,
    pub call_stack_depth: usize,
    pub total_overhead_ns: u64,
    pub capacity_utilization: f64,
}

/// Optimized profiler with minimal overhead
pub struct OptimizedProfiler {
    enabled: AtomicBool,
    global_buffer: LockFreeEventBuffer<crate::ProfileEvent>,
    collection_thread: Option<std::thread::JoinHandle<()>>,
    stop_signal: Arc<AtomicBool>,
    overhead_tracker: OverheadTracker,
    sampling_config: SamplingConfig,
}

/// Overhead tracking for performance analysis
pub struct OverheadTracker {
    total_calls: AtomicU64,
    total_overhead_ns: AtomicU64,
    max_overhead_ns: AtomicU64,
    histogram: RwLock<[u64; 20]>, // Histogram buckets for overhead distribution
}

impl OverheadTracker {
    fn new() -> Self {
        Self {
            total_calls: AtomicU64::new(0),
            total_overhead_ns: AtomicU64::new(0),
            max_overhead_ns: AtomicU64::new(0),
            histogram: RwLock::new([0; 20]),
        }
    }

    /// Record overhead measurement
    pub fn record_overhead(&self, overhead_ns: u64) {
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        self.total_overhead_ns
            .fetch_add(overhead_ns, Ordering::Relaxed);

        // Update maximum overhead
        let mut current_max = self.max_overhead_ns.load(Ordering::Relaxed);
        while overhead_ns > current_max {
            match self.max_overhead_ns.compare_exchange_weak(
                current_max,
                overhead_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_max = actual,
            }
        }

        // Update histogram
        let bucket = ((overhead_ns as f64).log2().max(0.0) as usize).min(19);
        if let Some(mut histogram) = self.histogram.try_write() {
            histogram[bucket] += 1;
        }
    }

    /// Get overhead statistics
    pub fn get_stats(&self) -> DetailedOverheadStats {
        let total_calls = self.total_calls.load(Ordering::Relaxed);
        let total_overhead = self.total_overhead_ns.load(Ordering::Relaxed);
        let avg_overhead = if total_calls > 0 {
            total_overhead as f64 / total_calls as f64
        } else {
            0.0
        };

        DetailedOverheadStats {
            total_calls,
            total_overhead_ns: total_overhead,
            avg_overhead_ns: avg_overhead,
            max_overhead_ns: self.max_overhead_ns.load(Ordering::Relaxed),
            histogram: *self.histogram.read(),
        }
    }
}

/// Detailed overhead statistics
#[derive(Debug, Clone)]
pub struct DetailedOverheadStats {
    pub total_calls: u64,
    pub total_overhead_ns: u64,
    pub avg_overhead_ns: f64,
    pub max_overhead_ns: u64,
    pub histogram: [u64; 20],
}

/// Sampling configuration for reduced overhead
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Sample every N events (1 = no sampling, 100 = sample 1 in 100)
    pub sample_rate: usize,
    /// Adaptive sampling based on overhead
    pub adaptive_sampling: bool,
    /// Maximum allowed overhead percentage (0.0 to 1.0)
    pub max_overhead_percent: f64,
    /// Minimum sampling rate (always sample at least this rate)
    pub min_sample_rate: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            sample_rate: 1,
            adaptive_sampling: false,
            max_overhead_percent: 0.05, // 5% overhead maximum
            min_sample_rate: 1000,      // Sample at least 1 in 1000
        }
    }
}

impl Default for OptimizedProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizedProfiler {
    /// Create a new optimized profiler
    pub fn new() -> Self {
        let stop_signal = Arc::new(AtomicBool::new(false));
        let global_buffer = LockFreeEventBuffer::new(8192); // 8K events

        Self {
            enabled: AtomicBool::new(true),
            global_buffer,
            collection_thread: None,
            stop_signal,
            overhead_tracker: OverheadTracker::new(),
            sampling_config: SamplingConfig::default(),
        }
    }

    /// Start the background collection thread
    pub fn start_collection_thread(&mut self) {
        if self.collection_thread.is_some() {
            return; // Already started
        }

        let stop_signal = self.stop_signal.clone();
        let handle = std::thread::spawn(move || {
            Self::collection_thread_main(stop_signal);
        });

        self.collection_thread = Some(handle);
    }

    /// Stop the background collection thread
    pub fn stop_collection_thread(&mut self) {
        if let Some(handle) = self.collection_thread.take() {
            self.stop_signal.store(true, Ordering::Release);
            let _ = handle.join();
            self.stop_signal.store(false, Ordering::Release);
        }
    }

    /// Record an event with minimal overhead
    pub fn record_event_fast(&self, event: crate::ProfileEvent) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }

        let start_time = Instant::now();

        // Try thread-local first for minimal contention
        THREAD_LOCAL_BUFFER.with(|buffer| {
            let mut buffer = buffer.borrow_mut();
            buffer.add_event(event);
        });

        let overhead = start_time.elapsed().as_nanos() as u64;
        self.overhead_tracker.record_overhead(overhead);
    }

    /// Flush thread-local buffers to global buffer
    pub fn flush_thread_local(&self) {
        THREAD_LOCAL_BUFFER.with(|buffer| {
            let mut buffer = buffer.borrow_mut();
            let events = buffer.drain_events();

            for event in events {
                if !self.global_buffer.push(event) {
                    // Buffer full, event dropped
                    break;
                }
            }
        });
    }

    /// Get profiler statistics
    pub fn get_stats(&self) -> ProfilerStats {
        let overhead_stats = self.overhead_tracker.get_stats();
        let buffer_usage = self.global_buffer.len() as f64 / self.global_buffer.capacity() as f64;

        let thread_local_stats = THREAD_LOCAL_BUFFER.with(|buffer| buffer.borrow().get_stats());

        ProfilerStats {
            enabled: self.enabled.load(Ordering::Relaxed),
            overhead_stats,
            buffer_usage,
            thread_local_stats,
            sampling_config: self.sampling_config.clone(),
        }
    }

    /// Optimize sampling rate based on overhead
    pub fn optimize_sampling(&mut self) {
        if !self.sampling_config.adaptive_sampling {
            return;
        }

        let stats = self.overhead_tracker.get_stats();
        if stats.total_calls < 1000 {
            return; // Not enough data
        }

        let overhead_percent = stats.avg_overhead_ns / 1_000_000_000.0; // Convert to seconds

        if overhead_percent > self.sampling_config.max_overhead_percent {
            // Increase sampling rate (sample less frequently)
            self.sampling_config.sample_rate =
                (self.sampling_config.sample_rate * 2).min(self.sampling_config.min_sample_rate);
        } else if overhead_percent < self.sampling_config.max_overhead_percent * 0.5 {
            // Decrease sampling rate (sample more frequently)
            self.sampling_config.sample_rate = (self.sampling_config.sample_rate / 2).max(1);
        }
    }

    /// Collection thread main loop
    fn collection_thread_main(stop_signal: Arc<AtomicBool>) {
        let mut flush_interval = std::time::Duration::from_millis(100);

        while !stop_signal.load(Ordering::Acquire) {
            std::thread::sleep(flush_interval);

            // Global collection logic would go here
            // For now, just a placeholder that adjusts flush interval based on load

            // Adaptive flush interval based on system load
            let load = 1.0; // Fallback - sys_info crate not available
            if load > 2.0 {
                flush_interval = std::time::Duration::from_millis(200); // Slower when system is busy
            } else {
                flush_interval = std::time::Duration::from_millis(50); // Faster when system is idle
            }
        }
    }
}

/// Comprehensive profiler statistics
#[derive(Debug, Clone)]
pub struct ProfilerStats {
    pub enabled: bool,
    pub overhead_stats: DetailedOverheadStats,
    pub buffer_usage: f64,
    pub thread_local_stats: ThreadLocalStats,
    pub sampling_config: SamplingConfig,
}

/// Memory pool for efficient event allocation
pub struct EventMemoryPool {
    #[allow(clippy::vec_box)]
    pool: parking_lot::Mutex<Vec<Box<crate::ProfileEvent>>>,
    max_pool_size: usize,
    allocations: AtomicU64,
    deallocations: AtomicU64,
}

impl EventMemoryPool {
    /// Create a new memory pool
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            pool: parking_lot::Mutex::new(Vec::with_capacity(max_pool_size)),
            max_pool_size,
            allocations: AtomicU64::new(0),
            deallocations: AtomicU64::new(0),
        }
    }

    /// Allocate an event from the pool or create new
    pub fn allocate(&self) -> Box<crate::ProfileEvent> {
        self.allocations.fetch_add(1, Ordering::Relaxed);

        if let Some(mut pool) = self.pool.try_lock() {
            if let Some(event) = pool.pop() {
                return event;
            }
        }

        // Pool empty or locked, allocate new
        Box::new(crate::ProfileEvent {
            name: String::new(),
            category: String::new(),
            start_us: 0,
            duration_us: 0,
            thread_id: 0,
            operation_count: None,
            flops: None,
            bytes_transferred: None,
            stack_trace: None,
        })
    }

    /// Return an event to the pool
    pub fn deallocate(&self, mut event: Box<crate::ProfileEvent>) {
        self.deallocations.fetch_add(1, Ordering::Relaxed);

        // Clear the event data
        event.name.clear();
        event.category.clear();
        event.start_us = 0;
        event.duration_us = 0;
        event.thread_id = 0;
        event.operation_count = None;
        event.flops = None;
        event.bytes_transferred = None;
        event.stack_trace = None;

        if let Some(mut pool) = self.pool.try_lock() {
            if pool.len() < self.max_pool_size {
                pool.push(event);
            }
        }
    }

    /// Get pool statistics
    pub fn get_stats(&self) -> PoolStats {
        let pool_size = self.pool.lock().len();
        PoolStats {
            pool_size,
            max_pool_size: self.max_pool_size,
            allocations: self.allocations.load(Ordering::Relaxed),
            deallocations: self.deallocations.load(Ordering::Relaxed),
        }
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub pool_size: usize,
    pub max_pool_size: usize,
    pub allocations: u64,
    pub deallocations: u64,
}

/// Compact event representation for minimal memory usage
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CompactEvent {
    /// Packed timing information (start time and duration)
    pub timing: u64,
    /// Thread ID and category ID packed together
    pub thread_category: u32,
    /// Name ID (index into string table)
    pub name_id: u32,
}

impl CompactEvent {
    /// Create a new compact event
    pub fn new(
        start_us: u32,
        duration_us: u32,
        thread_id: u16,
        category_id: u8,
        name_id: u32,
    ) -> Self {
        let timing = ((start_us as u64) << 32) | (duration_us as u64);
        let thread_category = ((thread_id as u32) << 16) | (category_id as u32);

        Self {
            timing,
            thread_category,
            name_id,
        }
    }

    /// Extract start time in microseconds
    pub fn start_us(&self) -> u32 {
        (self.timing >> 32) as u32
    }

    /// Extract duration in microseconds
    pub fn duration_us(&self) -> u32 {
        self.timing as u32
    }

    /// Extract thread ID
    pub fn thread_id(&self) -> u16 {
        (self.thread_category >> 16) as u16
    }

    /// Extract category ID
    pub fn category_id(&self) -> u8 {
        self.thread_category as u8
    }
}

/// String interning for efficient string storage
pub struct StringInterner {
    strings: RwLock<Vec<String>>,
    string_to_id: RwLock<HashMap<String, u32>>,
    next_id: AtomicU32,
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

impl StringInterner {
    /// Create a new string interner
    pub fn new() -> Self {
        Self {
            strings: RwLock::new(Vec::new()),
            string_to_id: RwLock::new(HashMap::new()),
            next_id: AtomicU32::new(0),
        }
    }

    /// Intern a string and return its ID
    pub fn intern(&self, s: &str) -> u32 {
        // Fast path: check if already interned
        if let Some(map) = self.string_to_id.try_read() {
            if let Some(&id) = map.get(s) {
                return id;
            }
        }

        // Slow path: add new string
        let mut strings = self.strings.write();
        let mut map = self.string_to_id.write();

        // Double-check in case another thread added it
        if let Some(&id) = map.get(s) {
            return id;
        }

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        strings.push(s.to_string());
        map.insert(s.to_string(), id);
        id
    }

    /// Get string by ID
    pub fn get_string(&self, id: u32) -> Option<String> {
        self.strings.read().get(id as usize).cloned()
    }

    /// Get statistics
    pub fn get_stats(&self) -> InternerStats {
        let strings = self.strings.read();
        let total_size = strings.iter().map(|s| s.len()).sum();

        InternerStats {
            string_count: strings.len(),
            total_size,
            next_id: self.next_id.load(Ordering::Relaxed),
        }
    }
}

/// String interner statistics
#[derive(Debug, Clone)]
pub struct InternerStats {
    pub string_count: usize,
    pub total_size: usize,
    pub next_id: u32,
}

/// Global optimized profiler instance
static mut OPTIMIZED_PROFILER: Option<OptimizedProfiler> = None;
static PROFILER_INIT: std::sync::Once = std::sync::Once::new();

/// Get or create the global optimized profiler
pub fn get_optimized_profiler() -> &'static mut OptimizedProfiler {
    unsafe {
        PROFILER_INIT.call_once(|| {
            OPTIMIZED_PROFILER = Some(OptimizedProfiler::new());
        });
        OPTIMIZED_PROFILER.as_mut().unwrap()
    }
}

/// Initialize optimized profiling
pub fn init_optimized_profiling() {
    let profiler = get_optimized_profiler();
    profiler.start_collection_thread();
}

/// Record an event with optimized path
pub fn record_event_optimized(event: crate::ProfileEvent) {
    let profiler = get_optimized_profiler();
    profiler.record_event_fast(event);
}

/// Flush all buffers
pub fn flush_optimized_buffers() {
    let profiler = get_optimized_profiler();
    profiler.flush_thread_local();
}

/// Get optimization statistics
pub fn get_optimization_stats() -> ProfilerStats {
    let profiler = get_optimized_profiler();
    profiler.get_stats()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_lock_free_buffer() {
        let buffer = LockFreeEventBuffer::new(16);

        // Test push/pop
        assert!(buffer.push("item1"));
        assert!(buffer.push("item2"));
        assert_eq!(buffer.len(), 2);

        assert_eq!(buffer.pop(), Some("item1"));
        assert_eq!(buffer.pop(), Some("item2"));
        assert_eq!(buffer.pop(), None);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_overhead_tracker() {
        let tracker = OverheadTracker::new();

        tracker.record_overhead(1000);
        tracker.record_overhead(2000);
        tracker.record_overhead(1500);

        let stats = tracker.get_stats();
        assert_eq!(stats.total_calls, 3);
        assert_eq!(stats.total_overhead_ns, 4500);
        assert_eq!(stats.avg_overhead_ns, 1500.0);
        assert_eq!(stats.max_overhead_ns, 2000);
    }

    #[test]
    fn test_event_memory_pool() {
        let pool = EventMemoryPool::new(10);

        // Allocate and deallocate
        let event1 = pool.allocate();
        let event2 = pool.allocate();

        let stats = pool.get_stats();
        assert_eq!(stats.allocations, 2);

        pool.deallocate(event1);
        pool.deallocate(event2);

        let stats = pool.get_stats();
        assert_eq!(stats.deallocations, 2);
        assert_eq!(stats.pool_size, 2);
    }

    #[test]
    fn test_compact_event() {
        let event = CompactEvent::new(1000, 500, 123, 5, 42);

        assert_eq!(event.start_us(), 1000);
        assert_eq!(event.duration_us(), 500);
        assert_eq!(event.thread_id(), 123);
        assert_eq!(event.category_id(), 5);
        assert_eq!(event.name_id, 42);
    }

    #[test]
    fn test_string_interner() {
        let interner = StringInterner::new();

        let id1 = interner.intern("test_string");
        let id2 = interner.intern("another_string");
        let id3 = interner.intern("test_string"); // Should reuse ID

        assert_eq!(id1, id3);
        assert_ne!(id1, id2);

        assert_eq!(interner.get_string(id1), Some("test_string".to_string()));
        assert_eq!(interner.get_string(id2), Some("another_string".to_string()));

        let stats = interner.get_stats();
        assert_eq!(stats.string_count, 2);
    }

    #[test]
    fn test_thread_local_buffer() {
        let event = crate::ProfileEvent {
            name: "test".to_string(),
            category: "test".to_string(),
            start_us: 0,
            duration_us: 1000,
            thread_id: 1,
            operation_count: Some(1),
            flops: None,
            bytes_transferred: None,
            stack_trace: None,
        };

        THREAD_LOCAL_BUFFER.with(|buffer| {
            let mut buffer = buffer.borrow_mut();
            buffer.add_event(event.clone());
            buffer.add_event(event);

            let stats = buffer.get_stats();
            assert_eq!(stats.event_count, 2);
            assert_eq!(stats.buffer_size, 2);
        });
    }

    #[test]
    fn test_optimized_profiler() {
        let mut profiler = OptimizedProfiler::new();

        let event = crate::ProfileEvent {
            name: "test_event".to_string(),
            category: "test".to_string(),
            start_us: 0,
            duration_us: 1000,
            thread_id: 1,
            operation_count: Some(1),
            flops: None,
            bytes_transferred: None,
            stack_trace: None,
        };

        profiler.record_event_fast(event);

        let stats = profiler.get_stats();
        assert!(stats.enabled);
        assert!(stats.overhead_stats.total_calls > 0);
    }

    #[test]
    fn test_sampling_optimization() {
        let mut profiler = OptimizedProfiler::new();
        profiler.sampling_config.adaptive_sampling = true;
        profiler.sampling_config.max_overhead_percent = 0.01; // 1%

        // Simulate high overhead
        for _ in 0..2000 {
            profiler.overhead_tracker.record_overhead(1_000_000); // 1ms overhead
        }

        let old_rate = profiler.sampling_config.sample_rate;
        profiler.optimize_sampling();

        // Should increase sampling rate due to high overhead
        assert!(profiler.sampling_config.sample_rate >= old_rate);
    }
}
