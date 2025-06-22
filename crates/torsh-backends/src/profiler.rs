//! Performance profiling and monitoring

use crate::Device;
use torsh_core::error::Result;
use std::time::{Duration, Instant};

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec, boxed::Box};

/// Profiler interface for performance monitoring
pub trait Profiler: Send + Sync {
    /// Start profiling
    fn start(&mut self) -> Result<()>;
    
    /// Stop profiling
    fn stop(&mut self) -> Result<()>;
    
    /// Begin a profiling event
    fn begin_event(&mut self, name: &str) -> Result<EventId>;
    
    /// End a profiling event
    fn end_event(&mut self, event_id: EventId) -> Result<()>;
    
    /// Record a marker event
    fn marker(&mut self, name: &str) -> Result<()>;
    
    /// Get profiling statistics
    fn stats(&self) -> ProfilerStats;
    
    /// Get recorded events
    fn events(&self) -> &[ProfilerEvent];
    
    /// Clear recorded events
    fn clear(&mut self);
    
    /// Generate a report
    fn report(&self) -> String;
    
    /// Check if profiling is enabled
    fn is_enabled(&self) -> bool;
}

/// Event ID for tracking profiling events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventId(pub u64);

/// Profiling event
#[derive(Debug, Clone)]
pub struct ProfilerEvent {
    /// Event ID
    pub id: EventId,
    
    /// Event name
    pub name: String,
    
    /// Event type
    pub event_type: EventType,
    
    /// Start timestamp
    pub start_time: Instant,
    
    /// End timestamp (for duration events)
    pub end_time: Option<Instant>,
    
    /// Duration in nanoseconds
    pub duration_ns: Option<u64>,
    
    /// Device this event occurred on
    pub device: Option<Device>,
    
    /// Additional metadata
    pub metadata: Vec<(String, String)>,
}

impl ProfilerEvent {
    /// Create a new event
    pub fn new(id: EventId, name: String, event_type: EventType) -> Self {
        Self {
            id,
            name,
            event_type,
            start_time: Instant::now(),
            end_time: None,
            duration_ns: None,
            device: None,
            metadata: Vec::new(),
        }
    }
    
    /// Finish the event
    pub fn finish(&mut self) {
        let now = Instant::now();
        self.end_time = Some(now);
        self.duration_ns = Some(now.duration_since(self.start_time).as_nanos() as u64);
    }
    
    /// Get event duration
    pub fn duration(&self) -> Option<Duration> {
        self.duration_ns.map(Duration::from_nanos)
    }
    
    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.push((key, value));
    }
}

/// Event type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventType {
    /// Kernel execution
    KernelExecution,
    
    /// Memory operation (copy, allocation, etc.)
    MemoryOperation,
    
    /// Device synchronization
    Synchronization,
    
    /// API call
    ApiCall,
    
    /// Custom event
    Custom(String),
    
    /// Marker (instant event)
    Marker,
}

/// Profiler statistics
#[derive(Debug, Clone)]
pub struct ProfilerStats {
    /// Total number of events recorded
    pub total_events: usize,
    
    /// Total profiling time
    pub total_time: Duration,
    
    /// Number of kernel executions
    pub kernel_executions: usize,
    
    /// Total kernel execution time
    pub kernel_time: Duration,
    
    /// Number of memory operations
    pub memory_operations: usize,
    
    /// Total memory operation time
    pub memory_time: Duration,
    
    /// Average kernel execution time
    pub avg_kernel_time: Duration,
    
    /// Peak memory usage during profiling
    pub peak_memory_usage: usize,
    
    /// Number of synchronization events
    pub synchronization_events: usize,
    
    /// Profiling overhead estimate
    pub overhead_ns: u64,
}

impl Default for ProfilerStats {
    fn default() -> Self {
        Self {
            total_events: 0,
            total_time: Duration::ZERO,
            kernel_executions: 0,
            kernel_time: Duration::ZERO,
            memory_operations: 0,
            memory_time: Duration::ZERO,
            avg_kernel_time: Duration::ZERO,
            peak_memory_usage: 0,
            synchronization_events: 0,
            overhead_ns: 0,
        }
    }
}

/// Simple profiler implementation
pub struct SimpleProfiler {
    /// Whether profiling is enabled
    enabled: bool,
    
    /// Recorded events
    events: Vec<ProfilerEvent>,
    
    /// Next event ID
    next_event_id: u64,
    
    /// Profiling start time
    start_time: Option<Instant>,
    
    /// Statistics
    stats: ProfilerStats,
}

impl SimpleProfiler {
    /// Create a new simple profiler
    pub fn new() -> Self {
        Self {
            enabled: false,
            events: Vec::new(),
            next_event_id: 1,
            start_time: None,
            stats: ProfilerStats::default(),
        }
    }
    
    /// Generate next event ID
    fn next_id(&mut self) -> EventId {
        let id = EventId(self.next_event_id);
        self.next_event_id += 1;
        id
    }
    
    /// Update statistics
    fn update_stats(&mut self) {
        self.stats.total_events = self.events.len();
        
        let mut kernel_times = Vec::new();
        let mut memory_times = Vec::new();
        
        for event in &self.events {
            if let Some(duration) = event.duration() {
                match event.event_type {
                    EventType::KernelExecution => {
                        self.stats.kernel_executions += 1;
                        self.stats.kernel_time += duration;
                        kernel_times.push(duration);
                    }
                    EventType::MemoryOperation => {
                        self.stats.memory_operations += 1;
                        self.stats.memory_time += duration;
                        memory_times.push(duration);
                    }
                    EventType::Synchronization => {
                        self.stats.synchronization_events += 1;
                    }
                    _ => {}
                }
            }
        }
        
        if !kernel_times.is_empty() {
            let total_ns: u64 = kernel_times.iter().map(|d| d.as_nanos() as u64).sum();
            self.stats.avg_kernel_time = Duration::from_nanos(total_ns / kernel_times.len() as u64);
        }
        
        if let Some(start) = self.start_time {
            self.stats.total_time = Instant::now().duration_since(start);
        }
    }
}

impl Default for SimpleProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Profiler for SimpleProfiler {
    fn start(&mut self) -> Result<()> {
        self.enabled = true;
        self.start_time = Some(Instant::now());
        self.events.clear();
        self.stats = ProfilerStats::default();
        Ok(())
    }
    
    fn stop(&mut self) -> Result<()> {
        self.enabled = false;
        self.update_stats();
        Ok(())
    }
    
    fn begin_event(&mut self, name: &str) -> Result<EventId> {
        if !self.enabled {
            return Ok(EventId(0));
        }
        
        let id = self.next_id();
        let event = ProfilerEvent::new(id, name.to_string(), EventType::Custom(name.to_string()));
        self.events.push(event);
        Ok(id)
    }
    
    fn end_event(&mut self, event_id: EventId) -> Result<()> {
        if !self.enabled || event_id.0 == 0 {
            return Ok(());
        }
        
        if let Some(event) = self.events.iter_mut().find(|e| e.id == event_id) {
            event.finish();
        }
        Ok(())
    }
    
    fn marker(&mut self, name: &str) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        
        let id = self.next_id();
        let mut event = ProfilerEvent::new(id, name.to_string(), EventType::Marker);
        event.finish();
        self.events.push(event);
        Ok(())
    }
    
    fn stats(&self) -> ProfilerStats {
        self.stats.clone()
    }
    
    fn events(&self) -> &[ProfilerEvent] {
        &self.events
    }
    
    fn clear(&mut self) {
        self.events.clear();
        self.stats = ProfilerStats::default();
    }
    
    fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Profiler Report ===\n");
        report.push_str(&format!("Total Events: {}\n", self.stats.total_events));
        report.push_str(&format!("Total Time: {:.2}ms\n", self.stats.total_time.as_secs_f64() * 1000.0));
        report.push_str(&format!("Kernel Executions: {}\n", self.stats.kernel_executions));
        report.push_str(&format!("Kernel Time: {:.2}ms\n", self.stats.kernel_time.as_secs_f64() * 1000.0));
        report.push_str(&format!("Memory Operations: {}\n", self.stats.memory_operations));
        report.push_str(&format!("Memory Time: {:.2}ms\n", self.stats.memory_time.as_secs_f64() * 1000.0));
        report.push_str(&format!("Avg Kernel Time: {:.2}μs\n", self.stats.avg_kernel_time.as_secs_f64() * 1_000_000.0));
        
        report.push_str("\n=== Events ===\n");
        for event in &self.events {
            if let Some(duration) = event.duration() {
                report.push_str(&format!(
                    "{}: {:.2}μs\n",
                    event.name,
                    duration.as_secs_f64() * 1_000_000.0
                ));
            } else {
                report.push_str(&format!("{}: (marker)\n", event.name));
            }
        }
        
        report
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

/// Scoped profiler event that automatically ends when dropped
pub struct ScopedEvent<'a> {
    profiler: &'a mut dyn Profiler,
    event_id: EventId,
}

impl<'a> ScopedEvent<'a> {
    /// Create a new scoped event
    pub fn new(profiler: &'a mut dyn Profiler, name: &str) -> Result<Self> {
        let event_id = profiler.begin_event(name)?;
        Ok(Self { profiler, event_id })
    }
}

impl Drop for ScopedEvent<'_> {
    fn drop(&mut self) {
        let _ = self.profiler.end_event(self.event_id);
    }
}

/// Macro for creating scoped profiling events
#[macro_export]
macro_rules! profile_scope {
    ($profiler:expr, $name:expr) => {
        let _scoped_event = $crate::profiler::ScopedEvent::new($profiler, $name)?;
    };
}

/// Profiler configuration
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Whether to enable profiling by default
    pub enabled: bool,
    
    /// Maximum number of events to store
    pub max_events: Option<usize>,
    
    /// Whether to collect detailed timing information
    pub detailed_timing: bool,
    
    /// Whether to track memory usage
    pub track_memory: bool,
    
    /// Event types to profile
    pub event_types: Vec<EventType>,
    
    /// Output format for reports
    pub output_format: OutputFormat,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_events: Some(10000),
            detailed_timing: true,
            track_memory: false,
            event_types: vec![
                EventType::KernelExecution,
                EventType::MemoryOperation,
                EventType::Synchronization,
            ],
            output_format: OutputFormat::Text,
        }
    }
}

/// Output format for profiler reports
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputFormat {
    /// Plain text
    Text,
    
    /// JSON format
    Json,
    
    /// Chrome tracing format
    ChromeTracing,
    
    /// CSV format
    Csv,
}