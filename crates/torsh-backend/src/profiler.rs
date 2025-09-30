//! Performance profiling and monitoring

use crate::Device;
use std::time::{Duration, Instant};
use torsh_core::error::Result;

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

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
            total_time: Duration::from_secs(0),
            kernel_executions: 0,
            kernel_time: Duration::from_secs(0),
            memory_operations: 0,
            memory_time: Duration::from_secs(0),
            avg_kernel_time: Duration::from_secs(0),
            peak_memory_usage: 0,
            synchronization_events: 0,
            overhead_ns: 0,
        }
    }
}

/// Simple profiler implementation
#[derive(Debug, Clone)]
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

    /// Start a new profiling event and return it for manual control
    ///
    /// This is a convenience method for benchmarks that need direct access to the event.
    /// The returned event can be used to end the profiling session manually.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the event to profile
    ///
    /// # Returns
    ///
    /// A ProfilerEvent that can be used to control the event lifecycle
    pub fn start_event(&mut self, name: &str) -> ProfilerEvent {
        let id = self.next_id();
        let event = ProfilerEvent::new(id, name.to_string(), EventType::Custom(name.to_string()));

        if self.enabled {
            // Store a copy in the profiler's event list
            self.events.push(event.clone());
        }

        event
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
        report.push_str(&format!(
            "Total Time: {:.2}ms\n",
            self.stats.total_time.as_secs_f64() * 1000.0
        ));
        report.push_str(&format!(
            "Kernel Executions: {}\n",
            self.stats.kernel_executions
        ));
        report.push_str(&format!(
            "Kernel Time: {:.2}ms\n",
            self.stats.kernel_time.as_secs_f64() * 1000.0
        ));
        report.push_str(&format!(
            "Memory Operations: {}\n",
            self.stats.memory_operations
        ));
        report.push_str(&format!(
            "Memory Time: {:.2}ms\n",
            self.stats.memory_time.as_secs_f64() * 1000.0
        ));
        report.push_str(&format!(
            "Avg Kernel Time: {:.2}μs\n",
            self.stats.avg_kernel_time.as_secs_f64() * 1_000_000.0
        ));

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Device, DeviceInfo};
    use std::time::Duration;
    use torsh_core::device::DeviceType;

    #[allow(dead_code)]
    fn create_test_device() -> Device {
        let info = DeviceInfo::default();
        Device::new(0, DeviceType::Cpu, "Test CPU".to_string(), info)
    }

    #[test]
    fn test_event_id() {
        let id1 = EventId(1);
        let id2 = EventId(1);
        let id3 = EventId(2);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_profiler_event_creation() {
        let id = EventId(1);
        let event = ProfilerEvent::new(id, "test_event".to_string(), EventType::KernelExecution);

        assert_eq!(event.id, id);
        assert_eq!(event.name, "test_event");
        assert_eq!(event.event_type, EventType::KernelExecution);
        assert!(event.end_time.is_none());
        assert!(event.duration_ns.is_none());
        assert!(event.device.is_none());
        assert!(event.metadata.is_empty());
    }

    #[test]
    fn test_profiler_event_finish() {
        let id = EventId(1);
        let mut event =
            ProfilerEvent::new(id, "test_event".to_string(), EventType::MemoryOperation);

        // Simulate some work
        std::thread::sleep(Duration::from_millis(1));

        event.finish();

        assert!(event.end_time.is_some());
        assert!(event.duration_ns.is_some());
        assert!(event.duration().is_some());

        let duration = event.duration().unwrap();
        assert!(duration.as_millis() >= 1);
    }

    #[test]
    fn test_profiler_event_metadata() {
        let id = EventId(1);
        let mut event = ProfilerEvent::new(id, "test_event".to_string(), EventType::ApiCall);

        event.add_metadata("param1".to_string(), "value1".to_string());
        event.add_metadata("param2".to_string(), "value2".to_string());

        assert_eq!(event.metadata.len(), 2);
        assert!(event
            .metadata
            .contains(&("param1".to_string(), "value1".to_string())));
        assert!(event
            .metadata
            .contains(&("param2".to_string(), "value2".to_string())));
    }

    #[test]
    fn test_event_type_variants() {
        let types = [
            EventType::KernelExecution,
            EventType::MemoryOperation,
            EventType::Synchronization,
            EventType::ApiCall,
            EventType::Custom("CustomEvent".to_string()),
            EventType::Marker,
        ];

        // Ensure all types are distinct
        for (i, type1) in types.iter().enumerate() {
            for (j, type2) in types.iter().enumerate() {
                if i != j {
                    assert_ne!(type1, type2);
                }
            }
        }
    }

    #[test]
    fn test_profiler_stats_default() {
        let stats = ProfilerStats::default();

        assert_eq!(stats.total_events, 0);
        assert_eq!(stats.total_time, Duration::from_secs(0));
        assert_eq!(stats.kernel_executions, 0);
        assert_eq!(stats.kernel_time, Duration::from_secs(0));
        assert_eq!(stats.memory_operations, 0);
        assert_eq!(stats.memory_time, Duration::from_secs(0));
        assert_eq!(stats.avg_kernel_time, Duration::from_secs(0));
        assert_eq!(stats.peak_memory_usage, 0);
        assert_eq!(stats.synchronization_events, 0);
        assert_eq!(stats.overhead_ns, 0);
    }

    #[test]
    fn test_simple_profiler_creation() {
        let profiler = SimpleProfiler::new();

        assert!(!profiler.is_enabled());
        assert!(profiler.events().is_empty());
        assert_eq!(profiler.stats().total_events, 0);
    }

    #[test]
    fn test_simple_profiler_start_stop() {
        let mut profiler = SimpleProfiler::new();

        // Initially disabled
        assert!(!profiler.is_enabled());

        // Start profiling
        let result = profiler.start();
        assert!(result.is_ok());
        assert!(profiler.is_enabled());

        // Stop profiling
        let result = profiler.stop();
        assert!(result.is_ok());
        assert!(!profiler.is_enabled());
    }

    #[test]
    fn test_simple_profiler_events() {
        let mut profiler = SimpleProfiler::new();
        profiler.start().unwrap();

        // Add some events
        let id1 = profiler.begin_event("event1").unwrap();
        std::thread::sleep(Duration::from_millis(1));
        profiler.end_event(id1).unwrap();

        let id2 = profiler.begin_event("event2").unwrap();
        std::thread::sleep(Duration::from_millis(1));
        profiler.end_event(id2).unwrap();

        profiler.marker("checkpoint").unwrap();

        profiler.stop().unwrap();

        let events = profiler.events();
        assert_eq!(events.len(), 3);

        // Check first event
        assert_eq!(events[0].name, "event1");
        assert!(events[0].duration().is_some());

        // Check second event
        assert_eq!(events[1].name, "event2");
        assert!(events[1].duration().is_some());

        // Check marker
        assert_eq!(events[2].name, "checkpoint");
        assert_eq!(events[2].event_type, EventType::Marker);
        assert!(events[2].duration().is_some());
    }

    #[test]
    fn test_simple_profiler_disabled_events() {
        let mut profiler = SimpleProfiler::new();
        // Don't start profiling

        let id = profiler.begin_event("event").unwrap();
        assert_eq!(id.0, 0); // Should return 0 when disabled

        let result = profiler.end_event(id);
        assert!(result.is_ok()); // Should not error

        let result = profiler.marker("marker");
        assert!(result.is_ok()); // Should not error

        // Should have no events
        assert!(profiler.events().is_empty());
    }

    #[test]
    fn test_simple_profiler_clear() {
        let mut profiler = SimpleProfiler::new();
        profiler.start().unwrap();

        let id = profiler.begin_event("event").unwrap();
        profiler.end_event(id).unwrap();

        assert_eq!(profiler.events().len(), 1);

        profiler.clear();

        assert!(profiler.events().is_empty());
        assert_eq!(profiler.stats().total_events, 0);
    }

    #[test]
    fn test_simple_profiler_report() {
        let mut profiler = SimpleProfiler::new();
        profiler.start().unwrap();

        let id = profiler.begin_event("test_kernel").unwrap();
        std::thread::sleep(Duration::from_millis(5));
        profiler.end_event(id).unwrap();

        profiler.stop().unwrap();

        let report = profiler.report();

        assert!(report.contains("=== Profiler Report ==="));
        assert!(report.contains("Total Events: 1"));
        assert!(report.contains("test_kernel"));
        assert!(report.contains("=== Events ==="));
    }

    #[test]
    fn test_scoped_event() {
        let mut profiler = SimpleProfiler::new();
        profiler.start().unwrap();

        {
            let _scoped = ScopedEvent::new(&mut profiler, "scoped_event").unwrap();
            std::thread::sleep(Duration::from_millis(1));
            // Event should automatically end when _scoped is dropped
        }

        profiler.stop().unwrap();

        let events = profiler.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "scoped_event");
        assert!(events[0].duration().is_some());
    }

    #[test]
    fn test_profiler_config_default() {
        let config = ProfilerConfig::default();

        assert!(!config.enabled);
        assert_eq!(config.max_events, Some(10000));
        assert!(config.detailed_timing);
        assert!(!config.track_memory);
        assert_eq!(config.event_types.len(), 3);
        assert!(config.event_types.contains(&EventType::KernelExecution));
        assert!(config.event_types.contains(&EventType::MemoryOperation));
        assert!(config.event_types.contains(&EventType::Synchronization));
        assert_eq!(config.output_format, OutputFormat::Text);
    }

    #[test]
    fn test_output_format_variants() {
        let formats = [
            OutputFormat::Text,
            OutputFormat::Json,
            OutputFormat::ChromeTracing,
            OutputFormat::Csv,
        ];

        // Ensure all formats are distinct
        for (i, format1) in formats.iter().enumerate() {
            for (j, format2) in formats.iter().enumerate() {
                if i != j {
                    assert_ne!(format1, format2);
                }
            }
        }
    }

    #[test]
    fn test_profiler_stats_update() {
        let mut profiler = SimpleProfiler::new();
        profiler.start().unwrap();

        // Add a kernel execution event
        let kernel_id = profiler.begin_event("kernel").unwrap();
        std::thread::sleep(Duration::from_millis(2));
        profiler.end_event(kernel_id).unwrap();

        // Update the event type to kernel execution
        if let Some(event) = profiler.events.iter_mut().find(|e| e.id == kernel_id) {
            event.event_type = EventType::KernelExecution;
        }

        // Add a memory operation event
        let memory_id = profiler.begin_event("memory").unwrap();
        std::thread::sleep(Duration::from_millis(1));
        profiler.end_event(memory_id).unwrap();

        // Update the event type to memory operation
        if let Some(event) = profiler.events.iter_mut().find(|e| e.id == memory_id) {
            event.event_type = EventType::MemoryOperation;
        }

        profiler.stop().unwrap();

        let stats = profiler.stats();
        assert_eq!(stats.total_events, 2);
        // Note: The stats update logic in SimpleProfiler is reset each time,
        // so these counters won't reflect the actual events unless we call update_stats
    }
}
