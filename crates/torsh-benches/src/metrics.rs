//! System metrics collection for benchmarks

use std::time::{Duration, Instant};

/// System metrics collector
pub struct MetricsCollector {
    start_time: Option<Instant>,
    memory_tracker: MemoryTracker,
    cpu_tracker: CpuTracker,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            start_time: None,
            memory_tracker: MemoryTracker::new(),
            cpu_tracker: CpuTracker::new(),
        }
    }

    /// Start collecting metrics
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.memory_tracker.start();
        self.cpu_tracker.start();
    }

    /// Stop collecting metrics and return results
    pub fn stop(&mut self) -> SystemMetrics {
        let elapsed = self
            .start_time
            .map(|start| start.elapsed())
            .unwrap_or(Duration::ZERO);

        let memory_stats = self.memory_tracker.stop();
        let cpu_stats = self.cpu_tracker.stop();

        SystemMetrics {
            elapsed_time: elapsed,
            memory_stats,
            cpu_stats,
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete system metrics
#[derive(Debug, Clone)]
pub struct SystemMetrics {
    pub elapsed_time: Duration,
    pub memory_stats: MemoryStats,
    pub cpu_stats: CpuStats,
}

impl SystemMetrics {
    /// Get memory efficiency (operations per MB)
    pub fn memory_efficiency(&self, operations: usize) -> f64 {
        if self.memory_stats.peak_usage_mb > 0.0 {
            operations as f64 / self.memory_stats.peak_usage_mb
        } else {
            0.0
        }
    }

    /// Get CPU utilization percentage
    pub fn cpu_utilization(&self) -> f64 {
        self.cpu_stats.average_usage_percent
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    /// Initial memory usage in MB
    pub initial_usage_mb: f64,

    /// Peak memory usage in MB
    pub peak_usage_mb: f64,

    /// Final memory usage in MB
    pub final_usage_mb: f64,

    /// Memory allocated during benchmark in MB
    pub allocated_mb: f64,

    /// Memory deallocated during benchmark in MB
    pub deallocated_mb: f64,

    /// Number of allocations
    pub allocation_count: usize,

    /// Number of deallocations
    pub deallocation_count: usize,
}

impl Default for MemoryStats {
    fn default() -> Self {
        Self {
            initial_usage_mb: 0.0,
            peak_usage_mb: 0.0,
            final_usage_mb: 0.0,
            allocated_mb: 0.0,
            deallocated_mb: 0.0,
            allocation_count: 0,
            deallocation_count: 0,
        }
    }
}

/// CPU usage statistics
#[derive(Debug, Clone)]
pub struct CpuStats {
    /// Average CPU usage percentage
    pub average_usage_percent: f64,

    /// Peak CPU usage percentage
    pub peak_usage_percent: f64,

    /// Number of CPU cores used
    pub cores_used: usize,

    /// Context switches during benchmark
    pub context_switches: usize,
}

impl Default for CpuStats {
    fn default() -> Self {
        Self {
            average_usage_percent: 0.0,
            peak_usage_percent: 0.0,
            cores_used: 1,
            context_switches: 0,
        }
    }
}

/// Memory usage tracker
pub struct MemoryTracker {
    initial_usage: Option<f64>,
    peak_usage: f64,
    samples: Vec<f64>,
    start_time: Option<Instant>,
}

impl MemoryTracker {
    pub fn new() -> Self {
        Self {
            initial_usage: None,
            peak_usage: 0.0,
            samples: Vec::new(),
            start_time: None,
        }
    }

    /// Start tracking memory usage
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.initial_usage = Some(Self::get_process_memory_mb());
        self.peak_usage = self.initial_usage.unwrap_or(0.0);
        self.samples.clear();
    }

    /// Stop tracking and return statistics
    pub fn stop(&mut self) -> MemoryStats {
        let current_usage = Self::get_process_memory_mb();
        let initial = self.initial_usage.unwrap_or(0.0);

        MemoryStats {
            initial_usage_mb: initial,
            peak_usage_mb: self.peak_usage,
            final_usage_mb: current_usage,
            allocated_mb: (current_usage - initial).max(0.0),
            deallocated_mb: (initial - current_usage).max(0.0),
            allocation_count: 0, // Would need system hook to track this
            deallocation_count: 0,
        }
    }

    /// Sample current memory usage
    pub fn sample(&mut self) {
        let usage = Self::get_process_memory_mb();
        self.samples.push(usage);
        if usage > self.peak_usage {
            self.peak_usage = usage;
        }
    }

    /// Get current process memory usage in MB
    fn get_process_memory_mb() -> f64 {
        // Simplified implementation - would use platform-specific APIs in production
        #[cfg(target_os = "linux")]
        {
            if let Ok(contents) = std::fs::read_to_string("/proc/self/status") {
                for line in contents.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<f64>() {
                                return kb / 1024.0; // Convert KB to MB
                            }
                        }
                    }
                }
            }
        }

        // Fallback: rough estimate based on allocation tracking
        // In a real implementation, this would use proper system APIs
        0.0
    }
}

impl Default for MemoryTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// CPU usage tracker
pub struct CpuTracker {
    samples: Vec<f64>,
    peak_usage: f64,
    start_time: Option<Instant>,
}

impl CpuTracker {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            peak_usage: 0.0,
            start_time: None,
        }
    }

    /// Start tracking CPU usage
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
        self.samples.clear();
        self.peak_usage = 0.0;
    }

    /// Stop tracking and return statistics
    pub fn stop(&mut self) -> CpuStats {
        let average_usage = if self.samples.is_empty() {
            0.0
        } else {
            self.samples.iter().sum::<f64>() / self.samples.len() as f64
        };

        CpuStats {
            average_usage_percent: average_usage,
            peak_usage_percent: self.peak_usage,
            cores_used: num_cpus::get(),
            context_switches: 0, // Would need system monitoring to track this
        }
    }

    /// Sample current CPU usage
    pub fn sample(&mut self) {
        let usage = Self::get_cpu_usage_percent();
        self.samples.push(usage);
        if usage > self.peak_usage {
            self.peak_usage = usage;
        }
    }

    /// Get current CPU usage percentage
    fn get_cpu_usage_percent() -> f64 {
        // Simplified implementation
        // In production, would use platform-specific APIs or libraries like sysinfo
        0.0
    }
}

impl Default for CpuTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance profiler for detailed analysis
pub struct PerformanceProfiler {
    events: Vec<ProfileEvent>,
    current_stack: Vec<String>,
    start_time: Instant,
}

impl PerformanceProfiler {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            current_stack: Vec::new(),
            start_time: Instant::now(),
        }
    }

    /// Begin a profiling event
    pub fn begin_event(&mut self, name: &str) {
        let event = ProfileEvent {
            name: name.to_string(),
            event_type: ProfileEventType::Begin,
            timestamp: self.start_time.elapsed(),
            thread_id: std::thread::current().id(),
            stack_depth: self.current_stack.len(),
        };

        self.current_stack.push(name.to_string());
        self.events.push(event);
    }

    /// End a profiling event
    pub fn end_event(&mut self, name: &str) {
        if let Some(stack_name) = self.current_stack.pop() {
            assert_eq!(stack_name, name, "Mismatched profiling events");
        }

        let event = ProfileEvent {
            name: name.to_string(),
            event_type: ProfileEventType::End,
            timestamp: self.start_time.elapsed(),
            thread_id: std::thread::current().id(),
            stack_depth: self.current_stack.len(),
        };

        self.events.push(event);
    }

    /// Add an instant marker
    pub fn marker(&mut self, name: &str) {
        let event = ProfileEvent {
            name: name.to_string(),
            event_type: ProfileEventType::Marker,
            timestamp: self.start_time.elapsed(),
            thread_id: std::thread::current().id(),
            stack_depth: self.current_stack.len(),
        };

        self.events.push(event);
    }

    /// Generate Chrome tracing format output
    pub fn export_chrome_trace(&self, path: &str) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        writeln!(file, "{{")?;
        writeln!(file, "  \"traceEvents\": [")?;

        for (i, event) in self.events.iter().enumerate() {
            let phase = match event.event_type {
                ProfileEventType::Begin => "B",
                ProfileEventType::End => "E",
                ProfileEventType::Marker => "i",
            };

            writeln!(
                file,
                "    {{\"name\": \"{}\", \"ph\": \"{}\", \"ts\": {}, \"pid\": 1, \"tid\": {:?}}}{}",
                event.name,
                phase,
                event.timestamp.as_micros(),
                event.thread_id,
                if i < self.events.len() - 1 { "," } else { "" }
            )?;
        }

        writeln!(file, "  ]")?;
        writeln!(file, "}}")?;

        Ok(())
    }

    /// Generate performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let mut durations = std::collections::HashMap::new();
        let mut stack = Vec::new();

        for event in &self.events {
            match event.event_type {
                ProfileEventType::Begin => {
                    stack.push((event.name.clone(), event.timestamp));
                }
                ProfileEventType::End => {
                    if let Some((name, start_time)) = stack.pop() {
                        assert_eq!(name, event.name);
                        let duration = event.timestamp - start_time;
                        durations
                            .entry(name)
                            .or_insert_with(Vec::new)
                            .push(duration);
                    }
                }
                ProfileEventType::Marker => {} // Instant events don't have duration
            }
        }

        let mut function_stats = Vec::new();
        for (name, times) in durations {
            let total_time: Duration = times.iter().sum();
            let avg_time = total_time / times.len() as u32;
            let min_time = times.iter().min().copied().unwrap_or_default();
            let max_time = times.iter().max().copied().unwrap_or_default();

            function_stats.push(FunctionStats {
                name,
                call_count: times.len(),
                total_time,
                average_time: avg_time,
                min_time,
                max_time,
            });
        }

        // Sort by total time descending
        function_stats.sort_by(|a, b| b.total_time.cmp(&a.total_time));

        PerformanceReport {
            function_stats,
            total_events: self.events.len(),
        }
    }
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Profiling event
#[derive(Debug, Clone)]
pub struct ProfileEvent {
    pub name: String,
    pub event_type: ProfileEventType,
    pub timestamp: Duration,
    pub thread_id: std::thread::ThreadId,
    pub stack_depth: usize,
}

/// Profiling event type
#[derive(Debug, Clone, Copy)]
pub enum ProfileEventType {
    Begin,
    End,
    Marker,
}

/// Performance analysis report
#[derive(Debug)]
pub struct PerformanceReport {
    pub function_stats: Vec<FunctionStats>,
    pub total_events: usize,
}

/// Statistics for a profiled function
#[derive(Debug)]
pub struct FunctionStats {
    pub name: String,
    pub call_count: usize,
    pub total_time: Duration,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
}

/// Scoped profiler that automatically ends when dropped
pub struct ScopedProfiler<'a> {
    profiler: &'a mut PerformanceProfiler,
    name: String,
}

impl<'a> ScopedProfiler<'a> {
    pub fn new(profiler: &'a mut PerformanceProfiler, name: &str) -> Self {
        profiler.begin_event(name);
        Self {
            profiler,
            name: name.to_string(),
        }
    }
}

impl<'a> Drop for ScopedProfiler<'a> {
    fn drop(&mut self) {
        self.profiler.end_event(&self.name);
    }
}

/// Macro for scoped profiling
#[macro_export]
macro_rules! profile {
    ($profiler:expr, $name:expr) => {
        let _scoped_profiler = $crate::metrics::ScopedProfiler::new($profiler, $name);
    };
}

// External dependency for CPU count
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
}
