//! Profiler integration for JIT compilation
//!
//! This module provides comprehensive profiling capabilities for JIT-compiled code,
//! including performance counters, sampling profilers, and external profiler integration.

use crate::{JitError, JitResult};
use indexmap::IndexMap;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Profiler manager for JIT compilation
#[derive(Debug)]
pub struct ProfilerManager {
    /// Active profiling sessions
    sessions: Arc<Mutex<IndexMap<String, ProfilingSession>>>,

    /// Performance counters
    counters: Arc<Mutex<PerformanceCounters>>,

    /// Profiler configuration
    config: ProfilerConfig,

    /// External profiler integrations
    external_profilers: Vec<Box<dyn ExternalProfiler>>,

    /// Sampling profiler
    sampling_profiler: Option<SamplingProfiler>,

    /// Global profiling statistics
    stats: Arc<Mutex<ProfilerStats>>,

    /// Session counter for generating unique IDs
    session_counter: Arc<Mutex<u64>>,
}

/// Profiling session for tracking execution
#[derive(Debug, Clone)]
pub struct ProfilingSession {
    /// Session ID
    pub id: String,

    /// Session name
    pub name: String,

    /// Start time
    pub start_time: Instant,

    /// Duration (if completed)
    pub duration: Option<Duration>,

    /// Performance events collected
    pub events: Vec<PerformanceEvent>,

    /// Function call stacks
    pub call_stacks: Vec<CallStack>,

    /// Memory allocation tracking
    pub memory_events: Vec<MemoryEvent>,

    /// Hardware performance counters
    pub hw_counters: HardwareCounters,

    /// Session metadata
    pub metadata: HashMap<String, String>,

    /// Session status
    pub status: SessionStatus,
}

/// Status of a profiling session
#[derive(Debug, Clone, PartialEq)]
pub enum SessionStatus {
    Active,
    Completed,
    Failed(String),
    Cancelled,
}

/// Performance event types
#[derive(Debug, Clone)]
pub enum PerformanceEvent {
    /// Function entry
    FunctionEntry {
        function_name: String,
        timestamp: Instant,
        thread_id: u64,
        address: u64,
    },

    /// Function exit
    FunctionExit {
        function_name: String,
        timestamp: Instant,
        thread_id: u64,
        duration: Duration,
    },

    /// Kernel launch (for GPU code)
    KernelLaunch {
        kernel_name: String,
        timestamp: Instant,
        grid_size: (u32, u32, u32),
        block_size: (u32, u32, u32),
    },

    /// Kernel completion
    KernelComplete {
        kernel_name: String,
        timestamp: Instant,
        duration: Duration,
        occupancy: f32,
    },

    /// Memory allocation
    MemoryAlloc {
        size: usize,
        address: u64,
        timestamp: Instant,
        alignment: usize,
    },

    /// Memory deallocation
    MemoryFree { address: u64, timestamp: Instant },

    /// Cache miss
    CacheMiss {
        level: u8,
        address: u64,
        timestamp: Instant,
    },

    /// Branch misprediction
    BranchMisprediction {
        address: u64,
        timestamp: Instant,
        target_address: u64,
    },

    /// Custom user event
    Custom {
        name: String,
        timestamp: Instant,
        data: HashMap<String, String>,
    },
}

/// Call stack representation
#[derive(Debug, Clone)]
pub struct CallStack {
    /// Timestamp when stack was captured
    pub timestamp: Instant,

    /// Thread ID
    pub thread_id: u64,

    /// Stack frames (bottom to top)
    pub frames: Vec<StackFrame>,

    /// Total depth
    pub depth: usize,
}

/// Stack frame information
#[derive(Debug, Clone)]
pub struct StackFrame {
    /// Function name
    pub function_name: String,

    /// Address
    pub address: u64,

    /// Source location (if available)
    pub source_location: Option<SourceLocation>,

    /// Inlined function information
    pub inlined: bool,

    /// Module name
    pub module_name: String,
}

/// Source location for profiling
#[derive(Debug, Clone)]
pub struct SourceLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
}

/// Memory event tracking
#[derive(Debug, Clone)]
pub enum MemoryEvent {
    /// Allocation
    Alloc {
        size: usize,
        address: u64,
        timestamp: Instant,
        stack_trace: Vec<StackFrame>,
    },

    /// Deallocation
    Free {
        address: u64,
        timestamp: Instant,
        stack_trace: Vec<StackFrame>,
    },

    /// Memory access
    Access {
        address: u64,
        size: usize,
        is_write: bool,
        timestamp: Instant,
    },

    /// Page fault
    PageFault {
        address: u64,
        timestamp: Instant,
        fault_type: PageFaultType,
    },
}

/// Page fault types
#[derive(Debug, Clone)]
pub enum PageFaultType {
    Major,
    Minor,
    Protection,
}

/// Hardware performance counters
#[derive(Debug, Clone, Default)]
pub struct HardwareCounters {
    /// CPU cycles
    pub cycles: u64,

    /// Instructions executed
    pub instructions: u64,

    /// Cache misses (L1, L2, L3)
    pub cache_misses: [u64; 3],

    /// Cache references
    pub cache_references: u64,

    /// Branch mispredictions
    pub branch_mispredictions: u64,

    /// Branch instructions
    pub branches: u64,

    /// Page faults
    pub page_faults: u64,

    /// Context switches
    pub context_switches: u64,

    /// CPU migrations
    pub cpu_migrations: u64,

    /// Custom counters
    pub custom_counters: HashMap<String, u64>,
}

/// Global performance counters
#[derive(Debug, Clone, Default)]
pub struct PerformanceCounters {
    /// Total compilation time
    pub total_compile_time: Duration,

    /// Total execution time
    pub total_execution_time: Duration,

    /// Number of compilations
    pub compilation_count: u64,

    /// Number of executions
    pub execution_count: u64,

    /// Memory usage statistics
    pub memory_stats: MemoryStats,

    /// Function call counts
    pub function_calls: HashMap<String, u64>,

    /// Kernel launch counts
    pub kernel_launches: HashMap<String, u64>,

    /// Error counts
    pub error_counts: HashMap<String, u64>,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Current memory usage
    pub current_usage: usize,

    /// Peak memory usage
    pub peak_usage: usize,

    /// Total allocations
    pub total_allocations: u64,

    /// Total deallocations
    pub total_deallocations: u64,

    /// Total bytes allocated
    pub total_bytes_allocated: u64,

    /// Total bytes freed
    pub total_bytes_freed: u64,

    /// Average allocation size
    pub avg_allocation_size: f64,

    /// Allocation histogram
    pub allocation_histogram: HashMap<usize, u64>,
}

/// Profiler configuration
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Enable profiling
    pub enabled: bool,

    /// Sampling frequency in Hz
    pub sampling_frequency: u32,

    /// Enable call stack collection
    pub collect_call_stacks: bool,

    /// Enable memory tracking
    pub track_memory: bool,

    /// Enable hardware counter collection
    pub collect_hardware_counters: bool,

    /// Maximum number of events per session
    pub max_events_per_session: usize,

    /// Enable external profiler integration
    pub enable_external_profilers: bool,

    /// Output format for profiling data
    pub output_format: ProfilerOutputFormat,

    /// Output directory
    pub output_directory: String,
}

/// Profiler output formats
#[derive(Debug, Clone)]
pub enum ProfilerOutputFormat {
    /// Chrome tracing format
    ChromeTracing,

    /// Linux perf format
    PerfData,

    /// Intel VTune format
    VTune,

    /// Custom JSON format
    Json,

    /// Binary format
    Binary,
}

/// Sampling profiler for continuous monitoring
#[derive(Debug)]
pub struct SamplingProfiler {
    /// Sampling thread handle
    thread_handle: Option<std::thread::JoinHandle<()>>,

    /// Sampling configuration
    config: SamplingConfig,

    /// Collected samples
    samples: Arc<Mutex<Vec<Sample>>>,

    /// Running flag
    running: Arc<Mutex<bool>>,
}

/// Sampling configuration
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Sampling interval
    pub interval: Duration,

    /// Enable stack trace collection
    pub collect_stacks: bool,

    /// Maximum stack depth
    pub max_stack_depth: usize,

    /// Target threads (empty = all threads)
    pub target_threads: Vec<u64>,
}

/// Profiling sample
#[derive(Debug, Clone)]
pub struct Sample {
    /// Timestamp
    pub timestamp: Instant,

    /// Thread ID
    pub thread_id: u64,

    /// CPU ID
    pub cpu_id: u32,

    /// Program counter
    pub pc: u64,

    /// Stack trace
    pub stack_trace: Option<Vec<StackFrame>>,

    /// CPU utilization
    pub cpu_utilization: f32,

    /// Memory usage
    pub memory_usage: usize,
}

/// External profiler trait
pub trait ExternalProfiler: Send + Sync + std::fmt::Debug {
    /// Start profiling
    fn start(&mut self) -> JitResult<()>;

    /// Stop profiling
    fn stop(&mut self) -> JitResult<()>;

    /// Add a function to profile
    fn add_function(&mut self, name: &str, address: u64, size: usize) -> JitResult<()>;

    /// Remove a function from profiling
    fn remove_function(&mut self, address: u64) -> JitResult<()>;

    /// Export profiling data
    fn export_data(&self, output_path: &str) -> JitResult<()>;

    /// Get profiler name
    fn name(&self) -> &str;
}

/// Linux perf profiler integration
#[derive(Debug)]
pub struct PerfProfiler {
    /// Perf session active
    active: bool,

    /// Function mappings
    function_map: HashMap<u64, String>,

    /// JIT dump file
    jit_dump_file: Option<std::fs::File>,

    /// Map file path
    map_file: Option<std::path::PathBuf>,
}

/// Intel VTune profiler integration
#[derive(Debug)]
pub struct VTuneProfiler {
    /// VTune session active
    active: bool,

    /// Function mappings
    function_map: HashMap<u64, String>,
}

/// Profiler statistics
#[derive(Debug, Clone, Default)]
pub struct ProfilerStats {
    /// Total sessions created
    pub total_sessions: u64,

    /// Active sessions
    pub active_sessions: u64,

    /// Total events collected
    pub total_events: u64,

    /// Total samples collected
    pub total_samples: u64,

    /// Profiling overhead percentage
    pub overhead_percentage: f32,

    /// Data export count
    pub export_count: u64,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sampling_frequency: 1000, // 1 KHz
            collect_call_stacks: true,
            track_memory: true,
            collect_hardware_counters: false, // Requires privileged access
            max_events_per_session: 1_000_000,
            enable_external_profilers: false,
            output_format: ProfilerOutputFormat::Json,
            output_directory: "/tmp/torsh_profiling".to_string(),
        }
    }
}

impl ProfilerManager {
    /// Create a new profiler manager
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            sessions: Arc::new(Mutex::new(IndexMap::new())),
            counters: Arc::new(Mutex::new(PerformanceCounters::default())),
            config,
            external_profilers: Vec::new(),
            sampling_profiler: None,
            stats: Arc::new(Mutex::new(ProfilerStats::default())),
            session_counter: Arc::new(Mutex::new(0)),
        }
    }

    /// Create a new profiler manager with default configuration
    pub fn with_defaults() -> Self {
        Self::new(ProfilerConfig::default())
    }

    /// Start a new profiling session
    pub fn start_session(&mut self, name: &str) -> JitResult<String> {
        if !self.config.enabled {
            return Err(JitError::RuntimeError("Profiling disabled".to_string()));
        }

        let session_id = {
            let mut counter = self.session_counter.lock().unwrap();
            *counter += 1;
            format!("session_{}", *counter)
        };
        let session = ProfilingSession {
            id: session_id.clone(),
            name: name.to_string(),
            start_time: Instant::now(),
            duration: None,
            events: Vec::new(),
            call_stacks: Vec::new(),
            memory_events: Vec::new(),
            hw_counters: HardwareCounters::default(),
            metadata: HashMap::new(),
            status: SessionStatus::Active,
        };

        {
            let mut sessions = self.sessions.lock().unwrap();
            sessions.insert(session_id.clone(), session);
        }

        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_sessions += 1;
            stats.active_sessions += 1;
        }

        // Start external profilers if enabled
        if self.config.enable_external_profilers {
            for profiler in &mut self.external_profilers {
                profiler.start()?;
            }
        }

        Ok(session_id)
    }

    /// Stop a profiling session
    pub fn stop_session(&mut self, session_id: &str) -> JitResult<()> {
        let mut sessions = self.sessions.lock().unwrap();

        if let Some(session) = sessions.get_mut(session_id) {
            session.duration = Some(session.start_time.elapsed());
            session.status = SessionStatus::Completed;

            let mut stats = self.stats.lock().unwrap();
            stats.active_sessions = stats.active_sessions.saturating_sub(1);
        } else {
            return Err(JitError::RuntimeError(format!(
                "Session {} not found",
                session_id
            )));
        }

        // Stop external profilers if no active sessions
        let active_count = {
            let stats = self.stats.lock().unwrap();
            stats.active_sessions
        };

        if active_count == 0 && self.config.enable_external_profilers {
            for profiler in &mut self.external_profilers {
                profiler.stop()?;
            }
        }

        Ok(())
    }

    /// Record a performance event
    pub fn record_event(&mut self, session_id: &str, event: PerformanceEvent) -> JitResult<()> {
        let mut sessions = self.sessions.lock().unwrap();

        if let Some(session) = sessions.get_mut(session_id) {
            if session.events.len() < self.config.max_events_per_session {
                session.events.push(event);

                let mut stats = self.stats.lock().unwrap();
                stats.total_events += 1;
            }
        }

        Ok(())
    }

    /// Record a call stack
    pub fn record_call_stack(&mut self, session_id: &str, call_stack: CallStack) -> JitResult<()> {
        if !self.config.collect_call_stacks {
            return Ok(());
        }

        let mut sessions = self.sessions.lock().unwrap();

        if let Some(session) = sessions.get_mut(session_id) {
            session.call_stacks.push(call_stack);
        }

        Ok(())
    }

    /// Start sampling profiler
    pub fn start_sampling(&mut self) -> JitResult<()> {
        if self.sampling_profiler.is_some() {
            return Err(JitError::RuntimeError(
                "Sampling profiler already running".to_string(),
            ));
        }

        let config = SamplingConfig {
            interval: Duration::from_nanos(1_000_000_000 / self.config.sampling_frequency as u64),
            collect_stacks: self.config.collect_call_stacks,
            max_stack_depth: 64,
            target_threads: Vec::new(),
        };

        let mut sampling_profiler = SamplingProfiler::new(config)?;
        sampling_profiler.start()?;

        self.sampling_profiler = Some(sampling_profiler);
        Ok(())
    }

    /// Stop sampling profiler
    pub fn stop_sampling(&mut self) -> JitResult<()> {
        if let Some(mut profiler) = self.sampling_profiler.take() {
            profiler.stop()?;
        }
        Ok(())
    }

    /// Add external profiler
    pub fn add_external_profiler(&mut self, profiler: Box<dyn ExternalProfiler>) {
        self.external_profilers.push(profiler);
    }

    /// Export profiling data
    pub fn export_session_data(&self, session_id: &str, output_path: &str) -> JitResult<()> {
        let sessions = self.sessions.lock().unwrap();

        if let Some(session) = sessions.get(session_id) {
            match self.config.output_format {
                ProfilerOutputFormat::Json => self.export_json(session, output_path)?,
                ProfilerOutputFormat::ChromeTracing => {
                    self.export_chrome_tracing(session, output_path)?
                }
                _ => {
                    return Err(JitError::RuntimeError(
                        "Unsupported export format".to_string(),
                    ))
                }
            }

            let mut stats = self.stats.lock().unwrap();
            stats.export_count += 1;
        }

        Ok(())
    }

    /// Export session data as JSON
    fn export_json(&self, session: &ProfilingSession, output_path: &str) -> JitResult<()> {
        use serde_json::json;

        // Build comprehensive JSON representation
        let events_json: Vec<_> = session
            .events
            .iter()
            .map(|event| {
                json!({
                    "event": format!("{:?}", event)
                })
            })
            .collect();

        let session_json = json!({
            "name": session.name,
            "id": session.id,
            "start_time": session.start_time.elapsed().as_micros(),
            "duration": session.duration.map(|d| d.as_micros()),
            "events": events_json,
            "event_count": session.events.len(),
            "call_stack_count": session.call_stacks.len(),
            "memory_event_count": session.memory_events.len()
        });

        std::fs::write(
            output_path,
            serde_json::to_string_pretty(&session_json)
                .map_err(|e| JitError::RuntimeError(format!("JSON serialization failed: {}", e)))?,
        )
        .map_err(|e| JitError::RuntimeError(format!("Failed to write JSON: {}", e)))?;

        Ok(())
    }

    /// Export session data as Chrome tracing format
    fn export_chrome_tracing(
        &self,
        session: &ProfilingSession,
        output_path: &str,
    ) -> JitResult<()> {
        use serde_json::json;

        // Convert events to Chrome tracing format
        // See https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/
        let mut trace_events = Vec::new();

        for event in &session.events {
            match event {
                PerformanceEvent::FunctionEntry {
                    function_name,
                    timestamp,
                    thread_id,
                    ..
                } => {
                    trace_events.push(json!({
                        "name": function_name,
                        "cat": "function",
                        "ph": "B",  // Begin
                        "ts": timestamp.elapsed().as_micros() as u64,
                        "pid": 1,
                        "tid": thread_id
                    }));
                }
                PerformanceEvent::FunctionExit {
                    function_name,
                    timestamp,
                    thread_id,
                    duration,
                } => {
                    trace_events.push(json!({
                        "name": function_name,
                        "cat": "function",
                        "ph": "E",  // End
                        "ts": timestamp.elapsed().as_micros() as u64,
                        "pid": 1,
                        "tid": thread_id,
                        "args": { "duration_us": duration.as_micros() }
                    }));
                }
                PerformanceEvent::MemoryAlloc {
                    size,
                    timestamp,
                    address,
                    ..
                } => {
                    trace_events.push(json!({
                        "name": "Memory Allocation",
                        "cat": "memory",
                        "ph": "i",  // Instant event
                        "ts": timestamp.elapsed().as_micros() as u64,
                        "pid": 1,
                        "tid": 1,
                        "s": "g",   // Global scope
                        "args": { "size": size, "address": format!("0x{:x}", address) }
                    }));
                }
                PerformanceEvent::CacheMiss {
                    level, timestamp, ..
                } => {
                    trace_events.push(json!({
                        "name": format!("L{} Cache Miss", level),
                        "cat": "cache",
                        "ph": "i",
                        "ts": timestamp.elapsed().as_micros() as u64,
                        "pid": 1,
                        "tid": 1,
                        "s": "t",   // Thread scope
                    }));
                }
                _ => {
                    // Generic event
                    trace_events.push(json!({
                        "name": format!("{:?}", event),
                        "cat": "general",
                        "ph": "i",
                        "ts": 0,
                        "pid": 1,
                        "tid": 1
                    }));
                }
            }
        }

        // Add metadata
        let metadata = json!({
            "process_name": { "1": session.name.clone() },
            "thread_name": { "1": "Main Thread" }
        });

        // Build final trace
        let trace = json!({
            "traceEvents": trace_events,
            "displayTimeUnit": "ms",
            "metadata": metadata
        });

        std::fs::write(
            output_path,
            serde_json::to_string_pretty(&trace)
                .map_err(|e| JitError::RuntimeError(format!("JSON serialization failed: {}", e)))?,
        )
        .map_err(|e| JitError::RuntimeError(format!("Failed to write tracing data: {}", e)))?;

        Ok(())
    }

    /// Get session data
    pub fn get_session(&self, session_id: &str) -> Option<ProfilingSession> {
        let sessions = self.sessions.lock().unwrap();
        sessions.get(session_id).cloned()
    }

    /// Get performance counters
    pub fn get_counters(&self) -> PerformanceCounters {
        let counters = self.counters.lock().unwrap();
        counters.clone()
    }

    /// Get profiler statistics
    pub fn get_stats(&self) -> ProfilerStats {
        let stats = self.stats.lock().unwrap();
        stats.clone()
    }

    /// Update performance counters
    pub fn update_counters<F>(&self, update_fn: F)
    where
        F: FnOnce(&mut PerformanceCounters),
    {
        let mut counters = self.counters.lock().unwrap();
        update_fn(&mut *counters);
    }
}

impl SamplingProfiler {
    /// Create a new sampling profiler
    pub fn new(config: SamplingConfig) -> JitResult<Self> {
        Ok(Self {
            thread_handle: None,
            config,
            samples: Arc::new(Mutex::new(Vec::new())),
            running: Arc::new(Mutex::new(false)),
        })
    }

    /// Start sampling
    pub fn start(&mut self) -> JitResult<()> {
        let running = self.running.clone();
        let samples = self.samples.clone();
        let config = self.config.clone();

        *running.lock().unwrap() = true;

        let thread_handle = std::thread::spawn(move || {
            let mut last_sample_time = Instant::now();

            while *running.lock().unwrap() {
                let now = Instant::now();

                // Collect sample with actual system information
                let sample = Sample {
                    timestamp: now,
                    thread_id: Self::get_thread_id(),
                    cpu_id: Self::get_cpu_id(),
                    pc: 0, // Program counter requires platform-specific code
                    stack_trace: Self::collect_stack_trace(),
                    cpu_utilization: Self::calculate_cpu_utilization(&last_sample_time, &now),
                    memory_usage: Self::get_memory_usage(),
                };

                {
                    let mut samples_guard = samples.lock().unwrap();
                    samples_guard.push(sample);
                }

                last_sample_time = now;

                std::thread::sleep(config.interval);
            }
        });

        self.thread_handle = Some(thread_handle);
        Ok(())
    }

    /// Stop sampling
    pub fn stop(&mut self) -> JitResult<()> {
        *self.running.lock().unwrap() = false;

        if let Some(handle) = self.thread_handle.take() {
            handle.join().map_err(|_| {
                JitError::RuntimeError("Failed to join sampling thread".to_string())
            })?;
        }

        Ok(())
    }

    /// Get collected samples
    pub fn get_samples(&self) -> Vec<Sample> {
        let samples = self.samples.lock().unwrap();
        samples.clone()
    }

    /// Get the current thread ID
    fn get_thread_id() -> u64 {
        // Use thread name hash for portability
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        std::thread::current().id().hash(&mut hasher);
        hasher.finish()
    }

    /// Get the current CPU ID
    fn get_cpu_id() -> u32 {
        // Platform-specific CPU ID retrieval
        // For cross-platform compatibility, default to 0
        0
    }

    /// Collect stack trace
    fn collect_stack_trace() -> Option<Vec<StackFrame>> {
        // Use backtrace-rs or similar library in production
        // For now, return a simple placeholder
        Some(vec![StackFrame {
            function_name: "sampling_thread".to_string(),
            address: 0,
            source_location: Some(SourceLocation {
                file: "profiler.rs".to_string(),
                line: 0,
                column: 0,
            }),
            inlined: false,
            module_name: "torsh_jit".to_string(),
        }])
    }

    /// Calculate CPU utilization
    fn calculate_cpu_utilization(_last_time: &Instant, _current_time: &Instant) -> f32 {
        // This would require platform-specific CPU time queries
        // Placeholder implementation
        (num_cpus::get() as f32) * 0.5 // Assume 50% utilization
    }

    /// Get current memory usage
    fn get_memory_usage() -> usize {
        // Platform-specific memory usage
        #[cfg(target_os = "linux")]
        {
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            if let Ok(kb) = kb_str.parse::<usize>() {
                                return kb * 1024; // Convert to bytes
                            }
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            use std::process::Command;
            if let Ok(output) = Command::new("ps")
                .args(["-o", "rss=", "-p", &std::process::id().to_string()])
                .output()
            {
                if let Ok(text) = String::from_utf8(output.stdout) {
                    if let Ok(kb) = text.trim().parse::<usize>() {
                        return kb * 1024; // Convert to bytes
                    }
                }
            }
        }

        0 // Default if not available
    }
}

impl ExternalProfiler for PerfProfiler {
    fn start(&mut self) -> JitResult<()> {
        // Initialize perf profiling
        // On Linux, perf uses /tmp/perf-<pid>.map for JIT symbol maps
        #[cfg(target_os = "linux")]
        {
            let pid = std::process::id();
            let map_file = format!("/tmp/perf-{}.map", pid);

            // Create or truncate the perf map file
            std::fs::write(&map_file, "")
                .map_err(|e| JitError::RuntimeError(format!("Failed to create perf map: {}", e)))?;

            self.map_file = Some(map_file.into());
        }

        self.active = true;
        Ok(())
    }

    fn stop(&mut self) -> JitResult<()> {
        // Finalize perf profiling
        self.active = false;
        Ok(())
    }

    fn add_function(&mut self, name: &str, address: u64, size: usize) -> JitResult<()> {
        // Add function to perf symbol map
        self.function_map.insert(address, name.to_string());

        // Write to perf map file format: <start_addr> <size> <symbol_name>
        #[cfg(target_os = "linux")]
        {
            if let Some(ref map_file) = self.map_file {
                use std::io::Write;
                let mut file = std::fs::OpenOptions::new()
                    .append(true)
                    .open(map_file)
                    .map_err(|e| {
                        JitError::RuntimeError(format!("Failed to open perf map: {}", e))
                    })?;

                writeln!(file, "{:x} {:x} {}", address, size, name).map_err(|e| {
                    JitError::RuntimeError(format!("Failed to write to perf map: {}", e))
                })?;
            }
        }

        Ok(())
    }

    fn remove_function(&mut self, address: u64) -> JitResult<()> {
        self.function_map.remove(&address);
        // Note: perf map files are append-only, so we can't remove entries
        Ok(())
    }

    fn export_data(&self, output_path: &str) -> JitResult<()> {
        // Export perf-compatible symbol map
        use std::io::Write;
        let mut file = std::fs::File::create(output_path)
            .map_err(|e| JitError::RuntimeError(format!("Failed to create export file: {}", e)))?;

        // Write function map in perf format
        for (address, name) in &self.function_map {
            writeln!(file, "{:x} 0 {}", address, name)
                .map_err(|e| JitError::RuntimeError(format!("Failed to write perf data: {}", e)))?;
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "PerfProfiler"
    }
}

impl ExternalProfiler for VTuneProfiler {
    fn start(&mut self) -> JitResult<()> {
        // Initialize VTune profiling using Intel JIT API
        // VTune uses a shared library interface for JIT notification
        #[cfg(target_os = "linux")]
        {
            // In a full implementation, this would dynamically load libittnotify
            // and call iJIT_NotifyEvent(iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED, ...)
            log::info!("VTune profiler started (stub implementation)");
        }

        #[cfg(target_os = "windows")]
        {
            log::info!("VTune profiler started on Windows (stub implementation)");
        }

        self.active = true;
        Ok(())
    }

    fn stop(&mut self) -> JitResult<()> {
        // Finalize VTune profiling
        self.active = false;
        Ok(())
    }

    fn add_function(&mut self, name: &str, address: u64, size: usize) -> JitResult<()> {
        // Add function to VTune using JIT API
        self.function_map.insert(address, name.to_string());

        if self.active {
            // In a full implementation, this would call:
            // iJIT_NotifyEvent(iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED, method_load_info)
            // where method_load_info contains: method_id, method_name,
            // method_load_address, method_size, line_number_table, etc.

            log::debug!(
                "Registered function '{}' at 0x{:x} (size: {}) with VTune",
                name,
                address,
                size
            );
        }

        Ok(())
    }

    fn remove_function(&mut self, address: u64) -> JitResult<()> {
        self.function_map.remove(&address);

        if self.active {
            // In a full implementation, this would call:
            // iJIT_NotifyEvent(iJVM_EVENT_TYPE_METHOD_UNLOAD_START, method_id)
            log::debug!("Unregistered function at 0x{:x} from VTune", address);
        }

        Ok(())
    }

    fn export_data(&self, output_path: &str) -> JitResult<()> {
        // Export VTune-compatible symbol data
        use std::io::Write;
        let mut file = std::fs::File::create(output_path)
            .map_err(|e| JitError::RuntimeError(format!("Failed to create export file: {}", e)))?;

        // Write header
        writeln!(file, "# VTune JIT Symbol Map")
            .map_err(|e| JitError::RuntimeError(format!("Write failed: {}", e)))?;
        writeln!(file, "# Format: <address> <size> <name>")
            .map_err(|e| JitError::RuntimeError(format!("Write failed: {}", e)))?;
        writeln!(file).map_err(|e| JitError::RuntimeError(format!("Write failed: {}", e)))?;

        // Write function map
        for (address, name) in &self.function_map {
            writeln!(file, "{:016x} 0000 {}", address, name).map_err(|e| {
                JitError::RuntimeError(format!("Failed to write VTune data: {}", e))
            })?;
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "VTuneProfiler"
    }
}

impl PerfProfiler {
    /// Create a new perf profiler
    pub fn new() -> Self {
        Self {
            active: false,
            function_map: HashMap::new(),
            jit_dump_file: None,
            map_file: None,
        }
    }
}

impl VTuneProfiler {
    /// Create a new VTune profiler
    pub fn new() -> Self {
        Self {
            active: false,
            function_map: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_manager_creation() {
        let manager = ProfilerManager::with_defaults();
        assert!(manager.config.enabled);
        assert_eq!(manager.config.sampling_frequency, 1000);
    }

    #[test]
    fn test_session_lifecycle() {
        let mut manager = ProfilerManager::with_defaults();

        let session_id = manager.start_session("test_session").unwrap();
        assert!(!session_id.is_empty());

        let session = manager.get_session(&session_id).unwrap();
        assert_eq!(session.name, "test_session");
        assert_eq!(session.status, SessionStatus::Active);

        manager.stop_session(&session_id).unwrap();

        let session = manager.get_session(&session_id).unwrap();
        assert_eq!(session.status, SessionStatus::Completed);
        assert!(session.duration.is_some());
    }

    #[test]
    fn test_performance_event_recording() {
        let mut manager = ProfilerManager::with_defaults();
        let session_id = manager.start_session("test_session").unwrap();

        let event = PerformanceEvent::FunctionEntry {
            function_name: "test_function".to_string(),
            timestamp: Instant::now(),
            thread_id: 1,
            address: 0x1000,
        };

        manager.record_event(&session_id, event).unwrap();

        let session = manager.get_session(&session_id).unwrap();
        assert_eq!(session.events.len(), 1);
    }

    #[test]
    fn test_external_profiler_integration() {
        let mut manager = ProfilerManager::with_defaults();
        let perf_profiler = Box::new(PerfProfiler::new());

        manager.add_external_profiler(perf_profiler);
        assert_eq!(manager.external_profilers.len(), 1);
    }
}
