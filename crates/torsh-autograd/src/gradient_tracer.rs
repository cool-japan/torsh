// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Gradient Computation Path Tracing
//!
//! This module provides comprehensive tracing support for gradient computation paths,
//! enabling detailed debugging, performance analysis, and understanding of autograd
//! execution flow.
//!
//! # Features
//!
//! - **Path Recording**: Captures complete gradient computation paths
//! - **Timeline Analysis**: Tracks timing and ordering of operations
//! - **Memory Tracking**: Monitors memory allocations during gradient computation
//! - **Dependency Analysis**: Identifies computation dependencies and critical paths
//! - **Visual Representation**: Generates human-readable trace outputs
//! - **Export Formats**: Supports JSON, Chrome Trace Format, and custom formats

use crate::error_handling::{AutogradError, AutogradResult};
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Unique identifier for a trace event
pub type TraceEventId = u64;

/// Unique identifier for a gradient computation path
pub type PathId = u64;

/// Gradient computation event captured during tracing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent {
    /// Unique event identifier
    pub id: TraceEventId,

    /// Parent event ID (if any)
    pub parent_id: Option<TraceEventId>,

    /// Path identifier this event belongs to
    pub path_id: PathId,

    /// Event type (operation, memory allocation, synchronization, etc.)
    pub event_type: EventType,

    /// Operation name
    pub operation: String,

    /// Event timestamp
    pub timestamp: DateTime<Utc>,

    /// Event duration (for begin/end paired events)
    pub duration: Option<Duration>,

    /// Memory allocated (in bytes)
    pub memory_allocated: Option<usize>,

    /// Memory deallocated (in bytes)
    pub memory_deallocated: Option<usize>,

    /// Input tensor IDs
    pub input_ids: Vec<String>,

    /// Output tensor IDs
    pub output_ids: Vec<String>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Type of trace event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    /// Start of an operation
    OperationBegin,

    /// End of an operation
    OperationEnd,

    /// Memory allocation
    MemoryAllocation,

    /// Memory deallocation
    MemoryDeallocation,

    /// Gradient computation
    GradientComputation,

    /// Backward pass start
    BackwardBegin,

    /// Backward pass end
    BackwardEnd,

    /// Checkpoint save
    CheckpointSave,

    /// Checkpoint restore
    CheckpointRestore,

    /// Synchronization point
    Synchronization,

    /// Custom event
    Custom,
}

/// Gradient computation path trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientPath {
    /// Unique path identifier
    pub id: PathId,

    /// Path name/description
    pub name: String,

    /// All events in this path
    pub events: Vec<TraceEvent>,

    /// Start timestamp
    pub start_time: DateTime<Utc>,

    /// End timestamp
    pub end_time: Option<DateTime<Utc>>,

    /// Total duration
    pub total_duration: Option<Duration>,

    /// Peak memory usage
    pub peak_memory: usize,

    /// Total memory allocated
    pub total_memory_allocated: usize,

    /// Number of operations
    pub operation_count: usize,
}

impl GradientPath {
    /// Create a new gradient path
    pub fn new(id: PathId, name: String) -> Self {
        Self {
            id,
            name,
            events: Vec::new(),
            start_time: Utc::now(),
            end_time: None,
            total_duration: None,
            peak_memory: 0,
            total_memory_allocated: 0,
            operation_count: 0,
        }
    }

    /// Add an event to this path
    pub fn add_event(&mut self, event: TraceEvent) {
        if let Some(mem) = event.memory_allocated {
            self.total_memory_allocated += mem;
            self.peak_memory = self.peak_memory.max(self.total_memory_allocated);
        }

        if let Some(mem) = event.memory_deallocated {
            self.total_memory_allocated = self.total_memory_allocated.saturating_sub(mem);
        }

        if matches!(
            event.event_type,
            EventType::OperationBegin | EventType::OperationEnd
        ) {
            self.operation_count += 1;
        }

        self.events.push(event);
    }

    /// Finalize the path
    pub fn finalize(&mut self) {
        self.end_time = Some(Utc::now());
        if let Some(end) = self.end_time {
            self.total_duration = Some(
                chrono::Duration::to_std(&end.signed_duration_since(self.start_time))
                    .unwrap_or(Duration::ZERO),
            );
        }
    }

    /// Get critical path (longest dependency chain)
    pub fn critical_path(&self) -> Vec<&TraceEvent> {
        // Build dependency graph
        let mut dependencies: HashMap<TraceEventId, Vec<TraceEventId>> = HashMap::new();
        let mut event_map: HashMap<TraceEventId, &TraceEvent> = HashMap::new();

        for event in &self.events {
            event_map.insert(event.id, event);
            if let Some(parent) = event.parent_id {
                dependencies.entry(parent).or_default().push(event.id);
            }
        }

        // Find longest path using dynamic programming
        let mut longest_paths: HashMap<TraceEventId, Vec<TraceEventId>> = HashMap::new();

        fn compute_longest_path(
            event_id: TraceEventId,
            dependencies: &HashMap<TraceEventId, Vec<TraceEventId>>,
            longest_paths: &mut HashMap<TraceEventId, Vec<TraceEventId>>,
        ) -> Vec<TraceEventId> {
            if let Some(path) = longest_paths.get(&event_id) {
                return path.clone();
            }

            let mut max_path = vec![event_id];

            if let Some(children) = dependencies.get(&event_id) {
                for &child in children {
                    let child_path = compute_longest_path(child, dependencies, longest_paths);
                    if child_path.len() + 1 > max_path.len() {
                        max_path = vec![event_id];
                        max_path.extend(child_path);
                    }
                }
            }

            longest_paths.insert(event_id, max_path.clone());
            max_path
        }

        // Find root events (no parents)
        let root_events: Vec<_> = self
            .events
            .iter()
            .filter(|e| e.parent_id.is_none())
            .map(|e| e.id)
            .collect();

        let mut critical = Vec::new();
        for root in root_events {
            let path = compute_longest_path(root, &dependencies, &mut longest_paths);
            if path.len() > critical.len() {
                critical = path;
            }
        }

        critical
            .iter()
            .filter_map(|id| event_map.get(id).copied())
            .collect()
    }

    /// Get operation statistics
    pub fn operation_stats(&self) -> HashMap<String, OperationStats> {
        let mut stats: HashMap<String, OperationStats> = HashMap::new();

        for event in &self.events {
            let stat = stats
                .entry(event.operation.clone())
                .or_insert_with(|| OperationStats {
                    count: 0,
                    total_duration: Duration::ZERO,
                    total_memory: 0,
                    min_duration: Duration::MAX,
                    max_duration: Duration::ZERO,
                });

            stat.count += 1;

            if let Some(duration) = event.duration {
                stat.total_duration += duration;
                stat.min_duration = stat.min_duration.min(duration);
                stat.max_duration = stat.max_duration.max(duration);
            }

            if let Some(mem) = event.memory_allocated {
                stat.total_memory += mem;
            }
        }

        stats
    }
}

/// Statistics for a specific operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStats {
    /// Number of times this operation occurred
    pub count: usize,

    /// Total duration across all invocations
    pub total_duration: Duration,

    /// Total memory allocated
    pub total_memory: usize,

    /// Minimum duration
    pub min_duration: Duration,

    /// Maximum duration
    pub max_duration: Duration,
}

impl OperationStats {
    /// Get average duration
    pub fn average_duration(&self) -> Duration {
        if self.count == 0 {
            Duration::ZERO
        } else {
            self.total_duration / self.count as u32
        }
    }

    /// Get average memory
    pub fn average_memory(&self) -> usize {
        if self.count == 0 {
            0
        } else {
            self.total_memory / self.count
        }
    }
}

/// Gradient tracer for capturing computation paths
pub struct GradientTracer {
    /// Whether tracing is enabled
    enabled: Arc<RwLock<bool>>,

    /// Current paths being traced
    active_paths: Arc<Mutex<HashMap<PathId, GradientPath>>>,

    /// Completed paths
    completed_paths: Arc<Mutex<Vec<GradientPath>>>,

    /// Next event ID
    next_event_id: Arc<Mutex<TraceEventId>>,

    /// Next path ID
    next_path_id: Arc<Mutex<PathId>>,

    /// Event buffer for asynchronous processing
    event_buffer: Arc<Mutex<VecDeque<TraceEvent>>>,

    /// Maximum number of events to buffer
    max_buffer_size: usize,

    /// Configuration
    config: TracerConfig,
}

/// Tracer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracerConfig {
    /// Whether to trace memory allocations
    pub trace_memory: bool,

    /// Whether to trace operation timings
    pub trace_timing: bool,

    /// Whether to capture metadata
    pub capture_metadata: bool,

    /// Minimum operation duration to record (filter out very fast ops)
    pub min_duration: Option<Duration>,

    /// Maximum number of events per path
    pub max_events_per_path: Option<usize>,

    /// Whether to enable async event processing
    pub async_processing: bool,
}

impl Default for TracerConfig {
    fn default() -> Self {
        Self {
            trace_memory: true,
            trace_timing: true,
            capture_metadata: true,
            min_duration: None,
            max_events_per_path: Some(100_000),
            async_processing: false,
        }
    }
}

impl GradientTracer {
    /// Create a new gradient tracer
    pub fn new(config: TracerConfig) -> Self {
        Self {
            enabled: Arc::new(RwLock::new(false)),
            active_paths: Arc::new(Mutex::new(HashMap::new())),
            completed_paths: Arc::new(Mutex::new(Vec::new())),
            next_event_id: Arc::new(Mutex::new(1)), // Start IDs from 1
            next_path_id: Arc::new(Mutex::new(1)),  // Start IDs from 1
            event_buffer: Arc::new(Mutex::new(VecDeque::new())),
            max_buffer_size: 10_000,
            config,
        }
    }

    /// Enable tracing
    pub fn enable(&self) {
        *self.enabled.write() = true;
    }

    /// Disable tracing
    pub fn disable(&self) {
        *self.enabled.write() = false;
    }

    /// Check if tracing is enabled
    pub fn is_enabled(&self) -> bool {
        *self.enabled.read()
    }

    /// Start a new gradient path
    pub fn start_path(&self, name: String) -> PathId {
        let path_id = {
            let mut next_id = self.next_path_id.lock();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let path = GradientPath::new(path_id, name);
        self.active_paths.lock().insert(path_id, path);

        path_id
    }

    /// End a gradient path
    pub fn end_path(&self, path_id: PathId) {
        let mut active = self.active_paths.lock();
        if let Some(mut path) = active.remove(&path_id) {
            path.finalize();
            self.completed_paths.lock().push(path);
        }
    }

    /// Record a trace event
    pub fn record_event(
        &self,
        path_id: PathId,
        event_type: EventType,
        operation: String,
        parent_id: Option<TraceEventId>,
    ) -> TraceEventId {
        if !self.is_enabled() {
            return 0;
        }

        let event_id = {
            let mut next_id = self.next_event_id.lock();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let event = TraceEvent {
            id: event_id,
            parent_id,
            path_id,
            event_type,
            operation,
            timestamp: Utc::now(),
            duration: None,
            memory_allocated: None,
            memory_deallocated: None,
            input_ids: Vec::new(),
            output_ids: Vec::new(),
            metadata: HashMap::new(),
        };

        if self.config.async_processing {
            let mut buffer = self.event_buffer.lock();
            buffer.push_back(event);

            // Prevent buffer overflow
            if buffer.len() > self.max_buffer_size {
                buffer.pop_front();
            }
        } else {
            self.add_event_to_path(path_id, event);
        }

        event_id
    }

    /// Record an operation with automatic begin/end events
    pub fn trace_operation<F, R>(
        &self,
        path_id: PathId,
        operation: String,
        parent_id: Option<TraceEventId>,
        f: F,
    ) -> AutogradResult<(R, TraceEventId)>
    where
        F: FnOnce() -> AutogradResult<R>,
    {
        if !self.is_enabled() {
            return f().map(|r| (r, 0));
        }

        let begin_id = self.record_event(
            path_id,
            EventType::OperationBegin,
            operation.clone(),
            parent_id,
        );

        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();

        let end_id = {
            let mut next_id = self.next_event_id.lock();
            let id = *next_id;
            *next_id += 1;
            id
        };

        // Filter based on minimum duration
        if let Some(min_dur) = self.config.min_duration {
            if duration < min_dur {
                return result.map(|r| (r, begin_id));
            }
        }

        let mut event = TraceEvent {
            id: end_id,
            parent_id: Some(begin_id),
            path_id,
            event_type: EventType::OperationEnd,
            operation,
            timestamp: Utc::now(),
            duration: Some(duration),
            memory_allocated: None,
            memory_deallocated: None,
            input_ids: Vec::new(),
            output_ids: Vec::new(),
            metadata: HashMap::new(),
        };

        if self.config.trace_timing {
            event.metadata.insert(
                "duration_micros".to_string(),
                duration.as_micros().to_string(),
            );
        }

        self.add_event_to_path(path_id, event);

        result.map(|r| (r, end_id))
    }

    /// Add memory allocation event
    pub fn record_memory_allocation(&self, path_id: PathId, bytes: usize, tensor_id: String) {
        if !self.is_enabled() || !self.config.trace_memory {
            return;
        }

        let event_id = {
            let mut next_id = self.next_event_id.lock();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let event = TraceEvent {
            id: event_id,
            parent_id: None,
            path_id,
            event_type: EventType::MemoryAllocation,
            operation: "memory_alloc".to_string(),
            timestamp: Utc::now(),
            duration: None,
            memory_allocated: Some(bytes),
            memory_deallocated: None,
            input_ids: Vec::new(),
            output_ids: vec![tensor_id],
            metadata: HashMap::new(),
        };

        self.add_event_to_path(path_id, event);
    }

    /// Add memory deallocation event
    pub fn record_memory_deallocation(&self, path_id: PathId, bytes: usize, tensor_id: String) {
        if !self.is_enabled() || !self.config.trace_memory {
            return;
        }

        let event_id = {
            let mut next_id = self.next_event_id.lock();
            let id = *next_id;
            *next_id += 1;
            id
        };

        let event = TraceEvent {
            id: event_id,
            parent_id: None,
            path_id,
            event_type: EventType::MemoryDeallocation,
            operation: "memory_dealloc".to_string(),
            timestamp: Utc::now(),
            duration: None,
            memory_allocated: None,
            memory_deallocated: Some(bytes),
            input_ids: vec![tensor_id],
            output_ids: Vec::new(),
            metadata: HashMap::new(),
        };

        self.add_event_to_path(path_id, event);
    }

    /// Add event to path
    fn add_event_to_path(&self, path_id: PathId, event: TraceEvent) {
        let mut paths = self.active_paths.lock();
        if let Some(path) = paths.get_mut(&path_id) {
            // Check max events limit
            if let Some(max_events) = self.config.max_events_per_path {
                if path.events.len() >= max_events {
                    return; // Drop event if limit reached
                }
            }

            path.add_event(event);
        }
    }

    /// Get all completed paths
    pub fn completed_paths(&self) -> Vec<GradientPath> {
        self.completed_paths.lock().clone()
    }

    /// Get a specific path
    pub fn get_path(&self, path_id: PathId) -> Option<GradientPath> {
        // Check active paths first
        if let Some(path) = self.active_paths.lock().get(&path_id) {
            return Some(path.clone());
        }

        // Then check completed paths
        self.completed_paths
            .lock()
            .iter()
            .find(|p| p.id == path_id)
            .cloned()
    }

    /// Clear all traces
    pub fn clear(&self) {
        self.active_paths.lock().clear();
        self.completed_paths.lock().clear();
        self.event_buffer.lock().clear();
    }

    /// Export to Chrome Trace Format (JSON)
    pub fn export_chrome_trace(&self) -> AutogradResult<String> {
        let paths = self.completed_paths.lock();

        let mut events = Vec::new();

        for path in paths.iter() {
            for event in &path.events {
                let mut trace_event = serde_json::json!({
                    "name": event.operation,
                    "cat": format!("{:?}", event.event_type),
                    "ph": match event.event_type {
                        EventType::OperationBegin => "B",
                        EventType::OperationEnd => "E",
                        _ => "i",
                    },
                    "ts": event.timestamp.timestamp_micros(),
                    "pid": path.id,
                    "tid": event.parent_id.unwrap_or(0),
                });

                if let Some(dur) = event.duration {
                    trace_event["dur"] = serde_json::json!(dur.as_micros());
                }

                if !event.metadata.is_empty() {
                    trace_event["args"] = serde_json::json!(event.metadata);
                }

                events.push(trace_event);
            }
        }

        let trace = serde_json::json!({
            "traceEvents": events,
            "displayTimeUnit": "ms",
        });

        serde_json::to_string_pretty(&trace).map_err(|e| AutogradError::Configuration {
            parameter: "serialization".to_string(),
            value: "trace".to_string(),
            reason: format!("Failed to serialize: {}", e),
            valid_range: None,
        })
    }

    /// Export to JSON
    pub fn export_json(&self) -> AutogradResult<String> {
        let paths = self.completed_paths.lock();
        serde_json::to_string_pretty(&*paths).map_err(|e| AutogradError::Configuration {
            parameter: "serialization".to_string(),
            value: "paths".to_string(),
            reason: format!("Failed to serialize: {}", e),
            valid_range: None,
        })
    }

    /// Generate text summary
    pub fn summary(&self) -> String {
        let paths = self.completed_paths.lock();
        let mut output = String::new();

        output.push_str("=== Gradient Tracer Summary ===\n\n");
        output.push_str(&format!("Total paths: {}\n\n", paths.len()));

        for path in paths.iter() {
            output.push_str(&format!("Path #{}: {}\n", path.id, path.name));
            output.push_str(&format!("  Events: {}\n", path.events.len()));
            output.push_str(&format!("  Operations: {}\n", path.operation_count));

            if let Some(duration) = path.total_duration {
                output.push_str(&format!("  Duration: {:?}\n", duration));
            }

            output.push_str(&format!("  Peak memory: {} bytes\n", path.peak_memory));
            output.push_str(&format!(
                "  Total allocated: {} bytes\n",
                path.total_memory_allocated
            ));

            let critical = path.critical_path();
            output.push_str(&format!("  Critical path length: {}\n", critical.len()));

            let stats = path.operation_stats();
            output.push_str("  Operation statistics:\n");

            let mut sorted_ops: Vec<_> = stats.iter().collect();
            sorted_ops.sort_by(|(_, a), (_, b)| b.total_duration.cmp(&a.total_duration));

            for (op, stat) in sorted_ops.iter().take(10) {
                output.push_str(&format!(
                    "    {}: count={}, avg_duration={:?}, total_memory={} bytes\n",
                    op,
                    stat.count,
                    stat.average_duration(),
                    stat.total_memory
                ));
            }

            output.push_str("\n");
        }

        output
    }
}

/// Global gradient tracer instance
static GLOBAL_TRACER: once_cell::sync::Lazy<GradientTracer> =
    once_cell::sync::Lazy::new(|| GradientTracer::new(TracerConfig::default()));

/// Get the global gradient tracer
pub fn global_tracer() -> &'static GradientTracer {
    &GLOBAL_TRACER
}

/// RAII guard for automatic path tracing
pub struct PathTraceGuard {
    path_id: PathId,
    tracer: &'static GradientTracer,
}

impl PathTraceGuard {
    /// Create a new trace guard
    pub fn new(name: String) -> Self {
        let tracer = global_tracer();
        let path_id = tracer.start_path(name);
        Self { path_id, tracer }
    }

    /// Get the path ID
    pub fn path_id(&self) -> PathId {
        self.path_id
    }

    /// Record an operation in this path
    pub fn trace_operation<F, R>(
        &self,
        operation: String,
        parent_id: Option<TraceEventId>,
        f: F,
    ) -> AutogradResult<(R, TraceEventId)>
    where
        F: FnOnce() -> AutogradResult<R>,
    {
        self.tracer
            .trace_operation(self.path_id, operation, parent_id, f)
    }
}

impl Drop for PathTraceGuard {
    fn drop(&mut self) {
        self.tracer.end_path(self.path_id);
    }
}

/// Macro for easy gradient path tracing
#[macro_export]
macro_rules! trace_gradient_path {
    ($name:expr, $body:block) => {{
        let _guard = $crate::gradient_tracer::PathTraceGuard::new($name.to_string());
        $body
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracer_basic() {
        let tracer = GradientTracer::new(TracerConfig::default());
        tracer.enable();

        let path_id = tracer.start_path("test_path".to_string());

        let event_id = tracer.record_event(
            path_id,
            EventType::OperationBegin,
            "matmul".to_string(),
            None,
        );

        assert!(event_id > 0);

        tracer.end_path(path_id);

        let paths = tracer.completed_paths();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].name, "test_path");
    }

    #[test]
    fn test_operation_tracing() {
        let tracer = GradientTracer::new(TracerConfig::default());
        tracer.enable();

        let path_id = tracer.start_path("operation_test".to_string());

        let result = tracer.trace_operation(path_id, "add".to_string(), None, || Ok(42));

        assert!(result.is_ok());
        let (value, _event_id) = result.unwrap();
        assert_eq!(value, 42);

        tracer.end_path(path_id);

        let paths = tracer.completed_paths();
        assert_eq!(paths.len(), 1);
        assert!(paths[0].events.len() > 0);
    }

    #[test]
    fn test_memory_tracing() {
        let tracer = GradientTracer::new(TracerConfig::default());
        tracer.enable();

        let path_id = tracer.start_path("memory_test".to_string());

        tracer.record_memory_allocation(path_id, 1024, "tensor_1".to_string());
        tracer.record_memory_allocation(path_id, 2048, "tensor_2".to_string());
        tracer.record_memory_deallocation(path_id, 1024, "tensor_1".to_string());

        tracer.end_path(path_id);

        let paths = tracer.completed_paths();
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].peak_memory, 3072);
    }

    #[test]
    fn test_critical_path() {
        let tracer = GradientTracer::new(TracerConfig::default());
        tracer.enable();

        let path_id = tracer.start_path("critical_path_test".to_string());

        let e1 = tracer.record_event(path_id, EventType::OperationBegin, "op1".to_string(), None);
        let e2 = tracer.record_event(
            path_id,
            EventType::OperationBegin,
            "op2".to_string(),
            Some(e1),
        );
        let _e3 = tracer.record_event(
            path_id,
            EventType::OperationBegin,
            "op3".to_string(),
            Some(e2),
        );

        tracer.end_path(path_id);

        let paths = tracer.completed_paths();
        assert_eq!(paths.len(), 1);

        let critical = paths[0].critical_path();
        assert_eq!(critical.len(), 3);
    }

    #[test]
    fn test_chrome_trace_export() {
        let tracer = GradientTracer::new(TracerConfig::default());
        tracer.enable();

        let path_id = tracer.start_path("export_test".to_string());
        tracer.record_event(path_id, EventType::OperationBegin, "op1".to_string(), None);
        tracer.end_path(path_id);

        let json = tracer.export_chrome_trace();
        assert!(json.is_ok());

        let json_str = json.unwrap();
        assert!(json_str.contains("traceEvents"));
    }

    #[test]
    fn test_path_guard() {
        let guard = PathTraceGuard::new("guard_test".to_string());
        global_tracer().enable();

        let path_id = guard.path_id();
        let result = guard.trace_operation("test_op".to_string(), None, || Ok(100));

        assert!(result.is_ok());
        drop(guard);

        let path = global_tracer().get_path(path_id);
        assert!(path.is_some());
    }
}
