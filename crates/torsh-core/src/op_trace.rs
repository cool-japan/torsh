//! Operation Tracing System for Step-by-Step Debugging
//!
//! Provides comprehensive operation tracing capabilities for debugging complex
//! tensor operations and computational graphs. Integrates with the runtime
//! configuration system for dynamic control.
//!
//! # Features
//!
//! - **Hierarchical Tracing**: Track operations with parent-child relationships
//! - **Input/Output Capture**: Record tensor shapes, dtypes, and values
//! - **Timing Information**: Measure operation execution time
//! - **Filtering**: Enable/disable tracing for specific operation types
//! - **Breakpoints**: Pause execution at specific operations
//! - **Trace Replay**: Analyze captured traces offline
//! - **Context Preservation**: Maintain operation context through call stacks
//!
//! # Examples
//!
//! ```rust
//! use torsh_core::op_trace::{OpTracer, TraceConfig, trace_operation};
//!
//! // Enable tracing
//! OpTracer::global().set_enabled(true);
//!
//! // Trace an operation (returns Option<TraceId>)
//! let trace_id = trace_operation("matmul", |tracer| {
//!     tracer.record_input("lhs", vec![128, 256]);
//!     tracer.record_input("rhs", vec![256, 512]);
//!
//!     // ... perform operation ...
//!
//!     tracer.record_output("result", vec![128, 512]);
//! });
//!
//! // Analyze traces if tracing was enabled
//! if let Some(id) = trace_id {
//!     if let Some(trace) = OpTracer::global().get_trace(id) {
//!         println!("Operation took {:?}", trace.duration);
//!     }
//! }
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::dtype::DType;
use crate::runtime_config::RuntimeConfig;

/// Global operation tracer instance
static OP_TRACER: OnceLock<Arc<Mutex<OpTracerInternal>>> = OnceLock::new();

/// Unique trace identifier
pub type TraceId = u64;

/// Operation trace configuration
#[derive(Debug, Clone)]
pub struct TraceConfig {
    /// Enable operation tracing
    pub enabled: bool,
    /// Maximum number of traces to keep
    pub max_traces: usize,
    /// Enable input value capture (expensive)
    pub capture_values: bool,
    /// Enable output value capture (expensive)
    pub capture_outputs: bool,
    /// Enable stack trace capture
    pub capture_stack_trace: bool,
    /// Filter: only trace operations matching these patterns
    pub operation_filters: Vec<String>,
    /// Maximum depth for hierarchical tracing (0 = unlimited)
    pub max_depth: usize,
    /// Enable automatic breakpoints on errors
    pub break_on_error: bool,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_traces: 10_000,
            capture_values: false,
            capture_outputs: false,
            capture_stack_trace: false,
            operation_filters: Vec::new(),
            max_depth: 0, // unlimited
            break_on_error: true,
        }
    }
}

/// Input/output metadata for traced operations
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    /// Tensor name/identifier
    pub name: String,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: Option<DType>,
    /// Total number of elements
    pub numel: usize,
    /// Size in bytes
    pub size_bytes: usize,
    /// Whether the tensor is contiguous
    pub is_contiguous: bool,
    /// Captured values (if enabled)
    pub values: Option<Vec<f64>>,
}

impl TensorMetadata {
    /// Create tensor metadata
    pub fn new(name: impl Into<String>, shape: Vec<usize>) -> Self {
        let numel = shape.iter().product();
        Self {
            name: name.into(),
            shape,
            dtype: None,
            numel,
            size_bytes: 0,
            is_contiguous: true,
            values: None,
        }
    }

    /// Set data type
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.size_bytes = self.numel * dtype.size();
        self.dtype = Some(dtype);
        self
    }

    /// Set contiguity
    pub fn with_contiguous(mut self, is_contiguous: bool) -> Self {
        self.is_contiguous = is_contiguous;
        self
    }

    /// Set captured values
    pub fn with_values(mut self, values: Vec<f64>) -> Self {
        self.values = Some(values);
        self
    }
}

/// Operation trace record
#[derive(Debug, Clone)]
pub struct OperationTrace {
    /// Unique trace identifier
    pub id: TraceId,
    /// Parent trace ID (for hierarchical operations)
    pub parent_id: Option<TraceId>,
    /// Operation name
    pub operation: String,
    /// Operation category (e.g., "matmul", "conv", "activation")
    pub category: Option<String>,
    /// Input tensors
    pub inputs: Vec<TensorMetadata>,
    /// Output tensors
    pub outputs: Vec<TensorMetadata>,
    /// Operation start time
    pub start_time: Instant,
    /// Operation duration
    pub duration: Option<Duration>,
    /// Depth in the operation hierarchy
    pub depth: usize,
    /// Stack trace (if captured)
    pub stack_trace: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Whether this operation resulted in an error
    pub had_error: bool,
    /// Error message if any
    pub error_message: Option<String>,
}

impl OperationTrace {
    /// Create a new operation trace
    fn new(id: TraceId, parent_id: Option<TraceId>, operation: String, depth: usize) -> Self {
        Self {
            id,
            parent_id,
            operation,
            category: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            start_time: Instant::now(),
            duration: None,
            depth,
            stack_trace: None,
            metadata: HashMap::new(),
            had_error: false,
            error_message: None,
        }
    }

    /// Set operation category
    pub fn set_category(&mut self, category: impl Into<String>) {
        self.category = Some(category.into());
    }

    /// Add input tensor metadata
    pub fn add_input(&mut self, input: TensorMetadata) {
        self.inputs.push(input);
    }

    /// Add output tensor metadata
    pub fn add_output(&mut self, output: TensorMetadata) {
        self.outputs.push(output);
    }

    /// Add metadata key-value pair
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Mark operation as complete
    pub fn complete(&mut self) {
        self.duration = Some(self.start_time.elapsed());
    }

    /// Mark operation as failed
    pub fn mark_error(&mut self, error: impl Into<String>) {
        self.had_error = true;
        self.error_message = Some(error.into());
        self.complete();
    }

    /// Check if this trace matches a filter pattern
    fn matches_filter(&self, filter: &str) -> bool {
        self.operation.contains(filter)
            || self.category.as_ref().map_or(false, |c| c.contains(filter))
    }
}

/// Trace builder for recording operation details
pub struct TraceBuilder {
    trace_id: TraceId,
}

impl TraceBuilder {
    fn new(trace_id: TraceId) -> Self {
        Self { trace_id }
    }

    /// Record input tensor
    pub fn record_input(&self, name: impl Into<String>, shape: Vec<usize>) {
        let metadata = TensorMetadata::new(name, shape);
        if let Some(tracer) = OP_TRACER.get() {
            if let Ok(mut tracer) = tracer.lock() {
                if let Some(trace) = tracer.traces.get_mut(&self.trace_id) {
                    trace.add_input(metadata);
                }
            }
        }
    }

    /// Record input tensor with dtype
    pub fn record_input_with_dtype(
        &self,
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: DType,
    ) {
        let metadata = TensorMetadata::new(name, shape).with_dtype(dtype);
        if let Some(tracer) = OP_TRACER.get() {
            if let Ok(mut tracer) = tracer.lock() {
                if let Some(trace) = tracer.traces.get_mut(&self.trace_id) {
                    trace.add_input(metadata);
                }
            }
        }
    }

    /// Record output tensor
    pub fn record_output(&self, name: impl Into<String>, shape: Vec<usize>) {
        let metadata = TensorMetadata::new(name, shape);
        if let Some(tracer) = OP_TRACER.get() {
            if let Ok(mut tracer) = tracer.lock() {
                if let Some(trace) = tracer.traces.get_mut(&self.trace_id) {
                    trace.add_output(metadata);
                }
            }
        }
    }

    /// Record output tensor with dtype
    pub fn record_output_with_dtype(
        &self,
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: DType,
    ) {
        let metadata = TensorMetadata::new(name, shape).with_dtype(dtype);
        if let Some(tracer) = OP_TRACER.get() {
            if let Ok(mut tracer) = tracer.lock() {
                if let Some(trace) = tracer.traces.get_mut(&self.trace_id) {
                    trace.add_output(metadata);
                }
            }
        }
    }

    /// Add custom metadata
    pub fn add_metadata(&self, key: impl Into<String>, value: impl Into<String>) {
        if let Some(tracer) = OP_TRACER.get() {
            if let Ok(mut tracer) = tracer.lock() {
                if let Some(trace) = tracer.traces.get_mut(&self.trace_id) {
                    trace.add_metadata(key, value);
                }
            }
        }
    }

    /// Set operation category
    pub fn set_category(&self, category: impl Into<String>) {
        if let Some(tracer) = OP_TRACER.get() {
            if let Ok(mut tracer) = tracer.lock() {
                if let Some(trace) = tracer.traces.get_mut(&self.trace_id) {
                    trace.set_category(category);
                }
            }
        }
    }
}

/// Internal operation tracer state
struct OpTracerInternal {
    config: TraceConfig,
    traces: HashMap<TraceId, OperationTrace>,
    trace_order: VecDeque<TraceId>,
    next_id: TraceId,
    current_depth: usize,
    depth_stack: Vec<TraceId>,
    breakpoints: HashMap<String, bool>, // operation -> enabled
}

impl OpTracerInternal {
    fn new() -> Self {
        Self {
            config: TraceConfig::default(),
            traces: HashMap::new(),
            trace_order: VecDeque::new(),
            next_id: 1,
            current_depth: 0,
            depth_stack: Vec::new(),
            breakpoints: HashMap::new(),
        }
    }

    fn should_trace(&self, operation: &str) -> bool {
        if !self.config.enabled {
            return false;
        }

        // Check depth limit
        if self.config.max_depth > 0 && self.current_depth >= self.config.max_depth {
            return false;
        }

        // Check filters
        if !self.config.operation_filters.is_empty() {
            return self
                .config
                .operation_filters
                .iter()
                .any(|f| operation.contains(f));
        }

        true
    }

    fn start_trace(&mut self, operation: String) -> Option<TraceId> {
        if !self.should_trace(&operation) {
            return None;
        }

        let trace_id = self.next_id;
        self.next_id += 1;

        let parent_id = self.depth_stack.last().copied();
        let trace = OperationTrace::new(trace_id, parent_id, operation, self.current_depth);

        self.traces.insert(trace_id, trace);
        self.trace_order.push_back(trace_id);
        self.depth_stack.push(trace_id);
        self.current_depth += 1;

        // Enforce max traces limit
        while self.trace_order.len() > self.config.max_traces {
            if let Some(old_id) = self.trace_order.pop_front() {
                self.traces.remove(&old_id);
            }
        }

        Some(trace_id)
    }

    fn complete_trace(&mut self, trace_id: TraceId) {
        if let Some(trace) = self.traces.get_mut(&trace_id) {
            trace.complete();
        }

        // Pop from depth stack
        if self.depth_stack.last() == Some(&trace_id) {
            self.depth_stack.pop();
            if self.current_depth > 0 {
                self.current_depth -= 1;
            }
        }
    }

    fn mark_error(&mut self, trace_id: TraceId, error: String) {
        if let Some(trace) = self.traces.get_mut(&trace_id) {
            trace.mark_error(error);
        }
    }
}

/// Operation tracer - main interface
pub struct OpTracer {
    inner: Arc<Mutex<OpTracerInternal>>,
}

impl OpTracer {
    /// Get the global operation tracer
    pub fn global() -> Self {
        let inner = OP_TRACER
            .get_or_init(|| Arc::new(Mutex::new(OpTracerInternal::new())))
            .clone();
        Self { inner }
    }

    /// Create a new isolated tracer (for testing)
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(OpTracerInternal::new())),
        }
    }

    /// Enable or disable tracing
    pub fn set_enabled(&self, enabled: bool) {
        if let Ok(mut tracer) = self.inner.lock() {
            tracer.config.enabled = enabled;
        }
    }

    /// Check if tracing is enabled
    pub fn is_enabled(&self) -> bool {
        self.inner.lock().map_or(false, |t| t.config.enabled)
    }

    /// Set trace configuration
    pub fn set_config(&self, config: TraceConfig) {
        if let Ok(mut tracer) = self.inner.lock() {
            tracer.config = config;
        }
    }

    /// Get current configuration
    pub fn get_config(&self) -> TraceConfig {
        self.inner
            .lock()
            .map_or(TraceConfig::default(), |t| t.config.clone())
    }

    /// Add an operation filter
    pub fn add_filter(&self, pattern: impl Into<String>) {
        if let Ok(mut tracer) = self.inner.lock() {
            tracer.config.operation_filters.push(pattern.into());
        }
    }

    /// Clear all filters
    pub fn clear_filters(&self) {
        if let Ok(mut tracer) = self.inner.lock() {
            tracer.config.operation_filters.clear();
        }
    }

    /// Set a breakpoint on an operation
    pub fn set_breakpoint(&self, operation: impl Into<String>) {
        if let Ok(mut tracer) = self.inner.lock() {
            tracer.breakpoints.insert(operation.into(), true);
        }
    }

    /// Remove a breakpoint
    pub fn remove_breakpoint(&self, operation: &str) {
        if let Ok(mut tracer) = self.inner.lock() {
            tracer.breakpoints.remove(operation);
        }
    }

    /// Check if a breakpoint is set
    pub fn has_breakpoint(&self, operation: &str) -> bool {
        self.inner.lock().map_or(false, |t| {
            t.breakpoints.get(operation).copied().unwrap_or(false)
        })
    }

    /// Get a trace by ID
    pub fn get_trace(&self, trace_id: TraceId) -> Option<OperationTrace> {
        self.inner.lock().ok()?.traces.get(&trace_id).cloned()
    }

    /// Get all traces
    pub fn get_all_traces(&self) -> Vec<OperationTrace> {
        self.inner.lock().map_or(Vec::new(), |t| {
            t.trace_order
                .iter()
                .filter_map(|id| t.traces.get(id).cloned())
                .collect()
        })
    }

    /// Get traces matching a filter
    pub fn get_filtered_traces(&self, filter: &str) -> Vec<OperationTrace> {
        self.inner.lock().map_or(Vec::new(), |t| {
            t.trace_order
                .iter()
                .filter_map(|id| t.traces.get(id))
                .filter(|trace| trace.matches_filter(filter))
                .cloned()
                .collect()
        })
    }

    /// Clear all traces
    pub fn clear_traces(&self) {
        if let Ok(mut tracer) = self.inner.lock() {
            tracer.traces.clear();
            tracer.trace_order.clear();
        }
    }

    /// Get trace statistics
    pub fn get_statistics(&self) -> TraceStatistics {
        let tracer = match self.inner.lock() {
            Ok(t) => t,
            Err(_) => return TraceStatistics::default(),
        };

        let total_traces = tracer.traces.len();
        let total_errors = tracer.traces.values().filter(|t| t.had_error).count();

        let total_duration: Duration = tracer.traces.values().filter_map(|t| t.duration).sum();

        let operations_by_type: HashMap<String, usize> =
            tracer
                .traces
                .values()
                .fold(HashMap::new(), |mut acc, trace| {
                    *acc.entry(trace.operation.clone()).or_insert(0) += 1;
                    acc
                });

        TraceStatistics {
            total_traces,
            total_errors,
            total_duration,
            operations_by_type,
        }
    }
}

impl Default for OpTracer {
    fn default() -> Self {
        Self::global()
    }
}

/// Trace statistics
#[derive(Debug, Clone, Default)]
pub struct TraceStatistics {
    pub total_traces: usize,
    pub total_errors: usize,
    pub total_duration: Duration,
    pub operations_by_type: HashMap<String, usize>,
}

/// Trace an operation with automatic completion
///
/// # Examples
///
/// ```rust
/// use torsh_core::op_trace::trace_operation;
///
/// let trace_id = trace_operation("my_operation", |tracer| {
///     tracer.record_input("input", vec![10, 20]);
///     // ... perform operation ...
///     tracer.record_output("output", vec![10, 20]);
/// });
/// ```
pub fn trace_operation<F>(operation: impl Into<String>, f: F) -> Option<TraceId>
where
    F: FnOnce(&TraceBuilder),
{
    let operation = operation.into();
    let tracer = OpTracer::global();

    // Check runtime config
    let runtime_config = RuntimeConfig::global();
    if !runtime_config.should_collect_metrics(&operation) {
        return None;
    }

    let trace_id = {
        let mut inner = tracer.inner.lock().ok()?;
        inner.start_trace(operation.clone())?
    };

    let builder = TraceBuilder::new(trace_id);
    f(&builder);

    {
        let mut inner = tracer.inner.lock().ok()?;
        inner.complete_trace(trace_id);
    }

    Some(trace_id)
}

/// Trace an operation that may fail
pub fn trace_operation_result<F, T, E>(operation: impl Into<String>, f: F) -> Result<T, E>
where
    F: FnOnce(&TraceBuilder) -> Result<T, E>,
    E: std::fmt::Display,
{
    let operation = operation.into();
    let tracer = OpTracer::global();

    let trace_id = {
        let mut inner = tracer.inner.lock().ok().ok_or_else(|| {
            // Can't really return a meaningful error here without the user's error type
            panic!("Failed to acquire tracer lock")
        })?;
        inner.start_trace(operation.clone())
    };

    let builder = trace_id.map(TraceBuilder::new);

    let result = match builder.as_ref() {
        Some(b) => f(b),
        None => f(&TraceBuilder::new(0)), // Dummy builder
    };

    if let Some(tid) = trace_id {
        let mut inner = tracer
            .inner
            .lock()
            .ok()
            .ok_or_else(|| panic!("Failed to acquire tracer lock"))?;

        match &result {
            Ok(_) => inner.complete_trace(tid),
            Err(e) => inner.mark_error(tid, e.to_string()),
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracer_enable_disable() {
        let tracer = OpTracer::new();
        assert!(!tracer.is_enabled());

        tracer.set_enabled(true);
        assert!(tracer.is_enabled());

        tracer.set_enabled(false);
        assert!(!tracer.is_enabled());
    }

    #[test]
    fn test_trace_operation() {
        let tracer = OpTracer::new();
        tracer.set_enabled(true);

        // Manually start/complete trace
        let trace_id = {
            let mut inner = tracer.inner.lock().unwrap();
            inner.start_trace("test_op".to_string()).unwrap()
        };

        assert!(tracer.get_trace(trace_id).is_some());

        {
            let mut inner = tracer.inner.lock().unwrap();
            inner.complete_trace(trace_id);
        }

        let trace = tracer.get_trace(trace_id).unwrap();
        assert_eq!(trace.operation, "test_op");
        assert!(trace.duration.is_some());
    }

    #[test]
    fn test_trace_with_inputs_outputs() {
        let tracer = OpTracer::new();
        tracer.set_enabled(true);

        let trace_id = {
            let mut inner = tracer.inner.lock().unwrap();
            inner.start_trace("matmul".to_string()).unwrap()
        };

        // Record inputs/outputs by directly accessing the tracer
        {
            let mut inner = tracer.inner.lock().unwrap();
            if let Some(trace) = inner.traces.get_mut(&trace_id) {
                trace.add_input(TensorMetadata::new("lhs", vec![10, 20]).with_dtype(DType::F32));
                trace.add_input(TensorMetadata::new("rhs", vec![20, 30]).with_dtype(DType::F32));
                trace
                    .add_output(TensorMetadata::new("result", vec![10, 30]).with_dtype(DType::F32));
            }
        }

        {
            let mut inner = tracer.inner.lock().unwrap();
            inner.complete_trace(trace_id);
        }

        let trace = tracer.get_trace(trace_id).unwrap();
        assert_eq!(trace.inputs.len(), 2);
        assert_eq!(trace.outputs.len(), 1);
        assert_eq!(trace.inputs[0].shape, vec![10, 20]);
        assert_eq!(trace.outputs[0].shape, vec![10, 30]);
    }

    #[test]
    fn test_trace_filtering() {
        let tracer = OpTracer::new();
        tracer.set_enabled(true);
        tracer.add_filter("matmul");

        // This should be traced
        let trace_id1 = {
            let mut inner = tracer.inner.lock().unwrap();
            inner.start_trace("matmul".to_string())
        };
        assert!(trace_id1.is_some());

        // This should not be traced
        let trace_id2 = {
            let mut inner = tracer.inner.lock().unwrap();
            inner.start_trace("add".to_string())
        };
        assert!(trace_id2.is_none());
    }

    #[test]
    fn test_trace_hierarchy() {
        let tracer = OpTracer::new();
        tracer.set_enabled(true);

        let parent_id = {
            let mut inner = tracer.inner.lock().unwrap();
            inner.start_trace("parent_op".to_string()).unwrap()
        };

        let child_id = {
            let mut inner = tracer.inner.lock().unwrap();
            inner.start_trace("child_op".to_string()).unwrap()
        };

        {
            let mut inner = tracer.inner.lock().unwrap();
            inner.complete_trace(child_id);
            inner.complete_trace(parent_id);
        }

        let parent_trace = tracer.get_trace(parent_id).unwrap();
        let child_trace = tracer.get_trace(child_id).unwrap();

        assert_eq!(parent_trace.depth, 0);
        assert_eq!(child_trace.depth, 1);
        assert_eq!(child_trace.parent_id, Some(parent_id));
    }

    #[test]
    fn test_breakpoints() {
        let tracer = OpTracer::new();

        tracer.set_breakpoint("critical_op");
        assert!(tracer.has_breakpoint("critical_op"));

        tracer.remove_breakpoint("critical_op");
        assert!(!tracer.has_breakpoint("critical_op"));
    }

    #[test]
    fn test_trace_statistics() {
        let tracer = OpTracer::new();
        tracer.set_enabled(true);

        // Create some traces
        for i in 0..5 {
            let trace_id = {
                let mut inner = tracer.inner.lock().unwrap();
                inner.start_trace(format!("op_{}", i)).unwrap()
            };

            let mut inner = tracer.inner.lock().unwrap();
            inner.complete_trace(trace_id);
        }

        let stats = tracer.get_statistics();
        assert_eq!(stats.total_traces, 5);
        assert_eq!(stats.total_errors, 0);
    }

    #[test]
    fn test_error_tracing() {
        let tracer = OpTracer::new();
        tracer.set_enabled(true);

        let trace_id = {
            let mut inner = tracer.inner.lock().unwrap();
            inner.start_trace("failing_op".to_string()).unwrap()
        };

        {
            let mut inner = tracer.inner.lock().unwrap();
            inner.mark_error(trace_id, "Test error".to_string());
        }

        let trace = tracer.get_trace(trace_id).unwrap();
        assert!(trace.had_error);
        assert_eq!(trace.error_message, Some("Test error".to_string()));

        let stats = tracer.get_statistics();
        assert_eq!(stats.total_errors, 1);
    }

    #[test]
    fn test_max_traces_limit() {
        let tracer = OpTracer::new();
        let mut config = TraceConfig::default();
        config.enabled = true;
        config.max_traces = 5;
        tracer.set_config(config);

        // Create more traces than the limit
        for i in 0..10 {
            let trace_id = {
                let mut inner = tracer.inner.lock().unwrap();
                inner.start_trace(format!("op_{}", i)).unwrap()
            };

            let mut inner = tracer.inner.lock().unwrap();
            inner.complete_trace(trace_id);
        }

        let all_traces = tracer.get_all_traces();
        assert_eq!(all_traces.len(), 5); // Should only keep the last 5
    }

    #[test]
    fn test_clear_traces() {
        let tracer = OpTracer::new();
        tracer.set_enabled(true);

        let trace_id = {
            let mut inner = tracer.inner.lock().unwrap();
            inner.start_trace("test_op".to_string()).unwrap()
        };

        assert!(tracer.get_trace(trace_id).is_some());

        tracer.clear_traces();
        assert!(tracer.get_trace(trace_id).is_none());
    }
}
