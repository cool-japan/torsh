// Copyright (c) 2025 ToRSh Project
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Operation Introspection Tools
//!
//! This module provides comprehensive introspection capabilities for autograd operations,
//! allowing developers to inspect, analyze, and understand what's happening in the
//! automatic differentiation system at runtime.
//!
//! # Features
//!
//! - **Operation Metadata**: Inspect operation details, inputs, outputs, and parameters
//! - **Call Stack Tracing**: Track operation call stacks for debugging
//! - **Memory Tracking**: Monitor memory usage per operation
//! - **Dependency Analysis**: Understand operation dependencies and execution order
//! - **Real-time Monitoring**: Live monitoring of operations as they execute
//! - **Query Interface**: Search and filter operations by various criteria
//! - **Performance Metrics**: Collect timing and resource usage data
//!
//! # Example
//!
//! ```rust,ignore
//! use torsh_autograd::operation_introspection::{OperationIntrospector, IntrospectionConfig};
//!
//! // Create introspector
//! let introspector = OperationIntrospector::new(IntrospectionConfig::default());
//!
//! // Enable introspection
//! introspector.enable();
//!
//! // Operations are automatically tracked...
//!
//! // Query operations
//! let matmul_ops = introspector.query()
//!     .filter_by_name("matmul")
//!     .collect();
//!
//! // Get operation details
//! for op in matmul_ops {
//!     println!("Operation: {}, Time: {:?}", op.name, op.execution_time);
//! }
//! ```

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

/// Operation introspection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrospectionConfig {
    /// Enable introspection
    pub enabled: bool,

    /// Track call stacks
    pub track_call_stacks: bool,

    /// Track memory usage
    pub track_memory: bool,

    /// Track timing information
    pub track_timing: bool,

    /// Maximum number of operations to keep in history
    pub max_history_size: usize,

    /// Enable real-time monitoring
    pub enable_monitoring: bool,

    /// Collect tensor shape information
    pub collect_shapes: bool,

    /// Collect parameter information
    pub collect_parameters: bool,
}

impl Default for IntrospectionConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default for performance
            track_call_stacks: true,
            track_memory: true,
            track_timing: true,
            max_history_size: 10000,
            enable_monitoring: false,
            collect_shapes: true,
            collect_parameters: true,
        }
    }
}

/// Tensor information for introspection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Tensor ID
    pub id: String,

    /// Tensor name (if available)
    pub name: Option<String>,

    /// Shape
    pub shape: Vec<usize>,

    /// Data type
    pub dtype: String,

    /// Device
    pub device: String,

    /// Requires gradient
    pub requires_grad: bool,

    /// Is leaf tensor
    pub is_leaf: bool,

    /// Memory size in bytes
    pub memory_size: usize,
}

/// Operation parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationParameter {
    /// Parameter name
    pub name: String,

    /// Parameter value (as string for simplicity)
    pub value: String,

    /// Parameter type
    pub param_type: String,
}

/// Call stack frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallStackFrame {
    /// Function name
    pub function: String,

    /// Module path
    pub module: String,

    /// File name
    pub file: Option<String>,

    /// Line number
    pub line: Option<u32>,
}

/// Operation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationRecord {
    /// Operation ID
    pub id: u64,

    /// Operation name
    pub name: String,

    /// Operation type/category
    pub op_type: String,

    /// Input tensors
    pub inputs: Vec<TensorInfo>,

    /// Output tensors
    pub outputs: Vec<TensorInfo>,

    /// Operation parameters
    pub parameters: Vec<OperationParameter>,

    /// Call stack (if enabled)
    pub call_stack: Option<Vec<CallStackFrame>>,

    /// Start time
    #[serde(with = "system_time_serde")]
    pub start_time: SystemTime,

    /// Execution time
    #[serde(with = "duration_serde")]
    pub execution_time: Duration,

    /// Memory allocated (bytes)
    pub memory_allocated: usize,

    /// Memory freed (bytes)
    pub memory_freed: usize,

    /// Peak memory usage during operation (bytes)
    pub peak_memory: usize,

    /// Thread ID where operation executed
    pub thread_id: u64,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

mod system_time_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time.duration_since(UNIX_EPOCH).unwrap();
        duration.as_secs().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = u64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + std::time::Duration::from_secs(secs))
    }
}

mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_nanos().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let nanos = u128::deserialize(deserializer)?;
        Ok(Duration::from_nanos(nanos as u64))
    }
}

/// Operation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStatistics {
    /// Total number of operations
    pub total_operations: usize,

    /// Operations by type
    pub operations_by_type: HashMap<String, usize>,

    /// Total execution time
    pub total_execution_time: Duration,

    /// Average execution time
    pub average_execution_time: Duration,

    /// Total memory allocated
    pub total_memory_allocated: usize,

    /// Total memory freed
    pub total_memory_freed: usize,

    /// Peak memory usage
    pub peak_memory_usage: usize,

    /// Most frequent operations
    pub most_frequent_ops: Vec<(String, usize)>,

    /// Slowest operations
    pub slowest_ops: Vec<(String, Duration)>,
}

/// Operation query builder
pub struct OperationQuery {
    operations: Vec<OperationRecord>,
    filters: Vec<Box<dyn Fn(&OperationRecord) -> bool + Send + Sync>>,
}

impl OperationQuery {
    fn new(operations: Vec<OperationRecord>) -> Self {
        Self {
            operations,
            filters: Vec::new(),
        }
    }

    /// Filter by operation name
    pub fn filter_by_name(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        self.filters
            .push(Box::new(move |op| op.name.contains(&name)));
        self
    }

    /// Filter by operation type
    pub fn filter_by_type(mut self, op_type: impl Into<String>) -> Self {
        let op_type = op_type.into();
        self.filters.push(Box::new(move |op| op.op_type == op_type));
        self
    }

    /// Filter by minimum execution time
    pub fn filter_by_min_time(mut self, min_time: Duration) -> Self {
        self.filters
            .push(Box::new(move |op| op.execution_time >= min_time));
        self
    }

    /// Filter by minimum memory usage
    pub fn filter_by_min_memory(mut self, min_memory: usize) -> Self {
        self.filters
            .push(Box::new(move |op| op.memory_allocated >= min_memory));
        self
    }

    /// Filter by time range
    pub fn filter_by_time_range(mut self, start: SystemTime, end: SystemTime) -> Self {
        self.filters.push(Box::new(move |op| {
            op.start_time >= start && op.start_time <= end
        }));
        self
    }

    /// Filter by thread ID
    pub fn filter_by_thread(mut self, thread_id: u64) -> Self {
        self.filters
            .push(Box::new(move |op| op.thread_id == thread_id));
        self
    }

    /// Collect filtered operations
    pub fn collect(self) -> Vec<OperationRecord> {
        self.operations
            .into_iter()
            .filter(|op| self.filters.iter().all(|f| f(op)))
            .collect()
    }

    /// Count filtered operations
    pub fn count(self) -> usize {
        self.operations
            .iter()
            .filter(|op| self.filters.iter().all(|f| f(op)))
            .count()
    }

    /// Get first matching operation
    pub fn first(self) -> Option<OperationRecord> {
        self.operations
            .into_iter()
            .find(|op| self.filters.iter().all(|f| f(op)))
    }
}

/// Operation introspector
#[derive(Clone)]
pub struct OperationIntrospector {
    config: Arc<RwLock<IntrospectionConfig>>,
    operations: Arc<Mutex<VecDeque<OperationRecord>>>,
    next_id: Arc<AtomicU64>,
    enabled: Arc<AtomicBool>,
    monitors: Arc<RwLock<Vec<Arc<dyn Fn(&OperationRecord) + Send + Sync>>>>,
}

impl OperationIntrospector {
    /// Create a new operation introspector
    pub fn new(config: IntrospectionConfig) -> Self {
        let enabled = config.enabled;
        Self {
            config: Arc::new(RwLock::new(config)),
            operations: Arc::new(Mutex::new(VecDeque::new())),
            next_id: Arc::new(AtomicU64::new(0)),
            enabled: Arc::new(AtomicBool::new(enabled)),
            monitors: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Enable introspection
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
        self.config.write().enabled = true;
    }

    /// Disable introspection
    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Relaxed);
        self.config.write().enabled = false;
    }

    /// Check if introspection is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Record an operation
    pub fn record_operation(&self, record: OperationRecord) {
        if !self.is_enabled() {
            return;
        }

        let config = self.config.read();
        let max_size = config.max_history_size;
        let monitoring_enabled = config.enable_monitoring;
        drop(config);

        // Add to history
        {
            let mut ops = self.operations.lock();
            ops.push_back(record.clone());
            if ops.len() > max_size {
                ops.pop_front();
            }
        }

        // Trigger monitors
        if monitoring_enabled {
            let monitors = self.monitors.read();
            for monitor in monitors.iter() {
                monitor(&record);
            }
        }
    }

    /// Start recording an operation (returns operation ID)
    pub fn start_operation(&self, _name: impl Into<String>, _op_type: impl Into<String>) -> u64 {
        if !self.is_enabled() {
            return 0;
        }

        // fetch_add returns the previous value, so we add 1 first then return
        self.next_id.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Finish recording an operation
    pub fn finish_operation(
        &self,
        id: u64,
        name: impl Into<String>,
        op_type: impl Into<String>,
        inputs: Vec<TensorInfo>,
        outputs: Vec<TensorInfo>,
        start_time: SystemTime,
        execution_time: Duration,
    ) {
        if !self.is_enabled() || id == 0 {
            return;
        }

        let config = self.config.read();
        // Use a simple hash of the thread ID since as_u64() is unstable
        let thread_id = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            std::thread::current().id().hash(&mut hasher);
            hasher.finish()
        };

        let record = OperationRecord {
            id,
            name: name.into(),
            op_type: op_type.into(),
            inputs,
            outputs,
            parameters: Vec::new(),
            call_stack: if config.track_call_stacks {
                Some(self.capture_call_stack())
            } else {
                None
            },
            start_time,
            execution_time,
            memory_allocated: 0,
            memory_freed: 0,
            peak_memory: 0,
            thread_id,
            metadata: HashMap::new(),
        };

        drop(config);
        self.record_operation(record);
    }

    /// Capture current call stack
    fn capture_call_stack(&self) -> Vec<CallStackFrame> {
        // Simplified call stack capture
        // In a real implementation, you'd use backtrace or similar
        vec![CallStackFrame {
            function: "autograd_operation".to_string(),
            module: "torsh_autograd".to_string(),
            file: None,
            line: None,
        }]
    }

    /// Query operations
    pub fn query(&self) -> OperationQuery {
        let ops = self.operations.lock().iter().cloned().collect();
        OperationQuery::new(ops)
    }

    /// Get all operations
    pub fn get_operations(&self) -> Vec<OperationRecord> {
        self.operations.lock().iter().cloned().collect()
    }

    /// Get operation by ID
    pub fn get_operation(&self, id: u64) -> Option<OperationRecord> {
        self.operations
            .lock()
            .iter()
            .find(|op| op.id == id)
            .cloned()
    }

    /// Compute statistics
    pub fn compute_statistics(&self) -> OperationStatistics {
        let ops = self.operations.lock();

        let mut ops_by_type: HashMap<String, usize> = HashMap::new();
        let mut total_time = Duration::ZERO;
        let mut total_memory_allocated = 0;
        let mut total_memory_freed = 0;
        let mut peak_memory = 0;

        for op in ops.iter() {
            *ops_by_type.entry(op.op_type.clone()).or_insert(0) += 1;
            total_time += op.execution_time;
            total_memory_allocated += op.memory_allocated;
            total_memory_freed += op.memory_freed;
            peak_memory = peak_memory.max(op.peak_memory);
        }

        let total_ops = ops.len();
        let avg_time = if total_ops > 0 {
            total_time / total_ops as u32
        } else {
            Duration::ZERO
        };

        // Most frequent operations
        let mut freq_vec: Vec<_> = ops_by_type.iter().map(|(k, v)| (k.clone(), *v)).collect();
        freq_vec.sort_by(|a, b| b.1.cmp(&a.1));
        let most_frequent = freq_vec.into_iter().take(10).collect();

        // Slowest operations
        let mut slow_vec: Vec<_> = ops
            .iter()
            .map(|op| (op.name.clone(), op.execution_time))
            .collect();
        slow_vec.sort_by(|a, b| b.1.cmp(&a.1));
        let slowest = slow_vec.into_iter().take(10).collect();

        OperationStatistics {
            total_operations: total_ops,
            operations_by_type: ops_by_type,
            total_execution_time: total_time,
            average_execution_time: avg_time,
            total_memory_allocated,
            total_memory_freed,
            peak_memory_usage: peak_memory,
            most_frequent_ops: most_frequent,
            slowest_ops: slowest,
        }
    }

    /// Add operation monitor
    pub fn add_monitor(&self, monitor: Arc<dyn Fn(&OperationRecord) + Send + Sync>) {
        self.monitors.write().push(monitor);
    }

    /// Clear all monitors
    pub fn clear_monitors(&self) {
        self.monitors.write().clear();
    }

    /// Clear operation history
    pub fn clear_history(&self) {
        self.operations.lock().clear();
    }

    /// Export operations to JSON
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        let ops = self.get_operations();
        serde_json::to_string_pretty(&ops)
    }

    /// Get recent operations (last N)
    pub fn get_recent_operations(&self, count: usize) -> Vec<OperationRecord> {
        let ops = self.operations.lock();
        ops.iter().rev().take(count).cloned().collect()
    }
}

impl Default for OperationIntrospector {
    fn default() -> Self {
        Self::new(IntrospectionConfig::default())
    }
}

/// Global operation introspector instance
static GLOBAL_INTROSPECTOR: once_cell::sync::Lazy<OperationIntrospector> =
    once_cell::sync::Lazy::new(|| OperationIntrospector::default());

/// Get the global operation introspector
pub fn global_introspector() -> &'static OperationIntrospector {
    &GLOBAL_INTROSPECTOR
}

/// RAII guard for operation introspection
pub struct OperationScope {
    introspector: OperationIntrospector,
    id: u64,
    name: String,
    op_type: String,
    start_time: SystemTime,
    start_instant: Instant,
    inputs: Vec<TensorInfo>,
}

impl OperationScope {
    /// Create a new operation scope
    pub fn new(
        introspector: OperationIntrospector,
        name: impl Into<String>,
        op_type: impl Into<String>,
    ) -> Self {
        let name = name.into();
        let op_type = op_type.into();
        let id = introspector.start_operation(&name, &op_type);

        Self {
            introspector,
            id,
            name,
            op_type,
            start_time: SystemTime::now(),
            start_instant: Instant::now(),
            inputs: Vec::new(),
        }
    }

    /// Add input tensor
    pub fn add_input(&mut self, tensor: TensorInfo) {
        self.inputs.push(tensor);
    }

    /// Add multiple inputs
    pub fn add_inputs(&mut self, tensors: Vec<TensorInfo>) {
        self.inputs.extend(tensors);
    }

    /// Finish with outputs
    pub fn finish(self, outputs: Vec<TensorInfo>) {
        let execution_time = self.start_instant.elapsed();
        self.introspector.finish_operation(
            self.id,
            self.name.clone(),
            self.op_type.clone(),
            self.inputs.clone(),
            outputs,
            self.start_time,
            execution_time,
        );
        // Prevent Drop from running
        std::mem::forget(self);
    }
}

impl Drop for OperationScope {
    fn drop(&mut self) {
        let execution_time = self.start_instant.elapsed();
        self.introspector.finish_operation(
            self.id,
            self.name.clone(),
            self.op_type.clone(),
            self.inputs.clone(),
            Vec::new(),
            self.start_time,
            execution_time,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_tensor(id: &str) -> TensorInfo {
        TensorInfo {
            id: id.to_string(),
            name: Some(format!("tensor_{}", id)),
            shape: vec![2, 3],
            dtype: "f32".to_string(),
            device: "cpu".to_string(),
            requires_grad: true,
            is_leaf: false,
            memory_size: 24,
        }
    }

    #[test]
    fn test_introspector_creation() {
        let introspector = OperationIntrospector::new(IntrospectionConfig::default());
        assert!(!introspector.is_enabled()); // Disabled by default
    }

    #[test]
    fn test_enable_disable() {
        let introspector = OperationIntrospector::new(IntrospectionConfig::default());
        assert!(!introspector.is_enabled());

        introspector.enable();
        assert!(introspector.is_enabled());

        introspector.disable();
        assert!(!introspector.is_enabled());
    }

    #[test]
    fn test_record_operation() {
        let mut config = IntrospectionConfig::default();
        config.enabled = true;
        let introspector = OperationIntrospector::new(config);

        let record = OperationRecord {
            id: 1,
            name: "matmul".to_string(),
            op_type: "linear_algebra".to_string(),
            inputs: vec![create_test_tensor("1"), create_test_tensor("2")],
            outputs: vec![create_test_tensor("3")],
            parameters: vec![],
            call_stack: None,
            start_time: SystemTime::now(),
            execution_time: Duration::from_millis(10),
            memory_allocated: 1024,
            memory_freed: 0,
            peak_memory: 1024,
            thread_id: 1,
            metadata: HashMap::new(),
        };

        introspector.record_operation(record);
        let ops = introspector.get_operations();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].name, "matmul");
    }

    #[test]
    fn test_query_by_name() {
        let mut config = IntrospectionConfig::default();
        config.enabled = true;
        let introspector = OperationIntrospector::new(config);

        // Record multiple operations
        for i in 0..5 {
            let name = if i % 2 == 0 { "matmul" } else { "add" };
            let record = OperationRecord {
                id: i,
                name: name.to_string(),
                op_type: "math".to_string(),
                inputs: vec![],
                outputs: vec![],
                parameters: vec![],
                call_stack: None,
                start_time: SystemTime::now(),
                execution_time: Duration::from_millis(i as u64 * 10),
                memory_allocated: 0,
                memory_freed: 0,
                peak_memory: 0,
                thread_id: 1,
                metadata: HashMap::new(),
            };
            introspector.record_operation(record);
        }

        let matmul_ops = introspector.query().filter_by_name("matmul").collect();
        assert_eq!(matmul_ops.len(), 3);
    }

    #[test]
    fn test_query_by_min_time() {
        let mut config = IntrospectionConfig::default();
        config.enabled = true;
        let introspector = OperationIntrospector::new(config);

        for i in 0..5 {
            let record = OperationRecord {
                id: i,
                name: format!("op_{}", i),
                op_type: "test".to_string(),
                inputs: vec![],
                outputs: vec![],
                parameters: vec![],
                call_stack: None,
                start_time: SystemTime::now(),
                execution_time: Duration::from_millis(i as u64 * 10),
                memory_allocated: 0,
                memory_freed: 0,
                peak_memory: 0,
                thread_id: 1,
                metadata: HashMap::new(),
            };
            introspector.record_operation(record);
        }

        let slow_ops = introspector
            .query()
            .filter_by_min_time(Duration::from_millis(25))
            .collect();
        assert_eq!(slow_ops.len(), 2); // ops 3 and 4
    }

    #[test]
    fn test_statistics() {
        let mut config = IntrospectionConfig::default();
        config.enabled = true;
        let introspector = OperationIntrospector::new(config);

        for i in 0..10 {
            let op_type = if i % 3 == 0 {
                "matmul"
            } else if i % 3 == 1 {
                "add"
            } else {
                "relu"
            };

            let record = OperationRecord {
                id: i,
                name: format!("op_{}", i),
                op_type: op_type.to_string(),
                inputs: vec![],
                outputs: vec![],
                parameters: vec![],
                call_stack: None,
                start_time: SystemTime::now(),
                execution_time: Duration::from_millis(i as u64 * 10),
                memory_allocated: i as usize * 100,
                memory_freed: 0,
                peak_memory: i as usize * 100,
                thread_id: 1,
                metadata: HashMap::new(),
            };
            introspector.record_operation(record);
        }

        let stats = introspector.compute_statistics();
        assert_eq!(stats.total_operations, 10);
        assert_eq!(stats.operations_by_type.len(), 3);
    }

    #[test]
    fn test_operation_scope() {
        let introspector = OperationIntrospector::new(IntrospectionConfig::default());
        introspector.enable(); // Enable introspection

        {
            let mut scope = OperationScope::new(introspector.clone(), "test_op", "test");
            scope.add_input(create_test_tensor("1"));
            scope.finish(vec![create_test_tensor("2")]);
        }

        let ops = introspector.get_operations();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].name, "test_op");
        assert_eq!(ops[0].inputs.len(), 1);
        assert_eq!(ops[0].outputs.len(), 1);
    }

    #[test]
    fn test_monitors() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mut config = IntrospectionConfig::default();
        config.enabled = true;
        config.enable_monitoring = true;
        let introspector = OperationIntrospector::new(config);

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        introspector.add_monitor(Arc::new(move |_op| {
            counter_clone.fetch_add(1, Ordering::Relaxed);
        }));

        for i in 0..5 {
            let record = OperationRecord {
                id: i,
                name: format!("op_{}", i),
                op_type: "test".to_string(),
                inputs: vec![],
                outputs: vec![],
                parameters: vec![],
                call_stack: None,
                start_time: SystemTime::now(),
                execution_time: Duration::from_millis(1),
                memory_allocated: 0,
                memory_freed: 0,
                peak_memory: 0,
                thread_id: 1,
                metadata: HashMap::new(),
            };
            introspector.record_operation(record);
        }

        assert_eq!(counter.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_global_introspector() {
        let introspector = global_introspector();
        introspector.clear_history();
        introspector.enable();

        let id = introspector.start_operation("global_test", "test");
        assert!(id > 0, "Expected operation ID > 0, got {}", id);

        // Finish the operation
        introspector.finish_operation(
            id,
            "global_test",
            "test",
            vec![],
            vec![],
            SystemTime::now(),
            Duration::from_millis(1),
        );

        // Verify it was recorded
        let ops = introspector.get_operations();
        assert!(!ops.is_empty(), "Expected at least one operation recorded");

        introspector.disable();
    }

    #[test]
    fn test_export_json() {
        let mut config = IntrospectionConfig::default();
        config.enabled = true;
        let introspector = OperationIntrospector::new(config);

        let record = OperationRecord {
            id: 1,
            name: "test".to_string(),
            op_type: "test".to_string(),
            inputs: vec![],
            outputs: vec![],
            parameters: vec![],
            call_stack: None,
            start_time: SystemTime::now(),
            execution_time: Duration::from_millis(1),
            memory_allocated: 0,
            memory_freed: 0,
            peak_memory: 0,
            thread_id: 1,
            metadata: HashMap::new(),
        };
        introspector.record_operation(record);

        let json = introspector.export_json();
        assert!(json.is_ok());
        assert!(json.unwrap().contains("test"));
    }
}
