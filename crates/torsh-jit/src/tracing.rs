//! Graph capture and tracing for JIT compilation
//!
//! This module provides automatic graph capture by tracing function execution,
//! similar to PyTorch's torch.jit.trace functionality.

use crate::graph::{ComputationGraph, Edge, Node, NodeId, Operation};
use crate::{JitError, JitResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use torsh_core::{DType, DeviceType, Shape};

/// Tracer for capturing computation graphs
#[derive(Debug)]
pub struct GraphTracer {
    /// Current graph being built
    graph: ComputationGraph,

    /// Mapping from traced values to nodes
    value_to_node: HashMap<TracedValueId, NodeId>,

    /// Next value ID
    next_value_id: TracedValueId,

    /// Whether tracing is active
    active: bool,

    /// Stack of operation contexts
    op_stack: Vec<OpContext>,

    /// Profiling data
    profiler: Option<Profiler>,
}

/// Profiler for collecting execution statistics
#[derive(Debug)]
pub struct Profiler {
    /// Operation timings
    pub op_timings: HashMap<String, Duration>,

    /// Memory usage tracking
    pub memory_usage: HashMap<String, usize>,

    /// Operation counts
    pub op_counts: HashMap<String, usize>,

    /// Start time for current operation
    current_op_start: Option<Instant>,

    /// Current operation name
    current_op_name: Option<String>,
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            op_timings: HashMap::new(),
            memory_usage: HashMap::new(),
            op_counts: HashMap::new(),
            current_op_start: None,
            current_op_name: None,
        }
    }

    pub fn start_op(&mut self, op_name: String) {
        self.current_op_start = Some(Instant::now());
        self.current_op_name = Some(op_name);
    }

    pub fn end_op(&mut self) {
        if let (Some(start), Some(name)) =
            (self.current_op_start.take(), self.current_op_name.take())
        {
            let duration = start.elapsed();
            *self
                .op_timings
                .entry(name.clone())
                .or_insert(Duration::ZERO) += duration;
            *self.op_counts.entry(name).or_insert(0) += 1;
        }
    }

    pub fn record_memory_usage(&mut self, op_name: String, bytes: usize) {
        self.memory_usage.insert(op_name, bytes);
    }

    pub fn get_total_time(&self) -> Duration {
        self.op_timings.values().sum()
    }

    pub fn get_slowest_ops(&self, count: usize) -> Vec<(String, Duration)> {
        let mut ops: Vec<_> = self
            .op_timings
            .iter()
            .map(|(name, duration)| (name.clone(), *duration))
            .collect();
        ops.sort_by(|a, b| b.1.cmp(&a.1));
        ops.into_iter().take(count).collect()
    }
}

/// Traced value identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TracedValueId(u64);

/// Operation context during tracing
#[derive(Debug, Clone)]
struct OpContext {
    /// Operation being traced
    #[allow(dead_code)]
    op: Operation,

    /// Input values
    #[allow(dead_code)]
    inputs: Vec<TracedValueId>,

    /// Operation metadata
    #[allow(dead_code)]
    metadata: OpMetadata,
}

/// Operation metadata
#[derive(Debug, Clone)]
struct OpMetadata {
    /// Source location (for debugging)
    #[allow(dead_code)]
    source_location: Option<String>,

    /// Operation name
    #[allow(dead_code)]
    name: Option<String>,

    /// Additional attributes
    #[allow(dead_code)]
    attributes: HashMap<String, String>,
}

/// Traced value representing a tensor in the computation
#[derive(Debug, Clone)]
pub struct TracedValue {
    /// Unique identifier
    pub id: TracedValueId,

    /// Shape of the tensor
    pub shape: Shape,

    /// Data type
    pub dtype: DType,

    /// Device placement
    pub device: DeviceType,

    /// Whether this value requires gradient
    pub requires_grad: bool,

    /// Reference to the tracer
    tracer: Arc<Mutex<GraphTracer>>,
}

impl GraphTracer {
    /// Create a new graph tracer
    pub fn new() -> Self {
        Self {
            graph: ComputationGraph::new(),
            value_to_node: HashMap::new(),
            next_value_id: TracedValueId(0),
            active: false,
            op_stack: Vec::new(),
            profiler: None,
        }
    }

    /// Create a new graph tracer with profiling enabled
    pub fn new_with_profiling() -> Self {
        Self {
            graph: ComputationGraph::new(),
            value_to_node: HashMap::new(),
            next_value_id: TracedValueId(0),
            active: false,
            op_stack: Vec::new(),
            profiler: Some(Profiler::new()),
        }
    }

    /// Enable profiling
    pub fn enable_profiling(&mut self) {
        self.profiler = Some(Profiler::new());
    }

    /// Disable profiling
    pub fn disable_profiling(&mut self) {
        self.profiler = None;
    }

    /// Get profiling data
    pub fn get_profiler(&self) -> Option<&Profiler> {
        self.profiler.as_ref()
    }

    /// Start tracing
    pub fn start_tracing(&mut self) {
        self.active = true;
        self.graph = ComputationGraph::new();
        self.value_to_node.clear();
        self.next_value_id = TracedValueId(0);
        self.op_stack.clear();
    }

    /// Stop tracing and return the captured graph
    pub fn stop_tracing(&mut self) -> ComputationGraph {
        self.active = false;
        std::mem::take(&mut self.graph)
    }

    /// Check if tracing is active
    pub fn is_tracing(&self) -> bool {
        self.active
    }

    /// Create a new traced value ID
    fn new_value_id(&mut self) -> TracedValueId {
        let id = self.next_value_id;
        self.next_value_id.0 += 1;
        id
    }

    /// Create an input value
    pub fn create_input(
        &mut self,
        name: impl Into<String>,
        shape: Shape,
        dtype: DType,
        device: DeviceType,
    ) -> TracedValueId {
        if !self.active {
            return TracedValueId(0);
        }

        let value_id = self.new_value_id();

        let mut node = Node::new(Operation::Input, name.into());
        node = node
            .with_output_shapes(vec![Some(shape)])
            .with_dtypes(vec![dtype])
            .with_device(device);
        node.inputs = vec![];
        node.is_output = false;

        let node_id = self.graph.add_node(node);
        self.graph.add_input(node_id);
        self.value_to_node.insert(value_id, node_id);

        value_id
    }

    /// Record an operation
    pub fn record_operation(
        &mut self,
        op: Operation,
        inputs: &[TracedValueId],
        output_shape: Shape,
        output_dtype: DType,
        output_device: DeviceType,
    ) -> JitResult<TracedValueId> {
        if !self.active {
            return Ok(TracedValueId(0));
        }

        let op_name = format!("{:?}", op);

        // Start profiling if enabled
        if let Some(ref mut profiler) = self.profiler {
            profiler.start_op(op_name.clone());
        }

        let output_id = self.new_value_id();

        // Create output node
        let mut node = Node::new(op.clone(), format!("{:?}_{}", op, output_id.0));
        node = node
            .with_output_shapes(vec![Some(output_shape.clone())])
            .with_dtypes(vec![output_dtype])
            .with_device(output_device);
        node.inputs = vec![];
        node.is_output = false;

        let node_id = self.graph.add_node(node);

        // Connect inputs
        for (i, &input_id) in inputs.iter().enumerate() {
            if let Some(&input_node_id) = self.value_to_node.get(&input_id) {
                self.graph.add_edge(
                    input_node_id,
                    node_id,
                    Edge {
                        src_output: 0,
                        dst_input: i,
                    },
                );
            }
        }

        self.value_to_node.insert(output_id, node_id);

        // Record memory usage estimate
        if let Some(ref mut profiler) = self.profiler {
            let memory_bytes = output_shape.numel() * dtype_size_bytes(output_dtype);
            profiler.record_memory_usage(op_name.clone(), memory_bytes);
            profiler.end_op();
        }

        Ok(output_id)
    }

    /// Mark a value as output
    pub fn mark_output(&mut self, value_id: TracedValueId) {
        if let Some(&node_id) = self.value_to_node.get(&value_id) {
            self.graph.add_output(node_id);
        }
    }

    /// Get the current graph (for inspection)
    pub fn get_graph(&self) -> &ComputationGraph {
        &self.graph
    }
}

impl TracedValue {
    /// Create a new traced value
    pub fn new(
        shape: Shape,
        dtype: DType,
        device: DeviceType,
        requires_grad: bool,
        tracer: Arc<Mutex<GraphTracer>>,
    ) -> Self {
        let id = {
            let mut t = tracer.lock().expect("lock should not be poisoned");
            t.new_value_id()
        };

        Self {
            id,
            shape,
            dtype,
            device,
            requires_grad,
            tracer,
        }
    }

    /// Create an input traced value
    pub fn input(
        name: impl Into<String>,
        shape: Shape,
        dtype: DType,
        device: DeviceType,
        tracer: Arc<Mutex<GraphTracer>>,
    ) -> Self {
        let id = {
            let mut t = tracer.lock().expect("lock should not be poisoned");
            t.create_input(name, shape.clone(), dtype, device)
        };

        Self {
            id,
            shape,
            dtype,
            device,
            requires_grad: false,
            tracer,
        }
    }

    /// Perform a unary operation
    pub fn unary_op(&self, op: Operation) -> JitResult<TracedValue> {
        let output_id = {
            let mut tracer = self.tracer.lock().expect("lock should not be poisoned");
            tracer.record_operation(op, &[self.id], self.shape.clone(), self.dtype, self.device)?
        };

        Ok(TracedValue {
            id: output_id,
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device,
            requires_grad: self.requires_grad,
            tracer: self.tracer.clone(),
        })
    }

    /// Perform a binary operation
    pub fn binary_op(&self, other: &TracedValue, op: Operation) -> JitResult<TracedValue> {
        // Determine output shape (simplified broadcasting)
        let output_shape = if self.shape.dims() == other.shape.dims() {
            self.shape.clone()
        } else {
            // Simplified: use larger shape
            if self.shape.numel() >= other.shape.numel() {
                self.shape.clone()
            } else {
                other.shape.clone()
            }
        };

        // Determine output type (use higher precision)
        let output_dtype = match (self.dtype, other.dtype) {
            (DType::F64, _) | (_, DType::F64) => DType::F64,
            (DType::F32, _) | (_, DType::F32) => DType::F32,
            _ => self.dtype,
        };

        let output_id = {
            let mut tracer = self.tracer.lock().expect("lock should not be poisoned");
            tracer.record_operation(
                op,
                &[self.id, other.id],
                output_shape.clone(),
                output_dtype,
                self.device,
            )?
        };

        Ok(TracedValue {
            id: output_id,
            shape: output_shape,
            dtype: output_dtype,
            device: self.device,
            requires_grad: self.requires_grad || other.requires_grad,
            tracer: self.tracer.clone(),
        })
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &TracedValue) -> JitResult<TracedValue> {
        // Compute output shape for matrix multiplication
        let self_dims = self.shape.dims();
        let other_dims = other.shape.dims();

        if self_dims.len() < 2 || other_dims.len() < 2 {
            return Err(JitError::GraphError(
                "Matrix multiplication requires at least 2D tensors".to_string(),
            ));
        }

        let m = self_dims[self_dims.len() - 2];
        let k1 = self_dims[self_dims.len() - 1];
        let k2 = other_dims[other_dims.len() - 2];
        let n = other_dims[other_dims.len() - 1];

        if k1 != k2 {
            return Err(JitError::GraphError(format!(
                "Matrix multiplication dimension mismatch: {} != {}",
                k1, k2
            )));
        }

        // Result shape: batch dims + [m, n]
        let mut output_dims = self_dims[..self_dims.len() - 2].to_vec();
        output_dims.push(m);
        output_dims.push(n);

        let output_shape = Shape::new(output_dims);
        let output_dtype = match (self.dtype, other.dtype) {
            (DType::F64, _) | (_, DType::F64) => DType::F64,
            (DType::F32, _) | (_, DType::F32) => DType::F32,
            _ => self.dtype,
        };

        let output_id = {
            let mut tracer = self.tracer.lock().expect("lock should not be poisoned");
            tracer.record_operation(
                Operation::MatMul,
                &[self.id, other.id],
                output_shape.clone(),
                output_dtype,
                self.device,
            )?
        };

        Ok(TracedValue {
            id: output_id,
            shape: output_shape,
            dtype: output_dtype,
            device: self.device,
            requires_grad: self.requires_grad || other.requires_grad,
            tracer: self.tracer.clone(),
        })
    }

    /// Reshape operation
    pub fn reshape(&self, new_shape: &[isize]) -> JitResult<TracedValue> {
        let output_shape = Shape::new(
            new_shape
                .iter()
                .map(|&dim| {
                    if dim == -1 {
                        // Infer dimension
                        let total_elements = self.shape.numel();
                        let known_elements: usize = new_shape
                            .iter()
                            .filter(|&&d| d != -1)
                            .map(|&d| d as usize)
                            .product();
                        if known_elements == 0 {
                            total_elements
                        } else {
                            total_elements / known_elements
                        }
                    } else {
                        dim as usize
                    }
                })
                .collect(),
        );

        let output_id = {
            let mut tracer = self.tracer.lock().expect("lock should not be poisoned");
            tracer.record_operation(
                Operation::Reshape {
                    shape: new_shape.to_vec(),
                },
                &[self.id],
                output_shape.clone(),
                self.dtype,
                self.device,
            )?
        };

        Ok(TracedValue {
            id: output_id,
            shape: output_shape,
            dtype: self.dtype,
            device: self.device,
            requires_grad: self.requires_grad,
            tracer: self.tracer.clone(),
        })
    }

    /// Mark this value as an output
    pub fn mark_as_output(&self) {
        let mut tracer = self.tracer.lock().expect("lock should not be poisoned");
        tracer.mark_output(self.id);
    }

    // Common operations
    pub fn add(&self, other: &TracedValue) -> JitResult<TracedValue> {
        self.binary_op(other, Operation::Add)
    }

    pub fn sub(&self, other: &TracedValue) -> JitResult<TracedValue> {
        self.binary_op(other, Operation::Sub)
    }

    pub fn mul(&self, other: &TracedValue) -> JitResult<TracedValue> {
        self.binary_op(other, Operation::Mul)
    }

    pub fn div(&self, other: &TracedValue) -> JitResult<TracedValue> {
        self.binary_op(other, Operation::Div)
    }

    pub fn relu(&self) -> JitResult<TracedValue> {
        self.unary_op(Operation::Relu)
    }

    pub fn sigmoid(&self) -> JitResult<TracedValue> {
        self.unary_op(Operation::Sigmoid)
    }

    pub fn tanh(&self) -> JitResult<TracedValue> {
        self.unary_op(Operation::Tanh)
    }

    pub fn exp(&self) -> JitResult<TracedValue> {
        self.unary_op(Operation::Exp)
    }

    pub fn log(&self) -> JitResult<TracedValue> {
        self.unary_op(Operation::Log)
    }
}

impl Default for GraphTracer {
    fn default() -> Self {
        Self::new()
    }
}

/// Trace a function and capture its computation graph
pub fn trace_function<F, I, O>(func: F, example_inputs: I) -> JitResult<ComputationGraph>
where
    F: FnOnce(I) -> O,
{
    let tracer = Arc::new(Mutex::new(GraphTracer::new()));

    // Start tracing
    {
        let mut t = tracer.lock().expect("lock should not be poisoned");
        t.start_tracing();
    }

    // Execute function
    let _outputs = func(example_inputs);

    // Stop tracing and get graph
    let graph = {
        let mut t = tracer.lock().expect("lock should not be poisoned");
        t.stop_tracing()
    };

    Ok(graph)
}

/// Get size in bytes for a data type
fn dtype_size_bytes(dtype: DType) -> usize {
    match dtype {
        DType::Bool | DType::I8 | DType::U8 | DType::QInt8 | DType::QUInt8 => 1,
        DType::I16 | DType::F16 | DType::BF16 => 2,
        DType::I32 | DType::F32 | DType::U32 | DType::QInt32 => 4,
        DType::I64 | DType::F64 | DType::C64 | DType::U64 => 8,
        DType::C128 => 16,
    }
}

/// Performance analysis results
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    /// Total execution time
    pub total_time: Duration,

    /// Per-operation statistics
    pub op_stats: HashMap<String, OpStats>,

    /// Memory usage statistics
    pub memory_stats: MemoryStats,

    /// Bottleneck analysis
    pub bottlenecks: Vec<Bottleneck>,
}

#[derive(Debug, Clone)]
pub struct OpStats {
    pub count: usize,
    pub total_time: Duration,
    pub avg_time: Duration,
    pub memory_usage: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub peak_usage: usize,
    pub total_allocated: usize,
    pub fragmentation_ratio: f32,
}

#[derive(Debug, Clone)]
pub struct Bottleneck {
    pub op_name: String,
    pub time_percentage: f32,
    pub memory_percentage: f32,
    pub recommendation: String,
}

impl Profiler {
    /// Generate performance analysis
    pub fn analyze(&self) -> PerformanceAnalysis {
        let total_time = self.get_total_time();
        let mut op_stats = HashMap::new();

        for (op_name, &op_time) in &self.op_timings {
            let count = self.op_counts.get(op_name).copied().unwrap_or(0);
            let memory_usage = self.memory_usage.get(op_name).copied().unwrap_or(0);

            op_stats.insert(
                op_name.clone(),
                OpStats {
                    count,
                    total_time: op_time,
                    avg_time: if count > 0 {
                        op_time / count as u32
                    } else {
                        Duration::ZERO
                    },
                    memory_usage,
                },
            );
        }

        let peak_usage = self.memory_usage.values().max().copied().unwrap_or(0);
        let total_allocated = self.memory_usage.values().sum();

        let memory_stats = MemoryStats {
            peak_usage,
            total_allocated,
            fragmentation_ratio: 0.0, // Would need more sophisticated tracking
        };

        let bottlenecks = self.identify_bottlenecks(&total_time);

        PerformanceAnalysis {
            total_time,
            op_stats,
            memory_stats,
            bottlenecks,
        }
    }

    fn identify_bottlenecks(&self, total_time: &Duration) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();
        let total_memory: usize = self.memory_usage.values().sum();

        for (op_name, &op_time) in &self.op_timings {
            let time_percentage = if total_time.as_nanos() > 0 {
                (op_time.as_nanos() as f32 / total_time.as_nanos() as f32) * 100.0
            } else {
                0.0
            };

            let memory_usage = self.memory_usage.get(op_name).copied().unwrap_or(0);
            let memory_percentage = if total_memory > 0 {
                (memory_usage as f32 / total_memory as f32) * 100.0
            } else {
                0.0
            };

            if time_percentage > 10.0 || memory_percentage > 20.0 {
                let recommendation = if time_percentage > 20.0 {
                    "Consider optimizing this operation - it's consuming significant compute time"
                        .to_string()
                } else if memory_percentage > 30.0 {
                    "High memory usage - consider reducing precision or using memory optimization"
                        .to_string()
                } else {
                    "Monitor this operation for potential optimization".to_string()
                };

                bottlenecks.push(Bottleneck {
                    op_name: op_name.clone(),
                    time_percentage,
                    memory_percentage,
                    recommendation,
                });
            }
        }

        bottlenecks.sort_by(|a, b| {
            b.time_percentage
                .partial_cmp(&a.time_percentage)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        bottlenecks
    }
}

/// Utility macro for easy tracing
#[macro_export]
macro_rules! trace {
    ($tracer:expr, $op:expr, $($input:expr),*) => {{
        let inputs = vec![$($input.id),*];
        // This would need to be expanded with proper shape/type inference
        $tracer.lock().expect("lock should not be poisoned").record_operation($op, &inputs, Shape::new(vec![1]), DType::F32, DeviceType::Cpu)
    }};
}

/// Source mapping for debugging JIT-compiled code
#[derive(Debug, Clone)]
pub struct SourceMap {
    /// Mapping from node IDs to source locations
    pub node_to_source: HashMap<NodeId, SourceLocation>,

    /// Mapping from generated code addresses to source locations
    pub code_to_source: HashMap<usize, SourceLocation>,

    /// Source file contents for reference
    pub source_files: HashMap<String, String>,

    /// Symbol table for debugging
    pub symbols: HashMap<String, SymbolInfo>,
}

/// Source location in original code
#[derive(Debug, Clone, PartialEq)]
pub struct SourceLocation {
    /// Source file name/path
    pub file: String,
    /// Line number (1-based)
    pub line: u32,
    /// Column number (1-based)
    pub column: u32,
    /// Length of the source span
    pub length: u32,
    /// Function or scope name
    pub function: Option<String>,
}

/// Symbol information for debugging
#[derive(Debug, Clone)]
pub struct SymbolInfo {
    /// Symbol name
    pub name: String,
    /// Symbol type (variable, function, etc.)
    pub symbol_type: SymbolType,
    /// Source location where defined
    pub definition: SourceLocation,
    /// Data type information
    pub data_type: Option<DType>,
    /// Shape information for tensors
    pub shape: Option<Shape>,
}

/// Types of symbols
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolType {
    Variable,
    Function,
    Parameter,
    Constant,
    Temporary,
}

/// Source map builder for creating debug information
pub struct SourceMapBuilder {
    source_map: SourceMap,
    current_file: Option<String>,
    current_function: Option<String>,
}

impl SourceMapBuilder {
    /// Create a new source map builder
    pub fn new() -> Self {
        Self {
            source_map: SourceMap {
                node_to_source: HashMap::new(),
                code_to_source: HashMap::new(),
                source_files: HashMap::new(),
                symbols: HashMap::new(),
            },
            current_file: None,
            current_function: None,
        }
    }

    /// Set the current source file
    pub fn set_current_file(&mut self, file: String, content: String) {
        self.current_file = Some(file.clone());
        self.source_map.source_files.insert(file, content);
    }

    /// Set the current function context
    pub fn set_current_function(&mut self, function: String) {
        self.current_function = Some(function);
    }

    /// Add a mapping from node to source location
    pub fn add_node_mapping(&mut self, node_id: NodeId, location: SourceLocation) {
        self.source_map.node_to_source.insert(node_id, location);
    }

    /// Add a mapping from generated code address to source location
    pub fn add_code_mapping(&mut self, address: usize, location: SourceLocation) {
        self.source_map.code_to_source.insert(address, location);
    }

    /// Add symbol information
    pub fn add_symbol(&mut self, symbol: SymbolInfo) {
        self.source_map.symbols.insert(symbol.name.clone(), symbol);
    }

    /// Create a source location with current context
    pub fn create_location(&self, line: u32, column: u32, length: u32) -> SourceLocation {
        SourceLocation {
            file: self
                .current_file
                .clone()
                .unwrap_or_else(|| "<unknown>".to_string()),
            line,
            column,
            length,
            function: self.current_function.clone(),
        }
    }

    /// Build the final source map
    pub fn build(self) -> SourceMap {
        self.source_map
    }
}

impl Default for SourceMapBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceMap {
    /// Create an empty source map
    pub fn new() -> Self {
        Self {
            node_to_source: HashMap::new(),
            code_to_source: HashMap::new(),
            source_files: HashMap::new(),
            symbols: HashMap::new(),
        }
    }

    /// Get source location for a node
    pub fn get_node_location(&self, node_id: NodeId) -> Option<&SourceLocation> {
        self.node_to_source.get(&node_id)
    }

    /// Get source location for a code address
    pub fn get_code_location(&self, address: usize) -> Option<&SourceLocation> {
        self.code_to_source.get(&address)
    }

    /// Get symbol information
    pub fn get_symbol(&self, name: &str) -> Option<&SymbolInfo> {
        self.symbols.get(name)
    }

    /// Get source line for a location
    pub fn get_source_line(&self, location: &SourceLocation) -> Option<String> {
        self.source_files.get(&location.file).and_then(|content| {
            content
                .lines()
                .nth((location.line - 1) as usize)
                .map(|s| s.to_string())
        })
    }

    /// Get context lines around a location
    pub fn get_source_context(
        &self,
        location: &SourceLocation,
        context_lines: u32,
    ) -> Vec<(u32, String)> {
        let mut lines = Vec::new();

        if let Some(content) = self.source_files.get(&location.file) {
            let file_lines: Vec<&str> = content.lines().collect();
            let start_line = location.line.saturating_sub(context_lines);
            let end_line = std::cmp::min(location.line + context_lines, file_lines.len() as u32);

            for line_num in start_line..end_line {
                if let Some(line_content) = file_lines.get(line_num as usize) {
                    lines.push((line_num + 1, line_content.to_string()));
                }
            }
        }

        lines
    }

    /// Find all locations in a file
    pub fn find_locations_in_file(&self, file: &str) -> Vec<(NodeId, &SourceLocation)> {
        self.node_to_source
            .iter()
            .filter(|(_, loc)| loc.file == file)
            .map(|(&node_id, loc)| (node_id, loc))
            .collect()
    }

    /// Find all symbols of a specific type
    pub fn find_symbols_by_type(&self, symbol_type: SymbolType) -> Vec<&SymbolInfo> {
        self.symbols
            .values()
            .filter(|symbol| symbol.symbol_type == symbol_type)
            .collect()
    }

    /// Generate debug information for external debuggers
    pub fn to_dwarf_info(&self) -> DebugInfo {
        DebugInfo {
            compilation_unit: self.current_file().unwrap_or_else(|| "<jit>".to_string()),
            functions: self.extract_function_info(),
            variables: self.extract_variable_info(),
            line_table: self.generate_line_table(),
        }
    }

    /// Get the current file being debugged
    fn current_file(&self) -> Option<String> {
        self.source_files.keys().next().cloned()
    }

    /// Extract function debug information
    fn extract_function_info(&self) -> Vec<FunctionDebugInfo> {
        let mut functions = Vec::new();
        let mut seen_functions = std::collections::HashSet::new();

        for symbol in self.symbols.values() {
            if symbol.symbol_type == SymbolType::Function && seen_functions.insert(&symbol.name) {
                functions.push(FunctionDebugInfo {
                    name: symbol.name.clone(),
                    start_location: symbol.definition.clone(),
                    parameters: self.find_function_parameters(&symbol.name),
                    local_variables: self.find_function_locals(&symbol.name),
                });
            }
        }

        functions
    }

    /// Find parameters for a function
    fn find_function_parameters(&self, function_name: &str) -> Vec<String> {
        self.symbols
            .values()
            .filter(|symbol| {
                symbol.symbol_type == SymbolType::Parameter
                    && symbol.definition.function.as_deref() == Some(function_name)
            })
            .map(|symbol| symbol.name.clone())
            .collect()
    }

    /// Find local variables for a function
    fn find_function_locals(&self, function_name: &str) -> Vec<String> {
        self.symbols
            .values()
            .filter(|symbol| {
                matches!(
                    symbol.symbol_type,
                    SymbolType::Variable | SymbolType::Temporary
                ) && symbol.definition.function.as_deref() == Some(function_name)
            })
            .map(|symbol| symbol.name.clone())
            .collect()
    }

    /// Extract variable debug information
    fn extract_variable_info(&self) -> Vec<VariableDebugInfo> {
        self.symbols
            .values()
            .filter(|symbol| {
                matches!(
                    symbol.symbol_type,
                    SymbolType::Variable | SymbolType::Parameter
                )
            })
            .map(|symbol| VariableDebugInfo {
                name: symbol.name.clone(),
                data_type: symbol.data_type.unwrap_or(DType::F32),
                location: symbol.definition.clone(),
                shape: symbol.shape.clone(),
            })
            .collect()
    }

    /// Generate line number table for debugging
    fn generate_line_table(&self) -> Vec<LineTableEntry> {
        let mut entries = Vec::new();

        for (&address, location) in &self.code_to_source {
            entries.push(LineTableEntry {
                address,
                file: location.file.clone(),
                line: location.line,
                column: location.column,
            });
        }

        entries.sort_by_key(|entry| entry.address);
        entries
    }
}

impl Default for SourceMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Debug information for external debuggers
#[derive(Debug, Clone)]
pub struct DebugInfo {
    pub compilation_unit: String,
    pub functions: Vec<FunctionDebugInfo>,
    pub variables: Vec<VariableDebugInfo>,
    pub line_table: Vec<LineTableEntry>,
}

/// Function debug information
#[derive(Debug, Clone)]
pub struct FunctionDebugInfo {
    pub name: String,
    pub start_location: SourceLocation,
    pub parameters: Vec<String>,
    pub local_variables: Vec<String>,
}

/// Variable debug information
#[derive(Debug, Clone)]
pub struct VariableDebugInfo {
    pub name: String,
    pub data_type: DType,
    pub location: SourceLocation,
    pub shape: Option<Shape>,
}

/// Line table entry for debugging
#[derive(Debug, Clone)]
pub struct LineTableEntry {
    pub address: usize,
    pub file: String,
    pub line: u32,
    pub column: u32,
}

/// Debugging utilities
pub struct DebugUtils;

impl DebugUtils {
    /// Create a source map from a computation graph with simple heuristics
    pub fn create_source_map_from_graph(graph: &ComputationGraph) -> SourceMap {
        let mut builder = SourceMapBuilder::new();
        builder.set_current_file(
            "generated.py".to_string(),
            "# JIT generated code".to_string(),
        );

        for (node_id, node) in graph.nodes() {
            let location = SourceLocation {
                file: "generated.py".to_string(),
                line: node_id.index() as u32 + 1,
                column: 1,
                length: node.name.len() as u32,
                function: Some("forward".to_string()),
            };

            builder.add_node_mapping(node_id, location.clone());

            // Add symbol for this node
            let symbol = SymbolInfo {
                name: node.name.clone(),
                symbol_type: SymbolType::Variable,
                definition: location,
                data_type: Some(node.dtype),
                shape: Some(node.output_shape.clone()),
            };
            builder.add_symbol(symbol);
        }

        builder.build()
    }

    /// Generate a stack trace from a source map and node ID
    pub fn generate_stack_trace(source_map: &SourceMap, node_id: NodeId) -> String {
        if let Some(location) = source_map.get_node_location(node_id) {
            let mut trace = String::new();
            trace.push_str(&format!(
                "  File \"{}\", line {}\n",
                location.file, location.line
            ));

            if let Some(source_line) = source_map.get_source_line(location) {
                trace.push_str(&format!("    {}\n", source_line.trim()));

                // Add caret pointing to the column
                let indent = " ".repeat(4 + location.column as usize - 1);
                let caret = "^".repeat(location.length as usize);
                trace.push_str(&format!("    {}{}\n", indent, caret));
            }

            if let Some(function) = &location.function {
                trace.push_str(&format!("    in {}\n", function));
            }

            trace
        } else {
            format!("  Unknown location for node {:?}\n", node_id)
        }
    }

    /// Format an error with source information
    pub fn format_error_with_source(
        source_map: &SourceMap,
        node_id: NodeId,
        error: &str,
    ) -> String {
        let mut formatted = String::new();
        formatted.push_str(&format!("JIT Error: {}\n", error));
        formatted.push_str("Traceback (most recent call last):\n");
        formatted.push_str(&Self::generate_stack_trace(source_map, node_id));
        formatted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_tracer() {
        let mut tracer = GraphTracer::new();
        tracer.start_tracing();

        // Create input
        let input_id =
            tracer.create_input("x", Shape::new(vec![32, 128]), DType::F32, DeviceType::Cpu);

        // Record ReLU operation
        let relu_id = tracer
            .record_operation(
                Operation::Relu,
                &[input_id],
                Shape::new(vec![32, 128]),
                DType::F32,
                DeviceType::Cpu,
            )
            .unwrap();

        tracer.mark_output(relu_id);

        let graph = tracer.stop_tracing();

        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs.len(), 1);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_traced_value_operations() {
        let tracer = Arc::new(Mutex::new(GraphTracer::new()));

        {
            let mut t = tracer.lock().expect("lock should not be poisoned");
            t.start_tracing();
        }

        let x = TracedValue::input(
            "x",
            Shape::new(vec![10, 20]),
            DType::F32,
            DeviceType::Cpu,
            tracer.clone(),
        );

        let y = TracedValue::input(
            "y",
            Shape::new(vec![10, 20]),
            DType::F32,
            DeviceType::Cpu,
            tracer.clone(),
        );

        let z = x.add(&y).unwrap();
        let w = z.relu().unwrap();

        w.mark_as_output();

        let graph = {
            let mut t = tracer.lock().expect("lock should not be poisoned");
            t.stop_tracing()
        };

        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 1);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_matmul_tracing() {
        let tracer = Arc::new(Mutex::new(GraphTracer::new()));

        {
            let mut t = tracer.lock().expect("lock should not be poisoned");
            t.start_tracing();
        }

        let a = TracedValue::input(
            "a",
            Shape::new(vec![32, 128]),
            DType::F32,
            DeviceType::Cpu,
            tracer.clone(),
        );

        let b = TracedValue::input(
            "b",
            Shape::new(vec![128, 64]),
            DType::F32,
            DeviceType::Cpu,
            tracer.clone(),
        );

        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape.dims(), &[32, 64]);

        c.mark_as_output();

        let graph = {
            let mut t = tracer.lock().expect("lock should not be poisoned");
            t.stop_tracing()
        };

        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_profiling() {
        let tracer = Arc::new(Mutex::new(GraphTracer::new_with_profiling()));

        {
            let mut t = tracer.lock().expect("lock should not be poisoned");
            t.start_tracing();
        }

        let x = TracedValue::input(
            "x",
            Shape::new(vec![1000, 1000]),
            DType::F32,
            DeviceType::Cpu,
            tracer.clone(),
        );

        let y = TracedValue::input(
            "y",
            Shape::new(vec![1000, 1000]),
            DType::F32,
            DeviceType::Cpu,
            tracer.clone(),
        );

        // Perform several operations
        let z1 = x.add(&y).unwrap();
        let z2 = z1.relu().unwrap();
        let z3 = z2.mul(&x).unwrap();

        z3.mark_as_output();

        let (graph, analysis) = {
            let mut t = tracer.lock().expect("lock should not be poisoned");
            let graph = t.stop_tracing();
            let analysis = t.get_profiler().map(|p| p.analyze());
            (graph, analysis)
        };

        assert!(graph.validate().is_ok());
        assert!(analysis.is_some());

        if let Some(analysis) = analysis {
            assert!(analysis.op_stats.contains_key("Add"));
            assert!(analysis.op_stats.contains_key("Relu"));
            assert!(analysis.op_stats.contains_key("Mul"));
            assert!(analysis.total_time >= Duration::ZERO);
        }
    }

    #[test]
    fn test_profiler_bottleneck_detection() {
        let mut profiler = Profiler::new();

        // Simulate some operations
        profiler.start_op("MatMul".to_string());
        std::thread::sleep(Duration::from_millis(10));
        profiler.end_op();

        profiler.start_op("Add".to_string());
        std::thread::sleep(Duration::from_millis(1));
        profiler.end_op();

        profiler.record_memory_usage("MatMul".to_string(), 1000000);
        profiler.record_memory_usage("Add".to_string(), 100000);

        let analysis = profiler.analyze();

        assert!(!analysis.bottlenecks.is_empty());
        assert!(analysis.bottlenecks[0].op_name == "MatMul");
        assert!(analysis.bottlenecks[0].time_percentage > 50.0);
    }
}
