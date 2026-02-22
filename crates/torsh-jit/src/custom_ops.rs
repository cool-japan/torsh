//! Custom operators for JIT compilation
//!
//! This module provides support for registering and using custom operators
//! in the JIT compiler, similar to PyTorch's custom operator functionality.

use crate::{JitError, JitResult, TensorRef};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use torsh_core::DType;

/// Type alias for custom operator function
pub type CustomOpFn = Box<dyn Fn(&[TensorRef]) -> JitResult<Vec<TensorRef>> + Send + Sync>;

/// Type alias for shape inference function
pub type ShapeInferenceFn = Box<dyn Fn(&[Vec<usize>]) -> JitResult<Vec<Vec<usize>>> + Send + Sync>;

/// Type alias for gradient function
pub type GradientFn =
    Box<dyn Fn(&[TensorRef], &[TensorRef]) -> JitResult<Vec<TensorRef>> + Send + Sync>;

/// Type alias for type validation function
pub type TypeValidatorFn = Box<dyn Fn(&[TensorRef]) -> JitResult<()> + Send + Sync>;

/// Type alias for memory optimization function
pub type MemoryOptimizerFn = Box<dyn Fn(&[TensorRef]) -> JitResult<MemoryLayout> + Send + Sync>;

/// Performance characteristics and hints for optimization
#[derive(Debug, Clone)]
pub struct PerformanceHints {
    /// Computational complexity estimate (e.g., O(n), O(n^2))
    pub complexity: ComplexityClass,

    /// Memory access pattern
    pub memory_pattern: MemoryAccessPattern,

    /// Whether the operation is vectorizable
    pub vectorizable: bool,

    /// Whether the operation can be parallelized
    pub parallelizable: bool,

    /// Preferred minimum tensor size for efficient execution
    pub min_efficient_size: Option<usize>,

    /// Cache behavior hints
    pub cache_friendly: bool,

    /// Whether output can be computed in-place
    pub supports_inplace: bool,
}

/// Computational complexity classes
#[derive(Debug, Clone)]
pub enum ComplexityClass {
    Constant,       // O(1)
    Linear,         // O(n)
    LinearLogN,     // O(n log n)
    Quadratic,      // O(n^2)
    Cubic,          // O(n^3)
    Exponential,    // O(2^n)
    Custom(String), // Custom complexity description
}

/// Memory access patterns for optimization
#[derive(Debug, Clone)]
pub enum MemoryAccessPattern {
    Sequential, // Sequential access
    Random,     // Random access
    Strided,    // Strided access
    Blocked,    // Block access
    Broadcast,  // Broadcasting pattern
}

/// Memory layout optimization information
#[derive(Debug, Clone)]
pub struct MemoryLayout {
    /// Preferred memory alignment
    pub alignment: Option<usize>,

    /// Stride pattern for optimal access
    pub strides: Option<Vec<usize>>,

    /// Whether data should be contiguous
    pub contiguous: bool,

    /// Memory pool hint
    pub pool_hint: Option<String>,
}

/// Fusion compatibility information
#[derive(Debug, Clone)]
pub struct FusionInfo {
    /// Operations this can be fused with
    pub fusable_with: Vec<String>,

    /// Operations this cannot be fused with
    pub non_fusable_with: Vec<String>,

    /// Whether this operation acts as a fusion barrier
    pub fusion_barrier: bool,

    /// Fusion priority (higher means more preferred for fusion)
    pub fusion_priority: i32,

    /// Elementwise operation flag
    pub is_elementwise: bool,

    /// Reduction operation flag
    pub is_reduction: bool,
}

impl Default for PerformanceHints {
    fn default() -> Self {
        Self {
            complexity: ComplexityClass::Linear,
            memory_pattern: MemoryAccessPattern::Sequential,
            vectorizable: false,
            parallelizable: false,
            min_efficient_size: None,
            cache_friendly: true,
            supports_inplace: false,
        }
    }
}

impl Default for MemoryLayout {
    fn default() -> Self {
        Self {
            alignment: None,
            strides: None,
            contiguous: true,
            pool_hint: None,
        }
    }
}

impl Default for FusionInfo {
    fn default() -> Self {
        Self {
            fusable_with: Vec::new(),
            non_fusable_with: Vec::new(),
            fusion_barrier: false,
            fusion_priority: 0,
            is_elementwise: false,
            is_reduction: false,
        }
    }
}

/// Custom operator definition
pub struct CustomOperator {
    /// Operator name
    pub name: String,

    /// Namespace (e.g., "torsh", "user")
    pub namespace: String,

    /// Full qualified name (namespace::name)
    pub qualified_name: String,

    /// Forward computation function
    pub forward_fn: CustomOpFn,

    /// Shape inference function
    pub shape_fn: Option<ShapeInferenceFn>,

    /// Gradient computation function (for autograd)
    pub gradient_fn: Option<GradientFn>,

    /// Input argument specifications
    pub input_specs: Vec<ArgumentSpec>,

    /// Output specifications
    pub output_specs: Vec<ArgumentSpec>,

    /// Whether this operator is differentiable
    pub is_differentiable: bool,

    /// Additional metadata
    pub metadata: HashMap<String, String>,

    /// Performance characteristics
    pub performance_hints: PerformanceHints,

    /// Type validation function
    pub type_validator: Option<TypeValidatorFn>,

    /// Memory optimization function
    pub memory_optimizer: Option<MemoryOptimizerFn>,

    /// Backend-specific implementations
    pub backend_impls: HashMap<String, CustomOpFn>,

    /// Fusion compatibility information
    pub fusion_info: FusionInfo,
}

/// Argument specification for custom operators
#[derive(Debug, Clone)]
pub struct ArgumentSpec {
    /// Argument name
    pub name: String,

    /// Expected data type
    pub dtype: Option<DType>,

    /// Expected shape (None for dynamic)
    pub shape: Option<Vec<Option<usize>>>,

    /// Whether this argument is optional
    pub optional: bool,

    /// Default value (if optional)
    pub default_value: Option<TensorRef>,
}

lazy_static::lazy_static! {
    static ref CUSTOM_OP_REGISTRY: Arc<RwLock<CustomOpRegistry>> =
        Arc::new(RwLock::new(CustomOpRegistry::new()));
}

/// Registry for managing custom operators
pub struct CustomOpRegistry {
    /// Map from qualified names to operators
    operators: HashMap<String, Arc<CustomOperator>>,

    /// Map from namespace to operator names
    namespaces: HashMap<String, Vec<String>>,
}

impl CustomOpRegistry {
    /// Create a new registry
    fn new() -> Self {
        Self {
            operators: HashMap::new(),
            namespaces: HashMap::new(),
        }
    }

    /// Register a custom operator
    pub fn register(&mut self, op: CustomOperator) -> JitResult<()> {
        let qualified_name = op.qualified_name.clone();
        let namespace = op.namespace.clone();
        let name = op.name.clone();

        // Check if operator already exists
        if self.operators.contains_key(&qualified_name) {
            return Err(JitError::RuntimeError(format!(
                "Operator {} already registered",
                qualified_name
            )));
        }

        // Add to registry
        self.operators.insert(qualified_name, Arc::new(op));

        // Update namespace mapping
        self.namespaces.entry(namespace).or_default().push(name);

        Ok(())
    }

    /// Get an operator by qualified name
    pub fn get(&self, qualified_name: &str) -> Option<Arc<CustomOperator>> {
        self.operators.get(qualified_name).cloned()
    }

    /// List all operators in a namespace
    pub fn list_namespace(&self, namespace: &str) -> Vec<String> {
        self.namespaces.get(namespace).cloned().unwrap_or_default()
    }

    /// List all registered operators
    pub fn list_all(&self) -> Vec<String> {
        self.operators.keys().cloned().collect()
    }
}

/// Builder for creating custom operators
pub struct CustomOpBuilder {
    name: String,
    namespace: String,
    forward_fn: Option<CustomOpFn>,
    shape_fn: Option<ShapeInferenceFn>,
    gradient_fn: Option<GradientFn>,
    input_specs: Vec<ArgumentSpec>,
    output_specs: Vec<ArgumentSpec>,
    is_differentiable: bool,
    metadata: HashMap<String, String>,
    performance_hints: PerformanceHints,
    type_validator: Option<TypeValidatorFn>,
    memory_optimizer: Option<MemoryOptimizerFn>,
    backend_impls: HashMap<String, CustomOpFn>,
    fusion_info: FusionInfo,
}

impl CustomOpBuilder {
    /// Create a new builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            namespace: "user".to_string(),
            forward_fn: None,
            shape_fn: None,
            gradient_fn: None,
            input_specs: Vec::new(),
            output_specs: Vec::new(),
            is_differentiable: false,
            metadata: HashMap::new(),
            performance_hints: PerformanceHints::default(),
            type_validator: None,
            memory_optimizer: None,
            backend_impls: HashMap::new(),
            fusion_info: FusionInfo::default(),
        }
    }

    /// Set namespace
    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = namespace.into();
        self
    }

    /// Set forward function
    pub fn forward<F>(mut self, f: F) -> Self
    where
        F: Fn(&[TensorRef]) -> JitResult<Vec<TensorRef>> + Send + Sync + 'static,
    {
        self.forward_fn = Some(Box::new(f));
        self
    }

    /// Set shape inference function
    pub fn shape_inference<F>(mut self, f: F) -> Self
    where
        F: Fn(&[Vec<usize>]) -> JitResult<Vec<Vec<usize>>> + Send + Sync + 'static,
    {
        self.shape_fn = Some(Box::new(f));
        self
    }

    /// Set gradient function
    pub fn gradient<F>(mut self, f: F) -> Self
    where
        F: Fn(&[TensorRef], &[TensorRef]) -> JitResult<Vec<TensorRef>> + Send + Sync + 'static,
    {
        self.gradient_fn = Some(Box::new(f));
        self.is_differentiable = true;
        self
    }

    /// Add input specification
    pub fn input(mut self, name: impl Into<String>, dtype: Option<DType>) -> Self {
        self.input_specs.push(ArgumentSpec {
            name: name.into(),
            dtype,
            shape: None,
            optional: false,
            default_value: None,
        });
        self
    }

    /// Add output specification
    pub fn output(mut self, name: impl Into<String>, dtype: Option<DType>) -> Self {
        self.output_specs.push(ArgumentSpec {
            name: name.into(),
            dtype,
            shape: None,
            optional: false,
            default_value: None,
        });
        self
    }

    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set performance hints
    pub fn performance_hints(mut self, hints: PerformanceHints) -> Self {
        self.performance_hints = hints;
        self
    }

    /// Set complexity class
    pub fn complexity(mut self, complexity: ComplexityClass) -> Self {
        self.performance_hints.complexity = complexity;
        self
    }

    /// Mark as vectorizable
    pub fn vectorizable(mut self, vectorizable: bool) -> Self {
        self.performance_hints.vectorizable = vectorizable;
        self
    }

    /// Mark as parallelizable
    pub fn parallelizable(mut self, parallelizable: bool) -> Self {
        self.performance_hints.parallelizable = parallelizable;
        self
    }

    /// Set memory access pattern
    pub fn memory_pattern(mut self, pattern: MemoryAccessPattern) -> Self {
        self.performance_hints.memory_pattern = pattern;
        self
    }

    /// Enable in-place computation
    pub fn supports_inplace(mut self, inplace: bool) -> Self {
        self.performance_hints.supports_inplace = inplace;
        self
    }

    /// Set type validator
    pub fn type_validator<F>(mut self, validator: F) -> Self
    where
        F: Fn(&[TensorRef]) -> JitResult<()> + Send + Sync + 'static,
    {
        self.type_validator = Some(Box::new(validator));
        self
    }

    /// Set memory optimizer
    pub fn memory_optimizer<F>(mut self, optimizer: F) -> Self
    where
        F: Fn(&[TensorRef]) -> JitResult<MemoryLayout> + Send + Sync + 'static,
    {
        self.memory_optimizer = Some(Box::new(optimizer));
        self
    }

    /// Add backend-specific implementation
    pub fn backend_impl<F>(mut self, backend: impl Into<String>, implementation: F) -> Self
    where
        F: Fn(&[TensorRef]) -> JitResult<Vec<TensorRef>> + Send + Sync + 'static,
    {
        self.backend_impls
            .insert(backend.into(), Box::new(implementation));
        self
    }

    /// Mark as elementwise operation
    pub fn elementwise(mut self, elementwise: bool) -> Self {
        self.fusion_info.is_elementwise = elementwise;
        self
    }

    /// Mark as reduction operation
    pub fn reduction(mut self, reduction: bool) -> Self {
        self.fusion_info.is_reduction = reduction;
        self
    }

    /// Set fusion priority
    pub fn fusion_priority(mut self, priority: i32) -> Self {
        self.fusion_info.fusion_priority = priority;
        self
    }

    /// Add fusable operations
    pub fn fusable_with(mut self, ops: Vec<String>) -> Self {
        self.fusion_info.fusable_with.extend(ops);
        self
    }

    /// Add non-fusable operations
    pub fn non_fusable_with(mut self, ops: Vec<String>) -> Self {
        self.fusion_info.non_fusable_with.extend(ops);
        self
    }

    /// Mark as fusion barrier
    pub fn fusion_barrier(mut self, barrier: bool) -> Self {
        self.fusion_info.fusion_barrier = barrier;
        self
    }

    /// Build and register the operator
    pub fn build(self) -> JitResult<()> {
        let forward_fn = self
            .forward_fn
            .ok_or_else(|| JitError::RuntimeError("Forward function not specified".to_string()))?;

        let qualified_name = format!("{}::{}", self.namespace, self.name);

        let op = CustomOperator {
            name: self.name,
            namespace: self.namespace,
            qualified_name,
            forward_fn,
            shape_fn: self.shape_fn,
            gradient_fn: self.gradient_fn,
            input_specs: self.input_specs,
            output_specs: self.output_specs,
            is_differentiable: self.is_differentiable,
            metadata: self.metadata,
            performance_hints: self.performance_hints,
            type_validator: self.type_validator,
            memory_optimizer: self.memory_optimizer,
            backend_impls: self.backend_impls,
            fusion_info: self.fusion_info,
        };

        register_custom_op(op)
    }
}

/// Register a custom operator
pub fn register_custom_op(op: CustomOperator) -> JitResult<()> {
    let mut registry = CUSTOM_OP_REGISTRY
        .write()
        .map_err(|_| JitError::RuntimeError("Failed to acquire registry lock".to_string()))?;
    registry.register(op)
}

/// Get a custom operator by qualified name
pub fn get_custom_op(qualified_name: &str) -> Option<Arc<CustomOperator>> {
    CUSTOM_OP_REGISTRY.read().ok()?.get(qualified_name)
}

/// List all custom operators
pub fn list_custom_ops() -> Vec<String> {
    CUSTOM_OP_REGISTRY
        .read()
        .map(|r| r.list_all())
        .unwrap_or_default()
}

/// List operators in a specific namespace
pub fn list_ops_in_namespace(namespace: &str) -> Vec<String> {
    CUSTOM_OP_REGISTRY
        .read()
        .map(|r| r.list_namespace(namespace))
        .unwrap_or_default()
}

/// Macro for easier custom operator registration
#[macro_export]
macro_rules! register_op {
    ($name:expr, $forward:expr) => {
        $crate::custom_ops::CustomOpBuilder::new($name)
            .forward($forward)
            .build()
    };

    ($name:expr, $forward:expr, $gradient:expr) => {
        $crate::custom_ops::CustomOpBuilder::new($name)
            .forward($forward)
            .gradient($gradient)
            .build()
    };
}

/// Example custom operators
pub mod examples {
    use super::*;

    /// Register example custom operators
    pub fn register_example_ops() -> JitResult<()> {
        // Custom ReLU6 operator
        CustomOpBuilder::new("relu6")
            .namespace("torsh")
            .forward(|inputs| {
                if inputs.is_empty() {
                    return Err(JitError::RuntimeError("ReLU6 requires 1 input".to_string()));
                }

                let input = &inputs[0];
                let mut output = input.clone();

                // Clamp values between 0 and 6
                for val in &mut output.data {
                    *val = val.clamp(0.0, 6.0);
                }

                Ok(vec![output])
            })
            .shape_inference(|shapes| {
                if shapes.is_empty() {
                    return Err(JitError::RuntimeError(
                        "ReLU6 requires 1 input shape".to_string(),
                    ));
                }
                Ok(vec![shapes[0].clone()])
            })
            .gradient(|inputs, grad_outputs| {
                let input = &inputs[0];
                let grad_output = &grad_outputs[0];
                let mut grad_input = grad_output.clone();

                // Gradient is 1 where 0 < input < 6, 0 elsewhere
                for (i, &val) in input.data.iter().enumerate() {
                    if val <= 0.0 || val >= 6.0 {
                        grad_input.data[i] = 0.0;
                    }
                }

                Ok(vec![grad_input])
            })
            .input("input", Some(DType::F32))
            .output("output", Some(DType::F32))
            .metadata("description", "ReLU with upper bound of 6")
            .build()?;

        // Custom GELU approximation
        CustomOpBuilder::new("gelu_approx")
            .namespace("torsh")
            .forward(|inputs| {
                if inputs.is_empty() {
                    return Err(JitError::RuntimeError("GELU requires 1 input".to_string()));
                }

                let input = &inputs[0];
                let mut output = input.clone();

                // GELU approximation: x * sigmoid(1.702 * x)
                for (i, &x) in input.data.iter().enumerate() {
                    let sigmoid = 1.0 / (1.0 + (-1.702 * x).exp());
                    output.data[i] = x * sigmoid;
                }

                Ok(vec![output])
            })
            .shape_inference(|shapes| {
                if shapes.is_empty() {
                    return Err(JitError::RuntimeError(
                        "GELU requires 1 input shape".to_string(),
                    ));
                }
                Ok(vec![shapes[0].clone()])
            })
            .input("input", Some(DType::F32))
            .output("output", Some(DType::F32))
            .metadata("description", "Gaussian Error Linear Unit approximation")
            .build()?;

        Ok(())
    }
}

/// Integration with JIT compiler for custom operators
pub mod jit_integration {
    use super::*;
    use crate::graph::Operation;

    /// Check if an operation is a custom operator
    pub fn is_custom_op(op: &Operation) -> bool {
        matches!(op, Operation::Custom(_))
    }

    /// Execute a custom operator
    pub fn execute_custom_op(op_name: &str, inputs: &[TensorRef]) -> JitResult<Vec<TensorRef>> {
        let op = get_custom_op(op_name).ok_or_else(|| {
            JitError::RuntimeError(format!("Custom operator {} not found", op_name))
        })?;

        // Validate inputs
        if inputs.len() != op.input_specs.len() {
            return Err(JitError::RuntimeError(format!(
                "Expected {} inputs, got {}",
                op.input_specs.len(),
                inputs.len()
            )));
        }

        // Execute forward function
        (op.forward_fn)(inputs)
    }

    /// Infer shapes for custom operator
    pub fn infer_custom_op_shapes(
        op_name: &str,
        input_shapes: &[Vec<usize>],
    ) -> JitResult<Vec<Vec<usize>>> {
        let op = get_custom_op(op_name).ok_or_else(|| {
            JitError::RuntimeError(format!("Custom operator {} not found", op_name))
        })?;

        if let Some(shape_fn) = &op.shape_fn {
            (shape_fn)(input_shapes)
        } else {
            // Default: output shapes match first input
            if input_shapes.is_empty() {
                Ok(vec![])
            } else {
                Ok(vec![input_shapes[0].clone()])
            }
        }
    }
}

/// Performance profiling for custom operators
pub mod profiling {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    /// Performance metrics for a custom operator
    #[derive(Debug, Clone)]
    pub struct OperatorMetrics {
        pub op_name: String,
        pub total_executions: u64,
        pub total_time: Duration,
        pub average_time: Duration,
        pub min_time: Duration,
        pub max_time: Duration,
        pub memory_usage: Option<usize>,
        pub cache_hits: u64,
        pub cache_misses: u64,
    }

    lazy_static::lazy_static! {
        pub static ref PROFILER: Arc<Mutex<CustomOpProfiler>> = Arc::new(Mutex::new(CustomOpProfiler::new()));
    }

    /// Profiler for custom operators
    pub struct CustomOpProfiler {
        metrics: HashMap<String, OperatorMetrics>,
        enabled: bool,
    }

    impl CustomOpProfiler {
        fn new() -> Self {
            Self {
                metrics: HashMap::new(),
                enabled: true,
            }
        }

        pub fn record_execution(
            &mut self,
            op_name: &str,
            duration: Duration,
            memory_usage: Option<usize>,
        ) {
            if !self.enabled {
                return;
            }

            let metrics =
                self.metrics
                    .entry(op_name.to_string())
                    .or_insert_with(|| OperatorMetrics {
                        op_name: op_name.to_string(),
                        total_executions: 0,
                        total_time: Duration::ZERO,
                        average_time: Duration::ZERO,
                        min_time: Duration::MAX,
                        max_time: Duration::ZERO,
                        memory_usage,
                        cache_hits: 0,
                        cache_misses: 0,
                    });

            metrics.total_executions += 1;
            metrics.total_time += duration;
            metrics.average_time = metrics.total_time / metrics.total_executions as u32;
            metrics.min_time = metrics.min_time.min(duration);
            metrics.max_time = metrics.max_time.max(duration);

            if let Some(mem) = memory_usage {
                metrics.memory_usage = Some(mem);
            }
        }

        pub fn record_cache_hit(&mut self, op_name: &str) {
            if let Some(metrics) = self.metrics.get_mut(op_name) {
                metrics.cache_hits += 1;
            }
        }

        pub fn record_cache_miss(&mut self, op_name: &str) {
            if let Some(metrics) = self.metrics.get_mut(op_name) {
                metrics.cache_misses += 1;
            }
        }

        pub fn get_metrics(&self, op_name: &str) -> Option<OperatorMetrics> {
            self.metrics.get(op_name).cloned()
        }

        pub fn get_all_metrics(&self) -> Vec<OperatorMetrics> {
            self.metrics.values().cloned().collect()
        }

        pub fn reset(&mut self) {
            self.metrics.clear();
        }

        pub fn enable(&mut self, enabled: bool) {
            self.enabled = enabled;
        }
    }

    /// Execute custom operator with profiling
    pub fn execute_with_profiling(
        op_name: &str,
        inputs: &[TensorRef],
    ) -> JitResult<Vec<TensorRef>> {
        let start = Instant::now();
        let result = jit_integration::execute_custom_op(op_name, inputs);
        let duration = start.elapsed();

        // Estimate memory usage
        let memory_usage = inputs
            .iter()
            .map(|t| t.data.len() * std::mem::size_of::<f32>())
            .sum();

        if let Ok(mut profiler) = PROFILER.lock() {
            profiler.record_execution(op_name, duration, Some(memory_usage));
        }

        result
    }

    /// Get profiling metrics for an operator
    pub fn get_operator_metrics(op_name: &str) -> Option<OperatorMetrics> {
        PROFILER.lock().ok()?.get_metrics(op_name)
    }

    /// Get all profiling metrics
    pub fn get_all_metrics() -> Vec<OperatorMetrics> {
        PROFILER
            .lock()
            .map(|p| p.get_all_metrics())
            .unwrap_or_default()
    }

    /// Reset profiling data
    pub fn reset_profiling() {
        if let Ok(mut profiler) = PROFILER.lock() {
            profiler.reset();
        }
    }

    /// Enable/disable profiling
    pub fn enable_profiling(enabled: bool) {
        if let Ok(mut profiler) = PROFILER.lock() {
            profiler.enable(enabled);
        }
    }
}

/// Caching for custom operator results
pub mod caching {
    use super::*;
    use std::collections::HashMap;
    use std::hash::{Hash, Hasher};
    use std::sync::{Arc, RwLock};

    /// Cache key for operator results
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct CacheKey {
        pub op_name: String,
        pub input_hashes: Vec<u64>,
        pub input_shapes: Vec<Vec<usize>>,
    }

    /// Cached result entry
    #[derive(Debug, Clone)]
    pub struct CacheEntry {
        pub result: Vec<TensorRef>,
        pub timestamp: std::time::SystemTime,
        pub hit_count: u64,
    }

    lazy_static::lazy_static! {
        static ref CACHE: Arc<RwLock<CustomOpCache>> = Arc::new(RwLock::new(CustomOpCache::new()));
    }

    /// Cache for custom operator results
    pub struct CustomOpCache {
        entries: HashMap<CacheKey, CacheEntry>,
        max_size: usize,
        enabled: bool,
    }

    impl CustomOpCache {
        fn new() -> Self {
            Self {
                entries: HashMap::new(),
                max_size: 1000,
                enabled: true,
            }
        }

        pub fn get(&mut self, key: &CacheKey) -> Option<Vec<TensorRef>> {
            if !self.enabled {
                return None;
            }

            if let Some(entry) = self.entries.get_mut(key) {
                entry.hit_count += 1;
                Some(entry.result.clone())
            } else {
                None
            }
        }

        pub fn insert(&mut self, key: CacheKey, result: Vec<TensorRef>) {
            if !self.enabled {
                return;
            }

            // Evict oldest entries if cache is full
            if self.entries.len() >= self.max_size {
                self.evict_oldest();
            }

            let entry = CacheEntry {
                result,
                timestamp: std::time::SystemTime::now(),
                hit_count: 0,
            };

            self.entries.insert(key, entry);
        }

        fn evict_oldest(&mut self) {
            if let Some((oldest_key, _)) =
                self.entries.iter().min_by_key(|(_, entry)| entry.timestamp)
            {
                let oldest_key = oldest_key.clone();
                self.entries.remove(&oldest_key);
            }
        }

        pub fn clear(&mut self) {
            self.entries.clear();
        }

        pub fn set_max_size(&mut self, size: usize) {
            self.max_size = size;
            while self.entries.len() > self.max_size {
                self.evict_oldest();
            }
        }

        pub fn enable(&mut self, enabled: bool) {
            self.enabled = enabled;
        }

        pub fn stats(&self) -> (usize, usize) {
            (self.entries.len(), self.max_size)
        }
    }

    /// Hash a tensor for caching
    fn hash_tensor(tensor: &TensorRef) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Hash the data length and first few elements for efficiency
        tensor.data.len().hash(&mut hasher);
        for (i, &val) in tensor.data.iter().enumerate() {
            if i >= 16 {
                break;
            } // Only hash first 16 elements
            val.to_bits().hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Create cache key for inputs
    fn create_cache_key(op_name: &str, inputs: &[TensorRef]) -> CacheKey {
        let input_hashes = inputs.iter().map(hash_tensor).collect();
        let input_shapes = inputs.iter().map(|t| vec![t.data.len()]).collect(); // Simplified shape

        CacheKey {
            op_name: op_name.to_string(),
            input_hashes,
            input_shapes,
        }
    }

    /// Execute custom operator with caching
    pub fn execute_with_caching(op_name: &str, inputs: &[TensorRef]) -> JitResult<Vec<TensorRef>> {
        let cache_key = create_cache_key(op_name, inputs);

        // Try to get from cache first
        if let Ok(mut cache) = CACHE.write() {
            if let Some(cached_result) = cache.get(&cache_key) {
                if let Ok(mut profiler) = profiling::PROFILER.lock() {
                    profiler.record_cache_hit(op_name);
                }
                return Ok(cached_result);
            }
        }

        // Cache miss - execute the operation
        let result = jit_integration::execute_custom_op(op_name, inputs)?;

        // Store result in cache
        if let Ok(mut cache) = CACHE.write() {
            cache.insert(cache_key, result.clone());
            if let Ok(mut profiler) = profiling::PROFILER.lock() {
                profiler.record_cache_miss(op_name);
            }
        }

        Ok(result)
    }

    /// Clear the cache
    pub fn clear_cache() {
        if let Ok(mut cache) = CACHE.write() {
            cache.clear();
        }
    }

    /// Set cache size
    pub fn set_cache_size(size: usize) {
        if let Ok(mut cache) = CACHE.write() {
            cache.set_max_size(size);
        }
    }

    /// Enable/disable caching
    pub fn enable_caching(enabled: bool) {
        if let Ok(mut cache) = CACHE.write() {
            cache.enable(enabled);
        }
    }

    /// Get cache statistics
    pub fn get_cache_stats() -> (usize, usize) {
        CACHE.read().map(|c| c.stats()).unwrap_or((0, 0))
    }
}

/// Advanced execution engine for custom operators
pub mod advanced_execution {
    use super::*;
    use crate::fusion::FusionStrategy;

    /// Execution context for custom operators
    #[derive(Debug, Clone)]
    pub struct ExecutionContext {
        pub backend: String,
        pub device_id: Option<usize>,
        pub use_cache: bool,
        pub use_profiling: bool,
        pub fusion_strategy: Option<FusionStrategy>,
        pub optimization_level: u8,
    }

    impl Default for ExecutionContext {
        fn default() -> Self {
            Self {
                backend: "cpu".to_string(),
                device_id: None,
                use_cache: true,
                use_profiling: true,
                fusion_strategy: Some(FusionStrategy::Default),
                optimization_level: 2,
            }
        }
    }

    /// Advanced executor for custom operators
    pub struct CustomOpExecutor {
        context: ExecutionContext,
    }

    impl CustomOpExecutor {
        /// Create a new executor with context
        pub fn new(context: ExecutionContext) -> Self {
            Self { context }
        }

        /// Execute custom operator with full optimization pipeline
        pub fn execute(&self, op_name: &str, inputs: &[TensorRef]) -> JitResult<Vec<TensorRef>> {
            // Get the operator
            let op = get_custom_op(op_name).ok_or_else(|| {
                JitError::RuntimeError(format!("Custom operator {} not found", op_name))
            })?;

            // Type validation
            if let Some(validator) = &op.type_validator {
                validator(inputs)?;
            }

            // Choose execution path based on context
            let result = if self.context.use_cache {
                caching::execute_with_caching(op_name, inputs)?
            } else if self.context.use_profiling {
                profiling::execute_with_profiling(op_name, inputs)?
            } else {
                jit_integration::execute_custom_op(op_name, inputs)?
            };

            // Memory optimization
            if let Some(optimizer) = &op.memory_optimizer {
                let _layout = optimizer(inputs)?;
                // Apply memory optimizations based on layout
            }

            Ok(result)
        }

        /// Execute with backend-specific implementation
        pub fn execute_with_backend(
            &self,
            op_name: &str,
            inputs: &[TensorRef],
        ) -> JitResult<Vec<TensorRef>> {
            let op = get_custom_op(op_name).ok_or_else(|| {
                JitError::RuntimeError(format!("Custom operator {} not found", op_name))
            })?;

            // Check for backend-specific implementation
            if let Some(backend_impl) = op.backend_impls.get(&self.context.backend) {
                return backend_impl(inputs);
            }

            // Fallback to default implementation
            self.execute(op_name, inputs)
        }

        /// Check if operations can be fused
        pub fn can_fuse_ops(&self, op_names: &[String]) -> bool {
            for (i, op_name) in op_names.iter().enumerate() {
                if let Some(op) = get_custom_op(op_name) {
                    // Check fusion barrier
                    if op.fusion_info.fusion_barrier {
                        return false;
                    }

                    // Check fusability with next operation
                    if i + 1 < op_names.len() {
                        let next_op_name = &op_names[i + 1];
                        if op.fusion_info.non_fusable_with.contains(next_op_name) {
                            return false;
                        }
                    }
                }
            }
            true
        }

        /// Execute fused operations
        pub fn execute_fused(
            &self,
            op_names: &[String],
            inputs: &[TensorRef],
        ) -> JitResult<Vec<TensorRef>> {
            if !self.can_fuse_ops(op_names) {
                return Err(JitError::FusionError(
                    "Operations cannot be fused".to_string(),
                ));
            }

            // For now, execute sequentially (in a real implementation, this would generate fused kernels)
            let mut current_inputs = inputs.to_vec();

            for op_name in op_names {
                current_inputs = self.execute(op_name, &current_inputs)?;
            }

            Ok(current_inputs)
        }

        /// Set execution context
        pub fn set_context(&mut self, context: ExecutionContext) {
            self.context = context;
        }

        /// Get execution context
        pub fn context(&self) -> &ExecutionContext {
            &self.context
        }
    }

    /// Create default executor
    pub fn create_executor() -> CustomOpExecutor {
        CustomOpExecutor::new(ExecutionContext::default())
    }

    /// Create executor with specific backend
    pub fn create_executor_with_backend(backend: &str) -> CustomOpExecutor {
        let mut context = ExecutionContext::default();
        context.backend = backend.to_string();
        CustomOpExecutor::new(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_op_builder() {
        let result = CustomOpBuilder::new("test_op")
            .namespace("test")
            .forward(|inputs| Ok(inputs.to_vec()))
            .input("x", Some(DType::F32))
            .output("y", Some(DType::F32))
            .build();

        assert!(result.is_ok());

        // Check that operator was registered
        let ops = list_ops_in_namespace("test");
        assert!(ops.contains(&"test_op".to_string()));
    }

    #[test]
    fn test_example_ops() {
        let result = examples::register_example_ops();
        assert!(result.is_ok());

        // Check that operators were registered
        let torsh_ops = list_ops_in_namespace("torsh");
        assert!(torsh_ops.contains(&"relu6".to_string()));
        assert!(torsh_ops.contains(&"gelu_approx".to_string()));
    }

    #[test]
    fn test_custom_op_execution() {
        // Register a simple custom op
        let registration_result = CustomOpBuilder::new("double")
            .namespace("test")
            .input("input", None)
            .forward(|inputs| {
                let mut output = inputs[0].clone();
                for val in &mut output.data {
                    *val *= 2.0;
                }
                Ok(vec![output])
            })
            .build();

        assert!(
            registration_result.is_ok(),
            "Failed to register custom op: {:?}",
            registration_result.err()
        );

        // Test execution
        let input = TensorRef {
            data: vec![1.0, 2.0, 3.0],
        };
        let result = jit_integration::execute_custom_op("test::double", &[input]);

        if result.is_err() {
            println!("Error executing custom op: {:?}", result.as_ref().err());
            println!(
                "Available ops in 'test' namespace: {:?}",
                list_ops_in_namespace("test")
            );
        }

        assert!(
            result.is_ok(),
            "Custom op execution failed: {:?}",
            result.as_ref().err()
        );
        let outputs = result.unwrap();
        assert_eq!(outputs[0].data, vec![2.0, 4.0, 6.0]);
    }
}
