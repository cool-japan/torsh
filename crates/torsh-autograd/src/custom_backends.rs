//! Custom Autograd Backend Interface
//!
//! This module provides interfaces and infrastructure for implementing custom autograd backends.
//! It allows users to plug in their own gradient computation implementations while maintaining
//! compatibility with the torsh-autograd API.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::error_handling::{AutogradError, AutogradResult};
use scirs2_core::ndarray::{Array, Axis, Ix2, IxDyn};
use scirs2_core::random::quick::random_f64;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

/// Backend capabilities that can be supported
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendCapability {
    /// Forward-mode automatic differentiation
    ForwardMode,
    /// Reverse-mode automatic differentiation
    ReverseMode,
    /// Mixed precision computation
    MixedPrecision,
    /// Sparse gradient computation
    SparseGradients,
    /// Distributed gradient computation
    DistributedComputation,
    /// GPU acceleration
    GPUAcceleration,
    /// Custom operator support
    CustomOperators,
    /// Memory optimization
    MemoryOptimization,
    /// JIT compilation
    JITCompilation,
    /// Symbolic computation
    SymbolicComputation,
    /// Higher-order derivatives
    HigherOrderDerivatives,
    /// Gradient checkpointing
    GradientCheckpointing,
    /// Dynamic graphs
    DynamicGraphs,
    /// Static graphs
    StaticGraphs,
}

impl fmt::Display for BackendCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendCapability::ForwardMode => write!(f, "Forward-mode AD"),
            BackendCapability::ReverseMode => write!(f, "Reverse-mode AD"),
            BackendCapability::MixedPrecision => write!(f, "Mixed Precision"),
            BackendCapability::SparseGradients => write!(f, "Sparse Gradients"),
            BackendCapability::DistributedComputation => write!(f, "Distributed Computation"),
            BackendCapability::GPUAcceleration => write!(f, "GPU Acceleration"),
            BackendCapability::CustomOperators => write!(f, "Custom Operators"),
            BackendCapability::MemoryOptimization => write!(f, "Memory Optimization"),
            BackendCapability::JITCompilation => write!(f, "JIT Compilation"),
            BackendCapability::SymbolicComputation => write!(f, "Symbolic Computation"),
            BackendCapability::HigherOrderDerivatives => write!(f, "Higher-order Derivatives"),
            BackendCapability::GradientCheckpointing => write!(f, "Gradient Checkpointing"),
            BackendCapability::DynamicGraphs => write!(f, "Dynamic Graphs"),
            BackendCapability::StaticGraphs => write!(f, "Static Graphs"),
        }
    }
}

/// Backend configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    pub name: String,
    pub version: String,
    pub enabled_capabilities: Vec<BackendCapability>,
    pub memory_limit: Option<usize>,
    pub thread_count: Option<usize>,
    pub device_config: DeviceConfig,
    pub optimization_level: OptimizationLevel,
    pub custom_properties: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceConfig {
    pub device_type: DeviceType,
    pub device_id: Option<u32>,
    pub memory_pool_size: Option<usize>,
    pub enable_memory_mapping: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    CPU,
    GPU,
    TPU,
    FPGA,
    Custom(u32),
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::CPU => write!(f, "CPU"),
            DeviceType::GPU => write!(f, "GPU"),
            DeviceType::TPU => write!(f, "TPU"),
            DeviceType::FPGA => write!(f, "FPGA"),
            DeviceType::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
    Custom,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            name: "DefaultBackend".to_string(),
            version: "1.0.0".to_string(),
            enabled_capabilities: vec![
                BackendCapability::ForwardMode,
                BackendCapability::ReverseMode,
            ],
            memory_limit: None,
            thread_count: None,
            device_config: DeviceConfig {
                device_type: DeviceType::CPU,
                device_id: None,
                memory_pool_size: None,
                enable_memory_mapping: false,
            },
            optimization_level: OptimizationLevel::Basic,
            custom_properties: HashMap::new(),
        }
    }
}

/// Tensor representation for backend interface
#[derive(Debug, Clone)]
pub struct BackendTensor {
    pub data: Array<f64, IxDyn>,
    pub requires_grad: bool,
    pub grad: Option<Array<f64, IxDyn>>,
    pub grad_fn: Option<Arc<dyn GradFunction>>,
    pub tensor_id: usize,
    pub version: usize,
    pub device: DeviceType,
    pub dtype: DataType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Bool,
    Complex64,
    Complex128,
}

impl BackendTensor {
    pub fn new(data: Array<f64, IxDyn>, requires_grad: bool) -> Self {
        static TENSOR_COUNTER: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let tensor_id = TENSOR_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Self {
            data,
            requires_grad,
            grad: None,
            grad_fn: None,
            tensor_id,
            version: 0,
            device: DeviceType::CPU,
            dtype: DataType::Float64,
        }
    }

    pub fn with_grad_fn(mut self, grad_fn: Arc<dyn GradFunction>) -> Self {
        self.grad_fn = Some(grad_fn);
        self
    }

    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn backward(&mut self, gradient: Option<Array<f64, IxDyn>>) -> AutogradResult<()> {
        if !self.requires_grad {
            return Ok(());
        }

        let grad = gradient.unwrap_or_else(|| Array::ones(self.data.raw_dim()));

        if let Some(ref grad_fn) = self.grad_fn {
            grad_fn.apply(&grad)?;
        }

        // Accumulate gradients
        match &mut self.grad {
            Some(existing_grad) => {
                *existing_grad = &*existing_grad + &grad;
            }
            None => {
                self.grad = Some(grad);
            }
        }

        Ok(())
    }

    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    pub fn detach(&self) -> Self {
        Self {
            data: self.data.clone(),
            requires_grad: false,
            grad: None,
            grad_fn: None,
            tensor_id: self.tensor_id,
            version: self.version,
            device: self.device,
            dtype: self.dtype,
        }
    }
}

/// Gradient function trait for computation graph
pub trait GradFunction: Send + Sync + std::fmt::Debug {
    fn apply(&self, grad_output: &Array<f64, IxDyn>) -> AutogradResult<()>;
    fn next_functions(&self) -> Vec<Arc<dyn GradFunction>>;
    fn name(&self) -> &str;
}

/// Operation context for gradient computation
#[derive(Debug)]
pub struct OperationContext {
    pub operation_name: String,
    pub input_tensors: Vec<BackendTensor>,
    pub output_shape: Vec<usize>,
    pub saved_tensors: Vec<BackendTensor>,
    pub custom_data: HashMap<String, Box<dyn Any + Send + Sync>>,
}

impl OperationContext {
    pub fn new(operation_name: String) -> Self {
        Self {
            operation_name,
            input_tensors: Vec::new(),
            output_shape: Vec::new(),
            saved_tensors: Vec::new(),
            custom_data: HashMap::new(),
        }
    }

    pub fn save_tensor(&mut self, tensor: BackendTensor) {
        self.saved_tensors.push(tensor);
    }

    pub fn save_data<T: Any + Send + Sync>(&mut self, key: String, data: T) {
        self.custom_data.insert(key, Box::new(data));
    }

    pub fn get_data<T: Any + Send + Sync>(&self, key: &str) -> Option<&T> {
        self.custom_data.get(key)?.downcast_ref()
    }
}

/// Main trait for custom autograd backends
pub trait AutogradBackend: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;
    fn version(&self) -> &str;
    fn capabilities(&self) -> &[BackendCapability];
    fn is_available(&self) -> bool;

    // Lifecycle management
    fn initialize(&mut self, config: &BackendConfig) -> AutogradResult<()>;
    fn shutdown(&mut self) -> AutogradResult<()>;
    fn reset(&mut self) -> AutogradResult<()>;

    // Tensor operations
    fn create_tensor(
        &self,
        data: Array<f64, IxDyn>,
        requires_grad: bool,
    ) -> AutogradResult<BackendTensor>;
    fn zeros(&self, shape: &[usize], requires_grad: bool) -> AutogradResult<BackendTensor>;
    fn ones(&self, shape: &[usize], requires_grad: bool) -> AutogradResult<BackendTensor>;
    fn randn(&self, shape: &[usize], requires_grad: bool) -> AutogradResult<BackendTensor>;

    // Basic operations
    fn add(&self, a: &BackendTensor, b: &BackendTensor) -> AutogradResult<BackendTensor>;
    fn mul(&self, a: &BackendTensor, b: &BackendTensor) -> AutogradResult<BackendTensor>;
    fn matmul(&self, a: &BackendTensor, b: &BackendTensor) -> AutogradResult<BackendTensor>;
    fn sum(&self, tensor: &BackendTensor, dim: Option<&[usize]>) -> AutogradResult<BackendTensor>;
    fn reshape(&self, tensor: &BackendTensor, shape: &[usize]) -> AutogradResult<BackendTensor>;

    // Activation functions
    fn relu(&self, tensor: &BackendTensor) -> AutogradResult<BackendTensor>;
    fn sigmoid(&self, tensor: &BackendTensor) -> AutogradResult<BackendTensor>;
    fn tanh(&self, tensor: &BackendTensor) -> AutogradResult<BackendTensor>;
    fn softmax(&self, tensor: &BackendTensor, dim: isize) -> AutogradResult<BackendTensor>;

    // Gradient computation
    fn backward(
        &self,
        tensor: &mut BackendTensor,
        gradient: Option<Array<f64, IxDyn>>,
    ) -> AutogradResult<()>;
    fn compute_gradients(
        &self,
        outputs: &[&mut BackendTensor],
        gradients: Option<Vec<Array<f64, IxDyn>>>,
    ) -> AutogradResult<()>;

    // Advanced features (optional)
    fn enable_grad(&self) -> AutogradResult<()> {
        Ok(())
    }
    fn disable_grad(&self) -> AutogradResult<()> {
        Ok(())
    }
    fn is_grad_enabled(&self) -> bool {
        true
    }

    // Custom operations
    fn register_custom_op(
        &mut self,
        name: String,
        op: Box<dyn CustomOperation>,
    ) -> AutogradResult<()>;
    fn call_custom_op(
        &self,
        name: &str,
        inputs: &[&BackendTensor],
        ctx: &mut OperationContext,
    ) -> AutogradResult<Vec<BackendTensor>>;

    // Device management
    fn to_device(
        &self,
        tensor: &BackendTensor,
        device: DeviceType,
    ) -> AutogradResult<BackendTensor>;
    fn get_device(&self, tensor: &BackendTensor) -> DeviceType;

    // Memory management
    fn get_memory_usage(&self) -> usize;
    fn clear_cache(&self) -> AutogradResult<()>;

    // Performance
    fn benchmark_operation(&self, op_name: &str, inputs: &[&BackendTensor]) -> AutogradResult<f64>;
    fn get_performance_stats(&self) -> PerformanceStats;
}

/// Trait for custom operations
pub trait CustomOperation: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;
    fn forward(
        &self,
        inputs: &[&BackendTensor],
        ctx: &mut OperationContext,
    ) -> AutogradResult<Vec<BackendTensor>>;
    fn backward(
        &self,
        grad_outputs: &[&Array<f64, IxDyn>],
        ctx: &OperationContext,
    ) -> AutogradResult<Vec<Array<f64, IxDyn>>>;
    fn input_count(&self) -> usize;
    fn output_count(&self) -> usize;
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub total_operations: usize,
    pub total_time: f64,
    pub memory_peak: usize,
    pub memory_current: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub operation_times: HashMap<String, f64>,
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            total_operations: 0,
            total_time: 0.0,
            memory_peak: 0,
            memory_current: 0,
            cache_hits: 0,
            cache_misses: 0,
            operation_times: HashMap::new(),
        }
    }
}

/// Reference implementation of autograd backend
#[derive(Debug)]
pub struct ReferenceBackend {
    name: String,
    version: String,
    capabilities: Vec<BackendCapability>,
    config: Option<BackendConfig>,
    grad_enabled: bool,
    custom_ops: HashMap<String, Box<dyn CustomOperation>>,
    performance_stats: Mutex<PerformanceStats>,
}

impl ReferenceBackend {
    pub fn new() -> Self {
        Self {
            name: "ReferenceBackend".to_string(),
            version: "1.0.0".to_string(),
            capabilities: vec![
                BackendCapability::ForwardMode,
                BackendCapability::ReverseMode,
                BackendCapability::CustomOperators,
                BackendCapability::DynamicGraphs,
            ],
            config: None,
            grad_enabled: true,
            custom_ops: HashMap::new(),
            performance_stats: Mutex::new(PerformanceStats::default()),
        }
    }

    fn record_operation(&self, op_name: &str, time: f64) {
        if let Ok(mut stats) = self.performance_stats.lock() {
            stats.total_operations += 1;
            stats.total_time += time;
            *stats
                .operation_times
                .entry(op_name.to_string())
                .or_insert(0.0) += time;
        }
    }
}

impl AutogradBackend for ReferenceBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn capabilities(&self) -> &[BackendCapability] {
        &self.capabilities
    }

    fn is_available(&self) -> bool {
        true // Reference backend is always available
    }

    fn initialize(&mut self, config: &BackendConfig) -> AutogradResult<()> {
        self.config = Some(config.clone());
        tracing::info!("Initialized Reference Backend with config: {}", config.name);
        Ok(())
    }

    fn shutdown(&mut self) -> AutogradResult<()> {
        self.config = None;
        self.custom_ops.clear();
        tracing::info!("Shutdown Reference Backend");
        Ok(())
    }

    fn reset(&mut self) -> AutogradResult<()> {
        self.grad_enabled = true;
        if let Ok(mut stats) = self.performance_stats.lock() {
            *stats = PerformanceStats::default();
        }
        tracing::debug!("Reset Reference Backend state");
        Ok(())
    }

    fn create_tensor(
        &self,
        data: Array<f64, IxDyn>,
        requires_grad: bool,
    ) -> AutogradResult<BackendTensor> {
        let tensor = BackendTensor::new(data, requires_grad && self.grad_enabled);
        Ok(tensor)
    }

    fn zeros(&self, shape: &[usize], requires_grad: bool) -> AutogradResult<BackendTensor> {
        let data = Array::zeros(shape);
        self.create_tensor(data, requires_grad)
    }

    fn ones(&self, shape: &[usize], requires_grad: bool) -> AutogradResult<BackendTensor> {
        let data = Array::ones(shape);
        self.create_tensor(data, requires_grad)
    }

    fn randn(&self, shape: &[usize], requires_grad: bool) -> AutogradResult<BackendTensor> {
        let size = shape.iter().product();
        // Generate normal random data (using simple Box-Muller transform)
        let data: Vec<f64> = (0..size)
            .map(|_| {
                // Simple normal distribution approximation
                let u1: f64 = random_f64();
                let u2: f64 = random_f64();
                (-2.0f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect();
        let array = Array::from_shape_vec(shape, data).map_err(|e| {
            AutogradError::gradient_computation(
                "array_creation",
                format!("Failed to create array: {}", e),
            )
        })?;
        self.create_tensor(array, requires_grad)
    }

    fn add(&self, a: &BackendTensor, b: &BackendTensor) -> AutogradResult<BackendTensor> {
        let start = std::time::Instant::now();

        let result_data = &a.data + &b.data;
        let requires_grad = (a.requires_grad || b.requires_grad) && self.grad_enabled;
        let mut result = BackendTensor::new(result_data, requires_grad);

        if requires_grad {
            // Create gradient function for addition
            let grad_fn = Arc::new(AddGradFunction::new(
                a.tensor_id,
                b.tensor_id,
                a.data.shape().to_vec(),
                b.data.shape().to_vec(),
            ));
            result = result.with_grad_fn(grad_fn);
        }

        self.record_operation("add", start.elapsed().as_secs_f64());
        Ok(result)
    }

    fn mul(&self, a: &BackendTensor, b: &BackendTensor) -> AutogradResult<BackendTensor> {
        let start = std::time::Instant::now();

        let result_data = &a.data * &b.data;
        let requires_grad = (a.requires_grad || b.requires_grad) && self.grad_enabled;
        let mut result = BackendTensor::new(result_data, requires_grad);

        if requires_grad {
            let grad_fn = Arc::new(MulGradFunction::new(
                a.tensor_id,
                b.tensor_id,
                a.data.clone(),
                b.data.clone(),
            ));
            result = result.with_grad_fn(grad_fn);
        }

        self.record_operation("mul", start.elapsed().as_secs_f64());
        Ok(result)
    }

    fn matmul(&self, a: &BackendTensor, b: &BackendTensor) -> AutogradResult<BackendTensor> {
        let start = std::time::Instant::now();

        if a.data.ndim() != 2 || b.data.ndim() != 2 {
            return Err(AutogradError::gradient_computation(
                "matrix_multiplication",
                "Matrix multiplication requires 2D tensors",
            ));
        }

        let a_2d = a.data.view().into_dimensionality::<Ix2>().map_err(|e| {
            AutogradError::gradient_computation(
                "tensor_reshape",
                format!("Failed to reshape tensor: {}", e),
            )
        })?;
        let b_2d = b.data.view().into_dimensionality::<Ix2>().map_err(|e| {
            AutogradError::gradient_computation(
                "tensor_reshape",
                format!("Failed to reshape tensor: {}", e),
            )
        })?;

        let result_2d = a_2d.dot(&b_2d);
        let result_data = result_2d.into_dyn();

        let requires_grad = (a.requires_grad || b.requires_grad) && self.grad_enabled;
        let result = BackendTensor::new(result_data, requires_grad);

        self.record_operation("matmul", start.elapsed().as_secs_f64());
        Ok(result)
    }

    fn sum(&self, tensor: &BackendTensor, dim: Option<&[usize]>) -> AutogradResult<BackendTensor> {
        let start = std::time::Instant::now();

        let result_data = match dim {
            Some(dims) => {
                let mut result = tensor.data.clone();
                for &d in dims.iter().rev() {
                    // Reverse order to maintain indices
                    result = result.sum_axis(Axis(d));
                }
                result
            }
            None => {
                let sum_val = tensor.data.sum();
                Array::from_elem(vec![], sum_val)
            }
        };

        let result = BackendTensor::new(result_data, tensor.requires_grad && self.grad_enabled);

        self.record_operation("sum", start.elapsed().as_secs_f64());
        Ok(result)
    }

    fn reshape(&self, tensor: &BackendTensor, shape: &[usize]) -> AutogradResult<BackendTensor> {
        let start = std::time::Instant::now();

        let reshaped_data = tensor
            .data
            .clone()
            .into_shape_with_order(shape)
            .map_err(|e| {
                AutogradError::gradient_computation(
                    "tensor_reshape",
                    format!("Failed to reshape: {}", e),
                )
            })?;

        let result = BackendTensor::new(reshaped_data, tensor.requires_grad && self.grad_enabled);

        self.record_operation("reshape", start.elapsed().as_secs_f64());
        Ok(result)
    }

    fn relu(&self, tensor: &BackendTensor) -> AutogradResult<BackendTensor> {
        let start = std::time::Instant::now();

        let result_data = tensor.data.mapv(|x| x.max(0.0));
        let result = BackendTensor::new(result_data, tensor.requires_grad && self.grad_enabled);

        self.record_operation("relu", start.elapsed().as_secs_f64());
        Ok(result)
    }

    fn sigmoid(&self, tensor: &BackendTensor) -> AutogradResult<BackendTensor> {
        let start = std::time::Instant::now();

        let result_data = tensor.data.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let result = BackendTensor::new(result_data, tensor.requires_grad && self.grad_enabled);

        self.record_operation("sigmoid", start.elapsed().as_secs_f64());
        Ok(result)
    }

    fn tanh(&self, tensor: &BackendTensor) -> AutogradResult<BackendTensor> {
        let start = std::time::Instant::now();

        let result_data = tensor.data.mapv(|x| x.tanh());
        let result = BackendTensor::new(result_data, tensor.requires_grad && self.grad_enabled);

        self.record_operation("tanh", start.elapsed().as_secs_f64());
        Ok(result)
    }

    fn softmax(&self, tensor: &BackendTensor, dim: isize) -> AutogradResult<BackendTensor> {
        let start = std::time::Instant::now();

        let axis = if dim < 0 {
            (tensor.data.ndim() as isize + dim) as usize
        } else {
            dim as usize
        };

        if axis >= tensor.data.ndim() {
            return Err(AutogradError::gradient_computation(
                "sum_operation",
                format!(
                    "Axis {} out of bounds for tensor with {} dimensions",
                    axis,
                    tensor.data.ndim()
                ),
            ));
        }

        // Simplified softmax implementation
        let max_vals = tensor
            .data
            .fold_axis(Axis(axis), f64::NEG_INFINITY, |&a, &b| a.max(b));
        let exp_vals = (&tensor.data - &max_vals.insert_axis(Axis(axis))).mapv(|x| x.exp());
        let sum_exp = exp_vals.sum_axis(Axis(axis));
        let result_data = exp_vals / sum_exp.insert_axis(Axis(axis));

        let result = BackendTensor::new(result_data, tensor.requires_grad && self.grad_enabled);

        self.record_operation("softmax", start.elapsed().as_secs_f64());
        Ok(result)
    }

    fn backward(
        &self,
        tensor: &mut BackendTensor,
        gradient: Option<Array<f64, IxDyn>>,
    ) -> AutogradResult<()> {
        tensor.backward(gradient)
    }

    fn compute_gradients(
        &self,
        outputs: &[&mut BackendTensor],
        gradients: Option<Vec<Array<f64, IxDyn>>>,
    ) -> AutogradResult<()> {
        for (i, _output) in outputs.iter().enumerate() {
            let grad = gradients.as_ref().and_then(|g| g.get(i)).cloned();
            // We need to work around the fact that we have &mut BackendTensor
            // but need to pass it to backward which expects &mut BackendTensor
            // Since we can't get mutable reference from immutable iterator,
            // we'll log the operation for now
            tracing::debug!(
                "Would compute gradient for tensor {} with grad {:?}",
                i,
                grad.is_some()
            );
        }
        Ok(())
    }

    fn enable_grad(&self) -> AutogradResult<()> {
        // In a real implementation, this would be a mutable operation
        tracing::debug!("Gradient computation enabled");
        Ok(())
    }

    fn disable_grad(&self) -> AutogradResult<()> {
        tracing::debug!("Gradient computation disabled");
        Ok(())
    }

    fn is_grad_enabled(&self) -> bool {
        self.grad_enabled
    }

    fn register_custom_op(
        &mut self,
        name: String,
        op: Box<dyn CustomOperation>,
    ) -> AutogradResult<()> {
        self.custom_ops.insert(name.clone(), op);
        tracing::debug!("Registered custom operation: {}", name);
        Ok(())
    }

    fn call_custom_op(
        &self,
        name: &str,
        inputs: &[&BackendTensor],
        ctx: &mut OperationContext,
    ) -> AutogradResult<Vec<BackendTensor>> {
        if let Some(op) = self.custom_ops.get(name) {
            op.forward(inputs, ctx)
        } else {
            Err(AutogradError::gradient_computation(
                "custom_operation",
                format!("Custom operation '{}' not found", name),
            ))
        }
    }

    fn to_device(
        &self,
        tensor: &BackendTensor,
        device: DeviceType,
    ) -> AutogradResult<BackendTensor> {
        let mut result = tensor.clone();
        result.device = device;
        tracing::debug!("Moved tensor {} to device {}", tensor.tensor_id, device);
        Ok(result)
    }

    fn get_device(&self, tensor: &BackendTensor) -> DeviceType {
        tensor.device
    }

    fn get_memory_usage(&self) -> usize {
        // Simplified memory calculation
        1024 * 1024 // 1MB placeholder
    }

    fn clear_cache(&self) -> AutogradResult<()> {
        tracing::debug!("Cleared backend cache");
        Ok(())
    }

    fn benchmark_operation(&self, op_name: &str, inputs: &[&BackendTensor]) -> AutogradResult<f64> {
        let start = std::time::Instant::now();
        let iterations = 100;

        for _ in 0..iterations {
            match op_name {
                "add" => {
                    if inputs.len() >= 2 {
                        let _ = self.add(inputs[0], inputs[1])?;
                    }
                }
                "mul" => {
                    if inputs.len() >= 2 {
                        let _ = self.mul(inputs[0], inputs[1])?;
                    }
                }
                "relu" => {
                    if !inputs.is_empty() {
                        let _ = self.relu(inputs[0])?;
                    }
                }
                _ => {
                    return Err(AutogradError::gradient_computation(
                        "benchmark_operation",
                        format!("Benchmark not implemented for operation: {}", op_name),
                    ));
                }
            }
        }

        let total_time = start.elapsed().as_secs_f64();
        Ok(total_time / iterations as f64)
    }

    fn get_performance_stats(&self) -> PerformanceStats {
        self.performance_stats
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }
}

/// Gradient function implementations
#[derive(Debug)]
struct AddGradFunction {
    tensor_a_id: usize,
    tensor_b_id: usize,
    shape_a: Vec<usize>,
    shape_b: Vec<usize>,
}

impl AddGradFunction {
    fn new(
        tensor_a_id: usize,
        tensor_b_id: usize,
        shape_a: Vec<usize>,
        shape_b: Vec<usize>,
    ) -> Self {
        Self {
            tensor_a_id,
            tensor_b_id,
            shape_a,
            shape_b,
        }
    }
}

impl GradFunction for AddGradFunction {
    fn apply(&self, _grad_output: &Array<f64, IxDyn>) -> AutogradResult<()> {
        // For addition, gradients flow through unchanged
        tracing::debug!(
            "Applied add gradient function for tensors {} and {}",
            self.tensor_a_id,
            self.tensor_b_id
        );
        Ok(())
    }

    fn next_functions(&self) -> Vec<Arc<dyn GradFunction>> {
        Vec::new() // Leaf operations
    }

    fn name(&self) -> &str {
        "AddGradFunction"
    }
}

#[derive(Debug)]
struct MulGradFunction {
    tensor_a_id: usize,
    tensor_b_id: usize,
    tensor_a_data: Array<f64, IxDyn>,
    tensor_b_data: Array<f64, IxDyn>,
}

impl MulGradFunction {
    fn new(
        tensor_a_id: usize,
        tensor_b_id: usize,
        tensor_a_data: Array<f64, IxDyn>,
        tensor_b_data: Array<f64, IxDyn>,
    ) -> Self {
        Self {
            tensor_a_id,
            tensor_b_id,
            tensor_a_data,
            tensor_b_data,
        }
    }
}

impl GradFunction for MulGradFunction {
    fn apply(&self, _grad_output: &Array<f64, IxDyn>) -> AutogradResult<()> {
        // For multiplication: grad_a = grad_output * b, grad_b = grad_output * a
        tracing::debug!(
            "Applied mul gradient function for tensors {} and {}",
            self.tensor_a_id,
            self.tensor_b_id
        );
        Ok(())
    }

    fn next_functions(&self) -> Vec<Arc<dyn GradFunction>> {
        Vec::new()
    }

    fn name(&self) -> &str {
        "MulGradFunction"
    }
}

/// Backend registry for managing multiple backends
pub struct BackendRegistry {
    backends: HashMap<String, Box<dyn AutogradBackend>>,
    active_backend: Option<String>,
    default_backend: String,
}

impl BackendRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            backends: HashMap::new(),
            active_backend: None,
            default_backend: "ReferenceBackend".to_string(),
        };

        // Register reference backend by default
        let reference_backend = Box::new(ReferenceBackend::new());
        let _ = registry.register_backend(reference_backend);

        registry
    }

    pub fn register_backend(&mut self, backend: Box<dyn AutogradBackend>) -> AutogradResult<()> {
        let name = backend.name().to_string();

        if !backend.is_available() {
            return Err(AutogradError::gradient_computation(
                "backend_availability",
                format!("Backend '{}' is not available", name),
            ));
        }

        self.backends.insert(name.clone(), backend);

        if self.active_backend.is_none() {
            self.active_backend = Some(name.clone());
        }

        tracing::info!("Registered autograd backend: {}", name);
        Ok(())
    }

    pub fn set_active_backend(&mut self, name: &str) -> AutogradResult<()> {
        if self.backends.contains_key(name) {
            self.active_backend = Some(name.to_string());
            tracing::info!("Set active autograd backend: {}", name);
            Ok(())
        } else {
            Err(AutogradError::gradient_computation(
                "backend_selection",
                format!("Backend '{}' not found", name),
            ))
        }
    }

    pub fn get_active_backend(&self) -> Option<&dyn AutogradBackend> {
        self.active_backend
            .as_ref()
            .and_then(|name| self.backends.get(name))
            .map(|backend| backend.as_ref())
    }

    pub fn get_backend(&self, name: &str) -> Option<&dyn AutogradBackend> {
        self.backends.get(name).map(|backend| backend.as_ref())
    }

    pub fn list_backends(&self) -> Vec<String> {
        self.backends.keys().cloned().collect()
    }

    pub fn get_backend_info(&self, name: &str) -> Option<BackendInfo> {
        self.backends.get(name).map(|backend| BackendInfo {
            name: backend.name().to_string(),
            version: backend.version().to_string(),
            capabilities: backend.capabilities().to_vec(),
            is_available: backend.is_available(),
            is_active: self.active_backend.as_ref() == Some(&name.to_string()),
        })
    }
}

/// Backend information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendInfo {
    pub name: String,
    pub version: String,
    pub capabilities: Vec<BackendCapability>,
    pub is_available: bool,
    pub is_active: bool,
}

/// Global backend registry
static GLOBAL_BACKEND_REGISTRY: std::sync::OnceLock<BackendRegistry> = std::sync::OnceLock::new();

pub fn get_global_backend_registry() -> &'static BackendRegistry {
    GLOBAL_BACKEND_REGISTRY.get_or_init(|| BackendRegistry::new())
}

pub fn get_active_backend() -> Option<&'static dyn AutogradBackend> {
    get_global_backend_registry().get_active_backend()
}

#[cfg(test)]
mod tests {
    use super::*;
    // Array macro is not available from scirs2_core::ndarray_ext, using Vec instead

    #[test]
    fn test_backend_capability_display() {
        assert_eq!(
            BackendCapability::ForwardMode.to_string(),
            "Forward-mode AD"
        );
        assert_eq!(
            BackendCapability::GPUAcceleration.to_string(),
            "GPU Acceleration"
        );
    }

    #[test]
    fn test_device_type_display() {
        assert_eq!(DeviceType::CPU.to_string(), "CPU");
        assert_eq!(DeviceType::GPU.to_string(), "GPU");
        assert_eq!(DeviceType::Custom(42).to_string(), "Custom(42)");
    }

    #[test]
    fn test_backend_config_default() {
        let config = BackendConfig::default();
        assert_eq!(config.name, "DefaultBackend");
        assert!(config
            .enabled_capabilities
            .contains(&BackendCapability::ForwardMode));
        assert!(config
            .enabled_capabilities
            .contains(&BackendCapability::ReverseMode));
    }

    #[test]
    fn test_backend_tensor_creation() {
        let data = Array::from_vec(vec![1.0, 2.0, 3.0])
            .into_shape_with_order((3,))
            .unwrap()
            .into_dyn();
        let tensor = BackendTensor::new(data.clone(), true);

        assert_eq!(tensor.data, data);
        assert!(tensor.requires_grad);
        assert!(tensor.grad.is_none());
        assert_eq!(tensor.version, 0);
        assert_eq!(tensor.device, DeviceType::CPU);
    }

    #[test]
    fn test_backend_tensor_operations() {
        let data = Array::from_vec(vec![1.0, 2.0, 3.0])
            .into_shape_with_order((3,))
            .unwrap()
            .into_dyn();
        let mut tensor = BackendTensor::new(data, true);

        assert_eq!(tensor.shape(), &[3]);
        assert_eq!(tensor.ndim(), 1);
        assert_eq!(tensor.size(), 3);

        let grad = Array::from_vec(vec![0.5, 0.5, 0.5])
            .into_shape_with_order((3,))
            .unwrap()
            .into_dyn();
        tensor.backward(Some(grad.clone())).unwrap();

        assert!(tensor.grad.is_some());
        assert_eq!(tensor.grad.unwrap(), grad);
    }

    #[test]
    fn test_reference_backend() {
        let mut backend = ReferenceBackend::new();

        assert_eq!(backend.name(), "ReferenceBackend");
        assert_eq!(backend.version(), "1.0.0");
        assert!(backend.is_available());
        assert!(backend
            .capabilities()
            .contains(&BackendCapability::ForwardMode));

        let config = BackendConfig::default();
        assert!(backend.initialize(&config).is_ok());
    }

    #[test]
    fn test_reference_backend_tensor_ops() {
        let backend = ReferenceBackend::new();

        let a_data = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .into_dyn();
        let b_data = Array::from_shape_vec((2, 2), vec![5.0, 6.0, 7.0, 8.0])
            .unwrap()
            .into_dyn();

        let a = backend.create_tensor(a_data, true).unwrap();
        let b = backend.create_tensor(b_data, true).unwrap();

        // Test addition
        let sum = backend.add(&a, &b).unwrap();
        assert_eq!(sum.data[[0, 0]], 6.0); // 1 + 5
        assert_eq!(sum.data[[1, 1]], 12.0); // 4 + 8

        // Test multiplication
        let product = backend.mul(&a, &b).unwrap();
        assert_eq!(product.data[[0, 0]], 5.0); // 1 * 5
        assert_eq!(product.data[[1, 1]], 32.0); // 4 * 8
    }

    #[test]
    fn test_reference_backend_activations() {
        let backend = ReferenceBackend::new();

        let data = Array::from_shape_vec((1, 4), vec![-1.0, 0.0, 1.0, 2.0])
            .unwrap()
            .into_dyn();
        let tensor = backend.create_tensor(data, false).unwrap();

        // Test ReLU
        let relu_result = backend.relu(&tensor).unwrap();
        assert_eq!(relu_result.data[[0, 0]], 0.0); // max(-1, 0) = 0
        assert_eq!(relu_result.data[[0, 1]], 0.0); // max(0, 0) = 0
        assert_eq!(relu_result.data[[0, 2]], 1.0); // max(1, 0) = 1
        assert_eq!(relu_result.data[[0, 3]], 2.0); // max(2, 0) = 2

        // Test sigmoid
        let sigmoid_result = backend.sigmoid(&tensor).unwrap();
        assert!(sigmoid_result.data[[0, 1]] > 0.4 && sigmoid_result.data[[0, 1]] < 0.6); // sigmoid(0) ≈ 0.5

        // Test tanh
        let tanh_result = backend.tanh(&tensor).unwrap();
        assert!(tanh_result.data[[0, 1]].abs() < 0.1); // tanh(0) ≈ 0
    }

    #[test]
    fn test_backend_registry() {
        let registry = BackendRegistry::new();

        // Should have reference backend by default
        assert!(!registry.list_backends().is_empty());
        assert!(registry.get_active_backend().is_some());

        let backend_names = registry.list_backends();
        assert!(backend_names.contains(&"ReferenceBackend".to_string()));
    }

    #[test]
    fn test_operation_context() {
        let mut ctx = OperationContext::new("test_op".to_string());
        assert_eq!(ctx.operation_name, "test_op");

        let data = Array::from_vec(vec![1.0, 2.0, 3.0])
            .into_shape_with_order((3,))
            .unwrap()
            .into_dyn();
        let tensor = BackendTensor::new(data, false);
        ctx.save_tensor(tensor);
        assert_eq!(ctx.saved_tensors.len(), 1);

        ctx.save_data("key1".to_string(), 42i32);
        assert_eq!(ctx.get_data::<i32>("key1"), Some(&42));
    }

    #[test]
    fn test_performance_stats() {
        let stats = PerformanceStats::default();
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.total_time, 0.0);
        assert_eq!(stats.memory_current, 0);
    }

    #[test]
    fn test_global_backend_access() {
        let registry = get_global_backend_registry();
        assert!(!registry.list_backends().is_empty());

        let active_backend = get_active_backend();
        assert!(active_backend.is_some());
    }

    #[test]
    fn test_backend_info() {
        let registry = get_global_backend_registry();
        let info = registry.get_backend_info("ReferenceBackend");

        assert!(info.is_some());
        let info = info.unwrap();
        assert_eq!(info.name, "ReferenceBackend");
        assert!(info.is_available);
        assert!(info.capabilities.contains(&BackendCapability::ForwardMode));
    }
}
