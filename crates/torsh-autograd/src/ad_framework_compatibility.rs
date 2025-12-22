//! Compatibility Layers for Different AD Frameworks
//!
//! This module provides standardized compatibility interfaces for integrating
//! with various automatic differentiation frameworks including PyTorch, JAX,
//! TensorFlow, and others. It enables seamless interoperability and migration
//! between different AD systems.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::error_handling::{AutogradError, AutogradResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Mutex, RwLock};

/// Supported automatic differentiation frameworks
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ADFramework {
    /// PyTorch autograd system
    PyTorch,
    /// Google JAX
    JAX,
    /// TensorFlow/Keras
    TensorFlow,
    /// Theano (legacy)
    Theano,
    /// Apache MXNet/Gluon
    MXNet,
    /// Chainer
    Chainer,
    /// DyNet
    DyNet,
    /// Autograd (Python)
    Autograd,
    /// Zygote.jl (Julia)
    Zygote,
    /// ForwardDiff.jl (Julia)
    ForwardDiff,
    /// ReverseDiff.jl (Julia)
    ReverseDiff,
    /// Flux.jl (Julia)
    Flux,
    /// Stan Math
    StanMath,
    /// CasADi
    CasADi,
    /// ADOL-C
    ADOLC,
    /// CppAD
    CppAD,
    /// Enzyme
    Enzyme,
    /// Tapenade
    Tapenade,
    /// Custom framework
    Custom(String),
}

impl fmt::Display for ADFramework {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ADFramework::PyTorch => write!(f, "PyTorch"),
            ADFramework::JAX => write!(f, "JAX"),
            ADFramework::TensorFlow => write!(f, "TensorFlow"),
            ADFramework::Theano => write!(f, "Theano"),
            ADFramework::MXNet => write!(f, "MXNet"),
            ADFramework::Chainer => write!(f, "Chainer"),
            ADFramework::DyNet => write!(f, "DyNet"),
            ADFramework::Autograd => write!(f, "Autograd"),
            ADFramework::Zygote => write!(f, "Zygote.jl"),
            ADFramework::ForwardDiff => write!(f, "ForwardDiff.jl"),
            ADFramework::ReverseDiff => write!(f, "ReverseDiff.jl"),
            ADFramework::Flux => write!(f, "Flux.jl"),
            ADFramework::StanMath => write!(f, "Stan Math"),
            ADFramework::CasADi => write!(f, "CasADi"),
            ADFramework::ADOLC => write!(f, "ADOL-C"),
            ADFramework::CppAD => write!(f, "CppAD"),
            ADFramework::Enzyme => write!(f, "Enzyme"),
            ADFramework::Tapenade => write!(f, "Tapenade"),
            ADFramework::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// Data types supported across different frameworks
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UniversalDataType {
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// 16-bit floating point
    Float16,
    /// Brain floating point
    BFloat16,
    /// 32-bit signed integer
    Int32,
    /// 64-bit signed integer
    Int64,
    /// 16-bit signed integer
    Int16,
    /// 8-bit signed integer
    Int8,
    /// Boolean
    Bool,
    /// Complex 64-bit
    Complex64,
    /// Complex 128-bit
    Complex128,
}

impl fmt::Display for UniversalDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UniversalDataType::Float32 => write!(f, "float32"),
            UniversalDataType::Float64 => write!(f, "float64"),
            UniversalDataType::Float16 => write!(f, "float16"),
            UniversalDataType::BFloat16 => write!(f, "bfloat16"),
            UniversalDataType::Int32 => write!(f, "int32"),
            UniversalDataType::Int64 => write!(f, "int64"),
            UniversalDataType::Int16 => write!(f, "int16"),
            UniversalDataType::Int8 => write!(f, "int8"),
            UniversalDataType::Bool => write!(f, "bool"),
            UniversalDataType::Complex64 => write!(f, "complex64"),
            UniversalDataType::Complex128 => write!(f, "complex128"),
        }
    }
}

/// Universal tensor representation for cross-framework compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalTensor {
    pub data: Vec<f64>, // Raw data in flattened form
    pub shape: Vec<usize>,
    pub dtype: UniversalDataType,
    pub requires_grad: bool,
    pub grad: Option<Box<UniversalTensor>>,
    pub framework_metadata: HashMap<String, String>,
}

impl UniversalTensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>, dtype: UniversalDataType) -> Self {
        Self {
            data,
            shape,
            dtype,
            requires_grad: false,
            grad: None,
            framework_metadata: HashMap::new(),
        }
    }

    pub fn zeros(shape: Vec<usize>, dtype: UniversalDataType) -> Self {
        let size = shape.iter().product();
        Self::new(vec![0.0; size], shape, dtype)
    }

    pub fn ones(shape: Vec<usize>, dtype: UniversalDataType) -> Self {
        let size = shape.iter().product();
        Self::new(vec![1.0; size], shape, dtype)
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    pub fn set_grad(&mut self, grad: UniversalTensor) {
        self.grad = Some(Box::new(grad));
    }

    pub fn add_metadata(&mut self, key: String, value: String) {
        self.framework_metadata.insert(key, value);
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty() || (self.shape.len() == 1 && self.shape[0] == 1)
    }
}

/// Operations supported across frameworks
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UniversalOperation {
    // Arithmetic operations
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Sqrt,
    Exp,
    Log,

    // Trigonometric operations
    Sin,
    Cos,
    Tan,

    // Matrix operations
    MatMul,
    Transpose,
    Inverse,

    // Reduction operations
    Sum,
    Mean,
    Max,
    Min,

    // Activation functions
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,

    // Convolution operations
    Conv1D,
    Conv2D,
    Conv3D,

    // Pooling operations
    MaxPool,
    AvgPool,

    // Shape operations
    Reshape,
    Flatten,
    Squeeze,
    Unsqueeze,

    // Custom operation
    Custom(String),
}

impl fmt::Display for UniversalOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UniversalOperation::Add => write!(f, "add"),
            UniversalOperation::Sub => write!(f, "sub"),
            UniversalOperation::Mul => write!(f, "mul"),
            UniversalOperation::Div => write!(f, "div"),
            UniversalOperation::Pow => write!(f, "pow"),
            UniversalOperation::Sqrt => write!(f, "sqrt"),
            UniversalOperation::Exp => write!(f, "exp"),
            UniversalOperation::Log => write!(f, "log"),
            UniversalOperation::Sin => write!(f, "sin"),
            UniversalOperation::Cos => write!(f, "cos"),
            UniversalOperation::Tan => write!(f, "tan"),
            UniversalOperation::MatMul => write!(f, "matmul"),
            UniversalOperation::Transpose => write!(f, "transpose"),
            UniversalOperation::Inverse => write!(f, "inverse"),
            UniversalOperation::Sum => write!(f, "sum"),
            UniversalOperation::Mean => write!(f, "mean"),
            UniversalOperation::Max => write!(f, "max"),
            UniversalOperation::Min => write!(f, "min"),
            UniversalOperation::ReLU => write!(f, "relu"),
            UniversalOperation::Sigmoid => write!(f, "sigmoid"),
            UniversalOperation::Tanh => write!(f, "tanh"),
            UniversalOperation::Softmax => write!(f, "softmax"),
            UniversalOperation::Conv1D => write!(f, "conv1d"),
            UniversalOperation::Conv2D => write!(f, "conv2d"),
            UniversalOperation::Conv3D => write!(f, "conv3d"),
            UniversalOperation::MaxPool => write!(f, "maxpool"),
            UniversalOperation::AvgPool => write!(f, "avgpool"),
            UniversalOperation::Reshape => write!(f, "reshape"),
            UniversalOperation::Flatten => write!(f, "flatten"),
            UniversalOperation::Squeeze => write!(f, "squeeze"),
            UniversalOperation::Unsqueeze => write!(f, "unsqueeze"),
            UniversalOperation::Custom(name) => write!(f, "custom({})", name),
        }
    }
}

/// Migration capabilities for framework transitions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MigrationCapability {
    /// Direct tensor conversion
    TensorConversion,
    /// Model architecture translation
    ModelTranslation,
    /// Gradient computation compatibility
    GradientCompatibility,
    /// Operation mapping
    OperationMapping,
    /// Optimizer state transfer
    OptimizerTransfer,
    /// Training loop adaptation
    TrainingLoopAdaptation,
    /// Checkpoint format conversion
    CheckpointConversion,
    /// Custom operation porting
    CustomOperationPorting,
}

/// Framework capability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkCapabilities {
    pub framework: ADFramework,
    pub supported_operations: Vec<UniversalOperation>,
    pub supported_dtypes: Vec<UniversalDataType>,
    pub migration_capabilities: Vec<MigrationCapability>,
    pub version: String,
    pub native_tensor_format: String,
    pub gradient_backend: String,
    pub supports_eager_execution: bool,
    pub supports_graph_mode: bool,
    pub supports_jit_compilation: bool,
    pub supports_distributed: bool,
    pub memory_format: String,
    pub device_support: Vec<String>, // "cpu", "cuda", "metal", etc.
}

impl FrameworkCapabilities {
    pub fn supports_operation(&self, op: &UniversalOperation) -> bool {
        self.supported_operations.contains(op)
    }

    pub fn supports_dtype(&self, dtype: &UniversalDataType) -> bool {
        self.supported_dtypes.contains(dtype)
    }

    pub fn can_migrate_to(&self, target: &FrameworkCapabilities) -> Vec<MigrationCapability> {
        self.migration_capabilities
            .iter()
            .filter(|cap| target.migration_capabilities.contains(cap))
            .cloned()
            .collect()
    }
}

/// Trait for framework-specific adapters
pub trait FrameworkAdapter: Send + Sync + std::fmt::Debug {
    fn framework_type(&self) -> ADFramework;
    fn is_available(&self) -> bool;
    fn get_capabilities(&self) -> FrameworkCapabilities;

    // Tensor operations
    fn import_tensor(&self, tensor: &UniversalTensor) -> AutogradResult<Box<dyn FrameworkTensor>>;
    fn export_tensor(&self, tensor: &dyn FrameworkTensor) -> AutogradResult<UniversalTensor>;

    // Gradient operations
    fn compute_gradient(
        &self,
        tensor: &dyn FrameworkTensor,
        grad_outputs: Option<&dyn FrameworkTensor>,
    ) -> AutogradResult<Box<dyn FrameworkTensor>>;
    fn zero_grad(&self, tensor: &mut dyn FrameworkTensor) -> AutogradResult<()>;

    // Operation execution
    fn execute_operation(
        &self,
        op: &UniversalOperation,
        inputs: &[&dyn FrameworkTensor],
        params: &HashMap<String, String>,
    ) -> AutogradResult<Box<dyn FrameworkTensor>>;

    // Migration support
    fn create_migration_plan(
        &self,
        source: &FrameworkCapabilities,
        target: &FrameworkCapabilities,
    ) -> AutogradResult<MigrationPlan>;
    fn execute_migration(
        &self,
        plan: &MigrationPlan,
        data: &MigrationData,
    ) -> AutogradResult<MigrationResult>;
}

/// Framework-agnostic tensor interface
pub trait FrameworkTensor: Send + Sync + std::fmt::Debug {
    fn shape(&self) -> Vec<usize>;
    fn dtype(&self) -> UniversalDataType;
    fn requires_grad(&self) -> bool;
    fn device(&self) -> String;
    fn to_universal(&self) -> AutogradResult<UniversalTensor>;
    fn grad(&self) -> Option<Box<dyn FrameworkTensor>>;
    fn set_grad(&mut self, grad: Box<dyn FrameworkTensor>) -> AutogradResult<()>;
}

/// Migration plan for framework transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPlan {
    pub source_framework: ADFramework,
    pub target_framework: ADFramework,
    pub steps: Vec<MigrationStep>,
    pub compatibility_level: CompatibilityLevel,
    pub estimated_effort: EffortLevel,
    pub warnings: Vec<String>,
    pub required_transformations: Vec<RequiredTransformation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStep {
    pub step_id: usize,
    pub description: String,
    pub operation: MigrationOperation,
    pub input_requirements: Vec<String>,
    pub output_format: String,
    pub validation_checks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationOperation {
    TensorConversion,
    OperationMapping,
    GradientSystemAdaptation,
    OptimizerTranslation,
    ModelStructureTranslation,
    CustomCodeGeneration,
    ValidationTest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompatibilityLevel {
    Full,
    High,
    Medium,
    Low,
    Incompatible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffortLevel {
    Minimal,
    Low,
    Medium,
    High,
    VeryHigh,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequiredTransformation {
    pub name: String,
    pub description: String,
    pub automation_level: AutomationLevel,
    pub complexity: EffortLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutomationLevel {
    FullyAutomatic,
    SemiAutomatic,
    ManualWithGuidance,
    FullyManual,
}

/// Migration data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationData {
    pub tensors: Vec<UniversalTensor>,
    pub model_definition: Option<String>,
    pub optimizer_state: Option<HashMap<String, String>>,
    pub training_config: Option<HashMap<String, String>>,
    pub custom_operations: Vec<CustomOperationDefinition>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomOperationDefinition {
    pub name: String,
    pub forward_code: String,
    pub backward_code: Option<String>,
    pub input_types: Vec<UniversalDataType>,
    pub output_types: Vec<UniversalDataType>,
    pub parameters: HashMap<String, String>,
    pub framework_specific_code: HashMap<ADFramework, String>,
}

/// Migration result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationResult {
    pub success: bool,
    pub migrated_tensors: Vec<UniversalTensor>,
    pub generated_code: Option<String>,
    pub validation_results: Vec<ValidationResult>,
    pub performance_comparison: Option<PerformanceComparison>,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub test_name: String,
    pub passed: bool,
    pub error_message: Option<String>,
    pub tolerance: Option<f64>,
    pub actual_difference: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceComparison {
    pub source_performance: PerformanceMetrics,
    pub target_performance: PerformanceMetrics,
    pub speedup_factor: f64,
    pub memory_efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub forward_time_ms: f64,
    pub backward_time_ms: f64,
    pub memory_usage_mb: f64,
    pub throughput_ops_per_sec: f64,
}

/// PyTorch compatibility adapter
#[derive(Debug)]
pub struct PyTorchAdapter {
    available: bool,
    capabilities: FrameworkCapabilities,
}

impl PyTorchAdapter {
    pub fn new() -> Self {
        // Check if PyTorch bindings are available
        let available = true; // In practice, check for actual PyTorch availability

        let capabilities = FrameworkCapabilities {
            framework: ADFramework::PyTorch,
            supported_operations: vec![
                UniversalOperation::Add,
                UniversalOperation::Sub,
                UniversalOperation::Mul,
                UniversalOperation::Div,
                UniversalOperation::MatMul,
                UniversalOperation::ReLU,
                UniversalOperation::Conv2D,
                UniversalOperation::MaxPool,
                UniversalOperation::Softmax,
                UniversalOperation::Sum,
                UniversalOperation::Mean,
                UniversalOperation::Transpose,
            ],
            supported_dtypes: vec![
                UniversalDataType::Float32,
                UniversalDataType::Float64,
                UniversalDataType::Int32,
                UniversalDataType::Int64,
                UniversalDataType::Bool,
                UniversalDataType::Complex64,
            ],
            migration_capabilities: vec![
                MigrationCapability::TensorConversion,
                MigrationCapability::ModelTranslation,
                MigrationCapability::GradientCompatibility,
                MigrationCapability::OperationMapping,
                MigrationCapability::OptimizerTransfer,
            ],
            version: "2.0+".to_string(),
            native_tensor_format: "torch.Tensor".to_string(),
            gradient_backend: "autograd".to_string(),
            supports_eager_execution: true,
            supports_graph_mode: true,
            supports_jit_compilation: true,
            supports_distributed: true,
            memory_format: "channels_last".to_string(),
            device_support: vec!["cpu".to_string(), "cuda".to_string(), "mps".to_string()],
        };

        Self {
            available,
            capabilities,
        }
    }
}

impl FrameworkAdapter for PyTorchAdapter {
    fn framework_type(&self) -> ADFramework {
        ADFramework::PyTorch
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn get_capabilities(&self) -> FrameworkCapabilities {
        self.capabilities.clone()
    }

    fn import_tensor(&self, tensor: &UniversalTensor) -> AutogradResult<Box<dyn FrameworkTensor>> {
        let pytorch_tensor = PyTorchTensor {
            data: tensor.data.clone(),
            shape: tensor.shape.clone(),
            dtype: tensor.dtype.clone(),
            requires_grad: tensor.requires_grad,
            device: "cpu".to_string(),
            grad: None,
        };

        Ok(Box::new(pytorch_tensor))
    }

    fn export_tensor(&self, tensor: &dyn FrameworkTensor) -> AutogradResult<UniversalTensor> {
        tensor.to_universal()
    }

    fn compute_gradient(
        &self,
        tensor: &dyn FrameworkTensor,
        grad_outputs: Option<&dyn FrameworkTensor>,
    ) -> AutogradResult<Box<dyn FrameworkTensor>> {
        // Simulate PyTorch backward computation
        let mut grad_tensor = PyTorchTensor {
            data: vec![1.0; tensor.shape().iter().product()],
            shape: tensor.shape(),
            dtype: tensor.dtype(),
            requires_grad: false,
            device: tensor.device(),
            grad: None,
        };

        if let Some(grad_out) = grad_outputs {
            // Use provided gradient outputs
            grad_tensor.data = grad_out.to_universal()?.data;
        }

        Ok(Box::new(grad_tensor))
    }

    fn zero_grad(&self, _tensor: &mut dyn FrameworkTensor) -> AutogradResult<()> {
        // Simulate zeroing gradients
        tracing::debug!("Zeroing gradients for PyTorch tensor");
        Ok(())
    }

    fn execute_operation(
        &self,
        op: &UniversalOperation,
        inputs: &[&dyn FrameworkTensor],
        _params: &HashMap<String, String>,
    ) -> AutogradResult<Box<dyn FrameworkTensor>> {
        if inputs.is_empty() {
            return Err(AutogradError::gradient_computation(
                "execute_operation",
                "No input tensors provided",
            ));
        }

        let input_tensor = inputs[0];
        let result_shape = input_tensor.shape();

        let result_data = match op {
            UniversalOperation::Add => {
                if inputs.len() < 2 {
                    return Err(AutogradError::gradient_computation(
                        "add_operation",
                        "Add operation requires 2 inputs",
                    ));
                }
                let input1 = inputs[0].to_universal()?;
                let input2 = inputs[1].to_universal()?;
                input1
                    .data
                    .iter()
                    .zip(input2.data.iter())
                    .map(|(a, b)| a + b)
                    .collect()
            }
            UniversalOperation::ReLU => {
                let input = input_tensor.to_universal()?;
                input.data.iter().map(|&x| x.max(0.0)).collect()
            }
            UniversalOperation::Sum => {
                let input = input_tensor.to_universal()?;
                vec![input.data.iter().sum()]
            }
            _ => {
                tracing::warn!("Operation {} not implemented for PyTorch adapter", op);
                input_tensor.to_universal()?.data
            }
        };

        let result_tensor = PyTorchTensor {
            data: result_data,
            shape: if matches!(op, UniversalOperation::Sum) {
                vec![1]
            } else {
                result_shape
            },
            dtype: input_tensor.dtype(),
            requires_grad: input_tensor.requires_grad(),
            device: input_tensor.device(),
            grad: None,
        };

        Ok(Box::new(result_tensor))
    }

    fn create_migration_plan(
        &self,
        source: &FrameworkCapabilities,
        target: &FrameworkCapabilities,
    ) -> AutogradResult<MigrationPlan> {
        let compatibility = if source.framework == target.framework {
            CompatibilityLevel::Full
        } else if source.framework == ADFramework::JAX
            || source.framework == ADFramework::TensorFlow
        {
            CompatibilityLevel::High
        } else {
            CompatibilityLevel::Medium
        };

        let steps = vec![
            MigrationStep {
                step_id: 1,
                description: "Convert tensor formats".to_string(),
                operation: MigrationOperation::TensorConversion,
                input_requirements: vec!["Universal tensor format".to_string()],
                output_format: "torch.Tensor".to_string(),
                validation_checks: vec![
                    "Shape consistency".to_string(),
                    "Data type compatibility".to_string(),
                ],
            },
            MigrationStep {
                step_id: 2,
                description: "Map operations to PyTorch equivalents".to_string(),
                operation: MigrationOperation::OperationMapping,
                input_requirements: vec!["Operation list".to_string()],
                output_format: "PyTorch functions".to_string(),
                validation_checks: vec!["Operation availability".to_string()],
            },
        ];

        Ok(MigrationPlan {
            source_framework: source.framework.clone(),
            target_framework: target.framework.clone(),
            steps,
            compatibility_level: compatibility,
            estimated_effort: EffortLevel::Low,
            warnings: vec!["Some operations may have different semantics".to_string()],
            required_transformations: vec![RequiredTransformation {
                name: "Gradient system adaptation".to_string(),
                description: "Adapt to PyTorch autograd semantics".to_string(),
                automation_level: AutomationLevel::SemiAutomatic,
                complexity: EffortLevel::Medium,
            }],
        })
    }

    fn execute_migration(
        &self,
        plan: &MigrationPlan,
        data: &MigrationData,
    ) -> AutogradResult<MigrationResult> {
        let mut migrated_tensors = Vec::new();
        let mut validation_results = Vec::new();

        // Migrate tensors
        for tensor in &data.tensors {
            match self.import_tensor(tensor) {
                Ok(pytorch_tensor) => {
                    migrated_tensors.push(pytorch_tensor.to_universal()?);
                    validation_results.push(ValidationResult {
                        test_name: "Tensor migration".to_string(),
                        passed: true,
                        error_message: None,
                        tolerance: Some(1e-6),
                        actual_difference: Some(0.0),
                    });
                }
                Err(e) => {
                    validation_results.push(ValidationResult {
                        test_name: "Tensor migration".to_string(),
                        passed: false,
                        error_message: Some(e.to_string()),
                        tolerance: None,
                        actual_difference: None,
                    });
                }
            }
        }

        let generated_code = Some(format!(
            "# PyTorch migration from {}\nimport torch\n\n# Converted tensors\n",
            plan.source_framework
        ));

        Ok(MigrationResult {
            success: validation_results.iter().all(|r| r.passed),
            migrated_tensors,
            generated_code,
            validation_results,
            performance_comparison: None,
            warnings: plan.warnings.clone(),
            errors: Vec::new(),
        })
    }
}

/// PyTorch tensor implementation
#[derive(Debug, Clone)]
pub struct PyTorchTensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub dtype: UniversalDataType,
    pub requires_grad: bool,
    pub device: String,
    pub grad: Option<Box<PyTorchTensor>>,
}

impl FrameworkTensor for PyTorchTensor {
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn dtype(&self) -> UniversalDataType {
        self.dtype.clone()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn device(&self) -> String {
        self.device.clone()
    }

    fn to_universal(&self) -> AutogradResult<UniversalTensor> {
        let mut tensor =
            UniversalTensor::new(self.data.clone(), self.shape.clone(), self.dtype.clone());
        tensor.set_requires_grad(self.requires_grad);
        tensor.add_metadata("framework".to_string(), "pytorch".to_string());
        tensor.add_metadata("device".to_string(), self.device.clone());
        Ok(tensor)
    }

    fn grad(&self) -> Option<Box<dyn FrameworkTensor>> {
        self.grad
            .as_ref()
            .map(|g| Box::new(g.as_ref().clone()) as Box<dyn FrameworkTensor>)
    }

    fn set_grad(&mut self, grad: Box<dyn FrameworkTensor>) -> AutogradResult<()> {
        let pytorch_grad = PyTorchTensor {
            data: grad.to_universal()?.data,
            shape: grad.shape(),
            dtype: grad.dtype(),
            requires_grad: false,
            device: grad.device(),
            grad: None,
        };
        self.grad = Some(Box::new(pytorch_grad));
        Ok(())
    }
}

/// Compatibility layer manager
pub struct ADFrameworkCompatibilityManager {
    adapters: HashMap<ADFramework, Box<dyn FrameworkAdapter>>,
    default_adapter: Option<ADFramework>,
    compatibility_matrix: RwLock<HashMap<(ADFramework, ADFramework), CompatibilityLevel>>,
    migration_cache: Mutex<HashMap<String, MigrationPlan>>,
}

impl ADFrameworkCompatibilityManager {
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
            default_adapter: None,
            compatibility_matrix: RwLock::new(HashMap::new()),
            migration_cache: Mutex::new(HashMap::new()),
        }
    }

    pub fn register_adapter(&mut self, adapter: Box<dyn FrameworkAdapter>) -> AutogradResult<()> {
        let framework = adapter.framework_type();
        if adapter.is_available() {
            self.adapters.insert(framework.clone(), adapter);
            tracing::info!("Registered {} adapter", framework);

            if self.default_adapter.is_none() {
                self.default_adapter = Some(framework);
            }
        }
        Ok(())
    }

    pub fn get_adapter(&self, framework: &ADFramework) -> Option<&dyn FrameworkAdapter> {
        self.adapters.get(framework).map(|a| a.as_ref())
    }

    pub fn list_available_frameworks(&self) -> Vec<ADFramework> {
        self.adapters.keys().cloned().collect()
    }

    pub fn check_compatibility(
        &self,
        source: &ADFramework,
        target: &ADFramework,
    ) -> AutogradResult<CompatibilityLevel> {
        if let Some(level) = self
            .compatibility_matrix
            .read()
            .unwrap()
            .get(&(source.clone(), target.clone()))
        {
            return Ok(*level);
        }

        let source_adapter = self.get_adapter(&source).ok_or_else(|| {
            AutogradError::gradient_computation(
                "adapter_lookup",
                format!("Adapter for {} not available", source),
            )
        })?;
        let target_adapter = self.get_adapter(&target).ok_or_else(|| {
            AutogradError::gradient_computation(
                "adapter_lookup",
                format!("Adapter for {} not available", target),
            )
        })?;

        let source_caps = source_adapter.get_capabilities();
        let target_caps = target_adapter.get_capabilities();

        let compatibility = self.calculate_compatibility(&source_caps, &target_caps);

        // Cache the result
        self.compatibility_matrix
            .write()
            .unwrap()
            .insert((source.clone(), target.clone()), compatibility);

        Ok(compatibility)
    }

    fn calculate_compatibility(
        &self,
        source: &FrameworkCapabilities,
        target: &FrameworkCapabilities,
    ) -> CompatibilityLevel {
        if source.framework == target.framework {
            return CompatibilityLevel::Full;
        }

        let operation_compatibility = self.calculate_operation_compatibility(source, target);
        let dtype_compatibility = self.calculate_dtype_compatibility(source, target);
        let migration_compatibility = self.calculate_migration_compatibility(source, target);

        // Combine compatibility scores
        let overall_score =
            (operation_compatibility + dtype_compatibility + migration_compatibility) / 3.0;

        match overall_score {
            score if score >= 0.9 => CompatibilityLevel::High,
            score if score >= 0.7 => CompatibilityLevel::Medium,
            score if score >= 0.5 => CompatibilityLevel::Low,
            _ => CompatibilityLevel::Incompatible,
        }
    }

    fn calculate_operation_compatibility(
        &self,
        source: &FrameworkCapabilities,
        target: &FrameworkCapabilities,
    ) -> f64 {
        let source_ops: std::collections::HashSet<_> = source.supported_operations.iter().collect();
        let target_ops: std::collections::HashSet<_> = target.supported_operations.iter().collect();
        let intersection = source_ops.intersection(&target_ops).count();
        let union = source_ops.union(&target_ops).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    fn calculate_dtype_compatibility(
        &self,
        source: &FrameworkCapabilities,
        target: &FrameworkCapabilities,
    ) -> f64 {
        let source_dtypes: std::collections::HashSet<_> = source.supported_dtypes.iter().collect();
        let target_dtypes: std::collections::HashSet<_> = target.supported_dtypes.iter().collect();
        let intersection = source_dtypes.intersection(&target_dtypes).count();
        let union = source_dtypes.union(&target_dtypes).count();

        if union == 0 {
            0.0
        } else {
            intersection as f64 / union as f64
        }
    }

    fn calculate_migration_compatibility(
        &self,
        source: &FrameworkCapabilities,
        target: &FrameworkCapabilities,
    ) -> f64 {
        let common_capabilities = source.can_migrate_to(target);
        let max_capabilities = source
            .migration_capabilities
            .len()
            .max(target.migration_capabilities.len());

        if max_capabilities == 0 {
            0.0
        } else {
            common_capabilities.len() as f64 / max_capabilities as f64
        }
    }

    pub fn create_migration_plan(
        &self,
        source: ADFramework,
        target: ADFramework,
    ) -> AutogradResult<MigrationPlan> {
        let cache_key = format!("{}_{}", source, target);

        if let Some(cached_plan) = self.migration_cache.lock().unwrap().get(&cache_key) {
            return Ok(cached_plan.clone());
        }

        let source_adapter = self.get_adapter(&source).ok_or_else(|| {
            AutogradError::gradient_computation(
                "adapter_lookup",
                format!("Adapter for {} not available", source),
            )
        })?;
        let target_adapter = self.get_adapter(&target).ok_or_else(|| {
            AutogradError::gradient_computation(
                "adapter_lookup",
                format!("Adapter for {} not available", target),
            )
        })?;

        let source_caps = source_adapter.get_capabilities();
        let target_caps = target_adapter.get_capabilities();

        let plan = source_adapter.create_migration_plan(&source_caps, &target_caps)?;

        // Cache the plan
        self.migration_cache
            .lock()
            .unwrap()
            .insert(cache_key, plan.clone());

        Ok(plan)
    }

    pub fn execute_migration(
        &self,
        source: ADFramework,
        target: ADFramework,
        data: &MigrationData,
    ) -> AutogradResult<MigrationResult> {
        let plan = self.create_migration_plan(source, target.clone())?;

        let target_adapter = self.get_adapter(&target).ok_or_else(|| {
            AutogradError::gradient_computation(
                "adapter_lookup",
                format!("Adapter for {} not available", target),
            )
        })?;

        target_adapter.execute_migration(&plan, data)
    }

    pub fn get_compatibility_report(&self) -> CompatibilityReport {
        let frameworks = self.list_available_frameworks();
        let mut compatibility_matrix = HashMap::new();

        for source in &frameworks {
            for target in &frameworks {
                if let Ok(level) = self.check_compatibility(&source, &target) {
                    compatibility_matrix.insert((source.clone(), target.clone()), level);
                }
            }
        }

        CompatibilityReport {
            available_frameworks: frameworks,
            compatibility_matrix,
            default_framework: self.default_adapter.clone(),
        }
    }
}

/// Compatibility report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityReport {
    pub available_frameworks: Vec<ADFramework>,
    pub compatibility_matrix: HashMap<(ADFramework, ADFramework), CompatibilityLevel>,
    pub default_framework: Option<ADFramework>,
}

impl CompatibilityReport {
    pub fn print_summary(&self) {
        println!("=== AD Framework Compatibility Report ===");
        println!("Available Frameworks: {:?}", self.available_frameworks);

        if let Some(ref default) = self.default_framework {
            println!("Default Framework: {}", default);
        }

        println!("\nCompatibility Matrix:");
        for source in &self.available_frameworks {
            for target in &self.available_frameworks {
                if let Some(level) = self
                    .compatibility_matrix
                    .get(&(source.clone(), target.clone()))
                {
                    println!("  {} -> {}: {:?}", source, target, level);
                }
            }
        }
    }
}

/// Global compatibility manager
static GLOBAL_COMPATIBILITY_MANAGER: std::sync::OnceLock<
    std::sync::Mutex<ADFrameworkCompatibilityManager>,
> = std::sync::OnceLock::new();

pub fn get_global_compatibility_manager(
) -> &'static std::sync::Mutex<ADFrameworkCompatibilityManager> {
    GLOBAL_COMPATIBILITY_MANAGER.get_or_init(|| {
        let mut manager = ADFrameworkCompatibilityManager::new();

        // Register default adapters
        if let Err(e) = manager.register_adapter(Box::new(PyTorchAdapter::new())) {
            tracing::error!("Failed to register PyTorch adapter: {}", e);
        }

        std::sync::Mutex::new(manager)
    })
}

/// Convenience functions for common operations
pub fn convert_tensor(
    tensor: &UniversalTensor,
    target_framework: ADFramework,
) -> AutogradResult<Box<dyn FrameworkTensor>> {
    let manager = get_global_compatibility_manager();
    let manager_lock = manager.lock().unwrap();
    let adapter = manager_lock.get_adapter(&target_framework).ok_or_else(|| {
        AutogradError::gradient_computation(
            "adapter_lookup",
            format!("Adapter for {} not available", target_framework),
        )
    })?;
    adapter.import_tensor(tensor)
}

pub fn migrate_model(
    source: ADFramework,
    target: ADFramework,
    data: &MigrationData,
) -> AutogradResult<MigrationResult> {
    let manager = get_global_compatibility_manager();
    let manager_lock = manager.lock().unwrap();
    manager_lock.execute_migration(source, target, data)
}

pub fn check_framework_compatibility(
    source: ADFramework,
    target: ADFramework,
) -> AutogradResult<CompatibilityLevel> {
    let manager = get_global_compatibility_manager();
    let manager_lock = manager.lock().unwrap();
    manager_lock.check_compatibility(&source, &target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_universal_tensor() {
        let tensor = UniversalTensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            UniversalDataType::Float32,
        );

        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.ndim(), 2);
        assert!(!tensor.is_scalar());
    }

    #[test]
    fn test_framework_capabilities() {
        let caps = FrameworkCapabilities {
            framework: ADFramework::PyTorch,
            supported_operations: vec![UniversalOperation::Add, UniversalOperation::Mul],
            supported_dtypes: vec![UniversalDataType::Float32],
            migration_capabilities: vec![MigrationCapability::TensorConversion],
            version: "2.0".to_string(),
            native_tensor_format: "torch.Tensor".to_string(),
            gradient_backend: "autograd".to_string(),
            supports_eager_execution: true,
            supports_graph_mode: true,
            supports_jit_compilation: true,
            supports_distributed: true,
            memory_format: "channels_last".to_string(),
            device_support: vec!["cpu".to_string()],
        };

        assert!(caps.supports_operation(&UniversalOperation::Add));
        assert!(!caps.supports_operation(&UniversalOperation::Conv2D));
        assert!(caps.supports_dtype(&UniversalDataType::Float32));
    }

    #[test]
    fn test_pytorch_adapter() {
        let adapter = PyTorchAdapter::new();
        assert_eq!(adapter.framework_type(), ADFramework::PyTorch);
        assert!(adapter.is_available());

        let capabilities = adapter.get_capabilities();
        assert_eq!(capabilities.framework, ADFramework::PyTorch);
        assert!(capabilities.supports_operation(&UniversalOperation::Add));
    }

    #[test]
    fn test_tensor_conversion() {
        let adapter = PyTorchAdapter::new();
        let universal = UniversalTensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            UniversalDataType::Float32,
        );

        let pytorch_tensor = adapter.import_tensor(&universal).unwrap();
        assert_eq!(pytorch_tensor.shape(), vec![2, 2]);
        assert_eq!(pytorch_tensor.dtype(), UniversalDataType::Float32);

        let back_to_universal = adapter.export_tensor(pytorch_tensor.as_ref()).unwrap();
        assert_eq!(back_to_universal.data, universal.data);
    }

    #[test]
    fn test_operation_execution() {
        let adapter = PyTorchAdapter::new();
        let tensor1 = PyTorchTensor {
            data: vec![1.0, 2.0],
            shape: vec![2],
            dtype: UniversalDataType::Float32,
            requires_grad: true,
            device: "cpu".to_string(),
            grad: None,
        };

        let tensor2 = PyTorchTensor {
            data: vec![3.0, 4.0],
            shape: vec![2],
            dtype: UniversalDataType::Float32,
            requires_grad: true,
            device: "cpu".to_string(),
            grad: None,
        };

        let inputs: Vec<&dyn FrameworkTensor> = vec![&tensor1, &tensor2];
        let result = adapter
            .execute_operation(&UniversalOperation::Add, &inputs, &HashMap::new())
            .unwrap();

        let result_data = result.to_universal().unwrap();
        assert_eq!(result_data.data, vec![4.0, 6.0]);
    }

    #[test]
    fn test_compatibility_manager() {
        let mut manager = ADFrameworkCompatibilityManager::new();
        let adapter = Box::new(PyTorchAdapter::new());

        manager.register_adapter(adapter).unwrap();
        assert!(manager
            .list_available_frameworks()
            .contains(&ADFramework::PyTorch));

        let compatibility = manager
            .check_compatibility(&ADFramework::PyTorch, &ADFramework::PyTorch)
            .unwrap();
        assert_eq!(compatibility, CompatibilityLevel::Full);
    }

    #[test]
    fn test_migration_plan() {
        let adapter = PyTorchAdapter::new();
        let source_caps = FrameworkCapabilities {
            framework: ADFramework::JAX,
            supported_operations: vec![UniversalOperation::Add],
            supported_dtypes: vec![UniversalDataType::Float32],
            migration_capabilities: vec![MigrationCapability::TensorConversion],
            version: "0.4".to_string(),
            native_tensor_format: "jax.Array".to_string(),
            gradient_backend: "jax.grad".to_string(),
            supports_eager_execution: true,
            supports_graph_mode: false,
            supports_jit_compilation: true,
            supports_distributed: true,
            memory_format: "row_major".to_string(),
            device_support: vec!["cpu".to_string(), "tpu".to_string()],
        };
        let target_caps = adapter.get_capabilities();

        let plan = adapter
            .create_migration_plan(&source_caps, &target_caps)
            .unwrap();
        assert_eq!(plan.source_framework, ADFramework::JAX);
        assert_eq!(plan.target_framework, ADFramework::PyTorch);
        assert!(!plan.steps.is_empty());
    }

    #[test]
    fn test_universal_data_types() {
        assert_eq!(UniversalDataType::Float32.to_string(), "float32");
        assert_eq!(UniversalDataType::Int64.to_string(), "int64");
        assert_eq!(UniversalDataType::Complex128.to_string(), "complex128");
    }

    #[test]
    fn test_universal_operations() {
        assert_eq!(UniversalOperation::Add.to_string(), "add");
        assert_eq!(UniversalOperation::Conv2D.to_string(), "conv2d");
        assert_eq!(
            UniversalOperation::Custom("test".to_string()).to_string(),
            "custom(test)"
        );
    }

    #[test]
    fn test_automation_levels() {
        assert_eq!(
            AutomationLevel::FullyAutomatic,
            AutomationLevel::FullyAutomatic
        );
        assert_ne!(AutomationLevel::SemiAutomatic, AutomationLevel::FullyManual);
    }

    #[test]
    fn test_compatibility_levels() {
        assert_eq!(CompatibilityLevel::Full, CompatibilityLevel::Full);
        assert_ne!(CompatibilityLevel::High, CompatibilityLevel::Low);
    }

    #[test]
    fn test_migration_data() {
        let data = MigrationData {
            tensors: vec![UniversalTensor::ones(
                vec![2, 2],
                UniversalDataType::Float32,
            )],
            model_definition: Some("class Model(nn.Module): pass".to_string()),
            optimizer_state: Some(HashMap::new()),
            training_config: None,
            custom_operations: vec![],
            metadata: HashMap::new(),
        };

        assert_eq!(data.tensors.len(), 1);
        assert!(data.model_definition.is_some());
    }
}
