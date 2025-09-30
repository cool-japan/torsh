//! C-compatible types and handles for ToRSh FFI
//!
//! This module defines all C-compatible types, opaque handles, enums, and type
//! conversions needed for the ToRSh C API. It provides a clean interface between
//! Rust's type system and C's more limited type system.
//!
//! # Overview
//!
//! The C API uses opaque handles to represent complex Rust objects safely:
//!
//! - **TensorHandle**: Represents tensor objects across the C boundary
//! - **ModuleHandle**: Represents neural network modules
//! - **OptimizerHandle**: Represents optimization algorithms
//!
//! # Type Safety
//!
//! All handles are opaque pointers that cannot be dereferenced from C code.
//! This ensures memory safety while allowing efficient data passing.
//!
//! # Examples
//!
//! ```c
//! // C usage example
//! TensorHandle tensor = torsh_tensor_new(data, shape, 2, TORSH_F32);
//! TorshDType dtype = torsh_tensor_dtype(tensor);
//! torsh_tensor_free(tensor);
//! ```

use std::os::raw::{c_char, c_float, c_int, c_void};
use torsh_core::DType;

/// Opaque handle for Tensor objects
///
/// This is an opaque pointer type that represents a tensor in C code.
/// C code cannot directly access the tensor data; it must use the provided
/// API functions to interact with tensors.
///
/// # Safety
///
/// - Never dereference this pointer from C code
/// - Always use the provided API functions
/// - Always call `torsh_tensor_free` when done with a tensor
#[repr(C)]
pub struct TorshTensor {
    _private: [u8; 0],
}

/// Type alias for tensor handle (used across language bindings)
///
/// This type alias provides a more semantic name for tensor handles
/// and is used consistently across different language bindings.
pub type TensorHandle = *mut TorshTensor;

/// Opaque handle for Module objects
///
/// Represents neural network modules (layers, networks) in C code.
/// Like tensor handles, this is opaque and must be manipulated through
/// the provided API functions.
///
/// # Supported Module Types
///
/// - Linear layers (fully connected)
/// - Convolutional layers (future)
/// - Activation layers (future)
/// - Custom modules (future)
#[repr(C)]
pub struct TorshModule {
    _private: [u8; 0],
}

/// Type alias for module handle
pub type ModuleHandle = *mut TorshModule;

/// Opaque handle for Optimizer objects
///
/// Represents optimization algorithms in C code. Optimizers maintain
/// internal state for training neural networks.
///
/// # Supported Optimizers
///
/// - SGD (Stochastic Gradient Descent)
/// - Adam (Adaptive Moment Estimation)
/// - AdamW (future)
/// - RMSprop (future)
#[repr(C)]
pub struct TorshOptimizer {
    _private: [u8; 0],
}

/// Type alias for optimizer handle
pub type OptimizerHandle = *mut TorshOptimizer;

/// Data types for C API
///
/// Represents the supported data types for tensors in the C API.
/// These correspond to common numerical types used in machine learning.
///
/// # Type Mapping
///
/// - `F32`: 32-bit floating point (most common for ML)
/// - `F64`: 64-bit floating point (high precision)
/// - `I32`: 32-bit signed integer
/// - `I64`: 64-bit signed integer
/// - `U8`: 8-bit unsigned integer (often used for images)
///
/// # Examples
///
/// ```c
/// // Create a float32 tensor
/// TensorHandle tensor = torsh_tensor_new(data, shape, 2, TORSH_F32);
///
/// // Check tensor type
/// TorshDType dtype = torsh_tensor_dtype(tensor);
/// if (dtype == TORSH_F32) {
///     printf("Tensor is float32\n");
/// }
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TorshDType {
    /// 32-bit floating point
    F32 = 0,
    /// 64-bit floating point
    F64 = 1,
    /// 32-bit signed integer
    I32 = 2,
    /// 64-bit signed integer
    I64 = 3,
    /// 8-bit unsigned integer
    U8 = 4,
}

/// Error codes for C API
///
/// Enhanced error code system with granular error types for better debugging
/// and error handling. These provide detailed information about what went
/// wrong during API calls, enabling more precise error recovery strategies.
///
/// # Error Categories
///
/// ## Success
/// - `Success`: Operation completed successfully
///
/// ## Input Validation Errors
/// - `InvalidArgument`: Invalid parameters passed to function
/// - `NullPointer`: Null pointer provided where non-null expected
/// - `InvalidShape`: Invalid tensor shape specification
/// - `InvalidIndex`: Array index out of bounds
/// - `InvalidDimension`: Invalid dimension specification
/// - `ShapeMismatch`: Tensor shapes are incompatible for operation
/// - `TypeMismatch`: Tensor types are incompatible
/// - `SizeMismatch`: Data size doesn't match expected size
///
/// ## Memory Management Errors
/// - `OutOfMemory`: Memory allocation failed
/// - `MemoryAlignment`: Memory alignment requirements not met
/// - `BufferOverflow`: Buffer overflow detected
/// - `MemoryCorruption`: Memory corruption detected
///
/// ## Device and Backend Errors
/// - `DeviceError`: Device operation failed
/// - `CudaError`: CUDA-specific error occurred
/// - `ComputeError`: Computation backend error
/// - `DeviceNotAvailable`: Requested device not available
///
/// ## Mathematical and Numerical Errors
/// - `NumericalError`: Numerical computation error (NaN, infinity)
/// - `ConvergenceError`: Algorithm failed to converge
/// - `PrecisionLoss`: Significant precision loss detected
/// - `DivisionByZero`: Division by zero attempted
///
/// ## System and I/O Errors
/// - `IoError`: Input/output operation failed
/// - `SerializationError`: Data serialization/deserialization failed
/// - `NetworkError`: Network communication error
/// - `FileSystemError`: File system operation failed
///
/// ## Threading and Concurrency Errors
/// - `ThreadError`: Threading operation failed
/// - `LockError`: Lock acquisition failed
/// - `ConcurrencyError`: Concurrency violation detected
///
/// ## Generic Error Types
/// - `RuntimeError`: General runtime error occurred
/// - `NotImplemented`: Requested feature is not implemented
/// - `InternalError`: Internal consistency check failed
///
/// # Usage Pattern
///
/// ```c
/// TorshError result = torsh_tensor_add(a, b, output);
/// if (result != TORSH_SUCCESS) {
///     const char* error_msg = torsh_get_last_error();
///     fprintf(stderr, "Error: %s\n", error_msg);
///
///     // Handle specific error types
///     switch (result) {
///         case TORSH_SHAPE_MISMATCH:
///             fprintf(stderr, "Tensor shapes incompatible\n");
///             break;
///         case TORSH_OUT_OF_MEMORY:
///             fprintf(stderr, "Insufficient memory\n");
///             break;
///         default:
///             fprintf(stderr, "Unknown error\n");
///     }
///     return -1;
/// }
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TorshError {
    /// Operation completed successfully
    Success = 0,

    // Input Validation Errors (1-10)
    /// Invalid argument provided to function
    InvalidArgument = 1,
    /// Null pointer provided where non-null expected
    NullPointer = 2,
    /// Invalid tensor shape specification
    InvalidShape = 3,
    /// Array index out of bounds
    InvalidIndex = 4,
    /// Invalid dimension specification
    InvalidDimension = 5,
    /// Tensor shapes are incompatible for operation
    ShapeMismatch = 6,
    /// Tensor data types are incompatible
    TypeMismatch = 7,
    /// Data size doesn't match expected size
    SizeMismatch = 8,

    // Memory Management Errors (11-20)
    /// Memory allocation failed
    OutOfMemory = 11,
    /// Memory alignment requirements not met
    MemoryAlignment = 12,
    /// Buffer overflow detected
    BufferOverflow = 13,
    /// Memory corruption detected
    MemoryCorruption = 14,

    // Device and Backend Errors (21-30)
    /// Device operation failed
    DeviceError = 21,
    /// CUDA-specific error occurred
    CudaError = 22,
    /// Computation backend error
    ComputeError = 23,
    /// Requested device not available
    DeviceNotAvailable = 24,

    // Mathematical and Numerical Errors (31-40)
    /// Numerical computation error (NaN, infinity)
    NumericalError = 31,
    /// Algorithm failed to converge
    ConvergenceError = 32,
    /// Significant precision loss detected
    PrecisionLoss = 33,
    /// Division by zero attempted
    DivisionByZero = 34,

    // System and I/O Errors (41-50)
    /// Input/output operation failed
    IoError = 41,
    /// Data serialization/deserialization failed
    SerializationError = 42,
    /// Network communication error
    NetworkError = 43,
    /// File system operation failed
    FileSystemError = 44,

    // Threading and Concurrency Errors (51-60)
    /// Threading operation failed
    ThreadError = 51,
    /// Lock acquisition failed
    LockError = 52,
    /// Concurrency violation detected
    ConcurrencyError = 53,

    // Generic Error Types (61-70)
    /// General runtime error occurred
    RuntimeError = 61,
    /// Requested feature is not implemented
    NotImplemented = 62,
    /// Internal consistency check failed
    InternalError = 63,
}

/// Internal implementation for tensors
///
/// This structure holds the actual tensor data in Rust. It is not exposed
/// to C code directly but is accessed through the opaque handle system.
///
/// # Memory Layout
///
/// The tensor data is stored in row-major order, which is compatible with
/// most C/C++ libraries and matches NumPy's default layout.
#[derive(Clone, Debug)]
pub struct TensorImpl {
    /// Raw tensor data (always stored as f32 for simplicity)
    pub data: Vec<f32>,
    /// Shape of the tensor (dimensions)
    pub shape: Vec<usize>,
    /// Data type of the original data
    pub dtype: DType,
}

impl TensorImpl {
    /// Create a new tensor implementation
    ///
    /// # Arguments
    ///
    /// * `data` - Vector of f32 values
    /// * `shape` - Shape dimensions
    /// * `dtype` - Original data type
    pub fn new(data: Vec<f32>, shape: Vec<usize>, dtype: DType) -> Self {
        Self { data, shape, dtype }
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Validate that the data length matches the shape
    pub fn is_valid(&self) -> bool {
        self.data.len() == self.numel()
    }
}

/// Internal implementation for neural network modules
///
/// Holds the parameters and configuration for a neural network module.
/// Currently supports linear layers with optional bias.
#[derive(Clone, Debug)]
pub struct ModuleImpl {
    /// Type of module (e.g., "linear", "conv2d")
    pub module_type: String,
    /// Number of input features
    pub in_features: usize,
    /// Number of output features
    pub out_features: usize,
    /// Whether bias is enabled
    pub bias: bool,
    /// Weight parameters (flattened)
    pub weight: Vec<f32>,
    /// Bias parameters (if enabled)
    pub bias_data: Option<Vec<f32>>,
}

impl ModuleImpl {
    /// Create a new linear module
    ///
    /// # Arguments
    ///
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    /// * `bias` - Whether to include bias parameters
    /// * `weight` - Weight parameters
    /// * `bias_data` - Bias parameters (if bias is true)
    pub fn new_linear(
        in_features: usize,
        out_features: usize,
        bias: bool,
        weight: Vec<f32>,
        bias_data: Option<Vec<f32>>,
    ) -> Self {
        Self {
            module_type: "linear".to_string(),
            in_features,
            out_features,
            bias,
            weight,
            bias_data,
        }
    }

    /// Get the expected weight size
    pub fn weight_size(&self) -> usize {
        self.in_features * self.out_features
    }

    /// Get the expected bias size
    pub fn bias_size(&self) -> Option<usize> {
        if self.bias {
            Some(self.out_features)
        } else {
            None
        }
    }

    /// Validate the module parameters
    pub fn is_valid(&self) -> bool {
        // Check weight size
        if self.weight.len() != self.weight_size() {
            return false;
        }

        // Check bias size if bias is enabled
        if self.bias {
            if let Some(ref bias_data) = self.bias_data {
                if bias_data.len() != self.out_features {
                    return false;
                }
            } else {
                return false; // Bias enabled but no bias data
            }
        }

        true
    }
}

/// Internal implementation for optimizers
///
/// Holds the configuration and state for optimization algorithms.
/// Supports different optimizer types with their specific parameters.
#[derive(Clone, Debug)]
pub struct OptimizerImpl {
    /// Type of optimizer (e.g., "sgd", "adam")
    pub optimizer_type: String,
    /// Learning rate
    pub learning_rate: f32,
    /// Momentum (for SGD)
    pub momentum: Option<f32>,
    /// Beta1 parameter (for Adam)
    pub beta1: Option<f32>,
    /// Beta2 parameter (for Adam)
    pub beta2: Option<f32>,
    /// Epsilon parameter (for Adam)
    pub epsilon: Option<f32>,
}

impl OptimizerImpl {
    /// Create a new SGD optimizer
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Learning rate for parameter updates
    /// * `momentum` - Momentum factor (0.0 for no momentum)
    pub fn new_sgd(learning_rate: f32, momentum: f32) -> Self {
        Self {
            optimizer_type: "sgd".to_string(),
            learning_rate,
            momentum: if momentum > 0.0 { Some(momentum) } else { None },
            beta1: None,
            beta2: None,
            epsilon: None,
        }
    }

    /// Create a new Adam optimizer
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - Learning rate for parameter updates
    /// * `beta1` - Exponential decay rate for first moment estimates
    /// * `beta2` - Exponential decay rate for second moment estimates
    /// * `epsilon` - Small constant for numerical stability
    pub fn new_adam(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        Self {
            optimizer_type: "adam".to_string(),
            learning_rate,
            momentum: None,
            beta1: Some(beta1),
            beta2: Some(beta2),
            epsilon: Some(epsilon),
        }
    }

    /// Get the optimizer type
    pub fn optimizer_type(&self) -> &str {
        &self.optimizer_type
    }

    /// Validate the optimizer configuration
    pub fn is_valid(&self) -> bool {
        match self.optimizer_type.as_str() {
            "sgd" => self.learning_rate > 0.0,
            "adam" => {
                self.learning_rate > 0.0
                    && self.beta1.map_or(false, |b| b >= 0.0 && b < 1.0)
                    && self.beta2.map_or(false, |b| b >= 0.0 && b < 1.0)
                    && self.epsilon.map_or(false, |e| e > 0.0)
            }
            _ => false,
        }
    }
}

/// Convert Rust DType to C API TorshDType
///
/// Provides type conversion from Rust's internal DType enum to the
/// C API's TorshDType enum for cross-language compatibility.
///
/// # Examples
///
/// ```rust
/// let rust_type = DType::F32;
/// let c_type: TorshDType = rust_type.into();
/// assert_eq!(c_type, TorshDType::F32);
/// ```
impl From<DType> for TorshDType {
    fn from(dtype: DType) -> Self {
        match dtype {
            DType::F32 => TorshDType::F32,
            DType::F64 => TorshDType::F64,
            DType::I32 => TorshDType::I32,
            DType::I64 => TorshDType::I64,
            DType::U8 => TorshDType::U8,
            _ => TorshDType::F32, // Default fallback for unsupported types
        }
    }
}

/// Convert C API TorshDType to Rust DType
///
/// Provides type conversion from the C API's TorshDType enum to
/// Rust's internal DType enum.
///
/// # Examples
///
/// ```rust
/// let c_type = TorshDType::F32;
/// let rust_type: DType = c_type.into();
/// assert_eq!(rust_type, DType::F32);
/// ```
impl From<TorshDType> for DType {
    fn from(dtype: TorshDType) -> Self {
        match dtype {
            TorshDType::F32 => DType::F32,
            TorshDType::F64 => DType::F64,
            TorshDType::I32 => DType::I32,
            TorshDType::I64 => DType::I64,
            TorshDType::U8 => DType::U8,
        }
    }
}

/// Type conversion utilities for C data types
impl TorshDType {
    /// Get the size in bytes of this data type
    ///
    /// Returns the number of bytes required to store one element
    /// of this data type.
    ///
    /// # Examples
    ///
    /// ```rust
    /// assert_eq!(TorshDType::F32.size_in_bytes(), 4);
    /// assert_eq!(TorshDType::F64.size_in_bytes(), 8);
    /// assert_eq!(TorshDType::U8.size_in_bytes(), 1);
    /// ```
    pub fn size_in_bytes(self) -> usize {
        match self {
            TorshDType::F32 => 4,
            TorshDType::F64 => 8,
            TorshDType::I32 => 4,
            TorshDType::I64 => 8,
            TorshDType::U8 => 1,
        }
    }

    /// Check if this is a floating point type
    ///
    /// Returns true if the type represents floating point numbers.
    pub fn is_floating_point(self) -> bool {
        matches!(self, TorshDType::F32 | TorshDType::F64)
    }

    /// Check if this is an integer type
    ///
    /// Returns true if the type represents integer numbers.
    pub fn is_integer(self) -> bool {
        matches!(self, TorshDType::I32 | TorshDType::I64 | TorshDType::U8)
    }

    /// Get a string representation of the type
    ///
    /// Returns a human-readable name for the data type.
    pub fn as_str(self) -> &'static str {
        match self {
            TorshDType::F32 => "f32",
            TorshDType::F64 => "f64",
            TorshDType::I32 => "i32",
            TorshDType::I64 => "i64",
            TorshDType::U8 => "u8",
        }
    }
}

/// Error code utilities
impl TorshError {
    /// Check if the error code indicates success
    ///
    /// Returns true if the error code represents successful operation.
    pub fn is_success(self) -> bool {
        self == TorshError::Success
    }

    /// Check if the error code indicates an error
    ///
    /// Returns true if the error code represents a failure condition.
    pub fn is_error(self) -> bool {
        self != TorshError::Success
    }

    /// Get a string representation of the error
    ///
    /// Returns a human-readable description of the error code.
    pub fn as_str(self) -> &'static str {
        match self {
            // Success
            TorshError::Success => "Success",

            // Input Validation Errors
            TorshError::InvalidArgument => "Invalid argument",
            TorshError::NullPointer => "Null pointer provided",
            TorshError::InvalidShape => "Invalid tensor shape",
            TorshError::InvalidIndex => "Array index out of bounds",
            TorshError::InvalidDimension => "Invalid dimension specification",
            TorshError::ShapeMismatch => "Tensor shapes incompatible",
            TorshError::TypeMismatch => "Tensor types incompatible",
            TorshError::SizeMismatch => "Data size mismatch",

            // Memory Management Errors
            TorshError::OutOfMemory => "Memory allocation failed",
            TorshError::MemoryAlignment => "Memory alignment requirements not met",
            TorshError::BufferOverflow => "Buffer overflow detected",
            TorshError::MemoryCorruption => "Memory corruption detected",

            // Device and Backend Errors
            TorshError::DeviceError => "Device operation failed",
            TorshError::CudaError => "CUDA error occurred",
            TorshError::ComputeError => "Computation backend error",
            TorshError::DeviceNotAvailable => "Requested device not available",

            // Mathematical and Numerical Errors
            TorshError::NumericalError => "Numerical computation error",
            TorshError::ConvergenceError => "Algorithm failed to converge",
            TorshError::PrecisionLoss => "Significant precision loss detected",
            TorshError::DivisionByZero => "Division by zero attempted",

            // System and I/O Errors
            TorshError::IoError => "Input/output operation failed",
            TorshError::SerializationError => "Serialization/deserialization failed",
            TorshError::NetworkError => "Network communication error",
            TorshError::FileSystemError => "File system operation failed",

            // Threading and Concurrency Errors
            TorshError::ThreadError => "Threading operation failed",
            TorshError::LockError => "Lock acquisition failed",
            TorshError::ConcurrencyError => "Concurrency violation detected",

            // Generic Error Types
            TorshError::RuntimeError => "Runtime error occurred",
            TorshError::NotImplemented => "Feature not implemented",
            TorshError::InternalError => "Internal consistency check failed",
        }
    }

    /// Get error category for grouped error handling
    ///
    /// Returns a category string for the error type, useful for implementing
    /// category-based error handling strategies.
    pub fn category(self) -> &'static str {
        match self {
            TorshError::Success => "success",

            TorshError::InvalidArgument
            | TorshError::NullPointer
            | TorshError::InvalidShape
            | TorshError::InvalidIndex
            | TorshError::InvalidDimension
            | TorshError::ShapeMismatch
            | TorshError::TypeMismatch
            | TorshError::SizeMismatch => "validation",

            TorshError::OutOfMemory
            | TorshError::MemoryAlignment
            | TorshError::BufferOverflow
            | TorshError::MemoryCorruption => "memory",

            TorshError::DeviceError
            | TorshError::CudaError
            | TorshError::ComputeError
            | TorshError::DeviceNotAvailable => "device",

            TorshError::NumericalError
            | TorshError::ConvergenceError
            | TorshError::PrecisionLoss
            | TorshError::DivisionByZero => "numerical",

            TorshError::IoError
            | TorshError::SerializationError
            | TorshError::NetworkError
            | TorshError::FileSystemError => "io",

            TorshError::ThreadError | TorshError::LockError | TorshError::ConcurrencyError => {
                "threading"
            }

            TorshError::RuntimeError | TorshError::NotImplemented | TorshError::InternalError => {
                "generic"
            }
        }
    }

    /// Check if error is recoverable
    ///
    /// Returns true if the error represents a potentially recoverable condition
    /// that might succeed if retried or handled appropriately.
    pub fn is_recoverable(self) -> bool {
        matches!(
            self,
            TorshError::Success
                | TorshError::DeviceNotAvailable
                | TorshError::ConvergenceError
                | TorshError::IoError
                | TorshError::NetworkError
                | TorshError::FileSystemError
                | TorshError::LockError
        )
    }

    /// Check if error indicates a critical system failure
    ///
    /// Returns true if the error represents a critical failure that likely
    /// requires immediate attention or system restart.
    pub fn is_critical(self) -> bool {
        matches!(
            self,
            TorshError::MemoryCorruption
                | TorshError::BufferOverflow
                | TorshError::InternalError
                | TorshError::ConcurrencyError
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_conversions() {
        // Test Rust to C conversion
        let rust_f32 = DType::F32;
        let c_f32: TorshDType = rust_f32.into();
        assert_eq!(c_f32, TorshDType::F32);

        // Test C to Rust conversion
        let c_f64 = TorshDType::F64;
        let rust_f64: DType = c_f64.into();
        assert_eq!(rust_f64, DType::F64);
    }

    #[test]
    fn test_dtype_utilities() {
        assert_eq!(TorshDType::F32.size_in_bytes(), 4);
        assert_eq!(TorshDType::F64.size_in_bytes(), 8);
        assert_eq!(TorshDType::I32.size_in_bytes(), 4);
        assert_eq!(TorshDType::I64.size_in_bytes(), 8);
        assert_eq!(TorshDType::U8.size_in_bytes(), 1);

        assert!(TorshDType::F32.is_floating_point());
        assert!(TorshDType::F64.is_floating_point());
        assert!(!TorshDType::I32.is_floating_point());

        assert!(!TorshDType::F32.is_integer());
        assert!(TorshDType::I32.is_integer());
        assert!(TorshDType::U8.is_integer());
    }

    #[test]
    fn test_dtype_string_representation() {
        assert_eq!(TorshDType::F32.as_str(), "f32");
        assert_eq!(TorshDType::F64.as_str(), "f64");
        assert_eq!(TorshDType::I32.as_str(), "i32");
        assert_eq!(TorshDType::I64.as_str(), "i64");
        assert_eq!(TorshDType::U8.as_str(), "u8");
    }

    #[test]
    fn test_error_utilities() {
        // Test basic success/error status
        assert!(TorshError::Success.is_success());
        assert!(!TorshError::Success.is_error());

        assert!(!TorshError::InvalidArgument.is_success());
        assert!(TorshError::InvalidArgument.is_error());

        // Test string representations
        assert_eq!(TorshError::Success.as_str(), "Success");
        assert_eq!(TorshError::InvalidArgument.as_str(), "Invalid argument");
        assert_eq!(TorshError::OutOfMemory.as_str(), "Memory allocation failed");
        assert_eq!(
            TorshError::ShapeMismatch.as_str(),
            "Tensor shapes incompatible"
        );
        assert_eq!(TorshError::CudaError.as_str(), "CUDA error occurred");
        assert_eq!(
            TorshError::NumericalError.as_str(),
            "Numerical computation error"
        );

        // Test error categories
        assert_eq!(TorshError::Success.category(), "success");
        assert_eq!(TorshError::InvalidArgument.category(), "validation");
        assert_eq!(TorshError::OutOfMemory.category(), "memory");
        assert_eq!(TorshError::CudaError.category(), "device");
        assert_eq!(TorshError::DivisionByZero.category(), "numerical");
        assert_eq!(TorshError::IoError.category(), "io");
        assert_eq!(TorshError::ThreadError.category(), "threading");
        assert_eq!(TorshError::RuntimeError.category(), "generic");

        // Test recoverability
        assert!(TorshError::Success.is_recoverable());
        assert!(TorshError::DeviceNotAvailable.is_recoverable());
        assert!(TorshError::IoError.is_recoverable());
        assert!(!TorshError::InvalidArgument.is_recoverable());
        assert!(!TorshError::OutOfMemory.is_recoverable());

        // Test critical errors
        assert!(!TorshError::Success.is_critical());
        assert!(!TorshError::InvalidArgument.is_critical());
        assert!(TorshError::MemoryCorruption.is_critical());
        assert!(TorshError::BufferOverflow.is_critical());
        assert!(TorshError::InternalError.is_critical());
    }

    #[test]
    fn test_tensor_impl() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = TensorImpl::new(data, shape, DType::F32);

        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.ndim(), 2);
        assert!(tensor.is_valid());

        // Test invalid tensor
        let invalid_data = vec![1.0, 2.0]; // Wrong length
        let invalid_tensor = TensorImpl::new(invalid_data, vec![2, 2], DType::F32);
        assert!(!invalid_tensor.is_valid());
    }

    #[test]
    fn test_module_impl() {
        let weight = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // 2x3 weight matrix
        let bias = Some(vec![0.1, 0.2]); // 2 bias terms

        let module = ModuleImpl::new_linear(3, 2, true, weight, bias);

        assert_eq!(module.module_type, "linear");
        assert_eq!(module.in_features, 3);
        assert_eq!(module.out_features, 2);
        assert!(module.bias);
        assert_eq!(module.weight_size(), 6);
        assert_eq!(module.bias_size(), Some(2));
        assert!(module.is_valid());

        // Test invalid module (wrong bias size)
        let invalid_bias = Some(vec![0.1]); // Wrong size
        let invalid_module = ModuleImpl::new_linear(3, 2, true, vec![0.0; 6], invalid_bias);
        assert!(!invalid_module.is_valid());
    }

    #[test]
    fn test_optimizer_impl() {
        // Test SGD optimizer
        let sgd = OptimizerImpl::new_sgd(0.01, 0.9);
        assert_eq!(sgd.optimizer_type(), "sgd");
        assert_eq!(sgd.learning_rate, 0.01);
        assert_eq!(sgd.momentum, Some(0.9));
        assert!(sgd.is_valid());

        // Test Adam optimizer
        let adam = OptimizerImpl::new_adam(0.001, 0.9, 0.999, 1e-8);
        assert_eq!(adam.optimizer_type(), "adam");
        assert_eq!(adam.learning_rate, 0.001);
        assert_eq!(adam.beta1, Some(0.9));
        assert_eq!(adam.beta2, Some(0.999));
        assert_eq!(adam.epsilon, Some(1e-8));
        assert!(adam.is_valid());

        // Test invalid optimizer (negative learning rate)
        let invalid_sgd = OptimizerImpl::new_sgd(-0.01, 0.9);
        assert!(!invalid_sgd.is_valid());
    }
}
