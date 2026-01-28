//! Core tensor operations and data structures
//!
//! This module provides the fundamental tensor operations including creation, data access,
//! gradient management, and basic tensor manipulation functionality.
//!
//! # Features
//!
//! - **Tensor creation**: Multiple constructors for different data sources
//! - **Memory management**: Copy-on-write semantics and reference cleanup
//! - **Data access**: Multi-dimensional and flat indexing
//! - **Gradient operations**: Full autograd integration
//! - **Device operations**: Device transfer and tensor detachment
//! - **Storage optimization**: Automatic memory mapping for large tensors

use std::path::PathBuf;
use std::sync::{Arc, RwLock, Weak};

use torsh_core::{
    device::DeviceType,
    dtype::{DType, FloatElement, TensorElement},
    error::{Result, TorshError},
    shape::Shape,
};

use crate::storage::TensorStorage;

// 🚀 Enhanced multi-backend GPU device management with SciRS2 integration
#[cfg(feature = "gpu")]
use crate::backend_integration::GpuBackendType;

/// Operation type for gradient computation
#[derive(Debug, Clone)]
pub enum Operation<T: TensorElement> {
    /// Leaf node (no operation)
    Leaf,
    /// Power operation: x^n
    Power {
        input: Arc<Tensor<T>>,
        exponent: f32,
    },
    /// Addition operation: a + b
    Add {
        lhs: Arc<Tensor<T>>,
        rhs: Arc<Tensor<T>>,
    },
    /// Multiplication operation: a * b
    Mul {
        lhs: Arc<Tensor<T>>,
        rhs: Arc<Tensor<T>>,
    },
    /// Custom operation with name and inputs
    Custom(String, Vec<Weak<Tensor<T>>>),
}

/// The main Tensor type for ToRSh
///
/// A tensor implementation with automatic memory mapping for large tensors
/// and efficient views with reference counting
#[derive(Clone)]
pub struct Tensor<T = f32>
where
    T: TensorElement,
{
    /// The data storage (automatically uses memory mapping for large tensors)
    pub(crate) storage: TensorStorage<T>,
    /// Shape of the tensor
    pub(crate) shape: Shape,
    /// Device information
    pub(crate) device: DeviceType,
    /// Whether gradients are required
    pub(crate) requires_grad: bool,
    /// Gradient tensor if computed
    pub(crate) grad: Arc<RwLock<Option<Tensor<T>>>>,
    /// Operation that created this tensor
    pub(crate) operation: Operation<T>,
    /// Custom strides for views (None means contiguous layout)
    pub(crate) strides: Option<Vec<usize>>,
    /// Offset into the storage for views (0 for base tensors)
    pub(crate) storage_offset: usize,
    /// Reference to base tensor for views (None for base tensors)
    pub(crate) base_tensor: Option<Weak<Tensor<T>>>,
}

impl<T: TensorElement + Copy> Tensor<T> {
    /// Clean up dead weak references in custom operations to improve memory efficiency
    pub fn cleanup_operation_refs(&mut self) {
        if let Operation::Custom(_, inputs) = &mut self.operation {
            inputs.retain(|weak_ref| weak_ref.strong_count() > 0);
        }
    }

    /// Create from raw data
    pub fn from_data(data: Vec<T>, shape: Vec<usize>, device: DeviceType) -> Result<Self> {
        let storage = TensorStorage::create_optimal(data)?;
        Ok(Self {
            storage,
            shape: Shape::new(shape),
            device,
            requires_grad: false,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: None,
            storage_offset: 0,
            base_tensor: None,
        })
    }

    /// 🚀 **Phase 7**: Create tensor with fast result storage (skips alignment copy)
    ///
    /// For SIMD operation results, uses simple InMemory storage to avoid
    /// the ~10µs overhead of copying data to aligned memory.
    ///
    /// # Performance
    /// - Skips AlignedVec copy (saves ~10µs for 50K elements)
    /// - Best for intermediate/result tensors
    /// - Input tensors should still use from_data for optimal SIMD input access
    pub fn from_data_fast(data: Vec<T>, shape: Vec<usize>, device: DeviceType) -> Self {
        let storage = TensorStorage::fast_result(data);
        Self {
            storage,
            shape: Shape::new(shape),
            device,
            requires_grad: false,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: None,
            storage_offset: 0,
            base_tensor: None,
        }
    }

    /// Create from raw data with explicit storage type
    pub fn from_data_with_storage(
        data: Vec<T>,
        shape: Vec<usize>,
        device: DeviceType,
        use_memory_mapping: bool,
    ) -> Result<Self> {
        let storage = if use_memory_mapping {
            TensorStorage::memory_mapped(data, None)?
        } else {
            TensorStorage::in_memory(data)
        };
        Ok(Self {
            storage,
            shape: Shape::new(shape),
            device,
            requires_grad: false,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: None,
            storage_offset: 0,
            base_tensor: None,
        })
    }

    /// Create from raw data with specified memory-mapped file path
    pub fn from_data_memory_mapped(
        data: Vec<T>,
        shape: Vec<usize>,
        device: DeviceType,
        file_path: PathBuf,
    ) -> Result<Self> {
        let storage = TensorStorage::memory_mapped(data, Some(file_path))?;
        Ok(Self {
            storage,
            shape: Shape::new(shape),
            device,
            requires_grad: false,
            grad: Arc::new(RwLock::new(None)),
            operation: Operation::Leaf,
            strides: None,
            storage_offset: 0,
            base_tensor: None,
        })
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: &[usize], device: DeviceType) -> Result<Self> {
        let numel = shape.iter().product();
        let data = vec![T::zero(); numel];
        Self::from_data(data, shape.to_vec(), device)
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: &[usize], device: DeviceType) -> Result<Self> {
        let numel = shape.iter().product();
        let data = vec![T::one(); numel];
        Self::from_data(data, shape.to_vec(), device)
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> Shape {
        self.shape.clone()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Get the data type
    pub fn dtype(&self) -> DType {
        T::dtype()
    }

    /// Convert tensor to a different data type
    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        // For now, just return a clone if the dtype is the same
        if T::dtype() == dtype {
            Ok(self.clone())
        } else {
            // TODO: Implement actual dtype conversion
            Err(TorshError::UnsupportedOperation {
                op: "dtype conversion".to_string(),
                dtype: format!("{:?} to {:?}", T::dtype(), dtype),
            })
        }
    }

    /// Get the device
    pub fn device(&self) -> DeviceType {
        self.device
    }

    /// Get element at multi-dimensional index
    pub fn get(&self, indices: &[usize]) -> Result<T>
    where
        T: Copy,
    {
        let flat_index = self.compute_flat_index(indices)?;
        self.storage.get(flat_index)
    }

    /// Get element at single flat index
    pub fn get_flat(&self, index: usize) -> Result<T>
    where
        T: Copy,
    {
        self.storage.get(index)
    }

    /// Set element at index (requires multi-dimensional indices for views)
    pub fn set(&self, indices: &[usize], value: T) -> Result<()>
    where
        T: Copy,
    {
        let flat_index = self.compute_flat_index(indices)?;
        self.storage.set(flat_index, value)
    }

    /// Get slice of elements
    pub fn get_slice(&self, start: usize, len: usize) -> Result<Vec<T>>
    where
        T: Copy,
    {
        self.storage.get_slice(start, len)
    }

    /// Set slice of elements
    pub fn set_slice(&self, start: usize, values: &[T]) -> Result<()>
    where
        T: Copy,
    {
        self.storage.set_slice(start, values)
    }

    /// Get all data as a vector (may be expensive for large memory-mapped tensors)
    /// For views, extracts only the data visible by this view
    pub fn to_vec(&self) -> Result<Vec<T>>
    where
        T: Copy,
    {
        if self.is_view() {
            // For views, we need to extract data according to strides and offsets
            let mut result = Vec::with_capacity(self.numel());
            let shape = self.shape.dims();
            let mut indices = vec![0; shape.len()];

            fn extract_recursive<T: TensorElement + Copy>(
                tensor: &Tensor<T>,
                indices: &mut [usize],
                dim: usize,
                result: &mut Vec<T>,
            ) -> Result<()> {
                let shape = tensor.shape.dims();
                if dim == shape.len() {
                    let flat_index = tensor.compute_flat_index(indices)?;
                    let value = tensor.storage.get(flat_index)?;
                    result.push(value);
                } else {
                    for i in 0..shape[dim] {
                        indices[dim] = i;
                        extract_recursive(tensor, indices, dim + 1, result)?;
                    }
                }
                Ok(())
            }

            extract_recursive(self, &mut indices, 0, &mut result)?;
            Ok(result)
        } else {
            self.storage.to_vec()
        }
    }

    /// Get storage type information
    pub fn storage_type(&self) -> &'static str {
        self.storage.storage_type()
    }

    /// Get estimated memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.storage.memory_usage()
    }

    /// Check if tensor uses memory mapping
    pub fn is_memory_mapped(&self) -> bool {
        matches!(self.storage, TensorStorage::MemoryMapped(_))
    }

    /// Check if this tensor is a view of another tensor
    pub fn is_view(&self) -> bool {
        self.base_tensor.is_some()
    }

    /// Get the strides for this tensor (either custom strides for views or default contiguous strides)
    pub fn strides(&self) -> Vec<usize> {
        if let Some(ref strides) = self.strides {
            strides.clone()
        } else {
            self.compute_default_strides()
        }
    }

    /// Compute default contiguous strides for the tensor's shape
    pub(crate) fn compute_default_strides(&self) -> Vec<usize> {
        let shape = self.shape.dims();
        let mut strides = vec![1; shape.len()];
        if shape.len() > 1 {
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        strides
    }

    /// Compute flat index with custom strides and offset for views
    fn compute_flat_index(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Expected {} indices, got {}",
                self.ndim(),
                indices.len()
            )));
        }

        // Validate indices
        let shape = self.shape.dims();
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= shape[i] {
                return Err(TorshError::IndexOutOfBounds {
                    index: idx,
                    size: shape[i],
                });
            }
        }

        // Compute flat index using strides
        let strides = self.strides();
        let flat_index = indices
            .iter()
            .zip(&strides)
            .map(|(idx, stride)| idx * stride)
            .sum::<usize>()
            + self.storage_offset;

        Ok(flat_index)
    }

    /// Execute a function with zero-copy access to tensor data (immutable)
    ///
    /// This enables SIMD operations without memory copies by providing direct
    /// access to the underlying buffer within a scoped context.
    ///
    /// # Arguments
    /// * `f` - Closure that receives `&[T]` and returns `Result<R>`
    ///
    /// # Returns
    /// Result from the closure
    ///
    /// # Performance
    /// - Zero memory copies for InMemory and Aligned storage
    /// - One allocation for MemoryMapped storage
    /// - Enables 2-4x SIMD speedup (per SciRS2 docs)
    ///
    /// # Examples
    /// ```ignore
    /// // Direct SIMD operation without copies
    /// let result = tensor.with_data_slice(|data| {
    ///     other_tensor.with_data_slice(|other_data| {
    ///         // Zero-copy SIMD addition
    ///         f32::simd_add(&data, &other_data)
    ///     })
    /// })?;
    /// ```
    pub fn with_data_slice<R, F>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&[T]) -> Result<R>,
        T: Copy,
    {
        self.storage.with_slice(f)
    }

    /// Execute a function with zero-copy access to tensor data (mutable)
    ///
    /// This enables in-place SIMD operations without memory copies.
    ///
    /// # Arguments
    /// * `f` - Closure that receives `&mut [T]` and returns `Result<R>`
    ///
    /// # Returns
    /// Result from the closure
    ///
    /// # Performance
    /// - Zero memory copies for InMemory storage
    /// - Not supported for MemoryMapped or Aligned storage (returns error)
    ///
    /// # Examples
    /// ```ignore
    /// // In-place SIMD operation without copies
    /// tensor.with_data_slice_mut(|data| {
    ///     other_tensor.with_data_slice(|other_data| {
    ///         // Zero-copy in-place SIMD addition
    ///         for (x, y) in data.iter_mut().zip(other_data.iter()) {
    ///             *x = *x + *y;
    ///         }
    ///         Ok(())
    ///     })
    /// })?;
    /// ```
    pub fn with_data_slice_mut<R, F>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&mut [T]) -> Result<R>,
        T: Copy,
    {
        self.storage.with_slice_mut(f)
    }

    /// Create a tensor of ones with the same shape as this tensor
    pub fn ones_like(&self) -> Result<Self> {
        let ones_data = vec![T::one(); self.numel()];
        Self::from_data(ones_data, self.shape().dims().to_vec(), self.device)
    }

    /// Create a tensor of zeros with the same shape as this tensor
    pub fn zeros_like(&self) -> Result<Self> {
        let zeros_data = vec![T::zero(); self.numel()];
        Self::from_data(zeros_data, self.shape().dims().to_vec(), self.device)
    }

    /// Set whether this tensor requires gradients
    pub fn requires_grad_(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    /// Get whether this tensor requires gradients
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set gradient tensor
    #[allow(dead_code)]
    pub fn set_grad(&self, grad: Option<Tensor<T>>) {
        let mut grad_lock = self.grad.write().expect("lock should not be poisoned");
        *grad_lock = grad;
    }

    /// Get mutable access to gradient
    pub fn grad_mut(&mut self) -> Option<&mut Self> {
        // For now, return None - would need to implement proper gradient access
        None
    }

    /// 🚀 Enhanced device transfer with multi-backend GPU support
    /// Automatically selects optimal transfer strategy and backend
    pub fn to<D: Into<DeviceType>>(self, device: D) -> Result<Self> {
        let target_device = device.into();
        if target_device == self.device {
            return Ok(self);
        }

        // Use SciRS2 backend integration for optimized device transfers
        self.to_device(target_device)
    }

    /// 🚀 Advanced multi-GPU distribution for parallel processing
    /// Automatically distributes tensor across multiple GPUs with optimal strategy
    /// Note: Actual implementation is in backend_integration.rs module
    #[cfg(feature = "gpu")]
    #[allow(dead_code)]
    pub fn distribute_multi_gpu_wrapper(&self, gpu_count: usize) -> Result<Vec<Self>> {
        // Forward to the implementation in backend_integration module
        // The distribute_multi_gpu method is implemented in backend_integration.rs
        // and should be available on self
        let _ = gpu_count;
        Err(TorshError::InvalidArgument(
            "Use the distribute_multi_gpu method from backend_integration module directly"
                .to_string(),
        ))
    }

    #[cfg(not(feature = "gpu"))]
    #[allow(dead_code)]
    pub fn distribute_multi_gpu_wrapper(&self, gpu_count: usize) -> Result<Vec<Self>> {
        // Attempted to distribute tensor across GPUs, but GPU feature is not enabled
        let _ = gpu_count; // Use parameter
        Err(TorshError::UnsupportedOperation {
            op: format!("multi-GPU distribution ({} GPUs)", gpu_count),
            dtype: "GPU feature not enabled".to_string(),
        })
    }

    /// 🚀 Get optimal GPU backend for current hardware
    #[cfg(feature = "gpu")]
    pub fn get_optimal_gpu_backend() -> Option<GpuBackendType> {
        // Try backends in order of preference
        let backends = [
            GpuBackendType::Cuda,
            GpuBackendType::Metal,
            GpuBackendType::Rocm,
            GpuBackendType::WebGpu,
            GpuBackendType::OpenCl,
        ];

        for backend in &backends {
            if Self::is_backend_available(backend.clone()) {
                return Some(backend.clone());
            }
        }
        None
    }

    /// Check if a specific GPU backend is available
    /// TODO: Temporarily disabled - backend types not yet available in scirs2_core
    #[cfg(feature = "gpu")]
    #[allow(dead_code)]
    fn is_backend_available(_backend: GpuBackendType) -> bool {
        // TODO: Implement when scirs2_core GPU backends are available
        // use scirs2_core::gpu::*;
        // match backend {
        //     GpuBackendType::Cuda => CudaBackend::is_available(),
        //     GpuBackendType::Metal => MetalBackend::is_available(),
        //     GpuBackendType::WebGpu => WebGpuBackend::is_available(),
        //     GpuBackendType::Rocm => RocmBackend::is_available(),
        //     GpuBackendType::OpenCl => OpenClBackend::is_available(),
        // }
        false
    }

    /// 🚀 Create tensor on optimal GPU device
    #[cfg(feature = "gpu")]
    pub fn zeros_gpu(shape: &[usize]) -> Result<Self> {
        let optimal_backend =
            Self::get_optimal_gpu_backend().ok_or_else(|| TorshError::UnsupportedOperation {
                op: "GPU tensor creation".to_string(),
                dtype: "No GPU backend available".to_string(),
            })?;

        let device = match optimal_backend {
            GpuBackendType::Cuda => DeviceType::Cuda(0),
            GpuBackendType::Metal => DeviceType::Metal(0),
            GpuBackendType::WebGpu => DeviceType::Wgpu(0),
            _ => {
                return Err(TorshError::UnsupportedOperation {
                    op: "GPU tensor creation".to_string(),
                    dtype: "Backend not supported for tensor creation".to_string(),
                })
            }
        };

        Self::zeros(shape, device)
    }

    /// 🚀 Create tensor on optimal GPU device filled with ones
    #[cfg(feature = "gpu")]
    pub fn ones_gpu(shape: &[usize]) -> Result<Self> {
        let optimal_backend =
            Self::get_optimal_gpu_backend().ok_or_else(|| TorshError::UnsupportedOperation {
                op: "GPU tensor creation".to_string(),
                dtype: "No GPU backend available".to_string(),
            })?;

        let device = match optimal_backend {
            GpuBackendType::Cuda => DeviceType::Cuda(0),
            GpuBackendType::Metal => DeviceType::Metal(0),
            GpuBackendType::WebGpu => DeviceType::Wgpu(0),
            _ => {
                return Err(TorshError::UnsupportedOperation {
                    op: "GPU tensor creation".to_string(),
                    dtype: "Backend not supported for tensor creation".to_string(),
                })
            }
        };

        Self::ones(shape, device)
    }

    /// Detach from the computation graph
    pub fn detach(&self) -> Self {
        let mut detached = self.clone();
        detached.requires_grad = false;
        detached
    }

    /// Get the gradient of this tensor (if it exists)
    pub fn grad(&self) -> Option<Self> {
        let grad_lock = self.grad.read().expect("lock should not be poisoned");
        grad_lock.as_ref().cloned()
    }

    /// Check if this tensor has a gradient
    pub fn has_grad(&self) -> bool {
        let grad_lock = self.grad.read().expect("lock should not be poisoned");
        grad_lock.is_some()
    }

    /// Zero the gradient
    pub fn zero_grad(&mut self) {
        let mut grad_lock = self.grad.write().expect("lock should not be poisoned");
        *grad_lock = None;
    }

    /// Backward pass (compute gradients) - integrated with autograd system
    pub fn backward(&self) -> Result<()>
    where
        T: FloatElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + Clone
            + std::fmt::Debug,
        f32: From<T>,
    {
        if !self.requires_grad {
            return Err(TorshError::AutogradError(
                "Called backward on tensor that doesn't require grad".to_string(),
            ));
        }

        if self.shape().numel() != 1 {
            return Err(TorshError::AutogradError(
                "Gradient can only be computed for scalar outputs".to_string(),
            ));
        }

        // Start backward computation with gradient of 1.0 for scalar output
        let grad_output = self.ones_like()?;
        self.backward_impl(&grad_output)?;
        Ok(())
    }

    /// Backward pass with gradient - integrated with autograd system
    pub fn backward_with_grad(&self, _gradient: Option<&Self>) -> Result<()>
    where
        T: FloatElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>
            + Clone
            + std::fmt::Debug,
        f32: From<T>,
    {
        if !self.requires_grad {
            return Err(TorshError::AutogradError(
                "Called backward on tensor that doesn't require grad".to_string(),
            ));
        }

        // TODO: Implement backward pass with gradient - currently autograd is handled at higher level
        // For now, return Ok since this will be handled by the autograd crate
        Ok(())
    }

    /// Internal backward implementation
    fn backward_impl(&self, grad_output: &Self) -> Result<()>
    where
        T: FloatElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
    {
        match &self.operation {
            Operation::Leaf => {
                // Accumulate gradient for leaf nodes
                let mut grad_lock = self.grad.write().expect("lock should not be poisoned");
                if let Some(existing_grad) = grad_lock.as_ref() {
                    // Add gradients if they exist
                    let new_grad = existing_grad.add_op(grad_output)?;
                    *grad_lock = Some(new_grad);
                } else {
                    // Set gradient if it doesn't exist
                    *grad_lock = Some(grad_output.clone());
                }
            }
            Operation::Power { input, exponent } => {
                if input.requires_grad {
                    // Compute gradient: d/dx(x^n) = n * x^(n-1)
                    let input_data = input.to_vec()?;
                    let grad_data: Vec<T> = input_data
                        .iter()
                        .map(|&x| {
                            let exp_minus_one = *exponent - 1.0;
                            let exp_t = T::from_f64(*exponent as f64)
                                .expect("f64 conversion should succeed");
                            let exp_minus_one_t = T::from_f64(exp_minus_one as f64)
                                .expect("f64 conversion should succeed");
                            exp_t * x.powf(exp_minus_one_t)
                        })
                        .collect();

                    let input_grad =
                        Self::from_data(grad_data, input.shape().dims().to_vec(), input.device)?;
                    let final_grad = input_grad.mul_op(grad_output)?;

                    // Recursively compute gradients
                    input.backward_impl(&final_grad)?;
                }
            }
            Operation::Add { lhs, rhs } => {
                // Gradient flows through both operands unchanged
                if lhs.requires_grad {
                    lhs.backward_impl(grad_output)?;
                }
                if rhs.requires_grad {
                    rhs.backward_impl(grad_output)?;
                }
            }
            Operation::Mul { lhs, rhs } => {
                // Product rule: d/dx(f*g) = f'*g + f*g'
                if lhs.requires_grad {
                    let lhs_grad = (**rhs).mul_op(grad_output)?;
                    lhs.backward_impl(&lhs_grad)?;
                }
                if rhs.requires_grad {
                    let rhs_grad = (**lhs).mul_op(grad_output)?;
                    rhs.backward_impl(&rhs_grad)?;
                }
            }
            Operation::Custom(op_name, inputs) => {
                // For custom operations, we need operation-specific gradient computation
                // For now, we'll just propagate the gradient to all inputs
                match op_name.as_str() {
                    "conv1d" | "conv2d" | "conv3d" => {
                        // Convolution backward pass would require specific implementation
                        // For now, we skip gradient computation for convolutions
                        // TODO: Implement proper convolution backward pass
                    }
                    _ => {
                        // For other custom operations, propagate gradient to all inputs
                        // Use weak references to prevent memory leaks from circular references
                        for weak_input in inputs {
                            if let Some(input) = weak_input.upgrade() {
                                if input.requires_grad {
                                    input.backward_impl(grad_output)?;
                                }
                            }
                            // Note: Dead weak references are automatically cleaned up
                            // when the Vec goes out of scope
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Comparison operations for tensors
impl<T: TensorElement + PartialOrd + Copy> Tensor<T> {
    /// Element-wise greater than comparison
    pub fn gt(&self, other: &Self) -> Result<Tensor<bool>> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }

        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a > b)
            .collect();

        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }

    /// Element-wise less than comparison
    pub fn lt(&self, other: &Self) -> Result<Tensor<bool>> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }

        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a < b)
            .collect();

        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }

    /// Element-wise greater than or equal comparison
    pub fn ge(&self, other: &Self) -> Result<Tensor<bool>> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }

        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a >= b)
            .collect();

        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }

    /// Element-wise less than or equal comparison
    pub fn le(&self, other: &Self) -> Result<Tensor<bool>> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }

        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a <= b)
            .collect();

        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }

    /// Element-wise equality comparison
    pub fn eq(&self, other: &Self) -> Result<Tensor<bool>> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }

        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a == b)
            .collect();

        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }

    /// Element-wise inequality comparison
    pub fn ne(&self, other: &Self) -> Result<Tensor<bool>> {
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }

        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;
        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a != b)
            .collect();

        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }

    /// Scalar comparison methods
    /// Element-wise equality comparison with scalar
    pub fn eq_scalar(&self, value: T) -> Result<Tensor<bool>>
    where
        T: PartialEq + Copy,
    {
        let self_data = self.to_vec()?;
        let result_data: Vec<bool> = self_data.iter().map(|&a| a == value).collect();

        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }

    /// Element-wise inequality comparison with scalar
    pub fn ne_scalar(&self, value: T) -> Result<Tensor<bool>>
    where
        T: PartialEq + Copy,
    {
        let self_data = self.to_vec()?;
        let result_data: Vec<bool> = self_data.iter().map(|&a| a != value).collect();

        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }

    /// Element-wise greater than comparison with scalar
    pub fn gt_scalar(&self, value: T) -> Result<Tensor<bool>>
    where
        T: PartialOrd + Copy,
    {
        let self_data = self.to_vec()?;
        let result_data: Vec<bool> = self_data.iter().map(|&a| a > value).collect();

        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }

    /// Element-wise less than comparison with scalar
    pub fn lt_scalar(&self, value: T) -> Result<Tensor<bool>>
    where
        T: PartialOrd + Copy,
    {
        let self_data = self.to_vec()?;
        let result_data: Vec<bool> = self_data.iter().map(|&a| a < value).collect();

        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }

    /// Element-wise less than or equal comparison with scalar
    pub fn le_scalar(&self, value: T) -> Result<Tensor<bool>>
    where
        T: PartialOrd + Copy,
    {
        let self_data = self.to_vec()?;
        let result_data: Vec<bool> = self_data.iter().map(|&a| a <= value).collect();

        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }

    /// Element-wise greater than or equal comparison with scalar
    pub fn ge_scalar(&self, value: T) -> Result<Tensor<bool>>
    where
        T: PartialOrd + Copy,
    {
        let self_data = self.to_vec()?;
        let result_data: Vec<bool> = self_data.iter().map(|&a| a >= value).collect();

        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
}

/// Shape manipulation operations for tensors
impl<T: TensorElement> Tensor<T> {
    /// Flatten tensor to 1D
    pub fn flatten(&self) -> Result<Self> {
        let total_elements = self.numel();
        self.view(&[total_elements as i32])
    }

    // broadcast_to has been moved to ops/manipulation.rs with proper implementation

    /// Conditional tensor selection - where condition is true, select from self, otherwise from other
    pub fn where_tensor(&self, condition: &Tensor<bool>, other: &Self) -> Result<Self> {
        // Verify all tensors have the same shape
        if self.shape != condition.shape || self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: condition.shape.dims().to_vec(),
            });
        }

        let self_data = self.to_vec()?;
        let condition_data = condition.to_vec()?;
        let other_data = other.to_vec()?;

        let result_data: Vec<T> = self_data
            .iter()
            .zip(condition_data.iter())
            .zip(other_data.iter())
            .map(
                |((&self_val, &cond), &other_val)| {
                    if cond {
                        self_val
                    } else {
                        other_val
                    }
                },
            )
            .collect();

        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }

    /// Add bias vector to tensor (element-wise addition)
    pub fn add_bias(&self, bias: &Self) -> Result<Self>
    where
        T: std::ops::Add<Output = T>,
    {
        self.add(bias)
    }
}

/// Logical operations for boolean tensors
impl Tensor<bool> {
    /// Element-wise logical AND operation
    pub fn logical_and(&self, other: &Self) -> Result<Self> {
        // Verify shapes match
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }

        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;

        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a && b)
            .collect();

        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }

    /// Element-wise logical OR operation
    pub fn logical_or(&self, other: &Self) -> Result<Self> {
        // Verify shapes match
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }

        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;

        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a || b)
            .collect();

        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }

    /// Element-wise logical XOR operation
    pub fn logical_xor(&self, other: &Self) -> Result<Self> {
        // Verify shapes match
        if self.shape != other.shape {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }

        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;

        let result_data: Vec<bool> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a ^ b)
            .collect();

        Tensor::from_data(result_data, self.shape.dims().to_vec(), self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_data(data, vec![2, 2], DeviceType::Cpu).unwrap();

        assert_eq!(tensor.shape().dims(), &[2, 2]);
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.device(), DeviceType::Cpu);
    }

    #[test]
    fn test_zeros_and_ones() {
        let zeros = Tensor::<f32>::zeros(&[3, 3], DeviceType::Cpu).unwrap();
        assert_eq!(zeros.numel(), 9);
        assert_eq!(zeros.get(&[0, 0]).unwrap(), 0.0);

        let ones = Tensor::<f32>::ones(&[2, 3], DeviceType::Cpu).unwrap();
        assert_eq!(ones.numel(), 6);
        assert_eq!(ones.get(&[1, 2]).unwrap(), 1.0);
    }

    #[test]
    fn test_tensor_indexing() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_data(data, vec![2, 3], DeviceType::Cpu).unwrap();

        assert_eq!(tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(tensor.get(&[0, 2]).unwrap(), 3.0);
        assert_eq!(tensor.get(&[1, 1]).unwrap(), 5.0);
    }

    #[test]
    fn test_tensor_properties() {
        let data = vec![1.0f32; 100];
        let tensor = Tensor::from_data(data, vec![10, 10], DeviceType::Cpu).unwrap();

        assert!(!tensor.is_view());
        assert!(!tensor.is_memory_mapped());
        assert!(!tensor.requires_grad());

        let with_grad = tensor.requires_grad_(true);
        assert!(with_grad.requires_grad());
    }

    #[test]
    fn test_ones_like_zeros_like() {
        let original =
            Tensor::<f32>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], DeviceType::Cpu)
                .unwrap();

        let zeros_like = original.zeros_like().unwrap();
        assert_eq!(zeros_like.shape().dims(), &[2, 2]);
        assert_eq!(zeros_like.get(&[0, 0]).unwrap(), 0.0);

        let ones_like = original.ones_like().unwrap();
        assert_eq!(ones_like.shape().dims(), &[2, 2]);
        assert_eq!(ones_like.get(&[1, 1]).unwrap(), 1.0);
    }
}
