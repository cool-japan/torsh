//! Element-wise arithmetic operations for tensors
//! 🚀 Enhanced with SciRS2 breakthrough hyperoptimized SIMD implementations
//! - Up to 14.17x speedup for medium arrays with TLB optimization
//! - 7.93x speedup for small arrays with cache-line aware processing
//! - 7.41x speedup for large arrays with software pipelining
//! - Adaptive selection automatically chooses optimal strategy

use crate::{FloatElement, Tensor, TensorElement};
use torsh_core::error::{Result, TorshError};
use super::simd::{should_use_simd, SimdOpType};

/// Element-wise arithmetic operations
impl<
        T: TensorElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
    > Tensor<T>
{
    /// Element-wise addition with broadcasting (ops module implementation)
    ///
    /// # SIMD Optimization (Phase 3/4)
    /// For f32 tensors with matching shapes, uses adaptive SIMD dispatch:
    /// - Small tensors (<512): Scalar (SIMD overhead not worth it)
    /// - Medium tensors (512-65K): Phase 3 SIMD (uninit buffer + scirs2 API)
    /// - Large tensors (>65K): Parallel SIMD (Rayon + SIMD chunks)
    pub fn add_op(&self, other: &Self) -> Result<Self> {
        // GPU dispatch: CUDA-device-only, f32-only, large tensors only.
        // Falls through to the CPU path if context init or kernel execution fails.
        #[cfg(feature = "gpu")]
        {
            use torsh_core::device::DeviceType;
            if matches!(self.device, DeviceType::Cuda(_))
                && self.shape() == other.shape()
                && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
                && self.shape().numel() >= 65536
            {
                static GPU_CTX: std::sync::OnceLock<
                    Option<scirs2_core::gpu::GpuContext>,
                > = std::sync::OnceLock::new();
                let ctx_opt = GPU_CTX.get_or_init(|| {
                    scirs2_core::gpu::GpuContext::new(scirs2_core::gpu::GpuBackend::Cuda).ok()
                });
                if let Some(ctx) = ctx_opt {
                    let kernel =
                        scirs2_core::gpu::kernels::elementwise::ElementwiseAddKernel::new();
                    use scirs2_core::gpu::kernels::GpuKernel as _;
                    let source = kernel
                        .source_for_backend(scirs2_core::gpu::GpuBackend::Cuda)
                        .unwrap_or_default();
                    let self_data = self.data()?;
                    let other_data = other.data()?;
                    // Transmute slices to f32 — safe because T == f32 is verified via TypeId above.
                    let self_f32: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            self_data.as_ptr() as *const f32,
                            self_data.len(),
                        )
                    };
                    let other_f32: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            other_data.as_ptr() as *const f32,
                            other_data.len(),
                        )
                    };
                    let n = self_data.len() as u32;
                    let buf_a = ctx.create_buffer_from_slice(self_f32);
                    let buf_b = ctx.create_buffer_from_slice(other_f32);
                    let buf_out = ctx.create_buffer::<f32>(self_data.len());
                    let workgroups = ((n + 255) / 256, 1, 1);
                    if ctx
                        .execute_kernel(
                            &source,
                            &[buf_a, buf_b, buf_out.clone()],
                            workgroups,
                            &[n],
                            &[],
                        )
                        .is_ok()
                    {
                        let result_f32 = ctx
                            .read_buffer(&buf_out)
                            .map_err(|e| TorshError::Other(e.to_string()))?;
                        // SAFETY: T == f32, so Vec<f32> has the same bit pattern as Vec<T>.
                        let result_t: Vec<T> = unsafe { std::mem::transmute(result_f32) };
                        let mut out = Self::from_data(
                            result_t,
                            self.shape().dims().to_vec(),
                            self.device.clone(),
                        )?;
                        if self.requires_grad || other.requires_grad {
                            use std::sync::Arc;
                            out.requires_grad = true;
                            out.operation = crate::Operation::Add {
                                lhs: Arc::new(self.clone()),
                                rhs: Arc::new(other.clone()),
                            };
                        }
                        return Ok(out);
                    }
                }
            }
        }

        let mut result = if self.shape() == other.shape() {
            // 🚀 Phase 3/4: Use adaptive SIMD for f32 tensors
            #[cfg(feature = "simd")]
            {
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                    // Use Phase 4 adaptive dispatch for f32
                    return {
                        let mut result = self.add_adaptive(other)?;
                        // Track the operation for gradient computation
                        if self.requires_grad || other.requires_grad {
                            use std::sync::Arc;
                            result.requires_grad = true;
                            result.operation = crate::Operation::Add {
                                lhs: Arc::new(self.clone()),
                                rhs: Arc::new(other.clone()),
                            };
                        }
                        Ok(result)
                    };
                }
            }
            // Fallback to scalar for non-f32 types
            self.element_wise_op(other, |a, b| a + b)?
        } else {
            self.broadcast_binary_op(other, |a, b| a + b)?
        };

        // Track the operation for gradient computation
        if self.requires_grad || other.requires_grad {
            use std::sync::Arc;
            result.requires_grad = true;
            result.operation = crate::Operation::Add {
                lhs: Arc::new(self.clone()),
                rhs: Arc::new(other.clone()),
            };
        }

        Ok(result)
    }

    /// Performs element-wise addition of two tensors with broadcasting support.
    ///
    /// Adds corresponding elements from `self` and `other`. If the tensors have
    /// different shapes, broadcasting rules are applied to make them compatible.
    ///
    /// # Broadcasting Rules
    /// - Dimensions are aligned from right to left
    /// - Dimension of size 1 can broadcast to any size
    /// - Missing dimensions are treated as size 1
    ///
    /// Examples of valid broadcasts:
    /// - `[3, 4]` + `[3, 4]` → `[3, 4]` (same shape)
    /// - `[3, 4]` + `[4]` → `[3, 4]` (broadcast last dimension)
    /// - `[3, 1]` + `[1, 4]` → `[3, 4]` (broadcast both)
    /// - `[3, 4, 5]` + `[5]` → `[3, 4, 5]` (broadcast to batch)
    ///
    /// # Performance
    /// For matching shapes with f32 type, automatically uses adaptive SIMD optimization:
    /// - Small tensors (<512 elements): Scalar operations
    /// - Medium tensors (512-65K): SIMD vectorization (up to 14x speedup)
    /// - Large tensors (>65K): Parallel SIMD (multi-threaded)
    ///
    /// # Gradient Tracking
    /// If either tensor has `requires_grad=true`, the operation is recorded
    /// in the computational graph for automatic differentiation.
    ///
    /// # Arguments
    /// * `other` - The tensor to add to `self`
    ///
    /// # Returns
    /// A new tensor containing the element-wise sum
    ///
    /// # Errors
    /// Returns error if the shapes are not compatible for broadcasting
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use torsh::Tensor;
    /// # use torsh_core::device::DeviceType;
    /// // Same shape addition
    /// let a = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu)?;
    /// let b = Tensor::from_data(vec![4.0, 5.0, 6.0], vec![3], DeviceType::Cpu)?;
    /// let c = a.add(&b)?;
    /// assert_eq!(c.data()?, vec![5.0, 7.0, 9.0]);
    ///
    /// // Broadcasting: [2,3] + [3] → [2,3]
    /// let a = Tensor::from_data(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     vec![2, 3],
    ///     DeviceType::Cpu
    /// )?;
    /// let b = Tensor::from_data(vec![10.0, 20.0, 30.0], vec![3], DeviceType::Cpu)?;
    /// let c = a.add(&b)?;  // Adds [10,20,30] to each row
    /// assert_eq!(c.data()?, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    ///
    /// // Neural network bias addition
    /// let activations = Tensor::randn(&[32, 128], DeviceType::Cpu)?;  // Batch output
    /// let bias = Tensor::randn(&[128], DeviceType::Cpu)?;             // Bias vector
    /// let output = activations.add(&bias)?;  // Broadcasts bias to all samples
    ///
    /// // Matrix addition for residual connections
    /// let x = Tensor::randn(&[64, 64], DeviceType::Cpu)?;
    /// let residual = Tensor::randn(&[64, 64], DeviceType::Cpu)?;
    /// let output = x.add(&residual)?;  // Element-wise sum
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.add(a, b)` or `a + b`
    ///
    /// See also: [`Self::add_scalar`], [`Self::add_`], [`Self::sub`], [`Self::mul`]
    pub fn add(&self, other: &Self) -> Result<Self> {
        self.add_op(other)
    }

    /// Element-wise subtraction with broadcasting
    ///
    /// # SIMD Optimization (Phase 3/4)
    /// For f32 tensors with matching shapes, uses adaptive SIMD dispatch:
    /// - Small tensors (<512): Scalar (SIMD overhead not worth it)
    /// - Medium tensors (512-65K): Phase 7 direct SIMD
    /// - Large tensors (>65K): Parallel SIMD
    pub fn sub(&self, other: &Self) -> Result<Self> {
        let mut result = if self.shape() == other.shape() {
            // 🚀 Phase 3/4: Use adaptive SIMD for f32 tensors
            #[cfg(feature = "simd")]
            {
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                    // Use Phase 4 adaptive dispatch for f32
                    return {
                        let mut result = self.sub_adaptive(other)?;
                        // Track the operation for gradient computation
                        if self.requires_grad || other.requires_grad {
                            use std::sync::Arc;
                            result.requires_grad = true;
                            result.operation = crate::Operation::Sub {
                                lhs: Arc::new(self.clone()),
                                rhs: Arc::new(other.clone()),
                            };
                        }
                        Ok(result)
                    };
                }
            }
            // Fallback to scalar for non-f32 types
            self.element_wise_op(other, |a, b| a - b)?
        } else {
            self.broadcast_binary_op(other, |a, b| a - b)?
        };

        // Track the operation for gradient computation
        if self.requires_grad || other.requires_grad {
            use std::sync::Arc;
            result.requires_grad = true;
            result.operation = crate::Operation::Sub {
                lhs: Arc::new(self.clone()),
                rhs: Arc::new(other.clone()),
            };
        }

        Ok(result)
    }

    /// Element-wise multiplication with broadcasting (ops module implementation)
    ///
    /// # SIMD Optimization (Phase 3/4)
    /// For f32 tensors with matching shapes, uses adaptive SIMD dispatch:
    /// - Small tensors (<512): Scalar (SIMD overhead not worth it)
    /// - Medium tensors (512-65K): Phase 3 SIMD (uninit buffer + scirs2 API)
    /// - Large tensors (>65K): Parallel SIMD (Rayon + SIMD chunks)
    pub fn mul_op(&self, other: &Self) -> Result<Self> {
        // GPU dispatch: CUDA-device-only, f32-only, large tensors only.
        // Falls through to the CPU path if context init or kernel execution fails.
        #[cfg(feature = "gpu")]
        {
            use torsh_core::device::DeviceType;
            if matches!(self.device, DeviceType::Cuda(_))
                && self.shape() == other.shape()
                && std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
                && self.shape().numel() >= 65536
            {
                static GPU_CTX: std::sync::OnceLock<
                    Option<scirs2_core::gpu::GpuContext>,
                > = std::sync::OnceLock::new();
                let ctx_opt = GPU_CTX.get_or_init(|| {
                    scirs2_core::gpu::GpuContext::new(scirs2_core::gpu::GpuBackend::Cuda).ok()
                });
                if let Some(ctx) = ctx_opt {
                    let kernel =
                        scirs2_core::gpu::kernels::elementwise::ElementwiseMulKernel::new();
                    use scirs2_core::gpu::kernels::GpuKernel as _;
                    let source = kernel
                        .source_for_backend(scirs2_core::gpu::GpuBackend::Cuda)
                        .unwrap_or_default();
                    let self_data = self.data()?;
                    let other_data = other.data()?;
                    // Transmute slices to f32 — safe because T == f32 is verified via TypeId above.
                    let self_f32: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            self_data.as_ptr() as *const f32,
                            self_data.len(),
                        )
                    };
                    let other_f32: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            other_data.as_ptr() as *const f32,
                            other_data.len(),
                        )
                    };
                    let n = self_data.len() as u32;
                    let buf_a = ctx.create_buffer_from_slice(self_f32);
                    let buf_b = ctx.create_buffer_from_slice(other_f32);
                    let buf_out = ctx.create_buffer::<f32>(self_data.len());
                    let workgroups = ((n + 255) / 256, 1, 1);
                    if ctx
                        .execute_kernel(
                            &source,
                            &[buf_a, buf_b, buf_out.clone()],
                            workgroups,
                            &[n],
                            &[],
                        )
                        .is_ok()
                    {
                        let result_f32 = ctx
                            .read_buffer(&buf_out)
                            .map_err(|e| TorshError::Other(e.to_string()))?;
                        // SAFETY: T == f32, so Vec<f32> has the same bit pattern as Vec<T>.
                        let result_t: Vec<T> = unsafe { std::mem::transmute(result_f32) };
                        let mut out = Self::from_data(
                            result_t,
                            self.shape().dims().to_vec(),
                            self.device,
                        )?;
                        if self.requires_grad || other.requires_grad {
                            use std::sync::Arc;
                            out.requires_grad = true;
                            out.operation = crate::Operation::Mul {
                                lhs: Arc::new(self.clone()),
                                rhs: Arc::new(other.clone()),
                            };
                        }
                        return Ok(out);
                    }
                }
            }
        }

        let mut result = if self.shape() == other.shape() {
            // 🚀 Phase 3/4: Use adaptive SIMD for f32 tensors
            #[cfg(feature = "simd")]
            {
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                    // Use Phase 4 adaptive dispatch for f32
                    return {
                        let mut result = self.mul_adaptive(other)?;
                        // Track the operation for gradient computation
                        if self.requires_grad || other.requires_grad {
                            use std::sync::Arc;
                            result.requires_grad = true;
                            result.operation = crate::Operation::Mul {
                                lhs: Arc::new(self.clone()),
                                rhs: Arc::new(other.clone()),
                            };
                        }
                        Ok(result)
                    };
                }
            }
            // Fallback to scalar for non-f32 types
            self.element_wise_op(other, |a, b| a * b)?
        } else {
            self.broadcast_binary_op(other, |a, b| a * b)?
        };

        // Track the operation for gradient computation
        if self.requires_grad || other.requires_grad {
            use std::sync::Arc;
            result.requires_grad = true;
            result.operation = crate::Operation::Mul {
                lhs: Arc::new(self.clone()),
                rhs: Arc::new(other.clone()),
            };
        }

        Ok(result)
    }

    /// Performs element-wise multiplication of two tensors with broadcasting support.
    ///
    /// Multiplies corresponding elements from `self` and `other`. If the tensors have
    /// different shapes, broadcasting rules are applied to make them compatible.
    ///
    /// # Broadcasting Rules
    /// - Dimensions are aligned from right to left
    /// - Dimension of size 1 can broadcast to any size
    /// - Missing dimensions are treated as size 1
    ///
    /// Examples of valid broadcasts:
    /// - `[3, 4]` * `[3, 4]` → `[3, 4]` (same shape)
    /// - `[3, 4]` * `[4]` → `[3, 4]` (broadcast last dimension)
    /// - `[3, 1]` * `[1, 4]` → `[3, 4]` (broadcast both)
    /// - `[3, 4, 5]` * `[5]` → `[3, 4, 5]` (broadcast to batch)
    ///
    /// # Performance
    /// For matching shapes with f32 type, automatically uses adaptive SIMD optimization:
    /// - Small tensors (<512 elements): Scalar operations
    /// - Medium tensors (512-65K): SIMD vectorization (up to 14x speedup)
    /// - Large tensors (>65K): Parallel SIMD (multi-threaded)
    ///
    /// # Gradient Tracking
    /// If either tensor has `requires_grad=true`, the operation is recorded
    /// in the computational graph for automatic differentiation.
    ///
    /// # Arguments
    /// * `other` - The tensor to multiply with `self`
    ///
    /// # Returns
    /// A new tensor containing the element-wise product
    ///
    /// # Errors
    /// Returns error if the shapes are not compatible for broadcasting
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use torsh::Tensor;
    /// # use torsh_core::device::DeviceType;
    /// // Same shape multiplication
    /// let a = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu)?;
    /// let b = Tensor::from_data(vec![4.0, 5.0, 6.0], vec![3], DeviceType::Cpu)?;
    /// let c = a.mul(&b)?;
    /// assert_eq!(c.data()?, vec![4.0, 10.0, 18.0]);
    ///
    /// // Broadcasting: [2,3] * [3] → [2,3]
    /// let a = Tensor::from_data(
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ///     vec![2, 3],
    ///     DeviceType::Cpu
    /// )?;
    /// let b = Tensor::from_data(vec![10.0, 20.0, 30.0], vec![3], DeviceType::Cpu)?;
    /// let c = a.mul(&b)?;  // Multiplies each row by [10,20,30]
    /// assert_eq!(c.data()?, vec![10.0, 40.0, 90.0, 40.0, 100.0, 180.0]);
    ///
    /// // Apply attention mask (element-wise gating)
    /// let features = Tensor::randn(&[32, 128], DeviceType::Cpu)?;
    /// let mask = Tensor::ones(&[32, 128], DeviceType::Cpu)?;  // Binary mask
    /// let masked_features = features.mul(&mask)?;  // Zero out masked positions
    ///
    /// // Feature scaling
    /// let x = Tensor::randn(&[64, 256], DeviceType::Cpu)?;
    /// let scale = Tensor::ones(&[256], DeviceType::Cpu)?;  // Learnable scale
    /// let scaled_x = x.mul(&scale)?;  // Scale each feature
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.mul(a, b)` or `a * b`
    ///
    /// Note: This is element-wise multiplication, not matrix multiplication.
    /// For matrix multiplication, use [`Self::matmul`].
    ///
    /// See also: [`Self::mul_scalar`], [`Self::mul_`], [`Self::matmul`], [`Self::div`]
    pub fn mul(&self, other: &Self) -> Result<Self> {
        self.mul_op(other)
    }

    /// Performs element-wise division of two tensors with broadcasting support.
    ///
    /// Divides corresponding elements of `self` by `other`. If the tensors have
    /// different shapes, broadcasting rules are applied to make them compatible.
    ///
    /// # Broadcasting Rules
    /// - Dimensions are aligned from right to left
    /// - Dimension of size 1 can broadcast to any size
    /// - Missing dimensions are treated as size 1
    ///
    /// Examples of valid broadcasts:
    /// - `[3, 4]` / `[3, 4]` → `[3, 4]` (same shape)
    /// - `[3, 4]` / `[4]` → `[3, 4]` (broadcast last dimension)
    /// - `[3, 1]` / `[1, 4]` → `[3, 4]` (broadcast both)
    /// - `[3, 4, 5]` / `[5]` → `[3, 4, 5]` (broadcast to batch)
    ///
    /// # Performance
    /// For matching shapes with f32 type, automatically uses adaptive SIMD optimization:
    /// - Small tensors (<512 elements): Scalar operations
    /// - Medium tensors (512-65K): SIMD vectorization
    /// - Large tensors (>65K): Parallel SIMD (multi-threaded)
    ///
    /// # Division by Zero
    /// Division by zero produces infinity (inf) or NaN according to IEEE 754 rules:
    /// - Positive number / 0.0 → inf
    /// - Negative number / 0.0 → -inf
    /// - 0.0 / 0.0 → NaN
    ///
    /// # Arguments
    /// * `other` - The tensor to divide by (denominator)
    ///
    /// # Returns
    /// A new tensor containing the element-wise quotient
    ///
    /// # Errors
    /// Returns error if the shapes are not compatible for broadcasting
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use torsh::Tensor;
    /// # use torsh_core::device::DeviceType;
    /// // Same shape division
    /// let a = Tensor::from_data(vec![10.0, 20.0, 30.0], vec![3], DeviceType::Cpu)?;
    /// let b = Tensor::from_data(vec![2.0, 4.0, 5.0], vec![3], DeviceType::Cpu)?;
    /// let c = a.div(&b)?;
    /// assert_eq!(c.data()?, vec![5.0, 5.0, 6.0]);
    ///
    /// // Broadcasting: [2,3] / [3] → [2,3]
    /// let a = Tensor::from_data(
    ///     vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    ///     vec![2, 3],
    ///     DeviceType::Cpu
    /// )?;
    /// let b = Tensor::from_data(vec![2.0, 5.0, 10.0], vec![3], DeviceType::Cpu)?;
    /// let c = a.div(&b)?;  // Divides each row by [2,5,10]
    /// assert_eq!(c.data()?, vec![5.0, 4.0, 3.0, 20.0, 10.0, 6.0]);
    ///
    /// // Normalize features (divide by standard deviation)
    /// let x = Tensor::randn(&[32, 128], DeviceType::Cpu)?;
    /// let std = Tensor::ones(&[128], DeviceType::Cpu)?;  // Feature std deviations
    /// let normalized = x.div(&std)?;  // Normalize each feature
    ///
    /// // Compute probabilities from logits
    /// let logits = Tensor::randn(&[64, 10], DeviceType::Cpu)?;
    /// let sum = logits.sum_dim(1, true)?;  // Sum over classes
    /// let probs = logits.div(&sum)?;  // Normalize to probabilities
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.div(a, b)` or `a / b`
    ///
    /// See also: [`Self::div_scalar`], [`Self::div_`], [`Self::mul`], [`Self::reciprocal`]
    pub fn div(&self, other: &Self) -> Result<Self> {
        if self.shape() == other.shape() {
            // 🚀 Phase 3/4: Use adaptive SIMD for f32 tensors
            #[cfg(feature = "simd")]
            {
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                    return self.div_adaptive(other);
                }
            }
            // Fallback to scalar for non-f32 types
            self.element_wise_op(other, |a, b| a / b)
        } else {
            self.broadcast_binary_op(other, |a, b| a / b)
        }
    }

    /// Generic broadcasting binary operation with comprehensive error handling
    pub fn broadcast_binary_op<F>(&self, other: &Self, op: F) -> Result<Self>
    where
        F: Fn(T, T) -> T + Send + Sync,
        T: Copy + Default + 'static,
    {
        use crate::broadcast::{BroadcastOps, BroadcastShape};

        // Validate the broadcast operation
        BroadcastOps::validate_broadcast_operation(
            self.shape().dims(),
            other.shape().dims(),
            "binary operation",
        )?;

        // If shapes are identical, use optimized path
        if self.shape() == other.shape() {
            return self.element_wise_op(other, op);
        }

        // Check if broadcasting is memory efficient
        if !self.shape().is_broadcast_efficient(&other.shape()) {
            eprintln!(
                "Warning: Broadcasting shapes {:?} and {:?} may use significant memory",
                self.shape().dims(),
                other.shape().dims()
            );
        }

        // Compute broadcasted shape using the new implementation
        let broadcast_shape = self.shape().broadcast_shape(&other.shape())?;
        let broadcast_dims = broadcast_shape.dims();
        let broadcast_size = broadcast_shape.numel();

        // Estimate memory requirements
        let element_size = std::mem::size_of::<T>();
        let _memory_required = BroadcastOps::estimate_broadcast_memory(
            self.shape().dims(),
            other.shape().dims(),
            element_size,
        )?;

        let self_data = self.data()?;
        let other_data = other.data()?;

        // Use pooled buffer to avoid repeated heap allocation for broadcast results.
        // T: 'static is satisfied by the outer where-clause on broadcast_binary_op.
        use crate::memory_pool::global_acquire_uninit;
        let mut buf = global_acquire_uninit::<T>(broadcast_size);
        {
            let uninit = buf.as_uninit_slice_mut();

            // Compute broadcasting for each element using optimized indexing
            for flat_idx in 0..broadcast_size {
                let broadcast_indices = BroadcastOps::flat_to_multi_index(flat_idx, broadcast_dims);

                let self_idx = BroadcastOps::compute_broadcast_index(
                    &broadcast_indices,
                    self.shape().dims(),
                    broadcast_dims,
                )?;
                let other_idx = BroadcastOps::compute_broadcast_index(
                    &broadcast_indices,
                    other.shape().dims(),
                    broadcast_dims,
                )?;

                let self_val = self_data[self_idx];
                let other_val = other_data[other_idx];

                // SAFETY: flat_idx < broadcast_size == buf.capacity(); element written exactly once.
                unsafe { uninit[flat_idx].write(op(self_val, other_val)) };
            }
        }
        let result_data = buf.into_vec(broadcast_size);

        Self::from_data(
            result_data,
            broadcast_dims.to_vec(),
            self.device,
        )
    }

    /// Element-wise power with scalar exponent
    pub fn pow_scalar_f32(&self, exponent: f32) -> Result<Self>
    where
        T: FloatElement,
    {
        let data = self.data()?;
        let result_data: Vec<T> = data
            .iter()
            .map(|&x| {
                let exp_val = T::from_f64(exponent as f64).ok_or_else(|| {
                    TorshError::ConversionError(format!(
                        "Cannot convert {exponent} to element type"
                    ))
                })?;
                Ok(x.powf(exp_val))
            })
            .collect::<Result<Vec<T>>>()?;

        let mut result = Self::from_data(result_data, self.shape().dims().to_vec(), self.device)?;
        result.requires_grad = self.requires_grad;

        // Track the operation for gradient computation
        if self.requires_grad {
            use std::sync::Arc;
            result.operation = crate::Operation::Power {
                input: Arc::new(self.clone()),
                exponent,
            };
        }

        Ok(result)
    }

    /// Element-wise power with tensor exponent
    pub fn pow_tensor(&self, exponent: &Self) -> Result<Self>
    where
        T: FloatElement,
    {
        // Check shapes are compatible for broadcasting
        if self.shape() != exponent.shape() {
            return Err(TorshError::BroadcastError {
                shape1: self.shape().dims().to_vec(),
                shape2: exponent.shape().dims().to_vec(),
            });
        }

        let self_data = self.data()?;
        let exp_data = exponent.data()?;

        let result_data: Vec<T> = self_data
            .iter()
            .zip(exp_data.iter())
            .map(|(&base, &exp)| base.powf(exp))
            .collect();

        let mut result = Self::from_data(result_data, self.shape().dims().to_vec(), self.device)?;
        result.requires_grad = self.requires_grad || exponent.requires_grad;

        Ok(result)
    }

    /// Negation (for float types) - legacy arithmetic implementation
    pub fn neg_float(&self) -> Result<Self>
    where
        T: FloatElement,
    {
        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| -x).collect();

        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Add scalar
    pub fn add_scalar(&self, scalar: T) -> Result<Self> {
        let self_data = self.data()?;
        let result_data: Vec<T> = self_data.iter().map(|&a| a + scalar).collect();

        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Divide by scalar
    pub fn div_scalar(&self, scalar: f32) -> Result<Self> {
        let self_data = self.data()?;

        let scalar_t = T::from_f64(scalar as f64).ok_or_else(|| {
            TorshError::ConversionError(format!(
                "Cannot convert scalar {} to target type",
                scalar
            ))
        })?;

        let result_data: Vec<T> = self_data
            .iter()
            .map(|&a| a / scalar_t)
            .collect();

        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Power by scalar
    pub fn pow_scalar(&self, exponent: f32) -> Result<Self>
    where
        T: FloatElement,
    {
        let data = self.data()?;
        let exp_t = T::from_f64(exponent as f64).ok_or_else(|| {
            TorshError::ConversionError(format!(
                "Cannot convert exponent {} to target type",
                exponent
            ))
        })?;

        let result_data: Vec<T> = data
            .iter()
            .map(|&x| x.powf(exp_t))
            .collect();

        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Clamp values between min and max (f32 version)
    pub fn clamp_f32(&self, min: f32, max: f32) -> Result<Self>
    where
        T: PartialOrd,
    {
        let min_t = T::from_f64(min as f64).ok_or_else(|| {
            TorshError::ConversionError(format!(
                "Cannot convert min value {} to target type",
                min
            ))
        })?;
        let max_t = T::from_f64(max as f64).ok_or_else(|| {
            TorshError::ConversionError(format!(
                "Cannot convert max value {} to target type",
                max
            ))
        })?;

        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&item| {
            if item < min_t {
                min_t
            } else if item > max_t {
                max_t
            } else {
                item
            }
        }).collect();

        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Computes the dot product (inner product) of two 1D tensors.
    ///
    /// For two vectors `a` and `b` of length `n`, computes the sum of element-wise products:
    /// `dot(a, b) = a[0]*b[0] + a[1]*b[1] + ... + a[n-1]*b[n-1]`
    ///
    /// This is a scalar-valued operation that measures the projection of one vector onto another.
    ///
    /// # Requirements
    /// - Both tensors must be 1-dimensional
    /// - Both tensors must have the same length
    ///
    /// # Performance
    /// Uses breakthrough hyperoptimized SIMD implementation with adaptive selection:
    /// - Small vectors (<512 elements): Standard scalar loop
    /// - Medium vectors (512-65K): Cache-optimized SIMD (7-14x speedup)
    /// - Large vectors (>65K): Parallel SIMD with software pipelining
    ///
    /// Performance characteristics:
    /// - Up to 14.17x speedup for medium arrays with TLB optimization
    /// - 7.93x speedup for small arrays with cache-line aware processing
    /// - 7.41x speedup for large arrays with software pipelining
    /// - Automatically selects optimal strategy based on vector size
    ///
    /// # Arguments
    /// * `other` - The second 1D tensor to compute dot product with
    ///
    /// # Returns
    /// A scalar value (type `T`) representing the dot product
    ///
    /// # Errors
    /// - Returns error if either tensor is not 1-dimensional
    /// - Returns error if the vectors have different lengths
    ///
    /// # Examples
    /// ```rust,no_run
    /// # use torsh::Tensor;
    /// # use torsh_core::device::DeviceType;
    /// // Basic dot product
    /// let a = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu)?;
    /// let b = Tensor::from_data(vec![4.0, 5.0, 6.0], vec![3], DeviceType::Cpu)?;
    /// let dot = a.dot_hyperoptimized(&b)?;
    /// assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 32
    ///
    /// // Cosine similarity computation
    /// let v1 = Tensor::randn(&[128], DeviceType::Cpu)?;
    /// let v2 = Tensor::randn(&[128], DeviceType::Cpu)?;
    /// let dot = v1.dot_hyperoptimized(&v2)?;
    /// let norm1 = v1.dot_hyperoptimized(&v1)?.sqrt();
    /// let norm2 = v2.dot_hyperoptimized(&v2)?.sqrt();
    /// let cosine_sim = dot / (norm1 * norm2);
    ///
    /// // Neural network: compute attention scores
    /// let query = Tensor::randn(&[512], DeviceType::Cpu)?;
    /// let key = Tensor::randn(&[512], DeviceType::Cpu)?;
    /// let attention_score = query.dot_hyperoptimized(&key)?;
    ///
    /// // Compute vector norm (L2 norm)
    /// let v = Tensor::from_data(vec![3.0, 4.0], vec![2], DeviceType::Cpu)?;
    /// let norm = v.dot_hyperoptimized(&v)?.sqrt();
    /// assert_eq!(norm, 5.0); // sqrt(3^2 + 4^2) = 5
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # PyTorch Compatibility
    /// Equivalent to `torch.dot(a, b)`
    ///
    /// Note: For matrix-vector or matrix-matrix products, use [`Self::matmul`].
    ///
    /// See also: [`Self::matmul`], [`Self::mul`], [`Self::outer`], [`Self::cross`]
    pub fn dot_hyperoptimized(&self, other: &Self) -> Result<T>
    where
        T: FloatElement + Copy + std::iter::Sum,
    {
        // Ensure both tensors are 1D and have the same shape
        if self.shape().dims().len() != 1 || other.shape().dims().len() != 1 {
            return Err(TorshError::InvalidArgument(
                "Dot product requires 1D tensors".to_string()
            ));
        }

        if self.shape() != other.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().to_vec(),
                got: other.shape().to_vec(),
            });
        }

        let self_data = self.data()?;
        let other_data = other.data()?;

        // Use hyperoptimized SIMD dot product for f32 tensors
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            #[cfg(feature = "simd")]
            {
                if should_use_simd(self.numel()) {
                    use scirs2_core::ndarray::ArrayView1;
                    use super::simd::simd_dot_f32;

                    // Cast to f32 for SIMD operations
                    let self_f32: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            self_data.as_ptr() as *const f32,
                            self_data.len(),
                        )
                    };
                    let other_f32: &[f32] = unsafe {
                        std::slice::from_raw_parts(
                            other_data.as_ptr() as *const f32,
                            other_data.len(),
                        )
                    };

                    let self_view = ArrayView1::from(self_f32);
                    let other_view = ArrayView1::from(other_f32);

                    // Use adaptive hyperoptimized SIMD dot product
                    let result_f32 = simd_dot_f32(&self_view, &other_view);
                    let result: T = unsafe { std::mem::transmute_copy::<f32, T>(&result_f32) };
                    return Ok(result);
                }
            }
        }

        // Fallback to standard dot product for non-f32 types or when SIMD is not beneficial
        let result: T = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a * b)
            .sum();

        Ok(result)
    }
}

// ✅ In-place operations for PyTorch compatibility
//
// NOTE: This module (`ops/arithmetic.rs`) is currently unreachable because
// `pub mod ops;` is disabled in `lib.rs` (it conflicts with the live API in
// `math_ops.rs`). The real `add_` / `sub_` / `mul_` / `div_` family lives in
// `math_ops.rs` — see that module for the implementation actually compiled
// into the crate. The bodies below are kept in lock-step with the live ones
// (single-lock slice mutation via `with_slice_mut`, broadcast-into-self
// support, never allocates a new buffer) so re-enabling `pub mod ops;` later
// will not require a rewrite of this file.
impl<
        T: TensorElement
            + Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
    > Tensor<T>
{
    /// Element-wise in-place addition: `self += other` (with broadcast of `other` into `self`)
    pub fn add_(&mut self, other: &Self) -> Result<&mut Self> {
        self.inplace_binary_op(other, "add_", |a, b| a + b)
    }

    /// Element-wise in-place subtraction: `self -= other` (with broadcast of `other` into `self`)
    pub fn sub_(&mut self, other: &Self) -> Result<&mut Self> {
        self.inplace_binary_op(other, "sub_", |a, b| a - b)
    }

    /// Element-wise in-place multiplication: `self *= other` (with broadcast of `other` into `self`)
    pub fn mul_(&mut self, other: &Self) -> Result<&mut Self> {
        self.inplace_binary_op(other, "mul_", |a, b| a * b)
    }

    /// Element-wise in-place division: `self /= other` (with broadcast of `other` into `self`)
    pub fn div_(&mut self, other: &Self) -> Result<&mut Self> {
        self.inplace_binary_op(other, "div_", |a, b| a / b)
    }

    /// Shared driver for `add_` / `sub_` / `mul_` / `div_`.
    ///
    /// Holds the storage write lock for the entire pass (one acquisition) and
    /// never allocates a new data buffer — broadcasting is done by reading
    /// `other` through `BroadcastOps::compute_broadcast_index` while writing
    /// straight into `self`'s slice.
    fn inplace_binary_op<F>(&mut self, other: &Self, op_name: &str, op: F) -> Result<&mut Self>
    where
        F: Fn(T, T) -> T,
    {
        if self.requires_grad {
            return Err(TorshError::InvalidArgument(format!(
                "In-place operation `{op_name}` on tensor that requires grad is not allowed"
            )));
        }

        let self_dims_vec = self.shape().dims().to_vec();
        let other_dims_vec = other.shape().dims().to_vec();
        let other_data = other.data()?;

        // Fast path: identical shapes — direct elementwise pass over the slice.
        if self_dims_vec == other_dims_vec {
            self.storage.with_slice_mut(|slice| {
                debug_assert_eq!(slice.len(), other_data.len());
                for (dst, src) in slice.iter_mut().zip(other_data.iter()) {
                    *dst = op(*dst, *src);
                }
                Ok(())
            })?;
            return Ok(self);
        }

        // Broadcast path: must be broadcast-compatible AND `self`'s shape must
        // already equal the broadcast shape (we cannot grow `self` in place).
        use crate::broadcast::BroadcastOps;
        let broadcast_shape =
            BroadcastOps::compute_broadcast_shape(&self_dims_vec, &other_dims_vec)?;
        if broadcast_shape != self_dims_vec {
            return Err(TorshError::ShapeMismatch {
                expected: self_dims_vec,
                got: broadcast_shape,
            });
        }

        // `other` broadcasts into `self`'s shape — walk every output index,
        // pull the (possibly repeated) value from `other`, write into `self`.
        let total = self_dims_vec.iter().product::<usize>();
        let self_dims = self_dims_vec.as_slice();
        let other_dims = other_dims_vec.as_slice();
        self.storage.with_slice_mut(|slice| {
            for flat_idx in 0..total {
                let multi_index = BroadcastOps::flat_to_multi_index(flat_idx, self_dims);
                let other_idx =
                    BroadcastOps::compute_broadcast_index(&multi_index, other_dims, self_dims)?;
                slice[flat_idx] = op(slice[flat_idx], other_data[other_idx]);
            }
            Ok(())
        })?;

        Ok(self)
    }

    /// In-place scalar addition: `self += scalar`
    pub fn add_scalar_(&mut self, scalar: T) -> Result<&mut Self> {
        self.inplace_scalar_op(scalar, "add_scalar_", |a, b| a + b)
    }

    /// In-place scalar subtraction: `self -= scalar`
    pub fn sub_scalar_(&mut self, scalar: T) -> Result<&mut Self> {
        self.inplace_scalar_op(scalar, "sub_scalar_", |a, b| a - b)
    }

    /// In-place scalar multiplication: `self *= scalar`
    pub fn mul_scalar_(&mut self, scalar: T) -> Result<&mut Self> {
        self.inplace_scalar_op(scalar, "mul_scalar_", |a, b| a * b)
    }

    /// In-place scalar division: `self /= scalar`
    pub fn div_scalar_(&mut self, scalar: T) -> Result<&mut Self> {
        self.inplace_scalar_op(scalar, "div_scalar_", |a, b| a / b)
    }

    /// Shared driver for scalar in-place ops; one lock acquisition for the whole pass.
    fn inplace_scalar_op<F>(&mut self, scalar: T, op_name: &str, op: F) -> Result<&mut Self>
    where
        F: Fn(T, T) -> T,
    {
        if self.requires_grad {
            return Err(TorshError::InvalidArgument(format!(
                "In-place operation `{op_name}` on tensor that requires grad is not allowed"
            )));
        }

        self.storage.with_slice_mut(|slice| {
            for dst in slice.iter_mut() {
                *dst = op(*dst, scalar);
            }
            Ok(())
        })?;

        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_add_inplace() {
        let mut a = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");
        let b = Tensor::from_data(vec![4.0f32, 5.0, 6.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");

        a.add_(&b).expect("add_ failed");
        let result = a.data().expect("data retrieval failed");

        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_mul_inplace() {
        let mut a = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");
        let b = Tensor::from_data(vec![2.0f32, 3.0, 4.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");

        a.mul_(&b).expect("mul_ failed");
        let result = a.data().expect("data retrieval failed");

        assert_eq!(result, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_sub_inplace() {
        let mut a = Tensor::from_data(vec![5.0f32, 7.0, 9.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");
        let b = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");

        a.sub_(&b).expect("sub_ failed");
        let result = a.data().expect("data retrieval failed");

        assert_eq!(result, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_div_inplace() {
        let mut a = Tensor::from_data(vec![6.0f32, 12.0, 18.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");
        let b = Tensor::from_data(vec![2.0f32, 3.0, 6.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");

        a.div_(&b).expect("div_ failed");
        let result = a.data().expect("data retrieval failed");

        assert_eq!(result, vec![3.0, 4.0, 3.0]);
    }

    #[test]
    fn test_add_scalar_inplace() {
        let mut tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");

        tensor.add_scalar_(10.0).expect("add_scalar_ failed");
        let result = tensor.data().expect("data retrieval failed");

        assert_eq!(result, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_mul_scalar_inplace() {
        let mut tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");

        tensor.mul_scalar_(2.0).expect("mul_scalar_ failed");
        let result = tensor.data().expect("data retrieval failed");

        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_div_scalar_inplace() {
        let mut tensor = Tensor::from_data(vec![10.0f32, 20.0, 30.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");

        tensor.div_scalar_(10.0).expect("div_scalar_ failed");
        let result = tensor.data().expect("data retrieval failed");

        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_inplace_method_chaining() {
        let mut tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");
        let b = Tensor::from_data(vec![1.0f32, 1.0, 1.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");

        // Test method chaining
        tensor.add_(&b).expect("add_ failed").mul_scalar_(2.0).expect("mul_scalar_ failed");
        let result = tensor.data().expect("data retrieval failed");

        assert_eq!(result, vec![4.0, 6.0, 8.0]); // (1+1)*2, (2+1)*2, (3+1)*2
    }

    #[test]
    fn test_inplace_shape_mismatch_error() {
        let mut a = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).expect("tensor creation failed");
        let b = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");

        assert!(a.add_(&b).is_err());
        assert!(a.mul_(&b).is_err());
    }

    #[test]
    fn test_sub_scalar_inplace() {
        let mut tensor =
            Tensor::from_data(vec![10.0f32, 20.0, 30.0], vec![3], DeviceType::Cpu)
                .expect("tensor creation failed");

        tensor
            .sub_scalar_(5.0)
            .expect("sub_scalar_ failed");
        let result = tensor.data().expect("data retrieval failed");

        assert_eq!(result, vec![5.0, 15.0, 25.0]);
    }

    #[test]
    fn test_add_inplace_broadcast_row_into_matrix() {
        // Broadcast a [3] vector into a [2, 3] matrix in place.
        // After: each row of `mat` should be its original row + [10, 20, 30].
        let mut mat = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .expect("tensor creation failed");
        let row = Tensor::from_data(vec![10.0f32, 20.0, 30.0], vec![3], DeviceType::Cpu)
            .expect("tensor creation failed");

        mat.add_(&row).expect("add_ broadcast failed");
        let result = mat.data().expect("data retrieval failed");
        assert_eq!(result, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_mul_inplace_broadcast_column_into_matrix() {
        // Broadcast a [2, 1] column into a [2, 3] matrix in place via mul_.
        let mut mat = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .expect("tensor creation failed");
        let col = Tensor::from_data(vec![10.0f32, 100.0], vec![2, 1], DeviceType::Cpu)
            .expect("tensor creation failed");

        mat.mul_(&col).expect("mul_ broadcast failed");
        let result = mat.data().expect("data retrieval failed");
        // Row 0 scaled by 10, row 1 scaled by 100.
        assert_eq!(result, vec![10.0, 20.0, 30.0, 400.0, 500.0, 600.0]);
    }

    #[test]
    fn test_inplace_broadcast_self_would_grow_errors() {
        // self=[3], other=[2, 3] — broadcast shape is [2, 3], which would force
        // `self` to grow. The in-place op must refuse instead of allocating.
        let mut small = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)
            .expect("tensor creation failed");
        let big = Tensor::from_data(
            vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .expect("tensor creation failed");

        assert!(small.add_(&big).is_err());
        assert!(small.sub_(&big).is_err());
    }

    // --- Regression tests for issue #43: sub must propagate requires_grad ---

    #[test]
    fn test_issue_43_sub_propagates_requires_grad() {
        // Both operands have requires_grad=true; result must too.
        let a = Tensor::from_data(vec![3.0f32, 4.0], vec![2], DeviceType::Cpu)
            .expect("tensor creation failed")
            .requires_grad_(true);
        let b = Tensor::from_data(vec![1.0f32, 1.0], vec![2], DeviceType::Cpu)
            .expect("tensor creation failed")
            .requires_grad_(true);

        let result = a.sub(&b).expect("sub failed");
        assert!(
            result.requires_grad(),
            "sub result must have requires_grad=true when either operand does"
        );
    }

    #[test]
    fn test_issue_43_sub_propagates_requires_grad_one_sided() {
        // Only one operand has requires_grad; result must still have it.
        let a = Tensor::from_data(vec![3.0f32, 4.0], vec![2], DeviceType::Cpu)
            .expect("tensor creation failed")
            .requires_grad_(true);
        let b = Tensor::from_data(vec![1.0f32, 1.0], vec![2], DeviceType::Cpu)
            .expect("tensor creation failed");

        let result = a.sub(&b).expect("sub failed");
        assert!(
            result.requires_grad(),
            "sub result must have requires_grad=true when lhs has it"
        );
    }

    #[test]
    fn test_issue_43_sub_backward() {
        // Use scalar tensors (numel=1) so backward() can be called directly.
        // d/dlhs (lhs - rhs) = +1;  d/drhs (lhs - rhs) = -1
        let lhs = Tensor::from_data(vec![5.0f32], vec![1], DeviceType::Cpu)
            .expect("tensor creation failed")
            .requires_grad_(true);
        let rhs = Tensor::from_data(vec![3.0f32], vec![1], DeviceType::Cpu)
            .expect("tensor creation failed")
            .requires_grad_(true);

        let result = lhs.sub(&rhs).expect("sub failed");
        assert!(result.requires_grad(), "sub result must track gradients");

        result.backward().expect("backward failed");

        let lhs_grad = lhs.grad().expect("lhs must have gradient after backward");
        let rhs_grad = rhs.grad().expect("rhs must have gradient after backward");

        let lhs_grad_data = lhs_grad.data().expect("lhs grad data");
        let rhs_grad_data = rhs_grad.data().expect("rhs grad data");

        // d(lhs - rhs)/dlhs = +1
        assert_eq!(lhs_grad_data, vec![1.0f32], "lhs grad should be +1");
        // d(lhs - rhs)/drhs = -1
        assert_eq!(rhs_grad_data, vec![-1.0f32], "rhs grad should be -1");
    }
}