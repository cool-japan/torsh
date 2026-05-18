//! Activation functions for tensors
//!
//! This module provides comprehensive activation functions commonly used in neural networks:
//! - ReLU and variants (ReLU, Leaky ReLU)
//! - Sigmoid and hyperbolic tangent (Sigmoid, Tanh)
//! - Advanced activations (GELU)
//! - Softmax functions (Softmax, Log Softmax)
//! - In-place versions for memory efficiency

use crate::{FloatElement, Tensor, TensorElement};
use torsh_core::error::{Result, TorshError};

/// Try to execute a unary activation on the GPU (f32 + CUDA only).
///
/// Returns `Some(result)` when the GPU dispatch succeeds end-to-end,
/// or `None` if the tensor is not an f32 CUDA tensor, the GPU context
/// cannot be initialised, the kernel call fails, or the resulting
/// tensor cannot be reconstructed.  Callers fall through to the
/// existing CPU implementation in that case.
///
/// The closure receives the live `GpuContext` and the input `GpuBuffer<f32>`
/// and returns the produced output buffer.
#[cfg(feature = "gpu")]
fn try_gpu_unary_f32<T, F>(input: &Tensor<T>, gpu_fn: F) -> Option<Tensor<T>>
where
    T: FloatElement + 'static,
    F: FnOnce(
        &scirs2_core::gpu::GpuContext,
        &scirs2_core::gpu::GpuBuffer<f32>,
    ) -> std::result::Result<
        scirs2_core::gpu::GpuBuffer<f32>,
        scirs2_core::gpu::GpuError,
    >,
{
    use scirs2_core::gpu::{GpuBackend, GpuContext};
    use std::any::TypeId;
    use std::sync::OnceLock;

    // Bail out unless T == f32 and the tensor lives on a CUDA device.
    if TypeId::of::<T>() != TypeId::of::<f32>() {
        return None;
    }
    if !matches!(input.device, crate::DeviceType::Cuda(_)) {
        return None;
    }

    static GPU_CTX: OnceLock<Option<GpuContext>> = OnceLock::new();
    let ctx = GPU_CTX
        .get_or_init(|| GpuContext::new(GpuBackend::Cuda).ok())
        .as_ref()?;

    let data = input.data().ok()?;
    // SAFETY: TypeId guard above guarantees T == f32, so &[T] is &[f32].
    let f32_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), data.len())
    };
    let input_buf = ctx.create_buffer_from_slice(f32_slice);
    let output_buf = gpu_fn(ctx, &input_buf).ok()?;
    let result_f32: Vec<f32> = output_buf.to_vec();
    // SAFETY: T == f32 (confirmed by TypeId), so Vec<f32> == Vec<T>.
    let result_t: Vec<T> = unsafe {
        let mut v = std::mem::ManuallyDrop::new(result_f32);
        Vec::from_raw_parts(v.as_mut_ptr().cast::<T>(), v.len(), v.capacity())
    };
    Tensor::<T>::from_data(result_t, input.shape().dims().to_vec(), input.device).ok()
}

/// Try to execute a parameterised unary activation on the GPU using the
/// generic `execute_kernel` dispatch path (f32 + CUDA only).
///
/// Unlike [`try_gpu_unary_f32`] (which uses pre-baked `GpuContext::*` methods
/// such as `gelu` / `relu`), this helper drives kernels that take auxiliary
/// scalar parameters (e.g. `negative_slope` for LeakyReLU) and that have no
/// dedicated convenience method on `GpuContext`.  It mirrors the architectural
/// pattern used by `ElementwiseAddKernel` in `ops/arithmetic.rs`.
///
/// Returns `Some(result)` when the GPU dispatch succeeds end-to-end, otherwise
/// `None` so the caller can fall back to the CPU path.
///
/// * `input`            — input tensor (must be f32 and on a CUDA device).
/// * `min_numel`        — threshold below which CPU is preferred.
/// * `source_for_backend` — closure producing the kernel source string for the
///                          selected backend.  Receives the GPU context's
///                          backend so the caller can pull the proper variant
///                          (CUDA / WGPU / Metal / OpenCL) from the kernel.
/// * `float_params`     — scalar parameters fed into the kernel as `f32`.
#[cfg(feature = "gpu")]
fn try_gpu_kernel_unary_f32<T, S>(
    input: &Tensor<T>,
    min_numel: usize,
    source_for_backend: S,
    float_params: &[f32],
) -> Option<Tensor<T>>
where
    T: FloatElement + 'static,
    S: FnOnce(scirs2_core::gpu::GpuBackend) -> Option<String>,
{
    use scirs2_core::gpu::{GpuBackend, GpuContext};
    use std::any::TypeId;
    use std::sync::OnceLock;

    if TypeId::of::<T>() != TypeId::of::<f32>() {
        return None;
    }
    if !matches!(input.device, crate::DeviceType::Cuda(_)) {
        return None;
    }
    let numel = input.shape().numel();
    if numel < min_numel {
        return None;
    }

    static GPU_CTX: OnceLock<Option<GpuContext>> = OnceLock::new();
    let ctx = GPU_CTX
        .get_or_init(|| GpuContext::new(GpuBackend::Cuda).ok())
        .as_ref()?;

    let source = source_for_backend(ctx.backend())?;
    let data = input.data().ok()?;
    // SAFETY: TypeId guard above guarantees T == f32, so &[T] is &[f32].
    let f32_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), data.len())
    };
    let input_buf = ctx.create_buffer_from_slice(f32_slice);
    let output_buf = ctx.create_buffer::<f32>(data.len());
    let n = data.len() as u32;
    let workgroups = (n.div_ceil(256), 1, 1);

    ctx.execute_kernel(
        &source,
        &[input_buf, output_buf.clone()],
        workgroups,
        &[n],
        float_params,
    )
    .ok()?;

    let result_f32: Vec<f32> = output_buf.to_vec();
    // SAFETY: T == f32 (confirmed by TypeId), so Vec<f32> == Vec<T>.
    let result_t: Vec<T> = unsafe {
        let mut v = std::mem::ManuallyDrop::new(result_f32);
        Vec::from_raw_parts(v.as_mut_ptr().cast::<T>(), v.len(), v.capacity())
    };
    Tensor::<T>::from_data(result_t, input.shape().dims().to_vec(), input.device).ok()
}

/// Activation functions for float tensors
impl<T: FloatElement> Tensor<T> {
    /// ReLU activation: f(x) = max(0, x)
    ///
    /// When compiled with `feature = "gpu"` and the tensor is an f32 CUDA
    /// tensor, computation is dispatched to `scirs2_core::gpu::GpuContext::relu`
    /// (which routes to the GPU when CUDA hardware is present and falls
    /// back to an optimised CPU implementation otherwise).  Any failure
    /// silently falls through to the standard CPU path below.
    pub fn relu(&self) -> Result<Self> {
        #[cfg(feature = "gpu")]
        if let Some(result) = try_gpu_unary_f32(self, |ctx, buf| ctx.relu(buf)) {
            return Ok(result);
        }

        let data = self.data()?;
        let result_data: Vec<T> = data
            .iter()
            .map(|&x| {
                let zero = <T as TensorElement>::zero();
                if x > zero {
                    x
                } else {
                    zero
                }
            })
            .collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// In-place ReLU activation
    pub fn relu_(&mut self) -> Result<()>
    where
        T: PartialOrd,
    {
        let zero = <T as TensorElement>::zero();

        self.data_mut_apply(|item| {
            if *item < zero {
                *item = zero;
            }
        })?;

        Ok(())
    }

    /// Leaky ReLU activation: f(x) = max(negative_slope * x, x)
    ///
    /// When compiled with `feature = "gpu"` and the tensor is an f32 CUDA
    /// tensor with at least 65 536 elements, computation is dispatched to
    /// `scirs2_core::gpu::kernels::ml::activation::LeakyReluKernel` via the
    /// generic `execute_kernel` path (the architectural pattern shared with
    /// `ElementwiseAddKernel` in `ops/arithmetic.rs`).  Any GPU failure
    /// silently falls through to the standard CPU path below.
    pub fn leaky_relu(&self, negative_slope: f64) -> Result<Self> {
        // ── GPU fast path (f32 on CUDA, large tensors only) ──────────────────
        #[cfg(feature = "gpu")]
        {
            use scirs2_core::gpu::kernels::{ml::activation::LeakyReluKernel, GpuKernel};
            let slope_f32 = negative_slope as f32;
            let kernel = LeakyReluKernel::new(slope_f32);
            if let Some(result) = try_gpu_kernel_unary_f32(
                self,
                65_536,
                |backend| kernel.source_for_backend(backend).ok(),
                &[slope_f32],
            ) {
                return Ok(result);
            }
        }

        // ── CPU path (generic fallback) ──────────────────────────────────────
        let data = self.data()?;
        let slope = T::from_f64(negative_slope).unwrap_or_else(|| <T as TensorElement>::zero());
        let zero = <T as TensorElement>::zero();
        let result_data: Vec<T> = data
            .iter()
            .map(|&x| if x > zero { x } else { x * slope })
            .collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Sigmoid activation: f(x) = 1 / (1 + exp(-x))
    ///
    /// When compiled with `feature = "gpu"` and the tensor is an f32 CUDA
    /// tensor, computation is dispatched to `scirs2_core::gpu::GpuContext::sigmoid`
    /// (which routes to the GPU when CUDA hardware is present and falls
    /// back to an optimised CPU implementation otherwise).  Any failure
    /// silently falls through to the standard CPU path below.
    pub fn sigmoid(&self) -> Result<Self> {
        #[cfg(feature = "gpu")]
        if let Some(result) = try_gpu_unary_f32(self, |ctx, buf| ctx.sigmoid(buf)) {
            return Ok(result);
        }

        let data = self.data()?;
        let result_data: Vec<T> = data
            .iter()
            .map(|&x| {
                let one = <T as TensorElement>::one();
                one / (one + (-x).exp())
            })
            .collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// In-place sigmoid activation
    pub fn sigmoid_(&mut self) -> Result<()> {
        let one = <T as TensorElement>::one();

        self.data_mut_apply(|item| {
            *item = one / (one + (-*item).exp());
        })?;

        Ok(())
    }

    /// Hyperbolic tangent activation: f(x) = tanh(x)
    ///
    /// When compiled with `feature = "gpu"` and the tensor is an f32 CUDA
    /// tensor, computation is dispatched to `scirs2_core::gpu::GpuContext::tanh`
    /// (which routes to the GPU when CUDA hardware is present and falls
    /// back to an optimised CPU implementation otherwise).  Any failure
    /// silently falls through to the standard CPU path below.
    pub fn tanh(&self) -> Result<Self> {
        #[cfg(feature = "gpu")]
        if let Some(result) = try_gpu_unary_f32(self, |ctx, buf| ctx.tanh(buf)) {
            return Ok(result);
        }

        let data = self.data()?;
        let result_data: Vec<T> = data.iter().map(|&x| x.tanh()).collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// In-place hyperbolic tangent activation
    pub fn tanh_(&mut self) -> Result<()> {
        self.data_mut_apply(|item| {
            *item = item.tanh();
        })?;

        Ok(())
    }

    /// GELU (Gaussian Error Linear Unit) activation
    /// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    ///
    /// When compiled with `feature = "gpu"` and the tensor is on a CUDA device
    /// with element type f32, computation is dispatched to `scirs2_core::gpu::GpuContext::gelu`
    /// which routes to the GPU when CUDA hardware is present and falls back to an
    /// optimised CPU implementation otherwise.  Any GPU initialisation failure
    /// silently falls through to the standard CPU path.
    pub fn gelu(&self) -> Result<Self> {
        // ── GPU fast path (f32 on CUDA only) ──────────────────────────────────
        #[cfg(feature = "gpu")]
        if let Some(result) = try_gpu_unary_f32(self, |ctx, buf| ctx.gelu(buf)) {
            return Ok(result);
        }

        // ── CPU path (generic fallback) ───────────────────────────────────────
        let data = self.data()?;
        // GELU(x) = x * Φ(x) where Φ(x) is the CDF of standard normal distribution
        // Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        let half = T::from_f64(0.5).unwrap_or_else(|| <T as TensorElement>::zero());
        let one = <T as TensorElement>::one();
        let c1 = T::from_f64(0.7978845608).unwrap_or_else(|| <T as TensorElement>::zero()); // sqrt(2/π)
        let c2 = T::from_f64(0.044715).unwrap_or_else(|| <T as TensorElement>::zero());

        let result_data: Vec<T> = data
            .iter()
            .map(|&x| {
                let x3 = x * x * x;
                let inner = c1 * (x + c2 * x3);
                half * x * (one + inner.tanh())
            })
            .collect();
        Self::from_data(
            result_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Softmax activation along specified dimension
    /// Computes softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    /// Uses numerical stability techniques (subtracting max)
    pub fn softmax(&self, dim: i32) -> Result<Self> {
        // For now, implement along the last dimension if dim == -1, otherwise use specified dim
        let data = self.data()?;
        let shape_binding = self.shape();
        let shape = shape_binding.dims();

        if shape.is_empty() {
            return Err(TorshError::InvalidOperation("Cannot compute softmax on empty tensor".to_string()));
        }

        // Handle negative dimension
        let actual_dim = if dim < 0 {
            (shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };

        if actual_dim >= shape.len() {
            return Err(TorshError::InvalidArgument(format!("Dimension {} out of range for tensor with {} dimensions", dim, shape.len())));
        }

        let dim_size = shape[actual_dim];
        let outer_size: usize = shape[..actual_dim].iter().product();
        let inner_size: usize = shape[actual_dim + 1..].iter().product();

        let mut result_data = vec![T::from_f64(0.0).expect("f64 conversion should succeed"); data.len()];

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base_idx = outer * dim_size * inner_size + inner;

                // Find max for numerical stability
                let mut max_val = data[base_idx];
                for d in 1..dim_size {
                    let idx = base_idx + d * inner_size;
                    if data[idx] > max_val {
                        max_val = data[idx];
                    }
                }

                // Compute exp(x - max) and sum
                let mut exp_sum = T::from_f64(0.0).expect("f64 conversion should succeed");
                let mut exp_values = vec![T::from_f64(0.0).expect("f64 conversion should succeed"); dim_size];

                for d in 0..dim_size {
                    let idx = base_idx + d * inner_size;
                    let exp_val = (data[idx] - max_val).exp();
                    exp_values[d] = exp_val;
                    exp_sum = exp_sum + exp_val;
                }

                // Compute softmax values
                for d in 0..dim_size {
                    let idx = base_idx + d * inner_size;
                    result_data[idx] = exp_values[d] / exp_sum;
                }
            }
        }

        Self::from_data(
            result_data,
            shape.to_vec(),
            self.device,
        )
    }

    /// Log softmax along dimension (numerically stable)
    /// Computes log(softmax(x_i)) = x_i - log(sum(exp(x_j))) for all j
    pub fn log_softmax(&self, dim: i32) -> Result<Self> {
        let data = self.data()?;
        let shape_binding = self.shape();
        let shape = shape_binding.dims();

        if shape.is_empty() {
            return Err(TorshError::InvalidOperation("Cannot compute log_softmax on empty tensor".to_string()));
        }

        // Handle negative dimension
        let actual_dim = if dim < 0 {
            (shape.len() as i32 + dim) as usize
        } else {
            dim as usize
        };

        if actual_dim >= shape.len() {
            return Err(TorshError::InvalidArgument(format!("Dimension {} out of range for tensor with {} dimensions", dim, shape.len())));
        }

        let dim_size = shape[actual_dim];
        let outer_size: usize = shape[..actual_dim].iter().product();
        let inner_size: usize = shape[actual_dim + 1..].iter().product();

        let mut result_data = vec![T::from_f64(0.0).expect("f64 conversion should succeed"); data.len()];

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base_idx = outer * dim_size * inner_size + inner;

                // Find max for numerical stability
                let mut max_val = data[base_idx];
                for d in 1..dim_size {
                    let idx = base_idx + d * inner_size;
                    if data[idx] > max_val {
                        max_val = data[idx];
                    }
                }

                // Compute log(sum(exp(x - max)))
                let mut exp_sum = T::from_f64(0.0).expect("f64 conversion should succeed");
                for d in 0..dim_size {
                    let idx = base_idx + d * inner_size;
                    exp_sum = exp_sum + (data[idx] - max_val).exp();
                }
                let log_sum_exp = exp_sum.ln();

                // Compute log_softmax values: x - max - log(sum_exp)
                for d in 0..dim_size {
                    let idx = base_idx + d * inner_size;
                    result_data[idx] = data[idx] - max_val - log_sum_exp;
                }
            }
        }

        Self::from_data(
            result_data,
            shape.to_vec(),
            self.device,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_relu() {
        let tensor = Tensor::from_data(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0], vec![5], DeviceType::Cpu).expect("tensor creation failed");
        let result = tensor.relu().expect("relu failed");
        let data = result.data().expect("data retrieval failed");
        assert_eq!(data.as_slice(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_inplace() {
        let mut tensor = Tensor::from_data(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0], vec![5], DeviceType::Cpu).expect("tensor creation failed");
        tensor.relu_().expect("relu_ failed");
        let data = tensor.data().expect("data retrieval failed");
        assert_eq!(data.as_slice(), &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_leaky_relu() {
        let tensor = Tensor::from_data(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0], vec![5], DeviceType::Cpu).expect("tensor creation failed");
        let result = tensor.leaky_relu(0.1).expect("leaky_relu failed");
        let data = result.data().expect("data retrieval failed");
        assert_eq!(data.as_slice(), &[-0.2, -0.1, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let tensor = Tensor::from_data(vec![0.0f32, 1.0, -1.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");
        let result = tensor.sigmoid().expect("sigmoid failed");
        let data = result.data().expect("data retrieval failed");

        // sigmoid(0) = 0.5, sigmoid(1) ≈ 0.731, sigmoid(-1) ≈ 0.269
        assert!((data[0] - 0.5).abs() < 1e-6);
        assert!((data[1] - 0.7310586).abs() < 1e-6);
        assert!((data[2] - 0.26894143).abs() < 1e-6);
    }

    #[test]
    fn test_tanh() {
        let tensor = Tensor::from_data(vec![0.0f32, 1.0, -1.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");
        let result = tensor.tanh().expect("tanh failed");
        let data = result.data().expect("data retrieval failed");

        // tanh(0) = 0, tanh(1) ≈ 0.761, tanh(-1) ≈ -0.761
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[1] - 0.7615942).abs() < 1e-6);
        assert!((data[2] - (-0.7615942)).abs() < 1e-6);
    }

    #[test]
    fn test_gelu() {
        let tensor = Tensor::from_data(vec![0.0f32, 1.0, -1.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");
        let result = tensor.gelu().expect("gelu failed");
        let data = result.data().expect("data retrieval failed");

        // GELU should preserve relative ordering and be smooth
        assert!((data[0] - 0.0).abs() < 1e-5); // GELU(0) ≈ 0
        assert!(data[1] > 0.8); // GELU(1) should be close to 1
        assert!(data[2] < -0.1); // GELU(-1) should be negative but closer to 0
    }

    #[test]
    fn test_softmax() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");
        let result = tensor.softmax(-1).expect("softmax failed");
        let data = result.data().expect("data retrieval failed");

        // Check that values sum to 1
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that larger input gives larger output
        assert!(data[2] > data[1]);
        assert!(data[1] > data[0]);
    }

    #[test]
    fn test_log_softmax() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).expect("tensor creation failed");
        let result = tensor.log_softmax(-1).expect("log_softmax failed");
        let data = result.data().expect("data retrieval failed");

        // log_softmax values should be negative (since softmax values are < 1)
        assert!(data[0] < 0.0);
        assert!(data[1] < 0.0);
        assert!(data[2] < 0.0);

        // Larger input should give larger (less negative) log_softmax output
        assert!(data[2] > data[1]);
        assert!(data[1] > data[0]);
    }

    #[test]
    fn test_softmax_2d() {
        let tensor = Tensor::from_data(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu
        ).expect("tensor creation failed");

        // Test softmax along last dimension (dim=-1)
        let result = tensor.softmax(-1).expect("softmax failed");
        let data = result.data().expect("data retrieval failed");

        // Check that each row sums to 1
        let row1_sum: f32 = data[0..3].iter().sum();
        let row2_sum: f32 = data[3..6].iter().sum();
        assert!((row1_sum - 1.0).abs() < 1e-6);
        assert!((row2_sum - 1.0).abs() < 1e-6);
    }
}