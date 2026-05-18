//! Transcendental and trigonometric math operations for tensors.
//!
//! This module is included by `math_ops.rs` via `#[path]` and re-exported with `pub use`.
//! It covers all floating-point transcendental functions:
//! - Square root, exponential, logarithms
//! - Trigonometric: sin, cos, tan, asin, acos, atan
//! - Hyperbolic: sinh, cosh, tanh
//! - Activation: GELU, leaky ReLU
//! - Rounding: floor, ceil, round, trunc, fract
//! - Power, negation, sign

use super::*;

// Mathematical functions for floating-point tensors
impl<T: TensorElement + Copy> Tensor<T>
where
    T: scirs2_core::numeric::Float + torsh_core::dtype::FloatElement,
{
    /// Square root of all elements
    pub fn sqrt(&self) -> Result<Self> {
        self.map(|x| x.sqrt())
    }

    /// Square of all elements
    pub fn square(&self) -> Result<Self> {
        self.map(|x| x * x)
    }

    /// Reciprocal square root of all elements (1/sqrt(x))
    pub fn rsqrt(&self) -> Result<Self> {
        self.map(|x| T::from(1.0).expect("numeric conversion should succeed") / x.sqrt())
    }

    /// Reciprocal of all elements (1/x)
    pub fn reciprocal(&self) -> Result<Self> {
        self.map(|x| T::from(1.0).expect("numeric conversion should succeed") / x)
    }

    /// Exponential of all elements
    pub fn exp(&self) -> Result<Self> {
        self.map(|x| x.exp())
    }

    /// Natural logarithm of all elements
    pub fn ln(&self) -> Result<Self> {
        self.map(|x| x.ln())
    }

    /// Logarithm base 10 of all elements
    pub fn log10(&self) -> Result<Self> {
        self.map(|x| x.log10())
    }

    /// Logarithm base 2 of all elements
    pub fn log2(&self) -> Result<Self> {
        self.map(|x| x.log2())
    }

    /// Natural logarithm of all elements
    pub fn log(&self) -> Result<Self> {
        self.map(|x| x.ln())
    }

    /// Sine of all elements
    pub fn sin(&self) -> Result<Self> {
        self.map(|x| x.sin())
    }

    /// Cosine of all elements
    pub fn cos(&self) -> Result<Self> {
        self.map(|x| x.cos())
    }

    /// Tangent of all elements
    pub fn tan(&self) -> Result<Self> {
        self.map(|x| x.tan())
    }

    /// GELU (Gaussian Error Linear Unit) activation function with GPU and SIMD optimization
    pub fn gelu(&self) -> Result<Self> {
        // ✅ SciRS2 GPU Acceleration - Use GPU for very large tensors (10x-100x speedup potential)
        #[cfg(feature = "gpu")]
        {
            if self.numel() > 50000 {
                if let Ok(result) = self.gpu_gelu() {
                    return Ok(result);
                }
            }
        }

        // ✅ SciRS2 SIMD Optimization - Vectorized GELU for f32 tensors
        #[cfg(feature = "simd")]
        {
            if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() && self.numel() > 1000 {
                return self.simd_gelu_f32();
            }
        }

        // ✅ SciRS2 Parallel Processing - Use parallel computation for medium tensors
        #[cfg(feature = "parallel")]
        {
            if self.numel() > 100 {
                return self.parallel_map(|x| self.compute_gelu_scalar(x));
            }
        }

        // Fallback to sequential processing
        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        self.map(|x| self.compute_gelu_scalar(x))
    }

    /// GPU-accelerated GELU activation function
    ///
    /// NOTE: The real GPU dispatch for GELU now lives in
    /// [`crate::ops::activation::Tensor::gelu`] via `try_gpu_unary_f32`, which
    /// routes f32 CUDA tensors to `scirs2_core::gpu::GpuContext::gelu`.  This
    /// legacy method is retained as a placeholder for downstream callers that
    /// might still reach for it but is otherwise unused; it returns an error
    /// to make any accidental use loud.
    #[cfg(feature = "gpu")]
    #[allow(dead_code)]
    fn gpu_gelu(&self) -> Result<Self>
    where
        T: torsh_core::dtype::FloatElement,
    {
        Err(TorshError::InvalidArgument(
            "Use Tensor::gelu(): the real GPU dispatch lives in ops::activation".to_string(),
        ))
    }

    /// Compute GELU for a single scalar value
    fn compute_gelu_scalar(&self, x: T) -> T {
        let pi = T::from(std::f64::consts::PI).expect("numeric conversion should succeed");
        let two = T::from(2.0).expect("numeric conversion should succeed");
        let sqrt_2_over_pi = (two / pi).sqrt();
        let point_044715 = T::from(0.044715).expect("numeric conversion should succeed");
        let one = <T as scirs2_core::numeric::One>::one();
        let half = T::from(0.5).expect("numeric conversion should succeed");

        let x_cubed = x * x * x;
        let tanh_input = sqrt_2_over_pi * (x + point_044715 * x_cubed);
        half * x * (one + tanh_input.tanh())
    }

    /// SIMD-optimized GELU activation function for f32 tensors
    #[cfg(feature = "simd")]
    fn simd_gelu_f32(&self) -> Result<Self> {
        use scirs2_core::ndarray::ArrayView1;

        let data = self.data()?;

        // Cast to f32 for SIMD operations
        let data_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len()) };

        // Create ArrayView1 for SIMD function
        let data_view = ArrayView1::from(data_f32);

        // Use scirs2_core SIMD-accelerated GELU
        let result_array = adaptive_simd::adaptive_simd_gelu_f32(&data_view);

        // Convert result back to T type
        let result_vec: Vec<T> = result_array
            .to_vec()
            .into_iter()
            .map(|f| unsafe { std::mem::transmute_copy::<f32, T>(&f) })
            .collect();

        Self::from_data(
            result_vec,
            self.shape().dims().to_vec(),
            self.device.clone(),
        )
    }

    /// Leaky ReLU activation function with negative slope
    pub fn leaky_relu(&self, negative_slope: T) -> Result<Self> {
        self.map(|x| {
            if x > scirs2_core::numeric::Zero::zero() {
                x
            } else {
                negative_slope * x
            }
        })
    }

    /// Arcsine of all elements
    pub fn asin(&self) -> Result<Self> {
        self.map(|x| x.asin())
    }

    /// Arccosine of all elements
    pub fn acos(&self) -> Result<Self> {
        self.map(|x| x.acos())
    }

    /// Arctangent of all elements
    pub fn atan(&self) -> Result<Self> {
        self.map(|x| x.atan())
    }

    /// Hyperbolic sine of all elements
    pub fn sinh(&self) -> Result<Self> {
        self.map(|x| x.sinh())
    }

    /// Hyperbolic cosine of all elements
    pub fn cosh(&self) -> Result<Self> {
        self.map(|x| x.cosh())
    }

    /// Hyperbolic tangent of all elements
    pub fn tanh(&self) -> Result<Self> {
        self.map(|x| x.tanh())
    }

    /// Power function (element-wise)
    pub fn pow(&self, exponent: T) -> Result<Self>
    where
        T: TensorElement + Into<f32>,
    {
        // Convert T to f32 for the Operation::Power storage
        let exponent_f32: f32 = exponent.into();

        let mut result = self.map(|x| x.powf(exponent))?;

        // Set up gradient computation if needed
        if self.requires_grad {
            result.requires_grad = true;
            result.operation = Operation::Power {
                input: Arc::new(self.clone()),
                exponent: exponent_f32,
            };
        }

        Ok(result)
    }

    /// Power function with scalar exponent (alias for pow)
    pub fn pow_scalar(&self, exponent: T) -> Result<Self>
    where
        T: TensorElement + Into<f32>,
    {
        self.pow(exponent)
    }

    /// Power function with tensor exponents
    pub fn pow_tensor(&self, exponent: &Self) -> Result<Self> {
        self.elementwise_operation(exponent, |base, exp| base.powf(exp))
    }

    /// Floor of all elements
    pub fn floor(&self) -> Result<Self> {
        self.map(|x| x.floor())
    }

    /// Ceiling of all elements
    pub fn ceil(&self) -> Result<Self> {
        self.map(|x| x.ceil())
    }

    /// Round to nearest integer
    pub fn round(&self) -> Result<Self> {
        self.map(|x| x.round())
    }

    /// Truncate to integer part
    pub fn trunc(&self) -> Result<Self> {
        self.map(|x| x.trunc())
    }

    /// Fractional part
    pub fn fract(&self) -> Result<Self> {
        self.map(|x| x.fract())
    }

    /// Negation of all elements
    pub fn neg(&self) -> Result<Self>
    where
        T: std::ops::Neg<Output = T>,
    {
        self.map(|x| -x)
    }

    /// Sign of all elements (-1, 0, or 1)
    pub fn sign(&self) -> Result<Self> {
        self.map(|x| {
            if x > <T as scirs2_core::numeric::Zero>::zero() {
                <T as scirs2_core::numeric::One>::one()
            } else if x < <T as scirs2_core::numeric::Zero>::zero() {
                -<T as scirs2_core::numeric::One>::one()
            } else {
                <T as scirs2_core::numeric::Zero>::zero()
            }
        })
    }
}
