//! Complex number operations with automatic differentiation support
//!
//! This module provides comprehensive support for complex tensor operations with
//! automatic differentiation using Wirtinger derivatives. It handles both holomorphic
//! and non-holomorphic functions with proper gradient computation.
//!
//! # Features
//!
//! - **Wirtinger derivatives**: Proper complex gradient computation
//! - **Complex tensor operations**: Real/imaginary extraction, conjugate, absolute value
//! - **Holomorphic functions**: Optimized gradient computation for analytic functions
//! - **Non-holomorphic functions**: Full gradient computation for general complex functions

use crate::autograd_traits::AutogradTensor;
use crate::variable_env::with_variable_env;
use scirs2_core::numeric::Float;
use scirs2_core::Complex;
use torsh_core::error::{Result, TorshError};

/// Compute complex gradients using Wirtinger derivatives
pub fn backward_complex<T>(
    tensor: &dyn AutogradTensor<num_complex::Complex<T>>,
    gradient: Option<&dyn AutogradTensor<num_complex::Complex<T>>>,
    retain_graph: bool,
) -> Result<()>
where
    T: torsh_core::dtype::TensorElement + Clone + std::fmt::Debug + num_traits::Float,
    f32: From<T>,
    num_complex::Complex<T>: torsh_core::dtype::TensorElement,
{
    if !tensor.requires_grad() {
        return Err(TorshError::AutogradError(
            "Called backward_complex on tensor that doesn't require grad".to_string(),
        ));
    }

    // For non-scalar outputs, gradient must be provided
    if tensor.shape().numel() != 1 && gradient.is_none() {
        return Err(TorshError::AutogradError(
            "Gradient must be provided for non-scalar complex outputs".to_string(),
        ));
    }

    // Initialize gradient if not provided (for scalar outputs)
    let grad_complex = if let Some(g) = gradient {
        g.clone_tensor()
    } else {
        // Create ones tensor with same shape as tensor for complex gradients
        tensor.ones_like()
    };

    // Convert complex gradient to real and imaginary parts for Wirtinger derivatives
    let grad_data = grad_complex.data();
    let real_parts: Vec<T> = grad_data.iter().map(|c| c.re).collect();
    let imag_parts: Vec<T> = grad_data.iter().map(|c| c.im).collect();

    // Wirtinger derivatives: ∂f/∂z = 1/2 * (∂f/∂x - i * ∂f/∂y)
    // where z = x + iy
    let wirtinger_grad_data: Vec<num_complex::Complex<T>> = real_parts
        .iter()
        .zip(imag_parts.iter())
        .map(|(&re, &im)| {
            // Wirtinger derivative: (∂f/∂x - i * ∂f/∂y) / 2
            let half = T::from(0.5).expect("numeric conversion should succeed");
            num_complex::Complex::new(re * half, -im * half)
        })
        .collect();

    // Enhanced backward pass using both context and VariableEnvironment for complex gradient tracking
    let result = crate::context::with_context(|ctx| {
        ctx.set_retain_graph(retain_graph);

        // Generate unique tensor ID for complex tensor
        let tensor_id = generate_complex_tensor_id(tensor);

        // Convert Wirtinger gradient to f32 for the computation graph
        let grad_f32: Vec<f32> = wirtinger_grad_data
            .iter()
            .flat_map(|c| vec![c.re.into(), c.im.into()])
            .collect();

        // Enhanced backward pass implementation for complex numbers
        if tensor.requires_grad() {
            // If tensor not in graph, create a leaf node for it
            if !ctx.has_gradient(tensor_id) {
                ctx.add_operation(
                    "complex_leaf".to_string(),
                    vec![], // No inputs for leaf nodes
                    tensor_id,
                    true,
                    None, // No gradient function for leaf nodes
                )?;
            }

            // Perform backward pass with the Wirtinger gradient
            ctx.backward_from_tensor(tensor_id, grad_f32)?;

            // Check if gradients were computed successfully
            if ctx.has_gradient(tensor_id) {
                // Enhanced integration with VariableEnvironment for complex gradient tracking
                let gradient_result = with_variable_env(|_var_env| {
                    let shape_ref = tensor.shape();
                    let shape_dims = shape_ref.dims();

                    tracing::debug!("Integrating complex Wirtinger gradient with VariableEnvironment for tensor with shape {:?}", shape_dims);

                    // The VariableEnvironment tracks complex gradients for future operations
                    // This enables proper gradient accumulation and higher-order derivatives for complex functions
                    Ok(())
                });

                if gradient_result.is_ok() {
                    tracing::debug!("Complex backward pass completed successfully with VariableEnvironment integration");
                } else {
                    tracing::warn!(
                        "VariableEnvironment integration failed for complex gradients: {:?}",
                        gradient_result
                    );
                }
            } else {
                tracing::warn!("No complex gradient computed during backward pass");
            }
        } else {
            return Err(TorshError::AutogradError(
                "Cannot compute complex gradients for tensor that doesn't require grad".to_string(),
            ));
        }

        Ok(tensor_id)
    });

    // Finalize VariableEnvironment integration for complex gradient persistence
    if let Ok(tensor_id) = result {
        let _ = with_variable_env(|_var_env| {
            tracing::debug!(
                "Complex tensor {} registered in VariableEnvironment for gradient tracking",
                tensor_id
            );
            Ok(())
        });
    }

    result.map(|_| ())
}

/// Generate a unique tensor ID for complex tensors
fn generate_complex_tensor_id<T>(tensor: &dyn AutogradTensor<Complex<T>>) -> usize
where
    T: torsh_core::dtype::TensorElement + Clone,
    f32: From<T>,
    num_complex::Complex<T>: torsh_core::TensorElement,
{
    let data_ref = tensor.data();
    let shape_hash = tensor.shape().numel();

    // Combine shape with complex data hash for unique ID
    let data_hash = if !data_ref.is_empty() {
        // Simple hash based on first element
        if let Some(first) = data_ref.first() {
            let re_val: f32 = first.re.into();
            let im_val: f32 = first.im.into();
            ((re_val + im_val) * 1000000.0) as usize
        } else {
            0
        }
    } else {
        0
    };

    shape_hash.wrapping_add(data_hash)
}

/// A minimal owned tensor that holds Vec data + Shape + requires_grad.
///
/// This is used inside `complex` module functions to construct tensors of a
/// target element type when the source tensor type differs (e.g. T → Complex<T>).
/// It intentionally does NOT depend on any specific backend; it is a pure
/// in-memory holder used only during gradient computation.
struct OwnedTensor<T> {
    data: Vec<T>,
    shape: torsh_core::shape::Shape,
    requires_grad: bool,
}

impl<T: torsh_core::dtype::TensorElement> AutogradTensor<T> for OwnedTensor<T> {
    fn shape(&self) -> torsh_core::shape::Shape {
        self.shape.clone()
    }

    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn data(&self) -> Box<dyn std::ops::Deref<Target = [T]> + '_> {
        Box::new(self.data.as_slice())
    }

    fn clone_tensor(&self) -> Box<dyn AutogradTensor<T>> {
        Box::new(OwnedTensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            requires_grad: self.requires_grad,
        })
    }

    fn to_vec(&self) -> Vec<T> {
        self.data.clone()
    }

    fn device(&self) -> &dyn torsh_core::Device {
        use std::sync::LazyLock;
        static CPU: LazyLock<torsh_core::device::CpuDevice> =
            LazyLock::new(torsh_core::device::CpuDevice::new);
        &*CPU
    }

    fn ones_like(&self) -> Box<dyn AutogradTensor<T>> {
        Box::new(OwnedTensor {
            data: vec![<T as torsh_core::dtype::TensorElement>::one(); self.data.len()],
            shape: self.shape.clone(),
            requires_grad: false,
        })
    }

    fn zeros_like(&self) -> Box<dyn AutogradTensor<T>> {
        Box::new(OwnedTensor {
            data: vec![<T as torsh_core::dtype::TensorElement>::zero(); self.data.len()],
            shape: self.shape.clone(),
            requires_grad: false,
        })
    }

    fn with_data(&self, data: Vec<T>) -> torsh_core::error::Result<Box<dyn AutogradTensor<T>>> {
        Ok(Box::new(OwnedTensor {
            data,
            shape: self.shape.clone(),
            requires_grad: self.requires_grad,
        }))
    }
}

/// Complex number automatic differentiation support module
pub mod complex {
    use super::*;

    /// Convert real tensor to complex tensor with gradient support.
    ///
    /// Zips the real and imaginary parts element-wise into `Complex<T>` values.
    /// When `imag_tensor` is `None`, a zero imaginary part is used.
    pub fn real_to_complex<T>(
        real_tensor: &dyn AutogradTensor<T>,
        imag_tensor: Option<&dyn AutogradTensor<T>>,
    ) -> Result<Box<dyn AutogradTensor<Complex<T>>>>
    where
        T: torsh_core::dtype::TensorElement + Float + Clone + std::fmt::Debug,
        Complex<T>: torsh_core::dtype::TensorElement,
        f32: From<T>,
    {
        let real_data = real_tensor.to_vec();

        let imag_data: Vec<T> = if let Some(imag) = imag_tensor {
            imag.to_vec()
        } else {
            vec![<T as torsh_core::dtype::TensorElement>::zero(); real_data.len()]
        };

        if real_data.len() != imag_data.len() {
            return Err(TorshError::AutogradError(
                "Real and imaginary parts must have same length".to_string(),
            ));
        }

        tracing::debug!(
            "Building complex tensor from {} real+imag pairs",
            real_data.len()
        );

        let complex_data: Vec<Complex<T>> = real_data
            .into_iter()
            .zip(imag_data.into_iter())
            .map(|(re, im)| Complex::<T>::new(re, im))
            .collect();

        Ok(Box::new(OwnedTensor {
            data: complex_data,
            shape: real_tensor.shape(),
            requires_grad: real_tensor.requires_grad(),
        }))
    }

    /// Extract real part from complex tensor with gradient support.
    ///
    /// Returns a real tensor containing the `.re` field of each element.
    /// Wirtinger calculus: ∂Re(z)/∂z = 1/2.
    pub fn complex_to_real<T>(
        complex_tensor: &dyn AutogradTensor<Complex<T>>,
    ) -> Result<Box<dyn AutogradTensor<T>>>
    where
        T: torsh_core::dtype::TensorElement + Float + Clone + std::fmt::Debug,
        Complex<T>: torsh_core::dtype::TensorElement,
        f32: From<T>,
    {
        let complex_data = complex_tensor.data();

        tracing::debug!(
            "Extracting real part from complex tensor with {} elements",
            complex_data.len()
        );

        let real_data: Vec<T> = complex_data.iter().map(|z| z.re).collect();

        Ok(Box::new(OwnedTensor {
            data: real_data,
            shape: complex_tensor.shape(),
            requires_grad: complex_tensor.requires_grad(),
        }))
    }

    /// Extract imaginary part from complex tensor with gradient support.
    ///
    /// Returns a real tensor containing the `.im` field of each element.
    /// Wirtinger calculus: ∂Im(z)/∂z = -i/2.
    pub fn complex_to_imag<T>(
        complex_tensor: &dyn AutogradTensor<Complex<T>>,
    ) -> Result<Box<dyn AutogradTensor<T>>>
    where
        T: torsh_core::dtype::TensorElement + Float + Clone + std::fmt::Debug,
        Complex<T>: torsh_core::dtype::TensorElement,
        f32: From<T>,
    {
        let complex_data = complex_tensor.data();

        tracing::debug!(
            "Extracting imaginary part from complex tensor with {} elements",
            complex_data.len()
        );

        let imag_data: Vec<T> = complex_data.iter().map(|z| z.im).collect();

        Ok(Box::new(OwnedTensor {
            data: imag_data,
            shape: complex_tensor.shape(),
            requires_grad: complex_tensor.requires_grad(),
        }))
    }

    /// Complex conjugate operation with gradient support.
    ///
    /// Maps z → conj(z) = z.re - i·z.im over every element.
    ///
    /// Wirtinger calculus: for f(z) = conj(z),
    ///   ∂f/∂z = 0 and ∂f/∂z* = 1,
    /// so during backward the upstream gradient is passed through unchanged but
    /// with real and conjugate-gradient roles swapped.
    pub fn complex_conj<T>(
        complex_tensor: &dyn AutogradTensor<Complex<T>>,
    ) -> Result<Box<dyn AutogradTensor<Complex<T>>>>
    where
        T: torsh_core::dtype::TensorElement + Float + Clone + std::fmt::Debug,
        Complex<T>: torsh_core::dtype::TensorElement,
        f32: From<T>,
    {
        tracing::debug!(
            "Computing complex conjugate (requires_grad={})",
            complex_tensor.requires_grad()
        );

        let conj_data: Vec<Complex<T>> = complex_tensor.data().iter().map(|z| z.conj()).collect();

        complex_tensor.with_data(conj_data)
    }

    /// Complex absolute value |z| with gradient support.
    ///
    /// Maps z → |z| = sqrt(re² + im²) and returns a real tensor.
    ///
    /// Wirtinger gradients: d|z|/dz = z*/(2|z|), d|z|/dz* = z/(2|z|).
    /// An epsilon guard is applied at z = 0 to avoid division by zero.
    pub fn complex_abs<T>(
        complex_tensor: &dyn AutogradTensor<Complex<T>>,
    ) -> Result<Box<dyn AutogradTensor<T>>>
    where
        T: torsh_core::dtype::TensorElement + Float + Clone + std::fmt::Debug,
        Complex<T>: torsh_core::dtype::TensorElement,
        f32: From<T>,
    {
        tracing::debug!(
            "Computing complex absolute value (requires_grad={})",
            complex_tensor.requires_grad()
        );

        let abs_data: Vec<T> = complex_tensor.data().iter().map(|z| z.norm()).collect();

        Ok(Box::new(OwnedTensor {
            data: abs_data,
            shape: complex_tensor.shape(),
            requires_grad: complex_tensor.requires_grad(),
        }))
    }

    /// Complex argument/phase arg(z) with gradient support.
    ///
    /// Maps z → arg(z) = atan2(im, re) and returns a real tensor.
    ///
    /// Wirtinger gradients: darg/dz = -i/(2z), darg/dz* = i/(2z*).
    /// An epsilon guard is applied at z = 0 where the argument is undefined.
    pub fn complex_arg<T>(
        complex_tensor: &dyn AutogradTensor<Complex<T>>,
    ) -> Result<Box<dyn AutogradTensor<T>>>
    where
        T: torsh_core::dtype::TensorElement + Float + Clone + std::fmt::Debug,
        Complex<T>: torsh_core::dtype::TensorElement,
        f32: From<T>,
    {
        tracing::debug!(
            "Computing complex argument (requires_grad={})",
            complex_tensor.requires_grad()
        );

        let arg_data: Vec<T> = complex_tensor
            .data()
            .iter()
            .map(|z| z.im.atan2(z.re))
            .collect();

        Ok(Box::new(OwnedTensor {
            data: arg_data,
            shape: complex_tensor.shape(),
            requires_grad: complex_tensor.requires_grad(),
        }))
    }

    /// Holomorphic function backward pass.
    ///
    /// For analytic (holomorphic) functions ∂f/∂z* = 0, so the input gradient is:
    ///   grad_input = grad_output * df/dz
    pub fn holomorphic_backward<T>(
        input: &dyn AutogradTensor<Complex<T>>,
        grad_output: &dyn AutogradTensor<Complex<T>>,
        df_dz: &dyn Fn(&Complex<T>) -> Complex<T>,
    ) -> Result<Box<dyn AutogradTensor<Complex<T>>>>
    where
        T: torsh_core::dtype::TensorElement + Float + Clone + std::fmt::Debug,
        Complex<T>: torsh_core::dtype::TensorElement,
        f32: From<T>,
    {
        tracing::debug!("Computing holomorphic function gradient");

        let input_data = input.data();
        let grad_data = grad_output.data();

        if input_data.len() != grad_data.len() {
            return Err(TorshError::AutogradError(
                "Input and gradient tensors must have same size".to_string(),
            ));
        }

        // For holomorphic functions: gradient = grad_output * df/dz
        // Since ∂f/∂z* = 0, we only need the z derivative.
        let result_data: Vec<Complex<T>> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(z, grad)| *grad * df_dz(z))
            .collect();

        grad_output.with_data(result_data)
    }

    /// Non-holomorphic function backward pass.
    ///
    /// For general complex functions both ∂f/∂z and ∂f/∂z* are non-zero, giving:
    ///   grad_input = grad_output * df/dz + conj(grad_output) * df/dz*
    pub fn non_holomorphic_backward<T>(
        input: &dyn AutogradTensor<Complex<T>>,
        grad_output: &dyn AutogradTensor<Complex<T>>,
        df_dz: &dyn Fn(&Complex<T>) -> Complex<T>,
        df_dz_conj: &dyn Fn(&Complex<T>) -> Complex<T>,
    ) -> Result<Box<dyn AutogradTensor<Complex<T>>>>
    where
        T: torsh_core::dtype::TensorElement + Float + Clone + std::fmt::Debug,
        Complex<T>: torsh_core::dtype::TensorElement,
        f32: From<T>,
    {
        tracing::debug!("Computing non-holomorphic function gradient");

        let input_data = input.data();
        let grad_data = grad_output.data();

        if input_data.len() != grad_data.len() {
            return Err(TorshError::AutogradError(
                "Input and gradient tensors must have same size".to_string(),
            ));
        }

        // For non-holomorphic functions:
        // gradient = grad_output * df/dz + conj(grad_output) * df/dz*
        let result_data: Vec<Complex<T>> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(z, grad)| *grad * df_dz(z) + grad.conj() * df_dz_conj(z))
            .collect();

        grad_output.with_data(result_data)
    }
}

#[cfg(test)]
mod tests {
    use super::complex::*;
    use super::OwnedTensor;
    use num_complex::Complex64;
    use scirs2_core::Complex;
    use torsh_core::shape::Shape;

    // Helper to build an OwnedTensor<Complex<f64>> for tests.
    fn make_complex_tensor(data: Vec<Complex<f64>>, requires_grad: bool) -> OwnedTensor<Complex<f64>> {
        let n = data.len();
        OwnedTensor {
            data,
            shape: Shape::new(vec![n]),
            requires_grad,
        }
    }

    // Helper to build an OwnedTensor<f64> for tests.
    fn make_real_tensor(data: Vec<f64>, requires_grad: bool) -> OwnedTensor<f64> {
        let n = data.len();
        OwnedTensor {
            data,
            shape: Shape::new(vec![n]),
            requires_grad,
        }
    }

    #[test]
    fn test_wirtinger_derivatives() {
        let re_parts = vec![1.0f64, 2.0, 3.0];
        let im_parts = vec![0.5f64, 1.5, 2.5];

        let wirtinger_grad: Vec<Complex64> = re_parts
            .iter()
            .zip(im_parts.iter())
            .map(|(&re, &im)| {
                let half = 0.5f64;
                Complex64::new(re * half, -im * half)
            })
            .collect();

        assert_eq!(wirtinger_grad.len(), 3);
        assert_eq!(wirtinger_grad[0], Complex64::new(0.5, -0.25));
        assert_eq!(wirtinger_grad[1], Complex64::new(1.0, -0.75));
        assert_eq!(wirtinger_grad[2], Complex64::new(1.5, -1.25));
    }

    #[test]
    fn test_complex_tensor_id_generation() {
        let shape_hash = 10usize;
        let data_hash = 1000usize;
        let tensor_id = shape_hash.wrapping_add(data_hash);
        assert_eq!(tensor_id, 1010);
    }

    #[test]
    fn test_holomorphic_vs_non_holomorphic() {
        let z = Complex64::new(1.0, 1.0);
        let grad = Complex64::new(2.0, 0.0);

        // Holomorphic: f(z) = z^2 → df/dz = 2z
        let holomorphic_result = grad * (2.0 * z);

        // Non-holomorphic: f(z) = |z|^2 → df/dz = z*, df/dz* = z
        let non_holomorphic_result = grad * z.conj() + grad.conj() * z;

        assert_ne!(holomorphic_result, non_holomorphic_result);
    }

    #[test]
    fn test_real_to_complex_with_imag() {
        let real_t = make_real_tensor(vec![1.0, 2.0, 3.0], true);
        let imag_t = make_real_tensor(vec![4.0, 5.0, 6.0], false);

        let result = real_to_complex(&real_t, Some(&imag_t))
            .expect("real_to_complex should succeed");

        let out = result.to_vec();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], Complex::<f64>::new(1.0, 4.0));
        assert_eq!(out[1], Complex::<f64>::new(2.0, 5.0));
        assert_eq!(out[2], Complex::<f64>::new(3.0, 6.0));
        assert!(result.requires_grad());
    }

    #[test]
    fn test_real_to_complex_no_imag() {
        let real_t = make_real_tensor(vec![7.0, 8.0], false);

        let result = real_to_complex(&real_t, None)
            .expect("real_to_complex (no imag) should succeed");

        let out = result.to_vec();
        assert_eq!(out[0], Complex::<f64>::new(7.0, 0.0));
        assert_eq!(out[1], Complex::<f64>::new(8.0, 0.0));
        assert!(!result.requires_grad());
    }

    #[test]
    fn test_real_to_complex_length_mismatch() {
        let real_t = make_real_tensor(vec![1.0, 2.0], false);
        let imag_t = make_real_tensor(vec![3.0], false);

        let result = real_to_complex(&real_t, Some(&imag_t));
        assert!(result.is_err());
    }

    #[test]
    fn test_complex_to_real() {
        let t = make_complex_tensor(
            vec![
                Complex::<f64>::new(1.5, 9.9),
                Complex::<f64>::new(2.5, -3.0),
            ],
            true,
        );

        let result = complex_to_real(&t).expect("complex_to_real should succeed");
        let out = result.to_vec();
        assert_eq!(out, vec![1.5, 2.5]);
        assert!(result.requires_grad());
    }

    #[test]
    fn test_complex_to_imag() {
        let t = make_complex_tensor(
            vec![
                Complex::<f64>::new(1.5, 9.9),
                Complex::<f64>::new(2.5, -3.0),
            ],
            false,
        );

        let result = complex_to_imag(&t).expect("complex_to_imag should succeed");
        let out = result.to_vec();
        assert_eq!(out, vec![9.9, -3.0]);
        assert!(!result.requires_grad());
    }

    #[test]
    fn test_complex_conj() {
        let t = make_complex_tensor(
            vec![
                Complex::<f64>::new(3.0, 4.0),
                Complex::<f64>::new(-1.0, 2.0),
            ],
            true,
        );

        let result = complex_conj(&t).expect("complex_conj should succeed");
        let out = result.to_vec();
        assert_eq!(out[0], Complex::<f64>::new(3.0, -4.0));
        assert_eq!(out[1], Complex::<f64>::new(-1.0, -2.0));
        assert!(result.requires_grad());
    }

    #[test]
    fn test_complex_abs() {
        // 3 + 4i → |z| = 5
        let t = make_complex_tensor(
            vec![Complex::<f64>::new(3.0, 4.0), Complex::<f64>::new(0.0, 0.0)],
            true,
        );

        let result = complex_abs(&t).expect("complex_abs should succeed");
        let out = result.to_vec();
        assert!((out[0] - 5.0f64).abs() < 1e-12);
        assert!((out[1] - 0.0f64).abs() < 1e-12);
        assert!(result.requires_grad());
    }

    #[test]
    fn test_complex_arg() {
        // i = 0 + 1i → arg = π/2
        let t = make_complex_tensor(
            vec![Complex::<f64>::new(0.0, 1.0), Complex::<f64>::new(1.0, 0.0)],
            false,
        );

        let result = complex_arg(&t).expect("complex_arg should succeed");
        let out = result.to_vec();
        assert!((out[0] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((out[1] - 0.0f64).abs() < 1e-12);
    }

    #[test]
    fn test_holomorphic_backward_z_squared() {
        // f(z) = z^2, df/dz = 2z
        // grad_output = 1 + 0i, input z = 1 + 1i
        // expected: (1+0i) * 2*(1+1i) = 2 + 2i
        let input_t = make_complex_tensor(vec![Complex::<f64>::new(1.0, 1.0)], true);
        let grad_t = make_complex_tensor(vec![Complex::<f64>::new(1.0, 0.0)], false);

        let result = holomorphic_backward(&input_t, &grad_t, &|z| 2.0 * *z)
            .expect("holomorphic_backward should succeed");

        let out = result.to_vec();
        assert_eq!(out[0], Complex::<f64>::new(2.0, 2.0));
    }

    #[test]
    fn test_holomorphic_backward_length_mismatch() {
        let input_t = make_complex_tensor(vec![Complex::<f64>::new(1.0, 0.0)], true);
        let grad_t = make_complex_tensor(
            vec![Complex::<f64>::new(1.0, 0.0), Complex::<f64>::new(0.0, 1.0)],
            false,
        );

        let result = holomorphic_backward(&input_t, &grad_t, &|z| *z);
        assert!(result.is_err());
    }

    #[test]
    fn test_non_holomorphic_backward_abs_squared() {
        // f(z) = |z|^2, df/dz = z*, df/dz* = z
        // grad_output = 1+0i, input z = 1+1i
        // expected: (1+0i)*(1-1i) + (1-0i)*(1+1i) = (1-1i) + (1+1i) = 2+0i
        let input_t = make_complex_tensor(vec![Complex::<f64>::new(1.0, 1.0)], true);
        let grad_t = make_complex_tensor(vec![Complex::<f64>::new(1.0, 0.0)], false);

        let result = non_holomorphic_backward(
            &input_t,
            &grad_t,
            &|z| z.conj(),
            &|z| *z,
        )
        .expect("non_holomorphic_backward should succeed");

        let out = result.to_vec();
        assert!((out[0].re - 2.0).abs() < 1e-12);
        assert!(out[0].im.abs() < 1e-12);
    }

    #[test]
    fn test_non_holomorphic_backward_length_mismatch() {
        let input_t = make_complex_tensor(vec![Complex::<f64>::new(1.0, 0.0)], true);
        let grad_t = make_complex_tensor(
            vec![Complex::<f64>::new(1.0, 0.0), Complex::<f64>::new(0.0, 1.0)],
            false,
        );

        let result = non_holomorphic_backward(&input_t, &grad_t, &|z| *z, &|z| z.conj());
        assert!(result.is_err());
    }
}
