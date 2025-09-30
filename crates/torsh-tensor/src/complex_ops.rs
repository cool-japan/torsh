//! Complex number operations for tensors
//!
//! This module provides specialized operations for complex-valued tensors including
//! complex conjugation, real/imaginary part extraction, and complex-specific
//! automatic differentiation support.
//!
//! # Features
//!
//! - **Complex conjugation**: Efficient complex conjugate operations
//! - **Component extraction**: Real and imaginary part access
//! - **Complex autograd**: Specialized gradient computation for complex numbers
//! - **Complex arithmetic**: Element-wise operations preserving complex structure
//! - **Magnitude and phase**: Polar representation support

use num_traits::Float;
use std::sync::Arc;
use torsh_core::{
    dtype::{ComplexElement, TensorElement},
    error::{Result, TorshError},
};

use crate::core_ops::{Operation, Tensor};

impl<T: ComplexElement + Copy> Tensor<T> {
    /// Complex conjugate for complex tensors
    pub fn complex_conj(&self) -> Result<Self>
    where
        T: Copy,
    {
        let data = self.to_vec()?;
        let conj_data: Vec<T> = data.iter().map(|&z| z.conj()).collect();
        let mut result = Self::from_data(conj_data, self.shape().dims().to_vec(), self.device)?;
        result.requires_grad = self.requires_grad;

        // Set up operation tracking for autograd
        if self.requires_grad {
            result.operation = Operation::Custom(
                "complex_conj".to_string(),
                vec![Arc::downgrade(&Arc::new(self.clone()))],
            );
        }

        Ok(result)
    }

    /// Get real part of complex tensor
    pub fn real(&self) -> Result<Tensor<T::Real>>
    where
        T::Real: TensorElement + Copy,
    {
        let data = self.to_vec()?;
        let real_data: Vec<T::Real> = data.iter().map(|x| x.real()).collect();
        Tensor::from_data(real_data, self.shape().dims().to_vec(), self.device)
    }

    /// Get imaginary part of complex tensor
    pub fn imag(&self) -> Result<Tensor<T::Real>>
    where
        T::Real: TensorElement + Copy,
    {
        let data = self.to_vec()?;
        let imag_data: Vec<T::Real> = data.iter().map(|x| x.imag()).collect();
        Tensor::from_data(imag_data, self.shape().dims().to_vec(), self.device)
    }

    /// Get magnitude (absolute value) of complex tensor
    pub fn abs(&self) -> Result<Tensor<T::Real>>
    where
        T::Real: TensorElement + Copy + num_traits::Float,
    {
        let data = self.to_vec()?;
        let abs_data: Vec<T::Real> = data.iter().map(|x| x.abs()).collect();
        Tensor::from_data(abs_data, self.shape().dims().to_vec(), self.device)
    }

    /// Get phase (argument) of complex tensor
    pub fn angle(&self) -> Result<Tensor<T::Real>>
    where
        T::Real: TensorElement + Copy + num_traits::Float,
    {
        let data = self.to_vec()?;
        let angle_data: Vec<T::Real> = data.iter().map(|x| x.arg()).collect();
        Tensor::from_data(angle_data, self.shape().dims().to_vec(), self.device)
    }

    /// Create complex tensor from real and imaginary parts
    pub fn complex(real: &Tensor<T::Real>, imag: &Tensor<T::Real>) -> Result<Self>
    where
        T::Real: TensorElement + Copy,
    {
        if real.shape() != imag.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: real.shape().dims().to_vec(),
                got: imag.shape().dims().to_vec(),
            });
        }

        let real_data = real.to_vec()?;
        let imag_data = imag.to_vec()?;

        let complex_data: Vec<T> = real_data
            .iter()
            .zip(imag_data.iter())
            .map(|(&r, &i)| T::new(r, i))
            .collect();

        Self::from_data(complex_data, real.shape().dims().to_vec(), real.device)
    }

    /// Create complex tensor from polar representation (magnitude and phase)
    pub fn polar(magnitude: &Tensor<T::Real>, phase: &Tensor<T::Real>) -> Result<Self>
    where
        T::Real: TensorElement + Copy + num_traits::Float,
    {
        if magnitude.shape() != phase.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: magnitude.shape().dims().to_vec(),
                got: phase.shape().dims().to_vec(),
            });
        }

        let mag_data = magnitude.to_vec()?;
        let phase_data = phase.to_vec()?;

        let complex_data: Vec<T> = mag_data
            .iter()
            .zip(phase_data.iter())
            .map(|(&mag, &phase)| {
                let real = mag * phase.cos();
                let imag = mag * phase.sin();
                T::new(real, imag)
            })
            .collect();

        Self::from_data(
            complex_data,
            magnitude.shape().dims().to_vec(),
            magnitude.device,
        )
    }

    /// Backward pass for complex tensors (compute gradients)
    ///
    /// Complex autograd follows PyTorch's approach where gradients are computed
    /// treating complex numbers as 2D vectors of real numbers.
    pub fn backward_complex(&self) -> Result<()>
    where
        T: Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
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

        // Create initial gradient of 1.0 + 0.0i for the output
        let output_grad_data = vec![T::new(
            <T::Real as TensorElement>::one(),
            <T::Real as TensorElement>::zero(),
        )];
        let output_grad = Self::from_data(output_grad_data, vec![], self.device)?;

        // Start backpropagation
        self.backward_complex_impl(&output_grad)?;

        Ok(())
    }

    /// Internal backward implementation for complex tensors
    fn backward_complex_impl(&self, grad_output: &Self) -> Result<()>
    where
        T: Copy
            + Default
            + std::ops::Add<Output = T>
            + std::ops::Sub<Output = T>
            + std::ops::Mul<Output = T>
            + std::ops::Div<Output = T>,
    {
        match &self.operation {
            Operation::Leaf => {
                // Accumulate gradient for leaf nodes
                let mut grad_lock = self.grad.write().unwrap();
                if let Some(existing_grad) = grad_lock.as_ref() {
                    // Add gradients if they exist
                    let new_grad = existing_grad.add_op(grad_output)?;
                    *grad_lock = Some(new_grad);
                } else {
                    // Set gradient if it doesn't exist
                    *grad_lock = Some(grad_output.clone());
                }
            }
            Operation::Add { lhs, rhs } => {
                // Gradient flows through both operands unchanged for complex addition
                if lhs.requires_grad {
                    lhs.backward_complex_impl(grad_output)?;
                }
                if rhs.requires_grad {
                    rhs.backward_complex_impl(grad_output)?;
                }
            }
            Operation::Mul { lhs, rhs } => {
                // Complex multiplication rule: d/dz(f*g) = f'*g + f*g'
                if lhs.requires_grad {
                    let lhs_grad = (**rhs).mul_op(grad_output)?;
                    lhs.backward_complex_impl(&lhs_grad)?;
                }
                if rhs.requires_grad {
                    let rhs_grad = (**lhs).mul_op(grad_output)?;
                    rhs.backward_complex_impl(&rhs_grad)?;
                }
            }
            Operation::Custom(op_name, inputs) => {
                match op_name.as_str() {
                    "complex_conj" => {
                        // Gradient of complex conjugate: d/dz(conj(f)) = conj(df/dz)
                        if let Some(weak_input) = inputs.first() {
                            if let Some(input) = weak_input.upgrade() {
                                if input.requires_grad {
                                    let conj_grad = grad_output.complex_conj()?;
                                    input.backward_complex_impl(&conj_grad)?;
                                }
                            }
                        }
                    }
                    "complex_abs" => {
                        // Gradient of abs(z) = z / |z| for z != 0
                        if let Some(weak_input) = inputs.first() {
                            if let Some(input) = weak_input.upgrade() {
                                if input.requires_grad {
                                    let input_data = input.to_vec()?;
                                    let grad_data = grad_output.to_vec()?;

                                    let input_grad_data: Vec<T> = input_data
                                        .iter()
                                        .zip(grad_data.iter())
                                        .map(|(&z, &grad)| {
                                            let abs_z = z.abs();
                                            if abs_z > T::Real::zero() {
                                                // Gradient is z / |z| * grad_output
                                                let z_normalized =
                                                    T::new(z.real() / abs_z, z.imag() / abs_z);
                                                T::new(
                                                    z_normalized.real() * grad.real()
                                                        - z_normalized.imag() * grad.imag(),
                                                    z_normalized.real() * grad.imag()
                                                        + z_normalized.imag() * grad.real(),
                                                )
                                            } else {
                                                T::new(T::Real::zero(), T::Real::zero())
                                            }
                                        })
                                        .collect();

                                    let input_grad = Self::from_data(
                                        input_grad_data,
                                        input.shape().dims().to_vec(),
                                        input.device,
                                    )?;
                                    input.backward_complex_impl(&input_grad)?;
                                }
                            }
                        }
                    }
                    _ => {
                        // For other custom operations, propagate gradient to all inputs
                        for weak_input in inputs {
                            if let Some(input) = weak_input.upgrade() {
                                if input.requires_grad {
                                    input.backward_complex_impl(grad_output)?;
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                // For other operations, fall back to regular backward pass
                // This would call the regular backward_impl method
                // Note: This is a simplified approach - in practice, each operation
                // would need its own complex-specific gradient computation
            }
        }

        Ok(())
    }

    /// Element-wise complex multiplication with proper gradient tracking
    pub fn complex_mul(&self, other: &Self) -> Result<Self>
    where
        T: std::ops::Mul<Output = T> + std::ops::Add<Output = T> + std::ops::Sub<Output = T>,
    {
        if self.shape() != other.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: other.shape().dims().to_vec(),
            });
        }

        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;

        let result_data: Vec<T> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| {
                // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
                T::new(
                    a.real() * b.real() - a.imag() * b.imag(),
                    a.real() * b.imag() + a.imag() * b.real(),
                )
            })
            .collect();

        let mut result = Self::from_data(result_data, self.shape().dims().to_vec(), self.device)?;

        // Set up gradient tracking
        if self.requires_grad || other.requires_grad {
            result.requires_grad = true;
            result.operation = Operation::Mul {
                lhs: Arc::new(self.clone()),
                rhs: Arc::new(other.clone()),
            };
        }

        Ok(result)
    }

    /// Element-wise complex addition with proper gradient tracking
    pub fn complex_add(&self, other: &Self) -> Result<Self>
    where
        T: std::ops::Add<Output = T>,
    {
        if self.shape() != other.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: other.shape().dims().to_vec(),
            });
        }

        let self_data = self.to_vec()?;
        let other_data = other.to_vec()?;

        let result_data: Vec<T> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| T::new(a.real() + b.real(), a.imag() + b.imag()))
            .collect();

        let mut result = Self::from_data(result_data, self.shape().dims().to_vec(), self.device)?;

        // Set up gradient tracking
        if self.requires_grad || other.requires_grad {
            result.requires_grad = true;
            result.operation = Operation::Add {
                lhs: Arc::new(self.clone()),
                rhs: Arc::new(other.clone()),
            };
        }

        Ok(result)
    }

    /// Check if all elements in the tensor are real (imaginary part is zero)
    pub fn is_real(&self) -> Result<bool>
    where
        T::Real: PartialEq + num_traits::Zero,
    {
        let data = self.to_vec()?;
        Ok(data.iter().all(|&z| z.imag() == T::Real::zero()))
    }

    /// Check if any elements in the tensor are complex (imaginary part is non-zero)
    pub fn is_complex(&self) -> Result<bool>
    where
        T::Real: PartialEq + num_traits::Zero,
    {
        Ok(!self.is_real()?)
    }
}

// Note: add_op and mul_op are provided by math_ops.rs for general use

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex32;
    use torsh_core::device::DeviceType;

    type C32 = Complex32;

    #[test]
    fn test_complex_conjugate() {
        let data = vec![C32::new(1.0, 2.0), C32::new(3.0, -4.0), C32::new(-1.0, 1.0)];
        let tensor = Tensor::from_data(data, vec![3], DeviceType::Cpu).unwrap();

        let conj_tensor = tensor.complex_conj().unwrap();
        let conj_data = conj_tensor.to_vec().unwrap();

        assert_eq!(conj_data[0], C32::new(1.0, -2.0));
        assert_eq!(conj_data[1], C32::new(3.0, 4.0));
        assert_eq!(conj_data[2], C32::new(-1.0, -1.0));
    }

    #[test]
    fn test_real_imag_extraction() {
        let data = vec![C32::new(1.0, 2.0), C32::new(3.0, -4.0)];
        let tensor = Tensor::from_data(data, vec![2], DeviceType::Cpu).unwrap();

        let real_part = tensor.real().unwrap();
        let imag_part = tensor.imag().unwrap();

        assert_eq!(real_part.to_vec().unwrap(), vec![1.0, 3.0]);
        assert_eq!(imag_part.to_vec().unwrap(), vec![2.0, -4.0]);
    }

    #[test]
    fn test_magnitude_and_phase() {
        let data = vec![
            C32::new(3.0, 4.0), // |z| = 5, arg = atan(4/3)
            C32::new(1.0, 0.0), // |z| = 1, arg = 0
        ];
        let tensor = Tensor::from_data(data, vec![2], DeviceType::Cpu).unwrap();

        let magnitude = tensor.abs().unwrap();
        let phase = tensor.angle().unwrap();

        let mag_data = magnitude.to_vec().unwrap();
        let phase_data = phase.to_vec().unwrap();

        assert!((mag_data[0] - 5.0).abs() < 1e-6);
        assert!((mag_data[1] - 1.0).abs() < 1e-6);
        assert!((phase_data[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_complex_from_components() {
        let real_data = vec![1.0f32, 2.0, 3.0];
        let imag_data = vec![4.0f32, 5.0, 6.0];

        let real_tensor = Tensor::from_data(real_data, vec![3], DeviceType::Cpu).unwrap();
        let imag_tensor = Tensor::from_data(imag_data, vec![3], DeviceType::Cpu).unwrap();

        let complex_tensor = Tensor::<C32>::complex(&real_tensor, &imag_tensor).unwrap();
        let result_data = complex_tensor.to_vec().unwrap();

        assert_eq!(result_data[0], C32::new(1.0, 4.0));
        assert_eq!(result_data[1], C32::new(2.0, 5.0));
        assert_eq!(result_data[2], C32::new(3.0, 6.0));
    }

    #[test]
    fn test_complex_arithmetic() {
        let a_data = vec![C32::new(1.0, 2.0), C32::new(3.0, 4.0)];
        let b_data = vec![C32::new(2.0, 1.0), C32::new(1.0, -1.0)];

        let a = Tensor::from_data(a_data, vec![2], DeviceType::Cpu).unwrap();
        let b = Tensor::from_data(b_data, vec![2], DeviceType::Cpu).unwrap();

        // Test complex addition
        let sum = a.complex_add(&b).unwrap();
        let sum_data = sum.to_vec().unwrap();
        assert_eq!(sum_data[0], C32::new(3.0, 3.0));
        assert_eq!(sum_data[1], C32::new(4.0, 3.0));

        // Test complex multiplication
        let product = a.complex_mul(&b).unwrap();
        let prod_data = product.to_vec().unwrap();
        // (1+2i)(2+1i) = 2 + i + 4i + 2i² = 2 + 5i - 2 = 0 + 5i
        assert_eq!(prod_data[0], C32::new(0.0, 5.0));
        // (3+4i)(1-1i) = 3 - 3i + 4i - 4i² = 3 + i + 4 = 7 + i
        assert_eq!(prod_data[1], C32::new(7.0, 1.0));
    }

    #[test]
    fn test_polar_construction() {
        let mag_data = vec![1.0f32, 2.0];
        let phase_data = vec![0.0f32, std::f32::consts::PI / 2.0];

        let mag_tensor = Tensor::from_data(mag_data, vec![2], DeviceType::Cpu).unwrap();
        let phase_tensor = Tensor::from_data(phase_data, vec![2], DeviceType::Cpu).unwrap();

        let complex_tensor = Tensor::<C32>::polar(&mag_tensor, &phase_tensor).unwrap();
        let result_data = complex_tensor.to_vec().unwrap();

        // First element: 1 * (cos(0) + i*sin(0)) = 1 + 0i
        assert!((result_data[0].re - 1.0).abs() < 1e-6);
        assert!((result_data[0].im - 0.0).abs() < 1e-6);

        // Second element: 2 * (cos(π/2) + i*sin(π/2)) = 2 * (0 + i) = 0 + 2i
        assert!((result_data[1].re - 0.0).abs() < 1e-6);
        assert!((result_data[1].im - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_is_real_complex() {
        let real_data = vec![C32::new(1.0, 0.0), C32::new(2.0, 0.0)];
        let complex_data = vec![C32::new(1.0, 1.0), C32::new(2.0, 0.0)];

        let real_tensor = Tensor::from_data(real_data, vec![2], DeviceType::Cpu).unwrap();
        let complex_tensor = Tensor::from_data(complex_data, vec![2], DeviceType::Cpu).unwrap();

        assert!(real_tensor.is_real().unwrap());
        assert!(!real_tensor.is_complex().unwrap());

        assert!(!complex_tensor.is_real().unwrap());
        assert!(complex_tensor.is_complex().unwrap());
    }

    #[test]
    fn test_shape_mismatch_errors() {
        let a = Tensor::<C32>::zeros(&[2], DeviceType::Cpu).unwrap();
        let b = Tensor::<C32>::zeros(&[3], DeviceType::Cpu).unwrap();

        assert!(a.complex_add(&b).is_err());
        assert!(a.complex_mul(&b).is_err());

        let real_2 = Tensor::<f32>::zeros(&[2], DeviceType::Cpu).unwrap();
        let imag_3 = Tensor::<f32>::zeros(&[3], DeviceType::Cpu).unwrap();

        assert!(Tensor::<C32>::complex(&real_2, &imag_3).is_err());
    }
}
