//! Type conversion operations for tensors
//!
//! This module provides comprehensive type conversion operations:
//! - Numeric type conversions (f32, f64, i32, i64, etc.)
//! - Boolean conversions
//! - Complex number conversions
//! - Device transfers (CPU, CUDA, etc.)
//! - Type promotion for mixed operations

use crate::{FloatElement, Tensor, TensorElement};
use torsh_core::error::{Result, TorshError};
use torsh_core::device::DeviceType;

/// Type conversion operations
impl<T: TensorElement> Tensor<T> {
    /// Convert tensor to f32 type
    pub fn to_f32(&self) -> Result<Tensor<f32>> {
        let data = self.data()?;
        let converted_data: std::result::Result<Vec<f32>, _> = data
            .iter()
            .map(|&x| {
                <T as TensorElement>::to_f64(&x)
                    .and_then(|f| if f.is_finite() && f >= f32::MIN as f64 && f <= f32::MAX as f64 {
                        Some(f as f32)
                    } else {
                        None
                    })
                    .ok_or_else(|| TorshError::InvalidArgument(
                        format!("Cannot convert value to f32: {}", f64::from_bits(<T as TensorElement>::to_f64(&x).unwrap_or(0.0) as u64))
                    ))
            })
            .collect();

        let converted_data = converted_data?;
        Tensor::from_data(
            converted_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Convert tensor to f64 type
    pub fn to_f64(&self) -> Result<Tensor<f64>> {
        let data = self.data()?;
        let converted_data: std::result::Result<Vec<f64>, _> = data
            .iter()
            .map(|&x| {
                <T as TensorElement>::to_f64(&x)
                    .ok_or_else(|| TorshError::InvalidArgument(
                        "Cannot convert value to f64".to_string()
                    ))
            })
            .collect();

        let converted_data = converted_data?;
        Tensor::from_data(
            converted_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Convert tensor to i32 type
    pub fn to_i32(&self) -> Result<Tensor<i32>> {
        let data = self.data()?;
        let converted_data: std::result::Result<Vec<i32>, _> = data
            .iter()
            .map(|&x| {
                <T as TensorElement>::to_f64(&x)
                    .and_then(|f| {
                        if f.is_finite() && f >= i32::MIN as f64 && f <= i32::MAX as f64 {
                            Some(f.round() as i32)
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| TorshError::InvalidArgument(
                        "Cannot convert value to i32: value out of range or not finite".to_string()
                    ))
            })
            .collect();

        let converted_data = converted_data?;
        Tensor::from_data(
            converted_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Convert tensor to i64 type
    pub fn to_i64(&self) -> Result<Tensor<i64>> {
        let data = self.data()?;
        let converted_data: std::result::Result<Vec<i64>, _> = data
            .iter()
            .map(|&x| {
                <T as TensorElement>::to_f64(&x)
                    .and_then(|f| {
                        if f.is_finite() && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
                            Some(f.round() as i64)
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| TorshError::InvalidArgument(
                        "Cannot convert value to i64: value out of range or not finite".to_string()
                    ))
            })
            .collect();

        let converted_data = converted_data?;
        Tensor::from_data(
            converted_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Convert tensor to bool type (non-zero values become true)
    pub fn to_bool(&self) -> Result<Tensor<bool>> {
        let data = self.data()?;
        let converted_data: Vec<bool> = data
            .iter()
            .map(|&x| {
                <T as TensorElement>::to_f64(&x)
                    .map(|f| f != 0.0)
                    .unwrap_or(false)
            })
            .collect();

        Tensor::from_data(
            converted_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }


    /// Move tensor to CPU device
    pub fn to_cpu(&self) -> Result<Self> {
        self.to_device(DeviceType::Cpu)
    }

    /// Move tensor to CUDA device (if available)
    pub fn to_cuda(&self, device_id: usize) -> Result<Self> {
        self.to_device(DeviceType::Cuda(device_id))
    }

    /// Generic type conversion to any TensorElement type
    pub fn to_tensor<U: TensorElement>(&self) -> Result<Tensor<U>> {
        let data = self.data()?;
        let converted_data: std::result::Result<Vec<U>, _> = data
            .iter()
            .map(|&x| {
                <T as TensorElement>::to_f64(&x)
                    .and_then(|f| U::from_f64(f))
                    .ok_or_else(|| TorshError::InvalidArgument(
                        format!("Cannot convert value to target type")
                    ))
            })
            .collect();

        let converted_data = converted_data?;
        Tensor::from_data(
            converted_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }
}

/// Float-specific conversions
impl<T: FloatElement> Tensor<T> {
    /// Convert to complex tensor (imaginary part is zero)
    pub fn to_complex32(&self) -> Result<Tensor<torsh_core::dtype::Complex32>> {
        let data = self.data()?;
        let complex_data: Vec<torsh_core::dtype::Complex32> = data
            .iter()
            .map(|&x| {
                let real = <T as TensorElement>::to_f64(&x).unwrap_or(0.0) as f32;
                torsh_core::dtype::Complex32::new(real, 0.0)
            })
            .collect();

        Tensor::from_data(
            complex_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Convert to complex64 tensor (imaginary part is zero)
    pub fn to_complex64(&self) -> Result<Tensor<torsh_core::dtype::Complex64>> {
        let data = self.data()?;
        let complex_data: Vec<torsh_core::dtype::Complex64> = data
            .iter()
            .map(|&x| {
                let real = <T as TensorElement>::to_f64(&x).unwrap_or(0.0);
                torsh_core::dtype::Complex64::new(real, 0.0)
            })
            .collect();

        Tensor::from_data(
            complex_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }
}

/// Complex tensor conversions
impl Tensor<torsh_core::dtype::Complex32> {
    /// Extract real part as f32 tensor
    pub fn real_part(&self) -> Result<Tensor<f32>> {
        let data = self.data()?;
        let real_data: Vec<f32> = data.iter().map(|x| x.re).collect();

        Tensor::from_data(
            real_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Extract imaginary part as f32 tensor
    pub fn imag_part(&self) -> Result<Tensor<f32>> {
        let data = self.data()?;
        let imag_data: Vec<f32> = data.iter().map(|x| x.im).collect();

        Tensor::from_data(
            imag_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Convert to magnitude (absolute value) tensor
    pub fn magnitude(&self) -> Result<Tensor<f32>> {
        let data = self.data()?;
        let mag_data: Vec<f32> = data.iter().map(|x| x.norm()).collect();

        Tensor::from_data(
            mag_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Convert to phase (angle) tensor
    pub fn phase(&self) -> Result<Tensor<f32>> {
        let data = self.data()?;
        let phase_data: Vec<f32> = data.iter().map(|x| x.arg()).collect();

        Tensor::from_data(
            phase_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }
}

/// Complex64 tensor conversions
impl Tensor<torsh_core::dtype::Complex64> {
    /// Extract real part as f64 tensor
    pub fn real_part(&self) -> Result<Tensor<f64>> {
        let data = self.data()?;
        let real_data: Vec<f64> = data.iter().map(|x| x.re).collect();

        Tensor::from_data(
            real_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Extract imaginary part as f64 tensor
    pub fn imag_part(&self) -> Result<Tensor<f64>> {
        let data = self.data()?;
        let imag_data: Vec<f64> = data.iter().map(|x| x.im).collect();

        Tensor::from_data(
            imag_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Convert to magnitude (absolute value) tensor
    pub fn magnitude(&self) -> Result<Tensor<f64>> {
        let data = self.data()?;
        let mag_data: Vec<f64> = data.iter().map(|x| x.norm()).collect();

        Tensor::from_data(
            mag_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }

    /// Convert to phase (angle) tensor
    pub fn phase(&self) -> Result<Tensor<f64>> {
        let data = self.data()?;
        let phase_data: Vec<f64> = data.iter().map(|x| x.arg()).collect();

        Tensor::from_data(
            phase_data,
            self.shape().dims().to_vec(),
            self.device,
        )
    }
}

/// Boolean tensor conversions
impl Tensor<bool> {
}

/// Type promotion utilities
pub fn promote_types<T1: TensorElement, T2: TensorElement>(
    tensor1: &Tensor<T1>,
    tensor2: &Tensor<T2>,
) -> Result<(Tensor<f64>, Tensor<f64>)> {
    // For simplicity, promote both tensors to f64
    // In a more sophisticated implementation, we would use type hierarchy
    let promoted1 = tensor1.to_f64()?;
    let promoted2 = tensor2.to_f64()?;
    Ok((promoted1, promoted2))
}

/// Create a complex tensor from real and imaginary parts
pub fn complex_from_parts<T: FloatElement>(
    real: &Tensor<T>,
    imag: &Tensor<T>,
) -> Result<Tensor<torsh_core::dtype::Complex64>> {
    if real.shape() != imag.shape() {
        return Err(TorshError::ShapeMismatch {
            expected: real.shape().dims().to_vec(),
            got: imag.shape().dims().to_vec(),
        });
    }

    let real_data = real.data()?;
    let imag_data = imag.data()?;

    let complex_data: Vec<torsh_core::dtype::Complex64> = real_data
        .iter()
        .zip(imag_data.iter())
        .map(|(&r, &i)| {
            let real_f64 = <T as TensorElement>::to_f64(&r).unwrap_or(0.0);
            let imag_f64 = <T as TensorElement>::to_f64(&i).unwrap_or(0.0);
            torsh_core::dtype::Complex64::new(real_f64, imag_f64)
        })
        .collect();

    Tensor::from_data(
        complex_data,
        real.shape().dims().to_vec(),
        real.device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_to_f32() {
        let tensor = Tensor::from_data(vec![1i32, 2, 3, 4], vec![4], DeviceType::Cpu).unwrap();
        let f32_tensor = tensor.to_f32().unwrap();
        let data = f32_tensor.data().unwrap();
        assert_eq!(data.as_slice(), &[1.0f32, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_to_i32() {
        let tensor = Tensor::from_data(vec![1.7f32, 2.3, -3.9, 4.0], vec![4], DeviceType::Cpu).unwrap();
        let i32_tensor = tensor.to_i32().unwrap();
        let data = i32_tensor.data().unwrap();
        assert_eq!(data.as_slice(), &[2i32, 2, -4, 4]); // Rounded values
    }

    #[test]
    fn test_to_bool() {
        let tensor = Tensor::from_data(vec![0.0f32, 1.0, -2.5, 0.0], vec![4], DeviceType::Cpu).unwrap();
        let bool_tensor = tensor.to_bool().unwrap();
        let data = bool_tensor.data().unwrap();
        assert_eq!(data.as_slice(), &[false, true, true, false]);
    }

    #[test]
    fn test_bool_to_f32() {
        let tensor = Tensor::from_data(vec![true, false, true, false], vec![4], DeviceType::Cpu).unwrap();
        let f32_tensor = tensor.to_f32().unwrap();
        let data = f32_tensor.data().unwrap();
        assert_eq!(data.as_slice(), &[1.0f32, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_to_device_cpu() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let cpu_tensor = tensor.to_cpu().unwrap();
        assert_eq!(cpu_tensor.device, DeviceType::Cpu);
    }

    #[test]
    fn test_to_complex32() {
        let tensor = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu).unwrap();
        let complex_tensor = tensor.to_complex32().unwrap();
        let data = complex_tensor.data().unwrap();

        assert_eq!(data[0].re, 1.0);
        assert_eq!(data[0].im, 0.0);
        assert_eq!(data[1].re, 2.0);
        assert_eq!(data[1].im, 0.0);
    }

    #[test]
    fn test_complex_real_imag_parts() {
        let real_data = vec![1.0f32, 2.0, 3.0];
        let imag_data = vec![4.0f32, 5.0, 6.0];

        let complex_data: Vec<torsh_core::dtype::Complex32> = real_data
            .iter()
            .zip(imag_data.iter())
            .map(|(&r, &i)| torsh_core::dtype::Complex32::new(r, i))
            .collect();

        let complex_tensor = Tensor::from_data(complex_data, vec![3], DeviceType::Cpu).unwrap();

        let real_part = complex_tensor.real_part().unwrap();
        let imag_part = complex_tensor.imag_part().unwrap();

        let real_data_result = real_part.data().unwrap();
        let imag_data_result = imag_part.data().unwrap();

        assert_eq!(real_data_result.as_slice(), &[1.0f32, 2.0, 3.0]);
        assert_eq!(imag_data_result.as_slice(), &[4.0f32, 5.0, 6.0]);
    }

    #[test]
    fn test_complex_magnitude() {
        let complex_data = vec![
            torsh_core::dtype::Complex32::new(3.0, 4.0), // magnitude = 5.0
            torsh_core::dtype::Complex32::new(0.0, 1.0), // magnitude = 1.0
        ];

        let complex_tensor = Tensor::from_data(complex_data, vec![2], DeviceType::Cpu).unwrap();
        let magnitude = complex_tensor.magnitude().unwrap();
        let mag_data = magnitude.data().unwrap();

        assert!((mag_data[0] - 5.0).abs() < 1e-6);
        assert!((mag_data[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_complex_from_parts() {
        let real = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).unwrap();
        let imag = Tensor::from_data(vec![3.0f32, 4.0], vec![2], DeviceType::Cpu).unwrap();

        let complex_tensor = complex_from_parts(&real, &imag).unwrap();
        let data = complex_tensor.data().unwrap();

        assert_eq!(data[0].re, 1.0);
        assert_eq!(data[0].im, 3.0);
        assert_eq!(data[1].re, 2.0);
        assert_eq!(data[1].im, 4.0);
    }

    #[test]
    fn test_promote_types() {
        let tensor1 = Tensor::from_data(vec![1i32, 2], vec![2], DeviceType::Cpu).unwrap();
        let tensor2 = Tensor::from_data(vec![3.5f32, 4.5], vec![2], DeviceType::Cpu).unwrap();

        let (promoted1, promoted2) = promote_types(&tensor1, &tensor2).unwrap();

        let data1 = promoted1.data().unwrap();
        let data2 = promoted2.data().unwrap();

        assert_eq!(data1.as_slice(), &[1.0f64, 2.0]);
        assert_eq!(data2.as_slice(), &[3.5f64, 4.5]);
    }

    #[test]
    fn test_conversion_error_handling() {
        // Test overflow handling
        let large_tensor = Tensor::from_data(
            vec![f64::MAX, f64::MIN],
            vec![2],
            DeviceType::Cpu
        ).unwrap();

        // This should fail because f64::MAX cannot be represented as f32
        assert!(large_tensor.to_f32().is_err());
    }
}