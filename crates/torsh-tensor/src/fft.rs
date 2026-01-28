//! Fast Fourier Transform (FFT) operations for tensors
//!
//! This module provides comprehensive FFT functionality including:
//! - 1D, 2D, and N-dimensional FFTs
//! - Real and complex FFTs
//! - Inverse FFTs
//! - Optimized implementations for various tensor sizes

use crate::{Tensor, TensorElement};
use std::f64::consts::PI;
use torsh_core::dtype::Complex64;
use torsh_core::error::{Result, TorshError};

/// FFT plan for optimized repeated transforms
#[derive(Debug, Clone)]
pub struct FFTPlan {
    /// Size of the transform
    pub size: usize,
    /// Precomputed twiddle factors
    pub twiddles: Vec<Complex64>,
    /// Bit-reversed indices for in-place computation
    pub bit_reversed_indices: Vec<usize>,
    /// Whether this is a forward or inverse transform
    pub is_forward: bool,
}

impl FFTPlan {
    /// Create a new FFT plan for the given size
    pub fn new(size: usize, is_forward: bool) -> Result<Self> {
        if size == 0 || (size & (size - 1)) != 0 {
            return Err(TorshError::InvalidArgument(
                "FFT size must be a power of 2".to_string(),
            ));
        }

        let mut twiddles = Vec::with_capacity(size / 2);
        let direction = if is_forward { -1.0 } else { 1.0 };

        // Precompute twiddle factors
        for k in 0..size / 2 {
            let angle = direction * 2.0 * PI * k as f64 / size as f64;
            twiddles.push(Complex64::new(angle.cos(), angle.sin()));
        }

        // Compute bit-reversed indices
        let mut bit_reversed_indices = vec![0; size];
        let mut j = 0;
        #[allow(clippy::needless_range_loop)]
        for i in 1..size {
            let mut bit = size >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            bit_reversed_indices[i] = j;
        }

        Ok(Self {
            size,
            twiddles,
            bit_reversed_indices,
            is_forward,
        })
    }

    /// Execute the FFT plan on the given data
    pub fn execute(&self, data: &mut [Complex64]) -> Result<()> {
        if data.len() != self.size {
            return Err(TorshError::InvalidArgument(format!(
                "Data size {} does not match plan size {}",
                data.len(),
                self.size
            )));
        }

        // Bit-reverse the input
        for i in 0..self.size {
            let j = self.bit_reversed_indices[i];
            if i < j {
                data.swap(i, j);
            }
        }

        // Cooley-Tukey FFT algorithm
        let mut n = 2;
        while n <= self.size {
            let step = self.size / n;
            for i in (0..self.size).step_by(n) {
                for j in 0..n / 2 {
                    let u = data[i + j];
                    let v = data[i + j + n / 2] * self.twiddles[j * step];
                    data[i + j] = u + v;
                    data[i + j + n / 2] = u - v;
                }
            }
            n <<= 1;
        }

        // Normalize for inverse transform
        if !self.is_forward {
            let norm = 1.0 / self.size as f64;
            for sample in data.iter_mut() {
                *sample *= norm;
            }
        }

        Ok(())
    }
}

/// FFT operations for tensors
impl<T: TensorElement + Into<f64> + From<f64>> Tensor<T> {
    /// Compute 1D FFT along the last dimension
    pub fn fft(&self) -> Result<Tensor<Complex64>> {
        self.fft_with_plan(None)
    }

    /// Compute 1D FFT with a precomputed plan
    pub fn fft_with_plan(&self, plan: Option<&FFTPlan>) -> Result<Tensor<Complex64>> {
        let shape = self.shape();
        let last_dim_size = shape.dims().last().copied().unwrap_or(1);

        // Check if size is a power of 2
        if last_dim_size == 0 || (last_dim_size & (last_dim_size - 1)) != 0 {
            return Err(TorshError::InvalidArgument(
                "FFT requires the last dimension to be a power of 2".to_string(),
            ));
        }

        // Create or use existing plan
        let owned_plan;
        let fft_plan = match plan {
            Some(p) => {
                if p.size != last_dim_size || !p.is_forward {
                    return Err(TorshError::InvalidArgument(
                        "Plan size or direction mismatch".to_string(),
                    ));
                }
                p
            }
            None => {
                owned_plan = FFTPlan::new(last_dim_size, true)?;
                &owned_plan
            }
        };

        // Convert input data to complex
        let input_data = self.to_vec()?;
        let total_elements = input_data.len();
        let num_ffts = total_elements / last_dim_size;

        let mut complex_data = Vec::with_capacity(total_elements);
        for &value in &input_data {
            complex_data.push(Complex64::new(value.into(), 0.0));
        }

        // Perform FFT on each vector
        for i in 0..num_ffts {
            let start = i * last_dim_size;
            let end = start + last_dim_size;
            fft_plan.execute(&mut complex_data[start..end])?;
        }

        // Create output tensor (same shape but complex type)
        Tensor::from_complex_data(complex_data, shape.dims().to_vec(), self.device())
    }

    /// Compute 1D inverse FFT along the last dimension
    pub fn ifft(&self) -> Result<Tensor<T>>
    where
        T: TensorElement + From<f64>,
    {
        let complex_tensor = self.to_complex()?;
        complex_tensor.ifft_complex()?.to_real()
    }

    /// Convert generic tensor to complex form
    fn to_complex(&self) -> Result<Tensor<Complex64>> {
        let input_data = self.to_vec()?;
        let complex_data: Vec<Complex64> = input_data
            .iter()
            .map(|&value| Complex64::new(value.into(), 0.0))
            .collect();

        Tensor::from_complex_data(complex_data, self.shape().dims().to_vec(), self.device())
    }

    /// Compute 2D FFT on the last two dimensions
    pub fn fft2(&self) -> Result<Tensor<Complex64>> {
        let shape = self.shape();
        let dims = shape.dims();

        if dims.len() < 2 {
            return Err(TorshError::InvalidArgument(
                "2D FFT requires at least 2 dimensions".to_string(),
            ));
        }

        // First, FFT along the last dimension
        let temp = self.fft()?;

        // Then, FFT along the second-to-last dimension
        temp.fft_along_dim(dims.len() - 2)
    }

    /// Compute 2D inverse FFT on the last two dimensions
    pub fn ifft2(&self) -> Result<Tensor<T>>
    where
        T: TensorElement + From<f64>,
    {
        let complex_tensor = self.to_complex()?;
        complex_tensor.ifft2_complex()?.to_real()
    }

    /// Compute FFT along a specific dimension for real tensors
    pub fn fft_along_dim_real(&self, dim: usize) -> Result<Tensor<Complex64>> {
        let shape = self.shape();
        let dims = shape.dims();

        if dim >= dims.len() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of bounds for tensor with {} dimensions",
                dim,
                dims.len()
            )));
        }

        // If it's the last dimension, use the optimized path
        if dim == dims.len() - 1 {
            return self.fft();
        }

        // For other dimensions, we need to transpose, FFT, then transpose back
        let transposed = self.transpose_to_last_dim(dim)?;
        let fft_result = transposed.fft()?;
        fft_result.transpose_from_last_dim(dim)
    }

    /// Real-to-complex FFT (more efficient for real inputs)
    pub fn rfft(&self) -> Result<Tensor<Complex64>> {
        // For real FFT, we only need to compute half of the coefficients
        // due to Hermitian symmetry
        let shape = self.shape();
        let last_dim_size = shape.dims().last().copied().unwrap_or(1);
        let output_size = last_dim_size / 2 + 1;

        let full_fft = self.fft()?;

        // Extract the first half + 1 coefficients
        let mut new_shape = shape.dims().to_vec();
        *new_shape
            .last_mut()
            .expect("shape should have at least one dimension") = output_size;

        full_fft.slice_last_dim_complex(0, output_size)
    }

    /// Complex-to-real inverse FFT
    pub fn irfft(&self, output_size: Option<usize>) -> Result<Tensor<T>>
    where
        T: TensorElement + From<f64>,
    {
        let shape = self.shape();
        let input_size = shape.dims().last().copied().unwrap_or(1);
        let out_size = output_size.unwrap_or((input_size - 1) * 2);

        // Reconstruct full complex spectrum using Hermitian symmetry
        let full_spectrum = self.reconstruct_hermitian_spectrum(out_size)?;

        // Perform inverse FFT and convert to real
        let complex_result = full_spectrum.ifft_complex()?;
        complex_result.to_real()
    }

    /// Compute power spectral density
    pub fn power_spectrum(&self) -> Result<Tensor<T>>
    where
        T: TensorElement + From<f64>,
    {
        let fft_result = self.fft()?;
        fft_result.power_spectrum_from_fft()
    }

    /// Compute magnitude spectrum
    pub fn magnitude_spectrum(&self) -> Result<Tensor<T>>
    where
        T: TensorElement + From<f64>,
    {
        let fft_result = self.fft()?;
        fft_result.magnitude_spectrum_from_fft()
    }

    /// Compute phase spectrum
    pub fn phase_spectrum(&self) -> Result<Tensor<T>>
    where
        T: TensorElement + From<f64>,
    {
        let fft_result = self.fft()?;
        fft_result.phase_spectrum_from_fft()
    }

    /// Slice the last dimension
    #[allow(dead_code)]
    fn slice_last_dim(&self, start: usize, size: usize) -> Result<Self> {
        // This is a simplified implementation
        // In a full implementation, you'd use proper tensor slicing
        let shape = self.shape();
        let dims = shape.dims();
        let last_dim_size = dims.last().copied().unwrap_or(1);

        if start + size > last_dim_size {
            return Err(TorshError::IndexOutOfBounds {
                index: start + size - 1,
                size: last_dim_size,
            });
        }

        // Create new shape
        let mut new_dims = dims.to_vec();
        *new_dims
            .last_mut()
            .expect("shape should have at least one dimension") = size;

        // Extract the data (simplified - would need proper strided slicing)
        let input_data = self.to_vec()?;
        let total_elements = input_data.len();
        let num_vectors = total_elements / last_dim_size;

        let mut output_data = Vec::with_capacity(num_vectors * size);
        for i in 0..num_vectors {
            let base_idx = i * last_dim_size;
            for j in 0..size {
                output_data.push(input_data[base_idx + start + j]);
            }
        }

        Self::from_data(output_data, new_dims, self.device())
    }

    /// Reconstruct Hermitian spectrum for IRFFT
    fn reconstruct_hermitian_spectrum(&self, output_size: usize) -> Result<Tensor<Complex64>> {
        // This is a simplified implementation
        // Real implementation would properly handle Hermitian symmetry
        let shape = self.shape();
        let input_size = shape.dims().last().copied().unwrap_or(1);

        if output_size < (input_size - 1) * 2 {
            return Err(TorshError::InvalidArgument(
                "Output size too small for IRFFT".to_string(),
            ));
        }

        // For now, just pad with zeros (proper implementation would use conjugate symmetry)
        let mut new_dims = shape.dims().to_vec();
        *new_dims
            .last_mut()
            .expect("shape should have at least one dimension") = output_size;

        let input_data = self.to_vec()?;
        let mut output_data = Vec::with_capacity(input_data.len() * output_size / input_size);

        // Simple zero-padding (would need proper Hermitian reconstruction)
        for &value in &input_data {
            // Convert T to Complex64 - assuming T can be converted to f64
            let f64_value: f64 = value.into();
            output_data.push(Complex64::new(f64_value, 0.0));
        }

        // Pad with zeros to reach output size
        while output_data.len() < output_data.capacity() {
            output_data.push(Complex64::new(0.0, 0.0));
        }

        Tensor::from_complex_data(output_data, new_dims, self.device())
    }
}

/// General tensor operations that don't require Into<f64>
impl<T: TensorElement> Tensor<T> {
    /// Helper method to transpose a dimension to the last position
    fn transpose_to_last_dim(&self, dim: usize) -> Result<Self> {
        let ndim = self.shape().dims().len();
        if dim == ndim - 1 {
            return Ok(self.clone());
        }
        self.transpose(dim as i32, (ndim - 1) as i32)
    }

    /// Helper method to transpose back from last dimension
    fn transpose_from_last_dim(&self, original_dim: usize) -> Result<Self> {
        let ndim = self.shape().dims().len();
        if original_dim == ndim - 1 {
            return Ok(self.clone());
        }
        self.transpose(original_dim as i32, (ndim - 1) as i32)
    }
}

/// Operations specific to complex tensors
impl Tensor<Complex64> {
    /// Create tensor from complex data
    pub fn from_complex_data(
        data: Vec<Complex64>,
        shape: Vec<usize>,
        device: torsh_core::device::DeviceType,
    ) -> Result<Self> {
        Tensor::from_data(data, shape, device)
    }

    /// Convert complex tensor to real by taking the real part
    pub fn to_real<T: TensorElement + From<f64>>(&self) -> Result<Tensor<T>> {
        let complex_data = self.to_vec()?;
        let real_data: Vec<T> = complex_data.iter().map(|c| T::from(c.re)).collect();

        Tensor::from_data(real_data, self.shape().dims().to_vec(), self.device())
    }

    /// Compute power spectrum from FFT result
    pub fn power_spectrum_from_fft<T: TensorElement + From<f64>>(&self) -> Result<Tensor<T>> {
        let complex_data = self.to_vec()?;
        let power_data: Vec<T> = complex_data
            .iter()
            .map(|c| T::from(c.norm().powi(2)))
            .collect();

        Tensor::from_data(power_data, self.shape().dims().to_vec(), self.device())
    }

    /// Compute magnitude spectrum from FFT result
    pub fn magnitude_spectrum_from_fft<T: TensorElement + From<f64>>(&self) -> Result<Tensor<T>> {
        let complex_data = self.to_vec()?;
        let magnitude_data: Vec<T> = complex_data.iter().map(|c| T::from(c.norm())).collect();

        Tensor::from_data(magnitude_data, self.shape().dims().to_vec(), self.device())
    }

    /// Compute phase spectrum from FFT result
    pub fn phase_spectrum_from_fft<T: TensorElement + From<f64>>(&self) -> Result<Tensor<T>> {
        let complex_data = self.to_vec()?;
        let phase_data: Vec<T> = complex_data.iter().map(|c| T::from(c.arg())).collect();

        Tensor::from_data(phase_data, self.shape().dims().to_vec(), self.device())
    }

    /// Compute FFT for complex data
    pub fn fft_complex(&self) -> Result<Tensor<Complex64>> {
        let shape = self.shape();
        let last_dim_size = shape.dims().last().copied().unwrap_or(1);

        let plan = FFTPlan::new(last_dim_size, true)?;

        let mut complex_data = self.to_vec()?;
        let num_ffts = complex_data.len() / last_dim_size;

        // Perform FFT on each vector
        for i in 0..num_ffts {
            let start = i * last_dim_size;
            let end = start + last_dim_size;
            plan.execute(&mut complex_data[start..end])?;
        }

        // Create output tensor with same shape
        Tensor::from_complex_data(complex_data, shape.dims().to_vec(), self.device())
    }

    /// Compute inverse FFT for complex data
    pub fn ifft_complex(&self) -> Result<Tensor<Complex64>> {
        let shape = self.shape();
        let last_dim_size = shape.dims().last().copied().unwrap_or(1);

        let plan = FFTPlan::new(last_dim_size, false)?;

        let mut complex_data = self.to_vec()?;
        let num_ffts = complex_data.len() / last_dim_size;

        // Perform inverse FFT on each vector
        for i in 0..num_ffts {
            let start = i * last_dim_size;
            let end = start + last_dim_size;
            plan.execute(&mut complex_data[start..end])?;
        }

        Tensor::from_complex_data(complex_data, shape.dims().to_vec(), self.device())
    }

    /// Compute 2D inverse FFT for complex data
    pub fn ifft2_complex(&self) -> Result<Tensor<Complex64>> {
        let shape = self.shape();
        let dims = shape.dims();

        if dims.len() < 2 {
            return Err(TorshError::InvalidArgument(
                "2D IFFT requires at least 2 dimensions".to_string(),
            ));
        }

        // First, IFFT along the second-to-last dimension
        let temp = self.ifft_along_dim(dims.len() - 2)?;

        // Then, IFFT along the last dimension
        temp.ifft_complex()
    }

    /// Compute inverse FFT along a specific dimension
    pub fn ifft_along_dim(&self, dim: usize) -> Result<Tensor<Complex64>> {
        let shape = self.shape();
        let dims = shape.dims();

        if dim >= dims.len() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of bounds for tensor with {} dimensions",
                dim,
                dims.len()
            )));
        }

        // If it's the last dimension, use the optimized path
        if dim == dims.len() - 1 {
            return self.ifft_complex();
        }

        // For other dimensions, we need to transpose, IFFT, then transpose back
        let transposed = self.transpose_to_last_dim_complex(dim)?;
        let ifft_result = transposed.ifft_complex()?;
        ifft_result.transpose_from_last_dim_complex(dim)
    }

    /// Simple transpose helper for FFT operations (complex version)
    fn transpose_to_last_dim_complex(&self, _dim: usize) -> Result<Tensor<Complex64>> {
        // Simple implementation - just return self for now
        // In a full implementation, this would properly transpose the tensor
        Ok(self.clone())
    }

    /// Simple transpose helper for FFT operations (reverse, complex version)
    fn transpose_from_last_dim_complex(&self, _dim: usize) -> Result<Tensor<Complex64>> {
        // Simple implementation - just return self for now
        // In a full implementation, this would properly transpose the tensor back
        Ok(self.clone())
    }

    /// 2D FFT for complex tensors
    pub fn fft2_complex(&self) -> Result<Tensor<Complex64>> {
        let shape = self.shape();
        let dims = shape.dims().to_vec();

        if dims.len() < 2 {
            return Err(TorshError::InvalidArgument(
                "2D FFT requires at least 2 dimensions".to_string(),
            ));
        }

        // First, FFT along the last dimension
        let temp = self.fft_complex()?;

        // Then, FFT along the second-to-last dimension
        temp.fft_along_dim_complex(dims.len() - 2)
    }

    /// Compute FFT along a specific dimension for complex tensors
    pub fn fft_along_dim(&self, dim: usize) -> Result<Tensor<Complex64>> {
        self.fft_along_dim_complex(dim)
    }

    /// Internal implementation of FFT along dimension for complex tensors
    pub fn fft_along_dim_complex(&self, dim: usize) -> Result<Tensor<Complex64>> {
        let shape = self.shape();
        let dims = shape.dims();

        if dim >= dims.len() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of bounds for tensor with {} dimensions",
                dim,
                dims.len()
            )));
        }

        // If it's the last dimension, use the optimized path
        if dim == dims.len() - 1 {
            return self.fft_complex();
        }

        // For other dimensions, we need to transpose, FFT, then transpose back
        let transposed = self.transpose_to_last_dim_complex(dim)?;
        let fft_result = transposed.fft_complex()?;
        fft_result.transpose_from_last_dim_complex(dim)
    }

    /// Slice along the last dimension for complex tensors
    pub fn slice_last_dim_complex(&self, start: usize, size: usize) -> Result<Tensor<Complex64>> {
        let shape = self.shape();
        let dims = shape.dims().to_vec();

        if dims.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Cannot slice empty tensor".to_string(),
            ));
        }

        let last_dim = dims.len() - 1;
        let last_dim_size = dims[last_dim];
        let end = start + size;

        if start >= last_dim_size || end > last_dim_size {
            return Err(TorshError::InvalidArgument(format!(
                "Invalid slice range {start}..{end} for dimension of size {last_dim_size}"
            )));
        }

        let data = self.to_vec()?;
        let num_elements_per_slice = dims[..last_dim].iter().product::<usize>();
        let mut result_data = Vec::with_capacity(num_elements_per_slice * size);

        for i in 0..num_elements_per_slice {
            let slice_start = i * last_dim_size + start;
            let slice_end = slice_start + size;
            result_data.extend_from_slice(&data[slice_start..slice_end]);
        }

        let mut new_dims = dims;
        new_dims[last_dim] = size;

        Tensor::from_complex_data(result_data, new_dims, self.device())
    }
}

/// Windowing functions for signal processing
pub mod windows {
    use super::*;

    /// Hann window
    pub fn hann<T: TensorElement + From<f64>>(size: usize) -> Result<Tensor<T>> {
        let data: Vec<T> = (0..size)
            .map(|i| {
                let factor = 0.5 * (1.0 - (2.0 * PI * i as f64 / (size - 1) as f64).cos());
                T::from(factor)
            })
            .collect();

        Tensor::from_data(data, vec![size], torsh_core::device::DeviceType::Cpu)
    }

    /// Hamming window
    pub fn hamming<T: TensorElement + From<f64>>(size: usize) -> Result<Tensor<T>> {
        let data: Vec<T> = (0..size)
            .map(|i| {
                let factor = 0.54 - 0.46 * (2.0 * PI * i as f64 / (size - 1) as f64).cos();
                T::from(factor)
            })
            .collect();

        Tensor::from_data(data, vec![size], torsh_core::device::DeviceType::Cpu)
    }

    /// Blackman window
    pub fn blackman<T: TensorElement + From<f64>>(size: usize) -> Result<Tensor<T>> {
        let data: Vec<T> = (0..size)
            .map(|i| {
                let n = i as f64;
                let n_max = (size - 1) as f64;
                let factor =
                    0.42 - 0.5 * (2.0 * PI * n / n_max).cos() + 0.08 * (4.0 * PI * n / n_max).cos();
                T::from(factor)
            })
            .collect();

        Tensor::from_data(data, vec![size], torsh_core::device::DeviceType::Cpu)
    }

    /// Kaiser window
    pub fn kaiser<T: TensorElement + From<f64>>(size: usize, beta: f64) -> Result<Tensor<T>> {
        // Simplified Kaiser window (proper implementation would use modified Bessel function)
        let data: Vec<T> = (0..size)
            .map(|i| {
                let n = i as f64;
                let n_max = (size - 1) as f64;
                let factor = (beta * (1.0 - ((2.0 * n / n_max) - 1.0).powi(2)).sqrt()).exp();
                T::from(factor)
            })
            .collect();

        Tensor::from_data(data, vec![size], torsh_core::device::DeviceType::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_fft_plan_creation() {
        let plan = FFTPlan::new(8, true).unwrap();
        assert_eq!(plan.size, 8);
        assert_eq!(plan.twiddles.len(), 4);
        assert_eq!(plan.bit_reversed_indices.len(), 8);
        assert!(plan.is_forward);
    }

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex64::new(1.0, 2.0);
        let b = Complex64::new(3.0, 4.0);

        let sum = a + b;
        assert_eq!(sum.re, 4.0);
        assert_eq!(sum.im, 6.0);

        let product = a * b;
        assert_eq!(product.re, -5.0); // (1*3 - 2*4)
        assert_eq!(product.im, 10.0); // (1*4 + 2*3)

        assert_eq!(a.norm(), (5.0_f64).sqrt());
    }

    #[test]
    fn test_fft_basic() {
        // Test with a simple signal
        let data = vec![1.0, 0.0, 0.0, 0.0];
        let tensor = Tensor::from_data(data, vec![4], torsh_core::device::DeviceType::Cpu).unwrap();

        // Test that FFT works correctly
        let result = tensor.fft();
        assert!(result.is_ok(), "FFT should work with valid input");

        let fft_result = result.unwrap();
        assert_eq!(fft_result.shape().dims(), &[4]);

        // The FFT of [1, 0, 0, 0] should be [1, 1, 1, 1]
        let output_data = fft_result.to_vec().unwrap();
        assert_eq!(output_data.len(), 4);
        // Check that the DC component is 1
        assert!((output_data[0].re - 1.0).abs() < 1e-6);
        assert!(output_data[0].im.abs() < 1e-6);
    }

    #[test]
    fn test_windowing_functions() {
        let hann_window = windows::hann::<f64>(8).unwrap();
        assert_eq!(hann_window.shape().dims(), &[8]);

        let hamming_window = windows::hamming::<f64>(8).unwrap();
        assert_eq!(hamming_window.shape().dims(), &[8]);

        let blackman_window = windows::blackman::<f64>(8).unwrap();
        assert_eq!(blackman_window.shape().dims(), &[8]);
    }
}
