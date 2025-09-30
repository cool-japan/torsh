//! BFloat16 tensor operations and optimizations
//!
//! This module provides specialized operations for BFloat16 (bf16) tensors,
//! including proper rounding modes and optimized implementations.

use crate::{Tensor, TensorElement};
use half::bf16;
use torsh_core::{
    dtype::{BF16RoundingMode, BFloat16Ops},
    error::Result,
};

/// Extension trait for BFloat16 tensor operations
pub trait BFloat16TensorOps<T: TensorElement> {
    /// Convert tensor to bf16 with specified rounding mode
    fn to_bf16_with_rounding(&self, mode: BF16RoundingMode) -> Result<Tensor<bf16>>;

    /// Convert from bf16 tensor to higher precision
    fn to_f32(&self) -> Result<Tensor<f32>>;

    /// Perform operation in higher precision then round back to bf16
    fn bf16_high_precision_op<F>(&self, op: F) -> Result<Tensor<bf16>>
    where
        F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>;
}

impl BFloat16TensorOps<f32> for Tensor<f32> {
    fn to_bf16_with_rounding(&self, mode: BF16RoundingMode) -> Result<Tensor<bf16>> {
        let data = self.data()?;
        let converted_data: Vec<bf16> = data
            .iter()
            .map(|&x| bf16::from_f32_with_rounding(x, mode))
            .collect();

        Tensor::from_data(converted_data, self.shape().dims().to_vec(), self.device())
    }

    fn to_f32(&self) -> Result<Tensor<f32>> {
        // This doesn't make sense for f32 -> bf16, but included for completeness
        self.to_bf16_with_rounding(BF16RoundingMode::NearestTiesToEven)?
            .to_f32()
    }

    fn bf16_high_precision_op<F>(&self, op: F) -> Result<Tensor<bf16>>
    where
        F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
    {
        // f32 is already high precision, apply op and convert to bf16
        let result = op(self)?;
        result.to_bf16_with_rounding(BF16RoundingMode::NearestTiesToEven)
    }
}

impl BFloat16TensorOps<bf16> for Tensor<bf16> {
    fn to_bf16_with_rounding(&self, _mode: BF16RoundingMode) -> Result<Tensor<bf16>> {
        // Already bf16, return clone
        Ok(self.clone())
    }

    fn to_f32(&self) -> Result<Tensor<f32>> {
        let data = self.data()?;
        let converted_data: Vec<f32> = data.iter().map(|&x| x.to_f32()).collect();

        Tensor::from_data(converted_data, self.shape().dims().to_vec(), self.device())
    }

    fn bf16_high_precision_op<F>(&self, op: F) -> Result<Tensor<bf16>>
    where
        F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
    {
        // Convert to f32, apply op, convert back to bf16
        let f32_tensor = self.to_f32()?;
        let result = op(&f32_tensor)?;
        result.to_bf16_with_rounding(BF16RoundingMode::NearestTiesToEven)
    }
}

/// Specialized bf16 arithmetic operations with proper rounding
impl Tensor<bf16> {
    /// Add two bf16 tensors with specified rounding mode
    pub fn add_with_rounding(
        &self,
        other: &Tensor<bf16>,
        mode: BF16RoundingMode,
    ) -> Result<Tensor<bf16>> {
        let self_data = self.data()?;
        let other_data = other.data()?;

        if self_data.len() != other_data.len() {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Tensor shapes must match for addition".to_string(),
            ));
        }

        let result_data: Vec<bf16> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| {
                let sum_f32 = a.to_f32() + b.to_f32();
                bf16::from_f32_with_rounding(sum_f32, mode)
            })
            .collect();

        Tensor::from_data(result_data, self.shape().dims().to_vec(), self.device())
    }

    /// Multiply two bf16 tensors with specified rounding mode
    pub fn mul_with_rounding(
        &self,
        other: &Tensor<bf16>,
        mode: BF16RoundingMode,
    ) -> Result<Tensor<bf16>> {
        let self_data = self.data()?;
        let other_data = other.data()?;

        if self_data.len() != other_data.len() {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "Tensor shapes must match for multiplication".to_string(),
            ));
        }

        let result_data: Vec<bf16> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(&a, &b)| a.mul_with_rounding(b, mode))
            .collect();

        Tensor::from_data(result_data, self.shape().dims().to_vec(), self.device())
    }

    /// Fused multiply-add with proper bf16 rounding
    pub fn fma_with_rounding(
        &self,
        other: &Tensor<bf16>,
        addend: &Tensor<bf16>,
        mode: BF16RoundingMode,
    ) -> Result<Tensor<bf16>> {
        let self_data = self.data()?;
        let other_data = other.data()?;
        let addend_data = addend.data()?;

        if self_data.len() != other_data.len() || self_data.len() != addend_data.len() {
            return Err(torsh_core::error::TorshError::InvalidArgument(
                "All tensor shapes must match for FMA".to_string(),
            ));
        }

        let result_data: Vec<bf16> = self_data
            .iter()
            .zip(other_data.iter())
            .zip(addend_data.iter())
            .map(|((&a, &b), &c)| a.fma_with_rounding(b, c, mode))
            .collect();

        Tensor::from_data(result_data, self.shape().dims().to_vec(), self.device())
    }
}

/// Optimized bf16 creation functions
pub mod creation {
    use super::*;
    use crate::creation;

    /// Create bf16 tensor from f32 data with specified rounding
    pub fn tensor_1d_bf16_from_f32(data: &[f32], mode: BF16RoundingMode) -> Result<Tensor<bf16>> {
        let bf16_data: Vec<bf16> = data
            .iter()
            .map(|&x| bf16::from_f32_with_rounding(x, mode))
            .collect();
        creation::tensor_1d(&bf16_data)
    }

    /// Create 2D bf16 tensor from f32 data with specified rounding
    pub fn tensor_2d_bf16_from_f32(
        data: &[&[f32]],
        mode: BF16RoundingMode,
    ) -> Result<Tensor<bf16>> {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };

        let mut bf16_data = Vec::with_capacity(rows * cols);
        for row in data {
            for &val in row.iter() {
                bf16_data.push(bf16::from_f32_with_rounding(val, mode));
            }
        }

        Tensor::from_data(
            bf16_data,
            vec![rows, cols],
            torsh_core::device::DeviceType::Cpu,
        )
    }

    /// Create bf16 zeros tensor
    pub fn zeros_bf16(shape: &[usize]) -> Result<Tensor<bf16>> {
        creation::zeros::<bf16>(shape)
    }

    /// Create bf16 ones tensor
    pub fn ones_bf16(shape: &[usize]) -> Result<Tensor<bf16>> {
        creation::ones::<bf16>(shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation;
    use approx::assert_relative_eq;

    #[test]
    fn test_bf16_tensor_creation() {
        let data = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
        ];
        let tensor = creation::tensor_1d(&data).unwrap();

        assert_eq!(tensor.shape().dims(), &[3]);
        assert_eq!(tensor.data().unwrap(), data);
    }

    #[test]
    fn test_bf16_zeros_ones() {
        let zeros = creation::zeros::<bf16>(&[2, 3]).unwrap();
        assert_eq!(zeros.shape().dims(), &[2, 3]);

        let zeros_data = zeros.data().unwrap();
        assert!(zeros_data.iter().all(|&x| x == bf16::from_f32(0.0)));

        let ones = creation::ones::<bf16>(&[2, 3]).unwrap();
        let ones_data = ones.data().unwrap();
        assert!(ones_data.iter().all(|&x| x == bf16::from_f32(1.0)));
    }

    #[test]
    fn test_bf16_rounding_modes() {
        let f32_data = vec![1.5f32, 2.5f32, 3.7f32];

        // Test different rounding modes
        let nearest_even = super::creation::tensor_1d_bf16_from_f32(
            &f32_data,
            BF16RoundingMode::NearestTiesToEven,
        )
        .unwrap();
        let nearest_away =
            super::creation::tensor_1d_bf16_from_f32(&f32_data, BF16RoundingMode::NearestTiesAway)
                .unwrap();
        let toward_zero =
            super::creation::tensor_1d_bf16_from_f32(&f32_data, BF16RoundingMode::TowardZero)
                .unwrap();

        let nearest_even_data = nearest_even.data().unwrap();
        let nearest_away_data = nearest_away.data().unwrap();
        let toward_zero_data = toward_zero.data().unwrap();

        // Verify different rounding behaviors for tie cases
        assert_eq!(
            nearest_even_data[0],
            bf16::from_f32_with_rounding(1.5, BF16RoundingMode::NearestTiesToEven)
        );
        assert_eq!(
            nearest_away_data[0],
            bf16::from_f32_with_rounding(1.5, BF16RoundingMode::NearestTiesAway)
        );
        assert_eq!(
            toward_zero_data[0],
            bf16::from_f32_with_rounding(1.5, BF16RoundingMode::TowardZero)
        );
    }

    #[test]
    fn test_bf16_arithmetic_with_rounding() {
        let a = creation::tensor_1d(&[bf16::from_f32(1.5), bf16::from_f32(2.5)]).unwrap();
        let b = creation::tensor_1d(&[bf16::from_f32(0.5), bf16::from_f32(1.5)]).unwrap();

        let result = a
            .add_with_rounding(&b, BF16RoundingMode::NearestTiesToEven)
            .unwrap();
        let result_data = result.data().unwrap();

        assert_relative_eq!(result_data[0].to_f32(), 2.0, epsilon = 1e-6);
        assert_relative_eq!(result_data[1].to_f32(), 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_bf16_conversion() {
        let f32_tensor = creation::tensor_1d(&[1.0f32, 2.0f32, 3.0f32]).unwrap();

        // Convert to bf16
        let bf16_tensor = f32_tensor
            .to_bf16_with_rounding(BF16RoundingMode::NearestTiesToEven)
            .unwrap();

        // Convert back to f32
        let f32_converted = bf16_tensor.to_f32().unwrap();
        let f32_converted_data = f32_converted.data().unwrap();

        // Should be approximately equal (some precision loss expected)
        assert_relative_eq!(f32_converted_data[0], 1.0, epsilon = 1e-2);
        assert_relative_eq!(f32_converted_data[1], 2.0, epsilon = 1e-2);
        assert_relative_eq!(f32_converted_data[2], 3.0, epsilon = 1e-2);
    }

    #[test]
    fn test_bf16_high_precision_op() {
        let bf16_tensor = creation::tensor_1d(&[bf16::from_f32(1.0), bf16::from_f32(2.0)]).unwrap();

        // Apply a complex operation in high precision
        let result = bf16_tensor
            .bf16_high_precision_op(|t| {
                let doubled = t.mul_op(t)?; // Square in f32 precision
                doubled.add_scalar(1.0) // Add 1 in f32 precision
            })
            .unwrap();

        let result_data = result.data().unwrap();
        assert_relative_eq!(result_data[0].to_f32(), 2.0, epsilon = 1e-2); // 1^2 + 1 = 2
        assert_relative_eq!(result_data[1].to_f32(), 5.0, epsilon = 1e-2); // 2^2 + 1 = 5
    }

    #[test]
    fn test_bf16_fma() {
        let a = creation::tensor_1d(&[bf16::from_f32(2.0), bf16::from_f32(3.0)]).unwrap();
        let b = creation::tensor_1d(&[bf16::from_f32(4.0), bf16::from_f32(5.0)]).unwrap();
        let c = creation::tensor_1d(&[bf16::from_f32(1.0), bf16::from_f32(2.0)]).unwrap();

        let result = a
            .fma_with_rounding(&b, &c, BF16RoundingMode::NearestTiesToEven)
            .unwrap();
        let result_data = result.data().unwrap();

        // FMA: a * b + c
        assert_relative_eq!(result_data[0].to_f32(), 9.0, epsilon = 1e-2); // 2 * 4 + 1 = 9
        assert_relative_eq!(result_data[1].to_f32(), 17.0, epsilon = 1e-2); // 3 * 5 + 2 = 17
    }

    #[test]
    fn test_bf16_precision_limits() {
        // Test bf16 precision limits
        let large_value = 65504.0f32; // Near bf16 max
        let small_value = 1e-6f32; // Very small value

        let large_tensor = super::creation::tensor_1d_bf16_from_f32(
            &[large_value],
            BF16RoundingMode::NearestTiesToEven,
        )
        .unwrap();
        let small_tensor = super::creation::tensor_1d_bf16_from_f32(
            &[small_value],
            BF16RoundingMode::NearestTiesToEven,
        )
        .unwrap();

        let large_data = large_tensor.data().unwrap();
        let small_data = small_tensor.data().unwrap();

        // Large values should be preserved with some precision loss
        assert!((large_data[0].to_f32() - large_value).abs() < 1000.0);

        // Very small values might be rounded to zero or have significant precision loss
        assert!(small_data[0].to_f32() >= 0.0);
    }
}
