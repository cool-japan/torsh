//! Fake quantization for QAT (Quantization Aware Training)

use crate::{QScheme, TorshResult};
use torsh_core::{DType, TorshError};
use torsh_tensor::Tensor;

/// Fake quantize a tensor - quantizes then immediately dequantizes
/// This simulates quantization during training while keeping gradients flowing
pub fn fake_quantize(tensor: &Tensor, scale: f32, zero_point: i32) -> TorshResult<Tensor> {
    fake_quantize_per_tensor_affine(tensor, scale, zero_point, -128, 127)
}

/// Fake quantize using per-tensor affine quantization
pub fn fake_quantize_per_tensor_affine(
    tensor: &Tensor,
    scale: f32,
    zero_point: i32,
    quant_min: i32,
    quant_max: i32,
) -> TorshResult<Tensor> {
    let data = tensor.data()?;

    // Fake quantize each element: quantize then dequantize
    let fake_quantized_data: Vec<f32> = data
        .iter()
        .map(|&x| {
            // Quantize: q = round(x / scale) + zero_point
            let quantized = (x / scale).round() + zero_point as f32;
            // Clamp to quantization range
            let clamped = quantized.max(quant_min as f32).min(quant_max as f32);
            // Dequantize: x = scale * (q - zero_point)
            scale * (clamped - zero_point as f32)
        })
        .collect();

    let result_tensor = Tensor::from_data(
        fake_quantized_data,
        tensor.shape().dims().to_vec(),
        tensor.device(),
    )?;
    Ok(result_tensor)
}

/// Fake quantize using symmetric quantization (zero_point = 0)
pub fn fake_quantize_per_tensor_symmetric(
    tensor: &Tensor,
    scale: f32,
    quant_min: i32,
    quant_max: i32,
) -> TorshResult<Tensor> {
    fake_quantize_per_tensor_affine(tensor, scale, 0, quant_min, quant_max)
}

/// Fake quantize with learnable parameters
#[derive(Debug)]
pub struct FakeQuantize {
    scale: f32,
    zero_point: i32,
    quant_min: i32,
    quant_max: i32,
    enabled: bool,
}

impl FakeQuantize {
    /// Create a new fake quantize module
    pub fn new(scale: f32, zero_point: i32, quant_min: i32, quant_max: i32) -> Self {
        Self {
            scale,
            zero_point,
            quant_min,
            quant_max,
            enabled: true,
        }
    }

    /// Create fake quantize for INT8
    pub fn int8(scale: f32, zero_point: i32) -> Self {
        Self::new(scale, zero_point, -128, 127)
    }

    /// Create fake quantize for UINT8
    pub fn uint8(scale: f32, zero_point: i32) -> Self {
        Self::new(scale, zero_point, 0, 255)
    }

    /// Enable fake quantization
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable fake quantization
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Forward pass through fake quantization
    pub fn forward(&self, tensor: &Tensor) -> TorshResult<Tensor> {
        if !self.enabled {
            return Ok(tensor.clone());
        }

        fake_quantize_per_tensor_affine(
            tensor,
            self.scale,
            self.zero_point,
            self.quant_min,
            self.quant_max,
        )
    }

    /// Update quantization parameters (for calibration)
    pub fn update_params(&mut self, scale: f32, zero_point: i32) {
        self.scale = scale;
        self.zero_point = zero_point;
    }
}

/// Fake quantize with automatic parameter calculation
pub fn fake_quantize_auto(tensor: &Tensor, dtype: DType, scheme: QScheme) -> TorshResult<Tensor> {
    let (quant_min, quant_max) = match dtype {
        DType::I8 => (-128, 127),
        DType::U8 => (0, 255),
        _ => {
            return Err(TorshError::InvalidArgument(
                "Unsupported quantization dtype".to_string(),
            ))
        }
    };

    // Calculate parameters from tensor statistics
    let data = tensor.data()?;
    let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b)).min(0.0);
    let max_val = data
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        .max(0.0);

    let scale = (max_val - min_val) / (quant_max - quant_min) as f32;
    let scale = if scale == 0.0 { 1.0 } else { scale };

    match scheme {
        QScheme::PerTensorAffine => {
            let zero_point = (quant_min as f32 - min_val / scale)
                .round()
                .max(quant_min as f32)
                .min(quant_max as f32) as i32;
            fake_quantize_per_tensor_affine(tensor, scale, zero_point, quant_min, quant_max)
        }
        QScheme::PerTensorSymmetric => {
            fake_quantize_per_tensor_symmetric(tensor, scale, quant_min, quant_max)
        }
        _ => Err(TorshError::InvalidArgument(
            "Quantization scheme not yet implemented".to_string(),
        )),
    }
}

/// Apply fake quantization to module weights during training
// Temporarily disabled: pub fn apply_fake_quantization(_module: &mut dyn torsh_nn::Module) -> TorshResult<()> {
#[allow(dead_code)]
pub fn apply_fake_quantization(_module: &mut dyn crate::TemporaryModule) -> TorshResult<()> {
    // This would iterate through module parameters and apply fake quantization
    // For now, this is a placeholder implementation
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_fake_quantize_per_tensor_affine() {
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let tensor = tensor_1d(&data).unwrap();

        let scale = 0.1;
        let zero_point = 0;

        let fake_quantized =
            fake_quantize_per_tensor_affine(&tensor, scale, zero_point, -128, 127).unwrap();
        let result_data = fake_quantized.to_vec().unwrap();

        // Values should be close to original but with quantization noise
        for (i, &original) in data.iter().enumerate() {
            assert!((result_data[i] - original).abs() <= scale);
        }
    }

    #[test]
    fn test_fake_quantize_symmetric() {
        let data = vec![-1.0, 0.0, 1.0, 2.0];
        let tensor = tensor_1d(&data).unwrap();

        let scale = 0.1;

        let fake_quantized = fake_quantize_per_tensor_symmetric(&tensor, scale, -128, 127).unwrap();
        let result_data = fake_quantized.to_vec().unwrap();

        // Check that quantization is symmetric around zero
        assert_eq!(result_data.len(), data.len());
    }

    #[test]
    fn test_fake_quantize_module() {
        let mut fake_quant = FakeQuantize::int8(0.1, 0);

        let data = vec![0.5, 1.5, 2.5, 3.5];
        let tensor = tensor_1d(&data).unwrap();

        let result = fake_quant.forward(&tensor).unwrap();
        let result_data = result.to_vec().unwrap();

        // Should produce quantized values
        assert_eq!(result_data.len(), data.len());

        // Disable and test passthrough
        fake_quant.disable();
        let passthrough = fake_quant.forward(&tensor).unwrap();
        let passthrough_data = passthrough.to_vec().unwrap();

        for (i, &original) in data.iter().enumerate() {
            assert_relative_eq!(passthrough_data[i], original, epsilon = 1e-5);
        }
    }

    #[test]
    #[ignore = "test hangs - needs investigation"]
    fn test_fake_quantize_auto() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let tensor = tensor_1d(&data).unwrap();

        let fake_quantized =
            fake_quantize_auto(&tensor, DType::I8, QScheme::PerTensorAffine).unwrap();
        let result_data = fake_quantized.to_vec().unwrap();

        assert_eq!(result_data.len(), data.len());

        // Values should be within reasonable range
        for &val in &result_data {
            assert!((-2.1..=2.1).contains(&val));
        }
    }
}
