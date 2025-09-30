//! Different quantization schemes and their implementations

use crate::quantization::{CalibrationMethod, QuantizationParams, QuantizationScheme};
use torsh_core::{
    dtype::DType,
    error::{Result, TorshError},
};
use torsh_tensor::Tensor;

/// Post-training quantization (PTQ) implementation
pub struct PostTrainingQuantization {
    scheme: QuantizationScheme,
    target_dtype: DType,
    calibration_method: CalibrationMethod,
}

impl PostTrainingQuantization {
    /// Create a new PTQ quantizer
    pub fn new(
        scheme: QuantizationScheme,
        target_dtype: DType,
        calibration_method: CalibrationMethod,
    ) -> Self {
        Self {
            scheme,
            target_dtype,
            calibration_method,
        }
    }

    /// Apply post-training quantization to a tensor
    pub fn quantize_tensor(&self, tensor: &Tensor) -> Result<(Tensor, QuantizationParams)> {
        let data = tensor.to_vec()?;
        let params = self.calculate_params(&data)?;
        let quantized = params.quantize(tensor)?;
        Ok((quantized, params))
    }

    /// Calculate quantization parameters for the given data
    fn calculate_params(&self, data: &[f32]) -> Result<QuantizationParams> {
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        match self.scheme {
            QuantizationScheme::Symmetric => {
                let scale = self.calculate_symmetric_scale(min_val, max_val)?;
                Ok(QuantizationParams::symmetric(
                    scale,
                    DType::F32,
                    self.target_dtype,
                ))
            }
            QuantizationScheme::Asymmetric => {
                let (scale, zero_point) = self.calculate_asymmetric_params(min_val, max_val)?;
                Ok(QuantizationParams::asymmetric(
                    scale,
                    zero_point,
                    DType::F32,
                    self.target_dtype,
                ))
            }
            QuantizationScheme::Dynamic => {
                // Dynamic quantization uses runtime statistics
                self.dynamic_quantization_params(data)
            }
            QuantizationScheme::KLDivergence => self.kl_divergence_params(data),
            QuantizationScheme::Percentile(percentile) => self.percentile_params(data, percentile),
        }
    }

    /// Calculate symmetric quantization scale
    fn calculate_symmetric_scale(&self, min_val: f32, max_val: f32) -> Result<f32> {
        let max_abs = max_val.abs().max(min_val.abs());
        let max_quant = match self.target_dtype {
            DType::I8 => 127.0,
            DType::U8 => 127.0, // Use symmetric range for U8 too
            DType::I16 => 32767.0,
            _ => {
                return Err(TorshError::InvalidArgument(
                    "Unsupported quantization dtype".to_string(),
                ))
            }
        };

        if max_abs == 0.0 {
            return Ok(1.0); // Avoid division by zero
        }

        Ok(max_abs / max_quant)
    }

    /// Calculate asymmetric quantization parameters
    fn calculate_asymmetric_params(&self, min_val: f32, max_val: f32) -> Result<(f32, i32)> {
        let (qmin, qmax) = match self.target_dtype {
            DType::U8 => (0.0, 255.0),
            DType::I8 => (-128.0, 127.0),
            DType::I16 => (-32768.0, 32767.0),
            _ => {
                return Err(TorshError::InvalidArgument(
                    "Unsupported quantization dtype".to_string(),
                ))
            }
        };

        let scale = (max_val - min_val) / (qmax - qmin);
        let zero_point = (qmin - min_val / scale).round() as i32;

        Ok((scale, zero_point))
    }

    /// Dynamic quantization parameters
    fn dynamic_quantization_params(&self, data: &[f32]) -> Result<QuantizationParams> {
        // For dynamic quantization, we calculate parameters based on runtime statistics
        // This is similar to symmetric but might use different calibration methods
        use crate::quantization::calibration::calculate_optimal_scale;

        let scale = calculate_optimal_scale(data, &self.calibration_method, self.target_dtype)?;
        Ok(QuantizationParams::symmetric(
            scale,
            DType::F32,
            self.target_dtype,
        ))
    }

    /// KL divergence-based quantization parameters
    fn kl_divergence_params(&self, data: &[f32]) -> Result<QuantizationParams> {
        let scale = self.find_optimal_scale_kl(data)?;
        Ok(QuantizationParams::symmetric(
            scale,
            DType::F32,
            self.target_dtype,
        ))
    }

    /// Percentile-based quantization parameters
    fn percentile_params(&self, data: &[f32], percentile: f32) -> Result<QuantizationParams> {
        let mut sorted_abs: Vec<f32> = data.iter().map(|&x| x.abs()).collect();
        sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = ((percentile / 100.0) * sorted_abs.len() as f32) as usize;
        let max_val = sorted_abs
            .get(index.min(sorted_abs.len() - 1))
            .copied()
            .unwrap_or(0.0);

        let scale = self.calculate_symmetric_scale(-max_val, max_val)?;
        Ok(QuantizationParams::symmetric(
            scale,
            DType::F32,
            self.target_dtype,
        ))
    }

    /// Find optimal scale using KL divergence
    fn find_optimal_scale_kl(&self, data: &[f32]) -> Result<f32> {
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
        let max_quant = match self.target_dtype {
            DType::I8 => 127.0,
            DType::U8 => 255.0,
            _ => {
                return Err(TorshError::InvalidArgument(
                    "Unsupported dtype for KL divergence".to_string(),
                ))
            }
        };

        let base_scale = max_val / max_quant;
        let mut best_scale = base_scale;
        let mut best_divergence = f32::INFINITY;

        // Try different scales and find the one with minimum KL divergence
        for multiplier in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0] {
            let scale = base_scale * multiplier;
            let divergence = self.calculate_kl_divergence(data, scale)?;

            if divergence < best_divergence {
                best_divergence = divergence;
                best_scale = scale;
            }
        }

        Ok(best_scale)
    }

    /// Calculate KL divergence for a given scale
    fn calculate_kl_divergence(&self, data: &[f32], scale: f32) -> Result<f32> {
        let max_quant = match self.target_dtype {
            DType::I8 => 127,
            DType::U8 => 255,
            _ => return Err(TorshError::InvalidArgument("Unsupported dtype".to_string())),
        };

        // Create histogram of original data
        let num_bins = 256;
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let bin_width = (max_val - min_val) / num_bins as f32;

        let mut original_hist = vec![0u32; num_bins];
        let mut quantized_hist = vec![0u32; num_bins];

        for &value in data {
            // Original histogram
            let bin_idx = ((value - min_val) / bin_width) as usize;
            let bin_idx = bin_idx.min(num_bins - 1);
            original_hist[bin_idx] += 1;

            // Quantized histogram
            let quantized = ((value / scale).round() as i32).clamp(-max_quant, max_quant);
            let dequantized = quantized as f32 * scale;
            let quant_bin_idx = ((dequantized - min_val) / bin_width) as usize;
            let quant_bin_idx = quant_bin_idx.min(num_bins - 1);
            quantized_hist[quant_bin_idx] += 1;
        }

        // Calculate KL divergence
        let total_count = data.len() as f32;
        let mut kl_div = 0.0;

        for i in 0..num_bins {
            let p = original_hist[i] as f32 / total_count;
            let q = quantized_hist[i] as f32 / total_count;

            if p > 0.0 && q > 0.0 {
                kl_div += p * (p / q).ln();
            }
        }

        Ok(kl_div)
    }
}

/// Quantization-Aware Training (QAT) implementation
pub struct QuantizationAwareTraining {
    fake_quantize: bool,
    #[allow(dead_code)]
    scheme: QuantizationScheme,
    target_dtype: DType,
}

impl QuantizationAwareTraining {
    /// Create a new QAT quantizer
    pub fn new(scheme: QuantizationScheme, target_dtype: DType) -> Self {
        Self {
            fake_quantize: true,
            scheme,
            target_dtype,
        }
    }

    /// Apply fake quantization during training
    pub fn fake_quantize_tensor(
        &self,
        tensor: &Tensor,
        params: &QuantizationParams,
    ) -> Result<Tensor> {
        if !self.fake_quantize {
            return Ok(tensor.clone());
        }

        // Simulate quantization during forward pass
        let quantized = params.quantize(tensor)?;
        let dequantized = params.dequantize(&quantized)?;

        Ok(dequantized)
    }

    /// Enable/disable fake quantization
    pub fn set_fake_quantize(&mut self, enabled: bool) {
        self.fake_quantize = enabled;
    }

    /// Create learnable quantization parameters
    pub fn learnable_params(
        &self,
        initial_scale: f32,
        initial_zero_point: i32,
    ) -> LearnableQuantParams {
        LearnableQuantParams::new(initial_scale, initial_zero_point, self.target_dtype)
    }
}

/// Learnable quantization parameters for QAT
#[derive(Debug, Clone)]
pub struct LearnableQuantParams {
    scale: Tensor,
    zero_point: Tensor,
    target_dtype: DType,
}

impl LearnableQuantParams {
    /// Create new learnable parameters
    pub fn new(initial_scale: f32, initial_zero_point: i32, target_dtype: DType) -> Self {
        let scale = Tensor::from_data(
            vec![initial_scale],
            vec![1],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap_or_else(|_| {
            Tensor::zeros(&[1], torsh_core::device::DeviceType::Cpu)
                .expect("Failed to create fallback zeros tensor")
        });
        let zero_point = Tensor::from_data(
            vec![initial_zero_point as f32],
            vec![1],
            torsh_core::device::DeviceType::Cpu,
        )
        .unwrap_or_else(|_| {
            Tensor::zeros(&[1], torsh_core::device::DeviceType::Cpu)
                .expect("Failed to create fallback zeros tensor")
        });

        Self {
            scale,
            zero_point,
            target_dtype,
        }
    }

    /// Get current quantization parameters
    pub fn current_params(&self) -> Result<QuantizationParams> {
        let scale_val = self.scale.to_vec()?[0];
        let zero_point_val = self.zero_point.to_vec()?[0] as i32;

        Ok(QuantizationParams::asymmetric(
            scale_val,
            zero_point_val,
            DType::F32,
            self.target_dtype,
        ))
    }

    /// Update parameters during training
    pub fn update_scale(&mut self, new_scale: f32) -> Result<()> {
        self.scale = Tensor::from_data(
            vec![new_scale],
            vec![1],
            torsh_core::device::DeviceType::Cpu,
        )?;
        Ok(())
    }

    /// Update zero point during training
    pub fn update_zero_point(&mut self, new_zero_point: i32) -> Result<()> {
        self.zero_point = Tensor::from_data(
            vec![new_zero_point as f32],
            vec![1],
            torsh_core::device::DeviceType::Cpu,
        )?;
        Ok(())
    }
}

/// Block-wise quantization for large models
pub struct BlockWiseQuantization {
    block_size: usize,
    scheme: QuantizationScheme,
    target_dtype: DType,
}

impl BlockWiseQuantization {
    /// Create a new block-wise quantizer
    pub fn new(block_size: usize, scheme: QuantizationScheme, target_dtype: DType) -> Self {
        Self {
            block_size,
            scheme,
            target_dtype,
        }
    }

    /// Quantize tensor in blocks
    pub fn quantize_blocks(&self, tensor: &Tensor) -> Result<(Tensor, Vec<QuantizationParams>)> {
        let data = tensor.to_vec()?;
        let mut quantized_data = Vec::new();
        let mut block_params = Vec::new();

        for chunk in data.chunks(self.block_size) {
            let ptq = PostTrainingQuantization::new(
                self.scheme.clone(),
                self.target_dtype,
                CalibrationMethod::MinMax,
            );

            let block_tensor = Tensor::from_data(
                chunk.to_vec(),
                vec![chunk.len()],
                torsh_core::device::DeviceType::Cpu,
            )?;
            let (quantized_block, params) = ptq.quantize_tensor(&block_tensor)?;

            let block_data: Vec<u8> = match self.target_dtype {
                DType::I8 => quantized_block
                    .to_vec()?
                    .into_iter()
                    .map(|x: f32| x as i8 as u8)
                    .collect(),
                DType::U8 => quantized_block
                    .to_vec()?
                    .into_iter()
                    .map(|x: f32| x as u8)
                    .collect(),
                _ => {
                    return Err(TorshError::InvalidArgument(
                        "Unsupported block quantization dtype".to_string(),
                    ))
                }
            };

            quantized_data.extend(block_data);
            block_params.push(params);
        }

        let float_data: Vec<f32> = quantized_data.into_iter().map(|x| x as f32).collect();
        let quantized_tensor = Tensor::from_data(
            float_data,
            tensor.shape().dims().to_vec(),
            torsh_core::device::DeviceType::Cpu,
        )?;
        Ok((quantized_tensor, block_params))
    }
}

/// Mixed precision quantization
pub struct MixedPrecisionQuantization {
    layer_configs: Vec<LayerQuantConfig>,
}

/// Configuration for per-layer quantization
#[derive(Debug, Clone)]
pub struct LayerQuantConfig {
    pub layer_name: String,
    pub weight_dtype: DType,
    pub activation_dtype: DType,
    pub scheme: QuantizationScheme,
    pub per_channel: bool,
}

impl MixedPrecisionQuantization {
    /// Create a new mixed precision quantizer
    pub fn new(layer_configs: Vec<LayerQuantConfig>) -> Self {
        Self { layer_configs }
    }

    /// Get quantization config for a specific layer
    pub fn get_layer_config(&self, layer_name: &str) -> Option<&LayerQuantConfig> {
        self.layer_configs
            .iter()
            .find(|config| config.layer_name == layer_name)
    }

    /// Apply layer-specific quantization
    pub fn quantize_layer(
        &self,
        layer_name: &str,
        weights: &Tensor,
        activations: &Tensor,
    ) -> Result<(Tensor, Tensor, QuantizationParams, QuantizationParams)> {
        let config = self.get_layer_config(layer_name).ok_or_else(|| {
            TorshError::InvalidArgument(format!(
                "No quantization config found for layer: {}",
                layer_name
            ))
        })?;

        // Quantize weights
        let weight_ptq = PostTrainingQuantization::new(
            config.scheme.clone(),
            config.weight_dtype,
            CalibrationMethod::MinMax,
        );
        let (quantized_weights, weight_params) = weight_ptq.quantize_tensor(weights)?;

        // Quantize activations
        let activation_ptq = PostTrainingQuantization::new(
            config.scheme.clone(),
            config.activation_dtype,
            CalibrationMethod::MinMax,
        );
        let (quantized_activations, activation_params) =
            activation_ptq.quantize_tensor(activations)?;

        Ok((
            quantized_weights,
            quantized_activations,
            weight_params,
            activation_params,
        ))
    }
}

/// Utility functions for quantization schemes
pub mod utils {
    use super::*;

    /// Create a simple INT8 symmetric quantization scheme
    pub fn int8_symmetric() -> PostTrainingQuantization {
        PostTrainingQuantization::new(
            QuantizationScheme::Symmetric,
            DType::I8,
            CalibrationMethod::MinMax,
        )
    }

    /// Create a UINT8 asymmetric quantization scheme
    pub fn uint8_asymmetric() -> PostTrainingQuantization {
        PostTrainingQuantization::new(
            QuantizationScheme::Asymmetric,
            DType::U8,
            CalibrationMethod::MinMax,
        )
    }

    /// Create a dynamic quantization scheme
    pub fn dynamic_int8() -> PostTrainingQuantization {
        PostTrainingQuantization::new(
            QuantizationScheme::Dynamic,
            DType::I8,
            CalibrationMethod::Entropy,
        )
    }

    /// Create a KL divergence-based quantization scheme
    pub fn kl_divergence_int8() -> PostTrainingQuantization {
        PostTrainingQuantization::new(
            QuantizationScheme::KLDivergence,
            DType::I8,
            CalibrationMethod::Entropy,
        )
    }

    /// Create a percentile-based quantization scheme
    pub fn percentile_int8(percentile: f32) -> PostTrainingQuantization {
        PostTrainingQuantization::new(
            QuantizationScheme::Percentile(percentile),
            DType::I8,
            CalibrationMethod::MinMax,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_post_training_quantization() {
        let data = vec![1.0, -1.0, 0.5, -0.5, 2.0, -2.0];
        let tensor = Tensor::from_data(data, vec![6], torsh_core::device::DeviceType::Cpu).unwrap();

        let ptq = PostTrainingQuantization::new(
            QuantizationScheme::Symmetric,
            DType::I8,
            CalibrationMethod::MinMax,
        );

        let result = ptq.quantize_tensor(&tensor);
        assert!(result.is_ok());

        let (_quantized, params) = result.unwrap();
        // Note: quantized tensor will have F32 dtype due to our implementation
        assert!(params.scale > 0.0);
    }

    #[test]
    fn test_qat_fake_quantization() {
        let tensor_data = vec![0.1f32; 10];
        let tensor =
            Tensor::from_data(tensor_data, vec![10], torsh_core::device::DeviceType::Cpu).unwrap();
        let params = QuantizationParams::symmetric(0.1, DType::F32, DType::I8);

        let qat = QuantizationAwareTraining::new(QuantizationScheme::Symmetric, DType::I8);
        let fake_quantized = qat.fake_quantize_tensor(&tensor, &params).unwrap();

        assert_eq!(fake_quantized.shape().dims(), tensor.shape().dims());
        // Note: tensor dtype will be F32 due to our implementation
    }

    #[test]
    fn test_block_wise_quantization() {
        let data: Vec<f32> = (0..100).map(|x| x as f32 / 10.0).collect();
        let tensor =
            Tensor::from_data(data, vec![100], torsh_core::device::DeviceType::Cpu).unwrap();

        let block_quant = BlockWiseQuantization::new(25, QuantizationScheme::Symmetric, DType::I8);

        let result = block_quant.quantize_blocks(&tensor);
        assert!(result.is_ok());

        let (quantized, params) = result.unwrap();
        assert_eq!(quantized.shape().dims(), tensor.shape().dims());
        assert_eq!(params.len(), 4); // 100 / 25 = 4 blocks
    }

    #[test]
    fn test_mixed_precision_quantization() {
        let layer_configs = vec![
            LayerQuantConfig {
                layer_name: "conv1".to_string(),
                weight_dtype: DType::I8,
                activation_dtype: DType::I8,
                scheme: QuantizationScheme::Symmetric,
                per_channel: true,
            },
            LayerQuantConfig {
                layer_name: "fc1".to_string(),
                weight_dtype: DType::I8,
                activation_dtype: DType::U8,
                scheme: QuantizationScheme::Asymmetric,
                per_channel: false,
            },
        ];

        let mixed_prec = MixedPrecisionQuantization::new(layer_configs);
        let config = mixed_prec.get_layer_config("conv1");
        assert!(config.is_some());
        assert_eq!(config.unwrap().weight_dtype, DType::I8);
    }
}
