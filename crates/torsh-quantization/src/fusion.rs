//! Operation fusion for quantized models
//!
//! This module provides operation fusion capabilities to optimize quantized models
//! by combining multiple operations into single kernels for better performance.

use crate::{QuantConfig, TorshResult};
use std::collections::HashMap;
use torsh_core::TorshError;
use torsh_tensor::Tensor;

/// Represents a fused operation
#[derive(Debug, Clone)]
pub struct FusedOp {
    /// Name of the fused operation
    pub name: String,
    /// Component operations that are fused
    pub component_ops: Vec<String>,
    /// Input tensors required
    pub inputs: Vec<String>,
    /// Output tensors produced
    pub outputs: Vec<String>,
    /// Quantization configuration for this fused op
    pub qconfig: Option<QuantConfig>,
}

/// Fusion patterns that can be applied
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FusionPattern {
    /// Conv + BatchNorm fusion
    ConvBN,
    /// Conv + BatchNorm + ReLU fusion
    ConvBNReLU,
    /// Linear + ReLU fusion
    LinearReLU,
    /// Conv + ReLU fusion
    ConvReLU,
    /// Add + ReLU fusion
    AddReLU,
    /// Mul + Add fusion (for quantized operations)
    MulAdd,
    /// Quantize + Dequantize elimination
    QuantDequant,
}

/// Operation fusion engine
pub struct FusionEngine {
    /// Available fusion patterns
    patterns: Vec<FusionPattern>,
    /// Configuration for different fusion patterns
    fusion_configs: HashMap<FusionPattern, QuantConfig>,
}

impl FusionEngine {
    /// Create a new fusion engine
    pub fn new() -> Self {
        // More specific patterns (longer) should be checked first
        let patterns = vec![
            FusionPattern::ConvBNReLU,
            FusionPattern::ConvBN,
            FusionPattern::LinearReLU,
            FusionPattern::ConvReLU,
            FusionPattern::AddReLU,
            FusionPattern::MulAdd,
            FusionPattern::QuantDequant,
        ];

        Self {
            patterns,
            fusion_configs: HashMap::new(),
        }
    }

    /// Add a fusion pattern to the engine
    pub fn add_pattern(&mut self, pattern: FusionPattern) {
        if !self.patterns.contains(&pattern) {
            self.patterns.push(pattern);
        }
    }

    /// Set quantization config for a fusion pattern
    pub fn set_fusion_config(&mut self, pattern: FusionPattern, config: QuantConfig) {
        self.fusion_configs.insert(pattern, config);
    }

    /// Apply fusion optimizations to a model
    pub fn fuse_model(&self, ops: &[String]) -> TorshResult<Vec<FusedOp>> {
        let mut fused_ops = Vec::new();
        let mut i = 0;

        while i < ops.len() {
            let mut found_pattern = false;

            // Try to match each fusion pattern
            for pattern in &self.patterns {
                if let Some(fused_op) = self.try_fuse_pattern(pattern, &ops[i..])? {
                    fused_ops.push(fused_op.clone());
                    i += fused_op.component_ops.len();
                    found_pattern = true;
                    break;
                }
            }

            if !found_pattern {
                // No fusion pattern matched, keep original operation
                fused_ops.push(FusedOp {
                    name: ops[i].clone(),
                    component_ops: vec![ops[i].clone()],
                    inputs: vec![format!("input_{}", i)],
                    outputs: vec![format!("output_{}", i)],
                    qconfig: None,
                });
                i += 1;
            }
        }

        Ok(fused_ops)
    }

    /// Try to match a specific fusion pattern at the current position
    fn try_fuse_pattern(
        &self,
        pattern: &FusionPattern,
        ops: &[String],
    ) -> TorshResult<Option<FusedOp>> {
        match pattern {
            FusionPattern::ConvBN => self.try_fuse_conv_bn(ops),
            FusionPattern::ConvBNReLU => self.try_fuse_conv_bn_relu(ops),
            FusionPattern::LinearReLU => self.try_fuse_linear_relu(ops),
            FusionPattern::ConvReLU => self.try_fuse_conv_relu(ops),
            FusionPattern::AddReLU => self.try_fuse_add_relu(ops),
            FusionPattern::MulAdd => self.try_fuse_mul_add(ops),
            FusionPattern::QuantDequant => self.try_fuse_quant_dequant(ops),
        }
    }

    /// Try to fuse Conv + BatchNorm
    fn try_fuse_conv_bn(&self, ops: &[String]) -> TorshResult<Option<FusedOp>> {
        if ops.len() >= 2 && ops[0].contains("conv") && ops[1].contains("batch_norm") {
            let fused_op = FusedOp {
                name: "fused_conv_bn".to_string(),
                component_ops: vec![ops[0].clone(), ops[1].clone()],
                inputs: vec!["input".to_string()],
                outputs: vec!["output".to_string()],
                qconfig: self.fusion_configs.get(&FusionPattern::ConvBN).cloned(),
            };
            Ok(Some(fused_op))
        } else {
            Ok(None)
        }
    }

    /// Try to fuse Conv + BatchNorm + ReLU
    fn try_fuse_conv_bn_relu(&self, ops: &[String]) -> TorshResult<Option<FusedOp>> {
        if ops.len() >= 3
            && ops[0].contains("conv")
            && ops[1].contains("batch_norm")
            && ops[2].contains("relu")
        {
            let fused_op = FusedOp {
                name: "fused_conv_bn_relu".to_string(),
                component_ops: vec![ops[0].clone(), ops[1].clone(), ops[2].clone()],
                inputs: vec!["input".to_string()],
                outputs: vec!["output".to_string()],
                qconfig: self.fusion_configs.get(&FusionPattern::ConvBNReLU).cloned(),
            };
            Ok(Some(fused_op))
        } else {
            Ok(None)
        }
    }

    /// Try to fuse Linear + ReLU
    fn try_fuse_linear_relu(&self, ops: &[String]) -> TorshResult<Option<FusedOp>> {
        if ops.len() >= 2 && ops[0].contains("linear") && ops[1].contains("relu") {
            let fused_op = FusedOp {
                name: "fused_linear_relu".to_string(),
                component_ops: vec![ops[0].clone(), ops[1].clone()],
                inputs: vec!["input".to_string()],
                outputs: vec!["output".to_string()],
                qconfig: self.fusion_configs.get(&FusionPattern::LinearReLU).cloned(),
            };
            Ok(Some(fused_op))
        } else {
            Ok(None)
        }
    }

    /// Try to fuse Conv + ReLU
    fn try_fuse_conv_relu(&self, ops: &[String]) -> TorshResult<Option<FusedOp>> {
        if ops.len() >= 2 && ops[0].contains("conv") && ops[1].contains("relu") {
            let fused_op = FusedOp {
                name: "fused_conv_relu".to_string(),
                component_ops: vec![ops[0].clone(), ops[1].clone()],
                inputs: vec!["input".to_string()],
                outputs: vec!["output".to_string()],
                qconfig: self.fusion_configs.get(&FusionPattern::ConvReLU).cloned(),
            };
            Ok(Some(fused_op))
        } else {
            Ok(None)
        }
    }

    /// Try to fuse Add + ReLU
    fn try_fuse_add_relu(&self, ops: &[String]) -> TorshResult<Option<FusedOp>> {
        if ops.len() >= 2 && ops[0].contains("add") && ops[1].contains("relu") {
            let fused_op = FusedOp {
                name: "fused_add_relu".to_string(),
                component_ops: vec![ops[0].clone(), ops[1].clone()],
                inputs: vec!["input1".to_string(), "input2".to_string()],
                outputs: vec!["output".to_string()],
                qconfig: self.fusion_configs.get(&FusionPattern::AddReLU).cloned(),
            };
            Ok(Some(fused_op))
        } else {
            Ok(None)
        }
    }

    /// Try to fuse Mul + Add (common in quantized models)
    fn try_fuse_mul_add(&self, ops: &[String]) -> TorshResult<Option<FusedOp>> {
        if ops.len() >= 2 && ops[0].contains("mul") && ops[1].contains("add") {
            let fused_op = FusedOp {
                name: "fused_mul_add".to_string(),
                component_ops: vec![ops[0].clone(), ops[1].clone()],
                inputs: vec![
                    "input1".to_string(),
                    "input2".to_string(),
                    "input3".to_string(),
                ],
                outputs: vec!["output".to_string()],
                qconfig: self.fusion_configs.get(&FusionPattern::MulAdd).cloned(),
            };
            Ok(Some(fused_op))
        } else {
            Ok(None)
        }
    }

    /// Try to eliminate Quantize + Dequantize pairs
    fn try_fuse_quant_dequant(&self, ops: &[String]) -> TorshResult<Option<FusedOp>> {
        if ops.len() >= 2 && ops[0].contains("quantize") && ops[1].contains("dequantize") {
            // This is an identity operation that can be eliminated
            let fused_op = FusedOp {
                name: "identity".to_string(),
                component_ops: vec![ops[0].clone(), ops[1].clone()],
                inputs: vec!["input".to_string()],
                outputs: vec!["output".to_string()],
                qconfig: None, // No quantization needed for identity
            };
            Ok(Some(fused_op))
        } else {
            Ok(None)
        }
    }

    /// Execute a fused operation
    pub fn execute_fused_op(
        &self,
        fused_op: &FusedOp,
        inputs: &[Tensor],
    ) -> TorshResult<Vec<Tensor>> {
        match fused_op.name.as_str() {
            "fused_conv_bn" => self.execute_conv_bn_fusion(inputs),
            "fused_conv_bn_relu" => self.execute_conv_bn_relu_fusion(inputs),
            "fused_linear_relu" => self.execute_linear_relu_fusion(inputs),
            "fused_conv_relu" => self.execute_conv_relu_fusion(inputs),
            "fused_add_relu" => self.execute_add_relu_fusion(inputs),
            "fused_mul_add" => self.execute_mul_add_fusion(inputs),
            "identity" => Ok(inputs.to_vec()), // Identity operation
            _ => {
                // Unknown fused operation, return error
                Err(TorshError::InvalidArgument(format!(
                    "Unknown fused operation: {}",
                    fused_op.name
                )))
            }
        }
    }

    /// Execute fused Conv + BatchNorm
    fn execute_conv_bn_fusion(&self, inputs: &[Tensor]) -> TorshResult<Vec<Tensor>> {
        if inputs.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Conv+BN fusion requires at least one input".to_string(),
            ));
        }

        // Simplified implementation - in practice this would perform
        // the actual fused convolution with batch normalization
        let input = &inputs[0];

        // Simulate batch normalization effects: (x - mean) / sqrt(variance + epsilon)
        // For simplicity, we'll apply a standardization-like operation
        let data = input.data()?;
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std_dev = (variance + 1e-5).sqrt();

        let normalized_data: Vec<f32> = data.iter().map(|x| (x - mean) / std_dev).collect();

        // Create output tensor with normalized data
        let output = Tensor::from_data(
            normalized_data,
            input.shape().dims().to_vec(),
            input.device(),
        )?;

        Ok(vec![output])
    }

    /// Execute fused Conv + BatchNorm + ReLU
    fn execute_conv_bn_relu_fusion(&self, inputs: &[Tensor]) -> TorshResult<Vec<Tensor>> {
        if inputs.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Conv+BN+ReLU fusion requires at least one input".to_string(),
            ));
        }

        // Simplified implementation
        let input = &inputs[0];
        let data = input.data()?;
        let relu_data: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();

        let output = Tensor::from_data(relu_data, input.shape().dims().to_vec(), input.device())?;

        Ok(vec![output])
    }

    /// Execute fused Linear + ReLU
    fn execute_linear_relu_fusion(&self, inputs: &[Tensor]) -> TorshResult<Vec<Tensor>> {
        if inputs.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Linear+ReLU fusion requires at least one input".to_string(),
            ));
        }

        // Simplified implementation
        let input = &inputs[0];
        let data = input.data()?;
        let relu_data: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();

        let output = Tensor::from_data(relu_data, input.shape().dims().to_vec(), input.device())?;

        Ok(vec![output])
    }

    /// Execute fused Conv + ReLU
    fn execute_conv_relu_fusion(&self, inputs: &[Tensor]) -> TorshResult<Vec<Tensor>> {
        if inputs.is_empty() {
            return Err(TorshError::InvalidArgument(
                "Conv+ReLU fusion requires at least one input".to_string(),
            ));
        }

        // Simplified implementation
        let input = &inputs[0];
        let data = input.data()?;
        let relu_data: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();

        let output = Tensor::from_data(relu_data, input.shape().dims().to_vec(), input.device())?;

        Ok(vec![output])
    }

    /// Execute fused Add + ReLU
    fn execute_add_relu_fusion(&self, inputs: &[Tensor]) -> TorshResult<Vec<Tensor>> {
        if inputs.len() < 2 {
            return Err(TorshError::InvalidArgument(
                "Add+ReLU fusion requires at least two inputs".to_string(),
            ));
        }

        let input1 = &inputs[0];
        let input2 = &inputs[1];

        if input1.shape() != input2.shape() {
            return Err(TorshError::InvalidArgument(
                "Input tensors must have the same shape for Add+ReLU fusion".to_string(),
            ));
        }

        let data1 = input1.data()?;
        let data2 = input2.data()?;

        let add_relu_data: Vec<f32> = data1
            .iter()
            .zip(data2.iter())
            .map(|(&x1, &x2)| (x1 + x2).max(0.0))
            .collect();

        let output = Tensor::from_data(
            add_relu_data,
            input1.shape().dims().to_vec(),
            input1.device(),
        )?;

        Ok(vec![output])
    }

    /// Execute fused Mul + Add
    fn execute_mul_add_fusion(&self, inputs: &[Tensor]) -> TorshResult<Vec<Tensor>> {
        if inputs.len() < 3 {
            return Err(TorshError::InvalidArgument(
                "Mul+Add fusion requires at least three inputs".to_string(),
            ));
        }

        let input1 = &inputs[0];
        let input2 = &inputs[1];
        let input3 = &inputs[2];

        if input1.shape() != input2.shape() || input1.shape() != input3.shape() {
            return Err(TorshError::InvalidArgument(
                "Input tensors must have the same shape for Mul+Add fusion".to_string(),
            ));
        }

        let data1 = input1.data()?;
        let data2 = input2.data()?;
        let data3 = input3.data()?;

        let mul_add_data: Vec<f32> = data1
            .iter()
            .zip(data2.iter())
            .zip(data3.iter())
            .map(|((&x1, &x2), &x3)| x1 * x2 + x3)
            .collect();

        let output = Tensor::from_data(
            mul_add_data,
            input1.shape().dims().to_vec(),
            input1.device(),
        )?;

        Ok(vec![output])
    }
}

impl Default for FusionEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Fusion optimization pass
pub struct FusionPass {
    /// The fusion engine
    engine: FusionEngine,
    /// Whether to enable aggressive fusion
    aggressive_fusion: bool,
}

impl FusionPass {
    /// Create a new fusion pass
    pub fn new() -> Self {
        Self {
            engine: FusionEngine::new(),
            aggressive_fusion: false,
        }
    }

    /// Enable or disable aggressive fusion
    pub fn set_aggressive_fusion(&mut self, enable: bool) {
        self.aggressive_fusion = enable;
    }

    /// Get the fusion engine
    pub fn engine(&self) -> &FusionEngine {
        &self.engine
    }

    /// Get the fusion engine mutably
    pub fn engine_mut(&mut self) -> &mut FusionEngine {
        &mut self.engine
    }

    /// Apply fusion pass to a model
    pub fn apply(&self, model_ops: &[String]) -> TorshResult<Vec<FusedOp>> {
        let mut fused_ops = self.engine.fuse_model(model_ops)?;

        if self.aggressive_fusion {
            // Apply additional aggressive fusion optimizations
            fused_ops = self.apply_aggressive_fusion(fused_ops)?;
        }

        Ok(fused_ops)
    }

    /// Apply aggressive fusion optimizations
    fn apply_aggressive_fusion(&self, ops: Vec<FusedOp>) -> TorshResult<Vec<FusedOp>> {
        let mut optimized_ops = Vec::new();
        let mut i = 0;

        while i < ops.len() {
            let current_op = &ops[i];

            // Try to fuse with next operation if possible
            if i + 1 < ops.len() {
                let next_op = &ops[i + 1];

                // Example: try to fuse two consecutive fused operations
                if self.can_fuse_operations(current_op, next_op) {
                    let combined_op = self.combine_operations(current_op, next_op)?;
                    optimized_ops.push(combined_op);
                    i += 2; // Skip next operation as it's been fused
                    continue;
                }
            }

            optimized_ops.push(current_op.clone());
            i += 1;
        }

        Ok(optimized_ops)
    }

    /// Check if two operations can be fused together
    fn can_fuse_operations(&self, op1: &FusedOp, op2: &FusedOp) -> bool {
        // Simple heuristic: check if output of first op matches input of second op
        op1.outputs.len() == 1 && op2.inputs.len() == 1 && op1.outputs[0] == op2.inputs[0]
    }

    /// Combine two operations into a single fused operation
    fn combine_operations(&self, op1: &FusedOp, op2: &FusedOp) -> TorshResult<FusedOp> {
        let mut combined_components = op1.component_ops.clone();
        combined_components.extend(op2.component_ops.clone());

        let combined_op = FusedOp {
            name: format!("combined_{}_{}", op1.name, op2.name),
            component_ops: combined_components,
            inputs: op1.inputs.clone(),
            outputs: op2.outputs.clone(),
            qconfig: op1.qconfig.clone().or(op2.qconfig.clone()),
        };

        Ok(combined_op)
    }
}

impl Default for FusionPass {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::tensor_1d;

    #[test]
    fn test_fusion_engine_creation() {
        let engine = FusionEngine::new();
        assert!(!engine.patterns.is_empty());
    }

    #[test]
    fn test_conv_bn_fusion() {
        let engine = FusionEngine::new();
        let ops = vec!["conv2d".to_string(), "batch_norm".to_string()];
        let fused_ops = engine.fuse_model(&ops).unwrap();

        assert_eq!(fused_ops.len(), 1);
        assert_eq!(fused_ops[0].name, "fused_conv_bn");
        assert_eq!(fused_ops[0].component_ops.len(), 2);
    }

    #[test]
    fn test_conv_bn_relu_fusion() {
        let engine = FusionEngine::new();
        let ops = vec![
            "conv2d".to_string(),
            "batch_norm".to_string(),
            "relu".to_string(),
        ];
        let fused_ops = engine.fuse_model(&ops).unwrap();

        assert_eq!(fused_ops.len(), 1);
        assert_eq!(fused_ops[0].name, "fused_conv_bn_relu");
        assert_eq!(fused_ops[0].component_ops.len(), 3);
    }

    #[test]
    fn test_linear_relu_fusion() {
        let engine = FusionEngine::new();
        let ops = vec!["linear".to_string(), "relu".to_string()];
        let fused_ops = engine.fuse_model(&ops).unwrap();

        assert_eq!(fused_ops.len(), 1);
        assert_eq!(fused_ops[0].name, "fused_linear_relu");
        assert_eq!(fused_ops[0].component_ops.len(), 2);
    }

    #[test]
    fn test_quant_dequant_elimination() {
        let engine = FusionEngine::new();
        let ops = vec!["quantize".to_string(), "dequantize".to_string()];
        let fused_ops = engine.fuse_model(&ops).unwrap();

        assert_eq!(fused_ops.len(), 1);
        assert_eq!(fused_ops[0].name, "identity");
        assert_eq!(fused_ops[0].component_ops.len(), 2);
    }

    #[test]
    fn test_no_fusion_fallback() {
        let engine = FusionEngine::new();
        let ops = vec!["unknown_op".to_string()];
        let fused_ops = engine.fuse_model(&ops).unwrap();

        assert_eq!(fused_ops.len(), 1);
        assert_eq!(fused_ops[0].name, "unknown_op");
        assert_eq!(fused_ops[0].component_ops.len(), 1);
    }

    #[test]
    fn test_add_relu_execution() {
        let engine = FusionEngine::new();

        let input1 = tensor_1d(&[1.0, -2.0, 3.0]).unwrap();
        let input2 = tensor_1d(&[2.0, 1.0, -1.0]).unwrap();
        let inputs = vec![input1, input2];

        let result = engine.execute_add_relu_fusion(&inputs).unwrap();
        assert_eq!(result.len(), 1);

        let output_data = result[0].to_vec().unwrap();
        assert_eq!(output_data, vec![3.0, 0.0, 2.0]); // (1+2), max(-2+1, 0), (3-1)
    }

    #[test]
    fn test_mul_add_execution() {
        let engine = FusionEngine::new();

        let input1 = tensor_1d(&[2.0, 3.0, 4.0]).unwrap();
        let input2 = tensor_1d(&[3.0, 2.0, 1.0]).unwrap();
        let input3 = tensor_1d(&[1.0, 1.0, 1.0]).unwrap();
        let inputs = vec![input1, input2, input3];

        let result = engine.execute_mul_add_fusion(&inputs).unwrap();
        assert_eq!(result.len(), 1);

        let output_data = result[0].to_vec().unwrap();
        assert_eq!(output_data, vec![7.0, 7.0, 5.0]); // (2*3+1), (3*2+1), (4*1+1)
    }

    #[test]
    fn test_fusion_pass() {
        let mut pass = FusionPass::new();
        pass.set_aggressive_fusion(true);

        let ops = vec![
            "conv2d".to_string(),
            "batch_norm".to_string(),
            "relu".to_string(),
        ];
        let fused_ops = pass.apply(&ops).unwrap();

        assert!(!fused_ops.is_empty());
        assert_eq!(fused_ops[0].name, "fused_conv_bn_relu");
    }

    #[test]
    fn test_fusion_config() {
        let mut engine = FusionEngine::new();
        let config = QuantConfig::int8();

        engine.set_fusion_config(FusionPattern::ConvReLU, config.clone());

        let ops = vec!["conv2d".to_string(), "relu".to_string()];
        let fused_ops = engine.fuse_model(&ops).unwrap();

        assert_eq!(fused_ops.len(), 1);
        assert!(fused_ops[0].qconfig.is_some());
    }
}
