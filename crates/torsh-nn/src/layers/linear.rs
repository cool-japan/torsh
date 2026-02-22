//! Linear (fully connected) layers

use crate::{Module, ModuleBase, Parameter};
use torsh_core::device::DeviceType;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
use torsh_core::error::Result;
use torsh_tensor::{creation::*, Tensor};

/// Linear (fully connected) layer
pub struct Linear {
    base: ModuleBase,
    in_features: usize,
    out_features: usize,
    use_bias: bool,
}

impl Linear {
    /// Create a new linear layer
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let mut base = ModuleBase::new();

        // Initialize weight with shape [in_features, out_features] for direct matmul
        // This way input[batch, in_features] @ weight[in_features, out_features] = output[batch, out_features]
        let weight = crate::init::xavier_uniform(&[in_features, out_features])
            .expect("Failed to initialize linear layer weight");
        base.register_parameter("weight".to_string(), Parameter::new(weight));

        if bias {
            let bias_tensor = zeros(&[out_features]).expect("Failed to create bias tensor");
            base.register_parameter("bias".to_string(), Parameter::new(bias_tensor));
        }

        Self {
            base,
            in_features,
            out_features,
            use_bias: bias,
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified linear transformation using basic tensor operations
        let weight = self.base.parameters["weight"].tensor().read().clone();

        // Compute input @ weight
        let output = input.matmul(&weight)?;

        if self.use_bias {
            let bias = self.base.parameters["bias"].tensor().read().clone();
            Ok(output.add(&bias)?)
        } else {
            Ok(output)
        }
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        self.base.parameters.clone()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        self.base.named_parameters()
    }
}

impl core::fmt::Debug for Linear {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Linear")
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("use_bias", &self.use_bias)
            .finish()
    }
}

/// Flatten layer to reshape tensor to 1D (except batch dimension)
pub struct Flatten {
    base: ModuleBase,
    start_dim: usize,
    end_dim: Option<usize>,
}

impl Flatten {
    /// Create a new flatten layer
    pub fn new() -> Self {
        Self {
            base: ModuleBase::new(),
            start_dim: 1,
            end_dim: None,
        }
    }

    /// Create a flatten layer with custom dimensions
    pub fn with_dims(start_dim: usize, end_dim: Option<usize>) -> Self {
        Self {
            base: ModuleBase::new(),
            start_dim,
            end_dim,
        }
    }
}

impl Module for Flatten {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        let dims = shape.dims();

        if dims.is_empty() {
            return Ok(input.clone());
        }

        let start = self.start_dim.min(dims.len());
        let end = self.end_dim.unwrap_or(dims.len()).min(dims.len());

        if start >= end {
            return Ok(input.clone());
        }

        // Calculate new shape
        let mut new_shape = Vec::new();

        // Keep dimensions before start_dim
        new_shape.extend_from_slice(&dims[..start]);

        // Flatten dimensions from start_dim to end_dim
        let flattened_size: usize = dims[start..end].iter().product();
        new_shape.push(flattened_size);

        // Keep dimensions after end_dim
        if end < dims.len() {
            new_shape.extend_from_slice(&dims[end..]);
        }

        let new_shape_i32: Vec<i32> = new_shape.iter().map(|&x| x as i32).collect();
        input.reshape(&new_shape_i32)
    }

    fn parameters(&self) -> HashMap<String, Parameter> {
        HashMap::new()
    }

    fn training(&self) -> bool {
        self.base.training()
    }

    fn train(&mut self) {
        self.base.set_training(true);
    }

    fn eval(&mut self) {
        self.base.set_training(false);
    }

    fn set_training(&mut self, training: bool) {
        self.base.set_training(training);
    }

    fn to_device(&mut self, device: DeviceType) -> Result<()> {
        self.base.to_device(device)
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        HashMap::new()
    }
}

impl core::fmt::Debug for Flatten {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Flatten")
            .field("start_dim", &self.start_dim)
            .field("end_dim", &self.end_dim)
            .finish()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // =========================================================================
    // LINEAR LAYER TESTS
    // =========================================================================

    #[test]
    fn test_linear_creation() {
        let layer = Linear::new(10, 5, true);
        assert_eq!(layer.in_features, 10);
        assert_eq!(layer.out_features, 5);
        assert!(layer.use_bias);

        let params = layer.parameters();
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }

    #[test]
    fn test_linear_no_bias() {
        let layer = Linear::new(10, 5, false);
        assert!(!layer.use_bias);

        let params = layer.parameters();
        assert!(params.contains_key("weight"));
        assert!(!params.contains_key("bias"));
    }

    #[test]
    fn test_linear_forward_basic() -> Result<()> {
        let mut layer = Linear::new(3, 2, false);

        // Set known weights for testing
        let weight = Tensor::from_vec(
            vec![
                1.0, 0.0, // Feature 0 weights
                0.0, 1.0, // Feature 1 weights
                0.0, 0.0, // Feature 2 weights
            ],
            &[3, 2],
        )?;
        *layer
            .base
            .parameters
            .get_mut("weight")
            .unwrap()
            .tensor()
            .write() = weight;

        let input = Tensor::from_vec(vec![2.0, 3.0, 1.0], &[1, 3])?;
        let output = layer.forward(&input)?;

        let output_data = output.to_vec()?;
        assert_eq!(output_data.len(), 2);
        assert_relative_eq!(output_data[0], 2.0, epsilon = 1e-5); // 2*1 + 3*0 + 1*0
        assert_relative_eq!(output_data[1], 3.0, epsilon = 1e-5); // 2*0 + 3*1 + 1*0

        Ok(())
    }

    #[test]
    fn test_linear_forward_with_bias() -> Result<()> {
        let mut layer = Linear::new(3, 2, true);

        let weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[3, 2])?;
        *layer
            .base
            .parameters
            .get_mut("weight")
            .unwrap()
            .tensor()
            .write() = weight;

        let bias = Tensor::from_vec(vec![0.5, -0.5], &[2])?;
        *layer
            .base
            .parameters
            .get_mut("bias")
            .unwrap()
            .tensor()
            .write() = bias;

        let input = Tensor::from_vec(vec![2.0, 3.0, 1.0], &[1, 3])?;
        let output = layer.forward(&input)?;

        let output_data = output.to_vec()?;
        assert_relative_eq!(output_data[0], 2.5, epsilon = 1e-5); // 2.0 + 0.5
        assert_relative_eq!(output_data[1], 2.5, epsilon = 1e-5); // 3.0 - 0.5

        Ok(())
    }

    #[test]
    fn test_linear_forward_batch() -> Result<()> {
        let mut layer = Linear::new(3, 2, false);

        let weight = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5], &[3, 2])?;
        *layer
            .base
            .parameters
            .get_mut("weight")
            .unwrap()
            .tensor()
            .write() = weight;

        // Batch of 2 samples
        let input = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, // Sample 1
                4.0, 5.0, 6.0, // Sample 2
            ],
            &[2, 3],
        )?;
        let output = layer.forward(&input)?;

        assert_eq!(output.shape().dims(), &[2, 2]);

        let output_data = output.to_vec()?;
        // Sample 1: [1*1 + 2*0 + 3*0.5, 1*0 + 2*1 + 3*0.5] = [2.5, 3.5]
        assert_relative_eq!(output_data[0], 2.5, epsilon = 1e-5);
        assert_relative_eq!(output_data[1], 3.5, epsilon = 1e-5);
        // Sample 2: [4*1 + 5*0 + 6*0.5, 4*0 + 5*1 + 6*0.5] = [7.0, 8.0]
        assert_relative_eq!(output_data[2], 7.0, epsilon = 1e-5);
        assert_relative_eq!(output_data[3], 8.0, epsilon = 1e-5);

        Ok(())
    }

    #[test]
    fn test_linear_training_mode() {
        let mut layer = Linear::new(10, 5, true);
        assert!(layer.training()); // Default is training mode

        layer.eval();
        assert!(!layer.training());

        layer.train();
        assert!(layer.training());

        layer.set_training(false);
        assert!(!layer.training());
    }

    #[test]
    fn test_linear_weight_shape() -> Result<()> {
        let layer = Linear::new(10, 5, true);
        let params = layer.parameters();

        let weight_arc = params["weight"].tensor();
        let weight = weight_arc.read();
        assert_eq!(weight.shape().dims(), &[10, 5]);

        let bias_arc = params["bias"].tensor();
        let bias = bias_arc.read();
        assert_eq!(bias.shape().dims(), &[5]);

        Ok(())
    }

    // =========================================================================
    // FLATTEN LAYER TESTS
    // =========================================================================

    #[test]
    fn test_flatten_creation() {
        let layer = Flatten::new();
        assert_eq!(layer.start_dim, 1);
        assert_eq!(layer.end_dim, None);

        let params = layer.parameters();
        assert!(params.is_empty()); // Flatten has no learnable parameters
    }

    #[test]
    fn test_flatten_with_dims() {
        let layer = Flatten::with_dims(2, Some(4));
        assert_eq!(layer.start_dim, 2);
        assert_eq!(layer.end_dim, Some(4));
    }

    #[test]
    fn test_flatten_basic() -> Result<()> {
        let layer = Flatten::new(); // Default: start_dim=1

        // Input shape: [batch=2, height=3, width=4]
        let input = Tensor::from_vec(vec![1.0; 24], &[2, 3, 4])?;
        let output = layer.forward(&input)?;

        // Should flatten from dim 1 onwards: [2, 3*4] = [2, 12]
        assert_eq!(output.shape().dims(), &[2, 12]);

        Ok(())
    }

    #[test]
    fn test_flatten_4d_tensor() -> Result<()> {
        let layer = Flatten::new();

        // Input shape: [batch=2, channels=3, height=4, width=5]
        let input = Tensor::from_vec(vec![1.0; 120], &[2, 3, 4, 5])?;
        let output = layer.forward(&input)?;

        // Should flatten from dim 1 onwards: [2, 3*4*5] = [2, 60]
        assert_eq!(output.shape().dims(), &[2, 60]);

        Ok(())
    }

    #[test]
    fn test_flatten_custom_dims() -> Result<()> {
        let layer = Flatten::with_dims(1, Some(3));

        // Input shape: [batch=2, dim1=3, dim2=4, dim3=5]
        let input = Tensor::from_vec(vec![1.0; 120], &[2, 3, 4, 5])?;
        let output = layer.forward(&input)?;

        // Should flatten from dim 1 to 3 (exclusive): [2, 3*4, 5] = [2, 12, 5]
        assert_eq!(output.shape().dims(), &[2, 12, 5]);

        Ok(())
    }

    #[test]
    fn test_flatten_all_dims() -> Result<()> {
        let layer = Flatten::with_dims(0, None);

        // Input shape: [2, 3, 4]
        let input = Tensor::from_vec(vec![1.0; 24], &[2, 3, 4])?;
        let output = layer.forward(&input)?;

        // Should flatten all dimensions: [2*3*4] = [24]
        assert_eq!(output.shape().dims(), &[24]);

        Ok(())
    }

    #[test]
    fn test_flatten_empty_tensor() -> Result<()> {
        let layer = Flatten::new();

        let input = Tensor::from_vec(vec![], &[0])?;
        let output = layer.forward(&input)?;

        // Empty tensor should remain empty
        assert_eq!(output.shape().dims(), &[0]);

        Ok(())
    }

    #[test]
    fn test_flatten_1d_tensor() -> Result<()> {
        let layer = Flatten::new();

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let output = layer.forward(&input)?;

        // 1D tensor with start_dim=1 should remain unchanged (start >= dims.len())
        assert_eq!(output.shape().dims(), &[3]);

        Ok(())
    }

    #[test]
    fn test_flatten_start_equals_end() -> Result<()> {
        let layer = Flatten::with_dims(2, Some(2));

        let input = Tensor::from_vec(vec![1.0; 24], &[2, 3, 4])?;
        let output = layer.forward(&input)?;

        // start == end should not flatten anything
        assert_eq!(output.shape().dims(), &[2, 3, 4]);

        Ok(())
    }

    #[test]
    fn test_flatten_preserves_data() -> Result<()> {
        let layer = Flatten::new();

        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::from_vec(input_data.clone(), &[2, 3])?;
        let output = layer.forward(&input)?;

        let output_data = output.to_vec()?;
        assert_eq!(output_data, input_data); // Data should be preserved

        Ok(())
    }

    #[test]
    fn test_flatten_training_mode() {
        let mut layer = Flatten::new();
        assert!(layer.training());

        layer.eval();
        assert!(!layer.training());

        layer.train();
        assert!(layer.training());
    }

    #[test]
    fn test_flatten_no_parameters() {
        let layer = Flatten::new();
        let params = layer.parameters();
        assert!(params.is_empty());

        let named_params = layer.named_parameters();
        assert!(named_params.is_empty());
    }
}
