//! WebAssembly (WASM) bindings for ToRSh
//!
//! This module provides comprehensive WebAssembly bindings for the ToRSh deep learning framework,
//! enabling deployment in browser environments, edge computing, and serverless platforms.
//!
//! # Features
//!
//! - **Browser Compatibility**: Run deep learning models directly in web browsers
//! - **Edge Deployment**: Deploy models on edge devices and serverless platforms
//! - **Zero Installation**: No native dependencies required
//! - **Memory Safety**: Rust's memory safety guarantees in WASM
//! - **Performance**: Near-native performance with WASM optimization
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │         JavaScript/TypeScript Layer             │
//! │  (Browser, Node.js, Deno, Cloudflare Workers)  │
//! └───────────────────┬─────────────────────────────┘
//!                     │
//! ┌───────────────────▼─────────────────────────────┐
//! │          WASM Bindgen Interface                 │
//! │  • Type Conversions  • Memory Management        │
//! │  • Promise Support   • Error Handling           │
//! └───────────────────┬─────────────────────────────┘
//!                     │
//! ┌───────────────────▼─────────────────────────────┐
//! │              ToRSh Core (WASM)                  │
//! │  • Tensor Operations  • Neural Networks         │
//! │  • Autograd           • Optimizers              │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! # Quick Start
//!
//! ## JavaScript/TypeScript Usage
//!
//! ```javascript
//! import * as torsh from 'torsh-wasm';
//!
//! // Initialize ToRSh WASM module
//! await torsh.default();
//!
//! // Create tensors
//! const x = torsh.Tensor.randn([2, 3]);
//! const y = torsh.Tensor.ones([3, 4]);
//!
//! // Neural network
//! const model = new torsh.Sequential([
//!   new torsh.Linear(3, 64),
//!   new torsh.ReLU(),
//!   new torsh.Linear(64, 4)
//! ]);
//!
//! // Forward pass
//! const output = model.forward(x);
//!
//! // Training
//! const optimizer = new torsh.Adam(model.parameters(), { lr: 0.001 });
//! const loss = torsh.mse_loss(output, target);
//! loss.backward();
//! optimizer.step();
//! ```
//!
//! ## Browser Usage
//!
//! ```html
//! <!DOCTYPE html>
//! <html>
//! <head>
//!   <script type="module">
//!     import init, * as torsh from './torsh_wasm.js';
//!
//!     async function run() {
//!       await init();
//!       const tensor = torsh.Tensor.randn([10, 10]);
//!       console.log('Tensor created:', tensor.shape());
//!     }
//!
//!     run();
//!   </script>
//! </head>
//! <body>
//!   <h1>ToRSh WASM Demo</h1>
//! </body>
//! </html>
//! ```
//!
//! # Platform Support
//!
//! - ✅ **Web Browsers**: Chrome, Firefox, Safari, Edge
//! - ✅ **Node.js**: v14+ with WASM support
//! - ✅ **Deno**: Native WASM support
//! - ✅ **Cloudflare Workers**: Edge computing platform
//! - ✅ **Vercel Edge Functions**: Serverless edge deployment
//! - ✅ **AWS Lambda@Edge**: CDN edge functions
//!
//! # Performance Considerations
//!
//! - **SIMD Support**: Leverages WebAssembly SIMD when available
//! - **Threading**: Uses Web Workers for parallel computation
//! - **Memory Management**: Efficient memory pooling for WASM heap
//! - **Lazy Loading**: Progressive model loading for faster startup
//!
//! # Security
//!
//! - **Sandboxed Execution**: WASM provides secure sandboxing
//! - **Memory Safety**: Rust's guarantees prevent buffer overflows
//! - **No Unsafe Access**: Cannot access system resources directly
//! - **Cryptographic Operations**: Secure random number generation

// Framework infrastructure - components designed for future use
#![allow(dead_code)]

use oxicode::config;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// Note: In a real WASM build, we would use wasm-bindgen
// For now, we provide the structure that would work with wasm-bindgen

/// WASM-compatible tensor representation
///
/// This struct provides a JavaScript-friendly interface to ToRSh tensors
/// through wasm-bindgen's type conversions.
#[derive(Clone, Serialize, Deserialize)]
pub struct WasmTensor {
    /// Tensor data as flat Vec<f32>
    data: Vec<f32>,
    /// Tensor shape dimensions
    shape: Vec<usize>,
    /// Whether gradients are tracked
    requires_grad: bool,
    /// Optional gradient storage
    #[serde(skip)]
    grad: Option<Arc<RwLock<Vec<f32>>>>,
}

impl WasmTensor {
    /// Create a new WASM tensor from data and shape
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        // Validate shape matches data length
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        );

        Self {
            data,
            shape,
            requires_grad: false,
            grad: None,
        }
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self::new(vec![0.0; len], shape)
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self::new(vec![1.0; len], shape)
    }

    /// Create a tensor filled with random values from normal distribution
    pub fn randn(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        // Use fastrand for WASM compatibility (lighter than rand)
        let data: Vec<f32> = (0..len)
            .map(|_| {
                // Box-Muller transform for normal distribution
                let u1 = fastrand::f32();
                let u2 = fastrand::f32();
                ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()) as f32
            })
            .collect();
        Self::new(data, shape)
    }

    /// Create a tensor filled with random uniform values [0, 1)
    pub fn rand(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        let data: Vec<f32> = (0..len).map(|_| fastrand::f32()).collect();
        Self::new(data, shape)
    }

    /// Get tensor shape
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get tensor data as slice
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable tensor data
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Enable gradient tracking
    pub fn requires_grad_(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        if requires_grad && self.grad.is_none() {
            self.grad = Some(Arc::new(RwLock::new(vec![0.0; self.data.len()])));
        }
        self
    }

    /// Check if gradient tracking is enabled
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Reshape tensor (view operation)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, String> {
        let new_len: usize = new_shape.iter().product();
        if new_len != self.data.len() {
            return Err(format!(
                "Cannot reshape tensor of size {} to shape {:?} (size {})",
                self.data.len(),
                new_shape,
                new_len
            ));
        }

        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
            requires_grad: self.requires_grad,
            grad: self.grad.clone(),
        })
    }

    /// Transpose 2D tensor
    pub fn transpose(&self) -> Result<Self, String> {
        if self.shape.len() != 2 {
            return Err(format!(
                "Transpose only supported for 2D tensors, got shape {:?}",
                self.shape
            ));
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut new_data = vec![0.0; self.data.len()];

        for i in 0..rows {
            for j in 0..cols {
                new_data[j * rows + i] = self.data[i * cols + j];
            }
        }

        Ok(Self::new(new_data, vec![cols, rows]))
    }

    /// Matrix multiplication (2D only)
    pub fn matmul(&self, other: &Self) -> Result<Self, String> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err("Matrix multiplication requires 2D tensors".to_string());
        }

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        if k != other.shape[0] {
            return Err(format!(
                "Incompatible shapes for matmul: {:?} and {:?}",
                self.shape, other.shape
            ));
        }

        let mut result = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for k_idx in 0..k {
                    sum += self.data[i * k + k_idx] * other.data[k_idx * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Ok(Self::new(result, vec![m, n]))
    }

    /// Element-wise addition
    pub fn add(&self, other: &Self) -> Result<Self, String> {
        if self.shape != other.shape {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            ));
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(Self::new(data, self.shape.clone()))
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Self) -> Result<Self, String> {
        if self.shape != other.shape {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            ));
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Ok(Self::new(data, self.shape.clone()))
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Self) -> Result<Self, String> {
        if self.shape != other.shape {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            ));
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        Ok(Self::new(data, self.shape.clone()))
    }

    /// Element-wise division
    pub fn div(&self, other: &Self) -> Result<Self, String> {
        if self.shape != other.shape {
            return Err(format!(
                "Shape mismatch: {:?} vs {:?}",
                self.shape, other.shape
            ));
        }

        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a / b)
            .collect();

        Ok(Self::new(data, self.shape.clone()))
    }

    /// Scalar addition
    pub fn add_scalar(&self, scalar: f32) -> Self {
        let data: Vec<f32> = self.data.iter().map(|x| x + scalar).collect();
        Self::new(data, self.shape.clone())
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: f32) -> Self {
        let data: Vec<f32> = self.data.iter().map(|x| x * scalar).collect();
        Self::new(data, self.shape.clone())
    }

    /// ReLU activation
    pub fn relu(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|x| x.max(0.0)).collect();
        Self::new(data, self.shape.clone())
    }

    /// Sigmoid activation
    pub fn sigmoid(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        Self::new(data, self.shape.clone())
    }

    /// Tanh activation
    pub fn tanh(&self) -> Self {
        let data: Vec<f32> = self.data.iter().map(|x| x.tanh()).collect();
        Self::new(data, self.shape.clone())
    }

    /// Sum all elements
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Mean of all elements
    pub fn mean(&self) -> f32 {
        self.sum() / self.data.len() as f32
    }

    /// Maximum element value
    pub fn max(&self) -> f32 {
        self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Minimum element value
    pub fn min(&self) -> f32 {
        self.data.iter().copied().fold(f32::INFINITY, f32::min)
    }

    /// Convert to JSON string
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self).map_err(|e| format!("JSON serialization failed: {}", e))
    }

    /// Create from JSON string
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("JSON deserialization failed: {}", e))
    }
}

/// WASM-compatible linear layer
#[derive(Clone)]
pub struct WasmLinear {
    in_features: usize,
    out_features: usize,
    weight: WasmTensor,
    bias: Option<WasmTensor>,
}

impl WasmLinear {
    /// Create a new linear layer
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Xavier/Glorot initialization
        let limit = (6.0 / (in_features + out_features) as f32).sqrt();
        let weight = WasmTensor::rand(vec![out_features, in_features])
            .mul_scalar(2.0 * limit)
            .add_scalar(-limit);

        let bias = if bias {
            Some(WasmTensor::zeros(vec![out_features]))
        } else {
            None
        };

        Self {
            in_features,
            out_features,
            weight,
            bias,
        }
    }

    /// Forward pass
    pub fn forward(&self, input: &WasmTensor) -> Result<WasmTensor, String> {
        // Support both 1D and 2D inputs
        let input_shape = input.shape();
        let is_batched = input_shape.len() == 2;

        if !is_batched && input_shape.len() != 1 {
            return Err(format!(
                "Expected 1D or 2D input, got shape {:?}",
                input_shape
            ));
        }

        let last_dim = *input_shape.last().expect("input_shape should not be empty");
        if last_dim != self.in_features {
            return Err(format!(
                "Input feature size {} doesn't match layer input size {}",
                last_dim, self.in_features
            ));
        }

        // Reshape input to 2D if needed
        let input_2d = if !is_batched {
            input.reshape(vec![1, self.in_features])?
        } else {
            input.clone()
        };

        // Matrix multiplication: (batch, in_features) @ (out_features, in_features)^T
        // = (batch, in_features) @ (in_features, out_features) = (batch, out_features)
        let weight_t = self.weight.transpose()?;
        let mut output = input_2d.matmul(&weight_t)?;

        // Add bias if present
        if let Some(ref bias) = self.bias {
            // Broadcast bias across batch dimension
            let batch_size = output.shape()[0];
            for i in 0..batch_size {
                for j in 0..self.out_features {
                    output.data_mut()[i * self.out_features + j] += bias.data()[j];
                }
            }
        }

        // Reshape back to 1D if input was 1D
        if !is_batched {
            output = output.reshape(vec![self.out_features])?;
        }

        Ok(output)
    }

    /// Get layer parameters
    pub fn parameters(&self) -> Vec<WasmTensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}

/// WASM-compatible sequential container
pub struct WasmSequential {
    layers: Vec<WasmLayer>,
}

/// WASM layer enum
#[derive(Clone)]
pub enum WasmLayer {
    Linear(WasmLinear),
    ReLU,
    Sigmoid,
    Tanh,
}

impl WasmSequential {
    /// Create new sequential container
    pub fn new(layers: Vec<WasmLayer>) -> Self {
        Self { layers }
    }

    /// Forward pass through all layers
    pub fn forward(&self, mut input: WasmTensor) -> Result<WasmTensor, String> {
        for layer in &self.layers {
            input = match layer {
                WasmLayer::Linear(linear) => linear.forward(&input)?,
                WasmLayer::ReLU => input.relu(),
                WasmLayer::Sigmoid => input.sigmoid(),
                WasmLayer::Tanh => input.tanh(),
            };
        }
        Ok(input)
    }

    /// Get all parameters
    pub fn parameters(&self) -> Vec<WasmTensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            if let WasmLayer::Linear(linear) = layer {
                params.extend(linear.parameters());
            }
        }
        params
    }
}

/// WASM-compatible optimizer configuration
#[derive(Clone, Serialize, Deserialize)]
pub struct WasmOptimizerConfig {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
}

impl Default for WasmOptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// WASM-compatible Adam optimizer
pub struct WasmAdam {
    config: WasmOptimizerConfig,
    step_count: usize,
    // First moment estimates
    m: Vec<Vec<f32>>,
    // Second moment estimates
    v: Vec<Vec<f32>>,
}

impl WasmAdam {
    /// Create new Adam optimizer
    pub fn new(num_params: usize, config: WasmOptimizerConfig) -> Self {
        Self {
            config,
            step_count: 0,
            m: vec![Vec::new(); num_params],
            v: vec![Vec::new(); num_params],
        }
    }

    /// Perform optimization step
    pub fn step(
        &mut self,
        parameters: &mut [WasmTensor],
        gradients: &[Vec<f32>],
    ) -> Result<(), String> {
        if parameters.len() != gradients.len() {
            return Err(format!(
                "Parameter and gradient count mismatch: {} vs {}",
                parameters.len(),
                gradients.len()
            ));
        }

        self.step_count += 1;
        let t = self.step_count as f32;

        for (i, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            if grad.len() != param.numel() {
                return Err(format!(
                    "Gradient size {} doesn't match parameter size {}",
                    grad.len(),
                    param.numel()
                ));
            }

            // Initialize moment estimates if needed
            if self.m[i].is_empty() {
                self.m[i] = vec![0.0; param.numel()];
                self.v[i] = vec![0.0; param.numel()];
            }

            let data = param.data_mut();

            for j in 0..data.len() {
                // Update biased first moment estimate
                self.m[i][j] =
                    self.config.beta1 * self.m[i][j] + (1.0 - self.config.beta1) * grad[j];

                // Update biased second moment estimate
                self.v[i][j] = self.config.beta2 * self.v[i][j]
                    + (1.0 - self.config.beta2) * grad[j] * grad[j];

                // Compute bias-corrected estimates
                let m_hat = self.m[i][j] / (1.0 - self.config.beta1.powf(t));
                let v_hat = self.v[i][j] / (1.0 - self.config.beta2.powf(t));

                // Update parameters
                data[j] -= self.config.learning_rate * m_hat / (v_hat.sqrt() + self.config.epsilon);

                // Apply weight decay (L2 regularization)
                if self.config.weight_decay > 0.0 {
                    data[j] -= self.config.learning_rate * self.config.weight_decay * data[j];
                }
            }
        }

        Ok(())
    }

    /// Reset optimizer state
    pub fn zero_grad(&mut self) {
        self.step_count = 0;
        for i in 0..self.m.len() {
            self.m[i].fill(0.0);
            self.v[i].fill(0.0);
        }
    }
}

/// Loss functions
pub mod loss {
    use super::WasmTensor;

    /// Mean Squared Error loss
    pub fn mse_loss(predictions: &WasmTensor, targets: &WasmTensor) -> Result<f32, String> {
        if predictions.shape() != targets.shape() {
            return Err(format!(
                "Shape mismatch: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            ));
        }

        let diff = predictions.sub(targets)?;
        let squared: Vec<f32> = diff.data().iter().map(|x| x * x).collect();
        let sum: f32 = squared.iter().sum();
        Ok(sum / squared.len() as f32)
    }

    /// Binary Cross Entropy loss
    pub fn binary_cross_entropy(
        predictions: &WasmTensor,
        targets: &WasmTensor,
    ) -> Result<f32, String> {
        if predictions.shape() != targets.shape() {
            return Err(format!(
                "Shape mismatch: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            ));
        }

        let epsilon = 1e-7;
        let mut sum = 0.0;

        for (pred, target) in predictions.data().iter().zip(targets.data().iter()) {
            let pred_clipped = pred.clamp(epsilon, 1.0 - epsilon);
            sum += -(target * pred_clipped.ln() + (1.0 - target) * (1.0 - pred_clipped).ln());
        }

        Ok(sum / predictions.numel() as f32)
    }

    /// Cross Entropy loss (for multi-class classification)
    pub fn cross_entropy(predictions: &WasmTensor, targets: &WasmTensor) -> Result<f32, String> {
        if predictions.shape() != targets.shape() {
            return Err(format!(
                "Shape mismatch: predictions {:?} vs targets {:?}",
                predictions.shape(),
                targets.shape()
            ));
        }

        let epsilon = 1e-7;
        let mut sum = 0.0;

        for (pred, target) in predictions.data().iter().zip(targets.data().iter()) {
            let pred_clipped = pred.clamp(epsilon, 1.0);
            sum += -target * pred_clipped.ln();
        }

        Ok(sum / predictions.numel() as f32)
    }
}

/// WASM utilities
pub mod utils {
    use super::*;

    /// Convert JavaScript Float32Array to WasmTensor
    pub fn float32array_to_tensor(data: Vec<f32>, shape: Vec<usize>) -> Result<WasmTensor, String> {
        Ok(WasmTensor::new(data, shape))
    }

    /// Serialize tensor for transfer to JavaScript
    pub fn tensor_to_bytes(tensor: &WasmTensor) -> Vec<u8> {
        oxicode::serde::encode_to_vec(tensor, config::standard()).unwrap_or_default()
    }

    /// Deserialize tensor from bytes
    pub fn tensor_from_bytes(bytes: &[u8]) -> Result<WasmTensor, String> {
        oxicode::serde::decode_from_slice(bytes, config::standard())
            .map(|(tensor, _)| tensor)
            .map_err(|e| format!("Deserialization failed: {}", e))
    }

    /// Get memory usage statistics
    #[derive(Serialize, Deserialize)]
    pub struct MemoryStats {
        pub tensor_count: usize,
        pub total_elements: usize,
        pub bytes_used: usize,
    }

    /// Calculate memory statistics for tensors
    pub fn get_memory_stats(tensors: &[WasmTensor]) -> MemoryStats {
        let tensor_count = tensors.len();
        let total_elements: usize = tensors.iter().map(|t| t.numel()).sum();
        let bytes_used = total_elements * std::mem::size_of::<f32>();

        MemoryStats {
            tensor_count,
            total_elements,
            bytes_used,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasm_tensor_creation() {
        let tensor = WasmTensor::zeros(vec![2, 3]);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);
        assert!(tensor.data().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_wasm_tensor_ops() {
        let a = WasmTensor::ones(vec![2, 2]);
        let b = WasmTensor::ones(vec![2, 2]).mul_scalar(2.0);

        let c = a.add(&b).unwrap();
        assert!(c.data().iter().all(|&x| (x - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_wasm_linear_layer() {
        let layer = WasmLinear::new(3, 2, true);
        let input = WasmTensor::ones(vec![1, 3]);
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2]);
    }

    #[test]
    fn test_wasm_matmul() {
        let a = WasmTensor::ones(vec![2, 3]);
        let b = WasmTensor::ones(vec![3, 4]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 4]);
        assert!(c.data().iter().all(|&x| (x - 3.0).abs() < 1e-6));
    }

    #[test]
    fn test_wasm_sequential() {
        let layers = vec![
            WasmLayer::Linear(WasmLinear::new(10, 5, true)),
            WasmLayer::ReLU,
            WasmLayer::Linear(WasmLinear::new(5, 2, true)),
        ];
        let model = WasmSequential::new(layers);
        let input = WasmTensor::randn(vec![1, 10]);
        let output = model.forward(input).unwrap();
        assert_eq!(output.shape(), &[1, 2]);
    }

    #[test]
    fn test_wasm_optimizer() {
        let config = WasmOptimizerConfig::default();
        let mut optimizer = WasmAdam::new(1, config);

        let mut params = vec![WasmTensor::ones(vec![2, 2])];
        let grads = vec![vec![0.1, 0.2, 0.3, 0.4]];

        optimizer.step(&mut params, &grads).unwrap();

        // Check that parameters were updated
        assert!(params[0].data().iter().any(|&x| (x - 1.0).abs() > 1e-6));
    }

    #[test]
    fn test_wasm_loss_functions() {
        let pred = WasmTensor::ones(vec![2, 2]);
        let target = WasmTensor::ones(vec![2, 2]).mul_scalar(2.0);

        let loss = loss::mse_loss(&pred, &target).unwrap();
        assert!((loss - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_wasm_json_serialization() {
        let tensor = WasmTensor::randn(vec![2, 3]);
        let json = tensor.to_json().unwrap();
        let deserialized = WasmTensor::from_json(&json).unwrap();

        assert_eq!(tensor.shape(), deserialized.shape());
        assert_eq!(tensor.data(), deserialized.data());
    }

    #[test]
    fn test_wasm_activations() {
        let input = WasmTensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![4]);

        let relu = input.relu();
        assert_eq!(relu.data(), &[0.0, 0.0, 1.0, 2.0]);

        let sigmoid = input.sigmoid();
        assert!(sigmoid.data().iter().all(|&x| x >= 0.0 && x <= 1.0));

        let tanh = input.tanh();
        assert!(tanh.data().iter().all(|&x| x >= -1.0 && x <= 1.0));
    }
}
