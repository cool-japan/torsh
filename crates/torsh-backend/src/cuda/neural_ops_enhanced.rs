//! Enhanced neural network operations with cuDNN integration

use crate::cuda::stream::CudaStream;
use crate::cuda::tensor_cores::TensorCoreContext;
use crate::error::BackendResult;
use cust::prelude::DevicePointer;

#[cfg(feature = "cudnn")]
use crate::cuda::cudnn::CudnnOps;

#[cfg(feature = "cudnn")]
use crate::cuda::cudnn::types::{ActivationMode, PoolingMode};

/// Enhanced neural operations that can use cuDNN and Tensor Cores when available
pub struct EnhancedNeuralOps {
    #[cfg(feature = "cudnn")]
    cudnn: Option<CudnnOps>,
    tensor_cores: Option<TensorCoreContext>,
}

impl EnhancedNeuralOps {
    /// Create new enhanced neural operations
    pub fn new() -> BackendResult<Self> {
        // Try to initialize Tensor Cores (assuming Compute 7.0+ for now)
        let tensor_cores = {
            let context = TensorCoreContext::new(8, 0); // Ampere as default
            if context.is_enabled() {
                Some(context)
            } else {
                None
            }
        };

        #[cfg(feature = "cudnn")]
        {
            let cudnn = match CudnnOps::new() {
                Ok(ops) => Some(ops),
                Err(_) => None, // Fall back to custom kernels if cuDNN fails
            };
            Ok(Self {
                cudnn,
                tensor_cores,
            })
        }
        #[cfg(not(feature = "cudnn"))]
        {
            Ok(Self { tensor_cores })
        }
    }

    /// Check if cuDNN is available
    pub fn has_cudnn(&self) -> bool {
        #[cfg(feature = "cudnn")]
        {
            self.cudnn.is_some()
        }
        #[cfg(not(feature = "cudnn"))]
        {
            false
        }
    }

    /// Check if Tensor Cores are available
    pub fn has_tensor_cores(&self) -> bool {
        self.tensor_cores.is_some()
    }

    /// Get mutable reference to tensor core context
    pub fn tensor_cores_mut(&mut self) -> Option<&mut TensorCoreContext> {
        self.tensor_cores.as_mut()
    }

    /// Perform 2D convolution with optimal backend selection
    pub fn conv2d_forward(
        &self,
        input: DevicePointer<f32>,
        weight: DevicePointer<f32>,
        bias: Option<DevicePointer<f32>>,
        output: DevicePointer<f32>,
        input_shape: (i32, i32, i32, i32),  // (N, C, H, W)
        weight_shape: (i32, i32, i32, i32), // (K, C, H, W)
        output_shape: (i32, i32, i32, i32), // (N, K, H, W)
        padding: (i32, i32),
        stride: (i32, i32),
        dilation: (i32, i32),
        stream: &CudaStream,
    ) -> BackendResult<()> {
        #[cfg(feature = "cudnn")]
        {
            if let Some(ref cudnn_ops) = self.cudnn {
                // Use cuDNN for optimized convolution
                return cudnn_ops.conv2d_forward(
                    input,
                    weight,
                    bias,
                    output,
                    input_shape,
                    weight_shape,
                    output_shape,
                    padding,
                    stride,
                    dilation,
                );
            }
        }

        // Fall back to custom CUDA kernels
        self.conv2d_fallback(
            input,
            weight,
            bias,
            output,
            input_shape,
            weight_shape,
            output_shape,
            padding,
            stride,
            dilation,
            stream,
        )
    }

    /// Fallback convolution using custom CUDA kernels
    #[allow(unused_variables)]
    fn conv2d_fallback(
        &self,
        input: DevicePointer<f32>,
        weight: DevicePointer<f32>,
        bias: Option<DevicePointer<f32>>,
        output: DevicePointer<f32>,
        input_shape: (i32, i32, i32, i32),
        weight_shape: (i32, i32, i32, i32),
        output_shape: (i32, i32, i32, i32),
        padding: (i32, i32),
        stride: (i32, i32),
        dilation: (i32, i32),
        stream: &CudaStream,
    ) -> BackendResult<()> {
        // Fallback convolution kernels not yet implemented
        // Requires custom CUDA kernel implementation
        Err(crate::error::BackendError::NotImplemented(
            "Convolution fallback kernel not implemented (requires cuDNN)".to_string(),
        ))
    }

    /// Perform ReLU activation with optimal backend selection
    pub fn relu_forward(
        &self,
        input: DevicePointer<f32>,
        output: DevicePointer<f32>,
        shape: (i32, i32, i32, i32),
    ) -> BackendResult<()> {
        #[cfg(feature = "cudnn")]
        {
            if let Some(ref cudnn_ops) = self.cudnn {
                // Use cuDNN for activation
                return cudnn_ops.activation_forward(ActivationMode::Relu, input, output, shape);
            }
        }

        // Fall back to custom CUDA kernels
        self.relu_fallback(input, output, shape)
    }

    /// Fallback ReLU using custom CUDA kernels
    #[allow(unused_variables)]
    fn relu_fallback(
        &self,
        input: DevicePointer<f32>,
        output: DevicePointer<f32>,
        shape: (i32, i32, i32, i32),
    ) -> BackendResult<()> {
        // Fallback ReLU kernels not yet implemented
        Err(crate::error::BackendError::NotImplemented(
            "ReLU fallback kernel not implemented (requires cuDNN)".to_string(),
        ))
    }

    /// Perform sigmoid activation with optimal backend selection
    pub fn sigmoid_forward(
        &self,
        input: DevicePointer<f32>,
        output: DevicePointer<f32>,
        shape: (i32, i32, i32, i32),
    ) -> BackendResult<()> {
        #[cfg(feature = "cudnn")]
        {
            if let Some(ref cudnn_ops) = self.cudnn {
                // Use cuDNN for activation
                return cudnn_ops.activation_forward(ActivationMode::Sigmoid, input, output, shape);
            }
        }

        // Fall back to custom CUDA kernels
        self.sigmoid_fallback(input, output, shape)
    }

    /// Fallback sigmoid using custom CUDA kernels
    #[allow(unused_variables)]
    fn sigmoid_fallback(
        &self,
        input: DevicePointer<f32>,
        output: DevicePointer<f32>,
        shape: (i32, i32, i32, i32),
    ) -> BackendResult<()> {
        // Fallback sigmoid kernels not yet implemented
        Err(crate::error::BackendError::NotImplemented(
            "Sigmoid fallback kernel not implemented (requires cuDNN)".to_string(),
        ))
    }

    /// Perform tanh activation with optimal backend selection
    pub fn tanh_forward(
        &self,
        input: DevicePointer<f32>,
        output: DevicePointer<f32>,
        shape: (i32, i32, i32, i32),
    ) -> BackendResult<()> {
        #[cfg(feature = "cudnn")]
        {
            if let Some(ref cudnn_ops) = self.cudnn {
                // Use cuDNN for activation
                return cudnn_ops.activation_forward(ActivationMode::Tanh, input, output, shape);
            }
        }

        // Fall back to custom CUDA kernels
        self.tanh_fallback(input, output, shape)
    }

    /// Fallback tanh using custom CUDA kernels
    #[allow(unused_variables)]
    fn tanh_fallback(
        &self,
        input: DevicePointer<f32>,
        output: DevicePointer<f32>,
        shape: (i32, i32, i32, i32),
    ) -> BackendResult<()> {
        // Fallback tanh kernels not yet implemented
        Err(crate::error::BackendError::NotImplemented(
            "Tanh fallback kernel not implemented (requires cuDNN)".to_string(),
        ))
    }

    /// Perform 2D max pooling with optimal backend selection
    pub fn maxpool2d_forward(
        &self,
        input: DevicePointer<f32>,
        output: DevicePointer<f32>,
        input_shape: (i32, i32, i32, i32),  // (N, C, H, W)
        output_shape: (i32, i32, i32, i32), // (N, C, H_out, W_out)
        kernel_size: (i32, i32),
        padding: (i32, i32),
        stride: (i32, i32),
        _stream: &CudaStream,
    ) -> BackendResult<()> {
        #[cfg(feature = "cudnn")]
        {
            if let Some(ref cudnn_ops) = self.cudnn {
                // Use cuDNN for optimized pooling
                return cudnn_ops.pooling2d_forward(
                    PoolingMode::Max,
                    input,
                    output,
                    input_shape,
                    output_shape,
                    kernel_size,
                    padding,
                    stride,
                );
            }
        }

        // Fallback maxpool kernels not yet implemented
        Err(crate::error::BackendError::NotImplemented(
            "MaxPool2D fallback kernel not implemented (requires cuDNN)".to_string(),
        ))
    }

    /// Perform 2D batch normalization with optimal backend selection
    pub fn batchnorm2d_forward(
        &self,
        input: DevicePointer<f32>,
        output: DevicePointer<f32>,
        weight: DevicePointer<f32>,
        bias: DevicePointer<f32>,
        running_mean: DevicePointer<f32>,
        running_var: DevicePointer<f32>,
        shape: (i32, i32, i32, i32), // (N, C, H, W)
        eps: f32,
        momentum: f32,
        training: bool,
        _stream: &CudaStream,
    ) -> BackendResult<()> {
        #[cfg(feature = "cudnn")]
        {
            if let Some(ref cudnn_ops) = self.cudnn {
                // Use cuDNN for optimized batch normalization
                return cudnn_ops.batchnorm_forward(
                    input,
                    output,
                    weight,
                    bias,
                    running_mean,
                    running_var,
                    eps as f64,
                    momentum as f64,
                    shape,
                    training,
                );
            }
        }

        // Fallback batchnorm kernels not yet implemented
        Err(crate::error::BackendError::NotImplemented(
            "BatchNorm2D fallback kernel not implemented (requires cuDNN)".to_string(),
        ))
    }

    /// Perform softmax
    #[allow(unused_variables)]
    pub fn softmax_forward(
        &self,
        input: DevicePointer<f32>,
        output: DevicePointer<f32>,
        batch_size: i32,
        classes: i32,
        stream: &CudaStream,
    ) -> BackendResult<()> {
        // Fallback softmax kernels not yet implemented
        Err(crate::error::BackendError::NotImplemented(
            "Softmax fallback kernel not implemented (requires custom kernel)".to_string(),
        ))
    }
}

impl Default for EnhancedNeuralOps {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            #[cfg(feature = "cudnn")]
            {
                Self {
                    cudnn: None,
                    tensor_cores: None,
                }
            }
            #[cfg(not(feature = "cudnn"))]
            {
                Self { tensor_cores: None }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_neural_ops_creation() {
        let ops = EnhancedNeuralOps::new();
        assert!(ops.is_ok());
    }

    #[test]
    fn test_cudnn_availability() {
        if let Ok(ops) = EnhancedNeuralOps::new() {
            // Test will depend on whether cuDNN is available in the environment
            let _has_cudnn = ops.has_cudnn();
        }
    }
}
