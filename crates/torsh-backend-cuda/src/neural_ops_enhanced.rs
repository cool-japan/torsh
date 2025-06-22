//! Enhanced neural network operations with cuDNN integration

use crate::error::{CudaError, CudaResult};
use crate::stream::CudaStream;

#[cfg(feature = "cudnn")]
use crate::cudnn::{CudnnOps, ActivationMode};

/// Enhanced neural operations that can use cuDNN when available
pub struct EnhancedNeuralOps {
    #[cfg(feature = "cudnn")]
    cudnn: Option<CudnnOps>,
}

impl EnhancedNeuralOps {
    /// Create new enhanced neural operations
    pub fn new() -> CudaResult<Self> {
        #[cfg(feature = "cudnn")] 
        {
            let cudnn = match CudnnOps::new() {
                Ok(ops) => Some(ops),
                Err(_) => None, // Fall back to custom kernels if cuDNN fails
            };
            Ok(Self { cudnn })
        }
        #[cfg(not(feature = "cudnn"))]
        {
            Ok(Self {})
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

    /// Perform 2D convolution with optimal backend selection
    pub fn conv2d_forward(
        &self,
        input: cust::DevicePointer<f32>,
        weight: cust::DevicePointer<f32>,
        bias: Option<cust::DevicePointer<f32>>,
        output: cust::DevicePointer<f32>,
        input_shape: (i32, i32, i32, i32), // (N, C, H, W)
        weight_shape: (i32, i32, i32, i32), // (K, C, H, W) 
        output_shape: (i32, i32, i32, i32), // (N, K, H, W)
        padding: (i32, i32),
        stride: (i32, i32),
        dilation: (i32, i32),
        stream: &CudaStream,
    ) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            if let Some(ref cudnn_ops) = self.cudnn {
                // Use cuDNN for optimized convolution
                return cudnn_ops.conv2d_forward(
                    input, weight, bias, output,
                    input_shape, weight_shape, output_shape,
                    padding, stride, dilation,
                );
            }
        }

        // Fall back to custom CUDA kernels
        self.conv2d_fallback(
            input, weight, bias, output,
            input_shape, weight_shape, output_shape,
            padding, stride, dilation, stream,
        )
    }

    /// Fallback convolution using custom CUDA kernels
    fn conv2d_fallback(
        &self,
        input: cust::DevicePointer<f32>,
        weight: cust::DevicePointer<f32>,
        bias: Option<cust::DevicePointer<f32>>,
        output: cust::DevicePointer<f32>,
        input_shape: (i32, i32, i32, i32),
        weight_shape: (i32, i32, i32, i32),
        output_shape: (i32, i32, i32, i32),
        padding: (i32, i32),
        stride: (i32, i32),
        dilation: (i32, i32),
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let bias_ptr = bias.map(|b| b.as_raw_mut()).unwrap_or(std::ptr::null_mut());
        
        crate::kernels::neural_ops::launch_conv2d_f32(
            input.as_raw_mut(),
            weight.as_raw_mut(),
            bias_ptr,
            output.as_raw_mut(),
            input_shape.0,  // batch_size
            input_shape.1,  // in_channels
            weight_shape.0, // out_channels
            input_shape.2,  // input_height
            input_shape.3,  // input_width
            weight_shape.2, // kernel_height
            weight_shape.3, // kernel_width
            padding.0,      // pad_h
            padding.1,      // pad_w
            stride.0,       // stride_h
            stride.1,       // stride_w
            dilation.0,     // dilation_h
            dilation.1,     // dilation_w
            stream.raw() as cuda_sys::CUstream,
        );
        
        Ok(())
    }

    /// Perform ReLU activation with optimal backend selection
    pub fn relu_forward(
        &self,
        input: cust::DevicePointer<f32>,
        output: cust::DevicePointer<f32>,
        shape: (i32, i32, i32, i32),
    ) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            if let Some(ref cudnn_ops) = self.cudnn {
                // Use cuDNN for activation
                return cudnn_ops.activation_forward(
                    ActivationMode::Relu,
                    input,
                    output,
                    shape,
                );
            }
        }

        // Fall back to custom CUDA kernels
        self.relu_fallback(input, output, shape)
    }

    /// Fallback ReLU using custom CUDA kernels
    fn relu_fallback(
        &self,
        input: cust::DevicePointer<f32>,
        output: cust::DevicePointer<f32>,
        shape: (i32, i32, i32, i32),
    ) -> CudaResult<()> {
        let size = (shape.0 * shape.1 * shape.2 * shape.3) as usize;
        
        // Use tensor operations for element-wise ReLU
        crate::kernels::tensor_ops::launch_relu_f32(
            input.as_raw_mut(),
            output.as_raw_mut(),
            size,
            std::ptr::null_mut() as cuda_sys::CUstream,
        );
        
        Ok(())
    }

    /// Perform sigmoid activation with optimal backend selection
    pub fn sigmoid_forward(
        &self,
        input: cust::DevicePointer<f32>,
        output: cust::DevicePointer<f32>,
        shape: (i32, i32, i32, i32),
    ) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            if let Some(ref cudnn_ops) = self.cudnn {
                // Use cuDNN for activation
                return cudnn_ops.activation_forward(
                    ActivationMode::Sigmoid,
                    input,
                    output,
                    shape,
                );
            }
        }

        // Fall back to custom CUDA kernels
        self.sigmoid_fallback(input, output, shape)
    }

    /// Fallback sigmoid using custom CUDA kernels
    fn sigmoid_fallback(
        &self,
        input: cust::DevicePointer<f32>,
        output: cust::DevicePointer<f32>,
        shape: (i32, i32, i32, i32),
    ) -> CudaResult<()> {
        let size = (shape.0 * shape.1 * shape.2 * shape.3) as usize;
        
        // Use tensor operations for element-wise sigmoid
        crate::kernels::tensor_ops::launch_sigmoid_f32(
            input.as_raw_mut(),
            output.as_raw_mut(),
            size,
            std::ptr::null_mut() as cuda_sys::CUstream,
        );
        
        Ok(())
    }

    /// Perform tanh activation with optimal backend selection
    pub fn tanh_forward(
        &self,
        input: cust::DevicePointer<f32>,
        output: cust::DevicePointer<f32>,
        shape: (i32, i32, i32, i32),
    ) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            if let Some(ref cudnn_ops) = self.cudnn {
                // Use cuDNN for activation
                return cudnn_ops.activation_forward(
                    ActivationMode::Tanh,
                    input,
                    output,
                    shape,
                );
            }
        }

        // Fall back to custom CUDA kernels
        self.tanh_fallback(input, output, shape)
    }

    /// Fallback tanh using custom CUDA kernels
    fn tanh_fallback(
        &self,
        input: cust::DevicePointer<f32>,
        output: cust::DevicePointer<f32>,
        shape: (i32, i32, i32, i32),
    ) -> CudaResult<()> {
        let size = (shape.0 * shape.1 * shape.2 * shape.3) as usize;
        
        // Use tensor operations for element-wise tanh
        crate::kernels::tensor_ops::launch_tanh_f32(
            input.as_raw_mut(),
            output.as_raw_mut(),
            size,
            std::ptr::null_mut() as cuda_sys::CUstream,
        );
        
        Ok(())
    }

    /// Perform 2D max pooling
    pub fn maxpool2d_forward(
        &self,
        input: cust::DevicePointer<f32>,
        output: cust::DevicePointer<f32>,
        input_shape: (i32, i32, i32, i32), // (N, C, H, W)
        output_shape: (i32, i32, i32, i32), // (N, C, H_out, W_out)
        kernel_size: (i32, i32),
        padding: (i32, i32),
        stride: (i32, i32),
        stream: &CudaStream,
    ) -> CudaResult<()> {
        // Max pooling uses custom kernels (cuDNN pooling can be added later)
        crate::kernels::neural_ops::launch_maxpool2d_f32(
            input.as_raw_mut(),
            output.as_raw_mut(),
            input_shape.0,  // batch_size
            input_shape.1,  // channels
            input_shape.2,  // input_height
            input_shape.3,  // input_width
            output_shape.2, // output_height
            output_shape.3, // output_width
            kernel_size.0,  // kernel_height
            kernel_size.1,  // kernel_width
            padding.0,      // pad_h
            padding.1,      // pad_w
            stride.0,       // stride_h
            stride.1,       // stride_w
            stream.raw() as cuda_sys::CUstream,
        );
        
        Ok(())
    }

    /// Perform 2D batch normalization
    pub fn batchnorm2d_forward(
        &self,
        input: cust::DevicePointer<f32>,
        output: cust::DevicePointer<f32>,
        weight: cust::DevicePointer<f32>,
        bias: cust::DevicePointer<f32>,
        running_mean: cust::DevicePointer<f32>,
        running_var: cust::DevicePointer<f32>,
        shape: (i32, i32, i32, i32), // (N, C, H, W)
        eps: f32,
        momentum: f32,
        training: bool,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        // Batch normalization uses custom kernels
        crate::kernels::neural_ops::launch_batchnorm2d_f32(
            input.as_raw_mut(),
            output.as_raw_mut(),
            weight.as_raw_mut(),
            bias.as_raw_mut(),
            running_mean.as_raw_mut(),
            running_var.as_raw_mut(),
            shape.0, // batch_size
            shape.1, // channels
            shape.2, // height
            shape.3, // width
            eps,
            momentum,
            training,
            stream.raw() as cuda_sys::CUstream,
        );
        
        Ok(())
    }

    /// Perform softmax
    pub fn softmax_forward(
        &self,
        input: cust::DevicePointer<f32>,
        output: cust::DevicePointer<f32>,
        batch_size: i32,
        classes: i32,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        // Softmax uses custom kernels
        crate::kernels::neural_ops::launch_softmax_f32(
            input.as_raw_mut(),
            output.as_raw_mut(),
            batch_size,
            classes,
            stream.raw() as cuda_sys::CUstream,
        );
        
        Ok(())
    }
}

impl Default for EnhancedNeuralOps {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| {
            #[cfg(feature = "cudnn")]
            {
                Self { cudnn: None }
            }
            #[cfg(not(feature = "cudnn"))]
            {
                Self {}
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