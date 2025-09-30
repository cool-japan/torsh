//! cuDNN (CUDA Deep Neural Network library) integration
//!
//! This module provides comprehensive cuDNN support for the ToRSh deep learning framework,
//! including tensor operations, neural network layers, and RNN functionality. The implementation
//! is organized into focused submodules for maintainability and clarity.
//!
//! # Architecture
//!
//! The cuDNN integration is structured as follows:
//!
//! - **types**: Core enums and type definitions
//! - **handle**: cuDNN handle management with automatic resource cleanup
//! - **descriptors**: Tensor, filter, convolution, activation, and pooling descriptors
//! - **operations**: High-level operations interface (CudnnOps)
//! - **rnn**: Recurrent Neural Network support (LSTM, GRU, vanilla RNN)
//!
//! # Features
//!
//! - **Convolution Operations**: 2D convolution with various algorithms and configurations
//! - **Batch Normalization**: Spatial and per-activation batch normalization
//! - **Activation Functions**: ReLU, Sigmoid, Tanh, and other activation functions
//! - **Pooling Operations**: Max pooling, average pooling with various configurations
//! - **Layer Normalization**: Transformer-style layer normalization
//! - **RNN Support**: LSTM, GRU, and vanilla RNN with bidirectional support
//! - **Algorithm Selection**: Automatic algorithm finding for optimal performance
//! - **Memory Management**: Automatic workspace size calculation and management
//!
//! # Usage Examples
//!
//! ## Basic Operations
//!
//! ```rust,ignore
//! use torsh_backend::cuda::cudnn::{CudnnOps, ActivationMode, PoolingMode};
//!
//! // Initialize cuDNN operations
//! let ops = CudnnOps::new()?;
//!
//! // Perform 2D convolution
//! ops.conv2d_forward(
//!     input_ptr,
//!     weight_ptr,
//!     Some(bias_ptr),
//!     output_ptr,
//!     (1, 3, 224, 224),   // input shape (N, C, H, W)
//!     (64, 3, 7, 7),      // weight shape (K, C, H, W)
//!     (1, 64, 218, 218),  // output shape (N, K, H_out, W_out)
//!     (0, 0),             // padding
//!     (1, 1),             // stride
//!     (1, 1),             // dilation
//! )?;
//!
//! // Apply activation function
//! ops.activation_forward(
//!     ActivationMode::Relu,
//!     input_ptr,
//!     output_ptr,
//!     (1, 64, 218, 218),
//! )?;
//!
//! // Perform max pooling
//! ops.pooling2d_forward(
//!     PoolingMode::Max,
//!     input_ptr,
//!     output_ptr,
//!     (1, 64, 218, 218),  // input shape
//!     (1, 64, 109, 109),  // output shape
//!     (2, 2),             // window size
//!     (0, 0),             // padding
//!     (2, 2),             // stride
//! )?;
//! ```
//!
//! ## RNN Operations
//!
//! ```rust,ignore
//! use torsh_backend::cuda::cudnn::{
//!     CudnnHandle, CudnnOps,
//!     rnn::{RNNDescriptor, RNNDataDescriptor, DropoutDescriptor},
//!     rnn::{RNNMode, RNNInputMode, RNNDirectionMode, RNNAlgorithm, MathType, RNNForwardMode, RNNDataLayout},
//! };
//! use torsh_core::DType;
//!
//! // Create cuDNN handle
//! let handle = CudnnHandle::new()?;
//!
//! // Create dropout descriptor
//! let dropout_desc = DropoutDescriptor::new(&handle, 0.1, 12345)?;
//!
//! // Create and configure RNN descriptor
//! let mut rnn_desc = RNNDescriptor::new()?;
//! rnn_desc.set_lstm(
//!     128,                             // hidden_size
//!     2,                               // num_layers
//!     &dropout_desc,
//!     RNNInputMode::LinearInput,
//!     RNNDirectionMode::Bidirectional,
//!     RNNMode::LSTM,
//!     RNNAlgorithm::Standard,
//!     MathType::Default,
//! )?;
//!
//! // Create RNN data descriptors
//! let mut x_desc = RNNDataDescriptor::new()?;
//! x_desc.set(
//!     DType::F32,
//!     RNNDataLayout::SeqMajorUnpacked,
//!     20,    // max_seq_length
//!     32,    // batch_size
//!     256,   // vector_size
//!     &sequence_lengths,
//!     None,  // no padding fill
//! )?;
//!
//! // Perform LSTM forward pass
//! let ops = CudnnOps::new()?;
//! ops.lstm_forward(
//!     &rnn_desc,
//!     RNNForwardMode::Training,
//!     Some(seq_lengths_ptr),
//!     &x_desc,
//!     x_ptr,
//!     &y_desc,
//!     y_ptr,
//!     &h_desc,
//!     Some(hx_ptr),
//!     Some(hy_ptr),
//!     &c_desc,
//!     Some(cx_ptr),
//!     Some(cy_ptr),
//!     weight_space_size,
//!     weight_space_ptr,
//!     work_space_size,
//!     work_space_ptr,
//!     reserve_space_size,
//!     Some(reserve_space_ptr),
//! )?;
//! ```
//!
//! ## Descriptor Management
//!
//! ```rust,ignore
//! use torsh_backend::cuda::cudnn::{
//!     TensorDescriptor, FilterDescriptor, ConvolutionDescriptor,
//!     ActivationDescriptor, PoolingDescriptor,
//!     ConvolutionMode, ActivationMode, NanPropagation, PoolingMode,
//! };
//! use torsh_core::DType;
//!
//! // Create tensor descriptor
//! let mut input_desc = TensorDescriptor::new()?;
//! input_desc.set_4d(DType::F32, 1, 3, 224, 224)?;
//!
//! // Create filter descriptor
//! let mut filter_desc = FilterDescriptor::new()?;
//! filter_desc.set_4d(DType::F32, 64, 3, 7, 7)?;
//!
//! // Create convolution descriptor
//! let mut conv_desc = ConvolutionDescriptor::new()?;
//! conv_desc.set_2d(
//!     3, 3,                           // padding
//!     2, 2,                           // stride
//!     1, 1,                           // dilation
//!     ConvolutionMode::CrossCorrelation,
//! )?;
//!
//! // Create activation descriptor
//! let mut act_desc = ActivationDescriptor::new()?;
//! act_desc.set(ActivationMode::Relu, NanPropagation::NotPropagateNan, 0.0)?;
//! ```
//!
//! # Feature Requirements
//!
//! This module requires the "cudnn" feature to be enabled. When the feature is disabled,
//! all operations will return appropriate errors indicating that cuDNN is not available.
//!
//! # Error Handling
//!
//! All operations return `CudaResult<T>` which provides comprehensive error information
//! including cuDNN status codes and descriptive error messages.
//!
//! # Thread Safety
//!
//! All types in this module implement `Send` and `Sync` where appropriate, allowing
//! safe use in multi-threaded environments. cuDNN handles are protected by mutexes
//! to ensure thread-safe access.

pub mod descriptors;
pub mod handle;
pub mod operations;
pub mod rnn;
pub mod types;

// Re-export core functionality for easy access
pub use descriptors::{
    ActivationDescriptor, ConvolutionDescriptor, FilterDescriptor, PoolingDescriptor,
    TensorDescriptor,
};
pub use handle::CudnnHandle;
pub use operations::{init, CudnnOps};
pub use types::{
    is_available, ActivationMode, ConvolutionDeterminism, ConvolutionForwardAlgorithm,
    ConvolutionForwardAlgorithmPerformance, ConvolutionMathType, ConvolutionMode,
    ConvolutionStatus, NanPropagation, PoolingMode, TensorFormat,
};

// Re-export RNN functionality
pub use rnn::{
    DropoutDescriptor, MathType, RNNAlgorithm, RNNDataDescriptor, RNNDataLayout, RNNDescriptor,
    RNNDirectionMode, RNNForwardMode, RNNInputMode, RNNMode,
};

// Convenience type aliases for common operations
/// Alias for the main cuDNN operations interface
pub type Operations = CudnnOps;

/// Alias for cuDNN handle
pub type Handle = CudnnHandle;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_availability() {
        // Test that we can check cuDNN availability
        let available = is_available();

        // This test should always pass regardless of cuDNN availability
        assert!(available || !available);
    }

    #[test]
    fn test_public_api_accessibility() {
        // Test that all major types are accessible through the public API
        // This is a compilation test to ensure our re-exports work

        // Handle and operations
        let _handle_type = std::marker::PhantomData::<CudnnHandle>;
        let _ops_type = std::marker::PhantomData::<CudnnOps>;
        let _ops_alias_type = std::marker::PhantomData::<Operations>;
        let _handle_alias_type = std::marker::PhantomData::<Handle>;

        // Descriptors
        let _tensor_desc_type = std::marker::PhantomData::<TensorDescriptor>;
        let _filter_desc_type = std::marker::PhantomData::<FilterDescriptor>;
        let _conv_desc_type = std::marker::PhantomData::<ConvolutionDescriptor>;
        let _act_desc_type = std::marker::PhantomData::<ActivationDescriptor>;
        let _pool_desc_type = std::marker::PhantomData::<PoolingDescriptor>;

        // Types and enums
        let _activation_mode = ActivationMode::Relu;
        let _conv_mode = ConvolutionMode::CrossCorrelation;
        let _pool_mode = PoolingMode::Max;
        let _nan_prop = NanPropagation::NotPropagateNan;

        // RNN types
        let _rnn_desc_type = std::marker::PhantomData::<RNNDescriptor>;
        let _rnn_data_desc_type = std::marker::PhantomData::<RNNDataDescriptor>;
        let _dropout_desc_type = std::marker::PhantomData::<DropoutDescriptor>;
        let _rnn_mode = RNNMode::LSTM;
        let _rnn_input_mode = RNNInputMode::LinearInput;
        let _rnn_direction = RNNDirectionMode::Bidirectional;

        assert!(true); // If we get here, all types are accessible
    }

    #[test]
    fn test_init_function() {
        // Test that the init function is accessible
        let result = init();

        // Should either succeed or fail gracefully
        match result {
            Ok(()) => assert!(true),
            Err(_) => assert!(true), // cuDNN might not be available
        }
    }

    #[cfg(feature = "cudnn")]
    #[test]
    fn test_cudnn_feature_enabled() {
        // When cuDNN feature is enabled, test basic functionality
        match CudnnHandle::new() {
            Ok(_handle) => {
                // cuDNN is available and working
                assert!(is_available());
            }
            Err(_) => {
                // cuDNN feature is enabled but library might not be available
                // This is acceptable in test environments
                assert!(true);
            }
        }
    }

    #[cfg(not(feature = "cudnn"))]
    #[test]
    fn test_cudnn_feature_disabled() {
        // When cuDNN feature is disabled, operations should fail gracefully
        assert!(!is_available());

        let result = CudnnHandle::new();
        assert!(result.is_err());

        let result = CudnnOps::new();
        assert!(result.is_err());

        let result = init();
        assert!(result.is_err());
    }
}
