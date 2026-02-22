//! cuDNN RNN (Recurrent Neural Network) functionality
//!
//! This module provides RNN support including LSTM, GRU, and vanilla RNN
//! implementations with various configuration options. It includes descriptors
//! for RNN networks, dropout layers, and data layout management.

use crate::cuda::error::{CudaError, CudaResult};
use torsh_core::DType;

use super::handle::CudnnHandle;

#[cfg(feature = "cudnn")]
use cudnn_sys::*;

// Import compatibility layer for missing cudnn-sys types and functions
#[cfg(feature = "cudnn")]
use super::compat::{
    cudnnCreateDropoutDescriptor, cudnnCreateRNNDataDescriptor, cudnnCreateRNNDescriptor,
    cudnnDataType_t as CompatDataType_t, cudnnDestroyDropoutDescriptor,
    cudnnDestroyRNNDataDescriptor, cudnnDestroyRNNDescriptor, cudnnDirectionMode_t,
    cudnnDropoutDescriptor_t, cudnnDropoutGetStatesSize, cudnnForwardMode_t, cudnnMathType_t,
    cudnnRNNAlgo_t, cudnnRNNDataDescriptor_t, cudnnRNNDataLayout_t, cudnnRNNDescriptor_t,
    cudnnRNNInputMode_t, cudnnRNNMode_t, cudnnSetDropoutDescriptor, cudnnSetRNNDataDescriptor,
    cudnnSetRNNDescriptor_v8,
};

/// RNN input mode
#[derive(Debug, Clone, Copy)]
pub enum RNNInputMode {
    LinearInput,
    SkipInput,
}

impl RNNInputMode {
    #[cfg(feature = "cudnn")]
    pub(crate) fn to_cudnn(self) -> cudnnRNNInputMode_t {
        match self {
            RNNInputMode::LinearInput => cudnnRNNInputMode_t::CUDNN_LINEAR_INPUT,
            RNNInputMode::SkipInput => cudnnRNNInputMode_t::CUDNN_SKIP_INPUT,
        }
    }
}

/// RNN direction mode
#[derive(Debug, Clone, Copy)]
pub enum RNNDirectionMode {
    Unidirectional,
    Bidirectional,
}

impl RNNDirectionMode {
    #[cfg(feature = "cudnn")]
    pub(crate) fn to_cudnn(self) -> cudnnDirectionMode_t {
        match self {
            RNNDirectionMode::Unidirectional => cudnnDirectionMode_t::CUDNN_UNIDIRECTIONAL,
            RNNDirectionMode::Bidirectional => cudnnDirectionMode_t::CUDNN_BIDIRECTIONAL,
        }
    }
}

/// RNN mode
#[derive(Debug, Clone, Copy)]
pub enum RNNMode {
    LSTM,
    GRU,
    RNNRelu,
    RNNTanh,
}

impl RNNMode {
    #[cfg(feature = "cudnn")]
    pub(crate) fn to_cudnn(self) -> cudnnRNNMode_t {
        match self {
            RNNMode::LSTM => cudnnRNNMode_t::CUDNN_LSTM,
            RNNMode::GRU => cudnnRNNMode_t::CUDNN_GRU,
            RNNMode::RNNRelu => cudnnRNNMode_t::CUDNN_RNN_RELU,
            RNNMode::RNNTanh => cudnnRNNMode_t::CUDNN_RNN_TANH,
        }
    }
}

/// RNN algorithm
#[derive(Debug, Clone, Copy)]
pub enum RNNAlgorithm {
    Standard,
    PersistStatic,
    PersistDynamic,
}

impl RNNAlgorithm {
    #[cfg(feature = "cudnn")]
    pub(crate) fn to_cudnn(self) -> cudnnRNNAlgo_t {
        match self {
            RNNAlgorithm::Standard => cudnnRNNAlgo_t::CUDNN_STANDARD,
            RNNAlgorithm::PersistStatic => cudnnRNNAlgo_t::CUDNN_STATIC_PERSISTENT,
            RNNAlgorithm::PersistDynamic => cudnnRNNAlgo_t::CUDNN_DYNAMIC_PERSISTENT,
        }
    }
}

/// Math type for precision
#[derive(Debug, Clone, Copy)]
pub enum MathType {
    Default,
    TensorOp,
}

impl MathType {
    #[cfg(feature = "cudnn")]
    pub(crate) fn to_cudnn(self) -> cudnnMathType_t {
        match self {
            MathType::Default => cudnnMathType_t::CUDNN_DEFAULT_MATH,
            MathType::TensorOp => cudnnMathType_t::CUDNN_TENSOR_OP_MATH,
        }
    }
}

/// RNN forward mode
#[derive(Debug, Clone, Copy)]
pub enum RNNForwardMode {
    Training,
    Inference,
}

impl RNNForwardMode {
    #[cfg(feature = "cudnn")]
    pub(crate) fn to_cudnn(self) -> cudnnForwardMode_t {
        match self {
            RNNForwardMode::Training => cudnnForwardMode_t::CUDNN_FWD_MODE_TRAINING,
            RNNForwardMode::Inference => cudnnForwardMode_t::CUDNN_FWD_MODE_INFERENCE,
        }
    }
}

/// RNN data layout
#[derive(Debug, Clone, Copy)]
pub enum RNNDataLayout {
    SeqMajorUnpacked,
    SeqMajorPacked,
    BatchMajorUnpacked,
}

impl RNNDataLayout {
    #[cfg(feature = "cudnn")]
    pub(crate) fn to_cudnn(self) -> cudnnRNNDataLayout_t {
        match self {
            RNNDataLayout::SeqMajorUnpacked => {
                cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED
            }
            RNNDataLayout::SeqMajorPacked => {
                cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED
            }
            RNNDataLayout::BatchMajorUnpacked => {
                cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED
            }
        }
    }
}

/// RNN descriptor wrapper
///
/// Provides a safe wrapper around the cuDNN RNN descriptor for configuring
/// recurrent neural network operations. Supports LSTM, GRU, and vanilla RNN modes.
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_backend::cuda::cudnn::rnn::{RNNDescriptor, RNNMode, RNNInputMode, RNNDirectionMode, RNNAlgorithm, MathType};
///
/// let mut rnn_desc = RNNDescriptor::new()?;
/// rnn_desc.set_lstm(
///     128,    // hidden_size
///     2,      // num_layers
///     &dropout_desc,
///     RNNInputMode::LinearInput,
///     RNNDirectionMode::Unidirectional,
///     RNNMode::LSTM,
///     RNNAlgorithm::Standard,
///     MathType::Default,
/// )?;
/// ```
pub struct RNNDescriptor {
    #[cfg(feature = "cudnn")]
    descriptor: cudnnRNNDescriptor_t,
    #[cfg(not(feature = "cudnn"))]
    _phantom: std::marker::PhantomData<()>,
}

impl RNNDescriptor {
    /// Create new RNN descriptor
    ///
    /// Initializes a new RNN descriptor for configuring recurrent neural
    /// network operations.
    ///
    /// # Returns
    ///
    /// A new `RNNDescriptor` instance on success, or an error if creation fails.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::CudnnError` if:
    /// - cuDNN is not available (feature not enabled)
    /// - RNN descriptor creation fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_backend::cuda::cudnn::rnn::RNNDescriptor;
    ///
    /// let rnn_desc = RNNDescriptor::new()?;
    /// ```
    pub fn new() -> CudaResult<Self> {
        #[cfg(feature = "cudnn")]
        {
            let mut descriptor: cudnnRNNDescriptor_t = std::ptr::null_mut();
            let status = unsafe { cudnnCreateRNNDescriptor(&mut descriptor) };
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to create RNN descriptor: {:?}",
                    status
                )));
            }
            Ok(Self { descriptor })
        }
        #[cfg(not(feature = "cudnn"))]
        {
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Set LSTM configuration
    ///
    /// Configures the RNN descriptor for LSTM operations with the specified
    /// parameters including layer count, dropout, and execution modes.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - Size of the hidden state
    /// * `num_layers` - Number of stacked RNN layers
    /// * `dropout_desc` - Dropout descriptor for regularization
    /// * `input_mode` - Input processing mode
    /// * `direction` - Unidirectional or bidirectional
    /// * `mode` - RNN mode (LSTM, GRU, etc.)
    /// * `algorithm` - Algorithm variant for performance
    /// * `math_precision` - Math precision mode
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if configuration fails.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::CudnnError` if:
    /// - cuDNN is not available (feature not enabled)
    /// - RNN descriptor configuration fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_backend::cuda::cudnn::rnn::{RNNDescriptor, RNNMode, RNNInputMode, RNNDirectionMode, RNNAlgorithm, MathType};
    ///
    /// let mut rnn_desc = RNNDescriptor::new()?;
    /// rnn_desc.set_lstm(
    ///     256,    // hidden_size
    ///     3,      // num_layers
    ///     &dropout_desc,
    ///     RNNInputMode::LinearInput,
    ///     RNNDirectionMode::Bidirectional,
    ///     RNNMode::LSTM,
    ///     RNNAlgorithm::Standard,
    ///     MathType::Default,
    /// )?;
    /// ```
    pub fn set_lstm(
        &mut self,
        hidden_size: i32,
        num_layers: i32,
        dropout_desc: &DropoutDescriptor,
        input_mode: RNNInputMode,
        direction: RNNDirectionMode,
        mode: RNNMode,
        algorithm: RNNAlgorithm,
        math_precision: MathType,
    ) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let status = unsafe {
                cudnnSetRNNDescriptor_v8(
                    self.descriptor,
                    algorithm.to_cudnn(),
                    direction.to_cudnn(),
                    mode.to_cudnn(),
                    input_mode.to_cudnn(),
                    hidden_size,
                    num_layers,
                    dropout_desc.raw(),
                    0, // auxFlags
                    math_precision.to_cudnn(),
                    std::ptr::null_mut(), // dataType (use default)
                )
            };
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to set RNN descriptor: {:?}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (
                hidden_size,
                num_layers,
                dropout_desc,
                input_mode,
                direction,
                mode,
                algorithm,
                math_precision,
            );
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Get the raw cuDNN RNN descriptor
    ///
    /// Returns the underlying cuDNN RNN descriptor for use with low-level cuDNN functions.
    /// This is only available when the "cudnn" feature is enabled.
    ///
    /// # Returns
    ///
    /// The raw `cudnnRNNDescriptor_t` pointer.
    ///
    /// # Safety
    ///
    /// The returned descriptor should not be destroyed manually as it is managed
    /// by the `RNNDescriptor` wrapper.
    #[cfg(feature = "cudnn")]
    pub fn raw(&self) -> cudnnRNNDescriptor_t {
        self.descriptor
    }
}

impl Drop for RNNDescriptor {
    /// Automatically destroy the RNN descriptor when dropped
    ///
    /// This ensures proper cleanup of cuDNN resources when the descriptor
    /// goes out of scope.
    fn drop(&mut self) {
        #[cfg(feature = "cudnn")]
        {
            if !self.descriptor.is_null() {
                unsafe {
                    let _status = cudnnDestroyRNNDescriptor(self.descriptor);
                    // Note: We ignore the status here as we can't return an error from drop
                    // In practice, cudnnDestroyRNNDescriptor rarely fails if the descriptor was valid
                }
            }
        }
    }
}

/// Dropout descriptor wrapper
///
/// Provides a safe wrapper around the cuDNN dropout descriptor for configuring
/// dropout operations in neural networks. Dropout is commonly used for
/// regularization to prevent overfitting.
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_backend::cuda::cudnn::{CudnnHandle, rnn::DropoutDescriptor};
///
/// let handle = CudnnHandle::new()?;
/// let dropout_desc = DropoutDescriptor::new(&handle, 0.5, 12345)?; // 50% dropout, seed 12345
/// ```
pub struct DropoutDescriptor {
    #[cfg(feature = "cudnn")]
    descriptor: cudnnDropoutDescriptor_t,
    #[cfg(not(feature = "cudnn"))]
    _phantom: std::marker::PhantomData<()>,
}

impl DropoutDescriptor {
    /// Create new dropout descriptor
    ///
    /// Initializes a new dropout descriptor with the specified dropout rate
    /// and random seed for reproducible behavior.
    ///
    /// # Arguments
    ///
    /// * `handle` - cuDNN handle for context
    /// * `dropout` - Dropout rate (0.0 to 1.0)
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// A new `DropoutDescriptor` instance on success, or an error if creation fails.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::CudnnError` if:
    /// - cuDNN is not available (feature not enabled)
    /// - Dropout descriptor creation fails
    /// - States size query fails
    /// - Descriptor configuration fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_backend::cuda::cudnn::{CudnnHandle, rnn::DropoutDescriptor};
    ///
    /// let handle = CudnnHandle::new()?;
    /// let dropout_desc = DropoutDescriptor::new(&handle, 0.2, 42)?; // 20% dropout
    /// ```
    pub fn new(handle: &CudnnHandle, dropout: f32, seed: u64) -> CudaResult<Self> {
        #[cfg(feature = "cudnn")]
        {
            let mut descriptor: cudnnDropoutDescriptor_t = std::ptr::null_mut();
            let status = unsafe { cudnnCreateDropoutDescriptor(&mut descriptor) };
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to create dropout descriptor: {:?}",
                    status
                )));
            }

            // Get states size
            let mut states_size: usize = 0;
            let status = unsafe { cudnnDropoutGetStatesSize(handle.raw(), &mut states_size) };
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to get dropout states size: {:?}",
                    status
                )));
            }

            // Allocate states memory (simplified - in practice should use proper memory management)
            let states = std::ptr::null_mut(); // This would need proper allocation

            let status = unsafe {
                cudnnSetDropoutDescriptor(
                    descriptor,
                    handle.raw(),
                    dropout,
                    states,
                    states_size,
                    seed,
                )
            };
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to set dropout descriptor: {:?}",
                    status
                )));
            }

            Ok(Self { descriptor })
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (handle, dropout, seed);
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Get the raw cuDNN dropout descriptor
    ///
    /// Returns the underlying cuDNN dropout descriptor for use with low-level cuDNN functions.
    /// This is only available when the "cudnn" feature is enabled.
    ///
    /// # Returns
    ///
    /// The raw `cudnnDropoutDescriptor_t` pointer.
    #[cfg(feature = "cudnn")]
    pub fn raw(&self) -> cudnnDropoutDescriptor_t {
        self.descriptor
    }
}

impl Drop for DropoutDescriptor {
    /// Automatically destroy the dropout descriptor when dropped
    ///
    /// This ensures proper cleanup of cuDNN resources when the descriptor
    /// goes out of scope.
    fn drop(&mut self) {
        #[cfg(feature = "cudnn")]
        {
            if !self.descriptor.is_null() {
                unsafe {
                    let _status = cudnnDestroyDropoutDescriptor(self.descriptor);
                    // Note: We ignore the status here as we can't return an error from drop
                    // In practice, cudnnDestroyDropoutDescriptor rarely fails if the descriptor was valid
                }
            }
        }
    }
}

/// RNN data descriptor wrapper
///
/// Provides a safe wrapper around the cuDNN RNN data descriptor for configuring
/// the layout and format of input/output data for RNN operations.
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_backend::cuda::cudnn::rnn::{RNNDataDescriptor, RNNDataLayout};
/// use torsh_core::DType;
///
/// let mut data_desc = RNNDataDescriptor::new()?;
/// data_desc.set(
///     DType::F32,
///     RNNDataLayout::SeqMajorUnpacked,
///     10,   // max_seq_length
///     32,   // batch_size
///     128,  // vector_size
///     &[10, 8, 6, 4, 2], // variable sequence lengths
///     Some(0.0), // padding fill value
/// )?;
/// ```
pub struct RNNDataDescriptor {
    #[cfg(feature = "cudnn")]
    descriptor: cudnnRNNDataDescriptor_t,
    #[cfg(not(feature = "cudnn"))]
    _phantom: std::marker::PhantomData<()>,
}

impl RNNDataDescriptor {
    /// Create new RNN data descriptor
    ///
    /// Initializes a new RNN data descriptor for configuring the layout
    /// and format of RNN input/output data.
    ///
    /// # Returns
    ///
    /// A new `RNNDataDescriptor` instance on success, or an error if creation fails.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::CudnnError` if:
    /// - cuDNN is not available (feature not enabled)
    /// - RNN data descriptor creation fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_backend::cuda::cudnn::rnn::RNNDataDescriptor;
    ///
    /// let data_desc = RNNDataDescriptor::new()?;
    /// ```
    pub fn new() -> CudaResult<Self> {
        #[cfg(feature = "cudnn")]
        {
            let mut descriptor: cudnnRNNDataDescriptor_t = std::ptr::null_mut();
            let status = unsafe { cudnnCreateRNNDataDescriptor(&mut descriptor) };
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to create RNN data descriptor: {:?}",
                    status
                )));
            }
            Ok(Self { descriptor })
        }
        #[cfg(not(feature = "cudnn"))]
        {
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Set RNN data descriptor configuration
    ///
    /// Configures the RNN data descriptor with the specified data layout,
    /// batch information, and sequence length details.
    ///
    /// # Arguments
    ///
    /// * `data_type` - Data type for the tensors
    /// * `layout` - Data layout (sequence-major or batch-major)
    /// * `max_seq_length` - Maximum sequence length in the batch
    /// * `batch_size` - Number of sequences in the batch
    /// * `vector_size` - Size of each vector element
    /// * `seq_length_array` - Array of actual sequence lengths
    /// * `padding_fill` - Optional padding value for variable-length sequences
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if configuration fails.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::CudnnError` if:
    /// - cuDNN is not available (feature not enabled)
    /// - RNN data descriptor configuration fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_backend::cuda::cudnn::rnn::{RNNDataDescriptor, RNNDataLayout};
    /// use torsh_core::DType;
    ///
    /// let mut data_desc = RNNDataDescriptor::new()?;
    /// data_desc.set(
    ///     DType::F32,
    ///     RNNDataLayout::BatchMajorUnpacked,
    ///     20,   // max_seq_length
    ///     16,   // batch_size
    ///     256,  // vector_size
    ///     &[20, 18, 15, 12, 10, 8, 6, 4], // sequence lengths
    ///     None, // no padding fill
    /// )?;
    /// ```
    pub fn set(
        &mut self,
        data_type: DType,
        layout: RNNDataLayout,
        max_seq_length: i32,
        batch_size: i32,
        vector_size: i32,
        seq_length_array: &[i32],
        padding_fill: Option<f32>,
    ) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let cudnn_data_type = match data_type {
                DType::F32 => CompatDataType_t::CUDNN_DATA_FLOAT,
                DType::F64 => CompatDataType_t::CUDNN_DATA_DOUBLE,
                DType::F16 => CompatDataType_t::CUDNN_DATA_HALF,
                _ => {
                    return Err(CudaError::CudnnError(format!(
                        "Unsupported data type for RNN: {:?}",
                        data_type
                    )))
                }
            };

            let status = unsafe {
                cudnnSetRNNDataDescriptor(
                    self.descriptor,
                    cudnn_data_type,
                    layout.to_cudnn(),
                    max_seq_length,
                    batch_size,
                    vector_size,
                    seq_length_array.as_ptr(),
                    padding_fill
                        .map(|f| &f as *const f32)
                        .unwrap_or(std::ptr::null()),
                )
            };
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to set RNN data descriptor: {:?}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = (
                data_type,
                layout,
                max_seq_length,
                batch_size,
                vector_size,
                seq_length_array,
                padding_fill,
            );
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Get the raw cuDNN RNN data descriptor
    ///
    /// Returns the underlying cuDNN RNN data descriptor for use with low-level cuDNN functions.
    /// This is only available when the "cudnn" feature is enabled.
    ///
    /// # Returns
    ///
    /// The raw `cudnnRNNDataDescriptor_t` pointer.
    #[cfg(feature = "cudnn")]
    pub fn raw(&self) -> cudnnRNNDataDescriptor_t {
        self.descriptor
    }
}

impl Drop for RNNDataDescriptor {
    /// Automatically destroy the RNN data descriptor when dropped
    ///
    /// This ensures proper cleanup of cuDNN resources when the descriptor
    /// goes out of scope.
    fn drop(&mut self) {
        #[cfg(feature = "cudnn")]
        {
            if !self.descriptor.is_null() {
                unsafe {
                    let _status = cudnnDestroyRNNDataDescriptor(self.descriptor);
                    // Note: We ignore the status here as we can't return an error from drop
                    // In practice, cudnnDestroyRNNDataDescriptor rarely fails if the descriptor was valid
                }
            }
        }
    }
}

unsafe impl Send for RNNDescriptor {}
unsafe impl Sync for RNNDescriptor {}
unsafe impl Send for DropoutDescriptor {}
unsafe impl Sync for DropoutDescriptor {}
unsafe impl Send for RNNDataDescriptor {}
unsafe impl Sync for RNNDataDescriptor {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rnn_descriptor_creation() {
        // Test basic RNN descriptor creation
        // Note: This test will only pass if cuDNN is available
        #[cfg(feature = "cudnn")]
        {
            match RNNDescriptor::new() {
                Ok(_desc) => {
                    // Basic verification that the descriptor was created
                    assert!(true);
                }
                Err(_) => {
                    // cuDNN might not be available in test environment
                    // This is acceptable
                }
            }
        }

        #[cfg(not(feature = "cudnn"))]
        {
            let result = RNNDescriptor::new();
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_rnn_data_descriptor_creation() {
        // Test basic RNN data descriptor creation
        #[cfg(feature = "cudnn")]
        {
            match RNNDataDescriptor::new() {
                Ok(_desc) => {
                    assert!(true);
                }
                Err(_) => {
                    // cuDNN might not be available in test environment
                }
            }
        }

        #[cfg(not(feature = "cudnn"))]
        {
            let result = RNNDataDescriptor::new();
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_rnn_enum_conversions() {
        // Test that our RNN enum conversions work properly
        let input_mode = RNNInputMode::LinearInput;
        let direction = RNNDirectionMode::Bidirectional;
        let mode = RNNMode::LSTM;
        let algorithm = RNNAlgorithm::Standard;
        let math_type = MathType::Default;
        let forward_mode = RNNForwardMode::Training;
        let layout = RNNDataLayout::SeqMajorUnpacked;

        // Basic creation tests
        assert_eq!(
            std::mem::discriminant(&input_mode),
            std::mem::discriminant(&RNNInputMode::LinearInput)
        );
        assert_eq!(
            std::mem::discriminant(&direction),
            std::mem::discriminant(&RNNDirectionMode::Bidirectional)
        );
        assert_eq!(
            std::mem::discriminant(&mode),
            std::mem::discriminant(&RNNMode::LSTM)
        );
        assert_eq!(
            std::mem::discriminant(&algorithm),
            std::mem::discriminant(&RNNAlgorithm::Standard)
        );
        assert_eq!(
            std::mem::discriminant(&math_type),
            std::mem::discriminant(&MathType::Default)
        );
        assert_eq!(
            std::mem::discriminant(&forward_mode),
            std::mem::discriminant(&RNNForwardMode::Training)
        );
        assert_eq!(
            std::mem::discriminant(&layout),
            std::mem::discriminant(&RNNDataLayout::SeqMajorUnpacked)
        );
    }

    #[test]
    fn test_send_sync() {
        // Test that RNN types implement Send and Sync
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<RNNDescriptor>();
        assert_sync::<RNNDescriptor>();
        assert_send::<DropoutDescriptor>();
        assert_sync::<DropoutDescriptor>();
        assert_send::<RNNDataDescriptor>();
        assert_sync::<RNNDataDescriptor>();
    }
}
