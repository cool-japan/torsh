//! Core types and enums for cuDNN operations
//!
//! This module provides fundamental data types, enums, and utility structures
//! used throughout the cuDNN integration system.

// Allow unreachable patterns for forward compatibility with cuDNN versions
#![allow(unreachable_patterns)]
// Allow unused imports as they are used conditionally with the cudnn feature
#![allow(unused_imports)]

#[cfg(feature = "cudnn")]
use cudnn_sys::*;

// Import compatibility types for missing cudnn-sys definitions
#[cfg(feature = "cudnn")]
use super::compat::{
    cudnnActivationMode_t, cudnnAddMode_t, cudnnBatchNormMode_t, cudnnConvolutionFwdAlgo_t,
    cudnnConvolutionMode_t, cudnnDataType_t, cudnnDirectionMode_t, cudnnForwardMode_t,
    cudnnMathType_t, cudnnNanPropagation_t, cudnnPoolingMode_t, cudnnRNNAlgo_t,
    cudnnRNNDataLayout_t, cudnnRNNInputMode_t, cudnnRNNMode_t, cudnnTensorFormat_t,
};

/// Convolution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvolutionMode {
    /// Cross-correlation mode
    CrossCorrelation,
    /// Convolution mode
    Convolution,
}

impl ConvolutionMode {
    #[cfg(feature = "cudnn")]
    pub(crate) fn to_cudnn(&self) -> cudnnConvolutionMode_t {
        match self {
            ConvolutionMode::CrossCorrelation => cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            ConvolutionMode::Convolution => cudnnConvolutionMode_t::CUDNN_CONVOLUTION,
        }
    }
}

/// Activation function modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationMode {
    /// Sigmoid activation
    Sigmoid,
    /// ReLU activation
    Relu,
    /// Tanh activation
    Tanh,
    /// Clipped ReLU activation
    ClippedRelu,
    /// ELU activation
    Elu,
    /// Identity activation (pass-through)
    Identity,
}

impl ActivationMode {
    #[cfg(feature = "cudnn")]
    pub(crate) fn to_cudnn(&self) -> cudnnActivationMode_t {
        match self {
            ActivationMode::Sigmoid => cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID,
            ActivationMode::Relu => cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
            ActivationMode::Tanh => cudnnActivationMode_t::CUDNN_ACTIVATION_TANH,
            ActivationMode::ClippedRelu => cudnnActivationMode_t::CUDNN_ACTIVATION_CLIPPED_RELU,
            ActivationMode::Elu => cudnnActivationMode_t::CUDNN_ACTIVATION_ELU,
            ActivationMode::Identity => cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY,
        }
    }
}

/// NaN propagation modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanPropagation {
    /// Propagate NaN values
    Propagate,
    /// Do not propagate NaN values
    NotPropagate,
}

impl NanPropagation {
    #[cfg(feature = "cudnn")]
    pub(crate) fn to_cudnn(&self) -> cudnnNanPropagation_t {
        match self {
            NanPropagation::Propagate => cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
            NanPropagation::NotPropagate => cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
        }
    }
}

/// Pooling modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingMode {
    /// Maximum pooling
    Max,
    /// Average pooling (including padding)
    AverageCountIncludePadding,
    /// Average pooling (excluding padding)
    AverageCountExcludePadding,
    /// Maximum pooling with index
    MaxDeterministic,
}

impl PoolingMode {
    #[cfg(feature = "cudnn")]
    pub(crate) fn to_cudnn(&self) -> cudnnPoolingMode_t {
        match self {
            PoolingMode::Max => cudnnPoolingMode_t::CUDNN_POOLING_MAX,
            PoolingMode::AverageCountIncludePadding => {
                cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
            }
            PoolingMode::AverageCountExcludePadding => {
                cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
            }
            PoolingMode::MaxDeterministic => cudnnPoolingMode_t::CUDNN_POOLING_MAX_DETERMINISTIC,
        }
    }
}

/// RNN input mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RNNInputMode {
    /// Linear input
    LinearInput,
    /// Skip input
    SkipInput,
}

impl RNNInputMode {
    #[cfg(feature = "cudnn")]
    pub(crate) fn to_cudnn(&self) -> cudnnRNNInputMode_t {
        match self {
            RNNInputMode::LinearInput => cudnnRNNInputMode_t::CUDNN_LINEAR_INPUT,
            RNNInputMode::SkipInput => cudnnRNNInputMode_t::CUDNN_SKIP_INPUT,
        }
    }

    #[cfg(feature = "cudnn")]
    pub(crate) fn from_cudnn(mode: cudnnRNNInputMode_t) -> Self {
        match mode {
            cudnnRNNInputMode_t::CUDNN_LINEAR_INPUT => RNNInputMode::LinearInput,
            cudnnRNNInputMode_t::CUDNN_SKIP_INPUT => RNNInputMode::SkipInput,
            _ => RNNInputMode::LinearInput, // default fallback
        }
    }
}

/// RNN direction mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RNNDirectionMode {
    /// Unidirectional RNN
    Unidirectional,
    /// Bidirectional RNN
    Bidirectional,
}

impl RNNDirectionMode {
    #[cfg(feature = "cudnn")]
    pub(crate) fn to_cudnn(&self) -> cudnnDirectionMode_t {
        match self {
            RNNDirectionMode::Unidirectional => cudnnDirectionMode_t::CUDNN_UNIDIRECTIONAL,
            RNNDirectionMode::Bidirectional => cudnnDirectionMode_t::CUDNN_BIDIRECTIONAL,
        }
    }

    #[cfg(feature = "cudnn")]
    pub(crate) fn from_cudnn(mode: cudnnDirectionMode_t) -> Self {
        match mode {
            cudnnDirectionMode_t::CUDNN_UNIDIRECTIONAL => RNNDirectionMode::Unidirectional,
            cudnnDirectionMode_t::CUDNN_BIDIRECTIONAL => RNNDirectionMode::Bidirectional,
            _ => RNNDirectionMode::Unidirectional, // default fallback
        }
    }
}

/// RNN mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RNNMode {
    /// ReLU RNN
    Relu,
    /// Tanh RNN
    Tanh,
    /// LSTM
    Lstm,
    /// GRU
    Gru,
}

impl RNNMode {
    #[cfg(feature = "cudnn")]
    pub(crate) fn to_cudnn(&self) -> cudnnRNNMode_t {
        match self {
            RNNMode::Relu => cudnnRNNMode_t::CUDNN_RNN_RELU,
            RNNMode::Tanh => cudnnRNNMode_t::CUDNN_RNN_TANH,
            RNNMode::Lstm => cudnnRNNMode_t::CUDNN_LSTM,
            RNNMode::Gru => cudnnRNNMode_t::CUDNN_GRU,
        }
    }

    #[cfg(feature = "cudnn")]
    pub(crate) fn from_cudnn(mode: cudnnRNNMode_t) -> Self {
        match mode {
            cudnnRNNMode_t::CUDNN_RNN_RELU => RNNMode::Relu,
            cudnnRNNMode_t::CUDNN_RNN_TANH => RNNMode::Tanh,
            cudnnRNNMode_t::CUDNN_LSTM => RNNMode::Lstm,
            cudnnRNNMode_t::CUDNN_GRU => RNNMode::Gru,
            _ => RNNMode::Tanh, // default fallback
        }
    }
}

/// RNN algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RNNAlgorithm {
    /// Standard algorithm
    Standard,
    /// Static persistent algorithm
    StaticPersistent,
    /// Dynamic persistent algorithm
    DynamicPersistent,
}

impl RNNAlgorithm {
    #[cfg(feature = "cudnn")]
    pub(crate) fn to_cudnn(&self) -> cudnnRNNAlgo_t {
        match self {
            RNNAlgorithm::Standard => cudnnRNNAlgo_t::CUDNN_STANDARD,
            RNNAlgorithm::StaticPersistent => cudnnRNNAlgo_t::CUDNN_STATIC_PERSISTENT,
            RNNAlgorithm::DynamicPersistent => cudnnRNNAlgo_t::CUDNN_DYNAMIC_PERSISTENT,
        }
    }

    #[cfg(feature = "cudnn")]
    pub(crate) fn from_cudnn(algo: cudnnRNNAlgo_t) -> Self {
        match algo {
            cudnnRNNAlgo_t::CUDNN_STANDARD => RNNAlgorithm::Standard,
            cudnnRNNAlgo_t::CUDNN_STATIC_PERSISTENT => RNNAlgorithm::StaticPersistent,
            cudnnRNNAlgo_t::CUDNN_DYNAMIC_PERSISTENT => RNNAlgorithm::DynamicPersistent,
            _ => RNNAlgorithm::Standard, // default fallback
        }
    }
}

/// Math type for precision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathType {
    /// Default math
    Default,
    /// Tensor operations math
    TensorOp,
    /// Tensor operations math with allow conversion
    TensorOpAllowConversion,
}

impl MathType {
    #[cfg(feature = "cudnn")]
    pub(crate) fn to_cudnn(&self) -> cudnnMathType_t {
        match self {
            MathType::Default => cudnnMathType_t::CUDNN_DEFAULT_MATH,
            MathType::TensorOp => cudnnMathType_t::CUDNN_TENSOR_OP_MATH,
            MathType::TensorOpAllowConversion => {
                cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION
            }
        }
    }

    #[cfg(feature = "cudnn")]
    pub(crate) fn from_cudnn(math_type: cudnnMathType_t) -> Self {
        match math_type {
            cudnnMathType_t::CUDNN_DEFAULT_MATH => MathType::Default,
            cudnnMathType_t::CUDNN_TENSOR_OP_MATH => MathType::TensorOp,
            cudnnMathType_t::CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION => {
                MathType::TensorOpAllowConversion
            }
            _ => MathType::Default, // default fallback
        }
    }
}

/// RNN forward mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RNNForwardMode {
    /// Training mode
    Training,
    /// Inference mode
    Inference,
}

impl RNNForwardMode {
    #[cfg(feature = "cudnn")]
    pub(crate) fn to_cudnn(&self) -> cudnnForwardMode_t {
        match self {
            RNNForwardMode::Training => cudnnForwardMode_t::CUDNN_FWD_MODE_TRAINING,
            RNNForwardMode::Inference => cudnnForwardMode_t::CUDNN_FWD_MODE_INFERENCE,
        }
    }

    #[cfg(feature = "cudnn")]
    pub(crate) fn from_cudnn(mode: cudnnForwardMode_t) -> Self {
        match mode {
            cudnnForwardMode_t::CUDNN_FWD_MODE_TRAINING => RNNForwardMode::Training,
            cudnnForwardMode_t::CUDNN_FWD_MODE_INFERENCE => RNNForwardMode::Inference,
            _ => RNNForwardMode::Inference, // default fallback
        }
    }
}

/// RNN data layout
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RNNDataLayout {
    /// Sequence major layout (seq_length, batch, vector_size)
    SeqMajorUnpacked,
    /// Batch major layout (batch, seq_length, vector_size)
    BatchMajorUnpacked,
    /// Sequence major packed layout
    SeqMajorPacked,
}

impl RNNDataLayout {
    #[cfg(feature = "cudnn")]
    pub(crate) fn to_cudnn(&self) -> cudnnRNNDataLayout_t {
        match self {
            RNNDataLayout::SeqMajorUnpacked => {
                cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED
            }
            RNNDataLayout::BatchMajorUnpacked => {
                cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED
            }
            RNNDataLayout::SeqMajorPacked => {
                cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED
            }
        }
    }

    #[cfg(feature = "cudnn")]
    pub(crate) fn from_cudnn(layout: cudnnRNNDataLayout_t) -> Self {
        match layout {
            cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED => {
                RNNDataLayout::SeqMajorUnpacked
            }
            cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED => {
                RNNDataLayout::BatchMajorUnpacked
            }
            cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED => {
                RNNDataLayout::SeqMajorPacked
            }
            _ => RNNDataLayout::SeqMajorUnpacked, // default fallback
        }
    }
}

/// Convolution forward algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvolutionForwardAlgorithm {
    /// Implicit GEMM algorithm
    #[default]
    ImplicitGemm,
    /// Implicit precomputed GEMM algorithm
    ImplicitPrecompGemm,
    /// GEMM algorithm
    Gemm,
    /// Direct algorithm
    Direct,
    /// FFT algorithm
    Fft,
    /// FFT tiling algorithm
    FftTiling,
    /// Winograd algorithm
    Winograd,
    /// Winograd non-fused algorithm
    WinogradNonfused,
    /// Count (used for iteration)
    Count,
}

impl ConvolutionForwardAlgorithm {
    #[cfg(feature = "cudnn")]
    pub(crate) fn to_cudnn(&self) -> cudnnConvolutionFwdAlgo_t {
        match self {
            ConvolutionForwardAlgorithm::ImplicitGemm => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
            }
            ConvolutionForwardAlgorithm::ImplicitPrecompGemm => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
            }
            ConvolutionForwardAlgorithm::Gemm => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM
            }
            ConvolutionForwardAlgorithm::Direct => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
            }
            ConvolutionForwardAlgorithm::Fft => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT
            }
            ConvolutionForwardAlgorithm::FftTiling => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
            }
            ConvolutionForwardAlgorithm::Winograd => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
            }
            ConvolutionForwardAlgorithm::WinogradNonfused => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
            }
            ConvolutionForwardAlgorithm::Count => {
                cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_COUNT
            }
        }
    }

    #[cfg(feature = "cudnn")]
    pub(crate) fn from_cudnn(algo: cudnnConvolutionFwdAlgo_t) -> Self {
        match algo {
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM => {
                ConvolutionForwardAlgorithm::ImplicitGemm
            }
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM => {
                ConvolutionForwardAlgorithm::ImplicitPrecompGemm
            }
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM => {
                ConvolutionForwardAlgorithm::Gemm
            }
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT => {
                ConvolutionForwardAlgorithm::Direct
            }
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT => {
                ConvolutionForwardAlgorithm::Fft
            }
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING => {
                ConvolutionForwardAlgorithm::FftTiling
            }
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD => {
                ConvolutionForwardAlgorithm::Winograd
            }
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED => {
                ConvolutionForwardAlgorithm::WinogradNonfused
            }
            cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_COUNT => {
                ConvolutionForwardAlgorithm::Count
            }
            _ => ConvolutionForwardAlgorithm::ImplicitGemm, // default fallback
        }
    }
}

/// Convolution forward algorithm performance information
#[derive(Debug, Clone, Default)]
pub struct ConvolutionForwardAlgorithmPerformance {
    /// Algorithm type
    pub algorithm: ConvolutionForwardAlgorithm,
    /// Status of the algorithm
    pub status: ConvolutionStatus,
    /// Execution time in milliseconds
    pub time: f32,
    /// Memory usage in bytes
    pub memory: usize,
    /// Determinism flag
    pub determinism: ConvolutionDeterminism,
    /// Math type used
    pub math_type: ConvolutionMathType,
}

/// Convolution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvolutionStatus {
    /// Success
    #[default]
    Success,
    /// Not supported
    NotSupported,
    /// Bad parameter
    BadParam,
    /// Mapping error
    MappingError,
    /// Execution failed
    ExecutionFailed,
    /// Internal error
    InternalError,
    /// Not initialized
    NotInitialized,
    /// Architecture mismatch
    ArchMismatch,
    /// Runtime prerequisite missing
    RuntimePrerequisiteMissing,
}

/// Convolution determinism
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvolutionDeterminism {
    /// Non-deterministic
    #[default]
    NonDeterministic,
    /// Deterministic
    Deterministic,
}

/// Convolution math type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvolutionMathType {
    /// Default math
    #[default]
    Default,
    /// Tensor operations
    TensorOp,
}

/// Tensor format for cuDNN operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorFormat {
    /// NCHW format (batch, channels, height, width)
    NCHW,
    /// NHWC format (batch, height, width, channels)
    NHWC,
    /// NCHWVectC format (vectorized channels)
    NCHWVectC,
}

impl Default for TensorFormat {
    fn default() -> Self {
        Self::NCHW
    }
}

/// Check if cuDNN is available
pub fn is_available() -> bool {
    #[cfg(feature = "cudnn")]
    {
        true
    }
    #[cfg(not(feature = "cudnn"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enum_conversions() {
        // Test that our enum conversions work properly
        let conv_mode = ConvolutionMode::CrossCorrelation;
        let activ_mode = ActivationMode::Relu;
        let pool_mode = PoolingMode::Max;
        let rnn_mode = RNNMode::Lstm;

        // Basic creation tests
        assert_eq!(conv_mode, ConvolutionMode::CrossCorrelation);
        assert_eq!(activ_mode, ActivationMode::Relu);
        assert_eq!(pool_mode, PoolingMode::Max);
        assert_eq!(rnn_mode, RNNMode::Lstm);
    }

    #[test]
    fn test_availability() {
        // Test availability function
        let available = is_available();

        #[cfg(feature = "cudnn")]
        assert!(available);

        #[cfg(not(feature = "cudnn"))]
        assert!(!available);
    }

    #[test]
    fn test_enum_debug() {
        // Test that enums can be debugged
        let conv_mode = ConvolutionMode::Convolution;
        let debug_str = format!("{:?}", conv_mode);
        assert!(debug_str.contains("Convolution"));
    }

    #[test]
    fn test_enum_equality() {
        // Test enum equality
        assert_eq!(
            ConvolutionMode::CrossCorrelation,
            ConvolutionMode::CrossCorrelation
        );
        assert_ne!(
            ConvolutionMode::CrossCorrelation,
            ConvolutionMode::Convolution
        );

        assert_eq!(ActivationMode::Relu, ActivationMode::Relu);
        assert_ne!(ActivationMode::Relu, ActivationMode::Sigmoid);
    }

    #[test]
    fn test_performance_struct() {
        // Test that we can create performance struct
        let perf = ConvolutionForwardAlgorithmPerformance {
            algorithm: ConvolutionForwardAlgorithm::ImplicitGemm,
            status: ConvolutionStatus::Success,
            time: 1.5,
            memory: 1024,
            determinism: ConvolutionDeterminism::Deterministic,
            math_type: ConvolutionMathType::Default,
        };

        assert_eq!(perf.algorithm, ConvolutionForwardAlgorithm::ImplicitGemm);
        assert_eq!(perf.status, ConvolutionStatus::Success);
        assert_eq!(perf.time, 1.5);
        assert_eq!(perf.memory, 1024);
    }
}
