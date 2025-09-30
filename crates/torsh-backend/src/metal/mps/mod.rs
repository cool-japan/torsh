//! Metal Performance Shaders (MPS) integration for optimized operations

use metal::CommandBuffer;
use metal::NSUInteger;
use objc2::runtime::Object;
use objc2::{class, msg_send, sel, ClassType};

use crate::metal::error::Result;

mod activation;
mod conv;
mod matmul;
mod mixed_precision;
mod networks;
mod neural_ops;
mod normalization;
mod pooling;

pub use activation::{ActivationType, MPSActivation};
pub use conv::{Conv2dParams, MPSConv2d};
pub use matmul::MPSMatMul;
pub use mixed_precision::{
    AMPConfig, MPSAutocast, MPSGradScaler, MPSMixedPrecision, MixedPrecisionStats, OptLevel,
};
pub use networks::{
    MPSConvBlock, MPSConvBlockBuilder, MPSFeedForward, MPSLayerNorm, MPSOptimizations,
    MPSResidualBlock, MPSTransformerEncoderLayer, MemoryLayout,
};
pub use neural_ops::{
    Conv2dParams as OptimizedConv2dParams, ConvolutionAlgorithm, MPSBatchNormalization,
    MPSFusedOps, MPSLinear, MPSMultiHeadAttention, MPSOptimizedConv2d,
};
pub use normalization::MPSBatchNorm;
pub use pooling::{MPSAvgPool2d, MPSMaxPool2d};

/// Base trait for MPS operations
pub trait MPSOperation {
    /// Encode the operation into a command buffer
    fn encode(&self, command_buffer: &CommandBuffer) -> Result<()>;
}

/// MPS data type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MPSDataType {
    Float16,
    Float32,
    Int8,
    UInt8,
}

impl MPSDataType {
    /// Convert to Metal Performance Shaders data type constant
    pub fn to_mps_constant(&self) -> u32 {
        match self {
            MPSDataType::Float16 => 0x10DE, // MPSDataTypeFloat16
            MPSDataType::Float32 => 0x10E0, // MPSDataTypeFloat32
            MPSDataType::Int8 => 0x1020,    // MPSDataTypeInt8
            MPSDataType::UInt8 => 0x1008,   // MPSDataTypeUInt8
        }
    }
}

/// MPS tensor descriptor
#[allow(dead_code)]
pub struct MPSTensorDescriptor {
    shape: Vec<usize>,
    dtype: MPSDataType,
    layout: MPSLayout,
}

/// MPS tensor layout
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MPSLayout {
    NCHW,
    NHWC,
}

/// Helper to create MPS matrix descriptor
pub(crate) unsafe fn create_matrix_descriptor(
    rows: usize,
    columns: usize,
    dtype: MPSDataType,
) -> *mut Object {
    let class = objc2::class!(MPSMatrixDescriptor);
    let descriptor: *mut Object = msg_send![class, alloc];
    let descriptor: *mut Object = msg_send![descriptor, init];

    // Set dimensions
    let _: () = msg_send![descriptor, setRows: rows as NSUInteger];
    let _: () = msg_send![descriptor, setColumns: columns as NSUInteger];

    // Set data type
    let _: () = msg_send![descriptor, setDataType: dtype.to_mps_constant()];

    // Set row bytes (assuming contiguous)
    let element_size = match dtype {
        MPSDataType::Float16 => 2,
        MPSDataType::Float32 => 4,
        MPSDataType::Int8 | MPSDataType::UInt8 => 1,
    };
    let row_bytes = columns * element_size;
    let _: () = msg_send![descriptor, setRowBytes: row_bytes as NSUInteger];

    descriptor
}

/// Helper to create MPS image descriptor
pub(crate) unsafe fn create_image_descriptor(
    width: usize,
    height: usize,
    channels: usize,
    _dtype: MPSDataType,
) -> *mut Object {
    let class = objc2::class!(MPSImageDescriptor);
    let descriptor: *mut Object = msg_send![class, alloc];

    // Use appropriate initializer based on channels
    let descriptor: *mut Object = if channels == 1 {
        msg_send![descriptor,
            initWithChannelFormat: 0x10DE // MTLPixelFormatR16Float
            width: width as NSUInteger
            height: height as NSUInteger
            featureChannels: channels as NSUInteger
        ]
    } else {
        msg_send![descriptor,
            initWithChannelFormat: 0x7310 // MTLPixelFormatRGBA16Float
            width: width as NSUInteger
            height: height as NSUInteger
            featureChannels: channels as NSUInteger
        ]
    };

    descriptor
}

/// Helper to create MPS CNN convolution descriptor
pub(crate) unsafe fn create_conv_descriptor(
    kernel_height: usize,
    kernel_width: usize,
    input_channels: usize,
    output_channels: usize,
) -> *mut Object {
    let class = objc2::class!(MPSCNNConvolutionDescriptor);
    let descriptor: *mut Object = msg_send![class, alloc];
    let descriptor: *mut Object = msg_send![descriptor, init];

    let _: () = msg_send![descriptor, setKernelHeight: kernel_height as NSUInteger];
    let _: () = msg_send![descriptor, setKernelWidth: kernel_width as NSUInteger];
    let _: () = msg_send![descriptor, setInputFeatureChannels: input_channels as NSUInteger];
    let _: () = msg_send![descriptor, setOutputFeatureChannels: output_channels as NSUInteger];

    descriptor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mps_data_type() {
        assert_eq!(MPSDataType::Float32.to_mps_constant(), 0x10E0);
        assert_eq!(MPSDataType::Float16.to_mps_constant(), 0x10DE);
    }
}
