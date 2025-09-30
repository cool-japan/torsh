//! Pooling operations using Metal Performance Shaders

use metal::foreign_types::ForeignType;
use metal::Device;
use metal::NSUInteger;
use objc2::runtime::Object;
use objc2::{msg_send, sel, ClassType};

use crate::metal::{
    buffer::MetalBuffer,
    error::{MetalError, Result},
};

/// Max pooling 2D using MPS
#[allow(dead_code)]
pub struct MPSMaxPool2d {
    pool: *mut Object,
    output: MetalBuffer,
}

impl MPSMaxPool2d {
    /// Create a new max pooling operation
    pub fn new(
        device: &Device,
        input_shape: &[usize], // [N, C, H, W]
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self> {
        unsafe {
            if input_shape.len() != 4 {
                return Err(MetalError::ShapeMismatch {
                    expected: vec![4],
                    got: vec![input_shape.len()],
                });
            }

            let (n, c, h, w) = (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            );
            let (kh, kw) = kernel_size;
            let (sh, sw) = stride;
            let (ph, pw) = padding;

            // Calculate output dimensions
            let out_h = (h + 2 * ph - kh) / sh + 1;
            let out_w = (w + 2 * pw - kw) / sw + 1;

            let output = MetalBuffer::zeros(
                &torsh_core::Shape::from(vec![n, c, out_h, out_w]),
                &torsh_core::DType::F32,
                &crate::metal::device::MetalDevice::new()?,
            )?;

            // Create MPS max pooling
            let class = objc2::class!(MPSCNNPoolingMax);
            let pool: *mut Object = msg_send![class, alloc];
            let pool: *mut Object = msg_send![pool,
                initWithDevice: device.as_ptr() as *mut Object
                kernelWidth: kw as NSUInteger
                kernelHeight: kh as NSUInteger
                strideInPixelsX: sw as NSUInteger
                strideInPixelsY: sh as NSUInteger
            ];

            // Set padding
            let _: () = msg_send![pool, setPaddingLeft: pw as NSUInteger];
            let _: () = msg_send![pool, setPaddingRight: pw as NSUInteger];
            let _: () = msg_send![pool, setPaddingTop: ph as NSUInteger];
            let _: () = msg_send![pool, setPaddingBottom: ph as NSUInteger];

            // Set edge mode
            let _: () = msg_send![pool, setEdgeMode: 0]; // MPSImageEdgeModeZero

            Ok(Self { pool, output })
        }
    }

    /// Get output buffer
    pub fn output(&self) -> &MetalBuffer {
        &self.output
    }
}

impl Drop for MPSMaxPool2d {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.pool, release];
        }
    }
}

/// Average pooling 2D using MPS
#[allow(dead_code)]
pub struct MPSAvgPool2d {
    pool: *mut Object,
    output: MetalBuffer,
}

impl MPSAvgPool2d {
    /// Create a new average pooling operation
    pub fn new(
        device: &Device,
        input_shape: &[usize], // [N, C, H, W]
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        count_include_pad: bool,
    ) -> Result<Self> {
        unsafe {
            if input_shape.len() != 4 {
                return Err(MetalError::ShapeMismatch {
                    expected: vec![4],
                    got: vec![input_shape.len()],
                });
            }

            let (n, c, h, w) = (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            );
            let (kh, kw) = kernel_size;
            let (sh, sw) = stride;
            let (ph, pw) = padding;

            // Calculate output dimensions
            let out_h = (h + 2 * ph - kh) / sh + 1;
            let out_w = (w + 2 * pw - kw) / sw + 1;

            let output = MetalBuffer::zeros(
                &torsh_core::Shape::from(vec![n, c, out_h, out_w]),
                &torsh_core::DType::F32,
                &crate::metal::device::MetalDevice::new()?,
            )?;

            // Create MPS average pooling
            let class = objc2::class!(MPSCNNPoolingAverage);
            let pool: *mut Object = msg_send![class, alloc];
            let pool: *mut Object = msg_send![pool,
                initWithDevice: device.as_ptr() as *mut Object
                kernelWidth: kw as NSUInteger
                kernelHeight: kh as NSUInteger
                strideInPixelsX: sw as NSUInteger
                strideInPixelsY: sh as NSUInteger
            ];

            // Set padding
            let _: () = msg_send![pool, setPaddingLeft: pw as NSUInteger];
            let _: () = msg_send![pool, setPaddingRight: pw as NSUInteger];
            let _: () = msg_send![pool, setPaddingTop: ph as NSUInteger];
            let _: () = msg_send![pool, setPaddingBottom: ph as NSUInteger];

            // Set edge mode based on count_include_pad
            let edge_mode = if count_include_pad { 0 } else { 1 }; // Zero vs Clamp
            let _: () = msg_send![pool, setEdgeMode: edge_mode];

            Ok(Self { pool, output })
        }
    }

    /// Get output buffer
    pub fn output(&self) -> &MetalBuffer {
        &self.output
    }
}

impl Drop for MPSAvgPool2d {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.pool, release];
        }
    }
}
