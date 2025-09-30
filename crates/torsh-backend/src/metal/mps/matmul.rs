//! Matrix multiplication using Metal Performance Shaders

use metal::foreign_types::ForeignType;
use metal::{CommandBuffer, Device, NSUInteger};
use objc2::runtime::{Bool, Object};
use objc2::{msg_send, sel, ClassType};

use crate::metal::{
    buffer::MetalBuffer,
    error::{metal_errors, MetalError, Result},
    mps::{create_matrix_descriptor, MPSDataType, MPSOperation},
};

/// Matrix multiplication using MPS
#[allow(dead_code)]
pub struct MPSMatMul {
    /// MPS matrix multiplication object
    matmul: *mut Object,
    /// Output buffer
    output: MetalBuffer,
    /// Dimensions
    m: usize,
    n: usize,
    k: usize,
}

impl MPSMatMul {
    /// Create a new MPS matrix multiplication operation
    ///
    /// Computes: C = alpha * A @ B + beta * C
    /// Where A is [M x K], B is [K x N], C is [M x N]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        a: &MetalBuffer,
        b: &MetalBuffer,
        c: Option<&MetalBuffer>,
        alpha: f32,
        beta: f32,
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<Self> {
        unsafe {
            // Get dimensions
            let a_shape = a.shape().dims();
            let b_shape = b.shape().dims();

            if a_shape.len() != 2 || b_shape.len() != 2 {
                return Err(MetalError::ShapeMismatch {
                    expected: vec![2],
                    got: vec![a_shape.len(), b_shape.len()],
                });
            }

            let (m, k_a) = if transpose_a {
                (a_shape[1], a_shape[0])
            } else {
                (a_shape[0], a_shape[1])
            };

            let (k_b, n) = if transpose_b {
                (b_shape[1], b_shape[0])
            } else {
                (b_shape[0], b_shape[1])
            };

            if k_a != k_b {
                return Err(MetalError::ShapeMismatch {
                    expected: vec![k_a],
                    got: vec![k_b],
                });
            }

            let k = k_a;

            // Create output buffer if not provided
            let output = if let Some(c_buffer) = c {
                c_buffer.clone()
            } else {
                MetalBuffer::zeros(
                    &torsh_core::Shape::from(vec![m, n]),
                    &torsh_core::DType::F32,
                    &crate::metal::device::MetalDevice::new()?,
                )?
            };

            // Create matrix descriptors
            let _a_desc = create_matrix_descriptor(
                if transpose_a { k } else { m },
                if transpose_a { m } else { k },
                MPSDataType::Float32,
            );

            let _b_desc = create_matrix_descriptor(
                if transpose_b { n } else { k },
                if transpose_b { k } else { n },
                MPSDataType::Float32,
            );

            let _c_desc = create_matrix_descriptor(m, n, MPSDataType::Float32);

            // Create MPS matrix multiplication
            let class = objc2::class!(MPSMatrixMultiplication);
            let matmul: *mut Object = msg_send![class, alloc];
            let matmul: *mut Object = msg_send![matmul,
                initWithDevice: device.as_ptr() as *mut Object
                transposeLeft: objc2::runtime::Bool::from(transpose_a)
                transposeRight: objc2::runtime::Bool::from(transpose_b)
                resultRows: m as NSUInteger
                resultColumns: n as NSUInteger
                interiorColumns: k as NSUInteger
                alpha: alpha as f64
                beta: beta as f64
            ];

            Ok(Self {
                matmul,
                output,
                m,
                n,
                k,
            })
        }
    }

    /// Encode the matrix multiplication
    pub fn encode_matmul(
        &self,
        command_buffer: &CommandBuffer,
        a: &MetalBuffer,
        b: &MetalBuffer,
    ) -> Result<()> {
        unsafe {
            // Create matrix objects
            let class = objc2::class!(MPSMatrix);

            let a_matrix: *mut Object = msg_send![class, alloc];
            let a_matrix: *mut Object = msg_send![a_matrix,
                initWithBuffer: a.buffer().as_ptr() as *mut Object
                descriptor: create_matrix_descriptor(self.m, self.k, MPSDataType::Float32)
            ];

            let b_matrix: *mut Object = msg_send![class, alloc];
            let b_matrix: *mut Object = msg_send![b_matrix,
                initWithBuffer: b.buffer().as_ptr() as *mut Object
                descriptor: create_matrix_descriptor(self.k, self.n, MPSDataType::Float32)
            ];

            let c_matrix: *mut Object = msg_send![class, alloc];
            let c_matrix: *mut Object = msg_send![c_matrix,
                initWithBuffer: self.output.buffer().as_ptr() as *mut Object
                descriptor: create_matrix_descriptor(self.m, self.n, MPSDataType::Float32)
            ];

            // Encode the operation
            let _: () = msg_send![self.matmul,
                encodeToCommandBuffer: command_buffer.as_ptr() as *mut Object
                leftMatrix: a_matrix
                rightMatrix: b_matrix
                resultMatrix: c_matrix
            ];

            Ok(())
        }
    }

    /// Get the output buffer
    pub fn output(&self) -> &MetalBuffer {
        &self.output
    }
}

impl MPSOperation for MPSMatMul {
    fn encode(&self, _command_buffer: &CommandBuffer) -> Result<()> {
        // This would need the input buffers passed in
        // For now, this is a placeholder
        Err(metal_errors::kernel_execution_error(
            "MPSMatMul::encode requires input buffers".to_string(), None
        ))
    }
}

impl Drop for MPSMatMul {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.matmul, release];
        }
    }
}
