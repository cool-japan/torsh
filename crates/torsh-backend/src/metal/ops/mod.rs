//! Tensor operations implementation for Metal backend

mod binary;
mod conv;
mod matmul;
mod pooling;
mod reduction;
mod unary;

pub use binary::*;
pub use conv::*;
pub use matmul::*;
pub use pooling::*;
pub use reduction::*;
pub use unary::*;

use crate::metal::{device::MetalDevice, error::Result};

/// Execute a Metal operation and wait for completion
pub fn execute_and_wait<F>(device: &MetalDevice, op: F) -> Result<()>
where
    F: FnOnce(&metal::ComputeCommandEncoderRef) -> Result<()>,
{
    let command_buffer = device.new_command_buffer()?;
    let encoder = command_buffer.new_compute_command_encoder();

    op(encoder)?;

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Ok(())
}
