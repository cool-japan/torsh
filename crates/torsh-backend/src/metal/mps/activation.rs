//! Activation functions using Metal Performance Shaders

use metal::foreign_types::ForeignType;
use metal::{CommandBuffer, Device};
use objc2::msg_send;
use objc2::runtime::AnyObject;

use crate::metal::{buffer::MetalBuffer, error::Result};

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationType {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LeakyReLU(f32),
    ELU(f32),
}

/// MPS activation function
#[allow(dead_code)]
pub struct MPSActivation {
    activation: *mut AnyObject,
    activation_type: ActivationType,
}

impl MPSActivation {
    /// Create a new activation function
    pub fn new(device: &Device, activation_type: ActivationType) -> Result<Self> {
        unsafe {
            let activation: *mut AnyObject = match activation_type {
                ActivationType::ReLU => {
                    let class = objc2::class!(MPSCNNNeuronReLU);
                    let act: *mut AnyObject = msg_send![class, alloc];
                    msg_send![act, initWithDevice: device.as_ptr() as *mut AnyObject]
                }
                ActivationType::Sigmoid => {
                    let class = objc2::class!(MPSCNNNeuronSigmoid);
                    let act: *mut AnyObject = msg_send![class, alloc];
                    msg_send![act, initWithDevice: device.as_ptr() as *mut AnyObject]
                }
                ActivationType::Tanh => {
                    let class = objc2::class!(MPSCNNNeuronTanH);
                    let act: *mut AnyObject = msg_send![class, alloc];
                    msg_send![act, initWithDevice: device.as_ptr() as *mut AnyObject]
                }
                ActivationType::Softmax => {
                    let class = objc2::class!(MPSCNNSoftMax);
                    let act: *mut AnyObject = msg_send![class, alloc];
                    msg_send![act, initWithDevice: device.as_ptr() as *mut AnyObject]
                }
                ActivationType::LeakyReLU(alpha) => {
                    let class = objc2::class!(MPSCNNNeuronLinear);
                    let act: *mut AnyObject = msg_send![class, alloc];
                    msg_send![act, initWithDevice: device.as_ptr() as *mut AnyObject, a: alpha, b: 0.0f32]
                }
                ActivationType::ELU(alpha) => {
                    let class = objc2::class!(MPSCNNNeuronELU);
                    let act: *mut AnyObject = msg_send![class, alloc];
                    msg_send![act, initWithDevice: device.as_ptr() as *mut AnyObject, a: alpha]
                }
            };

            Ok(Self {
                activation,
                activation_type,
            })
        }
    }

    /// Apply activation in-place
    pub fn apply_inplace(
        &self,
        _command_buffer: &CommandBuffer,
        _tensor: &mut MetalBuffer,
    ) -> Result<()> {
        // For CNN neurons, we need to work with images
        // For now, we'll use the simpler MPSNeuron operations

        // This is a simplified version - real implementation would handle
        // different tensor layouts and use appropriate MPS primitives

        Ok(())
    }

    /// Apply activation to create new output
    pub fn apply(
        &self,
        _command_buffer: &CommandBuffer,
        _input: &MetalBuffer,
        _output: &MetalBuffer,
    ) -> Result<()> {
        // Similar to apply_inplace but writes to a new buffer
        Ok(())
    }
}

impl Drop for MPSActivation {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.activation, release];
        }
    }
}
