//! Convolutional layers
//!
//! Organized by layer type:
//! - `types`: Struct definitions for all convolutional layer types
//! - `conv1d_traits`: `Module` + `Debug` impls for `Conv1d`
//! - `conv2d_traits`: `Module` + `Debug` impls for `Conv2d`
//! - `conv3d_traits`: `Module` + `Debug` impls for `Conv3d`
//! - `convtranspose1d_traits`: `Module` + `Debug` impls for `ConvTranspose1d`
//! - `convtranspose2d_traits`: `Module` + `Debug` impls for `ConvTranspose2d`
//! - `convtranspose3d_traits`: `Module` + `Debug` impls for `ConvTranspose3d`
//! - `depthwiseseparableconv_traits`: `Module` + `Debug` impls for `DepthwiseSeparableConv`
//! - `functions`: Tests

mod conv1d_traits;
mod conv2d_traits;
mod conv3d_traits;
mod convtranspose1d_traits;
mod convtranspose2d_traits;
mod convtranspose3d_traits;
mod depthwiseseparableconv_traits;
mod functions;
pub(crate) mod types;

pub use types::{
    Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d,
    DepthwiseSeparableConv,
};
