//! Pooling layers
//!
//! Organized by layer type:
//! - `types`: Struct definitions for all pooling layer types
//! - `maxpool2d_traits`: `Module` + `Debug` impls for `MaxPool2d`
//! - `maxpool1d_traits`: `Module` + `Debug` impls for `MaxPool1d`
//! - `maxpool3d_traits`: `Module` + `Debug` impls for `MaxPool3d`
//! - `avgpool2d_traits`: `Module` + `Debug` impls for `AvgPool2d`
//! - `adaptiveavgpool1d_traits`: `Module` + `Debug` impls for `AdaptiveAvgPool1d`
//! - `adaptiveavgpool2d_traits`: `Module` + `Debug` impls for `AdaptiveAvgPool2d`
//! - `adaptiveavgpool3d_traits`: `Module` + `Debug` impls for `AdaptiveAvgPool3d`
//! - `adaptivemaxpool1d_traits`: `Module` + `Debug` impls for `AdaptiveMaxPool1d`
//! - `adaptivemaxpool2d_traits`: `Module` + `Debug` impls for `AdaptiveMaxPool2d`
//! - `adaptivemaxpool3d_traits`: `Module` + `Debug` impls for `AdaptiveMaxPool3d`
//! - `fractionalmaxpool1d_traits`: `Module` + `Debug` impls for `FractionalMaxPool1d`
//! - `fractionalmaxpool2d_traits`: `Module` + `Debug` impls for `FractionalMaxPool2d`
//! - `fractionalmaxpool3d_traits`: `Module` + `Debug` impls for `FractionalMaxPool3d`
//! - `lppool1d_traits`: `Module` + `Debug` impls for `LPPool1d`
//! - `lppool2d_traits`: `Module` + `Debug` impls for `LPPool2d`
//! - `functions`: Tests

mod adaptiveavgpool1d_traits;
mod adaptiveavgpool2d_traits;
mod adaptiveavgpool3d_traits;
mod adaptivemaxpool1d_traits;
mod adaptivemaxpool2d_traits;
mod adaptivemaxpool3d_traits;
mod avgpool2d_traits;
mod fractionalmaxpool1d_traits;
mod fractionalmaxpool2d_traits;
mod fractionalmaxpool3d_traits;
mod functions;
mod lppool1d_traits;
mod lppool2d_traits;
mod maxpool1d_traits;
mod maxpool2d_traits;
mod maxpool3d_traits;
pub(crate) mod types;

pub use types::{
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d, AdaptiveMaxPool1d, AdaptiveMaxPool2d,
    AdaptiveMaxPool3d, AvgPool2d, FractionalMaxPool1d, FractionalMaxPool2d, FractionalMaxPool3d,
    GlobalPool, LPPool1d, LPPool2d, MaxPool1d, MaxPool2d, MaxPool3d,
};
