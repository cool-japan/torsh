//! Pooling layer modules
//!
//! This module contains pooling layer types and their trait implementations,
//! split from the original monolithic pooling.rs for maintainability.

// Type definitions for all pooling layers
pub mod types;

// Standalone functions
pub mod functions;

// Trait implementations (Module, Debug) for each pooling layer type
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
mod lppool1d_traits;
mod lppool2d_traits;
mod maxpool1d_traits;
mod maxpool2d_traits;
mod maxpool3d_traits;

// Re-export all public types and functions
pub use types::*;
