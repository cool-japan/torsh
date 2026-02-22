//! Command implementations for ToRSh CLI

#![allow(ambiguous_glob_reexports)]

pub mod benchmark;
pub mod dataset;
pub mod dev;
pub mod hub;
pub mod info;
pub mod init;
pub mod model;
pub mod train;
pub mod update;

// ✅ Beta.1 Enhanced Real Implementations
pub mod benchmark_real;
pub mod dataset_real;
pub mod quantize_real;
pub mod train_real;

// Re-export command structures
pub use benchmark::*;
pub use dataset::*;
pub use dev::*;
pub use hub::*;
pub use info::*;
pub use init::*;
pub use model::*;
pub use train::*;
pub use update::*;
