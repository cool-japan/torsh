//! ToRSh Package - Model packaging and distribution
//!
//! This module provides functionality similar to torch.package for creating
//! self-contained model packages that include code, weights, and dependencies.
//!
//! # Features
//!
//! - **Package Creation**: Create self-contained model packages with the [`Package`] struct
//! - **Builder Pattern**: Use [`PackageBuilder`] for convenient package creation
//! - **Import/Export**: Load and save packages with [`PackageImporter`] and [`PackageExporter`]
//! - **Resource Management**: Manage different types of resources with [`Resource`]
//! - **Version Management**: Handle semantic versioning with [`PackageVersion`]
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use torsh_package::{Package, PackageBuilder};
//!
//! // Create a package using the builder pattern
//! let package = PackageBuilder::new("my-model".to_string(), "1.0.0".to_string())
//!     .author("Your Name".to_string())
//!     .description("My awesome model".to_string())
//!     .add_dependency("torch", "2.0")
//!     .package();
//!
//! // Or create directly
//! let mut package = Package::new("my-model".to_string(), "1.0.0".to_string());
//! package.add_source_file("main", "fn main() { println!(\"Hello!\"); }").unwrap();
//!
//! // Save the package
//! package.save("my-model.torshpkg").unwrap();
//!
//! // Load a package
//! let loaded = Package::load("my-model.torshpkg").unwrap();
//! ```

#![allow(clippy::result_large_err)]
#![deny(missing_docs)]

// Core modules
pub mod builder;
pub mod compression;
pub mod delta;
pub mod dependency;
pub mod exporter;
pub mod format_compat;
pub mod importer;
pub mod lazy_resources;
pub mod manifest;
pub mod package;
pub mod resources;
pub mod utils;
pub mod version;

// Re-exports for convenience
pub use builder::{BuilderConfig, PackageBuilder};
pub use compression::{
    AdvancedCompressor, CompressionAlgorithm, CompressionConfig, CompressionLevel,
    CompressionResult, CompressionStats, CompressionStrategy, DecompressionResult,
    ParallelCompressor,
};
pub use delta::{
    DeltaOperation, DeltaPackageExt, DeltaPatch, DeltaPatchApplier, DeltaPatchBuilder,
};
pub use dependency::{
    DependencyConflict, DependencyGraph, DependencyResolver, DependencySpec, LocalPackageRegistry,
    PackageInfo, PackageRegistry, ResolutionStrategy, ResolvedDependency,
};
pub use exporter::{ExportConfig, PackageExporter};
pub use format_compat::{
    FormatCompatibilityManager, FormatConverter, HuggingFaceConverter, PackageFormat,
    PyTorchConverter,
};
pub use importer::{ImportConfig, PackageImporter};
pub use lazy_resources::{EvictionStrategy, LazyResource, LazyResourceManager};
pub use manifest::{ModuleInfo, PackageManifest, ResourceInfo};
pub use package::Package;
pub use resources::{Resource, ResourceType};
#[cfg(feature = "with-nn")]
pub use utils::export_module;
pub use utils::import_module;
pub use version::{CompatibilityChecker, PackageVersion, VersionComparator, VersionRequirement};

/// Package format version
///
/// This follows semantic versioning where:
/// - Major version changes indicate breaking format changes
/// - Minor version changes indicate backward-compatible additions
/// - Patch version changes indicate bug fixes
pub const PACKAGE_FORMAT_VERSION: &str = "1.0.0";

/// Default file extension for ToRSh packages
pub const PACKAGE_EXTENSION: &str = "torshpkg";

/// Maximum package size in bytes (1GB by default)
pub const DEFAULT_MAX_PACKAGE_SIZE: usize = 1024 * 1024 * 1024;

/// Maximum number of resources per package
pub const DEFAULT_MAX_RESOURCES: usize = 10000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(PACKAGE_FORMAT_VERSION, "1.0.0");
        assert_eq!(PACKAGE_EXTENSION, "torshpkg");
        assert!(DEFAULT_MAX_PACKAGE_SIZE > 0);
        assert!(DEFAULT_MAX_RESOURCES > 0);
    }

    #[test]
    fn test_format_version_parsing() {
        let version = semver::Version::parse(PACKAGE_FORMAT_VERSION).unwrap();
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 0);
        assert_eq!(version.patch, 0);
    }
}
