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
pub mod access_control;
pub mod audit;
pub mod audit_storage;
pub mod backup;
pub mod builder;
pub mod cdn;
pub mod cloud_storage;
pub mod compression;
pub mod delta;
pub mod dependency;
pub mod dependency_installer;
pub mod dependency_lockfile;
pub mod dependency_solver;
pub mod diagnostics;
pub mod exporter;
pub mod format_compat;
pub mod governance;
pub mod importer;
pub mod lazy_resources;
pub mod manifest;
pub mod mirror;
pub mod monitoring;
pub mod optimization;
pub mod package;
pub mod profiling;
pub mod replication;
pub mod resources;
pub mod sandbox;
pub mod security;
pub mod storage;
pub mod syslog_integration;
pub mod utils;
pub mod version;
pub mod vulnerability;

// Async and registry modules
#[cfg(feature = "async")]
pub mod async_ops;
pub mod registry;

// Re-exports for convenience
pub use access_control::{
    AccessCheckResult, AccessControlManager, AccessLevel, AclEntry, Organization,
    OrganizationMembership, PackageOwnership, Permission, Role, User,
};
pub use audit::{
    ActionResult, AuditEvent, AuditEventType, AuditListener, AuditLogConfig, AuditLogFormat,
    AuditLogger, AuditQuery, AuditSeverity, AuditStatistics,
};
pub use audit_storage::{
    AuditStorage, DatabaseStatistics, InMemoryStorage, PoolStatistics, PostgresStorage,
    PostgresStorageConfig, SqliteStorage, SqliteStorageConfig, SslMode, StorageStatistics,
};
pub use backup::{
    BackupConfig, BackupDestination, BackupManager, BackupMetadata, BackupStatistics,
    BackupStrategy, RecoveryPoint, RetentionPolicy, VerificationResult,
};
pub use builder::{BuilderConfig, PackageBuilder};
pub use cdn::{
    CacheControl, CdnConfig, CdnManager, CdnProvider, CdnRegion, CdnStatistics, EdgeNode,
    EdgeNodeStatus,
};
pub use cloud_storage::{
    AzureConfig, GcsConfig, MockAzureStorage, MockGcsStorage, MockS3Storage, S3Config,
};
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
pub use dependency_installer::{
    DownloadOptions, InstallationPlan, InstallationProgress, ParallelDependencyInstaller,
    PlannedPackage,
};
pub use dependency_lockfile::{
    LockedDependency, LockedPackage, LockfileGenerator, LockfileValidator, PackageLockfile,
};
pub use dependency_solver::{
    Assignment, CdclSolver, DependencySatSolver, DependencySolution, SatClause, SatLiteral,
    SatVariable, VersionConstraint,
};
pub use diagnostics::{
    DiagnosticIssue, DiagnosticReport, HealthStatus, IssueCategory, IssueSeverity,
    PackageDiagnostics, PackageStatistics, ResourceValidation, SecurityAssessment,
    ValidationResult,
};
pub use exporter::{ExportConfig, PackageExporter};
pub use format_compat::{
    FormatCompatibilityManager, FormatConverter, HuggingFaceConverter, PackageFormat,
    PyTorchConverter,
};
pub use governance::{
    ComplianceIssue, ComplianceLevel, ComplianceMetadata, ComplianceReport,
    IssueSeverity as ComplianceIssueSeverity, LineageEdge, LineageQueryResult, LineageRelation,
    LineageStatistics, LineageTracker, ProvenanceInfo, TransformationRecord,
};
pub use importer::{ImportConfig, PackageImporter};
pub use lazy_resources::{
    EvictionStrategy, LazyResource, LazyResourceManager, MappedResource, ResourceStreamWriter,
    StreamingResource,
};
pub use manifest::{ModuleInfo, PackageManifest, ResourceInfo};
pub use mirror::{
    FailoverConfig, Mirror, MirrorConfig, MirrorHealth, MirrorManager, MirrorSelection,
    MirrorStatistics, SelectionStrategy, SyncStatus,
};
pub use monitoring::{
    Alert, AlertSeverity, AlertThreshold, AnalyticsReport, MetricPoint, MetricType,
    MetricsCollector, PackageStats, RegionStats, TimeSeries, TimeSeriesStats, UserStats,
};
pub use optimization::{
    CompressibleResource, CompressionAnalysis, DeduplicationAnalysis, OptimizationOpportunity,
    OptimizationReport, OptimizationType, PackageOptimizer,
};
pub use package::Package;
pub use profiling::{
    global_profiler, profile, OperationProfiler, PackageOperation, ProfileEntry, ProfileGuard,
    ProfileStats,
};
pub use replication::{
    ConflictResolution, ConsistencyLevel, NodeStatus, OperationStatus, ReplicaMetadata,
    ReplicationConfig, ReplicationConflict, ReplicationManager, ReplicationNode,
    ReplicationOperation, ReplicationStatistics, ReplicationStrategy,
};
pub use resources::{Resource, ResourceType};
pub use sandbox::{
    CapabilitySet, FilesystemPolicy, NetworkPolicy, PortRange, ResourceLimits, ResourceMonitor,
    ResourceUsageStats, Sandbox, SandboxConfig, SandboxResult, SandboxViolation, ViolationSeverity,
    ViolationType,
};
pub use security::{
    EncryptedPackage, EncryptionAlgorithm, PackageEncryptor, PackageSignature, PackageSigner,
    SignatureAlgorithm,
};
pub use storage::{LocalStorage, StorageBackend, StorageManager, StorageObject, StorageStats};
pub use syslog_integration::{
    SyslogClient, SyslogConfig, SyslogFacility, SyslogProtocol, SyslogSeverity, SyslogStatistics,
    SyslogTransport,
};
#[cfg(feature = "with-nn")]
pub use utils::export_module;
pub use utils::{
    calculate_checksum, calculate_hash, estimate_compression_ratio, format_file_size,
    get_file_extension, get_relative_path, import_module, is_safe_path, normalize_path,
    parse_content_type, sanitize_filename, validate_package_metadata, validate_package_name,
    validate_resource_path, validate_version, verify_checksum, MemoryStats, PerformanceTimer,
};
pub use version::{CompatibilityChecker, PackageVersion, VersionComparator, VersionRequirement};
pub use vulnerability::{
    IssueType, ScanPolicy, ScanReport, SecurityIssue, Severity, VulnerabilityScanner,
};

#[cfg(feature = "async")]
pub use async_ops::{AsyncPackageLoader, AsyncPackageSaver, BackgroundProcessor, DownloadProgress};
pub use registry::{PackageCache, PackageMetadata, RegistryClient, RegistryConfig, SearchResult};

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
