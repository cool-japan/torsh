//! Comprehensive error checking and version compatibility for ToRSh backends
//!
//! This module provides robust error handling, version checking, and compatibility
//! management for different backend implementations and their dependencies.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::{BackendResult, BackendType};
use std::collections::HashMap;
use torsh_core::error::TorshError;

#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec::Vec};

/// Version information for backend components
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    /// Major version number
    pub major: u32,
    /// Minor version number
    pub minor: u32,
    /// Patch version number
    pub patch: u32,
    /// Pre-release identifier (alpha, beta, rc)
    pub pre_release: Option<String>,
    /// Build metadata
    pub build: Option<String>,
}

impl Version {
    /// Create a new version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            pre_release: None,
            build: None,
        }
    }

    /// Create a version with pre-release identifier
    pub fn with_pre_release(major: u32, minor: u32, patch: u32, pre_release: String) -> Self {
        Self {
            major,
            minor,
            patch,
            pre_release: Some(pre_release),
            build: None,
        }
    }

    /// Parse version from string (e.g., "1.2.3-alpha.1+build.123")
    pub fn parse(version_str: &str) -> Result<Version, VersionError> {
        let mut parts = version_str.split('+');
        let version_part = parts.next().ok_or(VersionError::InvalidFormat)?;
        let build = parts.next().map(|s| s.to_string());

        let mut version_pre = version_part.split('-');
        let version_core = version_pre.next().ok_or(VersionError::InvalidFormat)?;
        let pre_release = version_pre.next().map(|s| s.to_string());

        let core_parts: Vec<&str> = version_core.split('.').collect();
        if core_parts.len() != 3 {
            return Err(VersionError::InvalidFormat);
        }

        let major = core_parts[0]
            .parse()
            .map_err(|_| VersionError::InvalidNumber)?;
        let minor = core_parts[1]
            .parse()
            .map_err(|_| VersionError::InvalidNumber)?;
        let patch = core_parts[2]
            .parse()
            .map_err(|_| VersionError::InvalidNumber)?;

        Ok(Version {
            major,
            minor,
            patch,
            pre_release,
            build,
        })
    }

    /// Check if this version is compatible with another version
    pub fn is_compatible_with(&self, other: &Version) -> bool {
        // Same major version and this version is >= other
        self.major == other.major && self >= other
    }

    /// Check if this is a breaking change from another version
    pub fn is_breaking_change_from(&self, other: &Version) -> bool {
        self.major != other.major
    }

    /// Get string representation
    pub fn to_string(&self) -> String {
        let mut version = format!("{}.{}.{}", self.major, self.minor, self.patch);

        if let Some(ref pre) = self.pre_release {
            version.push('-');
            version.push_str(pre);
        }

        if let Some(ref build) = self.build {
            version.push('+');
            version.push_str(build);
        }

        version
    }
}

/// Version parsing and comparison errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VersionError {
    /// Invalid version format
    InvalidFormat,
    /// Invalid number in version string
    InvalidNumber,
    /// Incompatible versions
    IncompatibleVersions(String, String),
    /// Missing required version
    MissingVersion(String),
}

impl std::fmt::Display for VersionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VersionError::InvalidFormat => write!(f, "Invalid version format"),
            VersionError::InvalidNumber => write!(f, "Invalid number in version string"),
            VersionError::IncompatibleVersions(v1, v2) => {
                write!(f, "Incompatible versions: {} and {}", v1, v2)
            }
            VersionError::MissingVersion(component) => {
                write!(f, "Missing version for component: {}", component)
            }
        }
    }
}

impl std::error::Error for VersionError {}

/// Backend dependency information
#[derive(Debug, Clone)]
pub struct BackendDependency {
    /// Name of the dependency
    pub name: String,
    /// Required version range
    pub required_version: VersionRange,
    /// Current version (if available)
    pub current_version: Option<Version>,
    /// Whether this dependency is optional
    pub optional: bool,
    /// Features required from this dependency
    pub required_features: Vec<String>,
}

/// Version range specification
#[derive(Debug, Clone)]
pub enum VersionRange {
    /// Exact version match
    Exact(Version),
    /// Minimum version (inclusive)
    Minimum(Version),
    /// Range between two versions (inclusive)
    Range(Version, Version),
    /// Compatible with (same major version, >= minor.patch)
    Compatible(Version),
    /// Any version
    Any,
}

impl VersionRange {
    /// Check if a version satisfies this range
    pub fn satisfies(&self, version: &Version) -> bool {
        match self {
            VersionRange::Exact(required) => version == required,
            VersionRange::Minimum(min) => version >= min,
            VersionRange::Range(min, max) => version >= min && version <= max,
            VersionRange::Compatible(base) => version.is_compatible_with(base),
            VersionRange::Any => true,
        }
    }

    /// Get string representation of the range
    pub fn to_string(&self) -> String {
        match self {
            VersionRange::Exact(v) => format!("={}", v.to_string()),
            VersionRange::Minimum(v) => format!(">={}", v.to_string()),
            VersionRange::Range(min, max) => format!("{}-{}", min.to_string(), max.to_string()),
            VersionRange::Compatible(v) => format!("~{}", v.to_string()),
            VersionRange::Any => "*".to_string(),
        }
    }
}

/// Comprehensive error context with debugging information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation that failed
    pub operation: String,
    /// Backend type where error occurred
    pub backend_type: Option<BackendType>,
    /// Device information
    pub device_info: Option<String>,
    /// Error location (file:line)
    pub location: Option<String>,
    /// Timestamp when error occurred
    pub timestamp: Option<String>,
    /// Additional context information
    pub details: HashMap<String, String>,
    /// Suggested solutions
    pub suggestions: Vec<String>,
    /// Error severity level
    pub severity: ErrorSeverity,
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// Fatal error - operation cannot continue
    Fatal,
    /// Error - operation failed but system can recover
    Error,
    /// Warning - operation succeeded but with issues
    Warning,
    /// Info - informational message
    Info,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new(operation: &str) -> Self {
        Self {
            operation: operation.to_string(),
            backend_type: None,
            device_info: None,
            location: None,
            timestamp: None,
            details: HashMap::new(),
            suggestions: Vec::new(),
            severity: ErrorSeverity::Error,
        }
    }

    /// Add backend type to context
    pub fn with_backend(mut self, backend_type: BackendType) -> Self {
        self.backend_type = Some(backend_type);
        self
    }

    /// Add device information
    pub fn with_device(mut self, device_info: &str) -> Self {
        self.device_info = Some(device_info.to_string());
        self
    }

    /// Add location information
    pub fn with_location(mut self, file: &str, line: u32) -> Self {
        self.location = Some(format!("{}:{}", file, line));
        self
    }

    /// Add detail information
    pub fn with_detail(mut self, key: &str, value: &str) -> Self {
        self.details.insert(key.to_string(), value.to_string());
        self
    }

    /// Add suggestion for fixing the error
    pub fn with_suggestion(mut self, suggestion: &str) -> Self {
        self.suggestions.push(suggestion.to_string());
        self
    }

    /// Set error severity
    pub fn with_severity(mut self, severity: ErrorSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Format error context for display
    pub fn format(&self) -> String {
        let mut output = format!("Error in operation: {}", self.operation);

        if let Some(ref backend) = self.backend_type {
            output.push_str(&format!("\nBackend: {:?}", backend));
        }

        if let Some(ref device) = self.device_info {
            output.push_str(&format!("\nDevice: {}", device));
        }

        if let Some(ref location) = self.location {
            output.push_str(&format!("\nLocation: {}", location));
        }

        if !self.details.is_empty() {
            output.push_str("\nDetails:");
            for (key, value) in &self.details {
                output.push_str(&format!("\n  {}: {}", key, value));
            }
        }

        if !self.suggestions.is_empty() {
            output.push_str("\nSuggestions:");
            for suggestion in &self.suggestions {
                output.push_str(&format!("\n  - {}", suggestion));
            }
        }

        output
    }
}

/// Version compatibility checker
#[derive(Debug)]
pub struct VersionCompatibilityChecker {
    /// Known backend versions
    backend_versions: HashMap<BackendType, Version>,
    /// Backend dependencies
    dependencies: HashMap<BackendType, Vec<BackendDependency>>,
    /// Compatibility matrix
    compatibility_matrix: HashMap<(BackendType, BackendType), bool>,
}

impl Default for VersionCompatibilityChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl VersionCompatibilityChecker {
    /// Create a new compatibility checker
    pub fn new() -> Self {
        let mut checker = Self {
            backend_versions: HashMap::new(),
            dependencies: HashMap::new(),
            compatibility_matrix: HashMap::new(),
        };

        // Initialize with known versions and dependencies
        checker.initialize_known_versions();
        checker.initialize_dependencies();
        checker.build_compatibility_matrix();

        checker
    }

    /// Initialize known backend versions
    fn initialize_known_versions(&mut self) {
        // These would typically be detected at runtime
        self.backend_versions
            .insert(BackendType::Cpu, Version::new(0, 1, 0));

        #[cfg(feature = "cuda")]
        self.backend_versions
            .insert(BackendType::Cuda, Version::new(0, 1, 0));

        #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
        self.backend_versions
            .insert(BackendType::Metal, Version::new(0, 1, 0));

        #[cfg(feature = "webgpu")]
        self.backend_versions
            .insert(BackendType::WebGpu, Version::new(0, 1, 0));
    }

    /// Initialize known dependencies
    fn initialize_dependencies(&mut self) {
        // CPU backend dependencies
        let cpu_deps = vec![
            BackendDependency {
                name: "scirs2-core".to_string(),
                required_version: VersionRange::Compatible(Version::new(0, 1, 0)),
                current_version: None,
                optional: false,
                required_features: vec!["cpu".to_string()],
            },
            BackendDependency {
                name: "rayon".to_string(),
                required_version: VersionRange::Minimum(Version::new(1, 5, 0)),
                current_version: None,
                optional: false,
                required_features: vec![],
            },
        ];
        self.dependencies.insert(BackendType::Cpu, cpu_deps);

        // CUDA backend dependencies
        #[cfg(feature = "cuda")]
        {
            let cuda_deps = vec![
                BackendDependency {
                    name: "scirs2-core".to_string(),
                    required_version: VersionRange::Compatible(Version::new(0, 1, 0)),
                    current_version: None,
                    optional: false,
                    required_features: vec!["cuda".to_string()],
                },
                BackendDependency {
                    name: "cuda-runtime".to_string(),
                    required_version: VersionRange::Minimum(Version::new(11, 0, 0)),
                    current_version: None,
                    optional: false,
                    required_features: vec![],
                },
            ];
            self.dependencies.insert(BackendType::Cuda, cuda_deps);
        }
    }

    /// Build compatibility matrix between backends
    fn build_compatibility_matrix(&mut self) {
        // All backends are compatible with themselves
        for backend_type in [
            BackendType::Cpu,
            BackendType::Cuda,
            BackendType::Metal,
            BackendType::WebGpu,
        ] {
            self.compatibility_matrix
                .insert((backend_type, backend_type), true);
        }

        // Cross-backend compatibility (for data transfer, etc.)
        self.compatibility_matrix
            .insert((BackendType::Cpu, BackendType::Cuda), true);
        self.compatibility_matrix
            .insert((BackendType::Cuda, BackendType::Cpu), true);
        self.compatibility_matrix
            .insert((BackendType::Cpu, BackendType::Metal), true);
        self.compatibility_matrix
            .insert((BackendType::Metal, BackendType::Cpu), true);
        // WebGPU has limited compatibility
        self.compatibility_matrix
            .insert((BackendType::Cpu, BackendType::WebGpu), true);
        self.compatibility_matrix
            .insert((BackendType::WebGpu, BackendType::Cpu), true);
    }

    /// Check if a backend is compatible with current system
    pub fn check_backend_compatibility(
        &self,
        backend_type: BackendType,
    ) -> BackendResult<CompatibilityReport> {
        let mut report = CompatibilityReport::new(backend_type);

        // Check backend version
        if let Some(version) = self.backend_versions.get(&backend_type) {
            report.backend_version = Some(version.clone());
        } else {
            report
                .errors
                .push(format!("Backend {:?} version not found", backend_type));
        }

        // Check dependencies
        if let Some(deps) = self.dependencies.get(&backend_type) {
            for dep in deps {
                let dep_status = self.check_dependency(dep);
                report
                    .dependency_status
                    .insert(dep.name.clone(), dep_status);
            }
        }

        // Check runtime requirements
        self.check_runtime_requirements(backend_type, &mut report);

        Ok(report)
    }

    /// Check a specific dependency
    fn check_dependency(&self, dependency: &BackendDependency) -> DependencyStatus {
        // In a real implementation, this would check if the dependency is available
        // and what version is installed
        DependencyStatus {
            available: true, // Assume available for now
            current_version: dependency.current_version.clone(),
            satisfies_requirement: true, // Assume satisfied
            missing_features: Vec::new(),
        }
    }

    /// Check runtime requirements for a backend
    fn check_runtime_requirements(
        &self,
        backend_type: BackendType,
        report: &mut CompatibilityReport,
    ) {
        match backend_type {
            BackendType::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    // Check CUDA runtime
                    if !self.check_cuda_runtime() {
                        report.errors.push("CUDA runtime not available".to_string());
                        report
                            .suggestions
                            .push("Install CUDA toolkit and drivers".to_string());
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    report
                        .errors
                        .push("CUDA backend not compiled in".to_string());
                    report
                        .suggestions
                        .push("Recompile with --features cuda".to_string());
                }
            }
            BackendType::Metal => {
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    // Check Metal availability
                    if !self.check_metal_availability() {
                        report
                            .errors
                            .push("Metal not available on this system".to_string());
                    }
                }
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    report
                        .errors
                        .push("Metal backend only available on macOS".to_string());
                }
            }
            BackendType::WebGpu => {
                #[cfg(feature = "webgpu")]
                {
                    // Check WebGPU support
                    if !self.check_webgpu_support() {
                        report
                            .warnings
                            .push("WebGPU support may be limited".to_string());
                    }
                }
                #[cfg(not(feature = "webgpu"))]
                {
                    report
                        .errors
                        .push("WebGPU backend not compiled in".to_string());
                }
            }
            BackendType::Cpu => {
                // CPU backend should always be available
                // Check for optimal CPU features
                self.check_cpu_features(report);
            }
            _ => {}
        }
    }

    #[cfg(feature = "cuda")]
    fn check_cuda_runtime(&self) -> bool {
        // In a real implementation, this would check for CUDA libraries
        true // Assume available for now
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    fn check_metal_availability(&self) -> bool {
        // Check if Metal is available
        true // Assume available on macOS
    }

    #[cfg(feature = "webgpu")]
    fn check_webgpu_support(&self) -> bool {
        // Check WebGPU support
        true // Assume available
    }

    fn check_cpu_features(&self, _report: &mut CompatibilityReport) {
        // Check for optimal CPU features
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if !self.has_avx2() {
                _report
                    .warnings
                    .push("AVX2 not available - performance may be reduced".to_string());
                _report
                    .suggestions
                    .push("Consider using a newer CPU with AVX2 support".to_string());
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn has_avx2(&self) -> bool {
        std::arch::is_x86_feature_detected!("avx2")
    }

    /// Check compatibility between two backends
    pub fn check_cross_backend_compatibility(&self, from: BackendType, to: BackendType) -> bool {
        self.compatibility_matrix
            .get(&(from, to))
            .copied()
            .unwrap_or(false)
    }

    /// Get all compatible backends for a given backend
    pub fn get_compatible_backends(&self, backend_type: BackendType) -> Vec<BackendType> {
        let mut compatible = Vec::new();

        for other_backend in [
            BackendType::Cpu,
            BackendType::Cuda,
            BackendType::Metal,
            BackendType::WebGpu,
        ] {
            if self.check_cross_backend_compatibility(backend_type, other_backend) {
                compatible.push(other_backend);
            }
        }

        compatible
    }
}

/// Compatibility check report
#[derive(Debug, Clone)]
pub struct CompatibilityReport {
    /// Backend being checked
    pub backend_type: BackendType,
    /// Backend version
    pub backend_version: Option<Version>,
    /// Dependency status
    pub dependency_status: HashMap<String, DependencyStatus>,
    /// Compatibility errors
    pub errors: Vec<String>,
    /// Compatibility warnings
    pub warnings: Vec<String>,
    /// Suggestions for improvement
    pub suggestions: Vec<String>,
    /// Overall compatibility score (0.0 to 1.0)
    pub compatibility_score: f32,
}

impl CompatibilityReport {
    /// Create a new compatibility report
    pub fn new(backend_type: BackendType) -> Self {
        Self {
            backend_type,
            backend_version: None,
            dependency_status: HashMap::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
            suggestions: Vec::new(),
            compatibility_score: 0.0,
        }
    }

    /// Check if the backend is fully compatible
    pub fn is_compatible(&self) -> bool {
        self.errors.is_empty()
    }

    /// Calculate compatibility score
    pub fn calculate_score(&mut self) {
        let total_checks = 1 + self.dependency_status.len(); // Backend + dependencies
        let mut passed_checks = 0;

        // Backend version check
        if self.backend_version.is_some() {
            passed_checks += 1;
        }

        // Dependency checks
        for status in self.dependency_status.values() {
            if status.satisfies_requirement {
                passed_checks += 1;
            }
        }

        self.compatibility_score = passed_checks as f32 / total_checks as f32;

        // Reduce score for warnings
        let warning_penalty = self.warnings.len() as f32 * 0.1;
        self.compatibility_score = (self.compatibility_score - warning_penalty).max(0.0);
    }

    /// Format report for display
    pub fn format(&self) -> String {
        let mut output = format!("Compatibility Report for {:?} Backend\n", self.backend_type);
        output.push_str(&format!(
            "Score: {:.1}%\n\n",
            self.compatibility_score * 100.0
        ));

        if let Some(ref version) = self.backend_version {
            output.push_str(&format!("Backend Version: {}\n", version.to_string()));
        }

        if !self.dependency_status.is_empty() {
            output.push_str("\nDependencies:\n");
            for (name, status) in &self.dependency_status {
                let status_str = if status.satisfies_requirement {
                    "✓"
                } else {
                    "✗"
                };
                output.push_str(&format!("  {} {}\n", status_str, name));
            }
        }

        if !self.errors.is_empty() {
            output.push_str("\nErrors:\n");
            for error in &self.errors {
                output.push_str(&format!("  ✗ {}\n", error));
            }
        }

        if !self.warnings.is_empty() {
            output.push_str("\nWarnings:\n");
            for warning in &self.warnings {
                output.push_str(&format!("  ⚠ {}\n", warning));
            }
        }

        if !self.suggestions.is_empty() {
            output.push_str("\nSuggestions:\n");
            for suggestion in &self.suggestions {
                output.push_str(&format!("  💡 {}\n", suggestion));
            }
        }

        output
    }
}

/// Status of a dependency check
#[derive(Debug, Clone)]
pub struct DependencyStatus {
    /// Whether the dependency is available
    pub available: bool,
    /// Current version of the dependency
    pub current_version: Option<Version>,
    /// Whether current version satisfies requirement
    pub satisfies_requirement: bool,
    /// Missing required features
    pub missing_features: Vec<String>,
}

/// Enhanced error result extension for version compatibility
pub trait VersionErrorContextExt<T> {
    /// Add operation context to error
    fn with_operation(self, operation: &str) -> Result<T, TorshError>;

    /// Add backend context to error
    fn with_backend_context(
        self,
        backend_type: BackendType,
        operation: &str,
    ) -> Result<T, TorshError>;

    /// Add location context to error
    fn with_location_context(self, file: &str, line: u32) -> Result<T, TorshError>;
}

impl<T> VersionErrorContextExt<T> for Result<T, TorshError> {
    fn with_operation(self, operation: &str) -> Result<T, TorshError> {
        self.map_err(|e| {
            let context = ErrorContext::new(operation);
            TorshError::BackendError(format!("{}\n{}", e, context.format()))
        })
    }

    fn with_backend_context(
        self,
        backend_type: BackendType,
        operation: &str,
    ) -> Result<T, TorshError> {
        self.map_err(|e| {
            let context = ErrorContext::new(operation).with_backend(backend_type);
            TorshError::BackendError(format!("{}\n{}", e, context.format()))
        })
    }

    fn with_location_context(self, file: &str, line: u32) -> Result<T, TorshError> {
        self.map_err(|e| {
            let context = ErrorContext::new("unknown").with_location(file, line);
            TorshError::BackendError(format!("{}\n{}", e, context.format()))
        })
    }
}

/// Macro for adding location context to errors
#[macro_export]
macro_rules! error_with_location {
    ($result:expr) => {
        $result.with_location_context(file!(), line!())
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parsing() {
        let version = Version::parse("1.2.3").unwrap();
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);
        assert_eq!(version.pre_release, None);

        let version = Version::parse("2.0.0-alpha.1+build.123").unwrap();
        assert_eq!(version.major, 2);
        assert_eq!(version.pre_release, Some("alpha.1".to_string()));
        assert_eq!(version.build, Some("build.123".to_string()));
    }

    #[test]
    fn test_version_compatibility() {
        let v1 = Version::new(1, 2, 3);
        let v2 = Version::new(1, 2, 4);
        let v3 = Version::new(2, 0, 0);

        assert!(v2.is_compatible_with(&v1));
        assert!(!v1.is_compatible_with(&v2));
        assert!(v3.is_breaking_change_from(&v1));
    }

    #[test]
    fn test_version_range() {
        let version = Version::new(1, 2, 3);

        let exact = VersionRange::Exact(Version::new(1, 2, 3));
        assert!(exact.satisfies(&version));

        let min = VersionRange::Minimum(Version::new(1, 0, 0));
        assert!(min.satisfies(&version));

        let compat = VersionRange::Compatible(Version::new(1, 1, 0));
        assert!(compat.satisfies(&version));
    }

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("test_operation")
            .with_backend(BackendType::Cpu)
            .with_device("cpu:0")
            .with_detail("tensor_size", "1000")
            .with_suggestion("Reduce batch size");

        let formatted = context.format();
        assert!(formatted.contains("test_operation"));
        assert!(formatted.contains("Backend: Cpu"));
        assert!(formatted.contains("Device: cpu:0"));
    }

    #[test]
    fn test_compatibility_checker() {
        let checker = VersionCompatibilityChecker::new();

        let report = checker
            .check_backend_compatibility(BackendType::Cpu)
            .unwrap();
        assert_eq!(report.backend_type, BackendType::Cpu);

        let compatible = checker.get_compatible_backends(BackendType::Cpu);
        assert!(!compatible.is_empty());
    }

    #[test]
    fn test_compatibility_report() {
        let mut report = CompatibilityReport::new(BackendType::Cpu);
        report.backend_version = Some(Version::new(0, 1, 0));
        report.calculate_score();

        assert!(report.compatibility_score > 0.0);
        assert!(report.is_compatible());
    }

    #[test]
    fn test_cross_backend_compatibility() {
        let checker = VersionCompatibilityChecker::new();

        // CPU and CUDA should be compatible
        assert!(checker.check_cross_backend_compatibility(BackendType::Cpu, BackendType::Cuda));
        assert!(checker.check_cross_backend_compatibility(BackendType::Cuda, BackendType::Cpu));
    }
}
