//! Dependency lockfile management for reproducible builds
//!
//! This module provides functionality for generating and validating lockfiles
//! that ensure reproducible dependency resolution across different environments.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use torsh_core::error::{Result, TorshError};

use crate::dependency::{DependencySpec, ResolvedDependency};
use crate::version::PackageVersion;

/// Lockfile format version
pub const LOCKFILE_VERSION: &str = "1.0.0";

/// Package lockfile containing resolved dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageLockfile {
    /// Lockfile format version
    pub version: String,
    /// Root package name
    pub package_name: String,
    /// Root package version
    pub package_version: String,
    /// When the lockfile was generated
    pub generated_at: DateTime<Utc>,
    /// Locked dependencies with resolved versions
    pub dependencies: Vec<LockedDependency>,
    /// Metadata about the lockfile generation
    pub metadata: LockfileMetadata,
}

/// Locked dependency with exact version and integrity hash
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockedDependency {
    /// Package name
    pub name: String,
    /// Exact resolved version
    pub version: String,
    /// Version requirement that led to this resolution
    pub version_req: String,
    /// Integrity hash (SHA-256) of the package
    pub integrity: String,
    /// Dependencies of this package
    pub dependencies: Vec<String>,
    /// Whether this is an optional dependency
    pub optional: bool,
    /// Platform-specific requirement
    pub platform: Option<String>,
    /// Registry URL where package was resolved from
    pub registry: String,
}

/// Metadata about lockfile generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockfileMetadata {
    /// Total number of dependencies
    pub total_dependencies: usize,
    /// Dependency resolution strategy used
    pub resolution_strategy: String,
    /// Platform the lockfile was generated on
    pub platform: String,
    /// Generator tool and version
    pub generator: String,
}

/// Locked package representation for flat storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockedPackage {
    /// Package name
    pub name: String,
    /// Exact version
    pub version: String,
    /// Integrity hash
    pub integrity: String,
    /// Direct dependencies
    pub dependencies: HashMap<String, String>,
}

/// Lockfile generator for creating lockfiles from resolved dependencies
pub struct LockfileGenerator {
    /// Whether to include optional dependencies
    include_optional: bool,
    /// Whether to include platform-specific dependencies
    include_platform_specific: bool,
}

/// Lockfile validator for checking lockfile integrity
pub struct LockfileValidator {
    /// Whether to allow missing optional dependencies
    allow_missing_optional: bool,
    /// Whether to strict-check integrity hashes
    strict_integrity: bool,
}

impl PackageLockfile {
    /// Create a new lockfile
    pub fn new(package_name: String, package_version: String) -> Self {
        Self {
            version: LOCKFILE_VERSION.to_string(),
            package_name,
            package_version,
            generated_at: Utc::now(),
            dependencies: Vec::new(),
            metadata: LockfileMetadata {
                total_dependencies: 0,
                resolution_strategy: "highest".to_string(),
                platform: std::env::consts::OS.to_string(),
                generator: format!("torsh-package/{}", env!("CARGO_PKG_VERSION")),
            },
        }
    }

    /// Load lockfile from a file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let contents = fs::read_to_string(path)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to read lockfile: {}", e)))?;

        let lockfile: Self = serde_json::from_str(&contents)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to parse lockfile: {}", e)))?;

        // Validate lockfile format version
        let lockfile_version = PackageVersion::parse(&lockfile.version)
            .map_err(|e| TorshError::InvalidArgument(format!("Invalid lockfile version: {}", e)))?;

        let current_version = PackageVersion::parse(LOCKFILE_VERSION)
            .map_err(|e| TorshError::InvalidArgument(format!("Invalid current version: {}", e)))?;

        if lockfile_version.major != current_version.major {
            return Err(TorshError::InvalidArgument(format!(
                "Incompatible lockfile version: {} (expected major version {})",
                lockfile.version, current_version.major
            )));
        }

        Ok(lockfile)
    }

    /// Save lockfile to a file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let contents = serde_json::to_string_pretty(self).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to serialize lockfile: {}", e))
        })?;

        fs::write(path, contents)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to write lockfile: {}", e)))?;

        Ok(())
    }

    /// Add a locked dependency
    pub fn add_dependency(&mut self, dependency: LockedDependency) {
        self.dependencies.push(dependency);
        self.metadata.total_dependencies = self.dependencies.len();
    }

    /// Get a dependency by name
    pub fn get_dependency(&self, name: &str) -> Option<&LockedDependency> {
        self.dependencies.iter().find(|d| d.name == name)
    }

    /// Get all dependencies as a map
    pub fn get_dependency_map(&self) -> HashMap<String, String> {
        self.dependencies
            .iter()
            .map(|d| (d.name.clone(), d.version.clone()))
            .collect()
    }

    /// Check if lockfile is outdated (older than specified days)
    pub fn is_outdated(&self, days: i64) -> bool {
        let now = Utc::now();
        let age = now.signed_duration_since(self.generated_at);
        age.num_days() > days
    }

    /// Get statistics about the lockfile
    pub fn get_statistics(&self) -> LockfileStatistics {
        let mut optional_count = 0;
        let mut platform_specific_count = 0;
        let mut total_size = 0;

        for dep in &self.dependencies {
            if dep.optional {
                optional_count += 1;
            }
            if dep.platform.is_some() {
                platform_specific_count += 1;
            }
            // Estimate size from integrity hash (simplified)
            total_size += 1024; // Placeholder
        }

        LockfileStatistics {
            total_dependencies: self.dependencies.len(),
            optional_dependencies: optional_count,
            platform_specific_dependencies: platform_specific_count,
            estimated_total_size: total_size,
            generation_date: self.generated_at,
        }
    }
}

/// Lockfile statistics
#[derive(Debug, Clone)]
pub struct LockfileStatistics {
    /// Total number of dependencies
    pub total_dependencies: usize,
    /// Number of optional dependencies
    pub optional_dependencies: usize,
    /// Number of platform-specific dependencies
    pub platform_specific_dependencies: usize,
    /// Estimated total download size in bytes
    pub estimated_total_size: u64,
    /// When the lockfile was generated
    pub generation_date: DateTime<Utc>,
}

impl LockedDependency {
    /// Create a new locked dependency
    pub fn new(name: String, version: String, version_req: String, integrity: String) -> Self {
        Self {
            name,
            version,
            version_req,
            integrity,
            dependencies: Vec::new(),
            optional: false,
            platform: None,
            registry: "https://packages.torsh.rs".to_string(),
        }
    }

    /// Verify integrity hash against actual package data
    pub fn verify_integrity(&self, package_data: &[u8]) -> bool {
        let computed_hash = Self::compute_integrity(package_data);
        self.integrity == computed_hash
    }

    /// Compute integrity hash for package data
    pub fn compute_integrity(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("sha256-{}", hex::encode(hasher.finalize()))
    }
}

impl Default for LockfileGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl LockfileGenerator {
    /// Create a new lockfile generator
    pub fn new() -> Self {
        Self {
            include_optional: true,
            include_platform_specific: true,
        }
    }

    /// Set whether to include optional dependencies
    pub fn with_optional(mut self, include: bool) -> Self {
        self.include_optional = include;
        self
    }

    /// Set whether to include platform-specific dependencies
    pub fn with_platform_specific(mut self, include: bool) -> Self {
        self.include_platform_specific = include;
        self
    }

    /// Generate lockfile from resolved dependencies
    pub fn generate(
        &self,
        package_name: String,
        package_version: String,
        resolved: &[ResolvedDependency],
    ) -> Result<PackageLockfile> {
        let mut lockfile = PackageLockfile::new(package_name, package_version);

        for dep in resolved {
            // Skip optional dependencies if not included
            if dep.spec.optional && !self.include_optional {
                continue;
            }

            // Skip platform-specific dependencies if not included
            if dep.spec.platform.is_some() && !self.include_platform_specific {
                continue;
            }

            let locked_dep = LockedDependency {
                name: dep.spec.name.clone(),
                version: dep.resolved_version.clone(),
                version_req: dep.spec.version_req.clone(),
                integrity: "sha256-placeholder".to_string(), // Would compute from actual package
                dependencies: dep
                    .dependencies
                    .iter()
                    .map(|d| d.spec.name.clone())
                    .collect(),
                optional: dep.spec.optional,
                platform: dep.spec.platform.clone(),
                registry: "https://packages.torsh.rs".to_string(),
            };

            lockfile.add_dependency(locked_dep);
        }

        Ok(lockfile)
    }

    /// Generate lockfile from dependency specs
    pub fn generate_from_specs(
        &self,
        package_name: String,
        package_version: String,
        specs: &[DependencySpec],
    ) -> Result<PackageLockfile> {
        // Convert specs to resolved dependencies
        let resolved: Vec<_> = specs
            .iter()
            .map(|spec| ResolvedDependency {
                spec: spec.clone(),
                resolved_version: "0.0.0".to_string(), // Placeholder
                package_path: None,
                dependencies: Vec::new(),
            })
            .collect();

        self.generate(package_name, package_version, &resolved)
    }
}

impl Default for LockfileValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl LockfileValidator {
    /// Create a new lockfile validator
    pub fn new() -> Self {
        Self {
            allow_missing_optional: true,
            strict_integrity: true,
        }
    }

    /// Set whether to allow missing optional dependencies
    pub fn with_allow_missing_optional(mut self, allow: bool) -> Self {
        self.allow_missing_optional = allow;
        self
    }

    /// Set whether to strictly check integrity hashes
    pub fn with_strict_integrity(mut self, strict: bool) -> Self {
        self.strict_integrity = strict;
        self
    }

    /// Validate a lockfile
    pub fn validate(&self, lockfile: &PackageLockfile) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();

        // Check lockfile version
        let lockfile_version = PackageVersion::parse(&lockfile.version)
            .map_err(|e| TorshError::InvalidArgument(format!("Invalid lockfile version: {}", e)))?;

        let current_version = PackageVersion::parse(LOCKFILE_VERSION)
            .map_err(|e| TorshError::InvalidArgument(format!("Invalid current version: {}", e)))?;

        if lockfile_version.major != current_version.major {
            report.add_error(format!(
                "Incompatible lockfile version: {} (expected major version {})",
                lockfile.version, current_version.major
            ));
        }

        // Check for duplicate dependencies
        let mut seen = HashMap::new();
        for dep in &lockfile.dependencies {
            if let Some(existing) = seen.insert(&dep.name, &dep.version) {
                report.add_warning(format!(
                    "Duplicate dependency: {} (versions {} and {})",
                    dep.name, existing, dep.version
                ));
            }
        }

        // Validate each dependency
        for dep in &lockfile.dependencies {
            self.validate_dependency(dep, &mut report)?;
        }

        // Check for circular dependencies
        if let Err(e) = self.check_circular_dependencies(lockfile) {
            report.add_error(format!("Circular dependency detected: {}", e));
        }

        Ok(report)
    }

    /// Validate a single dependency
    fn validate_dependency(
        &self,
        dep: &LockedDependency,
        report: &mut ValidationReport,
    ) -> Result<()> {
        // Validate version format
        if PackageVersion::parse(&dep.version).is_err() {
            report.add_error(format!("Invalid version for {}: {}", dep.name, dep.version));
        }

        // Check integrity hash format
        if self.strict_integrity && !dep.integrity.starts_with("sha256-") {
            report.add_error(format!(
                "Invalid integrity hash format for {}: {}",
                dep.name, dep.integrity
            ));
        }

        // Check optional dependency handling
        if dep.optional && !self.allow_missing_optional {
            report.add_warning(format!(
                "Optional dependency {} may not be installed",
                dep.name
            ));
        }

        Ok(())
    }

    /// Check for circular dependencies in the lockfile
    fn check_circular_dependencies(&self, lockfile: &PackageLockfile) -> Result<()> {
        let mut graph: HashMap<String, Vec<String>> = HashMap::new();

        for dep in &lockfile.dependencies {
            graph.insert(dep.name.clone(), dep.dependencies.clone());
        }

        // Perform cycle detection using DFS
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for dep in &lockfile.dependencies {
            if !visited.contains(&dep.name) {
                if self.has_cycle(&dep.name, &graph, &mut visited, &mut rec_stack)? {
                    return Err(TorshError::InvalidArgument(format!(
                        "Circular dependency involving: {}",
                        dep.name
                    )));
                }
            }
        }

        Ok(())
    }

    /// DFS-based cycle detection
    fn has_cycle(
        &self,
        node: &str,
        graph: &HashMap<String, Vec<String>>,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> Result<bool> {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());

        if let Some(neighbors) = graph.get(node) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    if self.has_cycle(neighbor, graph, visited, rec_stack)? {
                        return Ok(true);
                    }
                } else if rec_stack.contains(neighbor) {
                    return Ok(true);
                }
            }
        }

        rec_stack.remove(node);
        Ok(false)
    }

    /// Compare two lockfiles and report differences
    pub fn compare_lockfiles(&self, old: &PackageLockfile, new: &PackageLockfile) -> LockfileDiff {
        let mut diff = LockfileDiff::new();

        let old_map: HashMap<_, _> = old
            .dependencies
            .iter()
            .map(|d| (&d.name, &d.version))
            .collect();

        let new_map: HashMap<_, _> = new
            .dependencies
            .iter()
            .map(|d| (&d.name, &d.version))
            .collect();

        // Find added, removed, and updated dependencies
        for (name, new_version) in &new_map {
            if let Some(old_version) = old_map.get(name) {
                if old_version != new_version {
                    diff.updated.push((
                        name.to_string(),
                        old_version.to_string(),
                        new_version.to_string(),
                    ));
                }
            } else {
                diff.added.push((name.to_string(), new_version.to_string()));
            }
        }

        for (name, old_version) in &old_map {
            if !new_map.contains_key(name) {
                diff.removed
                    .push((name.to_string(), old_version.to_string()));
            }
        }

        diff
    }
}

use std::collections::HashSet;

/// Validation report for lockfile validation
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Validation errors (critical issues)
    pub errors: Vec<String>,
    /// Validation warnings (non-critical issues)
    pub warnings: Vec<String>,
}

impl ValidationReport {
    /// Create a new validation report
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Add an error
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }

    /// Add a warning
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Check if validation passed
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Get summary of validation
    pub fn summary(&self) -> String {
        format!(
            "Validation: {} errors, {} warnings",
            self.errors.len(),
            self.warnings.len()
        )
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Difference between two lockfiles
#[derive(Debug, Clone)]
pub struct LockfileDiff {
    /// Added dependencies (name, version)
    pub added: Vec<(String, String)>,
    /// Removed dependencies (name, version)
    pub removed: Vec<(String, String)>,
    /// Updated dependencies (name, old_version, new_version)
    pub updated: Vec<(String, String, String)>,
}

impl LockfileDiff {
    /// Create a new lockfile diff
    pub fn new() -> Self {
        Self {
            added: Vec::new(),
            removed: Vec::new(),
            updated: Vec::new(),
        }
    }

    /// Check if there are any differences
    pub fn has_changes(&self) -> bool {
        !self.added.is_empty() || !self.removed.is_empty() || !self.updated.is_empty()
    }

    /// Get summary of differences
    pub fn summary(&self) -> String {
        format!(
            "{} added, {} removed, {} updated",
            self.added.len(),
            self.removed.len(),
            self.updated.len()
        )
    }
}

impl Default for LockfileDiff {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lockfile_creation() {
        let lockfile = PackageLockfile::new("test-package".to_string(), "1.0.0".to_string());
        assert_eq!(lockfile.package_name, "test-package");
        assert_eq!(lockfile.package_version, "1.0.0");
        assert_eq!(lockfile.version, LOCKFILE_VERSION);
    }

    #[test]
    fn test_locked_dependency() {
        let dep = LockedDependency::new(
            "test-dep".to_string(),
            "1.0.0".to_string(),
            "^1.0.0".to_string(),
            "sha256-abc123".to_string(),
        );

        assert_eq!(dep.name, "test-dep");
        assert_eq!(dep.version, "1.0.0");
        assert_eq!(dep.integrity, "sha256-abc123");
    }

    #[test]
    fn test_integrity_verification() {
        let data = b"test package data";
        let integrity = LockedDependency::compute_integrity(data);

        assert!(integrity.starts_with("sha256-"));

        let dep = LockedDependency::new(
            "test".to_string(),
            "1.0.0".to_string(),
            "^1.0.0".to_string(),
            integrity,
        );

        assert!(dep.verify_integrity(data));
        assert!(!dep.verify_integrity(b"different data"));
    }

    #[test]
    fn test_lockfile_generator() {
        let generator = LockfileGenerator::new().with_optional(true);

        let specs = vec![DependencySpec::new(
            "test-dep".to_string(),
            "^1.0.0".to_string(),
        )];

        let lockfile = generator
            .generate_from_specs("test-package".to_string(), "1.0.0".to_string(), &specs)
            .unwrap();

        assert_eq!(lockfile.dependencies.len(), 1);
        assert_eq!(lockfile.dependencies[0].name, "test-dep");
    }

    #[test]
    fn test_lockfile_validator() {
        let mut lockfile = PackageLockfile::new("test-package".to_string(), "1.0.0".to_string());

        let dep = LockedDependency::new(
            "test-dep".to_string(),
            "1.0.0".to_string(),
            "^1.0.0".to_string(),
            "sha256-abc123".to_string(),
        );

        lockfile.add_dependency(dep);

        let validator = LockfileValidator::new();
        let report = validator.validate(&lockfile).unwrap();

        assert!(report.is_valid());
    }

    #[test]
    fn test_lockfile_diff() {
        let mut old = PackageLockfile::new("test-package".to_string(), "1.0.0".to_string());
        old.add_dependency(LockedDependency::new(
            "dep1".to_string(),
            "1.0.0".to_string(),
            "^1.0.0".to_string(),
            "sha256-abc".to_string(),
        ));

        let mut new = PackageLockfile::new("test-package".to_string(), "1.0.0".to_string());
        new.add_dependency(LockedDependency::new(
            "dep1".to_string(),
            "1.1.0".to_string(),
            "^1.0.0".to_string(),
            "sha256-def".to_string(),
        ));
        new.add_dependency(LockedDependency::new(
            "dep2".to_string(),
            "2.0.0".to_string(),
            "^2.0.0".to_string(),
            "sha256-ghi".to_string(),
        ));

        let validator = LockfileValidator::new();
        let diff = validator.compare_lockfiles(&old, &new);

        assert_eq!(diff.updated.len(), 1);
        assert_eq!(diff.added.len(), 1);
        assert!(diff.has_changes());
    }
}
