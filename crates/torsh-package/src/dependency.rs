//! Dependency resolution and management for packages
//!
//! This module provides functionality for resolving package dependencies,
//! handling version conflicts, and automatically installing required packages.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;

use serde::{Deserialize, Serialize};
use torsh_core::error::{Result, TorshError};

use crate::package::Package;
use crate::version::{PackageVersion, VersionRequirement};

/// Dependency specification with version requirements
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DependencySpec {
    /// Name of the dependency package
    pub name: String,
    /// Version requirement (e.g., "^1.0.0", ">=2.0.0")
    pub version_req: String,
    /// Optional features to enable
    pub features: Vec<String>,
    /// Whether this dependency is optional
    pub optional: bool,
    /// Platform-specific requirements
    pub platform: Option<String>,
}

/// Resolved dependency with specific version
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedDependency {
    /// Dependency specification
    pub spec: DependencySpec,
    /// Resolved version
    pub resolved_version: String,
    /// Path to the package file
    pub package_path: Option<String>,
    /// Transitive dependencies
    pub dependencies: Vec<ResolvedDependency>,
}

/// Dependency resolution strategy
#[derive(Debug, Clone, Copy)]
pub enum ResolutionStrategy {
    /// Use the highest compatible version
    Highest,
    /// Use the lowest compatible version
    Lowest,
    /// Use the most recent stable version
    Stable,
}

/// Dependency resolver for handling complex dependency graphs
pub struct DependencyResolver {
    /// Resolution strategy to use
    strategy: ResolutionStrategy,
    /// Package registry to search for dependencies
    registry: Box<dyn PackageRegistry>,
    /// Maximum dependency depth to prevent infinite loops
    max_depth: usize,
    /// Enable parallel resolution
    parallel_resolution: bool,
}

/// Package registry trait for abstracting package sources
pub trait PackageRegistry: Send + Sync {
    /// Search for packages matching the given name pattern
    fn search_packages(&self, name_pattern: &str) -> Result<Vec<PackageInfo>>;

    /// Get available versions for a package
    fn get_versions(&self, package_name: &str) -> Result<Vec<String>>;

    /// Download a specific package version
    fn download_package(&self, name: &str, version: &str, dest_path: &Path) -> Result<()>;

    /// Get package metadata without downloading
    fn get_package_info(&self, name: &str, version: &str) -> Result<PackageInfo>;
}

/// Package information from registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageInfo {
    /// Package name
    pub name: String,
    /// Package version
    pub version: String,
    /// Package description
    pub description: Option<String>,
    /// Package author
    pub author: Option<String>,
    /// Package dependencies
    pub dependencies: Vec<DependencySpec>,
    /// Package size in bytes
    pub size: u64,
    /// Package checksum
    pub checksum: String,
    /// Registry URL where package can be found
    pub registry_url: String,
}

/// Dependency conflict information
#[derive(Debug, Clone)]
pub struct DependencyConflict {
    /// Package name with conflict
    pub package_name: String,
    /// Conflicting version requirements
    pub conflicts: Vec<(String, String)>, // (source, requirement)
    /// Suggested resolution
    pub suggestion: Option<String>,
}

/// Dependency graph for analyzing relationships
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Nodes in the graph (package name -> package info)
    nodes: HashMap<String, PackageInfo>,
    /// Edges in the graph (dependent -> [dependencies])
    edges: HashMap<String, Vec<String>>,
    /// Resolved versions
    resolved_versions: HashMap<String, String>,
}

impl DependencySpec {
    /// Create a new dependency specification
    pub fn new(name: String, version_req: String) -> Self {
        Self {
            name,
            version_req,
            features: Vec::new(),
            optional: false,
            platform: None,
        }
    }

    /// Add a feature requirement
    pub fn with_feature(mut self, feature: String) -> Self {
        self.features.push(feature);
        self
    }

    /// Mark as optional dependency
    pub fn optional(mut self) -> Self {
        self.optional = true;
        self
    }

    /// Add platform requirement
    pub fn for_platform(mut self, platform: String) -> Self {
        self.platform = Some(platform);
        self
    }

    /// Check if this dependency is compatible with the current platform
    pub fn is_compatible_platform(&self) -> bool {
        // Simplified platform check - in production, you'd check actual platform
        self.platform
            .as_ref()
            .map_or(true, |p| p == "any" || p == std::env::consts::OS)
    }

    /// Check if a version satisfies this dependency requirement
    pub fn is_satisfied_by(&self, version: &str) -> Result<bool> {
        let requirement = VersionRequirement::parse(&self.version_req).map_err(|e| {
            TorshError::InvalidArgument(format!("Invalid version requirement: {}", e))
        })?;
        let package_version = PackageVersion::parse(version)
            .map_err(|e| TorshError::InvalidArgument(format!("Invalid version: {}", e)))?;
        Ok(requirement.matches(&package_version))
    }
}

impl Default for DependencyResolver {
    fn default() -> Self {
        Self::new(Box::new(LocalPackageRegistry::default()))
    }
}

impl DependencyResolver {
    /// Create a new dependency resolver
    pub fn new(registry: Box<dyn PackageRegistry>) -> Self {
        Self {
            strategy: ResolutionStrategy::Highest,
            registry,
            max_depth: 100,
            parallel_resolution: false,
        }
    }

    /// Set resolution strategy
    pub fn with_strategy(mut self, strategy: ResolutionStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set maximum dependency depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Enable parallel resolution
    pub fn with_parallel_resolution(mut self, parallel: bool) -> Self {
        self.parallel_resolution = parallel;
        self
    }

    /// Resolve dependencies for a package
    pub fn resolve_dependencies(&self, package: &Package) -> Result<Vec<ResolvedDependency>> {
        let mut resolved = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Start with direct dependencies
        for (name, version_req) in &package.manifest.dependencies {
            let spec = DependencySpec::new(name.clone(), version_req.clone());
            queue.push_back((spec, 0)); // (spec, depth)
        }

        while let Some((spec, depth)) = queue.pop_front() {
            if depth >= self.max_depth {
                return Err(TorshError::InvalidArgument(format!(
                    "Maximum dependency depth exceeded for package: {}",
                    spec.name
                )));
            }

            if visited.contains(&spec.name) {
                continue;
            }

            if !spec.is_compatible_platform() {
                continue; // Skip platform-incompatible dependencies
            }

            // Resolve the specific version
            let resolved_version = self.resolve_version(&spec)?;
            let package_info = self
                .registry
                .get_package_info(&spec.name, &resolved_version)?;

            // Add transitive dependencies to queue
            for dep in &package_info.dependencies {
                if !visited.contains(&dep.name) && !dep.optional {
                    queue.push_back((dep.clone(), depth + 1));
                }
            }

            let resolved_dep = ResolvedDependency {
                spec: spec.clone(),
                resolved_version,
                package_path: None,       // Will be set during installation
                dependencies: Vec::new(), // Will be populated recursively
            };

            resolved.push(resolved_dep);
            visited.insert(spec.name.clone());
        }

        // Check for conflicts
        self.check_conflicts(&resolved)?;

        Ok(resolved)
    }

    /// Resolve a specific version for a dependency
    fn resolve_version(&self, spec: &DependencySpec) -> Result<String> {
        let available_versions = self.registry.get_versions(&spec.name)?;

        if available_versions.is_empty() {
            return Err(TorshError::InvalidArgument(format!(
                "No versions found for package: {}",
                spec.name
            )));
        }

        // Filter compatible versions
        let mut compatible_versions = Vec::new();
        for version in &available_versions {
            if spec.is_satisfied_by(version)? {
                compatible_versions.push(version.clone());
            }
        }

        if compatible_versions.is_empty() {
            return Err(TorshError::InvalidArgument(format!(
                "No compatible versions found for package: {} with requirement: {}",
                spec.name, spec.version_req
            )));
        }

        // Apply resolution strategy
        let selected_version = match self.strategy {
            ResolutionStrategy::Highest => self.select_highest_version(&compatible_versions)?,
            ResolutionStrategy::Lowest => self.select_lowest_version(&compatible_versions)?,
            ResolutionStrategy::Stable => self.select_stable_version(&compatible_versions)?,
        };

        Ok(selected_version)
    }

    /// Select the highest compatible version
    fn select_highest_version(&self, versions: &[String]) -> Result<String> {
        let mut parsed_versions: Vec<_> = versions
            .iter()
            .map(|v| (v, PackageVersion::parse(v)))
            .filter_map(|(v, parsed)| parsed.ok().map(|p| (v.clone(), p)))
            .collect();

        parsed_versions.sort_by(|a, b| b.1.cmp(&a.1));

        parsed_versions
            .first()
            .map(|(version, _)| version.clone())
            .ok_or_else(|| TorshError::InvalidArgument("No valid versions found".to_string()))
    }

    /// Select the lowest compatible version
    fn select_lowest_version(&self, versions: &[String]) -> Result<String> {
        let mut parsed_versions: Vec<_> = versions
            .iter()
            .map(|v| (v, PackageVersion::parse(v)))
            .filter_map(|(v, parsed)| parsed.ok().map(|p| (v.clone(), p)))
            .collect();

        parsed_versions.sort_by(|a, b| a.1.cmp(&b.1));

        parsed_versions
            .first()
            .map(|(version, _)| version.clone())
            .ok_or_else(|| TorshError::InvalidArgument("No valid versions found".to_string()))
    }

    /// Select the most recent stable version (non-prerelease)
    fn select_stable_version(&self, versions: &[String]) -> Result<String> {
        let mut stable_versions: Vec<_> = versions
            .iter()
            .map(|v| (v, PackageVersion::parse(v)))
            .filter_map(|(v, parsed)| {
                parsed.ok().and_then(|p| {
                    if p.pre_release.is_none() {
                        // No prerelease
                        Some((v.clone(), p))
                    } else {
                        None
                    }
                })
            })
            .collect();

        if stable_versions.is_empty() {
            // Fall back to highest version if no stable versions available
            return self.select_highest_version(versions);
        }

        stable_versions.sort_by(|a, b| b.1.cmp(&a.1));

        stable_versions
            .first()
            .map(|(version, _)| version.clone())
            .ok_or_else(|| TorshError::InvalidArgument("No stable versions found".to_string()))
    }

    /// Check for dependency conflicts
    fn check_conflicts(&self, resolved: &[ResolvedDependency]) -> Result<()> {
        let mut package_versions: HashMap<String, Vec<String>> = HashMap::new();

        for dep in resolved {
            package_versions
                .entry(dep.spec.name.clone())
                .or_default()
                .push(dep.resolved_version.clone());
        }

        let mut conflicts = Vec::new();
        for (package_name, versions) in &package_versions {
            let unique_versions: HashSet<_> = versions.iter().collect();
            if unique_versions.len() > 1 {
                let conflict = DependencyConflict {
                    package_name: package_name.clone(),
                    conflicts: versions
                        .iter()
                        .map(|v| (package_name.clone(), v.clone()))
                        .collect(),
                    suggestion: Some(format!("Use version {}", versions[0])),
                };
                conflicts.push(conflict);
            }
        }

        if !conflicts.is_empty() {
            let conflict_descriptions: Vec<String> = conflicts
                .iter()
                .map(|c| {
                    format!(
                        "Package '{}' has conflicting version requirements",
                        c.package_name
                    )
                })
                .collect();

            return Err(TorshError::InvalidArgument(format!(
                "Dependency conflicts detected: {}",
                conflict_descriptions.join(", ")
            )));
        }

        Ok(())
    }

    /// Install resolved dependencies
    pub fn install_dependencies(
        &self,
        resolved: &mut [ResolvedDependency],
        install_dir: &Path,
    ) -> Result<()> {
        for dep in resolved {
            let package_path = install_dir.join(format!(
                "{}-{}.torshpkg",
                dep.spec.name, dep.resolved_version
            ));

            // Download the package
            self.registry
                .download_package(&dep.spec.name, &dep.resolved_version, &package_path)?;

            // Update the package path
            dep.package_path = Some(package_path.to_string_lossy().to_string());
        }

        Ok(())
    }

    /// Build dependency graph for analysis
    pub fn build_dependency_graph(&self, package: &Package) -> Result<DependencyGraph> {
        let resolved = self.resolve_dependencies(package)?;
        let mut graph = DependencyGraph::new();

        for dep in &resolved {
            let package_info = self
                .registry
                .get_package_info(&dep.spec.name, &dep.resolved_version)?;
            graph.add_package(package_info);
        }

        Ok(graph)
    }
}

impl DependencyGraph {
    /// Create a new empty dependency graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            resolved_versions: HashMap::new(),
        }
    }

    /// Add a package to the graph
    pub fn add_package(&mut self, package_info: PackageInfo) {
        let package_name = package_info.name.clone();
        self.resolved_versions
            .insert(package_name.clone(), package_info.version.clone());

        // Add dependencies as edges
        let mut dependencies = Vec::new();
        for dep in &package_info.dependencies {
            dependencies.push(dep.name.clone());
        }
        self.edges.insert(package_name.clone(), dependencies);

        self.nodes.insert(package_name, package_info);
    }

    /// Get topological ordering of dependencies
    pub fn topological_sort(&self) -> Result<Vec<String>> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut in_stack = HashSet::new();

        for package_name in self.nodes.keys() {
            if !visited.contains(package_name) {
                self.topological_sort_util(package_name, &mut visited, &mut in_stack, &mut result)?;
            }
        }

        result.reverse();
        Ok(result)
    }

    /// Utility function for topological sort
    fn topological_sort_util(
        &self,
        package_name: &str,
        visited: &mut HashSet<String>,
        in_stack: &mut HashSet<String>,
        result: &mut Vec<String>,
    ) -> Result<()> {
        if in_stack.contains(package_name) {
            return Err(TorshError::InvalidArgument(format!(
                "Circular dependency detected involving package: {}",
                package_name
            )));
        }

        if visited.contains(package_name) {
            return Ok(());
        }

        visited.insert(package_name.to_string());
        in_stack.insert(package_name.to_string());

        if let Some(dependencies) = self.edges.get(package_name) {
            for dep in dependencies {
                self.topological_sort_util(dep, visited, in_stack, result)?;
            }
        }

        in_stack.remove(package_name);
        result.push(package_name.to_string());
        Ok(())
    }

    /// Get all packages in the graph
    pub fn get_packages(&self) -> &HashMap<String, PackageInfo> {
        &self.nodes
    }

    /// Get dependencies for a package
    pub fn get_dependencies(&self, package_name: &str) -> Option<&Vec<String>> {
        self.edges.get(package_name)
    }
}

/// Local package registry implementation (for testing and local development)
#[derive(Debug, Default)]
pub struct LocalPackageRegistry {
    /// Local package cache directory
    cache_dir: Option<String>,
    /// Available packages
    packages: HashMap<String, Vec<PackageInfo>>,
}

impl LocalPackageRegistry {
    /// Create a new local registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a package to the local registry
    pub fn add_package(&mut self, package_info: PackageInfo) {
        self.packages
            .entry(package_info.name.clone())
            .or_default()
            .push(package_info);
    }
}

impl PackageRegistry for LocalPackageRegistry {
    fn search_packages(&self, name_pattern: &str) -> Result<Vec<PackageInfo>> {
        let mut results = Vec::new();

        for (name, packages) in &self.packages {
            if name.contains(name_pattern) {
                results.extend(packages.iter().cloned());
            }
        }

        Ok(results)
    }

    fn get_versions(&self, package_name: &str) -> Result<Vec<String>> {
        let versions = self
            .packages
            .get(package_name)
            .map(|packages| packages.iter().map(|p| p.version.clone()).collect())
            .unwrap_or_default();

        Ok(versions)
    }

    fn download_package(&self, _name: &str, _version: &str, _dest_path: &Path) -> Result<()> {
        // Simulate download - in a real implementation, this would download from a registry
        Ok(())
    }

    fn get_package_info(&self, name: &str, version: &str) -> Result<PackageInfo> {
        let packages = self
            .packages
            .get(name)
            .ok_or_else(|| TorshError::InvalidArgument(format!("Package not found: {}", name)))?;

        packages
            .iter()
            .find(|p| p.version == version)
            .cloned()
            .ok_or_else(|| {
                TorshError::InvalidArgument(format!(
                    "Version {} not found for package: {}",
                    version, name
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_package_info(name: &str, version: &str) -> PackageInfo {
        PackageInfo {
            name: name.to_string(),
            version: version.to_string(),
            description: None,
            author: None,
            dependencies: Vec::new(),
            size: 1024,
            checksum: "abc123".to_string(),
            registry_url: "http://localhost".to_string(),
        }
    }

    #[test]
    fn test_dependency_spec_creation() {
        let spec = DependencySpec::new("test".to_string(), "^1.0.0".to_string())
            .with_feature("test-feature".to_string())
            .optional()
            .for_platform("linux".to_string());

        assert_eq!(spec.name, "test");
        assert_eq!(spec.version_req, "^1.0.0");
        assert_eq!(spec.features, vec!["test-feature"]);
        assert!(spec.optional);
        assert_eq!(spec.platform, Some("linux".to_string()));
    }

    #[test]
    fn test_dependency_spec_version_satisfaction() {
        let spec = DependencySpec::new("test".to_string(), "^1.0.0".to_string());

        assert!(spec.is_satisfied_by("1.0.0").unwrap());
        assert!(spec.is_satisfied_by("1.5.0").unwrap());
        assert!(!spec.is_satisfied_by("2.0.0").unwrap());
        assert!(!spec.is_satisfied_by("0.9.0").unwrap());
    }

    #[test]
    fn test_local_package_registry() {
        let mut registry = LocalPackageRegistry::new();
        let package_info = create_test_package_info("test-package", "1.0.0");

        registry.add_package(package_info.clone());

        let versions = registry.get_versions("test-package").unwrap();
        assert_eq!(versions, vec!["1.0.0"]);

        let retrieved_info = registry.get_package_info("test-package", "1.0.0").unwrap();
        assert_eq!(retrieved_info.name, package_info.name);
        assert_eq!(retrieved_info.version, package_info.version);
    }

    #[test]
    fn test_dependency_resolution_strategy() {
        let registry = Box::new(LocalPackageRegistry::new());
        let resolver = DependencyResolver::new(registry)
            .with_strategy(ResolutionStrategy::Highest)
            .with_max_depth(50);

        // Test that strategy is set correctly
        match resolver.strategy {
            ResolutionStrategy::Highest => (),
            _ => panic!("Strategy not set correctly"),
        }

        assert_eq!(resolver.max_depth, 50);
    }

    #[test]
    fn test_dependency_graph() {
        let mut graph = DependencyGraph::new();
        let package_info = create_test_package_info("test-package", "1.0.0");

        graph.add_package(package_info.clone());

        assert_eq!(graph.nodes.len(), 1);
        assert!(graph.nodes.contains_key("test-package"));
    }

    #[test]
    fn test_version_selection() {
        let resolver = DependencyResolver::default();
        let versions = vec![
            "1.0.0".to_string(),
            "1.5.0".to_string(),
            "2.0.0".to_string(),
        ];

        let highest = resolver.select_highest_version(&versions).unwrap();
        assert_eq!(highest, "2.0.0");

        let lowest = resolver.select_lowest_version(&versions).unwrap();
        assert_eq!(lowest, "1.0.0");
    }
}
