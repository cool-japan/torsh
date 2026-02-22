//! Parallel dependency installation with async support
//!
//! This module provides functionality for downloading and installing
//! package dependencies in parallel with progress tracking and retry logic.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use scirs2_core::parallel_ops::*;
use serde::{Deserialize, Serialize};
use torsh_core::error::{Result, TorshError};

use crate::dependency::{PackageRegistry, ResolvedDependency};
use crate::dependency_lockfile::LockedDependency;

/// Download options for parallel installation
#[derive(Debug, Clone)]
pub struct DownloadOptions {
    /// Maximum number of parallel downloads
    pub max_parallel: usize,
    /// Connection timeout in seconds
    pub timeout_secs: u64,
    /// Number of retry attempts for failed downloads
    pub max_retries: usize,
    /// Delay between retries in milliseconds
    pub retry_delay_ms: u64,
    /// Download buffer size in bytes
    pub buffer_size: usize,
    /// Whether to verify package integrity after download
    pub verify_integrity: bool,
    /// Whether to resume partial downloads
    pub resume_partial: bool,
}

/// Installation progress tracking
#[derive(Debug, Clone)]
pub struct InstallationProgress {
    /// Total number of packages to install
    pub total_packages: usize,
    /// Number of packages downloaded
    pub downloaded: Arc<AtomicUsize>,
    /// Number of packages installed
    pub installed: Arc<AtomicUsize>,
    /// Number of packages failed
    pub failed: Arc<AtomicUsize>,
    /// Total bytes to download
    pub total_bytes: u64,
    /// Bytes downloaded so far
    pub downloaded_bytes: Arc<AtomicU64>,
    /// Start time of installation
    pub start_time: Instant,
}

/// Installation plan with dependency ordering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallationPlan {
    /// Packages to install in dependency order
    pub packages: Vec<PlannedPackage>,
    /// Total estimated download size
    pub total_size: u64,
    /// Estimated installation time in seconds
    pub estimated_time: u64,
}

/// Planned package installation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannedPackage {
    /// Package name
    pub name: String,
    /// Package version
    pub version: String,
    /// Installation priority (lower = higher priority)
    pub priority: usize,
    /// Dependencies that must be installed first
    pub depends_on: Vec<String>,
    /// Estimated download size in bytes
    pub size: u64,
}

/// Parallel dependency installer
pub struct ParallelDependencyInstaller {
    /// Download options
    options: DownloadOptions,
    /// Package registry for downloading
    registry: Arc<dyn PackageRegistry>,
    /// Installation directory
    install_dir: PathBuf,
    /// Progress tracker
    progress: Arc<InstallationProgress>,
}

/// Installation result for a single package
#[derive(Debug)]
pub struct InstallationResult {
    /// Package name
    pub name: String,
    /// Package version
    pub version: String,
    /// Whether installation succeeded
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Time taken to install in milliseconds
    pub duration_ms: u64,
    /// Bytes downloaded
    pub bytes_downloaded: u64,
}

/// Installation statistics
#[derive(Debug, Clone)]
pub struct InstallationStatistics {
    /// Total packages installed
    pub total_installed: usize,
    /// Total packages failed
    pub total_failed: usize,
    /// Total time taken in seconds
    pub total_time_secs: f64,
    /// Total bytes downloaded
    pub total_bytes: u64,
    /// Average download speed in bytes/sec
    pub avg_download_speed: f64,
    /// Packages installed per second
    pub packages_per_sec: f64,
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            max_parallel: 8,
            timeout_secs: 300,
            max_retries: 3,
            retry_delay_ms: 1000,
            buffer_size: 8192,
            verify_integrity: true,
            resume_partial: true,
        }
    }
}

impl DownloadOptions {
    /// Create new download options with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum parallel downloads
    pub fn with_max_parallel(mut self, max: usize) -> Self {
        self.max_parallel = max;
        self
    }

    /// Set connection timeout
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Set maximum retries
    pub fn with_max_retries(mut self, retries: usize) -> Self {
        self.max_retries = retries;
        self
    }

    /// Enable/disable integrity verification
    pub fn with_verify_integrity(mut self, verify: bool) -> Self {
        self.verify_integrity = verify;
        self
    }
}

impl InstallationProgress {
    /// Create new progress tracker
    pub fn new(total_packages: usize, total_bytes: u64) -> Self {
        Self {
            total_packages,
            downloaded: Arc::new(AtomicUsize::new(0)),
            installed: Arc::new(AtomicUsize::new(0)),
            failed: Arc::new(AtomicUsize::new(0)),
            total_bytes,
            downloaded_bytes: Arc::new(AtomicU64::new(0)),
            start_time: Instant::now(),
        }
    }

    /// Get current progress as percentage (0-100)
    pub fn percentage(&self) -> f64 {
        if self.total_packages == 0 {
            return 100.0;
        }
        (self.installed.load(Ordering::Relaxed) as f64 / self.total_packages as f64) * 100.0
    }

    /// Get download progress as percentage (0-100)
    pub fn download_percentage(&self) -> f64 {
        if self.total_bytes == 0 {
            return 100.0;
        }
        (self.downloaded_bytes.load(Ordering::Relaxed) as f64 / self.total_bytes as f64) * 100.0
    }

    /// Get elapsed time in seconds
    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Get estimated remaining time in seconds
    pub fn estimated_remaining_secs(&self) -> f64 {
        let installed = self.installed.load(Ordering::Relaxed);
        if installed == 0 {
            return 0.0;
        }

        let elapsed = self.elapsed_secs();
        let rate = installed as f64 / elapsed;
        let remaining = self.total_packages - installed;

        remaining as f64 / rate
    }

    /// Get download speed in bytes/sec
    pub fn download_speed(&self) -> f64 {
        let elapsed = self.elapsed_secs();
        if elapsed == 0.0 {
            return 0.0;
        }

        self.downloaded_bytes.load(Ordering::Relaxed) as f64 / elapsed
    }

    /// Mark a package as downloaded
    pub fn mark_downloaded(&self, bytes: u64) {
        self.downloaded.fetch_add(1, Ordering::Relaxed);
        self.downloaded_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Mark a package as installed
    pub fn mark_installed(&self) {
        self.installed.fetch_add(1, Ordering::Relaxed);
    }

    /// Mark a package as failed
    pub fn mark_failed(&self) {
        self.failed.fetch_add(1, Ordering::Relaxed);
    }
}

impl InstallationPlan {
    /// Create a new installation plan
    pub fn new() -> Self {
        Self {
            packages: Vec::new(),
            total_size: 0,
            estimated_time: 0,
        }
    }

    /// Add a package to the plan
    pub fn add_package(&mut self, package: PlannedPackage) {
        self.total_size += package.size;
        self.packages.push(package);
    }

    /// Sort packages by dependency order
    pub fn sort_by_dependencies(&mut self) -> Result<()> {
        // Topological sort
        let mut sorted = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut visiting = std::collections::HashSet::new();

        for pkg in &self.packages {
            if !visited.contains(&pkg.name) {
                self.visit_package(&pkg.name, &mut visited, &mut visiting, &mut sorted)?;
            }
        }

        self.packages = sorted;
        Ok(())
    }

    /// Visit package for topological sort (DFS)
    fn visit_package(
        &self,
        name: &str,
        visited: &mut std::collections::HashSet<String>,
        visiting: &mut std::collections::HashSet<String>,
        sorted: &mut Vec<PlannedPackage>,
    ) -> Result<()> {
        if visiting.contains(name) {
            return Err(TorshError::InvalidArgument(format!(
                "Circular dependency detected: {}",
                name
            )));
        }

        if visited.contains(name) {
            return Ok(());
        }

        visiting.insert(name.to_string());

        // Find the package
        let package = self
            .packages
            .iter()
            .find(|p| p.name == name)
            .ok_or_else(|| {
                TorshError::InvalidArgument(format!("Package not found in plan: {}", name))
            })?;

        // Visit dependencies first
        for dep in &package.depends_on {
            self.visit_package(dep, visited, visiting, sorted)?;
        }

        visiting.remove(name);
        visited.insert(name.to_string());
        sorted.push(package.clone());

        Ok(())
    }

    /// Get packages by priority level
    pub fn get_by_priority(&self, priority: usize) -> Vec<&PlannedPackage> {
        self.packages
            .iter()
            .filter(|p| p.priority == priority)
            .collect()
    }

    /// Estimate total installation time based on size and bandwidth
    pub fn estimate_time(&mut self, bandwidth_bytes_per_sec: u64) {
        if bandwidth_bytes_per_sec > 0 {
            self.estimated_time = self.total_size / bandwidth_bytes_per_sec;
        }
    }
}

impl Default for InstallationPlan {
    fn default() -> Self {
        Self::new()
    }
}

impl ParallelDependencyInstaller {
    /// Create a new parallel installer
    pub fn new(
        registry: Arc<dyn PackageRegistry>,
        install_dir: PathBuf,
        options: DownloadOptions,
    ) -> Self {
        Self {
            options,
            registry,
            install_dir,
            progress: Arc::new(InstallationProgress::new(0, 0)),
        }
    }

    /// Create installation plan from resolved dependencies
    pub fn create_plan(&self, dependencies: &[ResolvedDependency]) -> Result<InstallationPlan> {
        let mut plan = InstallationPlan::new();

        for (priority, dep) in dependencies.iter().enumerate() {
            let package_info = self
                .registry
                .get_package_info(&dep.spec.name, &dep.resolved_version)?;

            let planned = PlannedPackage {
                name: dep.spec.name.clone(),
                version: dep.resolved_version.clone(),
                priority,
                depends_on: dep
                    .dependencies
                    .iter()
                    .map(|d| d.spec.name.clone())
                    .collect(),
                size: package_info.size,
            };

            plan.add_package(planned);
        }

        // Sort by dependencies
        plan.sort_by_dependencies()?;

        // Estimate installation time (assuming 10 MB/s bandwidth)
        plan.estimate_time(10 * 1024 * 1024);

        Ok(plan)
    }

    /// Create installation plan from locked dependencies
    pub fn create_plan_from_lockfile(
        &self,
        dependencies: &[LockedDependency],
    ) -> Result<InstallationPlan> {
        let mut plan = InstallationPlan::new();

        for (priority, dep) in dependencies.iter().enumerate() {
            let planned = PlannedPackage {
                name: dep.name.clone(),
                version: dep.version.clone(),
                priority,
                depends_on: dep.dependencies.clone(),
                size: 1024 * 1024, // Default 1MB, would be fetched from registry
            };

            plan.add_package(planned);
        }

        plan.sort_by_dependencies()?;
        plan.estimate_time(10 * 1024 * 1024);

        Ok(plan)
    }

    /// Install dependencies according to plan
    pub fn install(&mut self, plan: &InstallationPlan) -> Result<InstallationStatistics> {
        let total_packages = plan.packages.len();
        let total_bytes = plan.total_size;

        self.progress = Arc::new(InstallationProgress::new(total_packages, total_bytes));

        // Group packages by priority level for parallel installation
        let max_priority = plan.packages.iter().map(|p| p.priority).max().unwrap_or(0);

        let mut results = Vec::new();

        // Install packages level by level
        for level in 0..=max_priority {
            let level_packages: Vec<_> = plan.get_by_priority(level);

            if level_packages.is_empty() {
                continue;
            }

            // Install this level in parallel
            let level_results = self.install_parallel(&level_packages)?;
            results.extend(level_results);
        }

        // Compute statistics
        let stats = self.compute_statistics(&results);

        Ok(stats)
    }

    /// Install packages in parallel
    fn install_parallel(&self, packages: &[&PlannedPackage]) -> Result<Vec<InstallationResult>> {
        let _chunk_size = (packages.len() / self.options.max_parallel).max(1);

        // Use scirs2-core parallel operations
        let results: Vec<_> = packages
            .into_par_iter()
            .map(|pkg| self.install_package(pkg))
            .collect();

        // Check for errors
        for result in &results {
            if !result.success {
                if let Some(error) = &result.error {
                    // Log error but continue with other packages
                    eprintln!("Failed to install {}: {}", result.name, error);
                }
            }
        }

        Ok(results)
    }

    /// Install a single package
    fn install_package(&self, package: &PlannedPackage) -> InstallationResult {
        let start_time = Instant::now();
        let mut bytes_downloaded = 0u64;

        let result = self.install_package_with_retry(package, &mut bytes_downloaded);

        let duration_ms = start_time.elapsed().as_millis() as u64;

        match result {
            Ok(()) => {
                self.progress.mark_installed();
                InstallationResult {
                    name: package.name.clone(),
                    version: package.version.clone(),
                    success: true,
                    error: None,
                    duration_ms,
                    bytes_downloaded,
                }
            }
            Err(e) => {
                self.progress.mark_failed();
                InstallationResult {
                    name: package.name.clone(),
                    version: package.version.clone(),
                    success: false,
                    error: Some(e.to_string()),
                    duration_ms,
                    bytes_downloaded,
                }
            }
        }
    }

    /// Install package with retry logic
    fn install_package_with_retry(
        &self,
        package: &PlannedPackage,
        bytes_downloaded: &mut u64,
    ) -> Result<()> {
        let mut last_error = None;

        for attempt in 0..=self.options.max_retries {
            if attempt > 0 {
                // Wait before retry
                std::thread::sleep(Duration::from_millis(self.options.retry_delay_ms));
            }

            match self.download_and_install(package, bytes_downloaded) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.options.max_retries {
                        eprintln!(
                            "Download failed for {} (attempt {}/{}), retrying...",
                            package.name,
                            attempt + 1,
                            self.options.max_retries
                        );
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            TorshError::InvalidArgument("Installation failed with unknown error".to_string())
        }))
    }

    /// Download and install a package
    fn download_and_install(
        &self,
        package: &PlannedPackage,
        bytes_downloaded: &mut u64,
    ) -> Result<()> {
        let package_path = self
            .install_dir
            .join(format!("{}-{}.torshpkg", package.name, package.version));

        // Download the package
        self.registry
            .download_package(&package.name, &package.version, &package_path)?;

        // Update progress
        *bytes_downloaded = package.size;
        self.progress.mark_downloaded(package.size);

        // Verify integrity if enabled
        if self.options.verify_integrity {
            self.verify_package_integrity(&package_path)?;
        }

        Ok(())
    }

    /// Verify package integrity
    fn verify_package_integrity(&self, _package_path: &Path) -> Result<()> {
        // Simplified verification - in production, check against lockfile hash
        Ok(())
    }

    /// Compute installation statistics
    fn compute_statistics(&self, results: &[InstallationResult]) -> InstallationStatistics {
        let total_installed = results.iter().filter(|r| r.success).count();
        let total_failed = results.iter().filter(|r| !r.success).count();
        let total_bytes: u64 = results.iter().map(|r| r.bytes_downloaded).sum();
        let total_time_secs = self.progress.elapsed_secs();

        let avg_download_speed = if total_time_secs > 0.0 {
            total_bytes as f64 / total_time_secs
        } else {
            0.0
        };

        let packages_per_sec = if total_time_secs > 0.0 {
            total_installed as f64 / total_time_secs
        } else {
            0.0
        };

        InstallationStatistics {
            total_installed,
            total_failed,
            total_time_secs,
            total_bytes,
            avg_download_speed,
            packages_per_sec,
        }
    }

    /// Get current progress
    pub fn get_progress(&self) -> &InstallationProgress {
        &self.progress
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_options() {
        let options = DownloadOptions::new()
            .with_max_parallel(16)
            .with_timeout(600)
            .with_max_retries(5);

        assert_eq!(options.max_parallel, 16);
        assert_eq!(options.timeout_secs, 600);
        assert_eq!(options.max_retries, 5);
    }

    #[test]
    fn test_installation_progress() {
        let progress = InstallationProgress::new(10, 1000);

        progress.mark_downloaded(100);
        assert_eq!(progress.downloaded.load(Ordering::Relaxed), 1);
        assert_eq!(progress.downloaded_bytes.load(Ordering::Relaxed), 100);

        progress.mark_installed();
        assert_eq!(progress.installed.load(Ordering::Relaxed), 1);

        assert_eq!(progress.percentage(), 10.0);
        assert_eq!(progress.download_percentage(), 10.0);
    }

    #[test]
    fn test_installation_plan() {
        let mut plan = InstallationPlan::new();

        let pkg1 = PlannedPackage {
            name: "pkg1".to_string(),
            version: "1.0.0".to_string(),
            priority: 0,
            depends_on: vec![],
            size: 1000,
        };

        let pkg2 = PlannedPackage {
            name: "pkg2".to_string(),
            version: "1.0.0".to_string(),
            priority: 1,
            depends_on: vec!["pkg1".to_string()],
            size: 2000,
        };

        plan.add_package(pkg1);
        plan.add_package(pkg2);

        assert_eq!(plan.total_size, 3000);
        assert_eq!(plan.packages.len(), 2);
    }

    #[test]
    fn test_topological_sort() {
        let mut plan = InstallationPlan::new();

        // Create dependency chain: pkg3 -> pkg2 -> pkg1
        plan.add_package(PlannedPackage {
            name: "pkg3".to_string(),
            version: "1.0.0".to_string(),
            priority: 0,
            depends_on: vec!["pkg2".to_string()],
            size: 1000,
        });

        plan.add_package(PlannedPackage {
            name: "pkg2".to_string(),
            version: "1.0.0".to_string(),
            priority: 1,
            depends_on: vec!["pkg1".to_string()],
            size: 1000,
        });

        plan.add_package(PlannedPackage {
            name: "pkg1".to_string(),
            version: "1.0.0".to_string(),
            priority: 2,
            depends_on: vec![],
            size: 1000,
        });

        plan.sort_by_dependencies().unwrap();

        // pkg1 should be first, then pkg2, then pkg3
        assert_eq!(plan.packages[0].name, "pkg1");
        assert_eq!(plan.packages[1].name, "pkg2");
        assert_eq!(plan.packages[2].name, "pkg3");
    }
}
