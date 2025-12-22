//! Sandboxing system for safe package execution
//!
//! This module provides a secure execution environment for untrusted packages
//! with resource limits, file system restrictions, and network isolation.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use torsh_core::error::{Result, TorshError};

/// Sandbox configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Resource limits
    pub limits: ResourceLimits,
    /// File system access policy
    pub filesystem: FilesystemPolicy,
    /// Network access policy
    pub network: NetworkPolicy,
    /// Capability restrictions
    pub capabilities: CapabilitySet,
    /// Environment variables allowed
    pub allowed_env_vars: HashSet<String>,
    /// Maximum execution time
    pub max_execution_time: Duration,
}

/// Resource limits for sandboxed execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum CPU usage (percentage, 0-100)
    pub max_cpu_percent: u8,
    /// Maximum memory in bytes
    pub max_memory_bytes: u64,
    /// Maximum disk space in bytes
    pub max_disk_bytes: u64,
    /// Maximum number of open files
    pub max_open_files: u32,
    /// Maximum number of processes
    pub max_processes: u32,
    /// Maximum number of threads
    pub max_threads: u32,
}

/// File system access policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemPolicy {
    /// Read-only paths
    pub readonly_paths: Vec<PathBuf>,
    /// Read-write paths
    pub readwrite_paths: Vec<PathBuf>,
    /// Forbidden paths (blacklist)
    pub forbidden_paths: Vec<PathBuf>,
    /// Temporary directory for sandbox
    pub temp_dir: Option<PathBuf>,
    /// Whether to use virtual filesystem overlay
    pub use_overlay: bool,
}

/// Network access policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicy {
    /// Whether network access is allowed at all
    pub allowed: bool,
    /// Allowed hostnames (whitelist)
    pub allowed_hosts: Vec<String>,
    /// Allowed port ranges
    pub allowed_ports: Vec<PortRange>,
    /// Maximum bandwidth in bytes/sec
    pub max_bandwidth: u64,
    /// Whether to use network namespace isolation
    pub use_namespace: bool,
}

/// Port range for network policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortRange {
    /// Start port (inclusive)
    pub start: u16,
    /// End port (inclusive)
    pub end: u16,
}

/// Capability set for fine-grained permissions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilitySet {
    /// Can read from filesystem
    pub read_files: bool,
    /// Can write to filesystem
    pub write_files: bool,
    /// Can execute programs
    pub execute: bool,
    /// Can access network
    pub network: bool,
    /// Can create processes
    pub fork: bool,
    /// Can access system information
    pub system_info: bool,
    /// Can load dynamic libraries
    pub load_libraries: bool,
    /// Custom capabilities
    pub custom: HashMap<String, bool>,
}

/// Sandbox execution context
pub struct Sandbox {
    /// Sandbox configuration
    config: SandboxConfig,
    /// Sandbox ID for tracking
    id: String,
    /// Whether sandbox is active
    active: bool,
    /// Resource usage monitor
    monitor: ResourceMonitor,
}

/// Resource usage monitor
#[derive(Debug, Clone)]
pub struct ResourceMonitor {
    /// Current CPU usage percentage
    pub cpu_usage: f64,
    /// Current memory usage in bytes
    pub memory_usage: u64,
    /// Current disk usage in bytes
    pub disk_usage: u64,
    /// Number of open files
    pub open_files: u32,
    /// Number of active processes
    pub active_processes: u32,
}

/// Sandbox execution result
#[derive(Debug)]
pub struct SandboxResult<T> {
    /// Execution result
    pub result: Result<T>,
    /// Resource usage statistics
    pub resource_usage: ResourceUsageStats,
    /// Sandbox violations detected
    pub violations: Vec<SandboxViolation>,
}

/// Resource usage statistics after execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageStats {
    /// Peak CPU usage percentage
    pub peak_cpu: f64,
    /// Peak memory usage in bytes
    pub peak_memory: u64,
    /// Total disk reads in bytes
    pub disk_reads: u64,
    /// Total disk writes in bytes
    pub disk_writes: u64,
    /// Total execution time
    pub execution_time: Duration,
    /// Number of system calls made
    pub syscalls: u64,
}

/// Sandbox violation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxViolation {
    /// Type of violation
    pub violation_type: ViolationType,
    /// Description of the violation
    pub description: String,
    /// Severity level
    pub severity: ViolationSeverity,
    /// When the violation occurred
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Type of sandbox violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationType {
    /// Resource limit exceeded
    ResourceLimit,
    /// Forbidden file access
    FileAccess,
    /// Forbidden network access
    NetworkAccess,
    /// Missing capability
    CapabilityDenied,
    /// Execution timeout
    Timeout,
    /// Suspicious system call
    SuspiciousSystemCall,
}

/// Violation severity level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Low severity (logged but allowed)
    Low,
    /// Medium severity (warned)
    Medium,
    /// High severity (blocked)
    High,
    /// Critical severity (sandbox terminated)
    Critical,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            limits: ResourceLimits::default(),
            filesystem: FilesystemPolicy::default(),
            network: NetworkPolicy::default(),
            capabilities: CapabilitySet::minimal(),
            allowed_env_vars: HashSet::new(),
            max_execution_time: Duration::from_secs(300),
        }
    }
}

impl SandboxConfig {
    /// Create a restrictive sandbox configuration
    pub fn restrictive() -> Self {
        Self {
            limits: ResourceLimits::restrictive(),
            filesystem: FilesystemPolicy::readonly(),
            network: NetworkPolicy::deny_all(),
            capabilities: CapabilitySet::minimal(),
            allowed_env_vars: HashSet::new(),
            max_execution_time: Duration::from_secs(60),
        }
    }

    /// Create a permissive sandbox configuration (for trusted packages)
    pub fn permissive() -> Self {
        Self {
            limits: ResourceLimits::permissive(),
            filesystem: FilesystemPolicy::default(),
            network: NetworkPolicy::allow_all(),
            capabilities: CapabilitySet::full(),
            allowed_env_vars: std::env::vars().map(|(k, _)| k).collect(),
            max_execution_time: Duration::from_secs(3600),
        }
    }

    /// Validate sandbox configuration
    pub fn validate(&self) -> Result<()> {
        // Check resource limits
        if self.limits.max_cpu_percent > 100 {
            return Err(TorshError::InvalidArgument(
                "CPU limit cannot exceed 100%".to_string(),
            ));
        }

        if self.limits.max_memory_bytes == 0 {
            return Err(TorshError::InvalidArgument(
                "Memory limit cannot be zero".to_string(),
            ));
        }

        // Check filesystem policy
        for path in &self.filesystem.forbidden_paths {
            if !path.is_absolute() {
                return Err(TorshError::InvalidArgument(format!(
                    "Forbidden path must be absolute: {:?}",
                    path
                )));
            }
        }

        Ok(())
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_cpu_percent: 80,
            max_memory_bytes: 2 * 1024 * 1024 * 1024, // 2GB
            max_disk_bytes: 10 * 1024 * 1024 * 1024,  // 10GB
            max_open_files: 1024,
            max_processes: 100,
            max_threads: 100,
        }
    }
}

impl ResourceLimits {
    /// Create restrictive resource limits
    pub fn restrictive() -> Self {
        Self {
            max_cpu_percent: 50,
            max_memory_bytes: 512 * 1024 * 1024, // 512MB
            max_disk_bytes: 1024 * 1024 * 1024,  // 1GB
            max_open_files: 256,
            max_processes: 10,
            max_threads: 20,
        }
    }

    /// Create permissive resource limits
    pub fn permissive() -> Self {
        Self {
            max_cpu_percent: 100,
            max_memory_bytes: 16 * 1024 * 1024 * 1024, // 16GB
            max_disk_bytes: 100 * 1024 * 1024 * 1024,  // 100GB
            max_open_files: 4096,
            max_processes: 1000,
            max_threads: 1000,
        }
    }
}

impl Default for FilesystemPolicy {
    fn default() -> Self {
        Self {
            readonly_paths: vec![],
            readwrite_paths: vec![],
            forbidden_paths: vec![
                PathBuf::from("/etc/passwd"),
                PathBuf::from("/etc/shadow"),
                PathBuf::from("/root"),
            ],
            temp_dir: None,
            use_overlay: false,
        }
    }
}

impl FilesystemPolicy {
    /// Create readonly filesystem policy
    pub fn readonly() -> Self {
        Self {
            readonly_paths: vec![PathBuf::from("/")],
            readwrite_paths: vec![],
            forbidden_paths: vec![],
            temp_dir: Some(std::env::temp_dir()),
            use_overlay: true,
        }
    }

    /// Check if path is allowed for reading
    pub fn can_read(&self, path: &Path) -> bool {
        // Check if in forbidden paths
        for forbidden in &self.forbidden_paths {
            if path.starts_with(forbidden) {
                return false;
            }
        }

        // Check if in readonly or readwrite paths
        for allowed in self.readonly_paths.iter().chain(&self.readwrite_paths) {
            if path.starts_with(allowed) {
                return true;
            }
        }

        false
    }

    /// Check if path is allowed for writing
    pub fn can_write(&self, path: &Path) -> bool {
        // Check if in forbidden paths
        for forbidden in &self.forbidden_paths {
            if path.starts_with(forbidden) {
                return false;
            }
        }

        // Only readwrite paths are writable
        for allowed in &self.readwrite_paths {
            if path.starts_with(allowed) {
                return true;
            }
        }

        false
    }
}

impl Default for NetworkPolicy {
    fn default() -> Self {
        Self {
            allowed: false,
            allowed_hosts: vec![],
            allowed_ports: vec![],
            max_bandwidth: 10 * 1024 * 1024, // 10 MB/s
            use_namespace: false,
        }
    }
}

impl NetworkPolicy {
    /// Create deny-all network policy
    pub fn deny_all() -> Self {
        Self {
            allowed: false,
            ..Default::default()
        }
    }

    /// Create allow-all network policy
    pub fn allow_all() -> Self {
        Self {
            allowed: true,
            allowed_hosts: vec!["*".to_string()],
            allowed_ports: vec![PortRange {
                start: 1,
                end: 65535,
            }],
            max_bandwidth: u64::MAX,
            use_namespace: false,
        }
    }

    /// Check if host is allowed
    pub fn is_host_allowed(&self, host: &str) -> bool {
        if !self.allowed {
            return false;
        }

        if self.allowed_hosts.contains(&"*".to_string()) {
            return true;
        }

        self.allowed_hosts
            .iter()
            .any(|allowed| host == allowed || host.ends_with(&format!(".{}", allowed)))
    }

    /// Check if port is allowed
    pub fn is_port_allowed(&self, port: u16) -> bool {
        if !self.allowed {
            return false;
        }

        self.allowed_ports
            .iter()
            .any(|range| port >= range.start && port <= range.end)
    }
}

impl CapabilitySet {
    /// Create minimal capability set (very restrictive)
    pub fn minimal() -> Self {
        Self {
            read_files: true,
            write_files: false,
            execute: false,
            network: false,
            fork: false,
            system_info: false,
            load_libraries: false,
            custom: HashMap::new(),
        }
    }

    /// Create full capability set (permissive)
    pub fn full() -> Self {
        Self {
            read_files: true,
            write_files: true,
            execute: true,
            network: true,
            fork: true,
            system_info: true,
            load_libraries: true,
            custom: HashMap::new(),
        }
    }

    /// Check if a capability is granted
    pub fn has_capability(&self, capability: &str) -> bool {
        match capability {
            "read_files" => self.read_files,
            "write_files" => self.write_files,
            "execute" => self.execute,
            "network" => self.network,
            "fork" => self.fork,
            "system_info" => self.system_info,
            "load_libraries" => self.load_libraries,
            custom => self.custom.get(custom).copied().unwrap_or(false),
        }
    }
}

impl Sandbox {
    /// Create a new sandbox
    pub fn new(config: SandboxConfig) -> Result<Self> {
        config.validate()?;

        let id = uuid::Uuid::new_v4().to_string();

        Ok(Self {
            config,
            id,
            active: false,
            monitor: ResourceMonitor::new(),
        })
    }

    /// Activate the sandbox
    pub fn activate(&mut self) -> Result<()> {
        if self.active {
            return Err(TorshError::InvalidArgument(
                "Sandbox is already active".to_string(),
            ));
        }

        // Platform-specific sandbox activation
        #[cfg(target_os = "linux")]
        self.activate_linux()?;

        #[cfg(target_os = "macos")]
        self.activate_macos()?;

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        self.activate_generic()?;

        self.active = true;
        Ok(())
    }

    /// Deactivate the sandbox
    pub fn deactivate(&mut self) -> Result<()> {
        if !self.active {
            return Ok(());
        }

        // Platform-specific sandbox deactivation
        self.active = false;
        Ok(())
    }

    /// Execute a function in the sandbox
    pub fn execute<F, T>(&mut self, f: F) -> SandboxResult<T>
    where
        F: FnOnce() -> Result<T>,
    {
        let start_time = std::time::Instant::now();
        let mut violations = Vec::new();

        // Activate sandbox
        if let Err(e) = self.activate() {
            return SandboxResult {
                result: Err(e),
                resource_usage: ResourceUsageStats::default(),
                violations,
            };
        }

        // Execute function
        let result = f();

        // Deactivate sandbox
        let _ = self.deactivate();

        let execution_time = start_time.elapsed();

        // Collect resource usage
        let resource_usage = ResourceUsageStats {
            peak_cpu: self.monitor.cpu_usage,
            peak_memory: self.monitor.memory_usage,
            disk_reads: 0,  // Would be collected from OS
            disk_writes: 0, // Would be collected from OS
            execution_time,
            syscalls: 0, // Would be collected from tracing
        };

        // Check for violations
        if execution_time > self.config.max_execution_time {
            violations.push(SandboxViolation {
                violation_type: ViolationType::Timeout,
                description: format!(
                    "Execution time exceeded limit: {:?} > {:?}",
                    execution_time, self.config.max_execution_time
                ),
                severity: ViolationSeverity::High,
                timestamp: chrono::Utc::now(),
            });
        }

        SandboxResult {
            result,
            resource_usage,
            violations,
        }
    }

    /// Linux-specific sandbox activation
    #[cfg(target_os = "linux")]
    fn activate_linux(&mut self) -> Result<()> {
        // In production, would use:
        // - seccomp for system call filtering
        // - namespaces for isolation
        // - cgroups for resource limits
        // For now, this is a placeholder
        Ok(())
    }

    /// macOS-specific sandbox activation
    #[cfg(target_os = "macos")]
    fn activate_macos(&mut self) -> Result<()> {
        // In production, would use:
        // - sandbox-exec for sandboxing
        // - process sandboxing APIs
        // For now, this is a placeholder
        Ok(())
    }

    /// Generic sandbox activation (fallback)
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn activate_generic(&mut self) -> Result<()> {
        // Limited sandboxing capabilities on other platforms
        Ok(())
    }

    /// Get sandbox ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get sandbox configuration
    pub fn config(&self) -> &SandboxConfig {
        &self.config
    }

    /// Get resource monitor
    pub fn monitor(&self) -> &ResourceMonitor {
        &self.monitor
    }
}

impl ResourceMonitor {
    /// Create a new resource monitor
    pub fn new() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            disk_usage: 0,
            open_files: 0,
            active_processes: 0,
        }
    }

    /// Update resource usage
    pub fn update(&mut self) {
        // In production, would query actual system metrics
        // For now, placeholder values
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ResourceUsageStats {
    fn default() -> Self {
        Self {
            peak_cpu: 0.0,
            peak_memory: 0,
            disk_reads: 0,
            disk_writes: 0,
            execution_time: Duration::from_secs(0),
            syscalls: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sandbox_config() {
        let config = SandboxConfig::default();
        assert!(config.validate().is_ok());

        let restrictive = SandboxConfig::restrictive();
        assert!(restrictive.validate().is_ok());
        assert_eq!(restrictive.limits.max_cpu_percent, 50);

        let permissive = SandboxConfig::permissive();
        assert!(permissive.validate().is_ok());
        assert_eq!(permissive.limits.max_cpu_percent, 100);
    }

    #[test]
    fn test_resource_limits() {
        let limits = ResourceLimits::default();
        assert!(limits.max_cpu_percent <= 100);
        assert!(limits.max_memory_bytes > 0);

        let restrictive = ResourceLimits::restrictive();
        assert!(restrictive.max_memory_bytes < limits.max_memory_bytes);
    }

    #[test]
    fn test_filesystem_policy() {
        let policy = FilesystemPolicy::default();

        // Forbidden paths should not be readable
        assert!(!policy.can_read(Path::new("/etc/shadow")));
        assert!(!policy.can_write(Path::new("/etc/shadow")));

        let readonly = FilesystemPolicy::readonly();
        assert!(readonly.can_read(Path::new("/usr/bin/ls")));
        assert!(!readonly.can_write(Path::new("/usr/bin/ls")));
    }

    #[test]
    fn test_network_policy() {
        let deny_all = NetworkPolicy::deny_all();
        assert!(!deny_all.allowed);
        assert!(!deny_all.is_host_allowed("example.com"));
        assert!(!deny_all.is_port_allowed(80));

        let allow_all = NetworkPolicy::allow_all();
        assert!(allow_all.allowed);
        assert!(allow_all.is_host_allowed("example.com"));
        assert!(allow_all.is_port_allowed(80));
    }

    #[test]
    fn test_capability_set() {
        let minimal = CapabilitySet::minimal();
        assert!(minimal.read_files);
        assert!(!minimal.write_files);
        assert!(!minimal.execute);

        let full = CapabilitySet::full();
        assert!(full.read_files);
        assert!(full.write_files);
        assert!(full.execute);
    }

    #[test]
    fn test_sandbox_creation() {
        let config = SandboxConfig::default();
        let sandbox = Sandbox::new(config);
        assert!(sandbox.is_ok());
    }

    #[test]
    fn test_sandbox_execution() {
        let config = SandboxConfig::permissive();
        let mut sandbox = Sandbox::new(config).unwrap();

        let result = sandbox.execute(|| {
            // Simple test function
            Ok(42)
        });

        assert!(result.result.is_ok());
        assert_eq!(result.result.unwrap(), 42);
        assert!(result.violations.is_empty());
    }
}
