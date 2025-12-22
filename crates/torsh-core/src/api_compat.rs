//! API Compatibility and Deprecation Management
//!
//! This module provides infrastructure for managing API evolution,
//! deprecation warnings, and compatibility tracking across ToRSh versions.
//!
//! # Features
//!
//! - **Deprecation Warnings**: Track and emit warnings for deprecated APIs
//! - **Version Compatibility**: Check compatibility between ToRSh versions
//! - **Migration Guides**: Provide automated migration suggestions
//! - **Breaking Change Detection**: Identify breaking changes in API usage
//!
//! # Examples
//!
//! ```rust
//! use torsh_core::api_compat::{deprecation_warning, deprecation_warning_inline, Version};
//!
//! // Simple deprecation warning (requires prior registration)
//! deprecation_warning("old_function");
//!
//! // Emit a deprecation warning with inline info
//! deprecation_warning_inline(
//!     "another_old_function",
//!     Version::new(0, 1, 0),
//!     Version::new(0, 2, 0),
//!     Some("new_function")
//! );
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

/// Semantic version representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl Version {
    /// Create a new version
    pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse version from string (e.g., "0.1.0")
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return None;
        }

        let major = parts[0].parse().ok()?;
        let minor = parts[1].parse().ok()?;
        let patch = parts[2].parse().ok()?;

        Some(Self::new(major, minor, patch))
    }

    /// Check if this version is compatible with another version
    /// using semantic versioning rules
    pub fn is_compatible_with(&self, other: &Version) -> bool {
        // Major version must match for compatibility
        if self.major != other.major {
            return false;
        }

        // If major is 0, minor version must also match
        if self.major == 0 && self.minor != other.minor {
            return false;
        }

        true
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Current ToRSh version
pub const TORSH_VERSION: Version = Version::new(0, 1, 0);

/// Deprecation severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DeprecationSeverity {
    /// Soft deprecation - will be removed in future major version
    Soft,
    /// Hard deprecation - will be removed in next minor version
    Hard,
    /// Critical - will be removed in next patch version
    Critical,
}

/// Information about a deprecated API
#[derive(Debug, Clone)]
pub struct DeprecationInfo {
    /// Name of the deprecated API
    pub api_name: String,
    /// Version when API was deprecated
    pub deprecated_in: Version,
    /// Version when API will be removed
    pub removed_in: Version,
    /// Suggested replacement
    pub replacement: Option<String>,
    /// Deprecation reason
    pub reason: Option<String>,
    /// Migration guide URL or text
    pub migration_guide: Option<String>,
    /// Severity of deprecation
    pub severity: DeprecationSeverity,
}

impl DeprecationInfo {
    /// Create a new deprecation info
    pub fn new(api_name: impl Into<String>, deprecated_in: Version, removed_in: Version) -> Self {
        Self {
            api_name: api_name.into(),
            deprecated_in,
            removed_in,
            replacement: None,
            reason: None,
            migration_guide: None,
            severity: DeprecationSeverity::Soft,
        }
    }

    /// Set the replacement API
    pub fn with_replacement(mut self, replacement: impl Into<String>) -> Self {
        self.replacement = Some(replacement.into());
        self
    }

    /// Set the deprecation reason
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Set the migration guide
    pub fn with_migration_guide(mut self, guide: impl Into<String>) -> Self {
        self.migration_guide = Some(guide.into());
        self
    }

    /// Set the severity
    pub fn with_severity(mut self, severity: DeprecationSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Check if this API has been removed in the current version
    pub fn is_removed(&self) -> bool {
        TORSH_VERSION >= self.removed_in
    }

    /// Check if this API should show a warning
    pub fn should_warn(&self) -> bool {
        TORSH_VERSION >= self.deprecated_in && TORSH_VERSION < self.removed_in
    }

    /// Format a deprecation warning message
    pub fn format_warning(&self) -> String {
        let mut msg = format!(
            "API '{}' is deprecated since version {} and will be removed in version {}",
            self.api_name, self.deprecated_in, self.removed_in
        );

        if let Some(ref replacement) = self.replacement {
            msg.push_str(&format!(". Use '{}' instead", replacement));
        }

        if let Some(ref reason) = self.reason {
            msg.push_str(&format!(". Reason: {}", reason));
        }

        if let Some(ref guide) = self.migration_guide {
            msg.push_str(&format!(". Migration guide: {}", guide));
        }

        msg
    }
}

/// Global deprecation tracker
static DEPRECATION_TRACKER: OnceLock<Arc<Mutex<DeprecationTracker>>> = OnceLock::new();

/// Tracks deprecation warnings and API usage
struct DeprecationTracker {
    /// Registered deprecations
    deprecations: HashMap<String, DeprecationInfo>,
    /// Count of warnings emitted per API
    warning_counts: HashMap<String, usize>,
    /// Maximum warnings per API before suppressing
    max_warnings_per_api: usize,
    /// Whether to emit warnings to stderr
    emit_warnings: bool,
}

impl Default for DeprecationTracker {
    fn default() -> Self {
        Self {
            deprecations: HashMap::new(),
            warning_counts: HashMap::new(),
            max_warnings_per_api: 10,   // Limit repeated warnings
            emit_warnings: !cfg!(test), // Suppress in tests by default
        }
    }
}

impl DeprecationTracker {
    /// Register a deprecated API
    fn register(&mut self, info: DeprecationInfo) {
        self.deprecations.insert(info.api_name.clone(), info);
    }

    /// Emit a deprecation warning
    fn emit_warning(&mut self, api_name: &str) -> bool {
        // Check if API is registered
        let info = match self.deprecations.get(api_name) {
            Some(info) => info,
            None => return false,
        };

        // Check if we should warn
        if !info.should_warn() {
            return false;
        }

        // Check if we've exceeded max warnings
        let count = self.warning_counts.entry(api_name.to_string()).or_insert(0);
        if *count >= self.max_warnings_per_api {
            return false;
        }
        *count += 1;

        // Emit warning if enabled
        if self.emit_warnings {
            eprintln!("⚠️  DEPRECATION WARNING: {}", info.format_warning());

            if *count == self.max_warnings_per_api {
                eprintln!(
                    "⚠️  (Further warnings for '{}' will be suppressed)",
                    api_name
                );
            }
        }

        true
    }

    /// Get deprecation info for an API
    fn get_info(&self, api_name: &str) -> Option<&DeprecationInfo> {
        self.deprecations.get(api_name)
    }

    /// Get all registered deprecations
    fn get_all_deprecations(&self) -> Vec<DeprecationInfo> {
        self.deprecations.values().cloned().collect()
    }

    /// Get warning statistics
    fn get_warning_stats(&self) -> HashMap<String, usize> {
        self.warning_counts.clone()
    }

    /// Clear warning counts
    fn clear_warning_counts(&mut self) {
        self.warning_counts.clear();
    }

    /// Set whether to emit warnings
    fn set_emit_warnings(&mut self, emit: bool) {
        self.emit_warnings = emit;
    }

    /// Set maximum warnings per API
    fn set_max_warnings_per_api(&mut self, max: usize) {
        self.max_warnings_per_api = max;
    }
}

/// Get the global deprecation tracker
fn get_tracker() -> Arc<Mutex<DeprecationTracker>> {
    DEPRECATION_TRACKER
        .get_or_init(|| Arc::new(Mutex::new(DeprecationTracker::default())))
        .clone()
}

/// Register a deprecated API
///
/// # Examples
///
/// ```rust
/// use torsh_core::api_compat::{register_deprecation, DeprecationInfo, Version};
///
/// let info = DeprecationInfo::new("old_function", Version::new(0, 1, 0), Version::new(0, 2, 0))
///     .with_replacement("new_function")
///     .with_reason("Improved performance and API consistency");
///
/// register_deprecation(info);
/// ```
pub fn register_deprecation(info: DeprecationInfo) {
    get_tracker().lock().unwrap().register(info);
}

/// Emit a deprecation warning for an API
///
/// # Examples
///
/// ```rust
/// use torsh_core::api_compat::deprecation_warning;
///
/// deprecation_warning("old_function");
/// ```
pub fn deprecation_warning(api_name: &str) -> bool {
    get_tracker().lock().unwrap().emit_warning(api_name)
}

/// Convenience function to emit a deprecation warning with inline info
///
/// # Examples
///
/// ```rust
/// use torsh_core::api_compat::{deprecation_warning_inline, Version};
///
/// deprecation_warning_inline(
///     "old_function",
///     Version::new(0, 1, 0),
///     Version::new(0, 2, 0),
///     Some("new_function")
/// );
/// ```
pub fn deprecation_warning_inline(
    api_name: &str,
    deprecated_in: Version,
    removed_in: Version,
    replacement: Option<&str>,
) {
    let mut info = DeprecationInfo::new(api_name, deprecated_in, removed_in);
    if let Some(repl) = replacement {
        info = info.with_replacement(repl);
    }
    register_deprecation(info);
    deprecation_warning(api_name);
}

/// Get deprecation info for an API
pub fn get_deprecation_info(api_name: &str) -> Option<DeprecationInfo> {
    get_tracker().lock().unwrap().get_info(api_name).cloned()
}

/// Get all registered deprecations
pub fn get_all_deprecations() -> Vec<DeprecationInfo> {
    get_tracker().lock().unwrap().get_all_deprecations()
}

/// Get deprecation warning statistics
pub fn get_deprecation_stats() -> HashMap<String, usize> {
    get_tracker().lock().unwrap().get_warning_stats()
}

/// Clear deprecation warning counts
pub fn clear_deprecation_counts() {
    get_tracker().lock().unwrap().clear_warning_counts();
}

/// Configure deprecation warning behavior
pub fn configure_deprecation_warnings(emit: bool, max_per_api: usize) {
    let binding = get_tracker();
    let mut tracker = binding.lock().unwrap();
    tracker.set_emit_warnings(emit);
    tracker.set_max_warnings_per_api(max_per_api);
}

/// Reset the entire deprecation tracker (primarily for testing)
/// This clears all deprecations, warning counts, and resets configuration to defaults
#[cfg(test)]
pub fn reset_deprecation_tracker() {
    let binding = get_tracker();
    let mut tracker = binding.lock().unwrap();
    tracker.deprecations.clear();
    tracker.warning_counts.clear();
    tracker.max_warnings_per_api = 10;
    tracker.emit_warnings = false; // Safe default for tests
}

/// Generate a deprecation report
pub struct DeprecationReport {
    /// Deprecations that are currently active (should warn)
    pub active: Vec<DeprecationInfo>,
    /// Deprecations that have been removed
    pub removed: Vec<DeprecationInfo>,
    /// Deprecations pending (not yet deprecated)
    pub pending: Vec<DeprecationInfo>,
    /// Warning statistics
    pub warning_stats: HashMap<String, usize>,
}

impl DeprecationReport {
    /// Generate a deprecation report
    pub fn generate() -> Self {
        let binding = get_tracker();
        let tracker = binding.lock().unwrap();
        let all_deprecations = tracker.get_all_deprecations();

        let mut active = Vec::new();
        let mut removed = Vec::new();
        let mut pending = Vec::new();

        for info in all_deprecations {
            if info.is_removed() {
                removed.push(info);
            } else if info.should_warn() {
                active.push(info);
            } else {
                pending.push(info);
            }
        }

        Self {
            active,
            removed,
            pending,
            warning_stats: tracker.get_warning_stats(),
        }
    }

    /// Format the report as a string
    pub fn format(&self) -> String {
        let mut report = String::from("ToRSh API Deprecation Report\n");
        report.push_str("==============================\n\n");

        report.push_str(&format!("Current Version: {}\n\n", TORSH_VERSION));

        if !self.active.is_empty() {
            report.push_str(&format!("Active Deprecations ({})\n", self.active.len()));
            report.push_str("-------------------------\n");
            for info in &self.active {
                report.push_str(&format!(
                    "  • {} (deprecated in {}, removed in {})\n",
                    info.api_name, info.deprecated_in, info.removed_in
                ));
                if let Some(ref repl) = info.replacement {
                    report.push_str(&format!("    Replacement: {}\n", repl));
                }
                if let Some(count) = self.warning_stats.get(&info.api_name) {
                    report.push_str(&format!("    Warnings emitted: {}\n", count));
                }
            }
            report.push('\n');
        }

        if !self.removed.is_empty() {
            report.push_str(&format!("Removed APIs ({})\n", self.removed.len()));
            report.push_str("-----------------\n");
            for info in &self.removed {
                report.push_str(&format!(
                    "  • {} (removed in {})\n",
                    info.api_name, info.removed_in
                ));
            }
            report.push('\n');
        }

        if !self.pending.is_empty() {
            report.push_str(&format!("Pending Deprecations ({})\n", self.pending.len()));
            report.push_str("-------------------------\n");
            for info in &self.pending {
                report.push_str(&format!(
                    "  • {} (will be deprecated in {})\n",
                    info.api_name, info.deprecated_in
                ));
            }
            report.push('\n');
        }

        report
    }
}

/// Macro to mark a function as deprecated
#[macro_export]
macro_rules! deprecated {
    ($name:expr, $deprecated_in:expr, $removed_in:expr) => {
        $crate::api_compat::deprecation_warning_inline($name, $deprecated_in, $removed_in, None);
    };
    ($name:expr, $deprecated_in:expr, $removed_in:expr, $replacement:expr) => {
        $crate::api_compat::deprecation_warning_inline(
            $name,
            $deprecated_in,
            $removed_in,
            Some($replacement),
        );
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parsing() {
        let v = Version::parse("1.2.3").unwrap();
        assert_eq!(v, Version::new(1, 2, 3));

        assert!(Version::parse("invalid").is_none());
        assert!(Version::parse("1.2").is_none());
    }

    #[test]
    fn test_version_compatibility() {
        let v1 = Version::new(1, 2, 3);
        let v2 = Version::new(1, 3, 0);
        let v3 = Version::new(2, 0, 0);

        assert!(v1.is_compatible_with(&v2));
        assert!(!v1.is_compatible_with(&v3));

        // 0.x.y versions require minor match
        let v4 = Version::new(0, 1, 0);
        let v5 = Version::new(0, 1, 5);
        let v6 = Version::new(0, 2, 0);

        assert!(v4.is_compatible_with(&v5));
        assert!(!v4.is_compatible_with(&v6));
    }

    #[test]
    fn test_deprecation_info() {
        let info = DeprecationInfo::new("old_func", Version::new(0, 1, 0), Version::new(0, 2, 0))
            .with_replacement("new_func")
            .with_reason("Better performance");

        assert_eq!(info.api_name, "old_func");
        assert_eq!(info.replacement, Some("new_func".to_string()));
        assert_eq!(info.reason, Some("Better performance".to_string()));
    }

    #[test]
    fn test_deprecation_registration() {
        clear_deprecation_counts();

        let info = DeprecationInfo::new("test_api", Version::new(0, 0, 1), Version::new(1, 0, 0))
            .with_replacement("new_test_api");

        register_deprecation(info);

        let retrieved = get_deprecation_info("test_api").unwrap();
        assert_eq!(retrieved.api_name, "test_api");
        assert_eq!(retrieved.replacement, Some("new_test_api".to_string()));
    }

    #[test]
    fn test_deprecation_warning() {
        // Enable warnings and clear counts for this test
        configure_deprecation_warnings(true, 10);
        clear_deprecation_counts();

        let info = DeprecationInfo::new(
            "test_warning_api",
            Version::new(0, 0, 1),
            Version::new(1, 0, 0),
        );

        register_deprecation(info);
        deprecation_warning("test_warning_api");

        let stats = get_deprecation_stats();
        // The warning should have been recorded at least once
        let count = stats.get("test_warning_api").copied().unwrap_or(0);
        assert!(count >= 1, "Expected at least 1 warning, got {}", count);
    }

    #[test]
    fn test_deprecation_report() {
        clear_deprecation_counts();

        // Register some test deprecations
        register_deprecation(DeprecationInfo::new(
            "active_api",
            Version::new(0, 0, 1),
            Version::new(1, 0, 0),
        ));

        let report = DeprecationReport::generate();
        let formatted = report.format();

        assert!(formatted.contains("ToRSh API Deprecation Report"));
    }

    #[test]
    fn test_max_warnings_limit() {
        // Use a unique API name to avoid test isolation issues
        let unique_api = "limited_api_max_warnings_test_unique_v2";

        // Reset tracker completely first to avoid interference from other tests
        reset_deprecation_tracker();

        // Now configure with our desired settings
        configure_deprecation_warnings(false, 3);

        let info = DeprecationInfo::new(unique_api, Version::new(0, 0, 1), Version::new(1, 0, 0));

        register_deprecation(info);

        // Track actual warnings emitted (return value indicates if warning was actually counted)
        let mut actual_emitted = 0;
        for _ in 0..5 {
            if deprecation_warning(unique_api) {
                actual_emitted += 1;
            }
        }

        // Verify return values indicate warnings were limited
        assert_eq!(
            actual_emitted, 3,
            "Should have returned true for exactly 3 warnings"
        );

        let stats = get_deprecation_stats();
        assert_eq!(stats.get(unique_api), Some(&3)); // Should be limited to 3
    }
}
