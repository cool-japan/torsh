//! Package version handling and compatibility

use serde::{Deserialize, Serialize};
use std::fmt;

/// Package version with semantic versioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageVersion {
    /// Major version number (breaking changes)
    pub major: u32,
    /// Minor version number (backward-compatible additions)
    pub minor: u32,
    /// Patch version number (backward-compatible bug fixes)
    pub patch: u32,
    /// Pre-release version identifier (e.g., "alpha.1", "beta.2")
    pub pre_release: Option<String>,
    /// Build metadata (ignored in version precedence)
    pub build_metadata: Option<String>,
}

impl PackageVersion {
    /// Create a new version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            pre_release: None,
            build_metadata: None,
        }
    }

    /// Parse version from string
    pub fn parse(version: &str) -> Result<Self, String> {
        let semver = semver::Version::parse(version)
            .map_err(|e| format!("Invalid version format: {}", e))?;

        Ok(Self {
            major: semver.major as u32,
            minor: semver.minor as u32,
            patch: semver.patch as u32,
            pre_release: if semver.pre.is_empty() {
                None
            } else {
                Some(semver.pre.to_string())
            },
            build_metadata: if semver.build.is_empty() {
                None
            } else {
                Some(semver.build.to_string())
            },
        })
    }

    /// Check if this version is compatible with a requirement
    pub fn is_compatible_with(&self, requirement: &VersionRequirement) -> bool {
        requirement.matches(self)
    }

    /// Get the next major version
    pub fn next_major(&self) -> Self {
        Self::new(self.major + 1, 0, 0)
    }

    /// Get the next minor version
    pub fn next_minor(&self) -> Self {
        Self::new(self.major, self.minor + 1, 0)
    }

    /// Get the next patch version
    pub fn next_patch(&self) -> Self {
        Self::new(self.major, self.minor, self.patch + 1)
    }

    /// Check if this is a pre-release version
    pub fn is_pre_release(&self) -> bool {
        self.pre_release.is_some()
    }
}

impl PartialEq for PackageVersion {
    fn eq(&self, other: &Self) -> bool {
        // According to semantic versioning, build metadata should be ignored in comparisons
        self.major == other.major
            && self.minor == other.minor
            && self.patch == other.patch
            && self.pre_release == other.pre_release
        // Note: build_metadata is intentionally ignored
    }
}

impl Eq for PackageVersion {}

impl std::cmp::PartialOrd for PackageVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl std::cmp::Ord for PackageVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        // First compare major.minor.patch
        match (self.major, self.minor, self.patch).cmp(&(other.major, other.minor, other.patch)) {
            Ordering::Equal => {
                // When major.minor.patch are equal, handle pre-release semantics
                match (&self.pre_release, &other.pre_release) {
                    (None, None) => Ordering::Equal,
                    (None, Some(_)) => Ordering::Greater, // Release > pre-release
                    (Some(_), None) => Ordering::Less,    // Pre-release < release
                    (Some(a), Some(b)) => a.cmp(b),       // Compare pre-release strings
                }
                // Note: build metadata is ignored in version precedence per semver
            }
            other_ordering => other_ordering,
        }
    }
}

impl fmt::Display for PackageVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;

        if let Some(pre) = &self.pre_release {
            write!(f, "-{}", pre)?;
        }

        if let Some(build) = &self.build_metadata {
            write!(f, "+{}", build)?;
        }

        Ok(())
    }
}

impl From<semver::Version> for PackageVersion {
    fn from(v: semver::Version) -> Self {
        Self {
            major: v.major as u32,
            minor: v.minor as u32,
            patch: v.patch as u32,
            pre_release: if v.pre.is_empty() {
                None
            } else {
                Some(v.pre.to_string())
            },
            build_metadata: if v.build.is_empty() {
                None
            } else {
                Some(v.build.to_string())
            },
        }
    }
}

/// Version requirement specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionRequirement {
    /// Comparison operator to use
    pub comparator: VersionComparator,
    /// Version to compare against
    pub version: PackageVersion,
}

/// Version comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VersionComparator {
    /// Exact match
    Exact,
    /// Greater than
    GreaterThan,
    /// Greater than or equal
    GreaterThanOrEqual,
    /// Less than
    LessThan,
    /// Less than or equal
    LessThanOrEqual,
    /// Compatible (same major version, >= minor/patch)
    Compatible,
}

impl VersionRequirement {
    /// Create a new version requirement
    pub fn new(comparator: VersionComparator, version: PackageVersion) -> Self {
        Self {
            comparator,
            version,
        }
    }

    /// Parse requirement from string
    pub fn parse(req: &str) -> Result<Self, String> {
        let req = req.trim();

        if req.is_empty() {
            return Err("Empty version requirement".to_string());
        }

        // Handle different formats
        if let Some(version_str) = req.strip_prefix("^") {
            // Compatible version (^1.2.3)
            let version = PackageVersion::parse(version_str)?;
            Ok(Self::new(VersionComparator::Compatible, version))
        } else if let Some(version_str) = req.strip_prefix(">=") {
            let version = PackageVersion::parse(version_str)?;
            Ok(Self::new(VersionComparator::GreaterThanOrEqual, version))
        } else if let Some(version_str) = req.strip_prefix(">") {
            let version = PackageVersion::parse(version_str)?;
            Ok(Self::new(VersionComparator::GreaterThan, version))
        } else if let Some(version_str) = req.strip_prefix("<=") {
            let version = PackageVersion::parse(version_str)?;
            Ok(Self::new(VersionComparator::LessThanOrEqual, version))
        } else if let Some(version_str) = req.strip_prefix("<") {
            let version = PackageVersion::parse(version_str)?;
            Ok(Self::new(VersionComparator::LessThan, version))
        } else if let Some(version_str) = req.strip_prefix("=") {
            let version = PackageVersion::parse(version_str)?;
            Ok(Self::new(VersionComparator::Exact, version))
        } else {
            // Default to exact match
            let version = PackageVersion::parse(req)?;
            Ok(Self::new(VersionComparator::Exact, version))
        }
    }

    /// Check if a version matches this requirement
    pub fn matches(&self, version: &PackageVersion) -> bool {
        match self.comparator {
            VersionComparator::Exact => version == &self.version,
            VersionComparator::GreaterThan => version > &self.version,
            VersionComparator::GreaterThanOrEqual => version >= &self.version,
            VersionComparator::LessThan => version < &self.version,
            VersionComparator::LessThanOrEqual => version <= &self.version,
            VersionComparator::Compatible => {
                version.major == self.version.major
                    && (version.minor > self.version.minor
                        || (version.minor == self.version.minor
                            && version.patch >= self.version.patch))
            }
        }
    }
}

impl fmt::Display for VersionRequirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prefix = match self.comparator {
            VersionComparator::Exact => "=",
            VersionComparator::GreaterThan => ">",
            VersionComparator::GreaterThanOrEqual => ">=",
            VersionComparator::LessThan => "<",
            VersionComparator::LessThanOrEqual => "<=",
            VersionComparator::Compatible => "^",
        };

        write!(f, "{}{}", prefix, self.version)
    }
}

/// Version compatibility checker
pub struct CompatibilityChecker {
    requirements: Vec<VersionRequirement>,
}

impl CompatibilityChecker {
    /// Create a new compatibility checker
    pub fn new() -> Self {
        Self {
            requirements: Vec::new(),
        }
    }

    /// Add a requirement
    pub fn add_requirement(&mut self, requirement: VersionRequirement) {
        self.requirements.push(requirement);
    }

    /// Check if a version satisfies all requirements
    pub fn check(&self, version: &PackageVersion) -> bool {
        self.requirements.iter().all(|req| req.matches(version))
    }

    /// Get all requirements
    pub fn requirements(&self) -> &[VersionRequirement] {
        &self.requirements
    }
}

impl Default for CompatibilityChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parsing() {
        let v1 = PackageVersion::parse("1.2.3").unwrap();
        assert_eq!(v1.major, 1);
        assert_eq!(v1.minor, 2);
        assert_eq!(v1.patch, 3);

        let v2 = PackageVersion::parse("2.0.0-alpha.1").unwrap();
        assert_eq!(v2.major, 2);
        assert_eq!(v2.minor, 0);
        assert_eq!(v2.patch, 0);
        assert_eq!(v2.pre_release.as_deref(), Some("alpha.1"));
    }

    #[test]
    fn test_version_comparison() {
        let v1 = PackageVersion::new(1, 2, 3);
        let v2 = PackageVersion::new(1, 2, 4);
        let v3 = PackageVersion::new(1, 3, 0);
        let v4 = PackageVersion::new(2, 0, 0);

        assert!(v1 < v2);
        assert!(v2 < v3);
        assert!(v3 < v4);
    }

    #[test]
    fn test_version_requirement() {
        let req = VersionRequirement::parse("^1.2.3").unwrap();

        assert!(req.matches(&PackageVersion::new(1, 2, 3)));
        assert!(req.matches(&PackageVersion::new(1, 2, 4)));
        assert!(req.matches(&PackageVersion::new(1, 3, 0)));
        assert!(!req.matches(&PackageVersion::new(1, 2, 2)));
        assert!(!req.matches(&PackageVersion::new(2, 0, 0)));

        let req2 = VersionRequirement::parse(">=1.2.0").unwrap();
        assert!(req2.matches(&PackageVersion::new(1, 2, 0)));
        assert!(req2.matches(&PackageVersion::new(1, 3, 0)));
        assert!(req2.matches(&PackageVersion::new(2, 0, 0)));
        assert!(!req2.matches(&PackageVersion::new(1, 1, 9)));
    }

    #[test]
    fn test_compatibility_checker() {
        let mut checker = CompatibilityChecker::new();
        checker.add_requirement(VersionRequirement::parse(">=1.2.0").unwrap());
        checker.add_requirement(VersionRequirement::parse("<2.0.0").unwrap());

        assert!(checker.check(&PackageVersion::new(1, 2, 0)));
        assert!(checker.check(&PackageVersion::new(1, 5, 3)));
        assert!(!checker.check(&PackageVersion::new(1, 1, 9)));
        assert!(!checker.check(&PackageVersion::new(2, 0, 0)));
    }
}
