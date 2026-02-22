//! Package diagnostics and analysis
//!
//! This module provides comprehensive diagnostics for packages, including
//! health checks, integrity verification, and issue detection.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::error::Result;

use crate::package::Package;
use crate::utils::validate_package_metadata;

/// Package diagnostic report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticReport {
    /// Overall package health score (0-100)
    pub health_score: u8,
    /// Package health status
    pub status: HealthStatus,
    /// List of detected issues
    pub issues: Vec<DiagnosticIssue>,
    /// Package statistics
    pub statistics: PackageStatistics,
    /// Metadata validation results
    pub metadata_validation: ValidationResult,
    /// Resource validation results
    pub resource_validation: Vec<ResourceValidation>,
    /// Security assessment
    pub security: SecurityAssessment,
}

/// Package health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Package is healthy (score >= 90)
    Healthy,
    /// Package has minor issues (score 70-89)
    Warning,
    /// Package has significant issues (score 50-69)
    Degraded,
    /// Package has critical issues (score < 50)
    Critical,
}

/// A diagnostic issue found in the package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue category
    pub category: IssueCategory,
    /// Issue description
    pub description: String,
    /// Recommended action
    pub recommendation: String,
    /// Affected resources or components
    pub affected: Vec<String>,
}

/// Issue severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Informational
    Info,
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Issue categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IssueCategory {
    /// Metadata issues
    Metadata,
    /// Resource issues
    Resource,
    /// Dependency issues
    Dependency,
    /// Security issues
    Security,
    /// Performance issues
    Performance,
    /// Compatibility issues
    Compatibility,
}

/// Package statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageStatistics {
    /// Total package size in bytes
    pub total_size: u64,
    /// Number of resources
    pub resource_count: usize,
    /// Number of dependencies
    pub dependency_count: usize,
    /// Largest resource size
    pub largest_resource_size: u64,
    /// Smallest resource size
    pub smallest_resource_size: u64,
    /// Average resource size
    pub average_resource_size: u64,
    /// Resource type distribution
    pub resource_types: HashMap<String, usize>,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether validation passed
    pub passed: bool,
    /// Validation messages
    pub messages: Vec<String>,
}

/// Resource validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceValidation {
    /// Resource name
    pub name: String,
    /// Whether validation passed
    pub valid: bool,
    /// Validation issues
    pub issues: Vec<String>,
}

/// Security assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAssessment {
    /// Whether package is signed
    pub is_signed: bool,
    /// Whether package is encrypted
    pub is_encrypted: bool,
    /// Security issues found
    pub issues: Vec<String>,
    /// Security score (0-100)
    pub security_score: u8,
}

/// Package diagnostics analyzer
pub struct PackageDiagnostics {
    /// Strict validation mode
    pub strict_mode: bool,
    /// Check security features
    pub check_security: bool,
    /// Check performance issues
    pub check_performance: bool,
}

impl PackageDiagnostics {
    /// Create a new diagnostics analyzer
    pub fn new() -> Self {
        Self {
            strict_mode: false,
            check_security: true,
            check_performance: true,
        }
    }

    /// Run comprehensive diagnostics on a package
    pub fn diagnose(&self, package: &Package) -> Result<DiagnosticReport> {
        let mut issues = Vec::new();

        // Validate metadata
        let metadata_validation = self.validate_metadata(package, &mut issues)?;

        // Validate resources
        let resource_validation = self.validate_resources(package, &mut issues);

        // Check security
        let security = if self.check_security {
            self.assess_security(package, &mut issues)
        } else {
            SecurityAssessment {
                is_signed: false,
                is_encrypted: false,
                issues: Vec::new(),
                security_score: 50,
            }
        };

        // Check performance
        if self.check_performance {
            self.check_performance_issues(package, &mut issues);
        }

        // Calculate statistics
        let statistics = self.calculate_statistics(package);

        // Calculate health score
        let health_score = self.calculate_health_score(&issues, &security);
        let status = Self::determine_status(health_score);

        // Sort issues by severity
        issues.sort_by(|a, b| b.severity.cmp(&a.severity));

        Ok(DiagnosticReport {
            health_score,
            status,
            issues,
            statistics,
            metadata_validation,
            resource_validation,
            security,
        })
    }

    /// Validate package metadata
    fn validate_metadata(
        &self,
        package: &Package,
        issues: &mut Vec<DiagnosticIssue>,
    ) -> Result<ValidationResult> {
        let mut messages = Vec::new();
        let mut passed = true;

        // Validate package name and version
        let name = package.name();
        let version = package.get_version();

        match validate_package_metadata(name, version, None) {
            Ok(_) => messages.push("Package metadata is valid".to_string()),
            Err(e) => {
                passed = false;
                messages.push(format!("Metadata validation failed: {}", e));
                issues.push(DiagnosticIssue {
                    severity: IssueSeverity::High,
                    category: IssueCategory::Metadata,
                    description: format!("Invalid package metadata: {}", e),
                    recommendation: "Fix package name or version to comply with standards"
                        .to_string(),
                    affected: vec!["metadata".to_string()],
                });
            }
        }

        Ok(ValidationResult { passed, messages })
    }

    /// Validate resources
    fn validate_resources(
        &self,
        package: &Package,
        issues: &mut Vec<DiagnosticIssue>,
    ) -> Vec<ResourceValidation> {
        let mut validations = Vec::new();

        for (name, resource) in package.resources() {
            let mut resource_issues = Vec::new();
            let mut valid = true;

            // Validate resource path
            if let Err(e) = crate::utils::validate_resource_path(name) {
                valid = false;
                resource_issues.push(format!("Invalid path: {}", e));

                issues.push(DiagnosticIssue {
                    severity: IssueSeverity::High,
                    category: IssueCategory::Resource,
                    description: format!("Resource '{}' has invalid path", name),
                    recommendation: "Ensure resource paths do not contain special characters or path traversal sequences".to_string(),
                    affected: vec![name.clone()],
                });
            }

            // Check for excessively large resources
            if resource.size() > 100 * 1024 * 1024 {
                // 100 MB
                resource_issues.push("Resource size exceeds 100 MB".to_string());

                issues.push(DiagnosticIssue {
                    severity: IssueSeverity::Medium,
                    category: IssueCategory::Performance,
                    description: format!(
                        "Resource '{}' is very large ({} bytes)",
                        name,
                        resource.size()
                    ),
                    recommendation:
                        "Consider compressing the resource or splitting into smaller chunks"
                            .to_string(),
                    affected: vec![name.clone()],
                });
            }

            validations.push(ResourceValidation {
                name: name.clone(),
                valid,
                issues: resource_issues,
            });
        }

        validations
    }

    /// Assess package security
    fn assess_security(
        &self,
        package: &Package,
        issues: &mut Vec<DiagnosticIssue>,
    ) -> SecurityAssessment {
        let mut security_issues = Vec::new();

        // Check if package is signed
        let is_signed = package.metadata().signature.is_some();

        if !is_signed {
            issues.push(DiagnosticIssue {
                severity: IssueSeverity::Medium,
                category: IssueCategory::Security,
                description: "Package is not digitally signed".to_string(),
                recommendation: "Sign the package to ensure authenticity and integrity using the security module".to_string(),
                affected: vec!["package".to_string()],
            });
            security_issues.push("Package is not signed".to_string());
        }

        // Check for encryption
        let is_encrypted = package.resources().values().any(|resource| {
            resource.metadata.get("encryption").map_or(false, |v| {
                v == "true" || v.starts_with("aes") || v.starts_with("chacha")
            })
        });

        // Check for weak checksums
        let has_weak_checksums = package.resources().values().any(|resource| {
            !resource.metadata.contains_key("sha256") && !resource.metadata.contains_key("sha512")
        });

        if has_weak_checksums {
            issues.push(DiagnosticIssue {
                severity: IssueSeverity::Low,
                category: IssueCategory::Security,
                description: "Some resources lack strong checksums".to_string(),
                recommendation:
                    "Add SHA-256 or SHA-512 checksums to all resources for integrity verification"
                        .to_string(),
                affected: vec!["resources".to_string()],
            });
            security_issues.push("Missing strong checksums".to_string());
        }

        // Calculate security score (0-100)
        let mut security_score = 100;
        if !is_signed {
            security_score -= 30;
        }
        if !is_encrypted {
            security_score -= 15;
        }
        if has_weak_checksums {
            security_score -= 5;
        }

        SecurityAssessment {
            is_signed,
            is_encrypted,
            issues: security_issues,
            security_score: security_score.max(0),
        }
    }

    /// Check for performance issues
    fn check_performance_issues(&self, package: &Package, issues: &mut Vec<DiagnosticIssue>) {
        // Check for large uncompressed resources
        for (name, resource) in package.resources() {
            let size = resource.size();
            let is_compressed = resource.is_compressed();

            if size > 10 * 1024 * 1024 && !is_compressed {
                // > 10 MB uncompressed
                issues.push(DiagnosticIssue {
                    severity: IssueSeverity::Medium,
                    category: IssueCategory::Performance,
                    description: format!(
                        "Resource '{}' is large ({} bytes) and uncompressed",
                        name, size
                    ),
                    recommendation:
                        "Enable compression to reduce package size and improve download times"
                            .to_string(),
                    affected: vec![name.clone()],
                });
            }
        }

        // Check for excessive number of small resources
        let small_resources: Vec<_> = package
            .resources()
            .iter()
            .filter(|(_, resource)| resource.size() < 1024) // < 1 KB
            .map(|(name, _)| name.clone())
            .collect();

        if small_resources.len() > 100 {
            issues.push(DiagnosticIssue {
                severity: IssueSeverity::Low,
                category: IssueCategory::Performance,
                description: format!(
                    "Package contains {} very small resources (< 1 KB each)",
                    small_resources.len()
                ),
                recommendation:
                    "Consider bundling small resources into larger files to reduce overhead"
                        .to_string(),
                affected: small_resources,
            });
        }

        // Check for deep dependency trees
        let dependency_count = package.metadata().dependencies.len();
        if dependency_count > 50 {
            issues.push(DiagnosticIssue {
                severity: IssueSeverity::Medium,
                category: IssueCategory::Performance,
                description: format!("Package has {} dependencies", dependency_count),
                recommendation:
                    "Review dependencies and consider reducing the dependency tree depth"
                        .to_string(),
                affected: vec!["dependencies".to_string()],
            });
        }
    }

    /// Calculate package statistics
    fn calculate_statistics(&self, package: &Package) -> PackageStatistics {
        let total_size: usize = package.resources().values().map(|r| r.size()).sum();
        let resource_count = package.resources().len();
        let dependency_count = package.metadata().dependencies.len();
        let largest_resource_size = package
            .resources()
            .values()
            .map(|r| r.size())
            .max()
            .unwrap_or(0) as u64;

        PackageStatistics {
            total_size: total_size as u64,
            resource_count,
            dependency_count,
            largest_resource_size,
            smallest_resource_size: 0,
            average_resource_size: 0,
            resource_types: HashMap::new(),
        }
    }

    /// Calculate overall health score
    fn calculate_health_score(
        &self,
        issues: &[DiagnosticIssue],
        security: &SecurityAssessment,
    ) -> u8 {
        let mut score = 100u8;

        // Deduct points for issues
        for issue in issues {
            let deduction = match issue.severity {
                IssueSeverity::Critical => 25,
                IssueSeverity::High => 15,
                IssueSeverity::Medium => 8,
                IssueSeverity::Low => 3,
                IssueSeverity::Info => 0,
            };
            score = score.saturating_sub(deduction);
        }

        // Factor in security score
        let weighted_score = (score as f64 * 0.7 + security.security_score as f64 * 0.3) as u8;

        weighted_score
    }

    /// Determine health status from score
    fn determine_status(score: u8) -> HealthStatus {
        match score {
            90..=100 => HealthStatus::Healthy,
            70..=89 => HealthStatus::Warning,
            50..=69 => HealthStatus::Degraded,
            _ => HealthStatus::Critical,
        }
    }
}

impl Default for PackageDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostics_creation() {
        let diagnostics = PackageDiagnostics::new();
        assert!(!diagnostics.strict_mode);
        assert!(diagnostics.check_security);
        assert!(diagnostics.check_performance);
    }

    #[test]
    fn test_health_status() {
        assert_eq!(
            PackageDiagnostics::determine_status(95),
            HealthStatus::Healthy
        );
        assert_eq!(
            PackageDiagnostics::determine_status(80),
            HealthStatus::Warning
        );
        assert_eq!(
            PackageDiagnostics::determine_status(60),
            HealthStatus::Degraded
        );
        assert_eq!(
            PackageDiagnostics::determine_status(40),
            HealthStatus::Critical
        );
    }

    #[test]
    fn test_issue_severity_ordering() {
        assert!(IssueSeverity::Critical > IssueSeverity::High);
        assert!(IssueSeverity::High > IssueSeverity::Medium);
        assert!(IssueSeverity::Medium > IssueSeverity::Low);
        assert!(IssueSeverity::Low > IssueSeverity::Info);
    }

    #[test]
    fn test_diagnostic_issue_creation() {
        let issue = DiagnosticIssue {
            severity: IssueSeverity::High,
            category: IssueCategory::Security,
            description: "Test issue".to_string(),
            recommendation: "Fix it".to_string(),
            affected: vec!["test".to_string()],
        };

        assert_eq!(issue.severity, IssueSeverity::High);
        assert_eq!(issue.category, IssueCategory::Security);
    }

    #[test]
    fn test_security_assessment() {
        let assessment = SecurityAssessment {
            is_signed: true,
            is_encrypted: false,
            issues: Vec::new(),
            security_score: 100,
        };

        assert!(assessment.is_signed);
        assert!(!assessment.is_encrypted);
        assert_eq!(assessment.security_score, 100);
    }

    #[test]
    fn test_validation_result() {
        let result = ValidationResult {
            passed: true,
            messages: vec!["All checks passed".to_string()],
        };

        assert!(result.passed);
        assert_eq!(result.messages.len(), 1);
    }
}
