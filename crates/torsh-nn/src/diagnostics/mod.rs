//! Module diagnostics and analysis system
//!
//! This module provides comprehensive diagnostic capabilities for neural network
//! modules including health checking, statistical analysis, and troubleshooting.

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

use crate::{ParameterDiagnostics, ParameterStats};

/// Module information for debugging and introspection
#[derive(Debug, Clone)]
pub struct ModuleInfo {
    /// Module name (if available)
    pub name: String,
    /// Whether module is in training mode
    pub training: bool,
    /// Total number of parameters
    pub parameter_count: usize,
    /// Number of trainable parameters
    pub trainable_parameter_count: usize,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Whether module has child modules
    pub has_children: bool,
    /// Number of direct child modules
    pub children_count: usize,
}

impl core::fmt::Display for ModuleInfo {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "Module: {}", self.name)?;
        writeln!(f, "  Training: {}", self.training)?;
        writeln!(
            f,
            "  Parameters: {} ({} trainable)",
            self.parameter_count, self.trainable_parameter_count
        )?;
        writeln!(
            f,
            "  Memory: {:.2} MB",
            self.memory_usage_bytes as f64 / (1024.0 * 1024.0)
        )?;
        writeln!(
            f,
            "  Children: {} (has_children: {})",
            self.children_count, self.has_children
        )?;
        Ok(())
    }
}

/// Comprehensive module diagnostics
#[derive(Debug, Clone)]
pub struct ModuleDiagnostics {
    /// Basic module information
    pub module_info: ModuleInfo,
    /// Critical issues that need immediate attention
    pub issues: Vec<String>,
    /// Warnings that should be addressed
    pub warnings: Vec<String>,
    /// Parameter-specific diagnostics
    pub parameter_diagnostics: HashMap<String, ParameterDiagnostics>,
}

impl ModuleDiagnostics {
    /// Check if module has any critical issues
    pub fn has_issues(&self) -> bool {
        !self.issues.is_empty()
    }

    /// Check if module has any warnings
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Check if module is healthy (no issues or warnings)
    pub fn is_healthy(&self) -> bool {
        !self.has_issues() && !self.has_warnings()
    }

    /// Get a summary of the health status
    pub fn health_summary(&self) -> String {
        if self.is_healthy() {
            "Healthy".to_string()
        } else {
            format!(
                "{} issues, {} warnings",
                self.issues.len(),
                self.warnings.len()
            )
        }
    }
}

impl core::fmt::Display for ModuleDiagnostics {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "Module Diagnostics:")?;
        writeln!(f, "{}", self.module_info)?;

        if !self.issues.is_empty() {
            writeln!(f, "Issues:")?;
            for issue in &self.issues {
                writeln!(f, "  ❌ {}", issue)?;
            }
        }

        if !self.warnings.is_empty() {
            writeln!(f, "Warnings:")?;
            for warning in &self.warnings {
                writeln!(f, "  ⚠️  {}", warning)?;
            }
        }

        if self.is_healthy() {
            writeln!(f, "✅ Module appears healthy")?;
        }

        writeln!(f, "Parameter Health:")?;
        for (name, param_diag) in &self.parameter_diagnostics {
            if param_diag.issues.is_empty() && param_diag.warnings.is_empty() {
                writeln!(f, "  ✅ {}: Healthy", name)?;
            } else {
                writeln!(
                    f,
                    "  ⚠️  {}: {} issues, {} warnings",
                    name,
                    param_diag.issues.len(),
                    param_diag.warnings.len()
                )?;
            }
        }

        Ok(())
    }
}
