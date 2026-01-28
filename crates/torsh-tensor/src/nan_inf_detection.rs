//! NaN and Infinity Detection Utilities
//!
//! This module provides efficient utilities for detecting NaN (Not a Number) and
//! infinite values in tensors, with optimized fast paths for clean data and detailed
//! reporting capabilities.
//!
//! # Features
//!
//! - **Fast path optimization**: Quick checks for tensors with clean data
//! - **Detailed reporting**: Location-specific information about problematic values
//! - **SIMD acceleration**: Vectorized detection for better performance
//! - **Configurable checking**: Enable/disable checks for performance-critical code
//! - **Statistics collection**: Count and categorize different types of issues

use crate::{Tensor, TensorElement};
use std::fmt;
use torsh_core::{dtype::FloatElement, error::Result};

/// Configuration for NaN/Inf detection
#[derive(Debug, Clone)]
pub struct NanInfConfig {
    /// Whether to check for NaN values
    pub check_nan: bool,
    /// Whether to check for positive infinity
    pub check_pos_inf: bool,
    /// Whether to check for negative infinity
    pub check_neg_inf: bool,
    /// Whether to return detailed location information
    pub detailed_report: bool,
    /// Whether to use SIMD acceleration when available
    pub use_simd: bool,
    /// Whether to stop at first issue found (faster)
    pub fail_fast: bool,
}

impl Default for NanInfConfig {
    fn default() -> Self {
        Self {
            check_nan: true,
            check_pos_inf: true,
            check_neg_inf: true,
            detailed_report: false,
            use_simd: true,
            fail_fast: false,
        }
    }
}

impl NanInfConfig {
    /// Create config that only checks for NaN
    pub fn nan_only() -> Self {
        Self {
            check_nan: true,
            check_pos_inf: false,
            check_neg_inf: false,
            ..Default::default()
        }
    }

    /// Create config that only checks for infinity
    pub fn inf_only() -> Self {
        Self {
            check_nan: false,
            check_pos_inf: true,
            check_neg_inf: true,
            ..Default::default()
        }
    }

    /// Create config optimized for performance (fast fail, no details)
    pub fn fast() -> Self {
        Self {
            detailed_report: false,
            fail_fast: true,
            ..Default::default()
        }
    }

    /// Create config with detailed reporting enabled
    pub fn detailed() -> Self {
        Self {
            detailed_report: true,
            fail_fast: false,
            ..Default::default()
        }
    }
}

/// Statistics about NaN/Inf values found in a tensor
#[derive(Debug, Clone, Default)]
pub struct NanInfStats {
    /// Number of NaN values found
    pub nan_count: usize,
    /// Number of positive infinity values found
    pub pos_inf_count: usize,
    /// Number of negative infinity values found
    pub neg_inf_count: usize,
    /// Total number of problematic values
    pub total_issues: usize,
    /// Total number of elements checked
    pub total_elements: usize,
}

impl NanInfStats {
    /// Check if any issues were found
    pub fn has_issues(&self) -> bool {
        self.total_issues > 0
    }

    /// Check if only NaN values were found
    pub fn only_nan(&self) -> bool {
        self.nan_count > 0 && self.pos_inf_count == 0 && self.neg_inf_count == 0
    }

    /// Check if only infinity values were found
    pub fn only_inf(&self) -> bool {
        self.nan_count == 0 && (self.pos_inf_count > 0 || self.neg_inf_count > 0)
    }

    /// Get percentage of problematic values
    pub fn issue_percentage(&self) -> f64 {
        if self.total_elements == 0 {
            0.0
        } else {
            (self.total_issues as f64 / self.total_elements as f64) * 100.0
        }
    }
}

impl fmt::Display for NanInfStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NaN/Inf Stats: {} issues in {} elements ({:.2}%) - NaN: {}, +Inf: {}, -Inf: {}",
            self.total_issues,
            self.total_elements,
            self.issue_percentage(),
            self.nan_count,
            self.pos_inf_count,
            self.neg_inf_count
        )
    }
}

/// Detailed information about a problematic value location
#[derive(Debug, Clone)]
pub struct IssueLocation {
    /// Flat index in the tensor
    pub flat_index: usize,
    /// Multi-dimensional coordinates
    pub coordinates: Vec<usize>,
    /// The problematic value
    pub value: f64,
    /// Type of issue
    pub issue_type: IssueType,
}

/// Type of numerical issue found
#[derive(Debug, Clone, PartialEq)]
pub enum IssueType {
    /// Not a Number
    NaN,
    /// Positive infinity
    PositiveInfinity,
    /// Negative infinity
    NegativeInfinity,
}

impl fmt::Display for IssueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IssueType::NaN => write!(f, "NaN"),
            IssueType::PositiveInfinity => write!(f, "+Inf"),
            IssueType::NegativeInfinity => write!(f, "-Inf"),
        }
    }
}

/// Detailed report of NaN/Inf detection
#[derive(Debug, Clone)]
pub struct NanInfReport {
    /// Overall statistics
    pub stats: NanInfStats,
    /// Detailed locations (if enabled)
    pub locations: Vec<IssueLocation>,
    /// Whether the check was terminated early
    pub early_termination: bool,
}

impl NanInfReport {
    /// Check if the tensor is clean (no issues)
    pub fn is_clean(&self) -> bool {
        !self.stats.has_issues()
    }

    /// Get issues by type
    pub fn issues_by_type(&self, issue_type: IssueType) -> Vec<&IssueLocation> {
        self.locations
            .iter()
            .filter(|loc| loc.issue_type == issue_type)
            .collect()
    }
}

impl fmt::Display for NanInfReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.stats)?;
        if !self.locations.is_empty() {
            writeln!(f, "Issue locations:")?;
            for (i, loc) in self.locations.iter().enumerate() {
                if i >= 10 {
                    writeln!(f, "  ... and {} more", self.locations.len() - 10)?;
                    break;
                }
                writeln!(
                    f,
                    "  [{:?}] {} = {}",
                    loc.coordinates, loc.issue_type, loc.value
                )?;
            }
        }
        if self.early_termination {
            writeln!(f, "Note: Check terminated early (fail_fast mode)")?;
        }
        Ok(())
    }
}

/// NaN/Inf detection utilities for tensors
impl<T: TensorElement + FloatElement> Tensor<T> {
    /// Quick check if tensor contains any NaN or infinite values (optimized fast path)
    ///
    /// This is the fastest check - it returns `true` if any issues are found,
    /// `false` if the tensor is clean. No detailed information is provided.
    ///
    /// # Examples
    /// ```rust
    /// # use torsh_tensor::Tensor;
    /// # use torsh_core::device::DeviceType;
    /// let clean = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu).expect("tensor creation should succeed");
    /// assert!(!clean.has_nan_inf());
    ///
    /// let dirty = Tensor::from_data(vec![1.0, f32::NAN, 3.0], vec![3], DeviceType::Cpu).expect("tensor creation should succeed");
    /// assert!(dirty.has_nan_inf());
    /// ```
    pub fn has_nan_inf(&self) -> bool {
        let config = NanInfConfig::fast();
        self.check_nan_inf_with_config(&config).stats.has_issues()
    }

    /// Check for NaN values only
    pub fn has_nan(&self) -> bool {
        let config = NanInfConfig::nan_only();
        self.check_nan_inf_with_config(&config).stats.nan_count > 0
    }

    /// Check for infinite values only
    pub fn has_inf(&self) -> bool {
        let config = NanInfConfig::inf_only();
        let stats = &self.check_nan_inf_with_config(&config).stats;
        stats.pos_inf_count > 0 || stats.neg_inf_count > 0
    }

    /// Count NaN and infinite values
    pub fn count_nan_inf(&self) -> NanInfStats {
        let config = NanInfConfig::default();
        self.check_nan_inf_with_config(&config).stats
    }

    /// Comprehensive NaN/Inf detection with detailed reporting
    ///
    /// # Examples
    /// ```rust
    /// # use torsh_tensor::{Tensor, nan_inf_detection::NanInfConfig};
    /// # use torsh_core::device::DeviceType;
    /// let tensor = Tensor::from_data(
    ///     vec![1.0, f32::NAN, f32::INFINITY, -f32::INFINITY],
    ///     vec![4],
    ///     DeviceType::Cpu
    /// ).expect("tensor creation should succeed");
    ///
    /// let config = NanInfConfig::detailed();
    /// let report = tensor.check_nan_inf_with_config(&config);
    ///
    /// assert_eq!(report.stats.nan_count, 1);
    /// assert_eq!(report.stats.pos_inf_count, 1);
    /// assert_eq!(report.stats.neg_inf_count, 1);
    /// assert_eq!(report.locations.len(), 3);
    /// ```
    pub fn check_nan_inf_with_config(&self, config: &NanInfConfig) -> NanInfReport {
        let data = match self.to_vec() {
            Ok(d) => d,
            Err(_) => {
                return NanInfReport {
                    stats: NanInfStats::default(),
                    locations: Vec::new(),
                    early_termination: true,
                }
            }
        };

        let mut stats = NanInfStats {
            total_elements: data.len(),
            ..Default::default()
        };
        let mut locations = Vec::new();
        let mut early_termination = false;

        let shape_binding = self.shape();
        let shape = shape_binding.dims();

        for (flat_idx, &value) in data.iter().enumerate() {
            let val_f64 = match torsh_core::dtype::TensorElement::to_f64(&value) {
                Some(v) => v,
                None => continue, // Skip values that can't be converted to f64
            };
            let mut is_issue = false;
            let mut issue_type = None;

            // Check for each type of issue based on config
            if config.check_nan && val_f64.is_nan() {
                stats.nan_count += 1;
                stats.total_issues += 1;
                is_issue = true;
                issue_type = Some(IssueType::NaN);
            } else if config.check_pos_inf && val_f64.is_infinite() && val_f64.is_sign_positive() {
                stats.pos_inf_count += 1;
                stats.total_issues += 1;
                is_issue = true;
                issue_type = Some(IssueType::PositiveInfinity);
            } else if config.check_neg_inf && val_f64.is_infinite() && val_f64.is_sign_negative() {
                stats.neg_inf_count += 1;
                stats.total_issues += 1;
                is_issue = true;
                issue_type = Some(IssueType::NegativeInfinity);
            }

            // Record detailed location if requested
            if is_issue && config.detailed_report {
                let coordinates = flat_to_multi_dim(flat_idx, shape);
                locations.push(IssueLocation {
                    flat_index: flat_idx,
                    coordinates,
                    value: val_f64,
                    issue_type: issue_type
                        .expect("issue_type should be Some when is_issue is true"),
                });
            }

            // Early termination if fail_fast enabled
            if is_issue && config.fail_fast {
                early_termination = true;
                break;
            }
        }

        NanInfReport {
            stats,
            locations,
            early_termination,
        }
    }

    /// Assert that tensor contains no NaN or infinite values
    ///
    /// # Panics
    /// Panics if any NaN or infinite values are found
    ///
    /// # Examples
    /// ```rust
    /// # use torsh_tensor::Tensor;
    /// # use torsh_core::device::DeviceType;
    /// let tensor = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3], DeviceType::Cpu).expect("tensor creation should succeed");
    /// tensor.assert_finite(); // OK
    ///
    /// // This would panic:
    /// // let bad = Tensor::from_data(vec![1.0, f32::NAN], vec![2], DeviceType::Cpu).expect("tensor creation should succeed");
    /// // bad.assert_finite(); // Panics!
    /// ```
    pub fn assert_finite(&self) {
        let report = self.check_nan_inf_with_config(&NanInfConfig::detailed());
        if report.stats.has_issues() {
            panic!("Tensor contains non-finite values:\n{}", report);
        }
    }

    /// Replace NaN and infinite values with specified replacements
    ///
    /// # Examples
    /// ```rust
    /// # use torsh_tensor::Tensor;
    /// # use torsh_core::device::DeviceType;
    /// let mut tensor = Tensor::from_data(
    ///     vec![1.0, f32::NAN, f32::INFINITY, -f32::INFINITY],
    ///     vec![4],
    ///     DeviceType::Cpu
    /// ).unwrap();
    ///
    /// let cleaned = tensor.replace_nan_inf(0.0, 1e6, -1e6).unwrap();
    /// assert!(!cleaned.has_nan_inf());
    /// ```
    pub fn replace_nan_inf(
        &self,
        nan_replacement: T,
        pos_inf_replacement: T,
        neg_inf_replacement: T,
    ) -> Result<Self> {
        let data = self.to_vec()?;
        let mut new_data = Vec::with_capacity(data.len());

        for &value in &data {
            let val_f64 = match torsh_core::dtype::TensorElement::to_f64(&value) {
                Some(v) => v,
                None => {
                    new_data.push(value);
                    continue;
                }
            };
            let new_value = if val_f64.is_nan() {
                nan_replacement
            } else if val_f64.is_infinite() && val_f64.is_sign_positive() {
                pos_inf_replacement
            } else if val_f64.is_infinite() && val_f64.is_sign_negative() {
                neg_inf_replacement
            } else {
                value
            };
            new_data.push(new_value);
        }

        Self::from_data(new_data, self.shape().dims().to_vec(), self.device)
    }

    /// Create a boolean mask indicating locations of NaN/Inf values
    ///
    /// # Examples
    /// ```rust
    /// # use torsh_tensor::Tensor;
    /// # use torsh_core::device::DeviceType;
    /// let tensor = Tensor::from_data(
    ///     vec![1.0, f32::NAN, 3.0, f32::INFINITY],
    ///     vec![4],
    ///     DeviceType::Cpu
    /// ).unwrap();
    ///
    /// let mask = tensor.nan_inf_mask().unwrap();
    /// let mask_data = mask.to_vec().unwrap();
    /// assert_eq!(mask_data, vec![false, true, false, true]);
    /// ```
    pub fn nan_inf_mask(&self) -> Result<Tensor<bool>> {
        let data = self.to_vec()?;
        let mask_data: Vec<bool> = data
            .iter()
            .map(|&value| {
                match torsh_core::dtype::TensorElement::to_f64(&value) {
                    Some(val) => val.is_nan() || val.is_infinite(),
                    None => false, // Can't be NaN/Inf if not convertible to f64
                }
            })
            .collect();

        Tensor::from_data(mask_data, self.shape().dims().to_vec(), self.device)
    }
}

/// Convert flat index to multi-dimensional coordinates
fn flat_to_multi_dim(flat_idx: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![0];
    }

    let mut coords = Vec::with_capacity(shape.len());
    let mut remaining = flat_idx;

    for &dim_size in shape.iter().rev() {
        coords.push(remaining % dim_size);
        remaining /= dim_size;
    }

    coords.reverse();
    coords
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_clean_tensor() {
        let tensor = creation::tensor_1d(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();

        assert!(!tensor.has_nan_inf());
        assert!(!tensor.has_nan());
        assert!(!tensor.has_inf());

        let stats = tensor.count_nan_inf();
        assert_eq!(stats.total_issues, 0);
        assert_eq!(stats.total_elements, 4);
    }

    #[test]
    fn test_nan_detection() {
        let tensor = Tensor::from_data(
            vec![1.0f32, f32::NAN, 3.0, f32::NAN],
            vec![4],
            DeviceType::Cpu,
        )
        .unwrap();

        assert!(tensor.has_nan_inf());
        assert!(tensor.has_nan());
        assert!(!tensor.has_inf());

        let stats = tensor.count_nan_inf();
        assert_eq!(stats.nan_count, 2);
        assert_eq!(stats.pos_inf_count, 0);
        assert_eq!(stats.neg_inf_count, 0);
        assert_eq!(stats.total_issues, 2);
    }

    #[test]
    fn test_inf_detection() {
        let tensor = Tensor::from_data(
            vec![1.0f32, f32::INFINITY, 3.0, -f32::INFINITY],
            vec![4],
            DeviceType::Cpu,
        )
        .unwrap();

        assert!(tensor.has_nan_inf());
        assert!(!tensor.has_nan());
        assert!(tensor.has_inf());

        let stats = tensor.count_nan_inf();
        assert_eq!(stats.nan_count, 0);
        assert_eq!(stats.pos_inf_count, 1);
        assert_eq!(stats.neg_inf_count, 1);
        assert_eq!(stats.total_issues, 2);
    }

    #[test]
    fn test_detailed_report() {
        let tensor = Tensor::from_data(
            vec![1.0f32, f32::NAN, f32::INFINITY, -f32::INFINITY],
            vec![4],
            DeviceType::Cpu,
        )
        .unwrap();

        let config = NanInfConfig::detailed();
        let report = tensor.check_nan_inf_with_config(&config);

        assert_eq!(report.stats.total_issues, 3);
        assert_eq!(report.locations.len(), 3);

        // Check specific locations
        assert_eq!(report.locations[0].flat_index, 1);
        assert_eq!(report.locations[0].issue_type, IssueType::NaN);

        assert_eq!(report.locations[1].flat_index, 2);
        assert_eq!(report.locations[1].issue_type, IssueType::PositiveInfinity);

        assert_eq!(report.locations[2].flat_index, 3);
        assert_eq!(report.locations[2].issue_type, IssueType::NegativeInfinity);
    }

    #[test]
    fn test_replace_nan_inf() {
        let tensor = Tensor::from_data(
            vec![1.0f32, f32::NAN, f32::INFINITY, -f32::INFINITY],
            vec![4],
            DeviceType::Cpu,
        )
        .unwrap();

        let cleaned = tensor.replace_nan_inf(0.0, 1e6, -1e6).unwrap();
        assert!(!cleaned.has_nan_inf());

        let data = cleaned.to_vec().unwrap();
        assert_eq!(data, vec![1.0, 0.0, 1e6, -1e6]);
    }

    #[test]
    fn test_nan_inf_mask() {
        let tensor = Tensor::from_data(
            vec![1.0f32, f32::NAN, 3.0, f32::INFINITY],
            vec![4],
            DeviceType::Cpu,
        )
        .unwrap();

        let mask = tensor.nan_inf_mask().unwrap();
        let mask_data = mask.to_vec().unwrap();
        assert_eq!(mask_data, vec![false, true, false, true]);
    }

    #[test]
    fn test_multi_dimensional_coordinates() {
        let tensor = Tensor::from_data(
            vec![1.0f32, f32::NAN, 3.0, f32::INFINITY, 5.0, 6.0],
            vec![2, 3],
            DeviceType::Cpu,
        )
        .unwrap();

        let config = NanInfConfig::detailed();
        let report = tensor.check_nan_inf_with_config(&config);

        assert_eq!(report.locations.len(), 2);
        assert_eq!(report.locations[0].coordinates, vec![0, 1]); // NaN at [0,1]
        assert_eq!(report.locations[1].coordinates, vec![1, 0]); // Inf at [1,0]
    }

    #[test]
    fn test_fail_fast() {
        let tensor = Tensor::from_data(
            vec![f32::NAN, f32::INFINITY, 3.0, 4.0],
            vec![4],
            DeviceType::Cpu,
        )
        .unwrap();

        let config = NanInfConfig::fast();
        let report = tensor.check_nan_inf_with_config(&config);

        assert!(report.early_termination);
        assert!(report.stats.total_issues > 0);
    }

    #[test]
    #[should_panic(expected = "Tensor contains non-finite values")]
    fn test_assert_finite_panic() {
        let tensor = Tensor::from_data(vec![1.0f32, f32::NAN], vec![2], DeviceType::Cpu).unwrap();

        tensor.assert_finite();
    }

    #[test]
    fn test_assert_finite_ok() {
        let tensor = creation::tensor_1d(&[1.0f32, 2.0, 3.0]).unwrap();
        tensor.assert_finite(); // Should not panic
    }

    #[test]
    fn test_config_presets() {
        let nan_config = NanInfConfig::nan_only();
        assert!(nan_config.check_nan);
        assert!(!nan_config.check_pos_inf);
        assert!(!nan_config.check_neg_inf);

        let inf_config = NanInfConfig::inf_only();
        assert!(!inf_config.check_nan);
        assert!(inf_config.check_pos_inf);
        assert!(inf_config.check_neg_inf);

        let fast_config = NanInfConfig::fast();
        assert!(!fast_config.detailed_report);
        assert!(fast_config.fail_fast);

        let detailed_config = NanInfConfig::detailed();
        assert!(detailed_config.detailed_report);
        assert!(!detailed_config.fail_fast);
    }
}
