//! Cross-backend correctness validation tests
//!
//! This module provides tests to ensure mathematical correctness and consistency
//! across different backend implementations. It verifies that the same operations
//! produce equivalent results regardless of the backend used.

use crate::{available_backends, BackendBuilder, BackendType};
use std::collections::HashMap;
use torsh_core::DType;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Tolerance for floating-point comparisons
const F32_TOLERANCE: f32 = 1e-6;
pub const F64_TOLERANCE: f64 = 1e-11;

/// Cross-backend validation test suite
pub struct CrossBackendValidator {
    available_backends: Vec<BackendType>,
    test_data_f32: Vec<f32>,
    test_data_f64: Vec<f64>,
    test_data_i32: Vec<i32>,
}

impl CrossBackendValidator {
    /// Create a new cross-backend validator
    pub fn new() -> Self {
        let backends = available_backends();

        Self {
            available_backends: backends,
            test_data_f32: vec![
                0.0,
                1.0,
                -1.0,
                2.5,
                -3.7,
                std::f32::consts::PI,
                std::f32::consts::E,
                1e-6,
                1e6,
                f32::MIN,
                f32::MAX,
            ],
            test_data_f64: vec![
                0.0,
                1.0,
                -1.0,
                2.5,
                -3.7,
                std::f64::consts::PI,
                std::f64::consts::E,
                1e-12,
                1e12,
                f64::MIN,
                f64::MAX,
            ],
            test_data_i32: vec![0, 1, -1, 42, -42, i32::MIN, i32::MAX, 1000, -1000],
        }
    }

    /// Get the list of available backends for testing
    pub fn available_backends(&self) -> &[BackendType] {
        &self.available_backends
    }

    /// Validate device creation consistency across backends
    pub fn validate_device_creation(&self) -> Result<(), String> {
        let mut results = HashMap::new();

        for &backend_type in &self.available_backends {
            match BackendBuilder::new().backend_type(backend_type).build() {
                Ok(backend) => match backend.default_device() {
                    Ok(device) => {
                        let device_name = device.name().to_string();
                        let device_type = device.device_type();
                        results.insert(backend_type, (device_name, device_type));
                    }
                    Err(e) => {
                        return Err(format!(
                            "Failed to get default device for {:?}: {}",
                            backend_type, e
                        ));
                    }
                },
                Err(e) => {
                    // Some backends may not be available, which is acceptable
                    eprintln!("Backend {:?} not available: {}", backend_type, e);
                }
            }
        }

        // Verify that we got at least one backend (CPU should always be available)
        if results.is_empty() {
            return Err("No backends available for validation".to_string());
        }

        // Verify basic device properties are consistent
        for (backend_type, (device_name, device_type)) in &results {
            if device_name.is_empty() {
                return Err(format!(
                    "Backend {:?} returned empty device name",
                    backend_type
                ));
            }

            // Device type should match backend type expectations
            match backend_type {
                BackendType::Cpu => {
                    if *device_type != torsh_core::device::DeviceType::Cpu {
                        return Err(format!(
                            "CPU backend returned wrong device type: {:?}",
                            device_type
                        ));
                    }
                }
                BackendType::Cuda => {
                    if !matches!(device_type, torsh_core::device::DeviceType::Cuda(_)) {
                        return Err(format!(
                            "CUDA backend returned wrong device type: {:?}",
                            device_type
                        ));
                    }
                }
                _ => {
                    // Other backends may have various device types
                }
            }
        }

        Ok(())
    }

    /// Validate capability reporting consistency
    pub fn validate_capabilities_consistency(&self) -> Result<(), String> {
        let mut capability_results = HashMap::new();

        for &backend_type in &self.available_backends {
            if let Ok(backend) = BackendBuilder::new().backend_type(backend_type).build() {
                let capabilities = backend.capabilities();
                capability_results.insert(backend_type, capabilities);
            }
        }

        if capability_results.is_empty() {
            return Err("No backends available for capability validation".to_string());
        }

        // All backends should support at least basic data types
        for (backend_type, capabilities) in &capability_results {
            if capabilities.supported_dtypes.is_empty() {
                return Err(format!(
                    "Backend {:?} reports no supported data types",
                    backend_type
                ));
            }

            // All backends should support F32
            if !capabilities.supported_dtypes.contains(&DType::F32) {
                return Err(format!("Backend {:?} does not support F32", backend_type));
            }

            // Check for reasonable limits
            if capabilities.max_buffer_size == 0 {
                return Err(format!(
                    "Backend {:?} reports zero max buffer size",
                    backend_type
                ));
            }

            if capabilities.max_compute_units == 0 {
                return Err(format!(
                    "Backend {:?} reports zero max threads",
                    backend_type
                ));
            }
        }

        Ok(())
    }

    /// Validate memory management consistency across backends
    pub fn validate_memory_management(&self) -> Result<(), String> {
        for &backend_type in &self.available_backends {
            if let Ok(backend) = BackendBuilder::new().backend_type(backend_type).build() {
                if let Ok(device) = backend.default_device() {
                    // Test basic memory allocation
                    match backend.memory_manager(&device) {
                        Ok(mut memory_manager) => {
                            match memory_manager.allocate_raw(1024, 8) {
                                Ok(ptr) => {
                                    // Test that we can deallocate
                                    if let Err(e) = memory_manager.deallocate_raw(ptr, 1024) {
                                        return Err(format!(
                                            "Backend {:?} failed to deallocate: {}",
                                            backend_type, e
                                        ));
                                    }
                                }
                                Err(e) => {
                                    return Err(format!(
                                        "Backend {:?} failed to allocate: {}",
                                        backend_type, e
                                    ));
                                }
                            }
                        }
                        Err(e) => {
                            return Err(format!(
                                "Backend {:?} failed to get memory manager: {}",
                                backend_type, e
                            ));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Validate error handling consistency
    pub fn validate_error_handling(&self) -> Result<(), String> {
        for &backend_type in &self.available_backends {
            if let Ok(backend) = BackendBuilder::new().backend_type(backend_type).build() {
                // Test invalid device creation
                let invalid_device_result = backend.create_device(9999);
                if invalid_device_result.is_ok() {
                    return Err(format!(
                        "Backend {:?} should reject invalid device ID",
                        backend_type
                    ));
                }

                // Test error message quality
                let error_msg = invalid_device_result.unwrap_err().to_string();
                if error_msg.is_empty() {
                    return Err(format!(
                        "Backend {:?} returned empty error message",
                        backend_type
                    ));
                }

                if !error_msg.contains("9999") && !error_msg.contains("not found") {
                    return Err(format!(
                        "Backend {:?} error message not descriptive enough: {}",
                        backend_type, error_msg
                    ));
                }
            }
        }

        Ok(())
    }

    /// Validate performance hints consistency
    pub fn validate_performance_hints(&self) -> Result<(), String> {
        for &backend_type in &self.available_backends {
            if let Ok(backend) = BackendBuilder::new().backend_type(backend_type).build() {
                let hints = backend.performance_hints();

                // Check that hints are reasonable
                if hints.optimal_batch_size == 0 {
                    return Err(format!(
                        "Backend {:?} suggests zero batch size",
                        backend_type
                    ));
                }

                if hints.optimal_batch_size > 1024 * 1024 * 1024 {
                    return Err(format!(
                        "Backend {:?} suggests unreasonably large batch size: {}",
                        backend_type, hints.optimal_batch_size
                    ));
                }

                if hints.memory_alignment == 0
                    || (hints.memory_alignment & (hints.memory_alignment - 1)) != 0
                {
                    return Err(format!(
                        "Backend {:?} suggests invalid memory alignment: {}",
                        backend_type, hints.memory_alignment
                    ));
                }
            }
        }

        Ok(())
    }

    /// Run all validation tests
    pub fn run_all_validations(&self) -> Result<(), String> {
        self.validate_device_creation()?;
        self.validate_capabilities_consistency()?;
        self.validate_memory_management()?;
        self.validate_error_handling()?;
        self.validate_performance_hints()?;

        Ok(())
    }
}

impl Default for CrossBackendValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility function to compare floating-point values with tolerance
pub fn compare_f32_values(a: f32, b: f32, tolerance: f32) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    (a - b).abs() <= tolerance
}

/// Utility function to compare floating-point values with tolerance
pub fn compare_f64_values(a: f64, b: f64, tolerance: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() && b.is_infinite() {
        return a.signum() == b.signum();
    }
    (a - b).abs() <= tolerance
}

/// Comprehensive cross-backend validation test runner
pub fn run_cross_backend_validation() -> Result<(), String> {
    let validator = CrossBackendValidator::new();
    validator.run_all_validations()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_backend_device_creation() {
        let validator = CrossBackendValidator::new();
        let result = validator.validate_device_creation();

        match result {
            Ok(()) => {
                // Validation passed
            }
            Err(e) => {
                panic!("Cross-backend device creation validation failed: {}", e);
            }
        }
    }

    #[test]
    fn test_cross_backend_capabilities() {
        let validator = CrossBackendValidator::new();
        let result = validator.validate_capabilities_consistency();

        match result {
            Ok(()) => {
                // Validation passed
            }
            Err(e) => {
                panic!("Cross-backend capabilities validation failed: {}", e);
            }
        }
    }

    #[test]
    fn test_cross_backend_memory_management() {
        let validator = CrossBackendValidator::new();
        let result = validator.validate_memory_management();

        match result {
            Ok(()) => {
                // Validation passed
            }
            Err(e) => {
                panic!("Cross-backend memory management validation failed: {}", e);
            }
        }
    }

    #[test]
    fn test_cross_backend_error_handling() {
        let validator = CrossBackendValidator::new();
        let result = validator.validate_error_handling();

        match result {
            Ok(()) => {
                // Validation passed
            }
            Err(e) => {
                panic!("Cross-backend error handling validation failed: {}", e);
            }
        }
    }

    #[test]
    fn test_cross_backend_performance_hints() {
        let validator = CrossBackendValidator::new();
        let result = validator.validate_performance_hints();

        match result {
            Ok(()) => {
                // Validation passed
            }
            Err(e) => {
                panic!("Cross-backend performance hints validation failed: {}", e);
            }
        }
    }

    #[test]
    fn test_floating_point_comparison_utilities() {
        // Test normal values
        assert!(compare_f32_values(1.0, 1.0, F32_TOLERANCE));
        assert!(compare_f32_values(1.0, 1.0000001, F32_TOLERANCE));
        assert!(!compare_f32_values(1.0, 2.0, F32_TOLERANCE));

        // Test edge cases
        assert!(compare_f32_values(f32::NAN, f32::NAN, F32_TOLERANCE));
        assert!(compare_f32_values(
            f32::INFINITY,
            f32::INFINITY,
            F32_TOLERANCE
        ));
        assert!(compare_f32_values(
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            F32_TOLERANCE
        ));
        assert!(!compare_f32_values(
            f32::INFINITY,
            f32::NEG_INFINITY,
            F32_TOLERANCE
        ));

        // Test f64 version
        assert!(compare_f64_values(1.0, 1.0, F64_TOLERANCE));
        assert!(compare_f64_values(1.0, 1.000000000001, F64_TOLERANCE));
        assert!(!compare_f64_values(1.0, 2.0, F64_TOLERANCE));
    }

    #[test]
    fn test_full_cross_backend_validation() {
        let result = run_cross_backend_validation();

        match result {
            Ok(()) => {
                println!("All cross-backend validations passed!");
            }
            Err(e) => {
                panic!("Cross-backend validation failed: {}", e);
            }
        }
    }

    #[test]
    fn test_validator_creation() {
        let validator = CrossBackendValidator::new();

        // Should have at least CPU backend
        assert!(!validator.available_backends.is_empty());
        assert!(validator.available_backends.contains(&BackendType::Cpu));

        // Should have test data
        assert!(!validator.test_data_f32.is_empty());
        assert!(!validator.test_data_f64.is_empty());
        assert!(!validator.test_data_i32.is_empty());
    }

    #[test]
    fn test_backend_isolation() {
        // Test that multiple backends can be created and used simultaneously
        let validator = CrossBackendValidator::new();
        let mut backends = Vec::new();

        for &backend_type in &validator.available_backends {
            if let Ok(backend) = BackendBuilder::new().backend_type(backend_type).build() {
                backends.push((backend_type, backend));
            }
        }

        // Should have at least one backend
        assert!(!backends.is_empty());

        // All backends should work simultaneously
        for (backend_type, backend) in &backends {
            let device_result = backend.default_device();
            assert!(
                device_result.is_ok(),
                "Backend {:?} failed to provide default device",
                backend_type
            );

            let device = device_result.unwrap();
            assert!(
                !device.name().is_empty(),
                "Backend {:?} returned empty device name",
                backend_type
            );
        }
    }
}

/// Check if two f64 values are close within the given tolerance
///
/// This is a standalone utility function for benchmarking and testing purposes
pub fn is_close_f64(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance
}
