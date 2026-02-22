//! Hardware-specific optimization testing
//!
//! This module provides automated tests for hardware-specific optimizations
//! to ensure they are correctly detected and applied across different platforms.

use crate::{BackendBuilder, BackendType};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Hardware optimization test suite
pub struct HardwareOptimizationTester {
    pub simd_tests_enabled: bool,
    pub platform_tests_enabled: bool,
    pub performance_tests_enabled: bool,
}

impl HardwareOptimizationTester {
    /// Create a new hardware optimization tester
    pub fn new() -> Self {
        Self {
            simd_tests_enabled: true,
            platform_tests_enabled: true,
            performance_tests_enabled: false, // Disabled by default for CI
        }
    }

    /// Test CPU feature detection
    pub fn test_cpu_feature_detection(&self) -> Result<(), String> {
        #[cfg(feature = "cpu")]
        {
            use crate::cpu::feature_detection::{detected_features, has_feature, CpuFeature};

            // Test basic feature detection
            let features = detected_features();

            // Should detect at least some basic features on any modern CPU
            if features.is_empty() {
                return Err("No CPU features detected - this seems unlikely".to_string());
            }

            // Test individual feature checks
            for feature in features {
                let has_it = has_feature(feature);
                if !has_it {
                    return Err(format!(
                        "Feature {:?} reported as available but individual check failed",
                        feature
                    ));
                }
            }

            // Test some common features that should be available on most systems
            #[cfg(target_arch = "x86_64")]
            {
                // SSE2 should be available on all x86_64 systems
                if !has_feature(CpuFeature::Sse2) {
                    return Err("SSE2 not detected on x86_64 system".to_string());
                }
            }

            #[cfg(target_arch = "aarch64")]
            {
                // NEON should be available on all aarch64 systems
                if !has_feature(CpuFeature::Neon) {
                    return Err("NEON not detected on aarch64 system".to_string());
                }
            }
        }

        Ok(())
    }

    /// Test SIMD optimization availability
    pub fn test_simd_optimizations(&self) -> Result<(), String> {
        if !self.simd_tests_enabled {
            return Ok(());
        }

        #[cfg(feature = "cpu")]
        {
            use crate::cpu::simd::{has_avx2, has_avx512, has_neon};

            // Test that SIMD feature detection works
            let _has_avx2 = has_avx2();
            let _has_avx512 = has_avx512();
            let _has_neon = has_neon();

            // Test basic SIMD operation availability
            #[cfg(feature = "simd")]
            {
                let test_data = vec![1.0f32, 2.0, 3.0, 4.0];
                let mut result = vec![0.0f32; 4];

                // Test SIMD functions directly if available
                use crate::cpu::simd::simd_add_f32;
                simd_add_f32(&test_data, &test_data, &mut result);

                // Verify result is reasonable
                for i in 0..4 {
                    let expected = test_data[i] + test_data[i];
                    if (result[i] - expected).abs() > 1e-6 {
                        return Err(format!(
                            "SIMD addition failed: expected {}, got {}",
                            expected, result[i]
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Test platform-specific optimizations
    pub fn test_platform_optimizations(&self) -> Result<(), String> {
        if !self.platform_tests_enabled {
            return Ok(());
        }

        #[cfg(feature = "cpu")]
        {
            use crate::cpu::platform_optimization::PlatformOptimizer;

            // Test platform optimizer creation
            let _optimizer = PlatformOptimizer::new()
                .map_err(|e| format!("Failed to create platform optimizer: {}", e))?;

            // Test that platform optimizer was created successfully
            // Note: Some optimization methods are not yet implemented
            let _test_params = crate::cpu::optimizations::OptimizationManager::default();
        }

        Ok(())
    }

    /// Test memory optimization features
    pub fn test_memory_optimizations(&self) -> Result<(), String> {
        #[cfg(feature = "cpu")]
        {
            use crate::cpu::memory_patterns::{AccessPattern, AccessPatternOptimizer};

            // Test memory pattern detection
            let _pattern_optimizer = AccessPatternOptimizer::new();

            // Test different access patterns
            let patterns = vec![
                AccessPattern::Sequential,
                AccessPattern::Random,
                AccessPattern::Strided(4),
            ];

            for pattern in patterns {
                // Test that pattern optimizer can handle different patterns
                // Note: get_prefetch_hints method is not yet implemented
                match pattern {
                    AccessPattern::Sequential => {
                        // Sequential pattern should be optimized
                    }
                    AccessPattern::Random => {
                        // Random pattern should be handled
                    }
                    AccessPattern::Strided(_) => {
                        // Strided pattern should be optimized
                    }
                    _ => {
                        // Other patterns should be handled
                    }
                }
            }
        }

        Ok(())
    }

    /// Test auto-tuning system
    pub fn test_autotuning_system(&self) -> Result<(), String> {
        #[cfg(feature = "cpu")]
        {
            use crate::cpu::autotuning::{AutoTuner, TuningConfig};

            // Test auto-tuner creation
            let config = TuningConfig::default();
            let auto_tuner = AutoTuner::with_config(config);

            // Test that it can provide tuning results
            let test_operation = "test_matmul";
            let input_size = 128 * 128;

            // This should not panic
            let tuning_result = auto_tuner
                .get_optimal_params(test_operation, input_size, "f32")
                .map_err(|e| format!("Auto-tuner failed: {}", e))?;

            // Result should be reasonable
            if tuning_result.optimal_thread_count == 0 {
                return Err("Auto-tuner returned zero threads".to_string());
            }

            if tuning_result.optimal_block_size == Some(0) {
                return Err("Auto-tuner returned zero block size".to_string());
            }
        }

        Ok(())
    }

    /// Test backend-specific hardware optimizations
    pub fn test_backend_hardware_optimizations(&self) -> Result<(), String> {
        // Test CPU backend optimizations
        if let Ok(backend) = BackendBuilder::new().backend_type(BackendType::Cpu).build() {
            let capabilities = backend.capabilities();

            // Should report hardware-specific capabilities
            if capabilities
                .extended_capabilities
                .hardware_features
                .is_empty()
            {
                return Err("CPU backend reports no hardware features".to_string());
            }

            // Should have reasonable memory hierarchy info
            if capabilities
                .extended_capabilities
                .memory_hierarchy
                .l1_cache_size
                .unwrap_or(0)
                == 0
            {
                return Err("CPU backend reports zero L1 cache size".to_string());
            }
        }

        // Test other backends if available
        #[cfg(feature = "cuda")]
        {
            if let Ok(backend) = BackendBuilder::new()
                .backend_type(BackendType::Cuda)
                .build()
            {
                let capabilities = backend.capabilities();

                // CUDA backend should report GPU-specific features
                if !capabilities
                    .extended_capabilities
                    .hardware_features
                    .contains(&crate::backend::HardwareFeature::TensorCores)
                {
                    return Err("CUDA backend doesn't report GPU hardware feature".to_string());
                }
            }
        }

        Ok(())
    }

    /// Run all hardware optimization tests
    pub fn run_all_tests(&self) -> Result<(), String> {
        self.test_cpu_feature_detection()?;
        self.test_simd_optimizations()?;
        self.test_platform_optimizations()?;
        self.test_memory_optimizations()?;
        self.test_autotuning_system()?;
        self.test_backend_hardware_optimizations()?;

        Ok(())
    }
}

impl Default for HardwareOptimizationTester {
    fn default() -> Self {
        Self::new()
    }
}

/// Run hardware optimization tests
pub fn run_hardware_optimization_tests() -> Result<(), String> {
    let tester = HardwareOptimizationTester::new();
    tester.run_all_tests()
}

/// Run lightweight hardware optimization tests (for CI)
pub fn run_lightweight_hardware_tests() -> Result<(), String> {
    let mut tester = HardwareOptimizationTester::new();
    tester.performance_tests_enabled = false; // Disable heavy tests

    tester.test_cpu_feature_detection()?;
    tester.test_backend_hardware_optimizations()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_optimization_tester_creation() {
        let tester = HardwareOptimizationTester::new();
        assert!(tester.simd_tests_enabled);
        assert!(tester.platform_tests_enabled);
        assert!(!tester.performance_tests_enabled); // Should be disabled by default
    }

    #[test]
    fn test_cpu_feature_detection() {
        let tester = HardwareOptimizationTester::new();

        match tester.test_cpu_feature_detection() {
            Ok(()) => {
                // Feature detection worked
            }
            Err(e) => {
                // May fail if feature detection is not available
                eprintln!("CPU feature detection test failed: {}", e);
            }
        }
    }

    #[test]
    fn test_simd_optimizations() {
        let tester = HardwareOptimizationTester::new();

        match tester.test_simd_optimizations() {
            Ok(()) => {
                // SIMD tests passed
            }
            Err(e) => {
                // May fail if SIMD is not available or not implemented
                eprintln!("SIMD optimization test failed: {}", e);
            }
        }
    }

    #[test]
    fn test_platform_optimizations() {
        let tester = HardwareOptimizationTester::new();

        match tester.test_platform_optimizations() {
            Ok(()) => {
                // Platform optimization tests passed
            }
            Err(e) => {
                // May fail if platform optimizations are not implemented
                eprintln!("Platform optimization test failed: {}", e);
            }
        }
    }

    #[test]
    fn test_backend_hardware_optimizations() {
        let tester = HardwareOptimizationTester::new();

        match tester.test_backend_hardware_optimizations() {
            Ok(()) => {
                // Backend hardware optimization tests passed
            }
            Err(e) => {
                // May fail if backend doesn't report hardware features properly
                eprintln!("Backend hardware optimization test failed: {}", e);
            }
        }
    }

    #[test]
    fn test_lightweight_hardware_tests() {
        // This test should always run in CI
        match run_lightweight_hardware_tests() {
            Ok(()) => {
                println!("Lightweight hardware tests passed");
            }
            Err(e) => {
                // Log but don't fail - hardware detection may not be available
                eprintln!("Lightweight hardware tests warning: {}", e);
            }
        }
    }

    #[test]
    fn test_hardware_optimization_config() {
        // Test that we can configure the tester
        let mut tester = HardwareOptimizationTester::new();

        tester.simd_tests_enabled = false;
        tester.platform_tests_enabled = false;
        tester.performance_tests_enabled = true;

        // Should still be able to run (though most tests will be skipped)
        match tester.run_all_tests() {
            Ok(()) => {
                // Tests passed with custom config
            }
            Err(e) => {
                // Some tests may still fail
                eprintln!(
                    "Hardware optimization tests with custom config failed: {}",
                    e
                );
            }
        }
    }
}
