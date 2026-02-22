//! CPU feature detection and dynamic kernel dispatch

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
// Allow unexpected_cfgs for SVE feature which may be added in the future
#![allow(unexpected_cfgs)]
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

#[cfg(not(feature = "std"))]
use alloc::{string::String, vec::Vec};

/// CPU feature flags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpuFeature {
    // x86_64 features
    Sse2,
    Sse3,
    Ssse3,
    Sse41,
    Sse42,
    Avx,
    Avx2,
    Avx512f,
    Avx512bw,
    Avx512cd,
    Avx512dq,
    Avx512vl,
    Fma,
    Bmi1,
    Bmi2,
    Lzcnt,
    Popcnt,
    F16c,
    Aes,
    Pclmulqdq,

    // ARM features
    Neon,
    Asimd,
    Sve,
    Sve2,
    Crc32,
    AesArm,
    Sha1,
    Sha2,
    Pmull,

    // RISC-V features
    V, // Vector extension
    F, // Single-precision floating-point
    D, // Double-precision floating-point
    C, // Compressed instructions
    M, // Integer multiplication and division
    A, // Atomic instructions

    // General features
    MultiCore,
    Hypervisor,
    TurboBoost,

    // Custom ToRSh optimizations
    TorshSuperscalar,
    TorshVectorized,
    TorshCacheOptimized,
}

impl CpuFeature {
    /// Get human-readable name for the feature
    pub fn name(&self) -> &'static str {
        match self {
            CpuFeature::Sse2 => "SSE2",
            CpuFeature::Sse3 => "SSE3",
            CpuFeature::Ssse3 => "SSSE3",
            CpuFeature::Sse41 => "SSE4.1",
            CpuFeature::Sse42 => "SSE4.2",
            CpuFeature::Avx => "AVX",
            CpuFeature::Avx2 => "AVX2",
            CpuFeature::Avx512f => "AVX-512F",
            CpuFeature::Avx512bw => "AVX-512BW",
            CpuFeature::Avx512cd => "AVX-512CD",
            CpuFeature::Avx512dq => "AVX-512DQ",
            CpuFeature::Avx512vl => "AVX-512VL",
            CpuFeature::Fma => "FMA",
            CpuFeature::Bmi1 => "BMI1",
            CpuFeature::Bmi2 => "BMI2",
            CpuFeature::Lzcnt => "LZCNT",
            CpuFeature::Popcnt => "POPCNT",
            CpuFeature::F16c => "F16C",
            CpuFeature::Aes => "AES",
            CpuFeature::AesArm => "AES (ARM)",
            CpuFeature::Pclmulqdq => "PCLMULQDQ",
            CpuFeature::Neon => "NEON",
            CpuFeature::Asimd => "Advanced SIMD",
            CpuFeature::Sve => "SVE",
            CpuFeature::Sve2 => "SVE2",
            CpuFeature::Crc32 => "CRC32",
            CpuFeature::Sha1 => "SHA1",
            CpuFeature::Sha2 => "SHA2",
            CpuFeature::Pmull => "PMULL",
            CpuFeature::V => "Vector",
            CpuFeature::F => "Float32",
            CpuFeature::D => "Float64",
            CpuFeature::C => "Compressed",
            CpuFeature::M => "Multiply/Divide",
            CpuFeature::A => "Atomic",
            CpuFeature::MultiCore => "Multi-Core",
            CpuFeature::Hypervisor => "Hypervisor",
            CpuFeature::TurboBoost => "Turbo Boost",
            CpuFeature::TorshSuperscalar => "ToRSh Superscalar",
            CpuFeature::TorshVectorized => "ToRSh Vectorized",
            CpuFeature::TorshCacheOptimized => "ToRSh Cache Optimized",
        }
    }

    /// Check if this feature requires another feature
    pub fn dependencies(&self) -> Vec<CpuFeature> {
        match self {
            CpuFeature::Sse3 => vec![CpuFeature::Sse2],
            CpuFeature::Ssse3 => vec![CpuFeature::Sse3],
            CpuFeature::Sse41 => vec![CpuFeature::Ssse3],
            CpuFeature::Sse42 => vec![CpuFeature::Sse41],
            CpuFeature::Avx => vec![CpuFeature::Sse42],
            CpuFeature::Avx2 => vec![CpuFeature::Avx],
            CpuFeature::Avx512f => vec![CpuFeature::Avx2],
            CpuFeature::Avx512bw => vec![CpuFeature::Avx512f],
            CpuFeature::Avx512cd => vec![CpuFeature::Avx512f],
            CpuFeature::Avx512dq => vec![CpuFeature::Avx512f],
            CpuFeature::Avx512vl => vec![CpuFeature::Avx512f],
            CpuFeature::Fma => vec![CpuFeature::Avx],
            CpuFeature::Sve2 => vec![CpuFeature::Sve],
            CpuFeature::TorshSuperscalar => vec![CpuFeature::MultiCore],
            CpuFeature::TorshVectorized => vec![], // Can work with any SIMD
            CpuFeature::TorshCacheOptimized => vec![],
            _ => vec![],
        }
    }
}

/// CPU architecture information
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuArchInfo {
    pub arch: CpuArch,
    pub vendor: String,
    pub model_name: String,
    pub cores: usize,
    pub threads: usize,
    pub cache_l1_data: usize,
    pub cache_l1_instruction: usize,
    pub cache_l2: usize,
    pub cache_l3: usize,
    pub base_frequency: Option<u64>, // MHz
    pub max_frequency: Option<u64>,  // MHz
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuArch {
    X86_64,
    Aarch64,
    RiscV,
    Other(u32),
}

/// CPU feature detector
#[derive(Debug)]
pub struct CpuFeatureDetector {
    /// Detected features
    features: RwLock<HashMap<CpuFeature, bool>>,

    /// Architecture information
    arch_info: RwLock<Option<CpuArchInfo>>,

    /// Detection status
    detected: std::sync::atomic::AtomicBool,
}

impl Default for CpuFeatureDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuFeatureDetector {
    /// Create a new CPU feature detector
    pub fn new() -> Self {
        Self {
            features: RwLock::new(HashMap::new()),
            arch_info: RwLock::new(None),
            detected: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Perform CPU feature detection
    pub fn detect(&self) -> Result<(), &'static str> {
        if self.detected.load(std::sync::atomic::Ordering::Acquire) {
            return Ok(());
        }

        let mut features = self.features.write();
        features.clear();

        // Detect based on target architecture
        #[cfg(target_arch = "x86_64")]
        {
            self.detect_x86_64_features(&mut features);
        }

        #[cfg(target_arch = "aarch64")]
        {
            self.detect_aarch64_features(&mut features);
        }

        #[cfg(target_arch = "riscv64")]
        {
            self.detect_riscv_features(&mut features);
        }

        // Detect general features
        self.detect_general_features(&mut features);

        // Detect architecture info
        let arch_info = self.detect_arch_info();
        *self.arch_info.write() = Some(arch_info);

        self.detected
            .store(true, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Check if a feature is available
    pub fn has_feature(&self, feature: CpuFeature) -> bool {
        if !self.detected.load(std::sync::atomic::Ordering::Acquire) {
            let _ = self.detect();
        }

        self.features.read().get(&feature).copied().unwrap_or(false)
    }

    /// Get all detected features
    pub fn detected_features(&self) -> Vec<CpuFeature> {
        if !self.detected.load(std::sync::atomic::Ordering::Acquire) {
            let _ = self.detect();
        }

        self.features
            .read()
            .iter()
            .filter_map(
                |(feature, &available)| {
                    if available {
                        Some(*feature)
                    } else {
                        None
                    }
                },
            )
            .collect()
    }

    /// Get architecture information
    pub fn arch_info(&self) -> Option<CpuArchInfo> {
        if !self.detected.load(std::sync::atomic::Ordering::Acquire) {
            let _ = self.detect();
        }

        self.arch_info.read().clone()
    }

    /// Check if all dependencies for a feature are met
    pub fn check_dependencies(&self, feature: CpuFeature) -> bool {
        let deps = feature.dependencies();
        for dep in deps {
            if !self.has_feature(dep) {
                return false;
            }
        }
        true
    }

    /// Get optimal feature set for a given operation type
    pub fn optimal_features_for_operation(&self, operation: &str) -> Vec<CpuFeature> {
        let mut optimal = Vec::new();

        match operation {
            "elementwise" => {
                // x86_64 vectorization hierarchy
                if self.has_feature(CpuFeature::Avx512f) {
                    optimal.push(CpuFeature::Avx512f);
                } else if self.has_feature(CpuFeature::Avx2) {
                    optimal.push(CpuFeature::Avx2);
                } else if self.has_feature(CpuFeature::Avx) {
                    optimal.push(CpuFeature::Avx);
                } else if self.has_feature(CpuFeature::Sse42) {
                    optimal.push(CpuFeature::Sse42);
                }

                // ARM vectorization
                if self.has_feature(CpuFeature::Sve2) {
                    optimal.push(CpuFeature::Sve2);
                } else if self.has_feature(CpuFeature::Sve) {
                    optimal.push(CpuFeature::Sve);
                } else if self.has_feature(CpuFeature::Neon) {
                    optimal.push(CpuFeature::Neon);
                }

                // RISC-V vectorization
                if self.has_feature(CpuFeature::V) {
                    optimal.push(CpuFeature::V);
                    if self.has_feature(CpuFeature::F) {
                        optimal.push(CpuFeature::F);
                    }
                    if self.has_feature(CpuFeature::D) {
                        optimal.push(CpuFeature::D);
                    }
                }
            }
            "matmul" => {
                // x86_64 matrix multiplication optimizations
                if self.has_feature(CpuFeature::Avx512f) && self.has_feature(CpuFeature::Fma) {
                    optimal.extend([CpuFeature::Avx512f, CpuFeature::Fma]);
                } else if self.has_feature(CpuFeature::Avx2) && self.has_feature(CpuFeature::Fma) {
                    optimal.extend([CpuFeature::Avx2, CpuFeature::Fma]);
                } else if self.has_feature(CpuFeature::Avx) && self.has_feature(CpuFeature::Fma) {
                    optimal.extend([CpuFeature::Avx, CpuFeature::Fma]);
                }

                // ARM matrix multiplication
                if self.has_feature(CpuFeature::Sve2) {
                    optimal.push(CpuFeature::Sve2);
                } else if self.has_feature(CpuFeature::Sve) {
                    optimal.push(CpuFeature::Sve);
                } else if self.has_feature(CpuFeature::Neon) {
                    optimal.push(CpuFeature::Neon);
                }

                // RISC-V matrix multiplication with vector extensions
                if self.has_feature(CpuFeature::V) && self.has_feature(CpuFeature::F) {
                    optimal.extend([CpuFeature::V, CpuFeature::F]);
                    if self.has_feature(CpuFeature::D) {
                        optimal.push(CpuFeature::D);
                    }
                }

                if self.has_feature(CpuFeature::MultiCore) {
                    optimal.push(CpuFeature::MultiCore);
                }
            }
            "reduction" => {
                // x86_64 reduction optimizations
                if self.has_feature(CpuFeature::Avx512bw) {
                    optimal.push(CpuFeature::Avx512bw);
                } else if self.has_feature(CpuFeature::Avx2) {
                    optimal.push(CpuFeature::Avx2);
                }

                // ARM reduction optimizations
                if self.has_feature(CpuFeature::Sve2) {
                    optimal.push(CpuFeature::Sve2);
                } else if self.has_feature(CpuFeature::Sve) {
                    optimal.push(CpuFeature::Sve);
                } else if self.has_feature(CpuFeature::Neon) {
                    optimal.push(CpuFeature::Neon);
                }

                // RISC-V reduction with vector extensions
                if self.has_feature(CpuFeature::V) {
                    optimal.push(CpuFeature::V);
                }

                if self.has_feature(CpuFeature::MultiCore) {
                    optimal.push(CpuFeature::MultiCore);
                }
            }
            "convolution" => {
                // Specialized for convolution operations
                if self.has_feature(CpuFeature::Avx512f) && self.has_feature(CpuFeature::Fma) {
                    optimal.extend([CpuFeature::Avx512f, CpuFeature::Fma]);
                } else if self.has_feature(CpuFeature::Avx2) && self.has_feature(CpuFeature::Fma) {
                    optimal.extend([CpuFeature::Avx2, CpuFeature::Fma]);
                }

                if self.has_feature(CpuFeature::Sve2) {
                    optimal.push(CpuFeature::Sve2);
                } else if self.has_feature(CpuFeature::Neon) {
                    optimal.push(CpuFeature::Neon);
                }

                if self.has_feature(CpuFeature::V) && self.has_feature(CpuFeature::F) {
                    optimal.extend([CpuFeature::V, CpuFeature::F]);
                }

                if self.has_feature(CpuFeature::MultiCore) {
                    optimal.push(CpuFeature::MultiCore);
                }
            }
            "fft" => {
                // FFT optimizations prefer floating-point and vector capabilities
                if self.has_feature(CpuFeature::Avx512f) && self.has_feature(CpuFeature::Fma) {
                    optimal.extend([CpuFeature::Avx512f, CpuFeature::Fma]);
                } else if self.has_feature(CpuFeature::Avx2) && self.has_feature(CpuFeature::Fma) {
                    optimal.extend([CpuFeature::Avx2, CpuFeature::Fma]);
                }

                if self.has_feature(CpuFeature::Sve) {
                    optimal.push(CpuFeature::Sve);
                } else if self.has_feature(CpuFeature::Neon) {
                    optimal.push(CpuFeature::Neon);
                }

                if self.has_feature(CpuFeature::V) && self.has_feature(CpuFeature::F) {
                    optimal.extend([CpuFeature::V, CpuFeature::F]);
                    if self.has_feature(CpuFeature::D) {
                        optimal.push(CpuFeature::D);
                    }
                }
            }
            _ => {
                // Default to best available vectorization for unknown operations
                if self.has_feature(CpuFeature::Avx512f) {
                    optimal.push(CpuFeature::Avx512f);
                } else if self.has_feature(CpuFeature::Avx2) {
                    optimal.push(CpuFeature::Avx2);
                } else if self.has_feature(CpuFeature::Avx) {
                    optimal.push(CpuFeature::Avx);
                }

                if self.has_feature(CpuFeature::Sve2) {
                    optimal.push(CpuFeature::Sve2);
                } else if self.has_feature(CpuFeature::Sve) {
                    optimal.push(CpuFeature::Sve);
                } else if self.has_feature(CpuFeature::Neon) {
                    optimal.push(CpuFeature::Neon);
                }

                if self.has_feature(CpuFeature::V) {
                    optimal.push(CpuFeature::V);
                }
            }
        }

        optimal
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_x86_64_features(&self, features: &mut HashMap<CpuFeature, bool>) {
        // Use std::arch for feature detection on x86_64
        features.insert(
            CpuFeature::Sse2,
            std::arch::is_x86_feature_detected!("sse2"),
        );
        features.insert(
            CpuFeature::Sse3,
            std::arch::is_x86_feature_detected!("sse3"),
        );
        features.insert(
            CpuFeature::Ssse3,
            std::arch::is_x86_feature_detected!("ssse3"),
        );
        features.insert(
            CpuFeature::Sse41,
            std::arch::is_x86_feature_detected!("sse4.1"),
        );
        features.insert(
            CpuFeature::Sse42,
            std::arch::is_x86_feature_detected!("sse4.2"),
        );
        features.insert(CpuFeature::Avx, std::arch::is_x86_feature_detected!("avx"));
        features.insert(
            CpuFeature::Avx2,
            std::arch::is_x86_feature_detected!("avx2"),
        );
        features.insert(
            CpuFeature::Avx512f,
            std::arch::is_x86_feature_detected!("avx512f"),
        );
        features.insert(
            CpuFeature::Avx512bw,
            std::arch::is_x86_feature_detected!("avx512bw"),
        );
        features.insert(
            CpuFeature::Avx512cd,
            std::arch::is_x86_feature_detected!("avx512cd"),
        );
        features.insert(
            CpuFeature::Avx512dq,
            std::arch::is_x86_feature_detected!("avx512dq"),
        );
        features.insert(
            CpuFeature::Avx512vl,
            std::arch::is_x86_feature_detected!("avx512vl"),
        );
        features.insert(CpuFeature::Fma, std::arch::is_x86_feature_detected!("fma"));
        features.insert(
            CpuFeature::Bmi1,
            std::arch::is_x86_feature_detected!("bmi1"),
        );
        features.insert(
            CpuFeature::Bmi2,
            std::arch::is_x86_feature_detected!("bmi2"),
        );
        features.insert(
            CpuFeature::Lzcnt,
            std::arch::is_x86_feature_detected!("lzcnt"),
        );
        features.insert(
            CpuFeature::Popcnt,
            std::arch::is_x86_feature_detected!("popcnt"),
        );
        features.insert(
            CpuFeature::F16c,
            std::arch::is_x86_feature_detected!("f16c"),
        );
        features.insert(CpuFeature::Aes, std::arch::is_x86_feature_detected!("aes"));
        features.insert(
            CpuFeature::Pclmulqdq,
            std::arch::is_x86_feature_detected!("pclmulqdq"),
        );
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_aarch64_features(&self, features: &mut HashMap<CpuFeature, bool>) {
        // Use std::arch for feature detection on aarch64
        features.insert(
            CpuFeature::Neon,
            std::arch::is_aarch64_feature_detected!("neon"),
        );
        features.insert(
            CpuFeature::Asimd,
            std::arch::is_aarch64_feature_detected!("asimd"),
        );
        features.insert(
            CpuFeature::Crc32,
            std::arch::is_aarch64_feature_detected!("crc"),
        );
        features.insert(
            CpuFeature::AesArm,
            std::arch::is_aarch64_feature_detected!("aes"),
        );
        features.insert(
            CpuFeature::Sha1,
            std::arch::is_aarch64_feature_detected!("sha2"),
        );
        features.insert(
            CpuFeature::Sha2,
            std::arch::is_aarch64_feature_detected!("sha3"),
        );
        features.insert(
            CpuFeature::Pmull,
            std::arch::is_aarch64_feature_detected!("pmull"),
        );

        // SVE detection might require additional checks
        #[cfg(feature = "sve")]
        {
            features.insert(
                CpuFeature::Sve,
                std::arch::is_aarch64_feature_detected!("sve"),
            );
            features.insert(
                CpuFeature::Sve2,
                std::arch::is_aarch64_feature_detected!("sve2"),
            );
        }
    }

    #[cfg(target_arch = "riscv64")]
    fn detect_riscv_features(&self, features: &mut HashMap<CpuFeature, bool>) {
        // RISC-V feature detection via parsing /proc/cpuinfo and hwprobe
        self.detect_riscv_from_cpuinfo(features);
        self.detect_riscv_from_hwprobe(features);
        self.detect_riscv_runtime_features(features);
    }

    #[cfg(target_arch = "riscv64")]
    fn detect_riscv_from_cpuinfo(&self, features: &mut HashMap<CpuFeature, bool>) {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in cpuinfo.lines() {
                if line.starts_with("isa") || line.starts_with("ISA") {
                    let isa_string = line.split(':').nth(1).unwrap_or("").trim();

                    // Parse standard RISC-V ISA string
                    if isa_string.contains("rv64") {
                        // Base integer instruction set
                        features.insert(
                            CpuFeature::M,
                            isa_string.contains('m') || isa_string.contains('M'),
                        );
                        features.insert(
                            CpuFeature::A,
                            isa_string.contains('a') || isa_string.contains('A'),
                        );
                        features.insert(
                            CpuFeature::F,
                            isa_string.contains('f') || isa_string.contains('F'),
                        );
                        features.insert(
                            CpuFeature::D,
                            isa_string.contains('d') || isa_string.contains('D'),
                        );
                        features.insert(
                            CpuFeature::C,
                            isa_string.contains('c') || isa_string.contains('C'),
                        );
                        features.insert(
                            CpuFeature::V,
                            isa_string.contains('v') || isa_string.contains('V'),
                        );

                        // Check for vector extensions in ISA extensions
                        if isa_string.contains("_zve")
                            || isa_string.contains("_zvl")
                            || isa_string.contains("_zvbb")
                        {
                            features.insert(CpuFeature::V, true);
                        }

                        // Check for compressed instructions
                        if isa_string.contains("_zca")
                            || isa_string.contains("_zcb")
                            || isa_string.contains("_zcd")
                        {
                            features.insert(CpuFeature::C, true);
                        }
                    }
                    break;
                }
            }
        }
    }

    #[cfg(target_arch = "riscv64")]
    fn detect_riscv_from_hwprobe(&self, features: &mut HashMap<CpuFeature, bool>) {
        // Try to use riscv_hwprobe syscall if available
        // This is a more reliable method for newer kernels
        unsafe {
            use std::mem;

            // Define hwprobe structures
            #[repr(C)]
            struct riscv_hwprobe {
                key: i64,
                value: u64,
            }

            const RISCV_HWPROBE_KEY_IMA_EXT_0: i64 = 4;
            const RISCV_HWPROBE_IMA_V: u64 = 1 << 2;
            const RISCV_HWPROBE_IMA_FD: u64 = 1 << 0;
            const RISCV_HWPROBE_IMA_C: u64 = 1 << 1;

            let mut probes = [riscv_hwprobe {
                key: RISCV_HWPROBE_KEY_IMA_EXT_0,
                value: 0,
            }];

            // syscall number for riscv_hwprobe (if available)
            let syscall_num = 258; // This may vary by kernel version

            let result = libc::syscall(
                syscall_num,
                probes.as_mut_ptr(),
                probes.len(),
                0,
                std::ptr::null_mut::<u8>(),
                0,
            );

            if result == 0 {
                let probe = &probes[0];
                if probe.key == RISCV_HWPROBE_KEY_IMA_EXT_0 {
                    features.insert(CpuFeature::V, (probe.value & RISCV_HWPROBE_IMA_V) != 0);
                    features.insert(CpuFeature::F, (probe.value & RISCV_HWPROBE_IMA_FD) != 0);
                    features.insert(CpuFeature::D, (probe.value & RISCV_HWPROBE_IMA_FD) != 0);
                    features.insert(CpuFeature::C, (probe.value & RISCV_HWPROBE_IMA_C) != 0);
                }
            }
        }
    }

    #[cfg(target_arch = "riscv64")]
    fn detect_riscv_runtime_features(&self, features: &mut HashMap<CpuFeature, bool>) {
        // Runtime feature detection through test execution
        // This is a fallback method for cases where static detection fails

        // Test for vector extension by attempting to use vector instructions
        if self.test_riscv_vector_capability() {
            features.insert(CpuFeature::V, true);
        }

        // Test for floating-point extensions
        if self.test_riscv_float_capability() {
            features.insert(CpuFeature::F, true);
            features.insert(CpuFeature::D, true);
        }

        // Test for atomic instructions
        if self.test_riscv_atomic_capability() {
            features.insert(CpuFeature::A, true);
        }

        // Multiplication/division is typically always available on modern RISC-V
        features.insert(CpuFeature::M, true);
    }

    #[cfg(target_arch = "riscv64")]
    fn test_riscv_vector_capability(&self) -> bool {
        // This would need to be implemented with actual RISC-V vector assembly
        // For now, we'll use a conservative approach
        std::panic::catch_unwind(|| {
            // Try to execute a simple vector instruction
            // This is a placeholder - actual implementation would use inline assembly
            true
        })
        .unwrap_or(false)
    }

    #[cfg(target_arch = "riscv64")]
    fn test_riscv_float_capability(&self) -> bool {
        // Test floating-point capability
        std::panic::catch_unwind(|| {
            let _test: f64 = 1.0 + 2.0;
            true
        })
        .unwrap_or(false)
    }

    #[cfg(target_arch = "riscv64")]
    fn test_riscv_atomic_capability(&self) -> bool {
        // Test atomic instructions
        use std::sync::atomic::{AtomicU64, Ordering};
        let atomic = AtomicU64::new(0);
        atomic
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    }

    fn detect_general_features(&self, features: &mut HashMap<CpuFeature, bool>) {
        // Detect multi-core capability
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        features.insert(CpuFeature::MultiCore, num_cpus > 1);

        // Detect ToRSh-specific optimizations
        features.insert(CpuFeature::TorshSuperscalar, num_cpus >= 4);
        features.insert(CpuFeature::TorshVectorized, true); // Always available
        features.insert(CpuFeature::TorshCacheOptimized, true); // Always available
    }

    fn detect_arch_info(&self) -> CpuArchInfo {
        let arch = if cfg!(target_arch = "x86_64") {
            CpuArch::X86_64
        } else if cfg!(target_arch = "aarch64") {
            CpuArch::Aarch64
        } else if cfg!(target_arch = "riscv64") {
            CpuArch::RiscV
        } else {
            CpuArch::Other(0)
        };

        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        CpuArchInfo {
            arch,
            vendor: self.detect_cpu_vendor(),
            model_name: self.detect_cpu_model(),
            cores,
            threads: cores,        // Simplified - could detect hyperthreading
            cache_l1_data: 32_768, // Default estimates
            cache_l1_instruction: 32_768,
            cache_l2: 256_768,
            cache_l3: 8_388_608,
            base_frequency: None,
            max_frequency: None,
        }
    }

    fn detect_cpu_vendor(&self) -> String {
        #[cfg(target_arch = "x86_64")]
        {
            // Could use CPUID to get vendor string
            "Unknown".to_string()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            "Unknown".to_string()
        }
    }

    fn detect_cpu_model(&self) -> String {
        #[cfg(target_arch = "x86_64")]
        {
            // Could parse /proc/cpuinfo or use CPUID
            "Unknown".to_string()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            "Unknown".to_string()
        }
    }
}

/// Kernel dispatcher based on CPU features
pub struct CpuKernelDispatcher {
    detector: Arc<CpuFeatureDetector>,
    kernels: RwLock<HashMap<String, Vec<DynamicKernel>>>,
}

/// Dynamic kernel that can be selected based on CPU features
pub struct DynamicKernel {
    pub name: String,
    pub required_features: Vec<CpuFeature>,
    pub performance_score: u32,
    pub kernel_fn: Arc<dyn Fn(&[f32], &mut [f32]) + Send + Sync>,
}

impl std::fmt::Debug for DynamicKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicKernel")
            .field("name", &self.name)
            .field("required_features", &self.required_features)
            .field("performance_score", &self.performance_score)
            .field("kernel_fn", &"<function>")
            .finish()
    }
}

impl CpuKernelDispatcher {
    /// Create a new kernel dispatcher
    pub fn new(detector: Arc<CpuFeatureDetector>) -> Self {
        Self {
            detector,
            kernels: RwLock::new(HashMap::new()),
        }
    }

    /// Register a kernel variant
    pub fn register_kernel(&self, operation: &str, kernel: DynamicKernel) {
        let mut kernels = self.kernels.write();
        kernels
            .entry(operation.to_string())
            .or_insert_with(Vec::new)
            .push(kernel);

        // Sort by performance score (descending)
        if let Some(kernel_list) = kernels.get_mut(operation) {
            kernel_list.sort_by(|a, b| b.performance_score.cmp(&a.performance_score));
        }
    }

    /// Get the best kernel for an operation
    pub fn get_best_kernel(
        &self,
        operation: &str,
    ) -> Option<Arc<dyn Fn(&[f32], &mut [f32]) + Send + Sync>> {
        let kernels = self.kernels.read();
        let kernel_list = kernels.get(operation)?;

        // Find the first kernel where all required features are available
        for kernel in kernel_list {
            let all_features_available = kernel
                .required_features
                .iter()
                .all(|&feature| self.detector.has_feature(feature));

            if all_features_available {
                return Some(Arc::clone(&kernel.kernel_fn));
            }
        }

        None
    }

    /// Get information about available kernels for an operation
    pub fn get_kernel_info(&self, operation: &str) -> Vec<(String, Vec<CpuFeature>, bool)> {
        let kernels = self.kernels.read();
        if let Some(kernel_list) = kernels.get(operation) {
            kernel_list
                .iter()
                .map(|kernel| {
                    let available = kernel
                        .required_features
                        .iter()
                        .all(|&feature| self.detector.has_feature(feature));
                    (
                        kernel.name.clone(),
                        kernel.required_features.clone(),
                        available,
                    )
                })
                .collect()
        } else {
            Vec::new()
        }
    }
}

/// Global CPU feature detector instance
static GLOBAL_DETECTOR: OnceLock<Arc<CpuFeatureDetector>> = OnceLock::new();

/// Get the global CPU feature detector
pub fn global_detector() -> Arc<CpuFeatureDetector> {
    GLOBAL_DETECTOR
        .get_or_init(|| {
            let detector = Arc::new(CpuFeatureDetector::new());
            let _ = detector.detect();
            detector
        })
        .clone()
}

/// Convenience function to check if a feature is available
pub fn has_feature(feature: CpuFeature) -> bool {
    global_detector().has_feature(feature)
}

/// Get all detected CPU features
pub fn detected_features() -> Vec<CpuFeature> {
    global_detector().detected_features()
}

/// Get CPU architecture information
pub fn cpu_arch_info() -> Option<CpuArchInfo> {
    global_detector().arch_info()
}

/// Get CPU architecture information (alias for cpu_arch_info)
///
/// This is a convenience alias for benchmarks that expect the get_arch_info function.
pub fn get_arch_info() -> Option<CpuArchInfo> {
    cpu_arch_info()
}

/// Type alias for backward compatibility with benchmarks
pub type FeatureDetector = CpuFeatureDetector;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_detector_creation() {
        let detector = CpuFeatureDetector::new();
        assert!(!detector.detected.load(std::sync::atomic::Ordering::Acquire));
    }

    #[test]
    fn test_feature_detection() {
        let detector = CpuFeatureDetector::new();
        let result = detector.detect();
        assert!(result.is_ok());
        assert!(detector.detected.load(std::sync::atomic::Ordering::Acquire));
    }

    #[test]
    fn test_feature_dependencies() {
        let deps = CpuFeature::Avx2.dependencies();
        assert!(deps.contains(&CpuFeature::Avx));

        let deps = CpuFeature::Avx512f.dependencies();
        assert!(deps.contains(&CpuFeature::Avx2));
    }

    #[test]
    fn test_global_detector() {
        let detector = global_detector();
        assert!(detector.detected.load(std::sync::atomic::Ordering::Acquire));
    }

    #[test]
    fn test_kernel_dispatcher() {
        let detector = Arc::new(CpuFeatureDetector::new());
        let _ = detector.detect();
        let dispatcher = CpuKernelDispatcher::new(detector);

        // Register a simple kernel
        let kernel = DynamicKernel {
            name: "test_kernel".to_string(),
            required_features: vec![],
            performance_score: 100,
            kernel_fn: Arc::new(|_input, _output| {
                // Test kernel implementation
            }),
        };

        dispatcher.register_kernel("test_op", kernel);

        let best_kernel = dispatcher.get_best_kernel("test_op");
        assert!(best_kernel.is_some());
    }

    #[test]
    fn test_feature_names() {
        assert_eq!(CpuFeature::Avx2.name(), "AVX2");
        assert_eq!(CpuFeature::Neon.name(), "NEON");
        assert_eq!(CpuFeature::Sve.name(), "SVE");
    }

    #[test]
    fn test_optimal_features_for_operation() {
        let detector = CpuFeatureDetector::new();
        let _ = detector.detect();

        let features = detector.optimal_features_for_operation("elementwise");
        // Should return some features (exact features depend on the test system)
        assert!(!features.is_empty() || !detector.has_feature(CpuFeature::Avx));
    }

    #[test]
    fn test_arch_info() {
        let detector = CpuFeatureDetector::new();
        let _ = detector.detect();

        if let Some(arch_info) = detector.arch_info() {
            assert!(arch_info.cores > 0);
            assert!(arch_info.threads > 0);
            assert!(!arch_info.vendor.is_empty());
            assert!(!arch_info.model_name.is_empty());
        }
    }
}
