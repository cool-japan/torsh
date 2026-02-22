//! Core CPU detection and information gathering
//!
//! This module provides the main CpuInfo struct and comprehensive CPU detection
//! logic for both x86_64 and ARM64 architectures.

use super::{
    cache::CacheInfo,
    features::CpuFeatures,
    microarchitecture::{ArmMicroarchitecture, X86Microarchitecture},
    optimization::MicroarchOptimization,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::sync::OnceLock;

/// Global CPU detection results
static CPU_INFO: OnceLock<CpuInfo> = OnceLock::new();

/// Complete CPU information
#[derive(Debug, Clone)]
pub struct CpuInfo {
    /// Detected features
    pub features: CpuFeatures,
    /// Cache hierarchy
    pub cache: CacheInfo,
    /// Microarchitecture type (x86_64)
    pub x86_microarch: Option<X86Microarchitecture>,
    /// Microarchitecture type (ARM64)
    pub arm_microarch: Option<ArmMicroarchitecture>,
    /// Optimization parameters
    pub optimization: MicroarchOptimization,
    /// CPU vendor
    pub vendor: String,
    /// CPU model name
    pub model_name: String,
    /// Number of physical cores
    pub physical_cores: usize,
    /// Number of logical cores (including HT)
    pub logical_cores: usize,
    /// Base frequency in MHz
    pub base_frequency: f64,
    /// Maximum turbo frequency in MHz
    pub max_frequency: f64,
}

impl CpuInfo {
    /// Get global CPU information (lazy initialization)
    pub fn get() -> &'static CpuInfo {
        CPU_INFO.get_or_init(Self::detect)
    }

    /// Detect CPU information at runtime
    fn detect() -> Self {
        let mut info = Self::default();

        #[cfg(target_arch = "x86_64")]
        {
            info.detect_x86_features();
            info.detect_x86_microarchitecture();
            info.detect_x86_cache_info();
        }

        #[cfg(target_arch = "aarch64")]
        {
            info.detect_arm_features();
            info.detect_arm_microarchitecture();
            info.detect_arm_cache_info();
        }

        info.detect_topology();
        info.create_optimization_profile();

        info
    }

    /// Detect x86_64 CPU features using CPUID
    #[cfg(target_arch = "x86_64")]
    fn detect_x86_features(&mut self) {
        unsafe {
            // Check for CPUID availability
            if !has_cpuid() {
                return;
            }

            // Get basic CPUID information
            let cpuid_result = __cpuid(1);

            // Feature detection from CPUID.01H:EDX
            self.features.sse = (cpuid_result.edx & (1 << 25)) != 0;
            self.features.sse2 = (cpuid_result.edx & (1 << 26)) != 0;
            self.features.fma = (cpuid_result.ecx & (1 << 12)) != 0;
            self.features.popcnt = (cpuid_result.ecx & (1 << 23)) != 0;
            self.features.aes = (cpuid_result.ecx & (1 << 25)) != 0;
            self.features.avx = (cpuid_result.ecx & (1 << 28)) != 0;
            self.features.f16c = (cpuid_result.ecx & (1 << 29)) != 0;
            self.features.rdrand = (cpuid_result.ecx & (1 << 30)) != 0;

            // Feature detection from CPUID.01H:ECX
            self.features.sse3 = (cpuid_result.ecx & (1 << 0)) != 0;
            self.features.pclmul = (cpuid_result.ecx & (1 << 1)) != 0;
            self.features.ssse3 = (cpuid_result.ecx & (1 << 9)) != 0;
            self.features.sse4_1 = (cpuid_result.ecx & (1 << 19)) != 0;
            self.features.sse4_2 = (cpuid_result.ecx & (1 << 20)) != 0;
            self.features.movbe = (cpuid_result.ecx & (1 << 22)) != 0;
            self.features.xsave = (cpuid_result.ecx & (1 << 26)) != 0;

            // Extended features (CPUID.07H:EBX)
            let extended_result = __cpuid_count(7, 0);
            self.features.avx2 = (extended_result.ebx & (1 << 5)) != 0;
            self.features.bmi1 = (extended_result.ebx & (1 << 3)) != 0;
            self.features.bmi2 = (extended_result.ebx & (1 << 8)) != 0;
            self.features.rtm = (extended_result.ebx & (1 << 11)) != 0;
            self.features.hle = (extended_result.ebx & (1 << 4)) != 0;
            self.features.avx512f = (extended_result.ebx & (1 << 16)) != 0;
            self.features.avx512dq = (extended_result.ebx & (1 << 17)) != 0;
            self.features.rdseed = (extended_result.ebx & (1 << 18)) != 0;
            self.features.adx = (extended_result.ebx & (1 << 19)) != 0;
            self.features.avx512cd = (extended_result.ebx & (1 << 28)) != 0;
            self.features.avx512bw = (extended_result.ebx & (1 << 30)) != 0;
            self.features.avx512vl = (extended_result.ebx & (1 << 31)) != 0;

            // More extended features (CPUID.07H:ECX)
            self.features.prefetchw = (extended_result.ecx & (1 << 0)) != 0;
            self.features.avx512vnni = (extended_result.ecx & (1 << 11)) != 0;
            self.features.avx512bf16 = (extended_result.ecx & (1 << 5)) != 0;
            self.features.sha = (extended_result.ecx & (1 << 29)) != 0;

            // Extended function CPUID.80000001H
            let extended_fn = __cpuid(0x80000001);
            self.features.lzcnt = (extended_fn.ecx & (1 << 5)) != 0;
            self.features.fma4 = (extended_fn.ecx & (1 << 16)) != 0;
            self.features.rdtscp = (extended_fn.edx & (1 << 27)) != 0;

            // Detect vendor
            let vendor_result = __cpuid(0);
            let vendor_bytes = [
                (vendor_result.ebx as u32).to_le_bytes(),
                (vendor_result.edx as u32).to_le_bytes(),
                (vendor_result.ecx as u32).to_le_bytes(),
            ];
            self.vendor = String::from_utf8_lossy(&[
                vendor_bytes[0][0],
                vendor_bytes[0][1],
                vendor_bytes[0][2],
                vendor_bytes[0][3],
                vendor_bytes[1][0],
                vendor_bytes[1][1],
                vendor_bytes[1][2],
                vendor_bytes[1][3],
                vendor_bytes[2][0],
                vendor_bytes[2][1],
                vendor_bytes[2][2],
                vendor_bytes[2][3],
            ])
            .trim_end_matches('\0')
            .to_string();
        }
    }

    /// Detect x86_64 microarchitecture
    #[cfg(target_arch = "x86_64")]
    fn detect_x86_microarchitecture(&mut self) {
        // This is a simplified detection - real implementation would use more CPUID data
        self.x86_microarch = Some(if self.vendor.contains("Intel") {
            if self.features.avx512f {
                if self.features.avx512vnni {
                    X86Microarchitecture::IceLake
                } else {
                    X86Microarchitecture::Skylake
                }
            } else if self.features.avx2 {
                X86Microarchitecture::Haswell
            } else if self.features.avx {
                X86Microarchitecture::SandyBridge
            } else {
                X86Microarchitecture::Nehalem
            }
        } else if self.vendor.contains("AMD") {
            if self.features.avx2 {
                X86Microarchitecture::Zen2
            } else if self.features.avx {
                X86Microarchitecture::Bulldozer
            } else {
                X86Microarchitecture::K10
            }
        } else {
            X86Microarchitecture::Unknown
        });
    }

    /// Detect x86_64 cache information
    #[cfg(target_arch = "x86_64")]
    fn detect_x86_cache_info(&mut self) {
        unsafe {
            if !has_cpuid() {
                return;
            }

            // Use CPUID.04H for cache information
            for level in 0..4 {
                let cache_info = __cpuid_count(4, level);
                let cache_type = cache_info.eax & 0x1F;

                if cache_type == 0 {
                    break; // No more cache levels
                }

                let cache_level = (cache_info.eax >> 5) & 0x7;
                let line_size = ((cache_info.ebx & 0xFFF) + 1) as usize;
                let ways = (((cache_info.ebx >> 22) & 0x3FF) + 1) as usize;
                let sets = (cache_info.ecx + 1) as usize;
                let size = ways * sets * line_size;

                match (cache_level, cache_type) {
                    (1, 1) => {
                        // L1 data cache
                        self.cache.l1d_size = size;
                        self.cache.l1_line_size = line_size;
                        self.cache.l1_associativity = ways;
                    }
                    (1, 2) => {
                        // L1 instruction cache
                        self.cache.l1i_size = size;
                    }
                    (2, 3) => {
                        // L2 unified cache
                        self.cache.l2_size = size;
                        self.cache.l2_line_size = line_size;
                        self.cache.l2_associativity = ways;
                    }
                    (3, 3) => {
                        // L3 unified cache
                        self.cache.l3_size = size;
                        self.cache.l3_line_size = line_size;
                        self.cache.l3_associativity = ways;
                    }
                    _ => {}
                }
            }
        }
    }

    /// Detect ARM64 features (simplified)
    #[cfg(target_arch = "aarch64")]
    fn detect_arm_features(&mut self) {
        // ARM64 feature detection typically uses AT_HWCAP from auxiliary vector
        // For simplicity, we'll assume common features are available
        self.features.neon = true;
        self.features.fp = true;
        self.features.asimd = true;

        // Detect Apple Silicon specific features
        #[cfg(target_os = "macos")]
        {
            self.features.fp = true;
            self.features.asimd = true;
            self.features.crc32 = true;
            self.features.aes_arm = true;
            self.features.sha1 = true;
            self.features.sha256 = true;
        }
    }

    /// Detect ARM64 microarchitecture
    #[cfg(target_arch = "aarch64")]
    fn detect_arm_microarchitecture(&mut self) {
        #[cfg(target_os = "macos")]
        {
            // Enhanced Apple Silicon detection
            self.arm_microarch = Self::detect_apple_silicon_chip();
            self.vendor = "Apple".to_string();

            // Set Apple Silicon specific frequencies
            match self.arm_microarch {
                Some(ArmMicroarchitecture::M1) => {
                    self.base_frequency = 3200.0; // 3.2 GHz
                    self.max_frequency = 3200.0; // No turbo on M1
                }
                Some(ArmMicroarchitecture::M2) => {
                    self.base_frequency = 3500.0; // 3.5 GHz
                    self.max_frequency = 3500.0; // No turbo on M2
                }
                Some(ArmMicroarchitecture::M3) => {
                    self.base_frequency = 4000.0; // 4.0 GHz
                    self.max_frequency = 4000.0; // No turbo on M3
                }
                _ => {
                    self.base_frequency = 3000.0; // Default
                    self.max_frequency = 3000.0;
                }
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            self.arm_microarch = Some(ArmMicroarchitecture::CortexA76); // Generic default
            self.vendor = "ARM".to_string();
        }
    }

    /// Detect specific Apple Silicon chip
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    fn detect_apple_silicon_chip() -> Option<ArmMicroarchitecture> {
        use std::process::Command;

        // Try to detect using sysctl
        if let Ok(output) = Command::new("sysctl")
            .arg("-n")
            .arg("machdep.cpu.brand_string")
            .output()
        {
            let brand_string = String::from_utf8_lossy(&output.stdout);

            if brand_string.contains("M3") {
                return Some(ArmMicroarchitecture::M3);
            } else if brand_string.contains("M2") {
                return Some(ArmMicroarchitecture::M2);
            } else if brand_string.contains("M1") {
                return Some(ArmMicroarchitecture::M1);
            }
        }

        // Try to detect core count as a fallback
        if let Ok(output) = Command::new("sysctl")
            .arg("-n")
            .arg("hw.perflevel0.physicalcpu")
            .output()
        {
            if let Ok(cores_str) = String::from_utf8(output.stdout) {
                if let Ok(cores) = cores_str.trim().parse::<i32>() {
                    // M3 typically has 8-12 performance cores
                    // M2 typically has 8 performance cores
                    // M1 typically has 4-8 performance cores
                    return Some(match cores {
                        12 => ArmMicroarchitecture::M3,
                        8 => ArmMicroarchitecture::M2, // Could be M2 or M1 Pro/Max
                        4 => ArmMicroarchitecture::M1,
                        _ => ArmMicroarchitecture::M2, // Default to M2 for unknown
                    });
                }
            }
        }

        // Default fallback
        Some(ArmMicroarchitecture::M1)
    }

    /// Detect ARM64 cache information
    #[cfg(target_arch = "aarch64")]
    fn detect_arm_cache_info(&mut self) {
        // ARM64 cache info would typically be read from system registers
        // For Apple Silicon, use known values based on chip type
        #[cfg(target_os = "macos")]
        {
            match self.arm_microarch {
                Some(ArmMicroarchitecture::M1) => {
                    self.cache.l1d_size = 128 * 1024; // 128KB per core
                    self.cache.l1i_size = 128 * 1024; // 128KB per core
                    self.cache.l2_size = 12 * 1024 * 1024; // 12MB shared
                    self.cache.l1_line_size = 64;
                    self.cache.l2_line_size = 64;
                    self.cache.l1_associativity = 8;
                    self.cache.l2_associativity = 12;
                }
                Some(ArmMicroarchitecture::M2) => {
                    self.cache.l1d_size = 128 * 1024; // 128KB per core
                    self.cache.l1i_size = 128 * 1024; // 128KB per core
                    self.cache.l2_size = 16 * 1024 * 1024; // 16MB shared
                    self.cache.l1_line_size = 64;
                    self.cache.l2_line_size = 64;
                    self.cache.l1_associativity = 8;
                    self.cache.l2_associativity = 16;
                }
                Some(ArmMicroarchitecture::M3) => {
                    self.cache.l1d_size = 128 * 1024; // 128KB per core
                    self.cache.l1i_size = 128 * 1024; // 128KB per core
                    self.cache.l2_size = 18 * 1024 * 1024; // 18MB shared
                    self.cache.l1_line_size = 64;
                    self.cache.l2_line_size = 64;
                    self.cache.l1_associativity = 8;
                    self.cache.l2_associativity = 18;
                }
                _ => {
                    // Default Apple Silicon values
                    self.cache.l1d_size = 128 * 1024;
                    self.cache.l1i_size = 128 * 1024;
                    self.cache.l2_size = 12 * 1024 * 1024;
                    self.cache.l1_line_size = 64;
                    self.cache.l2_line_size = 64;
                    self.cache.l1_associativity = 8;
                    self.cache.l2_associativity = 12;
                }
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Generic ARM64 values
            self.cache.l1d_size = 64 * 1024; // 64KB
            self.cache.l1i_size = 64 * 1024; // 64KB
            self.cache.l2_size = 1024 * 1024; // 1MB
            self.cache.l1_line_size = 64;
            self.cache.l2_line_size = 64;
            self.cache.l1_associativity = 4;
            self.cache.l2_associativity = 8;
        }
    }

    /// Detect CPU topology (core count, threading)
    fn detect_topology(&mut self) {
        self.logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        // Simplified physical core detection
        self.physical_cores = if self.features.sse {
            // x86_64 with likely HT
            self.logical_cores / 2
        } else {
            self.logical_cores
        };

        // Set frequencies (would be detected from CPUID or system files in real implementation)
        self.base_frequency = 2400.0; // 2.4 GHz default
        self.max_frequency = 3600.0; // 3.6 GHz default
    }

    /// Create microarchitecture-specific optimization profile
    fn create_optimization_profile(&mut self) {
        self.optimization = match (self.x86_microarch, self.arm_microarch) {
            (Some(X86Microarchitecture::Skylake | X86Microarchitecture::IceLake), _) => {
                MicroarchOptimization {
                    optimal_vector_width: 64, // AVX-512
                    unroll_factor: 8,
                    matrix_block_size: 128,
                    prefetch_distance: 12,
                    branch_friendly: true,
                    prefer_fma: true,
                    cache_blocking: true,
                    software_prefetch: true,
                    memory_alignment: 64,
                    parallel_chunk_size: 2048,
                    ht_aware: true,
                    numa_aware: true,
                }
            }
            (Some(X86Microarchitecture::Haswell | X86Microarchitecture::Broadwell), _) => {
                MicroarchOptimization {
                    optimal_vector_width: 32, // AVX2
                    unroll_factor: 4,
                    matrix_block_size: 64,
                    prefetch_distance: 8,
                    branch_friendly: true,
                    prefer_fma: true,
                    cache_blocking: true,
                    software_prefetch: true,
                    memory_alignment: 32,
                    parallel_chunk_size: 1024,
                    ht_aware: true,
                    numa_aware: false,
                }
            }
            (Some(X86Microarchitecture::Zen2 | X86Microarchitecture::Zen3), _) => {
                MicroarchOptimization {
                    optimal_vector_width: 32, // AVX2
                    unroll_factor: 6,
                    matrix_block_size: 96,
                    prefetch_distance: 10,
                    branch_friendly: true,
                    prefer_fma: true,
                    cache_blocking: true,
                    software_prefetch: true,
                    memory_alignment: 32,
                    parallel_chunk_size: 1536,
                    ht_aware: false, // AMD doesn't use SMT as aggressively
                    numa_aware: true,
                }
            }
            (_, Some(ArmMicroarchitecture::M1)) => {
                MicroarchOptimization {
                    optimal_vector_width: 16, // NEON 128-bit
                    unroll_factor: 4,
                    matrix_block_size: 96,
                    prefetch_distance: 16,
                    branch_friendly: true,
                    prefer_fma: true,
                    cache_blocking: true,
                    software_prefetch: false, // Hardware prefetch is very good
                    memory_alignment: 16,
                    parallel_chunk_size: 1024,
                    ht_aware: false,
                    numa_aware: false, // Unified memory architecture
                }
            }
            (_, Some(ArmMicroarchitecture::M2)) => {
                MicroarchOptimization {
                    optimal_vector_width: 16, // NEON 128-bit
                    unroll_factor: 6,         // M2 has better execution units
                    matrix_block_size: 128,   // Larger L2 cache
                    prefetch_distance: 20,
                    branch_friendly: true,
                    prefer_fma: true,
                    cache_blocking: true,
                    software_prefetch: false, // Hardware prefetch is very good
                    memory_alignment: 16,
                    parallel_chunk_size: 1536, // Higher bandwidth
                    ht_aware: false,
                    numa_aware: false, // Unified memory architecture
                }
            }
            (_, Some(ArmMicroarchitecture::M3)) => {
                MicroarchOptimization {
                    optimal_vector_width: 16, // NEON 128-bit
                    unroll_factor: 8,         // M3 has even better execution units
                    matrix_block_size: 144,   // Larger L2 cache
                    prefetch_distance: 24,
                    branch_friendly: true,
                    prefer_fma: true,
                    cache_blocking: true,
                    software_prefetch: false, // Hardware prefetch is very good
                    memory_alignment: 16,
                    parallel_chunk_size: 2048, // Higher bandwidth
                    ht_aware: false,
                    numa_aware: false, // Unified memory architecture
                }
            }
            (
                _,
                Some(
                    ArmMicroarchitecture::CortexA76
                    | ArmMicroarchitecture::CortexA77
                    | ArmMicroarchitecture::CortexA78,
                ),
            ) => {
                MicroarchOptimization {
                    optimal_vector_width: 16, // NEON 128-bit
                    unroll_factor: 4,
                    matrix_block_size: 64, // Smaller cache than Apple Silicon
                    prefetch_distance: 8,
                    branch_friendly: true,
                    prefer_fma: true,
                    cache_blocking: true,
                    software_prefetch: true, // May benefit from software prefetch
                    memory_alignment: 16,
                    parallel_chunk_size: 512,
                    ht_aware: false,
                    numa_aware: true, // Some ARM64 systems have NUMA
                }
            }
            _ => MicroarchOptimization::default(),
        };
    }
}

impl Default for CpuInfo {
    fn default() -> Self {
        Self {
            features: CpuFeatures::default(),
            cache: CacheInfo::default(),
            x86_microarch: None,
            arm_microarch: None,
            optimization: MicroarchOptimization::default(),
            vendor: "Unknown".to_string(),
            model_name: "Unknown".to_string(),
            physical_cores: 4,
            logical_cores: 4,
            base_frequency: 2000.0,
            max_frequency: 3000.0,
        }
    }
}

// Helper functions
#[cfg(target_arch = "x86_64")]
fn has_cpuid() -> bool {
    true // CPUID is always available on x86_64
}

#[cfg(not(target_arch = "x86_64"))]
fn has_cpuid() -> bool {
    false
}

/// Detect x86 microarchitecture
pub fn detect_x86_microarchitecture() -> Option<X86Microarchitecture> {
    #[cfg(target_arch = "x86_64")]
    {
        if !has_cpuid() {
            return Some(X86Microarchitecture::Unknown);
        }

        unsafe {
            let cpuid = __cpuid(0);
            if cpuid.eax < 1 {
                return Some(X86Microarchitecture::Unknown);
            }

            let info = __cpuid(1);
            let family = ((info.eax >> 8) & 0xF) + ((info.eax >> 20) & 0xFF);
            let model = ((info.eax >> 4) & 0xF) | (((info.eax >> 16) & 0xF) << 4);

            // Intel detection
            if cpuid.ebx == 0x756e6547 && cpuid.edx == 0x49656e69 && cpuid.ecx == 0x6c65746e {
                return Some(match (family, model) {
                    (6, 0x1E..=0x1F) => X86Microarchitecture::Nehalem,
                    (6, 0x2A..=0x2D) => X86Microarchitecture::SandyBridge,
                    (6, 0x3A | 0x3B) => X86Microarchitecture::IvyBridge,
                    (6, 0x3C | 0x3E | 0x3F | 0x45 | 0x46) => X86Microarchitecture::Haswell,
                    (6, 0x3D | 0x47 | 0x4F | 0x56) => X86Microarchitecture::Broadwell,
                    (6, 0x4E | 0x5E | 0x8E) => X86Microarchitecture::Skylake,
                    (6, 0x97 | 0x9A) => X86Microarchitecture::AlderLake,
                    _ => X86Microarchitecture::Unknown,
                });
            }

            // AMD detection
            if cpuid.ebx == 0x68747541 && cpuid.edx == 0x69746e65 && cpuid.ecx == 0x444d4163 {
                return Some(match family {
                    0x17 => X86Microarchitecture::Zen,
                    0x19 => X86Microarchitecture::Zen3,
                    _ => X86Microarchitecture::Unknown,
                });
            }
        }
    }

    None
}

/// Detect ARM microarchitecture
pub fn detect_arm_microarchitecture() -> Option<ArmMicroarchitecture> {
    #[cfg(target_arch = "aarch64")]
    {
        // This is a simplified detection - in practice, we'd need to read
        // CPU identification registers or parse /proc/cpuinfo
        if cfg!(target_os = "macos") {
            Some(ArmMicroarchitecture::M1) // Assume Apple Silicon
        } else {
            Some(ArmMicroarchitecture::CortexA55) // Generic ARM64
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        None
    }
}
