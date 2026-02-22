//! Platform-specific CPU optimizations with microarchitecture detection
//!
//! This module provides advanced x86_64 and ARM64 optimizations that are specifically
//! tuned for different CPU microarchitectures and features.

use crate::error::{BackendError, BackendResult};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::sync::OnceLock;

/// CPU microarchitecture types for x86_64
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum X86Microarchitecture {
    /// Intel Core 2 / Penryn (SSE4.1)
    Core2,
    /// Intel Nehalem (SSE4.2)
    Nehalem,
    /// Intel Sandy Bridge (AVX)
    SandyBridge,
    /// Intel Ivy Bridge (enhanced AVX)
    IvyBridge,
    /// Intel Haswell (AVX2, FMA)
    Haswell,
    /// Intel Broadwell (enhanced AVX2)
    Broadwell,
    /// Intel Skylake (AVX-512 foundation)
    Skylake,
    /// Intel Kaby Lake
    KabyLake,
    /// Intel Coffee Lake
    CoffeeLake,
    /// Intel Ice Lake (enhanced AVX-512)
    IceLake,
    /// Intel Tiger Lake
    TigerLake,
    /// Intel Alder Lake (hybrid architecture)
    AlderLake,
    /// Intel Raptor Lake
    RaptorLake,
    /// Intel Meteor Lake
    MeteorLake,
    /// AMD K8
    K8,
    /// AMD K10
    K10,
    /// AMD Bulldozer
    Bulldozer,
    /// AMD Piledriver
    Piledriver,
    /// AMD Steamroller
    Steamroller,
    /// AMD Excavator
    Excavator,
    /// AMD Zen
    Zen,
    /// AMD Zen+
    ZenPlus,
    /// AMD Zen 2
    Zen2,
    /// AMD Zen 3
    Zen3,
    /// AMD Zen 4
    Zen4,
    /// Unknown/Generic x86_64
    Unknown,
}

/// ARM microarchitecture types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArmMicroarchitecture {
    /// Apple A7 (Cyclone)
    Cyclone,
    /// Apple A8 (Typhoon)  
    Typhoon,
    /// Apple A9 (Twister)
    Twister,
    /// Apple A10 (Hurricane)
    Hurricane,
    /// Apple A11 (Monsoon/Mistral)
    Bionic,
    /// Apple A12 (Vortex/Tempest)
    A12,
    /// Apple A13 (Lightning/Thunder)
    A13,
    /// Apple A14 (Firestorm/Icestorm)
    A14,
    /// Apple A15 (Avalanche/Blizzard)
    A15,
    /// Apple A16 (Everest/Sawtooth)
    A16,
    /// Apple M1 (Firestorm/Icestorm)
    M1,
    /// Apple M2 (Avalanche/Blizzard)
    M2,
    /// Apple M3 (Enhanced Avalanche/Blizzard)
    M3,
    /// ARM Cortex-A53
    CortexA53,
    /// ARM Cortex-A55
    CortexA55,
    /// ARM Cortex-A57
    CortexA57,
    /// ARM Cortex-A72
    CortexA72,
    /// ARM Cortex-A73
    CortexA73,
    /// ARM Cortex-A75
    CortexA75,
    /// ARM Cortex-A76
    CortexA76,
    /// ARM Cortex-A77
    CortexA77,
    /// ARM Cortex-A78
    CortexA78,
    /// ARM Cortex-X1
    CortexX1,
    /// ARM Cortex-A510
    CortexA510,
    /// ARM Cortex-A710
    CortexA710,
    /// ARM Cortex-X2
    CortexX2,
    /// ARM Cortex-A715
    CortexA715,
    /// ARM Cortex-X3
    CortexX3,
    /// ARM Neoverse V1
    NeoverseV1,
    /// ARM Neoverse N1
    NeoverseN1,
    /// ARM Neoverse N2
    NeoverseN2,
    /// Unknown/Generic ARM64
    Unknown,
}

/// CPU feature flags detected at runtime
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    // x86_64 features
    pub sse: bool,
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
    pub sse4_1: bool,
    pub sse4_2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512dq: bool,
    pub avx512cd: bool,
    pub avx512bw: bool,
    pub avx512vl: bool,
    pub avx512vnni: bool,
    pub avx512bf16: bool,
    pub avx512vp2intersect: bool,
    pub fma: bool,
    pub fma4: bool,
    pub bmi1: bool,
    pub bmi2: bool,
    pub lzcnt: bool,
    pub popcnt: bool,
    pub f16c: bool,
    pub rdrand: bool,
    pub rdseed: bool,
    pub aes: bool,
    pub pclmul: bool,
    pub sha: bool,
    pub adx: bool,
    pub prefetchw: bool,
    pub clflushopt: bool,
    pub clwb: bool,
    pub movbe: bool,
    pub rtm: bool,
    pub hle: bool,
    pub mpx: bool,
    pub xsave: bool,
    pub xsaveopt: bool,
    pub xgetbv: bool,
    pub invariant_tsc: bool,
    pub rdtscp: bool,

    // ARM64 features
    pub neon: bool,
    pub fp: bool,
    pub asimd: bool,
    pub aes_arm: bool,
    pub pmull: bool,
    pub sha1: bool,
    pub sha256: bool,
    pub crc32: bool,
    pub atomics: bool,
    pub fphp: bool,
    pub asimdhp: bool,
    pub cpuid: bool,
    pub asimdrdm: bool,
    pub jscvt: bool,
    pub fcma: bool,
    pub lrcpc: bool,
    pub dcpop: bool,
    pub sha3: bool,
    pub sm3: bool,
    pub sm4: bool,
    pub asimddp: bool,
    pub sha512: bool,
    pub sve: bool,
    pub sve2: bool,
    pub sveaes: bool,
    pub svepmull: bool,
    pub svebitperm: bool,
    pub svesha3: bool,
    pub svesm4: bool,
    pub flagm: bool,
    pub ssbs: bool,
    pub sb: bool,
    pub paca: bool,
    pub pacg: bool,
    pub dgh: bool,
    pub bf16: bool,
    pub i8mm: bool,
    pub rng: bool,
    pub bti: bool,
    pub mte: bool,
    pub ecv: bool,
    pub afp: bool,
    pub rpres: bool,
}

impl Default for CpuFeatures {
    fn default() -> Self {
        Self {
            // Initialize all features as false - will be detected at runtime
            sse: false,
            sse2: false,
            sse3: false,
            ssse3: false,
            sse4_1: false,
            sse4_2: false,
            avx: false,
            avx2: false,
            avx512f: false,
            avx512dq: false,
            avx512cd: false,
            avx512bw: false,
            avx512vl: false,
            avx512vnni: false,
            avx512bf16: false,
            avx512vp2intersect: false,
            fma: false,
            fma4: false,
            bmi1: false,
            bmi2: false,
            lzcnt: false,
            popcnt: false,
            f16c: false,
            rdrand: false,
            rdseed: false,
            aes: false,
            pclmul: false,
            sha: false,
            adx: false,
            prefetchw: false,
            clflushopt: false,
            clwb: false,
            movbe: false,
            rtm: false,
            hle: false,
            mpx: false,
            xsave: false,
            xsaveopt: false,
            xgetbv: false,
            invariant_tsc: false,
            rdtscp: false,
            neon: false,
            fp: false,
            asimd: false,
            aes_arm: false,
            pmull: false,
            sha1: false,
            sha256: false,
            crc32: false,
            atomics: false,
            fphp: false,
            asimdhp: false,
            cpuid: false,
            asimdrdm: false,
            jscvt: false,
            fcma: false,
            lrcpc: false,
            dcpop: false,
            sha3: false,
            sm3: false,
            sm4: false,
            asimddp: false,
            sha512: false,
            sve: false,
            sve2: false,
            sveaes: false,
            svepmull: false,
            svebitperm: false,
            svesha3: false,
            svesm4: false,
            flagm: false,
            ssbs: false,
            sb: false,
            paca: false,
            pacg: false,
            dgh: false,
            bf16: false,
            i8mm: false,
            rng: false,
            bti: false,
            mte: false,
            ecv: false,
            afp: false,
            rpres: false,
        }
    }
}

/// Cache hierarchy information
#[derive(Debug, Clone)]
pub struct CacheInfo {
    /// L1 instruction cache size in bytes
    pub l1i_size: usize,
    /// L1 data cache size in bytes
    pub l1d_size: usize,
    /// L1 cache associativity
    pub l1_associativity: usize,
    /// L1 cache line size in bytes
    pub l1_line_size: usize,
    /// L2 cache size in bytes
    pub l2_size: usize,
    /// L2 cache associativity
    pub l2_associativity: usize,
    /// L2 cache line size in bytes
    pub l2_line_size: usize,
    /// L3 cache size in bytes
    pub l3_size: usize,
    /// L3 cache associativity
    pub l3_associativity: usize,
    /// L3 cache line size in bytes
    pub l3_line_size: usize,
    /// Number of cores sharing L3 cache
    pub l3_sharing: usize,
    /// TLB entries for 4KB pages
    pub tlb_4kb_entries: usize,
    /// TLB entries for 2MB pages
    pub tlb_2mb_entries: usize,
    /// TLB entries for 1GB pages
    pub tlb_1gb_entries: usize,
}

impl Default for CacheInfo {
    fn default() -> Self {
        Self {
            l1i_size: 32 * 1024, // 32KB
            l1d_size: 32 * 1024, // 32KB
            l1_associativity: 8,
            l1_line_size: 64,
            l2_size: 256 * 1024, // 256KB
            l2_associativity: 8,
            l2_line_size: 64,
            l3_size: 8 * 1024 * 1024, // 8MB
            l3_associativity: 16,
            l3_line_size: 64,
            l3_sharing: 8,
            tlb_4kb_entries: 64,
            tlb_2mb_entries: 32,
            tlb_1gb_entries: 4,
        }
    }
}

/// Microarchitecture-specific optimization parameters
#[derive(Debug, Clone)]
pub struct MicroarchOptimization {
    /// Optimal vector width for SIMD operations
    pub optimal_vector_width: usize,
    /// Preferred unroll factor for loops
    pub unroll_factor: usize,
    /// Optimal block size for matrix operations
    pub matrix_block_size: usize,
    /// Memory prefetch distance (cache lines ahead)
    pub prefetch_distance: usize,
    /// Branch predictor friendly loop structure
    pub branch_friendly: bool,
    /// Instruction scheduling preferences
    pub prefer_fma: bool,
    /// Cache blocking strategy
    pub cache_blocking: bool,
    /// Software prefetching enabled
    pub software_prefetch: bool,
    /// Memory alignment requirements
    pub memory_alignment: usize,
    /// Optimal chunk size for parallel operations
    pub parallel_chunk_size: usize,
    /// Hyper-threading awareness
    pub ht_aware: bool,
    /// NUMA awareness
    pub numa_aware: bool,
}

impl Default for MicroarchOptimization {
    fn default() -> Self {
        Self {
            optimal_vector_width: 32, // 256-bit vectors (AVX2)
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
}

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

/// Platform-specific optimized operations
#[derive(Debug)]
pub struct PlatformOptimizedOps {
    cpu_info: &'static CpuInfo,
}

impl PlatformOptimizedOps {
    /// Create new platform-optimized operations
    pub fn new() -> Self {
        Self {
            cpu_info: CpuInfo::get(),
        }
    }

    /// Get the detected CPU information
    pub fn cpu_info(&self) -> &CpuInfo {
        self.cpu_info
    }

    /// Optimized vector dot product
    pub fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> BackendResult<f32> {
        if a.len() != b.len() {
            return Err(BackendError::InvalidArgument(
                "Vector lengths must match".to_string(),
            ));
        }

        #[cfg(target_arch = "x86_64")]
        {
            if self.cpu_info.features.avx2 {
                return Ok(self.dot_product_avx2(a, b));
            } else if self.cpu_info.features.sse {
                return Ok(self.dot_product_sse(a, b));
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if self.cpu_info.features.neon {
                return Ok(self.dot_product_neon(a, b));
            }
        }

        // Fallback to scalar implementation
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
    }

    /// AVX2 optimized dot product
    #[cfg(target_arch = "x86_64")]
    fn dot_product_avx2(&self, a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let mut sum = _mm256_setzero_ps();
            let len = a.len();
            let simd_len = len & !7; // Round down to multiple of 8

            // Process 8 elements at a time with AVX2
            for i in (0..simd_len).step_by(8) {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));

                if self.cpu_info.features.fma {
                    sum = _mm256_fmadd_ps(va, vb, sum);
                } else {
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
                }
            }

            // Horizontal sum of vector
            let sum_low = _mm256_castps256_ps128(sum);
            let sum_high = _mm256_extractf128_ps(sum, 1);
            let sum128 = _mm_add_ps(sum_low, sum_high);
            let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
            let mut result = _mm_cvtss_f32(sum32);

            // Handle remaining elements
            for i in simd_len..len {
                result += a[i] * b[i];
            }

            result
        }
    }

    /// SSE optimized dot product
    #[cfg(target_arch = "x86_64")]
    fn dot_product_sse(&self, a: &[f32], b: &[f32]) -> f32 {
        unsafe {
            let mut sum = _mm_setzero_ps();
            let len = a.len();
            let simd_len = len & !3; // Round down to multiple of 4

            // Process 4 elements at a time with SSE
            for i in (0..simd_len).step_by(4) {
                let va = _mm_loadu_ps(a.as_ptr().add(i));
                let vb = _mm_loadu_ps(b.as_ptr().add(i));
                sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
            }

            // Horizontal sum
            let sum64 = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
            let mut result = _mm_cvtss_f32(sum32);

            // Handle remaining elements
            for i in simd_len..len {
                result += a[i] * b[i];
            }

            result
        }
    }

    /// NEON optimized dot product
    #[cfg(target_arch = "aarch64")]
    fn dot_product_neon(&self, a: &[f32], b: &[f32]) -> f32 {
        use std::arch::aarch64::*;

        unsafe {
            let mut sum = vdupq_n_f32(0.0);
            let len = a.len();
            let simd_len = len & !3; // Round down to multiple of 4

            // Process 4 elements at a time with NEON
            for i in (0..simd_len).step_by(4) {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                sum = vfmaq_f32(sum, va, vb);
            }

            // Horizontal sum
            let sum_pair = vpadd_f32(vget_low_f32(sum), vget_high_f32(sum));
            let mut result = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);

            // Handle remaining elements
            for i in simd_len..len {
                result += a[i] * b[i];
            }

            result
        }
    }

    /// Optimized matrix multiplication with microarchitecture-specific blocking
    pub fn matrix_multiply_f32(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> BackendResult<()> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(BackendError::InvalidArgument(
                "Matrix dimension mismatch".to_string(),
            ));
        }

        let block_size = self.cpu_info.optimization.matrix_block_size;

        // Use cache-blocked matrix multiplication
        for ii in (0..m).step_by(block_size) {
            for jj in (0..n).step_by(block_size) {
                for kk in (0..k).step_by(block_size) {
                    let i_end = (ii + block_size).min(m);
                    let j_end = (jj + block_size).min(n);
                    let k_end = (kk + block_size).min(k);

                    self.matrix_multiply_block(a, b, c, ii, i_end, jj, j_end, kk, k_end, m, n, k);
                }
            }
        }

        Ok(())
    }

    /// Optimized block matrix multiplication
    fn matrix_multiply_block(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        i_start: usize,
        i_end: usize,
        j_start: usize,
        j_end: usize,
        k_start: usize,
        k_end: usize,
        _m: usize,
        n: usize,
        k: usize,
    ) {
        let unroll = self.cpu_info.optimization.unroll_factor;

        for i in i_start..i_end {
            for j in (j_start..j_end).step_by(unroll) {
                let j_limit = (j + unroll).min(j_end);

                for kk in k_start..k_end {
                    let a_val = a[i * k + kk];

                    // Unrolled inner loop
                    for jj in j..j_limit {
                        c[i * n + jj] += a_val * b[kk * n + jj];
                    }
                }
            }
        }
    }

    /// Get optimal chunk size for parallel operations
    pub fn get_optimal_parallel_chunk_size(&self, total_elements: usize) -> usize {
        let base_chunk = self.cpu_info.optimization.parallel_chunk_size;
        let num_threads = self.cpu_info.logical_cores;

        // Ensure we have enough work per thread
        let min_chunk = total_elements / (num_threads * 4);

        base_chunk.max(min_chunk).min(total_elements)
    }

    /// Check if operation should use software prefetching
    pub fn should_use_prefetch(&self, data_size: usize) -> bool {
        self.cpu_info.optimization.software_prefetch && data_size > self.cpu_info.cache.l3_size
    }

    /// Get optimal memory alignment for current platform
    pub fn get_memory_alignment(&self) -> usize {
        self.cpu_info.optimization.memory_alignment
    }

    /// Print detailed CPU information
    pub fn print_cpu_info(&self) {
        let info = self.cpu_info;

        println!("CPU Information:");
        println!("  Vendor: {}", info.vendor);
        println!("  Model: {}", info.model_name);
        println!(
            "  Cores: {} physical, {} logical",
            info.physical_cores, info.logical_cores
        );
        println!(
            "  Frequency: {:.1} MHz base, {:.1} MHz max",
            info.base_frequency, info.max_frequency
        );

        if let Some(arch) = info.x86_microarch {
            println!("  Microarchitecture: {:?}", arch);
        }
        if let Some(arch) = info.arm_microarch {
            println!("  Microarchitecture: {:?}", arch);
        }

        println!(
            "  Cache: L1D={}KB, L1I={}KB, L2={}KB, L3={}KB",
            info.cache.l1d_size / 1024,
            info.cache.l1i_size / 1024,
            info.cache.l2_size / 1024,
            info.cache.l3_size / 1024
        );

        println!("Features:");
        #[cfg(target_arch = "x86_64")]
        {
            if info.features.sse {
                print!(" SSE");
            }
            if info.features.sse2 {
                print!(" SSE2");
            }
            if info.features.sse3 {
                print!(" SSE3");
            }
            if info.features.ssse3 {
                print!(" SSSE3");
            }
            if info.features.sse4_1 {
                print!(" SSE4.1");
            }
            if info.features.sse4_2 {
                print!(" SSE4.2");
            }
            if info.features.avx {
                print!(" AVX");
            }
            if info.features.avx2 {
                print!(" AVX2");
            }
            if info.features.avx512f {
                print!(" AVX-512F");
            }
            if info.features.fma {
                print!(" FMA");
            }
            if info.features.bmi1 {
                print!(" BMI1");
            }
            if info.features.bmi2 {
                print!(" BMI2");
            }
            if info.features.aes {
                print!(" AES");
            }
            if info.features.sha {
                print!(" SHA");
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if info.features.neon {
                print!(" NEON");
            }
            if info.features.fp {
                print!(" FP");
            }
            if info.features.asimd {
                print!(" ASIMD");
            }
            if info.features.aes_arm {
                print!(" AES");
            }
            if info.features.sha1 {
                print!(" SHA1");
            }
            if info.features.sha256 {
                print!(" SHA256");
            }
            if info.features.crc32 {
                print!(" CRC32");
            }
            if info.features.sve {
                print!(" SVE");
            }
        }
        println!();

        println!("Optimizations:");
        println!(
            "  Vector width: {} bytes",
            info.optimization.optimal_vector_width
        );
        println!("  Unroll factor: {}", info.optimization.unroll_factor);
        println!(
            "  Matrix block size: {}",
            info.optimization.matrix_block_size
        );
        println!(
            "  Memory alignment: {} bytes",
            info.optimization.memory_alignment
        );
        println!(
            "  Parallel chunk size: {}",
            info.optimization.parallel_chunk_size
        );
        println!(
            "  Software prefetch: {}",
            info.optimization.software_prefetch
        );
        println!("  HT aware: {}", info.optimization.ht_aware);
        println!("  NUMA aware: {}", info.optimization.numa_aware);
    }
}

impl Default for PlatformOptimizedOps {
    fn default() -> Self {
        Self::new()
    }
}

// Enhanced implementations for platform optimizer
#[derive(Debug)]
pub struct PlatformOptimizer {
    pub features: CpuFeatures,
    pub x86_arch: Option<X86Microarchitecture>,
    pub arm_arch: Option<ArmMicroarchitecture>,
    pub optimized_ops: PlatformOptimizedOps,
}

pub struct CpuOptimizer;
pub struct OptimizedOperations;
pub struct OptimizationCache;

impl PlatformOptimizer {
    pub fn new() -> BackendResult<Self> {
        let features = detect_cpu_features()?;
        let x86_arch = detect_x86_microarchitecture();
        let arm_arch = detect_arm_microarchitecture();
        let optimized_ops = PlatformOptimizedOps::new();

        Ok(Self {
            features,
            x86_arch,
            arm_arch,
            optimized_ops,
        })
    }

    pub fn get_cpu_info(&self) -> String {
        format!(
            "CPU Features: AVX={}, AVX2={}, AVX512F={}, NEON={}, x86_arch={:?}, arm_arch={:?}",
            self.features.avx,
            self.features.avx2,
            self.features.avx512f,
            self.features.neon,
            self.x86_arch,
            self.arm_arch
        )
    }
}

// CPUID helper function for x86_64
#[cfg(target_arch = "x86_64")]
fn has_cpuid() -> bool {
    true // CPUID is always available on x86_64
}

#[cfg(not(target_arch = "x86_64"))]
fn has_cpuid() -> bool {
    false
}

/// Detect CPU features for the current platform
pub fn detect_cpu_features() -> BackendResult<CpuFeatures> {
    #[cfg(target_arch = "x86_64")]
    {
        if !has_cpuid() {
            return Ok(CpuFeatures::default());
        }

        unsafe {
            let cpuid = __cpuid(1);
            let sse = (cpuid.edx & (1 << 25)) != 0;
            let sse2 = (cpuid.edx & (1 << 26)) != 0;
            let sse3 = (cpuid.ecx & (1 << 0)) != 0;
            let ssse3 = (cpuid.ecx & (1 << 9)) != 0;
            let sse4_1 = (cpuid.ecx & (1 << 19)) != 0;
            let sse4_2 = (cpuid.ecx & (1 << 20)) != 0;

            let extended_features = __cpuid(7);
            let avx = (cpuid.ecx & (1 << 28)) != 0;
            let avx2 = (extended_features.ebx & (1 << 5)) != 0;
            let avx512f = (extended_features.ebx & (1 << 16)) != 0;
            let avx512cd = (extended_features.ebx & (1 << 28)) != 0;
            let fma = (cpuid.ecx & (1 << 12)) != 0;

            Ok(CpuFeatures {
                sse,
                sse2,
                sse3,
                ssse3,
                sse4_1,
                sse4_2,
                avx,
                avx2,
                avx512f,
                avx512cd,
                fma,
                neon: false,
                ..Default::default()
            })
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        Ok(CpuFeatures {
            neon: true, // NEON is standard on AArch64
            ..Default::default()
        })
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        Ok(CpuFeatures::default())
    }
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

impl Default for PlatformOptimizer {
    fn default() -> Self {
        Self::new().unwrap_or_else(|_| Self {
            features: CpuFeatures::default(),
            x86_arch: None,
            arm_arch: None,
            optimized_ops: PlatformOptimizedOps::new(),
        })
    }
}

impl CpuOptimizer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizedOperations {
    pub fn new() -> Self {
        Self
    }
}

impl Default for OptimizedOperations {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationCache {
    pub fn new() -> Self {
        Self
    }
}

impl Default for OptimizationCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_info_detection() {
        let cpu_info = CpuInfo::get();

        // Basic sanity checks
        assert!(cpu_info.logical_cores > 0);
        assert!(cpu_info.physical_cores > 0);
        assert!(cpu_info.physical_cores <= cpu_info.logical_cores);
        assert!(cpu_info.base_frequency > 0.0);
        assert!(cpu_info.max_frequency >= cpu_info.base_frequency);

        // Cache sizes should be reasonable
        assert!(cpu_info.cache.l1d_size >= 16 * 1024); // At least 16KB
        assert!(cpu_info.cache.l2_size >= 128 * 1024); // At least 128KB
        assert!(cpu_info.cache.l1_line_size >= 32); // At least 32 bytes

        println!(
            "Detected CPU: {} cores, {}KB L1, {}KB L2, {}KB L3",
            cpu_info.logical_cores,
            cpu_info.cache.l1d_size / 1024,
            cpu_info.cache.l2_size / 1024,
            cpu_info.cache.l3_size / 1024
        );
    }

    #[test]
    fn test_platform_optimized_ops() {
        let ops = PlatformOptimizedOps::new();

        // Test dot product
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let result = ops.dot_product_f32(&a, &b).unwrap();
        assert_eq!(result, 40.0); // 1*2 + 2*3 + 3*4 + 4*5 = 40

        // Test with mismatched lengths
        let c = vec![1.0, 2.0];
        assert!(ops.dot_product_f32(&a, &c).is_err());

        // Test matrix multiplication
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 matrix
        let mut c = vec![0.0; 4]; // 2x2 result

        ops.matrix_multiply_f32(&a, &b, &mut c, 2, 2, 2).unwrap();
        // Expected: [1*5+2*7, 1*6+2*8] = [19, 22]
        //          [3*5+4*7, 3*6+4*8] = [43, 50]
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_optimization_parameters() {
        let ops = PlatformOptimizedOps::new();
        let cpu_info = ops.cpu_info();

        // Check that optimization parameters are reasonable
        assert!(cpu_info.optimization.optimal_vector_width >= 16);
        assert!(cpu_info.optimization.unroll_factor >= 2);
        assert!(cpu_info.optimization.matrix_block_size >= 32);
        assert!(cpu_info.optimization.memory_alignment >= 16);
        assert!(cpu_info.optimization.parallel_chunk_size >= 64);

        // Test chunk size calculation
        let chunk_size = ops.get_optimal_parallel_chunk_size(10000);
        assert!(chunk_size > 0);
        assert!(chunk_size <= 10000);

        // Test memory alignment
        let alignment = ops.get_memory_alignment();
        assert!(alignment.is_power_of_two());
        assert!(alignment >= 16);
    }

    #[test]
    fn test_feature_detection() {
        let cpu_info = CpuInfo::get();

        #[cfg(target_arch = "x86_64")]
        {
            // Most modern x86_64 CPUs should have these
            assert!(cpu_info.features.sse);
            assert!(cpu_info.features.sse2);

            // Print detected features for debugging
            println!(
                "x86_64 features: AVX={}, AVX2={}, AVX-512F={}, FMA={}",
                cpu_info.features.avx,
                cpu_info.features.avx2,
                cpu_info.features.avx512f,
                cpu_info.features.fma
            );
        }

        #[cfg(target_arch = "aarch64")]
        {
            // Most ARM64 systems should have NEON
            assert!(cpu_info.features.neon);
            assert!(cpu_info.features.fp);

            println!(
                "ARM64 features: NEON={}, FP={}, ASIMD={}, AES={}",
                cpu_info.features.neon,
                cpu_info.features.fp,
                cpu_info.features.asimd,
                cpu_info.features.aes_arm
            );
        }
    }

    #[test]
    fn test_microarchitecture_detection() {
        let cpu_info = CpuInfo::get();

        #[cfg(target_arch = "x86_64")]
        {
            assert!(cpu_info.x86_microarch.is_some());
            println!(
                "Detected x86_64 microarchitecture: {:?}",
                cpu_info.x86_microarch
            );
        }

        #[cfg(target_arch = "aarch64")]
        {
            assert!(cpu_info.arm_microarch.is_some());
            println!(
                "Detected ARM64 microarchitecture: {:?}",
                cpu_info.arm_microarch
            );
        }

        assert!(!cpu_info.vendor.is_empty());
        println!("CPU vendor: {}", cpu_info.vendor);
    }
}
