//! Enhanced x86_64 optimizations with microarchitecture-specific tuning
//!
//! This module provides advanced optimizations specifically tuned for different
//! x86_64 microarchitectures, including Intel and AMD processors.

use crate::cpu::platform_optimization::{X86Microarchitecture, CpuFeatures};
use crate::error::BackendResult;
use torsh_core::error::TorshError;
use std::sync::OnceLock;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

/// Enhanced x86_64 optimizer with microarchitecture-specific tuning
#[derive(Debug)]
pub struct EnhancedX86Optimizer {
    microarch: X86Microarchitecture,
    features: CpuFeatures,
    optimization_params: OptimizationParameters,
    vector_unit_config: VectorUnitConfig,
    cache_config: CacheConfiguration,
    execution_units: ExecutionUnitInfo,
}

/// Optimization parameters tuned for specific microarchitectures
#[derive(Debug, Clone)]
pub struct OptimizationParameters {
    /// Optimal unroll factor for loops
    pub loop_unroll_factor: usize,
    /// Preferred vector width in bits
    pub preferred_vector_width: usize,
    /// Maximum memory bandwidth utilization
    pub max_memory_bandwidth_utilization: f64,
    /// Instruction scheduling window size
    pub scheduling_window_size: usize,
    /// Branch prediction threshold
    pub branch_prediction_threshold: f64,
    /// Cache blocking factors
    pub cache_blocking_factors: CacheBlockingFactors,
    /// Parallel execution parameters
    pub parallel_params: ParallelExecutionParams,
}

/// Cache blocking factors for different cache levels
#[derive(Debug, Clone)]
pub struct CacheBlockingFactors {
    pub l1_block_size: usize,
    pub l2_block_size: usize,
    pub l3_block_size: usize,
    pub tlb_block_size: usize,
}

/// Parallel execution parameters
#[derive(Debug, Clone)]
pub struct ParallelExecutionParams {
    pub optimal_thread_count: usize,
    pub work_stealing_threshold: usize,
    pub chunk_size_multiplier: f64,
    pub load_balancing_strategy: LoadBalancingStrategy,
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    Static,
    Dynamic,
    WorkStealing,
    NUMA_Aware,
}

/// Vector unit configuration for different microarchitectures
#[derive(Debug, Clone)]
pub struct VectorUnitConfig {
    pub avx512_available: bool,
    pub avx2_optimal: bool,
    pub fma_units: usize,
    pub vector_register_count: usize,
    pub optimal_vector_size: usize,
    pub mask_register_count: usize,
    pub memory_units: usize,
}

/// Cache configuration information
#[derive(Debug, Clone)]
pub struct CacheConfiguration {
    pub l1i_size: usize,
    pub l1d_size: usize,
    pub l2_size: usize,
    pub l3_size: usize,
    pub l1_associativity: usize,
    pub l2_associativity: usize,
    pub l3_associativity: usize,
    pub cache_line_size: usize,
    pub tlb_entries: usize,
    pub prefetch_distance: usize,
}

/// Execution unit information
#[derive(Debug, Clone)]
pub struct ExecutionUnitInfo {
    pub integer_units: usize,
    pub fp_units: usize,
    pub vector_units: usize,
    pub load_units: usize,
    pub store_units: usize,
    pub branch_units: usize,
    pub issue_width: usize,
    pub retire_width: usize,
}

impl EnhancedX86Optimizer {
    /// Create a new enhanced x86_64 optimizer
    pub fn new() -> BackendResult<Self> {
        let microarch = Self::detect_microarchitecture()?;
        let features = Self::detect_cpu_features()?;
        let optimization_params = Self::get_optimization_parameters(&microarch, &features);
        let vector_unit_config = Self::get_vector_unit_config(&microarch, &features);
        let cache_config = Self::detect_cache_configuration(&microarch)?;
        let execution_units = Self::get_execution_unit_info(&microarch);
        
        Ok(Self {
            microarch,
            features,
            optimization_params,
            vector_unit_config,
            cache_config,
            execution_units,
        })
    }
    
    /// Detect the CPU microarchitecture
    fn detect_microarchitecture() -> BackendResult<X86Microarchitecture> {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::__cpuid;
            
            unsafe {
                // Get vendor ID
                let vendor_info = __cpuid(0);
                let vendor = [vendor_info.ebx, vendor_info.edx, vendor_info.ecx];
                
                // Get CPU info
                let cpu_info = __cpuid(1);
                let family = (cpu_info.eax >> 8) & 0xF;
                let model = (cpu_info.eax >> 4) & 0xF;
                let extended_family = (cpu_info.eax >> 20) & 0xFF;
                let extended_model = (cpu_info.eax >> 16) & 0xF;
                
                let display_family = if family == 0xF {
                    family + extended_family
                } else {
                    family
                };
                
                let display_model = if family == 0x6 || family == 0xF {
                    (extended_model << 4) + model
                } else {
                    model
                };
                
                // Intel detection
                if vendor == [0x756e6547, 0x49656e69, 0x6c65746e] { // "GenuineIntel"
                    return Ok(Self::detect_intel_microarch(display_family, display_model));
                }
                
                // AMD detection
                if vendor == [0x68747541, 0x69746e65, 0x444d4163] { // "AuthenticAMD"
                    return Ok(Self::detect_amd_microarch(display_family, display_model));
                }
            }
        }
        
        Ok(X86Microarchitecture::Unknown)
    }
    
    /// Detect Intel microarchitecture
    #[cfg(target_arch = "x86_64")]
    fn detect_intel_microarch(family: u32, model: u32) -> X86Microarchitecture {
        match family {
            0x6 => match model {
                0x1A | 0x1E | 0x1F | 0x2E => X86Microarchitecture::Nehalem,
                0x25 | 0x2C | 0x2F => X86Microarchitecture::Nehalem, // Westmere
                0x2A | 0x2D => X86Microarchitecture::SandyBridge,
                0x3A | 0x3E => X86Microarchitecture::IvyBridge,
                0x3C | 0x3F | 0x45 | 0x46 => X86Microarchitecture::Haswell,
                0x3D | 0x47 | 0x4F | 0x56 => X86Microarchitecture::Broadwell,
                0x4E | 0x5E => X86Microarchitecture::Skylake,
                0x8E | 0x9E => X86Microarchitecture::KabyLake,
                0x66 => X86Microarchitecture::CoffeeLake,
                0x7D | 0x7E => X86Microarchitecture::IceLake,
                0x8C | 0x8D => X86Microarchitecture::TigerLake,
                0x97 | 0x9A => X86Microarchitecture::AlderLake,
                0xB7 | 0xBA => X86Microarchitecture::RaptorLake,
                0xAA | 0xAC => X86Microarchitecture::MeteorLake,
                _ => X86Microarchitecture::Unknown,
            },
            _ => X86Microarchitecture::Unknown,
        }
    }
    
    /// Detect AMD microarchitecture
    #[cfg(target_arch = "x86_64")]
    fn detect_amd_microarch(family: u32, model: u32) -> X86Microarchitecture {
        match family {
            0x15 => match model {
                0x00..=0x0F => X86Microarchitecture::Bulldozer,
                0x10..=0x1F => X86Microarchitecture::Piledriver,
                0x30..=0x3F => X86Microarchitecture::Steamroller,
                0x60..=0x7F => X86Microarchitecture::Excavator,
                _ => X86Microarchitecture::Unknown,
            },
            0x17 => match model {
                0x01 | 0x08 | 0x11 | 0x18 => X86Microarchitecture::Zen,
                0x31 | 0x38 => X86Microarchitecture::Zen2,
                _ => X86Microarchitecture::ZenPlus,
            },
            0x19 => match model {
                0x21 | 0x50 => X86Microarchitecture::Zen3,
                0x40..=0x4F => X86Microarchitecture::Zen4,
                _ => X86Microarchitecture::Zen3,
            },
            _ => X86Microarchitecture::Unknown,
        }
    }
    
    /// Detect CPU features using CPUID
    fn detect_cpu_features() -> BackendResult<CpuFeatures> {
        let mut features = CpuFeatures::default();
        
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{__cpuid, __cpuid_count};
            
            unsafe {
                // Basic features (CPUID 1)
                let info = __cpuid(1);
                features.sse = (info.edx & (1 << 25)) != 0;
                features.sse2 = (info.edx & (1 << 26)) != 0;
                features.sse3 = (info.ecx & (1 << 0)) != 0;
                features.ssse3 = (info.ecx & (1 << 9)) != 0;
                features.sse4_1 = (info.ecx & (1 << 19)) != 0;
                features.sse4_2 = (info.ecx & (1 << 20)) != 0;
                features.popcnt = (info.ecx & (1 << 23)) != 0;
                features.aes = (info.ecx & (1 << 25)) != 0;
                features.avx = (info.ecx & (1 << 28)) != 0;
                features.rdrand = (info.ecx & (1 << 30)) != 0;
                features.f16c = (info.ecx & (1 << 29)) != 0;
                features.fma = (info.ecx & (1 << 12)) != 0;
                
                // Extended features (CPUID 7)
                let ext_info = __cpuid_count(7, 0);
                features.avx2 = (ext_info.ebx & (1 << 5)) != 0;
                features.bmi1 = (ext_info.ebx & (1 << 3)) != 0;
                features.bmi2 = (ext_info.ebx & (1 << 8)) != 0;
                features.avx512f = (ext_info.ebx & (1 << 16)) != 0;
                features.avx512dq = (ext_info.ebx & (1 << 17)) != 0;
                features.avx512cd = (ext_info.ebx & (1 << 28)) != 0;
                features.avx512bw = (ext_info.ebx & (1 << 30)) != 0;
                features.avx512vl = (ext_info.ebx & (1 << 31)) != 0;
                features.sha = (ext_info.ebx & (1 << 29)) != 0;
                features.adx = (ext_info.ebx & (1 << 19)) != 0;
                features.rdseed = (ext_info.ebx & (1 << 18)) != 0;
                features.clflushopt = (ext_info.ebx & (1 << 23)) != 0;
                features.clwb = (ext_info.ebx & (1 << 24)) != 0;
                features.avx512vnni = (ext_info.ecx & (1 << 11)) != 0;
                features.avx512bf16 = (ext_info.eax & (1 << 5)) != 0;
                
                // Extended CPUID features
                let max_extended = __cpuid(0x80000000).eax;
                if max_extended >= 0x80000001 {
                    let ext_info = __cpuid(0x80000001);
                    features.lzcnt = (ext_info.ecx & (1 << 5)) != 0;
                    features.fma4 = (ext_info.ecx & (1 << 16)) != 0;
                    features.prefetchw = (ext_info.ecx & (1 << 8)) != 0;
                }
            }
        }
        
        Ok(features)
    }
    
    /// Get optimization parameters for specific microarchitecture
    fn get_optimization_parameters(microarch: &X86Microarchitecture, features: &CpuFeatures) -> OptimizationParameters {
        let (loop_unroll, vector_width, memory_bw, sched_window, branch_thresh) = match microarch {
            X86Microarchitecture::Haswell | X86Microarchitecture::Broadwell => {
                (8, if features.avx2 { 256 } else { 128 }, 0.85, 192, 0.95)
            }
            X86Microarchitecture::Skylake | X86Microarchitecture::KabyLake | X86Microarchitecture::CoffeeLake => {
                (8, if features.avx512f { 512 } else { 256 }, 0.90, 224, 0.96)
            }
            X86Microarchitecture::IceLake | X86Microarchitecture::TigerLake => {
                (12, 512, 0.92, 256, 0.97)
            }
            X86Microarchitecture::AlderLake | X86Microarchitecture::RaptorLake => {
                (16, 512, 0.94, 288, 0.98)
            }
            X86Microarchitecture::Zen2 | X86Microarchitecture::Zen3 => {
                (8, 256, 0.88, 192, 0.95)
            }
            X86Microarchitecture::Zen4 => {
                (12, 512, 0.90, 256, 0.96)
            }
            _ => (4, 128, 0.80, 128, 0.90),
        };
        
        let cache_blocking = match microarch {
            X86Microarchitecture::Haswell | X86Microarchitecture::Broadwell => {
                CacheBlockingFactors {
                    l1_block_size: 16 * 1024,  // 16KB for L1D
                    l2_block_size: 128 * 1024, // 128KB for L2
                    l3_block_size: 4 * 1024 * 1024, // 4MB for L3
                    tlb_block_size: 2 * 1024 * 1024, // 2MB pages
                }
            }
            X86Microarchitecture::Skylake..=X86Microarchitecture::RaptorLake => {
                CacheBlockingFactors {
                    l1_block_size: 24 * 1024,  // 32KB for L1D
                    l2_block_size: 192 * 1024, // 256KB for L2
                    l3_block_size: 6 * 1024 * 1024, // 6MB+ for L3
                    tlb_block_size: 2 * 1024 * 1024, // 2MB pages
                }
            }
            X86Microarchitecture::Zen2..=X86Microarchitecture::Zen4 => {
                CacheBlockingFactors {
                    l1_block_size: 24 * 1024,  // 32KB for L1D
                    l2_block_size: 384 * 1024, // 512KB for L2
                    l3_block_size: 8 * 1024 * 1024, // 8MB+ for L3
                    tlb_block_size: 2 * 1024 * 1024, // 2MB pages
                }
            }
            _ => {
                CacheBlockingFactors {
                    l1_block_size: 12 * 1024,
                    l2_block_size: 96 * 1024,
                    l3_block_size: 2 * 1024 * 1024,
                    tlb_block_size: 2 * 1024 * 1024,
                }
            }
        };
        
        let parallel_params = ParallelExecutionParams {
            optimal_thread_count: num_cpus::get(),
            work_stealing_threshold: 1000,
            chunk_size_multiplier: match microarch {
                X86Microarchitecture::AlderLake | X86Microarchitecture::RaptorLake => 1.5, // Hybrid architecture
                _ => 1.0,
            },
            load_balancing_strategy: LoadBalancingStrategy::NUMA_Aware,
        };
        
        OptimizationParameters {
            loop_unroll_factor: loop_unroll,
            preferred_vector_width: vector_width,
            max_memory_bandwidth_utilization: memory_bw,
            scheduling_window_size: sched_window,
            branch_prediction_threshold: branch_thresh,
            cache_blocking_factors: cache_blocking,
            parallel_params,
        }
    }
    
    /// Get vector unit configuration
    fn get_vector_unit_config(microarch: &X86Microarchitecture, features: &CpuFeatures) -> VectorUnitConfig {
        match microarch {
            X86Microarchitecture::Haswell | X86Microarchitecture::Broadwell => {
                VectorUnitConfig {
                    avx512_available: false,
                    avx2_optimal: true,
                    fma_units: 2,
                    vector_register_count: 16,
                    optimal_vector_size: 256,
                    mask_register_count: 0,
                    memory_units: 2,
                }
            }
            X86Microarchitecture::Skylake..=X86Microarchitecture::RaptorLake => {
                VectorUnitConfig {
                    avx512_available: features.avx512f,
                    avx2_optimal: !features.avx512f, // AVX-512 can be slower on some workloads
                    fma_units: 2,
                    vector_register_count: if features.avx512f { 32 } else { 16 },
                    optimal_vector_size: if features.avx512f { 512 } else { 256 },
                    mask_register_count: if features.avx512f { 8 } else { 0 },
                    memory_units: 2,
                }
            }
            X86Microarchitecture::Zen2..=X86Microarchitecture::Zen4 => {
                VectorUnitConfig {
                    avx512_available: microarch == &X86Microarchitecture::Zen4 && features.avx512f,
                    avx2_optimal: true,
                    fma_units: 2,
                    vector_register_count: if features.avx512f { 32 } else { 16 },
                    optimal_vector_size: if features.avx512f { 512 } else { 256 },
                    mask_register_count: if features.avx512f { 8 } else { 0 },
                    memory_units: 3, // AMD has wider memory subsystem
                }
            }
            _ => {
                VectorUnitConfig {
                    avx512_available: false,
                    avx2_optimal: features.avx2,
                    fma_units: 1,
                    vector_register_count: 16,
                    optimal_vector_size: if features.avx2 { 256 } else { 128 },
                    mask_register_count: 0,
                    memory_units: 1,
                }
            }
        }
    }
    
    /// Detect cache configuration
    fn detect_cache_configuration(microarch: &X86Microarchitecture) -> BackendResult<CacheConfiguration> {
        // These values are based on known microarchitecture specifications
        let config = match microarch {
            X86Microarchitecture::Haswell | X86Microarchitecture::Broadwell => {
                CacheConfiguration {
                    l1i_size: 32 * 1024,
                    l1d_size: 32 * 1024,
                    l2_size: 256 * 1024,
                    l3_size: 8 * 1024 * 1024,
                    l1_associativity: 8,
                    l2_associativity: 8,
                    l3_associativity: 16,
                    cache_line_size: 64,
                    tlb_entries: 64,
                    prefetch_distance: 128,
                }
            }
            X86Microarchitecture::Skylake..=X86Microarchitecture::RaptorLake => {
                CacheConfiguration {
                    l1i_size: 32 * 1024,
                    l1d_size: 32 * 1024,
                    l2_size: 256 * 1024,
                    l3_size: 12 * 1024 * 1024, // Varies by SKU
                    l1_associativity: 8,
                    l2_associativity: 4,
                    l3_associativity: 12,
                    cache_line_size: 64,
                    tlb_entries: 64,
                    prefetch_distance: 192,
                }
            }
            X86Microarchitecture::Zen2 | X86Microarchitecture::Zen3 => {
                CacheConfiguration {
                    l1i_size: 32 * 1024,
                    l1d_size: 32 * 1024,
                    l2_size: 512 * 1024,
                    l3_size: 16 * 1024 * 1024, // Per CCX
                    l1_associativity: 8,
                    l2_associativity: 8,
                    l3_associativity: 16,
                    cache_line_size: 64,
                    tlb_entries: 64,
                    prefetch_distance: 256,
                }
            }
            X86Microarchitecture::Zen4 => {
                CacheConfiguration {
                    l1i_size: 32 * 1024,
                    l1d_size: 32 * 1024,
                    l2_size: 1024 * 1024,
                    l3_size: 32 * 1024 * 1024,
                    l1_associativity: 8,
                    l2_associativity: 8,
                    l3_associativity: 16,
                    cache_line_size: 64,
                    tlb_entries: 128,
                    prefetch_distance: 512,
                }
            }
            _ => {
                // Conservative defaults
                CacheConfiguration {
                    l1i_size: 32 * 1024,
                    l1d_size: 32 * 1024,
                    l2_size: 256 * 1024,
                    l3_size: 4 * 1024 * 1024,
                    l1_associativity: 4,
                    l2_associativity: 8,
                    l3_associativity: 12,
                    cache_line_size: 64,
                    tlb_entries: 64,
                    prefetch_distance: 64,
                }
            }
        };
        
        Ok(config)
    }
    
    /// Get execution unit information
    fn get_execution_unit_info(microarch: &X86Microarchitecture) -> ExecutionUnitInfo {
        match microarch {
            X86Microarchitecture::Haswell | X86Microarchitecture::Broadwell => {
                ExecutionUnitInfo {
                    integer_units: 4,
                    fp_units: 3,
                    vector_units: 2,
                    load_units: 2,
                    store_units: 1,
                    branch_units: 1,
                    issue_width: 4,
                    retire_width: 4,
                }
            }
            X86Microarchitecture::Skylake..=X86Microarchitecture::RaptorLake => {
                ExecutionUnitInfo {
                    integer_units: 4,
                    fp_units: 3,
                    vector_units: 2,
                    load_units: 2,
                    store_units: 1,
                    branch_units: 1,
                    issue_width: 4,
                    retire_width: 4,
                }
            }
            X86Microarchitecture::Zen2 | X86Microarchitecture::Zen3 => {
                ExecutionUnitInfo {
                    integer_units: 4,
                    fp_units: 4,
                    vector_units: 2,
                    load_units: 3,
                    store_units: 2,
                    branch_units: 1,
                    issue_width: 6,
                    retire_width: 6,
                }
            }
            X86Microarchitecture::Zen4 => {
                ExecutionUnitInfo {
                    integer_units: 4,
                    fp_units: 4,
                    vector_units: 2,
                    load_units: 3,
                    store_units: 2,
                    branch_units: 1,
                    issue_width: 6,
                    retire_width: 6,
                }
            }
            _ => {
                ExecutionUnitInfo {
                    integer_units: 2,
                    fp_units: 2,
                    vector_units: 1,
                    load_units: 1,
                    store_units: 1,
                    branch_units: 1,
                    issue_width: 2,
                    retire_width: 2,
                }
            }
        }
    }
    
    /// Get optimal parameters for matrix multiplication
    pub fn get_matmul_params(&self, m: usize, n: usize, k: usize) -> MatmulParams {
        let vector_width = self.vector_unit_config.optimal_vector_size / 32; // 32-bit elements
        let cache_blocking = &self.optimization_params.cache_blocking_factors;
        
        // Calculate blocking factors based on cache sizes and access patterns
        let (block_m, block_n, block_k) = if self.features.avx512f && m * n * k > 1_000_000 {
            // Large matrices with AVX-512
            (
                (cache_blocking.l1_block_size / (k * 4)).min(256).max(32),
                (cache_blocking.l2_block_size / (m * 4)).min(512).max(64),
                (cache_blocking.l3_block_size / (m * n * 4)).min(1024).max(128),
            )
        } else if self.features.avx2 {
            // Medium matrices with AVX2
            (
                (cache_blocking.l1_block_size / (k * 4)).min(128).max(16),
                (cache_blocking.l2_block_size / (m * 4)).min(256).max(32),
                (cache_blocking.l3_block_size / (m * n * 4)).min(512).max(64),
            )
        } else {
            // Small matrices or basic SIMD
            (
                (cache_blocking.l1_block_size / (k * 4)).min(64).max(8),
                (cache_blocking.l2_block_size / (m * 4)).min(128).max(16),
                (cache_blocking.l3_block_size / (m * n * 4)).min(256).max(32),
            )
        };
        
        MatmulParams {
            block_m,
            block_n,
            block_k,
            vector_width,
            unroll_factor: self.optimization_params.loop_unroll_factor,
            use_fma: self.features.fma,
            prefer_avx512: self.vector_unit_config.avx512_available && m * n * k > 100_000,
        }
    }
    
    /// Get optimal parameters for convolution
    pub fn get_conv_params(&self, batch: usize, channels: usize, height: usize, width: usize) -> ConvParams {
        let total_elements = batch * channels * height * width;
        let vector_width = self.vector_unit_config.optimal_vector_size / 32;
        
        let (tile_h, tile_w, unroll_channels) = match self.microarch {
            X86Microarchitecture::Haswell..=X86Microarchitecture::RaptorLake => {
                if total_elements > 1_000_000 {
                    (16, 16, 8) // Large images
                } else {
                    (8, 8, 4) // Small images
                }
            }
            X86Microarchitecture::Zen2..=X86Microarchitecture::Zen4 => {
                if total_elements > 1_000_000 {
                    (16, 16, 16) // AMD has better memory bandwidth
                } else {
                    (8, 8, 8)
                }
            }
            _ => (4, 4, 2),
        };
        
        ConvParams {
            tile_height: tile_h,
            tile_width: tile_w,
            channel_unroll: unroll_channels,
            vector_width,
            use_fma: self.features.fma,
            winograd_threshold: 64, // Use Winograd for kernels >= 3x3 on large feature maps
        }
    }
    
    /// Get microarchitecture information
    pub fn get_microarch_info(&self) -> MicroarchInfo {
        MicroarchInfo {
            name: format!("{:?}", self.microarch),
            vendor: if matches!(self.microarch, 
                X86Microarchitecture::Zen..=X86Microarchitecture::Zen4 |
                X86Microarchitecture::Bulldozer..=X86Microarchitecture::Excavator) {
                "AMD".to_string()
            } else {
                "Intel".to_string()
            },
            features: self.features,
            optimization_params: self.optimization_params.clone(),
            cache_config: self.cache_config.clone(),
        }
    }
}

/// Matrix multiplication optimization parameters
#[derive(Debug, Clone)]
pub struct MatmulParams {
    pub block_m: usize,
    pub block_n: usize,
    pub block_k: usize,
    pub vector_width: usize,
    pub unroll_factor: usize,
    pub use_fma: bool,
    pub prefer_avx512: bool,
}

/// Convolution optimization parameters
#[derive(Debug, Clone)]
pub struct ConvParams {
    pub tile_height: usize,
    pub tile_width: usize,
    pub channel_unroll: usize,
    pub vector_width: usize,
    pub use_fma: bool,
    pub winograd_threshold: usize,
}

/// Microarchitecture information
#[derive(Debug, Clone)]
pub struct MicroarchInfo {
    pub name: String,
    pub vendor: String,
    pub features: CpuFeatures,
    pub optimization_params: OptimizationParameters,
    pub cache_config: CacheConfiguration,
}

/// Global optimizer instance
static GLOBAL_OPTIMIZER: OnceLock<EnhancedX86Optimizer> = OnceLock::new();

/// Get the global optimizer instance
pub fn get_optimizer() -> &'static EnhancedX86Optimizer {
    GLOBAL_OPTIMIZER.get_or_init(|| {
        EnhancedX86Optimizer::new().unwrap_or_else(|_| {
            // Fallback configuration
            EnhancedX86Optimizer {
                microarch: X86Microarchitecture::Unknown,
                features: CpuFeatures::default(),
                optimization_params: OptimizationParameters {
                    loop_unroll_factor: 4,
                    preferred_vector_width: 128,
                    max_memory_bandwidth_utilization: 0.8,
                    scheduling_window_size: 128,
                    branch_prediction_threshold: 0.9,
                    cache_blocking_factors: CacheBlockingFactors {
                        l1_block_size: 16 * 1024,
                        l2_block_size: 256 * 1024,
                        l3_block_size: 4 * 1024 * 1024,
                        tlb_block_size: 2 * 1024 * 1024,
                    },
                    parallel_params: ParallelExecutionParams {
                        optimal_thread_count: num_cpus::get(),
                        work_stealing_threshold: 1000,
                        chunk_size_multiplier: 1.0,
                        load_balancing_strategy: LoadBalancingStrategy::Dynamic,
                    },
                },
                vector_unit_config: VectorUnitConfig {
                    avx512_available: false,
                    avx2_optimal: false,
                    fma_units: 1,
                    vector_register_count: 16,
                    optimal_vector_size: 128,
                    mask_register_count: 0,
                    memory_units: 1,
                },
                cache_config: CacheConfiguration {
                    l1i_size: 32 * 1024,
                    l1d_size: 32 * 1024,
                    l2_size: 256 * 1024,
                    l3_size: 4 * 1024 * 1024,
                    l1_associativity: 4,
                    l2_associativity: 8,
                    l3_associativity: 12,
                    cache_line_size: 64,
                    tlb_entries: 64,
                    prefetch_distance: 64,
                },
                execution_units: ExecutionUnitInfo {
                    integer_units: 2,
                    fp_units: 2,
                    vector_units: 1,
                    load_units: 1,
                    store_units: 1,
                    branch_units: 1,
                    issue_width: 2,
                    retire_width: 2,
                },
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_optimizer_creation() {
        let optimizer = EnhancedX86Optimizer::new();
        assert!(optimizer.is_ok());
    }
    
    #[test]
    fn test_feature_detection() {
        let features = EnhancedX86Optimizer::detect_cpu_features();
        assert!(features.is_ok());
        
        // At minimum, SSE2 should be available on x86_64
        #[cfg(target_arch = "x86_64")]
        {
            let features = features.unwrap();
            assert!(features.sse2);
        }
    }
    
    #[test]
    fn test_matmul_params() {
        let optimizer = get_optimizer();
        let params = optimizer.get_matmul_params(128, 128, 128);
        
        assert!(params.block_m > 0);
        assert!(params.block_n > 0);
        assert!(params.block_k > 0);
        assert!(params.vector_width > 0);
    }
    
    #[test]
    fn test_conv_params() {
        let optimizer = get_optimizer();
        let params = optimizer.get_conv_params(1, 32, 224, 224);
        
        assert!(params.tile_height > 0);
        assert!(params.tile_width > 0);
        assert!(params.channel_unroll > 0);
        assert!(params.vector_width > 0);
    }
    
    #[test]
    fn test_microarch_info() {
        let optimizer = get_optimizer();
        let info = optimizer.get_microarch_info();
        
        assert!(!info.name.is_empty());
        assert!(!info.vendor.is_empty());
    }
    
    #[test]
    fn test_cache_configuration() {
        let optimizer = get_optimizer();
        let cache = &optimizer.cache_config;
        
        assert!(cache.l1d_size > 0);
        assert!(cache.l2_size > 0);
        assert!(cache.l3_size > 0);
        assert!(cache.cache_line_size > 0);
    }
}