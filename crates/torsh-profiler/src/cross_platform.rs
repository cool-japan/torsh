//! Cross-platform Profiling Support
//!
//! This module provides platform-specific profiling implementations for:
//! - ARM64 (AArch64) - Apple Silicon, ARM servers
//! - RISC-V - Emerging RISC architecture
//! - WebAssembly (WASM) - Browser and edge computing
//!
//! # Features
//!
//! - Platform detection and capability discovery
//! - Architecture-specific performance counters
//! - Timer implementations with fallbacks
//! - Memory profiling adapters
//! - Portable profiling API

use serde::{Deserialize, Serialize};
use std::time::Instant;
use torsh_core::{Result as TorshResult, TorshError};

/// Platform architecture types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlatformArch {
    X86_64,
    ARM64,
    RISCV64,
    WASM32,
    WASM64,
    Unknown,
}

impl PlatformArch {
    /// Detect the current platform architecture
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        return Self::X86_64;

        #[cfg(target_arch = "aarch64")]
        return Self::ARM64;

        #[cfg(target_arch = "riscv64")]
        return Self::RISCV64;

        #[cfg(all(target_arch = "wasm32", not(target_pointer_width = "64")))]
        return Self::WASM32;

        #[cfg(all(target_arch = "wasm32", target_pointer_width = "64"))]
        return Self::WASM64;

        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64",
            target_arch = "wasm32"
        )))]
        return Self::Unknown;
    }

    /// Check if the platform is ARM-based
    pub fn is_arm(&self) -> bool {
        matches!(self, Self::ARM64)
    }

    /// Check if the platform is RISC-V
    pub fn is_riscv(&self) -> bool {
        matches!(self, Self::RISCV64)
    }

    /// Check if the platform is WebAssembly
    pub fn is_wasm(&self) -> bool {
        matches!(self, Self::WASM32 | Self::WASM64)
    }

    /// Check if the platform supports hardware performance counters
    pub fn supports_hardware_counters(&self) -> bool {
        matches!(self, Self::X86_64 | Self::ARM64 | Self::RISCV64)
    }
}

impl std::fmt::Display for PlatformArch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::X86_64 => write!(f, "x86_64"),
            Self::ARM64 => write!(f, "ARM64 (AArch64)"),
            Self::RISCV64 => write!(f, "RISC-V 64-bit"),
            Self::WASM32 => write!(f, "WebAssembly 32-bit"),
            Self::WASM64 => write!(f, "WebAssembly 64-bit"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Platform capabilities for profiling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCapabilities {
    pub arch: PlatformArch,
    pub has_rdtsc: bool,
    pub has_pmu: bool,
    pub has_simd: bool,
    pub simd_width: usize,
    pub cache_line_size: usize,
    pub supports_atomics: bool,
    pub supports_threads: bool,
    pub timer_resolution_ns: u64,
}

impl PlatformCapabilities {
    /// Detect platform capabilities
    pub fn detect() -> Self {
        let arch = PlatformArch::detect();

        let (has_rdtsc, has_pmu, has_simd, simd_width) = match arch {
            PlatformArch::X86_64 => (true, true, true, 256), // AVX2
            PlatformArch::ARM64 => (false, true, true, 128), // NEON
            PlatformArch::RISCV64 => (false, true, true, 128), // RVV
            PlatformArch::WASM32 | PlatformArch::WASM64 => (false, false, true, 128), // SIMD128
            PlatformArch::Unknown => (false, false, false, 0),
        };

        let cache_line_size = match arch {
            PlatformArch::X86_64 => 64,
            PlatformArch::ARM64 => 64,
            PlatformArch::RISCV64 => 64,
            PlatformArch::WASM32 | PlatformArch::WASM64 => 0, // No cache in WASM
            PlatformArch::Unknown => 64,
        };

        let supports_atomics = !arch.is_wasm(); // WASM atomics require SharedArrayBuffer
        let supports_threads = !arch.is_wasm(); // WASM threads require special features

        // Estimate timer resolution
        let timer_resolution_ns = match arch {
            PlatformArch::X86_64 => 1,   // RDTSC nanosecond precision
            PlatformArch::ARM64 => 10,   // ARM generic timer ~10ns
            PlatformArch::RISCV64 => 10, // RISC-V timer ~10ns
            PlatformArch::WASM32 | PlatformArch::WASM64 => 1000, // performance.now() ~1μs
            PlatformArch::Unknown => 1000,
        };

        Self {
            arch,
            has_rdtsc,
            has_pmu,
            has_simd,
            simd_width,
            cache_line_size,
            supports_atomics,
            supports_threads,
            timer_resolution_ns,
        }
    }
}

/// Cross-platform high-resolution timer
pub struct CrossPlatformTimer {
    start: Instant,
    capabilities: PlatformCapabilities,
}

impl CrossPlatformTimer {
    /// Create a new timer
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            capabilities: PlatformCapabilities::detect(),
        }
    }

    /// Start timing
    pub fn start(&mut self) {
        self.start = Instant::now();
    }

    /// Get elapsed time in microseconds
    pub fn elapsed_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }

    /// Get elapsed time in nanoseconds
    pub fn elapsed_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    /// Get platform-specific timestamp counter (if available)
    pub fn get_cycle_count(&self) -> Option<u64> {
        #[cfg(target_arch = "x86_64")]
        {
            // Use RDTSC on x86_64
            Some(unsafe { std::arch::x86_64::_rdtsc() })
        }

        #[cfg(target_arch = "aarch64")]
        {
            // ARM64 - use PMCCNTR_EL0 (requires privileged access)
            // Fallback to Instant-based timing
            None
        }

        #[cfg(target_arch = "riscv64")]
        {
            // RISC-V - use RDCYCLE
            // Note: Requires M-mode or counter delegation
            None
        }

        #[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
        {
            // WASM has no cycle counter
            None
        }

        #[cfg(not(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64",
            target_arch = "wasm32",
            target_arch = "wasm64"
        )))]
        {
            None
        }
    }

    /// Get platform capabilities
    pub fn capabilities(&self) -> &PlatformCapabilities {
        &self.capabilities
    }
}

impl Default for CrossPlatformTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// ARM64-specific profiling features
#[cfg(target_arch = "aarch64")]
pub mod arm64 {
    use super::*;

    /// ARM64 performance counter types
    #[derive(Debug, Clone, Copy)]
    pub enum ARM64Counter {
        CycleCount,
        InstructionCount,
        CacheMisses,
        BranchMisses,
        L1DCacheAccess,
        L1DCacheMiss,
        L2CacheAccess,
        L2CacheMiss,
    }

    /// ARM64 NEON SIMD information
    pub struct NeonInfo {
        pub available: bool,
        pub register_width: usize,
        pub num_registers: usize,
    }

    impl NeonInfo {
        pub fn detect() -> Self {
            Self {
                available: true, // NEON is mandatory in ARMv8-A
                register_width: 128,
                num_registers: 32,
            }
        }
    }

    /// Apple Silicon specific features
    #[cfg(target_os = "macos")]
    pub mod apple_silicon {
        /// Detect if running on Apple Silicon
        pub fn is_apple_silicon() -> bool {
            cfg!(all(target_arch = "aarch64", target_os = "macos"))
        }

        /// Get performance core count (P-cores)
        pub fn performance_core_count() -> usize {
            // This is a simplified detection
            // Real implementation would use sysctl
            num_cpus::get() / 2
        }

        /// Get efficiency core count (E-cores)
        pub fn efficiency_core_count() -> usize {
            num_cpus::get() / 2
        }
    }
}

/// RISC-V specific profiling features
#[cfg(target_arch = "riscv64")]
pub mod riscv {
    use super::*;

    /// RISC-V performance counter types
    #[derive(Debug, Clone, Copy)]
    pub enum RISCVCounter {
        CycleCount,
        InstructionCount,
        Time,
    }

    /// RISC-V Vector (RVV) extension information
    pub struct RVVInfo {
        pub available: bool,
        pub vlen: usize, // Vector register length in bits
    }

    impl RVVInfo {
        pub fn detect() -> Self {
            Self {
                available: false, // Detect via CSR or runtime check
                vlen: 0,
            }
        }
    }
}

/// WebAssembly specific profiling features
#[cfg(any(target_arch = "wasm32", target_arch = "wasm64"))]
pub mod wasm {
    use super::*;

    /// WASM runtime environment
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum WasmRuntime {
        Browser,
        Node,
        Wasmtime,
        Wasmer,
        Unknown,
    }

    impl WasmRuntime {
        pub fn detect() -> Self {
            // Detection would require JS interop
            Self::Unknown
        }
    }

    /// WASM SIMD support
    pub struct WasmSimdInfo {
        pub available: bool,
        pub simd128: bool,
    }

    impl WasmSimdInfo {
        pub fn detect() -> Self {
            Self {
                available: true,
                simd128: true, // Most modern WASM runtimes support SIMD128
            }
        }
    }

    /// WASM memory profiling (different from native)
    pub struct WasmMemoryProfiler {
        initial_pages: usize,
        max_pages: Option<usize>,
    }

    impl WasmMemoryProfiler {
        pub fn new() -> Self {
            Self {
                initial_pages: 256,     // Default 16MB (64KB per page)
                max_pages: Some(65536), // Maximum 4GB
            }
        }

        pub fn current_memory_pages(&self) -> usize {
            // Would use wasm memory.size instruction
            self.initial_pages
        }

        pub fn memory_bytes(&self) -> usize {
            self.current_memory_pages() * 65536 // 64KB per page
        }
    }

    impl Default for WasmMemoryProfiler {
        fn default() -> Self {
            Self::new()
        }
    }
}

/// Cross-platform profiler adapter
pub struct CrossPlatformProfiler {
    capabilities: PlatformCapabilities,
    timer: CrossPlatformTimer,
}

impl CrossPlatformProfiler {
    /// Create a new cross-platform profiler
    pub fn new() -> Self {
        Self {
            capabilities: PlatformCapabilities::detect(),
            timer: CrossPlatformTimer::new(),
        }
    }

    /// Get platform information
    pub fn platform_info(&self) -> String {
        format!(
            "Platform: {}\n\
             Hardware Counters: {}\n\
             SIMD: {} (width: {} bits)\n\
             Cache Line: {} bytes\n\
             Atomics: {}\n\
             Threads: {}\n\
             Timer Resolution: {} ns",
            self.capabilities.arch,
            if self.capabilities.has_pmu {
                "Yes"
            } else {
                "No"
            },
            if self.capabilities.has_simd {
                "Yes"
            } else {
                "No"
            },
            self.capabilities.simd_width,
            self.capabilities.cache_line_size,
            if self.capabilities.supports_atomics {
                "Yes"
            } else {
                "No"
            },
            if self.capabilities.supports_threads {
                "Yes"
            } else {
                "No"
            },
            self.capabilities.timer_resolution_ns
        )
    }

    /// Start profiling
    pub fn start(&mut self) {
        self.timer.start();
    }

    /// Stop profiling and get elapsed time
    pub fn stop(&self) -> u64 {
        self.timer.elapsed_us()
    }

    /// Get capabilities
    pub fn capabilities(&self) -> &PlatformCapabilities {
        &self.capabilities
    }

    /// Check if running on a specific architecture
    pub fn is_architecture(&self, arch: PlatformArch) -> bool {
        self.capabilities.arch == arch
    }

    /// Get recommended profiling strategy for the platform
    pub fn recommended_strategy(&self) -> ProfilingStrategy {
        match self.capabilities.arch {
            PlatformArch::X86_64 => ProfilingStrategy::HardwareCounters,
            PlatformArch::ARM64 => ProfilingStrategy::Hybrid,
            PlatformArch::RISCV64 => ProfilingStrategy::Sampling,
            PlatformArch::WASM32 | PlatformArch::WASM64 => ProfilingStrategy::Lightweight,
            PlatformArch::Unknown => ProfilingStrategy::Basic,
        }
    }
}

impl Default for CrossPlatformProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Recommended profiling strategy based on platform
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfilingStrategy {
    /// Use hardware performance counters (x86_64)
    HardwareCounters,
    /// Hybrid approach (ARM64)
    Hybrid,
    /// Sampling-based profiling (RISC-V)
    Sampling,
    /// Lightweight instrumentation (WASM)
    Lightweight,
    /// Basic timing only
    Basic,
}

impl std::fmt::Display for ProfilingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HardwareCounters => write!(f, "Hardware Counters"),
            Self::Hybrid => write!(f, "Hybrid"),
            Self::Sampling => write!(f, "Sampling"),
            Self::Lightweight => write!(f, "Lightweight"),
            Self::Basic => write!(f, "Basic"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_detection() {
        let arch = PlatformArch::detect();
        println!("Detected architecture: {}", arch);
        assert_ne!(arch, PlatformArch::Unknown);
    }

    #[test]
    fn test_capabilities_detection() {
        let caps = PlatformCapabilities::detect();
        println!("Platform capabilities:");
        println!("  Architecture: {}", caps.arch);
        println!("  RDTSC: {}", caps.has_rdtsc);
        println!("  PMU: {}", caps.has_pmu);
        println!("  SIMD: {} (width: {})", caps.has_simd, caps.simd_width);
        println!("  Cache line: {} bytes", caps.cache_line_size);
        println!("  Atomics: {}", caps.supports_atomics);
        println!("  Threads: {}", caps.supports_threads);
        println!("  Timer resolution: {} ns", caps.timer_resolution_ns);

        assert!(caps.cache_line_size > 0 || caps.arch.is_wasm());
    }

    #[test]
    fn test_cross_platform_timer() {
        let mut timer = CrossPlatformTimer::new();
        timer.start();

        // Simulate some work
        std::thread::sleep(std::time::Duration::from_micros(100));

        let elapsed = timer.elapsed_us();
        println!("Elapsed time: {} μs", elapsed);
        assert!(elapsed >= 100);
    }

    #[test]
    fn test_cross_platform_profiler() {
        let mut profiler = CrossPlatformProfiler::new();
        println!("{}", profiler.platform_info());

        profiler.start();
        std::thread::sleep(std::time::Duration::from_micros(100));
        let elapsed = profiler.stop();

        println!("Profiled time: {} μs", elapsed);
        assert!(elapsed >= 100);

        let strategy = profiler.recommended_strategy();
        println!("Recommended strategy: {}", strategy);
    }

    #[test]
    fn test_architecture_checks() {
        let profiler = CrossPlatformProfiler::new();
        let arch = profiler.capabilities().arch;

        assert_eq!(profiler.is_architecture(arch), true);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_arm64_neon() {
        let neon = arm64::NeonInfo::detect();
        assert!(neon.available);
        assert_eq!(neon.register_width, 128);
        println!(
            "NEON: {} registers of {} bits",
            neon.num_registers, neon.register_width
        );
    }

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn test_apple_silicon() {
        if arm64::apple_silicon::is_apple_silicon() {
            println!("Running on Apple Silicon");
            println!(
                "P-cores: {}",
                arm64::apple_silicon::performance_core_count()
            );
            println!("E-cores: {}", arm64::apple_silicon::efficiency_core_count());
        }
    }
}
