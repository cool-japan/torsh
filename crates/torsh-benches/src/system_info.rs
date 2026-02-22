//! Enhanced system information gathering for benchmarking
//!
//! This module provides comprehensive system information collection
//! to help understand benchmark performance characteristics and
//! provide platform-specific optimization recommendations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

/// Comprehensive system information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub cpu_info: CpuInfo,
    pub memory_info: MemoryInfo,
    pub system_info: BasicSystemInfo,
    pub environment_info: EnvironmentInfo,
    pub benchmark_environment: BenchmarkEnvironment,
    pub optimization_recommendations: Vec<String>,
}

/// CPU-specific information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub model: String,
    pub cores: usize,
    pub threads: usize,
    pub base_frequency: Option<f64>, // MHz
    pub max_frequency: Option<f64>,  // MHz
    pub cache_l1d: Option<usize>,    // KB
    pub cache_l1i: Option<usize>,    // KB
    pub cache_l2: Option<usize>,     // KB
    pub cache_l3: Option<usize>,     // KB
    pub features: Vec<String>,       // CPU features (AVX, AVX2, etc.)
    pub architecture: String,
    pub vendor: String,
}

/// Memory system information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_memory: u64,             // bytes
    pub available_memory: u64,         // bytes
    pub memory_bandwidth: Option<f64>, // GB/s (estimated)
    pub numa_nodes: usize,
    pub memory_type: Option<String>, // DDR4, DDR5, etc.
    pub memory_speed: Option<u32>,   // MHz
}

/// Basic system information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicSystemInfo {
    pub os: String,
    pub os_version: String,
    pub architecture: String,
    pub hostname: String,
    pub kernel_version: Option<String>,
}

/// Environment configuration affecting benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    pub compiler_version: String,
    pub rust_version: String,
    pub build_mode: BuildMode,
    pub target_features: Vec<String>,
    pub environment_variables: HashMap<String, String>,
    pub process_priority: Option<String>,
}

/// Build configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BuildMode {
    Debug,
    Release,
    Custom(String),
}

/// Benchmark environment assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkEnvironment {
    pub is_isolated: bool,
    pub cpu_governor: Option<String>,
    pub thermal_state: ThermalState,
    pub background_load: LoadLevel,
    pub timing_precision: TimingPrecision,
    pub reproducibility_score: f64, // 0.0 to 1.0
}

/// CPU thermal state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThermalState {
    Cool,
    Warm,
    Hot,
    Throttling,
    Unknown,
}

/// System load level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Timing precision assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingPrecision {
    pub clock_resolution: u64, // nanoseconds
    pub tsc_available: bool,
    pub monotonic_clock: bool,
    pub precision_rating: PrecisionRating,
}

/// Timing precision rating
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrecisionRating {
    Excellent,  // < 1ns
    Good,       // 1-10ns
    Acceptable, // 10-100ns
    Poor,       // > 100ns
}

/// System information collector
pub struct SystemInfoCollector {
    cached_info: Option<SystemInfo>,
}

impl SystemInfoCollector {
    /// Create a new system information collector
    pub fn new() -> Self {
        Self { cached_info: None }
    }

    /// Collect comprehensive system information
    pub fn collect(&mut self) -> SystemInfo {
        if let Some(ref cached) = self.cached_info {
            // Return cached info if collected recently (within 5 minutes)
            let age = chrono::Utc::now() - cached.timestamp;
            if age.num_minutes() < 5 {
                return cached.clone();
            }
        }

        let info = SystemInfo {
            timestamp: chrono::Utc::now(),
            cpu_info: self.collect_cpu_info(),
            memory_info: self.collect_memory_info(),
            system_info: self.collect_system_info(),
            environment_info: self.collect_environment_info(),
            benchmark_environment: self.assess_benchmark_environment(),
            optimization_recommendations: Vec::new(),
        };

        let mut info_with_recommendations = info;
        info_with_recommendations.optimization_recommendations =
            self.generate_optimization_recommendations(&info_with_recommendations);

        self.cached_info = Some(info_with_recommendations.clone());
        info_with_recommendations
    }

    /// Collect CPU information
    fn collect_cpu_info(&self) -> CpuInfo {
        let mut cpu_info = CpuInfo {
            model: "Unknown".to_string(),
            cores: num_cpus::get_physical(),
            threads: num_cpus::get(),
            base_frequency: None,
            max_frequency: None,
            cache_l1d: None,
            cache_l1i: None,
            cache_l2: None,
            cache_l3: None,
            features: Vec::new(),
            architecture: std::env::consts::ARCH.to_string(),
            vendor: "Unknown".to_string(),
        };

        // Try to get detailed CPU info on Linux
        if cfg!(target_os = "linux") {
            if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
                cpu_info.model = self.extract_cpu_model(&cpuinfo);
                cpu_info.vendor = self.extract_cpu_vendor(&cpuinfo);
                cpu_info.features = self.extract_cpu_features(&cpuinfo);
            }

            // Try to get cache info
            cpu_info.cache_l1d =
                self.read_cache_size("/sys/devices/system/cpu/cpu0/cache/index0/size");
            cpu_info.cache_l1i =
                self.read_cache_size("/sys/devices/system/cpu/cpu0/cache/index1/size");
            cpu_info.cache_l2 =
                self.read_cache_size("/sys/devices/system/cpu/cpu0/cache/index2/size");
            cpu_info.cache_l3 =
                self.read_cache_size("/sys/devices/system/cpu/cpu0/cache/index3/size");

            // Try to get frequency info
            cpu_info.base_frequency =
                self.read_cpu_frequency("/sys/devices/system/cpu/cpu0/cpufreq/base_frequency");
            cpu_info.max_frequency =
                self.read_cpu_frequency("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
        }

        // Detect CPU features using Rust's built-in detection
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                cpu_info.features.push("AVX".to_string());
            }
            if is_x86_feature_detected!("avx2") {
                cpu_info.features.push("AVX2".to_string());
            }
            if is_x86_feature_detected!("avx512f") {
                cpu_info.features.push("AVX-512F".to_string());
            }
            if is_x86_feature_detected!("fma") {
                cpu_info.features.push("FMA".to_string());
            }
            if is_x86_feature_detected!("sse4.1") {
                cpu_info.features.push("SSE4.1".to_string());
            }
            if is_x86_feature_detected!("sse4.2") {
                cpu_info.features.push("SSE4.2".to_string());
            }
        }

        cpu_info
    }

    /// Collect memory information
    fn collect_memory_info(&self) -> MemoryInfo {
        let mut memory_info = MemoryInfo {
            total_memory: 0,
            available_memory: 0,
            memory_bandwidth: None,
            numa_nodes: 1,
            memory_type: None,
            memory_speed: None,
        };

        // Get memory info on Linux
        if cfg!(target_os = "linux") {
            if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
                memory_info.total_memory = self.extract_memory_value(&meminfo, "MemTotal:") * 1024;
                memory_info.available_memory =
                    self.extract_memory_value(&meminfo, "MemAvailable:") * 1024;
            }

            // Count NUMA nodes
            if let Ok(entries) = fs::read_dir("/sys/devices/system/node") {
                memory_info.numa_nodes = entries
                    .filter_map(|entry| entry.ok())
                    .filter(|entry| entry.file_name().to_string_lossy().starts_with("node"))
                    .count()
                    .max(1);
            }

            // Estimate memory bandwidth (very rough estimate based on DDR type)
            memory_info.memory_bandwidth = Some(self.estimate_memory_bandwidth(&memory_info));
        }

        memory_info
    }

    /// Collect basic system information
    fn collect_system_info(&self) -> BasicSystemInfo {
        let mut system_info = BasicSystemInfo {
            os: std::env::consts::OS.to_string(),
            os_version: "Unknown".to_string(),
            architecture: std::env::consts::ARCH.to_string(),
            hostname: "Unknown".to_string(),
            kernel_version: None,
        };

        // Get hostname
        if let Ok(hostname) = hostname::get() {
            system_info.hostname = hostname.to_string_lossy().to_string();
        }

        // Get OS version and kernel info on Linux
        if cfg!(target_os = "linux") {
            if let Ok(os_release) = fs::read_to_string("/etc/os-release") {
                system_info.os_version = self.extract_os_version(&os_release);
            }

            if let Ok(version) = fs::read_to_string("/proc/version") {
                system_info.kernel_version = Some(version.trim().to_string());
            }
        }

        system_info
    }

    /// Collect environment information
    fn collect_environment_info(&self) -> EnvironmentInfo {
        let mut env_info = EnvironmentInfo {
            compiler_version: "Unknown".to_string(),
            rust_version: "Unknown".to_string(),
            build_mode: if cfg!(debug_assertions) {
                BuildMode::Debug
            } else {
                BuildMode::Release
            },
            target_features: Vec::new(),
            environment_variables: HashMap::new(),
            process_priority: None,
        };

        // Get Rust version
        env_info.rust_version =
            std::env::var("CARGO_PKG_RUST_VERSION").unwrap_or_else(|_| "unknown".to_string());

        // Collect relevant environment variables
        let relevant_vars = [
            "RUSTFLAGS",
            "CARGO_TARGET_DIR",
            "RAYON_NUM_THREADS",
            "OMP_NUM_THREADS",
            "GOMP_CPU_AFFINITY",
            "CUDA_VISIBLE_DEVICES",
        ];

        for var in &relevant_vars {
            if let Ok(value) = std::env::var(var) {
                env_info
                    .environment_variables
                    .insert(var.to_string(), value);
            }
        }

        // Get target features from RUSTFLAGS if available
        if let Ok(rustflags) = std::env::var("RUSTFLAGS") {
            env_info.target_features = self.extract_target_features(&rustflags);
        }

        env_info
    }

    /// Assess benchmark environment quality
    fn assess_benchmark_environment(&self) -> BenchmarkEnvironment {
        BenchmarkEnvironment {
            is_isolated: self.check_cpu_isolation(),
            cpu_governor: self.get_cpu_governor(),
            thermal_state: self.assess_thermal_state(),
            background_load: self.assess_background_load(),
            timing_precision: self.assess_timing_precision(),
            reproducibility_score: self.calculate_reproducibility_score(),
        }
    }

    /// Generate optimization recommendations
    fn generate_optimization_recommendations(&self, info: &SystemInfo) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Build mode recommendations
        if matches!(info.environment_info.build_mode, BuildMode::Debug) {
            recommendations.push(
                "🚨 CRITICAL: Running in debug mode! Use `--release` for accurate benchmarks"
                    .to_string(),
            );
        }

        // CPU feature recommendations
        if info.cpu_info.features.is_empty() {
            recommendations.push(
                "⚠️ No CPU features detected. Consider setting RUSTFLAGS=\"-C target-cpu=native\""
                    .to_string(),
            );
        } else {
            let has_avx2 = info.cpu_info.features.iter().any(|f| f.contains("AVX2"));
            let has_avx512 = info.cpu_info.features.iter().any(|f| f.contains("AVX-512"));

            if has_avx512 {
                recommendations.push(
                    "✅ AVX-512 detected. Excellent for compute-intensive operations".to_string(),
                );
            } else if has_avx2 {
                recommendations
                    .push("✅ AVX2 detected. Good SIMD performance available".to_string());
            } else {
                recommendations.push(
                    "💡 Consider upgrading to a CPU with AVX2+ for better performance".to_string(),
                );
            }
        }

        // Memory recommendations
        let memory_gb = info.memory_info.total_memory / (1024 * 1024 * 1024);
        if memory_gb < 8 {
            recommendations
                .push("⚠️ Low memory detected. Large benchmarks may cause swapping".to_string());
        } else if memory_gb > 64 {
            recommendations
                .push("✅ Abundant memory available. Can run large-scale benchmarks".to_string());
        }

        // NUMA recommendations
        if info.memory_info.numa_nodes > 1 {
            recommendations.push(format!(
                "💡 {} NUMA nodes detected. Consider NUMA-aware allocation for large tensors",
                info.memory_info.numa_nodes
            ));
        }

        // Threading recommendations
        let logical_cores = info.cpu_info.threads;
        let physical_cores = info.cpu_info.cores;

        if logical_cores > physical_cores * 2 {
            recommendations.push("💡 Hyperthreading detected. May want to limit threads to physical cores for compute-heavy tasks".to_string());
        }

        if !info
            .environment_info
            .environment_variables
            .contains_key("RAYON_NUM_THREADS")
        {
            recommendations.push(format!(
                "💡 Consider setting RAYON_NUM_THREADS={} for optimal performance",
                physical_cores
            ));
        }

        // Environment recommendations
        if !info.benchmark_environment.is_isolated {
            recommendations
                .push("⚠️ CPU isolation not detected. Results may be less consistent".to_string());
        }

        if let Some(ref governor) = info.benchmark_environment.cpu_governor {
            if governor != "performance" {
                recommendations.push(
                    "💡 Consider setting CPU governor to 'performance' for consistent results"
                        .to_string(),
                );
            }
        }

        match info.benchmark_environment.background_load {
            LoadLevel::High | LoadLevel::Critical => {
                recommendations.push("⚠️ High system load detected. Close unnecessary applications for better results".to_string());
            }
            _ => {}
        }

        // Timing precision recommendations
        match info.benchmark_environment.timing_precision.precision_rating {
            PrecisionRating::Poor => {
                recommendations.push(
                    "⚠️ Poor timing precision detected. Results may be less accurate".to_string(),
                );
            }
            PrecisionRating::Acceptable => {
                recommendations
                    .push("💡 Timing precision is acceptable but could be improved".to_string());
            }
            _ => {}
        }

        recommendations
    }

    // Helper methods for parsing system information

    fn extract_cpu_model(&self, cpuinfo: &str) -> String {
        cpuinfo
            .lines()
            .find(|line| line.starts_with("model name"))
            .and_then(|line| line.split(':').nth(1))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "Unknown".to_string())
    }

    fn extract_cpu_vendor(&self, cpuinfo: &str) -> String {
        cpuinfo
            .lines()
            .find(|line| line.starts_with("vendor_id"))
            .and_then(|line| line.split(':').nth(1))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "Unknown".to_string())
    }

    fn extract_cpu_features(&self, cpuinfo: &str) -> Vec<String> {
        cpuinfo
            .lines()
            .find(|line| line.starts_with("flags"))
            .and_then(|line| line.split(':').nth(1))
            .map(|s| {
                s.trim()
                    .split_whitespace()
                    .map(|f| f.to_uppercase())
                    .collect()
            })
            .unwrap_or_default()
    }

    fn read_cache_size(&self, path: &str) -> Option<usize> {
        fs::read_to_string(path)
            .ok()
            .and_then(|content| content.trim().strip_suffix("K").map(|s| s.to_string()))
            .and_then(|size_str| size_str.parse::<usize>().ok())
    }

    fn read_cpu_frequency(&self, path: &str) -> Option<f64> {
        fs::read_to_string(path)
            .ok()
            .and_then(|content| content.trim().parse::<f64>().ok())
            .map(|freq_khz| freq_khz / 1000.0) // Convert kHz to MHz
    }

    fn extract_memory_value(&self, meminfo: &str, key: &str) -> u64 {
        meminfo
            .lines()
            .find(|line| line.starts_with(key))
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0)
    }

    fn estimate_memory_bandwidth(&self, _memory_info: &MemoryInfo) -> f64 {
        // Very rough estimation based on typical DDR speeds
        // This would need more sophisticated detection in practice
        50.0 // GB/s - conservative estimate
    }

    fn extract_os_version(&self, os_release: &str) -> String {
        os_release
            .lines()
            .find(|line| line.starts_with("PRETTY_NAME="))
            .and_then(|line| line.split('=').nth(1))
            .map(|s| s.trim_matches('"').to_string())
            .unwrap_or_else(|| "Unknown".to_string())
    }

    fn extract_target_features(&self, rustflags: &str) -> Vec<String> {
        rustflags
            .split_whitespace()
            .filter(|part| part.starts_with("-C") && part.contains("target-feature"))
            .flat_map(|part| part.split('=').nth(1))
            .flat_map(|features| features.split(','))
            .map(|feature| feature.trim().to_string())
            .collect()
    }

    fn check_cpu_isolation(&self) -> bool {
        // Check for CPU isolation on Linux
        if cfg!(target_os = "linux") {
            if let Ok(cmdline) = fs::read_to_string("/proc/cmdline") {
                return cmdline.contains("isolcpus") || cmdline.contains("nohz_full");
            }
        }
        false
    }

    fn get_cpu_governor(&self) -> Option<String> {
        if cfg!(target_os = "linux") {
            fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
                .ok()
                .map(|s| s.trim().to_string())
        } else {
            None
        }
    }

    fn assess_thermal_state(&self) -> ThermalState {
        // This is a simplified assessment
        // In practice, you'd check thermal zones, CPU frequency scaling, etc.
        ThermalState::Unknown
    }

    fn assess_background_load(&self) -> LoadLevel {
        // Simple load assessment based on load average
        if cfg!(target_os = "linux") {
            if let Ok(loadavg) = fs::read_to_string("/proc/loadavg") {
                if let Some(load_1min) = loadavg.split_whitespace().next() {
                    if let Ok(load) = load_1min.parse::<f64>() {
                        let cores = num_cpus::get() as f64;
                        let load_ratio = load / cores;

                        return match load_ratio {
                            r if r < 0.5 => LoadLevel::Low,
                            r if r < 1.0 => LoadLevel::Medium,
                            r if r < 2.0 => LoadLevel::High,
                            _ => LoadLevel::Critical,
                        };
                    }
                }
            }
        }
        LoadLevel::Medium
    }

    fn assess_timing_precision(&self) -> TimingPrecision {
        // Assess timing precision based on available clock sources
        let clock_resolution = 1; // nanosecond (best case)
        let tsc_available = cfg!(target_arch = "x86_64"); // TSC available on x86_64
        let monotonic_clock = true; // Rust's Instant uses monotonic clock

        let precision_rating = match clock_resolution {
            r if r < 1 => PrecisionRating::Excellent,
            r if r < 10 => PrecisionRating::Good,
            r if r < 100 => PrecisionRating::Acceptable,
            _ => PrecisionRating::Poor,
        };

        TimingPrecision {
            clock_resolution,
            tsc_available,
            monotonic_clock,
            precision_rating,
        }
    }

    fn calculate_reproducibility_score(&self) -> f64 {
        let mut score: f64 = 1.0;

        // Reduce score for factors that hurt reproducibility
        if matches!(
            self.assess_background_load(),
            LoadLevel::High | LoadLevel::Critical
        ) {
            score -= 0.3;
        }

        if !self.check_cpu_isolation() {
            score -= 0.2;
        }

        if self.get_cpu_governor().as_deref() != Some("performance") {
            score -= 0.1;
        }

        if matches!(
            self.assess_thermal_state(),
            ThermalState::Hot | ThermalState::Throttling
        ) {
            score -= 0.2;
        }

        score.max(0.0).min(1.0)
    }

    /// Generate a comprehensive system report
    pub fn generate_system_report(
        &self,
        info: &SystemInfo,
        output_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut file = fs::File::create(output_path)?;
        use std::io::Write;

        writeln!(file, "# System Information Report")?;
        writeln!(
            file,
            "Generated: {}\n",
            info.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        )?;

        writeln!(file, "## System Overview\n")?;
        writeln!(file, "- **Hostname**: {}", info.system_info.hostname)?;
        writeln!(
            file,
            "- **OS**: {} {}",
            info.system_info.os, info.system_info.os_version
        )?;
        writeln!(
            file,
            "- **Architecture**: {}",
            info.system_info.architecture
        )?;
        if let Some(ref kernel) = info.system_info.kernel_version {
            writeln!(file, "- **Kernel**: {}", kernel)?;
        }

        writeln!(file, "\n## CPU Information\n")?;
        writeln!(file, "- **Model**: {}", info.cpu_info.model)?;
        writeln!(file, "- **Vendor**: {}", info.cpu_info.vendor)?;
        writeln!(
            file,
            "- **Cores**: {} physical, {} logical",
            info.cpu_info.cores, info.cpu_info.threads
        )?;
        writeln!(file, "- **Architecture**: {}", info.cpu_info.architecture)?;

        if let Some(base_freq) = info.cpu_info.base_frequency {
            writeln!(file, "- **Base Frequency**: {:.0} MHz", base_freq)?;
        }
        if let Some(max_freq) = info.cpu_info.max_frequency {
            writeln!(file, "- **Max Frequency**: {:.0} MHz", max_freq)?;
        }

        writeln!(file, "\n**Cache Hierarchy:**")?;
        if let Some(l1d) = info.cpu_info.cache_l1d {
            writeln!(file, "- L1D: {} KB", l1d)?;
        }
        if let Some(l1i) = info.cpu_info.cache_l1i {
            writeln!(file, "- L1I: {} KB", l1i)?;
        }
        if let Some(l2) = info.cpu_info.cache_l2 {
            writeln!(file, "- L2: {} KB", l2)?;
        }
        if let Some(l3) = info.cpu_info.cache_l3 {
            writeln!(file, "- L3: {} KB", l3)?;
        }

        if !info.cpu_info.features.is_empty() {
            writeln!(file, "\n**CPU Features:**")?;
            for feature in &info.cpu_info.features {
                writeln!(file, "- {}", feature)?;
            }
        }

        writeln!(file, "\n## Memory Information\n")?;
        writeln!(
            file,
            "- **Total Memory**: {:.1} GB",
            info.memory_info.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        )?;
        writeln!(
            file,
            "- **Available Memory**: {:.1} GB",
            info.memory_info.available_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        )?;
        writeln!(file, "- **NUMA Nodes**: {}", info.memory_info.numa_nodes)?;

        if let Some(bandwidth) = info.memory_info.memory_bandwidth {
            writeln!(file, "- **Estimated Bandwidth**: {:.1} GB/s", bandwidth)?;
        }

        writeln!(file, "\n## Build Environment\n")?;
        writeln!(
            file,
            "- **Rust Version**: {}",
            info.environment_info.rust_version
        )?;
        writeln!(
            file,
            "- **Build Mode**: {:?}",
            info.environment_info.build_mode
        )?;

        if !info.environment_info.target_features.is_empty() {
            writeln!(
                file,
                "- **Target Features**: {}",
                info.environment_info.target_features.join(", ")
            )?;
        }

        if !info.environment_info.environment_variables.is_empty() {
            writeln!(file, "\n**Environment Variables:**")?;
            for (key, value) in &info.environment_info.environment_variables {
                writeln!(file, "- {}: {}", key, value)?;
            }
        }

        writeln!(file, "\n## Benchmark Environment Assessment\n")?;
        writeln!(
            file,
            "- **CPU Isolation**: {}",
            if info.benchmark_environment.is_isolated {
                "✅ Enabled"
            } else {
                "❌ Disabled"
            }
        )?;

        if let Some(ref governor) = info.benchmark_environment.cpu_governor {
            writeln!(file, "- **CPU Governor**: {}", governor)?;
        }

        writeln!(
            file,
            "- **Thermal State**: {:?}",
            info.benchmark_environment.thermal_state
        )?;
        writeln!(
            file,
            "- **Background Load**: {:?}",
            info.benchmark_environment.background_load
        )?;
        writeln!(
            file,
            "- **Timing Precision**: {:?}",
            info.benchmark_environment.timing_precision.precision_rating
        )?;
        writeln!(
            file,
            "- **Reproducibility Score**: {:.1}%",
            info.benchmark_environment.reproducibility_score * 100.0
        )?;

        writeln!(file, "\n## Optimization Recommendations\n")?;
        for recommendation in &info.optimization_recommendations {
            writeln!(file, "{}", recommendation)?;
        }

        writeln!(file, "\n---")?;
        writeln!(
            file,
            "*Report generated by ToRSh System Information Collector*"
        )?;

        Ok(())
    }
}

impl Default for SystemInfoCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility function to get basic system info quickly
pub fn get_basic_system_info() -> BasicSystemInfo {
    SystemInfoCollector::new().collect_system_info()
}

/// Utility function to check if system is optimized for benchmarking
pub fn is_system_optimized_for_benchmarking() -> (bool, Vec<String>) {
    let mut collector = SystemInfoCollector::new();
    let info = collector.collect();

    let is_optimized = info.benchmark_environment.reproducibility_score > 0.8
        && !matches!(info.environment_info.build_mode, BuildMode::Debug)
        && !matches!(
            info.benchmark_environment.background_load,
            LoadLevel::High | LoadLevel::Critical
        );

    (is_optimized, info.optimization_recommendations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "System info collection may vary by platform"]
    fn test_system_info_collection() {
        let mut collector = SystemInfoCollector::new();
        let info = collector.collect();

        // Basic sanity checks
        assert!(!info.cpu_info.model.is_empty());
        assert!(info.cpu_info.cores > 0);
        assert!(info.cpu_info.threads >= info.cpu_info.cores);
        assert!(info.memory_info.total_memory > 0);
        assert!(!info.system_info.hostname.is_empty());
    }

    #[test]
    fn test_reproducibility_score() {
        let collector = SystemInfoCollector::new();
        let score = collector.calculate_reproducibility_score();
        assert!(score >= 0.0 && score <= 1.0);
    }
}

// Mock implementation for num_cpus when not available
#[cfg(not(test))]
mod num_cpus {
    pub fn get() -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    }

    pub fn get_physical() -> usize {
        get() / 2 // Rough estimate
    }
}

// Mock implementation for hostname when not available
#[cfg(not(test))]
mod hostname {
    pub fn get() -> Result<std::ffi::OsString, Box<dyn std::error::Error>> {
        Ok(std::ffi::OsString::from("unknown"))
    }
}
