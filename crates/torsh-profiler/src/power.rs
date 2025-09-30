//! Power profiling capabilities for energy-efficient performance monitoring
//!
//! This module provides comprehensive power profiling functionality including
//! CPU, GPU, memory, and system-level power monitoring across different platforms.

use crate::ProfileEvent;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};

/// Power measurement units
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum PowerUnit {
    Watts,
    Milliwatts,
    Microwatts,
    Joules,
    WattHours,
    KilowattHours,
}

impl std::fmt::Display for PowerUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PowerUnit::Watts => write!(f, "W"),
            PowerUnit::Milliwatts => write!(f, "mW"),
            PowerUnit::Microwatts => write!(f, "µW"),
            PowerUnit::Joules => write!(f, "J"),
            PowerUnit::WattHours => write!(f, "Wh"),
            PowerUnit::KilowattHours => write!(f, "kWh"),
        }
    }
}

/// Power domains that can be monitored
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Hash, Eq)]
pub enum PowerDomain {
    Cpu,
    Gpu,
    Memory,
    Storage,
    Network,
    Display,
    System,
    Package, // Intel package power
    Core,    // Intel core power
    Uncore,  // Intel uncore power
    Dram,    // DRAM power
    Custom(String),
}

impl std::fmt::Display for PowerDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PowerDomain::Cpu => write!(f, "CPU"),
            PowerDomain::Gpu => write!(f, "GPU"),
            PowerDomain::Memory => write!(f, "Memory"),
            PowerDomain::Storage => write!(f, "Storage"),
            PowerDomain::Network => write!(f, "Network"),
            PowerDomain::Display => write!(f, "Display"),
            PowerDomain::System => write!(f, "System"),
            PowerDomain::Package => write!(f, "Package"),
            PowerDomain::Core => write!(f, "Core"),
            PowerDomain::Uncore => write!(f, "Uncore"),
            PowerDomain::Dram => write!(f, "DRAM"),
            PowerDomain::Custom(name) => write!(f, "{name}"),
        }
    }
}

/// Power measurement sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSample {
    pub timestamp: SystemTime,
    pub domain: PowerDomain,
    pub power: f64,
    pub unit: PowerUnit,
    pub voltage: Option<f64>,
    pub current: Option<f64>,
    pub frequency: Option<f64>,
    pub temperature: Option<f64>,
}

/// Power profiling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerConfig {
    pub enabled_domains: Vec<PowerDomain>,
    pub sampling_rate_hz: f64,
    pub enable_rapl: bool,           // Intel RAPL
    pub enable_nvidia_ml: bool,      // NVIDIA Management Library
    pub enable_amd_adl: bool,        // AMD Display Library
    pub enable_apple_smc: bool,      // Apple System Management Controller
    pub enable_cpu_freq: bool,       // CPU frequency scaling monitoring
    pub enable_thermal: bool,        // Thermal monitoring
    pub baseline_power: Option<f64>, // Baseline system power for calculation
    pub power_cap: Option<f64>,      // Power limit for warnings
}

impl Default for PowerConfig {
    fn default() -> Self {
        Self {
            enabled_domains: vec![
                PowerDomain::Cpu,
                PowerDomain::Gpu,
                PowerDomain::Memory,
                PowerDomain::System,
            ],
            sampling_rate_hz: 10.0, // 10 Hz sampling
            enable_rapl: true,
            enable_nvidia_ml: true,
            enable_amd_adl: true,
            enable_apple_smc: true,
            enable_cpu_freq: true,
            enable_thermal: true,
            baseline_power: None,
            power_cap: None,
        }
    }
}

/// Power profiling statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerStats {
    pub domain: PowerDomain,
    pub samples_count: u64,
    pub min_power: f64,
    pub max_power: f64,
    pub average_power: f64,
    pub total_energy: f64,
    pub peak_to_average_ratio: f64,
    pub power_efficiency: f64, // Operations per watt
    pub thermal_throttling_events: u64,
    pub power_limit_events: u64,
}

/// Power efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerEfficiency {
    pub operations_per_watt: f64,
    pub energy_per_operation: f64,
    pub gflops_per_watt: f64,
    pub bytes_per_joule: f64,
    pub performance_per_watt: f64,
    pub energy_delay_product: f64,
}

/// Power profiler implementation
pub struct PowerProfiler {
    config: PowerConfig,
    samples: Vec<PowerSample>,
    rapl_monitor: Option<RaplMonitor>,
    nvidia_monitor: Option<NvidiaMonitor>,
    amd_monitor: Option<AmdMonitor>,
    apple_monitor: Option<AppleMonitor>,
    cpu_freq_monitor: Option<CpuFreqMonitor>,
    thermal_monitor: Option<ThermalMonitor>,
    baseline_power: f64,
    last_sample_time: Option<SystemTime>,
}

impl PowerProfiler {
    pub fn new(config: PowerConfig) -> Result<Self> {
        let rapl_monitor = if config.enable_rapl {
            Some(RaplMonitor::new()?)
        } else {
            None
        };

        let nvidia_monitor = if config.enable_nvidia_ml {
            Some(NvidiaMonitor::new()?)
        } else {
            None
        };

        let amd_monitor = if config.enable_amd_adl {
            Some(AmdMonitor::new()?)
        } else {
            None
        };

        let apple_monitor = if config.enable_apple_smc {
            Some(AppleMonitor::new()?)
        } else {
            None
        };

        let cpu_freq_monitor = if config.enable_cpu_freq {
            Some(CpuFreqMonitor::new()?)
        } else {
            None
        };

        let thermal_monitor = if config.enable_thermal {
            Some(ThermalMonitor::new()?)
        } else {
            None
        };

        let baseline_power = config.baseline_power.unwrap_or(0.0);

        Ok(Self {
            config,
            samples: Vec::new(),
            rapl_monitor,
            nvidia_monitor,
            amd_monitor,
            apple_monitor,
            cpu_freq_monitor,
            thermal_monitor,
            baseline_power,
            last_sample_time: None,
        })
    }

    /// Start power monitoring
    pub fn start_monitoring(&mut self) -> Result<()> {
        if let Some(rapl) = &mut self.rapl_monitor {
            rapl.start()?;
        }
        if let Some(nvidia) = &mut self.nvidia_monitor {
            nvidia.start()?;
        }
        if let Some(amd) = &mut self.amd_monitor {
            amd.start()?;
        }
        if let Some(apple) = &mut self.apple_monitor {
            apple.start()?;
        }
        if let Some(cpu_freq) = &mut self.cpu_freq_monitor {
            cpu_freq.start()?;
        }
        if let Some(thermal) = &mut self.thermal_monitor {
            thermal.start()?;
        }

        self.last_sample_time = Some(SystemTime::now());
        Ok(())
    }

    /// Stop power monitoring
    pub fn stop_monitoring(&mut self) -> Result<()> {
        if let Some(rapl) = &mut self.rapl_monitor {
            rapl.stop()?;
        }
        if let Some(nvidia) = &mut self.nvidia_monitor {
            nvidia.stop()?;
        }
        if let Some(amd) = &mut self.amd_monitor {
            amd.stop()?;
        }
        if let Some(apple) = &mut self.apple_monitor {
            apple.stop()?;
        }
        if let Some(cpu_freq) = &mut self.cpu_freq_monitor {
            cpu_freq.stop()?;
        }
        if let Some(thermal) = &mut self.thermal_monitor {
            thermal.stop()?;
        }

        Ok(())
    }

    /// Collect power samples from all enabled monitors
    pub fn collect_samples(&mut self) -> Result<Vec<PowerSample>> {
        let mut new_samples = Vec::new();
        let timestamp = SystemTime::now();

        // Check if enough time has passed since last sample
        if let Some(last_time) = self.last_sample_time {
            let elapsed = timestamp
                .duration_since(last_time)
                .unwrap_or(Duration::ZERO);
            let sample_interval = Duration::from_secs_f64(1.0 / self.config.sampling_rate_hz);

            if elapsed < sample_interval {
                return Ok(new_samples);
            }
        }

        // Collect from Intel RAPL
        if let Some(rapl) = &self.rapl_monitor {
            new_samples.extend(rapl.get_samples(timestamp)?);
        }

        // Collect from NVIDIA GPU
        if let Some(nvidia) = &self.nvidia_monitor {
            new_samples.extend(nvidia.get_samples(timestamp)?);
        }

        // Collect from AMD GPU
        if let Some(amd) = &self.amd_monitor {
            new_samples.extend(amd.get_samples(timestamp)?);
        }

        // Collect from Apple SMC
        if let Some(apple) = &self.apple_monitor {
            new_samples.extend(apple.get_samples(timestamp)?);
        }

        // Collect CPU frequency data
        if let Some(cpu_freq) = &self.cpu_freq_monitor {
            new_samples.extend(cpu_freq.get_samples(timestamp)?);
        }

        // Collect thermal data
        if let Some(thermal) = &self.thermal_monitor {
            new_samples.extend(thermal.get_samples(timestamp)?);
        }

        // Filter samples based on enabled domains
        new_samples.retain(|sample| self.config.enabled_domains.contains(&sample.domain));

        self.samples.extend(new_samples.clone());
        self.last_sample_time = Some(timestamp);

        Ok(new_samples)
    }

    /// Calculate power statistics for a specific domain
    pub fn calculate_power_stats(&self, domain: &PowerDomain) -> Option<PowerStats> {
        let domain_samples: Vec<_> = self
            .samples
            .iter()
            .filter(|s| s.domain == *domain)
            .collect();

        if domain_samples.is_empty() {
            return None;
        }

        let powers: Vec<f64> = domain_samples.iter().map(|s| s.power).collect();
        let min_power = powers.iter().copied().fold(f64::INFINITY, f64::min);
        let max_power = powers.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let average_power = powers.iter().sum::<f64>() / powers.len() as f64;

        // Calculate total energy (simple integration)
        let total_energy = if domain_samples.len() > 1 {
            let time_span = domain_samples
                .last()
                .unwrap()
                .timestamp
                .duration_since(domain_samples.first().unwrap().timestamp)
                .unwrap_or(Duration::ZERO)
                .as_secs_f64();
            average_power * time_span / 3600.0 // Convert to Wh
        } else {
            0.0
        };

        let peak_to_average_ratio = if average_power > 0.0 {
            max_power / average_power
        } else {
            0.0
        };

        Some(PowerStats {
            domain: domain.clone(),
            samples_count: domain_samples.len() as u64,
            min_power,
            max_power,
            average_power,
            total_energy,
            peak_to_average_ratio,
            power_efficiency: 0.0, // Would need performance data to calculate
            thermal_throttling_events: 0, // Would need thermal event data
            power_limit_events: 0, // Would need power limit event data
        })
    }

    /// Calculate power efficiency metrics
    pub fn calculate_power_efficiency(&self, events: &[ProfileEvent]) -> PowerEfficiency {
        let total_operations = events.len() as f64;
        let total_flops: f64 = events.iter().filter_map(|e| e.flops).sum::<u64>() as f64;
        let total_bytes: f64 = events
            .iter()
            .filter_map(|e| e.bytes_transferred)
            .sum::<u64>() as f64;
        let total_duration = events.iter().map(|e| e.duration_us).sum::<u64>() as f64 / 1_000_000.0;

        // Calculate average power consumption during profiling period
        let avg_power = if !self.samples.is_empty() {
            self.samples.iter().map(|s| s.power).sum::<f64>() / self.samples.len() as f64
        } else {
            0.0
        };

        let total_energy = avg_power * total_duration / 3600.0; // Wh

        let operations_per_watt = if avg_power > 0.0 {
            total_operations / avg_power
        } else {
            0.0
        };

        let energy_per_operation = if total_operations > 0.0 {
            total_energy / total_operations
        } else {
            0.0
        };

        let gflops_per_watt = if avg_power > 0.0 {
            total_flops / avg_power / 1_000_000_000.0
        } else {
            0.0
        };

        let bytes_per_joule = if total_energy > 0.0 {
            total_bytes / (total_energy * 3600.0) // Convert Wh to J
        } else {
            0.0
        };

        let performance_per_watt = if avg_power > 0.0 && total_duration > 0.0 {
            total_operations / total_duration / avg_power
        } else {
            0.0
        };

        let energy_delay_product = total_energy * total_duration;

        PowerEfficiency {
            operations_per_watt,
            energy_per_operation,
            gflops_per_watt,
            bytes_per_joule,
            performance_per_watt,
            energy_delay_product,
        }
    }

    /// Export power data to CSV
    pub fn export_csv(&self, path: &str) -> Result<()> {
        let mut csv = String::new();
        csv.push_str("timestamp,domain,power_w,voltage_v,current_a,frequency_hz,temperature_c\n");

        for sample in &self.samples {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                sample
                    .timestamp
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                sample.domain,
                sample.power,
                sample.voltage.unwrap_or(0.0),
                sample.current.unwrap_or(0.0),
                sample.frequency.unwrap_or(0.0),
                sample.temperature.unwrap_or(0.0),
            ));
        }

        fs::write(path, csv)?;
        Ok(())
    }

    /// Get power samples for a specific time range
    pub fn get_samples_in_range(&self, start: SystemTime, end: SystemTime) -> Vec<PowerSample> {
        self.samples
            .iter()
            .filter(|s| s.timestamp >= start && s.timestamp <= end)
            .cloned()
            .collect()
    }

    /// Clear collected samples
    pub fn clear_samples(&mut self) {
        self.samples.clear();
    }

    /// Get total number of samples
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

/// Intel RAPL (Running Average Power Limit) monitor
pub struct RaplMonitor {
    enabled: bool,
    msr_files: HashMap<PowerDomain, String>,
}

impl RaplMonitor {
    pub fn new() -> Result<Self> {
        let mut msr_files = HashMap::new();

        // Check for RAPL MSR files (requires root access on Linux)
        if Path::new("/dev/cpu/0/msr").exists() {
            msr_files.insert(
                PowerDomain::Package,
                "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj".to_string(),
            );
            msr_files.insert(
                PowerDomain::Core,
                "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj".to_string(),
            );
            msr_files.insert(
                PowerDomain::Uncore,
                "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:1/energy_uj".to_string(),
            );
            msr_files.insert(
                PowerDomain::Dram,
                "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:2/energy_uj".to_string(),
            );
        }

        Ok(Self {
            enabled: !msr_files.is_empty(),
            msr_files,
        })
    }

    pub fn start(&mut self) -> Result<()> {
        if !self.enabled {
            return Err(anyhow::anyhow!("RAPL not available"));
        }
        Ok(())
    }

    pub fn stop(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn get_samples(&self, timestamp: SystemTime) -> Result<Vec<PowerSample>> {
        let mut samples = Vec::new();

        for (domain, file_path) in &self.msr_files {
            if let Ok(energy_str) = fs::read_to_string(file_path) {
                if let Ok(energy_uj) = energy_str.trim().parse::<f64>() {
                    // Convert microjoules to watts (would need previous reading for accurate calculation)
                    let power = energy_uj / 1_000_000.0; // Simplified conversion

                    samples.push(PowerSample {
                        timestamp,
                        domain: domain.clone(),
                        power,
                        unit: PowerUnit::Watts,
                        voltage: None,
                        current: None,
                        frequency: None,
                        temperature: None,
                    });
                }
            }
        }

        Ok(samples)
    }
}

/// NVIDIA GPU power monitor using NVIDIA Management Library (stub implementation)
pub struct NvidiaMonitor {
    enabled: bool,
}

impl NvidiaMonitor {
    pub fn new() -> Result<Self> {
        // In a real implementation, this would initialize NVML
        Ok(Self { enabled: false })
    }

    pub fn start(&mut self) -> Result<()> {
        // Initialize NVML and get device handles
        Ok(())
    }

    pub fn stop(&mut self) -> Result<()> {
        // Cleanup NVML
        Ok(())
    }

    pub fn get_samples(&self, timestamp: SystemTime) -> Result<Vec<PowerSample>> {
        let mut samples = Vec::new();

        if self.enabled {
            // In real implementation, would call nvmlDeviceGetPowerUsage
            samples.push(PowerSample {
                timestamp,
                domain: PowerDomain::Gpu,
                power: 150.0, // Placeholder value
                unit: PowerUnit::Watts,
                voltage: Some(1.0),
                current: Some(150.0),
                frequency: Some(1_500_000_000.0),
                temperature: Some(65.0),
            });
        }

        Ok(samples)
    }
}

/// AMD GPU power monitor using AMD Display Library (stub implementation)
pub struct AmdMonitor {
    enabled: bool,
}

impl AmdMonitor {
    pub fn new() -> Result<Self> {
        // In a real implementation, this would initialize ADL
        Ok(Self { enabled: false })
    }

    pub fn start(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn stop(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn get_samples(&self, timestamp: SystemTime) -> Result<Vec<PowerSample>> {
        let mut samples = Vec::new();

        if self.enabled {
            // In real implementation, would call ADL power functions
            samples.push(PowerSample {
                timestamp,
                domain: PowerDomain::Gpu,
                power: 120.0, // Placeholder value
                unit: PowerUnit::Watts,
                voltage: None,
                current: None,
                frequency: Some(1_200_000_000.0),
                temperature: Some(70.0),
            });
        }

        Ok(samples)
    }
}

/// Apple System Management Controller monitor (stub implementation)
pub struct AppleMonitor {
    enabled: bool,
}

impl AppleMonitor {
    pub fn new() -> Result<Self> {
        // In a real implementation, this would check for macOS and SMC access
        let enabled = cfg!(target_os = "macos");
        Ok(Self { enabled })
    }

    pub fn start(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn stop(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn get_samples(&self, timestamp: SystemTime) -> Result<Vec<PowerSample>> {
        let mut samples = Vec::new();

        if self.enabled {
            // In real implementation, would read SMC keys for power data
            samples.push(PowerSample {
                timestamp,
                domain: PowerDomain::System,
                power: 25.0, // Placeholder value
                unit: PowerUnit::Watts,
                voltage: None,
                current: None,
                frequency: None,
                temperature: Some(45.0),
            });
        }

        Ok(samples)
    }
}

/// CPU frequency monitor
pub struct CpuFreqMonitor {
    enabled: bool,
    cpu_count: usize,
}

impl CpuFreqMonitor {
    pub fn new() -> Result<Self> {
        let cpu_count = num_cpus::get();
        let enabled = Path::new("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").exists();

        Ok(Self { enabled, cpu_count })
    }

    pub fn start(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn stop(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn get_samples(&self, timestamp: SystemTime) -> Result<Vec<PowerSample>> {
        let mut samples = Vec::new();

        if self.enabled {
            for cpu in 0..self.cpu_count {
                let freq_path =
                    format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq");
                if let Ok(freq_str) = fs::read_to_string(&freq_path) {
                    if let Ok(freq_khz) = freq_str.trim().parse::<f64>() {
                        samples.push(PowerSample {
                            timestamp,
                            domain: PowerDomain::Custom(format!("CPU{cpu}")),
                            power: 0.0, // Frequency monitoring doesn't directly give power
                            unit: PowerUnit::Watts,
                            voltage: None,
                            current: None,
                            frequency: Some(freq_khz * 1000.0), // Convert to Hz
                            temperature: None,
                        });
                    }
                }
            }
        }

        Ok(samples)
    }
}

/// Thermal monitor
pub struct ThermalMonitor {
    enabled: bool,
    thermal_zones: Vec<String>,
}

impl ThermalMonitor {
    pub fn new() -> Result<Self> {
        let mut thermal_zones = Vec::new();

        // Find thermal zones on Linux
        if let Ok(entries) = fs::read_dir("/sys/class/thermal") {
            for entry in entries.flatten() {
                if entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("thermal_zone")
                {
                    thermal_zones.push(entry.path().to_string_lossy().to_string());
                }
            }
        }

        Ok(Self {
            enabled: !thermal_zones.is_empty(),
            thermal_zones,
        })
    }

    pub fn start(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn stop(&mut self) -> Result<()> {
        Ok(())
    }

    pub fn get_samples(&self, timestamp: SystemTime) -> Result<Vec<PowerSample>> {
        let mut samples = Vec::new();

        if self.enabled {
            for (i, zone_path) in self.thermal_zones.iter().enumerate() {
                let temp_path = format!("{zone_path}/temp");
                if let Ok(temp_str) = fs::read_to_string(&temp_path) {
                    if let Ok(temp_millic) = temp_str.trim().parse::<f64>() {
                        let temp_celsius = temp_millic / 1000.0;

                        samples.push(PowerSample {
                            timestamp,
                            domain: PowerDomain::Custom(format!("ThermalZone{i}")),
                            power: 0.0, // Temperature monitoring doesn't directly give power
                            unit: PowerUnit::Watts,
                            voltage: None,
                            current: None,
                            frequency: None,
                            temperature: Some(temp_celsius),
                        });
                    }
                }
            }
        }

        Ok(samples)
    }
}

/// Public API functions
/// Create a power profiler with default configuration
pub fn create_power_profiler() -> Result<PowerProfiler> {
    PowerProfiler::new(PowerConfig::default())
}

/// Create a power profiler with custom configuration
pub fn create_power_profiler_with_config(config: PowerConfig) -> Result<PowerProfiler> {
    PowerProfiler::new(config)
}

/// Calculate power efficiency from profile events and power samples
pub fn calculate_power_efficiency(
    events: &[ProfileEvent],
    power_samples: &[PowerSample],
) -> PowerEfficiency {
    // This is a simplified implementation
    let total_operations = events.len() as f64;
    let avg_power = if !power_samples.is_empty() {
        power_samples.iter().map(|s| s.power).sum::<f64>() / power_samples.len() as f64
    } else {
        0.0
    };

    PowerEfficiency {
        operations_per_watt: if avg_power > 0.0 {
            total_operations / avg_power
        } else {
            0.0
        },
        energy_per_operation: if total_operations > 0.0 {
            avg_power / total_operations
        } else {
            0.0
        },
        gflops_per_watt: 0.0,      // Would need FLOPS data
        bytes_per_joule: 0.0,      // Would need bytes data
        performance_per_watt: 0.0, // Would need performance metric
        energy_delay_product: 0.0, // Would need timing data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_config_creation() {
        let config = PowerConfig::default();
        assert!(config.enabled_domains.contains(&PowerDomain::Cpu));
        assert!(config.enabled_domains.contains(&PowerDomain::Gpu));
        assert_eq!(config.sampling_rate_hz, 10.0);
    }

    #[test]
    fn test_power_sample_creation() {
        let sample = PowerSample {
            timestamp: SystemTime::now(),
            domain: PowerDomain::Cpu,
            power: 50.0,
            unit: PowerUnit::Watts,
            voltage: Some(1.2),
            current: Some(41.67),
            frequency: Some(3_500_000_000.0),
            temperature: Some(60.0),
        };

        assert_eq!(sample.domain, PowerDomain::Cpu);
        assert_eq!(sample.power, 50.0);
        assert_eq!(sample.unit, PowerUnit::Watts);
    }

    #[test]
    fn test_power_profiler_creation() {
        let config = PowerConfig::default();
        let profiler = PowerProfiler::new(config);
        assert!(profiler.is_ok());
    }

    #[test]
    fn test_power_stats_calculation() {
        let mut profiler = PowerProfiler::new(PowerConfig::default()).unwrap();

        // Add some sample data
        profiler.samples = vec![
            PowerSample {
                timestamp: SystemTime::now(),
                domain: PowerDomain::Cpu,
                power: 50.0,
                unit: PowerUnit::Watts,
                voltage: None,
                current: None,
                frequency: None,
                temperature: None,
            },
            PowerSample {
                timestamp: SystemTime::now(),
                domain: PowerDomain::Cpu,
                power: 60.0,
                unit: PowerUnit::Watts,
                voltage: None,
                current: None,
                frequency: None,
                temperature: None,
            },
        ];

        let stats = profiler.calculate_power_stats(&PowerDomain::Cpu).unwrap();
        assert_eq!(stats.samples_count, 2);
        assert_eq!(stats.min_power, 50.0);
        assert_eq!(stats.max_power, 60.0);
        assert_eq!(stats.average_power, 55.0);
    }

    #[test]
    fn test_power_efficiency_calculation() {
        let events = vec![ProfileEvent {
            name: "test_op".to_string(),
            category: "test".to_string(),
            start_us: 0,
            duration_us: 1_000_000, // 1 second
            thread_id: 0,
            operation_count: Some(1),
            flops: Some(1_000_000_000),         // 1 GFLOP
            bytes_transferred: Some(1_000_000), // 1 MB
            stack_trace: Some("test trace".to_string()),
        }];

        let power_samples = vec![PowerSample {
            timestamp: SystemTime::now(),
            domain: PowerDomain::Cpu,
            power: 100.0, // 100W
            unit: PowerUnit::Watts,
            voltage: None,
            current: None,
            frequency: None,
            temperature: None,
        }];

        let efficiency = calculate_power_efficiency(&events, &power_samples);
        assert_eq!(efficiency.operations_per_watt, 0.01); // 1 operation / 100W
    }

    #[test]
    fn test_power_domain_display() {
        assert_eq!(PowerDomain::Cpu.to_string(), "CPU");
        assert_eq!(PowerDomain::Gpu.to_string(), "GPU");
        assert_eq!(
            PowerDomain::Custom("Custom".to_string()).to_string(),
            "Custom"
        );
    }

    #[test]
    fn test_power_unit_display() {
        assert_eq!(PowerUnit::Watts.to_string(), "W");
        assert_eq!(PowerUnit::Milliwatts.to_string(), "mW");
        assert_eq!(PowerUnit::Joules.to_string(), "J");
    }

    #[test]
    fn test_rapl_monitor_creation() {
        let monitor = RaplMonitor::new();
        assert!(monitor.is_ok());
    }

    #[test]
    fn test_nvidia_monitor_creation() {
        let monitor = NvidiaMonitor::new();
        assert!(monitor.is_ok());
    }

    #[test]
    fn test_sample_filtering() {
        let mut profiler = PowerProfiler::new(PowerConfig {
            enabled_domains: vec![PowerDomain::Cpu],
            ..PowerConfig::default()
        })
        .unwrap();

        profiler.samples = vec![
            PowerSample {
                timestamp: SystemTime::now(),
                domain: PowerDomain::Cpu,
                power: 50.0,
                unit: PowerUnit::Watts,
                voltage: None,
                current: None,
                frequency: None,
                temperature: None,
            },
            PowerSample {
                timestamp: SystemTime::now(),
                domain: PowerDomain::Gpu,
                power: 150.0,
                unit: PowerUnit::Watts,
                voltage: None,
                current: None,
                frequency: None,
                temperature: None,
            },
        ];

        let cpu_samples: Vec<_> = profiler
            .samples
            .iter()
            .filter(|s| s.domain == PowerDomain::Cpu)
            .collect();

        assert_eq!(cpu_samples.len(), 1);
        assert_eq!(cpu_samples[0].power, 50.0);
    }
}
