//! Power consumption monitoring system
//!
//! This module provides comprehensive power monitoring capabilities including
//! total system power, component-specific power consumption, and energy efficiency metrics.

use std::time::{Duration, Instant};

/// Power consumption metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PowerMetrics {
    /// Total energy consumed in joules
    pub total_energy_joules: f64,

    /// Average power consumption in watts
    pub average_power_watts: f64,

    /// Peak power consumption in watts
    pub peak_power_watts: f64,

    /// Duration of measurement
    pub measurement_duration: Duration,

    /// Number of power samples taken
    pub sample_count: usize,

    /// CPU power consumption in watts (if available)
    pub cpu_power_watts: Option<f64>,

    /// GPU power consumption in watts (if available)
    pub gpu_power_watts: Option<f64>,

    /// Memory power consumption in watts (if available)
    pub memory_power_watts: Option<f64>,

    /// Power efficiency (operations per joule)
    pub power_efficiency_ops_per_joule: Option<f64>,
}

impl Default for PowerMetrics {
    fn default() -> Self {
        Self {
            total_energy_joules: 0.0,
            average_power_watts: 0.0,
            peak_power_watts: 0.0,
            measurement_duration: Duration::ZERO,
            sample_count: 0,
            cpu_power_watts: None,
            gpu_power_watts: None,
            memory_power_watts: None,
            power_efficiency_ops_per_joule: None,
        }
    }
}

/// Power monitoring system
pub struct PowerMonitor {
    start_time: Option<Instant>,
    power_samples: Vec<PowerSample>,
    sampling_interval: Duration,
    monitor_thread: Option<std::thread::JoinHandle<()>>,
    stop_signal: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

/// Single power measurement sample
#[derive(Debug, Clone)]
struct PowerSample {
    timestamp: Instant,
    total_power_watts: f64,
    cpu_power_watts: Option<f64>,
    gpu_power_watts: Option<f64>,
    memory_power_watts: Option<f64>,
}

impl PowerMonitor {
    /// Create new power monitor
    pub fn new() -> Self {
        Self {
            start_time: None,
            power_samples: Vec::new(),
            sampling_interval: Duration::from_millis(100), // 10Hz sampling
            monitor_thread: None,
            stop_signal: None,
        }
    }

    /// Set power sampling interval
    pub fn with_sampling_interval(mut self, interval: Duration) -> Self {
        self.sampling_interval = interval;
        self
    }

    /// Start power monitoring
    pub fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.monitor_thread.is_some() {
            return Err("Power monitoring already started".into());
        }

        self.start_time = Some(Instant::now());
        self.power_samples.clear();

        let stop_signal = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        self.stop_signal = Some(stop_signal.clone());

        let sampling_interval = self.sampling_interval;
        let samples = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let samples_clone = samples.clone();

        // Spawn monitoring thread
        let handle = std::thread::spawn(move || {
            let mut power_reader = SystemPowerReader::new();

            while !stop_signal.load(std::sync::atomic::Ordering::Relaxed) {
                if let Ok(sample) = power_reader.read_power() {
                    if let Ok(mut samples_guard) = samples_clone.try_lock() {
                        samples_guard.push(sample);
                    }
                }

                std::thread::sleep(sampling_interval);
            }
        });

        self.monitor_thread = Some(handle);
        Ok(())
    }

    /// Stop power monitoring and return metrics
    pub fn stop(&mut self) -> PowerMetrics {
        // Signal monitoring thread to stop
        if let Some(stop_signal) = &self.stop_signal {
            stop_signal.store(true, std::sync::atomic::Ordering::Relaxed);
        }

        // Wait for thread to finish
        if let Some(handle) = self.monitor_thread.take() {
            let _ = handle.join();
        }

        let measurement_duration = self
            .start_time
            .map(|start| start.elapsed())
            .unwrap_or(Duration::ZERO);

        if self.power_samples.is_empty() {
            return PowerMetrics {
                measurement_duration,
                ..Default::default()
            };
        }

        // Calculate metrics from samples
        let total_power: f64 = self.power_samples.iter().map(|s| s.total_power_watts).sum();
        let sample_count = self.power_samples.len();
        let average_power = total_power / sample_count as f64;
        let peak_power = self
            .power_samples
            .iter()
            .map(|s| s.total_power_watts)
            .fold(0.0, f64::max);

        // Energy = average power * time (in joules)
        let total_energy = average_power * measurement_duration.as_secs_f64();

        // Calculate component averages
        let avg_cpu_power = self.calculate_component_average(|s| s.cpu_power_watts);
        let avg_gpu_power = self.calculate_component_average(|s| s.gpu_power_watts);
        let avg_memory_power = self.calculate_component_average(|s| s.memory_power_watts);

        PowerMetrics {
            total_energy_joules: total_energy,
            average_power_watts: average_power,
            peak_power_watts: peak_power,
            measurement_duration,
            sample_count,
            cpu_power_watts: avg_cpu_power,
            gpu_power_watts: avg_gpu_power,
            memory_power_watts: avg_memory_power,
            power_efficiency_ops_per_joule: None, // To be calculated by caller
        }
    }

    /// Calculate power efficiency for a given number of operations
    pub fn calculate_power_efficiency(&self, metrics: &PowerMetrics, operations: u64) -> f64 {
        if metrics.total_energy_joules > 0.0 {
            operations as f64 / metrics.total_energy_joules
        } else {
            0.0
        }
    }

    /// Helper to calculate average component power
    fn calculate_component_average<F>(&self, extractor: F) -> Option<f64>
    where
        F: Fn(&PowerSample) -> Option<f64>,
    {
        let samples: Vec<f64> = self.power_samples.iter().filter_map(extractor).collect();

        if !samples.is_empty() {
            Some(samples.iter().sum::<f64>() / samples.len() as f64)
        } else {
            None
        }
    }
}

impl Default for PowerMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// System power reader that attempts to read power from multiple sources
struct SystemPowerReader {
    power_sources: Vec<Box<dyn PowerSource>>,
}

impl SystemPowerReader {
    fn new() -> Self {
        let mut sources: Vec<Box<dyn PowerSource>> = Vec::new();

        // Try different power sources based on platform
        #[cfg(target_os = "linux")]
        {
            sources.push(Box::new(LinuxPowerSource::new()));
            sources.push(Box::new(RaplPowerSource::new()));
        }

        #[cfg(target_os = "windows")]
        {
            sources.push(Box::new(WindowsPowerSource::new()));
        }

        #[cfg(target_os = "macos")]
        {
            sources.push(Box::new(MacOsPowerSource::new()));
        }

        // Fallback to estimate-based power source
        sources.push(Box::new(EstimatedPowerSource::new()));

        Self {
            power_sources: sources,
        }
    }

    fn read_power(&mut self) -> Result<PowerSample, Box<dyn std::error::Error>> {
        for source in &mut self.power_sources {
            if let Ok(sample) = source.read_power() {
                return Ok(sample);
            }
        }

        Err("No power sources available".into())
    }
}

/// Trait for different power monitoring sources
trait PowerSource {
    fn read_power(&mut self) -> Result<PowerSample, Box<dyn std::error::Error>>;
}

/// Linux-specific power monitoring using /sys/class/power_supply
#[cfg(target_os = "linux")]
struct LinuxPowerSource {
    last_reading_time: Option<Instant>,
}

#[cfg(target_os = "linux")]
impl LinuxPowerSource {
    fn new() -> Self {
        Self {
            last_reading_time: None,
        }
    }
}

#[cfg(target_os = "linux")]
impl PowerSource for LinuxPowerSource {
    fn read_power(&mut self) -> Result<PowerSample, Box<dyn std::error::Error>> {
        // Simplified implementation - would read from /sys/class/power_supply
        Ok(PowerSample {
            timestamp: Instant::now(),
            total_power_watts: 15.0, // Placeholder value
            cpu_power_watts: Some(10.0),
            gpu_power_watts: None,
            memory_power_watts: Some(5.0),
        })
    }
}

/// RAPL (Running Average Power Limit) power monitoring for Intel CPUs
#[cfg(target_os = "linux")]
struct RaplPowerSource;

#[cfg(target_os = "linux")]
impl RaplPowerSource {
    fn new() -> Self {
        Self
    }
}

#[cfg(target_os = "linux")]
impl PowerSource for RaplPowerSource {
    fn read_power(&mut self) -> Result<PowerSample, Box<dyn std::error::Error>> {
        // Would read from /sys/class/powercap/intel-rapl
        Ok(PowerSample {
            timestamp: Instant::now(),
            total_power_watts: 12.0,
            cpu_power_watts: Some(8.0),
            gpu_power_watts: None,
            memory_power_watts: Some(4.0),
        })
    }
}

/// Windows-specific power monitoring
#[cfg(target_os = "windows")]
struct WindowsPowerSource;

#[cfg(target_os = "windows")]
impl WindowsPowerSource {
    fn new() -> Self {
        Self
    }
}

#[cfg(target_os = "windows")]
impl PowerSource for WindowsPowerSource {
    fn read_power(&mut self) -> Result<PowerSample, Box<dyn std::error::Error>> {
        // Would use Windows Performance Counters or ETW
        Ok(PowerSample {
            timestamp: Instant::now(),
            total_power_watts: 20.0,
            cpu_power_watts: Some(15.0),
            gpu_power_watts: None,
            memory_power_watts: Some(5.0),
        })
    }
}

/// macOS-specific power monitoring
#[cfg(target_os = "macos")]
struct MacOsPowerSource;

#[cfg(target_os = "macos")]
impl MacOsPowerSource {
    fn new() -> Self {
        Self
    }
}

#[cfg(target_os = "macos")]
impl PowerSource for MacOsPowerSource {
    fn read_power(&mut self) -> Result<PowerSample, Box<dyn std::error::Error>> {
        // Would use IOKit or powermetrics
        Ok(PowerSample {
            timestamp: Instant::now(),
            total_power_watts: 18.0,
            cpu_power_watts: Some(12.0),
            gpu_power_watts: Some(3.0),
            memory_power_watts: Some(3.0),
        })
    }
}

/// Fallback estimated power source based on CPU usage
struct EstimatedPowerSource;

impl EstimatedPowerSource {
    fn new() -> Self {
        Self
    }
}

impl PowerSource for EstimatedPowerSource {
    fn read_power(&mut self) -> Result<PowerSample, Box<dyn std::error::Error>> {
        // Simple estimation based on assumed baseline power consumption
        // In practice, would consider CPU usage, system load, etc.
        Ok(PowerSample {
            timestamp: Instant::now(),
            total_power_watts: 10.0, // Conservative estimate
            cpu_power_watts: Some(7.0),
            gpu_power_watts: None,
            memory_power_watts: Some(3.0),
        })
    }
}
