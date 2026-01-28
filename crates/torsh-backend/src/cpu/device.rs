//! CPU Device Implementation

use crate::error::BackendResult;
use crate::{Device, DeviceFeature, DeviceInfo};
use torsh_core::device::DeviceType;

#[cfg(not(feature = "std"))]
use alloc::string::String;

/// CPU device representation
#[derive(Debug, Clone)]
pub struct CpuDevice {
    id: usize,
    num_cores: usize,
}

impl CpuDevice {
    /// Create a new CPU device
    pub fn new(id: usize, num_cores: usize) -> BackendResult<Self> {
        Ok(Self { id, num_cores })
    }

    /// Get the device ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get the number of CPU cores
    pub fn num_cores(&self) -> usize {
        self.num_cores
    }

    /// Convert to abstract Device
    pub fn to_device(&self) -> Device {
        let info = DeviceInfo {
            vendor: Self::get_cpu_vendor(),
            driver_version: "N/A".to_string(),
            total_memory: Self::get_total_memory(),
            available_memory: Self::get_available_memory(),
            compute_units: self.num_cores,
            max_work_group_size: usize::MAX,
            max_work_group_dimensions: vec![usize::MAX, 1, 1],
            clock_frequency_mhz: 3000,   // Typical CPU frequency
            memory_bandwidth_gbps: 50.0, // Typical DDR4 bandwidth
            peak_gflops: self.num_cores as f32 * 10.0, // Rough estimate
            features: vec![
                DeviceFeature::DoublePrecision,
                DeviceFeature::UnifiedMemory,
                DeviceFeature::AtomicOperations,
                DeviceFeature::Profiling,
                DeviceFeature::ConcurrentExecution,
                DeviceFeature::AsyncMemory,
                DeviceFeature::FastMath,
            ],
            properties: vec![
                ("num_cores".to_string(), self.num_cores.to_string()),
                ("architecture".to_string(), Self::get_cpu_architecture()),
            ],
        };

        Device::new(
            self.id,
            DeviceType::Cpu,
            format!("CPU ({} cores)", self.num_cores),
            info,
        )
    }

    /// Get total system memory in bytes
    fn get_total_memory() -> usize {
        // ✅ Pure Rust: Use sysinfo instead of libc::sysctlbyname
        #[cfg(feature = "cpu")]
        {
            use sysinfo::System;
            let mut sys = System::new_all();
            sys.refresh_memory();
            return sys.total_memory() as usize;
        }

        #[cfg(not(feature = "cpu"))]
        {
            // Fallback for no_std or when cpu feature disabled
            8 * 1024 * 1024 * 1024 // 8GB
        }
    }

    /// Get available system memory in bytes
    fn get_available_memory() -> usize {
        // ✅ Pure Rust: Use sysinfo for accurate available memory
        #[cfg(feature = "cpu")]
        {
            use sysinfo::System;
            let mut sys = System::new_all();
            sys.refresh_memory();
            return sys.available_memory() as usize;
        }

        #[cfg(not(feature = "cpu"))]
        {
            // Fallback: assume 80% of total memory is available
            Self::get_total_memory() * 8 / 10
        }
    }

    /// Get CPU vendor information
    fn get_cpu_vendor() -> String {
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(info) = raw_cpuid::CpuId::new().get_vendor_info() {
                return info.as_str().to_string();
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            return "ARM".to_string();
        }

        #[allow(unreachable_code)]
        "Unknown".to_string()
    }

    /// Get CPU architecture information
    fn get_cpu_architecture() -> String {
        #[cfg(target_arch = "x86_64")]
        {
            "x86_64".to_string()
        }
        #[cfg(target_arch = "aarch64")]
        {
            "aarch64".to_string()
        }
        #[cfg(target_arch = "arm")]
        {
            "arm".to_string()
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "arm")))]
        {
            env!("TARGET").to_string()
        }
    }
}

impl From<CpuDevice> for Device {
    fn from(cpu_device: CpuDevice) -> Self {
        cpu_device.to_device()
    }
}

impl From<&CpuDevice> for Device {
    fn from(cpu_device: &CpuDevice) -> Self {
        cpu_device.to_device()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_device_creation() {
        let device = CpuDevice::new(0, 8).unwrap();
        assert_eq!(device.id(), 0);
        assert_eq!(device.num_cores(), 8);
        let device_info = device.to_device();
        assert_eq!(device_info.device_type(), DeviceType::Cpu);
        assert!(device_info.info().total_memory > 0);
    }

    #[test]
    fn test_cpu_device_conversion() {
        let cpu_device = CpuDevice::new(0, 4).unwrap();
        let device: Device = cpu_device.clone().into();

        assert_eq!(device.device_type(), DeviceType::Cpu);
        assert_eq!(device.id(), 0);
        assert_eq!(device.info().compute_units, 4);
    }
}
