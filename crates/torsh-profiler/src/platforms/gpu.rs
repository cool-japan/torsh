//! GPU profiling platform integrations

#![allow(ambiguous_glob_reexports)]
#![allow(unexpected_cfgs)]

// Re-export from existing modules for backward compatibility
pub use crate::amd::*;
pub use crate::cuda::*;
pub use crate::nsight::*;

/// Unified GPU profiling interface
pub struct GpuProfilerPlatform {
    pub cuda_profiler: Option<crate::cuda::CudaProfiler>,
    pub nsight_profiler: Option<crate::nsight::NsightProfiler>,
    pub amd_profiler: Option<crate::amd::AMDProfiler>,
    pub gpu_vendor: GpuVendor,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Unknown,
}

impl GpuProfilerPlatform {
    pub fn new() -> Self {
        Self {
            cuda_profiler: None,
            nsight_profiler: None,
            amd_profiler: None,
            gpu_vendor: Self::detect_gpu_vendor(),
        }
    }

    fn detect_gpu_vendor() -> GpuVendor {
        // Simple vendor detection logic - in real implementation this would
        // check the actual GPU hardware
        if cfg!(feature = "cuda") {
            GpuVendor::Nvidia
        } else if cfg!(feature = "rocm") {
            GpuVendor::Amd
        } else {
            GpuVendor::Unknown
        }
    }

    pub fn with_optimal_profiler(mut self) -> Self {
        match self.gpu_vendor {
            GpuVendor::Nvidia => {
                self.cuda_profiler = Some(crate::cuda::CudaProfiler::new(0)); // Default to device 0
                self.nsight_profiler = Some(crate::nsight::create_nsight_profiler());
            }
            GpuVendor::Amd => {
                self.amd_profiler = Some(crate::amd::AMDProfiler::new());
            }
            _ => {}
        }
        self
    }

    pub fn start_profiling(&mut self) -> crate::TorshResult<()> {
        match self.gpu_vendor {
            GpuVendor::Nvidia => {
                if let Some(ref mut cuda) = self.cuda_profiler {
                    // CUDA profiler start logic
                }
                if let Some(ref mut nsight) = self.nsight_profiler {
                    // NSight profiler start logic
                }
            }
            GpuVendor::Amd => {
                if let Some(ref mut amd) = self.amd_profiler {
                    // AMD profiler start logic
                }
            }
            _ => {}
        }
        Ok(())
    }

    pub fn stop_profiling(&mut self) -> crate::TorshResult<()> {
        // Stop all active GPU profilers
        Ok(())
    }

    pub fn get_gpu_stats(&self) -> crate::TorshResult<GpuStats> {
        match self.gpu_vendor {
            GpuVendor::Nvidia => {
                if let Some(ref cuda) = self.cuda_profiler {
                    return Ok(GpuStats {
                        vendor: self.gpu_vendor.clone(),
                        memory_used: 0, // Would get from CUDA API
                        memory_total: 0,
                        utilization: 0.0,
                        temperature: None,
                    });
                }
            }
            GpuVendor::Amd => {
                if let Some(ref amd) = self.amd_profiler {
                    return Ok(GpuStats {
                        vendor: self.gpu_vendor.clone(),
                        memory_used: 0, // Would get from ROCm API
                        memory_total: 0,
                        utilization: 0.0,
                        temperature: None,
                    });
                }
            }
            _ => {}
        }

        Err(crate::TorshError::Other(
            "No GPU profiler available".to_string(),
        ))
    }
}

#[derive(Debug, Clone)]
pub struct GpuStats {
    pub vendor: GpuVendor,
    pub memory_used: u64,
    pub memory_total: u64,
    pub utilization: f64,
    pub temperature: Option<f64>,
}

impl Default for GpuProfilerPlatform {
    fn default() -> Self {
        Self::new()
    }
}
