use crate::error::{FfiError, FfiResult};

/// Device types for tensor operations
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceType {
    /// CPU device
    CPU,
    /// CUDA GPU device with device index
    CUDA(usize),
    /// Metal GPU device (future support)
    Metal(usize),
    /// WebGPU device (future support)
    WebGPU(usize),
}

impl DeviceType {
    /// Get the device name as a string
    pub fn name(&self) -> String {
        match self {
            DeviceType::CPU => "cpu".to_string(),
            DeviceType::CUDA(idx) => format!("cuda:{}", idx),
            DeviceType::Metal(idx) => format!("metal:{}", idx),
            DeviceType::WebGPU(idx) => format!("webgpu:{}", idx),
        }
    }

    /// Parse device string into DeviceType
    pub fn from_string(device_str: &str) -> FfiResult<Self> {
        if device_str == "cpu" {
            Ok(DeviceType::CPU)
        } else if device_str.starts_with("cuda:") {
            let idx_str = &device_str[5..];
            let idx = idx_str
                .parse::<usize>()
                .map_err(|_| FfiError::DeviceTransfer {
                    message: format!("Invalid CUDA device index: {}", idx_str),
                })?;
            Ok(DeviceType::CUDA(idx))
        } else if device_str.starts_with("metal:") {
            let idx_str = &device_str[6..];
            let idx = idx_str
                .parse::<usize>()
                .map_err(|_| FfiError::DeviceTransfer {
                    message: format!("Invalid Metal device index: {}", idx_str),
                })?;
            Ok(DeviceType::Metal(idx))
        } else if device_str.starts_with("webgpu:") {
            let idx_str = &device_str[7..];
            let idx = idx_str
                .parse::<usize>()
                .map_err(|_| FfiError::DeviceTransfer {
                    message: format!("Invalid WebGPU device index: {}", idx_str),
                })?;
            Ok(DeviceType::WebGPU(idx))
        } else {
            Err(FfiError::DeviceTransfer {
                message: format!("Unsupported device type: {}", device_str),
            })
        }
    }

    /// Check if this device is available
    pub fn is_available(&self) -> bool {
        match self {
            DeviceType::CPU => true,
            DeviceType::CUDA(idx) => {
                // For now, we'll use a simple check - in a real implementation,
                // this would check if CUDA is available and the device exists
                *idx < device_count_cuda()
            }
            DeviceType::Metal(_) => false,  // Not implemented yet
            DeviceType::WebGPU(_) => false, // Not implemented yet
        }
    }

    /// Get device compute capability or properties
    pub fn properties(&self) -> DeviceProperties {
        match self {
            DeviceType::CPU => DeviceProperties {
                name: "CPU".to_string(),
                memory_total: get_system_memory(),
                memory_available: get_available_memory(),
                compute_capability: "N/A".to_string(),
                multi_processor_count: num_cpus::get(),
                is_integrated: false,
            },
            DeviceType::CUDA(idx) => DeviceProperties {
                name: format!("CUDA Device {}", idx),
                memory_total: get_cuda_memory(*idx).unwrap_or(0),
                memory_available: get_cuda_available_memory(*idx).unwrap_or(0),
                compute_capability: get_cuda_compute_capability(*idx)
                    .unwrap_or("Unknown".to_string()),
                multi_processor_count: get_cuda_sm_count(*idx).unwrap_or(0),
                is_integrated: false,
            },
            DeviceType::Metal(idx) => DeviceProperties {
                name: format!("Metal Device {}", idx),
                memory_total: 0,
                memory_available: 0,
                compute_capability: "Not implemented".to_string(),
                multi_processor_count: 0,
                is_integrated: true,
            },
            DeviceType::WebGPU(idx) => DeviceProperties {
                name: format!("WebGPU Device {}", idx),
                memory_total: 0,
                memory_available: 0,
                compute_capability: "Not implemented".to_string(),
                multi_processor_count: 0,
                is_integrated: false,
            },
        }
    }
}

/// Device properties information
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    pub name: String,
    pub memory_total: usize,
    pub memory_available: usize,
    pub compute_capability: String,
    pub multi_processor_count: usize,
    pub is_integrated: bool,
}

/// Device utility functions (simplified implementations)
#[allow(dead_code)]
fn device_count_cuda() -> usize {
    // In a real implementation, this would query CUDA runtime
    // For now, we'll assume 0 or 1 device based on availability
    if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
        1
    } else {
        0
    }
}

#[allow(dead_code)]
fn get_system_memory() -> usize {
    // Get total system memory (simplified)
    8 * 1024 * 1024 * 1024 // 8GB default
}

#[allow(dead_code)]
fn get_available_memory() -> usize {
    // Get available system memory (simplified)
    4 * 1024 * 1024 * 1024 // 4GB default
}

#[allow(dead_code)]
fn get_cuda_memory(device_idx: usize) -> Option<usize> {
    // In a real implementation, this would query CUDA device memory
    if device_idx < device_count_cuda() {
        Some(8 * 1024 * 1024 * 1024) // 8GB default
    } else {
        None
    }
}

#[allow(dead_code)]
fn get_cuda_available_memory(device_idx: usize) -> Option<usize> {
    // In a real implementation, this would query available CUDA memory
    if device_idx < device_count_cuda() {
        Some(6 * 1024 * 1024 * 1024) // 6GB available
    } else {
        None
    }
}

#[allow(dead_code)]
fn get_cuda_compute_capability(device_idx: usize) -> Option<String> {
    // In a real implementation, this would query CUDA compute capability
    if device_idx < device_count_cuda() {
        Some("7.5".to_string()) // Common compute capability
    } else {
        None
    }
}

#[allow(dead_code)]
fn get_cuda_sm_count(device_idx: usize) -> Option<usize> {
    // In a real implementation, this would query CUDA SM count
    if device_idx < device_count_cuda() {
        Some(68) // Common SM count for modern GPUs
    } else {
        None
    }
}
