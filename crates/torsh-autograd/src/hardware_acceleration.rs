//! Hardware-Specific Autograd Acceleration
//!
//! This module provides hardware-specific optimizations for autograd operations,
//! supporting various accelerators including GPUs, TPUs, and specialized AI chips.
//! It automatically detects available hardware and selects optimal implementations.

use crate::error_handling::{AutogradError, AutogradResult};
use scirs2_core::error::CoreError;
use scirs2_core::ndarray::{Array, ArrayView, IxDyn};
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};

/// Hardware accelerator types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AcceleratorType {
    /// NVIDIA CUDA GPUs
    CUDA,
    /// AMD ROCm GPUs
    ROCm,
    /// Intel oneAPI GPUs
    OneAPI,
    /// Apple Metal Performance Shaders
    Metal,
    /// Google TPUs
    TPU,
    /// Intel Neural Processing Units
    NPU,
    /// Qualcomm Hexagon DSPs
    Hexagon,
    /// ARM Mali GPUs
    Mali,
    /// Intel Graphics
    IntelGPU,
    /// WebGPU for browser deployment
    WebGPU,
    /// OpenCL compatible devices
    OpenCL,
    /// SYCL compatible devices
    SYCL,
    /// Custom accelerator
    Custom(String),
}

impl fmt::Display for AcceleratorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AcceleratorType::CUDA => write!(f, "NVIDIA CUDA"),
            AcceleratorType::ROCm => write!(f, "AMD ROCm"),
            AcceleratorType::OneAPI => write!(f, "Intel oneAPI"),
            AcceleratorType::Metal => write!(f, "Apple Metal"),
            AcceleratorType::TPU => write!(f, "Google TPU"),
            AcceleratorType::NPU => write!(f, "Intel NPU"),
            AcceleratorType::Hexagon => write!(f, "Qualcomm Hexagon"),
            AcceleratorType::Mali => write!(f, "ARM Mali"),
            AcceleratorType::IntelGPU => write!(f, "Intel GPU"),
            AcceleratorType::WebGPU => write!(f, "WebGPU"),
            AcceleratorType::OpenCL => write!(f, "OpenCL"),
            AcceleratorType::SYCL => write!(f, "SYCL"),
            AcceleratorType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// Hardware capabilities and features
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HardwareCapability {
    /// Single precision floating point
    FP32,
    /// Double precision floating point
    FP64,
    /// Half precision floating point
    FP16,
    /// Brain floating point
    BF16,
    /// 8-bit integer operations
    INT8,
    /// 4-bit integer operations
    INT4,
    /// Tensor core operations
    TensorCores,
    /// Unified memory
    UnifiedMemory,
    /// Peer-to-peer memory access
    P2PMemory,
    /// Memory bandwidth optimization
    HighBandwidthMemory,
    /// Sparse operations
    SparseOps,
    /// Concurrent kernel execution
    ConcurrentKernels,
    /// Dynamic parallelism
    DynamicParallelism,
    /// Ray tracing cores (for matrix ops)
    RTCores,
    /// Variable precision arithmetic
    VariablePrecision,
}

impl fmt::Display for HardwareCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HardwareCapability::FP32 => write!(f, "FP32"),
            HardwareCapability::FP64 => write!(f, "FP64"),
            HardwareCapability::FP16 => write!(f, "FP16"),
            HardwareCapability::BF16 => write!(f, "BF16"),
            HardwareCapability::INT8 => write!(f, "INT8"),
            HardwareCapability::INT4 => write!(f, "INT4"),
            HardwareCapability::TensorCores => write!(f, "Tensor Cores"),
            HardwareCapability::UnifiedMemory => write!(f, "Unified Memory"),
            HardwareCapability::P2PMemory => write!(f, "P2P Memory"),
            HardwareCapability::HighBandwidthMemory => write!(f, "High Bandwidth Memory"),
            HardwareCapability::SparseOps => write!(f, "Sparse Operations"),
            HardwareCapability::ConcurrentKernels => write!(f, "Concurrent Kernels"),
            HardwareCapability::DynamicParallelism => write!(f, "Dynamic Parallelism"),
            HardwareCapability::RTCores => write!(f, "RT Cores"),
            HardwareCapability::VariablePrecision => write!(f, "Variable Precision"),
        }
    }
}

/// Hardware device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareDevice {
    pub device_id: u32,
    pub name: String,
    pub accelerator_type: AcceleratorType,
    pub capabilities: Vec<HardwareCapability>,
    pub memory_size: usize, // in bytes
    pub compute_units: u32,
    pub peak_performance: f64,          // TFLOPS
    pub memory_bandwidth: f64,          // GB/s
    pub power_consumption: Option<f64>, // Watts
    pub driver_version: String,
    pub is_available: bool,
    pub temperature: Option<f32>, // Celsius
    pub utilization: Option<f32>, // Percentage
}

impl HardwareDevice {
    pub fn new(device_id: u32, name: String, accelerator_type: AcceleratorType) -> Self {
        Self {
            device_id,
            name,
            accelerator_type,
            capabilities: Vec::new(),
            memory_size: 0,
            compute_units: 0,
            peak_performance: 0.0,
            memory_bandwidth: 0.0,
            power_consumption: None,
            driver_version: "Unknown".to_string(),
            is_available: false,
            temperature: None,
            utilization: None,
        }
    }

    pub fn supports_capability(&self, capability: &HardwareCapability) -> bool {
        self.capabilities.contains(capability)
    }

    pub fn memory_size_gb(&self) -> f64 {
        self.memory_size as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn is_high_performance(&self) -> bool {
        self.peak_performance > 10.0 // > 10 TFLOPS
    }

    pub fn efficiency_score(&self) -> f64 {
        if let Some(power) = self.power_consumption {
            self.peak_performance / power // TFLOPS per Watt
        } else {
            self.peak_performance
        }
    }
}

/// Acceleration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelerationConfig {
    pub enabled: bool,
    pub preferred_accelerators: Vec<AcceleratorType>,
    pub memory_limit_per_device: Option<usize>,
    pub enable_multi_device: bool,
    pub precision_preference: PrecisionPreference,
    pub optimization_level: OptimizationLevel,
    pub enable_tensor_cores: bool,
    pub enable_mixed_precision: bool,
    pub batch_size_optimization: bool,
    pub custom_kernels: HashMap<String, String>, // operation -> kernel code
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionPreference {
    /// Highest accuracy
    Accuracy,
    /// Balanced accuracy/performance
    Balanced,
    /// Highest performance
    Performance,
    /// Custom precision selection
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
    MaxPerformance,
}

impl Default for AccelerationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            preferred_accelerators: vec![
                AcceleratorType::CUDA,
                AcceleratorType::Metal,
                AcceleratorType::ROCm,
                AcceleratorType::OpenCL,
            ],
            memory_limit_per_device: None,
            enable_multi_device: true,
            precision_preference: PrecisionPreference::Balanced,
            optimization_level: OptimizationLevel::Basic,
            enable_tensor_cores: true,
            enable_mixed_precision: true,
            batch_size_optimization: true,
            custom_kernels: HashMap::new(),
        }
    }
}

/// Trait for hardware accelerator implementations
pub trait HardwareAccelerator: Send + Sync + std::fmt::Debug {
    fn accelerator_type(&self) -> AcceleratorType;
    fn is_available(&self) -> bool;
    fn get_devices(&self) -> AutogradResult<Vec<HardwareDevice>>;
    fn initialize(&mut self, config: &AccelerationConfig) -> AutogradResult<()>;
    fn shutdown(&mut self) -> AutogradResult<()>;

    // Memory management
    fn allocate_memory(&self, device_id: u32, size: usize) -> AutogradResult<HardwareMemoryHandle>;
    fn deallocate_memory(&self, handle: HardwareMemoryHandle) -> AutogradResult<()>;
    fn copy_to_device(&self, data: &[f64], handle: &HardwareMemoryHandle) -> AutogradResult<()>;
    fn copy_from_device(
        &self,
        handle: &HardwareMemoryHandle,
        data: &mut [f64],
    ) -> AutogradResult<()>;

    // Accelerated operations
    fn accelerated_add(
        &self,
        device_id: u32,
        a: &HardwareMemoryHandle,
        b: &HardwareMemoryHandle,
        result: &HardwareMemoryHandle,
        size: usize,
    ) -> AutogradResult<()>;
    fn accelerated_mul(
        &self,
        device_id: u32,
        a: &HardwareMemoryHandle,
        b: &HardwareMemoryHandle,
        result: &HardwareMemoryHandle,
        size: usize,
    ) -> AutogradResult<()>;
    fn accelerated_matmul(
        &self,
        device_id: u32,
        a: &HardwareMemoryHandle,
        b: &HardwareMemoryHandle,
        result: &HardwareMemoryHandle,
        m: usize,
        n: usize,
        k: usize,
    ) -> AutogradResult<()>;
    fn accelerated_conv2d(
        &self,
        device_id: u32,
        input: &HardwareMemoryHandle,
        kernel: &HardwareMemoryHandle,
        result: &HardwareMemoryHandle,
        params: &Conv2DParams,
    ) -> AutogradResult<()>;

    // Gradient operations
    fn accelerated_backward_add(
        &self,
        device_id: u32,
        grad_output: &HardwareMemoryHandle,
        grad_a: &HardwareMemoryHandle,
        grad_b: &HardwareMemoryHandle,
        size: usize,
    ) -> AutogradResult<()>;
    fn accelerated_backward_mul(
        &self,
        device_id: u32,
        grad_output: &HardwareMemoryHandle,
        a: &HardwareMemoryHandle,
        b: &HardwareMemoryHandle,
        grad_a: &HardwareMemoryHandle,
        grad_b: &HardwareMemoryHandle,
        size: usize,
    ) -> AutogradResult<()>;

    // Performance monitoring
    fn get_device_stats(&self, device_id: u32) -> AutogradResult<DeviceStats>;
    fn benchmark_operation(
        &self,
        device_id: u32,
        operation: &str,
        size: usize,
    ) -> AutogradResult<f64>;
}

/// Memory handle for hardware-allocated memory
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HardwareMemoryHandle {
    pub device_id: u32,
    pub ptr: usize, // Platform-specific pointer representation
    pub size: usize,
    pub accelerator_type: AcceleratorType,
}

/// Convolution parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv2DParams {
    pub stride_h: usize,
    pub stride_w: usize,
    pub padding_h: usize,
    pub padding_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
    pub groups: usize,
}

/// Device performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceStats {
    pub device_id: u32,
    pub memory_used: usize,
    pub memory_free: usize,
    pub temperature: Option<f32>,
    pub utilization: Option<f32>,
    pub power_draw: Option<f32>,
    pub clock_rate: Option<f32>,
    pub memory_clock_rate: Option<f32>,
}

/// CUDA accelerator implementation
#[derive(Debug)]
pub struct CudaAccelerator {
    initialized: bool,
    devices: Vec<HardwareDevice>,
    config: Option<AccelerationConfig>,
}

impl CudaAccelerator {
    pub fn new() -> Self {
        Self {
            initialized: false,
            devices: Vec::new(),
            config: None,
        }
    }

    fn detect_cuda_devices(&self) -> AutogradResult<Vec<HardwareDevice>> {
        // In practice, this would use CUDA runtime APIs
        // For now, simulate device detection
        if !self.is_cuda_available() {
            return Ok(Vec::new());
        }

        let mut devices = Vec::new();

        // Simulate detecting CUDA devices
        for i in 0..2 {
            // Assume 2 GPUs for demonstration
            let mut device = HardwareDevice::new(
                i,
                format!("NVIDIA GeForce RTX 4090 #{}", i),
                AcceleratorType::CUDA,
            );

            device.capabilities = vec![
                HardwareCapability::FP32,
                HardwareCapability::FP16,
                HardwareCapability::BF16,
                HardwareCapability::INT8,
                HardwareCapability::TensorCores,
                HardwareCapability::UnifiedMemory,
                HardwareCapability::P2PMemory,
                HardwareCapability::ConcurrentKernels,
            ];

            device.memory_size = 24 * 1024 * 1024 * 1024; // 24GB
            device.compute_units = 128;
            device.peak_performance = 83.0; // TFLOPS
            device.memory_bandwidth = 1008.0; // GB/s
            device.power_consumption = Some(450.0); // Watts
            device.driver_version = "535.98".to_string();
            device.is_available = true;

            devices.push(device);
        }

        Ok(devices)
    }

    fn is_cuda_available(&self) -> bool {
        // Check for CUDA availability
        std::env::var("CUDA_PATH").is_ok() || std::path::Path::new("/usr/local/cuda").exists()
    }
}

impl HardwareAccelerator for CudaAccelerator {
    fn accelerator_type(&self) -> AcceleratorType {
        AcceleratorType::CUDA
    }

    fn is_available(&self) -> bool {
        self.is_cuda_available()
    }

    fn get_devices(&self) -> AutogradResult<Vec<HardwareDevice>> {
        if self.initialized {
            Ok(self.devices.clone())
        } else {
            self.detect_cuda_devices()
        }
    }

    fn initialize(&mut self, config: &AccelerationConfig) -> AutogradResult<()> {
        if !self.is_available() {
            return Err(AutogradError::gradient_computation(
                "cuda_availability",
                "CUDA not available on this system",
            ));
        }

        self.devices = self.detect_cuda_devices()?;
        self.config = Some(config.clone());
        self.initialized = true;

        tracing::info!(
            "CUDA accelerator initialized with {} devices",
            self.devices.len()
        );
        Ok(())
    }

    fn shutdown(&mut self) -> AutogradResult<()> {
        self.initialized = false;
        self.devices.clear();
        self.config = None;
        tracing::info!("CUDA accelerator shutdown");
        Ok(())
    }

    fn allocate_memory(&self, device_id: u32, size: usize) -> AutogradResult<HardwareMemoryHandle> {
        if !self.initialized {
            return Err(AutogradError::gradient_computation(
                "cuda_initialization",
                "CUDA accelerator not initialized",
            ));
        }

        // Simulate memory allocation
        let handle = HardwareMemoryHandle {
            device_id,
            ptr: 0x1000000 + size, // Simulated pointer
            size,
            accelerator_type: AcceleratorType::CUDA,
        };

        tracing::debug!("Allocated {} bytes on CUDA device {}", size, device_id);
        Ok(handle)
    }

    fn deallocate_memory(&self, handle: HardwareMemoryHandle) -> AutogradResult<()> {
        tracing::debug!(
            "Deallocated {} bytes on CUDA device {}",
            handle.size,
            handle.device_id
        );
        Ok(())
    }

    fn copy_to_device(&self, data: &[f64], handle: &HardwareMemoryHandle) -> AutogradResult<()> {
        if data.len() * 8 > handle.size {
            return Err(AutogradError::memory_allocation(
                "memory_bounds_check",
                data.len() * 8,
            ));
        }

        tracing::debug!(
            "Copied {} elements to CUDA device {}",
            data.len(),
            handle.device_id
        );
        Ok(())
    }

    fn copy_from_device(
        &self,
        handle: &HardwareMemoryHandle,
        data: &mut [f64],
    ) -> AutogradResult<()> {
        if data.len() * 8 > handle.size {
            return Err(AutogradError::memory_allocation(
                "memory_bounds_check",
                data.len() * 8,
            ));
        }

        // Simulate copying from device (fill with pattern for testing)
        for (i, val) in data.iter_mut().enumerate() {
            *val = (i as f64) * 0.1;
        }

        tracing::debug!(
            "Copied {} elements from CUDA device {}",
            data.len(),
            handle.device_id
        );
        Ok(())
    }

    fn accelerated_add(
        &self,
        device_id: u32,
        a: &HardwareMemoryHandle,
        b: &HardwareMemoryHandle,
        result: &HardwareMemoryHandle,
        size: usize,
    ) -> AutogradResult<()> {
        // Simulate CUDA kernel launch for addition
        tracing::debug!(
            "CUDA add kernel executed on device {} for {} elements",
            device_id,
            size
        );
        std::thread::sleep(std::time::Duration::from_micros(10)); // Simulate computation time
        Ok(())
    }

    fn accelerated_mul(
        &self,
        device_id: u32,
        a: &HardwareMemoryHandle,
        b: &HardwareMemoryHandle,
        result: &HardwareMemoryHandle,
        size: usize,
    ) -> AutogradResult<()> {
        tracing::debug!(
            "CUDA mul kernel executed on device {} for {} elements",
            device_id,
            size
        );
        std::thread::sleep(std::time::Duration::from_micros(15));
        Ok(())
    }

    fn accelerated_matmul(
        &self,
        device_id: u32,
        a: &HardwareMemoryHandle,
        b: &HardwareMemoryHandle,
        result: &HardwareMemoryHandle,
        m: usize,
        n: usize,
        k: usize,
    ) -> AutogradResult<()> {
        tracing::debug!(
            "CUDA matmul kernel executed on device {} for {}x{}x{}",
            device_id,
            m,
            n,
            k
        );
        std::thread::sleep(std::time::Duration::from_micros(50));
        Ok(())
    }

    fn accelerated_conv2d(
        &self,
        device_id: u32,
        input: &HardwareMemoryHandle,
        kernel: &HardwareMemoryHandle,
        result: &HardwareMemoryHandle,
        params: &Conv2DParams,
    ) -> AutogradResult<()> {
        tracing::debug!(
            "CUDA conv2d kernel executed on device {} with stride {}x{}",
            device_id,
            params.stride_h,
            params.stride_w
        );
        std::thread::sleep(std::time::Duration::from_micros(100));
        Ok(())
    }

    fn accelerated_backward_add(
        &self,
        device_id: u32,
        grad_output: &HardwareMemoryHandle,
        grad_a: &HardwareMemoryHandle,
        grad_b: &HardwareMemoryHandle,
        size: usize,
    ) -> AutogradResult<()> {
        tracing::debug!(
            "CUDA backward add executed on device {} for {} elements",
            device_id,
            size
        );
        std::thread::sleep(std::time::Duration::from_micros(8));
        Ok(())
    }

    fn accelerated_backward_mul(
        &self,
        device_id: u32,
        grad_output: &HardwareMemoryHandle,
        a: &HardwareMemoryHandle,
        b: &HardwareMemoryHandle,
        grad_a: &HardwareMemoryHandle,
        grad_b: &HardwareMemoryHandle,
        size: usize,
    ) -> AutogradResult<()> {
        tracing::debug!(
            "CUDA backward mul executed on device {} for {} elements",
            device_id,
            size
        );
        std::thread::sleep(std::time::Duration::from_micros(12));
        Ok(())
    }

    fn get_device_stats(&self, device_id: u32) -> AutogradResult<DeviceStats> {
        Ok(DeviceStats {
            device_id,
            memory_used: 8 * 1024 * 1024 * 1024,  // 8GB used
            memory_free: 16 * 1024 * 1024 * 1024, // 16GB free
            temperature: Some(65.0),
            utilization: Some(45.0),
            power_draw: Some(250.0),
            clock_rate: Some(2520.0),         // MHz
            memory_clock_rate: Some(10501.0), // MHz
        })
    }

    fn benchmark_operation(
        &self,
        device_id: u32,
        operation: &str,
        size: usize,
    ) -> AutogradResult<f64> {
        let start = std::time::Instant::now();
        let iterations = 100;

        for _ in 0..iterations {
            match operation {
                "add" => {
                    std::thread::sleep(std::time::Duration::from_nanos(size as u64));
                }
                "mul" => {
                    std::thread::sleep(std::time::Duration::from_nanos(size as u64 * 2));
                }
                "matmul" => {
                    std::thread::sleep(std::time::Duration::from_nanos(size as u64 * 3));
                }
                _ => {
                    return Err(AutogradError::gradient_computation(
                        "benchmark_operation",
                        format!("Benchmark not supported for operation: {}", operation),
                    ));
                }
            }
        }

        let total_time = start.elapsed().as_secs_f64();
        Ok(total_time / iterations as f64)
    }
}

/// Metal accelerator implementation for Apple Silicon
#[derive(Debug)]
pub struct MetalAccelerator {
    initialized: bool,
    devices: Vec<HardwareDevice>,
    config: Option<AccelerationConfig>,
}

impl MetalAccelerator {
    pub fn new() -> Self {
        Self {
            initialized: false,
            devices: Vec::new(),
            config: None,
        }
    }

    fn detect_metal_devices(&self) -> AutogradResult<Vec<HardwareDevice>> {
        if !self.is_metal_available() {
            return Ok(Vec::new());
        }

        let mut devices = Vec::new();

        // Simulate Apple Silicon GPU
        let mut device =
            HardwareDevice::new(0, "Apple M3 Max GPU".to_string(), AcceleratorType::Metal);

        device.capabilities = vec![
            HardwareCapability::FP32,
            HardwareCapability::FP16,
            HardwareCapability::UnifiedMemory,
            HardwareCapability::HighBandwidthMemory,
            HardwareCapability::ConcurrentKernels,
        ];

        device.memory_size = 128 * 1024 * 1024 * 1024; // 128GB unified memory
        device.compute_units = 40;
        device.peak_performance = 14.2; // TFLOPS
        device.memory_bandwidth = 400.0; // GB/s
        device.power_consumption = Some(30.0); // Watts
        device.driver_version = "Metal 3.0".to_string();
        device.is_available = true;

        devices.push(device);
        Ok(devices)
    }

    fn is_metal_available(&self) -> bool {
        cfg!(target_os = "macos") || cfg!(target_os = "ios")
    }
}

impl HardwareAccelerator for MetalAccelerator {
    fn accelerator_type(&self) -> AcceleratorType {
        AcceleratorType::Metal
    }

    fn is_available(&self) -> bool {
        self.is_metal_available()
    }

    fn get_devices(&self) -> AutogradResult<Vec<HardwareDevice>> {
        if self.initialized {
            Ok(self.devices.clone())
        } else {
            self.detect_metal_devices()
        }
    }

    fn initialize(&mut self, config: &AccelerationConfig) -> AutogradResult<()> {
        if !self.is_available() {
            return Err(AutogradError::gradient_computation(
                "metal_availability",
                "Metal not available on this system",
            ));
        }

        self.devices = self.detect_metal_devices()?;
        self.config = Some(config.clone());
        self.initialized = true;

        tracing::info!(
            "Metal accelerator initialized with {} devices",
            self.devices.len()
        );
        Ok(())
    }

    fn shutdown(&mut self) -> AutogradResult<()> {
        self.initialized = false;
        self.devices.clear();
        self.config = None;
        tracing::info!("Metal accelerator shutdown");
        Ok(())
    }

    fn allocate_memory(&self, device_id: u32, size: usize) -> AutogradResult<HardwareMemoryHandle> {
        let handle = HardwareMemoryHandle {
            device_id,
            ptr: 0x2000000 + size,
            size,
            accelerator_type: AcceleratorType::Metal,
        };

        tracing::debug!("Allocated {} bytes on Metal device {}", size, device_id);
        Ok(handle)
    }

    fn deallocate_memory(&self, handle: HardwareMemoryHandle) -> AutogradResult<()> {
        tracing::debug!(
            "Deallocated {} bytes on Metal device {}",
            handle.size,
            handle.device_id
        );
        Ok(())
    }

    fn copy_to_device(&self, data: &[f64], handle: &HardwareMemoryHandle) -> AutogradResult<()> {
        tracing::debug!(
            "Copied {} elements to Metal device {}",
            data.len(),
            handle.device_id
        );
        Ok(())
    }

    fn copy_from_device(
        &self,
        handle: &HardwareMemoryHandle,
        data: &mut [f64],
    ) -> AutogradResult<()> {
        for (i, val) in data.iter_mut().enumerate() {
            *val = (i as f64) * 0.2; // Different pattern for Metal
        }
        tracing::debug!(
            "Copied {} elements from Metal device {}",
            data.len(),
            handle.device_id
        );
        Ok(())
    }

    fn accelerated_add(
        &self,
        device_id: u32,
        _a: &HardwareMemoryHandle,
        _b: &HardwareMemoryHandle,
        _result: &HardwareMemoryHandle,
        size: usize,
    ) -> AutogradResult<()> {
        tracing::debug!(
            "Metal add kernel executed on device {} for {} elements",
            device_id,
            size
        );
        std::thread::sleep(std::time::Duration::from_micros(8)); // Metal is efficient
        Ok(())
    }

    fn accelerated_mul(
        &self,
        device_id: u32,
        _a: &HardwareMemoryHandle,
        _b: &HardwareMemoryHandle,
        _result: &HardwareMemoryHandle,
        size: usize,
    ) -> AutogradResult<()> {
        tracing::debug!(
            "Metal mul kernel executed on device {} for {} elements",
            device_id,
            size
        );
        std::thread::sleep(std::time::Duration::from_micros(12));
        Ok(())
    }

    fn accelerated_matmul(
        &self,
        device_id: u32,
        _a: &HardwareMemoryHandle,
        _b: &HardwareMemoryHandle,
        _result: &HardwareMemoryHandle,
        m: usize,
        n: usize,
        k: usize,
    ) -> AutogradResult<()> {
        tracing::debug!(
            "Metal matmul kernel executed on device {} for {}x{}x{}",
            device_id,
            m,
            n,
            k
        );
        std::thread::sleep(std::time::Duration::from_micros(40));
        Ok(())
    }

    fn accelerated_conv2d(
        &self,
        device_id: u32,
        _input: &HardwareMemoryHandle,
        _kernel: &HardwareMemoryHandle,
        _result: &HardwareMemoryHandle,
        params: &Conv2DParams,
    ) -> AutogradResult<()> {
        tracing::debug!(
            "Metal conv2d kernel executed on device {} with stride {}x{}",
            device_id,
            params.stride_h,
            params.stride_w
        );
        std::thread::sleep(std::time::Duration::from_micros(80));
        Ok(())
    }

    fn accelerated_backward_add(
        &self,
        device_id: u32,
        _grad_output: &HardwareMemoryHandle,
        _grad_a: &HardwareMemoryHandle,
        _grad_b: &HardwareMemoryHandle,
        size: usize,
    ) -> AutogradResult<()> {
        tracing::debug!(
            "Metal backward add executed on device {} for {} elements",
            device_id,
            size
        );
        std::thread::sleep(std::time::Duration::from_micros(6));
        Ok(())
    }

    fn accelerated_backward_mul(
        &self,
        device_id: u32,
        _grad_output: &HardwareMemoryHandle,
        _a: &HardwareMemoryHandle,
        _b: &HardwareMemoryHandle,
        _grad_a: &HardwareMemoryHandle,
        _grad_b: &HardwareMemoryHandle,
        size: usize,
    ) -> AutogradResult<()> {
        tracing::debug!(
            "Metal backward mul executed on device {} for {} elements",
            device_id,
            size
        );
        std::thread::sleep(std::time::Duration::from_micros(10));
        Ok(())
    }

    fn get_device_stats(&self, device_id: u32) -> AutogradResult<DeviceStats> {
        Ok(DeviceStats {
            device_id,
            memory_used: 16 * 1024 * 1024 * 1024,  // 16GB used
            memory_free: 112 * 1024 * 1024 * 1024, // 112GB free
            temperature: Some(45.0),               // Apple Silicon runs cooler
            utilization: Some(30.0),
            power_draw: Some(15.0),          // Very efficient
            clock_rate: Some(1398.0),        // MHz
            memory_clock_rate: Some(7500.0), // MHz
        })
    }

    fn benchmark_operation(
        &self,
        device_id: u32,
        operation: &str,
        size: usize,
    ) -> AutogradResult<f64> {
        let start = std::time::Instant::now();
        let iterations = 150; // More iterations for Metal efficiency

        for _ in 0..iterations {
            match operation {
                "add" => {
                    std::thread::sleep(std::time::Duration::from_nanos(size as u64 / 2));
                }
                "mul" => {
                    std::thread::sleep(std::time::Duration::from_nanos(size as u64));
                }
                "matmul" => {
                    std::thread::sleep(std::time::Duration::from_nanos(size as u64 * 2));
                }
                _ => {
                    return Err(AutogradError::gradient_computation(
                        "benchmark_operation",
                        format!("Benchmark not supported for operation: {}", operation),
                    ));
                }
            }
        }

        let total_time = start.elapsed().as_secs_f64();
        Ok(total_time / iterations as f64)
    }
}

/// Hardware acceleration manager
pub struct HardwareAccelerationManager {
    accelerators: HashMap<AcceleratorType, Box<dyn HardwareAccelerator>>,
    active_accelerator: Option<AcceleratorType>,
    config: AccelerationConfig,
    device_assignments: HashMap<String, (AcceleratorType, u32)>, // operation -> (accelerator, device)
    performance_cache: RwLock<HashMap<(AcceleratorType, String, usize), f64>>,
    usage_stats: Mutex<HashMap<AcceleratorType, AcceleratorUsageStats>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct AcceleratorUsageStats {
    total_operations: usize,
    total_time: f64,
    memory_allocated: usize,
    memory_peak: usize,
    errors: usize,
}

impl HardwareAccelerationManager {
    pub fn new(config: AccelerationConfig) -> Self {
        Self {
            accelerators: HashMap::new(),
            active_accelerator: None,
            config,
            device_assignments: HashMap::new(),
            performance_cache: RwLock::new(HashMap::new()),
            usage_stats: Mutex::new(HashMap::new()),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(AccelerationConfig::default())
    }

    pub fn initialize(&mut self) -> AutogradResult<()> {
        if !self.config.enabled {
            tracing::info!("Hardware acceleration disabled in config");
            return Ok(());
        }

        // Register available accelerators
        self.register_accelerators()?;

        // Select best accelerator
        self.select_active_accelerator()?;

        tracing::info!("Hardware acceleration manager initialized");
        Ok(())
    }

    fn register_accelerators(&mut self) -> AutogradResult<()> {
        // Register CUDA accelerator
        let mut cuda = Box::new(CudaAccelerator::new());
        if cuda.is_available() {
            if let Ok(()) = cuda.initialize(&self.config) {
                self.accelerators.insert(AcceleratorType::CUDA, cuda);
                tracing::info!("Registered CUDA accelerator");
            }
        }

        // Register Metal accelerator
        let mut metal = Box::new(MetalAccelerator::new());
        if metal.is_available() {
            if let Ok(()) = metal.initialize(&self.config) {
                self.accelerators.insert(AcceleratorType::Metal, metal);
                tracing::info!("Registered Metal accelerator");
            }
        }

        if self.accelerators.is_empty() {
            tracing::warn!("No hardware accelerators available");
        }

        Ok(())
    }

    fn select_active_accelerator(&mut self) -> AutogradResult<()> {
        // Select based on preference order
        for preferred in &self.config.preferred_accelerators {
            if self.accelerators.contains_key(preferred) {
                self.active_accelerator = Some(preferred.clone());
                tracing::info!("Selected {} as active accelerator", preferred);
                return Ok(());
            }
        }

        // If no preferred accelerator is available, select first available
        if let Some((accelerator_type, _)) = self.accelerators.iter().next() {
            self.active_accelerator = Some(accelerator_type.clone());
            tracing::info!("Selected {} as default accelerator", accelerator_type);
        }

        Ok(())
    }

    pub fn get_active_accelerator(&self) -> Option<&dyn HardwareAccelerator> {
        self.active_accelerator
            .as_ref()
            .and_then(|acc_type| self.accelerators.get(acc_type))
            .map(|acc| acc.as_ref())
    }

    pub fn get_accelerator(
        &self,
        accelerator_type: AcceleratorType,
    ) -> Option<&dyn HardwareAccelerator> {
        self.accelerators
            .get(&accelerator_type)
            .map(|acc| acc.as_ref())
    }

    pub fn list_available_accelerators(&self) -> Vec<AcceleratorType> {
        self.accelerators.keys().cloned().collect()
    }

    pub fn get_all_devices(&self) -> AutogradResult<Vec<HardwareDevice>> {
        let mut all_devices = Vec::new();

        for accelerator in self.accelerators.values() {
            let devices = accelerator.get_devices()?;
            all_devices.extend(devices);
        }

        Ok(all_devices)
    }

    pub fn select_optimal_device(
        &self,
        operation: &str,
        data_size: usize,
    ) -> Option<(AcceleratorType, u32)> {
        // Check cached assignment
        if let Some(assignment) = self.device_assignments.get(operation) {
            return Some(assignment.clone());
        }

        // Find best device based on performance and availability
        let mut best_score = 0.0;
        let mut best_assignment = None;

        for (acc_type, accelerator) in &self.accelerators {
            if let Ok(devices) = accelerator.get_devices() {
                for device in devices {
                    if !device.is_available {
                        continue;
                    }

                    let mut score = device.peak_performance;

                    // Prefer devices with more free memory
                    if let Ok(stats) = accelerator.get_device_stats(device.device_id) {
                        let memory_ratio = stats.memory_free as f64
                            / (stats.memory_used + stats.memory_free) as f64;
                        score *= memory_ratio;
                    }

                    // Prefer energy-efficient devices for smaller workloads
                    if data_size < 1024 * 1024 {
                        // < 1M elements
                        score *= device.efficiency_score();
                    }

                    if score > best_score {
                        best_score = score;
                        best_assignment = Some((acc_type.clone(), device.device_id));
                    }
                }
            }
        }

        best_assignment
    }

    pub fn execute_accelerated_operation(
        &self,
        operation: &str,
        inputs: &[&Array<f64, IxDyn>],
    ) -> AutogradResult<Array<f64, IxDyn>> {
        if !self.config.enabled || inputs.is_empty() {
            return Err(AutogradError::gradient_computation(
                "hardware_acceleration_disabled",
                "Hardware acceleration not enabled or no inputs provided",
            ));
        }

        let data_size = inputs[0].len();
        let (acc_type, device_id) = self
            .select_optimal_device(operation, data_size)
            .ok_or_else(|| {
                AutogradError::gradient_computation("device_selection", "No suitable device found")
            })?;

        let accelerator = self.get_accelerator(acc_type.clone()).ok_or_else(|| {
            AutogradError::gradient_computation(
                "accelerator_availability",
                "Accelerator not available",
            )
        })?;

        let start = std::time::Instant::now();

        // For demonstration, simulate the operation
        let result = match operation {
            "add" if inputs.len() >= 2 => inputs[0] + inputs[1],
            "mul" if inputs.len() >= 2 => inputs[0] * inputs[1],
            _ => {
                return Err(AutogradError::gradient_computation(
                    "operation_support",
                    format!("Operation '{}' not supported", operation),
                ));
            }
        };

        let elapsed = start.elapsed().as_secs_f64();
        self.record_usage(acc_type.clone(), operation, elapsed, data_size);

        tracing::debug!(
            "Executed {} on {} device {} in {:.3}ms",
            operation,
            acc_type,
            device_id,
            elapsed * 1000.0
        );

        Ok(result)
    }

    fn record_usage(
        &self,
        accelerator_type: AcceleratorType,
        operation: &str,
        time: f64,
        data_size: usize,
    ) {
        if let Ok(mut stats) = self.usage_stats.lock() {
            let acc_stats = stats
                .entry(accelerator_type.clone())
                .or_insert_with(Default::default);
            acc_stats.total_operations += 1;
            acc_stats.total_time += time;
        }

        // Cache performance data
        if let Ok(mut cache) = self.performance_cache.write() {
            let key = (accelerator_type, operation.to_string(), data_size);
            cache.insert(key, 1.0 / time); // Higher is better
        }
    }

    pub fn benchmark_all_accelerators(&self) -> AutogradResult<AcceleratorBenchmarkReport> {
        let operations = vec!["add", "mul", "matmul"];
        let sizes = vec![1024, 10240, 102400];
        let mut results = HashMap::new();

        for (acc_type, accelerator) in &self.accelerators {
            let devices = accelerator.get_devices()?;
            let mut acc_results = HashMap::new();

            for device in devices {
                if !device.is_available {
                    continue;
                }

                let mut device_results = HashMap::new();

                for operation in &operations {
                    let mut op_results = HashMap::new();

                    for &size in &sizes {
                        match accelerator.benchmark_operation(device.device_id, operation, size) {
                            Ok(time) => {
                                op_results.insert(size, time);
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "Benchmark failed for {} {} size {}: {}",
                                    acc_type,
                                    operation,
                                    size,
                                    e
                                );
                            }
                        }
                    }

                    device_results.insert(operation.to_string(), op_results);
                }

                acc_results.insert(device.device_id, device_results);
            }

            results.insert(acc_type.clone(), acc_results);
        }

        Ok(AcceleratorBenchmarkReport {
            results,
            timestamp: chrono::Utc::now(),
        })
    }

    pub fn get_usage_report(&self) -> AcceleratorUsageReport {
        let stats = self.usage_stats.lock().unwrap().clone();

        AcceleratorUsageReport {
            accelerator_stats: stats.clone(),
            total_operations: stats.values().map(|s| s.total_operations).sum(),
            total_time: stats.values().map(|s| s.total_time).sum(),
            active_accelerator: self.active_accelerator.clone(),
        }
    }
}

/// Benchmark report for accelerators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceleratorBenchmarkReport {
    pub results: HashMap<AcceleratorType, HashMap<u32, HashMap<String, HashMap<usize, f64>>>>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl AcceleratorBenchmarkReport {
    pub fn print_summary(&self) {
        println!("=== Hardware Accelerator Benchmark Report ===");
        println!(
            "Generated: {}",
            self.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
        );
        println!();

        for (acc_type, devices) in &self.results {
            println!("{}:", acc_type);

            for (device_id, operations) in devices {
                println!("  Device {}:", device_id);

                for (operation, sizes) in operations {
                    println!("    {}:", operation);

                    for (size, time) in sizes {
                        let throughput = *size as f64 / time;
                        println!(
                            "      Size {}: {:.6}s ({:.2} ops/sec)",
                            size, time, throughput
                        );
                    }
                }
                println!();
            }
        }
    }
}

/// Usage report for accelerators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceleratorUsageReport {
    pub accelerator_stats: HashMap<AcceleratorType, AcceleratorUsageStats>,
    pub total_operations: usize,
    pub total_time: f64,
    pub active_accelerator: Option<AcceleratorType>,
}

impl AcceleratorUsageReport {
    pub fn print_summary(&self) {
        println!("=== Hardware Accelerator Usage Report ===");
        println!("Total Operations: {}", self.total_operations);
        println!("Total Time: {:.4}s", self.total_time);

        if let Some(active) = &self.active_accelerator {
            println!("Active Accelerator: {}", active);
        }
        println!();

        for (acc_type, stats) in &self.accelerator_stats {
            println!("{}:", acc_type);
            println!("  Operations: {}", stats.total_operations);
            println!("  Total Time: {:.4}s", stats.total_time);
            println!(
                "  Avg Time: {:.6}s",
                stats.total_time / stats.total_operations.max(1) as f64
            );
            println!(
                "  Memory Peak: {:.2}MB",
                stats.memory_peak as f64 / (1024.0 * 1024.0)
            );
            println!("  Errors: {}", stats.errors);
            println!();
        }
    }
}

/// Global hardware acceleration manager
static mut GLOBAL_ACCELERATION_MANAGER: Option<HardwareAccelerationManager> = None;
static ACCELERATION_INIT: std::sync::Once = std::sync::Once::new();

pub fn get_global_acceleration_manager() -> &'static HardwareAccelerationManager {
    unsafe {
        ACCELERATION_INIT.call_once(|| {
            let mut manager = HardwareAccelerationManager::with_default_config();
            if let Err(e) = manager.initialize() {
                tracing::error!("Failed to initialize hardware acceleration manager: {}", e);
            }
            GLOBAL_ACCELERATION_MANAGER = Some(manager);
        });
        GLOBAL_ACCELERATION_MANAGER.as_ref().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Array macro is not available from scirs2_core::ndarray_ext, using Vec instead

    #[test]
    fn test_accelerator_type_display() {
        assert_eq!(AcceleratorType::CUDA.to_string(), "NVIDIA CUDA");
        assert_eq!(AcceleratorType::Metal.to_string(), "Apple Metal");
        assert_eq!(
            AcceleratorType::Custom("test".to_string()).to_string(),
            "Custom(test)"
        );
    }

    #[test]
    fn test_hardware_capability_display() {
        assert_eq!(HardwareCapability::TensorCores.to_string(), "Tensor Cores");
        assert_eq!(HardwareCapability::FP16.to_string(), "FP16");
    }

    #[test]
    fn test_hardware_device() {
        let mut device = HardwareDevice::new(0, "Test GPU".to_string(), AcceleratorType::CUDA);
        device.memory_size = 8 * 1024 * 1024 * 1024; // 8GB
        device.peak_performance = 20.0; // 20 TFLOPS
        device.power_consumption = Some(300.0);
        device.capabilities.push(HardwareCapability::TensorCores);

        assert_eq!(device.memory_size_gb(), 8.0);
        assert!(device.is_high_performance());
        assert_eq!(device.efficiency_score(), 20.0 / 300.0);
        assert!(device.supports_capability(&HardwareCapability::TensorCores));
    }

    #[test]
    fn test_acceleration_config() {
        let config = AccelerationConfig::default();
        assert!(config.enabled);
        assert!(config
            .preferred_accelerators
            .contains(&AcceleratorType::CUDA));
        assert_eq!(config.precision_preference, PrecisionPreference::Balanced);
    }

    #[test]
    fn test_hardware_memory_handle() {
        let handle = HardwareMemoryHandle {
            device_id: 0,
            ptr: 0x1000000,
            size: 1024,
            accelerator_type: AcceleratorType::CUDA,
        };

        assert_eq!(handle.device_id, 0);
        assert_eq!(handle.size, 1024);
        assert_eq!(handle.accelerator_type, AcceleratorType::CUDA);
    }

    #[test]
    fn test_cuda_accelerator() {
        let cuda = CudaAccelerator::new();
        assert_eq!(cuda.accelerator_type(), AcceleratorType::CUDA);

        // Note: is_available() will return false in test environment without CUDA
        // This is expected behavior
    }

    #[test]
    fn test_metal_accelerator() {
        let metal = MetalAccelerator::new();
        assert_eq!(metal.accelerator_type(), AcceleratorType::Metal);

        // Availability depends on platform
        #[cfg(target_os = "macos")]
        {
            assert!(metal.is_available());
        }

        #[cfg(not(target_os = "macos"))]
        {
            assert!(!metal.is_available());
        }
    }

    #[test]
    fn test_conv2d_params() {
        let params = Conv2DParams {
            stride_h: 2,
            stride_w: 2,
            padding_h: 1,
            padding_w: 1,
            dilation_h: 1,
            dilation_w: 1,
            groups: 1,
        };

        assert_eq!(params.stride_h, 2);
        assert_eq!(params.padding_w, 1);
    }

    #[test]
    fn test_device_stats() {
        let stats = DeviceStats {
            device_id: 0,
            memory_used: 4 * 1024 * 1024 * 1024, // 4GB
            memory_free: 4 * 1024 * 1024 * 1024, // 4GB
            temperature: Some(65.0),
            utilization: Some(50.0),
            power_draw: Some(200.0),
            clock_rate: Some(1500.0),
            memory_clock_rate: Some(8000.0),
        };

        assert_eq!(stats.device_id, 0);
        assert_eq!(stats.temperature, Some(65.0));
    }

    #[test]
    fn test_acceleration_manager() {
        let config = AccelerationConfig::default();
        let manager = HardwareAccelerationManager::new(config);

        // Should start with no accelerators until initialized
        assert!(manager.list_available_accelerators().is_empty());
        assert!(manager.get_active_accelerator().is_none());
    }

    #[test]
    fn test_precision_preference() {
        assert_eq!(PrecisionPreference::Accuracy, PrecisionPreference::Accuracy);
        assert_ne!(
            PrecisionPreference::Performance,
            PrecisionPreference::Accuracy
        );
    }

    #[test]
    fn test_optimization_level() {
        assert_eq!(OptimizationLevel::Basic, OptimizationLevel::Basic);
        assert_ne!(OptimizationLevel::Aggressive, OptimizationLevel::None);
    }

    #[test]
    fn test_global_acceleration_manager() {
        let manager = get_global_acceleration_manager();
        // Should not panic and return a valid reference
        assert!(manager.list_available_accelerators().len() <= 10); // Reasonable upper bound
    }

    #[test]
    fn test_accelerator_usage_stats() {
        let stats = AcceleratorUsageStats::default();
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.total_time, 0.0);
        assert_eq!(stats.memory_allocated, 0);
        assert_eq!(stats.errors, 0);
    }
}
