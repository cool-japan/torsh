//! Device abstraction and management system
//!
//! This module provides a comprehensive device abstraction layer for ToRSh that supports
//! multiple compute backends including CPU, CUDA, Metal, and WebGPU. The system is designed
//! to be modular, extensible, and provides both type-safe and runtime device operations.
//!
//! # Architecture
//!
//! The device system is organized into specialized modules:
//!
//! - [`types`] - Core device types and fundamental definitions
//! - [`capabilities`] - Device capability detection and SIMD support
//! - [`sync`] - Device synchronization primitives and coordination
//! - [`core`] - Core device trait definitions and interfaces
//! - [`phantom`] - Phantom device types for compile-time safety
//! - [`typed`] - Strongly typed device programming interfaces
//! - [`implementations`] - Concrete device implementations for different backends
//! - [`management`] - Device discovery and lifecycle management
//! - [`transfer`] - Cross-device memory transfer and scheduling
//! - [`discovery`] - Device enumeration and intelligent selection
//!
//! # Basic Usage
//!
//! ```ignore
//! use torsh_core::device::{DeviceType, Device, DeviceManager};
//!
//! // Basic device creation
//! let cpu_device = CpuDevice::new();
//! println!("Device: {}", cpu_device.name());
//!
//! // Device discovery and management
//! let mut manager = DeviceManager::new();
//! manager.discover_devices()?;
//! let best_device = manager.get_best_device()?;
//!
//! // Memory transfers
//! let transfer = TransferRequest::new(DeviceType::Cpu, DeviceType::Cuda(0), 1024);
//! let handle = transfer_manager.execute_transfer(transfer)?;
//! handle.wait()?;
//! ```
//!
//! # Type-Safe Device Programming
//!
//! ```ignore
//! use torsh_core::device::{PhantomCuda, DeviceHandle, TypedDevice};
//!
//! // Compile-time type safety
//! let cuda_device: DeviceHandle<PhantomCuda<0>> = create_cuda_device()?;
//! assert!(cuda_device.is_cuda());
//! assert!(!cuda_device.is_cpu());
//!
//! // Type-safe operations
//! let result = cuda_device.execute_typed_operation(my_operation)?;
//! ```
//!
//! # Advanced Features
//!
//! The device system provides advanced features including:
//!
//! - **Device Discovery**: Intelligent device selection based on workload characteristics
//! - **Memory Transfer**: Optimized cross-device memory transfers with P2P support
//! - **Synchronization**: Events, barriers, and streams for coordinating operations
//! - **Type Safety**: Compile-time device type checking with phantom types
//! - **Capabilities**: Comprehensive device capability detection and SIMD support
//! - **Management**: Lifecycle management, health monitoring, and resource allocation

// Module declarations
pub mod capabilities;
pub mod core;
pub mod discovery;
pub mod implementations;
pub mod management;
pub mod phantom;
pub mod sync;
pub mod transfer;
pub mod typed;
pub mod types;

// Core re-exports for backward compatibility
pub use self::core::{
    global_device_registry, initialize_global_registry, Device, DeviceContext,
    DeviceFactory as CoreDeviceFactory, DeviceLifecycle, DeviceMemoryInfo, DeviceRegistry,
    DeviceState, RegistryStatistics,
};
pub use self::types::{parse_device_string, DeviceType};

// Capability detection
pub use self::capabilities::{DeviceCapabilities, PciInfo, SimdFeatures, ThermalInfo};

// Synchronization primitives
pub use self::sync::{
    DeviceAsync, DeviceBarrier, DeviceEvent, DeviceMutex, DeviceStream, DeviceSyncManager,
    StreamPriority, SyncStatistics,
};

// Phantom types for compile-time safety
pub use self::phantom::{
    DeviceCompatible, DeviceHandle, DeviceOperation, DeviceRequirements, NoRequirements,
    PhantomCpu, PhantomCuda, PhantomDevice, PhantomDeviceManager, PhantomMetal, PhantomWgpu,
    RequiresCpu, RequiresCuda, RequiresGpu, SameDevice, TransferCompatible,
};

// Typed device programming
pub use self::typed::{
    CudaKernel, DeviceRequirements as TypedDeviceRequirements, GpuKernel, GpuMemoryInfo,
    MetalShader, TypedCpuDevice, TypedCudaDevice, TypedDevice, TypedDeviceBuilder,
    TypedDeviceFactory, TypedDeviceInstance, TypedDeviceOperation, TypedDeviceSelector,
    TypedGpuDevice, TypedMetalDevice, TypedOperationValidator,
};

// Concrete implementations
pub use self::implementations::{CpuDevice, DeviceFactory, SimdLevel};

#[cfg(feature = "cuda")]
pub use self::implementations::CudaDevice;

#[cfg(target_os = "macos")]
pub use self::implementations::MetalDevice;

#[cfg(feature = "wgpu")]
pub use self::implementations::WgpuDevice;

// Device management
pub use self::management::{
    global_device_manager, initialize_global_manager, AllocationStrategy, DeviceHealth,
    DeviceManager, DiscoveryConfig as ManagementDiscoveryConfig, HealthConfig, HealthMonitor,
    ManagerConfig, ManagerStatistics,
};

// Memory transfer
pub use self::transfer::{
    BandwidthConfig, BandwidthManager, P2PManager, TransferConfig, TransferHandle, TransferId,
    TransferManager, TransferMethod, TransferPriority, TransferRequest, TransferResult,
    TransferStatistics, TransferStatus,
};

// Device discovery and selection
pub use self::discovery::{
    CapabilityRequirements, DeviceDiscovery, DeviceOption, DevicePreference, DeviceRecommendation,
    DiscoveredDevice, DiscoveryConfig, DiscoveryStatistics, PerformanceEstimate, PlatformInfo,
    UseCase, WorkloadProfile, WorkloadType,
};

// Utility re-exports
pub use self::capabilities::utils as capability_utils;
pub use self::core::utils as device_utils;
pub use self::discovery::utils as discovery_utils;
pub use self::implementations::utils as implementation_utils;
pub use self::management::utils as management_utils;
pub use self::phantom::utils as phantom_utils;
pub use self::sync::utils as sync_utils;
pub use self::transfer::utils as transfer_utils;
pub use self::typed::utils as typed_utils;
pub use self::types::utils as device_type_utils;

// Legacy type aliases for backward compatibility
pub type DeviceId = String;
pub type DeviceIndex = usize;

/// Prelude for common device operations
///
/// This module provides convenient imports for the most commonly used device types and operations.
///
/// ```ignore
/// use torsh_core::device::prelude::*;
///
/// let device = CpuDevice::new();
/// let manager = DeviceManager::new();
/// let discovery = DeviceDiscovery::new();
/// ```
pub mod prelude {
    // Core types
    pub use super::{Device, DeviceCapabilities, DeviceType};

    // Common implementations
    pub use super::CpuDevice;
    #[cfg(feature = "cuda")]
    pub use super::CudaDevice;
    #[cfg(target_os = "macos")]
    pub use super::MetalDevice;
    #[cfg(feature = "wgpu")]
    pub use super::WgpuDevice;

    // Management and discovery
    pub use super::{DeviceDiscovery, DeviceFactory, DeviceManager};

    // Synchronization
    pub use super::{DeviceBarrier, DeviceEvent, DeviceStream};

    // Transfer
    pub use super::{TransferHandle, TransferManager, TransferRequest};

    // Phantom types
    pub use super::{DeviceHandle, PhantomCpu, PhantomCuda, PhantomMetal, PhantomWgpu};

    // Typed interfaces
    pub use super::{TypedDevice, TypedDeviceFactory};

    // Utility functions
    pub use super::{global_device_manager, parse_device_string};
}

/// Advanced device programming constructs
///
/// This module provides access to advanced device programming features for users who need
/// fine-grained control over device operations.
pub mod advanced {
    // Core advanced interfaces
    pub use super::core::{DeviceContext, DeviceLifecycle, DeviceRegistry};

    // Advanced synchronization
    pub use super::sync::{DeviceAsync, DeviceMutex, DeviceSyncManager};

    // Phantom type programming
    pub use super::phantom::{
        DeviceCompatible, DeviceOperation, PhantomDeviceManager, SameDevice, TransferCompatible,
    };

    // Advanced typed programming
    pub use super::typed::{
        CudaKernel, GpuKernel, MetalShader, TypedDeviceBuilder, TypedDeviceOperation,
        TypedDeviceSelector, TypedOperationValidator,
    };

    // Advanced transfer operations
    pub use super::transfer::{BandwidthManager, P2PManager, TransferConfig};

    // Advanced discovery
    pub use super::discovery::{CapabilityRequirements, DeviceRecommendation, WorkloadProfile};

    // Platform information
    pub use super::capabilities::{PciInfo, SimdFeatures, ThermalInfo};
    pub use super::discovery::PlatformInfo;
}

/// Platform-specific functionality
///
/// This module provides access to platform-specific device features and optimizations.
pub mod platform {
    // CPU platform features
    pub use super::capabilities::SimdFeatures;
    pub use super::implementations::SimdLevel;

    // CUDA platform features
    #[cfg(feature = "cuda")]
    pub mod cuda {
        pub use crate::device::implementations::CudaDevice;
        pub use crate::device::phantom::PhantomCuda;
        pub use crate::device::typed::{CudaKernel, TypedCudaDevice};
    }

    // Metal platform features
    #[cfg(target_os = "macos")]
    pub mod metal {
        pub use crate::device::implementations::MetalDevice;
        pub use crate::device::phantom::PhantomMetal;
        pub use crate::device::typed::{MetalShader, TypedMetalDevice};
    }

    // WebGPU platform features
    #[cfg(feature = "wgpu")]
    pub mod wgpu {
        pub use crate::device::implementations::WgpuDevice;
        pub use crate::device::phantom::PhantomWgpu;
    }
}

/// Convenience functions for common device operations
///
/// This module provides high-level convenience functions that combine multiple device
/// operations for common use cases.
pub mod convenience {
    use super::*;
    use crate::error::Result;

    /// Get the best available device for training workloads
    pub fn get_best_training_device() -> Result<Option<std::sync::Arc<dyn Device>>> {
        discovery_utils::quick_select_for_training()
    }

    /// Get the best available device for inference workloads
    pub fn get_best_inference_device() -> Result<Option<std::sync::Arc<dyn Device>>> {
        discovery_utils::quick_select_for_inference()
    }

    /// Get the best available GPU device
    pub fn get_best_gpu_device() -> Result<Option<std::sync::Arc<dyn Device>>> {
        discovery_utils::get_best_gpu()
    }

    /// Create a CPU device
    pub fn create_cpu_device() -> CpuDevice {
        CpuDevice::new()
    }

    /// Create a CUDA device if available
    #[cfg(feature = "cuda")]
    pub fn create_cuda_device(index: usize) -> Result<CudaDevice> {
        CudaDevice::new(index)
    }

    /// Create a Metal device if available
    #[cfg(target_os = "macos")]
    pub fn create_metal_device(index: usize) -> Result<MetalDevice> {
        MetalDevice::new(index)
    }

    /// Create a WebGPU device if available
    #[cfg(feature = "wgpu")]
    pub fn create_wgpu_device(index: usize) -> Result<WgpuDevice> {
        WgpuDevice::new(index)
    }

    /// Parse device type from string
    pub fn parse_device_type(s: &str) -> Result<DeviceType> {
        parse_device_string(s)
    }

    /// Get device capabilities
    pub fn get_device_capabilities(device_type: DeviceType) -> Result<DeviceCapabilities> {
        DeviceCapabilities::detect(device_type)
    }

    /// Check if a device type is available
    pub fn is_device_available(device_type: DeviceType) -> bool {
        DeviceFactory::is_device_type_available(device_type)
    }

    /// Get all available device types
    pub fn get_available_device_types() -> Vec<DeviceType> {
        DeviceFactory::available_device_types()
    }

    /// Create a device manager with all devices discovered
    pub fn create_device_manager() -> Result<DeviceManager> {
        management_utils::create_manager_with_all_devices()
    }

    /// Create a transfer manager with optimal configuration
    pub fn create_transfer_manager() -> TransferManager {
        transfer_utils::create_optimized_manager()
    }

    /// Create a device discovery engine and scan for devices
    pub fn create_device_discovery() -> Result<DeviceDiscovery> {
        discovery_utils::create_and_scan()
    }

    /// Get a summary of all discovered devices
    pub fn get_device_summary() -> Result<Vec<String>> {
        discovery_utils::create_device_summary()
    }

    /// Check if high-performance devices are available
    pub fn has_high_performance_devices() -> Result<bool> {
        discovery_utils::has_high_performance_devices()
    }
}

/// Error types related to device operations
pub mod error {
    // Re-export device-related error variants from the main error module
    pub use crate::error::TorshError::*;
}

// Global initialization function for the device system
/// Initialize the device system with default configuration
pub fn initialize_device_system() -> Result<(), crate::error::TorshError> {
    // Initialize the global device manager
    let manager_config = management::ManagerConfig::default();
    management::initialize_global_manager(manager_config)?;

    // Initialize the global device registry
    let _registry = core::global_device_registry();

    // Register standard device factories if available
    // Note: In a real implementation, this would register actual factories
    // For now, we use the DeviceFactory from implementations

    Ok(())
}

/// Get system information about available devices
pub fn get_system_info() -> SystemInfo {
    let manager = global_device_manager();
    let stats = manager.statistics();

    let available_types = DeviceFactory::available_device_types();

    SystemInfo {
        total_devices: stats.total_devices,
        available_devices: stats.available_devices,
        device_types: available_types,
        has_cuda: cfg!(feature = "cuda"),
        has_metal: cfg!(target_os = "macos"),
        has_wgpu: cfg!(feature = "wgpu"),
    }
}

/// System information structure
#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub total_devices: usize,
    pub available_devices: usize,
    pub device_types: Vec<DeviceType>,
    pub has_cuda: bool,
    pub has_metal: bool,
    pub has_wgpu: bool,
}

impl std::fmt::Display for SystemInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Device System: {} devices ({} available), CUDA: {}, Metal: {}, WebGPU: {}",
            self.total_devices,
            self.available_devices,
            self.has_cuda,
            self.has_metal,
            self.has_wgpu
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_integration() {
        // Test that all major components can be imported and used together
        let device_type = DeviceType::Cpu;
        assert_eq!(device_type.to_string(), "cpu");

        let cpu_device = CpuDevice::new();
        assert_eq!(cpu_device.device_type(), DeviceType::Cpu);

        let manager = DeviceManager::new();
        assert_eq!(manager.device_count(), 0);

        let discovery = DeviceDiscovery::new();
        let _count = discovery.scan_devices().unwrap();
    }

    #[test]
    fn test_prelude_imports() {
        use super::prelude::*;

        let device_type = DeviceType::Cpu;
        let cpu_device = CpuDevice::new();
        let manager = DeviceManager::new();
        let discovery = DeviceDiscovery::new();

        assert_eq!(device_type, cpu_device.device_type());
        assert_eq!(manager.device_count(), 0);
        assert!(
            !discovery.get_discovered_devices().is_empty() || discovery.scan_devices().unwrap() > 0
        );
    }

    #[test]
    fn test_convenience_functions() {
        let cpu_device = convenience::create_cpu_device();
        assert_eq!(cpu_device.device_type(), DeviceType::Cpu);

        let device_type = convenience::parse_device_type("cpu").unwrap();
        assert_eq!(device_type, DeviceType::Cpu);

        let available_types = convenience::get_available_device_types();
        assert!(available_types.contains(&DeviceType::Cpu));

        assert!(convenience::is_device_available(DeviceType::Cpu));
    }

    #[test]
    fn test_system_info() {
        let info = get_system_info();

        // Should have at least some devices after global manager initialization
        // Note: total_devices is unsigned, so check it's reasonable instead
        assert!(!info.device_types.is_empty());

        // Check feature flags
        assert_eq!(info.has_cuda, cfg!(feature = "cuda"));
        assert_eq!(info.has_metal, cfg!(target_os = "macos"));
        assert_eq!(info.has_wgpu, cfg!(feature = "wgpu"));
    }

    #[test]
    fn test_device_system_initialization() {
        let result = initialize_device_system();
        assert!(result.is_ok());

        // After initialization, global manager should be available
        let global_manager = global_device_manager();
        let _stats = global_manager.statistics();
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that old-style imports still work
        let device_type = DeviceType::Cpu;
        let capabilities = DeviceCapabilities::detect(device_type).unwrap();
        let cpu_device = CpuDevice::new();

        assert_eq!(device_type, DeviceType::Cpu);
        assert!(capabilities.total_memory() > 0);
        assert_eq!(cpu_device.name(), "CPU");

        // Test parsing
        let parsed = parse_device_string("cpu").unwrap();
        assert_eq!(parsed, DeviceType::Cpu);
    }

    #[test]
    fn test_phantom_types() {
        use super::phantom::*;

        assert_eq!(PhantomCpu::device_type(), DeviceType::Cpu);
        assert_eq!(PhantomCuda::<0>::device_type(), DeviceType::Cuda(0));

        assert!(PhantomCpu::is_compatible::<PhantomCpu>());
        assert!(!PhantomCpu::is_compatible::<PhantomCuda<0>>());
    }

    #[test]
    fn test_advanced_features() {
        use super::advanced::*;

        let lifecycle = DeviceLifecycle::new();
        assert!(!lifecycle.is_ready()); // Starts uninitialized

        let sync_manager = DeviceSyncManager::new();
        let stats = sync_manager.statistics();
        assert_eq!(stats.total_streams, 0);
    }
}
