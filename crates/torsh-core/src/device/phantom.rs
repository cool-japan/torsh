//! Phantom device types for compile-time safety
//!
//! This module provides phantom types and zero-cost abstractions that enable
//! compile-time device type safety and validation without runtime overhead.

use crate::device::{Device, DeviceType};
use std::marker::PhantomData;

/// Phantom device marker trait for compile-time device type information
///
/// This trait is used to mark types with specific device information at compile time,
/// enabling type-safe device operations without runtime overhead.
pub trait PhantomDevice: 'static + std::fmt::Debug + Send + Sync {
    /// The device type this phantom represents
    const DEVICE_TYPE: DeviceType;

    /// Get the device type (compile-time constant)
    fn device_type() -> DeviceType {
        Self::DEVICE_TYPE
    }

    /// Check if this phantom device is compatible with another
    fn is_compatible<Other: PhantomDevice>() -> bool {
        Self::DEVICE_TYPE == Other::DEVICE_TYPE
    }

    /// Get the device name as a compile-time string
    fn device_name() -> &'static str;

    /// Check if this device requires GPU features
    fn requires_gpu() -> bool {
        !matches!(Self::DEVICE_TYPE, DeviceType::Cpu)
    }

    /// Check if this device supports peer-to-peer operations
    fn supports_p2p() -> bool {
        matches!(Self::DEVICE_TYPE, DeviceType::Cuda(_))
    }
}

/// CPU phantom device marker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhantomCpu;

impl PhantomDevice for PhantomCpu {
    const DEVICE_TYPE: DeviceType = DeviceType::Cpu;

    fn device_name() -> &'static str {
        "CPU"
    }
}

/// CUDA phantom device marker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhantomCuda<const INDEX: usize>;

impl<const INDEX: usize> PhantomDevice for PhantomCuda<INDEX> {
    const DEVICE_TYPE: DeviceType = DeviceType::Cuda(INDEX);

    fn device_name() -> &'static str {
        "CUDA"
    }
}

/// Metal phantom device marker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhantomMetal<const INDEX: usize>;

impl<const INDEX: usize> PhantomDevice for PhantomMetal<INDEX> {
    const DEVICE_TYPE: DeviceType = DeviceType::Metal(INDEX);

    fn device_name() -> &'static str {
        "Metal"
    }
}

/// WebGPU phantom device marker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhantomWgpu<const INDEX: usize>;

impl<const INDEX: usize> PhantomDevice for PhantomWgpu<INDEX> {
    const DEVICE_TYPE: DeviceType = DeviceType::Wgpu(INDEX);

    fn device_name() -> &'static str {
        "WebGPU"
    }
}

/// Type-safe device handle with phantom device information
///
/// This wrapper provides compile-time device type safety for device operations.
/// The phantom type parameter carries device information at the type level.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::device::{DeviceHandle, PhantomCpu, PhantomCuda};
///
/// // CPU device handle
/// let cpu_handle = DeviceHandle::<PhantomCpu>::new(cpu_device);
///
/// // CUDA device handle with index 0
/// let cuda_handle = DeviceHandle::<PhantomCuda<0>>::new(cuda_device);
///
/// // Compile-time device type checking
/// assert!(cpu_handle.is_cpu());
/// assert!(cuda_handle.is_gpu());
/// ```
#[derive(Debug)]
pub struct DeviceHandle<P: PhantomDevice> {
    device: Box<dyn Device>,
    _phantom: PhantomData<P>,
}

impl<P: PhantomDevice> DeviceHandle<P> {
    /// Create a new device handle with phantom type information
    pub fn new(device: Box<dyn Device>) -> Result<Self, crate::error::TorshError> {
        if device.device_type() != P::DEVICE_TYPE {
            return Err(crate::error::TorshError::InvalidArgument(format!(
                "Device type mismatch: expected {:?}, got {:?}",
                P::DEVICE_TYPE,
                device.device_type()
            )));
        }

        Ok(Self {
            device,
            _phantom: PhantomData,
        })
    }

    /// Create an unchecked device handle (unsafe)
    ///
    /// # Safety
    /// The caller must ensure that the device type matches the phantom type.
    pub unsafe fn new_unchecked(device: Box<dyn Device>) -> Self {
        Self {
            device,
            _phantom: PhantomData,
        }
    }

    /// Get the underlying device
    pub fn device(&self) -> &dyn Device {
        self.device.as_ref()
    }

    /// Get the underlying device mutably
    pub fn device_mut(&mut self) -> &mut dyn Device {
        self.device.as_mut()
    }

    /// Get the phantom device type
    pub fn phantom_device_type() -> DeviceType {
        P::DEVICE_TYPE
    }

    /// Check if this is a CPU device (compile-time)
    pub const fn is_cpu() -> bool {
        matches!(P::DEVICE_TYPE, DeviceType::Cpu)
    }

    /// Check if this is a GPU device (compile-time)
    pub const fn is_gpu() -> bool {
        !matches!(P::DEVICE_TYPE, DeviceType::Cpu)
    }

    /// Check if this is a CUDA device (compile-time)
    pub const fn is_cuda() -> bool {
        matches!(P::DEVICE_TYPE, DeviceType::Cuda(_))
    }

    /// Check if this is a Metal device (compile-time)
    pub const fn is_metal() -> bool {
        matches!(P::DEVICE_TYPE, DeviceType::Metal(_))
    }

    /// Check if this is a WebGPU device (compile-time)
    pub const fn is_wgpu() -> bool {
        matches!(P::DEVICE_TYPE, DeviceType::Wgpu(_))
    }

    /// Convert to a different phantom device type (with runtime check)
    pub fn cast<Q: PhantomDevice>(
        self,
    ) -> Result<DeviceHandle<Q>, (Self, crate::error::TorshError)> {
        if self.device.device_type() != Q::DEVICE_TYPE {
            let error = crate::error::TorshError::InvalidArgument(format!(
                "Cannot cast device from {:?} to {:?}",
                P::DEVICE_TYPE,
                Q::DEVICE_TYPE
            ));
            return Err((self, error));
        }

        Ok(DeviceHandle {
            device: self.device,
            _phantom: PhantomData,
        })
    }

    /// Convert to a different phantom device type (unsafe, no runtime check)
    ///
    /// # Safety
    /// The caller must ensure that the device type matches the target phantom type.
    pub unsafe fn cast_unchecked<Q: PhantomDevice>(self) -> DeviceHandle<Q> {
        DeviceHandle {
            device: self.device,
            _phantom: PhantomData,
        }
    }

    /// Extract the underlying device, consuming the handle
    pub fn into_device(self) -> Box<dyn Device> {
        self.device
    }
}

impl<P: PhantomDevice> Clone for DeviceHandle<P> {
    fn clone(&self) -> Self {
        let cloned_device = self.device.clone_device().expect("Failed to clone device");

        Self {
            device: cloned_device,
            _phantom: PhantomData,
        }
    }
}

/// Compile-time device compatibility checker
///
/// This trait provides compile-time guarantees about device compatibility
/// for operations that require specific device types or combinations.
pub trait DeviceCompatible<Other> {
    /// Check if the devices are compatible at compile time
    const COMPATIBLE: bool;

    /// Get compatibility information
    fn compatibility_info() -> &'static str;
}

impl<P: PhantomDevice> DeviceCompatible<P> for P {
    const COMPATIBLE: bool = true;

    fn compatibility_info() -> &'static str {
        "Same device type - always compatible"
    }
}

// DeviceCompatible is already implemented generically above

/// Type-level operation constraints
///
/// This trait allows operations to specify their device requirements at the type level,
/// enabling compile-time validation of device compatibility.
pub trait DeviceOperation<P: PhantomDevice> {
    /// The result type of this operation
    type Output;

    /// Device requirements for this operation
    type Requirements: DeviceRequirements;

    /// Execute the operation on the given device
    fn execute(device: &DeviceHandle<P>) -> Result<Self::Output, crate::error::TorshError>;

    /// Check if the operation is supported on this device type (compile-time)
    const SUPPORTED: bool = Self::Requirements::SATISFIED_BY_DEVICE;
}

/// Device requirements trait for compile-time requirement checking
pub trait DeviceRequirements {
    /// Whether this requirement is satisfied by the device
    const SATISFIED_BY_DEVICE: bool;

    /// Description of the requirements
    fn description() -> &'static str;
}

/// Requirement for GPU device
#[derive(Debug, Clone, Copy)]
pub struct RequiresGpu;

impl DeviceRequirements for RequiresGpu {
    const SATISFIED_BY_DEVICE: bool = false; // Will be specialized for GPU types

    fn description() -> &'static str {
        "Requires GPU device"
    }
}

/// Requirement for CPU device
#[derive(Debug, Clone, Copy)]
pub struct RequiresCpu;

impl DeviceRequirements for RequiresCpu {
    const SATISFIED_BY_DEVICE: bool = false; // Will be specialized for CPU type

    fn description() -> &'static str {
        "Requires CPU device"
    }
}

/// Requirement for CUDA device
#[derive(Debug, Clone, Copy)]
pub struct RequiresCuda;

impl DeviceRequirements for RequiresCuda {
    const SATISFIED_BY_DEVICE: bool = false; // Will be specialized for CUDA types

    fn description() -> &'static str {
        "Requires CUDA device"
    }
}

/// No specific device requirements
#[derive(Debug, Clone, Copy)]
pub struct NoRequirements;

impl DeviceRequirements for NoRequirements {
    const SATISFIED_BY_DEVICE: bool = true;

    fn description() -> &'static str {
        "No specific device requirements"
    }
}

/// Device constraint that requires two devices to be the same type
#[derive(Debug)]
pub struct SameDevice<P1: PhantomDevice, P2: PhantomDevice> {
    _phantom: PhantomData<(P1, P2)>,
}

impl<P1: PhantomDevice, P2: PhantomDevice> SameDevice<P1, P2> {
    /// Check if the constraint is satisfied
    pub fn is_satisfied() -> bool {
        match (P1::DEVICE_TYPE, P2::DEVICE_TYPE) {
            (DeviceType::Cpu, DeviceType::Cpu) => true,
            (DeviceType::Cuda(a), DeviceType::Cuda(b)) => a == b,
            (DeviceType::Metal(a), DeviceType::Metal(b)) => a == b,
            (DeviceType::Wgpu(a), DeviceType::Wgpu(b)) => a == b,
            _ => false,
        }
    }
}

/// Device constraint that allows transfer between compatible devices
#[derive(Debug)]
pub struct TransferCompatible<P1: PhantomDevice, P2: PhantomDevice> {
    _phantom: PhantomData<(P1, P2)>,
}

impl<P1: PhantomDevice, P2: PhantomDevice> TransferCompatible<P1, P2> {
    /// Check if transfer is supported (compile-time)
    pub const SUPPORTED: bool = true; // All devices support some form of transfer

    /// Get the estimated transfer cost
    pub fn transfer_cost() -> u32 {
        match (P1::DEVICE_TYPE, P2::DEVICE_TYPE) {
            (DeviceType::Cpu, DeviceType::Cpu) => 0,
            (DeviceType::Cuda(a), DeviceType::Cuda(b)) if a == b => 0,
            (DeviceType::Metal(a), DeviceType::Metal(b)) if a == b => 0,
            (DeviceType::Wgpu(a), DeviceType::Wgpu(b)) if a == b => 0,
            (DeviceType::Cpu, DeviceType::Cuda(_)) => 100,
            (DeviceType::Cuda(_), DeviceType::Cpu) => 100,
            (DeviceType::Cpu, DeviceType::Metal(_)) => 80,
            (DeviceType::Metal(_), DeviceType::Cpu) => 80,
            _ => 200, // Cross-GPU transfers
        }
    }
}

/// Type-safe device manager that maintains phantom type information
#[derive(Debug)]
pub struct PhantomDeviceManager<P: PhantomDevice> {
    handles: Vec<DeviceHandle<P>>,
    _phantom: PhantomData<P>,
}

impl<P: PhantomDevice> PhantomDeviceManager<P> {
    /// Create a new phantom device manager
    pub fn new() -> Self {
        Self {
            handles: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Add a device handle
    pub fn add_device(&mut self, handle: DeviceHandle<P>) {
        self.handles.push(handle);
    }

    /// Get the number of managed devices
    pub fn device_count(&self) -> usize {
        self.handles.len()
    }

    /// Get a device handle by index
    pub fn get_device(&self, index: usize) -> Option<&DeviceHandle<P>> {
        self.handles.get(index)
    }

    /// Get a mutable device handle by index
    pub fn get_device_mut(&mut self, index: usize) -> Option<&mut DeviceHandle<P>> {
        self.handles.get_mut(index)
    }

    /// Remove a device handle by index
    pub fn remove_device(&mut self, index: usize) -> Option<DeviceHandle<P>> {
        if index < self.handles.len() {
            Some(self.handles.remove(index))
        } else {
            None
        }
    }

    /// Get all device handles
    pub fn devices(&self) -> &[DeviceHandle<P>] {
        &self.handles
    }

    /// Clear all devices
    pub fn clear(&mut self) {
        self.handles.clear();
    }

    /// Execute an operation on all devices
    pub fn execute_on_all<Op>(
        &self,
        _operation: Op,
    ) -> Vec<Result<Op::Output, crate::error::TorshError>>
    where
        Op: DeviceOperation<P> + Clone,
    {
        self.handles
            .iter()
            .map(|handle| Op::execute(handle))
            .collect()
    }
}

impl<P: PhantomDevice> Default for PhantomDeviceManager<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for phantom device operations
pub mod utils {
    use super::*;

    /// Create a type-safe device handle from a runtime device
    pub fn create_phantom_handle<P: PhantomDevice>(
        device: Box<dyn Device>,
    ) -> Result<DeviceHandle<P>, crate::error::TorshError> {
        DeviceHandle::<P>::new(device)
    }

    /// Check device compatibility at runtime with phantom type information
    pub fn check_phantom_compatibility<P1: PhantomDevice, P2: PhantomDevice>() -> bool {
        P1::DEVICE_TYPE == P2::DEVICE_TYPE
    }

    /// Get the transfer cost between two phantom device types
    pub fn phantom_transfer_cost<P1: PhantomDevice, P2: PhantomDevice>() -> u32 {
        TransferCompatible::<P1, P2>::transfer_cost()
    }

    /// Create a device manager for a specific phantom device type
    pub fn create_phantom_manager<P: PhantomDevice>() -> PhantomDeviceManager<P> {
        PhantomDeviceManager::new()
    }

    /// Verify that an operation is supported on a phantom device type
    pub fn verify_operation_support<P: PhantomDevice, Op: DeviceOperation<P>>() -> bool {
        Op::SUPPORTED
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::core::Device;
    use std::any::Any;

    // Mock device for testing
    #[derive(Debug)]
    struct MockDevice {
        device_type: DeviceType,
    }

    impl MockDevice {
        fn new(device_type: DeviceType) -> Self {
            Self { device_type }
        }
    }

    impl Device for MockDevice {
        fn device_type(&self) -> DeviceType {
            self.device_type
        }

        fn name(&self) -> &str {
            "Mock Device"
        }

        fn is_available(&self) -> Result<bool, crate::error::TorshError> {
            Ok(true)
        }

        fn capabilities(
            &self,
        ) -> Result<crate::device::DeviceCapabilities, crate::error::TorshError> {
            crate::device::DeviceCapabilities::detect(self.device_type)
        }

        fn synchronize(&self) -> Result<(), crate::error::TorshError> {
            Ok(())
        }

        fn reset(&self) -> Result<(), crate::error::TorshError> {
            Ok(())
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }

        fn clone_device(&self) -> Result<Box<dyn Device>, crate::error::TorshError> {
            Ok(Box::new(MockDevice::new(self.device_type)))
        }
    }

    #[test]
    fn test_phantom_device_markers() {
        assert_eq!(PhantomCpu::device_type(), DeviceType::Cpu);
        assert_eq!(PhantomCuda::<0>::device_type(), DeviceType::Cuda(0));
        assert_eq!(PhantomMetal::<1>::device_type(), DeviceType::Metal(1));
        assert_eq!(PhantomWgpu::<2>::device_type(), DeviceType::Wgpu(2));

        assert_eq!(PhantomCpu::device_name(), "CPU");
        assert_eq!(PhantomCuda::<0>::device_name(), "CUDA");
        assert_eq!(PhantomMetal::<0>::device_name(), "Metal");
        assert_eq!(PhantomWgpu::<0>::device_name(), "WebGPU");
    }

    #[test]
    fn test_phantom_device_properties() {
        assert!(!PhantomCpu::requires_gpu());
        assert!(PhantomCuda::<0>::requires_gpu());
        assert!(PhantomMetal::<0>::requires_gpu());
        assert!(PhantomWgpu::<0>::requires_gpu());

        assert!(!PhantomCpu::supports_p2p());
        assert!(PhantomCuda::<0>::supports_p2p());
        assert!(!PhantomMetal::<0>::supports_p2p());
        assert!(!PhantomWgpu::<0>::supports_p2p());
    }

    #[test]
    fn test_device_handle() {
        let mock_device = Box::new(MockDevice::new(DeviceType::Cpu));
        let _handle = DeviceHandle::<PhantomCpu>::new(mock_device).unwrap();

        assert_eq!(
            DeviceHandle::<PhantomCpu>::phantom_device_type(),
            DeviceType::Cpu
        );
        assert!(DeviceHandle::<PhantomCpu>::is_cpu());
        assert!(!DeviceHandle::<PhantomCpu>::is_gpu());
        assert!(!DeviceHandle::<PhantomCpu>::is_cuda());
    }

    #[test]
    fn test_device_handle_type_mismatch() {
        let mock_device = Box::new(MockDevice::new(DeviceType::Cuda(0)));
        let result = DeviceHandle::<PhantomCpu>::new(mock_device);
        assert!(result.is_err());
    }

    #[test]
    fn test_device_compatibility() {
        assert!(PhantomCpu::is_compatible::<PhantomCpu>());
        assert!(!PhantomCpu::is_compatible::<PhantomCuda<0>>());
        assert!(PhantomCuda::<0>::is_compatible::<PhantomCuda<0>>());
        assert!(!PhantomCuda::<0>::is_compatible::<PhantomCuda<1>>());
    }

    #[test]
    fn test_phantom_device_manager() {
        let mut manager = PhantomDeviceManager::<PhantomCpu>::new();
        assert_eq!(manager.device_count(), 0);

        let mock_device = Box::new(MockDevice::new(DeviceType::Cpu));
        let handle = DeviceHandle::<PhantomCpu>::new(mock_device).unwrap();
        manager.add_device(handle);

        assert_eq!(manager.device_count(), 1);
        assert!(manager.get_device(0).is_some());
        assert!(manager.get_device(1).is_none());

        let removed = manager.remove_device(0);
        assert!(removed.is_some());
        assert_eq!(manager.device_count(), 0);
    }

    #[test]
    fn test_transfer_cost_constants() {
        assert_eq!(
            TransferCompatible::<PhantomCpu, PhantomCpu>::transfer_cost(),
            0
        );
        assert_eq!(
            TransferCompatible::<PhantomCpu, PhantomCuda<0>>::transfer_cost(),
            100
        );
        assert_eq!(
            TransferCompatible::<PhantomCpu, PhantomMetal<0>>::transfer_cost(),
            80
        );
    }

    #[test]
    fn test_utils_functions() {
        assert!(utils::check_phantom_compatibility::<PhantomCpu, PhantomCpu>());
        assert!(!utils::check_phantom_compatibility::<
            PhantomCpu,
            PhantomCuda<0>,
        >());

        let cost = utils::phantom_transfer_cost::<PhantomCpu, PhantomCuda<0>>();
        assert_eq!(cost, 100);

        let manager = utils::create_phantom_manager::<PhantomCpu>();
        assert_eq!(manager.device_count(), 0);
    }
}
