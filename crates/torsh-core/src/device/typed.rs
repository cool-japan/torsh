//! Strongly typed device programming interfaces
//!
//! This module provides advanced type-safe device programming constructs that
//! leverage Rust's type system to ensure correctness and safety of device operations.

use crate::device::phantom::{
    DeviceHandle, PhantomCpu, PhantomCuda, PhantomDevice, PhantomMetal, PhantomWgpu,
};
use crate::device::{Device, DeviceCapabilities, DeviceType};
use crate::error::Result;
use std::marker::PhantomData;

/// Strongly typed device interface that enforces type safety at compile time
///
/// This trait extends the basic Device interface with strong typing guarantees,
/// ensuring that operations are only performed on compatible device types.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::device::{TypedDevice, TypedCpuDevice};
///
/// fn process_on_cpu<D: TypedDevice<DeviceType = CpuDeviceType>>(device: &D) -> Result<()> {
///     // This function can only be called with CPU devices
///     device.execute_cpu_operation()?;
///     Ok(())
/// }
/// ```
pub trait TypedDevice: Device {
    /// The phantom device type this device represents
    type PhantomType: PhantomDevice;

    /// Get the phantom device type
    fn phantom_type() -> Self::PhantomType;

    /// Execute a typed operation on this device
    fn execute_typed_operation<Op>(&self, operation: Op) -> Result<Op::Output>
    where
        Op: TypedDeviceOperation<Self::PhantomType>;

    /// Check if this device supports a typed operation at compile time
    fn supports_typed_operation<Op>() -> bool
    where
        Op: TypedDeviceOperation<Self::PhantomType>,
    {
        Op::is_supported()
    }

    /// Convert to a device handle with phantom type information
    fn to_typed_handle(self) -> Result<DeviceHandle<Self::PhantomType>>
    where
        Self: Sized,
    {
        DeviceHandle::new(Box::new(self))
    }
}

/// Typed operation that can be executed on specific device types
///
/// This trait allows operations to specify their type requirements and provides
/// compile-time guarantees about device compatibility.
pub trait TypedDeviceOperation<P: PhantomDevice> {
    /// The output type of this operation
    type Output;

    /// Execute the operation
    fn execute(&self, device: &dyn Device) -> Result<Self::Output>;

    /// Check if this operation is supported on the device type (compile-time)
    fn is_supported() -> bool {
        true // Default implementation - operations can override for specific requirements
    }

    /// Get operation metadata
    fn operation_name() -> &'static str;

    /// Get device requirements
    fn device_requirements() -> DeviceRequirements;
}

/// Device requirements specification
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceRequirements {
    /// Minimum memory required in bytes
    pub min_memory: Option<u64>,
    /// Requires GPU
    pub requires_gpu: bool,
    /// Requires specific compute capability
    pub min_compute_capability: Option<(u32, u32)>,
    /// Required features
    pub required_features: Vec<String>,
}

impl DeviceRequirements {
    /// Create basic requirements
    pub fn basic() -> Self {
        Self {
            min_memory: None,
            requires_gpu: false,
            min_compute_capability: None,
            required_features: Vec::new(),
        }
    }

    /// Require GPU device
    pub fn gpu() -> Self {
        Self {
            min_memory: None,
            requires_gpu: true,
            min_compute_capability: None,
            required_features: Vec::new(),
        }
    }

    /// Require minimum memory
    pub fn with_memory(mut self, min_memory: u64) -> Self {
        self.min_memory = Some(min_memory);
        self
    }

    /// Require specific features
    pub fn with_features(mut self, features: Vec<String>) -> Self {
        self.required_features = features;
        self
    }

    /// Check if requirements are satisfied by device capabilities
    pub fn satisfied_by(&self, capabilities: &DeviceCapabilities) -> bool {
        if self.requires_gpu && capabilities.device_type().is_cpu() {
            return false;
        }

        if let Some(min_mem) = self.min_memory {
            if capabilities.total_memory() < min_mem {
                return false;
            }
        }

        for feature in &self.required_features {
            if !capabilities.supports_feature(feature) {
                return false;
            }
        }

        true
    }
}

/// CPU-specific typed device
pub trait TypedCpuDevice: TypedDevice<PhantomType = PhantomCpu> {
    /// Execute CPU-specific operation
    fn execute_cpu_operation<F, T>(&self, operation: F) -> Result<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static;

    /// Get CPU core count
    fn cpu_core_count(&self) -> Result<u32>;

    /// Check SIMD support
    fn supports_simd(&self) -> Result<bool>;
}

/// GPU-specific typed device trait
pub trait TypedGpuDevice<P: PhantomDevice>: TypedDevice<PhantomType = P>
where
    P: PhantomDevice,
{
    /// Execute GPU kernel
    fn execute_kernel<K>(&self, kernel: K) -> Result<K::Output>
    where
        K: GpuKernel<P>;

    /// Get GPU memory info
    fn gpu_memory_info(&self) -> Result<GpuMemoryInfo>;

    /// Synchronize GPU operations
    fn gpu_synchronize(&self) -> Result<()>;
}

/// CUDA-specific typed device
pub trait TypedCudaDevice<const INDEX: usize>: TypedGpuDevice<PhantomCuda<INDEX>> {
    /// Launch CUDA kernel
    fn launch_cuda_kernel<K>(
        &self,
        kernel: K,
        grid_size: (u32, u32, u32),
        block_size: (u32, u32, u32),
    ) -> Result<K::Output>
    where
        K: CudaKernel;

    /// Get CUDA compute capability
    fn cuda_compute_capability(&self) -> Result<(u32, u32)>;

    /// Check if peer access is available to another CUDA device
    fn can_access_peer(&self, peer_index: usize) -> Result<bool>;
}

/// Metal-specific typed device
pub trait TypedMetalDevice<const INDEX: usize>: TypedGpuDevice<PhantomMetal<INDEX>> {
    /// Execute Metal compute shader
    fn execute_metal_shader<S>(&self, shader: S) -> Result<S::Output>
    where
        S: MetalShader;

    /// Get Metal device registry
    fn metal_device_registry(&self) -> Result<MetalDeviceRegistry>;
}

/// GPU memory information
#[derive(Debug, Clone)]
pub struct GpuMemoryInfo {
    pub total: u64,
    pub free: u64,
    pub used: u64,
}

impl GpuMemoryInfo {
    pub fn utilization_percent(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.used as f64 / self.total as f64) * 100.0
        }
    }
}

/// GPU kernel trait
pub trait GpuKernel<P: PhantomDevice> {
    type Output;

    fn execute(&self, device: &dyn Device) -> Result<Self::Output>;
    fn kernel_name() -> &'static str;
}

/// CUDA kernel trait
pub trait CudaKernel {
    type Output;

    fn execute(&self, device: &dyn Device) -> Result<Self::Output>;
    fn kernel_name() -> &'static str;
    fn required_compute_capability() -> Option<(u32, u32)> {
        None
    }
}

/// Metal shader trait
pub trait MetalShader {
    type Output;

    fn execute(&self, device: &dyn Device) -> Result<Self::Output>;
    fn shader_name() -> &'static str;
}

/// Metal device registry placeholder
#[derive(Debug, Clone)]
pub struct MetalDeviceRegistry {
    // Placeholder for Metal-specific registry information
}

/// Typed device builder for creating strongly typed device instances
#[derive(Debug)]
pub struct TypedDeviceBuilder<P: PhantomDevice> {
    device_type: DeviceType,
    requirements: DeviceRequirements,
    _phantom: PhantomData<P>,
}

impl<P: PhantomDevice> TypedDeviceBuilder<P> {
    /// Create a new typed device builder
    pub fn new() -> Self {
        Self {
            device_type: P::DEVICE_TYPE,
            requirements: DeviceRequirements::basic(),
            _phantom: PhantomData,
        }
    }

    /// Set memory requirements
    pub fn with_memory(mut self, min_memory: u64) -> Self {
        self.requirements = self.requirements.with_memory(min_memory);
        self
    }

    /// Set feature requirements
    pub fn with_features(mut self, features: Vec<String>) -> Self {
        self.requirements = self.requirements.with_features(features);
        self
    }

    /// Build the typed device
    pub fn build(self) -> Result<TypedDeviceInstance<P>> {
        // In a real implementation, this would create the actual device
        // For now, we'll create a mock instance
        TypedDeviceInstance::new(self.device_type, self.requirements)
    }
}

impl<P: PhantomDevice> Default for TypedDeviceBuilder<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Strongly typed device instance
#[derive(Debug)]
pub struct TypedDeviceInstance<P: PhantomDevice> {
    device_type: DeviceType,
    requirements: DeviceRequirements,
    capabilities: Option<DeviceCapabilities>,
    _phantom: PhantomData<P>,
}

impl<P: PhantomDevice> TypedDeviceInstance<P> {
    /// Create a new typed device instance
    pub fn new(device_type: DeviceType, requirements: DeviceRequirements) -> Result<Self> {
        if device_type != P::DEVICE_TYPE {
            return Err(crate::error::TorshError::InvalidArgument(format!(
                "Device type mismatch: expected {:?}, got {:?}",
                P::DEVICE_TYPE,
                device_type
            )));
        }

        Ok(Self {
            device_type,
            requirements,
            capabilities: None,
            _phantom: PhantomData,
        })
    }

    /// Get the device requirements
    pub fn requirements(&self) -> &DeviceRequirements {
        &self.requirements
    }

    /// Check if this instance satisfies the requirements
    pub fn validate_requirements(&mut self) -> Result<bool> {
        let caps = self.capabilities()?;
        Ok(self.requirements.satisfied_by(&caps))
    }

    /// Get the phantom device type
    pub fn phantom_device_type() -> DeviceType {
        P::DEVICE_TYPE
    }
}

impl<P: PhantomDevice> Device for TypedDeviceInstance<P> {
    fn device_type(&self) -> DeviceType {
        self.device_type
    }

    fn name(&self) -> &str {
        P::device_name()
    }

    fn is_available(&self) -> Result<bool> {
        Ok(true) // Mock implementation
    }

    fn capabilities(&self) -> Result<DeviceCapabilities> {
        DeviceCapabilities::detect(self.device_type)
    }

    fn synchronize(&self) -> Result<()> {
        Ok(()) // Mock implementation
    }

    fn reset(&self) -> Result<()> {
        Ok(()) // Mock implementation
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn clone_device(&self) -> Result<Box<dyn Device>> {
        Ok(Box::new(TypedDeviceInstance {
            device_type: self.device_type,
            requirements: self.requirements.clone(),
            capabilities: self.capabilities.clone(),
            _phantom: PhantomData::<P>,
        }))
    }
}

impl<P: PhantomDevice> TypedDevice for TypedDeviceInstance<P> {
    type PhantomType = P;

    fn phantom_type() -> Self::PhantomType {
        // This is a bit tricky - we need to construct a PhantomDevice instance
        // For now, we'll use a placeholder approach
        unsafe { std::mem::zeroed() } // This is unsafe but works for phantom types
    }

    fn execute_typed_operation<Op>(&self, operation: Op) -> Result<Op::Output>
    where
        Op: TypedDeviceOperation<Self::PhantomType>,
    {
        if !Op::is_supported() {
            return Err(crate::error::TorshError::General(
                crate::error::GeneralError::UnsupportedOperation {
                    op: Op::operation_name().to_string(),
                    dtype: self.name().to_string(),
                },
            ));
        }

        let caps = self.capabilities()?;
        if !Op::device_requirements().satisfied_by(&caps) {
            return Err(crate::error::TorshError::General(
                crate::error::GeneralError::RuntimeError(
                    "Device does not meet operation requirements".to_string(),
                ),
            ));
        }

        operation.execute(self)
    }
}

/// Typed device factory for creating strongly typed devices
#[derive(Debug)]
pub struct TypedDeviceFactory;

impl TypedDeviceFactory {
    /// Create a CPU device
    pub fn create_cpu() -> Result<TypedDeviceInstance<PhantomCpu>> {
        TypedDeviceBuilder::<PhantomCpu>::new().build()
    }

    /// Create a CUDA device
    pub fn create_cuda<const INDEX: usize>() -> Result<TypedDeviceInstance<PhantomCuda<INDEX>>> {
        TypedDeviceBuilder::<PhantomCuda<INDEX>>::new()
            .with_features(vec!["cuda".to_string()])
            .build()
    }

    /// Create a Metal device
    pub fn create_metal<const INDEX: usize>() -> Result<TypedDeviceInstance<PhantomMetal<INDEX>>> {
        TypedDeviceBuilder::<PhantomMetal<INDEX>>::new()
            .with_features(vec!["metal".to_string()])
            .build()
    }

    /// Create a WebGPU device
    pub fn create_wgpu<const INDEX: usize>() -> Result<TypedDeviceInstance<PhantomWgpu<INDEX>>> {
        TypedDeviceBuilder::<PhantomWgpu<INDEX>>::new()
            .with_features(vec!["wgpu".to_string()])
            .build()
    }

    /// Create a device from a device type with runtime checking
    pub fn create_from_type(device_type: DeviceType) -> Result<Box<dyn Device>> {
        match device_type {
            DeviceType::Cpu => Ok(Box::new(Self::create_cpu()?)),
            DeviceType::Cuda(0) => Ok(Box::new(Self::create_cuda::<0>()?)),
            DeviceType::Metal(0) => Ok(Box::new(Self::create_metal::<0>()?)),
            DeviceType::Wgpu(0) => Ok(Box::new(Self::create_wgpu::<0>()?)),
            _ => Err(crate::error::TorshError::General(
                crate::error::GeneralError::UnsupportedOperation {
                    op: format!("Device type {:?}", device_type),
                    dtype: "N/A".to_string(),
                },
            )),
        }
    }
}

/// Type-level device selection based on requirements
pub struct TypedDeviceSelector<P: PhantomDevice> {
    requirements: DeviceRequirements,
    _phantom: PhantomData<P>,
}

impl<P: PhantomDevice> TypedDeviceSelector<P> {
    /// Create a new device selector
    pub fn new(requirements: DeviceRequirements) -> Self {
        Self {
            requirements,
            _phantom: PhantomData,
        }
    }

    /// Select the best device that meets the requirements
    pub fn select_best_device<'a>(
        &self,
        candidates: &[&'a dyn Device],
    ) -> Result<Option<&'a dyn Device>> {
        let mut best_device = None;
        let mut best_score = 0;

        for &device in candidates {
            if device.device_type() != P::DEVICE_TYPE {
                continue;
            }

            let caps = device.capabilities()?;
            if !self.requirements.satisfied_by(&caps) {
                continue;
            }

            let score = caps.compute_score();
            if score > best_score {
                best_score = score;
                best_device = Some(device);
            }
        }

        Ok(best_device)
    }

    /// Check if any candidate meets the requirements
    pub fn has_suitable_device(&self, candidates: &[&dyn Device]) -> Result<bool> {
        Ok(self.select_best_device(candidates)?.is_some())
    }
}

/// Compile-time device operation validation
pub struct TypedOperationValidator<P: PhantomDevice, Op: TypedDeviceOperation<P>> {
    _phantom: PhantomData<(P, Op)>,
}

impl<P: PhantomDevice, Op: TypedDeviceOperation<P>> TypedOperationValidator<P, Op> {
    /// Validate that the operation can be executed on the device type
    pub const VALID: bool = true; // In a real implementation, this would use const evaluation

    /// Get validation result
    pub fn validate() -> bool {
        Op::is_supported()
    }

    /// Get operation requirements
    pub fn requirements() -> DeviceRequirements {
        Op::device_requirements()
    }
}

/// Utility functions for typed device programming
pub mod utils {
    use super::*;

    /// Create a typed device builder for a specific phantom type
    pub fn builder<P: PhantomDevice>() -> TypedDeviceBuilder<P> {
        TypedDeviceBuilder::new()
    }

    /// Create a device selector with requirements
    pub fn selector<P: PhantomDevice>(requirements: DeviceRequirements) -> TypedDeviceSelector<P> {
        TypedDeviceSelector::new(requirements)
    }

    /// Validate that an operation can be executed on a device type
    pub fn validate_operation<P: PhantomDevice, Op: TypedDeviceOperation<P>>() -> bool {
        TypedOperationValidator::<P, Op>::validate()
    }

    /// Convert a runtime device to a typed device if compatible
    pub fn try_convert_to_typed<P: PhantomDevice>(
        device: Box<dyn Device>,
    ) -> std::result::Result<TypedDeviceInstance<P>, (Box<dyn Device>, crate::error::TorshError)>
    {
        if device.device_type() != P::DEVICE_TYPE {
            let error = crate::error::TorshError::InvalidArgument(format!(
                "Device type mismatch: expected {:?}, got {:?}",
                P::DEVICE_TYPE,
                device.device_type()
            ));
            return Err((device, error));
        }

        match TypedDeviceInstance::new(device.device_type(), DeviceRequirements::basic()) {
            Ok(typed_device) => Ok(typed_device),
            Err(error) => Err((device, error)),
        }
    }

    /// Check if a device satisfies typed requirements
    pub fn satisfies_requirements<P: PhantomDevice>(
        device: &dyn Device,
        requirements: &DeviceRequirements,
    ) -> Result<bool> {
        if device.device_type() != P::DEVICE_TYPE {
            return Ok(false);
        }

        let caps = device.capabilities()?;
        Ok(requirements.satisfied_by(&caps))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::core::Device;

    // Mock typed operation for testing
    struct MockOperation;

    impl TypedDeviceOperation<PhantomCpu> for MockOperation {
        type Output = u32;

        fn execute(&self, _device: &dyn Device) -> Result<Self::Output> {
            Ok(42)
        }

        fn operation_name() -> &'static str {
            "MockOperation"
        }

        fn device_requirements() -> DeviceRequirements {
            DeviceRequirements::basic()
        }
    }

    #[test]
    fn test_device_requirements() {
        let basic = DeviceRequirements::basic();
        assert!(!basic.requires_gpu);
        assert!(basic.min_memory.is_none());

        let gpu_req = DeviceRequirements::gpu().with_memory(1024 * 1024 * 1024);
        assert!(gpu_req.requires_gpu);
        assert_eq!(gpu_req.min_memory, Some(1024 * 1024 * 1024));
    }

    #[test]
    fn test_typed_device_builder() {
        let builder = TypedDeviceBuilder::<PhantomCpu>::new()
            .with_memory(512 * 1024 * 1024)
            .with_features(vec!["avx".to_string()]);

        let device = builder.build().unwrap();
        assert_eq!(device.device_type(), DeviceType::Cpu);
        assert_eq!(device.requirements().min_memory, Some(512 * 1024 * 1024));
    }

    #[test]
    fn test_typed_device_factory() {
        let cpu_device = TypedDeviceFactory::create_cpu().unwrap();
        assert_eq!(cpu_device.device_type(), DeviceType::Cpu);

        let runtime_device = TypedDeviceFactory::create_from_type(DeviceType::Cpu).unwrap();
        assert_eq!(runtime_device.device_type(), DeviceType::Cpu);
    }

    #[test]
    fn test_typed_operation_execution() {
        let device = TypedDeviceFactory::create_cpu().unwrap();
        let operation = MockOperation;

        let result = device.execute_typed_operation(operation).unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_device_selector() {
        let requirements = DeviceRequirements::basic();
        let selector = TypedDeviceSelector::<PhantomCpu>::new(requirements);

        let device = TypedDeviceFactory::create_cpu().unwrap();
        let candidates = vec![&device as &dyn Device];

        let selected = selector.select_best_device(&candidates).unwrap();
        assert!(selected.is_some());
        assert!(selector.has_suitable_device(&candidates).unwrap());
    }

    #[test]
    fn test_operation_validator() {
        assert!(TypedOperationValidator::<PhantomCpu, MockOperation>::validate());

        let requirements = TypedOperationValidator::<PhantomCpu, MockOperation>::requirements();
        assert!(!requirements.requires_gpu);
    }

    #[test]
    fn test_utils_functions() {
        let builder = utils::builder::<PhantomCpu>();
        let device = builder.build().unwrap();

        let requirements = DeviceRequirements::basic();
        let satisfies =
            utils::satisfies_requirements::<PhantomCpu>(&device, &requirements).unwrap();
        assert!(satisfies);

        assert!(utils::validate_operation::<PhantomCpu, MockOperation>());
    }

    #[test]
    fn test_gpu_memory_info() {
        let memory_info = GpuMemoryInfo {
            total: 8 * 1024 * 1024 * 1024, // 8GB
            free: 6 * 1024 * 1024 * 1024,  // 6GB
            used: 2 * 1024 * 1024 * 1024,  // 2GB
        };

        assert_eq!(memory_info.utilization_percent(), 25.0);
    }
}
