//! Core device trait definitions and interfaces
//!
//! This module provides the fundamental Device trait and related interfaces
//! that form the foundation of the device abstraction layer in ToRSh.

use crate::device::{DeviceCapabilities, DeviceType};
use crate::error::Result;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

/// Core device trait that all device implementations must satisfy
///
/// This trait defines the essential interface for all compute devices in ToRSh.
/// It provides methods for device identification, capability querying, memory
/// management, and lifecycle operations.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::device::{Device, DeviceType};
///
/// fn use_device<D: Device>(device: &D) -> Result<()> {
///     println!("Device: {}", device.name());
///     println!("Type: {:?}", device.device_type());
///     println!("Available: {}", device.is_available()?);
///
///     if device.is_available()? {
///         device.synchronize()?;
///     }
///     Ok(())
/// }
/// ```
pub trait Device: Debug + Send + Sync + 'static {
    /// Get the device type
    fn device_type(&self) -> DeviceType;

    /// Get a human-readable name for the device
    fn name(&self) -> &str;

    /// Check if the device is currently available for use
    fn is_available(&self) -> Result<bool>;

    /// Get device capabilities
    fn capabilities(&self) -> Result<DeviceCapabilities>;

    /// Synchronize the device (wait for all operations to complete)
    fn synchronize(&self) -> Result<()>;

    /// Reset the device to a clean state
    fn reset(&self) -> Result<()>;

    /// Get device-specific information as Any trait object
    fn as_any(&self) -> &dyn Any;

    /// Get device-specific mutable information as Any trait object
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Clone the device (for shared ownership)
    fn clone_device(&self) -> Result<Box<dyn Device>>;

    /// Check if this device is the same as another device
    fn is_same_device(&self, other: &dyn Device) -> bool {
        self.device_type() == other.device_type()
    }

    /// Get the unique device identifier
    fn device_id(&self) -> String {
        match self.device_type() {
            DeviceType::Cpu => "cpu".to_string(),
            DeviceType::Cuda(idx) => format!("cuda:{}", idx),
            DeviceType::Metal(idx) => format!("metal:{}", idx),
            DeviceType::Wgpu(idx) => format!("wgpu:{}", idx),
        }
    }

    /// Check if the device supports a specific feature
    fn supports_feature(&self, feature: &str) -> Result<bool> {
        Ok(self.capabilities()?.supports_feature(feature))
    }

    /// Get current memory usage information
    fn memory_info(&self) -> Result<DeviceMemoryInfo> {
        let caps = self.capabilities()?;
        Ok(DeviceMemoryInfo {
            total: caps.total_memory(),
            available: caps.available_memory(),
            used: caps.total_memory() - caps.available_memory(),
        })
    }

    /// Perform device-specific cleanup
    fn cleanup(&mut self) -> Result<()> {
        // Default implementation - devices can override for specific cleanup
        Ok(())
    }
}

/// Device memory information
#[derive(Debug, Clone, Copy)]
pub struct DeviceMemoryInfo {
    /// Total memory in bytes
    pub total: u64,
    /// Available memory in bytes
    pub available: u64,
    /// Used memory in bytes
    pub used: u64,
}

impl DeviceMemoryInfo {
    /// Get memory utilization as a percentage (0.0 to 100.0)
    pub fn utilization_percent(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.used as f64 / self.total as f64) * 100.0
        }
    }

    /// Get available memory as a percentage (0.0 to 100.0)
    pub fn available_percent(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.available as f64 / self.total as f64) * 100.0
        }
    }

    /// Check if memory usage is above a threshold
    pub fn is_memory_pressure(&self, threshold_percent: f64) -> bool {
        self.utilization_percent() > threshold_percent
    }
}

impl std::fmt::Display for DeviceMemoryInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Memory(total={:.1}MB, used={:.1}MB, available={:.1}MB, utilization={:.1}%)",
            self.total as f64 / (1024.0 * 1024.0),
            self.used as f64 / (1024.0 * 1024.0),
            self.available as f64 / (1024.0 * 1024.0),
            self.utilization_percent()
        )
    }
}

/// Device state enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceState {
    /// Device is uninitialized
    Uninitialized,
    /// Device is initializing
    Initializing,
    /// Device is ready for use
    Ready,
    /// Device is busy with operations
    Busy,
    /// Device is in error state
    Error,
    /// Device is being reset
    Resetting,
    /// Device is shutting down
    ShuttingDown,
}

impl std::fmt::Display for DeviceState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceState::Uninitialized => write!(f, "Uninitialized"),
            DeviceState::Initializing => write!(f, "Initializing"),
            DeviceState::Ready => write!(f, "Ready"),
            DeviceState::Busy => write!(f, "Busy"),
            DeviceState::Error => write!(f, "Error"),
            DeviceState::Resetting => write!(f, "Resetting"),
            DeviceState::ShuttingDown => write!(f, "Shutting Down"),
        }
    }
}

/// Device lifecycle manager for handling device state transitions
#[derive(Debug)]
pub struct DeviceLifecycle {
    state: Mutex<DeviceState>,
    error_info: Mutex<Option<String>>,
    initialization_time: Mutex<Option<std::time::Instant>>,
}

impl DeviceLifecycle {
    /// Create a new device lifecycle manager
    pub fn new() -> Self {
        Self {
            state: Mutex::new(DeviceState::Uninitialized),
            error_info: Mutex::new(None),
            initialization_time: Mutex::new(None),
        }
    }

    /// Get the current device state
    pub fn state(&self) -> DeviceState {
        *self.state.lock().expect("lock should not be poisoned")
    }

    /// Set the device state
    pub fn set_state(&self, new_state: DeviceState) -> Result<()> {
        let mut state = self.state.lock().expect("lock should not be poisoned");
        match (*state, new_state) {
            // Valid transitions
            (DeviceState::Uninitialized, DeviceState::Initializing) => {
                *self
                    .initialization_time
                    .lock()
                    .expect("lock should not be poisoned") = Some(std::time::Instant::now());
            }
            (DeviceState::Uninitialized, DeviceState::Ready) => {} // Allow direct transition to ready
            (DeviceState::Initializing, DeviceState::Ready) => {}
            (DeviceState::Ready, DeviceState::Busy) => {}
            (DeviceState::Busy, DeviceState::Ready) => {}
            (_, DeviceState::Error) => {} // Can transition to error from any state
            (_, DeviceState::Resetting) => {} // Can reset from any state
            (DeviceState::Resetting, DeviceState::Ready) => {}
            (DeviceState::Resetting, DeviceState::Uninitialized) => {} // Can go back to uninitialized after reset
            (_, DeviceState::ShuttingDown) => {} // Can shutdown from any state

            // Invalid transitions
            (current, target) => {
                return Err(crate::error::TorshError::InvalidState(format!(
                    "Invalid state transition from {:?} to {:?}",
                    current, target
                )));
            }
        }
        *state = new_state;
        Ok(())
    }

    /// Set error state with error information
    pub fn set_error(&self, error_info: String) -> Result<()> {
        *self.error_info.lock().expect("lock should not be poisoned") = Some(error_info);
        self.set_state(DeviceState::Error)
    }

    /// Get error information if in error state
    pub fn error_info(&self) -> Option<String> {
        self.error_info
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Get initialization time if available
    pub fn initialization_time(&self) -> Option<std::time::Duration> {
        self.initialization_time
            .lock()
            .expect("lock should not be poisoned")
            .map(|start| start.elapsed())
    }

    /// Check if device is ready for operations
    pub fn is_ready(&self) -> bool {
        matches!(self.state(), DeviceState::Ready)
    }

    /// Check if device is in error state
    pub fn is_error(&self) -> bool {
        matches!(self.state(), DeviceState::Error)
    }

    /// Reset to uninitialized state
    pub fn reset(&self) -> Result<()> {
        self.set_state(DeviceState::Resetting)?;
        *self.error_info.lock().expect("lock should not be poisoned") = None;
        *self
            .initialization_time
            .lock()
            .expect("lock should not be poisoned") = None;
        self.set_state(DeviceState::Uninitialized)
    }
}

impl Default for DeviceLifecycle {
    fn default() -> Self {
        Self::new()
    }
}

/// Device context for managing device-specific resources and state
#[derive(Debug)]
pub struct DeviceContext {
    device_type: DeviceType,
    lifecycle: DeviceLifecycle,
    properties: Mutex<HashMap<String, String>>,
    resource_handles: Mutex<Vec<Box<dyn Any + Send + Sync>>>,
}

impl DeviceContext {
    /// Create a new device context
    pub fn new(device_type: DeviceType) -> Self {
        Self {
            device_type,
            lifecycle: DeviceLifecycle::new(),
            properties: Mutex::new(HashMap::new()),
            resource_handles: Mutex::new(Vec::new()),
        }
    }

    /// Get the device type
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    /// Get the lifecycle manager
    pub fn lifecycle(&self) -> &DeviceLifecycle {
        &self.lifecycle
    }

    /// Set a device property
    pub fn set_property(&self, key: String, value: String) {
        let mut props = self.properties.lock().expect("lock should not be poisoned");
        props.insert(key, value);
    }

    /// Get a device property
    pub fn get_property(&self, key: &str) -> Option<String> {
        let props = self.properties.lock().expect("lock should not be poisoned");
        props.get(key).cloned()
    }

    /// Get all device properties
    pub fn properties(&self) -> HashMap<String, String> {
        self.properties
            .lock()
            .expect("lock should not be poisoned")
            .clone()
    }

    /// Add a resource handle
    pub fn add_resource<T: Any + Send + Sync + 'static>(&self, resource: T) {
        let mut handles = self
            .resource_handles
            .lock()
            .expect("lock should not be poisoned");
        handles.push(Box::new(resource));
    }

    /// Clear all resources
    pub fn clear_resources(&self) {
        let mut handles = self
            .resource_handles
            .lock()
            .expect("lock should not be poisoned");
        handles.clear();
    }

    /// Get the number of managed resources
    pub fn resource_count(&self) -> usize {
        let handles = self
            .resource_handles
            .lock()
            .expect("lock should not be poisoned");
        handles.len()
    }
}

/// Device factory trait for creating device instances
pub trait DeviceFactory: Debug + Send + Sync {
    /// Create a device of the specified type
    fn create_device(&self, device_type: DeviceType) -> Result<Box<dyn Device>>;

    /// Check if the factory supports creating devices of the given type
    fn supports_device_type(&self, device_type: DeviceType) -> bool;

    /// Get the name of this factory
    fn factory_name(&self) -> &str;

    /// Get supported device types
    fn supported_device_types(&self) -> Vec<DeviceType>;
}

/// Device registry for managing device factories and instances
#[derive(Debug)]
pub struct DeviceRegistry {
    factories: Mutex<HashMap<DeviceType, Box<dyn DeviceFactory>>>,
    devices: Mutex<HashMap<String, Arc<dyn Device>>>,
}

impl DeviceRegistry {
    /// Create a new device registry
    pub fn new() -> Self {
        Self {
            factories: Mutex::new(HashMap::new()),
            devices: Mutex::new(HashMap::new()),
        }
    }

    /// Register a device factory
    pub fn register_factory<F: DeviceFactory + 'static>(&self, factory: F) -> Result<()> {
        let mut factories = self.factories.lock().expect("lock should not be poisoned");
        let device_types = factory.supported_device_types();

        for device_type in device_types {
            if factories.contains_key(&device_type) {
                return Err(crate::error::TorshError::InvalidArgument(format!(
                    "Factory for device type {:?} already registered",
                    device_type
                )));
            }
        }

        let factory_box = Box::new(factory);
        let supported_types = factory_box.supported_device_types();
        if let Some(&first_type) = supported_types.first() {
            factories.insert(first_type, factory_box);
        }

        Ok(())
    }

    /// Create a device using registered factories
    pub fn create_device(&self, device_type: DeviceType) -> Result<Box<dyn Device>> {
        let factories = self.factories.lock().expect("lock should not be poisoned");

        match factories.get(&device_type) {
            Some(factory) => factory.create_device(device_type),
            None => Err(crate::error::TorshError::General(
                crate::error::GeneralError::DeviceError(format!(
                    "No factory registered for device type {:?}",
                    device_type
                )),
            )),
        }
    }

    /// Get or create a cached device instance
    pub fn get_or_create_device(&self, device_type: DeviceType) -> Result<Arc<dyn Device>> {
        let device_id = format!("{:?}", device_type);

        {
            let devices = self.devices.lock().expect("lock should not be poisoned");
            if let Some(device) = devices.get(&device_id) {
                return Ok(device.clone());
            }
        }

        // Create new device
        let device = self.create_device(device_type)?;
        let arc_device: Arc<dyn Device> = unsafe { Arc::from_raw(Box::into_raw(device)) };

        {
            let mut devices = self.devices.lock().expect("lock should not be poisoned");
            devices.insert(device_id, arc_device.clone());
        }

        Ok(arc_device)
    }

    /// Get all registered device types
    pub fn registered_device_types(&self) -> Vec<DeviceType> {
        let factories = self.factories.lock().expect("lock should not be poisoned");
        factories.keys().copied().collect()
    }

    /// Clear all cached devices
    pub fn clear_devices(&self) {
        let mut devices = self.devices.lock().expect("lock should not be poisoned");
        devices.clear();
    }

    /// Get registry statistics
    pub fn statistics(&self) -> RegistryStatistics {
        let factories = self.factories.lock().expect("lock should not be poisoned");
        let devices = self.devices.lock().expect("lock should not be poisoned");

        RegistryStatistics {
            registered_factories: factories.len(),
            cached_devices: devices.len(),
            supported_device_types: factories.keys().copied().collect(),
        }
    }
}

impl Default for DeviceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Registry statistics
#[derive(Debug, Clone)]
pub struct RegistryStatistics {
    pub registered_factories: usize,
    pub cached_devices: usize,
    pub supported_device_types: Vec<DeviceType>,
}

impl std::fmt::Display for RegistryStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Registry(factories={}, cached_devices={}, types={:?})",
            self.registered_factories, self.cached_devices, self.supported_device_types
        )
    }
}

/// Global device registry instance
static GLOBAL_REGISTRY: std::sync::OnceLock<DeviceRegistry> = std::sync::OnceLock::new();

/// Get the global device registry
pub fn global_device_registry() -> &'static DeviceRegistry {
    GLOBAL_REGISTRY.get_or_init(DeviceRegistry::new)
}

/// Initialize the global device registry with custom factories
pub fn initialize_global_registry<F>(init_fn: F) -> Result<()>
where
    F: FnOnce(&DeviceRegistry) -> Result<()>,
{
    let registry = global_device_registry();
    init_fn(registry)
}

/// Utility functions for device operations
pub mod utils {
    use super::*;

    /// Check if two devices are compatible for operations
    pub fn devices_compatible(a: &dyn Device, b: &dyn Device) -> bool {
        a.device_type() == b.device_type()
    }

    /// Find the best device among a list based on capabilities
    pub fn find_best_device<'a>(devices: &'a [&'a dyn Device]) -> Result<Option<&'a dyn Device>> {
        if devices.is_empty() {
            return Ok(None);
        }

        let mut best_device = devices[0];
        let mut best_score = 0u64;

        for &device in devices {
            if !device.is_available()? {
                continue;
            }

            let caps = device.capabilities()?;
            let score = caps.compute_score();

            if score > best_score {
                best_score = score;
                best_device = device;
            }
        }

        Ok(if best_score > 0 {
            Some(best_device)
        } else {
            None
        })
    }

    /// Synchronize multiple devices
    pub fn synchronize_devices(devices: &[&dyn Device]) -> Result<()> {
        for device in devices {
            device.synchronize()?;
        }
        Ok(())
    }

    /// Check if all devices are available
    pub fn all_devices_available(devices: &[&dyn Device]) -> Result<bool> {
        for device in devices {
            if !device.is_available()? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Get memory information for multiple devices
    pub fn get_devices_memory_info(devices: &[&dyn Device]) -> Result<Vec<DeviceMemoryInfo>> {
        devices.iter().map(|device| device.memory_info()).collect()
    }

    /// Filter devices by available memory threshold
    pub fn filter_devices_by_memory<'a>(
        devices: &'a [&'a dyn Device],
        min_available_mb: u64,
    ) -> Result<Vec<&'a dyn Device>> {
        let mut filtered = Vec::new();

        for &device in devices {
            let memory_info = device.memory_info()?;
            let available_mb = memory_info.available / (1024 * 1024);

            if available_mb >= min_available_mb {
                filtered.push(device);
            }
        }

        Ok(filtered)
    }

    /// Create a device summary string
    pub fn device_summary(device: &dyn Device) -> Result<String> {
        let caps = device.capabilities()?;
        let memory_info = device.memory_info()?;

        Ok(format!(
            "{} - {} ({:.1}MB available, {:.1}% used)",
            device.name(),
            caps.device_type(),
            memory_info.available as f64 / (1024.0 * 1024.0),
            memory_info.utilization_percent()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock device implementation for testing
    #[derive(Debug)]
    struct MockDevice {
        device_type: DeviceType,
        name: String,
        available: bool,
    }

    impl MockDevice {
        fn new(device_type: DeviceType, name: String) -> Self {
            Self {
                device_type,
                name,
                available: true,
            }
        }
    }

    impl Device for MockDevice {
        fn device_type(&self) -> DeviceType {
            self.device_type
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn is_available(&self) -> Result<bool> {
            Ok(self.available)
        }

        fn capabilities(&self) -> Result<DeviceCapabilities> {
            DeviceCapabilities::detect(self.device_type)
        }

        fn synchronize(&self) -> Result<()> {
            Ok(())
        }

        fn reset(&self) -> Result<()> {
            Ok(())
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }

        fn clone_device(&self) -> Result<Box<dyn Device>> {
            Ok(Box::new(MockDevice {
                device_type: self.device_type,
                name: self.name.clone(),
                available: self.available,
            }))
        }
    }

    #[test]
    fn test_device_lifecycle() {
        let lifecycle = DeviceLifecycle::new();
        assert_eq!(lifecycle.state(), DeviceState::Uninitialized);
        assert!(!lifecycle.is_ready());

        lifecycle.set_state(DeviceState::Initializing).unwrap();
        lifecycle.set_state(DeviceState::Ready).unwrap();
        assert!(lifecycle.is_ready());

        lifecycle.set_error("Test error".to_string()).unwrap();
        assert!(lifecycle.is_error());
        assert_eq!(lifecycle.error_info(), Some("Test error".to_string()));

        lifecycle.reset().unwrap();
        assert_eq!(lifecycle.state(), DeviceState::Uninitialized);
        assert!(lifecycle.error_info().is_none());
    }

    #[test]
    fn test_device_context() {
        let context = DeviceContext::new(DeviceType::Cpu);
        assert_eq!(context.device_type(), DeviceType::Cpu);

        context.set_property("test_prop".to_string(), "test_value".to_string());
        assert_eq!(
            context.get_property("test_prop"),
            Some("test_value".to_string())
        );

        context.add_resource(42u32);
        assert_eq!(context.resource_count(), 1);

        context.clear_resources();
        assert_eq!(context.resource_count(), 0);
    }

    #[test]
    fn test_device_memory_info() {
        let memory_info = DeviceMemoryInfo {
            total: 1024 * 1024 * 1024,    // 1GB
            available: 512 * 1024 * 1024, // 512MB
            used: 512 * 1024 * 1024,      // 512MB
        };

        assert_eq!(memory_info.utilization_percent(), 50.0);
        assert_eq!(memory_info.available_percent(), 50.0);
        assert!(!memory_info.is_memory_pressure(75.0));
        assert!(memory_info.is_memory_pressure(25.0));
    }

    #[test]
    fn test_mock_device() {
        let device = MockDevice::new(DeviceType::Cpu, "Test CPU".to_string());
        assert_eq!(device.device_type(), DeviceType::Cpu);
        assert_eq!(device.name(), "Test CPU");
        assert!(device.is_available().unwrap());
        assert_eq!(device.device_id(), "cpu");

        let cloned = device.clone_device().unwrap();
        assert!(device.is_same_device(cloned.as_ref()));
    }

    #[test]
    fn test_device_registry() {
        let registry = DeviceRegistry::new();

        // Note: We can't easily test factory registration without implementing
        // a full mock factory, which would require more complex setup

        let stats = registry.statistics();
        assert_eq!(stats.registered_factories, 0);
        assert_eq!(stats.cached_devices, 0);
    }

    #[test]
    fn test_utils_functions() {
        let device1 = MockDevice::new(DeviceType::Cpu, "CPU 1".to_string());
        let device2 = MockDevice::new(DeviceType::Cpu, "CPU 2".to_string());
        let device3 = MockDevice::new(DeviceType::Cuda(0), "GPU 1".to_string());

        assert!(utils::devices_compatible(&device1, &device2));
        assert!(!utils::devices_compatible(&device1, &device3));

        let devices = vec![&device1 as &dyn Device, &device2, &device3];
        assert!(utils::all_devices_available(&devices).unwrap());

        utils::synchronize_devices(&devices).unwrap();

        let summary = utils::device_summary(&device1).unwrap();
        assert!(summary.contains("CPU 1"));
    }
}
