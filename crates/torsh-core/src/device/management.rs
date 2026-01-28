//! Device discovery and management
//!
//! This module provides high-level device management capabilities including
//! device discovery, lifecycle management, resource allocation, and health monitoring.

use crate::device::core::{DeviceLifecycle, DeviceState};
#[cfg(test)]
use crate::device::implementations::CpuDevice;
use crate::device::implementations::DeviceFactory;
use crate::device::{Device, DeviceCapabilities, DeviceType};
use crate::error::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Device manager for coordinating multiple devices
///
/// Provides centralized management of device resources, including discovery,
/// allocation, health monitoring, and load balancing across available devices.
///
/// # Examples
///
/// ```ignore
/// use torsh_core::device::DeviceManager;
///
/// let mut manager = DeviceManager::new();
/// manager.discover_devices()?;
///
/// // Get the best available device
/// let device = manager.get_best_device()?;
/// println!("Selected device: {}", device.name());
///
/// // Get devices by type
/// let gpu_devices = manager.get_devices_by_type(DeviceType::Cuda(0))?;
/// ```
#[derive(Debug)]
pub struct DeviceManager {
    devices: RwLock<HashMap<String, Arc<dyn Device>>>,
    device_states: RwLock<HashMap<String, Arc<DeviceLifecycle>>>,
    device_health: RwLock<HashMap<String, DeviceHealth>>,
    allocation_strategy: AllocationStrategy,
    health_monitor: Arc<HealthMonitor>,
    discovery_config: DiscoveryConfig,
}

impl DeviceManager {
    /// Create a new device manager
    pub fn new() -> Self {
        Self {
            devices: RwLock::new(HashMap::new()),
            device_states: RwLock::new(HashMap::new()),
            device_health: RwLock::new(HashMap::new()),
            allocation_strategy: AllocationStrategy::BestFit,
            health_monitor: Arc::new(HealthMonitor::new()),
            discovery_config: DiscoveryConfig::default(),
        }
    }

    /// Create device manager with specific configuration
    pub fn with_config(config: ManagerConfig) -> Self {
        Self {
            devices: RwLock::new(HashMap::new()),
            device_states: RwLock::new(HashMap::new()),
            device_health: RwLock::new(HashMap::new()),
            allocation_strategy: config.allocation_strategy,
            health_monitor: Arc::new(HealthMonitor::with_config(config.health_config)),
            discovery_config: config.discovery_config,
        }
    }

    /// Discover all available devices
    pub fn discover_devices(&self) -> Result<usize> {
        let mut discovered_count = 0;

        // Discover CPU devices
        if self.discovery_config.enable_cpu {
            self.discover_cpu_devices()?;
            discovered_count += 1;
        }

        // Discover CUDA devices
        if self.discovery_config.enable_cuda {
            discovered_count += self.discover_cuda_devices()?;
        }

        // Discover Metal devices
        if self.discovery_config.enable_metal {
            discovered_count += self.discover_metal_devices()?;
        }

        // Discover WebGPU devices
        if self.discovery_config.enable_wgpu {
            discovered_count += self.discover_wgpu_devices()?;
        }

        // Start health monitoring for discovered devices
        self.start_health_monitoring()?;

        Ok(discovered_count)
    }

    /// Get device by ID
    pub fn get_device(&self, device_id: &str) -> Option<Arc<dyn Device>> {
        let devices = self.devices.read().expect("lock should not be poisoned");
        devices.get(device_id).cloned()
    }

    /// Get all devices
    pub fn get_all_devices(&self) -> Vec<Arc<dyn Device>> {
        let devices = self.devices.read().expect("lock should not be poisoned");
        devices.values().cloned().collect()
    }

    /// Get devices by type
    pub fn get_devices_by_type(&self, device_type: DeviceType) -> Vec<Arc<dyn Device>> {
        let devices = self.devices.read().expect("lock should not be poisoned");
        devices
            .values()
            .filter(|device| device.device_type() == device_type)
            .cloned()
            .collect()
    }

    /// Get the best available device based on allocation strategy
    pub fn get_best_device(&self) -> Result<Option<Arc<dyn Device>>> {
        let devices = self.get_available_devices()?;
        if devices.is_empty() {
            return Ok(None);
        }

        match self.allocation_strategy {
            AllocationStrategy::BestFit => self.select_best_fit_device(&devices),
            AllocationStrategy::LoadBalanced => self.select_load_balanced_device(&devices),
            AllocationStrategy::Fastest => self.select_fastest_device(&devices),
            AllocationStrategy::MostMemory => self.select_most_memory_device(&devices),
        }
    }

    /// Get devices that are currently available
    pub fn get_available_devices(&self) -> Result<Vec<Arc<dyn Device>>> {
        let devices = self.devices.read().expect("lock should not be poisoned");
        let mut available = Vec::new();

        for device in devices.values() {
            if device.is_available()? && self.is_device_healthy(device.as_ref())? {
                available.push(device.clone());
            }
        }

        Ok(available)
    }

    /// Add a device to the manager
    pub fn add_device(&self, device: Box<dyn Device>) -> Result<String> {
        let device_id = device.device_id();
        let arc_device: Arc<dyn Device> = device.into();

        // Create lifecycle management for the device
        let lifecycle = Arc::new(DeviceLifecycle::new());
        lifecycle.set_state(DeviceState::Ready)?;

        // Initialize health tracking
        let health = DeviceHealth::new();

        {
            let mut devices = self.devices.write().expect("lock should not be poisoned");
            let mut states = self
                .device_states
                .write()
                .expect("lock should not be poisoned");
            let mut health_map = self
                .device_health
                .write()
                .expect("lock should not be poisoned");

            devices.insert(device_id.clone(), arc_device);
            states.insert(device_id.clone(), lifecycle);
            health_map.insert(device_id.clone(), health);
        }

        Ok(device_id)
    }

    /// Remove a device from the manager
    pub fn remove_device(&self, device_id: &str) -> Option<Arc<dyn Device>> {
        let mut devices = self.devices.write().expect("lock should not be poisoned");
        let mut states = self
            .device_states
            .write()
            .expect("lock should not be poisoned");
        let mut health_map = self
            .device_health
            .write()
            .expect("lock should not be poisoned");

        states.remove(device_id);
        health_map.remove(device_id);
        devices.remove(device_id)
    }

    /// Get device count
    pub fn device_count(&self) -> usize {
        let devices = self.devices.read().expect("lock should not be poisoned");
        devices.len()
    }

    /// Get manager statistics
    pub fn statistics(&self) -> ManagerStatistics {
        let devices = self.devices.read().expect("lock should not be poisoned");
        let health_map = self
            .device_health
            .read()
            .expect("lock should not be poisoned");

        let total_devices = devices.len();
        let available_devices = devices
            .values()
            .filter(|device| device.is_available().unwrap_or(false))
            .count();

        let healthy_devices = health_map
            .values()
            .filter(|health| health.is_healthy())
            .count();

        let device_types = devices
            .values()
            .map(|device| device.device_type())
            .collect::<std::collections::HashSet<_>>()
            .len();

        ManagerStatistics {
            total_devices,
            available_devices,
            healthy_devices,
            device_types,
        }
    }

    /// Synchronize all devices
    pub fn synchronize_all(&self) -> Result<()> {
        let devices = self.devices.read().expect("lock should not be poisoned");
        for device in devices.values() {
            device.synchronize()?;
        }
        Ok(())
    }

    /// Reset all devices
    pub fn reset_all(&self) -> Result<()> {
        let devices = self.devices.read().expect("lock should not be poisoned");
        for device in devices.values() {
            device.reset()?;
        }
        Ok(())
    }

    fn discover_cpu_devices(&self) -> Result<()> {
        let cpu_device = DeviceFactory::create_device(DeviceType::Cpu)?;
        self.add_device(cpu_device)?;
        Ok(())
    }

    fn discover_cuda_devices(&self) -> Result<usize> {
        #[allow(unused_mut)] // mut needed for conditional compilation features
        let mut count = 0;

        #[cfg(feature = "cuda")]
        {
            // In a real implementation, this would query CUDA runtime for device count
            for index in 0..2 {
                if let Ok(device) = DeviceFactory::create_device(DeviceType::Cuda(index)) {
                    self.add_device(device)?;
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    fn discover_metal_devices(&self) -> Result<usize> {
        #[allow(unused_mut)] // mut needed for conditional compilation features
        let mut count = 0;

        #[cfg(target_os = "macos")]
        {
            // On macOS, typically there's one Metal device
            if let Ok(device) = DeviceFactory::create_device(DeviceType::Metal(0)) {
                self.add_device(device)?;
                count += 1;
            }
        }

        Ok(count)
    }

    fn discover_wgpu_devices(&self) -> Result<usize> {
        #[allow(unused_mut)] // mut needed for conditional compilation features
        let mut count = 0;

        #[cfg(feature = "wgpu")]
        {
            // WebGPU device discovery would enumerate available adapters
            if let Ok(device) = DeviceFactory::create_device(DeviceType::Wgpu(0)) {
                self.add_device(device)?;
                count += 1;
            }
        }

        Ok(count)
    }

    fn start_health_monitoring(&self) -> Result<()> {
        let devices = self.devices.read().expect("lock should not be poisoned");
        for (device_id, device) in devices.iter() {
            self.health_monitor
                .add_device(device_id.clone(), device.clone())?;
        }
        Ok(())
    }

    fn is_device_healthy(&self, device: &dyn Device) -> Result<bool> {
        let health_map = self
            .device_health
            .read()
            .expect("lock should not be poisoned");
        let device_id = device.device_id();
        Ok(health_map
            .get(&device_id)
            .map(|health| health.is_healthy())
            .unwrap_or(false))
    }

    fn select_best_fit_device(
        &self,
        devices: &[Arc<dyn Device>],
    ) -> Result<Option<Arc<dyn Device>>> {
        if devices.is_empty() {
            return Ok(None);
        }

        let mut best_device = None;
        let mut best_score = 0;

        for device in devices {
            let caps = device.capabilities()?;
            let score = caps.compute_score();
            if score > best_score {
                best_score = score;
                best_device = Some(device.clone());
            }
        }

        Ok(best_device)
    }

    fn select_load_balanced_device(
        &self,
        devices: &[Arc<dyn Device>],
    ) -> Result<Option<Arc<dyn Device>>> {
        // Simple load balancing - select device with least utilization
        // In a real implementation, this would track actual device usage
        self.select_best_fit_device(devices)
    }

    fn select_fastest_device(
        &self,
        devices: &[Arc<dyn Device>],
    ) -> Result<Option<Arc<dyn Device>>> {
        if devices.is_empty() {
            return Ok(None);
        }

        let mut fastest_device = None;
        let mut best_speed = 0;

        for device in devices {
            let caps = device.capabilities()?;
            let speed = caps.clock_rate().unwrap_or(1000) * caps.compute_units();
            if speed > best_speed {
                best_speed = speed;
                fastest_device = Some(device.clone());
            }
        }

        Ok(fastest_device)
    }

    fn select_most_memory_device(
        &self,
        devices: &[Arc<dyn Device>],
    ) -> Result<Option<Arc<dyn Device>>> {
        if devices.is_empty() {
            return Ok(None);
        }

        let mut best_device = None;
        let mut most_memory = 0;

        for device in devices {
            let caps = device.capabilities()?;
            let memory = caps.available_memory();
            if memory > most_memory {
                most_memory = memory;
                best_device = Some(device.clone());
            }
        }

        Ok(best_device)
    }
}

impl Default for DeviceManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Device allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    /// Select the device with the best overall performance
    BestFit,
    /// Balance load across available devices
    LoadBalanced,
    /// Select the fastest device
    Fastest,
    /// Select the device with the most available memory
    MostMemory,
}

/// Device health information
#[derive(Debug, Clone)]
pub struct DeviceHealth {
    is_healthy: bool,
    last_check: Instant,
    error_count: u32,
    temperature: Option<f32>,
    memory_pressure: Option<f32>,
}

impl Default for DeviceHealth {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceHealth {
    pub fn new() -> Self {
        Self {
            is_healthy: true,
            last_check: Instant::now(),
            error_count: 0,
            temperature: None,
            memory_pressure: None,
        }
    }

    pub fn is_healthy(&self) -> bool {
        self.is_healthy
    }

    pub fn error_count(&self) -> u32 {
        self.error_count
    }

    pub fn temperature(&self) -> Option<f32> {
        self.temperature
    }

    pub fn memory_pressure(&self) -> Option<f32> {
        self.memory_pressure
    }

    pub fn update_health(&mut self, healthy: bool) {
        self.is_healthy = healthy;
        self.last_check = Instant::now();
        if !healthy {
            self.error_count += 1;
        }
    }

    pub fn set_temperature(&mut self, temp: f32) {
        self.temperature = Some(temp);
    }

    pub fn set_memory_pressure(&mut self, pressure: f32) {
        self.memory_pressure = Some(pressure);
    }
}

/// Health monitor for tracking device health
#[derive(Debug)]
pub struct HealthMonitor {
    monitored_devices: Mutex<HashMap<String, Arc<dyn Device>>>,
    #[allow(dead_code)] // Health check interval - future implementation
    check_interval: Duration,
    #[allow(dead_code)] // Health monitoring configuration - future implementation
    config: HealthConfig,
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthMonitor {
    pub fn new() -> Self {
        Self {
            monitored_devices: Mutex::new(HashMap::new()),
            check_interval: Duration::from_secs(30),
            config: HealthConfig::default(),
        }
    }

    pub fn with_config(config: HealthConfig) -> Self {
        Self {
            monitored_devices: Mutex::new(HashMap::new()),
            check_interval: config.check_interval,
            config,
        }
    }

    pub fn add_device(&self, device_id: String, device: Arc<dyn Device>) -> Result<()> {
        let mut devices = self
            .monitored_devices
            .lock()
            .expect("lock should not be poisoned");
        devices.insert(device_id, device);
        Ok(())
    }

    pub fn remove_device(&self, device_id: &str) {
        let mut devices = self
            .monitored_devices
            .lock()
            .expect("lock should not be poisoned");
        devices.remove(device_id);
    }

    pub fn check_device_health(&self, device: &dyn Device) -> Result<bool> {
        // Basic health checks
        if !device.is_available()? {
            return Ok(false);
        }

        // Check capabilities are still accessible
        let _caps = device.capabilities()?;

        // In a real implementation, this would include:
        // - Temperature monitoring
        // - Memory usage checks
        // - Error rate monitoring
        // - Performance degradation detection

        Ok(true)
    }
}

/// Health monitoring configuration
#[derive(Debug, Clone)]
pub struct HealthConfig {
    pub check_interval: Duration,
    pub temperature_threshold: Option<f32>,
    pub memory_pressure_threshold: Option<f32>,
    pub error_rate_threshold: Option<f32>,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            temperature_threshold: Some(85.0),    // 85°C
            memory_pressure_threshold: Some(0.9), // 90%
            error_rate_threshold: Some(0.05),     // 5%
        }
    }
}

/// Device discovery configuration
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    pub enable_cpu: bool,
    pub enable_cuda: bool,
    pub enable_metal: bool,
    pub enable_wgpu: bool,
    pub auto_discovery: bool,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            enable_cpu: true,
            enable_cuda: cfg!(feature = "cuda"),
            enable_metal: cfg!(target_os = "macos"),
            enable_wgpu: cfg!(feature = "wgpu"),
            auto_discovery: true,
        }
    }
}

/// Manager configuration
#[derive(Debug, Clone)]
pub struct ManagerConfig {
    pub allocation_strategy: AllocationStrategy,
    pub discovery_config: DiscoveryConfig,
    pub health_config: HealthConfig,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            allocation_strategy: AllocationStrategy::BestFit,
            discovery_config: DiscoveryConfig::default(),
            health_config: HealthConfig::default(),
        }
    }
}

/// Manager statistics
#[derive(Debug, Clone)]
pub struct ManagerStatistics {
    pub total_devices: usize,
    pub available_devices: usize,
    pub healthy_devices: usize,
    pub device_types: usize,
}

impl std::fmt::Display for ManagerStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DeviceManager(total={}, available={}, healthy={}, types={})",
            self.total_devices, self.available_devices, self.healthy_devices, self.device_types
        )
    }
}

/// Global device manager instance
static GLOBAL_MANAGER: std::sync::OnceLock<DeviceManager> = std::sync::OnceLock::new();

/// Get the global device manager
pub fn global_device_manager() -> &'static DeviceManager {
    GLOBAL_MANAGER.get_or_init(|| {
        let manager = DeviceManager::new();
        // Auto-discover devices on first access
        let _ = manager.discover_devices();
        manager
    })
}

/// Initialize the global device manager with custom configuration
pub fn initialize_global_manager(config: ManagerConfig) -> Result<()> {
    // If already initialized, just return Ok
    if GLOBAL_MANAGER.get().is_some() {
        return Ok(());
    }

    let manager = DeviceManager::with_config(config);
    manager.discover_devices()?;

    GLOBAL_MANAGER.set(manager).map_err(|_| {
        crate::error::TorshError::InvalidState(
            "Global device manager already initialized".to_string(),
        )
    })?;

    Ok(())
}

/// Utility functions for device management
pub mod utils {
    use super::*;

    /// Create a manager with all available devices
    pub fn create_manager_with_all_devices() -> Result<DeviceManager> {
        let manager = DeviceManager::new();
        manager.discover_devices()?;
        Ok(manager)
    }

    /// Get device counts by type
    pub fn get_device_counts_by_type(manager: &DeviceManager) -> HashMap<DeviceType, usize> {
        let devices = manager.get_all_devices();
        let mut counts = HashMap::new();

        for device in devices {
            let device_type = device.device_type();
            *counts.entry(device_type).or_insert(0) += 1;
        }

        counts
    }

    /// Check if any GPU devices are available
    pub fn has_gpu_devices(manager: &DeviceManager) -> bool {
        let devices = manager.get_all_devices();
        devices.iter().any(|device| device.device_type().is_gpu())
    }

    /// Get the fastest device of each type
    pub fn get_fastest_device_per_type(
        manager: &DeviceManager,
    ) -> Result<HashMap<DeviceType, Arc<dyn Device>>> {
        let devices = manager.get_available_devices()?;
        let mut fastest_per_type = HashMap::new();

        for device in devices {
            let device_type = device.device_type();
            let caps = device.capabilities()?;
            let speed = caps.clock_rate().unwrap_or(1000) * caps.compute_units();

            match fastest_per_type.get(&device_type) {
                Some((_current_device, current_speed)) => {
                    if speed > *current_speed {
                        fastest_per_type.insert(device_type, (device, speed));
                    }
                }
                None => {
                    fastest_per_type.insert(device_type, (device, speed));
                }
            }
        }

        Ok(fastest_per_type
            .into_iter()
            .map(|(device_type, (device, _))| (device_type, device))
            .collect())
    }

    /// Create a summary of all devices
    pub fn create_device_summary(manager: &DeviceManager) -> Vec<String> {
        let devices = manager.get_all_devices();
        devices
            .iter()
            .map(|device| {
                let caps = device.capabilities().unwrap_or_else(|_| {
                    // Fallback capabilities
                    DeviceCapabilities::detect(device.device_type())
                        .expect("CPU device capabilities detection should always succeed")
                });

                format!(
                    "{} - {} ({:.1}MB, {} cores)",
                    device.name(),
                    device.device_type(),
                    caps.total_memory_mb(),
                    caps.compute_units()
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_manager_creation() {
        let manager = DeviceManager::new();
        assert_eq!(manager.device_count(), 0);

        let stats = manager.statistics();
        assert_eq!(stats.total_devices, 0);
    }

    #[test]
    fn test_device_discovery() {
        let manager = DeviceManager::new();
        let discovered = manager.discover_devices().unwrap();
        assert!(discovered > 0); // At least CPU should be discovered

        let devices = manager.get_all_devices();
        assert!(!devices.is_empty());

        // CPU device should always be available
        let cpu_devices = manager.get_devices_by_type(DeviceType::Cpu);
        assert_eq!(cpu_devices.len(), 1);
    }

    #[test]
    fn test_device_addition_and_removal() {
        let manager = DeviceManager::new();

        let cpu_device = DeviceFactory::create_device(DeviceType::Cpu).unwrap();
        let device_id = manager.add_device(cpu_device).unwrap();

        assert_eq!(manager.device_count(), 1);
        assert!(manager.get_device(&device_id).is_some());

        let removed = manager.remove_device(&device_id);
        assert!(removed.is_some());
        assert_eq!(manager.device_count(), 0);
    }

    #[test]
    fn test_best_device_selection() {
        let manager = DeviceManager::new();
        manager.discover_devices().unwrap();

        let best_device = manager.get_best_device().unwrap();
        assert!(best_device.is_some());

        let available_devices = manager.get_available_devices().unwrap();
        assert!(!available_devices.is_empty());
    }

    #[test]
    fn test_device_health() {
        let mut health = DeviceHealth::new();
        assert!(health.is_healthy());
        assert_eq!(health.error_count(), 0);

        health.update_health(false);
        assert!(!health.is_healthy());
        assert_eq!(health.error_count(), 1);

        health.set_temperature(65.0);
        assert_eq!(health.temperature(), Some(65.0));
    }

    #[test]
    fn test_health_monitor() {
        let monitor = HealthMonitor::new();
        let cpu_device = Arc::new(CpuDevice::new()) as Arc<dyn Device>;

        monitor
            .add_device("cpu".to_string(), cpu_device.clone())
            .unwrap();

        let is_healthy = monitor.check_device_health(cpu_device.as_ref()).unwrap();
        assert!(is_healthy);

        monitor.remove_device("cpu");
    }

    #[test]
    fn test_manager_with_config() {
        let config = ManagerConfig {
            allocation_strategy: AllocationStrategy::Fastest,
            discovery_config: DiscoveryConfig {
                enable_cpu: true,
                enable_cuda: false,
                enable_metal: false,
                enable_wgpu: false,
                auto_discovery: true,
            },
            health_config: HealthConfig::default(),
        };

        let manager = DeviceManager::with_config(config);
        manager.discover_devices().unwrap();

        assert!(manager.device_count() >= 1); // At least CPU
    }

    #[test]
    fn test_utils_functions() {
        let manager = utils::create_manager_with_all_devices().unwrap();
        assert!(manager.device_count() > 0);

        let counts = utils::get_device_counts_by_type(&manager);
        assert!(counts.contains_key(&DeviceType::Cpu));

        let summary = utils::create_device_summary(&manager);
        assert!(!summary.is_empty());

        // CPU should always be available, so no GPU detection isn't guaranteed
        let _has_gpu = utils::has_gpu_devices(&manager);
    }

    #[test]
    fn test_global_manager() {
        let global = global_device_manager();
        assert!(global.device_count() > 0); // Should have auto-discovered devices
    }
}
