//! Memory pinning functionality for optimized GPU transfers
//!
//! This module provides abstractions and implementations for memory pinning,
//! which can significantly improve performance when transferring data between
//! CPU and GPU memory.

use torsh_core::{device::DeviceType, error::Result};

/// Trait for memory pinning operations
///
/// Memory pinning allocates page-locked memory that can be transferred to/from
/// GPU memory more efficiently than regular pageable memory. This is particularly
/// important for high-throughput data loading scenarios.
///
/// # Examples
///
/// ```no_run
/// use torsh_data::dataloader::memory::{MemoryPinning, CpuMemoryPinner};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let pinner = CpuMemoryPinner;
/// let data: Vec<i32> = vec![1, 2, 3, 4, 5];
/// let pinned_data = pinner.pin_memory(data)?;
///
/// // Check if pinning is supported for this type
/// println!("Supports pinning: {}", <CpuMemoryPinner as MemoryPinning<Vec<i32>>>::supports_pinning(&pinner));
/// # Ok(())
/// # }
/// ```
pub trait MemoryPinning<T> {
    /// Pin memory for GPU transfers
    ///
    /// # Arguments
    ///
    /// * `data` - The data to pin in memory
    ///
    /// # Returns
    ///
    /// The data with memory pinning applied (if supported)
    fn pin_memory(&self, data: T) -> Result<T>;

    /// Check if memory pinning is supported
    ///
    /// # Returns
    ///
    /// True if this implementation supports memory pinning, false otherwise
    fn supports_pinning(&self) -> bool;

    /// Get information about pinning capabilities
    ///
    /// # Returns
    ///
    /// String describing the pinning implementation
    fn pinning_info(&self) -> String {
        if self.supports_pinning() {
            "Memory pinning supported".to_string()
        } else {
            "Memory pinning not supported".to_string()
        }
    }
}

/// CPU memory pinning implementation (no-op for CPU)
///
/// This implementation provides a no-op memory pinning for CPU-only scenarios.
/// Since CPU memory doesn't require pinning for CPU operations, this simply
/// returns the data unchanged.
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_data::dataloader::memory::{CpuMemoryPinner, MemoryPinning};
///
/// let pinner = CpuMemoryPinner;
/// let data = vec![1, 2, 3, 4, 5];
/// let result = pinner.pin_memory(data).unwrap();
/// assert!(!pinner.supports_pinning());
/// ```
#[derive(Debug, Clone, Default)]
pub struct CpuMemoryPinner;

impl CpuMemoryPinner {
    /// Create a new CPU memory pinner
    pub fn new() -> Self {
        Self
    }
}

impl<T> MemoryPinning<T> for CpuMemoryPinner {
    fn pin_memory(&self, data: T) -> Result<T> {
        // CPU doesn't need pinning, return as-is
        Ok(data)
    }

    fn supports_pinning(&self) -> bool {
        false
    }

    fn pinning_info(&self) -> String {
        "CPU memory pinner (no-op implementation)".to_string()
    }
}

/// CUDA memory pinning implementation
///
/// This implementation provides memory pinning for CUDA GPU scenarios.
/// When enabled with the "cuda" feature, it can allocate page-locked memory
/// for faster transfers between CPU and GPU.
///
/// # Examples
///
/// ```rust,ignore
/// #[cfg(feature = "cuda")]
/// use torsh_data::dataloader::memory::{CudaMemoryPinner, MemoryPinning};
///
/// #[cfg(feature = "cuda")]
/// {
///     let pinner = CudaMemoryPinner::new(0)?;
///     assert!(pinner.supports_pinning());
/// }
/// ```
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CudaMemoryPinner {
    device_id: usize,
}

#[cfg(feature = "cuda")]
impl CudaMemoryPinner {
    /// Create a new CUDA memory pinner for the specified device
    ///
    /// # Arguments
    ///
    /// * `device_id` - The CUDA device ID to use for memory pinning
    ///
    /// # Returns
    ///
    /// A new CudaMemoryPinner or an error if the device is not available
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_data::dataloader::memory::CudaMemoryPinner;
    ///
    /// let pinner = CudaMemoryPinner::new(0)?;
    /// ```
    pub fn new(device_id: usize) -> Result<Self> {
        // Verify device availability
        #[cfg(feature = "cuda")]
        {
            // For now, assume CUDA is available if the feature is enabled
            // This should be replaced with proper torsh-backend integration
            // when the dependency is added
        }

        Ok(Self { device_id })
    }

    /// Get the device ID this pinner is configured for
    ///
    /// # Returns
    ///
    /// The CUDA device ID
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Set the device ID for this pinner
    ///
    /// # Arguments
    ///
    /// * `device_id` - The new CUDA device ID
    pub fn set_device_id(&mut self, device_id: usize) {
        self.device_id = device_id;
    }
}

#[cfg(feature = "cuda")]
impl<T> MemoryPinning<torsh_tensor::Tensor<T>> for CudaMemoryPinner
where
    T: torsh_core::dtype::TensorElement,
{
    fn pin_memory(&self, tensor: torsh_tensor::Tensor<T>) -> Result<torsh_tensor::Tensor<T>> {
        // For CUDA, we would:
        // 1. Check if tensor is on CPU
        // 2. Allocate page-locked (pinned) memory
        // 3. Copy data to pinned memory
        // 4. Return tensor backed by pinned memory

        // This is a simplified implementation - in practice would use CUDA APIs
        // to allocate page-locked memory for faster GPU transfers

        // For now, return the tensor as-is since the full CUDA implementation
        // would require more complex memory management
        Ok(tensor)
    }

    fn supports_pinning(&self) -> bool {
        true
    }

    fn pinning_info(&self) -> String {
        format!("CUDA memory pinner for device {}", self.device_id)
    }
}

/// Memory pinning manager that selects appropriate pinner based on device
///
/// This manager provides a unified interface for memory pinning across different
/// device types, automatically selecting the appropriate pinner implementation
/// based on the target device.
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_data::dataloader::memory::MemoryPinningManager;
/// use torsh_core::device::DeviceType;
///
/// let mut manager = MemoryPinningManager::new();
///
/// // Check if pinning is supported for CUDA
/// let cuda_supported = manager.supports_pinning(Some(DeviceType::Cuda(0)));
/// ```
#[derive(Debug)]
pub struct MemoryPinningManager {
    cpu_pinner: CpuMemoryPinner,
    #[cfg(feature = "cuda")]
    cuda_pinners: std::collections::HashMap<usize, CudaMemoryPinner>,
}

impl MemoryPinningManager {
    /// Create a new memory pinning manager
    ///
    /// # Returns
    ///
    /// A new MemoryPinningManager with default settings
    pub fn new() -> Self {
        Self {
            cpu_pinner: CpuMemoryPinner::new(),
            #[cfg(feature = "cuda")]
            cuda_pinners: std::collections::HashMap::new(),
        }
    }

    /// Pin memory for the appropriate device
    ///
    /// This method automatically selects the appropriate pinning implementation
    /// based on the target device type.
    ///
    /// # Arguments
    ///
    /// * `data` - The tensor data to pin
    /// * `target_device` - The target device for the data
    ///
    /// # Returns
    ///
    /// The tensor with memory pinning applied if supported
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_data::dataloader::memory::MemoryPinningManager;
    /// use torsh_core::device::DeviceType;
    /// use torsh_tensor::Tensor;
    ///
    /// let mut manager = MemoryPinningManager::new();
    /// let tensor = Tensor::zeros(&[2, 3]);
    /// let pinned = manager.pin_memory(tensor, Some(DeviceType::Cpu))?;
    /// ```
    pub fn pin_memory<T>(
        &mut self,
        data: torsh_tensor::Tensor<T>,
        target_device: Option<DeviceType>,
    ) -> Result<torsh_tensor::Tensor<T>>
    where
        T: torsh_core::dtype::TensorElement,
    {
        match target_device {
            Some(DeviceType::Cuda(device_id)) => {
                #[cfg(feature = "cuda")]
                {
                    if !self.cuda_pinners.contains_key(&device_id) {
                        let pinner = CudaMemoryPinner::new(device_id)?;
                        self.cuda_pinners.insert(device_id, pinner);
                    }

                    if let Some(pinner) = self.cuda_pinners.get(&device_id) {
                        pinner.pin_memory(data)
                    } else {
                        Ok(data)
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = device_id; // Suppress unused warning
                                       // CUDA not available, fall back to CPU
                    self.cpu_pinner.pin_memory(data)
                }
            }
            _ => {
                // CPU or other devices
                self.cpu_pinner.pin_memory(data)
            }
        }
    }

    /// Pin memory for vector data
    ///
    /// A convenience method for pinning vector data that doesn't require tensor operations.
    ///
    /// # Arguments
    ///
    /// * `data` - The vector data to pin
    /// * `target_device` - The target device for the data
    ///
    /// # Returns
    ///
    /// The vector with memory pinning applied if supported
    pub fn pin_vector_memory<T>(
        &mut self,
        data: Vec<T>,
        target_device: Option<DeviceType>,
    ) -> Result<Vec<T>> {
        match target_device {
            Some(DeviceType::Cuda(_device_id)) => {
                #[cfg(feature = "cuda")]
                {
                    // For vectors, we would typically convert to pinned memory
                    // This is a simplified implementation
                    Ok(data)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    // CUDA not available, return as-is
                    Ok(data)
                }
            }
            _ => {
                // CPU or other devices, no pinning needed
                Ok(data)
            }
        }
    }

    /// Check if pinning is supported for the target device
    ///
    /// # Arguments
    ///
    /// * `target_device` - The device to check pinning support for
    ///
    /// # Returns
    ///
    /// True if pinning is supported for the target device, false otherwise
    pub fn supports_pinning(&self, target_device: Option<DeviceType>) -> bool {
        match target_device {
            Some(DeviceType::Cuda(_)) => {
                #[cfg(feature = "cuda")]
                return true;
                #[cfg(not(feature = "cuda"))]
                return false;
            }
            _ => false,
        }
    }

    /// Get information about available pinning implementations
    ///
    /// # Returns
    ///
    /// String describing available pinning capabilities
    pub fn available_pinners(&self) -> String {
        // mut needed when cuda feature is enabled for info.push()
        #[allow(unused_mut)]
        let mut info = vec!["CPU (no-op)".to_string()];

        #[cfg(feature = "cuda")]
        {
            if !self.cuda_pinners.is_empty() {
                let devices: Vec<String> = self
                    .cuda_pinners
                    .keys()
                    .map(|id| format!("CUDA device {}", id))
                    .collect();
                info.push(format!("CUDA: {}", devices.join(", ")));
            } else {
                info.push("CUDA (available but no devices initialized)".to_string());
            }
        }

        format!("Available pinners: {}", info.join(", "))
    }

    /// Clear all cached CUDA pinners
    ///
    /// This can be useful for memory management or when device availability changes.
    #[cfg(feature = "cuda")]
    pub fn clear_cuda_pinners(&mut self) {
        self.cuda_pinners.clear();
    }

    /// Get the number of initialized CUDA pinners
    ///
    /// # Returns
    ///
    /// The number of CUDA pinners currently cached
    #[cfg(feature = "cuda")]
    pub fn cuda_pinner_count(&self) -> usize {
        self.cuda_pinners.len()
    }

    /// Pre-initialize a CUDA pinner for a specific device
    ///
    /// This can be useful for warming up the pinner before actual use.
    ///
    /// # Arguments
    ///
    /// * `device_id` - The CUDA device ID to initialize
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of initialization
    #[cfg(feature = "cuda")]
    pub fn initialize_cuda_pinner(&mut self, device_id: usize) -> Result<()> {
        if !self.cuda_pinners.contains_key(&device_id) {
            let pinner = CudaMemoryPinner::new(device_id)?;
            self.cuda_pinners.insert(device_id, pinner);
        }
        Ok(())
    }
}

impl Default for MemoryPinningManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for memory pinning operations
#[derive(Debug, Clone)]
pub struct PinningConfig {
    /// Whether to enable memory pinning
    pub enabled: bool,
    /// Target device for pinning
    pub target_device: Option<DeviceType>,
    /// Whether to pre-initialize pinners
    pub pre_initialize: bool,
}

impl Default for PinningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            target_device: None,
            pre_initialize: false,
        }
    }
}

impl PinningConfig {
    /// Create a new pinning configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable memory pinning
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set the target device for pinning
    pub fn target_device(mut self, device: DeviceType) -> Self {
        self.target_device = Some(device);
        self
    }

    /// Enable pre-initialization of pinners
    pub fn pre_initialize(mut self, pre_init: bool) -> Self {
        self.pre_initialize = pre_init;
        self
    }

    /// Create a configuration for CUDA pinning
    pub fn cuda(device_id: usize) -> Self {
        Self {
            enabled: true,
            target_device: Some(DeviceType::Cuda(device_id)),
            pre_initialize: false,
        }
    }

    /// Create a configuration for CPU (no-op) pinning
    pub fn cpu() -> Self {
        Self {
            enabled: false,
            target_device: Some(DeviceType::Cpu),
            pre_initialize: false,
        }
    }
}

/// Utility functions for memory pinning
pub mod utils {
    use super::*;

    /// Determine if memory pinning would be beneficial for the given configuration
    ///
    /// # Arguments
    ///
    /// * `source_device` - Device where data currently resides
    /// * `target_device` - Device where data will be used
    /// * `data_size` - Size of data in bytes
    ///
    /// # Returns
    ///
    /// True if pinning would likely improve performance
    pub fn should_pin_memory(
        source_device: DeviceType,
        target_device: DeviceType,
        data_size: usize,
    ) -> bool {
        match (source_device, target_device) {
            (DeviceType::Cpu, DeviceType::Cuda(_)) => {
                // CPU to GPU transfers benefit from pinning, especially for larger data
                data_size > 1024 // Pin for data larger than 1KB
            }
            (DeviceType::Cuda(_), DeviceType::Cpu) => {
                // GPU to CPU transfers can also benefit
                data_size > 1024
            }
            _ => false, // Same device or other combinations don't need pinning
        }
    }

    /// Estimate the memory overhead of pinning
    ///
    /// # Arguments
    ///
    /// * `data_size` - Size of data to be pinned in bytes
    ///
    /// # Returns
    ///
    /// Estimated additional memory usage in bytes
    pub fn estimate_pinning_overhead(data_size: usize) -> usize {
        // Pinned memory typically has minimal overhead
        // This is a conservative estimate
        data_size / 100 // 1% overhead estimate
    }

    /// Check if the system supports memory pinning for the given device
    ///
    /// # Arguments
    ///
    /// * `device` - Device to check support for
    ///
    /// # Returns
    ///
    /// True if the system supports pinning for this device type
    pub fn system_supports_pinning(device: DeviceType) -> bool {
        match device {
            DeviceType::Cuda(_) => {
                #[cfg(feature = "cuda")]
                return true;
                #[cfg(not(feature = "cuda"))]
                return false;
            }
            _ => false,
        }
    }

    /// Create an optimal pinning configuration for the given scenario
    ///
    /// # Arguments
    ///
    /// * `source_device` - Source device type
    /// * `target_device` - Target device type
    /// * `data_size` - Size of data to be transferred
    ///
    /// # Returns
    ///
    /// Optimal PinningConfig for the scenario
    pub fn optimal_pinning_config(
        source_device: DeviceType,
        target_device: DeviceType,
        data_size: usize,
    ) -> PinningConfig {
        if should_pin_memory(source_device, target_device, data_size) {
            PinningConfig::new()
                .enabled(true)
                .target_device(target_device)
                .pre_initialize(data_size > 1024 * 1024) // Pre-init for large transfers
        } else {
            PinningConfig::new().enabled(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_memory_pinner() {
        let pinner: CpuMemoryPinner = CpuMemoryPinner::new();
        let pinner_trait: &dyn MemoryPinning<Vec<i32>> = &pinner;
        assert!(!pinner_trait.supports_pinning());

        let data = vec![1, 2, 3, 4, 5];
        let result = pinner.pin_memory(data.clone()).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_memory_pinning_manager_creation() {
        let manager = MemoryPinningManager::new();
        assert!(!manager.supports_pinning(Some(DeviceType::Cpu)));
    }

    #[test]
    fn test_memory_pinning_manager_cpu() {
        let mut manager = MemoryPinningManager::new();
        let data = vec![1, 2, 3, 4, 5];
        let result = manager
            .pin_vector_memory(data.clone(), Some(DeviceType::Cpu))
            .unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_pinning_config() {
        let config = PinningConfig::new()
            .enabled(true)
            .target_device(DeviceType::Cpu)
            .pre_initialize(true);

        assert!(config.enabled);
        assert_eq!(config.target_device, Some(DeviceType::Cpu));
        assert!(config.pre_initialize);
    }

    #[test]
    fn test_pinning_config_cuda() {
        let config = PinningConfig::cuda(0);
        assert!(config.enabled);
        assert_eq!(config.target_device, Some(DeviceType::Cuda(0)));
        assert!(!config.pre_initialize);
    }

    #[test]
    fn test_pinning_config_cpu() {
        let config = PinningConfig::cpu();
        assert!(!config.enabled);
        assert_eq!(config.target_device, Some(DeviceType::Cpu));
        assert!(!config.pre_initialize);
    }

    #[test]
    fn test_should_pin_memory() {
        // CPU to CUDA should pin for large data
        assert!(utils::should_pin_memory(
            DeviceType::Cpu,
            DeviceType::Cuda(0),
            2048
        ));

        // Small data shouldn't be pinned
        assert!(!utils::should_pin_memory(
            DeviceType::Cpu,
            DeviceType::Cuda(0),
            512
        ));

        // Same device shouldn't pin
        assert!(!utils::should_pin_memory(
            DeviceType::Cpu,
            DeviceType::Cpu,
            2048
        ));
    }

    #[test]
    fn test_estimate_pinning_overhead() {
        let data_size = 1000;
        let overhead = utils::estimate_pinning_overhead(data_size);
        assert_eq!(overhead, 10); // 1% of 1000
    }

    #[test]
    fn test_system_supports_pinning() {
        // CPU should not support pinning
        assert!(!utils::system_supports_pinning(DeviceType::Cpu));

        // CUDA support depends on feature flag
        #[cfg(feature = "cuda")]
        assert!(utils::system_supports_pinning(DeviceType::Cuda(0)));

        #[cfg(not(feature = "cuda"))]
        assert!(!utils::system_supports_pinning(DeviceType::Cuda(0)));
    }

    #[test]
    fn test_optimal_pinning_config() {
        // Large CPU to CUDA transfer should enable pinning
        let config = utils::optimal_pinning_config(DeviceType::Cpu, DeviceType::Cuda(0), 2048);
        assert!(config.enabled);
        assert_eq!(config.target_device, Some(DeviceType::Cuda(0)));

        // Small transfer should not enable pinning
        let config = utils::optimal_pinning_config(DeviceType::Cpu, DeviceType::Cuda(0), 512);
        assert!(!config.enabled);

        // Very large transfer should enable pre-initialization
        let config =
            utils::optimal_pinning_config(DeviceType::Cpu, DeviceType::Cuda(0), 2 * 1024 * 1024);
        assert!(config.enabled);
        assert!(config.pre_initialize);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_memory_pinner() {
        let pinner = CudaMemoryPinner::new(0).unwrap();
        let pinner_trait: &dyn MemoryPinning<torsh_tensor::Tensor<f32>> = &pinner;
        assert!(pinner_trait.supports_pinning());
        assert_eq!(pinner.device_id(), 0);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_memory_pinning_manager_cuda() {
        let mut manager = MemoryPinningManager::new();
        assert!(manager.supports_pinning(Some(DeviceType::Cuda(0))));

        // Test initialization
        manager.initialize_cuda_pinner(0).unwrap();
        assert_eq!(manager.cuda_pinner_count(), 1);

        // Test clearing
        manager.clear_cuda_pinners();
        assert_eq!(manager.cuda_pinner_count(), 0);
    }
}
