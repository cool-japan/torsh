//! Zero-copy memory transfer implementations for ToRSh backends
//!
//! This module provides efficient zero-copy host-device and device-device transfers
//! where supported by the underlying hardware and drivers.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::error::{BackendError, BackendResult};
use crate::{Device, MemoryManager};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use torsh_core::device::DeviceType;

#[cfg(feature = "cuda")]
use crate::cuda::CudaDevice as SciRs2CudaDevice;

// Temporary mock for scirs2_cuda when CUDA is not available
#[cfg(all(feature = "cuda", not(cuda_available)))]
mod scirs2_cuda {
    // Mock CUDA device type for fallback scenarios
    #[derive(Debug)]
    pub struct MockCudaDevice {
        id: usize,
    }

    pub mod memory {
        pub enum MemoryAdvice {
            SetPreferredLocation(u32),
            SetAccessedBy(u32),
            SetReadMostly,
            UnsetReadMostly,
        }

        pub async fn prefetch_async(
            _device: &crate::cuda::CudaDevice,
            _ptr: *const u8,
            _size: usize,
        ) -> Result<(), String> {
            Err("CUDA not available".to_string())
        }

        pub async fn set_advice(
            _ptr: *const u8,
            _size: usize,
            _advice: MemoryAdvice,
        ) -> Result<(), String> {
            Err("CUDA not available".to_string())
        }

        pub async fn copy_peer_to_peer(
            _src: *const u8,
            _dst: *mut u8,
            _size: usize,
        ) -> Result<(), String> {
            Err("CUDA not available".to_string())
        }

        pub async fn copy_host_to_device_async(
            _src: *const u8,
            _dst: *mut u8,
            _size: usize,
        ) -> Result<(), String> {
            Err("CUDA not available".to_string())
        }

        pub async fn copy_device_to_host_async(
            _src: *const u8,
            _dst: *mut u8,
            _size: usize,
        ) -> Result<(), String> {
            Err("CUDA not available".to_string())
        }

        pub fn copy_host_to_device(
            _src: *const u8,
            _dst: *mut u8,
            _size: usize,
        ) -> Result<(), String> {
            Err("CUDA not available".to_string())
        }

        pub fn copy_device_to_host(
            _src: *const u8,
            _dst: *mut u8,
            _size: usize,
        ) -> Result<(), String> {
            Err("CUDA not available".to_string())
        }
    }

    // Add missing synchronize function for CUDA - this is at the scirs2_cuda module level
    pub fn synchronize(_device: &crate::cuda::CudaDevice) -> Result<(), String> {
        Err("CUDA not available".to_string())
    }
}

#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
use crate::metal::MetalDevice as SciRs2MetalDevice;

// Temporary mock for scirs2_metal since scirs2_core doesn't have a metal module yet
#[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
mod scirs2_metal {
    pub mod memory {
        use crate::metal::device::MetalDevice;

        pub enum CpuCacheMode {
            WriteCombined,
        }

        pub fn set_cpu_cache_mode(
            _device: &MetalDevice,
            _ptr: *mut u8,
            _mode: CpuCacheMode,
        ) -> Result<(), String> {
            // Mock implementation - in real implementation would set Metal cache mode
            Ok(())
        }

        pub async fn copy_host_to_device_async(
            _device: &MetalDevice,
            _src_ptr: *const u8,
            _dst_ptr: *mut u8,
            _size: usize,
        ) -> Result<(), String> {
            // Mock implementation - in real implementation would use Metal async copy
            Ok(())
        }

        pub async fn copy_device_to_host_async(
            _device: &MetalDevice,
            _src_ptr: *const u8,
            _dst_ptr: *mut u8,
            _size: usize,
        ) -> Result<(), String> {
            // Mock implementation - in real implementation would use Metal async copy
            Ok(())
        }

        pub fn copy_host_to_device(
            _device: &MetalDevice,
            _src_ptr: *const u8,
            _dst_ptr: *mut u8,
            _size: usize,
        ) -> Result<(), String> {
            // Mock implementation - in real implementation would use Metal sync copy
            Ok(())
        }

        pub fn copy_device_to_host(
            _device: &MetalDevice,
            _src_ptr: *const u8,
            _dst_ptr: *mut u8,
            _size: usize,
        ) -> Result<(), String> {
            // Mock implementation - in real implementation would use Metal sync copy
            Ok(())
        }
    }

    pub fn synchronize(_device: &crate::metal::device::MetalDevice) -> Result<(), String> {
        // Mock implementation - in real implementation would synchronize Metal commands
        Ok(())
    }
}

/// Transfer mode for zero-copy operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferMode {
    /// Synchronous transfer (blocking)
    Synchronous,
    /// Asynchronous transfer (non-blocking)
    Asynchronous,
    /// Streaming transfer (for large data)
    Streaming,
    /// Peer-to-peer direct transfer
    PeerToPeer,
}

/// Transfer direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// Host to device transfer
    HostToDevice,
    /// Device to host transfer
    DeviceToHost,
    /// Device to device transfer (same device)
    DeviceToDevice,
    /// Cross-device transfer (different devices)
    CrossDevice,
}

/// Zero-copy capability flags
#[derive(Debug, Clone, Copy)]
pub struct ZeroCopyCapabilities {
    /// Supports unified memory (host-device accessible)
    pub unified_memory: bool,
    /// Supports peer-to-peer access
    pub peer_to_peer: bool,
    /// Supports memory mapping
    pub memory_mapping: bool,
    /// Supports direct GPU access
    pub direct_gpu_access: bool,
    /// Supports pinned host memory
    pub pinned_memory: bool,
    /// Supports memory advice hints
    pub memory_advice: bool,
    /// Supports asynchronous transfers
    pub async_transfers: bool,
    /// Supports streaming transfers
    pub streaming_transfers: bool,
}

impl Default for ZeroCopyCapabilities {
    fn default() -> Self {
        Self {
            unified_memory: false,
            peer_to_peer: false,
            memory_mapping: false,
            direct_gpu_access: false,
            pinned_memory: false,
            memory_advice: false,
            async_transfers: false,
            streaming_transfers: false,
        }
    }
}

impl ZeroCopyCapabilities {
    /// Check if any zero-copy features are supported
    pub fn has_any_capabilities(&self) -> bool {
        self.unified_memory
            || self.peer_to_peer
            || self.memory_mapping
            || self.direct_gpu_access
            || self.pinned_memory
            || self.async_transfers
    }

    /// Get capabilities score (0.0 to 1.0)
    pub fn capability_score(&self) -> f32 {
        let mut score = 0.0;
        let total_features = 8.0;

        if self.unified_memory {
            score += 1.0;
        }
        if self.peer_to_peer {
            score += 1.0;
        }
        if self.memory_mapping {
            score += 1.0;
        }
        if self.direct_gpu_access {
            score += 1.0;
        }
        if self.pinned_memory {
            score += 1.0;
        }
        if self.memory_advice {
            score += 1.0;
        }
        if self.async_transfers {
            score += 1.0;
        }
        if self.streaming_transfers {
            score += 1.0;
        }

        score / total_features
    }

    /// Get recommended transfer mode for given capabilities
    pub fn recommended_transfer_mode(&self) -> TransferMode {
        if self.streaming_transfers {
            TransferMode::Streaming
        } else if self.async_transfers {
            TransferMode::Asynchronous
        } else if self.peer_to_peer {
            TransferMode::PeerToPeer
        } else {
            TransferMode::Synchronous
        }
    }
}

/// Zero-copy transfer descriptor
#[derive(Debug, Clone)]
pub struct ZeroCopyTransfer {
    /// Source device
    pub source_device: Device,
    /// Destination device
    pub destination_device: Device,
    /// Transfer direction
    pub direction: TransferDirection,
    /// Transfer mode
    pub mode: TransferMode,
    /// Source memory pointer
    pub source_ptr: *mut u8,
    /// Destination memory pointer
    pub destination_ptr: *mut u8,
    /// Size in bytes
    pub size: usize,
    /// Memory alignment requirement
    pub alignment: usize,
    /// Priority level (0 = highest, higher numbers = lower priority)
    pub priority: u32,
    /// Optional stream/queue for asynchronous operations
    pub stream_id: Option<u64>,
}

unsafe impl Send for ZeroCopyTransfer {}
unsafe impl Sync for ZeroCopyTransfer {}

impl ZeroCopyTransfer {
    /// Create a new zero-copy transfer descriptor
    pub fn new(
        source_device: Device,
        destination_device: Device,
        source_ptr: *mut u8,
        destination_ptr: *mut u8,
        size: usize,
    ) -> Self {
        let direction = if source_device.device_type() == DeviceType::Cpu
            && destination_device.device_type() != DeviceType::Cpu
        {
            TransferDirection::HostToDevice
        } else if source_device.device_type() != DeviceType::Cpu
            && destination_device.device_type() == DeviceType::Cpu
        {
            TransferDirection::DeviceToHost
        } else if source_device.id() == destination_device.id() {
            TransferDirection::DeviceToDevice
        } else {
            TransferDirection::CrossDevice
        };

        Self {
            source_device,
            destination_device,
            direction,
            mode: TransferMode::Synchronous,
            source_ptr,
            destination_ptr,
            size,
            alignment: 1,
            priority: 1,
            stream_id: None,
        }
    }

    /// Set transfer mode
    pub fn with_mode(mut self, mode: TransferMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set alignment requirement
    pub fn with_alignment(mut self, alignment: usize) -> Self {
        self.alignment = alignment;
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Set stream ID for asynchronous operations
    pub fn with_stream(mut self, stream_id: u64) -> Self {
        self.stream_id = Some(stream_id);
        self
    }

    /// Check if transfer can be zero-copy
    pub fn is_zero_copy_possible(&self, capabilities: &ZeroCopyCapabilities) -> bool {
        match self.direction {
            TransferDirection::HostToDevice | TransferDirection::DeviceToHost => {
                capabilities.unified_memory || capabilities.pinned_memory
            }
            TransferDirection::DeviceToDevice => capabilities.memory_mapping,
            TransferDirection::CrossDevice => capabilities.peer_to_peer,
        }
    }

    /// Estimate transfer bandwidth (bytes per second)
    pub fn estimate_bandwidth(&self, device_type: DeviceType) -> u64 {
        match (device_type, self.direction) {
            (DeviceType::Cuda(_), TransferDirection::HostToDevice) => {
                if self.alignment >= 256 {
                    25_000_000_000 // 25 GB/s for well-aligned transfers
                } else {
                    12_000_000_000 // 12 GB/s for unaligned
                }
            }
            (DeviceType::Cuda(_), TransferDirection::DeviceToHost) => {
                if self.alignment >= 256 {
                    20_000_000_000 // 20 GB/s
                } else {
                    10_000_000_000 // 10 GB/s
                }
            }
            (DeviceType::Cuda(_), TransferDirection::CrossDevice) => 50_000_000_000, // 50 GB/s NVLink
            (DeviceType::Metal(_), TransferDirection::HostToDevice) => 40_000_000_000, // 40 GB/s unified memory
            (DeviceType::Metal(_), TransferDirection::DeviceToHost) => 40_000_000_000,
            (DeviceType::Wgpu(_), TransferDirection::HostToDevice) => 8_000_000_000, // 8 GB/s
            (DeviceType::Wgpu(_), TransferDirection::DeviceToHost) => 6_000_000_000, // 6 GB/s
            (DeviceType::Cpu, _) => 50_000_000_000, // 50 GB/s DDR4/5
            _ => 1_000_000_000,                     // 1 GB/s fallback
        }
    }

    /// Estimate transfer time in microseconds
    pub fn estimate_transfer_time_us(&self, device_type: DeviceType) -> u64 {
        let bandwidth = self.estimate_bandwidth(device_type);
        if bandwidth == 0 {
            u64::MAX
        } else {
            (self.size as u64 * 1_000_000) / bandwidth
        }
    }
}

/// Zero-copy transfer statistics
#[derive(Debug, Default, Clone)]
pub struct ZeroCopyStats {
    /// Total number of transfers attempted
    pub total_transfers: u64,
    /// Number of successful zero-copy transfers
    pub zero_copy_transfers: u64,
    /// Number of fallback copies
    pub fallback_transfers: u64,
    /// Total bytes transferred via zero-copy
    pub zero_copy_bytes: u64,
    /// Total bytes transferred via fallback
    pub fallback_bytes: u64,
    /// Total transfer time in microseconds
    pub total_transfer_time_us: u64,
    /// Average transfer bandwidth in bytes per second
    pub average_bandwidth: f64,
    /// Number of transfer errors
    pub error_count: u64,
}

impl ZeroCopyStats {
    /// Calculate zero-copy success rate
    pub fn zero_copy_success_rate(&self) -> f64 {
        if self.total_transfers == 0 {
            0.0
        } else {
            (self.zero_copy_transfers as f64) / (self.total_transfers as f64)
        }
    }

    /// Calculate bandwidth efficiency (actual vs theoretical)
    pub fn bandwidth_efficiency(&self, theoretical_bandwidth: u64) -> f64 {
        if theoretical_bandwidth == 0 {
            0.0
        } else {
            self.average_bandwidth / (theoretical_bandwidth as f64)
        }
    }

    /// Calculate error rate
    pub fn error_rate(&self) -> f64 {
        if self.total_transfers == 0 {
            0.0
        } else {
            (self.error_count as f64) / (self.total_transfers as f64)
        }
    }

    /// Update statistics with a new transfer
    pub fn update_transfer(
        &mut self,
        bytes: u64,
        time_us: u64,
        was_zero_copy: bool,
        was_error: bool,
    ) {
        self.total_transfers += 1;

        if was_error {
            self.error_count += 1;
            return;
        }

        if was_zero_copy {
            self.zero_copy_transfers += 1;
            self.zero_copy_bytes += bytes;
        } else {
            self.fallback_transfers += 1;
            self.fallback_bytes += bytes;
        }

        self.total_transfer_time_us += time_us;

        // Update average bandwidth
        let total_bytes = self.zero_copy_bytes + self.fallback_bytes;
        if self.total_transfer_time_us > 0 {
            self.average_bandwidth =
                (total_bytes as f64) / (self.total_transfer_time_us as f64 / 1_000_000.0);
        }
    }
}

/// Zero-copy transfer manager
pub struct ZeroCopyManager {
    /// Device capabilities cache
    capabilities: Arc<RwLock<HashMap<String, ZeroCopyCapabilities>>>,
    /// Transfer statistics
    stats: Arc<RwLock<ZeroCopyStats>>,
    /// Memory managers for each device
    memory_managers: HashMap<String, Arc<dyn MemoryManager>>,
    /// SciRS2 CUDA devices for actual zero-copy operations
    #[cfg(feature = "cuda")]
    cuda_devices: HashMap<String, Arc<SciRs2CudaDevice>>,
    /// SciRS2 Metal devices for actual zero-copy operations
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    metal_devices: HashMap<String, Arc<SciRs2MetalDevice>>,
}

impl ZeroCopyManager {
    /// Create a new zero-copy transfer manager
    pub fn new() -> Self {
        Self {
            capabilities: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(ZeroCopyStats::default())),
            memory_managers: HashMap::new(),
            #[cfg(feature = "cuda")]
            cuda_devices: HashMap::new(),
            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            metal_devices: HashMap::new(),
        }
    }

    /// Register a device with its capabilities
    pub fn register_device(
        &mut self,
        device: &Device,
        capabilities: ZeroCopyCapabilities,
        memory_manager: Arc<dyn MemoryManager>,
    ) -> BackendResult<()> {
        let device_key = format!("{}:{}", device.device_type(), device.id());

        {
            let mut caps = self.capabilities.write().unwrap();
            caps.insert(device_key.clone(), capabilities);
        }

        self.memory_managers.insert(device_key, memory_manager);
        Ok(())
    }

    /// Register a CUDA device for zero-copy operations
    #[cfg(feature = "cuda")]
    pub fn register_cuda_device(
        &mut self,
        device: &Device,
        scirs2_device: Arc<SciRs2CudaDevice>,
        capabilities: ZeroCopyCapabilities,
        memory_manager: Arc<dyn MemoryManager>,
    ) -> BackendResult<()> {
        let device_key = format!("{}:{}", device.device_type(), device.id());

        // Register device capabilities and memory manager
        self.register_device(device, capabilities, memory_manager)?;

        // Register SciRS2 device for actual operations
        self.cuda_devices.insert(device_key, scirs2_device);
        Ok(())
    }

    /// Register a Metal device for zero-copy operations
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    pub fn register_metal_device(
        &mut self,
        device: &Device,
        scirs2_device: Arc<SciRs2MetalDevice>,
        capabilities: ZeroCopyCapabilities,
        memory_manager: Arc<dyn MemoryManager>,
    ) -> BackendResult<()> {
        let device_key = format!("{}:{}", device.device_type(), device.id());

        // Register device capabilities and memory manager
        self.register_device(device, capabilities, memory_manager)?;

        // Register SciRS2 device for actual operations
        self.metal_devices.insert(device_key, scirs2_device);
        Ok(())
    }

    /// Get device capabilities
    pub fn get_capabilities(&self, device: &Device) -> Option<ZeroCopyCapabilities> {
        let device_key = format!("{}:{}", device.device_type(), device.id());
        let caps = self.capabilities.read().unwrap();
        caps.get(&device_key).copied()
    }

    /// Check if zero-copy transfer is possible between devices
    pub fn can_zero_copy(&self, source: &Device, destination: &Device) -> bool {
        let source_caps = self.get_capabilities(source);
        let dest_caps = self.get_capabilities(destination);

        match (source_caps, dest_caps) {
            (Some(src), Some(dst)) => {
                // Check specific transfer compatibility
                if source.id() == destination.id() {
                    // Same device - check memory mapping
                    src.memory_mapping && dst.memory_mapping
                } else if source.device_type() == DeviceType::Cpu {
                    // Host to device - check unified memory or pinned memory
                    dst.unified_memory || dst.pinned_memory
                } else if destination.device_type() == DeviceType::Cpu {
                    // Device to host - check unified memory or pinned memory
                    src.unified_memory || src.pinned_memory
                } else {
                    // Device to device - check peer-to-peer
                    src.peer_to_peer && dst.peer_to_peer
                }
            }
            _ => false,
        }
    }

    /// Perform zero-copy transfer
    pub async fn transfer(&mut self, transfer: &ZeroCopyTransfer) -> BackendResult<bool> {
        let start_time = std::time::Instant::now();

        // Check if zero-copy is possible
        if !self.can_zero_copy(&transfer.source_device, &transfer.destination_device) {
            return self.fallback_transfer(transfer, start_time).await;
        }

        // Attempt zero-copy transfer based on direction and capabilities
        let result = match transfer.direction {
            TransferDirection::HostToDevice => self.host_to_device_transfer(transfer).await,
            TransferDirection::DeviceToHost => self.device_to_host_transfer(transfer).await,
            TransferDirection::DeviceToDevice => self.device_to_device_transfer(transfer).await,
            TransferDirection::CrossDevice => self.cross_device_transfer(transfer).await,
        };

        let elapsed_us = start_time.elapsed().as_micros() as u64;
        let was_zero_copy = result.is_ok() && result.as_ref().unwrap_or(&false) == &true;
        let was_error = result.is_err();

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.update_transfer(transfer.size as u64, elapsed_us, was_zero_copy, was_error);
        }

        result
    }

    /// Host to device zero-copy transfer
    async fn host_to_device_transfer(&self, transfer: &ZeroCopyTransfer) -> BackendResult<bool> {
        let dest_caps = self
            .get_capabilities(&transfer.destination_device)
            .ok_or_else(|| {
                BackendError::BackendError("Destination device not registered".to_string())
            })?;

        if dest_caps.unified_memory {
            // Use unified memory - data is already accessible by device
            self.unified_memory_transfer(transfer).await
        } else if dest_caps.pinned_memory {
            // Use pinned host memory with DMA transfer
            self.pinned_memory_transfer(transfer).await
        } else {
            Err(BackendError::BackendError(
                "No zero-copy method available for host to device transfer".to_string(),
            ))
        }
    }

    /// Device to host zero-copy transfer
    async fn device_to_host_transfer(&self, transfer: &ZeroCopyTransfer) -> BackendResult<bool> {
        let source_caps = self
            .get_capabilities(&transfer.source_device)
            .ok_or_else(|| {
                BackendError::BackendError("Source device not registered".to_string())
            })?;

        if source_caps.unified_memory {
            // Use unified memory - data is already accessible by host
            self.unified_memory_transfer(transfer).await
        } else if source_caps.pinned_memory {
            // Use pinned host memory with DMA transfer
            self.pinned_memory_transfer(transfer).await
        } else {
            Err(BackendError::BackendError(
                "No zero-copy method available for device to host transfer".to_string(),
            ))
        }
    }

    /// Device to device zero-copy transfer (same device)
    async fn device_to_device_transfer(&self, transfer: &ZeroCopyTransfer) -> BackendResult<bool> {
        let device_caps = self
            .get_capabilities(&transfer.source_device)
            .ok_or_else(|| BackendError::BackendError("Device not registered".to_string()))?;

        if device_caps.memory_mapping {
            // Use memory mapping for device-local transfer
            self.memory_mapped_transfer(transfer).await
        } else {
            Err(BackendError::BackendError(
                "No zero-copy method available for device to device transfer".to_string(),
            ))
        }
    }

    /// Cross-device zero-copy transfer
    #[allow(unused_unsafe)]
    async fn cross_device_transfer(&self, transfer: &ZeroCopyTransfer) -> BackendResult<bool> {
        let source_caps = self
            .get_capabilities(&transfer.source_device)
            .ok_or_else(|| {
                BackendError::BackendError("Source device not registered".to_string())
            })?;
        let dest_caps = self
            .get_capabilities(&transfer.destination_device)
            .ok_or_else(|| {
                BackendError::BackendError("Destination device not registered".to_string())
            })?;

        if source_caps.peer_to_peer && dest_caps.peer_to_peer {
            // Use peer-to-peer transfer (e.g., NVLink, PCIe P2P)
            self.peer_to_peer_transfer(transfer).await
        } else {
            Err(BackendError::BackendError(
                "No zero-copy method available for cross-device transfer".to_string(),
            ))
        }
    }

    /// Unified memory transfer implementation
    async fn unified_memory_transfer(&self, transfer: &ZeroCopyTransfer) -> BackendResult<bool> {
        let device_key = format!(
            "{}:{}",
            transfer.destination_device.device_type(),
            transfer.destination_device.id()
        );

        match transfer.destination_device.device_type() {
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                if let Some(cuda_device) = self.cuda_devices.get(&device_key) {
                    // Use SciRS2 CUDA unified memory
                    #[allow(unused_unsafe)]
                    unsafe {
                        scirs2_cuda::memory::prefetch_async(
                            cuda_device.as_ref(),
                            transfer.source_ptr,
                            transfer.size,
                        )
                        .await
                        .map_err(|e| {
                            BackendError::BackendError(format!(
                                "CUDA unified memory prefetch failed: {}",
                                e
                            ))
                        })?;

                        // Set memory advice for optimal access
                        scirs2_cuda::memory::set_advice(
                            transfer.source_ptr,
                            transfer.size,
                            scirs2_cuda::memory::MemoryAdvice::SetPreferredLocation(
                                transfer.destination_device.id() as u32,
                            ),
                        )
                        .await
                        .map_err(|e| {
                            BackendError::BackendError(format!("CUDA memory advice failed: {}", e))
                        })?;
                    }
                    Ok(true)
                } else {
                    Err(BackendError::BackendError(
                        "CUDA device not registered for unified memory".to_string(),
                    ))
                }
            }

            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            DeviceType::Metal(_) => {
                if let Some(metal_device) = self.metal_devices.get(&device_key) {
                    // Use SciRS2 Metal unified memory
                    #[allow(unused_unsafe)]
                    unsafe {
                        scirs2_metal::memory::set_cpu_cache_mode(
                            metal_device,
                            transfer.source_ptr,
                            scirs2_metal::memory::CpuCacheMode::WriteCombined,
                        )
                        .map_err(|e| {
                            BackendError::BackendError(format!("Metal cache mode failed: {}", e))
                        })?;

                        // For Metal, unified memory is automatically managed by the system
                        // No explicit prefetch needed
                    }
                    Ok(true)
                } else {
                    Err(BackendError::BackendError(
                        "Metal device not registered for unified memory".to_string(),
                    ))
                }
            }

            _ => {
                // Fallback to memory manager for other device types
                if let Some(memory_manager) = self.memory_managers.get(&device_key) {
                    let _ = memory_manager.set_memory_advice(
                        transfer.source_ptr,
                        transfer.size,
                        crate::memory::MemoryAdvice::SetPreferredLocation,
                    );
                    let _ = memory_manager.prefetch_to_device(transfer.source_ptr, transfer.size);
                    Ok(true)
                } else {
                    Err(BackendError::BackendError(
                        "Memory manager not found for device".to_string(),
                    ))
                }
            }
        }
    }

    /// Pinned memory transfer implementation
    async fn pinned_memory_transfer(&self, transfer: &ZeroCopyTransfer) -> BackendResult<bool> {
        #[allow(unused_variables)]
        let device_key = format!(
            "{}:{}",
            transfer.destination_device.device_type(),
            transfer.destination_device.id()
        );

        match transfer.destination_device.device_type() {
            #[cfg(feature = "cuda")]
            DeviceType::Cuda(_) => {
                if let Some(cuda_device) = self.cuda_devices.get(&device_key) {
                    match transfer.mode {
                        TransferMode::Asynchronous => {
                            self.launch_cuda_async_transfer(cuda_device, transfer).await
                        }
                        TransferMode::Streaming => {
                            self.launch_cuda_streaming_transfer(cuda_device, transfer)
                                .await
                        }
                        _ => self.launch_cuda_sync_transfer(cuda_device, transfer).await,
                    }
                } else {
                    Err(BackendError::BackendError(
                        "CUDA device not registered for pinned memory".to_string(),
                    ))
                }
            }

            #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
            DeviceType::Metal(_) => {
                if let Some(metal_device) = self.metal_devices.get(&device_key) {
                    match transfer.mode {
                        TransferMode::Asynchronous => {
                            self.launch_metal_async_transfer(metal_device, transfer)
                                .await
                        }
                        TransferMode::Streaming => {
                            self.launch_metal_streaming_transfer(metal_device, transfer)
                                .await
                        }
                        _ => {
                            self.launch_metal_sync_transfer(metal_device, transfer)
                                .await
                        }
                    }
                } else {
                    Err(BackendError::BackendError(
                        "Metal device not registered for pinned memory".to_string(),
                    ))
                }
            }

            _ => {
                // Fallback to generic implementation
                match transfer.mode {
                    TransferMode::Asynchronous => self.launch_async_dma(transfer).await,
                    TransferMode::Streaming => self.launch_streaming_transfer(transfer).await,
                    _ => self.launch_sync_dma(transfer).await,
                }
            }
        }
    }

    /// Memory mapped transfer implementation
    async fn memory_mapped_transfer(&self, transfer: &ZeroCopyTransfer) -> BackendResult<bool> {
        // For memory mapped transfers, we can use device-local copy operations
        // This avoids host involvement

        if transfer.source_ptr.is_null() || transfer.destination_ptr.is_null() {
            return Err(BackendError::InvalidArgument(
                "Null pointer in memory mapped transfer".to_string(),
            ));
        }

        // Use device-optimized memory copy (e.g., GPU kernels)
        self.launch_device_copy(transfer).await
    }

    /// Peer-to-peer transfer implementation
    async fn peer_to_peer_transfer(&self, transfer: &ZeroCopyTransfer) -> BackendResult<bool> {
        // For peer-to-peer transfers, devices can directly access each other's memory
        // This is particularly efficient with technologies like NVLink

        let source_caps = self.get_capabilities(&transfer.source_device).unwrap();
        let dest_caps = self.get_capabilities(&transfer.destination_device).unwrap();

        if !source_caps.peer_to_peer || !dest_caps.peer_to_peer {
            return Err(BackendError::BackendError(
                "Peer-to-peer not supported on one or both devices".to_string(),
            ));
        }

        // Launch peer-to-peer DMA transfer
        self.launch_p2p_transfer(transfer).await
    }

    /// Fallback transfer using conventional copying
    async fn fallback_transfer(
        &mut self,
        transfer: &ZeroCopyTransfer,
        start_time: std::time::Instant,
    ) -> BackendResult<bool> {
        // Perform conventional memory copy
        if transfer.source_ptr.is_null() || transfer.destination_ptr.is_null() {
            return Err(BackendError::InvalidArgument(
                "Null pointer in fallback transfer".to_string(),
            ));
        }

        // Safety: This is unsafe as it involves raw pointer operations
        // In a real implementation, this would use proper device APIs
        unsafe {
            std::ptr::copy_nonoverlapping(
                transfer.source_ptr,
                transfer.destination_ptr,
                transfer.size,
            );
        }

        let elapsed_us = start_time.elapsed().as_micros() as u64;

        // Update statistics for fallback transfer
        {
            let mut stats = self.stats.write().unwrap();
            stats.update_transfer(transfer.size as u64, elapsed_us, false, false);
        }

        Ok(false) // Not zero-copy
    }

    /// Launch asynchronous DMA transfer
    async fn launch_async_dma(&self, transfer: &ZeroCopyTransfer) -> BackendResult<bool> {
        // Implementation would use device-specific async DMA APIs
        // For now, simulate async operation
        #[cfg(feature = "async")]
        tokio::task::yield_now().await;

        // Simulate DMA copy
        if !transfer.source_ptr.is_null() && !transfer.destination_ptr.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    transfer.source_ptr,
                    transfer.destination_ptr,
                    transfer.size,
                );
            }
        }

        Ok(true)
    }

    /// Launch streaming transfer
    async fn launch_streaming_transfer(&self, transfer: &ZeroCopyTransfer) -> BackendResult<bool> {
        // For streaming transfers, break large transfers into smaller chunks
        const CHUNK_SIZE: usize = 64 * 1024 * 1024; // 64MB chunks

        let num_chunks = (transfer.size + CHUNK_SIZE - 1) / CHUNK_SIZE;

        for chunk in 0..num_chunks {
            let chunk_offset = chunk * CHUNK_SIZE;
            let chunk_size = std::cmp::min(CHUNK_SIZE, transfer.size - chunk_offset);

            if chunk_size == 0 {
                break;
            }

            // Create chunk transfer
            let chunk_transfer = ZeroCopyTransfer {
                source_ptr: unsafe { transfer.source_ptr.add(chunk_offset) },
                destination_ptr: unsafe { transfer.destination_ptr.add(chunk_offset) },
                size: chunk_size,
                mode: TransferMode::Asynchronous,
                ..transfer.clone()
            };

            // Launch chunk transfer
            self.launch_async_dma(&chunk_transfer).await?;

            // Yield between chunks
            #[cfg(feature = "async")]
            tokio::task::yield_now().await;
        }

        Ok(true)
    }

    /// Launch synchronous DMA transfer
    async fn launch_sync_dma(&self, transfer: &ZeroCopyTransfer) -> BackendResult<bool> {
        // Implementation would use device-specific sync DMA APIs
        if !transfer.source_ptr.is_null() && !transfer.destination_ptr.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    transfer.source_ptr,
                    transfer.destination_ptr,
                    transfer.size,
                );
            }
            Ok(true)
        } else {
            Err(BackendError::InvalidArgument(
                "Null pointer in sync DMA transfer".to_string(),
            ))
        }
    }

    /// Launch device-local copy operation
    async fn launch_device_copy(&self, transfer: &ZeroCopyTransfer) -> BackendResult<bool> {
        // Implementation would use device-specific copy kernels
        if !transfer.source_ptr.is_null() && !transfer.destination_ptr.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    transfer.source_ptr,
                    transfer.destination_ptr,
                    transfer.size,
                );
            }
            Ok(true)
        } else {
            Err(BackendError::InvalidArgument(
                "Null pointer in device copy".to_string(),
            ))
        }
    }

    /// Launch peer-to-peer transfer
    #[allow(unused_unsafe)]
    async fn launch_p2p_transfer(&self, transfer: &ZeroCopyTransfer) -> BackendResult<bool> {
        #[allow(unused_variables)]
        let source_key = format!(
            "{}:{}",
            transfer.source_device.device_type(),
            transfer.source_device.id()
        );
        #[allow(unused_variables)]
        let dest_key = format!(
            "{}:{}",
            transfer.destination_device.device_type(),
            transfer.destination_device.id()
        );

        match (
            transfer.source_device.device_type(),
            transfer.destination_device.device_type(),
        ) {
            #[cfg(feature = "cuda")]
            (DeviceType::Cuda(_), DeviceType::Cuda(_)) => {
                if let (Some(_src_device), Some(_dst_device)) = (
                    self.cuda_devices.get(&source_key),
                    self.cuda_devices.get(&dest_key),
                ) {
                    // Use SciRS2 CUDA peer-to-peer transfer
                    unsafe {
                        scirs2_cuda::memory::copy_peer_to_peer(
                            transfer.source_ptr,
                            transfer.destination_ptr,
                            transfer.size,
                        )
                        .await
                        .map_err(|e| {
                            BackendError::BackendError(format!("CUDA P2P transfer failed: {}", e))
                        })?;
                    }
                    Ok(true)
                } else {
                    Err(BackendError::BackendError(
                        "CUDA devices not registered for P2P transfer".to_string(),
                    ))
                }
            }

            _ => {
                // Fallback to conventional copy for non-CUDA P2P
                if !transfer.source_ptr.is_null() && !transfer.destination_ptr.is_null() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            transfer.source_ptr,
                            transfer.destination_ptr,
                            transfer.size,
                        );
                    }
                    Ok(true)
                } else {
                    Err(BackendError::InvalidArgument(
                        "Null pointer in P2P transfer".to_string(),
                    ))
                }
            }
        }
    }

    /// Launch CUDA asynchronous transfer
    #[cfg(feature = "cuda")]
    #[allow(unused_unsafe)]
    async fn launch_cuda_async_transfer(
        &self,
        _cuda_device: &SciRs2CudaDevice,
        transfer: &ZeroCopyTransfer,
    ) -> BackendResult<bool> {
        unsafe {
            match transfer.direction {
                TransferDirection::HostToDevice => {
                    scirs2_cuda::memory::copy_host_to_device_async(
                        transfer.source_ptr,
                        transfer.destination_ptr,
                        transfer.size,
                    )
                    .await
                    .map_err(|e| {
                        BackendError::BackendError(format!("CUDA H2D async transfer failed: {}", e))
                    })?;
                }
                TransferDirection::DeviceToHost => {
                    scirs2_cuda::memory::copy_device_to_host_async(
                        transfer.source_ptr,
                        transfer.destination_ptr,
                        transfer.size,
                    )
                    .await
                    .map_err(|e| {
                        BackendError::BackendError(format!("CUDA D2H async transfer failed: {}", e))
                    })?;
                }
                _ => {
                    return Err(BackendError::InvalidArgument(
                        "Invalid transfer direction for CUDA async transfer".to_string(),
                    ));
                }
            }
        }
        Ok(true)
    }

    /// Launch CUDA synchronous transfer
    #[cfg(feature = "cuda")]
    async fn launch_cuda_sync_transfer(
        &self,
        _cuda_device: &SciRs2CudaDevice,
        transfer: &ZeroCopyTransfer,
    ) -> BackendResult<bool> {
        #[allow(unused_unsafe)]
        unsafe {
            match transfer.direction {
                TransferDirection::HostToDevice => {
                    scirs2_cuda::memory::copy_host_to_device(
                        transfer.source_ptr,
                        transfer.destination_ptr,
                        transfer.size,
                    )
                    .map_err(|e| {
                        BackendError::BackendError(format!("CUDA H2D sync transfer failed: {}", e))
                    })?;
                }
                TransferDirection::DeviceToHost => {
                    scirs2_cuda::memory::copy_device_to_host(
                        transfer.source_ptr,
                        transfer.destination_ptr,
                        transfer.size,
                    )
                    .map_err(|e| {
                        BackendError::BackendError(format!("CUDA D2H sync transfer failed: {}", e))
                    })?;
                }
                _ => {
                    return Err(BackendError::InvalidArgument(
                        "Invalid transfer direction for CUDA sync transfer".to_string(),
                    ));
                }
            }
        }
        Ok(true)
    }

    /// Launch CUDA streaming transfer
    #[cfg(feature = "cuda")]
    async fn launch_cuda_streaming_transfer(
        &self,
        cuda_device: &SciRs2CudaDevice,
        transfer: &ZeroCopyTransfer,
    ) -> BackendResult<bool> {
        const CHUNK_SIZE: usize = 64 * 1024 * 1024; // 64MB chunks for CUDA
        let num_chunks = (transfer.size + CHUNK_SIZE - 1) / CHUNK_SIZE;

        for chunk in 0..num_chunks {
            let chunk_offset = chunk * CHUNK_SIZE;
            let chunk_size = std::cmp::min(CHUNK_SIZE, transfer.size - chunk_offset);

            if chunk_size == 0 {
                break;
            }

            // Create chunk transfer
            let chunk_transfer = ZeroCopyTransfer {
                source_ptr: unsafe { transfer.source_ptr.add(chunk_offset) },
                destination_ptr: unsafe { transfer.destination_ptr.add(chunk_offset) },
                size: chunk_size,
                ..transfer.clone()
            };

            // Launch chunk transfer asynchronously
            self.launch_cuda_async_transfer(cuda_device, &chunk_transfer)
                .await?;

            // Yield between chunks for better scheduling
            #[cfg(feature = "async")]
            tokio::task::yield_now().await;
        }

        // Synchronize all transfers
        scirs2_cuda::synchronize(cuda_device).map_err(|e| {
            BackendError::BackendError(format!("CUDA streaming sync failed: {}", e))
        })?;

        Ok(true)
    }

    /// Launch Metal asynchronous transfer
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    async fn launch_metal_async_transfer(
        &self,
        metal_device: &SciRs2MetalDevice,
        transfer: &ZeroCopyTransfer,
    ) -> BackendResult<bool> {
        #[allow(unused_unsafe)]
        unsafe {
            match transfer.direction {
                TransferDirection::HostToDevice => {
                    scirs2_metal::memory::copy_host_to_device_async(
                        metal_device,
                        transfer.source_ptr,
                        transfer.destination_ptr,
                        transfer.size,
                    )
                    .await
                    .map_err(|e| {
                        BackendError::BackendError(format!(
                            "Metal H2D async transfer failed: {}",
                            e
                        ))
                    })?;
                }
                TransferDirection::DeviceToHost => {
                    scirs2_metal::memory::copy_device_to_host_async(
                        metal_device,
                        transfer.source_ptr,
                        transfer.destination_ptr,
                        transfer.size,
                    )
                    .await
                    .map_err(|e| {
                        BackendError::BackendError(format!(
                            "Metal D2H async transfer failed: {}",
                            e
                        ))
                    })?;
                }
                _ => {
                    return Err(BackendError::InvalidArgument(
                        "Invalid transfer direction for Metal async transfer".to_string(),
                    ));
                }
            }
        }
        Ok(true)
    }

    /// Launch Metal synchronous transfer
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    async fn launch_metal_sync_transfer(
        &self,
        metal_device: &SciRs2MetalDevice,
        transfer: &ZeroCopyTransfer,
    ) -> BackendResult<bool> {
        #[allow(unused_unsafe)]
        unsafe {
            match transfer.direction {
                TransferDirection::HostToDevice => {
                    scirs2_metal::memory::copy_host_to_device(
                        metal_device,
                        transfer.source_ptr,
                        transfer.destination_ptr,
                        transfer.size,
                    )
                    .map_err(|e| {
                        BackendError::BackendError(format!("Metal H2D sync transfer failed: {}", e))
                    })?;
                }
                TransferDirection::DeviceToHost => {
                    scirs2_metal::memory::copy_device_to_host(
                        metal_device,
                        transfer.source_ptr,
                        transfer.destination_ptr,
                        transfer.size,
                    )
                    .map_err(|e| {
                        BackendError::BackendError(format!("Metal D2H sync transfer failed: {}", e))
                    })?;
                }
                _ => {
                    return Err(BackendError::InvalidArgument(
                        "Invalid transfer direction for Metal sync transfer".to_string(),
                    ));
                }
            }
        }
        Ok(true)
    }

    /// Launch Metal streaming transfer
    #[cfg(all(feature = "metal", target_os = "macos", target_arch = "aarch64"))]
    async fn launch_metal_streaming_transfer(
        &self,
        metal_device: &SciRs2MetalDevice,
        transfer: &ZeroCopyTransfer,
    ) -> BackendResult<bool> {
        const CHUNK_SIZE: usize = 32 * 1024 * 1024; // 32MB chunks for Metal
        let num_chunks = (transfer.size + CHUNK_SIZE - 1) / CHUNK_SIZE;

        for chunk in 0..num_chunks {
            let chunk_offset = chunk * CHUNK_SIZE;
            let chunk_size = std::cmp::min(CHUNK_SIZE, transfer.size - chunk_offset);

            if chunk_size == 0 {
                break;
            }

            // Create chunk transfer
            let chunk_transfer = ZeroCopyTransfer {
                source_ptr: unsafe { transfer.source_ptr.add(chunk_offset) },
                destination_ptr: unsafe { transfer.destination_ptr.add(chunk_offset) },
                size: chunk_size,
                ..transfer.clone()
            };

            // Launch chunk transfer asynchronously
            self.launch_metal_async_transfer(metal_device, &chunk_transfer)
                .await?;

            // Yield between chunks
            #[cfg(feature = "async")]
            tokio::task::yield_now().await;
        }

        // Synchronize all transfers
        scirs2_metal::synchronize(metal_device).map_err(|e| {
            BackendError::BackendError(format!("Metal streaming sync failed: {}", e))
        })?;

        Ok(true)
    }

    /// Get transfer statistics
    pub fn get_stats(&self) -> ZeroCopyStats {
        self.stats.read().unwrap().clone()
    }

    /// Reset transfer statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write().unwrap();
        *stats = ZeroCopyStats::default();
    }

    /// Get optimal transfer mode for given transfer
    pub fn get_optimal_transfer_mode(&self, transfer: &ZeroCopyTransfer) -> TransferMode {
        let source_caps = self.get_capabilities(&transfer.source_device);
        let dest_caps = self.get_capabilities(&transfer.destination_device);

        match (source_caps, dest_caps) {
            (Some(src), Some(dst)) => {
                // Choose based on capabilities and transfer size
                if transfer.size > 100 * 1024 * 1024
                    && src.streaming_transfers
                    && dst.streaming_transfers
                {
                    TransferMode::Streaming
                } else if src.async_transfers && dst.async_transfers {
                    TransferMode::Asynchronous
                } else if transfer.direction == TransferDirection::CrossDevice
                    && src.peer_to_peer
                    && dst.peer_to_peer
                {
                    TransferMode::PeerToPeer
                } else {
                    TransferMode::Synchronous
                }
            }
            _ => TransferMode::Synchronous,
        }
    }

    /// Optimize transfer parameters
    pub fn optimize_transfer(&self, mut transfer: ZeroCopyTransfer) -> ZeroCopyTransfer {
        // Set optimal transfer mode
        transfer.mode = self.get_optimal_transfer_mode(&transfer);

        // Optimize alignment for better performance
        if transfer.alignment < 256 && transfer.size > 1024 * 1024 {
            transfer.alignment = 256; // Optimize for large transfers
        }

        // Set appropriate priority based on size
        transfer.priority = if transfer.size > 100 * 1024 * 1024 {
            0 // High priority for large transfers
        } else {
            1 // Normal priority
        };

        transfer
    }
}

impl Default for ZeroCopyManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for zero-copy operations
pub mod utils {
    use super::*;

    /// Detect zero-copy capabilities for different device types
    pub fn detect_capabilities(device_type: DeviceType) -> ZeroCopyCapabilities {
        match device_type {
            DeviceType::Cuda(_) => ZeroCopyCapabilities {
                unified_memory: true,
                peer_to_peer: true,
                memory_mapping: true,
                direct_gpu_access: true,
                pinned_memory: true,
                memory_advice: true,
                async_transfers: true,
                streaming_transfers: true,
            },
            DeviceType::Metal(_) => ZeroCopyCapabilities {
                unified_memory: true,
                peer_to_peer: false, // Limited P2P support
                memory_mapping: true,
                direct_gpu_access: true,
                pinned_memory: true,
                memory_advice: true,
                async_transfers: true,
                streaming_transfers: true,
            },
            DeviceType::Wgpu(_) => ZeroCopyCapabilities {
                unified_memory: false,
                peer_to_peer: false,
                memory_mapping: true,
                direct_gpu_access: false,
                pinned_memory: false,
                memory_advice: false,
                async_transfers: true,
                streaming_transfers: false,
            },
            DeviceType::Cpu => ZeroCopyCapabilities {
                unified_memory: true,
                peer_to_peer: false,
                memory_mapping: true,
                direct_gpu_access: false,
                pinned_memory: true,
                memory_advice: false,
                async_transfers: false,
                streaming_transfers: false,
            },
        }
    }

    /// Check if pointers are properly aligned for zero-copy
    pub fn check_alignment(ptr: *const u8, alignment: usize) -> bool {
        if alignment == 0 || (alignment & (alignment - 1)) != 0 {
            return false; // Invalid alignment (must be power of 2)
        }
        (ptr as usize).is_multiple_of(alignment)
    }

    /// Calculate optimal chunk size for streaming transfers
    pub fn optimal_chunk_size(total_size: usize, device_type: DeviceType) -> usize {
        let base_chunk_size = match device_type {
            DeviceType::Cuda(_) => 64 * 1024 * 1024,  // 64MB
            DeviceType::Metal(_) => 32 * 1024 * 1024, // 32MB
            DeviceType::Wgpu(_) => 16 * 1024 * 1024,  // 16MB
            DeviceType::Cpu => 128 * 1024 * 1024,     // 128MB
        };

        // Adjust chunk size based on total size
        if total_size < base_chunk_size {
            total_size
        } else {
            std::cmp::min(base_chunk_size, total_size / 8) // At least 8 chunks
        }
    }

    /// Estimate transfer efficiency
    pub fn estimate_efficiency(
        transfer: &ZeroCopyTransfer,
        capabilities: &ZeroCopyCapabilities,
    ) -> f32 {
        if !transfer.is_zero_copy_possible(capabilities) {
            return 0.0; // No zero-copy possible
        }

        let mut efficiency: f32 = 1.0;

        // Reduce efficiency for suboptimal alignment
        if transfer.alignment < 256 {
            efficiency *= 0.8;
        }

        // Reduce efficiency for small transfers
        if transfer.size < 4096 {
            efficiency *= 0.5;
        }

        // Boost efficiency for optimal transfer modes
        match transfer.mode {
            TransferMode::Streaming if transfer.size > 100 * 1024 * 1024 => efficiency *= 1.2,
            TransferMode::PeerToPeer if transfer.direction == TransferDirection::CrossDevice => {
                efficiency *= 1.3
            }
            TransferMode::Asynchronous => efficiency *= 1.1,
            _ => {}
        }

        efficiency.min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Device, DeviceInfo};
    use std::ptr::null_mut;

    fn create_test_device(device_type: DeviceType, id: usize) -> Device {
        let info = DeviceInfo::default();
        Device::new(
            id,
            device_type,
            format!("Test {:?} {}", device_type, id),
            info,
        )
    }

    #[test]
    fn test_zero_copy_capabilities_default() {
        let caps = ZeroCopyCapabilities::default();
        assert!(!caps.has_any_capabilities());
        assert_eq!(caps.capability_score(), 0.0);
        assert_eq!(caps.recommended_transfer_mode(), TransferMode::Synchronous);
    }

    #[test]
    fn test_zero_copy_capabilities_scoring() {
        let mut caps = ZeroCopyCapabilities::default();
        caps.unified_memory = true;
        caps.async_transfers = true;

        assert!(caps.has_any_capabilities());
        assert_eq!(caps.capability_score(), 0.25); // 2/8 features
        assert_eq!(caps.recommended_transfer_mode(), TransferMode::Asynchronous);
    }

    #[test]
    fn test_zero_copy_transfer_creation() {
        let cpu_device = create_test_device(DeviceType::Cpu, 0);
        let gpu_device = create_test_device(DeviceType::Cuda(1), 1);

        let transfer = ZeroCopyTransfer::new(cpu_device, gpu_device, null_mut(), null_mut(), 1024);

        assert_eq!(transfer.direction, TransferDirection::HostToDevice);
        assert_eq!(transfer.mode, TransferMode::Synchronous);
        assert_eq!(transfer.size, 1024);
        assert_eq!(transfer.alignment, 1);
        assert_eq!(transfer.priority, 1);
        assert!(transfer.stream_id.is_none());
    }

    #[test]
    fn test_zero_copy_transfer_direction_detection() {
        let cpu_device = create_test_device(DeviceType::Cpu, 0);
        let gpu_device1 = create_test_device(DeviceType::Cuda(1), 1);
        let gpu_device2 = create_test_device(DeviceType::Cuda(2), 2);

        // Host to device
        let transfer = ZeroCopyTransfer::new(
            cpu_device.clone(),
            gpu_device1.clone(),
            null_mut(),
            null_mut(),
            1024,
        );
        assert_eq!(transfer.direction, TransferDirection::HostToDevice);

        // Device to host
        let transfer = ZeroCopyTransfer::new(
            gpu_device1.clone(),
            cpu_device,
            null_mut(),
            null_mut(),
            1024,
        );
        assert_eq!(transfer.direction, TransferDirection::DeviceToHost);

        // Device to device (same)
        let transfer = ZeroCopyTransfer::new(
            gpu_device1.clone(),
            gpu_device1.clone(),
            null_mut(),
            null_mut(),
            1024,
        );
        assert_eq!(transfer.direction, TransferDirection::DeviceToDevice);

        // Cross device
        let transfer = ZeroCopyTransfer::new(
            gpu_device1.clone(),
            gpu_device2,
            null_mut(),
            null_mut(),
            1024,
        );
        assert_eq!(transfer.direction, TransferDirection::CrossDevice);
    }

    #[test]
    fn test_zero_copy_transfer_builder() {
        let cpu_device = create_test_device(DeviceType::Cpu, 0);
        let gpu_device = create_test_device(DeviceType::Cuda(1), 1);

        let transfer = ZeroCopyTransfer::new(cpu_device, gpu_device, null_mut(), null_mut(), 1024)
            .with_mode(TransferMode::Asynchronous)
            .with_alignment(256)
            .with_priority(0)
            .with_stream(42);

        assert_eq!(transfer.mode, TransferMode::Asynchronous);
        assert_eq!(transfer.alignment, 256);
        assert_eq!(transfer.priority, 0);
        assert_eq!(transfer.stream_id, Some(42));
    }

    #[test]
    fn test_zero_copy_transfer_zero_copy_possible() {
        let cpu_device = create_test_device(DeviceType::Cpu, 0);
        let gpu_device = create_test_device(DeviceType::Cuda(1), 1);

        let transfer = ZeroCopyTransfer::new(cpu_device, gpu_device, null_mut(), null_mut(), 1024);

        let caps_unified = ZeroCopyCapabilities {
            unified_memory: true,
            ..Default::default()
        };

        let caps_pinned = ZeroCopyCapabilities {
            pinned_memory: true,
            ..Default::default()
        };

        let caps_none = ZeroCopyCapabilities::default();

        assert!(transfer.is_zero_copy_possible(&caps_unified));
        assert!(transfer.is_zero_copy_possible(&caps_pinned));
        assert!(!transfer.is_zero_copy_possible(&caps_none));
    }

    #[test]
    fn test_zero_copy_transfer_bandwidth_estimation() {
        let cpu_device = create_test_device(DeviceType::Cpu, 0);
        let gpu_device = create_test_device(DeviceType::Cuda(1), 1);

        let transfer = ZeroCopyTransfer::new(cpu_device, gpu_device, null_mut(), null_mut(), 1024)
            .with_alignment(256);

        let bandwidth = transfer.estimate_bandwidth(DeviceType::Cuda(1));
        assert_eq!(bandwidth, 25_000_000_000); // Well-aligned CUDA transfer

        let transfer_unaligned = ZeroCopyTransfer::new(
            create_test_device(DeviceType::Cpu, 0),
            create_test_device(DeviceType::Cuda(1), 1),
            null_mut(),
            null_mut(),
            1024,
        )
        .with_alignment(1);

        let bandwidth_unaligned = transfer_unaligned.estimate_bandwidth(DeviceType::Cuda(1));
        assert_eq!(bandwidth_unaligned, 12_000_000_000); // Unaligned CUDA transfer
    }

    #[test]
    fn test_zero_copy_stats() {
        let mut stats = ZeroCopyStats::default();

        // Test initial state
        assert_eq!(stats.zero_copy_success_rate(), 0.0);
        assert_eq!(stats.error_rate(), 0.0);

        // Update with successful zero-copy transfer
        stats.update_transfer(1024, 100, true, false);
        assert_eq!(stats.total_transfers, 1);
        assert_eq!(stats.zero_copy_transfers, 1);
        assert_eq!(stats.zero_copy_success_rate(), 1.0);

        // Update with fallback transfer
        stats.update_transfer(512, 200, false, false);
        assert_eq!(stats.total_transfers, 2);
        assert_eq!(stats.fallback_transfers, 1);
        assert_eq!(stats.zero_copy_success_rate(), 0.5);

        // Update with error
        stats.update_transfer(256, 50, false, true);
        assert_eq!(stats.total_transfers, 3);
        assert_eq!(stats.error_count, 1);
        assert!((stats.error_rate() - (1.0 / 3.0)).abs() < 0.001);
    }

    #[test]
    fn test_zero_copy_manager_creation() {
        let manager = ZeroCopyManager::new();
        assert!(manager.capabilities.read().unwrap().is_empty());

        let stats = manager.get_stats();
        assert_eq!(stats.total_transfers, 0);
    }

    #[test]
    fn test_utils_detect_capabilities() {
        let cuda_caps = utils::detect_capabilities(DeviceType::Cuda(0));
        assert!(cuda_caps.unified_memory);
        assert!(cuda_caps.peer_to_peer);
        assert!(cuda_caps.streaming_transfers);

        let webgpu_caps = utils::detect_capabilities(DeviceType::Wgpu(0));
        assert!(!webgpu_caps.unified_memory);
        assert!(!webgpu_caps.peer_to_peer);
        assert!(!webgpu_caps.streaming_transfers);
    }

    #[test]
    fn test_utils_check_alignment() {
        let ptr = 0x1000 as *const u8; // 4KB aligned

        assert!(utils::check_alignment(ptr, 16));
        assert!(utils::check_alignment(ptr, 256));
        assert!(utils::check_alignment(ptr, 4096));
        assert!(!utils::check_alignment(ptr, 8192));

        // Test invalid alignments
        assert!(!utils::check_alignment(ptr, 0));
        assert!(!utils::check_alignment(ptr, 3)); // Not power of 2
    }

    #[test]
    fn test_utils_optimal_chunk_size() {
        let cuda_chunk = utils::optimal_chunk_size(1024 * 1024 * 1024, DeviceType::Cuda(0));
        assert_eq!(cuda_chunk, 64 * 1024 * 1024); // 64MB for large transfers

        let small_chunk = utils::optimal_chunk_size(1024, DeviceType::Cuda(0));
        assert_eq!(small_chunk, 1024); // Use full size for small transfers
    }

    #[test]
    fn test_utils_estimate_efficiency() {
        let cpu_device = create_test_device(DeviceType::Cpu, 0);
        let gpu_device = create_test_device(DeviceType::Cuda(1), 1);

        let transfer =
            ZeroCopyTransfer::new(cpu_device, gpu_device, null_mut(), null_mut(), 1024 * 1024)
                .with_alignment(256)
                .with_mode(TransferMode::Asynchronous);

        let caps = ZeroCopyCapabilities {
            unified_memory: true,
            async_transfers: true,
            ..Default::default()
        };

        let efficiency = utils::estimate_efficiency(&transfer, &caps);
        assert!(efficiency > 0.0); // Should be positive for valid zero-copy transfers
        assert!(efficiency <= 1.0); // Efficiency is capped at 1.0
    }
}
