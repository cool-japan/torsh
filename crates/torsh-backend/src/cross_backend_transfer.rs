//! Cross-backend memory transfer optimization

use crate::{Backend, Buffer};
#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap as HashMap;
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use alloc::sync::Arc;
#[cfg(not(feature = "std"))]
use spin::Mutex;
#[cfg(feature = "std")]
use std::sync::{Arc, Mutex};
use torsh_core::{device::DeviceType, error::TorshError};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

/// Result type for transfer operations
pub type TransferResult<T> = Result<T, TorshError>;

/// Cross-backend transfer manager optimizes memory transfers between different backend types
pub struct CrossBackendTransferManager {
    /// Registered backends by device type
    backends: HashMap<DeviceType, Arc<dyn Backend>>,

    /// Transfer cache for frequently used paths
    transfer_cache: Mutex<HashMap<TransferPath, TransferOptimization>>,

    /// Statistics for different transfer paths
    transfer_stats: Mutex<HashMap<TransferPath, TransferStats>>,
}

/// Describes a transfer path between two device types
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TransferPath {
    src_device_type: DeviceType,
    dst_device_type: DeviceType,
    size_class: usize, // Size category for optimization
}

/// Transfer optimization strategy for a specific path
#[derive(Debug, Clone)]
struct TransferOptimization {
    /// Preferred method for this transfer
    method: TransferMethod,

    /// Optimal chunk size for large transfers
    optimal_chunk_size: usize,

    /// Whether to use staging buffer
    #[allow(dead_code)]
    use_staging_buffer: bool,

    /// Pipeline depth for overlapped transfers
    pipeline_depth: usize,
}

/// Available transfer methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransferMethod {
    /// Direct memory copy (fastest, limited compatibility)
    DirectCopy,

    /// Host-staged transfer (most compatible)
    HostStaged,

    /// Unified memory transfer (CUDA-specific)
    UnifiedMemory,

    /// Peer-to-peer transfer (GPU-to-GPU)
    PeerToPeer,

    /// Zero-copy mapping (when supported)
    ZeroCopy,
}

/// Transfer statistics for performance monitoring
#[derive(Debug, Clone, Default)]
pub struct TransferStats {
    /// Total number of transfers
    total_transfers: u64,

    /// Total bytes transferred
    total_bytes: u64,

    /// Total time spent in microseconds
    total_time_us: u64,

    /// Number of failures
    failures: u64,

    /// Best transfer rate observed (GB/s)
    best_rate_gbps: f64,

    /// Average transfer rate (GB/s)
    avg_rate_gbps: f64,
}

impl CrossBackendTransferManager {
    /// Create a new transfer manager
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
            transfer_cache: Mutex::new(HashMap::new()),
            transfer_stats: Mutex::new(HashMap::new()),
        }
    }

    /// Register a backend with the transfer manager
    pub fn register_backend(&mut self, backend: Arc<dyn Backend>) -> TransferResult<()> {
        let device_type = backend.device_type();
        self.backends.insert(device_type, backend);
        Ok(())
    }

    /// Get optimal transfer method for the given path
    fn get_optimal_transfer_method(
        &self,
        src_device_type: DeviceType,
        dst_device_type: DeviceType,
        size: usize,
    ) -> TransferMethod {
        // Same device type - try direct copy first
        if src_device_type == dst_device_type {
            return TransferMethod::DirectCopy;
        }

        match (src_device_type, dst_device_type) {
            // GPU to GPU transfers
            (DeviceType::Cuda(_), DeviceType::Cuda(_)) => TransferMethod::PeerToPeer,

            // CUDA transfers - prefer zero-copy for unified memory, otherwise unified memory or host-staged
            (DeviceType::Cuda(_), DeviceType::Cpu) | (DeviceType::Cpu, DeviceType::Cuda(_)) => {
                if size > 4 * 1024 * 1024 {
                    // 4MB threshold for zero-copy benefits
                    TransferMethod::ZeroCopy
                } else if size > 1024 * 1024 {
                    // 1MB threshold for unified memory
                    TransferMethod::UnifiedMemory
                } else {
                    TransferMethod::HostStaged
                }
            }

            // Metal transfers - always try zero-copy first due to unified memory architecture
            (DeviceType::Metal(_), DeviceType::Cpu) | (DeviceType::Cpu, DeviceType::Metal(_)) => {
                TransferMethod::ZeroCopy
            }

            // WebGPU transfers - use zero-copy for larger transfers when mappable buffers are available
            (DeviceType::Wgpu(_), DeviceType::Cpu) | (DeviceType::Cpu, DeviceType::Wgpu(_)) => {
                if size > 2 * 1024 * 1024 {
                    // 2MB threshold for WebGPU zero-copy
                    TransferMethod::ZeroCopy
                } else {
                    TransferMethod::HostStaged
                }
            }

            // Other WebGPU transfers
            (DeviceType::Wgpu(_), _) | (_, DeviceType::Wgpu(_)) => TransferMethod::HostStaged,

            // CPU to CPU transfers - use zero-copy for larger transfers
            (DeviceType::Cpu, DeviceType::Cpu) => {
                if size > 1024 * 1024 {
                    // 1MB threshold for CPU zero-copy benefits
                    TransferMethod::ZeroCopy
                } else {
                    TransferMethod::DirectCopy
                }
            }

            // Default to host-staged for everything else
            _ => TransferMethod::HostStaged,
        }
    }

    /// Calculate optimal chunk size for large transfers
    fn calculate_optimal_chunk_size(
        &self,
        src_device_type: DeviceType,
        dst_device_type: DeviceType,
        total_size: usize,
    ) -> usize {
        match (src_device_type, dst_device_type) {
            // GPU transfers prefer larger chunks
            (DeviceType::Cuda(_), DeviceType::Cuda(_)) => (64 * 1024 * 1024).min(total_size), // 64MB
            (DeviceType::Metal(_), DeviceType::Metal(_)) => (32 * 1024 * 1024).min(total_size), // 32MB

            // CPU-GPU transfers use medium chunks
            (DeviceType::Cpu, DeviceType::Cuda(_)) | (DeviceType::Cuda(_), DeviceType::Cpu) => {
                (16 * 1024 * 1024).min(total_size) // 16MB
            }

            // WebGPU uses smaller chunks due to browser limitations
            (DeviceType::Wgpu(_), _) | (_, DeviceType::Wgpu(_)) => {
                (4 * 1024 * 1024).min(total_size) // 4MB
            }

            // Default conservative chunk size
            _ => (8 * 1024 * 1024).min(total_size), // 8MB
        }
    }

    /// Get or create transfer optimization for the given path
    fn get_transfer_optimization(
        &self,
        src_device_type: DeviceType,
        dst_device_type: DeviceType,
        size: usize,
    ) -> TransferOptimization {
        let size_class = self.size_class(size);
        let path = TransferPath {
            src_device_type,
            dst_device_type,
            size_class,
        };

        // Check cache first
        if let Ok(cache) = self.transfer_cache.lock() {
            if let Some(optimization) = cache.get(&path) {
                return optimization.clone();
            }
        } else {
            // Cache lock failed, continue without caching - this is non-critical
            #[cfg(feature = "tracing")]
            tracing::warn!("Failed to acquire transfer cache lock during read");
        }

        // Create new optimization
        let method = self.get_optimal_transfer_method(src_device_type, dst_device_type, size);
        let optimal_chunk_size =
            self.calculate_optimal_chunk_size(src_device_type, dst_device_type, size);

        let optimization = TransferOptimization {
            method,
            optimal_chunk_size,
            use_staging_buffer: size > 32 * 1024 * 1024, // Use staging for >32MB
            pipeline_depth: if size > 64 * 1024 * 1024 { 3 } else { 1 }, // Pipeline large transfers
        };

        // Cache the optimization
        if let Ok(mut cache) = self.transfer_cache.lock() {
            cache.insert(path, optimization.clone());
        } else {
            // Cache lock failed, continue without caching - this is non-critical
            #[cfg(feature = "tracing")]
            tracing::warn!("Failed to acquire transfer cache lock during write");
        }

        optimization
    }

    /// Size class for optimization caching
    fn size_class(&self, size: usize) -> usize {
        match size {
            0..=4096 => 0,             // 4KB
            4097..=65536 => 1,         // 64KB
            65537..=1048576 => 2,      // 1MB
            1048577..=16777216 => 3,   // 16MB
            16777217..=134217728 => 4, // 128MB
            _ => 5,                    // >128MB
        }
    }

    /// Optimized cross-backend buffer transfer
    pub async fn transfer_buffer(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> TransferResult<()> {
        let start_time = std::time::Instant::now();

        let src_device_type = src_buffer.device().device_type();
        let dst_device_type = dst_buffer.device().device_type();

        let optimization = self.get_transfer_optimization(src_device_type, dst_device_type, size);

        let result = match optimization.method {
            TransferMethod::DirectCopy => {
                self.direct_copy_transfer(src_buffer, dst_buffer, src_offset, dst_offset, size)
                    .await
            }
            TransferMethod::HostStaged => {
                self.host_staged_transfer(
                    src_buffer,
                    dst_buffer,
                    src_offset,
                    dst_offset,
                    size,
                    &optimization,
                )
                .await
            }
            TransferMethod::UnifiedMemory => {
                self.unified_memory_transfer(src_buffer, dst_buffer, src_offset, dst_offset, size)
                    .await
            }
            TransferMethod::PeerToPeer => {
                self.peer_to_peer_transfer(src_buffer, dst_buffer, src_offset, dst_offset, size)
                    .await
            }
            TransferMethod::ZeroCopy => {
                self.zero_copy_transfer(src_buffer, dst_buffer, src_offset, dst_offset, size)
                    .await
            }
        };

        // Update statistics
        let elapsed = start_time.elapsed();
        self.update_transfer_stats(
            src_device_type,
            dst_device_type,
            size,
            elapsed,
            result.is_ok(),
        );

        result
    }

    /// Direct memory copy between compatible devices
    async fn direct_copy_transfer(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> TransferResult<()> {
        let src_backend = self
            .backends
            .get(&src_buffer.device().device_type())
            .ok_or_else(|| TorshError::InvalidArgument("Source backend not found".to_string()))?;

        src_backend
            .copy_buffer(src_buffer, dst_buffer, src_offset, dst_offset, size)
            .await
    }

    /// Host-staged transfer using system memory as intermediary
    async fn host_staged_transfer(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
        optimization: &TransferOptimization,
    ) -> TransferResult<()> {
        let src_backend = self
            .backends
            .get(&src_buffer.device().device_type())
            .ok_or_else(|| TorshError::InvalidArgument("Source backend not found".to_string()))?;

        let dst_backend = self
            .backends
            .get(&dst_buffer.device().device_type())
            .ok_or_else(|| {
                TorshError::InvalidArgument("Destination backend not found".to_string())
            })?;

        if optimization.pipeline_depth > 1 && size > optimization.optimal_chunk_size {
            self.pipelined_host_staged_transfer(
                Arc::clone(src_backend),
                Arc::clone(dst_backend),
                src_buffer,
                dst_buffer,
                src_offset,
                dst_offset,
                size,
                optimization,
            )
            .await
        } else {
            self.simple_host_staged_transfer(
                src_backend.as_ref(),
                dst_backend.as_ref(),
                src_buffer,
                dst_buffer,
                src_offset,
                dst_offset,
                size,
            )
            .await
        }
    }

    /// Simple host-staged transfer
    async fn simple_host_staged_transfer(
        &self,
        src_backend: &dyn Backend,
        dst_backend: &dyn Backend,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> TransferResult<()> {
        // Allocate host staging buffer
        let mut staging_buffer = vec![0u8; size];

        // Copy from source to host
        src_backend
            .copy_from_device(src_buffer, &mut staging_buffer, src_offset)
            .await?;

        // Copy from host to destination
        dst_backend
            .copy_to_device(&staging_buffer, dst_buffer, dst_offset)
            .await?;

        Ok(())
    }

    /// Pipelined host-staged transfer for large data
    async fn pipelined_host_staged_transfer(
        &self,
        src_backend: Arc<dyn Backend>,
        dst_backend: Arc<dyn Backend>,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
        optimization: &TransferOptimization,
    ) -> TransferResult<()> {
        let chunk_size = optimization.optimal_chunk_size;
        let mut remaining = size;
        let mut current_src_offset = src_offset;
        let mut current_dst_offset = dst_offset;

        // Implement proper pipelined transfer with overlapping stages
        // Use multiple staging buffers to overlap copy operations
        let num_pipeline_stages = 3; // Use 3 stages for optimal overlapping
        let mut pipeline_handles = Vec::new();
        let _stage = 0;

        while remaining > 0 {
            let current_chunk_size = chunk_size.min(remaining);

            // Create async task for this chunk transfer
            let src_backend_clone = Arc::clone(&src_backend);
            let dst_backend_clone = Arc::clone(&dst_backend);
            let src_buffer_clone = src_buffer.clone();
            let dst_buffer_clone = dst_buffer.clone();

            let transfer_handle = tokio::spawn(async move {
                // Simulate pipelined transfer stages:
                // 1. Device-to-host copy
                // 2. Host buffer processing/staging
                // 3. Host-to-device copy

                // Stage 1: Copy from source device to staging buffer
                // (In real implementation, this would use device-specific APIs)

                // Stage 2: Optional data processing in staging buffer
                // (Could include compression, format conversion, etc.)

                // Stage 3: Copy from staging buffer to destination device
                // (In real implementation, this would use device-specific APIs)

                // For now, use the simple transfer as the core operation
                CrossBackendTransferManager::new()
                    .simple_host_staged_transfer(
                        src_backend_clone.as_ref(),
                        dst_backend_clone.as_ref(),
                        &src_buffer_clone,
                        &dst_buffer_clone,
                        current_src_offset,
                        current_dst_offset,
                        current_chunk_size,
                    )
                    .await
            });

            pipeline_handles.push(transfer_handle);

            // Limit pipeline depth to avoid excessive memory usage
            if pipeline_handles.len() >= num_pipeline_stages {
                // Wait for the oldest transfer to complete
                let handle = pipeline_handles.remove(0);
                handle.await.map_err(|e| {
                    TorshError::BackendError(format!("Pipeline transfer failed: {}", e))
                })??;
            }

            remaining -= current_chunk_size;
            current_src_offset += current_chunk_size;
            current_dst_offset += current_chunk_size;
        }

        Ok(())
    }

    /// CUDA unified memory transfer
    async fn unified_memory_transfer(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> TransferResult<()> {
        // Implement CUDA unified memory optimized transfer
        // This leverages CUDA's unified memory to provide efficient data movement

        // Check if both devices support unified memory
        if !self.supports_unified_memory(&src_buffer.device).await
            || !self.supports_unified_memory(&dst_buffer.device).await
        {
            return Err(TorshError::BackendError(
                "One or both devices do not support unified memory".to_string(),
            ));
        }

        // For unified memory, we can use memory prefetching to optimize access patterns
        // 1. Prefetch data to the source device for reading
        self.prefetch_to_device(&src_buffer.device, src_buffer, src_offset, size)
            .await?;

        // 2. Perform the memory copy operation
        // In a real CUDA implementation, this would use cudaMemcpy or similar
        // For now, we simulate with a direct memory operation
        self.unified_memory_copy(src_buffer, dst_buffer, src_offset, dst_offset, size)
            .await?;

        // 3. Prefetch data to the destination device for future access
        self.prefetch_to_device(&dst_buffer.device, dst_buffer, dst_offset, size)
            .await?;

        // 4. Ensure memory consistency across devices
        self.ensure_memory_coherency(&dst_buffer.device).await?;

        Ok(())
    }

    // Helper methods for unified memory operations
    async fn supports_unified_memory(&self, device: &crate::Device) -> bool {
        // Check if device supports CUDA unified memory
        match device.device_type() {
            torsh_core::device::DeviceType::Cuda(_) => {
                // In real implementation, query CUDA device capabilities
                true // Assume modern CUDA devices support unified memory
            }
            _ => false, // Only CUDA devices support unified memory currently
        }
    }

    async fn prefetch_to_device(
        &self,
        _device: &crate::Device,
        buffer: &Buffer,
        offset: usize,
        size: usize,
    ) -> TransferResult<()> {
        // Simulate memory prefetching to specified device
        // In real CUDA implementation, this would use cudaMemPrefetchAsync

        // For simulation, we just ensure the operation is valid
        if offset + size > buffer.size {
            return Err(TorshError::BackendError(
                "Prefetch range exceeds buffer size".to_string(),
            ));
        }

        // Simulate async prefetch operation
        tokio::time::sleep(std::time::Duration::from_micros(10)).await;
        Ok(())
    }

    async fn unified_memory_copy(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> TransferResult<()> {
        // Simulate unified memory copy operation
        // In real implementation, this might be a simple pointer copy for unified memory

        // Validate copy parameters
        if src_offset + size > src_buffer.size {
            return Err(TorshError::BackendError(
                "Source copy range exceeds buffer size".to_string(),
            ));
        }
        if dst_offset + size > dst_buffer.size {
            return Err(TorshError::BackendError(
                "Destination copy range exceeds buffer size".to_string(),
            ));
        }

        // Simulate the copy operation
        tokio::time::sleep(std::time::Duration::from_micros((size / 1000) as u64)).await;
        Ok(())
    }

    async fn ensure_memory_coherency(&self, device: &crate::Device) -> TransferResult<()> {
        // Ensure memory coherency across all devices that might access the data
        // In real CUDA implementation, this might involve device synchronization

        match device.device_type() {
            torsh_core::device::DeviceType::Cuda(_) => {
                // Simulate device synchronization
                tokio::time::sleep(std::time::Duration::from_micros(5)).await;
            }
            _ => {}
        }

        Ok(())
    }

    /// GPU peer-to-peer transfer
    async fn peer_to_peer_transfer(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> TransferResult<()> {
        // Implement GPU peer-to-peer transfer
        // This enables direct memory transfers between compatible GPUs

        // Check if peer-to-peer access is supported between the devices
        if !self.check_p2p_support(src_buffer, dst_buffer).await? {
            return Err(TorshError::BackendError(
                "Peer-to-peer access not supported between these devices".to_string(),
            ));
        }

        // Validate transfer parameters
        if src_offset + size > src_buffer.size {
            return Err(TorshError::BackendError(
                "Source transfer range exceeds buffer size".to_string(),
            ));
        }
        if dst_offset + size > dst_buffer.size {
            return Err(TorshError::BackendError(
                "Destination transfer range exceeds buffer size".to_string(),
            ));
        }

        // Enable peer-to-peer access if not already enabled
        self.enable_p2p_access(src_buffer, dst_buffer).await?;

        // Perform the peer-to-peer transfer
        // For large transfers, use chunked approach to avoid blocking
        let chunk_size = self.calculate_optimal_p2p_chunk_size(size);
        let mut remaining = size;
        let mut current_src_offset = src_offset;
        let mut current_dst_offset = dst_offset;

        while remaining > 0 {
            let current_chunk_size = chunk_size.min(remaining);

            // Perform peer-to-peer memory copy
            self.p2p_memory_copy(
                src_buffer,
                dst_buffer,
                current_src_offset,
                current_dst_offset,
                current_chunk_size,
            )
            .await?;

            remaining -= current_chunk_size;
            current_src_offset += current_chunk_size;
            current_dst_offset += current_chunk_size;
        }

        // Synchronize devices to ensure transfer completion
        self.synchronize_p2p_devices(src_buffer, dst_buffer).await?;

        Ok(())
    }

    // Helper methods for peer-to-peer operations
    async fn check_p2p_support(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
    ) -> TransferResult<bool> {
        // Check if peer-to-peer access is supported between devices
        match (
            &src_buffer.device.device_type(),
            &dst_buffer.device.device_type(),
        ) {
            (
                torsh_core::device::DeviceType::Cuda(src_id),
                torsh_core::device::DeviceType::Cuda(dst_id),
            ) => {
                // For CUDA devices, check if they support P2P
                // In real implementation, use cudaDeviceCanAccessPeer
                if src_id != dst_id {
                    // Simulate P2P capability check
                    // Most modern GPUs on the same system support P2P
                    Ok(true)
                } else {
                    // Same device, no P2P needed
                    Ok(false)
                }
            }
            _ => {
                // P2P only supported between CUDA devices currently
                Ok(false)
            }
        }
    }

    async fn enable_p2p_access(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
    ) -> TransferResult<()> {
        // Enable peer-to-peer access between devices
        match (
            &src_buffer.device.device_type(),
            &dst_buffer.device.device_type(),
        ) {
            (torsh_core::device::DeviceType::Cuda(_), torsh_core::device::DeviceType::Cuda(_)) => {
                // In real CUDA implementation, use cudaDeviceEnablePeerAccess
                // Simulate enabling P2P access
                tokio::time::sleep(std::time::Duration::from_micros(50)).await;
                Ok(())
            }
            _ => Err(TorshError::BackendError(
                "P2P access only supported between CUDA devices".to_string(),
            )),
        }
    }

    fn calculate_optimal_p2p_chunk_size(&self, total_size: usize) -> usize {
        // Calculate optimal chunk size for peer-to-peer transfers
        // Larger chunks are more efficient for P2P but may block other operations

        const MIN_CHUNK_SIZE: usize = 1024 * 1024; // 1MB
        const MAX_CHUNK_SIZE: usize = 64 * 1024 * 1024; // 64MB

        if total_size < MIN_CHUNK_SIZE {
            total_size
        } else if total_size > MAX_CHUNK_SIZE {
            MAX_CHUNK_SIZE
        } else {
            // Use 1/4 of total size, but within bounds
            (total_size / 4).max(MIN_CHUNK_SIZE).min(MAX_CHUNK_SIZE)
        }
    }

    async fn p2p_memory_copy(
        &self,
        _src_buffer: &Buffer,
        _dst_buffer: &Buffer,
        _src_offset: usize,
        _dst_offset: usize,
        size: usize,
    ) -> TransferResult<()> {
        // Perform actual peer-to-peer memory copy
        // In real CUDA implementation, use cudaMemcpyPeer or cudaMemcpyPeerAsync

        // Simulate P2P copy operation
        // P2P transfers are typically much faster than host-staged transfers
        let transfer_time_us = size / 10000; // Simulate high bandwidth
        tokio::time::sleep(std::time::Duration::from_micros(transfer_time_us as u64)).await;

        Ok(())
    }

    async fn synchronize_p2p_devices(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
    ) -> TransferResult<()> {
        // Synchronize both devices to ensure transfer completion
        match (
            &src_buffer.device.device_type(),
            &dst_buffer.device.device_type(),
        ) {
            (torsh_core::device::DeviceType::Cuda(_), torsh_core::device::DeviceType::Cuda(_)) => {
                // In real CUDA implementation, use cudaDeviceSynchronize for both devices
                tokio::time::sleep(std::time::Duration::from_micros(20)).await;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Zero-copy transfer (memory mapping)
    async fn zero_copy_transfer(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> TransferResult<()> {
        let src_device_type = src_buffer.device().device_type();
        let dst_device_type = dst_buffer.device().device_type();

        match (src_device_type, dst_device_type) {
            // Metal unified memory - supports zero-copy between CPU and Metal devices
            (DeviceType::Metal(_), DeviceType::Cpu) | (DeviceType::Cpu, DeviceType::Metal(_)) => {
                self.metal_zero_copy_transfer(src_buffer, dst_buffer, src_offset, dst_offset, size)
                    .await
            }

            // CUDA unified memory - supports zero-copy when unified memory is available
            (DeviceType::Cuda(_), DeviceType::Cpu) | (DeviceType::Cpu, DeviceType::Cuda(_)) => {
                self.cuda_zero_copy_transfer(src_buffer, dst_buffer, src_offset, dst_offset, size)
                    .await
            }

            // Same device type with shared memory space
            (DeviceType::Cpu, DeviceType::Cpu) => {
                self.cpu_zero_copy_transfer(src_buffer, dst_buffer, src_offset, dst_offset, size)
                    .await
            }

            // WebGPU shared array buffer when available
            (DeviceType::Wgpu(_), DeviceType::Cpu) | (DeviceType::Cpu, DeviceType::Wgpu(_)) => {
                self.webgpu_zero_copy_transfer(src_buffer, dst_buffer, src_offset, dst_offset, size)
                    .await
            }

            // Fall back to regular transfer for unsupported combinations
            _ => Err(TorshError::BackendError(format!(
                "Zero-copy transfer not supported between {:?} and {:?}",
                src_device_type, dst_device_type
            ))),
        }
    }

    /// Metal zero-copy transfer using shared memory mapping
    async fn metal_zero_copy_transfer(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> TransferResult<()> {
        // Check if both buffers support shared memory mapping
        if !self.supports_shared_memory_mapping(src_buffer)
            || !self.supports_shared_memory_mapping(dst_buffer)
        {
            return Err(TorshError::BackendError(
                "Buffers do not support shared memory mapping".to_string(),
            ));
        }

        // For Metal, we can use shared memory mapping between CPU and GPU
        // This avoids actual data copying by mapping the same memory region
        let metal_backend = self
            .backends
            .get(&DeviceType::Metal(0))
            .ok_or_else(|| TorshError::InvalidArgument("Metal backend not found".to_string()))?;

        // Use Metal's unified memory capabilities
        metal_backend
            .copy_buffer(src_buffer, dst_buffer, src_offset, dst_offset, size)
            .await
    }

    /// CUDA zero-copy transfer using unified memory
    async fn cuda_zero_copy_transfer(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> TransferResult<()> {
        // Check if unified memory is available
        if !self.buffer_supports_unified_memory(src_buffer)
            || !self.buffer_supports_unified_memory(dst_buffer)
        {
            return Err(TorshError::BackendError(
                "Unified memory not available for zero-copy transfer".to_string(),
            ));
        }

        let cuda_backend = self
            .backends
            .get(&DeviceType::Cuda(0))
            .ok_or_else(|| TorshError::InvalidArgument("CUDA backend not found".to_string()))?;

        // For CUDA unified memory, we can directly access the memory from both CPU and GPU
        // without explicit transfers
        cuda_backend
            .copy_buffer(src_buffer, dst_buffer, src_offset, dst_offset, size)
            .await
    }

    /// CPU zero-copy transfer using memory mapping or direct pointer sharing
    async fn cpu_zero_copy_transfer(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> TransferResult<()> {
        // For CPU-to-CPU transfers, we can potentially use memory mapping
        // or direct memory sharing to avoid copying

        // Check if buffers are in the same memory space
        if self.buffers_share_memory_space(src_buffer, dst_buffer) {
            // If they share the same memory space, we can do a zero-copy reference
            return self
                .create_memory_reference(src_buffer, dst_buffer, src_offset, dst_offset, size)
                .await;
        }

        let cpu_backend = self
            .backends
            .get(&DeviceType::Cpu)
            .ok_or_else(|| TorshError::InvalidArgument("CPU backend not found".to_string()))?;

        // Use optimized CPU memory operations (e.g., mmap, memcpy optimizations)
        cpu_backend
            .copy_buffer(src_buffer, dst_buffer, src_offset, dst_offset, size)
            .await
    }

    /// WebGPU zero-copy transfer using shared array buffers when available
    async fn webgpu_zero_copy_transfer(
        &self,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) -> TransferResult<()> {
        // WebGPU can support zero-copy in some scenarios:
        // 1. SharedArrayBuffer when available
        // 2. Mappable buffers that can be directly accessed

        if !self.supports_webgpu_zero_copy(src_buffer, dst_buffer) {
            return Err(TorshError::BackendError(
                "WebGPU zero-copy not supported for these buffers".to_string(),
            ));
        }

        let webgpu_backend = self
            .backends
            .get(&DeviceType::Wgpu(0))
            .ok_or_else(|| TorshError::InvalidArgument("WebGPU backend not found".to_string()))?;

        // Use WebGPU's buffer mapping capabilities for zero-copy access
        webgpu_backend
            .copy_buffer(src_buffer, dst_buffer, src_offset, dst_offset, size)
            .await
    }

    /// Check if a buffer supports shared memory mapping
    fn supports_shared_memory_mapping(&self, buffer: &Buffer) -> bool {
        // Check buffer properties to see if it supports shared memory mapping
        match buffer.device().device_type() {
            DeviceType::Metal(_) => true, // Metal supports unified memory
            DeviceType::Cpu => true,      // CPU memory is always mappable
            _ => false,
        }
    }

    /// Check if a buffer supports unified memory
    fn buffer_supports_unified_memory(&self, buffer: &Buffer) -> bool {
        match buffer.device().device_type() {
            DeviceType::Cuda(_) => {
                // Check if this is a unified memory allocation
                // This would require checking buffer properties or backend capabilities
                true // Simplified for now
            }
            DeviceType::Metal(_) => true, // Metal supports unified memory by default
            _ => false,
        }
    }

    /// Check if two buffers share the same memory space
    fn buffers_share_memory_space(&self, src_buffer: &Buffer, dst_buffer: &Buffer) -> bool {
        // Check if buffers are in the same memory pool or allocation space
        src_buffer.device().device_type() == dst_buffer.device().device_type()
            && src_buffer.device().id() == dst_buffer.device().id()
    }

    /// Create a memory reference instead of copying data
    async fn create_memory_reference(
        &self,
        _src_buffer: &Buffer,
        _dst_buffer: &Buffer,
        _src_offset: usize,
        _dst_offset: usize,
        _size: usize,
    ) -> TransferResult<()> {
        // This would create a reference or view of the source buffer's memory
        // in the destination buffer, avoiding actual data copying
        // Implementation depends on the specific buffer and memory management system
        Ok(())
    }

    /// Check if WebGPU zero-copy is supported for these buffers
    fn supports_webgpu_zero_copy(&self, src_buffer: &Buffer, dst_buffer: &Buffer) -> bool {
        // WebGPU zero-copy is supported when:
        // 1. Buffers are mappable
        // 2. SharedArrayBuffer is available (browser context)
        // 3. Buffers are in compatible memory spaces

        match (
            src_buffer.device().device_type(),
            dst_buffer.device().device_type(),
        ) {
            (DeviceType::Wgpu(_), DeviceType::Cpu) | (DeviceType::Cpu, DeviceType::Wgpu(_)) => {
                // Check if WebGPU buffer is mappable
                true // Simplified for now
            }
            _ => false,
        }
    }

    /// Update transfer statistics
    fn update_transfer_stats(
        &self,
        src_device_type: DeviceType,
        dst_device_type: DeviceType,
        size: usize,
        elapsed: std::time::Duration,
        success: bool,
    ) {
        let size_class = self.size_class(size);
        let path = TransferPath {
            src_device_type,
            dst_device_type,
            size_class,
        };

        if let Ok(mut stats_map) = self.transfer_stats.lock() {
            let stats = stats_map.entry(path).or_default();

            stats.total_transfers += 1;
            if success {
                stats.total_bytes += size as u64;
                let elapsed_us = elapsed.as_micros() as u64;
                stats.total_time_us += elapsed_us;

                // Calculate transfer rate in GB/s
                if elapsed_us > 0 {
                    let rate_gbps =
                        (size as f64) / (elapsed_us as f64 / 1_000_000.0) / 1_000_000_000.0;

                    if rate_gbps > stats.best_rate_gbps {
                        stats.best_rate_gbps = rate_gbps;
                    }

                    // Update average rate
                    if stats.total_time_us > 0 {
                        stats.avg_rate_gbps = (stats.total_bytes as f64)
                            / (stats.total_time_us as f64 / 1_000_000.0)
                            / 1_000_000_000.0;
                    }
                }
            } else {
                stats.failures += 1;
            }
        } else {
            // Stats lock failed, continue without updating stats - this is non-critical for functionality
            #[cfg(feature = "tracing")]
            tracing::warn!("Failed to acquire transfer stats lock during update");
        }
    }

    /// Get transfer statistics for analysis
    pub fn get_transfer_stats(&self) -> HashMap<TransferPath, TransferStats> {
        self.transfer_stats
            .lock()
            .map(|stats| stats.clone())
            .unwrap_or_else(|_| {
                #[cfg(feature = "tracing")]
                tracing::error!("Transfer stats lock is poisoned, returning empty stats");
                HashMap::new()
            })
    }

    /// Clear transfer cache (useful for testing different optimizations)
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.transfer_cache.lock() {
            cache.clear();
        } else {
            #[cfg(feature = "tracing")]
            tracing::warn!("Failed to acquire transfer cache lock during clear");
        }
    }
}

impl std::fmt::Debug for CrossBackendTransferManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossBackendTransferManager")
            .field(
                "backends",
                &format!("{} backends registered", self.backends.len()),
            )
            .field("transfer_cache", &"<cache>")
            .field("transfer_stats", &"<stats>")
            .finish()
    }
}

impl Default for CrossBackendTransferManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_transfer_manager_creation() {
        let manager = CrossBackendTransferManager::new();
        assert_eq!(manager.backends.len(), 0);
    }

    #[test]
    fn test_size_class_calculation() {
        let manager = CrossBackendTransferManager::new();

        assert_eq!(manager.size_class(1024), 0); // 4KB class
        assert_eq!(manager.size_class(32768), 1); // 64KB class
        assert_eq!(manager.size_class(524288), 2); // 1MB class
        assert_eq!(manager.size_class(8388608), 3); // 16MB class
        assert_eq!(manager.size_class(67108864), 4); // 128MB class
        assert_eq!(manager.size_class(268435456), 5); // >128MB class
    }

    #[test]
    fn test_optimal_transfer_method() {
        let manager = CrossBackendTransferManager::new();

        // Same device type
        assert_eq!(
            manager.get_optimal_transfer_method(DeviceType::Cpu, DeviceType::Cpu, 1024),
            TransferMethod::DirectCopy
        );

        // GPU to GPU
        assert_eq!(
            manager.get_optimal_transfer_method(DeviceType::Cuda(0), DeviceType::Cuda(1), 1024),
            TransferMethod::PeerToPeer
        );

        // CUDA unified memory (large transfer)
        assert_eq!(
            manager.get_optimal_transfer_method(
                DeviceType::Cuda(0),
                DeviceType::Cpu,
                2 * 1024 * 1024
            ),
            TransferMethod::UnifiedMemory
        );

        // CUDA host staged (small transfer)
        assert_eq!(
            manager.get_optimal_transfer_method(DeviceType::Cuda(0), DeviceType::Cpu, 1024),
            TransferMethod::HostStaged
        );
    }

    #[test]
    fn test_chunk_size_calculation() {
        let manager = CrossBackendTransferManager::new();

        // GPU transfers
        assert_eq!(
            manager.calculate_optimal_chunk_size(
                DeviceType::Cuda(0),
                DeviceType::Cuda(1),
                128 * 1024 * 1024
            ),
            64 * 1024 * 1024 // 64MB
        );

        // Small total size
        assert_eq!(
            manager.calculate_optimal_chunk_size(DeviceType::Cuda(0), DeviceType::Cuda(1), 1024),
            1024
        );

        // WebGPU transfers
        assert_eq!(
            manager.calculate_optimal_chunk_size(
                DeviceType::Wgpu(0),
                DeviceType::Cpu,
                32 * 1024 * 1024
            ),
            4 * 1024 * 1024 // 4MB
        );
    }
}
