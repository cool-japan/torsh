//! Multi-GPU support for CUDA backend

use super::{CudaBackend, CudaBuffer, CudaDevice, CudaStream};
use crate::cuda::cuda_sys_compat as cuda_sys;
use crate::cuda::error::{CudaError, CudaResult};
use std::collections::HashMap;
use std::sync::Arc;

/// Multi-GPU context for managing multiple CUDA devices
#[derive(Debug)]
pub struct MultiGpuContext {
    devices: Vec<Arc<CudaDevice>>,
    backends: Vec<Arc<CudaBackend>>,
    streams: HashMap<usize, Arc<CudaStream>>,
    peer_access: HashMap<(usize, usize), bool>,
}

impl MultiGpuContext {
    /// Create a new multi-GPU context with specified devices
    pub fn new(device_ids: Vec<usize>) -> CudaResult<Self> {
        if device_ids.is_empty() {
            return Err(CudaError::InvalidDevice { device_id: 0 });
        }

        // Verify device count
        let device_count = crate::cuda::device_count()? as usize;
        for &id in &device_ids {
            if id >= device_count {
                return Err(CudaError::InvalidDevice { device_id: id });
            }
        }

        // Create devices and backends
        let mut devices = Vec::new();
        let mut backends = Vec::new();
        let mut streams = HashMap::new();

        for &device_id in &device_ids {
            // Set device as current before creating resources
            crate::cuda::set_device(device_id)?;

            let device = Arc::new(CudaDevice::new(device_id)?);
            let backend = Arc::new(CudaBackend::new(super::backend::CudaBackendConfig {
                device_id,
                ..Default::default()
            })?);
            let stream = Arc::new(CudaStream::new()?);

            devices.push(device);
            backends.push(backend);
            streams.insert(device_id, stream);
        }

        // Enable peer access between devices
        let mut peer_access = HashMap::new();
        for i in 0..device_ids.len() {
            for j in 0..device_ids.len() {
                if i != j {
                    let can_access = Self::can_access_peer(device_ids[i], device_ids[j])?;
                    if can_access {
                        Self::enable_peer_access(device_ids[i], device_ids[j])?;
                        peer_access.insert((device_ids[i], device_ids[j]), true);
                    }
                }
            }
        }

        Ok(Self {
            devices,
            backends,
            streams,
            peer_access,
        })
    }

    /// Create context with all available GPUs
    pub fn all_gpus() -> CudaResult<Self> {
        let device_count = crate::cuda::device_count()? as usize;
        let device_ids: Vec<usize> = (0..device_count).collect();
        Self::new(device_ids)
    }

    /// Get the number of devices in this context
    pub fn num_devices(&self) -> usize {
        self.devices.len()
    }

    /// Get device by index
    pub fn device(&self, index: usize) -> Option<&Arc<CudaDevice>> {
        self.devices.get(index)
    }

    /// Get backend by device ID
    pub fn backend(&self, device_id: usize) -> Option<&Arc<CudaBackend>> {
        self.devices
            .iter()
            .position(|d| d.id() == device_id)
            .and_then(|idx| self.backends.get(idx))
    }

    /// Get stream for device
    pub fn stream(&self, device_id: usize) -> Option<&Arc<CudaStream>> {
        self.streams.get(&device_id)
    }

    /// Check if peer access is enabled between two devices
    pub fn has_peer_access(&self, src_device: usize, dst_device: usize) -> bool {
        self.peer_access
            .get(&(src_device, dst_device))
            .copied()
            .unwrap_or(false)
    }

    /// Copy data between devices
    pub async fn copy_between_devices<T: Clone + Send + Sync + Default + 'static>(
        &self,
        src: &CudaBuffer<T>,
        dst: &mut CudaBuffer<T>,
        src_device: usize,
        dst_device: usize,
    ) -> CudaResult<()> {
        if src_device == dst_device {
            // Same device, use regular copy
            dst.copy_from(src)?;
            return Ok(());
        }

        // Check if peer access is available
        if self.has_peer_access(src_device, dst_device) {
            // Direct peer-to-peer copy
            unsafe {
                cuda_sys::cudaMemcpyPeerAsync(
                    dst.device_ptr().as_raw() as *mut _,
                    dst_device as i32,
                    src.device_ptr().as_raw() as *const _,
                    src_device as i32,
                    src.size_bytes(),
                    self.stream(dst_device)
                        .ok_or_else(|| CudaError::InvalidDevice {
                            device_id: dst_device,
                        })?
                        .stream(),
                );
            }
        } else {
            // Copy through host memory
            let mut host_buffer = vec![Default::default(); src.len()];
            src.copy_to_host(&mut host_buffer)?;
            dst.copy_from_host(&host_buffer)?;
        }

        Ok(())
    }

    /// Broadcast data from one device to all others
    pub async fn broadcast<T: Clone + Send + Sync + Default + 'static>(
        &self,
        src: &CudaBuffer<T>,
        src_device: usize,
        dst_buffers: &mut [CudaBuffer<T>],
    ) -> CudaResult<()> {
        if dst_buffers.len() != self.devices.len() - 1 {
            return Err(CudaError::InvalidDevice {
                device_id: dst_buffers.len(),
            });
        }

        // Copy to all other devices
        let mut dst_idx = 0;
        for (_i, device) in self.devices.iter().enumerate() {
            if device.id() != src_device {
                self.copy_between_devices(src, &mut dst_buffers[dst_idx], src_device, device.id())
                    .await?;
                dst_idx += 1;
            }
        }

        Ok(())
    }

    /// Reduce data from all devices to one
    pub async fn reduce<T: Clone + Send + Sync + Default + 'static>(
        &self,
        src_buffers: &[CudaBuffer<T>],
        dst: &mut CudaBuffer<T>,
        dst_device: usize,
        op: ReduceOp,
    ) -> CudaResult<()> {
        if src_buffers.len() != self.devices.len() {
            return Err(CudaError::InvalidDevice {
                device_id: src_buffers.len(),
            });
        }

        // For now, implement sum reduction only
        match op {
            ReduceOp::Sum => {
                // Copy first buffer to destination
                let src_device = self.devices[0].id();
                if src_device == dst_device {
                    dst.copy_from(&src_buffers[0])?;
                } else {
                    self.copy_between_devices(&src_buffers[0], dst, src_device, dst_device)
                        .await?;
                }

                // Add remaining buffers
                for (i, src) in src_buffers.iter().enumerate().skip(1) {
                    let src_device = self.devices[i].id();
                    if src_device == dst_device {
                        // Same device, direct add
                        let backend = self
                            .backend(dst_device)
                            .expect("backend for dst_device should exist");
                        // Note: This is a simplified version, would need proper type handling
                        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                            unsafe {
                                let src_f32 =
                                    std::mem::transmute::<&CudaBuffer<T>, &CudaBuffer<f32>>(src);
                                // Create a temporary output buffer to avoid borrow conflict
                                let mut output_buffer = dst.clone();
                                let output_f32 = std::mem::transmute::<
                                    &mut CudaBuffer<T>,
                                    &mut CudaBuffer<f32>,
                                >(
                                    &mut output_buffer
                                );
                                let dst_f32 = std::mem::transmute::<&CudaBuffer<T>, &CudaBuffer<f32>>(
                                    &*(dst as *const _),
                                );
                                backend.elementwise_add_f32(src_f32, dst_f32, output_f32, None)?;
                                // Copy result back to dst
                                std::ptr::copy_nonoverlapping(
                                    &output_buffer as *const _ as *const u8,
                                    dst as *mut _ as *mut u8,
                                    std::mem::size_of::<CudaBuffer<T>>(),
                                );
                                std::mem::forget(output_buffer);
                            }
                        }
                    } else {
                        // Copy to destination device then add
                        let mut temp = dst.clone();
                        self.copy_between_devices(src, &mut temp, src_device, dst_device)
                            .await?;

                        let backend = self
                            .backend(dst_device)
                            .expect("backend for dst_device should exist");
                        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
                            unsafe {
                                let temp_f32 =
                                    std::mem::transmute::<&CudaBuffer<T>, &CudaBuffer<f32>>(&temp);
                                // Create a temporary output buffer to avoid borrow conflict
                                let mut output_buffer = dst.clone();
                                let output_f32 = std::mem::transmute::<
                                    &mut CudaBuffer<T>,
                                    &mut CudaBuffer<f32>,
                                >(
                                    &mut output_buffer
                                );
                                let dst_f32 = std::mem::transmute::<&CudaBuffer<T>, &CudaBuffer<f32>>(
                                    &*(dst as *const _),
                                );
                                backend.elementwise_add_f32(temp_f32, dst_f32, output_f32, None)?;
                                // Copy result back to dst
                                std::ptr::copy_nonoverlapping(
                                    &output_buffer as *const _ as *const u8,
                                    dst as *mut _ as *mut u8,
                                    std::mem::size_of::<CudaBuffer<T>>(),
                                );
                                std::mem::forget(output_buffer);
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(CudaError::UnsupportedOperation {
                    op: format!("Reduce operation {:?}", op),
                    dtype: "".to_string(),
                });
            }
        }

        Ok(())
    }

    /// All-reduce operation across all devices
    pub async fn all_reduce<T: Clone + Send + Sync + Default + 'static>(
        &self,
        buffers: &mut [CudaBuffer<T>],
        op: ReduceOp,
    ) -> CudaResult<()> {
        if buffers.len() != self.devices.len() {
            return Err(CudaError::InvalidDevice {
                device_id: buffers.len(),
            });
        }

        // First reduce to device 0
        let mut result = buffers[0].clone();
        self.reduce(buffers, &mut result, self.devices[0].id(), op)
            .await?;

        // Then broadcast to all devices
        for (i, device) in self.devices.iter().enumerate() {
            if i > 0 {
                self.copy_between_devices(
                    &result,
                    &mut buffers[i],
                    self.devices[0].id(),
                    device.id(),
                )
                .await?;
            } else {
                buffers[0].copy_from(&result)?;
            }
        }

        Ok(())
    }

    /// Synchronize all devices
    pub fn synchronize_all(&self) -> CudaResult<()> {
        for backend in &self.backends {
            backend.synchronize()?;
        }
        Ok(())
    }

    /// Check if two devices can access each other's memory
    fn can_access_peer(device1: usize, device2: usize) -> CudaResult<bool> {
        let mut can_access: i32 = 0;
        unsafe {
            let result = cuda_sys::cudaDeviceCanAccessPeer(
                &mut can_access as *mut _,
                device1 as i32,
                device2 as i32,
            );
            if result != crate::cuda::cudaSuccess {
                return Err(CudaError::Context {
                    message: format!(
                        "Failed to check peer access between devices {} and {}",
                        device1, device2
                    ),
                });
            }
        }
        Ok(can_access != 0)
    }

    /// Enable peer access between two devices
    fn enable_peer_access(src_device: usize, dst_device: usize) -> CudaResult<()> {
        // Set current device
        crate::cuda::set_device(src_device)?;

        unsafe {
            let result = ::cuda_sys::cudart::cudaDeviceEnablePeerAccess(dst_device as i32, 0);
            // Ignore error if peer access is already enabled
            if result != crate::cuda::cudaSuccess
                && result != ::cuda_sys::cudart::cudaError_t::PeerAccessAlreadyEnabled
            {
                return Err(CudaError::Context {
                    message: format!(
                        "Failed to enable peer access from device {} to device {}",
                        src_device, dst_device
                    ),
                });
            }
        }
        Ok(())
    }
}

/// Reduction operations for multi-GPU collectives
#[derive(Debug, Clone, Copy)]
pub enum ReduceOp {
    Sum,
    Product,
    Min,
    Max,
    Average,
}

/// Data parallel model wrapper for multi-GPU training
pub struct DataParallel<M> {
    module: M,
    device_ids: Vec<usize>,
    output_device: usize,
    dim: usize,
    context: Arc<MultiGpuContext>,
}

impl<M> DataParallel<M> {
    /// Create a new data parallel wrapper
    pub fn new(
        module: M,
        device_ids: Vec<usize>,
        output_device: Option<usize>,
        dim: usize,
    ) -> CudaResult<Self> {
        let output_device = output_device.unwrap_or_else(|| device_ids[0]);
        let context = Arc::new(MultiGpuContext::new(device_ids.clone())?);

        Ok(Self {
            module,
            device_ids,
            output_device,
            dim,
            context,
        })
    }

    /// Get the module reference
    pub fn module(&self) -> &M {
        &self.module
    }

    /// Get the module mutable reference
    pub fn module_mut(&mut self) -> &mut M {
        &mut self.module
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::DType;

    #[test]
    fn test_multi_gpu_context_creation() {
        if crate::is_available() && crate::cuda::device_count().unwrap_or(0) > 1 {
            let context = MultiGpuContext::new(vec![0, 1]);
            assert!(context.is_ok());

            let context = context.unwrap();
            assert_eq!(context.num_devices(), 2);
        }
    }

    #[tokio::test]
    async fn test_peer_to_peer_copy() {
        if crate::is_available() && crate::cuda::device_count().unwrap_or(0) > 1 {
            let context = MultiGpuContext::new(vec![0, 1]).unwrap();

            // Create buffers on different devices
            let backend0 = context.backend(0).unwrap();
            let backend1 = context.backend(1).unwrap();

            crate::cuda::set_device(0).unwrap();
            let mut src = backend0.create_buffer::<f32>(1024, DType::F32).unwrap();

            crate::cuda::set_device(1).unwrap();
            let mut dst = backend1.create_buffer::<f32>(1024, DType::F32).unwrap();

            // Initialize source buffer
            let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
            src.copy_from_host(&data).unwrap();

            // Copy between devices
            context
                .copy_between_devices(&src, &mut dst, 0, 1)
                .await
                .unwrap();

            // Verify
            let mut result = vec![0.0; 1024];
            dst.copy_to_host(&mut result).unwrap();
            assert_eq!(result, data);
        }
    }

    #[tokio::test]
    async fn test_all_reduce() {
        if crate::is_available() && crate::cuda::device_count().unwrap_or(0) > 1 {
            let context = MultiGpuContext::new(vec![0, 1]).unwrap();

            // Create buffers on each device
            let mut buffers = Vec::new();
            for (i, backend) in context.backends.iter().enumerate() {
                crate::cuda::set_device(context.devices[i].id()).unwrap();
                let mut buffer = backend.create_buffer::<f32>(4, DType::F32).unwrap();

                // Initialize with device-specific values
                let data = vec![1.0 + i as f32; 4];
                buffer.copy_from_host(&data).unwrap();
                buffers.push(buffer);
            }

            // All-reduce sum
            context
                .all_reduce(&mut buffers, ReduceOp::Sum)
                .await
                .unwrap();

            // Verify - each element should be sum of all device values
            for buffer in &buffers {
                let mut result = vec![0.0; 4];
                buffer.copy_to_host(&mut result).unwrap();
                assert_eq!(result, vec![3.0; 4]); // 1.0 + 2.0 = 3.0
            }
        }
    }
}
