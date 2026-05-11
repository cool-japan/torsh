//! Multi-GPU support for CUDA backend

use super::{CudaBackend, CudaBuffer, CudaDevice, CudaStream};
use crate::cuda::cuda_sys_compat as cuda_sys;
use crate::cuda::error::{CudaError, CudaResult};
use std::collections::HashMap;
use std::sync::Arc;

// ──────────────────────────────────────────────────────────────────────────────
// P3.1 — Type-safe dispatch trait
// ──────────────────────────────────────────────────────────────────────────────

/// Marker trait for GPU-reducible element types.
///
/// Provides type-safe element-wise operations used by ring all-reduce without
/// any `unsafe { mem::transmute }` or `mem::forget` hacks.  For now the
/// arithmetic is performed on the CPU side (GPU kernels require actual CUDA
/// hardware present); the trait is still generic enough for future GPU kernels
/// to call.
pub trait ReducibleElement: Clone + Send + Sync + 'static + std::fmt::Debug {
    /// Apply the given reduction operation element-wise to two values.
    fn apply_reduce_op(a: Self, b: Self, op: ReduceOp) -> Self;
    /// Scale a value by a floating-point factor (used for Average).
    fn scale(val: Self, factor: f64) -> Self;
    /// If `Self == f32`, return a view of the slice; otherwise `None`.
    fn as_f32_slice(data: &[Self]) -> Option<&[f32]>;
    /// If `Self == f64`, return a view of the slice; otherwise `None`.
    fn as_f64_slice(data: &[Self]) -> Option<&[f64]>;
}

impl ReducibleElement for f32 {
    fn apply_reduce_op(a: Self, b: Self, op: ReduceOp) -> Self {
        match op {
            ReduceOp::Sum => a + b,
            ReduceOp::Product => a * b,
            ReduceOp::Min => if a < b { a } else { b },
            ReduceOp::Max => if a > b { a } else { b },
            // Average: accumulate as sum first; division by N is applied
            // once at the end of the scatter-reduce phase by the caller.
            ReduceOp::Average => a + b,
        }
    }

    fn scale(val: Self, factor: f64) -> Self {
        val * (factor as f32)
    }

    fn as_f32_slice(data: &[Self]) -> Option<&[f32]> {
        Some(data)
    }

    fn as_f64_slice(_data: &[Self]) -> Option<&[f64]> {
        None
    }
}

impl ReducibleElement for f64 {
    fn apply_reduce_op(a: Self, b: Self, op: ReduceOp) -> Self {
        match op {
            ReduceOp::Sum => a + b,
            ReduceOp::Product => a * b,
            ReduceOp::Min => if a < b { a } else { b },
            ReduceOp::Max => if a > b { a } else { b },
            ReduceOp::Average => a + b,
        }
    }

    fn scale(val: Self, factor: f64) -> Self {
        val * factor
    }

    fn as_f32_slice(_data: &[Self]) -> Option<&[f32]> {
        None
    }

    fn as_f64_slice(data: &[Self]) -> Option<&[f64]> {
        Some(data)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// P3.3 — Ring all-reduce (pure-CPU simulation + CUDA path stub)
// ──────────────────────────────────────────────────────────────────────────────

/// Bandwidth-optimal ring all-reduce over a set of per-device host-side
/// buffers.
///
/// The algorithm runs in two phases, each with `N-1` steps where `N` is the
/// number of devices:
///
/// **Scatter-reduce phase** — each device sends one chunk of its data to its
/// right neighbour and receives a chunk from its left neighbour.  The received
/// chunk is reduced element-wise into the local copy.  After this phase every
/// device owns one chunk that is fully reduced across all devices.
///
/// **All-gather phase** — each device forwards its fully-reduced chunk to its
/// right neighbour.  After this phase every device holds the same fully-reduced
/// data.
///
/// For `Average` the final division by `N` is applied once to every element
/// after the scatter-reduce phase.
///
/// # Arguments
/// * `buffers` — one mutable `Vec<T>` per device; all must have equal length.
/// * `op` — the commutative, associative reduction to apply.
pub fn ring_all_reduce<T: ReducibleElement>(
    buffers: &mut [Vec<T>],
    op: ReduceOp,
) -> CudaResult<()> {
    let n = buffers.len();
    if n == 0 {
        return Err(CudaError::InvalidDevice { device_id: 0 });
    }
    if n == 1 {
        // Single device: nothing to communicate.
        return Ok(());
    }

    let len = buffers[0].len();
    // All buffers must be the same length.
    if buffers.iter().any(|b| b.len() != len) {
        return Err(CudaError::UnsupportedOperation {
            op: "ring_all_reduce".to_string(),
            dtype: "mismatched buffer lengths".to_string(),
        });
    }
    if len == 0 {
        return Ok(());
    }

    // Divide data into N chunks.  The last chunk absorbs any remainder.
    let base_chunk = len / n;
    let remainder = len % n;

    // Chunk range helper: chunk index → (start, end) in the buffer.
    let chunk_range = |chunk_idx: usize| -> (usize, usize) {
        let start = chunk_idx * base_chunk + chunk_idx.min(remainder);
        let extra = if chunk_idx < remainder { 1 } else { 0 };
        let end = start + base_chunk + extra;
        (start, end)
    };

    // ── Phase 1: Scatter-reduce ───────────────────────────────────────────────
    // After step `s`, device `i` has accumulated data for chunk `(i - s) % N`.
    // We simulate the ring by iterating steps and performing the sends/receives
    // in-order (no actual concurrency needed for a CPU simulation).
    for step in 0..n - 1 {
        // Each device `i` sends chunk `(i + n - step) % n` to device `(i + 1) % n`
        // and receives from device `(i + n - 1) % n`.
        // Process right-to-left to avoid clobbering data before it is read.
        for i in (0..n).rev() {
            let send_chunk = (i + n - step) % n;
            let recv_from = (i + n - 1) % n;
            let (s_start, s_end) = chunk_range(send_chunk);
            // Extract the data to send from `recv_from`'s buffer (that device
            // is the *sender* for device `i` in this step).
            let send_data: Vec<T> = buffers[recv_from][s_start..s_end].to_vec();
            // Reduce into device `i`'s copy.
            for (local_el, incoming) in buffers[i][s_start..s_end].iter_mut().zip(send_data.iter()) {
                *local_el = T::apply_reduce_op(local_el.clone(), incoming.clone(), op);
            }
        }
    }

    // If averaging, divide every element by N once (sum → mean).
    if matches!(op, ReduceOp::Average) {
        let inv = 1.0_f64 / n as f64;
        for buf in buffers.iter_mut() {
            for el in buf.iter_mut() {
                *el = T::scale(el.clone(), inv);
            }
        }
    }

    // ── Phase 2: All-gather ───────────────────────────────────────────────────
    // After scatter-reduce, device `i` holds the correct data for chunk `i`.
    // We broadcast each reduced chunk around the ring so that every device
    // ends up with all chunks.
    for step in 0..n - 1 {
        // Each device `i` sends chunk `(i + 1 + n - step) % n` to `(i+1) % n`.
        for i in (0..n).rev() {
            let send_chunk = (i + 1 + n - step) % n;
            let recv_from = (i + n - 1) % n;
            let (s_start, s_end) = chunk_range(send_chunk);
            let send_data: Vec<T> = buffers[recv_from][s_start..s_end].to_vec();
            // Overwrite (no reduction — data is already fully reduced).
            buffers[i][s_start..s_end].clone_from_slice(&send_data);
        }
    }

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Multi-GPU context
// ──────────────────────────────────────────────────────────────────────────────

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

    /// Reduce data from all devices to one.
    ///
    /// All `ReduceOp` variants are now supported.  The implementation copies
    /// each source buffer to the destination device and applies the operation
    /// element-wise on the host for correctness (GPU kernels require actual
    /// CUDA hardware).
    pub async fn reduce<T: ReducibleElement + Default>(
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

        // Bring the first buffer onto the destination device as a host Vec.
        let src0_device = self.devices[0].id();
        let mut acc: Vec<T> = {
            let mut tmp: Vec<T> = vec![Default::default(); src_buffers[0].len()];
            if src0_device == dst_device {
                src_buffers[0].copy_to_host(&mut tmp)?;
            } else {
                // Copy through host memory — already handled by copy_to_host.
                src_buffers[0].copy_to_host(&mut tmp)?;
            }
            tmp
        };

        // Accumulate the remaining buffers one by one.
        for src in src_buffers.iter().skip(1) {
            let mut incoming: Vec<T> = vec![Default::default(); src.len()];
            src.copy_to_host(&mut incoming)?;
            for (a, b) in acc.iter_mut().zip(incoming.iter()) {
                *a = T::apply_reduce_op(a.clone(), b.clone(), op);
            }
        }

        // For Average, divide by device count once.
        if matches!(op, ReduceOp::Average) {
            let inv = 1.0_f64 / self.devices.len() as f64;
            for el in acc.iter_mut() {
                *el = T::scale(el.clone(), inv);
            }
        }

        dst.copy_from_host(&acc)?;
        Ok(())
    }

    /// All-reduce operation across all devices.
    ///
    /// Uses the bandwidth-optimal ring all-reduce algorithm (`ring_all_reduce`)
    /// for 2+ devices and a trivial no-op for a single device.  All
    /// `ReduceOp` variants are supported via the `ReducibleElement` trait,
    /// completely eliminating the previous `unsafe { mem::transmute }` and
    /// `mem::forget` hacks.
    ///
    /// The function works on host-side staging buffers (one per device) and
    /// writes the result back to each `CudaBuffer`.  When running on real CUDA
    /// hardware the per-device `CudaBuffer`s are staged through host memory;
    /// this keeps the implementation sound without P2P for types that do not
    /// have a CUDA kernel.
    pub async fn all_reduce<T: ReducibleElement + Default>(
        &self,
        buffers: &mut [CudaBuffer<T>],
        op: ReduceOp,
    ) -> CudaResult<()> {
        if buffers.len() != self.devices.len() {
            return Err(CudaError::InvalidDevice {
                device_id: buffers.len(),
            });
        }

        let n = self.devices.len();
        let buf_len = if !buffers.is_empty() { buffers[0].len() } else { 0 };

        if n == 1 || buf_len == 0 {
            // Nothing to do.
            return Ok(());
        }

        // Stage all device buffers onto host.
        let mut host_bufs: Vec<Vec<T>> = buffers
            .iter()
            .map(|b| {
                let mut v = vec![Default::default(); b.len()];
                b.copy_to_host(&mut v)?;
                Ok(v)
            })
            .collect::<CudaResult<_>>()?;

        // Perform ring all-reduce entirely on host vectors.
        ring_all_reduce(&mut host_bufs, op)?;

        // Write results back to each device buffer.
        for (buf, host) in buffers.iter_mut().zip(host_bufs.iter()) {
            buf.copy_from_host(host)?;
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

// ──────────────────────────────────────────────────────────────────────────────
// ReduceOp enum — P3.2 all variants
// ──────────────────────────────────────────────────────────────────────────────

/// Reduction operations for multi-GPU collectives.
///
/// All variants are now supported by `ring_all_reduce` and `all_reduce`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReduceOp {
    /// Element-wise sum across devices.
    Sum,
    /// Element-wise product across devices.
    Product,
    /// Element-wise minimum across devices.
    Min,
    /// Element-wise maximum across devices.
    Max,
    /// Element-wise average (sum / device_count) across devices.
    Average,
}

// ──────────────────────────────────────────────────────────────────────────────
// DataParallel wrapper
// ──────────────────────────────────────────────────────────────────────────────

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

// ──────────────────────────────────────────────────────────────────────────────
// Existing GPU integration tests (require real CUDA hardware)
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::DType;

    #[test]
    fn test_multi_gpu_context_creation() {
        if crate::is_available() && crate::cuda::device_count().unwrap_or(0) > 1 {
            let context = MultiGpuContext::new(vec![0, 1]);
            assert!(context.is_ok());

            let context = context.expect("operation should succeed");
            assert_eq!(context.num_devices(), 2);
        }
    }

    #[tokio::test]
    async fn test_peer_to_peer_copy() {
        if crate::is_available() && crate::cuda::device_count().unwrap_or(0) > 1 {
            let context =
                MultiGpuContext::new(vec![0, 1]).expect("Multi Gpu Context should succeed");

            // Create buffers on different devices
            let backend0 = context.backend(0).expect("backend should be initialized");
            let backend1 = context.backend(1).expect("backend should be initialized");

            crate::cuda::set_device(0).expect("cuda should succeed");
            let mut src = backend0
                .create_buffer::<f32>(1024, DType::F32)
                .expect("operation should succeed");

            crate::cuda::set_device(1).expect("cuda should succeed");
            let mut dst = backend1
                .create_buffer::<f32>(1024, DType::F32)
                .expect("operation should succeed");

            // Initialize source buffer
            let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
            src.copy_from_host(&data)
                .expect("copy from host memory should succeed");

            // Copy between devices
            context
                .copy_between_devices(&src, &mut dst, 0, 1)
                .await
                .expect("operation should succeed");

            // Verify
            let mut result = vec![0.0; 1024];
            dst.copy_to_host(&mut result)
                .expect("copy to host memory should succeed");
            assert_eq!(result, data);
        }
    }

    /// Updated to use the new ring all-reduce path.
    #[tokio::test]
    #[ignore = "Requires 2+ CUDA GPUs"]
    async fn test_all_reduce() {
        if crate::is_available() && crate::cuda::device_count().unwrap_or(0) > 1 {
            let context =
                MultiGpuContext::new(vec![0, 1]).expect("Multi Gpu Context should succeed");

            // Create buffers on each device
            let mut buffers = Vec::new();
            for (i, backend) in context.backends.iter().enumerate() {
                crate::cuda::set_device(context.devices[i].id()).expect("operation should succeed");
                let mut buffer = backend
                    .create_buffer::<f32>(4, DType::F32)
                    .expect("operation should succeed");

                // Initialize with device-specific values
                let data = vec![1.0 + i as f32; 4];
                buffer
                    .copy_from_host(&data)
                    .expect("copy from host memory should succeed");
                buffers.push(buffer);
            }

            // All-reduce sum via ring algorithm
            context
                .all_reduce(&mut buffers, ReduceOp::Sum)
                .await
                .expect("operation should succeed");

            // Verify — each element should be sum of all device values (1.0 + 2.0 = 3.0)
            for buffer in &buffers {
                let mut result = vec![0.0; 4];
                buffer
                    .copy_to_host(&mut result)
                    .expect("copy to host memory should succeed");
                assert_eq!(result, vec![3.0; 4]);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// P3.4 — Comprehensive CPU-side tests (no CUDA hardware required)
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests_p3 {
    use super::*;

    const EPS_F32: f32 = 1e-6;
    const EPS_F64: f64 = 1e-12;

    // ── ReducibleElement for f32 ───────────────────────────────────────────────

    #[test]
    fn test_reducible_f32_sum() {
        assert!((f32::apply_reduce_op(2.0_f32, 3.0_f32, ReduceOp::Sum) - 5.0).abs() < EPS_F32);
        assert!((f32::apply_reduce_op(-1.0_f32, 1.0_f32, ReduceOp::Sum) - 0.0).abs() < EPS_F32);
    }

    #[test]
    fn test_reducible_f32_product() {
        assert!(
            (f32::apply_reduce_op(2.0_f32, 3.0_f32, ReduceOp::Product) - 6.0).abs() < EPS_F32
        );
        assert!(
            (f32::apply_reduce_op(-2.0_f32, 4.0_f32, ReduceOp::Product) - (-8.0)).abs() < EPS_F32
        );
    }

    #[test]
    fn test_reducible_f32_min() {
        assert!((f32::apply_reduce_op(3.0_f32, 7.0_f32, ReduceOp::Min) - 3.0).abs() < EPS_F32);
        assert!((f32::apply_reduce_op(-5.0_f32, 2.0_f32, ReduceOp::Min) - (-5.0)).abs() < EPS_F32);
    }

    #[test]
    fn test_reducible_f32_max() {
        assert!((f32::apply_reduce_op(3.0_f32, 7.0_f32, ReduceOp::Max) - 7.0).abs() < EPS_F32);
        assert!((f32::apply_reduce_op(-5.0_f32, 2.0_f32, ReduceOp::Max) - 2.0).abs() < EPS_F32);
    }

    #[test]
    fn test_reducible_f32_mean() {
        // Average op accumulates as sum; the caller applies the /N divisor.
        let acc = f32::apply_reduce_op(3.0_f32, 5.0_f32, ReduceOp::Average);
        assert!((acc - 8.0).abs() < EPS_F32, "accumulation should be sum before /N");
        let scaled = f32::scale(8.0_f32, 0.5_f64);
        assert!((scaled - 4.0).abs() < EPS_F32, "scale by 0.5 → 4.0");
    }

    #[test]
    fn test_reducible_f32_type_slices() {
        let v: Vec<f32> = vec![1.0, 2.0, 3.0];
        assert!(f32::as_f32_slice(&v).is_some());
        assert!(f32::as_f64_slice(&v).is_none());
    }

    // ── ReducibleElement for f64 ───────────────────────────────────────────────

    #[test]
    fn test_reducible_f64_all_ops() {
        // Sum
        assert!((f64::apply_reduce_op(1.5_f64, 2.5_f64, ReduceOp::Sum) - 4.0).abs() < EPS_F64);
        // Product
        assert!((f64::apply_reduce_op(3.0_f64, 4.0_f64, ReduceOp::Product) - 12.0).abs() < EPS_F64);
        // Min
        assert!((f64::apply_reduce_op(10.0_f64, -3.0_f64, ReduceOp::Min) - (-3.0)).abs() < EPS_F64);
        // Max
        assert!((f64::apply_reduce_op(10.0_f64, -3.0_f64, ReduceOp::Max) - 10.0).abs() < EPS_F64);
        // Average accumulation
        let acc = f64::apply_reduce_op(6.0_f64, 2.0_f64, ReduceOp::Average);
        assert!((acc - 8.0).abs() < EPS_F64);
        let scaled = f64::scale(acc, 0.5_f64);
        assert!((scaled - 4.0).abs() < EPS_F64);
        // Type slice checks
        let v: Vec<f64> = vec![1.0, 2.0];
        assert!(f64::as_f64_slice(&v).is_some());
        assert!(f64::as_f32_slice(&v).is_none());
    }

    // ── ring_all_reduce correctness ────────────────────────────────────────────

    /// 4 devices, each holding [1.0, 2.0, 3.0, 4.0].
    /// After sum all-reduce every device should hold [4.0, 8.0, 12.0, 16.0].
    #[test]
    fn test_ring_all_reduce_correctness() {
        let n = 4usize;
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut buffers: Vec<Vec<f32>> = (0..n).map(|_| data.clone()).collect();

        ring_all_reduce(&mut buffers, ReduceOp::Sum).expect("ring_all_reduce should succeed");

        let expected: Vec<f32> = vec![4.0, 8.0, 12.0, 16.0];
        for (dev_idx, buf) in buffers.iter().enumerate() {
            for (i, (&got, &exp)) in buf.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < EPS_F32,
                    "device {dev_idx} element {i}: got {got}, expected {exp}"
                );
            }
        }
    }

    /// 3 devices with different starting values, verifying product correctness.
    ///
    /// Device 0: [1.0, 2.0, 3.0]
    /// Device 1: [4.0, 5.0, 6.0]
    /// Device 2: [7.0, 8.0, 9.0]
    /// Expected product: [28.0, 80.0, 162.0]
    #[test]
    fn test_ring_all_reduce_product() {
        let mut buffers: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        ring_all_reduce(&mut buffers, ReduceOp::Product).expect("ring_all_reduce should succeed");

        let expected: Vec<f32> = vec![28.0, 80.0, 162.0];
        for (dev_idx, buf) in buffers.iter().enumerate() {
            for (i, (&got, &exp)) in buf.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < EPS_F32,
                    "device {dev_idx} element {i}: got {got}, expected {exp}"
                );
            }
        }
    }

    /// 4 devices with varying element values, verifying min correctness.
    ///
    /// Each device holds a permutation; global min per position:
    ///   pos 0: min(3,1,4,2) = 1
    ///   pos 1: min(9,2,7,5) = 2
    #[test]
    fn test_ring_all_reduce_min() {
        let mut buffers: Vec<Vec<f32>> = vec![
            vec![3.0, 9.0],
            vec![1.0, 2.0],
            vec![4.0, 7.0],
            vec![2.0, 5.0],
        ];

        ring_all_reduce(&mut buffers, ReduceOp::Min).expect("ring_all_reduce should succeed");

        let expected: Vec<f32> = vec![1.0, 2.0];
        for (dev_idx, buf) in buffers.iter().enumerate() {
            for (i, (&got, &exp)) in buf.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < EPS_F32,
                    "device {dev_idx} element {i}: got {got}, expected {exp}"
                );
            }
        }
    }

    /// 2 devices, average of [2.0, 4.0, 6.0] and [4.0, 8.0, 12.0] = [3.0, 6.0, 9.0]
    #[test]
    fn test_ring_all_reduce_average() {
        let mut buffers: Vec<Vec<f32>> = vec![
            vec![2.0, 4.0, 6.0],
            vec![4.0, 8.0, 12.0],
        ];

        ring_all_reduce(&mut buffers, ReduceOp::Average).expect("ring_all_reduce should succeed");

        let expected: Vec<f32> = vec![3.0, 6.0, 9.0];
        for (dev_idx, buf) in buffers.iter().enumerate() {
            for (i, (&got, &exp)) in buf.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < EPS_F32,
                    "device {dev_idx} element {i}: got {got}, expected {exp}"
                );
            }
        }
    }

    /// Single-device degenerate case — buffer must be unchanged.
    #[test]
    fn test_ring_all_reduce_single_device() {
        let mut buffers: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0]];
        ring_all_reduce(&mut buffers, ReduceOp::Sum).expect("ring_all_reduce should succeed");
        assert_eq!(buffers[0], vec![1.0, 2.0, 3.0]);
    }

    /// f64 all-reduce with max operation.
    #[test]
    fn test_ring_all_reduce_f64_max() {
        let mut buffers: Vec<Vec<f64>> = vec![
            vec![1.0, 5.0, 2.0],
            vec![3.0, 2.0, 8.0],
            vec![2.0, 9.0, 1.0],
        ];

        ring_all_reduce(&mut buffers, ReduceOp::Max).expect("ring_all_reduce should succeed");

        let expected: Vec<f64> = vec![3.0, 9.0, 8.0];
        for (dev_idx, buf) in buffers.iter().enumerate() {
            for (i, (&got, &exp)) in buf.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < EPS_F64,
                    "device {dev_idx} element {i}: got {got}, expected {exp}"
                );
            }
        }
    }

    /// Non-power-of-two buffer length to exercise the remainder chunk handling.
    #[test]
    fn test_ring_all_reduce_nonaligned_length() {
        // 3 devices, 7-element buffers (7 / 3 = 2 rem 1)
        let mut buffers: Vec<Vec<f32>> = vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            vec![3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
        ];

        ring_all_reduce(&mut buffers, ReduceOp::Sum).expect("ring_all_reduce should succeed");

        let expected = vec![6.0_f32; 7];
        for (dev_idx, buf) in buffers.iter().enumerate() {
            for (i, (&got, &exp)) in buf.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < EPS_F32,
                    "device {dev_idx} element {i}: got {got}, expected {exp}"
                );
            }
        }
    }

    // ── GPU tests (require actual CUDA hardware) ───────────────────────────────

    #[test]
    #[ignore = "Requires 2+ CUDA GPUs"]
    fn test_gpu_all_reduce_sum() {
        // This test requires real CUDA hardware with at least 2 devices.
        // It is intentionally left as a skeleton; exercise via `cargo nextest run
        // --ignored` on a machine with multiple GPUs.
        use torsh_core::DType;
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime should start");
        rt.block_on(async {
            if !crate::is_available() || crate::cuda::device_count().unwrap_or(0) < 2 {
                return;
            }
            let ctx = MultiGpuContext::new(vec![0, 1]).expect("context");
            let mut bufs: Vec<_> = ctx
                .backends
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    crate::cuda::set_device(ctx.devices[i].id()).expect("set device");
                    let mut buf = b.create_buffer::<f32>(4, DType::F32).expect("create buffer");
                    buf.copy_from_host(&vec![1.0_f32 + i as f32; 4]).expect("copy");
                    buf
                })
                .collect();
            ctx.all_reduce(&mut bufs, ReduceOp::Sum).await.expect("all_reduce");
            for buf in &bufs {
                let mut v = vec![0.0_f32; 4];
                buf.copy_to_host(&mut v).expect("copy to host");
                for &x in &v {
                    assert!((x - 3.0).abs() < EPS_F32);
                }
            }
        });
    }
}
