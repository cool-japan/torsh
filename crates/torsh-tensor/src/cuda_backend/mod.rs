//! ToRSh-owned thin CUDA backend implementing [`oxicuda_backend::ComputeBackend`].
//!
//! This is a lean adapter built directly on the oxicuda *leaf* crates
//! (`oxicuda-driver` / `oxicuda-launch` / `oxicuda-ptx`) so ToRSh avoids
//! depending on the large `oxicuda` umbrella facade (which unconditionally
//! compiles unrelated modules: collective / distributed / pipeline_parallel /
//! tensor_backend / …). The elementwise / reduction compute path is ported
//! from oxicuda's umbrella `backend` module — see [`ptx_ops`].
//!
//! Scope: real GPU `unary` / `binary` / `reduce` (PTX), plus device memory
//! management (`alloc` / `free` / `copy_htod` / `copy_dtoh`) and
//! `synchronize`. `gemm` / `conv2d_forward` / `attention` return
//! [`BackendError::Unsupported`] for now.
//!
//! ## Context model
//!
//! `init` creates **one persistent regular [`Context`]** and keeps it for the
//! backend's lifetime; every allocation and every kernel launch runs inside
//! that single context, so device pointers are valid across all operations and
//! kernels can access the memory. Per launch we create a cheap throwaway
//! [`Stream`](oxicuda_driver::Stream) in that context — *not* a throwaway
//! context. (Creating/destroying a CUDA **context** per launch, as some
//! reference code does, churns context state and yields "invalid context" on
//! the second op; a stream is cheap and safe to recreate.) The CUDA
//! current-context is per-thread, so [`activate_gpu`](CudaBackend::activate_gpu)
//! re-binds the context before every operation.

use std::sync::{Arc, Mutex};

use oxicuda_backend::{
    BackendError, BackendResult, BackendTranspose, BinaryOp, ComputeBackend, ReduceOp, UnaryOp,
};
use oxicuda_driver::device::Device;
use oxicuda_driver::ffi::CUdeviceptr;
use oxicuda_driver::loader::try_driver;
use oxicuda_driver::Context;

mod ptx_ops;

/// ToRSh's CUDA compute backend.
///
/// On a host without a CUDA GPU, [`init`](ComputeBackend::init) still succeeds
/// (the type is always constructible) but GPU operations return
/// [`BackendError::DeviceError`] / `NotInitialized`.
#[derive(Debug)]
pub(crate) struct CudaBackend {
    initialized: bool,
    /// The single persistent context that owns all device memory and kernels.
    /// `Some` once a GPU was found during [`init`](ComputeBackend::init).
    context: Mutex<Option<Arc<Context>>>,
}

impl CudaBackend {
    /// Create a new, uninitialized CUDA backend.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            initialized: false,
            context: Mutex::new(None),
        }
    }

    /// Returns `true` if a live GPU context was created during [`init`](ComputeBackend::init).
    pub(crate) fn has_gpu_context(&self) -> bool {
        self.context.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    /// Check that the backend is initialized.
    fn check_init(&self) -> BackendResult<()> {
        if self.initialized {
            Ok(())
        } else {
            Err(BackendError::NotInitialized)
        }
    }

    /// Re-binds the persistent context on the calling thread (CUDA's
    /// current-context is per-thread) and returns a clone of it.
    pub(super) fn activate_gpu(&self) -> BackendResult<Arc<Context>> {
        let guard = self
            .context
            .lock()
            .map_err(|_| BackendError::DeviceError("backend context lock poisoned".into()))?;
        let ctx = guard
            .as_ref()
            .ok_or_else(|| BackendError::DeviceError("no CUDA GPU context available".into()))?;
        ctx.set_current()
            .map_err(|e| BackendError::DeviceError(e.to_string()))?;
        Ok(Arc::clone(ctx))
    }
}

impl Default for CudaBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeBackend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    fn init(&mut self) -> BackendResult<()> {
        if self.initialized {
            return Ok(());
        }
        // Attempt real CUDA initialisation. On a host without a GPU the driver
        // library is absent and we degrade gracefully: the backend is marked
        // initialised, but GPU operations return an error.
        if oxicuda_driver::init().is_ok() {
            if let Ok(dev) = Device::get(0) {
                // `Context::new` creates a regular context and makes it current.
                if let Ok(ctx) = Context::new(&dev) {
                    if let Ok(mut guard) = self.context.lock() {
                        *guard = Some(Arc::new(ctx));
                    }
                }
            }
        }
        self.initialized = true;
        Ok(())
    }

    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn gemm(
        &self,
        _trans_a: BackendTranspose,
        _trans_b: BackendTranspose,
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f64,
        _a_ptr: u64,
        _lda: usize,
        _b_ptr: u64,
        _ldb: usize,
        _beta: f64,
        _c_ptr: u64,
        _ldc: usize,
    ) -> BackendResult<()> {
        self.check_init()?;
        // TODO(torsh-cuda-gemm): wire `oxicuda-blas` here (the umbrella's
        // gemm_impl pattern) when ToRSh's GPU matmul path is implemented.
        Err(BackendError::Unsupported(
            "gemm is not yet implemented in ToRSh's thin CUDA backend".into(),
        ))
    }

    fn conv2d_forward(
        &self,
        _input_ptr: u64,
        _input_shape: &[usize],
        _filter_ptr: u64,
        _filter_shape: &[usize],
        _output_ptr: u64,
        _output_shape: &[usize],
        _stride: &[usize],
        _padding: &[usize],
    ) -> BackendResult<()> {
        self.check_init()?;
        // TODO(torsh-cuda-dnn): wire `oxicuda-dnn` if GPU conv is ever needed here.
        Err(BackendError::Unsupported(
            "conv2d_forward is not implemented in ToRSh's thin CUDA backend".into(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        _q_ptr: u64,
        _k_ptr: u64,
        _v_ptr: u64,
        _o_ptr: u64,
        _batch: usize,
        _heads: usize,
        _seq_q: usize,
        _seq_kv: usize,
        _head_dim: usize,
        _scale: f64,
        _causal: bool,
    ) -> BackendResult<()> {
        self.check_init()?;
        Err(BackendError::Unsupported(
            "attention is not implemented in ToRSh's thin CUDA backend".into(),
        ))
    }

    fn reduce(
        &self,
        op: ReduceOp,
        input_ptr: u64,
        output_ptr: u64,
        shape: &[usize],
        axis: usize,
    ) -> BackendResult<()> {
        self.check_init()?;
        if shape.is_empty() {
            return Err(BackendError::InvalidArgument(
                "shape must not be empty".into(),
            ));
        }
        if axis >= shape.len() {
            return Err(BackendError::InvalidArgument(format!(
                "axis {} out of bounds for shape with {} dimensions",
                axis,
                shape.len()
            )));
        }
        ptx_ops::reduce_axis(self, op, input_ptr, output_ptr, shape, axis)
    }

    fn unary(&self, op: UnaryOp, input_ptr: u64, output_ptr: u64, n: usize) -> BackendResult<()> {
        self.check_init()?;
        if n == 0 {
            return Ok(());
        }
        ptx_ops::unary_elementwise(self, op, input_ptr, output_ptr, n)
    }

    fn binary(
        &self,
        op: BinaryOp,
        a_ptr: u64,
        b_ptr: u64,
        output_ptr: u64,
        n: usize,
    ) -> BackendResult<()> {
        self.check_init()?;
        if n == 0 {
            return Ok(());
        }
        ptx_ops::binary_elementwise(self, op, a_ptr, b_ptr, output_ptr, n)
    }

    fn synchronize(&self) -> BackendResult<()> {
        self.check_init()?;
        if !self.has_gpu_context() {
            return Ok(());
        }
        let ctx = self.activate_gpu()?;
        ctx.synchronize()
            .map_err(|e| BackendError::DeviceError(e.to_string()))
    }

    fn alloc(&self, bytes: usize) -> BackendResult<u64> {
        self.check_init()?;
        if bytes == 0 {
            return Err(BackendError::InvalidArgument(
                "cannot allocate 0 bytes".into(),
            ));
        }
        self.activate_gpu()?; // ensure the owning context is current on this thread
        let api = try_driver().map_err(|e| BackendError::DeviceError(e.to_string()))?;
        let mut ptr: CUdeviceptr = 0;
        oxicuda_driver::error::check(unsafe { (api.cu_mem_alloc_v2)(&mut ptr, bytes) }).map_err(
            |e| match e {
                oxicuda_driver::CudaError::OutOfMemory => BackendError::OutOfMemory,
                other => BackendError::DeviceError(other.to_string()),
            },
        )?;
        Ok(ptr)
    }

    fn free(&self, ptr: u64) -> BackendResult<()> {
        self.check_init()?;
        self.activate_gpu()?;
        let api = try_driver().map_err(|e| BackendError::DeviceError(e.to_string()))?;
        oxicuda_driver::error::check(unsafe { (api.cu_mem_free_v2)(ptr) })
            .map_err(|e| BackendError::DeviceError(e.to_string()))
    }

    fn copy_htod(&self, dst: u64, src: &[u8]) -> BackendResult<()> {
        self.check_init()?;
        if src.is_empty() {
            return Ok(());
        }
        self.activate_gpu()?;
        let api = try_driver().map_err(|e| BackendError::DeviceError(e.to_string()))?;
        oxicuda_driver::error::check(unsafe {
            (api.cu_memcpy_htod_v2)(dst, src.as_ptr().cast(), src.len())
        })
        .map_err(|e| BackendError::DeviceError(e.to_string()))
    }

    fn copy_dtoh(&self, dst: &mut [u8], src: u64) -> BackendResult<()> {
        self.check_init()?;
        if dst.is_empty() {
            return Ok(());
        }
        self.activate_gpu()?;
        let api = try_driver().map_err(|e| BackendError::DeviceError(e.to_string()))?;
        oxicuda_driver::error::check(unsafe {
            (api.cu_memcpy_dtoh_v2)(dst.as_mut_ptr().cast(), src, dst.len())
        })
        .map_err(|e| BackendError::DeviceError(e.to_string()))
    }
}
