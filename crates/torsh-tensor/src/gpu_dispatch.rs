//! GPU compute dispatch for ToRSh, backed by oxicuda's `ComputeBackend` trait.
//!
//! This module is the migration target that replaces the former
//! `scirs2_core::gpu` dependency.  ToRSh tensors keep owning all
//! dtype / shape / autograd semantics; this layer only marshals the `f32`
//! payload of CUDA-device tensors through the flat
//! [`oxicuda_backend::ComputeBackend`] op interface using the
//! host → device → op → host transfer model (the same per-op transfer model
//! the previous `scirs2_core::gpu` path used).
//!
//! ## Backend selection
//!
//! The concrete GPU backend (oxicuda's `CudaBackend`, which delegates to
//! `oxicuda-driver` / `oxicuda-blas` / `oxicuda-dnn`) lives in the `oxicuda`
//! *umbrella* crate.  At the time of writing the umbrella is not yet published
//! at `0.3` on crates.io (only the leaf crates are), so [`active_backend`]
//! returns `None` and every dispatch declines — callers then fall back to
//! ToRSh's native CPU / SIMD implementations.  This is the **same observable
//! behaviour** as the previous `scirs2_core::gpu` path, whose
//! `GpuContext::new(GpuBackend::Cuda)` also failed and fell back to the CPU.
//!
//! Wiring the real CUDA backend is a one-step change once the umbrella
//! publishes — see the `TODO(oxicuda-umbrella-0.3)` block in
//! [`active_backend`].
//!
//! The marshalling helpers ([`run_unary_f32`], [`run_binary_f32`]) are
//! exercised against the real [`oxicuda_backend::CpuBackend`] in the unit
//! tests, so the op mapping and buffer handling are verified end-to-end even
//! without a GPU present.

use crate::{Tensor, TensorElement};
use oxicuda_backend::{BackendResult, ComputeBackend};

// Re-export the op vocabulary so call sites depend on `crate::gpu_dispatch::*`
// rather than naming the `oxicuda_backend` crate directly.
pub use oxicuda_backend::{BinaryOp, ReduceOp, UnaryOp};

/// View an `f32` slice as its raw native-endian bytes (plain-old-data reinterpret).
#[inline]
fn f32_as_bytes(data: &[f32]) -> &[u8] {
    // SAFETY: `f32` is plain-old-data; every byte of its representation is a
    // valid `u8`.  The returned slice borrows `data` for the same lifetime and
    // spans exactly `size_of_val(data)` bytes.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), std::mem::size_of_val(data)) }
}

/// Decode native-endian `f32` values from a byte buffer.
#[inline]
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Allocate one device buffer per entry in `sizes`, run `f` with the resulting
/// pointers, then free every buffer (even on error).  If an allocation fails
/// partway through, the already-allocated buffers are freed before returning.
fn with_buffers<R>(
    backend: &dyn ComputeBackend,
    sizes: &[usize],
    f: impl FnOnce(&[u64]) -> BackendResult<R>,
) -> BackendResult<R> {
    let mut ptrs: Vec<u64> = Vec::with_capacity(sizes.len());
    for &size in sizes {
        match backend.alloc(size) {
            Ok(ptr) => ptrs.push(ptr),
            Err(err) => {
                for &ptr in &ptrs {
                    let _ = backend.free(ptr);
                }
                return Err(err);
            }
        }
    }
    let result = f(&ptrs);
    for &ptr in &ptrs {
        let _ = backend.free(ptr);
    }
    result
}

/// Execute a unary `f32` op on `input` through `backend`
/// (alloc → copy-in → op → copy-out → free).
fn run_unary_f32(
    backend: &dyn ComputeBackend,
    op: UnaryOp,
    input: &[f32],
) -> BackendResult<Vec<f32>> {
    let n = input.len();
    let bytes = std::mem::size_of_val(input);
    with_buffers(backend, &[bytes, bytes], |ptrs| {
        let (input_ptr, output_ptr) = (ptrs[0], ptrs[1]);
        backend.copy_htod(input_ptr, f32_as_bytes(input))?;
        backend.unary(op, input_ptr, output_ptr, n)?;
        let mut out = vec![0u8; bytes];
        backend.copy_dtoh(&mut out, output_ptr)?;
        Ok(bytes_to_f32(&out))
    })
}

/// Execute a binary elementwise `f32` op on `(a, b)` through `backend`.
///
/// `a` and `b` must have equal length.
fn run_binary_f32(
    backend: &dyn ComputeBackend,
    op: BinaryOp,
    a: &[f32],
    b: &[f32],
) -> BackendResult<Vec<f32>> {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let bytes = std::mem::size_of_val(a);
    with_buffers(backend, &[bytes, bytes, bytes], |ptrs| {
        let (a_ptr, b_ptr, out_ptr) = (ptrs[0], ptrs[1], ptrs[2]);
        backend.copy_htod(a_ptr, f32_as_bytes(a))?;
        backend.copy_htod(b_ptr, f32_as_bytes(b))?;
        backend.binary(op, a_ptr, b_ptr, out_ptr, n)?;
        let mut out = vec![0u8; bytes];
        backend.copy_dtoh(&mut out, out_ptr)?;
        Ok(bytes_to_f32(&out))
    })
}

/// Return the active GPU compute backend, or `None` when no GPU backend is
/// available in this build / environment (in which case callers fall back to
/// their native CPU implementation).
fn active_backend() -> Option<&'static dyn ComputeBackend> {
    // Real CUDA path: ToRSh's own thin `CudaBackend` over the oxicuda leaf
    // crates (built lazily, once). It is adopted only when a physical GPU
    // device was found during init; otherwise we return None and callers use
    // their native CPU path.
    #[cfg(feature = "cuda")]
    {
        use crate::cuda_backend::CudaBackend;
        use std::sync::OnceLock;

        static BACKEND: OnceLock<Option<CudaBackend>> = OnceLock::new();
        let backend = BACKEND.get_or_init(|| {
            let mut backend = CudaBackend::new();
            backend.init().ok()?;
            if backend.has_gpu_context() {
                Some(backend)
            } else {
                None
            }
        });
        return backend.as_ref().map(|b| b as &dyn ComputeBackend);
    }

    // No CUDA feature: no GPU backend is wired, so decline (CPU fallback).
    #[cfg(not(feature = "cuda"))]
    None
}

/// Attempt a unary activation on the GPU for `f32` CUDA tensors.
///
/// Returns `Some(result)` only when **all** of the following hold:
/// 1. `T == f32`,
/// 2. the tensor lives on a CUDA device, and
/// 3. a GPU backend is available and the dispatch succeeds end-to-end.
///
/// Otherwise returns `None`, signalling the caller to use its CPU path.
pub fn try_unary_f32<T: TensorElement>(input: &Tensor<T>, op: UnaryOp) -> Option<Tensor<T>> {
    use std::any::TypeId;

    if TypeId::of::<T>() != TypeId::of::<f32>() {
        return None;
    }
    if !matches!(input.device, crate::DeviceType::Cuda(_)) {
        return None;
    }
    let backend = active_backend()?;

    let data = input.data().ok()?;
    // SAFETY: the `TypeId` guard above guarantees `T == f32`, so `&[T]` is `&[f32]`.
    let f32_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), data.len()) };
    let result_f32 = run_unary_f32(backend, op, f32_slice).ok()?;

    // SAFETY: `T == f32` (confirmed by `TypeId`), so `Vec<f32>` is `Vec<T>`.
    let result_t: Vec<T> = unsafe {
        let mut v = std::mem::ManuallyDrop::new(result_f32);
        Vec::from_raw_parts(v.as_mut_ptr().cast::<T>(), v.len(), v.capacity())
    };
    Tensor::<T>::from_data(result_t, input.shape().dims().to_vec(), input.device).ok()
}

/// Attempt a binary elementwise op on the GPU for equal-shaped `f32` CUDA tensors.
///
/// Same gating as [`try_unary_f32`]; returns `None` (CPU fallback) unless both
/// tensors are `f32`, on a CUDA device, equal length, and a GPU backend
/// dispatches successfully.
pub fn try_binary_f32<T: TensorElement>(
    lhs: &Tensor<T>,
    rhs: &Tensor<T>,
    op: BinaryOp,
) -> Option<Tensor<T>> {
    use std::any::TypeId;

    if TypeId::of::<T>() != TypeId::of::<f32>() {
        return None;
    }
    if !matches!(lhs.device, crate::DeviceType::Cuda(_)) {
        return None;
    }
    let backend = active_backend()?;

    let lhs_data = lhs.data().ok()?;
    let rhs_data = rhs.data().ok()?;
    if lhs_data.len() != rhs_data.len() {
        return None;
    }
    // SAFETY: the `TypeId` guard guarantees `T == f32`, so `&[T]` is `&[f32]`.
    let lhs_f32: &[f32] =
        unsafe { std::slice::from_raw_parts(lhs_data.as_ptr().cast::<f32>(), lhs_data.len()) };
    let rhs_f32: &[f32] =
        unsafe { std::slice::from_raw_parts(rhs_data.as_ptr().cast::<f32>(), rhs_data.len()) };
    let result_f32 = run_binary_f32(backend, op, lhs_f32, rhs_f32).ok()?;

    // SAFETY: `T == f32` (confirmed by `TypeId`), so `Vec<f32>` is `Vec<T>`.
    let result_t: Vec<T> = unsafe {
        let mut v = std::mem::ManuallyDrop::new(result_f32);
        Vec::from_raw_parts(v.as_mut_ptr().cast::<T>(), v.len(), v.capacity())
    };
    Tensor::<T>::from_data(result_t, lhs.shape().dims().to_vec(), lhs.device).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxicuda_backend::CpuBackend;

    #[test]
    fn unary_relu_through_compute_backend() {
        let mut backend = CpuBackend::new();
        backend.init().expect("backend init");
        let out = run_unary_f32(&backend, UnaryOp::Relu, &[-2.0, -0.5, 0.0, 1.5, 3.0])
            .expect("relu dispatch");
        assert_eq!(out, vec![0.0, 0.0, 0.0, 1.5, 3.0]);
        // Every scratch buffer must have been released.
        assert_eq!(backend.live_allocations(), 0);
    }

    #[test]
    fn unary_sigmoid_through_compute_backend() {
        let mut backend = CpuBackend::new();
        backend.init().expect("backend init");
        let out = run_unary_f32(&backend, UnaryOp::Sigmoid, &[0.0]).expect("sigmoid dispatch");
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert_eq!(backend.live_allocations(), 0);
    }

    #[test]
    fn binary_add_and_mul_through_compute_backend() {
        let mut backend = CpuBackend::new();
        backend.init().expect("backend init");
        let a = [1.0f32, 2.0, 3.0];
        let b = [10.0f32, 20.0, 30.0];
        assert_eq!(
            run_binary_f32(&backend, BinaryOp::Add, &a, &b).expect("add dispatch"),
            vec![11.0, 22.0, 33.0]
        );
        assert_eq!(
            run_binary_f32(&backend, BinaryOp::Mul, &a, &b).expect("mul dispatch"),
            vec![10.0, 40.0, 90.0]
        );
        assert_eq!(backend.live_allocations(), 0);
    }

    #[test]
    fn cpu_tensor_declines_gpu_dispatch() {
        // An f32 CPU tensor must never take the GPU path (device guard).
        let tensor = Tensor::from_data(vec![1.0f32, -1.0], vec![2], crate::DeviceType::Cpu)
            .expect("tensor creation");
        assert!(try_unary_f32(&tensor, UnaryOp::Relu).is_none());
    }

    // ── Real-GPU end-to-end tests (require an actual CUDA device) ────────────
    // These run only with `--features cuda`. When no device is present they
    // skip (the dispatch declines), so they are safe on CPU-only CI too.

    #[cfg(feature = "cuda")]
    #[test]
    fn unary_relu_runs_on_real_gpu() {
        let Some(backend) = active_backend() else {
            eprintln!("no CUDA device available; skipping real-GPU relu test");
            return;
        };
        let input = vec![-2.0f32, -0.5, 0.0, 1.5, 3.0, -7.0, 4.0, 0.25];
        // Call the helper directly so a backend error is surfaced (not swallowed).
        let got = run_unary_f32(backend, UnaryOp::Relu, &input).expect("GPU relu dispatch failed");
        let expect: Vec<f32> = input.iter().map(|&x| x.max(0.0)).collect();
        assert_eq!(got, expect, "GPU relu result must match CPU reference");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tensor_path_unary_dispatches_to_gpu() {
        if active_backend().is_none() {
            return;
        }
        // Realistic path: f32 Cuda tensor -> try_unary_f32 -> GPU. `expect`
        // (not silent CPU fallback) proves the GPU dispatch actually succeeded.
        let t = Tensor::from_data(
            vec![-1.0f32, 2.0, -3.0, 4.0],
            vec![4],
            crate::DeviceType::Cuda(0),
        )
        .expect("cuda tensor");
        let out = try_unary_f32(&t, UnaryOp::Relu).expect("Tensor GPU path returned None");
        assert_eq!(out.to_vec().expect("vec"), vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn binary_add_runs_on_real_gpu() {
        let Some(backend) = active_backend() else {
            return;
        };
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [10.0f32, 20.0, 30.0, 40.0];
        let got = run_binary_f32(backend, BinaryOp::Add, &a, &b).expect("GPU add dispatch failed");
        assert_eq!(got, vec![11.0, 22.0, 33.0, 44.0]);
    }
}
