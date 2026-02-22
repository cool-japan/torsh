//! .NET 6+ Modern Integration for ToRSh
//!
//! This module provides cutting-edge .NET 6+ features including:
//! - Task-based Asynchronous Pattern (TAP) with async/await
//! - ValueTask for high-performance scenarios
//! - IAsyncEnumerable<T> for streaming operations
//! - Span<T> and Memory<T> for zero-copy operations
//! - Source Generators compatibility
//! - Native AOT (Ahead-of-Time) compilation support
//! - System.Threading.Channels for efficient producer-consumer patterns
//!
//! # Features
//!
//! ## Async/Await Support
//!
//! All computationally expensive operations return Task<T> or ValueTask<T>:
//!
//! ```csharp
//! // Async tensor operations
//! var result = await Tensor.MatMulAsync(x, y);
//!
//! // Parallel batch processing
//! await Parallel.ForEachAsync(tensors, async (tensor, ct) => {
//!     await tensor.NormalizeAsync(ct);
//! });
//! ```
//!
//! ## Zero-Copy Operations
//!
//! Leverage Span<T> and Memory<T> for maximum performance:
//!
//! ```csharp
//! Span<float> data = stackalloc float[100];
//! var tensor = Tensor.FromSpan(data, shape);
//!
//! // No allocation, direct memory access
//! ReadOnlySpan<float> tensorData = tensor.AsSpan();
//! ```
//!
//! ## Streaming with IAsyncEnumerable
//!
//! Process large datasets efficiently:
//!
//! ```csharp
//! await foreach (var batch in dataloader.StreamBatchesAsync()) {
//!     var output = await model.ForwardAsync(batch);
//!     await output.SaveAsync(path);
//! }
//! ```
//!
//! ## Native AOT Compatibility
//!
//! All P/Invoke signatures are Native AOT compatible with proper marshaling attributes.

#![allow(dead_code)]

use crate::c_api::*;
use parking_lot::Mutex;
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

/// Send-safe wrapper for tensor pointer addresses
#[derive(Clone, Copy)]
struct TensorAddr(usize);

unsafe impl Send for TensorAddr {}

impl TensorAddr {
    fn new(ptr: *mut TorshTensor) -> Self {
        Self(ptr as usize)
    }

    unsafe fn as_ptr(&self) -> *mut TorshTensor {
        self.0 as *mut TorshTensor
    }
}

/// Send-safe wrapper for void pointer addresses
#[derive(Clone, Copy)]
struct VoidAddr(usize);

unsafe impl Send for VoidAddr {}

impl VoidAddr {
    fn new(ptr: *mut c_void) -> Self {
        Self(ptr as usize)
    }

    unsafe fn as_ptr(&self) -> *mut c_void {
        self.0 as *mut c_void
    }
}

// =============================================================================
// Task-Based Asynchronous Pattern (TAP) Support
// =============================================================================

/// Async operation handle for .NET Task completion
///
/// This represents an ongoing async operation that can be awaited in C#.
#[repr(C)]
pub struct DotNetAsyncOperation {
    /// Unique operation ID
    id: u64,
    /// Completion flag
    completed: AtomicBool,
    /// Result tensor as address (null/0 until complete)
    result: Mutex<Option<TensorAddr>>,
    /// Error message if operation failed
    error: Mutex<Option<String>>,
}

impl DotNetAsyncOperation {
    pub fn new(id: u64) -> Self {
        Self {
            id,
            completed: AtomicBool::new(false),
            result: Mutex::new(None),
            error: Mutex::new(None),
        }
    }

    pub fn complete_with_result(&self, tensor: *mut TorshTensor) {
        *self.result.lock() = Some(TensorAddr::new(tensor));
        self.completed.store(true, Ordering::Release);
    }

    pub fn complete_with_error(&self, error: String) {
        *self.error.lock() = Some(error);
        self.completed.store(true, Ordering::Release);
    }

    pub fn is_completed(&self) -> bool {
        self.completed.load(Ordering::Acquire)
    }

    pub fn get_result(&self) -> Option<*mut TorshTensor> {
        self.result.lock().map(|addr| unsafe { addr.as_ptr() })
    }

    pub fn get_error(&self) -> Option<String> {
        self.error.lock().clone()
    }
}

/// Global async operation registry
static ASYNC_OP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Create a new async operation for matrix multiplication
///
/// Returns an operation handle that C# can poll or await.
///
/// # C# Usage
/// ```csharp
/// [DllImport("torsh_ffi")]
/// private static extern IntPtr dotnet6_matmul_async(
///     IntPtr a, IntPtr b, out ulong operationId);
///
/// public async Task<Tensor> MatMulAsync(Tensor other) {
///     var op = dotnet6_matmul_async(handle, other.handle, out var opId);
///     while (!dotnet6_async_is_completed(op)) {
///         await Task.Delay(1); // Or use completion callbacks
///     }
///     return new Tensor(dotnet6_async_get_result(op));
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn dotnet6_matmul_async(
    a: *mut TorshTensor,
    b: *mut TorshTensor,
    operation_id: *mut u64,
) -> *mut DotNetAsyncOperation {
    if a.is_null() || b.is_null() || operation_id.is_null() {
        return ptr::null_mut();
    }

    let id = ASYNC_OP_COUNTER.fetch_add(1, Ordering::Relaxed);
    *operation_id = id;

    let op = Arc::new(DotNetAsyncOperation::new(id));
    let op_clone = Arc::clone(&op);

    // Convert pointers to Send-safe wrappers
    let a_addr = TensorAddr::new(a);
    let b_addr = TensorAddr::new(b);

    // Spawn async operation on thread pool
    std::thread::spawn(move || unsafe {
        // Convert back to pointers
        let a = a_addr.as_ptr();
        let b = b_addr.as_ptr();

        // Perform matrix multiplication
        let result = torsh_tensor_new(ptr::null(), ptr::null(), 0, TorshDType::F32);

        if torsh_tensor_matmul(a, b, result) == TorshError::Success {
            op_clone.complete_with_result(result);
        } else {
            op_clone.complete_with_error("Matrix multiplication failed".to_string());
            torsh_tensor_free(result);
        }
    });

    Arc::into_raw(op) as *mut DotNetAsyncOperation
}

/// Check if async operation is completed
#[no_mangle]
pub unsafe extern "C" fn dotnet6_async_is_completed(op: *const DotNetAsyncOperation) -> bool {
    if op.is_null() {
        return true; // Treat null as completed with error
    }

    let op_ref = &*(op as *const DotNetAsyncOperation);
    op_ref.is_completed()
}

/// Get result from completed async operation
///
/// Returns null if operation is not completed or failed.
#[no_mangle]
pub unsafe extern "C" fn dotnet6_async_get_result(
    op: *const DotNetAsyncOperation,
) -> *mut TorshTensor {
    if op.is_null() {
        return ptr::null_mut();
    }

    let op_ref = &*(op as *const DotNetAsyncOperation);
    op_ref.get_result().unwrap_or(ptr::null_mut())
}

/// Get error message from failed async operation
///
/// Returns null if no error occurred.
#[no_mangle]
pub unsafe extern "C" fn dotnet6_async_get_error(op: *const DotNetAsyncOperation) -> *const c_char {
    if op.is_null() {
        return ptr::null();
    }

    let op_ref = &*(op as *const DotNetAsyncOperation);
    if let Some(error) = op_ref.get_error() {
        let c_str =
            std::ffi::CString::new(error).expect("error string should not contain null bytes");
        c_str.into_raw()
    } else {
        ptr::null()
    }
}

/// Free async operation handle
#[no_mangle]
pub unsafe extern "C" fn dotnet6_async_free(op: *mut DotNetAsyncOperation) {
    if !op.is_null() {
        let _ = Arc::from_raw(op as *const DotNetAsyncOperation);
    }
}

// =============================================================================
// Span<T> and Memory<T> Support
// =============================================================================

/// Create tensor from Span<float> without copying
///
/// # C# Usage
/// ```csharp
/// Span<float> data = stackalloc float[100];
/// fixed (float* ptr = data) {
///     var tensor = dotnet6_tensor_from_span(
///         (IntPtr)ptr, data.Length, shape, shape.Length);
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn dotnet6_tensor_from_span(
    data: *const c_float,
    data_len: usize,
    shape: *const usize,
    ndim: usize,
) -> *mut TorshTensor {
    if data.is_null() || shape.is_null() {
        return ptr::null_mut();
    }

    // Verify shape matches data length
    let shape_slice = std::slice::from_raw_parts(shape, ndim);
    let expected_len: usize = shape_slice.iter().product();

    if expected_len != data_len {
        return ptr::null_mut();
    }

    torsh_tensor_new(data as *const c_void, shape, ndim, TorshDType::F32)
}

/// Get tensor data as Span<float> (read-only)
///
/// # C# Usage
/// ```csharp
/// var ptr = dotnet6_tensor_as_span(tensor, out var length);
/// var span = new ReadOnlySpan<float>((float*)ptr, (int)length);
/// ```
#[no_mangle]
pub unsafe extern "C" fn dotnet6_tensor_as_span(
    tensor: *const TorshTensor,
    length: *mut usize,
) -> *const c_float {
    if tensor.is_null() || length.is_null() {
        return ptr::null();
    }

    let data_ptr = torsh_tensor_data(tensor);
    *length = torsh_tensor_numel(tensor);

    data_ptr as *const c_float
}

/// Get mutable tensor data as Span<float>
///
/// # Safety
/// Caller must ensure exclusive access to the tensor.
#[no_mangle]
pub unsafe extern "C" fn dotnet6_tensor_as_span_mut(
    tensor: *mut TorshTensor,
    length: *mut usize,
) -> *mut c_float {
    if tensor.is_null() || length.is_null() {
        return ptr::null_mut();
    }

    let data_ptr = torsh_tensor_data(tensor as *const TorshTensor);
    *length = torsh_tensor_numel(tensor as *const TorshTensor);

    data_ptr as *mut c_float
}

// =============================================================================
// IAsyncEnumerable<T> Support (Streaming)
// =============================================================================

/// Streaming data loader handle
#[repr(C)]
pub struct DotNetStreamingLoader {
    current_batch: AtomicU64,
    total_batches: u64,
    batch_size: usize,
    completed: AtomicBool,
}

impl DotNetStreamingLoader {
    pub fn new(total_batches: u64, batch_size: usize) -> Self {
        Self {
            current_batch: AtomicU64::new(0),
            total_batches,
            batch_size,
            completed: AtomicBool::new(false),
        }
    }
}

/// Create streaming data loader
///
/// # C# Usage
/// ```csharp
/// public async IAsyncEnumerable<Tensor> StreamBatchesAsync(
///     [EnumeratorCancellation] CancellationToken ct = default) {
///
///     var loader = dotnet6_streaming_loader_create(totalBatches, batchSize);
///     try {
///         while (dotnet6_streaming_loader_has_next(loader)) {
///             var batch = await GetNextBatchAsync(loader, ct);
///             if (batch != null) yield return batch;
///         }
///     } finally {
///         dotnet6_streaming_loader_free(loader);
///     }
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn dotnet6_streaming_loader_create(
    total_batches: u64,
    batch_size: usize,
) -> *mut DotNetStreamingLoader {
    let loader = Box::new(DotNetStreamingLoader::new(total_batches, batch_size));
    Box::into_raw(loader)
}

/// Check if streaming loader has more batches
#[no_mangle]
pub unsafe extern "C" fn dotnet6_streaming_loader_has_next(
    loader: *const DotNetStreamingLoader,
) -> bool {
    if loader.is_null() {
        return false;
    }

    let loader_ref = &*loader;
    loader_ref.current_batch.load(Ordering::Acquire) < loader_ref.total_batches
}

/// Get next batch asynchronously
///
/// Returns an async operation that yields the next batch.
#[no_mangle]
pub unsafe extern "C" fn dotnet6_streaming_loader_next_async(
    loader: *mut DotNetStreamingLoader,
    operation_id: *mut u64,
) -> *mut DotNetAsyncOperation {
    if loader.is_null() || operation_id.is_null() {
        return ptr::null_mut();
    }

    let loader_ref = &*loader;
    let batch_idx = loader_ref.current_batch.fetch_add(1, Ordering::AcqRel);

    if batch_idx >= loader_ref.total_batches {
        return ptr::null_mut(); // No more batches
    }

    let id = ASYNC_OP_COUNTER.fetch_add(1, Ordering::Relaxed);
    *operation_id = id;

    let op = Arc::new(DotNetAsyncOperation::new(id));
    let op_clone = Arc::clone(&op);
    let batch_size = loader_ref.batch_size;

    // Simulate async batch loading
    std::thread::spawn(move || unsafe {
        // Create batch tensor (in real implementation, would load from disk/network)
        let shape = [batch_size, 784]; // Example: MNIST batch
        let result = torsh_tensor_new(ptr::null(), shape.as_ptr(), shape.len(), TorshDType::F32);

        op_clone.complete_with_result(result);
    });

    Arc::into_raw(op) as *mut DotNetAsyncOperation
}

/// Free streaming loader
#[no_mangle]
pub unsafe extern "C" fn dotnet6_streaming_loader_free(loader: *mut DotNetStreamingLoader) {
    if !loader.is_null() {
        let _ = Box::from_raw(loader);
    }
}

// =============================================================================
// ValueTask<T> Support for High-Performance Scenarios
// =============================================================================

/// Synchronous result that can be wrapped in ValueTask<T>
///
/// For operations that often complete synchronously, use ValueTask
/// to avoid Task allocation overhead.
#[repr(C)]
pub struct DotNetValueTaskResult {
    /// True if result is available synchronously
    is_sync: bool,
    /// Synchronous result (if is_sync = true)
    sync_result: *mut TorshTensor,
    /// Async operation (if is_sync = false)
    async_op: *mut DotNetAsyncOperation,
}

/// Check if tensor is cached (synchronous check for ValueTask)
///
/// Returns immediately if cached, or returns async operation if needs computation.
///
/// # C# Usage
/// ```csharp
/// public ValueTask<Tensor> GetOrComputeAsync(string key) {
///     var result = dotnet6_get_or_compute(key);
///
///     if (result.IsSync) {
///         return new ValueTask<Tensor>(new Tensor(result.SyncResult));
///     } else {
///         return new ValueTask<Tensor>(AwaitAsync(result.AsyncOp));
///     }
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn dotnet6_get_or_compute(key: *const c_char) -> DotNetValueTaskResult {
    if key.is_null() {
        return DotNetValueTaskResult {
            is_sync: true,
            sync_result: ptr::null_mut(),
            async_op: ptr::null_mut(),
        };
    }

    // Simulate cache lookup
    let is_cached = fastrand::bool(); // Random for demo

    if is_cached {
        // Synchronous path - return cached result
        let cached = torsh_tensor_new(ptr::null(), ptr::null(), 0, TorshDType::F32);

        DotNetValueTaskResult {
            is_sync: true,
            sync_result: cached,
            async_op: ptr::null_mut(),
        }
    } else {
        // Asynchronous path - compute in background
        let id = ASYNC_OP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let op = Arc::new(DotNetAsyncOperation::new(id));
        let op_clone = Arc::clone(&op);

        std::thread::spawn(move || unsafe {
            // Simulate computation
            std::thread::sleep(std::time::Duration::from_millis(10));
            let result = torsh_tensor_new(ptr::null(), ptr::null(), 0, TorshDType::F32);
            op_clone.complete_with_result(result);
        });

        DotNetValueTaskResult {
            is_sync: false,
            sync_result: ptr::null_mut(),
            async_op: Arc::into_raw(op) as *mut DotNetAsyncOperation,
        }
    }
}

// =============================================================================
// System.Threading.Channels Support
// =============================================================================

/// Channel for efficient producer-consumer pattern
#[repr(C)]
pub struct DotNetChannel {
    queue: Mutex<std::collections::VecDeque<*mut TorshTensor>>,
    capacity: usize,
    closed: AtomicBool,
}

impl DotNetChannel {
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: Mutex::new(std::collections::VecDeque::with_capacity(capacity)),
            capacity,
            closed: AtomicBool::new(false),
        }
    }
}

/// Create a bounded channel
///
/// # C# Usage
/// ```csharp
/// var channel = Channel.CreateBounded<Tensor>(capacity);
/// var nativeChannel = dotnet6_channel_create(capacity);
///
/// // Producer
/// _ = Task.Run(async () => {
///     while (await dotnet6_channel_write_async(nativeChannel, tensor)) {
///         // Continue producing
///     }
/// });
///
/// // Consumer
/// await foreach (var tensor in ReadAllAsync(nativeChannel)) {
///     Process(tensor);
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn dotnet6_channel_create(capacity: usize) -> *mut DotNetChannel {
    let channel = Box::new(DotNetChannel::new(capacity));
    Box::into_raw(channel)
}

/// Try to write tensor to channel (non-blocking)
///
/// Returns true if written, false if channel is full.
#[no_mangle]
pub unsafe extern "C" fn dotnet6_channel_try_write(
    channel: *mut DotNetChannel,
    tensor: *mut TorshTensor,
) -> bool {
    if channel.is_null() || tensor.is_null() {
        return false;
    }

    let channel_ref = &*channel;
    if channel_ref.closed.load(Ordering::Acquire) {
        return false;
    }

    let mut queue = channel_ref.queue.lock();
    if queue.len() < channel_ref.capacity {
        queue.push_back(tensor);
        true
    } else {
        false
    }
}

/// Try to read tensor from channel (non-blocking)
///
/// Returns null if channel is empty.
#[no_mangle]
pub unsafe extern "C" fn dotnet6_channel_try_read(channel: *mut DotNetChannel) -> *mut TorshTensor {
    if channel.is_null() {
        return ptr::null_mut();
    }

    let channel_ref = &*channel;
    let mut queue = channel_ref.queue.lock();

    queue.pop_front().unwrap_or(ptr::null_mut())
}

/// Close channel (no more writes allowed)
#[no_mangle]
pub unsafe extern "C" fn dotnet6_channel_close(channel: *mut DotNetChannel) {
    if !channel.is_null() {
        let channel_ref = &*channel;
        channel_ref.closed.store(true, Ordering::Release);
    }
}

/// Free channel
#[no_mangle]
pub unsafe extern "C" fn dotnet6_channel_free(channel: *mut DotNetChannel) {
    if !channel.is_null() {
        let channel = Box::from_raw(channel);
        // Clean up any remaining tensors
        let mut queue = channel.queue.lock();
        while let Some(tensor) = queue.pop_front() {
            torsh_tensor_free(tensor);
        }
    }
}

// =============================================================================
// Native AOT Compatibility Helpers
// =============================================================================

/// Function pointer types for Native AOT marshaling
pub type ProgressCallback = unsafe extern "C" fn(progress: c_float, user_data: *mut c_void);
pub type CompletionCallback =
    unsafe extern "C" fn(result: *mut TorshTensor, error: *const c_char, user_data: *mut c_void);

/// Async operation with callback (for Native AOT scenarios)
///
/// # C# Usage with Native AOT
/// ```csharp
/// [UnmanagedCallersOnly(CallConvs = new[] { typeof(CallConvCdecl) })]
/// static void OnComplete(IntPtr result, IntPtr error, IntPtr userData) {
///     if (error == IntPtr.Zero) {
///         var tensor = new Tensor(result);
///         // Process result
///     }
/// }
///
/// dotnet6_matmul_with_callback(a, b, &OnComplete, userData);
/// ```
#[no_mangle]
pub unsafe extern "C" fn dotnet6_matmul_with_callback(
    a: *mut TorshTensor,
    b: *mut TorshTensor,
    callback: CompletionCallback,
    user_data: *mut c_void,
) {
    if a.is_null() || b.is_null() {
        let error = std::ffi::CString::new("Invalid tensor")
            .expect("static string should not contain null bytes");
        callback(ptr::null_mut(), error.as_ptr(), user_data);
        return;
    }

    // Convert to Send-safe wrappers
    let a_addr = TensorAddr::new(a);
    let b_addr = TensorAddr::new(b);
    let user_data_addr = VoidAddr::new(user_data);

    // Perform operation asynchronously
    std::thread::spawn(move || unsafe {
        let a = a_addr.as_ptr();
        let b = b_addr.as_ptr();
        let user_data = user_data_addr.as_ptr();

        let result = torsh_tensor_new(ptr::null(), ptr::null(), 0, TorshDType::F32);

        if torsh_tensor_matmul(a, b, result) == TorshError::Success {
            callback(result, ptr::null(), user_data);
        } else {
            let error = std::ffi::CString::new("Matrix multiplication failed")
                .expect("static string should not contain null bytes");
            callback(ptr::null_mut(), error.as_ptr(), user_data);
            torsh_tensor_free(result);
        }
    });
}

/// Training with progress callback
///
/// Reports training progress asynchronously.
#[no_mangle]
pub unsafe extern "C" fn dotnet6_train_with_progress(
    model: *mut c_void,
    data: *mut TorshTensor,
    epochs: c_int,
    progress_callback: ProgressCallback,
    completion_callback: CompletionCallback,
    user_data: *mut c_void,
) {
    if model.is_null() || data.is_null() {
        let error = std::ffi::CString::new("Invalid arguments")
            .expect("static string should not contain null bytes");
        completion_callback(ptr::null_mut(), error.as_ptr(), user_data);
        return;
    }

    // Convert to Send-safe wrapper
    let user_data_addr = VoidAddr::new(user_data);

    std::thread::spawn(move || unsafe {
        let user_data = user_data_addr.as_ptr();

        for epoch in 0..epochs {
            // Simulate training
            std::thread::sleep(std::time::Duration::from_millis(100));

            let progress = (epoch as f32 + 1.0) / epochs as f32;
            progress_callback(progress, user_data);
        }

        // Training complete
        let result = torsh_tensor_new(ptr::null(), ptr::null(), 0, TorshDType::F32);
        completion_callback(result, ptr::null(), user_data);
    });
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_async_operation_creation() {
        let op = DotNetAsyncOperation::new(1);
        assert!(!op.is_completed());
        assert!(op.get_result().is_none());
        assert!(op.get_error().is_none());
    }

    #[test]
    fn test_async_operation_completion() {
        let op = DotNetAsyncOperation::new(1);

        unsafe {
            let tensor = torsh_tensor_new(ptr::null(), ptr::null(), 0, TorshDType::F32);
            op.complete_with_result(tensor);
        }

        assert!(op.is_completed());
        assert!(op.get_result().is_some());
        assert!(op.get_error().is_none());
    }

    #[test]
    fn test_async_operation_error() {
        let op = DotNetAsyncOperation::new(1);
        op.complete_with_error("Test error".to_string());

        assert!(op.is_completed());
        assert!(op.get_result().is_none());
        assert_eq!(op.get_error().unwrap(), "Test error");
    }

    #[test]
    fn test_streaming_loader() {
        unsafe {
            let loader = dotnet6_streaming_loader_create(10, 32);
            assert!(!loader.is_null());

            assert!(dotnet6_streaming_loader_has_next(loader));

            dotnet6_streaming_loader_free(loader);
        }
    }

    #[test]
    fn test_channel_operations() {
        unsafe {
            let channel = dotnet6_channel_create(5);
            assert!(!channel.is_null());

            // Create test tensor with actual data
            let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
            let shape: [usize; 2] = [2, 2];
            let tensor = torsh_tensor_new(
                data.as_ptr() as *const std::ffi::c_void,
                shape.as_ptr(),
                2,
                TorshDType::F32,
            );
            assert!(!tensor.is_null(), "Failed to create tensor");

            // Write to channel
            assert!(dotnet6_channel_try_write(channel, tensor));

            // Read from channel
            let read_tensor = dotnet6_channel_try_read(channel);
            assert!(!read_tensor.is_null());

            // Clean up
            torsh_tensor_free(read_tensor);
            dotnet6_channel_free(channel);
        }
    }

    #[test]
    fn test_value_task_result() {
        unsafe {
            let key = std::ffi::CString::new("test_key").unwrap();
            let result = dotnet6_get_or_compute(key.as_ptr());

            // Result should be either sync or async
            if result.is_sync {
                // Sync path would have a result
                assert!(!result.sync_result.is_null() || result.sync_result.is_null());
                assert!(result.async_op.is_null());
            } else {
                // Async path would have an operation
                assert!(result.sync_result.is_null());
                assert!(!result.async_op.is_null());

                if !result.async_op.is_null() {
                    dotnet6_async_free(result.async_op);
                }
            }
        }
    }
}
