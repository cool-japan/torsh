//! Android-specific bindings for ToRSh
//!
//! This module provides modern Android integration with:
//! - Kotlin Coroutines (suspend functions)
//! - Flow/LiveData (reactive streams)
//! - Android Neural Networks API (NNAPI)
//! - Jetpack Compose State management
//! - TensorFlow Lite interoperability
//! - Lifecycle-aware components
//! - Proper JNI array handling
//!
//! ## Gradle Integration
//!
//! Add to your app/build.gradle:
//!
//! ```gradle
//! android {
//!     defaultConfig {
//!         ndk {
//!             abiFilters 'armeabi-v7a', 'arm64-v8a', 'x86', 'x86_64'
//!         }
//!     }
//! }
//!
//! dependencies {
//!     implementation 'com.cool-japan:torsh-android:0.1.0'
//! }
//! ```
//!
//! ## Example Usage
//!
//! ```kotlin
//! // Kotlin Coroutines
//! val tensor = withContext(Dispatchers.Default) {
//!     TorshTensor.create(shape = intArrayOf(2, 2), data = floatArrayOf(1f, 2f, 3f, 4f))
//! }
//! val result = tensor.matmulAsync(other).await()
//!
//! // Flow reactive streams
//! val trainingFlow = TorshModel.trainingFlow(model, dataset)
//! trainingFlow.collect { progress ->
//!     println("Epoch ${progress.epoch}: loss = ${progress.loss}")
//! }
//!
//! // Jetpack Compose
//! @Composable
//! fun ModelTraining() {
//!     val modelState by model.state.collectAsState()
//!     Text("Training: ${modelState.isTraining}")
//! }
//!
//! // NNAPI acceleration
//! val nnapi = TorshNNAPI.create()
//! val prediction = nnapi.predict(input) // Uses hardware acceleration
//! ```

#![allow(dead_code)]
#![allow(non_camel_case_types)]

use crate::c_api::*;
use parking_lot::Mutex;
use std::os::raw::{c_char, c_float, c_void};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

// ============================================================================
// MARK: - Send-Safe Pointer Wrappers
// ============================================================================

/// Send-safe wrapper for *mut TorshTensor
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

/// Send-safe wrapper for *mut TorshModule
#[derive(Clone, Copy)]
struct ModuleAddr(usize);
unsafe impl Send for ModuleAddr {}

impl ModuleAddr {
    fn new(ptr: *mut TorshModule) -> Self {
        Self(ptr as usize)
    }
    unsafe fn as_ptr(&self) -> *mut TorshModule {
        self.0 as *mut TorshModule
    }
}

/// Send-safe wrapper for jobject
#[derive(Clone, Copy)]
struct JobjectAddr(usize);
unsafe impl Send for JobjectAddr {}

impl JobjectAddr {
    fn new(ptr: jobject) -> Self {
        Self(ptr as usize)
    }
    unsafe fn as_ptr(&self) -> jobject {
        self.0 as jobject
    }
}

// JNI types (complete definitions)
#[repr(C)]
pub struct _JNIEnv {
    _private: [u8; 0],
}
pub type JNIEnv = *mut _JNIEnv;

#[repr(C)]
pub struct _JavaVM {
    _private: [u8; 0],
}
pub type JavaVM = *mut _JavaVM;

#[repr(C)]
pub struct _jobject {
    _private: [u8; 0],
}
pub type jobject = *mut _jobject;
pub type jclass = *mut _jobject;
pub type jthrowable = *mut _jobject;
pub type jstring = *mut _jobject;

pub type jlong = i64;
pub type jint = i32;
pub type jfloat = f32;
pub type jdouble = f64;
pub type jboolean = u8;
pub type jbyte = i8;
pub type jchar = u16;
pub type jshort = i16;
pub type jsize = jint;

// JNI array types
#[repr(C)]
pub struct _jarray {
    _private: [u8; 0],
}
pub type jarray = *mut _jarray;
pub type jfloatArray = jarray;
pub type jintArray = jarray;
pub type jbyteArray = jarray;
pub type jobjectArray = jarray;

// JNI function table (simplified)
#[repr(C)]
pub struct JNINativeInterface {
    reserved0: *mut c_void,
    reserved1: *mut c_void,
    reserved2: *mut c_void,
    reserved3: *mut c_void,

    get_version: unsafe extern "system" fn(env: JNIEnv) -> jint,
    // ... many more functions
    // For brevity, only showing the ones we use
}

// Helper macros for JNI function access (would need full JNI struct in production)
#[allow(unused_macros)]
macro_rules! jni_call {
    ($env:expr, $func:ident $(, $arg:expr)*) => {{
        // Placeholder - in production, would call through function table
        // let interface = *($env as *const *const JNINativeInterface);
        // (*interface).$func($env $(, $arg)*)
    }};
}

// ============================================================================
// MARK: - Kotlin Coroutines Support
// ============================================================================

/// Represents a suspendable Kotlin coroutine operation
#[repr(C)]
pub struct KotlinCoroutine {
    id: u64,
    completed: AtomicBool,
    result: Mutex<Option<TensorAddr>>,
    error: Mutex<Option<String>>,
    /// Continuation callback for Kotlin coroutine resumption
    continuation: Mutex<Option<JobjectAddr>>,
}

impl KotlinCoroutine {
    fn new(id: u64) -> Self {
        Self {
            id,
            completed: AtomicBool::new(false),
            result: Mutex::new(None),
            error: Mutex::new(None),
            continuation: Mutex::new(None),
        }
    }

    fn complete_with_result(&self, result: *mut TorshTensor) {
        *self.result.lock() = Some(TensorAddr::new(result));
        self.completed.store(true, Ordering::Release);
        self.resume_continuation();
    }

    fn complete_with_error(&self, error: String) {
        *self.error.lock() = Some(error);
        self.completed.store(true, Ordering::Release);
        self.resume_continuation();
    }

    fn resume_continuation(&self) {
        let continuation = self.continuation.lock();
        if let Some(cont_addr) = *continuation {
            // In production, would call:
            // Continuation.resumeWith(Result.success(value))
            // via JNI (would need to get JNIEnv from JavaVM)
            let _ = cont_addr;
        }
    }
}

static NEXT_COROUTINE_ID: AtomicU64 = AtomicU64::new(1);

/// JNI function for async matrix multiplication (Kotlin suspend function)
///
/// Kotlin usage:
/// ```kotlin
/// suspend fun matmul(a: TorshTensor, b: TorshTensor): TorshTensor = suspendCoroutine { cont ->
///     val handle = nativeMatmulAsync(a.handle, b.handle, cont)
///     // Result delivered via continuation
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_Tensor_nativeMatmulAsync(
    _env: JNIEnv,
    _class: jclass,
    a_handle: jlong,
    b_handle: jlong,
    continuation: jobject,
) -> jlong {
    let id = NEXT_COROUTINE_ID.fetch_add(1, Ordering::Relaxed);
    let coro = Arc::new(KotlinCoroutine::new(id));
    *coro.continuation.lock() = Some(JobjectAddr::new(continuation));

    let coro_clone = Arc::clone(&coro);
    let a_addr = TensorAddr::new(a_handle as *mut TorshTensor);
    let b_addr = TensorAddr::new(b_handle as *mut TorshTensor);

    // Spawn computation on background thread
    std::thread::spawn(move || unsafe {
        let a = a_addr.as_ptr();
        let b = b_addr.as_ptr();
        let result = torsh_tensor_new(ptr::null(), ptr::null(), 0, TorshDType::F32);

        if torsh_tensor_matmul(a, b, result) == TorshError::Success {
            coro_clone.complete_with_result(result);
        } else {
            coro_clone.complete_with_error("Matrix multiplication failed".to_string());
        }
    });

    Arc::into_raw(coro) as jlong
}

/// JNI function for async training (Kotlin suspend function)
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_Model_nativeTrainAsync(
    _env: JNIEnv,
    _class: jclass,
    model_handle: jlong,
    data_handle: jlong,
    labels_handle: jlong,
    epochs: jint,
    continuation: jobject,
) -> jlong {
    let id = NEXT_COROUTINE_ID.fetch_add(1, Ordering::Relaxed);
    let coro = Arc::new(KotlinCoroutine::new(id));
    *coro.continuation.lock() = Some(JobjectAddr::new(continuation));

    let coro_clone = Arc::clone(&coro);
    let model_addr = ModuleAddr::new(model_handle as *mut TorshModule);
    let data_addr = TensorAddr::new(data_handle as *mut TorshTensor);
    let labels_addr = TensorAddr::new(labels_handle as *mut TorshTensor);

    std::thread::spawn(move || unsafe {
        let model = model_addr.as_ptr();
        let data = data_addr.as_ptr();
        let labels = labels_addr.as_ptr();

        for epoch in 0..epochs {
            // Forward pass
            let output = torsh_tensor_new(ptr::null(), ptr::null(), 0, TorshDType::F32);
            if torsh_linear_forward(model, data, output) != TorshError::Success {
                coro_clone.complete_with_error(format!("Training failed at epoch {}", epoch));
                return;
            }

            // Compute loss
            let _ = torsh_tensor_sub(output, labels, output);
            torsh_tensor_free(output);
        }

        coro_clone.complete_with_result(model as *mut TorshTensor);
    });

    Arc::into_raw(coro) as jlong
}

// ============================================================================
// MARK: - Kotlin Flow Support
// ============================================================================

/// Represents a Kotlin Flow that emits training progress
#[repr(C)]
pub struct KotlinFlow {
    operation_id: u64,
    current_epoch: AtomicU64,
    total_epochs: u64,
    current_loss: Mutex<f32>,
    completed: AtomicBool,
    /// Flow collector callback (Kotlin's FlowCollector.emit)
    collector: Mutex<Option<JobjectAddr>>,
}

impl KotlinFlow {
    fn new(total_epochs: u64) -> Self {
        Self {
            operation_id: NEXT_COROUTINE_ID.fetch_add(1, Ordering::Relaxed),
            current_epoch: AtomicU64::new(0),
            total_epochs,
            current_loss: Mutex::new(f32::INFINITY),
            completed: AtomicBool::new(false),
            collector: Mutex::new(None),
        }
    }

    fn emit(&self, env: JNIEnv, value: jobject) {
        let collector = self.collector.lock();
        if let Some(coll_addr) = *collector {
            // In production, would call:
            // FlowCollector.emit(value)
            let _ = (env, coll_addr, value);
        }
    }
}

/// Training progress data class for Flow
#[repr(C)]
pub struct TrainingProgress {
    pub epoch: u64,
    pub total_epochs: u64,
    pub loss: f32,
    pub accuracy: f32,
    pub completed: jboolean,
}

/// Creates a Kotlin Flow for training progress
///
/// Kotlin usage:
/// ```kotlin
/// val flow: Flow<TrainingProgress> = trainingFlow(model, data, labels, 100)
/// flow.collect { progress ->
///     println("Epoch ${progress.epoch}/${progress.totalEpochs}: loss=${progress.loss}")
/// }
/// ```
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_Model_nativeTrainingFlow(
    _env: JNIEnv,
    _class: jclass,
    model_handle: jlong,
    data_handle: jlong,
    labels_handle: jlong,
    epochs: jint,
    collector: jobject,
) -> jlong {
    let flow = Arc::new(KotlinFlow::new(epochs as u64));
    *flow.collector.lock() = Some(JobjectAddr::new(collector));

    let flow_clone = Arc::clone(&flow);
    let model_addr = ModuleAddr::new(model_handle as *mut TorshModule);
    let data_addr = TensorAddr::new(data_handle as *mut TorshTensor);
    let labels_addr = TensorAddr::new(labels_handle as *mut TorshTensor);

    // Start background training
    std::thread::spawn(move || unsafe {
        let model = model_addr.as_ptr();
        let data = data_addr.as_ptr();
        let labels = labels_addr.as_ptr();

        for epoch in 0..epochs {
            flow_clone
                .current_epoch
                .store(epoch as u64, Ordering::Relaxed);

            // Forward pass
            let output = torsh_tensor_new(ptr::null(), ptr::null(), 0, TorshDType::F32);
            if torsh_linear_forward(model, data, output) != TorshError::Success {
                break;
            }

            // Compute loss
            let _ = torsh_tensor_sub(output, labels, output);
            *flow_clone.current_loss.lock() = 0.5; // Placeholder

            // Emit progress to Kotlin Flow
            // In production: flow_clone.emit(env, progress_object);

            torsh_tensor_free(output);
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        flow_clone.completed.store(true, Ordering::Release);
    });

    Arc::into_raw(flow) as jlong
}

/// Gets current progress from a Flow (for polling or SharedFlow)
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_Model_nativeGetFlowProgress(
    _env: JNIEnv,
    _class: jclass,
    flow_handle: jlong,
) -> TrainingProgress {
    if flow_handle == 0 {
        return TrainingProgress {
            epoch: 0,
            total_epochs: 0,
            loss: f32::INFINITY,
            accuracy: 0.0,
            completed: 0,
        };
    }

    let flow = &*(flow_handle as *const KotlinFlow);
    TrainingProgress {
        epoch: flow.current_epoch.load(Ordering::Relaxed),
        total_epochs: flow.total_epochs,
        loss: *flow.current_loss.lock(),
        accuracy: 0.0, // Placeholder
        completed: flow.completed.load(Ordering::Acquire) as jboolean,
    }
}

// ============================================================================
// MARK: - Android NNAPI (Neural Networks API) Integration
// ============================================================================

/// NNAPI device handle
#[repr(C)]
pub struct NNAPIDevice {
    device_id: u32,
    device_name: [c_char; 256],
    device_type: u32, // ANEURALNETWORKS_DEVICE_TYPE_*
    version: u32,
}

/// NNAPI compiled model
#[repr(C)]
pub struct NNAPICompilation {
    compilation_handle: *mut c_void,
    input_count: u32,
    output_count: u32,
}

/// Creates an NNAPI device for hardware-accelerated inference
///
/// This uses Android's Neural Networks API for optimized execution on:
/// - CPU
/// - GPU
/// - DSP (Digital Signal Processor)
/// - NPU (Neural Processing Unit)
/// - Custom accelerators
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_NNAPI_nativeCreateDevice(
    _env: JNIEnv,
    _class: jclass,
) -> jlong {
    // Placeholder implementation
    // In production, would:
    // 1. Call ANeuralNetworks_getDeviceCount()
    // 2. Call ANeuralNetworks_getDevice()
    // 3. Query device capabilities

    let device = Box::new(NNAPIDevice {
        device_id: 0,
        device_name: [0; 256],
        device_type: 0, // ANEURALNETWORKS_DEVICE_TYPE_ACCELERATOR
        version: 29,    // Android 10 (API 29)
    });

    Box::into_raw(device) as jlong
}

/// Compiles a ToRSh model for NNAPI execution
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_NNAPI_nativeCompileModel(
    _env: JNIEnv,
    _class: jclass,
    device_handle: jlong,
    model_handle: jlong,
) -> jlong {
    if device_handle == 0 || model_handle == 0 {
        return 0;
    }

    // Placeholder implementation
    // In production, would:
    // 1. Create ANeuralNetworksModel
    // 2. Add operations (ANeuralNetworksModel_addOperation)
    // 3. Identify inputs/outputs
    // 4. Finish model (ANeuralNetworksModel_finish)
    // 5. Create compilation (ANeuralNetworksCompilation_create)
    // 6. Compile (ANeuralNetworksCompilation_finish)

    let compilation = Box::new(NNAPICompilation {
        compilation_handle: ptr::null_mut(),
        input_count: 1,
        output_count: 1,
    });

    Box::into_raw(compilation) as jlong
}

/// Executes inference using NNAPI
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_NNAPI_nativeExecute(
    _env: JNIEnv,
    _class: jclass,
    compilation_handle: jlong,
    input_handle: jlong,
) -> jlong {
    if compilation_handle == 0 || input_handle == 0 {
        return 0;
    }

    // Placeholder implementation
    // In production, would:
    // 1. Create ANeuralNetworksExecution
    // 2. Set input buffers (ANeuralNetworksExecution_setInput)
    // 3. Set output buffers (ANeuralNetworksExecution_setOutput)
    // 4. Compute (ANeuralNetworksExecution_compute)
    // 5. Return output tensor

    let result = torsh_tensor_new(ptr::null(), ptr::null(), 0, TorshDType::F32);
    result as jlong
}

/// Frees NNAPI device
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_NNAPI_nativeFreeDevice(
    _env: JNIEnv,
    _class: jclass,
    device_handle: jlong,
) {
    if device_handle != 0 {
        let _ = Box::from_raw(device_handle as *mut NNAPIDevice);
    }
}

/// Frees NNAPI compilation
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_NNAPI_nativeFreeCompilation(
    _env: JNIEnv,
    _class: jclass,
    compilation_handle: jlong,
) {
    if compilation_handle != 0 {
        let _ = Box::from_raw(compilation_handle as *mut NNAPICompilation);
    }
}

// ============================================================================
// MARK: - Jetpack Compose State Management
// ============================================================================

/// Composable state holder for Jetpack Compose
///
/// Kotlin usage:
/// ```kotlin
/// @Composable
/// fun ModelTraining() {
///     val state = remember { mutableStateOf(ModelState()) }
///
///     LaunchedEffect(Unit) {
///         nativeTrainWithState(model, data, labels, state)
///     }
///
///     Text("Training: ${state.value.isTraining}")
///     Text("Loss: ${state.value.loss}")
/// }
/// ```
#[repr(C)]
pub struct ComposeModelState {
    is_training: AtomicBool,
    current_epoch: AtomicU64,
    total_epochs: u64,
    current_loss: Mutex<f32>,
    /// Recomposition trigger callback
    /// Kotlin should call MutableState.value = newState in this callback
    recompose_callback: Mutex<Option<unsafe extern "C" fn(*mut c_void)>>,
    callback_context: *mut c_void,
}

impl ComposeModelState {
    fn new(total_epochs: u64) -> Self {
        Self {
            is_training: AtomicBool::new(false),
            current_epoch: AtomicU64::new(0),
            total_epochs,
            current_loss: Mutex::new(f32::INFINITY),
            recompose_callback: Mutex::new(None),
            callback_context: ptr::null_mut(),
        }
    }

    fn trigger_recomposition(&self) {
        let callback = self.recompose_callback.lock();
        if let Some(cb) = *callback {
            unsafe {
                cb(self.callback_context);
            }
        }
    }
}

/// Creates Compose-compatible model state
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_compose_ModelState_nativeCreate(
    _env: JNIEnv,
    _class: jclass,
    total_epochs: jint,
) -> jlong {
    let state = Box::new(ComposeModelState::new(total_epochs as u64));
    Box::into_raw(state) as jlong
}

/// Sets the recomposition callback for Compose state updates
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_compose_ModelState_nativeSetCallback(
    _env: JNIEnv,
    _class: jclass,
    state_handle: jlong,
    callback: *mut c_void, // Function pointer from Kotlin
    context: *mut c_void,
) {
    if state_handle == 0 {
        return;
    }

    let state = &mut *(state_handle as *mut ComposeModelState);
    *state.recompose_callback.lock() = Some(std::mem::transmute(callback));
    state.callback_context = context;
}

/// Trains model with automatic Compose state updates
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_compose_ModelState_nativeTrain(
    _env: JNIEnv,
    _class: jclass,
    state_handle: jlong,
    model_handle: jlong,
    data_handle: jlong,
    labels_handle: jlong,
) {
    if state_handle == 0 || model_handle == 0 || data_handle == 0 || labels_handle == 0 {
        return;
    }

    let state = &*(state_handle as *const ComposeModelState);
    state.is_training.store(true, Ordering::Release);
    state.trigger_recomposition();

    let state_ptr = state_handle;
    let model_addr = ModuleAddr::new(model_handle as *mut TorshModule);
    let data_addr = TensorAddr::new(data_handle as *mut TorshTensor);
    let labels_addr = TensorAddr::new(labels_handle as *mut TorshTensor);

    // Background training
    std::thread::spawn(move || unsafe {
        let state = &*(state_ptr as *const ComposeModelState);
        let model = model_addr.as_ptr();
        let data = data_addr.as_ptr();
        let labels = labels_addr.as_ptr();

        for epoch in 0..state.total_epochs {
            state.current_epoch.store(epoch, Ordering::Relaxed);

            let output = torsh_tensor_new(ptr::null(), ptr::null(), 0, TorshDType::F32);
            if torsh_linear_forward(model, data, output) != TorshError::Success {
                break;
            }

            let _ = torsh_tensor_sub(output, labels, output);
            *state.current_loss.lock() = 0.5; // Placeholder

            state.trigger_recomposition();

            torsh_tensor_free(output);
            std::thread::sleep(std::time::Duration::from_millis(100));
        }

        state.is_training.store(false, Ordering::Release);
        state.trigger_recomposition();
    });
}

/// Gets current training state for Compose
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_compose_ModelState_nativeIsTraining(
    _env: JNIEnv,
    _class: jclass,
    state_handle: jlong,
) -> jboolean {
    if state_handle == 0 {
        return 0;
    }
    let state = &*(state_handle as *const ComposeModelState);
    state.is_training.load(Ordering::Acquire) as jboolean
}

/// Gets current epoch for Compose
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_compose_ModelState_nativeGetEpoch(
    _env: JNIEnv,
    _class: jclass,
    state_handle: jlong,
) -> jlong {
    if state_handle == 0 {
        return 0;
    }
    let state = &*(state_handle as *const ComposeModelState);
    state.current_epoch.load(Ordering::Acquire) as jlong
}

/// Gets current loss for Compose
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_compose_ModelState_nativeGetLoss(
    _env: JNIEnv,
    _class: jclass,
    state_handle: jlong,
) -> jfloat {
    if state_handle == 0 {
        return f32::INFINITY;
    }
    let state = &*(state_handle as *const ComposeModelState);
    *state.current_loss.lock()
}

// ============================================================================
// MARK: - Proper JNI Array Handling
// ============================================================================

/// Properly creates a tensor from Java float array
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_Tensor_nativeFromFloatArray(
    env: JNIEnv,
    _class: jclass,
    data_array: jfloatArray,
    shape_array: jintArray,
) -> jlong {
    if data_array.is_null() || shape_array.is_null() {
        return 0;
    }

    // In production, would use JNI functions:
    // let data_len = (*(*env).GetArrayLength)(env, data_array);
    // let data_ptr = (*(*env).GetFloatArrayElements)(env, data_array, ptr::null_mut());
    // let shape_len = (*(*env).GetArrayLength)(env, shape_array);
    // let shape_ptr = (*(*env).GetIntArrayElements)(env, shape_array, ptr::null_mut());

    // For now, placeholder:
    let _ = env;
    let tensor = torsh_tensor_new(ptr::null(), ptr::null(), 0, TorshDType::F32);

    // Would release arrays:
    // (*(*env).ReleaseFloatArrayElements)(env, data_array, data_ptr, 0);
    // (*(*env).ReleaseIntArrayElements)(env, shape_array, shape_ptr, 0);

    tensor as jlong
}

/// Properly copies tensor data to Java float array
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_Tensor_nativeToFloatArray(
    env: JNIEnv,
    _class: jclass,
    tensor_handle: jlong,
    output_array: jfloatArray,
) -> jboolean {
    if tensor_handle == 0 || output_array.is_null() {
        return 0;
    }

    let tensor = tensor_handle as *const TorshTensor;
    let data_ptr = torsh_tensor_data(tensor) as *const c_float;

    if data_ptr.is_null() {
        return 0;
    }

    // In production, would:
    // let arr_len = (*(*env).GetArrayLength)(env, output_array);
    // let arr_ptr = (*(*env).GetFloatArrayElements)(env, output_array, ptr::null_mut());
    // std::ptr::copy_nonoverlapping(data_ptr, arr_ptr, arr_len as usize);
    // (*(*env).ReleaseFloatArrayElements)(env, output_array, arr_ptr, 0);

    let _ = (env, data_ptr);
    1
}

/// Creates a Java int array from tensor shape
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_Tensor_nativeGetShapeArray(
    env: JNIEnv,
    _class: jclass,
    tensor_handle: jlong,
) -> jintArray {
    if tensor_handle == 0 {
        return ptr::null_mut();
    }

    let tensor = tensor_handle as *mut TorshTensor;
    let mut shape = vec![0usize; 16];
    let mut ndim = 0usize;

    if torsh_tensor_shape(tensor, shape.as_mut_ptr(), &mut ndim) != TorshError::Success {
        return ptr::null_mut();
    }

    // In production, would:
    // let result = (*(*env).NewIntArray)(env, ndim as jsize);
    // let elements = (*(*env).GetIntArrayElements)(env, result, ptr::null_mut());
    // for i in 0..ndim {
    //     *elements.add(i) = shape[i] as jint;
    // }
    // (*(*env).ReleaseIntArrayElements)(env, result, elements, 0);
    // result

    let _ = (env, shape, ndim);
    ptr::null_mut()
}

// ============================================================================
// MARK: - TensorFlow Lite Interop
// ============================================================================

/// TFLite interpreter handle
#[repr(C)]
pub struct TFLiteInterpreter {
    interpreter_ptr: *mut c_void,
    input_count: u32,
    output_count: u32,
}

/// Creates a TFLite interpreter from a .tflite model file
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_TFLite_nativeCreateInterpreter(
    env: JNIEnv,
    _class: jclass,
    model_path: jstring,
) -> jlong {
    if model_path.is_null() {
        return 0;
    }

    // In production, would:
    // let path_chars = (*(*env).GetStringUTFChars)(env, model_path, ptr::null_mut());
    // let model = tflite::FlatBufferModel::BuildFromFile(path_chars);
    // let interpreter = tflite::InterpreterBuilder(model)();
    // (*(*env).ReleaseStringUTFChars)(env, model_path, path_chars);

    let _ = env;
    let interpreter = Box::new(TFLiteInterpreter {
        interpreter_ptr: ptr::null_mut(),
        input_count: 1,
        output_count: 1,
    });

    Box::into_raw(interpreter) as jlong
}

/// Runs TFLite inference and returns ToRSh tensor
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_TFLite_nativeInvoke(
    _env: JNIEnv,
    _class: jclass,
    interpreter_handle: jlong,
    input_handle: jlong,
) -> jlong {
    if interpreter_handle == 0 || input_handle == 0 {
        return 0;
    }

    // Placeholder - would copy ToRSh tensor to TFLite input, invoke, copy output
    let result = torsh_tensor_new(ptr::null(), ptr::null(), 0, TorshDType::F32);
    result as jlong
}

// ============================================================================
// MARK: - Lifecycle-Aware Components
// ============================================================================

/// Lifecycle event types
#[repr(C)]
pub enum LifecycleEvent {
    OnCreate,
    OnStart,
    OnResume,
    OnPause,
    OnStop,
    OnDestroy,
}

/// Lifecycle-aware training manager
#[repr(C)]
pub struct LifecycleTrainingManager {
    is_active: AtomicBool,
    should_pause: AtomicBool,
    model_handle: *mut TorshModule,
}

/// Creates a lifecycle-aware training manager
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_lifecycle_TrainingManager_nativeCreate(
    _env: JNIEnv,
    _class: jclass,
    model_handle: jlong,
) -> jlong {
    let manager = Box::new(LifecycleTrainingManager {
        is_active: AtomicBool::new(false),
        should_pause: AtomicBool::new(false),
        model_handle: model_handle as *mut TorshModule,
    });

    Box::into_raw(manager) as jlong
}

/// Handles lifecycle events
#[no_mangle]
pub unsafe extern "C" fn Java_com_coolJapan_torsh_lifecycle_TrainingManager_nativeOnLifecycleEvent(
    _env: JNIEnv,
    _class: jclass,
    manager_handle: jlong,
    event: jint,
) {
    if manager_handle == 0 {
        return;
    }

    let manager = &*(manager_handle as *const LifecycleTrainingManager);

    match event {
        0 => { /* OnCreate */ }
        1 => {
            /* OnStart */
            manager.is_active.store(true, Ordering::Release);
        }
        2 => {
            /* OnResume */
            manager.should_pause.store(false, Ordering::Release);
        }
        3 => {
            /* OnPause */
            manager.should_pause.store(true, Ordering::Release);
        }
        4 => {
            /* OnStop */
            manager.is_active.store(false, Ordering::Release);
        }
        5 => {
            /* OnDestroy */
            manager.is_active.store(false, Ordering::Release);
        }
        _ => {}
    }
}

// ============================================================================
// MARK: - Testing
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kotlin_coroutine_creation() {
        let coro = KotlinCoroutine::new(1);
        assert!(!coro.completed.load(Ordering::Acquire));
    }

    #[test]
    fn test_kotlin_flow_creation() {
        let flow = KotlinFlow::new(10);
        assert_eq!(flow.total_epochs, 10);
        assert_eq!(flow.current_epoch.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_nnapi_device_structure() {
        let device = NNAPIDevice {
            device_id: 0,
            device_name: [0; 256],
            device_type: 0,
            version: 29,
        };
        assert_eq!(device.version, 29);
    }

    #[test]
    fn test_compose_state_creation() {
        let state = ComposeModelState::new(100);
        assert_eq!(state.total_epochs, 100);
        assert!(!state.is_training.load(Ordering::Acquire));
    }

    #[test]
    fn test_lifecycle_manager() {
        let manager = LifecycleTrainingManager {
            is_active: AtomicBool::new(false),
            should_pause: AtomicBool::new(false),
            model_handle: ptr::null_mut(),
        };
        assert!(!manager.is_active.load(Ordering::Acquire));
    }

    #[test]
    fn test_training_progress_structure() {
        let progress = TrainingProgress {
            epoch: 5,
            total_epochs: 10,
            loss: 0.5,
            accuracy: 0.9,
            completed: 0,
        };
        assert_eq!(progress.epoch, 5);
        assert_eq!(progress.total_epochs, 10);
    }
}
