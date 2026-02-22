//! Utility functions for C API
//!
//! This module provides utility functionality for the ToRSh C API,
//! including initialization, cleanup, device management, error handling,
//! and version information.

use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::{Mutex, OnceLock};

use super::types::TorshError;

// Global state for error handling
static LAST_ERROR: OnceLock<Mutex<Option<String>>> = OnceLock::new();

// Global state for device management
static mut DEVICE_TYPE: c_int = 0; // 0 = CPU
static mut DEVICE_ID: c_int = 0;

// Version information
static VERSION_STRING: &str = "0.1.0-alpha.2\0";

/// Get the global error store
pub fn get_last_error() -> &'static Mutex<Option<String>> {
    LAST_ERROR.get_or_init(|| Mutex::new(None))
}

/// Set last error message
pub fn set_last_error(message: String) {
    if let Ok(mut last_error) = get_last_error().lock() {
        *last_error = Some(message);
    }
}

// =============================================================================
// Version and Library Information
// =============================================================================

/// Get version string
#[no_mangle]
pub unsafe extern "C" fn torsh_version() -> *const c_char {
    VERSION_STRING.as_ptr() as *const c_char
}

// =============================================================================
// Library Initialization and Cleanup
// =============================================================================

/// Initialize ToRSh library
#[no_mangle]
pub unsafe extern "C" fn torsh_init() -> TorshError {
    // Initialize global state
    if let Ok(mut tensor_store) = super::tensor::get_tensor_store().lock() {
        tensor_store.clear();
    }
    if let Ok(mut module_store) = super::neural::get_module_store().lock() {
        module_store.clear();
    }
    if let Ok(mut optimizer_store) = super::optimizer::get_optimizer_store().lock() {
        optimizer_store.clear();
    }
    if let Ok(mut last_error) = get_last_error().lock() {
        *last_error = None;
    }

    TorshError::Success
}

/// Cleanup ToRSh library
#[no_mangle]
pub unsafe extern "C" fn torsh_cleanup() -> TorshError {
    // Clear all stores
    if let Ok(mut tensor_store) = super::tensor::get_tensor_store().lock() {
        tensor_store.clear();
    }
    if let Ok(mut module_store) = super::neural::get_module_store().lock() {
        module_store.clear();
    }
    if let Ok(mut optimizer_store) = super::optimizer::get_optimizer_store().lock() {
        optimizer_store.clear();
    }
    if let Ok(mut last_error) = get_last_error().lock() {
        *last_error = None;
    }

    TorshError::Success
}

// =============================================================================
// Device Management
// =============================================================================

/// Set device
#[no_mangle]
pub unsafe extern "C" fn torsh_set_device(device_type: c_int, device_id: c_int) -> TorshError {
    // 0 = CPU, 1 = CUDA, 2 = Metal, etc.
    if device_type < 0 || device_id < 0 {
        set_last_error("Invalid device parameters".to_string());
        return TorshError::InvalidArgument;
    }

    DEVICE_TYPE = device_type;
    DEVICE_ID = device_id;

    TorshError::Success
}

/// Get current device type
#[no_mangle]
pub unsafe extern "C" fn torsh_get_device_type() -> c_int {
    DEVICE_TYPE
}

/// Get current device ID
#[no_mangle]
pub unsafe extern "C" fn torsh_get_device_id() -> c_int {
    DEVICE_ID
}

// =============================================================================
// CUDA Support Information
// =============================================================================

/// Check CUDA availability
#[no_mangle]
pub unsafe extern "C" fn torsh_cuda_is_available() -> c_int {
    // In a real implementation, this would check for CUDA runtime
    // For now, always return false (0) as we're using CPU backend
    0
}

/// Get CUDA device count
#[no_mangle]
pub unsafe extern "C" fn torsh_cuda_device_count() -> c_int {
    // In a real implementation, this would query CUDA device count
    // For now, always return 0 as we're using CPU backend
    0
}

/// Get CUDA device name
#[no_mangle]
pub unsafe extern "C" fn torsh_cuda_device_name(_device_id: c_int) -> *const c_char {
    // In a real implementation, this would return the actual device name
    // For now, return null as we don't have CUDA support
    ptr::null()
}

/// Get CUDA device memory
#[no_mangle]
pub unsafe extern "C" fn torsh_cuda_device_memory(_device_id: c_int) -> usize {
    // In a real implementation, this would return available memory in bytes
    // For now, return 0 as we don't have CUDA support
    0
}

// =============================================================================
// Error Handling
// =============================================================================

/// Get last error message
#[no_mangle]
pub unsafe extern "C" fn torsh_get_last_error() -> *const c_char {
    if let Ok(last_error) = get_last_error().lock() {
        if let Some(ref error_msg) = *last_error {
            return error_msg.as_bytes().as_ptr() as *const c_char;
        }
    }

    ptr::null()
}

/// Clear last error
#[no_mangle]
pub unsafe extern "C" fn torsh_clear_last_error() {
    if let Ok(mut last_error) = get_last_error().lock() {
        *last_error = None;
    }
}

/// Check if there is a pending error
#[no_mangle]
pub unsafe extern "C" fn torsh_has_error() -> c_int {
    if let Ok(last_error) = get_last_error().lock() {
        if last_error.is_some() {
            return 1; // true
        }
    }
    0 // false
}

// =============================================================================
// Memory and Performance Information
// =============================================================================

/// Get memory usage statistics
#[no_mangle]
pub unsafe extern "C" fn torsh_get_memory_usage() -> usize {
    // In a real implementation, this would return actual memory usage
    // For now, return 0 as a placeholder
    0
}

/// Get peak memory usage
#[no_mangle]
pub unsafe extern "C" fn torsh_get_peak_memory_usage() -> usize {
    // In a real implementation, this would return peak memory usage
    // For now, return 0 as a placeholder
    0
}

/// Reset memory usage tracking
#[no_mangle]
pub unsafe extern "C" fn torsh_reset_memory_tracking() -> TorshError {
    // In a real implementation, this would reset memory tracking counters
    // For now, just return success
    TorshError::Success
}

/// Get number of threads used by ToRSh
#[no_mangle]
pub unsafe extern "C" fn torsh_get_num_threads() -> c_int {
    // In a real implementation, this would return the actual thread count
    // For now, return number of logical CPUs
    num_cpus::get() as c_int
}

/// Set number of threads for ToRSh operations
#[no_mangle]
pub unsafe extern "C" fn torsh_set_num_threads(num_threads: c_int) -> TorshError {
    if num_threads <= 0 {
        set_last_error("Number of threads must be positive".to_string());
        return TorshError::InvalidArgument;
    }

    // In a real implementation, this would configure the thread pool
    // For now, just return success
    TorshError::Success
}

// =============================================================================
// Configuration and Settings
// =============================================================================

/// Enable or disable automatic mixed precision
#[no_mangle]
pub unsafe extern "C" fn torsh_set_autocast(_enabled: c_int) -> TorshError {
    // In a real implementation, this would configure automatic mixed precision
    // For now, just return success
    TorshError::Success
}

/// Check if automatic mixed precision is enabled
#[no_mangle]
pub unsafe extern "C" fn torsh_is_autocast_enabled() -> c_int {
    // In a real implementation, this would return the actual autocast state
    // For now, return false (0)
    0
}

/// Set random seed for reproducible results
#[no_mangle]
pub unsafe extern "C" fn torsh_set_random_seed(seed: u64) -> TorshError {
    // In a real implementation, this would set the global random seed
    fastrand::seed(seed);
    TorshError::Success
}

/// Enable or disable gradient computation
#[no_mangle]
pub unsafe extern "C" fn torsh_set_grad_enabled(_enabled: c_int) -> TorshError {
    // In a real implementation, this would configure gradient computation
    // For now, just return success
    TorshError::Success
}

/// Check if gradient computation is enabled
#[no_mangle]
pub unsafe extern "C" fn torsh_is_grad_enabled() -> c_int {
    // In a real implementation, this would return the actual gradient state
    // For now, return true (1)
    1
}

// =============================================================================
// Debug and Profiling
// =============================================================================

/// Enable or disable debug mode
#[no_mangle]
pub unsafe extern "C" fn torsh_set_debug_mode(_enabled: c_int) -> TorshError {
    // In a real implementation, this would configure debug output
    // For now, just return success
    TorshError::Success
}

/// Check if debug mode is enabled
#[no_mangle]
pub unsafe extern "C" fn torsh_is_debug_mode_enabled() -> c_int {
    // In a real implementation, this would return the actual debug state
    // For now, return false (0)
    0
}

/// Start performance profiling
#[no_mangle]
pub unsafe extern "C" fn torsh_start_profiling() -> TorshError {
    // In a real implementation, this would start performance profiling
    // For now, just return success
    TorshError::Success
}

/// Stop performance profiling and return results
#[no_mangle]
pub unsafe extern "C" fn torsh_stop_profiling() -> *const c_char {
    // In a real implementation, this would return profiling results as JSON
    // For now, return empty string
    "\0".as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        unsafe {
            let version = torsh_version();
            assert!(!version.is_null());
        }
    }

    #[test]
    fn test_initialization() {
        unsafe {
            let result = torsh_init();
            assert_eq!(result, TorshError::Success);

            let cleanup_result = torsh_cleanup();
            assert_eq!(cleanup_result, TorshError::Success);
        }
    }

    #[test]
    fn test_device_management() {
        unsafe {
            let result = torsh_set_device(0, 0);
            assert_eq!(result, TorshError::Success);

            let device_type = torsh_get_device_type();
            assert_eq!(device_type, 0);

            let device_id = torsh_get_device_id();
            assert_eq!(device_id, 0);
        }
    }

    #[test]
    fn test_error_handling() {
        unsafe {
            torsh_clear_last_error();
            assert_eq!(torsh_has_error(), 0);

            set_last_error("Test error".to_string());
            assert_eq!(torsh_has_error(), 1);

            torsh_clear_last_error();
            assert_eq!(torsh_has_error(), 0);
        }
    }

    #[test]
    fn test_cuda_info() {
        unsafe {
            assert_eq!(torsh_cuda_is_available(), 0);
            assert_eq!(torsh_cuda_device_count(), 0);
        }
    }

    #[test]
    fn test_random_seed() {
        unsafe {
            let result = torsh_set_random_seed(12345);
            assert_eq!(result, TorshError::Success);
        }
    }
}
