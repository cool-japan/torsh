//! cuDNN handle management
//!
//! This module provides the cuDNN handle wrapper for managing cuDNN contexts
//! and stream operations.

use crate::error::{CudaError, CudaResult};
use crate::stream::CudaStream;

#[cfg(feature = "cudnn")]
use cudnn_sys::*;

/// cuDNN handle wrapper
///
/// Provides a safe wrapper around the cuDNN handle with automatic resource management.
/// The handle is automatically destroyed when the wrapper is dropped.
///
/// # Examples
///
/// ```rust,ignore
/// use torsh_backend::cuda::cudnn::CudnnHandle;
///
/// let mut handle = CudnnHandle::new()?;
/// // Use handle for cuDNN operations
/// ```
pub struct CudnnHandle {
    #[cfg(feature = "cudnn")]
    handle: cudnnHandle_t,
    #[cfg(not(feature = "cudnn"))]
    _phantom: std::marker::PhantomData<()>,
}

unsafe impl Send for CudnnHandle {}
unsafe impl Sync for CudnnHandle {}

impl CudnnHandle {
    /// Create new cuDNN handle
    ///
    /// Initializes a new cuDNN context handle. This must be called before
    /// performing any cuDNN operations.
    ///
    /// # Returns
    ///
    /// A new `CudnnHandle` instance on success, or an error if handle creation fails.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::CudnnError` if:
    /// - cuDNN is not available (feature not enabled)
    /// - cuDNN handle creation fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_backend::cuda::cudnn::CudnnHandle;
    ///
    /// match CudnnHandle::new() {
    ///     Ok(handle) => {
    ///         // Use the handle for cuDNN operations
    ///         println!("cuDNN handle created successfully");
    ///     }
    ///     Err(e) => {
    ///         eprintln!("Failed to create cuDNN handle: {}", e);
    ///     }
    /// }
    /// ```
    pub fn new() -> CudaResult<Self> {
        #[cfg(feature = "cudnn")]
        {
            let mut handle: cudnnHandle_t = std::ptr::null_mut();
            let status = unsafe { cudnnCreate(&mut handle) };
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to create cuDNN handle: {:?}",
                    status
                )));
            }
            Ok(Self { handle })
        }
        #[cfg(not(feature = "cudnn"))]
        {
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Set stream for cuDNN operations
    ///
    /// Associates a CUDA stream with this cuDNN handle. All subsequent cuDNN
    /// operations using this handle will execute on the specified stream.
    ///
    /// # Arguments
    ///
    /// * `stream` - The CUDA stream to associate with this handle
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if stream setting fails.
    ///
    /// # Errors
    ///
    /// Returns `CudaError::CudnnError` if:
    /// - cuDNN is not available (feature not enabled)
    /// - Stream setting fails
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_backend::cuda::{CudaStream, cudnn::CudnnHandle};
    ///
    /// let mut handle = CudnnHandle::new()?;
    /// let stream = CudaStream::new()?;
    /// handle.set_stream(&stream)?;
    /// ```
    pub fn set_stream(&mut self, stream: &CudaStream) -> CudaResult<()> {
        #[cfg(feature = "cudnn")]
        {
            let status = unsafe { cudnnSetStream(self.handle, stream.raw() as cudaStream_t) };
            if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
                return Err(CudaError::CudnnError(format!(
                    "Failed to set cuDNN stream: {:?}",
                    status
                )));
            }
            Ok(())
        }
        #[cfg(not(feature = "cudnn"))]
        {
            let _ = stream;
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Get the raw cuDNN handle
    ///
    /// Returns the underlying cuDNN handle for use with low-level cuDNN functions.
    /// This is only available when the "cudnn" feature is enabled.
    ///
    /// # Returns
    ///
    /// The raw `cudnnHandle_t` pointer.
    ///
    /// # Safety
    ///
    /// The returned handle should not be destroyed manually as it is managed
    /// by the `CudnnHandle` wrapper.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use torsh_backend::cuda::cudnn::CudnnHandle;
    ///
    /// let handle = CudnnHandle::new()?;
    /// let raw_handle = handle.raw();
    /// // Use raw_handle with cuDNN functions
    /// ```
    #[cfg(feature = "cudnn")]
    pub fn raw(&self) -> cudnnHandle_t {
        self.handle
    }

    /// Check if this handle is valid
    ///
    /// Returns whether the handle has been properly initialized and is ready for use.
    ///
    /// # Returns
    ///
    /// `true` if the handle is valid, `false` otherwise.
    pub fn is_valid(&self) -> bool {
        #[cfg(feature = "cudnn")]
        {
            !self.handle.is_null()
        }
        #[cfg(not(feature = "cudnn"))]
        {
            false
        }
    }

    /// Get cuDNN version information
    ///
    /// Returns the version of the cuDNN library being used.
    ///
    /// # Returns
    ///
    /// The cuDNN version as a tuple (major, minor, patch) on success.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDNN is not available.
    pub fn get_version() -> CudaResult<(i32, i32, i32)> {
        #[cfg(feature = "cudnn")]
        {
            let version = unsafe { cudnnGetVersion() };
            let major = (version / 1000) as i32;
            let minor = ((version % 1000) / 100) as i32;
            let patch = (version % 100) as i32;
            Ok((major, minor, patch))
        }
        #[cfg(not(feature = "cudnn"))]
        {
            Err(CudaError::CudnnError(
                "cuDNN not available - feature not enabled".to_string(),
            ))
        }
    }

    /// Get cuDNN error string
    ///
    /// Converts a cuDNN status code to a human-readable error string.
    ///
    /// # Arguments
    ///
    /// * `status` - The cuDNN status code to convert
    ///
    /// # Returns
    ///
    /// A string description of the error.
    #[cfg(feature = "cudnn")]
    pub fn get_error_string(status: cudnnStatus_t) -> String {
        unsafe {
            let ptr = cudnnGetErrorString(status);
            if ptr.is_null() {
                "Unknown cuDNN error".to_string()
            } else {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        }
    }

    /// Set math type for this handle
    ///
    /// Configures the math type (precision mode) for operations performed
    /// with this handle.
    ///
    /// # Arguments
    ///
    /// * `math_type` - The math type to use
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if setting fails.
    #[cfg(feature = "cudnn")]
    pub fn set_math_type(&mut self, math_type: cudnnMathType_t) -> CudaResult<()> {
        let status = unsafe { cudnnSetMathType(self.handle, math_type) };
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(CudaError::CudnnError(format!(
                "Failed to set math type: {}",
                Self::get_error_string(status)
            )));
        }
        Ok(())
    }

    /// Get the current math type for this handle
    ///
    /// Returns the math type currently configured for this handle.
    ///
    /// # Returns
    ///
    /// The current math type on success, or an error if retrieval fails.
    #[cfg(feature = "cudnn")]
    pub fn get_math_type(&self) -> CudaResult<cudnnMathType_t> {
        let mut math_type: cudnnMathType_t = cudnnMathType_t::CUDNN_DEFAULT_MATH;
        let status = unsafe { cudnnGetMathType(self.handle, &mut math_type) };
        if status != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
            return Err(CudaError::CudnnError(format!(
                "Failed to get math type: {}",
                Self::get_error_string(status)
            )));
        }
        Ok(math_type)
    }
}

impl Drop for CudnnHandle {
    /// Automatically destroy the cuDNN handle when dropped
    ///
    /// This ensures proper cleanup of cuDNN resources when the handle
    /// goes out of scope.
    fn drop(&mut self) {
        #[cfg(feature = "cudnn")]
        {
            if !self.handle.is_null() {
                unsafe {
                    let _status = cudnnDestroy(self.handle);
                    // Note: We ignore the status here as we can't return an error from drop
                    // In practice, cudnnDestroy rarely fails if the handle was valid
                }
            }
        }
    }
}

impl Default for CudnnHandle {
    /// Create a default cuDNN handle
    ///
    /// This will attempt to create a new cuDNN handle. If creation fails,
    /// this will panic. For error handling, use `CudnnHandle::new()` instead.
    fn default() -> Self {
        Self::new().expect("Failed to create default cuDNN handle")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_creation() {
        // Test basic handle creation
        // Note: This test will only pass if cuDNN is available
        #[cfg(feature = "cudnn")]
        {
            match CudnnHandle::new() {
                Ok(handle) => {
                    assert!(handle.is_valid());
                }
                Err(_) => {
                    // cuDNN might not be available in test environment
                    // This is acceptable
                }
            }
        }

        #[cfg(not(feature = "cudnn"))]
        {
            let result = CudnnHandle::new();
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_handle_validity() {
        // Test handle validity checking
        #[cfg(feature = "cudnn")]
        {
            if let Ok(handle) = CudnnHandle::new() {
                assert!(handle.is_valid());
            }
        }

        #[cfg(not(feature = "cudnn"))]
        {
            // When cuDNN is not available, handles are always invalid
            if let Ok(handle) = CudnnHandle::new() {
                assert!(!handle.is_valid());
            }
        }
    }

    #[test]
    fn test_version_info() {
        // Test version information retrieval
        #[cfg(feature = "cudnn")]
        {
            match CudnnHandle::get_version() {
                Ok((major, minor, patch)) => {
                    assert!(major >= 7); // cuDNN 7.x or higher
                    assert!(minor >= 0);
                    assert!(patch >= 0);
                }
                Err(_) => {
                    // cuDNN might not be available in test environment
                }
            }
        }

        #[cfg(not(feature = "cudnn"))]
        {
            let result = CudnnHandle::get_version();
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_error_string() {
        // Test error string conversion
        #[cfg(feature = "cudnn")]
        {
            let error_str = CudnnHandle::get_error_string(cudnnStatus_t::CUDNN_STATUS_SUCCESS);
            assert!(!error_str.is_empty());

            let error_str = CudnnHandle::get_error_string(cudnnStatus_t::CUDNN_STATUS_BAD_PARAM);
            assert!(!error_str.is_empty());
            assert!(
                error_str.to_lowercase().contains("param")
                    || error_str.to_lowercase().contains("parameter")
            );
        }
    }

    #[test]
    fn test_math_type_operations() {
        // Test math type setting and getting
        #[cfg(feature = "cudnn")]
        {
            if let Ok(mut handle) = CudnnHandle::new() {
                // Test setting and getting math type
                let math_type = cudnnMathType_t::CUDNN_DEFAULT_MATH;
                if handle.set_math_type(math_type).is_ok() {
                    if let Ok(retrieved_type) = handle.get_math_type() {
                        assert_eq!(retrieved_type, math_type);
                    }
                }
            }
        }
    }

    #[test]
    fn test_send_sync() {
        // Test that CudnnHandle implements Send and Sync
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<CudnnHandle>();
        assert_sync::<CudnnHandle>();
    }
}
