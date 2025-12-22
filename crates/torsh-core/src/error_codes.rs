//! Standard Error Codes for Interoperability
//!
//! Provides standardized error codes that can be used across different programming
//! languages and systems. This module maps ToRSh errors to POSIX-like error codes
//! for better C/C++ interoperability and consistent error handling in FFI contexts.
//!
//! # Features
//!
//! - **POSIX-compatible Codes**: Standard error codes similar to errno
//! - **Language Interoperability**: Easy mapping for FFI and C bindings
//! - **Error Categories**: Organized by error type for better handling
//! - **Bidirectional Mapping**: Convert between ToRSh errors and standard codes
//! - **Human-Readable**: Error code descriptions and recovery suggestions
//!
//! # Examples
//!
//! ```rust
//! use torsh_core::error_codes::{StandardErrorCode, ErrorCodeMapper};
//! use torsh_core::error::TorshError;
//!
//! let error = TorshError::InvalidArgument("test error".to_string());
//! let code = ErrorCodeMapper::from_torsh_error(&error);
//!
//! assert_eq!(code, StandardErrorCode::InvalidArgument);
//! assert_eq!(code.to_errno(), 22); // EINVAL
//! ```

use crate::error::TorshError;
use std::fmt;

/// Standard error codes for interoperability
///
/// These codes are designed to be compatible with POSIX errno values
/// and provide a standardized interface for error handling across
/// different programming languages and systems.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum StandardErrorCode {
    /// Success (no error) - errno 0
    Success = 0,

    /// Operation not permitted - errno 1 (EPERM)
    OperationNotPermitted = 1,

    /// No such file or directory - errno 2 (ENOENT)
    NoSuchEntity = 2,

    /// I/O error - errno 5 (EIO)
    IOError = 5,

    /// Bad file descriptor - errno 9 (EBADF)
    BadDescriptor = 9,

    /// Out of memory - errno 12 (ENOMEM)
    OutOfMemory = 12,

    /// Permission denied - errno 13 (EACCES)
    PermissionDenied = 13,

    /// Bad address - errno 14 (EFAULT)
    BadAddress = 14,

    /// Device or resource busy - errno 16 (EBUSY)
    DeviceBusy = 16,

    /// File exists - errno 17 (EEXIST)
    AlreadyExists = 17,

    /// Invalid argument - errno 22 (EINVAL)
    InvalidArgument = 22,

    /// Too many open files - errno 24 (EMFILE)
    TooManyFiles = 24,

    /// Function not implemented - errno 38 (ENOSYS)
    NotImplemented = 38,

    /// Directory not empty - errno 39 (ENOTEMPTY)
    NotEmpty = 39,

    /// Operation not supported - errno 95 (EOPNOTSUPP)
    NotSupported = 95,

    /// Operation timed out - errno 110 (ETIMEDOUT)
    TimedOut = 110,

    /// Operation canceled - errno 125 (ECANCELED)
    Canceled = 125,

    /// Custom error codes (1000+)
    ///
    /// Shape-related errors
    ShapeMismatch = 1001,
    InvalidShape = 1002,
    BroadcastError = 1003,
    DimensionError = 1004,

    /// DType-related errors
    DTypeMismatch = 1011,
    InvalidDType = 1012,
    ConversionError = 1013,

    /// Device-related errors
    DeviceNotAvailable = 1021,
    DeviceMismatch = 1022,
    DeviceError = 1023,
    CudaError = 1024,

    /// Memory-related errors
    AllocationFailed = 1031,
    AlignmentError = 1032,
    MemoryError = 1033,

    /// Backend-related errors
    BackendNotAvailable = 1041,
    BackendError = 1042,
    UnsupportedOperation = 1043,

    /// Runtime errors
    RuntimeError = 1051,
    InvalidState = 1052,
    NotInitialized = 1053,

    /// Unknown or unclassified error
    Unknown = 9999,
}

impl StandardErrorCode {
    /// Convert to POSIX errno value
    ///
    /// Returns the integer errno value that corresponds to this error code.
    /// Custom error codes (>= 1000) return their own value.
    #[inline]
    pub const fn to_errno(self) -> i32 {
        self as i32
    }

    /// Get error code category
    pub const fn category(&self) -> ErrorCategory {
        match self {
            Self::Success => ErrorCategory::Success,
            Self::OperationNotPermitted
            | Self::PermissionDenied
            | Self::DeviceBusy
            | Self::NotSupported => ErrorCategory::Permission,

            Self::NoSuchEntity | Self::NotEmpty => ErrorCategory::NotFound,

            Self::IOError | Self::BackendError | Self::DeviceError | Self::CudaError => {
                ErrorCategory::IO
            }

            Self::BadDescriptor | Self::BadAddress => ErrorCategory::InvalidReference,

            Self::OutOfMemory
            | Self::AllocationFailed
            | Self::AlignmentError
            | Self::MemoryError => ErrorCategory::Memory,

            Self::InvalidArgument
            | Self::InvalidShape
            | Self::InvalidDType
            | Self::DimensionError => ErrorCategory::InvalidInput,

            Self::ShapeMismatch | Self::DTypeMismatch | Self::DeviceMismatch => {
                ErrorCategory::Mismatch
            }

            Self::TooManyFiles => ErrorCategory::ResourceExhausted,

            Self::NotImplemented | Self::BackendNotAvailable | Self::UnsupportedOperation => {
                ErrorCategory::NotImplemented
            }

            Self::AlreadyExists => ErrorCategory::AlreadyExists,

            Self::BroadcastError | Self::ConversionError => ErrorCategory::Conversion,

            Self::TimedOut => ErrorCategory::Timeout,

            Self::Canceled => ErrorCategory::Canceled,

            Self::RuntimeError | Self::InvalidState | Self::NotInitialized => {
                ErrorCategory::Runtime
            }

            Self::DeviceNotAvailable => ErrorCategory::Unavailable,

            Self::Unknown => ErrorCategory::Unknown,
        }
    }

    /// Get human-readable error description
    pub const fn description(&self) -> &'static str {
        match self {
            Self::Success => "Success",
            Self::OperationNotPermitted => "Operation not permitted",
            Self::NoSuchEntity => "No such entity",
            Self::IOError => "Input/output error",
            Self::BadDescriptor => "Bad file descriptor",
            Self::OutOfMemory => "Out of memory",
            Self::PermissionDenied => "Permission denied",
            Self::BadAddress => "Bad address",
            Self::DeviceBusy => "Device or resource busy",
            Self::AlreadyExists => "Already exists",
            Self::InvalidArgument => "Invalid argument",
            Self::TooManyFiles => "Too many open files",
            Self::NotImplemented => "Function not implemented",
            Self::NotEmpty => "Directory not empty",
            Self::NotSupported => "Operation not supported",
            Self::TimedOut => "Operation timed out",
            Self::Canceled => "Operation canceled",
            Self::ShapeMismatch => "Shape mismatch",
            Self::InvalidShape => "Invalid shape",
            Self::BroadcastError => "Broadcasting error",
            Self::DimensionError => "Dimension error",
            Self::DTypeMismatch => "Data type mismatch",
            Self::InvalidDType => "Invalid data type",
            Self::ConversionError => "Type conversion error",
            Self::DeviceNotAvailable => "Device not available",
            Self::DeviceMismatch => "Device mismatch",
            Self::DeviceError => "Device error",
            Self::CudaError => "CUDA error",
            Self::AllocationFailed => "Memory allocation failed",
            Self::AlignmentError => "Memory alignment error",
            Self::MemoryError => "Memory error",
            Self::BackendNotAvailable => "Backend not available",
            Self::BackendError => "Backend error",
            Self::UnsupportedOperation => "Unsupported operation",
            Self::RuntimeError => "Runtime error",
            Self::InvalidState => "Invalid state",
            Self::NotInitialized => "Not initialized",
            Self::Unknown => "Unknown error",
        }
    }

    /// Get recovery suggestion for this error
    pub const fn recovery_suggestion(&self) -> &'static str {
        match self {
            Self::Success => "No action needed",
            Self::OutOfMemory | Self::AllocationFailed => "Free memory or reduce allocation size",
            Self::InvalidArgument | Self::InvalidShape | Self::InvalidDType => {
                "Check input parameters and types"
            }
            Self::ShapeMismatch | Self::DTypeMismatch | Self::DeviceMismatch => {
                "Ensure operand shapes/types/devices match"
            }
            Self::DeviceNotAvailable | Self::BackendNotAvailable => {
                "Check device availability and capabilities"
            }
            Self::NotImplemented | Self::UnsupportedOperation => {
                "Use an alternative operation or backend"
            }
            Self::TimedOut => "Increase timeout or check system performance",
            Self::Canceled => "Retry the operation if appropriate",
            Self::PermissionDenied | Self::OperationNotPermitted => "Check permissions",
            Self::DeviceBusy => "Wait for device to become available",
            Self::IOError | Self::DeviceError => "Check hardware and drivers",
            Self::CudaError => "Check CUDA installation and GPU status",
            Self::AlignmentError => "Ensure proper memory alignment",
            Self::NotInitialized => "Initialize the component before use",
            Self::InvalidState => "Check operation preconditions",
            _ => "Consult documentation for details",
        }
    }

    /// Check if this is a recoverable error
    pub const fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::TimedOut
                | Self::DeviceBusy
                | Self::Canceled
                | Self::DeviceNotAvailable
                | Self::BackendNotAvailable
                | Self::TooManyFiles
        )
    }

    /// Check if this error indicates a programming bug
    pub const fn is_bug(&self) -> bool {
        matches!(
            self,
            Self::InvalidArgument
                | Self::InvalidShape
                | Self::InvalidDType
                | Self::DimensionError
                | Self::BadDescriptor
                | Self::BadAddress
                | Self::NotInitialized
                | Self::InvalidState
        )
    }

    /// Get severity level (0-5, higher is more severe)
    pub const fn severity(&self) -> u8 {
        match self.category() {
            ErrorCategory::Success => 0,
            ErrorCategory::Canceled => 1,
            ErrorCategory::NotFound | ErrorCategory::AlreadyExists => 2,
            ErrorCategory::InvalidInput | ErrorCategory::Mismatch | ErrorCategory::Conversion => 3,
            ErrorCategory::Permission
            | ErrorCategory::NotImplemented
            | ErrorCategory::Timeout
            | ErrorCategory::Runtime
            | ErrorCategory::Unavailable => 4,
            ErrorCategory::Memory
            | ErrorCategory::IO
            | ErrorCategory::InvalidReference
            | ErrorCategory::ResourceExhausted => 5,
            ErrorCategory::Unknown => 3,
        }
    }
}

impl fmt::Display for StandardErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (code {}): {}",
            self.description(),
            self.to_errno(),
            self.recovery_suggestion()
        )
    }
}

/// Error category for grouping related errors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// No error
    Success,
    /// Permission or access denied
    Permission,
    /// Entity not found
    NotFound,
    /// Input/output error
    IO,
    /// Invalid reference or pointer
    InvalidReference,
    /// Memory allocation failure
    Memory,
    /// Invalid input or argument
    InvalidInput,
    /// Type or shape mismatch
    Mismatch,
    /// Resource exhausted
    ResourceExhausted,
    /// Not implemented
    NotImplemented,
    /// Already exists
    AlreadyExists,
    /// Conversion error
    Conversion,
    /// Operation timeout
    Timeout,
    /// Operation canceled
    Canceled,
    /// Runtime error
    Runtime,
    /// Resource unavailable
    Unavailable,
    /// Unknown error
    Unknown,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => write!(f, "Success"),
            Self::Permission => write!(f, "Permission"),
            Self::NotFound => write!(f, "Not Found"),
            Self::IO => write!(f, "I/O Error"),
            Self::InvalidReference => write!(f, "Invalid Reference"),
            Self::Memory => write!(f, "Memory Error"),
            Self::InvalidInput => write!(f, "Invalid Input"),
            Self::Mismatch => write!(f, "Mismatch"),
            Self::ResourceExhausted => write!(f, "Resource Exhausted"),
            Self::NotImplemented => write!(f, "Not Implemented"),
            Self::AlreadyExists => write!(f, "Already Exists"),
            Self::Conversion => write!(f, "Conversion Error"),
            Self::Timeout => write!(f, "Timeout"),
            Self::Canceled => write!(f, "Canceled"),
            Self::Runtime => write!(f, "Runtime Error"),
            Self::Unavailable => write!(f, "Unavailable"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Mapper between ToRSh errors and standard error codes
pub struct ErrorCodeMapper;

impl ErrorCodeMapper {
    /// Convert a TorshError to a standard error code
    pub fn from_torsh_error(error: &TorshError) -> StandardErrorCode {
        match error {
            TorshError::InvalidDimension { .. } => StandardErrorCode::DimensionError,

            TorshError::ShapeMismatch { .. } => StandardErrorCode::ShapeMismatch,

            TorshError::InvalidShape(_) => StandardErrorCode::InvalidShape,

            TorshError::BroadcastError { .. } => StandardErrorCode::BroadcastError,

            TorshError::Shape(_) => {
                // All shape errors map to shape-related codes
                StandardErrorCode::InvalidShape
            }

            TorshError::DeviceMismatch => StandardErrorCode::DeviceMismatch,

            TorshError::AllocationError(_) => StandardErrorCode::AllocationFailed,

            TorshError::BackendError(_) => StandardErrorCode::BackendError,

            TorshError::UnsupportedOperation { .. } => StandardErrorCode::UnsupportedOperation,

            TorshError::InvalidArgument(_) => StandardErrorCode::InvalidArgument,

            TorshError::NotImplemented(_) => StandardErrorCode::NotImplemented,

            TorshError::IoError(_) => StandardErrorCode::IOError,

            TorshError::IndexOutOfBounds { .. }
            | TorshError::IndexError { .. }
            | TorshError::Index(_) => StandardErrorCode::InvalidArgument,

            TorshError::ConversionError(_) => StandardErrorCode::ConversionError,

            TorshError::DeviceError(_) => StandardErrorCode::DeviceError,

            TorshError::RuntimeError(_) => StandardErrorCode::RuntimeError,

            TorshError::InvalidState(_) => StandardErrorCode::InvalidState,

            TorshError::InvalidOperation(_) => StandardErrorCode::InvalidArgument,

            TorshError::ConfigError(_) => StandardErrorCode::InvalidArgument,

            TorshError::SynchronizationError(_) => StandardErrorCode::RuntimeError,

            TorshError::General(_) | TorshError::WithContext { .. } => {
                // For general/context errors, try to infer from the error category
                StandardErrorCode::RuntimeError
            }

            _ => StandardErrorCode::Unknown,
        }
    }

    /// Convert a standard error code to an errno value
    #[inline]
    pub fn to_errno(code: StandardErrorCode) -> i32 {
        code.to_errno()
    }

    /// Get error details from a TorshError
    pub fn get_error_details(error: &TorshError) -> ErrorDetails {
        let code = Self::from_torsh_error(error);
        ErrorDetails {
            code,
            category: code.category(),
            description: code.description(),
            recovery_suggestion: code.recovery_suggestion(),
            is_recoverable: code.is_recoverable(),
            is_bug: code.is_bug(),
            severity: code.severity(),
            original_error: format!("{:?}", error),
        }
    }
}

/// Detailed error information
#[derive(Debug, Clone)]
pub struct ErrorDetails {
    /// Standard error code
    pub code: StandardErrorCode,
    /// Error category
    pub category: ErrorCategory,
    /// Human-readable description
    pub description: &'static str,
    /// Recovery suggestion
    pub recovery_suggestion: &'static str,
    /// Whether the error is recoverable
    pub is_recoverable: bool,
    /// Whether this indicates a bug
    pub is_bug: bool,
    /// Severity level (0-5)
    pub severity: u8,
    /// Original error details
    pub original_error: String,
}

impl fmt::Display for ErrorDetails {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Error: {} (code {})",
            self.description,
            self.code.to_errno()
        )?;
        writeln!(f, "Category: {}", self.category)?;
        writeln!(f, "Severity: {}/5", self.severity)?;
        writeln!(f, "Recoverable: {}", self.is_recoverable)?;
        writeln!(f, "Likely Bug: {}", self.is_bug)?;
        writeln!(f, "Suggestion: {}", self.recovery_suggestion)?;
        writeln!(f, "Details: {}", self.original_error)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_to_errno() {
        assert_eq!(StandardErrorCode::Success.to_errno(), 0);
        assert_eq!(StandardErrorCode::InvalidArgument.to_errno(), 22);
        assert_eq!(StandardErrorCode::OutOfMemory.to_errno(), 12);
        assert_eq!(StandardErrorCode::NotImplemented.to_errno(), 38);
    }

    #[test]
    fn test_error_categories() {
        assert_eq!(
            StandardErrorCode::OutOfMemory.category(),
            ErrorCategory::Memory
        );
        assert_eq!(
            StandardErrorCode::InvalidShape.category(),
            ErrorCategory::InvalidInput
        );
        assert_eq!(
            StandardErrorCode::ShapeMismatch.category(),
            ErrorCategory::Mismatch
        );
    }

    #[test]
    fn test_recoverability() {
        assert!(StandardErrorCode::TimedOut.is_recoverable());
        assert!(StandardErrorCode::DeviceBusy.is_recoverable());
        assert!(!StandardErrorCode::OutOfMemory.is_recoverable());
        assert!(!StandardErrorCode::InvalidArgument.is_recoverable());
    }

    #[test]
    fn test_bug_detection() {
        assert!(StandardErrorCode::InvalidArgument.is_bug());
        assert!(StandardErrorCode::InvalidShape.is_bug());
        assert!(!StandardErrorCode::OutOfMemory.is_bug());
        assert!(!StandardErrorCode::TimedOut.is_bug());
    }

    #[test]
    fn test_severity() {
        assert_eq!(StandardErrorCode::Success.severity(), 0);
        assert_eq!(StandardErrorCode::Canceled.severity(), 1);
        assert!(StandardErrorCode::OutOfMemory.severity() == 5);
        assert!(StandardErrorCode::InvalidArgument.severity() == 3);
    }

    #[test]
    fn test_torsh_error_mapping() {
        let error = TorshError::InvalidDimension { dim: 5, ndim: 3 };
        let code = ErrorCodeMapper::from_torsh_error(&error);
        assert_eq!(code, StandardErrorCode::DimensionError);

        let error = TorshError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![3, 2],
        };
        let code = ErrorCodeMapper::from_torsh_error(&error);
        assert_eq!(code, StandardErrorCode::ShapeMismatch);

        let error = TorshError::AllocationError("failed to allocate memory".to_string());
        let code = ErrorCodeMapper::from_torsh_error(&error);
        assert_eq!(code, StandardErrorCode::AllocationFailed);
    }

    #[test]
    fn test_error_details() {
        let error = TorshError::InvalidDimension { dim: 5, ndim: 3 };
        let details = ErrorCodeMapper::get_error_details(&error);

        assert_eq!(details.code, StandardErrorCode::DimensionError);
        assert_eq!(details.category, ErrorCategory::InvalidInput);
        // DimensionError is marked as a bug since it indicates invalid input
        assert!(details.code.is_bug());
        assert!(!details.is_recoverable);
        assert_eq!(details.severity, 3);
    }

    #[test]
    fn test_display_formatting() {
        let code = StandardErrorCode::InvalidArgument;
        let formatted = format!("{}", code);
        assert!(formatted.contains("Invalid argument"));
        assert!(formatted.contains("22"));
    }

    #[test]
    fn test_all_error_codes_have_descriptions() {
        let codes = [
            StandardErrorCode::Success,
            StandardErrorCode::OutOfMemory,
            StandardErrorCode::InvalidArgument,
            StandardErrorCode::ShapeMismatch,
            StandardErrorCode::DTypeMismatch,
            StandardErrorCode::DeviceNotAvailable,
        ];

        for code in &codes {
            assert!(!code.description().is_empty());
            assert!(!code.recovery_suggestion().is_empty());
        }
    }

    #[test]
    fn test_custom_error_codes() {
        assert_eq!(StandardErrorCode::ShapeMismatch.to_errno(), 1001);
        assert_eq!(StandardErrorCode::DTypeMismatch.to_errno(), 1011);
        assert_eq!(StandardErrorCode::DeviceNotAvailable.to_errno(), 1021);
    }

    #[test]
    fn test_error_category_display() {
        assert_eq!(format!("{}", ErrorCategory::Memory), "Memory Error");
        assert_eq!(format!("{}", ErrorCategory::InvalidInput), "Invalid Input");
        assert_eq!(format!("{}", ErrorCategory::Mismatch), "Mismatch");
    }
}
