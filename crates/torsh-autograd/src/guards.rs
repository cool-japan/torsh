//! RAII guard implementations for gradient mode management
//!
//! This module provides RAII (Resource Acquisition Is Initialization) guards for
//! managing gradient computation modes with automatic cleanup. Guards ensure that
//! gradient mode changes are properly restored when leaving scope.
//!
//! # Features
//!
//! - **NoGradGuard**: Disable gradients within a scope
//! - **EnableGradGuard**: Enable gradients within a scope
//! - **Automatic cleanup**: Mode restoration via Drop trait
//! - **Nested contexts**: Support for multiple guard levels

use crate::grad_mode::{pop_grad_enabled, push_grad_enabled};

/// Guard for disabling gradient computation
///
/// This guard disables gradient computation when created and automatically
/// restores the previous gradient mode when dropped. It's particularly useful
/// for inference operations where gradients are not needed.
///
/// # Examples
///
/// ```rust
/// use torsh_autograd::guards::NoGradGuard;
///
/// {
///     let _guard = NoGradGuard::new();
///     // Gradients are disabled in this scope
///     // ... inference operations ...
/// } // Guard is dropped here, gradient mode restored
/// ```
pub struct NoGradGuard;

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl NoGradGuard {
    /// Create a new NoGradGuard and disable gradients
    pub fn new() -> Self {
        push_grad_enabled(false);
        Self
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        pop_grad_enabled();
    }
}

/// Guard for enabling gradient computation
///
/// This guard enables gradient computation when created and automatically
/// restores the previous gradient mode when dropped. It's useful for enabling
/// gradients in specific scopes when they were previously disabled.
///
/// # Examples
///
/// ```rust
/// use torsh_autograd::guards::EnableGradGuard;
///
/// {
///     let _guard = EnableGradGuard::new();
///     // Gradients are enabled in this scope
///     // ... training operations ...
/// } // Guard is dropped here, gradient mode restored
/// ```
pub struct EnableGradGuard;

impl Default for EnableGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl EnableGradGuard {
    /// Create a new EnableGradGuard and enable gradients
    pub fn new() -> Self {
        push_grad_enabled(true);
        Self
    }
}

impl Drop for EnableGradGuard {
    fn drop(&mut self) {
        pop_grad_enabled();
    }
}

/// Disable gradient computation
///
/// Convenience function that returns a NoGradGuard for disabling gradients.
///
/// # Examples
///
/// ```rust
/// use torsh_autograd::guards::no_grad;
///
/// let _guard = no_grad();
/// // Gradients disabled until guard is dropped
/// ```
pub fn no_grad() -> NoGradGuard {
    NoGradGuard::new()
}

/// Enable gradient computation
///
/// Convenience function that returns an EnableGradGuard for enabling gradients.
///
/// # Examples
///
/// ```rust
/// use torsh_autograd::guards::enable_grad;
///
/// let _guard = enable_grad();
/// // Gradients enabled until guard is dropped
/// ```
pub fn enable_grad() -> EnableGradGuard {
    EnableGradGuard::new()
}

/// Custom guard for setting a specific gradient mode
///
/// This guard allows setting a specific gradient mode (true or false) and
/// automatically restores the previous mode when dropped.
pub struct GradModeGuard {
    _guard: Box<dyn std::any::Any>,
}

impl GradModeGuard {
    /// Create a guard with a specific gradient mode
    pub fn new(enabled: bool) -> Self {
        let guard: Box<dyn std::any::Any> = if enabled {
            Box::new(EnableGradGuard::new())
        } else {
            Box::new(NoGradGuard::new())
        };

        Self { _guard: guard }
    }

    /// Create a guard that disables gradients
    pub fn no_grad() -> Self {
        Self::new(false)
    }

    /// Create a guard that enables gradients
    pub fn enable_grad() -> Self {
        Self::new(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grad_mode::{is_grad_enabled, set_grad_enabled};

    #[test]
    fn test_no_grad_guard() {
        set_grad_enabled(true);
        assert!(is_grad_enabled());

        {
            let _guard = NoGradGuard::new();
            assert!(!is_grad_enabled());
        } // Guard dropped here

        assert!(is_grad_enabled()); // Should be restored
    }

    #[test]
    fn test_enable_grad_guard() {
        set_grad_enabled(false);
        assert!(!is_grad_enabled());

        {
            let _guard = EnableGradGuard::new();
            assert!(is_grad_enabled());
        } // Guard dropped here

        assert!(!is_grad_enabled()); // Should be restored
    }

    #[test]
    fn test_convenience_functions() {
        set_grad_enabled(true);

        {
            let _guard = no_grad();
            assert!(!is_grad_enabled());
        }

        assert!(is_grad_enabled());

        set_grad_enabled(false);

        {
            let _guard = enable_grad();
            assert!(is_grad_enabled());
        }

        assert!(!is_grad_enabled());
    }

    #[test]
    fn test_nested_guards() {
        set_grad_enabled(true);

        {
            let _outer = no_grad();
            assert!(!is_grad_enabled());

            {
                let _inner = enable_grad();
                assert!(is_grad_enabled());
            }

            assert!(!is_grad_enabled());
        }

        assert!(is_grad_enabled());
    }

    #[test]
    fn test_custom_grad_mode_guard() {
        set_grad_enabled(true);

        {
            let _guard = GradModeGuard::new(false);
            assert!(!is_grad_enabled());
        }

        assert!(is_grad_enabled());

        set_grad_enabled(false);

        {
            let _guard = GradModeGuard::new(true);
            assert!(is_grad_enabled());
        }

        assert!(!is_grad_enabled());
    }

    #[test]
    fn test_guard_defaults() {
        set_grad_enabled(true);

        {
            let _guard = NoGradGuard::default();
            assert!(!is_grad_enabled());
        }

        assert!(is_grad_enabled());

        set_grad_enabled(false);

        {
            let _guard = EnableGradGuard::default();
            assert!(is_grad_enabled());
        }

        assert!(!is_grad_enabled());
    }
}
