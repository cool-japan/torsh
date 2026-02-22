//! Module construction and configuration system
//!
//! This module provides standardized construction patterns and configuration
//! management for neural network modules.

use torsh_core::device::DeviceType;
use torsh_core::error::Result;

// Conditional imports for std/no_std compatibility
#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use hashbrown::HashMap;

#[cfg(feature = "serialize")]
use serde_json;

/// Helper trait for module construction patterns
///
/// This trait provides standardized construction patterns for modules,
/// ensuring consistent error handling and ergonomics across all implementations.
pub trait ModuleConstruct {
    /// Type returned by the constructor
    type Output;

    /// Attempt to create a module, returning Result for error handling
    ///
    /// This is the primary constructor that should be implemented.
    fn try_new() -> Result<Self::Output>;

    /// Create a module with panic on error (for convenience)
    ///
    /// This method provides a convenient interface for cases where
    /// construction failure is not expected.
    fn new() -> Self::Output
    where
        Self::Output: Sized,
    {
        Self::try_new().expect("Module construction failed")
    }

    /// Create a module with default parameters
    ///
    /// Default implementation delegates to `try_new()`. Override if your
    /// module supports different default configurations.
    fn default() -> Result<Self::Output> {
        Self::try_new()
    }

    /// Create a module with a specific configuration
    ///
    /// Default implementation delegates to `try_new()`. Override if your
    /// module supports configuration-based construction.
    fn with_config(_config: &ModuleConfig) -> Result<Self::Output> {
        Self::try_new()
    }
}

/// Generic module configuration
///
/// This provides a standard configuration interface that can be extended
/// by specific module types.
#[derive(Debug, Clone)]
pub struct ModuleConfig {
    /// Training mode
    pub training: bool,
    /// Target device
    pub device: DeviceType,
    /// Whether to use bias terms
    pub bias: bool,
    /// Dropout probability
    pub dropout: f32,
    /// Custom parameters
    #[cfg(feature = "serialize")]
    pub custom: HashMap<String, serde_json::Value>,
    /// Custom parameters (placeholder when serialize feature is disabled)
    #[cfg(not(feature = "serialize"))]
    pub custom: HashMap<String, String>,
}

impl Default for ModuleConfig {
    fn default() -> Self {
        Self {
            training: true,
            device: DeviceType::Cpu,
            bias: true,
            dropout: 0.0,
            custom: HashMap::new(),
        }
    }
}

impl ModuleConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set training mode
    pub fn training(mut self, training: bool) -> Self {
        self.training = training;
        self
    }

    /// Set device
    pub fn device(mut self, device: DeviceType) -> Self {
        self.device = device;
        self
    }

    /// Set bias usage
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Set dropout probability
    pub fn dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Add a custom parameter
    #[cfg(feature = "serialize")]
    pub fn custom_param<T: serde::Serialize>(mut self, name: &str, value: T) -> Self {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.custom.insert(name.to_string(), json_value);
        }
        self
    }

    /// Add a custom parameter (simplified version when serialize feature is disabled)
    #[cfg(not(feature = "serialize"))]
    pub fn custom_param<T: std::fmt::Display>(mut self, name: &str, value: T) -> Self {
        self.custom.insert(name.to_string(), value.to_string());
        self
    }

    /// Get a custom parameter
    #[cfg(feature = "serialize")]
    pub fn get_custom<T: serde::de::DeserializeOwned>(&self, name: &str) -> Option<T> {
        self.custom
            .get(name)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Get a custom parameter (simplified version when serialize feature is disabled)
    #[cfg(not(feature = "serialize"))]
    pub fn get_custom(&self, name: &str) -> Option<String> {
        self.custom.get(name).cloned()
    }
}

/// Macro to implement standardized constructors
#[macro_export]
macro_rules! impl_module_constructor {
    ($module_type:ty, $constructor:expr) => {
        impl ModuleConstruct for $module_type {
            type Output = $module_type;

            fn try_new() -> Result<Self::Output> {
                $constructor
            }
        }
    };
}
