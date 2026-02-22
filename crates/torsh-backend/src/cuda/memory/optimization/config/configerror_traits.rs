//! # ConfigError - Trait Implementations
//!
//! This module contains trait implementations for `ConfigError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//! - `Error`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ConfigError;

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::ConfigurationNotFound(id) => {
                write!(f, "Configuration not found: {}", id)
            }
            ConfigError::ConfigurationConflict(msg) => {
                write!(f, "Configuration conflict: {}", msg)
            }
            ConfigError::ConfigurationInvalid(msg) => {
                write!(f, "Configuration invalid: {}", msg)
            }
            ConfigError::ValidationFailed(errors) => {
                write!(f, "Validation failed: {:?}", errors)
            }
            ConfigError::VersionNotFound(msg) => write!(f, "Version not found: {}", msg),
            ConfigError::ParseError(field) => {
                write!(f, "Parse error in field: {}", field)
            }
            ConfigError::SerializationError(msg) => {
                write!(f, "Serialization error: {}", msg)
            }
            ConfigError::PersistenceError(msg) => write!(f, "Persistence error: {}", msg),
            ConfigError::BackupError(msg) => write!(f, "Backup error: {}", msg),
            ConfigError::RestoreError(msg) => write!(f, "Restore error: {}", msg),
            ConfigError::SynchronizationError(msg) => {
                write!(f, "Synchronization error: {}", msg)
            }
            ConfigError::MigrationError(msg) => write!(f, "Migration error: {}", msg),
            ConfigError::TemplateError(msg) => write!(f, "Template error: {}", msg),
            ConfigError::SchemaError(msg) => write!(f, "Schema error: {}", msg),
            ConfigError::ImportValidationFailed(errors) => {
                write!(f, "Import validation failed: {:?}", errors)
            }
            ConfigError::ExportError(msg) => write!(f, "Export error: {}", msg),
            ConfigError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            ConfigError::AccessDenied(msg) => write!(f, "Access denied: {}", msg),
            ConfigError::LockError => write!(f, "Failed to acquire lock"),
            ConfigError::IOError(msg) => write!(f, "I/O error: {}", msg),
            ConfigError::NetworkError(msg) => write!(f, "Network error: {}", msg),
        }
    }
}

impl std::error::Error for ConfigError {}

