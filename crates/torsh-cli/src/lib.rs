//! ToRSh CLI Library
//!
//! This library provides the core functionality for the ToRSh command-line interface,
//! including configuration management, command implementations, and utilities.

pub mod commands;
pub mod config;
pub mod utils;

// Re-export commonly used types
pub use config::Config;

/// CLI version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const VERSION_MAJOR: u32 = 0;
pub const VERSION_MINOR: u32 = 1;
pub const VERSION_PATCH: u32 = 0;

/// Check if CLI version is compatible with required version
pub fn check_version(required_major: u32, required_minor: u32) -> anyhow::Result<()> {
    if VERSION_MAJOR < required_major
        || (VERSION_MAJOR == required_major && VERSION_MINOR < required_minor)
    {
        anyhow::bail!(
            "ToRSh CLI version {}.{} or higher required, but got {}.{}",
            required_major,
            required_minor,
            VERSION_MAJOR,
            VERSION_MINOR
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_check() {
        assert!(check_version(0, 1).is_ok());
        assert!(check_version(1, 0).is_err());
    }
}
