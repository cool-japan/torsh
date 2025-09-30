//! File integrity validation and security configuration
//!
//! This module handles file hash verification, model source validation,
//! and security policy enforcement.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use torsh_core::error::{Result, TorshError};

use super::signing::ModelSignature;

/// Calculate SHA-256 hash of a file
pub fn calculate_file_hash<P: AsRef<Path>>(file_path: P) -> Result<String> {
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0; 8192];

    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    let result = hasher.finalize();
    Ok(hex::encode(result))
}

/// Verify file integrity by comparing hashes
pub fn verify_file_integrity<P: AsRef<Path>>(file_path: P, expected_hash: &str) -> Result<bool> {
    let actual_hash = calculate_file_hash(file_path)?;
    Ok(actual_hash == expected_hash)
}

/// Security configuration for downloads and model loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Require signatures for all models
    pub require_signatures: bool,
    /// Only accept models signed by trusted keys
    pub require_trusted_keys: bool,
    /// Maximum age of signatures in seconds
    pub max_signature_age: Option<u64>,
    /// List of blocked model sources
    pub blocked_sources: Vec<String>,
    /// List of allowed model sources (if empty, all sources are allowed)
    pub allowed_sources: Vec<String>,
    /// Verify file integrity on load
    pub verify_integrity: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            require_signatures: false,
            require_trusted_keys: false,
            max_signature_age: Some(30 * 24 * 3600), // 30 days
            blocked_sources: Vec::new(),
            allowed_sources: Vec::new(),
            verify_integrity: true,
        }
    }
}

/// Validate a model source against security configuration
pub fn validate_model_source(url: &str, config: &SecurityConfig) -> Result<()> {
    // Check blocked sources
    for blocked in &config.blocked_sources {
        if url.contains(blocked) {
            return Err(TorshError::SecurityError(format!(
                "Model source '{}' is blocked",
                blocked
            )));
        }
    }

    // Check allowed sources (if specified)
    if !config.allowed_sources.is_empty() {
        let is_allowed = config
            .allowed_sources
            .iter()
            .any(|allowed| url.contains(allowed));
        if !is_allowed {
            return Err(TorshError::SecurityError(
                "Model source is not in the allowed list".to_string(),
            ));
        }
    }

    Ok(())
}

/// Validate signature age
pub fn validate_signature_age(signature: &ModelSignature, max_age: Option<u64>) -> Result<()> {
    if let Some(max_age_secs) = max_age {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let age = current_time.saturating_sub(signature.timestamp);
        if age > max_age_secs {
            return Err(TorshError::SecurityError(format!(
                "Signature is too old: {} seconds (max: {})",
                age, max_age_secs
            )));
        }
    }

    Ok(())
}