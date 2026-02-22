//! # RegistryConfig - Trait Implementations
//!
//! This module contains trait implementations for `RegistryConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{RegistryConfig, RetryConfig};

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            registry_url: Some("https://huggingface.co".to_string()),
            api_key: None,
            timeout_seconds: 300,
            max_concurrent: 4,
            compression: true,
            mirrors: vec![
                "https://mirror1.torsh.rs".to_string(),
                "https://mirror2.torsh.rs".to_string(),
            ],
            retry_config: RetryConfig::default(),
            bandwidth_limit: None,
            cache_ttl_seconds: 3600,
            enable_p2p: false,
            user_agent: "ToRSh Model Zoo/1.0".to_string(),
        }
    }
}
