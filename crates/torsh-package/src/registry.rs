//! Package registry client
//!
//! This module provides a client for interacting with package registries,
//! allowing users to publish, download, and search for packages.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use torsh_core::error::{Result, TorshError};

use crate::package::Package;

/// Registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Registry URL
    pub url: String,
    /// Authentication token
    pub token: Option<String>,
    /// API version
    pub api_version: String,
    /// Timeout in seconds
    pub timeout_secs: u64,
}

/// Package metadata from registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageMetadata {
    /// Package name
    pub name: String,
    /// Package version
    pub version: String,
    /// Package author
    pub author: Option<String>,
    /// Package description
    pub description: Option<String>,
    /// Package license
    pub license: Option<String>,
    /// Download count
    pub downloads: u64,
    /// Published date
    pub published_at: chrono::DateTime<chrono::Utc>,
    /// Package size in bytes
    pub size_bytes: u64,
    /// Package checksum
    pub checksum: String,
}

/// Search result from registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Total number of results
    pub total: usize,
    /// Current page
    pub page: usize,
    /// Results per page
    pub per_page: usize,
    /// Package results
    pub packages: Vec<PackageMetadata>,
}

/// Package registry client
pub struct RegistryClient {
    /// Registry configuration
    _config: RegistryConfig,
    /// HTTP client (placeholder)
    _client: Option<()>,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            url: "https://registry.torsh.ai".to_string(),
            token: None,
            api_version: "v1".to_string(),
            timeout_secs: 30,
        }
    }
}

impl RegistryConfig {
    /// Create a new registry config
    pub fn new(url: String) -> Self {
        Self {
            url,
            ..Default::default()
        }
    }

    /// Set authentication token
    pub fn with_token(mut self, token: String) -> Self {
        self.token = Some(token);
        self
    }

    /// Set API version
    pub fn with_api_version(mut self, api_version: String) -> Self {
        self.api_version = api_version;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }
}

impl RegistryClient {
    /// Create a new registry client
    pub fn new(config: RegistryConfig) -> Self {
        Self {
            _config: config,
            _client: None,
        }
    }

    /// Create a client with default configuration
    pub fn with_defaults() -> Self {
        Self::new(RegistryConfig::default())
    }

    /// Get the configuration
    pub fn config(&self) -> &RegistryConfig {
        &self._config
    }

    /// Publish a package to the registry
    pub async fn publish(&self, package: &Package) -> Result<PackageMetadata> {
        // Validate package before publishing
        package.verify()?;

        // In a real implementation, this would:
        // 1. Serialize the package
        // 2. Calculate checksum
        // 3. Upload to registry
        // 4. Return metadata

        // Placeholder implementation
        let metadata = PackageMetadata {
            name: package.name().to_string(),
            version: package.get_version().to_string(),
            author: package.metadata().author.clone(),
            description: package.metadata().description.clone(),
            license: package.metadata().license.clone(),
            downloads: 0,
            published_at: chrono::Utc::now(),
            size_bytes: 0, // Would be actual package size
            checksum: "placeholder".to_string(),
        };

        Ok(metadata)
    }

    /// Download a package from the registry
    pub async fn download(&self, name: &str, version: &str) -> Result<Package> {
        // In a real implementation, this would:
        // 1. Query registry for package location
        // 2. Download package file
        // 3. Verify checksum
        // 4. Deserialize and return package

        // Placeholder implementation
        Err(TorshError::InvalidArgument(format!(
            "Package {} version {} not found in registry",
            name, version
        )))
    }

    /// Search for packages in the registry
    pub async fn search(&self, _query: &str, page: usize, per_page: usize) -> Result<SearchResult> {
        // In a real implementation, this would query the registry API

        // Placeholder implementation
        Ok(SearchResult {
            total: 0,
            page,
            per_page,
            packages: Vec::new(),
        })
    }

    /// Get package metadata without downloading
    pub async fn get_metadata(&self, name: &str, version: &str) -> Result<PackageMetadata> {
        // In a real implementation, this would query the registry API

        // Placeholder implementation
        Err(TorshError::InvalidArgument(format!(
            "Package {} version {} not found",
            name, version
        )))
    }

    /// List all versions of a package
    pub async fn list_versions(&self, _name: &str) -> Result<Vec<String>> {
        // In a real implementation, this would query the registry API

        // Placeholder implementation
        Ok(Vec::new())
    }

    /// Check if a package exists in the registry
    pub async fn exists(&self, name: &str, version: &str) -> Result<bool> {
        match self.get_metadata(name, version).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Yank (deprecate) a package version
    pub async fn yank(&self, _name: &str, _version: &str) -> Result<()> {
        // In a real implementation, this would mark the version as yanked

        Ok(())
    }

    /// Unyank a previously yanked version
    pub async fn unyank(&self, _name: &str, _version: &str) -> Result<()> {
        // In a real implementation, this would remove the yank marker

        Ok(())
    }

    /// Get download statistics for a package
    pub async fn get_stats(&self, _name: &str) -> Result<HashMap<String, u64>> {
        // In a real implementation, this would return download stats per version

        Ok(HashMap::new())
    }
}

/// Local package cache for registry
pub struct PackageCache {
    /// Cache directory
    cache_dir: std::path::PathBuf,
    /// Maximum cache size in bytes
    max_size_bytes: u64,
    /// Cache index
    index: HashMap<String, PackageMetadata>,
}

impl PackageCache {
    /// Create a new package cache
    pub fn new<P: AsRef<Path>>(cache_dir: P, max_size_bytes: u64) -> Result<Self> {
        let cache_dir = cache_dir.as_ref().to_path_buf();

        if !cache_dir.exists() {
            std::fs::create_dir_all(&cache_dir)
                .map_err(|e| TorshError::IoError(format!("Failed to create cache dir: {}", e)))?;
        }

        Ok(Self {
            cache_dir,
            max_size_bytes,
            index: HashMap::new(),
        })
    }

    /// Get a package from cache
    pub fn get(&self, name: &str, version: &str) -> Result<Option<Package>> {
        let cache_key = format!("{}_{}", name, version);
        let cache_file = self.cache_dir.join(format!("{}.torsh", cache_key));

        if cache_file.exists() {
            Package::load(&cache_file).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Store a package in cache
    pub fn put(&mut self, package: &Package) -> Result<()> {
        let cache_key = format!("{}_{}", package.name(), package.get_version());
        let cache_file = self.cache_dir.join(format!("{}.torsh", cache_key));

        package.save(&cache_file)?;

        // Update index
        let metadata = PackageMetadata {
            name: package.name().to_string(),
            version: package.get_version().to_string(),
            author: package.metadata().author.clone(),
            description: package.metadata().description.clone(),
            license: package.metadata().license.clone(),
            downloads: 0,
            published_at: chrono::Utc::now(),
            size_bytes: cache_file.metadata().map(|m| m.len()).unwrap_or(0),
            checksum: "".to_string(),
        };

        self.index.insert(cache_key, metadata);

        // Check cache size and evict if necessary
        self.evict_if_needed()?;

        Ok(())
    }

    /// Clear the cache
    pub fn clear(&mut self) -> Result<()> {
        std::fs::remove_dir_all(&self.cache_dir)
            .map_err(|e| TorshError::IoError(format!("Failed to clear cache: {}", e)))?;

        std::fs::create_dir_all(&self.cache_dir)
            .map_err(|e| TorshError::IoError(format!("Failed to recreate cache dir: {}", e)))?;

        self.index.clear();

        Ok(())
    }

    /// Get current cache size in bytes
    pub fn size(&self) -> u64 {
        self.index.values().map(|m| m.size_bytes).sum()
    }

    /// Evict entries if cache size exceeds maximum
    fn evict_if_needed(&mut self) -> Result<()> {
        while self.size() > self.max_size_bytes && !self.index.is_empty() {
            // Find oldest entry
            if let Some((key, _)) = self
                .index
                .iter()
                .min_by_key(|(_, m)| m.published_at)
                .map(|(k, m)| (k.clone(), m.clone()))
            {
                let cache_file = self.cache_dir.join(format!("{}.torsh", key));
                if cache_file.exists() {
                    std::fs::remove_file(&cache_file).map_err(|e| {
                        TorshError::IoError(format!("Failed to remove cache file: {}", e))
                    })?;
                }
                self.index.remove(&key);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_registry_config() {
        let config = RegistryConfig::new("https://my-registry.com".to_string())
            .with_token("secret_token".to_string())
            .with_api_version("v2".to_string())
            .with_timeout(60);

        assert_eq!(config.url, "https://my-registry.com");
        assert_eq!(config.token, Some("secret_token".to_string()));
        assert_eq!(config.api_version, "v2");
        assert_eq!(config.timeout_secs, 60);
    }

    #[test]
    fn test_registry_client_creation() {
        let config = RegistryConfig::default();
        let client = RegistryClient::new(config);

        assert!(client.config().url.contains("registry"));
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_publish_placeholder() {
        let client = RegistryClient::with_defaults();
        let package = Package::new("test_model".to_string(), "1.0.0".to_string());

        let metadata = client.publish(&package).await.unwrap();

        assert_eq!(metadata.name, "test_model");
        assert_eq!(metadata.version, "1.0.0");
    }

    #[test]
    fn test_package_cache() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let mut cache = PackageCache::new(temp_dir.path(), 10_000_000)?;

        let package = Package::new("cached_model".to_string(), "1.0.0".to_string());

        cache.put(&package)?;

        let retrieved = cache.get("cached_model", "1.0.0")?.unwrap();

        assert_eq!(retrieved.name(), package.name());

        Ok(())
    }

    #[test]
    fn test_cache_eviction() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let mut cache = PackageCache::new(temp_dir.path(), 1000)?; // Small cache

        // Add packages until eviction happens
        for i in 0..10 {
            let mut package = Package::new(format!("model_{}", i), "1.0.0".to_string());
            // Add some data to make packages larger
            package.add_source_file("code", &"x".repeat(200))?;
            cache.put(&package)?;
        }

        // Cache should have evicted some entries
        assert!(cache.index.len() < 10);

        Ok(())
    }

    #[test]
    fn test_cache_clear() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let mut cache = PackageCache::new(temp_dir.path(), 10_000_000)?;

        let package = Package::new("test".to_string(), "1.0.0".to_string());
        cache.put(&package)?;

        cache.clear()?;

        assert_eq!(cache.index.len(), 0);
        assert_eq!(cache.size(), 0);

        Ok(())
    }
}
