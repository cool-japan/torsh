//! Cache management for ToRSh Hub

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use torsh_core::error::{Result, TorshError};

/// Cache manager for hub models and repositories
#[derive(Debug, Clone)]
pub struct CacheManager {
    cache_dir: PathBuf,
    metadata_file: PathBuf,
    metadata: CacheMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheMetadata {
    version: String,
    repositories: HashMap<String, RepoMetadata>,
    models: HashMap<String, ModelMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RepoMetadata {
    owner: String,
    name: String,
    branch: String,
    last_updated: chrono::DateTime<chrono::Utc>,
    size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelMetadata {
    repo: String,
    name: String,
    version: String,
    hash: String,
    size_bytes: u64,
    last_accessed: chrono::DateTime<chrono::Utc>,
}

impl CacheManager {
    /// Create a new cache manager
    pub fn new(cache_dir: &Path) -> Result<Self> {
        fs::create_dir_all(cache_dir)?;

        let metadata_file = cache_dir.join("cache_metadata.json");
        let metadata = if metadata_file.exists() {
            load_metadata(&metadata_file)?
        } else {
            CacheMetadata {
                version: "1.0".to_string(),
                repositories: HashMap::new(),
                models: HashMap::new(),
            }
        };

        Ok(Self {
            cache_dir: cache_dir.to_path_buf(),
            metadata_file,
            metadata,
        })
    }

    /// Get repository directory
    pub fn get_repo_dir(&self, owner: &str, repo: &str, branch: &str) -> PathBuf {
        self.cache_dir
            .join("repositories")
            .join(format!("{}__{}__{}", owner, repo, branch))
    }

    /// Get model cache path
    pub fn get_model_path(&self, repo: &str, model: &str, version: &str) -> PathBuf {
        self.cache_dir
            .join("models")
            .join(repo.replace('/', "__"))
            .join(format!("{}_{}.torsh", model, version))
    }

    /// Check if repository is cached
    pub fn is_repo_cached(&self, owner: &str, repo: &str, branch: &str) -> bool {
        let key = format!("{}/{}/{}", owner, repo, branch);
        self.metadata.repositories.contains_key(&key)
    }

    /// Check if model is cached
    pub fn is_model_cached(&self, repo: &str, model: &str, version: &str) -> bool {
        let key = format!("{}/{}@{}", repo, model, version);
        self.metadata.models.contains_key(&key)
    }

    /// Add repository to cache
    pub fn add_repo(
        &mut self,
        owner: &str,
        repo: &str,
        branch: &str,
        size_bytes: u64,
    ) -> Result<()> {
        let key = format!("{}/{}/{}", owner, repo, branch);
        self.metadata.repositories.insert(
            key,
            RepoMetadata {
                owner: owner.to_string(),
                name: repo.to_string(),
                branch: branch.to_string(),
                last_updated: chrono::Utc::now(),
                size_bytes,
            },
        );

        self.save_metadata()
    }

    /// Add model to cache
    pub fn add_model(
        &mut self,
        repo: &str,
        model: &str,
        version: &str,
        hash: &str,
        size_bytes: u64,
    ) -> Result<()> {
        let key = format!("{}/{}@{}", repo, model, version);
        self.metadata.models.insert(
            key,
            ModelMetadata {
                repo: repo.to_string(),
                name: model.to_string(),
                version: version.to_string(),
                hash: hash.to_string(),
                size_bytes,
                last_accessed: chrono::Utc::now(),
            },
        );

        self.save_metadata()
    }

    /// Update model access time
    pub fn touch_model(&mut self, repo: &str, model: &str, version: &str) -> Result<()> {
        let key = format!("{}/{}@{}", repo, model, version);
        if let Some(metadata) = self.metadata.models.get_mut(&key) {
            metadata.last_accessed = chrono::Utc::now();
            self.save_metadata()?;
        }
        Ok(())
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStats {
        let total_repos = self.metadata.repositories.len();
        let total_models = self.metadata.models.len();
        let total_size = self
            .metadata
            .repositories
            .values()
            .map(|r| r.size_bytes)
            .sum::<u64>()
            + self
                .metadata
                .models
                .values()
                .map(|m| m.size_bytes)
                .sum::<u64>();

        let oldest_access = self
            .metadata
            .models
            .values()
            .map(|m| m.last_accessed)
            .min()
            .unwrap_or_else(chrono::Utc::now);

        let newest_access = self
            .metadata
            .models
            .values()
            .map(|m| m.last_accessed)
            .max()
            .unwrap_or_else(chrono::Utc::now);

        CacheStats {
            total_repositories: total_repos,
            total_models: total_models,
            total_size_bytes: total_size,
            total_size_formatted: format_bytes(total_size),
            oldest_access,
            newest_access,
            hit_rate: self.calculate_hit_rate(),
        }
    }

    /// Perform cache cleanup based on size and age limits
    pub fn cleanup_cache(
        &mut self,
        max_size_bytes: u64,
        max_age_days: u32,
    ) -> Result<CacheCleanupResult> {
        let mut cleanup_result = CacheCleanupResult::default();
        let cutoff_date = chrono::Utc::now() - chrono::Duration::days(max_age_days as i64);

        // First pass: Remove old models
        let mut models_to_remove = Vec::new();
        for (key, metadata) in &self.metadata.models {
            if metadata.last_accessed < cutoff_date {
                models_to_remove.push(key.clone());
            }
        }

        for key in models_to_remove {
            if let Some(metadata) = self.metadata.models.remove(&key) {
                let model_path =
                    self.get_model_path(&metadata.repo, &metadata.name, &metadata.version);
                if model_path.exists() {
                    fs::remove_file(&model_path)?;
                    cleanup_result.freed_bytes += metadata.size_bytes;
                    cleanup_result.models_removed += 1;
                }
            }
        }

        // Second pass: Remove models by size if still over limit
        let current_size = self.get_cache_stats().total_size_bytes;
        if current_size > max_size_bytes {
            let size_to_free = current_size - max_size_bytes;
            let mut candidates: Vec<_> = self.metadata.models.iter().collect();
            candidates.sort_by_key(|(_, metadata)| metadata.last_accessed);

            let mut freed_bytes = 0u64;
            let mut models_to_remove = Vec::new();

            for (key, metadata) in candidates {
                if freed_bytes >= size_to_free {
                    break;
                }
                models_to_remove.push(key.clone());
                freed_bytes += metadata.size_bytes;
            }

            for key in models_to_remove {
                if let Some(metadata) = self.metadata.models.remove(&key) {
                    let model_path =
                        self.get_model_path(&metadata.repo, &metadata.name, &metadata.version);
                    if model_path.exists() {
                        fs::remove_file(&model_path)?;
                        cleanup_result.freed_bytes += metadata.size_bytes;
                        cleanup_result.models_removed += 1;
                    }
                }
            }
        }

        self.save_metadata()?;
        cleanup_result.cleanup_duration = std::time::Instant::now().elapsed();

        Ok(cleanup_result)
    }

    /// Compress cache files to save disk space
    pub fn compress_cache(&mut self) -> Result<CompressionResult> {
        let mut compression_result = CompressionResult::default();
        let cache_models_dir = self.cache_dir.join("models");

        if !cache_models_dir.exists() {
            return Ok(compression_result);
        }

        // First collect the keys and metadata info to avoid borrowing conflicts
        let models_info: Vec<(String, String, String, String)> = self
            .metadata
            .models
            .iter()
            .map(|(key, metadata)| {
                (
                    key.clone(),
                    metadata.repo.clone(),
                    metadata.name.clone(),
                    metadata.version.clone(),
                )
            })
            .collect();

        for (key, repo, name, version) in models_info {
            let model_path = self.get_model_path(&repo, &name, &version);
            let metadata = self.metadata.models.get_mut(&key).unwrap();
            let compressed_path = model_path.with_extension("torsh.gz");

            if model_path.exists() && !compressed_path.exists() {
                match compress_file(&model_path, &compressed_path) {
                    Ok(compression_stats) => {
                        // Update metadata to point to compressed file
                        let old_size = metadata.size_bytes;
                        metadata.size_bytes = compression_stats.compressed_size;

                        // Remove original file
                        fs::remove_file(&model_path)?;

                        compression_result.files_compressed += 1;
                        compression_result.original_bytes += compression_stats.original_size;
                        compression_result.compressed_bytes += compression_stats.compressed_size;
                        compression_result.bytes_saved +=
                            old_size - compression_stats.compressed_size;
                    }
                    Err(e) => {
                        eprintln!("Failed to compress {}: {}", model_path.display(), e);
                        compression_result.compression_failures += 1;
                    }
                }
            }
        }

        self.save_metadata()?;
        Ok(compression_result)
    }

    /// Calculate cache hit rate (simplified implementation)
    fn calculate_hit_rate(&self) -> f32 {
        // This would need to be implemented with proper hit/miss tracking
        // For now, return a placeholder based on cache usage
        if self.metadata.models.is_empty() {
            0.0
        } else {
            0.85 // Placeholder value
        }
    }

    /// Validate cache integrity
    pub fn validate_cache(&self) -> Result<CacheValidationResult> {
        let mut validation_result = CacheValidationResult::default();

        // Check model files
        for (key, metadata) in &self.metadata.models {
            let model_path = self.get_model_path(&metadata.repo, &metadata.name, &metadata.version);
            let compressed_path = model_path.with_extension("torsh.gz");

            if model_path.exists() {
                validation_result.valid_models += 1;
            } else if compressed_path.exists() {
                validation_result.valid_models += 1;
            } else {
                validation_result.missing_files.push(key.clone());
                validation_result.invalid_models += 1;
            }
        }

        // Check repository directories
        for (key, metadata) in &self.metadata.repositories {
            let repo_dir = self.get_repo_dir(&metadata.owner, &metadata.name, &metadata.branch);
            if repo_dir.exists() {
                validation_result.valid_repositories += 1;
            } else {
                validation_result.missing_directories.push(key.clone());
                validation_result.invalid_repositories += 1;
            }
        }

        Ok(validation_result)
    }

    /// Get cache size
    pub fn get_cache_size(&self) -> u64 {
        let repo_size: u64 = self
            .metadata
            .repositories
            .values()
            .map(|r| r.size_bytes)
            .sum();

        let model_size: u64 = self.metadata.models.values().map(|m| m.size_bytes).sum();

        repo_size + model_size
    }

    /// Clean old cache entries
    pub fn clean_cache(
        &mut self,
        max_size_bytes: Option<u64>,
        max_age_days: Option<u64>,
    ) -> Result<()> {
        let now = chrono::Utc::now();

        // Remove old models
        if let Some(max_age) = max_age_days {
            let cutoff = now - chrono::Duration::days(max_age as i64);

            let old_models: Vec<String> = self
                .metadata
                .models
                .iter()
                .filter(|(_, m)| m.last_accessed < cutoff)
                .map(|(k, _)| k.clone())
                .collect();

            for key in old_models {
                if let Some(model) = self.metadata.models.remove(&key) {
                    let path = self.get_model_path(&model.repo, &model.name, &model.version);
                    let _ = fs::remove_file(path);
                }
            }
        }

        // Remove to fit size limit
        if let Some(max_size) = max_size_bytes {
            while self.get_cache_size() > max_size {
                // Find least recently used model
                let lru_key = self
                    .metadata
                    .models
                    .iter()
                    .min_by_key(|(_, m)| m.last_accessed)
                    .map(|(k, _)| k.clone());

                if let Some(key) = lru_key {
                    if let Some(model) = self.metadata.models.remove(&key) {
                        let path = self.get_model_path(&model.repo, &model.name, &model.version);
                        let _ = fs::remove_file(path);
                    }
                } else {
                    break;
                }
            }
        }

        self.save_metadata()
    }

    /// Clear entire cache
    pub fn clear_cache(&mut self) -> Result<()> {
        // Remove all files
        if self.cache_dir.exists() {
            fs::remove_dir_all(&self.cache_dir)?;
            fs::create_dir_all(&self.cache_dir)?;
        }

        // Reset metadata
        self.metadata.repositories.clear();
        self.metadata.models.clear();

        self.save_metadata()
    }

    /// Save metadata to disk
    fn save_metadata(&self) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| TorshError::SerializationError(e.to_string()))?;
        let mut file = File::create(&self.metadata_file)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }
}

/// Load metadata from file
fn load_metadata(path: &Path) -> Result<CacheMetadata> {
    let mut file = File::open(path)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;

    serde_json::from_str(&content).map_err(|e| TorshError::SerializationError(e.to_string()))
}

/// Get directory size recursively
pub fn get_dir_size(path: &Path) -> Result<u64> {
    let mut size = 0;

    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                size += get_dir_size(&path)?;
            } else {
                size += entry.metadata()?.len();
            }
        }
    } else if path.is_file() {
        size = fs::metadata(path)?.len();
    }

    Ok(size)
}

/// Verify file hash
pub fn verify_file_hash(path: &Path, expected_hash: &str) -> Result<bool> {
    use sha2::{Digest, Sha256};

    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 8192];

    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    let result = hasher.finalize();
    let actual_hash = hex::encode(result);

    Ok(actual_hash == expected_hash)
}

/// Cache statistics structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_repositories: usize,
    pub total_models: usize,
    pub total_size_bytes: u64,
    pub total_size_formatted: String,
    pub oldest_access: chrono::DateTime<chrono::Utc>,
    pub newest_access: chrono::DateTime<chrono::Utc>,
    pub hit_rate: f32,
}

/// Cache cleanup result
#[derive(Debug, Clone, Default)]
pub struct CacheCleanupResult {
    pub models_removed: usize,
    pub repositories_removed: usize,
    pub freed_bytes: u64,
    pub cleanup_duration: std::time::Duration,
}

/// Compression result
#[derive(Debug, Clone, Default)]
pub struct CompressionResult {
    pub files_compressed: usize,
    pub original_bytes: u64,
    pub compressed_bytes: u64,
    pub bytes_saved: u64,
    pub compression_failures: usize,
}

/// Cache validation result
#[derive(Debug, Clone, Default)]
pub struct CacheValidationResult {
    pub valid_models: usize,
    pub invalid_models: usize,
    pub valid_repositories: usize,
    pub invalid_repositories: usize,
    pub missing_files: Vec<String>,
    pub missing_directories: Vec<String>,
}

/// File compression statistics
#[derive(Debug, Clone)]
pub struct FileCompressionStats {
    pub original_size: u64,
    pub compressed_size: u64,
    pub compression_ratio: f32,
}

/// Format bytes into human-readable string
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    const THRESHOLD: f64 = 1024.0;

    if bytes == 0 {
        return "0 B".to_string();
    }

    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= THRESHOLD && unit_index < UNITS.len() - 1 {
        size /= THRESHOLD;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

/// Compress a file using gzip compression
pub fn compress_file(input_path: &Path, output_path: &Path) -> Result<FileCompressionStats> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::copy;

    let input_file = File::open(input_path)?;
    let output_file = File::create(output_path)?;
    let mut encoder = GzEncoder::new(output_file, Compression::default());

    let original_size = input_file.metadata()?.len();

    let mut input_reader = std::io::BufReader::new(input_file);
    copy(&mut input_reader, &mut encoder)?;
    encoder.finish()?;

    let compressed_size = fs::metadata(output_path)?.len();
    let compression_ratio = if original_size > 0 {
        compressed_size as f32 / original_size as f32
    } else {
        1.0
    };

    Ok(FileCompressionStats {
        original_size,
        compressed_size,
        compression_ratio,
    })
}

/// Decompress a gzip file
pub fn decompress_file(input_path: &Path, output_path: &Path) -> Result<()> {
    use flate2::read::GzDecoder;
    use std::io::copy;

    let input_file = File::open(input_path)?;
    let mut decoder = GzDecoder::new(input_file);
    let mut output_file = File::create(output_path)?;

    copy(&mut decoder, &mut output_file)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_cache_manager() {
        let temp_dir = TempDir::new().unwrap();
        let mut cache = CacheManager::new(temp_dir.path()).unwrap();

        // Test repository caching
        assert!(!cache.is_repo_cached("owner", "repo", "main"));
        cache.add_repo("owner", "repo", "main", 1000).unwrap();
        assert!(cache.is_repo_cached("owner", "repo", "main"));

        // Test model caching
        assert!(!cache.is_model_cached("owner/repo", "model", "v1.0"));
        cache
            .add_model("owner/repo", "model", "v1.0", "hash123", 2000)
            .unwrap();
        assert!(cache.is_model_cached("owner/repo", "model", "v1.0"));

        // Test cache size
        assert_eq!(cache.get_cache_size(), 3000);
    }

    #[test]
    fn test_cache_cleaning() {
        let temp_dir = TempDir::new().unwrap();
        let mut cache = CacheManager::new(temp_dir.path()).unwrap();

        // Add some models
        cache
            .add_model("repo1", "model1", "v1", "hash1", 1000)
            .unwrap();

        // Sleep a tiny bit to ensure different timestamps
        std::thread::sleep(std::time::Duration::from_millis(10));

        cache
            .add_model("repo2", "model2", "v1", "hash2", 2000)
            .unwrap();

        // Touch the first model to make it more recently accessed
        cache.touch_model("repo1", "model1", "v1").unwrap();

        // Clean with size limit - should remove the least recently accessed (model2)
        cache.clean_cache(Some(1500), None).unwrap();

        // Should have removed model2 (2000 bytes) and kept model1 (1000 bytes)
        assert_eq!(cache.metadata.models.len(), 1);
        assert!(cache.is_model_cached("repo1", "model1", "v1"));
    }
}
