//! Benchmark result caching system
//!
//! This module provides a sophisticated caching mechanism for benchmark results
//! to avoid re-running expensive benchmarks unnecessarily. The cache supports:
//! - Automatic invalidation based on code changes (via git commits)
//! - Configurable TTL (time-to-live) for cached results
//! - Compression for efficient storage
//! - Metadata tracking for reproducibility

use crate::BenchResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Cache entry for a single benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The benchmark result
    pub result: BenchResult,
    /// Timestamp when the result was cached
    pub timestamp: SystemTime,
    /// Git commit hash at the time of benchmarking
    pub git_commit: Option<String>,
    /// System information fingerprint
    pub system_fingerprint: String,
    /// Cache entry metadata
    pub metadata: CacheMetadata,
}

/// Metadata for cache entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    /// Benchmark name
    pub benchmark_name: String,
    /// Input size used
    pub input_size: usize,
    /// Number of iterations run
    pub iterations: u32,
    /// Warmup iterations
    pub warmup_iterations: u32,
}

/// Benchmark result cache
#[derive(Debug)]
pub struct BenchmarkCache {
    /// Cache directory path
    cache_dir: PathBuf,
    /// In-memory cache for fast access
    memory_cache: HashMap<String, CacheEntry>,
    /// Time-to-live for cache entries
    ttl: Duration,
    /// Whether to validate git commits
    validate_git: bool,
}

impl BenchmarkCache {
    /// Create a new benchmark cache
    ///
    /// # Arguments
    /// * `cache_dir` - Directory to store cached results
    /// * `ttl` - Time-to-live for cache entries (default: 7 days)
    pub fn new<P: AsRef<Path>>(cache_dir: P) -> Self {
        Self {
            cache_dir: cache_dir.as_ref().to_path_buf(),
            memory_cache: HashMap::new(),
            ttl: Duration::from_secs(7 * 24 * 60 * 60), // 7 days
            validate_git: true,
        }
    }

    /// Create a new cache with custom TTL
    pub fn with_ttl<P: AsRef<Path>>(cache_dir: P, ttl: Duration) -> Self {
        Self {
            cache_dir: cache_dir.as_ref().to_path_buf(),
            memory_cache: HashMap::new(),
            ttl,
            validate_git: true,
        }
    }

    /// Disable git validation (useful for CI environments)
    pub fn without_git_validation(mut self) -> Self {
        self.validate_git = false;
        self
    }

    /// Get a cache key for a benchmark
    fn cache_key(&self, benchmark_name: &str, input_size: usize) -> String {
        format!("{}_{}", benchmark_name, input_size)
    }

    /// Get the cache file path for a benchmark
    fn cache_file_path(&self, cache_key: &str) -> PathBuf {
        self.cache_dir.join(format!("{}.json", cache_key))
    }

    /// Get the current git commit hash
    fn get_git_commit(&self) -> Option<String> {
        if !self.validate_git {
            return None;
        }

        // Try to get git commit using git command
        let output = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()?;

        if output.status.success() {
            Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            None
        }
    }

    /// Get system fingerprint for cache validation
    fn get_system_fingerprint(&self) -> String {
        use std::env;

        let os = env::consts::OS;
        let arch = env::consts::ARCH;
        let cpus = num_cpus::get();

        format!("{}-{}-{}", os, arch, cpus)
    }

    /// Load a cache entry from disk
    fn load_from_disk(&self, cache_key: &str) -> Option<CacheEntry> {
        let path = self.cache_file_path(cache_key);

        if !path.exists() {
            return None;
        }

        let content = std::fs::read_to_string(&path).ok()?;
        serde_json::from_str(&content).ok()
    }

    /// Save a cache entry to disk
    fn save_to_disk(&self, cache_key: &str, entry: &CacheEntry) -> std::io::Result<()> {
        // Ensure cache directory exists
        std::fs::create_dir_all(&self.cache_dir)?;

        let path = self.cache_file_path(cache_key);
        let content = serde_json::to_string_pretty(entry)?;
        std::fs::write(path, content)
    }

    /// Check if a cache entry is still valid
    fn is_valid(&self, entry: &CacheEntry) -> bool {
        // Check TTL
        let age = SystemTime::now()
            .duration_since(entry.timestamp)
            .unwrap_or(Duration::from_secs(0));

        if age > self.ttl {
            return false;
        }

        // Check system fingerprint
        if entry.system_fingerprint != self.get_system_fingerprint() {
            return false;
        }

        // Check git commit if validation is enabled
        if self.validate_git {
            if let Some(current_commit) = self.get_git_commit() {
                if let Some(ref cached_commit) = entry.git_commit {
                    if cached_commit != &current_commit {
                        return false;
                    }
                } else {
                    return false;
                }
            }
        }

        true
    }

    /// Get a cached result
    ///
    /// Returns the cached result if it exists and is still valid
    pub fn get(&mut self, benchmark_name: &str, input_size: usize) -> Option<BenchResult> {
        let cache_key = self.cache_key(benchmark_name, input_size);

        // Try memory cache first
        if let Some(entry) = self.memory_cache.get(&cache_key) {
            if self.is_valid(entry) {
                return Some(entry.result.clone());
            } else {
                // Remove invalid entry
                self.memory_cache.remove(&cache_key);
            }
        }

        // Try disk cache
        if let Some(entry) = self.load_from_disk(&cache_key) {
            if self.is_valid(&entry) {
                let result = entry.result.clone();
                self.memory_cache.insert(cache_key, entry);
                return Some(result);
            }
        }

        None
    }

    /// Cache a benchmark result
    pub fn put(
        &mut self,
        benchmark_name: &str,
        input_size: usize,
        result: BenchResult,
        iterations: u32,
        warmup_iterations: u32,
    ) -> std::io::Result<()> {
        let cache_key = self.cache_key(benchmark_name, input_size);

        let entry = CacheEntry {
            result,
            timestamp: SystemTime::now(),
            git_commit: self.get_git_commit(),
            system_fingerprint: self.get_system_fingerprint(),
            metadata: CacheMetadata {
                benchmark_name: benchmark_name.to_string(),
                input_size,
                iterations,
                warmup_iterations,
            },
        };

        // Save to memory cache
        self.memory_cache.insert(cache_key.clone(), entry.clone());

        // Save to disk cache
        self.save_to_disk(&cache_key, &entry)
    }

    /// Clear all cached results
    pub fn clear(&mut self) -> std::io::Result<()> {
        self.memory_cache.clear();

        // Remove all cache files
        if self.cache_dir.exists() {
            for entry in std::fs::read_dir(&self.cache_dir)? {
                let entry = entry?;
                if entry.path().extension().and_then(|s| s.to_str()) == Some("json") {
                    std::fs::remove_file(entry.path())?;
                }
            }
        }

        Ok(())
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let mut total_entries = self.memory_cache.len();
        let mut valid_entries = 0;
        let mut expired_entries = 0;
        let mut invalid_system = 0;
        let mut invalid_git = 0;

        for entry in self.memory_cache.values() {
            let age = SystemTime::now()
                .duration_since(entry.timestamp)
                .unwrap_or(Duration::from_secs(0));

            if age > self.ttl {
                expired_entries += 1;
            } else if entry.system_fingerprint != self.get_system_fingerprint() {
                invalid_system += 1;
            } else if self.validate_git {
                if let Some(current_commit) = self.get_git_commit() {
                    if let Some(ref cached_commit) = entry.git_commit {
                        if cached_commit != &current_commit {
                            invalid_git += 1;
                        } else {
                            valid_entries += 1;
                        }
                    } else {
                        invalid_git += 1;
                    }
                } else {
                    valid_entries += 1;
                }
            } else {
                valid_entries += 1;
            }
        }

        // Count disk cache entries
        if let Ok(entries) = std::fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                if entry.path().extension().and_then(|s| s.to_str()) == Some("json") {
                    total_entries += 1;
                }
            }
        }

        CacheStats {
            total_entries,
            valid_entries,
            expired_entries,
            invalid_system,
            invalid_git,
        }
    }

    /// Prune invalid cache entries
    pub fn prune(&mut self) -> std::io::Result<usize> {
        let mut pruned = 0;

        // Collect keys to remove to avoid borrow checker issues
        let keys_to_remove: Vec<String> = self
            .memory_cache
            .iter()
            .filter_map(|(key, entry)| {
                if !self.is_valid(entry) {
                    Some(key.clone())
                } else {
                    None
                }
            })
            .collect();

        // Remove invalid entries
        for key in keys_to_remove {
            self.memory_cache.remove(&key);
            pruned += 1;
        }

        // Prune disk cache
        if self.cache_dir.exists() {
            for entry in std::fs::read_dir(&self.cache_dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.extension().and_then(|s| s.to_str()) == Some("json") {
                    if let Some(cache_entry) = self.load_from_disk(
                        path.file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or_default(),
                    ) {
                        if !self.is_valid(&cache_entry) {
                            std::fs::remove_file(path)?;
                            pruned += 1;
                        }
                    }
                }
            }
        }

        Ok(pruned)
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total number of cache entries
    pub total_entries: usize,
    /// Number of valid entries
    pub valid_entries: usize,
    /// Number of expired entries
    pub expired_entries: usize,
    /// Number of entries with invalid system fingerprint
    pub invalid_system: usize,
    /// Number of entries with invalid git commit
    pub invalid_git: usize,
}

impl CacheStats {
    /// Calculate hit rate (valid / total)
    pub fn hit_rate(&self) -> f64 {
        if self.total_entries == 0 {
            0.0
        } else {
            self.valid_entries as f64 / self.total_entries as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_cache_creation() {
        let temp_dir = env::temp_dir().join("torsh_bench_cache_test");
        let cache = BenchmarkCache::new(&temp_dir);

        assert_eq!(cache.cache_dir, temp_dir);
        assert_eq!(cache.ttl, Duration::from_secs(7 * 24 * 60 * 60));
    }

    #[test]
    fn test_cache_key() {
        let temp_dir = env::temp_dir().join("torsh_bench_cache_test");
        let cache = BenchmarkCache::new(&temp_dir);

        let key = cache.cache_key("matmul", 1024);
        assert_eq!(key, "matmul_1024");
    }

    #[test]
    fn test_system_fingerprint() {
        let temp_dir = env::temp_dir().join("torsh_bench_cache_test");
        let cache = BenchmarkCache::new(&temp_dir);

        let fingerprint = cache.get_system_fingerprint();
        assert!(!fingerprint.is_empty());
        assert!(fingerprint.contains(env::consts::OS));
        assert!(fingerprint.contains(env::consts::ARCH));
    }

    #[test]
    fn test_cache_put_get() {
        let temp_dir = env::temp_dir().join("torsh_bench_cache_test_put_get");
        let mut cache = BenchmarkCache::new(&temp_dir).without_git_validation();

        let result = BenchResult {
            name: "test_bench".to_string(),
            size: 100,
            dtype: torsh_core::dtype::DType::F32,
            mean_time_ns: 1000.0,
            std_dev_ns: 50.0,
            throughput: Some(1e9),
            memory_usage: None,
            peak_memory: None,
            metrics: std::collections::HashMap::new(),
        };

        assert!(cache.put("test_bench", 100, result.clone(), 10, 2).is_ok());

        let cached = cache.get("test_bench", 100);
        assert!(cached.is_some());

        let cached = cached.unwrap();
        assert_eq!(cached.name, "test_bench");
        assert_eq!(cached.mean_time_ns, 1000.0);

        // Cleanup
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_cache_stats() {
        let temp_dir = env::temp_dir().join("torsh_bench_cache_test_stats");
        let mut cache = BenchmarkCache::new(&temp_dir).without_git_validation();

        let result = BenchResult {
            name: "test".to_string(),
            size: 100,
            dtype: torsh_core::dtype::DType::F32,
            mean_time_ns: 1000.0,
            std_dev_ns: 50.0,
            throughput: None,
            memory_usage: None,
            peak_memory: None,
            metrics: std::collections::HashMap::new(),
        };

        assert!(cache.put("test1", 100, result.clone(), 10, 2).is_ok());
        assert!(cache.put("test2", 200, result.clone(), 10, 2).is_ok());

        let stats = cache.stats();
        assert!(stats.total_entries >= 2);
        assert!(stats.valid_entries >= 2);

        // Cleanup
        let _ = std::fs::remove_dir_all(temp_dir);
    }

    #[test]
    fn test_cache_clear() {
        let temp_dir = env::temp_dir().join("torsh_bench_cache_test_clear");
        let mut cache = BenchmarkCache::new(&temp_dir).without_git_validation();

        let result = BenchResult {
            name: "test".to_string(),
            size: 100,
            dtype: torsh_core::dtype::DType::F32,
            mean_time_ns: 1000.0,
            std_dev_ns: 50.0,
            throughput: None,
            memory_usage: None,
            peak_memory: None,
            metrics: std::collections::HashMap::new(),
        };

        assert!(cache.put("test", 100, result, 10, 2).is_ok());
        assert!(cache.get("test", 100).is_some());

        assert!(cache.clear().is_ok());
        assert!(cache.get("test", 100).is_none());

        // Cleanup
        let _ = std::fs::remove_dir_all(temp_dir);
    }
}
