//! Smart caching for expensive special function computations

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Cache entry with timestamp for TTL management
#[derive(Clone)]
struct CacheEntry<T> {
    value: T,
    timestamp: Instant,
    access_count: usize,
}

/// Smart cache for special function computations
pub struct SmartCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    cache: Arc<Mutex<HashMap<K, CacheEntry<V>>>>,
    max_size: usize,
    ttl: Duration,
    hit_count: Arc<Mutex<usize>>,
    miss_count: Arc<Mutex<usize>>,
}

impl<K, V> SmartCache<K, V>
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    /// Create a new smart cache with specified capacity and TTL
    pub fn new(max_size: usize, ttl: Duration) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            max_size,
            ttl,
            hit_count: Arc::new(Mutex::new(0)),
            miss_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Get value from cache or compute it using the provided function
    pub fn get_or_compute<F>(&self, key: K, compute_fn: F) -> V
    where
        F: FnOnce() -> V,
    {
        {
            let mut cache = self.cache.lock().unwrap();

            // Check if key exists and is not expired
            if let Some(entry) = cache.get_mut(&key) {
                if entry.timestamp.elapsed() < self.ttl {
                    entry.access_count += 1;
                    *self.hit_count.lock().unwrap() += 1;
                    return entry.value.clone();
                } else {
                    // Remove expired entry
                    cache.remove(&key);
                }
            }
        }

        // Cache miss - compute the value
        let value = compute_fn();
        *self.miss_count.lock().unwrap() += 1;

        // Store in cache
        self.insert(key, value.clone());
        value
    }

    /// Insert a value into the cache
    pub fn insert(&self, key: K, value: V) {
        let mut cache = self.cache.lock().unwrap();

        // If cache is full, remove least recently used entries
        if cache.len() >= self.max_size {
            self.evict_lru(&mut cache);
        }

        let entry = CacheEntry {
            value,
            timestamp: Instant::now(),
            access_count: 1,
        };

        cache.insert(key, entry);
    }

    /// Evict least recently used entries
    fn evict_lru(&self, cache: &mut HashMap<K, CacheEntry<V>>) {
        // Find the entry with lowest access count and oldest timestamp
        let mut lru_key = None;
        let mut min_access_count = usize::MAX;
        let mut oldest_time = Instant::now();

        for (key, entry) in cache.iter() {
            if entry.access_count < min_access_count
                || (entry.access_count == min_access_count && entry.timestamp < oldest_time)
            {
                min_access_count = entry.access_count;
                oldest_time = entry.timestamp;
                lru_key = Some(key.clone());
            }
        }

        if let Some(key) = lru_key {
            cache.remove(&key);
        }
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f64 {
        let hits = *self.hit_count.lock().unwrap() as f64;
        let misses = *self.miss_count.lock().unwrap() as f64;
        let total = hits + misses;

        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.lock().unwrap().clear();
        *self.hit_count.lock().unwrap() = 0;
        *self.miss_count.lock().unwrap() = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.lock().unwrap();
        CacheStats {
            size: cache.len(),
            max_size: self.max_size,
            hit_rate: self.hit_rate(),
            hits: *self.hit_count.lock().unwrap(),
            misses: *self.miss_count.lock().unwrap(),
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub size: usize,
    pub max_size: usize,
    pub hit_rate: f64,
    pub hits: usize,
    pub misses: usize,
}

/// Cached special function key for floating point inputs
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct FloatKey {
    // Store as scaled integer for exact hashing
    value: i64,
    function_id: u32,
}

impl FloatKey {
    /// Create a new FloatKey from f64, using scaled integer representation
    pub fn new(value: f64, function_id: u32) -> Self {
        // Scale by 1e6 and round to get deterministic integer representation
        let scaled = (value * 1_000_000.0).round() as i64;
        Self {
            value: scaled,
            function_id,
        }
    }

    /// Get the original float value (approximately)
    pub fn to_f64(&self) -> f64 {
        self.value as f64 / 1_000_000.0
    }
}

// Global cache instance for special functions
lazy_static::lazy_static! {
    static ref GLOBAL_CACHE: SmartCache<FloatKey, f64> =
        SmartCache::new(10000, Duration::from_secs(300)); // 5 minutes TTL
}

/// Function IDs for caching
pub mod function_ids {
    pub const GAMMA: u32 = 1;
    pub const LGAMMA: u32 = 2;
    pub const BESSEL_J0: u32 = 3;
    pub const BESSEL_J1: u32 = 4;
    pub const BESSEL_Y0: u32 = 5;
    pub const BESSEL_Y1: u32 = 6;
    pub const BESSEL_K0: u32 = 7;
    pub const BESSEL_K1: u32 = 8;
    pub const ERF: u32 = 9;
    pub const ERFC: u32 = 10;
}

/// Cached computation wrapper for expensive functions
pub fn cached_compute<F>(value: f64, function_id: u32, compute_fn: F) -> f64
where
    F: FnOnce() -> f64,
{
    let key = FloatKey::new(value, function_id);
    GLOBAL_CACHE.get_or_compute(key, compute_fn)
}

/// Get global cache statistics
pub fn cache_stats() -> CacheStats {
    GLOBAL_CACHE.stats()
}

/// Clear global cache
pub fn clear_cache() {
    GLOBAL_CACHE.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TorshResult;
    use std::thread;

    #[test]
    fn test_cache_basic_functionality() -> TorshResult<()> {
        let cache: SmartCache<i32, String> = SmartCache::new(2, Duration::from_secs(1));

        let value1 = cache.get_or_compute(1, || "value1".to_string());
        assert_eq!(value1, "value1");

        let value2 = cache.get_or_compute(1, || "should_not_compute".to_string());
        assert_eq!(value2, "value1"); // Should use cached value

        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);
        Ok(())
    }

    #[test]
    fn test_cache_expiration() -> TorshResult<()> {
        let cache: SmartCache<i32, String> = SmartCache::new(10, Duration::from_millis(50));

        cache.insert(1, "value1".to_string());

        // Should get cached value
        let value = cache.get_or_compute(1, || "new_value".to_string());
        assert_eq!(value, "value1");

        // Wait for expiration
        thread::sleep(Duration::from_millis(100));

        // Should compute new value
        let value = cache.get_or_compute(1, || "new_value".to_string());
        assert_eq!(value, "new_value");
        Ok(())
    }

    #[test]
    fn test_float_key() -> TorshResult<()> {
        let key1 = FloatKey::new(1.23456, function_ids::GAMMA);
        let key2 = FloatKey::new(1.23456, function_ids::GAMMA);
        let key3 = FloatKey::new(1.23457, function_ids::GAMMA);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);

        assert!((key1.to_f64() - 1.23456).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_cached_compute() -> TorshResult<()> {
        clear_cache();

        let mut call_count = 0;
        let expensive_fn = || {
            call_count += 1;
            1.23456
        };

        // First call should compute
        let result1 = cached_compute(1.0, function_ids::GAMMA, expensive_fn);
        assert_eq!(result1, 1.23456);

        // Second call should use cache (call_count won't increment)
        let result2 = cached_compute(1.0, function_ids::GAMMA, || 9.99999);
        assert_eq!(result2, 1.23456);
        Ok(())
    }
}
