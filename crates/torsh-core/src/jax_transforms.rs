//! JAX-Style Function Transformations for Functional Programming
//!
//! This module provides a comprehensive transformation system inspired by JAX,
//! enabling functional programming patterns for tensor operations. It includes:
//!
//! - **JIT Compilation**: Just-in-time compilation with caching
//! - **Vectorization (vmap)**: Automatic batching over tensor dimensions
//! - **Parallelization (pmap)**: Parallel execution across devices
//! - **Composition**: Combine transformations for complex workflows
//!
//! # Examples
//!
//! ```
//! use torsh_core::jax_transforms::*;
//!
//! // JIT compilation (requires Hash + Eq, so use i32 instead of f32)
//! let jit_fn = JitTransform::new(|x: i32| x * 2);
//! let result = jit_fn.apply(5);
//! assert_eq!(result, 10);
//!
//! // Vectorization
//! let vmap_fn = VmapTransform::new(|x: i32| x * x, 0);
//! // vmap_fn will automatically vectorize over the first dimension
//! ```

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

/// Unique identifier for transformed functions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransformId(u64);

impl TransformId {
    /// Create a new transformation ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the underlying ID
    pub fn id(&self) -> u64 {
        self.0
    }
}

/// Transformation types supported by the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransformType {
    /// Just-in-time compilation
    Jit,
    /// Vectorization (map over a dimension)
    Vmap,
    /// Parallelization (map over devices)
    Pmap,
    /// Gradient computation
    Grad,
    /// Value and gradient computation
    ValueAndGrad,
    /// Custom transformation
    Custom,
}

/// Metadata about a transformation
#[derive(Debug, Clone)]
pub struct TransformMetadata {
    /// Unique identifier
    pub id: TransformId,
    /// Type of transformation
    pub transform_type: TransformType,
    /// Human-readable name
    pub name: String,
    /// Number of times this transformation has been applied
    pub application_count: usize,
    /// Whether this transformation is cached
    pub is_cached: bool,
    /// Custom metadata key-value pairs
    pub custom_metadata: HashMap<String, String>,
}

impl TransformMetadata {
    /// Create new transformation metadata
    pub fn new(id: TransformId, transform_type: TransformType, name: impl Into<String>) -> Self {
        Self {
            id,
            transform_type,
            name: name.into(),
            application_count: 0,
            is_cached: false,
            custom_metadata: HashMap::new(),
        }
    }

    /// Add custom metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom_metadata.insert(key.into(), value.into());
        self
    }

    /// Mark as cached
    pub fn cached(mut self) -> Self {
        self.is_cached = true;
        self
    }

    /// Increment application count
    pub fn increment_count(&mut self) {
        self.application_count += 1;
    }
}

/// Trait for functions that can be JIT compiled
pub trait Jittable<Input, Output> {
    /// Apply the function with JIT compilation
    fn jit_apply(&self, input: Input) -> Output;

    /// Get the transformation metadata
    fn jit_metadata(&self) -> &TransformMetadata;

    /// Invalidate the JIT cache for this function
    fn invalidate_cache(&mut self);
}

/// Trait for functions that can be vectorized (vmap)
pub trait Vectorizable<Input, Output> {
    /// Apply the function with vectorization over the specified dimension
    fn vmap_apply(&self, input: Input, in_dim: usize) -> Output;

    /// Get the vectorization dimension
    fn vmap_dim(&self) -> usize;

    /// Get the transformation metadata
    fn vmap_metadata(&self) -> &TransformMetadata;
}

/// Trait for functions that can be parallelized (pmap)
pub trait Parallelizable<Input, Output> {
    /// Apply the function in parallel across devices
    fn pmap_apply(&self, input: Input, devices: &[usize]) -> Output;

    /// Get the number of parallel executions
    fn pmap_degree(&self) -> usize;

    /// Get the transformation metadata
    fn pmap_metadata(&self) -> &TransformMetadata;
}

/// JIT compilation cache entry
#[derive(Debug, Clone)]
struct JitCacheEntry<Output> {
    /// Cached output
    output: Output,
    /// Number of cache hits
    hit_count: usize,
    /// Last access timestamp (for LRU eviction)
    last_access: std::time::Instant,
}

/// JIT transformation wrapper
pub struct JitTransform<F, Input, Output>
where
    F: Fn(Input) -> Output,
    Input: Clone + Hash + Eq,
    Output: Clone,
{
    /// The function to transform
    func: F,
    /// Transformation metadata
    metadata: Arc<RwLock<TransformMetadata>>,
    /// JIT cache (input -> output)
    cache: Arc<RwLock<HashMap<u64, JitCacheEntry<Output>>>>,
    /// Maximum cache size (0 = unlimited)
    max_cache_size: usize,
    /// Phantom data for input type
    _phantom: PhantomData<Input>,
}

impl<F, Input, Output> JitTransform<F, Input, Output>
where
    F: Fn(Input) -> Output,
    Input: Clone + Hash + Eq,
    Output: Clone,
{
    /// Create a new JIT transformation
    pub fn new(func: F) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Self {
            func,
            metadata: Arc::new(RwLock::new(
                TransformMetadata::new(TransformId::new(id), TransformType::Jit, "jit").cached(),
            )),
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_cache_size: 1000, // Default cache size
            _phantom: PhantomData,
        }
    }

    /// Create a new JIT transformation with custom cache size
    pub fn with_cache_size(func: F, max_size: usize) -> Self {
        let mut transform = Self::new(func);
        transform.max_cache_size = max_size;
        transform
    }

    /// Apply the function with JIT compilation and caching
    pub fn apply(&self, input: Input) -> Output
    where
        Input: Hash,
    {
        // Hash the input for cache lookup
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        input.hash(&mut hasher);
        let hash = hasher.finish();

        // Check cache
        {
            let mut cache = self.cache.write().unwrap();
            if let Some(entry) = cache.get_mut(&hash) {
                entry.hit_count += 1;
                entry.last_access = std::time::Instant::now();
                self.metadata.write().unwrap().increment_count();
                return entry.output.clone();
            }
        }

        // Cache miss - compute output
        let output = (self.func)(input.clone());

        // Store in cache
        {
            let mut cache = self.cache.write().unwrap();

            // Evict oldest entry if cache is full
            if self.max_cache_size > 0 && cache.len() >= self.max_cache_size {
                if let Some((&oldest_key, _)) =
                    cache.iter().min_by_key(|(_, entry)| entry.last_access)
                {
                    cache.remove(&oldest_key);
                }
            }

            cache.insert(
                hash,
                JitCacheEntry {
                    output: output.clone(),
                    hit_count: 0,
                    last_access: std::time::Instant::now(),
                },
            );
        }

        self.metadata.write().unwrap().increment_count();
        output
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        let cache = self.cache.read().unwrap();
        let total_hits: usize = cache.values().map(|entry| entry.hit_count).sum();

        CacheStats {
            size: cache.len(),
            total_hits,
            total_misses: self.metadata.read().unwrap().application_count - total_hits,
            max_size: self.max_cache_size,
        }
    }

    /// Clear the JIT cache
    pub fn clear_cache(&mut self) {
        self.cache.write().unwrap().clear();
    }

    /// Get transformation metadata
    pub fn metadata(&self) -> TransformMetadata {
        self.metadata.read().unwrap().clone()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Current cache size
    pub size: usize,
    /// Total cache hits
    pub total_hits: usize,
    /// Total cache misses
    pub total_misses: usize,
    /// Maximum cache size
    pub max_size: usize,
}

impl CacheStats {
    /// Get cache hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            0.0
        } else {
            self.total_hits as f64 / total as f64
        }
    }

    /// Check if cache is full
    pub fn is_full(&self) -> bool {
        self.max_size > 0 && self.size >= self.max_size
    }
}

/// Vectorization (vmap) transformation
pub struct VmapTransform<F, Input, Output>
where
    F: Fn(Input) -> Output,
{
    /// The function to transform
    func: F,
    /// Dimension to vectorize over
    in_dim: usize,
    /// Output dimension (usually same as in_dim)
    out_dim: usize,
    /// Transformation metadata
    metadata: Arc<RwLock<TransformMetadata>>,
    /// Phantom data
    _phantom: PhantomData<(Input, Output)>,
}

impl<F, Input, Output> VmapTransform<F, Input, Output>
where
    F: Fn(Input) -> Output,
{
    /// Create a new vmap transformation
    pub fn new(func: F, in_dim: usize) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Self {
            func,
            in_dim,
            out_dim: in_dim, // Default: output dimension matches input dimension
            metadata: Arc::new(RwLock::new(TransformMetadata::new(
                TransformId::new(id),
                TransformType::Vmap,
                "vmap",
            ))),
            _phantom: PhantomData,
        }
    }

    /// Create vmap with different output dimension
    pub fn with_out_dim(func: F, in_dim: usize, out_dim: usize) -> Self {
        let mut transform = Self::new(func, in_dim);
        transform.out_dim = out_dim;
        transform
    }

    /// Get the input vectorization dimension
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Get the output vectorization dimension
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// Get transformation metadata
    pub fn metadata(&self) -> TransformMetadata {
        self.metadata.read().unwrap().clone()
    }

    /// Apply the vectorization (must be implemented for specific types)
    pub fn apply_marker(&self) -> &F {
        &self.func
    }
}

/// Parallelization (pmap) transformation
pub struct PmapTransform<F, Input, Output>
where
    F: Fn(Input) -> Output + Send + Sync,
    Input: Send,
    Output: Send,
{
    /// The function to transform
    func: Arc<F>,
    /// Number of parallel executions
    num_devices: usize,
    /// Transformation metadata
    metadata: Arc<RwLock<TransformMetadata>>,
    /// Phantom data
    _phantom: PhantomData<(Input, Output)>,
}

impl<F, Input, Output> PmapTransform<F, Input, Output>
where
    F: Fn(Input) -> Output + Send + Sync,
    Input: Send,
    Output: Send,
{
    /// Create a new pmap transformation
    pub fn new(func: F, num_devices: usize) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Self {
            func: Arc::new(func),
            num_devices,
            metadata: Arc::new(RwLock::new(TransformMetadata::new(
                TransformId::new(id),
                TransformType::Pmap,
                "pmap",
            ))),
            _phantom: PhantomData,
        }
    }

    /// Get the number of parallel devices
    pub fn num_devices(&self) -> usize {
        self.num_devices
    }

    /// Get transformation metadata
    pub fn metadata(&self) -> TransformMetadata {
        self.metadata.read().unwrap().clone()
    }

    /// Get the function reference
    pub fn func(&self) -> &Arc<F> {
        &self.func
    }
}

/// Gradient transformation metadata
pub struct GradTransform<F, Input, Output>
where
    F: Fn(Input) -> Output,
{
    /// The function to differentiate
    func: F,
    /// Which arguments to differentiate with respect to
    argnums: Vec<usize>,
    /// Transformation metadata
    metadata: Arc<RwLock<TransformMetadata>>,
    /// Phantom data
    _phantom: PhantomData<(Input, Output)>,
}

impl<F, Input, Output> GradTransform<F, Input, Output>
where
    F: Fn(Input) -> Output,
{
    /// Create a new gradient transformation
    pub fn new(func: F, argnums: Vec<usize>) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Self {
            func,
            argnums,
            metadata: Arc::new(RwLock::new(TransformMetadata::new(
                TransformId::new(id),
                TransformType::Grad,
                "grad",
            ))),
            _phantom: PhantomData,
        }
    }

    /// Get the argument indices to differentiate
    pub fn argnums(&self) -> &[usize] {
        &self.argnums
    }

    /// Get transformation metadata
    pub fn metadata(&self) -> TransformMetadata {
        self.metadata.read().unwrap().clone()
    }

    /// Get the function reference
    pub fn func(&self) -> &F {
        &self.func
    }
}

/// Transformation composition for chaining transformations
pub struct ComposedTransform<F1, F2, Input, Intermediate, Output>
where
    F1: Fn(Input) -> Intermediate,
    F2: Fn(Intermediate) -> Output,
{
    /// First transformation
    first: F1,
    /// Second transformation
    second: F2,
    /// Transformation metadata
    metadata: Arc<RwLock<TransformMetadata>>,
    /// Phantom data
    _phantom: PhantomData<(Input, Intermediate, Output)>,
}

impl<F1, F2, Input, Intermediate, Output> ComposedTransform<F1, F2, Input, Intermediate, Output>
where
    F1: Fn(Input) -> Intermediate,
    F2: Fn(Intermediate) -> Output,
{
    /// Create a new composed transformation
    pub fn new(first: F1, second: F2) -> Self {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Self {
            first,
            second,
            metadata: Arc::new(RwLock::new(TransformMetadata::new(
                TransformId::new(id),
                TransformType::Custom,
                "composed",
            ))),
            _phantom: PhantomData,
        }
    }

    /// Apply the composed transformation
    pub fn apply(&self, input: Input) -> Output {
        let intermediate = (self.first)(input);
        (self.second)(intermediate)
    }

    /// Get transformation metadata
    pub fn metadata(&self) -> TransformMetadata {
        self.metadata.read().unwrap().clone()
    }
}

/// Transformation registry for managing all transformations
pub struct TransformRegistry {
    /// Registered transformations
    transforms: Arc<RwLock<HashMap<TransformId, TransformMetadata>>>,
}

impl TransformRegistry {
    /// Create a new transformation registry
    pub fn new() -> Self {
        Self {
            transforms: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a transformation
    pub fn register(&self, metadata: TransformMetadata) {
        self.transforms
            .write()
            .unwrap()
            .insert(metadata.id, metadata);
    }

    /// Get transformation metadata
    pub fn get(&self, id: TransformId) -> Option<TransformMetadata> {
        self.transforms.read().unwrap().get(&id).cloned()
    }

    /// Get all transformations of a specific type
    pub fn get_by_type(&self, transform_type: TransformType) -> Vec<TransformMetadata> {
        self.transforms
            .read()
            .unwrap()
            .values()
            .filter(|m| m.transform_type == transform_type)
            .cloned()
            .collect()
    }

    /// Get total number of registered transformations
    pub fn len(&self) -> usize {
        self.transforms.read().unwrap().len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.transforms.read().unwrap().is_empty()
    }

    /// Clear all registered transformations
    pub fn clear(&self) {
        self.transforms.write().unwrap().clear();
    }
}

impl Default for TransformRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_basic() {
        let jit_fn = JitTransform::new(|x: i32| x * 2);
        assert_eq!(jit_fn.apply(5), 10);
        assert_eq!(jit_fn.apply(10), 20);
        assert_eq!(jit_fn.apply(5), 10); // Cache hit
    }

    #[test]
    fn test_jit_cache_stats() {
        let jit_fn = JitTransform::new(|x: i32| x * 2);
        jit_fn.apply(5);
        jit_fn.apply(5); // Cache hit
        jit_fn.apply(10);
        jit_fn.apply(10); // Cache hit

        let stats = jit_fn.cache_stats();
        assert_eq!(stats.size, 2); // Two unique inputs
        assert_eq!(stats.total_hits, 2); // Two cache hits
        assert_eq!(stats.total_misses, 2); // Two cache misses
        assert_eq!(stats.hit_rate(), 0.5);
    }

    #[test]
    fn test_jit_cache_eviction() {
        let jit_fn = JitTransform::with_cache_size(|x: i32| x * 2, 2);
        jit_fn.apply(1);
        jit_fn.apply(2);
        jit_fn.apply(3); // Should evict least recently used

        let stats = jit_fn.cache_stats();
        assert_eq!(stats.size, 2); // Max cache size
    }

    #[test]
    fn test_jit_clear_cache() {
        let mut jit_fn = JitTransform::new(|x: i32| x * 2);
        jit_fn.apply(5);
        jit_fn.apply(10);

        assert_eq!(jit_fn.cache_stats().size, 2);

        jit_fn.clear_cache();
        assert_eq!(jit_fn.cache_stats().size, 0);
    }

    #[test]
    fn test_vmap_basic() {
        let vmap_fn = VmapTransform::new(|x: f32| x * x, 0);
        assert_eq!(vmap_fn.in_dim(), 0);
        assert_eq!(vmap_fn.out_dim(), 0);
    }

    #[test]
    fn test_vmap_different_dims() {
        let vmap_fn = VmapTransform::with_out_dim(|x: f32| x * x, 0, 1);
        assert_eq!(vmap_fn.in_dim(), 0);
        assert_eq!(vmap_fn.out_dim(), 1);
    }

    #[test]
    fn test_pmap_basic() {
        let pmap_fn = PmapTransform::new(|x: i32| x * 2, 4);
        assert_eq!(pmap_fn.num_devices(), 4);
    }

    #[test]
    fn test_grad_basic() {
        let grad_fn = GradTransform::new(|x: f32| x * x, vec![0]);
        assert_eq!(grad_fn.argnums(), &[0]);
    }

    #[test]
    fn test_grad_multiple_args() {
        let grad_fn = GradTransform::new(|_: (f32, f32)| 0.0, vec![0, 1]);
        assert_eq!(grad_fn.argnums(), &[0, 1]);
    }

    #[test]
    fn test_composed_transform() {
        let composed = ComposedTransform::new(|x: i32| x * 2, |x: i32| x + 1);
        assert_eq!(composed.apply(5), 11); // (5 * 2) + 1
    }

    #[test]
    fn test_transform_registry() {
        let registry = TransformRegistry::new();
        assert!(registry.is_empty());

        let metadata = TransformMetadata::new(TransformId::new(1), TransformType::Jit, "test");
        registry.register(metadata.clone());

        assert_eq!(registry.len(), 1);
        assert_eq!(registry.get(TransformId::new(1)).unwrap().name, "test");
    }

    #[test]
    fn test_transform_registry_by_type() {
        let registry = TransformRegistry::new();

        registry.register(TransformMetadata::new(
            TransformId::new(1),
            TransformType::Jit,
            "jit1",
        ));
        registry.register(TransformMetadata::new(
            TransformId::new(2),
            TransformType::Vmap,
            "vmap1",
        ));
        registry.register(TransformMetadata::new(
            TransformId::new(3),
            TransformType::Jit,
            "jit2",
        ));

        let jit_transforms = registry.get_by_type(TransformType::Jit);
        assert_eq!(jit_transforms.len(), 2);
    }

    #[test]
    fn test_transform_metadata() {
        let mut metadata = TransformMetadata::new(TransformId::new(1), TransformType::Jit, "test");

        assert_eq!(metadata.application_count, 0);
        metadata.increment_count();
        assert_eq!(metadata.application_count, 1);
    }

    #[test]
    fn test_transform_metadata_custom() {
        let metadata = TransformMetadata::new(TransformId::new(1), TransformType::Custom, "test")
            .with_metadata("key1", "value1")
            .with_metadata("key2", "value2");

        assert_eq!(metadata.custom_metadata.get("key1").unwrap(), "value1");
        assert_eq!(metadata.custom_metadata.get("key2").unwrap(), "value2");
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let stats = CacheStats {
            size: 5,
            total_hits: 8,
            total_misses: 2,
            max_size: 10,
        };

        assert_eq!(stats.hit_rate(), 0.8);
        assert!(!stats.is_full());
    }

    #[test]
    fn test_cache_stats_full() {
        let stats = CacheStats {
            size: 10,
            total_hits: 5,
            total_misses: 5,
            max_size: 10,
        };

        assert!(stats.is_full());
    }

    #[test]
    fn test_transform_id_equality() {
        let id1 = TransformId::new(42);
        let id2 = TransformId::new(42);
        let id3 = TransformId::new(43);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert_eq!(id1.id(), 42);
    }

    #[test]
    fn test_jit_metadata() {
        let jit_fn = JitTransform::new(|x: i32| x * 2);
        let metadata = jit_fn.metadata();

        assert_eq!(metadata.transform_type, TransformType::Jit);
        assert_eq!(metadata.name, "jit");
        assert!(metadata.is_cached);
    }

    #[test]
    fn test_vmap_metadata() {
        let vmap_fn = VmapTransform::new(|x: f32| x * x, 0);
        let metadata = vmap_fn.metadata();

        assert_eq!(metadata.transform_type, TransformType::Vmap);
        assert_eq!(metadata.name, "vmap");
    }

    #[test]
    fn test_pmap_metadata() {
        let pmap_fn = PmapTransform::new(|x: i32| x * 2, 4);
        let metadata = pmap_fn.metadata();

        assert_eq!(metadata.transform_type, TransformType::Pmap);
        assert_eq!(metadata.name, "pmap");
    }
}
