# Best Practices for Data Loading in ToRSh

This guide provides comprehensive best practices for efficient and effective data loading in the ToRSh deep learning framework.

## Table of Contents

1. [General Principles](#general-principles)
2. [Dataset Design](#dataset-design)
3. [DataLoader Configuration](#dataloader-configuration)
4. [Transform Optimization](#transform-optimization)
5. [Memory Management](#memory-management)
6. [Performance Optimization](#performance-optimization)
7. [Error Handling and Debugging](#error-handling-and-debugging)
8. [Security Considerations](#security-considerations)
9. [Testing and Validation](#testing-and-validation)
10. [Production Deployment](#production-deployment)

## General Principles

### 1. Data Pipeline Design

**Design for Scalability**
```rust
// Good: Separates data loading, transformation, and batching concerns
let dataset = MyDataset::new(data_path)?;
let transform = MyTransformPipeline::new();
let dataloader = DataLoader::builder(dataset)
    .batch_size(32)
    .shuffle(true)
    .num_workers(4)
    .transform(transform)
    .build();

// Avoid: Tightly coupled data loading and processing
let dataloader = MonolithicDataLoader::new(data_path, batch_size, transform_config);
```

**Lazy Loading Principle**
```rust
// Good: Load data only when needed
#[derive(Clone)]
pub struct LazyImageDataset {
    image_paths: Vec<PathBuf>,
    transform: Option<ImageTransform>,
}

impl Dataset for LazyImageDataset {
    type Item = Tensor<f32>;
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        // Load image only when requested
        let image = load_image(&self.image_paths[index])?;
        let tensor = image_to_tensor(image)?;
        
        if let Some(ref transform) = self.transform {
            transform.transform(tensor)
        } else {
            Ok(tensor)
        }
    }
}

// Avoid: Loading all data upfront
pub struct EagerImageDataset {
    images: Vec<Tensor<f32>>, // Memory intensive!
}
```

### 2. Thread Safety

**Always Implement Send + Sync**
```rust
// Good: Thread-safe dataset
#[derive(Clone)]
pub struct ThreadSafeDataset {
    data: Arc<Vec<DataItem>>,
    metadata: Arc<Metadata>,
}

unsafe impl Send for ThreadSafeDataset {}
unsafe impl Sync for ThreadSafeDataset {}

// Good: Use Arc for shared immutable data
#[derive(Clone)]
pub struct SharedDataset {
    config: Arc<DatasetConfig>,
    cache: Arc<RwLock<HashMap<usize, CachedItem>>>,
}
```

**Avoid Shared Mutable State**
```rust
// Avoid: Shared mutable state without proper synchronization
pub struct BadDataset {
    counter: usize, // Not thread-safe!
}

// Better: Use atomic operations for shared counters
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct GoodDataset {
    access_counter: Arc<AtomicUsize>,
}

impl Dataset for GoodDataset {
    fn get(&self, index: usize) -> Result<Self::Item> {
        self.access_counter.fetch_add(1, Ordering::Relaxed);
        // ... rest of implementation
    }
}
```

## Dataset Design

### 1. Efficient Data Storage

**Use Memory Mapping for Large Files**
```rust
use memmap2::MmapOptions;

#[derive(Clone)]
pub struct MmapDataset {
    mmap: Arc<memmap2::Mmap>,
    item_size: usize,
    num_items: usize,
}

impl MmapDataset {
    pub fn new<P: AsRef<Path>>(path: P, item_size: usize) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        let num_items = mmap.len() / item_size;
        
        Ok(Self {
            mmap: Arc::new(mmap),
            item_size,
            num_items,
        })
    }
}

impl Dataset for MmapDataset {
    type Item = &[u8];
    
    fn len(&self) -> usize {
        self.num_items
    }
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        let start = index * self.item_size;
        let end = start + self.item_size;
        Ok(&self.mmap[start..end])
    }
}
```

**Implement Hierarchical Data Access**
```rust
// Good: Hierarchical dataset structure
pub struct HierarchicalDataset {
    metadata_index: HashMap<String, Vec<usize>>,
    data_chunks: Vec<DataChunk>,
}

impl HierarchicalDataset {
    pub fn get_by_category(&self, category: &str) -> Result<CategoryDataset> {
        let indices = self.metadata_index.get(category)
            .ok_or_else(|| DataError::CategoryNotFound(category.to_string()))?;
        
        Ok(CategoryDataset {
            parent: self,
            indices: indices.clone(),
        })
    }
}
```

### 2. Robust Data Validation

**Validate Data at Creation Time**
```rust
impl TensorDataset<f32> {
    pub fn new(tensors: Vec<Tensor<f32>>) -> Result<Self> {
        // Validate all tensors have compatible shapes
        if tensors.is_empty() {
            return Err(DataError::EmptyDataset);
        }
        
        let first_shape = tensors[0].shape();
        let expected_batch_dim = first_shape.dims()[0];
        
        for (i, tensor) in tensors.iter().enumerate() {
            let shape = tensor.shape();
            if shape.dims()[0] != expected_batch_dim {
                return Err(DataError::InvalidShape {
                    tensor_index: i,
                    expected: expected_batch_dim,
                    actual: shape.dims()[0],
                });
            }
        }
        
        Ok(Self { tensors })
    }
}
```

**Implement Data Integrity Checks**
```rust
pub trait DataIntegrity {
    fn verify_integrity(&self) -> Result<()>;
    fn repair_if_possible(&mut self) -> Result<bool>;
}

impl DataIntegrity for FileDataset {
    fn verify_integrity(&self) -> Result<()> {
        for (i, path) in self.file_paths.iter().enumerate() {
            if !path.exists() {
                return Err(DataError::MissingFile {
                    index: i,
                    path: path.clone(),
                });
            }
            
            // Check file size, format, etc.
            let metadata = std::fs::metadata(path)?;
            if metadata.len() == 0 {
                return Err(DataError::CorruptedFile {
                    index: i,
                    path: path.clone(),
                    reason: "Empty file".to_string(),
                });
            }
        }
        Ok(())
    }
    
    fn repair_if_possible(&mut self) -> Result<bool> {
        let mut repaired = false;
        self.file_paths.retain(|path| {
            if !path.exists() {
                eprintln!("Warning: Removing missing file: {:?}", path);
                repaired = true;
                false
            } else {
                true
            }
        });
        Ok(repaired)
    }
}
```

### 3. Flexible Dataset Composition

**Use Composition over Inheritance**
```rust
// Good: Compositional design
pub struct CompositeDataset<D1, D2> {
    dataset1: D1,
    dataset2: D2,
    combine_fn: fn(D1::Item, D2::Item) -> CombinedItem,
}

impl<D1: Dataset, D2: Dataset> Dataset for CompositeDataset<D1, D2> {
    type Item = CombinedItem;
    
    fn len(&self) -> usize {
        std::cmp::min(self.dataset1.len(), self.dataset2.len())
    }
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        let item1 = self.dataset1.get(index)?;
        let item2 = self.dataset2.get(index)?;
        Ok((self.combine_fn)(item1, item2))
    }
}

// Usage
let features = FeatureDataset::new(feature_data);
let labels = LabelDataset::new(label_data);
let combined = CompositeDataset::new(features, labels, |f, l| (f, l));
```

## DataLoader Configuration

### 1. Optimal Batch Size Selection

**Consider Memory and Computation Trade-offs**
```rust
pub struct BatchSizeOptimizer {
    available_memory: usize,
    item_size_estimate: usize,
    target_memory_usage: f32, // 0.8 = 80% of available memory
}

impl BatchSizeOptimizer {
    pub fn recommend_batch_size(&self, dataset_len: usize) -> usize {
        let memory_based = (self.available_memory as f32 * self.target_memory_usage) 
            / self.item_size_estimate as f32;
        
        let power_of_2 = (memory_based.log2().floor() as u32).min(10); // Cap at 1024
        let recommended = 2_usize.pow(power_of_2);
        
        // Ensure we don't exceed dataset size
        std::cmp::min(recommended, dataset_len)
    }
}

// Usage
let optimizer = BatchSizeOptimizer {
    available_memory: get_available_memory(),
    item_size_estimate: estimate_item_size(&dataset),
    target_memory_usage: 0.7,
};

let batch_size = optimizer.recommend_batch_size(dataset.len());
```

### 2. Worker Configuration

**Determine Optimal Worker Count**
```rust
pub fn optimal_worker_count() -> usize {
    let cpu_count = num_cpus::get();
    let available_memory = get_available_memory();
    
    // Rule of thumb: 1-2 workers per CPU core, limited by memory
    let cpu_based = (cpu_count as f32 * 1.5) as usize;
    let memory_based = available_memory / (512 * 1024 * 1024); // 512MB per worker
    
    std::cmp::max(1, std::cmp::min(cpu_based, memory_based))
}

// Configure DataLoader with optimal settings
let dataloader = DataLoader::builder(dataset)
    .batch_size(32)
    .num_workers(optimal_worker_count())
    .pin_memory(true) // If using GPU
    .drop_last(true)  // For consistent batch sizes
    .build();
```

### 3. Sampling Strategies

**Choose Appropriate Sampling for Your Use Case**
```rust
// For balanced training
let balanced_sampler = WeightedRandomSampler::new(
    compute_class_weights(&dataset)?,
    dataset.len(),
    true, // replacement
)?;

// For distributed training
let distributed_sampler = sampler.into_distributed(
    world_size,
    rank,
);

// For curriculum learning
let curriculum_sampler = CurriculumSampler::new(
    dataset.len(),
    difficulty_scores,
    initial_easy_ratio: 0.8,
);
```

## Transform Optimization

### 1. Transform Pipeline Design

**Order Transforms by Computational Cost**
```rust
// Good: Cheap operations first, expensive operations last
let transform_pipeline = TransformPipeline::new()
    .add(ValidateInput)        // Fast validation
    .add(Normalize::new(mean, std))  // Simple arithmetic
    .add(Resize::new(224, 224))      // Moderate cost
    .add(RandomAugmentation::new())  // Most expensive
    .build();

// Avoid: Expensive operations early in pipeline
let bad_pipeline = TransformPipeline::new()
    .add(ExpensiveAugmentation::new())  // Wasteful if validation fails
    .add(ValidateInput)
    .add(SimpleNormalize)
    .build();
```

**Use Conditional Transforms Wisely**
```rust
// Good: Skip expensive operations when not needed
let conditional_augmentation = ExpensiveAugmentation::new()
    .when(|item| item.requires_augmentation())
    .with_probability(0.5); // Only apply 50% of the time

// Cache expensive computations
let cached_transform = CacheTransform::new(
    ExpensiveTransform::new(),
    |item| item.compute_cache_key(), // Key function
);
```

### 2. Memory-Efficient Transforms

**Prefer In-Place Operations**
```rust
// Good: In-place normalization
pub struct InPlaceNormalize {
    mean: f32,
    std: f32,
}

impl Transform<&mut Tensor<f32>> for InPlaceNormalize {
    type Output = ();
    
    fn transform(&self, tensor: &mut Tensor<f32>) -> Result<()> {
        // Modify tensor in-place to avoid allocation
        tensor.sub_scalar_(self.mean)?;
        tensor.div_scalar_(self.std)?;
        Ok(())
    }
}

// Use copy-on-write semantics when appropriate
pub struct CowTransform<T> {
    inner: T,
}

impl<T: Transform<Tensor<f32>, Output = Tensor<f32>>> Transform<Tensor<f32>> for CowTransform<T> {
    type Output = Tensor<f32>;
    
    fn transform(&self, input: Tensor<f32>) -> Result<Self::Output> {
        // Try in-place operation first
        if input.ref_count() == 1 {
            // Safe to modify in-place
            let mut tensor = input;
            self.inner.transform_inplace(&mut tensor)?;
            Ok(tensor)
        } else {
            // Need to copy
            let copied = input.clone();
            self.inner.transform(copied)
        }
    }
}
```

## Memory Management

### 1. Memory Pool Usage

**Implement Memory Pools for Frequent Allocations**
```rust
use std::sync::Mutex;

pub struct TensorPool {
    pools: Vec<Mutex<Vec<Tensor<f32>>>>,
    max_pool_size: usize,
}

impl TensorPool {
    pub fn new(max_pool_size: usize) -> Self {
        let num_sizes = 10; // Support different tensor sizes
        let pools = (0..num_sizes)
            .map(|_| Mutex::new(Vec::new()))
            .collect();
        
        Self { pools, max_pool_size }
    }
    
    pub fn get_tensor(&self, shape: &[usize]) -> Tensor<f32> {
        let pool_idx = self.shape_to_pool_index(shape);
        
        if let Ok(mut pool) = self.pools[pool_idx].try_lock() {
            if let Some(tensor) = pool.pop() {
                // Reuse existing tensor
                tensor.resize_(shape).unwrap();
                return tensor;
            }
        }
        
        // Create new tensor
        Tensor::zeros(shape).unwrap()
    }
    
    pub fn return_tensor(&self, tensor: Tensor<f32>) {
        let shape = tensor.shape().dims();
        let pool_idx = self.shape_to_pool_index(shape);
        
        if let Ok(mut pool) = self.pools[pool_idx].try_lock() {
            if pool.len() < self.max_pool_size {
                pool.push(tensor);
            }
        }
        // Otherwise, let tensor drop naturally
    }
}

// Global tensor pool
lazy_static::lazy_static! {
    static ref TENSOR_POOL: TensorPool = TensorPool::new(100);
}
```

### 2. Memory Usage Monitoring

**Track Memory Usage**
```rust
pub struct MemoryTracker {
    peak_usage: AtomicUsize,
    current_usage: AtomicUsize,
    allocation_count: AtomicUsize,
}

impl MemoryTracker {
    pub fn track_allocation(&self, size: usize) {
        let new_usage = self.current_usage.fetch_add(size, Ordering::Relaxed) + size;
        
        // Update peak usage
        let mut peak = self.peak_usage.load(Ordering::Relaxed);
        while new_usage > peak {
            match self.peak_usage.compare_exchange_weak(
                peak,
                new_usage,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
        
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn track_deallocation(&self, size: usize) {
        self.current_usage.fetch_sub(size, Ordering::Relaxed);
    }
    
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            current_usage: self.current_usage.load(Ordering::Relaxed),
            peak_usage: self.peak_usage.load(Ordering::Relaxed),
            allocation_count: self.allocation_count.load(Ordering::Relaxed),
        }
    }
}
```

## Performance Optimization

### 1. CPU Optimization

**Use SIMD When Possible**
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn vectorized_normalize(data: &mut [f32], mean: f32, std: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                vectorized_normalize_avx2(data, mean, std);
                return;
            }
        }
    }
    
    // Fallback scalar implementation
    for value in data.iter_mut() {
        *value = (*value - mean) / std;
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn vectorized_normalize_avx2(data: &mut [f32], mean: f32, std: f32) {
    let mean_vec = _mm256_set1_ps(mean);
    let std_vec = _mm256_set1_ps(std);
    
    let chunks = data.chunks_exact_mut(8);
    let remainder = chunks.remainder();
    
    for chunk in chunks {
        let values = _mm256_loadu_ps(chunk.as_ptr());
        let normalized = _mm256_div_ps(
            _mm256_sub_ps(values, mean_vec),
            std_vec
        );
        _mm256_storeu_ps(chunk.as_mut_ptr(), normalized);
    }
    
    // Handle remainder with scalar code
    for value in remainder {
        *value = (*value - mean) / std;
    }
}
```

**Leverage Parallelism**
```rust
use rayon::prelude::*;

impl Transform<Vec<f32>> for ParallelNormalize {
    type Output = Vec<f32>;
    
    fn transform(&self, input: Vec<f32>) -> Result<Self::Output> {
        Ok(input
            .into_par_iter()
            .map(|x| (x - self.mean) / self.std)
            .collect())
    }
    
    fn transform_batch(&self, inputs: Vec<Vec<f32>>) -> Result<Vec<Self::Output>> {
        Ok(inputs
            .into_par_iter()
            .map(|input| self.transform(input).unwrap())
            .collect())
    }
}
```

### 2. I/O Optimization

**Use Asynchronous I/O**
```rust
use tokio::fs::File;
use tokio::io::AsyncReadExt;

#[derive(Clone)]
pub struct AsyncFileDataset {
    file_paths: Arc<Vec<PathBuf>>,
    chunk_size: usize,
}

impl AsyncFileDataset {
    pub async fn get_async(&self, index: usize) -> Result<Vec<u8>> {
        let path = &self.file_paths[index];
        let mut file = File::open(path).await?;
        
        let metadata = file.metadata().await?;
        let file_size = metadata.len() as usize;
        
        let mut buffer = Vec::with_capacity(file_size);
        file.read_to_end(&mut buffer).await?;
        
        Ok(buffer)
    }
}

// Prefetch data asynchronously
pub struct PrefetchingDataset<D> {
    inner: D,
    prefetch_buffer: Arc<Mutex<VecDeque<(usize, D::Item)>>>,
    buffer_size: usize,
}

impl<D: Dataset + Send + Sync + 'static> PrefetchingDataset<D>
where
    D::Item: Send + 'static,
{
    pub fn new(inner: D, buffer_size: usize) -> Self {
        let dataset = Self {
            inner,
            prefetch_buffer: Arc::new(Mutex::new(VecDeque::new())),
            buffer_size,
        };
        
        // Start prefetching thread
        dataset.start_prefetching();
        dataset
    }
    
    fn start_prefetching(&self) {
        let inner = Arc::new(self.inner.clone());
        let buffer = self.prefetch_buffer.clone();
        let buffer_size = self.buffer_size;
        
        tokio::spawn(async move {
            for i in 0..inner.len() {
                // Check if buffer is full
                {
                    let buf = buffer.lock().unwrap();
                    if buf.len() >= buffer_size {
                        // Wait for space
                        drop(buf);
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        continue;
                    }
                }
                
                // Load item asynchronously
                if let Ok(item) = inner.get(i) {
                    let mut buf = buffer.lock().unwrap();
                    buf.push_back((i, item));
                }
            }
        });
    }
}
```

### 3. GPU Optimization

**Efficient GPU Memory Transfer**
```rust
use torsh_core::device::DeviceType;

pub struct GPUOptimizedDataLoader<D> {
    cpu_dataloader: DataLoader<D>,
    device: DeviceType,
    pin_memory: bool,
    prefetch_factor: usize,
}

impl<D: Dataset> GPUOptimizedDataLoader<D>
where
    D::Item: Send + 'static,
{
    pub fn new(dataloader: DataLoader<D>, device: DeviceType) -> Self {
        Self {
            cpu_dataloader: dataloader,
            device,
            pin_memory: true,
            prefetch_factor: 2,
        }
    }
    
    pub fn iter_gpu(&self) -> GPUDataLoaderIterator<D> {
        GPUDataLoaderIterator::new(
            self.cpu_dataloader.iter(),
            self.device.clone(),
            self.prefetch_factor,
        )
    }
}

pub struct GPUDataLoaderIterator<D: Dataset> {
    cpu_iter: DataLoaderIterator<D>,
    device: DeviceType,
    transfer_queue: VecDeque<Tensor<f32>>,
    prefetch_factor: usize,
}

impl<D: Dataset> Iterator for GPUDataLoaderIterator<D> {
    type Item = Result<Tensor<f32>>;
    
    fn next(&mut self) -> Option<Self::Item> {
        // Start async GPU transfer for future batches
        while self.transfer_queue.len() < self.prefetch_factor {
            if let Some(cpu_batch) = self.cpu_iter.next() {
                match cpu_batch {
                    Ok(batch) => {
                        // Asynchronously transfer to GPU
                        let gpu_batch = batch.to_device(&self.device);
                        self.transfer_queue.push_back(gpu_batch);
                    }
                    Err(e) => return Some(Err(e)),
                }
            } else {
                break;
            }
        }
        
        self.transfer_queue.pop_front().map(Ok)
    }
}
```

## Error Handling and Debugging

### 1. Comprehensive Error Types

**Define Rich Error Types**
```rust
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("Dataset is empty")]
    EmptyDataset,
    
    #[error("Index {index} out of bounds (size: {size})")]
    IndexOutOfBounds { index: usize, size: usize },
    
    #[error("Invalid tensor shape at index {tensor_index}: expected {expected}, got {actual}")]
    InvalidShape {
        tensor_index: usize,
        expected: usize,
        actual: usize,
    },
    
    #[error("File not found: {path}")]
    FileNotFound { path: PathBuf },
    
    #[error("Corrupted data at index {index}: {reason}")]
    CorruptedData { index: usize, reason: String },
    
    #[error("Transform failed: {transform_name} at step {step}: {reason}")]
    TransformFailed {
        transform_name: String,
        step: usize,
        reason: String,
    },
    
    #[error("Memory allocation failed: requested {size} bytes")]
    MemoryAllocationFailed { size: usize },
    
    #[error("Timeout while loading data: {timeout_ms}ms")]
    Timeout { timeout_ms: u64 },
    
    #[error("Worker thread panicked: {worker_id}")]
    WorkerPanic { worker_id: usize },
}

impl DataError {
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            DataError::EmptyDataset => ErrorSeverity::Fatal,
            DataError::IndexOutOfBounds { .. } => ErrorSeverity::Error,
            DataError::CorruptedData { .. } => ErrorSeverity::Warning,
            DataError::Timeout { .. } => ErrorSeverity::Warning,
            _ => ErrorSeverity::Error,
        }
    }
    
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            DataError::Timeout { .. } | 
            DataError::CorruptedData { .. }
        )
    }
}
```

### 2. Debugging Utilities

**Add Debugging and Profiling Support**
```rust
pub struct DebugDataLoader<D> {
    inner: DataLoader<D>,
    debug_config: DebugConfig,
    stats: Arc<Mutex<LoadingStats>>,
}

#[derive(Debug, Clone)]
pub struct DebugConfig {
    pub log_slow_loads: bool,
    pub slow_threshold_ms: u64,
    pub sample_data: bool,
    pub track_memory: bool,
}

#[derive(Debug, Default)]
pub struct LoadingStats {
    pub total_loads: usize,
    pub slow_loads: usize,
    pub total_time_ms: u64,
    pub avg_time_ms: f64,
    pub memory_peak_mb: f64,
}

impl<D: Dataset> DebugDataLoader<D> {
    pub fn new(dataloader: DataLoader<D>, config: DebugConfig) -> Self {
        Self {
            inner: dataloader,
            debug_config: config,
            stats: Arc::new(Mutex::new(LoadingStats::default())),
        }
    }
    
    pub fn get_stats(&self) -> LoadingStats {
        self.stats.lock().unwrap().clone()
    }
}

impl<D: Dataset> Iterator for DebugDataLoader<D> {
    type Item = Result<D::Item>;
    
    fn next(&mut self) -> Option<Self::Item> {
        let start_time = std::time::Instant::now();
        let start_memory = if self.debug_config.track_memory {
            get_current_memory_usage()
        } else {
            0.0
        };
        
        let result = self.inner.next();
        
        let elapsed_ms = start_time.elapsed().as_millis() as u64;
        
        // Update statistics
        {
            let mut stats = self.stats.lock().unwrap();
            stats.total_loads += 1;
            stats.total_time_ms += elapsed_ms;
            stats.avg_time_ms = stats.total_time_ms as f64 / stats.total_loads as f64;
            
            if elapsed_ms > self.debug_config.slow_threshold_ms {
                stats.slow_loads += 1;
                if self.debug_config.log_slow_loads {
                    eprintln!("Slow data load: {}ms", elapsed_ms);
                }
            }
            
            if self.debug_config.track_memory {
                let current_memory = get_current_memory_usage();
                stats.memory_peak_mb = stats.memory_peak_mb.max(current_memory);
            }
        }
        
        result
    }
}
```

## Security Considerations

### 1. Input Validation

**Validate All External Data**
```rust
pub struct SecureDataset {
    inner: Box<dyn Dataset<Item = Vec<u8>>>,
    validator: DataValidator,
}

pub struct DataValidator {
    max_file_size: usize,
    allowed_extensions: HashSet<String>,
    virus_scanner: Option<Box<dyn VirusScanner>>,
}

impl DataValidator {
    pub fn validate_file(&self, path: &Path) -> Result<()> {
        // Check file extension
        if let Some(ext) = path.extension() {
            let ext_str = ext.to_string_lossy().to_lowercase();
            if !self.allowed_extensions.contains(&ext_str) {
                return Err(SecurityError::InvalidFileType(ext_str));
            }
        }
        
        // Check file size
        let metadata = std::fs::metadata(path)?;
        if metadata.len() as usize > self.max_file_size {
            return Err(SecurityError::FileTooLarge {
                size: metadata.len() as usize,
                max_allowed: self.max_file_size,
            });
        }
        
        // Virus scan if available
        if let Some(ref scanner) = self.virus_scanner {
            scanner.scan_file(path)?;
        }
        
        Ok(())
    }
}

impl Dataset for SecureDataset {
    type Item = Vec<u8>;
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        let data = self.inner.get(index)?;
        
        // Additional runtime validation
        if data.len() > self.validator.max_file_size {
            return Err(SecurityError::DataTooLarge {
                size: data.len(),
                max_allowed: self.validator.max_file_size,
            }.into());
        }
        
        Ok(data)
    }
}
```

### 2. Sandboxing and Resource Limits

**Implement Resource Limits**
```rust
use std::time::{Duration, Instant};

pub struct ResourceLimitedDataLoader<D> {
    inner: DataLoader<D>,
    memory_limit: usize,
    time_limit: Duration,
    current_memory: Arc<AtomicUsize>,
}

impl<D: Dataset> ResourceLimitedDataLoader<D> {
    pub fn with_limits(
        dataloader: DataLoader<D>,
        memory_limit: usize,
        time_limit: Duration,
    ) -> Self {
        Self {
            inner: dataloader,
            memory_limit,
            time_limit,
            current_memory: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl<D: Dataset> Iterator for ResourceLimitedDataLoader<D> {
    type Item = Result<D::Item>;
    
    fn next(&mut self) -> Option<Self::Item> {
        let start_time = Instant::now();
        
        // Check memory limit
        let current_memory = self.current_memory.load(Ordering::Relaxed);
        if current_memory > self.memory_limit {
            return Some(Err(ResourceError::MemoryLimitExceeded {
                current: current_memory,
                limit: self.memory_limit,
            }.into()));
        }
        
        // Load with timeout
        let result = match timeout(self.time_limit, self.inner.next()) {
            Ok(Some(result)) => result,
            Ok(None) => return None,
            Err(_) => return Some(Err(ResourceError::TimeoutExceeded {
                timeout: self.time_limit,
            }.into())),
        };
        
        Some(result)
    }
}
```

## Testing and Validation

### 1. Comprehensive Test Coverage

**Test All Components**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    
    #[test]
    fn test_dataset_bounds_checking() {
        let dataset = create_test_dataset(100);
        
        // Valid access
        assert!(dataset.get(0).is_ok());
        assert!(dataset.get(99).is_ok());
        
        // Invalid access
        assert!(dataset.get(100).is_err());
        assert!(dataset.get(1000).is_err());
    }
    
    #[test]
    fn test_dataloader_deterministic_with_seed() {
        let dataset = create_test_dataset(100);
        
        let loader1 = DataLoader::builder(dataset.clone())
            .shuffle(true)
            .random_seed(42)
            .build();
            
        let loader2 = DataLoader::builder(dataset)
            .shuffle(true)
            .random_seed(42)
            .build();
        
        let batches1: Vec<_> = loader1.collect();
        let batches2: Vec<_> = loader2.collect();
        
        assert_eq!(batches1.len(), batches2.len());
        for (b1, b2) in batches1.iter().zip(batches2.iter()) {
            assert_tensors_equal(b1, b2);
        }
    }
    
    proptest! {
        #[test]
        fn test_transform_properties(
            data in prop::collection::vec(-1000.0f32..1000.0, 1..1000)
        ) {
            let transform = NormalizeTransform::new(0.0, 1.0);
            
            // Test that transform preserves vector length
            let original_len = data.len();
            let transformed = transform.transform(data.clone()).unwrap();
            prop_assert_eq!(transformed.len(), original_len);
            
            // Test that inverse transform works
            let denormalize = NormalizeTransform::new(0.0, 1.0).inverse();
            let roundtrip = denormalize.transform(transformed).unwrap();
            
            for (orig, rt) in data.iter().zip(roundtrip.iter()) {
                prop_assert!((orig - rt).abs() < 1e-5);
            }
        }
    }
}
```

### 2. Performance Testing

**Benchmark Critical Paths**
```rust
#[cfg(test)]
mod benchmarks {
    use super::*;
    use criterion::{criterion_group, criterion_main, Criterion};
    
    fn benchmark_dataloader_throughput(c: &mut Criterion) {
        let dataset = create_large_test_dataset(10000);
        let dataloader = DataLoader::builder(dataset)
            .batch_size(32)
            .num_workers(4)
            .build();
        
        c.bench_function("dataloader_throughput", |b| {
            b.iter(|| {
                let mut count = 0;
                for batch in dataloader.iter() {
                    count += batch.unwrap().len();
                    if count >= 1000 { break; } // Benchmark first 1000 items
                }
            });
        });
    }
    
    criterion_group!(benches, benchmark_dataloader_throughput);
    criterion_main!(benches);
}
```

## Production Deployment

### 1. Configuration Management

**Use Configuration Files**
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct DataLoaderConfig {
    pub batch_size: usize,
    pub num_workers: usize,
    pub shuffle: bool,
    pub drop_last: bool,
    pub pin_memory: bool,
    pub timeout_ms: Option<u64>,
    pub prefetch_factor: usize,
    pub memory_limit_mb: Option<usize>,
    pub cache_size_mb: Option<usize>,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            num_workers: num_cpus::get(),
            shuffle: true,
            drop_last: false,
            pin_memory: cfg!(feature = "cuda"),
            timeout_ms: Some(30000),
            prefetch_factor: 2,
            memory_limit_mb: None,
            cache_size_mb: Some(512),
        }
    }
}

pub fn create_dataloader_from_config<D: Dataset>(
    dataset: D,
    config: &DataLoaderConfig,
) -> Result<DataLoader<D>> {
    let mut builder = DataLoader::builder(dataset)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .shuffle(config.shuffle)
        .drop_last(config.drop_last)
        .pin_memory(config.pin_memory);
    
    if let Some(timeout) = config.timeout_ms {
        builder = builder.timeout(Duration::from_millis(timeout));
    }
    
    Ok(builder.build())
}
```

### 2. Monitoring and Metrics

**Implement Comprehensive Monitoring**
```rust
use prometheus::{Counter, Histogram, Gauge, register_counter, register_histogram, register_gauge};

pub struct DataLoaderMetrics {
    batches_loaded: Counter,
    load_duration: Histogram,
    memory_usage: Gauge,
    error_count: Counter,
}

impl DataLoaderMetrics {
    pub fn new() -> Result<Self> {
        Ok(Self {
            batches_loaded: register_counter!(
                "dataloader_batches_total",
                "Total number of batches loaded"
            )?,
            load_duration: register_histogram!(
                "dataloader_load_duration_seconds",
                "Time spent loading batches"
            )?,
            memory_usage: register_gauge!(
                "dataloader_memory_usage_bytes",
                "Current memory usage"
            )?,
            error_count: register_counter!(
                "dataloader_errors_total",
                "Total number of loading errors"
            )?,
        })
    }
    
    pub fn record_batch_loaded(&self, duration: Duration) {
        self.batches_loaded.inc();
        self.load_duration.observe(duration.as_secs_f64());
    }
    
    pub fn record_error(&self) {
        self.error_count.inc();
    }
    
    pub fn update_memory_usage(&self, bytes: usize) {
        self.memory_usage.set(bytes as f64);
    }
}
```

This comprehensive best practices guide covers all aspects of efficient and robust data loading in the ToRSh framework. Follow these patterns to build high-performance, reliable data pipelines for your machine learning applications.