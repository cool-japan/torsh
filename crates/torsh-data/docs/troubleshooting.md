# Troubleshooting Guide for ToRSh Data Loading

This guide helps diagnose and resolve common issues when using the ToRSh data loading framework.

## Table of Contents

1. [Common Error Messages](#common-error-messages)
2. [Performance Issues](#performance-issues)
3. [Memory Problems](#memory-problems)
4. [Threading and Concurrency Issues](#threading-and-concurrency-issues)
5. [GPU-Related Problems](#gpu-related-problems)
6. [Data Corruption and Validation](#data-corruption-and-validation)
7. [Integration Issues](#integration-issues)
8. [Debugging Tools and Techniques](#debugging-tools-and-techniques)
9. [Configuration Problems](#configuration-problems)
10. [Platform-Specific Issues](#platform-specific-issues)

## Common Error Messages

### IndexError: Index out of bounds

**Error Message:**
```
IndexError: Index 150 out of bounds (size: 100)
```

**Cause:** Accessing dataset with invalid index

**Solutions:**
```rust
// Problem: Incorrect dataset length calculation
impl Dataset for MyDataset {
    fn len(&self) -> usize {
        self.data.len() + 10  // Wrong! Adding extra length
    }
}

// Solution: Return correct length
impl Dataset for MyDataset {
    fn len(&self) -> usize {
        self.data.len()  // Correct
    }
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        // Always validate bounds
        if index >= self.len() {
            return Err(DataError::IndexOutOfBounds {
                index,
                size: self.len(),
            }.into());
        }
        // ... rest of implementation
    }
}

// Additional validation for edge cases
fn create_safe_indices(dataset_len: usize, batch_size: usize) -> Vec<usize> {
    (0..dataset_len)
        .step_by(batch_size)
        .map(|i| i.min(dataset_len - 1))  // Clamp to valid range
        .collect()
}
```

### DataError: Shape mismatch in batch collation

**Error Message:**
```
DataError: Cannot collate tensors with shapes [224, 224, 3] and [256, 256, 3]
```

**Cause:** Inconsistent tensor shapes in batch

**Solutions:**
```rust
// Problem: Mixed image sizes in dataset
let dataset = ImageDataset::new(image_paths); // Images have different sizes

// Solution 1: Consistent preprocessing
let transform = Compose::new(vec![
    Box::new(Resize::new((224, 224))),  // Ensure consistent size
    Box::new(ToTensor),
]);
let dataset = TransformedDataset::new(dataset, transform);

// Solution 2: Custom collate function for variable sizes
struct VariableSizeCollate;

impl Collate<Tensor<f32>> for VariableSizeCollate {
    type Output = Vec<Tensor<f32>>;  // Return list instead of batched tensor
    
    fn collate(&self, batch: Vec<Tensor<f32>>) -> Result<Self::Output> {
        // Don't stack - return as is for variable sizes
        Ok(batch)
    }
}

// Solution 3: Padding for NLP sequences
struct PaddedCollate {
    pad_token: i32,
}

impl Collate<Vec<i32>> for PaddedCollate {
    type Output = Tensor<i32>;
    
    fn collate(&self, batch: Vec<Vec<i32>>) -> Result<Self::Output> {
        let max_len = batch.iter().map(|seq| seq.len()).max().unwrap_or(0);
        
        let mut padded_batch = Vec::new();
        for mut seq in batch {
            seq.resize(max_len, self.pad_token);
            padded_batch.extend(seq);
        }
        
        Tensor::from_slice(&padded_batch, &[batch.len(), max_len])
    }
}
```

### MemoryError: Out of memory

**Error Message:**
```
MemoryError: Failed to allocate 4GB for tensor
```

**Cause:** Insufficient memory for large batches or datasets

**Solutions:**
```rust
// Problem: Batch size too large
let dataloader = DataLoader::builder(dataset)
    .batch_size(1024)  // Too large!
    .build();

// Solution 1: Reduce batch size
let dataloader = DataLoader::builder(dataset)
    .batch_size(32)    // More reasonable
    .build();

// Solution 2: Implement memory monitoring
struct MemoryAwareDataLoader<D> {
    inner: DataLoader<D>,
    max_memory_mb: usize,
    current_batch_size: usize,
    min_batch_size: usize,
    max_batch_size: usize,
}

impl<D: Dataset> MemoryAwareDataLoader<D> {
    pub fn new(dataset: D, initial_batch_size: usize, max_memory_mb: usize) -> Self {
        let dataloader = DataLoader::builder(dataset)
            .batch_size(initial_batch_size)
            .build();
        
        Self {
            inner: dataloader,
            max_memory_mb,
            current_batch_size: initial_batch_size,
            min_batch_size: 1,
            max_batch_size: initial_batch_size * 4,
        }
    }
    
    fn adapt_batch_size(&mut self) -> bool {
        let current_memory = get_memory_usage_mb();
        
        if current_memory > self.max_memory_mb * 80 / 100 {  // 80% threshold
            // Reduce batch size
            self.current_batch_size = (self.current_batch_size * 3 / 4).max(self.min_batch_size);
            true
        } else if current_memory < self.max_memory_mb * 50 / 100 {  // 50% threshold
            // Increase batch size
            self.current_batch_size = (self.current_batch_size * 5 / 4).min(self.max_batch_size);
            true
        } else {
            false
        }
    }
}

// Solution 3: Use streaming for large datasets
struct StreamingDataset<T> {
    data_source: Box<dyn Iterator<Item = Result<T>> + Send>,
}

impl<T> IterableDataset for StreamingDataset<T> {
    type Item = T;
    type Iter = Box<dyn Iterator<Item = Result<T>> + Send>;
    
    fn iter(&self) -> Self::Iter {
        // Stream data instead of loading all at once
        Box::new(self.data_source.by_ref())
    }
}
```

### WorkerError: Worker thread panicked

**Error Message:**
```
WorkerError: Worker thread 2 panicked: called `Result::unwrap()` on an `Err` value
```

**Cause:** Exception in worker thread, often due to unhandled errors

**Solutions:**
```rust
// Problem: Unhandled errors in worker threads
fn worker_function(dataset: &Dataset, indices: Vec<usize>) {
    for index in indices {
        let item = dataset.get(index).unwrap();  // This can panic!
        process_item(item);
    }
}

// Solution: Proper error handling
fn robust_worker_function(dataset: &Dataset, indices: Vec<usize>) -> Result<Vec<ProcessedItem>> {
    let mut results = Vec::new();
    
    for index in indices {
        match dataset.get(index) {
            Ok(item) => {
                match process_item(item) {
                    Ok(processed) => results.push(processed),
                    Err(e) => {
                        eprintln!("Warning: Failed to process item {}: {}", index, e);
                        // Continue with next item instead of panicking
                        continue;
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: Failed to load item {}: {}", index, e);
                // Option 1: Skip item
                continue;
                // Option 2: Return error (depends on use case)
                // return Err(e);
            }
        }
    }
    
    Ok(results)
}

// Enhanced worker with panic recovery
struct RobustWorker {
    worker_id: usize,
    panic_count: AtomicUsize,
    max_panics: usize,
}

impl RobustWorker {
    fn run_with_recovery<F, R>(&self, work: F) -> Result<R>
    where
        F: FnOnce() -> Result<R> + UnwindSafe,
    {
        match std::panic::catch_unwind(|| work()) {
            Ok(result) => result,
            Err(panic_info) => {
                let panic_count = self.panic_count.fetch_add(1, Ordering::Relaxed);
                
                eprintln!("Worker {} panicked (count: {}): {:?}", 
                         self.worker_id, panic_count + 1, panic_info);
                
                if panic_count >= self.max_panics {
                    return Err(WorkerError::TooManyPanics {
                        worker_id: self.worker_id,
                        panic_count: panic_count + 1,
                    }.into());
                }
                
                // Return a recoverable error instead of propagating panic
                Err(WorkerError::WorkerPanic {
                    worker_id: self.worker_id,
                }.into())
            }
        }
    }
}
```

## Performance Issues

### Slow Data Loading

**Symptoms:**
- High CPU wait time
- Low GPU utilization
- Training stalls waiting for data

**Diagnosis:**
```rust
// Add performance monitoring
struct PerformanceDiagnostic {
    load_times: Vec<Duration>,
    batch_sizes: Vec<usize>,
    start_time: Instant,
}

impl PerformanceDiagnostic {
    fn diagnose_bottlenecks(&self) -> Vec<BottleneckType> {
        let mut bottlenecks = Vec::new();
        
        let avg_load_time = self.load_times.iter().sum::<Duration>() / self.load_times.len() as u32;
        let total_samples: usize = self.batch_sizes.iter().sum();
        let elapsed = self.start_time.elapsed();
        let throughput = total_samples as f64 / elapsed.as_secs_f64();
        
        if avg_load_time > Duration::from_millis(100) {
            bottlenecks.push(BottleneckType::SlowDataLoading);
        }
        
        if throughput < 100.0 {  // samples per second
            bottlenecks.push(BottleneckType::LowThroughput);
        }
        
        let cpu_usage = get_cpu_utilization();
        if cpu_usage > 90.0 {
            bottlenecks.push(BottleneckType::CPUBound);
        }
        
        let io_wait = get_io_wait_percentage();
        if io_wait > 30.0 {
            bottlenecks.push(BottleneckType::IOBound);
        }
        
        bottlenecks
    }
}

#[derive(Debug)]
enum BottleneckType {
    SlowDataLoading,
    LowThroughput,
    CPUBound,
    IOBound,
    MemoryBound,
}
```

**Solutions:**

1. **Optimize I/O:**
```rust
// Use memory mapping for large files
let dataset = MmapDataset::new("large_file.bin", item_size)?;

// Implement prefetching
let prefetch_dataset = PrefetchDataset::new(dataset, prefetch_size: 100);

// Use SSD storage
// Consider network-attached storage for distributed training
```

2. **Increase Parallelism:**
```rust
// Optimize worker count
let optimal_workers = std::cmp::min(
    num_cpus::get(),
    available_memory_gb * 2,  // Rule of thumb
);

let dataloader = DataLoader::builder(dataset)
    .num_workers(optimal_workers)
    .pin_memory(true)  // For GPU training
    .build();
```

3. **Cache Frequently Used Data:**
```rust
struct CachedDataset<D> {
    inner: D,
    cache: Arc<RwLock<LruCache<usize, D::Item>>>,
    cache_hit_rate: AtomicUsize,
    total_accesses: AtomicUsize,
}

impl<D: Dataset> CachedDataset<D>
where
    D::Item: Clone,
{
    fn get_cache_stats(&self) -> (f64, usize) {
        let hits = self.cache_hit_rate.load(Ordering::Relaxed);
        let total = self.total_accesses.load(Ordering::Relaxed);
        let hit_rate = if total > 0 { hits as f64 / total as f64 } else { 0.0 };
        (hit_rate, total)
    }
}
```

### Memory Leaks

**Symptoms:**
- Gradually increasing memory usage
- Out of memory errors after long runs
- Poor performance due to memory pressure

**Diagnosis:**
```rust
// Memory leak detector
struct MemoryLeakDetector {
    snapshots: Vec<MemorySnapshot>,
    check_interval: Duration,
    leak_threshold_mb: f64,
}

#[derive(Debug, Clone)]
struct MemorySnapshot {
    timestamp: Instant,
    rss_mb: f64,
    heap_mb: f64,
    active_objects: usize,
}

impl MemoryLeakDetector {
    fn check_for_leaks(&mut self) -> Option<MemoryLeak> {
        let current_snapshot = self.take_snapshot();
        self.snapshots.push(current_snapshot.clone());
        
        if self.snapshots.len() < 3 {
            return None;
        }
        
        // Check trend over last few snapshots
        let recent_snapshots = &self.snapshots[self.snapshots.len()-3..];
        let memory_trend = self.calculate_trend(recent_snapshots);
        
        if memory_trend > self.leak_threshold_mb {
            Some(MemoryLeak {
                trend_mb_per_sec: memory_trend,
                current_usage_mb: current_snapshot.rss_mb,
                snapshots: recent_snapshots.to_vec(),
            })
        } else {
            None
        }
    }
    
    fn calculate_trend(&self, snapshots: &[MemorySnapshot]) -> f64 {
        if snapshots.len() < 2 {
            return 0.0;
        }
        
        let time_diff = snapshots.last().unwrap().timestamp
            .duration_since(snapshots.first().unwrap().timestamp);
        let memory_diff = snapshots.last().unwrap().rss_mb 
            - snapshots.first().unwrap().rss_mb;
        
        memory_diff / time_diff.as_secs_f64()
    }
}
```

**Solutions:**

1. **Use RAII and Smart Pointers:**
```rust
// Problem: Manual memory management
struct BadDataset {
    raw_data: *mut u8,
    size: usize,
}

// Solution: Use smart pointers
struct GoodDataset {
    data: Arc<Vec<u8>>,  // Automatic cleanup
    metadata: Box<Metadata>,
}

// Use memory pools for frequent allocations
struct PoolAllocatedDataset {
    pool: Arc<MemoryPool<DataItem>>,
    items: Vec<PoolPtr<DataItem>>,
}
```

2. **Implement Proper Drop:**
```rust
struct ResourceDataset {
    file_handles: Vec<File>,
    gpu_buffers: Vec<CudaBuffer>,
}

impl Drop for ResourceDataset {
    fn drop(&mut self) {
        // Explicit cleanup of resources
        for buffer in &mut self.gpu_buffers {
            buffer.free();
        }
        // Files are automatically closed by their Drop impl
    }
}
```

## Threading and Concurrency Issues

### Race Conditions

**Symptoms:**
- Inconsistent results between runs
- Occasional crashes or panics
- Data corruption

**Solutions:**
```rust
// Problem: Unsynchronized access to shared state
struct UnsafeDataset {
    cache: HashMap<usize, DataItem>,  // Not thread-safe!
    access_count: usize,
}

// Solution: Proper synchronization
struct SafeDataset {
    cache: Arc<RwLock<HashMap<usize, DataItem>>>,
    access_count: AtomicUsize,
}

impl Dataset for SafeDataset {
    fn get(&self, index: usize) -> Result<Self::Item> {
        self.access_count.fetch_add(1, Ordering::Relaxed);
        
        // Read-only access
        {
            let cache = self.cache.read().unwrap();
            if let Some(item) = cache.get(&index) {
                return Ok(item.clone());
            }
        }
        
        // Write access (separate scope to avoid deadlock)
        let item = self.load_item(index)?;
        {
            let mut cache = self.cache.write().unwrap();
            cache.insert(index, item.clone());
        }
        
        Ok(item)
    }
}

// Lock-free alternatives for high-performance scenarios
use crossbeam::queue::SegQueue;

struct LockFreeQueue<T> {
    queue: SegQueue<T>,
}

impl<T> LockFreeQueue<T> {
    fn push(&self, item: T) {
        self.queue.push(item);
    }
    
    fn pop(&self) -> Option<T> {
        self.queue.pop()
    }
}
```

### Deadlocks

**Symptoms:**
- Application hangs
- No progress in data loading
- High CPU usage with no work being done

**Solutions:**
```rust
// Problem: Lock ordering issues
struct DeadlockProneDataset {
    cache1: Mutex<HashMap<usize, DataItem>>,
    cache2: Mutex<HashMap<String, MetaData>>,
}

impl DeadlockProneDataset {
    fn bad_method(&self, index: usize, key: &str) {
        let _c1 = self.cache1.lock().unwrap();  // Lock A
        let _c2 = self.cache2.lock().unwrap();  // Lock B
    }
    
    fn another_bad_method(&self, key: &str, index: usize) {
        let _c2 = self.cache2.lock().unwrap();  // Lock B first
        let _c1 = self.cache1.lock().unwrap();  // Lock A second - DEADLOCK!
    }
}

// Solution: Consistent lock ordering
struct DeadlockFreeDataset {
    cache1: Mutex<HashMap<usize, DataItem>>,
    cache2: Mutex<HashMap<String, MetaData>>,
}

impl DeadlockFreeDataset {
    fn safe_method(&self, index: usize, key: &str) {
        // Always acquire locks in the same order
        let _c1 = self.cache1.lock().unwrap();  // Always lock A first
        let _c2 = self.cache2.lock().unwrap();  // Always lock B second
    }
    
    // Even better: Avoid multiple locks
    fn better_method(&self, index: usize, key: &str) -> Result<CombinedResult> {
        let cache1_data = {
            let cache = self.cache1.lock().unwrap();
            cache.get(&index).cloned()
        };  // Lock released here
        
        let cache2_data = {
            let cache = self.cache2.lock().unwrap();
            cache.get(key).cloned()
        };  // Lock released here
        
        Ok(CombinedResult { cache1_data, cache2_data })
    }
}

// Timeout-based locking for deadlock detection
use std::time::Duration;

fn try_with_timeout<T, F>(mutex: &Mutex<T>, timeout: Duration, f: F) -> Result<R>
where
    F: FnOnce(&mut T) -> R,
{
    match mutex.try_lock_for(timeout) {
        Some(mut guard) => Ok(f(&mut *guard)),
        None => Err(DataError::LockTimeout { timeout }.into()),
    }
}
```

## GPU-Related Problems

### CUDA Out of Memory

**Error Message:**
```
CUDA Error: out of memory (error code: 2)
```

**Solutions:**
```rust
// Monitor GPU memory usage
struct GPUMemoryMonitor {
    peak_usage: AtomicUsize,
    current_usage: AtomicUsize,
    allocation_count: AtomicUsize,
}

impl GPUMemoryMonitor {
    fn allocate(&self, size: usize) -> Result<GPUPtr> {
        let available = get_gpu_memory_available()?;
        
        if size > available {
            // Try to free some memory first
            self.run_garbage_collection();
            
            let available_after_gc = get_gpu_memory_available()?;
            if size > available_after_gc {
                return Err(GPUError::OutOfMemory {
                    requested: size,
                    available: available_after_gc,
                }.into());
            }
        }
        
        let ptr = cuda_malloc(size)?;
        self.current_usage.fetch_add(size, Ordering::Relaxed);
        self.allocation_count.fetch_add(1, Ordering::Relaxed);
        
        Ok(ptr)
    }
    
    fn run_garbage_collection(&self) {
        // Force collection of unused GPU memory
        cuda_empty_cache();
        
        // Optional: Move some data back to CPU if needed
        self.offload_to_cpu_if_needed();
    }
}

// Implement GPU memory pools
struct GPUMemoryPool {
    free_blocks: Vec<Vec<GPUBlock>>,
    size_classes: Vec<usize>,
}

impl GPUMemoryPool {
    fn get_block(&mut self, size: usize) -> Result<GPUBlock> {
        let size_class = self.find_size_class(size);
        
        if let Some(block) = self.free_blocks[size_class].pop() {
            Ok(block)
        } else {
            // Allocate new block
            self.allocate_new_block(self.size_classes[size_class])
        }
    }
}
```

### GPU Transfer Issues

**Symptoms:**
- Slow training despite fast loading
- High CPU-GPU transfer overhead

**Solutions:**
```rust
// Asynchronous GPU transfers
struct AsyncGPUDataLoader<D> {
    cpu_loader: DataLoader<D>,
    transfer_queue: VecDeque<TransferJob>,
    gpu_streams: Vec<CudaStream>,
}

struct TransferJob {
    cpu_data: Tensor<f32>,
    gpu_promise: Promise<Tensor<f32>>,
    stream_id: usize,
}

impl<D: Dataset> AsyncGPUDataLoader<D> {
    fn start_async_transfer(&mut self, cpu_batch: Tensor<f32>) -> Future<Tensor<f32>> {
        let stream_id = self.get_next_stream();
        let (promise, future) = create_promise();
        
        let job = TransferJob {
            cpu_data: cpu_batch,
            gpu_promise: promise,
            stream_id,
        };
        
        self.transfer_queue.push_back(job);
        self.process_transfer_queue();
        
        future
    }
    
    fn process_transfer_queue(&mut self) {
        while let Some(job) = self.transfer_queue.pop_front() {
            let stream = &self.gpu_streams[job.stream_id];
            
            // Start async transfer
            let gpu_tensor = job.cpu_data.to_device_async(&self.device, stream)?;
            
            // Set callback for completion
            stream.add_callback(move || {
                job.gpu_promise.set(gpu_tensor);
            });
        }
    }
}

// Pinned memory for faster transfers
struct PinnedMemoryDataLoader<D> {
    inner: DataLoader<D>,
    pinned_buffers: Vec<PinnedBuffer>,
    buffer_index: AtomicUsize,
}

impl<D: Dataset> PinnedMemoryDataLoader<D> {
    fn new(inner: DataLoader<D>, num_buffers: usize) -> Self {
        let pinned_buffers = (0..num_buffers)
            .map(|_| PinnedBuffer::new(estimate_buffer_size(&inner)))
            .collect();
        
        Self {
            inner,
            pinned_buffers,
            buffer_index: AtomicUsize::new(0),
        }
    }
    
    fn get_next_batch(&mut self) -> Result<Tensor<f32>> {
        let buffer_idx = self.buffer_index.fetch_add(1, Ordering::Relaxed) 
            % self.pinned_buffers.len();
        let buffer = &mut self.pinned_buffers[buffer_idx];
        
        // Load data into pinned memory
        let cpu_batch = self.inner.next().unwrap()?;
        buffer.copy_from_tensor(&cpu_batch);
        
        // Fast transfer from pinned memory
        buffer.to_gpu_async()
    }
}
```

## Data Corruption and Validation

### Detecting Corrupted Data

**Symptoms:**
- Training loss becomes NaN
- Inconsistent model performance
- Unexpected tensor values

**Solutions:**
```rust
// Data validation layer
struct ValidatedDataset<D> {
    inner: D,
    validator: DataValidator,
    corruption_count: AtomicUsize,
    total_samples: AtomicUsize,
}

struct DataValidator {
    validate_range: bool,
    min_value: f32,
    max_value: f32,
    validate_shape: bool,
    expected_shape: Vec<usize>,
    validate_checksum: bool,
}

impl<D: Dataset> ValidatedDataset<D>
where
    D::Item: Validateable,
{
    fn get(&self, index: usize) -> Result<D::Item> {
        let item = self.inner.get(index)?;
        self.total_samples.fetch_add(1, Ordering::Relaxed);
        
        match self.validator.validate(&item) {
            Ok(()) => Ok(item),
            Err(validation_error) => {
                self.corruption_count.fetch_add(1, Ordering::Relaxed);
                
                eprintln!("Data corruption detected at index {}: {:?}", 
                         index, validation_error);
                
                // Decide whether to return error or try to recover
                if self.should_attempt_recovery(&validation_error) {
                    self.attempt_recovery(index, item, validation_error)
                } else {
                    Err(validation_error.into())
                }
            }
        }
    }
    
    fn attempt_recovery(
        &self, 
        index: usize, 
        mut item: D::Item, 
        error: ValidationError
    ) -> Result<D::Item> {
        match error {
            ValidationError::OutOfRange { .. } => {
                // Clamp values to valid range
                item.clamp(self.validator.min_value, self.validator.max_value);
                Ok(item)
            }
            ValidationError::InvalidShape { .. } => {
                // Try to reshape or pad
                item.reshape_or_pad(&self.validator.expected_shape)
            }
            ValidationError::NaNDetected { .. } => {
                // Replace NaN with zeros
                item.replace_nan_with_zero();
                Ok(item)
            }
            _ => Err(error.into()),
        }
    }
    
    pub fn get_corruption_rate(&self) -> f64 {
        let corrupted = self.corruption_count.load(Ordering::Relaxed);
        let total = self.total_samples.load(Ordering::Relaxed);
        
        if total > 0 {
            corrupted as f64 / total as f64
        } else {
            0.0
        }
    }
}

trait Validateable {
    fn validate(&self, validator: &DataValidator) -> Result<(), ValidationError>;
    fn clamp(&mut self, min: f32, max: f32);
    fn reshape_or_pad(&mut self, target_shape: &[usize]) -> Result<Self, ValidationError>;
    fn replace_nan_with_zero(&mut self);
}

#[derive(Debug)]
enum ValidationError {
    OutOfRange { value: f32, min: f32, max: f32 },
    InvalidShape { actual: Vec<usize>, expected: Vec<usize> },
    NaNDetected { location: String },
    ChecksumMismatch { expected: u64, actual: u64 },
}
```

### Checksum Validation

```rust
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

struct ChecksumDataset<D> {
    inner: D,
    checksums: HashMap<usize, u64>,
    validation_mode: ChecksumMode,
}

enum ChecksumMode {
    Strict,    // Fail on mismatch
    Warning,   // Log warning but continue
    Disabled,  // No validation
}

impl<D: Dataset> ChecksumDataset<D>
where
    D::Item: Hash,
{
    fn compute_checksum(item: &D::Item) -> u64 {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        hasher.finish()
    }
    
    fn validate_checksum(&self, index: usize, item: &D::Item) -> Result<()> {
        let computed = Self::compute_checksum(item);
        
        if let Some(&expected) = self.checksums.get(&index) {
            match self.validation_mode {
                ChecksumMode::Strict => {
                    if computed != expected {
                        return Err(ValidationError::ChecksumMismatch {
                            expected,
                            actual: computed,
                        }.into());
                    }
                }
                ChecksumMode::Warning => {
                    if computed != expected {
                        eprintln!("Warning: Checksum mismatch at index {}: expected {}, got {}", 
                                index, expected, computed);
                    }
                }
                ChecksumMode::Disabled => {}
            }
        }
        
        Ok(())
    }
}
```

## Debugging Tools and Techniques

### Logging and Tracing

```rust
use tracing::{debug, error, info, instrument, span, warn, Level};

#[instrument(level = "debug")]
impl<D: Dataset> Dataset for DebugDataset<D> {
    type Item = D::Item;
    
    #[instrument(skip(self), fields(index = %index))]
    fn get(&self, index: usize) -> Result<Self::Item> {
        let span = span!(Level::DEBUG, "dataset_get", index = %index);
        let _enter = span.enter();
        
        debug!("Loading item at index {}", index);
        
        let start_time = Instant::now();
        let result = self.inner.get(index);
        let elapsed = start_time.elapsed();
        
        match &result {
            Ok(_) => {
                debug!("Successfully loaded item {} in {:?}", index, elapsed);
                if elapsed > Duration::from_millis(100) {
                    warn!("Slow data loading: {}ms for index {}", elapsed.as_millis(), index);
                }
            }
            Err(e) => {
                error!("Failed to load item {}: {:?}", index, e);
            }
        }
        
        result
    }
}

// Performance tracing
struct TracingDataLoader<D> {
    inner: DataLoader<D>,
    tracer: Arc<Mutex<Tracer>>,
}

struct Tracer {
    events: Vec<TraceEvent>,
    start_time: Instant,
}

#[derive(Debug, Clone)]
struct TraceEvent {
    timestamp: Duration,
    event_type: String,
    metadata: HashMap<String, String>,
    duration: Option<Duration>,
}

impl<D: Dataset> TracingDataLoader<D> {
    fn trace_batch_load<F, R>(&self, batch_id: usize, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        
        let event = TraceEvent {
            timestamp: start.duration_since(self.tracer.lock().unwrap().start_time),
            event_type: "batch_load".to_string(),
            metadata: {
                let mut map = HashMap::new();
                map.insert("batch_id".to_string(), batch_id.to_string());
                map
            },
            duration: Some(duration),
        };
        
        self.tracer.lock().unwrap().events.push(event);
        result
    }
    
    pub fn export_trace(&self, path: &Path) -> Result<()> {
        let tracer = self.tracer.lock().unwrap();
        let json = serde_json::to_string_pretty(&tracer.events)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}
```

### Memory Debugging

```rust
// Memory usage tracker
struct MemoryDebugger {
    allocations: HashMap<*const u8, AllocationInfo>,
    total_allocated: usize,
    peak_allocated: usize,
}

#[derive(Debug)]
struct AllocationInfo {
    size: usize,
    stack_trace: String,
    timestamp: Instant,
}

impl MemoryDebugger {
    fn track_allocation(&mut self, ptr: *const u8, size: usize) {
        let stack_trace = get_stack_trace();
        
        self.allocations.insert(ptr, AllocationInfo {
            size,
            stack_trace,
            timestamp: Instant::now(),
        });
        
        self.total_allocated += size;
        self.peak_allocated = self.peak_allocated.max(self.total_allocated);
    }
    
    fn track_deallocation(&mut self, ptr: *const u8) {
        if let Some(info) = self.allocations.remove(&ptr) {
            self.total_allocated -= info.size;
        }
    }
    
    fn find_leaks(&self) -> Vec<LeakReport> {
        let now = Instant::now();
        
        self.allocations.iter()
            .filter(|(_, info)| now.duration_since(info.timestamp) > Duration::from_secs(300))
            .map(|(ptr, info)| LeakReport {
                ptr: *ptr,
                size: info.size,
                age: now.duration_since(info.timestamp),
                stack_trace: info.stack_trace.clone(),
            })
            .collect()
    }
}

#[derive(Debug)]
struct LeakReport {
    ptr: *const u8,
    size: usize,
    age: Duration,
    stack_trace: String,
}

fn get_stack_trace() -> String {
    // Implementation would use backtrace crate
    "stack trace placeholder".to_string()
}
```

### Integration Testing

```rust
// Comprehensive integration test
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_end_to_end_data_pipeline() {
        // Create test dataset
        let dataset = create_test_dataset(1000);
        
        // Test various configurations
        let configs = vec![
            DataLoaderConfig { batch_size: 32, num_workers: 1, shuffle: false },
            DataLoaderConfig { batch_size: 64, num_workers: 4, shuffle: true },
            DataLoaderConfig { batch_size: 16, num_workers: 2, shuffle: true },
        ];
        
        for config in configs {
            test_dataloader_config(dataset.clone(), config);
        }
    }
    
    fn test_dataloader_config(dataset: TestDataset, config: DataLoaderConfig) {
        let dataloader = DataLoader::builder(dataset.clone())
            .batch_size(config.batch_size)
            .num_workers(config.num_workers)
            .shuffle(config.shuffle)
            .build();
        
        let mut total_samples = 0;
        let mut batch_count = 0;
        let mut load_times = Vec::new();
        
        for batch_result in dataloader {
            let start = Instant::now();
            let batch = batch_result.expect("Batch loading failed");
            let load_time = start.elapsed();
            
            // Validate batch
            assert!(!batch.is_empty(), "Empty batch returned");
            assert!(batch.len() <= config.batch_size, "Batch size exceeded");
            
            total_samples += batch.len();
            batch_count += 1;
            load_times.push(load_time);
            
            // Test some batches then break to avoid long test times
            if batch_count >= 10 {
                break;
            }
        }
        
        // Performance assertions
        let avg_load_time = load_times.iter().sum::<Duration>() / load_times.len() as u32;
        assert!(avg_load_time < Duration::from_millis(1000), 
               "Average load time too high: {:?}", avg_load_time);
        
        println!("Config {:?}: {} batches, {} samples, avg load time: {:?}", 
                config, batch_count, total_samples, avg_load_time);
    }
}
```

This comprehensive troubleshooting guide covers the most common issues encountered in data loading and provides practical solutions for each problem category. Use this guide to quickly diagnose and resolve issues in your ToRSh data loading pipelines.