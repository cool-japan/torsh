# Performance Guide for ToRSh Data Loading

This guide provides detailed performance optimization strategies for the ToRSh data loading framework.

## Table of Contents

1. [Performance Analysis Framework](#performance-analysis-framework)
2. [CPU Optimization](#cpu-optimization)
3. [Memory Optimization](#memory-optimization)
4. [I/O Optimization](#io-optimization)
5. [GPU Optimization](#gpu-optimization)
6. [Network and Distributed Optimization](#network-and-distributed-optimization)
7. [Profiling and Benchmarking](#profiling-and-benchmarking)
8. [Performance Tuning Strategies](#performance-tuning-strategies)
9. [Common Performance Pitfalls](#common-performance-pitfalls)
10. [Platform-Specific Optimizations](#platform-specific-optimizations)

## Performance Analysis Framework

### 1. Performance Metrics

**Key Performance Indicators (KPIs)**
```rust
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub throughput_samples_per_sec: f64,
    pub throughput_batches_per_sec: f64,
    pub latency_avg_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub memory_usage_mb: f64,
    pub memory_peak_mb: f64,
    pub cpu_utilization_percent: f64,
    pub io_wait_percent: f64,
    pub cache_hit_rate_percent: f64,
    pub queue_depth_avg: f64,
}

impl PerformanceMetrics {
    pub fn efficiency_score(&self) -> f64 {
        // Composite score: throughput * (1 - latency_penalty) * (1 - memory_penalty)
        let latency_penalty = (self.latency_p95_ms / 1000.0).min(0.5);
        let memory_penalty = (self.memory_usage_mb / 2048.0).min(0.5);
        
        self.throughput_samples_per_sec * (1.0 - latency_penalty) * (1.0 - memory_penalty)
    }
}
```

**Performance Monitor**
```rust
use std::time::{Duration, Instant};
use std::collections::VecDeque;

pub struct PerformanceMonitor {
    sample_times: VecDeque<Instant>,
    batch_sizes: VecDeque<usize>,
    latencies: VecDeque<Duration>,
    memory_samples: VecDeque<usize>,
    window_size: usize,
}

impl PerformanceMonitor {
    pub fn new(window_size: usize) -> Self {
        Self {
            sample_times: VecDeque::with_capacity(window_size),
            batch_sizes: VecDeque::with_capacity(window_size),
            latencies: VecDeque::with_capacity(window_size),
            memory_samples: VecDeque::with_capacity(window_size),
            window_size,
        }
    }
    
    pub fn record_batch(&mut self, batch_size: usize, latency: Duration) {
        let now = Instant::now();
        
        if self.sample_times.len() >= self.window_size {
            self.sample_times.pop_front();
            self.batch_sizes.pop_front();
            self.latencies.pop_front();
        }
        
        self.sample_times.push_back(now);
        self.batch_sizes.push_back(batch_size);
        self.latencies.push_back(latency);
        
        // Record memory usage
        if let Ok(memory) = get_current_memory_usage() {
            if self.memory_samples.len() >= self.window_size {
                self.memory_samples.pop_front();
            }
            self.memory_samples.push_back(memory);
        }
    }
    
    pub fn compute_metrics(&self) -> PerformanceMetrics {
        if self.sample_times.len() < 2 {
            return PerformanceMetrics::default();
        }
        
        let total_samples: usize = self.batch_sizes.iter().sum();
        let time_span = self.sample_times.back().unwrap()
            .duration_since(*self.sample_times.front().unwrap());
        
        let throughput_samples = total_samples as f64 / time_span.as_secs_f64();
        let throughput_batches = self.batch_sizes.len() as f64 / time_span.as_secs_f64();
        
        let mut latencies_ms: Vec<f64> = self.latencies.iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();
        latencies_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let latency_avg = latencies_ms.iter().sum::<f64>() / latencies_ms.len() as f64;
        let latency_p95 = percentile(&latencies_ms, 0.95);
        let latency_p99 = percentile(&latencies_ms, 0.99);
        
        let memory_avg = self.memory_samples.iter().sum::<usize>() as f64 
            / self.memory_samples.len() as f64 / 1024.0 / 1024.0; // Convert to MB
        let memory_peak = *self.memory_samples.iter().max().unwrap_or(&0) as f64 
            / 1024.0 / 1024.0;
        
        PerformanceMetrics {
            throughput_samples_per_sec: throughput_samples,
            throughput_batches_per_sec: throughput_batches,
            latency_avg_ms: latency_avg,
            latency_p95_ms: latency_p95,
            latency_p99_ms: latency_p99,
            memory_usage_mb: memory_avg,
            memory_peak_mb: memory_peak,
            cpu_utilization_percent: get_cpu_utilization(),
            io_wait_percent: get_io_wait_percentage(),
            cache_hit_rate_percent: 0.0, // To be implemented
            queue_depth_avg: 0.0, // To be implemented
        }
    }
}

fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    let index = (p * (sorted_data.len() - 1) as f64) as usize;
    sorted_data[index.min(sorted_data.len() - 1)]
}
```

## CPU Optimization

### 1. SIMD Vectorization

**Vectorized Data Processing**
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct SIMDOptimizedTransform;

impl SIMDOptimizedTransform {
    #[inline]
    pub fn normalize_f32_slice(&self, data: &mut [f32], mean: f32, std: f32) {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { self.normalize_avx2(data, mean, std); }
                return;
            }
            if is_x86_feature_detected!("sse4.1") {
                unsafe { self.normalize_sse41(data, mean, std); }
                return;
            }
        }
        
        // Fallback scalar implementation
        let inv_std = 1.0 / std;
        for value in data.iter_mut() {
            *value = (*value - mean) * inv_std;
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn normalize_avx2(&self, data: &mut [f32], mean: f32, std: f32) {
        let mean_vec = _mm256_set1_ps(mean);
        let inv_std_vec = _mm256_set1_ps(1.0 / std);
        
        let chunks = data.chunks_exact_mut(8);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let values = _mm256_loadu_ps(chunk.as_ptr());
            let centered = _mm256_sub_ps(values, mean_vec);
            let normalized = _mm256_mul_ps(centered, inv_std_vec);
            _mm256_storeu_ps(chunk.as_mut_ptr(), normalized);
        }
        
        // Handle remainder
        let inv_std = 1.0 / std;
        for value in remainder {
            *value = (*value - mean) * inv_std;
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.1")]
    unsafe fn normalize_sse41(&self, data: &mut [f32], mean: f32, std: f32) {
        let mean_vec = _mm_set1_ps(mean);
        let inv_std_vec = _mm_set1_ps(1.0 / std);
        
        let chunks = data.chunks_exact_mut(4);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let values = _mm_loadu_ps(chunk.as_ptr());
            let centered = _mm_sub_ps(values, mean_vec);
            let normalized = _mm_mul_ps(centered, inv_std_vec);
            _mm_storeu_ps(chunk.as_mut_ptr(), normalized);
        }
        
        // Handle remainder
        let inv_std = 1.0 / std;
        for value in remainder {
            *value = (*value - mean) * inv_std;
        }
    }
}
```

### 2. Parallel Processing

**Work-Stealing Task Pool**
```rust
use rayon::prelude::*;
use crossbeam::channel::{bounded, Receiver, Sender};
use std::sync::Arc;

pub struct WorkStealingDataLoader<D> {
    dataset: Arc<D>,
    work_queue: Sender<WorkItem>,
    result_queue: Receiver<ProcessedBatch>,
    num_workers: usize,
    batch_size: usize,
}

#[derive(Debug)]
struct WorkItem {
    indices: Vec<usize>,
    batch_id: usize,
}

#[derive(Debug)]
struct ProcessedBatch {
    batch_id: usize,
    data: Vec<Tensor<f32>>,
    processing_time: Duration,
}

impl<D: Dataset + Send + Sync + 'static> WorkStealingDataLoader<D>
where
    D::Item: Send + 'static,
{
    pub fn new(dataset: D, batch_size: usize, num_workers: usize) -> Self {
        let dataset = Arc::new(dataset);
        let (work_sender, work_receiver) = bounded(num_workers * 2);
        let (result_sender, result_receiver) = bounded(num_workers);
        
        // Spawn worker threads
        for worker_id in 0..num_workers {
            let dataset = Arc::clone(&dataset);
            let work_receiver = work_receiver.clone();
            let result_sender = result_sender.clone();
            
            std::thread::spawn(move || {
                Self::worker_loop(worker_id, dataset, work_receiver, result_sender);
            });
        }
        
        Self {
            dataset,
            work_queue: work_sender,
            result_queue: result_receiver,
            num_workers,
            batch_size,
        }
    }
    
    fn worker_loop(
        worker_id: usize,
        dataset: Arc<D>,
        work_receiver: Receiver<WorkItem>,
        result_sender: Sender<ProcessedBatch>,
    ) {
        while let Ok(work_item) = work_receiver.recv() {
            let start_time = Instant::now();
            
            // Process batch in parallel
            let batch_data: Result<Vec<_>, _> = work_item.indices
                .into_par_iter()
                .map(|idx| dataset.get(idx))
                .collect();
            
            match batch_data {
                Ok(data) => {
                    let processed_batch = ProcessedBatch {
                        batch_id: work_item.batch_id,
                        data,
                        processing_time: start_time.elapsed(),
                    };
                    
                    if result_sender.send(processed_batch).is_err() {
                        // Receiver dropped, exit worker
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("Worker {} failed to process batch {}: {:?}", 
                             worker_id, work_item.batch_id, e);
                }
            }
        }
    }
    
    pub fn adaptive_batch_size(&mut self, target_latency_ms: u64) -> usize {
        // Dynamically adjust batch size based on processing time
        let current_latency = self.measure_average_latency();
        
        if current_latency > Duration::from_millis(target_latency_ms) {
            self.batch_size = (self.batch_size * 3 / 4).max(1);
        } else if current_latency < Duration::from_millis(target_latency_ms / 2) {
            self.batch_size = (self.batch_size * 5 / 4).min(256);
        }
        
        self.batch_size
    }
}
```

### 3. CPU Cache Optimization

**Cache-Friendly Data Layout**
```rust
// Structure of Arrays (SoA) layout for better cache performance
pub struct SoADataset {
    features: Vec<f32>,     // All features contiguous
    labels: Vec<i32>,       // All labels contiguous
    metadata: Vec<u64>,     // All metadata contiguous
    item_count: usize,
    feature_dim: usize,
}

impl SoADataset {
    pub fn new(features: Vec<Vec<f32>>, labels: Vec<i32>, metadata: Vec<u64>) -> Self {
        let item_count = labels.len();
        let feature_dim = features[0].len();
        
        // Flatten features into SoA layout
        let mut flattened_features = Vec::with_capacity(item_count * feature_dim);
        for feature_vec in features {
            flattened_features.extend(feature_vec);
        }
        
        Self {
            features: flattened_features,
            labels,
            metadata,
            item_count,
            feature_dim,
        }
    }
    
    #[inline]
    pub fn get_feature_slice(&self, index: usize) -> &[f32] {
        let start = index * self.feature_dim;
        let end = start + self.feature_dim;
        &self.features[start..end]
    }
    
    // Prefetch next items to improve cache performance
    #[inline]
    pub fn prefetch_range(&self, start_index: usize, count: usize) {
        #[cfg(target_arch = "x86_64")]
        {
            for i in start_index..start_index + count {
                if i < self.item_count {
                    let feature_start = i * self.feature_dim;
                    unsafe {
                        // Prefetch features
                        std::arch::x86_64::_mm_prefetch(
                            self.features.as_ptr().add(feature_start) as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                        
                        // Prefetch labels and metadata
                        std::arch::x86_64::_mm_prefetch(
                            self.labels.as_ptr().add(i) as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
            }
        }
    }
}

impl Dataset for SoADataset {
    type Item = (Vec<f32>, i32, u64);
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.item_count {
            return Err(DataError::IndexOutOfBounds {
                index,
                size: self.item_count,
            }.into());
        }
        
        // Prefetch next few items
        self.prefetch_range(index + 1, 4);
        
        let features = self.get_feature_slice(index).to_vec();
        let label = self.labels[index];
        let metadata = self.metadata[index];
        
        Ok((features, label, metadata))
    }
}
```

## Memory Optimization

### 1. Memory Pool Management

**Advanced Memory Pool**
```rust
use std::sync::atomic::{AtomicUsize, Ordering};
use parking_lot::Mutex;

pub struct AdvancedMemoryPool<T> {
    pools: Vec<Mutex<Vec<Box<T>>>>,
    allocation_counter: AtomicUsize,
    deallocation_counter: AtomicUsize,
    total_allocated: AtomicUsize,
    max_pool_size: usize,
    size_classes: Vec<usize>,
}

impl<T: Default> AdvancedMemoryPool<T> {
    pub fn new(size_classes: Vec<usize>, max_pool_size: usize) -> Self {
        let pools = size_classes.iter()
            .map(|_| Mutex::new(Vec::new()))
            .collect();
        
        Self {
            pools,
            allocation_counter: AtomicUsize::new(0),
            deallocation_counter: AtomicUsize::new(0),
            total_allocated: AtomicUsize::new(0),
            max_pool_size,
            size_classes,
        }
    }
    
    pub fn allocate(&self, size_hint: usize) -> Box<T> {
        let pool_index = self.find_size_class(size_hint);
        
        if let Some(pool_index) = pool_index {
            if let Some(item) = self.pools[pool_index].lock().pop() {
                self.allocation_counter.fetch_add(1, Ordering::Relaxed);
                return item;
            }
        }
        
        // Create new item
        let item = Box::new(T::default());
        self.allocation_counter.fetch_add(1, Ordering::Relaxed);
        self.total_allocated.fetch_add(1, Ordering::Relaxed);
        item
    }
    
    pub fn deallocate(&self, item: Box<T>, size_hint: usize) {
        let pool_index = self.find_size_class(size_hint);
        
        if let Some(pool_index) = pool_index {
            let mut pool = self.pools[pool_index].lock();
            if pool.len() < self.max_pool_size {
                pool.push(item);
                self.deallocation_counter.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        
        // Let item drop naturally
        self.deallocation_counter.fetch_add(1, Ordering::Relaxed);
    }
    
    fn find_size_class(&self, size: usize) -> Option<usize> {
        self.size_classes.iter()
            .position(|&class_size| size <= class_size)
    }
    
    pub fn get_stats(&self) -> PoolStats {
        PoolStats {
            allocations: self.allocation_counter.load(Ordering::Relaxed),
            deallocations: self.deallocation_counter.load(Ordering::Relaxed),
            total_allocated: self.total_allocated.load(Ordering::Relaxed),
            pool_sizes: self.pools.iter()
                .map(|pool| pool.lock().len())
                .collect(),
        }
    }
}

#[derive(Debug)]
pub struct PoolStats {
    pub allocations: usize,
    pub deallocations: usize,
    pub total_allocated: usize,
    pub pool_sizes: Vec<usize>,
}
```

### 2. Zero-Copy Operations

**Zero-Copy Tensor Views**
```rust
use std::ptr::NonNull;

pub struct ZeroCopyTensorView<T> {
    data: NonNull<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> ZeroCopyTensorView<T> {
    pub unsafe fn from_raw_parts(
        data: *mut T,
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> Self {
        let len = shape.iter().product();
        Self {
            data: NonNull::new(data).expect("Null pointer"),
            shape,
            strides,
            len,
            _phantom: std::marker::PhantomData,
        }
    }
    
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.data.as_ptr(), self.len)
        }
    }
    
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.data.as_ptr(), self.len)
        }
    }
    
    // Create view without copying data
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Result<ZeroCopyTensorView<T>> {
        if ranges.len() != self.shape.len() {
            return Err(DataError::InvalidSlice.into());
        }
        
        let mut offset = 0;
        let mut new_shape = Vec::new();
        
        for (i, range) in ranges.iter().enumerate() {
            if range.end > self.shape[i] {
                return Err(DataError::InvalidSlice.into());
            }
            offset += range.start * self.strides[i];
            new_shape.push(range.end - range.start);
        }
        
        unsafe {
            Ok(ZeroCopyTensorView::from_raw_parts(
                self.data.as_ptr().add(offset),
                new_shape,
                self.strides.clone(),
            ))
        }
    }
}

// Safe wrapper for managed memory
pub struct ManagedTensorView<T> {
    view: ZeroCopyTensorView<T>,
    _owner: Arc<dyn Any + Send + Sync>, // Keep owner alive
}

impl<T> ManagedTensorView<T> {
    pub fn new<O: Any + Send + Sync>(
        owner: Arc<O>,
        data: *mut T,
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> Self {
        let view = unsafe {
            ZeroCopyTensorView::from_raw_parts(data, shape, strides)
        };
        
        Self {
            view,
            _owner: owner,
        }
    }
    
    pub fn as_slice(&self) -> &[T] {
        self.view.as_slice()
    }
}
```

## I/O Optimization

### 1. Asynchronous I/O

**Async File Reading with Batching**
```rust
use tokio::fs::File;
use tokio::io::{AsyncReadExt, BufReader};
use futures::stream::{self, StreamExt};

pub struct AsyncBatchFileReader {
    file_paths: Vec<PathBuf>,
    batch_size: usize,
    concurrent_limit: usize,
    buffer_size: usize,
}

impl AsyncBatchFileReader {
    pub fn new(
        file_paths: Vec<PathBuf>,
        batch_size: usize,
        concurrent_limit: usize,
    ) -> Self {
        Self {
            file_paths,
            batch_size,
            concurrent_limit,
            buffer_size: 64 * 1024, // 64KB buffer
        }
    }
    
    pub async fn read_batch(&self, batch_indices: Vec<usize>) -> Result<Vec<Vec<u8>>> {
        let file_futures = batch_indices.into_iter().map(|idx| {
            let path = self.file_paths[idx].clone();
            let buffer_size = self.buffer_size;
            
            async move {
                let file = File::open(&path).await?;
                let mut reader = BufReader::with_capacity(buffer_size, file);
                let mut buffer = Vec::new();
                reader.read_to_end(&mut buffer).await?;
                Ok::<Vec<u8>, std::io::Error>(buffer)
            }
        });
        
        // Process files concurrently with limit
        let results: Result<Vec<_>, _> = stream::iter(file_futures)
            .buffered(self.concurrent_limit)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect();
        
        results.map_err(|e| DataError::IoError(e.to_string()).into())
    }
}

// Async dataset with prefetching
pub struct AsyncPrefetchDataset<D> {
    inner: D,
    prefetch_queue: Arc<tokio::sync::Mutex<VecDeque<(usize, D::Item)>>>,
    prefetch_size: usize,
    current_index: AtomicUsize,
}

impl<D: Dataset + Send + Sync + 'static> AsyncPrefetchDataset<D>
where
    D::Item: Send + 'static,
{
    pub fn new(inner: D, prefetch_size: usize) -> Self {
        let dataset = Self {
            inner,
            prefetch_queue: Arc::new(tokio::sync::Mutex::new(VecDeque::new())),
            prefetch_size,
            current_index: AtomicUsize::new(0),
        };
        
        dataset.start_prefetching();
        dataset
    }
    
    fn start_prefetching(&self) {
        let inner = Arc::new(self.inner.clone());
        let queue = Arc::clone(&self.prefetch_queue);
        let prefetch_size = self.prefetch_size;
        let current_index = Arc::new(AtomicUsize::new(0));
        
        tokio::spawn(async move {
            loop {
                let current = current_index.load(Ordering::Relaxed);
                
                // Check queue size
                {
                    let queue_guard = queue.lock().await;
                    if queue_guard.len() >= prefetch_size {
                        drop(queue_guard);
                        tokio::time::sleep(Duration::from_millis(10)).await;
                        continue;
                    }
                }
                
                if current >= inner.len() {
                    break;
                }
                
                // Prefetch next item
                if let Ok(item) = inner.get(current) {
                    let mut queue_guard = queue.lock().await;
                    queue_guard.push_back((current, item));
                    current_index.store(current + 1, Ordering::Relaxed);
                }
            }
        });
    }
    
    pub async fn get_async(&self, index: usize) -> Result<D::Item> {
        // First check prefetch queue
        {
            let mut queue = self.prefetch_queue.lock().await;
            if let Some(pos) = queue.iter().position(|(i, _)| *i == index) {
                let (_, item) = queue.remove(pos).unwrap();
                return Ok(item);
            }
        }
        
        // Fallback to synchronous get
        self.inner.get(index)
    }
}
```

### 2. Memory-Mapped I/O

**High-Performance Memory Mapping**
```rust
use memmap2::{Mmap, MmapOptions};

pub struct HighPerfMmapDataset {
    mmap: Mmap,
    item_offsets: Vec<u64>,
    item_sizes: Vec<u32>,
    compression: Option<CompressionType>,
    cache: Arc<RwLock<LruCache<usize, CachedItem>>>,
}

#[derive(Clone)]
enum CompressionType {
    None,
    Lz4,
    Zstd,
}

#[derive(Clone)]
struct CachedItem {
    data: Vec<u8>,
    access_time: Instant,
    access_count: u32,
}

impl HighPerfMmapDataset {
    pub fn new<P: AsRef<Path>>(
        data_file: P,
        index_file: P,
        compression: Option<CompressionType>,
        cache_size: usize,
    ) -> Result<Self> {
        // Memory map the data file
        let file = File::open(data_file)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        // Load index
        let index_data = std::fs::read(index_file)?;
        let (item_offsets, item_sizes) = Self::parse_index(&index_data)?;
        
        let cache = Arc::new(RwLock::new(LruCache::new(cache_size)));
        
        Ok(Self {
            mmap,
            item_offsets,
            item_sizes,
            compression,
            cache,
        })
    }
    
    fn get_raw_data(&self, index: usize) -> Result<&[u8]> {
        if index >= self.item_offsets.len() {
            return Err(DataError::IndexOutOfBounds {
                index,
                size: self.item_offsets.len(),
            }.into());
        }
        
        let offset = self.item_offsets[index] as usize;
        let size = self.item_sizes[index] as usize;
        
        if offset + size > self.mmap.len() {
            return Err(DataError::CorruptedData {
                index,
                reason: "Offset/size exceeds file bounds".to_string(),
            }.into());
        }
        
        Ok(&self.mmap[offset..offset + size])
    }
    
    fn decompress_data(&self, compressed: &[u8]) -> Result<Vec<u8>> {
        match self.compression {
            None => Ok(compressed.to_vec()),
            Some(CompressionType::Lz4) => {
                lz4::block::decompress(compressed, None)
                    .map_err(|e| DataError::DecompressionFailed(e.to_string()).into())
            }
            Some(CompressionType::Zstd) => {
                zstd::bulk::decompress(compressed, 0)
                    .map_err(|e| DataError::DecompressionFailed(e.to_string()).into())
            }
        }
    }
}

impl Dataset for HighPerfMmapDataset {
    type Item = Vec<u8>;
    
    fn get(&self, index: usize) -> Result<Self::Item> {
        // Check cache first
        {
            let mut cache = self.cache.write().unwrap();
            if let Some(cached) = cache.get_mut(&index) {
                cached.access_count += 1;
                cached.access_time = Instant::now();
                return Ok(cached.data.clone());
            }
        }
        
        // Read from memory-mapped file
        let raw_data = self.get_raw_data(index)?;
        let data = self.decompress_data(raw_data)?;
        
        // Update cache
        {
            let mut cache = self.cache.write().unwrap();
            cache.put(index, CachedItem {
                data: data.clone(),
                access_time: Instant::now(),
                access_count: 1,
            });
        }
        
        Ok(data)
    }
}
```

## GPU Optimization

### 1. Efficient GPU Memory Management

**GPU Memory Pool**
```rust
use torsh_core::device::DeviceType;

pub struct GPUMemoryPool {
    device: DeviceType,
    free_blocks: Vec<Vec<GPUMemoryBlock>>, // Size-segregated free blocks
    allocated_blocks: HashMap<*mut u8, GPUMemoryBlock>,
    total_allocated: AtomicUsize,
    peak_allocated: AtomicUsize,
    size_classes: Vec<usize>,
}

#[derive(Debug, Clone)]
struct GPUMemoryBlock {
    ptr: *mut u8,
    size: usize,
    size_class: usize,
    allocated_at: Instant,
}

impl GPUMemoryPool {
    pub fn new(device: DeviceType) -> Self {
        let size_classes = vec![
            1024,      // 1KB
            4096,      // 4KB
            16384,     // 16KB
            65536,     // 64KB
            262144,    // 256KB
            1048576,   // 1MB
            4194304,   // 4MB
            16777216,  // 16MB
            67108864,  // 64MB
        ];
        
        let free_blocks = size_classes.iter()
            .map(|_| Vec::new())
            .collect();
        
        Self {
            device,
            free_blocks,
            allocated_blocks: HashMap::new(),
            total_allocated: AtomicUsize::new(0),
            peak_allocated: AtomicUsize::new(0),
            size_classes,
        }
    }
    
    pub fn allocate(&mut self, size: usize) -> Result<*mut u8> {
        let size_class_idx = self.find_size_class(size);
        let actual_size = self.size_classes[size_class_idx];
        
        // Try to reuse existing block
        if let Some(block) = self.free_blocks[size_class_idx].pop() {
            self.allocated_blocks.insert(block.ptr, block.clone());
            return Ok(block.ptr);
        }
        
        // Allocate new block
        let ptr = self.allocate_gpu_memory(actual_size)?;
        let block = GPUMemoryBlock {
            ptr,
            size: actual_size,
            size_class: size_class_idx,
            allocated_at: Instant::now(),
        };
        
        self.allocated_blocks.insert(ptr, block);
        
        let new_total = self.total_allocated.fetch_add(actual_size, Ordering::Relaxed) + actual_size;
        let mut peak = self.peak_allocated.load(Ordering::Relaxed);
        while new_total > peak {
            match self.peak_allocated.compare_exchange_weak(
                peak, new_total, Ordering::Relaxed, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
        
        Ok(ptr)
    }
    
    pub fn deallocate(&mut self, ptr: *mut u8) -> Result<()> {
        if let Some(block) = self.allocated_blocks.remove(&ptr) {
            self.total_allocated.fetch_sub(block.size, Ordering::Relaxed);
            
            // Return to free pool for reuse
            if self.free_blocks[block.size_class].len() < 10 { // Limit pool size
                self.free_blocks[block.size_class].push(block);
            } else {
                // Actually free the memory
                self.free_gpu_memory(ptr)?;
            }
        }
        
        Ok(())
    }
    
    fn find_size_class(&self, size: usize) -> usize {
        self.size_classes.iter()
            .position(|&class_size| size <= class_size)
            .unwrap_or(self.size_classes.len() - 1)
    }
}
```

### 2. Asynchronous GPU Operations

**Multi-Stream GPU Processing**
```rust
pub struct MultiStreamGPUProcessor {
    streams: Vec<CudaStream>,
    current_stream: AtomicUsize,
    pending_operations: Arc<Mutex<VecDeque<GPUOperation>>>,
    completion_queue: Arc<Mutex<VecDeque<CompletedOperation>>>,
}

#[derive(Debug)]
struct GPUOperation {
    operation_id: u64,
    stream_id: usize,
    operation_type: OperationType,
    data: Vec<u8>,
    callback: Box<dyn Fn(Result<Vec<u8>>) + Send>,
}

#[derive(Debug)]
enum OperationType {
    Transfer,
    Normalize,
    Resize,
    Transform,
}

impl MultiStreamGPUProcessor {
    pub fn new(num_streams: usize) -> Result<Self> {
        let streams = (0..num_streams)
            .map(|_| CudaStream::new())
            .collect::<Result<Vec<_>, _>>()?;
        
        let processor = Self {
            streams,
            current_stream: AtomicUsize::new(0),
            pending_operations: Arc::new(Mutex::new(VecDeque::new())),
            completion_queue: Arc::new(Mutex::new(VecDeque::new())),
        };
        
        processor.start_processing_threads();
        Ok(processor)
    }
    
    pub fn submit_async<F>(&self, data: Vec<u8>, operation: OperationType, callback: F) 
    where
        F: Fn(Result<Vec<u8>>) + Send + 'static,
    {
        let stream_id = self.current_stream.fetch_add(1, Ordering::Relaxed) % self.streams.len();
        let operation_id = generate_operation_id();
        
        let gpu_op = GPUOperation {
            operation_id,
            stream_id,
            operation_type: operation,
            data,
            callback: Box::new(callback),
        };
        
        self.pending_operations.lock().unwrap().push_back(gpu_op);
    }
    
    fn start_processing_threads(&self) {
        for stream_id in 0..self.streams.len() {
            let pending_ops = Arc::clone(&self.pending_operations);
            let completion_queue = Arc::clone(&self.completion_queue);
            
            std::thread::spawn(move || {
                loop {
                    let op = {
                        let mut pending = pending_ops.lock().unwrap();
                        pending.iter()
                            .position(|op| op.stream_id == stream_id)
                            .map(|pos| pending.remove(pos).unwrap())
                    };
                    
                    if let Some(operation) = op {
                        let result = Self::execute_operation(&operation);
                        
                        let completed = CompletedOperation {
                            operation_id: operation.operation_id,
                            result,
                            callback: operation.callback,
                        };
                        
                        completion_queue.lock().unwrap().push_back(completed);
                    } else {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }
            });
        }
        
        // Completion thread
        let completion_queue = Arc::clone(&self.completion_queue);
        std::thread::spawn(move || {
            loop {
                if let Some(completed) = completion_queue.lock().unwrap().pop_front() {
                    (completed.callback)(completed.result);
                } else {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
        });
    }
}
```

## Profiling and Benchmarking

### 1. Comprehensive Profiler

**DataLoader Profiler**
```rust
use std::time::{Duration, Instant};
use std::collections::HashMap;

pub struct DataLoaderProfiler {
    events: Vec<ProfileEvent>,
    start_time: Instant,
    active_spans: HashMap<u64, Instant>,
    next_span_id: AtomicU64,
    enable_detailed_tracing: bool,
}

#[derive(Debug, Clone)]
pub struct ProfileEvent {
    pub timestamp: Duration,
    pub event_type: EventType,
    pub span_id: u64,
    pub duration: Option<Duration>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum EventType {
    DatasetGet,
    Transform,
    Collate,
    BatchLoad,
    GPUTransfer,
    CacheHit,
    CacheMiss,
    MemoryAllocation,
    MemoryDeallocation,
}

impl DataLoaderProfiler {
    pub fn new(enable_detailed_tracing: bool) -> Self {
        Self {
            events: Vec::new(),
            start_time: Instant::now(),
            active_spans: HashMap::new(),
            next_span_id: AtomicU64::new(0),
            enable_detailed_tracing,
        }
    }
    
    pub fn start_span(&mut self, event_type: EventType) -> ProfileSpan {
        let span_id = self.next_span_id.fetch_add(1, Ordering::Relaxed);
        let start_time = Instant::now();
        
        self.active_spans.insert(span_id, start_time);
        
        if self.enable_detailed_tracing {
            self.events.push(ProfileEvent {
                timestamp: start_time.duration_since(self.start_time),
                event_type: event_type.clone(),
                span_id,
                duration: None,
                metadata: HashMap::new(),
            });
        }
        
        ProfileSpan {
            span_id,
            event_type,
            start_time,
            profiler: self as *mut Self,
        }
    }
    
    pub fn end_span(&mut self, span: ProfileSpan) {
        if let Some(start_time) = self.active_spans.remove(&span.span_id) {
            let duration = span.start_time.elapsed();
            
            self.events.push(ProfileEvent {
                timestamp: start_time.duration_since(self.start_time),
                event_type: span.event_type,
                span_id: span.span_id,
                duration: Some(duration),
                metadata: HashMap::new(),
            });
        }
    }
    
    pub fn generate_report(&self) -> ProfileReport {
        let mut event_stats: HashMap<String, EventStats> = HashMap::new();
        
        for event in &self.events {
            if let Some(duration) = event.duration {
                let event_name = format!("{:?}", event.event_type);
                let stats = event_stats.entry(event_name).or_insert(EventStats::default());
                
                stats.count += 1;
                stats.total_duration += duration;
                stats.min_duration = stats.min_duration.min(duration);
                stats.max_duration = stats.max_duration.max(duration);
                stats.durations.push(duration);
            }
        }
        
        // Calculate percentiles
        for stats in event_stats.values_mut() {
            stats.durations.sort();
            stats.median = percentile_duration(&stats.durations, 0.5);
            stats.p95 = percentile_duration(&stats.durations, 0.95);
            stats.p99 = percentile_duration(&stats.durations, 0.99);
            stats.avg_duration = stats.total_duration / stats.count as u32;
        }
        
        ProfileReport {
            total_duration: self.start_time.elapsed(),
            event_stats,
            total_events: self.events.len(),
        }
    }
}

pub struct ProfileSpan {
    span_id: u64,
    event_type: EventType,
    start_time: Instant,
    profiler: *mut DataLoaderProfiler,
}

impl Drop for ProfileSpan {
    fn drop(&mut self) {
        unsafe {
            (*self.profiler).end_span(ProfileSpan {
                span_id: self.span_id,
                event_type: self.event_type.clone(),
                start_time: self.start_time,
                profiler: self.profiler,
            });
        }
    }
}

#[derive(Debug, Default)]
pub struct EventStats {
    pub count: usize,
    pub total_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub avg_duration: Duration,
    pub median: Duration,
    pub p95: Duration,
    pub p99: Duration,
    durations: Vec<Duration>,
}

#[derive(Debug)]
pub struct ProfileReport {
    pub total_duration: Duration,
    pub event_stats: HashMap<String, EventStats>,
    pub total_events: usize,
}

impl ProfileReport {
    pub fn print_summary(&self) {
        println!("=== DataLoader Performance Report ===");
        println!("Total Duration: {:?}", self.total_duration);
        println!("Total Events: {}", self.total_events);
        println!();
        
        for (event_type, stats) in &self.event_stats {
            println!("{}: {} events", event_type, stats.count);
            println!("  Total: {:?}", stats.total_duration);
            println!("  Avg: {:?}", stats.avg_duration);
            println!("  Min: {:?}", stats.min_duration);
            println!("  Max: {:?}", stats.max_duration);
            println!("  Median: {:?}", stats.median);
            println!("  P95: {:?}", stats.p95);
            println!("  P99: {:?}", stats.p99);
            println!();
        }
    }
}
```

This comprehensive performance guide provides detailed strategies for optimizing every aspect of data loading in the ToRSh framework. Apply these techniques based on your specific performance requirements and bottlenecks.