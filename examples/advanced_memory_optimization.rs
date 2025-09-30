//! Advanced Memory Optimization Techniques Demo
//!
//! This example demonstrates sophisticated memory optimization strategies including:
//! - Gradient checkpointing for memory-efficient training
//! - Dynamic memory pool management
//! - Memory-mapped datasets for large-scale training
//! - Activation compression and quantization
//! - Zero-copy operations and unified memory
//! - Memory-aware batch sizing
//! - Advanced garbage collection strategies

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use torsh::prelude::*;

/// Mock function for GPU memory usage - returns a dummy value
fn get_gpu_memory_usage(_device_id: usize) -> Result<u64> {
    // Mock implementation - return 1GB in bytes
    Ok(1024 * 1024 * 1024)
}

/// Mock function for cache emptying - no-op
fn empty_cache() {
    // Mock implementation - no-op
}

/// Simple FeedForward implementation
pub struct FeedForward {
    linear1: Linear,
    linear2: Linear,
    dropout: Dropout,
}

impl FeedForward {
    pub fn new(
        hidden_dim: usize,
        intermediate_dim: usize,
        dropout: f32,
        _device: DeviceType,
    ) -> Self {
        Self {
            linear1: Linear::new(hidden_dim, intermediate_dim, true),
            linear2: Linear::new(intermediate_dim, hidden_dim, true),
            dropout: Dropout::new(dropout),
        }
    }
}

impl Module for FeedForward {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let hidden = self.linear1.forward(input)?;
        let hidden = hidden.relu()?; // Use ReLU activation
        let hidden = self.dropout.forward(&hidden)?;
        self.linear2.forward(&hidden)
    }
}

/// Mock checkpoint function - just executes the closure
fn checkpoint<F, T>(f: F) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    f()
}

/// Memory optimization configuration
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub enable_checkpointing: bool,
    pub checkpoint_ratio: f64,
    pub enable_activation_compression: bool,
    pub compression_ratio: f64,
    pub enable_unified_memory: bool,
    pub memory_pool_size_gb: f64,
    pub max_memory_usage_ratio: f64,
    pub enable_memory_mapping: bool,
    pub prefetch_factor: usize,
    pub gc_threshold_mb: f64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enable_checkpointing: true,
            checkpoint_ratio: 0.5,
            enable_activation_compression: true,
            compression_ratio: 0.25,
            enable_unified_memory: false,
            memory_pool_size_gb: 8.0,
            max_memory_usage_ratio: 0.9,
            enable_memory_mapping: true,
            prefetch_factor: 4,
            gc_threshold_mb: 1024.0,
        }
    }
}

/// Advanced memory monitor
pub struct MemoryMonitor {
    peak_usage: Arc<Mutex<f64>>,
    current_usage: Arc<Mutex<f64>>,
    allocation_history: Arc<Mutex<Vec<(std::time::Instant, f64)>>>,
    gc_events: Arc<Mutex<Vec<std::time::Instant>>>,
    memory_pressure_callbacks: Vec<Box<dyn Fn(f64) -> bool + Send + Sync>>,
}

impl MemoryMonitor {
    pub fn new() -> Self {
        Self {
            peak_usage: Arc::new(Mutex::new(0.0)),
            current_usage: Arc::new(Mutex::new(0.0)),
            allocation_history: Arc::new(Mutex::new(Vec::new())),
            gc_events: Arc::new(Mutex::new(Vec::new())),
            memory_pressure_callbacks: Vec::new(),
        }
    }

    pub fn update_usage(&self, usage_mb: f64) {
        let mut current = self.current_usage.lock().unwrap();
        *current = usage_mb;

        let mut peak = self.peak_usage.lock().unwrap();
        if usage_mb > *peak {
            *peak = usage_mb;
        }

        let mut history = self.allocation_history.lock().unwrap();
        history.push((std::time::Instant::now(), usage_mb));

        // Keep only last 1000 entries
        if history.len() > 1000 {
            history.drain(0..500);
        }

        // Check memory pressure
        self.check_memory_pressure(usage_mb);
    }

    fn check_memory_pressure(&self, usage_mb: f64) {
        for callback in &self.memory_pressure_callbacks {
            callback(usage_mb);
        }
    }

    pub fn record_gc_event(&self) {
        let mut gc_events = self.gc_events.lock().unwrap();
        gc_events.push(std::time::Instant::now());
    }

    pub fn get_memory_stats(&self) -> MemoryStats {
        let current = *self.current_usage.lock().unwrap();
        let peak = *self.peak_usage.lock().unwrap();
        let history = self.allocation_history.lock().unwrap();
        let gc_count = self.gc_events.lock().unwrap().len();

        let avg_usage = if !history.is_empty() {
            history.iter().map(|(_, usage)| usage).sum::<f64>() / history.len() as f64
        } else {
            0.0
        };

        MemoryStats {
            current_mb: current,
            peak_mb: peak,
            average_mb: avg_usage,
            gc_count,
        }
    }
}

#[derive(Debug)]
pub struct MemoryStats {
    pub current_mb: f64,
    pub peak_mb: f64,
    pub average_mb: f64,
    pub gc_count: usize,
}

/// Memory-efficient transformer block with checkpointing
pub struct CheckpointedTransformerBlock {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    dropout: Dropout,
    checkpoint_attention: bool,
    checkpoint_ffn: bool,
}

impl CheckpointedTransformerBlock {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout_rate: f64,
        checkpoint_attention: bool,
        checkpoint_ffn: bool,
    ) -> Result<Self> {
        Ok(Self {
            attention: MultiHeadAttention::new(d_model, num_heads, dropout_rate as f32, true, DeviceType::Cpu)?,
            feed_forward: FeedForward::new(d_model, d_ff, dropout_rate as f32, DeviceType::Cpu),
            norm1: LayerNorm::new(vec![d_model], 1e-5, true, DeviceType::Cpu)?,
            norm2: LayerNorm::new(vec![d_model], 1e-5, true, DeviceType::Cpu)?,
            dropout: Dropout::new(dropout_rate as f32),
            checkpoint_attention,
            checkpoint_ffn,
        })
    }
}

impl Module for CheckpointedTransformerBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Self-attention with optional checkpointing
        let attn_fn = |input: &Tensor| -> Result<Tensor> {
            let normed = self.norm1.forward(input)?;
            let (attn_out, _) = self.attention.forward(&normed, &normed, &normed, None)?;
            let dropped = self.dropout.forward(&attn_out)?;
            input.add(&dropped)
        };

        let x = if self.checkpoint_attention {
            checkpoint(|| attn_fn(x))?
        } else {
            attn_fn(x)?
        };

        // Feed-forward with optional checkpointing
        let ffn_fn = |input: &Tensor| -> Result<Tensor> {
            let normed = self.norm2.forward(input)?;
            let ffn_out = self.feed_forward.forward(&normed)?;
            let dropped = self.dropout.forward(&ffn_out)?;
            input.add(&dropped)
        };

        if self.checkpoint_ffn {
            checkpoint(|| ffn_fn(&x))
        } else {
            ffn_fn(&x)
        }
    }
}

/// Memory-efficient data loader with memory mapping
pub struct MemoryMappedDataLoader {
    mmap_file: memmap2::Mmap,
    indices: Vec<usize>,
    batch_size: usize,
    current_batch: usize,
    prefetch_queue: Arc<Mutex<Vec<Tensor>>>,
    prefetch_thread: Option<std::thread::JoinHandle<()>>,
    sample_size: usize,
}

impl MemoryMappedDataLoader {
    pub fn new(
        file_path: &str,
        batch_size: usize,
        sample_size: usize,
        shuffle: bool,
        prefetch_factor: usize,
    ) -> Result<Self> {
        use memmap2::MmapOptions;
        use std::fs::File;

        let file = File::open(file_path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        let num_samples = mmap.len() / sample_size;
        let mut indices: Vec<usize> = (0..num_samples).collect();

        if shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }

        let prefetch_queue = Arc::new(Mutex::new(Vec::with_capacity(prefetch_factor)));

        Ok(Self {
            mmap_file: mmap,
            indices,
            batch_size,
            current_batch: 0,
            prefetch_queue,
            prefetch_thread: None,
            sample_size,
        })
    }

    pub fn start_prefetching(&mut self) {
        let mmap_ptr = self.mmap_file.as_ptr();
        let indices = self.indices.clone();
        let batch_size = self.batch_size;
        let sample_size = self.sample_size;
        let queue = Arc::clone(&self.prefetch_queue);

        self.prefetch_thread = Some(std::thread::spawn(move || {
            for batch_start in (0..indices.len()).step_by(batch_size) {
                let batch_end = std::cmp::min(batch_start + batch_size, indices.len());
                let mut batch_data = Vec::new();

                for i in batch_start..batch_end {
                    let sample_idx = indices[i];
                    let sample_offset = sample_idx * sample_size;

                    unsafe {
                        let sample_ptr = mmap_ptr.add(sample_offset);
                        let sample_slice = std::slice::from_raw_parts(sample_ptr, sample_size);

                        // Convert to tensor (assuming f32 data)
                        let tensor_data: Vec<f32> = sample_slice
                            .chunks_exact(4)
                            .map(|chunk| {
                                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                            })
                            .collect();

                        let tensor = Tensor::from_data(tensor_data, vec![sample_size / 4], DeviceType::Cpu).unwrap();
                        batch_data.push(tensor);
                    }
                }

                // Add batch to queue
                let mut queue = queue.lock().unwrap();
                let batch_refs: Vec<&Tensor> = batch_data.iter().collect();
                queue.push(Tensor::cat(&batch_refs, 0).unwrap());

                // Limit queue size to prevent memory bloat
                while queue.len() > 4 {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }
        }));
    }

    pub fn next_batch(&mut self) -> Option<Tensor> {
        let mut queue = self.prefetch_queue.lock().unwrap();
        if !queue.is_empty() {
            Some(queue.remove(0))
        } else {
            None
        }
    }
}

/// Activation compression utilities
pub struct ActivationCompressor {
    compression_ratio: f64,
    quantization_bits: u8,
    enable_huffman: bool,
}

impl ActivationCompressor {
    pub fn new(compression_ratio: f64, quantization_bits: u8, enable_huffman: bool) -> Self {
        Self {
            compression_ratio,
            quantization_bits,
            enable_huffman,
        }
    }

    pub fn compress(&self, tensor: &Tensor) -> Result<CompressedActivation> {
        // Quantization-based compression
        let (quantized, scale, zero_point) = self.quantize_tensor(tensor)?;

        // Optional Huffman encoding
        let compressed_data = if self.enable_huffman {
            self.huffman_encode(&quantized)?
        } else {
            // Convert f32 tensor data to u8 bytes
            let f32_data = quantized.to_vec()?;
            f32_data.into_iter()
                .flat_map(|f| f.to_le_bytes())
                .collect()
        };

        Ok(CompressedActivation {
            data: compressed_data,
            original_shape: tensor.shape().clone(),
            scale,
            zero_point,
            compression_method: if self.enable_huffman {
                CompressionMethod::QuantizedHuffman
            } else {
                CompressionMethod::Quantized
            },
        })
    }

    pub fn decompress(&self, compressed: &CompressedActivation) -> Result<Tensor> {
        let quantized_data = match compressed.compression_method {
            CompressionMethod::QuantizedHuffman => self.huffman_decode(&compressed.data)?,
            CompressionMethod::Quantized => compressed.data.clone(),
        };

        // Reconstruct quantized tensor
        let quantized_tensor = Tensor::from_data(quantized_data, compressed.original_shape.dims().to_vec(), DeviceType::Cpu)?;

        // Dequantize
        self.dequantize_tensor(&quantized_tensor, compressed.scale, compressed.zero_point)
    }

    fn quantize_tensor(&self, tensor: &Tensor) -> Result<(Tensor, f64, u8)> {
        let min_val = tensor.min()?.item::<f32>()? as f64;
        let max_val = tensor.max()?.item::<f32>()? as f64;

        let num_levels = (1 << self.quantization_bits) - 1;
        let scale = (max_val - min_val) / num_levels as f64;
        let zero_point = (-min_val / scale).round() as u8;

        let quantized = tensor
            .sub_scalar(min_val as f32)?
            .div_scalar(scale as f32)?
            .add_scalar(zero_point as f32)?
            .clamp(0.0, num_levels as f32)?
            .to_dtype(DType::U8)?;

        Ok((quantized, scale, zero_point))
    }

    fn dequantize_tensor(&self, quantized: &Tensor, scale: f64, zero_point: u8) -> Result<Tensor> {
        quantized
            .to_dtype(DType::F32)?
            .sub_scalar(zero_point as f32)?
            .mul_scalar(scale as f32)
    }

    fn huffman_encode(&self, data: &Tensor) -> Result<Vec<u8>> {
        // Simplified Huffman encoding implementation for f32 data
        let data_vec = data.to_vec()?;

        // Convert f32 to bytes for encoding
        let mut byte_data = Vec::new();
        for value in data_vec {
            byte_data.extend_from_slice(&value.to_le_bytes());
        }

        // Simple RLE encoding
        let mut encoded = Vec::new();
        let mut i = 0;
        while i < byte_data.len() {
            let current = byte_data[i];
            let mut count = 1;

            while i + count < byte_data.len() && byte_data[i + count] == current && count < 255 {
                count += 1;
            }

            encoded.push(current);
            encoded.push(count as u8);
            i += count;
        }

        Ok(encoded)
    }

    fn huffman_decode(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Simplified Huffman decoding (actually RLE decoding)
        let mut decoded = Vec::new();

        for chunk in data.chunks_exact(2) {
            let value = chunk[0];
            let count = chunk[1] as usize;

            for _ in 0..count {
                decoded.push(value);
            }
        }

        Ok(decoded)
    }
}

#[derive(Debug, Clone)]
pub struct CompressedActivation {
    data: Vec<u8>,
    original_shape: Shape,
    scale: f64,
    zero_point: u8,
    compression_method: CompressionMethod,
}

#[derive(Debug, Clone)]
pub enum CompressionMethod {
    Quantized,
    QuantizedHuffman,
}

/// Memory-aware model with dynamic batch sizing
pub struct MemoryAwareModel {
    base_model: Sequential,
    memory_monitor: Arc<MemoryMonitor>,
    compressor: ActivationCompressor,
    current_batch_size: usize,
    max_batch_size: usize,
    min_batch_size: usize,
    memory_config: MemoryConfig,
}

impl MemoryAwareModel {
    pub fn new(
        base_model: Sequential,
        memory_monitor: Arc<MemoryMonitor>,
        memory_config: MemoryConfig,
    ) -> Self {
        let compressor = ActivationCompressor::new(
            memory_config.compression_ratio,
            8,    // quantization_bits
            true, // enable_huffman
        );

        Self {
            base_model,
            memory_monitor,
            compressor,
            current_batch_size: 32,
            max_batch_size: 128,
            min_batch_size: 8,
            memory_config,
        }
    }

    pub fn adaptive_forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let memory_usage = get_gpu_memory_usage(0)? as f64 / 1024.0 / 1024.0;
        self.memory_monitor.update_usage(memory_usage);

        // Adjust batch size based on memory pressure
        self.adjust_batch_size(memory_usage);

        // Forward pass with optional activation compression
        if self.memory_config.enable_activation_compression {
            self.forward_with_compression(input)
        } else {
            self.base_model.forward(input)
        }
    }

    fn adjust_batch_size(&mut self, memory_usage_mb: f64) {
        let max_memory_mb = self.memory_config.memory_pool_size_gb * 1024.0;
        let usage_ratio = memory_usage_mb / max_memory_mb;

        if usage_ratio > self.memory_config.max_memory_usage_ratio {
            // Reduce batch size
            self.current_batch_size = std::cmp::max(
                self.min_batch_size,
                (self.current_batch_size as f64 * 0.8) as usize,
            );
        } else if usage_ratio < 0.5 {
            // Increase batch size
            self.current_batch_size = std::cmp::min(
                self.max_batch_size,
                (self.current_batch_size as f64 * 1.2) as usize,
            );
        }
    }

    fn forward_with_compression(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();

        for (i, layer) in self.base_model.modules().into_iter().enumerate() {
            x = layer.forward(&x)?;

            // Compress activations for certain layers
            if i % 3 == 0 && self.should_compress_layer(i) {
                let compressed = self.compressor.compress(&x)?;
                x = self.compressor.decompress(&compressed)?;
            }
        }

        Ok(x)
    }

    fn should_compress_layer(&self, layer_idx: usize) -> bool {
        // Compress every 3rd layer or based on memory pressure
        let memory_stats = self.memory_monitor.get_memory_stats();
        let max_memory_mb = self.memory_config.memory_pool_size_gb * 1024.0;
        let usage_ratio = memory_stats.current_mb / max_memory_mb;

        layer_idx % 3 == 0 || usage_ratio > 0.8
    }

    pub fn get_optimal_batch_size(&self) -> usize {
        self.current_batch_size
    }
}

/// Advanced memory pool manager
pub struct AdvancedMemoryPool {
    pool_size_bytes: usize,
    allocated_blocks: HashMap<usize, (usize, bool)>, // (size, is_free)
    free_blocks: std::collections::BTreeMap<usize, Vec<usize>>, // size -> list of offsets
    total_allocated: usize,
    fragmentation_threshold: f64,
}

impl AdvancedMemoryPool {
    pub fn new(pool_size_gb: f64) -> Self {
        let pool_size_bytes = (pool_size_gb * 1024.0 * 1024.0 * 1024.0) as usize;

        let mut free_blocks = std::collections::BTreeMap::new();
        free_blocks.insert(pool_size_bytes, vec![0]);

        Self {
            pool_size_bytes,
            allocated_blocks: HashMap::new(),
            free_blocks,
            total_allocated: 0,
            fragmentation_threshold: 0.2,
        }
    }

    pub fn allocate(&mut self, size: usize) -> Option<usize> {
        // Find best fit block
        let mut found_allocation = None;
        let mut cleanup_size = None;

        for (&block_size, offsets) in self.free_blocks.range_mut(size..) {
            if let Some(offset) = offsets.pop() {
                found_allocation = Some((offset, block_size));
                if offsets.is_empty() {
                    cleanup_size = Some(block_size);
                }
                break;
            }
        }

        if let Some((offset, block_size)) = found_allocation {
            self.allocated_blocks.insert(offset, (size, false));
            self.total_allocated += size;

            // Split block if necessary
            if block_size > size {
                let remaining_size = block_size - size;
                let remaining_offset = offset + size;

                self.free_blocks
                    .entry(remaining_size)
                    .or_default()
                    .push(remaining_offset);
            }

            // Clean up empty size classes
            if let Some(cleanup_size) = cleanup_size {
                self.free_blocks.remove(&cleanup_size);
            }

            Some(offset)
        } else {
            None
        }
    }

    pub fn deallocate(&mut self, offset: usize) -> bool {
        if let Some((size, _)) = self.allocated_blocks.remove(&offset) {
            self.total_allocated -= size;

            // Try to coalesce with adjacent free blocks
            let coalesced_block = self.coalesce_free_blocks(offset, size);

            self.free_blocks
                .entry(coalesced_block.1)
                .or_default()
                .push(coalesced_block.0);

            true
        } else {
            false
        }
    }

    fn coalesce_free_blocks(&mut self, offset: usize, size: usize) -> (usize, usize) {
        let mut start = offset;
        let mut total_size = size;

        // Check for adjacent free blocks and merge them
        // This is a simplified implementation
        (start, total_size)
    }

    pub fn get_fragmentation_ratio(&self) -> f64 {
        if self.pool_size_bytes == 0 {
            return 0.0;
        }

        let free_space = self.pool_size_bytes - self.total_allocated;
        let largest_free_block = self
            .free_blocks
            .iter()
            .last()
            .map(|(&size, _)| size)
            .unwrap_or(0);

        if free_space == 0 {
            1.0
        } else {
            1.0 - (largest_free_block as f64 / free_space as f64)
        }
    }

    pub fn defragment(&mut self) {
        // Simplified defragmentation
        if self.get_fragmentation_ratio() > self.fragmentation_threshold {
            // In a real implementation, this would compact memory
            println!("Memory pool fragmentation detected, defragmenting...");
        }
    }
}

/// Training function with advanced memory optimization
pub fn run_memory_optimized_training() -> Result<()> {
    println!("Starting memory-optimized training...");

    let config = MemoryConfig::default();
    let memory_monitor = Arc::new(MemoryMonitor::new());

    // Create memory pool
    let mut memory_pool = AdvancedMemoryPool::new(config.memory_pool_size_gb);

    // Create model with memory awareness
    let base_model = Sequential::new()
        .add(Linear::new(784, 1024, true))
        .add(ReLU::new())
        .add(Linear::new(1024, 512, true))
        .add(ReLU::new())
        .add(Linear::new(512, 10, true));

    let mut model = MemoryAwareModel::new(base_model, Arc::clone(&memory_monitor), config.clone());

    // Create memory-mapped data loader
    let mut dataloader = MemoryMappedDataLoader::new(
        "./data/train.bin",
        32,      // batch_size
        784 * 4, // sample_size (784 floats * 4 bytes)
        true,    // shuffle
        config.prefetch_factor,
    )?;

    dataloader.start_prefetching();

    // Training loop with memory monitoring
    for epoch in 0..10 {
        println!("Epoch {}", epoch + 1);

        let mut batch_count = 0;
        while let Some(batch) = dataloader.next_batch() {
            // Forward pass with memory optimization
            let output = model.adaptive_forward(&batch)?;

            // Memory monitoring
            if batch_count % 100 == 0 {
                let stats = memory_monitor.get_memory_stats();
                println!(
                    "Batch {}: Memory usage: {:.2}MB (peak: {:.2}MB), Batch size: {}",
                    batch_count,
                    stats.current_mb,
                    stats.peak_mb,
                    model.get_optimal_batch_size()
                );

                // Check fragmentation
                let fragmentation = memory_pool.get_fragmentation_ratio();
                if fragmentation > 0.3 {
                    println!("High fragmentation detected: {:.2}", fragmentation);
                    memory_pool.defragment();
                }
            }

            batch_count += 1;

            // Simulate memory pressure
            if batch_count > 1000 {
                break;
            }
        }

        // Trigger garbage collection
        if config.enable_unified_memory {
            empty_cache();
            memory_monitor.record_gc_event();
        }
    }

    // Print final statistics
    let final_stats = memory_monitor.get_memory_stats();
    println!("\n=== Memory Training Summary ===");
    println!("Peak memory usage: {:.2}MB", final_stats.peak_mb);
    println!("Average memory usage: {:.2}MB", final_stats.average_mb);
    println!("Garbage collection events: {}", final_stats.gc_count);
    println!(
        "Final fragmentation: {:.2}",
        memory_pool.get_fragmentation_ratio()
    );

    Ok(())
}

fn main() -> Result<()> {
    run_memory_optimized_training()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_monitor() {
        let monitor = MemoryMonitor::new();
        monitor.update_usage(1024.0);
        monitor.update_usage(2048.0);

        let stats = monitor.get_memory_stats();
        assert_eq!(stats.current_mb, 2048.0);
        assert_eq!(stats.peak_mb, 2048.0);
    }

    #[test]
    fn test_activation_compressor() {
        let compressor = ActivationCompressor::new(0.25, 8, false);
        let tensor = randn(&[4, 4]);

        let compressed = compressor.compress(&tensor).unwrap();
        let decompressed = compressor.decompress(&compressed).unwrap();

        assert_eq!(tensor.shape(), decompressed.shape());
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = AdvancedMemoryPool::new(1.0); // 1GB

        let addr1 = pool.allocate(1024).unwrap();
        let addr2 = pool.allocate(2048).unwrap();

        assert_ne!(addr1, addr2);
        assert!(pool.deallocate(addr1));
        assert!(pool.deallocate(addr2));
    }
}
