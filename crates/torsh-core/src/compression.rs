//! Tensor compression schemes and pruning metadata for ToRSh Core
//!
//! This module provides comprehensive tensor compression techniques including:
//! - Magnitude-based pruning with threshold detection
//! - Structured pruning patterns (block-wise, channel-wise, attention head-wise)
//! - Compression encoding schemes (run-length, Huffman, delta encoding)
//! - Quantization-aware compression metadata

use crate::dtype::DType;
use crate::shape::Shape;

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

/// Pruning strategies for tensor compression
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PruningStrategy {
    /// Magnitude-based pruning: remove weights below threshold
    Magnitude { threshold_percentile: u8 },

    /// Structured pruning: remove entire blocks
    BlockWise { block_size: (usize, usize) },

    /// Channel-wise pruning for convolutional layers
    ChannelWise { channels_to_prune: usize },

    /// Attention head pruning for transformer models
    AttentionHead { heads_to_prune: usize },

    /// Movement pruning: prune based on weight movement during training
    Movement { sensitivity: f32 },

    /// Gradual magnitude pruning with schedule
    GradualMagnitude {
        initial_sparsity: f32,
        final_sparsity: f32,
    },
}

impl PruningStrategy {
    /// Get the expected sparsity for this strategy
    pub fn expected_sparsity(&self) -> f32 {
        match self {
            Self::Magnitude {
                threshold_percentile,
            } => *threshold_percentile as f32 / 100.0,
            Self::GradualMagnitude { final_sparsity, .. } => *final_sparsity,
            _ => 0.5, // Default 50% for other strategies
        }
    }

    /// Check if this is a structured pruning strategy
    pub fn is_structured(&self) -> bool {
        matches!(
            self,
            Self::BlockWise { .. } | Self::ChannelWise { .. } | Self::AttentionHead { .. }
        )
    }
}

/// Pruning metadata tracking which elements/blocks are pruned
#[derive(Debug, Clone)]
pub struct PruningMetadata {
    /// Pruning strategy used
    strategy: PruningStrategy,

    /// Pruned element indices (for unstructured pruning)
    pruned_indices: Option<Vec<usize>>,

    /// Pruned block indices (for structured pruning)
    pruned_blocks: Option<Vec<(usize, usize)>>,

    /// Pruned channel indices (for channel-wise pruning)
    pruned_channels: Option<Vec<usize>>,

    /// Actual sparsity achieved
    achieved_sparsity: f32,

    /// Original tensor shape before pruning
    original_shape: Shape,

    /// Threshold value used (for magnitude pruning)
    threshold_value: Option<f32>,

    /// Compression ratio (original_size / pruned_size)
    compression_ratio: f32,
}

impl PruningMetadata {
    /// Create new pruning metadata
    pub fn new(strategy: PruningStrategy, original_shape: Shape, achieved_sparsity: f32) -> Self {
        let compression_ratio = 1.0 / (1.0 - achieved_sparsity);

        Self {
            strategy,
            pruned_indices: None,
            pruned_blocks: None,
            pruned_channels: None,
            achieved_sparsity,
            original_shape,
            threshold_value: None,
            compression_ratio,
        }
    }

    /// Set pruned indices for unstructured pruning
    pub fn with_indices(mut self, indices: Vec<usize>) -> Self {
        self.pruned_indices = Some(indices);
        self
    }

    /// Set pruned blocks for structured pruning
    pub fn with_blocks(mut self, blocks: Vec<(usize, usize)>) -> Self {
        self.pruned_blocks = Some(blocks);
        self
    }

    /// Set pruned channels
    pub fn with_channels(mut self, channels: Vec<usize>) -> Self {
        self.pruned_channels = Some(channels);
        self
    }

    /// Set threshold value
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold_value = Some(threshold);
        self
    }

    /// Get pruning strategy
    pub fn strategy(&self) -> PruningStrategy {
        self.strategy
    }

    /// Get achieved sparsity
    pub fn sparsity(&self) -> f32 {
        self.achieved_sparsity
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        self.compression_ratio
    }

    /// Get number of pruned elements
    pub fn num_pruned_elements(&self) -> usize {
        if let Some(ref indices) = self.pruned_indices {
            indices.len()
        } else if let Some(ref blocks) = self.pruned_blocks {
            blocks.len()
        } else if let Some(ref channels) = self.pruned_channels {
            channels.len()
        } else {
            0
        }
    }

    /// Check if an element is pruned
    pub fn is_element_pruned(&self, index: usize) -> bool {
        if let Some(ref indices) = self.pruned_indices {
            indices.binary_search(&index).is_ok()
        } else {
            false
        }
    }

    /// Get memory savings in bytes
    pub fn memory_savings(&self, dtype: DType) -> usize {
        let total_elements = self.original_shape.numel();
        let pruned_elements = (total_elements as f32 * self.achieved_sparsity) as usize;
        pruned_elements * dtype.size()
    }
}

/// Compression encoding schemes for sparse tensor indices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompressionEncoding {
    /// No encoding (raw values)
    Raw,

    /// Run-length encoding for consecutive indices
    RunLength,

    /// Delta encoding for sequential indices
    Delta,

    /// Huffman encoding for variable-length codes
    Huffman,

    /// Bitmap encoding for dense regions
    Bitmap,

    /// Hybrid encoding combining multiple schemes
    Hybrid,
}

impl CompressionEncoding {
    /// Get the expected compression ratio for typical sparse tensors
    pub fn expected_compression_ratio(&self) -> f32 {
        match self {
            Self::Raw => 1.0,
            Self::RunLength => 2.0,
            Self::Delta => 1.5,
            Self::Huffman => 2.5,
            Self::Bitmap => 3.0,
            Self::Hybrid => 3.5,
        }
    }

    /// Check if encoding requires sorted indices
    pub fn requires_sorted_indices(&self) -> bool {
        matches!(self, Self::RunLength | Self::Delta | Self::Hybrid)
    }
}

/// Run-length encoded index sequence
#[derive(Debug, Clone)]
pub struct RunLengthEncoded {
    /// Starting indices of runs
    start_indices: Vec<usize>,

    /// Lengths of runs
    run_lengths: Vec<usize>,

    /// Total number of elements encoded
    total_elements: usize,
}

impl RunLengthEncoded {
    /// Create new run-length encoding from sorted indices
    pub fn encode(indices: &[usize]) -> Self {
        if indices.is_empty() {
            return Self {
                start_indices: vec![],
                run_lengths: vec![],
                total_elements: 0,
            };
        }

        let mut start_indices = Vec::new();
        let mut run_lengths = Vec::new();

        let mut current_start = indices[0];
        let mut current_length = 1;

        for i in 1..indices.len() {
            if indices[i] == indices[i - 1] + 1 {
                // Continue current run
                current_length += 1;
            } else {
                // Start new run
                start_indices.push(current_start);
                run_lengths.push(current_length);
                current_start = indices[i];
                current_length = 1;
            }
        }

        // Push last run
        start_indices.push(current_start);
        run_lengths.push(current_length);

        Self {
            start_indices,
            run_lengths,
            total_elements: indices.len(),
        }
    }

    /// Decode run-length encoding back to indices
    pub fn decode(&self) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.total_elements);

        for (start, length) in self.start_indices.iter().zip(self.run_lengths.iter()) {
            for offset in 0..*length {
                indices.push(start + offset);
            }
        }

        indices
    }

    /// Get compression ratio (original_size / compressed_size)
    pub fn compression_ratio(&self) -> f32 {
        if self.start_indices.is_empty() {
            return 1.0;
        }

        let original_size = self.total_elements * std::mem::size_of::<usize>();
        let compressed_size =
            (self.start_indices.len() + self.run_lengths.len()) * std::mem::size_of::<usize>();

        original_size as f32 / compressed_size as f32
    }

    /// Get number of runs
    pub fn num_runs(&self) -> usize {
        self.start_indices.len()
    }
}

/// Delta-encoded index sequence
#[derive(Debug, Clone)]
pub struct DeltaEncoded {
    /// Base index (first element)
    base_index: usize,

    /// Delta values (differences between consecutive indices)
    deltas: Vec<i32>,

    /// Total number of elements
    total_elements: usize,
}

impl DeltaEncoded {
    /// Create new delta encoding from sorted indices
    pub fn encode(indices: &[usize]) -> Self {
        if indices.is_empty() {
            return Self {
                base_index: 0,
                deltas: vec![],
                total_elements: 0,
            };
        }

        let base_index = indices[0];
        let mut deltas = Vec::with_capacity(indices.len() - 1);

        for i in 1..indices.len() {
            let delta = (indices[i] as i64 - indices[i - 1] as i64) as i32;
            deltas.push(delta);
        }

        Self {
            base_index,
            deltas,
            total_elements: indices.len(),
        }
    }

    /// Decode delta encoding back to indices
    pub fn decode(&self) -> Vec<usize> {
        if self.total_elements == 0 {
            return vec![];
        }

        let mut indices = Vec::with_capacity(self.total_elements);
        indices.push(self.base_index);

        let mut current = self.base_index as i64;
        for &delta in &self.deltas {
            current += delta as i64;
            indices.push(current as usize);
        }

        indices
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        if self.total_elements == 0 {
            return 1.0;
        }

        let original_size = self.total_elements * std::mem::size_of::<usize>();
        let compressed_size =
            std::mem::size_of::<usize>() + self.deltas.len() * std::mem::size_of::<i32>();

        original_size as f32 / compressed_size as f32
    }
}

/// Bitmap encoding for dense index regions
#[derive(Debug, Clone)]
pub struct BitmapEncoded {
    /// Starting index of bitmap
    start_index: usize,

    /// Bitmap (1 bit per element)
    bitmap: Vec<u64>,

    /// Number of elements in bitmap
    num_elements: usize,

    /// Number of set bits
    num_set_bits: usize,
}

impl BitmapEncoded {
    /// Create bitmap encoding from indices within a range
    pub fn encode(indices: &[usize], start: usize, end: usize) -> Self {
        let num_elements = end - start;
        let num_words = (num_elements + 63) / 64;
        let mut bitmap = vec![0u64; num_words];
        let mut num_set_bits = 0;

        for &idx in indices {
            if idx >= start && idx < end {
                let bit_pos = idx - start;
                let word_idx = bit_pos / 64;
                let bit_idx = bit_pos % 64;
                bitmap[word_idx] |= 1u64 << bit_idx;
                num_set_bits += 1;
            }
        }

        Self {
            start_index: start,
            bitmap,
            num_elements,
            num_set_bits,
        }
    }

    /// Decode bitmap back to indices
    pub fn decode(&self) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.num_set_bits);

        for (word_idx, &word) in self.bitmap.iter().enumerate() {
            if word == 0 {
                continue;
            }

            for bit_idx in 0..64 {
                if (word & (1u64 << bit_idx)) != 0 {
                    let idx = self.start_index + word_idx * 64 + bit_idx;
                    if idx < self.start_index + self.num_elements {
                        indices.push(idx);
                    }
                }
            }
        }

        indices
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        if self.num_set_bits == 0 {
            return 1.0;
        }

        let original_size = self.num_set_bits * std::mem::size_of::<usize>();
        let compressed_size =
            std::mem::size_of::<usize>() + self.bitmap.len() * std::mem::size_of::<u64>();

        original_size as f32 / compressed_size as f32
    }

    /// Get density (ratio of set bits to total bits)
    pub fn density(&self) -> f32 {
        self.num_set_bits as f32 / self.num_elements as f32
    }
}

/// Compression statistics and analysis
#[derive(Debug, Clone)]
pub struct CompressionAnalysis {
    /// Original size in bytes
    pub original_size: usize,

    /// Compressed size in bytes
    pub compressed_size: usize,

    /// Compression ratio
    pub compression_ratio: f32,

    /// Space savings in bytes
    pub space_savings: usize,

    /// Encoding used
    pub encoding: CompressionEncoding,

    /// Sparsity of data
    pub sparsity: f32,

    /// Compression efficiency score (0-100)
    pub efficiency_score: u8,
}

impl CompressionAnalysis {
    /// Create compression analysis
    pub fn new(
        original_size: usize,
        compressed_size: usize,
        encoding: CompressionEncoding,
        sparsity: f32,
    ) -> Self {
        let compression_ratio = if compressed_size > 0 {
            original_size as f32 / compressed_size as f32
        } else {
            1.0
        };

        let space_savings = original_size.saturating_sub(compressed_size);

        // Calculate efficiency score (how close to theoretical maximum)
        let theoretical_max = encoding.expected_compression_ratio();
        let efficiency_score = ((compression_ratio / theoretical_max) * 100.0).min(100.0) as u8;

        Self {
            original_size,
            compressed_size,
            compression_ratio,
            space_savings,
            encoding,
            sparsity,
            efficiency_score,
        }
    }

    /// Check if compression is beneficial
    pub fn is_beneficial(&self) -> bool {
        self.compression_ratio > 1.1 // At least 10% savings
    }

    /// Get space savings percentage
    pub fn savings_percentage(&self) -> f32 {
        (self.space_savings as f32 / self.original_size as f32) * 100.0
    }
}

/// Compression strategy selector based on data characteristics
#[derive(Debug, Clone)]
pub struct CompressionSelector {
    /// Sparsity threshold for using compression
    sparsity_threshold: f32,

    /// Preferred encoding methods in priority order
    preferred_encodings: Vec<CompressionEncoding>,
}

impl CompressionSelector {
    /// Create new compression selector
    pub fn new() -> Self {
        Self {
            sparsity_threshold: 0.3, // 30% sparsity minimum
            preferred_encodings: vec![
                CompressionEncoding::Hybrid,
                CompressionEncoding::Huffman,
                CompressionEncoding::Bitmap,
                CompressionEncoding::RunLength,
                CompressionEncoding::Delta,
            ],
        }
    }

    /// Set sparsity threshold
    pub fn with_sparsity_threshold(mut self, threshold: f32) -> Self {
        self.sparsity_threshold = threshold;
        self
    }

    /// Get preferred encodings
    pub fn preferred_encodings(&self) -> &[CompressionEncoding] {
        &self.preferred_encodings
    }

    /// Select best compression encoding for indices
    pub fn select_encoding(&self, indices: &[usize], total_size: usize) -> CompressionEncoding {
        if indices.is_empty() {
            return CompressionEncoding::Raw;
        }

        let sparsity = 1.0 - (indices.len() as f32 / total_size as f32);

        // Don't compress if below threshold
        if sparsity < self.sparsity_threshold {
            return CompressionEncoding::Raw;
        }

        // Check for consecutive patterns (good for RLE)
        let consecutive_ratio = self.calculate_consecutive_ratio(indices);
        if consecutive_ratio > 0.7 {
            return CompressionEncoding::RunLength;
        }

        // Check for small deltas (good for delta encoding)
        let avg_delta = self.calculate_average_delta(indices);
        if avg_delta < 10.0 {
            return CompressionEncoding::Delta;
        }

        // Check for dense regions (good for bitmap)
        if self.has_dense_regions(indices) {
            return CompressionEncoding::Bitmap;
        }

        // Default to hybrid for complex patterns
        CompressionEncoding::Hybrid
    }

    fn calculate_consecutive_ratio(&self, indices: &[usize]) -> f32 {
        if indices.len() < 2 {
            return 0.0;
        }

        let mut consecutive_count = 0;
        for i in 1..indices.len() {
            if indices[i] == indices[i - 1] + 1 {
                consecutive_count += 1;
            }
        }

        consecutive_count as f32 / (indices.len() - 1) as f32
    }

    fn calculate_average_delta(&self, indices: &[usize]) -> f32 {
        if indices.len() < 2 {
            return 0.0;
        }

        let mut total_delta = 0i64;
        for i in 1..indices.len() {
            total_delta += (indices[i] as i64 - indices[i - 1] as i64).abs();
        }

        total_delta as f32 / (indices.len() - 1) as f32
    }

    fn has_dense_regions(&self, indices: &[usize]) -> bool {
        if indices.len() < 10 {
            return false;
        }

        // Check if there's a region with >80% density
        let min_idx = *indices.iter().min().expect("reduction should succeed");
        let max_idx = *indices.iter().max().expect("reduction should succeed");
        let range = max_idx - min_idx + 1;

        if range == 0 {
            return false;
        }

        let density = indices.len() as f32 / range as f32;
        density > 0.8
    }
}

impl Default for CompressionSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Magnitude threshold calculator for pruning
#[derive(Debug, Clone)]
pub struct MagnitudeThresholdCalculator;

impl MagnitudeThresholdCalculator {
    /// Calculate threshold from percentile
    pub fn from_percentile(values: &[f32], percentile: u8) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted_values: Vec<f32> = values.iter().map(|v| v.abs()).collect();
        sorted_values.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("absolute values should be comparable (no NaN)")
        });

        let index = ((percentile as f32 / 100.0) * sorted_values.len() as f32) as usize;
        let index = index.min(sorted_values.len() - 1);

        sorted_values[index]
    }

    /// Calculate threshold from top-k selection
    pub fn from_top_k(values: &[f32], k: usize) -> f32 {
        if values.is_empty() || k == 0 {
            return 0.0;
        }

        let mut sorted_values: Vec<f32> = values.iter().map(|v| v.abs()).collect();
        sorted_values.sort_by(|a, b| {
            b.partial_cmp(a)
                .expect("absolute values should be comparable (no NaN)")
        });

        let k = k.min(sorted_values.len());
        sorted_values[k - 1]
    }

    /// Calculate threshold from standard deviation
    pub fn from_std_dev(values: &[f32], num_std_dev: f32) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();

        mean.abs() - num_std_dev * std_dev
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pruning_strategy() {
        let strategy = PruningStrategy::Magnitude {
            threshold_percentile: 50,
        };
        assert_eq!(strategy.expected_sparsity(), 0.5);
        assert!(!strategy.is_structured());

        let structured = PruningStrategy::BlockWise { block_size: (4, 4) };
        assert!(structured.is_structured());
    }

    #[test]
    fn test_run_length_encoding() {
        let indices = vec![0, 1, 2, 3, 10, 11, 12, 20];
        let encoded = RunLengthEncoded::encode(&indices);

        assert_eq!(encoded.num_runs(), 3);
        assert_eq!(encoded.decode(), indices);
        assert!(encoded.compression_ratio() > 1.0);
    }

    #[test]
    fn test_delta_encoding() {
        let indices = vec![5, 10, 15, 20, 25];
        let encoded = DeltaEncoded::encode(&indices);

        assert_eq!(encoded.decode(), indices);
        assert!(encoded.compression_ratio() > 1.0);
    }

    #[test]
    fn test_bitmap_encoding() {
        let indices = vec![0, 1, 3, 5, 7];
        let encoded = BitmapEncoded::encode(&indices, 0, 10);

        assert_eq!(encoded.num_set_bits, 5);
        assert_eq!(encoded.decode(), indices);
        assert_eq!(encoded.density(), 0.5);
    }

    #[test]
    fn test_compression_selector() {
        let selector = CompressionSelector::new();

        // Consecutive indices should use RLE
        let consecutive = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let encoding = selector.select_encoding(&consecutive, 100);
        assert_eq!(encoding, CompressionEncoding::RunLength);

        // Small deltas should use delta encoding
        let small_deltas = vec![0, 1, 3, 4, 6, 7, 9, 10];
        let encoding = selector.select_encoding(&small_deltas, 100);
        assert!(matches!(
            encoding,
            CompressionEncoding::Delta | CompressionEncoding::RunLength
        ));
    }

    #[test]
    fn test_magnitude_threshold() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // 50th percentile with 10 values at index 5 gives 6.0
        let threshold = MagnitudeThresholdCalculator::from_percentile(&values, 50);
        assert!((threshold - 6.0).abs() < 0.1);

        // Top-3 selection gets the 3rd largest value which is 8.0
        let threshold = MagnitudeThresholdCalculator::from_top_k(&values, 3);
        assert!((threshold - 8.0).abs() < 0.1);
    }

    #[test]
    fn test_pruning_metadata() {
        let shape = Shape::new(vec![10, 10]);
        let metadata = PruningMetadata::new(
            PruningStrategy::Magnitude {
                threshold_percentile: 50,
            },
            shape,
            0.5,
        )
        .with_indices(vec![0, 1, 2, 3, 4])
        .with_threshold(0.1);

        assert_eq!(metadata.sparsity(), 0.5);
        assert_eq!(metadata.compression_ratio(), 2.0);
        assert_eq!(metadata.num_pruned_elements(), 5);
        assert!(metadata.is_element_pruned(2));
        assert!(!metadata.is_element_pruned(10));
    }

    #[test]
    fn test_compression_analysis() {
        let analysis = CompressionAnalysis::new(1000, 250, CompressionEncoding::Huffman, 0.75);

        assert_eq!(analysis.compression_ratio, 4.0);
        assert_eq!(analysis.space_savings, 750);
        assert!(analysis.is_beneficial());
        assert_eq!(analysis.savings_percentage(), 75.0);
    }
}
