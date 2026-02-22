use crate::{OptimizerError, OptimizerResult};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// Configuration for sparse parameter updates
#[derive(Debug, Clone)]
pub struct SparseUpdateConfig {
    /// Sparsity threshold - gradients below this magnitude are considered zero
    pub sparsity_threshold: f32,
    /// Whether to use coordinate-wise sparsity detection
    pub coordinate_wise_sparsity: bool,
    /// Whether to track sparse patterns over time
    pub track_sparse_patterns: bool,
    /// Minimum sparsity ratio to trigger sparse updates (0.0 to 1.0)
    pub min_sparsity_ratio: f32,
    /// Whether to use compressed sparse representations
    pub use_compression: bool,
    /// Block size for block-sparse updates
    pub block_size: usize,
    /// Whether to use adaptive sparsity thresholds
    pub adaptive_threshold: bool,
}

impl Default for SparseUpdateConfig {
    fn default() -> Self {
        Self {
            sparsity_threshold: 1e-8,
            coordinate_wise_sparsity: true,
            track_sparse_patterns: true,
            min_sparsity_ratio: 0.1, // At least 10% sparse to use sparse representation
            use_compression: true,
            block_size: 32,
            adaptive_threshold: true,
        }
    }
}

/// Sparse gradient representation using Compressed Sparse Row (CSR) format
#[derive(Debug, Clone)]
pub struct SparseGradient {
    /// Non-zero values
    pub values: Vec<f32>,
    /// Column indices of non-zero values
    pub indices: Vec<usize>,
    /// Row pointers (for 2D gradients)
    pub row_ptr: Vec<usize>,
    /// Shape of the original dense gradient
    pub shape: Vec<usize>,
    /// Total number of elements
    pub total_elements: usize,
    /// Sparsity ratio (0.0 = dense, 1.0 = completely sparse)
    pub sparsity_ratio: f32,
}

impl SparseGradient {
    /// Create a sparse gradient from a dense gradient
    pub fn from_dense(gradient: &[f32], shape: Vec<usize>, threshold: f32) -> Self {
        let mut values = Vec::new();
        let mut indices = Vec::new();

        for (i, &val) in gradient.iter().enumerate() {
            if val.abs() > threshold {
                values.push(val);
                indices.push(i);
            }
        }

        let total_elements = gradient.len();
        let sparsity_ratio = 1.0 - (values.len() as f32 / total_elements as f32);

        // For 1D gradients, create a simple row pointer
        let row_ptr = if shape.len() == 1 {
            vec![0, values.len()]
        } else {
            // For multi-dimensional gradients, compute proper row pointers
            Self::compute_row_pointers(&indices, &shape)
        };

        Self {
            values,
            indices,
            row_ptr,
            shape,
            total_elements,
            sparsity_ratio,
        }
    }

    /// Convert back to dense gradient
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0.0; self.total_elements];

        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            dense[idx] = val;
        }

        dense
    }

    /// Check if this gradient is sparse enough to benefit from sparse representation
    pub fn is_worth_sparse(&self, min_sparsity_ratio: f32) -> bool {
        self.sparsity_ratio >= min_sparsity_ratio
    }

    /// Get memory footprint in bytes
    pub fn memory_footprint(&self) -> usize {
        self.values.len() * 4 + // f32 values
        self.indices.len() * 8 + // usize indices  
        self.row_ptr.len() * 8 + // usize row pointers
        self.shape.len() * 8 + // usize shape
        24 // other fields
    }

    /// Get compression ratio compared to dense representation
    pub fn compression_ratio(&self) -> f32 {
        let dense_size = self.total_elements * 4; // 4 bytes per f32
        let sparse_size = self.memory_footprint();
        dense_size as f32 / sparse_size as f32
    }

    /// Add another sparse gradient (element-wise addition)
    pub fn add(&mut self, other: &SparseGradient) -> Result<(), OptimizerError> {
        if self.shape != other.shape {
            return Err(OptimizerError::InvalidParameter(
                "Cannot add sparse gradients with different shapes".to_string(),
            ));
        }

        // Convert both to dense, add, then convert back to sparse
        let mut dense_self = self.to_dense();
        let dense_other = other.to_dense();

        for (a, b) in dense_self.iter_mut().zip(dense_other.iter()) {
            *a += b;
        }

        // Update self with the result
        let threshold = self
            .values
            .iter()
            .chain(other.values.iter())
            .map(|&x| x.abs())
            .fold(0.0f32, |acc, x| acc.max(x))
            * 1e-6;

        let result = Self::from_dense(&dense_self, self.shape.clone(), threshold);

        self.values = result.values;
        self.indices = result.indices;
        self.row_ptr = result.row_ptr;
        self.sparsity_ratio = result.sparsity_ratio;

        Ok(())
    }

    /// Scale the sparse gradient by a scalar
    pub fn scale(&mut self, factor: f32) {
        for val in &mut self.values {
            *val *= factor;
        }
    }

    /// Compute L2 norm of the sparse gradient
    pub fn norm(&self) -> f32 {
        self.values.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    // Private helper methods

    fn compute_row_pointers(indices: &[usize], shape: &[usize]) -> Vec<usize> {
        if shape.len() != 2 {
            // For non-2D shapes, return simple row pointer
            return vec![0, indices.len()];
        }

        let rows = shape[0];
        let cols = shape[1];
        let mut row_ptr = vec![0; rows + 1];

        for &idx in indices {
            let row = idx / cols;
            if row < rows {
                row_ptr[row + 1] += 1;
            }
        }

        // Convert counts to cumulative sums
        for i in 1..row_ptr.len() {
            row_ptr[i] += row_ptr[i - 1];
        }

        row_ptr
    }
}

/// Block-sparse gradient for structured sparsity
#[derive(Debug, Clone)]
pub struct BlockSparseGradient {
    /// Non-zero blocks
    pub blocks: Vec<Vec<f32>>,
    /// Block indices (row, col) for 2D or flat index for 1D
    pub block_indices: Vec<(usize, usize)>,
    /// Block size
    pub block_size: usize,
    /// Shape of the original gradient
    pub shape: Vec<usize>,
    /// Number of blocks in each dimension
    pub block_shape: Vec<usize>,
    /// Sparsity ratio at block level
    pub block_sparsity_ratio: f32,
}

impl BlockSparseGradient {
    /// Create block-sparse gradient from dense gradient
    pub fn from_dense(
        gradient: &[f32],
        shape: Vec<usize>,
        block_size: usize,
        threshold: f32,
    ) -> Self {
        match shape.len() {
            1 => Self::from_dense_1d(gradient, shape, block_size, threshold),
            2 => Self::from_dense_2d(gradient, shape, block_size, threshold),
            _ => panic!("Unsupported gradient dimensionality for block-sparse representation"),
        }
    }

    /// Convert back to dense gradient
    pub fn to_dense(&self) -> Vec<f32> {
        let total_elements: usize = self.shape.iter().product();
        let mut dense = vec![0.0; total_elements];

        for (block, &(block_row, block_col)) in self.blocks.iter().zip(self.block_indices.iter()) {
            if self.shape.len() == 1 {
                let start_idx = block_row * self.block_size;
                for (i, &val) in block.iter().enumerate() {
                    if start_idx + i < dense.len() {
                        dense[start_idx + i] = val;
                    }
                }
            } else if self.shape.len() == 2 {
                let rows = self.shape[0];
                let cols = self.shape[1];
                let start_row = block_row * self.block_size;
                let start_col = block_col * self.block_size;

                for (i, &val) in block.iter().enumerate() {
                    let row = start_row + i / self.block_size;
                    let col = start_col + i % self.block_size;
                    if row < rows && col < cols {
                        dense[row * cols + col] = val;
                    }
                }
            }
        }

        dense
    }

    /// Check if block representation is beneficial
    pub fn is_worth_block_sparse(&self, min_sparsity_ratio: f32) -> bool {
        self.block_sparsity_ratio >= min_sparsity_ratio
    }

    // Private helper methods

    fn from_dense_1d(
        gradient: &[f32],
        shape: Vec<usize>,
        block_size: usize,
        threshold: f32,
    ) -> Self {
        let total_elements = shape[0];
        let num_blocks = (total_elements + block_size - 1) / block_size;

        let mut blocks = Vec::new();
        let mut block_indices = Vec::new();

        for block_idx in 0..num_blocks {
            let start_idx = block_idx * block_size;
            let end_idx = (start_idx + block_size).min(total_elements);

            let block_data: Vec<f32> = gradient[start_idx..end_idx].to_vec();

            // Check if block has significant values
            let block_norm = block_data.iter().map(|&x| x * x).sum::<f32>().sqrt();

            if block_norm > threshold {
                blocks.push(block_data);
                block_indices.push((block_idx, 0));
            }
        }

        let block_sparsity_ratio = 1.0 - (blocks.len() as f32 / num_blocks as f32);

        Self {
            blocks,
            block_indices,
            block_size,
            shape,
            block_shape: vec![num_blocks],
            block_sparsity_ratio,
        }
    }

    fn from_dense_2d(
        gradient: &[f32],
        shape: Vec<usize>,
        block_size: usize,
        threshold: f32,
    ) -> Self {
        let rows = shape[0];
        let cols = shape[1];
        let block_rows = (rows + block_size - 1) / block_size;
        let block_cols = (cols + block_size - 1) / block_size;

        let mut blocks = Vec::new();
        let mut block_indices = Vec::new();

        for block_row in 0..block_rows {
            for block_col in 0..block_cols {
                let mut block_data = Vec::new();

                let start_row = block_row * block_size;
                let end_row = (start_row + block_size).min(rows);
                let start_col = block_col * block_size;
                let end_col = (start_col + block_size).min(cols);

                for row in start_row..end_row {
                    for col in start_col..end_col {
                        let idx = row * cols + col;
                        block_data.push(gradient[idx]);
                    }
                }

                // Check if block has significant values
                let block_norm = block_data.iter().map(|&x| x * x).sum::<f32>().sqrt();

                if block_norm > threshold {
                    blocks.push(block_data);
                    block_indices.push((block_row, block_col));
                }
            }
        }

        let total_blocks = block_rows * block_cols;
        let block_sparsity_ratio = 1.0 - (blocks.len() as f32 / total_blocks as f32);

        Self {
            blocks,
            block_indices,
            block_size,
            shape,
            block_shape: vec![block_rows, block_cols],
            block_sparsity_ratio,
        }
    }
}

/// Sparse pattern tracker for analyzing sparsity patterns over time
#[derive(Debug, Clone)]
pub struct SparsePatternTracker {
    /// Parameter ID
    pub parameter_id: String,
    /// Historical sparsity patterns (indices of non-zero elements)
    pub pattern_history: VecDeque<HashSet<usize>>,
    /// Maximum history length
    pub max_history: usize,
    /// Stable sparse indices (consistently sparse)
    pub stable_sparse_indices: HashSet<usize>,
    /// Stable dense indices (consistently non-zero)
    pub stable_dense_indices: HashSet<usize>,
    /// Stability threshold (fraction of history where pattern must be consistent)
    pub stability_threshold: f32,
}

use std::collections::VecDeque;

impl SparsePatternTracker {
    /// Create a new pattern tracker
    pub fn new(parameter_id: String, max_history: usize, stability_threshold: f32) -> Self {
        Self {
            parameter_id,
            pattern_history: VecDeque::with_capacity(max_history),
            max_history,
            stable_sparse_indices: HashSet::new(),
            stable_dense_indices: HashSet::new(),
            stability_threshold,
        }
    }

    /// Update with a new sparse pattern
    pub fn update(&mut self, sparse_gradient: &SparseGradient) {
        let non_zero_indices: HashSet<usize> = sparse_gradient.indices.iter().cloned().collect();

        // Add to history
        self.pattern_history.push_back(non_zero_indices);

        // Remove old history if necessary
        if self.pattern_history.len() > self.max_history {
            self.pattern_history.pop_front();
        }

        // Update stable patterns if we have enough history
        if self.pattern_history.len() >= (self.max_history as f32 * 0.5) as usize {
            self.update_stable_patterns(sparse_gradient.total_elements);
        }
    }

    /// Get sparsity statistics
    pub fn get_statistics(&self) -> SparsePatternStatistics {
        let total_patterns = self.pattern_history.len();

        let average_sparsity = if total_patterns > 0 {
            let total_sparse: usize = self
                .pattern_history
                .iter()
                .map(|pattern| pattern.len())
                .sum();
            total_sparse as f32 / total_patterns as f32
        } else {
            0.0
        };

        let pattern_stability = if total_patterns > 1 {
            // Measure how much patterns change between consecutive updates
            let mut stability_sum = 0.0;
            for i in 1..total_patterns {
                let prev = &self.pattern_history[i - 1];
                let curr = &self.pattern_history[i];
                let intersection = prev.intersection(curr).count();
                let union = prev.union(curr).count();
                stability_sum += intersection as f32 / union.max(1) as f32;
            }
            stability_sum / (total_patterns - 1) as f32
        } else {
            1.0
        };

        SparsePatternStatistics {
            parameter_id: self.parameter_id.clone(),
            average_sparsity,
            pattern_stability,
            stable_sparse_count: self.stable_sparse_indices.len(),
            stable_dense_count: self.stable_dense_indices.len(),
            total_patterns: total_patterns,
        }
    }

    /// Check if an index is consistently sparse
    pub fn is_consistently_sparse(&self, index: usize) -> bool {
        self.stable_sparse_indices.contains(&index)
    }

    /// Check if an index is consistently dense
    pub fn is_consistently_dense(&self, index: usize) -> bool {
        self.stable_dense_indices.contains(&index)
    }

    // Private methods

    fn update_stable_patterns(&mut self, total_elements: usize) {
        let history_len = self.pattern_history.len();
        let required_count = (history_len as f32 * self.stability_threshold) as usize;

        // Count how often each index appears as non-zero
        let mut index_counts: HashMap<usize, usize> = HashMap::new();
        for pattern in &self.pattern_history {
            for &idx in pattern {
                *index_counts.entry(idx).or_insert(0) += 1;
            }
        }

        // Update stable patterns
        self.stable_dense_indices.clear();
        self.stable_sparse_indices.clear();

        for idx in 0..total_elements {
            let count = index_counts.get(&idx).copied().unwrap_or(0);

            if count >= required_count {
                self.stable_dense_indices.insert(idx);
            } else if count == 0 && history_len >= required_count {
                self.stable_sparse_indices.insert(idx);
            }
        }
    }
}

/// Statistics about sparse patterns
#[derive(Debug, Clone)]
pub struct SparsePatternStatistics {
    pub parameter_id: String,
    pub average_sparsity: f32,
    pub pattern_stability: f32,
    pub stable_sparse_count: usize,
    pub stable_dense_count: usize,
    pub total_patterns: usize,
}

impl std::fmt::Display for SparsePatternStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Sparse Pattern Statistics for {}:", self.parameter_id)?;
        writeln!(
            f,
            "  Average Sparsity: {:.1}%",
            (1.0 - self.average_sparsity) * 100.0
        )?;
        writeln!(f, "  Pattern Stability: {:.3}", self.pattern_stability)?;
        writeln!(f, "  Stable Sparse: {}", self.stable_sparse_count)?;
        writeln!(f, "  Stable Dense: {}", self.stable_dense_count)?;
        writeln!(f, "  Total Patterns: {}", self.total_patterns)?;
        Ok(())
    }
}

/// Sparse update manager
pub struct SparseUpdateManager {
    config: SparseUpdateConfig,
    pattern_trackers: HashMap<String, SparsePatternTracker>,
    adaptive_thresholds: HashMap<String, f32>,
    compression_stats: HashMap<String, CompressionStats>,
}

#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub total_updates: usize,
    pub sparse_updates: usize,
    pub average_compression_ratio: f32,
    pub memory_saved_bytes: usize,
}

impl SparseUpdateManager {
    /// Create a new sparse update manager
    pub fn new(config: SparseUpdateConfig) -> Self {
        Self {
            config,
            pattern_trackers: HashMap::new(),
            adaptive_thresholds: HashMap::new(),
            compression_stats: HashMap::new(),
        }
    }

    /// Process a gradient and return sparse representation if beneficial
    pub fn process_gradient(
        &mut self,
        parameter_id: String,
        gradient: Vec<f32>,
        shape: Vec<usize>,
    ) -> SparseUpdateResult {
        let threshold = self.get_threshold(&parameter_id, &gradient);

        // Create sparse representation
        let sparse_gradient = SparseGradient::from_dense(&gradient, shape.clone(), threshold);

        // Update pattern tracking if enabled
        if self.config.track_sparse_patterns {
            let tracker = self
                .pattern_trackers
                .entry(parameter_id.clone())
                .or_insert_with(|| SparsePatternTracker::new(parameter_id.clone(), 100, 0.8));
            tracker.update(&sparse_gradient);
        }

        // Update compression statistics
        self.update_compression_stats(&parameter_id, &sparse_gradient, gradient.len());

        // Decide whether to use sparse representation
        if sparse_gradient.is_worth_sparse(self.config.min_sparsity_ratio) {
            if self.config.use_compression {
                SparseUpdateResult::Sparse(sparse_gradient)
            } else {
                SparseUpdateResult::Dense(gradient)
            }
        } else {
            SparseUpdateResult::Dense(gradient)
        }
    }

    /// Process gradient with block-sparse representation
    pub fn process_gradient_block_sparse(
        &mut self,
        parameter_id: String,
        gradient: Vec<f32>,
        shape: Vec<usize>,
    ) -> SparseUpdateResult {
        let threshold = self.get_threshold(&parameter_id, &gradient);

        // Create block-sparse representation
        let block_sparse =
            BlockSparseGradient::from_dense(&gradient, shape, self.config.block_size, threshold);

        if block_sparse.is_worth_block_sparse(self.config.min_sparsity_ratio) {
            SparseUpdateResult::BlockSparse(block_sparse)
        } else {
            SparseUpdateResult::Dense(gradient)
        }
    }

    /// Get sparsity statistics for a parameter
    pub fn get_parameter_statistics(&self, parameter_id: &str) -> Option<SparsePatternStatistics> {
        self.pattern_trackers
            .get(parameter_id)
            .map(|tracker| tracker.get_statistics())
    }

    /// Get all compression statistics
    pub fn get_compression_statistics(&self) -> &HashMap<String, CompressionStats> {
        &self.compression_stats
    }

    /// Set custom threshold for a parameter
    pub fn set_parameter_threshold(&mut self, parameter_id: String, threshold: f32) {
        self.adaptive_thresholds.insert(parameter_id, threshold);
    }

    /// Get overall statistics
    pub fn get_overall_statistics(&self) -> OverallSparseStatistics {
        let total_parameters = self.compression_stats.len();

        let total_updates: usize = self
            .compression_stats
            .values()
            .map(|stats| stats.total_updates)
            .sum();

        let total_sparse_updates: usize = self
            .compression_stats
            .values()
            .map(|stats| stats.sparse_updates)
            .sum();

        let average_compression_ratio = if total_parameters > 0 {
            self.compression_stats
                .values()
                .map(|stats| stats.average_compression_ratio)
                .sum::<f32>()
                / total_parameters as f32
        } else {
            1.0
        };

        let total_memory_saved: usize = self
            .compression_stats
            .values()
            .map(|stats| stats.memory_saved_bytes)
            .sum();

        let sparse_ratio = if total_updates > 0 {
            total_sparse_updates as f32 / total_updates as f32
        } else {
            0.0
        };

        OverallSparseStatistics {
            total_parameters,
            total_updates,
            total_sparse_updates,
            sparse_ratio,
            average_compression_ratio,
            total_memory_saved,
        }
    }

    // Private methods

    fn get_threshold(&self, parameter_id: &str, gradient: &[f32]) -> f32 {
        if let Some(&custom_threshold) = self.adaptive_thresholds.get(parameter_id) {
            return custom_threshold;
        }

        if !self.config.adaptive_threshold {
            return self.config.sparsity_threshold;
        }

        // Adaptive threshold based on gradient statistics
        let gradient_magnitude = gradient.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let gradient_std = {
            let mean = gradient.iter().sum::<f32>() / gradient.len() as f32;
            let variance =
                gradient.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / gradient.len() as f32;
            variance.sqrt()
        };

        // Use a fraction of the standard deviation as threshold
        let adaptive_threshold = gradient_std * 0.1;
        adaptive_threshold
            .max(self.config.sparsity_threshold)
            .min(gradient_magnitude * 0.01)
    }

    fn update_compression_stats(
        &mut self,
        parameter_id: &str,
        sparse_gradient: &SparseGradient,
        original_size: usize,
    ) {
        let stats = self
            .compression_stats
            .entry(parameter_id.to_string())
            .or_insert_with(|| CompressionStats {
                total_updates: 0,
                sparse_updates: 0,
                average_compression_ratio: 1.0,
                memory_saved_bytes: 0,
            });

        stats.total_updates += 1;

        if sparse_gradient.is_worth_sparse(self.config.min_sparsity_ratio) {
            stats.sparse_updates += 1;

            let compression_ratio = sparse_gradient.compression_ratio();
            stats.average_compression_ratio = (stats.average_compression_ratio
                * (stats.sparse_updates - 1) as f32
                + compression_ratio)
                / stats.sparse_updates as f32;

            let original_bytes = original_size * 4; // 4 bytes per f32
            let compressed_bytes = sparse_gradient.memory_footprint();
            stats.memory_saved_bytes += original_bytes.saturating_sub(compressed_bytes);
        }
    }
}

/// Result of sparse update processing
#[derive(Debug)]
pub enum SparseUpdateResult {
    /// Use dense representation
    Dense(Vec<f32>),
    /// Use sparse representation
    Sparse(SparseGradient),
    /// Use block-sparse representation
    BlockSparse(BlockSparseGradient),
}

/// Overall sparse update statistics
#[derive(Debug, Clone)]
pub struct OverallSparseStatistics {
    pub total_parameters: usize,
    pub total_updates: usize,
    pub total_sparse_updates: usize,
    pub sparse_ratio: f32,
    pub average_compression_ratio: f32,
    pub total_memory_saved: usize,
}

impl std::fmt::Display for OverallSparseStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Overall Sparse Update Statistics:")?;
        writeln!(f, "  Total Parameters: {}", self.total_parameters)?;
        writeln!(f, "  Total Updates: {}", self.total_updates)?;
        writeln!(f, "  Sparse Updates: {}", self.total_sparse_updates)?;
        writeln!(f, "  Sparse Ratio: {:.1}%", self.sparse_ratio * 100.0)?;
        writeln!(
            f,
            "  Average Compression: {:.2}x",
            self.average_compression_ratio
        )?;
        writeln!(
            f,
            "  Memory Saved: {:.2} MB",
            self.total_memory_saved as f64 / 1024.0 / 1024.0
        )?;
        Ok(())
    }
}

/// Trait for optimizers that support sparse updates
pub trait SparseUpdateSupport {
    /// Apply a sparse gradient update
    fn apply_sparse_update(
        &mut self,
        parameter_id: &str,
        sparse_gradient: &SparseGradient,
    ) -> Result<(), OptimizerError>;

    /// Apply a block-sparse gradient update
    fn apply_block_sparse_update(
        &mut self,
        parameter_id: &str,
        block_sparse: &BlockSparseGradient,
    ) -> Result<(), OptimizerError>;

    /// Apply a dense gradient update
    fn apply_dense_update(
        &mut self,
        parameter_id: &str,
        gradient: &[f32],
    ) -> Result<(), OptimizerError>;

    /// Get parameter shape for sparse processing
    fn get_parameter_shape(&self, parameter_id: &str) -> Option<Vec<usize>>;
}

/// Wrapper optimizer that adds sparse update functionality
pub struct SparseUpdateOptimizer<T> {
    inner: T,
    sparse_manager: SparseUpdateManager,
    enabled: bool,
}

impl<T> SparseUpdateOptimizer<T>
where
    T: SparseUpdateSupport,
{
    /// Create a new sparse update optimizer wrapper
    pub fn new(inner: T, config: SparseUpdateConfig) -> Self {
        Self {
            inner,
            sparse_manager: SparseUpdateManager::new(config),
            enabled: true,
        }
    }

    /// Enable or disable sparse updates
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get the inner optimizer
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Get the inner optimizer mutably
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Get the sparse manager
    pub fn sparse_manager(&self) -> &SparseUpdateManager {
        &self.sparse_manager
    }

    /// Submit gradients for sparse processing
    pub fn submit_gradients(
        &mut self,
        gradients: HashMap<String, Vec<f32>>,
    ) -> Result<(), OptimizerError> {
        for (parameter_id, gradient) in gradients {
            if !self.enabled {
                self.inner.apply_dense_update(&parameter_id, &gradient)?;
                continue;
            }

            // Get parameter shape
            let shape = self
                .inner
                .get_parameter_shape(&parameter_id)
                .unwrap_or_else(|| vec![gradient.len()]);

            // Process gradient for sparsity
            match self
                .sparse_manager
                .process_gradient(parameter_id.clone(), gradient, shape)
            {
                SparseUpdateResult::Dense(dense_gradient) => {
                    self.inner
                        .apply_dense_update(&parameter_id, &dense_gradient)?;
                }
                SparseUpdateResult::Sparse(sparse_gradient) => {
                    self.inner
                        .apply_sparse_update(&parameter_id, &sparse_gradient)?;
                }
                SparseUpdateResult::BlockSparse(block_sparse) => {
                    self.inner
                        .apply_block_sparse_update(&parameter_id, &block_sparse)?;
                }
            }
        }

        Ok(())
    }

    /// Get statistics for a specific parameter
    pub fn get_parameter_statistics(&self, parameter_id: &str) -> Option<SparsePatternStatistics> {
        self.sparse_manager.get_parameter_statistics(parameter_id)
    }

    /// Get overall statistics
    pub fn get_statistics(&self) -> OverallSparseStatistics {
        self.sparse_manager.get_overall_statistics()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_gradient() {
        let dense = vec![0.0, 1.5, 0.0, 0.0, 2.3, 0.0, -1.1, 0.0];
        let shape = vec![8];
        let threshold = 0.1;

        let sparse = SparseGradient::from_dense(&dense, shape, threshold);

        assert_eq!(sparse.values, vec![1.5, 2.3, -1.1]);
        assert_eq!(sparse.indices, vec![1, 4, 6]);
        assert!(sparse.sparsity_ratio > 0.5);

        let recovered = sparse.to_dense();
        assert_eq!(recovered, dense);
    }

    #[test]
    fn test_block_sparse_gradient() {
        let dense = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5, 1.5, 0.0, 0.0, 0.5, 0.8,
        ];
        let shape = vec![4, 4];
        let block_size = 2;
        let threshold = 0.1;

        let block_sparse = BlockSparseGradient::from_dense(&dense, shape, block_size, threshold);

        // Should have 2 non-zero blocks
        assert_eq!(block_sparse.blocks.len(), 2);
        assert!(block_sparse.block_sparsity_ratio > 0.0);

        let recovered = block_sparse.to_dense();
        // Check that significant values are preserved
        assert!((recovered[0] - 1.0).abs() < 1e-6);
        assert!((recovered[10] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_pattern_tracker() {
        let mut tracker = SparsePatternTracker::new("test_param".to_string(), 5, 0.8);

        // Simulate consistent pattern
        for _ in 0..10 {
            let dense = vec![1.0, 0.0, 2.0, 0.0, 0.0];
            let sparse = SparseGradient::from_dense(&dense, vec![5], 0.1);
            tracker.update(&sparse);
        }

        let stats = tracker.get_statistics();
        assert!(stats.pattern_stability > 0.5);
        assert_eq!(stats.stable_dense_count, 2); // indices 0 and 2
        assert_eq!(stats.stable_sparse_count, 3); // indices 1, 3, and 4
    }

    #[derive(Debug)]
    struct MockOptimizer {
        updates_received: HashMap<String, usize>,
    }

    impl MockOptimizer {
        fn new() -> Self {
            Self {
                updates_received: HashMap::new(),
            }
        }
    }

    impl SparseUpdateSupport for MockOptimizer {
        fn apply_sparse_update(
            &mut self,
            parameter_id: &str,
            _sparse_gradient: &SparseGradient,
        ) -> Result<(), OptimizerError> {
            *self
                .updates_received
                .entry(parameter_id.to_string())
                .or_insert(0) += 1;
            Ok(())
        }

        fn apply_block_sparse_update(
            &mut self,
            parameter_id: &str,
            _block_sparse: &BlockSparseGradient,
        ) -> Result<(), OptimizerError> {
            *self
                .updates_received
                .entry(parameter_id.to_string())
                .or_insert(0) += 1;
            Ok(())
        }

        fn apply_dense_update(
            &mut self,
            parameter_id: &str,
            _gradient: &[f32],
        ) -> Result<(), OptimizerError> {
            *self
                .updates_received
                .entry(parameter_id.to_string())
                .or_insert(0) += 1;
            Ok(())
        }

        fn get_parameter_shape(&self, _parameter_id: &str) -> Option<Vec<usize>> {
            Some(vec![8])
        }
    }

    #[test]
    fn test_sparse_update_optimizer() {
        let config = SparseUpdateConfig {
            sparsity_threshold: 0.1,
            min_sparsity_ratio: 0.3,
            ..Default::default()
        };

        let optimizer = MockOptimizer::new();
        let mut sparse_optimizer = SparseUpdateOptimizer::new(optimizer, config);

        // Submit gradients
        let mut gradients = HashMap::new();
        gradients.insert(
            "param1".to_string(),
            vec![0.0, 1.5, 0.0, 0.0, 2.3, 0.0, -1.1, 0.0],
        ); // Sparse
        gradients.insert(
            "param2".to_string(),
            vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
        ); // Dense

        sparse_optimizer.submit_gradients(gradients).unwrap();

        // Check that updates were received
        assert_eq!(
            sparse_optimizer.inner().updates_received.get("param1"),
            Some(&1)
        );
        assert_eq!(
            sparse_optimizer.inner().updates_received.get("param2"),
            Some(&1)
        );

        // Check statistics
        let stats = sparse_optimizer.get_statistics();
        assert_eq!(stats.total_parameters, 2);
        assert!(stats.total_updates > 0);
    }
}
