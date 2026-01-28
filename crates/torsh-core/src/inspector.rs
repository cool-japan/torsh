//! Tensor Inspector
//!
//! A comprehensive debugging and visualization tool for tensors that provides
//! detailed memory layout information, statistics, and debugging utilities.

use crate::error::Result;
use crate::{DType, Device, Shape};
use std::collections::HashMap;
use std::fmt;

/// Comprehensive tensor inspection and debugging utility
pub struct TensorInspector {
    /// Statistics tracking for tensor analysis
    stats: InspectionStats,
    /// Configuration for inspection behavior
    config: InspectorConfig,
}

/// Configuration for tensor inspector behavior
#[derive(Debug, Clone)]
pub struct InspectorConfig {
    /// Maximum number of elements to display in data preview
    pub max_preview_elements: usize,
    /// Whether to show detailed memory layout
    pub show_memory_layout: bool,
    /// Whether to compute and show statistics
    pub show_statistics: bool,
    /// Whether to validate tensor properties
    pub validate_properties: bool,
    /// Precision for floating point display
    pub float_precision: usize,
}

impl Default for InspectorConfig {
    fn default() -> Self {
        Self {
            max_preview_elements: 20,
            show_memory_layout: true,
            show_statistics: true,
            validate_properties: true,
            float_precision: 6,
        }
    }
}

/// Statistics collected during tensor inspection
#[derive(Debug, Default)]
pub struct InspectionStats {
    /// Number of tensors inspected
    pub tensors_inspected: usize,
    /// Total elements inspected
    pub total_elements: usize,
    /// Memory usage breakdown by dtype
    pub memory_by_dtype: HashMap<String, usize>,
    /// Device usage statistics
    pub device_usage: HashMap<String, usize>,
}

/// Detailed inspection result for a tensor
#[derive(Debug)]
pub struct TensorInspection {
    /// Basic tensor properties
    pub properties: TensorProperties,
    /// Memory layout information
    pub memory_layout: MemoryLayout,
    /// Statistics (if computed)
    pub statistics: Option<TensorStatistics>,
    /// Validation results
    pub validation: ValidationResults,
    /// Data preview (limited elements)
    pub data_preview: DataPreview,
}

/// Basic tensor properties
#[derive(Debug)]
pub struct TensorProperties {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub device: String,
    pub total_elements: usize,
    pub memory_bytes: usize,
    pub is_contiguous: bool,
    pub ndim: usize,
}

/// Memory layout information
#[derive(Debug)]
pub struct MemoryLayout {
    pub strides: Vec<usize>,
    pub stride_bytes: Vec<usize>,
    pub offset: usize,
    pub alignment: usize,
    pub memory_efficiency: f32,
    pub layout_pattern: LayoutPattern,
    pub visual_representation: String,
    pub cache_behavior: CacheBehaviorAnalysis,
    pub memory_fragmentation: MemoryFragmentation,
}

/// Pattern of memory layout
#[derive(Debug, PartialEq)]
pub enum LayoutPattern {
    /// Row-major (C-style) layout
    RowMajor,
    /// Column-major (Fortran-style) layout
    ColumnMajor,
    /// Strided layout with custom pattern
    Strided,
    /// Broadcast layout with replicated dimensions
    Broadcast,
    /// Unknown or complex pattern
    Unknown,
}

/// Statistical information about tensor data
#[derive(Debug)]
pub struct TensorStatistics {
    pub min_value: f64,
    pub max_value: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub zero_count: usize,
    pub inf_count: usize,
    pub nan_count: usize,
    pub unique_values: Option<usize>,
}

/// Validation results for tensor properties
#[derive(Debug)]
pub struct ValidationResults {
    pub is_valid: bool,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
    pub recommendations: Vec<String>,
}

/// Preview of tensor data with smart sampling
#[derive(Debug)]
pub struct DataPreview {
    pub sample_elements: Vec<String>,
    pub sample_indices: Vec<Vec<usize>>,
    pub is_truncated: bool,
    pub total_shown: usize,
}

/// Cache behavior analysis for memory access patterns
#[derive(Debug)]
pub struct CacheBehaviorAnalysis {
    pub cache_line_utilization: f32,
    pub spatial_locality_score: f32,
    pub temporal_locality_score: f32,
    pub prefetch_efficiency: f32,
    pub cache_miss_probability: f32,
    pub access_pattern: AccessPattern,
}

/// Memory access pattern classification
#[derive(Debug, PartialEq)]
pub enum AccessPattern {
    /// Sequential access (good for cache)
    Sequential,
    /// Random access (poor for cache)
    Random,
    /// Strided access (moderate for cache)
    Strided { stride: usize },
    /// Block-wise access (good for cache)
    BlockWise { block_size: usize },
    /// Transpose-like access (poor for cache)
    Transpose,
    /// Unknown pattern
    Unknown,
}

/// Memory fragmentation analysis
#[derive(Debug)]
pub struct MemoryFragmentation {
    pub fragmentation_ratio: f32,
    pub wasted_bytes: usize,
    pub alignment_waste: usize,
    pub padding_bytes: usize,
    pub memory_holes: Vec<MemoryHole>,
}

/// Represents a hole in memory allocation
#[derive(Debug)]
pub struct MemoryHole {
    pub offset: usize,
    pub size: usize,
    pub reason: String,
}

/// Performance profiling information
#[derive(Debug)]
pub struct PerformanceProfile {
    pub operation_name: String,
    pub execution_time_ns: u64,
    pub memory_bandwidth_gbps: f32,
    pub cache_hit_rate: f32,
    pub cpu_utilization: f32,
    pub memory_peak_usage: usize,
    pub bottleneck_analysis: Vec<String>,
}

impl TensorInspector {
    /// Create a new tensor inspector with default configuration
    pub fn new() -> Self {
        Self {
            stats: InspectionStats::default(),
            config: InspectorConfig::default(),
        }
    }

    /// Create a new tensor inspector with custom configuration
    pub fn with_config(config: InspectorConfig) -> Self {
        Self {
            stats: InspectionStats::default(),
            config,
        }
    }

    /// Inspect a tensor and return detailed analysis
    pub fn inspect<T>(
        &mut self,
        shape: &Shape,
        dtype: DType,
        device: &dyn Device,
        data: Option<&[T]>,
    ) -> Result<TensorInspection>
    where
        T: fmt::Debug + Clone + 'static,
    {
        // Update statistics
        self.stats.tensors_inspected += 1;
        self.stats.total_elements += shape.numel();

        let dtype_str = format!("{dtype:?}");
        *self
            .stats
            .memory_by_dtype
            .entry(dtype_str.clone())
            .or_insert(0) += shape.numel() * dtype.size_bytes();

        let device_str = device.name().to_string();
        *self
            .stats
            .device_usage
            .entry(device_str.clone())
            .or_insert(0) += 1;

        // Build inspection result
        let properties = self.analyze_properties(shape, dtype, device)?;
        let memory_layout = self.analyze_memory_layout(shape)?;
        let statistics = if self.config.show_statistics && data.is_some() {
            Some(self.compute_statistics(data.expect("data should be Some after is_some check"))?)
        } else {
            None
        };
        let validation = self.validate_tensor(shape, dtype, device, data)?;
        let data_preview = self.create_data_preview(shape, data)?;

        Ok(TensorInspection {
            properties,
            memory_layout,
            statistics,
            validation,
            data_preview,
        })
    }

    /// Get cumulative inspection statistics
    pub fn get_stats(&self) -> &InspectionStats {
        &self.stats
    }

    /// Reset inspection statistics
    pub fn reset_stats(&mut self) {
        self.stats = InspectionStats::default();
    }

    /// Update inspector configuration
    pub fn update_config(&mut self, config: InspectorConfig) {
        self.config = config;
    }

    fn analyze_properties(
        &self,
        shape: &Shape,
        dtype: DType,
        device: &dyn Device,
    ) -> Result<TensorProperties> {
        Ok(TensorProperties {
            shape: shape.dims().to_vec(),
            dtype: format!("{dtype:?}"),
            device: device.name().to_string(),
            total_elements: shape.numel(),
            memory_bytes: shape.numel() * dtype.size_bytes(),
            is_contiguous: true, // Simplified - would need stride analysis
            ndim: shape.ndim(),
        })
    }

    fn analyze_memory_layout(&self, shape: &Shape) -> Result<MemoryLayout> {
        let dims = shape.dims();
        let strides = self.compute_strides(dims);
        let stride_bytes: Vec<usize> = strides.iter().map(|&s| s * 4).collect(); // Assuming f32

        let layout_pattern = self.detect_layout_pattern(&strides);
        let memory_efficiency = self.compute_memory_efficiency(dims, &strides);
        let visual_representation = self.create_visual_representation(dims, &strides);
        let cache_behavior = self.analyze_cache_behavior(dims, &strides);
        let memory_fragmentation = self.analyze_memory_fragmentation(dims, &strides);

        Ok(MemoryLayout {
            strides,
            stride_bytes,
            offset: 0,
            alignment: 32, // Typical SIMD alignment
            memory_efficiency,
            layout_pattern,
            visual_representation,
            cache_behavior,
            memory_fragmentation,
        })
    }

    fn compute_strides(&self, dims: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        strides
    }

    fn detect_layout_pattern(&self, strides: &[usize]) -> LayoutPattern {
        if strides.is_empty() {
            return LayoutPattern::Unknown;
        }

        // Check for row-major (strides decrease)
        let mut is_row_major = true;
        for i in 0..strides.len().saturating_sub(1) {
            if strides[i] < strides[i + 1] {
                is_row_major = false;
                break;
            }
        }

        if is_row_major {
            return LayoutPattern::RowMajor;
        }

        // Check for column-major (strides increase)
        let mut is_col_major = true;
        for i in 0..strides.len().saturating_sub(1) {
            if strides[i] > strides[i + 1] {
                is_col_major = false;
                break;
            }
        }

        if is_col_major {
            LayoutPattern::ColumnMajor
        } else {
            LayoutPattern::Strided
        }
    }

    fn compute_memory_efficiency(&self, dims: &[usize], strides: &[usize]) -> f32 {
        if dims.is_empty() || strides.is_empty() {
            return 1.0;
        }

        let total_elements: usize = dims.iter().product();
        let max_stride = strides.iter().max().unwrap_or(&1);
        let theoretical_min_memory = total_elements;
        let actual_memory_span = max_stride + 1;

        if actual_memory_span == 0 {
            1.0
        } else {
            (theoretical_min_memory as f32) / (actual_memory_span as f32)
        }
    }

    fn compute_statistics<T>(&self, data: &[T]) -> Result<TensorStatistics>
    where
        T: fmt::Debug + Clone + 'static,
    {
        // This is a simplified version that would need proper numeric trait bounds
        // For now, return placeholder statistics
        Ok(TensorStatistics {
            min_value: 0.0,
            max_value: 1.0,
            mean: 0.5,
            std_dev: 0.25,
            zero_count: 0,
            inf_count: 0,
            nan_count: 0,
            unique_values: Some(data.len().min(1000)),
        })
    }

    fn validate_tensor<T>(
        &self,
        shape: &Shape,
        dtype: DType,
        device: &dyn Device,
        data: Option<&[T]>,
    ) -> Result<ValidationResults>
    where
        T: fmt::Debug + Clone + 'static,
    {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        let mut recommendations = Vec::new();

        // Validate shape
        if shape.dims().is_empty() {
            warnings.push("Scalar tensor (empty shape)".to_string());
        }

        if shape.dims().contains(&0) {
            warnings.push("Tensor contains zero-sized dimensions".to_string());
        }

        if shape.numel() > 1_000_000_000 {
            warnings.push("Very large tensor (>1B elements) - consider chunking".to_string());
        }

        // Validate dtype
        if matches!(dtype, DType::F16 | DType::BF16) {
            recommendations.push("Consider using F32 for better numerical stability".to_string());
        }

        // Validate device compatibility
        match device.device_type() {
            crate::device::DeviceType::Cpu => {
                if shape.numel() > 10_000_000 {
                    recommendations
                        .push("Large tensor on CPU - consider GPU acceleration".to_string());
                }
            }
            crate::device::DeviceType::Cuda(_) => {
                if shape.numel() < 1000 {
                    recommendations
                        .push("Small tensor on GPU - CPU might be more efficient".to_string());
                }
            }
            _ => {}
        }

        // Validate data consistency
        if let Some(tensor_data) = data {
            if tensor_data.len() != shape.numel() {
                errors.push(format!(
                    "Data length ({}) doesn't match shape elements ({})",
                    tensor_data.len(),
                    shape.numel()
                ));
            }
        }

        let is_valid = errors.is_empty();

        Ok(ValidationResults {
            is_valid,
            warnings,
            errors,
            recommendations,
        })
    }

    fn create_data_preview<T>(&self, shape: &Shape, data: Option<&[T]>) -> Result<DataPreview>
    where
        T: fmt::Debug + Clone + 'static,
    {
        let Some(tensor_data) = data else {
            return Ok(DataPreview {
                sample_elements: vec!["<data not available>".to_string()],
                sample_indices: vec![],
                is_truncated: false,
                total_shown: 0,
            });
        };

        let max_elements = self.config.max_preview_elements;
        let total_elements = tensor_data.len();
        let is_truncated = total_elements > max_elements;
        let elements_to_show = total_elements.min(max_elements);

        let mut sample_elements = Vec::new();
        let mut sample_indices = Vec::new();

        if total_elements <= max_elements {
            // Show all elements
            for (i, element) in tensor_data.iter().enumerate().take(elements_to_show) {
                sample_elements.push(format!("{element:?}"));
                sample_indices.push(self.linear_to_multi_index(i, shape.dims()));
            }
        } else {
            // Smart sampling: corners, center, and some random points
            let dims = shape.dims();

            // Sample corners (first few and last few elements)
            let corner_count = (max_elements / 3).max(1);
            for (i, element) in tensor_data
                .iter()
                .enumerate()
                .take(corner_count.min(total_elements))
            {
                sample_elements.push(format!("{element:?}"));
                sample_indices.push(self.linear_to_multi_index(i, dims));
            }

            // Sample center region
            let center_idx = total_elements / 2;
            let center_count = (max_elements / 3).max(1);
            let start_center = center_idx.saturating_sub(center_count / 2);
            for i in 0..center_count {
                let idx = (start_center + i).min(total_elements - 1);
                if idx >= corner_count && idx < total_elements - corner_count {
                    let element = tensor_data[idx].clone();
                    sample_elements.push(format!("{element:?}"));
                    sample_indices.push(self.linear_to_multi_index(idx, dims));
                }
            }

            // Sample end
            let end_count = max_elements - sample_elements.len();
            let start_end = total_elements.saturating_sub(end_count);
            for (i, element) in tensor_data
                .iter()
                .enumerate()
                .take(total_elements)
                .skip(start_end)
            {
                if sample_elements.len() < max_elements {
                    sample_elements.push(format!("{element:?}"));
                    sample_indices.push(self.linear_to_multi_index(i, dims));
                }
            }
        }

        let total_shown = sample_elements.len();
        Ok(DataPreview {
            sample_elements,
            sample_indices,
            is_truncated,
            total_shown,
        })
    }

    fn linear_to_multi_index(&self, linear_idx: usize, dims: &[usize]) -> Vec<usize> {
        let mut indices = Vec::with_capacity(dims.len());
        let mut remaining = linear_idx;

        for &dim in dims.iter().rev() {
            if dim > 0 {
                indices.push(remaining % dim);
                remaining /= dim;
            } else {
                indices.push(0);
            }
        }

        indices.reverse();
        indices
    }

    /// Create a visual representation of memory layout
    fn create_visual_representation(&self, dims: &[usize], strides: &[usize]) -> String {
        let mut visual = String::new();

        visual.push_str("Memory Layout Visualization:\n");

        // Show dimension structure
        for (i, (dim, stride)) in dims.iter().zip(strides.iter()).enumerate() {
            let indent = "  ".repeat(i);
            visual.push_str(&format!("{indent}Dim {i}: size={dim}, stride={stride}\n"));
        }

        // Show access pattern visualization for 2D case
        if dims.len() == 2 {
            visual.push_str("\nAccess Pattern (2D):\n");
            let rows = dims[0].min(4);
            let cols = dims[1].min(8);

            for i in 0..rows {
                visual.push_str("  ");
                for j in 0..cols {
                    let offset = i * strides[0] + j * strides[1];
                    visual.push_str(&format!("{offset:4}"));
                }
                if dims[1] > 8 {
                    visual.push_str(" ...");
                }
                visual.push('\n');
            }
            if dims[0] > 4 {
                visual.push_str("  ...\n");
            }
        }

        // Show stride efficiency
        visual.push_str("\nStride Analysis:\n");
        for (i, &stride) in strides.iter().enumerate() {
            let is_unit = stride == 1;
            let efficiency = if i == strides.len() - 1 { "✓" } else { "○" };
            visual.push_str(&format!(
                "  Dim {i}: stride={stride} {}",
                if is_unit { "(contiguous)" } else { "(strided)" }
            ));
            visual.push_str(&format!(" {efficiency}\n"));
        }

        visual
    }

    /// Analyze cache behavior based on memory access patterns
    fn analyze_cache_behavior(&self, dims: &[usize], strides: &[usize]) -> CacheBehaviorAnalysis {
        let cache_line_size = 64; // Typical cache line size in bytes
        let element_size = 4; // Assuming f32
        let elements_per_cache_line = cache_line_size / element_size;

        // Calculate cache line utilization
        let min_stride = strides.iter().min().copied().unwrap_or(1);
        let cache_line_utilization = if min_stride == 1 {
            1.0 // Perfect utilization for contiguous access
        } else {
            (1.0 / min_stride as f32).min(1.0)
        };

        // Spatial locality score (higher is better)
        let spatial_locality_score = if min_stride == 1 {
            1.0 // Perfect spatial locality
        } else {
            (elements_per_cache_line as f32 / min_stride as f32).min(1.0)
        };

        // Temporal locality score (simplified heuristic)
        let temporal_locality_score = if dims.len() > 1 {
            0.5 // Moderate temporal locality for multi-dimensional
        } else {
            0.8 // Good temporal locality for 1D
        };

        // Prefetch efficiency
        let prefetch_efficiency = if strides.iter().all(|&s| s <= elements_per_cache_line) {
            0.9 // Hardware prefetcher will work well
        } else {
            0.3 // Hardware prefetcher less effective
        };

        // Cache miss probability
        let cache_miss_probability = 1.0 - cache_line_utilization * 0.8;

        // Determine access pattern
        let access_pattern = self.classify_access_pattern(dims, strides);

        CacheBehaviorAnalysis {
            cache_line_utilization,
            spatial_locality_score,
            temporal_locality_score,
            prefetch_efficiency,
            cache_miss_probability,
            access_pattern,
        }
    }

    /// Classify memory access pattern
    fn classify_access_pattern(&self, dims: &[usize], strides: &[usize]) -> AccessPattern {
        if strides.is_empty() {
            return AccessPattern::Unknown;
        }

        // Check for sequential access
        if strides.iter().min() == Some(&1) {
            return AccessPattern::Sequential;
        }

        // Check for uniform stride
        if strides.len() == 1 {
            return AccessPattern::Strided { stride: strides[0] };
        }

        // Check for transpose pattern
        if strides.len() == 2 {
            let is_transpose = strides[0] == 1 && strides[1] > dims[1];
            if is_transpose {
                return AccessPattern::Transpose;
            }
        }

        // Check for block-wise access
        if dims.len() >= 2 && strides[strides.len() - 1] == 1 {
            let block_size = dims[dims.len() - 1];
            return AccessPattern::BlockWise { block_size };
        }

        // Default to strided if we can't classify
        AccessPattern::Strided {
            stride: strides.iter().min().copied().unwrap_or(1),
        }
    }

    /// Analyze memory fragmentation
    fn analyze_memory_fragmentation(
        &self,
        dims: &[usize],
        strides: &[usize],
    ) -> MemoryFragmentation {
        let element_size = 4; // Assuming f32
        let total_elements: usize = dims.iter().product();
        let theoretical_min_bytes = total_elements * element_size;

        // Calculate actual memory span
        let max_stride = strides.iter().max().copied().unwrap_or(1);
        let max_dim_idx = strides.iter().position(|&s| s == max_stride).unwrap_or(0);
        let max_dim_size = dims.get(max_dim_idx).copied().unwrap_or(1);
        let actual_span = max_stride * max_dim_size;
        let actual_bytes = actual_span * element_size;

        // Calculate wasted bytes
        let wasted_bytes = actual_bytes.saturating_sub(theoretical_min_bytes);

        // Calculate alignment waste (assume 32-byte alignment)
        let alignment = 32;
        let alignment_waste = (alignment - (theoretical_min_bytes % alignment)) % alignment;

        // Calculate padding bytes (simplified)
        let padding_bytes = if strides.iter().any(|&s| s > 1) {
            wasted_bytes / 2 // Heuristic: half of waste is padding
        } else {
            0
        };

        // Calculate fragmentation ratio
        let fragmentation_ratio = if actual_bytes > 0 {
            wasted_bytes as f32 / actual_bytes as f32
        } else {
            0.0
        };

        // Identify memory holes
        let mut memory_holes = Vec::new();
        if wasted_bytes > 0 {
            memory_holes.push(MemoryHole {
                offset: theoretical_min_bytes,
                size: wasted_bytes,
                reason: "Strided access pattern".to_string(),
            });
        }

        MemoryFragmentation {
            fragmentation_ratio,
            wasted_bytes,
            alignment_waste,
            padding_bytes,
            memory_holes,
        }
    }

    /// Profile operation performance
    pub fn profile_operation<F, T>(
        &self,
        operation_name: &str,
        operation: F,
    ) -> (T, PerformanceProfile)
    where
        F: FnOnce() -> T,
    {
        let start = std::time::Instant::now();
        let result = operation();
        let duration = start.elapsed();

        // Create performance profile with basic metrics
        let profile = PerformanceProfile {
            operation_name: operation_name.to_string(),
            execution_time_ns: duration.as_nanos() as u64,
            memory_bandwidth_gbps: 0.0, // Would need actual memory access measurements
            cache_hit_rate: 0.0,        // Would need hardware counters
            cpu_utilization: 0.0,       // Would need system monitoring
            memory_peak_usage: 0,       // Would need memory tracking
            bottleneck_analysis: vec![
                "Enable hardware performance counters for detailed analysis".to_string()
            ],
        };

        (result, profile)
    }
}

impl Default for TensorInspector {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TensorInspection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Tensor Inspection ===")?;

        // Properties
        writeln!(f, "\nProperties:")?;
        writeln!(f, "  Shape: {:?}", self.properties.shape)?;
        writeln!(f, "  DType: {}", self.properties.dtype)?;
        writeln!(f, "  Device: {}", self.properties.device)?;
        writeln!(f, "  Elements: {}", self.properties.total_elements)?;
        writeln!(f, "  Memory: {} bytes", self.properties.memory_bytes)?;
        writeln!(f, "  Dimensions: {}", self.properties.ndim)?;
        writeln!(f, "  Contiguous: {}", self.properties.is_contiguous)?;

        // Memory Layout
        writeln!(f, "\nMemory Layout:")?;
        writeln!(f, "  Strides: {:?}", self.memory_layout.strides)?;
        writeln!(f, "  Pattern: {:?}", self.memory_layout.layout_pattern)?;
        writeln!(
            f,
            "  Efficiency: {:.2}%",
            self.memory_layout.memory_efficiency * 100.0
        )?;
        writeln!(f, "  Alignment: {} bytes", self.memory_layout.alignment)?;

        // Visual representation
        writeln!(f, "\n{}", self.memory_layout.visual_representation)?;

        // Cache behavior analysis
        writeln!(f, "\nCache Behavior:")?;
        writeln!(
            f,
            "  Cache line utilization: {:.2}%",
            self.memory_layout.cache_behavior.cache_line_utilization * 100.0
        )?;
        writeln!(
            f,
            "  Spatial locality: {:.2}/1.0",
            self.memory_layout.cache_behavior.spatial_locality_score
        )?;
        writeln!(
            f,
            "  Temporal locality: {:.2}/1.0",
            self.memory_layout.cache_behavior.temporal_locality_score
        )?;
        writeln!(
            f,
            "  Prefetch efficiency: {:.2}%",
            self.memory_layout.cache_behavior.prefetch_efficiency * 100.0
        )?;
        writeln!(
            f,
            "  Cache miss probability: {:.2}%",
            self.memory_layout.cache_behavior.cache_miss_probability * 100.0
        )?;
        writeln!(
            f,
            "  Access pattern: {:?}",
            self.memory_layout.cache_behavior.access_pattern
        )?;

        // Memory fragmentation
        writeln!(f, "\nMemory Fragmentation:")?;
        writeln!(
            f,
            "  Fragmentation ratio: {:.2}%",
            self.memory_layout.memory_fragmentation.fragmentation_ratio * 100.0
        )?;
        writeln!(
            f,
            "  Wasted bytes: {}",
            self.memory_layout.memory_fragmentation.wasted_bytes
        )?;
        writeln!(
            f,
            "  Alignment waste: {} bytes",
            self.memory_layout.memory_fragmentation.alignment_waste
        )?;
        writeln!(
            f,
            "  Padding bytes: {}",
            self.memory_layout.memory_fragmentation.padding_bytes
        )?;
        if !self
            .memory_layout
            .memory_fragmentation
            .memory_holes
            .is_empty()
        {
            writeln!(f, "  Memory holes:")?;
            for hole in &self.memory_layout.memory_fragmentation.memory_holes {
                writeln!(
                    f,
                    "    - Offset {}: {} bytes ({})",
                    hole.offset, hole.size, hole.reason
                )?;
            }
        }

        // Statistics
        if let Some(ref stats) = self.statistics {
            writeln!(f, "\nStatistics:")?;
            writeln!(f, "  Min: {:.6}", stats.min_value)?;
            writeln!(f, "  Max: {:.6}", stats.max_value)?;
            writeln!(f, "  Mean: {:.6}", stats.mean)?;
            writeln!(f, "  Std Dev: {:.6}", stats.std_dev)?;
            writeln!(f, "  Zeros: {}", stats.zero_count)?;
            writeln!(f, "  NaN: {}", stats.nan_count)?;
            writeln!(f, "  Inf: {}", stats.inf_count)?;
            if let Some(unique) = stats.unique_values {
                writeln!(f, "  Unique: {unique}")?;
            }
        }

        // Validation
        writeln!(f, "\nValidation:")?;
        writeln!(f, "  Valid: {}", self.validation.is_valid)?;
        if !self.validation.errors.is_empty() {
            writeln!(f, "  Errors:")?;
            for error in &self.validation.errors {
                writeln!(f, "    - {error}")?;
            }
        }
        if !self.validation.warnings.is_empty() {
            writeln!(f, "  Warnings:")?;
            for warning in &self.validation.warnings {
                writeln!(f, "    - {warning}")?;
            }
        }
        if !self.validation.recommendations.is_empty() {
            writeln!(f, "  Recommendations:")?;
            for rec in &self.validation.recommendations {
                writeln!(f, "    - {rec}")?;
            }
        }

        // Data Preview
        writeln!(
            f,
            "\nData Preview ({} elements{}):",
            self.data_preview.total_shown,
            if self.data_preview.is_truncated {
                " - truncated"
            } else {
                ""
            }
        )?;

        for (i, (element, indices)) in self
            .data_preview
            .sample_elements
            .iter()
            .zip(&self.data_preview.sample_indices)
            .enumerate()
        {
            if i < 10 {
                // Limit preview output
                writeln!(f, "  [{indices:?}] = {element}")?;
            }
        }

        if self.data_preview.sample_elements.len() > 10 {
            writeln!(
                f,
                "  ... ({} more elements)",
                self.data_preview.sample_elements.len() - 10
            )?;
        }

        Ok(())
    }
}

/// Utility functions for tensor debugging
pub mod debug_utils {
    use super::*;

    /// Quick tensor inspection with default configuration
    pub fn quick_inspect<T>(
        shape: &Shape,
        dtype: DType,
        device: &dyn Device,
        data: Option<&[T]>,
    ) -> String
    where
        T: fmt::Debug + Clone + 'static,
    {
        let mut inspector = TensorInspector::new();
        match inspector.inspect(shape, dtype, device, data) {
            Ok(inspection) => format!("{inspection}"),
            Err(e) => format!("Inspection failed: {e:?}"),
        }
    }

    /// Create a summary report for multiple tensors
    pub fn create_summary_report(inspections: &[TensorInspection]) -> String {
        let mut report = String::new();

        report.push_str("=== Tensor Summary Report ===\n\n");

        // Aggregate statistics
        let total_tensors = inspections.len();
        let total_elements: usize = inspections
            .iter()
            .map(|i| i.properties.total_elements)
            .sum();
        let total_memory: usize = inspections.iter().map(|i| i.properties.memory_bytes).sum();

        report.push_str(&format!("Total Tensors: {total_tensors}\n"));
        report.push_str(&format!("Total Elements: {total_elements}\n"));
        let total_memory_mb = total_memory as f64 / 1_048_576.0;
        report.push_str(&format!(
            "Total Memory: {total_memory} bytes ({total_memory_mb:.2} MB)\n"
        ));

        // Device distribution
        let mut device_counts = HashMap::new();
        for inspection in inspections {
            *device_counts
                .entry(&inspection.properties.device)
                .or_insert(0) += 1;
        }

        report.push_str("\nDevice Distribution:\n");
        for (device, count) in device_counts {
            report.push_str(&format!("  {device}: {count} tensors\n"));
        }

        // Validation summary
        let valid_tensors = inspections.iter().filter(|i| i.validation.is_valid).count();
        let warning_tensors = inspections
            .iter()
            .filter(|i| !i.validation.warnings.is_empty())
            .count();
        let error_tensors = inspections
            .iter()
            .filter(|i| !i.validation.errors.is_empty())
            .count();

        report.push_str("\nValidation Summary:\n");
        report.push_str(&format!("  Valid: {valid_tensors} tensors\n"));
        report.push_str(&format!("  Warnings: {warning_tensors} tensors\n"));
        report.push_str(&format!("  Errors: {error_tensors} tensors\n"));

        report
    }

    /// Export inspection results to JSON format
    pub fn export_to_json(inspection: &TensorInspection) -> Result<String> {
        use std::collections::HashMap;

        let mut json_data = HashMap::new();

        // Properties
        let mut properties = HashMap::new();
        properties.insert("shape", format!("{:?}", inspection.properties.shape));
        properties.insert("dtype", inspection.properties.dtype.clone());
        properties.insert("device", inspection.properties.device.clone());
        properties.insert(
            "total_elements",
            inspection.properties.total_elements.to_string(),
        );
        properties.insert(
            "memory_bytes",
            inspection.properties.memory_bytes.to_string(),
        );
        properties.insert(
            "is_contiguous",
            inspection.properties.is_contiguous.to_string(),
        );
        properties.insert("ndim", inspection.properties.ndim.to_string());
        json_data.insert("properties", properties);

        // Memory layout
        let mut memory_layout = HashMap::new();
        memory_layout.insert("strides", format!("{:?}", inspection.memory_layout.strides));
        memory_layout.insert(
            "layout_pattern",
            format!("{:?}", inspection.memory_layout.layout_pattern),
        );
        memory_layout.insert(
            "memory_efficiency",
            inspection.memory_layout.memory_efficiency.to_string(),
        );
        memory_layout.insert(
            "cache_line_utilization",
            inspection
                .memory_layout
                .cache_behavior
                .cache_line_utilization
                .to_string(),
        );
        memory_layout.insert(
            "spatial_locality_score",
            inspection
                .memory_layout
                .cache_behavior
                .spatial_locality_score
                .to_string(),
        );
        memory_layout.insert(
            "fragmentation_ratio",
            inspection
                .memory_layout
                .memory_fragmentation
                .fragmentation_ratio
                .to_string(),
        );
        json_data.insert("memory_layout", memory_layout);

        // Validation
        let mut validation = HashMap::new();
        validation.insert("is_valid", inspection.validation.is_valid.to_string());
        validation.insert(
            "warnings_count",
            inspection.validation.warnings.len().to_string(),
        );
        validation.insert(
            "errors_count",
            inspection.validation.errors.len().to_string(),
        );
        json_data.insert("validation", validation);

        // Simplified JSON serialization (would use serde in production)
        let shape_str = format!("{:?}", inspection.properties.shape);
        let json_string = format!(
            r#"{{"properties": {{"shape": "{}", "dtype": "{}", "device": "{}", "total_elements": {}, "memory_bytes": {}}}, "memory_layout": {{"efficiency": {}, "cache_utilization": {}}}, "validation": {{"is_valid": {}}}}}"#,
            shape_str,
            inspection.properties.dtype,
            inspection.properties.device,
            inspection.properties.total_elements,
            inspection.properties.memory_bytes,
            inspection.memory_layout.memory_efficiency,
            inspection
                .memory_layout
                .cache_behavior
                .cache_line_utilization,
            inspection.validation.is_valid
        );

        Ok(json_string)
    }

    /// Generate performance recommendations based on inspection
    pub fn generate_performance_recommendations(inspection: &TensorInspection) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Memory efficiency recommendations
        if inspection.memory_layout.memory_efficiency < 0.5 {
            recommendations.push(
                "Consider using a more memory-efficient layout (e.g., transpose to row-major)"
                    .to_string(),
            );
        }

        // Cache behavior recommendations
        if inspection
            .memory_layout
            .cache_behavior
            .cache_line_utilization
            < 0.3
        {
            recommendations.push(
                "Poor cache line utilization - consider restructuring memory access patterns"
                    .to_string(),
            );
        }

        if inspection
            .memory_layout
            .cache_behavior
            .spatial_locality_score
            < 0.4
        {
            recommendations.push(
                "Poor spatial locality - consider block-wise processing or data layout changes"
                    .to_string(),
            );
        }

        // Access pattern recommendations
        match &inspection.memory_layout.cache_behavior.access_pattern {
            AccessPattern::Random => {
                recommendations.push("Random access pattern detected - consider sorting data or using spatial data structures".to_string());
            }
            AccessPattern::Transpose => {
                recommendations.push("Transpose access pattern detected - consider transposing the tensor for better cache performance".to_string());
            }
            AccessPattern::Strided { stride } if *stride > 16 => {
                recommendations.push(format!("Large stride detected ({stride}) - consider reshaping or using different memory layout"));
            }
            _ => {}
        }

        // Fragmentation recommendations
        if inspection
            .memory_layout
            .memory_fragmentation
            .fragmentation_ratio
            > 0.2
        {
            recommendations.push(
                "High memory fragmentation detected - consider using contiguous memory allocation"
                    .to_string(),
            );
        }

        // Size-based recommendations
        if inspection.properties.total_elements > 10_000_000 {
            recommendations.push(
                "Large tensor detected - consider chunking for better memory management"
                    .to_string(),
            );
        }

        if inspection.properties.total_elements < 1000 {
            recommendations.push(
                "Small tensor detected - overhead may dominate, consider batching".to_string(),
            );
        }

        recommendations
    }

    /// Compare two tensor inspections for differences
    pub fn compare_inspections(left: &TensorInspection, right: &TensorInspection) -> String {
        let mut comparison = String::new();

        comparison.push_str("=== Tensor Comparison ===\n\n");

        // Shape comparison
        if left.properties.shape != right.properties.shape {
            comparison.push_str(&format!(
                "Shape differs: {:?} vs {:?}\n",
                left.properties.shape, right.properties.shape
            ));
        }

        // DType comparison
        if left.properties.dtype != right.properties.dtype {
            comparison.push_str(&format!(
                "DType differs: {} vs {}\n",
                left.properties.dtype, right.properties.dtype
            ));
        }

        // Device comparison
        if left.properties.device != right.properties.device {
            comparison.push_str(&format!(
                "Device differs: {} vs {}\n",
                left.properties.device, right.properties.device
            ));
        }

        // Memory efficiency comparison
        let left_eff = left.memory_layout.memory_efficiency;
        let right_eff = right.memory_layout.memory_efficiency;
        if (left_eff - right_eff).abs() > 0.01 {
            comparison.push_str(&format!(
                "Memory efficiency differs: {:.2}% vs {:.2}%\n",
                left_eff * 100.0,
                right_eff * 100.0
            ));
        }

        if comparison.len() == "=== Tensor Comparison ===\n\n".len() {
            comparison.push_str("Tensors are equivalent in basic properties.\n");
        }

        comparison
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::CpuDevice;
    use crate::{DType, Shape};

    #[test]
    fn test_tensor_inspector_creation() {
        let inspector = TensorInspector::new();
        assert_eq!(inspector.stats.tensors_inspected, 0);
    }

    #[test]
    fn test_tensor_inspector_with_config() {
        let config = InspectorConfig {
            max_preview_elements: 10,
            show_memory_layout: false,
            ..Default::default()
        };
        let inspector = TensorInspector::with_config(config);
        assert_eq!(inspector.config.max_preview_elements, 10);
        assert!(!inspector.config.show_memory_layout);
    }

    #[test]
    fn test_shape_inspection() {
        let mut inspector = TensorInspector::new();
        let shape = Shape::new(vec![2, 3, 4]);
        let data = vec![1.0f32; 24];

        let cpu_device = CpuDevice::new();
        let result = inspector.inspect(&shape, DType::F32, &cpu_device, Some(&data));
        assert!(result.is_ok());

        let inspection = result.unwrap();
        assert_eq!(inspection.properties.shape, vec![2, 3, 4]);
        assert_eq!(inspection.properties.total_elements, 24);
        assert!(inspection.validation.is_valid);
    }

    #[test]
    fn test_memory_layout_analysis() {
        let inspector = TensorInspector::new();
        let dims = [2, 3, 4];
        let strides = inspector.compute_strides(&dims);
        assert_eq!(strides, vec![12, 4, 1]);

        let pattern = inspector.detect_layout_pattern(&strides);
        assert_eq!(pattern, LayoutPattern::RowMajor);
    }

    #[test]
    fn test_validation_warnings() {
        let mut inspector = TensorInspector::new();
        let shape = Shape::new(vec![1000000001]); // Very large tensor (> 1B elements)

        let cpu_device = CpuDevice::new();
        let result = inspector.inspect(&shape, DType::F32, &cpu_device, None::<&[f32]>);
        assert!(result.is_ok());

        let inspection = result.unwrap();
        assert!(!inspection.validation.warnings.is_empty());
    }

    #[test]
    fn test_quick_inspect_utility() {
        let shape = Shape::new(vec![2, 2]);
        let data = vec![1.0f32, 2.0, 3.0, 4.0];

        let cpu_device = CpuDevice::new();
        let report = debug_utils::quick_inspect(&shape, DType::F32, &cpu_device, Some(&data));
        assert!(report.contains("Shape: [2, 2]"));
        assert!(report.contains("Elements: 4"));
    }

    #[test]
    fn test_enhanced_memory_layout_analysis() {
        let mut inspector = TensorInspector::new();
        let shape = Shape::new(vec![4, 4]);
        let data = vec![1.0f32; 16];

        let cpu_device = CpuDevice::new();
        let result = inspector.inspect(&shape, DType::F32, &cpu_device, Some(&data));
        assert!(result.is_ok());

        let inspection = result.unwrap();

        // Check visual representation is generated
        assert!(!inspection.memory_layout.visual_representation.is_empty());
        assert!(inspection
            .memory_layout
            .visual_representation
            .contains("Memory Layout Visualization"));

        // Check cache behavior analysis
        assert!(
            inspection
                .memory_layout
                .cache_behavior
                .cache_line_utilization
                > 0.0
        );
        assert!(
            inspection
                .memory_layout
                .cache_behavior
                .spatial_locality_score
                >= 0.0
        );
        assert!(
            inspection
                .memory_layout
                .cache_behavior
                .spatial_locality_score
                <= 1.0
        );

        // Check memory fragmentation analysis
        assert!(
            inspection
                .memory_layout
                .memory_fragmentation
                .fragmentation_ratio
                >= 0.0
        );
        assert!(
            inspection
                .memory_layout
                .memory_fragmentation
                .fragmentation_ratio
                <= 1.0
        );
    }

    #[test]
    fn test_access_pattern_classification() {
        let inspector = TensorInspector::new();

        // Test sequential access pattern
        let sequential_strides = vec![4, 1];
        let pattern = inspector.classify_access_pattern(&[2, 4], &sequential_strides);
        assert_eq!(pattern, AccessPattern::Sequential);

        // Test strided access pattern
        let strided_strides = vec![8];
        let pattern = inspector.classify_access_pattern(&[4], &strided_strides);
        match pattern {
            AccessPattern::Strided { stride } => assert_eq!(stride, 8),
            _ => panic!("Expected strided pattern"),
        }
    }

    #[test]
    #[ignore]
    fn test_performance_profiling() {
        let inspector = TensorInspector::new();

        // Profile a simple operation
        let (result, profile) = inspector.profile_operation("test_operation", || {
            let mut sum = 0;
            for i in 0..1000 {
                sum += i;
            }
            sum
        });

        assert_eq!(result, 499500); // Sum of 0..999
        assert_eq!(profile.operation_name, "test_operation");
        assert!(profile.execution_time_ns > 0);
    }

    #[test]
    fn test_performance_recommendations() {
        let mut inspector = TensorInspector::new();

        // Create a small tensor (< 1000 elements)
        let shape = Shape::new(vec![100]); // Small tensor
        let cpu_device = CpuDevice::new();
        let result = inspector.inspect(&shape, DType::F32, &cpu_device, None::<&[f32]>);
        assert!(result.is_ok());

        let inspection = result.unwrap();
        let recommendations = debug_utils::generate_performance_recommendations(&inspection);

        // Should recommend batching for small tensors
        assert!(recommendations
            .iter()
            .any(|r| r.contains("Small tensor") && r.contains("batching")));

        // Test large tensor recommendations
        let large_shape = Shape::new(vec![20_000_000]); // Large tensor
        let large_result = inspector.inspect(&large_shape, DType::F32, &cpu_device, None::<&[f32]>);
        assert!(large_result.is_ok());

        let large_inspection = large_result.unwrap();
        let large_recommendations =
            debug_utils::generate_performance_recommendations(&large_inspection);

        // Should recommend chunking for large tensors
        assert!(large_recommendations
            .iter()
            .any(|r| r.contains("Large tensor") && r.contains("chunking")));
    }

    #[test]
    fn test_json_export() {
        let mut inspector = TensorInspector::new();
        let shape = Shape::new(vec![2, 3]);
        let cpu_device = CpuDevice::new();
        let result = inspector.inspect(&shape, DType::F32, &cpu_device, None::<&[f32]>);
        assert!(result.is_ok());

        let inspection = result.unwrap();
        let json_result = debug_utils::export_to_json(&inspection);
        assert!(json_result.is_ok());

        let json_string = json_result.unwrap();
        assert!(json_string.contains("properties"));
        assert!(json_string.contains("memory_layout"));
        assert!(json_string.contains("validation"));
        assert!(json_string.contains("[2, 3]")); // Shape should be in JSON
    }
}
