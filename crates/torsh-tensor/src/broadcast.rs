//! Tensor broadcasting operations with comprehensive error handling
//!
//! Broadcasting allows operations between tensors of different shapes by automatically
//! expanding dimensions according to NumPy/PyTorch broadcasting rules.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use torsh_core::error::{Result, TorshError};
use torsh_core::Shape;

/// Error types specific to broadcasting operations
#[derive(Debug, Clone)]
pub enum BroadcastError {
    /// Shapes are not compatible for broadcasting
    IncompatibleShapes {
        shape1: Vec<usize>,
        shape2: Vec<usize>,
        reason: String,
    },
    /// Dimension size mismatch
    DimensionMismatch {
        dim: usize,
        size1: usize,
        size2: usize,
    },
    /// Shape computation overflow
    ShapeOverflow { attempted_shape: Vec<usize> },
    /// Memory allocation failure
    MemoryError {
        required_size: usize,
        available_size: Option<usize>,
    },
}

impl std::fmt::Display for BroadcastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BroadcastError::IncompatibleShapes {
                shape1,
                shape2,
                reason,
            } => {
                write!(
                    f,
                    "Cannot broadcast shapes {shape1:?} and {shape2:?}: {reason}"
                )
            }
            BroadcastError::DimensionMismatch { dim, size1, size2 } => {
                write!(
                    f,
                    "Dimension {dim} mismatch: {size1} vs {size2} (neither is 1)"
                )
            }
            BroadcastError::ShapeOverflow { attempted_shape } => {
                write!(
                    f,
                    "Broadcast shape {attempted_shape:?} would overflow memory limits"
                )
            }
            BroadcastError::MemoryError {
                required_size,
                available_size,
            } => {
                if let Some(available) = available_size {
                    write!(
                        f,
                        "Insufficient memory for broadcast: need {required_size} bytes, have {available} bytes"
                    )
                } else {
                    write!(f, "Memory allocation failed for {required_size} bytes")
                }
            }
        }
    }
}

impl std::error::Error for BroadcastError {}

/// Broadcasting utilities for tensors
pub struct BroadcastOps;

impl BroadcastOps {
    /// Check if two shapes are compatible for broadcasting
    ///
    /// Broadcasting rules (from right to left):
    /// 1. Dimensions must be equal, or one of them is 1, or one is missing
    /// 2. Missing dimensions are treated as 1
    pub fn are_shapes_compatible(shape1: &[usize], shape2: &[usize]) -> Result<bool> {
        let ndim1 = shape1.len();
        let ndim2 = shape2.len();
        let max_ndim = ndim1.max(ndim2);

        for i in 0..max_ndim {
            let dim1 = if i < ndim1 {
                shape1[ndim1 - 1 - i]
            } else {
                1 // Missing dimensions are treated as 1
            };

            let dim2 = if i < ndim2 {
                shape2[ndim2 - 1 - i]
            } else {
                1 // Missing dimensions are treated as 1
            };

            // Check broadcasting compatibility
            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Compute the broadcasted shape for two input shapes
    pub fn compute_broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Result<Vec<usize>> {
        if !Self::are_shapes_compatible(shape1, shape2)? {
            return Err(TorshError::BroadcastError {
                shape1: shape1.to_vec(),
                shape2: shape2.to_vec(),
            });
        }

        let ndim1 = shape1.len();
        let ndim2 = shape2.len();
        let max_ndim = ndim1.max(ndim2);
        let mut result_shape = Vec::with_capacity(max_ndim);

        for i in 0..max_ndim {
            let dim1 = if i < ndim1 { shape1[ndim1 - 1 - i] } else { 1 };

            let dim2 = if i < ndim2 { shape2[ndim2 - 1 - i] } else { 1 };

            // The result dimension is the maximum of the two
            let result_dim = dim1.max(dim2);

            // Check for potential overflow
            if result_dim > usize::MAX / 2 {
                return Err(TorshError::InvalidArgument(
                    BroadcastError::ShapeOverflow {
                        attempted_shape: result_shape.clone(),
                    }
                    .to_string(),
                ));
            }

            result_shape.push(result_dim);
        }

        // Reverse to get proper order (we computed from right to left)
        result_shape.reverse();

        // Check total size doesn't overflow
        let total_elements = result_shape.iter().product::<usize>();
        if total_elements > isize::MAX as usize {
            return Err(TorshError::InvalidArgument(
                BroadcastError::ShapeOverflow {
                    attempted_shape: result_shape,
                }
                .to_string(),
            ));
        }

        Ok(result_shape)
    }

    /// Compute the linear index for a tensor in a broadcasted operation
    pub fn compute_broadcast_index(
        multi_index: &[usize],
        original_shape: &[usize],
        broadcast_shape: &[usize],
    ) -> Result<usize> {
        let orig_ndim = original_shape.len();
        let broadcast_ndim = broadcast_shape.len();

        if multi_index.len() != broadcast_ndim {
            return Err(TorshError::InvalidArgument(format!(
                "Multi-index length {} doesn't match broadcast shape dimensions {}",
                multi_index.len(),
                broadcast_ndim
            )));
        }

        let mut linear_index = 0;
        let mut stride = 1;

        // Process from right to left (last dimension first)
        for i in 0..broadcast_ndim {
            let broadcast_dim_idx = broadcast_ndim - 1 - i;
            let broadcast_coord = multi_index[broadcast_dim_idx];

            // Map to original shape coordinates
            let orig_coord = if i < orig_ndim {
                let orig_dim_idx = orig_ndim - 1 - i;
                let orig_dim_size = original_shape[orig_dim_idx];

                if orig_dim_size == 1 {
                    0 // Broadcasting: use index 0 for size-1 dimensions
                } else {
                    broadcast_coord // Use the coordinate directly
                }
            } else {
                0 // Missing dimensions are treated as 0 coordinate
            };

            // Add contribution to linear index
            linear_index += orig_coord * stride;

            // Update stride for next dimension
            if i < orig_ndim {
                let orig_dim_idx = orig_ndim - 1 - i;
                stride *= original_shape[orig_dim_idx];
            }
        }

        Ok(linear_index)
    }

    /// Convert flat index to multi-dimensional index
    pub fn flat_to_multi_index(flat_index: usize, shape: &[usize]) -> Vec<usize> {
        let mut multi_index = Vec::with_capacity(shape.len());
        let mut remaining = flat_index;

        for &dim_size in shape.iter().rev() {
            multi_index.push(remaining % dim_size);
            remaining /= dim_size;
        }

        multi_index.reverse();
        multi_index
    }

    /// Validate broadcasting operation parameters
    pub fn validate_broadcast_operation(
        shape1: &[usize],
        shape2: &[usize],
        operation_name: &str,
    ) -> Result<()> {
        // Empty shapes are allowed for scalar tensors
        // Only check for zero dimensions in non-empty shapes

        // Check for zero dimensions
        if shape1.contains(&0) || shape2.contains(&0) {
            return Err(TorshError::InvalidArgument(format!(
                "Cannot perform {operation_name} operation on tensors with zero-sized dimensions"
            )));
        }

        // Check maximum number of dimensions
        const MAX_DIMENSIONS: usize = 32; // Reasonable limit for memory and performance
        if shape1.len() > MAX_DIMENSIONS || shape2.len() > MAX_DIMENSIONS {
            return Err(TorshError::InvalidArgument(format!(
                "Too many dimensions for {operation_name} operation (max: {MAX_DIMENSIONS})"
            )));
        }

        // Check broadcasting compatibility
        if !Self::are_shapes_compatible(shape1, shape2)? {
            return Err(TorshError::InvalidArgument(
                BroadcastError::IncompatibleShapes {
                    shape1: shape1.to_vec(),
                    shape2: shape2.to_vec(),
                    reason: format!("Shapes not compatible for {operation_name} operation"),
                }
                .to_string(),
            ));
        }

        Ok(())
    }

    /// Estimate memory requirements for a broadcast operation
    pub fn estimate_broadcast_memory(
        shape1: &[usize],
        shape2: &[usize],
        element_size: usize,
    ) -> Result<usize> {
        let broadcast_shape = Self::compute_broadcast_shape(shape1, shape2)?;
        let num_elements = broadcast_shape.iter().product::<usize>();

        // Check for overflow
        let memory_required = num_elements.checked_mul(element_size).ok_or_else(|| {
            TorshError::InvalidArgument(
                BroadcastError::MemoryError {
                    required_size: usize::MAX,
                    available_size: None,
                }
                .to_string(),
            )
        })?;

        Ok(memory_required)
    }

    /// Get detailed broadcasting information for debugging
    pub fn get_broadcast_info(shape1: &[usize], shape2: &[usize]) -> Result<BroadcastInfo> {
        let broadcast_shape = Self::compute_broadcast_shape(shape1, shape2)?;
        let expansion_factor1 =
            broadcast_shape.iter().product::<usize>() / shape1.iter().product::<usize>();
        let expansion_factor2 =
            broadcast_shape.iter().product::<usize>() / shape2.iter().product::<usize>();

        Ok(BroadcastInfo {
            original_shape1: shape1.to_vec(),
            original_shape2: shape2.to_vec(),
            broadcast_shape,
            expansion_factor1,
            expansion_factor2,
            is_memory_efficient: expansion_factor1 <= 2 && expansion_factor2 <= 2,
        })
    }

    /// Compute pre-computed strides for efficient broadcasting operations
    pub fn compute_broadcast_strides(
        shape1: &[usize],
        shape2: &[usize],
        broadcast_shape: &[usize],
    ) -> Result<BroadcastStrides> {
        let original_strides1 = Self::compute_strides(shape1);
        let original_strides2 = Self::compute_strides(shape2);

        let broadcast_strides1 =
            Self::compute_broadcast_strides_for_shape(shape1, broadcast_shape, &original_strides1)?;
        let broadcast_strides2 =
            Self::compute_broadcast_strides_for_shape(shape2, broadcast_shape, &original_strides2)?;

        Ok(BroadcastStrides {
            original_strides1,
            original_strides2,
            broadcast_strides1,
            broadcast_strides2,
            broadcast_shape: broadcast_shape.to_vec(),
        })
    }

    /// Compute strides for a given shape (row-major/C-style)
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        if shape.is_empty() {
            return Vec::new();
        }

        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Compute broadcast strides for a specific shape to match broadcast shape
    fn compute_broadcast_strides_for_shape(
        original_shape: &[usize],
        broadcast_shape: &[usize],
        original_strides: &[usize],
    ) -> Result<Vec<usize>> {
        let orig_ndim = original_shape.len();
        let broadcast_ndim = broadcast_shape.len();
        let mut broadcast_strides = vec![0; broadcast_ndim];

        for i in 0..broadcast_ndim {
            let broadcast_dim_idx = broadcast_ndim - 1 - i;

            if i < orig_ndim {
                let orig_dim_idx = orig_ndim - 1 - i;
                let orig_size = original_shape[orig_dim_idx];
                let broadcast_size = broadcast_shape[broadcast_dim_idx];

                if orig_size == broadcast_size {
                    // No broadcasting needed for this dimension
                    broadcast_strides[broadcast_dim_idx] = original_strides[orig_dim_idx];
                } else if orig_size == 1 {
                    // Broadcasting: stride becomes 0 to repeat the single element
                    broadcast_strides[broadcast_dim_idx] = 0;
                } else {
                    return Err(TorshError::InvalidArgument(format!(
                        "Cannot broadcast dimension {orig_dim_idx}: original size {orig_size}, broadcast size {broadcast_size}"
                    )));
                }
            } else {
                // Missing dimension in original shape, treat as size 1 with stride 0
                broadcast_strides[broadcast_dim_idx] = 0;
            }
        }

        Ok(broadcast_strides)
    }

    /// Detect common broadcasting patterns for optimization
    pub fn detect_broadcast_pattern(shape1: &[usize], shape2: &[usize]) -> BroadcastPattern {
        // Scalar broadcasting (one operand is scalar)
        if shape1.is_empty() || shape2.is_empty() {
            return BroadcastPattern::Scalar;
        }

        // Element-wise (same shape)
        if shape1 == shape2 {
            return BroadcastPattern::ElementWise;
        }

        // Matrix-vector broadcasting (2D with 1D)
        if (shape1.len() == 2 && shape2.len() == 1) || (shape1.len() == 1 && shape2.len() == 2) {
            return BroadcastPattern::MatrixVector;
        }

        // Vector-scalar broadcasting (one operand is 1D, but not matrix-vector case)
        if shape1.len() == 1 || shape2.len() == 1 {
            return BroadcastPattern::VectorScalar;
        }

        // Check for size-1 dimension patterns
        let has_size_1_dims = shape1.contains(&1) || shape2.contains(&1);
        if has_size_1_dims {
            return BroadcastPattern::Size1Dimension;
        }

        // Default to general broadcasting
        BroadcastPattern::General
    }

    /// Create optimized broadcasting preview with cost estimation
    pub fn create_broadcast_preview(
        shape1: &[usize],
        shape2: &[usize],
        element_size: usize,
    ) -> BroadcastPreview {
        // Check compatibility first
        let compatible = Self::are_shapes_compatible(shape1, shape2).unwrap_or_default();

        if !compatible {
            return BroadcastPreview {
                success: false,
                broadcast_shape: None,
                memory_required: None,
                expansion_factor1: None,
                expansion_factor2: None,
                is_memory_efficient: false,
                operation_cost: OperationCost {
                    computational_complexity: 0,
                    memory_access_pattern: MemoryAccessPattern::Sequential,
                    cache_efficiency: 0.0,
                    estimated_runtime_ms: 0.0,
                },
                error_message: Some("Shapes are not compatible for broadcasting".to_string()),
            };
        }

        // Compute broadcast details
        let broadcast_shape = match Self::compute_broadcast_shape(shape1, shape2) {
            Ok(shape) => shape,
            Err(e) => {
                return BroadcastPreview {
                    success: false,
                    broadcast_shape: None,
                    memory_required: None,
                    expansion_factor1: None,
                    expansion_factor2: None,
                    is_memory_efficient: false,
                    operation_cost: OperationCost {
                        computational_complexity: 0,
                        memory_access_pattern: MemoryAccessPattern::Sequential,
                        cache_efficiency: 0.0,
                        estimated_runtime_ms: 0.0,
                    },
                    error_message: Some(format!("Error computing broadcast shape: {e}")),
                };
            }
        };

        let memory_required = Self::estimate_broadcast_memory(shape1, shape2, element_size).ok();

        let info = Self::get_broadcast_info(shape1, shape2)
            .expect("broadcast info should be available after shape validation");
        let pattern = Self::detect_broadcast_pattern(shape1, shape2);
        let cost = Self::estimate_operation_cost(&pattern, &broadcast_shape, element_size);

        BroadcastPreview {
            success: true,
            broadcast_shape: Some(broadcast_shape),
            memory_required,
            expansion_factor1: Some(info.expansion_factor1),
            expansion_factor2: Some(info.expansion_factor2),
            is_memory_efficient: info.is_memory_efficient,
            operation_cost: cost,
            error_message: None,
        }
    }

    /// Estimate operation cost for different broadcasting patterns
    fn estimate_operation_cost(
        pattern: &BroadcastPattern,
        broadcast_shape: &[usize],
        element_size: usize,
    ) -> OperationCost {
        let num_elements = broadcast_shape.iter().product::<usize>();
        let memory_bytes = num_elements * element_size;

        let (complexity, access_pattern, cache_efficiency, runtime_factor) = match pattern {
            BroadcastPattern::Scalar => (num_elements, MemoryAccessPattern::Sequential, 0.95, 1.0),
            BroadcastPattern::ElementWise => {
                (num_elements, MemoryAccessPattern::Sequential, 0.9, 1.0)
            }
            BroadcastPattern::VectorScalar => {
                (num_elements, MemoryAccessPattern::Sequential, 0.8, 1.2)
            }
            BroadcastPattern::MatrixVector => {
                let stride = broadcast_shape.last().unwrap_or(&1);
                (
                    num_elements,
                    MemoryAccessPattern::Strided { stride: *stride },
                    0.7,
                    1.5,
                )
            }
            BroadcastPattern::Size1Dimension => {
                (num_elements, MemoryAccessPattern::Random, 0.6, 2.0)
            }
            BroadcastPattern::General => (num_elements, MemoryAccessPattern::Random, 0.5, 2.5),
        };

        // Estimate runtime based on memory bandwidth and complexity
        let estimated_runtime_ms = (memory_bytes as f64 / 1e9) * runtime_factor; // Assume 1GB/s bandwidth

        OperationCost {
            computational_complexity: complexity,
            memory_access_pattern: access_pattern,
            cache_efficiency,
            estimated_runtime_ms,
        }
    }
}

/// Broadcasting patterns for optimization
#[derive(Debug, Clone, PartialEq)]
pub enum BroadcastPattern {
    /// Scalar broadcasting (one operand is scalar)
    Scalar,
    /// Element-wise operation (same shapes)
    ElementWise,
    /// Vector-scalar broadcasting
    VectorScalar,
    /// Matrix-vector broadcasting
    MatrixVector,
    /// Broadcasting with size-1 dimensions
    Size1Dimension,
    /// General broadcasting case
    General,
}

/// Information about a broadcasting operation
#[derive(Debug, Clone)]
pub struct BroadcastInfo {
    pub original_shape1: Vec<usize>,
    pub original_shape2: Vec<usize>,
    pub broadcast_shape: Vec<usize>,
    pub expansion_factor1: usize,
    pub expansion_factor2: usize,
    pub is_memory_efficient: bool,
}

/// Pre-computed strides for efficient broadcasting operations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BroadcastStrides {
    pub original_strides1: Vec<usize>,
    pub original_strides2: Vec<usize>,
    pub broadcast_strides1: Vec<usize>,
    pub broadcast_strides2: Vec<usize>,
    pub broadcast_shape: Vec<usize>,
}

/// Cache key for broadcasting operations
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BroadcastCacheKey {
    shape1: Vec<usize>,
    shape2: Vec<usize>,
}

/// Cached broadcasting computation results
#[derive(Debug, Clone)]
pub struct BroadcastCacheEntry {
    pub broadcast_shape: Vec<usize>,
    pub strides: BroadcastStrides,
    pub info: BroadcastInfo,
    access_count: usize,
    last_accessed: std::time::SystemTime,
}

/// Global broadcasting cache for repeated operations
static BROADCAST_CACHE: std::sync::LazyLock<
    Arc<Mutex<HashMap<BroadcastCacheKey, BroadcastCacheEntry>>>,
> = std::sync::LazyLock::new(|| Arc::new(Mutex::new(HashMap::new())));

/// Cache manager for broadcasting operations
pub struct BroadcastCache;

impl BroadcastCache {
    /// Get cached broadcast result or compute and cache it
    pub fn get_or_compute(
        shape1: &[usize],
        shape2: &[usize],
        config: &BroadcastCacheConfig,
    ) -> Result<BroadcastCacheEntry> {
        if !config.enable_cache {
            return Self::compute_fresh(shape1, shape2);
        }

        let key = BroadcastCacheKey {
            shape1: shape1.to_vec(),
            shape2: shape2.to_vec(),
        };

        let mut cache = BROADCAST_CACHE.lock().expect("lock should not be poisoned");

        // Check if entry exists and is not expired
        if let Some(entry) = cache.get_mut(&key) {
            let now = std::time::SystemTime::now();
            let age = now
                .duration_since(entry.last_accessed)
                .unwrap_or_default()
                .as_secs();

            if age < config.ttl_seconds {
                entry.access_count += 1;
                entry.last_accessed = now;
                return Ok(entry.clone());
            } else {
                // Entry expired, remove it
                cache.remove(&key);
            }
        }

        // Compute fresh result
        let mut entry = Self::compute_fresh(shape1, shape2)?;
        entry.access_count = 1;
        entry.last_accessed = std::time::SystemTime::now();

        // Evict old entries if cache is full
        if cache.len() >= config.max_entries {
            Self::evict_lru(&mut cache);
        }

        // Insert new entry
        cache.insert(key, entry.clone());
        Ok(entry)
    }

    /// Compute fresh broadcast result without caching
    fn compute_fresh(shape1: &[usize], shape2: &[usize]) -> Result<BroadcastCacheEntry> {
        let broadcast_shape = BroadcastOps::compute_broadcast_shape(shape1, shape2)?;
        let strides = BroadcastOps::compute_broadcast_strides(shape1, shape2, &broadcast_shape)?;
        let info = BroadcastOps::get_broadcast_info(shape1, shape2)?;

        Ok(BroadcastCacheEntry {
            broadcast_shape,
            strides,
            info,
            access_count: 0,
            last_accessed: std::time::SystemTime::now(),
        })
    }

    /// Evict least recently used entry
    fn evict_lru(cache: &mut HashMap<BroadcastCacheKey, BroadcastCacheEntry>) {
        if let Some((oldest_key, _)) = cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(k, v)| (k.clone(), v.clone()))
        {
            cache.remove(&oldest_key);
        }
    }

    /// Clear the cache
    pub fn clear() {
        let mut cache = BROADCAST_CACHE.lock().expect("lock should not be poisoned");
        cache.clear();
    }

    /// Get cache statistics
    pub fn get_stats() -> BroadcastCacheStats {
        let cache = BROADCAST_CACHE.lock().expect("lock should not be poisoned");
        let total_accesses: usize = cache.values().map(|entry| entry.access_count).sum();

        BroadcastCacheStats {
            total_entries: cache.len(),
            total_accesses,
            hit_rate: if total_accesses > 0 {
                cache.len() as f64 / total_accesses as f64
            } else {
                0.0
            },
        }
    }
}

/// Broadcasting cache statistics
#[derive(Debug, Clone)]
pub struct BroadcastCacheStats {
    pub total_entries: usize,
    pub total_accesses: usize,
    pub hit_rate: f64,
}

/// Configuration for broadcasting cache
pub struct BroadcastCacheConfig {
    pub max_entries: usize,
    pub ttl_seconds: u64,
    pub enable_cache: bool,
}

impl Default for BroadcastCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl_seconds: 300, // 5 minutes
            enable_cache: true,
        }
    }
}

/// Broadcasting preview result for dry-run functionality
#[derive(Debug, Clone)]
pub struct BroadcastPreview {
    pub success: bool,
    pub broadcast_shape: Option<Vec<usize>>,
    pub memory_required: Option<usize>,
    pub expansion_factor1: Option<usize>,
    pub expansion_factor2: Option<usize>,
    pub is_memory_efficient: bool,
    pub operation_cost: OperationCost,
    pub error_message: Option<String>,
}

/// Cost estimation for broadcasting operations
#[derive(Debug, Clone)]
pub struct OperationCost {
    pub computational_complexity: usize,
    pub memory_access_pattern: MemoryAccessPattern,
    pub cache_efficiency: f64,
    pub estimated_runtime_ms: f64,
}

/// Memory access patterns for optimization
#[derive(Debug, Clone)]
pub enum MemoryAccessPattern {
    Sequential,
    Strided { stride: usize },
    Random,
    Broadcast { expansion_factor: usize },
}

/// Extension trait for Shape to add broadcasting methods
pub trait BroadcastShape {
    /// Check if this shape is compatible for broadcasting with another shape
    fn broadcast_compatible(&self, other: &Self) -> bool;

    /// Compute the broadcasted shape with another shape
    fn broadcast_shape(&self, other: &Self) -> Result<Shape>;

    /// Check if broadcasting would be memory efficient
    fn is_broadcast_efficient(&self, other: &Self) -> bool;
}

impl BroadcastShape for Shape {
    fn broadcast_compatible(&self, other: &Self) -> bool {
        BroadcastOps::are_shapes_compatible(self.dims(), other.dims()).unwrap_or(false)
    }

    fn broadcast_shape(&self, other: &Self) -> Result<Shape> {
        let result_dims = BroadcastOps::compute_broadcast_shape(self.dims(), other.dims())?;
        Ok(Shape::new(result_dims))
    }

    fn is_broadcast_efficient(&self, other: &Self) -> bool {
        if let Ok(info) = BroadcastOps::get_broadcast_info(self.dims(), other.dims()) {
            info.is_memory_efficient
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_compatibility() {
        // Compatible shapes
        assert!(BroadcastOps::are_shapes_compatible(&[3, 4], &[1, 4]).unwrap());
        assert!(BroadcastOps::are_shapes_compatible(&[3, 1], &[3, 4]).unwrap());
        assert!(BroadcastOps::are_shapes_compatible(&[1], &[3, 4]).unwrap());
        assert!(BroadcastOps::are_shapes_compatible(&[], &[3]).unwrap());

        // Incompatible shapes
        assert!(!BroadcastOps::are_shapes_compatible(&[3, 4], &[2, 4]).unwrap());
        assert!(!BroadcastOps::are_shapes_compatible(&[3, 2], &[4, 3]).unwrap());
    }

    #[test]
    fn test_broadcast_shape_computation() {
        // Basic broadcasting
        let result = BroadcastOps::compute_broadcast_shape(&[3, 4], &[1, 4]).unwrap();
        assert_eq!(result, vec![3, 4]);

        let result = BroadcastOps::compute_broadcast_shape(&[3, 1], &[3, 4]).unwrap();
        assert_eq!(result, vec![3, 4]);

        // Different number of dimensions
        let result = BroadcastOps::compute_broadcast_shape(&[1], &[3, 4]).unwrap();
        assert_eq!(result, vec![3, 4]);

        let result = BroadcastOps::compute_broadcast_shape(&[], &[3]).unwrap();
        assert_eq!(result, vec![3]);
    }

    #[test]
    fn test_broadcast_index_computation() {
        // Test basic broadcasting index computation
        let multi_index = vec![1, 2];
        let original_shape = vec![1, 3];
        let broadcast_shape = vec![2, 3];

        let linear_index =
            BroadcastOps::compute_broadcast_index(&multi_index, &original_shape, &broadcast_shape)
                .unwrap();

        // For shape [1, 3] with broadcast coordinates [1, 2]:
        // - dimension 0: coordinate 1 -> maps to 0 (broadcast)
        // - dimension 1: coordinate 2 -> maps to 2
        // Linear index = 0 * 3 + 2 = 2
        assert_eq!(linear_index, 2);
    }

    #[test]
    fn test_flat_to_multi_index() {
        let shape = vec![2, 3, 4];

        // Test conversion for flat index 0
        let multi_index = BroadcastOps::flat_to_multi_index(0, &shape);
        assert_eq!(multi_index, vec![0, 0, 0]);

        // Test conversion for flat index 5
        let multi_index = BroadcastOps::flat_to_multi_index(5, &shape);
        assert_eq!(multi_index, vec![0, 1, 1]);

        // Test conversion for flat index 23 (last index)
        let multi_index = BroadcastOps::flat_to_multi_index(23, &shape);
        assert_eq!(multi_index, vec![1, 2, 3]);
    }

    #[test]
    fn test_validation() {
        // Valid operation
        assert!(BroadcastOps::validate_broadcast_operation(&[3, 4], &[1, 4], "add").is_ok());

        // Invalid: incompatible shapes
        assert!(BroadcastOps::validate_broadcast_operation(&[3, 4], &[2, 5], "add").is_err());

        // Valid: scalar with non-scalar (empty shape allowed for scalars)
        assert!(BroadcastOps::validate_broadcast_operation(&[], &[3], "add").is_ok());

        // Invalid: zero dimension
        assert!(BroadcastOps::validate_broadcast_operation(&[3, 0], &[3, 1], "add").is_err());
    }

    #[test]
    fn test_memory_estimation() {
        let shape1 = vec![2, 3];
        let shape2 = vec![1, 3];
        let element_size = std::mem::size_of::<f32>();

        let memory_required =
            BroadcastOps::estimate_broadcast_memory(&shape1, &shape2, element_size).unwrap();

        // Broadcast shape should be [2, 3] = 6 elements
        // Memory = 6 * sizeof(f32) = 6 * 4 = 24 bytes
        assert_eq!(memory_required, 6 * element_size);
    }

    #[test]
    fn test_broadcast_info() {
        let shape1 = vec![1, 4];
        let shape2 = vec![3, 1];

        let info = BroadcastOps::get_broadcast_info(&shape1, &shape2).unwrap();

        assert_eq!(info.original_shape1, vec![1, 4]);
        assert_eq!(info.original_shape2, vec![3, 1]);
        assert_eq!(info.broadcast_shape, vec![3, 4]);
        assert_eq!(info.expansion_factor1, 3); // (3*4) / (1*4) = 3
        assert_eq!(info.expansion_factor2, 4); // (3*4) / (3*1) = 4
        assert!(!info.is_memory_efficient); // expansion factors > 2
    }

    #[test]
    fn test_shape_trait_extension() {
        let shape1 = Shape::new(vec![3, 4]);
        let shape2 = Shape::new(vec![1, 4]);
        let shape3 = Shape::new(vec![2, 5]);

        // Test compatibility
        assert!(shape1.broadcast_compatible(&shape2));
        assert!(!shape1.broadcast_compatible(&shape3));

        // Test broadcast shape computation
        let broadcast_result = shape1.broadcast_shape(&shape2).unwrap();
        assert_eq!(broadcast_result.dims(), &[3, 4]);
    }

    #[test]
    fn test_error_messages() {
        // Test incompatible shapes error
        let result = BroadcastOps::compute_broadcast_shape(&[3, 4], &[2, 5]);
        assert!(result.is_err());
        if let Err(TorshError::InvalidArgument(err)) = result {
            let msg = err.to_string();
            assert!(msg.contains("Cannot broadcast"));
        }

        // Test validation error
        let result = BroadcastOps::validate_broadcast_operation(&[3, 0], &[3, 1], "multiply");
        assert!(result.is_err());
    }

    #[test]
    fn test_broadcast_strides() {
        let shape1 = vec![1, 4];
        let shape2 = vec![3, 1];
        let broadcast_shape = vec![3, 4];

        let strides =
            BroadcastOps::compute_broadcast_strides(&shape1, &shape2, &broadcast_shape).unwrap();

        // Original strides for [1, 4] should be [4, 1]
        assert_eq!(strides.original_strides1, vec![4, 1]);
        // Original strides for [3, 1] should be [1, 1]
        assert_eq!(strides.original_strides2, vec![1, 1]);

        // Broadcast strides for shape1 [1, 4] -> [3, 4] should be [0, 1] (dim 0 broadcasts)
        assert_eq!(strides.broadcast_strides1, vec![0, 1]);
        // Broadcast strides for shape2 [3, 1] -> [3, 4] should be [1, 0] (dim 1 broadcasts)
        assert_eq!(strides.broadcast_strides2, vec![1, 0]);
    }

    #[test]
    fn test_broadcast_pattern_detection() {
        // Scalar pattern
        assert_eq!(
            BroadcastOps::detect_broadcast_pattern(&[], &[3, 4]),
            BroadcastPattern::Scalar
        );
        assert_eq!(
            BroadcastOps::detect_broadcast_pattern(&[3, 4], &[]),
            BroadcastPattern::Scalar
        );

        // Element-wise pattern
        assert_eq!(
            BroadcastOps::detect_broadcast_pattern(&[3, 4], &[3, 4]),
            BroadcastPattern::ElementWise
        );

        // Vector-scalar pattern (1D with scalar-like)
        assert_eq!(
            BroadcastOps::detect_broadcast_pattern(&[1], &[4]),
            BroadcastPattern::VectorScalar
        );
        assert_eq!(
            BroadcastOps::detect_broadcast_pattern(&[4], &[1]),
            BroadcastPattern::VectorScalar
        );

        // Matrix-vector pattern (2D with 1D)
        assert_eq!(
            BroadcastOps::detect_broadcast_pattern(&[3, 4], &[4]),
            BroadcastPattern::MatrixVector
        );
        assert_eq!(
            BroadcastOps::detect_broadcast_pattern(&[4], &[3, 4]),
            BroadcastPattern::MatrixVector
        );

        // Size-1 dimension pattern
        assert_eq!(
            BroadcastOps::detect_broadcast_pattern(&[1, 4], &[3, 4]),
            BroadcastPattern::Size1Dimension
        );
        assert_eq!(
            BroadcastOps::detect_broadcast_pattern(&[3, 1], &[3, 4]),
            BroadcastPattern::Size1Dimension
        );

        // General pattern
        assert_eq!(
            BroadcastOps::detect_broadcast_pattern(&[2, 3, 4], &[5, 2, 3, 4]),
            BroadcastPattern::General
        );
    }

    #[test]
    fn test_broadcast_preview() {
        let shape1 = vec![1, 4];
        let shape2 = vec![3, 1];
        let element_size = std::mem::size_of::<f32>();

        let preview = BroadcastOps::create_broadcast_preview(&shape1, &shape2, element_size);

        assert!(preview.success);
        assert_eq!(preview.broadcast_shape, Some(vec![3, 4]));
        assert_eq!(preview.memory_required, Some(12 * element_size)); // 3*4 elements
        assert_eq!(preview.expansion_factor1, Some(3)); // (3*4) / (1*4) = 3
        assert_eq!(preview.expansion_factor2, Some(4)); // (3*4) / (3*1) = 4
        assert!(!preview.is_memory_efficient); // factors > 2
        assert!(preview.error_message.is_none());

        // Test incompatible shapes
        let preview_fail = BroadcastOps::create_broadcast_preview(&[3, 4], &[2, 5], element_size);
        assert!(!preview_fail.success);
        assert!(preview_fail.error_message.is_some());
    }

    #[test]
    fn test_broadcast_cache() {
        // Clear cache first
        BroadcastCache::clear();

        let config = BroadcastCacheConfig::default();
        let shape1 = vec![1, 4];
        let shape2 = vec![3, 1];

        // First access should compute and cache
        let entry1 = BroadcastCache::get_or_compute(&shape1, &shape2, &config).unwrap();
        assert_eq!(entry1.broadcast_shape, vec![3, 4]);

        // Second access should hit cache
        let entry2 = BroadcastCache::get_or_compute(&shape1, &shape2, &config).unwrap();
        assert_eq!(entry2.broadcast_shape, vec![3, 4]);

        // Verify cache statistics
        let stats = BroadcastCache::get_stats();
        assert!(stats.total_entries > 0);
        assert!(stats.total_accesses > 0);

        // Test cache disabled
        let config_no_cache = BroadcastCacheConfig {
            enable_cache: false,
            ..Default::default()
        };
        let entry3 = BroadcastCache::get_or_compute(&shape1, &shape2, &config_no_cache).unwrap();
        assert_eq!(entry3.broadcast_shape, vec![3, 4]);
    }

    #[test]
    fn test_stride_computation() {
        // Test basic stride computation
        let shape = vec![2, 3, 4];
        let strides = BroadcastOps::compute_strides(&shape);
        assert_eq!(strides, vec![12, 4, 1]); // [3*4, 4, 1]

        // Test empty shape
        let empty_shape = vec![];
        let empty_strides = BroadcastOps::compute_strides(&empty_shape);
        assert_eq!(empty_strides, Vec::<usize>::new());

        // Test single dimension
        let single_shape = vec![5];
        let single_strides = BroadcastOps::compute_strides(&single_shape);
        assert_eq!(single_strides, vec![1]);
    }

    #[test]
    fn test_operation_cost_estimation() {
        let shape1 = vec![3, 4];
        let shape2 = vec![3, 4];
        let element_size = std::mem::size_of::<f32>();

        let preview = BroadcastOps::create_broadcast_preview(&shape1, &shape2, element_size);
        assert!(preview.success);

        let cost = &preview.operation_cost;
        assert_eq!(cost.computational_complexity, 12); // 3*4 elements
        assert!(matches!(
            cost.memory_access_pattern,
            MemoryAccessPattern::Sequential
        ));
        assert!(cost.cache_efficiency > 0.8); // Element-wise should be efficient
        assert!(cost.estimated_runtime_ms >= 0.0);
    }
}
