//! Hybrid sparse formats that combine multiple sparse representations
//!
//! This module provides intelligent sparse format selection and hybrid formats
//! that can optimize storage and computation based on matrix characteristics.

use crate::{
    CooTensor, CscTensor, CsrTensor, DiaTensor, EllTensor, SparseFormat, SparseTensor, TorshResult,
};
use std::collections::HashMap;
use torsh_core::{Shape, TorshError};
use torsh_tensor::Tensor;

/// Type alias for block-based sparse triplets
type BlockTriplets = HashMap<(usize, usize), Vec<(usize, usize, f32)>>;

/// Hybrid sparse tensor that can store different regions in different formats
pub struct HybridTensor {
    /// Map of regions to their sparse representations
    regions: HashMap<RegionId, Box<dyn SparseTensor + Send + Sync>>,
    /// Overall shape of the tensor
    shape: Shape,
    /// Number of non-zero elements across all regions
    nnz: usize,
}

/// Identifier for different regions within a hybrid tensor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegionId {
    /// Starting row
    row_start: usize,
    /// Starting column
    col_start: usize,
    /// Number of rows in region
    rows: usize,
    /// Number of columns in region
    cols: usize,
}

impl RegionId {
    /// Create a new region identifier
    pub fn new(row_start: usize, col_start: usize, rows: usize, cols: usize) -> Self {
        Self {
            row_start,
            col_start,
            rows,
            cols,
        }
    }
}

impl HybridTensor {
    /// Create a new hybrid tensor from regions
    pub fn new(
        regions: HashMap<RegionId, Box<dyn SparseTensor + Send + Sync>>,
        shape: Shape,
    ) -> TorshResult<Self> {
        let nnz = regions.values().map(|region| region.nnz()).sum();

        // Validate that regions don't overlap and fit within shape
        Self::validate_regions(&regions, &shape)?;

        Ok(Self {
            regions,
            shape,
            nnz,
        })
    }

    /// Create a hybrid tensor by automatically partitioning a sparse tensor
    pub fn from_sparse<T: SparseTensor + Send + Sync + 'static>(
        sparse: T,
        partition_strategy: PartitionStrategy,
    ) -> TorshResult<Self> {
        let shape = sparse.shape().clone();
        let regions = Self::partition_tensor(Box::new(sparse), partition_strategy)?;
        Self::new(regions, shape)
    }

    /// Partition a sparse tensor into regions using the given strategy
    fn partition_tensor(
        sparse: Box<dyn SparseTensor + Send + Sync>,
        strategy: PartitionStrategy,
    ) -> TorshResult<HashMap<RegionId, Box<dyn SparseTensor + Send + Sync>>> {
        match strategy {
            PartitionStrategy::BlockBased { block_size } => {
                Self::partition_block_based(sparse, block_size)
            }
            PartitionStrategy::DensityBased { threshold } => {
                Self::partition_density_based(sparse, threshold)
            }
            PartitionStrategy::PatternBased => Self::partition_pattern_based(sparse),
        }
    }

    /// Partition tensor into fixed-size blocks
    fn partition_block_based(
        sparse: Box<dyn SparseTensor + Send + Sync>,
        block_size: (usize, usize),
    ) -> TorshResult<HashMap<RegionId, Box<dyn SparseTensor + Send + Sync>>> {
        let shape = sparse.shape();
        let (rows, cols) = (shape.dims()[0], shape.dims()[1]);
        let (block_rows, block_cols) = block_size;

        let mut regions = HashMap::new();
        let coo = sparse.to_coo()?;
        let triplets = coo.triplets();

        // Group triplets by blocks
        let mut block_triplets: BlockTriplets = HashMap::new();

        for (row, col, val) in triplets {
            let block_row = row / block_rows;
            let block_col = col / block_cols;
            block_triplets
                .entry((block_row, block_col))
                .or_default()
                .push((row % block_rows, col % block_cols, val));
        }

        // Create sparse tensors for each non-empty block
        for ((block_row, block_col), triplets) in block_triplets {
            let row_start = block_row * block_rows;
            let col_start = block_col * block_cols;
            let actual_rows = std::cmp::min(block_rows, rows - row_start);
            let actual_cols = std::cmp::min(block_cols, cols - col_start);

            let region_id = RegionId::new(row_start, col_start, actual_rows, actual_cols);

            // Create COO tensor for this block
            let (block_rows_vec, block_cols_vec, block_vals): (Vec<_>, Vec<_>, Vec<_>) =
                triplets.into_iter().fold(
                    (Vec::new(), Vec::new(), Vec::new()),
                    |(mut rows, mut cols, mut vals), (r, c, v)| {
                        rows.push(r);
                        cols.push(c);
                        vals.push(v);
                        (rows, cols, vals)
                    },
                );

            let block_shape = Shape::new(vec![actual_rows, actual_cols]);
            let block_coo =
                CooTensor::new(block_rows_vec, block_cols_vec, block_vals, block_shape)?;

            // Choose best format for this block
            let block_tensor = Self::select_optimal_format_for_block(&block_coo)?;
            regions.insert(region_id, block_tensor);
        }

        Ok(regions)
    }

    /// Partition tensor based on density patterns
    fn partition_density_based(
        sparse: Box<dyn SparseTensor + Send + Sync>,
        density_threshold: f32,
    ) -> TorshResult<HashMap<RegionId, Box<dyn SparseTensor + Send + Sync>>> {
        // For simplicity, we'll use a grid-based approach and check density
        let block_size = (64, 64); // Default block size for density analysis
        let mut regions = Self::partition_block_based(sparse, block_size)?;

        // Re-format regions based on their density
        let mut optimized_regions = HashMap::new();
        for (region_id, tensor) in regions.drain() {
            let density = 1.0 - tensor.sparsity();
            let optimized_tensor = if density > density_threshold {
                // High density: use format optimized for dense operations
                let coo = tensor.to_coo()?;
                Box::new(CsrTensor::from_coo(&coo)?) as Box<dyn SparseTensor + Send + Sync>
            } else {
                // Low density: keep in COO for flexibility
                Box::new(tensor.to_coo()?) as Box<dyn SparseTensor + Send + Sync>
            };
            optimized_regions.insert(region_id, optimized_tensor);
        }

        Ok(optimized_regions)
    }

    /// Partition tensor based on structural patterns
    fn partition_pattern_based(
        sparse: Box<dyn SparseTensor + Send + Sync>,
    ) -> TorshResult<HashMap<RegionId, Box<dyn SparseTensor + Send + Sync>>> {
        let coo = sparse.to_coo()?;
        let triplets = coo.triplets();
        let shape = sparse.shape();

        // Analyze patterns: diagonal, block diagonal, banded, etc.
        let pattern = Self::analyze_sparsity_pattern(&triplets, shape)?;

        match pattern {
            SparsityPattern::Diagonal => {
                // Use DIA format for the entire matrix
                let mut regions = HashMap::new();
                let region_id = RegionId::new(0, 0, shape.dims()[0], shape.dims()[1]);
                let dia_tensor = DiaTensor::from_coo(&coo)?;
                regions.insert(
                    region_id,
                    Box::new(dia_tensor) as Box<dyn SparseTensor + Send + Sync>,
                );
                Ok(regions)
            }
            SparsityPattern::BlockDiagonal { block_size } => {
                // Partition into diagonal blocks
                Self::partition_block_based(sparse, block_size)
            }
            SparsityPattern::Banded { bandwidth: _ } => {
                // Use specialized banded format or ELL
                let mut regions = HashMap::new();
                let region_id = RegionId::new(0, 0, shape.dims()[0], shape.dims()[1]);
                let ell_tensor = EllTensor::from_coo(&coo)?;
                regions.insert(
                    region_id,
                    Box::new(ell_tensor) as Box<dyn SparseTensor + Send + Sync>,
                );
                Ok(regions)
            }
            SparsityPattern::Random => {
                // Use block-based partitioning with optimal format selection
                Self::partition_block_based(sparse, (32, 32))
            }
        }
    }

    /// Analyze the sparsity pattern of triplets
    pub fn analyze_sparsity_pattern(
        triplets: &[(usize, usize, f32)],
        shape: &Shape,
    ) -> TorshResult<SparsityPattern> {
        let (rows, cols) = (shape.dims()[0], shape.dims()[1]);

        // Check for diagonal pattern
        let diagonal_count = triplets.iter().filter(|(r, c, _)| r == c).count();
        let diagonal_ratio = diagonal_count as f32 / triplets.len() as f32;

        if diagonal_ratio > 0.8 {
            return Ok(SparsityPattern::Diagonal);
        }

        // Check for banded pattern
        let max_bandwidth = triplets
            .iter()
            .map(|(r, c, _)| (*r as i32 - *c as i32).unsigned_abs() as usize)
            .max()
            .unwrap_or(0);

        let effective_bandwidth = std::cmp::min(max_bandwidth, std::cmp::min(rows, cols) / 4);

        if effective_bandwidth < std::cmp::min(rows, cols) / 8 {
            return Ok(SparsityPattern::Banded {
                bandwidth: effective_bandwidth,
            });
        }

        // Check for block diagonal pattern
        // Simple heuristic: if most non-zeros are within small blocks along diagonal
        let block_size = 16;
        let mut block_diagonal_count = 0;

        for (r, c, _) in triplets {
            let block_r = r / block_size;
            let block_c = c / block_size;
            if block_r == block_c {
                block_diagonal_count += 1;
            }
        }

        let block_diagonal_ratio = block_diagonal_count as f32 / triplets.len() as f32;

        if block_diagonal_ratio > 0.6 {
            return Ok(SparsityPattern::BlockDiagonal {
                block_size: (block_size, block_size),
            });
        }

        Ok(SparsityPattern::Random)
    }

    /// Select optimal sparse format for a block based on its characteristics
    fn select_optimal_format_for_block(
        coo: &CooTensor,
    ) -> TorshResult<Box<dyn SparseTensor + Send + Sync>> {
        let shape = coo.shape();
        let nnz = coo.nnz();
        let total_elements = shape.numel();
        let density = nnz as f32 / total_elements as f32;

        // Decision tree for format selection
        if density > 0.1 {
            // High density: use CSR for efficient matrix operations
            Ok(Box::new(CsrTensor::from_coo(coo)?))
        } else if nnz < 100 {
            // Very sparse: stay with COO for simplicity
            Ok(Box::new(coo.clone()))
        } else {
            // Medium sparsity: use CSR for general operations
            Ok(Box::new(CsrTensor::from_coo(coo)?))
        }
    }

    /// Validate that regions don't overlap and fit within the tensor shape
    fn validate_regions(
        regions: &HashMap<RegionId, Box<dyn SparseTensor + Send + Sync>>,
        shape: &Shape,
    ) -> TorshResult<()> {
        let (total_rows, total_cols) = (shape.dims()[0], shape.dims()[1]);

        for (region_id, tensor) in regions {
            // Check bounds
            if region_id.row_start + region_id.rows > total_rows
                || region_id.col_start + region_id.cols > total_cols
            {
                return Err(TorshError::InvalidArgument(
                    "Region extends beyond tensor bounds".to_string(),
                ));
            }

            // Check tensor shape matches region
            let tensor_shape = tensor.shape();
            if tensor_shape.dims() != [region_id.rows, region_id.cols] {
                return Err(TorshError::InvalidArgument(
                    "Region tensor shape doesn't match region dimensions".to_string(),
                ));
            }
        }

        // Check for overlapping regions
        let region_ids: Vec<&RegionId> = regions.keys().collect();
        for i in 0..region_ids.len() {
            for j in (i + 1)..region_ids.len() {
                let region1 = region_ids[i];
                let region2 = region_ids[j];

                // Check if regions overlap
                let r1_end_row = region1.row_start + region1.rows;
                let r1_end_col = region1.col_start + region1.cols;
                let r2_end_row = region2.row_start + region2.rows;
                let r2_end_col = region2.col_start + region2.cols;

                // Regions overlap if they intersect in both row and column dimensions
                let rows_overlap =
                    !(r1_end_row <= region2.row_start || r2_end_row <= region1.row_start);
                let cols_overlap =
                    !(r1_end_col <= region2.col_start || r2_end_col <= region1.col_start);

                if rows_overlap && cols_overlap {
                    return Err(TorshError::InvalidArgument(format!(
                        "Regions overlap: [{}, {}, {}, {}] and [{}, {}, {}, {}]",
                        region1.row_start,
                        region1.col_start,
                        region1.rows,
                        region1.cols,
                        region2.row_start,
                        region2.col_start,
                        region2.rows,
                        region2.cols
                    )));
                }
            }
        }

        Ok(())
    }
}

impl SparseTensor for HybridTensor {
    fn format(&self) -> SparseFormat {
        SparseFormat::Coo // Hybrid doesn't have a single format
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn dtype(&self) -> torsh_core::DType {
        torsh_core::DType::F32 // Assume f32 for now
    }

    fn device(&self) -> torsh_core::device::DeviceType {
        torsh_core::device::DeviceType::Cpu // Assume CPU for now
    }

    fn nnz(&self) -> usize {
        self.nnz
    }

    fn to_dense(&self) -> TorshResult<Tensor> {
        use torsh_tensor::creation::zeros;

        let dense = zeros::<f32>(&[self.shape.dims()[0], self.shape.dims()[1]])?;

        for (region_id, tensor) in &self.regions {
            let region_dense = tensor.to_dense()?;

            // Copy region data to the appropriate location in the dense tensor
            for i in 0..region_id.rows {
                for j in 0..region_id.cols {
                    let global_row = region_id.row_start + i;
                    let global_col = region_id.col_start + j;
                    let value = region_dense.get(&[i, j])?;
                    dense.set(&[global_row, global_col], value)?;
                }
            }
        }

        Ok(dense)
    }

    fn to_coo(&self) -> TorshResult<CooTensor> {
        let mut all_row_indices = Vec::new();
        let mut all_col_indices = Vec::new();
        let mut all_values = Vec::new();

        for (region_id, tensor) in &self.regions {
            let region_coo = tensor.to_coo()?;
            let triplets = region_coo.triplets();

            for (row, col, val) in triplets {
                all_row_indices.push(region_id.row_start + row);
                all_col_indices.push(region_id.col_start + col);
                all_values.push(val);
            }
        }

        CooTensor::new(
            all_row_indices,
            all_col_indices,
            all_values,
            self.shape.clone(),
        )
    }

    fn to_csr(&self) -> TorshResult<CsrTensor> {
        let coo = self.to_coo()?;
        CsrTensor::from_coo(&coo)
    }

    fn to_csc(&self) -> TorshResult<CscTensor> {
        let coo = self.to_coo()?;
        CscTensor::from_coo(&coo)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Strategy for partitioning a sparse tensor
#[derive(Debug, Clone)]
pub enum PartitionStrategy {
    /// Partition into fixed-size blocks
    BlockBased { block_size: (usize, usize) },
    /// Partition based on density thresholds
    DensityBased { threshold: f32 },
    /// Partition based on detected sparsity patterns
    PatternBased,
}

/// Detected sparsity patterns
#[derive(Debug, Clone)]
pub enum SparsityPattern {
    /// Diagonal matrix
    Diagonal,
    /// Block diagonal matrix
    BlockDiagonal { block_size: (usize, usize) },
    /// Banded matrix
    Banded { bandwidth: usize },
    /// Random sparsity pattern
    Random,
}

/// Automatically select the best sparse format for a given tensor
pub fn auto_select_format(dense: &Tensor, threshold: f32) -> TorshResult<SparseFormat> {
    let shape = dense.shape();
    if shape.ndim() != 2 {
        return Err(TorshError::InvalidArgument(
            "Can only select format for 2D tensors".to_string(),
        ));
    }

    let (rows, cols) = (shape.dims()[0], shape.dims()[1]);
    let total_elements = rows * cols;

    // Count non-zero elements
    let mut nnz = 0;
    let mut diagonal_nnz = 0;
    let mut max_bandwidth = 0;

    for i in 0..rows {
        for j in 0..cols {
            let val = dense.get(&[i, j])?;
            if val.abs() > threshold {
                nnz += 1;
                if i == j {
                    diagonal_nnz += 1;
                }
                max_bandwidth =
                    std::cmp::max(max_bandwidth, (i as i32 - j as i32).unsigned_abs() as usize);
            }
        }
    }

    let density = nnz as f32 / total_elements as f32;
    let diagonal_ratio = diagonal_nnz as f32 / nnz as f32;

    // Decision tree for format selection
    if diagonal_ratio > 0.8 {
        Ok(SparseFormat::Dia)
    } else if density > 0.1 {
        Ok(SparseFormat::Csr) // High density, good for matrix ops
    } else if max_bandwidth < std::cmp::min(rows, cols) / 8 {
        Ok(SparseFormat::Ell) // Banded structure
    } else if nnz < 1000 {
        Ok(SparseFormat::Coo) // Small, use flexible format
    } else if cols > rows * 2 {
        Ok(SparseFormat::Csc) // Tall and narrow, column-major
    } else {
        Ok(SparseFormat::Csr) // Default to row-major
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use torsh_tensor::creation::{eye, zeros};

    #[test]
    fn test_auto_format_selection() {
        // Test diagonal matrix
        let diagonal = eye::<f32>(10).unwrap();
        let format = auto_select_format(&diagonal, 0.0).unwrap();
        assert_eq!(format, SparseFormat::Dia);

        // Test dense-ish matrix
        let dense_ish = zeros::<f32>(&[5, 5]).unwrap();
        for i in 0..5 {
            for j in 0..5 {
                if (i + j) % 2 == 0 {
                    dense_ish.set(&[i, j], 1.0).unwrap();
                }
            }
        }
        let format = auto_select_format(&dense_ish, 0.0).unwrap();
        // Should be CSR due to high density
        assert_eq!(format, SparseFormat::Csr);
    }

    #[test]
    fn test_sparsity_pattern_analysis() {
        // Test diagonal pattern
        let triplets = vec![(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)];
        let shape = Shape::new(vec![3, 3]);
        let pattern = HybridTensor::analyze_sparsity_pattern(&triplets, &shape).unwrap();
        matches!(pattern, SparsityPattern::Diagonal);
    }

    #[test]
    fn test_hybrid_tensor_creation() {
        // Create a simple hybrid tensor with one region
        let mut regions = HashMap::new();
        let coo = CooTensor::new(
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 2.0],
            Shape::new(vec![2, 2]),
        )
        .unwrap();

        let region_id = RegionId::new(0, 0, 2, 2);
        regions.insert(
            region_id,
            Box::new(coo) as Box<dyn SparseTensor + Send + Sync>,
        );

        let hybrid = HybridTensor::new(regions, Shape::new(vec![2, 2])).unwrap();
        assert_eq!(hybrid.nnz(), 2);
        assert_eq!(hybrid.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_hybrid_tensor_to_dense() {
        // Create hybrid tensor and convert to dense
        let mut regions = HashMap::new();
        let coo = CooTensor::new(
            vec![0, 1],
            vec![0, 1],
            vec![3.0, 4.0],
            Shape::new(vec![2, 2]),
        )
        .unwrap();

        let region_id = RegionId::new(0, 0, 2, 2);
        regions.insert(
            region_id,
            Box::new(coo) as Box<dyn SparseTensor + Send + Sync>,
        );

        let hybrid = HybridTensor::new(regions, Shape::new(vec![2, 2])).unwrap();
        let dense = hybrid.to_dense().unwrap();

        assert_relative_eq!(dense.get(&[0, 0]).unwrap(), 3.0);
        assert_relative_eq!(dense.get(&[1, 1]).unwrap(), 4.0);
        assert_relative_eq!(dense.get(&[0, 1]).unwrap(), 0.0);
        assert_relative_eq!(dense.get(&[1, 0]).unwrap(), 0.0);
    }
}
