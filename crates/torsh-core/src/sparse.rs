// Sparse tensor metadata and storage formats for ToRSh Core
// Supports COO (Coordinate), CSR (Compressed Sparse Row), and CSC (Compressed Sparse Column) formats

use crate::dtype::DType;
use crate::error::TorshError;
use crate::shape::Shape;

use std::fmt;
use std::sync::Arc;

/// Sparse tensor storage formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SparseFormat {
    /// Coordinate format: stores (indices, values) pairs
    /// Memory: O(nnz) for indices + values
    /// Good for: construction, element access
    COO,

    /// Compressed Sparse Row format
    /// Memory: O(nnz) values + O(nnz) column indices + O(rows+1) row pointers
    /// Good for: matrix-vector multiplication, row access
    CSR,

    /// Compressed Sparse Column format
    /// Memory: O(nnz) values + O(nnz) row indices + O(cols+1) column pointers
    /// Good for: matrix-vector multiplication (transposed), column access
    CSC,

    /// Block Sparse Row format for structured sparsity
    /// Good for: GPU acceleration, structured pruning
    BSR,

    /// Diagonal format for diagonal and band matrices
    /// Good for: diagonal matrices, finite difference operators
    DIA,

    /// ELLPack format for GPU-optimized sparse operations
    /// Good for: GPU kernels with regular sparsity patterns
    ELL,
}

impl SparseFormat {
    /// Get human-readable name
    pub fn name(self) -> &'static str {
        match self {
            Self::COO => "COO",
            Self::CSR => "CSR",
            Self::CSC => "CSC",
            Self::BSR => "BSR",
            Self::DIA => "DIA",
            Self::ELL => "ELL",
        }
    }

    /// Check if format supports efficient row access
    pub fn supports_row_access(self) -> bool {
        matches!(self, Self::CSR | Self::BSR)
    }

    /// Check if format supports efficient column access
    pub fn supports_column_access(self) -> bool {
        matches!(self, Self::CSC)
    }

    /// Check if format is suitable for GPU operations
    pub fn is_gpu_friendly(self) -> bool {
        matches!(self, Self::CSR | Self::ELL | Self::BSR)
    }
}

impl fmt::Display for SparseFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Sparse tensor metadata containing format and structural information
#[derive(Debug, Clone)]
pub struct SparseMetadata {
    /// Storage format
    format: SparseFormat,

    /// Number of non-zero elements
    nnz: usize,

    /// Sparsity ratio (0.0 = dense, 1.0 = all zeros)
    sparsity: f32,

    /// Whether indices are sorted
    indices_sorted: bool,

    /// Whether duplicates have been summed
    duplicates_summed: bool,

    /// Block size for BSR format
    block_size: Option<(usize, usize)>,

    /// Number of diagonals for DIA format
    num_diagonals: Option<usize>,

    /// ELLPack row width
    ell_width: Option<usize>,

    /// Compression statistics
    compression_stats: CompressionStats,
}

/// Statistics about sparse tensor compression efficiency
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Theoretical dense size in bytes
    dense_size_bytes: usize,

    /// Actual sparse storage size in bytes
    sparse_size_bytes: usize,

    /// Compression ratio (dense_size / sparse_size)
    compression_ratio: f32,

    /// Memory overhead from indices storage
    #[allow(dead_code)] // Index overhead tracking - future implementation
    index_overhead_bytes: usize,
}

impl SparseMetadata {
    /// Create new sparse metadata
    pub fn new(
        format: SparseFormat,
        nnz: usize,
        total_elements: usize,
        dense_size_bytes: usize,
        sparse_size_bytes: usize,
    ) -> Self {
        let sparsity = 1.0 - (nnz as f32 / total_elements as f32);
        let compression_ratio = dense_size_bytes as f32 / sparse_size_bytes as f32;

        Self {
            format,
            nnz,
            sparsity,
            indices_sorted: false,
            duplicates_summed: false,
            block_size: None,
            num_diagonals: None,
            ell_width: None,
            compression_stats: CompressionStats {
                dense_size_bytes,
                sparse_size_bytes,
                compression_ratio,
                index_overhead_bytes: sparse_size_bytes - (nnz * 4), // Rough estimate
            },
        }
    }

    /// Get storage format
    pub fn format(&self) -> SparseFormat {
        self.format
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// Get sparsity ratio (fraction of zero elements)
    pub fn sparsity(&self) -> f32 {
        self.sparsity
    }

    /// Get density ratio (fraction of non-zero elements)
    pub fn density(&self) -> f32 {
        1.0 - self.sparsity
    }

    /// Check if indices are sorted
    pub fn indices_sorted(&self) -> bool {
        self.indices_sorted
    }

    /// Mark indices as sorted
    pub fn set_indices_sorted(&mut self, sorted: bool) {
        self.indices_sorted = sorted;
    }

    /// Check if duplicates have been summed
    pub fn duplicates_summed(&self) -> bool {
        self.duplicates_summed
    }

    /// Mark duplicates as summed
    pub fn set_duplicates_summed(&mut self, summed: bool) {
        self.duplicates_summed = summed;
    }

    /// Get block size for BSR format
    pub fn block_size(&self) -> Option<(usize, usize)> {
        self.block_size
    }

    /// Set block size for BSR format
    pub fn set_block_size(&mut self, size: (usize, usize)) {
        self.block_size = Some(size);
    }

    /// Get compression statistics
    pub fn compression_stats(&self) -> &CompressionStats {
        &self.compression_stats
    }

    /// Check if sparse representation is beneficial
    pub fn is_beneficial(&self) -> bool {
        self.compression_stats.compression_ratio > 1.2 // At least 20% savings
    }

    /// Estimate memory savings compared to dense representation
    pub fn memory_savings_bytes(&self) -> i64 {
        self.compression_stats.dense_size_bytes as i64
            - self.compression_stats.sparse_size_bytes as i64
    }

    /// Get format-specific information as string
    pub fn format_info(&self) -> String {
        match self.format {
            SparseFormat::BSR => {
                if let Some((bm, bn)) = self.block_size {
                    format!("BSR({}x{})", bm, bn)
                } else {
                    "BSR".to_string()
                }
            }
            SparseFormat::DIA => {
                if let Some(ndiag) = self.num_diagonals {
                    format!("DIA({})", ndiag)
                } else {
                    "DIA".to_string()
                }
            }
            SparseFormat::ELL => {
                if let Some(width) = self.ell_width {
                    format!("ELL({})", width)
                } else {
                    "ELL".to_string()
                }
            }
            _ => self.format.name().to_string(),
        }
    }
}

impl fmt::Display for SparseMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SparseMetadata({}, nnz={}, sparsity={:.2}%, compression={:.1}x)",
            self.format_info(),
            self.nnz,
            self.sparsity * 100.0,
            self.compression_stats.compression_ratio
        )
    }
}

/// COO (Coordinate) format sparse tensor indices
#[derive(Debug, Clone)]
pub struct CooIndices {
    /// Row indices (length = nnz)
    pub rows: Vec<usize>,

    /// Column indices (length = nnz)
    pub cols: Vec<usize>,

    /// Higher dimension indices for tensors with ndim > 2
    pub extra_dims: Vec<Vec<usize>>,
}

impl CooIndices {
    /// Create new COO indices for 2D tensor
    pub fn new_2d(rows: Vec<usize>, cols: Vec<usize>) -> Self {
        assert_eq!(
            rows.len(),
            cols.len(),
            "Row and column indices must have same length"
        );

        Self {
            rows,
            cols,
            extra_dims: Vec::new(),
        }
    }

    /// Create new COO indices for N-D tensor
    pub fn new_nd(indices: Vec<Vec<usize>>) -> Self {
        let nnz = indices.first().map_or(0, |dim| dim.len());

        // Validate all dimensions have same length
        for (i, dim_indices) in indices.iter().enumerate() {
            assert_eq!(
                dim_indices.len(),
                nnz,
                "Dimension {} indices length mismatch: expected {}, got {}",
                i,
                nnz,
                dim_indices.len()
            );
        }

        if indices.len() < 2 {
            panic!("N-D tensor must have at least 2 dimensions");
        }

        Self {
            rows: indices[0].clone(),
            cols: indices[1].clone(),
            extra_dims: if indices.len() > 2 {
                indices[2..].to_vec()
            } else {
                Vec::new()
            },
        }
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.rows.len()
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        2 + self.extra_dims.len()
    }

    /// Check if indices are sorted in lexicographic order
    pub fn is_sorted(&self) -> bool {
        for i in 1..self.rows.len() {
            if self.rows[i] < self.rows[i - 1] {
                return false;
            }
            if self.rows[i] == self.rows[i - 1] && self.cols[i] < self.cols[i - 1] {
                return false;
            }
        }
        true
    }

    /// Sort indices in lexicographic order, returning permutation
    pub fn sort(&mut self) -> Vec<usize> {
        let mut perm: Vec<usize> = (0..self.nnz()).collect();

        // Sort by lexicographic order
        perm.sort_by(|&a, &b| {
            // Compare rows first
            match self.rows[a].cmp(&self.rows[b]) {
                std::cmp::Ordering::Equal => {
                    // Rows equal, compare columns
                    match self.cols[a].cmp(&self.cols[b]) {
                        std::cmp::Ordering::Equal => {
                            // Compare extra dimensions
                            for dim_indices in &self.extra_dims {
                                match dim_indices[a].cmp(&dim_indices[b]) {
                                    std::cmp::Ordering::Equal => continue,
                                    other => return other,
                                }
                            }
                            std::cmp::Ordering::Equal
                        }
                        other => other,
                    }
                }
                other => other,
            }
        });

        // Apply permutation
        let orig_rows = self.rows.clone();
        let orig_cols = self.cols.clone();
        let orig_extra: Vec<_> = self.extra_dims.clone();

        for (i, &p) in perm.iter().enumerate() {
            self.rows[i] = orig_rows[p];
            self.cols[i] = orig_cols[p];
            for (dim_idx, orig_dim) in orig_extra.iter().enumerate() {
                self.extra_dims[dim_idx][i] = orig_dim[p];
            }
        }

        perm
    }
}

/// CSR (Compressed Sparse Row) format indices
#[derive(Debug, Clone)]
pub struct CsrIndices {
    /// Row pointers (length = nrows + 1)
    pub row_ptrs: Vec<usize>,

    /// Column indices (length = nnz)
    pub col_indices: Vec<usize>,
}

impl CsrIndices {
    /// Create new CSR indices
    pub fn new(row_ptrs: Vec<usize>, col_indices: Vec<usize>) -> Self {
        // Validate structure
        let nnz = col_indices.len();
        let _nrows = row_ptrs.len().saturating_sub(1);

        assert_eq!(
            *row_ptrs.last().unwrap_or(&0),
            nnz,
            "Last row pointer must equal nnz"
        );

        // Validate row pointers are non-decreasing
        for i in 1..row_ptrs.len() {
            assert!(
                row_ptrs[i] >= row_ptrs[i - 1],
                "Row pointers must be non-decreasing"
            );
        }

        Self {
            row_ptrs,
            col_indices,
        }
    }

    /// Convert from COO format
    pub fn from_coo(coo: &CooIndices, nrows: usize) -> Self {
        let _nnz = coo.nnz();
        let mut row_ptrs = vec![0; nrows + 1];

        // Count elements per row
        for &row in &coo.rows {
            if row < nrows {
                row_ptrs[row + 1] += 1;
            }
        }

        // Convert counts to cumulative sums
        for i in 1..=nrows {
            row_ptrs[i] += row_ptrs[i - 1];
        }

        // Create column indices array (assume COO is already sorted)
        let col_indices = coo.cols.clone();

        Self::new(row_ptrs, col_indices)
    }

    /// Get number of rows
    pub fn nrows(&self) -> usize {
        self.row_ptrs.len().saturating_sub(1)
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.col_indices.len()
    }

    /// Get range of column indices for a row
    pub fn row_range(&self, row: usize) -> Option<std::ops::Range<usize>> {
        if row >= self.nrows() {
            return None;
        }
        Some(self.row_ptrs[row]..self.row_ptrs[row + 1])
    }
}

/// Sparse tensor storage trait
pub trait SparseStorage: Send + Sync + std::fmt::Debug {
    /// Get sparse metadata
    fn metadata(&self) -> &SparseMetadata;

    /// Get element count
    fn nnz(&self) -> usize {
        self.metadata().nnz()
    }

    /// Get storage format
    fn format(&self) -> SparseFormat {
        self.metadata().format()
    }

    /// Check if representation is beneficial vs dense
    fn is_beneficial(&self) -> bool {
        self.metadata().is_beneficial()
    }

    /// Convert to COO format if possible
    fn to_coo(&self) -> Result<Arc<dyn SparseStorage>, TorshError>;

    /// Convert to CSR format if possible
    fn to_csr(&self) -> Result<Arc<dyn SparseStorage>, TorshError>;

    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize;
}

/// COO format sparse storage
#[derive(Debug)]
pub struct CooStorage {
    metadata: SparseMetadata,
    indices: CooIndices,
    values: Vec<u8>, // Raw bytes for type-erased storage
    dtype: DType,
    shape: Shape,
}

impl CooStorage {
    /// Create new COO storage
    pub fn new(
        indices: CooIndices,
        values: Vec<u8>,
        dtype: DType,
        shape: Shape,
    ) -> Result<Self, TorshError> {
        let nnz = indices.nnz();
        let expected_value_size = nnz * dtype.size();

        if values.len() != expected_value_size {
            return Err(TorshError::InvalidArgument(format!(
                "Value buffer size mismatch: expected {}, got {}",
                expected_value_size,
                values.len()
            )));
        }

        let total_elements: usize = shape.dims().iter().product();
        let dense_size = total_elements * dtype.size();
        let sparse_size = values.len() + indices.rows.len() * 8 + indices.cols.len() * 8; // Rough estimate

        let metadata = SparseMetadata::new(
            SparseFormat::COO,
            nnz,
            total_elements,
            dense_size,
            sparse_size,
        );

        Ok(Self {
            metadata,
            indices,
            values,
            dtype,
            shape,
        })
    }

    /// Get indices reference
    pub fn indices(&self) -> &CooIndices {
        &self.indices
    }

    /// Get mutable indices reference
    pub fn indices_mut(&mut self) -> &mut CooIndices {
        &mut self.indices
    }

    /// Get values as raw bytes
    pub fn values_bytes(&self) -> &[u8] {
        &self.values
    }

    /// Get data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}

impl SparseStorage for CooStorage {
    fn metadata(&self) -> &SparseMetadata {
        &self.metadata
    }

    fn to_coo(&self) -> Result<Arc<dyn SparseStorage>, TorshError> {
        // Already COO format, return clone
        Ok(Arc::new(CooStorage {
            metadata: self.metadata.clone(),
            indices: self.indices.clone(),
            values: self.values.clone(),
            dtype: self.dtype,
            shape: self.shape.clone(),
        }))
    }

    fn to_csr(&self) -> Result<Arc<dyn SparseStorage>, TorshError> {
        if self.shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "CSR format only supports 2D tensors".to_string(),
            ));
        }

        let nrows = self.shape.dims()[0];
        let csr_indices = CsrIndices::from_coo(&self.indices, nrows);

        Ok(Arc::new(CsrStorage {
            metadata: {
                let mut meta = self.metadata.clone();
                meta.format = SparseFormat::CSR;
                meta
            },
            indices: csr_indices,
            values: self.values.clone(),
            dtype: self.dtype,
            shape: self.shape.clone(),
        }))
    }

    fn memory_usage(&self) -> usize {
        self.values.len()
            + self.indices.rows.len() * std::mem::size_of::<usize>()
            + self.indices.cols.len() * std::mem::size_of::<usize>()
            + self
                .indices
                .extra_dims
                .iter()
                .map(|dim| dim.len() * std::mem::size_of::<usize>())
                .sum::<usize>()
    }
}

/// CSR format sparse storage
#[derive(Debug)]
pub struct CsrStorage {
    metadata: SparseMetadata,
    indices: CsrIndices,
    values: Vec<u8>,
    dtype: DType,
    shape: Shape,
}

impl CsrStorage {
    /// Create new CSR storage
    pub fn new(
        indices: CsrIndices,
        values: Vec<u8>,
        dtype: DType,
        shape: Shape,
    ) -> Result<Self, TorshError> {
        if shape.ndim() != 2 {
            return Err(TorshError::InvalidArgument(
                "CSR format only supports 2D tensors".to_string(),
            ));
        }

        let nnz = indices.nnz();
        let expected_value_size = nnz * dtype.size();

        if values.len() != expected_value_size {
            return Err(TorshError::InvalidArgument(format!(
                "Value buffer size mismatch: expected {}, got {}",
                expected_value_size,
                values.len()
            )));
        }

        let total_elements: usize = shape.dims().iter().product();
        let dense_size = total_elements * dtype.size();
        let sparse_size = values.len() + indices.row_ptrs.len() * 8 + indices.col_indices.len() * 8;

        let metadata = SparseMetadata::new(
            SparseFormat::CSR,
            nnz,
            total_elements,
            dense_size,
            sparse_size,
        );

        Ok(Self {
            metadata,
            indices,
            values,
            dtype,
            shape,
        })
    }

    /// Get indices reference
    pub fn indices(&self) -> &CsrIndices {
        &self.indices
    }
}

impl SparseStorage for CsrStorage {
    fn metadata(&self) -> &SparseMetadata {
        &self.metadata
    }

    fn to_coo(&self) -> Result<Arc<dyn SparseStorage>, TorshError> {
        // Convert CSR back to COO
        let mut rows = Vec::with_capacity(self.nnz());
        let mut cols = Vec::with_capacity(self.nnz());

        for row in 0..self.indices.nrows() {
            let range = self.indices.row_range(row).unwrap();
            for col_idx in range {
                rows.push(row);
                cols.push(self.indices.col_indices[col_idx]);
            }
        }

        let coo_indices = CooIndices::new_2d(rows, cols);

        Ok(Arc::new(CooStorage {
            metadata: {
                let mut meta = self.metadata.clone();
                meta.format = SparseFormat::COO;
                meta
            },
            indices: coo_indices,
            values: self.values.clone(),
            dtype: self.dtype,
            shape: self.shape.clone(),
        }))
    }

    fn to_csr(&self) -> Result<Arc<dyn SparseStorage>, TorshError> {
        // Already CSR format
        Ok(Arc::new(CsrStorage {
            metadata: self.metadata.clone(),
            indices: self.indices.clone(),
            values: self.values.clone(),
            dtype: self.dtype,
            shape: self.shape.clone(),
        }))
    }

    fn memory_usage(&self) -> usize {
        self.values.len()
            + self.indices.row_ptrs.len() * std::mem::size_of::<usize>()
            + self.indices.col_indices.len() * std::mem::size_of::<usize>()
    }
}

/// Utilities for sparse tensor operations
pub mod utils {
    use super::*;

    /// Analyze sparsity patterns in dense data
    pub fn analyze_sparsity(data: &[f32], shape: &[usize]) -> SparseAnalysis {
        let total_elements = data.len();
        let mut nnz = 0;
        let mut pattern_info = PatternInfo::default();

        // Count non-zeros and analyze patterns
        for (idx, &value) in data.iter().enumerate() {
            if value != 0.0 {
                nnz += 1;
                pattern_info.update(idx, shape);
            }
        }

        let sparsity = 1.0 - (nnz as f32 / total_elements as f32);

        SparseAnalysis {
            sparsity,
            nnz,
            total_elements,
            pattern_info,
        }
    }

    /// Recommend optimal sparse format based on sparsity analysis
    pub fn recommend_format(analysis: &SparseAnalysis, shape: &[usize]) -> FormatRecommendation {
        let sparsity = analysis.sparsity;
        let nnz = analysis.nnz;

        // Simple heuristics for format recommendation
        if sparsity < 0.5 {
            return FormatRecommendation {
                format: None, // Dense is better
                reason: "Low sparsity, dense representation more efficient".to_string(),
                confidence: 0.9,
            };
        }

        if shape.len() == 2 {
            // 2D matrix
            let (nrows, ncols) = (shape[0], shape[1]);

            if analysis.pattern_info.has_structured_rows {
                return FormatRecommendation {
                    format: Some(SparseFormat::CSR),
                    reason: "Good row locality, CSR optimal for row-wise operations".to_string(),
                    confidence: 0.8,
                };
            }

            if analysis.pattern_info.has_structured_cols {
                return FormatRecommendation {
                    format: Some(SparseFormat::CSC),
                    reason: "Good column locality, CSC optimal for column-wise operations"
                        .to_string(),
                    confidence: 0.8,
                };
            }

            if nnz < (nrows + ncols) * 10 {
                return FormatRecommendation {
                    format: Some(SparseFormat::COO),
                    reason: "Very sparse matrix, COO has lowest overhead".to_string(),
                    confidence: 0.7,
                };
            }

            return FormatRecommendation {
                format: Some(SparseFormat::CSR),
                reason: "General 2D sparse matrix, CSR is default choice".to_string(),
                confidence: 0.6,
            };
        }

        // N-D tensor
        FormatRecommendation {
            format: Some(SparseFormat::COO),
            reason: "Multi-dimensional tensor, COO supports arbitrary dimensions".to_string(),
            confidence: 0.8,
        }
    }

    /// Convert dense data to optimal sparse format
    pub fn densify_to_sparse<T>(
        data: &[T],
        shape: &Shape,
        dtype: DType,
        threshold: Option<f64>,
    ) -> Result<Arc<dyn SparseStorage>, TorshError>
    where
        T: Clone + PartialEq + Into<f64> + Default,
    {
        let threshold = threshold.unwrap_or(1e-12);
        let zero = T::default();

        // Find non-zero elements
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (linear_idx, value) in data.iter().enumerate() {
            let abs_val = value.clone().into().abs();
            if abs_val > threshold && *value != zero {
                // Convert linear index to multi-dimensional indices
                let multi_idx = linear_to_multidim(linear_idx, shape.dims());
                indices.push(multi_idx);
                values.push(value.clone());
            }
        }

        if indices.is_empty() {
            return Err(TorshError::InvalidArgument(
                "No non-zero elements found".to_string(),
            ));
        }

        // Convert to bytes (simplified - real implementation would handle type properly)
        let value_bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| {
                let val_f64 = v.clone().into();
                val_f64.to_ne_bytes()
            })
            .collect();

        // Create COO indices
        let dims = shape.dims();
        match dims.len() {
            1 => {
                let rows: Vec<usize> = indices.iter().map(|idx| idx[0]).collect();
                let cols = vec![0; rows.len()]; // Dummy column for 1D
                let coo_indices = CooIndices::new_2d(rows, cols);
                CooStorage::new(coo_indices, value_bytes, dtype, shape.clone())
                    .map(|storage| Arc::new(storage) as Arc<dyn SparseStorage>)
            }
            2 => {
                let rows: Vec<usize> = indices.iter().map(|idx| idx[0]).collect();
                let cols: Vec<usize> = indices.iter().map(|idx| idx[1]).collect();
                let coo_indices = CooIndices::new_2d(rows, cols);
                CooStorage::new(coo_indices, value_bytes, dtype, shape.clone())
                    .map(|storage| Arc::new(storage) as Arc<dyn SparseStorage>)
            }
            _ => {
                let transposed_indices: Vec<Vec<usize>> = (0..dims.len())
                    .map(|dim| indices.iter().map(|idx| idx[dim]).collect())
                    .collect();
                let coo_indices = CooIndices::new_nd(transposed_indices);
                CooStorage::new(coo_indices, value_bytes, dtype, shape.clone())
                    .map(|storage| Arc::new(storage) as Arc<dyn SparseStorage>)
            }
        }
    }

    // Helper function to convert linear index to multi-dimensional
    fn linear_to_multidim(linear_idx: usize, shape: &[usize]) -> Vec<usize> {
        let mut result = Vec::with_capacity(shape.len());
        let mut remaining = linear_idx;

        for &dim_size in shape.iter().rev() {
            result.push(remaining % dim_size);
            remaining /= dim_size;
        }

        result.reverse();
        result
    }

    /// Analysis results for sparse data
    #[derive(Debug, Clone)]
    pub struct SparseAnalysis {
        pub sparsity: f32,
        pub nnz: usize,
        pub total_elements: usize,
        pub pattern_info: PatternInfo,
    }

    /// Information about sparsity patterns
    #[derive(Debug, Clone, Default)]
    pub struct PatternInfo {
        pub has_structured_rows: bool,
        pub has_structured_cols: bool,
        pub has_diagonal_structure: bool,
        pub has_block_structure: bool,
        pub block_size: Option<(usize, usize)>,
    }

    impl PatternInfo {
        fn update(&mut self, idx: usize, shape: &[usize]) {
            // Simplified pattern detection logic
            // Real implementation would be more sophisticated
            if shape.len() == 2 {
                let (_nrows, ncols) = (shape[0], shape[1]);
                let row = idx / ncols;
                let col = idx % ncols;

                // Check for diagonal elements
                if row == col {
                    self.has_diagonal_structure = true;
                }

                // Simple heuristics for structured patterns
                if row.is_multiple_of(4) && col.is_multiple_of(4) {
                    self.has_block_structure = true;
                    self.block_size = Some((4, 4));
                }
            }
        }
    }

    /// Format recommendation result
    #[derive(Debug, Clone)]
    pub struct FormatRecommendation {
        pub format: Option<SparseFormat>,
        pub reason: String,
        pub confidence: f32, // 0.0 - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Shape;

    #[test]
    fn test_sparse_metadata_creation() {
        let metadata = SparseMetadata::new(
            SparseFormat::COO,
            1000,  // nnz
            10000, // total elements
            40000, // dense size (10k * 4 bytes)
            8000,  // sparse size
        );

        assert_eq!(metadata.format(), SparseFormat::COO);
        assert_eq!(metadata.nnz(), 1000);
        assert_eq!(metadata.sparsity(), 0.9); // 90% sparse
        assert!(metadata.is_beneficial()); // 5x compression
    }

    #[test]
    fn test_coo_indices_creation() {
        let rows = vec![0, 1, 2, 1];
        let cols = vec![1, 0, 2, 2];

        let indices = CooIndices::new_2d(rows.clone(), cols.clone());

        assert_eq!(indices.nnz(), 4);
        assert_eq!(indices.ndim(), 2);
        assert_eq!(indices.rows, rows);
        assert_eq!(indices.cols, cols);
    }

    #[test]
    fn test_coo_indices_sorting() {
        let mut indices = CooIndices::new_2d(
            vec![2, 1, 0, 1], // rows
            vec![0, 2, 1, 0], // cols
        );

        assert!(!indices.is_sorted());

        let _perm = indices.sort();

        // After sorting: [(0,1), (1,0), (1,2), (2,0)]
        assert_eq!(indices.rows, vec![0, 1, 1, 2]);
        assert_eq!(indices.cols, vec![1, 0, 2, 0]);
        assert!(indices.is_sorted());
    }

    #[test]
    fn test_csr_from_coo() {
        let coo_indices = CooIndices::new_2d(
            vec![0, 0, 1, 2, 2], // rows
            vec![1, 2, 0, 1, 2], // cols
        );

        let csr_indices = CsrIndices::from_coo(&coo_indices, 3);

        assert_eq!(csr_indices.nrows(), 3);
        assert_eq!(csr_indices.nnz(), 5);
        assert_eq!(csr_indices.row_ptrs, vec![0, 2, 3, 5]);
        assert_eq!(csr_indices.col_indices, vec![1, 2, 0, 1, 2]);
    }

    #[test]
    fn test_coo_storage_creation() {
        let indices = CooIndices::new_2d(vec![0, 1], vec![1, 0]);
        let values = [1.0_f32.to_ne_bytes(), 2.0_f32.to_ne_bytes()].concat();
        let shape = Shape::new(vec![2, 2]);

        let storage = CooStorage::new(indices, values, DType::F32, shape).unwrap();

        assert_eq!(storage.nnz(), 2);
        assert_eq!(storage.format(), SparseFormat::COO);
        assert_eq!(storage.dtype(), DType::F32);
    }

    #[test]
    fn test_format_conversion() {
        let indices = CooIndices::new_2d(vec![0, 1], vec![1, 0]);
        let values = [1.0_f32.to_ne_bytes(), 2.0_f32.to_ne_bytes()].concat();
        let shape = Shape::new(vec![2, 2]);

        let coo_storage = CooStorage::new(indices, values, DType::F32, shape).unwrap();

        // Convert COO -> CSR
        let csr_storage = coo_storage.to_csr().unwrap();
        assert_eq!(csr_storage.format(), SparseFormat::CSR);

        // Convert CSR -> COO
        let coo_again = csr_storage.to_coo().unwrap();
        assert_eq!(coo_again.format(), SparseFormat::COO);
    }

    #[test]
    fn test_sparsity_analysis() {
        let data = vec![0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let shape = vec![3, 3];

        let analysis = utils::analyze_sparsity(&data, &shape);

        assert_eq!(analysis.nnz, 3);
        assert_eq!(analysis.total_elements, 9);
        assert!((analysis.sparsity - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_format_recommendation() {
        // High sparsity case
        let analysis = utils::SparseAnalysis {
            sparsity: 0.9,
            nnz: 100,
            total_elements: 1000,
            pattern_info: utils::PatternInfo::default(),
        };

        let shape = vec![100, 10];
        let recommendation = utils::recommend_format(&analysis, &shape);

        assert!(recommendation.format.is_some());
        assert!(recommendation.confidence > 0.0);
    }
}
