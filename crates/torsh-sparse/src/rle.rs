//! Run-Length Encoded (RLE) sparse tensor format
//!
//! This format efficiently stores sparse tensors with consecutive non-zero elements
//! by using run-length encoding. It's particularly efficient for patterns with
//! consecutive sequences of non-zero values.

use crate::{CooTensor, CscTensor, CsrTensor, SparseFormat, SparseTensor, TorshResult};
use std::collections::HashMap;
use torsh_core::{device::DeviceType, DType, Shape, TorshError};
use torsh_tensor::Tensor;

/// Run-Length Encoded sparse tensor
///
/// This format stores consecutive non-zero elements as runs, where each run
/// contains the starting position, length, and values for that sequence.
#[derive(Debug, Clone)]
pub struct RleTensor {
    /// Runs of consecutive non-zero elements
    runs: Vec<Run>,
    /// Shape of the tensor
    shape: Shape,
    /// Data type
    dtype: DType,
    /// Device type
    device: DeviceType,
    /// Number of non-zero elements
    nnz: usize,
}

/// A run of consecutive non-zero elements
#[derive(Debug, Clone)]
pub struct Run {
    /// Starting row index
    pub row: usize,
    /// Starting column index
    pub col: usize,
    /// Length of the run
    pub length: usize,
    /// Values in the run
    pub values: Vec<f32>,
}

impl Run {
    /// Create a new run
    pub fn new(row: usize, col: usize, length: usize, values: Vec<f32>) -> Self {
        Self {
            row,
            col,
            length,
            values,
        }
    }

    /// Get the ending position of the run
    pub fn end_pos(&self) -> (usize, usize) {
        // For now, assume runs are horizontal (same row)
        (self.row, self.col + self.length - 1)
    }

    /// Check if a position is contained in this run
    pub fn contains(&self, row: usize, col: usize) -> bool {
        if row != self.row {
            return false;
        }
        col >= self.col && col < self.col + self.length
    }

    /// Get the value at a specific position within the run
    pub fn get_value(&self, row: usize, col: usize) -> Option<f32> {
        if self.contains(row, col) {
            let offset = col - self.col;
            self.values.get(offset).copied()
        } else {
            None
        }
    }
}

impl RleTensor {
    /// Create a new RLE tensor from runs
    pub fn new(
        runs: Vec<Run>,
        shape: Shape,
        dtype: DType,
        device: DeviceType,
    ) -> TorshResult<Self> {
        let nnz = runs.iter().map(|run| run.length).sum();

        // Validate runs don't overlap and fit within shape
        Self::validate_runs(&runs, &shape)?;

        Ok(Self {
            runs,
            shape,
            dtype,
            device,
            nnz,
        })
    }

    /// Create RLE tensor from dense tensor
    pub fn from_dense(dense: &Tensor, threshold: f32) -> TorshResult<Self> {
        let shape = dense.shape();
        let dtype = dense.dtype();
        let device = dense.device();

        if shape.dims().len() != 2 {
            return Err(TorshError::InvalidShape(
                "RLE format only supports 2D tensors".to_string(),
            ));
        }

        let (rows, cols) = (shape.dims()[0], shape.dims()[1]);
        let data = dense.to_vec()?;

        let mut runs = Vec::new();

        // Extract runs row by row
        for row in 0..rows {
            let mut col = 0;
            while col < cols {
                let idx = row * cols + col;
                let value = data[idx];

                if value.abs() > threshold {
                    // Start of a run
                    let run_start = col;
                    let mut run_values = vec![value];
                    col += 1;

                    // Continue the run
                    while col < cols {
                        let next_idx = row * cols + col;
                        let next_value = data[next_idx];

                        if next_value.abs() > threshold {
                            run_values.push(next_value);
                            col += 1;
                        } else {
                            break;
                        }
                    }

                    runs.push(Run::new(row, run_start, run_values.len(), run_values));
                } else {
                    col += 1;
                }
            }
        }

        Self::new(runs, shape, dtype, device)
    }

    /// Create RLE tensor from COO tensor
    pub fn from_coo(coo: &CooTensor) -> TorshResult<Self> {
        let shape = coo.shape().clone();
        let dtype = coo.dtype();
        let device = coo.device();

        let triplets = coo.triplets();

        // Group elements by row and sort by column
        let mut row_elements: HashMap<usize, Vec<(usize, f32)>> = HashMap::new();
        for (row, col, value) in triplets.iter() {
            row_elements.entry(*row).or_default().push((*col, *value));
        }

        let mut runs = Vec::new();

        // Process each row
        for (row, mut elements) in row_elements {
            // Sort by column
            elements.sort_by_key(|&(col, _)| col);

            let mut i = 0;
            while i < elements.len() {
                let (start_col, start_val) = elements[i];
                let mut run_values = vec![start_val];
                let mut current_col = start_col;

                // Try to extend the run
                i += 1;
                while i < elements.len() && elements[i].0 == current_col + 1 {
                    current_col = elements[i].0;
                    run_values.push(elements[i].1);
                    i += 1;
                }

                runs.push(Run::new(row, start_col, run_values.len(), run_values));
            }
        }

        Self::new(runs, shape, dtype, device)
    }

    /// Get the runs
    pub fn runs(&self) -> &[Run] {
        &self.runs
    }

    /// Get value at specific position
    pub fn get(&self, row: usize, col: usize) -> Option<f32> {
        for run in &self.runs {
            if let Some(value) = run.get_value(row, col) {
                return Some(value);
            }
        }
        None
    }

    /// Validate that runs don't overlap and fit within shape
    fn validate_runs(runs: &[Run], shape: &Shape) -> TorshResult<()> {
        let (rows, cols) = (shape.dims()[0], shape.dims()[1]);

        for run in runs {
            if run.row >= rows || run.col >= cols || run.col + run.length > cols {
                return Err(TorshError::InvalidShape(
                    "Run extends outside tensor bounds".to_string(),
                ));
            }

            if run.values.len() != run.length {
                return Err(TorshError::InvalidShape(
                    "Run length doesn't match values length".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Calculate compression ratio compared to dense storage
    pub fn compression_ratio(&self) -> f32 {
        let dense_size = self.shape.numel() * std::mem::size_of::<f32>();
        let rle_size = self.runs.len()
            * (std::mem::size_of::<usize>() * 3 + // row, col, length
            std::mem::size_of::<f32>() * self.runs.iter().map(|r| r.length).sum::<usize>());

        if rle_size == 0 {
            1.0
        } else {
            dense_size as f32 / rle_size as f32
        }
    }
}

impl SparseTensor for RleTensor {
    fn format(&self) -> SparseFormat {
        SparseFormat::Rle
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> DeviceType {
        self.device
    }

    fn nnz(&self) -> usize {
        self.nnz
    }

    fn to_dense(&self) -> TorshResult<Tensor> {
        let (rows, cols) = (self.shape.dims()[0], self.shape.dims()[1]);
        let mut data = vec![0.0f32; rows * cols];

        for run in &self.runs {
            for (i, &value) in run.values.iter().enumerate() {
                let row = run.row;
                let col = run.col + i;
                let idx = row * cols + col;
                data[idx] = value;
            }
        }

        Tensor::from_vec(data, self.shape.dims())
    }

    fn to_coo(&self) -> TorshResult<CooTensor> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut values = Vec::new();

        for run in &self.runs {
            for (i, &value) in run.values.iter().enumerate() {
                rows.push(run.row);
                cols.push(run.col + i);
                values.push(value);
            }
        }

        CooTensor::new(rows, cols, values, self.shape.clone())
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

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::zeros;

    #[test]
    fn test_rle_creation() {
        let runs = vec![
            Run::new(0, 1, 3, vec![1.0, 2.0, 3.0]),
            Run::new(1, 0, 2, vec![4.0, 5.0]),
        ];

        let shape = Shape::new(vec![2, 4]);
        let rle = RleTensor::new(runs, shape, DType::F32, DeviceType::Cpu).unwrap();

        assert_eq!(rle.nnz(), 5);
        assert_eq!(rle.runs().len(), 2);
        assert_eq!(rle.get(0, 1), Some(1.0));
        assert_eq!(rle.get(0, 2), Some(2.0));
        assert_eq!(rle.get(1, 0), Some(4.0));
        assert_eq!(rle.get(0, 0), None);
    }

    #[test]
    fn test_rle_from_dense() {
        // Test with all zeros first
        let dense = zeros::<f32>(&[2, 4]).unwrap();
        let rle = RleTensor::from_dense(&dense, 0.0).unwrap();
        assert_eq!(rle.nnz(), 0); // All zeros initially

        // Test with actual non-zero values
        // Create a pattern with consecutive elements:
        // Row 0: [0, 1, 2, 0]
        // Row 1: [3, 4, 0, 0]
        // This should create 2 runs:
        // - Run 1: row 0, cols 1-2, values [1, 2]
        // - Run 2: row 1, cols 0-1, values [3, 4]
        let data = vec![
            0.0, 1.0, 2.0, 0.0, // Row 0
            3.0, 4.0, 0.0, 0.0, // Row 1
        ];
        let dense_with_values = Tensor::from_vec(data, &[2, 4]).unwrap();

        let rle_with_values = RleTensor::from_dense(&dense_with_values, 0.1).unwrap();

        // Should have 4 non-zero elements total
        assert_eq!(rle_with_values.nnz(), 4);

        // Should have 2 runs
        assert_eq!(rle_with_values.runs().len(), 2);

        // Verify first run (row 0, cols 1-2, values [1, 2])
        let run1 = &rle_with_values.runs()[0];
        assert_eq!(run1.row, 0);
        assert_eq!(run1.col, 1);
        assert_eq!(run1.length, 2);
        assert_eq!(run1.values, vec![1.0, 2.0]);

        // Verify second run (row 1, cols 0-1, values [3, 4])
        let run2 = &rle_with_values.runs()[1];
        assert_eq!(run2.row, 1);
        assert_eq!(run2.col, 0);
        assert_eq!(run2.length, 2);
        assert_eq!(run2.values, vec![3.0, 4.0]);

        // Test element access
        assert_eq!(rle_with_values.get(0, 0), None); // 0.0 -> None (below threshold)
        assert_eq!(rle_with_values.get(0, 1), Some(1.0));
        assert_eq!(rle_with_values.get(0, 2), Some(2.0));
        assert_eq!(rle_with_values.get(0, 3), None); // 0.0 -> None
        assert_eq!(rle_with_values.get(1, 0), Some(3.0));
        assert_eq!(rle_with_values.get(1, 1), Some(4.0));
        assert_eq!(rle_with_values.get(1, 2), None); // 0.0 -> None
        assert_eq!(rle_with_values.get(1, 3), None); // 0.0 -> None
    }

    #[test]
    fn test_rle_conversion() {
        let runs = vec![Run::new(0, 1, 2, vec![1.0, 2.0])];

        let shape = Shape::new(vec![2, 3]);
        let rle = RleTensor::new(runs, shape, DType::F32, DeviceType::Cpu).unwrap();

        let coo = rle.to_coo().unwrap();
        assert_eq!(coo.nnz(), 2);

        let dense = rle.to_dense().unwrap();
        assert_eq!(dense.shape(), rle.shape);
    }

    #[test]
    fn test_run_contains() {
        let run = Run::new(0, 5, 3, vec![1.0, 2.0, 3.0]);

        assert!(run.contains(0, 5));
        assert!(run.contains(0, 6));
        assert!(run.contains(0, 7));
        assert!(!run.contains(0, 4));
        assert!(!run.contains(0, 8));
        assert!(!run.contains(1, 5));
    }

    #[test]
    fn test_compression_ratio() {
        let runs = vec![Run::new(0, 0, 2, vec![1.0, 2.0])];

        let shape = Shape::new(vec![10, 10]);
        let rle = RleTensor::new(runs, shape, DType::F32, DeviceType::Cpu).unwrap();

        let ratio = rle.compression_ratio();
        assert!(ratio > 1.0); // Should be compressed
    }
}
