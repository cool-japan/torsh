//! Tensor indexing and slicing operations

use crate::{Tensor, TensorElement};
use torsh_core::error::{Result, TorshError};

/// Index type for tensor indexing
#[derive(Debug, Clone)]
pub enum TensorIndex {
    /// Single index
    Index(i64),
    /// Range of indices
    Range(Option<i64>, Option<i64>, Option<i64>), // start, stop, step
    /// All indices (:)
    All,
    /// List of indices
    List(Vec<i64>),
    /// Boolean mask
    Mask(Tensor<bool>),
}

impl TensorIndex {
    /// Create a range index
    pub fn range(start: Option<i64>, stop: Option<i64>) -> Self {
        TensorIndex::Range(start, stop, None)
    }

    /// Create a range index with step
    pub fn range_step(start: Option<i64>, stop: Option<i64>, step: i64) -> Self {
        TensorIndex::Range(start, stop, Some(step))
    }
}

/// Indexing implementation
impl<T: TensorElement> Tensor<T> {
    /// Index into the tensor
    pub fn index(&self, _indices: &[TensorIndex]) -> Result<Self> {
        // TODO: Implement indexing using scirs2
        Ok(self.clone())
    }

    /// Get a single element (1D indexing)
    pub fn get(&self, index: usize) -> Result<T> {
        if self.ndim() != 1 {
            return Err(TorshError::InvalidShape(
                "get() can only be used on 1D tensors".to_string(),
            ));
        }

        if index >= self.shape().dims()[0] {
            return Err(TorshError::IndexOutOfBounds {
                index,
                size: self.shape().dims()[0],
            });
        }

        // TODO: Actually get the value from scirs2
        Ok(T::zero())
    }

    /// Get a single element (2D indexing)
    pub fn get_2d(&self, row: usize, col: usize) -> Result<T> {
        if self.ndim() != 2 {
            return Err(TorshError::InvalidShape(
                "get_2d() can only be used on 2D tensors".to_string(),
            ));
        }

        let shape = self.shape();
        if row >= shape.dims()[0] || col >= shape.dims()[1] {
            return Err(TorshError::IndexOutOfBounds {
                index: row * shape.dims()[1] + col,
                size: shape.numel(),
            });
        }

        // TODO: Actually get the value from scirs2
        Ok(T::zero())
    }

    /// Select along a dimension
    pub fn select(&self, _dim: i32, _index: i64) -> Result<Self> {
        // TODO: Implement using scirs2
        Ok(self.clone())
    }

    /// Slice along a dimension
    pub fn slice(
        &self,
        _dim: i32,
        _start: Option<i64>,
        _end: Option<i64>,
        _step: Option<i64>,
    ) -> Result<Self> {
        // TODO: Implement using scirs2
        Ok(self.clone())
    }

    /// Narrow along a dimension
    pub fn narrow(&self, _dim: i32, _start: i64, _length: usize) -> Result<Self> {
        // TODO: Implement using scirs2
        Ok(self.clone())
    }

    /// Boolean indexing (masking)
    pub fn masked_select(&self, mask: &Tensor<bool>) -> Result<Self> {
        if self.shape() != mask.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: mask.shape().dims().to_vec(),
            });
        }

        // TODO: Implement using scirs2
        Ok(self.clone())
    }

    /// Gather values along an axis
    pub fn gather(&self, _dim: i32, _index: &Tensor<i64>) -> Result<Self> {
        // TODO: Implement using scirs2
        Ok(self.clone())
    }

    /// Scatter values along an axis
    pub fn scatter(&self, _dim: i32, _index: &Tensor<i64>, _src: &Self) -> Result<Self> {
        // TODO: Implement using scirs2
        Ok(self.clone())
    }

    /// Take values at indices
    pub fn take(&self, _indices: &Tensor<i64>) -> Result<Self> {
        // TODO: Implement using scirs2
        Ok(self.clone())
    }

    /// Put values at indices
    pub fn put(&self, _indices: &Tensor<i64>, _values: &Self) -> Result<Self> {
        // TODO: Implement using scirs2
        Ok(self.clone())
    }
}

/// Convenience macros for indexing
#[macro_export]
macro_rules! idx {
    // Single index: idx![5]
    ($idx:expr) => {
        vec![TensorIndex::Index($idx)]
    };

    // Multiple indices: idx![1, 2, 3]
    ($($idx:expr),+ $(,)?) => {
        vec![$(TensorIndex::Index($idx)),+]
    };
}

#[macro_export]
macro_rules! s {
    // Full slice: s![..]
    (..) => {
        TensorIndex::All
    };

    // To end: s![..5]
    (.. $stop:expr) => {
        TensorIndex::range(None, Some($stop))
    };

    // Range (comma syntax): s![1, 5]
    ($start:expr, $stop:expr) => {
        TensorIndex::range(Some($start), Some($stop))
    };

    // Range with step (comma syntax): s![1, 5, 2]
    ($start:expr, $stop:expr, $step:expr) => {
        TensorIndex::range_step(Some($start), Some($stop), $step)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_macros() {
        // Test single index
        let indices = idx![5];
        assert_eq!(indices.len(), 1);

        // Test multiple indices
        let indices = idx![1, 2, 3];
        assert_eq!(indices.len(), 3);

        // Test slice macros
        let _all = s![..];
        let _range = s![1, 5];
        let _range_step = s![1, 10, 2];
        let _to = s![..7];
    }
}
