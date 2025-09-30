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
    /// List of indices (fancy indexing)
    List(Vec<i64>),
    /// Boolean mask
    Mask(Tensor<bool>),
    /// Ellipsis (...) - represents multiple ':' to fill remaining dimensions
    Ellipsis,
    /// Newaxis (None) - adds a dimension of size 1
    NewAxis,
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
    pub fn index(&self, indices: &[TensorIndex]) -> Result<Self> {
        // Validate number of indices (NewAxis and Ellipsis don't consume tensor dimensions)
        let consuming_indices = indices
            .iter()
            .filter(|idx| !matches!(idx, TensorIndex::NewAxis | TensorIndex::Ellipsis))
            .count();

        if consuming_indices > self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Too many indices for tensor: tensor has {} dimensions but {} consuming indices were provided",
                self.ndim(),
                consuming_indices
            )));
        }

        // Handle ellipsis by expanding indices first
        let expanded_indices = self.expand_ellipsis(indices)?;

        // Process each expanded index to determine the output shape and extraction logic
        let mut output_shape = Vec::new();
        let mut slices = Vec::new();
        let mut input_dim_idx = 0; // Track which input tensor dimension we're accessing

        for index in expanded_indices.iter() {
            if let TensorIndex::NewAxis = index {
                // NewAxis doesn't consume input dimensions, just adds a new dimension of size 1
                output_shape.push(1);
                slices.push((0, 1, 1));
                // Don't increment input_dim_idx for NewAxis
                continue;
            }

            // For all other indices, we need to get the dimension size from the input tensor
            let dim_size = if input_dim_idx < self.ndim() {
                self.shape().dims()[input_dim_idx]
            } else {
                return Err(TorshError::InvalidArgument(format!(
                    "Index {} beyond tensor dimensions (tensor has {} dimensions)",
                    input_dim_idx,
                    self.ndim()
                )));
            };

            match index {
                TensorIndex::Index(idx) => {
                    // Single index - this dimension is removed
                    let idx = if *idx < 0 {
                        (dim_size as i64 + idx) as usize
                    } else {
                        *idx as usize
                    };

                    if idx >= dim_size {
                        return Err(TorshError::IndexOutOfBounds {
                            index: idx,
                            size: dim_size,
                        });
                    }

                    slices.push((idx, idx + 1, 1));
                    // Single index doesn't add an output dimension, but consumes input dimension
                    input_dim_idx += 1;
                }
                TensorIndex::Range(start, stop, step) => {
                    let step = step.unwrap_or(1);
                    if step == 0 {
                        return Err(TorshError::InvalidArgument(
                            "Step cannot be zero".to_string(),
                        ));
                    }

                    let start = start
                        .map(|s| {
                            if s < 0 {
                                (dim_size as i64 + s).max(0) as usize
                            } else {
                                s.min(dim_size as i64) as usize
                            }
                        })
                        .unwrap_or(0);

                    let stop = stop
                        .map(|s| {
                            if s < 0 {
                                (dim_size as i64 + s).max(0) as usize
                            } else {
                                s.min(dim_size as i64) as usize
                            }
                        })
                        .unwrap_or(dim_size);

                    let size = if step > 0 {
                        ((stop as i64 - start as i64 + step - 1) / step).max(0) as usize
                    } else {
                        ((stop as i64 - start as i64 + step + 1) / step).max(0) as usize
                    };

                    output_shape.push(size);
                    slices.push((start, stop, step as usize));
                    input_dim_idx += 1;
                }
                TensorIndex::All => {
                    output_shape.push(dim_size);
                    slices.push((0, dim_size, 1));
                    input_dim_idx += 1;
                }
                TensorIndex::List(indices_list) => {
                    // Fancy indexing with list of indices
                    for &idx in indices_list {
                        let normalized_idx = if idx < 0 {
                            (dim_size as i64 + idx) as usize
                        } else {
                            idx as usize
                        };

                        if normalized_idx >= dim_size {
                            return Err(TorshError::IndexOutOfBounds {
                                index: normalized_idx,
                                size: dim_size,
                            });
                        }
                    }

                    output_shape.push(indices_list.len());
                    // Store list indices as a special slice marker
                    slices.push((0, indices_list.len(), 0)); // step=0 indicates list indexing
                    input_dim_idx += 1;
                }
                TensorIndex::Mask(mask) => {
                    // Boolean mask indexing - dimension is flattened
                    if mask.ndim() != 1 {
                        return Err(TorshError::InvalidArgument(
                            "Boolean mask must be 1D for single dimension indexing".to_string(),
                        ));
                    }

                    if mask.numel() != dim_size {
                        return Err(TorshError::ShapeMismatch {
                            expected: vec![dim_size],
                            got: mask.shape().dims().to_vec(),
                        });
                    }

                    // Count True values to determine output size
                    let mask_data = mask.to_vec()?;
                    let true_count = mask_data.iter().filter(|&&x| x).count();

                    output_shape.push(true_count);
                    // Store mask as special slice marker
                    slices.push((0, true_count, 0)); // step=0 indicates mask indexing
                    input_dim_idx += 1;
                }
                TensorIndex::NewAxis => {
                    // This should not happen since NewAxis is handled earlier
                    return Err(TorshError::InvalidArgument(
                        "NewAxis should be handled before this point".to_string(),
                    ));
                }
                TensorIndex::Ellipsis => {
                    // This should not happen since ellipsis is expanded earlier
                    return Err(TorshError::InvalidArgument(
                        "Ellipsis should be expanded before processing".to_string(),
                    ));
                }
            }
        }

        // If all indices were single indices, we need at least one dimension
        if output_shape.is_empty() {
            output_shape.push(1);
        }

        // Use specialized extraction logic for advanced indexing
        if expanded_indices
            .iter()
            .any(|idx| matches!(idx, TensorIndex::List(_) | TensorIndex::Mask(_)))
        {
            self.extract_advanced_indexing(&expanded_indices, &output_shape)
        } else {
            self.extract_basic_indexing(&expanded_indices, &output_shape, &slices)
        }
    }

    /// Extract data using basic indexing (ranges, single indices, all)
    fn extract_basic_indexing(
        &self,
        indices: &[TensorIndex],
        output_shape: &[usize],
        slices: &[(usize, usize, usize)],
    ) -> Result<Self> {
        let input_data = self.to_vec()?;

        let output_size = output_shape.iter().product();
        let mut output_data = Vec::with_capacity(output_size);

        let input_strides = self.compute_strides();
        let output_strides = compute_strides_from_shape(output_shape);

        for out_idx in 0..output_size {
            // Convert flat index to multi-dimensional indices
            let mut out_indices = vec![0; output_shape.len()];
            let mut remaining = out_idx;
            for (i, &stride) in output_strides.iter().enumerate() {
                out_indices[i] = remaining / stride;
                remaining %= stride;
            }

            // Map output indices to input indices using slices
            let mut input_flat_idx = 0;
            let mut out_dim = 0;
            let mut input_dim = 0;

            for (slice_idx, &(start, _, step)) in slices.iter().enumerate() {
                // Skip NewAxis dimensions in input tensor
                if slice_idx < indices.len() && matches!(indices[slice_idx], TensorIndex::NewAxis) {
                    out_dim += 1;
                    continue;
                }

                // Ensure we don't exceed input dimensions
                if input_dim >= input_strides.len() {
                    break;
                }

                let idx = if slice_idx < indices.len()
                    && matches!(indices[slice_idx], TensorIndex::Index(_))
                {
                    start
                } else {
                    start + out_indices[out_dim] * step
                };
                input_flat_idx += idx * input_strides[input_dim];

                if !(slice_idx < indices.len()
                    && matches!(indices[slice_idx], TensorIndex::Index(_)))
                {
                    out_dim += 1;
                }
                input_dim += 1;
            }

            output_data.push(input_data[input_flat_idx]);
        }

        Self::from_data(output_data, output_shape.to_vec(), self.device)
    }

    /// Extract data using advanced indexing (lists, masks)
    fn extract_advanced_indexing(
        &self,
        indices: &[TensorIndex],
        output_shape: &[usize],
    ) -> Result<Self> {
        let input_data = self.to_vec()?;

        let output_size = output_shape.iter().product();
        let mut output_data = Vec::with_capacity(output_size);

        let input_strides = self.compute_strides();
        let output_strides = compute_strides_from_shape(output_shape);

        for out_idx in 0..output_size {
            // Convert flat index to multi-dimensional indices
            let mut out_indices = vec![0; output_shape.len()];
            let mut remaining = out_idx;
            for (i, &stride) in output_strides.iter().enumerate() {
                out_indices[i] = remaining / stride;
                remaining %= stride;
            }

            // Map output indices to input indices using advanced indexing
            let mut input_flat_idx = 0;
            let mut out_dim = 0;

            for (dim_idx, index) in indices.iter().enumerate() {
                if dim_idx >= self.ndim() {
                    break;
                }

                let input_idx = match index {
                    TensorIndex::Index(idx) => {
                        let dim_size = self.shape().dims()[dim_idx];

                        if *idx < 0 {
                            (dim_size as i64 + idx) as usize
                        } else {
                            *idx as usize
                        }
                    }
                    TensorIndex::Range(start, _stop, step) => {
                        let dim_size = self.shape().dims()[dim_idx];
                        let step = step.unwrap_or(1);
                        let start = start
                            .map(|s| {
                                if s < 0 {
                                    (dim_size as i64 + s).max(0) as usize
                                } else {
                                    s.min(dim_size as i64) as usize
                                }
                            })
                            .unwrap_or(0);

                        start + out_indices[out_dim] * (step as usize)
                    }
                    TensorIndex::All => out_indices[out_dim],
                    TensorIndex::List(indices_list) => {
                        // Fancy indexing: use the list index
                        let list_idx = out_indices[out_dim];
                        if list_idx >= indices_list.len() {
                            return Err(TorshError::IndexOutOfBounds {
                                index: list_idx,
                                size: indices_list.len(),
                            });
                        }

                        let actual_idx = indices_list[list_idx];
                        let dim_size = self.shape().dims()[dim_idx];

                        if actual_idx < 0 {
                            (dim_size as i64 + actual_idx) as usize
                        } else {
                            actual_idx as usize
                        }
                    }
                    TensorIndex::Mask(mask) => {
                        // Boolean mask indexing
                        let mask_data = mask.to_vec()?;

                        // Find the nth True value in the mask
                        let target_true_idx = out_indices[out_dim];
                        let mut true_count = 0;
                        let mut found_idx = None;
                        for (i, &mask_val) in mask_data.iter().enumerate() {
                            if mask_val {
                                if true_count == target_true_idx {
                                    found_idx = Some(i);
                                    break;
                                }
                                true_count += 1;
                            }
                        }

                        match found_idx {
                            Some(idx) => idx,
                            None => {
                                return Err(TorshError::IndexOutOfBounds {
                                    index: target_true_idx,
                                    size: true_count,
                                });
                            }
                        }
                    }
                    TensorIndex::NewAxis => {
                        // NewAxis doesn't consume input dimensions
                        continue;
                    }
                    TensorIndex::Ellipsis => {
                        // Ellipsis should be handled in shape computation
                        out_indices[out_dim]
                    }
                };

                input_flat_idx += input_idx * input_strides[dim_idx];

                // Only advance output dimension for non-index operations
                if !matches!(index, TensorIndex::Index(_) | TensorIndex::NewAxis) {
                    out_dim += 1;
                }
            }

            // Handle remaining dimensions
            for stride in input_strides
                .iter()
                .skip(indices.len())
                .take(self.ndim() - indices.len())
            {
                if out_dim < out_indices.len() {
                    input_flat_idx += out_indices[out_dim] * stride;
                    out_dim += 1;
                }
            }

            if input_flat_idx >= input_data.len() {
                return Err(TorshError::IndexOutOfBounds {
                    index: input_flat_idx,
                    size: input_data.len(),
                });
            }

            output_data.push(input_data[input_flat_idx]);
        }

        Self::from_data(output_data, output_shape.to_vec(), self.device)
    }

    /// Expand ellipsis into explicit All indices
    fn expand_ellipsis(&self, indices: &[TensorIndex]) -> Result<Vec<TensorIndex>> {
        let mut expanded = Vec::new();
        let mut found_ellipsis = false;

        // Count non-ellipsis, non-newaxis indices to determine how many dimensions ellipsis should expand to
        let non_expanding_indices = indices
            .iter()
            .filter(|idx| !matches!(idx, TensorIndex::Ellipsis | TensorIndex::NewAxis))
            .count();

        for index in indices {
            match index {
                TensorIndex::Ellipsis => {
                    if found_ellipsis {
                        return Err(TorshError::InvalidArgument(
                            "Only one ellipsis (...) is allowed per indexing operation".to_string(),
                        ));
                    }
                    found_ellipsis = true;

                    // Calculate how many dimensions the ellipsis should expand to
                    let ellipsis_dims = if self.ndim() >= non_expanding_indices {
                        self.ndim() - non_expanding_indices
                    } else {
                        0
                    };

                    // Expand ellipsis to All indices
                    for _ in 0..ellipsis_dims {
                        expanded.push(TensorIndex::All);
                    }
                }
                _ => {
                    expanded.push(index.clone());
                }
            }
        }

        // If no ellipsis was found, add implicit trailing All indices for remaining dimensions
        if !found_ellipsis {
            let current_dims = expanded
                .iter()
                .filter(|idx| !matches!(idx, TensorIndex::NewAxis))
                .count();

            for _ in current_dims..self.ndim() {
                expanded.push(TensorIndex::All);
            }
        }

        Ok(expanded)
    }

    /// Get a single element (1D indexing)
    pub fn get_1d(&self, index: usize) -> Result<T> {
        if self.ndim() != 1 {
            return Err(TorshError::InvalidShape(
                "get_1d() can only be used on 1D tensors".to_string(),
            ));
        }

        if index >= self.shape().dims()[0] {
            return Err(TorshError::IndexOutOfBounds {
                index,
                size: self.shape().dims()[0],
            });
        }

        let data = self.data()?;
        Ok(data[index])
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

        let data = self.to_vec()?;

        let index = row * shape.dims()[1] + col;
        Ok(data[index])
    }

    /// Get a single element (3D indexing)
    pub fn get_3d(&self, x: usize, y: usize, z: usize) -> Result<T> {
        if self.ndim() != 3 {
            return Err(TorshError::InvalidShape(
                "get_3d() can only be used on 3D tensors".to_string(),
            ));
        }

        let shape = self.shape();
        if x >= shape.dims()[0] || y >= shape.dims()[1] || z >= shape.dims()[2] {
            return Err(TorshError::IndexOutOfBounds {
                index: x * shape.dims()[1] * shape.dims()[2] + y * shape.dims()[2] + z,
                size: shape.numel(),
            });
        }

        let data = self.to_vec()?;

        let index = x * shape.dims()[1] * shape.dims()[2] + y * shape.dims()[2] + z;
        Ok(data[index])
    }

    /// Set a single element (1D indexing)
    pub fn set_1d(&mut self, index: usize, value: T) -> Result<()> {
        if self.ndim() != 1 {
            return Err(TorshError::InvalidShape(
                "set_1d() can only be used on 1D tensors".to_string(),
            ));
        }

        if index >= self.shape().dims()[0] {
            return Err(TorshError::IndexOutOfBounds {
                index,
                size: self.shape().dims()[0],
            });
        }

        let mut data = self.to_vec()?;
        data[index] = value;
        *self = Self::from_data(data, self.shape().dims().to_vec(), self.device())?;
        Ok(())
    }

    /// Set a single element (2D indexing)
    pub fn set_2d(&mut self, row: usize, col: usize, value: T) -> Result<()> {
        if self.ndim() != 2 {
            return Err(TorshError::InvalidShape(
                "set_2d() can only be used on 2D tensors".to_string(),
            ));
        }

        let shape = self.shape();
        if row >= shape.dims()[0] || col >= shape.dims()[1] {
            return Err(TorshError::IndexOutOfBounds {
                index: row * shape.dims()[1] + col,
                size: shape.numel(),
            });
        }

        let mut data = self.to_vec()?;
        let index = row * shape.dims()[1] + col;
        data[index] = value;
        *self = Self::from_data(data, self.shape().dims().to_vec(), self.device())?;
        Ok(())
    }

    /// Set a single element (3D indexing)
    pub fn set_3d(&mut self, x: usize, y: usize, z: usize, value: T) -> Result<()> {
        if self.ndim() != 3 {
            return Err(TorshError::InvalidShape(
                "set_3d() can only be used on 3D tensors".to_string(),
            ));
        }

        let shape = self.shape();
        if x >= shape.dims()[0] || y >= shape.dims()[1] || z >= shape.dims()[2] {
            return Err(TorshError::IndexOutOfBounds {
                index: x * shape.dims()[1] * shape.dims()[2] + y * shape.dims()[2] + z,
                size: shape.numel(),
            });
        }

        let mut data = self.to_vec()?;
        let index = x * shape.dims()[1] * shape.dims()[2] + y * shape.dims()[2] + z;
        data[index] = value;
        *self = Self::from_data(data, self.shape().dims().to_vec(), self.device())?;
        Ok(())
    }

    /// Select along a dimension
    pub fn select(&self, dim: i32, index: i64) -> Result<Self> {
        let ndim = self.ndim() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;

        if dim >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                self.ndim()
            )));
        }

        let dim_size = self.shape().dims()[dim] as i64;
        let index = if index < 0 { dim_size + index } else { index };

        if index < 0 || index >= dim_size {
            return Err(TorshError::IndexOutOfBounds {
                index: index as usize,
                size: dim_size as usize,
            });
        }

        // Create index array for slicing
        let mut indices = Vec::new();
        for d in 0..self.ndim() {
            if d == dim {
                indices.push(TensorIndex::Index(index));
            } else {
                indices.push(TensorIndex::All);
            }
        }

        // Use the existing index function
        self.index(&indices)
    }

    /// Slice along a dimension with PyTorch-style parameters
    pub fn slice_with_step(
        &self,
        dim: i32,
        start: Option<i64>,
        end: Option<i64>,
        step: Option<i64>,
    ) -> Result<Self> {
        let ndim = self.ndim() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;

        if dim >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                self.ndim()
            )));
        }

        // Create index array for slicing
        let mut indices = Vec::new();
        for d in 0..self.ndim() {
            if d == dim {
                indices.push(TensorIndex::Range(start, end, step));
            } else {
                indices.push(TensorIndex::All);
            }
        }

        // Use the existing index function
        self.index(&indices)
    }

    /// Narrow along a dimension
    pub fn narrow(&self, dim: i32, start: i64, length: usize) -> Result<Self> {
        let ndim = self.ndim() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;

        if dim >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                self.ndim()
            )));
        }

        let dim_size = self.shape().dims()[dim] as i64;
        let start = if start < 0 { dim_size + start } else { start };

        if start < 0 || start >= dim_size {
            return Err(TorshError::InvalidArgument(format!(
                "Start index {start} out of range for dimension {dim} with size {dim_size}"
            )));
        }

        let end = start + length as i64;
        if end > dim_size {
            return Err(TorshError::InvalidArgument(format!(
                "End index {end} out of range for dimension {dim} with size {dim_size}"
            )));
        }

        // Create index array for slicing
        let mut indices = Vec::new();
        for d in 0..self.ndim() {
            if d == dim {
                indices.push(TensorIndex::Range(Some(start), Some(end), None));
            } else {
                indices.push(TensorIndex::All);
            }
        }

        // Use the existing index function
        self.index(&indices)
    }

    /// Boolean indexing (masking)
    pub fn masked_select(&self, mask: &Tensor<bool>) -> Result<Self> {
        if self.shape() != mask.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: mask.shape().dims().to_vec(),
            });
        }

        let self_data = self.data()?;
        let mask_data = mask.data()?;

        // Collect all elements where mask is true
        let mut selected_data = Vec::new();
        for (i, &mask_val) in mask_data.iter().enumerate() {
            if mask_val {
                selected_data.push(self_data[i]);
            }
        }

        // Return 1D tensor with selected elements
        Self::from_data(
            selected_data.clone(),
            vec![selected_data.len()],
            self.device,
        )
    }

    pub fn take(&self, indices: &Tensor<i64>) -> Result<Self> {
        let self_data = self.data()?;

        let indices_data = indices.data()?;

        let self_size = self.shape().numel();
        let output_shape = indices.shape().dims().to_vec();
        let output_size = indices.shape().numel();
        let mut output_data = Vec::with_capacity(output_size);

        // Take elements at the given flat indices
        for &idx in indices_data.iter() {
            let idx = if idx < 0 {
                (self_size as i64 + idx) as usize
            } else {
                idx as usize
            };

            if idx >= self_size {
                return Err(TorshError::IndexOutOfBounds {
                    index: idx,
                    size: self_size,
                });
            }

            output_data.push(self_data[idx]);
        }

        Self::from_data(output_data, output_shape, self.device)
    }

    /// Put values at indices
    pub fn put(&self, indices: &Tensor<i64>, values: &Self) -> Result<Self> {
        let self_data = self.data()?;

        let indices_data = indices.data()?;
        let values_data = values.data()?;

        // Check that indices and values have the same shape
        if indices.shape() != values.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: indices.shape().dims().to_vec(),
                got: values.shape().dims().to_vec(),
            });
        }

        let self_size = self.shape().numel();
        let mut output_data = self_data.clone();

        // Put values at the given flat indices
        for (i, &idx) in indices_data.iter().enumerate() {
            let idx = if idx < 0 {
                (self_size as i64 + idx) as usize
            } else {
                idx as usize
            };

            if idx >= self_size {
                return Err(TorshError::IndexOutOfBounds {
                    index: idx,
                    size: self_size,
                });
            }

            output_data[idx] = values_data[i];
        }

        Self::from_data(output_data, self.shape().dims().to_vec(), self.device)
    }

    /// Select indices along a dimension
    pub fn index_select(&self, dim: i32, index: &Tensor<i64>) -> Result<Self> {
        let ndim = self.ndim() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;

        if dim >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                self.ndim()
            )));
        }

        // Index must be 1D
        if index.ndim() != 1 {
            return Err(TorshError::InvalidShape(
                "index_select expects a 1D index tensor".to_string(),
            ));
        }

        // Calculate output shape
        let mut output_shape = self.shape().dims().to_vec();
        output_shape[dim] = index.shape().dims()[0];

        let output_size: usize = output_shape.iter().product();
        let mut output_data = Vec::with_capacity(output_size);

        let self_data = self.data()?;

        let index_data = index.data()?;

        // Compute strides
        let self_strides = self.compute_strides();
        let _output_strides = Self::compute_strides_for_shape(&output_shape);

        // Select elements
        for out_idx in 0..output_size {
            // Convert flat index to multi-dimensional index
            let mut indices = vec![0; self.ndim()];
            let mut remaining = out_idx;
            for i in (0..self.ndim()).rev() {
                indices[i] = remaining % output_shape[i];
                remaining /= output_shape[i];
            }

            // For the selected dimension, use the index from the index tensor
            let select_idx = indices[dim];
            let selected_value = index_data[select_idx] as usize;

            if selected_value >= self.shape().dims()[dim] {
                return Err(TorshError::IndexOutOfBounds {
                    index: selected_value,
                    size: self.shape().dims()[dim],
                });
            }

            indices[dim] = selected_value;

            // Compute flat index in source tensor
            let src_flat_idx = indices
                .iter()
                .zip(&self_strides)
                .map(|(idx, stride)| idx * stride)
                .sum::<usize>();

            output_data.push(self_data[src_flat_idx]);
        }

        Self::from_data(output_data, output_shape, self.device)
    }

    /// Compute strides for the tensor's shape
    pub(crate) fn compute_strides(&self) -> Vec<usize> {
        Self::compute_strides_for_shape(self.shape().dims())
    }

    /// Compute strides for a given shape
    pub(crate) fn compute_strides_for_shape(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}

/// Helper function to compute strides from shape
fn compute_strides_from_shape(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
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

    // Ellipsis: s![ellipsis]
    (ellipsis) => {
        TensorIndex::Ellipsis
    };

    // NewAxis: s![None]
    (None) => {
        TensorIndex::NewAxis
    };
}

/// Advanced indexing macros
#[macro_export]
macro_rules! fancy_idx {
    // List indexing: fancy_idx![0, 2, 1]
    [$($idx:expr),+ $(,)?] => {
        TensorIndex::List(vec![$($idx),+])
    };
}

#[macro_export]
macro_rules! mask_idx {
    // Boolean mask indexing: mask_idx![mask_tensor]
    [$mask:expr] => {
        TensorIndex::Mask($mask)
    };
}

/// Convenient indexing syntax
impl<T: TensorElement> Tensor<T> {
    /// Advanced indexing with list of indices (fancy indexing)
    pub fn index_with_list(&self, dim: i32, indices: &[i64]) -> Result<Self> {
        let ndim = self.ndim() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;

        if dim >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                self.ndim()
            )));
        }

        let mut index_spec = vec![TensorIndex::All; self.ndim()];
        index_spec[dim] = TensorIndex::List(indices.to_vec());

        self.index(&index_spec)
    }

    /// Boolean mask indexing for a specific dimension
    pub fn index_with_mask(&self, dim: i32, mask: &Tensor<bool>) -> Result<Self> {
        let ndim = self.ndim() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;

        if dim >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                self.ndim()
            )));
        }

        let mut index_spec = vec![TensorIndex::All; self.ndim()];
        index_spec[dim] = TensorIndex::Mask(mask.clone());

        self.index(&index_spec)
    }

    /// Global boolean mask indexing (flattens to 1D result)
    pub fn mask_select(&self, mask: &Tensor<bool>) -> Result<Self> {
        if self.shape() != mask.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: self.shape().dims().to_vec(),
                got: mask.shape().dims().to_vec(),
            });
        }

        let self_data = self.data()?;

        let mask_data = mask.data()?;

        // Collect all elements where mask is true
        let mut selected_data = Vec::new();
        for (i, &mask_val) in mask_data.iter().enumerate() {
            if mask_val {
                selected_data.push(self_data[i]);
            }
        }

        // Return 1D tensor with selected elements
        Self::from_data(
            selected_data.clone(),
            vec![selected_data.len()],
            self.device,
        )
    }

    /// Create boolean mask from condition
    pub fn where_condition<F>(&self, condition: F) -> Result<Tensor<bool>>
    where
        F: Fn(&T) -> bool,
        T: Clone,
    {
        let data = self.data()?;

        let mask_data: Vec<bool> = data.iter().map(condition).collect();

        Tensor::from_data(mask_data, self.shape().dims().to_vec(), self.device)
    }

    /// Scatter values along an axis using indices (indexing version)
    pub fn scatter_indexed(&self, dim: i32, index: &Tensor<i64>, src: &Self) -> Result<Self> {
        let ndim = self.ndim() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;

        if dim >= self.ndim() {
            return Err(TorshError::InvalidArgument(format!(
                "Dimension {} out of range for tensor with {} dimensions",
                dim,
                self.ndim()
            )));
        }

        let self_shape_binding = self.shape();
        let self_shape = self_shape_binding.dims();
        let index_shape_binding = index.shape();
        let index_shape = index_shape_binding.dims();
        let src_shape_binding = src.shape();
        let src_shape = src_shape_binding.dims();

        // Validate shapes
        if index_shape != src_shape {
            return Err(TorshError::ShapeMismatch {
                expected: index_shape.to_vec(),
                got: src_shape.to_vec(),
            });
        }

        if index_shape.len() != self_shape.len() {
            return Err(TorshError::InvalidArgument(
                "Index tensor must have same number of dimensions as input tensor".to_string(),
            ));
        }

        // Start with a copy of self
        let mut result_data = self.data()?.clone();
        let index_data = index.data()?;
        let src_data = src.data()?;
        let self_strides = self.compute_strides();

        let index_size = index_shape.iter().product();

        // Process each element in the index tensor
        for flat_idx in 0..index_size {
            // Convert flat index to multi-dimensional coordinates
            let mut coords = Vec::new();
            let mut temp_idx = flat_idx;

            for &dim_size in index_shape.iter().rev() {
                coords.push(temp_idx % dim_size);
                temp_idx /= dim_size;
            }
            coords.reverse();

            // Get the index value for the scatter dimension
            let scatter_idx = index_data[flat_idx];
            let dim_size = self_shape[dim] as i64;
            let scatter_idx = if scatter_idx < 0 {
                dim_size + scatter_idx
            } else {
                scatter_idx
            };

            if scatter_idx < 0 || scatter_idx >= dim_size {
                return Err(TorshError::IndexOutOfBounds {
                    index: scatter_idx as usize,
                    size: dim_size as usize,
                });
            }

            // Calculate destination index in result tensor
            coords[dim] = scatter_idx as usize;
            let mut dest_idx = 0;
            for (coord, &stride) in coords.iter().zip(self_strides.iter()) {
                dest_idx += coord * stride;
            }

            result_data[dest_idx] = src_data[flat_idx];
        }

        Self::from_data(result_data, self_shape.to_vec(), self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::{tensor_2d, zeros};

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

        // Test advanced indexing macros
        let _fancy = fancy_idx![0, 2, 1];
        let _ellipsis = s![ellipsis];
        let _newaxis = s![None];
    }

    #[test]
    fn test_get_set() {
        let tensor = tensor_2d(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]).unwrap();

        // Test get
        assert_eq!(tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(tensor.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(tensor.get(&[1, 2]).unwrap(), 6.0);

        // Test set
        tensor.set(&[1, 1], 10.0).unwrap();
        assert_eq!(tensor.get(&[1, 1]).unwrap(), 10.0);

        // Test out of bounds
        assert!(tensor.get(&[2, 0]).is_err());
        assert!(tensor.set(&[0, 3], 0.0).is_err());
    }

    #[test]
    fn test_gather() {
        // Create a 3x3 tensor
        let tensor = tensor_2d(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]).unwrap();

        // Create indices for gathering along dim=1
        let indices = tensor_2d(&[&[0i64, 2, 1], &[1, 0, 2], &[2, 1, 0]]).unwrap();

        let result = tensor.gather(1, &indices).unwrap();

        // Expected: [[1, 3, 2], [5, 4, 6], [9, 8, 7]]
        assert_eq!(result.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(result.get(&[0, 1]).unwrap(), 3.0);
        assert_eq!(result.get(&[0, 2]).unwrap(), 2.0);
        assert_eq!(result.get(&[1, 0]).unwrap(), 5.0);
        assert_eq!(result.get(&[1, 1]).unwrap(), 4.0);
        assert_eq!(result.get(&[1, 2]).unwrap(), 6.0);
        assert_eq!(result.get(&[2, 0]).unwrap(), 9.0);
        assert_eq!(result.get(&[2, 1]).unwrap(), 8.0);
        assert_eq!(result.get(&[2, 2]).unwrap(), 7.0);
    }

    #[test]
    fn test_scatter() {
        // Create a 3x3 tensor of zeros
        let tensor = zeros::<f32>(&[3, 3]).unwrap();

        // Create indices for scattering along dim=1
        let indices = tensor_2d(&[&[0i64, 2, 1], &[1, 0, 2], &[2, 1, 0]]).unwrap();

        // Source values
        let src = tensor_2d(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]).unwrap();

        let result = tensor.scatter(1, &indices, &src).unwrap();

        // Expected: [[1, 3, 2], [5, 4, 6], [9, 8, 7]]
        assert_eq!(result.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(result.get(&[0, 1]).unwrap(), 3.0);
        assert_eq!(result.get(&[0, 2]).unwrap(), 2.0);
        assert_eq!(result.get(&[1, 0]).unwrap(), 5.0);
        assert_eq!(result.get(&[1, 1]).unwrap(), 4.0);
        assert_eq!(result.get(&[1, 2]).unwrap(), 6.0);
        assert_eq!(result.get(&[2, 0]).unwrap(), 9.0);
        assert_eq!(result.get(&[2, 1]).unwrap(), 8.0);
        assert_eq!(result.get(&[2, 2]).unwrap(), 7.0);
    }

    #[test]
    fn test_index_select() {
        // Create a 3x4 tensor
        let tensor = tensor_2d(&[
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
        ])
        .unwrap();

        // Select rows 0 and 2
        let row_indices = crate::creation::tensor_1d(&[0i64, 2]).unwrap();
        let result = tensor.index_select(0, &row_indices).unwrap();

        assert_eq!(result.shape().dims(), &[2, 4]);
        assert_eq!(result.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(result.get(&[0, 3]).unwrap(), 4.0);
        assert_eq!(result.get(&[1, 0]).unwrap(), 9.0);
        assert_eq!(result.get(&[1, 3]).unwrap(), 12.0);

        // Select columns 1 and 3
        let col_indices = crate::creation::tensor_1d(&[1i64, 3]).unwrap();
        let result = tensor.index_select(1, &col_indices).unwrap();

        assert_eq!(result.shape().dims(), &[3, 2]);
        assert_eq!(result.get(&[0, 0]).unwrap(), 2.0);
        assert_eq!(result.get(&[0, 1]).unwrap(), 4.0);
        assert_eq!(result.get(&[2, 0]).unwrap(), 10.0);
        assert_eq!(result.get(&[2, 1]).unwrap(), 12.0);
    }

    #[test]
    fn test_list_indexing() {
        // Test fancy indexing with list of indices
        let tensor = tensor_2d(&[
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
        ])
        .unwrap();

        // Select rows 0 and 2 using list indexing
        let indices = vec![TensorIndex::List(vec![0, 2]), TensorIndex::All];
        let result = tensor.index(&indices).unwrap();

        assert_eq!(result.shape().dims(), &[2, 4]);
        assert_eq!(result.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(result.get(&[0, 3]).unwrap(), 4.0);
        assert_eq!(result.get(&[1, 0]).unwrap(), 9.0);
        assert_eq!(result.get(&[1, 3]).unwrap(), 12.0);

        // Test index_with_list convenience method
        let result2 = tensor.index_with_list(0, &[0, 2]).unwrap();
        assert_eq!(result.shape(), result2.shape());
        assert_eq!(result.get(&[0, 0]).unwrap(), result2.get(&[0, 0]).unwrap());
    }

    #[test]
    fn test_boolean_mask_indexing() {
        use crate::creation::tensor_1d;

        // Create test tensor
        let tensor = tensor_1d(&[10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();

        // Create boolean mask
        let mask = Tensor::from_data(
            vec![true, false, true, false, true],
            vec![5],
            crate::DeviceType::Cpu,
        )
        .unwrap();

        // Test mask_select (global mask)
        let result = tensor.mask_select(&mask).unwrap();
        assert_eq!(result.shape().dims(), &[3]);
        assert_eq!(result.get(&[0]).unwrap(), 10.0);
        assert_eq!(result.get(&[1]).unwrap(), 30.0);
        assert_eq!(result.get(&[2]).unwrap(), 50.0);

        // Test dimensional mask indexing
        let result2 = tensor.index_with_mask(0, &mask).unwrap();
        assert_eq!(result2.shape().dims(), &[3]);
        assert_eq!(result2.get(&[0]).unwrap(), 10.0);
        assert_eq!(result2.get(&[1]).unwrap(), 30.0);
        assert_eq!(result2.get(&[2]).unwrap(), 50.0);
    }

    #[test]
    fn test_where_condition() {
        use crate::creation::tensor_1d;

        let tensor = tensor_1d(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

        // Create mask for values > 3.0
        let mask = tensor.where_condition(|&x| x > 3.0).unwrap();

        {
            let mask_data = mask.data().unwrap();
            assert!(!mask_data[0]); // 1.0 <= 3.0
            assert!(!mask_data[1]); // 2.0 <= 3.0
            assert!(!mask_data[2]); // 3.0 <= 3.0
            assert!(mask_data[3]); // 4.0 > 3.0
            assert!(mask_data[4]); // 5.0 > 3.0
        } // Explicitly drop the lock

        // Use the mask to select elements
        let selected = tensor.mask_select(&mask).unwrap();
        assert_eq!(selected.shape().dims(), &[2]);
        assert_eq!(selected.get(&[0]).unwrap(), 4.0);
        assert_eq!(selected.get(&[1]).unwrap(), 5.0);
    }

    #[test]
    fn test_newaxis_indexing() {
        use crate::creation::tensor_1d;

        let tensor = tensor_1d(&[1.0, 2.0, 3.0]).unwrap();

        // Add new axis at beginning
        let indices = vec![TensorIndex::NewAxis, TensorIndex::All];
        let result = tensor.index(&indices).unwrap();
        assert_eq!(result.shape().dims(), &[1, 3]);

        // Add new axis at end
        let indices = vec![TensorIndex::All, TensorIndex::NewAxis];
        let result = tensor.index(&indices).unwrap();
        assert_eq!(result.shape().dims(), &[3, 1]);

        // Add multiple new axes
        let indices = vec![
            TensorIndex::NewAxis,
            TensorIndex::All,
            TensorIndex::NewAxis,
            TensorIndex::NewAxis,
        ];
        let result = tensor.index(&indices).unwrap();
        assert_eq!(result.shape().dims(), &[1, 3, 1, 1]);
    }

    #[test]
    fn test_ellipsis_indexing() {
        // Create 3D tensor
        let tensor = crate::creation::zeros::<f32>(&[2, 3, 4]).unwrap();

        // Test ellipsis in middle
        let indices = vec![TensorIndex::Index(0), TensorIndex::Ellipsis];
        let result = tensor.index(&indices).unwrap();
        assert_eq!(result.shape().dims(), &[3, 4]);

        // Test ellipsis at end
        let indices = vec![TensorIndex::Index(1), TensorIndex::Ellipsis];
        let result = tensor.index(&indices).unwrap();
        assert_eq!(result.shape().dims(), &[3, 4]);
    }

    #[test]
    fn test_complex_indexing() {
        // Test combination of different indexing types
        let tensor = tensor_2d(&[
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
            &[13.0, 14.0, 15.0, 16.0],
        ])
        .unwrap();

        // Combine list indexing with range indexing
        let indices = vec![
            TensorIndex::List(vec![0, 2, 3]),
            TensorIndex::Range(Some(1), Some(4), None),
        ];
        let result = tensor.index(&indices).unwrap();

        assert_eq!(result.shape().dims(), &[3, 3]);
        assert_eq!(result.get(&[0, 0]).unwrap(), 2.0); // tensor[0, 1]
        assert_eq!(result.get(&[1, 0]).unwrap(), 10.0); // tensor[2, 1]
        assert_eq!(result.get(&[2, 2]).unwrap(), 16.0); // tensor[3, 3]
    }

    #[test]
    fn test_negative_indexing() {
        use crate::creation::tensor_1d;

        let tensor = tensor_1d(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

        // Test negative single index
        let indices = vec![TensorIndex::Index(-1)];
        let result = tensor.index(&indices).unwrap();
        assert_eq!(result.numel(), 1);
        assert_eq!(result.item().unwrap(), 5.0);

        // Test negative range
        let indices = vec![TensorIndex::Range(Some(-3), Some(-1), None)];
        let result = tensor.index(&indices).unwrap();
        assert_eq!(result.shape().dims(), &[2]);
        assert_eq!(result.get(&[0]).unwrap(), 3.0);
        assert_eq!(result.get(&[1]).unwrap(), 4.0);

        // Test negative list indexing
        let indices = vec![TensorIndex::List(vec![-1, -2, 0])];
        let result = tensor.index(&indices).unwrap();
        assert_eq!(result.shape().dims(), &[3]);
        assert_eq!(result.get(&[0]).unwrap(), 5.0); // -1 -> index 4
        assert_eq!(result.get(&[1]).unwrap(), 4.0); // -2 -> index 3
        assert_eq!(result.get(&[2]).unwrap(), 1.0); // 0 -> index 0
    }
}
