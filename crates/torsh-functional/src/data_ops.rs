//! Data operations including unique, bincount, and histogram
//!
//! This module provides data analysis and manipulation operations
//! that are commonly used for data preprocessing and analysis.

use std::collections::HashMap;
use torsh_core::{Result as TorshResult, TorshError};
use torsh_tensor::Tensor;

/// Find unique elements in a tensor
///
/// Returns a tensor containing the unique elements of the input tensor.
/// Similar to torch.unique().
///
/// # Arguments
/// * `input` - Input tensor
/// * `sorted` - Whether to sort the unique elements
/// * `return_inverse` - Whether to return the indices to reconstruct the original tensor
/// * `return_counts` - Whether to return the count of each unique element
/// * `dim` - The dimension to apply unique over. If None, the unique of the flattened input is returned
///
/// # Returns
/// * `output` - Tensor containing unique elements
/// * `inverse_indices` - Optional tensor of indices to reconstruct the original tensor
/// * `counts` - Optional tensor with the count of each unique element
pub fn unique(
    input: &Tensor,
    sorted: bool,
    return_inverse: bool,
    return_counts: bool,
    dim: Option<i32>,
) -> TorshResult<(Tensor, Option<Tensor>, Option<Tensor>)> {
    // For now, implement the flattened version (dim = None)
    if dim.is_some() {
        return Err(TorshError::Other(
            "unique with dim parameter not yet implemented".to_string(),
        ));
    }

    // Flatten the input tensor
    let flattened = input.view(&[-1])?;
    let data = flattened.to_vec()?;
    let len = flattened.shape().numel();

    // Extract values - for now only support f32
    // Note: This is a simplified implementation
    let values: Vec<f32> = data;

    // Create a map to track unique values and their first occurrence
    let mut unique_map: HashMap<i32, (usize, usize)> = HashMap::new(); // value -> (first_index, count)
    let mut unique_values = Vec::new();
    let mut inverse_indices_vec = vec![0usize; len];

    for (idx, &val) in values.iter().enumerate() {
        // Convert f32 to i32 for hashmap (assuming integer values for now)
        let key = val as i32;

        match unique_map.get_mut(&key) {
            Some((first_idx, count)) => {
                *count += 1;
                inverse_indices_vec[idx] = *first_idx;
            }
            None => {
                let unique_idx = unique_values.len();
                unique_map.insert(key, (unique_idx, 1));
                unique_values.push(val);
                inverse_indices_vec[idx] = unique_idx;
            }
        }
    }

    // Sort if requested
    let mut sorted_indices: Vec<usize> = (0..unique_values.len()).collect();
    if sorted {
        sorted_indices.sort_by(|&a, &b| unique_values[a].partial_cmp(&unique_values[b]).unwrap());

        // Reorder unique values
        let sorted_unique: Vec<f32> = sorted_indices.iter().map(|&i| unique_values[i]).collect();
        unique_values = sorted_unique;

        // Update inverse indices for sorted order
        if return_inverse {
            let mut index_map = vec![0; sorted_indices.len()];
            for (new_idx, &old_idx) in sorted_indices.iter().enumerate() {
                index_map[old_idx] = new_idx;
            }
            for idx in &mut inverse_indices_vec {
                *idx = index_map[*idx];
            }
        }
    }

    // Create output tensor
    let unique_len = unique_values.len();
    let output = Tensor::from_vec(unique_values.clone(), &[unique_len])?;

    // Create inverse indices tensor if requested
    let inverse_indices = if return_inverse {
        let inverse_data: Vec<f32> = inverse_indices_vec.iter().map(|&i| i as f32).collect();
        Some(Tensor::from_vec(inverse_data, &[len])?)
    } else {
        None
    };

    // Create counts tensor if requested
    let counts = if return_counts {
        let mut counts_vec = vec![0usize; unique_values.len()];
        for (idx, count) in unique_map.values() {
            if sorted {
                // Find the sorted position
                let sorted_pos = sorted_indices.iter().position(|&i| i == *idx).unwrap();
                counts_vec[sorted_pos] = *count;
            } else {
                counts_vec[*idx] = *count;
            }
        }

        let counts_data: Vec<f32> = counts_vec.iter().map(|&c| c as f32).collect();
        Some(Tensor::from_vec(counts_data, &[unique_values.len()])?)
    } else {
        None
    };

    Ok((output, inverse_indices, counts))
}

/// Count number of occurrences of each value in an array of non-negative integers
///
/// Similar to torch.bincount().
///
/// # Arguments
/// * `input` - 1D tensor of non-negative integers
/// * `weights` - Optional weights tensor, same shape as input
/// * `minlength` - Minimum number of bins (output size will be at least this)
///
/// # Returns
/// Tensor of size max(input) + 1 if minlength is None, else max(max(input) + 1, minlength)
pub fn bincount(
    input: &Tensor,
    weights: Option<&Tensor>,
    minlength: Option<usize>,
) -> TorshResult<Tensor> {
    // Check input is 1D
    if input.ndim() != 1 {
        return Err(TorshError::dimension_error(
            "input must be 1-dimensional",
            "bincount",
        ));
    }

    // Check weights shape if provided
    if let Some(w) = weights {
        if w.shape() != input.shape() {
            return Err(TorshError::ShapeMismatch {
                expected: input.shape().dims().to_vec(),
                got: w.shape().dims().to_vec(),
            });
        }
    }

    let data = input.to_vec()?;
    let values: Vec<f32> = data;

    // Check all values are non-negative integers
    let mut max_val = 0i32;
    for &val in &values {
        if val < 0.0 || val.fract() != 0.0 {
            return Err(TorshError::InvalidArgument(
                "bincount: input must contain only non-negative integers".to_string(),
            ));
        }
        max_val = max_val.max(val as i32);
    }

    // Determine output size
    let output_size = if let Some(min_len) = minlength {
        (max_val as usize + 1).max(min_len)
    } else {
        max_val as usize + 1
    };

    // Count occurrences
    let mut counts = vec![0.0f32; output_size];

    if let Some(weights_tensor) = weights {
        // Weighted bincount
        let weights_data = weights_tensor.to_vec()?;
        let weights_values: Vec<f32> = weights_data;

        for (i, &val) in values.iter().enumerate() {
            let idx = val as usize;
            counts[idx] += weights_values[i];
        }
    } else {
        // Unweighted bincount
        for &val in &values {
            let idx = val as usize;
            counts[idx] += 1.0;
        }
    }

    Tensor::from_vec(counts, &[output_size])
}

/// Compute histogram of a tensor
///
/// Similar to torch.histogram() and torch.histc().
///
/// # Arguments
/// * `input` - Input tensor
/// * `bins` - Number of equal-width bins
/// * `min` - Lower end of the range (inclusive). If None, uses the minimum value in input
/// * `max` - Upper end of the range (inclusive). If None, uses the maximum value in input
/// * `density` - If true, the result is normalized to form a probability density
///
/// # Returns
/// * `hist` - The histogram tensor
/// * `bin_edges` - The edges of the bins
pub fn histogram(
    input: &Tensor,
    bins: usize,
    min: Option<f32>,
    max: Option<f32>,
    density: bool,
) -> TorshResult<(Tensor, Tensor)> {
    if bins == 0 {
        return Err(TorshError::InvalidArgument(
            "histogram: bins must be > 0".to_string(),
        ));
    }

    // Flatten the input
    let flattened = input.view(&[-1])?;
    let data = flattened.to_vec()?;
    let len = flattened.shape().numel();

    // Extract values
    let values: Vec<f32> = data;

    // Find min and max if not provided
    let min_val = min.unwrap_or_else(|| values.iter().cloned().fold(f32::INFINITY, f32::min));
    let max_val = max.unwrap_or_else(|| values.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    if min_val > max_val {
        return Err(TorshError::InvalidArgument(
            "histogram: min must be less than or equal to max".to_string(),
        ));
    }

    // Handle edge case where all values are the same
    let range = if max_val == min_val {
        1.0
    } else {
        max_val - min_val
    };

    // Compute bin edges
    let mut bin_edges = vec![0.0f32; bins + 1];
    for i in 0..=bins {
        bin_edges[i] = min_val + (i as f32 / bins as f32) * range;
    }

    // Count values in each bin
    let mut hist = vec![0.0f32; bins];
    for &val in &values {
        if val >= min_val && val <= max_val {
            // Find the bin index
            let mut bin_idx = ((val - min_val) / range * bins as f32) as usize;

            // Handle the edge case where val == max_val
            if bin_idx >= bins {
                bin_idx = bins - 1;
            }

            hist[bin_idx] += 1.0;
        }
    }

    // Normalize if density is requested
    if density {
        let bin_width = range / bins as f32;
        let total_count = len as f32;
        for h in &mut hist {
            *h /= total_count * bin_width;
        }
    }

    let hist_tensor = Tensor::from_vec(hist, &[bins])?;

    let edges_tensor = Tensor::from_vec(bin_edges, &[bins + 1])?;

    Ok((hist_tensor, edges_tensor))
}

/// Compute histogram of a tensor with specified bin edges
///
/// Similar to torch.histogram() with bin edges.
///
/// # Arguments
/// * `input` - Input tensor
/// * `bin_edges` - Monotonically increasing tensor of bin edges
/// * `density` - If true, the result is normalized to form a probability density
///
/// # Returns
/// The histogram tensor
pub fn histogram_with_edges(
    input: &Tensor,
    bin_edges: &Tensor,
    density: bool,
) -> TorshResult<Tensor> {
    // Check bin_edges is 1D
    if bin_edges.ndim() != 1 {
        return Err(TorshError::dimension_error(
            "bin_edges must be 1-dimensional",
            "histogram",
        ));
    }

    let num_edges = bin_edges.shape().numel();
    if num_edges < 2 {
        return Err(TorshError::InvalidArgument(
            "histogram: bin_edges must have at least 2 elements".to_string(),
        ));
    }

    let bins = num_edges - 1;

    // Get bin edges data
    let edges_data = bin_edges.to_vec()?;
    let edges: Vec<f32> = edges_data;

    // Check edges are monotonically increasing
    for i in 1..edges.len() {
        if edges[i] < edges[i - 1] {
            return Err(TorshError::InvalidArgument(
                "histogram: bin_edges must be monotonically increasing".to_string(),
            ));
        }
    }

    // Flatten input
    let flattened = input.view(&[-1])?;
    let data = flattened.to_vec()?;
    let len = flattened.shape().numel();

    // Extract values
    let values: Vec<f32> = data;

    // Count values in each bin
    let mut hist = vec![0.0f32; bins];
    for &val in &values {
        // Binary search for the bin
        if val >= edges[0] && val <= edges[bins] {
            let mut left = 0;
            let mut right = bins;

            while left < right {
                let mid = left + (right - left) / 2;
                if val < edges[mid + 1] {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }

            // Handle edge case where val == edges[bins]
            if left >= bins {
                left = bins - 1;
            }

            hist[left] += 1.0;
        }
    }

    // Normalize if density is requested
    if density {
        let total_count = len as f32;
        for i in 0..bins {
            let bin_width = edges[i + 1] - edges[i];
            hist[i] /= total_count * bin_width;
        }
    }

    Tensor::from_vec(hist, &[bins])
}

/// Count the frequency of each value in an array of non-negative integers
///
/// This is an alias for bincount without weights.
pub fn value_counts(input: &Tensor) -> TorshResult<Tensor> {
    bincount(input, None, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::tensor;

    #[test]
    fn test_unique_basic() {
        let input = tensor![3.0, 1.0, 2.0, 1.0, 3.0, 2.0].unwrap();
        let (output, inverse, counts) = unique(&input, true, true, true, None).unwrap();

        // Check unique values are sorted
        let unique_data = output.data().unwrap();
        let unique_vals: Vec<f32> = unique_data.clone();
        assert_eq!(unique_vals, vec![1.0, 2.0, 3.0]);

        // Check counts
        if let Some(counts_tensor) = counts {
            let counts_data = counts_tensor.data().unwrap();
            let counts_vals: Vec<f32> = counts_data.clone();
            assert_eq!(counts_vals, vec![2.0, 2.0, 2.0]);
        }

        // Check inverse indices can reconstruct original
        if let Some(inv) = inverse {
            let inv_data = inv.data().unwrap();
            let inv_vals: Vec<f32> = inv_data.clone();
            let reconstructed: Vec<f32> = inv_vals
                .iter()
                .map(|&idx| unique_vals[idx as usize])
                .collect();
            assert_eq!(reconstructed, vec![3.0, 1.0, 2.0, 1.0, 3.0, 2.0]);
        }
    }

    #[test]
    fn test_bincount_basic() {
        let input = tensor![0.0, 1.0, 1.0, 3.0, 2.0, 1.0, 3.0].unwrap();
        let output = bincount(&input, None, None).unwrap();

        let data = output.data().unwrap();
        let counts: Vec<f32> = data.clone();
        assert_eq!(counts, vec![1.0, 3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_bincount_weighted() {
        let input = tensor![0.0, 1.0, 1.0, 2.0, 2.0, 2.0].unwrap();
        let weights = tensor![1.0, 2.0, 3.0, 4.0, 5.0, 6.0].unwrap();
        let output = bincount(&input, Some(&weights), None).unwrap();

        let data = output.data().unwrap();
        let weighted_counts: Vec<f32> = data.clone();
        assert_eq!(weighted_counts, vec![1.0, 5.0, 15.0]);
    }

    #[test]
    fn test_histogram_basic() {
        let input = tensor![1.0, 2.0, 3.0, 4.0, 5.0].unwrap();
        let (hist, edges) = histogram(&input, 5, Some(1.0), Some(5.0), false).unwrap();

        let hist_data = hist.data().unwrap();
        let hist_vals: Vec<f32> = hist_data.clone();
        assert_eq!(hist_vals, vec![1.0, 1.0, 1.0, 1.0, 1.0]);

        let edges_data = edges.data().unwrap();
        let edges_vals: Vec<f32> = edges_data.clone();
        assert_eq!(edges_vals.len(), 6); // bins + 1
    }

    #[test]
    fn test_histogram_density() {
        let input = tensor![1.0, 2.0, 3.0, 4.0, 5.0].unwrap();
        let (hist, _) = histogram(&input, 5, Some(1.0), Some(5.0), true).unwrap();

        let hist_data = hist.data().unwrap();
        let hist_vals: Vec<f32> = hist_data.clone();

        // With density=true, the integral of histogram * bin_width should be 1
        let bin_width = 4.0 / 5.0; // (5.0 - 1.0) / 5
        let integral: f32 = hist_vals.iter().sum::<f32>() * bin_width;
        assert!((integral - 1.0).abs() < 1e-6);
    }
}
