//! Dataset trait and implementations

use torsh_core::error::Result;
use torsh_tensor::Tensor;

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

/// A map-style dataset
///
/// Represents a dataset that supports random access with a known length.
pub trait Dataset: Send + Sync {
    /// The type of items returned by the dataset
    type Item;

    /// Returns the number of items in the dataset
    fn len(&self) -> usize;

    /// Returns true if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a single item from the dataset
    fn get(&self, index: usize) -> Result<Self::Item>;
}

/// An iterable-style dataset
///
/// Represents a dataset that can be iterated over but may not support
/// random access or have a known length.
pub trait IterableDataset: Send + Sync {
    /// The type of items returned by the dataset
    type Item;
    /// The iterator type
    type Iter: Iterator<Item = Result<Self::Item>> + Send;

    /// Create an iterator over the dataset
    fn iter(&self) -> Self::Iter;
}

/// A simple dataset wrapping tensors
pub struct TensorDataset<T = f32>
where
    T: torsh_core::dtype::TensorElement,
{
    tensors: Vec<Tensor<T>>,
}

impl<T: torsh_core::dtype::TensorElement> TensorDataset<T> {
    /// Create a new tensor dataset from a vector of tensors
    pub fn new(tensors: Vec<Tensor<T>>) -> Self {
        // Verify all tensors have the same first dimension
        if !tensors.is_empty() {
            let first_dim = tensors[0].size(0).unwrap_or(0);
            for tensor in &tensors[1..] {
                assert_eq!(
                    tensor.size(0).unwrap_or(0),
                    first_dim,
                    "All tensors must have the same first dimension"
                );
            }
        }

        Self { tensors }
    }

    /// Create from a single tensor, treating the first dimension as the dataset size
    pub fn from_tensor(tensor: Tensor<T>) -> Self {
        Self::new(vec![tensor])
    }

    /// Create from multiple tensors (e.g., features and labels)
    pub fn from_tensors(tensors: Vec<Tensor<T>>) -> Self {
        Self::new(tensors)
    }
}

impl<T: torsh_core::dtype::TensorElement> Dataset for TensorDataset<T> {
    type Item = Vec<Tensor<T>>;

    fn len(&self) -> usize {
        if self.tensors.is_empty() {
            0
        } else {
            self.tensors[0].size(0).unwrap_or(0)
        }
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.len() {
            return Err(torsh_core::error::TorshError::IndexError {
                index,
                size: self.len(),
            });
        }

        // Extract the index-th element from each tensor
        let mut items = Vec::with_capacity(self.tensors.len());
        for tensor in &self.tensors {
            // TODO: Implement proper indexing when available
            // For now, return a clone of the whole tensor
            items.push(tensor.clone());
        }

        Ok(items)
    }
}

/// A dataset that concatenates multiple datasets
pub struct ConcatDataset<D: Dataset> {
    datasets: Vec<D>,
    cumulative_sizes: Vec<usize>,
}

impl<D: Dataset> ConcatDataset<D> {
    /// Create a new concatenated dataset
    pub fn new(datasets: Vec<D>) -> Self {
        let mut cumulative_sizes = Vec::with_capacity(datasets.len());
        let mut total = 0;

        for dataset in &datasets {
            total += dataset.len();
            cumulative_sizes.push(total);
        }

        Self {
            datasets,
            cumulative_sizes,
        }
    }

    /// Find which dataset an index belongs to
    fn dataset_idx(&self, index: usize) -> Option<(usize, usize)> {
        for (dataset_idx, &cumsum) in self.cumulative_sizes.iter().enumerate() {
            if index < cumsum {
                let dataset_offset = if dataset_idx == 0 {
                    0
                } else {
                    self.cumulative_sizes[dataset_idx - 1]
                };
                return Some((dataset_idx, index - dataset_offset));
            }
        }
        None
    }
}

impl<D: Dataset> Dataset for ConcatDataset<D> {
    type Item = D::Item;

    fn len(&self) -> usize {
        self.cumulative_sizes.last().copied().unwrap_or(0)
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if let Some((dataset_idx, sample_idx)) = self.dataset_idx(index) {
            self.datasets[dataset_idx].get(sample_idx)
        } else {
            Err(torsh_core::error::TorshError::IndexError {
                index,
                size: self.len(),
            })
        }
    }
}

/// A subset of a dataset
pub struct Subset<D: Dataset> {
    dataset: D,
    indices: Vec<usize>,
}

impl<D: Dataset> Subset<D> {
    /// Create a new subset with the given indices
    pub fn new(dataset: D, indices: Vec<usize>) -> Self {
        Self { dataset, indices }
    }
}

impl<D: Dataset> Dataset for Subset<D> {
    type Item = D::Item;

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.indices.len() {
            return Err(torsh_core::error::TorshError::IndexError {
                index,
                size: self.len(),
            });
        }

        let actual_index = self.indices[index];
        self.dataset.get(actual_index)
    }
}

/// Split a dataset into train and validation sets
pub fn random_split<D>(
    dataset: D,
    lengths: &[usize],
    generator: Option<u64>,
) -> Result<Vec<Subset<D>>>
where
    D: Dataset + Clone,
{
    let total_length: usize = lengths.iter().sum();
    if total_length != dataset.len() {
        return Err(torsh_core::error::TorshError::InvalidArgument(format!(
            "Sum of lengths {} does not equal dataset length {}",
            total_length,
            dataset.len()
        )));
    }

    // Create indices
    let mut indices: Vec<usize> = (0..dataset.len()).collect();

    // Shuffle indices if generator seed is provided
    if let Some(seed) = generator {
        use rand::rngs::StdRng;
        use rand::{seq::SliceRandom, SeedableRng};

        let mut rng = StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
    }

    // Split indices according to lengths
    let mut subsets = Vec::with_capacity(lengths.len());
    let mut offset = 0;

    for &length in lengths {
        let subset_indices = indices[offset..offset + length].to_vec();
        subsets.push(Subset::new(dataset.clone(), subset_indices));
        offset += length;
    }

    Ok(subsets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;

    #[test]
    fn test_tensor_dataset() {
        let data = ones::<f32>(&[10, 3]);
        let labels = zeros::<f32>(&[10]);

        let dataset = TensorDataset::from_tensors(vec![data, labels]);
        assert_eq!(dataset.len(), 10);

        let item = dataset.get(0).unwrap();
        assert_eq!(item.len(), 2);
    }

    #[test]
    fn test_concat_dataset() {
        let ds1 = TensorDataset::from_tensor(ones::<f32>(&[5, 3]));
        let ds2 = TensorDataset::from_tensor(zeros::<f32>(&[3, 3]));

        let concat = ConcatDataset::new(vec![ds1, ds2]);
        assert_eq!(concat.len(), 8);

        // Test dataset index calculation
        assert_eq!(concat.dataset_idx(0), Some((0, 0)));
        assert_eq!(concat.dataset_idx(4), Some((0, 4)));
        assert_eq!(concat.dataset_idx(5), Some((1, 0)));
        assert_eq!(concat.dataset_idx(7), Some((1, 2)));
        assert_eq!(concat.dataset_idx(8), None);
    }

    #[test]
    fn test_subset() {
        let dataset = TensorDataset::from_tensor(ones::<f32>(&[10, 3]));
        let subset = Subset::new(dataset, vec![0, 2, 4, 6, 8]);

        assert_eq!(subset.len(), 5);
        assert!(subset.get(0).is_ok());
        assert!(subset.get(5).is_err());
    }
}
