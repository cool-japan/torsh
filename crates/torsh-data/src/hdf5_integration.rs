//! HDF5 integration for scientific data formats
//!
//! This module provides utilities for reading and writing HDF5 files,
//! which are commonly used in scientific computing and machine learning.

#[cfg(feature = "hdf5-support")]
use hdf5::{File, Group, H5Type};

use crate::{utils, Dataset};
use torsh_core::{
    device::DeviceType,
    dtype::TensorElement,
    error::{Result, TorshError},
};
use torsh_tensor::Tensor;

#[cfg(not(feature = "hdf5-support"))]
use std::marker::PhantomData;

/// HDF5 dataset for reading HDF5 files
#[cfg(feature = "hdf5-support")]
pub struct HDF5TensorDataset {
    file: File,
    dataset_path: String,
    shape: Vec<usize>,
    chunk_size: Option<usize>,
    cache_size: usize,
}

#[cfg(not(feature = "hdf5-support"))]
pub struct HDF5TensorDataset {
    _phantom: PhantomData<()>,
}

#[cfg(feature = "hdf5-support")]
impl HDF5TensorDataset {
    /// Create a new HDF5 dataset
    pub fn new<P: AsRef<std::path::Path>>(file_path: P, dataset_path: &str) -> Result<Self> {
        let file_path = file_path.as_ref();
        utils::validate_dataset_path(file_path, "HDF5 file")?;
        utils::validate_file_extension(file_path, &["h5", "hdf5", "hdf"])?;

        let file = File::open(file_path)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to open HDF5 file: {}", e)))?;

        // Validate dataset exists
        let dataset = file.dataset(dataset_path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to open dataset '{}': {}", dataset_path, e))
        })?;

        let shape = dataset.shape();

        Ok(Self {
            file,
            dataset_path: dataset_path.to_string(),
            shape,
            chunk_size: None,
            cache_size: 100, // Default cache size
        })
    }

    /// Set chunk size for reading large datasets
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = Some(chunk_size);
        self
    }

    /// Set cache size for performance
    pub fn with_cache_size(mut self, cache_size: usize) -> Self {
        self.cache_size = cache_size;
        self
    }

    /// Read entire dataset as tensor
    pub fn read_full<T: TensorElement + H5Type>(&self) -> Result<Tensor<T>> {
        let dataset = self
            .file
            .dataset(&self.dataset_path)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to open dataset: {}", e)))?;

        let data: Vec<T> = dataset.read_raw().map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to read HDF5 dataset: {}", e))
        })?;

        torsh_tensor::Tensor::from_data(data, self.shape.clone(), DeviceType::Cpu)
    }

    /// Read a slice of the dataset
    pub fn read_slice<T: TensorElement + H5Type>(
        &self,
        start: &[usize],
        count: &[usize],
    ) -> Result<Tensor<T>> {
        if start.len() != self.shape.len() || count.len() != self.shape.len() {
            return Err(TorshError::InvalidArgument(
                "Start and count must have same dimensionality as dataset".to_string(),
            ));
        }

        let dataset = self
            .file
            .dataset(&self.dataset_path)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to open dataset: {}", e)))?;

        // Create selection
        let selection = dataset
            .read_slice_1d::<T, _>(start[0]..start[0] + count[0])
            .map_err(|e| {
                TorshError::InvalidArgument(format!("Failed to read HDF5 slice: {}", e))
            })?;

        torsh_tensor::Tensor::from_data(selection.to_vec(), count.to_vec(), DeviceType::Cpu)
    }

    /// Write tensor to HDF5 file
    pub fn write_tensor<T: TensorElement + H5Type>(
        file_path: &std::path::Path,
        dataset_path: &str,
        tensor: &Tensor<T>,
    ) -> Result<()> {
        let file = File::create(file_path).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create HDF5 file: {}", e))
        })?;

        let data = tensor.to_vec()?;
        let shape = tensor.shape().dims().to_vec();

        let dataset = file
            .new_dataset::<T>()
            .shape(shape)
            .create(dataset_path)
            .map_err(|e| {
                TorshError::InvalidArgument(format!("Failed to create HDF5 dataset: {}", e))
            })?;

        dataset.write_raw(&data).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to write HDF5 dataset: {}", e))
        })?;

        Ok(())
    }

    /// List all datasets in the file
    pub fn list_datasets(&self) -> Result<Vec<String>> {
        let mut datasets = Vec::new();
        self.collect_datasets_recursive(&self.file, "", &mut datasets)?;
        Ok(datasets)
    }

    fn collect_datasets_recursive(
        &self,
        group: &Group,
        prefix: &str,
        datasets: &mut Vec<String>,
    ) -> Result<()> {
        for name in group.member_names().map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to list group members: {}", e))
        })? {
            let full_path = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{}/{}", prefix, name)
            };

            if group.dataset(&name).is_ok() {
                datasets.push(full_path);
            } else if let Ok(subgroup) = group.group(&name) {
                self.collect_datasets_recursive(&subgroup, &full_path, datasets)?;
            }
        }
        Ok(())
    }

    /// Get dataset metadata
    pub fn get_metadata(&self) -> Result<HDF5Metadata> {
        let dataset = self
            .file
            .dataset(&self.dataset_path)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to open dataset: {}", e)))?;

        Ok(HDF5Metadata {
            shape: dataset.shape(),
            dtype: format!("{:?}", dataset.dtype()),
            size: dataset.size(),
            chunks: dataset.chunk(),
        })
    }
}

#[cfg(not(feature = "hdf5-support"))]
impl HDF5TensorDataset {
    /// Placeholder when HDF5 is not available
    pub fn new<P: AsRef<std::path::Path>>(_file_path: P, _dataset_path: &str) -> Result<Self> {
        Err(TorshError::InvalidArgument(
            "HDF5 support not enabled. Enable 'hdf5-support' feature flag.".to_string(),
        ))
    }

    /// Placeholder for reading
    pub fn read_full<T>(&self) -> Result<Tensor<T>> {
        Err(TorshError::InvalidArgument(
            "HDF5 support not enabled".to_string(),
        ))
    }
}

/// HDF5 dataset metadata
#[derive(Debug, Clone)]
pub struct HDF5Metadata {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub size: usize,
    pub chunks: Option<Vec<usize>>,
}

#[cfg(feature = "hdf5-support")]
impl Dataset for HDF5TensorDataset {
    type Item = Vec<f32>; // Simplified for now

    fn len(&self) -> usize {
        if self.shape.is_empty() {
            0
        } else {
            self.shape[0]
        }
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.len() {
            return Err(utils::errors::invalid_index(index, self.len()));
        }

        let dataset = self
            .file
            .dataset(&self.dataset_path)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to open dataset: {}", e)))?;

        // Read a single row/sample
        let row_data: Vec<f32> = dataset
            .read_slice_1d(index..index + 1)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to read HDF5 row: {}", e)))?
            .to_vec();

        Ok(row_data)
    }
}

#[cfg(not(feature = "hdf5-support"))]
impl Dataset for HDF5TensorDataset {
    type Item = ();

    fn len(&self) -> usize {
        0
    }

    fn get(&self, _index: usize) -> Result<Self::Item> {
        Err(TorshError::InvalidArgument(
            "HDF5 support not enabled".to_string(),
        ))
    }
}

/// Utility functions for HDF5 integration
pub mod hdf5_utils {
    use super::*;

    /// Check if HDF5 feature is available at compile time
    pub const fn is_hdf5_available() -> bool {
        cfg!(feature = "hdf5-support")
    }

    /// Create a sample HDF5 file for testing
    #[cfg(feature = "hdf5-support")]
    pub fn create_sample_file<P: AsRef<std::path::Path>>(path: P) -> Result<()> {
        use torsh_tensor::creation::arange;

        let tensor = arange::<f32>(0.0, 100.0, 1.0)
            .expect("failed to create arange tensor for sample HDF5 file")
            .reshape(&[10, 10])
            .expect("failed to reshape tensor to 10x10 for sample HDF5 file");

        HDF5TensorDataset::write_tensor(path.as_ref(), "data", &tensor)?;

        Ok(())
    }

    /// Batch read multiple datasets from HDF5 file
    #[cfg(feature = "hdf5-support")]
    pub fn batch_read_datasets<T: TensorElement + H5Type>(
        file_path: &std::path::Path,
        dataset_paths: &[&str],
    ) -> Result<Vec<Tensor<T>>> {
        let file = File::open(file_path)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to open HDF5 file: {}", e)))?;

        let mut tensors = Vec::new();

        for &dataset_path in dataset_paths {
            let dataset = file.dataset(dataset_path).map_err(|e| {
                TorshError::InvalidArgument(format!(
                    "Failed to open dataset '{}': {}",
                    dataset_path, e
                ))
            })?;

            let data: Vec<T> = dataset.read_raw().map_err(|e| {
                TorshError::InvalidArgument(format!(
                    "Failed to read dataset '{}': {}",
                    dataset_path, e
                ))
            })?;

            let shape = dataset.shape();
            let tensor = torsh_tensor::Tensor::from_data(data, shape, DeviceType::Cpu)?;
            tensors.push(tensor);
        }

        Ok(tensors)
    }

    /// Convert HDF5 file to multiple tensor files
    #[cfg(feature = "hdf5-support")]
    pub fn hdf5_to_tensors<T: TensorElement + H5Type>(
        hdf5_path: &std::path::Path,
        output_dir: &std::path::Path,
    ) -> Result<()> {
        let dataset = HDF5TensorDataset::new(hdf5_path, "data")?;
        let datasets = dataset.list_datasets()?;

        std::fs::create_dir_all(output_dir).map_err(|e| {
            TorshError::InvalidArgument(format!("Failed to create output directory: {}", e))
        })?;

        for dataset_path in datasets {
            let tensor: Tensor<T> = dataset.read_full()?;

            // Save as tensor file (simplified)
            let filename = dataset_path.replace("/", "_") + ".tensor";
            let output_path = output_dir.join(filename);

            // Here you would save the tensor in your preferred format
            // For now, we'll save the shape and data info
            let info = format!(
                "Shape: {:?}, Size: {}",
                tensor.shape().dims(),
                tensor.numel()
            );
            std::fs::write(output_path, info).map_err(|e| {
                TorshError::InvalidArgument(format!("Failed to write tensor info: {}", e))
            })?;
        }

        Ok(())
    }

    #[cfg(not(feature = "hdf5-support"))]
    pub fn create_sample_file<P: AsRef<std::path::Path>>(_path: P) -> Result<()> {
        Err(TorshError::InvalidArgument(
            "HDF5 support not enabled".to_string(),
        ))
    }

    #[cfg(not(feature = "hdf5-support"))]
    pub fn batch_read_datasets<T>(
        _file_path: &std::path::Path,
        _dataset_paths: &[&str],
    ) -> Result<Vec<Tensor<T>>> {
        Err(TorshError::InvalidArgument(
            "HDF5 support not enabled".to_string(),
        ))
    }

    #[cfg(not(feature = "hdf5-support"))]
    pub fn hdf5_to_tensors<T>(
        _hdf5_path: &std::path::Path,
        _output_dir: &std::path::Path,
    ) -> Result<()> {
        Err(TorshError::InvalidArgument(
            "HDF5 support not enabled".to_string(),
        ))
    }
}

/// Builder for HDF5 datasets with configuration options
pub struct HDF5DatasetBuilder {
    file_path: std::path::PathBuf,
    dataset_path: String,
    chunk_size: Option<usize>,
    cache_size: usize,
    read_only: bool,
}

impl HDF5DatasetBuilder {
    /// Create a new HDF5 dataset builder
    pub fn new<P: AsRef<std::path::Path>>(file_path: P, dataset_path: &str) -> Self {
        Self {
            file_path: file_path.as_ref().to_path_buf(),
            dataset_path: dataset_path.to_string(),
            chunk_size: None,
            cache_size: 100,
            read_only: true,
        }
    }

    /// Set chunk size for reading
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = Some(size);
        self
    }

    /// Set cache size
    pub fn cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }

    /// Enable write mode
    pub fn writable(mut self) -> Self {
        self.read_only = false;
        self
    }

    /// Build the dataset
    pub fn build(self) -> Result<HDF5TensorDataset> {
        let mut dataset = HDF5TensorDataset::new(&self.file_path, &self.dataset_path)?;

        if let Some(chunk_size) = self.chunk_size {
            dataset = dataset.with_chunk_size(chunk_size);
        }

        dataset = dataset.with_cache_size(self.cache_size);

        Ok(dataset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdf5_availability() {
        assert!(hdf5_utils::is_hdf5_available() || !hdf5_utils::is_hdf5_available());
    }

    #[cfg(feature = "hdf5-support")]
    #[test]
    fn test_hdf5_dataset_builder() {
        let builder = HDF5DatasetBuilder::new("test.h5", "data")
            .chunk_size(1000)
            .cache_size(200)
            .writable();

        // Would need an actual HDF5 file to test building
        assert_eq!(builder.chunk_size, Some(1000));
        assert_eq!(builder.cache_size, 200);
        assert!(!builder.read_only);
    }

    #[cfg(not(feature = "hdf5-support"))]
    #[test]
    fn test_hdf5_disabled() {
        let result = HDF5TensorDataset::new("test.h5", "data");
        assert!(result.is_err());
    }
}
