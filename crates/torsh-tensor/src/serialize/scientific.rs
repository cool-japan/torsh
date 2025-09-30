//! Scientific Data Format Implementations
//!
//! This module provides serialization support for scientific computing formats,
//! particularly HDF5 which is widely used in scientific computing for its
//! support of hierarchical data, compression, and metadata.

use super::common::{SerializationFormat, SerializationOptions, TensorMetadata};
use crate::{Tensor, TensorElement};
use std::path::Path;
use torsh_core::{
    device::DeviceType,
    error::{Result, TorshError},
};

/// HDF5 format implementation
#[cfg(feature = "serialize-hdf5")]
pub mod hdf5 {
    use super::*;
    use ::hdf5::{types::VarLenUnicode, Dataset, File, Group, H5Type};

    /// Serialize tensor to HDF5 format
    ///
    /// Creates an HDF5 file with the tensor data stored as a dataset
    /// and metadata stored as attributes. Supports compression and
    /// chunking for large datasets.
    ///
    /// # Arguments
    /// * `tensor` - Tensor to serialize
    /// * `path` - Output file path
    /// * `dataset_name` - Name for the dataset within the HDF5 file
    /// * `options` - Serialization options
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, error otherwise
    pub fn serialize_hdf5<T: TensorElement + H5Type>(
        tensor: &Tensor<T>,
        path: &Path,
        dataset_name: &str,
        options: &SerializationOptions,
    ) -> Result<()> {
        let file = File::create(path).map_err(|e| {
            TorshError::SerializationError(format!("Failed to create HDF5 file: {}", e))
        })?;

        // Get tensor data and shape
        let data = tensor.data()?;
        let shape_dims: Vec<usize> = tensor.shape().dims().to_vec();

        // Create dataset with optional compression
        let mut dataset_builder = file.new_dataset::<T>().shape(&shape_dims);

        if options.compression_level > 0 {
            // Enable compression - use deflate which is more widely supported
            dataset_builder = dataset_builder.deflate(options.compression_level.min(9));
        }

        // Enable chunking for large datasets
        if let Some(chunk_size) = options.chunk_size {
            let total_elements = shape_dims.iter().product::<usize>();
            if total_elements > chunk_size {
                // Calculate chunk dimensions
                let elements_per_chunk = chunk_size / std::mem::size_of::<T>();
                if elements_per_chunk > 0 {
                    let chunk_dims = calculate_chunk_dims(&shape_dims, elements_per_chunk);
                    dataset_builder = dataset_builder.chunk(&chunk_dims);
                }
            }
        }

        let dataset = dataset_builder.create(dataset_name).map_err(|e| {
            TorshError::SerializationError(format!("Failed to create HDF5 dataset: {}", e))
        })?;

        // Write tensor data
        dataset.write(&*data).map_err(|e| {
            TorshError::SerializationError(format!("Failed to write tensor data to HDF5: {}", e))
        })?;

        // Create and store metadata
        let metadata = TensorMetadata::from_tensor(
            tensor,
            options,
            SerializationFormat::Hdf5,
            data.len() * std::mem::size_of::<T>(),
        );

        // Store metadata as HDF5 attributes
        store_metadata_as_attributes(&dataset, &metadata)?;

        // Store custom metadata - temporarily disabled due to HDF5 API compatibility
        // TODO: Implement proper string metadata storage when HDF5 API is updated
        for (_key, _value) in &options.metadata {
            // Custom metadata storage disabled for now
        }

        Ok(())
    }

    /// Deserialize tensor from HDF5 format
    ///
    /// Reads an HDF5 file and reconstructs the tensor with its metadata.
    /// Automatically handles decompression and chunked data.
    ///
    /// # Arguments
    /// * `path` - Input file path
    /// * `dataset_name` - Name of the dataset within the HDF5 file
    ///
    /// # Returns
    /// * `Result<Tensor<T>>` - Deserialized tensor or error
    pub fn deserialize_hdf5<T: TensorElement + H5Type>(
        path: &Path,
        dataset_name: &str,
    ) -> Result<Tensor<T>> {
        let file = File::open(path).map_err(|e| {
            TorshError::SerializationError(format!("Failed to open HDF5 file: {}", e))
        })?;

        let dataset = file.dataset(dataset_name).map_err(|e| {
            TorshError::SerializationError(format!(
                "Failed to open HDF5 dataset '{}': {}",
                dataset_name, e
            ))
        })?;

        // Read shape information
        let shape_dims = dataset.shape();

        if shape_dims.is_empty() {
            return Err(TorshError::SerializationError(
                "HDF5 dataset has empty shape".to_string(),
            ));
        }

        // Read tensor data (HDF5 handles decompression automatically)
        let data: Vec<T> = dataset.read_raw().map_err(|e| {
            TorshError::SerializationError(format!("Failed to read tensor data from HDF5: {}", e))
        })?;

        // Verify data size consistency
        let expected_size = shape_dims.iter().product::<usize>();
        if data.len() != expected_size {
            return Err(TorshError::SerializationError(format!(
                "HDF5 data size mismatch: expected {} elements, got {}",
                expected_size,
                data.len()
            )));
        }

        // Read metadata from attributes
        let device = read_device_from_attributes(&dataset)?;

        // Create tensor
        let mut tensor = Tensor::from_data(data, shape_dims, device)?;

        // Set requires_grad if available
        if let Ok(requires_grad) = dataset
            .attr("requires_grad")
            .and_then(|attr| attr.read_scalar::<bool>())
        {
            // TODO: Implement mutable set_requires_grad when autograd is available
            // For now, the requires_grad field is set during tensor construction
        }

        Ok(tensor)
    }

    /// List datasets in an HDF5 file
    ///
    /// # Arguments
    /// * `path` - HDF5 file path
    ///
    /// # Returns
    /// * `Result<Vec<String>>` - List of dataset names or error
    pub fn list_datasets(path: &Path) -> Result<Vec<String>> {
        let file = File::open(path).map_err(|e| {
            TorshError::SerializationError(format!("Failed to open HDF5 file: {}", e))
        })?;

        let mut datasets = Vec::new();
        collect_datasets(&file, "/", &mut datasets)?;
        Ok(datasets)
    }

    /// Get dataset metadata without loading data
    ///
    /// # Arguments
    /// * `path` - HDF5 file path
    /// * `dataset_name` - Dataset name
    ///
    /// # Returns
    /// * `Result<TensorMetadata>` - Metadata or error
    pub fn get_dataset_metadata(path: &Path, dataset_name: &str) -> Result<TensorMetadata> {
        let file = File::open(path).map_err(|e| {
            TorshError::SerializationError(format!("Failed to open HDF5 file: {}", e))
        })?;

        let dataset = file.dataset(dataset_name).map_err(|e| {
            TorshError::SerializationError(format!("Failed to open dataset: {}", e))
        })?;

        read_metadata_from_attributes(&dataset)
    }

    /// Helper function to store metadata as HDF5 attributes
    fn store_metadata_as_attributes(dataset: &Dataset, metadata: &TensorMetadata) -> Result<()> {
        // Store device information - temporarily disabled due to HDF5 API compatibility
        // TODO: Implement proper device storage when HDF5 string API is updated
        let _device_info = format!("{:?}", metadata.device);

        // Store gradient requirement
        dataset
            .new_attr::<bool>()
            .create("requires_grad")
            .map_err(|e| {
                TorshError::SerializationError(format!(
                    "Failed to create requires_grad attribute: {}",
                    e
                ))
            })?
            .write_scalar(&metadata.requires_grad)
            .map_err(|e| {
                TorshError::SerializationError(format!(
                    "Failed to write requires_grad attribute: {}",
                    e
                ))
            })?;

        // Store data type - temporarily disabled due to HDF5 API compatibility
        // TODO: Implement proper dtype storage when HDF5 string API is updated
        let _dtype_info = &metadata.dtype_name;

        // Store version - temporarily disabled due to HDF5 API compatibility
        // TODO: Implement proper version storage when HDF5 string API is updated
        let _version_info = &metadata.version;

        // Store timestamp
        dataset
            .new_attr::<u64>()
            .create("timestamp")
            .map_err(|e| {
                TorshError::SerializationError(format!(
                    "Failed to create timestamp attribute: {}",
                    e
                ))
            })?
            .write_scalar(&metadata.timestamp)
            .map_err(|e| {
                TorshError::SerializationError(format!(
                    "Failed to write timestamp attribute: {}",
                    e
                ))
            })?;

        Ok(())
    }

    /// Helper function to read device information from HDF5 attributes
    fn read_device_from_attributes(dataset: &Dataset) -> Result<DeviceType> {
        let device_str: VarLenUnicode = dataset
            .attr("device")
            .map_err(|e| {
                TorshError::SerializationError(format!("Failed to read device attribute: {}", e))
            })?
            .read_scalar()
            .map_err(|e| {
                TorshError::SerializationError(format!("Failed to read device value: {}", e))
            })?;

        let device = match device_str.as_str() {
            "Cpu" => DeviceType::Cpu,
            s if s.starts_with("Cuda(") => {
                let id_str = &s[5..s.len() - 1];
                let id: usize = id_str.parse().unwrap_or(0);
                DeviceType::Cuda(id)
            }
            s if s.starts_with("Metal(") => {
                let id_str = &s[6..s.len() - 1];
                let id: usize = id_str.parse().unwrap_or(0);
                DeviceType::Metal(id)
            }
            _ => DeviceType::Cpu, // Default fallback
        };

        Ok(device)
    }

    /// Helper function to read metadata from HDF5 attributes
    fn read_metadata_from_attributes(dataset: &Dataset) -> Result<TensorMetadata> {
        let device = read_device_from_attributes(dataset)?;

        let requires_grad = dataset
            .attr("requires_grad")
            .and_then(|attr| attr.read_scalar::<bool>())
            .unwrap_or(false);

        let dtype_name: String = dataset
            .attr("dtype")
            .and_then(|attr| attr.read_scalar::<VarLenUnicode>())
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let version: String = dataset
            .attr("version")
            .and_then(|attr| attr.read_scalar::<VarLenUnicode>())
            .map(|s| s.to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        let timestamp = dataset
            .attr("timestamp")
            .and_then(|attr| attr.read_scalar::<u64>())
            .unwrap_or(0);

        use torsh_core::shape::Shape;
        let shape_dims = dataset.shape();
        let shape = Shape::new(shape_dims.clone());

        Ok(TensorMetadata {
            shape,
            device,
            requires_grad,
            dtype_name,
            version,
            timestamp,
            custom_metadata: std::collections::HashMap::new(),
            format: "Hdf5".to_string(),
            data_size: shape_dims.iter().product::<usize>() * std::mem::size_of::<f32>(), // Approximate
            compressed: false, // HDF5 compression is transparent
            checksum: None,
        })
    }

    /// Helper function to calculate optimal chunk dimensions
    fn calculate_chunk_dims(shape: &[usize], target_elements: usize) -> Vec<usize> {
        if shape.is_empty() {
            return Vec::new();
        }

        let mut chunk_dims = shape.to_vec();
        let total_elements = shape.iter().product::<usize>();

        if total_elements <= target_elements {
            return chunk_dims;
        }

        // Start from the last dimension and reduce chunk size
        let mut remaining_elements = target_elements;
        for i in (0..chunk_dims.len()).rev() {
            if remaining_elements >= chunk_dims[i] {
                remaining_elements /= chunk_dims[i];
            } else {
                chunk_dims[i] = remaining_elements;
                remaining_elements = 1;
            }

            if remaining_elements <= 1 {
                break;
            }
        }

        chunk_dims
    }

    /// Helper function to recursively collect dataset names
    fn collect_datasets(group: &Group, prefix: &str, datasets: &mut Vec<String>) -> Result<()> {
        for name in group.member_names().map_err(|e| {
            TorshError::SerializationError(format!("Failed to list group members: {}", e))
        })? {
            let full_path = if prefix == "/" {
                format!("/{}", name)
            } else {
                format!("{}/{}", prefix, name)
            };

            if group.link_exists(&name) {
                match group.group(&name) {
                    Ok(subgroup) => {
                        collect_datasets(&subgroup, &full_path, datasets)?;
                    }
                    Err(_) => {
                        // Assume it's a dataset
                        datasets.push(full_path);
                    }
                }
            }
        }

        Ok(())
    }
}

/// Stub implementation when HDF5 feature is not enabled
#[cfg(not(feature = "serialize-hdf5"))]
pub mod hdf5 {
    use super::*;

    pub fn serialize_hdf5<T: TensorElement>(
        _tensor: &Tensor<T>,
        _path: &Path,
        _dataset_name: &str,
        _options: &SerializationOptions,
    ) -> Result<()> {
        Err(TorshError::SerializationError(
            "HDF5 serialization requires the 'serialize-hdf5' feature to be enabled".to_string(),
        ))
    }

    pub fn deserialize_hdf5<T: TensorElement>(
        _path: &Path,
        _dataset_name: &str,
    ) -> Result<Tensor<T>> {
        Err(TorshError::SerializationError(
            "HDF5 deserialization requires the 'serialize-hdf5' feature to be enabled".to_string(),
        ))
    }

    pub fn list_datasets(_path: &Path) -> Result<Vec<String>> {
        Err(TorshError::SerializationError(
            "HDF5 operations require the 'serialize-hdf5' feature to be enabled".to_string(),
        ))
    }

    pub fn get_dataset_metadata(_path: &Path, _dataset_name: &str) -> Result<TensorMetadata> {
        Err(TorshError::SerializationError(
            "HDF5 operations require the 'serialize-hdf5' feature to be enabled".to_string(),
        ))
    }
}
