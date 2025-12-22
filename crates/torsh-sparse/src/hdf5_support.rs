//! HDF5 sparse matrix support
//!
//! This module provides I/O functionality for sparse matrices using the HDF5 format,
//! which is widely used in scientific computing for efficient storage of large datasets.

#[cfg(feature = "hdf5_support")]
use chrono;
#[cfg(feature = "hdf5_support")]
use hdf5::{File, Group};

use crate::*;
use std::collections::HashMap;
use std::path::Path;
use torsh_core::Result as TorshResult;

/// HDF5 sparse matrix metadata
#[derive(Debug, Clone)]
pub struct Hdf5SparseMetadata {
    /// Matrix format (COO, CSR, CSC, etc.)
    pub format: String,
    /// Data type (f32, f64, etc.)
    pub dtype: String,
    /// Matrix shape [rows, cols]
    pub shape: Vec<usize>,
    /// Number of non-zero elements
    pub nnz: usize,
    /// Sparsity ratio
    pub sparsity: f64,
    /// Creation timestamp
    pub created: String,
    /// ToRSh version
    pub torsh_version: String,
    /// Additional custom attributes
    pub attributes: HashMap<String, String>,
}

impl Hdf5SparseMetadata {
    /// Create new metadata
    pub fn new(
        format: String,
        dtype: String,
        shape: Vec<usize>,
        nnz: usize,
        sparsity: f64,
    ) -> Self {
        #[cfg(feature = "hdf5_support")]
        let created = chrono::Utc::now().to_rfc3339();
        #[cfg(not(feature = "hdf5_support"))]
        let created = "unknown".to_string();

        Self {
            format,
            dtype,
            shape,
            nnz,
            sparsity,
            created,
            torsh_version: env!("CARGO_PKG_VERSION").to_string(),
            attributes: HashMap::new(),
        }
    }

    /// Add custom attribute
    pub fn add_attribute(&mut self, key: String, value: String) {
        self.attributes.insert(key, value);
    }
}

/// HDF5 sparse matrix group structure
pub struct Hdf5SparseGroup {
    /// Group name in HDF5 file
    pub name: String,
    /// Metadata
    pub metadata: Hdf5SparseMetadata,
    /// Data arrays organization
    pub data_structure: Hdf5SparseDataStructure,
}

/// HDF5 data structure for different sparse formats
#[derive(Debug, Clone)]
pub enum Hdf5SparseDataStructure {
    /// COO format: separate arrays for rows, cols, values
    Coo {
        rows_dataset: String,
        cols_dataset: String,
        values_dataset: String,
    },
    /// CSR format: row pointers, column indices, values
    Csr {
        row_ptr_dataset: String,
        col_indices_dataset: String,
        values_dataset: String,
    },
    /// CSC format: column pointers, row indices, values
    Csc {
        col_ptr_dataset: String,
        row_indices_dataset: String,
        values_dataset: String,
    },
    /// BSR format: block data with row/col pointers
    Bsr {
        block_data_dataset: String,
        row_ptr_dataset: String,
        col_indices_dataset: String,
        block_shape_dataset: String,
    },
}

impl Hdf5SparseDataStructure {
    /// Get dataset names for the format
    pub fn dataset_names(&self) -> Vec<String> {
        match self {
            Self::Coo {
                rows_dataset,
                cols_dataset,
                values_dataset,
            } => {
                vec![
                    rows_dataset.clone(),
                    cols_dataset.clone(),
                    values_dataset.clone(),
                ]
            }
            Self::Csr {
                row_ptr_dataset,
                col_indices_dataset,
                values_dataset,
            } => {
                vec![
                    row_ptr_dataset.clone(),
                    col_indices_dataset.clone(),
                    values_dataset.clone(),
                ]
            }
            Self::Csc {
                col_ptr_dataset,
                row_indices_dataset,
                values_dataset,
            } => {
                vec![
                    col_ptr_dataset.clone(),
                    row_indices_dataset.clone(),
                    values_dataset.clone(),
                ]
            }
            Self::Bsr {
                block_data_dataset,
                row_ptr_dataset,
                col_indices_dataset,
                block_shape_dataset,
            } => {
                vec![
                    block_data_dataset.clone(),
                    row_ptr_dataset.clone(),
                    col_indices_dataset.clone(),
                    block_shape_dataset.clone(),
                ]
            }
        }
    }
}

/// HDF5 sparse matrix I/O utilities
pub struct Hdf5SparseIO;

impl Hdf5SparseIO {
    /// Export sparse tensor to HDF5 file
    #[cfg(feature = "hdf5_support")]
    pub fn export_to_hdf5(
        sparse: &dyn SparseTensor,
        filepath: &Path,
        group_name: &str,
    ) -> TorshResult<()> {
        let file = File::create(filepath).map_err(|e| {
            torsh_core::TorshError::Other(format!("Failed to create HDF5 file: {}", e))
        })?;

        let group = file
            .create_group(group_name)
            .map_err(|e| torsh_core::TorshError::Other(format!("Failed to create group: {}", e)))?;

        // Create metadata
        let shape = sparse.shape();
        let metadata = Hdf5SparseMetadata::new(
            format!("{:?}", sparse.format()),
            format!("{:?}", sparse.dtype()),
            shape.dims().to_vec(),
            sparse.nnz(),
            sparse.sparsity() as f64,
        );

        // Write metadata attributes
        Self::write_metadata(&group, &metadata)?;

        // Export data based on format
        match sparse.format() {
            SparseFormat::Coo => Self::export_coo_data(&group, sparse)?,
            SparseFormat::Csr => Self::export_csr_data(&group, sparse)?,
            SparseFormat::Csc => Self::export_csc_data(&group, sparse)?,
            _ => {
                // Convert to COO format for other formats
                let coo = sparse.to_coo()?;
                Self::export_coo_data(&group, &coo)?;
            }
        }

        Ok(())
    }

    /// Import sparse tensor from HDF5 file
    #[cfg(feature = "hdf5_support")]
    pub fn import_from_hdf5(
        filepath: &Path,
        group_name: &str,
    ) -> TorshResult<Box<dyn SparseTensor>> {
        let file = File::open(filepath).map_err(|e| {
            torsh_core::TorshError::Other(format!("Failed to open HDF5 file: {}", e))
        })?;

        let group = file
            .group(group_name)
            .map_err(|e| torsh_core::TorshError::Other(format!("Failed to open group: {}", e)))?;

        // Read metadata
        let metadata = Self::read_metadata(&group)?;

        // Import data based on format
        match metadata.format.as_str() {
            "Coo" => Self::import_coo_data(&group, &metadata),
            "Csr" => Self::import_csr_data(&group, &metadata),
            "Csc" => Self::import_csc_data(&group, &metadata),
            _ => {
                // Default to COO if format is unknown
                Self::import_coo_data(&group, &metadata)
            }
        }
    }

    #[cfg(feature = "hdf5_support")]
    fn write_metadata(group: &Group, metadata: &Hdf5SparseMetadata) -> TorshResult<()> {
        // Write basic attributes as byte arrays (HDF5 compatible)
        let format_bytes = metadata.format.as_bytes();
        group
            .new_attr::<u8>()
            .shape(format_bytes.len())
            .create("format")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .write(format_bytes)
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        let dtype_bytes = metadata.dtype.as_bytes();
        group
            .new_attr::<u8>()
            .shape(dtype_bytes.len())
            .create("dtype")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .write(dtype_bytes)
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        group
            .new_attr::<usize>()
            .shape([metadata.shape.len()])
            .create("shape")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .write(&metadata.shape)
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        group
            .new_attr::<usize>()
            .shape(())
            .create("nnz")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .write_scalar(&metadata.nnz)
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        group
            .new_attr::<f64>()
            .shape(())
            .create("sparsity")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .write_scalar(&metadata.sparsity)
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        let created_bytes = metadata.created.as_bytes();
        group
            .new_attr::<u8>()
            .shape(created_bytes.len())
            .create("created")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .write(created_bytes)
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        let version_bytes = metadata.torsh_version.as_bytes();
        group
            .new_attr::<u8>()
            .shape(version_bytes.len())
            .create("torsh_version")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .write(version_bytes)
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        // Write custom attributes
        for (key, value) in &metadata.attributes {
            let value_bytes = value.as_bytes();
            group
                .new_attr::<u8>()
                .shape(value_bytes.len())
                .create(format!("custom_{}", key).as_str())
                .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
                .write(value_bytes)
                .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        }

        Ok(())
    }

    #[cfg(feature = "hdf5_support")]
    fn read_metadata(group: &Group) -> TorshResult<Hdf5SparseMetadata> {
        let format_array = group
            .attr("format")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .read()
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let format_bytes: Vec<u8> = format_array.to_vec();
        let format = String::from_utf8(format_bytes)
            .map_err(|e| torsh_core::TorshError::Other(format!("UTF-8 error: {}", e)))?;

        let dtype_array = group
            .attr("dtype")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .read()
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let dtype_bytes: Vec<u8> = dtype_array.to_vec();
        let dtype = String::from_utf8(dtype_bytes)
            .map_err(|e| torsh_core::TorshError::Other(format!("UTF-8 error: {}", e)))?;

        let shape_array = group
            .attr("shape")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .read()
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let shape: Vec<usize> = shape_array.to_vec();

        let nnz: usize = group
            .attr("nnz")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .read_scalar()
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        let sparsity: f64 = group
            .attr("sparsity")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .read_scalar()
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        let created_array = group
            .attr("created")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .read()
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let created_bytes: Vec<u8> = created_array.to_vec();
        let created = String::from_utf8(created_bytes)
            .map_err(|e| torsh_core::TorshError::Other(format!("UTF-8 error: {}", e)))?;

        let version_array = group
            .attr("torsh_version")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .read()
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let version_bytes: Vec<u8> = version_array.to_vec();
        let torsh_version = String::from_utf8(version_bytes)
            .map_err(|e| torsh_core::TorshError::Other(format!("UTF-8 error: {}", e)))?;

        // Read custom attributes
        let mut attributes = HashMap::new();

        // Read custom attributes by iterating through all attributes
        // The HDF5 Rust API provides attribute iteration capabilities
        let attr_names: Vec<String> = group.attr_names().map_err(|e| {
            torsh_core::TorshError::Other(format!("HDF5 error reading attribute names: {}", e))
        })?;

        for attr_name in attr_names {
            // Only process attributes with "custom_" prefix
            if let Some(custom_key) = attr_name.strip_prefix("custom_") {
                // Try to read the custom attribute as a byte array
                match group.attr(&attr_name) {
                    Ok(attr) => {
                        // Read the attribute as a raw array, then convert to bytes
                        match attr.read_raw() {
                            Ok(attr_data) => {
                                match String::from_utf8(attr_data) {
                                    Ok(attr_value) => {
                                        attributes.insert(custom_key.to_string(), attr_value);
                                    }
                                    Err(_) => {
                                        // Skip non-UTF8 custom attributes
                                        continue;
                                    }
                                }
                            }
                            Err(_) => {
                                // Skip attributes that can't be read
                                continue;
                            }
                        }
                    }
                    Err(_) => {
                        // Skip attributes that can't be opened
                        continue;
                    }
                }
            }
        }

        Ok(Hdf5SparseMetadata {
            format,
            dtype,
            shape,
            nnz,
            sparsity,
            created,
            torsh_version,
            attributes,
        })
    }

    #[cfg(feature = "hdf5_support")]
    fn export_coo_data(group: &Group, sparse: &dyn SparseTensor) -> TorshResult<()> {
        let coo = sparse.to_coo()?;
        let triplets = coo.triplets();

        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut values = Vec::new();

        for (row, col, val) in triplets {
            rows.push(row as u64);
            cols.push(col as u64);
            values.push(val as f64);
        }

        // Create datasets
        group
            .new_dataset::<u64>()
            .shape([rows.len()])
            .create("rows")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .write(&rows)
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        group
            .new_dataset::<u64>()
            .shape([cols.len()])
            .create("cols")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .write(&cols)
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        group
            .new_dataset::<f64>()
            .shape([values.len()])
            .create("values")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .write(&values)
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        Ok(())
    }

    #[cfg(feature = "hdf5_support")]
    fn import_coo_data(
        group: &Group,
        metadata: &Hdf5SparseMetadata,
    ) -> TorshResult<Box<dyn SparseTensor>> {
        let rows_dataset = group
            .dataset("rows")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let cols_dataset = group
            .dataset("cols")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let values_dataset = group
            .dataset("values")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        let rows_array = rows_dataset
            .read()
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let rows: Vec<u64> = rows_array.to_vec();
        let cols_array = cols_dataset
            .read()
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let cols: Vec<u64> = cols_array.to_vec();
        let values_array = values_dataset
            .read()
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let values: Vec<f64> = values_array.to_vec();

        let shape = Shape::new(metadata.shape.clone());
        let mut coo = CooTensor::empty(shape, DType::F32)?;

        for ((row, col), val) in rows.iter().zip(cols.iter()).zip(values.iter()) {
            coo.insert(*row as usize, *col as usize, *val as f32)?;
        }

        Ok(Box::new(coo))
    }

    #[cfg(feature = "hdf5_support")]
    fn export_csr_data(group: &Group, sparse: &dyn SparseTensor) -> TorshResult<()> {
        let csr = sparse.to_csr()?;
        let row_ptr = csr.row_ptr();
        let col_indices = csr.col_indices();
        let values = csr.values();

        let row_ptr_vec: Vec<u64> = row_ptr.iter().map(|&x| x as u64).collect();
        let col_indices_vec: Vec<u64> = col_indices.iter().map(|&x| x as u64).collect();
        let values_vec: Vec<f64> = values.iter().map(|&x| x as f64).collect();

        group
            .new_dataset::<u64>()
            .shape([row_ptr_vec.len()])
            .create("row_ptr")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .write(&row_ptr_vec)
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        group
            .new_dataset::<u64>()
            .shape([col_indices_vec.len()])
            .create("col_indices")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .write(&col_indices_vec)
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        group
            .new_dataset::<f64>()
            .shape([values_vec.len()])
            .create("values")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .write(&values_vec)
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        Ok(())
    }

    #[cfg(feature = "hdf5_support")]
    fn import_csr_data(
        group: &Group,
        metadata: &Hdf5SparseMetadata,
    ) -> TorshResult<Box<dyn SparseTensor>> {
        let row_ptr_dataset = group
            .dataset("row_ptr")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let col_indices_dataset = group
            .dataset("col_indices")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let values_dataset = group
            .dataset("values")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        let row_ptr_array = row_ptr_dataset
            .read()
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let row_ptr: Vec<u64> = row_ptr_array.to_vec();
        let col_indices_array = col_indices_dataset
            .read()
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let col_indices: Vec<u64> = col_indices_array.to_vec();
        let values_array = values_dataset
            .read()
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let values: Vec<f64> = values_array.to_vec();

        let shape = Shape::new(metadata.shape.clone());
        let row_ptr_vec: Vec<usize> = row_ptr.iter().map(|&x| x as usize).collect();
        let col_indices_vec: Vec<usize> = col_indices.iter().map(|&x| x as usize).collect();
        let values_vec: Vec<f32> = values.iter().map(|&x| x as f32).collect();

        let csr = CsrTensor::from_raw_parts(row_ptr_vec, col_indices_vec, values_vec, shape)?;
        Ok(Box::new(csr))
    }

    #[cfg(feature = "hdf5_support")]
    fn export_csc_data(group: &Group, sparse: &dyn SparseTensor) -> TorshResult<()> {
        let csc = sparse.to_csc()?;
        let col_ptr = csc.col_ptr();
        let row_indices = csc.row_indices();
        let values = csc.values();

        let col_ptr_vec: Vec<u64> = col_ptr.iter().map(|&x| x as u64).collect();
        let row_indices_vec: Vec<u64> = row_indices.iter().map(|&x| x as u64).collect();
        let values_vec: Vec<f64> = values.iter().map(|&x| x as f64).collect();

        group
            .new_dataset::<u64>()
            .shape([col_ptr_vec.len()])
            .create("col_ptr")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .write(&col_ptr_vec)
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        group
            .new_dataset::<u64>()
            .shape([row_indices_vec.len()])
            .create("row_indices")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .write(&row_indices_vec)
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        group
            .new_dataset::<f64>()
            .shape([values_vec.len()])
            .create("values")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?
            .write(&values_vec)
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        Ok(())
    }

    #[cfg(feature = "hdf5_support")]
    fn import_csc_data(
        group: &Group,
        metadata: &Hdf5SparseMetadata,
    ) -> TorshResult<Box<dyn SparseTensor>> {
        let col_ptr_dataset = group
            .dataset("col_ptr")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let row_indices_dataset = group
            .dataset("row_indices")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let values_dataset = group
            .dataset("values")
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;

        let col_ptr_array = col_ptr_dataset
            .read()
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let col_ptr: Vec<u64> = col_ptr_array.to_vec();
        let row_indices_array = row_indices_dataset
            .read()
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let row_indices: Vec<u64> = row_indices_array.to_vec();
        let values_array = values_dataset
            .read()
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        let values: Vec<f64> = values_array.to_vec();

        let shape = Shape::new(metadata.shape.clone());
        let col_ptr_vec: Vec<usize> = col_ptr.iter().map(|&x| x as usize).collect();
        let row_indices_vec: Vec<usize> = row_indices.iter().map(|&x| x as usize).collect();
        let values_vec: Vec<f32> = values.iter().map(|&x| x as f32).collect();

        let csc = CscTensor::from_raw_parts(col_ptr_vec, row_indices_vec, values_vec, shape)?;
        Ok(Box::new(csc))
    }

    /// List all sparse matrices in an HDF5 file
    #[cfg(feature = "hdf5_support")]
    pub fn list_sparse_matrices(filepath: &Path) -> TorshResult<Vec<String>> {
        let file = File::open(filepath).map_err(|e| {
            torsh_core::TorshError::Other(format!("Failed to open HDF5 file: {}", e))
        })?;

        let mut matrix_names = Vec::new();

        let members = file
            .member_names()
            .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
        for group_name in members {
            if let Ok(group) = file.group(&group_name) {
                // Check if it's a sparse matrix group by looking for required attributes
                let attr_names = group
                    .attr_names()
                    .map_err(|e| torsh_core::TorshError::Other(format!("HDF5 error: {}", e)))?;
                if attr_names.contains(&"format".to_string())
                    && attr_names.contains(&"nnz".to_string())
                {
                    matrix_names.push(group_name);
                }
            }
        }

        Ok(matrix_names)
    }

    /// Get metadata for a sparse matrix in HDF5 file without loading the data
    #[cfg(feature = "hdf5_support")]
    pub fn get_metadata(filepath: &Path, group_name: &str) -> TorshResult<Hdf5SparseMetadata> {
        let file = File::open(filepath).map_err(|e| {
            torsh_core::TorshError::Other(format!("Failed to open HDF5 file: {}", e))
        })?;

        let group = file
            .group(group_name)
            .map_err(|e| torsh_core::TorshError::Other(format!("Failed to open group: {}", e)))?;

        Self::read_metadata(&group)
    }
}

/// Convenience functions for HDF5 sparse matrix I/O
pub mod hdf5_convenience {
    #[cfg(feature = "hdf5_support")]
    use super::*;
    #[cfg(feature = "hdf5_support")]
    use crate::{SparseFormat, SparseTensor, TorshResult};
    #[cfg(feature = "hdf5_support")]
    use std::path::Path;

    /// Export sparse tensor to HDF5 file with default group name
    #[cfg(feature = "hdf5_support")]
    pub fn save_sparse_matrix(sparse: &dyn SparseTensor, filepath: &Path) -> TorshResult<()> {
        Hdf5SparseIO::export_to_hdf5(sparse, filepath, "sparse_matrix")
    }

    /// Load sparse tensor from HDF5 file with default group name
    #[cfg(feature = "hdf5_support")]
    pub fn load_sparse_matrix(filepath: &Path) -> TorshResult<Box<dyn SparseTensor>> {
        Hdf5SparseIO::import_from_hdf5(filepath, "sparse_matrix")
    }

    /// Save multiple sparse matrices to a single HDF5 file
    #[cfg(feature = "hdf5_support")]
    pub fn save_sparse_matrices(
        matrices: &[(String, &dyn SparseTensor)],
        filepath: &Path,
    ) -> TorshResult<()> {
        let file = File::create(filepath).map_err(|e| {
            torsh_core::TorshError::Other(format!("Failed to create HDF5 file: {}", e))
        })?;

        for (name, matrix) in matrices {
            let group = file.create_group(name).map_err(|e| {
                torsh_core::TorshError::Other(format!("Failed to create group: {}", e))
            })?;

            // Use the internal export methods directly
            let shape = matrix.shape();
            let metadata = Hdf5SparseMetadata::new(
                format!("{:?}", matrix.format()),
                format!("{:?}", matrix.dtype()),
                shape.dims().to_vec(),
                matrix.nnz(),
                matrix.sparsity() as f64,
            );

            Hdf5SparseIO::write_metadata(&group, &metadata)?;

            match matrix.format() {
                SparseFormat::Coo => Hdf5SparseIO::export_coo_data(&group, *matrix)?,
                SparseFormat::Csr => Hdf5SparseIO::export_csr_data(&group, *matrix)?,
                SparseFormat::Csc => Hdf5SparseIO::export_csc_data(&group, *matrix)?,
                _ => {
                    let coo = matrix.to_coo()?;
                    Hdf5SparseIO::export_coo_data(&group, &coo)?;
                }
            }
        }

        Ok(())
    }

    /// Load multiple sparse matrices from a single HDF5 file
    #[cfg(feature = "hdf5_support")]
    pub fn load_sparse_matrices(
        filepath: &Path,
    ) -> TorshResult<Vec<(String, Box<dyn SparseTensor>)>> {
        let matrix_names = Hdf5SparseIO::list_sparse_matrices(filepath)?;
        let mut matrices = Vec::new();

        for name in matrix_names {
            let matrix = Hdf5SparseIO::import_from_hdf5(filepath, &name)?;
            matrices.push((name, matrix));
        }

        Ok(matrices)
    }
}

// Re-export convenience functions at module level for easier access
#[cfg(feature = "hdf5_support")]
pub use hdf5_convenience::*;

// Without HDF5 feature, provide stub implementations
#[cfg(not(feature = "hdf5_support"))]
pub fn save_sparse_matrix(_sparse: &dyn SparseTensor, _filepath: &Path) -> TorshResult<()> {
    Err(torsh_core::TorshError::Other(
        "HDF5 support not enabled. Enable the 'hdf5_support' feature to use this functionality."
            .to_string(),
    ))
}

#[cfg(not(feature = "hdf5_support"))]
pub fn load_sparse_matrix(_filepath: &Path) -> TorshResult<Box<dyn SparseTensor>> {
    Err(torsh_core::TorshError::Other(
        "HDF5 support not enabled. Enable the 'hdf5_support' feature to use this functionality."
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coo::CooTensor;
    use std::path::PathBuf;
    use torsh_core::{DType, Shape};

    #[test]
    fn test_metadata_creation() {
        let metadata = Hdf5SparseMetadata::new(
            "Coo".to_string(),
            "Float32".to_string(),
            vec![100, 100],
            50,
            0.995,
        );

        assert_eq!(metadata.format, "Coo");
        assert_eq!(metadata.nnz, 50);
        assert_eq!(metadata.sparsity, 0.995);
        assert!(!metadata.created.is_empty());
        assert!(!metadata.torsh_version.is_empty());
    }

    #[test]
    fn test_data_structure_dataset_names() {
        let coo_structure = Hdf5SparseDataStructure::Coo {
            rows_dataset: "rows".to_string(),
            cols_dataset: "cols".to_string(),
            values_dataset: "values".to_string(),
        };

        let names = coo_structure.dataset_names();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"rows".to_string()));
        assert!(names.contains(&"cols".to_string()));
        assert!(names.contains(&"values".to_string()));
    }

    #[cfg(feature = "hdf5_support")]
    #[test]
    fn test_hdf5_export_import() {
        let shape = Shape::new(vec![3, 3]);
        let mut coo = CooTensor::empty(shape.clone(), DType::F32).unwrap();

        coo.insert(0, 0, 1.0).unwrap();
        coo.insert(1, 1, 2.0).unwrap();
        coo.insert(2, 2, 3.0).unwrap();

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_sparse.h5");

        // Export
        Hdf5SparseIO::export_to_hdf5(&coo, &file_path, "test_matrix").unwrap();

        // Import
        let imported = Hdf5SparseIO::import_from_hdf5(&file_path, "test_matrix").unwrap();

        assert_eq!(imported.nnz(), 3);
        assert_eq!(imported.shape(), &shape);

        // Clean up
        let _ = std::fs::remove_file(file_path);
    }

    #[cfg(feature = "hdf5_support")]
    #[test]
    fn test_convenience_functions() {
        let shape = Shape::new(vec![2, 2]);
        let mut coo = CooTensor::empty(shape.clone(), DType::F32).unwrap();

        coo.insert(0, 0, 1.0).unwrap();
        coo.insert(1, 1, 2.0).unwrap();

        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_convenience.h5");

        // Save using convenience function
        save_sparse_matrix(&coo, &file_path).unwrap();

        // Load using convenience function
        let loaded = load_sparse_matrix(&file_path).unwrap();

        assert_eq!(loaded.nnz(), 2);
        assert_eq!(loaded.shape(), &shape);

        // Clean up
        let _ = std::fs::remove_file(file_path);
    }

    #[test]
    fn test_stub_functions_without_feature() {
        let shape = Shape::new(vec![2, 2]);
        let coo = CooTensor::empty(shape, DType::F32).unwrap();
        let temp_path = PathBuf::from("/tmp/test.h5");

        // These should return errors when HDF5 feature is not enabled
        #[cfg(not(feature = "hdf5_support"))]
        {
            assert!(save_sparse_matrix(&coo, &temp_path).is_err());
            assert!(load_sparse_matrix(&temp_path).is_err());
        }

        // Avoid unused variable warning when feature is enabled
        #[cfg(feature = "hdf5_support")]
        {
            let _ = (coo, temp_path);
        }
    }
}
