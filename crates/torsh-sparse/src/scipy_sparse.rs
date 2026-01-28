//! SciPy sparse matrix interoperability
//!
//! This module provides conversion between ToRSh sparse tensors and SciPy sparse matrices,
//! enabling seamless integration with Python scientific computing ecosystem.

#[cfg(feature = "scipy")]
use numpy::{PyArray1, PyReadonlyArray1};
#[cfg(feature = "scipy")]
use pyo3::prelude::*;
#[cfg(feature = "scipy")]
use pyo3::types::PyDict;
#[cfg(feature = "scipy")]
use pyo3::Bound;

use crate::*;
use std::collections::HashMap;
use torsh_core::Result as TorshResult;

/// Supported SciPy sparse matrix formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScipyFormat {
    /// Compressed Sparse Row (csr_matrix)
    Csr,
    /// Compressed Sparse Column (csc_matrix)
    Csc,
    /// Coordinate format (coo_matrix)
    Coo,
    /// Block Sparse Row (bsr_matrix)
    Bsr,
    /// Diagonal format (dia_matrix)
    Dia,
}

impl From<SparseFormat> for ScipyFormat {
    fn from(format: SparseFormat) -> Self {
        match format {
            SparseFormat::Coo => ScipyFormat::Coo,
            SparseFormat::Csr => ScipyFormat::Csr,
            SparseFormat::Csc => ScipyFormat::Csc,
            SparseFormat::Bsr => ScipyFormat::Bsr,
            SparseFormat::Dia => ScipyFormat::Dia,
            SparseFormat::Ell => ScipyFormat::Csr, // ELL -> CSR fallback
            SparseFormat::Rle => ScipyFormat::Csr, // RLE -> CSR fallback
            SparseFormat::Symmetric => ScipyFormat::Csr, // Symmetric -> CSR fallback
            SparseFormat::Dsr => ScipyFormat::Csr, // DSR -> CSR fallback
        }
    }
}

impl From<ScipyFormat> for SparseFormat {
    fn from(format: ScipyFormat) -> Self {
        match format {
            ScipyFormat::Coo => SparseFormat::Coo,
            ScipyFormat::Csr => SparseFormat::Csr,
            ScipyFormat::Csc => SparseFormat::Csc,
            ScipyFormat::Bsr => SparseFormat::Bsr,
            ScipyFormat::Dia => SparseFormat::Dia,
        }
    }
}

/// SciPy sparse matrix representation for data exchange
#[derive(Debug, Clone)]
pub struct ScipySparseData {
    /// Matrix format
    pub format: ScipyFormat,
    /// Matrix shape (rows, cols)
    pub shape: (usize, usize),
    /// Data values
    pub data: Vec<f64>,
    /// Row indices (for COO and CSR) or column indices (for CSC)
    pub indices: Vec<usize>,
    /// Row pointers (for CSR) or column pointers (for CSC) or coordinate rows (for COO)
    pub indptr_or_row: Vec<usize>,
    /// Block size for BSR format
    pub blocksize: Option<(usize, usize)>,
    /// Number of diagonals for DIA format
    pub diagonals: Option<Vec<i32>>,
}

impl ScipySparseData {
    /// Create new SciPy sparse data
    pub fn new(format: ScipyFormat, shape: (usize, usize)) -> Self {
        Self {
            format,
            shape,
            data: Vec::new(),
            indices: Vec::new(),
            indptr_or_row: Vec::new(),
            blocksize: None,
            diagonals: None,
        }
    }

    /// Create from COO data
    pub fn from_coo(
        shape: (usize, usize),
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Self {
        Self {
            format: ScipyFormat::Coo,
            shape,
            data: values,
            indices: col_indices,
            indptr_or_row: row_indices,
            blocksize: None,
            diagonals: None,
        }
    }

    /// Create from CSR data
    pub fn from_csr(
        shape: (usize, usize),
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Self {
        Self {
            format: ScipyFormat::Csr,
            shape,
            data: values,
            indices: col_indices,
            indptr_or_row: row_ptr,
            blocksize: None,
            diagonals: None,
        }
    }

    /// Create from CSC data
    pub fn from_csc(
        shape: (usize, usize),
        col_ptr: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Self {
        Self {
            format: ScipyFormat::Csc,
            shape,
            data: values,
            indices: row_indices,
            indptr_or_row: col_ptr,
            blocksize: None,
            diagonals: None,
        }
    }
}

/// SciPy sparse matrix integration utilities
pub struct ScipySparseIntegration;

impl ScipySparseIntegration {
    /// Convert ToRSh sparse tensor to SciPy sparse data
    pub fn to_scipy_data(sparse: &dyn SparseTensor) -> TorshResult<ScipySparseData> {
        let shape = sparse.shape();
        let (rows, cols) = (shape.dims()[0], shape.dims()[1]);

        match sparse.format() {
            SparseFormat::Coo => {
                let coo = sparse.to_coo()?;
                let triplets = coo.triplets();

                let mut row_indices = Vec::new();
                let mut col_indices = Vec::new();
                let mut values = Vec::new();

                for (row, col, val) in triplets {
                    row_indices.push(row);
                    col_indices.push(col);
                    values.push(val as f64);
                }

                Ok(ScipySparseData::from_coo(
                    (rows, cols),
                    row_indices,
                    col_indices,
                    values,
                ))
            }
            SparseFormat::Csr => {
                let csr = sparse.to_csr()?;
                let row_ptr = csr.row_ptr().to_vec();
                let col_indices = csr.col_indices().to_vec();
                let values = csr.values().iter().map(|&v| v as f64).collect();

                Ok(ScipySparseData::from_csr(
                    (rows, cols),
                    row_ptr,
                    col_indices,
                    values,
                ))
            }
            SparseFormat::Csc => {
                let csc = sparse.to_csc()?;
                let col_ptr = csc.col_ptr().to_vec();
                let row_indices = csc.row_indices().to_vec();
                let values = csc.values().iter().map(|&v| v as f64).collect();

                Ok(ScipySparseData::from_csc(
                    (rows, cols),
                    col_ptr,
                    row_indices,
                    values,
                ))
            }
            _ => {
                // Convert other formats to COO first
                let coo = sparse.to_coo()?;
                Self::to_scipy_data(&coo)
            }
        }
    }

    /// Convert SciPy sparse data to ToRSh sparse tensor
    pub fn from_scipy_data(
        data: &ScipySparseData,
    ) -> TorshResult<Box<dyn SparseTensor + Send + Sync>> {
        let shape = Shape::new(vec![data.shape.0, data.shape.1]);

        match data.format {
            ScipyFormat::Coo => {
                let mut rows = Vec::new();
                let mut cols = Vec::new();
                let mut values = Vec::new();

                for i in 0..data.data.len() {
                    rows.push(data.indptr_or_row[i]);
                    cols.push(data.indices[i]);
                    values.push(data.data[i] as f32);
                }

                let coo = CooTensor::new(rows, cols, values, shape)?;
                Ok(Box::new(coo))
            }
            ScipyFormat::Csr => {
                let row_ptr = &data.indptr_or_row;
                let col_indices = &data.indices;
                let values: Vec<f32> = data.data.iter().map(|&v| v as f32).collect();

                let csr =
                    CsrTensor::from_raw_parts(row_ptr.clone(), col_indices.clone(), values, shape)?;

                Ok(Box::new(csr))
            }
            ScipyFormat::Csc => {
                let col_ptr = &data.indptr_or_row;
                let row_indices = &data.indices;
                let values: Vec<f32> = data.data.iter().map(|&v| v as f32).collect();

                let csc =
                    CscTensor::from_raw_parts(col_ptr.clone(), row_indices.clone(), values, shape)?;

                Ok(Box::new(csc))
            }
            _ => {
                // Convert to COO first, then to target format
                let coo_data = ScipySparseData {
                    format: ScipyFormat::Coo,
                    ..data.clone()
                };
                let coo = Self::from_scipy_data(&coo_data)?;
                convert_sparse_format(coo.as_ref(), data.format.into())
            }
        }
    }

    /// Serialize sparse tensor to dictionary format compatible with SciPy
    pub fn to_dict(sparse: &dyn SparseTensor) -> TorshResult<HashMap<String, Vec<f64>>> {
        let scipy_data = Self::to_scipy_data(sparse)?;

        let mut dict = HashMap::new();
        dict.insert("data".to_string(), scipy_data.data);
        dict.insert(
            "indices".to_string(),
            scipy_data.indices.iter().map(|&x| x as f64).collect(),
        );
        dict.insert(
            "indptr".to_string(),
            scipy_data.indptr_or_row.iter().map(|&x| x as f64).collect(),
        );
        dict.insert(
            "shape".to_string(),
            vec![scipy_data.shape.0 as f64, scipy_data.shape.1 as f64],
        );

        Ok(dict)
    }

    /// Generate Python code to create equivalent SciPy sparse matrix
    pub fn to_python_code(sparse: &dyn SparseTensor, var_name: &str) -> TorshResult<String> {
        let scipy_data = Self::to_scipy_data(sparse)?;
        let format_name = match scipy_data.format {
            ScipyFormat::Coo => "coo_matrix",
            ScipyFormat::Csr => "csr_matrix",
            ScipyFormat::Csc => "csc_matrix",
            ScipyFormat::Bsr => "bsr_matrix",
            ScipyFormat::Dia => "dia_matrix",
        };

        let mut code = String::new();
        code.push_str("import numpy as np\n");
        code.push_str("from scipy.sparse import ");
        code.push_str(format_name);
        code.push_str("\n\n");

        match scipy_data.format {
            ScipyFormat::Coo => {
                code.push_str("# COO format data\n");
                code.push_str(&format!("row = np.array({:?})\n", scipy_data.indptr_or_row));
                code.push_str(&format!("col = np.array({:?})\n", scipy_data.indices));
                code.push_str(&format!("data = np.array({:?})\n", scipy_data.data));
                code.push_str(&format!("shape = {:?}\n", scipy_data.shape));
                code.push_str(&format!(
                    "{var_name} = {format_name}((data, (row, col)), shape=shape)\n"
                ));
            }
            ScipyFormat::Csr | ScipyFormat::Csc => {
                let ptr_name = "indptr";
                code.push_str(&format!("# {} format data\n", format_name.to_uppercase()));
                code.push_str(&format!("data = np.array({:?})\n", scipy_data.data));
                code.push_str(&format!("indices = np.array({:?})\n", scipy_data.indices));
                code.push_str(&format!(
                    "{} = np.array({:?})\n",
                    ptr_name, scipy_data.indptr_or_row
                ));
                code.push_str(&format!("shape = {:?}\n", scipy_data.shape));
                code.push_str(&format!(
                    "{var_name} = {format_name}((data, indices, {ptr_name}), shape=shape)\n"
                ));
            }
            _ => {
                // Fallback to COO for other formats
                code.push_str(&format!(
                    "# Note: {format_name} format converted to COO for compatibility\n"
                ));
                code.push_str(&format!("row = np.array({:?})\n", scipy_data.indptr_or_row));
                code.push_str(&format!("col = np.array({:?})\n", scipy_data.indices));
                code.push_str(&format!("data = np.array({:?})\n", scipy_data.data));
                code.push_str(&format!("shape = {:?}\n", scipy_data.shape));
                code.push_str(&format!(
                    "{var_name} = coo_matrix((data, (row, col)), shape=shape)\n"
                ));
            }
        }

        Ok(code)
    }
}

/// Python bindings for SciPy sparse integration (when scipy feature is enabled)
#[cfg(feature = "scipy")]
pub mod python_bindings {
    use super::*;

    /// Export sparse tensor to Python SciPy format
    #[pyfunction]
    pub fn torsh_to_scipy(
        py: Python,
        format: &str,
        shape: (usize, usize),
        data: Vec<f64>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
    ) -> PyResult<Py<PyAny>> {
        let scipy = py.import("scipy.sparse")?;

        let data_array = PyArray1::from_vec(py, data);
        let indices_array = PyArray1::from_vec(py, indices);
        let indptr_array = PyArray1::from_vec(py, indptr);

        let args = (
            data_array
                .into_pyobject(py)
                .expect("PyArray conversion should succeed"),
            indices_array
                .into_pyobject(py)
                .expect("PyArray conversion should succeed"),
            indptr_array
                .into_pyobject(py)
                .expect("PyArray conversion should succeed"),
        );

        let kwargs = PyDict::new(py);
        kwargs.set_item("shape", shape)?;

        let matrix_class = scipy.getattr(format)?;
        let result = matrix_class.call(args, Some(&kwargs))?;

        Ok(result.unbind())
    }

    /// Import sparse tensor from Python SciPy format
    #[pyfunction]
    pub fn scipy_to_torsh(
        _py: Python,
        scipy_matrix: &Bound<PyAny>,
    ) -> PyResult<(String, (usize, usize), Vec<f64>, Vec<usize>, Vec<usize>)> {
        // Get format
        let format_attr = scipy_matrix.getattr("format")?;
        let format: String = format_attr.extract()?;

        // Get shape
        let shape_attr = scipy_matrix.getattr("shape")?;
        let shape: (usize, usize) = shape_attr.extract()?;

        // Convert to COO format for universal handling
        let coo_matrix = scipy_matrix.call_method0("tocoo")?;

        // Extract data arrays
        let data_attr = coo_matrix.getattr("data")?;
        let row_attr = coo_matrix.getattr("row")?;
        let col_attr = coo_matrix.getattr("col")?;

        let data: PyReadonlyArray1<f64> = data_attr.extract()?;
        let row: PyReadonlyArray1<i32> = row_attr.extract()?;
        let col: PyReadonlyArray1<i32> = col_attr.extract()?;

        let data_vec = data.as_slice()?.to_vec();
        let row_vec: Vec<usize> = row.as_slice()?.iter().map(|&x| x as usize).collect();
        let col_vec: Vec<usize> = col.as_slice()?.iter().map(|&x| x as usize).collect();

        Ok((format, shape, data_vec, col_vec, row_vec))
    }
}

/// Macro to simplify SciPy conversion
#[macro_export]
macro_rules! to_scipy {
    ($sparse:expr) => {
        ScipySparseIntegration::to_scipy_data($sparse)
    };
    ($sparse:expr, $format:expr) => {{
        let scipy_data = ScipySparseIntegration::to_scipy_data($sparse)?;
        let converted = convert_sparse_format($sparse, $format)?;
        ScipySparseIntegration::to_scipy_data(converted.as_ref())
    }};
}

/// Macro to simplify creation from SciPy data
#[macro_export]
macro_rules! from_scipy {
    ($data:expr) => {
        ScipySparseIntegration::from_scipy_data($data)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coo::CooTensor;
    use torsh_core::{DType, Shape};

    #[test]
    fn test_scipy_data_conversion() {
        let shape = Shape::new(vec![3, 3]);
        let mut coo = CooTensor::empty(shape.clone(), DType::F32).unwrap();

        // Create a simple sparse matrix
        coo.insert(0, 0, 1.0).unwrap();
        coo.insert(1, 1, 2.0).unwrap();
        coo.insert(2, 2, 3.0).unwrap();
        coo.insert(0, 2, 4.0).unwrap();

        // Convert to SciPy data
        let scipy_data = ScipySparseIntegration::to_scipy_data(&coo).unwrap();

        assert_eq!(scipy_data.format, ScipyFormat::Coo);
        assert_eq!(scipy_data.shape, (3, 3));
        assert_eq!(scipy_data.data.len(), 4);

        // Convert back to ToRSh
        let restored = ScipySparseIntegration::from_scipy_data(&scipy_data).unwrap();
        assert_eq!(restored.nnz(), 4);
        assert_eq!(restored.shape(), &shape);
    }

    #[test]
    fn test_python_code_generation() {
        let shape = Shape::new(vec![2, 2]);
        let mut coo = CooTensor::empty(shape, DType::F32).unwrap();

        coo.insert(0, 0, 1.0).unwrap();
        coo.insert(1, 1, 2.0).unwrap();

        let code = ScipySparseIntegration::to_python_code(&coo, "matrix").unwrap();

        assert!(code.contains("import numpy as np"));
        assert!(code.contains("from scipy.sparse import"));
        assert!(code.contains("matrix ="));
    }

    #[test]
    fn test_dict_conversion() {
        let shape = Shape::new(vec![2, 2]);
        let mut coo = CooTensor::empty(shape, DType::F32).unwrap();

        coo.insert(0, 0, 1.0).unwrap();
        coo.insert(1, 1, 2.0).unwrap();

        let dict = ScipySparseIntegration::to_dict(&coo).unwrap();

        assert!(dict.contains_key("data"));
        assert!(dict.contains_key("indices"));
        assert!(dict.contains_key("indptr"));
        assert!(dict.contains_key("shape"));

        assert_eq!(dict["shape"], vec![2.0, 2.0]);
        assert_eq!(dict["data"].len(), 2);
    }

    #[test]
    fn test_format_conversion() {
        assert_eq!(ScipyFormat::from(SparseFormat::Coo), ScipyFormat::Coo);
        assert_eq!(ScipyFormat::from(SparseFormat::Csr), ScipyFormat::Csr);
        assert_eq!(ScipyFormat::from(SparseFormat::Csc), ScipyFormat::Csc);
        assert_eq!(ScipyFormat::from(SparseFormat::Ell), ScipyFormat::Csr);

        assert_eq!(SparseFormat::from(ScipyFormat::Coo), SparseFormat::Coo);
        assert_eq!(SparseFormat::from(ScipyFormat::Csr), SparseFormat::Csr);
        assert_eq!(SparseFormat::from(ScipyFormat::Csc), SparseFormat::Csc);
    }

    #[test]
    fn test_macro_usage() {
        let shape = Shape::new(vec![2, 2]);
        let mut coo = CooTensor::empty(shape, DType::F32).unwrap();

        coo.insert(0, 0, 1.0).unwrap();
        coo.insert(1, 1, 2.0).unwrap();

        let scipy_data = to_scipy!(&coo).unwrap();
        assert_eq!(scipy_data.data.len(), 2);

        let restored = from_scipy!(&scipy_data).unwrap();
        assert_eq!(restored.nnz(), 2);
    }
}
