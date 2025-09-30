//! Pandas support for ToRSh tensors
//!
//! This module provides comprehensive integration with Pandas, enabling seamless conversion
//! between ToRSh tensors and Pandas DataFrames/Series, as well as access to Pandas'
//! data manipulation and analysis functionality.

use crate::error::FfiError;
use crate::numpy_compatibility::NumpyCompat;
use crate::python::tensor::PyTensor;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyAny, PyDict, PyModule, PyTuple};
use pyo3::Bound;
use std::collections::HashMap;

/// Pandas integration layer providing data manipulation capabilities
#[pyclass(name = "PandasSupport")]
#[derive(Debug)]
pub struct PandasSupport {
    /// NumPy compatibility layer for array operations
    numpy_compat: NumpyCompat,
    /// Cached Pandas module reference
    pandas_module: Option<Py<PyModule>>,
    /// Configuration for Pandas operations
    config: PandasConfig,
    /// Type mappings between ToRSh and Pandas
    type_mappings: HashMap<String, String>,
}

/// Configuration for Pandas integration
#[derive(Debug, Clone)]
pub struct PandasConfig {
    /// Default index type for DataFrames
    pub default_index_type: String,
    /// Handling of missing values
    pub missing_value_strategy: MissingValueStrategy,
    /// Memory optimization settings
    pub optimize_memory: bool,
    /// Maximum rows to display
    pub max_display_rows: usize,
    /// Precision for floating point display
    pub float_precision: usize,
}

impl Default for PandasConfig {
    fn default() -> Self {
        Self {
            default_index_type: "range".to_string(),
            missing_value_strategy: MissingValueStrategy::DropNA,
            optimize_memory: true,
            max_display_rows: 100,
            float_precision: 4,
        }
    }
}

/// Strategy for handling missing values
#[derive(Debug, Clone)]
pub enum MissingValueStrategy {
    /// Drop rows/columns with missing values
    DropNA,
    /// Fill missing values with a constant
    FillValue(f64),
    /// Forward fill missing values
    ForwardFill,
    /// Backward fill missing values
    BackwardFill,
    /// Interpolate missing values
    Interpolate,
}

/// Result of data analysis operations
#[pyclass(name = "DataAnalysisResult")]
#[derive(Debug, Clone)]
pub struct DataAnalysisResult {
    /// Primary result tensor/data
    #[pyo3(get)]
    pub data: PyTensor,
    /// Statistical summary
    #[pyo3(get)]
    pub statistics: HashMap<String, f64>,
    /// Metadata about the analysis
    #[pyo3(get)]
    pub metadata: HashMap<String, String>,
    /// Column information
    #[pyo3(get)]
    pub columns: Vec<String>,
}

/// DataFrame representation compatible with Pandas
#[pyclass(name = "TorshDataFrame")]
#[derive(Debug, Clone)]
pub struct TorshDataFrame {
    /// Underlying tensor data
    #[pyo3(get)]
    pub data: PyTensor,
    /// Column names
    #[pyo3(get)]
    pub columns: Vec<String>,
    /// Index information
    #[pyo3(get)]
    pub index: Vec<String>,
    /// Data types for each column
    #[pyo3(get)]
    pub dtypes: HashMap<String, String>,
    /// Shape information
    #[pyo3(get)]
    pub shape: (usize, usize),
}

/// Series representation compatible with Pandas
#[pyclass(name = "TorshSeries")]
#[derive(Debug, Clone)]
pub struct TorshSeries {
    /// Underlying tensor data
    #[pyo3(get)]
    pub data: PyTensor,
    /// Series name
    #[pyo3(get)]
    pub name: Option<String>,
    /// Index information
    #[pyo3(get)]
    pub index: Vec<String>,
    /// Data type
    #[pyo3(get)]
    pub dtype: String,
    /// Length
    #[pyo3(get)]
    pub length: usize,
}

#[pymethods]
impl PandasSupport {
    /// Create a new Pandas support instance
    #[new]
    pub fn new() -> PyResult<Self> {
        let mut type_mappings = HashMap::new();

        // Set up type mappings between ToRSh and Pandas
        type_mappings.insert("f32".to_string(), "float32".to_string());
        type_mappings.insert("f64".to_string(), "float64".to_string());
        type_mappings.insert("i32".to_string(), "int32".to_string());
        type_mappings.insert("i64".to_string(), "int64".to_string());
        type_mappings.insert("bool".to_string(), "bool".to_string());

        Ok(Self {
            numpy_compat: NumpyCompat::new(),
            pandas_module: None,
            config: PandasConfig::default(),
            type_mappings,
        })
    }

    /// Configure Pandas support settings
    pub fn configure(&mut self, _py: Python, config: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(index_type) = config.get_item("default_index_type")? {
            self.config.default_index_type = index_type.extract()?;
        }
        if let Some(optimize) = config.get_item("optimize_memory")? {
            self.config.optimize_memory = optimize.extract()?;
        }
        if let Some(max_rows) = config.get_item("max_display_rows")? {
            self.config.max_display_rows = max_rows.extract()?;
        }
        if let Some(precision) = config.get_item("float_precision")? {
            self.config.float_precision = precision.extract()?;
        }
        Ok(())
    }

    /// Convert ToRSh tensor to Pandas DataFrame
    pub fn to_dataframe(
        &self,
        py: Python,
        tensor: &PyTensor,
        columns: Option<Vec<String>>,
        index: Option<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        let pandas = self.get_pandas_module(py)?;
        let numpy_array = self
            .numpy_compat
            .to_numpy_array(&tensor.data, &tensor.shape)
            .map_err(|e| FfiError::InvalidConversion { message: e })?;

        // Create DataFrame constructor arguments
        let mut kwargs = PyDict::new(py);
        kwargs.set_item("data", numpy_array)?;

        if let Some(cols) = columns {
            kwargs.set_item("columns", cols)?;
        }

        if let Some(idx) = index {
            kwargs.set_item("index", idx)?;
        }

        let dataframe = pandas.call_method(py, "DataFrame", (), Some(&kwargs))?;
        Ok(dataframe)
    }

    /// Convert Pandas DataFrame to ToRSh tensor
    pub fn from_dataframe(
        &self,
        py: Python,
        dataframe: Bound<'_, PyAny>,
    ) -> PyResult<TorshDataFrame> {
        // Get the underlying numpy array
        let values = dataframe.getattr("values")?;
        // TODO: Implement proper NumPy array conversion
        return Err(FfiError::UnsupportedOperation {
            operation: "DataFrame to Tensor conversion not implemented".to_string(),
        }
        .into());

        /*
        // Extract metadata
        let columns_py = dataframe.getattr("columns")?;
        let columns: Vec<String> = columns_py.call_method("tolist", (), None)?.extract()?;

        let index_py = dataframe.getattr("index")?;
        let index: Vec<String> = index_py.call_method("tolist", (), None)?.extract()?;

        let shape_py = dataframe.getattr("shape")?;
        let shape_tuple: &PyTuple = shape_py.downcast()?;
        let shape = (
            shape_tuple.get_item(0)?.extract()?,
            shape_tuple.get_item(1)?.extract()?,
        );

        // Get data types
        let dtypes_py = dataframe.getattr("dtypes")?;
        let mut dtypes = HashMap::new();
        for (i, col) in columns.iter().enumerate() {
            let dtype_str: String = dtypes_py.get_item(i)?.str()?.extract()?;
            dtypes.insert(col.clone(), dtype_str);
        }

        Ok(TorshDataFrame {
            data: tensor,
            columns,
            index,
            dtypes,
            shape,
        })
        */
    }

    /// Convert ToRSh tensor to Pandas Series
    pub fn to_series(
        &self,
        py: Python,
        tensor: &PyTensor,
        name: Option<String>,
        index: Option<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        let pandas = self.get_pandas_module(py)?;
        let numpy_array = self
            .numpy_compat
            .to_numpy_array(&tensor.data, &tensor.shape)
            .map_err(|e| FfiError::InvalidConversion { message: e })?;

        let mut kwargs = PyDict::new(py);
        kwargs.set_item("data", numpy_array)?;

        if let Some(name_str) = name {
            kwargs.set_item("name", name_str)?;
        }

        if let Some(idx) = index {
            kwargs.set_item("index", idx)?;
        }

        let series = pandas.call_method(py, "Series", (), Some(&kwargs))?;
        Ok(series)
    }

    /// Convert Pandas Series to ToRSh tensor
    pub fn from_series(&self, py: Python, series: Bound<'_, PyAny>) -> PyResult<TorshSeries> {
        // Get the underlying numpy array
        let _values = series.getattr("values")?;
        // TODO: Implement proper NumPy array conversion
        return Err(FfiError::UnsupportedOperation {
            operation: "Series to Tensor conversion not implemented".to_string(),
        }
        .into());

        /*
        // Extract metadata
        let name_py = series.getattr("name")?;
        let name = if name_py.is_none() {
            None
        } else {
            Some(name_py.extract()?)
        };

        let index_py = series.getattr("index")?;
        let index: Vec<String> = index_py.call_method("tolist", (), None)?.extract()?;

        let dtype_py = series.getattr("dtype")?;
        let dtype: String = dtype_py.str()?.extract()?;

        let length: usize = series.call_method("__len__", (), None)?.extract()?;

        Ok(TorshSeries {
            data: tensor,
            name,
            index,
            dtype,
            length,
        })
        */
    }

    /// Perform data grouping operations
    pub fn groupby_analysis(
        &self,
        py: Python,
        dataframe: Bound<'_, PyAny>,
        group_by: Vec<String>,
        aggregation: &str,
    ) -> PyResult<DataAnalysisResult> {
        let grouped = dataframe.call_method1("groupby", (group_by.clone(),))?;

        let result = match aggregation {
            "mean" => grouped.call_method("mean", (), None)?,
            "sum" => grouped.call_method("sum", (), None)?,
            "count" => grouped.call_method("count", (), None)?,
            "std" => grouped.call_method("std", (), None)?,
            "var" => grouped.call_method("var", (), None)?,
            "min" => grouped.call_method("min", (), None)?,
            "max" => grouped.call_method("max", (), None)?,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown aggregation: {}",
                    aggregation
                )))
            }
        };

        // Convert result back to tensor
        let torsh_df = self.from_dataframe(py, result)?;

        // Generate statistics
        let mut statistics = HashMap::new();
        statistics.insert("num_groups".to_string(), group_by.len() as f64);
        statistics.insert("aggregation_type".to_string(), aggregation.len() as f64);

        // Generate metadata
        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "groupby".to_string());
        metadata.insert("aggregation".to_string(), aggregation.to_string());
        metadata.insert("group_columns".to_string(), group_by.join(","));

        Ok(DataAnalysisResult {
            data: torsh_df.data,
            statistics,
            metadata,
            columns: torsh_df.columns,
        })
    }

    /// Perform statistical analysis
    pub fn statistical_analysis(
        &self,
        py: Python,
        dataframe: Bound<'_, PyAny>,
    ) -> PyResult<DataAnalysisResult> {
        let describe_result = dataframe.call_method("describe", (), None)?;
        let corr_result = dataframe.call_method("corr", (), None)?;

        // Convert correlation matrix to tensor
        let corr_df = self.from_dataframe(py, corr_result)?;

        // Extract statistical summaries
        let mut statistics = HashMap::new();
        let describe_df = self.from_dataframe(py, describe_result)?;

        // Get basic stats
        let mean_values = dataframe.call_method("mean", (), None)?;
        let std_values = dataframe.call_method("std", (), None)?;
        let min_values = dataframe.call_method("min", (), None)?;
        let max_values = dataframe.call_method("max", (), None)?;

        statistics.insert(
            "mean_of_means".to_string(),
            mean_values.call_method("mean", (), None)?.extract()?,
        );
        statistics.insert(
            "mean_of_stds".to_string(),
            std_values.call_method("mean", (), None)?.extract()?,
        );
        statistics.insert(
            "global_min".to_string(),
            min_values.call_method("min", (), None)?.extract()?,
        );
        statistics.insert(
            "global_max".to_string(),
            max_values.call_method("max", (), None)?.extract()?,
        );

        let mut metadata = HashMap::new();
        metadata.insert("operation".to_string(), "statistical_analysis".to_string());
        metadata.insert(
            "includes".to_string(),
            "correlation,describe,summary".to_string(),
        );

        Ok(DataAnalysisResult {
            data: corr_df.data,
            statistics,
            metadata,
            columns: corr_df.columns,
        })
    }

    /// Handle missing values according to strategy
    pub fn handle_missing_values(
        &self,
        py: Python,
        dataframe: Bound<'_, PyAny>,
        strategy: &str,
        value: Option<f64>,
    ) -> PyResult<Py<PyAny>> {
        match strategy {
            "dropna" => Ok(dataframe.call_method("dropna", (), None)?.into()),
            "fillna" => {
                let fill_value = value.unwrap_or(0.0);
                Ok(dataframe.call_method1("fillna", (fill_value,))?.into())
            }
            "ffill" => {
                let kwargs = [("method", "ffill")].into_py_dict(py)?;
                Ok(dataframe.call_method("fillna", (), Some(&kwargs))?.into())
            }
            "bfill" => {
                let kwargs = [("method", "bfill")].into_py_dict(py)?;
                Ok(dataframe.call_method("fillna", (), Some(&kwargs))?.into())
            }
            "interpolate" => Ok(dataframe.call_method("interpolate", (), None)?.into()),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown missing value strategy: {}",
                strategy
            ))),
        }
    }

    /// Perform data filtering and selection
    pub fn filter_data(
        &self,
        py: Python,
        dataframe: Bound<'_, PyAny>,
        query: &str,
    ) -> PyResult<Py<PyAny>> {
        Ok(dataframe.call_method1("query", (query,))?.into())
    }

    /// Merge/join DataFrames
    pub fn merge_dataframes(
        &self,
        py: Python,
        left: Bound<'_, PyAny>,
        right: Bound<'_, PyAny>,
        on: Vec<String>,
        how: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let pandas = self.get_pandas_module(py)?;
        let how_str = how.unwrap_or("inner");

        // TODO: Implement proper DataFrame merging
        Err(FfiError::UnsupportedOperation {
            operation: "DataFrame merging not implemented".to_string(),
        }
        .into())
    }

    /// Pivot table operations
    pub fn pivot_table(
        &self,
        py: Python,
        dataframe: Bound<'_, PyAny>,
        values: Vec<String>,
        index: Vec<String>,
        columns: Vec<String>,
        aggfunc: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        // TODO: Implement proper pivot table operations
        Err(FfiError::UnsupportedOperation {
            operation: "Pivot table operations not implemented".to_string(),
        }
        .into())
    }

    /// Time series operations
    pub fn time_series_analysis(
        &self,
        py: Python,
        series: Bound<'_, PyAny>,
        freq: Option<&str>,
        window: Option<usize>,
    ) -> PyResult<DataAnalysisResult> {
        // TODO: Implement proper time series analysis
        Err(FfiError::UnsupportedOperation {
            operation: "Time series analysis not implemented".to_string(),
        }
        .into())
    }

    /// Get Pandas version information
    pub fn get_pandas_version(&self, py: Python) -> PyResult<String> {
        let pandas = self.get_pandas_module(py)?;
        let version: String = pandas.getattr(py, "__version__")?.extract(py)?;
        Ok(version)
    }

    /// Export DataFrame to various formats
    pub fn export_dataframe(
        &self,
        py: Python,
        dataframe: Bound<'_, PyAny>,
        format: &str,
        path: &str,
    ) -> PyResult<()> {
        match format.to_lowercase().as_str() {
            "csv" => {
                dataframe.call_method1("to_csv", (path,))?;
            }
            "json" => {
                dataframe.call_method1("to_json", (path,))?;
            }
            "excel" => {
                dataframe.call_method1("to_excel", (path,))?;
            }
            "parquet" => {
                dataframe.call_method1("to_parquet", (path,))?;
            }
            "hdf5" | "hdf" => {
                let kwargs = [("mode", "w")].into_py_dict(py)?;
                dataframe.call_method("to_hdf", (path, "data"), Some(&kwargs))?;
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unsupported export format: {}",
                    format
                )))
            }
        }
        Ok(())
    }

    /// Import DataFrame from various formats
    pub fn import_dataframe(&self, py: Python, format: &str, path: &str) -> PyResult<Py<PyAny>> {
        let pandas = self.get_pandas_module(py)?;

        match format.to_lowercase().as_str() {
            "csv" => pandas.call_method1(py, "read_csv", (path,)),
            "json" => pandas.call_method1(py, "read_json", (path,)),
            "excel" => pandas.call_method1(py, "read_excel", (path,)),
            "parquet" => pandas.call_method1(py, "read_parquet", (path,)),
            "hdf5" | "hdf" => pandas.call_method(py, "read_hdf", (path, "data"), None),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported import format: {}",
                format
            ))),
        }
    }
}

impl PandasSupport {
    /// Get or cache Pandas module
    fn get_pandas_module(&self, py: Python) -> PyResult<Py<PyModule>> {
        if let Some(module) = &self.pandas_module {
            return Ok(module.clone_ref(py));
        }

        let module = py.import("pandas")?;
        Ok(module.into())
    }
}

/// Create Pandas support utilities
pub fn create_pandas_utilities(py: Python) -> PyResult<Bound<PyDict>> {
    let utils = PyDict::new(py);

    // Add utility functions
    utils.set_item("create_support", py.get_type::<PandasSupport>())?;
    utils.set_item("TorshDataFrame", py.get_type::<TorshDataFrame>())?;
    utils.set_item("TorshSeries", py.get_type::<TorshSeries>())?;
    utils.set_item("DataAnalysisResult", py.get_type::<DataAnalysisResult>())?;

    Ok(utils)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pandas_config() {
        let config = PandasConfig::default();
        assert_eq!(config.default_index_type, "range");
        assert_eq!(config.max_display_rows, 100);
        assert!(config.optimize_memory);
    }

    #[test]
    fn test_missing_value_strategy() {
        let strategy = MissingValueStrategy::FillValue(42.0);
        match strategy {
            MissingValueStrategy::FillValue(val) => assert_eq!(val, 42.0),
            _ => panic!("Expected FillValue strategy"),
        }
    }
}
