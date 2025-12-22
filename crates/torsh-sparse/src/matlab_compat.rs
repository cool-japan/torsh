//! MATLAB sparse matrix compatibility
//!
//! This module provides conversion between ToRSh sparse tensors and MATLAB sparse matrices,
//! enabling seamless integration with MATLAB scientific computing environment.

#[cfg(feature = "matlab")]
use matfile::MatFile;
#[cfg(feature = "matlab")]
use serde::{Deserialize, Serialize};

use crate::*;
use std::collections::HashMap;
use std::path::Path;
use torsh_core::{DType, Result as TorshResult, Shape};

/// MATLAB sparse matrix representation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "matlab", derive(Serialize, Deserialize))]
pub struct MatlabSparseMatrix {
    /// Matrix data values
    pub data: Vec<f64>,
    /// Row indices (1-based, MATLAB convention)
    pub row_indices: Vec<usize>,
    /// Column indices (1-based, MATLAB convention)
    pub col_indices: Vec<usize>,
    /// Matrix dimensions [rows, cols]
    pub size: [usize; 2],
    /// Matrix name in MATLAB workspace
    pub name: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl MatlabSparseMatrix {
    /// Create new MATLAB sparse matrix
    pub fn new(name: String, rows: usize, cols: usize) -> Self {
        Self {
            data: Vec::new(),
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            size: [rows, cols],
            name,
            metadata: HashMap::new(),
        }
    }

    /// Add a non-zero element (using 1-based indexing)
    pub fn add_element(&mut self, row: usize, col: usize, value: f64) {
        self.row_indices.push(row + 1); // Convert to 1-based
        self.col_indices.push(col + 1); // Convert to 1-based
        self.data.push(value);
    }

    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Add metadata entry
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Generate MATLAB code to reconstruct the matrix
    pub fn to_matlab_code(&self) -> String {
        let mut code = String::new();

        code.push_str(&format!("% Sparse matrix: {}\n", self.name));
        code.push_str(&format!(
            "% Size: {}x{}, NNZ: {}\n",
            self.size[0],
            self.size[1],
            self.nnz()
        ));

        if !self.metadata.is_empty() {
            code.push_str("% Metadata:\n");
            for (key, value) in &self.metadata {
                code.push_str(&format!("% {key}: {value}\n"));
            }
        }

        code.push('\n');

        if self.data.is_empty() {
            code.push_str(&format!(
                "{} = sparse({}, {});\n",
                self.name, self.size[0], self.size[1]
            ));
        } else {
            code.push_str(&format!(
                "I = [{}];\n",
                self.row_indices
                    .iter()
                    .map(|&x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));

            code.push_str(&format!(
                "J = [{}];\n",
                self.col_indices
                    .iter()
                    .map(|&x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));

            code.push_str(&format!(
                "V = [{}];\n",
                self.data
                    .iter()
                    .map(|&x| format!("{x:.16e}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));

            code.push_str(&format!(
                "{} = sparse(I, J, V, {}, {});\n",
                self.name, self.size[0], self.size[1]
            ));
        }

        code
    }

    /// Export to MATLAB script file
    pub fn to_matlab_file(&self, filepath: &Path) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let code = self.to_matlab_code();
        let mut file = File::create(filepath)?;
        file.write_all(code.as_bytes())?;
        Ok(())
    }
}

/// MATLAB compatibility utilities
pub struct MatlabSparseCompat;

impl MatlabSparseCompat {
    /// Convert ToRSh sparse tensor to MATLAB sparse matrix
    pub fn to_matlab(sparse: &dyn SparseTensor, name: String) -> TorshResult<MatlabSparseMatrix> {
        let shape = sparse.shape();
        let (rows, cols) = (shape.dims()[0], shape.dims()[1]);

        let mut matlab_matrix = MatlabSparseMatrix::new(name, rows, cols);

        // Add metadata
        matlab_matrix.add_metadata("format".to_string(), format!("{:?}", sparse.format()));
        matlab_matrix.add_metadata("dtype".to_string(), format!("{:?}", sparse.dtype()));
        matlab_matrix.add_metadata("nnz".to_string(), sparse.nnz().to_string());
        matlab_matrix.add_metadata("sparsity".to_string(), format!("{:.4}", sparse.sparsity()));

        // Convert to COO format for universal handling
        let coo = sparse.to_coo()?;
        let triplets = coo.triplets();

        for (row, col, val) in triplets {
            matlab_matrix.add_element(row, col, val as f64);
        }

        Ok(matlab_matrix)
    }

    /// Convert MATLAB sparse matrix to ToRSh sparse tensor
    pub fn from_matlab(matlab: &MatlabSparseMatrix) -> TorshResult<Box<dyn SparseTensor>> {
        let shape = Shape::new(vec![matlab.size[0], matlab.size[1]]);
        let mut coo = CooTensor::empty(shape, DType::F32)?;

        for i in 0..matlab.data.len() {
            let row = matlab.row_indices[i] - 1; // Convert from 1-based to 0-based
            let col = matlab.col_indices[i] - 1; // Convert from 1-based to 0-based
            let val = matlab.data[i] as f32;
            coo.insert(row, col, val)?;
        }

        Ok(Box::new(coo))
    }

    /// Export sparse tensor to MATLAB .mat file (via script generation)
    ///
    /// Note: Direct .mat file export is not currently implemented due to matfile crate API limitations.
    /// This function generates a MATLAB script that can be run to create the sparse matrix.
    /// For a complete working solution, use `export_to_matlab_script()` which creates a .m file
    /// with the matrix data embedded as code.
    #[cfg(feature = "matlab")]
    pub fn export_to_mat_file(
        sparse: &dyn SparseTensor,
        filepath: &Path,
        variable_name: &str,
    ) -> TorshResult<()> {
        // Alternative implementation: Generate MATLAB script instead
        let matlab_matrix = Self::to_matlab(sparse, variable_name.to_string())?;
        let matlab_code = matlab_matrix.to_matlab_code();

        // Change extension to .m for MATLAB script
        let script_path = filepath.with_extension("m");

        std::fs::write(&script_path, matlab_code)
            .map_err(|e| TorshError::IoError(format!("Failed to write MATLAB script: {}", e)))?;

        eprintln!(
            "Note: Exported as MATLAB script (.m) instead of .mat file.\n\
             To use in MATLAB: run the script '{}'",
            script_path.display()
        );

        Ok(())
    }

    /// Import sparse tensor from MATLAB .mat file
    ///
    /// Note: Direct .mat file import is not currently implemented due to matfile crate API limitations.
    /// As a workaround:
    /// 1. Export your MATLAB matrix to text format (save as CSV or use MATLAB's `save -ascii`)
    /// 2. Use the Matrix Market format for interchange (`mmwrite` in MATLAB)
    /// 3. Use HDF5 format which is supported by both MATLAB and torsh-sparse
    #[cfg(feature = "matlab")]
    pub fn import_from_mat_file(
        filepath: &Path,
        variable_name: &str,
    ) -> TorshResult<Box<dyn SparseTensor>> {
        // Attempt to use matfile crate for reading (basic implementation)
        use std::io::Cursor;

        let file_data = std::fs::read(filepath)
            .map_err(|e| TorshError::IoError(format!("Failed to read .mat file: {}", e)))?;

        // Create a cursor for the Read trait
        let cursor = Cursor::new(file_data);

        // Try to parse the .mat file
        let mat_file = MatFile::parse(cursor).map_err(|e| {
            TorshError::InvalidArgument(format!(
                "Failed to parse .mat file: {}. Consider using Matrix Market or HDF5 format instead.",
                e
            ))
        })?;

        // Try to find the variable in the .mat file
        let _array = mat_file.find_by_name(variable_name).ok_or_else(|| {
            TorshError::InvalidArgument(format!(
                "Variable '{}' not found in .mat file. Available variables: {:?}",
                variable_name,
                mat_file
                    .arrays()
                    .iter()
                    .map(|a| a.name())
                    .collect::<Vec<_>>()
            ))
        })?;

        // Check if it's a sparse array
        // Note: This is a basic implementation and may need adjustment based on
        // the actual matfile crate API and MATLAB sparse matrix encoding

        // For now, return an error with guidance
        Err(TorshError::InvalidArgument(format!(
            ".mat file import is partially implemented. The file was parsed successfully, \
                 but extracting sparse matrix data requires additional matfile crate features. \
                 Please use Matrix Market (.mtx) or HDF5 (.h5) format for reliable import/export. \
                 Variable '{}' was found in the file.",
            variable_name
        )))
    }

    /// Create MATLAB script to load sparse matrix from components
    pub fn create_load_script(variable_name: &str) -> String {
        format!(
            r#"function {variable_name} = load_sparse_matrix(data_file)
    % Load sparse matrix components from MAT file
    % Usage: matrix = load_sparse_matrix('matrix_data.mat')
    
    load(data_file);
    
    % Reconstruct sparse matrix
    {variable_name} = sparse({variable_name}_rows, {variable_name}_cols, {variable_name}_data, {variable_name}_size(1), {variable_name}_size(2));
    
    fprintf('Loaded sparse matrix: %dx%d with %d non-zeros\n', ...
        size({variable_name}, 1), size({variable_name}, 2), nnz({variable_name}));
end
"#
        )
    }

    /// Generate MATLAB analysis script for sparse matrix
    pub fn create_analysis_script(variable_name: &str) -> String {
        format!(
            r#"function analyze_sparse_matrix({variable_name})
    % Analyze sparse matrix properties
    % Usage: analyze_sparse_matrix(matrix)
    
    fprintf('=== Sparse Matrix Analysis ===\n');
    fprintf('Size: %dx%d\n', size({variable_name}, 1), size({variable_name}, 2));
    fprintf('Non-zeros: %d\n', nnz({variable_name}));
    fprintf('Density: %.4f%%\n', 100 * nnz({variable_name}) / numel({variable_name}));
    fprintf('Memory usage: %.2f KB\n', whos('{variable_name}').bytes / 1024);
    
    % Pattern analysis
    figure;
    subplot(2, 2, 1);
    spy({variable_name});
    title('Sparsity Pattern');
    
    subplot(2, 2, 2);
    histogram(nonzeros({variable_name}), 'EdgeColor', 'none');
    title('Value Distribution');
    xlabel('Value');
    ylabel('Frequency');
    
    subplot(2, 2, 3);
    [rows, cols] = find({variable_name});
    histogram(rows, 'EdgeColor', 'none');
    title('Row Distribution');
    xlabel('Row Index');
    ylabel('Frequency');
    
    subplot(2, 2, 4);
    histogram(cols, 'EdgeColor', 'none');
    title('Column Distribution');
    xlabel('Column Index');
    ylabel('Frequency');
    
    sgtitle(sprintf('Analysis of %s', inputname(1)));
end
"#
        )
    }

    /// Export complete MATLAB package with matrix and analysis tools
    pub fn export_matlab_package(
        sparse: &dyn SparseTensor,
        output_dir: &Path,
        variable_name: &str,
    ) -> TorshResult<()> {
        use std::fs;

        // Create output directory
        fs::create_dir_all(output_dir).map_err(|e| {
            torsh_core::TorshError::Other(format!("Failed to create directory: {e}"))
        })?;

        // Export matrix as MATLAB script
        let matlab_matrix = Self::to_matlab(sparse, variable_name.to_string())?;
        let script_path = output_dir.join(format!("{variable_name}.m"));
        matlab_matrix
            .to_matlab_file(&script_path)
            .map_err(|e| torsh_core::TorshError::Other(format!("Failed to write script: {e}")))?;

        // Export .mat file if feature is enabled
        #[cfg(feature = "matlab")]
        {
            let mat_path = output_dir.join(format!("{}.mat", variable_name));
            Self::export_to_mat_file(sparse, &mat_path, variable_name)?;
        }

        // Create loader script
        let loader_script = Self::create_load_script(variable_name);
        let loader_path = output_dir.join(format!("load_{variable_name}.m"));
        fs::write(&loader_path, loader_script)
            .map_err(|e| torsh_core::TorshError::Other(format!("Failed to write loader: {e}")))?;

        // Create analysis script
        let analysis_script = Self::create_analysis_script(variable_name);
        let analysis_path = output_dir.join(format!("analyze_{variable_name}.m"));
        fs::write(&analysis_path, analysis_script)
            .map_err(|e| torsh_core::TorshError::Other(format!("Failed to write analysis: {e}")))?;

        // Create README
        let readme_content = format!(
            r#"# MATLAB Sparse Matrix Package: {}

This package contains a sparse matrix exported from ToRSh in MATLAB-compatible format.

## Files:
- `{}.m` - MATLAB script to create the sparse matrix
- `{}.mat` - Binary MAT file with matrix data (if available)
- `load_{}.m` - Function to load matrix from MAT file
- `analyze_{}.m` - Function to analyze matrix properties

## Usage:

### Loading the matrix:
```matlab
% Method 1: Run the script
{}

% Method 2: Load from MAT file (if available)
{} = load_{}('data.mat');
```

### Analyzing the matrix:
```matlab
analyze_{}({});
```

## Matrix Properties:
- Size: {}x{}
- Non-zeros: {}
- Density: {:.4}%
- Format: {:?}
- Data type: {:?}
"#,
            variable_name,
            variable_name,
            variable_name,
            variable_name,
            variable_name,
            variable_name,
            variable_name,
            variable_name,
            variable_name,
            variable_name,
            sparse.shape().dims()[0],
            sparse.shape().dims()[1],
            sparse.nnz(),
            (1.0 - sparse.sparsity()) * 100.0,
            sparse.format(),
            sparse.dtype()
        );

        let readme_path = output_dir.join("README.md");
        fs::write(&readme_path, readme_content)
            .map_err(|e| torsh_core::TorshError::Other(format!("Failed to write README: {e}")))?;

        Ok(())
    }
}

/// Convenience function to export sparse tensor to MATLAB script
pub fn export_to_matlab_script(
    sparse: &dyn SparseTensor,
    filepath: &Path,
    variable_name: &str,
) -> TorshResult<()> {
    let matlab_matrix = MatlabSparseCompat::to_matlab(sparse, variable_name.to_string())?;
    matlab_matrix.to_matlab_file(filepath).map_err(|e| {
        torsh_core::TorshError::Other(format!("Failed to write MATLAB script: {e}"))
    })?;
    Ok(())
}

/// Convenience function to create MATLAB sparse matrix from triplets
pub fn matlab_sparse_from_triplets(
    rows: Vec<usize>,
    cols: Vec<usize>,
    values: Vec<f64>,
    shape: (usize, usize),
    name: String,
) -> MatlabSparseMatrix {
    let mut matlab_matrix = MatlabSparseMatrix::new(name, shape.0, shape.1);

    for ((row, col), val) in rows.iter().zip(cols.iter()).zip(values.iter()) {
        matlab_matrix.add_element(*row, *col, *val);
    }

    matlab_matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coo::CooTensor;
    use torsh_core::{DType, Shape};

    #[test]
    fn test_matlab_matrix_creation() {
        let mut matlab_matrix = MatlabSparseMatrix::new("test_matrix".to_string(), 3, 3);

        matlab_matrix.add_element(0, 0, 1.0);
        matlab_matrix.add_element(1, 1, 2.0);
        matlab_matrix.add_element(2, 2, 3.0);

        assert_eq!(matlab_matrix.nnz(), 3);
        assert_eq!(matlab_matrix.size, [3, 3]);
        assert_eq!(matlab_matrix.row_indices, vec![1, 2, 3]); // 1-based
        assert_eq!(matlab_matrix.col_indices, vec![1, 2, 3]); // 1-based
    }

    #[test]
    fn test_torsh_to_matlab_conversion() {
        let shape = Shape::new(vec![3, 3]);
        let mut coo = CooTensor::empty(shape, DType::F32).unwrap();

        coo.insert(0, 0, 1.0).unwrap();
        coo.insert(1, 1, 2.0).unwrap();
        coo.insert(2, 2, 3.0).unwrap();

        let matlab_matrix = MatlabSparseCompat::to_matlab(&coo, "test".to_string()).unwrap();

        assert_eq!(matlab_matrix.nnz(), 3);
        assert_eq!(matlab_matrix.size, [3, 3]);
        assert!(matlab_matrix.metadata.contains_key("format"));
        assert!(matlab_matrix.metadata.contains_key("nnz"));
    }

    #[test]
    fn test_matlab_to_torsh_conversion() {
        let mut matlab_matrix = MatlabSparseMatrix::new("test".to_string(), 2, 2);
        matlab_matrix.add_element(0, 0, 1.0);
        matlab_matrix.add_element(1, 1, 2.0);

        let sparse = MatlabSparseCompat::from_matlab(&matlab_matrix).unwrap();

        assert_eq!(sparse.nnz(), 2);
        assert_eq!(sparse.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_matlab_code_generation() {
        let mut matlab_matrix = MatlabSparseMatrix::new("A".to_string(), 2, 2);
        matlab_matrix.add_element(0, 0, 1.0);
        matlab_matrix.add_element(1, 1, 2.0);
        matlab_matrix.add_metadata("test".to_string(), "value".to_string());

        let code = matlab_matrix.to_matlab_code();

        assert!(code.contains("A = sparse"));
        assert!(code.contains("% Metadata:"));
        assert!(code.contains("test: value"));
        assert!(code.contains("I = [1, 2]"));
        assert!(code.contains("J = [1, 2]"));
    }

    #[test]
    fn test_triplets_to_matlab() {
        let rows = vec![0, 1, 2];
        let cols = vec![0, 1, 2];
        let values = vec![1.0, 2.0, 3.0];

        let matlab_matrix =
            matlab_sparse_from_triplets(rows, cols, values, (3, 3), "identity".to_string());

        assert_eq!(matlab_matrix.nnz(), 3);
        assert_eq!(matlab_matrix.name, "identity");
        assert_eq!(matlab_matrix.row_indices, vec![1, 2, 3]); // 1-based
    }

    #[test]
    fn test_analysis_script_generation() {
        let script = MatlabSparseCompat::create_analysis_script("matrix");

        assert!(script.contains("function analyze_sparse_matrix"));
        assert!(script.contains("spy(matrix)"));
        assert!(script.contains("histogram"));
        assert!(script.contains("Sparsity Pattern"));
    }

    #[test]
    fn test_load_script_generation() {
        let script = MatlabSparseCompat::create_load_script("data");

        assert!(script.contains("function data = load_sparse_matrix"));
        assert!(script.contains("data = sparse"));
        assert!(script.contains("data_rows, data_cols, data_data"));
    }

    #[test]
    fn test_matlab_script_export() {
        let shape = Shape::new(vec![2, 2]);
        let mut coo = CooTensor::empty(shape, DType::F32).unwrap();
        coo.insert(0, 0, 1.0).unwrap();
        coo.insert(1, 1, 2.0).unwrap();

        let temp_dir = std::env::temp_dir();
        let script_path = temp_dir.join("test_matrix.m");

        export_to_matlab_script(&coo, &script_path, "test_matrix").unwrap();

        // Verify file was created
        assert!(script_path.exists());

        // Clean up
        let _ = std::fs::remove_file(script_path);
    }
}
