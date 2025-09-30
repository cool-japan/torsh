//! Matrix Market I/O support for sparse matrices
//!
//! This module provides functionality to read and write sparse matrices
//! in the Matrix Market exchange format, which is widely used in scientific computing.

use crate::{CooTensor, SparseTensor, TorshResult};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::str::FromStr;
use torsh_core::{Shape, TorshError};

/// Matrix Market coordinate format specification
#[derive(Debug, Clone)]
pub struct MatrixMarketHeader {
    /// Matrix Market banner line
    pub banner: String,
    /// Matrix type (matrix, vector)
    pub object: MatrixMarketObject,
    /// Storage format (coordinate, array)
    pub format: MatrixMarketFormat,
    /// Field type (real, complex, integer, pattern)
    pub field: MatrixMarketField,
    /// Symmetry type (general, symmetric, skew-symmetric, hermitian)
    pub symmetry: MatrixMarketSymmetry,
    /// Matrix dimensions and number of entries
    pub size_info: MatrixMarketSize,
    /// Optional comments
    pub comments: Vec<String>,
}

/// Matrix Market object type
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixMarketObject {
    Matrix,
    Vector,
}

/// Matrix Market format type
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixMarketFormat {
    Coordinate,
    Array,
}

/// Matrix Market field type
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixMarketField {
    Real,
    Complex,
    Integer,
    Pattern,
}

/// Matrix Market symmetry type
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixMarketSymmetry {
    General,
    Symmetric,
    SkewSymmetric,
    Hermitian,
}

/// Matrix size information
#[derive(Debug, Clone)]
pub struct MatrixMarketSize {
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Number of non-zero entries
    pub nnz: usize,
}

/// Matrix Market reader/writer
pub struct MatrixMarketIO;

impl MatrixMarketIO {
    /// Read a sparse matrix from Matrix Market format
    pub fn read_from_file(file_path: &str) -> TorshResult<(CooTensor, MatrixMarketHeader)> {
        let file = File::open(file_path)
            .map_err(|e| TorshError::IoError(format!("Failed to open file {file_path}: {e}")))?;

        let reader = BufReader::new(file);
        Self::read_from_reader(reader)
    }

    /// Read from any reader implementing BufRead
    pub fn read_from_reader<R: BufRead>(
        mut reader: R,
    ) -> TorshResult<(CooTensor, MatrixMarketHeader)> {
        let header = Self::parse_header(&mut reader)?;
        let tensor = Self::parse_data(&mut reader, &header)?;
        Ok((tensor, header))
    }

    /// Write a sparse matrix to Matrix Market format
    pub fn write_to_file(
        tensor: &dyn SparseTensor,
        file_path: &str,
        field: MatrixMarketField,
        symmetry: MatrixMarketSymmetry,
    ) -> TorshResult<()> {
        let mut file = File::create(file_path)
            .map_err(|e| TorshError::IoError(format!("Failed to create file {file_path}: {e}")))?;

        Self::write_to_writer(tensor, &mut file, field, symmetry)
    }

    /// Write to any writer implementing Write
    pub fn write_to_writer<W: Write>(
        tensor: &dyn SparseTensor,
        writer: &mut W,
        field: MatrixMarketField,
        symmetry: MatrixMarketSymmetry,
    ) -> TorshResult<()> {
        // Convert to COO format for writing
        let coo = tensor.to_coo()?;
        let shape = coo.shape();
        let triplets = coo.triplets();

        // Write header
        Self::write_header(writer, &coo, &field, &symmetry)?;

        // Write size line
        writeln!(
            writer,
            "{} {} {}",
            shape.dims()[0],
            shape.dims()[1],
            triplets.len()
        )
        .map_err(|e| TorshError::IoError(format!("Failed to write size line: {e}")))?;

        // Write data
        Self::write_data(writer, &triplets, &field, &symmetry)?;

        Ok(())
    }

    /// Parse Matrix Market header
    fn parse_header<R: BufRead>(reader: &mut R) -> TorshResult<MatrixMarketHeader> {
        let mut line = String::new();

        // Read banner line
        reader
            .read_line(&mut line)
            .map_err(|e| TorshError::IoError(format!("Failed to read banner: {e}")))?;

        let banner = line.trim().to_string();

        // Parse banner components
        let parts: Vec<&str> = banner.split_whitespace().collect();
        if parts.len() != 5 || parts[0] != "%%MatrixMarket" {
            return Err(TorshError::InvalidArgument(
                "Invalid Matrix Market banner".to_string(),
            ));
        }

        let object = Self::parse_object(parts[1])?;
        let format = Self::parse_format(parts[2])?;
        let field = Self::parse_field(parts[3])?;
        let symmetry = Self::parse_symmetry(parts[4])?;

        // Read comments
        let mut comments = Vec::new();
        loop {
            line.clear();
            reader
                .read_line(&mut line)
                .map_err(|e| TorshError::IoError(format!("Failed to read line: {e}")))?;

            let trimmed = line.trim();
            if let Some(stripped) = trimmed.strip_prefix('%') {
                comments.push(stripped.trim().to_string());
            } else {
                break;
            }
        }

        // Parse size line
        let size_parts: Vec<&str> = line.split_whitespace().collect();
        if size_parts.len() != 3 {
            return Err(TorshError::InvalidArgument(
                "Invalid size line format".to_string(),
            ));
        }

        let rows = usize::from_str(size_parts[0])
            .map_err(|_| TorshError::InvalidArgument("Invalid row count".to_string()))?;
        let cols = usize::from_str(size_parts[1])
            .map_err(|_| TorshError::InvalidArgument("Invalid column count".to_string()))?;
        let nnz = usize::from_str(size_parts[2])
            .map_err(|_| TorshError::InvalidArgument("Invalid nnz count".to_string()))?;

        let size_info = MatrixMarketSize { rows, cols, nnz };

        Ok(MatrixMarketHeader {
            banner,
            object,
            format,
            field,
            symmetry,
            size_info,
            comments,
        })
    }

    /// Parse data section
    fn parse_data<R: BufRead>(
        reader: &mut R,
        header: &MatrixMarketHeader,
    ) -> TorshResult<CooTensor> {
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        let mut line = String::new();
        for _ in 0..header.size_info.nnz {
            line.clear();
            reader
                .read_line(&mut line)
                .map_err(|e| TorshError::IoError(format!("Failed to read data line: {e}")))?;

            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() < 2 {
                return Err(TorshError::InvalidArgument(
                    "Invalid data line format".to_string(),
                ));
            }

            // Matrix Market uses 1-based indexing
            let row = usize::from_str(parts[0])
                .map_err(|_| TorshError::InvalidArgument("Invalid row index".to_string()))?
                - 1;
            let col = usize::from_str(parts[1])
                .map_err(|_| TorshError::InvalidArgument("Invalid column index".to_string()))?
                - 1;

            // Parse value based on field type
            let value = match header.field {
                MatrixMarketField::Real => {
                    if parts.len() < 3 {
                        return Err(TorshError::InvalidArgument(
                            "Missing value for real field".to_string(),
                        ));
                    }
                    f32::from_str(parts[2]).map_err(|_| {
                        TorshError::InvalidArgument("Invalid real value".to_string())
                    })?
                }
                MatrixMarketField::Integer => {
                    if parts.len() < 3 {
                        return Err(TorshError::InvalidArgument(
                            "Missing value for integer field".to_string(),
                        ));
                    }
                    i32::from_str(parts[2]).map_err(|_| {
                        TorshError::InvalidArgument("Invalid integer value".to_string())
                    })? as f32
                }
                MatrixMarketField::Pattern => 1.0, // Pattern matrices have implicit value of 1
                MatrixMarketField::Complex => {
                    return Err(TorshError::UnsupportedOperation {
                        op: "Matrix Market Complex field".to_string(),
                        dtype: "Complex".to_string(),
                    });
                }
            };

            row_indices.push(row);
            col_indices.push(col);
            values.push(value);

            // Handle symmetry by adding symmetric entries
            if header.symmetry != MatrixMarketSymmetry::General && row != col {
                match header.symmetry {
                    MatrixMarketSymmetry::Symmetric => {
                        row_indices.push(col);
                        col_indices.push(row);
                        values.push(value);
                    }
                    MatrixMarketSymmetry::SkewSymmetric => {
                        row_indices.push(col);
                        col_indices.push(row);
                        values.push(-value);
                    }
                    MatrixMarketSymmetry::Hermitian => {
                        // For real matrices, Hermitian is the same as symmetric
                        row_indices.push(col);
                        col_indices.push(row);
                        values.push(value);
                    }
                    _ => {}
                }
            }
        }

        let shape = Shape::new(vec![header.size_info.rows, header.size_info.cols]);
        CooTensor::new(row_indices, col_indices, values, shape)
    }

    /// Write header section
    fn write_header<W: Write>(
        writer: &mut W,
        _tensor: &CooTensor,
        field: &MatrixMarketField,
        symmetry: &MatrixMarketSymmetry,
    ) -> TorshResult<()> {
        let object_str = "matrix";
        let format_str = "coordinate";
        let field_str = match *field {
            MatrixMarketField::Real => "real",
            MatrixMarketField::Integer => "integer",
            MatrixMarketField::Pattern => "pattern",
            MatrixMarketField::Complex => "complex",
        };
        let symmetry_str = match *symmetry {
            MatrixMarketSymmetry::General => "general",
            MatrixMarketSymmetry::Symmetric => "symmetric",
            MatrixMarketSymmetry::SkewSymmetric => "skew-symmetric",
            MatrixMarketSymmetry::Hermitian => "hermitian",
        };

        writeln!(
            writer,
            "%%MatrixMarket {object_str} {format_str} {field_str} {symmetry_str}"
        )
        .map_err(|e| TorshError::IoError(format!("Failed to write header: {e}")))?;

        // Write generation comment
        writeln!(writer, "% Generated by ToRSh sparse tensor library")
            .map_err(|e| TorshError::IoError(format!("Failed to write comment: {e}")))?;

        Ok(())
    }

    /// Write data section
    fn write_data<W: Write>(
        writer: &mut W,
        triplets: &[(usize, usize, f32)],
        field: &MatrixMarketField,
        symmetry: &MatrixMarketSymmetry,
    ) -> TorshResult<()> {
        // Filter triplets based on symmetry to avoid duplicates
        let filtered_triplets: Vec<_> = match *symmetry {
            MatrixMarketSymmetry::General => triplets.iter().collect(),
            MatrixMarketSymmetry::Symmetric
            | MatrixMarketSymmetry::SkewSymmetric
            | MatrixMarketSymmetry::Hermitian => {
                // Only write upper triangular part (including diagonal)
                triplets.iter().filter(|(r, c, _)| r <= c).collect()
            }
        };

        for (row, col, value) in filtered_triplets {
            match *field {
                MatrixMarketField::Real => {
                    writeln!(writer, "{} {} {:.16e}", row + 1, col + 1, value)
                        .map_err(|e| TorshError::IoError(format!("Failed to write data: {e}")))?;
                }
                MatrixMarketField::Integer => {
                    writeln!(writer, "{} {} {}", row + 1, col + 1, *value as i32)
                        .map_err(|e| TorshError::IoError(format!("Failed to write data: {e}")))?;
                }
                MatrixMarketField::Pattern => {
                    writeln!(writer, "{} {}", row + 1, col + 1)
                        .map_err(|e| TorshError::IoError(format!("Failed to write data: {e}")))?;
                }
                MatrixMarketField::Complex => {
                    return Err(TorshError::UnsupportedOperation {
                        op: "Matrix Market Complex field writing".to_string(),
                        dtype: "Complex".to_string(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Parse object type
    fn parse_object(s: &str) -> TorshResult<MatrixMarketObject> {
        match s.to_lowercase().as_str() {
            "matrix" => Ok(MatrixMarketObject::Matrix),
            "vector" => Ok(MatrixMarketObject::Vector),
            _ => Err(TorshError::InvalidArgument(format!(
                "Unknown object type: {s}"
            ))),
        }
    }

    /// Parse format type
    fn parse_format(s: &str) -> TorshResult<MatrixMarketFormat> {
        match s.to_lowercase().as_str() {
            "coordinate" => Ok(MatrixMarketFormat::Coordinate),
            "array" => Ok(MatrixMarketFormat::Array),
            _ => Err(TorshError::InvalidArgument(format!(
                "Unknown format type: {s}"
            ))),
        }
    }

    /// Parse field type
    fn parse_field(s: &str) -> TorshResult<MatrixMarketField> {
        match s.to_lowercase().as_str() {
            "real" => Ok(MatrixMarketField::Real),
            "complex" => Ok(MatrixMarketField::Complex),
            "integer" => Ok(MatrixMarketField::Integer),
            "pattern" => Ok(MatrixMarketField::Pattern),
            _ => Err(TorshError::InvalidArgument(format!(
                "Unknown field type: {s}"
            ))),
        }
    }

    /// Parse symmetry type
    fn parse_symmetry(s: &str) -> TorshResult<MatrixMarketSymmetry> {
        match s.to_lowercase().as_str() {
            "general" => Ok(MatrixMarketSymmetry::General),
            "symmetric" => Ok(MatrixMarketSymmetry::Symmetric),
            "skew-symmetric" => Ok(MatrixMarketSymmetry::SkewSymmetric),
            "hermitian" => Ok(MatrixMarketSymmetry::Hermitian),
            _ => Err(TorshError::InvalidArgument(format!(
                "Unknown symmetry type: {s}"
            ))),
        }
    }
}

/// Utility functions for Matrix Market format
pub struct MatrixMarketUtils;

impl MatrixMarketUtils {
    /// Detect symmetry in a sparse matrix
    pub fn detect_symmetry(tensor: &dyn SparseTensor) -> TorshResult<MatrixMarketSymmetry> {
        let coo = tensor.to_coo()?;
        let triplets = coo.triplets();
        let shape = coo.shape();

        if shape.dims()[0] != shape.dims()[1] {
            return Ok(MatrixMarketSymmetry::General); // Non-square matrices can't be symmetric
        }

        // Build maps for efficient lookup
        let mut entries: HashMap<(usize, usize), f32> = HashMap::new();
        for (r, c, v) in &triplets {
            entries.insert((*r, *c), *v);
        }

        let mut is_symmetric = true;
        let mut is_skew_symmetric = true;
        let tolerance = 1e-12;

        for (r, c, v) in &triplets {
            if let Some(&v_transpose) = entries.get(&(*c, *r)) {
                // Check symmetric
                if (*v - v_transpose).abs() > tolerance {
                    is_symmetric = false;
                }

                // Check skew-symmetric
                if (*v + v_transpose).abs() > tolerance {
                    is_skew_symmetric = false;
                }
            } else {
                // No corresponding transpose entry
                if v.abs() > tolerance {
                    is_symmetric = false;
                    is_skew_symmetric = false;
                }
            }
        }

        // Check diagonal for skew-symmetric (should be zero)
        if is_skew_symmetric {
            for (r, c, v) in &triplets {
                if *r == *c && v.abs() > tolerance {
                    is_skew_symmetric = false;
                    break;
                }
            }
        }

        if is_symmetric {
            Ok(MatrixMarketSymmetry::Symmetric)
        } else if is_skew_symmetric {
            Ok(MatrixMarketSymmetry::SkewSymmetric)
        } else {
            Ok(MatrixMarketSymmetry::General)
        }
    }

    /// Determine appropriate field type for a tensor
    pub fn detect_field_type(tensor: &dyn SparseTensor) -> MatrixMarketField {
        let coo = match tensor.to_coo() {
            Ok(coo) => coo,
            Err(_) => return MatrixMarketField::Real,
        };

        let triplets = coo.triplets();

        // Check if all values are exactly 1.0 (pattern matrix)
        let all_ones = triplets.iter().all(|(_, _, v)| (*v - 1.0).abs() < 1e-15);
        if all_ones {
            return MatrixMarketField::Pattern;
        }

        // Check if all values are integers
        let all_integers = triplets.iter().all(|(_, _, v)| (v.fract()).abs() < 1e-15);
        if all_integers {
            return MatrixMarketField::Integer;
        }

        MatrixMarketField::Real
    }

    /// Convert tensor to optimal Matrix Market representation
    pub fn optimize_for_matrix_market(
        tensor: &dyn SparseTensor,
    ) -> TorshResult<(CooTensor, MatrixMarketField, MatrixMarketSymmetry)> {
        let coo = tensor.to_coo()?;
        let field = Self::detect_field_type(tensor);
        let symmetry = Self::detect_symmetry(tensor)?;

        Ok((coo, field, symmetry))
    }

    /// Validate Matrix Market file format
    pub fn validate_file(file_path: &str) -> TorshResult<bool> {
        let file = File::open(file_path)
            .map_err(|e| TorshError::IoError(format!("Failed to open file: {e}")))?;

        let mut reader = BufReader::new(file);
        let mut line = String::new();

        // Check banner line
        reader
            .read_line(&mut line)
            .map_err(|e| TorshError::IoError(format!("Failed to read banner: {e}")))?;

        if !line.trim().starts_with("%%MatrixMarket") {
            return Ok(false);
        }

        // Try to parse header completely
        match MatrixMarketIO::parse_header(&mut reader) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Get file statistics without loading the entire matrix
    pub fn get_file_info(file_path: &str) -> TorshResult<MatrixMarketHeader> {
        let file = File::open(file_path)
            .map_err(|e| TorshError::IoError(format!("Failed to open file: {e}")))?;

        let mut reader = BufReader::new(file);
        MatrixMarketIO::parse_header(&mut reader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use torsh_core::Shape;

    #[test]
    fn test_matrix_market_roundtrip() {
        // Create a simple COO tensor
        let coo = CooTensor::new(
            vec![0, 1, 2],
            vec![0, 1, 2],
            vec![1.0, 2.0, 3.0],
            Shape::new(vec![3, 3]),
        )
        .unwrap();

        // Write to string
        let mut output = Vec::new();
        MatrixMarketIO::write_to_writer(
            &coo,
            &mut output,
            MatrixMarketField::Real,
            MatrixMarketSymmetry::General,
        )
        .unwrap();

        let output_str = String::from_utf8(output).unwrap();

        // Read back
        let cursor = Cursor::new(output_str.as_bytes());
        let (read_coo, header) = MatrixMarketIO::read_from_reader(cursor).unwrap();

        // Verify
        assert_eq!(header.field, MatrixMarketField::Real);
        assert_eq!(header.symmetry, MatrixMarketSymmetry::General);
        assert_eq!(read_coo.nnz(), 3);
    }

    #[test]
    fn test_symmetry_detection() {
        // Create symmetric matrix
        let coo = CooTensor::new(
            vec![0, 1, 1, 2],
            vec![1, 0, 2, 1],
            vec![1.0, 1.0, 2.0, 2.0],
            Shape::new(vec![3, 3]),
        )
        .unwrap();

        let symmetry = MatrixMarketUtils::detect_symmetry(&coo).unwrap();
        assert_eq!(symmetry, MatrixMarketSymmetry::Symmetric);
    }

    #[test]
    fn test_field_type_detection() {
        // Test pattern matrix (all 1s)
        let pattern_coo = CooTensor::new(
            vec![0, 1, 2],
            vec![0, 1, 2],
            vec![1.0, 1.0, 1.0],
            Shape::new(vec![3, 3]),
        )
        .unwrap();

        let field = MatrixMarketUtils::detect_field_type(&pattern_coo);
        assert_eq!(field, MatrixMarketField::Pattern);

        // Test integer matrix
        let int_coo = CooTensor::new(
            vec![0, 1, 2],
            vec![0, 1, 2],
            vec![1.0, 2.0, 3.0],
            Shape::new(vec![3, 3]),
        )
        .unwrap();

        let field = MatrixMarketUtils::detect_field_type(&int_coo);
        assert_eq!(field, MatrixMarketField::Integer);
    }

    #[test]
    fn test_header_parsing() {
        let header_text = "%%MatrixMarket matrix coordinate real general\n% Test matrix\n3 3 3\n";
        let mut cursor = Cursor::new(header_text.as_bytes());
        let header = MatrixMarketIO::parse_header(&mut cursor).unwrap();

        assert_eq!(header.object, MatrixMarketObject::Matrix);
        assert_eq!(header.format, MatrixMarketFormat::Coordinate);
        assert_eq!(header.field, MatrixMarketField::Real);
        assert_eq!(header.symmetry, MatrixMarketSymmetry::General);
        assert_eq!(header.size_info.rows, 3);
        assert_eq!(header.size_info.cols, 3);
        assert_eq!(header.size_info.nnz, 3);
    }
}
