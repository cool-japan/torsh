# ToRSh Sparse Format Reference

## Overview

This document provides detailed technical specifications for all sparse tensor formats supported by ToRSh-Sparse. Each format is optimized for specific use cases and access patterns.

## Format Comparison Table

| Format | Storage | Random Access | Sequential Access | Memory Overhead | Best Use Case |
|--------|---------|---------------|-------------------|-----------------|---------------|
| COO    | O(nnz)  | O(1)          | O(nnz)           | Low             | Construction, format conversion |
| CSR    | O(nnz)  | O(log n)      | O(nnz)           | Low             | Row-wise operations, SpMV |
| CSC    | O(nnz)  | O(log n)      | O(nnz)           | Low             | Column-wise operations, SpMV^T |
| BSR    | O(nnz)  | O(log n)      | O(nnz)           | Medium          | Block-structured matrices |
| DIA    | O(ndiag*n) | O(1)       | O(ndiag*n)       | High            | Diagonal/banded matrices |
| ELL    | O(maxnnz*n) | O(1)      | O(maxnnz*n)      | High            | Regular patterns, GPU |
| DSR    | O(nnz)  | O(log n)      | O(nnz)           | Medium          | Dynamic sparsity patterns |

## Coordinate Format (COO)

### Structure

The COO format stores sparse tensors as three arrays:
- `row_indices`: Row coordinates of non-zero elements
- `col_indices`: Column coordinates of non-zero elements  
- `values`: Non-zero values

### Memory Layout

```
Matrix:     COO Representation:
[1 0 3]     row_indices = [0, 0, 1, 2]
[0 2 0]     col_indices = [0, 2, 1, 2]
[0 0 4]     values      = [1, 3, 2, 4]
```

### Properties

- **Storage**: 3 * nnz elements (2 integers + 1 value per non-zero)
- **Memory overhead**: ~12-24 bytes per non-zero (depending on integer size)
- **Advantages**: 
  - Simple construction
  - Efficient for random insertions
  - Easy format conversion
  - Supports duplicate entries
- **Disadvantages**:
  - Not optimal for arithmetic operations
  - No locality for row/column access

### Operations Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Construction | O(nnz) | O(nnz) |
| Element access | O(nnz) | O(1) |
| Matrix-vector mult | O(nnz) | O(n) |
| Transpose | O(nnz) | O(nnz) |
| Addition | O(nnz1 + nnz2) | O(nnz1 + nnz2) |

### API Reference

```rust
use torsh_sparse::COOTensor;

// Creation
let coo = COOTensor::new(row_indices, col_indices, values, shape)?;
let coo = COOTensor::from_triplets(triplets, shape)?;
let coo = COOTensor::from_dense(&dense_tensor)?;

// Access
let value = coo.get(row, col)?;
let nnz = coo.nnz();
let shape = coo.shape();

// Conversion
let csr = coo.to_csr()?;
let csc = coo.to_csc()?;
let dense = coo.to_dense()?;
```

## Compressed Sparse Row (CSR)

### Structure

The CSR format uses three arrays:
- `row_ptrs`: Pointers to the start of each row in col_indices/values
- `col_indices`: Column indices of non-zero elements
- `values`: Non-zero values

### Memory Layout

```
Matrix:     CSR Representation:
[1 0 3]     row_ptrs    = [0, 2, 3, 4]
[0 2 0]     col_indices = [0, 2, 1, 2]
[0 0 4]     values      = [1, 3, 2, 4]
```

### Properties

- **Storage**: (n + 1) + 2 * nnz elements
- **Memory overhead**: ~8-16 bytes per non-zero + 4-8 bytes per row
- **Advantages**:
  - Efficient row-wise access
  - Optimal for matrix-vector multiplication
  - Cache-friendly for row operations
  - Compressed storage
- **Disadvantages**:
  - Expensive column access
  - Difficult to modify sparsity pattern

### Operations Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Construction from COO | O(nnz + n) | O(nnz + n) |
| Row access | O(1) | O(1) |
| Element access | O(log(nnz_row)) | O(1) |
| Matrix-vector mult | O(nnz) | O(n) |
| Transpose | O(nnz + n) | O(nnz + m) |

### API Reference

```rust
use torsh_sparse::CSRTensor;

// Creation
let csr = CSRTensor::new(row_ptrs, col_indices, values, shape)?;
let csr = CSRTensor::from_coo(&coo)?;
let csr = CSRTensor::from_dense(&dense_tensor)?;

// Access
let value = csr.get(row, col)?;
let row_slice = csr.get_row(row)?;
let nnz = csr.nnz();

// Operations
let result = csr.matvec(&vector)?;
let transposed = csr.transpose()?;
let sum = csr.sum()?;
```

## Compressed Sparse Column (CSC)

### Structure

The CSC format uses three arrays:
- `col_ptrs`: Pointers to the start of each column in row_indices/values
- `row_indices`: Row indices of non-zero elements
- `values`: Non-zero values

### Memory Layout

```
Matrix:     CSC Representation:
[1 0 3]     col_ptrs    = [0, 1, 2, 4]
[0 2 0]     row_indices = [0, 1, 0, 2]
[0 0 4]     values      = [1, 2, 3, 4]
```

### Properties

- **Storage**: (m + 1) + 2 * nnz elements
- **Memory overhead**: ~8-16 bytes per non-zero + 4-8 bytes per column
- **Advantages**:
  - Efficient column-wise access
  - Optimal for transpose matrix-vector multiplication
  - Cache-friendly for column operations
- **Disadvantages**:
  - Expensive row access
  - Difficult to modify sparsity pattern

### Operations Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Construction from COO | O(nnz + m) | O(nnz + m) |
| Column access | O(1) | O(1) |
| Element access | O(log(nnz_col)) | O(1) |
| Matrix-vector mult | O(nnz) | O(m) |
| Transpose | O(nnz + m) | O(nnz + n) |

### API Reference

```rust
use torsh_sparse::CSCTensor;

// Creation
let csc = CSCTensor::new(col_ptrs, row_indices, values, shape)?;
let csc = CSCTensor::from_coo(&coo)?;
let csc = CSCTensor::from_csr(&csr)?;

// Access
let value = csc.get(row, col)?;
let col_slice = csc.get_col(col)?;
let nnz = csc.nnz();

// Operations
let result = csc.vecmat(&vector)?; // vector^T * matrix
let transposed = csc.transpose()?;
```

## Block Sparse Row (BSR)

### Structure

The BSR format extends CSR to work with dense blocks:
- `block_ptrs`: Pointers to the start of each block row
- `block_indices`: Block column indices
- `blocks`: Dense blocks stored in row-major order

### Memory Layout

```
Matrix (2x2 blocks):  BSR Representation:
[1 2 | 0 0]          block_ptrs    = [0, 1, 2]
[3 4 | 0 0]          block_indices = [0, 1]
[----+----]          blocks        = [1,2,3,4, 5,6,7,8]
[0 0 | 5 6]
[0 0 | 7 8]
```

### Properties

- **Storage**: (n_block_rows + 1) + nnz_blocks + nnz_blocks * block_size²
- **Memory overhead**: Higher than CSR but enables vectorized operations
- **Advantages**:
  - Vectorized operations on blocks
  - Efficient for block-structured matrices
  - Better cache utilization
- **Disadvantages**:
  - Memory waste for sparse blocks
  - Complex construction

### Operations Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Construction | O(nnz_blocks * block_size²) | O(nnz_blocks * block_size²) |
| Block access | O(1) | O(1) |
| Element access | O(log(nnz_block_row)) | O(1) |
| Matrix-vector mult | O(nnz_blocks * block_size²) | O(n) |

### API Reference

```rust
use torsh_sparse::BSRTensor;

// Creation
let bsr = BSRTensor::new(block_ptrs, block_indices, blocks, block_size, shape)?;
let bsr = BSRTensor::from_csr(&csr, block_size)?;
let bsr = BSRTensor::from_dense(&dense_tensor, block_size)?;

// Access
let value = bsr.get(row, col)?;
let block = bsr.get_block(block_row, block_col)?;
let nnz_blocks = bsr.nnz_blocks();

// Operations
let result = bsr.matvec(&vector)?;
let transposed = bsr.transpose()?;
```

## Diagonal Format (DIA)

### Structure

The DIA format stores diagonals explicitly:
- `diagonals`: Values stored by diagonal
- `offsets`: Diagonal offsets from main diagonal

### Memory Layout

```
Matrix:     DIA Representation:
[1 2 0]     diagonals = [*, 1, 4]  <- diagonal -1
[0 3 4]                 [1, 3, 7]  <- diagonal 0
[0 0 7]                 [2, 4, *]  <- diagonal 1
            offsets   = [-1, 0, 1]
```

### Properties

- **Storage**: n_diag * n elements
- **Memory overhead**: High for sparse diagonals, minimal for dense diagonals
- **Advantages**:
  - Excellent for banded matrices
  - Predictable memory access patterns
  - Efficient diagonal operations
- **Disadvantages**:
  - Wastes memory for irregular patterns
  - Not suitable for general sparse matrices

### Operations Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Construction | O(n_diag * n) | O(n_diag * n) |
| Diagonal access | O(1) | O(1) |
| Element access | O(1) | O(1) |
| Matrix-vector mult | O(n_diag * n) | O(n) |

### API Reference

```rust
use torsh_sparse::DIATensor;

// Creation
let dia = DIATensor::new(diagonals, offsets, shape)?;
let dia = DIATensor::from_csr(&csr)?;
let dia = DIATensor::from_dense(&dense_tensor)?;

// Access
let value = dia.get(row, col)?;
let diagonal = dia.get_diagonal(offset)?;
let n_diag = dia.n_diag();

// Operations
let result = dia.matvec(&vector)?;
let sum = dia.sum()?;
```

## ELLPACK Format (ELL)

### Structure

The ELL format uses padded arrays:
- `indices`: Column indices padded to max_nnz_per_row
- `values`: Values padded to max_nnz_per_row

### Memory Layout

```
Matrix:     ELL Representation (max_nnz_per_row = 2):
[1 0 3]     indices = [0, 2]  <- row 0
[0 2 0]               [1, *]  <- row 1 (padded)
[0 0 4]               [2, *]  <- row 2 (padded)
            values  = [1, 3]
                      [2, *]
                      [4, *]
```

### Properties

- **Storage**: max_nnz_per_row * n * 2 elements
- **Memory overhead**: Very high for irregular patterns
- **Advantages**:
  - Coalesced memory access
  - Excellent for GPU computation
  - Predictable memory layout
- **Disadvantages**:
  - Significant memory waste
  - Only suitable for regular patterns

### Operations Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Construction | O(max_nnz_per_row * n) | O(max_nnz_per_row * n) |
| Row access | O(1) | O(1) |
| Element access | O(max_nnz_per_row) | O(1) |
| Matrix-vector mult | O(max_nnz_per_row * n) | O(n) |

### API Reference

```rust
use torsh_sparse::ELLTensor;

// Creation
let ell = ELLTensor::new(indices, values, max_nnz_per_row, shape)?;
let ell = ELLTensor::from_csr(&csr)?;
let ell = ELLTensor::from_dense(&dense_tensor)?;

// Access
let value = ell.get(row, col)?;
let row_slice = ell.get_row(row)?;
let max_nnz = ell.max_nnz_per_row();

// Operations
let result = ell.matvec(&vector)?;
let efficiency = ell.storage_efficiency()?;
```

## Dynamic Sparse Row (DSR)

### Structure

The DSR format uses dynamic data structures:
- `rows`: Vector of BTreeMap<column_index, value>
- Each row maintains sorted column indices

### Memory Layout

```
Matrix:     DSR Representation:
[1 0 3]     rows[0] = BTreeMap{0: 1.0, 2: 3.0}
[0 2 0]     rows[1] = BTreeMap{1: 2.0}
[0 0 4]     rows[2] = BTreeMap{2: 4.0}
```

### Properties

- **Storage**: ~nnz * (key + value + overhead) bytes
- **Memory overhead**: Higher than static formats due to tree structure
- **Advantages**:
  - Efficient insertion/deletion
  - Maintains sorted order
  - Good for dynamic sparsity patterns
- **Disadvantages**:
  - Memory overhead
  - Slower bulk operations

### Operations Complexity

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Construction | O(nnz * log(max_nnz_per_row)) | O(nnz) |
| Element access | O(log(nnz_row)) | O(1) |
| Element insertion | O(log(nnz_row)) | O(1) |
| Matrix-vector mult | O(nnz) | O(n) |

### API Reference

```rust
use torsh_sparse::DSRTensor;

// Creation
let mut dsr = DSRTensor::new(shape)?;
let dsr = DSRTensor::from_csr(&csr)?;
let dsr = DSRTensor::from_dense(&dense_tensor)?;

// Access and modification
let value = dsr.get(row, col)?;
dsr.set(row, col, value)?;
dsr.remove(row, col)?;

// Operations
let result = dsr.matvec(&vector)?;
let transposed = dsr.transpose()?;
```

## Format Selection Guidelines

### Use COO When:
- Building sparse matrices incrementally
- Converting between formats
- Performing one-time operations
- Working with unsorted data

### Use CSR When:
- Performing matrix-vector multiplication
- Row-wise operations are primary
- Memory efficiency is important
- Sparsity pattern is stable

### Use CSC When:
- Performing transpose matrix-vector multiplication
- Column-wise operations are primary
- Solving linear systems (some solvers prefer CSC)
- Working with tall, skinny matrices

### Use BSR When:
- Matrix has block structure
- Blocks are reasonably dense
- Vectorized operations are beneficial
- Working with finite element matrices

### Use DIA When:
- Matrix is diagonal or banded
- Diagonal operations are common
- Memory access patterns are predictable
- Working with differential operators

### Use ELL When:
- Regular sparsity patterns
- GPU computation is primary
- Memory bandwidth is crucial
- All rows have similar number of non-zeros

### Use DSR When:
- Sparsity pattern changes frequently
- Interactive applications
- Incremental matrix construction
- Need efficient insertion/deletion

## Performance Considerations

### Memory Access Patterns

1. **Sequential Access**: CSR/CSC for row/column-wise operations
2. **Random Access**: COO for arbitrary element access
3. **Blocked Access**: BSR for block-structured operations
4. **Predictable Access**: DIA/ELL for regular patterns

### Cache Performance

1. **Cache-Friendly**: CSR for row operations, CSC for column operations
2. **Cache-Unfriendly**: Random access in COO
3. **Block-Friendly**: BSR for block operations
4. **Streaming**: DIA/ELL for predictable patterns

### Parallel Processing

1. **Row-Parallel**: CSR enables easy row-wise parallelization
2. **Column-Parallel**: CSC enables easy column-wise parallelization
3. **Block-Parallel**: BSR enables block-wise parallelization
4. **Element-Parallel**: COO enables element-wise parallelization

### GPU Considerations

1. **Coalesced Access**: ELL provides best memory coalescing
2. **Warp Efficiency**: Regular patterns (DIA/ELL) improve warp utilization
3. **Memory Bandwidth**: CSR/CSC balance memory usage and bandwidth
4. **Dynamic Patterns**: Avoid DSR on GPU due to pointer chasing

## Conclusion

Each sparse format in ToRSh-Sparse is designed for specific use cases and access patterns. Understanding these characteristics is crucial for optimal performance. The library provides automatic format selection and optimization tools to help choose the best format for your specific application.

For more information on performance optimization and format selection, see the [Performance Guide](PERFORMANCE_GUIDE.md) and [Sparse Guide](SPARSE_GUIDE.md).