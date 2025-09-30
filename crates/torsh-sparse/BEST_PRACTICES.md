# ToRSh Sparse Best Practices Guide

## Overview

This guide provides best practices, design patterns, and optimization techniques for using ToRSh-Sparse effectively in production applications. Following these practices will help you build robust, efficient, and maintainable sparse tensor applications.

## Table of Contents

1. [Format Selection Guidelines](#format-selection-guidelines)
2. [Performance Best Practices](#performance-best-practices)
3. [Memory Management](#memory-management)
4. [Error Handling and Robustness](#error-handling-and-robustness)
5. [Code Organization](#code-organization)
6. [Testing Strategies](#testing-strategies)
7. [Deployment Considerations](#deployment-considerations)
8. [Common Pitfalls](#common-pitfalls)
9. [Design Patterns](#design-patterns)
10. [Optimization Checklist](#optimization-checklist)

## Format Selection Guidelines

### Choose the Right Format for Your Use Case

#### CSR (Compressed Sparse Row)
**Use when:**
- Performing frequent row-wise operations
- Matrix-vector multiplication is primary operation
- Need efficient sequential row access
- Building general-purpose sparse matrix libraries

**Avoid when:**
- Primarily doing column operations
- Frequent random element insertion/deletion
- Memory is extremely constrained

```rust
// Good: Row-wise operations with CSR
let csr_matrix = CSRTensor::from_triplets(triplets, shape)?;
for row_idx in 0..csr_matrix.nrows() {
    let row = csr_matrix.get_row(row_idx)?;
    // Process row efficiently
}

// Bad: Column operations with CSR (inefficient)
for col_idx in 0..csr_matrix.ncols() {
    let col = csr_matrix.get_col(col_idx)?; // Slow!
}
```

#### CSC (Compressed Sparse Column)
**Use when:**
- Performing frequent column-wise operations
- Transpose matrix-vector multiplication
- Linear algebra algorithms requiring column access
- Interfacing with column-major libraries

**Example:**
```rust
// Good: Column-wise operations with CSC
let csc_matrix = CSCTensor::from_csr(&csr_matrix)?;
for col_idx in 0..csc_matrix.ncols() {
    let col = csc_matrix.get_col(col_idx)?;
    // Process column efficiently
}
```

#### COO (Coordinate)
**Use when:**
- Building sparse matrices incrementally
- Converting between formats
- One-time operations on unsorted data
- Parallel construction from multiple threads

**Avoid when:**
- Performing arithmetic operations repeatedly
- Need optimized matrix-vector multiplication
- Memory usage is critical

```rust
// Good: Incremental construction with COO
let mut triplets = Vec::new();
for data_point in data_stream {
    triplets.push((data_point.row, data_point.col, data_point.value));
}
let coo_matrix = COOTensor::from_triplets(triplets, shape)?;

// Convert to optimized format for operations
let csr_matrix = CSRTensor::from_coo(&coo_matrix)?;
```

#### Specialized Formats (BSR, DIA, ELL)
**Use when:**
- Matrix has specific structural patterns
- Performance is critical
- Memory bandwidth is the bottleneck

```rust
// BSR for block-structured matrices
if has_block_structure(&matrix) {
    let block_size = detect_optimal_block_size(&matrix)?;
    let bsr_matrix = BSRTensor::from_csr(&csr_matrix, block_size)?;
}

// DIA for banded matrices
if is_banded(&matrix) {
    let dia_matrix = DIATensor::from_csr(&csr_matrix)?;
}
```

### Dynamic Format Selection

```rust
use torsh_sparse::{auto_select_format, OperationType};

fn select_optimal_format(matrix: &CSRTensor, operations: &[OperationType]) -> Result<SparseFormat, TorshError> {
    // Analyze matrix characteristics
    let analysis = analyze_sparsity_pattern(matrix)?;
    
    // Consider operation types
    let format = if analysis.is_diagonal {
        SparseFormat::DIA
    } else if analysis.has_block_structure && analysis.density > 0.1 {
        SparseFormat::BSR
    } else if operations.contains(&OperationType::MatVec) {
        SparseFormat::CSR
    } else if operations.contains(&OperationType::VecMat) {
        SparseFormat::CSC
    } else {
        auto_select_format(matrix, operations)?
    };
    
    Ok(format)
}
```

## Performance Best Practices

### 1. Minimize Format Conversions

```rust
// Bad: Multiple conversions in loop
for iteration in 0..num_iterations {
    let csc_matrix = CSCTensor::from_csr(&csr_matrix)?; // Expensive!
    let result = csc_matrix.vecmat(&vector)?;
}

// Good: Convert once, reuse
let csc_matrix = CSCTensor::from_csr(&csr_matrix)?;
for iteration in 0..num_iterations {
    let result = csc_matrix.vecmat(&vector)?;
}
```

### 2. Use Unified Interface for Complex Workflows

```rust
use torsh_sparse::UnifiedSparseTensor;

// Good: Automatic optimization
let unified = UnifiedSparseTensor::from_csr(csr_matrix)?;
let optimized = unified.optimize_for_operations(&[
    OperationType::MatVec,
    OperationType::Transpose,
    OperationType::Addition,
])?;

// Performs operations with optimal formats
let result1 = optimized.matvec(&vector)?;      // Uses CSR
let result2 = optimized.transpose()?;          // Uses CSC
let result3 = optimized.add(&other_matrix)?;   // Uses optimal format
```

### 3. Leverage Memory Pools

```rust
use torsh_sparse::memory_management::MemoryPool;

fn efficient_batch_processing(matrices: &[CSRTensor]) -> Result<Vec<f64>, TorshError> {
    // Create memory pool
    let pool = MemoryPool::new(1_000_000_000)?; // 1GB pool
    
    let mut results = Vec::new();
    for matrix in matrices {
        // Allocate temporary matrix from pool
        let temp_matrix = pool.allocate_like(matrix)?;
        
        // Perform operations using pool memory
        let result = pool.multiply_matrices(matrix, &temp_matrix)?;
        results.push(result.sum()?);
        
        // Memory automatically returned to pool
    }
    
    Ok(results)
}
```

### 4. Use SIMD and Parallel Operations

```rust
use torsh_sparse::parallel::*;
use torsh_sparse::kernels::simd::*;

// Enable SIMD optimizations
if supports_avx2() {
    enable_avx2_kernels();
}

// Use parallel operations for large matrices
let config = ParallelConfig::new()
    .num_threads(8)
    .chunk_size(1000)
    .load_balancing(LoadBalancing::Dynamic);

let result = parallel_spmv(&large_matrix, &vector, &config)?;
```

### 5. Profile and Optimize Hot Paths

```rust
use torsh_sparse::profiling::*;

fn optimized_computation() -> Result<(), TorshError> {
    let profiler = Profiler::new();
    profiler.enable();
    
    // Your computation here
    heavy_sparse_computation()?;
    
    let report = profiler.get_report();
    
    // Identify bottlenecks
    for (operation, timing) in &report.operation_times {
        if timing.total_time > 100.0 { // > 100ms
            println!("Bottleneck: {} took {:.2}ms", operation, timing.total_time);
        }
    }
    
    Ok(())
}
```

## Memory Management

### 1. Monitor Memory Usage

```rust
use torsh_sparse::memory_management::*;

struct MemoryAwareApplication {
    memory_budget: usize,
    current_usage: usize,
    matrices: Vec<Box<dyn SparseTensor>>,
}

impl MemoryAwareApplication {
    fn add_matrix(&mut self, matrix: Box<dyn SparseTensor>) -> Result<(), TorshError> {
        let matrix_size = matrix.memory_usage();
        
        if self.current_usage + matrix_size > self.memory_budget {
            self.cleanup_unused_matrices()?;
        }
        
        if self.current_usage + matrix_size <= self.memory_budget {
            self.current_usage += matrix_size;
            self.matrices.push(matrix);
            Ok(())
        } else {
            Err(TorshError::OutOfMemory { 
                requested: matrix_size,
                available: self.memory_budget - self.current_usage,
            })
        }
    }
    
    fn cleanup_unused_matrices(&mut self) -> Result<(), TorshError> {
        // Implement LRU or other cleanup strategy
        Ok(())
    }
}
```

### 2. Use Streaming for Large Datasets

```rust
fn process_large_dataset(file_path: &str) -> Result<(), TorshError> {
    let chunk_size = 10000; // Process 10k rows at a time
    let mut row_offset = 0;
    
    loop {
        // Load chunk
        let chunk = load_sparse_chunk(file_path, row_offset, chunk_size)?;
        if chunk.nnz() == 0 {
            break; // End of file
        }
        
        // Process chunk
        let result = process_sparse_chunk(&chunk)?;
        save_result(&result, row_offset)?;
        
        row_offset += chunk_size;
        
        // Force garbage collection periodically
        if row_offset % (chunk_size * 10) == 0 {
            force_gc()?;
        }
    }
    
    Ok(())
}
```

### 3. Implement Custom Memory Allocators

```rust
struct SparseMatrixAllocator {
    pool: MemoryPool,
    allocation_strategy: AllocationStrategy,
}

impl SparseMatrixAllocator {
    fn allocate_optimized<T: SparseTensor>(&self, 
                                         shape: (usize, usize), 
                                         nnz: usize) -> Result<T, TorshError> {
        match self.allocation_strategy {
            AllocationStrategy::MemoryFirst => {
                // Optimize for memory usage
                self.pool.allocate_compressed(shape, nnz)
            },
            AllocationStrategy::SpeedFirst => {
                // Optimize for speed
                self.pool.allocate_cache_aligned(shape, nnz)
            },
            AllocationStrategy::Balanced => {
                // Balance memory and speed
                self.pool.allocate_balanced(shape, nnz)
            },
        }
    }
}
```

## Error Handling and Robustness

### 1. Comprehensive Error Handling

```rust
use torsh_sparse::{TorshError, Result};

fn robust_sparse_operation(matrix: &CSRTensor, vector: &[f64]) -> Result<Vec<f64>> {
    // Validate inputs
    if matrix.ncols() != vector.len() {
        return Err(TorshError::DimensionMismatch {
            expected: matrix.ncols(),
            actual: vector.len(),
        });
    }
    
    // Check for numerical issues
    if matrix.has_inf_or_nan() {
        return Err(TorshError::NumericalError {
            message: "Matrix contains infinite or NaN values".to_string(),
        });
    }
    
    // Perform operation with error handling
    match matrix.matvec(vector) {
        Ok(result) => {
            // Validate result
            if result.iter().any(|&x| !x.is_finite()) {
                Err(TorshError::NumericalError {
                    message: "Result contains non-finite values".to_string(),
                })
            } else {
                Ok(result)
            }
        },
        Err(e) => {
            // Log error and provide context
            log::error!("Matrix-vector multiplication failed: {:?}", e);
            Err(e)
        }
    }
}
```

### 2. Input Validation

```rust
fn validate_sparse_matrix(matrix: &dyn SparseTensor) -> Result<()> {
    // Check dimensions
    let (nrows, ncols) = matrix.shape();
    if nrows == 0 || ncols == 0 {
        return Err(TorshError::InvalidInput {
            message: "Matrix dimensions must be positive".to_string(),
        });
    }
    
    // Check for valid indices
    for (row, col, _) in matrix.triplets() {
        if row >= nrows || col >= ncols {
            return Err(TorshError::IndexOutOfBounds {
                index: (row, col),
                shape: (nrows, ncols),
            });
        }
    }
    
    // Check for numerical validity
    for (_, _, value) in matrix.triplets() {
        if !value.is_finite() {
            return Err(TorshError::NumericalError {
                message: format!("Non-finite value found: {}", value),
            });
        }
    }
    
    Ok(())
}
```

### 3. Graceful Degradation

```rust
fn adaptive_sparse_computation(matrix: &CSRTensor, 
                              vector: &[f64]) -> Result<Vec<f64>> {
    // Try optimized algorithm first
    match optimized_spmv(matrix, vector) {
        Ok(result) => Ok(result),
        Err(TorshError::OutOfMemory { .. }) => {
            log::warn!("Falling back to memory-efficient algorithm");
            memory_efficient_spmv(matrix, vector)
        },
        Err(TorshError::UnsupportedOperation { .. }) => {
            log::warn!("Falling back to general algorithm");
            general_spmv(matrix, vector)
        },
        Err(e) => Err(e),
    }
}
```

## Code Organization

### 1. Modular Design

```rust
// sparse_ops.rs - Core operations
pub struct SparseOperations {
    memory_pool: MemoryPool,
    profiler: Option<Profiler>,
}

impl SparseOperations {
    pub fn new(memory_budget: usize) -> Result<Self> {
        Ok(Self {
            memory_pool: MemoryPool::new(memory_budget)?,
            profiler: None,
        })
    }
    
    pub fn enable_profiling(&mut self) {
        self.profiler = Some(Profiler::new());
    }
    
    pub fn multiply(&self, a: &CSRTensor, b: &CSRTensor) -> Result<CSRTensor> {
        if let Some(ref profiler) = self.profiler {
            profiler.start_operation("sparse_multiply")?;
        }
        
        let result = self.memory_pool.multiply_matrices(a, b)?;
        
        if let Some(ref profiler) = self.profiler {
            profiler.end_operation("sparse_multiply")?;
        }
        
        Ok(result)
    }
}

// format_manager.rs - Format selection and conversion
pub struct FormatManager {
    cache: HashMap<String, Box<dyn SparseTensor>>,
    auto_optimize: bool,
}

impl FormatManager {
    pub fn get_optimal_format(&self, 
                             matrix_id: &str, 
                             operations: &[OperationType]) -> Result<&dyn SparseTensor> {
        if self.auto_optimize {
            self.get_optimized_for_operations(matrix_id, operations)
        } else {
            self.get_cached(matrix_id)
        }
    }
}

// application.rs - High-level application logic
pub struct SparseApplication {
    operations: SparseOperations,
    format_manager: FormatManager,
    config: ApplicationConfig,
}
```

### 2. Configuration Management

```rust
#[derive(Debug, Clone)]
pub struct SparseConfig {
    pub memory_budget: usize,
    pub enable_profiling: bool,
    pub auto_format_selection: bool,
    pub parallel_threshold: usize,
    pub simd_enabled: bool,
    pub cache_size: usize,
}

impl Default for SparseConfig {
    fn default() -> Self {
        Self {
            memory_budget: 1_000_000_000, // 1GB
            enable_profiling: false,
            auto_format_selection: true,
            parallel_threshold: 10000,
            simd_enabled: true,
            cache_size: 100,
        }
    }
}

impl SparseConfig {
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();
        
        if let Ok(budget) = std::env::var("TORSH_MEMORY_BUDGET") {
            config.memory_budget = budget.parse()?;
        }
        
        if let Ok(profiling) = std::env::var("TORSH_ENABLE_PROFILING") {
            config.enable_profiling = profiling.parse()?;
        }
        
        Ok(config)
    }
}
```

### 3. Trait-Based Design

```rust
pub trait SparseComputation {
    type Input;
    type Output;
    type Error;
    
    fn compute(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;
    fn validate_input(&self, input: &Self::Input) -> Result<(), Self::Error>;
    fn estimated_memory_usage(&self, input: &Self::Input) -> usize;
}

pub struct MatrixVectorMultiplication {
    config: ComputationConfig,
}

impl SparseComputation for MatrixVectorMultiplication {
    type Input = (CSRTensor, Vec<f64>);
    type Output = Vec<f64>;
    type Error = TorshError;
    
    fn compute(&self, (matrix, vector): Self::Input) -> Result<Self::Output, Self::Error> {
        self.validate_input(&(matrix, vector))?;
        
        if self.estimated_memory_usage(&(matrix, vector)) > self.config.memory_limit {
            self.compute_streaming((matrix, vector))
        } else {
            matrix.matvec(&vector)
        }
    }
    
    // ... implement other methods
}
```

## Testing Strategies

### 1. Property-Based Testing

```rust
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    
    // Generate random sparse matrices for testing
    fn sparse_matrix_strategy() -> impl Strategy<Value = CSRTensor> {
        (1usize..100, 1usize..100, 0.01f64..0.5f64)
            .prop_flat_map(|(rows, cols, density)| {
                let nnz = ((rows * cols) as f64 * density) as usize;
                prop::collection::vec(
                    (0..rows, 0..cols, -10.0..10.0f64),
                    nnz..nnz+1
                ).prop_map(move |triplets| {
                    CSRTensor::from_triplets(triplets, (rows, cols)).unwrap()
                })
            })
    }
    
    proptest! {
        #[test]
        fn matrix_vector_multiplication_properties(
            matrix in sparse_matrix_strategy(),
            vector in prop::collection::vec(-10.0..10.0f64, 1..100)
        ) {
            if matrix.ncols() == vector.len() {
                let result = matrix.matvec(&vector);
                prop_assert!(result.is_ok());
                
                let result = result.unwrap();
                prop_assert_eq!(result.len(), matrix.nrows());
                prop_assert!(result.iter().all(|&x| x.is_finite()));
            }
        }
        
        #[test]
        fn format_conversion_preserves_data(matrix in sparse_matrix_strategy()) {
            let coo = COOTensor::from_csr(&matrix).unwrap();
            let csr_back = CSRTensor::from_coo(&coo).unwrap();
            
            prop_assert_eq!(matrix.shape(), csr_back.shape());
            prop_assert_eq!(matrix.nnz(), csr_back.nnz());
            
            // Check that all elements are preserved
            for i in 0..matrix.nrows() {
                for j in 0..matrix.ncols() {
                    prop_assert!((matrix.get(i, j).unwrap() - csr_back.get(i, j).unwrap()).abs() < 1e-10);
                }
            }
        }
    }
}
```

### 2. Performance Regression Testing

```rust
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn benchmark_matrix_vector_multiplication() {
        let matrix = create_test_matrix(10000, 0.01);
        let vector = vec![1.0; 10000];
        
        let start = Instant::now();
        let _result = matrix.matvec(&vector).unwrap();
        let duration = start.elapsed();
        
        // Assert performance doesn't regress
        assert!(duration.as_millis() < 100, 
               "Matrix-vector multiplication took {}ms, expected <100ms", 
               duration.as_millis());
    }
    
    #[test]
    fn memory_usage_test() {
        let initial_memory = get_memory_usage();
        
        {
            let large_matrix = create_test_matrix(50000, 0.001);
            let current_memory = get_memory_usage();
            
            // Check memory usage is reasonable
            let memory_increase = current_memory - initial_memory;
            let expected_memory = large_matrix.nnz() * 16; // Approximate
            
            assert!(memory_increase <= expected_memory * 2,
                   "Memory usage {} exceeds expected {}", 
                   memory_increase, expected_memory);
        }
        
        // Force garbage collection and check for leaks
        force_gc();
        let final_memory = get_memory_usage();
        
        assert!((final_memory as i64 - initial_memory as i64).abs() < 1_000_000,
               "Potential memory leak detected");
    }
}
```

### 3. Integration Testing

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn full_workflow_test() {
        // Test complete workflow from creation to computation
        let triplets = vec![(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0)];
        let coo = COOTensor::from_triplets(triplets, (3, 3)).unwrap();
        
        // Test format conversions
        let csr = CSRTensor::from_coo(&coo).unwrap();
        let csc = CSCTensor::from_csr(&csr).unwrap();
        let bsr = BSRTensor::from_csr(&csr, (1, 1)).unwrap();
        
        // Test operations on all formats
        let vector = vec![1.0, 2.0, 3.0];
        
        let result_csr = csr.matvec(&vector).unwrap();
        let result_csc = csc.matvec(&vector).unwrap();
        let result_bsr = bsr.matvec(&vector).unwrap();
        
        // All results should be identical
        for i in 0..3 {
            assert!((result_csr[i] - result_csc[i]).abs() < 1e-10);
            assert!((result_csr[i] - result_bsr[i]).abs() < 1e-10);
        }
    }
}
```

## Deployment Considerations

### 1. Environment-Specific Optimizations

```rust
pub fn configure_for_environment() -> SparseConfig {
    let mut config = SparseConfig::default();
    
    // Detect hardware capabilities
    if has_avx2_support() {
        config.simd_enabled = true;
        log::info!("AVX2 support detected, enabling SIMD optimizations");
    }
    
    // Adjust for available memory
    let available_memory = get_available_memory();
    config.memory_budget = (available_memory * 0.8) as usize; // Use 80% of available
    
    // Adjust for CPU count
    let cpu_count = num_cpus::get();
    config.parallel_threshold = 10000 / cpu_count; // Scale with CPU count
    
    config
}
```

### 2. Monitoring and Observability

```rust
use tracing::{info, warn, error, debug};

pub struct SparseMetrics {
    operation_counts: HashMap<String, u64>,
    operation_times: HashMap<String, Duration>,
    memory_usage: u64,
    cache_hit_rate: f64,
}

impl SparseMetrics {
    pub fn record_operation(&mut self, operation: &str, duration: Duration) {
        *self.operation_counts.entry(operation.to_string()).or_insert(0) += 1;
        *self.operation_times.entry(operation.to_string()).or_insert(Duration::ZERO) += duration;
        
        info!("Operation {} completed in {:?}", operation, duration);
        
        if duration > Duration::from_millis(1000) {
            warn!("Slow operation detected: {} took {:?}", operation, duration);
        }
    }
    
    pub fn export_metrics(&self) -> serde_json::Value {
        serde_json::json!({
            "operation_counts": self.operation_counts,
            "average_operation_times": self.operation_times
                .iter()
                .map(|(op, total_time)| {
                    let count = self.operation_counts.get(op).unwrap_or(&1);
                    (op, total_time.as_millis() / *count as u128)
                })
                .collect::<HashMap<_, _>>(),
            "memory_usage_mb": self.memory_usage / 1_000_000,
            "cache_hit_rate": self.cache_hit_rate,
        })
    }
}
```

### 3. Configuration Management

```rust
#[derive(Serialize, Deserialize)]
pub struct DeploymentConfig {
    pub sparse_config: SparseConfig,
    pub logging_level: String,
    pub metrics_endpoint: Option<String>,
    pub health_check_interval: Duration,
}

impl DeploymentConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&content)?;
        Ok(config)
    }
    
    pub fn from_env() -> Result<Self> {
        let sparse_config = SparseConfig::from_env()?;
        
        Ok(Self {
            sparse_config,
            logging_level: std::env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
            metrics_endpoint: std::env::var("METRICS_ENDPOINT").ok(),
            health_check_interval: Duration::from_secs(
                std::env::var("HEALTH_CHECK_INTERVAL")
                    .unwrap_or_else(|_| "30".to_string())
                    .parse()
                    .unwrap_or(30)
            ),
        })
    }
}
```

## Common Pitfalls

### 1. Inefficient Format Usage

```rust
// BAD: Using COO for repeated operations
let coo_matrix = COOTensor::from_triplets(triplets, shape)?;
for _ in 0..1000 {
    let result = coo_matrix.matvec(&vector)?; // Inefficient!
}

// GOOD: Convert to efficient format once
let coo_matrix = COOTensor::from_triplets(triplets, shape)?;
let csr_matrix = CSRTensor::from_coo(&coo_matrix)?;
for _ in 0..1000 {
    let result = csr_matrix.matvec(&vector)?; // Efficient!
}
```

### 2. Memory Leaks in Long-Running Applications

```rust
// BAD: Accumulating temporary matrices
let mut results = Vec::new();
for data in large_dataset {
    let temp_matrix = process_data(data)?;
    let result = expensive_operation(&temp_matrix)?;
    results.push(result);
    // temp_matrix not explicitly freed
}

// GOOD: Explicit memory management
let memory_pool = MemoryPool::new(memory_budget)?;
let mut results = Vec::new();
for data in large_dataset {
    let temp_matrix = memory_pool.process_data(data)?;
    let result = expensive_operation(&temp_matrix)?;
    results.push(result);
    memory_pool.deallocate(temp_matrix)?; // Explicit cleanup
    
    if results.len() % 1000 == 0 {
        memory_pool.garbage_collect()?; // Periodic cleanup
    }
}
```

### 3. Ignoring Numerical Stability

```rust
// BAD: No numerical validation
fn risky_computation(matrix: &CSRTensor) -> Result<f64> {
    let result = matrix.norm(2.0)?;
    Ok(result * 1e20) // Could overflow
}

// GOOD: Proper numerical handling
fn safe_computation(matrix: &CSRTensor) -> Result<f64> {
    let norm = matrix.norm(2.0)?;
    
    // Check for potential overflow
    if norm > f64::MAX / 1e20 {
        return Err(TorshError::NumericalError {
            message: "Computation would overflow".to_string(),
        });
    }
    
    // Check for underflow
    if norm < f64::MIN_POSITIVE * 1e20 {
        return Err(TorshError::NumericalError {
            message: "Computation would underflow".to_string(),
        });
    }
    
    Ok(norm * 1e20)
}
```

## Design Patterns

### 1. Builder Pattern for Complex Matrices

```rust
pub struct SparseMatrixBuilder {
    triplets: Vec<(usize, usize, f64)>,
    shape: Option<(usize, usize)>,
    format: SparseFormat,
    sorted: bool,
    deduplicated: bool,
}

impl SparseMatrixBuilder {
    pub fn new() -> Self {
        Self {
            triplets: Vec::new(),
            shape: None,
            format: SparseFormat::CSR,
            sorted: false,
            deduplicated: false,
        }
    }
    
    pub fn add_triplet(mut self, row: usize, col: usize, value: f64) -> Self {
        self.triplets.push((row, col, value));
        self
    }
    
    pub fn shape(mut self, shape: (usize, usize)) -> Self {
        self.shape = Some(shape);
        self
    }
    
    pub fn format(mut self, format: SparseFormat) -> Self {
        self.format = format;
        self
    }
    
    pub fn sorted(mut self) -> Self {
        self.sorted = true;
        self
    }
    
    pub fn deduplicated(mut self) -> Self {
        self.deduplicated = true;
        self
    }
    
    pub fn build(self) -> Result<Box<dyn SparseTensor>> {
        let shape = self.shape.ok_or_else(|| TorshError::InvalidInput {
            message: "Shape must be specified".to_string(),
        })?;
        
        let mut triplets = self.triplets;
        
        if self.sorted {
            triplets.sort_by_key(|&(r, c, _)| (r, c));
        }
        
        if self.deduplicated {
            // Deduplicate triplets
            let mut dedup_triplets = Vec::new();
            let mut last_key = None;
            let mut sum = 0.0;
            
            for (r, c, v) in triplets {
                let key = (r, c);
                if last_key == Some(key) {
                    sum += v;
                } else {
                    if let Some((lr, lc)) = last_key {
                        if sum != 0.0 {
                            dedup_triplets.push((lr, lc, sum));
                        }
                    }
                    last_key = Some(key);
                    sum = v;
                }
            }
            
            if let Some((lr, lc)) = last_key {
                if sum != 0.0 {
                    dedup_triplets.push((lr, lc, sum));
                }
            }
            
            triplets = dedup_triplets;
        }
        
        match self.format {
            SparseFormat::COO => {
                let coo = COOTensor::from_triplets(triplets, shape)?;
                Ok(Box::new(coo))
            },
            SparseFormat::CSR => {
                let coo = COOTensor::from_triplets(triplets, shape)?;
                let csr = CSRTensor::from_coo(&coo)?;
                Ok(Box::new(csr))
            },
            SparseFormat::CSC => {
                let coo = COOTensor::from_triplets(triplets, shape)?;
                let csc = CSCTensor::from_coo(&coo)?;
                Ok(Box::new(csc))
            },
            _ => Err(TorshError::UnsupportedOperation {
                op: format!("Building matrix in {:?} format", self.format),
            }),
        }
    }
}
```

### 2. Strategy Pattern for Algorithm Selection

```rust
pub trait SpMVStrategy {
    fn multiply(&self, matrix: &CSRTensor, vector: &[f64]) -> Result<Vec<f64>>;
    fn estimated_memory(&self, matrix: &CSRTensor) -> usize;
    fn estimated_time(&self, matrix: &CSRTensor) -> Duration;
}

pub struct StandardSpMV;
pub struct ParallelSpMV { num_threads: usize }
pub struct MemoryEfficientSpMV { chunk_size: usize }
pub struct CacheOptimizedSpMV;

impl SpMVStrategy for StandardSpMV {
    fn multiply(&self, matrix: &CSRTensor, vector: &[f64]) -> Result<Vec<f64>> {
        matrix.matvec(vector)
    }
    
    fn estimated_memory(&self, matrix: &CSRTensor) -> usize {
        matrix.memory_usage() + vector.len() * 8
    }
    
    fn estimated_time(&self, matrix: &CSRTensor) -> Duration {
        Duration::from_millis(matrix.nnz() / 1000000) // 1M ops per ms
    }
}

pub struct SpMVContext {
    strategy: Box<dyn SpMVStrategy>,
}

impl SpMVContext {
    pub fn new(strategy: Box<dyn SpMVStrategy>) -> Self {
        Self { strategy }
    }
    
    pub fn auto_select_strategy(matrix: &CSRTensor, 
                               constraints: &PerformanceConstraints) -> Self {
        let strategies: Vec<Box<dyn SpMVStrategy>> = vec![
            Box::new(StandardSpMV),
            Box::new(ParallelSpMV { num_threads: 8 }),
            Box::new(MemoryEfficientSpMV { chunk_size: 1000 }),
            Box::new(CacheOptimizedSpMV),
        ];
        
        let best_strategy = strategies
            .into_iter()
            .min_by_key(|strategy| {
                let memory = strategy.estimated_memory(matrix);
                let time = strategy.estimated_time(matrix);
                
                if memory > constraints.max_memory {
                    return usize::MAX; // Invalid strategy
                }
                
                time.as_millis() as usize
            })
            .unwrap();
        
        Self { strategy: best_strategy }
    }
    
    pub fn execute(&self, matrix: &CSRTensor, vector: &[f64]) -> Result<Vec<f64>> {
        self.strategy.multiply(matrix, vector)
    }
}
```

### 3. Observer Pattern for Progress Tracking

```rust
pub trait ProgressObserver {
    fn on_operation_start(&self, operation: &str, total_work: usize);
    fn on_progress(&self, completed_work: usize);
    fn on_operation_complete(&self, operation: &str, duration: Duration);
}

pub struct ConsoleProgressObserver;

impl ProgressObserver for ConsoleProgressObserver {
    fn on_operation_start(&self, operation: &str, total_work: usize) {
        println!("Starting {}: {} items to process", operation, total_work);
    }
    
    fn on_progress(&self, completed_work: usize) {
        print!("\rProgress: {}", completed_work);
        std::io::stdout().flush().unwrap();
    }
    
    fn on_operation_complete(&self, operation: &str, duration: Duration) {
        println!("\n{} completed in {:?}", operation, duration);
    }
}

pub struct ProgressTracker {
    observers: Vec<Box<dyn ProgressObserver>>,
}

impl ProgressTracker {
    pub fn new() -> Self {
        Self { observers: Vec::new() }
    }
    
    pub fn add_observer(&mut self, observer: Box<dyn ProgressObserver>) {
        self.observers.push(observer);
    }
    
    pub fn start_operation(&self, operation: &str, total_work: usize) {
        for observer in &self.observers {
            observer.on_operation_start(operation, total_work);
        }
    }
    
    pub fn report_progress(&self, completed_work: usize) {
        for observer in &self.observers {
            observer.on_progress(completed_work);
        }
    }
    
    pub fn complete_operation(&self, operation: &str, duration: Duration) {
        for observer in &self.observers {
            observer.on_operation_complete(operation, duration);
        }
    }
}
```

## Optimization Checklist

### Pre-Deployment Checklist

- [ ] **Format Selection**: Verified optimal sparse format for each use case
- [ ] **Memory Management**: Implemented memory pools and garbage collection
- [ ] **Error Handling**: Comprehensive error handling with graceful degradation
- [ ] **Performance Testing**: Benchmarked critical paths and identified bottlenecks
- [ ] **Memory Testing**: Verified no memory leaks in long-running scenarios
- [ ] **Numerical Stability**: Validated numerical algorithms for edge cases
- [ ] **Configuration**: Environment-specific configuration management
- [ ] **Monitoring**: Implemented metrics collection and alerting
- [ ] **Documentation**: API documentation and usage examples
- [ ] **Testing**: Unit tests, integration tests, and property-based tests

### Performance Optimization Checklist

- [ ] **SIMD**: Enabled SIMD optimizations where available
- [ ] **Parallelization**: Used parallel algorithms for large matrices
- [ ] **Cache Optimization**: Optimized memory access patterns
- [ ] **Algorithm Selection**: Used optimal algorithms for each operation
- [ ] **Memory Allocation**: Minimized dynamic memory allocation
- [ ] **Format Conversion**: Minimized unnecessary format conversions
- [ ] **Batch Processing**: Batched operations to reduce overhead
- [ ] **Streaming**: Implemented streaming for large datasets
- [ ] **Profiling**: Regular performance profiling and optimization
- [ ] **Hardware Utilization**: Optimized for target hardware architecture

## Conclusion

Following these best practices will help you build robust, efficient, and maintainable sparse tensor applications with ToRSh-Sparse. Remember to:

1. **Profile before optimizing** - Identify actual bottlenecks
2. **Choose the right format** - Match format to access patterns
3. **Manage memory carefully** - Use pools and monitor usage
4. **Handle errors gracefully** - Plan for failure scenarios
5. **Test comprehensively** - Include performance and stress tests
6. **Monitor in production** - Track metrics and performance
7. **Document thoroughly** - Make code maintainable for others

The sparse tensor domain has many subtleties, and these practices will help you navigate them successfully while building high-performance applications.

For more detailed information, see the [Sparse Guide](SPARSE_GUIDE.md), [Format Reference](FORMAT_REFERENCE.md), and [Performance Guide](PERFORMANCE_GUIDE.md).