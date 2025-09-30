# Arrow 55.x Migration Plan for torsh-data

## Overview

This document outlines the migration strategy for upgrading from Apache Arrow 50.0 to Arrow 55.x in the torsh-data crate. The migration aims to leverage performance improvements, new features, and enhanced stability while maintaining backward compatibility where possible.

## Current State Analysis

### Current Arrow Dependencies
```toml
# Current Cargo.toml
[dependencies]
arrow = { version = "50.0", optional = true }
parquet = { version = "50.0", optional = true }
```

### Issues with Current Version
1. **Version Conflicts**: Ecosystem packages may require newer Arrow versions
2. **Missing Features**: Arrow 55.x includes performance improvements and new APIs
3. **Security Updates**: Newer versions include important security fixes
4. **Deprecation Warnings**: Some APIs used may be deprecated in current version

## Target Version Analysis

### Arrow 55.x New Features
1. **Improved Compute Kernels**: Faster arithmetic and comparison operations
2. **Enhanced Schema Evolution**: Better support for schema changes
3. **New Data Types**: Additional primitive and complex types
4. **Performance Optimizations**: Memory usage and CPU performance improvements
5. **Better Error Handling**: More descriptive error messages and recovery

### Breaking Changes in 55.x
1. **API Changes**: Some method signatures have changed
2. **Trait Reorganization**: Trait bounds and implementations updated
3. **Memory Layout**: Internal memory representation optimizations
4. **Deprecation Removals**: Previously deprecated APIs removed

## Migration Strategy

### Phase 1: Dependency Analysis and Preparation

#### 1.1 Dependency Audit
```bash
# Check current dependency tree
cargo tree | grep arrow
cargo tree | grep parquet

# Identify conflicting versions
cargo tree -d | grep -E "(arrow|parquet)"
```

#### 1.2 Feature Compatibility Matrix
| Feature | Arrow 50.0 | Arrow 55.x | Migration Required |
|---------|------------|------------|-------------------|
| Array creation | ✓ | ✓ (improved) | Minor API changes |
| Schema handling | ✓ | ✓ (enhanced) | Method signature updates |
| Compute operations | ✓ | ✓ (faster) | No changes |
| I/O operations | ✓ | ✓ (optimized) | Error handling updates |
| Memory management | ✓ | ✓ (improved) | Memory pool API changes |

#### 1.3 Risk Assessment
- **Low Risk**: Basic array operations, schema creation
- **Medium Risk**: Complex compute operations, custom kernels
- **High Risk**: Direct memory manipulation, unsafe code

### Phase 2: Gradual Migration

#### 2.1 Update Dependencies
```toml
# Updated Cargo.toml
[dependencies]
arrow = { version = "55", optional = true, features = ["compute", "io"] }
parquet = { version = "55", optional = true }

# Pin to specific version initially
# arrow = { version = "=55.0.0", optional = true }
```

#### 2.2 Conditional Compilation
```rust
// Maintain compatibility during transition
#[cfg(feature = "arrow-50")]
mod arrow_50_impl {
    use arrow_50 as arrow;
    // Implementation for Arrow 50.x
}

#[cfg(feature = "arrow-55")]
mod arrow_55_impl {
    use arrow_55 as arrow;
    // Implementation for Arrow 55.x
}
```

#### 2.3 API Adaptation Layer
```rust
// Create compatibility layer for API changes
pub trait ArrowCompatibility {
    type ArrayType;
    type SchemaType;
    
    fn create_array(&self, data: Vec<i32>) -> Result<Self::ArrayType>;
    fn create_schema(&self, fields: Vec<Field>) -> Self::SchemaType;
}

#[cfg(feature = "arrow-55")]
impl ArrowCompatibility for Arrow55Impl {
    type ArrayType = arrow::array::Int32Array;
    type SchemaType = arrow::datatypes::Schema;
    
    fn create_array(&self, data: Vec<i32>) -> Result<Self::ArrayType> {
        // Updated API for Arrow 55.x
        arrow::array::Int32Array::from(data)
    }
}
```

### Phase 3: Code Updates

#### 3.1 Array Creation Updates
```rust
// Old Arrow 50.x approach
fn create_int_array_old(data: Vec<i32>) -> Result<Int32Array> {
    let array_data = ArrayData::builder(DataType::Int32)
        .len(data.len())
        .add_buffer(Buffer::from_slice_ref(&data))
        .build()?;
    Ok(Int32Array::from(array_data))
}

// New Arrow 55.x approach
fn create_int_array_new(data: Vec<i32>) -> Result<Int32Array> {
    // Simplified API in 55.x
    Ok(Int32Array::from(data))
}
```

#### 3.2 Schema Handling Updates
```rust
// Updated schema creation for Arrow 55.x
fn create_enhanced_schema() -> Result<Schema> {
    let fields = vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
        Field::new("values", DataType::List(
            Arc::new(Field::new("item", DataType::Float64, true))
        ), true),
    ];
    
    // New metadata handling in 55.x
    let metadata = HashMap::from([
        ("version".to_string(), "2.0".to_string()),
        ("created_by".to_string(), "torsh-data".to_string()),
    ]);
    
    Ok(Schema::new_with_metadata(fields, metadata))
}
```

#### 3.3 Error Handling Updates
```rust
// Enhanced error handling in Arrow 55.x
fn handle_arrow_errors() -> Result<(), TorshError> {
    match arrow_operation() {
        Ok(result) => Ok(result),
        Err(arrow::error::ArrowError::ComputeError(msg)) => {
            Err(TorshError::ComputationError(format!("Arrow compute error: {}", msg)))
        },
        Err(arrow::error::ArrowError::IoError(source)) => {
            Err(TorshError::IoError(format!("Arrow I/O error: {}", source)))
        },
        Err(e) => {
            Err(TorshError::ArrowError(e.to_string()))
        }
    }
}
```

### Phase 4: Performance Optimization

#### 4.1 Memory Pool Utilization
```rust
// Leverage improved memory pools in Arrow 55.x
use arrow::memory_pool::{MemoryPool, TrackingMemoryPool};

pub struct OptimizedArrowDataset {
    memory_pool: Arc<dyn MemoryPool>,
    data: Vec<RecordBatch>,
}

impl OptimizedArrowDataset {
    pub fn new_with_pool(pool: Arc<dyn MemoryPool>) -> Self {
        Self {
            memory_pool: pool,
            data: Vec::new(),
        }
    }
    
    pub fn memory_usage(&self) -> usize {
        if let Some(tracking_pool) = self.memory_pool.as_any()
            .downcast_ref::<TrackingMemoryPool>() {
            tracking_pool.bytes_allocated()
        } else {
            0
        }
    }
}
```

#### 4.2 Compute Kernel Optimization
```rust
// Use enhanced compute kernels in Arrow 55.x
use arrow::compute::{kernels, KernelContext};

fn optimized_arithmetic_operations(
    left: &dyn Array,
    right: &dyn Array,
) -> Result<ArrayRef> {
    let ctx = KernelContext::default();
    
    // Enhanced error handling and performance in 55.x
    match (left.data_type(), right.data_type()) {
        (DataType::Int32, DataType::Int32) => {
            let left = left.as_any().downcast_ref::<Int32Array>().unwrap();
            let right = right.as_any().downcast_ref::<Int32Array>().unwrap();
            Ok(Arc::new(kernels::arithmetic::add_wrapping(&ctx, left, right)?))
        },
        (DataType::Float64, DataType::Float64) => {
            let left = left.as_any().downcast_ref::<Float64Array>().unwrap();
            let right = right.as_any().downcast_ref::<Float64Array>().unwrap();
            Ok(Arc::new(kernels::arithmetic::add(&ctx, left, right)?))
        },
        _ => Err(ArrowError::ComputeError("Unsupported types".to_string())),
    }
}
```

### Phase 5: Testing and Validation

#### 5.1 Compatibility Testing
```rust
#[cfg(test)]
mod arrow_migration_tests {
    use super::*;
    
    #[test]
    fn test_array_creation_compatibility() {
        let data = vec![1, 2, 3, 4, 5];
        
        #[cfg(feature = "arrow-50")]
        let array_50 = create_int_array_old(data.clone()).unwrap();
        
        #[cfg(feature = "arrow-55")]
        let array_55 = create_int_array_new(data.clone()).unwrap();
        
        // Verify same behavior
        assert_eq!(array_55.len(), data.len());
        for (i, &expected) in data.iter().enumerate() {
            assert_eq!(array_55.value(i), expected);
        }
    }
    
    #[test]
    fn test_performance_regression() {
        let start = std::time::Instant::now();
        
        // Perform standard operations
        let _result = create_large_dataset();
        
        let duration = start.elapsed();
        
        // Ensure no significant performance regression
        assert!(duration.as_millis() < 1000, "Performance regression detected");
    }
}
```

#### 5.2 Integration Testing
```rust
#[cfg(test)]
mod integration_tests {
    #[test]
    fn test_parquet_roundtrip() {
        // Test full roundtrip with new Arrow version
        let schema = create_test_schema();
        let batch = create_test_batch(&schema);
        
        // Write to Parquet
        let buffer = write_parquet_buffer(&batch).unwrap();
        
        // Read back from Parquet
        let read_batch = read_parquet_buffer(&buffer).unwrap();
        
        // Verify data integrity
        assert_batches_eq(&[batch], &[read_batch]);
    }
    
    #[test]
    fn test_arrow_dataset_integration() {
        let dataset = ArrowDataset::from_batches(create_test_batches()).unwrap();
        
        // Test all dataset operations
        assert_eq!(dataset.len(), 1000);
        
        let sample = dataset.get(0).unwrap();
        assert!(sample.is_ok());
        
        // Test iteration
        let mut count = 0;
        for item in dataset.iter() {
            assert!(item.is_ok());
            count += 1;
        }
        assert_eq!(count, 1000);
    }
}
```

### Phase 6: Documentation and Migration Guide

#### 6.1 Breaking Changes Documentation
```markdown
## Breaking Changes in Arrow 55.x Migration

### API Changes
1. **Array Creation**: Simplified constructors available
2. **Error Types**: New error variants for better debugging
3. **Memory Management**: Updated memory pool interfaces

### Migration Steps
1. Update Cargo.toml dependencies
2. Update import statements
3. Adapt error handling patterns
4. Test thoroughly before deployment
```

#### 6.2 Performance Benchmarks
```rust
// Benchmark migration performance impact
#[cfg(test)]
mod benchmarks {
    use criterion::{black_box, criterion_group, criterion_main, Criterion};
    
    fn benchmark_array_operations(c: &mut Criterion) {
        c.bench_function("arrow_55_array_creation", |b| {
            b.iter(|| {
                let data: Vec<i32> = (0..1000).collect();
                black_box(create_int_array_new(data))
            })
        });
        
        c.bench_function("arrow_55_compute_operations", |b| {
            let left = create_test_array();
            let right = create_test_array();
            b.iter(|| {
                black_box(optimized_arithmetic_operations(&left, &right))
            })
        });
    }
    
    criterion_group!(benches, benchmark_array_operations);
    criterion_main!(benches);
}
```

## Risk Mitigation

### 1. Gradual Rollout
- Enable Arrow 55.x via feature flag initially
- Test in development environments first
- Monitor performance metrics closely
- Provide rollback plan if issues arise

### 2. Compatibility Layer
- Maintain abstraction over Arrow APIs
- Provide migration utilities for users
- Document all breaking changes clearly
- Offer support during transition period

### 3. Testing Strategy
- Comprehensive unit test coverage
- Integration testing with real datasets
- Performance regression testing
- Cross-platform compatibility validation

## Timeline and Milestones

### Week 1-2: Preparation
- [ ] Dependency analysis
- [ ] Breaking changes assessment
- [ ] Test environment setup
- [ ] Migration plan finalization

### Week 3-4: Core Migration
- [ ] Update dependencies
- [ ] Implement compatibility layer
- [ ] Update core Arrow operations
- [ ] Fix compilation errors

### Week 5-6: Testing and Optimization
- [ ] Run comprehensive test suite
- [ ] Performance benchmarking
- [ ] Fix any regressions
- [ ] Documentation updates

### Week 7-8: Validation and Release
- [ ] Final testing in production-like environment
- [ ] User acceptance testing
- [ ] Release preparation
- [ ] Migration guide publication

## Success Criteria

1. **Functionality**: All existing features work with Arrow 55.x
2. **Performance**: No significant performance regression (< 5% impact)
3. **Compatibility**: Smooth migration path for existing users
4. **Stability**: No new crashes or memory issues
5. **Documentation**: Complete migration guide and updated docs

## Rollback Plan

If critical issues are discovered during migration:

1. **Immediate**: Revert to Arrow 50.x via feature flag
2. **Short-term**: Fix critical issues in isolated environment
3. **Long-term**: Gradual re-introduction of Arrow 55.x with fixes

## Conclusion

The migration to Arrow 55.x will provide significant benefits in terms of performance, features, and ecosystem compatibility. The proposed gradual migration strategy minimizes risk while ensuring a smooth transition for all users of torsh-data.