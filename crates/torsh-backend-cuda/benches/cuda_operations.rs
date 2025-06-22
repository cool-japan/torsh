use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use torsh_backend_cuda::{CudaBackend, CudaBackendConfig};
use torsh_core::DType;

fn benchmark_elementwise_operations(c: &mut Criterion) {
    // Only run benchmarks if CUDA is available
    if !torsh_backend_cuda::is_available() {
        return;
    }
    
    let rt = tokio::runtime::Runtime::new().unwrap();
    let config = CudaBackendConfig::default();
    let backend = rt.block_on(CudaBackend::initialize(config)).unwrap();
    
    let mut group = c.benchmark_group("cuda_elementwise");
    
    for size in [1024, 4096, 16384, 65536].iter() {
        group.bench_with_input(
            BenchmarkId::new("add_f32", size),
            size,
            |b, &size| {
                let mut a = backend.create_buffer::<f32>(size, DType::F32).unwrap();
                let mut b_buf = backend.create_buffer::<f32>(size, DType::F32).unwrap();
                let mut output = backend.create_buffer::<f32>(size, DType::F32).unwrap();
                
                // Initialize with test data
                let data_a = vec![1.0f32; size];
                let data_b = vec![2.0f32; size];
                a.copy_from_host(&data_a).unwrap();
                b_buf.copy_from_host(&data_b).unwrap();
                
                b.iter(|| {
                    backend.elementwise_add_f32(&a, &b_buf, &mut output, None).unwrap();
                    backend.synchronize().unwrap();
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_matrix_operations(c: &mut Criterion) {
    if !torsh_backend_cuda::is_available() {
        return;
    }
    
    let rt = tokio::runtime::Runtime::new().unwrap();
    let config = CudaBackendConfig::default();
    let backend = rt.block_on(CudaBackend::initialize(config)).unwrap();
    
    let mut group = c.benchmark_group("cuda_matrix");
    
    for size in [128, 256, 512, 1024].iter() {
        group.bench_with_input(
            BenchmarkId::new("matmul_f32", size),
            size,
            |b, &size| {
                let mut a = backend.create_buffer::<f32>(size * size, DType::F32).unwrap();
                let mut b_mat = backend.create_buffer::<f32>(size * size, DType::F32).unwrap();
                let mut output = backend.create_buffer::<f32>(size * size, DType::F32).unwrap();
                
                // Initialize with random-like data
                let data_a: Vec<f32> = (0..size*size).map(|i| (i as f32) * 0.001).collect();
                let data_b: Vec<f32> = (0..size*size).map(|i| (i as f32) * 0.002).collect();
                a.copy_from_host(&data_a).unwrap();
                b_mat.copy_from_host(&data_b).unwrap();
                
                b.iter(|| {
                    backend.matmul_f32(&a, &b_mat, &mut output, size, size, size, None).unwrap();
                    backend.synchronize().unwrap();
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_elementwise_operations, benchmark_matrix_operations);
criterion_main!(benches);