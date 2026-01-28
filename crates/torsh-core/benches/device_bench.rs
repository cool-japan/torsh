use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;
use torsh_core::device::{CpuDevice, Device, DeviceType};

fn device_creation_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("device_creation");

    // Benchmark device creation
    group.bench_function("create_cpu_device", |b| b.iter(CpuDevice::new));

    group.bench_function("create_cuda_device_type", |b| {
        b.iter(|| DeviceType::Cuda(0))
    });

    group.bench_function("create_metal_device_type", |b| {
        b.iter(|| DeviceType::Metal(0))
    });

    group.bench_function("create_webgpu_device_type", |b| {
        b.iter(|| DeviceType::Wgpu(0))
    });

    group.finish();
}

fn device_comparison_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("device_comparison");

    let cpu1 = CpuDevice::new();
    let cpu2 = CpuDevice::new();
    let _cuda1 = CpuDevice::new(); // Placeholder for CUDA device
    let _cuda2 = CpuDevice::new(); // Placeholder for CUDA device

    // Benchmark device type equality
    group.bench_function("equal_cpu_device_types", |b| {
        b.iter(|| cpu1.device_type() == cpu2.device_type())
    });

    group.bench_function("different_device_types", |b| {
        b.iter(|| cpu1.device_type() == DeviceType::Cuda(0))
    });

    group.bench_function("device_type_comparison", |b| {
        b.iter(|| std::hint::black_box(DeviceType::Cpu) == std::hint::black_box(DeviceType::Cpu))
    });

    // Benchmark device type hashing
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    group.bench_function("hash_cpu_device_type", |b| {
        b.iter(|| {
            let mut hasher = DefaultHasher::new();
            cpu1.device_type().hash(&mut hasher);
            hasher.finish()
        })
    });

    group.bench_function("hash_cuda_device_type", |b| {
        b.iter(|| {
            let mut hasher = DefaultHasher::new();
            DeviceType::Cuda(0).hash(&mut hasher);
            hasher.finish()
        })
    });

    group.finish();
}

fn device_capabilities_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("device_capabilities");
    group.measurement_time(Duration::from_secs(5));

    let cpu = CpuDevice::new();
    let cuda = CpuDevice::new(); // Placeholder for CUDA device

    // Benchmark capability queries
    group.bench_function("cpu_capabilities", |b| b.iter(|| cpu.capabilities()));

    group.bench_function("cuda_capabilities", |b| b.iter(|| cuda.capabilities()));

    // Benchmark specific capability checks
    group.bench_function("check_device_name", |b| b.iter(|| cpu.name()));

    group.bench_function("check_device_availability", |b| {
        b.iter(|| cpu.is_available())
    });

    group.bench_function("check_device_synchronize", |b| b.iter(|| cpu.synchronize()));

    group.finish();
}

fn device_type_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("device_type");

    // Benchmark device type checks
    let cpu = CpuDevice::new();
    let cuda = CpuDevice::new(); // Placeholder for CUDA device

    group.bench_function("is_cpu", |b| {
        b.iter(|| cpu.device_type() == DeviceType::Cpu)
    });

    group.bench_function("is_cuda", |b| {
        b.iter(|| cuda.device_type() == DeviceType::Cuda(0))
    });

    // Benchmark device type creation
    group.bench_function("create_device_type_cpu", |b| b.iter(|| DeviceType::Cpu));

    group.bench_function("create_device_type_cuda", |b| {
        b.iter(|| DeviceType::Cuda(0))
    });

    group.finish();
}

fn device_availability_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("device_availability");
    group.measurement_time(Duration::from_secs(3));

    // Benchmark device availability checks (simplified for CPU only)
    group.bench_function("cpu_available", |b| b.iter(|| true));

    group.bench_function("device_type_match", |b| {
        b.iter(|| matches!(DeviceType::Cpu, DeviceType::Cpu))
    });

    group.finish();
}

fn device_memory_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("device_memory");

    let devices = vec![
        CpuDevice::new(),
        CpuDevice::new(), // Placeholder for different device
    ];

    for (i, device) in devices.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("memory_info", i), device, |b, device| {
            b.iter(|| device.memory_info())
        });

        group.bench_with_input(BenchmarkId::new("device_name", i), device, |b, device| {
            b.iter(|| device.name())
        });

        group.bench_with_input(
            BenchmarkId::new("device_synchronize", i),
            device,
            |b, device| b.iter(|| device.synchronize()),
        );
    }

    group.finish();
}

fn device_string_operations_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("device_string_ops");

    let devices = vec![
        CpuDevice::new(),
        CpuDevice::new(), // Placeholder for CUDA device
        CpuDevice::new(), // Placeholder for Metal device
    ];

    // Benchmark string formatting
    for (i, device) in devices.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("device_debug_format", i),
            device,
            |b, device| b.iter(|| format!("{:?}", device)),
        );

        group.bench_with_input(
            BenchmarkId::new("device_name_string", i),
            device,
            |b, device| b.iter(|| device.name().to_string()),
        );
    }

    // Benchmark device type string representation
    group.bench_function("device_type_to_string", |b| {
        b.iter(|| DeviceType::Cpu.to_string())
    });

    group.bench_function("cuda_device_type_to_string", |b| {
        b.iter(|| DeviceType::Cuda(0).to_string())
    });

    group.finish();
}

criterion_group!(
    benches,
    device_creation_benchmarks,
    device_comparison_benchmarks,
    device_capabilities_benchmarks,
    device_type_benchmarks,
    device_availability_benchmarks,
    device_memory_benchmarks,
    device_string_operations_benchmarks
);
criterion_main!(benches);
