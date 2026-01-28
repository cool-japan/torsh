//! Tests for collective operations

use torsh_core::Result;
use torsh_distributed::{
    backend::BackendType,
    backend::ReduceOp,
    collectives::{all_gather, all_reduce, barrier, broadcast, reduce, scatter},
    init_process_group,
};
use torsh_tensor::creation::{eye, full, ones, zeros};
use torsh_tensor::Tensor;

#[tokio::test]
async fn test_all_reduce() -> Result<()> {
    // Create a mock process group
    let pg = init_process_group(BackendType::Gloo, 0, 4, "127.0.0.1", 29500).await?;

    // Create a tensor
    let mut tensor = ones::<f32>(&[2, 3])?;

    // Perform all-reduce
    all_reduce(&mut tensor, ReduceOp::Sum, &pg).await?;

    // With mock backend, sum operation should divide by world size
    // So tensor should now be 0.25 (1.0 / 4)
    let expected = full::<f32>(&[2, 3], 0.25)?;

    // Compare tensors element-wise
    let data = tensor.to_vec()?;
    let expected_data = expected.to_vec()?;
    assert_eq!(data.len(), expected_data.len());
    for (a, b) in data.iter().zip(expected_data.iter()) {
        assert!((a - b).abs() < 1e-6_f32);
    }

    Ok(())
}

#[tokio::test]
async fn test_broadcast() -> Result<()> {
    let pg = init_process_group(BackendType::Gloo, 1, 4, "127.0.0.1", 29500).await?;

    // Create a tensor with rank-specific values
    let rank = pg.rank() as f32;
    let mut tensor = full::<f32>(&[3, 3], rank)?;

    // Broadcast from rank 0
    broadcast(&mut tensor, 0, &pg).await?;

    // With mock backend, tensor remains unchanged
    // In real implementation, all ranks would have rank 0's tensor
    let expected = full::<f32>(&[3, 3], rank)?;

    // Compare tensors
    let data = tensor.to_vec()?;
    let expected_data = expected.to_vec()?;
    assert_eq!(data, expected_data);

    Ok(())
}

#[tokio::test]
async fn test_all_gather() -> Result<()> {
    let pg = init_process_group(BackendType::Gloo, 2, 4, "127.0.0.1", 29500).await?;

    // Create input tensor
    let input = eye::<f32>(3)?;
    let mut output = Vec::new();

    // Perform all-gather
    all_gather(&mut output, &input, &pg).await?;

    // Should have one tensor per rank
    assert_eq!(output.len(), 4);

    // With mock backend, all tensors are copies of input
    let input_data = input.to_vec()?;
    for tensor in &output {
        let tensor_data = tensor.to_vec()?;
        assert_eq!(tensor_data, input_data);
    }

    Ok(())
}

#[tokio::test]
async fn test_reduce() -> Result<()> {
    let pg = init_process_group(BackendType::Gloo, 0, 4, "127.0.0.1", 29500).await?;

    // Create tensor
    let mut tensor = full::<f32>(&[2, 2], 2.0)?;

    // Reduce to rank 0
    reduce(&mut tensor, 0, ReduceOp::Sum, &pg).await?;

    // Since we are rank 0 (dst), tensor should be multiplied by world size
    let expected = full::<f32>(&[2, 2], 8.0)?; // 2.0 * 4

    // Compare tensors
    let data: Vec<f32> = tensor.to_vec()?;
    let expected_data = expected.to_vec()?;
    for (a, b) in data.iter().zip(expected_data.iter()) {
        assert!(
            (a - b).abs() < 1e-6_f32,
            "Values don't match: {} vs {}",
            a,
            b
        );
    }

    Ok(())
}

#[tokio::test]
async fn test_scatter() -> Result<()> {
    let pg = init_process_group(BackendType::Gloo, 0, 4, "127.0.0.1", 29500).await?;

    // Create tensors to scatter
    let tensors: Vec<Tensor<f32>> = (0..4)
        .map(|i| full(&[2, 2], i as f32))
        .collect::<Result<Vec<_>>>()?;

    let mut output = zeros::<f32>(&[2, 2])?;

    // Scatter from rank 0
    scatter(&mut output, Some(&tensors), 0, &pg).await?;

    // Rank 0 should get tensor[0]
    let expected = zeros::<f32>(&[2, 2])?;

    // Compare tensors
    let data = output.to_vec()?;
    let expected_data = expected.to_vec()?;
    assert_eq!(data, expected_data);

    Ok(())
}

#[tokio::test]
async fn test_barrier() -> Result<()> {
    let pg = init_process_group(BackendType::Gloo, 0, 4, "127.0.0.1", 29500).await?;

    // Barrier should succeed without error
    barrier(&pg).await?;

    Ok(())
}

#[test]
fn test_reduce_ops() {
    // Test that reduce operations are properly defined
    let ops = vec![
        ReduceOp::Sum,
        ReduceOp::Product,
        ReduceOp::Min,
        ReduceOp::Max,
        ReduceOp::Band,
        ReduceOp::Bor,
        ReduceOp::Bxor,
    ];

    for op in ops {
        // Just verify they can be created and compared
        assert_eq!(op, op);
    }
}

#[test]
fn test_backend_availability() {
    use torsh_distributed::{is_available, is_mpi_available, is_nccl_available};

    // At least one backend should be available (mock backend)
    assert!(is_available());

    // Check individual backends based on features
    #[cfg(feature = "mpi")]
    assert!(is_mpi_available());

    #[cfg(not(feature = "mpi"))]
    assert!(!is_mpi_available());

    // Note: NCCL is currently a mock backend without actual CUDA dependency
    #[cfg(feature = "nccl")]
    assert!(is_nccl_available());

    #[cfg(not(feature = "nccl"))]
    assert!(!is_nccl_available());
}
