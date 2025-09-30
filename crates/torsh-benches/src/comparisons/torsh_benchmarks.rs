//! ToRSh benchmark implementations
//!
//! This module contains the core ToRSh benchmark implementations used for
//! performance comparisons with other tensor libraries.

use crate::Benchmarkable;

/// ToRSh matrix multiplication benchmark
pub struct TorshMatmulBench;

impl Benchmarkable for TorshMatmulBench {
    type Input = (torsh_tensor::Tensor<f32>, torsh_tensor::Tensor<f32>);
    type Output = Result<torsh_tensor::Tensor<f32>, torsh_core::error::TorshError>;

    fn setup(&mut self, size: usize) -> Self::Input {
        let a = torsh_tensor::creation::rand::<f32>(&[size, size]).unwrap();
        let b = torsh_tensor::creation::rand::<f32>(&[size, size]).unwrap();
        (a, b)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        input.0.matmul(&input.1)
    }

    fn flops(&self, size: usize) -> usize {
        2 * size * size * size
    }
}

/// Element-wise operations comparison
pub struct TorshElementwiseBench;

impl Benchmarkable for TorshElementwiseBench {
    type Input = (torsh_tensor::Tensor<f32>, torsh_tensor::Tensor<f32>);
    type Output = Result<torsh_tensor::Tensor<f32>, torsh_core::error::TorshError>;

    fn setup(&mut self, size: usize) -> Self::Input {
        let a = torsh_tensor::creation::rand::<f32>(&[size]).unwrap();
        let b = torsh_tensor::creation::rand::<f32>(&[size]).unwrap();
        (a, b)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        input.0.add(&input.1)
    }

    fn flops(&self, size: usize) -> usize {
        size
    }
}