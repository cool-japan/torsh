//! NDArray comparison benchmarks
//!
//! This module provides benchmark implementations for comparing ToRSh
//! performance with NDArray operations.

use crate::Benchmarkable;
use std::hint::black_box;

#[cfg(feature = "compare-external")]
use scirs2_autograd::ndarray::{Array, Array2};
#[cfg(feature = "compare-external")]
use scirs2_core::random::Random;

/// NDArray matrix multiplication benchmark (for comparison)
#[cfg(feature = "compare-external")]
pub struct NdarrayMatmulBench;

#[cfg(feature = "compare-external")]
impl Benchmarkable for NdarrayMatmulBench {
    type Input = (Array2<f32>, Array2<f32>);
    type Output = Array2<f32>;

    fn setup(&mut self, size: usize) -> Self::Input {
        use scirs2_autograd::ndarray::Array;
        use scirs2_core::random::Random;
        let mut rng = Random::default();
        let a = Array::random((size, size), &mut rng);
        let b = Array::random((size, size), &mut rng);
        (a, b)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        black_box(input.0.dot(&input.1))
    }

    fn flops(&self, size: usize) -> usize {
        2 * size * size * size
    }
}

#[cfg(feature = "compare-external")]
pub struct NdarrayElementwiseBench;

#[cfg(feature = "compare-external")]
impl Benchmarkable for NdarrayElementwiseBench {
    type Input = (
        Array<f32, ndarray::Dim<[usize; 1]>>,
        Array<f32, ndarray::Dim<[usize; 1]>>,
    );
    type Output = Array<f32, ndarray::Dim<[usize; 1]>>;

    fn setup(&mut self, size: usize) -> Self::Input {
        use scirs2_autograd::ndarray::Array;
        use scirs2_core::random::Random;
        let mut rng = Random::default();
        let a = Array::random(size, &mut rng);
        let b = Array::random(size, &mut rng);
        (a, b)
    }

    fn run(&mut self, input: &Self::Input) -> Self::Output {
        black_box(&input.0 + &input.1)
    }

    fn flops(&self, size: usize) -> usize {
        size
    }
}