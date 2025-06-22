use approx::assert_relative_eq;
use torsh_tensor::creation::*;
use torsh_tensor::prelude::*;

#[test]
fn test_tensor_creation() {
    // Test scalar tensor
    let scalar = tensor![5.0f32];
    assert_eq!(scalar.shape().dims(), &[] as &[usize]);
    assert_eq!(scalar.numel(), 1);
    assert_eq!(scalar.item(), 5.0f32);

    // Test 1D tensor
    let vec1d = tensor![1.0f32, 2.0f32, 3.0f32];
    assert_eq!(vec1d.shape().dims(), &[3]);
    assert_eq!(vec1d.numel(), 3);

    // Test 2D tensor
    let mat2d = tensor_2d![[1.0f32, 2.0f32], [3.0f32, 4.0f32]];
    assert_eq!(mat2d.shape().dims(), &[2, 2]);
    assert_eq!(mat2d.numel(), 4);
}

#[test]
fn test_zeros_ones() {
    let z = zeros::<f32>(&[3, 4]);
    assert_eq!(z.shape().dims(), &[3, 4]);
    assert_eq!(z.numel(), 12);

    let o = ones::<f32>(&[2, 2]);
    assert_eq!(o.shape().dims(), &[2, 2]);
    assert_eq!(o.numel(), 4);

    let e = eye::<f32>(3);
    assert_eq!(e.shape().dims(), &[3, 3]);
}

#[test]
fn test_basic_operations() {
    let a = tensor![1.0, 2.0, 3.0];
    let b = tensor![4.0, 5.0, 6.0];

    // Addition
    let c = a.add(&b).unwrap();
    let expected = vec![5.0, 7.0, 9.0];
    assert_eq!(c.to_vec(), expected);

    // Subtraction
    let d = b.sub(&a).unwrap();
    let expected = vec![3.0, 3.0, 3.0];
    assert_eq!(d.to_vec(), expected);

    // Multiplication
    let e = a.mul(&b).unwrap();
    let expected = vec![4.0, 10.0, 18.0];
    assert_eq!(e.to_vec(), expected);

    // Division
    let f = b.div(&a).unwrap();
    assert_relative_eq!(f.to_vec()[0], 4.0, epsilon = 1e-6);
    assert_relative_eq!(f.to_vec()[1], 2.5, epsilon = 1e-6);
    assert_relative_eq!(f.to_vec()[2], 2.0, epsilon = 1e-6);
}

#[test]
fn test_matrix_multiplication() {
    let a = tensor_2d![[1.0, 2.0], [3.0, 4.0]];
    let b = tensor_2d![[5.0, 6.0], [7.0, 8.0]];

    let c = a.matmul(&b).unwrap();
    // Expected: [[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]]
    //         = [[19, 22], [43, 50]]
    let expected = vec![19.0, 22.0, 43.0, 50.0];
    assert_eq!(c.to_vec(), expected);
}

#[test]
fn test_transpose() {
    let a = tensor_2d![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let t = a.t().unwrap();
    assert_eq!(t.shape().dims(), &[3, 2]);
    let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    assert_eq!(t.to_vec(), expected);
}

#[test]
fn test_reductions() {
    let a = tensor_2d![[1.0, 2.0], [3.0, 4.0]];

    // Sum
    let sum = a.sum().unwrap();
    assert_eq!(sum.item(), 10.0);

    // Mean
    let mean = a.mean().unwrap();
    assert_eq!(mean.item(), 2.5);

    // Max
    let max = a.max().unwrap();
    assert_eq!(max.item(), 4.0);

    // Min
    let min = a.min().unwrap();
    assert_eq!(min.item(), 1.0);
}

#[test]
fn test_activations() {
    let a = tensor_2d![[-1.0, 0.0, 1.0, 2.0]];

    // ReLU
    let relu = a.relu().unwrap();
    let expected = vec![0.0, 0.0, 1.0, 2.0];
    assert_eq!(relu.to_vec(), expected);

    // Sigmoid
    let sigmoid = a.sigmoid().unwrap();
    assert_relative_eq!(sigmoid.to_vec()[1], 0.5, epsilon = 1e-6);

    // Tanh
    let tanh = a.tanh().unwrap();
    assert_relative_eq!(tanh.to_vec()[1], 0.0, epsilon = 1e-6);
}

#[test]
fn test_broadcasting() {
    let a = ones::<f32>(&[3, 1]);
    let b = ones::<f32>(&[1, 4]);

    let c = a.add(&b).unwrap();
    assert_eq!(c.shape().dims(), &[3, 4]);
    assert_eq!(c.to_vec(), vec![2.0; 12]);
}

#[test]
fn test_gradient_tracking() {
    let x = tensor![2.0].requires_grad_(true);
    assert!(x.requires_grad());

    let y = x.pow(2.0).unwrap();
    assert!(y.requires_grad());

    // Backward pass
    y.backward().unwrap();

    // Check gradient
    let grad = x.grad().unwrap();
    assert_eq!(grad.item(), 4.0); // dy/dx = 2x = 4
}

#[test]
fn test_random_tensors() {
    let r = rand::<f32>(&[3, 3]);
    assert_eq!(r.shape().dims(), &[3, 3]);

    // Check values are in [0, 1)
    for val in r.to_vec() {
        assert!(val >= 0.0 && val < 1.0);
    }

    let n = randn::<f32>(&[2, 2]);
    assert_eq!(n.shape().dims(), &[2, 2]);
}

#[test]
fn test_arange_linspace() {
    let a = arange(0.0f32, 5.0, 1.0).unwrap();
    assert_eq!(a.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);

    let l = linspace(0.0f32, 1.0, 5);
    let expected = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    for (actual, expected) in l.to_vec().iter().zip(expected.iter()) {
        assert_relative_eq!(actual, expected, epsilon = 1e-6);
    }
}

#[test]
fn test_scalar_operations() {
    let a = tensor![1.0, 2.0, 3.0];

    let b = a.add_scalar(5.0).unwrap();
    assert_eq!(b.to_vec(), vec![6.0, 7.0, 8.0]);

    let c = a.mul_scalar(2.0).unwrap();
    assert_eq!(c.to_vec(), vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_shape_errors() {
    // Create tensors with truly incompatible shapes for broadcasting
    let a = tensor_2d![[1.0, 2.0, 3.0]]; // shape [1, 3]
    let b = tensor_2d![[4.0, 5.0], [6.0, 7.0]]; // shape [2, 2]

    // Incompatible shapes for element-wise ops (3 vs 2 in last dimension)
    assert!(a.add(&b).is_err());

    // Incompatible shapes for matmul (3 vs 2 for inner dimensions)
    let c = tensor_2d![[1.0, 2.0], [3.0, 4.0]]; // shape [2, 2]
    assert!(a.matmul(&c).is_err()); // [1,3] @ [2,2] should fail
}
