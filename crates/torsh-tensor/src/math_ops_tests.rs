use super::*;
use torsh_core::device::DeviceType;

#[test]
fn test_scalar_operations() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let tensor = Tensor::from_data(data, vec![4], DeviceType::Cpu)
        .expect("failed to create tensor for scalar ops");

    let result = tensor.add_scalar(5.0).expect("add_scalar should succeed");
    assert_eq!(
        result.data().expect("failed to get add_scalar result data"),
        vec![6.0, 7.0, 8.0, 9.0]
    );

    let result = tensor.mul_scalar(2.0).expect("mul_scalar should succeed");
    assert_eq!(
        result.data().expect("failed to get mul_scalar result data"),
        vec![2.0, 4.0, 6.0, 8.0]
    );

    let result = tensor.sub_scalar(1.0).expect("sub_scalar should succeed");
    assert_eq!(
        result.data().expect("failed to get sub_scalar result data"),
        vec![0.0, 1.0, 2.0, 3.0]
    );

    let result = tensor.div_scalar(2.0).expect("div_scalar should succeed");
    assert_eq!(
        result.data().expect("failed to get div_scalar result data"),
        vec![0.5, 1.0, 1.5, 2.0]
    );
}

#[test]
fn test_elementwise_operations() {
    let a = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)
        .expect("failed to create tensor a");
    let b = Tensor::from_data(vec![4.0f32, 5.0, 6.0], vec![3], DeviceType::Cpu)
        .expect("failed to create tensor b");

    let result = a.add(&b).expect("elementwise add should succeed");
    assert_eq!(
        result.data().expect("failed to get add result data"),
        vec![5.0, 7.0, 9.0]
    );

    let result = a.sub(&b).expect("elementwise sub should succeed");
    assert_eq!(
        result.data().expect("failed to get sub result data"),
        vec![-3.0, -3.0, -3.0]
    );

    let result = a.mul(&b).expect("elementwise mul should succeed");
    assert_eq!(
        result.data().expect("failed to get mul result data"),
        vec![4.0, 10.0, 18.0]
    );

    let result = b.div(&a).expect("elementwise div should succeed");
    assert_eq!(
        result.data().expect("failed to get div result data"),
        vec![4.0, 2.5, 2.0]
    );
}

#[test]
fn test_mathematical_functions() {
    let data = vec![1.0f32, 4.0, 9.0, 16.0];
    let tensor = Tensor::from_data(data, vec![4], DeviceType::Cpu)
        .expect("failed to create tensor for math functions");

    let sqrt_result = tensor.sqrt().expect("sqrt should succeed");
    assert_eq!(
        sqrt_result.data().expect("failed to get sqrt result data"),
        vec![1.0, 2.0, 3.0, 4.0]
    );

    let data2 = vec![0.0f32, 1.0, 2.0];
    let tensor2 = Tensor::from_data(data2, vec![3], DeviceType::Cpu)
        .expect("failed to create tensor2 for exp");

    let exp_result = tensor2.exp().expect("exp should succeed");
    let expected_exp = vec![1.0, std::f32::consts::E, std::f32::consts::E.powi(2)];
    for (got, &expected) in exp_result
        .data()
        .expect("failed to get exp result data")
        .iter()
        .zip(&expected_exp)
    {
        assert!((got - expected).abs() < 1e-6);
    }
}

#[test]
fn test_trigonometric_functions() {
    let data = vec![0.0f32, std::f32::consts::PI / 2.0, std::f32::consts::PI];
    let tensor = Tensor::from_data(data, vec![3], DeviceType::Cpu)
        .expect("failed to create tensor for trig functions");

    let sin_result = tensor.sin().expect("sin should succeed");
    let sin_data = sin_result.data().expect("failed to get sin result data");
    assert!((sin_data[0] - 0.0).abs() < 1e-6);
    assert!((sin_data[1] - 1.0).abs() < 1e-6);
    assert!((sin_data[2] - 0.0).abs() < 1e-6);

    let cos_result = tensor.cos().expect("cos should succeed");
    let cos_data = cos_result.data().expect("failed to get cos result data");
    assert!((cos_data[0] - 1.0).abs() < 1e-6);
    assert!((cos_data[1] - 0.0).abs() < 1e-6);
    assert!((cos_data[2] - (-1.0)).abs() < 1e-6);
}

#[test]
fn test_operator_overloads() {
    let a = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)
        .expect("failed to create tensor a for operator overloads");
    let b = Tensor::from_data(vec![4.0f32, 5.0, 6.0], vec![3], DeviceType::Cpu)
        .expect("failed to create tensor b for operator overloads");

    let result = &a + &b;
    assert_eq!(
        result.data().expect("failed to get add operator result"),
        vec![5.0, 7.0, 9.0]
    );

    let result = &b - &a;
    assert_eq!(
        result.data().expect("failed to get sub operator result"),
        vec![3.0, 3.0, 3.0]
    );

    let result = &a * &b;
    assert_eq!(
        result.data().expect("failed to get mul operator result"),
        vec![4.0, 10.0, 18.0]
    );

    let result = &b / &a;
    assert_eq!(
        result.data().expect("failed to get div operator result"),
        vec![4.0, 2.5, 2.0]
    );

    let neg_result = -&a;
    assert_eq!(
        neg_result
            .data()
            .expect("failed to get neg operator result"),
        vec![-1.0, -2.0, -3.0]
    );
}

#[test]
fn test_power_operations() {
    let data = vec![2.0f32, 3.0, 4.0];
    let tensor = Tensor::from_data(data, vec![3], DeviceType::Cpu)
        .expect("failed to create tensor for power ops");

    let pow_result = tensor.pow(2.0).expect("pow should succeed");
    assert_eq!(
        pow_result.data().expect("failed to get pow result data"),
        vec![4.0, 9.0, 16.0]
    );

    let exponents = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)
        .expect("failed to create exponents tensor");
    let pow_tensor_result = tensor
        .pow_tensor(&exponents)
        .expect("pow_tensor should succeed");
    assert_eq!(
        pow_tensor_result
            .data()
            .expect("failed to get pow_tensor result data"),
        vec![2.0, 9.0, 64.0]
    );
}

#[test]
fn test_rounding_functions() {
    let data = vec![1.2f32, 2.7, -1.5, -2.3];
    let tensor = Tensor::from_data(data, vec![4], DeviceType::Cpu)
        .expect("failed to create tensor for rounding");

    let floor_result = tensor.floor().expect("floor should succeed");
    assert_eq!(
        floor_result
            .data()
            .expect("failed to get floor result data"),
        vec![1.0, 2.0, -2.0, -3.0]
    );

    let ceil_result = tensor.ceil().expect("ceil should succeed");
    assert_eq!(
        ceil_result.data().expect("failed to get ceil result data"),
        vec![2.0, 3.0, -1.0, -2.0]
    );

    let round_result = tensor.round().expect("round should succeed");
    assert_eq!(
        round_result
            .data()
            .expect("failed to get round result data"),
        vec![1.0, 3.0, -2.0, -2.0]
    );
}

#[test]
fn test_sign_function() {
    let data = vec![-3.0f32, 0.0, 5.0, -1.0];
    let tensor = Tensor::from_data(data, vec![4], DeviceType::Cpu)
        .expect("failed to create tensor for sign");

    let sign_result = tensor.sign().expect("sign should succeed");
    assert_eq!(
        sign_result.data().expect("failed to get sign result data"),
        vec![-1.0, 0.0, 1.0, -1.0]
    );
}

#[test]
fn test_shape_mismatch_error() {
    let a = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu)
        .expect("failed to create tensor a for shape mismatch test");
    let b = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)
        .expect("failed to create tensor b for shape mismatch test");

    assert!(a.add(&b).is_err());
    assert!(a.mul(&b).is_err());
}

// In-place operation tests
#[test]
fn test_relu_inplace() {
    let mut tensor =
        Tensor::from_data(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor for relu inplace");

    tensor.relu_().expect("relu_ should succeed");
    let result = tensor.data().expect("failed to get relu_ result data");

    assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_sigmoid_inplace() {
    let mut tensor = Tensor::from_data(vec![0.0f32], vec![1], DeviceType::Cpu)
        .expect("failed to create tensor for sigmoid inplace");

    tensor.sigmoid_().expect("sigmoid_ should succeed");
    let result = tensor.data().expect("failed to get sigmoid_ result data");

    // sigmoid(0) = 0.5
    assert!((result[0] - 0.5).abs() < 1e-6);
}

#[test]
fn test_tanh_inplace() {
    let mut tensor = Tensor::from_data(vec![0.0f32], vec![1], DeviceType::Cpu)
        .expect("failed to create tensor for tanh inplace");

    tensor.tanh_().expect("tanh_ should succeed");
    let result = tensor.data().expect("failed to get tanh_ result data");

    // tanh(0) = 0
    assert!(result[0].abs() < 1e-6);
}

#[test]
fn test_clamp_inplace() {
    let mut tensor =
        Tensor::from_data(vec![-2.0f32, -1.0, 0.0, 1.0, 2.0], vec![5], DeviceType::Cpu)
            .expect("failed to create tensor for clamp inplace");

    tensor.clamp_(-1.0, 1.0).expect("clamp_ should succeed");
    let result = tensor.data().expect("failed to get clamp_ result data");

    assert_eq!(result, vec![-1.0, -1.0, 0.0, 1.0, 1.0]);
}

#[test]
fn test_inplace_with_requires_grad_error() {
    let mut tensor = Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu)
        .expect("failed to create tensor for requires_grad test");
    tensor.requires_grad = true;

    // In-place operations should fail on tensors with requires_grad=true
    assert!(tensor.relu_().is_err());
    assert!(tensor.sigmoid_().is_err());
    assert!(tensor.tanh_().is_err());
}

// Block B: out-of-place f32 SIMD fast-path tests

#[test]
fn test_f32_add_simd_fast_path() {
    // Tensor large enough to trigger SIMD path (>=1024 elements)
    let n = 4096;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
    let a = Tensor::<f32>::from_data(a_data.clone(), vec![n], DeviceType::Cpu)
        .expect("failed to create tensor a");
    let b = Tensor::<f32>::from_data(b_data.clone(), vec![n], DeviceType::Cpu)
        .expect("failed to create tensor b");
    let result = a.add(&b).expect("add should succeed");
    let result_data = result.data().expect("failed to get add result data");
    for (i, (&got, (&aa, &bb))) in result_data
        .iter()
        .zip(a_data.iter().zip(b_data.iter()))
        .enumerate()
    {
        assert!(
            (got - (aa + bb)).abs() < 1e-5,
            "mismatch at i={}: got={}, expected={}",
            i,
            got,
            aa + bb
        );
    }
}

#[test]
fn test_f32_sub_simd_fast_path() {
    let n = 4096;
    let a_data: Vec<f32> = (0..n).map(|i| (i * 3) as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let a = Tensor::<f32>::from_data(a_data.clone(), vec![n], DeviceType::Cpu)
        .expect("failed to create tensor a");
    let b = Tensor::<f32>::from_data(b_data.clone(), vec![n], DeviceType::Cpu)
        .expect("failed to create tensor b");
    let result = a.sub(&b).expect("sub should succeed");
    let result_data = result.data().expect("failed to get sub result data");
    for (i, (&got, (&aa, &bb))) in result_data
        .iter()
        .zip(a_data.iter().zip(b_data.iter()))
        .enumerate()
    {
        assert!(
            (got - (aa - bb)).abs() < 1e-5,
            "mismatch at i={}: got={}, expected={}",
            i,
            got,
            aa - bb
        );
    }
}

#[test]
fn test_f32_mul_simd_fast_path() {
    let n = 4096;
    let a_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 + 0.5).collect();
    let a = Tensor::<f32>::from_data(a_data.clone(), vec![n], DeviceType::Cpu)
        .expect("failed to create tensor a");
    let b = Tensor::<f32>::from_data(b_data.clone(), vec![n], DeviceType::Cpu)
        .expect("failed to create tensor b");
    let result = a.mul(&b).expect("mul should succeed");
    let result_data = result.data().expect("failed to get mul result data");
    for (i, (&got, (&aa, &bb))) in result_data
        .iter()
        .zip(a_data.iter().zip(b_data.iter()))
        .enumerate()
    {
        assert!(
            (got - aa * bb).abs() < 1e-4,
            "mismatch at i={}: got={}, expected={}",
            i,
            got,
            aa * bb
        );
    }
}

#[test]
fn test_f32_div_simd_fast_path() {
    let n = 4096;
    let a_data: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let a = Tensor::<f32>::from_data(a_data.clone(), vec![n], DeviceType::Cpu)
        .expect("failed to create tensor a");
    let b = Tensor::<f32>::from_data(b_data.clone(), vec![n], DeviceType::Cpu)
        .expect("failed to create tensor b");
    let result = a.div(&b).expect("div should succeed");
    let result_data = result.data().expect("failed to get div result data");
    for (i, (&got, (&aa, &bb))) in result_data
        .iter()
        .zip(a_data.iter().zip(b_data.iter()))
        .enumerate()
    {
        assert!(
            (got - aa / bb).abs() < 1e-4,
            "mismatch at i={}: got={}, expected={}",
            i,
            got,
            aa / bb
        );
    }
}

// Block C: in-place tensor×tensor f32 SIMD fast-path tests

#[test]
fn test_f32_add_inplace_simd() {
    let n = 4096;
    let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
    let ref_data: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(a, b)| a + b)
        .collect();
    let mut a = Tensor::<f32>::from_data(a_data, vec![n], DeviceType::Cpu)
        .expect("failed to create tensor a");
    let b = Tensor::<f32>::from_data(b_data, vec![n], DeviceType::Cpu)
        .expect("failed to create tensor b");
    a.add_(&b).expect("add_ should succeed");
    let result = a.data().expect("failed to get add_ result data");
    for (i, (&got, &exp)) in result.iter().zip(ref_data.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "mismatch at i={}: got={}, expected={}",
            i,
            got,
            exp
        );
    }
}

#[test]
fn test_f32_sub_inplace_simd() {
    let n = 4096;
    let a_data: Vec<f32> = (0..n).map(|i| (i * 3) as f32).collect();
    let b_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let ref_data: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(a, b)| a - b)
        .collect();
    let mut a = Tensor::<f32>::from_data(a_data, vec![n], DeviceType::Cpu)
        .expect("failed to create tensor a");
    let b = Tensor::<f32>::from_data(b_data, vec![n], DeviceType::Cpu)
        .expect("failed to create tensor b");
    a.sub_(&b).expect("sub_ should succeed");
    let result = a.data().expect("failed to get sub_ result data");
    for (i, (&got, &exp)) in result.iter().zip(ref_data.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "mismatch at i={}: got={}, expected={}",
            i,
            got,
            exp
        );
    }
}

#[test]
fn test_f32_mul_inplace_simd() {
    let n = 4096;
    let a_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 + 0.5).collect();
    let ref_data: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(a, b)| a * b)
        .collect();
    let mut a = Tensor::<f32>::from_data(a_data, vec![n], DeviceType::Cpu)
        .expect("failed to create tensor a");
    let b = Tensor::<f32>::from_data(b_data, vec![n], DeviceType::Cpu)
        .expect("failed to create tensor b");
    a.mul_(&b).expect("mul_ should succeed");
    let result = a.data().expect("failed to get mul_ result data");
    for (i, (&got, &exp)) in result.iter().zip(ref_data.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "mismatch at i={}: got={}, expected={}",
            i,
            got,
            exp
        );
    }
}

#[test]
fn test_f32_div_inplace_simd() {
    let n = 4096;
    let a_data: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
    let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 + 1.0).collect();
    let ref_data: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter())
        .map(|(a, b)| a / b)
        .collect();
    let mut a = Tensor::<f32>::from_data(a_data, vec![n], DeviceType::Cpu)
        .expect("failed to create tensor a");
    let b = Tensor::<f32>::from_data(b_data, vec![n], DeviceType::Cpu)
        .expect("failed to create tensor b");
    a.div_(&b).expect("div_ should succeed");
    let result = a.data().expect("failed to get div_ result data");
    for (i, (&got, &exp)) in result.iter().zip(ref_data.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "mismatch at i={}: got={}, expected={}",
            i,
            got,
            exp
        );
    }
}

// Block D: in-place activation f32 SIMD fast-path tests

#[test]
fn test_f32_relu_inplace_simd() {
    let n = 4096;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) - (n as f32 / 2.0)).collect();
    let expected: Vec<f32> = data
        .iter()
        .map(|&x| if x < 0.0 { 0.0 } else { x })
        .collect();
    let mut t =
        Tensor::<f32>::from_data(data, vec![n], DeviceType::Cpu).expect("failed to create tensor");
    t.relu_().expect("relu_ should succeed");
    let result = t.data().expect("failed to get relu_ result data");
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "mismatch at i={}: got={}, expected={}",
            i,
            got,
            exp
        );
    }
}

#[test]
fn test_f32_leaky_relu_inplace_simd() {
    let n = 4096;
    let slope = 0.01f32;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) - (n as f32 / 2.0)).collect();
    let expected: Vec<f32> = data
        .iter()
        .map(|&x| if x >= 0.0 { x } else { slope * x })
        .collect();
    let mut t =
        Tensor::<f32>::from_data(data, vec![n], DeviceType::Cpu).expect("failed to create tensor");
    t.leaky_relu_(slope).expect("leaky_relu_ should succeed");
    let result = t.data().expect("failed to get leaky_relu_ result data");
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "mismatch at i={}: got={}, expected={}",
            i,
            got,
            exp
        );
    }
}

#[test]
fn test_f32_clamp_inplace_simd() {
    let n = 4096;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) - (n as f32 / 2.0)).collect();
    let min_val = -100.0f32;
    let max_val = 100.0f32;
    let expected: Vec<f32> = data
        .iter()
        .map(|&x| {
            if x < min_val {
                min_val
            } else if x > max_val {
                max_val
            } else {
                x
            }
        })
        .collect();
    let mut t =
        Tensor::<f32>::from_data(data, vec![n], DeviceType::Cpu).expect("failed to create tensor");
    t.clamp_(min_val, max_val).expect("clamp_ should succeed");
    let result = t.data().expect("failed to get clamp_ result data");
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "mismatch at i={}: got={}, expected={}",
            i,
            got,
            exp
        );
    }
}

// --- Regression tests for issue #43: sub must propagate requires_grad ---

#[test]
fn test_issue_43_sub_propagates_requires_grad() {
    // Both operands have requires_grad=true; result must too.
    let a = Tensor::from_data(vec![3.0f32, 4.0], vec![2], DeviceType::Cpu)
        .expect("tensor creation failed")
        .requires_grad_(true);
    let b = Tensor::from_data(vec![1.0f32, 1.0], vec![2], DeviceType::Cpu)
        .expect("tensor creation failed")
        .requires_grad_(true);

    let result = a.sub(&b).expect("sub should succeed");
    assert!(
        result.requires_grad(),
        "sub result must have requires_grad=true when either operand does"
    );
}

#[test]
fn test_issue_43_sub_propagates_requires_grad_one_sided() {
    // Only lhs operand has requires_grad; result must still have it.
    let a = Tensor::from_data(vec![3.0f32, 4.0], vec![2], DeviceType::Cpu)
        .expect("tensor creation failed")
        .requires_grad_(true);
    let b = Tensor::from_data(vec![1.0f32, 1.0], vec![2], DeviceType::Cpu)
        .expect("tensor creation failed");

    let result = a.sub(&b).expect("sub should succeed");
    assert!(
        result.requires_grad(),
        "sub result must have requires_grad=true when lhs has it"
    );
}

#[test]
fn test_issue_43_sub_no_requires_grad_when_both_false() {
    // Neither operand has requires_grad; result should not either.
    let a = Tensor::from_data(vec![3.0f32, 4.0], vec![2], DeviceType::Cpu)
        .expect("tensor creation failed");
    let b = Tensor::from_data(vec![1.0f32, 1.0], vec![2], DeviceType::Cpu)
        .expect("tensor creation failed");

    let result = a.sub(&b).expect("sub should succeed");
    assert!(
        !result.requires_grad(),
        "sub result must NOT have requires_grad when neither operand does"
    );
}

#[test]
fn test_issue_43_sub_backward() {
    // Use scalar (single-element) tensors so backward() can be called directly
    // on the sub result without needing a reduction.
    // d/dlhs (lhs - rhs) = +1;  d/drhs (lhs - rhs) = -1
    let lhs = Tensor::from_data(vec![5.0f32], vec![1], DeviceType::Cpu)
        .expect("tensor creation failed")
        .requires_grad_(true);
    let rhs = Tensor::from_data(vec![3.0f32], vec![1], DeviceType::Cpu)
        .expect("tensor creation failed")
        .requires_grad_(true);

    let result = lhs.sub(&rhs).expect("sub should succeed");
    assert!(result.requires_grad(), "sub result must track gradients");

    // result is a scalar (numel=1), so backward() is valid
    result.backward().expect("backward should succeed");

    let lhs_grad = lhs.grad().expect("lhs must have gradient after backward");
    let rhs_grad = rhs.grad().expect("rhs must have gradient after backward");

    let lhs_grad_data = lhs_grad.data().expect("lhs grad data");
    let rhs_grad_data = rhs_grad.data().expect("rhs grad data");

    // d(lhs - rhs)/dlhs = +1
    assert_eq!(lhs_grad_data, vec![1.0f32], "lhs grad should be +1");
    // d(lhs - rhs)/drhs = -1
    assert_eq!(rhs_grad_data, vec![-1.0f32], "rhs grad should be -1");
}

// --- In-place broadcast tests for add_ / sub_ / mul_ / div_ ---

#[test]
fn test_add_inplace_same_shape_small() {
    // Small (< 1024) so it exercises the generic-T equal-shape path,
    // not the f32 SIMD fast path.
    let mut a = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)
        .expect("tensor a creation failed");
    let b = Tensor::from_data(vec![10.0f32, 20.0, 30.0], vec![3], DeviceType::Cpu)
        .expect("tensor b creation failed");

    a.add_(&b).expect("add_ should succeed");
    assert_eq!(a.data().expect("data"), vec![11.0, 22.0, 33.0]);
}

#[test]
fn test_mul_inplace_same_shape_small() {
    let mut a = Tensor::from_data(vec![2.0f32, 3.0, 4.0], vec![3], DeviceType::Cpu)
        .expect("tensor a creation failed");
    let b = Tensor::from_data(vec![5.0f32, 6.0, 7.0], vec![3], DeviceType::Cpu)
        .expect("tensor b creation failed");

    a.mul_(&b).expect("mul_ should succeed");
    assert_eq!(a.data().expect("data"), vec![10.0, 18.0, 28.0]);
}

#[test]
fn test_add_inplace_broadcast_row_into_matrix() {
    // Bias-add pattern: [2, 3] += [3] should add the row vector to each row of self.
    let mut mat = Tensor::from_data(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DeviceType::Cpu,
    )
    .expect("matrix creation failed");
    let bias = Tensor::from_data(vec![10.0f32, 20.0, 30.0], vec![3], DeviceType::Cpu)
        .expect("bias creation failed");

    mat.add_(&bias).expect("broadcast add_ should succeed");
    assert_eq!(
        mat.data().expect("data"),
        vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
    );
}

#[test]
fn test_mul_inplace_broadcast_column_into_matrix() {
    // Per-row scaling: [2, 3] *= [2, 1] should multiply each row by a single scalar.
    let mut mat = Tensor::from_data(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        DeviceType::Cpu,
    )
    .expect("matrix creation failed");
    let col = Tensor::from_data(vec![10.0f32, 100.0], vec![2, 1], DeviceType::Cpu)
        .expect("col creation failed");

    mat.mul_(&col).expect("broadcast mul_ should succeed");
    assert_eq!(
        mat.data().expect("data"),
        vec![10.0, 20.0, 30.0, 400.0, 500.0, 600.0]
    );
}

#[test]
fn test_sub_inplace_broadcast_scalar_like_into_matrix() {
    // [2, 2] -= [1, 1] (broadcasts a single scalar into every element).
    let mut mat = Tensor::from_data(vec![10.0f32, 20.0, 30.0, 40.0], vec![2, 2], DeviceType::Cpu)
        .expect("matrix creation failed");
    let s = Tensor::from_data(vec![5.0f32], vec![1, 1], DeviceType::Cpu)
        .expect("scalar-shaped tensor creation failed");

    mat.sub_(&s).expect("broadcast sub_ should succeed");
    assert_eq!(mat.data().expect("data"), vec![5.0, 15.0, 25.0, 35.0]);
}

#[test]
fn test_div_inplace_broadcast_row_into_matrix() {
    let mut mat = Tensor::from_data(
        vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
        vec![2, 3],
        DeviceType::Cpu,
    )
    .expect("matrix creation failed");
    let row = Tensor::from_data(vec![2.0f32, 4.0, 5.0], vec![3], DeviceType::Cpu)
        .expect("row creation failed");

    mat.div_(&row).expect("broadcast div_ should succeed");
    assert_eq!(
        mat.data().expect("data"),
        vec![5.0, 5.0, 6.0, 20.0, 12.5, 12.0]
    );
}

#[test]
fn test_inplace_broadcast_self_would_grow_errors() {
    // self=[3], other=[2, 3] → broadcast shape would be [2, 3]; cannot grow self.
    let mut small = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)
        .expect("small creation failed");
    let big = Tensor::from_data(
        vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0],
        vec![2, 3],
        DeviceType::Cpu,
    )
    .expect("big creation failed");

    assert!(small.add_(&big).is_err(), "add_ must refuse to grow self");
    assert!(small.sub_(&big).is_err(), "sub_ must refuse to grow self");
    assert!(small.mul_(&big).is_err(), "mul_ must refuse to grow self");
    assert!(small.div_(&big).is_err(), "div_ must refuse to grow self");
}

#[test]
fn test_inplace_incompatible_shapes_error() {
    // [2] vs [3] is not broadcast-compatible at all.
    let mut a =
        Tensor::from_data(vec![1.0f32, 2.0], vec![2], DeviceType::Cpu).expect("a creation failed");
    let b = Tensor::from_data(vec![1.0f32, 2.0, 3.0], vec![3], DeviceType::Cpu)
        .expect("b creation failed");

    assert!(a.add_(&b).is_err());
    assert!(a.mul_(&b).is_err());
}
