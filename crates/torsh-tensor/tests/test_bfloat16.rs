use half::bf16;
use torsh_tensor::creation;

#[test]
fn test_bf16_tensor_creation() {
    let data = vec![
        bf16::from_f32(1.0),
        bf16::from_f32(2.0),
        bf16::from_f32(3.0),
    ];
    let tensor = creation::tensor_1d(&data).unwrap();

    assert_eq!(tensor.shape().dims(), &[3]);
    assert_eq!(tensor.data().unwrap(), data);
}

#[test]
fn test_bf16_zeros_ones() {
    let zeros = creation::zeros::<bf16>(&[2, 3]).unwrap();
    assert_eq!(zeros.shape().dims(), &[2, 3]);

    let zeros_data = zeros.data().unwrap();
    assert!(zeros_data.iter().all(|&x| x == bf16::from_f32(0.0)));

    let ones = creation::ones::<bf16>(&[2, 3]).unwrap();
    let ones_data = ones.data().unwrap();
    assert!(ones_data.iter().all(|&x| x == bf16::from_f32(1.0)));
}

#[test]
fn test_bf16_arithmetic() {
    let a = creation::tensor_1d(&[bf16::from_f32(1.5), bf16::from_f32(2.5)]).unwrap();
    let b = creation::tensor_1d(&[bf16::from_f32(0.5), bf16::from_f32(1.5)]).unwrap();

    let result = a.add_op(&b).unwrap();
    let result_data = result.data().unwrap();

    assert!((result_data[0].to_f32() - 2.0).abs() < 1e-6);
    assert!((result_data[1].to_f32() - 4.0).abs() < 1e-6);
}

#[test]
fn test_bf16_precision_characteristics() {
    // Test that bf16 behaves as expected for precision limits
    let large_value = 65504.0f32; // Near bf16 max
    let bf16_large = bf16::from_f32(large_value);
    let tensor = creation::tensor_1d(&[bf16_large]).unwrap();
    let data = tensor.data().unwrap();

    // Should preserve large values with some precision loss
    assert!((data[0].to_f32() - large_value).abs() < 1000.0);

    // Test small values
    let small_value = 1e-6f32;
    let bf16_small = bf16::from_f32(small_value);
    let small_tensor = creation::tensor_1d(&[bf16_small]).unwrap();
    let small_data = small_tensor.data().unwrap();

    // bf16 might not represent very small numbers accurately
    assert!(small_data[0].to_f32() >= 0.0);
}
