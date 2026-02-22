//! Property-Based Testing for Quantization
//!
//! This module implements comprehensive property-based tests using proptest
//! to automatically discover edge cases and verify correctness properties
//! across a wide range of inputs.

use proptest::prelude::*;
use scirs2_core::Distribution;
use torsh_core::DType;
use torsh_quantization::{
    algorithms::{dequantize_per_tensor_affine, quantize_per_tensor_affine},
    config::{ObserverType, QScheme, QuantConfig},
    observers::Observer,
    simd_ops::{
        dequantize_per_tensor_affine_simd, find_min_max_simd, quantize_per_tensor_affine_simd,
    },
    specialized::{
        quantize_binary, quantize_group_wise, quantize_int4_per_tensor, quantize_ternary,
    },
};
use torsh_tensor::creation::tensor_1d;

// =============================================================================
// Property-Based Test Strategies
// =============================================================================

/// Generate valid floating-point values (no NaN/Inf)
fn valid_f32() -> impl Strategy<Value = f32> {
    prop::num::f32::NORMAL
}

/// Generate valid tensor data (arrays of valid floats)
fn valid_tensor_data(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(valid_f32(), min_len..=max_len)
}

/// Generate valid quantization scales (positive, non-zero)
fn valid_scale() -> impl Strategy<Value = f32> {
    0.001_f32..1000.0_f32
}

/// Generate valid zero points for INT8
fn valid_zero_point_i8() -> impl Strategy<Value = i32> {
    -128_i32..=127_i32
}

/// Generate valid zero points for INT4
fn _valid_zero_point_i4() -> impl Strategy<Value = i32> {
    -8_i32..=7_i32
}

/// Generate valid quantization configurations
fn valid_quant_config() -> impl Strategy<Value = QuantConfig> {
    prop_oneof![
        Just(QuantConfig::int8()),
        Just(QuantConfig::int4()),
        Just(QuantConfig::binary()),
        Just(QuantConfig::ternary()),
        (0usize..10usize).prop_map(|ch| QuantConfig::per_channel(ch)),
        (16usize..128usize).prop_map(|gs| QuantConfig::group_wise(0, gs)),
    ]
}

// =============================================================================
// Core Quantization Properties
// =============================================================================

proptest! {
    /// Property: Quantized values should be within the valid quantization range
    #[test]
    fn prop_quantized_values_in_range(
        data in valid_tensor_data(4, 1024),
        scale in valid_scale(),
        zero_point in valid_zero_point_i8(),
    ) {
        let tensor = tensor_1d(&data).unwrap();
        let (quantized, _, _) = quantize_per_tensor_affine(&tensor, scale, zero_point).unwrap();
        let quant_data = quantized.data().unwrap();

        // All values should be in INT8 range
        for &val in quant_data.iter() {
            prop_assert!(val >= -128.0 && val <= 127.0,
                "Quantized value {} outside INT8 range [-128, 127]", val);
        }
    }

    /// Property: Dequantize(Quantize(x)) should approximately equal x
    #[test]
    fn prop_quantize_dequantize_roundtrip(
        data in valid_tensor_data(4, 256),
        scale in valid_scale(),
        zero_point in valid_zero_point_i8(),
    ) {
        let original_data = &data;

        // Calculate data range to check if quantization is feasible
        let min_val = original_data.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = original_data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let data_range = (max_val - min_val).abs();

        // Skip cases where the data range is too large for the scale
        // This would require more than 256 distinct values to represent accurately
        let required_range = data_range / scale;
        if !required_range.is_finite() || required_range > 256.0 * 1000.0 {
            // Skip extreme cases that can't be accurately quantized to INT8
            return Ok(());
        }

        // Also skip cases with extreme dynamic range
        // Check all pairwise ratios in the data
        let abs_values: Vec<f32> = original_data.iter().map(|&x| x.abs()).filter(|&x| x > 1e-20).collect();
        if abs_values.len() >= 2 {
            let min_abs = abs_values.iter().copied().fold(f32::INFINITY, f32::min);
            let max_abs = abs_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let ratio = max_abs / min_abs;
            if !ratio.is_finite() || ratio > 1e4 {
                // Skip cases with extreme ratios between values
                return Ok(());
            }
        }

        let tensor = tensor_1d(original_data).unwrap();
        let (quantized, final_scale, final_zp) = quantize_per_tensor_affine(&tensor, scale, zero_point).unwrap();
        let dequantized = dequantize_per_tensor_affine(&quantized, final_scale, final_zp).unwrap();

        let original_tensor_data = tensor.data().unwrap();
        let recovered_data = dequantized.data().unwrap();

        // Calculate maximum expected quantization error based on the quantization step size
        let max_error = final_scale * 1.5; // Allow for rounding + tolerance

        for (i, (&orig, &recovered)) in original_tensor_data.iter().zip(recovered_data.iter()).enumerate() {
            let error = (orig - recovered).abs();

            prop_assert!(error <= max_error,
                "Roundtrip error {} at index {} exceeds threshold {} (scale={}, orig={}, recovered={})",
                error, i, max_error, final_scale, orig, recovered);
        }
    }

    /// Property: Scale should always be positive after quantization
    #[test]
    fn prop_scale_always_positive(
        data in valid_tensor_data(4, 128),
        scale in valid_scale(),
        zero_point in valid_zero_point_i8(),
    ) {
        let tensor = tensor_1d(&data).unwrap();
        let (_, final_scale, _) = quantize_per_tensor_affine(&tensor, scale, zero_point).unwrap();
        prop_assert!(final_scale > 0.0, "Scale must be positive, got {}", final_scale);
    }

    /// Property: Zero point should be within INT8 range
    #[test]
    fn prop_zero_point_in_range(
        data in valid_tensor_data(4, 128),
        scale in valid_scale(),
        zero_point in valid_zero_point_i8(),
    ) {
        let tensor = tensor_1d(&data).unwrap();
        let (_, _, final_zp) = quantize_per_tensor_affine(&tensor, scale, zero_point).unwrap();
        prop_assert!(final_zp >= -128 && final_zp <= 127,
            "Zero point {} outside INT8 range", final_zp);
    }
}

// =============================================================================
// SIMD Operations Properties
// =============================================================================

proptest! {
    /// Property: SIMD quantization should match scalar quantization
    #[test]
    fn prop_simd_quantization_matches_scalar(
        data in valid_tensor_data(16, 256),
        scale in valid_scale(),
        zero_point in valid_zero_point_i8(),
    ) {
        // SIMD version
        let mut simd_output = vec![0.0_f32; data.len()];
        quantize_per_tensor_affine_simd(&data, scale, zero_point, &mut simd_output).unwrap();

        // Scalar version
        let tensor = tensor_1d(&data).unwrap();
        let (scalar_quantized, _, _) = quantize_per_tensor_affine(&tensor, scale, zero_point).unwrap();
        let scalar_output = scalar_quantized.data().unwrap();

        // Results should match exactly (both use same algorithm)
        for (i, (&simd_val, &scalar_val)) in simd_output.iter().zip(scalar_output.iter()).enumerate() {
            prop_assert!((simd_val - scalar_val).abs() < 0.1,
                "SIMD/scalar mismatch at index {}: SIMD={}, scalar={}",
                i, simd_val, scalar_val);
        }
    }

    /// Property: SIMD dequantization should match scalar dequantization
    #[test]
    fn prop_simd_dequantization_matches_scalar(
        data in valid_tensor_data(16, 256),
        scale in valid_scale(),
        zero_point in valid_zero_point_i8(),
    ) {
        // Create quantized data
        let tensor = tensor_1d(&data).unwrap();
        let (quantized, _, _) = quantize_per_tensor_affine(&tensor, scale, zero_point).unwrap();
        let quant_data = quantized.data().unwrap();

        // SIMD dequantization
        let mut simd_output = vec![0.0_f32; quant_data.len()];
        dequantize_per_tensor_affine_simd(&quant_data, scale, zero_point, &mut simd_output).unwrap();

        // Scalar dequantization
        let scalar_dequantized = dequantize_per_tensor_affine(&quantized, scale, zero_point).unwrap();
        let scalar_output = scalar_dequantized.data().unwrap();

        // Results should match exactly
        for (i, (&simd_val, &scalar_val)) in simd_output.iter().zip(scalar_output.iter()).enumerate() {
            prop_assert!((simd_val - scalar_val).abs() < 1e-5,
                "SIMD/scalar dequant mismatch at index {}: SIMD={}, scalar={}",
                i, simd_val, scalar_val);
        }
    }

    /// Property: SIMD min/max should match iterator min/max
    #[test]
    fn prop_simd_minmax_correct(data in valid_tensor_data(4, 512)) {
        let (simd_min, simd_max) = find_min_max_simd(&data).unwrap();

        let iter_min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let iter_max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        prop_assert!((simd_min - iter_min).abs() < 1e-6,
            "SIMD min {} doesn't match iterator min {}", simd_min, iter_min);
        prop_assert!((simd_max - iter_max).abs() < 1e-6,
            "SIMD max {} doesn't match iterator max {}", simd_max, iter_max);
    }
}

// =============================================================================
// Observer Properties
// =============================================================================

proptest! {
    /// Property: Observer should produce valid scale and zero point
    #[test]
    fn prop_observer_produces_valid_params(
        test_data in valid_tensor_data(8, 256),
        observer_type in prop_oneof![
            Just(ObserverType::MinMax),
            Just(ObserverType::Histogram),
            Just(ObserverType::Percentile),
        ],
    ) {
        let tensor = tensor_1d(&test_data).unwrap();
        let mut observer = Observer::new(observer_type);
        observer.update(&tensor).unwrap();

        let (scale, zero_point) = observer.calculate_qparams(DType::I8).unwrap();

        prop_assert!(scale > 0.0, "Scale must be positive, got {}", scale);
        prop_assert!(zero_point >= -128 && zero_point <= 127,
            "Zero point {} outside INT8 range", zero_point);
    }

    /// Property: Multiple observer updates should maintain valid parameters
    #[test]
    fn prop_observer_consistent_across_updates(
        test_data1 in valid_tensor_data(8, 128),
        test_data2 in valid_tensor_data(8, 128),
        test_data3 in valid_tensor_data(8, 128),
    ) {
        let mut observer = Observer::new(ObserverType::MinMax);

        let tensor1 = tensor_1d(&test_data1).unwrap();
        let tensor2 = tensor_1d(&test_data2).unwrap();
        let tensor3 = tensor_1d(&test_data3).unwrap();

        observer.update(&tensor1).unwrap();
        observer.update(&tensor2).unwrap();
        observer.update(&tensor3).unwrap();

        let (scale, zero_point) = observer.calculate_qparams(DType::I8).unwrap();

        prop_assert!(scale > 0.0);
        prop_assert!(zero_point >= -128 && zero_point <= 127);
    }
}

// =============================================================================
// Specialized Quantization Properties
// =============================================================================

proptest! {
    /// Property: INT4 quantized values should be in [-8, 7] range
    #[test]
    fn prop_int4_values_in_range(data in valid_tensor_data(8, 256)) {
        let tensor = tensor_1d(&data).unwrap();
        let config = QuantConfig::int4();
        let (quantized, _, _) = quantize_int4_per_tensor(&tensor, &config).unwrap();
        let quant_data = quantized.data().unwrap();

        for &val in quant_data.iter() {
            prop_assert!(val >= -8.0 && val <= 7.0,
                "INT4 quantized value {} outside range [-8, 7]", val);
        }
    }

    /// Property: Binary quantization should produce only {-1, +1} values
    #[test]
    fn prop_binary_values_only_binary(data in valid_tensor_data(8, 256)) {
        let tensor = tensor_1d(&data).unwrap();
        let (quantized, _, _) = quantize_binary(&tensor).unwrap();
        let quant_data = quantized.data().unwrap();

        for &val in quant_data.iter() {
            prop_assert!(val == -1.0 || val == 1.0,
                "Binary quantized value {} not in {{-1, +1}}", val);
        }
    }

    /// Property: Ternary quantization should produce only {-1, 0, +1} values
    #[test]
    fn prop_ternary_values_only_ternary(data in valid_tensor_data(8, 256)) {
        let tensor = tensor_1d(&data).unwrap();
        let (quantized, _, _) = quantize_ternary(&tensor).unwrap();
        let quant_data = quantized.data().unwrap();

        for &val in quant_data.iter() {
            prop_assert!(val == -1.0 || val == 0.0 || val == 1.0,
                "Ternary quantized value {} not in {{-1, 0, +1}}", val);
        }
    }

    /// Property: Group-wise quantization should handle groups correctly
    #[test]
    fn prop_groupwise_handles_groups(
        data in valid_tensor_data(64, 512),
        group_size in 8_usize..=64_usize,
    ) {
        // Ensure data length is divisible by group_size
        let adjusted_len = (data.len() / group_size) * group_size;
        if adjusted_len < 8 {
            return Ok(());
        }
        let trimmed_data = &data[..adjusted_len];

        let tensor = tensor_1d(trimmed_data).unwrap();
        let config = QuantConfig::group_wise(0, group_size);
        let result = quantize_group_wise(&tensor, 0, group_size, &config);

        // Should succeed for valid group sizes
        prop_assert!(result.is_ok(), "Group-wise quantization failed for group_size={}", group_size);
    }
}

// =============================================================================
// Configuration Properties
// =============================================================================

proptest! {
    /// Property: All preset configurations should be valid
    #[test]
    fn prop_preset_configs_valid(config in valid_quant_config()) {
        prop_assert!(config.validate().is_ok(),
            "Preset configuration {:?} failed validation", config.scheme);
    }

    /// Property: Configuration scheme should match expected dtype
    #[test]
    fn prop_config_dtype_matches_scheme(config in valid_quant_config()) {
        match config.scheme {
            QScheme::PerTensorAffine | QScheme::PerChannelAffine => {
                prop_assert_eq!(config.dtype, DType::I8, "INT8 schemes should use I8 dtype");
            }
            QScheme::Int4PerTensor | QScheme::Int4PerChannel => {
                prop_assert_eq!(config.dtype, DType::I8, "INT4 schemes use I8 dtype");
            }
            QScheme::Binary | QScheme::Ternary => {
                prop_assert_eq!(config.dtype, DType::I8, "Binary/Ternary use I8 dtype");
            }
            _ => {}
        }
    }
}

// =============================================================================
// Edge Cases and Boundary Conditions
// =============================================================================

proptest! {
    /// Property: Quantization should handle all-zero tensors
    #[test]
    fn prop_handles_all_zeros(len in 4_usize..=256_usize) {
        let data = vec![0.0_f32; len];
        let tensor = tensor_1d(&data).unwrap();

        let mut observer = Observer::new(ObserverType::MinMax);
        observer.update(&tensor).unwrap();
        let (scale, zero_point) = observer.calculate_qparams(DType::I8).unwrap();

        prop_assert!(scale > 0.0, "Scale should be positive even for all-zero tensor");

        let result = quantize_per_tensor_affine(&tensor, scale, zero_point);
        prop_assert!(result.is_ok(), "Should handle all-zero tensors");
    }

    /// Property: Quantization should handle constant tensors
    #[test]
    fn prop_handles_constant_values(
        len in 4_usize..=256_usize,
        constant in valid_f32(),
    ) {
        let data = vec![constant; len];
        let tensor = tensor_1d(&data).unwrap();

        let mut observer = Observer::new(ObserverType::MinMax);
        observer.update(&tensor).unwrap();
        let (scale, zero_point) = observer.calculate_qparams(DType::I8).unwrap();

        prop_assert!(scale > 0.0);

        let result = quantize_per_tensor_affine(&tensor, scale, zero_point);
        prop_assert!(result.is_ok(), "Should handle constant tensors");
    }

    /// Property: Quantization should be deterministic
    #[test]
    fn prop_quantization_deterministic(
        data in valid_tensor_data(16, 128),
        scale in valid_scale(),
        zero_point in valid_zero_point_i8(),
    ) {
        let tensor = tensor_1d(&data).unwrap();

        let (q1, s1, zp1) = quantize_per_tensor_affine(&tensor, scale, zero_point).unwrap();
        let (q2, s2, zp2) = quantize_per_tensor_affine(&tensor, scale, zero_point).unwrap();

        prop_assert_eq!(s1, s2, "Scale should be deterministic");
        prop_assert_eq!(zp1, zp2, "Zero point should be deterministic");

        let d1 = q1.data().unwrap();
        let d2 = q2.data().unwrap();

        for (i, (&v1, &v2)) in d1.iter().zip(d2.iter()).enumerate() {
            prop_assert!((v1 - v2).abs() < 1e-6,
                "Values should be deterministic at index {}: {} vs {}", i, v1, v2);
        }
    }
}

// =============================================================================
// Numerical Stability Properties
// =============================================================================

proptest! {
    /// Property: Quantization should not introduce NaN or Inf
    #[test]
    fn prop_no_nan_inf_after_quantization(
        data in valid_tensor_data(8, 256),
        scale in valid_scale(),
        zero_point in valid_zero_point_i8(),
    ) {
        let tensor = tensor_1d(&data).unwrap();
        let (quantized, _, _) = quantize_per_tensor_affine(&tensor, scale, zero_point).unwrap();
        let quant_data = quantized.data().unwrap();

        for &val in quant_data.iter() {
            prop_assert!(!val.is_nan(), "Quantization produced NaN");
            prop_assert!(!val.is_infinite(), "Quantization produced Inf");
        }
    }

    /// Property: Dequantization should not introduce NaN or Inf
    #[test]
    fn prop_no_nan_inf_after_dequantization(
        data in valid_tensor_data(8, 256),
        scale in valid_scale(),
        zero_point in valid_zero_point_i8(),
    ) {
        let tensor = tensor_1d(&data).unwrap();
        let (quantized, s, zp) = quantize_per_tensor_affine(&tensor, scale, zero_point).unwrap();
        let dequantized = dequantize_per_tensor_affine(&quantized, s, zp).unwrap();
        let dequant_data = dequantized.data().unwrap();

        for &val in dequant_data.iter() {
            prop_assert!(!val.is_nan(), "Dequantization produced NaN");
            prop_assert!(!val.is_infinite(), "Dequantization produced Inf");
        }
    }

    /// Property: Very small scales should not cause overflow
    #[test]
    fn prop_handles_small_scales(
        data in valid_tensor_data(8, 128),
        scale in 1e-6_f32..=1e-3_f32,
        zero_point in valid_zero_point_i8(),
    ) {
        let tensor = tensor_1d(&data).unwrap();
        let result = quantize_per_tensor_affine(&tensor, scale, zero_point);

        prop_assert!(result.is_ok(), "Should handle very small scales");

        let (quantized, _, _) = result.unwrap();
        let quant_data = quantized.data().unwrap();

        for &val in quant_data.iter() {
            prop_assert!(!val.is_infinite(), "Small scale caused overflow");
            prop_assert!(val >= -128.0 && val <= 127.0, "Value {} out of range", val);
        }
    }

    /// Property: Large dynamic ranges should be handled correctly
    #[test]
    fn prop_handles_large_dynamic_range(
        min_val in -1000.0_f32..=-10.0_f32,
        max_val in 10.0_f32..=1000.0_f32,
        len in 8_usize..=128_usize,
    ) {
        use scirs2_core::random::{thread_rng, Uniform};

        let mut rng = thread_rng();
        let dist = Uniform::new(min_val, max_val).unwrap();
        let data: Vec<f32> = (0..len).map(|_| dist.sample(&mut rng)).collect();

        let tensor = tensor_1d(&data).unwrap();
        let mut observer = Observer::new(ObserverType::MinMax);
        observer.update(&tensor).unwrap();

        let (scale, zero_point) = observer.calculate_qparams(DType::I8).unwrap();

        prop_assert!(scale > 0.0);
        prop_assert!(scale.is_finite());

        let result = quantize_per_tensor_affine(&tensor, scale, zero_point);
        prop_assert!(result.is_ok(), "Should handle large dynamic ranges");
    }
}
