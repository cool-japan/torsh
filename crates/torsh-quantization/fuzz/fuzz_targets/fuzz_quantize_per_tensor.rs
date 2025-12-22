//! Fuzz target for per-tensor quantization
//!
//! This fuzzer tests the robustness of per-tensor quantization
//! against arbitrary inputs, looking for panics, overflows, and
//! invalid outputs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use torsh_quantization::algorithms::quantize_per_tensor_affine;
use torsh_tensor::creation::tensor_1d;

#[derive(Debug, arbitrary::Arbitrary)]
struct FuzzInput {
    /// Tensor data (limited size for performance)
    data: Vec<f32>,
    /// Quantization scale
    scale: f32,
    /// Zero point
    zero_point: i32,
}

fuzz_target!(|input: FuzzInput| {
    // Sanitize inputs to avoid trivial rejections
    let data: Vec<f32> = input
        .data
        .iter()
        .take(1024) // Limit size
        .copied()
        .filter(|x| x.is_finite()) // Only finite values
        .collect();

    if data.is_empty() || data.len() < 2 {
        return;
    }

    let scale = if input.scale.is_finite() && input.scale > 0.0 {
        input.scale.clamp(1e-6, 1e6)
    } else {
        1.0
    };

    let zero_point = input.zero_point.clamp(-128, 127);

    // Create tensor and quantize
    if let Ok(tensor) = tensor_1d(&data) {
        if let Ok((quantized, result_scale, result_zp)) =
            quantize_per_tensor_affine(&tensor, scale, zero_point)
        {
            // Verify invariants
            assert!(result_scale > 0.0, "Scale must be positive");
            assert!(result_scale.is_finite(), "Scale must be finite");
            assert!(
                result_zp >= -128 && result_zp <= 127,
                "Zero point must be in I8 range"
            );

            // Verify quantized values are in range
            if let Ok(quant_data) = quantized.data() {
                for &val in quant_data.iter() {
                    assert!(val.is_finite(), "Quantized value must be finite");
                    assert!(
                        val >= -128.0 && val <= 127.0,
                        "Quantized value {} out of range",
                        val
                    );
                }
            }
        }
    }
});
