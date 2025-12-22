//! Fuzz target for specialized quantization schemes
//!
//! Tests INT4, binary, and ternary quantization schemes
//! for robustness against various inputs.

#![no_main]

use libfuzzer_sys::fuzz_target;
use torsh_quantization::config::QuantConfig;
use torsh_quantization::specialized::{
    quantize_binary, quantize_int4_per_tensor, quantize_ternary,
};
use torsh_tensor::creation::tensor_1d;

#[derive(Debug, arbitrary::Arbitrary)]
struct FuzzInput {
    /// Scheme selector
    scheme: u8,
    /// Tensor data
    data: Vec<f32>,
}

fuzz_target!(|input: FuzzInput| {
    // Sanitize data
    let clean_data: Vec<f32> = input
        .data
        .iter()
        .take(512)
        .copied()
        .filter(|x| x.is_finite())
        .collect();

    if clean_data.len() < 4 {
        return;
    }

    if let Ok(tensor) = tensor_1d(&clean_data) {
        match input.scheme % 3 {
            0 => {
                // Test INT4 quantization
                let config = QuantConfig::int4();
                if let Ok((quantized, scale, zp)) = quantize_int4_per_tensor(&tensor, &config) {
                    assert!(scale > 0.0 && scale.is_finite());
                    assert!(zp >= -128 && zp <= 127);

                    if let Ok(data) = quantized.data() {
                        for &val in data.iter() {
                            assert!(val.is_finite());
                            assert!(
                                val >= -8.0 && val <= 7.0,
                                "INT4 value {} out of range",
                                val
                            );
                        }
                    }
                }
            }
            1 => {
                // Test binary quantization
                if let Ok((quantized, scale, zp)) = quantize_binary(&tensor) {
                    assert!(scale > 0.0 && scale.is_finite());
                    assert!(zp >= -128 && zp <= 127);

                    if let Ok(data) = quantized.data() {
                        for &val in data.iter() {
                            assert!(val.is_finite());
                            assert!(
                                val == -1.0 || val == 1.0,
                                "Binary value {} not in {{-1, 1}}",
                                val
                            );
                        }
                    }
                }
            }
            _ => {
                // Test ternary quantization
                if let Ok((quantized, scale, zp)) = quantize_ternary(&tensor) {
                    assert!(scale > 0.0 && scale.is_finite());
                    assert!(zp >= -128 && zp <= 127);

                    if let Ok(data) = quantized.data() {
                        for &val in data.iter() {
                            assert!(val.is_finite());
                            assert!(
                                val == -1.0 || val == 0.0 || val == 1.0,
                                "Ternary value {} not in {{-1, 0, 1}}",
                                val
                            );
                        }
                    }
                }
            }
        }
    }
});
