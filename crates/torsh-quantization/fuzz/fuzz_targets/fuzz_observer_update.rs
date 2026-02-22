//! Fuzz target for observer updates
//!
//! Tests the robustness of observer parameter calculation
//! against various tensor distributions and edge cases.

#![no_main]

use libfuzzer_sys::fuzz_target;
use torsh_core::DType;
use torsh_quantization::config::ObserverType;
use torsh_quantization::observers::Observer;
use torsh_tensor::creation::tensor_1d;

#[derive(Debug, arbitrary::Arbitrary)]
struct FuzzInput {
    /// Observer type selection
    observer_selector: u8,
    /// Multiple tensor updates
    tensors: Vec<Vec<f32>>,
}

fuzz_target!(|input: FuzzInput| {
    // Select observer type
    let observer_type = match input.observer_selector % 4 {
        0 => ObserverType::MinMax,
        1 => ObserverType::Histogram,
        2 => ObserverType::Percentile,
        _ => ObserverType::MovingAverage,
    };

    let mut observer = Observer::new(observer_type);

    // Process multiple tensor updates
    for tensor_data in input.tensors.iter().take(10) {
        // Sanitize data
        let clean_data: Vec<f32> = tensor_data
            .iter()
            .take(256)
            .copied()
            .filter(|x| x.is_finite())
            .collect();

        if clean_data.len() < 2 {
            continue;
        }

        // Update observer
        if let Ok(tensor) = tensor_1d(&clean_data) {
            let _ = observer.update(&tensor);
        }
    }

    // Calculate quantization parameters
    if let Ok((scale, zero_point)) = observer.calculate_qparams(DType::I8) {
        // Verify invariants
        assert!(scale > 0.0, "Scale must be positive");
        assert!(scale.is_finite(), "Scale must be finite");
        assert!(
            zero_point >= -128 && zero_point <= 127,
            "Zero point {} must be in I8 range",
            zero_point
        );
    }
});
