//! Quick test to verify Bessel function fixes

use crate::{scirs2_integration::bessel_j0_scirs2, TorshResult};
use torsh_core::device::DeviceType;
use torsh_tensor::Tensor;

pub fn test_bessel_j0_fix() -> TorshResult<()> {
    println!("Testing Bessel J0 function...");

    let input = Tensor::from_data(vec![0.0f32, 1.0, 2.0], vec![3], DeviceType::Cpu)?;
    match bessel_j0_scirs2(&input) {
        Ok(result) => {
            let data = result.data()?;
            println!("J0(0) = {} (expected ~1.0)", data[0]);
            println!("J0(1) = {} (expected ~0.7652)", data[1]);
            println!("J0(2) = {} (expected ~0.2239)", data[2]);

            // Verify values are reasonable
            if (data[0] - 1.0).abs() < 0.01
                && (data[1] - 0.7652).abs() < 0.01
                && (data[2] - 0.2239).abs() < 0.01
            {
                println!("✓ Bessel J0 functions working correctly!");
            } else {
                println!("✗ Bessel J0 functions have unexpected values");
            }
        }
        Err(e) => println!("✗ Error testing Bessel J0: {e:?}"),
    }
    Ok(())
}
