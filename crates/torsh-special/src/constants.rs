//! Mathematical constants for special functions
//!
//! This module provides commonly used mathematical constants in special function
//! computations, including high-precision values and derived constants.

/// Euler-Mascheroni constant γ ≈ 0.5772156649015329
pub const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Catalan's constant G ≈ 0.9159655941772190
pub const CATALAN: f64 = 0.915_965_594_177_219;

/// Golden ratio φ = (1 + √5)/2 ≈ 1.6180339887498948
pub const GOLDEN_RATIO: f64 = 1.618_033_988_749_895;

/// Natural logarithm of 2π
pub const LN_2PI: f64 = 1.837_877_066_409_345_5;

/// Square root of 2π
pub const SQRT_2PI: f64 = 2.506_628_274_631_000_5;

/// Apéry's constant ζ(3) ≈ 1.2020569031595942
pub const APERY: f64 = 1.202_056_903_159_594_3;

/// π²/6 = ζ(2) ≈ 1.6449340668482264
pub const PI_SQUARED_OVER_6: f64 = 1.644_934_066_848_226_4;

/// Natural logarithm of π
pub const LN_PI: f64 = 1.144_729_885_849_4;

/// Reciprocal of the golden ratio (φ - 1) ≈ 0.6180339887498948
pub const GOLDEN_RATIO_RECIPROCAL: f64 = 0.618_033_988_749_894_9;

/// Feigenbaum's first constant δ ≈ 4.6692016091029906
pub const FEIGENBAUM_DELTA: f64 = 4.669_201_609_102_99;

/// Feigenbaum's second constant α ≈ 2.5029078750958930
pub const FEIGENBAUM_ALPHA: f64 = 2.502_907_875_095_893;

/// Mills' constant A ≈ 1.3063778838630806
pub const MILLS: f64 = 1.306_377_883_863_081;

/// Plastic number ρ ≈ 1.3247179572447460
pub const PLASTIC_NUMBER: f64 = 1.324_717_957_244_746;

/// Khinchin's constant K ≈ 2.6854520010653064
pub const KHINCHIN: f64 = 2.685_452_001_065_306_4;

/// Lévy's constant γ ≈ 3.2758229187218120
pub const LEVY: f64 = 3.275_822_918_721_812;

/// Ramanujan's constant e^(π*√163) ≈ 2.625374126407687e+17
/// Note: This is a transcendental number, not an integer despite appearing close to one
pub const RAMANUJAN: f64 = 2.625_374_126_407_687e17;

/// Glaisher-Kinkelin constant A ≈ 1.2824271291006226
pub const GLAISHER_KINKELIN: f64 = 1.282_427_129_100_622_6;

/// Dottie number (solution to cos(x) = x) ≈ 0.7390851332151607
pub const DOTTIE: f64 = 0.739_085_133_215_160_7;

/// Common special function utility constants
pub mod utility {
    use super::*;

    /// 1/√(2π) - useful in normal distribution calculations
    pub const INV_SQRT_2PI: f64 = 1.0 / SQRT_2PI;

    /// log(√(2π)) = 0.5 * ln(2π)
    pub const LOG_SQRT_2PI: f64 = LN_2PI * 0.5;

    /// 2/√π - useful in error function calculations
    pub const TWO_OVER_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;

    /// √(π/2) - useful in various special functions
    pub const SQRT_PI_OVER_2: f64 = 1.253_314_137_315_5;

    /// ln(4π) - appears in several special function normalizations
    pub const LN_4PI: f64 = 2.531_024_246_969_291;

    /// Stirling's constant √(2π) used in factorial approximations
    pub const STIRLING_CONSTANT: f64 = SQRT_2PI;

    /// 1/(2π) - frequently used in Fourier transforms and probability
    pub const INV_2PI: f64 = 1.0 / (2.0 * std::f64::consts::PI);

    /// π/4 - used in many trigonometric and integration contexts
    pub const PI_OVER_4: f64 = std::f64::consts::FRAC_PI_4;

    /// 3π/4 - common in phase calculations
    pub const THREE_PI_OVER_4: f64 = 3.0 * std::f64::consts::FRAC_PI_4;

    /// √(π/8) - appears in various statistical contexts
    pub const SQRT_PI_OVER_8: f64 = 0.626_657_068_657_750_2;

    /// ln(2π)/2 - half of LN_2PI, useful in Stirling approximations
    pub const HALF_LN_2PI: f64 = super::LN_2PI * 0.5;
}

/// Physical and mathematical constants for specialized applications
pub mod physics {
    /// Fine structure constant α ≈ 1/137.035999084
    pub const FINE_STRUCTURE: f64 = 7.297_352_566_4e-3;

    /// Planck constant h (in J⋅s)
    pub const PLANCK: f64 = 6.626_070_15e-34;

    /// Speed of light in vacuum c (in m/s)
    pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

    /// Boltzmann constant k_B (in J/K)
    pub const BOLTZMANN: f64 = 1.380_649e-23;
}

/// Commonly used fractions as high-precision constants
pub mod fractions {
    /// 1/3
    pub const ONE_THIRD: f64 = 1.0 / 3.0;

    /// 2/3
    pub const TWO_THIRDS: f64 = 2.0 / 3.0;

    /// 1/6
    pub const ONE_SIXTH: f64 = 1.0 / 6.0;

    /// 5/6
    pub const FIVE_SIXTHS: f64 = 5.0 / 6.0;

    /// 1/12
    pub const ONE_TWELFTH: f64 = 1.0 / 12.0;

    /// 7/12
    pub const SEVEN_TWELFTHS: f64 = 7.0 / 12.0;
}

/// High-precision Riemann zeta function values at integer points
pub mod zeta_values {
    /// ζ(2) = π²/6
    pub const ZETA_2: f64 = super::PI_SQUARED_OVER_6;

    /// ζ(3) = Apéry's constant
    pub const ZETA_3: f64 = super::APERY;

    /// ζ(4) = π⁴/90 ≈ 1.0823232337111382
    pub const ZETA_4: f64 = 1.082_323_233_711_138;

    /// ζ(5) ≈ 1.0369277551433699
    pub const ZETA_5: f64 = 1.036_927_755_143_37;

    /// ζ(6) = π⁶/945 ≈ 1.0173430619844491
    pub const ZETA_6: f64 = 1.017_343_061_984_449;
}

/// Bernoulli numbers (first few values)
pub mod bernoulli {
    /// B₀ = 1
    pub const B0: f64 = 1.0;

    /// B₁ = -1/2 (by convention, sometimes defined as +1/2)
    pub const B1: f64 = -0.5;

    /// B₂ = 1/6
    pub const B2: f64 = 1.0 / 6.0;

    /// B₄ = -1/30
    pub const B4: f64 = -1.0 / 30.0;

    /// B₆ = 1/42
    pub const B6: f64 = 1.0 / 42.0;

    /// B₈ = -1/30
    pub const B8: f64 = -1.0 / 30.0;

    /// B₁₀ = 5/66
    pub const B10: f64 = 5.0 / 66.0;

    /// B₁₂ = -691/2730
    pub const B12: f64 = -691.0 / 2730.0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts;

    #[test]
    fn test_mathematical_constants() {
        // Test that our constants are reasonable approximations
        assert!((EULER_GAMMA - 0.5772156649).abs() < 1e-9);
        assert!((CATALAN - 0.9159655942).abs() < 1e-9);
        assert!((GOLDEN_RATIO - 1.6180339887).abs() < 1e-9);
        assert!((PI_SQUARED_OVER_6 - (consts::PI.powi(2) / 6.0)).abs() < 1e-15);
    }

    #[test]
    fn test_utility_constants() {
        use utility::*;

        // Verify derived constants
        assert!((INV_SQRT_2PI - (1.0 / SQRT_2PI)).abs() < 1e-15);
        assert!((LOG_SQRT_2PI - (0.5 * LN_2PI)).abs() < 1e-15);
        assert!((TWO_OVER_SQRT_PI - (2.0 / consts::PI.sqrt())).abs() < 1e-15);
    }

    #[test]
    fn test_zeta_values() {
        use zeta_values::*;

        // Basic sanity checks
        assert!(ZETA_2 > 1.6 && ZETA_2 < 1.7);
        assert!(ZETA_3 > 1.2 && ZETA_3 < 1.3);
        assert!(ZETA_4 > 1.0 && ZETA_4 < 1.1);
    }

    #[test]
    fn test_bernoulli_numbers() {
        use bernoulli::*;

        // Test known Bernoulli numbers
        assert_eq!(B0, 1.0);
        assert_eq!(B1, -0.5);
        assert!((B2 - 1.0 / 6.0).abs() < 1e-15);
        assert!((B4 + 1.0 / 30.0).abs() < 1e-15);
    }
}
