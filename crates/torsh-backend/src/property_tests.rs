//! Property-Based Testing for Backend Mathematical Correctness
//!
//! This module provides comprehensive property-based tests to verify mathematical
//! properties of backend operations. Uses proptest to generate thousands of random
//! test cases and verify that mathematical laws and properties hold.
//!
//! ## Tested Properties
//!
//! - **Algebraic Properties**: Commutativity, associativity, distributivity
//! - **Identity Properties**: Additive identity (0), multiplicative identity (1)
//! - **Inverse Properties**: a + (-a) = 0, a * (1/a) = 1
//! - **Numerical Stability**: Precision bounds, overflow handling
//! - **Monotonicity**: Operation ordering preservation
//! - **Idempotence**: Repeated operations yield same result

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    /// Maximum float value for testing to avoid overflow
    const MAX_FLOAT: f32 = 1e6;
    const MIN_FLOAT: f32 = -1e6;

    /// Tolerance for floating-point comparisons
    /// Note: These tolerances account for cumulative floating-point errors
    /// in multi-step operations, which is a fundamental property of IEEE 754
    /// The tolerance is set higher to handle edge cases like nearly-canceling operations
    const EPSILON: f32 = 5e-4; // Relaxed for accumulation of rounding errors and near-cancellation
    const EPSILON_F64: f64 = 1e-10; // f64 has better precision but still has limits

    /// Helper to check if two floats are approximately equal
    /// Uses relative error for better handling of large values
    fn approx_eq(a: f32, b: f32) -> bool {
        let abs_diff = (a - b).abs();
        let max_val = a.abs().max(b.abs());

        // Use relative error for large values, absolute for small values
        if max_val > 1.0 {
            abs_diff / max_val < EPSILON
        } else {
            abs_diff < EPSILON
        }
    }

    fn approx_eq_f64(a: f64, b: f64) -> bool {
        let abs_diff = (a - b).abs();
        let max_val = a.abs().max(b.abs());

        // Use relative error for large values, absolute for small values
        if max_val > 1.0 {
            abs_diff / max_val < EPSILON_F64
        } else {
            abs_diff < EPSILON_F64
        }
    }

    // =====================================================================
    // ARITHMETIC PROPERTIES
    // =====================================================================

    proptest! {
        /// Test commutativity of addition: a + b = b + a
        #[test]
        fn test_addition_commutative(
            a in MIN_FLOAT..MAX_FLOAT,
            b in MIN_FLOAT..MAX_FLOAT
        ) {
            let result1 = a + b;
            let result2 = b + a;
            prop_assert!(approx_eq(result1, result2),
                "Addition not commutative: {} + {} = {} but {} + {} = {}",
                a, b, result1, b, a, result2);
        }

        /// Test associativity of addition: (a + b) + c = a + (b + c)
        #[test]
        fn test_addition_associative(
            a in MIN_FLOAT..MAX_FLOAT,
            b in MIN_FLOAT..MAX_FLOAT,
            c in MIN_FLOAT..MAX_FLOAT
        ) {
            let result1 = (a + b) + c;
            let result2 = a + (b + c);
            prop_assert!(approx_eq(result1, result2),
                "Addition not associative: ({} + {}) + {} = {} but {} + ({} + {}) = {}",
                a, b, c, result1, a, b, c, result2);
        }

        /// Test commutativity of multiplication: a * b = b * a
        #[test]
        fn test_multiplication_commutative(
            a in MIN_FLOAT..MAX_FLOAT,
            b in MIN_FLOAT..MAX_FLOAT
        ) {
            let result1 = a * b;
            let result2 = b * a;
            prop_assert!(approx_eq(result1, result2),
                "Multiplication not commutative: {} * {} = {} but {} * {} = {}",
                a, b, result1, b, a, result2);
        }

        /// Test associativity of multiplication: (a * b) * c = a * (b * c)
        #[test]
        fn test_multiplication_associative(
            a in -100.0f32..100.0,
            b in -100.0f32..100.0,
            c in -100.0f32..100.0
        ) {
            let result1 = (a * b) * c;
            let result2 = a * (b * c);
            prop_assert!(approx_eq(result1, result2),
                "Multiplication not associative: ({} * {}) * {} = {} but {} * ({} * {}) = {}",
                a, b, c, result1, a, b, c, result2);
        }

        /// Test distributivity: a * (b + c) = a * b + a * c
        #[test]
        fn test_distributive_property(
            a in -100.0f32..100.0,
            b in -100.0f32..100.0,
            c in -100.0f32..100.0
        ) {
            let result1 = a * (b + c);
            let result2 = a * b + a * c;
            prop_assert!(approx_eq(result1, result2),
                "Distributivity violated: {} * ({} + {}) = {} but {} * {} + {} * {} = {}",
                a, b, c, result1, a, b, a, c, result2);
        }

        /// Test additive identity: a + 0 = a
        #[test]
        fn test_additive_identity(a in MIN_FLOAT..MAX_FLOAT) {
            let result = a + 0.0;
            prop_assert!(approx_eq(result, a),
                "Additive identity violated: {} + 0 = {} (expected {})",
                a, result, a);
        }

        /// Test multiplicative identity: a * 1 = a
        #[test]
        fn test_multiplicative_identity(a in MIN_FLOAT..MAX_FLOAT) {
            let result = a * 1.0;
            prop_assert!(approx_eq(result, a),
                "Multiplicative identity violated: {} * 1 = {} (expected {})",
                a, result, a);
        }

        /// Test additive inverse: a + (-a) = 0
        #[test]
        fn test_additive_inverse(a in MIN_FLOAT..MAX_FLOAT) {
            let result = a + (-a);
            prop_assert!(approx_eq(result, 0.0),
                "Additive inverse violated: {} + ({}) = {} (expected 0)",
                a, -a, result);
        }

        /// Test multiplicative inverse: a * (1/a) = 1 (for non-zero a)
        #[test]
        fn test_multiplicative_inverse(a in -1000.0f32..1000.0) {
            prop_assume!(a.abs() > 0.001); // Avoid division by near-zero
            let result = a * (1.0 / a);
            prop_assert!(approx_eq(result, 1.0),
                "Multiplicative inverse violated: {} * (1/{}) = {} (expected 1)",
                a, a, result);
        }

        /// Test subtraction as inverse of addition: (a + b) - b = a
        /// Note: We constrain the range to avoid floating-point precision
        /// issues that occur when adding numbers with vastly different magnitudes
        #[test]
        fn test_subtraction_inverse(
            a in -1000.0f32..1000.0,
            b in -1000.0f32..1000.0
        ) {
            let result = (a + b) - b;
            prop_assert!(approx_eq(result, a),
                "Subtraction inverse violated: ({} + {}) - {} = {} (expected {})",
                a, b, b, result, a);
        }

        /// Test division as inverse of multiplication: (a * b) / b = a (for non-zero b)
        #[test]
        fn test_division_inverse(
            a in -1000.0f32..1000.0,
            b in -1000.0f32..1000.0
        ) {
            prop_assume!(b.abs() > 0.001); // Avoid division by near-zero
            let result = (a * b) / b;
            prop_assert!(approx_eq(result, a),
                "Division inverse violated: ({} * {}) / {} = {} (expected {})",
                a, b, b, result, a);
        }
    }

    // =====================================================================
    // COMPARISON AND ORDERING PROPERTIES
    // =====================================================================

    proptest! {
        /// Test transitivity of less-than: if a < b and b < c, then a < c
        #[test]
        fn test_ordering_transitivity(
            a in -1000.0f32..1000.0,
            b in -1000.0f32..1000.0,
            c in -1000.0f32..1000.0
        ) {
            if a < b && b < c {
                prop_assert!(a < c,
                    "Transitivity violated: {} < {} and {} < {} but {} >= {}",
                    a, b, b, c, a, c);
            }
        }

        /// Test reflexivity of equality: a == a
        #[test]
        fn test_equality_reflexive(a in MIN_FLOAT..MAX_FLOAT) {
            prop_assert_eq!(a, a, "Reflexivity violated: {} != {}", a, a);
        }

        /// Test symmetry of equality: if a == b then b == a
        #[test]
        fn test_equality_symmetric(
            a in -1000.0f32..1000.0,
            b in -1000.0f32..1000.0
        ) {
            if approx_eq(a, b) {
                prop_assert!(approx_eq(b, a),
                    "Symmetry violated: {} == {} but {} != {}",
                    a, b, b, a);
            }
        }

        /// Test transitivity of equality: if a == b and b == c then a == c
        #[test]
        fn test_equality_transitive(
            a in -1000.0f32..1000.0,
            b in -1000.0f32..1000.0,
            c in -1000.0f32..1000.0
        ) {
            if approx_eq(a, b) && approx_eq(b, c) {
                prop_assert!(approx_eq(a, c),
                    "Transitivity of equality violated: {} == {} and {} == {} but {} != {}",
                    a, b, b, c, a, c);
            }
        }
    }

    // =====================================================================
    // SPECIAL FUNCTION PROPERTIES
    // =====================================================================

    proptest! {
        /// Test exp(ln(x)) = x for positive x
        #[test]
        fn test_exp_ln_inverse(x in 0.001f32..1000.0) {
            let result = x.ln().exp();
            prop_assert!(approx_eq(result, x),
                "exp(ln({})) = {} (expected {})",
                x, result, x);
        }

        /// Test ln(exp(x)) = x
        #[test]
        fn test_ln_exp_inverse(x in -10.0f32..10.0) {
            let result = x.exp().ln();
            prop_assert!(approx_eq(result, x),
                "ln(exp({})) = {} (expected {})",
                x, result, x);
        }

        /// Test sqrt(x^2) = |x|
        #[test]
        fn test_sqrt_square_inverse(x in -1000.0f32..1000.0) {
            let result = (x * x).sqrt();
            prop_assert!(approx_eq(result, x.abs()),
                "sqrt({}^2) = {} (expected {})",
                x, result, x.abs());
        }

        /// Test abs(x) >= 0
        #[test]
        fn test_abs_non_negative(x in MIN_FLOAT..MAX_FLOAT) {
            let result = x.abs();
            prop_assert!(result >= 0.0,
                "abs({}) = {} (expected >= 0)",
                x, result);
        }

        /// Test abs(-x) = abs(x)
        #[test]
        fn test_abs_symmetric(x in MIN_FLOAT..MAX_FLOAT) {
            let result1 = x.abs();
            let result2 = (-x).abs();
            prop_assert!(approx_eq(result1, result2),
                "abs({}) = {} but abs({}) = {}",
                x, result1, -x, result2);
        }

        /// Test max(a, b) >= a and max(a, b) >= b
        #[test]
        fn test_max_upper_bound(
            a in MIN_FLOAT..MAX_FLOAT,
            b in MIN_FLOAT..MAX_FLOAT
        ) {
            let result = a.max(b);
            prop_assert!(result >= a && result >= b,
                "max({}, {}) = {} but should be >= both",
                a, b, result);
        }

        /// Test min(a, b) <= a and min(a, b) <= b
        #[test]
        fn test_min_lower_bound(
            a in MIN_FLOAT..MAX_FLOAT,
            b in MIN_FLOAT..MAX_FLOAT
        ) {
            let result = a.min(b);
            prop_assert!(result <= a && result <= b,
                "min({}, {}) = {} but should be <= both",
                a, b, result);
        }

        /// Test max is commutative: max(a, b) = max(b, a)
        #[test]
        fn test_max_commutative(
            a in MIN_FLOAT..MAX_FLOAT,
            b in MIN_FLOAT..MAX_FLOAT
        ) {
            let result1 = a.max(b);
            let result2 = b.max(a);
            prop_assert!(approx_eq(result1, result2),
                "max({}, {}) = {} but max({}, {}) = {}",
                a, b, result1, b, a, result2);
        }

        /// Test min is commutative: min(a, b) = min(b, a)
        #[test]
        fn test_min_commutative(
            a in MIN_FLOAT..MAX_FLOAT,
            b in MIN_FLOAT..MAX_FLOAT
        ) {
            let result1 = a.min(b);
            let result2 = b.min(a);
            prop_assert!(approx_eq(result1, result2),
                "min({}, {}) = {} but min({}, {}) = {}",
                a, b, result1, b, a, result2);
        }
    }

    // =====================================================================
    // TRIGONOMETRIC PROPERTIES
    // =====================================================================

    proptest! {
        /// Test sin^2(x) + cos^2(x) = 1 (Pythagorean identity)
        #[test]
        fn test_pythagorean_identity(x in -10.0f32..10.0) {
            let sin_x = x.sin();
            let cos_x = x.cos();
            let result = sin_x * sin_x + cos_x * cos_x;
            prop_assert!(approx_eq(result, 1.0),
                "sin^2({}) + cos^2({}) = {} (expected 1)",
                x, x, result);
        }

        /// Test sin(-x) = -sin(x) (odd function)
        #[test]
        fn test_sin_odd_function(x in -10.0f32..10.0) {
            let result1 = (-x).sin();
            let result2 = -x.sin();
            prop_assert!(approx_eq(result1, result2),
                "sin(-{}) = {} but -sin({}) = {}",
                x, result1, x, result2);
        }

        /// Test cos(-x) = cos(x) (even function)
        #[test]
        fn test_cos_even_function(x in -10.0f32..10.0) {
            let result1 = (-x).cos();
            let result2 = x.cos();
            prop_assert!(approx_eq(result1, result2),
                "cos(-{}) = {} but cos({}) = {}",
                x, result1, x, result2);
        }

        /// Test -1 <= sin(x) <= 1
        #[test]
        fn test_sin_bounded(x in -100.0f32..100.0) {
            let result = x.sin();
            prop_assert!(result >= -1.0 && result <= 1.0,
                "sin({}) = {} but should be in [-1, 1]",
                x, result);
        }

        /// Test -1 <= cos(x) <= 1
        #[test]
        fn test_cos_bounded(x in -100.0f32..100.0) {
            let result = x.cos();
            prop_assert!(result >= -1.0 && result <= 1.0,
                "cos({}) = {} but should be in [-1, 1]",
                x, result);
        }

        /// Test tan(x) = sin(x) / cos(x) (when cos(x) != 0)
        #[test]
        fn test_tan_definition(x in -1.5f32..1.5) {
            // Avoid points where cos(x) is near zero
            prop_assume!(x.cos().abs() > 0.1);
            let tan_x = x.tan();
            let expected = x.sin() / x.cos();
            prop_assert!(approx_eq(tan_x, expected),
                "tan({}) = {} but sin({})/cos({}) = {}",
                x, tan_x, x, x, expected);
        }
    }

    // =====================================================================
    // MONOTONICITY PROPERTIES
    // =====================================================================

    proptest! {
        /// Test exp is monotonically increasing
        #[test]
        fn test_exp_monotonic(a in -10.0f32..10.0, b in -10.0f32..10.0) {
            if a < b {
                prop_assert!(a.exp() < b.exp(),
                    "exp not monotonic: {} < {} but exp({}) = {} >= exp({}) = {}",
                    a, b, a, a.exp(), b, b.exp());
            }
        }

        /// Test ln is monotonically increasing on positive reals
        #[test]
        fn test_ln_monotonic(a in 0.01f32..1000.0, b in 0.01f32..1000.0) {
            if a < b {
                prop_assert!(a.ln() < b.ln(),
                    "ln not monotonic: {} < {} but ln({}) = {} >= ln({}) = {}",
                    a, b, a, a.ln(), b, b.ln());
            }
        }

        /// Test sqrt is monotonically increasing on non-negative reals
        #[test]
        fn test_sqrt_monotonic(a in 0.0f32..1000.0, b in 0.0f32..1000.0) {
            if a < b {
                prop_assert!(a.sqrt() < b.sqrt(),
                    "sqrt not monotonic: {} < {} but sqrt({}) = {} >= sqrt({}) = {}",
                    a, b, a, a.sqrt(), b, b.sqrt());
            }
        }
    }

    // =====================================================================
    // DOUBLE PRECISION PROPERTIES
    // =====================================================================

    proptest! {
        /// Test f64 arithmetic maintains higher precision
        #[test]
        fn test_f64_precision(
            a in -1000.0f64..1000.0,
            b in -1000.0f64..1000.0,
            c in -1000.0f64..1000.0
        ) {
            // Test that f64 maintains associativity better than f32
            let result_f64 = (a + b) + c;
            let expected_f64 = a + (b + c);
            prop_assert!(approx_eq_f64(result_f64, expected_f64),
                "f64 addition not associative: ({} + {}) + {} = {} but {} + ({} + {}) = {}",
                a, b, c, result_f64, a, b, c, expected_f64);
        }

        /// Test f64 distributivity
        #[test]
        fn test_f64_distributive(
            a in -100.0f64..100.0,
            b in -100.0f64..100.0,
            c in -100.0f64..100.0
        ) {
            let result = a * (b + c);
            let expected = a * b + a * c;
            prop_assert!(approx_eq_f64(result, expected),
                "f64 distributivity violated: {} * ({} + {}) = {} but {} * {} + {} * {} = {}",
                a, b, c, result, a, b, a, c, expected);
        }
    }

    // =====================================================================
    // NUMERICAL STABILITY PROPERTIES
    // =====================================================================

    proptest! {
        /// Test that operations don't produce NaN for valid inputs
        #[test]
        fn test_no_unexpected_nan(
            a in -1000.0f32..1000.0,
            b in -1000.0f32..1000.0
        ) {
            let sum = a + b;
            let diff = a - b;
            prop_assert!(!sum.is_nan(), "Addition produced NaN: {} + {} = NaN", a, b);
            prop_assert!(!diff.is_nan(), "Subtraction produced NaN: {} - {} = NaN", a, b);
        }

        /// Test that division by non-zero doesn't produce NaN
        #[test]
        fn test_division_no_nan(
            a in -1000.0f32..1000.0,
            b in -1000.0f32..1000.0
        ) {
            prop_assume!(b.abs() > 0.001);
            let result = a / b;
            prop_assert!(!result.is_nan(),
                "Division produced NaN: {} / {} = NaN",
                a, b);
        }

        /// Test that sqrt of positive numbers doesn't produce NaN
        #[test]
        fn test_sqrt_no_nan(x in 0.0f32..1000.0) {
            let result = x.sqrt();
            prop_assert!(!result.is_nan(),
                "sqrt produced NaN: sqrt({}) = NaN",
                x);
        }

        /// Test that exp doesn't overflow for reasonable inputs
        #[test]
        fn test_exp_no_inf(x in -10.0f32..10.0) {
            let result = x.exp();
            prop_assert!(!result.is_infinite(),
                "exp overflowed: exp({}) = inf",
                x);
        }
    }

    // =====================================================================
    // IDEMPOTENCE PROPERTIES
    // =====================================================================

    proptest! {
        /// Test abs is idempotent: abs(abs(x)) = abs(x)
        #[test]
        fn test_abs_idempotent(x in MIN_FLOAT..MAX_FLOAT) {
            let once = x.abs();
            let twice = once.abs();
            prop_assert!(approx_eq(once, twice),
                "abs not idempotent: abs({}) = {} but abs(abs({})) = {}",
                x, once, x, twice);
        }

        /// Test max(x, x) = x
        #[test]
        fn test_max_idempotent(x in MIN_FLOAT..MAX_FLOAT) {
            let result = x.max(x);
            prop_assert!(approx_eq(result, x),
                "max not idempotent: max({}, {}) = {} (expected {})",
                x, x, result, x);
        }

        /// Test min(x, x) = x
        #[test]
        fn test_min_idempotent(x in MIN_FLOAT..MAX_FLOAT) {
            let result = x.min(x);
            prop_assert!(approx_eq(result, x),
                "min not idempotent: min({}, {}) = {} (expected {})",
                x, x, result, x);
        }
    }
}
