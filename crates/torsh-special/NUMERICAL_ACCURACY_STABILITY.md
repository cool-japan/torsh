# Numerical Accuracy and Stability Documentation

## Overview

This document provides comprehensive information about the numerical accuracy guarantees, stability analysis, and implementation details for all special functions in the `torsh-special` crate.

## Accuracy Specifications

### General Accuracy Guarantees

| Function Category | Standard Accuracy | Fast Approximations | SIMD Optimized | Cached Values |
|------------------|-------------------|-------------------|----------------|---------------|
| Gamma Functions | 1e-12 (f64), 1e-6 (f32) | 1e-3 to 1e-4 | 1e-10 (f64) | Full precision |
| Bessel Functions | 1e-10 to 1e-12 | 1e-2 to 1e-3 | 1e-10 | Full precision |
| Error Functions | 1e-12 (f64), 1e-6 (f32) | 1e-3 to 1e-4 | 1e-11 | Full precision |
| Elliptic Functions | 1e-10 to 1e-12 | 1e-3 | 1e-10 | Full precision |
| Complex Functions | 1e-10 to 1e-12 | N/A | N/A | Full precision |

### Function-Specific Accuracy

#### Gamma Functions
- **`gamma(x)`**: Machine precision for x > 0, reflection formula accuracy for x < 0
- **`lgamma(x)`**: Better than 1e-12 for all finite x
- **`digamma(x)`**: Accuracy degrades near negative integers (as expected mathematically)
- **`polygamma(n, x)`**: Accuracy decreases with increasing order n

#### Bessel Functions
- **Regular Bessel (J₀, J₁, Jₙ)**: 1e-10 to 1e-12 accuracy across all domains
- **Neumann Functions (Y₀, Y₁, Yₙ)**: 1e-10 accuracy, may degrade near zero
- **Modified Bessel (I₀, I₁, Iₙ)**: 1e-10 accuracy, overflow protection for large arguments
- **Modified Bessel (K₀, K₁, Kₙ)**: 1e-10 accuracy, underflow protection for large arguments

#### Error Functions
- **`erf(x)`**: Machine precision for |x| < 10, asymptotic accuracy for large |x|
- **`erfc(x)`**: Maintained precision even for large positive x using continued fractions
- **`erfcx(x)`**: Scaled version maintains accuracy for very large x
- **`erfinv(x)`**: Accuracy degrades near ±1 (mathematical limitation)

## Numerical Stability Analysis

### Condition Numbers and Sensitivity

#### Well-Conditioned Functions
These functions have low condition numbers and are numerically stable:
- `erf(x)` for moderate |x|
- `gamma(x)` for x > 1
- `bessel_j0(x)`, `bessel_j1(x)` for moderate x
- `sin(x)`, `cos(x)` for moderate x

#### Ill-Conditioned Regions
Special care is taken in these mathematically challenging regions:

1. **Near Poles and Singularities**
   - `gamma(x)` near negative integers: Uses reflection formula
   - `digamma(x)` near negative integers: Series expansion with careful cancellation
   - `bessel_yn(x)` near x=0: Uses recurrence relations

2. **Large Argument Behavior**
   - Asymptotic expansions for `gamma(x)` when |x| > 50
   - Exponentially scaled functions (`erfcx`, `bessel_ive`) for overflow prevention
   - Continued fractions for `erfc(x)` when x > 8

3. **Small Argument Expansions**
   - Taylor series for functions near x=0
   - Special handling for `log(gamma(x))` near x=1,2

### Overflow and Underflow Protection

#### Overflow Prevention
- **Gamma functions**: Use `lgamma` when `gamma` would overflow
- **Exponential functions**: Automatic scaling and logarithmic computation
- **Modified Bessel**: Switch to exponentially scaled versions for large arguments

#### Underflow Handling
- **Small results**: Gradual underflow to zero instead of sudden cutoff
- **Denormal numbers**: Proper handling of subnormal floating-point values
- **Relative error maintenance**: Preserve relative accuracy even for tiny results

## Implementation Strategies

### Algorithmic Approaches

#### Polynomial Approximations
- **Chebyshev polynomials**: Used for smooth functions in bounded intervals
- **Padé approximants**: Rational functions for better asymptotic behavior
- **Minimax polynomials**: Optimal uniform approximation for critical ranges

#### Series Expansions
- **Taylor series**: For small argument expansions
- **Asymptotic series**: For large argument behavior
- **Convergent series**: For intermediate ranges

#### Special Techniques
- **Continued fractions**: For `erfc`, incomplete gamma functions
- **Recurrence relations**: For Bessel functions, orthogonal polynomials
- **Reflection formulas**: For extending domains (gamma function)
- **Duplication formulas**: For argument reduction

### Domain Decomposition

Most functions use domain-specific algorithms:

```rust
fn optimized_function(x: f64) -> f64 {
    if x.abs() < SMALL_THRESHOLD {
        small_argument_series(x)
    } else if x.abs() > LARGE_THRESHOLD {
        asymptotic_expansion(x)
    } else {
        intermediate_algorithm(x)
    }
}
```

### Error Analysis Framework

#### Forward Error Analysis
- Bounds on absolute error: |f_computed(x) - f_true(x)|
- Bounds on relative error: |f_computed(x) - f_true(x)| / |f_true(x)|

#### Backward Error Analysis
- Input perturbation that explains the computed result
- Condition number estimation for stability assessment

## Testing and Validation

### Test Coverage

#### Reference Values
- **NIST Digital Library**: High-precision reference values
- **Mathematica/Maple**: Independent verification for edge cases
- **Published tables**: Classical mathematical handbooks

#### Test Categories
1. **Basic functionality tests**: Verify correct implementation
2. **Accuracy tests**: Compare against high-precision references
3. **Edge case tests**: Boundary conditions, special values
4. **Stress tests**: Extreme arguments, pathological cases

#### Continuous Integration
- Automated testing across different platforms
- Regression detection for numerical accuracy
- Performance benchmarking for optimization validation

### Accuracy Measurement

#### Error Metrics
- **Absolute error**: `|computed - reference|`
- **Relative error**: `|computed - reference| / |reference|`
- **ULP error**: Error in units of last place

#### Statistical Analysis
- **Mean accuracy**: Average error across test sets
- **Worst-case accuracy**: Maximum observed error
- **Distribution of errors**: Understanding error patterns

## Optimization Trade-offs

### Accuracy vs Performance

| Optimization Level | Relative Performance | Typical Accuracy | Use Case |
|-------------------|---------------------|------------------|----------|
| Standard | 1.0x (baseline) | 1e-12 | Scientific computing |
| SIMD Optimized | 2-4x faster | 1e-10 | Large tensor operations |
| Fast Approximations | 5-10x faster | 1e-3 to 1e-4 | Real-time applications |
| Smart Cached | Variable speedup | Full precision | Repeated computations |

### Memory vs Accuracy Trade-offs

#### Lookup Tables
- **Advantages**: Constant-time access, perfect accuracy for tabulated values
- **Disadvantages**: Memory usage, interpolation errors between points
- **Implementation**: Strategic use for frequently accessed values

#### Precomputed Coefficients
- **Polynomial coefficients**: Stored for different domains
- **Rational function coefficients**: Numerator and denominator polynomials
- **Asymptotic coefficients**: For large-argument expansions

## Error Handling and Edge Cases

### Input Validation

#### Domain Checking
```rust
fn validated_function(x: f64) -> Result<f64, Error> {
    if !x.is_finite() {
        return Err(Error::InvalidInput("Non-finite input"));
    }
    if x < DOMAIN_MIN || x > DOMAIN_MAX {
        return Err(Error::OutOfDomain);
    }
    Ok(unsafe_implementation(x))
}
```

#### Special Value Handling
- **NaN propagation**: Consistent NaN handling across all functions
- **Infinity handling**: Mathematically correct limits
- **Zero handling**: Proper signed zero behavior

### Error Recovery

#### Graceful Degradation
- **Reduced precision**: Fall back to lower accuracy when high precision fails
- **Alternative algorithms**: Switch methods when primary algorithm fails
- **Approximation fallbacks**: Use fast approximations as last resort

#### Error Reporting
- **Detailed error messages**: Specific information about failure modes
- **Context preservation**: Maintain information about failed inputs
- **Recovery suggestions**: Guidance for user corrections

## Platform-Specific Considerations

### Floating-Point Standards

#### IEEE 754 Compliance
- **Rounding modes**: Consistent behavior across platforms
- **Exception handling**: Proper handling of overflow, underflow, invalid operations
- **Denormal numbers**: Correct processing of subnormal values

#### Platform Differences
- **x86 vs ARM**: Different floating-point implementations
- **32-bit vs 64-bit**: Precision and range differences
- **Compiler optimizations**: Ensuring mathematical correctness preservation

### SIMD Optimizations

#### Vector Instructions
- **AVX2/AVX-512**: Modern x86 vector processing
- **NEON**: ARM vector instructions
- **Accuracy preservation**: Maintaining precision in vectorized operations

#### Parallel Processing
- **Thread safety**: All functions are thread-safe and reentrant
- **Memory ordering**: Proper synchronization for cached values
- **Reproducibility**: Consistent results across parallel executions

## Usage Guidelines

### When to Use Each Optimization Level

#### Standard Functions (Default)
Use for:
- Scientific computing requiring maximum accuracy
- Financial calculations where precision is critical
- Small to medium datasets where performance is not critical

#### SIMD Optimized Functions
Use for:
- Large tensor operations (>1000 elements)
- Machine learning inference pipelines
- Signal processing applications

#### Fast Approximations
Use for:
- Real-time applications where speed is critical
- Graphics and gaming applications
- Situations where 0.1% error is acceptable

#### Smart Caching
Use for:
- Repeated computations with same inputs
- Monte Carlo simulations
- Iterative algorithms with function reevaluation

### Best Practices

#### Input Preprocessing
- **Argument reduction**: Use mathematical identities to map inputs to well-conditioned domains
- **Range checking**: Validate inputs before computation
- **Special case handling**: Check for exact mathematical special values

#### Result Postprocessing
- **Range validation**: Ensure results are within expected mathematical bounds
- **Consistency checking**: Verify mathematical relationships between related functions
- **Error estimation**: Provide accuracy estimates when possible

## Conclusion

The `torsh-special` crate provides industrial-strength implementations of special mathematical functions with:

- **Comprehensive accuracy guarantees** across all supported functions
- **Robust numerical stability** through careful algorithm selection
- **Flexible optimization levels** balancing accuracy and performance
- **Extensive testing and validation** ensuring reliability
- **Clear documentation** of limitations and trade-offs

This foundation enables confident use in production applications ranging from high-precision scientific computing to real-time numerical processing.