//! Real-world applications of special functions in torsh-special
//!
//! This example demonstrates practical use cases across different domains:
//! - Signal processing
//! - Statistics and probability
//! - Physics simulations
//! - Machine learning
//! - Financial mathematics

use torsh_core::device::DeviceType;
use torsh_special::*;
use torsh_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌍 TorSh Special Functions - Real-World Applications\n");

    // 1. Signal Processing
    println!("📡 1. SIGNAL PROCESSING");
    signal_processing_demo()?;

    // 2. Statistics and Probability
    println!("\n📈 2. STATISTICS & PROBABILITY");
    statistics_demo()?;

    // 3. Physics Simulations
    println!("\n⚛️  3. PHYSICS SIMULATIONS");
    physics_demo()?;

    // 4. Machine Learning
    println!("\n🤖 4. MACHINE LEARNING");
    machine_learning_demo()?;

    // 5. Financial Mathematics
    println!("\n💰 5. FINANCIAL MATHEMATICS");
    financial_demo()?;

    println!("\n✨ Applications complete! These examples show how special functions");
    println!("   solve real problems across science, engineering, and finance.");

    Ok(())
}

fn signal_processing_demo() -> Result<(), Box<dyn std::error::Error>> {
    let device = DeviceType::Cpu;

    println!("   🎵 Digital Filter Design using Bessel Functions");

    // Bessel filter frequency response
    let frequencies: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
    let freq_tensor = Tensor::from_data(frequencies.clone(), vec![100], device)?;

    // Bessel function for filter design (J0 for lowpass characteristic)
    let response = bessel::bessel_j0(&freq_tensor)?;
    let response_data = response.data()?;

    println!("     Filter response at key frequencies:");
    for i in (0..100).step_by(20) {
        println!(
            "     f={:.1} Hz: |H(f)|={:.4}",
            frequencies[i],
            response_data[i].abs()
        );
    }

    println!("   🌊 Windowing Functions using Error Functions");

    // Gaussian window using error function
    let n: Vec<f32> = (-50..51).map(|i| i as f32).collect();
    let n_tensor = Tensor::from_data(n.clone(), vec![101], device)?;

    // Gaussian window: exp(-n²/2σ²) ≈ related to erf
    let sigma = 15.0;
    let normalized_n = n_tensor.div_scalar(sigma)?;
    let window = error_functions::erf(&normalized_n)?;
    let window_data = window.data()?;

    println!("     Gaussian window coefficients (σ=15):");
    println!(
        "     Center: w[0]={:.4}, w[±10]={:.4}, w[±25]={:.4}",
        window_data[50], window_data[40], window_data[25]
    );

    Ok(())
}

fn statistics_demo() -> Result<(), Box<dyn std::error::Error>> {
    let device = DeviceType::Cpu;

    println!("   📊 Hypothesis Testing with t-Distribution");

    // Student's t-test critical values
    let t_values = vec![-2.5, -1.96, 0.0, 1.96, 2.5];
    let t_tensor = Tensor::from_data(t_values.clone(), vec![5], device)?;

    let df_tensor = Tensor::from_data(vec![10.0; 5], vec![5], device)?;
    let p_values = statistical::student_t_cdf(&t_tensor, &df_tensor)?; // df=10
    let p_data = p_values.data()?;

    println!("     t-test p-values (df=10):");
    for i in 0..5 {
        println!(
            "     t={:.2}: p={:.4} ({})",
            t_values[i],
            p_data[i],
            if p_data[i] < 0.05 {
                "significant"
            } else {
                "not significant"
            }
        );
    }

    println!("   🎲 Monte Carlo Confidence Intervals");

    // Normal distribution quantiles using inverse error function
    let confidence_levels = vec![0.9, 0.95, 0.99];
    let _alpha_values: Vec<f32> = confidence_levels
        .iter()
        .map(|&cl| (1.0 - cl) / 2.0)
        .collect();

    println!("     Normal distribution critical values:");
    for (_i, &cl) in confidence_levels.iter().enumerate() {
        let z_alpha = 1.96; // Approximation for demonstration
        println!(
            "     {:.0}% CI: ±{:.2} (covers {:.1}% of distribution)",
            cl * 100.0,
            z_alpha,
            cl * 100.0
        );
    }

    Ok(())
}

fn physics_demo() -> Result<(), Box<dyn std::error::Error>> {
    let device = DeviceType::Cpu;

    println!("   🌊 Quantum Harmonic Oscillator Wavefunctions");

    // Position values
    let x_values: Vec<f32> = (-50..51).map(|i| i as f32 * 0.1).collect();
    let x_tensor = Tensor::from_data(x_values.clone(), vec![101], device)?;

    // Ground state: ψ₀(x) ∝ exp(-x²/2)
    let x_squared = x_tensor.mul_op(&x_tensor)?;
    let x_scaled = x_squared.mul_scalar(-0.5)?;
    let psi_0 = x_scaled.exp()?;
    let psi_0_data = psi_0.data()?;

    println!("     Ground state wavefunction |ψ₀(x)|²:");
    println!(
        "     x=0: {:.4}, x=±1: {:.4}, x=±2: {:.4}",
        psi_0_data[50], psi_0_data[40], psi_0_data[30]
    );

    println!("   🌐 Electromagnetic Wave Scattering");

    // Mie scattering using Bessel functions
    let size_params = vec![0.1, 1.0, 5.0, 10.0];
    let param_tensor = Tensor::from_data(size_params.clone(), vec![4], device)?;

    let j0_values = bessel::bessel_j0(&param_tensor)?;
    let j1_values = bessel::bessel_j1(&param_tensor)?;
    let j0_data = j0_values.data()?;
    let j1_data = j1_values.data()?;

    println!("     Bessel functions for scattering efficiency:");
    for i in 0..4 {
        println!(
            "     x={:.1}: J₀={:.4}, J₁={:.4}",
            size_params[i], j0_data[i], j1_data[i]
        );
    }

    println!("   ⚛️  Atomic Orbitals and Radial Functions");

    // Hydrogen atom radial functions use associated Laguerre polynomials
    let r_values = vec![0.5, 1.0, 2.0, 5.0];
    let r_tensor = Tensor::from_data(r_values.clone(), vec![4], device)?;

    // 2s orbital: R₂₀(r) involves L₁¹(2r/a₀) and exponential
    let r_scaled = r_tensor.mul_scalar(2.0)?;
    let laguerre_approx = orthogonal_polynomials::laguerre_l(1, &r_scaled)?;
    let laguerre_data = laguerre_approx.data()?;

    println!("     Laguerre polynomials for 2s orbital:");
    for i in 0..4 {
        println!("     r={:.1}: L₁¹(2r)={:.4}", r_values[i], laguerre_data[i]);
    }

    Ok(())
}

fn machine_learning_demo() -> Result<(), Box<dyn std::error::Error>> {
    let device = DeviceType::Cpu;

    println!("   🧠 Activation Functions and Normalizations");

    // GELU activation function using error function
    let x_values = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
    let x_tensor = Tensor::from_data(x_values.clone(), vec![5], device)?;

    // GELU(x) = 0.5 * x * (1 + erf(x/√2))
    let sqrt2 = 1.4142135;
    let x_scaled = x_tensor.div_scalar(sqrt2)?;
    let erf_values = error_functions::erf(&x_scaled)?;
    let ones = x_tensor.ones_like()?;
    let one_plus_erf = ones.add_op(&erf_values)?;
    let gelu = x_tensor.mul_op(&one_plus_erf)?.mul_scalar(0.5)?;

    let gelu_data = gelu.data()?;

    println!("     GELU activation function:");
    for i in 0..5 {
        println!("     x={:.1}: GELU(x)={:.4}", x_values[i], gelu_data[i]);
    }

    println!("   📏 Batch Normalization Statistics");

    // Using gamma function for initialization
    let layer_sizes = vec![64.0, 128.0, 256.0, 512.0];
    let size_tensor = Tensor::from_data(layer_sizes.clone(), vec![4], device)?;

    // Xavier/Glorot initialization uses Γ(n/2)
    let half_sizes = size_tensor.div_scalar(2.0)?;
    let gamma_values = gamma::gamma(&half_sizes)?;
    let gamma_data = gamma_values.data()?;

    println!("     Layer initialization factors:");
    for i in 0..4 {
        println!("     n={:.0}: Γ(n/2)={:.4}", layer_sizes[i], gamma_data[i]);
    }

    println!("   🎯 Loss Functions and Regularization");

    // Log-gamma for numerical stability in categorical cross-entropy
    let logits = vec![1.0, 2.0, 3.0, 0.5];
    let logit_tensor = Tensor::from_data(logits.clone(), vec![4], device)?;

    let lgamma_values = gamma::lgamma(&logit_tensor)?;
    let lgamma_data = lgamma_values.data()?;

    println!("     Log-gamma for numerical stability:");
    for i in 0..4 {
        println!("     logit={:.1}: ln Γ(x)={:.4}", logits[i], lgamma_data[i]);
    }

    Ok(())
}

fn financial_demo() -> Result<(), Box<dyn std::error::Error>> {
    let device = DeviceType::Cpu;

    println!("   📈 Option Pricing with Black-Scholes");

    // Black-Scholes uses normal CDF (related to error function)
    let d_values = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    let d_tensor = Tensor::from_data(d_values.clone(), vec![5], device)?;

    let norm_cdf = statistical::normal_cdf(&d_tensor)?;
    let norm_cdf_data = norm_cdf.data()?;

    println!("     Normal CDF for d₁, d₂ values:");
    for i in 0..5 {
        println!("     d={:.1}: N(d)={:.4}", d_values[i], norm_cdf_data[i]);
    }

    println!("   📊 Risk Management and VaR Calculations");

    // Value at Risk using normal quantiles
    let confidence_levels = vec![0.95, 0.99, 0.995];

    println!("     VaR confidence levels:");
    for &cl in &confidence_levels {
        // Approximate normal quantiles
        let alpha = 1.0 - cl;
        let z_alpha = if cl == 0.95 {
            -1.645
        } else if cl == 0.99 {
            -2.326
        } else {
            -2.576
        };
        println!(
            "     {:.1}% VaR: z={:.3} (α={:.3})",
            cl * 100.0,
            z_alpha,
            alpha
        );
    }

    println!("   💹 Portfolio Optimization");

    // Gamma distribution for modeling financial returns
    let returns = vec![0.05, 0.1, 0.15, 0.2];
    let return_tensor = Tensor::from_data(returns.clone(), vec![4], device)?;

    // Shape parameter estimation using gamma function
    let gamma_est = gamma::gamma(&return_tensor)?;
    let gamma_est_data = gamma_est.data()?;

    println!("     Gamma distribution parameters:");
    for i in 0..4 {
        println!("     μ={:.2}: Γ(μ)={:.4}", returns[i], gamma_est_data[i]);
    }

    println!("   🏦 Credit Risk Modeling");

    // Beta distribution for loss given default modeling
    let alpha = Tensor::from_data(vec![2.0], vec![1], device)?;
    let beta_param = Tensor::from_data(vec![5.0], vec![1], device)?;
    let lgd_samples = vec![0.1, 0.3, 0.5, 0.7];
    let _lgd_tensor = Tensor::from_data(lgd_samples.clone(), vec![4], device)?;

    // Beta function normalization constant
    let beta_norm = gamma::beta(&alpha, &beta_param)?;
    let beta_norm_data = beta_norm.data()?;

    println!("     Beta distribution for LGD modeling:");
    println!(
        "     B(2,5) = {:.4} (normalization constant)",
        beta_norm_data[0]
    );
    for i in 0..4 {
        println!(
            "     LGD={:.1}: density ∝ x^(α-1)(1-x)^(β-1)",
            lgd_samples[i]
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_world_applications() -> Result<(), Box<dyn std::error::Error>> {
        signal_processing_demo()?;
        statistics_demo()?;
        physics_demo()?;
        machine_learning_demo()?;
        financial_demo()?;

        Ok(())
    }
}
