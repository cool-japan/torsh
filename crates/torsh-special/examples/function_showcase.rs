//! Showcase of all major function categories in torsh-special
//!
//! This example demonstrates the breadth of special functions available,
//! organized by mathematical category.

use torsh_core::device::DeviceType;
use torsh_special::*;
use torsh_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎯 TorSh Special Functions - Function Showcase\n");

    let device = DeviceType::Cpu;
    let x = Tensor::from_data(vec![0.5, 1.0, 2.0], vec![3], device)?;

    println!("📊 Input values: x = [0.5, 1.0, 2.0]\n");

    // 1. Gamma Functions
    println!("🔬 1. GAMMA FUNCTIONS");
    showcase_gamma_functions(&x)?;

    // 2. Error Functions
    println!("\n📈 2. ERROR FUNCTIONS");
    showcase_error_functions(&x)?;

    // 3. Bessel Functions
    println!("\n🌀 3. BESSEL FUNCTIONS");
    showcase_bessel_functions(&x)?;

    // 4. Elliptic Functions
    println!("\n🔄 4. ELLIPTIC FUNCTIONS");
    showcase_elliptic_functions(&x)?;

    // 5. Trigonometric Functions
    println!("\n📐 5. SPECIAL TRIGONOMETRIC");
    showcase_trigonometric_functions(&x)?;

    // 6. Statistical Functions
    println!("\n📊 6. STATISTICAL FUNCTIONS");
    showcase_statistical_functions(&x)?;

    // 7. Complex Functions
    println!("\n🔢 7. COMPLEX FUNCTIONS");
    showcase_complex_functions()?;

    println!("\n✨ Complete! All function categories demonstrated.");
    println!("   See documentation for detailed parameter information.");

    Ok(())
}

fn showcase_gamma_functions(x: &Tensor<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let gamma_result = gamma::gamma(x)?;
    let lgamma_result = gamma::lgamma(x)?;
    let digamma_result = gamma::digamma(x)?;

    let gamma_data = gamma_result.data()?;
    let lgamma_data = lgamma_result.data()?;
    let digamma_data = digamma_result.data()?;

    println!(
        "  Γ(x)     = [{:.4}, {:.4}, {:.4}]",
        gamma_data[0], gamma_data[1], gamma_data[2]
    );
    println!(
        "  ln Γ(x)  = [{:.4}, {:.4}, {:.4}]",
        lgamma_data[0], lgamma_data[1], lgamma_data[2]
    );
    println!(
        "  ψ(x)     = [{:.4}, {:.4}, {:.4}]",
        digamma_data[0], digamma_data[1], digamma_data[2]
    );

    // Beta function example
    let a = Tensor::from_data(vec![1.0], vec![1], x.device())?;
    let b = Tensor::from_data(vec![2.0], vec![1], x.device())?;
    let beta_result = gamma::beta(&a, &b)?;
    let beta_data = beta_result.data()?;
    println!("  B(1,2)   = {:.4}", beta_data[0]);

    Ok(())
}

fn showcase_error_functions(x: &Tensor<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let erf_result = error_functions::erf(x)?;
    let erfc_result = error_functions::erfc(x)?;
    let erfcx_result = scirs2_integration::erfcx(x)?;

    let erf_data = erf_result.data()?;
    let erfc_data = erfc_result.data()?;
    let erfcx_data = erfcx_result.data()?;

    println!(
        "  erf(x)   = [{:.4}, {:.4}, {:.4}]",
        erf_data[0], erf_data[1], erf_data[2]
    );
    println!(
        "  erfc(x)  = [{:.4}, {:.4}, {:.4}]",
        erfc_data[0], erfc_data[1], erfc_data[2]
    );
    println!(
        "  erfcx(x) = [{:.4}, {:.4}, {:.4}]",
        erfcx_data[0], erfcx_data[1], erfcx_data[2]
    );

    // Fresnel integrals
    let fresnel_s = error_functions::fresnel_s(x)?;
    let fresnel_c = error_functions::fresnel_c(x)?;
    let fs_data = fresnel_s.data()?;
    let fc_data = fresnel_c.data()?;

    println!(
        "  S(x)     = [{:.4}, {:.4}, {:.4}]",
        fs_data[0], fs_data[1], fs_data[2]
    );
    println!(
        "  C(x)     = [{:.4}, {:.4}, {:.4}]",
        fc_data[0], fc_data[1], fc_data[2]
    );

    Ok(())
}

fn showcase_bessel_functions(x: &Tensor<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let j0_result = bessel::bessel_j0(x)?;
    let j1_result = bessel::bessel_j1(x)?;
    let y0_result = bessel::bessel_y0(x)?;
    let i0_result = bessel::bessel_i0(x)?;
    let k0_result = bessel::bessel_k0(x)?;

    let j0_data = j0_result.data()?;
    let j1_data = j1_result.data()?;
    let y0_data = y0_result.data()?;
    let i0_data = i0_result.data()?;
    let k0_data = k0_result.data()?;

    println!(
        "  J₀(x)    = [{:.4}, {:.4}, {:.4}]",
        j0_data[0], j0_data[1], j0_data[2]
    );
    println!(
        "  J₁(x)    = [{:.4}, {:.4}, {:.4}]",
        j1_data[0], j1_data[1], j1_data[2]
    );
    println!(
        "  Y₀(x)    = [{:.4}, {:.4}, {:.4}]",
        y0_data[0], y0_data[1], y0_data[2]
    );
    println!(
        "  I₀(x)    = [{:.4}, {:.4}, {:.4}]",
        i0_data[0], i0_data[1], i0_data[2]
    );
    println!(
        "  K₀(x)    = [{:.4}, {:.4}, {:.4}]",
        k0_data[0], k0_data[1], k0_data[2]
    );

    // Higher order example
    let j2_result = bessel::bessel_jn(2, x)?;
    let j2_data = j2_result.data()?;
    println!(
        "  J₂(x)    = [{:.4}, {:.4}, {:.4}]",
        j2_data[0], j2_data[1], j2_data[2]
    );

    Ok(())
}

fn showcase_elliptic_functions(x: &Tensor<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let k_result = elliptic::elliptic_k(x)?;
    let e_result = elliptic::elliptic_e(x)?;

    let k_data = k_result.data()?;
    let e_data = e_result.data()?;

    println!(
        "  K(k)     = [{:.4}, {:.4}, {:.4}]",
        k_data[0], k_data[1], k_data[2]
    );
    println!(
        "  E(k)     = [{:.4}, {:.4}, {:.4}]",
        e_data[0], e_data[1], e_data[2]
    );

    // Jacobi functions (for k < 1)
    let k_valid = Tensor::from_data(vec![0.5], vec![1], x.device())?;
    let sn_result = elliptic::jacobi_sn(&k_valid, &k_valid)?;
    let cn_result = elliptic::jacobi_cn(&k_valid, &k_valid)?;
    let dn_result = elliptic::jacobi_dn(&k_valid, &k_valid)?;

    let sn_data = sn_result.data()?;
    let cn_data = cn_result.data()?;
    let dn_data = dn_result.data()?;

    println!("  sn(0.5,0.5) = {:.4}", sn_data[0]);
    println!("  cn(0.5,0.5) = {:.4}", cn_data[0]);
    println!("  dn(0.5,0.5) = {:.4}", dn_data[0]);

    Ok(())
}

fn showcase_trigonometric_functions(x: &Tensor<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let sinc_result = trigonometric::sinc(x)?;
    let sinc_unnorm_result = trigonometric::sinc_unnormalized(x)?;

    let sinc_data = sinc_result.data()?;
    let sinc_unnorm_data = sinc_unnorm_result.data()?;

    println!(
        "  sinc(x)        = [{:.4}, {:.4}, {:.4}]",
        sinc_data[0], sinc_data[1], sinc_data[2]
    );
    println!(
        "  sinc_unnorm(x) = [{:.4}, {:.4}, {:.4}]",
        sinc_unnorm_data[0], sinc_unnorm_data[1], sinc_unnorm_data[2]
    );

    // Spherical Bessel functions
    let sph_j0_result = trigonometric::spherical_j0(x)?;
    let sph_y0_result = trigonometric::spherical_y0(x)?;

    let sph_j0_data = sph_j0_result.data()?;
    let sph_y0_data = sph_y0_result.data()?;

    println!(
        "  j₀(x)          = [{:.4}, {:.4}, {:.4}]",
        sph_j0_data[0], sph_j0_data[1], sph_j0_data[2]
    );
    println!(
        "  y₀(x)          = [{:.4}, {:.4}, {:.4}]",
        sph_y0_data[0], sph_y0_data[1], sph_y0_data[2]
    );

    Ok(())
}

fn showcase_statistical_functions(x: &Tensor<f32>) -> Result<(), Box<dyn std::error::Error>> {
    let normal_cdf_result = statistical::normal_cdf(x)?;
    let normal_pdf_result = statistical::normal_pdf(x)?;

    let normal_cdf_data = normal_cdf_result.data()?;
    let normal_pdf_data = normal_pdf_result.data()?;

    println!(
        "  Φ(x)     = [{:.4}, {:.4}, {:.4}] (normal CDF)",
        normal_cdf_data[0], normal_cdf_data[1], normal_cdf_data[2]
    );
    println!(
        "  φ(x)     = [{:.4}, {:.4}, {:.4}] (normal PDF)",
        normal_pdf_data[0], normal_pdf_data[1], normal_pdf_data[2]
    );

    // Incomplete beta (for valid parameters)
    let a = Tensor::from_data(vec![2.0], vec![1], x.device())?;
    let b = Tensor::from_data(vec![3.0], vec![1], x.device())?;
    let p = Tensor::from_data(vec![0.5], vec![1], x.device())?;
    let inc_beta_result = statistical::incomplete_beta(&a, &b, &p)?;
    let inc_beta_data = inc_beta_result.data()?;

    println!("  I₀.₅(2,3) = {:.4} (incomplete beta)", inc_beta_data[0]);

    Ok(())
}

fn showcase_complex_functions() -> Result<(), Box<dyn std::error::Error>> {
    use torsh_core::dtype::Complex32;
    use torsh_special::complex::*;

    println!("  Complex number examples (z = 1 + i):");

    let device = DeviceType::Cpu;
    let z = Complex32::new(1.0, 1.0);
    let z_tensor = Tensor::from_data(vec![z], vec![1], device)?;

    let gamma_z = complex_gamma_c32(&z_tensor)?;
    let erf_z = complex_erf_c32(&z_tensor)?;

    let gamma_data = gamma_z.data()?;
    let erf_data = erf_z.data()?;

    println!(
        "  Γ(1+i)   = {:.4} + {:.4}i",
        gamma_data[0].re, gamma_data[0].im
    );
    println!(
        "  erf(1+i) = {:.4} + {:.4}i",
        erf_data[0].re, erf_data[0].im
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_showcase_functions() -> Result<(), Box<dyn std::error::Error>> {
        let device = DeviceType::Cpu;
        let x = Tensor::from_data(vec![0.5, 1.0, 2.0], vec![3], device)?;

        showcase_gamma_functions(&x)?;
        showcase_error_functions(&x)?;
        showcase_bessel_functions(&x)?;
        showcase_elliptic_functions(&x)?;
        showcase_trigonometric_functions(&x)?;
        showcase_statistical_functions(&x)?;
        showcase_complex_functions()?;

        Ok(())
    }
}
