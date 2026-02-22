//! Spheroidal Wave Functions Example
//!
//! This example demonstrates the use of spheroidal wave functions for:
//! - Electromagnetic scattering from prolate spheroids
//! - Acoustic wave propagation in spheroidal geometries
//! - Eigenvalue computation for spheroidal wave equations
//!
//! Run with: cargo run --example spheroidal_wave_functions

use torsh_special::{
    oblate_angular, oblate_radial, prolate_angular, prolate_radial, spheroidal_eigenvalue,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Spheroidal Wave Functions Demonstration ===\n");

    // Example 1: Electromagnetic Scattering from a Prolate Spheroid
    println!("1. ELECTROMAGNETIC SCATTERING FROM PROLATE SPHEROID");
    println!("   Scenario: Radio wave scattering from a prolate spheroidal antenna");
    println!("   Frequency: 1 GHz, Spheroid semi-major axis: 1 m\n");

    // Parameters for prolate spheroid
    let n = 2; // Degree
    let m = 0; // Order (azimuthally symmetric)
    let c = 2.0; // Spheroidicity parameter (related to k*d where k is wavenumber, d is interfocal distance)

    println!("   Computing angular distribution (θ dependence):");
    let angles: Vec<f64> = vec![0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0];
    for &theta_deg in &angles {
        let theta = theta_deg.to_radians();
        let eta = theta.cos(); // η = cos(θ)

        match prolate_angular(n, m, c, eta) {
            Ok(s_value) => {
                println!(
                    "     θ = {:3}°  →  S_{}^{}(c={}, η={:.3}) = {:.6}",
                    theta_deg, n, m, c, eta, s_value
                );
            }
            Err(e) => println!("     Error at θ = {}°: {}", theta_deg, e),
        }
    }

    println!("\n   Computing radial distribution (ξ dependence):");
    let radii: Vec<f64> = vec![1.0, 1.5, 2.0, 3.0, 5.0];
    for &xi in &radii {
        match prolate_radial(n, m, c, xi) {
            Ok(r_value) => {
                println!(
                    "     ξ = {:.1}  →  R_{}^{}(c={}, ξ={:.1}) = {:.6}",
                    xi, n, m, c, xi, r_value
                );
            }
            Err(e) => println!("     Error at ξ = {}: {}", xi, e),
        }
    }

    // Example 2: Acoustic Wave Propagation in Oblate Cavity
    println!("\n2. ACOUSTIC WAVE PROPAGATION IN OBLATE CAVITY");
    println!("   Scenario: Sound wave resonance in an oblate spheroidal cavity");
    println!("   Application: Musical instrument acoustics\n");

    let n_acoustic = 3;
    let m_acoustic = 1;
    let c_acoustic = 1.5;

    println!("   Angular modes at different positions:");
    let eta_values: Vec<f64> = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
    for &eta in &eta_values {
        match oblate_angular(n_acoustic, m_acoustic, c_acoustic, eta) {
            Ok(s_value) => {
                println!(
                    "     η = {:5.2}  →  S_{}^{}(c={}, η) = {:.6}",
                    eta, n_acoustic, m_acoustic, c_acoustic, s_value
                );
            }
            Err(e) => println!("     Error at η = {}: {}", eta, e),
        }
    }

    println!("\n   Radial modes (oblate coordinates):");
    let xi_oblate: Vec<f64> = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    for &xi in &xi_oblate {
        match oblate_radial(n_acoustic, m_acoustic, c_acoustic, xi) {
            Ok(r_value) => {
                println!(
                    "     ξ = {:.2}  →  R_{}^{}(c={}, ξ) = {:.6}",
                    xi, n_acoustic, m_acoustic, c_acoustic, r_value
                );
            }
            Err(e) => println!("     Error at ξ = {}: {}", xi, e),
        }
    }

    // Example 3: Eigenvalue Computation for Different Modes
    println!("\n3. EIGENVALUE COMPUTATION FOR SPHEROIDAL MODES");
    println!("   Computing eigenvalues λ_nm(c) for various modes\n");

    println!("   For c = 0 (spherical limit):");
    println!("   ┌──────┬──────┬──────────┬────────────┬──────────────┐");
    println!("   │  n   │  m   │  λ(c=0)  │  Expected  │ Verification │");
    println!("   ├──────┼──────┼──────────┼────────────┼──────────────┤");

    let modes = vec![(0, 0), (1, 0), (2, 0), (2, 1), (3, 0), (3, 2)];
    for &(n, m) in &modes {
        match spheroidal_eigenvalue(n, m, 0.0) {
            Ok(lambda) => {
                let expected = (n * (n + 1)) as f64;
                let matches = (lambda - expected).abs() < 1e-10;
                let check = if matches { "✓" } else { "✗" };
                println!(
                    "   │  {}   │  {}   │ {:8.3}  │  {:8.3}  │      {}       │",
                    n, m, lambda, expected, check
                );
            }
            Err(e) => println!("   │  {}   │  {}   │  Error: {}  │", n, m, e),
        }
    }
    println!("   └──────┴──────┴──────────┴────────────┴──────────────┘");

    println!("\n   For c = 1.0 (perturbation from spherical):");
    println!("   ┌──────┬──────┬───────────┬──────────────┐");
    println!("   │  n   │  m   │  λ(c=1)   │  λ(c=0)      │");
    println!("   ├──────┼──────┼───────────┼──────────────┤");

    for &(n, m) in &modes {
        match (
            spheroidal_eigenvalue(n, m, 1.0),
            spheroidal_eigenvalue(n, m, 0.0),
        ) {
            (Ok(lambda_1), Ok(lambda_0)) => {
                println!(
                    "   │  {}   │  {}   │ {:9.4}  │  {:9.4}  │",
                    n, m, lambda_1, lambda_0
                );
            }
            _ => println!("   │  {}   │  {}   │   Error   │", n, m),
        }
    }
    println!("   └──────┴──────┴───────────┴──────────────┘");

    // Example 4: Application - Antenna Radiation Pattern
    println!("\n4. ANTENNA RADIATION PATTERN ANALYSIS");
    println!("   Prolate spheroidal antenna with aspect ratio 3:1");
    println!("   Computing normalized radiation pattern\n");

    let n_antenna = 1;
    let m_antenna = 0;
    let c_antenna = 3.0;

    println!("   Radiation pattern (normalized to maximum):");
    println!("   ┌──────────┬────────────┬──────────────┐");
    println!("   │ Angle θ  │  S_nm(η)   │  Normalized  │");
    println!("   ├──────────┼────────────┼──────────────┤");

    let mut max_value = 0.0f64;
    let pattern_angles: Vec<f64> = (0..=180).step_by(15).map(|x| x as f64).collect();
    let pattern_values: Vec<(f64, f64)> = pattern_angles
        .iter()
        .filter_map(|&angle| {
            let eta = angle.to_radians().cos();
            prolate_angular(n_antenna, m_antenna, c_antenna, eta)
                .ok()
                .map(|val| {
                    max_value = max_value.max(val.abs());
                    (angle, val)
                })
        })
        .collect();

    for (angle, value) in &pattern_values {
        let normalized = if max_value > 0.0 {
            value / max_value
        } else {
            0.0
        };
        let bar_length = (normalized.abs() * 20.0) as usize;
        let bar = "█".repeat(bar_length);
        println!(
            "   │  {:5.0}°   │ {:9.5}  │ {:<20} │ {:.2}",
            angle, value, bar, normalized
        );
    }
    println!("   └──────────┴────────────┴──────────────┘");

    // Example 5: Wave Scattering Cross Section
    println!("\n5. SCATTERING CROSS SECTION COMPUTATION");
    println!("   Computing relative scattering efficiency for different c values\n");

    println!("   ┌────────┬─────────────┬──────────────────────┐");
    println!("   │   c    │  λ_20(c)    │  Scattering Weight   │");
    println!("   ├────────┼─────────────┼──────────────────────┤");

    let c_values: Vec<f64> = vec![0.0, 0.5, 1.0, 2.0, 3.0, 5.0];
    for &c in &c_values {
        match spheroidal_eigenvalue(2, 0, c) {
            Ok(lambda) => {
                // Scattering weight is related to eigenvalue perturbation
                let lambda_0 = 6.0; // n(n+1) for n=2
                let weight = (lambda - lambda_0).abs();
                let bar_length = (weight * 5.0).min(20.0) as usize;
                let bar = "▓".repeat(bar_length);
                println!(
                    "   │ {:6.2} │ {:10.5}  │ {:<20} │ {:.4}",
                    c, lambda, bar, weight
                );
            }
            Err(e) => println!("   │ {:6.2} │   Error: {}  │", c, e),
        }
    }
    println!("   └────────┴─────────────┴──────────────────────┘");

    // Example 6: Comparison of Prolate vs Oblate Modes
    println!("\n6. PROLATE VS OBLATE MODE COMPARISON");
    println!("   Comparing angular functions at η = 0.5 (45° from axis)\n");

    let eta_compare = 0.5;
    println!("   ┌─────┬─────┬────────────────┬────────────────┐");
    println!("   │  n  │  m  │  Prolate S_nm  │  Oblate S_nm   │");
    println!("   ├─────┼─────┼────────────────┼────────────────┤");

    let compare_modes = vec![(1, 0), (1, 1), (2, 0), (2, 1), (3, 0)];
    for &(n, m) in &compare_modes {
        match (
            prolate_angular(n, m, 2.0, eta_compare),
            oblate_angular(n, m, 2.0, eta_compare),
        ) {
            (Ok(prolate), Ok(oblate)) => {
                println!(
                    "   │  {}  │  {}  │  {:12.6}  │  {:12.6}  │",
                    n, m, prolate, oblate
                );
            }
            _ => println!("   │  {}  │  {}  │     Error      │", n, m),
        }
    }
    println!("   └─────┴─────┴────────────────┴────────────────┘");

    // Summary and Applications
    println!("\n=== APPLICATIONS SUMMARY ===");
    println!("Spheroidal wave functions are essential for:");
    println!("  • Electromagnetic scattering: radar cross sections, antenna design");
    println!("  • Acoustics: resonance in ellipsoidal cavities, musical instruments");
    println!("  • Quantum mechanics: molecular orbitals, diatomic molecules");
    println!("  • Astrophysics: gravitational wave propagation");
    println!("  • Geophysics: seismic wave propagation in Earth's ellipsoidal shape");

    println!("\n=== NUMERICAL CHARACTERISTICS ===");
    println!("  • Series expansions valid for |c| < 5");
    println!("  • Asymptotic approximations for |c| ≥ 5");
    println!("  • Eigenvalues computed via perturbation theory");
    println!("  • Proper handling of boundary conditions at η = ±1, ξ = 1");

    Ok(())
}
