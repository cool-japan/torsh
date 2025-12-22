//! Comprehensive demonstration of Session 7 metrics enhancements
//!
//! This example showcases:
//! - Time series forecasting metrics
//! - Regression diagnostics
//! - Explainability metrics
//! - Robustness metrics
//!
//! Run with: cargo run --example comprehensive_diagnostics_demo

use torsh_core::DeviceType;
use torsh_metrics::*;
use torsh_tensor::creation::from_vec;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  ToRSh Metrics - Session 7 Features Demonstration            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // 1. TIME SERIES METRICS DEMONSTRATION
    // ========================================================================
    println!("┌───────────────────────────────────────────────────────────────┐");
    println!("│ 1. TIME SERIES FORECASTING METRICS                            │");
    println!("└───────────────────────────────────────────────────────────────┘");

    // Simulated time series: actual vs forecasted sales
    let y_true = from_vec(
        vec![100.0, 110.0, 105.0, 115.0, 120.0, 118.0, 125.0, 130.0],
        &[8],
        DeviceType::Cpu,
    )?;
    let y_pred = from_vec(
        vec![98.0, 112.0, 103.0, 117.0, 119.0, 120.0, 124.0, 132.0],
        &[8],
        DeviceType::Cpu,
    )?;
    let y_train = from_vec(vec![90.0, 95.0, 92.0, 98.0, 100.0], &[5], DeviceType::Cpu)?;

    // MASE - Mean Absolute Scaled Error
    let mase_score = mase(&y_true, &y_pred, Some(&y_train), 1)?;
    println!("  MASE (Mean Absolute Scaled Error):     {:.4}", mase_score);
    println!("    └─ Values < 1: Better than naive forecast");

    // SMAPE - Symmetric Mean Absolute Percentage Error
    let smape_score = smape(&y_true, &y_pred)?;
    println!(
        "  SMAPE (Symmetric MAPE):                 {:.2}%",
        smape_score
    );

    // Theil's U Statistic
    let theil_score = theil_u(&y_true, &y_pred)?;
    println!(
        "  Theil's U Statistic:                    {:.4}",
        theil_score
    );
    println!("    └─ Values < 1: Better than no-change forecast");

    // Mean Directional Accuracy
    let mda_score = mean_directional_accuracy(&y_true, &y_pred)?;
    println!(
        "  Mean Directional Accuracy:              {:.2}%",
        mda_score
    );

    // DTW Distance
    let dtw_dist = dtw_distance(&y_true, &y_pred)?;
    println!("  DTW Distance (similarity measure):      {:.4}", dtw_dist);

    // ========================================================================
    // 2. REGRESSION DIAGNOSTICS DEMONSTRATION
    // ========================================================================
    println!("\n┌───────────────────────────────────────────────────────────────┐");
    println!("│ 2. REGRESSION DIAGNOSTICS                                     │");
    println!("└───────────────────────────────────────────────────────────────┘");

    // Simulated regression: house prices
    let y_true_reg = from_vec(
        vec![250.0, 300.0, 280.0, 350.0, 320.0, 290.0, 310.0, 340.0],
        &[8],
        DeviceType::Cpu,
    )?;
    let y_pred_reg = from_vec(
        vec![245.0, 305.0, 275.0, 355.0, 315.0, 295.0, 308.0, 345.0],
        &[8],
        DeviceType::Cpu,
    )?;

    // Features for diagnostic tests (size, location_score, age, bedrooms)
    let features = from_vec(
        vec![
            1500.0, 0.8, 5.0, 3.0, // House 1
            2000.0, 0.9, 3.0, 4.0, // House 2
            1800.0, 0.7, 8.0, 3.0, // House 3
            2500.0, 0.95, 2.0, 5.0, // House 4
            2200.0, 0.85, 4.0, 4.0, // House 5
            1700.0, 0.75, 6.0, 3.0, // House 6
            1900.0, 0.8, 5.0, 3.0, // House 7
            2400.0, 0.92, 3.0, 5.0, // House 8
        ],
        &[8, 4],
        DeviceType::Cpu,
    )?;

    // Residual Diagnostics
    let residual_diag = ResidualDiagnostics::new(&y_true_reg, &y_pred_reg)?;
    println!("  Residual Diagnostics:");
    println!("    Mean:                {:.4}", residual_diag.mean);
    println!("    Std Dev:             {:.4}", residual_diag.std_dev);
    println!("    Skewness:            {:.4}", residual_diag.skewness);
    println!("    Kurtosis:            {:.4}", residual_diag.kurtosis);
    println!(
        "    Approx Normal:       {}",
        residual_diag.is_approximately_normal()
    );

    // Durbin-Watson Test (autocorrelation in residuals)
    let dw_stat = durbin_watson(&y_true_reg, &y_pred_reg)?;
    println!("  Durbin-Watson Statistic:                {:.4}", dw_stat);
    println!("    └─ ~2: No autocorrelation, 0-2: Positive, 2-4: Negative");

    // Cook's Distance (influential points)
    let cooks_d = cooks_distance(&y_true_reg, &y_pred_reg, &features)?;
    let max_cooks = cooks_d
        .iter()
        .fold(0.0f64, |max, &d| if d > max { d } else { max });
    println!("  Max Cook's Distance:                    {:.4}", max_cooks);
    println!("    └─ Values > 1 indicate influential points");

    // VIF (Variance Inflation Factor) for first feature
    let vif_feature_0 = variance_inflation_factor(&features, 0)?;
    println!(
        "  VIF (Feature 0 - house size):           {:.4}",
        vif_feature_0
    );
    println!("    └─ VIF > 10: High multicollinearity");

    // Condition Number
    let cond_num = condition_number(&features)?;
    println!("  Condition Number:                       {:.4}", cond_num);
    println!("    └─ > 30: Potential multicollinearity");

    // Comprehensive Report
    let diag_report = RegressionDiagnosticReport::generate(&y_true_reg, &y_pred_reg, &features)?;
    println!("\n{}", diag_report.format());

    // ========================================================================
    // 3. EXPLAINABILITY METRICS DEMONSTRATION
    // ========================================================================
    println!("┌───────────────────────────────────────────────────────────────┐");
    println!("│ 3. EXPLAINABILITY & INTERPRETABILITY METRICS                  │");
    println!("└───────────────────────────────────────────────────────────────┘");

    // Feature importances from 3 different runs (simulated)
    let importances = vec![
        vec![0.45, 0.30, 0.15, 0.10], // Run 1
        vec![0.47, 0.28, 0.16, 0.09], // Run 2
        vec![0.44, 0.31, 0.14, 0.11], // Run 3
    ];

    let stability = feature_importance_stability(&importances)?;
    println!("  Feature Importance Stability:           {:.4}", stability);
    println!("    └─ Higher = more consistent rankings");

    // Attributions from different explanation methods (SHAP, LIME, etc.)
    let attributions = vec![
        vec![0.8, 0.5, 0.3, 0.2],     // SHAP values
        vec![0.75, 0.52, 0.28, 0.22], // LIME values
    ];

    let agreement = attribution_agreement(&attributions)?;
    println!("  Attribution Agreement:                  {:.4}", agreement);
    println!("    └─ 1.0 = Perfect agreement between methods");

    // Faithfulness - correlation with actual model behavior
    let attribution = vec![0.8, 0.5, 0.3, 0.2];
    let perturbation_effects = vec![0.85, 0.48, 0.32, 0.18];
    let faithfulness = explanation_faithfulness(&attribution, &perturbation_effects)?;
    println!(
        "  Explanation Faithfulness:               {:.4}",
        faithfulness
    );
    println!("    └─ How well explanation matches model");

    // Completeness - coverage of important features
    let true_importances = vec![0.9, 0.6, 0.3, 0.1];
    let completeness = explanation_completeness(&attribution, &true_importances, 3)?;
    println!(
        "  Explanation Completeness (top-3):       {:.4}",
        completeness
    );
    println!("    └─ Fraction of important features identified");

    // Feature monotonicity
    let feature_vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let predictions = vec![10.0, 15.0, 18.0, 22.0, 25.0];
    let monotonicity = feature_monotonicity(&feature_vals, &predictions)?;
    println!(
        "  Feature Monotonicity:                   {:.4}",
        monotonicity
    );
    println!("    └─ 1.0 = Perfectly monotonic relationship");

    // Interaction strength
    let individual_effects = 5.0;
    let joint_effect = 7.0;
    let interaction = interaction_strength(individual_effects, joint_effect);
    println!(
        "  Interaction Strength:                   {:.4}",
        interaction
    );
    println!("    └─ Non-additive effects magnitude");

    // ========================================================================
    // 4. ROBUSTNESS METRICS DEMONSTRATION
    // ========================================================================
    println!("\n┌───────────────────────────────────────────────────────────────┐");
    println!("│ 4. ROBUSTNESS & RELIABILITY METRICS                           │");
    println!("└───────────────────────────────────────────────────────────────┘");

    // Simulated clean vs adversarial predictions
    let clean_pred = from_vec(
        vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        &[8],
        DeviceType::Cpu,
    )?;
    let adv_pred = from_vec(
        vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0], // 2 flipped
        &[8],
        DeviceType::Cpu,
    )?;
    let targets = from_vec(
        vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        &[8],
        DeviceType::Cpu,
    )?;

    let adv_acc = adversarial_accuracy(&clean_pred, &adv_pred, &targets)?;
    println!("  Adversarial Accuracy:                   {:.4}", adv_acc);
    println!("    └─ Accuracy maintained under attack");

    let attack_sr = attack_success_rate(&clean_pred, &adv_pred, &targets)?;
    println!("  Attack Success Rate:                    {:.4}", attack_sr);
    println!("    └─ Fraction of clean correct flipped");

    // Noise sensitivity
    let noisy_pred = from_vec(
        vec![0.05, 0.98, 0.03, 1.02, 0.02, 0.97, 0.01, 1.03],
        &[8],
        DeviceType::Cpu,
    )?;
    let noise_sens = noise_sensitivity(&clean_pred, &noisy_pred)?;
    println!(
        "  Noise Sensitivity:                      {:.4}",
        noise_sens
    );
    println!("    └─ Lower = more robust to noise");

    // Confidence stability
    let clean_conf = from_vec(
        vec![0.95, 0.92, 0.88, 0.90, 0.94, 0.91, 0.89, 0.93],
        &[8],
        DeviceType::Cpu,
    )?;
    let pert_conf = from_vec(
        vec![0.92, 0.90, 0.85, 0.88, 0.91, 0.89, 0.87, 0.91],
        &[8],
        DeviceType::Cpu,
    )?;
    let conf_stab = confidence_stability(&clean_conf, &pert_conf)?;
    println!("  Confidence Stability:                   {:.4}", conf_stab);
    println!("    └─ Higher = more stable confidences");

    // OOD Detection
    let in_dist_conf = vec![0.95, 0.92, 0.90, 0.93];
    let out_dist_conf = vec![0.55, 0.60, 0.58, 0.62];
    let ood_score = ood_detection_score(&in_dist_conf, &out_dist_conf)?;
    println!("  OOD Detection Score:                    {:.4}", ood_score);
    println!("    └─ Ability to detect out-of-distribution inputs");

    // Corruption robustness
    let clean_acc_val = 0.95;
    let corrupted_accs = vec![0.88, 0.90, 0.86, 0.89];
    let corr_robust = corruption_robustness(clean_acc_val, &corrupted_accs)?;
    println!(
        "  Corruption Robustness:                  {:.4}",
        corr_robust
    );
    println!("    └─ Accuracy retention under corruptions");

    // Certified robustness
    let margins = vec![0.5, 0.3, 0.8, 0.4];
    let gradients = vec![0.1, 0.15, 0.2, 0.1];
    let cert_radius = certified_robustness_radius(&margins, &gradients)?;
    println!(
        "  Certified Robustness Radius:            {:.4}",
        cert_radius
    );
    println!("    └─ Minimum perturbation to change prediction");

    // Comprehensive Robustness Report
    let corrupted = vec![0.88, 0.90, 0.86];
    let robust_report = RobustnessReport::new(
        &clean_pred,
        &adv_pred,
        &targets,
        &clean_conf,
        &pert_conf,
        &corrupted,
    )?;
    println!("\n{}", robust_report.format());

    // ========================================================================
    // SUMMARY
    // ========================================================================
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║  SUMMARY - Session 7 Enhancements                             ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║  ✅ Time Series Metrics:      9 metrics implemented           ║");
    println!("║  ✅ Regression Diagnostics:   8 diagnostic tools              ║");
    println!("║  ✅ Explainability Metrics:   7 interpretability measures     ║");
    println!("║  ✅ Robustness Metrics:       8 reliability assessments       ║");
    println!("║                                                               ║");
    println!("║  Total: 32 new metrics + 4 comprehensive reports              ║");
    println!("║  All metrics follow SciRS2 POLICY for consistency             ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    Ok(())
}
