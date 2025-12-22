//! Regression diagnostic metrics and tools
//!
//! This module provides comprehensive diagnostics for regression models,
//! including residual analysis, influence measures, and model validation.
//! All implementations follow SciRS2 POLICY.

use torsh_core::error::TorshError;
use torsh_tensor::Tensor;

/// Residual diagnostics for regression models
#[derive(Debug, Clone)]
pub struct ResidualDiagnostics {
    /// Raw residuals (y - ŷ)
    pub residuals: Vec<f64>,
    /// Standardized residuals
    pub standardized_residuals: Vec<f64>,
    /// Mean of residuals (should be close to 0)
    pub mean: f64,
    /// Standard deviation of residuals
    pub std_dev: f64,
    /// Skewness of residuals
    pub skewness: f64,
    /// Kurtosis of residuals
    pub kurtosis: f64,
}

impl ResidualDiagnostics {
    /// Compute residual diagnostics
    pub fn new(y_true: &Tensor, y_pred: &Tensor) -> Result<Self, TorshError> {
        let true_vec = y_true.to_vec()?;
        let pred_vec = y_pred.to_vec()?;

        if true_vec.len() != pred_vec.len() {
            return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
        }

        // Calculate residuals
        let residuals: Vec<f64> = true_vec
            .iter()
            .zip(pred_vec.iter())
            .map(|(&t, &p)| t as f64 - p as f64)
            .collect();

        // Mean
        let mean = residuals.iter().sum::<f64>() / residuals.len() as f64;

        // Standard deviation
        let variance =
            residuals.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / residuals.len() as f64;
        let std_dev = variance.sqrt();

        // Standardized residuals
        let standardized_residuals: Vec<f64> = if std_dev > 1e-10 {
            residuals.iter().map(|&r| (r - mean) / std_dev).collect()
        } else {
            vec![0.0; residuals.len()]
        };

        // Skewness
        let skewness = if std_dev > 1e-10 {
            let third_moment = residuals
                .iter()
                .map(|&r| ((r - mean) / std_dev).powi(3))
                .sum::<f64>()
                / residuals.len() as f64;
            third_moment
        } else {
            0.0
        };

        // Kurtosis (excess kurtosis)
        let kurtosis = if std_dev > 1e-10 {
            let fourth_moment = residuals
                .iter()
                .map(|&r| ((r - mean) / std_dev).powi(4))
                .sum::<f64>()
                / residuals.len() as f64;
            fourth_moment - 3.0 // Excess kurtosis
        } else {
            0.0
        };

        Ok(Self {
            residuals,
            standardized_residuals,
            mean,
            std_dev,
            skewness,
            kurtosis,
        })
    }

    /// Check if residuals appear normally distributed
    pub fn is_approximately_normal(&self) -> bool {
        // Rough heuristic: skewness near 0 and excess kurtosis near 0
        self.skewness.abs() < 0.5 && self.kurtosis.abs() < 1.0
    }
}

/// Durbin-Watson statistic for testing autocorrelation in residuals
///
/// Values range from 0 to 4:
/// - 2: no autocorrelation
/// - 0-2: positive autocorrelation
/// - 2-4: negative autocorrelation
pub fn durbin_watson(y_true: &Tensor, y_pred: &Tensor) -> Result<f64, TorshError> {
    let true_vec = y_true.to_vec()?;
    let pred_vec = y_pred.to_vec()?;

    if true_vec.len() != pred_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    if true_vec.len() < 2 {
        return Err(TorshError::InvalidArgument(
            "Need at least 2 observations".to_string(),
        ));
    }

    let residuals: Vec<f64> = true_vec
        .iter()
        .zip(pred_vec.iter())
        .map(|(&t, &p)| t as f64 - p as f64)
        .collect();

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 1..residuals.len() {
        let diff = residuals[i] - residuals[i - 1];
        numerator += diff * diff;
    }

    for &r in &residuals {
        denominator += r * r;
    }

    if denominator < 1e-10 {
        return Err(TorshError::InvalidArgument(
            "Residual variance too small".to_string(),
        ));
    }

    Ok(numerator / denominator)
}

/// Leverage (hat values) for each observation
///
/// High leverage points have greater potential to influence the regression line.
/// Note: This is a simplified calculation assuming features are available.
pub fn calculate_leverage(features: &Tensor) -> Result<Vec<f64>, TorshError> {
    let feature_vec = features.to_vec()?;
    let shape = features.shape();
    let dims = shape.dims();

    if dims.len() != 2 {
        return Err(TorshError::InvalidArgument(
            "Features must be 2D (n_samples, n_features)".to_string(),
        ));
    }

    let n_samples = dims[0];
    let n_features = dims[1];

    // For simplicity, we calculate leverage based on distance from feature means
    // True leverage requires (X'X)^-1 which needs matrix operations
    let mut feature_means = vec![0.0; n_features];
    for i in 0..n_samples {
        for j in 0..n_features {
            feature_means[j] += feature_vec[i * n_features + j] as f64;
        }
    }
    for mean in &mut feature_means {
        *mean /= n_samples as f64;
    }

    // Calculate Mahalanobis-like distance (simplified)
    let mut leverages = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut distance = 0.0;
        for j in 0..n_features {
            let val = feature_vec[i * n_features + j] as f64;
            distance += (val - feature_means[j]).powi(2);
        }
        // Normalize leverage to be between 0 and 1
        leverages.push(distance.sqrt() / n_features as f64);
    }

    Ok(leverages)
}

/// Cook's distance - measure of influence of each observation
///
/// Points with Cook's distance > 1 are often considered influential.
/// Points with Cook's distance > 4/n warrant investigation.
pub fn cooks_distance(
    y_true: &Tensor,
    y_pred: &Tensor,
    features: &Tensor,
) -> Result<Vec<f64>, TorshError> {
    let true_vec = y_true.to_vec()?;
    let pred_vec = y_pred.to_vec()?;

    if true_vec.len() != pred_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    let residuals: Vec<f64> = true_vec
        .iter()
        .zip(pred_vec.iter())
        .map(|(&t, &p)| t as f64 - p as f64)
        .collect();

    // Calculate MSE
    let mse = residuals.iter().map(|&r| r * r).sum::<f64>() / residuals.len() as f64;

    if mse < 1e-10 {
        return Err(TorshError::InvalidArgument("MSE too small".to_string()));
    }

    // Get leverage
    let leverages = calculate_leverage(features)?;

    // Calculate Cook's distance
    let n_features = features.shape().dims()[1];
    let mut cooks_d = Vec::with_capacity(residuals.len());

    for i in 0..residuals.len() {
        let standardized_res = residuals[i] / mse.sqrt();
        let h = leverages[i];

        // Simplified Cook's D formula
        let d = (standardized_res.powi(2) / n_features as f64) * (h / (1.0 - h).max(1e-10));
        cooks_d.push(d);
    }

    Ok(cooks_d)
}

/// DFFITS - measure of how much a prediction changes when that observation is deleted
pub fn dffits(y_true: &Tensor, y_pred: &Tensor, features: &Tensor) -> Result<Vec<f64>, TorshError> {
    let true_vec = y_true.to_vec()?;
    let pred_vec = y_pred.to_vec()?;

    if true_vec.len() != pred_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    let residuals: Vec<f64> = true_vec
        .iter()
        .zip(pred_vec.iter())
        .map(|(&t, &p)| t as f64 - p as f64)
        .collect();

    let mse = residuals.iter().map(|&r| r * r).sum::<f64>() / residuals.len() as f64;

    if mse < 1e-10 {
        return Err(TorshError::InvalidArgument("MSE too small".to_string()));
    }

    let leverages = calculate_leverage(features)?;

    let mut dffits_values = Vec::with_capacity(residuals.len());

    for i in 0..residuals.len() {
        let standardized_res = residuals[i] / mse.sqrt();
        let h = leverages[i];

        // DFFITS = standardized_residual * sqrt(h / (1 - h))
        let dffits_val = standardized_res * (h / (1.0 - h).max(1e-10)).sqrt();
        dffits_values.push(dffits_val);
    }

    Ok(dffits_values)
}

/// Variance Inflation Factor (VIF) for detecting multicollinearity
///
/// VIF > 10 indicates high multicollinearity
/// VIF > 5 suggests moderate multicollinearity
pub fn variance_inflation_factor(features: &Tensor, feature_idx: usize) -> Result<f64, TorshError> {
    let feature_vec = features.to_vec()?;
    let shape = features.shape();
    let dims = shape.dims();

    if dims.len() != 2 {
        return Err(TorshError::InvalidArgument(
            "Features must be 2D".to_string(),
        ));
    }

    let n_samples = dims[0];
    let n_features = dims[1];

    if feature_idx >= n_features {
        return Err(TorshError::InvalidArgument(
            "Feature index out of bounds".to_string(),
        ));
    }

    // Extract target feature
    let mut target_feature = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        target_feature.push(feature_vec[i * n_features + feature_idx] as f64);
    }

    // Calculate mean
    let mean = target_feature.iter().sum::<f64>() / n_samples as f64;

    // Calculate total sum of squares
    let tss = target_feature
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>();

    if tss < 1e-10 {
        return Err(TorshError::InvalidArgument(
            "Feature variance too small".to_string(),
        ));
    }

    // For a proper VIF, we would regress this feature on all other features
    // This is a simplified version using correlation-based approximation
    let mut total_correlation = 0.0;
    let mut count = 0;

    for j in 0..n_features {
        if j != feature_idx {
            let mut other_feature = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                other_feature.push(feature_vec[i * n_features + j] as f64);
            }

            let other_mean = other_feature.iter().sum::<f64>() / n_samples as f64;

            // Calculate correlation
            let mut numerator = 0.0;
            let mut denom1 = 0.0;
            let mut denom2 = 0.0;

            for i in 0..n_samples {
                let x_dev = target_feature[i] - mean;
                let y_dev = other_feature[i] - other_mean;
                numerator += x_dev * y_dev;
                denom1 += x_dev * x_dev;
                denom2 += y_dev * y_dev;
            }

            if denom1 > 1e-10 && denom2 > 1e-10 {
                let corr = numerator / (denom1 * denom2).sqrt();
                total_correlation += corr.abs();
                count += 1;
            }
        }
    }

    let avg_correlation = if count > 0 {
        total_correlation / count as f64
    } else {
        0.0
    };

    // Approximate VIF from average correlation
    let r_squared = avg_correlation.powi(2);
    if r_squared >= 1.0 - 1e-10 {
        Ok(1000.0) // Cap at large value to avoid infinity
    } else {
        Ok(1.0 / (1.0 - r_squared))
    }
}

/// Condition number of feature matrix - measure of multicollinearity
///
/// Large condition numbers (> 30) indicate multicollinearity
pub fn condition_number(features: &Tensor) -> Result<f64, TorshError> {
    let feature_vec = features.to_vec()?;
    let shape = features.shape();
    let dims = shape.dims();

    if dims.len() != 2 {
        return Err(TorshError::InvalidArgument(
            "Features must be 2D".to_string(),
        ));
    }

    let n_features = dims[1];
    let n_samples = dims[0];

    // Calculate feature variances as approximation
    let mut max_variance: f64 = 0.0;
    let mut min_variance: f64 = f64::INFINITY;

    for j in 0..n_features {
        let mut feature = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            feature.push(feature_vec[i * n_features + j] as f64);
        }

        let mean = feature.iter().sum::<f64>() / n_samples as f64;
        let variance = feature.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;

        max_variance = max_variance.max(variance);
        if variance > 1e-10 {
            min_variance = min_variance.min(variance);
        }
    }

    if min_variance < 1e-10 || !min_variance.is_finite() {
        return Ok(1000.0); // High condition number for ill-conditioned matrix
    }

    Ok((max_variance / min_variance).sqrt())
}

/// Heteroscedasticity test (Breusch-Pagan test statistic)
///
/// Tests whether residual variance depends on the values of predictors.
/// Large values indicate heteroscedasticity.
pub fn breusch_pagan_test(
    y_true: &Tensor,
    y_pred: &Tensor,
    features: &Tensor,
) -> Result<f64, TorshError> {
    let true_vec = y_true.to_vec()?;
    let pred_vec = y_pred.to_vec()?;

    if true_vec.len() != pred_vec.len() {
        return Err(TorshError::InvalidArgument("Size mismatch".to_string()));
    }

    let residuals: Vec<f64> = true_vec
        .iter()
        .zip(pred_vec.iter())
        .map(|(&t, &p)| t as f64 - p as f64)
        .collect();

    // Squared residuals
    let squared_residuals: Vec<f64> = residuals.iter().map(|&r| r * r).collect();

    // Mean of squared residuals
    let mean_sq_res = squared_residuals.iter().sum::<f64>() / squared_residuals.len() as f64;

    // Standardized squared residuals
    let standardized: Vec<f64> = squared_residuals
        .iter()
        .map(|&sr| sr / mean_sq_res)
        .collect();

    // Calculate correlation with features (simplified test statistic)
    let shape = features.shape();
    let dims = shape.dims();
    let n_features = dims[1];
    let n_samples = dims[0];
    let feature_vec = features.to_vec()?;

    let mut test_statistic = 0.0;

    for j in 0..n_features {
        let mut feature = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            feature.push(feature_vec[i * n_features + j] as f64);
        }

        let feature_mean = feature.iter().sum::<f64>() / n_samples as f64;
        let std_mean = standardized.iter().sum::<f64>() / n_samples as f64;

        // Calculate correlation
        let mut numerator = 0.0;
        let mut denom1 = 0.0;
        let mut denom2 = 0.0;

        for i in 0..n_samples {
            let f_dev = feature[i] - feature_mean;
            let s_dev = standardized[i] - std_mean;
            numerator += f_dev * s_dev;
            denom1 += f_dev * f_dev;
            denom2 += s_dev * s_dev;
        }

        if denom1 > 1e-10 && denom2 > 1e-10 {
            let corr = numerator / (denom1 * denom2).sqrt();
            test_statistic += corr.abs();
        }
    }

    Ok(test_statistic)
}

/// Comprehensive diagnostic report for regression
#[derive(Debug, Clone)]
pub struct RegressionDiagnosticReport {
    pub residual_diagnostics: ResidualDiagnostics,
    pub durbin_watson: f64,
    pub cooks_distance_max: f64,
    pub high_leverage_points: usize,
    pub influential_points: usize,
}

impl RegressionDiagnosticReport {
    /// Generate comprehensive diagnostic report
    pub fn generate(
        y_true: &Tensor,
        y_pred: &Tensor,
        features: &Tensor,
    ) -> Result<Self, TorshError> {
        let residual_diagnostics = ResidualDiagnostics::new(y_true, y_pred)?;
        let durbin_watson_stat = durbin_watson(y_true, y_pred)?;
        let cooks_d = cooks_distance(y_true, y_pred, features)?;
        let leverages = calculate_leverage(features)?;

        let n = y_true.to_vec()?.len();
        let leverage_threshold = 2.0 * features.shape().dims()[1] as f64 / n as f64;
        let cooks_threshold = 4.0 / n as f64;

        let high_leverage_points = leverages
            .iter()
            .filter(|&&h| h > leverage_threshold)
            .count();
        let influential_points = cooks_d.iter().filter(|&&d| d > cooks_threshold).count();
        let cooks_distance_max = cooks_d
            .iter()
            .fold(0.0f64, |max, &d| if d > max { d } else { max });

        Ok(Self {
            residual_diagnostics,
            durbin_watson: durbin_watson_stat,
            cooks_distance_max,
            high_leverage_points,
            influential_points,
        })
    }

    /// Format report as string
    pub fn format(&self) -> String {
        let mut report = String::new();
        report.push_str("╔═══════════════════════════════════════════════╗\n");
        report.push_str("║      Regression Diagnostics Report           ║\n");
        report.push_str("╠═══════════════════════════════════════════════╣\n");
        report.push_str(&format!(
            "║ Residual Mean:     {:.6}                 ║\n",
            self.residual_diagnostics.mean
        ));
        report.push_str(&format!(
            "║ Residual Std Dev:  {:.6}                 ║\n",
            self.residual_diagnostics.std_dev
        ));
        report.push_str(&format!(
            "║ Skewness:          {:.6}                 ║\n",
            self.residual_diagnostics.skewness
        ));
        report.push_str(&format!(
            "║ Kurtosis:          {:.6}                 ║\n",
            self.residual_diagnostics.kurtosis
        ));
        report.push_str(&format!(
            "║ Durbin-Watson:     {:.6}                 ║\n",
            self.durbin_watson
        ));
        report.push_str(&format!(
            "║ Max Cook's D:      {:.6}                 ║\n",
            self.cooks_distance_max
        ));
        report.push_str(&format!(
            "║ High Leverage Pts: {}                        ║\n",
            self.high_leverage_points
        ));
        report.push_str(&format!(
            "║ Influential Pts:   {}                        ║\n",
            self.influential_points
        ));

        let normality = if self.residual_diagnostics.is_approximately_normal() {
            "Yes"
        } else {
            "No"
        };
        report.push_str(&format!(
            "║ Approx Normal:     {:<4}                     ║\n",
            normality
        ));

        report.push_str("╚═══════════════════════════════════════════════╝\n");

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::from_vec;

    #[test]
    fn test_residual_diagnostics() {
        let y_true = from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            &[5],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();
        let y_pred = from_vec(
            vec![1.1, 1.9, 3.1, 3.9, 5.1],
            &[5],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();

        let diag = ResidualDiagnostics::new(&y_true, &y_pred).unwrap();
        assert!(diag.mean.abs() < 0.2);
        assert!(diag.std_dev > 0.0);
    }

    #[test]
    fn test_durbin_watson_no_autocorrelation() {
        let y_true = from_vec(
            vec![1.0, 2.0, 1.5, 2.5, 2.0],
            &[5],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();
        let y_pred = from_vec(
            vec![1.1, 2.1, 1.6, 2.6, 2.1],
            &[5],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();

        let dw = durbin_watson(&y_true, &y_pred).unwrap();
        assert!(dw >= 0.0 && dw <= 4.0);
    }

    #[test]
    fn test_leverage_calculation() {
        let features = from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();

        let leverages = calculate_leverage(&features).unwrap();
        assert_eq!(leverages.len(), 3);
        assert!(leverages.iter().all(|&h| h >= 0.0));
    }

    #[test]
    fn test_cooks_distance() {
        let y_true = from_vec(vec![1.0, 2.0, 3.0], &[3], torsh_core::DeviceType::Cpu).unwrap();
        let y_pred = from_vec(vec![1.1, 2.0, 2.9], &[3], torsh_core::DeviceType::Cpu).unwrap();
        let features = from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();

        let cooks_d = cooks_distance(&y_true, &y_pred, &features).unwrap();
        assert_eq!(cooks_d.len(), 3);
        assert!(cooks_d.iter().all(|&d| d >= 0.0));
    }

    #[test]
    fn test_dffits() {
        let y_true = from_vec(vec![1.0, 2.0, 3.0], &[3], torsh_core::DeviceType::Cpu).unwrap();
        let y_pred = from_vec(vec![1.1, 2.0, 2.9], &[3], torsh_core::DeviceType::Cpu).unwrap();
        let features = from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();

        let dffits_vals = dffits(&y_true, &y_pred, &features).unwrap();
        assert_eq!(dffits_vals.len(), 3);
    }

    #[test]
    fn test_vif_calculation() {
        let features = from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            &[3, 3],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();

        let vif = variance_inflation_factor(&features, 0).unwrap();
        assert!(vif >= 1.0);
    }

    #[test]
    fn test_condition_number() {
        let features = from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();

        let cn = condition_number(&features).unwrap();
        assert!(cn >= 1.0);
    }

    #[test]
    fn test_breusch_pagan() {
        let y_true = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], torsh_core::DeviceType::Cpu).unwrap();
        let y_pred = from_vec(vec![1.1, 2.1, 2.9, 4.1], &[4], torsh_core::DeviceType::Cpu).unwrap();
        let features = from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[4, 2],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();

        let bp = breusch_pagan_test(&y_true, &y_pred, &features).unwrap();
        assert!(bp >= 0.0);
    }

    #[test]
    fn test_diagnostic_report() {
        let y_true = from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4], torsh_core::DeviceType::Cpu).unwrap();
        let y_pred = from_vec(vec![1.1, 2.0, 3.1, 3.9], &[4], torsh_core::DeviceType::Cpu).unwrap();
        let features = from_vec(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[4, 2],
            torsh_core::DeviceType::Cpu,
        )
        .unwrap();

        let report = RegressionDiagnosticReport::generate(&y_true, &y_pred, &features).unwrap();
        let formatted = report.format();
        assert!(formatted.contains("Regression Diagnostics Report"));
    }
}
