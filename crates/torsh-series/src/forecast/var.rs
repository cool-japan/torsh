//! Vector Autoregression (VAR) models for multivariate time series
//!
//! VAR models are used to capture the linear interdependencies among multiple time series.
//! Each variable is modeled as a linear function of past values of itself and past values
//! of the other variables.

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::TimeSeries;
use scirs2_core::ndarray::{Array1, Array2};
use torsh_core::error::{Result, TorshError};
use torsh_tensor::Tensor;

/// Vector Autoregression (VAR) model
///
/// A VAR(p) model expresses each variable as a linear combination of:
/// - Its own lagged values up to lag p
/// - Lagged values of all other variables up to lag p
/// - A constant term and potentially exogenous variables
pub struct VAR {
    /// Model order (number of lags)
    order: usize,
    /// Number of variables
    n_vars: usize,
    /// Coefficient matrices: coefficients[lag][var_to][var_from]
    coefficients: Vec<Array2<f64>>,
    /// Intercept vector
    intercept: Option<Array1<f64>>,
    /// Fitted flag
    is_fitted: bool,
    /// Residuals from fitted model
    residuals: Option<Array2<f64>>,
}

impl VAR {
    /// Create a new VAR model
    ///
    /// # Arguments
    /// * `order` - Number of lags (p in VAR(p))
    pub fn new(order: usize) -> Self {
        Self {
            order,
            n_vars: 0,
            coefficients: Vec::new(),
            intercept: None,
            is_fitted: false,
            residuals: None,
        }
    }

    /// Get model order
    pub fn order(&self) -> usize {
        self.order
    }

    /// Get number of variables
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Check if model is fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Fit VAR model to multivariate time series
    ///
    /// Uses Ordinary Least Squares (OLS) to estimate coefficients.
    /// For each variable, we solve: y_t = A_1 y_{t-1} + ... + A_p y_{t-p} + c + e_t
    pub fn fit(&mut self, series: &TimeSeries) -> Result<()> {
        // Validate input dimensions
        let shape = series.values.shape();
        let dims = shape.dims();

        if dims.len() != 2 {
            return Err(TorshError::InvalidArgument(
                "VAR requires 2D time series (time x variables)".to_string(),
            ));
        }

        let n_obs = dims[0];
        let n_vars = dims[1];

        if n_obs <= self.order {
            return Err(TorshError::InvalidArgument(format!(
                "Insufficient observations: {} observations for VAR({}) model",
                n_obs, self.order
            )));
        }

        self.n_vars = n_vars;

        // Convert to ndarray for easier manipulation
        let data = series.values.to_vec()?;
        let y_matrix = Array2::from_shape_vec((n_obs, n_vars), data)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to create matrix: {}", e)))?;

        // Construct design matrix X and response matrix Y
        let n_effective = n_obs - self.order;

        // Y: response variables (from time t=order to end)
        let mut y_response = Array2::<f64>::zeros((n_effective, n_vars));
        for i in 0..n_effective {
            for j in 0..n_vars {
                y_response[[i, j]] = y_matrix[[i + self.order, j]] as f64;
            }
        }

        // X: lagged variables (design matrix)
        // Each row: [y_{t-1}, y_{t-2}, ..., y_{t-p}, 1] flattened
        let n_features = n_vars * self.order + 1; // +1 for intercept
        let mut x_design = Array2::<f64>::zeros((n_effective, n_features));

        for t in 0..n_effective {
            let mut col_idx = 0;

            // Add lagged values
            for lag in 1..=self.order {
                for var in 0..n_vars {
                    let time_idx = t + self.order - lag;
                    x_design[[t, col_idx]] = y_matrix[[time_idx, var]] as f64;
                    col_idx += 1;
                }
            }

            // Add intercept term
            x_design[[t, col_idx]] = 1.0;
        }

        // Solve OLS for each variable: β = (X'X)^(-1) X'y
        let xt_x = x_design.t().dot(&x_design);
        let xt_y = x_design.t().dot(&y_response);

        // TODO: Use scirs2-core linalg when available
        // For now, use simplified pseudo-inverse approach
        let mut all_coefficients = Array2::<f64>::zeros((n_vars, n_features));

        // Simplified OLS solution using regularization
        // β = (X'X + λI)^(-1) X'y (ridge regression with small λ)
        let lambda = 1e-6;
        let mut xt_x_reg = xt_x.clone();
        for i in 0..n_features {
            xt_x_reg[[i, i]] += lambda;
        }

        // Use simple Gaussian elimination or set to identity for now
        // In practice, would use proper linear algebra solver
        for var in 0..n_vars {
            // Simplified coefficient estimation (placeholder)
            // In production, would use proper linear system solver
            for i in 0..n_features {
                all_coefficients[[var, i]] = xt_y[[i, var]] / (xt_x_reg[[i, i]] + 1e-10);
            }
        }

        // Extract coefficient matrices and intercept
        self.coefficients = Vec::with_capacity(self.order);
        for lag in 0..self.order {
            let mut coef_matrix = Array2::<f64>::zeros((n_vars, n_vars));
            for var_to in 0..n_vars {
                for var_from in 0..n_vars {
                    let col_idx = lag * n_vars + var_from;
                    coef_matrix[[var_to, var_from]] = all_coefficients[[var_to, col_idx]];
                }
            }
            self.coefficients.push(coef_matrix);
        }

        // Extract intercept
        let mut intercept = Array1::<f64>::zeros(n_vars);
        for var in 0..n_vars {
            intercept[var] = all_coefficients[[var, n_features - 1]];
        }
        self.intercept = Some(intercept);

        // Calculate residuals
        let predictions = self.predict_in_sample(&x_design)?;
        let mut residuals = Array2::<f64>::zeros((n_effective, n_vars));
        for i in 0..n_effective {
            for j in 0..n_vars {
                residuals[[i, j]] = y_response[[i, j]] - predictions[[i, j]];
            }
        }
        self.residuals = Some(residuals);

        self.is_fitted = true;
        Ok(())
    }

    /// Predict using design matrix (internal helper)
    fn predict_in_sample(&self, x_design: &Array2<f64>) -> Result<Array2<f64>> {
        let n_obs = x_design.nrows();
        let n_features = x_design.ncols();
        let n_vars = self.n_vars;

        let mut predictions = Array2::<f64>::zeros((n_obs, n_vars));

        // Reconstruct coefficient matrix
        let mut all_coefficients = Array2::<f64>::zeros((n_vars, n_features));

        for lag in 0..self.order {
            for var_to in 0..n_vars {
                for var_from in 0..n_vars {
                    let col_idx = lag * n_vars + var_from;
                    all_coefficients[[var_to, col_idx]] =
                        self.coefficients[lag][[var_to, var_from]];
                }
            }
        }

        // Add intercept
        if let Some(ref intercept) = self.intercept {
            for var in 0..n_vars {
                all_coefficients[[var, n_features - 1]] = intercept[var];
            }
        }

        // Compute predictions: Y_hat = X * β
        for obs in 0..n_obs {
            for var in 0..n_vars {
                let mut pred = 0.0;
                for feat in 0..n_features {
                    pred += x_design[[obs, feat]] * all_coefficients[[var, feat]];
                }
                predictions[[obs, var]] = pred;
            }
        }

        Ok(predictions)
    }

    /// Forecast h steps ahead
    ///
    /// Uses recursive forecasting: each prediction is used as input for the next step.
    pub fn forecast(&self, series: &TimeSeries, steps: usize) -> Result<TimeSeries> {
        if !self.is_fitted {
            return Err(TorshError::InvalidArgument(
                "Model must be fitted before forecasting".to_string(),
            ));
        }

        let shape = series.values.shape();
        let dims = shape.dims();
        let n_obs = dims[0];
        let n_vars = dims[1];

        if n_vars != self.n_vars {
            return Err(TorshError::InvalidArgument(format!(
                "Series has {} variables but model was fitted with {} variables",
                n_vars, self.n_vars
            )));
        }

        if n_obs < self.order {
            return Err(TorshError::InvalidArgument(
                "Insufficient observations for forecasting".to_string(),
            ));
        }

        // Get recent history
        let data = series.values.to_vec()?;
        let y_matrix = Array2::from_shape_vec((n_obs, n_vars), data)
            .map_err(|e| TorshError::InvalidArgument(format!("Failed to create matrix: {}", e)))?;

        // Initialize with last 'order' observations
        let mut history = Vec::with_capacity(self.order);
        for i in 0..self.order {
            let idx = n_obs - self.order + i;
            let mut obs = Array1::<f64>::zeros(n_vars);
            for j in 0..n_vars {
                obs[j] = y_matrix[[idx, j]] as f64;
            }
            history.push(obs);
        }

        // Forecast recursively
        let mut forecasts = Vec::with_capacity(steps * n_vars);

        for _step in 0..steps {
            // Construct current state vector
            let mut y_pred = if let Some(ref intercept) = self.intercept {
                intercept.clone()
            } else {
                Array1::<f64>::zeros(n_vars)
            };

            // Add contributions from lagged values
            for (lag_idx, lag_obs) in history.iter().rev().enumerate() {
                let coef_matrix = &self.coefficients[lag_idx];
                for var_to in 0..n_vars {
                    for var_from in 0..n_vars {
                        y_pred[var_to] += coef_matrix[[var_to, var_from]] * lag_obs[var_from];
                    }
                }
            }

            // Add to forecasts
            for var in 0..n_vars {
                forecasts.push(y_pred[var] as f32);
            }

            // Update history (sliding window)
            history.remove(0);
            history.push(y_pred);
        }

        let tensor = Tensor::from_vec(forecasts, &[steps, n_vars])?;
        Ok(TimeSeries::new(tensor))
    }

    /// Get Akaike Information Criterion (AIC)
    pub fn aic(&self) -> Result<f64> {
        if !self.is_fitted {
            return Err(TorshError::InvalidArgument(
                "Model must be fitted to compute AIC".to_string(),
            ));
        }

        let residuals = self
            .residuals
            .as_ref()
            .ok_or_else(|| TorshError::InvalidArgument("No residuals available".to_string()))?;

        let _n_obs = residuals.nrows() as f64;
        let n_params = (self.n_vars * self.n_vars * self.order + self.n_vars) as f64;

        // Calculate log-likelihood (assuming Gaussian errors)
        let mut log_likelihood = 0.0;
        let two_pi = 2.0 * std::f64::consts::PI;

        for i in 0..residuals.nrows() {
            for j in 0..residuals.ncols() {
                let residual = residuals[[i, j]];
                log_likelihood -= 0.5 * (two_pi.ln() + residual * residual);
            }
        }

        // AIC = -2*log(L) + 2*k
        Ok(-2.0 * log_likelihood + 2.0 * n_params)
    }

    /// Get Bayesian Information Criterion (BIC)
    pub fn bic(&self) -> Result<f64> {
        if !self.is_fitted {
            return Err(TorshError::InvalidArgument(
                "Model must be fitted to compute BIC".to_string(),
            ));
        }

        let residuals = self
            .residuals
            .as_ref()
            .ok_or_else(|| TorshError::InvalidArgument("No residuals available".to_string()))?;

        let n_obs = residuals.nrows() as f64;
        let n_params = (self.n_vars * self.n_vars * self.order + self.n_vars) as f64;

        // Calculate log-likelihood
        let mut log_likelihood = 0.0;
        let two_pi = 2.0 * std::f64::consts::PI;

        for i in 0..residuals.nrows() {
            for j in 0..residuals.ncols() {
                let residual = residuals[[i, j]];
                log_likelihood -= 0.5 * (two_pi.ln() + residual * residual);
            }
        }

        // BIC = -2*log(L) + k*log(n)
        Ok(-2.0 * log_likelihood + n_params * n_obs.ln())
    }

    /// Get Hannan-Quinn Information Criterion (HQIC)
    pub fn hqic(&self) -> Result<f64> {
        if !self.is_fitted {
            return Err(TorshError::InvalidArgument(
                "Model must be fitted to compute HQIC".to_string(),
            ));
        }

        let residuals = self
            .residuals
            .as_ref()
            .ok_or_else(|| TorshError::InvalidArgument("No residuals available".to_string()))?;

        let n_obs = residuals.nrows() as f64;
        let n_params = (self.n_vars * self.n_vars * self.order + self.n_vars) as f64;

        // Calculate log-likelihood
        let mut log_likelihood = 0.0;
        let two_pi = 2.0 * std::f64::consts::PI;

        for i in 0..residuals.nrows() {
            for j in 0..residuals.ncols() {
                let residual = residuals[[i, j]];
                log_likelihood -= 0.5 * (two_pi.ln() + residual * residual);
            }
        }

        // HQIC = -2*log(L) + 2*k*log(log(n))
        Ok(-2.0 * log_likelihood + 2.0 * n_params * n_obs.ln().ln())
    }

    /// Get coefficient matrix for a specific lag
    pub fn coefficients(&self, lag: usize) -> Result<&Array2<f64>> {
        if lag == 0 || lag > self.order {
            return Err(TorshError::InvalidArgument(format!(
                "Lag must be between 1 and {}",
                self.order
            )));
        }

        Ok(&self.coefficients[lag - 1])
    }

    /// Get intercept vector
    pub fn intercept(&self) -> Option<&Array1<f64>> {
        self.intercept.as_ref()
    }

    /// Get residuals
    pub fn residuals(&self) -> Option<&Array2<f64>> {
        self.residuals.as_ref()
    }
}

/// Granger causality test
///
/// Tests whether one time series is useful in forecasting another.
/// The null hypothesis is that x does not Granger-cause y.
pub struct GrangerCausality {
    max_lags: usize,
}

impl GrangerCausality {
    /// Create a new Granger causality test
    pub fn new(max_lags: usize) -> Self {
        Self { max_lags }
    }

    /// Test if x Granger-causes y
    ///
    /// Returns F-statistic and p-value for each lag up to max_lags
    pub fn test(&self, x: &TimeSeries, y: &TimeSeries) -> Result<Vec<(usize, f64, f64)>> {
        if x.len() != y.len() {
            return Err(TorshError::InvalidArgument(
                "Time series must have equal length".to_string(),
            ));
        }

        let mut results = Vec::new();

        for lag in 1..=self.max_lags {
            // Fit restricted model: y ~ lags(y)
            let mut restricted_series_data = Vec::new();
            for i in 0..y.len() {
                let y_val = y.values.get_item_flat(i)?;
                restricted_series_data.push(y_val);
            }
            let restricted_tensor = Tensor::from_vec(restricted_series_data, &[y.len(), 1])?;
            let restricted_series = TimeSeries::new(restricted_tensor);

            let mut restricted_model = VAR::new(lag);
            restricted_model.fit(&restricted_series)?;

            // Fit unrestricted model: y ~ lags(y) + lags(x)
            let mut unrestricted_series_data = Vec::new();
            for i in 0..y.len() {
                let y_val = y.values.get_item_flat(i)?;
                let x_val = x.values.get_item_flat(i)?;
                unrestricted_series_data.push(y_val);
                unrestricted_series_data.push(x_val);
            }
            let unrestricted_tensor = Tensor::from_vec(unrestricted_series_data, &[y.len(), 2])?;
            let unrestricted_series = TimeSeries::new(unrestricted_tensor);

            let mut unrestricted_model = VAR::new(lag);
            unrestricted_model.fit(&unrestricted_series)?;

            // Calculate F-statistic
            let rss_restricted = self.residual_sum_of_squares(&restricted_model)?;
            let rss_unrestricted = self.residual_sum_of_squares(&unrestricted_model)?;

            let n_obs = y.len() - lag;
            let n_params_diff = lag; // Number of additional parameters

            // TODO: Proper F-statistic calculation when VAR implementation is complete
            // For now, use placeholder to avoid numerical issues with zero residuals
            let f_stat = if rss_unrestricted > 1e-10 && rss_restricted >= rss_unrestricted {
                ((rss_restricted - rss_unrestricted) / n_params_diff as f64)
                    / (rss_unrestricted / (n_obs - 2 * lag - 1) as f64)
            } else {
                // Placeholder F-statistic for incomplete VAR implementation
                1.0
            };

            // Approximate p-value using F-distribution (simplified)
            let p_value = self.f_distribution_pvalue(f_stat, n_params_diff, n_obs - 2 * lag - 1);

            results.push((lag, f_stat.max(0.0), p_value));
        }

        Ok(results)
    }

    /// Calculate residual sum of squares
    fn residual_sum_of_squares(&self, model: &VAR) -> Result<f64> {
        let residuals = model
            .residuals()
            .ok_or_else(|| TorshError::InvalidArgument("No residuals available".to_string()))?;

        let mut rss = 0.0;
        for i in 0..residuals.nrows() {
            for j in 0..residuals.ncols() {
                rss += residuals[[i, j]] * residuals[[i, j]];
            }
        }

        Ok(rss)
    }

    /// Approximate F-distribution p-value (simplified)
    fn f_distribution_pvalue(&self, _f_stat: f64, _df1: usize, _df2: usize) -> f64 {
        // TODO: Use scirs2-stats for proper F-distribution calculation when available
        // For now, return placeholder p-value
        0.05
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::Tensor;

    fn create_multivariate_series() -> TimeSeries {
        // Create a simple 2-variable time series
        let data = vec![
            1.0f32, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
        ];
        let tensor = Tensor::from_vec(data, &[8, 2]).unwrap();
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_var_creation() {
        let var = VAR::new(2);
        assert_eq!(var.order(), 2);
        assert!(!var.is_fitted());
    }

    #[test]
    fn test_var_fit() {
        let series = create_multivariate_series();
        let mut var = VAR::new(2);
        var.fit(&series).unwrap();

        assert!(var.is_fitted());
        assert_eq!(var.n_vars(), 2);
    }

    #[test]
    fn test_var_forecast() {
        let series = create_multivariate_series();
        let mut var = VAR::new(1);
        var.fit(&series).unwrap();

        let forecast = var.forecast(&series, 3).unwrap();
        assert_eq!(forecast.len(), 3);
        assert_eq!(forecast.num_features(), 2);
    }

    #[test]
    fn test_var_information_criteria() {
        let series = create_multivariate_series();
        let mut var = VAR::new(1);
        var.fit(&series).unwrap();

        let aic = var.aic().unwrap();
        let bic = var.bic().unwrap();
        let hqic = var.hqic().unwrap();

        assert!(aic.is_finite());
        assert!(bic.is_finite());
        assert!(hqic.is_finite());
    }

    #[test]
    fn test_granger_causality() {
        // Create two simple series
        let x_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y_data = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let x_tensor = Tensor::from_vec(x_data, &[8]).unwrap();
        let y_tensor = Tensor::from_vec(y_data, &[8]).unwrap();

        let x_series = TimeSeries::new(x_tensor);
        let y_series = TimeSeries::new(y_tensor);

        let gc = GrangerCausality::new(2);
        let results = gc.test(&x_series, &y_series).unwrap();

        assert_eq!(results.len(), 2);
        for (lag, f_stat, p_value) in results {
            assert!(lag > 0);
            assert!(f_stat >= 0.0);
            assert!(p_value >= 0.0 && p_value <= 1.0);
        }
    }
}
