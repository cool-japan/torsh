//! ARIMA (AutoRegressive Integrated Moving Average) models

use crate::TimeSeries;
use torsh_tensor::{creation::zeros, Tensor};

/// ARIMA model for time series forecasting
pub struct ARIMA {
    p: usize, // AR order
    d: usize, // Differencing order
    q: usize, // MA order
    seasonal_order: Option<(usize, usize, usize, usize)>,
}

impl ARIMA {
    /// Create a new ARIMA model
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self {
            p,
            d,
            q,
            seasonal_order: None,
        }
    }

    /// Add seasonal component (P, D, Q, s)
    pub fn seasonal(mut self, p: usize, d: usize, q: usize, s: usize) -> Self {
        self.seasonal_order = Some((p, d, q, s));
        self
    }

    /// Get AR order
    pub fn p(&self) -> usize {
        self.p
    }

    /// Get differencing order
    pub fn d(&self) -> usize {
        self.d
    }

    /// Get MA order
    pub fn q(&self) -> usize {
        self.q
    }

    /// Get seasonal order
    pub fn seasonal_order(&self) -> Option<(usize, usize, usize, usize)> {
        self.seasonal_order
    }

    /// Fit ARIMA model
    pub fn fit(&mut self, _series: &TimeSeries) {
        // TODO: Fit model using scirs2-series or implement MLE estimation
    }

    /// Forecast future values
    pub fn forecast(&self, steps: usize) -> TimeSeries {
        let values = zeros(&[steps, 1]).unwrap();
        TimeSeries::new(values)
    }

    /// Get model information criteria
    pub fn aic(&self) -> f64 {
        // TODO: Calculate Akaike Information Criterion
        0.0
    }

    /// Get Bayesian Information Criterion
    pub fn bic(&self) -> f64 {
        // TODO: Calculate Bayesian Information Criterion
        0.0
    }

    /// Get model residuals
    pub fn residuals(&self) -> Option<TimeSeries> {
        // TODO: Return fitted residuals
        None
    }
}

/// SARIMA model (Seasonal ARIMA)
pub struct SARIMA {
    arima: ARIMA,
}

impl SARIMA {
    /// Create a new SARIMA model
    pub fn new(
        p: usize,
        d: usize,
        q: usize,
        seasonal_p: usize,
        seasonal_d: usize,
        seasonal_q: usize,
        seasonal_period: usize,
    ) -> Self {
        Self {
            arima: ARIMA::new(p, d, q).seasonal(
                seasonal_p,
                seasonal_d,
                seasonal_q,
                seasonal_period,
            ),
        }
    }

    /// Fit SARIMA model
    pub fn fit(&mut self, series: &TimeSeries) {
        self.arima.fit(series);
    }

    /// Forecast future values
    pub fn forecast(&self, steps: usize) -> TimeSeries {
        self.arima.forecast(steps)
    }
}

/// Auto ARIMA model selection
pub struct AutoARIMA {
    max_p: usize,
    max_d: usize,
    max_q: usize,
    seasonal: bool,
    information_criterion: String,
}

impl AutoARIMA {
    /// Create a new Auto ARIMA selector
    pub fn new() -> Self {
        Self {
            max_p: 5,
            max_d: 2,
            max_q: 5,
            seasonal: true,
            information_criterion: "aic".to_string(),
        }
    }

    /// Set maximum orders to search
    pub fn with_max_order(mut self, max_p: usize, max_d: usize, max_q: usize) -> Self {
        self.max_p = max_p;
        self.max_d = max_d;
        self.max_q = max_q;
        self
    }

    /// Set information criterion for model selection
    pub fn with_criterion(mut self, criterion: &str) -> Self {
        self.information_criterion = criterion.to_string();
        self
    }

    /// Automatically select and fit best ARIMA model
    pub fn fit(&self, _series: &TimeSeries) -> ARIMA {
        // TODO: Implement grid search over parameter space
        // For now, return a simple ARIMA(1,1,1)
        ARIMA::new(1, 1, 1)
    }
}

impl Default for AutoARIMA {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::creation::*;

    fn create_test_series() -> TimeSeries {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::from_vec(data, &[8]).unwrap();
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_arima_creation() {
        let arima = ARIMA::new(2, 1, 1);
        assert_eq!(arima.p(), 2);
        assert_eq!(arima.d(), 1);
        assert_eq!(arima.q(), 1);
        assert!(arima.seasonal_order().is_none());
    }

    #[test]
    fn test_arima_seasonal() {
        let arima = ARIMA::new(1, 1, 1).seasonal(1, 1, 1, 12);
        assert_eq!(arima.seasonal_order(), Some((1, 1, 1, 12)));
    }

    #[test]
    fn test_arima_forecast() {
        let series = create_test_series();
        let mut arima = ARIMA::new(1, 1, 1);
        arima.fit(&series);
        let forecast = arima.forecast(5);

        assert_eq!(forecast.len(), 5);
        assert_eq!(forecast.num_features(), 1);
    }

    #[test]
    fn test_sarima_creation() {
        let sarima = SARIMA::new(1, 1, 1, 1, 1, 1, 12);
        let forecast = sarima.forecast(3);
        assert_eq!(forecast.len(), 3);
    }

    #[test]
    fn test_auto_arima() {
        let series = create_test_series();
        let auto_arima = AutoARIMA::new().with_max_order(3, 2, 3);
        let _model = auto_arima.fit(&series);

        // Test that fit completes without errors
    }
}
