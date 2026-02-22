//! Exponential smoothing models

use crate::TimeSeries;
use torsh_tensor::creation::zeros;

/// Holt-Winters exponential smoothing
pub struct HoltWinters {
    seasonal: Option<String>, // "additive" or "multiplicative"
    period: Option<usize>,
    alpha: f32, // Level smoothing
    beta: f32,  // Trend smoothing
    gamma: f32, // Seasonal smoothing
    // Fitted state
    level: Option<f32>,
    trend: Option<f32>,
    seasonal_components: Option<Vec<f32>>,
}

impl HoltWinters {
    /// Create a new Holt-Winters model
    pub fn new() -> Self {
        Self {
            seasonal: None,
            period: None,
            alpha: 0.3,
            beta: 0.1,
            gamma: 0.1,
            level: None,
            trend: None,
            seasonal_components: None,
        }
    }

    /// Set seasonal component type
    pub fn seasonal(mut self, seasonal_type: &str, period: usize) -> Self {
        self.seasonal = Some(seasonal_type.to_string());
        self.period = Some(period);
        self
    }

    /// Set smoothing parameters
    pub fn with_params(mut self, alpha: f32, beta: f32, gamma: f32) -> Self {
        self.alpha = alpha;
        self.beta = beta;
        self.gamma = gamma;
        self
    }

    /// Get level smoothing parameter
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Get trend smoothing parameter
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// Get seasonal smoothing parameter
    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Get seasonal configuration
    pub fn seasonal_config(&self) -> (Option<&str>, Option<usize>) {
        (self.seasonal.as_deref(), self.period)
    }

    /// Fit the model
    ///
    /// Implements Holt-Winters exponential smoothing with optional seasonal component.
    ///
    /// # Formulas
    ///
    /// **No Seasonality (Holt's method):**
    /// - Level: l_t = α * y_t + (1 - α) * (l_(t-1) + b_(t-1))
    /// - Trend: b_t = β * (l_t - l_(t-1)) + (1 - β) * b_(t-1)
    ///
    /// **Additive Seasonality:**
    /// - Level: l_t = α * (y_t - s_(t-m)) + (1 - α) * (l_(t-1) + b_(t-1))
    /// - Trend: b_t = β * (l_t - l_(t-1)) + (1 - β) * b_(t-1)
    /// - Seasonal: s_t = γ * (y_t - l_t) + (1 - γ) * s_(t-m)
    ///
    /// **Multiplicative Seasonality:**
    /// - Level: l_t = α * (y_t / s_(t-m)) + (1 - α) * (l_(t-1) + b_(t-1))
    /// - Trend: b_t = β * (l_t - l_(t-1)) + (1 - β) * b_(t-1)
    /// - Seasonal: s_t = γ * (y_t / l_t) + (1 - γ) * s_(t-m)
    pub fn fit(&mut self, series: &TimeSeries) {
        let data = series.values.to_vec().unwrap_or_default();
        let n = data.len();

        if n < 2 {
            self.level = None;
            self.trend = None;
            self.seasonal_components = None;
            return;
        }

        // Determine if we have seasonality
        let has_seasonal = self.seasonal.is_some() && self.period.is_some();
        let period = self.period.unwrap_or(1);
        let is_multiplicative = self
            .seasonal
            .as_ref()
            .map(|s| s == "multiplicative")
            .unwrap_or(false);

        if has_seasonal && n < period * 2 {
            // Not enough data for seasonal model
            self.level = Some(data[n - 1]);
            self.trend = Some(0.0);
            self.seasonal_components = Some(vec![1.0; period]);
            return;
        }

        // Initialize level and trend
        let mut level = if has_seasonal {
            // Initialize level as mean of first season
            data[..period.min(n)].iter().sum::<f32>() / period.min(n) as f32
        } else {
            data[0]
        };

        let mut trend = if n >= 2 {
            if has_seasonal && n >= period * 2 {
                // Initialize trend from difference of seasonal averages
                let first_season: f32 = data[..period].iter().sum::<f32>() / period as f32;
                let second_season: f32 =
                    data[period..(2 * period).min(n)].iter().sum::<f32>() / period as f32;
                (second_season - first_season) / period as f32
            } else {
                data[1] - data[0]
            }
        } else {
            0.0
        };

        // Initialize seasonal components
        let mut seasonal_comp = if has_seasonal {
            let mut s = vec![0.0; period];
            // Initialize with deseasonalized values
            for i in 0..period.min(n) {
                if is_multiplicative && level.abs() > 1e-8 {
                    s[i] = data[i] / level;
                } else {
                    s[i] = data[i] - level;
                }
            }
            // Normalize: additive should sum to 0, multiplicative should average to 1
            if is_multiplicative {
                let mean = s[..period.min(n)].iter().sum::<f32>() / period.min(n) as f32;
                if mean.abs() > 1e-8 {
                    for i in 0..period.min(n) {
                        s[i] /= mean;
                    }
                }
            } else {
                let mean = s[..period.min(n)].iter().sum::<f32>() / period.min(n) as f32;
                for i in 0..period.min(n) {
                    s[i] -= mean;
                }
            }
            s
        } else {
            vec![]
        };

        // Apply Holt-Winters smoothing equations
        for t in 0..n {
            let y_t = data[t];
            let s_idx = t % period;

            let (deseasonalized, prev_seasonal) = if has_seasonal && t >= period {
                let prev_s = seasonal_comp[(t - period) % period];
                if is_multiplicative {
                    if prev_s.abs() > 1e-8 {
                        (y_t / prev_s, prev_s)
                    } else {
                        (y_t, 1.0)
                    }
                } else {
                    (y_t - prev_s, prev_s)
                }
            } else if has_seasonal {
                // Use initial seasonal component
                let prev_s = seasonal_comp[s_idx];
                if is_multiplicative {
                    if prev_s.abs() > 1e-8 {
                        (y_t / prev_s, prev_s)
                    } else {
                        (y_t, 1.0)
                    }
                } else {
                    (y_t - prev_s, prev_s)
                }
            } else {
                (y_t, 0.0)
            };

            let prev_level = level;

            // Update level
            level = self.alpha * deseasonalized + (1.0 - self.alpha) * (prev_level + trend);

            // Update trend
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend;

            // Update seasonal component
            if has_seasonal {
                if is_multiplicative {
                    if level.abs() > 1e-8 {
                        seasonal_comp[s_idx] =
                            self.gamma * (y_t / level) + (1.0 - self.gamma) * prev_seasonal;
                    }
                } else {
                    seasonal_comp[s_idx] =
                        self.gamma * (y_t - level) + (1.0 - self.gamma) * prev_seasonal;
                }
            }
        }

        self.level = Some(level);
        self.trend = Some(trend);
        self.seasonal_components = if has_seasonal {
            Some(seasonal_comp)
        } else {
            None
        };
    }

    /// Forecast future values
    ///
    /// Generates forecasts using the fitted Holt-Winters model.
    ///
    /// # Formulas
    ///
    /// **No Seasonality (Holt's method):**
    /// - ŷ_(t+h) = l_t + h * b_t
    ///
    /// **Additive Seasonality:**
    /// - ŷ_(t+h) = l_t + h * b_t + s_(t+h-m)
    ///
    /// **Multiplicative Seasonality:**
    /// - ŷ_(t+h) = (l_t + h * b_t) * s_(t+h-m)
    ///
    /// where m is the seasonal period
    pub fn forecast(&self, steps: usize) -> TimeSeries {
        let level = self.level.unwrap_or(0.0);
        let trend = self.trend.unwrap_or(0.0);

        let has_seasonal = self.seasonal_components.is_some();
        let period = self.period.unwrap_or(1);
        let is_multiplicative = self
            .seasonal
            .as_ref()
            .map(|s| s == "multiplicative")
            .unwrap_or(false);

        let forecast_values: Vec<f32> = (1..=steps)
            .map(|h| {
                // Base forecast (level + trend)
                let base = level + (h as f32) * trend;

                // Apply seasonal component if present
                if has_seasonal {
                    if let Some(ref seasonal) = self.seasonal_components {
                        let s_idx = (h - 1) % period;
                        let seasonal_factor = seasonal[s_idx];

                        if is_multiplicative {
                            base * seasonal_factor
                        } else {
                            base + seasonal_factor
                        }
                    } else {
                        base
                    }
                } else {
                    base
                }
            })
            .collect();

        let forecast_tensor = torsh_tensor::Tensor::from_vec(forecast_values, &[steps])
            .expect("tensor creation should succeed");
        TimeSeries::new(forecast_tensor)
    }

    /// Get model state (level, trend, seasonal)
    ///
    /// Returns the fitted state of the model after calling `fit()`:
    /// - level: The final smoothed level component
    /// - trend: The final smoothed trend component
    /// - seasonal: The final seasonal component vector (length = period)
    ///
    /// Returns (None, None, None) if the model has not been fitted.
    pub fn state(&self) -> (Option<f32>, Option<f32>, Option<Vec<f32>>) {
        (self.level, self.trend, self.seasonal_components.clone())
    }
}

impl Default for HoltWinters {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple exponential smoothing
pub struct SimpleExpSmoothing {
    alpha: f32,
    level: Option<f32>,
}

/// Type alias for SimpleExpSmoothing
pub type SimpleExponentialSmoothing = SimpleExpSmoothing;

impl SimpleExpSmoothing {
    /// Create a new simple exponential smoothing model
    pub fn new(alpha: f32) -> Self {
        Self { alpha, level: None }
    }

    /// Get smoothing parameter
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Set smoothing parameter
    pub fn set_alpha(&mut self, alpha: f32) {
        self.alpha = alpha;
    }

    /// Get current level
    pub fn level(&self) -> Option<f32> {
        self.level
    }

    /// Fit and forecast
    ///
    /// Performs simple exponential smoothing on the series and forecasts future values.
    /// Formula: s_t = α * y_t + (1 - α) * s_(t-1)
    /// Forecast: ŷ_(t+h) = s_t (flat forecast)
    pub fn fit_predict(&self, series: &TimeSeries, steps: usize) -> TimeSeries {
        // Fit the model to get the final smoothed level
        let data = series.values.to_vec().unwrap_or_default();
        if data.is_empty() {
            let values = zeros(&[steps, 1]).expect("tensor creation should succeed");
            return TimeSeries::new(values);
        }

        // Initialize with first observation
        let mut level = data[0];

        // Apply exponential smoothing
        for &y_t in &data[1..] {
            level = self.alpha * y_t + (1.0 - self.alpha) * level;
        }

        // Forecast: all future values equal the final level (flat forecast)
        let forecast_values: Vec<f32> = vec![level; steps];
        let forecast_tensor = torsh_tensor::Tensor::from_vec(forecast_values, &[steps])
            .expect("tensor creation should succeed");
        TimeSeries::new(forecast_tensor)
    }

    /// Fit the model
    ///
    /// Computes the smoothed level from the time series and stores it.
    pub fn fit(&mut self, series: &TimeSeries) {
        let data = series.values.to_vec().unwrap_or_default();
        if data.is_empty() {
            self.level = None;
            return;
        }

        // Initialize with first observation
        let mut level = data[0];

        // Apply exponential smoothing: s_t = α * y_t + (1 - α) * s_(t-1)
        for &y_t in &data[1..] {
            level = self.alpha * y_t + (1.0 - self.alpha) * level;
        }

        self.level = Some(level);
    }

    /// Forecast future values
    ///
    /// Uses the fitted level to generate flat forecasts.
    pub fn forecast(&self, steps: usize) -> TimeSeries {
        let level = self.level.unwrap_or(0.0);

        // Simple exponential smoothing produces flat forecasts
        let forecast_values: Vec<f32> = vec![level; steps];
        let forecast_tensor = torsh_tensor::Tensor::from_vec(forecast_values, &[steps])
            .expect("tensor creation should succeed");
        TimeSeries::new(forecast_tensor)
    }
}

/// Double exponential smoothing (Holt's method)
pub struct DoubleExpSmoothing {
    alpha: f32, // Level smoothing
    beta: f32,  // Trend smoothing
    level: Option<f32>,
    trend: Option<f32>,
}

impl DoubleExpSmoothing {
    /// Create a new double exponential smoothing model
    pub fn new(alpha: f32, beta: f32) -> Self {
        Self {
            alpha,
            beta,
            level: None,
            trend: None,
        }
    }

    /// Get smoothing parameters
    pub fn params(&self) -> (f32, f32) {
        (self.alpha, self.beta)
    }

    /// Get current state
    pub fn state(&self) -> (Option<f32>, Option<f32>) {
        (self.level, self.trend)
    }

    /// Fit the model using Holt's double exponential smoothing
    ///
    /// Computes both level and trend components:
    /// - Level: l_t = α * y_t + (1 - α) * (l_(t-1) + b_(t-1))
    /// - Trend: b_t = β * (l_t - l_(t-1)) + (1 - β) * b_(t-1))
    pub fn fit(&mut self, series: &TimeSeries) {
        let data = series.values.to_vec().unwrap_or_default();
        if data.len() < 2 {
            self.level = None;
            self.trend = None;
            return;
        }

        // Initialize level with first observation
        let mut level = data[0];

        // Initialize trend with the difference between first two observations
        let mut trend = data[1] - data[0];

        // Apply Holt's method to all observations
        for &y_t in &data[1..] {
            let prev_level = level;

            // Update level: l_t = α * y_t + (1 - α) * (l_(t-1) + b_(t-1))
            level = self.alpha * y_t + (1.0 - self.alpha) * (prev_level + trend);

            // Update trend: b_t = β * (l_t - l_(t-1)) + (1 - β) * b_(t-1)
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend;
        }

        self.level = Some(level);
        self.trend = Some(trend);
    }

    /// Forecast future values using fitted level and trend
    ///
    /// Formula: ŷ_(t+h) = l_t + h * b_t
    /// where l_t is the level and b_t is the trend at time t
    pub fn forecast(&self, steps: usize) -> TimeSeries {
        let level = self.level.unwrap_or(0.0);
        let trend = self.trend.unwrap_or(0.0);

        // Generate forecasts: forecast(h) = level + h * trend
        let forecast_values: Vec<f32> = (1..=steps).map(|h| level + (h as f32) * trend).collect();

        let forecast_tensor = torsh_tensor::Tensor::from_vec(forecast_values, &[steps])
            .expect("tensor creation should succeed");
        TimeSeries::new(forecast_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_tensor::Tensor;

    fn create_test_series() -> TimeSeries {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let tensor = Tensor::from_vec(data, &[8]).unwrap();
        TimeSeries::new(tensor)
    }

    #[test]
    fn test_holt_winters_creation() {
        let hw = HoltWinters::new();
        assert_eq!(hw.alpha(), 0.3);
        assert_eq!(hw.beta(), 0.1);
        assert_eq!(hw.gamma(), 0.1);
        assert_eq!(hw.seasonal_config(), (None, None));
    }

    #[test]
    fn test_holt_winters_seasonal() {
        let hw = HoltWinters::new().seasonal("additive", 12);
        let (seasonal_type, period) = hw.seasonal_config();
        assert_eq!(seasonal_type, Some("additive"));
        assert_eq!(period, Some(12));
    }

    #[test]
    fn test_holt_winters_params() {
        let hw = HoltWinters::new().with_params(0.5, 0.2, 0.3);
        assert_eq!(hw.alpha(), 0.5);
        assert_eq!(hw.beta(), 0.2);
        assert_eq!(hw.gamma(), 0.3);
    }

    #[test]
    fn test_holt_winters_forecast() {
        let series = create_test_series();
        let mut hw = HoltWinters::new();
        hw.fit(&series);
        let forecast = hw.forecast(3);

        assert_eq!(forecast.len(), 3);
    }

    #[test]
    fn test_simple_exp_smoothing_creation() {
        let ses = SimpleExpSmoothing::new(0.3);
        assert_eq!(ses.alpha(), 0.3);
        assert_eq!(ses.level(), None);
    }

    #[test]
    fn test_simple_exp_smoothing_params() {
        let mut ses = SimpleExpSmoothing::new(0.3);
        ses.set_alpha(0.5);
        assert_eq!(ses.alpha(), 0.5);
    }

    #[test]
    fn test_simple_exp_smoothing_forecast() {
        let series = create_test_series();
        let ses = SimpleExpSmoothing::new(0.3);
        let forecast = ses.fit_predict(&series, 2);

        assert_eq!(forecast.len(), 2);
    }

    #[test]
    fn test_double_exp_smoothing_creation() {
        let des = DoubleExpSmoothing::new(0.3, 0.1);
        let (alpha, beta) = des.params();
        assert_eq!(alpha, 0.3);
        assert_eq!(beta, 0.1);
        assert_eq!(des.state(), (None, None));
    }

    #[test]
    fn test_double_exp_smoothing_forecast() {
        let series = create_test_series();
        let mut des = DoubleExpSmoothing::new(0.3, 0.1);
        des.fit(&series);
        let forecast = des.forecast(3);

        assert_eq!(forecast.len(), 3);
    }

    #[test]
    fn test_holt_winters_default() {
        let hw = HoltWinters::default();
        assert_eq!(hw.alpha(), 0.3);
        assert_eq!(hw.beta(), 0.1);
        assert_eq!(hw.gamma(), 0.1);
    }

    #[test]
    fn test_holt_winters_fit_updates_state() {
        let series = create_test_series(); // [1,2,3,4,5,6,7,8]
        let mut hw = HoltWinters::new();

        // Before fitting, state should be None
        assert_eq!(hw.state(), (None, None, None));

        // Fit the model
        hw.fit(&series);

        // After fitting, state should be Some
        let (level, trend, seasonal) = hw.state();
        assert!(level.is_some());
        assert!(trend.is_some());
        assert!(seasonal.is_none()); // No seasonality specified

        // Level should be positive (series is trending upwards)
        assert!(level.unwrap() > 0.0);
        // Trend should be positive (series is increasing)
        assert!(trend.unwrap() > 0.0);
    }

    #[test]
    fn test_holt_winters_seasonal_additive() {
        // Create a series with clear seasonal pattern (period 4)
        let data = vec![
            1.0f32, 2.0, 3.0, 4.0, // First cycle
            2.0, 3.0, 4.0, 5.0, // Second cycle (with trend)
            3.0, 4.0, 5.0, 6.0, // Third cycle (with trend)
        ];
        let tensor = Tensor::from_vec(data, &[12]).unwrap();
        let series = TimeSeries::new(tensor);

        let mut hw = HoltWinters::new()
            .seasonal("additive", 4)
            .with_params(0.3, 0.1, 0.2);
        hw.fit(&series);

        let (level, trend, seasonal) = hw.state();
        assert!(level.is_some());
        assert!(trend.is_some());
        assert!(seasonal.is_some());

        // Seasonal component should have length = period
        let seasonal_comp = seasonal.unwrap();
        assert_eq!(seasonal_comp.len(), 4);

        // Forecast should use seasonal pattern
        let forecast = hw.forecast(8);
        assert_eq!(forecast.len(), 8);
    }

    #[test]
    fn test_holt_winters_seasonal_multiplicative() {
        // Create a series with multiplicative seasonality (period 4)
        let data = vec![
            1.0f32, 2.0, 1.5, 0.5, // First cycle
            2.0, 4.0, 3.0, 1.0, // Second cycle (scaled up)
            3.0, 6.0, 4.5, 1.5, // Third cycle (scaled up more)
        ];
        let tensor = Tensor::from_vec(data, &[12]).unwrap();
        let series = TimeSeries::new(tensor);

        let mut hw = HoltWinters::new()
            .seasonal("multiplicative", 4)
            .with_params(0.3, 0.1, 0.2);
        hw.fit(&series);

        let (level, trend, seasonal) = hw.state();
        assert!(level.is_some());
        assert!(trend.is_some());
        assert!(seasonal.is_some());

        // Seasonal component should have length = period
        let seasonal_comp = seasonal.unwrap();
        assert_eq!(seasonal_comp.len(), 4);

        // For multiplicative model, seasonal components should average to ~1.0
        let avg: f32 = seasonal_comp.iter().sum::<f32>() / seasonal_comp.len() as f32;
        assert!((avg - 1.0).abs() < 0.5); // Allow some tolerance

        // Forecast should use multiplicative seasonal pattern
        let forecast = hw.forecast(4);
        assert_eq!(forecast.len(), 4);
    }

    #[test]
    fn test_holt_winters_no_seasonal() {
        // Test Holt's method (no seasonality)
        let series = create_test_series();
        let mut hw = HoltWinters::new().with_params(0.5, 0.2, 0.0);
        hw.fit(&series);

        let (level, trend, seasonal) = hw.state();
        assert!(level.is_some());
        assert!(trend.is_some());
        assert!(seasonal.is_none());

        // Forecast should follow linear trend
        let forecast = hw.forecast(3);
        assert_eq!(forecast.len(), 3);

        // Forecasts should be increasing (positive trend)
        let f_data = forecast.values.to_vec().unwrap();
        assert!(f_data[0] > 0.0);
        assert!(f_data[1] > f_data[0]);
        assert!(f_data[2] > f_data[1]);
    }
}
