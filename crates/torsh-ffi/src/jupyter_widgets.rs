//! Jupyter widgets integration for ToRSh tensors
//!
//! This module provides interactive Jupyter widgets for visualizing and manipulating
//! ToRSh tensors directly within Jupyter notebooks, enabling real-time data exploration
//! and model analysis.

use crate::pandas_support::PandasSupport;
use crate::plotting_utilities::{PlotResult, PlottingUtilities};
use crate::tensor::PyTensor;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};
use pyo3::Bound;
use std::collections::HashMap;

/// Jupyter widgets integration for interactive data exploration
#[pyclass(name = "JupyterWidgets")]
#[derive(Debug)]
pub struct JupyterWidgets {
    /// Plotting utilities for visualization
    plotting: PlottingUtilities,
    /// Pandas support for data manipulation
    pandas: PandasSupport,
    /// Cached widget modules
    widget_modules: HashMap<String, Py<PyModule>>,
    /// Widget configuration
    config: WidgetConfig,
}

/// Configuration for Jupyter widgets
#[derive(Debug, Clone)]
pub struct WidgetConfig {
    /// Default widget layout
    pub layout: WidgetLayout,
    /// Enable real-time updates
    pub realtime_updates: bool,
    /// Update frequency in milliseconds
    pub update_frequency: u64,
    /// Enable interactive controls
    pub interactive_controls: bool,
    /// Widget theme
    pub theme: String,
}

/// Widget layout configuration
#[derive(Debug, Clone)]
pub struct WidgetLayout {
    /// Widget width
    pub width: String,
    /// Widget height
    pub height: String,
    /// Border configuration
    pub border: String,
    /// Margin configuration
    pub margin: String,
    /// Padding configuration
    pub padding: String,
}

impl Default for WidgetConfig {
    fn default() -> Self {
        Self {
            layout: WidgetLayout {
                width: "100%".to_string(),
                height: "400px".to_string(),
                border: "1px solid #ccc".to_string(),
                margin: "10px".to_string(),
                padding: "10px".to_string(),
            },
            realtime_updates: true,
            update_frequency: 100, // 100ms
            interactive_controls: true,
            theme: "light".to_string(),
        }
    }
}

/// Interactive tensor visualization widget
#[pyclass(name = "TensorVisualizationWidget")]
#[derive(Debug, Clone)]
pub struct TensorVisualizationWidget {
    /// Widget ID
    #[pyo3(get)]
    pub widget_id: String,
    /// Current tensor data
    #[pyo3(get)]
    pub tensor_data: PyTensor,
    /// Visualization type
    #[pyo3(get)]
    pub viz_type: String,
    /// Widget configuration
    #[pyo3(get)]
    pub config: HashMap<String, String>,
    /// Interactive controls state
    #[pyo3(get)]
    pub controls: HashMap<String, f64>,
}

/// Real-time monitoring widget for training metrics
#[pyclass(name = "TrainingMonitorWidget")]
#[derive(Debug, Clone)]
pub struct TrainingMonitorWidget {
    /// Widget ID
    #[pyo3(get)]
    pub widget_id: String,
    /// Training metrics history
    #[pyo3(get)]
    pub metrics_history: HashMap<String, Vec<f64>>,
    /// Current epoch
    #[pyo3(get)]
    pub current_epoch: usize,
    /// Widget layout
    #[pyo3(get)]
    pub layout: HashMap<String, String>,
    /// Auto-refresh enabled
    #[pyo3(get)]
    pub auto_refresh: bool,
}

/// Interactive data exploration widget
#[pyclass(name = "DataExplorationWidget")]
#[derive(Debug, Clone)]
pub struct DataExplorationWidget {
    /// Widget ID
    #[pyo3(get)]
    pub widget_id: String,
    /// Current dataset
    #[pyo3(get)]
    pub dataset: PyTensor,
    /// Selected features
    #[pyo3(get)]
    pub selected_features: Vec<usize>,
    /// Filter criteria
    #[pyo3(get)]
    pub filters: HashMap<String, String>,
    /// Visualization settings
    #[pyo3(get)]
    pub viz_settings: HashMap<String, String>,
}

#[pymethods]
impl JupyterWidgets {
    /// Create a new Jupyter widgets instance
    #[new]
    pub fn new() -> PyResult<Self> {
        Ok(Self {
            plotting: PlottingUtilities::new()?,
            pandas: PandasSupport::new()?,
            widget_modules: HashMap::new(),
            config: WidgetConfig::default(),
        })
    }

    /// Configure widget settings
    pub fn configure(&mut self, _py: Python, config: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(realtime) = config.get_item("realtime_updates")? {
            self.config.realtime_updates = realtime.extract()?;
        }
        if let Some(frequency) = config.get_item("update_frequency")? {
            self.config.update_frequency = frequency.extract()?;
        }
        if let Some(interactive) = config.get_item("interactive_controls")? {
            self.config.interactive_controls = interactive.extract()?;
        }
        if let Some(theme) = config.get_item("theme")? {
            self.config.theme = theme.extract()?;
        }
        Ok(())
    }

    /// Create an interactive tensor visualization widget
    pub fn create_tensor_widget(
        &self,
        py: Python,
        tensor: &PyTensor,
        viz_type: &str,
        title: Option<&str>,
    ) -> PyResult<TensorVisualizationWidget> {
        let ipywidgets = self.get_widget_module(py, "ipywidgets")?;
        let widget_id = format!("tensor_viz_{}", uuid::Uuid::new_v4().simple());

        // Create interactive plot based on tensor dimensions
        let plot_result = match tensor.shape().len() {
            1 => {
                // 1D tensor - line plot or histogram
                match viz_type {
                    "line" => {
                        let x = PyTensor::arange(0.0, tensor.shape()[0] as f64, 1.0, None)?;
                        self.plotting.line_plot(
                            py,
                            &x,
                            tensor,
                            title,
                            Some("Index"),
                            Some("Value"),
                            None,
                        )?
                    }
                    "histogram" => self.plotting.histogram(
                        py,
                        tensor,
                        Some(30),
                        Some(false),
                        title,
                        Some("Value"),
                        Some("Frequency"),
                    )?,
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Unsupported 1D visualization type: {}",
                            viz_type
                        )))
                    }
                }
            }
            2 => {
                // 2D tensor - heatmap or scatter plot
                match viz_type {
                    "heatmap" => {
                        self.plotting
                            .heatmap(py, tensor, title, Some("viridis"), Some(true))?
                    }
                    "scatter" => {
                        let shape = tensor.shape();
                        if shape[1] >= 2 {
                            let x = tensor.select(1, 0)?;
                            let y = tensor.select(1, 1)?;
                            self.plotting.scatter_plot(
                                py,
                                &x,
                                &y,
                                None,
                                None,
                                title,
                                Some("X"),
                                Some("Y"),
                            )?
                        } else {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "2D tensor needs at least 2 columns for scatter plot",
                            ));
                        }
                    }
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Unsupported 2D visualization type: {}",
                            viz_type
                        )))
                    }
                }
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Tensors with more than 2 dimensions require dimension reduction for visualization"
                ));
            }
        };

        // Create widget configuration
        let mut config = HashMap::new();
        config.insert("width".to_string(), self.config.layout.width.clone());
        config.insert("height".to_string(), self.config.layout.height.clone());
        config.insert("theme".to_string(), self.config.theme.clone());

        // Create interactive controls if enabled
        let mut controls = HashMap::new();
        if self.config.interactive_controls {
            controls.insert("zoom".to_string(), 1.0);
            controls.insert("pan_x".to_string(), 0.0);
            controls.insert("pan_y".to_string(), 0.0);
            controls.insert("rotation".to_string(), 0.0);
        }

        Ok(TensorVisualizationWidget {
            widget_id,
            tensor_data: tensor.clone(),
            viz_type: viz_type.to_string(),
            config,
            controls,
        })
    }

    /// Create a training monitoring widget
    pub fn create_training_monitor(
        &self,
        _py: Python,
        metrics: Vec<String>,
        title: Option<&str>,
    ) -> PyResult<TrainingMonitorWidget> {
        let widget_id = format!("training_monitor_{}", uuid::Uuid::new_v4().simple());

        // Initialize metrics history
        let mut metrics_history = HashMap::new();
        for metric in &metrics {
            metrics_history.insert(metric.clone(), Vec::new());
        }

        // Create layout configuration
        let mut layout = HashMap::new();
        layout.insert("width".to_string(), self.config.layout.width.clone());
        layout.insert("height".to_string(), self.config.layout.height.clone());
        layout.insert(
            "title".to_string(),
            title.unwrap_or("Training Metrics").to_string(),
        );

        Ok(TrainingMonitorWidget {
            widget_id,
            metrics_history,
            current_epoch: 0,
            layout,
            auto_refresh: self.config.realtime_updates,
        })
    }

    /// Update training monitor with new metrics
    pub fn update_training_monitor(
        &self,
        py: Python,
        monitor: &mut TrainingMonitorWidget,
        epoch: usize,
        metrics: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        monitor.current_epoch = epoch;

        // Update metrics history
        for (key, value) in metrics.iter() {
            let metric_name: String = key.extract()?;
            let metric_value: f64 = value.extract()?;

            if let Some(history) = monitor.metrics_history.get_mut(&metric_name) {
                history.push(metric_value);
            } else {
                monitor
                    .metrics_history
                    .insert(metric_name, vec![metric_value]);
            }
        }

        // Create updated plot
        if monitor.auto_refresh {
            self.refresh_training_monitor(py, monitor)?;
        }

        Ok(())
    }

    /// Refresh training monitor visualization
    pub fn refresh_training_monitor(
        &self,
        py: Python,
        monitor: &TrainingMonitorWidget,
    ) -> PyResult<PlotResult> {
        let num_metrics = monitor.metrics_history.len();
        if num_metrics == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No metrics to display",
            ));
        }

        // Create subplots for each metric
        let epochs: Vec<f64> = (0..=monitor.current_epoch).map(|i| i as f64).collect();
        let epochs_tensor = PyTensor::from_vec(epochs, None)?;

        // For now, create a single plot with the first metric
        // In a full implementation, this would create subplots
        if let Some((metric_name, values)) = monitor.metrics_history.iter().next() {
            if values.len() > 0 {
                let values_tensor = PyTensor::from_vec(values.clone(), None)?;
                return self.plotting.line_plot(
                    py,
                    &epochs_tensor,
                    &values_tensor,
                    Some(&format!("Training Progress - {}", metric_name)),
                    Some("Epoch"),
                    Some(metric_name),
                    Some("-o"),
                );
            }
        }

        Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "No valid metrics data to display",
        ))
    }

    /// Create a data exploration widget
    pub fn create_data_explorer(
        &self,
        _py: Python,
        dataset: &PyTensor,
        _feature_names: Option<Vec<String>>,
    ) -> PyResult<DataExplorationWidget> {
        let widget_id = format!("data_explorer_{}", uuid::Uuid::new_v4().simple());

        // Initialize selected features (all by default)
        let binding = dataset.shape();
        let num_features = binding.last().unwrap_or(&1);
        let selected_features: Vec<usize> = (0..*num_features).collect();

        // Initialize filters and visualization settings
        let filters = HashMap::new();
        let mut viz_settings = HashMap::new();
        viz_settings.insert("plot_type".to_string(), "scatter".to_string());
        viz_settings.insert("color_scheme".to_string(), "viridis".to_string());
        viz_settings.insert("point_size".to_string(), "5".to_string());

        Ok(DataExplorationWidget {
            widget_id,
            dataset: dataset.clone(),
            selected_features,
            filters,
            viz_settings,
        })
    }

    /// Update data exploration visualization
    pub fn update_data_explorer(
        &self,
        py: Python,
        explorer: &mut DataExplorationWidget,
        selected_features: Option<Vec<usize>>,
        viz_type: Option<&str>,
    ) -> PyResult<PlotResult> {
        if let Some(features) = selected_features {
            explorer.selected_features = features;
        }

        if let Some(viz) = viz_type {
            explorer
                .viz_settings
                .insert("plot_type".to_string(), viz.to_string());
        }

        // Create visualization based on selected features
        let plot_type = explorer
            .viz_settings
            .get("plot_type")
            .unwrap_or(&"scatter".to_string())
            .clone();

        match plot_type.as_str() {
            "scatter" => {
                if explorer.selected_features.len() >= 2 {
                    let x_idx = explorer.selected_features[0];
                    let y_idx = explorer.selected_features[1];

                    let x = explorer.dataset.select(1, x_idx as i32)?;
                    let y = explorer.dataset.select(1, y_idx as i32)?;

                    self.plotting.scatter_plot(
                        py,
                        &x,
                        &y,
                        None,
                        None,
                        Some("Data Exploration - Scatter Plot"),
                        Some(&format!("Feature {}", x_idx)),
                        Some(&format!("Feature {}", y_idx)),
                    )
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Scatter plot requires at least 2 selected features",
                    ))
                }
            }
            "histogram" => {
                if !explorer.selected_features.is_empty() {
                    let feature_idx = explorer.selected_features[0];
                    let feature_data = explorer.dataset.select(1, feature_idx as i32)?;

                    self.plotting.histogram(
                        py,
                        &feature_data,
                        Some(30),
                        Some(false),
                        Some(&format!("Feature {} Distribution", feature_idx)),
                        Some("Value"),
                        Some("Frequency"),
                    )
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Histogram requires at least 1 selected feature",
                    ))
                }
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unsupported visualization type: {}",
                plot_type
            ))),
        }
    }

    /// Create interactive parameter tuning widget
    pub fn create_parameter_tuner(
        &self,
        py: Python,
        parameters: &Bound<'_, PyDict>,
        callback: Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let ipywidgets = self.get_widget_module(py, "ipywidgets")?;

        // Create interactive widgets for each parameter
        let mut widgets = Vec::new();

        for (key, value) in parameters.iter() {
            let param_name: String = key.extract()?;

            // Create appropriate widget based on parameter type
            let widget = if value.is_instance_of::<pyo3::types::PyFloat>() {
                let float_val: f64 = value.extract()?;
                ipywidgets.call_method(
                    py,
                    "FloatSlider",
                    (),
                    Some(
                        [
                            ("description", param_name.into_pyobject(py)),
                            ("value", float_val.into_pyobject(py)),
                            ("min", (float_val * 0.1).into_pyobject(py)),
                            ("max", (float_val * 2.0).into_pyobject(py)),
                            ("step", (float_val * 0.01).into_pyobject(py)),
                        ]
                        .into_py_dict(py),
                    ),
                )?
            } else if value.is_instance_of::<pyo3::types::PyInt>() {
                let int_val: i64 = value.extract()?;
                ipywidgets.call_method(
                    py,
                    "IntSlider",
                    (),
                    Some(
                        [
                            ("description", param_name.into_pyobject(py)),
                            ("value", int_val.into_pyobject(py)),
                            ("min", (int_val / 10).into_pyobject(py)),
                            ("max", (int_val * 2).into_pyobject(py)),
                            ("step", 1.into_pyobject(py)),
                        ]
                        .into_py_dict(py),
                    ),
                )?
            } else {
                continue; // Skip unsupported parameter types
            };

            widgets.push(widget);
        }

        // Create interactive widget with callback
        let interactive = ipywidgets.call_method(py, "interactive", (callback,), None)?;

        Ok(interactive)
    }

    /// Export widget as HTML
    pub fn export_widget_html(&self, py: Python, widget: Bound<'_, PyAny>, filename: &str) -> PyResult<()> {
        let embed = self.get_widget_module(py, "ipywidgets.embed")?;
        embed.call_method(py, "embed_minimal_html", (filename, widget), None)?;
        Ok(())
    }

    /// Get available widget themes
    pub fn get_available_themes(&self) -> Vec<String> {
        vec![
            "light".to_string(),
            "dark".to_string(),
            "jupyter".to_string(),
            "colab".to_string(),
        ]
    }
}

impl JupyterWidgets {
    /// Get or cache widget module
    fn get_widget_module(&self, py: Python, module_name: &str) -> PyResult<Py<PyModule>> {
        if let Some(module) = self.widget_modules.get(module_name) {
            return Ok(module.clone_ref(py));
        }

        let module = py.import(module_name)?;
        Ok(module.into())
    }
}

/// Create Jupyter widgets utilities
pub fn create_jupyter_utilities(py: Python) -> PyResult<Bound<PyDict>> {
    let utils = PyDict::new(py);

    // Add utility functions
    utils.set_item("create_widgets", py.get_type::<JupyterWidgets>())?;
    utils.set_item(
        "TensorVisualizationWidget",
        py.get_type::<TensorVisualizationWidget>(),
    )?;
    utils.set_item(
        "TrainingMonitorWidget",
        py.get_type::<TrainingMonitorWidget>(),
    )?;
    utils.set_item(
        "DataExplorationWidget",
        py.get_type::<DataExplorationWidget>(),
    )?;

    Ok(utils)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_widget_config() {
        let config = WidgetConfig::default();
        assert_eq!(config.layout.width, "100%");
        assert_eq!(config.layout.height, "400px");
        assert!(config.realtime_updates);
        assert!(config.interactive_controls);
    }

    #[test]
    fn test_widget_layout() {
        let layout = WidgetLayout {
            width: "800px".to_string(),
            height: "600px".to_string(),
            border: "2px solid #000".to_string(),
            margin: "5px".to_string(),
            padding: "15px".to_string(),
        };
        assert_eq!(layout.width, "800px");
        assert_eq!(layout.height, "600px");
    }
}
