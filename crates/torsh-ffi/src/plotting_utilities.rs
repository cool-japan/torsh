//! Plotting utilities for ToRSh tensors
//!
//! This module provides comprehensive plotting and visualization capabilities for ToRSh tensors,
//! integrating with popular Python plotting libraries like Matplotlib, Seaborn, and Plotly
//! to create publication-quality visualizations.

use crate::numpy_compatibility::NumpyCompat;
use crate::python::tensor::PyTensor;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule, PyTuple};
use pyo3::Bound;
use std::collections::HashMap;

/// Plotting utilities for data visualization
#[pyclass(name = "PlottingUtilities")]
#[derive(Debug)]
pub struct PlottingUtilities {
    /// NumPy compatibility layer for array operations
    numpy_compat: NumpyCompat,
    /// Cached plotting library modules
    plot_modules: HashMap<String, Py<PyModule>>,
    /// Default plotting configuration
    config: PlotConfig,
    /// Color schemes and palettes
    color_schemes: HashMap<String, Vec<String>>,
}

/// Configuration for plotting operations
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// Default figure size
    pub figure_size: (f64, f64),
    /// Default DPI for plots
    pub dpi: u32,
    /// Default color scheme
    pub color_scheme: String,
    /// Font configuration
    pub font_config: FontConfig,
    /// Grid configuration
    pub show_grid: bool,
    /// Interactive mode
    pub interactive: bool,
    /// Export format preferences
    pub export_formats: Vec<String>,
}

/// Font configuration for plots
#[derive(Debug, Clone)]
pub struct FontConfig {
    /// Font family
    pub family: String,
    /// Font size
    pub size: u32,
    /// Font weight
    pub weight: String,
}

impl Default for PlotConfig {
    fn default() -> Self {
        Self {
            figure_size: (10.0, 6.0),
            dpi: 300,
            color_scheme: "viridis".to_string(),
            font_config: FontConfig {
                family: "DejaVu Sans".to_string(),
                size: 12,
                weight: "normal".to_string(),
            },
            show_grid: true,
            interactive: false,
            export_formats: vec!["png".to_string(), "pdf".to_string(), "svg".to_string()],
        }
    }
}

/// Plot result containing figure and metadata
#[pyclass(name = "PlotResult")]
#[derive(Debug, Clone)]
pub struct PlotResult {
    /// Figure object from matplotlib
    #[pyo3(get)]
    pub figure: Option<Py<PyAny>>,
    /// Axes objects
    #[pyo3(get)]
    pub axes: Option<Py<PyAny>>,
    /// Plot metadata
    #[pyo3(get)]
    pub metadata: HashMap<String, String>,
    /// Export paths
    #[pyo3(get)]
    pub exported_paths: Vec<String>,
}

/// Statistical plot configuration
#[pyclass(name = "StatPlotConfig")]
#[derive(Debug, Clone)]
pub struct StatPlotConfig {
    /// Show confidence intervals
    #[pyo3(get)]
    pub show_confidence: bool,
    /// Confidence level
    #[pyo3(get)]
    pub confidence_level: f64,
    /// Kernel for density estimation
    #[pyo3(get)]
    pub kernel: String,
    /// Number of bins for histograms
    #[pyo3(get)]
    pub bins: Option<usize>,
    /// Statistical test annotations
    #[pyo3(get)]
    pub show_stats: bool,
}

#[pymethods]
impl PlottingUtilities {
    /// Create a new plotting utilities instance
    #[new]
    pub fn new() -> PyResult<Self> {
        let mut color_schemes = HashMap::new();

        // Add predefined color schemes
        color_schemes.insert(
            "viridis".to_string(),
            vec!["#440154", "#3b528b", "#21908c", "#5dc863", "#fde725"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        color_schemes.insert(
            "plasma".to_string(),
            vec!["#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        color_schemes.insert(
            "cool".to_string(),
            vec!["#00ffff", "#80bfff", "#bf80ff", "#ff80bf", "#ff0080"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );
        color_schemes.insert(
            "warm".to_string(),
            vec!["#ffff00", "#ffbf00", "#ff8000", "#ff4000", "#ff0000"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
        );

        Ok(Self {
            numpy_compat: NumpyCompatLayer::new()?,
            plot_modules: HashMap::new(),
            config: PlotConfig::default(),
            color_schemes,
        })
    }

    /// Configure plotting settings
    pub fn configure(&mut self, py: Python, config: &Bound<'_, PyDict>) -> PyResult<()> {
        if let Some(figsize) = config.get_item("figure_size")? {
            let size_tuple: &PyTuple = figsize.downcast()?;
            self.config.figure_size = (
                size_tuple.get_item(0)?.extract()?,
                size_tuple.get_item(1)?.extract()?,
            );
        }
        if let Some(dpi) = config.get_item("dpi")? {
            self.config.dpi = dpi.extract()?;
        }
        if let Some(scheme) = config.get_item("color_scheme")? {
            self.config.color_scheme = scheme.extract()?;
        }
        if let Some(grid) = config.get_item("show_grid")? {
            self.config.show_grid = grid.extract()?;
        }
        if let Some(interactive) = config.get_item("interactive")? {
            self.config.interactive = interactive.extract()?;
        }
        Ok(())
    }

    /// Create a line plot
    pub fn line_plot(
        &self,
        py: Python,
        x: &PyTensor,
        y: &PyTensor,
        title: Option<&str>,
        xlabel: Option<&str>,
        ylabel: Option<&str>,
        style: Option<&str>,
    ) -> PyResult<PlotResult> {
        let plt = self.get_matplotlib_module(py)?;

        // Convert tensors to numpy arrays
        let x_np = self.numpy_compat.to_numpy_array(py, x)?;
        let y_np = self.numpy_compat.to_numpy_array(py, y)?;

        // Create figure and axes
        let (fig, ax) = self.create_figure(py)?;

        // Plot data
        let plot_style = style.unwrap_or("-");
        ax.call_method(py, "plot", (x_np, y_np, plot_style), None)?;

        // Set labels and title
        if let Some(title_str) = title {
            ax.call_method1(py, "set_title", (title_str,))?;
        }
        if let Some(xlabel_str) = xlabel {
            ax.call_method1(py, "set_xlabel", (xlabel_str,))?;
        }
        if let Some(ylabel_str) = ylabel {
            ax.call_method1(py, "set_ylabel", (ylabel_str,))?;
        }

        // Apply configuration
        self.apply_plot_config(py, &ax)?;

        let mut metadata = HashMap::new();
        metadata.insert("plot_type".to_string(), "line".to_string());
        metadata.insert(
            "data_points".to_string(),
            format!("{}", x.shape().iter().product::<usize>()),
        );

        Ok(PlotResult {
            figure: Some(fig),
            axes: Some(ax),
            metadata,
            exported_paths: Vec::new(),
        })
    }

    /// Create a scatter plot
    pub fn scatter_plot(
        &self,
        py: Python,
        x: &PyTensor,
        y: &PyTensor,
        colors: Option<&PyTensor>,
        sizes: Option<&PyTensor>,
        title: Option<&str>,
        xlabel: Option<&str>,
        ylabel: Option<&str>,
    ) -> PyResult<PlotResult> {
        let plt = self.get_matplotlib_module(py)?;

        let x_np = self.numpy_compat.to_numpy_array(py, x)?;
        let y_np = self.numpy_compat.to_numpy_array(py, y)?;

        let (fig, ax) = self.create_figure(py)?;

        // Prepare scatter arguments
        let mut kwargs = PyDict::new(py);
        if let Some(c) = colors {
            let c_np = self.numpy_compat.to_numpy_array(py, c)?;
            kwargs.set_item("c", c_np)?;
        }
        if let Some(s) = sizes {
            let s_np = self.numpy_compat.to_numpy_array(py, s)?;
            kwargs.set_item("s", s_np)?;
        }
        kwargs.set_item("alpha", 0.7)?;

        ax.call_method(py, "scatter", (x_np, y_np), Some(kwargs))?;

        // Set labels and title
        if let Some(title_str) = title {
            ax.call_method1(py, "set_title", (title_str,))?;
        }
        if let Some(xlabel_str) = xlabel {
            ax.call_method1(py, "set_xlabel", (xlabel_str,))?;
        }
        if let Some(ylabel_str) = ylabel {
            ax.call_method1(py, "set_ylabel", (ylabel_str,))?;
        }

        self.apply_plot_config(py, &ax)?;

        let mut metadata = HashMap::new();
        metadata.insert("plot_type".to_string(), "scatter".to_string());
        metadata.insert(
            "data_points".to_string(),
            format!("{}", x.shape().iter().product::<usize>()),
        );

        Ok(PlotResult {
            figure: Some(fig),
            axes: Some(ax),
            metadata,
            exported_paths: Vec::new(),
        })
    }

    /// Create a histogram
    pub fn histogram(
        &self,
        py: Python,
        data: &PyTensor,
        bins: Option<usize>,
        density: Option<bool>,
        title: Option<&str>,
        xlabel: Option<&str>,
        ylabel: Option<&str>,
    ) -> PyResult<PlotResult> {
        let data_np = self.numpy_compat.to_numpy_array(py, data)?;
        let (fig, ax) = self.create_figure(py)?;

        let mut kwargs = PyDict::new(py);
        kwargs.set_item("bins", bins.unwrap_or(30))?;
        kwargs.set_item("density", density.unwrap_or(false))?;
        kwargs.set_item("alpha", 0.7)?;
        kwargs.set_item("edgecolor", "black")?;

        ax.call_method(py, "hist", (data_np,), Some(kwargs))?;

        if let Some(title_str) = title {
            ax.call_method1(py, "set_title", (title_str,))?;
        }
        if let Some(xlabel_str) = xlabel {
            ax.call_method1(py, "set_xlabel", (xlabel_str,))?;
        }
        if let Some(ylabel_str) = ylabel {
            ax.call_method1(py, "set_ylabel", (ylabel_str,))?;
        }

        self.apply_plot_config(py, &ax)?;

        let mut metadata = HashMap::new();
        metadata.insert("plot_type".to_string(), "histogram".to_string());
        metadata.insert("bins".to_string(), bins.unwrap_or(30).to_string());

        Ok(PlotResult {
            figure: Some(fig),
            axes: Some(ax),
            metadata,
            exported_paths: Vec::new(),
        })
    }

    /// Create a heatmap
    pub fn heatmap(
        &self,
        py: Python,
        data: &PyTensor,
        title: Option<&str>,
        colormap: Option<&str>,
        show_colorbar: Option<bool>,
    ) -> PyResult<PlotResult> {
        let data_np = self.numpy_compat.to_numpy_array(py, data)?;
        let (fig, ax) = self.create_figure(py)?;

        let mut kwargs = PyDict::new(py);
        kwargs.set_item("cmap", colormap.unwrap_or(&self.config.color_scheme))?;
        kwargs.set_item("interpolation", "nearest")?;

        let im = ax.call_method(py, "imshow", (data_np,), Some(kwargs))?;

        if show_colorbar.unwrap_or(true) {
            fig.call_method1(py, "colorbar", (im,))?;
        }

        if let Some(title_str) = title {
            ax.call_method1(py, "set_title", (title_str,))?;
        }

        self.apply_plot_config(py, &ax)?;

        let mut metadata = HashMap::new();
        metadata.insert("plot_type".to_string(), "heatmap".to_string());
        metadata.insert(
            "colormap".to_string(),
            colormap.unwrap_or(&self.config.color_scheme).to_string(),
        );

        Ok(PlotResult {
            figure: Some(fig),
            axes: Some(ax),
            metadata,
            exported_paths: Vec::new(),
        })
    }

    /// Create a box plot
    pub fn box_plot(
        &self,
        py: Python,
        data_list: Vec<PyTensor>,
        labels: Option<Vec<String>>,
        title: Option<&str>,
        ylabel: Option<&str>,
    ) -> PyResult<PlotResult> {
        let (fig, ax) = self.create_figure(py)?;

        // Convert all tensors to numpy arrays
        let mut data_arrays = Vec::new();
        for tensor in &data_list {
            let np_array = self.numpy_compat.to_numpy_array(py, tensor)?;
            data_arrays.push(np_array);
        }

        let mut kwargs = PyDict::new(py);
        kwargs.set_item("patch_artist", true)?;

        let bp = ax.call_method(py, "boxplot", (data_arrays,), Some(kwargs))?;

        if let Some(labels_vec) = labels {
            ax.call_method1(py, "set_xticklabels", (labels_vec,))?;
        }

        if let Some(title_str) = title {
            ax.call_method1(py, "set_title", (title_str,))?;
        }
        if let Some(ylabel_str) = ylabel {
            ax.call_method1(py, "set_ylabel", (ylabel_str,))?;
        }

        self.apply_plot_config(py, &ax)?;

        let mut metadata = HashMap::new();
        metadata.insert("plot_type".to_string(), "boxplot".to_string());
        metadata.insert("num_groups".to_string(), data_list.len().to_string());

        Ok(PlotResult {
            figure: Some(fig),
            axes: Some(ax),
            metadata,
            exported_paths: Vec::new(),
        })
    }

    /// Create a 3D surface plot
    pub fn surface_plot(
        &self,
        py: Python,
        x: &PyTensor,
        y: &PyTensor,
        z: &PyTensor,
        title: Option<&str>,
        colormap: Option<&str>,
    ) -> PyResult<PlotResult> {
        let plt = self.get_matplotlib_module(py)?;
        let mpl_toolkits = py.import("mpl_toolkits.mplot3d")?;

        let x_np = self.numpy_compat.to_numpy_array(py, x)?;
        let y_np = self.numpy_compat.to_numpy_array(py, y)?;
        let z_np = self.numpy_compat.to_numpy_array(py, z)?;

        // Create 3D figure
        let fig = plt.call_method(
            py,
            "figure",
            (),
            Some([("figsize", self.config.figure_size.to_object(py))].into_py_dict(py)),
        )?;

        let ax = fig.call_method(
            py,
            "add_subplot",
            (111,),
            Some([("projection", "3d")].into_py_dict(py)),
        )?;

        let mut kwargs = PyDict::new(py);
        kwargs.set_item("cmap", colormap.unwrap_or(&self.config.color_scheme))?;
        kwargs.set_item("alpha", 0.8)?;

        ax.call_method(py, "plot_surface", (x_np, y_np, z_np), Some(kwargs))?;

        if let Some(title_str) = title {
            ax.call_method1(py, "set_title", (title_str,))?;
        }

        let mut metadata = HashMap::new();
        metadata.insert("plot_type".to_string(), "surface".to_string());
        metadata.insert("dimensions".to_string(), "3D".to_string());

        Ok(PlotResult {
            figure: Some(fig),
            axes: Some(ax),
            metadata,
            exported_paths: Vec::new(),
        })
    }

    /// Save plot to file
    pub fn save_plot(
        &self,
        py: Python,
        plot_result: &mut PlotResult,
        filename: &str,
        format: Option<&str>,
        dpi: Option<u32>,
    ) -> PyResult<()> {
        if let Some(fig) = &plot_result.figure {
            let mut kwargs = PyDict::new(py);
            kwargs.set_item("dpi", dpi.unwrap_or(self.config.dpi))?;
            kwargs.set_item("bbox_inches", "tight")?;

            if let Some(fmt) = format {
                kwargs.set_item("format", fmt)?;
            }

            fig.call_method(py, "savefig", (filename,), Some(kwargs))?;
            plot_result.exported_paths.push(filename.to_string());
        }
        Ok(())
    }

    /// Create statistical plots using Seaborn
    pub fn statistical_plot(
        &self,
        py: Python,
        data: &PyTensor,
        plot_type: &str,
        config: &StatPlotConfig,
    ) -> PyResult<PlotResult> {
        let sns = self.get_seaborn_module(py)?;
        let data_np = self.numpy_compat.to_numpy_array(py, data)?;

        let (fig, ax) = self.create_figure(py)?;

        match plot_type {
            "distplot" | "histplot" => {
                let mut kwargs = PyDict::new(py);
                if let Some(bins) = config.bins {
                    kwargs.set_item("bins", bins)?;
                }
                kwargs.set_item("kde", true)?;
                sns.call_method(py, "histplot", (data_np,), Some(kwargs))?;
            }
            "violinplot" => {
                sns.call_method1(py, "violinplot", (data_np,))?;
            }
            "boxplot" => {
                sns.call_method1(py, "boxplot", (data_np,))?;
            }
            "kde" | "kdeplot" => {
                sns.call_method1(py, "kdeplot", (data_np,))?;
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown statistical plot type: {}",
                    plot_type
                )))
            }
        }

        self.apply_plot_config(py, &ax)?;

        let mut metadata = HashMap::new();
        metadata.insert(
            "plot_type".to_string(),
            format!("statistical_{}", plot_type),
        );
        metadata.insert("library".to_string(), "seaborn".to_string());

        Ok(PlotResult {
            figure: Some(fig),
            axes: Some(ax),
            metadata,
            exported_paths: Vec::new(),
        })
    }

    /// Create interactive plots using Plotly
    pub fn interactive_plot(
        &self,
        py: Python,
        x: &PyTensor,
        y: &PyTensor,
        plot_type: &str,
        title: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let plotly = self.get_plotly_module(py)?;
        let go = plotly.getattr("graph_objects")?;

        let x_np = self.numpy_compat.to_numpy_array(py, x)?;
        let y_np = self.numpy_compat.to_numpy_array(py, y)?;

        let trace = match plot_type {
            "scatter" => go.call_method(
                py,
                "Scatter",
                (),
                Some([("x", x_np), ("y", y_np), ("mode", "markers")].into_py_dict(py)),
            )?,
            "line" => go.call_method(
                py,
                "Scatter",
                (),
                Some([("x", x_np), ("y", y_np), ("mode", "lines")].into_py_dict(py)),
            )?,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown interactive plot type: {}",
                    plot_type
                )))
            }
        };

        let mut layout_dict = PyDict::new(py);
        if let Some(title_str) = title {
            layout_dict.set_item("title", title_str)?;
        }

        let layout = go.call_method(py, "Layout", (), Some(layout_dict))?;
        let fig = go.call_method(
            py,
            "Figure",
            (),
            Some([("data", vec![trace].to_object(py)), ("layout", layout)].into_py_dict(py)),
        )?;

        Ok(fig)
    }

    /// Get available color schemes
    pub fn get_color_schemes(&self) -> Vec<String> {
        self.color_schemes.keys().cloned().collect()
    }

    /// Add custom color scheme
    pub fn add_color_scheme(&mut self, name: String, colors: Vec<String>) {
        self.color_schemes.insert(name, colors);
    }
}

impl PlottingUtilities {
    /// Get or cache matplotlib module
    fn get_matplotlib_module(&self, py: Python) -> PyResult<Py<PyModule>> {
        if let Some(module) = self.plot_modules.get("matplotlib") {
            return Ok(module.clone());
        }

        let module = py.import("matplotlib.pyplot")?;
        Ok(module.into())
    }

    /// Get or cache seaborn module
    fn get_seaborn_module(&self, py: Python) -> PyResult<Py<PyModule>> {
        if let Some(module) = self.plot_modules.get("seaborn") {
            return Ok(module.clone());
        }

        let module = py.import("seaborn")?;
        Ok(module.into())
    }

    /// Get or cache plotly module
    fn get_plotly_module(&self, py: Python) -> PyResult<Py<PyModule>> {
        if let Some(module) = self.plot_modules.get("plotly") {
            return Ok(module.clone());
        }

        let module = py.import("plotly")?;
        Ok(module.into())
    }

    /// Create figure and axes with default configuration
    fn create_figure(&self, py: Python) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let plt = self.get_matplotlib_module(py)?;

        let kwargs = [
            ("figsize", self.config.figure_size.to_object(py)),
            ("dpi", self.config.dpi.to_object(py)),
        ]
        .into_py_dict(py);

        let result = plt.call_method(py, "subplots", (), Some(kwargs))?;
        let result_tuple: &PyTuple = result.downcast()?;

        let fig = result_tuple.get_item(0)?.into();
        let ax = result_tuple.get_item(1)?.into();

        Ok((fig, ax))
    }

    /// Apply configuration to plot
    fn apply_plot_config(&self, py: Python, ax: &Py<PyAny>) -> PyResult<()> {
        if self.config.show_grid {
            ax.call_method1(py, "grid", (true,))?;
        }

        // Apply font configuration
        let font_dict = [
            ("family", self.config.font_config.family.to_object(py)),
            ("size", self.config.font_config.size.to_object(py)),
            ("weight", self.config.font_config.weight.to_object(py)),
        ]
        .into_py_dict(py);

        let matplotlib = py.import("matplotlib")?;
        matplotlib.call_method(py, "rc", ("font",), Some(font_dict))?;

        Ok(())
    }
}

/// Create plotting utilities
pub fn create_plotting_utilities(py: Python) -> PyResult<Bound<PyDict>> {
    let utils = PyDict::new_bound(py);

    // Add utility functions
    utils.set_item("create_plotter", py.get_type::<PlottingUtilities>())?;
    utils.set_item("PlotResult", py.get_type::<PlotResult>())?;
    utils.set_item("StatPlotConfig", py.get_type::<StatPlotConfig>())?;

    Ok(utils)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plot_config() {
        let config = PlotConfig::default();
        assert_eq!(config.figure_size, (10.0, 6.0));
        assert_eq!(config.dpi, 300);
        assert!(config.show_grid);
    }

    #[test]
    fn test_font_config() {
        let font_config = FontConfig {
            family: "Arial".to_string(),
            size: 14,
            weight: "bold".to_string(),
        };
        assert_eq!(font_config.family, "Arial");
        assert_eq!(font_config.size, 14);
    }
}
