#!/usr/bin/env python3
"""
Comprehensive examples demonstrating ToRSh integration with SciPy, Pandas, and Plotting utilities.

This script showcases:
1. SciPy integration for scientific computing
2. Pandas support for data manipulation
3. Plotting utilities for visualization
4. Jupyter widgets for interactive exploration
"""

import numpy as np
import torch  # For comparison/conversion
import torsh

def main():
    print("=== ToRSh Integration Examples ===\n")
    
    # Create sample data
    print("1. Creating sample data...")
    data = torsh.randn([100, 3])  # 100 samples, 3 features
    labels = torsh.randint(0, 2, [100])  # Binary classification labels
    
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print()
    
    # === SciPy Integration Examples ===
    print("2. SciPy Integration Examples")
    print("-" * 40)
    
    # Create SciPy integration instance
    scipy_integration = torsh.SciPyIntegration()
    
    # Example 1: Linear algebra operations
    print("2.1 Linear Algebra Operations")
    matrix = torsh.randn([5, 5])
    
    # Eigenvalue decomposition
    eigenresult = scipy_integration.eigendecomposition(matrix, compute_eigenvectors=True)
    print(f"Eigenvalues shape: {eigenresult.result.shape}")
    if eigenresult.secondary:
        print(f"Eigenvectors shape: {eigenresult.secondary.shape}")
    
    # SVD decomposition
    u, s, vt = scipy_integration.svd(matrix, full_matrices=False)
    print(f"SVD - U: {u.shape}, S: {s.shape}, VT: {vt.shape}")
    
    # Example 2: Optimization
    print("\n2.2 Optimization Example")
    initial_guess = torsh.zeros([2])
    
    # Define objective function (Python function for SciPy)
    def objective(x):
        return (x[0] - 1)**2 + (x[1] - 2)**2
    
    # Note: In practice, you'd pass a Python function to SciPy
    # opt_result = scipy_integration.minimize(objective, initial_guess)
    print("Optimization would minimize objective function")
    
    # Example 3: Signal processing
    print("\n2.3 Signal Processing")
    signal = torsh.randn([1000])  # Random signal
    
    # Apply digital filter
    filtered_result = scipy_integration.filter_signal(
        signal, "lowpass", cutoff=0.1, sample_rate=1.0
    )
    print(f"Filtered signal shape: {filtered_result.signal.shape}")
    
    # FFT
    fft_result = scipy_integration.fft(signal)
    print(f"FFT result shape: {fft_result.shape}")
    
    # Example 4: Statistical tests
    print("\n2.4 Statistical Tests")
    sample1 = torsh.randn([50])
    sample2 = torsh.randn([50]) + 0.5  # Shifted mean
    
    # Independent t-test
    statistic, p_value = scipy_integration.statistical_test(
        sample1, sample2, "ttest_ind"
    )
    print(f"T-test: statistic={statistic:.4f}, p-value={p_value:.4f}")
    
    print()
    
    # === Pandas Support Examples ===
    print("3. Pandas Support Examples")
    print("-" * 40)
    
    # Create Pandas support instance
    pandas_support = torsh.PandasSupport()
    
    # Example 1: Convert tensor to DataFrame
    print("3.1 Tensor to DataFrame Conversion")
    df_tensor = torsh.randn([20, 4])
    columns = ["feature_1", "feature_2", "feature_3", "target"]
    
    # Convert to pandas DataFrame
    dataframe = pandas_support.to_dataframe(
        df_tensor, columns=columns
    )
    print("Created DataFrame from tensor")
    
    # Convert back to ToRSh
    torsh_df = pandas_support.from_dataframe(dataframe)
    print(f"Converted back - shape: {torsh_df.shape}, columns: {torsh_df.columns}")
    
    # Example 2: Data analysis operations
    print("\n3.2 Data Analysis Operations")
    
    # Statistical analysis
    stats_result = pandas_support.statistical_analysis(dataframe)
    print(f"Statistical analysis completed, correlation matrix shape: {stats_result.data.shape}")
    print(f"Statistics: {list(stats_result.statistics.keys())}")
    
    # Group-by analysis
    # First create a categorical column
    category_data = torsh.randint(0, 3, [20])  # 3 categories
    category_df = pandas_support.to_series(category_data, name="category")
    
    print("Group-by analysis would be performed with categorical data")
    
    # Example 3: Time series analysis
    print("\n3.3 Time Series Analysis")
    time_series = torsh.randn([100])
    ts_series = pandas_support.to_series(time_series, name="values")
    
    # Time series analysis
    ts_result = pandas_support.time_series_analysis(
        ts_series, freq="D", window=7
    )
    print(f"Time series analysis - rolling mean shape: {ts_result.data.shape}")
    
    # Example 4: Data import/export
    print("\n3.4 Data Import/Export")
    
    # Export to different formats
    print("Data can be exported to: CSV, JSON, Excel, Parquet, HDF5")
    # pandas_support.export_dataframe(dataframe, "csv", "sample_data.csv")
    
    print()
    
    # === Plotting Utilities Examples ===
    print("4. Plotting Utilities Examples")
    print("-" * 40)
    
    # Create plotting utilities instance
    plotter = torsh.PlottingUtilities()
    
    # Example 1: Line plot
    print("4.1 Line Plot")
    x_data = torsh.linspace(0, 10, 100)
    y_data = torsh.sin(x_data)
    
    line_plot = plotter.line_plot(
        x_data, y_data, 
        title="Sine Wave",
        xlabel="x", ylabel="sin(x)"
    )
    print(f"Line plot created - type: {line_plot.metadata['plot_type']}")
    
    # Save plot
    # plotter.save_plot(line_plot, "sine_wave.png")
    
    # Example 2: Scatter plot
    print("\n4.2 Scatter Plot")
    x_scatter = torsh.randn([100])
    y_scatter = x_scatter * 2 + torsh.randn([100]) * 0.5  # Linear relationship with noise
    
    scatter_plot = plotter.scatter_plot(
        x_scatter, y_scatter,
        title="Scatter Plot Example",
        xlabel="X values", ylabel="Y values"
    )
    print(f"Scatter plot created - {scatter_plot.metadata['data_points']} points")
    
    # Example 3: Histogram
    print("\n4.3 Histogram")
    hist_data = torsh.randn([1000])
    
    histogram = plotter.histogram(
        hist_data, bins=30, density=False,
        title="Normal Distribution",
        xlabel="Value", ylabel="Frequency"
    )
    print(f"Histogram created - {histogram.metadata['bins']} bins")
    
    # Example 4: Heatmap
    print("\n4.4 Heatmap")
    heatmap_data = torsh.randn([10, 10])
    
    heatmap = plotter.heatmap(
        heatmap_data,
        title="Random Data Heatmap",
        colormap="viridis"
    )
    print(f"Heatmap created - colormap: {heatmap.metadata['colormap']}")
    
    # Example 5: Box plot
    print("\n4.5 Box Plot")
    box_data = [
        torsh.randn([50]),
        torsh.randn([50]) + 1,
        torsh.randn([50]) + 2
    ]
    
    box_plot = plotter.box_plot(
        box_data,
        labels=["Group 1", "Group 2", "Group 3"],
        title="Group Comparison",
        ylabel="Values"
    )
    print(f"Box plot created - {box_plot.metadata['num_groups']} groups")
    
    # Example 6: Statistical plots
    print("\n4.6 Statistical Plots")
    stat_config = torsh.StatPlotConfig(
        show_confidence=True,
        confidence_level=0.95,
        bins=25,
        show_stats=True
    )
    
    stat_plot = plotter.statistical_plot(
        hist_data, "histplot", stat_config
    )
    print(f"Statistical plot created - library: {stat_plot.metadata['library']}")
    
    # Example 7: Interactive plots (would require Plotly)
    print("\n4.7 Interactive Plots")
    print("Interactive plots can be created with Plotly integration")
    # interactive_plot = plotter.interactive_plot(x_scatter, y_scatter, "scatter", title="Interactive Scatter")
    
    print()
    
    # === Jupyter Widgets Examples ===
    print("5. Jupyter Widgets Examples")
    print("-" * 40)
    
    # Create Jupyter widgets instance
    jupyter_widgets = torsh.JupyterWidgets()
    
    # Example 1: Tensor visualization widget
    print("5.1 Tensor Visualization Widget")
    viz_tensor = torsh.randn([50, 2])  # 2D data for scatter plot
    
    tensor_widget = jupyter_widgets.create_tensor_widget(
        viz_tensor, "scatter", title="Tensor Visualization"
    )
    print(f"Created tensor widget - ID: {tensor_widget.widget_id}")
    print(f"Visualization type: {tensor_widget.viz_type}")
    
    # Example 2: Training monitor widget
    print("\n5.2 Training Monitor Widget")
    metrics = ["loss", "accuracy", "val_loss", "val_accuracy"]
    
    training_monitor = jupyter_widgets.create_training_monitor(
        metrics, title="Training Progress"
    )
    print(f"Created training monitor - ID: {training_monitor.widget_id}")
    print(f"Monitoring metrics: {list(training_monitor.metrics_history.keys())}")
    
    # Simulate training updates
    for epoch in range(5):
        epoch_metrics = {
            "loss": 1.0 - epoch * 0.1,
            "accuracy": 0.5 + epoch * 0.1,
            "val_loss": 1.1 - epoch * 0.08,
            "val_accuracy": 0.45 + epoch * 0.09
        }
        
        # jupyter_widgets.update_training_monitor(training_monitor, epoch, epoch_metrics)
        print(f"Updated training monitor for epoch {epoch}")
    
    # Example 3: Data exploration widget
    print("\n5.3 Data Exploration Widget")
    exploration_data = torsh.randn([100, 5])  # 5 features
    
    data_explorer = jupyter_widgets.create_data_explorer(
        exploration_data,
        feature_names=["feat_1", "feat_2", "feat_3", "feat_4", "feat_5"]
    )
    print(f"Created data explorer - ID: {data_explorer.widget_id}")
    print(f"Dataset shape: {data_explorer.dataset.shape}")
    print(f"Selected features: {data_explorer.selected_features}")
    
    # Update exploration
    # updated_plot = jupyter_widgets.update_data_explorer(
    #     data_explorer, selected_features=[0, 1], viz_type="scatter"
    # )
    
    # Example 4: Parameter tuning widget
    print("\n5.4 Parameter Tuning Widget")
    parameters = {
        "learning_rate": 0.01,
        "batch_size": 32,
        "hidden_size": 128,
        "dropout": 0.5
    }
    
    def parameter_callback(**kwargs):
        """Callback function for parameter updates"""
        print(f"Parameters updated: {kwargs}")
        # In practice, this would retrain/update your model
    
    print("Parameter tuning widget would create interactive sliders")
    print(f"Parameters to tune: {list(parameters.keys())}")
    
    # Available themes
    print(f"\n5.5 Available Widget Themes")
    themes = jupyter_widgets.get_available_themes()
    print(f"Available themes: {themes}")
    
    print()
    
    # === Performance Comparison Example ===
    print("6. Performance Comparison")
    print("-" * 40)
    
    # Benchmark operations
    print("6.1 SciPy Operations Benchmark")
    benchmark_tensor = torsh.randn([100, 100])
    
    # scipy_benchmarks = scipy_integration.benchmark_operations(
    #     tensor_size=[100, 100], num_iterations=10
    # )
    # print(f"Benchmark results: {scipy_benchmarks}")
    
    print("Benchmarking would measure performance of various operations")
    
    print()
    
    # === Integration with External Libraries ===
    print("7. External Library Integration")
    print("-" * 40)
    
    # NumPy conversion
    print("7.1 NumPy Integration")
    torsh_tensor = torsh.randn([5, 5])
    numpy_array = torsh.to_numpy(torsh_tensor)
    print(f"Converted to NumPy - shape: {numpy_array.shape}, dtype: {numpy_array.dtype}")
    
    back_to_torsh = torsh.from_numpy(numpy_array)
    print(f"Converted back to ToRSh - shape: {back_to_torsh.shape}")
    
    # PyTorch conversion (if available)
    print("\n7.2 PyTorch Integration")
    try:
        # This would require PyTorch tensor interop implementation
        print("PyTorch conversion would be available with tensor interop")
        print("torsh_tensor -> torch_tensor -> torsh_tensor")
    except Exception as e:
        print("PyTorch integration not yet implemented")
    
    print()
    print("=== Integration Examples Complete ===")

if __name__ == "__main__":
    main()