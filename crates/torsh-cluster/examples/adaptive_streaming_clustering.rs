//! Comprehensive example demonstrating advanced streaming clustering features
//!
//! This example showcases:
//! 1. Adaptive epsilon selection for DBSCAN
//! 2. Sliding window clustering for non-stationary streams
//! 3. Advanced drift detection methods
//! 4. Online K-Means with concept drift adaptation

use scirs2_core::random::{thread_rng, Rng};
use torsh_cluster::{
    algorithms::incremental::{
        IncrementalClustering, OnlineKMeans, SlidingWindowConfig, SlidingWindowKMeans,
    },
    traits::Fit,
    utils::{
        adaptive::{suggest_dbscan_params, suggest_epsilon},
        drift_detection::{CompositeDriftDetector, DriftStatus, PageHinkleyTest, ADWIN},
    },
    DBSCAN,
};
use torsh_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh Cluster: Advanced Streaming Clustering Demo ===\n");

    // =====================================================================
    // Part 1: Adaptive Epsilon Selection for DBSCAN
    // =====================================================================
    println!("## Part 1: Adaptive Epsilon Selection for DBSCAN");
    println!("------------------------------------------------");

    // Generate sample data with clusters
    let sample_data = generate_clustered_data(200, 2);

    // Automatically suggest DBSCAN parameters
    let (suggested_eps, suggested_min_samples) = suggest_dbscan_params(&sample_data, "auto", None)?;

    println!("Suggested DBSCAN parameters:");
    println!("  Epsilon (eps):     {:.4}", suggested_eps);
    println!("  Min samples:       {}", suggested_min_samples);

    // Try different epsilon selection methods
    let eps_elbow = suggest_epsilon(&sample_data, 4, "elbow", None)?;
    let eps_knee = suggest_epsilon(&sample_data, 4, "knee", None)?;
    let eps_90th = suggest_epsilon(&sample_data, 4, "percentile", Some(90.0))?;

    println!("\nEpsilon comparison across methods:");
    println!("  Elbow method:      {:.4}", eps_elbow);
    println!("  Knee method:       {:.4}", eps_knee);
    println!("  90th percentile:   {:.4}", eps_90th);

    // Run DBSCAN with suggested parameters
    let dbscan = DBSCAN::new(suggested_eps, suggested_min_samples);
    let dbscan_result = dbscan.fit(&sample_data)?;

    println!("\nDBSCAN Results:");
    println!("  Clusters found:    {}", dbscan_result.n_clusters);
    println!("  Noise points:      {}", dbscan_result.noise_points.len());
    println!(
        "  Core samples:      {}",
        dbscan_result.core_sample_indices.len()
    );

    // =====================================================================
    // Part 2: Sliding Window Clustering for Non-Stationary Streams
    // =====================================================================
    println!("\n## Part 2: Sliding Window Clustering");
    println!("-----------------------------------");

    let config = SlidingWindowConfig {
        n_clusters: 3,
        window_size: 100,
        recompute_frequency: 25,
        max_iters: 10,
        tolerance: 1e-4,
        random_state: Some(42),
    };

    let mut sliding_window = SlidingWindowKMeans::new(config)?;

    println!("Simulating non-stationary data stream...");
    println!("  Window size:            {}", 100);
    println!("  Recompute frequency:    {} points", 25);
    println!("  Number of clusters:     {}", 3);

    // Simulate a stream with shifting distribution
    let mut phase_stats = vec![];

    // Phase 1: Initial distribution (0-150)
    println!("\nPhase 1: Cluster centers around (0,0), (5,5), (10,10)");
    for i in 0..150 {
        let cluster_id = i % 3;
        let x = (cluster_id * 5) as f32 + (i % 10) as f32 * 0.1;
        let y = (cluster_id * 5) as f32 + (i % 10) as f32 * 0.1;

        let point = Tensor::from_vec(vec![x, y], &[2])?;
        sliding_window.update_single(&point)?;

        if i % 50 == 49 {
            let result = sliding_window.get_current_result()?;
            phase_stats.push((
                "Phase 1",
                i + 1,
                result.window_fill,
                result.n_recomputations,
            ));
        }
    }

    // Phase 2: Shifted distribution (150-300)
    println!("Phase 2: Clusters shift to (10,10), (15,15), (20,20)");
    for i in 150..300 {
        let cluster_id = i % 3;
        let x = 10.0 + (cluster_id * 5) as f32 + (i % 10) as f32 * 0.1;
        let y = 10.0 + (cluster_id * 5) as f32 + (i % 10) as f32 * 0.1;

        let point = Tensor::from_vec(vec![x, y], &[2])?;
        sliding_window.update_single(&point)?;

        if i % 50 == 49 {
            let result = sliding_window.get_current_result()?;
            phase_stats.push((
                "Phase 2",
                i + 1,
                result.window_fill,
                result.n_recomputations,
            ));
        }
    }

    println!("\nSliding Window Statistics:");
    println!(
        "  {:^10} | {:^15} | {:^12} | {:^15}",
        "Phase", "Points Seen", "Window Fill", "Recomputations"
    );
    println!("  {}", "-".repeat(60));
    for (phase, points, fill, recomp) in phase_stats {
        println!(
            "  {:^10} | {:^15} | {:^12} | {:^15}",
            phase, points, fill, recomp
        );
    }

    let final_result = sliding_window.get_current_result()?;
    println!("\nFinal state:");
    println!("  Window contains only recent points from shifted distribution");
    println!("  Total recomputations: {}", final_result.n_recomputations);

    // =====================================================================
    // Part 3: Advanced Drift Detection
    // =====================================================================
    println!("\n## Part 3: Advanced Drift Detection");
    println!("----------------------------------");

    // Test different drift detectors
    test_page_hinkley_detector()?;
    test_adwin_detector()?;
    test_composite_detector()?;

    // =====================================================================
    // Part 4: Online K-Means with Drift Detection
    // =====================================================================
    println!("\n## Part 4: Online K-Means with Integrated Drift Detection");
    println!("-------------------------------------------------------");

    let mut online_kmeans = OnlineKMeans::new(3)?.drift_threshold(0.15).random_state(42);

    let mut drift_detector = PageHinkleyTest::new(0.005, 20.0, 0.999);

    println!("Processing stream with drift detection...");

    let mut drift_detected_at = vec![];

    // Stable phase
    for i in 0..200 {
        let cluster_id = i % 3;
        let x = (cluster_id * 3) as f32 + (i % 5) as f32 * 0.1;
        let y = (cluster_id * 3) as f32;

        let point = Tensor::from_vec(vec![x, y], &[2])?;
        online_kmeans.update_single(&point)?;

        // Monitor clustering quality
        if i > 10 && i % 10 == 0 {
            let result = online_kmeans.get_current_result()?;
            let quality = result.avg_intra_cluster_distance;

            let status = drift_detector.update(quality);
            if status == DriftStatus::Drift {
                drift_detected_at.push(("Stable phase", i));
            }
        }
    }

    // Drift phase - introduce new distribution
    for i in 200..400 {
        let cluster_id = i % 3;
        let x = 20.0 + (cluster_id * 5) as f32 + (i % 5) as f32 * 0.2;
        let y = 20.0 + (cluster_id * 5) as f32;

        let point = Tensor::from_vec(vec![x, y], &[2])?;
        online_kmeans.update_single(&point)?;

        // Monitor clustering quality
        if i % 10 == 0 {
            let result = online_kmeans.get_current_result()?;
            let quality = result.avg_intra_cluster_distance;

            let status = drift_detector.update(quality);
            if status == DriftStatus::Drift {
                drift_detected_at.push(("Drift phase", i));
                drift_detector.reset(); // Reset after detection
            }
        }
    }

    println!("\nDrift detections:");
    if drift_detected_at.is_empty() {
        println!("  No drift detected (distribution change was gradual)");
    } else {
        for (phase, point_num) in drift_detected_at {
            println!("  Detected at point {} ({})", point_num, phase);
        }
    }

    let final_online_result = online_kmeans.get_current_result()?;
    println!("\nFinal Online K-Means state:");
    println!("  Points processed:  {}", final_online_result.n_points_seen);
    println!(
        "  Learning rate:     {:.6}",
        final_online_result.current_learning_rate
    );

    // =====================================================================
    // Summary
    // =====================================================================
    println!("\n## Summary");
    println!("----------");
    println!("This example demonstrated:");
    println!("  ✓ Automatic parameter selection for DBSCAN");
    println!("  ✓ Sliding window clustering for non-stationary data");
    println!("  ✓ Multiple drift detection algorithms (Page-Hinkley, ADWIN, DDM)");
    println!("  ✓ Online K-Means with adaptive learning and drift monitoring");
    println!("\nThese features enable robust clustering on real-world streaming data!");

    Ok(())
}

/// Test Page-Hinkley drift detector
fn test_page_hinkley_detector() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n### Page-Hinkley Test");

    let mut ph = PageHinkleyTest::new(0.005, 50.0, 0.9999);
    let mut rng = thread_rng();

    // Simulate stable then drift
    let mut detections = 0;
    let mut warning_count = 0;

    // Stable phase (mean = 5.0)
    for _ in 0..100 {
        let value = 5.0 + (rng.random::<f64>() - 0.5) * 0.5;
        let status = ph.update(value);
        if status == DriftStatus::Warning {
            warning_count += 1;
        }
    }

    // Drift phase (mean = 8.0)
    for _ in 0..100 {
        let value = 8.0 + (rng.random::<f64>() - 0.5) * 0.5;
        let status = ph.update(value);
        if status == DriftStatus::Drift {
            detections += 1;
        } else if status == DriftStatus::Warning {
            warning_count += 1;
        }
    }

    println!("  Drift detections:  {}", detections);
    println!("  Warnings:          {}", warning_count);
    println!("  PH statistic:      {:.4}", ph.get_statistic());

    Ok(())
}

/// Test ADWIN drift detector
fn test_adwin_detector() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n### ADWIN (Adaptive Windowing)");

    let mut adwin = ADWIN::new(0.002, 1000);
    let mut rng = thread_rng();

    // Stable phase
    for _ in 0..150 {
        let value = 10.0 + (rng.random::<f64>() - 0.5) * 2.0;
        let _ = adwin.update(value);
    }

    let window_before_drift = adwin.window_size();

    // Drift phase
    for _ in 0..150 {
        let value = 20.0 + (rng.random::<f64>() - 0.5) * 2.0;
        let _ = adwin.update(value);
    }

    println!("  Window before drift: {}", window_before_drift);
    println!("  Window after drift:  {}", adwin.window_size());
    println!("  Drift detections:    {}", adwin.n_detections());

    Ok(())
}

/// Test composite drift detector
fn test_composite_detector() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n### Composite Detector (Majority Voting)");

    let mut detector = CompositeDriftDetector::new();
    let mut rng = thread_rng();

    let mut drift_count = 0;
    let mut warning_count = 0;

    // Stable phase
    for _ in 0..100 {
        let value = 15.0 + (rng.random::<f64>() - 0.5);
        let error = if rng.random::<f64>() < 0.1 { 1.0 } else { 0.0 };
        let _ = detector.update(value, Some(error));
    }

    // Drift phase
    for _ in 0..100 {
        let value = 25.0 + (rng.random::<f64>() - 0.5);
        let error = if rng.random::<f64>() < 0.5 { 1.0 } else { 0.0 };
        let status = detector.update(value, Some(error));

        match status {
            DriftStatus::Drift => drift_count += 1,
            DriftStatus::Warning => warning_count += 1,
            DriftStatus::Stable => {}
        }
    }

    println!("  Drift detections:  {}", drift_count);
    println!("  Warnings:          {}", warning_count);

    let (ph_stat, adwin_size, ddm_error) = detector.get_individual_statuses();
    println!("  Individual detector states:");
    println!("    PH statistic:    {:.4}", ph_stat);
    println!("    ADWIN window:    {}", adwin_size);
    println!("    DDM error rate:  {:.4}", ddm_error);

    Ok(())
}

/// Generate sample clustered data
fn generate_clustered_data(n_samples: usize, n_features: usize) -> Tensor {
    let mut data = vec![];
    let mut rng = thread_rng();

    let n_clusters = 3;
    let cluster_centers = vec![(0.0, 0.0), (5.0, 5.0), (10.0, 10.0)];

    for i in 0..n_samples {
        let cluster_id = i % n_clusters;
        let (cx, cy) = cluster_centers[cluster_id];

        // Add some noise
        let x = cx + (rng.random::<f32>() - 0.5) * 1.0;
        let y = cy + (rng.random::<f32>() - 0.5) * 1.0;

        data.push(x);
        if n_features > 1 {
            data.push(y);
        }

        // Add more features if needed
        for _ in 2..n_features {
            data.push((rng.random::<f32>() - 0.5) * 0.5);
        }
    }

    Tensor::from_vec(data, &[n_samples, n_features]).unwrap()
}
