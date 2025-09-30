//! Customer Segmentation Example using Gaussian Mixture Models
//!
//! This example demonstrates how to use Gaussian Mixture Models (GMM) for customer
//! segmentation, a common business analytics task where we want to identify distinct
//! customer groups based on their behavior and characteristics.

use scirs2_core::random::prelude::*;
use std::collections::HashMap;
use torsh_cluster::{
    algorithms::{gaussian_mixture::CovarianceType, GaussianMixture},
    evaluation::metrics::{gap_statistic::*, *},
    traits::*,
};
use torsh_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("👥 Customer Segmentation with Gaussian Mixture Models");
    println!("====================================================");

    // Create synthetic customer data
    let (customer_data, customer_info) = create_customer_dataset()?;

    println!("📊 Customer Dataset Information:");
    println!("  • Total customers: {}", customer_info.total_customers);
    println!("  • Features: {:?}", customer_info.feature_names);
    println!("  • Data shape: {:?}", customer_data.shape().dims());

    // Find optimal number of customer segments
    let optimal_segments = find_optimal_segments(&customer_data)?;

    // Perform customer segmentation with different covariance types
    segment_customers(&customer_data, &customer_info, optimal_segments)?;

    // Analyze customer segments in detail
    analyze_customer_segments(&customer_data, &customer_info, optimal_segments)?;

    println!("\n✅ Customer segmentation analysis completed!");
    println!("💼 Business Applications:");
    println!("   • Targeted marketing campaigns for each segment");
    println!("   • Personalized product recommendations");
    println!("   • Pricing strategy optimization");
    println!("   • Customer lifetime value prediction");
    println!("   • Churn risk assessment by segment");

    Ok(())
}

/// Find the optimal number of customer segments using Gap Statistic
fn find_optimal_segments(data: &Tensor) -> Result<usize, Box<dyn std::error::Error>> {
    println!("\n🔍 Finding Optimal Number of Customer Segments");
    println!("==============================================");

    let config = GapStatisticConfig {
        max_k: 8,
        n_refs: 15,
        random_state: Some(42),
        ..Default::default()
    };

    let mut gap_stat = GapStatistic::new(config);
    let result = gap_stat.compute(data)?;

    println!("📊 Gap Statistic Analysis:");
    for (i, &gap) in result.gap_values.iter().enumerate() {
        let k = i + 1;
        let wk = result.wk_values[i];
        let marker = if k == result.optimal_k {
            " ← Optimal"
        } else {
            ""
        };
        println!("  k={}: Gap={:.4}, Wk={:.2}{}", k, gap, wk, marker);
    }

    println!("🎯 Recommended number of segments: {}", result.optimal_k);

    Ok(result.optimal_k)
}

/// Perform customer segmentation using different GMM configurations
fn segment_customers(
    data: &Tensor,
    info: &CustomerDatasetInfo,
    n_segments: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 Customer Segmentation Analysis");
    println!("================================");

    let covariance_types = [
        ("Full Covariance", CovarianceType::Full),
        ("Diagonal Covariance", CovarianceType::Diag),
        ("Spherical Covariance", CovarianceType::Spherical),
    ];

    let mut best_model = None;
    let mut best_bic = f64::INFINITY;

    for (name, cov_type) in &covariance_types {
        println!("\n📈 Testing {} Model", name);

        let gmm = GaussianMixture::new(n_segments)
            .covariance_type(*cov_type)
            .max_iters(200)
            .tolerance(1e-6)
            .reg_covar(1e-6)
            .random_state(42);

        let start = std::time::Instant::now();
        let result = gmm.fit(data)?;
        let duration = start.elapsed();

        // Evaluate model quality
        let silhouette = silhouette_score(data, &result.labels)?;
        let ch_score = calinski_harabasz_score(data, &result.labels)?;
        let db_score = davies_bouldin_score(data, &result.labels)?;

        println!("  ✓ Log-likelihood: {:.4}", result.log_likelihood);
        println!("  ✓ AIC: {:.4}", result.aic);
        println!("  ✓ BIC: {:.4}", result.bic);
        println!(
            "  ✓ Converged: {} ({} iterations)",
            result.converged, result.n_iter
        );
        println!("  ✓ Silhouette Score: {:.4}", silhouette);
        println!("  ✓ Calinski-Harabasz Score: {:.4}", ch_score);
        println!("  ✓ Davies-Bouldin Score: {:.4}", db_score);
        println!("  ⏱️  Training Time: {:?}", duration);

        // Track best model by BIC
        if result.bic < best_bic {
            best_bic = result.bic;
            best_model = Some((name, result, cov_type));
        }
    }

    if let Some((best_name, best_result, _)) = best_model {
        println!("\n🏆 Best Model: {} (BIC: {:.4})", best_name, best_bic);
        analyze_segment_characteristics(data, info, &best_result.labels)?;
    }

    Ok(())
}

/// Analyze characteristics of each customer segment
fn analyze_segment_characteristics(
    data: &Tensor,
    info: &CustomerDatasetInfo,
    labels: &Tensor,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n📊 Customer Segment Characteristics");
    println!("==================================");

    let data_vec = data.to_vec()?;
    let labels_vec = labels.to_vec()?;
    let shape = data.shape();
    let data_shape = shape.dims();
    let n_samples = data_shape[0];
    let n_features = data_shape[1];

    // Group customers by segment
    let mut segments: HashMap<i32, Vec<usize>> = HashMap::new();
    for (idx, &label) in labels_vec.iter().enumerate() {
        segments
            .entry(label as i32)
            .or_insert_with(Vec::new)
            .push(idx);
    }

    for (&segment_id, customer_indices) in &segments {
        println!(
            "\n🎯 Segment {} ({} customers, {:.1}% of total)",
            segment_id,
            customer_indices.len(),
            customer_indices.len() as f64 / n_samples as f64 * 100.0
        );

        // Calculate mean values for each feature in this segment
        let mut feature_means = vec![0.0; n_features];
        for &customer_idx in customer_indices {
            for feature_idx in 0..n_features {
                feature_means[feature_idx] +=
                    data_vec[customer_idx * n_features + feature_idx] as f64;
            }
        }

        for feature_mean in &mut feature_means {
            *feature_mean /= customer_indices.len() as f64;
        }

        // Display feature characteristics
        for (feature_idx, &mean_value) in feature_means.iter().enumerate() {
            let feature_name = &info.feature_names[feature_idx];
            let interpretation = interpret_feature_value(feature_name, mean_value);
            println!(
                "  • {}: {:.2} ({})",
                feature_name, mean_value, interpretation
            );
        }

        // Generate segment profile
        let profile = generate_segment_profile(segment_id, &feature_means, &info.feature_names);
        println!("  📋 Profile: {}", profile);
    }

    Ok(())
}

/// Analyze customer segments in business context
fn analyze_customer_segments(
    data: &Tensor,
    info: &CustomerDatasetInfo,
    n_segments: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n💼 Business Impact Analysis");
    println!("===========================");

    // Use the best GMM model for business analysis
    let gmm = GaussianMixture::new(n_segments)
        .covariance_type(CovarianceType::Full)
        .max_iters(200)
        .random_state(42);

    let result = gmm.fit(data)?;
    let labels_vec = result.labels.to_vec()?;

    // Calculate segment business metrics
    let mut segments: HashMap<i32, Vec<usize>> = HashMap::new();
    for (idx, &label) in labels_vec.iter().enumerate() {
        segments
            .entry(label as i32)
            .or_insert_with(Vec::new)
            .push(idx);
    }

    for (&segment_id, customer_indices) in &segments {
        let segment_size = customer_indices.len();
        let segment_percentage = segment_size as f64 / labels_vec.len() as f64 * 100.0;

        println!("\n🎯 Segment {} Business Analysis", segment_id);
        println!(
            "  • Size: {} customers ({:.1}%)",
            segment_size, segment_percentage
        );

        // Simulate business metrics (in practice, these would be calculated from real data)
        let avg_clv = simulate_customer_lifetime_value(segment_id);
        let churn_risk = simulate_churn_risk(segment_id);
        let engagement_score = simulate_engagement_score(segment_id);
        let revenue_potential = simulate_revenue_potential(segment_id, segment_size);

        println!("  • Average Customer Lifetime Value: ${:.0}", avg_clv);
        println!("  • Churn Risk: {:.1}%", churn_risk * 100.0);
        println!("  • Engagement Score: {:.2}/5.0", engagement_score);
        println!(
            "  • Total Revenue Potential: ${:.0}K",
            revenue_potential / 1000.0
        );

        // Marketing recommendations
        let recommendations =
            generate_marketing_recommendations(segment_id, avg_clv, churn_risk, engagement_score);
        println!("  📢 Marketing Recommendations:");
        for rec in recommendations {
            println!("     • {}", rec);
        }
    }

    Ok(())
}

/// Create synthetic customer dataset
fn create_customer_dataset() -> Result<(Tensor, CustomerDatasetInfo), Box<dyn std::error::Error>> {
    let n_customers = 1000;
    let feature_names = vec![
        "Age".to_string(),
        "Annual Income ($k)".to_string(),
        "Spending Score (1-100)".to_string(),
        "Years as Customer".to_string(),
        "Monthly Purchases".to_string(),
        "Support Tickets".to_string(),
    ];

    let mut data = Vec::with_capacity(n_customers * feature_names.len());

    // Generate customers with different profiles
    for i in 0..n_customers {
        let customer_type = i % 4; // 4 different customer types

        let (age, income, spending, years, purchases, tickets) = match customer_type {
            0 => {
                // Young professionals
                let age = 25.0 + thread_rng().gen::<f32>() * 10.0;
                let income = 45.0 + thread_rng().gen::<f32>() * 20.0;
                let spending = 60.0 + thread_rng().gen::<f32>() * 30.0;
                let years = 1.0 + thread_rng().gen::<f32>() * 3.0;
                let purchases = 5.0 + thread_rng().gen::<f32>() * 10.0;
                let tickets = thread_rng().gen::<f32>() * 2.0;
                (age, income, spending, years, purchases, tickets)
            }
            1 => {
                // Established families
                let age = 35.0 + thread_rng().gen::<f32>() * 15.0;
                let income = 60.0 + thread_rng().gen::<f32>() * 40.0;
                let spending = 40.0 + thread_rng().gen::<f32>() * 40.0;
                let years = 3.0 + thread_rng().gen::<f32>() * 7.0;
                let purchases = 8.0 + thread_rng().gen::<f32>() * 12.0;
                let tickets = 1.0 + thread_rng().gen::<f32>() * 3.0;
                (age, income, spending, years, purchases, tickets)
            }
            2 => {
                // High-value customers
                let age = 40.0 + thread_rng().gen::<f32>() * 20.0;
                let income = 80.0 + thread_rng().gen::<f32>() * 50.0;
                let spending = 70.0 + thread_rng().gen::<f32>() * 25.0;
                let years = 5.0 + thread_rng().gen::<f32>() * 10.0;
                let purchases = 15.0 + thread_rng().gen::<f32>() * 15.0;
                let tickets = 0.5 + thread_rng().gen::<f32>() * 2.0;
                (age, income, spending, years, purchases, tickets)
            }
            _ => {
                // Budget-conscious customers
                let age = 30.0 + thread_rng().gen::<f32>() * 25.0;
                let income = 30.0 + thread_rng().gen::<f32>() * 25.0;
                let spending = 20.0 + thread_rng().gen::<f32>() * 30.0;
                let years = 2.0 + thread_rng().gen::<f32>() * 8.0;
                let purchases = 2.0 + thread_rng().gen::<f32>() * 6.0;
                let tickets = 2.0 + thread_rng().gen::<f32>() * 4.0;
                (age, income, spending, years, purchases, tickets)
            }
        };

        data.extend_from_slice(&[age, income, spending, years, purchases, tickets]);
    }

    let data_tensor = Tensor::from_vec(data, &[n_customers, feature_names.len()])?;
    let info = CustomerDatasetInfo {
        total_customers: n_customers,
        feature_names,
    };

    Ok((data_tensor, info))
}

/// Interpret feature values in business context
fn interpret_feature_value(feature_name: &str, value: f64) -> &'static str {
    match feature_name {
        "Age" => {
            if value < 30.0 {
                "Young"
            } else if value < 50.0 {
                "Middle-aged"
            } else {
                "Mature"
            }
        }
        "Annual Income ($k)" => {
            if value < 40.0 {
                "Lower income"
            } else if value < 80.0 {
                "Middle income"
            } else {
                "High income"
            }
        }
        "Spending Score (1-100)" => {
            if value < 40.0 {
                "Low spender"
            } else if value < 70.0 {
                "Moderate spender"
            } else {
                "High spender"
            }
        }
        "Years as Customer" => {
            if value < 3.0 {
                "New customer"
            } else if value < 7.0 {
                "Regular customer"
            } else {
                "Loyal customer"
            }
        }
        "Monthly Purchases" => {
            if value < 5.0 {
                "Infrequent buyer"
            } else if value < 15.0 {
                "Regular buyer"
            } else {
                "Frequent buyer"
            }
        }
        "Support Tickets" => {
            if value < 1.0 {
                "Low maintenance"
            } else if value < 3.0 {
                "Moderate support"
            } else {
                "High maintenance"
            }
        }
        _ => "Unknown",
    }
}

/// Generate a descriptive profile for a customer segment
fn generate_segment_profile(
    segment_id: i32,
    feature_means: &[f64],
    feature_names: &[String],
) -> String {
    let age_interp = interpret_feature_value(&feature_names[0], feature_means[0]);
    let income_interp = interpret_feature_value(&feature_names[1], feature_means[1]);
    let spending_interp = interpret_feature_value(&feature_names[2], feature_means[2]);
    let loyalty_interp = interpret_feature_value(&feature_names[3], feature_means[3]);

    format!(
        "{} {} {} {}",
        age_interp, income_interp, spending_interp, loyalty_interp
    )
}

/// Simulate customer lifetime value
fn simulate_customer_lifetime_value(segment_id: i32) -> f64 {
    match segment_id {
        0 => 5000.0 + thread_rng().gen::<f64>() * 3000.0, // Young professionals
        1 => 8000.0 + thread_rng().gen::<f64>() * 4000.0, // Established families
        2 => 15000.0 + thread_rng().gen::<f64>() * 8000.0, // High-value
        _ => 3000.0 + thread_rng().gen::<f64>() * 2000.0, // Budget-conscious
    }
}

/// Simulate churn risk (0.0 to 1.0)
fn simulate_churn_risk(segment_id: i32) -> f64 {
    match segment_id {
        0 => 0.15 + thread_rng().gen::<f64>() * 0.1, // Young professionals - moderate churn
        1 => 0.05 + thread_rng().gen::<f64>() * 0.05, // Established families - low churn
        2 => 0.03 + thread_rng().gen::<f64>() * 0.03, // High-value - very low churn
        _ => 0.25 + thread_rng().gen::<f64>() * 0.15, // Budget-conscious - high churn
    }
}

/// Simulate engagement score (1.0 to 5.0)
fn simulate_engagement_score(segment_id: i32) -> f64 {
    match segment_id {
        0 => 3.5 + thread_rng().gen::<f64>() * 1.0, // Young professionals - good engagement
        1 => 3.0 + thread_rng().gen::<f64>() * 1.0, // Established families - moderate engagement
        2 => 4.0 + thread_rng().gen::<f64>() * 0.8, // High-value - high engagement
        _ => 2.5 + thread_rng().gen::<f64>() * 1.0, // Budget-conscious - lower engagement
    }
}

/// Simulate total revenue potential for segment
fn simulate_revenue_potential(segment_id: i32, segment_size: usize) -> f64 {
    let base_revenue_per_customer = match segment_id {
        0 => 2000.0, // Young professionals
        1 => 3000.0, // Established families
        2 => 5000.0, // High-value
        _ => 1000.0, // Budget-conscious
    };

    base_revenue_per_customer * segment_size as f64 * (0.8 + thread_rng().gen::<f64>() * 0.4)
}

/// Generate marketing recommendations for segment
fn generate_marketing_recommendations(
    segment_id: i32,
    clv: f64,
    churn_risk: f64,
    engagement: f64,
) -> Vec<String> {
    let mut recommendations = Vec::new();

    // CLV-based recommendations
    if clv > 10000.0 {
        recommendations.push("VIP program enrollment".to_string());
        recommendations.push("Personal account manager assignment".to_string());
    } else if clv > 5000.0 {
        recommendations.push("Premium tier upgrade offers".to_string());
    } else {
        recommendations.push("Value-focused promotions".to_string());
    }

    // Churn risk recommendations
    if churn_risk > 0.2 {
        recommendations.push("Retention campaign priority".to_string());
        recommendations.push("Customer satisfaction survey".to_string());
    } else if churn_risk > 0.1 {
        recommendations.push("Loyalty program enrollment".to_string());
    }

    // Engagement-based recommendations
    if engagement > 4.0 {
        recommendations.push("Brand ambassador program".to_string());
        recommendations.push("Beta testing opportunities".to_string());
    } else if engagement < 3.0 {
        recommendations.push("Re-engagement campaign".to_string());
        recommendations.push("Product education content".to_string());
    }

    recommendations
}

/// Information about the customer dataset
#[derive(Debug)]
struct CustomerDatasetInfo {
    total_customers: usize,
    feature_names: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_customer_dataset_creation() -> Result<(), Box<dyn std::error::Error>> {
        let (data, info) = create_customer_dataset()?;

        assert_eq!(data.shape().dims()[0], info.total_customers);
        assert_eq!(data.shape().dims()[1], info.feature_names.len());

        Ok(())
    }

    #[test]
    fn test_customer_segmentation() -> Result<(), Box<dyn std::error::Error>> {
        let (data, _) = create_customer_dataset()?;

        let gmm = GaussianMixture::new(4)
            .covariance_type(CovarianceType::Diag)
            .max_iters(50)
            .random_state(42);

        let result = gmm.fit(&data)?;

        // Check that segmentation produces valid results
        assert!(result.converged || result.n_iter == 50);
        assert!(result.log_likelihood.is_finite());
        assert!(result.aic > 0.0);
        assert!(result.bic > 0.0);

        Ok(())
    }
}
