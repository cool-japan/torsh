//! Cloud Providers Integration Demo
//!
//! This example demonstrates cloud provider integrations for profiling on AWS, Azure, and GCP,
//! including SageMaker, Azure ML, and Vertex AI support.

use torsh_profiler::cloud_providers::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh Profiler: Cloud Providers Integration Demo ===\n");

    // ========================================
    // Part 1: Cloud Provider Detection
    // ========================================
    println!("1. Cloud Provider Detection");
    println!("   Automatically detecting cloud environment\n");

    let provider = CloudProvider::detect();
    println!("   Detected cloud provider: {}", provider);

    let metadata = CloudInstanceMetadata::detect()?;
    println!("\n   Instance Metadata:");
    println!("      Provider: {}", metadata.provider);
    println!("      Instance ID: {}", metadata.instance_id);
    println!("      Instance Type: {}", metadata.instance_type);
    println!("      Region: {}", metadata.region);
    println!("      CPU Count: {}", metadata.cpu_count);
    println!("      GPU Count: {}", metadata.gpu_count);
    println!("      Spot Instance: {}", metadata.is_spot);

    // ========================================
    // Part 2: Multi-Cloud Profiler
    // ========================================
    println!("\n2. Multi-Cloud Profiler");
    println!("   Creating cloud-agnostic profiler\n");

    let profiler = MultiCloudProfiler::new()?;
    println!("   Running on: {}", profiler.provider());
    println!(
        "   Instance: {} ({})",
        profiler.metadata().instance_id,
        profiler.metadata().instance_type
    );

    println!("\n   Cloud-Specific Recommendations:");
    println!(
        "      Export destination: {}",
        profiler.recommended_export_destination()
    );
    println!(
        "      Estimated cost: ${:.2}/hour",
        profiler.estimated_cost_per_hour()
    );

    let tags = profiler.recommended_tags();
    println!("\n   Recommended tags:");
    for (key, value) in tags {
        println!("      {}: {}", key, value);
    }

    // ========================================
    // Part 3: AWS Integration
    // ========================================
    println!("\n3. AWS Integration (SageMaker, ECS, EKS)");
    println!("   Configuring profiling for AWS services\n");

    let mut aws_profiler = aws::AWSProfiler::default();

    // Configure for SageMaker training job
    let sagemaker_config = aws::SageMakerProfilingConfig {
        job_name: "pytorch-distributed-training".to_string(),
        s3_bucket: "my-sagemaker-bucket".to_string(),
        s3_prefix: "profiling/2024-10-24".to_string(),
        profiling_interval_seconds: 60,
        detailed_profiling: true,
        framework: "pytorch".to_string(),
    };

    aws_profiler.configure_sagemaker(sagemaker_config.clone());
    println!("   SageMaker Configuration:");
    println!("      Training Job: {}", sagemaker_config.job_name);
    println!("      S3 Bucket: {}", sagemaker_config.s3_bucket);
    println!("      Framework: {}", sagemaker_config.framework);
    println!(
        "      Profiling Interval: {}s",
        sagemaker_config.profiling_interval_seconds
    );

    // Simulate S3 export
    let s3_path =
        aws_profiler.export_to_s3(&sagemaker_config.s3_bucket, &sagemaker_config.s3_prefix)?;
    println!("\n   ✓ Profiling data exported to: {}", s3_path);

    // ECS Task Configuration
    println!("\n   ECS Task Profiling:");
    let ecs_config = aws::ECSTaskProfilingConfig {
        task_definition: "arn:aws:ecs:us-east-1:123456789:task-definition/torsh-training:1"
            .to_string(),
        cluster: "ml-training-cluster".to_string(),
        service: Some("torsh-training-service".to_string()),
        log_group: "/aws/ecs/torsh-training".to_string(),
        container_insights: true,
    };
    println!("      Task Definition: {}", ecs_config.task_definition);
    println!("      Cluster: {}", ecs_config.cluster);
    println!("      Log Group: {}", ecs_config.log_group);
    println!(
        "      Container Insights: {}",
        ecs_config.container_insights
    );

    // ========================================
    // Part 4: Azure Integration
    // ========================================
    println!("\n4. Azure Integration (Azure ML, AKS)");
    println!("   Configuring profiling for Azure services\n");

    let mut azure_profiler = azure::AzureProfiler::default();

    // Configure for Azure Machine Learning
    let azureml_config = azure::AzureMLProfilingConfig {
        workspace_name: "torsh-ml-workspace".to_string(),
        resource_group: "ml-resources".to_string(),
        experiment_name: "distributed-training-v1".to_string(),
        storage_account: "torshmlstorage".to_string(),
        container_name: "profiling-data".to_string(),
        application_insights: true,
    };

    azure_profiler.configure_azureml(azureml_config.clone());
    println!("   Azure ML Configuration:");
    println!("      Workspace: {}", azureml_config.workspace_name);
    println!("      Resource Group: {}", azureml_config.resource_group);
    println!("      Experiment: {}", azureml_config.experiment_name);
    println!("      Storage Account: {}", azureml_config.storage_account);
    println!(
        "      Application Insights: {}",
        azureml_config.application_insights
    );

    // Simulate Azure Blob Storage export
    let blob_path = azure_profiler
        .export_to_blob_storage(&azureml_config.container_name, "profiling-2024-10-24.json")?;
    println!("\n   ✓ Profiling data exported to: {}", blob_path);

    // ========================================
    // Part 5: GCP Integration
    // ========================================
    println!("\n5. GCP Integration (Vertex AI, GKE)");
    println!("   Configuring profiling for GCP services\n");

    let mut gcp_profiler = gcp::GCPProfiler::default();

    // Configure for Vertex AI
    let vertex_config = gcp::VertexAIProfilingConfig {
        project_id: "my-ml-project".to_string(),
        location: "us-central1".to_string(),
        pipeline_id: "pytorch-training-pipeline-v1".to_string(),
        gcs_bucket: "torsh-vertex-profiling".to_string(),
        gcs_prefix: "experiments/2024-10-24".to_string(),
        cloud_profiler: true,
        tensorboard_profiling: true,
    };

    gcp_profiler.configure_vertex_ai(vertex_config.clone());
    println!("   Vertex AI Configuration:");
    println!("      Project ID: {}", vertex_config.project_id);
    println!("      Location: {}", vertex_config.location);
    println!("      Pipeline ID: {}", vertex_config.pipeline_id);
    println!("      GCS Bucket: {}", vertex_config.gcs_bucket);
    println!("      Cloud Profiler: {}", vertex_config.cloud_profiler);
    println!(
        "      TensorBoard Profiling: {}",
        vertex_config.tensorboard_profiling
    );

    // Simulate GCS export
    let gcs_path =
        gcp_profiler.export_to_gcs(&vertex_config.gcs_bucket, "profiling-2024-10-24.json")?;
    println!("\n   ✓ Profiling data exported to: {}", gcs_path);

    // ========================================
    // Part 6: Cost Optimization Analysis
    // ========================================
    println!("\n6. Cost Optimization Analysis");
    println!("   Analyzing profiling costs across cloud providers\n");

    println!("   Estimated Hourly Costs:");
    println!("   {}", "-".repeat(60));

    // AWS
    let aws_instances = vec![
        ("p3.2xlarge (V100)", 3.06),
        ("p3.8xlarge (4x V100)", 12.24),
        ("p4d.24xlarge (8x A100)", 32.77),
        ("g5.xlarge (A10G)", 1.006),
    ];

    println!("\n   AWS EC2 Instances:");
    for (name, cost) in aws_instances {
        println!("      {:<30} ${:.2}/hour", name, cost);
    }

    // Azure
    let azure_instances = vec![
        ("NC6 (K80)", 0.90),
        ("NC12 (2x K80)", 1.80),
        ("NCv3-24 (4x V100)", 10.32),
        ("NDv2-40 (8x V100)", 22.03),
    ];

    println!("\n   Azure Virtual Machines:");
    for (name, cost) in azure_instances {
        println!("      {:<30} ${:.2}/hour", name, cost);
    }

    // GCP
    let gcp_instances = vec![
        ("n1-standard-4 + K80", 0.70),
        ("n1-standard-4 + T4", 0.95),
        ("n1-standard-16 + 4x V100", 8.74),
        ("a2-highgpu-8g (8x A100)", 16.65),
    ];

    println!("\n   GCP Compute Engine:");
    for (name, cost) in gcp_instances {
        println!("      {:<30} ${:.2}/hour", name, cost);
    }

    println!("\n   {}", "-".repeat(60));
    println!("   Note: Costs are approximate and vary by region and commitment");

    // ========================================
    // Part 7: Deployment Recommendations
    // ========================================
    println!("\n7. Cloud-Specific Deployment Recommendations");
    println!("   {}", "=".repeat(60));

    println!("\n   AWS Deployment:");
    println!("      • Use SageMaker for managed training with built-in profiling");
    println!("      • Export to S3 for durable storage");
    println!("      • Use CloudWatch for metrics and alerting");
    println!("      • Consider Spot instances for cost savings (up to 90%)");
    println!("      • Use EFS for shared file systems across training instances");

    println!("\n   Azure Deployment:");
    println!("      • Use Azure ML for end-to-end ML workflow");
    println!("      • Export to Azure Blob Storage");
    println!("      • Use Application Insights for telemetry");
    println!("      • Consider Low Priority VMs for cost optimization");
    println!("      • Use Azure Files for shared storage");

    println!("\n   GCP Deployment:");
    println!("      • Use Vertex AI for scalable ML training");
    println!("      • Export to Google Cloud Storage");
    println!("      • Use Cloud Monitoring and Cloud Profiler");
    println!("      • Consider Preemptible VMs for cost savings (up to 80%)");
    println!("      • Use Filestore for shared NFS storage");

    println!("\n   {}", "=".repeat(60));

    // ========================================
    // Part 8: Best Practices
    // ========================================
    println!("\n8. Multi-Cloud Best Practices");
    println!("   {}", "=".repeat(60));

    println!("\n   Performance:");
    println!("      ✓ Profile on same instance type as production");
    println!("      ✓ Use local SSD for fast profiling data writes");
    println!("      ✓ Configure appropriate profiling intervals (30-60s)");
    println!("      ✓ Use sampling for production workloads");

    println!("\n   Cost Optimization:");
    println!("      ✓ Use spot/preemptible instances for development");
    println!("      ✓ Export profiling data to object storage immediately");
    println!("      ✓ Clean up old profiling data regularly");
    println!("      ✓ Use lifecycle policies for automatic archival");

    println!("\n   Security:");
    println!("      ✓ Use IAM roles for cloud service authentication");
    println!("      ✓ Encrypt profiling data at rest and in transit");
    println!("      ✓ Use private endpoints for sensitive workloads");
    println!("      ✓ Enable audit logging for compliance");

    println!("\n   Reliability:");
    println!("      ✓ Use managed services (SageMaker, Azure ML, Vertex AI)");
    println!("      ✓ Configure automatic retries for export failures");
    println!("      ✓ Monitor profiler health with cloud-native tools");
    println!("      ✓ Set up alerts for profiling anomalies");

    println!("\n   {}", "=".repeat(60));

    println!("\n=== Demo Complete ===");
    println!("\nKey Features Demonstrated:");
    println!("  ✓ Automatic cloud provider detection");
    println!("  ✓ AWS SageMaker and ECS integration");
    println!("  ✓ Azure Machine Learning integration");
    println!("  ✓ Google Cloud Vertex AI integration");
    println!("  ✓ Multi-cloud cost analysis");
    println!("  ✓ Cloud-specific optimization recommendations");
    println!("  ✓ Unified profiling API across all providers");
    println!(
        "\nThe cloud provider integrations enable seamless profiling across AWS, Azure, and GCP!"
    );

    Ok(())
}
