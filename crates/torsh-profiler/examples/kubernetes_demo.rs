//! Kubernetes Operator Demo
//!
//! This example demonstrates the Kubernetes operator capabilities for cloud-native profiling,
//! including ProfilingJob CRDs, ConfigMap generation, and Prometheus integration.

use std::collections::HashMap;
use torsh_profiler::kubernetes::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToRSh Profiler: Kubernetes Operator Demo ===\n");

    // ========================================
    // Part 1: Create Kubernetes Profiler Operator
    // ========================================
    println!("1. Creating Kubernetes Profiler Operator");
    println!("   Setting up operator in 'torsh-profiler' namespace\n");

    let mut operator = KubernetesProfilerOperator::new("torsh-profiler".to_string());
    println!("   ✓ Operator created successfully");

    // ========================================
    // Part 2: Define ProfilingJob CRD
    // ========================================
    println!("\n2. Creating ProfilingJob Custom Resource");
    println!("   Defining profiling job for distributed training\n");

    let mut labels = HashMap::new();
    labels.insert("app".to_string(), "torsh-training".to_string());
    labels.insert("version".to_string(), "v0.1.0".to_string());

    let mut annotations = HashMap::new();
    annotations.insert(
        "description".to_string(),
        "Profile distributed training job".to_string(),
    );

    let mut match_labels = HashMap::new();
    match_labels.insert("app".to_string(), "training".to_string());
    match_labels.insert("framework".to_string(), "torsh".to_string());

    let profiling_job = ProfilingJob {
        api_version: "profiler.torsh.dev/v1".to_string(),
        kind: "ProfilingJob".to_string(),
        metadata: ProfilingJobMetadata {
            name: "distributed-training-profile".to_string(),
            namespace: "default".to_string(),
            labels,
            annotations,
        },
        spec: ProfilingJobSpec {
            selector: PodSelector {
                match_labels,
                match_expressions: vec![],
            },
            profiling_config: ProfilingConfig {
                enable_cpu: true,
                enable_gpu: true,
                enable_memory: true,
                enable_network: true,
                enable_distributed: true,
                stack_trace_depth: 15,
                max_overhead_percent: 5.0,
                custom_params: HashMap::new(),
            },
            export_config: ExportConfig {
                prometheus: Some(PrometheusExportConfig {
                    pushgateway_url: "http://prometheus-pushgateway:9091".to_string(),
                    job_name: "torsh-profiler".to_string(),
                    labels: vec![("env".to_string(), "production".to_string())]
                        .into_iter()
                        .collect(),
                    push_interval_seconds: 30,
                }),
                cloudwatch: None,
                grafana: Some(GrafanaExportConfig {
                    api_url: "http://grafana:3000".to_string(),
                    api_token: "grafana-api-token".to_string(),
                    dashboard_uid: "torsh-profiling".to_string(),
                    auto_update: true,
                }),
                object_storage: Some(ObjectStorageConfig {
                    provider: "s3".to_string(),
                    bucket: "torsh-profiling-data".to_string(),
                    prefix: "distributed-training".to_string(),
                    credentials_secret: "aws-credentials".to_string(),
                    upload_interval_seconds: 300,
                }),
            },
            duration_seconds: 7200, // 2 hours
            sampling_rate: 1.0,
            distributed: true,
        },
        status: None,
    };

    println!("   ProfilingJob Specification:");
    println!("      Name: {}", profiling_job.metadata.name);
    println!("      Namespace: {}", profiling_job.metadata.namespace);
    println!(
        "      Duration: {} seconds",
        profiling_job.spec.duration_seconds
    );
    println!(
        "      Sampling Rate: {:.0}%",
        profiling_job.spec.sampling_rate * 100.0
    );
    println!("      Distributed: {}", profiling_job.spec.distributed);

    // Create the job
    operator.create_job(profiling_job)?;
    println!("\n   ✓ ProfilingJob created successfully");

    // ========================================
    // Part 3: Register Pod Profilers
    // ========================================
    println!("\n3. Registering Pod Profiler Instances");
    println!("   Simulating distributed training pods\n");

    for i in 0..4 {
        let instance = PodProfilerInstance {
            pod_name: format!("training-pod-{}", i),
            pod_namespace: "default".to_string(),
            node_name: format!("worker-node-{}", i % 2),
            active: true,
            events_collected: i as u64 * 1000,
            start_time: chrono::Utc::now(),
            last_export: None,
        };

        operator.register_pod(instance.clone());
        println!(
            "   ✓ Registered: {} on {}",
            instance.pod_name, instance.node_name
        );
    }

    let active_pods = operator.active_pods();
    println!("\n   Active profiler instances: {}", active_pods.len());

    // ========================================
    // Part 4: Generate Kubernetes Resources
    // ========================================
    println!("\n4. Generating Kubernetes Resources");
    println!("   Creating ConfigMaps, Services, and ServiceMonitors\n");

    // Generate ConfigMap
    let configmap = operator.generate_configmap("distributed-training-profile")?;
    println!("   ConfigMap YAML:");
    println!("   {}", "-".repeat(60));
    for line in configmap.lines().take(15) {
        println!("   {}", line);
    }
    println!("   {} [truncated]", "-".repeat(60));

    // Generate Service
    let service = operator.generate_metrics_service("distributed-training-profile")?;
    println!("\n   Service YAML:");
    println!("   {}", "-".repeat(60));
    for line in service.lines().take(20) {
        println!("   {}", line);
    }
    println!("   {} [truncated]", "-".repeat(60));

    // Generate ServiceMonitor for Prometheus Operator
    let service_monitor = operator.generate_service_monitor("distributed-training-profile")?;
    println!("\n   ServiceMonitor YAML:");
    println!("   {}", "-".repeat(60));
    for line in service_monitor.lines().take(18) {
        println!("   {}", line);
    }
    println!("   {}", "-".repeat(60));

    // ========================================
    // Part 5: Helm Chart Generation
    // ========================================
    println!("\n5. Generating Helm Chart");
    println!("   Creating Helm chart for easy deployment\n");

    let chart_yaml = HelmChartGenerator::generate_chart_yaml();
    println!("   Chart.yaml:");
    println!("   {}", "-".repeat(60));
    for line in chart_yaml.lines().take(12) {
        println!("   {}", line);
    }
    println!("   {}", "-".repeat(60));

    let values_yaml =
        HelmChartGenerator::generate_values_yaml("my-training-job", &ProfilingConfig::default());
    println!("\n   values.yaml:");
    println!("   {}", "-".repeat(60));
    for line in values_yaml.lines().take(25) {
        println!("   {}", line);
    }
    println!("   {} [truncated]", "-".repeat(60));

    let deployment_yaml = HelmChartGenerator::generate_deployment_template();
    println!("\n   templates/deployment.yaml:");
    println!("   {}", "-".repeat(60));
    for line in deployment_yaml.lines().take(30) {
        println!("   {}", line);
    }
    println!("   {} [truncated]", "-".repeat(60));

    // ========================================
    // Part 6: Operator State Export
    // ========================================
    println!("\n6. Exporting Operator State");

    let state = operator.export_state()?;
    println!("   Operator State:");
    println!("{}", state);

    // ========================================
    // Part 7: List All Jobs
    // ========================================
    println!("\n7. Listing ProfilingJobs");

    let jobs = operator.list_jobs();
    println!("   Total ProfilingJobs: {}", jobs.len());
    for job in jobs {
        println!(
            "      • {} (namespace: {})",
            job.metadata.name, job.metadata.namespace
        );
    }

    // ========================================
    // Part 8: Usage Instructions
    // ========================================
    println!("\n8. Deployment Instructions");
    println!("   {}", "=".repeat(60));
    println!("\n   To deploy the ToRSh Profiler operator:");
    println!("\n   1. Create the namespace:");
    println!("      $ kubectl create namespace torsh-profiler");
    println!("\n   2. Install the Helm chart:");
    println!("      $ helm install torsh-profiler ./torsh-profiler-chart");
    println!("\n   3. Apply the ProfilingJob:");
    println!("      $ kubectl apply -f profiling-job.yaml");
    println!("\n   4. Check the status:");
    println!("      $ kubectl get profilingjobs -n default");
    println!("\n   5. View metrics:");
    println!("      $ kubectl port-forward svc/distributed-training-profile-metrics 9090:9090");
    println!("      Open http://localhost:9090/metrics");
    println!("\n   6. Access Grafana dashboard:");
    println!("      $ kubectl port-forward svc/grafana 3000:3000");
    println!("      Open http://localhost:3000");
    println!("\n   {}", "=".repeat(60));

    println!("\n=== Demo Complete ===");
    println!("\nKey Features Demonstrated:");
    println!("  ✓ ProfilingJob CRD for declarative profiling");
    println!("  ✓ Automatic ConfigMap generation for profiling configuration");
    println!("  ✓ Service and ServiceMonitor creation for Prometheus");
    println!("  ✓ Helm chart generation for easy deployment");
    println!("  ✓ Multi-pod profiling coordination");
    println!("  ✓ Integration with Grafana and CloudWatch");
    println!("  ✓ S3/object storage export support");
    println!("\nThe Kubernetes operator enables cloud-native profiling at scale!");

    Ok(())
}
