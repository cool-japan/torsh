//! Kubernetes Operator for Cloud-Native Profiling
//!
//! This module provides Kubernetes integration for ToRSh profiling, enabling:
//! - Automatic profiling of training workloads in Kubernetes
//! - ConfigMap-based profiling configuration
//! - Metrics export to Prometheus/Grafana
//! - Custom Resource Definitions (CRDs) for profiling jobs
//! - Pod-level and cluster-level profiling coordination
//! - Integration with Kubernetes monitoring stack
//!
//! # Features
//!
//! - ProfilingJob CRD for declarative profiling
//! - Automatic discovery of training pods
//! - Real-time metrics aggregation across pods
//! - Integration with Kubernetes events
//! - Support for distributed training profiling
//! - CloudWatch/Prometheus/Grafana export
//! - Alert integration with Kubernetes monitoring

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use torsh_core::{Result as TorshResult, TorshError};

/// Kubernetes namespace for profiling resources
pub const PROFILER_NAMESPACE: &str = "torsh-profiler";

/// Kubernetes ProfilingJob Custom Resource Definition (CRD)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingJob {
    /// API version
    pub api_version: String,
    /// Resource kind
    pub kind: String,
    /// Metadata
    pub metadata: ProfilingJobMetadata,
    /// Specification
    pub spec: ProfilingJobSpec,
    /// Status
    pub status: Option<ProfilingJobStatus>,
}

/// ProfilingJob metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingJobMetadata {
    /// Job name
    pub name: String,
    /// Namespace
    pub namespace: String,
    /// Labels
    pub labels: HashMap<String, String>,
    /// Annotations
    pub annotations: HashMap<String, String>,
}

/// ProfilingJob specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingJobSpec {
    /// Target pod selector
    pub selector: PodSelector,
    /// Profiling configuration
    pub profiling_config: ProfilingConfig,
    /// Export configuration
    pub export_config: ExportConfig,
    /// Duration in seconds (0 = unlimited)
    pub duration_seconds: u64,
    /// Sampling rate (1.0 = all events, 0.1 = 10% of events)
    pub sampling_rate: f64,
    /// Enable distributed profiling coordination
    pub distributed: bool,
}

/// Pod selector for targeting profiling jobs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodSelector {
    /// Match labels
    pub match_labels: HashMap<String, String>,
    /// Match expressions
    pub match_expressions: Vec<LabelSelectorRequirement>,
}

/// Label selector requirement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelSelectorRequirement {
    /// Label key
    pub key: String,
    /// Operator (In, NotIn, Exists, DoesNotExist)
    pub operator: String,
    /// Values (for In, NotIn operators)
    pub values: Vec<String>,
}

/// Profiling configuration for Kubernetes jobs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    /// Enable CPU profiling
    pub enable_cpu: bool,
    /// Enable GPU profiling
    pub enable_gpu: bool,
    /// Enable memory profiling
    pub enable_memory: bool,
    /// Enable network profiling
    pub enable_network: bool,
    /// Enable distributed profiling
    pub enable_distributed: bool,
    /// Stack trace depth
    pub stack_trace_depth: usize,
    /// Overhead threshold (percentage)
    pub max_overhead_percent: f64,
    /// Custom profiling parameters
    pub custom_params: HashMap<String, String>,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enable_cpu: true,
            enable_gpu: true,
            enable_memory: true,
            enable_network: false,
            enable_distributed: false,
            stack_trace_depth: 10,
            max_overhead_percent: 5.0,
            custom_params: HashMap::new(),
        }
    }
}

/// Export configuration for profiling data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Export to Prometheus
    pub prometheus: Option<PrometheusExportConfig>,
    /// Export to CloudWatch
    pub cloudwatch: Option<CloudWatchExportConfig>,
    /// Export to Grafana
    pub grafana: Option<GrafanaExportConfig>,
    /// Export to S3/object storage
    pub object_storage: Option<ObjectStorageConfig>,
}

/// Prometheus export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrometheusExportConfig {
    /// Prometheus pushgateway URL
    pub pushgateway_url: String,
    /// Job name
    pub job_name: String,
    /// Additional labels
    pub labels: HashMap<String, String>,
    /// Push interval in seconds
    pub push_interval_seconds: u64,
}

/// CloudWatch export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudWatchExportConfig {
    /// AWS region
    pub region: String,
    /// Namespace
    pub namespace: String,
    /// Dimension mappings
    pub dimensions: HashMap<String, String>,
    /// Publish interval in seconds
    pub publish_interval_seconds: u64,
}

/// Grafana export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrafanaExportConfig {
    /// Grafana API URL
    pub api_url: String,
    /// API token
    pub api_token: String,
    /// Dashboard UID
    pub dashboard_uid: String,
    /// Auto-update dashboard
    pub auto_update: bool,
}

/// Object storage configuration (S3, GCS, Azure Blob)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectStorageConfig {
    /// Storage provider (s3, gcs, azure)
    pub provider: String,
    /// Bucket/container name
    pub bucket: String,
    /// Prefix/path
    pub prefix: String,
    /// Credentials (from Kubernetes secret)
    pub credentials_secret: String,
    /// Upload interval in seconds
    pub upload_interval_seconds: u64,
}

/// ProfilingJob status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingJobStatus {
    /// Job phase (Pending, Running, Completed, Failed)
    pub phase: String,
    /// Start time
    pub start_time: Option<String>,
    /// Completion time
    pub completion_time: Option<String>,
    /// Number of profiled pods
    pub profiled_pods: usize,
    /// Total events collected
    pub total_events: u64,
    /// Total overhead (percentage)
    pub total_overhead_percent: f64,
    /// Status message
    pub message: Option<String>,
    /// Conditions
    pub conditions: Vec<ProfilingCondition>,
}

/// ProfilingJob condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingCondition {
    /// Condition type
    pub condition_type: String,
    /// Status (True, False, Unknown)
    pub status: String,
    /// Last transition time
    pub last_transition_time: String,
    /// Reason
    pub reason: String,
    /// Message
    pub message: String,
}

/// Kubernetes Profiling Operator
pub struct KubernetesProfilerOperator {
    /// Operator namespace
    namespace: String,
    /// Active profiling jobs
    jobs: HashMap<String, ProfilingJob>,
    /// Pod profiler instances
    pod_profilers: HashMap<String, PodProfilerInstance>,
}

/// Pod-level profiler instance
#[derive(Debug, Clone)]
pub struct PodProfilerInstance {
    /// Pod name
    pub pod_name: String,
    /// Pod namespace
    pub pod_namespace: String,
    /// Node name
    pub node_name: String,
    /// Profiler active
    pub active: bool,
    /// Events collected
    pub events_collected: u64,
    /// Start time
    pub start_time: chrono::DateTime<chrono::Utc>,
    /// Last export time
    pub last_export: Option<chrono::DateTime<chrono::Utc>>,
}

impl KubernetesProfilerOperator {
    /// Create a new Kubernetes profiler operator
    pub fn new(namespace: String) -> Self {
        Self {
            namespace,
            jobs: HashMap::new(),
            pod_profilers: HashMap::new(),
        }
    }

    /// Create a new profiling job
    pub fn create_job(&mut self, job: ProfilingJob) -> TorshResult<()> {
        let job_name = job.metadata.name.clone();

        if self.jobs.contains_key(&job_name) {
            return Err(TorshError::InvalidArgument(format!(
                "ProfilingJob {} already exists",
                job_name
            )));
        }

        self.jobs.insert(job_name, job);
        Ok(())
    }

    /// Delete a profiling job
    pub fn delete_job(&mut self, job_name: &str) -> TorshResult<()> {
        self.jobs.remove(job_name).ok_or_else(|| {
            TorshError::operation_error(&format!("ProfilingJob {} not found", job_name))
        })?;
        Ok(())
    }

    /// Get job status
    pub fn get_job_status(&self, job_name: &str) -> TorshResult<ProfilingJobStatus> {
        let job = self.jobs.get(job_name).ok_or_else(|| {
            TorshError::operation_error(&format!("ProfilingJob {} not found", job_name))
        })?;

        job.status
            .clone()
            .ok_or_else(|| TorshError::operation_error("Job status not available"))
    }

    /// List all profiling jobs
    pub fn list_jobs(&self) -> Vec<&ProfilingJob> {
        self.jobs.values().collect()
    }

    /// Register a pod profiler instance
    pub fn register_pod(&mut self, instance: PodProfilerInstance) {
        let key = format!("{}/{}", instance.pod_namespace, instance.pod_name);
        self.pod_profilers.insert(key, instance);
    }

    /// Unregister a pod profiler instance
    pub fn unregister_pod(&mut self, pod_namespace: &str, pod_name: &str) {
        let key = format!("{}/{}", pod_namespace, pod_name);
        self.pod_profilers.remove(&key);
    }

    /// Get all active pod profilers
    pub fn active_pods(&self) -> Vec<&PodProfilerInstance> {
        self.pod_profilers.values().filter(|p| p.active).collect()
    }

    /// Generate ConfigMap for profiling configuration
    pub fn generate_configmap(&self, job_name: &str) -> TorshResult<String> {
        let job = self.jobs.get(job_name).ok_or_else(|| {
            TorshError::operation_error(&format!("ProfilingJob {} not found", job_name))
        })?;

        let configmap = format!(
            r#"apiVersion: v1
kind: ConfigMap
metadata:
  name: {}-config
  namespace: {}
data:
  profiling.json: |
    {}
"#,
            job_name,
            job.metadata.namespace,
            serde_json::to_string_pretty(&job.spec.profiling_config).map_err(|e| {
                TorshError::operation_error(&format!("JSON serialization failed: {}", e))
            })?
        );

        Ok(configmap)
    }

    /// Generate Kubernetes service for metrics export
    pub fn generate_metrics_service(&self, job_name: &str) -> TorshResult<String> {
        let job = self.jobs.get(job_name).ok_or_else(|| {
            TorshError::operation_error(&format!("ProfilingJob {} not found", job_name))
        })?;

        let service = format!(
            r#"apiVersion: v1
kind: Service
metadata:
  name: {}-metrics
  namespace: {}
  labels:
    app: torsh-profiler
    job: {}
spec:
  type: ClusterIP
  selector:
    app: torsh-profiler
    job: {}
  ports:
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
  - name: http
    port: 8080
    targetPort: 8080
    protocol: TCP
"#,
            job_name, job.metadata.namespace, job_name, job_name
        );

        Ok(service)
    }

    /// Generate ServiceMonitor for Prometheus Operator
    pub fn generate_service_monitor(&self, job_name: &str) -> TorshResult<String> {
        let job = self.jobs.get(job_name).ok_or_else(|| {
            TorshError::operation_error(&format!("ProfilingJob {} not found", job_name))
        })?;

        let service_monitor = format!(
            r#"apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: {}-monitor
  namespace: {}
  labels:
    app: torsh-profiler
    job: {}
spec:
  selector:
    matchLabels:
      app: torsh-profiler
      job: {}
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
"#,
            job_name, job.metadata.namespace, job_name, job_name
        );

        Ok(service_monitor)
    }

    /// Export operator state to JSON
    pub fn export_state(&self) -> TorshResult<String> {
        #[derive(Serialize)]
        struct OperatorState<'a> {
            namespace: &'a str,
            active_jobs: usize,
            active_pods: usize,
            total_events: u64,
        }

        let total_events: u64 = self
            .pod_profilers
            .values()
            .map(|p| p.events_collected)
            .sum();

        let state = OperatorState {
            namespace: &self.namespace,
            active_jobs: self.jobs.len(),
            active_pods: self.active_pods().len(),
            total_events,
        };

        serde_json::to_string_pretty(&state)
            .map_err(|e| TorshError::operation_error(&format!("JSON export failed: {}", e)))
    }
}

impl Default for KubernetesProfilerOperator {
    fn default() -> Self {
        Self::new(PROFILER_NAMESPACE.to_string())
    }
}

/// Helm chart values generator for ToRSh Profiler
pub struct HelmChartGenerator;

impl HelmChartGenerator {
    /// Generate values.yaml for Helm chart
    pub fn generate_values_yaml(job_name: &str, config: &ProfilingConfig) -> String {
        format!(
            r#"# ToRSh Profiler Helm Chart Values

replicaCount: 1

image:
  repository: torsh/profiler
  pullPolicy: IfNotPresent
  tag: "latest"

nameOverride: "{}"
fullnameOverride: "{}-profiler"

serviceAccount:
  create: true
  name: torsh-profiler

profiling:
  enabled: true
  cpuProfiling: {}
  gpuProfiling: {}
  memoryProfiling: {}
  networkProfiling: {}
  distributedProfiling: {}
  stackTraceDepth: {}
  maxOverheadPercent: {}

metrics:
  enabled: true
  port: 9090
  serviceMonitor:
    enabled: true
    interval: 30s

resources:
  limits:
    cpu: 500m
    memory: 512Mi
  requests:
    cpu: 100m
    memory: 128Mi

nodeSelector: {{}}

tolerations: []

affinity: {{}}
"#,
            job_name,
            job_name,
            config.enable_cpu,
            config.enable_gpu,
            config.enable_memory,
            config.enable_network,
            config.enable_distributed,
            config.stack_trace_depth,
            config.max_overhead_percent
        )
    }

    /// Generate Chart.yaml for Helm chart
    pub fn generate_chart_yaml() -> String {
        r#"apiVersion: v2
name: torsh-profiler
description: A Helm chart for ToRSh Profiler Operator
type: application
version: 0.1.0
appVersion: "0.1.0-alpha.2"
keywords:
  - machine-learning
  - profiling
  - pytorch
  - rust
maintainers:
  - name: ToRSh Team
    email: torsh@example.com
"#
        .to_string()
    }

    /// Generate deployment.yaml template
    pub fn generate_deployment_template() -> String {
        r#"apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "torsh-profiler.fullname" . }}
  labels:
    {{- include "torsh-profiler.labels" . | nindent 4 }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      {{- include "torsh-profiler.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "torsh-profiler.selectorLabels" . | nindent 8 }}
    spec:
      serviceAccountName: {{ include "torsh-profiler.serviceAccountName" . }}
      containers:
      - name: profiler
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
        imagePullPolicy: {{ .Values.image.pullPolicy }}
        ports:
        - name: metrics
          containerPort: 9090
          protocol: TCP
        - name: http
          containerPort: 8080
          protocol: TCP
        env:
        - name: PROFILER_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        resources:
          {{- toYaml .Values.resources | nindent 12 }}
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
"#
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_creation() {
        let operator = KubernetesProfilerOperator::new("default".to_string());
        assert_eq!(operator.namespace, "default");
        assert_eq!(operator.jobs.len(), 0);
        assert_eq!(operator.pod_profilers.len(), 0);
    }

    #[test]
    fn test_job_creation() {
        let mut operator = KubernetesProfilerOperator::default();

        let job = ProfilingJob {
            api_version: "profiler.torsh.dev/v1".to_string(),
            kind: "ProfilingJob".to_string(),
            metadata: ProfilingJobMetadata {
                name: "test-job".to_string(),
                namespace: "default".to_string(),
                labels: HashMap::new(),
                annotations: HashMap::new(),
            },
            spec: ProfilingJobSpec {
                selector: PodSelector {
                    match_labels: vec![("app".to_string(), "training".to_string())]
                        .into_iter()
                        .collect(),
                    match_expressions: vec![],
                },
                profiling_config: ProfilingConfig::default(),
                export_config: ExportConfig {
                    prometheus: None,
                    cloudwatch: None,
                    grafana: None,
                    object_storage: None,
                },
                duration_seconds: 3600,
                sampling_rate: 1.0,
                distributed: false,
            },
            status: None,
        };

        operator.create_job(job).unwrap();
        assert_eq!(operator.jobs.len(), 1);
        assert!(operator.jobs.contains_key("test-job"));
    }

    #[test]
    fn test_pod_registration() {
        let mut operator = KubernetesProfilerOperator::default();

        let instance = PodProfilerInstance {
            pod_name: "training-pod-1".to_string(),
            pod_namespace: "default".to_string(),
            node_name: "node-1".to_string(),
            active: true,
            events_collected: 0,
            start_time: chrono::Utc::now(),
            last_export: None,
        };

        operator.register_pod(instance);
        assert_eq!(operator.pod_profilers.len(), 1);
        assert_eq!(operator.active_pods().len(), 1);
    }

    #[test]
    fn test_configmap_generation() {
        let mut operator = KubernetesProfilerOperator::default();

        let job = ProfilingJob {
            api_version: "profiler.torsh.dev/v1".to_string(),
            kind: "ProfilingJob".to_string(),
            metadata: ProfilingJobMetadata {
                name: "test-job".to_string(),
                namespace: "default".to_string(),
                labels: HashMap::new(),
                annotations: HashMap::new(),
            },
            spec: ProfilingJobSpec {
                selector: PodSelector {
                    match_labels: HashMap::new(),
                    match_expressions: vec![],
                },
                profiling_config: ProfilingConfig::default(),
                export_config: ExportConfig {
                    prometheus: None,
                    cloudwatch: None,
                    grafana: None,
                    object_storage: None,
                },
                duration_seconds: 3600,
                sampling_rate: 1.0,
                distributed: false,
            },
            status: None,
        };

        operator.create_job(job).unwrap();

        let configmap = operator.generate_configmap("test-job").unwrap();
        assert!(configmap.contains("kind: ConfigMap"));
        assert!(configmap.contains("test-job-config"));
    }

    #[test]
    fn test_service_generation() {
        let mut operator = KubernetesProfilerOperator::default();

        let job = ProfilingJob {
            api_version: "profiler.torsh.dev/v1".to_string(),
            kind: "ProfilingJob".to_string(),
            metadata: ProfilingJobMetadata {
                name: "test-job".to_string(),
                namespace: "default".to_string(),
                labels: HashMap::new(),
                annotations: HashMap::new(),
            },
            spec: ProfilingJobSpec {
                selector: PodSelector {
                    match_labels: HashMap::new(),
                    match_expressions: vec![],
                },
                profiling_config: ProfilingConfig::default(),
                export_config: ExportConfig {
                    prometheus: None,
                    cloudwatch: None,
                    grafana: None,
                    object_storage: None,
                },
                duration_seconds: 3600,
                sampling_rate: 1.0,
                distributed: false,
            },
            status: None,
        };

        operator.create_job(job).unwrap();

        let service = operator.generate_metrics_service("test-job").unwrap();
        assert!(service.contains("kind: Service"));
        assert!(service.contains("test-job-metrics"));
    }

    #[test]
    fn test_helm_chart_generation() {
        let values =
            HelmChartGenerator::generate_values_yaml("my-job", &ProfilingConfig::default());
        assert!(values.contains("my-job"));
        assert!(values.contains("cpuProfiling: true"));

        let chart = HelmChartGenerator::generate_chart_yaml();
        assert!(chart.contains("torsh-profiler"));

        let deployment = HelmChartGenerator::generate_deployment_template();
        assert!(deployment.contains("Deployment"));
    }

    #[test]
    fn test_state_export() {
        let operator = KubernetesProfilerOperator::default();
        let state = operator.export_state().unwrap();
        assert!(state.contains("namespace"));
        assert!(state.contains("active_jobs"));
    }
}
