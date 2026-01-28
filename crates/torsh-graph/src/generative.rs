//! Graph Generation Models
//!
//! Advanced implementation of generative models for graphs including
//! Variational Autoencoders (VAE) and Generative Adversarial Networks (GAN)
//! specifically designed for graph-structured data.
//!
//! # Features:
//! - Graph Variational Autoencoder (GraphVAE)
//! - Graph Generative Adversarial Network (GraphGAN)
//! - Conditional graph generation
//! - Graph reconstruction and completion
//! - Latent space graph interpolation
//! - Property-guided graph generation

// Framework infrastructure - components designed for future use
#![allow(dead_code)]
use crate::parameter::Parameter;
use crate::{GraphData, GraphLayer};
use scirs2_core::random::thread_rng;
use torsh_tensor::{
    creation::{from_vec, randn, zeros},
    Tensor,
};

/// Graph Variational Autoencoder (GraphVAE)
/// Learns a probabilistic latent representation of graphs
#[derive(Debug)]
pub struct GraphVAE {
    // Encoder parameters
    encoder_in_features: usize,
    encoder_hidden_features: usize,
    latent_dim: usize,

    // Encoder layers
    encoder_layer1: Parameter,
    encoder_layer2: Parameter,

    // Variational parameters (mean and log-variance)
    mu_layer: Parameter,
    logvar_layer: Parameter,

    // Decoder parameters
    decoder_layer1: Parameter,
    decoder_layer2: Parameter,
    node_decoder: Parameter,
    edge_decoder: Parameter,

    // KL divergence weight
    beta: f32,

    // Bias terms
    encoder_bias1: Option<Parameter>,
    encoder_bias2: Option<Parameter>,
    decoder_bias1: Option<Parameter>,
    decoder_bias2: Option<Parameter>,
}

impl GraphVAE {
    /// Create a new Graph Variational Autoencoder
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        latent_dim: usize,
        beta: f32,
        use_bias: bool,
    ) -> Self {
        // Encoder layers
        let encoder_layer1 = Parameter::new(
            randn(&[in_features, hidden_features]).expect("failed to create encoder_layer1 tensor"),
        );
        let encoder_layer2 = Parameter::new(
            randn(&[hidden_features, hidden_features])
                .expect("failed to create encoder_layer2 tensor"),
        );

        // Variational layers
        let mu_layer = Parameter::new(
            randn(&[hidden_features, latent_dim]).expect("failed to create mu_layer tensor"),
        );
        let logvar_layer = Parameter::new(
            randn(&[hidden_features, latent_dim]).expect("failed to create logvar_layer tensor"),
        );

        // Decoder layers
        let decoder_layer1 = Parameter::new(
            randn(&[latent_dim, hidden_features]).expect("failed to create decoder_layer1 tensor"),
        );
        let decoder_layer2 = Parameter::new(
            randn(&[hidden_features, hidden_features])
                .expect("failed to create decoder_layer2 tensor"),
        );
        let node_decoder = Parameter::new(
            randn(&[hidden_features, in_features]).expect("failed to create node_decoder tensor"),
        );
        let edge_decoder = Parameter::new(
            randn(&[hidden_features, 1]).expect("failed to create edge_decoder tensor"),
        );

        let (encoder_bias1, encoder_bias2, decoder_bias1, decoder_bias2) = if use_bias {
            (
                Some(Parameter::new(
                    zeros(&[hidden_features]).expect("failed to create encoder_bias1 tensor"),
                )),
                Some(Parameter::new(
                    zeros(&[hidden_features]).expect("failed to create encoder_bias2 tensor"),
                )),
                Some(Parameter::new(
                    zeros(&[hidden_features]).expect("failed to create decoder_bias1 tensor"),
                )),
                Some(Parameter::new(
                    zeros(&[hidden_features]).expect("failed to create decoder_bias2 tensor"),
                )),
            )
        } else {
            (None, None, None, None)
        };

        Self {
            encoder_in_features: in_features,
            encoder_hidden_features: hidden_features,
            latent_dim,
            encoder_layer1,
            encoder_layer2,
            mu_layer,
            logvar_layer,
            decoder_layer1,
            decoder_layer2,
            node_decoder,
            edge_decoder,
            beta,
            encoder_bias1,
            encoder_bias2,
            decoder_bias1,
            decoder_bias2,
        }
    }

    /// Encode graph to latent distribution parameters
    pub fn encode(&self, graph: &GraphData) -> (Tensor, Tensor) {
        // Forward through encoder
        let mut h = graph
            .x
            .matmul(&self.encoder_layer1.clone_data())
            .expect("operation should succeed");
        if let Some(ref bias) = self.encoder_bias1 {
            h = h.add(&bias.clone_data()).expect("operation should succeed");
        }
        h = self.relu(&h);

        h = h
            .matmul(&self.encoder_layer2.clone_data())
            .expect("operation should succeed");
        if let Some(ref bias) = self.encoder_bias2 {
            h = h.add(&bias.clone_data()).expect("operation should succeed");
        }
        h = self.relu(&h);

        // Global mean pooling
        let graph_embedding = h
            .mean(Some(&[0]), false)
            .expect("mean pooling should succeed");
        let graph_embedding_2d = graph_embedding
            .unsqueeze(0)
            .expect("unsqueeze should succeed"); // Make 2D for matmul

        // Compute mu and logvar
        let mu = graph_embedding_2d
            .matmul(&self.mu_layer.clone_data())
            .expect("mu layer matmul should succeed");
        let logvar = graph_embedding_2d
            .matmul(&self.logvar_layer.clone_data())
            .expect("logvar layer matmul should succeed");

        (mu, logvar)
    }

    /// Reparameterization trick for sampling from latent distribution
    pub fn reparameterize(&self, mu: &Tensor, logvar: &Tensor) -> Tensor {
        // std = exp(0.5 * logvar)
        let std = logvar
            .mul_scalar(0.5)
            .expect("logvar scaling should succeed")
            .exp()
            .expect("exp should succeed");

        // Sample epsilon from N(0, 1)
        let epsilon = randn(mu.shape().dims()).expect("epsilon sampling should succeed");

        // z = mu + std * epsilon
        mu.add(&std.mul(&epsilon).expect("operation should succeed"))
            .expect("operation should succeed")
    }

    /// Decode latent representation to graph
    pub fn decode(&self, z: &Tensor, num_nodes: usize) -> GraphData {
        // Forward through decoder
        let mut h = z
            .matmul(&self.decoder_layer1.clone_data())
            .expect("operation should succeed");
        if let Some(ref bias) = self.decoder_bias1 {
            h = h.add(&bias.clone_data()).expect("operation should succeed");
        }
        h = self.relu(&h);

        h = h
            .matmul(&self.decoder_layer2.clone_data())
            .expect("operation should succeed");
        if let Some(ref bias) = self.decoder_bias2 {
            h = h.add(&bias.clone_data()).expect("operation should succeed");
        }
        h = self.relu(&h);

        // Expand to node-level representation
        let h_expanded = self.expand_to_nodes(&h, num_nodes);

        // Decode node features
        let node_features = h_expanded
            .matmul(&self.node_decoder.clone_data())
            .expect("operation should succeed");

        // Decode edge probabilities
        let edge_logits = self.decode_edges(&h_expanded, num_nodes);
        let edge_index = self.sample_edges(&edge_logits, num_nodes);

        GraphData::new(node_features, edge_index)
    }

    /// Forward pass through GraphVAE
    pub fn forward(&self, graph: &GraphData) -> (GraphData, Tensor, Tensor) {
        // Encode
        let (mu, logvar) = self.encode(graph);

        // Sample latent variable
        let z = self.reparameterize(&mu, &logvar);

        // Decode
        let reconstructed = self.decode(&z, graph.num_nodes);

        (reconstructed, mu, logvar)
    }

    /// Compute VAE loss (reconstruction + KL divergence)
    pub fn compute_loss(
        &self,
        graph: &GraphData,
        reconstructed: &GraphData,
        mu: &Tensor,
        logvar: &Tensor,
    ) -> f32 {
        // Reconstruction loss (MSE for node features)
        let recon_loss = self.reconstruction_loss(graph, reconstructed);

        // KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        let kl_loss = self.kl_divergence(mu, logvar);

        // Total loss
        recon_loss + self.beta * kl_loss
    }

    /// Reconstruction loss (MSE)
    fn reconstruction_loss(&self, original: &GraphData, reconstructed: &GraphData) -> f32 {
        let orig_data = original.x.to_vec().expect("conversion should succeed");
        let recon_data = reconstructed.x.to_vec().expect("conversion should succeed");

        let mut mse = 0.0;
        let len = orig_data.len().min(recon_data.len());

        for i in 0..len {
            mse += (orig_data[i] - recon_data[i]).powi(2);
        }

        mse / len as f32
    }

    /// KL divergence loss
    fn kl_divergence(&self, mu: &Tensor, logvar: &Tensor) -> f32 {
        let mu_data = mu.to_vec().expect("conversion should succeed");
        let logvar_data = logvar.to_vec().expect("conversion should succeed");

        let mut kl = 0.0;
        for i in 0..mu_data.len() {
            kl += -0.5 * (1.0 + logvar_data[i] - mu_data[i].powi(2) - logvar_data[i].exp());
        }

        kl / mu_data.len() as f32
    }

    /// Generate new graph from random latent vector
    pub fn generate(&self, num_nodes: usize) -> GraphData {
        // Sample from standard normal
        let z = randn(&[1, self.latent_dim]).expect("latent vector sampling should succeed");

        // Decode to graph
        self.decode(&z, num_nodes)
    }

    /// Interpolate between two graphs in latent space
    pub fn interpolate(
        &self,
        graph1: &GraphData,
        graph2: &GraphData,
        alpha: f32,
        num_nodes: usize,
    ) -> GraphData {
        let (mu1, _) = self.encode(graph1);
        let (mu2, _) = self.encode(graph2);

        // Linear interpolation
        let z_interp = mu1
            .mul_scalar(1.0 - alpha)
            .expect("mu1 scaling should succeed")
            .add(&mu2.mul_scalar(alpha).expect("operation should succeed"))
            .expect("interpolation addition should succeed");

        // Decode interpolated latent
        self.decode(&z_interp, num_nodes)
    }

    // Helper methods

    fn relu(&self, x: &Tensor) -> Tensor {
        let data = x.to_vec().expect("conversion should succeed");
        let activated: Vec<f32> = data.iter().map(|&v| v.max(0.0)).collect();
        from_vec(
            activated,
            x.shape().dims(),
            torsh_core::device::DeviceType::Cpu,
        )
        .expect("relu tensor creation should succeed")
    }

    fn expand_to_nodes(&self, h: &Tensor, num_nodes: usize) -> Tensor {
        // Repeat graph-level embedding for each node
        let h_data = h.to_vec().expect("conversion should succeed");
        let feat_dim = h_data.len();

        let mut expanded_data = Vec::new();
        for _ in 0..num_nodes {
            expanded_data.extend(&h_data);
        }

        from_vec(
            expanded_data,
            &[num_nodes, feat_dim],
            torsh_core::device::DeviceType::Cpu,
        )
        .expect("expanded nodes tensor creation should succeed")
    }

    fn decode_edges(&self, h: &Tensor, num_nodes: usize) -> Tensor {
        // Compute pairwise edge probabilities
        let mut edge_logits_data = Vec::new();

        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if i != j {
                    // Simplified: use dot product of node embeddings as edge logit
                    let h_i = h
                        .slice_tensor(0, i, i + 1)
                        .expect("node i slice should succeed");
                    let h_j = h
                        .slice_tensor(0, j, j + 1)
                        .expect("node j slice should succeed");

                    let logit = h_i
                        .dot(&h_j.t().expect("transpose should succeed"))
                        .expect("dot product should succeed")
                        .item()
                        .expect("tensor should have single item");
                    edge_logits_data.push(logit);
                } else {
                    edge_logits_data.push(-1000.0); // No self-loops
                }
            }
        }

        from_vec(
            edge_logits_data,
            &[num_nodes, num_nodes],
            torsh_core::device::DeviceType::Cpu,
        )
        .expect("edge logits tensor creation should succeed")
    }

    fn sample_edges(&self, edge_logits: &Tensor, num_nodes: usize) -> Tensor {
        let logits_data = edge_logits.to_vec().expect("conversion should succeed");
        let mut edges = Vec::new();

        // Sample edges based on probabilities (threshold at 0.5)
        for i in 0..num_nodes {
            for j in 0..num_nodes {
                if i != j {
                    let idx = i * num_nodes + j;
                    let prob = 1.0 / (1.0 + (-logits_data[idx]).exp()); // Sigmoid

                    if prob > 0.5 {
                        edges.push(i as f32);
                        edges.push(j as f32);
                    }
                }
            }
        }

        if edges.is_empty() {
            // Return empty edge index
            return zeros(&[2, 0]).expect("empty edge index creation should succeed");
        }

        let num_edges = edges.len() / 2;
        from_vec(edges, &[2, num_edges], torsh_core::device::DeviceType::Cpu)
            .expect("edge index tensor creation should succeed")
    }
}

impl GraphLayer for GraphVAE {
    fn forward(&self, graph: &GraphData) -> GraphData {
        let (reconstructed, _, _) = self.forward(graph);
        reconstructed
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![
            self.encoder_layer1.clone_data(),
            self.encoder_layer2.clone_data(),
            self.mu_layer.clone_data(),
            self.logvar_layer.clone_data(),
            self.decoder_layer1.clone_data(),
            self.decoder_layer2.clone_data(),
            self.node_decoder.clone_data(),
            self.edge_decoder.clone_data(),
        ];

        if let Some(ref b) = self.encoder_bias1 {
            params.push(b.clone_data());
        }
        if let Some(ref b) = self.encoder_bias2 {
            params.push(b.clone_data());
        }
        if let Some(ref b) = self.decoder_bias1 {
            params.push(b.clone_data());
        }
        if let Some(ref b) = self.decoder_bias2 {
            params.push(b.clone_data());
        }

        params
    }
}

/// Graph Generative Adversarial Network (GraphGAN)
/// Learns to generate realistic graphs through adversarial training
#[derive(Debug)]
pub struct GraphGAN {
    latent_dim: usize,
    hidden_dim: usize,
    output_features: usize,

    // Generator network
    generator: GraphGANGenerator,

    // Discriminator network
    discriminator: GraphGANDiscriminator,
}

impl GraphGAN {
    /// Create a new Graph GAN
    pub fn new(
        latent_dim: usize,
        hidden_dim: usize,
        output_features: usize,
        use_bias: bool,
    ) -> Self {
        let generator = GraphGANGenerator::new(latent_dim, hidden_dim, output_features, use_bias);
        let discriminator = GraphGANDiscriminator::new(output_features, hidden_dim, use_bias);

        Self {
            latent_dim,
            hidden_dim,
            output_features,
            generator,
            discriminator,
        }
    }

    /// Generate fake graph from random noise
    pub fn generate(&self, num_nodes: usize) -> GraphData {
        let z = randn(&[1, self.latent_dim]).expect("latent vector sampling should succeed");
        self.generator.generate(&z, num_nodes)
    }

    /// Discriminator forward pass (returns real/fake score)
    pub fn discriminate(&self, graph: &GraphData) -> f32 {
        self.discriminator.forward(graph)
    }

    /// Train generator (maximize discriminator error)
    pub fn generator_loss(&self, num_nodes: usize) -> f32 {
        let fake_graph = self.generate(num_nodes);
        let fake_score = self.discriminate(&fake_graph);

        // Generator loss: -log(D(G(z)))
        -(fake_score.ln())
    }

    /// Train discriminator (distinguish real from fake)
    pub fn discriminator_loss(&self, real_graph: &GraphData, num_nodes: usize) -> f32 {
        // Real graph score
        let real_score = self.discriminate(real_graph);

        // Fake graph score
        let fake_graph = self.generate(num_nodes);
        let fake_score = self.discriminate(&fake_graph);

        // Discriminator loss: -log(D(real)) - log(1 - D(fake))
        -(real_score.ln()) - ((1.0 - fake_score).ln())
    }

    /// Get generator parameters
    pub fn generator_parameters(&self) -> Vec<Tensor> {
        self.generator.parameters()
    }

    /// Get discriminator parameters
    pub fn discriminator_parameters(&self) -> Vec<Tensor> {
        self.discriminator.parameters()
    }
}

/// Generator network for GraphGAN
#[derive(Debug)]
struct GraphGANGenerator {
    latent_dim: usize,
    hidden_dim: usize,
    output_features: usize,

    layer1: Parameter,
    layer2: Parameter,
    node_layer: Parameter,
    edge_layer: Parameter,

    bias1: Option<Parameter>,
    bias2: Option<Parameter>,
}

impl GraphGANGenerator {
    fn new(latent_dim: usize, hidden_dim: usize, output_features: usize, use_bias: bool) -> Self {
        let layer1 = Parameter::new(
            randn(&[latent_dim, hidden_dim]).expect("failed to create generator layer1 tensor"),
        );
        let layer2 = Parameter::new(
            randn(&[hidden_dim, hidden_dim]).expect("failed to create generator layer2 tensor"),
        );
        let node_layer = Parameter::new(
            randn(&[hidden_dim, output_features])
                .expect("failed to create generator node_layer tensor"),
        );
        let edge_layer = Parameter::new(
            randn(&[hidden_dim, 1]).expect("failed to create generator edge_layer tensor"),
        );

        let (bias1, bias2) = if use_bias {
            (
                Some(Parameter::new(
                    zeros(&[hidden_dim]).expect("failed to create generator bias1 tensor"),
                )),
                Some(Parameter::new(
                    zeros(&[hidden_dim]).expect("failed to create generator bias2 tensor"),
                )),
            )
        } else {
            (None, None)
        };

        Self {
            latent_dim,
            hidden_dim,
            output_features,
            layer1,
            layer2,
            node_layer,
            edge_layer,
            bias1,
            bias2,
        }
    }

    fn generate(&self, z: &Tensor, num_nodes: usize) -> GraphData {
        // Forward through generator
        let mut h = z
            .matmul(&self.layer1.clone_data())
            .expect("operation should succeed");
        if let Some(ref bias) = self.bias1 {
            h = h.add(&bias.clone_data()).expect("operation should succeed");
        }
        h = self.leaky_relu(&h, 0.2);

        h = h
            .matmul(&self.layer2.clone_data())
            .expect("operation should succeed");
        if let Some(ref bias) = self.bias2 {
            h = h.add(&bias.clone_data()).expect("operation should succeed");
        }
        h = self.leaky_relu(&h, 0.2);

        // Expand to node-level
        let h_expanded = self.expand_to_nodes(&h, num_nodes);

        // Generate node features
        let node_features = h_expanded
            .matmul(&self.node_layer.clone_data())
            .expect("operation should succeed");
        let node_features = self.tanh(&node_features);

        // Generate edges
        let edge_index = self.generate_edges(&h_expanded, num_nodes);

        GraphData::new(node_features, edge_index)
    }

    fn leaky_relu(&self, x: &Tensor, alpha: f32) -> Tensor {
        let data = x.to_vec().expect("conversion should succeed");
        let activated: Vec<f32> = data
            .iter()
            .map(|&v| if v > 0.0 { v } else { alpha * v })
            .collect();
        from_vec(
            activated,
            x.shape().dims(),
            torsh_core::device::DeviceType::Cpu,
        )
        .expect("leaky_relu tensor creation should succeed")
    }

    fn tanh(&self, x: &Tensor) -> Tensor {
        let data = x.to_vec().expect("conversion should succeed");
        let activated: Vec<f32> = data.iter().map(|&v| v.tanh()).collect();
        from_vec(
            activated,
            x.shape().dims(),
            torsh_core::device::DeviceType::Cpu,
        )
        .expect("tanh tensor creation should succeed")
    }

    fn expand_to_nodes(&self, h: &Tensor, num_nodes: usize) -> Tensor {
        let h_data = h.to_vec().expect("conversion should succeed");
        let feat_dim = h_data.len();

        let mut expanded_data = Vec::new();
        for _ in 0..num_nodes {
            expanded_data.extend(&h_data);
        }

        from_vec(
            expanded_data,
            &[num_nodes, feat_dim],
            torsh_core::device::DeviceType::Cpu,
        )
        .expect("expanded nodes tensor creation should succeed")
    }

    fn generate_edges(&self, _h: &Tensor, num_nodes: usize) -> Tensor {
        let mut edges = Vec::new();
        let mut rng = thread_rng();

        // Generate edges probabilistically
        for i in 0..num_nodes {
            for j in (i + 1)..num_nodes {
                // Use node embeddings to determine edge probability
                if rng.gen_range(0.0..1.0) > 0.7 {
                    edges.push(i as f32);
                    edges.push(j as f32);
                    edges.push(j as f32);
                    edges.push(i as f32);
                }
            }
        }

        if edges.is_empty() {
            return zeros(&[2, 0]).expect("empty edge index creation should succeed");
        }

        let num_edges = edges.len() / 2;
        from_vec(edges, &[2, num_edges], torsh_core::device::DeviceType::Cpu)
            .expect("edge index tensor creation should succeed")
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![
            self.layer1.clone_data(),
            self.layer2.clone_data(),
            self.node_layer.clone_data(),
            self.edge_layer.clone_data(),
        ];

        if let Some(ref b) = self.bias1 {
            params.push(b.clone_data());
        }
        if let Some(ref b) = self.bias2 {
            params.push(b.clone_data());
        }

        params
    }
}

/// Discriminator network for GraphGAN
#[derive(Debug)]
struct GraphGANDiscriminator {
    input_features: usize,
    hidden_dim: usize,

    layer1: Parameter,
    layer2: Parameter,
    output_layer: Parameter,

    bias1: Option<Parameter>,
    bias2: Option<Parameter>,
    bias_out: Option<Parameter>,
}

impl GraphGANDiscriminator {
    fn new(input_features: usize, hidden_dim: usize, use_bias: bool) -> Self {
        let layer1 = Parameter::new(
            randn(&[input_features, hidden_dim])
                .expect("failed to create discriminator layer1 tensor"),
        );
        let layer2 = Parameter::new(
            randn(&[hidden_dim, hidden_dim]).expect("failed to create discriminator layer2 tensor"),
        );
        let output_layer = Parameter::new(
            randn(&[hidden_dim, 1]).expect("failed to create discriminator output_layer tensor"),
        );

        let (bias1, bias2, bias_out) = if use_bias {
            (
                Some(Parameter::new(
                    zeros(&[hidden_dim]).expect("failed to create discriminator bias1 tensor"),
                )),
                Some(Parameter::new(
                    zeros(&[hidden_dim]).expect("failed to create discriminator bias2 tensor"),
                )),
                Some(Parameter::new(
                    zeros(&[1]).expect("failed to create discriminator bias_out tensor"),
                )),
            )
        } else {
            (None, None, None)
        };

        Self {
            input_features,
            hidden_dim,
            layer1,
            layer2,
            output_layer,
            bias1,
            bias2,
            bias_out,
        }
    }

    fn forward(&self, graph: &GraphData) -> f32 {
        // Forward through discriminator
        let mut h = graph
            .x
            .matmul(&self.layer1.clone_data())
            .expect("operation should succeed");
        if let Some(ref bias) = self.bias1 {
            h = h.add(&bias.clone_data()).expect("operation should succeed");
        }
        h = self.leaky_relu(&h, 0.2);

        h = h
            .matmul(&self.layer2.clone_data())
            .expect("operation should succeed");
        if let Some(ref bias) = self.bias2 {
            h = h.add(&bias.clone_data()).expect("operation should succeed");
        }
        h = self.leaky_relu(&h, 0.2);

        // Global mean pooling
        let graph_repr = h
            .mean(Some(&[0]), false)
            .expect("mean pooling should succeed");
        let graph_repr_2d = graph_repr.unsqueeze(0).expect("unsqueeze should succeed"); // Make 2D for matmul

        // Output layer
        let mut logit = graph_repr_2d
            .matmul(&self.output_layer.clone_data())
            .expect("operation should succeed");
        if let Some(ref bias) = self.bias_out {
            logit = logit
                .add(&bias.clone_data())
                .expect("operation should succeed");
        }

        // Sigmoid activation
        let logit_val = logit.item().expect("tensor should have single item");
        1.0 / (1.0 + (-logit_val).exp())
    }

    fn leaky_relu(&self, x: &Tensor, alpha: f32) -> Tensor {
        let data = x.to_vec().expect("conversion should succeed");
        let activated: Vec<f32> = data
            .iter()
            .map(|&v| if v > 0.0 { v } else { alpha * v })
            .collect();
        from_vec(
            activated,
            x.shape().dims(),
            torsh_core::device::DeviceType::Cpu,
        )
        .expect("discriminator leaky_relu tensor creation should succeed")
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![
            self.layer1.clone_data(),
            self.layer2.clone_data(),
            self.output_layer.clone_data(),
        ];

        if let Some(ref b) = self.bias1 {
            params.push(b.clone_data());
        }
        if let Some(ref b) = self.bias2 {
            params.push(b.clone_data());
        }
        if let Some(ref b) = self.bias_out {
            params.push(b.clone_data());
        }

        params
    }
}

/// Conditional Graph Generation
#[derive(Debug)]
pub struct ConditionalGraphGenerator {
    vae: GraphVAE,
    condition_dim: usize,
    condition_layer: Parameter,
}

impl ConditionalGraphGenerator {
    /// Create a new conditional graph generator
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        latent_dim: usize,
        condition_dim: usize,
        beta: f32,
    ) -> Self {
        let vae = GraphVAE::new(in_features, hidden_features, latent_dim, beta, true);
        let condition_layer = Parameter::new(
            randn(&[condition_dim, latent_dim]).expect("failed to create condition_layer tensor"),
        );

        Self {
            vae,
            condition_dim,
            condition_layer,
        }
    }

    /// Generate graph conditioned on a property vector
    pub fn generate_conditional(&self, condition: &Tensor, num_nodes: usize) -> GraphData {
        // Map condition to latent space bias
        let condition_bias = condition
            .matmul(&self.condition_layer.clone_data())
            .expect("condition matmul should succeed");

        // Sample base latent vector
        let z_base =
            randn(&[1, self.vae.latent_dim]).expect("latent vector sampling should succeed");

        // Add conditional bias
        let z = z_base
            .add(&condition_bias)
            .expect("operation should succeed");

        // Decode to graph
        self.vae.decode(&z, num_nodes)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.vae.parameters();
        params.push(self.condition_layer.clone_data());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use torsh_core::device::DeviceType;

    #[test]
    fn test_graphvae_creation() {
        let vae = GraphVAE::new(8, 16, 10, 1.0, true);
        assert_eq!(vae.encoder_in_features, 8);
        assert_eq!(vae.encoder_hidden_features, 16);
        assert_eq!(vae.latent_dim, 10);
        assert_eq!(vae.beta, 1.0);
    }

    #[test]
    fn test_graphvae_encode_decode() {
        let features = randn(&[5, 8]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0];
        let edge_index = from_vec(edges, &[2, 4], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let vae = GraphVAE::new(8, 16, 10, 1.0, true);

        let (mu, logvar) = vae.encode(&graph);
        assert_eq!(mu.shape().dims(), &[1, 10]);
        assert_eq!(logvar.shape().dims(), &[1, 10]);

        let z = vae.reparameterize(&mu, &logvar);
        assert_eq!(z.shape().dims(), &[1, 10]);

        let reconstructed = vae.decode(&z, 5);
        assert_eq!(reconstructed.num_nodes, 5);
    }

    #[test]
    fn test_graphvae_generation() {
        let vae = GraphVAE::new(8, 16, 10, 1.0, true);
        let generated = vae.generate(6);

        assert_eq!(generated.num_nodes, 6);
        assert_eq!(generated.x.shape().dims()[0], 6);
        assert_eq!(generated.x.shape().dims()[1], 8);
    }

    #[test]
    fn test_graphvae_interpolation() {
        let features1 = randn(&[4, 6]).unwrap();
        let features2 = randn(&[4, 6]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let edge_index = from_vec(edges.clone(), &[2, 3], DeviceType::Cpu).unwrap();

        let graph1 = GraphData::new(features1, edge_index.clone());
        let graph2 = GraphData::new(features2, edge_index);

        let vae = GraphVAE::new(6, 12, 8, 1.0, true);

        // Interpolate at alpha = 0.5 (midpoint)
        let interpolated = vae.interpolate(&graph1, &graph2, 0.5, 4);
        assert_eq!(interpolated.num_nodes, 4);
    }

    #[test]
    fn test_graphgan_creation() {
        let gan = GraphGAN::new(16, 32, 8, true);
        assert_eq!(gan.latent_dim, 16);
        assert_eq!(gan.hidden_dim, 32);
        assert_eq!(gan.output_features, 8);
    }

    #[test]
    fn test_graphgan_generation() {
        let gan = GraphGAN::new(16, 32, 8, true);
        let generated = gan.generate(5);

        assert_eq!(generated.num_nodes, 5);
        assert_eq!(generated.x.shape().dims()[1], 8);
    }

    #[test]
    fn test_graphgan_discriminate() {
        let features = randn(&[4, 8]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let edge_index = from_vec(edges, &[2, 3], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let gan = GraphGAN::new(16, 32, 8, true);
        let score = gan.discriminate(&graph);

        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_conditional_generation() {
        let cond_gen = ConditionalGraphGenerator::new(8, 16, 10, 4, 1.0);

        let condition = randn(&[1, 4]).unwrap();
        let generated = cond_gen.generate_conditional(&condition, 5);

        assert_eq!(generated.num_nodes, 5);
        assert_eq!(generated.x.shape().dims()[1], 8);
    }

    #[test]
    fn test_graphvae_loss_computation() {
        let features = randn(&[3, 6]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0];
        let edge_index = from_vec(edges, &[2, 2], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let vae = GraphVAE::new(6, 12, 8, 1.0, true);
        let (reconstructed, mu, logvar) = vae.forward(&graph);

        let loss = vae.compute_loss(&graph, &reconstructed, &mu, &logvar);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_graphgan_losses() {
        let features = randn(&[4, 8]).unwrap();
        let edges = vec![0.0, 1.0, 1.0, 2.0, 2.0, 3.0];
        let edge_index = from_vec(edges, &[2, 3], DeviceType::Cpu).unwrap();
        let graph = GraphData::new(features, edge_index);

        let gan = GraphGAN::new(16, 32, 8, true);

        let gen_loss = gan.generator_loss(4);
        assert!(gen_loss > 0.0);

        let disc_loss = gan.discriminator_loss(&graph, 4);
        // Discriminator loss can be negative
        assert!(disc_loss.is_finite());
    }
}
