# torsh-graph

Graph neural network components for ToRSh - powered by SciRS2.

## Overview

This crate provides comprehensive graph neural network (GNN) implementations with a PyTorch Geometric-compatible API. It leverages `scirs2-graph` for high-performance graph operations while maintaining full integration with ToRSh's autograd system and neural network modules.

## Features

- **Graph Representations**: Adjacency matrices, edge lists, COO/CSR formats
- **Message Passing Layers**: GCN, GAT, GraphSAGE, GIN, EdgeConv
- **Pooling Operations**: Global pooling, TopK pooling, SAGPool, DiffPool
- **Graph Convolutions**: Spectral and spatial convolutions
- **Attention Mechanisms**: Graph attention, multi-head attention, transformer layers
- **Graph Generation**: Erdős-Rényi, Barabási-Albert, Watts-Strogatz
- **Graph Utilities**: Subgraph sampling, neighborhood aggregation, batching
- **Heterogeneous Graphs**: Support for multiple node/edge types
- **Temporal Graphs**: Dynamic graph neural networks
- **Explainability**: GNNExplainer, attention visualization

## Usage

### Basic Graph Construction

```rust
use torsh_graph::prelude::*;
use torsh_tensor::prelude::*;

// Create a simple graph with 5 nodes
let num_nodes = 5;

// Define edges as pairs of node indices
let edge_index = tensor![[0, 1, 1, 2, 2, 3, 3, 4],  // source nodes
                         [1, 0, 2, 1, 3, 2, 4, 3]]; // target nodes

// Node features (5 nodes, 3 features each)
let x = randn(&[5, 3])?;

// Create graph
let graph = Graph::new(edge_index, Some(x))?;

println!("Number of nodes: {}", graph.num_nodes());
println!("Number of edges: {}", graph.num_edges());
println!("Is directed: {}", graph.is_directed());
```

### Graph Neural Network Layers

#### Graph Convolutional Network (GCN)

```rust
use torsh_graph::nn::*;
use torsh_nn::prelude::*;

// Single GCN layer
let gcn_layer = GCNConv::new(
    in_channels: 16,
    out_channels: 32,
    improved: false,
    cached: true,
    add_self_loops: true,
    normalize: true,
    bias: true,
)?;

// Forward pass
let x = randn(&[num_nodes, 16])?;
let edge_index = graph.edge_index();
let output = gcn_layer.forward(&x, &edge_index, None)?;

// Multi-layer GCN
struct GCN {
    conv1: GCNConv,
    conv2: GCNConv,
    conv3: GCNConv,
}

impl GCN {
    fn new(in_channels: usize, hidden_channels: usize, out_channels: usize) -> Self {
        Self {
            conv1: GCNConv::new(in_channels, hidden_channels, false, true, true, true, true).unwrap(),
            conv2: GCNConv::new(hidden_channels, hidden_channels, false, true, true, true, true).unwrap(),
            conv3: GCNConv::new(hidden_channels, out_channels, false, true, true, true, true).unwrap(),
        }
    }
}

impl Module for GCN {
    fn forward(&self, x: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x, edge_index, None)?;
        let x = F::relu(&x);
        let x = F::dropout(&x, 0.5, self.training)?;

        let x = self.conv2.forward(&x, edge_index, None)?;
        let x = F::relu(&x);
        let x = F::dropout(&x, 0.5, self.training)?;

        let x = self.conv3.forward(&x, edge_index, None)?;
        Ok(x)
    }
}
```

#### Graph Attention Network (GAT)

```rust
use torsh_graph::nn::*;

// GAT layer with multi-head attention
let gat_layer = GATConv::new(
    in_channels: 16,
    out_channels: 32,
    heads: 8,              // Number of attention heads
    concat: true,          // Concatenate or average attention heads
    negative_slope: 0.2,   // LeakyReLU negative slope
    dropout: 0.6,
    add_self_loops: true,
    bias: true,
)?;

let output = gat_layer.forward(&x, &edge_index, None, None)?;
println!("Output shape: {:?}", output.shape());  // [num_nodes, 32 * 8] if concat=true

// Multi-layer GAT
struct GAT {
    conv1: GATConv,
    conv2: GATConv,
}

impl Module for GAT {
    fn forward(&self, x: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x, edge_index, None, None)?;
        let x = F::elu(&x);
        let x = F::dropout(&x, 0.6, self.training)?;

        let x = self.conv2.forward(&x, edge_index, None, None)?;
        Ok(x)
    }
}
```

#### GraphSAGE

```rust
use torsh_graph::nn::*;

// GraphSAGE layer with different aggregation methods
let sage_mean = SAGEConv::new(
    in_channels: 16,
    out_channels: 32,
    aggr: "mean",  // Options: "mean", "max", "sum", "lstm"
    normalize: true,
    root_weight: true,
    bias: true,
)?;

let output = sage_mean.forward(&x, &edge_index)?;

// Using max aggregation
let sage_max = SAGEConv::new(16, 32, "max", true, true, true)?;

// Using LSTM aggregation
let sage_lstm = SAGEConv::new(16, 32, "lstm", true, true, true)?;
```

#### Graph Isomorphism Network (GIN)

```rust
use torsh_graph::nn::*;

// GIN layer with MLP
let mlp = Sequential::new()
    .add(Linear::new(16, 64, true))
    .add(ReLU::new(false))
    .add(Linear::new(64, 32, true));

let gin_layer = GINConv::new(
    nn: mlp,
    eps: 0.0,      // Initial epsilon for weighting self-loops
    train_eps: false,
)?;

let output = gin_layer.forward(&x, &edge_index)?;
```

#### EdgeConv (Dynamic Graph CNN)

```rust
use torsh_graph::nn::*;

// EdgeConv for point cloud processing
let edge_conv = EdgeConv::new(
    nn: Linear::new(32, 64, true),  // MLP applied to edge features
    aggr: "max",
)?;

let output = edge_conv.forward(&x, &edge_index)?;
```

### Graph Pooling

#### Global Pooling

```rust
use torsh_graph::nn::pooling::*;

// Global mean pooling
let global_mean = global_mean_pool(&x, &batch)?;

// Global max pooling
let global_max = global_max_pool(&x, &batch)?;

// Global sum pooling
let global_sum = global_add_pool(&x, &batch)?;

// Global attention pooling
let global_attn = GlobalAttention::new(
    gate_nn: Linear::new(64, 1, true),
    nn: Some(Linear::new(64, 64, true)),
)?;
let output = global_attn.forward(&x, &batch)?;
```

#### Hierarchical Pooling

```rust
use torsh_graph::nn::pooling::*;

// TopK pooling
let topk_pool = TopKPooling::new(
    in_channels: 64,
    ratio: 0.5,        // Keep top 50% of nodes
    min_score: None,
    multiplier: 1.0,
)?;

let (x_pooled, edge_index_pooled, batch_pooled, perm, score) =
    topk_pool.forward(&x, &edge_index, None, &batch, None)?;

// SAGPool (Self-Attention Graph Pooling)
let sag_pool = SAGPooling::new(
    in_channels: 64,
    ratio: 0.5,
    gnn: GCNConv::new(64, 1, false, false, true, true, false)?,
    min_score: None,
    multiplier: 1.0,
)?;

// DiffPool (Differentiable Pooling)
let diff_pool = DiffPooling::new(
    in_channels: 64,
    hidden_channels: 64,
    num_clusters: 10,
)?;

let (x_pooled, adj_pooled, link_loss, ent_loss) =
    diff_pool.forward(&x, &adj, &mask)?;
```

### Heterogeneous Graphs

```rust
use torsh_graph::hetero::*;

// Create heterogeneous graph with different node types
let hetero_graph = HeteroGraph::new();

// Add node types
hetero_graph.add_node_type("user", user_features)?;
hetero_graph.add_node_type("item", item_features)?;

// Add edge types with relations
hetero_graph.add_edge_type(
    ("user", "rates", "item"),
    user_item_edges,
    Some(edge_features),
)?;

hetero_graph.add_edge_type(
    ("user", "follows", "user"),
    user_user_edges,
    None,
)?;

// Heterogeneous GNN layer
let hetero_conv = HeteroConv::new()
    .add_conv(("user", "rates", "item"), GCNConv::new(64, 64, false, false, true, true, true)?)
    .add_conv(("user", "follows", "user"), GATConv::new(64, 64, 4, true, 0.2, 0.0, true, true)?);

let output = hetero_conv.forward(&hetero_graph)?;
```

### Temporal Graphs

```rust
use torsh_graph::temporal::*;

// Temporal graph with snapshots
let temporal_graph = TemporalGraph::new(num_snapshots: 10);

for t in 0..10 {
    temporal_graph.add_snapshot(t, edge_index_t, node_features_t)?;
}

// Temporal GNN
let tgnn = TGCN::new(
    in_channels: 16,
    hidden_channels: 32,
    num_layers: 2,
)?;

// Process temporal sequence
let outputs = tgnn.forward(&temporal_graph)?;

// Dynamic edge RNN
let edge_rnn = DynamicEdgeRNN::new(
    node_features: 16,
    edge_features: 8,
    hidden_size: 32,
)?;
```

### Graph Generation

```rust
use torsh_graph::generation::*;

// Erdős-Rényi random graph
let er_graph = erdos_renyi_graph(num_nodes: 100, p: 0.1, directed: false)?;

// Barabási-Albert preferential attachment
let ba_graph = barabasi_albert_graph(num_nodes: 100, m: 3)?;

// Watts-Strogatz small-world
let ws_graph = watts_strogatz_graph(num_nodes: 100, k: 4, p: 0.3)?;

// Stochastic block model
let sbm_graph = stochastic_block_model(
    block_sizes: &[30, 30, 40],
    p_matrix: &[[0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8]],
)?;
```

### Graph Utilities

#### Batching

```rust
use torsh_graph::data::*;

// Create a batch from multiple graphs
let graphs = vec![graph1, graph2, graph3];
let batch = Batch::from_data_list(&graphs)?;

// Batch contains:
// - batch.x: Concatenated node features
// - batch.edge_index: Concatenated edges with offset indices
// - batch.batch: Tensor indicating which graph each node belongs to

// Unbatch
let graphs_recovered = batch.to_data_list()?;
```

#### Neighborhood Sampling

```rust
use torsh_graph::sampling::*;

// Sample k-hop neighborhood
let subgraph = k_hop_subgraph(
    node_idx: &[5, 10, 15],  // Center nodes
    num_hops: 2,
    edge_index: &edge_index,
    relabel_nodes: true,
)?;

// Random walk sampling
let walks = random_walk(
    start: &[0, 1, 2],
    edge_index: &edge_index,
    walk_length: 10,
)?;

// Neighbor sampling (for GraphSAGE)
let neighbor_sampler = NeighborSampler::new(
    edge_index: edge_index.clone(),
    sizes: vec![25, 10],  // Sample 25 neighbors in 1st hop, 10 in 2nd hop
    node_idx: None,
    num_nodes: Some(num_nodes),
    batch_size: 128,
)?;

for batch in neighbor_sampler {
    // Train on sampled subgraph
}
```

### Graph Classification

```rust
use torsh_graph::prelude::*;

struct GraphClassifier {
    conv1: GCNConv,
    conv2: GCNConv,
    conv3: GCNConv,
    lin: Linear,
}

impl GraphClassifier {
    fn new(num_features: usize, num_classes: usize) -> Self {
        Self {
            conv1: GCNConv::new(num_features, 64, false, false, true, true, true).unwrap(),
            conv2: GCNConv::new(64, 64, false, false, true, true, true).unwrap(),
            conv3: GCNConv::new(64, 64, false, false, true, true, true).unwrap(),
            lin: Linear::new(64, num_classes, true),
        }
    }
}

impl Module for GraphClassifier {
    fn forward(&self, data: &Batch) -> Result<Tensor> {
        let x = &data.x;
        let edge_index = &data.edge_index;
        let batch = &data.batch;

        // Graph convolutions
        let x = self.conv1.forward(x, edge_index, None)?;
        let x = F::relu(&x);

        let x = self.conv2.forward(&x, edge_index, None)?;
        let x = F::relu(&x);

        let x = self.conv3.forward(&x, edge_index, None)?;

        // Global pooling
        let x = global_mean_pool(&x, batch)?;

        // Classification head
        let x = F::dropout(&x, 0.5, self.training)?;
        let x = self.lin.forward(&x)?;

        Ok(x)
    }
}
```

### Node Classification

```rust
use torsh_graph::prelude::*;

// Load citation network (e.g., Cora, CiteSeer, PubMed)
let dataset = Planetoid::new("./data", "Cora")?;
let data = dataset.get(0)?;

// Split into train/val/test
let train_mask = &data.train_mask;
let val_mask = &data.val_mask;
let test_mask = &data.test_mask;

// Define model
let model = GCN::new(
    dataset.num_features(),
    64,
    dataset.num_classes(),
);

// Training loop
for epoch in 0..200 {
    model.train();
    optimizer.zero_grad();

    let out = model.forward(&data.x, &data.edge_index)?;
    let loss = F::cross_entropy(&out.index_select(0, train_mask)?, &data.y.index_select(0, train_mask)?, None, "mean", None)?;

    loss.backward()?;
    optimizer.step();
}
```

### Explainability

```rust
use torsh_graph::explainability::*;

// GNNExplainer for model interpretation
let explainer = GNNExplainer::new(
    model: &model,
    epochs: 200,
    lr: 0.01,
    return_type: "log_prob",
)?;

// Explain prediction for a specific node
let (node_feat_mask, edge_mask) = explainer.explain_node(
    node_idx: 10,
    x: &data.x,
    edge_index: &data.edge_index,
)?;

// Visualize important edges
visualize_subgraph(
    node_idx: 10,
    edge_index: &data.edge_index,
    edge_mask: &edge_mask,
    threshold: 0.5,
)?;

// Attention weights visualization for GAT
let attention_weights = gat_layer.get_attention_weights()?;
plot_attention_graph(&data.edge_index, &attention_weights)?;
```

## Advanced Examples

### Link Prediction

```rust
use torsh_graph::prelude::*;

// Negative sampling for link prediction
let neg_edge_index = negative_sampling(
    edge_index: &data.edge_index,
    num_nodes: data.num_nodes(),
    num_neg_samples: Some(data.num_edges()),
)?;

// Model for link prediction
struct LinkPredictor {
    encoder: GCN,
}

impl LinkPredictor {
    fn decode(&self, z: &Tensor, edge_index: &Tensor) -> Result<Tensor> {
        let row = edge_index.select(0, 0)?;
        let col = edge_index.select(0, 1)?;

        let src = z.index_select(0, &row)?;
        let dst = z.index_select(0, &col)?;

        // Dot product decoder
        let scores = (src * dst).sum(-1)?;
        Ok(scores)
    }

    fn forward(&self, x: &Tensor, edge_index: &Tensor, pos_edge_index: &Tensor, neg_edge_index: &Tensor) -> Result<Tensor> {
        let z = self.encoder.forward(x, edge_index)?;

        let pos_score = self.decode(&z, pos_edge_index)?;
        let neg_score = self.decode(&z, neg_edge_index)?;

        // Binary cross-entropy loss
        let pos_loss = F::binary_cross_entropy_with_logits(&pos_score, &ones_like(&pos_score)?, None, None, "mean")?;
        let neg_loss = F::binary_cross_entropy_with_logits(&neg_score, &zeros_like(&neg_score)?, None, None, "mean")?;

        Ok(pos_loss + neg_loss)
    }
}
```

## Integration with SciRS2

This crate leverages the SciRS2 ecosystem for:

- Graph algorithms and data structures through `scirs2-graph`
- Sparse matrix operations via `scirs2-core`
- Spatial indexing through `scirs2-spatial`
- Automatic differentiation via `scirs2-autograd`

All implementations follow the [SciRS2 POLICY](https://github.com/cool-japan/scirs/blob/master/SCIRS2_POLICY.md) for optimal performance and maintainability.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](../../LICENSE-APACHE))
- MIT license ([LICENSE-MIT](../../LICENSE-MIT))

at your option.
