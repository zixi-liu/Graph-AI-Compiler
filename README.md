# Graph-AI-Compiler
Project from Google Research - Predict how fast an AI model runs.

An AI model can be represented as a graph, where a node is a tensor operation (e.g. matrix multiplication, convolution, etc), and an edge represents a tensor. A compilation configuration controls how the compiler transforms the graph for a specific optimization pass.

## Problen Statement 

We have very large graph as input. How to design learning model and algorithm that can fit into gpu memory? 

## Data Description

#### Tile Files

Suppose a graph (representing a kernel) with n nodes and m edges. In addition, suppose we compile the graph with c different configurations, and run each on a TPU.
- Key "node_feat": contains float32 matrix with shape (n, 140).
- Key "node_opcode" contains int32 vector with shape (n, ). 
- Key "edge_index" contains int32 matrix with shape (m, 2). If entry i is = [u, v] (where 0 <= u, v < n), then there is a directed edge from node u <- v, where u is a tensor operation consuming the output tensor of v (a reverse of a typical definition of an edge direction).
- Key "config_feat" contains float32 matrix with shape (c, 24) with row j containing the (graph-level) configuration feature vector.
- Keys "config_runtime" and "config_runtime_normalizers": both are int64 vectors of length c. Entry j stores the runtime (in nanoseconds) of the given graph compiled with configuration j and a default configuration, respectively.

#### Configuration Files



## Related Topics
[**What is a TPU**](https://jax.readthedocs.io/en/latest/pallas/tpu.html#what-is-a-tpu)
