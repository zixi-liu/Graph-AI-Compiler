# Graph-AI-Compiler
Project from Google Research - Predict how fast an AI model runs.

An AI model can be represented as a graph, where a node is a tensor operation (e.g. matrix multiplication, convolution, etc), and an edge represents a tensor. A compilation configuration controls how the compiler transforms the graph for a specific optimization pass. In particular, Alice can control two types of configurations/optimizations:

A layout configuration control how tensors in the graph are laid out in the physical memory, by specifying the dimension order of each input and output of an operation node.
A tile configuration controls the tile size of each fused subgraph.

## Related Topics
[**What is a TPU**](https://jax.readthedocs.io/en/latest/pallas/tpu.html#what-is-a-tpu)
