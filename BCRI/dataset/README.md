# Datasets

The experiments use five citation networks. Nodes are papers, edges are citations, and labels are research categories. Only numerical features and graph connectivity are model inputs.

| Dataset | Source | Nodes | Stored edge entries | Features | Classes |
|---|---|---:|---:|---:|---:|
| Cora | PyG `Planetoid(name="Cora")` | 2708 | 10556 | 1433 | 7 |
| CiteSeer | PyG `Planetoid(name="CiteSeer")` | 3327 | 9104 | 3703 | 6 |
| CoraFull | PyG `CoraFull()` | 19793 | 126842 | 8710 | 70 |
| Cora-ML | PyG `CitationFull(name="Cora_ML")` | 2995 | 16316 | 2879 | 7 |
| ogbn-arxiv | OGB `PygNodePropPredDataset(name="ogbn-arxiv")` | 169343 | 1166243 | 128 | 40 |

The four PyG graphs store both directions of undirected edges. The ogbn-arxiv count describes its original directed graph. BCRI symmetrizes connections when constructing diffusion transitions.

## Download and prepare

From the project root, run:

```bash
python dataset/prepare_data.py
```

To prepare selected datasets:

```bash
python dataset/prepare_data.py --dataset Cora CiteSeer
python dataset/prepare_data.py --dataset ogbn-arxiv
```

The script downloads through the public PyG/OGB loaders, exports only `x`, `edge_index`, and `y`, and verifies tensor fingerprints embedded in the script. No separate metadata file is required. Existing files are checked rather than overwritten; `--overwrite` explicitly replaces them. Downloads are cached in `.downloads/`.

Expected layout:

```text
dataset/
  README.md
  prepare_data.py
  Cora/Cora.pt
  CiteSeer/CiteSeer.pt
  CoraFull/CoraFull.pt
  Cora_ML/Cora-ML.pt
  ogbn-arxiv/ogbn-arxiv.pt
```

## Offline preparation

If the public host is unreachable, supply an existing trusted PyG Data file or tensor dictionary:

```bash
python dataset/prepare_data.py --dataset CiteSeer --source-file path/to/input.pt
```

Verify already prepared graphs without downloading:

```bash
python dataset/prepare_data.py --verify-only
```

Fingerprint checks cover tensor shapes, dtypes, graph contents and node ordering. Different CiteSeer variants or differently ordered nodes are not silently substituted.

## Sources

- [Planetoid: Cora and CiteSeer](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.Planetoid.html)
- [CoraFull](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.CoraFull.html)
- [CitationFull: Cora-ML](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.CitationFull.html)
- [Open Graph Benchmark: ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/)

Dataset licenses and terms remain those of their original providers. Original public masks and the OGB temporal split are not used as BCRI's few-shot class partitions.
