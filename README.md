# BCRI

Code for **Background-Contrasted Relational Inference for Few-Shot Node Classification**.

The main program pretrains a local encoder on unlabeled pseudo tasks, freezes it, and trains the geometry and contribution policies. At test time, network parameters remain frozen while task geometry and class prototypes are constructed.

## Installation

Use Python 3.11 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The implementation was tested with PyTorch 2.6.0, PyTorch Geometric 2.6.1, CUDA 12.4 and an NVIDIA RTX 4090. For another CUDA build, install the appropriate PyTorch wheel first.

## Prepare the datasets

```bash
python dataset/prepare_data.py
```

This prepares Cora, CiteSeer, CoraFull, Cora-ML and ogbn-arxiv using their public dataset loaders. Existing numerical files are verified and kept. Data sources, expected paths and offline preparation are described in [dataset/README.md](dataset/README.md).

## Run

Run one dataset or one setting:

```bash
python main.py --dataset CiteSeer
python main.py --key CoraFull/5way_3shot_seed_1234
```

The default protocol uses seed 1234, five fresh runs per setting, and 100 test tasks per run with ten queries per class. The seed is initialized once per setting and the random stream continues across repeats. Reported standard deviations are population standard deviations of run means.

For a single run, add `--repeats 1`. Preview settings without loading data:

```bash
python main.py --dataset CiteSeer --repeats 1
python main.py --list-configs
```

Training and evaluation progress is written to standard output as JSON. Runtime accuracies are fractions. The program keeps selected parameters and task tensors in memory; it does not save checkpoints.

## Files

| File | Purpose |
|---|---|
| `main.py` | Main entry point |
| `config.py` | Dataset, task and training settings |
| `data.py` | Graph loading, pseudo tasks and evaluation tasks |
| `encoder.py` | Local attention encoder |
| `pipeline.py` | Encoder lifecycle and task preparation |
| `geometry.py` | Diffusion, background construction and geometry policy |
| `inference.py` | Query contribution updates and prototype inference |
| `train.py` | Optimization and development model selection |

The protocol is transductive: all graph nodes may enter pseudo tasks without true labels. Development labels select models; test query labels are scored only after prediction. The ogbn-arxiv tasks use class-based few-shot partitions rather than its official temporal split.

Pseudo-task construction and parts of encoder pretraining follow or adapt [VISION](https://github.com/KEAML-JLU/VISION). 
