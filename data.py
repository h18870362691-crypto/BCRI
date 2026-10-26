"""Numeric graphs, pseudo tasks, class-disjoint evaluation and scoring."""
import random
import re
from pathlib import Path
import torch
from config import canonical_dataset, dataset_config, validate_task, EVALUATION_PROTOCOL
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree

# data
ROOT = Path(__file__).resolve().parent
_SEQUENCE = None


def continuous_running(entry=None):
    if _SEQUENCE is None:
        return False
    if entry is None:
        return True
    return (entry.get('evaluation_protocol')==EVALUATION_PROTOCOL and
            (entry['dataset'],entry['way'],entry['shot'],entry['seed'])==_SEQUENCE['group'])


def start_sequence(key):
    """Seed once, before the fixed class split and all five fresh trainings."""
    global _SEQUENCE
    e=entry(key)
    if e.get('evaluation_protocol')!=EVALUATION_PROTOCOL:
        raise ValueError('A continuous sequence requires a repeat-qualified setting key')
    random.seed(e['seed']);np.random.seed(e['seed']);torch.manual_seed(e['seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(e['seed'])
    _SEQUENCE=dict(group=(e['dataset'],e['way'],e['shot'],e['seed']),
                   repeat=0,classes=None,panels={})


def activate_repeat(key):
    e=entry(key)
    if not continuous_running(e) or e['repeat']!=_SEQUENCE['repeat']+1:
        raise ValueError('Run repeats consecutively in the same seeded process, starting at repeat 1')
    _SEQUENCE['repeat']=e['repeat']
    _SEQUENCE['panels']={}


def end_sequence():
    global _SEQUENCE
    _SEQUENCE=None


def graph_path(name):
    name=canonical_dataset(name)
    dataset_root=(ROOT/'dataset').resolve()
    path=(dataset_root/dataset_config(name)['graph']).resolve()
    if path.is_relative_to(dataset_root) and path.is_file():
        return path
    raise FileNotFoundError(f"Missing numeric graph for {name}: dataset/{dataset_config(name)['graph']}")


def load_graph(path):
    raw=torch.load(path,map_location='cpu',weights_only=False)
    graph={k:raw[k] for k in ('x','edge_index','y')}
    x=graph['x'].detach().float()
    if x.layout!=torch.strided:
        x=x.to_dense()
    edge=graph['edge_index'].long()
    y=graph['y'].long().reshape(-1)
    if x.ndim!=2 or len(x)!=len(y) or edge.ndim!=2 or edge.shape[0]!=2:
        raise ValueError('Graph x, edge_index and y are not node-aligned')
    if not torch.isfinite(x).all() or (edge.numel() and (edge.min()<0 or edge.max()>=len(x))):
        raise ValueError('Nonfinite features or invalid edge endpoints')
    return dict(x=x,edge_index=edge,y=y)


def entry(key):
    name,setting=key.split('/',1)
    name=canonical_dataset(name)
    match=re.fullmatch(r'(\d+)way_(\d+)shot_seed_(\d+)(?:_repeat_(\d+))?',setting)
    if match is None:
        raise ValueError('Setting must be DATASET/Nway_Kshot_seed_S')
    way,shot,seed=map(int,match.groups()[:3])
    repeat=int(match.group(4)) if match.group(4) else None
    profile=validate_task(name,way,shot)
    path=graph_path(name)
    return dict(key=name+'/'+setting,dataset=name,way=way,shot=shot,seed=seed,repeat=repeat,
                evaluation_protocol=EVALUATION_PROTOCOL if repeat else 'legacy-split-varying-v1',
                graph=str(path.relative_to(ROOT)),profile=profile,
                center=True,global_weight=1.0,bcpr_config=dict(temperature=.2,anchor=1.0))


def class_partition(e,truth):
    labels=sorted(set(truth.tolist()))
    counts=e['profile']['class_split']
    if len(labels)!=sum(counts):
        raise ValueError('Graph class count differs from the declared VISION split')
    if e.get('repeat') is not None and not continuous_running(e):
        raise ValueError('Initialize the single-seed sequence before constructing repeat models')
    if continuous_running(e) and _SEQUENCE['classes'] is not None:
        return {key:list(value) for key,value in _SEQUENCE['classes'].items()}
    if continuous_running(e):random.shuffle(labels)
    else:random.Random(e['seed']).shuffle(labels)
    n,d,_=counts
    result=dict(train=labels[:n],dev=labels[n:n+d],test=labels[n+d:])
    assert set(result['dev']).isdisjoint(result['test'])
    if continuous_running(e):_SEQUENCE['classes']={key:list(value) for key,value in result.items()}
    return result


def episode_ids(e,graph,panel,count):
    """Sampling uses labels only to define evaluation episodes, never model inputs."""
    if panel not in ('development','test') or count<1:
        raise ValueError('Invalid evaluation panel or episode count')
    classes=e['classes']['dev' if panel=='development' else 'test']
    way,shot,queries=e['way'],e['shot'],e['profile']['queries_per_class']
    pools={c:torch.where(graph['y']==c)[0].tolist() for c in classes}
    if any(len(v)<shot+queries for v in pools.values()):
        raise ValueError('An evaluation class has too few nodes for the requested shot/query count')
    continuous=continuous_running(e)
    if e.get('repeat') is not None and (not continuous or _SEQUENCE['repeat']!=e['repeat']):
        raise ValueError('Activate the matching repeat before sampling evaluation tasks')
    cache_key=(e['key'],panel,count)
    if continuous and cache_key in _SEQUENCE['panels']:
        return {k:v.clone() for k,v in _SEQUENCE['panels'][cache_key].items()}
    rng=random if continuous else random.Random(e['seed']+(202 if panel=='development' else 303))
    support,query=[],[]
    for _ in range(count):
        s,q=[],[]
        for c in rng.sample(classes,way):
            ids=rng.sample(pools[c],shot+queries)
            s.extend(ids[:shot]);q.extend(ids[shot:])
        support.append(s);query.append(q)
    packet=dict(support=torch.tensor(support),query=torch.tensor(query))
    if continuous:_SEQUENCE['panels'][cache_key]={k:v.clone() for k,v in packet.items()}
    return packet


def targets(e,packet,panel,graph):
    """Evaluator-only truth; never passed to any representation or policy network."""
    if panel not in ('development','test'):
        raise ValueError('Unknown evaluation split')
    y=graph['y']
    support=y[packet['support']].reshape(-1,e['way'],e['shot'])
    if not bool((support==support[:,:,:1]).all()):
        raise ValueError('Support groups contain mixed classes')
    labels=support[:,:,0]
    expected=set(e['classes']['dev' if panel=='development' else 'test'])
    if not set(labels.flatten().tolist()).issubset(expected):
        raise ValueError('Episode classes belong to a different evaluation split')
    match=y[packet['query']][:,:,None].eq(labels[:,None])
    if not bool((match.sum(-1)==1).all()):
        raise ValueError('Query class is missing or ambiguous in supports')
    return match.long().argmax(-1)


def metrics(logits,truth):
    logits,truth=logits.cpu(),truth.cpu()
    if logits.shape[:2]!=truth.shape or not torch.isfinite(logits).all():
        raise ValueError('Invalid evaluation logits or targets')
    correct=int(logits.argmax(-1).eq(truth).sum())
    ce=float(torch.nn.functional.cross_entropy(logits.double().reshape(-1,logits.shape[-1]),truth.reshape(-1)))
    return dict(correct=correct,total=truth.numel(),accuracy=correct/truth.numel(),ce=ce)

# tasks
def precompute_adaptive_task_features(data, return_components=False):
    x, edge_index = (data.x, data.edge_index)
    num_nodes = data.num_nodes
    x_raw = F.normalize(x, p=2, dim=1)
    edge_index_with_self_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index_with_self_loops
    deg_val = degree(row, num_nodes, dtype=x.dtype)
    deg_inv_sqrt = deg_val.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    A_norm = torch.sparse_coo_tensor(edge_index_with_self_loops, edge_weight, size=(num_nodes, num_nodes))
    x_smooth = torch.sparse.mm(A_norm, x_raw)
    x_smooth = F.normalize(x_smooth, p=2, dim=1)
    consistency_scores = torch.sum(x_raw * x_smooth, dim=1)
    gate = (consistency_scores.clamp(-1, 1) + 1) / 2
    x_task = (1 - gate.unsqueeze(1)) * x_raw + gate.unsqueeze(1) * x_smooth
    if return_components:
        return (x_task, x_raw, x_smooth)
    return x_task

def unsupervised_task_generator(train_idx, n_way, k_shot, m_query, features, pool_size=1000):
    device = features.device
    num_samples_per_class = k_shot + m_query
    train_idx_tensor = torch.from_numpy(train_idx)
    anchors_indices_in_train = torch.randperm(len(train_idx_tensor))[:n_way]
    selected_anchors = train_idx_tensor[anchors_indices_in_train]
    anchor_features = features[selected_anchors.to(device)]
    train_idx_set = set(train_idx)
    anchors_set = set(selected_anchors.tolist())
    potential_candidates_list = list(train_idx_set - anchors_set)
    actual_pool_size = min(pool_size, len(potential_candidates_list))
    if actual_pool_size < n_way * num_samples_per_class:
        actual_pool_size = len(potential_candidates_list)
        if actual_pool_size < n_way * num_samples_per_class:
            raise ValueError('Not enough nodes in train_idx to create a task without duplicates.')
    candidate_nodes_list = random.sample(potential_candidates_list, actual_pool_size)
    candidate_nodes = torch.tensor(candidate_nodes_list, device=device)
    candidate_features = features[candidate_nodes]
    sim_matrix = torch.matmul(anchor_features, candidate_features.T)
    used_mask = torch.zeros(actual_pool_size, dtype=torch.bool, device=device)
    sorted_sim_indices = torch.argsort(sim_matrix, dim=1, descending=True)
    samples_by_anchor_list = []
    for i in range(n_way):
        anchor_sorted_indices = sorted_sim_indices[i]
        available_indices = anchor_sorted_indices[~used_mask[anchor_sorted_indices]]
        selected_indices = available_indices[:num_samples_per_class]
        samples_by_anchor_list.append(candidate_nodes[selected_indices])
        used_mask[selected_indices] = True
    samples_by_anchor = torch.stack(samples_by_anchor_list)
    support_nodes_by_anchor = samples_by_anchor[:, :k_shot]
    query_nodes_by_anchor = samples_by_anchor[:, k_shot:]
    id_support = support_nodes_by_anchor.flatten().cpu().numpy()
    id_query = query_nodes_by_anchor.flatten().cpu().numpy()
    return (id_support, id_query, selected_anchors.numpy())
