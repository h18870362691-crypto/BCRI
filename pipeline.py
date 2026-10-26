"""Memory-only encoder lifecycle and task input preparation."""
import random
from types import SimpleNamespace
import numpy as np
import torch
import data as D
from encoder import LocalGraphEncoder
from encoder import local_batch, encode
from data import unsupervised_task_generator

# runtime
DEVICE='cuda'


def setup():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False


def seed(value):
    if D.continuous_running():
        return
    random.seed(value);np.random.seed(value)
    torch.manual_seed(value);torch.cuda.manual_seed_all(value)


def sliced(data,start,stop,total):
    return {k:(v[start:stop] if (isinstance(v,torch.Tensor) and v.ndim and len(v)==total)
               or (k=='parts' and isinstance(v,list)) else v) for k,v in data.items()}


def sampled_neighbors(edge_index,n,limit,random_seed):
    u,v=edge_index.cpu().numpy()
    row=np.concatenate((u,v));col=np.concatenate((v,u))
    order=np.argsort(row,kind='stable')
    col=col[order];counts=np.bincount(row,minlength=n)
    boundaries=np.concatenate(([0],counts.cumsum()))
    result=np.full((n,limit),-1,dtype=np.int64)
    rng=np.random if D.continuous_running() else np.random.default_rng(random_seed)
    for i in range(n):
        choices=col[boundaries[i]:boundaries[i+1]]
        if len(choices)>limit:
            choices=choices[rng.choice(len(choices),limit,replace=False)]
        result[i,:len(choices)]=choices
    return torch.from_numpy(result)


class Encoder:
    def __init__(self,key):
        self.e=D.entry(key)
        self.graph=D.load_graph(D.ROOT/self.e['graph'])
        self.e['classes']=D.class_partition(self.e,self.graph['y'])
        self.profile=self.e['profile']
        self.x=self.graph['x'].to(DEVICE)
        self.adj=None
        self.resample_neighbors(0)
        seed(self.e['seed']+1701)
        self.model=LocalGraphEncoder(self.x.shape[-1],self.profile['hidden_dim'],
            self.profile['dropout'],self.profile['heads'],self.profile['layers']).to(DEVICE)
        self.model.train().requires_grad_(True)
        self._task_features=None
        self._geometry=None
        self.pretraining_metadata=None

    def resample_neighbors(self,step):
        self.adj=sampled_neighbors(self.graph['edge_index'],len(self.x),self.profile['max_neighbors'],
                                  self.e['seed']+3701+step).to(self.x.device)

    def pseudo_features(self):
        if self._task_features is None:
            from data import precompute_adaptive_task_features
            numeric=SimpleNamespace(x=self.graph['x'],edge_index=self.graph['edge_index'],num_nodes=len(self.x))
            with torch.no_grad():
                features=precompute_adaptive_task_features(numeric)
            self._task_features=features if self.e['dataset']=='Cora' else features.to(self.x.device)
        return self._task_features

    def geometry(self):
        if self._geometry is None:
            from geometry import DiffusionGeometry
            # Neither true labels nor encoder predictions enter field construction.
            self._geometry=DiffusionGeometry(dict(x=self.graph['x'],edge_index=self.graph['edge_index']),
                                              self.e['dataset'],str(self.x.device))
        return self._geometry

    def freeze(self):
        self.model.eval().requires_grad_(False)
        self.initial={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
        self.frozen_neighbors=self.adj.detach().cpu().clone()

    def assert_frozen(self):
        assert not any(p.requires_grad for p in self.model.parameters())
        assert all(torch.equal(v.detach().cpu(),self.initial[k]) for k,v in self.model.state_dict().items())
        assert torch.equal(self.adj.cpu(),self.frozen_neighbors)

    def context(self,support,query):
        return local_batch(self.x,self.adj,support,query)

    def features(self,context):
        return encode(self.model,*context)


def local_gram(views):
    raw,smooth,latent=views
    return (((raw.double()@raw.double().T+smooth.double()@smooth.double().T)
             +latent.double()@latent.double().T)/3).float().double()

# preparation
def sample_pseudo(encoder,pool_size=4096):
    e=encoder.e
    support,query,_=unsupervised_task_generator(np.arange(len(encoder.x)),e['way'],e['shot'],
        e['profile']['queries_per_class'],encoder.pseudo_features(),pool_size=pool_size)
    return torch.from_numpy(support),torch.from_numpy(query)


def prepare_policy_tasks(encoder,count,validation_count):
    seed(encoder.e['seed']+101)
    support,query=[],[]
    for _ in range(count):
        s,q=sample_pseudo(encoder)
        support.append(s);query.append(q)
    support,query=torch.stack(support),torch.stack(query)
    validation=D.episode_ids(encoder.e,encoder.graph,'development',validation_count)
    encoder._task_features=None
    all_support=torch.cat((support,validation['support']))
    all_query=torch.cat((query,validation['query']))
    packet,bg=encoder.geometry().build(all_support,all_query,encoder.e)
    total=count+validation_count
    return (sliced(packet,0,count,total),sliced(bg,0,count,total),
            sliced(packet,count,total,total),sliced(bg,count,total,total))


def prepare_evaluation(encoder,panel,count):
    ids=D.episode_ids(encoder.e,encoder.graph,panel,count)
    return encoder.geometry().build(ids['support'],ids['query'],encoder.e)
