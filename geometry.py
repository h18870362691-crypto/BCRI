"""Module one: streamed matched backgrounds and learned task geometry."""
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
import json
from sklearn.cluster import MiniBatchKMeans
from torch import nn

# geometry
CONFIG = dict(holdout_seed=261015)
DEPTHS = (1, 4, 16, 40)
BACKGROUNDS = ('degree', 'feature', 'local', 'community')

def seedof(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)

def nonnegative_sqrt(values):
    """Same clipped square root, with zero gradient on inactive eigenvalues."""
    result = torch.zeros_like(values)
    active = values > 0
    result[active] = values[active].sqrt()
    return result

def embedding(gram, center):
    eig, vec = torch.linalg.eigh(gram.double())
    assert float(eig.min()) > -3e-05, ('nonPSD', float(eig.min()))
    k = min(40, gram.shape[-1])
    z = vec[..., -k:] * nonnegative_sqrt(eig[..., -k:]).unsqueeze(-2)
    if center:
        z = z - z.mean(-2, keepdim=True)
    return F.normalize(z, dim=-1, eps=1e-12).float()

def outer_partition(entry, key, index):
    way, shot = (entry['way'], entry['shot'])
    ns = way * shot
    if shot < 3:
        return torch.arange(ns), torch.empty(0, dtype=torch.long)
    pos = seedof(f"{key}:{index}:holdout:{CONFIG['holdout_seed']}") % shot
    held = torch.arange(way) * shot + pos
    train = torch.tensor([c * shot + j for c in range(way) for j in range(shot) if j != pos])
    return (train, held)

def psd_cosine(kernel):
    eig, vec = torch.linalg.eigh(kernel.double())
    value = vec * nonnegative_sqrt(eig).unsqueeze(-2)
    value = torch.nn.functional.normalize(value, dim=-1, eps=1e-12)
    return value @ value.transpose(-1, -2)

def partition(e, key, i):
    fit, _ = outer_partition(e, key, i)
    order = np.random.default_rng(seedof(f'{key}:{i}:background_roles')).permutation(4)
    return dict(fit=fit.tolist(), proposal=order[:2].tolist())

def source_bases(pack, bg):
    real = pack['real_relations'].permute(0, 3, 1, 2).double()
    real[:, 3] = pack['real_geometry_gram'].double()
    mean = bg['mean'].double()
    drop = mean[:, 2, 3]
    worlds = torch.cat((real[:, None], mean), 1)
    fifth = torch.cat((drop[:, None], mean[:, :, 3]), 1)
    positive = torch.cat((worlds, fifth[:, :, None]), 2)
    old = (pack['real_relations'] - pack['increment']).permute(0, 3, 1, 2).double()
    correction = torch.cat((old[:, None], mean), 1)
    correction = torch.cat((correction, correction[:, :, 3:4]), 2)
    return (positive, correction)

def kernels(theta, positive, correction, worlds=(0,)):
    theta = theta.to(positive.device)
    weights = theta[..., :5]
    lambdas = theta[..., 5:]
    raw = torch.einsum('bnp,bs pij->bnsij'.replace(' ', ''), weights, positive[:, list(worlds)])
    subtract = torch.einsum('bnp,bnk,bkpij->bnij', weights, lambdas, correction)
    raw = raw - subtract[:, :, None]
    shape = raw.shape
    flat = raw.reshape(-1, shape[-1], shape[-1])
    no_correction = lambdas.sum(-1) == 0
    required = (~no_correction)[:, :, None].expand(shape[:3]).reshape(-1)
    result = flat.clone()
    if bool(required.any()):
        result[required] = psd_cosine(flat[required])
    result = result.reshape(shape)
    return result

def worlds(theta, positive, correction, local, entry, parts):
    """Construct the real task geometry and two matched reference spaces."""
    batch, points, _ = theta.shape
    real = kernels(theta, positive, correction, (0,))
    indices = torch.tensor([[x + 1 for x in p['proposal']] for p in parts], device=positive.device)
    background = positive[torch.arange(batch, device=positive.device)[:, None], indices]
    reference = torch.einsum('bnp,bvpij->bnvij', theta[..., :5], background)
    kernel = torch.cat((real, reference), 2)
    weight = entry['global_weight']
    gram = (local[:, None, None] + weight * kernel) / (1 + weight)
    nodes = gram.shape[-1]
    return embedding(gram.reshape(-1, nodes, nodes), entry['center']).reshape(batch, points, 3, nodes, -1)

# graph fields
def transition(u,v,n):
    degree=torch.bincount(u,minlength=n)
    isolated=torch.where(degree==0)[0]
    rows=torch.cat((u,isolated));cols=torch.cat((v,isolated))
    values=degree.clamp_min(1)[rows].float().reciprocal()
    op=torch.sparse_coo_tensor(torch.stack((rows,cols)),values,(n,n)).coalesce().to_sparse_csr()
    return op,degree


def grouped_permutation(groups,seed):
    order=np.argsort(groups,kind='stable')
    cuts=np.flatnonzero(np.diff(groups[order]))+1
    rng=np.random.default_rng(seed);permutation=np.arange(len(groups))
    for ids in np.split(order,cuts):
        permutation[ids]=rng.permutation(ids)
    return torch.from_numpy(permutation)


class DiffusionGeometry:
    @torch.no_grad()
    def __init__(self,numeric,dataset,device='cuda',feature_block=256):
        if feature_block<1:
            raise ValueError('Feature block must be positive')
        self.device=device
        self.feature_block=feature_block
        self.x=F.normalize(numeric['x'].float().to(device),dim=-1)
        n=len(self.x)
        u,v=numeric['edge_index'].long().to(device)
        valid=u!=v
        codes=torch.cat((u[valid]*n+v[valid],v[valid]*n+u[valid])).unique(sorted=True)
        u,v=codes.div(n,rounding_mode='floor'),codes%n
        self.op,degree=transition(u,v,n)
        self.mean=(self.x*degree[:,None]).sum(0)/degree.sum().clamp_min(1)
        clusters=min(32,n)
        feature=MiniBatchKMeans(n_clusters=clusters,random_state=261015,n_init=1,
            batch_size=2048,max_iter=60).fit_predict(self.x.cpu().numpy())
        generator=torch.Generator(device=device).manual_seed(261015)
        sketch=torch.randn(n,8,generator=generator,device=device)
        for _ in range(12):
            sketch=torch.sparse.mm(self.op,sketch)
        community=MiniBatchKMeans(n_clusters=clusters,random_state=261016,n_init=1,
            batch_size=2048,max_iter=60).fit_predict(F.normalize(sketch,dim=-1).cpu().numpy())
        degree_cpu=degree.cpu().numpy()
        _,joint=np.unique(np.stack((degree_cpu,feature),1),axis=0,return_inverse=True)
        groups=dict(matched=joint,degree=degree_cpu,feature=feature,community=community)
        self.worlds=[dict(kind='real',operator=self.op)]
        for kind in ['matched',*BACKGROUNDS]:
            for replicate in range(2):
                seed=seedof(dataset+kind+str(replicate))
                if kind=='local':
                    a,b=u[u<v],v[u<v]
                    generator=torch.Generator(device=device).manual_seed(seed)
                    keep=torch.rand(len(a),generator=generator,device=device)>=.15
                    op,_=transition(torch.cat((a[keep],b[keep])),torch.cat((b[keep],a[keep])),n)
                    self.worlds.append(dict(kind=kind,operator=op))
                else:
                    permutation=grouped_permutation(groups[kind],seed).to(device)
                    inverse=torch.empty_like(permutation);inverse[permutation]=torch.arange(n,device=device)
                    self.worlds.append(dict(kind=kind,operator=self.op,permutation=permutation,inverse=inverse))

    @torch.no_grad()
    def world_grams(self,support,query,world):
        ids=torch.cat((support,query),1).to(self.device)
        batch,nodes=ids.shape
        grams=torch.zeros(len(DEPTHS),batch,nodes,nodes,dtype=torch.float64,device=self.device)
        norms=torch.zeros(len(DEPTHS),batch,nodes,dtype=torch.float64,device=self.device)
        for start in range(0,self.x.shape[1],self.feature_block):
            x=self.x[:,start:start+self.feature_block]
            mean=self.mean[start:start+self.feature_block]
            state=x.clone()
            for step in range(1,41):
                if 'permutation' in world:
                    propagated=torch.sparse.mm(world['operator'],state[world['inverse']])[world['permutation']]
                else:
                    propagated=torch.sparse.mm(world['operator'],state)
                state=.15*x+.85*propagated
                if step in DEPTHS:
                    level=DEPTHS.index(step)
                    for first in range(0,batch,16):
                        z=(state[ids[first:first+16]]-mean).double()
                        grams[level,first:first+16]+=z@z.transpose(-1,-2)
                        norms[level,first:first+16]+=z.square().sum(-1)
        scale=norms.clamp_min(1e-24).sqrt()
        return (grams/scale[:,:,:,None]/scale[:,:,None,:]).cpu()

    @torch.no_grad()
    def build(self,support,query,e):
        real=None;matched=[];background={kind:[] for kind in BACKGROUNDS}
        for index,world in enumerate(self.worlds):
            gram=self.world_grams(support,query,world)
            if world['kind']=='real':real=gram
            elif world['kind']=='matched':matched.append(gram.float())
            else:background[world['kind']].append(gram)
            print(json.dumps(dict(stage='geometry',world_done=index+1,worlds=len(self.worlds),
                                  kind=world['kind'],tasks=len(support))),flush=True)
        relations=real.permute(1,2,3,0).float()
        old=torch.stack(matched).mean(0).permute(1,2,3,0)
        packet=dict(support=support,query=query,real_relations=relations,
                    increment=relations-old,real_geometry_gram=real[-1])
        means=torch.stack([torch.stack(background[kind]).mean(0).permute(1,0,2,3)
                           for kind in BACKGROUNDS],1).float()
        bg=dict(mean=means,parts=[partition(e,e['key'],i) for i in range(len(support))])
        return packet,bg

# learnable policies
def support_separation(kernel, same, other):
    """Missing within-class pairs carry no separation evidence, rather than NaN."""
    if not bool(same.any()) or not bool(other.any()):
        return kernel.new_zeros(kernel.shape[:-2])
    return kernel[..., same].mean(-1) - kernel[..., other].mean(-1)

def geometry_state(positive, correction, local, parts, way, shot):
    """Use support-pair statistics and permitted matched backgrounds."""
    labels = torch.arange(way, device=positive.device).repeat_interleave(shot)
    task, backgrounds, masks = ([], [], [])
    for p, c, l, part in zip(positive, correction, local, parts):
        fit = torch.tensor(part['fit'], device=p.device)
        y = labels[fit]
        pairs = torch.triu(torch.ones(len(fit), len(fit), device=p.device, dtype=torch.bool), 1)
        same = pairs & y[:, None].eq(y[None, :])
        other = pairs & ~y[:, None].eq(y[None, :])
        real = p[0][:, fit][:, :, fit]
        controls = c[:, :, fit][:, :, :, fit]
        real_sep = support_separation(real, same, other)
        control_sep = support_separation(controls, same, other)
        gamma = real_sep[None] - control_sep
        allowed = torch.zeros(5, device=p.device)
        allowed[[0] + [i + 1 for i in part['proposal']]] = 1
        small = l[fit][:, fit]
        local_sep = support_separation(small, same, other)
        global_state = torch.cat((real_sep, local_sep[None], (gamma * allowed[:, None]).sum(0) / allowed.sum(), real_sep.new_tensor([way / 10, shot / 10])))
        token = torch.cat((control_sep, gamma, controls.abs().mean((-1, -2))), -1)
        task.append(global_state)
        backgrounds.append(token)
        masks.append(allowed)
    return (torch.stack(task).float(), torch.stack(backgrounds).float(), torch.stack(masks).float())

class BackgroundGeometry(nn.Module):
    """A utility gate directly controls suppression, rather than merely labeling it."""

    def __init__(self):
        super().__init__()
        self.background = nn.Sequential(nn.LayerNorm(28), nn.Linear(28, 32), nn.GELU())
        self.utility = nn.Linear(32, 1)
        self.strength = nn.Linear(32, 1)
        self.depth = nn.Sequential(nn.LayerNorm(45), nn.Linear(45, 32), nn.GELU(), nn.Linear(32, 5))
        nn.init.zeros_(self.depth[-1].weight)
        nn.init.zeros_(self.depth[-1].bias)
        self.depth[-1].bias.data[3] = 4.0
        nn.init.zeros_(self.strength.weight)
        nn.init.constant_(self.strength.bias, -3.0)

    def forward(self, state):
        task, tokens, allowed = state
        h = self.background(torch.cat((task[:, None].expand(-1, 5, -1), tokens), -1))
        utility = self.utility(h).sigmoid().squeeze(-1) * allowed
        strength = F.softplus(self.strength(h).squeeze(-1)) * utility
        suppression = 0.5 * strength / (1 + strength.sum(-1, keepdim=True))
        pooled = (h * utility[..., None]).sum(1) / utility.sum(-1, keepdim=True).clamp_min(1e-06)
        weights = self.depth(torch.cat((task, pooled), -1)).softmax(-1)
        return (torch.cat((weights, suppression), -1).double(), utility)
