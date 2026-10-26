"""Module two: background-gain contribution policy and prototype inference."""
import torch
from torch import nn
from torch.nn import functional as F
from geometry import BackgroundGeometry
import copy
import pipeline as R
import geometry as G
from geometry import geometry_state

# learnable policies
def node_state(z, center, probability, previous, temperature, iteration):
    """Class/row permutation invariant features, normalized for varying way/shot."""
    way = probability.shape[-1]
    nq = probability.shape[1]
    query = z[:, :, -nq:]
    dist = (query[:, :, :, None] - center[:, :, None]).square().sum(-1)
    evidence = (-dist / temperature).softmax(-1)
    real, background = (evidence[:, 0], evidence[:, 1:])
    midpoint = (real[:, None] + background) / 2
    js = 0.5 * ((real[:, None] * (real[:, None].clamp_min(1e-30).log() - midpoint.clamp_min(1e-30).log())).sum(-1) + (background * (background.clamp_min(1e-30).log() - midpoint.clamp_min(1e-30).log())).sum(-1))
    top = probability.topk(2, dim=-1).values
    entropy = -(probability * probability.clamp_min(1e-30).log()).sum(-1) / __import__('math').log(way)
    predicted = probability.argmax(-1)
    real_evidence = real.gather(-1, predicted[..., None]).squeeze(-1)
    bg_evidence = background.gather(-1, predicted[:, None, :, None].expand(-1, len(background[0]), -1, 1)).squeeze(-1).amax(1)
    imbalance = (probability.mean(1) - 1 / way).abs().sum(-1) / 2
    return torch.stack((top[..., 0], top[..., 0] - top[..., 1], entropy, js.amax(1) / __import__('math').log(2), real_evidence - bg_evidence, (probability - previous).abs().sum(-1) / 2, torch.full_like(entropy, iteration), imbalance[:, None].expand_as(entropy)), -1)

# bidirectional
DEFAULTS = dict(steps=10, probe=0.1, budget=1.0, evidence_temperature=0.01, evidence_scale=1.0, background_loss_weight=0.1)

class BidirectionalController(nn.Module):
    """Learn add/withdraw/keep; evidence has a fixed positive logit effect."""

    def __init__(self, controls=None):
        super().__init__()
        self.controls = dict(DEFAULTS, **controls or {})
        self.net = nn.Sequential(nn.LayerNorm(9), nn.Linear(9, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.0)
        self.net[-1].bias.data[2] = 1.0

    def forward(self, state, evidence):
        logits = self.net(state.float())
        c = self.controls
        bias = c['evidence_scale'] * torch.tanh(evidence / c['evidence_temperature'])
        bias = torch.cat((bias, torch.zeros_like(bias[..., :1])), -1)
        return (logits + bias.to(logits.dtype)).softmax(-1)

def initialize_policies(seed, device, controls=None):
    if R.D.continuous_running():
        policy=BackgroundGeometry().to(device)
        controller=BidirectionalController(controls).to(device)
        return controller,policy
    torch.manual_seed(seed + 8121)
    policy = BackgroundGeometry().to(device)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 9187)
        controller = BidirectionalController(controls).to(device)
    return (controller, policy)

def initial_state(z, way, shot, cfg):
    ns = way * shot
    support = z[:, :, :ns].reshape(len(z), z.shape[1], way, shot, -1)
    source = F.normalize(support.mean(3), dim=-1, eps=1e-12)
    scale = shot * cfg['anchor'] / support[:, 0].sum(2).norm(dim=-1).clamp_min(1e-12)
    known = F.one_hot(torch.arange(way, device=z.device).repeat_interleave(shot), way).to(z.dtype)
    coefficient = known[None] * scale[:, None]
    support_moment = torch.einsum('bvnd,bnc->bvcd', z[:, :, :ns], coefficient)
    query_coefficient = z.new_zeros(len(z), z.shape[2] - ns, way)
    center = F.normalize(support_moment, dim=-1, eps=1e-12)
    return (source, coefficient, support_moment, query_coefficient, center)

def readout(query, center, temperature):
    distance = (query[:, :, None] - center[:, None]).square().sum(-1)
    return (-distance / temperature).log_softmax(-1)

def bounded_update(coefficient, probability, actions, budget):
    add, withdraw = (actions[..., 0:1], actions[..., 1:2])
    retained = (1 - withdraw) * coefficient
    room = (budget - retained.sum(-1, keepdim=True)).clamp_min(0)
    return retained + add * room * probability

def _leaveout_value(center, moments, counts, source, anchor):
    fit = counts[:, None] * center.square().sum(-1) - 2 * (center * moments).sum(-1)
    return (fit + anchor * (center - source[:, :, None]).square().sum(-1)).sum(-1)

@torch.no_grad()
def directional_gains(query, source, support_moment, coefficient, probability, cfg, controls):
    """Two one-query probes; fixed U/geometry/anchor, excluding self-fit."""
    q = query.shape[2]
    assert q > 1
    u = probability.detach()
    moment = support_moment + torch.einsum('bvqd,bqc->bvcd', query, coefficient)
    center = F.normalize(moment, dim=-1, eps=1e-12)
    total = torch.einsum('bvqd,bqc->bvcd', query, u)
    moments = total[:, :, None] - query[:, :, :, None] * u[:, None, :, :, None]
    counts = u.sum(1)[:, None] - u
    anchor = cfg['anchor'] * controls['shot']
    baseline = _leaveout_value(center[:, :, None], moments, counts, source, anchor)
    mass = coefficient.sum(-1, keepdim=True)
    deltas = (controls['probe'] * (controls['budget'] - mass).clamp_min(0) * u, -controls['probe'] * coefficient)
    gains = []
    for delta in deltas:
        candidate = F.normalize(moment[:, :, None] + query[:, :, :, None] * delta[:, None, :, :, None], dim=-1, eps=1e-12)
        value = _leaveout_value(candidate, moments, counts, source, anchor)
        gains.append((baseline - value) / (q - 1))
    gains = torch.stack(gains, -1)
    real = gains[:, 0]
    background = gains[:, 1:].amax(1)
    evidence = real - background.clamp_min(0)
    return dict(real=real, background=background, excess=real - background, evidence=evidence)

def background_penalty(query, source, old_center, new_center, probability, cfg, shot):
    query, source, u = (query.detach(), source.detach(), probability.detach())
    moments = torch.einsum('bvqd,bqc->bvcd', query, u)
    count = u.sum(1)[:, None]

    def value(center):
        fit = count * center.square().sum(-1) - 2 * (center * moments).sum(-1)
        return (fit + shot * cfg['anchor'] * (center - source).square().sum(-1)).sum(-1)
    gain = (value(old_center.detach()) - value(new_center)) / query.shape[2]
    score = gain[:, 0] - gain[:, 1:].amax(-1).clamp_min(0)
    return (-score).relu().mean()

def infer(z, way, shot, cfg, controller, controls=None, return_aux=False, trace=None):
    c = dict(DEFAULTS, **controls or {})
    c['shot'] = shot
    z = z.double()
    if controller is None:
        raise ValueError('BCRI requires its query contribution controller')
    assert z.shape[1] >= 2 and c['steps'] > 0 and (c['budget'] > 0)
    query = z[:, :, way * shot:]
    source, support_coefficient, support_moment, a, center = initial_state(z, way, shot, cfg)
    previous = readout(query[:, 0], center[:, 0], cfg['temperature']).exp()
    penalties = []
    for iteration in range(c['steps']):
        u = readout(query[:, 0], center[:, 0], cfg['temperature']).exp()
        gains = directional_gains(query, source, support_moment, a, u, cfg, c)
        state = node_state(z, center, u, previous, cfg['temperature'], iteration / max(1, c['steps'] - 1))
        state = torch.cat((state, a.sum(-1, keepdim=True) / c['budget']), -1)
        actions = controller(state, gains['evidence']).double()
        new_a = bounded_update(a, u, actions, c['budget'])
        new_center = F.normalize(support_moment + torch.einsum('bvqd,bqc->bvcd', query, new_a), dim=-1, eps=1e-12)
        if return_aux:
            penalties.append(background_penalty(query, source, center, new_center, u, cfg, shot))
        if trace is not None:
            trace.append(dict(coefficient=new_a, actions=actions, gains=gains, support_coefficient=support_coefficient, center=new_center))
        previous, a, center = (u, new_a, new_center)
    log = readout(query[:, 0], center[:, 0], cfg['temperature'])
    if return_aux:
        return (log, dict(background_loss=torch.stack(penalties).mean()))
    return log

def make_solver(controller, controls=None):

    def solve(z, way, shot, cfg, return_aux=False, trace=None):
        return infer(z, way, shot, cfg, controller, controls, return_aux, trace)
    return solve

# model
class Predictor:
    def __init__(self,key,bundle,packet,background,limit=None):
        self.key = key
        self.e = {k:bundle.entry[k] for k in ('way','shot','center','global_weight','bcpr_config')}
        if 'controller_config' in bundle.entry:
            self.e['controller_config']=bundle.entry['controller_config']
        self.encoder = bundle.encoder
        if self.encoder.pretraining_metadata is None:
            raise RuntimeError('Complete encoder pretraining before policy inference')
        self.encoder.assert_frozen()
        self.model = self.encoder.model.eval().requires_grad_(False)
        self.device = str(self.encoder.x.device)
        self.set_policies(bundle.query_controller,bundle.geometry_policy)
        self.packet,self.background=packet,background
        self.task_count = len(self.packet['support'])
        if limit is not None:
            if not 0 < limit <= self.task_count:
                raise ValueError('Requested episode count exceeds available inputs')
            self.packet = R.sliced(self.packet,0,limit,self.task_count)
            self.background = R.sliced(self.background,0,limit,self.task_count)
            self.task_count = limit
        self.local = None
        if len(self.background['parts']) != self.task_count:
            raise RuntimeError('Task/background count mismatch')

    def set_policies(self,controller,policy):
        self.query_controller = copy.deepcopy(controller).cpu().eval().requires_grad_(False)
        self.geometry_policy = copy.deepcopy(policy).cpu().eval().requires_grad_(False)
        self.solve = make_solver(self.query_controller,self.e.get('controller_config'))

    def precompute_local(self):
        self.local = torch.stack([self.local_geometry(s,q) for s,q in
                                  zip(self.packet['support'],self.packet['query'])])

    def local_geometry(self,support,query):
        with torch.no_grad():
            return R.local_gram(self.encoder.features(self.encoder.context(support,query))).cpu()

    def infer_batch(self,start,stop):
        with torch.no_grad():
            packet = R.sliced(self.packet,start,stop,self.task_count)
            bg = R.sliced(self.background,start,stop,self.task_count)
            local = self.local[start:stop] if self.local is not None else torch.stack([
                self.local_geometry(s,q) for s,q in zip(packet['support'],packet['query'])])
            pos,cor = G.source_bases(packet,bg)
            theta,_ = self.geometry_policy(geometry_state(pos,cor,local,bg['parts'],self.e['way'],self.e['shot']))
            worlds = G.worlds(theta[:,None],pos,cor,local,self.e,bg['parts'])[:,0]
            logits = self.solve(worlds,self.e['way'],self.e['shot'],self.e['bcpr_config'])
            return logits.cpu()
