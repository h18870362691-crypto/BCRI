"""Two-stage pseudo-task training and development-only early stopping."""
import math
import hashlib
import json
import time
import torch
from torch import nn
from torch.nn import functional as F
import data as D
import pipeline as R
from config import pretraining_config
from pipeline import sample_pseudo
import gc
from types import SimpleNamespace
import geometry as G
from config import training_config
from geometry import geometry_state
from inference import DEFAULTS, initialize_policies, make_solver
from pipeline import prepare_policy_tasks
from inference import Predictor

# early stopping
class EarlyStopping:
    def __init__(self, patience, min_delta=0.0):
        if patience < 1 or min_delta < 0:
            raise ValueError('Invalid early-stopping configuration')
        self.patience, self.min_delta = patience, min_delta
        self.best_accuracy = -math.inf
        self.best_epoch = None
        self.bad_checks = 0
        self.best_states = None

    def update(self, epoch, accuracy, modules):
        if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError('Invalid development accuracy')
        improved = accuracy > self.best_accuracy + self.min_delta
        if improved:
            self.best_accuracy, self.best_epoch, self.bad_checks = accuracy, epoch, 0
            self.best_states = [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                                for m in modules]
        else:
            self.bad_checks += 1
        return improved, self.bad_checks >= self.patience

    def restore(self, modules):
        if self.best_states is None:
            raise RuntimeError('No development checkpoint was selected')
        for module, state in zip(modules, self.best_states, strict=True):
            module.load_state_dict(state, strict=True)

# pretrain
def parameter_digest(module):
    h=hashlib.sha256()
    for value in module.state_dict().values():
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class EpisodicReadout(nn.Module):
    """Training-only cosine-to-support head; not used by BCRI prototype inference."""
    def __init__(self):
        super().__init__()
        self.log_scale=nn.Parameter(torch.tensor(math.log(100.)))

    def forward(self,query,support,way,shot):
        scores=(query@support.T)*self.log_scale.clamp(max=math.log(100.)).exp()
        return scores.reshape(len(query),way,shot).logsumexp(-1)


def episode_forward(encoder,head,support,query,cfg,augment):
    x,adj,ids=encoder.context(support,query)
    if augment:
        adj=adj.masked_fill((torch.rand_like(adj.float())<cfg['edge_drop'])&(adj>=0),-1)
        x=x+cfg['input_noise']*torch.randn_like(x)
    z=encoder.model(x,adj)
    ns=len(support)
    s,q=z[ids[:ns]],z[ids[ns:]]
    return head(q,s,encoder.e['way'],encoder.e['shot']),q,s


def pseudo_loss(encoder,head,support,query,cfg):
    log,q,s=episode_forward(encoder,head,support,query,cfg,True)
    e=encoder.e
    target=torch.arange(e['way'],device=log.device).repeat_interleave(len(query)//e['way'])
    labels=torch.arange(e['way'],device=log.device).repeat_interleave(e['shot'])
    same=target[:,None].eq(labels[None])
    contrast=-((q@s.T/cfg['contrast_temperature']).log_softmax(-1)*same).sum(-1).div(same.sum(-1)).mean()
    classification=F.cross_entropy(log,target,label_smoothing=cfg['label_smoothing'])
    return classification+cfg['contrast_weight']*contrast


@torch.no_grad()
def evaluate_encoder(encoder,head,packet,truth,cfg):
    encoder.model.eval();head.eval()
    logs=[episode_forward(encoder,head,s,q,cfg,False)[0].cpu()
          for s,q in zip(packet['support'],packet['query'])]
    return D.metrics(torch.stack(logs),truth)


@torch.enable_grad()
def pretrain_encoder(key,device='cuda',max_steps=None,patience=None,
                     validation_tasks=None,validate_every=None):
    R.DEVICE=device;R.setup()
    encoder=R.Encoder(key)
    e=encoder.e
    cfg=pretraining_config(e['dataset'])
    steps=cfg['max_steps'] if max_steps is None else max_steps
    patience=cfg['patience'] if patience is None else patience
    validation_tasks=cfg['validation_tasks'] if validation_tasks is None else validation_tasks
    validate_every=cfg['validate_every'] if validate_every is None else validate_every
    if min(steps,patience,validation_tasks,validate_every)<1:
        raise ValueError('Encoder training counts must be positive')
    initial_digest=parameter_digest(encoder.model)
    encoder.pseudo_features()
    validation=D.episode_ids(e,encoder.graph,'development',validation_tasks)
    truth=D.targets(e,validation,'development',encoder.graph)
    R.seed(e['seed']+2701)
    head=EpisodicReadout().to(device)
    parameters=list(encoder.model.parameters())+list(head.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=steps,eta_min=1e-6)
    stopper=EarlyStopping(patience,cfg['min_delta'])
    history=[];best_neighbors=None;reason='max_steps';started=time.monotonic()
    for step in range(1,steps+1):
        if cfg['resample_neighbors_every'] and step%cfg['resample_neighbors_every']==0:
            encoder.resample_neighbors(step)
        encoder.model.train();head.train()
        support,query=sample_pseudo(encoder,cfg['pool_size'])
        optimizer.zero_grad(set_to_none=True)
        loss=pseudo_loss(encoder,head,support,query,cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite encoder pretraining loss')
        loss.backward()
        gradients=[p.grad for p in encoder.model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError('Invalid encoder gradients')
        torch.nn.utils.clip_grad_norm_(parameters,cfg['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();scheduler.step()
        if step%validate_every==0 or step==steps:
            score=evaluate_encoder(encoder,head,validation,truth,cfg)
            improved,stop=stopper.update(step,score['accuracy'],(encoder.model,head))
            if improved:
                best_neighbors=encoder.adj.detach().cpu().clone()
            history.append(dict(step=step,**score,improved=improved,bad_checks=stopper.bad_checks))
            print(json.dumps(dict(stage='encoder_development',key=e['key'],step=step,max_steps=steps,
                train_loss=float(loss.detach()),**score,best_step=stopper.best_epoch,
                bad_checks=stopper.bad_checks,patience=patience,seconds=round(time.monotonic()-started,2))),flush=True)
            if stop:
                reason='patience'
                break
    stopper.restore((encoder.model,head))
    encoder.adj=best_neighbors.to(device)
    optimizer.zero_grad(set_to_none=True)
    encoder.freeze()
    encoder.assert_frozen()
    selected_digest=parameter_digest(encoder.model)
    encoder.pretraining_metadata=dict(stage='encoder_pretraining',fresh_initialization=True,
        pretrained_weights_loaded=False,random_seed=e['seed'] if D.continuous_running(e) else e['seed']+1701,
        repeat=e.get('repeat'),initialization_rng='continuing global stream' if D.continuous_running(e) else 'legacy per-stage seed',
        max_steps=steps,completed_steps=step,
        best_step=stopper.best_epoch,best_development_accuracy=stopper.best_accuracy,
        patience=patience,validate_every=validate_every,validation_tasks=validation_tasks,
        stop_reason=reason,best_parameters_and_neighbors_restored=True,
        encoder_updated=(selected_digest!=initial_digest),initial_parameter_sha256=initial_digest,
        selected_parameter_sha256=selected_digest,validation_history=history,
        objective=f"pseudo-query CE with label smoothing + {cfg['contrast_weight']} pseudo-label contrast",
        configuration=dict(cfg,max_steps=steps,patience=patience,validate_every=validate_every,validation_tasks=validation_tasks),
        lr=cfg['lr'],test_query_targets_used_for_training_or_selection=False,weights_saved=False,
        standard_training=(steps==cfg['max_steps'] and patience==cfg['patience']
                           and validate_every==cfg['validate_every'] and validation_tasks==cfg['validation_tasks']))
    return encoder

# train
METHOD_ID = 'bcri'


def validation_metrics(predictor, truth, controller, policy, batch=40):
    predictor.set_policies(controller, policy)
    with torch.no_grad():
        logits = torch.cat([predictor.infer_batch(i, min(i+batch, predictor.task_count))
                            for i in range(0, predictor.task_count, batch)])
    return D.metrics(logits, truth)


def train_policies(key, device='cuda', max_epochs=None, tasks=None,
                   patience=None, validation_tasks=None, pretrain_steps=None,
                   pretrain_validate_every=None, encoder_patience=None, encoder=None):
    R.DEVICE = device
    R.setup()
    e = D.entry(key)
    cfg = training_config(e['dataset'])
    max_epochs = cfg['max_epochs'] if max_epochs is None else max_epochs
    tasks = cfg['tasks'] if tasks is None else tasks
    patience = cfg['patience'] if patience is None else patience
    validation_tasks = cfg['validation_tasks'] if validation_tasks is None else validation_tasks
    if min(max_epochs, tasks, patience, validation_tasks) < 1:
        raise ValueError('Training counts and patience must be positive')
    if encoder is None:
        encoder=pretrain_encoder(key,device,max_steps=pretrain_steps,patience=encoder_patience,
            validation_tasks=validation_tasks,validate_every=pretrain_validate_every)
    if encoder.pretraining_metadata is None:
        raise ValueError('Stage two requires a development-selected pretrained encoder')
    if encoder.e['key']!=e['key'] or encoder.x.device.type!=torch.device(device).type:
        raise ValueError('Pretrained encoder belongs to a different setting or device')
    encoder.assert_frozen()
    e=encoder.e
    packet,bg,validation_packet,validation_background=prepare_policy_tasks(encoder,tasks,validation_tasks)
    with torch.no_grad():
        local = torch.stack([R.local_gram(encoder.features(encoder.context(s,q))).cpu()
                             for s,q in zip(packet['support'],packet['query'])]).to(device)
        pos, cor = G.source_bases(packet,bg)
        pos, cor = pos.to(device), cor.to(device)
        state = geometry_state(pos,cor,local,bg['parts'],e['way'],e['shot'])
    controls = dict(DEFAULTS, **e.get('controller_config',{}))
    R.seed(e['seed']+8121)
    controller, policy = initialize_policies(e['seed'],device,controls)
    bundle = SimpleNamespace(encoder=encoder,query_controller=controller,geometry_policy=policy,entry=e)
    validation = Predictor(key,bundle,validation_packet,validation_background,limit=validation_tasks)
    validation.precompute_local()
    truth = D.targets(e,validation.packet,'development',encoder.graph)
    parameters = list(controller.parameters()) + list(policy.parameters())
    optimizer = torch.optim.AdamW(parameters,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    stopper = EarlyStopping(patience,cfg['min_delta'])
    solve = make_solver(controller,controls)
    queries = packet['query'].shape[1]
    if queries % e['way']:
        raise ValueError('Pseudo queries must be grouped evenly by pseudo class')
    target = torch.arange(e['way'],device=device).repeat_interleave(queries//e['way'])
    history = []
    updates = 0
    started = time.monotonic()
    reason = 'max_epochs'
    with torch.enable_grad():
        for epoch in range(1,max_epochs+1):
            controller.train(); policy.train()
            order = (torch.randperm(tasks) if D.continuous_running(e) else
                     torch.randperm(tasks,generator=torch.Generator().manual_seed(e['seed']+5500+epoch)))
            for start in range(0,tasks,cfg['batch_size']):
                indices = order[start:start+cfg['batch_size']]
                ix = indices.to(device)
                parts = [bg['parts'][int(i)] for i in indices]
                optimizer.zero_grad(set_to_none=True)
                theta,_ = policy(tuple(t[ix] for t in state))
                worlds = G.worlds(theta[:,None],pos[ix],cor[ix],local[ix],e,parts)[:,0]
                log,aux = solve(worlds,e['way'],e['shot'],e['bcpr_config'],return_aux=True)
                classification = F.nll_loss(log.reshape(-1,e['way']),target[None].expand(len(ix),-1).reshape(-1))
                loss = classification + controls['background_loss_weight']*aux['background_loss']
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                gradients = [p.grad for p in parameters if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                    raise FloatingPointError('Invalid policy gradient')
                torch.nn.utils.clip_grad_norm_(parameters,cfg['gradient_clip'],error_if_nonfinite=True)
                optimizer.step()
                updates += 1
            if epoch % cfg['validate_every'] == 0 or epoch == max_epochs:
                score = validation_metrics(validation,truth,controller,policy)
                improved, stop = stopper.update(epoch,score['accuracy'],(controller,policy))
                history.append(dict(epoch=epoch,**score,improved=improved,bad_checks=stopper.bad_checks))
                print(json.dumps(dict(stage='development',key=key,epoch=epoch,max_epochs=max_epochs,
                    train_loss=float(loss.detach()),**score,best_epoch=stopper.best_epoch,
                    bad_checks=stopper.bad_checks,patience=patience,seconds=round(time.monotonic()-started,2))),flush=True)
                if stop:
                    reason = 'patience'
                    break
    stopper.restore((controller,policy))
    encoder.assert_frozen()
    optimizer.zero_grad(set_to_none=True)
    controller.eval().requires_grad_(False)
    policy.eval().requires_grad_(False)
    del optimizer,validation,solve,pos,cor,local,state
    gc.collect()
    torch.cuda.empty_cache()
    bundle.metadata = dict(method_id=METHOD_ID,pretraining=encoder.pretraining_metadata,
        graph_source=e['graph'],class_partition=e['classes'],
        evaluation_protocol=e.get('evaluation_protocol'),repeat=e.get('repeat'),
        task_seeds=(dict(base_seed=e['seed'],rng='continuous across all repeats; no per-stage reset')
                    if D.continuous_running(e) else
                    dict(encoder_pseudo=e['seed']+2701,policy_pseudo=e['seed']+101,
                         development=e['seed']+202,test=e['seed']+303)),
        max_epochs=max_epochs,completed_epochs=epoch,
        best_epoch=stopper.best_epoch,best_development_accuracy=stopper.best_accuracy,
        stop_reason=reason,patience=patience,validation_tasks=validation_tasks,
        selection_scope='this setting and its own development classes only',
        selection_metric='strict development accuracy improvement; ties consume patience',
        pseudo_tasks=tasks,optimizer_updates=updates,best_parameters_restored=True,
        validation_history=history,encoder_frozen=True,weights_saved=False,
        fully_unrolled=True,controller_config=controls,
        standard_training=(max_epochs==cfg['max_epochs'] and tasks==cfg['tasks']
                           and patience==cfg['patience'] and validation_tasks==cfg['validation_tasks']))
    return bundle
