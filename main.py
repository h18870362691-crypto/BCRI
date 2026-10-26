"""BCRI main."""
import argparse
import json
from config import canonical_dataset, planned_settings, DEFAULT_SEED, REPEATS, EVALUATION_PROTOCOL, TEST_EPISODES

# main
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',nargs='?',choices=['evaluate','train','pretrain'],default='evaluate')
    parser.add_argument('--key')
    parser.add_argument('--dataset')
    parser.add_argument('--seed',type=int,default=DEFAULT_SEED)
    parser.add_argument('--repeats',type=int,default=REPEATS)
    parser.add_argument('--list-configs',action='store_true')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--max-epochs',type=int)
    parser.add_argument('--pretrain-steps',type=int)
    parser.add_argument('--pretrain-eval-every',type=int)
    parser.add_argument('--encoder-patience',type=int)
    parser.add_argument('--patience',type=int)
    parser.add_argument('--pseudo-tasks',type=int)
    parser.add_argument('--validation-tasks',type=int)
    parser.add_argument('--episodes',type=int,default=TEST_EPISODES)
    parser.add_argument('--batch-size',type=int,default=40)
    args = parser.parse_args()
    if args.repeats<1:
        parser.error('Repeat count must be positive')
    if args.list_configs:
        selected = canonical_dataset(args.dataset) if args.dataset else None
        for setting in planned_settings(args.seed,args.repeats):
            if selected is None or setting['dataset']==selected:
                print(json.dumps(setting,ensure_ascii=False))
        return
    if args.episodes<1 or args.batch_size<1:
        parser.error('Episode count and batch size must be positive')
    import gc
    import torch
    import data as D
    from train import train_policies, METHOD_ID
    from inference import Predictor
    from train import pretrain_encoder
    from pipeline import prepare_evaluation
    dataset = canonical_dataset(args.dataset) if args.dataset else None
    if args.key:
        name,setting = args.key.split('/',1)
        key = canonical_dataset(name)+'/'+setting
        if dataset and canonical_dataset(name)!=dataset:
            parser.error('--key and --dataset disagree')
        parsed=D.entry(key)
        if parsed.get('repeat') not in (None,1):
            parser.error('A continuous sequence must start at repeat 1; select the setting group')
        group_key=key.rsplit('_repeat_',1)[0]
        keys=[group_key+f'_repeat_{i}' for i in range(1,args.repeats+1)]
    elif dataset is not None:
        keys=[s['key'] for s in planned_settings(args.seed,args.repeats) if s['dataset']==dataset]
    else:
        settings=planned_settings(args.seed,args.repeats)
        ready=set()
        for name in dict.fromkeys(s['dataset'] for s in settings):
            try:
                D.graph_path(name);ready.add(name)
            except FileNotFoundError:
                pass
        keys=[s['key'] for s in settings if s['dataset'] in ready]
    if not keys:
        parser.error('No numeric graph is available; inspect --list-configs and provide a dataset/*.pt graph')
    rows = []
    active_group=None
    for key in keys:
        group_key=key.rsplit('_repeat_',1)[0]
        if group_key!=active_group:
            D.start_sequence(key)
            active_group=group_key
        D.activate_repeat(key)
        if args.mode=='pretrain':
            encoder=pretrain_encoder(key,args.device,args.pretrain_steps,args.encoder_patience,
                                     args.validation_tasks,args.pretrain_eval_every)
            print(json.dumps(dict(method_id=METHOD_ID,key=key,pretraining=encoder.pretraining_metadata)),flush=True)
            del encoder
            gc.collect();torch.cuda.empty_cache()
            continue
        bundle = train_policies(key,args.device,args.max_epochs,args.pseudo_tasks,args.patience,
            args.validation_tasks,args.pretrain_steps,args.pretrain_eval_every,args.encoder_patience)
        row = dict(method_id=METHOD_ID,key=key,seed=bundle.entry['seed'],repeat=bundle.entry['repeat'],
                   evaluation_protocol=EVALUATION_PROTOCOL,training=bundle.metadata)
        if args.mode=='evaluate':
            packet,background=prepare_evaluation(bundle.encoder,'test',args.episodes)
            predictor = Predictor(key,bundle,packet,background)
            logs = []
            for start in range(0,predictor.task_count,args.batch_size):
                stop = min(start+args.batch_size,predictor.task_count)
                logs.append(predictor.infer_batch(start,stop))
                print(json.dumps(dict(stage='test',key=key,done=stop,total=predictor.task_count)),flush=True)
            # Scoring targets are materialized only after predictions are complete.
            truth = D.targets(bundle.entry,predictor.packet,'test',bundle.encoder.graph)
            row.update(tasks=predictor.task_count,metrics=D.metrics(torch.cat(logs),truth))
            del predictor,logs,truth,packet,background
        rows.append(row)
        print(json.dumps(row),flush=True)
        del bundle
        gc.collect();torch.cuda.empty_cache()
        if row['repeat']==args.repeats and args.mode=='evaluate':
            import statistics
            group_rows=[r for r in rows if r['key'].rsplit('_repeat_',1)[0]==group_key]
            values=[r['metrics']['accuracy'] for r in group_rows]
            print(json.dumps(dict(stage='repeated_run_summary',group_key=group_key,seed=row['seed'],
                repeats=len(values),mean_accuracy=statistics.mean(values),
                std_accuracy=statistics.pstdev(values),std_ddof=0,
                per_run_accuracy=values,class_partition=group_rows[0]['training']['class_partition'])),flush=True)
    D.end_sequence()
    print(json.dumps(dict(method_id=METHOD_ID,settings=len(keys),mode=args.mode,
                          weights_saved=False,panels_saved=False,caches_saved=False)),flush=True)


if __name__=='__main__':
    main()
