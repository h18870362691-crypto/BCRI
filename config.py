"""BCRI config."""
import copy

# config
DATASETS = {
    'Cora': dict(class_split=(3, 2, 2), dropout=0.1, patience=5, shots=(1, 3, 5),
                 graph='Cora/Cora.pt', default_way=2),
    'CiteSeer': dict(class_split=(2, 2, 2), dropout=0.5, patience=5, shots=(1, 3, 5),
                     graph='CiteSeer/CiteSeer.pt', default_way=2),
    'CoraFull': dict(class_split=(40, 15, 15), dropout=0.1, patience=20, ways=(5, 10),
                    graph='CoraFull/CoraFull.pt', default_way=5),
    'Cora-ML': dict(class_split=(3, 2, 2), dropout=0.4, patience=5, shots=(1, 3, 5),
                   graph='Cora_ML/Cora-ML.pt', default_way=2),
    'ogbn-arxiv': dict(class_split=(20, 10, 10), dropout=0.3, patience=20, ways=(5, 10),
                      graph='ogbn-arxiv/ogbn-arxiv.pt', default_way=5),
}
ALIASES = {name.casefold().replace('_', '-'): name for name in DATASETS}
ALIASES.update({'citeseer': 'CiteSeer', 'coraml': 'Cora-ML', 'cora-ml': 'Cora-ML',
                'cora-full': 'CoraFull', 'arxiv': 'ogbn-arxiv'})
SHOTS = (3, 5)
DEFAULT_SEED = 1234
REPEATS = 5
STD_DDOF = 0
TEST_EPISODES = 100
EVALUATION_PROTOCOL = 'vision-style-single-seed-continuous-v1'
SEEDS = (DEFAULT_SEED,)


def canonical_dataset(name):
    try:
        return ALIASES[name.casefold().replace('_', '-')]
    except KeyError:
        raise ValueError('Unsupported dataset: ' + str(name)) from None


def dataset_config(name):
    name = canonical_dataset(name)
    result = copy.deepcopy(DATASETS[name])
    result.update(dataset=name, hidden_dim=256, heads=4, layers=2,
                  queries_per_class=10, max_neighbors=30,
                  reference='VISION dataset splits/dropout/patience; BCRI way defaults are experiment choices')
    return result


def training_config(name):
    profile = dataset_config(name)
    # BCRI epochs contain 200 pseudo tasks; they are not VISION's single-task steps.
    return dict(lr=0.001, max_epochs=50, patience=profile['patience'],
                validate_every=1, validation_tasks=100, min_delta=0.0,
                tasks=200, batch_size=8, weight_decay=0.0001, gradient_clip=1.0)


def pretraining_config(name):
    profile = dataset_config(name)
    return dict(lr=0.0002, max_steps=10000, validate_every=20,
                validation_tasks=100, patience=profile['patience'], min_delta=0.0,
                weight_decay=0.0001, gradient_clip=1.0, label_smoothing=0.1,
                contrast_weight=0.5, contrast_temperature=0.1,
                edge_drop=0.2, input_noise=0.02, pool_size=4096,
                resample_neighbors_every=0 if profile['dataset']=='ogbn-arxiv' else 50)


def validate_task(name, way, shot, queries=10):
    profile = dataset_config(name)
    if not 2 <= way <= min(profile['class_split'][1:]):
        raise ValueError(f"{profile['dataset']} requires 2 <= way <= {min(profile['class_split'][1:])}")
    if shot < 1 or queries < 1:
        raise ValueError('BCRI requires shot >= 1 and positive queries per class')
    return profile


def planned_settings(seed=DEFAULT_SEED, repeats=REPEATS):
    if repeats < 1:
        raise ValueError('Repeat count must be positive')
    return [dict(key=f"{name}/{way}way_{shot}shot_seed_{seed}_repeat_{repeat}",
                 group_key=f"{name}/{way}way_{shot}shot_seed_{seed}",
                 dataset=name, way=way, shot=shot, seed=seed, repeat=repeat,
                 test_episodes=TEST_EPISODES,
                 evaluation_protocol=EVALUATION_PROTOCOL,
                 pretraining=pretraining_config(name), training=training_config(name), profile=dataset_config(name))
            for name, p in DATASETS.items() for way in p.get('ways', (p['default_way'],))
            for shot in p.get('shots', SHOTS) for repeat in range(1,repeats+1)]
