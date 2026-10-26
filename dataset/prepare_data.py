"""Download/convert numerical graphs and verify the released tensor fingerprints."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parent
NAMES=['Cora','CiteSeer','CoraFull','Cora-ML','ogbn-arxiv']
MANIFEST={'Cora': {'path': 'Cora/Cora.pt',
          'nodes': 2708,
          'features': 1433,
          'stored_edges': 10556,
          'classes': 7,
          'tensors': {'x': {'shape': [2708, 1433],
                            'dtype': 'torch.float32',
                            'sha256': 'f0faab5177bcc12f5688f042c8e0ed24ffb9baa8efc3ae7cde440d42524c9075'},
                      'edge_index': {'shape': [2, 10556],
                                     'dtype': 'torch.int64',
                                     'sha256': '299cb541efef8bc7bdf70e07e44b206c5ce6419ab16e878cb585acd5a695b459'},
                      'y': {'shape': [2708],
                            'dtype': 'torch.int64',
                            'sha256': 'd0018d7ffd314eb25a4904973f2c7eb1b7be135d668de42d453714724f8e9d03'}}},
 'CiteSeer': {'path': 'CiteSeer/CiteSeer.pt',
              'nodes': 3327,
              'features': 3703,
              'stored_edges': 9104,
              'classes': 6,
              'tensors': {'x': {'shape': [3327, 3703],
                                'dtype': 'torch.float32',
                                'sha256': 'cb9a425333dd8d1ae650e0eb4f3d65b14920aa6d5f0d4efa56d60f18f8d8c600'},
                          'edge_index': {'shape': [2, 9104],
                                         'dtype': 'torch.int64',
                                         'sha256': '0826c26dc48439ab46d52ca6998839846b8d67562bb25aab06b6fea1b01e020c'},
                          'y': {'shape': [3327],
                                'dtype': 'torch.int64',
                                'sha256': '448f8c04e02516b2784cfd3c5d82049f84ea77d6ef94c65ee19f8ea8b792b0fa'}}},
 'CoraFull': {'path': 'CoraFull/CoraFull.pt',
              'nodes': 19793,
              'features': 8710,
              'stored_edges': 126842,
              'classes': 70,
              'tensors': {'x': {'shape': [19793, 8710],
                                'dtype': 'torch.float32',
                                'sha256': '77a2c6ee719cc52811549c08ffb120f5469d976b1ae21f3ec38058603cf38685'},
                          'edge_index': {'shape': [2, 126842],
                                         'dtype': 'torch.int64',
                                         'sha256': 'c88426defea354deb76c98008dfc3a6eb3e1c73d533ffd7ceff166dced090267'},
                          'y': {'shape': [19793],
                                'dtype': 'torch.int64',
                                'sha256': 'c0337b253b331a78e7613a8b8038867d9fae42d6c6cec44b3ec55b05b82e4991'}}},
 'Cora-ML': {'path': 'Cora_ML/Cora-ML.pt',
             'nodes': 2995,
             'features': 2879,
             'stored_edges': 16316,
             'classes': 7,
             'tensors': {'x': {'shape': [2995, 2879],
                               'dtype': 'torch.float32',
                               'sha256': '3651f9330dec4dfb0794c5ec64c9da40f3ef846986dda697f1101d4b204069bc'},
                         'edge_index': {'shape': [2, 16316],
                                        'dtype': 'torch.int64',
                                        'sha256': '6fb8f7acfeec68ee98358467afd4a53d24c5b2c882bbbdddca1b7174e27362b4'},
                         'y': {'shape': [2995],
                               'dtype': 'torch.int64',
                               'sha256': 'a9f950be5894ac3a1dd69108dfc843dda81157569b0a44a515e744bc1b6e8eb7'}}},
 'ogbn-arxiv': {'path': 'ogbn-arxiv/ogbn-arxiv.pt',
                'nodes': 169343,
                'features': 128,
                'stored_edges': 1166243,
                'classes': 40,
                'tensors': {'x': {'shape': [169343, 128],
                                  'dtype': 'torch.float32',
                                  'sha256': '40adbe7fe3b8762a41a4a061dd3709434867fb80c81800038d4300eb17a62a9c'},
                            'edge_index': {'shape': [2, 1166243],
                                           'dtype': 'torch.int64',
                                           'sha256': 'f6e1bed505b2ca37423abbcd46fa012f73049676c958002f9de100bc7889502c'},
                            'y': {'shape': [169343],
                                  'dtype': 'torch.int64',
                                  'sha256': '32135eb09ea9429084c5f84b79ebf8fc4c044ac74e630e941210450c3cfe25b6'}}}}


def tensors(raw):
    import torch
    result={k:raw[k].detach().cpu() for k in ['x','edge_index','y']}
    if result['x'].layout!=torch.strided:result['x']=result['x'].to_dense()
    result['x']=result['x'].float().contiguous()
    result['edge_index']=result['edge_index'].long().contiguous()
    result['y']=result['y'].long().reshape(-1).contiguous()
    x,edge,y=(result[k] for k in ['x','edge_index','y'])
    if x.ndim!=2 or edge.ndim!=2 or edge.shape[0]!=2 or len(x)!=len(y):
        raise ValueError('Graph fields are not node-aligned')
    if not torch.isfinite(x).all() or (edge.numel() and (edge.min()<0 or edge.max()>=len(x))):
        raise ValueError('Nonfinite attributes or invalid edge endpoints')
    return result


def verify(graph,expected):
    import torch
    observed=dict(nodes=len(graph['x']),features=graph['x'].shape[1],stored_edges=graph['edge_index'].shape[1],
                  classes=int(torch.unique(graph['y']).numel()))
    for name,value in observed.items():
        if value!=expected[name]:raise ValueError(f'{name}: expected {expected[name]}, got {value}')
    for name,value in graph.items():
        ref=expected['tensors'][name]
        if list(value.shape)!=ref['shape'] or str(value.dtype)!=ref['dtype']:
            raise ValueError(f'{name}: tensor shape or dtype mismatch')
        digest=hashlib.sha256(value.numpy().tobytes()).hexdigest()
        if digest!=ref['sha256']:
            raise ValueError(f'{name}: fingerprint mismatch; check dataset version, preprocessing and node order')
    return observed


def download(name,cache):
    if name=='ogbn-arxiv':
        import torch
        from ogb.nodeproppred import PygNodePropPredDataset
        from torch_geometric.data import Data
        from torch_geometric.data.data import DataEdgeAttr,DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage,NodeStorage,EdgeStorage
        # OGB 1.3.6 uses torch.load without an explicit weights_only argument.
        # Allow only the PyG graph classes required by its processed cache.
        with torch.serialization.safe_globals([Data,DataEdgeAttr,DataTensorAttr,GlobalStorage,NodeStorage,EdgeStorage]):
            return PygNodePropPredDataset(name='ogbn-arxiv',root=str(cache))[0]
    from torch_geometric.datasets import Planetoid,CoraFull,CitationFull
    if name in ['Cora','CiteSeer']:return Planetoid(root=str(cache/name),name=name)[0]
    if name=='CoraFull':return CoraFull(root=str(cache/name))[0]
    return CitationFull(root=str(cache/name),name='Cora_ML')[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',nargs='+',choices=NAMES,default=NAMES)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--source-file',type=Path,help='Convert one existing trusted tensor dictionary or PyG Data file')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    if args.source_file and len(args.dataset)!=1:parser.error('--source-file requires exactly one dataset')
    if args.verify_only and (args.source_file or args.overwrite):parser.error('--verify-only cannot convert or overwrite')
    import torch
    manifest=MANIFEST
    for name in args.dataset:
        expected=manifest[name];destination=args.root/expected['path']
        if destination.exists() and not args.overwrite:
            graph=tensors(torch.load(destination,map_location='cpu',weights_only=False))
            print(json.dumps(dict(dataset=name,status='verified_existing',**verify(graph,expected))))
            continue
        if args.verify_only:raise FileNotFoundError(f'Missing graph: {destination}')
        raw=torch.load(args.source_file,map_location='cpu',weights_only=False) if args.source_file else download(name,args.root/'.downloads')
        graph=tensors(raw);observed=verify(graph,expected)
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix('.tmp')
        torch.save(graph,temporary);temporary.replace(destination)
        print(json.dumps(dict(dataset=name,status='prepared_and_verified',**observed)))


if __name__=='__main__':main()
