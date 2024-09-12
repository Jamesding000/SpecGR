import numpy as np
import json
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
import yaml
import re
import random
from datasets import load_dataset
import sys
import os

def load_yaml(yaml_file):
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    
    with open(yaml_file, 'r') as f:
        config = yaml.load(f, Loader=loader)
    
    return config

def update_config(base_config, hyper_params):
    config = base_config.copy()
    for key, value in hyper_params.items():
        keys = key.split('/')
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return config

def load_config(config_path, base_config_path=None):
    base_config_path = base_config_path or 'configs/base.yaml'
    base_config = load_yaml(base_config_path)
    exp_config = load_yaml(config_path)
    return update_config(base_config, exp_config)

def load_dataset_splits(domain, splits):
    data_files = {split: f'{domain}.{split}.csv' for split in splits}
    return load_dataset(f'dataset/{domain}', data_files=data_files)

def unique(x, dim=0):
    unique, inverse, counts = torch.unique(x, dim=dim, sorted=True, return_inverse=True, return_counts=True)
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return unique, index

def torch_in(query_tensor, reference_tensor):
    query_tensor = query_tensor.unsqueeze(1)
    reference_tensor = reference_tensor.unsqueeze(0)
    matches = (query_tensor == reference_tensor).all(dim=2)
    return matches.any(dim=1)

def safe_topk(tensor, k, dim=-1):
    if tensor.numel() == 0:
        return tensor, torch.empty(0, dtype=torch.long).to(tensor.device)
    if k > tensor.size(dim):
        return torch.sort(tensor, dim=dim, descending=True)
    else:
        return torch.topk(tensor, k, dim=dim)

def repeat_interleave_with_expand(tensor, num_repeats, dim=0):
    # Add a new dimension at the specified dimension
    tensor = tensor.unsqueeze(dim + 1)

    # Expand this new dimension to the number of repeats
    expanded_tensor = tensor.expand(*tensor.size()[:dim + 1], num_repeats, *tensor.size()[dim + 2:])
    
    # Flatten the new dimension into the original specified dimension
    expanded_tensor = expanded_tensor.reshape(-1, *tensor.size()[dim + 2:])
    
    return expanded_tensor

def gather_indicies(output, gather_index):
    gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
    output_tensor = output.gather(dim=1, index=gather_index)
    return output_tensor.squeeze(1)

def load_semantic_ids(config, saved_id_path=None):
    domain = config['dataset']
    exp_id = config['exp_id']
    saved_id_path = saved_id_path or get_saved_id_path(domain, exp_id)
    
    semantic_ids = np.fromfile(saved_id_path, dtype=np.int64).reshape(-1, config['RQ-VAE']['num_layers']+1)

    if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
        print('Semantic ids loaded from:', saved_id_path)
        print(semantic_ids)
    
    return semantic_ids

def load_item_embeddings(config, saved_emb_path=None):
    domain = config['dataset']
    saved_emb_path = saved_emb_path or get_saved_emb_path(domain)
    
    item_embeddings = np.fromfile(saved_emb_path, dtype=np.float32).reshape(-1, config['RQ-VAE']['sent_emb_dim'])

    if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
        print('Item embeddings loaded from:', saved_emb_path)
        print(item_embeddings.round(4))
        print(item_embeddings.shape)
    
    return item_embeddings

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_devices(devices_str):
    try:
        devices = json.loads(devices_str)
    except json.JSONDecodeError:
        devices = int(devices_str)
    return devices

def get_model_ckpt_path(model_name, domain, exp_id, suffix=''):
    model_ckpt_dir = os.path.join('results', model_name)
    os.makedirs(model_ckpt_dir, exist_ok=True)
    model_ckpt_file = f"{domain}{exp_id}_best{suffix}.pt"
    return os.path.join(model_ckpt_dir, model_ckpt_file)

def get_logfile_path(model_name, exp_id, suffix=''):
    log_dir = os.path.join('logs', model_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"exp{exp_id}{suffix}"
    return os.path.join(log_dir, log_file)

def get_saved_id_path(domain, exp_id):
    saved_id_dir = 'semantic_ids'
    os.makedirs(saved_id_dir, exist_ok=True)
    saved_id_file = f"{domain}{exp_id}.semantic_id"
    return os.path.join(saved_id_dir, saved_id_file)

def get_saved_emb_path(domain):
    saved_emb_dir = os.path.join('dataset', domain)
    os.makedirs(saved_emb_dir, exist_ok=True)
    saved_emb_file = f'{domain}.sent_emb'
    return os.path.join(saved_emb_dir, saved_emb_file)

def get_model(model_name):
    pass

def get_tokenizer(model_name):
    pass

def get_evaluator(model_name):
    pass