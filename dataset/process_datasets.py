import re
import os
import html
import json
import numpy as np
import yaml
import argparse
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
from datasets import load_dataset
from utils import load_config, get_saved_emb_path
from datetime import datetime

DATASET_SPLITS = ['train', 'valid', 'test', 'valid_in_sample', 'valid_cold_start', 'test_in_sample', 'test_cold_start']

def list_to_str(l): 
    if isinstance(l, list):
        return list_to_str(', '.join(l))
    else:
        return l

def clean_text(raw_text): 
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text=re.sub(r'[^\x00-\x7F]', ' ', text)
    return text

def feature_process(feature): 
    sentence = ""
    if isinstance(feature, float):
        sentence += str(feature)
        sentence += '.'
    elif isinstance(feature, list) and len(feature) > 0:
        for v in feature:
            sentence += clean_text(v)
            sentence += ', '
        sentence = sentence[:-2]
        sentence += '.'
    else:
        sentence = clean_text(feature)
    return sentence + ' '

def clean_metadata(example):
    meta_text = ''
    features_needed = ['title', 'features', 'categories', 'description']
    for feature in features_needed:
        meta_text += feature_process(example[feature])
    example['cleaned_metadata'] = meta_text
    return example

def get_attribute_Amazon(meta_dataset, data_maps, attribute_core):

    attributes = defaultdict(int)
    for iid, cates in zip(meta_dataset['parent_asin'], meta_dataset['categories']):
        for cate in cates[1:]: # delete the main category
            attributes[cate] +=1

    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    for iid, cates in zip(meta_dataset['parent_asin'], meta_dataset['categories']):
        new_meta[iid] = []

        for cate in cates[1:]:
            if attributes[cate] >= attribute_core:
                new_meta[iid].append(cate)

    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []

    for iid, attributes in new_meta.items():
        item_id = data_maps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
    print(f'before delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, Avg.:{np.mean(attribute_lens):.4f}')
    
    data_maps['attribute2id'] = attribute2id
    data_maps['id2attribute'] = id2attribute
    data_maps['attributeid2num'] = attributeid2num
    data_maps['item2attributes'] = items2attributes
    return len(attribute2id), np.mean(attribute_lens), data_maps

def process_meta(config, data_maps, require_attributes):
    domain = config['dataset']

    meta_dataset = load_dataset(
        'McAuley-Lab/Amazon-Reviews-2023',
        f'raw_meta_{domain}',
        split='full',
        trust_remote_code=True
    )

    meta_dataset = meta_dataset.filter(lambda t: t['parent_asin'] in data_maps['item2id'])
    print(f'{len(meta_dataset)} of {len(data_maps["item2id"]) - 1} items have meta data.')

    meta_dataset = meta_dataset.map(
        clean_metadata,
        num_proc=config['num_workers']
    )
    
    if require_attributes:
        print('Begin extracting item attributes ...')
        attribute_num, avg_attribute, data_maps = get_attribute_Amazon(meta_dataset, data_maps, attribute_core=0)
        print(f'{attribute_num} different item attributes, average number of attributes per item: {avg_attribute}')

    id2meta = {0: '[PAD]'}
    for parent_asin, cleaned_metadata in zip(meta_dataset['parent_asin'], meta_dataset['cleaned_metadata']):
        item_id = data_maps['item2id'][parent_asin]
        id2meta[item_id] = cleaned_metadata

    data_maps['id2meta'] = id2meta

    return data_maps

def generate_sentence_embedding(data_maps, config, device):
    sentence_emb_model = SentenceTransformer('sentence-transformers/sentence-t5-base').to(device)
    sorted_text = []    # 1-base, sorted_text[0] -> item_id=1
    for i in range(1, len(data_maps['item2id'])):
        sorted_text.append(data_maps['id2meta'][str(i)])
    embeddings = sentence_emb_model.encode(sorted_text, convert_to_numpy=True, batch_size=512, show_progress_bar=True, device=device)
    
    print('embeddings', embeddings)
    # Add a padding row to the embeddings matrix
    padding_row = np.zeros((1, embeddings.shape[1]), dtype=embeddings.dtype)
    padded_embeddings = np.concatenate((padding_row, embeddings), axis=0)
    padded_embeddings.tofile(get_saved_emb_path(config['dataset']))

def remap_id(config, datasets, sorted_items):
    user2id = {'[PAD]': 0}
    id2user = ['[PAD]']
    item2id = {'[PAD]': 0}
    id2item = ['[PAD]']
    
    for i in range(len(sorted_items)):
        item, time = sorted_items[i]
        item2id[item] = i + 1
        id2item.append(item)
    
    for split in ['train', 'valid', 'test']:
        ds = datasets[split]
        for user, item in zip(ds['user_id'], ds['parent_asin']):
            if user not in user2id:
                user2id[user] = len(id2user)
                id2user.append(user)
    
    for split in ['train', 'valid', 'test']:
        datasets[split] = datasets[split].map(
            lambda t: {
                'user_id': user2id[t['user_id']],
                'item_id': item2id[t['parent_asin']],
                'history': ' '.join([str(item2id[item]) for item in t['history'].split(' ')]),
            },
            num_proc=config['num_workers']
        )
    
    data_maps = {'user2id': user2id, 'id2user': id2user, 'item2id': item2id, 'id2item': id2item}

    return datasets, data_maps

from rich.console import Console
from rich.table import Table

def format_variable_name(variable_name):
    # Convert snake_case to Title Case
    return ' '.join(word.capitalize() for word in variable_name.split('_'))

def print_dataset_info_to_console(dataset_info):
    console = Console()

    # Create a table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Dataset Info", style="dim")
    table.add_column("Value")

    # Add rows to the table in sections
    sections = [
        ("Item Count", ["train_item_count", "valid_item_count", "test_item_count"]),
        ("Timestamp Range", ["train_timestamp_min", "train_timestamp_max", "valid_timestamp_min", "valid_timestamp_max", "test_timestamp_min", "test_timestamp_max"]),
        ("Item IDs", ["in_sample_item_ids", "unseen_item_ids"]),
        ("Counts", ["user_count", "item_count"]),
        ("Dataset Sizes", ["train_size", "valid_size", "test_size", "valid_in_sample_size", "valid_cold_start_size", "test_in_sample_size", "test_cold_start_size"]),
    ]

    for section_name, keys in sections:
        table.add_row(f"[bold]{section_name}[/bold]", "", end_section=True)  # Section header

        for key in keys:
            if key in dataset_info:
                value = dataset_info[key]
                if isinstance(value, tuple):
                    value_str = f"{value[0]} to {value[1]}"
                elif isinstance(value, float):
                    value_str = f"{value:.2f}"
                elif isinstance(value, int) and "timestamp" in key:
                    # Convert timestamp to a readable date without time
                    value_str = datetime.fromtimestamp(value).strftime('%Y-%m-%d')
                else:
                    value_str = str(value)

                table.add_row(format_variable_name(key), value_str)
        
        table.add_row("", "", end_section=True)  # Add separator after each section

    # Print the table to the console
    console.print(table)


def process_raw_dataset_by_timestamp(config_path, require_attributes=False):
    config = load_config(config_path)
    
    print(config)
    
    domain = config['dataset']
    dataset_info = {}

    if all(os.path.exists(f'dataset/{domain}/{domain}.{split}.csv') for split in DATASET_SPLITS):
        print(f'All required splits found in dataset/{domain}')
        return
    
    print('Process raw dataset from scratch')
    
    # Load the dataset
    dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"5core_timestamp_w_his_{domain}",
        trust_remote_code=True
    )

    # Extract item2date mapping before processing the dataset
    item2startdate = {}
    for split in ['train', 'valid', 'test']:
        ds = dataset[split].sort('timestamp').map(
            lambda t: {'timestamp': int(t['timestamp'])}, num_proc=config['num_workers'])
        for user_id, item_id, timestamp in zip(ds['user_id'], ds['parent_asin'], ds['timestamp']):
            if item_id in item2startdate:
                if timestamp < item2startdate[item_id]:
                    item2startdate[item_id] = timestamp // 1000
            else:
                item2startdate[item_id] = timestamp // 1000

        dataset_info[f'{split}_item_count'] = len(item2startdate)
        if split == 'train':
            unseen_start_index = len(item2startdate)
            config['unseen_start_index'] = unseen_start_index
        if split == 'valid':
            config['test_start_index'] = len(item2startdate)

    # Write the updated data back to the file
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)
    
    for split in ['train', 'valid', 'test']:
        dates = np.array(dataset[split]['timestamp'], dtype=int) // 1000
        dataset_info[f'{split}_timestamp_range'] = (dates.min(), dates.max())
        dataset_info[f'{split}_timestamp_min'] = datetime.fromtimestamp(dates.min()).strftime('%Y-%m-%d %H:%M:%S')
        dataset_info[f'{split}_timestamp_max'] = datetime.fromtimestamp(dates.max()).strftime('%Y-%m-%d %H:%M:%S')

    last_seen_timestamp = np.array(dataset['train']['timestamp'], dtype=int).max() // 1000
    dataset_info['last_seen_timestamp'] = last_seen_timestamp

    # Apply Filtering and History Window
    datasets = {}
    for split in ['train', 'valid', 'test']:
        filtered_dataset = dataset[split].filter(lambda t: len(t['history']) > 0)
        truncated_dataset = filtered_dataset.map(
            lambda t: {
                'history': ' '.join(t['history'].split(' ')[-config['max_history_len']:])
            },
            num_proc=config['num_workers']
        )
        datasets[split] = truncated_dataset

    # Checking that data filtering doesn't remove any item in the item pool
    for split in ['train', 'valid', 'test']:
        ds = datasets[split]
        for parent_asin, history in zip(ds['parent_asin'], ds['history']):
            items_in_history = history.split(' ')
            for item in items_in_history:
                assert item in item2startdate

    selected_item_date = {}
    for split in ['train', 'valid', 'test']:
        ds = datasets[split]
        for parent_asin, history in zip(ds['parent_asin'], ds['history']):
            items_in_history = history.split(' ')
            if parent_asin not in selected_item_date:
                selected_item_date[parent_asin] = item2startdate[parent_asin]
            for item in items_in_history:
                if item not in selected_item_date:
                    selected_item_date[item] = item2startdate[item]
    
    assert len(selected_item_date) == len(item2startdate)
    
    # Sort the dictionary by date
    sorted_items = sorted(selected_item_date.items(), key=lambda x: x[1])
    dates = [value for key, value in sorted_items]
        
    dataset_info['in_sample_item_ids'] = (1, unseen_start_index)
    dataset_info['unseen_item_ids'] = (unseen_start_index + 1, len(sorted_items))
    
    # Remap IDs
    datasets, data_maps = remap_id(config, datasets, sorted_items)
    dataset_info['user_count'] = len(data_maps['user2id'])
    dataset_info['item_count'] = len(data_maps['item2id'])
    
    datasets['valid_in_sample'] = datasets['valid'].filter(lambda t: t['item_id'] <= unseen_start_index)
    datasets['valid_cold_start'] = datasets['valid'].filter(lambda t: t['item_id'] > unseen_start_index)
    datasets['test_in_sample'] = datasets['test'].filter(lambda t: t['item_id'] <= unseen_start_index)
    datasets['test_cold_start'] = datasets['test'].filter(lambda t: t['item_id'] > unseen_start_index)
    
    # Update dataset sizes in the dictionary
    dataset_info['train_size'] = len(datasets['train'])
    dataset_info['valid_size'] = len(datasets['valid'])
    dataset_info['test_size'] = len(datasets['test'])
    dataset_info['valid_in_sample_size'] = len(datasets['valid_in_sample'])
    dataset_info['valid_cold_start_size'] = len(datasets['valid_cold_start'])
    dataset_info['test_in_sample_size'] = len(datasets['test_in_sample'])
    dataset_info['test_cold_start_size'] = len(datasets['test_cold_start'])

    # Process metadata
    data_maps = process_meta(config, data_maps, require_attributes)

    # Save interaction files
    for split in DATASET_SPLITS:
        dataset = datasets[split]
        output_dir = f'dataset/{domain}'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{domain}.{split}.csv')
        with open(output_path, 'w') as f:
            f.write('user_id,history,item_id\n')
            for user_id, history, item_id in zip(dataset['user_id'], dataset['history'], dataset['item_id']):
                f.write(f"{user_id},{history},{item_id}\n")

    # Save data maps
    output_path = f'dataset/{domain}/{domain}.data_maps'
    with open(output_path, 'w') as f:
        json.dump(data_maps, f)

    print_dataset_info_to_console(dataset_info)

def process_datasets(config_path, device, require_attributes=False):
    
    config = load_config(config_path)
    domain = config['dataset']
    
    process_raw_dataset_by_timestamp(config_path, require_attributes)
    
    with open(f'dataset/{domain}/{domain}.data_maps', 'r') as f:
        data_maps = json.load(f)

    if not os.path.exists(f'dataset/{domain}/{domain}.sent_emb'):
        print('Generate sentence embeddings')
        generate_sentence_embedding(data_maps, config, device)
    else:
        print(f'Sentence embeddings found at dataset/{domain}/{domain}.sent_emb')
    
    print('Data Processing Complete.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/Beauty7_new.yaml', help='Path to the config file.')
    parser.add_argument('--device', type=int, default=0, help='GPU ID to use for training. Default is 0 (cuda:0).')
    parser.add_argument("-a", "--require_attributes", action="store_true", help="If set, require to extract attributes mappings")
    args = parser.parse_args()
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    process_datasets(args.config, args.device, args.require_attributes)