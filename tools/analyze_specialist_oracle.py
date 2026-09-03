"""Meta-Album specialist/oracle and paired-checkpoint analysis.

This tool intentionally does not depend on or modify analyze_anchor_response.py.
It evaluates two YAML-defined specialist checkpoints and one reference
checkpoint on identical episodes, then runs a class-balanced query-split/
cross-fit analysis.  The optional paired-checkpoint mode evaluates exactly two
checkpoints on one shared set of single-domain episodes and records the sampled
class/image identities without running an oracle or bootstrap analysis.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from contextlib import nullcontext


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch import amp
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import yaml

import datasets
import models
import utils


def seed_everything(seed=0):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
  del worker_id
  worker_seed = torch.initial_seed() % (2 ** 32)
  random.seed(worker_seed)
  np.random.seed(worker_seed)


def capture_rng_state():
  state = {
    'python': random.getstate(),
    'numpy': np.random.get_state(),
    'torch': torch.get_rng_state(),
  }
  if torch.cuda.is_available():
    state['cuda'] = torch.cuda.get_rng_state_all()
  return state


def restore_rng_state(state):
  random.setstate(state['python'])
  np.random.set_state(state['numpy'])
  torch.set_rng_state(state['torch'])
  if 'cuda' in state:
    torch.cuda.set_rng_state_all(state['cuda'])


def resolve_project_path(path):
  path = os.path.expanduser(str(path))
  if not os.path.isabs(path):
    path = os.path.join(PROJECT_ROOT, path)
  return os.path.abspath(path)


def autocast_for(device, enabled, dtype=None):
  if device.type == 'cuda':
    return amp.autocast('cuda', enabled=enabled, dtype=dtype)
  return nullcontext()


def torch_load_checkpoint(path):
  try:
    return torch.load(path, map_location='cpu', weights_only=False)
  except TypeError:
    return torch.load(path, map_location='cpu')


def sha256_file(path, chunk_size=1024 * 1024):
  digest = hashlib.sha256()
  with open(path, 'rb') as f:
    while True:
      chunk = f.read(chunk_size)
      if not chunk:
        break
      digest.update(chunk)
  return digest.hexdigest()


def checkpoint_parameter_fingerprint(checkpoint):
  """Fingerprint model and optional gradient-transport parameters."""
  digest = hashlib.sha256()
  found = False
  for section in (
      'encoder_state_dict', 'classifier_state_dict',
      'gradient_transport_state_dict'):
    state_dict = checkpoint.get(section)
    if not isinstance(state_dict, dict):
      continue
    for name in sorted(state_dict):
      value = state_dict[name]
      if not torch.is_tensor(value):
        continue
      found = True
      tensor = value.detach().cpu().contiguous()
      digest.update(section.encode('utf-8'))
      digest.update(name.encode('utf-8'))
      digest.update(str(tensor.dtype).encode('ascii'))
      digest.update(str(tuple(tensor.shape)).encode('ascii'))
      digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
  if not found:
    raise ValueError(
      'checkpoint has no encoder/classifier tensors to fingerprint')
  return digest.hexdigest()


def name_slug(name):
  """Return a stable, CSV-safe spelling for a YAML model/cluster name."""
  slug = re.sub(r'[^0-9A-Za-z]+', '_', str(name)).strip('_').lower()
  if not slug:
    raise ValueError('model and cluster names must contain a letter or digit')
  return slug


def analysis_mode(config):
  mode = str(config.get('mode', 'specialist_oracle')).strip().lower()
  aliases = {
    'specialist-oracle': 'specialist_oracle',
    'paired': 'paired_checkpoints',
    'paired-checkpoint': 'paired_checkpoints',
    'paired-checkpoints': 'paired_checkpoints',
    'two_model': 'paired_checkpoints',
  }
  return aliases.get(mode, mode)


def paired_model_names(config):
  return tuple((config.get('models') or {}).keys())


def configured_model_names(config):
  if analysis_mode(config) == 'paired_checkpoints':
    return paired_model_names(config)
  return analysis_model_names(config)


def cluster_names(config):
  return tuple(config.get('clusters', {}).keys())


def specialist_model_by_cluster(config):
  return dict(config['model_roles']['specialists'])


def specialist_model_names(config):
  specialists = specialist_model_by_cluster(config)
  return tuple(specialists[cluster] for cluster in cluster_names(config))


def reference_model_name(config):
  return config['model_roles']['reference']


def analysis_model_names(config):
  return specialist_model_names(config) + (reference_model_name(config),)


def infer_model_roles(config):
  """Resolve explicit roles, or infer the legacy ordered three-model schema."""
  clusters = tuple(config.get('clusters', {}).keys())
  models_config = config.get('models') or {}
  model_names = tuple(models_config.keys())
  roles = dict(config.get('model_roles') or {})
  explicit_specialists = roles.get('specialists')
  explicit_reference = roles.get('reference')

  if explicit_specialists is None:
    missing = [name for name in clusters if name not in models_config]
    if missing:
      raise ValueError(
        'model_roles.specialists is required when cluster names do not match '
        'specialist model names; missing models for {}'.format(missing))
    specialists = {name: name for name in clusters}
  else:
    specialists = {
      str(cluster): str(model)
      for cluster, model in dict(explicit_specialists).items()
    }

  if explicit_reference is None:
    specialist_names = set(specialists.values())
    remaining = [name for name in model_names if name not in specialist_names]
    if len(remaining) != 1:
      raise ValueError(
        'model_roles.reference is required unless exactly one model remains '
        'after the specialist models')
    reference = remaining[0]
  else:
    reference = str(explicit_reference)
  return {'specialists': specialists, 'reference': reference}


def standard_csv_fields(config):
  fields = ['task_id', 'cluster', 'domain']
  for name in analysis_model_names(config):
    slug = name_slug(name)
    fields.extend([
      '{}_query_loss'.format(slug),
      '{}_query_accuracy'.format(slug),
    ])
  fields.extend([
    'loss_selected_specialist', 'specialist_loss_tie',
    'domain_specialist',
    'domain_router_query_loss', 'domain_router_query_accuracy',
    'oracle_query_loss', 'oracle_query_accuracy',
  ])
  return fields


def query_split_csv_fields(config):
  fields = ['task_id', 'cluster', 'domain']
  for name in analysis_model_names(config):
    slug = name_slug(name)
    for split in ('a', 'b'):
      fields.extend([
        '{}_{}_query_loss'.format(slug, split),
        '{}_{}_query_accuracy'.format(slug, split),
      ])
  reference_slug = name_slug(reference_model_name(config))
  fields.extend([
    'specialist_selected_on_a', 'specialist_selected_on_b',
    'a_to_b_query_loss', 'a_to_b_query_accuracy',
    'b_to_a_query_loss', 'b_to_a_query_accuracy',
    'cross_fitted_oracle_query_loss',
    'cross_fitted_oracle_query_accuracy',
    'cross_fitted_{}_query_loss'.format(reference_slug),
    'cross_fitted_{}_query_accuracy'.format(reference_slug),
    'in_sample_oracle_query_loss', 'in_sample_oracle_query_accuracy',
    '{}_split_oracle_query_loss'.format(reference_slug),
  ])
  return fields


def paired_csv_fields(config):
  fields = [
    'task_id', 'domain', 'class_combination', 'class_sample_mapping',
    'episode_sample_sha256',
  ]
  for name in paired_model_names(config):
    slug = name_slug(name)
    fields.extend([
      '{}_query_loss'.format(slug),
      '{}_query_accuracy'.format(slug),
    ])
  fields.extend(['query_loss_difference', 'query_accuracy_difference'])
  return fields


def standard_csv_float_fields(config):
  return tuple(
    field for field in standard_csv_fields(config)
    if field.endswith('_query_loss') or field.endswith('_query_accuracy'))


def normalized_config(config):
  config = dict(config)
  config['mode'] = analysis_mode(config)
  config['clusters'] = {
    str(name): [str(domain) for domain in domains]
    for name, domains in (config.get('clusters') or {}).items()
  }
  config['models'] = {
    str(name): dict(model_config)
    for name, model_config in (config.get('models') or {}).items()
  }
  if config['mode'] == 'specialist_oracle':
    config['model_roles'] = infer_model_roles(config)
  config['inner_args'] = utils.config_inner_args(
    dict(config.get('inner_args') or {}))
  return config


def require_equal(label, actual, expected):
  if actual != expected:
    raise ValueError('{} must be {!r}, got {!r}'.format(
      label, expected, actual))


def validate_specialist_oracle_config(config):
  require_equal('dataset', config.get('dataset'), 'meta-album')
  require_equal('seed', int(config.get('seed', 0)), 0)
  clusters = cluster_names(config)
  if len(clusters) != 2:
    raise ValueError(
      'specialist-oracle analysis requires exactly two YAML clusters, got {}'
      .format(list(clusters)))
  for cluster in clusters:
    domains = config['clusters'][cluster]
    if not domains:
      raise ValueError('clusters.{} must contain at least one domain'.format(
        cluster))
    if len(domains) != len(set(domains)):
      raise ValueError(
        'clusters.{} contains duplicate domains'.format(cluster))
  all_domains = [
    domain for cluster in clusters
    for domain in config['clusters'][cluster]
  ]
  if len(all_domains) != len(set(all_domains)):
    raise ValueError('a domain may belong to only one cluster')

  tasks_per_cluster = int(config.get('tasks_per_cluster', 500))
  if tasks_per_cluster <= 0:
    raise ValueError('tasks_per_cluster must be positive')

  test_config = config.get('test') or {}
  expected_test = {
    'split': 'meta-test',
    'n_way': 5,
    'n_shot': 1,
    'normalization': False,
    'transform': None,
  }
  for key, expected in expected_test.items():
    require_equal('test.{}'.format(key), test_config.get(key), expected)
  if int(test_config.get('n_episode', 0)) <= 0:
    raise ValueError('test.n_episode must be positive')
  if int(test_config.get('n_query', 0)) <= 0:
    raise ValueError('test.n_query must be positive')
  if int(test_config.get('image_size', 0)) <= 0:
    raise ValueError('test.image_size must be positive')

  query_split = config.get('query_split') or {}
  require_equal('query_split.enabled', query_split.get('enabled'), True)
  split_n_query = int(query_split.get('n_query', 0))
  split_query = int(query_split.get('split_query', 0))
  if split_query <= 0 or split_n_query != 2 * split_query:
    raise ValueError(
      'query_split requires positive n_query == 2 * split_query')

  require_equal('inner_args.reset_classifier',
                config['inner_args'].get('reset_classifier'), False)
  if int(config['inner_args'].get('n_step', 0)) <= 0:
    raise ValueError('inner_args.n_step must be positive')
  if not isinstance(config['inner_args'].get('first_order'), bool):
    raise ValueError('inner_args.first_order must be a boolean')
  for key in ('encoder_lr', 'classifier_lr'):
    if float(config['inner_args'].get(key, 0.0)) < 0.0:
      raise ValueError('inner_args.{} must be non-negative'.format(key))
  if 'bn' not in config['inner_args'].get('frozen', []):
    raise ValueError("inner_args.frozen must include 'bn'")

  roles = config.get('model_roles') or {}
  specialists = dict(roles.get('specialists') or {})
  if set(specialists) != set(clusters):
    raise ValueError(
      'model_roles.specialists must map exactly the two YAML clusters; got {}'
      .format(list(specialists)))
  role_model_names = analysis_model_names(config)
  if len(set(role_model_names)) != 3:
    raise ValueError('the two specialists and reference must be distinct')
  if set(config.get('models', {})) != set(role_model_names):
    raise ValueError(
      'models must contain exactly the models named by model_roles; got {} '
      'but roles require {}'.format(
        list(config.get('models', {})), list(role_model_names)))
  slugs = [name_slug(name) for name in role_model_names]
  if len(set(slugs)) != len(slugs):
    raise ValueError('model names must have distinct CSV-safe spellings')
  for name in role_model_names:
    model_config = config['models'][name]
    if not model_config.get('load'):
      raise ValueError('models.{}.load is required'.format(name))
    require_equal('models.{}.use_gradient_transport'.format(name),
                  model_config.get('use_gradient_transport'), False)

  manifest = config.get('class_split_manifest')
  if not manifest:
    raise ValueError('class_split_manifest is required')
  test_manifest = test_config.get('class_split_manifest', manifest)
  if resolve_project_path(test_manifest) != resolve_project_path(manifest):
    raise ValueError(
      'test.class_split_manifest must match class_split_manifest')

  bootstrap = config.get('bootstrap') or {}
  if int(bootstrap.get('n_resamples', 10000)) <= 0:
    raise ValueError('bootstrap.n_resamples must be positive')
  confidence = float(bootstrap.get('confidence', 0.95))
  require_equal('bootstrap.confidence', confidence, 0.95)


def validate_paired_config(config):
  require_equal('dataset', config.get('dataset'), 'meta-album')
  domain = config.get('domain')
  if not isinstance(domain, str) or not domain.strip():
    raise ValueError('domain must name one Meta-Album domain')
  config['domain'] = domain.strip()

  if int(config.get('n_episodes', 0)) <= 0:
    raise ValueError('n_episodes must be positive')

  test_config = config.get('test') or {}
  require_equal('test.split', test_config.get('split'), 'meta-test')
  for key in ('image_size', 'n_way', 'n_shot', 'n_query', 'n_episode'):
    if int(test_config.get(key, 0)) <= 0:
      raise ValueError('test.{} must be positive'.format(key))

  models_config = config.get('models') or {}
  model_names = paired_model_names(config)
  if len(model_names) != 2:
    raise ValueError(
      'paired-checkpoint analysis requires exactly two YAML models, got {}'
      .format(list(model_names)))
  if len({name_slug(name) for name in model_names}) != 2:
    raise ValueError('paired model names must have distinct CSV-safe spellings')
  resolved_loads = []
  for name in model_names:
    model_config = models_config[name]
    if not isinstance(model_config, dict):
      raise ValueError('models.{} must be an object'.format(name))
    if not model_config.get('load'):
      raise ValueError('models.{}.load is required'.format(name))
    resolved_loads.append(resolve_project_path(model_config['load']))
    if 'use_gradient_transport' in model_config and not isinstance(
        model_config['use_gradient_transport'], bool):
      raise ValueError(
        'models.{}.use_gradient_transport must be a boolean'.format(name))
  if len(set(resolved_loads)) != 2:
    raise ValueError('paired model checkpoint paths must be distinct')

  manifest = config.get('class_split_manifest')
  if not manifest:
    raise ValueError('class_split_manifest is required')
  test_manifest = test_config.get('class_split_manifest', manifest)
  if resolve_project_path(test_manifest) != resolve_project_path(manifest):
    raise ValueError(
      'test.class_split_manifest must match class_split_manifest')

  inner_args = config['inner_args']
  if int(inner_args.get('n_step', -1)) < 0:
    raise ValueError('inner_args.n_step must be non-negative')
  if not isinstance(inner_args.get('reset_classifier'), bool):
    raise ValueError('inner_args.reset_classifier must be a boolean')
  if not isinstance(inner_args.get('first_order'), bool):
    raise ValueError('inner_args.first_order must be a boolean')
  for key in ('encoder_lr', 'classifier_lr'):
    if float(inner_args.get(key, 0.0)) < 0.0:
      raise ValueError('inner_args.{} must be non-negative'.format(key))

  amp_dtype_name = str(config.get('amp_dtype', 'float16')).lower()
  if amp_dtype_name not in ('float16', 'fp16', 'bfloat16', 'bf16'):
    raise ValueError(
      'amp_dtype must be one of: float16, fp16, bfloat16, bf16')


def validate_config(config):
  mode = analysis_mode(config)
  if mode == 'specialist_oracle':
    return validate_specialist_oracle_config(config)
  if mode == 'paired_checkpoints':
    return validate_paired_config(config)
  raise ValueError(
    "mode must be 'specialist_oracle' or 'paired_checkpoints', got {!r}"
    .format(mode))


def configured_domains(config):
  if analysis_mode(config) == 'paired_checkpoints':
    return (config['domain'],)
  return tuple(
    domain for cluster in cluster_names(config)
    for domain in config['clusters'][cluster])


def validate_manifest(config):
  manifest_path = resolve_project_path(config['class_split_manifest'])
  if not os.path.isfile(manifest_path):
    raise FileNotFoundError(
      'class split manifest not found: {}'.format(manifest_path))
  with open(manifest_path, 'r', encoding='utf-8') as f:
    manifest = json.load(f)
  if not isinstance(manifest, dict):
    raise ValueError('class split manifest must contain a JSON object')

  counts = {}
  for domain in configured_domains(config):
    if domain not in manifest:
      raise KeyError('manifest is missing domain {}'.format(domain))
    entry = manifest[domain]
    if not isinstance(entry, dict):
      raise ValueError('manifest entry {} must be an object'.format(domain))
    split_sets = {}
    for split in ('train', 'val', 'test'):
      values = entry.get(split)
      if not isinstance(values, list):
        raise ValueError(
          'manifest {}.{} must be a list'.format(domain, split))
      if len(values) != len(set(map(str, values))):
        raise ValueError(
          'manifest {}.{} contains duplicates'.format(domain, split))
      split_sets[split] = set(map(str, values))
    overlaps = {
      'train_val': split_sets['train'] & split_sets['val'],
      'train_test': split_sets['train'] & split_sets['test'],
      'val_test': split_sets['val'] & split_sets['test'],
    }
    nonempty_overlaps = {
      key: sorted(value) for key, value in overlaps.items() if value
    }
    if nonempty_overlaps:
      raise ValueError(
        'manifest {} has overlapping splits: {}'.format(
          domain, nonempty_overlaps))
    if len(split_sets['test']) < int(config['test']['n_way']):
      raise ValueError(
        'manifest {} test split has fewer than n_way classes'.format(
          domain))
    counts[domain] = {
      split: len(split_sets[split]) for split in ('train', 'val', 'test')
      }

  return {
    'path': manifest_path,
    'sha256': sha256_file(manifest_path),
    'requested_split': 'meta-test',
    'manifest_key_used': 'test',
    'train_val_test_disjoint': True,
    'all_requested_domains_present': True,
    'class_counts': counts,
  }


def balanced_task_specs(config, seed):
  tasks_per_cluster = int(config['tasks_per_cluster'])
  specs = []
  task_id = 0
  for cluster_index, cluster in enumerate(cluster_names(config)):
    domains = config['clusters'][cluster]
    base, remainder = divmod(tasks_per_cluster, len(domains))
    scheduled_domains = []
    for domain_index, domain in enumerate(domains):
      scheduled_domains.extend(
        [domain] * (base + int(domain_index < remainder)))
    random.Random(seed + 1009 * cluster_index).shuffle(scheduled_domains)
    for domain in scheduled_domains:
      specs.append({
        'task_id': task_id,
        'cluster': cluster,
        'domain': domain,
      })
      task_id += 1
  return specs


def paired_task_specs(config):
  return [
    {'task_id': task_id, 'domain': config['domain']}
    for task_id in range(int(config['n_episodes']))
  ]


def task_balance_report(specs):
  clusters = tuple(dict.fromkeys(spec['cluster'] for spec in specs))
  cluster_counts = Counter(spec['cluster'] for spec in specs)
  domain_counts = {
    cluster: Counter(
      spec['domain'] for spec in specs if spec['cluster'] == cluster)
    for cluster in clusters
  }
  for cluster in clusters:
    values = list(domain_counts[cluster].values())
    if not values or max(values) - min(values) > 1:
      raise RuntimeError(
        '{} task allocation is not maximally balanced'.format(cluster))
  return {
    'total_tasks': len(specs),
    'cluster_counts': dict(cluster_counts),
    'domain_counts': {
      cluster: dict(domain_counts[cluster]) for cluster in clusters
    },
    'max_within_cluster_domain_count_difference': {
      cluster: (
        max(domain_counts[cluster].values()) -
        min(domain_counts[cluster].values()))
      for cluster in clusters
    },
  }


def make_domain_test_config(config, domain, n_query, return_metadata=False):
  test_config = dict(config['test'])
  test_config['root_path'] = resolve_project_path(
    test_config.get('root_path') or config.get('data_root'))
  test_config['class_split_manifest'] = resolve_project_path(
    config['class_split_manifest'])
  test_config['split'] = 'meta-test'
  test_config['domains'] = [domain]
  test_config['n_query'] = int(n_query)
  test_config['n_batch'] = 1
  test_config['n_episode'] = 1
  test_config['return_metadata'] = bool(return_metadata)
  return test_config


class ScheduledSingleDomainEpisodes(Dataset):
  """Episode dataset whose task-to-domain schedule is explicit and balanced."""

  def __init__(self, config, task_specs, n_query):
    super().__init__()
    self.task_specs = list(task_specs)
    self.domain_datasets = {}
    all_domains = [
      domain for cluster in cluster_names(config)
      for domain in config['clusters'][cluster]
    ]
    for domain in all_domains:
      domain_config = make_domain_test_config(config, domain, n_query)
      dataset = datasets.make(config['dataset'], **domain_config)
      if dataset.domains != [domain]:
        raise RuntimeError(
          'task dataset for {} is not single-domain'.format(domain))
      if dataset.split != 'meta-test':
        raise RuntimeError(
          'task dataset for {} is not meta-test'.format(domain))
      if resolve_project_path(dataset.class_split_manifest) != \
          resolve_project_path(config['class_split_manifest']):
        raise RuntimeError(
          'task dataset for {} does not use the requested manifest'.format(
            domain))
      self.domain_datasets[domain] = dataset

  def __len__(self):
    return len(self.task_specs)

  def __getitem__(self, index):
    spec = self.task_specs[index]
    episode = self.domain_datasets[spec['domain']][0]
    return (
      spec['task_id'], spec['cluster'], spec['domain'],
      episode[0], episode[1], episode[2], episode[3])


class PairedSingleDomainEpisodes(Dataset):
  """Sample each scheduled single-domain episode once, with provenance."""

  def __init__(self, config, task_specs):
    super().__init__()
    self.task_specs = list(task_specs)
    domain = config['domain']
    domain_config = make_domain_test_config(
      config, domain, int(config['test']['n_query']), return_metadata=True)
    self.episode_dataset = datasets.make(config['dataset'], **domain_config)
    if self.episode_dataset.domains != [domain]:
      raise RuntimeError('paired task dataset is not single-domain')
    if not self.episode_dataset.return_metadata:
      raise RuntimeError('paired task dataset metadata is disabled')

  def __len__(self):
    return len(self.task_specs)

  def __getitem__(self, index):
    spec = self.task_specs[index]
    episode = self.episode_dataset[0]
    return (
      spec['task_id'], spec['domain'], episode[0], episode[1], episode[2],
      episode[3], episode[4])

  def validate_episode_metadata(self, metadata):
    dataset = self.episode_dataset
    if metadata.get('domain') != dataset.domains[0]:
      raise RuntimeError('episode metadata domain does not match dataset')
    class_names = tuple(metadata.get('class_names') or ())
    support = tuple(metadata.get('support_pickle_indices') or ())
    query = tuple(metadata.get('query_pickle_indices') or ())
    n_way = int(dataset.n_way)
    if not (len(class_names) == len(support) == len(query) == n_way):
      raise RuntimeError('episode metadata does not contain n_way classes')
    if len(set(class_names)) != n_way:
      raise RuntimeError('episode metadata class names are not unique')

    class_to_pool = {
      class_name: set(int(i) for i in indices)
      for class_name, indices in zip(
        dataset.domain_cat_keys[0], dataset.domain_catlocs[0])
    }
    for class_name, support_indices, query_indices in zip(
        class_names, support, query):
      if class_name not in class_to_pool:
        raise RuntimeError(
          'episode metadata contains unknown class {}'.format(class_name))
      support_indices = tuple(int(i) for i in support_indices)
      query_indices = tuple(int(i) for i in query_indices)
      if len(support_indices) != int(dataset.n_shot):
        raise RuntimeError('episode metadata support count is incorrect')
      if len(query_indices) != int(dataset.n_query):
        raise RuntimeError('episode metadata query count is incorrect')
      sampled = support_indices + query_indices
      if len(set(sampled)) != len(sampled):
        raise RuntimeError(
          'episode metadata repeats a support/query pickle index')
      if not set(sampled).issubset(class_to_pool[class_name]):
        raise RuntimeError(
          'episode metadata maps an index to the wrong class')
    return True


def collate_scheduled_episodes(items):
  task_ids = [item[0] for item in items]
  clusters = [item[1] for item in items]
  domains = [item[2] for item in items]
  episode_batch = datasets.collate_fn([
    (item[3], item[4], item[5], item[6]) for item in items
  ])
  return task_ids, clusters, domains, episode_batch


def collate_paired_episodes(items):
  task_ids = [item[0] for item in items]
  domains = [item[1] for item in items]
  episode_batch = datasets.collate_fn([
    (item[2], item[3], item[4], item[5]) for item in items
  ])
  metadata = [item[6] for item in items]
  return task_ids, domains, episode_batch, metadata


def make_loader(config, task_specs, n_query, seed, device):
  dataset = ScheduledSingleDomainEpisodes(config, task_specs, n_query)
  num_workers = int(config.get('num_workers', 0))
  generator = torch.Generator()
  generator.manual_seed(seed)
  loader_kwargs = {
    'dataset': dataset,
    'batch_size': int(config['test']['n_episode']),
    'shuffle': False,
    'collate_fn': collate_scheduled_episodes,
    'num_workers': num_workers,
    'pin_memory': device.type == 'cuda',
    'worker_init_fn': seed_worker,
    'generator': generator,
  }
  if num_workers > 0:
    loader_kwargs['prefetch_factor'] = int(config.get('prefetch_factor', 2))
    loader_kwargs['persistent_workers'] = bool(
      config.get('persistent_workers', True))
  return DataLoader(**loader_kwargs)


def make_paired_loader(config, task_specs, seed, device):
  dataset = PairedSingleDomainEpisodes(config, task_specs)
  num_workers = int(config.get('num_workers', 0))
  generator = torch.Generator()
  generator.manual_seed(seed)
  loader_kwargs = {
    'dataset': dataset,
    'batch_size': int(config['test']['n_episode']),
    'shuffle': False,
    'collate_fn': collate_paired_episodes,
    'num_workers': num_workers,
    'pin_memory': device.type == 'cuda',
    'worker_init_fn': seed_worker,
    'generator': generator,
  }
  if num_workers > 0:
    loader_kwargs['prefetch_factor'] = int(config.get('prefetch_factor', 2))
    loader_kwargs['persistent_workers'] = bool(
      config.get('persistent_workers', True))
  return DataLoader(**loader_kwargs)


def load_analysis_models(config, args, device):
  records = {}
  parameter_fingerprints = {}
  file_hashes = {}
  resolved_paths = {}
  model_names = configured_model_names(config)
  expected_encoder = config.get('encoder')
  paired_mode = analysis_mode(config) == 'paired_checkpoints'

  for name in model_names:
    model_config = config['models'][name]
    checkpoint_path = resolve_project_path(model_config['load'])
    if not os.path.isfile(checkpoint_path):
      raise FileNotFoundError(
        '{} checkpoint not found: {}'.format(name, checkpoint_path))
    resolved_paths[name] = checkpoint_path
    utils.log('loading {} checkpoint: {}'.format(name, checkpoint_path))
    checkpoint = torch_load_checkpoint(checkpoint_path)
    checkpoint_config = dict(checkpoint.get('config') or {})
    checkpoint_encoder = checkpoint.get('encoder')
    if not checkpoint_encoder:
      raise ValueError('{} checkpoint encoder is missing'.format(name))
    if expected_encoder is None:
      expected_encoder = checkpoint_encoder
    require_equal('{} checkpoint encoder'.format(name),
                  checkpoint_encoder, expected_encoder)
    require_equal('{} checkpoint classifier'.format(name),
                  checkpoint.get('classifier'), 'logistic')
    classifier_args = checkpoint.get('classifier_args') or {}
    require_equal('{} checkpoint classifier n_way'.format(name),
                  int(classifier_args.get('n_way', -1)),
                  int(config['test']['n_way']))

    file_hash = sha256_file(checkpoint_path)
    parameter_fingerprint = checkpoint_parameter_fingerprint(checkpoint)
    file_hashes[name] = file_hash
    parameter_fingerprints[name] = parameter_fingerprint

    precision_config = dict(checkpoint_config)
    if not paired_mode and 'use_amp' not in config:
      # Preserve the oracle tool's historical disabled-by-default AMP policy.
      precision_config['use_amp'] = False
    for key in ('use_amp', 'amp_dtype', 'allow_tf32'):
      if key in config:
        precision_config[key] = config[key]
    use_amp, amp_dtype, amp_dtype_name, _, allow_tf32 = \
      utils.config_cuda_precision(precision_config)

    if paired_mode:
      use_gradient_transport = model_config.get(
        'use_gradient_transport', config.get(
          'use_gradient_transport',
          checkpoint_config.get('use_gradient_transport', False)))
    else:
      use_gradient_transport = False

    model = models.load(
      checkpoint, load_clf=(not config['inner_args']['reset_classifier']))
    if args.efficient:
      model.go_efficient()
    model.eval()
    model.cpu()
    if config.get('_parallel'):
      model = nn.DataParallel(model)

    records[name] = {
      'name': name,
      'load': checkpoint_path,
      'use_gradient_transport': bool(use_gradient_transport),
      'precision': {
        'use_amp': bool(use_amp),
        'amp_dtype': amp_dtype,
        'amp_dtype_name': amp_dtype_name,
        'autocast_enabled': bool(use_amp and device.type == 'cuda'),
        'effective_compute_dtype': (
          amp_dtype_name
          if use_amp and device.type == 'cuda' else 'float32'),
        'allow_tf32': bool(allow_tf32),
      },
      'checkpoint_config': checkpoint_config,
      'checkpoint_bn_args': (
        (checkpoint.get('encoder_args') or
         checkpoint_config.get('encoder_args') or {}).get('bn_args')),
      'file_sha256': file_hash,
      'parameter_sha256': parameter_fingerprint,
      'model': model,
    }
    utils.log(
      '{} params: {}, gradient transport: {}, amp: {} ({}), tf32: {}'
      .format(
        name, utils.compute_n_params(model),
        'enabled' if use_gradient_transport else 'disabled',
        'enabled' if use_amp else 'disabled',
        amp_dtype_name if use_amp else 'float32',
        'enabled' if allow_tf32 else 'disabled'))

  if len(set(resolved_paths.values())) != len(model_names):
    raise ValueError('all analysis model paths must be distinct')
  if len(set(file_hashes.values())) != len(model_names):
    raise ValueError(
      'all analysis checkpoint files must be distinct')
  if len(set(parameter_fingerprints.values())) != len(model_names):
    raise ValueError(
      'all analysis model parameters must be distinct')
  config['encoder'] = expected_encoder

  keep_on_gpu = (
    bool(config.get('keep_models_on_gpu', True))
    if args.keep_models_on_gpu is None else args.keep_models_on_gpu)
  keep_on_gpu = bool(keep_on_gpu and device.type == 'cuda')
  if keep_on_gpu:
    for record in records.values():
      record['model'].to(device)

  checkpoint_report = {
    'all_paths_distinct': True,
    'all_file_hashes_distinct': True,
    'all_parameter_hashes_distinct': True,
    'encoder': expected_encoder,
    'encoder_verified_as_configured': True,
    # Kept for consumers of the original ConvNet4 report schema.
    'encoder_verified_as_convnet4': expected_encoder == 'convnet4',
    'classifier_verified_as_5way_logistic': (
      int(config['test']['n_way']) == 5),
    'classifier_verified_as_configured_n_way_logistic': True,
    'gradient_transport_disabled_for_all': all(
      not records[name]['use_gradient_transport'] for name in model_names),
    'models': {
      name: {
        'load': records[name]['load'],
        'file_sha256': records[name]['file_sha256'],
        'parameter_sha256': records[name]['parameter_sha256'],
        'use_gradient_transport': records[name]['use_gradient_transport'],
        'precision': {
          key: value for key, value in records[name]['precision'].items()
          if key != 'amp_dtype'
        },
        'checkpoint_bn_args': records[name]['checkpoint_bn_args'],
      }
      for name in model_names
    },
  }
  return records, keep_on_gpu, checkpoint_report


def move_batch_to_device(batch, device):
  non_blocking = device.type == 'cuda'
  return tuple(
    tensor.to(device, non_blocking=non_blocking) for tensor in batch)


def batch_storage_signature(batch):
  return tuple(
    (tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype),
     int(tensor._version))
    for tensor in batch)


def evaluate_model(
        record,
        batch,
        inner_args,
        n_way,
        device,
        use_amp,
        keep_on_gpu,
        amp_dtype=None,
        allow_tf32=None):
  model = record['model']
  if not keep_on_gpu:
    model.to(device)
  model.eval()

  if allow_tf32 is not None:
    allow_tf32 = bool(allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision(
      'high' if allow_tf32 else 'highest')

  x_shot, x_query, y_shot, y_query = batch
  if inner_args['reset_classifier']:
    if isinstance(model, nn.DataParallel):
      model.module.reset_classifier()
    else:
      model.reset_classifier()

  with autocast_for(device, use_amp, dtype=amp_dtype):
    logits = model(
      x_shot,
      x_query,
      y_shot,
      inner_args,
      meta_train=False,
      use_gradient_transport=record.get('use_gradient_transport', False))
    logits = logits.view(-1, n_way)
    labels = y_query.view(-1)
    element_losses = F.cross_entropy(logits, labels, reduction='none')
    task_losses = element_losses.view(y_query.size(0), -1).mean(dim=1)
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).float()
    task_accuracies = correct.view(y_query.size(0), -1).mean(dim=1)

  losses = task_losses.detach().float().cpu().numpy()
  accuracies = task_accuracies.detach().float().cpu().numpy()
  if not np.isfinite(losses).all() or not np.isfinite(accuracies).all():
    raise FloatingPointError(
      '{} produced NaN or Inf query metrics'.format(record['name']))

  if not keep_on_gpu:
    model.cpu()
    if device.type == 'cuda':
      torch.cuda.empty_cache()
  return losses.tolist(), accuracies.tolist()


def split_class_balanced_query_batch(batch, n_way, n_query, split_query):
  x_shot, x_query, y_shot, y_query = batch
  if n_query != split_query * 2:
    raise ValueError(
      'query split requires n_query == 2 * split_query, got {} and {}'.format(
        n_query, split_query))

  query_a, labels_a, query_b, labels_b = [], [], [], []
  for episode in range(x_query.size(0)):
    episode_query_a = []
    episode_labels_a = []
    episode_query_b = []
    episode_labels_b = []
    for cls in range(n_way):
      class_indices = (y_query[episode] == cls).nonzero(
        as_tuple=False).flatten()
      if class_indices.numel() != n_query:
        raise ValueError(
          'task {} class {} has {} query examples; expected {}'.format(
            episode, cls, class_indices.numel(), n_query))
      indices_a = class_indices[:split_query]
      indices_b = class_indices[split_query:]
      episode_query_a.append(x_query[episode, indices_a])
      episode_labels_a.append(y_query[episode, indices_a])
      episode_query_b.append(x_query[episode, indices_b])
      episode_labels_b.append(y_query[episode, indices_b])

    query_a.append(torch.cat(episode_query_a, dim=0))
    labels_a.append(torch.cat(episode_labels_a, dim=0))
    query_b.append(torch.cat(episode_query_b, dim=0))
    labels_b.append(torch.cat(episode_labels_b, dim=0))

  batch_a = (
    x_shot,
    torch.stack(query_a, dim=0),
    y_shot,
    torch.stack(labels_a, dim=0),
  )
  batch_b = (
    x_shot,
    torch.stack(query_b, dim=0),
    y_shot,
    torch.stack(labels_b, dim=0),
  )
  for split_name, split_batch in (('A', batch_a), ('B', batch_b)):
    split_labels = split_batch[3]
    for cls in range(n_way):
      counts = (split_labels == cls).sum(dim=1)
      if not torch.all(counts == split_query):
        raise RuntimeError(
          'query split {} is not class-balanced'.format(split_name))
  return batch_a, batch_b


def evaluate_standard_tasks(
        config,
        model_records,
        loader,
        device,
        keep_on_gpu):
  results = {
    name: {'loss': [], 'accuracy': []}
    for name in analysis_model_names(config)
  }
  evaluated_ids = {name: [] for name in analysis_model_names(config)}
  seen_metadata = []
  inner_args = config['inner_args']
  n_way = int(config['test']['n_way'])

  for task_ids, clusters, domains, cpu_batch in tqdm(
      loader, desc='specialist oracle', leave=False):
    batch = move_batch_to_device(cpu_batch, device)
    signature = batch_storage_signature(batch)
    seen_metadata.extend(zip(task_ids, clusters, domains))
    for name in analysis_model_names(config):
      if batch_storage_signature(batch) != signature:
        raise RuntimeError('episode tensors changed between checkpoints')
      losses, accuracies = evaluate_model(
        model_records[name], batch, inner_args, n_way, device,
        model_records[name]['precision']['use_amp'], keep_on_gpu,
        amp_dtype=model_records[name]['precision']['amp_dtype'],
        allow_tf32=model_records[name]['precision']['allow_tf32'])
      if batch_storage_signature(batch) != signature:
        raise RuntimeError('{} modified the episode tensors'.format(name))
      if len(losses) != len(task_ids):
        raise RuntimeError('{} returned the wrong task count'.format(name))
      results[name]['loss'].extend(losses)
      results[name]['accuracy'].extend(accuracies)
      evaluated_ids[name].extend(task_ids)

  return results, evaluated_ids, seen_metadata


def canonical_episode_metadata(task_id, domain, metadata):
  class_names = tuple(str(name) for name in metadata['class_names'])
  support = tuple(
    tuple(int(index) for index in values)
    for values in metadata['support_pickle_indices'])
  query = tuple(
    tuple(int(index) for index in values)
    for values in metadata['query_pickle_indices'])
  mapping = {
    class_name: {
      'episode_label': episode_label,
      'support_pickle_indices': list(support[episode_label]),
      'query_pickle_indices': list(query[episode_label]),
    }
    for episode_label, class_name in enumerate(class_names)
  }
  identity = {
    'domain': str(domain),
    'class_sample_mapping': mapping,
  }
  identity_json = json.dumps(
    identity, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
  return {
    'task_id': int(task_id),
    'domain': str(domain),
    'class_combination': json.dumps(
      sorted(class_names), ensure_ascii=False, separators=(',', ':')),
    'class_sample_mapping': json.dumps(
      mapping, ensure_ascii=False, sort_keys=True, separators=(',', ':')),
    'episode_sample_sha256': hashlib.sha256(
      identity_json.encode('utf-8')).hexdigest(),
  }


def evaluate_paired_tasks(
        config,
        model_records,
        loader,
        device,
        keep_on_gpu):
  model_names = paired_model_names(config)
  results = {
    name: {'loss': [], 'accuracy': []} for name in model_names
  }
  evaluated_ids = {name: [] for name in model_names}
  evaluated_sample_ids = {name: [] for name in model_names}
  metadata_records = []
  inner_args = config['inner_args']
  n_way = int(config['test']['n_way'])

  for task_ids, domains, cpu_batch, metadata_batch in tqdm(
      loader, desc='paired checkpoints', leave=False):
    if not (len(task_ids) == len(domains) == len(metadata_batch)):
      raise RuntimeError('paired loader returned inconsistent batch lengths')
    batch = move_batch_to_device(cpu_batch, device)
    storage_signature = batch_storage_signature(batch)
    batch_records = []
    for task_id, domain, metadata in zip(
        task_ids, domains, metadata_batch):
      loader.dataset.validate_episode_metadata(metadata)
      if domain != config['domain']:
        raise RuntimeError('paired loader returned an unexpected domain')
      batch_records.append(
        canonical_episode_metadata(task_id, domain, metadata))
    metadata_records.extend(batch_records)
    batch_sample_ids = [
      record['episode_sample_sha256'] for record in batch_records
    ]
    shared_eval_rng_state = capture_rng_state()
    post_eval_rng_state = None

    for model_index, name in enumerate(model_names):
      restore_rng_state(shared_eval_rng_state)
      if batch_storage_signature(batch) != storage_signature:
        raise RuntimeError('episode tensors changed between checkpoints')
      precision = model_records[name]['precision']
      losses, accuracies = evaluate_model(
        model_records[name], batch, inner_args, n_way, device,
        precision['use_amp'], keep_on_gpu,
        amp_dtype=precision['amp_dtype'],
        allow_tf32=precision['allow_tf32'])
      if batch_storage_signature(batch) != storage_signature:
        raise RuntimeError('{} modified the episode tensors'.format(name))
      if len(losses) != len(task_ids):
        raise RuntimeError('{} returned the wrong task count'.format(name))
      results[name]['loss'].extend(losses)
      results[name]['accuracy'].extend(accuracies)
      evaluated_ids[name].extend(task_ids)
      evaluated_sample_ids[name].extend(batch_sample_ids)
      if model_index == 0:
        post_eval_rng_state = capture_rng_state()
    # Advance the run as if one model had been evaluated, while ensuring both
    # checkpoints saw identical stochastic classifier-reset/eval RNG state.
    restore_rng_state(post_eval_rng_state)

  return (
    results, evaluated_ids, evaluated_sample_ids, metadata_records)


def evaluate_query_split_tasks(
        config,
        model_records,
        loader,
        device,
        keep_on_gpu):
  results = {
    name: {
      'A': {'loss': [], 'accuracy': []},
      'B': {'loss': [], 'accuracy': []},
    }
    for name in analysis_model_names(config)
  }
  evaluated_ids = {name: [] for name in analysis_model_names(config)}
  seen_metadata = []
  inner_args = config['inner_args']
  n_way = int(config['test']['n_way'])
  n_query = int(config['query_split']['n_query'])
  split_query = int(config['query_split']['split_query'])

  for task_ids, clusters, domains, cpu_batch in tqdm(
      loader, desc='query-split cross-fit', leave=False):
    batch = move_batch_to_device(cpu_batch, device)
    batch_a, batch_b = split_class_balanced_query_batch(
      batch, n_way, n_query, split_query)
    signature_a = batch_storage_signature(batch_a)
    signature_b = batch_storage_signature(batch_b)
    seen_metadata.extend(zip(task_ids, clusters, domains))

    for name in analysis_model_names(config):
      if (batch_storage_signature(batch_a) != signature_a or
          batch_storage_signature(batch_b) != signature_b):
        raise RuntimeError(
          'query-split episode tensors changed between checkpoints')
      for split_name, split_batch in (('A', batch_a), ('B', batch_b)):
        losses, accuracies = evaluate_model(
          model_records[name], split_batch, inner_args, n_way, device,
          model_records[name]['precision']['use_amp'], keep_on_gpu,
          amp_dtype=model_records[name]['precision']['amp_dtype'],
          allow_tf32=model_records[name]['precision']['allow_tf32'])
        if (batch_storage_signature(batch_a) != signature_a or
            batch_storage_signature(batch_b) != signature_b):
          raise RuntimeError(
            '{} modified query-split episode tensors'.format(name))
        if len(losses) != len(task_ids):
          raise RuntimeError(
            '{} split {} returned the wrong task count'.format(
              name, split_name))
        results[name][split_name]['loss'].extend(losses)
        results[name][split_name]['accuracy'].extend(accuracies)
      evaluated_ids[name].extend(task_ids)

  return results, evaluated_ids, seen_metadata


def as_finite_array(values, label):
  array = np.asarray(values, dtype=np.float64)
  if array.ndim != 1 or array.size == 0:
    raise ValueError('{} must be a non-empty vector'.format(label))
  if not np.isfinite(array).all():
    raise FloatingPointError('{} contains NaN or Inf'.format(label))
  return array


def safe_relative_percent(numerator, denominator):
  if denominator == 0.0:
    return None
  return float(numerator / denominator * 100.0)


def estimate_interval(estimate, samples, confidence):
  alpha = (1.0 - confidence) / 2.0
  low, high = np.quantile(samples, [alpha, 1.0 - alpha])
  return {
    'estimate': float(estimate),
    'low': float(low),
    'high': float(high),
  }


def bootstrap_mean_ci(values, n_resamples, confidence, seed):
  values = as_finite_array(values, 'bootstrap values')
  rng = np.random.default_rng(seed)
  samples = np.empty(n_resamples, dtype=np.float64)
  chunk_size = 256
  for start in range(0, n_resamples, chunk_size):
    size = min(chunk_size, n_resamples - start)
    indices = rng.integers(0, values.size, size=(size, values.size))
    samples[start:start + size] = values[indices].mean(axis=1)
  return estimate_interval(float(values.mean()), samples, confidence)


def paired_bootstrap_ci(
        candidate,
        reference,
        n_resamples,
        confidence,
        seed):
  candidate = as_finite_array(candidate, 'paired candidate')
  reference = as_finite_array(reference, 'paired reference')
  if candidate.shape != reference.shape:
    raise ValueError('paired bootstrap arrays must have identical shapes')

  rng = np.random.default_rng(seed)
  candidate_samples = np.empty(n_resamples, dtype=np.float64)
  reference_samples = np.empty(n_resamples, dtype=np.float64)
  relative_samples = np.empty(n_resamples, dtype=np.float64)
  relative_absolute_samples = np.empty(n_resamples, dtype=np.float64)
  chunk_size = 256
  for start in range(0, n_resamples, chunk_size):
    size = min(chunk_size, n_resamples - start)
    indices = rng.integers(0, candidate.size,
                           size=(size, candidate.size))
    candidate_mean = candidate[indices].mean(axis=1)
    reference_mean = reference[indices].mean(axis=1)
    candidate_samples[start:start + size] = candidate_mean
    reference_samples[start:start + size] = reference_mean
    relative_samples[start:start + size] = np.divide(
      reference_mean - candidate_mean,
      reference_mean,
      out=np.full(size, np.nan, dtype=np.float64),
      where=reference_mean != 0.0) * 100.0
    relative_absolute_samples[start:start + size] = np.divide(
      np.abs(candidate_mean - reference_mean),
      reference_mean,
      out=np.full(size, np.nan, dtype=np.float64),
      where=reference_mean != 0.0) * 100.0

  candidate_mean = float(candidate.mean())
  reference_mean = float(reference.mean())
  difference_samples = candidate_samples - reference_samples
  gain_samples = -difference_samples
  finite_relative = relative_samples[np.isfinite(relative_samples)]
  finite_relative_absolute = relative_absolute_samples[
    np.isfinite(relative_absolute_samples)]
  output = {
    'candidate_mean': estimate_interval(
      candidate_mean, candidate_samples, confidence),
    'reference_mean': estimate_interval(
      reference_mean, reference_samples, confidence),
    'candidate_minus_reference': estimate_interval(
      candidate_mean - reference_mean, difference_samples, confidence),
    'absolute_difference': estimate_interval(
      abs(candidate_mean - reference_mean),
      np.abs(difference_samples), confidence),
    'reference_minus_candidate_gain': estimate_interval(
      reference_mean - candidate_mean, gain_samples, confidence),
  }
  point_relative = safe_relative_percent(
    reference_mean - candidate_mean, reference_mean)
  output['relative_gain_percent'] = (
    estimate_interval(point_relative, finite_relative, confidence)
    if point_relative is not None and finite_relative.size else None)
  point_relative_absolute = safe_relative_percent(
    abs(candidate_mean - reference_mean), reference_mean)
  output['relative_absolute_difference_percent'] = (
    estimate_interval(
      point_relative_absolute, finite_relative_absolute, confidence)
    if (point_relative_absolute is not None and
        finite_relative_absolute.size) else None)
  return output


def query_split_noise_floor_bootstrap(
        losses_a,
        losses_b,
        n_resamples,
        confidence,
        seed):
  losses_a = as_finite_array(losses_a, 'noise-floor A losses')
  losses_b = as_finite_array(losses_b, 'noise-floor B losses')
  if losses_a.shape != losses_b.shape:
    raise ValueError('noise-floor arrays must have identical shapes')

  rng = np.random.default_rng(seed)
  best_single_samples = np.empty(n_resamples, dtype=np.float64)
  oracle_samples = np.empty(n_resamples, dtype=np.float64)
  relative_samples = np.empty(n_resamples, dtype=np.float64)
  chunk_size = 256
  for start in range(0, n_resamples, chunk_size):
    size = min(chunk_size, n_resamples - start)
    indices = rng.integers(0, losses_a.size,
                           size=(size, losses_a.size))
    sampled_a = losses_a[indices]
    sampled_b = losses_b[indices]
    best_single = np.minimum(
      sampled_a.mean(axis=1), sampled_b.mean(axis=1))
    oracle = np.minimum(sampled_a, sampled_b).mean(axis=1)
    gain = best_single - oracle
    best_single_samples[start:start + size] = best_single
    oracle_samples[start:start + size] = oracle
    relative_samples[start:start + size] = np.divide(
      gain,
      best_single,
      out=np.full(size, np.nan, dtype=np.float64),
      where=best_single != 0.0) * 100.0

  mean_a = float(losses_a.mean())
  mean_b = float(losses_b.mean())
  best_single = min(mean_a, mean_b)
  oracle = float(np.minimum(losses_a, losses_b).mean())
  gain = best_single - oracle
  gain_samples = best_single_samples - oracle_samples
  finite_relative = relative_samples[np.isfinite(relative_samples)]
  relative = safe_relative_percent(gain, best_single)
  return {
    'best_single_loss': estimate_interval(
      best_single, best_single_samples, confidence),
    'oracle_loss': estimate_interval(oracle, oracle_samples, confidence),
    'absolute_gain': estimate_interval(gain, gain_samples, confidence),
    'relative_gain_percent': (
      estimate_interval(relative, finite_relative, confidence)
      if relative is not None and finite_relative.size else None),
  }


def average_tie_ranks(values):
  values = np.asarray(values, dtype=np.float64)
  order = np.argsort(values, kind='mergesort')
  sorted_values = values[order]
  ranks = np.empty(values.size, dtype=np.float64)
  start = 0
  while start < values.size:
    end = start + 1
    while end < values.size and sorted_values[end] == sorted_values[start]:
      end += 1
    # scipy.stats.rankdata(method='average') uses one-based ranks.
    average_rank = 0.5 * ((start + 1) + end)
    ranks[order[start:end]] = average_rank
    start = end
  return ranks


def spearman_correlation(left, right):
  left = as_finite_array(left, 'Spearman left')
  right = as_finite_array(right, 'Spearman right')
  if left.shape != right.shape:
    raise ValueError('Spearman arrays must have identical shapes')
  if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
    return None
  left_ranks = average_tie_ranks(left)
  right_ranks = average_tie_ranks(right)
  correlation = np.corrcoef(left_ranks, right_ranks)[0, 1]
  if not np.isfinite(correlation):
    return None
  return float(correlation)


def paired_loss_diagnostics_bootstrap(
        left_losses,
        right_losses,
        n_resamples,
        confidence,
        seed,
        tolerance=1e-12,
        left_name='natural',
        right_name='technical'):
  left_losses = as_finite_array(left_losses, 'diagnostic left losses')
  right_losses = as_finite_array(right_losses, 'diagnostic right losses')
  if left_losses.shape != right_losses.shape:
    raise ValueError('diagnostic loss arrays must have identical shapes')

  ties = np.isclose(
    left_losses, right_losses, rtol=0.0, atol=tolerance)
  left_win = ((left_losses < right_losses) & ~ties).astype(
    np.float64)
  right_win = ((right_losses < left_losses) & ~ties).astype(
    np.float64)
  tie = ties.astype(np.float64)
  rng = np.random.default_rng(seed)
  left_samples = np.empty(n_resamples, dtype=np.float64)
  right_samples = np.empty(n_resamples, dtype=np.float64)
  tie_samples = np.empty(n_resamples, dtype=np.float64)
  correlation_samples = np.empty(n_resamples, dtype=np.float64)
  chunk_size = 128
  for start in range(0, n_resamples, chunk_size):
    size = min(chunk_size, n_resamples - start)
    indices = rng.integers(
      0, left_losses.size, size=(size, left_losses.size))
    left_samples[start:start + size] = left_win[indices].mean(axis=1)
    right_samples[start:start + size] = right_win[indices].mean(
      axis=1)
    tie_samples[start:start + size] = tie[indices].mean(axis=1)
    for offset, sampled_indices in enumerate(indices):
      correlation = spearman_correlation(
        left_losses[sampled_indices], right_losses[sampled_indices])
      correlation_samples[start + offset] = (
        np.nan if correlation is None else correlation)

  finite_correlation = correlation_samples[
    np.isfinite(correlation_samples)]
  point_correlation = spearman_correlation(
    left_losses, right_losses)
  left_slug = name_slug(left_name)
  right_slug = name_slug(right_name)
  return {
    '{}_win_rate'.format(left_slug): estimate_interval(
      float(left_win.mean()), left_samples, confidence),
    '{}_win_rate'.format(right_slug): estimate_interval(
      float(right_win.mean()), right_samples, confidence),
    'tie_rate': estimate_interval(float(tie.mean()), tie_samples, confidence),
    'loss_spearman': (
      estimate_interval(
        point_correlation, finite_correlation, confidence)
      if point_correlation is not None and finite_correlation.size else None),
  }


def winner_rates(left, right, left_name, right_name, tolerance=1e-12):
  left = as_finite_array(left, 'winner left')
  right = as_finite_array(right, 'winner right')
  if left.shape != right.shape:
    raise ValueError('winner arrays must have identical shapes')
  ties = np.isclose(left, right, rtol=0.0, atol=tolerance)
  left_wins = (left < right) & ~ties
  right_wins = (right < left) & ~ties
  total = left.size
  left_slug = name_slug(left_name)
  right_slug = name_slug(right_name)
  return {
    '{}_wins'.format(left_slug): int(left_wins.sum()),
    '{}_wins'.format(right_slug): int(right_wins.sum()),
    'ties': int(ties.sum()),
    '{}_win_rate'.format(left_slug): float(left_wins.sum() / total),
    '{}_win_rate'.format(right_slug): float(
      right_wins.sum() / total),
    'tie_rate': float(ties.sum() / total),
  }


def metric_summary(losses, accuracies, bootstrap, seed):
  losses = as_finite_array(losses, 'metric losses')
  accuracies = as_finite_array(accuracies, 'metric accuracies')
  return {
    'mean_query_loss': float(losses.mean()),
    'mean_query_accuracy': float(accuracies.mean()),
    'paired_bootstrap_ci95': {
      'mean_query_loss': bootstrap_mean_ci(
        losses, bootstrap['n_resamples'], bootstrap['confidence'], seed),
      'mean_query_accuracy': bootstrap_mean_ci(
        accuracies, bootstrap['n_resamples'], bootstrap['confidence'],
        seed + 1),
    },
  }


def comparison_summary(
        candidate_losses,
        reference_losses,
        candidate_accuracies,
        reference_accuracies,
        bootstrap,
        seed):
  candidate_losses = as_finite_array(candidate_losses, 'candidate losses')
  reference_losses = as_finite_array(reference_losses, 'reference losses')
  candidate_accuracies = as_finite_array(
    candidate_accuracies, 'candidate accuracies')
  reference_accuracies = as_finite_array(
    reference_accuracies, 'reference accuracies')
  candidate_loss = float(candidate_losses.mean())
  reference_loss = float(reference_losses.mean())
  signed_difference = candidate_loss - reference_loss
  gain = reference_loss - candidate_loss
  candidate_accuracy = float(candidate_accuracies.mean())
  reference_accuracy = float(reference_accuracies.mean())
  return {
    'candidate_mean_query_loss': candidate_loss,
    'reference_mean_query_loss': reference_loss,
    'candidate_minus_reference_loss': signed_difference,
    'absolute_loss_difference': abs(signed_difference),
    'relative_absolute_loss_difference_percent': safe_relative_percent(
      abs(signed_difference), reference_loss),
    'reference_minus_candidate_loss_gain': gain,
    'relative_loss_gain_percent': safe_relative_percent(gain, reference_loss),
    'candidate_mean_query_accuracy': candidate_accuracy,
    'reference_mean_query_accuracy': reference_accuracy,
    'candidate_minus_reference_accuracy': (
      candidate_accuracy - reference_accuracy),
    'paired_bootstrap_ci95': {
      'loss': paired_bootstrap_ci(
        candidate_losses, reference_losses,
        bootstrap['n_resamples'], bootstrap['confidence'], seed),
      'accuracy': paired_bootstrap_ci(
        candidate_accuracies, reference_accuracies,
        bootstrap['n_resamples'], bootstrap['confidence'], seed + 1),
    },
  }


def bootstrap_config(config, args):
  bootstrap = dict(config.get('bootstrap') or {})
  if args.bootstrap_resamples is not None:
    bootstrap['n_resamples'] = args.bootstrap_resamples
  resolved = {
    'n_resamples': int(bootstrap.get('n_resamples', 10000)),
    'confidence': float(bootstrap.get('confidence', 0.95)),
    'seed': int(bootstrap.get('seed', 1729)),
  }
  if resolved['n_resamples'] <= 0:
    raise ValueError('bootstrap resample count must be positive')
  return resolved


def _resolve_row_config(task_specs, results, config):
  if config is not None:
    return config
  clusters = tuple(dict.fromkeys(spec['cluster'] for spec in task_specs))
  models_config = {str(name): {} for name in results}
  temporary = {
    'clusters': {str(cluster): [] for cluster in clusters},
    'models': models_config,
  }
  temporary['model_roles'] = infer_model_roles(temporary)
  return temporary


def make_standard_rows(task_specs, results, config=None):
  config = _resolve_row_config(task_specs, results, config)
  specialist_by_cluster = specialist_model_by_cluster(config)
  specialists = specialist_model_names(config)
  reference = reference_model_name(config)
  rows = []
  for index, spec in enumerate(task_specs):
    values = {}
    model_metrics = {}
    for model_name in analysis_model_names(config):
      loss = float(results[model_name]['loss'][index])
      accuracy = float(results[model_name]['accuracy'][index])
      model_metrics[model_name] = {'loss': loss, 'accuracy': accuracy}
      slug = name_slug(model_name)
      values['{}_query_loss'.format(slug)] = loss
      values['{}_query_accuracy'.format(slug)] = accuracy

    first, second = specialists
    first_loss = model_metrics[first]['loss']
    second_loss = model_metrics[second]['loss']
    # Exact ties are deterministically routed to the first YAML specialist.
    # The tie flag keeps
    # this convention explicit in both the CSV and summary.
    selected_specialist = (
      first if first_loss <= second_loss else second)
    specialist_tie = bool(np.isclose(
      first_loss, second_loss, rtol=0.0, atol=1e-12))
    domain_specialist = specialist_by_cluster[spec['cluster']]
    oracle_loss = model_metrics[selected_specialist]['loss']
    oracle_accuracy = model_metrics[selected_specialist]['accuracy']
    domain_router_loss = model_metrics[domain_specialist]['loss']
    domain_router_accuracy = model_metrics[domain_specialist]['accuracy']

    rows.append({
      'task_id': spec['task_id'],
      'cluster': spec['cluster'],
      'domain': spec['domain'],
      **values,
      'loss_selected_specialist': selected_specialist,
      'specialist_loss_tie': specialist_tie,
      'domain_specialist': domain_specialist,
      'domain_router_query_loss': domain_router_loss,
      'domain_router_query_accuracy': domain_router_accuracy,
      'oracle_query_loss': oracle_loss,
      'oracle_query_accuracy': oracle_accuracy,
    })
  return rows


def make_query_split_rows(task_specs, results, config=None):
  config = _resolve_row_config(task_specs, results, config)
  specialists = specialist_model_names(config)
  reference = reference_model_name(config)
  reference_slug = name_slug(reference)
  rows = []
  for index, spec in enumerate(task_specs):
    values = {}
    for model_name in analysis_model_names(config):
      slug = name_slug(model_name)
      for split_name in ('A', 'B'):
        split_slug = split_name.lower()
        values['{}_{}_query_loss'.format(slug, split_slug)] = float(
          results[model_name][split_name]['loss'][index])
        values['{}_{}_query_accuracy'.format(slug, split_slug)] = float(
          results[model_name][split_name]['accuracy'][index])

    first, second = specialists
    first_slug = name_slug(first)
    second_slug = name_slug(second)
    select_on_a = (
      first
      if values['{}_a_query_loss'.format(first_slug)] <=
      values['{}_a_query_loss'.format(second_slug)] else second)
    select_on_b = (
      first
      if values['{}_b_query_loss'.format(first_slug)] <=
      values['{}_b_query_loss'.format(second_slug)] else second)

    a_model = name_slug(select_on_a)
    b_model = name_slug(select_on_b)
    a_to_b_loss = values['{}_b_query_loss'.format(a_model)]
    a_to_b_accuracy = values['{}_b_query_accuracy'.format(a_model)]
    b_to_a_loss = values['{}_a_query_loss'.format(b_model)]
    b_to_a_accuracy = values['{}_a_query_accuracy'.format(b_model)]

    oracle_a_model = name_slug(select_on_a)
    oracle_b_model = name_slug(select_on_b)
    in_sample_oracle_loss = 0.5 * (
      values['{}_a_query_loss'.format(oracle_a_model)] +
      values['{}_b_query_loss'.format(oracle_b_model)])
    in_sample_oracle_accuracy = 0.5 * (
      values['{}_a_query_accuracy'.format(oracle_a_model)] +
      values['{}_b_query_accuracy'.format(oracle_b_model)])

    row = {
      'task_id': spec['task_id'],
      'cluster': spec['cluster'],
      'domain': spec['domain'],
      **values,
      'specialist_selected_on_a': select_on_a,
      'specialist_selected_on_b': select_on_b,
      'a_to_b_query_loss': a_to_b_loss,
      'a_to_b_query_accuracy': a_to_b_accuracy,
      'b_to_a_query_loss': b_to_a_loss,
      'b_to_a_query_accuracy': b_to_a_accuracy,
      'cross_fitted_oracle_query_loss': 0.5 * (
        a_to_b_loss + b_to_a_loss),
      'cross_fitted_oracle_query_accuracy': 0.5 * (
        a_to_b_accuracy + b_to_a_accuracy),
      'cross_fitted_{}_query_loss'.format(reference_slug): 0.5 * (
        values['{}_a_query_loss'.format(reference_slug)] +
        values['{}_b_query_loss'.format(reference_slug)]),
      'cross_fitted_{}_query_accuracy'.format(reference_slug): 0.5 * (
        values['{}_a_query_accuracy'.format(reference_slug)] +
        values['{}_b_query_accuracy'.format(reference_slug)]),
      'in_sample_oracle_query_loss': in_sample_oracle_loss,
      'in_sample_oracle_query_accuracy': in_sample_oracle_accuracy,
      '{}_split_oracle_query_loss'.format(reference_slug): min(
        values['{}_a_query_loss'.format(reference_slug)],
        values['{}_b_query_loss'.format(reference_slug)]),
    }
    rows.append(row)
  return rows


def make_paired_rows(metadata_records, results, config):
  first, second = paired_model_names(config)
  rows = []
  for index, metadata in enumerate(metadata_records):
    values = {}
    for name in (first, second):
      slug = name_slug(name)
      values['{}_query_loss'.format(slug)] = float(
        results[name]['loss'][index])
      values['{}_query_accuracy'.format(slug)] = float(
        results[name]['accuracy'][index])
    rows.append({
      **metadata,
      **values,
      # A signed first-minus-second difference makes the ordering in YAML
      # authoritative and avoids implying specialist/reference semantics.
      'query_loss_difference': float(
        results[first]['loss'][index] - results[second]['loss'][index]),
      'query_accuracy_difference': float(
        results[first]['accuracy'][index] -
        results[second]['accuracy'][index]),
    })
  return rows


def paired_protocol_summary(config, checkpoint_report, device):
  inner_args = config['inner_args']
  return {
    'phase': 'meta-test',
    'device_type': device.type,
    'encoder': config.get('encoder'),
    'image_size': int(config['test']['image_size']),
    'n_way': int(config['test']['n_way']),
    'n_shot': int(config['test']['n_shot']),
    'n_query': int(config['test']['n_query']),
    'episode_batch_size': int(config['test']['n_episode']),
    'inner_loop': {
      'reset_classifier': bool(inner_args['reset_classifier']),
      'n_step': int(inner_args['n_step']),
      'encoder_lr': float(inner_args['encoder_lr']),
      'classifier_lr': float(inner_args['classifier_lr']),
      'momentum': float(inner_args['momentum']),
      'weight_decay': float(inner_args['weight_decay']),
      'first_order': bool(inner_args['first_order']),
      'frozen': list(inner_args['frozen']),
    },
    'batch_norm': {
      'model_mode': 'eval',
      'frozen_during_inner_loop': 'bn' in inner_args['frozen'],
      'checkpoint_encoder_bn_args_by_model': {
        name: checkpoint_report['models'][name]['checkpoint_bn_args']
        for name in paired_model_names(config)
      },
    },
    'precision_by_model': {
      name: checkpoint_report['models'][name]['precision']
      for name in paired_model_names(config)
    },
    'gradient_transport_by_model': {
      name: checkpoint_report['models'][name]['use_gradient_transport']
      for name in paired_model_names(config)
    },
    'evaluation_rng_state_synchronized_between_models': True,
    'normalization': config['test'].get('normalization', True),
    'transform': config['test'].get('transform'),
    'val_transform': config['test'].get('val_transform'),
  }


def compute_paired_summary(
        config,
        rows,
        results,
        outputs,
        checkpoint_report,
        manifest_report,
        validation,
        device):
  first, second = paired_model_names(config)
  model_summaries = {}
  for name in (first, second):
    losses = as_finite_array(results[name]['loss'], '{} losses'.format(name))
    accuracies = as_finite_array(
      results[name]['accuracy'], '{} accuracies'.format(name))
    model_summaries[name] = {
      'mean_query_loss': float(losses.mean()),
      'mean_query_accuracy': float(accuracies.mean()),
    }
  loss_differences = as_finite_array(
    [row['query_loss_difference'] for row in rows], 'loss differences')
  accuracy_differences = as_finite_array(
    [row['query_accuracy_difference'] for row in rows],
    'accuracy differences')
  return {
    'mode': 'paired_checkpoints',
    'config_source': config.get('_config_source'),
    'dataset': config['dataset'],
    'domain': config['domain'],
    'seed': int(config.get('seed', 0)),
    'n_episodes': len(rows),
    'model_order': [first, second],
    'difference_definition': '{} minus {}'.format(first, second),
    'descriptive_metrics_only': True,
    'bootstrap_performed': False,
    'models': model_summaries,
    'differences': {
      'mean_query_loss_difference': float(loss_differences.mean()),
      'mean_query_accuracy_difference': float(
        accuracy_differences.mean()),
    },
    'protocol': paired_protocol_summary(
      config, checkpoint_report, device),
    'checkpoints': checkpoint_report,
    'manifest': manifest_report,
    'metadata_schema': {
      'class_combination': 'sorted JSON array of real class names',
      'class_sample_mapping': (
        'JSON object keyed by real class name; each value records the '
        'episode label and support/query domain-pickle indices'),
      'episode_sample_sha256': (
        'SHA-256 of domain and class/sample mapping'),
    },
    'validation': validation,
    'outputs': outputs,
  }


def protocol_summary(config, n_query):
  return {
    'phase': 'meta-test',
    'encoder': config.get('encoder'),
    'image_size': int(config['test']['image_size']),
    'n_way': int(config['test']['n_way']),
    'n_shot': int(config['test']['n_shot']),
    'n_query': int(n_query),
    'n_step': int(config['inner_args']['n_step']),
    'encoder_lr': float(config['inner_args']['encoder_lr']),
    'classifier_lr': float(config['inner_args']['classifier_lr']),
    'momentum': float(config['inner_args']['momentum']),
    'weight_decay': float(config['inner_args']['weight_decay']),
    'first_order': bool(config['inner_args']['first_order']),
    'frozen': list(config['inner_args']['frozen']),
    'normalization': config['test']['normalization'],
    'transform': config['test']['transform'],
    'use_gradient_transport': False,
  }


def compute_standard_summary(
        config,
        rows,
        results,
        outputs,
        checkpoint_report,
        manifest_report,
        validation,
        bootstrap):
  model_names = analysis_model_names(config)
  specialists = specialist_model_names(config)
  reference = reference_model_name(config)
  arrays = {
    name: {
      metric: as_finite_array(results[name][metric],
                              '{} {}'.format(name, metric))
      for metric in ('loss', 'accuracy')
    }
    for name in model_names
  }
  router_losses = as_finite_array(
    [row['domain_router_query_loss'] for row in rows], 'router losses')
  router_accuracies = as_finite_array(
    [row['domain_router_query_accuracy'] for row in rows],
    'router accuracies')
  oracle_losses = as_finite_array(
    [row['oracle_query_loss'] for row in rows], 'oracle losses')
  oracle_accuracies = as_finite_array(
    [row['oracle_query_accuracy'] for row in rows], 'oracle accuracies')

  cluster_masks = {
    cluster: np.asarray(
      [row['cluster'] == cluster for row in rows], dtype=bool)
    for cluster in cluster_names(config)
  }
  first, second = specialists
  win_rates = {
    'overall': winner_rates(
      arrays[first]['loss'], arrays[second]['loss'], first, second),
  }
  correlations = {
    'overall': spearman_correlation(
      arrays[first]['loss'], arrays[second]['loss']),
  }
  for cluster, mask in cluster_masks.items():
    win_rates[cluster] = winner_rates(
      arrays[first]['loss'][mask],
      arrays[second]['loss'][mask], first, second)
    correlations[cluster] = spearman_correlation(
      arrays[first]['loss'][mask], arrays[second]['loss'][mask])

  diagnostic_bootstrap = {}
  diagnostic_groups = {'overall': np.ones(len(rows), dtype=bool)}
  diagnostic_groups.update(cluster_masks)
  for offset, (group_name, mask) in enumerate(diagnostic_groups.items()):
    diagnostic_bootstrap[group_name] = paired_loss_diagnostics_bootstrap(
      arrays[first]['loss'][mask], arrays[second]['loss'][mask],
      bootstrap['n_resamples'], bootstrap['confidence'],
      bootstrap['seed'] + 500 + 10 * offset,
      left_name=first, right_name=second)

  base_seed = bootstrap['seed']
  model_summaries = {}
  for offset, name in enumerate(model_names):
    model_summaries[name] = {
      'load': checkpoint_report['models'][name]['load'],
      'use_gradient_transport': False,
      **metric_summary(
        arrays[name]['loss'], arrays[name]['accuracy'], bootstrap,
        base_seed + 10 * offset),
    }

  oracle_comparison = comparison_summary(
    oracle_losses, arrays[reference]['loss'],
    oracle_accuracies, arrays[reference]['accuracy'],
    bootstrap, base_seed + 200)
  router_comparison = comparison_summary(
    router_losses, arrays[reference]['loss'],
    router_accuracies, arrays[reference]['accuracy'],
    bootstrap, base_seed + 300)
  summary = {
    'mode': 'specialist_oracle',
    'dataset': config['dataset'],
    'seed': int(config.get('seed', 0)),
    'n_tasks': len(rows),
    'n_tasks_per_cluster': int(config['tasks_per_cluster']),
    'clusters': config['clusters'],
    'model_roles': config['model_roles'],
    'reference_model': reference,
    'protocol': protocol_summary(config, config['test']['n_query']),
    'checkpoints': checkpoint_report,
    'models': model_summaries,
    'domain_router': {
      'routing_rule': '; '.join(
        '{} checkpoint for {} tasks'.format(model, cluster)
        for cluster, model in specialist_model_by_cluster(config).items()),
      **metric_summary(
        router_losses, router_accuracies, bootstrap, base_seed + 100),
    },
    'per_task_oracle': {
      'selection_rule': (
        'lower query loss between {} and {}; exact ties select {}'.format(
          first, second, first)),
      **metric_summary(
        oracle_losses, oracle_accuracies, bootstrap, base_seed + 110),
    },
    'oracle_vs_reference': oracle_comparison,
    'domain_router_vs_reference': router_comparison,
    'specialist_loss_win_rates': win_rates,
    'specialist_loss_spearman': correlations,
    'specialist_diagnostics_paired_bootstrap_ci95': diagnostic_bootstrap,
    'bootstrap': bootstrap,
    'manifest_validation': manifest_report,
    'smoke_validation': validation,
    'outputs': outputs,
  }
  if name_slug(reference) == 'union':
    # Preserve the original ConvNet4 JSON keys for downstream analyses.
    summary['oracle_vs_union'] = oracle_comparison
    summary['domain_router_vs_union'] = router_comparison
  return summary


def compute_query_split_summary(
        config,
        rows,
        results,
        outputs,
        checkpoint_report,
        manifest_report,
        validation,
        bootstrap,
        standard_summary):
  base_seed = bootstrap['seed'] + 10000
  model_names = analysis_model_names(config)
  specialists = specialist_model_names(config)
  reference = reference_model_name(config)
  reference_slug = name_slug(reference)
  union_a_loss = as_finite_array(
    results[reference]['A']['loss'], '{} A losses'.format(reference))
  union_b_loss = as_finite_array(
    results[reference]['B']['loss'], '{} B losses'.format(reference))
  union_a_accuracy = as_finite_array(
    results[reference]['A']['accuracy'],
    '{} A accuracies'.format(reference))
  union_b_accuracy = as_finite_array(
    results[reference]['B']['accuracy'],
    '{} B accuracies'.format(reference))
  split_oracle_mask = union_a_loss <= union_b_loss
  split_oracle_accuracy = np.where(
    split_oracle_mask, union_a_accuracy, union_b_accuracy)
  best_single_split = (
    'A' if union_a_loss.mean() <= union_b_loss.mean() else 'B')
  best_single_loss = float(min(union_a_loss.mean(), union_b_loss.mean()))
  split_oracle_loss = float(np.minimum(union_a_loss, union_b_loss).mean())
  noise_floor_gain = best_single_loss - split_oracle_loss

  crossfit_losses = as_finite_array(
    [row['cross_fitted_oracle_query_loss'] for row in rows],
    'cross-fitted oracle losses')
  crossfit_accuracies = as_finite_array(
    [row['cross_fitted_oracle_query_accuracy'] for row in rows],
    'cross-fitted oracle accuracies')
  crossfit_union_losses = as_finite_array(
    [row['cross_fitted_{}_query_loss'.format(reference_slug)] for row in rows],
    'cross-fitted {} losses'.format(reference))
  crossfit_union_accuracies = as_finite_array(
    [row['cross_fitted_{}_query_accuracy'.format(reference_slug)]
     for row in rows],
    'cross-fitted {} accuracies'.format(reference))
  in_sample_losses = as_finite_array(
    [row['in_sample_oracle_query_loss'] for row in rows],
    'query-split in-sample oracle losses')
  in_sample_accuracies = as_finite_array(
    [row['in_sample_oracle_query_accuracy'] for row in rows],
    'query-split in-sample oracle accuracies')

  model_summaries = {}
  for model_offset, name in enumerate(model_names):
    model_summaries[name] = {}
    for split_offset, split_name in enumerate(('A', 'B')):
      model_summaries[name][split_name] = metric_summary(
        results[name][split_name]['loss'],
        results[name][split_name]['accuracy'],
        bootstrap,
        base_seed + 100 * model_offset + 10 * split_offset)

  selection_counts = {
    'selected_on_a': dict(Counter(
      row['specialist_selected_on_a'] for row in rows)),
    'selected_on_b': dict(Counter(
      row['specialist_selected_on_b'] for row in rows)),
  }
  for key in selection_counts:
    counts = selection_counts[key]
    for specialist in specialists:
      counts[specialist] = int(counts.get(specialist, 0))
      counts['{}_rate'.format(specialist)] = float(
        counts[specialist] / len(rows))

  noise_floor = {
    'definition': (
      'Same {} checkpoint and support set; class-balanced A/B query '
      'halves. Gain = best global split mean loss minus per-task minimum '
      'A/B loss, matching analyze_anchor_response.py.'.format(reference)),
    'reference_model': reference,
    'reference_a': {
      'mean_query_loss': float(union_a_loss.mean()),
      'mean_query_accuracy': float(union_a_accuracy.mean()),
    },
    'reference_b': {
      'mean_query_loss': float(union_b_loss.mean()),
      'mean_query_accuracy': float(union_b_accuracy.mean()),
    },
    'loss_comparison': winner_rates(
      union_a_loss, union_b_loss, 'A', 'B'),
    'accuracy_comparison_higher_is_better': {},
    'loss_spearman': spearman_correlation(union_a_loss, union_b_loss),
    'best_single_split': best_single_split,
    'best_single_loss': best_single_loss,
    'oracle_loss': split_oracle_loss,
    'oracle_accuracy_corresponding_to_lower_loss': float(
      split_oracle_accuracy.mean()),
    'absolute_gain': noise_floor_gain,
    'relative_gain_percent': safe_relative_percent(
      noise_floor_gain, best_single_loss),
    'paired_bootstrap_ci95': query_split_noise_floor_bootstrap(
      union_a_loss, union_b_loss,
      bootstrap['n_resamples'], bootstrap['confidence'], base_seed + 1000),
  }
  accuracy_ties = np.isclose(
    union_a_accuracy, union_b_accuracy, rtol=0.0, atol=1e-12)
  noise_floor['accuracy_comparison_higher_is_better'] = {
    'a_wins': int(((union_a_accuracy > union_b_accuracy) &
                   ~accuracy_ties).sum()),
    'b_wins': int(((union_b_accuracy > union_a_accuracy) &
                   ~accuracy_ties).sum()),
    'ties': int(accuracy_ties.sum()),
  }

  crossfit_comparison = comparison_summary(
    crossfit_losses, crossfit_union_losses,
    crossfit_accuracies, crossfit_union_accuracies,
    bootstrap, base_seed + 2000)
  in_sample_comparison = comparison_summary(
    in_sample_losses, crossfit_union_losses,
    in_sample_accuracies, crossfit_union_accuracies,
    bootstrap, base_seed + 3000)
  primary_comparison = standard_summary['oracle_vs_reference']
  primary_gain = primary_comparison['reference_minus_candidate_loss_gain']
  crossfit_gain = crossfit_comparison['reference_minus_candidate_loss_gain']
  in_sample_gain = in_sample_comparison['reference_minus_candidate_loss_gain']

  comparison_to_noise_floor = {
    'primary_specialist_oracle_gain': primary_gain,
    'primary_specialist_oracle_relative_gain_percent': (
      primary_comparison['relative_loss_gain_percent']),
    'query_split_in_sample_specialist_oracle_gain': in_sample_gain,
    'query_split_in_sample_specialist_oracle_relative_gain_percent': (
      in_sample_comparison['relative_loss_gain_percent']),
    'cross_fitted_specialist_oracle_gain': crossfit_gain,
    'cross_fitted_specialist_oracle_relative_gain_percent': (
      crossfit_comparison['relative_loss_gain_percent']),
    'query_split_noise_floor_gain': noise_floor_gain,
    'query_split_noise_floor_relative_gain_percent': (
      noise_floor['relative_gain_percent']),
    'primary_gain_minus_noise_floor': primary_gain - noise_floor_gain,
    'cross_fitted_gain_minus_noise_floor': crossfit_gain - noise_floor_gain,
    'primary_gain_to_noise_floor_ratio': (
      float(primary_gain / noise_floor_gain)
      if noise_floor_gain != 0.0 else None),
    'cross_fitted_gain_to_noise_floor_ratio': (
      float(crossfit_gain / noise_floor_gain)
      if noise_floor_gain != 0.0 else None),
  }

  summary = {
    'mode': 'class_balanced_query_split',
    'dataset': config['dataset'],
    'seed': int(config.get('seed', 0)),
    'n_tasks': len(rows),
    'n_tasks_per_cluster': int(config['tasks_per_cluster']),
    'clusters': config['clusters'],
    'model_roles': config['model_roles'],
    'reference_model': reference,
    'protocol': {
      **protocol_summary(config, config['query_split']['n_query']),
      'n_query_per_split_per_class': int(
        config['query_split']['split_query']),
      'class_balanced_splits': True,
    },
    'checkpoints': checkpoint_report,
    'models_by_split': model_summaries,
    'reference_query_split_noise_floor': noise_floor,
    'cross_fitted_oracle': {
      'definition': (
        'select {}/{} on A and evaluate on B, then select on B and evaluate '
        'on A; average both directions per task'.format(*specialists)),
      'selection_counts': selection_counts,
      **metric_summary(
        crossfit_losses, crossfit_accuracies, bootstrap, base_seed + 4000),
      'vs_reference': crossfit_comparison,
    },
    'query_split_in_sample_oracle': {
      **metric_summary(
        in_sample_losses, in_sample_accuracies, bootstrap, base_seed + 5000),
      'vs_reference': in_sample_comparison,
    },
    'specialist_gain_vs_query_split_noise_floor': comparison_to_noise_floor,
    'bootstrap': bootstrap,
    'manifest_validation': manifest_report,
    'smoke_validation': validation,
    'outputs': outputs,
  }
  if reference_slug == 'union':
    # Preserve the original ConvNet4 JSON keys for downstream analyses.
    noise_floor['union_a'] = noise_floor['reference_a']
    noise_floor['union_b'] = noise_floor['reference_b']
    summary['union_query_split_noise_floor'] = noise_floor
    summary['cross_fitted_oracle']['vs_union'] = crossfit_comparison
    summary['query_split_in_sample_oracle']['vs_union'] = in_sample_comparison
  return summary


def validate_data_files(config):
  root_path = resolve_project_path(
    config['test'].get('root_path') or config.get('data_root'))
  missing = []
  paths = {}
  for domain in configured_domains(config):
    path = os.path.join(root_path, domain + '.pickle')
    paths[domain] = path
    if not os.path.isfile(path):
      missing.append(path)
  if missing:
    raise FileNotFoundError(
      'missing Meta-Album domain pickle(s): {}'.format(', '.join(missing)))
  return {
    'root_path': root_path,
    'all_domain_pickles_present': True,
    'domain_pickle_paths': paths,
  }


def validate_paired_completed_run(
        config,
        task_specs,
        rows,
        results,
        evaluated_ids,
        evaluated_sample_ids,
        metadata_records):
  expected_ids = [spec['task_id'] for spec in task_specs]
  expected_sample_ids = [
    record['episode_sample_sha256'] for record in metadata_records
  ]
  model_names = paired_model_names(config)
  if len(rows) != int(config['n_episodes']):
    raise RuntimeError('paired output row count differs from n_episodes')
  if len(metadata_records) != len(task_specs):
    raise RuntimeError('paired metadata count differs from task count')

  for name in model_names:
    if evaluated_ids[name] != expected_ids:
      raise RuntimeError(
        '{} did not evaluate the exact paired task IDs'.format(name))
    if evaluated_sample_ids[name] != expected_sample_ids:
      raise RuntimeError(
        '{} did not evaluate the exact paired samples'.format(name))
    for metric in ('loss', 'accuracy'):
      values = results[name][metric]
      if len(values) != len(task_specs):
        raise RuntimeError(
          '{} has missing {} results'.format(name, metric))
      if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise FloatingPointError(
          '{} {} contains NaN or Inf'.format(name, metric))

  for spec, row in zip(task_specs, rows):
    if row['task_id'] != spec['task_id'] or row['domain'] != config['domain']:
      raise RuntimeError('paired row task/domain metadata is incorrect')
    combination = json.loads(row['class_combination'])
    mapping = json.loads(row['class_sample_mapping'])
    if combination != sorted(combination):
      raise RuntimeError('class_combination is not sorted')
    if combination != sorted(mapping):
      raise RuntimeError(
        'class_combination and class_sample_mapping disagree')
    for model_name in model_names:
      slug = name_slug(model_name)
      for field in ('{}_query_loss'.format(slug),
                    '{}_query_accuracy'.format(slug)):
        if not np.isfinite(float(row[field])):
          raise FloatingPointError('{} contains NaN or Inf'.format(field))
    if not np.isfinite(float(row['query_loss_difference'])):
      raise FloatingPointError('query_loss_difference contains NaN or Inf')
    if not np.isfinite(float(row['query_accuracy_difference'])):
      raise FloatingPointError(
        'query_accuracy_difference contains NaN or Inf')

  return {
    'metadata_matches_sampled_pickle_classes_and_indices': True,
    'class_combination_sorted_and_matches_mapping': True,
    'same_task_ids_used_by_both_models': True,
    'same_sample_ids_used_by_both_models': True,
    'same_in_memory_episode_batch_used_by_both_models': True,
    'evaluation_rng_state_synchronized_between_models': True,
    'each_episode_sampled_once': True,
    'each_task_is_single_domain': True,
    'task_count_matches_n_episodes': True,
    'no_nan_or_infinite_results': True,
    'no_missing_results': True,
    'meta_test_manifest_enforced_by_dataset': True,
    'manifest_train_val_test_disjoint': True,
  }


def validate_completed_run(
        config,
        task_specs,
        rows,
        results,
        evaluated_ids,
        seen_metadata,
        query_split=False):
  expected_ids = [spec['task_id'] for spec in task_specs]
  expected_metadata = [
    (spec['task_id'], spec['cluster'], spec['domain'])
    for spec in task_specs
  ]
  if list(seen_metadata) != expected_metadata:
    raise RuntimeError('loader task metadata differs from the task schedule')
  for name in analysis_model_names(config):
    if evaluated_ids[name] != expected_ids:
      raise RuntimeError(
        '{} did not evaluate the exact scheduled task IDs'.format(name))

  expected_per_cluster = int(config['tasks_per_cluster'])
  balance = task_balance_report(task_specs)
  expected_cluster_counts = {
    cluster: expected_per_cluster for cluster in cluster_names(config)
  }
  if balance['cluster_counts'] != expected_cluster_counts:
    raise RuntimeError(
      'task counts do not match tasks_per_cluster: {}'.format(
        balance['cluster_counts']))
  expected_rows = len(cluster_names(config)) * expected_per_cluster
  if len(rows) != expected_rows:
    raise RuntimeError(
      'output row count does not equal {}'.format(expected_rows))

  finite = True
  missing = False
  if query_split:
    for name in analysis_model_names(config):
      for split_name in ('A', 'B'):
        for metric in ('loss', 'accuracy'):
          values = results[name][split_name][metric]
          if len(values) != len(task_specs):
            missing = True
          finite = finite and bool(np.isfinite(values).all())
  else:
    for name in analysis_model_names(config):
      for metric in ('loss', 'accuracy'):
        values = results[name][metric]
        if len(values) != len(task_specs):
          missing = True
        finite = finite and bool(np.isfinite(values).all())
    for row in rows:
      missing = missing or any(row.get(field) is None
                               for field in standard_csv_float_fields(config))
      finite = finite and all(
        np.isfinite(float(row[field]))
        for field in standard_csv_float_fields(config))
  if not finite:
    raise FloatingPointError('completed analysis contains NaN or Inf')
  if missing:
    raise RuntimeError('completed analysis contains missing results')

  return {
    'checkpoints_distinct': True,
    'task_counts_match_yaml': True,
    'task_counts_are_500_plus_500': (
      len(cluster_names(config)) == 2 and expected_per_cluster == 500),
    'task_balance': balance,
    'same_task_ids_used_by_all_three_models': True,
    'same_in_memory_episode_batch_used_by_all_three_models': True,
    'each_task_is_single_domain': True,
    'no_nan_or_infinite_results': True,
    'no_missing_results': True,
    'meta_test_manifest_enforced_by_dataset': True,
    'manifest_train_val_test_disjoint': True,
    'class_balanced_query_split_verified': bool(query_split),
  }


def write_csv(path, rows, fieldnames):
  with open(path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def write_json(path, value):
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(value, f, indent=2, allow_nan=False)


def log_standard_summary(summary):
  utils.log('specialist oracle tasks: {}'.format(summary['n_tasks']))
  for name in summary['models']:
    metrics = summary['models'][name]
    utils.log('{}: loss={:.6f}, accuracy={:.6f}'.format(
      name, metrics['mean_query_loss'], metrics['mean_query_accuracy']))
  oracle = summary['per_task_oracle']
  oracle_comparison = summary['oracle_vs_reference']
  router_comparison = summary['domain_router_vs_reference']
  reference = summary['reference_model']
  utils.log(
    'oracle: loss={:.6f}, accuracy={:.6f}, {} gain={:.6f}'.format(
      oracle['mean_query_loss'], oracle['mean_query_accuracy'], reference,
      oracle_comparison['reference_minus_candidate_loss_gain']))
  utils.log('domain-router: loss={:.6f}, {} gain={:.6f}'.format(
    summary['domain_router']['mean_query_loss'], reference,
    router_comparison['reference_minus_candidate_loss_gain']))


def log_query_split_summary(summary):
  noise = summary['reference_query_split_noise_floor']
  crossfit = summary['cross_fitted_oracle']
  reference = summary['reference_model']
  utils.log('query-split tasks: {}'.format(summary['n_tasks']))
  utils.log('{} query-split noise-floor gain={:.6f}'.format(
    reference, noise['absolute_gain']))
  utils.log(
    'cross-fitted oracle: loss={:.6f}, accuracy={:.6f}, {} gain={:.6f}'
    .format(
      crossfit['mean_query_loss'], crossfit['mean_query_accuracy'], reference,
      crossfit['vs_reference']['reference_minus_candidate_loss_gain']))


def log_paired_summary(summary):
  first, second = summary['model_order']
  utils.log(
    'paired checkpoints: domain={}, episodes={}'.format(
      summary['domain'], summary['n_episodes']))
  for name in (first, second):
    metrics = summary['models'][name]
    utils.log('{}: loss={:.6f}, accuracy={:.6f}'.format(
      name, metrics['mean_query_loss'], metrics['mean_query_accuracy']))
  utils.log('{}: mean loss difference={:.6f}, mean accuracy difference={:.6f}'
            .format(
              summary['difference_definition'],
              summary['differences']['mean_query_loss_difference'],
              summary['differences']['mean_query_accuracy_difference']))


def run_paired_analysis(config, args):
  seed = int(args.seed if args.seed is not None else config.get('seed', 0))
  config['seed'] = seed
  seed_everything(seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  manifest_report = validate_manifest(config)
  manifest_report['data_files'] = validate_data_files(config)
  model_records, keep_on_gpu, checkpoint_report = load_analysis_models(
    config, args, device)
  task_specs = paired_task_specs(config)

  if args.validate_only:
    report = {
      'mode': 'paired_checkpoints',
      'config_valid': True,
      'seed': seed,
      'domain': config['domain'],
      'n_episodes': len(task_specs),
      'checkpoints': checkpoint_report,
      'manifest': manifest_report,
      'note': (
        'validate-only checks inputs and the task count; run without '
        '--validate-only to verify sampled metadata, paired batches, and '
        'metrics.'),
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    return

  default_output_dir = os.path.join(
    'save', 'paired_checkpoint_{}_{}_vs_{}'.format(
      name_slug(config['domain']),
      name_slug(paired_model_names(config)[0]),
      name_slug(paired_model_names(config)[1])))
  output_dir = resolve_project_path(
    args.output_dir or config.get('output_dir') or default_output_dir)
  os.makedirs(output_dir, exist_ok=True)
  outputs = {
    'paired_checkpoint_tasks_csv': os.path.join(
      output_dir, 'paired_checkpoint_tasks.csv'),
    'paired_checkpoint_summary_json': os.path.join(
      output_dir, 'paired_checkpoint_summary.json'),
  }
  loader = make_paired_loader(config, task_specs, seed, device)
  results, evaluated_ids, evaluated_sample_ids, metadata_records = \
    evaluate_paired_tasks(
      config, model_records, loader, device, keep_on_gpu)
  rows = make_paired_rows(metadata_records, results, config)
  validation = validate_paired_completed_run(
    config, task_specs, rows, results, evaluated_ids,
    evaluated_sample_ids, metadata_records)
  summary = compute_paired_summary(
    config, rows, results, outputs, checkpoint_report, manifest_report,
    validation, device)
  write_csv(outputs['paired_checkpoint_tasks_csv'], rows,
            paired_csv_fields(config))
  write_json(outputs['paired_checkpoint_summary_json'], summary)
  log_paired_summary(summary)
  utils.log('paired task CSV: {}'.format(
    outputs['paired_checkpoint_tasks_csv']))
  utils.log('paired summary JSON: {}'.format(
    outputs['paired_checkpoint_summary_json']))


def run_analysis(config, args):
  validate_config(config)
  if analysis_mode(config) == 'paired_checkpoints':
    return run_paired_analysis(config, args)
  seed = int(args.seed if args.seed is not None else config.get('seed', 0))
  require_equal('resolved protocol seed', seed, 0)
  config['seed'] = seed
  bootstrap = bootstrap_config(config, args)
  seed_everything(seed)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  manifest_report = validate_manifest(config)
  data_file_report = validate_data_files(config)
  manifest_report['data_files'] = data_file_report
  model_records, keep_on_gpu, checkpoint_report = load_analysis_models(
    config, args, device)

  task_specs = balanced_task_specs(config, seed)
  balance = task_balance_report(task_specs)
  expected_per_cluster = int(config['tasks_per_cluster'])
  require_equal(
    'total task count', balance['total_tasks'],
    len(cluster_names(config)) * expected_per_cluster)
  for cluster in cluster_names(config):
    require_equal(
      '{} task count'.format(cluster),
      balance['cluster_counts'].get(cluster), expected_per_cluster)

  if args.validate_only:
    report = {
      'config_valid': True,
      'checkpoints': checkpoint_report,
      'manifest': manifest_report,
      'task_balance': balance,
      'note': (
        'validate-only checks inputs and the deterministic schedule; run '
        'without --validate-only to verify evaluated metrics and shared '
        'episode batches.'),
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    return

  output_dir = resolve_project_path(
    args.output_dir or config.get('output_dir') or '.')
  os.makedirs(output_dir, exist_ok=True)
  outputs = {
    'specialist_oracle_tasks_csv': os.path.join(
      output_dir, 'specialist_oracle_tasks.csv'),
    'specialist_oracle_summary_json': os.path.join(
      output_dir, 'specialist_oracle_summary.json'),
    'query_split_tasks_csv': os.path.join(
      output_dir, 'query_split_tasks.csv'),
    'query_split_summary_json': os.path.join(
      output_dir, 'query_split_summary.json'),
  }

  standard_loader = make_loader(
    config, task_specs, int(config['test']['n_query']), seed, device)
  standard_results, standard_ids, standard_metadata = \
    evaluate_standard_tasks(
      config, model_records, standard_loader, device, keep_on_gpu)
  standard_rows = make_standard_rows(
    task_specs, standard_results, config=config)
  standard_validation = validate_completed_run(
    config, task_specs, standard_rows, standard_results,
    standard_ids, standard_metadata, query_split=False)
  standard_summary = compute_standard_summary(
    config, standard_rows, standard_results, outputs,
    checkpoint_report, manifest_report, standard_validation, bootstrap)
  write_csv(
    outputs['specialist_oracle_tasks_csv'], standard_rows,
    standard_csv_fields(config))
  write_json(outputs['specialist_oracle_summary_json'], standard_summary)
  log_standard_summary(standard_summary)

  # Release the decoded n_query=15 datasets before constructing their
  # n_query=30 counterparts. This matters for the full 15-domain album.
  del standard_loader
  gc.collect()
  if device.type == 'cuda':
    torch.cuda.empty_cache()

  seed_everything(seed)
  query_loader = make_loader(
    config, task_specs, int(config['query_split']['n_query']), seed, device)
  query_results, query_ids, query_metadata = evaluate_query_split_tasks(
    config, model_records, query_loader, device, keep_on_gpu)
  query_rows = make_query_split_rows(
    task_specs, query_results, config=config)
  query_validation = validate_completed_run(
    config, task_specs, query_rows, query_results,
    query_ids, query_metadata, query_split=True)
  query_summary = compute_query_split_summary(
    config, query_rows, query_results, outputs,
    checkpoint_report, manifest_report, query_validation, bootstrap,
    standard_summary)

  comparison = query_summary[
    'specialist_gain_vs_query_split_noise_floor']
  standard_summary['specialist_gain_vs_query_split_noise_floor'] = comparison
  write_csv(outputs['query_split_tasks_csv'], query_rows,
            query_split_csv_fields(config))
  write_json(outputs['query_split_summary_json'], query_summary)
  write_json(outputs['specialist_oracle_summary_json'], standard_summary)
  log_query_split_summary(query_summary)

  utils.log('specialist task CSV: {}'.format(
    outputs['specialist_oracle_tasks_csv']))
  utils.log('specialist summary JSON: {}'.format(
    outputs['specialist_oracle_summary_json']))
  utils.log('query-split task CSV: {}'.format(
    outputs['query_split_tasks_csv']))
  utils.log('query-split summary JSON: {}'.format(
    outputs['query_split_summary_json']))


def parse_args():
  parser = argparse.ArgumentParser(
    description=(
      'Meta-Album specialist/oracle or paired-checkpoint analysis'))
  parser.add_argument('--config', required=True,
                      help='analysis YAML')
  parser.add_argument('--gpu', type=str, default='0',
                      help='GPU device number(s), or -1 for CPU')
  parser.add_argument('--seed', type=int, default=None,
                      help='override config seed (protocol default: 0)')
  parser.add_argument('--output-dir', type=str, default=None,
                      help='override output directory')
  parser.add_argument('--bootstrap-resamples', type=int, default=None,
                      help='override paired bootstrap resample count')
  parser.add_argument('--efficient', action='store_true',
                      help='enable gradient checkpointing')
  parser.add_argument(
    '--keep-models-on-gpu',
    action=argparse.BooleanOptionalAction,
    default=None,
    help='keep all analysis models on GPU')
  parser.add_argument(
    '--validate-only', action='store_true',
    help='validate checkpoints, manifest, and the YAML task schedule')
  return parser.parse_args()


if __name__ == '__main__':
  cli_args = parse_args()
  config_path = os.path.abspath(os.path.expanduser(cli_args.config))
  with open(config_path, 'r', encoding='utf-8') as f:
    analysis_config = yaml.load(f, Loader=yaml.FullLoader)
  analysis_config = normalized_config(analysis_config)
  analysis_config['_config_source'] = {
    'path': config_path,
    'sha256': sha256_file(config_path),
  }
  if len(cli_args.gpu.split(',')) > 1:
    analysis_config['_parallel'] = True
    analysis_config['_gpu'] = cli_args.gpu
  utils.set_gpu(cli_args.gpu)
  run_analysis(analysis_config, cli_args)
