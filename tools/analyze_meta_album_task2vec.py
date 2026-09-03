"""Train-only Task2Vec geometry for raw Meta-Album domains.

This diagnostic is intentionally independent from AG-Meta's MAML/Gradient
Transport models. It reads raw/processed Meta-Album domain directories in the
standard ``labels.csv + images/`` layout, uses only manifest train classes,
fits a fresh linear head per balanced repeat, and estimates a Monte-Carlo
diagonal Fisher embedding from a fixed ImageNet-pretrained ResNet-34 probe.

Important: the Task2Vec path never reads the 84x84 AG-Meta pickle files.
``--image-size 0`` (default) chooses the largest common size up to 224 that
does not upscale any image selected for the analysis.
"""

import argparse
import csv
import glob
import hashlib
import json
import os
import platform
import random
import sys
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
  sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler
import torchvision
from torchvision import transforms
from torchvision.models import resnet34

try:
  from torchvision.models import ResNet34_Weights
except ImportError:
  ResNet34_Weights = None

from datasets.meta_album import _load_class_split_manifest

DEFAULT_DATA_ROOT = 'materials/task2vec_album_raw'
DEFAULT_MANIFEST_ROOT = 'materials/album'
DEFAULT_OUTPUT_DIR = 'outputs/task2vec_meta_album'
OLD_CLUSTERS = {
  'A': ['BRD', 'DOG', 'AWA'],
  'B': ['MD_MIX', 'PLK'],
}
OUTPUT_FILENAMES = (
  'task2vec_embeddings.npz',
  'domain_names.json',
  'task2vec_distance_matrix.csv',
  'cosine_distance_matrix.csv',
  'clustering_results.json',
  'dendrogram.png',
  'pca_domains.png',
  'sampling_summary.json',
  'run_config.json',
)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


class RawMetaAlbumSample(Dataset):
  """Indexed view over raw Meta-Album image paths."""

  def __init__(self, image_paths, image_indices, targets, transform):
    self.image_paths = list(image_paths)
    self.image_indices = np.asarray(image_indices, dtype=np.int64)
    self.targets = np.asarray(targets, dtype=np.int64)
    self.transform = transform

  def __len__(self):
    return len(self.image_indices)

  def __getitem__(self, index):
    path = self.image_paths[int(self.image_indices[index])]
    with Image.open(path) as image:
      image = image.convert('RGB')
      tensor = self.transform(image)
    return tensor, int(self.targets[index])


def absolute_path(path):
  return os.path.abspath(os.path.expanduser(path))


def write_json(path, value):
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(value, f, indent=2, allow_nan=False)


def sha256_file(path):
  digest = hashlib.sha256()
  with open(path, 'rb') as f:
    for block in iter(lambda: f.read(1024 * 1024), b''):
      digest.update(block)
  return digest.hexdigest()


def derived_seed(base_seed, domain, repeat, purpose):
  payload = '{}\0{}\0{}\0{}'.format(
    int(base_seed), domain, int(repeat), purpose).encode('utf-8')
  return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big')


def seed_everything(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
  if hasattr(torch.backends, 'cudnn'):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
  try:
    torch.use_deterministic_algorithms(True, warn_only=True)
  except (AttributeError, TypeError):
    pass


def validate_positive(name, value):
  if value <= 0:
    raise ValueError('{} must be positive, got {}'.format(name, value))


def validate_args(args):
  for name in (
      'classes_per_domain', 'images_per_class', 'num_repeats',
      'batch_size', 'head_epochs', 'fisher_num_samples',
      'fisher_batch_size', 'top_pairs'):
    validate_positive('--' + name.replace('_', '-'), getattr(args, name))
  if args.image_size < 0:
    raise ValueError('--image-size must be 0 (auto/no-upscale) or positive')
  if args.num_workers < 0:
    raise ValueError('--num-workers cannot be negative')
  if args.head_learning_rate <= 0:
    raise ValueError('--head-learning-rate must be positive')
  if args.head_weight_decay < 0:
    raise ValueError('--head-weight-decay cannot be negative')


def resolve_device(value):
  if value == 'auto':
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  device = torch.device(value)
  if device.type == 'cuda' and not torch.cuda.is_available():
    raise RuntimeError('CUDA requested but unavailable; use --device cpu')
  return device


def prepare_output_dir(output_dir, overwrite):
  output_dir = absolute_path(output_dir)
  existing = [
    name for name in OUTPUT_FILENAMES
    if os.path.exists(os.path.join(output_dir, name))
  ]
  if existing and not overwrite:
    raise FileExistsError(
      'refusing to overwrite an existing Task2Vec run in {} (found {}). '
      'Use another --output-dir or pass --overwrite.'.format(
        output_dir, ', '.join(existing)))
  os.makedirs(output_dir, exist_ok=True)
  return output_dir


def _read_info_json(domain_dir):
  path = os.path.join(domain_dir, 'info.json')
  if not os.path.isfile(path):
    return {}
  with open(path, 'r', encoding='utf-8') as f:
    value = json.load(f)
  return value if isinstance(value, dict) else {}


def _domain_aliases(domain_dir):
  info = _read_info_json(domain_dir)
  aliases = {os.path.basename(os.path.normpath(domain_dir))}
  for key in ('dataset_name', 'dataset_id', 'name', 'abbreviation'):
    value = info.get(key)
    if isinstance(value, str) and value.strip():
      aliases.add(value.strip())
  return aliases


def discover_raw_domains(data_root, requested_domains=None):
  """Discover standard Meta-Album folders containing labels.csv + images/."""
  data_root = absolute_path(data_root)
  if not os.path.isdir(data_root):
    raise FileNotFoundError('raw Task2Vec data root not found: {}'.format(data_root))

  candidates = []
  for root, dirs, files in os.walk(data_root):
    if 'labels.csv' in files and os.path.isdir(os.path.join(root, 'images')):
      candidates.append(absolute_path(root))
      dirs[:] = []

  if not candidates:
    raise RuntimeError(
      'no raw Meta-Album domain folders found under {}. Expected '
      '<domain>/labels.csv and <domain>/images/.'.format(data_root))

  alias_to_paths = {}
  for path in candidates:
    for alias in _domain_aliases(path):
      alias_to_paths.setdefault(alias.casefold(), []).append(path)

  if requested_domains:
    domains = [str(x) for x in requested_domains]
    if len(domains) != len(set(domains)):
      raise ValueError('--domains contains duplicates')
  else:
    # Use folder basenames as canonical names when no subset is requested.
    domains = sorted(
      [os.path.basename(os.path.normpath(path)) for path in candidates],
      key=str.casefold)

  resolved = {}
  for domain in domains:
    matches = alias_to_paths.get(domain.casefold(), [])
    if not matches:
      raise FileNotFoundError(
        "raw domain '{}' not found under {}. Discovered aliases: {}".format(
          domain, data_root,
          ', '.join(sorted(alias_to_paths.keys())[:50])))
    unique = sorted(set(matches))
    if len(unique) != 1:
      raise ValueError(
        "raw domain '{}' is ambiguous: {}".format(domain, ', '.join(unique)))
    resolved[domain] = unique[0]
  return dict(sorted(resolved.items(), key=lambda item: item[0].casefold()))


def load_raw_meta_album_domain(domain_dir):
  """Load image paths and class labels from standard Meta-Album metadata."""
  domain_dir = absolute_path(domain_dir)
  info = _read_info_json(domain_dir)
  labels_path = os.path.join(domain_dir, 'labels.csv')
  images_dir = os.path.join(domain_dir, 'images')
  if not os.path.isfile(labels_path):
    raise FileNotFoundError('labels.csv missing: {}'.format(labels_path))
  if not os.path.isdir(images_dir):
    raise FileNotFoundError('images/ missing: {}'.format(images_dir))

  image_column = str(info.get('image_column_name') or 'FILE_NAME')
  category_column = str(info.get('category_column_name') or 'CATEGORY')

  image_paths = []
  labels = []
  with open(labels_path, 'r', encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    if not reader.fieldnames:
      raise ValueError('labels.csv has no header: {}'.format(labels_path))
    if image_column not in reader.fieldnames or category_column not in reader.fieldnames:
      raise KeyError(
        '{} must contain columns {!r} and {!r}; got {}'.format(
          labels_path, image_column, category_column, reader.fieldnames))
    for row in reader:
      filename = str(row[image_column]).strip()
      category = str(row[category_column]).strip()
      if not filename or not category:
        continue
      path = os.path.join(images_dir, filename)
      if not os.path.isfile(path):
        raise FileNotFoundError(
          'labels.csv references missing image: {}'.format(path))
      if os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS:
        # Do not reject uncommon-but-readable extensions solely by suffix.
        pass
      image_paths.append(absolute_path(path))
      labels.append(category)

  if not image_paths:
    raise RuntimeError('no usable rows found in {}'.format(labels_path))
  return image_paths, np.asarray(labels, dtype=str), {
    'domain_dir': domain_dir,
    'labels_csv': labels_path,
    'images_dir': images_dir,
    'image_column_name': image_column,
    'category_column_name': category_column,
    'raw_image_count': len(image_paths),
    'info_json': os.path.join(domain_dir, 'info.json') if info else None,
  }


def _candidate_manifest_domains(path):
  try:
    with open(path, 'r', encoding='utf-8') as f:
      value = json.load(f)
  except (OSError, json.JSONDecodeError):
    return set()
  if not isinstance(value, dict):
    return set()
  return {
    str(domain) for domain, entry in value.items()
    if isinstance(entry, dict) and
    all(isinstance(entry.get(split), list)
        for split in ('train', 'val', 'test'))
  }


def discover_domain_manifests(manifest_root, domains, explicit_paths=None):
  """Resolve one canonical train/val/test split per requested domain."""
  manifest_root = absolute_path(manifest_root)
  explicit = bool(explicit_paths)
  if explicit:
    candidates = [absolute_path(path) for path in explicit_paths]
    missing = [path for path in candidates if not os.path.isfile(path)]
    if missing:
      raise FileNotFoundError(
        'class split manifest(s) not found: {}'.format(', '.join(missing)))
  else:
    candidates = sorted(
      glob.glob(os.path.join(manifest_root, '*.json')),
      key=lambda path: os.path.basename(path).casefold())

  resolved = {}
  sources = {}
  for path in candidates:
    candidate_domains = _candidate_manifest_domains(path)
    for domain in sorted(set(domains) & candidate_domains):
      split = _load_class_split_manifest(path, [domain])[domain]
      equivalent = domain in resolved and all(
        set(resolved[domain][name]) == set(split[name])
        for name in ('train', 'val', 'test'))
      if domain in resolved and not equivalent:
        raise ValueError(
          "conflicting class split manifests for domain '{}': {} and {}. "
          'Pass the intended file with --class-split-manifest.'.format(
            domain, sources[domain][0], absolute_path(path)))
      resolved[domain] = split
      sources.setdefault(domain, []).append(absolute_path(path))

  unresolved = sorted(set(domains) - set(resolved))
  if explicit and unresolved:
    raise KeyError(
      'explicit manifest(s) do not cover requested domain(s): {}'.format(
        ', '.join(unresolved)))
  if not resolved:
    raise RuntimeError(
      'no usable train/val/test manifest covers the raw domains. '
      'Use --class-split-manifest explicitly.')
  return resolved, sources, unresolved, candidates


def validate_manifest_labels(domain, labels, split, sources):
  actual = set(np.unique(labels).tolist())
  manifest_sets = {
    name: set(str(value) for value in split[name])
    for name in ('train', 'val', 'test')
  }
  assigned = set().union(*manifest_sets.values())
  unknown = sorted(assigned - actual)
  unassigned = sorted(actual - assigned)
  if unknown:
    raise ValueError(
      "manifest for '{}' names classes absent from raw labels: {} "
      '(source: {})'.format(domain, ', '.join(unknown[:10]), ', '.join(sources)))
  if unassigned:
    raise ValueError(
      "manifest for '{}' leaves raw classes unassigned: {} "
      '(source: {})'.format(domain, ', '.join(unassigned[:10]), ', '.join(sources)))
  if not split['train']:
    raise ValueError("domain '{}' has no train classes".format(domain))
  return {
    'total_classes': len(actual),
    'train_classes': len(manifest_sets['train']),
    'val_classes_excluded': len(manifest_sets['val']),
    'test_classes_excluded': len(manifest_sets['test']),
    'full_manifest_coverage': True,
    'splits_are_disjoint': True,
  }


def sample_train_classes(
    labels, train_classes, classes_per_domain, images_per_class, seed):
  rng = np.random.default_rng(seed)
  train_classes = np.asarray(sorted(str(x) for x in train_classes), dtype=str)
  n_classes = min(classes_per_domain, len(train_classes))
  selected = rng.choice(train_classes, size=n_classes, replace=False).tolist()

  image_indices = []
  targets = []
  class_records = []
  for target, class_name in enumerate(selected):
    available = np.flatnonzero(labels == class_name)
    if not len(available):
      raise RuntimeError(
        "train class '{}' has no raw images".format(class_name))
    n_images = min(images_per_class, len(available))
    chosen = rng.choice(available, size=n_images, replace=False)
    image_indices.extend(chosen.tolist())
    targets.extend([target] * n_images)
    class_records.append({
      'class_name': class_name,
      'available_images': int(len(available)),
      'sampled_images': int(n_images),
      'used_fewer_images_than_requested': bool(n_images < images_per_class),
    })

  order = rng.permutation(len(image_indices))
  image_indices = np.asarray(image_indices, dtype=np.int64)[order]
  targets = np.asarray(targets, dtype=np.int64)[order]
  return image_indices, targets, {
    'sampling_seed': int(seed),
    'requested_classes': int(classes_per_domain),
    'available_train_classes': int(len(train_classes)),
    'sampled_classes': int(n_classes),
    'used_fewer_classes_than_requested': bool(n_classes < classes_per_domain),
    'requested_images_per_class': int(images_per_class),
    'sampled_images_total': int(len(image_indices)),
    'selected_classes': class_records,
  }


def inspect_image_sizes(paths):
  sizes = []
  for path in sorted(set(paths)):
    with Image.open(path) as image:
      width, height = image.size
    sizes.append((int(width), int(height)))
  if not sizes:
    raise RuntimeError('cannot infer probe size from an empty image set')
  short_sides = [min(w, h) for w, h in sizes]
  return {
    'inspected_images': len(sizes),
    'min_short_side': int(min(short_sides)),
    'max_short_side': int(max(short_sides)),
    'min_width': int(min(w for w, _ in sizes)),
    'max_width': int(max(w for w, _ in sizes)),
    'min_height': int(min(h for _, h in sizes)),
    'max_height': int(max(h for _, h in sizes)),
    'unique_sizes_preview': [list(x) for x in sorted(set(sizes))[:20]],
  }


def resolve_probe_image_size(requested_size, selected_paths, max_auto_size=224):
  report = inspect_image_sizes(selected_paths)
  if requested_size == 0:
    target = min(int(max_auto_size), report['min_short_side'])
    mode = 'auto_no_upscale'
  else:
    target = int(requested_size)
    mode = 'explicit_no_upscale'
    if report['min_short_side'] < target:
      raise ValueError(
        '--image-size {} would upscale at least one selected raw image '
        '(minimum short side is {}). Choose <= {} or use --image-size 0.'
        .format(target, report['min_short_side'],
                report['min_short_side']))
  report.update({
    'requested_image_size': int(requested_size),
    'resolved_probe_image_size': int(target),
    'mode': mode,
    'upscaling_allowed': False,
    'auto_cap': int(max_auto_size),
  })
  return target, report


def make_probe(device, image_size):
  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  if ResNet34_Weights is not None:
    weights = ResNet34_Weights.DEFAULT
    probe = resnet34(weights=weights)
    weights_name = '{}.{}'.format(ResNet34_Weights.__name__, weights.name)
    weights_url = getattr(weights, 'url', None)
  else:
    probe = resnet34(pretrained=True)
    weights_name = 'legacy torchvision pretrained=True'
    weights_url = None

  # Resize shorter side to target and center-crop. resolve_probe_image_size()
  # guarantees target <= every selected short side, so this never upscales.
  transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
  ])

  feature_dim = int(probe.fc.in_features)
  probe.fc = nn.Identity()
  probe.eval()
  probe.to(device)
  for parameter in probe.parameters():
    parameter.requires_grad_(True)

  return probe, transform, feature_dim, {
    'architecture': 'torchvision.models.resnet34',
    'weights': weights_name,
    'weights_url': weights_url,
    'feature_dim': feature_dim,
    'image_size': int(image_size),
    'input_policy': 'resize/crop without upscaling',
    'normalization_mean': mean,
    'normalization_std': std,
    'probe_mode': 'eval (fixed ImageNet BatchNorm running statistics)',
  }


def probe_state_sha256(probe):
  digest = hashlib.sha256()
  for name, tensor in probe.state_dict().items():
    digest.update(name.encode('utf-8'))
    array = tensor.detach().cpu().contiguous().numpy()
    digest.update(str(array.dtype).encode('ascii'))
    digest.update(str(array.shape).encode('ascii'))
    digest.update(array.tobytes())
  return digest.hexdigest()


def fisher_layers(probe, include_batchnorm):
  accepted = (nn.Conv2d, nn.BatchNorm2d) if include_batchnorm else (nn.Conv2d,)
  result = [
    (name, module) for name, module in probe.named_modules()
    if isinstance(module, accepted) and module.weight is not None
  ]
  if not result:
    raise RuntimeError('no feature-extractor Fisher layers selected')
  return result


def embedding_layout(layers):
  layout = []
  component_names = []
  start = 0
  for name, module in layers:
    length = int(module.weight.shape[0])
    layout.append({
      'module': name,
      'type': type(module).__name__,
      'start': start,
      'end_exclusive': start + length,
      'weight_shape': list(module.weight.shape),
    })
    component_names.extend(
      '{}.weight[{}]'.format(name, index) for index in range(length))
    start += length
  return layout, component_names


def make_image_loader(dataset, batch_size, num_workers, device):
  return DataLoader(
    dataset,
    batch_size=min(batch_size, len(dataset)),
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device.type == 'cuda'),
    drop_last=False)


def make_balanced_fisher_sampler(targets, num_samples, seed):
  targets = np.asarray(targets, dtype=np.int64)
  if targets.ndim != 1 or not len(targets):
    raise ValueError('Fisher targets must be a non-empty 1D array')
  validate_positive('--fisher-num-samples', num_samples)
  classes, counts = np.unique(targets, return_counts=True)
  if len(classes) < 2:
    raise ValueError('Task2Vec Fisher requires at least two classes')
  class_counts = {int(cls): int(count) for cls, count in zip(classes, counts)}
  weights = np.asarray(
    [1.0 / class_counts[int(target)] for target in targets], dtype=np.float64)
  generator = torch.Generator()
  generator.manual_seed(int(seed))
  sampler = WeightedRandomSampler(
    weights=torch.as_tensor(weights, dtype=torch.double),
    num_samples=int(num_samples),
    replacement=True,
    generator=generator)
  return sampler, {
    'sampling': 'class-balanced WeightedRandomSampler with replacement',
    'num_samples': int(num_samples),
    'seed': int(seed),
    'class_counts_in_source_subset': class_counts,
  }


def make_fisher_loader(
    dataset, targets, batch_size, num_workers, device, num_samples, seed):
  sampler, sampler_report = make_balanced_fisher_sampler(
    targets, num_samples, seed)
  loader = DataLoader(
    dataset,
    batch_size=min(batch_size, num_samples),
    sampler=sampler,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device.type == 'cuda'),
    drop_last=True)
  return loader, sampler_report


def cache_probe_features(probe, loader, device):
  batches = []
  probe.eval()
  with torch.no_grad():
    for images, _ in loader:
      images = images.to(device, non_blocking=(device.type == 'cuda'))
      features = probe(images)
      if features.ndim != 2:
        raise RuntimeError(
          'ResNet-34 returned {}, expected [N,D]'.format(tuple(features.shape)))
      batches.append(features.detach().float().cpu())
  return torch.cat(batches, dim=0)


def fit_linear_head(
    features, targets, num_classes, feature_dim, device, args, seed):
  seed_everything(seed)
  head = nn.Linear(feature_dim, num_classes).to(device)
  optimizer = torch.optim.Adam(
    head.parameters(),
    lr=args.head_learning_rate,
    weight_decay=args.head_weight_decay)
  generator = torch.Generator()
  generator.manual_seed(seed)
  dataset = TensorDataset(features, torch.as_tensor(targets, dtype=torch.long))
  loader = DataLoader(
    dataset,
    batch_size=min(args.batch_size, len(dataset)),
    shuffle=True,
    num_workers=0,
    generator=generator,
    drop_last=False)

  last_loss = None
  head.train()
  for _ in range(args.head_epochs):
    loss_sum = 0.0
    n_seen = 0
    for batch_features, batch_targets in loader:
      batch_features = batch_features.to(device)
      batch_targets = batch_targets.to(device)
      optimizer.zero_grad(set_to_none=True)
      loss = F.cross_entropy(head(batch_features), batch_targets)
      loss.backward()
      optimizer.step()
      loss_sum += float(loss.detach().cpu()) * len(batch_targets)
      n_seen += len(batch_targets)
    last_loss = loss_sum / n_seen

  head.eval()
  correct = 0
  with torch.no_grad():
    for batch_features, batch_targets in DataLoader(
        dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=False):
      predictions = head(batch_features.to(device)).argmax(dim=1).cpu()
      correct += int((predictions == batch_targets).sum())
  for parameter in head.parameters():
    parameter.requires_grad_(False)
  return head, {
    'initialization_and_shuffle_seed': int(seed),
    'epochs': int(args.head_epochs),
    'learning_rate': float(args.head_learning_rate),
    'weight_decay': float(args.head_weight_decay),
    'final_training_loss': float(last_loss),
    'training_accuracy': float(correct / len(dataset)),
  }


def monte_carlo_filter_fisher(
    probe, head, loader, layers, device, seed, sampler_report=None):
  seed_everything(seed)
  try:
    generator = torch.Generator(device=device)
  except TypeError:
    generator = torch.Generator(device=device.type)
  generator.manual_seed(seed)
  probe.eval()
  head.eval()

  length = sum(int(module.weight.shape[0]) for _, module in layers)
  accumulator = np.zeros(length, dtype=np.float64)
  n_batches = 0
  n_examples = 0
  for images, _ in loader:
    images = images.to(device, non_blocking=(device.type == 'cuda'))
    probe.zero_grad(set_to_none=True)
    logits = head(probe(images))
    probabilities = F.softmax(logits, dim=-1)
    sampled_targets = torch.multinomial(
      probabilities, 1, replacement=True, generator=generator).view(-1)
    loss = F.cross_entropy(logits, sampled_targets)
    loss.backward()

    start = 0
    for name, module in layers:
      gradient = module.weight.grad
      if gradient is None:
        raise RuntimeError("Fisher gradient missing for '{}'".format(name))
      value = gradient.detach().float().square().reshape(
        gradient.shape[0], -1).mean(dim=1)
      end = start + len(value)
      accumulator[start:end] += value.cpu().numpy()
      start = end
    n_batches += 1
    n_examples += len(images)

  probe.zero_grad(set_to_none=True)
  if not n_batches:
    raise RuntimeError('Fisher loader produced no minibatches')
  embedding = (accumulator / n_batches).astype(np.float32)
  if not np.isfinite(embedding).all():
    raise FloatingPointError('Task2Vec embedding contains NaN or Inf')
  report = {
    'label_sampling_seed': int(seed),
    'examples_drawn_after_drop_last': int(n_examples),
    'minibatches': int(n_batches),
    'estimator': (
      'Monte Carlo predictive-label diagonal Fisher; class-balanced '
      'replacement sampling; squared gradients of minibatch-mean '
      'cross-entropy averaged across minibatches'),
  }
  if sampler_report is not None:
    report['input_sampler'] = sampler_report
  return embedding, report


def cosine_distance(left, right, epsilon=1e-12):
  left = np.asarray(left, dtype=np.float64)
  right = np.asarray(right, dtype=np.float64)
  left_norm = float(np.linalg.norm(left))
  right_norm = float(np.linalg.norm(right))
  if left_norm <= epsilon and right_norm <= epsilon:
    return 0.0
  if left_norm <= epsilon or right_norm <= epsilon:
    return 1.0
  similarity = float(np.dot(left, right) / (left_norm * right_norm))
  return float(1.0 - np.clip(similarity, -1.0, 1.0))


def task2vec_distance(left, right, epsilon=1e-12):
  left = np.asarray(left, dtype=np.float64)
  right = np.asarray(right, dtype=np.float64)
  denominator = left + right
  scaled_left = np.divide(
    left, denominator, out=np.zeros_like(left),
    where=np.abs(denominator) > epsilon)
  scaled_right = np.divide(
    right, denominator, out=np.zeros_like(right),
    where=np.abs(denominator) > epsilon)
  return cosine_distance(scaled_left, scaled_right, epsilon=epsilon)


def pairwise_distances(embeddings, distance_function):
  n = len(embeddings)
  matrix = np.zeros((n, n), dtype=np.float64)
  for left in range(n):
    for right in range(left + 1, n):
      value = distance_function(embeddings[left], embeddings[right])
      matrix[left, right] = matrix[right, left] = value
  if not np.isfinite(matrix).all():
    raise FloatingPointError('distance matrix contains NaN or Inf')
  return matrix


def write_distance_csv(path, domain_names, matrix):
  with open(path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['domain'] + list(domain_names))
    for domain, row in zip(domain_names, matrix):
      writer.writerow([domain] + ['{:.12g}'.format(value) for value in row])


def agglomerative_labels(distance_matrix, n_clusters):
  kwargs = {'n_clusters': n_clusters, 'linkage': 'average'}
  try:
    model = AgglomerativeClustering(metric='precomputed', **kwargs)
  except TypeError:
    model = AgglomerativeClustering(affinity='precomputed', **kwargs)
  return model.fit_predict(distance_matrix)


def labels_to_clusters(domain_names, labels):
  result = {}
  for domain, label in zip(domain_names, labels):
    result.setdefault(str(int(label)), []).append(domain)
  return dict(sorted(result.items(), key=lambda item: int(item[0])))


def clustering_analysis(domain_names, distance_matrix):
  n_domains = len(domain_names)
  if n_domains < 2:
    raise ValueError('at least two domains required')
  condensed = squareform(distance_matrix, checks=True)
  linkage_matrix = hierarchy.linkage(
    condensed, method='average', optimal_ordering=True)
  fixed_k = {}
  silhouette = {}
  max_candidate = min(10, n_domains - 1)
  for k in range(2, max(6, max_candidate + 1)):
    if k >= n_domains:
      if k <= 5:
        fixed_k[str(k)] = {
          'available': False,
          'reason': 'requires more than {} domains'.format(k),
        }
      continue
    labels = agglomerative_labels(distance_matrix, k)
    try:
      score = float(silhouette_score(
        distance_matrix, labels, metric='precomputed'))
      if not np.isfinite(score):
        score = None
    except ValueError:
      score = None
    if score is not None:
      silhouette[str(k)] = score
    if k <= 5:
      fixed_k[str(k)] = {
        'available': True,
        'clusters': labels_to_clusters(domain_names, labels),
        'silhouette_score': score,
      }
  suggested_k = int(max(silhouette, key=silhouette.get)) if silhouette else None
  return linkage_matrix, {
    'distance': 'symmetric_task2vec',
    'hierarchical_linkage': 'average',
    'linkage_matrix': linkage_matrix.tolist(),
    'fixed_k_agglomerative': fixed_k,
    'exploratory_silhouette_scores': silhouette,
    'suggested_k_by_max_silhouette': suggested_k,
    'suggested_k_is_exploratory_not_confirmatory': True,
  }


def plot_dendrogram(path, domain_names, linkage_matrix):
  width = max(9.0, 0.55 * len(domain_names))
  fig, axis = plt.subplots(figsize=(width, 6.5))
  hierarchy.dendrogram(
    linkage_matrix, labels=domain_names, leaf_rotation=45,
    leaf_font_size=9, ax=axis)
  axis.set_title('Meta-Album domain hierarchy (Task2Vec distance)')
  axis.set_ylabel('Average-linkage distance')
  fig.tight_layout()
  fig.savefig(path, dpi=180, bbox_inches='tight')
  plt.close(fig)


def pca_projection(embeddings):
  model = PCA(n_components=2, svd_solver='full')
  coordinates = model.fit_transform(np.asarray(embeddings, dtype=np.float64))
  coordinates = np.nan_to_num(coordinates)
  explained = np.nan_to_num(model.explained_variance_ratio_)
  return coordinates, explained


def plot_projection(path, domain_names, coordinates, explained_variance):
  fig, axis = plt.subplots(figsize=(9, 7))
  axis.scatter(coordinates[:, 0], coordinates[:, 1], s=55, alpha=0.85)
  for domain, (x, y) in zip(domain_names, coordinates):
    axis.annotate(domain, (x, y), xytext=(4, 4),
                  textcoords='offset points', fontsize=9)
  axis.set_xlabel('PC1 ({:.1%} variance)'.format(explained_variance[0]))
  axis.set_ylabel('PC2 ({:.1%} variance)'.format(explained_variance[1]))
  axis.set_title('PCA of Meta-Album Task2Vec domain embeddings')
  fig.tight_layout()
  fig.savefig(path, dpi=180, bbox_inches='tight')
  plt.close(fig)


def maybe_plot_umap(output_dir, domain_names, embeddings, seed, enabled):
  result = {'enabled': bool(enabled), 'created': False, 'output': None}
  if not enabled:
    result['reason'] = 'disabled'
    return result
  try:
    import umap
  except ImportError:
    result['reason'] = 'umap-learn not installed'
    return result
  if len(domain_names) < 4:
    result['reason'] = 'needs at least four domains'
    return result
  n_neighbors = min(15, len(domain_names) - 1)
  reducer = umap.UMAP(
    n_components=2, n_neighbors=n_neighbors, metric='cosine',
    random_state=seed, transform_seed=seed)
  coordinates = reducer.fit_transform(embeddings)
  path = os.path.join(output_dir, 'umap_domains.png')
  fig, axis = plt.subplots(figsize=(9, 7))
  axis.scatter(coordinates[:, 0], coordinates[:, 1], s=55, alpha=0.85)
  for domain, (x, y) in zip(domain_names, coordinates):
    axis.annotate(domain, (x, y), xytext=(4, 4),
                  textcoords='offset points', fontsize=9)
  axis.set_title('UMAP of Meta-Album Task2Vec domain embeddings')
  fig.tight_layout()
  fig.savefig(path, dpi=180, bbox_inches='tight')
  plt.close(fig)
  result.update({
    'created': True, 'output': path, 'n_neighbors': n_neighbors,
    'metric': 'cosine'})
  return result


def finite_mean(values):
  return float(np.mean(values)) if values else None


def old_cluster_analysis(domain_names, distance_matrix):
  index = {domain: i for i, domain in enumerate(domain_names)}
  present = {
    name: [domain for domain in members if domain in index]
    for name, members in OLD_CLUSTERS.items()
  }
  missing = {
    name: [domain for domain in members if domain not in index]
    for name, members in OLD_CLUSTERS.items()
  }

  def within(members):
    return [
      float(distance_matrix[index[members[i]], index[members[j]]])
      for i in range(len(members))
      for j in range(i + 1, len(members))
    ]

  within_a_values = within(present['A'])
  within_b_values = within(present['B'])
  between_values = [
    float(distance_matrix[index[left], index[right]])
    for left in present['A'] for right in present['B']
  ]
  within_a_mean = finite_mean(within_a_values)
  within_b_mean = finite_mean(within_b_values)
  between_mean = finite_mean(between_values)

  pooled_within = finite_mean(within_a_values + within_b_values)
  pooled_ratio = (
    float(between_mean / pooled_within)
    if between_mean is not None and pooled_within is not None
    and pooled_within > 0 else None)

  balanced_within = None
  balanced_ratio = None
  if within_a_mean is not None and within_b_mean is not None:
    balanced_within = float((within_a_mean + within_b_mean) / 2.0)
    if between_mean is not None and balanced_within > 0:
      balanced_ratio = float(between_mean / balanced_within)

  return {
    'definition': OLD_CLUSTERS,
    'complete_comparison': not any(missing.values()),
    'comparison_is_partial': bool(any(missing.values())),
    'available_members': present,
    'missing_members': missing,
    'within_A_average_distance': within_a_mean,
    'within_B_average_distance': within_b_mean,
    'A_to_B_average_distance': between_mean,
    'within_cluster_distance': pooled_within,
    'within_cluster_distance_definition': (
      'pooled mean over all available unordered within-A and within-B pairs'),
    'separation_ratio': pooled_ratio,
    'separation_ratio_definition': (
      'A_to_B_average_distance / pooled within_cluster_distance'),
    'balanced_within_cluster_distance': balanced_within,
    'balanced_within_cluster_distance_definition': (
      '(within_A_average_distance + within_B_average_distance) / 2'),
    'balanced_separation_ratio': balanced_ratio,
    'balanced_separation_ratio_definition': (
      'A_to_B_average_distance / balanced_within_cluster_distance'),
  }


def nearest_neighbors(domain_names, distance_matrix, requested):
  index = {domain: i for i, domain in enumerate(domain_names)}
  result = {}
  for domain in requested:
    if domain not in index:
      result[domain] = {'available': False, 'neighbors': []}
      continue
    position = index[domain]
    order = np.argsort(distance_matrix[position])
    result[domain] = {
      'available': True,
      'neighbors': [
        {'domain': domain_names[j],
         'distance': float(distance_matrix[position, j])}
        for j in order if j != position
      ],
    }
  return result


def ranked_pairs(domain_names, distance_matrix, limit):
  pairs = [
    {'left': domain_names[i], 'right': domain_names[j],
     'distance': float(distance_matrix[i, j])}
    for i in range(len(domain_names))
    for j in range(i + 1, len(domain_names))
  ]
  pairs.sort(key=lambda item: item['distance'])
  return pairs[:limit], list(reversed(pairs[-limit:]))


def print_summary(
    closest, distant, clustering, old_comparison, nearest, top_pairs):
  print('\nClosest domain pairs:')
  for rank, pair in enumerate(closest, 1):
    print('{}. {} - {} : {:.6f}'.format(
      rank, pair['left'], pair['right'], pair['distance']))
  print('\nMost distant domain pairs:')
  for rank, pair in enumerate(distant, 1):
    print('{}. {} - {} : {:.6f}'.format(
      rank, pair['left'], pair['right'], pair['distance']))

  for k in range(2, 6):
    record = clustering['fixed_k_agglomerative'][str(k)]
    print('\nClusters for k={}:'.format(k))
    if not record['available']:
      print('  unavailable: {}'.format(record['reason']))
      continue
    for label, members in record['clusters'].items():
      print('  Cluster {}: {}'.format(label, members))

  print('\nPrevious Conv4-selected clusters under Task2Vec:')
  for key in (
      'within_A_average_distance', 'within_B_average_distance',
      'A_to_B_average_distance', 'within_cluster_distance',
      'separation_ratio', 'balanced_within_cluster_distance',
      'balanced_separation_ratio'):
    value = old_comparison[key]
    print('  {}: {}'.format(
      key, 'unavailable' if value is None else '{:.6f}'.format(value)))

  print('\nNearest neighbors of previous-cluster domains:')
  for domain in OLD_CLUSTERS['A'] + OLD_CLUSTERS['B']:
    record = nearest[domain]
    if not record['available']:
      print('  {}: unavailable'.format(domain))
      continue
    preview = record['neighbors'][:top_pairs]
    print('  {}: {}'.format(
      domain, ', '.join('{} ({:.6f})'.format(
        item['domain'], item['distance']) for item in preview)))


def environment_report(device):
  return {
    'python': platform.python_version(),
    'platform': platform.platform(),
    'numpy': np.__version__,
    'torch': torch.__version__,
    'torchvision': torchvision.__version__,
    'device': str(device),
    'cuda_available': bool(torch.cuda.is_available()),
    'cuda_version': torch.version.cuda,
    'cudnn_version': (
      torch.backends.cudnn.version() if torch.cuda.is_available() else None),
  }


def git_commit():
  try:
    import subprocess
    result = subprocess.run(
      ['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT,
      capture_output=True, text=True, check=True)
    return result.stdout.strip()
  except Exception:
    return None


def run(args):
  validate_args(args)
  seed_everything(args.seed)
  device = resolve_device(args.device)
  data_root = absolute_path(args.data_root)
  manifest_root = absolute_path(args.manifest_root)

  requested_domains = args.domains
  if not requested_domains and args.class_split_manifest:
    # Explicit manifests define the intended canonical domain set.
    requested_domains = sorted(set().union(*[
      _candidate_manifest_domains(path)
      for path in args.class_split_manifest
    ]), key=str.casefold)

  raw_paths = discover_raw_domains(data_root, requested_domains)
  split_by_domain, manifest_sources, skipped, manifest_candidates = \
    discover_domain_manifests(
      manifest_root, list(raw_paths), args.class_split_manifest)

  if skipped:
    print('Skipping raw domains without safe train-class manifest: {}'.format(
      ', '.join(skipped)))
  domain_names = [
    domain for domain in raw_paths if domain in split_by_domain
  ]
  skipped_too_few = [
    domain for domain in domain_names
    if len(split_by_domain[domain]['train']) < 2
  ]
  domain_names = [d for d in domain_names if d not in set(skipped_too_few)]
  if len(domain_names) < 2 and not args.validate_only:
    raise RuntimeError('at least two manifest-covered raw domains are required')

  print('raw data root: {}'.format(data_root))
  print('manifest root: {}'.format(manifest_root))
  print('device: {}'.format(device))
  print('detected raw domains: {} ({})'.format(
    len(raw_paths), ', '.join(raw_paths)))
  print('manifest-covered train-only domains: {} ({})'.format(
    len(domain_names), ', '.join(domain_names)))

  # Load only metadata/paths and pre-compute deterministic repeat sampling.
  domain_data = {}
  all_selected_paths = []
  for domain in domain_names:
    image_paths, labels, raw_report = load_raw_meta_album_domain(
      raw_paths[domain])
    validation = validate_manifest_labels(
      domain, labels, split_by_domain[domain], manifest_sources[domain])
    plans = []
    for repeat in range(args.num_repeats):
      sampling_seed = derived_seed(args.seed, domain, repeat, 'sampling')
      head_seed = derived_seed(args.seed, domain, repeat, 'head')
      fisher_seed = derived_seed(args.seed, domain, repeat, 'fisher')
      indices, targets, sample_record = sample_train_classes(
        labels, split_by_domain[domain]['train'],
        args.classes_per_domain, args.images_per_class, sampling_seed)
      sample_record.update({
        'repeat': int(repeat),
        'head_seed': int(head_seed),
        'fisher_seed': int(fisher_seed),
      })
      plans.append((indices, targets, sample_record))
      all_selected_paths.extend([image_paths[int(i)] for i in indices])
    domain_data[domain] = {
      'image_paths': image_paths,
      'labels': labels,
      'raw_report': raw_report,
      'manifest_validation': validation,
      'plans': plans,
    }

  probe_image_size, image_size_report = resolve_probe_image_size(
    args.image_size, all_selected_paths)
  print('probe input size: {}x{} (no upscaling)'.format(
    probe_image_size, probe_image_size))
  print('selected raw image short-side range: {}..{}'.format(
    image_size_report['min_short_side'],
    image_size_report['max_short_side']))

  sampling_summary = {
    'protocol': {
      'input_source': 'raw Meta-Album labels.csv + images/',
      'pickle_input_used': False,
      'classes_per_domain': int(args.classes_per_domain),
      'images_per_class': int(args.images_per_class),
      'num_repeats': int(args.num_repeats),
      'fisher_num_samples_per_repeat': int(args.fisher_num_samples),
      'fisher_batch_size': int(args.fisher_batch_size),
      'base_seed': int(args.seed),
      'only_manifest_train_classes_used': True,
      'class_sampling_without_replacement': True,
      'fisher_sampling_with_replacement': True,
      'repeat_embeddings_averaged_arithmetically': True,
      'image_size_policy': image_size_report,
    },
    'detected_domains': list(raw_paths),
    'analyzed_domains': domain_names,
    'skipped_without_manifest': skipped,
    'skipped_with_fewer_than_two_train_classes': skipped_too_few,
    'domains': {},
  }

  for domain in domain_names:
    item = domain_data[domain]
    sampling_summary['domains'][domain] = {
      'raw_data': item['raw_report'],
      'manifest_sources': manifest_sources[domain],
      'manifest_source_sha256': {
        path: sha256_file(path) for path in manifest_sources[domain]},
      'manifest_validation': item['manifest_validation'],
      'repeats': [plan[2] for plan in item['plans']],
    }

  if args.validate_only:
    print(json.dumps(sampling_summary, indent=2, allow_nan=False))
    print(
      'Validation complete: raw images/manifests are valid; no model loaded '
      'and no outputs written.')
    return

  output_dir = prepare_output_dir(args.output_dir, args.overwrite)
  probe, transform, feature_dim, probe_report = make_probe(
    device, probe_image_size)
  layers = fisher_layers(probe, args.include_batchnorm)
  layout, component_names = embedding_layout(layers)
  initial_probe_hash = probe_state_sha256(probe)
  probe_report.update({
    'state_sha256_before_analysis': initial_probe_hash,
    'embedding_dimension': len(component_names),
    'fisher_modules': len(layers),
    'include_batchnorm_scales': bool(args.include_batchnorm),
    'classifier_excluded_from_embedding': True,
    'image_size_resolution_report': image_size_report,
  })
  print('probe: {} / {}'.format(
    probe_report['architecture'], probe_report['weights']))
  print('Task2Vec embedding dimension: {}'.format(len(component_names)))

  repeat_embeddings_by_domain = []
  repeat_seeds_by_domain = []
  for domain_index, domain in enumerate(domain_names):
    item = domain_data[domain]
    image_paths = item['image_paths']
    repeat_embeddings = []
    repeat_seeds = []

    for repeat, (indices, targets, sample_record) in enumerate(item['plans']):
      print('[{}/{}] {} repeat {}/{}: {} classes, {} raw images'.format(
        domain_index + 1, len(domain_names), domain,
        repeat + 1, args.num_repeats,
        sample_record['sampled_classes'],
        sample_record['sampled_images_total']))

      dataset = RawMetaAlbumSample(
        image_paths, indices, targets, transform)
      image_loader = make_image_loader(
        dataset, args.batch_size, args.num_workers, device)
      features = cache_probe_features(probe, image_loader, device)
      head, head_report = fit_linear_head(
        features, targets, sample_record['sampled_classes'], feature_dim,
        device, args, sample_record['head_seed'])
      fisher_loader, fisher_sampler_report = make_fisher_loader(
        dataset, targets, args.fisher_batch_size, args.num_workers, device,
        args.fisher_num_samples, sample_record['fisher_seed'])
      embedding, fisher_report = monte_carlo_filter_fisher(
        probe, head, fisher_loader, layers, device,
        sample_record['fisher_seed'], sampler_report=fisher_sampler_report)

      sample_record['head_fit'] = head_report
      sample_record['fisher'] = fisher_report
      sample_record['embedding_l1_norm'] = float(np.linalg.norm(
        embedding, ord=1))
      sample_record['embedding_l2_norm'] = float(np.linalg.norm(embedding))
      repeat_embeddings.append(embedding)
      repeat_seeds.append([
        sample_record['sampling_seed'], sample_record['head_seed'],
        sample_record['fisher_seed']])
      del dataset, image_loader, fisher_loader, features, head
      if device.type == 'cuda':
        torch.cuda.empty_cache()

    repeat_embeddings_by_domain.append(np.stack(repeat_embeddings, axis=0))
    repeat_seeds_by_domain.append(np.asarray(repeat_seeds, dtype=np.uint64))

  final_probe_hash = probe_state_sha256(probe)
  if final_probe_hash != initial_probe_hash:
    raise RuntimeError('fixed probe parameters changed during analysis')
  probe_report['state_sha256_after_analysis'] = final_probe_hash
  probe_report['fixed_probe_hash_verified_unchanged'] = True

  repeat_embeddings = np.stack(repeat_embeddings_by_domain, axis=0)
  embeddings = repeat_embeddings.mean(axis=1, dtype=np.float64).astype(
    np.float32)
  repeat_seeds = np.stack(repeat_seeds_by_domain, axis=0)
  task2vec_matrix = pairwise_distances(embeddings, task2vec_distance)
  cosine_matrix = pairwise_distances(embeddings, cosine_distance)

  np.savez_compressed(
    os.path.join(output_dir, 'task2vec_embeddings.npz'),
    embeddings=embeddings,
    repeat_embeddings=repeat_embeddings,
    domain_names=np.asarray(domain_names, dtype=str),
    component_names=np.asarray(component_names, dtype=str),
    embedding_layout_json=np.asarray(json.dumps(layout), dtype=str),
    repeat_seeds=repeat_seeds)
  write_json(os.path.join(output_dir, 'domain_names.json'), domain_names)
  write_distance_csv(
    os.path.join(output_dir, 'task2vec_distance_matrix.csv'),
    domain_names, task2vec_matrix)
  write_distance_csv(
    os.path.join(output_dir, 'cosine_distance_matrix.csv'),
    domain_names, cosine_matrix)

  linkage_matrix, clustering = clustering_analysis(
    domain_names, task2vec_matrix)
  plot_dendrogram(
    os.path.join(output_dir, 'dendrogram.png'),
    domain_names, linkage_matrix)
  coordinates, explained = pca_projection(embeddings)
  plot_projection(
    os.path.join(output_dir, 'pca_domains.png'),
    domain_names, coordinates, explained)
  clustering['pca'] = {
    'coordinates': {
      domain: [float(value) for value in row]
      for domain, row in zip(domain_names, coordinates)},
    'explained_variance_ratio': [float(x) for x in explained],
    'input': 'arithmetic-mean Task2Vec embeddings',
  }
  clustering['umap'] = maybe_plot_umap(
    output_dir, domain_names, embeddings, args.seed, args.umap)
  old_comparison = old_cluster_analysis(domain_names, task2vec_matrix)
  nearest = nearest_neighbors(
    domain_names, task2vec_matrix,
    OLD_CLUSTERS['A'] + OLD_CLUSTERS['B'])
  closest, distant = ranked_pairs(
    domain_names, task2vec_matrix, args.top_pairs)
  clustering['previous_conv4_clusters'] = old_comparison
  clustering['nearest_neighbors_previous_cluster_domains'] = nearest
  clustering['closest_domain_pairs'] = closest
  clustering['most_distant_domain_pairs'] = distant
  write_json(
    os.path.join(output_dir, 'clustering_results.json'), clustering)
  write_json(
    os.path.join(output_dir, 'sampling_summary.json'), sampling_summary)

  run_config = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'project_root': PROJECT_ROOT,
    'git_commit': git_commit(),
    'raw_data_root': data_root,
    'manifest_root': manifest_root,
    'output_dir': output_dir,
    'arguments': vars(args),
    'environment': environment_report(device),
    'probe': probe_report,
    'manifests': {
      'explicit_paths': (
        [absolute_path(path) for path in args.class_split_manifest]
        if args.class_split_manifest else []),
      'auto_discovery_candidates': [
        absolute_path(path) for path in manifest_candidates],
      'sources_by_domain': manifest_sources,
      'domains_without_manifest_skipped': skipped,
    },
    'methodology': {
      'name': 'Task2Vec-style Monte Carlo diagonal Fisher',
      'probe_independent_from_maml': True,
      'maml_checkpoint_used': False,
      'pickle_input_used': False,
      'raw_input_format': 'Meta-Album labels.csv + images/',
      'feature_extractor_updated': False,
      'domain_specific_component': 'fresh linear classification head only',
      'head_excluded_from_embedding': True,
      'input_upscaling': False,
      'primary_distance': (
        'cosine(Fa/(Fa+Fb), Fb/(Fa+Fb)); symmetric Task2Vec distance'),
      'secondary_distance': 'ordinary cosine distance',
      'references': [
        'https://openaccess.thecvf.com/content_ICCV_2019/html/'
        'Achille_Task2Vec_Task_Embedding_for_Meta-Learning_ICCV_2019_paper.html',
        'https://github.com/awslabs/aws-cv-task2vec',
      ],
    },
    'outputs': sorted(set(os.listdir(output_dir)) | {'run_config.json'}),
  }
  write_json(os.path.join(output_dir, 'run_config.json'), run_config)
  print_summary(
    closest, distant, clustering, old_comparison, nearest, args.top_pairs)
  print('\nOutputs written to: {}'.format(output_dir))


def parse_args(argv=None):
  parser = argparse.ArgumentParser(
    description=(
      'Compute train-only Task2Vec geometry from raw Meta-Album images; '
      '84x84 AG-Meta pickles are never used.'))
  parser.add_argument(
    '--data-root', default=DEFAULT_DATA_ROOT,
    help='raw Meta-Album root containing domain folders with labels.csv + images/')
  parser.add_argument(
    '--manifest-root', default=DEFAULT_MANIFEST_ROOT,
    help='directory containing AG-Meta train/val/test class-split manifests')
  parser.add_argument(
    '--class-split-manifest', action='append', default=None,
    help='explicit split manifest; may be repeated')
  parser.add_argument(
    '--domains', nargs='+', default=None,
    help='optional domain subset; strongly recommended with explicit manifest')
  parser.add_argument('--classes-per-domain', type=int, default=10)
  parser.add_argument('--images-per-class', type=int, default=20)
  parser.add_argument('--num-repeats', type=int, default=5)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--device', default='auto')
  parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
  parser.add_argument('--batch-size', type=int, default=32)
  parser.add_argument('--fisher-batch-size', type=int, default=64)
  parser.add_argument('--num-workers', type=int, default=2)
  parser.add_argument('--head-epochs', type=int, default=10)
  parser.add_argument('--head-learning-rate', type=float, default=4e-4)
  parser.add_argument('--head-weight-decay', type=float, default=1e-4)
  parser.add_argument('--fisher-num-samples', type=int, default=10000)
  parser.add_argument(
    '--image-size', type=int, default=0,
    help=(
      '0 = automatically choose min(224, minimum selected short side), '
      'so no raw image is upscaled; positive values are allowed only if '
      'every selected image is at least that large'))
  parser.add_argument(
    '--include-batchnorm', action=argparse.BooleanOptionalAction,
    default=True)
  parser.add_argument(
    '--umap', action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument('--top-pairs', type=int, default=5)
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--validate-only', action='store_true')
  return parser.parse_args(argv)


if __name__ == '__main__':
  run(parse_args())
