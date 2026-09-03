import os
import shutil
import time
import torch

import numpy as np
import scipy.stats as stats


_log_path = None

def set_log_path(path):
  global _log_path
  _log_path = path


def log(obj, filename='log.txt'):
  print(obj)
  if _log_path is not None:
    with open(os.path.join(_log_path, filename), 'a') as f:
      print(obj, file=f)


class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0.
    self.avg = 0.
    self.sum = 0.
    self.count = 0.

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

  def item(self):
    return self.avg


class Timer(object):
  def __init__(self):
    self.start()

  def start(self):
    self.v = time.time()

  def end(self):
    return time.time() - self.v


def set_gpu(gpu: str):
    """Safely set GPU device if available, otherwise stay on CPU."""
    if not torch.cuda.is_available() or gpu == '-1':
        print("⚠️ CUDA not available or GPU disabled, running on CPU.")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        return

    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        torch.cuda.set_device(0)
        print(f"✅ Using CUDA device(s): {gpu}")
    except Exception as e:
        print(f"⚠️ Could not set CUDA device, fallback to CPU. ({e})")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''


def ensure_path(path, remove=True):
  basename = os.path.basename(path.rstrip('/'))
  if os.path.exists(path):
    if remove and (basename.startswith('_')
      or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
      shutil.rmtree(path)
      os.makedirs(path)
  else:
    os.makedirs(path)


def time_str(t):
  if t >= 3600:
    return '{:.1f}h'.format(t / 3600)
  if t >= 60:
    return '{:.1f}m'.format(t / 60)
  return '{:.1f}s'.format(t)


def compute_acc(pred, label, reduction='mean'):
  result = (pred == label).float()
  if reduction == 'none':
    return result.detach()
  elif reduction == 'mean':
    return result.mean().item()


def compute_n_params(model, return_str=True):
  n_params = 0
  for p in model.parameters():
    n_params += p.numel()
  if return_str:
    if n_params >= 1e6:
      return '{:.1f}M'.format(n_params / 1e6)
    else:
      return '{:.1f}K'.format(n_params / 1e3)
  else:
    return n_params


def mean_confidence_interval(data, confidence=0.95):
  a = 1.0 * np.array(data)
  stderr = stats.sem(a)
  h = stderr * stats.t.ppf((1 + confidence) / 2., len(a) - 1)
  return h


def _default(inner_args, key, default):
  # inner_args.get(key) or default would silently discard explicit falsy
  # values like 0 (e.g. encoder_lr: 0, n_step: 0); only fall back to the
  # default when the key is absent or explicitly null.
  value = inner_args.get(key)
  return default if value is None else value


def config_inner_args(inner_args):
  if inner_args is None:
    inner_args = dict()

  inner_args['reset_classifier'] = _default(inner_args, 'reset_classifier', False)
  inner_args['n_step'] = _default(inner_args, 'n_step', 5)
  inner_args['encoder_lr'] = _default(inner_args, 'encoder_lr', 0.01)
  inner_args['classifier_lr'] = _default(inner_args, 'classifier_lr', 0.01)
  inner_args['momentum'] = _default(inner_args, 'momentum', 0.)
  inner_args['weight_decay'] = _default(inner_args, 'weight_decay', 0.)
  inner_args['first_order'] = _default(inner_args, 'first_order', False)
  inner_args['frozen'] = _default(inner_args, 'frozen', [])

  return inner_args


def config_cuda_precision(config):
  use_amp = bool(config.get('use_amp', True))
  amp_dtype_name = str(config.get('amp_dtype', 'float16')).lower()
  amp_dtypes = {
    'float16': torch.float16,
    'fp16': torch.float16,
    'bfloat16': torch.bfloat16,
    'bf16': torch.bfloat16,
  }
  if amp_dtype_name not in amp_dtypes:
    raise ValueError(
      "amp_dtype must be one of: float16, fp16, bfloat16, bf16")

  amp_dtype = amp_dtypes[amp_dtype_name]
  amp_dtype_name = (
    'bfloat16' if amp_dtype == torch.bfloat16 else 'float16')
  if use_amp and amp_dtype == torch.bfloat16 \
      and not torch.cuda.is_bf16_supported():
    raise RuntimeError(
      'amp_dtype=bfloat16 was requested, but this CUDA device does not '
      'support bfloat16. Set use_amp: false for full FP32 training.')

  allow_tf32 = bool(config.get('allow_tf32', False))
  torch.backends.cuda.matmul.allow_tf32 = allow_tf32
  torch.backends.cudnn.allow_tf32 = allow_tf32
  torch.set_float32_matmul_precision('high' if allow_tf32 else 'highest')

  # FP16 needs dynamic loss scaling. BF16 has the same exponent range as
  # FP32, so scaling is unnecessary and can obscure whether Adam stepped.
  use_grad_scaler = use_amp and amp_dtype == torch.float16
  return use_amp, amp_dtype, amp_dtype_name, use_grad_scaler, allow_tf32
