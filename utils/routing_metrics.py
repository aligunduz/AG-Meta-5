from threading import Lock

import numpy as np


class AnchorRoutingMeter:
  """Accumulates router outputs without affecting model gradients."""

  def __init__(self, n_anchors):
    self.n_anchors = n_anchors
    self.lock = Lock()
    self.reset()

  def reset(self):
    with self.lock:
      self.phase = 'train'
      self.stats = {}

  def set_phase(self, phase):
    with self.lock:
      self.phase = phase

  def hook(self, module, inputs, output):
    weights = output['weights'].detach().double().cpu().numpy()
    indices = output['indices'].detach().cpu().numpy()

    nearest = np.bincount(
      indices[:, 0], minlength=self.n_anchors)
    selected = np.bincount(
      indices.reshape(-1), minlength=self.n_anchors)

    with self.lock:
      if self.phase not in self.stats:
        self.stats[self.phase] = {
          'tasks': 0,
          'nearest': np.zeros(self.n_anchors, dtype=np.int64),
          'selected': np.zeros(self.n_anchors, dtype=np.int64),
          'weight_sum': np.zeros(self.n_anchors),
          'effective_sum': 0.,
        }

      stats = self.stats[self.phase]
      stats['tasks'] += len(weights)
      stats['nearest'] += nearest
      stats['selected'] += selected
      stats['weight_sum'] += weights.sum(axis=0)
      stats['effective_sum'] += (
        1. / np.square(weights).sum(axis=1)
      ).sum()

  def metrics(self):
    result = {}
    with self.lock:
      for phase, stats in self.stats.items():
        n = stats['tasks']
        if n == 0:
          continue

        prefix = f'{phase}/routing'
        result[f'{prefix}/tasks'] = int(n)
        result[f'{prefix}/effective_anchors'] = float(
          stats['effective_sum'] / n)

        for k in range(self.n_anchors):
          anchor = f'{prefix}/anchor_{k}'
          result[f'{anchor}/nearest_count'] = int(stats['nearest'][k])
          result[f'{anchor}/selected_count'] = int(stats['selected'][k])
          result[f'{anchor}/selected_fraction'] = float(
            stats['selected'][k] / n)
          result[f'{anchor}/weight_share'] = float(
            stats['weight_sum'][k] / n)

    return result