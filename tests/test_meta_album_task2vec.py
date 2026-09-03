import json
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.analyze_meta_album_task2vec import (
  cosine_distance,
  derived_seed,
  discover_domain_manifests,
  discover_raw_domains,
  load_raw_meta_album_domain,
  make_balanced_fisher_sampler,
  old_cluster_analysis,
  resolve_probe_image_size,
  sample_train_classes,
  task2vec_distance,
)


class Task2VecAnalysisTests(unittest.TestCase):

  def test_derived_seeds_are_stable_and_purpose_specific(self):
    first = derived_seed(7, 'BRD', 2, 'sampling')
    self.assertEqual(first, derived_seed(7, 'BRD', 2, 'sampling'))
    self.assertNotEqual(first, derived_seed(7, 'BRD', 2, 'fisher'))
    self.assertNotEqual(first, derived_seed(7, 'DOG', 2, 'sampling'))

  def test_balanced_sampler_uses_only_requested_train_classes(self):
    labels = np.asarray(
      ['train_a'] * 2 + ['train_b'] * 4 + ['test'] * 20, dtype=str)
    indices, targets, report = sample_train_classes(
      labels, ['train_a', 'train_b'],
      classes_per_domain=3, images_per_class=3, seed=11)

    sampled_labels = set(labels[indices].tolist())
    self.assertEqual(sampled_labels, {'train_a', 'train_b'})
    self.assertNotIn('test', sampled_labels)
    self.assertEqual(len(indices), 5)
    self.assertEqual(set(targets.tolist()), {0, 1})
    self.assertTrue(report['used_fewer_classes_than_requested'])

  def test_fisher_sampler_is_deterministic_and_fixed_size(self):
    targets = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int64)
    sampler_a, report_a = make_balanced_fisher_sampler(
      targets, num_samples=1000, seed=123)
    sampler_b, report_b = make_balanced_fisher_sampler(
      targets, num_samples=1000, seed=123)

    draw_a = list(iter(sampler_a))
    draw_b = list(iter(sampler_b))
    self.assertEqual(draw_a, draw_b)
    self.assertEqual(len(draw_a), 1000)
    self.assertEqual(report_a, report_b)

    sampled_targets = targets[np.asarray(draw_a, dtype=np.int64)]
    counts = np.bincount(sampled_targets, minlength=2)
    self.assertLess(abs(int(counts[0]) - int(counts[1])), 120)

  def test_task2vec_distance_is_symmetric_and_identity_is_zero(self):
    left = np.asarray([0.2, 1.0, 0.0, 2.0])
    right = np.asarray([1.0, 0.5, 0.0, 3.0])
    self.assertAlmostEqual(task2vec_distance(left, left), 0.0)
    self.assertAlmostEqual(
      task2vec_distance(left, right), task2vec_distance(right, left))
    self.assertGreaterEqual(task2vec_distance(left, right), 0.0)
    self.assertAlmostEqual(cosine_distance(left, left), 0.0)

  def test_manifest_auto_discovery_rejects_conflicts(self):
    first = {'BRD': {'train': ['a'], 'val': ['b'], 'test': ['c']}}
    second = {'BRD': {'train': ['b'], 'val': ['a'], 'test': ['c']}}
    with tempfile.TemporaryDirectory() as directory:
      with open(os.path.join(directory, 'one.json'), 'w', encoding='utf-8') as f:
        json.dump(first, f)
      with open(os.path.join(directory, 'two.json'), 'w', encoding='utf-8') as f:
        json.dump(second, f)
      with self.assertRaisesRegex(ValueError, 'conflicting class split'):
        discover_domain_manifests(directory, ['BRD'])

  def test_old_cluster_separation_reports_pooled_and_balanced(self):
    domains = ['BRD', 'DOG', 'AWA', 'MD_MIX', 'PLK']
    matrix = np.asarray([
      [0, 1, 1, 4, 4],
      [1, 0, 1, 4, 4],
      [1, 1, 0, 4, 4],
      [4, 4, 4, 0, 3],
      [4, 4, 4, 3, 0],
    ], dtype=float)
    report = old_cluster_analysis(domains, matrix)

    self.assertEqual(report['within_A_average_distance'], 1.0)
    self.assertEqual(report['within_B_average_distance'], 3.0)
    self.assertEqual(report['A_to_B_average_distance'], 4.0)

    # pooled within = (1 + 1 + 1 + 3) / 4 = 1.5
    self.assertAlmostEqual(report['within_cluster_distance'], 1.5)
    self.assertAlmostEqual(report['separation_ratio'], 4.0 / 1.5)

    # balanced gives A and B equal cluster weight: (1 + 3) / 2 = 2
    self.assertAlmostEqual(report['balanced_within_cluster_distance'], 2.0)
    self.assertAlmostEqual(report['balanced_separation_ratio'], 2.0)

  def test_raw_domain_loader_reads_labels_csv_and_images(self):
    with tempfile.TemporaryDirectory() as directory:
      domain_dir = os.path.join(directory, 'BRD')
      images_dir = os.path.join(domain_dir, 'images')
      os.makedirs(images_dir)

      rows = [
        ('a.jpg', 'class_a'),
        ('b.jpg', 'class_b'),
      ]
      for filename, _ in rows:
        Image.new('RGB', (128, 128)).save(os.path.join(images_dir, filename))

      with open(os.path.join(domain_dir, 'labels.csv'), 'w',
                encoding='utf-8', newline='') as f:
        f.write('FILE_NAME,CATEGORY,SUPER_CATEGORY\n')
        for filename, category in rows:
          f.write('{},{},NAN\n'.format(filename, category))

      with open(os.path.join(domain_dir, 'info.json'), 'w',
                encoding='utf-8') as f:
        json.dump({
          'dataset_name': 'BRD',
          'image_column_name': 'FILE_NAME',
          'category_column_name': 'CATEGORY',
        }, f)

      discovered = discover_raw_domains(directory, ['BRD'])
      self.assertEqual(discovered['BRD'], os.path.abspath(domain_dir))

      paths, labels, report = load_raw_meta_album_domain(domain_dir)
      self.assertEqual(len(paths), 2)
      self.assertEqual(labels.tolist(), ['class_a', 'class_b'])
      self.assertEqual(report['raw_image_count'], 2)

  def test_auto_image_size_never_upscales(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = []
      for idx, size in enumerate([(128, 128), (160, 140), (256, 256)]):
        path = os.path.join(directory, '{}.jpg'.format(idx))
        Image.new('RGB', size).save(path)
        paths.append(path)

      resolved, report = resolve_probe_image_size(0, paths)
      self.assertEqual(resolved, 128)
      self.assertFalse(report['upscaling_allowed'])

      with self.assertRaisesRegex(ValueError, 'would upscale'):
        resolve_probe_image_size(224, paths)


if __name__ == '__main__':
  unittest.main()
