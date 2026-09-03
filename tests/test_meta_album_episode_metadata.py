import json
import os
import pickle
import tempfile
import unittest

import numpy as np
import torch

from datasets.meta_album import MetaAlbumCrossDomain


class MetaAlbumEpisodeMetadataTests(unittest.TestCase):

  def make_domain(self, directory):
    labels = []
    images = []
    for class_index, class_name in enumerate(
        ('train_class', 'val_class', 'zebra', 'ant')):
      for image_index in range(4):
        labels.append(class_name)
        images.append(np.full(
          (6, 6, 3), class_index * 20 + image_index, dtype=np.uint8))
    with open(os.path.join(directory, 'TEX.pickle'), 'wb') as f:
      pickle.dump({'data': images, 'labels': labels}, f)
    manifest_path = os.path.join(directory, 'split.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
      json.dump({
        'TEX': {
          'train': ['train_class'],
          'val': ['val_class'],
          'test': ['zebra', 'ant'],
        },
      }, f)
    return labels, manifest_path

  def make_dataset(
      self, directory, manifest_path, return_metadata,
      allow_unassigned_classes=False):
    return MetaAlbumCrossDomain(
      root_path=directory,
      split='meta-test',
      domains=['TEX'],
      image_size=6,
      normalization=False,
      transform=None,
      n_batch=1,
      n_episode=1,
      n_way=2,
      n_shot=1,
      n_query=2,
      class_split_manifest=manifest_path,
      allow_unassigned_classes=allow_unassigned_classes,
      return_metadata=return_metadata)

  def test_metadata_uses_exact_episode_draw_and_pickle_indices(self):
    with tempfile.TemporaryDirectory() as directory:
      labels, manifest_path = self.make_domain(directory)
      legacy = self.make_dataset(directory, manifest_path, False)
      with_metadata = self.make_dataset(directory, manifest_path, True)

      np.random.seed(123)
      legacy_episode = legacy[0]
      np.random.seed(123)
      metadata_episode = with_metadata[0]

    self.assertEqual(len(legacy_episode), 4)
    self.assertEqual(len(metadata_episode), 5)
    for legacy_value, metadata_value in zip(
        legacy_episode, metadata_episode[:4]):
      self.assertTrue(torch.equal(legacy_value, metadata_value))

    metadata = metadata_episode[4]
    self.assertEqual(metadata['domain'], 'TEX')
    self.assertEqual(len(metadata['class_names']), 2)
    for class_name, support, query in zip(
        metadata['class_names'], metadata['support_pickle_indices'],
        metadata['query_pickle_indices']):
      self.assertEqual(len(support), 1)
      self.assertEqual(len(query), 2)
      self.assertEqual(len(set(support + query)), 3)
      for pickle_index in support + query:
        self.assertEqual(labels[pickle_index], class_name)

  def test_unassigned_class_option_coexists_with_episode_metadata(self):
    with tempfile.TemporaryDirectory() as directory:
      _, manifest_path = self.make_domain(directory)
      with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({
          'TEX': {
            'train': ['train_class'],
            'val': [],
            'test': ['zebra', 'ant'],
          },
        }, f)

      with self.assertRaisesRegex(ValueError, 'does not assign'):
        self.make_dataset(directory, manifest_path, return_metadata=False)

      dataset = self.make_dataset(
        directory, manifest_path, return_metadata=True,
        allow_unassigned_classes=True)
      episode = dataset[0]

    self.assertTrue(dataset.allow_unassigned_classes)
    self.assertEqual(len(episode), 5)
    self.assertEqual(set(episode[4]['class_names']), {'zebra', 'ant'})


if __name__ == '__main__':
  unittest.main()
