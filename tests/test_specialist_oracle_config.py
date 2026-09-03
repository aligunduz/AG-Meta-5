import argparse
import csv
import json
import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn
import yaml

from tools import analyze_specialist_oracle as analysis


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_CONFIG = os.path.join(
  PROJECT_ROOT, 'configs', 'analysis',
  'specialist_oracle_meta_album_5way_1shot.yaml')
RESNET_CONFIG = os.path.join(
  PROJECT_ROOT, 'configs', 'analysis',
  'specialist_oracle_resnet18_fomaml_ab_5way_1shot.yaml')
PAIRED_CONFIG = os.path.join(
  PROJECT_ROOT, 'configs', 'analysis',
  'paired_checkpoint_meta_album_5way_1shot.yaml')


def load_config(path):
  with open(path, 'r', encoding='utf-8') as f:
    return analysis.normalized_config(
      yaml.load(f, Loader=yaml.FullLoader))


class DummyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.weight = nn.Parameter(torch.zeros(1))

  def go_efficient(self):
    return None


class RecordingModel(nn.Module):
  def __init__(self, n_way):
    super().__init__()
    self.weight = nn.Parameter(torch.zeros(1))
    self.n_way = n_way
    self.batch_signatures = []

  def forward(
      self, x_shot, x_query, y_shot, inner_args, meta_train,
      use_gradient_transport):
    del inner_args, meta_train, use_gradient_transport
    self.batch_signatures.append((
      x_shot.data_ptr(), x_query.data_ptr(), y_shot.data_ptr()))
    return torch.zeros(
      x_query.shape[0], x_query.shape[1], self.n_way,
      device=x_query.device) + self.weight


class MetadataValidator(object):
  def validate_episode_metadata(self, metadata):
    return bool(metadata)


class OneBatchLoader(object):
  def __init__(self, batch):
    self.batch = batch
    self.dataset = MetadataValidator()

  def __iter__(self):
    yield self.batch


class SpecialistOracleConfigTests(unittest.TestCase):

  def test_paired_template_does_not_require_oracle_roles_or_query_split(self):
    config = load_config(PAIRED_CONFIG)
    analysis.validate_config(config)

    self.assertEqual(config['mode'], 'paired_checkpoints')
    self.assertEqual(config['domain'], 'TEX')
    self.assertEqual(config['n_episodes'], 500)
    self.assertTrue(config['test']['allow_unassigned_classes'])
    self.assertNotIn('model_roles', config)
    self.assertNotIn('query_split', config)
    self.assertNotIn('bootstrap', config)
    self.assertEqual(analysis.paired_model_names(config), ('C', 'mixed3'))
    self.assertEqual(analysis.paired_csv_fields(config), [
      'task_id', 'domain', 'class_combination', 'class_sample_mapping',
      'episode_sample_sha256',
      'c_query_loss', 'c_query_accuracy',
      'mixed3_query_loss', 'mixed3_query_accuracy',
      'query_loss_difference', 'query_accuracy_difference',
    ])

  def test_paired_rows_record_sorted_classes_mapping_and_differences(self):
    config = load_config(PAIRED_CONFIG)
    metadata = analysis.canonical_episode_metadata(3, 'TEX', {
      'class_names': ('zebra', 'ant'),
      'support_pickle_indices': ((8,), (2,)),
      'query_pickle_indices': ((9, 10), (3, 4)),
    })
    results = {
      'C': {'loss': [0.2], 'accuracy': [0.8]},
      'mixed3': {'loss': [0.3], 'accuracy': [0.75]},
    }
    row = analysis.make_paired_rows([metadata], results, config)[0]

    self.assertEqual(json.loads(row['class_combination']), ['ant', 'zebra'])
    mapping = json.loads(row['class_sample_mapping'])
    self.assertEqual(mapping['zebra']['support_pickle_indices'], [8])
    self.assertEqual(mapping['ant']['query_pickle_indices'], [3, 4])
    self.assertAlmostEqual(row['query_loss_difference'], -0.1)
    self.assertAlmostEqual(row['query_accuracy_difference'], 0.05)

  def test_paired_evaluation_passes_same_batch_to_both_models(self):
    config = load_config(PAIRED_CONFIG)
    config['domain'] = 'TEX'
    config['test']['n_way'] = 2
    config['test']['n_shot'] = 1
    config['test']['n_query'] = 2
    config['inner_args']['n_step'] = 0
    x_shot = torch.randn(1, 2, 3, 2, 2)
    x_query = torch.randn(1, 4, 3, 2, 2)
    y_shot = torch.tensor([[0, 1]])
    y_query = torch.tensor([[0, 0, 1, 1]])
    metadata = {
      'domain': 'TEX',
      'class_names': ('a', 'b'),
      'support_pickle_indices': ((1,), (10,)),
      'query_pickle_indices': ((2, 3), (11, 12)),
    }
    loader = OneBatchLoader((
      [0], ['TEX'], (x_shot, x_query, y_shot, y_query), [metadata]))
    models_by_name = {}
    records = {}
    for name in analysis.paired_model_names(config):
      model = RecordingModel(n_way=2)
      models_by_name[name] = model
      records[name] = {
        'name': name,
        'model': model,
        'use_gradient_transport': False,
        'precision': {
          'use_amp': False,
          'amp_dtype': torch.float16,
          'allow_tf32': False,
        },
      }

    results, ids, sample_ids, metadata_records = \
      analysis.evaluate_paired_tasks(
        config, records, loader, torch.device('cpu'), keep_on_gpu=False)

    first_signature = models_by_name['C'].batch_signatures[0]
    second_signature = models_by_name['mixed3'].batch_signatures[0]
    self.assertEqual(first_signature, second_signature)
    self.assertEqual(ids['C'], ids['mixed3'])
    self.assertEqual(sample_ids['C'], sample_ids['mixed3'])
    self.assertEqual(len(metadata_records), 1)
    self.assertEqual(len(results['C']['loss']), 1)

  def test_legacy_conv4_config_and_csv_schema_remain_compatible(self):
    config = load_config(LEGACY_CONFIG)
    analysis.validate_config(config)

    self.assertEqual(config['model_roles'], {
      'specialists': {
        'NATURAL': 'NATURAL',
        'TECHNICAL': 'TECHNICAL',
      },
      'reference': 'UNION',
    })
    self.assertEqual(config['test']['image_size'], 84)
    self.assertEqual(config['inner_args']['n_step'], 10)
    self.assertEqual(analysis.standard_csv_fields(config), [
      'task_id', 'cluster', 'domain',
      'natural_query_loss', 'natural_query_accuracy',
      'technical_query_loss', 'technical_query_accuracy',
      'union_query_loss', 'union_query_accuracy',
      'loss_selected_specialist', 'specialist_loss_tie',
      'domain_specialist',
      'domain_router_query_loss', 'domain_router_query_accuracy',
      'oracle_query_loss', 'oracle_query_accuracy',
    ])
    self.assertIn(
      'cross_fitted_union_query_loss',
      analysis.query_split_csv_fields(config))
    self.assertIn(
      'union_split_oracle_query_loss',
      analysis.query_split_csv_fields(config))

  def test_legacy_summary_json_keys_remain_compatible(self):
    config = load_config(LEGACY_CONFIG)
    specs = [
      {'task_id': 0, 'cluster': 'NATURAL', 'domain': 'BRD'},
      {'task_id': 1, 'cluster': 'TECHNICAL', 'domain': 'RESISC'},
    ]
    results = {
      'NATURAL': {'loss': [0.2, 0.7], 'accuracy': [0.9, 0.4]},
      'TECHNICAL': {'loss': [0.6, 0.1], 'accuracy': [0.5, 1.0]},
      'UNION': {'loss': [0.4, 0.4], 'accuracy': [0.7, 0.7]},
    }
    rows = analysis.make_standard_rows(specs, results, config=config)
    checkpoint_report = {
      'models': {name: {'load': name} for name in results},
    }
    bootstrap = {'n_resamples': 8, 'confidence': 0.95, 'seed': 3}
    summary = analysis.compute_standard_summary(
      config, rows, results, {}, checkpoint_report, {}, {}, bootstrap)

    self.assertIs(
      summary['oracle_vs_union'], summary['oracle_vs_reference'])
    self.assertIs(
      summary['domain_router_vs_union'],
      summary['domain_router_vs_reference'])

    split_results = {
      name: {
        'A': {'loss': values['loss'], 'accuracy': values['accuracy']},
        'B': {
          'loss': list(reversed(values['loss'])),
          'accuracy': list(reversed(values['accuracy'])),
        },
      }
      for name, values in results.items()
    }
    split_rows = analysis.make_query_split_rows(
      specs, split_results, config=config)
    split_summary = analysis.compute_query_split_summary(
      config, split_rows, split_results, {}, checkpoint_report, {}, {},
      bootstrap, summary)
    self.assertIs(
      split_summary['union_query_split_noise_floor'],
      split_summary['reference_query_split_noise_floor'])
    self.assertIn('vs_union', split_summary['cross_fitted_oracle'])

  def test_resnet_fomaml_config_has_requested_protocol(self):
    config = load_config(RESNET_CONFIG)
    analysis.validate_config(config)

    self.assertEqual(config['encoder'], 'resnet18')
    self.assertEqual(
      config['class_split_manifest'],
      './materials/album_128/geometry_cluster_ab_class_split.json')
    self.assertEqual(
      config['test']['class_split_manifest'],
      './materials/album_128/geometry_cluster_ab_class_split.json')
    self.assertEqual(config['clusters'], {
      'A': ['ACT_40', 'ACT_410', 'SPT'],
      'B': ['RESISC', 'RSICB', 'RSD'],
    })
    self.assertEqual(config['model_roles'], {
      'specialists': {'A': 'A', 'B': 'B'},
      'reference': 'AB',
    })
    self.assertEqual(config['models']['A']['load'],
      './save/FOMAML_resnet18_geometry_cluster_a/max-va.pth')
    self.assertEqual(config['models']['B']['load'],
      './save/FOMAML_resnet18_geometry_cluster_b/max-va.pth')
    self.assertEqual(config['models']['AB']['load'],
      './save/FOMAML_resnet18_geometry_cluster_ab/max-va.pth')
    self.assertEqual(config['test']['image_size'], 128)
    self.assertEqual(config['inner_args']['n_step'], 5)
    self.assertTrue(config['inner_args']['first_order'])
    self.assertEqual(config['query_split']['n_query'], 30)
    self.assertEqual(config['query_split']['split_query'], 15)
    self.assertIn('ab_query_loss', analysis.standard_csv_fields(config))
    self.assertIn(
      'cross_fitted_ab_query_loss',
      analysis.query_split_csv_fields(config))

  def test_cluster_and_domain_names_are_config_driven(self):
    config = load_config(RESNET_CONFIG)
    config['clusters'] = {
      'VISION': ['CUSTOM_ONE', 'CUSTOM_TWO'],
      'MIXED': ['CUSTOM_THREE'],
    }
    config['models'] = {
      'left-model': config['models']['A'],
      'right-model': config['models']['B'],
      'shared-model': config['models']['AB'],
    }
    config['model_roles'] = {
      'specialists': {
        'VISION': 'left-model',
        'MIXED': 'right-model',
      },
      'reference': 'shared-model',
    }
    analysis.validate_config(config)

    specs = analysis.balanced_task_specs(config, seed=0)
    self.assertEqual(
      {spec['domain'] for spec in specs},
      {'CUSTOM_ONE', 'CUSTOM_TWO', 'CUSTOM_THREE'})
    self.assertEqual(
      set(analysis.task_balance_report(specs)['cluster_counts']),
      {'VISION', 'MIXED'})

  def test_dynamic_rows_use_a_b_ab_names_and_roles(self):
    config = load_config(RESNET_CONFIG)
    specs = [
      {'task_id': 0, 'cluster': 'A', 'domain': 'ACT_40'},
      {'task_id': 1, 'cluster': 'B', 'domain': 'BRD'},
    ]
    results = {
      'A': {'loss': [0.2, 0.8], 'accuracy': [0.9, 0.3]},
      'B': {'loss': [0.5, 0.1], 'accuracy': [0.6, 1.0]},
      'AB': {'loss': [0.4, 0.4], 'accuracy': [0.7, 0.7]},
    }
    rows = analysis.make_standard_rows(specs, results, config=config)

    self.assertEqual(rows[0]['loss_selected_specialist'], 'A')
    self.assertEqual(rows[1]['loss_selected_specialist'], 'B')
    self.assertEqual(rows[0]['domain_specialist'], 'A')
    self.assertEqual(rows[1]['domain_specialist'], 'B')
    self.assertEqual(rows[0]['a_query_loss'], 0.2)
    self.assertEqual(rows[1]['b_query_loss'], 0.1)
    self.assertEqual(rows[0]['ab_query_loss'], 0.4)

    split_results = {
      name: {
        'A': {
          'loss': values['loss'],
          'accuracy': values['accuracy'],
        },
        'B': {
          'loss': list(reversed(values['loss'])),
          'accuracy': list(reversed(values['accuracy'])),
        },
      }
      for name, values in results.items()
    }
    split_rows = analysis.make_query_split_rows(
      specs, split_results, config=config)
    self.assertIn('a_a_query_loss', split_rows[0])
    self.assertIn('b_b_query_loss', split_rows[0])
    self.assertIn('ab_a_query_loss', split_rows[0])
    self.assertIn('cross_fitted_ab_query_loss', split_rows[0])

  def test_checkpoint_validation_accepts_resnet18(self):
    config = load_config(RESNET_CONFIG)
    with tempfile.TemporaryDirectory() as directory:
      checkpoints = {}
      for index, name in enumerate(analysis.analysis_model_names(config)):
        path = os.path.join(directory, '{}.pth'.format(name))
        with open(path, 'wb') as f:
          f.write(name.encode('utf-8'))
        config['models'][name]['load'] = path
        checkpoints[path] = {
          'encoder': 'resnet18',
          'classifier': 'logistic',
          'classifier_args': {'n_way': 5},
          'encoder_state_dict': {
            'weight': torch.tensor([float(index)]),
          },
          'classifier_state_dict': {
            'weight': torch.tensor([float(index + 10)]),
          },
        }

      args = argparse.Namespace(
        efficient=False, keep_models_on_gpu=False)
      with mock.patch.object(
          analysis, 'torch_load_checkpoint',
          side_effect=lambda path: checkpoints[path]), mock.patch.object(
            analysis.models, 'load', side_effect=lambda *args, **kwargs: DummyModel()):
        records, keep_on_gpu, report = analysis.load_analysis_models(
          config, args, torch.device('cpu'))

    self.assertEqual(set(records), {'A', 'B', 'AB'})
    self.assertFalse(keep_on_gpu)
    self.assertEqual(report['encoder'], 'resnet18')
    self.assertTrue(report['encoder_verified_as_configured'])
    self.assertFalse(report['encoder_verified_as_convnet4'])

  def test_task_csv_and_summary_json_writers_remain_available(self):
    config = load_config(RESNET_CONFIG)
    row = {field: 0 for field in analysis.standard_csv_fields(config)}
    row.update({'task_id': 7, 'cluster': 'A', 'domain': 'ACT_40'})
    summary = {
      'mode': 'specialist_oracle',
      'reference_model': 'AB',
      'n_tasks': 1,
    }
    with tempfile.TemporaryDirectory() as directory:
      csv_path = os.path.join(directory, 'specialist_oracle_tasks.csv')
      json_path = os.path.join(directory, 'specialist_oracle_summary.json')
      analysis.write_csv(
        csv_path, [row], analysis.standard_csv_fields(config))
      analysis.write_json(json_path, summary)
      with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        written_rows = list(csv.DictReader(f))
      with open(json_path, 'r', encoding='utf-8') as f:
        written_summary = json.load(f)

    self.assertEqual(written_rows[0]['domain'], 'ACT_40')
    self.assertEqual(written_summary['reference_model'], 'AB')


if __name__ == '__main__':
  unittest.main()
