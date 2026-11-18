import copy
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from uemo.data import load_records, setting_key, validate_record
from uemo.metrics import summarize, report
from uemo.model import choice_loss
from uemo.calibrate import fit_temperature
from uemo.prepare_data import extract_archive
from uemo.train import packed_batches


def record(identifier='000000001'):
    return dict(id=identifier, source='Example', source_id='Example_'+identifier,
                modality=['text'], text='An example.', audio_path=[], image_path=[], video_path=[],
                instruction='Choose an emotion.', choices=['happy', 'sad'], gt_answer='happy')


class CoreTests(unittest.TestCase):
    def test_valid_choice_gradient_and_global_tail(self):
        torch.manual_seed(4)
        logits = torch.randn(3, 12, dtype=torch.float64, requires_grad=True)
        ids, counts, targets = [2, 4, 7], [2, 3, 2], [0, 2, 1]
        losses = choice_loss(logits, targets, ids, counts)
        independent = torch.stack([torch.nn.functional.cross_entropy(
            logits[i, ids[:count]][None], torch.tensor([targets[i]])) for i, count in enumerate(counts)])
        torch.testing.assert_close(losses, independent)
        grad = torch.autograd.grad(losses.mean(), logits, retain_graph=True)[0]
        # Four ranks with a three-sample tail: one rank has zero rows.
        summed = torch.zeros_like(logits)
        for rank in range(4):
            if rank < 3:
                summed += torch.autograd.grad(losses[rank]/3, logits, retain_graph=True)[0]
        torch.testing.assert_close(summed, grad)
        self.assertEqual(float(grad[0, 7]), 0.0)
        self.assertEqual(float(grad[:, [0, 1, 3, 5, 6, 8, 9, 10, 11]].abs().sum()), 0.0)

    def test_metrics_respect_shuffled_answer_contents(self):
        first = dict(record(), prediction='happy', probabilities=[.8, .2])
        second = dict(record('000000002'), choices=['sad', 'happy'], gt_answer='sad',
                      prediction='happy', probabilities=[.4, .6])
        result = summarize([first, second])
        self.assertAlmostEqual(result['accuracy'], .5)
        self.assertAlmostEqual(result['macro_f1'], 1/3)
        self.assertAlmostEqual(result['ece10'], .4)
        self.assertAlmostEqual(result['brier'], .4)

    def test_cremad_settings_are_distinct(self):
        audio = dict(record(), source='CREMA-D', subtask='CREMA-D', modality=['audio'],
                     prediction='happy', probabilities=[.8, .2])
        video = dict(audio, id='000000002', modality=['video', 'audio'])
        self.assertNotEqual(setting_key(audio), setting_key(video))
        self.assertEqual(report([audio, video])['setting_count'], 2)

    def test_context_schema_and_duplicate_ids(self):
        row = record()
        row['context'] = dict(modality=['text'], text='Earlier context.', audio_path=[], image_path=[], video_path=[])
        validate_record(row)
        bad = copy.deepcopy(row); bad['context']['modality'] = ['audio']
        with self.assertRaises(ValueError):
            validate_record(bad)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'metadata.json'; path.write_text(json.dumps([row, row]))
            with self.assertRaisesRegex(ValueError, 'Duplicate ID'):
                load_records(path)

    def test_packing_falls_back_without_token_changes(self):
        rows = [record(f'{i:09d}') for i in range(1, 4)]
        def prepared(processor, row, root):
            return {'text': row['id']}
        def batched(processor, items):
            # Simulate a processor whose batched media expansion changes tokens.
            return {'input_ids': torch.tensor([[int(x['text']), 7 if len(items) == 1 else 9] for x in items]),
                    'attention_mask': torch.ones(len(items), 2, dtype=torch.long)}
        with patch('uemo.data.prepare', prepared), patch('uemo.data.batch_prepare', batched):
            chunks = list(packed_batches(None, rows, Path('.'), 8, 12000))
        self.assertEqual(sum(len(targets) for _, targets, _ in chunks), 3)
        self.assertEqual([int(inputs['input_ids'][0, 0]) for inputs, _, _ in chunks], [1, 2, 3])
        self.assertTrue(all(int(inputs['input_ids'][0, 1]) == 7 for inputs, _, _ in chunks))

    def test_archive_reuse_and_traversal_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp)/'image-00001.tar'
            with tarfile.open(archive, 'w') as handle:
                member = tarfile.TarInfo('image/example.bin'); member.size = 3
                handle.addfile(member, io.BytesIO(b'abc'))
            extract_archive(archive); extract_archive(archive)
            self.assertEqual((Path(tmp)/'image/example.bin').read_bytes(), b'abc')
            (Path(tmp)/'image/example.bin').write_bytes(b'bad')
            with self.assertRaises(ValueError):
                extract_archive(archive)
            with tarfile.open(archive, 'w') as handle:
                member = tarfile.TarInfo('../escape'); member.size = 1
                handle.addfile(member, io.BytesIO(b'x'))
            with self.assertRaises(ValueError):
                extract_archive(archive)

    def test_temperature_fit_uses_saved_logits(self):
        rows = [dict(record(f'{i:09d}'), gt_answer='sad', option_logits=[4., 0.]) for i in range(1, 5)]
        fit = fit_temperature(rows)
        self.assertGreater(fit['temperature'], 0)
        self.assertLessEqual(fit['fit_ece10_after'], fit['fit_ece10_before'])
        for row in rows:
            self.assertEqual(int(np.argmax(row['option_logits'])), int(np.argmax(np.array(row['option_logits'])/fit['temperature'])))


if __name__ == '__main__':
    unittest.main()
