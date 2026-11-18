import json
import tempfile
import unittest
from pathlib import Path

from uemo.data import load_records


class MetadataFormatTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(id='000000001', source='Example', source_id='Example_000000001',
                        subtask='Emotion', modality=['text'], text='Hello.', audio_path=[],
                        image_path=[], video_path=[], instruction='Identify the emotion.',
                        choices=['happy', 'sad'], gt_answer='happy')

    def test_formats_preserve_order_choices_and_context(self):
        first = dict(self.row, context=dict(modality=['text'], text='Earlier.',
                                           audio_path=[], image_path=[], video_path=[]))
        second = dict(self.row, id='000000002', source_id='Example_000000002',
                      instruction='What emotion is conveyed?', choices=['sad', 'happy'])
        rows = [first, second]
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'fixed.json', Path(tmp)/'diverse.jsonl'
            a.write_text(json.dumps(rows), encoding='utf-8')
            b.write_text('\n' + '\n\n'.join(json.dumps(r) for r in rows)+'\n', encoding='utf-8')
            self.assertEqual(load_records(a), rows)
            self.assertEqual(load_records(b), rows)

    def test_jsonl_keeps_validation_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'metadata.jsonl'
            for rows, message in [([self.row, self.row], 'Duplicate ID'),
                                  ([dict(self.row, gt_answer='unknown')], 'Invalid choices/answer')]:
                p.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, message):
                    load_records(p)
            p.write_text('\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'nonempty'):
                load_records(p)
            p.write_text(json.dumps(self.row)+'\n{bad json}\n', encoding='utf-8')
            with self.assertRaises(json.JSONDecodeError):
                load_records(p)


if __name__ == '__main__':
    unittest.main()
