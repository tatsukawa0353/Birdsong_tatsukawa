import json
from pathlib import Path
import tempfile
import unittest
from Deeplearning.issue17_artifacts import compress_records,read_record


class CompleteRecordStorageTests(unittest.TestCase):
    def test_every_original_value_survives_compression_and_loader_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'runs/constant_1d/TD3/lambda1/seed_0';run.mkdir(parents=True)
            rows=[dict(episodes=i,reward=-i*.1234567891234567,action=[i/4096]) for i in range(4096)]
            path=run/'training_trace.json';path.write_text(json.dumps(rows,indent=2))
            result=compress_records(root)
            self.assertFalse(path.exists())
            self.assertEqual(read_record(path),rows)
            self.assertTrue(result['lossless'] and result['no_thinning'])
            self.assertEqual(len(result['entries']),1)
            self.assertLess(result['compressed_bytes'],result['original_bytes'])
