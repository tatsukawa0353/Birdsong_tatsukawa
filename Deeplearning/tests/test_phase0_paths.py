"""Migration compatibility and guards against replacing integrated unique data."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from Deeplearning import legacy_paths


class Phase0PathTests(unittest.TestCase):
    def test_old_protection_path_resolves_identical_bytes_after_real_move(self):
        with TemporaryDirectory(prefix="birdsong-phase0-path-") as temporary:
            repo=Path(temporary); parent=repo/"Deeplearning/results/phase0_fixed_target_20261108"
            old=repo/"Deeplearning/results/phase0"
            old.mkdir(parents=True); parent.mkdir()
            (old/"unique_glide.npz").write_bytes(b"unique legacy payload")
            expected=hashlib.sha256((old/"unique_glide.npz").read_bytes()).hexdigest()
            with patch.object(legacy_paths,"REPO",repo),patch.object(legacy_paths,"PHASE0_ROOT",parent):
                self.assertEqual(legacy_paths.legacy_phase0_root(),old)
                old.rename(parent/"legacy_random_target")
                path=legacy_paths.protected_path("Deeplearning/results/phase0/unique_glide.npz")
                self.assertEqual(path,parent/"legacy_random_target/unique_glide.npz")
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),expected)
                self.assertFalse(old.exists())

    def test_issue14_replacement_and_report_cannot_overwrite_integrated_results(self):
        from Deeplearning import issue14_transaction
        from Deeplearning.issue14_report import generate_report
        with TemporaryDirectory(prefix="birdsong-phase0-guard-") as temporary:
            root=Path(temporary)/"Deeplearning"
            parent=root/"results/phase0_fixed_target_20261108"
            (parent/"reward_ablation").mkdir(parents=True)
            marker=parent/"reward_ablation/valuable.txt";marker.write_text("preserve")
            with patch.object(issue14_transaction,"ROOT",root),patch.object(issue14_transaction,"DEFAULT_OUTPUT",parent):
                with self.assertRaises(RuntimeError): issue14_transaction.replace_results({},1)
            with self.assertRaises(ValueError): generate_report(parent)
            self.assertEqual(marker.read_text(),"preserve")


if __name__=="__main__":unittest.main()
