import tempfile
from pathlib import Path
import unittest
from Deeplearning.issue17_cleanup import checked_file,safe_unlink
from Deeplearning.issue16 import digest


class CleanupBoundaryTests(unittest.TestCase):
    def test_deletes_only_matching_regular_file(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'owned';base.mkdir();path=base/'old.json';path.write_text('old')
            with self.assertRaises(RuntimeError):safe_unlink(path,base,'wrong hash')
            self.assertTrue(path.exists())
            safe_unlink(path,base,digest(path))
            self.assertFalse(path.exists())

    def test_parent_traversal_and_symlink_targets_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'owned';base.mkdir()
            outside=root/'outside';outside.mkdir();target=outside/'keep';target.write_text('keep')
            link=base/'link';link.symlink_to(target)
            parent=base/'parent';parent.symlink_to(outside,target_is_directory=True)
            for path in (link,parent/'keep',base/'..'/'outside'/'keep',target):
                with self.assertRaises(RuntimeError):checked_file(path,base)
            self.assertTrue(target.exists() and link.is_symlink() and parent.is_symlink())
