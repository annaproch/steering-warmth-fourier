"""Cache isolation and paired scoring checks; no model weights needed."""
import unittest

import torch
from transformers.cache_utils import DynamicCache

from branch_pilot import clone_cache, prefix, result_row


class BranchChecks(unittest.TestCase):
    def test_cache_branches_are_isolated_and_interleaved(self):
        keys = torch.arange(12.0).reshape(2, 1, 3, 2)
        original = DynamicCache([(keys.clone(), (keys + 100).clone())])
        branched = clone_cache(original)
        branched.batch_repeat_interleave(2)
        self.assertEqual(branched.layers[0].keys.shape[0], 4)
        for i in range(4):
            torch.testing.assert_close(branched.layers[0].keys[i], keys[i // 2])
        branched.layers[0].keys[0].zero_()
        torch.testing.assert_close(original.layers[0].keys, keys)
        torch.testing.assert_close(branched.layers[0].keys[1], keys[0])

    def test_both_labels_required(self):
        partial = result_row({"prediction": [3, 0]}, (0, 0), (3, 5), "test", "both")
        self.assertTrue(partial["day_ok"])
        self.assertFalse(partial["month_ok"])
        self.assertFalse(partial["both_ok"])
        complete = result_row({"prediction": [3, 5]}, (0, 0), (3, 5), "test", "both")
        self.assertTrue(complete["both_ok"])

    def test_prefix_has_no_readout_question(self):
        self.assertEqual(prefix(0, 0), "Record: weekday=Monday; month=January.\nEnd of record.")
        self.assertTrue(prefix(3, 5).endswith("End of record."))


if __name__ == "__main__":
    unittest.main()
