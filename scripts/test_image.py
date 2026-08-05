import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robokit.image import lememory_preprocess
from robokit.policies.memvla_lora import _preprocess


class LeMemoryPreprocessTest(unittest.TestCase):
    def test_realsense_frame_becomes_square_without_black_bars(self):
        image = np.full((240, 320, 3), 127, dtype=np.uint8)

        result = np.asarray(lememory_preprocess(image))

        self.assertEqual(result.shape, (224, 224, 3))
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.all(result == 127))

    def test_policy_and_collection_share_identical_transform(self):
        columns = np.linspace(0, 255, 320, dtype=np.uint8)
        image = np.repeat(columns[None, :, None], 240, axis=0)
        image = np.repeat(image, 3, axis=2)

        collection = np.asarray(lememory_preprocess(image))
        inference = np.asarray(_preprocess(image))

        np.testing.assert_array_equal(collection, inference)
        self.assertGreater(int(collection[:, 0].mean()), 0)
        self.assertLess(int(collection[:, -1].mean()), 255)

    def test_rejects_invalid_mode(self):
        with self.assertRaisesRegex(ValueError, "deploy\\|train_aligned"):
            lememory_preprocess(np.zeros((10, 10, 3), dtype=np.uint8), mode="pi05")


if __name__ == "__main__":
    unittest.main()
