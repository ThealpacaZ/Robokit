#!/usr/bin/env python
"""episode 编号水位的纯软件回归测试；不碰硬件、不写真实数据目录。

publish_batch.sh 冻结一批时把 HDF5 **搬进**批次目录（本地只留一份），任务目录可能被搬空。
只按现存文件编号的话下一段会退回 0.hdf5，跟远端已发布的段同名不同内容——这个测试盯的就是
那道水位（<任务目录>/.next_index）。
"""

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from robokit.recorder import next_episode_index


class EpisodeNumberingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.addCleanup(self.tmp.cleanup)

    def _touch(self, *names):
        for name in names:
            open(os.path.join(self.dir, name), "w").close()

    def _mark(self, value):
        with open(os.path.join(self.dir, ".next_index"), "w") as f:
            f.write(str(value))

    def test_empty_dir_starts_at_zero(self):
        self.assertEqual(next_episode_index(self.dir), 0)

    def test_counts_from_existing_files(self):
        self._touch("0.hdf5", "1.hdf5", "2.hdf5")
        self.assertEqual(next_episode_index(self.dir), 3)

    def test_marker_carries_numbering_across_a_publish(self):
        # 0..4 已发布并被搬走，目录里只剩 config.json 和水位文件
        self._mark(5)
        self.assertEqual(next_episode_index(self.dir), 5)

    def test_existing_files_win_when_marker_is_behind(self):
        # 水位是上一批发布时写的，之后又采了 5..7；现存文件更靠后就听现存文件的
        self._mark(5)
        self._touch("5.hdf5", "6.hdf5", "7.hdf5")
        self.assertEqual(next_episode_index(self.dir), 8)

    def test_corrupt_marker_falls_back_to_files(self):
        self._touch("0.hdf5", "1.hdf5")
        self._mark("garbage")
        self.assertEqual(next_episode_index(self.dir), 2)

    def test_non_episode_files_are_ignored(self):
        self._touch("config.json", "clean_report.json", "3.hdf5.tmp")
        self.assertEqual(next_episode_index(self.dir), 0)


if __name__ == "__main__":
    unittest.main()
