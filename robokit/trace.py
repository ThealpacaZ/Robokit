"""结构化执行轨迹（JSONL）。真机部署和仿真回放写同一种格式，事后用同一套脚本分析。

每行一个 JSON 对象。ChunkExecutor 每下发一步写一条；客户端每轮推理写一条 event 记录。
"""
import json
import os
import threading

import numpy as np


def _default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


class TraceWriter:
    def __init__(self, path):
        self.path = str(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record):
        line = json.dumps(record, default=_default, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")

    def close(self):
        with self._lock:
            if not self._fh.closed:
                self._fh.flush()
                self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class MemoryTrace:
    """内存版 trace：测试和诊断可直接读取 records，不落盘。接口与 TraceWriter 兼容。"""

    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)

    def close(self):
        pass

    def dump(self, path):
        with TraceWriter(path) as w:
            for record in self.records:
                w.write(record)


def read_trace(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
