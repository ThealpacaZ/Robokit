"""程序化构建 RLDS 数据集（等价于 tfds build，但不依赖 tfds CLI 的额外组件）。

用法:
    conda activate pi0_demo   # 或任何含 tensorflow/tfds 的环境
    python build.py --data /path/to/datasets/任务名 [--out ~/tensorflow_datasets] [--overwrite]

转换前必须先跑过 scripts/clean.py：目录里还留着被判 bad 的 episode 会直接拒绝转换
（--allow-unclean 可跳过）。
"""
import argparse
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="任务目录（含 N.hdf5）")
    parser.add_argument("--out", default=os.path.expanduser("~/tensorflow_datasets"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-unclean", action="store_true",
                        help="跳过 clean_report.json 检查（默认有 bad episode 就拒绝转换）")
    args = parser.parse_args()

    os.environ["ROBOKIT_DATA_DIR"] = args.data
    if args.allow_unclean:
        os.environ["ROBOKIT_ALLOW_UNCLEAN"] = "1"
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "robokit_dataset"))
    from robokit_dataset_dataset_builder import RobokitDataset

    if args.overwrite:
        target = os.path.join(args.out, "robokit_dataset")
        if os.path.exists(target):
            shutil.rmtree(target)

    builder = RobokitDataset(data_dir=args.out)
    builder.download_and_prepare()
    print(f"\nRLDS dataset written to {builder.data_dir}")
    print(builder.info)


if __name__ == "__main__":
    main()
