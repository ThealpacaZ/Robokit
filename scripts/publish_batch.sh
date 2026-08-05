#!/usr/bin/env bash
# 分批发布：把任务目录里还没发过的 episode **搬进**批次目录冻结，
# 清洗 → 转 RLDS → 把 HDF5 和 RLDS 一起传 HuggingFace → 逐文件比对远端大小 → 删本地 RLDS。
# 发布期间可以继续采下一批，新采的段留给下一次发布。
#
#   ./scripts/publish_batch.sh "stack cups"                     # 发到默认仓库
#   ./scripts/publish_batch.sh --repo lememory "stack cups"     # 指定仓库（不带 owner 就补 shaohuan1/）
#   ./scripts/publish_batch.sh --repo lememory "stack cups" 66 115   # 只发这个编号范围
#   ./scripts/publish_batch.sh --audit --repo lememory          # 核对整个仓库（所有任务）
#   ./scripts/publish_batch.sh --audit "stack cups"             # 只核对一个任务
#   KEEP_RLDS=1 ./scripts/publish_batch.sh "stack cups"         # 本机还要用 RLDS 就别让它删
#
# 先决条件：hf 已登录且是 role=write 的 token（`hf auth login`）。
#
# 一个仓库装多个任务，一个任务一个目录，任务下面一批一个目录：
#   <slug>/<批次>/hdf5/N.hdf5                       原始 HDF5
#   <slug>/<批次>/robokit_dataset/1.1.0/*.tfrecord  RLDS
#   <slug>/<批次>/source_meta/{config,clean_report}.json
# 本地按仓库分开镜像同一套结构（upload-large-folder 没有 --path-in-repo，
# 仓库内路径 = 相对上传根的路径，所以上传根就是"仓库名"那一层）：
#   datasets/_batches/<仓库>/<slug>/<批次>/hdf5/N.hdf5
#   ~/tensorflow_datasets/<仓库>/<slug>/<批次>/robokit_dataset/...
#
# 磁盘上每段 HDF5 任何时刻**只有一份**：发布前在 datasets/<任务名>/，冻结时 mv 进批次目录
# （同一分区，瞬间完成，不复制不硬链接）。RLDS 是派生物，核对完远端就地删。
#
# 台账 = <批次>.json（manifest，几十 KB，就放在任务目录下、和批次目录并排）：这一批的编号
# 范围 + 每个已上传文件的大小。批次数据目录随时可以 rm -rf（--audit 通过之后），manifest
# 留着就还能核对远端；整个 datasets/ 删掉也只是丢掉「下一批从哪接着发」和核对能力。
#
# 断点续跑：整条流水线幂等，断在哪一步重跑同一条命令即可。
#   - 冻结：批次目录已存在就复用，一律以它的**实际内容**为准。不带参数重跑会自动接着发
#           那个还没校验通过的批次，不会另起一批把它落下。
#   - 清洗：clean_report.json 在就跳过。
#   - 转换：只认「dataset_info.json 存在 且 episode 数等于批次段数」为完成；
#           半截产物（TFDS 中断会留 .incomplete*）一律判为未完成并整体重转。
#           TFDS 本身不支持转换中途续跑，所以按批切分——一批最多重来约 10 分钟。
#   - 上传：upload-large-folder 原生续传（状态在 <上传根>/.cache/huggingface/）。
#   - 校验通过才写 manifest 的 verified；已 verified 的批次直接跳过。
#
# 为什么必须先把这一批搬出任务目录：rlds builder 在构造时（_probe）和生成时
# （_generate_examples）各 glob 一次数据目录，两次之间新落盘的 episode 会绕过 clean 硬闸
# 被静默转进训练集。搬走之后这一批的内容不再变化，边发边采才是安全的。
#
# 编号水位：搬走后任务目录里可能一段都不剩，recorder 的 next_episode_index 会从 0 重新编号、
# 跟已发布的段撞名。所以冻结前先把 <任务目录>/.next_index 写成 TO+1，recorder 会读它。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

AUDIT=0
REPO_ARG=""
DROP_LOCAL="${DROP_LOCAL:-0}"
while [ $# -gt 0 ]; do
  case "$1" in
    --audit)      AUDIT=1; shift ;;
    --repo)       REPO_ARG="${2:?--repo 后面要跟仓库名}"; shift 2 ;;
    --repo=*)     REPO_ARG="${1#--repo=}"; shift ;;
    --drop-local) DROP_LOCAL=1; shift ;;   # 校验通过后删掉本地已发布的 HDF5
    *) break ;;
  esac
done

# 仓库：--repo 优先，其次 HF_REPO，最后默认值；不带 owner 的裸名字自动补上 HF_OWNER。
HF_OWNER="${HF_OWNER:-shaohuan1}"
HF_REPO="${REPO_ARG:-${HF_REPO:-Memoryvla}}"
case "$HF_REPO" in */*) ;; *) HF_REPO="$HF_OWNER/$HF_REPO" ;; esac
REPO_NAME="${HF_REPO##*/}"

PY="${ROBOKIT_TFDS_PYTHON:-$HOME/miniconda3/envs/pi0_demo/bin/python}"
HF="${ROBOKIT_HF_BIN:-$HOME/miniconda3/envs/pi0_demo/bin/hf}"
TFDS_ROOT="${TFDS_ROOT:-$HOME/tensorflow_datasets}"

BATCH_ROOT="datasets/_batches/$REPO_NAME"   # HDF5 的上传根
OUT_ROOT="$TFDS_ROOT/$REPO_NAME"            # RLDS 的上传根
mkdir -p "$BATCH_ROOT"

TASK="${1:-}"
SLUG="$(echo "$TASK" | tr ' ' '_')"

# ---------- --audit：只读核对，别用 HF 网页目测 ----------
# HF 网页按字典序排（一个批次目录的第一屏是 100.hdf5，66.hdf5 排在第 103 位），而且一个任务的
# episode 会跨多个批次目录，光看网页极易误判成"没传全"。用这个，不要用眼睛。
# 只读 manifest 和远端、不看本地数据，所以批次目录删掉之后照样能核对。
# 不给任务名就核对这个仓库里的所有任务。
if [ "$AUDIT" = 1 ]; then
  export ALL_PROXY= all_proxy=
  exec "$PY" - "$HF_REPO" "$BATCH_ROOT" "$SLUG" <<'PYEOF'
import glob, json, os, re, sys
from huggingface_hub import HfApi

repo, batch_root, slug = sys.argv[1:4]
tasks = [slug] if slug else sorted(
    os.path.basename(d) for d in glob.glob(f"{batch_root}/*")
    if os.path.isdir(d) and glob.glob(f"{d}/*.json"))
if not tasks:
    sys.exit(f"{batch_root} 下没有任何任务的 manifest：这个仓库还没从本机发过东西")

remote = {e.path: e.size for e in
          HfApi().list_repo_tree(repo, repo_type="dataset", recursive=True)
          if getattr(e, "size", None) is not None}

problems, accounted, held = [], set(), 0
print(f"仓库 {repo}")
for task in tasks:
    manifests = sorted(glob.glob(f"{batch_root}/{task}/*.json"),
                       key=lambda p: json.load(open(p))["from"])
    if not manifests:
        problems.append(f"{task}: 没有 manifest"); continue
    published, quarantined, task_held = set(), set(), 0
    print(f"\n  {task}")
    for mp in manifests:
        m = json.load(open(mp))
        # 清洗隔离的段本来就没打算发，不能算成"缺号"
        quarantined |= set(m.get("quarantined", []))
        published |= set(range(m["from"], m["to"] + 1)) - set(m.get("quarantined", []))
        accounted |= set(m["files"])
        n_bad = 0
        for path, size in m["files"].items():
            if path not in remote:
                problems.append(f"{path} 远端缺失"); n_bad += 1
            elif remote[path] != size:
                problems.append(f"{path} 大小不符 manifest {size} 远端 {remote[path]}"); n_bad += 1
        if not m.get("verified"):
            problems.append(f"{task}/{m['batch']}: manifest 没有 verified，上一次没跑完")

        local = [os.path.getsize(p)
                 for p in glob.glob(f"{batch_root}/{task}/{m['batch']}/hdf5/*.hdf5")]
        task_held += sum(local)
        print(f"    {m['batch']:<12} episode {m['from']}..{m['to']} ({m['n']} 段)  "
              f"{len(m['files']) - n_bad}/{len(m['files'])} 个文件已核对  "
              + (f"本地还占 {sum(local) / 2**30:.1f} GiB" if local else "本地已删"))

    gaps = [i for i in range(min(published), max(published) + 1)
            if i not in published and i not in quarantined]
    if gaps:
        problems.append(f"{task}: 已发布编号有缺口 {gaps[:20]}")
    if quarantined:
        print(f"    清洗隔离（不发，编号留空）: {sorted(quarantined)}")

    # 任务目录里剩下的就是还没发的段（已发布的在冻结时就搬走了）
    src = f"datasets/{json.load(open(manifests[0]))['task']}"
    todo = sorted(int(os.path.basename(p)[:-5]) for p in glob.glob(f"{src}/*.hdf5")
                  if re.fullmatch(r"\d+\.hdf5", os.path.basename(p)))
    held += task_held
    print(f"    合计已发布 {len(published)} 段 | 还有 {len(todo)} 段没发"
          + (f"：{todo[0]}..{todo[-1]}" if todo else "")
          + f" | 本地占 {task_held / 2**30:.1f} GiB")

# 只在核对整个仓库时报"远端多出来的"，否则同仓库里别的任务会被算成多余文件。
extra = [] if slug else [p for p in remote if p not in accounted and p != "README.md"]
if extra:
    print(f"\n  远端另有 {len(extra)} 个文件不在本机 manifest 里"
          f"（老布局 <slug>_<批次>/ 或别的机器发的），例：{sorted(extra)[:3]}")

if problems:
    print(f"\n★ {len(problems)} 处问题：")
    for p in problems[:30]:
        print("  " + p)
    sys.exit(1)
print("\n核对通过：manifest 记的每个文件远端都在、大小一致")
if held:
    print(f"可以删本地已发布数据了（manifest 是 .json，不受影响）：\n  rm -rf {batch_root}/*/*/")
PYEOF
fi

[ -n "$TASK" ] || { echo "用法: publish_batch.sh [--repo 仓库] [--audit] \"任务名\" [起始编号 结束编号]" >&2; exit 1; }
SRC="datasets/$TASK"
TASK_ROOT="$BATCH_ROOT/$SLUG"

# ---------- 1. 决定这一批的编号范围与批次名 ----------
# 没发完的批次优先：上一次断在哪一批，不带参数重跑就接着发那一批。没有这一步的话，
# 任务目录里剩下的段会被当成新的一批，断掉的那批就永远留在本地、再也不会被碰。
PENDING=$("$PY" - "$TASK_ROOT" <<'PYEOF'
import glob, json, os, re, sys
task_root = sys.argv[1]
for d in sorted(glob.glob(f"{task_root}/*/hdf5")):
    batch = os.path.basename(os.path.dirname(d))
    if not re.fullmatch(r"b\d+_\d+", batch):
        continue                                  # 手工建的怪名字，不自动接管
    mp = f"{task_root}/{batch}.json"
    if not (os.path.exists(mp) and json.load(open(mp)).get("verified")):
        print(batch); break
PYEOF
)

if [ $# -ge 3 ]; then
  FROM="$2"; TO="$3"
  BATCH="${BATCH:-b${FROM}_${TO}}"
elif [ -n "$PENDING" ]; then
  # 范围从批次名 b<起>_<止> 反解，不用占位值：上一次可能刚建完目录就断了，一段都还没搬进去，
  # 那时批次目录是空的，只有批次名还记得这一批本来要发哪些段。
  BATCH="$PENDING"
  FROM="${PENDING#b}"; FROM="${FROM%%_*}"; TO="${PENDING##*_}"
  echo "==> 发现没发完的批次 $BATCH（episode $FROM..$TO），先把它发完；要发别的批次就显式给编号范围"
else
  # 已发布的段在冻结时就搬走了，所以任务目录里剩下的**就是**还没发的段。
  [ -d "$SRC" ] || { echo "没有任务目录 $SRC" >&2; exit 1; }
  NUMS=$(ls "$SRC"/*.hdf5 2>/dev/null | sed 's/.*\///;s/\.hdf5//' | sort -n) || true
  [ -n "${NUMS:-}" ] || { echo "$SRC 里没有待发布的 .hdf5（都发过了？用 --audit 看）"; exit 0; }
  FROM=$(echo "$NUMS" | head -1)
  TO=$(echo "$NUMS" | tail -1)
  BATCH="${BATCH:-b${FROM}_${TO}}"
fi

DATA_DIR="$TASK_ROOT/$BATCH/hdf5"
OUT_DIR="$OUT_ROOT/$SLUG/$BATCH"
MANIFEST="$TASK_ROOT/$BATCH.json"
PREFIX="$SLUG/$BATCH"                        # 仓库内前缀 = 相对上传根的路径

if "$PY" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('verified') else 1)" \
       "$MANIFEST" 2>/dev/null; then
  echo "==> 批次 $BATCH 早已发布并校验通过（$MANIFEST），不重发。核对用 --audit"
  exit 0
fi
echo "==> $HF_REPO : $PREFIX（任务 $TASK）"

# ---------- 2. 冻结：把这一批搬进批次目录 ----------
if [ -d "$DATA_DIR" ] && ls "$DATA_DIR"/*.hdf5 >/dev/null 2>&1; then
  echo "==> 批次目录已存在，复用：$DATA_DIR"
else
  mkdir -p "$DATA_DIR"
  # 先抬编号水位再搬文件：反过来的话，中间这一刻 recorder 看到空目录就会从 0 重新编号。
  NEXT=$((TO + 1)); OLD=$(cat "$SRC/.next_index" 2>/dev/null || echo 0)
  [ "$OLD" -gt "$NEXT" ] && NEXT="$OLD"
  echo "$NEXT" > "$SRC/.next_index"
  for i in $(seq "$FROM" "$TO"); do
    [ -f "$DATA_DIR/$i.hdf5" ] && continue        # 上次搬到一半，接着搬
    [ -f "$SRC/$i.hdf5" ] || { echo "缺 $SRC/$i.hdf5" >&2; exit 1; }
    mv "$SRC/$i.hdf5" "$DATA_DIR/$i.hdf5"
  done
  echo "==> 冻结 $((TO-FROM+1)) 段 -> $DATA_DIR（move，本地仍然只有一份）"
fi

# 采集配置单独补：批次目录已存在时上面整段跳过，而上一次可能正是断在搬文件的中途。
if [ ! -f "$DATA_DIR/config.json" ]; then
  [ -f "$SRC/config.json" ] || { echo "缺 $SRC/config.json，补不出这一批的来源凭据" >&2; exit 1; }
  cp "$SRC/config.json" "$DATA_DIR/config.json"
fi

# 之后一律以批次目录的**实际内容**为准：断点重跑时冻结整段跳过，而这期间任务目录里
# 可能又多了几十段，拿调用方给的意图范围继续算会把没发过的段一起当成这一批。
ACTUAL=$(ls "$DATA_DIR"/*.hdf5 | sed 's/.*\///;s/\.hdf5//' | sort -n)
# 清洗 --quarantine 把 bad 段挪进 hdf5/_quarantine/：它们不转、不传、不核对，但仍属于这一批。
# 编号连续性按「留下的 + 隔离的」并集算，否则隔离掉一段这批就再也发不出去了。
QUAR=$( (ls "$DATA_DIR"/_quarantine/*.hdf5 2>/dev/null || true) | sed 's/.*\///;s/\.hdf5//' | sort -n)
UNION=$(printf '%s\n%s\n' "$ACTUAL" "$QUAR" | sed '/^$/d' | sort -n)
FROM=$(echo "$UNION" | head -1)
TO=$(echo "$UNION" | tail -1)
N=$(echo "$ACTUAL" | wc -l)
QUAR_CSV=$(echo "$QUAR" | sed '/^$/d' | paste -sd, -)
echo "==> 这一批实际内容：编号 $FROM..$TO，$N 段要发" \
     "${QUAR_CSV:+，清洗隔离 $QUAR_CSV 不发}"
[ "$(echo "$UNION" | wc -l)" -eq "$((TO - FROM + 1))" ] || {
  echo "批次编号不连续（既不在 hdf5/ 也不在 hdf5/_quarantine/），请人工核对 $DATA_DIR" >&2; exit 1; }

# ---------- 3. 清洗（有 bad 段就停；转换那道闸也会拦） ----------
if [ -f "$DATA_DIR/clean_report.json" ]; then
  echo "==> 清洗报告已存在，跳过"
else
  echo "==> 清洗"
  "$PY" scripts/clean.py --data "$DATA_DIR"
fi

# ---------- 4. 转 RLDS ----------
# NO_USE_EMBED=1：MemoryVLA 训练只读 language_instruction 原文，从不读 language_embedding，
# 那 512 维 USE 向量对这套栈是死重量。跳过省约 1 GB 下载 + 1.5 GB 常驻内存。
# 换成会消费 language_embedding 的训练栈时必须去掉这个变量重转。
rlds_done() {
  "$PY" - "$OUT_DIR" "$N" <<'PYEOF'
import glob, json, sys
out, want = sys.argv[1], int(sys.argv[2])
infos = glob.glob(f"{out}/robokit_dataset/*/dataset_info.json")
if not infos:
    sys.exit(1)                                   # 没有 info = 没转完（或只剩半截 .incomplete*）
info = json.load(open(infos[0]))
got = sum(int(x) for s in info.get("splits", []) for x in s.get("shardLengths", []))
sys.exit(0 if got == want else 1)
PYEOF
}
if rlds_done; then
  echo "==> RLDS 已完整（$N 段），跳过转换"
else
  [ -d "$OUT_DIR" ] && { echo "==> 检出半截/不匹配的 RLDS 产物，整体重转"; rm -rf "$OUT_DIR"; }
  # 转换耗时随分辨率涨得比像素数还快（PNG 编码是大头）：320×240 实测 9 s/段，
  # 640×480 像素多 4 倍但实测 ~75 s/段，是线性外推的两倍。所以查实测表，不做线性外推。
  SEC=$(ROBOKIT_HDF5="$(ls "$DATA_DIR"/*.hdf5 | head -1)" "$PY" - <<'PYEOF'
import os
MEASURED = {(240, 320): 9, (480, 640): 75}        # (高,宽) -> 秒/段，本机实测
try:
    import h5py
    with h5py.File(os.environ["ROBOKIT_HDF5"], "r") as f:
        g = f["observations/images"]
        h, w = g[next(iter(g))].shape[1:3]
    sec = MEASURED.get((h, w))
    if sec is None:                                # 没测过的分辨率：从最近的实测点按像素比推
        (bh, bw), base = min(MEASURED.items(), key=lambda kv: abs(kv[0][0] * kv[0][1] - h * w))
        sec = max(1, round(base * h * w / (bh * bw)))
    print(sec)
except Exception:
    print(9)
PYEOF
)
  echo "==> 转 RLDS -> $OUT_DIR（约 $SEC s/段，$N 段约 $((N * SEC / 60)) 分钟）"
  NO_USE_EMBED=1 nice -n 10 "$PY" rlds/build.py \
    --data "$DATA_DIR" --out "$OUT_DIR" --overwrite
  rlds_done || { echo "转换结束但 episode 数与批次不符，中止" >&2; exit 1; }
fi

# ---------- 5. 上传：HDF5 与 RLDS 一起 ----------
# ALL_PROXY 是 socks://，hf CLI 底层的 httpx 不认，必须清掉；
# HF_HUB_DISABLE_XET=1 否则在 "Finished hashing" 后卡到 2 KB/s。
# upload-large-folder 没有 --path-in-repo，仓库内路径 = 相对上传根的路径，所以两个上传根分别取
# BATCH_ROOT（HDF5）和 OUT_ROOT（RLDS），都天然带上 <slug>/<批次>/ 前缀。断了重跑即续传。
export ALL_PROXY= all_proxy= HF_HUB_DISABLE_XET=1

# --exclude 不是多余的：hf 的 --include 用 fnmatch，`*` 会跨过 `/`，
# 所以 "hdf5/*.hdf5" 连 hdf5/_quarantine/52.hdf5 一起匹配上，清洗判 bad 的段会被传上去。
echo "==> 上传 HDF5（$N 段）-> $HF_REPO:$PREFIX/hdf5/"
"$HF" upload-large-folder "$HF_REPO" "$BATCH_ROOT" --repo-type dataset \
  --include "$PREFIX/hdf5/*.hdf5" --exclude "*/_quarantine/*" --num-workers 8 || UPLOAD_HDF5_FAILED=1
if [ "${UPLOAD_HDF5_FAILED:-0}" = 1 ]; then echo "HDF5 上传失败，重跑本命令续传" >&2; exit 1; fi

echo "==> 上传 RLDS -> $HF_REPO:$PREFIX/robokit_dataset/"
"$HF" upload-large-folder "$HF_REPO" "$OUT_ROOT" --repo-type dataset \
  --include "$PREFIX/robokit_dataset/**" --num-workers 8

# 来源凭据：采集配置与清洗结论
echo "==> 上传来源凭据"
"$HF" upload "$HF_REPO" "$DATA_DIR/config.json" \
  "$PREFIX/source_meta/config.json" --repo-type dataset
"$HF" upload "$HF_REPO" "$DATA_DIR/clean_report.json" \
  "$PREFIX/source_meta/clean_report.json" --repo-type dataset

# ---------- 6. 逐文件比对远端大小，通过了才写 manifest ----------
# manifest 是这一批唯一的台账：本地数据删掉以后，--audit 就只认它。
echo "==> 校验远端"
"$PY" - "$HF_REPO" "$MANIFEST" "$TASK" "$SLUG" "$BATCH" "$FROM" "$TO" "$N" "$(date -Iseconds)" "$QUAR_CSV" \
  "$DATA_DIR|$PREFIX/hdf5|.hdf5" \
  "$DATA_DIR|$PREFIX/source_meta|.json" \
  "$OUT_DIR/robokit_dataset|$PREFIX/robokit_dataset|" <<'PYEOF'
import json, os, sys
from huggingface_hub import HfApi

# 前 10 个参数是元信息，之后每个参数是 本地目录|仓库前缀|后缀过滤（空 = 全部）。
# 批次的 hdf5 目录里同时躺着 .hdf5 和 config/clean_report 两个 .json，它们上传到不同前缀，
# 所以按后缀拆成两条；不拆的话 .json 会被判成"hdf5/ 下远端缺失"。
repo, manifest_path, task, slug, batch = sys.argv[1:6]
first, last, n, stamp = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), sys.argv[9]
quarantined = [int(x) for x in sys.argv[10].split(",") if x]
pairs = [a.split("|") for a in sys.argv[11:]]

api, bad, files = HfApi(), [], {}
for local_root, prefix, suffix in pairs:
    remote = {
        e.path[len(prefix) + 1:]: e.size
        for e in api.list_repo_tree(repo, repo_type="dataset", path_in_repo=prefix, recursive=True)
        if getattr(e, "size", None) is not None
    }
    for dirpath, dirnames, names in os.walk(local_root):
        dirnames[:] = [d for d in dirnames if d != "_quarantine"]   # 隔离段不传，也不该核对
        for name in names:
            if suffix and not name.endswith(suffix):
                continue
            p = os.path.join(dirpath, name)
            rel = os.path.relpath(p, local_root)
            size = os.path.getsize(p)
            files[f"{prefix}/{rel}"] = size
            if rel not in remote:
                bad.append(f"缺失: {prefix}/{rel}")
            elif remote[rel] != size:
                bad.append(f"大小不符: {prefix}/{rel} 本地 {size} 远端 {remote[rel]}")
if bad:
    print("\n".join(bad)); sys.exit(1)

os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump({"task": task, "slug": slug, "batch": batch, "repo": repo,
               "from": first, "to": last, "n": n, "quarantined": quarantined,
               "verified": stamp, "files": files},
              f, ensure_ascii=False, indent=1)
print(f"校验通过：{len(files)} 个文件本地与远端大小全一致 -> {manifest_path}")
PYEOF

# ---------- 7. 删本地 RLDS ----------
# RLDS 是 HDF5 的派生物，而且刚刚逐文件核对上了远端，本地留着就是纯粹的第二份数据。
# 本机要用（训练/调试）就 KEEP_RLDS=1，或者从 HF 拉，或者重转一次（约 9 s/段）。
if [ "${KEEP_RLDS:-0}" = 1 ]; then
  echo "==> KEEP_RLDS=1，保留本地 RLDS：$OUT_DIR"
else
  rm -rf "$OUT_DIR"
  echo "==> 已删本地 RLDS：$OUT_DIR（远端已核对；要用就 KEEP_RLDS=1 重跑或重转）"
fi

# ---------- 7b. --drop-local：连本地 HDF5 一起删（磁盘紧张时边发边放空间） ----------
# 删除清单**只**取自刚写完的 manifest 里 <前缀>/hdf5/ 那些条目，也就是逐文件比对过
# 「远端存在且大小一致」的那批。清洗隔离的段不在 manifest 里（根本没上传），所以永远
# 不会被这里删掉——它们是那几段唯一的副本。clean_report.json / config.json 也留着。
if [ "$DROP_LOCAL" = 1 ]; then
  "$PY" - "$MANIFEST" "$DATA_DIR" "$PREFIX/hdf5/" <<'PYEOF'
import json, os, sys
manifest_path, data_dir, prefix = sys.argv[1:4]
manifest = json.load(open(manifest_path))
if not manifest.get("verified"):
    sys.exit("manifest 没有 verified，拒绝删本地数据")      # 兜底：只删核对过的
freed = 0
for path in manifest["files"]:
    if not path.startswith(prefix):
        continue
    local = os.path.join(data_dir, os.path.basename(path))
    if os.path.isfile(local):
        freed += os.path.getsize(local)
        os.remove(local)
print(f"==> --drop-local：已删 {freed / 2**30:.1f} GiB 本地 HDF5"
      f"（只删 manifest 里核对过的；隔离段与报告保留）")
PYEOF
fi

# ---------- 8. 刷新仓库首页索引 ----------
# HF 网页按字典序列目录（第一屏是 100.hdf5，66.hdf5 在第 103 位），而且 episode 跨多个
# 批次目录，光靠翻页目测必然误判成"没传全"。把真实范围写在首页，一眼可查。
# 索引从**远端文件列表**现算，不是从本机 manifest：一个仓库装多个任务，别的任务可能是别的
# 机器发的、或者是老布局，只写本机知道的那部分会把它们从首页抹掉。
CARD=$(mktemp); trap 'rm -f "$CARD"' EXIT
export ALL_PROXY= all_proxy=
"$PY" - "$HF_REPO" > "$CARD" <<'PYEOF'
import re, sys
from collections import defaultdict
from huggingface_hub import HfApi

repo = sys.argv[1]
paths = [e.path for e in HfApi().list_repo_tree(repo, repo_type="dataset", recursive=True)]

eps = defaultdict(list)          # (task, batch) -> [episode 编号]
rlds = set()                     # 有 RLDS 的 (task, batch)
for p in paths:
    # 新布局 <slug>/<批次>/...；老布局 <slug>_<批次>/... （批次名一律 b 开头）
    m = re.fullmatch(r"([^/]+)/(b[^/]*)/(hdf5/(\d+)\.hdf5|robokit_dataset/.+)", p) \
        or re.fullmatch(r"([^/]+?)_(b\d+(?:_\d+)?)/(hdf5/(\d+)\.hdf5|robokit_dataset/.+)", p)
    if not m:
        continue
    key = (m.group(1), m.group(2))
    if m.group(4):
        eps[key].append(int(m.group(4)))
    else:
        rlds.add(key)

print(f"# {repo.split('/')[-1]}（robokit 采集）")
print()
print("一个任务一个目录，任务下面一批一个目录：")
print("`<任务>/<批次>/hdf5/` 原始 HDF5、`robokit_dataset/` RLDS、`source_meta/` 采集配置与清洗报告。")
for task in sorted({k[0] for k in eps}):
    batches = sorted((k[1] for k in eps if k[0] == task),
                     key=lambda b: min(eps[(task, b)]))
    total = sum(len(eps[(task, b)]) for b in batches)
    print()
    print(f"## {task}（{total} 段）")
    print()
    print("| 批次 | episode | 段数 | RLDS |")
    print("|---|---|---|---|")
    for b in batches:
        ns = sorted(eps[(task, b)])
        missing = [i for i in range(ns[0], ns[-1] + 1) if i not in set(ns)]
        flag = "✓" if (task, b) in rlds else "—"
        print(f"| `{b}` | {ns[0]}..{ns[-1]} | {len(ns)}"
              + (f" ★空号 {missing[:5]}" if missing else "") + f" | {flag} |")
print()
print("> **文件列表按字典序显示**（`100.hdf5` 排在 `66.hdf5` 前面），翻页目测容易误判成缺文件。")
print("> ★空号 = 该编号远端没有，通常是清洗判 bad 被隔离、或采集时被丢弃，不一定是漏传。")
print("> 上表是按远端实际文件列表现算的。逐文件核对大小用 "
      "`./scripts/publish_batch.sh --audit --repo <仓库>`。")
PYEOF
"$HF" upload "$HF_REPO" "$CARD" "README.md" --repo-type dataset >/dev/null
echo "==> 已刷新仓库首页索引（按远端实际文件列表现算，含仓库里的其它任务）"

echo "==> 完成。这一批本地只剩 HDF5 一份：$DATA_DIR"
echo "    远端已逐文件核对通过，随时可以 rm -rf 它（manifest $MANIFEST 留着，--audit 照样能核对）"
