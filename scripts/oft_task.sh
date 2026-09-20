#!/usr/bin/env bash
# 一条命令切 OpenVLA-OFT 的任务：重起服务端并换成该任务的反归一化键，等到就绪，
# 再把真机端要跑的命令打出来。
#
#   bash scripts/oft_task.sh                 # 列出所有任务
#   bash scripts/oft_task.sh buttons         # 切到按钮任务
#   bash scripts/oft_task.sh buttons --run   # 切完直接跑真机（带 --demo 录像）
#
# 为什么要这一步：OFT 是多任务模型，提示词决定做什么，反归一化键决定动作的物理
# 尺度，两者必须同时换。只换提示词不换键不会报错，只会让整条轨迹按别的任务的
# 尺度走 —— 这种错很难从现象上看出来。
set -uo pipefail
HOST=${ROBOKIT_SSH_HOST:-westb2}
REMOTE_REPO=${REMOTE_REPO:-/root/robokit}
PY=${REMOTE_PYTHON:-/root/autodl-tmp/envs/openvla_oft/bin/python}
MODEL=openvla-oft-alltask
LOG=/root/autodl-tmp/outputs/eval-services/$MODEL-sync.log

# 短名 | 提示词（逐字，OFT 不带 <control mode> 后缀）| 反归一化键后缀
TASKS="
buttons|Press and hold the red and blue buttons in sequence for a while|press_and_hold_the_red_and_blue_buttons_in_sequence_for_a_while
cover|Cover the building block with a cup, then lift up the cup covering the block.|cover_the_building_block_with_a_cup_then_lift_up_the_cup_covering_the_block_b0
drawer3|Put the three items on the table into the drawer, then close the drawer.|put_the_three_items_on_the_table_into_the_drawer_then_close_the_drawer
drawer|Open the drawer, put the fruit and the cup from the table inside, and close the drawer.|open_the_drawer_put_the_fruit_and_the_cup_from_the_table_inside_and_close_the_drawer
stack|Stack one cup on top of another cup|stack_one_cup_on_top_of_another_cup_b0
waxapple|Place the banana, kiwi, and wax apple into the basket in that order.|place_the_banana_kiwi_and_wax_apple_into_the_basket_in_that_order
grape|Place the banana, mangoe, and grape into the basket in that order.|place_the_banana_mangoe_and_grape_into_the_basket_in_that_order
cherry|Place the banana, kiwi, and cherry into the basket in that order.|place_the_banana_kiwi_and_cherry_into_the_basket_in_that_order
"

list_tasks() {
  printf '%-10s %s\n' "短名" "提示词"
  echo "$TASKS" | while IFS='|' read -r name prompt _; do
    [ -n "$name" ] && printf '%-10s %s\n' "$name" "$prompt"
  done
}

[ $# -ge 1 ] || { echo "用法: bash scripts/oft_task.sh <短名> [--run]"; echo; list_tasks; exit 0; }

NAME=$1; RUN=${2:-}
LINE=$(echo "$TASKS" | grep "^$NAME|" || true)
[ -n "$LINE" ] || { echo "没有这个任务: $NAME"; echo; list_tasks; exit 1; }
PROMPT=$(echo "$LINE" | cut -d'|' -f2)
KEY=robokit_eefbase_$(echo "$LINE" | cut -d'|' -f3)

echo "任务   : $PROMPT"
echo "反归一化: $KEY"
echo "切换服务端（$HOST）…"
# 走 serve_ctl 而不是直接 setsid：它记 pidfile，下次 stop/status 才找得到这个服务。
# serve_ctl 把 model/mode 之后的参数原样透传给 serve_policy.py。
ssh "$HOST" "cd $REMOTE_REPO && bash scripts/serve_ctl.sh stop $MODEL sync >/dev/null 2>&1; sleep 4; \
  cd $REMOTE_REPO && CUDA_VISIBLE_DEVICES=0 bash scripts/serve_ctl.sh start $MODEL sync --unnorm-key '$KEY'" || exit 1

printf '等待加载'
for _ in $(seq 1 40); do
  sleep 10; printf '.'
  state=$(ssh "$HOST" "grep -E 'listening on|Traceback|Error' $LOG 2>/dev/null | grep -v Warning | tail -1" 2>/dev/null)
  case "$state" in
    *"listening on"*) echo; echo "服务就绪。"; break ;;
    *Traceback*|*Error*) echo; echo "服务启动失败："; echo "$state"; exit 1 ;;
  esac
done
ssh "$HOST" "grep -o 'unnorm_key=[^ ]*' $LOG | tail -1" 2>/dev/null

CMD="python scripts/run_policy.py --model $MODEL --mode sync --demo -L \"$PROMPT\""
echo
if [ "$RUN" = "--run" ]; then
  echo "> $CMD"; eval "$CMD"
else
  echo "真机端跑："; echo "$CMD"
fi
