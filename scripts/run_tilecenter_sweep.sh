#!/usr/bin/env bash
set -e

TASK_CFG=configs/tasks/tile_center.yaml
DATA_ROOT=data/micro/gt
SAVE_ROOT=outputs/tile_center
STEPS=${STEPS:-150}
DCW=${DCW:-0.5}

for s in 2 3 4 5 6 7 8 9; do
  python sample_condition.py \
    --task-config ${TASK_CFG} \
    --data-root ${DATA_ROOT} \
    --save-root ${SAVE_ROOT} \
    --steps ${STEPS} \
    --dc-weight ${DCW} \
    --s ${s}
done
