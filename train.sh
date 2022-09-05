#!/bin/bash

DELAY="${1:-0}"
PLAN_HORIZON="$(($DELAY + 1))"

echo "Executing with DELAY $DELAY and PLAN_HORIZON $PLAN_HORIZON"

python mbexp.py -logdir ./log/DATS \
    -env zinc-coating-v0 \
    -o exp_cfg.exp_cfg.plan_hor $PLAN_HORIZON \
    -o exp_cfg.sim_cfg.delay_hor $DELAY \
    -o ctrl_cfg.prop_cfg.delay_step $DELAY \
    -ca opt-type CEM \
    -ca model-type PE \
    -ca prop-type E
