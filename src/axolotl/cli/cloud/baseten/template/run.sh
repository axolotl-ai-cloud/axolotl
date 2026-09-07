#!/bin/bash
set -eux

export NCCL_SOCKET_IFNAME="^docker0,lo"
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=1800000

axolotl preprocess train.yaml
axolotl train train.yaml --launcher ${AXOLOTL_LAUNCHER} ${AXOLOTL_LAUNCHER_ARGS}
