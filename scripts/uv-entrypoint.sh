#!/bin/bash

if [ -f /workspace/axolotl/scripts/cuda13_env.sh ]; then
    source /workspace/axolotl/scripts/cuda13_env.sh
fi

exec "$@"
