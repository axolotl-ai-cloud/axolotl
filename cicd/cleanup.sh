#!/bin/bash
set -e

# cleanup old cache files for datasets processing and intermediate mappings
find /workspace/data/huggingface-cache/hub/datasets -name "cache-*" -type f -mtime +1 -exec rm {} \;
find /workspace/data/huggingface-cache/hub/datasets -name "*.lock" -type f -mtime +1 -exec rm {} \;
