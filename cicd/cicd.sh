#!/bin/bash

pytest --ignore=tests/e2e/ /workspace/axolotl/tests/
pytest /workspace/axolotl/tests/e2e/patched/
pytest --ignore=tests/e2e/patched/ /workspace/axolotl/tests/e2e/
