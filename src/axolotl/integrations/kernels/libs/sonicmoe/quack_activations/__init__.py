# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CuTe DSL epilogues registered into the hub sonic-moe kernel's vendored quack at runtime.

quack keys its on-disk kernel cache on its own sources; this directory is added to that
fingerprint (``quack.cache.EXTRA_SOURCE_DIRS``) when an epilogue from here is registered, so an
edit here recompiles instead of loading stale kernels. Keep only epilogue sources in it.
"""
