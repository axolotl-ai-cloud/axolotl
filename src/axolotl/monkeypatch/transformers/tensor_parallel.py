"""patches to fix broken tensor parallelism in transformers"""

import sys

import transformers.integrations.tensor_parallel


def distribute_model(model, distributed_config, device_mesh, tp_size):
    res = transformers.integrations.tensor_parallel.distribute_model(
        model,
        distributed_config,
        device_mesh,
        tp_size,
    )
    model._tp_size = tp_size  # pylint: disable=protected-access
    model._device_mesh = device_mesh  # pylint: disable=protected-access
    return res


def patch_tp_fix():
    transformers.integrations.tensor_parallel.distribute_model = distribute_model
    setattr(
        sys.modules["transformers.integrations.tensor_parallel"],
        "distribute_model",
        distribute_model,
    )
