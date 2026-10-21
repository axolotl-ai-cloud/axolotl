"""Memoize ``ModuleUtilsMixin.num_parameters`` per model and argument set.

HF Trainer calls ``floating_point_ops`` on every micro-step, and that recounts the
non-embedding parameters by walking ``named_modules`` and ``named_parameters``.
"""

from functools import wraps

from transformers.modeling_utils import ModuleUtilsMixin, PreTrainedModel

_CACHE_ATTR = "_axolotl_num_parameters_cache"


def _clear_cache(model) -> None:
    model.__dict__.pop(_CACHE_ATTR, None)


def patch_num_parameters():
    original = ModuleUtilsMixin.num_parameters
    if getattr(original, "_axolotl_cached", False):
        return

    @wraps(original)
    def num_parameters(
        self, only_trainable: bool = False, exclude_embeddings: bool = False
    ) -> int:
        # requires_grad can change at any time (freezing, adapters), so trainable
        # counts are never cached.
        if only_trainable:
            return original(
                self, only_trainable=True, exclude_embeddings=exclude_embeddings
            )
        # Kept on the instance (not an lru_cache keyed on self) so models can be freed.
        cache = self.__dict__.setdefault(_CACHE_ATTR, {})
        key = bool(exclude_embeddings)
        if key not in cache:
            cache[key] = original(self, exclude_embeddings=exclude_embeddings)
        return cache[key]

    num_parameters._axolotl_cached = True  # type: ignore[attr-defined]
    ModuleUtilsMixin.num_parameters = num_parameters  # type: ignore[method-assign]

    original_resize = PreTrainedModel.resize_token_embeddings

    @wraps(original_resize)
    def resize_token_embeddings(self, *args, **kwargs):
        result = original_resize(self, *args, **kwargs)
        _clear_cache(self)
        return result

    PreTrainedModel.resize_token_embeddings = resize_token_embeddings  # type: ignore[method-assign]
