import pytest
import torch._dynamo.config as dynamo_config


@pytest.fixture(autouse=True)
def _restore_dynamo_recompile_limit():
    """A flex ``EncoderCanvasBackend`` raises dynamo's recompile limit process-wide."""
    with dynamo_config.patch(recompile_limit=dynamo_config.recompile_limit):
        yield
