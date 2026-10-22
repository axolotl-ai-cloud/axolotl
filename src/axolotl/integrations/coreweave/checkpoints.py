"""Publish immutable checkpoint artifacts before requesting a hot-load."""

import uuid

from axolotl.core.publishing.chain import ChainWriter
from axolotl.core.publishing.publisher import ChainPublisher
from axolotl.utils.delta_checkpoint import snapshot_weights

from .destination import CoreWeaveDestination


class CheckpointPublisher(ChainPublisher):
    """Advance the delta base only after upload and replica convergence succeed."""

    def __init__(self, config, client, directory, s3=None):
        self.config, self.client = config, client
        self.destination = CoreWeaveDestination(config, client, s3)
        self.s3 = self.destination.s3.client
        super().__init__(
            ChainWriter(
                directory,
                max_deltas=config.max_deltas,
                max_shard_size=config.max_shard_size,
            ),
            self.destination,
        )

    @property
    def directory(self):
        return self.chain.workdir

    @property
    def base(self):
        return self.chain.baseline

    def new_identity(self, step):
        return f"step-{step:08d}-{uuid.uuid4().hex}"

    def publish(self, weights, model_config, step, *, fingerprint=None):
        self.check_ready()
        return super().publish(
            snapshot_weights(weights),
            self.new_identity(step),
            config=model_config,
            fingerprint=fingerprint,
            copy=False,
        )

    def upload(self, directory, identity):
        """Upload referenced shards first and the checkpoint index last."""
        self.destination.upload(directory, identity)
