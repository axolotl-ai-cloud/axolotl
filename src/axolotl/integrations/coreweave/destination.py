"""CoreWeave Dedicated Inference as a checkpoint destination: S3 upload plus hot-load."""

from axolotl.core.publishing.chain import StagedCheckpoint
from axolotl.core.publishing.destinations import Destination, S3Destination
from axolotl.utils.schemas.checkpoint_publish import MAX_CHAIN_DELTAS


class CoreWeaveDestination(Destination):
    """Succeed only once every replica serves the uploaded snapshot."""

    max_deltas = MAX_CHAIN_DELTAS
    exports = ("merged",)

    @classmethod
    def check_export(cls, export: str) -> None:
        if export not in cls.exports:
            raise ValueError(
                "CoreWeave hot-load takes full model weights; use "
                "checkpoint_publish.export: merged"
            )

    def __init__(self, config, client, s3=None):
        self.config, self.client = config, client
        if s3 is None:
            try:
                import boto3
            except ImportError as exc:
                raise ImportError(
                    "Install axolotl[coreweave] to upload checkpoints"
                ) from exc

            s3 = boto3.Session(profile_name=config.s3_profile).client(
                "s3", endpoint_url=config.s3_endpoint_url
            )
        self.s3 = S3Destination(config.bucket, config.path_prefix, client=s3)

    def upload(self, directory, identity):
        self.s3.upload(directory, identity)

    def publish(self, staged: StagedCheckpoint) -> None:
        self.client.ready_identity = None
        self.upload(staged.path, staged.identity)
        self.client.hot_load(staged.identity, base_identity=staged.base_identity)

    def publish_files(self, directory, identity, names):
        self.s3.publish_files(directory, identity, names)

    def publish_index(self, directory, identity, *, base_identity, full):
        self.client.ready_identity = None
        self.s3.publish_index(
            directory, identity, base_identity=base_identity, full=full
        )
        self.client.hot_load(identity, base_identity=base_identity)

    def recover(self, staged: StagedCheckpoint, operation=None) -> None:
        """Confirm an existing or retried hot-load without issuing a duplicate."""
        if operation is None:
            self.publish(staged)
        else:
            self.client.wait_for_hot_load(operation, staged.identity)
