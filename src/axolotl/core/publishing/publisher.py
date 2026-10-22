"""Compose a chain writer with a destination under commit-after-publish semantics."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .chain import ChainWriter, StagedCheckpoint
from .destinations import Destination

if TYPE_CHECKING:
    from .exporters import Exporter


class ChainPublisher:
    """Advance the committed baseline only after the destination accepted a checkpoint.

    A failed publish stays ``pending``: ``recover`` completes it, ``abandon`` drops
    it so the next publish is built against the last commit again.
    """

    def __init__(self, chain: ChainWriter, destination: Destination):
        self.chain, self.destination = chain, destination
        if (
            destination.max_deltas is not None
            and chain.max_deltas > destination.max_deltas
        ):
            chain.max_deltas = destination.max_deltas
        self.pending: StagedCheckpoint | None = None

    @property
    def identity(self) -> str | None:
        return self.chain.identity

    @property
    def delta_count(self) -> int:
        return self.chain.delta_count

    @property
    def baseline(self):
        return self.chain.baseline

    def check_ready(self):
        if self.pending is not None:
            raise RuntimeError(
                "A checkpoint publication is pending; recover it before publishing again"
            )

    def publish(
        self,
        tensors: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
        identity: str,
        *,
        config: dict | None = None,
        fingerprint: dict | None = None,
        layout=None,
        copy: bool = True,
    ) -> str | None:
        """Publish and commit; returns the committed identity (unchanged if no delta)."""
        self.check_ready()
        staged = self.chain.write(
            tensors,
            identity,
            config=config,
            fingerprint=fingerprint,
            layout=layout,
            copy=copy,
        )
        return self.submit(staged)

    def publish_export(
        self,
        exporter: "Exporter",
        adapter: Path | str,
        identity: str,
        *,
        fingerprint: dict | None = None,
    ) -> str | None:
        """Stream an exporter's tensors straight into a chain session and publish."""
        self.check_ready()
        adapter = Path(adapter)
        self.destination.check_export(exporter.kind)
        exporter.validate(adapter)
        if fingerprint is None:
            fingerprint = exporter.fingerprint(adapter)
        with self.chain.begin(
            identity, fingerprint=fingerprint, layout=exporter.layout(adapter)
        ) as session:
            config = exporter.export(
                adapter, lambda name, tensor: session.add(name, tensor, copy=False)
            )
            staged = session.finish(config)
        return self.submit(staged)

    def submit(self, staged: StagedCheckpoint) -> str | None:
        self.check_ready()
        if staged.empty:
            self.chain.discard(staged)
            return self.chain.identity
        self.pending = staged
        self.destination.publish(staged)
        self.commit()
        return staged.identity

    def recover(self, operation: Any = None):
        if self.pending is None:
            raise RuntimeError("No pending checkpoint to recover")
        self.destination.recover(self.pending, operation)
        self.commit()

    def abandon(self):
        if self.pending is not None:
            self.chain.discard(self.pending)
            self.pending = None

    def commit(self):
        assert self.pending is not None
        self.chain.commit(self.pending)
        self.pending = None
