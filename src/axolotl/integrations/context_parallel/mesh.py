"""Expose Ringmaster's two USP axes without changing Accelerate's CP mesh."""

from torch.distributed.device_mesh import DeviceMesh


class RingmasterMesh:
    """Keep the full CP group and derive ring/ulysses subgroups within it."""

    def __init__(self, mesh, *, ring_size, ulysses_size):
        names = list(mesh.mesh_dim_names)
        cp_axis = names.index("cp")
        shape = list(mesh.mesh.shape)
        if shape[cp_axis] != ring_size * ulysses_size:
            raise ValueError("Ringmaster degrees must multiply to the CP mesh size")
        names[cp_axis : cp_axis + 1] = ["cp_ring", "cp_ulysses"]
        shape[cp_axis : cp_axis + 1] = [ring_size, ulysses_size]
        self._cp = mesh["cp"]
        self._split = DeviceMesh(
            mesh.device_type, mesh.mesh.reshape(shape), mesh_dim_names=tuple(names)
        )
        self.mesh_dim_names = ("cp", *names)

    def __getitem__(self, name):
        return self._cp if name == "cp" else self._split[name]
