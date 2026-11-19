"""Guard for the unimplemented optional upstream graph module.

The upstream release contains an empty class declaration here. This local
compatibility fix makes the package valid Python but does not implement graph
memory; attempts to enable it must fail clearly.
"""


class GraphMem:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "LightMem's optional graph memory is not implemented in the upstream "
            "release. Set graph_mem=False or provide a verified implementation."
        )
