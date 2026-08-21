"""Import this before brax/playground training under JAX>=0.10."""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _device_put_replicated(value, devices):
  """Emulate removed jax.device_put_replicated for brax pmap training."""
  devices = list(devices)
  mesh = Mesh(np.array(devices), axis_names=("i",))
  sharding = NamedSharding(mesh, P("i"))

  def _put(x):
    x = jnp.asarray(x)
    # Leading device axis expected by brax._unpmap(...squeeze(0))
    x = jnp.broadcast_to(x, (len(devices),) + x.shape)
    return jax.device_put(x, sharding)

  return jax.tree_util.tree_map(_put, value)


def apply() -> None:
  # Always install shim compatible with brax>=0.14 on JAX 0.10+.
  jax.device_put_replicated = _device_put_replicated  # type: ignore[attr-defined]


apply()
