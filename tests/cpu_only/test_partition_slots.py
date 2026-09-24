import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import AbstractMesh, AxisType, Mesh, NamedSharding, PartitionSpec as P

from jaxmg._cusolvermp_layout import (
    infer_mesh_and_matrix_specs,
    partition_slots_from_mesh,
    validate_2d_matrix_specs,
)


def _slots(mesh, specs):
    row_axis, col_axis, grid = validate_2d_matrix_specs(mesh, specs)
    return partition_slots_from_mesh(
        mesh, row_axis=row_axis, col_axis=col_axis, grid=grid, caller="test"
    )


def _axis_index_slots(mesh, specs):
    """Grid slots as `jax.lax.axis_index` sees them, from XLA's partition id."""
    sizes = dict(zip(mesh.axis_names, mesh.axis_sizes))
    strides = {
        name: int(np.prod(mesh.axis_sizes[i + 1 :], dtype=np.int64))
        for i, name in enumerate(mesh.axis_names)
    }
    row_axis, col_axis = specs
    cols = 1 if col_axis is None else sizes[col_axis]

    def coord(p, axis):
        return 0 if axis is None else (p // strides[axis]) % sizes[axis]

    return tuple(
        coord(p, row_axis) * cols + coord(p, col_axis)
        for p in range(int(np.prod(mesh.axis_sizes)))
    )


@pytest.mark.parametrize(
    "shape, names, specs",
    [
        ((2, 3), ("r", "c"), P("r", "c")),
        ((2, 3), ("r", "c"), P("c", "r")),
        ((6,), ("x",), P("x", None)),
        ((6,), ("x",), P(None, "x")),
        ((4, 2), ("a", "b"), P("b", "a")),
    ],
)
def test_partition_slots_match_axis_index(shape, names, specs):
    """Slots only need the abstract mesh, and agree with jax.lax.axis_index."""
    mesh = AbstractMesh(shape, names)
    assert _slots(mesh, specs) == _axis_index_slots(mesh, specs)


def test_partition_slots_row_and_column_major():
    mesh = AbstractMesh((2, 3), ("r", "c"))
    # Partitions are laid out row-major over the mesh, so P("r", "c") gives the
    # identity, and the transposed matrix sharding a column-major order.
    assert _slots(mesh, P("r", "c")) == (0, 1, 2, 3, 4, 5)
    assert _slots(mesh, P("c", "r")) == (0, 2, 4, 1, 3, 5)


def test_partition_slots_rejects_extra_mesh_axes():
    mesh = AbstractMesh((2, 3), ("r", "c"))
    with pytest.raises(ValueError, match="exactly the axes"):
        _slots(mesh, P("r", None))


def _single_device_mesh(axis_type):
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    return Mesh(devices, ("r", "c"), axis_types=(axis_type, axis_type))


@pytest.mark.parametrize("axis_type", [AxisType.Auto, AxisType.Explicit])
def test_infer_reads_the_sharding_of_a_eagerly(axis_type):
    mesh = _single_device_mesh(axis_type)
    a = jax.device_put(jnp.ones((4, 4)), NamedSharding(mesh, P("c", "r")))
    inferred_mesh, specs = infer_mesh_and_matrix_specs(a, mesh=None, matrix_specs=None)
    assert inferred_mesh.axis_names == ("r", "c")
    assert specs == P("c", "r")


@pytest.mark.parametrize(
    "axis_type, expected",
    [
        # Under jit, Auto shardings are not part of the type of A, so the
        # default follows the axes of the mesh; Explicit ones are.
        (AxisType.Auto, P("r", "c")),
        (AxisType.Explicit, P("c", "r")),
    ],
)
def test_infer_under_jit_without_concrete_mesh(axis_type, expected):
    """Under jit only the abstract mesh is available, which is all we need."""
    mesh = _single_device_mesh(axis_type)
    a = jax.device_put(jnp.ones((4, 4)), NamedSharding(mesh, P("c", "r")))
    seen = {}

    @jax.jit
    def f(a):
        seen["mesh"], seen["specs"] = infer_mesh_and_matrix_specs(
            a, mesh=None, matrix_specs=None
        )
        return a

    f(a)
    assert isinstance(seen["mesh"], AbstractMesh)
    assert seen["mesh"].axis_names == ("r", "c")
    assert seen["specs"] == expected


def test_infer_falls_back_to_the_context_mesh():
    mesh = Mesh(np.asarray(jax.devices()[:1], dtype=object), ("x",))
    with jax.set_mesh(mesh):
        inferred_mesh, specs = infer_mesh_and_matrix_specs(
            np.ones((4, 4)), mesh=None, matrix_specs=None
        )
    assert inferred_mesh.axis_names == ("x",)
    assert specs == P("x", None)


def test_infer_explicit_arguments_take_precedence():
    mesh = _single_device_mesh(AxisType.Auto)
    a = jax.device_put(jnp.ones((4, 4)), NamedSharding(mesh, P("c", "r")))
    other = AbstractMesh((1, 1), ("x", "y"))
    inferred_mesh, specs = infer_mesh_and_matrix_specs(
        a, mesh=other, matrix_specs=P("y", "x")
    )
    assert inferred_mesh == other
    assert specs == P("y", "x")


def test_infer_without_any_mesh_raises():
    with pytest.raises(ValueError, match="could not find a mesh"):
        infer_mesh_and_matrix_specs(np.ones((4, 4)), mesh=None, matrix_specs=None)
