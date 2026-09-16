import os
import sys
import traceback
from functools import partial

import jax

if not jax.config.jax_enable_x64:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P

from cusolvermp_case_utils import (
    SolverCase,
    dtype_from_name,
    emit,
    global_array_to_numpy,
    local_device_id_for_process,
    make_process_mesh,
    native_status_words,
    select_gpu_allocator,
)


coord_addr = sys.argv[1]
proc_id = int(sys.argv[2])
num_procs = int(sys.argv[3])
case_name = sys.argv[4]
dtype_name = sys.argv[5]
interface = os.environ.get("JAXMG_TEST_INTERFACE", "public")

select_gpu_allocator(proc_id)

jax.distributed.initialize(
    coordinator_address=coord_addr,
    num_processes=num_procs,
    process_id=proc_id,
    local_device_ids=[local_device_id_for_process(proc_id)],
    coordinator_bind_address=coord_addr if proc_id == 0 else None,
)

from jaxmg import polar, polar_shardmap_ctx
from jaxmg._cusolvermp_status import _CUSOLVERMP_POLAR_STATUS_SIZE


def run_case() -> None:
    """Run one padded polar decomposition and validate its requested factors."""
    dtype = dtype_from_name(dtype_name)
    compute_h = case_name != "padded_u"
    process_rows, process_cols = num_procs, 1
    m, n, tile_size = 384, 192, 128
    case = SolverCase(
        process_rows=process_rows,
        process_cols=process_cols,
        grid_order="row_major",
        n=m,
        tile_size=tile_size,
    )
    mesh = make_process_mesh(case)
    matrix_specs = P("pr", "pc")

    # A positive rectangular diagonal has an exact polar decomposition while
    # exercising different padded local shapes for A and H.
    a_host = np.zeros((m, n), dtype=np.dtype(dtype))
    diagonal = np.linspace(1.0, 2.0, n).astype(a_host.real.dtype)
    a_host[np.arange(n), np.arange(n)] = diagonal
    a_dev = jax.device_put(a_host, NamedSharding(mesh, matrix_specs))

    if interface == "context":

        @partial(jax.jit, donate_argnums=(0,), static_argnames=("tile_size",))
        def decomposition(_a, *, tile_size):
            return polar_shardmap_ctx(
                _a,
                tile_size,
                mesh=mesh,
                matrix_specs=matrix_specs,
                compute_h=compute_h,
            )

        outputs = decomposition(a_dev, tile_size=tile_size)
    else:
        outputs = polar(
            a_dev,
            tile_size,
            mesh=mesh,
            matrix_specs=matrix_specs,
            compute_h=compute_h,
            return_status=True,
        )

    if compute_h:
        up, h, status = outputs
    else:
        up, status = outputs
        h = None
    up.block_until_ready()
    if h is not None:
        h.block_until_ready()
    status.block_until_ready()

    status_words = native_status_words(status)
    assert status_words.size % _CUSOLVERMP_POLAR_STATUS_SIZE == 0, status_words
    assert np.all(status_words[::_CUSOLVERMP_POLAR_STATUS_SIZE] == 0), status_words
    assert np.all(
        status_words[20::_CUSOLVERMP_POLAR_STATUS_SIZE] == int(compute_h)
    ), status_words
    assert np.all(status_words[26::_CUSOLVERMP_POLAR_STATUS_SIZE] == 1), status_words
    assert np.all(status_words[27::_CUSOLVERMP_POLAR_STATUS_SIZE] == 0), status_words

    up_host = global_array_to_numpy(up)
    np.testing.assert_allclose(
        up_host.conj().T @ up_host,
        np.eye(n, dtype=up_host.dtype),
        rtol=5e-3,
        atol=5e-3,
    )
    if h is not None:
        h_host = global_array_to_numpy(h)
        np.testing.assert_allclose(h_host, h_host.conj().T, rtol=5e-3, atol=5e-3)
        np.testing.assert_allclose(up_host @ h_host, a_host, rtol=5e-3, atol=5e-3)

    emit(
        "GPU_TEST_RESULT",
        {
            "proc": proc_id,
            "name": case_name,
            "dtype": dtype_name,
            "status": "ok",
            "interface": interface,
            "compute_h": compute_h,
            "params": {
                "m": m,
                "n": n,
                "tile_size": tile_size,
                "process_rows": process_rows,
                "process_cols": process_cols,
            },
        },
    )
    multihost_utils.sync_global_devices(
        f"polar_{case_name}_{dtype_name}_{num_procs}_complete"
    )


def main() -> None:
    try:
        run_case()
    except Exception:
        emit(
            "GPU_TEST_RESULT",
            {
                "proc": proc_id,
                "name": case_name,
                "dtype": dtype_name,
                "status": "fail",
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        emit(
            "GPU_TEST_SUMMARY",
            {"proc": proc_id, "name": case_name, "dtype": dtype_name},
        )


if __name__ == "__main__":
    main()
