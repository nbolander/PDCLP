"""Standalone driver mirroring peripy_fracture_demo.ipynb, so we can iterate
on the PeriPy API from the command line without reloading JupyterLab.

Run:
    conda activate peripy_env
    python test_demo.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import meshio

from peripy.model import Model, initial_crack_helper
from peripy.integrators import EulerCromerCL


# --------------------------------------------------------------------------- #
# Toggles — keep the smoke test cheap while we iterate.
# --------------------------------------------------------------------------- #
RUN_SIMULATE = True      # set False to stop after Model construction
N_STEPS = 2000
WRITE_EVERY = 200
SAVE_PLOT = True


# --------------------------------------------------------------------------- #
# 1. Mesh generation
# --------------------------------------------------------------------------- #
L_x, L_y = 1.0, 0.5
nx, ny = 81, 41
dx = L_x / (nx - 1)
dy = L_y / (ny - 1)

xs = np.linspace(0.0, L_x, nx)
ys = np.linspace(0.0, L_y, ny)
xx, yy = np.meshgrid(xs, ys)
points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny)])
nnodes = points.shape[0]

tris = []
for j in range(ny - 1):
    for i in range(nx - 1):
        n00 = j * nx + i
        n10 = j * nx + i + 1
        n01 = (j + 1) * nx + i
        n11 = (j + 1) * nx + i + 1
        tris.append([n00, n10, n11])
        tris.append([n00, n11, n01])
tris = np.asarray(tris, dtype=np.int64)

lines = []
for i in range(nx - 1):
    lines.append([i, i + 1])
for i in range(nx - 1):
    lines.append([(ny - 1) * nx + i, (ny - 1) * nx + i + 1])
for j in range(ny - 1):
    lines.append([j * nx, (j + 1) * nx])
for j in range(ny - 1):
    lines.append([j * nx + (nx - 1), (j + 1) * nx + (nx - 1)])
lines = np.asarray(lines, dtype=np.int64)

mesh_path = pathlib.Path('plate.msh')
meshio.write_points_cells(
    str(mesh_path),
    points=points,
    cells=[('line', lines), ('triangle', tris)],
    file_format='gmsh22',
    binary=False,
)
print(f'[mesh] {mesh_path} :: {nnodes} nodes | {len(tris)} tris | {len(lines)} lines')


# --------------------------------------------------------------------------- #
# 2. Material + PD parameters
# --------------------------------------------------------------------------- #
E = 72.0e9
rho = 2440.0
thickness = 0.01

horizon = 3.015 * dx
critical_stretch = 5.0e-4
bond_stiffness = (9.0 * E) / (np.pi * thickness * horizon ** 3)

# PeriPy wants density as a (nnodes,) ndarray, not a scalar.
density = np.full(nnodes, rho, dtype=np.float64)

print(f'[mat]  E={E:.2e} rho={rho} thickness={thickness}')
print(f'[PD]   dx={dx:.4e} horizon={horizon:.4e} bond_k={bond_stiffness:.3e} s0={critical_stretch}')


# --------------------------------------------------------------------------- #
# 3. Pre-crack + BC callables
# --------------------------------------------------------------------------- #
NOTCH_Y = L_y / 2.0
NOTCH_LENGTH = 0.2 * L_x


@initial_crack_helper
def is_crack(icoord, jcoord):
    if (icoord[1] - NOTCH_Y) * (jcoord[1] - NOTCH_Y) > 0.0:
        return False
    x_mid = 0.5 * (icoord[0] + jcoord[0])
    return x_mid <= NOTCH_LENGTH


BC_LAYER = 3.0 * dx   # thicker clamp zone so the loaded layer doesn't self-fracture


def is_displacement_boundary(x):
    bnd = [None, None, None]
    if x[1] > L_y - BC_LAYER:
        bnd[1] = 1
    elif x[1] < BC_LAYER:
        bnd[1] = -1
    return bnd


# --------------------------------------------------------------------------- #
# 4. Build model
# --------------------------------------------------------------------------- #
print('[model] constructing ...')
t0 = time.perf_counter()
integrator = EulerCromerCL(damping=0.0, dt=1.0e-8)
model = Model(
    mesh_file=str(mesh_path),
    integrator=integrator,
    horizon=horizon,
    critical_stretch=critical_stretch,
    bond_stiffness=bond_stiffness,
    dimensions=2,
    density=density,
    initial_crack=is_crack,
    is_displacement_boundary=is_displacement_boundary,
)
print(f'[model] built in {time.perf_counter() - t0:.2f}s  '
      f'nnodes={model.nnodes}  max_neigh={model.max_neighbours}')


# --------------------------------------------------------------------------- #
# 5. Optional: run a short simulation to smoke-test simulate()
# --------------------------------------------------------------------------- #
if RUN_SIMULATE:
    print(f'[sim]  running {N_STEPS} steps ...')
    t0 = time.perf_counter()
    # displacement_bc_magnitudes[i] is the CUMULATIVE displacement applied at
    # step i (not per-step increment). We ramp from 0 up to u_max on each
    # edge. With critical_stretch = 5e-4 and plate height 0.5 m, we need edge
    # displacement > ~1.25e-4 m to exceed critical; ramp to 5e-4 m to drive
    # propagation well past initiation.
    u_max = 3.0e-4
    displacement_bc_magnitudes = np.linspace(0.0, u_max, N_STEPS)
    u, damage, connectivity, force, ud, data = model.simulate(
        steps=N_STEPS,
        displacement_bc_magnitudes=displacement_bc_magnitudes,
        write=WRITE_EVERY,
    )
    elapsed = time.perf_counter() - t0
    print(f'[sim]  done in {elapsed:.2f}s')
    print(f'       u.shape       = {u.shape}')
    print(f'       damage.shape  = {damage.shape}')
    print(f'       force.shape   = {force.shape}')
    print(f'       ud.shape      = {ud.shape}')
    print(f'       damage range  = [{damage.min():.3f}, {damage.max():.3f}]')
    print(f'       broken bonds  = {int((damage > 0.0).sum())} / {damage.size} nodes with damage')

    if SAVE_PLOT:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        coords = model.coords[:, :2]
        scale = 50.0
        deformed = coords + scale * u[:, :2]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        sc0 = axes[0].scatter(coords[:, 0], coords[:, 1], c=damage,
                              cmap='inferno', s=6, vmin=0.0, vmax=1.0)
        axes[0].set_title('Damage field (undeformed)')
        axes[0].set_aspect('equal')
        plt.colorbar(sc0, ax=axes[0], label='damage')

        sc1 = axes[1].scatter(deformed[:, 0], deformed[:, 1], c=u[:, 1],
                              cmap='coolwarm', s=6)
        axes[1].set_title(f'Deformed (x{scale:.0f}), coloured by u_y')
        axes[1].set_aspect('equal')
        plt.colorbar(sc1, ax=axes[1], label='u_y [m]')

        plt.tight_layout()
        out = 'demo_output.png'
        plt.savefig(out, dpi=120)
        print(f'[plot] saved {out}')

print('OK')
