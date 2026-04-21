"""Standalone driver for an *elastic* PeriPy demo — the companion to the
fracture case. Same plate geometry, but:

  - no initial crack
  - load is ramped well below the critical stretch (so no bonds break)
  - we use dynamic-relaxation damping so the plate settles into a
    quasi-static equilibrium instead of ringing forever

At the end we compare the simulated stress response to the analytical
prediction sigma = E * eps, so we can verify PeriPy recovers linear elasticity
in the safe-load regime.

Run:
    conda activate peripy_env
    python test_elastic.py
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
import meshio

from peripy.model import Model
from peripy.integrators import EulerCromerCL


# --------------------------------------------------------------------------- #
# Knobs
# --------------------------------------------------------------------------- #
RUN_SIMULATE = True
N_STEPS = 4000           # longer than fracture run - dynamic relaxation needs
                         # time to damp out waves
WRITE_EVERY = 500
SAVE_PLOT = True


# --------------------------------------------------------------------------- #
# 1. Mesh  (identical to fracture demo so results are comparable)
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

mesh_path = pathlib.Path('plate_elastic.msh')
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

density = np.full(nnodes, rho, dtype=np.float64)

# Pick a peak edge displacement that stays comfortably below critical.
# With L_y = 0.5 m and critical_stretch = 5e-4, critical edge disp per side is
# about 1.25e-4 m. Use ~25% of that so we're firmly in the elastic regime.
U_MAX = 3.0e-5          # m (per edge, cumulative at final step)
eps_applied = 2.0 * U_MAX / L_y  # total imposed axial strain

print(f'[mat]  E={E:.2e} rho={rho} thickness={thickness}')
print(f'[PD]   dx={dx:.4e} horizon={horizon:.4e} bond_k={bond_stiffness:.3e} s0={critical_stretch}')
print(f'[load] u_max/edge={U_MAX:.2e} m -> strain={eps_applied:.2e} ({eps_applied/critical_stretch:.1%} of critical)')


# --------------------------------------------------------------------------- #
# 3. Boundary conditions — no pre-crack, just uniaxial tension
# --------------------------------------------------------------------------- #
BC_LAYER = 3.0 * dx


def is_displacement_boundary(x):
    """Top edge pulled in +y, bottom edge pulled in -y. Middle is free."""
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

# Dynamic relaxation damping — rule of thumb for a 2D PD plate is
# eta ~ 2 * sqrt(rho * c_max) where c_max ~ bond_stiffness * horizon volume.
# A value in the same order as rho * wave_speed / horizon works well in practice.
wave_speed = np.sqrt(E / rho)
damping = 2.0 * rho * wave_speed / horizon   # ~ 7e8 kg/(m^3 s)
print(f'[damp] damping = {damping:.3e} kg/(m^3 s)  (quasi-static relaxation)')

integrator = EulerCromerCL(damping=damping, dt=1.0e-8)
model = Model(
    mesh_file=str(mesh_path),
    integrator=integrator,
    horizon=horizon,
    critical_stretch=critical_stretch,
    bond_stiffness=bond_stiffness,
    dimensions=2,
    density=density,
    is_displacement_boundary=is_displacement_boundary,
    # no initial_crack here - fully intact elastic body
)
print(f'[model] built in {time.perf_counter() - t0:.2f}s  '
      f'nnodes={model.nnodes}  max_neigh={model.max_neighbours}')


# --------------------------------------------------------------------------- #
# 5. Run
# --------------------------------------------------------------------------- #
if RUN_SIMULATE:
    print(f'[sim]  running {N_STEPS} steps ...')
    t0 = time.perf_counter()

    # Ramp cumulative edge displacement up to U_MAX and then hold, so the
    # dynamic damping can settle the plate.
    hold_start = N_STEPS // 2
    ramp = np.linspace(0.0, U_MAX, hold_start)
    hold = np.full(N_STEPS - hold_start, U_MAX)
    displacement_bc_magnitudes = np.concatenate([ramp, hold])

    u, damage, connectivity, force, ud, data = model.simulate(
        steps=N_STEPS,
        displacement_bc_magnitudes=displacement_bc_magnitudes,
        write=WRITE_EVERY,
    )
    elapsed = time.perf_counter() - t0
    print(f'[sim]  done in {elapsed:.2f}s')
    print(f'       u.shape       = {u.shape}')
    print(f'       damage range  = [{damage.min():.3e}, {damage.max():.3e}]')
    print(f'       damaged nodes = {int((damage > 0.0).sum())} / {damage.size}')

    # ---- Validation: measured vs analytical axial strain ----
    coords = model.coords[:, :2]
    # Pick a strip in the middle of the plate and fit a linear variation
    # u_y(y) = eps * y + const.
    middle_mask = (np.abs(coords[:, 0] - L_x / 2.0) < 0.05 * L_x)
    y_mid = coords[middle_mask, 1]
    uy_mid = u[middle_mask, 1]
    # Least-squares slope
    A = np.vstack([y_mid, np.ones_like(y_mid)]).T
    slope, intercept = np.linalg.lstsq(A, uy_mid, rcond=None)[0]

    sigma_analytical = E * eps_applied           # Pa
    print('\n[validate] uniaxial tension')
    print(f'  applied total strain (top-bottom opening / L_y) = {eps_applied:.3e}')
    print(f'  measured axial strain (du_y/dy, midplane fit)    = {slope:.3e}')
    print(f'  relative error = {100 * (slope - eps_applied) / eps_applied:+.2f} %')
    print(f'  predicted stress sigma = E * eps                 = {sigma_analytical:.3e} Pa')

    # ---- Plot ----
    if SAVE_PLOT:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        scale = 500.0  # displacements are tiny in the elastic regime
        deformed = coords + scale * u[:, :2]

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.0))

        sc0 = axes[0].scatter(coords[:, 0], coords[:, 1], c=u[:, 1],
                              cmap='coolwarm', s=6)
        axes[0].set_title('u_y (undeformed)')
        axes[0].set_xlabel('x [m]'); axes[0].set_ylabel('y [m]')
        axes[0].set_aspect('equal')
        plt.colorbar(sc0, ax=axes[0], label='u_y [m]')

        sc1 = axes[1].scatter(deformed[:, 0], deformed[:, 1], c=u[:, 1],
                              cmap='coolwarm', s=6)
        axes[1].set_title(f'Deformed (x{scale:.0f})')
        axes[1].set_xlabel('x [m]'); axes[1].set_ylabel('y [m]')
        axes[1].set_aspect('equal')
        plt.colorbar(sc1, ax=axes[1], label='u_y [m]')

        # u_y profile along mid-width vs linear prediction
        ys_plot = np.linspace(0, L_y, 100)
        uy_lin = eps_applied * (ys_plot - L_y / 2.0)
        axes[2].plot(uy_mid * 1e6, y_mid, 'o', ms=3, alpha=0.5,
                     label='PeriPy (midplane)')
        axes[2].plot(uy_lin * 1e6, ys_plot, 'k--', lw=1.5,
                     label=f'E*eps*y  (applied eps={eps_applied:.1e})')
        axes[2].set_xlabel('u_y [μm]'); axes[2].set_ylabel('y [m]')
        axes[2].set_title('Axial u_y profile')
        axes[2].legend(loc='best'); axes[2].grid(alpha=0.3)

        plt.tight_layout()
        out = 'elastic_output.png'
        plt.savefig(out, dpi=120)
        print(f'[plot] saved {out}')

print('OK')
