# PDCLP — PeriPy Demo Notebooks

Two companion Jupyter notebooks built on top of the
[PeriPy](https://github.com/alan-turing-institute/PeriPy) peridynamics
library. The two demos share the same plate geometry and material, so you
can swap one for the other and see exactly where the elastic regime ends
and brittle fracture takes over.

## Contents

| File | What it shows |
|---|---|
| `peripy_fracture_demo.ipynb` | 2-D mode-I fracture: notched plate loaded past critical stretch; crack propagates across the plate. |
| `peripy_elastic_demo.ipynb`  | Same plate, sub-critical load: pure elastic deformation, validated against linear elasticity. |
| `test_demo.py`               | Fast-iteration standalone driver for the fracture case. |
| `test_elastic.py`            | Fast-iteration standalone driver for the elastic case. |
| `requirements.txt`           | Python dependencies. |

Both notebooks:
1. Generate a triangular mesh of a 1.0 m × 0.5 m plate with boundary `line`
   elements (required by PeriPy's mesh loader).
2. Define material and peridynamic parameters.
3. Build a PeriPy `Model` with the `EulerCromerCL` dynamic integrator.
4. Run the simulation.
5. Visualise the result with matplotlib.

The **elastic demo** additionally compares the measured mid-plane axial
strain to the analytical prediction `σ = E · ε` to validate PeriPy's
linear-elastic response.

## Quick start

```bash
# 1. Clone this repo
git clone https://github.com/nbolander/PDCLP.git
cd PDCLP

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install numpy scipy matplotlib meshio h5py tqdm scikit-learn cython pyopencl jupyter

# 4. Install PeriPy from source (more reliable than PyPI on Windows)
git clone https://github.com/alan-turing-institute/PeriPy.git
cd PeriPy
pip install -e .
cd ..

# 5. Launch Jupyter
jupyter lab peripy_fracture_demo.ipynb   # or peripy_elastic_demo.ipynb
```

## Fast iteration

If you're debugging the PeriPy API and don't want the JupyterLab
reload-rerun cycle, just run the standalone scripts:

```bash
python test_demo.py       # fracture
python test_elastic.py    # elastic + validation against sigma = E*eps
```

They write `demo_output.png` / `elastic_output.png` for visual inspection.

## Requirements

- Python 3.9+ (Python 3.10 tested; IPython 8.x pulls traitlets ≥ 5.14 —
  on Python 3.8 environments this fails with a cryptic
  `'type' object is not subscriptable` error at kernel start-up).
- An OpenCL runtime (Intel OpenCL Runtime, POCL, or a GPU vendor driver).
- C/C++ build tools (MSVC Build Tools on Windows, GCC on Linux) to compile
  PeriPy's Cython extensions.

## Physics summary

The bond-based 2-D plane-stress micro-modulus used in both notebooks is

```
c = 9 · E / (π · t · δ³)
```

where `t` is plate thickness and `δ` is the horizon. Bonds break
irreversibly once their stretch exceeds `s_0 = 5·10⁻⁴`. For glass-like
parameters (`E = 72 GPa`, `ρ = 2440 kg/m³`, plate height `L_y = 0.5 m`),
the critical edge displacement per side is `L_y/2 · s_0 = 1.25·10⁻⁴ m`.
- The **fracture** demo ramps the load to `3·10⁻⁴ m` (2.4× critical).
- The **elastic** demo ramps only to `3·10⁻⁵ m` (0.24× critical) and
  holds, so dynamic-relaxation damping can settle the plate into an
  elastostatic equilibrium.

## License

The PeriPy library is distributed under the MIT License. Content in this
repo is provided as-is for educational purposes.
