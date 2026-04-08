# PDCLP — PeriPy Fracture Demo

A minimal Jupyter notebook demonstrating a 2-D mode-I fracture simulation
built on top of the [PeriPy](https://github.com/alan-turing-institute/PeriPy)
peridynamics library.

## Contents

- `peripy_fracture_demo.ipynb` — self-contained notebook that:
  1. Generates a triangular mesh of a rectangular plate.
  2. Writes it as a Gmsh 2.2 `.msh` file for PeriPy to consume.
  3. Defines a horizontal pre-notch, material properties, and displacement
     boundary conditions.
  4. Runs a bond-based peridynamics simulation with PeriPy's OpenCL Euler
     integrator.
  5. Plots the final damage field and deformed configuration with
     matplotlib.
- `requirements.txt` — Python dependencies.

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
jupyter notebook peripy_fracture_demo.ipynb
```

## Requirements

- Python 3.8+ (Python 3.10 recommended)
- An OpenCL runtime installed on your machine (Intel OpenCL Runtime, POCL,
  or a GPU vendor driver). PeriPy uses OpenCL for its fast integrators.
- C/C++ build tools (MSVC Build Tools on Windows, GCC on Linux) to compile
  PeriPy's Cython extensions.

## Notes on the physics

The notebook models a brittle, glass-like plate (`E = 72 GPa`,
`rho = 2440 kg/m^3`, `s_0 = 5e-4`) with a horizontal notch on the left
edge. Loading is mode-I through prescribed displacements on the top and
bottom edges. The bond-based micro-modulus formula used for the 2-D case is

```
c = 9 E / (pi * t * delta^3)
```

where `t` is plate thickness and `delta` is the horizon.

## License

The PeriPy library is distributed under the MIT License. Content in this
repo is provided as-is for educational purposes.
