# Year 2 Computing Project — 3D Optical Ray Tracer

A Python-based 3D optical ray tracer developed as part of a Year 2 computing project. The package models the propagation of light rays through simple optical systems, including refracting and reflecting surfaces and compound lenses.

## Features

* Trace individual rays through 3D space
* Generate and propagate bundles of rays
* Model spherical and planar optical surfaces
* Simulate refraction and reflection
* Model plano-convex and bi-convex lenses
* Calculate focal points and optical behaviour
* Visualise ray paths and analyse optical systems
* Automated testing using `pytest`

## Requirements

The project requires Python **3.9 or later** and uses:

* NumPy
* SciPy
* Matplotlib
* pytest

## Installation

Clone the repository:

```bash
git clone https://github.com/Harschul/Year-2-Computing-Project.git
cd Year-2-Computing-Project
```

Install the package and its dependencies:

```bash
pip install -e .
```

Alternatively, dependencies can be installed from:

```bash
pip install -r requirements.txt
```

## Example

```python
from raytracer.rays import RayBundle
from raytracer.lenses import PlanoConvex
from raytracer.elements import OutputPlane

# Create a bundle of parallel rays
rays = RayBundle(rmax=5, nrings=5, multi=6)

# Define an optical system
lens = PlanoConvex(
    z_0=100,
    curvature=0.02,
    thickness=5,
    aperture=50
)

screen = OutputPlane(z_0=200)

# Trace the rays through the system
rays.propagate_bundle([lens, screen])

# Plot the resulting ray paths
rays.track_plot()
```

## Project Structure

```text
raytracer/
├── analysis.py    # Optical analysis and plotting tools
├── elements.py    # Optical surfaces and output planes
├── lenses.py      # Compound lens models
├── physics.py     # Reflection and refraction calculations
├── rays.py        # Ray and RayBundle classes
└── genpolar.py    # Ray-bundle position generation

tests/             # Automated test suite
plots_for_marking/ # Generated project plots
```

## Testing

Run the test suite from the project root with:

```bash
pytest
```

