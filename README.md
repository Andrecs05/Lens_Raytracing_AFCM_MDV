# Lens Ray Tracing with Chromatic Aberration Simulation

A Python-based optical ray tracing simulator for analyzing chromatic aberration effects in telescope systems. This project implements matrix-based optics to model simple and achromatic telescope designs, demonstrating how different wavelengths of light focus at different points due to material dispersion.

## 🔭 Overview

This project simulates the optical behavior of telescope systems by:
- Modeling refractive index dispersion in optical glasses (NBK7, NSF2, NBAF10, NSF6HT)
- Calculating ray transfer matrices for thick lenses and doublet systems
- Simulating chromatic aberration effects on astronomical images
- Comparing simple telescope designs with achromatic doublet configurations

## 🚀 Features

- **Matrix-based Ray Tracing**: Uses ABCD matrix formalism for optical system modeling
- **Chromatic Aberration Simulation**: Separates RGB channels to demonstrate wavelength-dependent focusing
- **Multiple Glass Types**: Implements Sellmeier equations for various optical glasses
- **Image Processing**: Processes real astronomical images through simulated optical systems
- **Comparative Analysis**: Includes both simple and achromatic telescope designs
- **Visualization**: Generates before/after comparisons and channel-separated outputs

## 📁 Project Structure

```
lens_raytracing_AFCM_MDV/
├── src/                          # Core simulation modules
│   ├── matrix_formation.py       # Optical matrix calculations
│   ├── refractive_idxs.py        # Sellmeier equations for glass materials
│   └── utilities.py              # Telescope simulation functions
├── notebooks/                    # Jupyter notebooks for analysis
│   ├── simple_telescope.ipynb    # Basic telescope simulation
│   └── achromat_telescope.ipynb  # Achromatic doublet simulation
├── assets/                       # Sample astronomical images
│   ├── PIA01464.jpg             # Jupiter image
│   ├── PIA02270.jpg             # Saturn image
│   └── moon_SC_2016_07_15-1.jpg # Moon image
├── results/                      # Output images
└── scripts/                      # Utility scripts
```

## 🛠️ Installation

### Prerequisites
- Python 3.13+
- UV package manager (recommended) or pip

### Using UV (Recommended)
```bash
# Clone the repository
git clone https://github.com/Andrecs05/lens_raytracing_AFCM_MDV.git
cd lens_raytracing_AFCM_MDV

# Install dependencies
uv sync
```

### Using pip
```bash
pip install -r requirements.txt
```

### Required Dependencies
- `numpy>=2.3.2` - Numerical computations
- `opencv-python>=4.11.0.86` - Image processing
- `matplotlib>=3.10.5` - Plotting and visualization
- `scipy>=1.16.1` - Scientific computing and interpolation
- `scikit-image>=0.25.2` - Image analysis (SSIM calculations)
- `pandas>=2.3.2` - Data manipulation

## 📊 Usage

### Quick Start with Jupyter Notebooks

1. **Simple Telescope Analysis**:
   ```bash
   jupyter notebook notebooks/simple_telescope.ipynb
   ```
   - Demonstrates chromatic aberration in a basic two-lens telescope
   - Shows RGB channel separation effects
   - Calculates focal length variations across wavelengths

2. **Achromatic Telescope Analysis**:
   ```bash
   jupyter notebook notebooks/achromat_telescope.ipynb
   ```
   - Models doublet lens systems for chromatic aberration correction
   - Compares multiple glass combinations
   - Demonstrates improved color correction

### Python API Usage

```python
from src.utilities import get_parameters, telescope
from src.refractive_idxs import refractive_index_NBK7
import cv2
import numpy as np

# Load an astronomical image
img = cv2.imread("assets/PIA01464.jpg")
R, G, B = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Define telescope parameters
n1 = 1.0  # Air
R1_o, R2_o = 308.2, -308.2  # Objective lens radii (mm)
R1_e, R2_e = -52.1, 52.1    # Eyepiece lens radii (mm)
d_o, d_e = 5.1, 3.5          # Lens thicknesses (mm)

# Calculate refractive indices for different wavelengths
nR = refractive_index_NBK7(0.700)  # Red
nG = refractive_index_NBK7(0.550)  # Green  
nB = refractive_index_NBK7(0.435)  # Blue

# Get telescope parameters for each channel
params_R = get_parameters(R, n1, nR, R1_o, R2_o, d_o, R1_e, R2_e, d_e)
params_G = get_parameters(G, n1, nG, R1_o, R2_o, d_o, R1_e, R2_e, d_e)
params_B = get_parameters(B, n1, nB, R1_o, R2_o, d_o, R1_e, R2_e, d_e)

# Simulate telescope for each color channel
final_R = telescope(R, *params_R)
final_G = telescope(G, *params_G)
final_B = telescope(B, *params_B)
```

## 🧮 Technical Details

### Optical Matrix Formalism

The project uses 2×2 ABCD matrices to model optical elements:

- **Thick Lens**: Models refractive surfaces with finite thickness
- **Translation**: Propagation through homogeneous media  
- **Doublet**: Combination of two lens elements for aberration correction

### Refractive Index Models

Implements Sellmeier equations for accurate dispersion modeling:
- **NBK7**: Standard crown glass
- **NSF2**: Dense flint glass  
- **NBAF10**: Barium crown glass
- **NSF6HT**: High-index flint glass

### Chromatic Aberration Analysis

The simulation separates white light into RGB components and traces each wavelength independently, revealing:
- Focal length variations (longitudinal chromatic aberration)
- Image magnification differences  
- Color fringing effects in the final image

## 📈 Sample Results

The simulation produces:
- **Original vs. Processed Images**: Side-by-side comparisons showing aberration effects
- **Channel Separation**: Individual R, G, B channel outputs highlighting chromatic dispersion
- **Quantitative Metrics**: SSIM values for color channel correlation analysis
- **Focal Length Analysis**: Numerical data on chromatic aberration magnitude

## 🎓 Educational Applications

This project is ideal for:
- **Optics Courses**: Demonstrating fundamental ray tracing principles
- **Astronomy Education**: Understanding telescope design trade-offs
- **Computer Vision**: Image processing and geometric transformations

## 📝 License

This project is part of academic research in optics. Please cite appropriately if used in academic work.

## 📚 References

- Hecht, E. (2016). *Optics* (5th Edition)
- Sellmeier coefficients from optical glass manufacturer datasheets
