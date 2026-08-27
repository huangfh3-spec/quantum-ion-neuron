# Programmable memristive quantum spiking
dynamics in a trapped ion

This repository contains the code and data required to reproduce the figures and results reported in the paper:

**“Programmable memristive quantum spiking
dynamics in a trapped ion.”**

In this work, a  quantum neuron is implemented using a trapped-ion platform.
By engineering dissipation and gain channels, the system exhibits diverse neuronal firing patterns such as phasic, bursting, and adaptive responses.
The nonlinear internal dynamics of this quantum neuron further enable it to perform nonlinear classification tasks such as XOR.

This repository provides all scripts and datasets necessary to reproduce the figures in the manuscript.

---

# Repository Structure

```
quantum-spiking-neuron/
│
├── README.md
├── environment.yml
├── paths.py
├── run_all_figures.py
│
├── scripts/                # Python scripts that generate figures
│   ├── fig2A.py
│   ├── fig2BandfigS5.py
│   ├── fig2C.py
│   ├── fig2D.py
│   ├── fig3A.py
│   ├── fig3B.py
│   ├── fig3C.py
│   ├── fig3D.py
│   ├── fig4AB.py
│   ├── fig4C1.py
│   ├── fig4C2.py
│   ├── fig4C3.py
│   ├── fig4C4.py
│   ├── figS1.py
│   ├── figS2A.py
│   ├── figS2B.py
│   ├── figS2C.py
│   ├── figS2D.py
│   ├── figS3A.py
│   ├── figS3B.py
│   ├── figS3C.py
│   ├── figS3D.py
│   ├── figS4.py
│   └── figS6.py
├── data/                   # Experimental and simulation data
│   ├── S_x.csv
│   ├── S_y.csv
│   ├── adaptive_population.xls
│   ├── bursting_population.xls
│   ├── High_High_data.xls
│   ├── High_Low_data.xls
│   ├── Low_High_data.xls
│   ├── Low_Low_data.xls
│   ├── I_high_waveform.xls
│   ├── I_low_waveform.xls
│   ├── I_q_exp.xls
│   └── I_q_sim.xls
│
└── figures/                # Generated output figures
    ├── Fig2A.PDF
    ├── Fig2B.PDF
    └── ...
```

---

# Environment Setup

Create the conda environment required to run the code:

```
conda env create -f environment.yml
conda activate qutip_repro
```

The environment includes the following main dependencies:

* Python
* NumPy
* SciPy
* Matplotlib
* Pandas
* QuTiP

---

# Reproducing the Figures

All figures can be reproduced automatically using:

```
python run_all_figures.py
```

This script sequentially executes all figure-generation scripts located in the `scripts/` directory.

---

# Reproducing Individual Figures

You can also generate figures individually.

For example:

### Fig.2B and Fig.S5

```
python scripts/fig2BandfigS5.py
```

### Fig.3A

```
python scripts/fig3A.py
```

### Supplementary Fig.S2A

```
python scripts/figS2A.py
```


---

# Path Management

All scripts use the shared configuration file:

```
paths.py
```

This file defines the main project directories:

* `DATA` — path to the `data/` directory


Each script automatically locates the project root directory, ensuring that the code runs correctly regardless of the current working directory.

---

# Data

The folder `data/` contains the experimental and simulation data used to generate the figures in the manuscript.

These datasets include:

* population measurements
* simulation results
* S_x and S_y expectation values
* robustness_analysis

All scripts read the corresponding datasets directly from this directory.

---

# Citation

If you use this code or data in your research, please cite the associated paper:

```
@article{Huang2026QuantumNeuron,
title = {Programmable memristive quantum spiking
dynamics in a trapped ion},
author = {Huang, Fuhua and ...},
journal = {...},
year = {2026}
}
```

---

# License

This project is released under the MIT License.
