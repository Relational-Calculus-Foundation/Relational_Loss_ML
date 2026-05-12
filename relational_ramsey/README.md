# Ramsey State Equation – A Relational Calculus Framework

**Structural Pressure and Phase Transitions in Complex Networks**

[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.19757717-blue)](https://doi.org/10.5281/zenodo.19757717)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Open Source Collective](https://img.shields.io/badge/OSC-Relational%20Calculus%20Foundation-green)](https://opencollective.com/relational-calculus-foundation)

---

## Overview

This repository contains the complete executable framework behind the paper **“Structural Pressure and Phase Transitions in Ramsey Graphs $R(3,k)$: A Thermodynamic and Geometric Approach”**. We demonstrate that complex networks—whether computational, infrastructural, or financial—obey a universal **Ramsey State Equation** derived from Relational Calculus.

Instead of exhaustive combinatorial enumeration, we reduce network behaviour to two dimensionless parameters:
*   **Structural Pressure** ($P = \frac{m}{2k}$)
*   **Specific Volume** ($V = \frac{2n}{k}$)

These ratios collapse seemingly unrelated systems onto a single phase diagram, revealing harmonic equilibrium points, geometric transitions ($2\pi$), and critical thresholds that predict systemic collapse.

## Repository Contents

```text
.
├── README.md                           # This file
├── paper/
│   └── ramsey_state_equation.pdf       # Full paper (preprint)
├── demo_cybersecurity/
│   ├── cyber_sonar.py                  # Topological anomaly detection on CIC‑IDS‑2017
│   ├── ramsey_phase_diagram.png        # Phase diagram output
│   └── README.md                       # Instructions for reproducing the cybersecurity demo
├── demo_power_grid/
│   ├── blackout_simulator.py           # Cascading failure analysis (2003 Northeast Blackout)
│   ├── blackout_phase_diagram.png      # Phase diagram output
│   └── README.md                       # Instructions for reproducing the power‑grid demo
├── demo_financial/
│   ├── interbank_analyzer.py           # Interbank network stress‑test (2008 crisis)
│   ├── interbank_phase_diagram.png     # Phase diagram output
│   └── README.md                       # Instructions for reproducing the financial demo
└── references/
    └── bibliography.bib                # BibTeX file with all references
```

## Getting Started

### Prerequisites
*   Python 3.8+
*   Required libraries: `pandas`, `numpy`, `matplotlib`, `networkx`

Install the dependencies:

```bash
pip install -r requirements.txt
```

### Running the Demos
Each demo is self‑contained and can be executed independently:

```bash
# Cybersecurity demo (CIC‑IDS‑2017 dataset)
cd demo_cybersecurity
python cyber_sonar.py

# Power grid demo (2003 Northeast Blackout)
cd demo_power_grid
python blackout_simulator.py

# Financial demo (2008 interbank crisis)
cd demo_financial
python interbank_analyzer.py
```

Detailed instructions and data sources are provided in the `README.md` files inside each demo folder.

## Key Results

| Domain | Network Type | Metric | Critical Behaviour |
| :--- | :--- | :--- | :--- |
| **Cybersecurity** | Enterprise traffic | $P$ | Spikes beyond 7.0 during lateral movement attacks |
| **Power Grid** | Transmission lines | $P$ | Drops precipitously minutes before cascading blackout |
| **Financial** | Interbank lending | $P$ | Hysteresis cycle: super-critical bubble $\to$ sub-critical freeze |

All three systems exhibit the same **dimensionless phase diagram** predicted by the Ramsey State Equation, confirming the domain‑agnostic nature of Relational Calculus.

## The Relational Calculus Connection

The framework used throughout this work is **Relational Calculus** — a meta‑mathematical protocol that translates constrained systems into dimensionless, scale‑invariant templates (see Concas 2026).

The three‑step protocol is:
1.  **Identify the intrinsic capacity** (the “North Star”) of the system.
2.  **Anchor all measurements** to that capacity, producing dimensionless ratios.
3.  **Seek the invariant template** that those ratios obey across all scales.

The Structural Pressure $P$ and Specific Volume $V$ are the dimensionless ratios that naturally emerge for any network constrained by connectivity and fragmentation limits. Their behaviour is identical across the three demos presented here.

## Citation

If you use this code or the accompanying paper in your research, please cite:

```bibtex
@article{concas2026ramsey,
  title   = {Structural Pressure and Phase Transitions in {R}amsey Graphs {R}(3,k): A Thermodynamic and Geometric Approach},
  author  = {Concas, Massimiliano},
  journal = {Zenodo},
  year    = {2026},
  doi     = {10.5281/zenodo.XXXXXXXXX}
}
```

For the foundational work on Relational Calculus:

```bibtex
@techreport{concas2026blueprint,
  title  = {The Intrinsic Blueprint: An Introduction to Relational Calculus},
  author = {Concas, Massimiliano},
  year   = {2026},
  doi    = {10.5281/zenodo.19757717}
}
```

## License

This project is licensed under the MIT License – see the [LICENSE](./LICENSE) file for details.

## Acknowledgements

*   **Brendan McKay** for maintaining the public Ramsey graph databases.
*   **CIC‑IDS‑2017 team** for the labelled intrusion detection dataset.
*   **NERC / U.S.–Canada Power System Outage Task Force** for the 2003 blackout data.
*   The **Open Source Collective** for fiscal hosting of the Relational Calculus Foundation.

*Relational Calculus Foundation – Democratizing computational efficiency through dimensionless geometry.*
