# Physics-Informed Neural Networks for Compressible Flows with Shocks

This repository implements **Physics-Informed Neural Networks (PINNs)** for solving the **compressible Euler equations** in the presence of **shock waves**.

The framework incorporates a **learnable global artificial viscosity model**, enabling **robust, data-free shock capturing** without requiring labeled training data.

The code includes standard **1D and 2D Riemann benchmark problems**, which are widely used to validate numerical methods for compressible flows.

---

## 📂 Repository Structure
```
.
├── 1DRiemann
│   ├── main.py
│   └── src.py
│
├── 2DRiemann
│   ├── main.py
│   └── src.py
│
└── README.md
```
---

## ▶ Running the Code

### 1D Riemann Problems

Navigate to the `1DRiemann` directory and run:

```bash
python main.py --case SST

or

python main.py --case LST

Available cases:
	•	SST → Sod Shock Tube
	•	LST → Lax Shock Tube
```

### 2D Riemann Problems

Navigate to the 2DRiemann directory and run:

```bash
python main.py --case CONFIG1

or

python main.py --case CONFIG2

or

python main.py --case CONFIG3

```
Each case corresponds to a distinct 2D Riemann initial configuration.

📖 Citation

If you use this code in your research, please cite:

@article{kumar2026shocks,
  title = {A Robust Data-Free Physics-Informed Neural Network for Compressible Flows with Shocks}},
  author = {Prashant Kumar and Rajesh Ranjan},
  doi = {https://doi.org/10.1016/j.compfluid.2026.106975},
  url = {https://www.sciencedirect.com/science/article/pii/S0045793026000174},
  year = {2026},
  journal = {Computer & Fluids}
}


