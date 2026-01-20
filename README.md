Physics-Informed Neural Networks for Compressible Flows with Shocks

This repository implements Physics-Informed Neural Networks (PINNs) for solving the compressible Euler equations in the presence of shocks.
The framework incorporates a learnable global artificial viscosity model, enabling robust, data-free shock capturing without the need for labeled data.

The code includes standard 1D and 2D Riemann benchmark problems, widely used to validate numerical methods for compressible flows.

⸻

📂 Repository Structure
.
├── 1DRiemann/
│   ├── src.py        # PINN model, PDE residuals, loss functions
│   ├── main.py       # Driver script for 1D Riemann problems
│
├── 2DRiemann/
│   ├── src.py        # PINN model and PDE formulation for 2D problems
│   └── main.py       # Driver script for 2D Riemann problems

⸻

Running the Code

▶ 1D Riemann Problems

Navigate to the 1DRiemann directory and run:

python main.py --case SST

or

python main.py --case LST

where:
	•	SST → Sod Shock Tube
	•	LST → Lax Shock Tube

⸻

▶ 2D Riemann Problems

Navigate to the 2DRiemann directory and run:

python main.py --case CONFIG1

or

python main.py --case CONFIG2

or

python main.py --case CONFIG3

Each case corresponds to a distinct 2D Riemann initial configuration.

⸻

📖 Citation

If you use this code in your research, please cite:

@article{kumar2026shocks,
  title   = {A Robust Data-Free Physics-Informed Neural Network for Compressible Flows with Shocks},
  author  = {Kumar, Prashant and Ranjan, Rajesh},
  journal = {Computers \& Fluids},
  year    = {2026}
}

