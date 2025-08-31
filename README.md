# FlexAf: Differentiable Feature Selection and Training for Hardware-Aware Neural Networks

**FlexAf** is a framework for **differentiable feature selection** and **hardware-aware neural network training**, designed for mixed-signal, resource-constrained, and flexible hardware platforms.  
It integrates **stochastic feature gating**, **cost-aware regularization**, and optional **lottery ticket pruning** to jointly optimize classification accuracy and feature extraction costs.

---

## 🔧 Key Features

- 🔬 **Differentiable Feature Selection**  
  Embedded stochastic gating layer enables end-to-end feature selection during training.

- 💡 **Cost-Aware Regularization**  
  Incorporates per-feature hardware area costs into the optimization objective.

- 🧩 **Stochastic Gating Layer**  
  Gumbel–Sigmoid gates enable gradient-based optimization and interpretable feature importance.

- ✂️ **Feature Pruning**  
  Automatic pruning via thresholding on learned gate probabilities.

- 🏹 **Lottery Ticket Pruning**  
  Optional pruning-aware retraining to further reduce model complexity.

- ⚙️ **Hardware-Aware Optimization**  
  Look-Up Table (LUT)-based hardware cost modeling for analog feature extraction.

- 🔗 **End-to-End Pipeline**  
  Supports flexible, mixed-signal wearables for healthcare at the extreme edge. All MLP designs are **fully evaluated** in hardware using EDA tools (e.g., Synopsys suite).

---

## 📦 Technologies Used

- Python 3  
- PyTorch (for differentiable gating and training)  
- Gumbel–Sigmoid distributions for stochastic gating  
- Conda environment management  
- YAML-based experiment configuration  

---

## ⚙️ Installation

Set up the environment using Conda:

```bash
# Clone the repository
git clone https://github.com/your-username/flexaf.git
cd flexaf

# Create and activate the conda environment
conda env create -f env.yml
conda activate flexaf
```

## 🚀 Usage

The main entry point is ```main.py```.
All experiment parameters are defined in run/args.yaml, which controls:
- Training hyperparameters
- Gating and cost-regularization settings
- Hardware cost constraints
- Pruning thresholds
 
To run the default experiment use:
```bash
./run/main.sh
```
To run differentiable feature selection optimization
```bash
./run/gates.sh
```
To run lottery ticket pruning and retraining
```bash
./run/lottery_ticket.sh
```

## 📚 Citation

To cite this work in bibtex:
```bibtex
@inproceedings{Shatta:ICCAD2025:Invited:analogFEx,
    author    = {Maha Shatta and Konstantinos Balaskas and Paula Carolina Lozano Duarte and Georgios Panagopoulos and Mehdi B. Tahoori and Georgios Zervakis},
    title     = {Invited Paper: Feature-to-Classifier Co-Design for Mixed-Signal Smart Flexible Wearables for Healthcare at the Extreme Edge},
    booktitle = {Proceedings of the IEEE/ACM International Conference on Computer-Aided Design (ICCAD)},
    year      = {2025},
    month     = oct
}
```

## 🙏 Acknowledgment
This work is partially supported by:
- European Research Council (ERC) (Grant No. 101052764)
- H.F.R.I. “Basic Research Financing (Horizontal support of all Sciences)” under the National Recovery and Resilience Plan “Greece 2.0” (Project No. 17048)
