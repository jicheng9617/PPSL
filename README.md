# Parametric Pareto Set Learning (PPSL)

[![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python)](https://www.python.org/downloads/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/license/MIT)

Official implementation of the paper [**"Parametric Pareto Set Learning: Amortizing Multi-Objective Optimization with Parameters"**](https://ieeexplore.ieee.org/document/11236473), published in IEEE Transactions on Evolutionary Computation.

## 🎯 Overview

PPSL addresses the challenge of solving an infinite number of multi-objective optimization problems where optimal solutions must adapt to varying parameters. Unlike traditional methods that generate only finite solution sets, PPSL learns a unified mapping from parameters to the entire Pareto set, enabling real-time inference of optimal solutions for any parameter setting.

### Key Features

- 🚀 **Amortized Optimization**: Shifts computational burden from online solving to offline training
- 🧠 **Hypernetwork Architecture**: Generates PS model parameters conditioned on input parameters
- ⚡ **Low-Rank Adaptation (LoRA)**: Achieves computational efficiency and scalability
- 🎯 **Continuous Pareto Set Learning**: Captures the entire Pareto set structure across parameter space
- 🔄 **Real-time Inference**: Millisecond-level solution generation after training

### Applications

1. **Dynamic Multiobjective Optimization Problems (DMOPs)**: Where objectives change over time
2. **Multiobjective Optimization with Shared Components**: Where design variables must share identical settings for manufacturing efficiency

## 🛠️ Installation

#### Prerequisites
* Python ≥ 3.8  
* PyTorch ≥ 1.12 (GPU supported; CUDA optional)  

#### Setup
1. Clone the repository:
```bash
git clone (available upon acceptance)
cd ppsl
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

#### Required Packages
```apache
torch>=1.12.0
numpy>=1.21.0
pymoo>=0.6.0
matplotlib>=3.5.0
scipy>=1.7.0
```

## 🚀 Quick Start
#### Interactive Demo
Explore PPSL through our Jupyter notebook:
```bash
jupyter notebook example.ipynb
```

## 📁 Repository Structure
```clean
ppsl/
├── experiment_dmop.py      # Experiments for Dynamic MOPs
├── experiment_mopsc.py     # Experiments for MOPs with Shared Components
├── trainer.py              # Training methods (fixed/random/black-box)
├── model.py                # Hypernetwork and PS models (LoRA/non-LoRA)
├── example.ipynb           # Interactive demonstration
├── problems/               # Problem definitions
│   ├── problem_dyn.py      # Dynamic MOP benchmarks
│   └── problem_f_re.py     # RE problems
├── results/                # Experimental results
└── requirements.txt        # Package dependencies
```

## 🎓 Citation
If you find this work useful, please cite our paper:
```bibtex
@article{ppsl2024,
  title={Parametric Pareto Set Learning: Amortizing Multi-Objective Optimization with Parameters},
}
```

## 🙏 Acknowledgments
- Built on [PyMOO](https://pymoo.org/) framework
- Inspired by recent advances in amortized optimization and [Pareto set learning](https://github.com/Xi-L/EPSL)
