# 📦 jmstate

**jmstate** is a Python package for **multi-state nonlinear joint modeling**.  
It leverages **PyTorch** for automatic differentiation and vectorized computation, making it efficient and scalable. The package provides a flexible framework where you can use **neural networks as regression and link functions**, while still offering simpler built-in options like parametric baseline hazards.

With **jmstate**, you can model longitudinal data jointly with multi-state transitions (e.g. health progression), capture nonlinear effects, and perform inference in complex real-world settings.

---

## ✨ Features

- **Multi-State Joint Modeling**  
  Supports subjects moving through multiple states with transition intensities that depend on longitudinal trajectories and covariates.

- **Nonlinear Flexibility**  
  Use neural networks (or any PyTorch model) as regression or link functions.

- **Built-in Tools**  
  Includes default baseline hazards, regression, link functions, and analysis utilities.

- **Automatic Differentiation & GPU Support**  
  Powered by PyTorch for efficient gradient computation and vectorization.

- **Analysis & Visualization**  
  Tools for state occupancy probabilities, hazard estimation, and residual diagnostics.

---

## 🚀 Installation

```bash
pip install jmstate
```

---

## 📖 Learn More

For tutorials, API reference, visit the official site:  
👉 [jmstate Documentation](https://felixlaplante0.gitlab.io/jmstate)
