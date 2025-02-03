# HHL Algorithm: Quantum Linear System Solver

This repository contains an implementation and detailed explanation of the HHL algorithm (Harrow-Hassidim-Lloyd), a quantum method designed to solve systems of linear equations of the form \( A\vec{x} = \vec{b} \). The HHL algorithm provides an exponential speedup over classical methods under certain conditions, such as matrix sparsity and a low condition number.
## Contents

- **Introduction to the HHL Algorithm**: Theoretical description and applications.
- **Implementation**: Detailed code using [Qiskit](https://qiskit.org/) to simulate the algorithm in a quantum environment.
- **Examples**: Practical cases demonstrating the algorithm's application to different linear systems.
- **References**: Links to additional resources and recommended readings.

## Requirements

- Python.
- Required libraries:
  - Qiskit
  - NumPy
  - Matplotlib

It is recommended to use a virtual environment to manage dependencies.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DanielCondeTorres/HHL.git
   vi hhl_exercise_DCT.py
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install module
   ```

## Usage

The main file of the project is `hhl_exercise_DCT.py`, an script that includes:

- Theoretical explanations of the HHL algorithm.
- Step-by-step implementation in Qiskit.
- Visualizations and result analysis.

To run the script:

1. Start a terminal:
   ```bash
   python hhl_exercise_DCT.py
   ```


## Experimentation with HHL

### Simulating Different Qubit Numbers

Set `num_state_qubits` to be the qubit number \(n\). In this case, the function will output a $(2^n \times 2^n)$ matrix. Perform simulations for qubit numbers $(n \leq 5)$. 

**Question:** What is the maximum number of qubits \(M\) that you can simulate on your machine? State the model of your PC's processor and, if available, the graphics card or GPU.

### Using Qiskit HHL Class

Use the Qiskit HHL class documented in the Jupyter notebook `hhl_tutorial.ipynb` to provide explicit quantum circuits that prepare a quantum solution for the linear system. To achieve this, use the `state` property of the `LinearSolverResult` object returned by `HHL.solve()` and the `print` command.

### Estimating Properties for HHL Solutions

Use the HHL quantum algorithm to estimate properties of the HHL solutions for different systems of equations. Choose a system \(Ax = b\) defined by a tridiagonal matrix \(A\) that you can solve using HHL with at most \(M \leq 5\) qubits. Generate it using the `TridiagonalToeplitz` class:

```python
TridiagonalToeplitz(num_state_qubits, main_diag, off_diag, tolerance=0.01, evolution_time=1.0, trotter_steps=1, name='tridi')
```

[Documentation](https://docs.quantum.ibm.com/api/qiskit/0.40/qiskit.algorithms.linear_solvers.TridiagonalToeplitz)

Solve the system for your chosen matrix and report the following properties of the solutions:

1. **Vector norm** $(|x|)$ of the solution vector (use the `euclidean_norm` property).
2. **Average of the vector entries**.
3. **Inner product** $(\langle x | B | x \rangle)$ where $(B)$ is a Hermitian observable of your choice.
4. **Verification**: Use the classical NumPy solver `linalg.LinearSolverResult` to verify your solutions.

### Influence of Matrix Entry Size on Performance

Study the influence of performance concerning the size of the matrix entries for tridiagonal matrices. Use different ranges for matrix entries:

1. \(a, b\) are randomly chosen real numbers in the interval \([-1, 1]\).
2. \(a, b\) are randomly chosen \(\log(n)\)-digit numbers in the interval \([-n, n]\).
3. \(a, b\) are randomly chosen \(n\)-digit numbers in \([-2n, 2n]\).

**Question:** What is the influence of the size of the range on the cost of running HHL?

