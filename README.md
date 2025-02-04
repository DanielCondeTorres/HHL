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
#### Code
The first step is to load the important libraries
 ```
import numpy as np
# pylint: disable=line-too-long
from qiskit.algorithms import NumPyLinearSolver
from qiskit.algorithms.linear_solvers.hhl import HHL
from qiskit.algorithms.linear_solvers.matrices import TridiagonalToeplitz
# %% [code]
import numpy as np
import matplotlib.pyplot as plt
# Import Qiskit and the HHL-related classes.
from qiskit import Aer
from qiskit.algorithms.linear_solvers.observables import AbsoluteAverage, MatrixFunctional
from qiskit.circuit.library import ZGate
from qiskit.quantum_info import Statevector

 ```

Then, we want to create  TridiagonalToeplitz matrix, that depends on the number of qubits and solve it with the HHL algorihm
```
def simulate_hhl(num_state_qubits: int,
                 a: float,
                 b: float,
                 vector: np.array = None,
                 evolution_time: float = 1.0,
                 trotter_steps: int = 1,
                 tolerance: float = 0.01,
                 name: str = 'tridi' ,
                observable: np.array = MatrixFunctional(1, 1 / 2)) -> tuple:
    """
    Simulates the HHL algorithm for solving the linear system A x = b,
    where A is a Tridiagonal Toeplitz matrix of size 2^n x 2^n.

    Parameters:
        num_state_qubits (int): n (the number of qubits), yielding a matrix of size 2^n x 2^n.
        a (float): Value for the main diagonal of A.
        b (float): Value for the off-diagonals of A.
        vector (np.array, optional): Right-hand side vector; if not provided, defaults to [1, 0, ..., 0].
        evolution_time (float): Evolution time for the Hamiltonian simulation (passed to the matrix constructor).
        trotter_steps (int): Number of trotter steps (passed to the matrix constructor).
        tolerance (float): Tolerance for the matrix approximation.
        name (str): Name for the matrix.
        observable (np.array): Observable for the HHL solution.

    Returns:
        tuple: (result, tridi_matrix, vector), where result is the HHL solution object.
    """
    try:
        # Create the Tridiagonal Toeplitz matrix
        tridi_matrix = TridiagonalToeplitz(num_state_qubits,
                                           main_diag=a,
                                           off_diag=b,
                                           tolerance=tolerance,
                                           evolution_time=evolution_time,
                                           trotter_steps=trotter_steps,
                                           name=name)
        # Default vector if none provided
        if vector is None:
            vector = np.array([1] + [0]*(2**num_state_qubits - 1))
        
        # Create an instance of HHL and solve the system
        hhl_solver = HHL()
        result = hhl_solver.solve(tridi_matrix, vector)
        
        print(f"\n--- HHL Result for {num_state_qubits} qubits ---")
        print(result)
        print("\nQuantum circuit used to prepare the solution state:")
        # This prints the quantum circuit. (Depending on Qiskit version, the attribute name may differ.)
        print(result.state.draw())
        print('Euclidean Norm: ',result.euclidean_norm)
        average_solution = HHL().solve(tridi_matrix,vector,AbsoluteAverage())
        print('Average: ',average_solution)
        functional_solution = HHL().solve(tridi_matrix, vector, observable)
        print(f"Inner product of observable with B: {functional_solution.observable}")        
        return result, tridi_matrix, vector

    except Exception as e:
        print(f"Simulation failed for n = {num_state_qubits}. Error: {e}")
        return None, None, None

```

To answer the question,  What is the maximum number of qubits \(M\) that you can simulate on your machine? We can create a small code:
```
max_qubits = 0
for n in range(1, 20):
    solution = simulate_hhl(n,a=1, b=-1/3)
    if solution:
        max_qubits = n
    else:
        break
```
That will give us a maximum number of 5.
In my case, the computer used has the following specifications

```
# Get processor information
processor_info = cpuinfo.get_cpu_info()
print("Processor:", processor_info['brand_raw'])

# Get GPU information
gpus = GPUtil.getGPUs()
if gpus:
    for gpu in gpus:
        print(f"GPU: {gpu.name}")
else:
    print("No GPU found.")

```
Output:
```
Processor: Apple M3 Max
No GPU found.
```

### Using Qiskit HHL Class

Use the Qiskit HHL class documented in the Jupyter notebook `hhl_tutorial.ipynb` to provide explicit quantum circuits that prepare a quantum solution for the linear system. To achieve this, use the `state` property of the `LinearSolverResult` object returned by `HHL.solve()` and the `print` command.


This part is donde in the previous function, particularlly:


```
print(result.state.draw())
```

### Estimating Properties for HHL Solutions

Use the HHL quantum algorithm to estimate properties of the HHL solutions for different systems of equations. Choose a system $(Ax = b)$ defined by a tridiagonal matrix $(A)$ that you can solve using HHL with at most $(M \leq 5)$ qubits. Generate it using the `TridiagonalToeplitz` class:

```python
TridiagonalToeplitz(num_state_qubits, main_diag, off_diag, tolerance=0.01, evolution_time=1.0, trotter_steps=1, name='tridi')
```

[Documentation](https://docs.quantum.ibm.com/api/qiskit/0.40/qiskit.algorithms.linear_solvers.TridiagonalToeplitz)

Solve the system for your chosen matrix and report the following properties of the solutions:

1. **Vector norm** $(|x|)$ of the solution vector (use the `euclidean_norm` property).
2. **Average of the vector entries**.
3. **Inner product** $(\langle x | B | x \rangle)$ where $(B)$ is a Hermitian observable of your choice.
4. **Verification**: Use the classical NumPy solver `linalg.LinearSolverResult` to verify your solutions.

Steps 1 to 3 are done in the same function here:

```
print('Euclidean Norm: ',result.euclidean_norm)
average_solution = HHL().solve(tridi_matrix,vector,AbsoluteAverage())
print('Average: ',average_solution)
functional_solution = HHL().solve(tridi_matrix, vector, observable)
print(f"Inner product of observable with B: {functional_solution.observable}")  
```
While the classical solution can be obtained from:

```
def verify_solution(matrix, vector):
    classical_solution = NumPyLinearSolver().solve(matrix, vector / np.linalg.norm(vector))
    print("Classical solution:", classical_solution.state)
```

### Influence of Matrix Entry Size on Performance

Study the influence of performance concerning the size of the matrix entries for tridiagonal matrices. Use different ranges for matrix entries:

1. \(a, b\) are randomly chosen real numbers in the interval \([-1, 1]\).
2. \(a, b\) are randomly chosen \(\log(n)\)-digit numbers in the interval \([-n, n]\).
3. \(a, b\) are randomly chosen \(n\)-digit numbers in $([-2^n, 2^n])$.

**Question:** What is the influence of the size of the range on the cost of running HHL?

