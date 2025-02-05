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
from qiskit.algorithms import HHL
from qiskit.algorithms.linear_solvers.matrices import TridiagonalToeplitz
from qiskit.circuit.library import ZGate
from qiskit.algorithms.linear_solvers.observables import AbsoluteAverage, MatrixFunctional
from qiskit.quantum_info import Statevector
#inputs
matrix = np.array([[1, -1/3], [-1/3, 1]])
vector = np.array([1, 0])



naive_hhl_solution = HHL().solve(matrix, vector)
tridi_matrix = TridiagonalToeplitz(1, 1, -1 / 3)
tridi_solution = HHL().solve(tridi_matrix, vector)


 

# %% [code]
def simulate_hhl(num_state_qubits: int,
                 a: float = 1,
                 b: float = -1/3,
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

        # Plot the solution vector

        print(f"\n--- HHL Result for {num_state_qubits} qubits ---")
        print(result)
        print("\nQuantum circuit used to prepare the solution state:")
        # This prints the quantum circuit. (Depending on Qiskit version, the attribute name may differ.)
        print('PITAAAA',result.state)
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
print('END')
max_qubits = 0
for n in range(1, 2):
    solution = simulate_hhl(n,a=1, b=-1/3)
    if solution:
        max_qubits = n
    else:
        break

print(f"Maximum number of qubits M that can be simulated: {max_qubits}")



# %% [code]
from qiskit.algorithms.linear_solvers.observables import AbsoluteAverage, MatrixFunctional

def estimate_properties(solution,tridi_matrix,vector): #give to it result
    # (a) Euclidean norm
    euclidean_norm = solution.euclidean_norm
    print(f"Euclidean norm: {euclidean_norm}")

    # (b) Average of the vector entries
    average_solution = HHL().solve(tridi_matrix, vector, AbsoluteAverage())
    print(f"Average of vector entries: {average_solution.observable}")

    # (c) Inner product with a Hermitian observable B
    observable = MatrixFunctional(1, 1 / 2)  # Example observable
    functional_solution = HHL().solve(tridi_matrix, vector, observable)
    print(f"Inner product with B: {functional_solution.observable}")
    return euclidean_norm, average_solution.observable, functional_solution.observable




def get_solution_vector(solution):
    """Extracts and normalizes simulated state vector
    from LinearSolverResult."""
    solution_vector = Statevector(solution.state).data[16:18].real
    print("Quantum solution:",solution_vector)
    norm = solution.euclidean_norm
    return norm * solution_vector / np.linalg.norm(solution_vector)


def generar_listas_binarias_a_decimal(X):
    # Validamos que X sea mayor que 1
    if X < 2:
        raise ValueError("El valor de X debe ser mayor o igual a 2.")
    
    # Primera lista: comienza con 1 y luego todos 0's
    lista1 = ['1'] + ['0'] * (X - 1)
    
    # Segunda lista: comienza y termina con 1, el resto 0's
    lista2 = ['1'] + ['0'] * (X - 2) + ['1']
    
    # Convertimos las listas binarias a números decimales
    decimal1 = int(''.join(lista1), 2)
    decimal2 = int(''.join(lista2), 2)
    
    return  decimal1, decimal2



def verify_solution(matrix, vector, result):
    classical_solution=NumPyLinearSolver().solve(matrix,vector/np.linalg.norm(vector))
    print("Classical solution:")
    print(classical_solution.state)
    print('------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print("Quantum solution:",Statevector(result.state).data   )
    solution=get_solution_vector(result)      
    # plt.bar(range(len(solution)), solution,color='orange',label='Quantum Solution')
    # plt.bar(range(len(classical_solution.state)), classical_solution.state,color='blue',label='Classical Solution',alpha=0.5)     
    # plt.show()
    print('QUANTUM',solution)
    print('CLASSICAL',classical_solution.state)
    return classical_solution
matrix = np.array([[1, -1/3], [-1/3, 1]])
vector = np.array([1, 0])
sol = simulate_hhl(1, a=1, b=-1/3)
result, tridi_matrix, vectors  = sol
print('RESULTAAAAAAA',vectors)
verify_solution(tridi_matrix, vectors, result)
print('NAIVW',len(naive_hhl_solution.state))
print(result.state.num_qubits)
primero, segundo = generar_listas_binarias_a_decimal(result.state.num_qubits)



print(primero,segundo)





















