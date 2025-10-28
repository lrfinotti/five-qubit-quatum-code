# ---
# jupyter:
#   jupytext:
#     default_lexer: ipython3
#     formats: ipynb,py:percent,md:myst
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister, Parameter, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator

from qiskit_aer import AerSimulator
from qiskit import transpile

import pylatexenc

# for testing
from random import randint

import numpy as np
# make numpy output look better
np_version = int(np.version.version.split(".")[0])
if np_version >= 2:
    np.set_printoptions(legacy="1.25")

# %% [markdown]
# # Five Qubit Error-Correcting Code

# %% [markdown]
# ## Stabilizer Group

# %% [markdown]
# We follow Nielsen and Chuang's [Quantum Computation and Quantum Information](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview), Section 10.5.6.
#
# Generators of the stabilizer group:
#
# $$
#   \begin{align*}
#     g_0 &= X \otimes Z \otimes Z \otimes X \otimes \mathbb{I}. \\
#     g_1 &= \mathbb{I} \otimes X \otimes Z \otimes Z \otimes X ,\\
#     g_2 &= X \otimes \mathbb{I} \otimes X \otimes Z \otimes Z ,\\
#     g_3 &= Z \otimes X \otimes \mathbb{I} \otimes X \otimes Z, \\
#   \end{align*}
# $$

# %%
g_gates_str = [
    ["x", "i", "x", "z", "z"],
    ["z", "x", "i", "x", "z"],
    ["x", "z", "z", "x", "i"],
    ["i", "x", "z", "z", "x"],
]

# %%
g = []

for gates in g_gates_str:
    quantum_register = QuantumRegister(size=len(gates), name="x")
    circuit = QuantumCircuit(quantum_register)
    for j, gate in enumerate(gates):
        if gate == "x":
            circuit.x(j)
        elif gate == "z":
            circuit.z(j)
    g.append(circuit)

# %%
i = 2
g[i].draw("mpl")

# %% [markdown]
# ## Encoding
#
# We follow the circuit provided in Chandak, Mardia, and Tolunay's [Implementation and analysis of stabilizer codes in pyQuil](https://shubhamchandak94.github.io/reports/stabilizer_code_report.pdf)

# %%
quantum_register = QuantumRegister(size=5, name="x")
encoder_circ = QuantumCircuit(quantum_register)

encoder_circ.h(quantum_register[:-1])
encoder_circ.z(-1)
for i in range(4):
    encoder_circ.cx(i, 4)
encoder_circ.cz(0, 4)
for i in range(0, 4, 2):
    encoder_circ.cz(i, i + 1)
for i in range(1, 5, 2):
    encoder_circ.cz(i, i + 1)

encoder_circ.draw("mpl")

# %%
logical_0 = Statevector(encoder_circ)

# %% [markdown]
# Let's check that the coefficients are what we expect.  They should all be real:

# %%
for x in logical_0.data:
    if x.imag != 0:
        print(f"{x} is not real!")
        break
else:
    print("All real!")

# %% [markdown]
# They should all be $0$ or $\pm 1/4$:

# %%
a = np.abs(4 * logical_0.data)
np.all(np.isclose(a, 0) | np.isclose(a, 1))


# %%
def binary_digits(a, n):
    """
    Given a and n, returns the first n digits of the binary representation of a.

    INPUTS:
    * a: A positive integer;
    * n: the number of binary digits.

    OUTPUT:
    An array of length n containing the first n binary digits of a, from letft to right.
    """
    return (a % 2 ** np.arange(1, n + 1)) // 2 ** np.arange(n)


for i, coef in enumerate(4 * logical_0.data):
    if coef != 0:
        print(f"{np.rint(coef.real).astype(int):>3}: {binary_digits(i, 5)}")

# %% [markdown]
# To encode $\left| 1_L \right\rangle$ we need to use $\left| 0 \right\rangle^{\otimes 4} \left| 0 \right\rangle$ as input:

# %%
quantum_register = QuantumRegister(size=5, name="x")
circ = QuantumCircuit(quantum_register)

circ.x(-1)
circ.compose(encoder_circ, inplace=True)

circ.draw("mpl")

# %%
logical_1 = Statevector(circ)

# %% [markdown]
# Let's check that the coefficients are what we expect.  They should all be real:

# %%
for x in logical_1.data:
    if x.imag != 0:
        print(f"{x} is not real!")
        break
else:
    print("All real!")

# %% [markdown]
# They should all be $0$ or $\pm 1/4$:

# %%
a = np.abs(4 * logical_1.data)
np.all(np.isclose(a, 0) | np.isclose(a, 1))

# %%
for i, coef in enumerate(4 * logical_1.data):
    if coef != 0:
        print(f"{np.rint(coef.real).astype(int):>3}: {binary_digits(i, 5)}")

# %% [markdown]
# Let's test against the $g_i$'s:

# %%
quantum_register_1 = QuantumRegister(size=5, name="x")
set_1 = QuantumCircuit(quantum_register_1)
set_1.x(-1)

for i, gi in enumerate(g):
    output_0 = Statevector(encoder_circ.compose(gi))
    output_1 = Statevector(set_1.compose(encoder_circ.compose(gi)))

    cond = np.array_equal(output_0.data, logical_0.data) and np.array_equal(output_1, logical_1.data)

    if not cond:
        print("Failed!")
        break
else:
    print("It worked!")

# %% [markdown]
# ## Error Correction

# %%
register_size = len(g_gates_str[0])
checks_size = len(g_gates_str)

quantum_register = QuantumRegister(size=register_size, name="x")
checks_register = QuantumRegister(size=checks_size, name="c")

code_circuit = QuantumCircuit(quantum_register, checks_register)

for i, gates in enumerate(g_gates_str):
    for j, gate in enumerate(gates):
        if gate == "x":
            code_circuit.cx(quantum_register[j], checks_register[i])
    if "z" in gates:
        code_circuit.h(checks_register[i])
        for j, gate in enumerate(gates):
            if gate == "z":
                code_circuit.cx(checks_register[i], quantum_register[j])
        code_circuit.h(checks_register[i])
    code_circuit.barrier()

code_circuit.draw("mpl")

# %%

# %%

# %%
syndromes = ClassicalRegister(size=5, name="s")

code_circuit.add_register(syndromes)
