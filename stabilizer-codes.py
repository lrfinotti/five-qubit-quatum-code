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
# from random import random, randint, choice

import numpy as np
# make numpy output look better
np_version = int(np.version.version.split(".")[0])
if np_version >= 2:
    np.set_printoptions(legacy="1.25")

import matplotlib.pyplot as plt
# # %matplotlib inline

# style
# plt.rcParams['figure.figsize'] = (10, 8)
# plt.style.use('fivethirtyeight')
plt.style.use("ggplot")

# %% [markdown]
# # Examples of Stabilizer Codes

# %% [markdown]
# ## Five-Qubit Error-Correcting Code

# %% [markdown]
# We implement here the [Quantum Five-Qubit Error Correcting Code](https://en.wikipedia.org/wiki/Five-qubit_error_correcting_code), which is a $[[5, 1, 3]]$ code.  We then run experiments with two logical qubits having the probability of getting a *Pauli error* (i.e., and $X$, $Y$, or $Z$ error) in each of the physical qubits with a given probability $p$, for a few values of $p$.  So, the probability of an error occurring in each physical qubit is $3p$, and each error, $X$, $Y$, and $Z$ has equal probability of occurring.  We also present a visualization of the results.

# %% [markdown]
# ### Stabilizer Group

# %% [markdown]
# We follow Nielsen and Chuang's [Quantum Computation and Quantum Information](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview), Section 10.5.6.
#
# The generators of the stabilizer group are:
#
# $$
#   \begin{align*}
#     g_0 &= X \otimes Z \otimes Z \otimes X \otimes \mathbb{I}. \\
#     g_1 &= \mathbb{I} \otimes X \otimes Z \otimes Z \otimes X ,\\
#     g_2 &= X \otimes \mathbb{I} \otimes X \otimes Z \otimes Z ,\\
#     g_3 &= Z \otimes X \otimes \mathbb{I} \otimes X \otimes Z, \\
#   \end{align*}
# $$
#
# Let's encode this information:

# %%
g5_gates_str = [
    ["x", "z", "z", "x", "i"],
    ["i", "x", "z", "z", "x"],
    ["x", "i", "x", "z", "z"],
    ["z", "x", "i", "x", "z"],
]


# %% [markdown]
# Now we create a function to generate the circuits for each element of the group:

# %%
def stabilizer_gen(gates_str):
    """
    Given a list containing lists of the tensor factors of the generating set of
    the stabilizer group, returns a list containing circuits for each generator.

    INPUT: A list containing of the tensor factors of the generating set of the
           stabilizer group.  Each gate must be a string of "x", "z", or "i" (for
           the identity.)  For example:
           [
              ["x", "z", "z", "x", "i"],
              ["i", "x", "z", "z", "x"],
              ["x", "i", "x", "z", "z"],
              ["z", "x", "i", "x", "z"],
           ]

    OUTPUT: A list containing circuits for each generator.
    """
    g = []
    for gates in gates_str:
        quantum_register = QuantumRegister(size=len(gates), name="x")
        circuit = QuantumCircuit(quantum_register)
        for j, gate in enumerate(gates):
            if gate == "x":
                circuit.x(j)
            elif gate == "z":
                circuit.z(j)
        g.append(circuit)
    return g


# %% [markdown]
# Let's save the generators in `g5`.  We can use it to test our encoding circuit.

# %%
g5 = stabilizer_gen(g5_gates_str)

# %% [markdown]
# Let's take a look at one of them:

# %%
i = 2
g5[i].draw("mpl")


# %% [markdown]
# ### Encoding
#
# According to Nielsen and Chuang, the logical qubits are given by:
#
# $$
# \begin{align*}
# \left| 0 \right\rangle_L 
# & = \frac{1}{4} (\phantom{+} \left|00000\right\rangle -\left|11000\right\rangle + \left|10100\right\rangle -\left|01100\right\rangle \\
# &\qquad +\left|10010\right\rangle + \left|01010\right\rangle -\left|00110\right\rangle -\left|11110\right\rangle \\
# & \qquad -\left|10001\right\rangle + \left|01001\right\rangle + \left|00101\right\rangle-\left|11101\right\rangle ,\\ 
# & \qquad-\left|00011\right\rangle -\left|11011\right\rangle-\left|10111\right\rangle -\left|01111\right\rangle) \\
# \left| 1 \right\rangle_L 
# &= \frac{1}{4} (-\left|10000\right\rangle -\left|01000\right\rangle -\left|00100\right\rangle -\left|11100\right\rangle \\
# &\qquad -\left|00010\right\rangle +\left|11010\right\rangle +\left|10110\right\rangle -\left|01110\right\rangle \\
# &\qquad  -\left|00001\right\rangle -\left|11001\right\rangle +\left|10101\right\rangle +\left|01101\right\rangle \\
# &\qquad -\left|10011\right\rangle +\left|01011\right\rangle -\left|00111\right\rangle +\left|11111\right\rangle).
# \end{align*}
# $$
#
# We follow a clever construction from [Stack Exchange](https://quantumcomputing.stackexchange.com/)'s thread [Nielsen&Chuang 5-qubit quantum error-correction encoding gate](https://quantumcomputing.stackexchange.com/questions/14264/nielsenchuang-5-qubit-quantum-error-correction-encoding-gate) for the encoding:

# %%
def five_qubit_encoder():
    """
    Returns the encoding circuit for the Five-Qubit Error Correction Code.
    """
    quantum_register = QuantumRegister(size=5, name="x")
    encoder_circ = QuantumCircuit(quantum_register)

    encoder_circ.z(0)
    encoder_circ.h(quantum_register[1:])
    for i in range(4):
        encoder_circ.cx(4 - i, 0)
    encoder_circ.cz(0, 4)
    for i in range(1, 5, 2):
        encoder_circ.cz(i, i + 1)
    for i in range(0, 4, 2):
        encoder_circ.cz(i, i + 1)
    return encoder_circ


# %% [markdown]
# Let's visualize it:

# %%
encoder5 = five_qubit_encoder()

encoder5.draw("mpl")

# %% [markdown]
# #### Testing the Encoding Circuit

# %% [markdown]
# Let's check that this encoding circuit works.  We start with $\left| 0 \right\rangle_L$.

# %%
logical5_0 = Statevector(encoder5)

# %% [markdown]
# Firstly, the scalars in the linear combination should all be real:

# %%
for x in logical5_0.data:
    if x.imag != 0:
        print(f"{x} is not real!")
        break
else:
    print("All real!")

# %% [markdown]
# They should also all be $0$ or $\pm 1/4$:

# %%
a = np.abs(4 * logical5_0.data)
np.all(np.isclose(a, 0) | np.isclose(a, 1))


# %% [markdown]
# Let's print the result to compare to expected value of $\left| 0 \right\rangle_L$:

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


for i, coef in enumerate(4 * logical5_0.data):
    if coef != 0:
        print(f"{np.rint(coef.real).astype(int):>3}: {binary_digits(i, 5)}")

# %% [markdown]
# So, it matches!
#
# Now, we repeat the idea for $\left| 1 \right\rangle_L$.  To encode it, we need to use $\left|10000\right\rangle$ as input:

# %%
quantum_register = QuantumRegister(size=5, name="x")
circ = QuantumCircuit(quantum_register)

circ.x(0)
circ.barrier()
circ.compose(encoder5, inplace=True)

circ.draw("mpl")

# %% [markdown]
# Let's save the state:

# %%
logical5_1 = Statevector(circ)

# %% [markdown]
# Again, we check that the coefficients are all real:

# %%
for x in logical5_1.data:
    if x.imag != 0:
        print(f"{x} is not real!")
        break
else:
    print("All real!")

# %% [markdown]
# Now we check if they are all either $0$ or $\pm 1/4$:

# %%
a = np.abs(4 * logical5_1.data)
np.all(np.isclose(a, 0) | np.isclose(a, 1))

# %% [markdown]
# And we print the result to compare to the expected expression for $\left| 1 \right\rangle_L$:

# %%
for i, coef in enumerate(4 * logical5_1.data):
    if coef != 0:
        print(f"{np.rint(coef.real).astype(int):>3}: {binary_digits(i, 5)}")


# %% [markdown]
# Again, it matches!
#
# Finally, let's check that $\left| 0 \right\rangle_L$ and $\left| 1 \right\rangle_L$ are indeed $+1$-eigenvalues of the stabilizer group generators, i.e., the $g_i$'s.
#
# Let's write a more general function for that:

# %%
def test_logical_qubit(encoder, stabilizer_group):
    """
    Test if a given encoder yields +1-eigenstates of the stabilizer group.

    INPUTS:
    * encoder: a circuit with the encorder for the stabilizer code;
    * stabilizer_group: a list of circuits for the generators of the stabilizer
      group.

    OUTPUT: True, if the encoder gives +1-eigenstates, and False otherwise.
    """
    n_qubits = encoder.num_qubits

    for qubit in [0, 1]:
        quantum_register = QuantumRegister(size=n_qubits, name="x")
        circ = QuantumCircuit(quantum_register)

        if qubit == 1:
            circ.x(0)

        circ.compose(encoder, inplace=True)

        logical = Statevector(circ)
        for i, gi in enumerate(stabilizer_group):
            output = Statevector(circ.compose(gi))
            if not np.array_equal(output.data, logical.data):
                return False

    return True


# %% [markdown]
# We can now check:

# %%
test_logical_qubit(encoder5, g5)


# %% [markdown]
# ### Error Correction

# %% [markdown]
# We now introduce the error correction.  We follow [Bernard Zygelman](https://www.physics.unlv.edu/~bernard/)'s [Five and Seven Qubit Codes](https://www.physics.unlv.edu/~bernard/MATH_book/Chap9/Notebook9_3.pdf).
#
# The idea is that if we have a single qubit operator $M$ with eigenvalues $\pm 1$, then we can determine if an eigenvector $\left| \psi \right\rangle$ has eigenvalue $1$ or $-1$ via:
#
# <img src="M-meas.png" alt="Single Qubit Eigenvalue Flip Circuit" width="500"/>
#
# If the measurement yields $0$, then $\left| \psi \right\rangle$ has eigenvalue $+1$, and if it yields $1$, then it has eigenvalue $-1$:
#
# $$
# \begin{align*}
# \left| \psi \right\rangle \left| 0 \right\rangle
# &\mapsto \frac{1}{\sqrt{2}} \left(\left| \psi \right\rangle \left| 0 \right\rangle +  \left| \psi \right\rangle \left| 1 \right\rangle \right) \\
# &\mapsto \frac{1}{\sqrt{2}} \left(\left| \psi \right\rangle \left| 0 \right\rangle  \pm  \left| \psi \right\rangle \left| 1 \right\rangle \right) \\
# &= \left| \psi \right\rangle \otimes \frac{1}{\sqrt{2}} \left( \left| 0 \right\rangle \pm \left| 1 \right\rangle \right) \\
# &\mapsto \left| \psi \right\rangle \left| (1 - (\pm 1)) /2 \right\rangle.
# \end{align*}
# $$
#
# We can apply this to $M=X$ and $M=Z$ to detect errors: suppose that $g_i$ has a tensor factor of $X$ in the $j$-th qubit. Then, we have that $X_j g_i = g_i X_j$, and if $\left| \psi \right\rangle$ is a $+1$-eigenstate of $g_i$, then
# $$
# g_i \left(X_j \left| \psi \right\rangle \right) = X_j \left(g_i \left| \psi \right\rangle \right) = X_j \left| \psi \right\rangle,
# $$
# i.e., $X_j \left| \psi \right\rangle$ is also a $+1$-eigenstate of $g_i$, and our circuit above yields a measurement of $0$.
#
#
# On the other hand, since $XZ=-ZX$ and $XY = -YX$, we have that $U_j g_i = -g_i U_j$ for $U$ either $Z$ or $Y$, and then
# $$
# g_i \left(U_j \left| \psi \right\rangle \right) = -U_j \left(g_i \left| \psi \right\rangle \right) = -U_j \left| \psi \right\rangle,
# $$
# i.e., $U_j \left| \psi \right\rangle$ is now a $-1$-eigenstate, and our circuit above yields a measurement of $1$.
#
# Similarly, if we have instead a tensor factor of $Z$ in the $j$-th qubit, we obtain a measurement of $0$ if we have an $Z$ error in the $j$-th qubit and $1$ if we have either a $Z$ or $Y$ error.
#
# Here is the implementation:

# %%
def stabilizer_error_correction_circ(encoder, gates_str, barrier=False):
    """
    Produces a circuit with the encoding and error measurements  for a stabilizer
    code.

    INPUTS:
    * encoder: the encoder for the code;
    * gates_str: a list containing of the tensor factors of the generating set of
                the stabilizer group.  Each gate must be a string of "x", "z", or
                "i" (for the identity.)  For example:
                   [
                      ["x", "z", "z", "x", "i"],
                      ["i", "x", "z", "z", "x"],
                      ["x", "i", "x", "z", "z"],
                      ["z", "x", "i", "x", "z"],
                   ]
    * barrier: a boolan determining if barriers should be added between encoding
               and steps of encoding.  (Default: False)

    OUTPUT: A circuit that encodes, checks for errors, and measure the checks in
            classical bits (syndromes).
    """
    register_size = len(gates_str[0])
    checks_size = len(gates_str)

    quantum_register = QuantumRegister(size=register_size, name="x")
    checks_register = AncillaRegister(size=checks_size, name="c")

    code_circ = QuantumCircuit(quantum_register, checks_register)

    if barrier:
        code_circ.barrier()

    # error detection
    code_circ.h(checks_register)
    if barrier:
        code_circ.barrier()

    for i, gates in enumerate(gates_str):
        for j, gate in enumerate(gates):
            if gate == "x":
                code_circ.cx(checks_register[i], quantum_register[j])
            elif gate == "z":
                code_circ.cz(checks_register[i], quantum_register[j])
        if barrier:
            code_circ.barrier()

    code_circ.h(checks_register)
    if barrier:
        code_circ.barrier()

    # add syndromes
    syndromes = ClassicalRegister(size=checks_size, name="s")
    code_circ.add_register(syndromes)
    code_circ.measure(checks_register, syndromes)

    return code_circ


# %% [markdown]
# Let's view the circuit:

# %%
code5_circuit = stabilizer_error_correction_circ(encoder5, g5_gates_str, barrier=True)
code5_circuit.draw("mpl")


# %% [markdown]
# We can also produce a table that gives the measurements for each Pauli error:

# %%
def error_table(gates_str):
    """
    Produces the error table given the generators of a stabilizer error.

    INPUT: A list containing of the tensor factors of the generating set of the
           stabilizer group.  Each gate must be a string of "x", "z", or "i" (for
           the identity.)  For example:
           [
              ["x", "z", "z", "x", "i"],
              ["i", "x", "z", "z", "x"],
              ["x", "i", "x", "z", "z"],
              ["z", "x", "i", "x", "z"],
           ]

    OUTPUT: A dictionary with keys as tuples made of a string and and integer and
            values lists of 0's and 1's.  For example a key ("x", 2) corresponds
            to X_2 (X in the tensor factor 2 and identity in all other factors), 
            and the list in the value corresponds to the measurements in the ancilla
            for that error.
    """
    n_qubits = len(gates_str[0])
    n_syndromes = len(gates_str)

    # save table in dictionary:
    decode_dict = {}

    for qubit_error in ["x", "z", "y"]:  # possible errors
        for i in range(n_qubits):  # qubit for the error
            syndrome = [0] * n_syndromes
            for j, gi in enumerate(gates_str):  # gates
                if gi[i] not in [qubit_error, "i"]:
                    syndrome[j] = 1
            decode_dict[(qubit_error, i)] = syndrome

    return decode_dict


# %%
decode5_dict = error_table(g5_gates_str)

for error, syndrome in decode5_dict.items():
    print(f"{error[0]}_{error[1]}: {syndrome}")


# %% [markdown]
# So, we have:
#
# | Error |   Measurement  | | Error |   Measurement  | | Error |   Measurement  |
# |-------|----------------|-|-------|----------------|-|-------|----------------|
# | $X_0$ | $(0, 0, 0, 1)$ | | $Z_0$ | $(1, 0, 1, 0)$ | | $Y_0$ | $(1, 0, 1, 1)$ |
# | $X_1$ | $(1, 0, 0, 0)$ | | $Z_1$ | $(0, 1, 0, 1)$ | | $Y_1$ | $(1, 1, 0, 1)$ |
# | $X_2$ | $(1, 1, 0, 0)$ | | $Z_2$ | $(0, 0, 1, 0)$ | | $Y_2$ | $(1, 1, 1, 0)$ |
# | $X_3$ | $(0, 1, 1, 0)$ | | $Z_3$ | $(1, 0, 0, 1)$ | | $Y_3$ | $(1, 1, 1, 1)$ |
# | $X_4$ | $(0, 0, 1, 1)$ | | $Z_4$ | $(0, 1, 0, 0)$ | | $Y_4$ | $(0, 1, 1, 1)$ |
#
#
# So, the measurements tell us which error occurred, and we can fix it by applying the same operator.  For instance, if we measure $(0,0,1,0)$ we apply $Z_2$ to the circuit.

# %% [markdown]
# Let's create a function that apply tests the measurements and applies the corrections based on the error table.  **Note that this function modifies the given circuit, and does not output a new one.**  
#
# (Unfortunately, although the process is algorithmic, I was not able to find a way to implement it more generally, since we cannot add "if tests" in a loop.)

# %%
def add_5_qubit_correction(circuit, qubits, syndromes, decode_dict):
    """
    Adds correction for five-qubit code circuit (in-place) using a decoding table.

    INPUTS:
    * circuit: the circuit containing the five-qubit code;
    * qubtis: register for the qubits and checks;
    * syndromes: syndromes containing the measurements;
    * decode_dict: dictionary containing the decoding table.
    """
    for error, values in decode_dict.items():
        with circuit.if_test((syndromes[0], values[0])):
            with circuit.if_test((syndromes[1], values[1])):
                with circuit.if_test((syndromes[2], values[2])):
                    with circuit.if_test((syndromes[3], values[3])):
                        if error[0] == "x":
                            circuit.x(qubits[error[1]])
                        elif error[0] == "z":
                            circuit.z(qubits[error[1]])
                        elif error[0] == "y":
                            circuit.y(qubits[error[1]])


# %% [markdown]
# Let's then add corrections to our circuit:

# %%
add_5_qubit_correction(
    code5_circuit, code5_circuit.qubits[0:5], code5_circuit.clbits, decode5_dict
)

code5_circuit.draw("mpl")


# %% [markdown]
# Let's now test the correction of a single Pauli error.  The following function creates a code circuit and adds a specific error to a specific qubit after the encoding.

# %%
def test_error_circuit(
    encoder,
    gates_str,
    add_corrections,
    qubit,
    error_gate,
    error_position,
    barrier=False,
):
    """
    Creates a code circuit with an error introduced to test the error-correction.

    INPUTS:
    * encoder: a circuit with the encorder for the stabilizer code;
    * gates_str: a list containing of the tensor factors of the generating set of
                the stabilizer group.  Each gate must be a string of "x", "z", or
                "i" (for the identity.)  For example:
                   [
                      ["x", "z", "z", "x", "i"],
                      ["i", "x", "z", "z", "x"],
                      ["x", "i", "x", "z", "z"],
                      ["z", "x", "i", "x", "z"],
                   ];
    * add_corrections: a function that adds the correction step for the code;
    * qubit: a boolean, or interger 0 or 1, for the qubit to be encoded;
    * error_gate: a string for the error gate, i.e., "x", "y", or "z";
    * error_position: the index for the qubit at which the error is introduced;
    * barrier: a boolan determining if barriers should be added between encoding
               and steps of encoding.  (Default: False.)

    OUTPUT: A circuit that sets the qubit to be encoded, followed by the encoder
            and the code circuit with corrections.
    """
    n_qubits = len(gates_str[0])
    n_checks = len(gates_str)

    decode_dict = error_table(gates_str)

    code_circuit = stabilizer_error_correction_circ(encoder, gates_str, barrier=barrier)
    add_corrections(
        code_circuit, code_circuit.qubits[:n_qubits], code_circuit.clbits, decode_dict
    )

    quantum_register = QuantumRegister(size=n_qubits, name="x")
    checks_register = QuantumRegister(size=n_checks, name="c")
    syndromes = ClassicalRegister(size=n_checks, name="s")

    qubit_measurements = ClassicalRegister(size=n_qubits, name="meas")

    test_circuit = QuantumCircuit(quantum_register, checks_register, syndromes)

    # encode correct value
    if qubit == 1:
        test_circuit.x(quantum_register[0])
        if barrier:
            test_circuit.barrier()

    # encode
    test_circuit.compose(encoder, inplace=True)
    if barrier:
        test_circuit.barrier()

    # add error
    if error_gate == "x":
        test_circuit.x(quantum_register[error_position])
    elif error_gate == "y":
        test_circuit.y(quantum_register[error_position])
    elif error_gate == "z":
        test_circuit.z(quantum_register[error_position])
    if barrier:
        test_circuit.barrier()

    # add code
    test_circuit.compose(code_circuit, inplace=True)
    if barrier:
        test_circuit.barrier()

    # decode
    test_circuit.compose(encoder.inverse(), inplace=True)
    if barrier:
        test_circuit.barrier()

    # measure
    test_circuit.add_register(qubit_measurements)
    test_circuit.measure(quantum_register, qubit_measurements)

    return test_circuit


# %% [markdown]
# Let's print an example of the circuit:

# %%
# change these to test:
encoded_qubit = False
error_gate = "x"
error_position = 2

test_error_circuit(
    encoder5, g5_gates_str, add_5_qubit_correction, encoded_qubit, error_gate, error_position, barrier=True
).draw("mpl")


# %% [markdown]
# Now, we write a function to test the error correction for a particular error:

# %%
def test_error_one_sim(
    encoder, gates_str, add_correction, qubit, error_gate, error_position
):
    """
    Tests if a code circuit with an error introduced can correct the error.

    INPUTS:
    * encoder: a circuit with the encorder for the stabilizer code;
    * gates_str: a list containing of the tensor factors of the generating set of
                the stabilizer group.  Each gate must be a string of "x", "z", or
                "i" (for the identity.)  For example:
                   [
                      ["x", "z", "z", "x", "i"],
                      ["i", "x", "z", "z", "x"],
                      ["x", "i", "x", "z", "z"],
                      ["z", "x", "i", "x", "z"],
                   ];
    * add_corrections: a function that adds the correction step for the code;
    * qubit: a boolean, or interger 0 or 1, for the qubit to be encoded;
    * error_gate: a string for the error gate, i.e., "x", "y", or "z";
    * error_position: the index for the qubit at which the error is introduced.

    OUTPUT: True, if the circuit could correct the error and False otherwise.
    """
    n_qubits = len(gates_str[0])

    circ = test_error_circuit(
        encoder,
        gates_str,
        add_correction,
        qubit,
        error_gate,
        error_position,
        barrier=False,
    )

    simulator = AerSimulator()

    # Transpile the circuit for the backend
    compiled_circuit = transpile(circ, simulator)

    # Run the circuit -- shot probably could be 1...
    job = simulator.run(compiled_circuit, shots=10)

    # Get the measurement counts
    counts = job.result().get_counts()

    return list(counts.keys())[0].split()[0][::-1] == str(int(qubit)) + (n_qubits - 1) * "0"


# %% [markdown]
# One test:

# %%
# change these to test:
encoded_qubit = True
error_gate = "x"
error_position = 2

test_error_one_sim(
    encoder5,
    g5_gates_str,
    add_5_qubit_correction,
    encoded_qubit,
    error_gate,
    error_position,
)


# %% [markdown]
# We can now do a full test for all possible one-qubit errors:

# %%
def test_error_full(encoder, gates_str, add_correction):
    """
    Tests if a code circuit with an error introduced can correct the error.

    INPUTS:
    * encoder: a circuit with the encorder for the stabilizer code;
    * gates_str: a list containing of the tensor factors of the generating set of
                the stabilizer group.  Each gate must be a string of "x", "z", or
                "i" (for the identity.)  For example:
                   [
                      ["x", "z", "z", "x", "i"],
                      ["i", "x", "z", "z", "x"],
                      ["x", "i", "x", "z", "z"],
                      ["z", "x", "i", "x", "z"],
                   ];
    * add_corrections: a function that adds the correction step for the code.

    OUTPUT: True, if the circuit could correct all possible single Pauli error in
            any qubit and with any encoded qubit, and False otherwise.
    """
    n_qubits = len(gates_str[0])

    for qubit in [True, False]:
        for error_gate in ["x", "y", "z"]:
            for error_position in range(n_qubits):
                if not test_error_one_sim(
                    encoder,
                    gates_str,
                    add_correction,
                    qubit,
                    error_gate,
                    error_position,
                ):
                    print(f"Failed for {qubit = }, {error_gate = }, {error_position = }")
                    return False

    return True


# %%
test_error_full(encoder5, g5_gates_str, add_5_qubit_correction)


# %% [markdown]
# So, indeed the code can fully correct a single Pauli error in any encoded qubit.

# %% [markdown]
# ### Two Qubit Encoding/Decoding and Test

# %% [markdown]
# We now need to produce a circuit that encodes two qubits using the five-qubit code, but introduces error according to a given probability.  But let's create a more general function, that can do it for other stabilizer codes and any number of qubits to be encoded:

# %%
def n_qubit_plus_errors_circ(
    n, gates_str, encoder, add_correction, qubits, p, barrier=False
):
    """
    Produces a circuit that prepares the state for n qubits to be encoded followed
    by the encoders (for each qubit), random individual errors in each qubit of X,
    Y, or Z with probability p for each one and each qubit, followed by the
    correction, decoding for all qubits, and measuments.

    INPUTS:
    * n: the number of qubits to be encoded;
    * encoder: a circuit with the encorder for the stabilizer code;
    * add_corrections: a function that adds the correction step for the code;
    * qubits: a list/tuple of booleans, or intergers 0 or 1, that represent the
              qubits to be encoded;
    * p: the probability that an X, Y, or Z error occurs in a physical qubit
         (so the probability that at least one error occur in a physical qubit
         is 3 * p, and and each error has the same probability);
    * barrier: a boolan determining if barriers should be added between encoding
               and steps of encoding.  (Default: False.)

    OUTPUTS:
    * a circuit that prepares the state for the qubits to be encoded, followed by the
      encoders (for each qubit), random individual errors in each qubit of X, Y, or Z
      with probability p for each one and each qubit, followed by the correction,
      decoding for all qubits, and measuments;
    * a list of strings with errors for each qubit, with "i" representing the identity.
    """
    n_qubits = len(gates_str[0])
    n_ancillas = len(gates_str)

    code_circ = stabilizer_error_correction_circ(encoder, gates_str, barrier=barrier)
    decode_dict = error_table(gates_str)

    # registers
    quantum_register = QuantumRegister(size=n * n_qubits, name="x")
    checks_register = AncillaRegister(size=n * n_ancillas, name="c")
    syndromes = ClassicalRegister(size=n * n_ancillas, name="s")

    # circuit
    circ = QuantumCircuit(
        quantum_register,
        checks_register,
        syndromes,
    )

    # set initial state
    for i in range(n):
        if qubits[i]:
            circ.x(quantum_register[i * n_qubits])
        if barrier:
            circ.barrier()

    # encoders
    for i in range(n):
        circ.compose(
            encoder,
            quantum_register[i * n_qubits : (i + 1) * n_qubits],
            inplace=True,
        )
        if barrier:
            circ.barrier()

    # random errors
    error_occurred = ["i"] * (n * n_qubits)
    # for i in range(n * n_qubits):
    #     rnd = np.random.random(3)
    #     # X error
    #     if rnd[0] < p:
    #         circ.x(quantum_register[i])
    #         error_occurred[i] = "x"
    #     # Y error
    #     if rnd[1] < p:
    #         circ.y(quantum_register[i])
    #         if error_occurred[i] == "i":
    #             error_occurred[i] = "y"
    #         else:
    #             error_occurred[i] += "y"
    #     # Z error
    #     if rnd[2] < p:
    #         circ.z(quantum_register[i])
    #         if error_occurred[i] == "i":
    #             error_occurred[i] = "z"
    #         else:
    #             error_occurred[i] += "z"
    #     # reverse order to see it as composition (right to left)
    #     error_occurred[i] = error_occurred[i][::-1]
    for i in range(n * n_qubits):
        rnd = np.random.random()
        if rnd < 3 * p:
            error = np.random.choice(["x", "y", "z"])
            error_occurred[i] = error
            if error == "x":
                circ.x(quantum_register[i])
            elif error == "y":
                circ.y(quantum_register[i])
            else:
                circ.z(quantum_register[i])
    if barrier:
        circ.barrier()

    # encoding + correction
    for i in range(n):
        circ.compose(
            code_circ,
            qubits=list(range(i * n_qubits, (i + 1) * n_qubits))
            + list(
                range(
                    n * n_qubits + i * n_ancillas, n * n_qubits + (i + 1) * n_ancillas
                )
            ),
            clbits=syndromes[i * n_ancillas : (i + 1) * n_ancillas],
            inplace=True,
        )
        if barrier:
            circ.barrier()

    # recovery
    for i in range(n):
        add_correction(
            circ,
            quantum_register[i * n_qubits : (i + 1) * n_qubits],
            syndromes[i * n_ancillas : (i + 1) * n_ancillas],
            decode_dict,
        )
        if barrier:
            circ.barrier()

    # decoding
    for i in range(n):
        circ.compose(
            encoder.inverse(),
            quantum_register[i * n_qubits : (i + 1) * n_qubits],
            inplace=True,
        )
        if barrier:
            circ.barrier()

    # add measurements
    qubit_measurements = ClassicalRegister(size=n * n_qubits, name="meas")
    circ.add_register(qubit_measurements)
    circ.measure(quantum_register, qubit_measurements)

    return circ, error_occurred


# %% [markdown]
# We can now apply it to our particular case:

# %%
def two_qubit_plus_errors_5_circ(qubits, p, barrier=False):
    """
    Produces a circuit that prepares the state for 2 qubits to be encoded using
    the 5-qubit code, followed by the encoders (for each qubit), random individual
    errors in each qubit of X, Y, or Z with probability p for each one and each
    qubit, followed by the correction, decoding for all qubits, and measuments.

    INPUTS:
    * qubits: a list/tuple of booleans, or intergers 0 or 1, that represent the
              qubits to be encoded;
    * p: the probability that an X, Y, or Z error occurs in a physical qubit
         (so the probability that at least one error occur in a physical qubit
         is 3 * p, and and each error has the same probability);
    * barrier: a boolan determining if barriers should be added between encoding
               and steps of encoding.  (Default: False.)

    OUTPUTS:
    * a circuit that prepares the state for the qubits to be encoded, followed by the
      encoders (for each qubit), random individual errors in each qubit of X, Y, or Z
      with probability p for each one and each qubit, followed by the correction,
      decoding for all qubits, and measuments;
    * a list of strings with errors for each qubit, with "i" representing the identity.
    """

    g5_gates_str = [
        ["x", "z", "z", "x", "i"],
        ["i", "x", "z", "z", "x"],
        ["x", "i", "x", "z", "z"],
        ["z", "x", "i", "x", "z"],
    ]

    return n_qubit_plus_errors_circ(
        2,
        g5_gates_str,
        five_qubit_encoder(),
        add_5_qubit_correction,
        qubits,
        p,
        barrier=barrier,
    )


# %% [markdown]
# Let's test it then:

# %%
qubits = (0, 1)
p = 0.1

circ, error = two_qubit_plus_errors_5_circ(qubits, p, barrier=True)
circ.draw("mpl")

# %%
error


# %% [markdown]
# To perform tests and provide an empirical probability the our code we fix all errors when there is a probability of $p$ that a Pauli error occurs in each encoded qubit, we need to perform simulations.
#
# Again, we will start with a more general function, that works for different stabilizer codes, and use $n$ encoded qubits:

# %%
def n_qubit_plus_errors_sim(
    n, gates_str, encoder, add_correction, qubits, p, shots=10
):
    """
    Produces a circuit that prepares the state for n qubits to be encoded followed
    by the encoders (for each qubit), random individual errors in each qubit of X,
    Y, or Z with probability p for each one and each qubit, followed by the
    correction, decoding for all qubits, and measuments, then runs a simulation and
    verifies if the error was corrected.

    INPUTS:
    * n: the number of qubits to be encoded;
    * gates_str: a list containing of the tensor factors of the generating set of
                the stabilizer group.  Each gate must be a string of "x", "z", or
                "i" (for the identity.)  For example:
                   [
                      ["x", "z", "z", "x", "i"],
                      ["i", "x", "z", "z", "x"],
                      ["x", "i", "x", "z", "z"],
                      ["z", "x", "i", "x", "z"],
                   ];
    * encoder: a circuit with the encorder for the stabilizer code;
    * add_corrections: a function that adds the correction step for the code;
    * qubits: a list/tuple of booleans, or intergers 0 or 1, that represent the
              qubits to be encoded;
    * p: the probability that an X, Y, or Z error occurs in a physical qubit
         (so the probability that at least one error occur in a physical qubit
         is 3 * p, and and each error has the same probability);
    * shots: number of shots to be used in the simulation. (Default: 10.)

    OUTPUTS:
    * a boolean, with True if the circuit corrected the error;
    * a list of strings with errors for each qubit, with "i" representing the identity.
    """

    circ, error_occurred = n_qubit_plus_errors_circ(
        n, gates_str, encoder, add_correction, qubits, p, barrier=False
    )

    n_qubits = (circ.num_qubits - circ.num_ancillas) // n

    # simulation
    simulator = AerSimulator()

    # Transpile the circuit for the backend
    compiled_circuit = transpile(circ, simulator)

    # Run the circuit
    job = simulator.run(compiled_circuit, shots=shots)

    # Get the measurement counts
    counts = job.result().get_counts()

    # check results
    # more than one result?
    results = {qubits.split()[0] for qubits in counts}
    if len(results) > 1:
        return False, error_occurred

    # REVERSE the result!
    result = results.pop()[::-1]

    # get the logical form of each qubit
    result_split = [result[i * n_qubits: (i + 1) * n_qubits] for i in range(n)]

    # check
    for res, qubit in zip(result_split, qubits):
        if res != str(int(qubit)) + "0" * (n_qubits - 1):
            return False, error_occurred

    return True, error_occurred


# %% [markdown]
# And, again, we use it for our particular case:

# %%
def two_qubit_plus_errors_5_sim(qubits, p, shots=10):
    """
    Produces a circuit that prepares the state for 2 qubits to be encoded using
    the 5-qubit code, followed by the encoders (for each qubit), random individual
    errors in each qubit of X, Y, or Z with probability p for each one and each
    qubit, followed by the correction, decoding for all qubits, and measuments,
    then runs a simulation and verifies if the error was corrected.

    INPUTS:
    * qubits: a list/tuple of booleans, or intergers 0 or 1, that represent the
              qubits to be encoded;
    * p: the probability that an X, Y, or Z error occurs in a physical qubit
         (so the probability that at least one error occur in a physical qubit
         is 3 * p, and and each error has the same probability);
    * shots: number of shots to be used in the simulation. (Default: 10.)

    OUTPUTS:
    * a boolean, with True if the circuit corrected the error;
    * a list of strings with errors for each qubit, with "i" representing the identity.
    """
    g5_gates_str = [
        ["x", "z", "z", "x", "i"],
        ["i", "x", "z", "z", "x"],
        ["x", "i", "x", "z", "z"],
        ["z", "x", "i", "x", "z"],
    ]

    return n_qubit_plus_errors_sim(
        2,
        g5_gates_str,
        five_qubit_encoder(),
        add_5_qubit_correction,
        qubits,
        p,
    )


# %% [markdown]
# So, let's try it once:

# %%
qubits = (1, 1)
p = 0.05

two_qubit_plus_errors_5_sim(qubits, p)

# %% [markdown]
# Again, if only at most one error occurs in each of the first and last five qubits, the code should correct the error.

# %% [markdown]
# Let's now collect data for different values of $p$.  (**Note:** It can take a long time to run it!)

# %%
# %%time
max_prob = 0.17
step = 0.01

xs = np.arange(0, max_prob + step, step)
ys = np.zeros_like(xs)

# number of tries for each p
number_of_tries = 100

for i, p in enumerate(xs):
    count = 0
    for _ in range(number_of_tries):
        qubits = np.random.randint(0, 2, 2)
        res, _ = two_qubit_plus_errors_5_sim(qubits, p)
        if res:
            count += 1
        ys[i] = count / number_of_tries

# %% [markdown]
# Here are is the table with the percentage of correctly decoded pairs of qubits:

# %%
print(f"{'p':^6} | percentage")
print("------ | ---------- ")
for x, y in zip(xs, ys):
    print(f"{x:^6.2f} | {y:^10.2f}")


# %% [markdown]
# The probability that any error occur in a qubit is $3p$.  So if we have $n$ qubits and the probability that we have at most one error, in which case our code will certainly produce the correct result, is:
# $$
# (1 - 3p)^n + n \cdot (1 - 3p)^{n-1} \cdot 3p = (1 - 3p)^{n-1} \left(1 - 3p + n \cdot 3p \right) = (1 - 3p)^{n-1} \left(1 + 3(n-1)p\right).
# $$
# In practice, although with low probability, the code could fix more errors "by accident", so that should approximate the lower bound for the percentage of errors corrected.
#
# Let's add it as a function:

# %%
def lower_probability(n_qubits, p):
    """
    Give the probability we get at most one error of probability 3 * p in all
    n_qubits qubits.

    INPUTS:
    * n_qubits: number of qubits;
    * p: 1/3 of the probability that a Pauli error occurs.

    OUTPUT:
    the probability we get at most one error occurs.
    """
    return (1 - 3 * p) ** (n_qubits - 1) * (1 + 3 * (n_qubits - 1) * p)


# %% [markdown]
# Here is the corresponding plot:

# %%
ys_est = lower_probability(5, xs)

plt.plot(xs, ys_est, alpha=0.4, label="Estimates")
plt.plot(xs, ys, "--o", label="Empirical Probability")

plt.title("Empirical Probability of Correcting Errors")
plt.xlabel("$p$ (Probability of Pauli Error)")
plt.ylabel("Empirical Probability")
plt.legend()

# plt.savefig("5-qb.png")

plt.show()

# %% [markdown]
# Below is the hard coded data from one run, with 100 examples per $p$ between $0$ and $0.17$ in steps of $0.1$:

# %%
xs_found_1000_5qb = np.array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ,
       0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17])
ys_found_1000_5qb = np.array([1.   , 0.99 , 0.951, 0.889, 0.839, 0.809, 0.754, 0.679, 0.613,
       0.585, 0.527, 0.473, 0.387, 0.395, 0.34 , 0.312, 0.296, 0.3])

# %% [markdown]
# Here is the table for that run:
#
# |   $p$   | percentage  |
# |---------|:-----------:|
# | $0.00$  |   $1.000$   |
# | $0.01$  |   $0.990$   |
# | $0.02$  |   $0.951$   |
# | $0.03$  |   $0.889$   |
# | $0.04$  |   $0.839$   |
# | $0.05$  |   $0.809$   |
# | $0.06$  |   $0.754$   |
# | $0.07$  |   $0.679$   |
# | $0.08$  |   $0.613$   |
# | $0.09$  |   $0.585$   |
# | $0.10$  |   $0.527$   |
# | $0.11$  |   $0.473$   |
# | $0.12$  |   $0.387$   |
# | $0.13$  |   $0.395$   |
# | $0.14$  |   $0.340$   |
# | $0.15$  |   $0.312$   |
# | $0.16$  |   $0.296$   |
# | $0.17$  |   $0.300$   |
#
# Here is the corresponding graph:
#
# <img src="5-qb-1000.png" alt="Percentage Corrected for One Run"/>

# %% [markdown]
# ## Steane's Code

# %% [markdown]
# We can apply similar ideas, and the functions created above, to other stabilizer codes.  Now we turn our attention to [Steane code](https://en.wikipedia.org/wiki/Steane_code), which is a $[[7,4,3]]$ code.  In this example, we will only verify the correction, for probability $3p$ that an error occurs, for one encoded qubit.

# %% [markdown]
# ### Stabilizer Group

# %% [markdown]
# Again, we first encode the generating set for the stabilizer group, as found in [Wikipedia](https://en.wikipedia.org/wiki/Steane_code) (or from the matrix in [EC Zoo](https://errorcorrectionzoo.org/)'s [Steane Code](https://errorcorrectionzoo.org/c/steane)):

# %%
g7_gates_str = [
    ["i", "i", "i", "x", "x", "x", "x"],
    ["i", "x", "x", "i", "i", "x", "x"],
    ["x", "i", "x", "i", "x", "i", "x"],
    ["i", "i", "i", "z", "z", "z", "z"],
    ["i", "z", "z", "i", "i", "z", "z"],
    ["z", "i", "z", "i", "z", "i", "z"],
]

# %% [markdown]
# As before, we can use the `stabilizer_gen` function to create circuits for each generator.

# %%
g7 = stabilizer_gen(g7_gates_str)


# %% [markdown]
# ### Encoding

# %% [markdown]
# We follow [MITRE STEM](https://stem.mitre.org/)'s [Steane's Error Correction Code](https://stem.mitre.org/quantum/error-correction-codes/steane-ecc.html).  Here is the encoder circuit given:

# %%
def steanes_encoder():
    """
    Returns the encoding circuit for the Steane's Error Correction Code.
    """
    quantum_register = QuantumRegister(size=7, name="x")
    encoder = QuantumCircuit(quantum_register)

    encoder.h(quantum_register[4:])
    encoder.cx(0, [1, 2])
    encoder.cx(6, [0, 1, 3])
    encoder.cx(5, [0, 2, 3])
    encoder.cx(4, [1, 2, 3])

    return encoder


# %%
encoder7 = steanes_encoder()

encoder7.draw("mpl")

# %% [markdown]
# As seen in [EC Zoo](https://errorcorrectionzoo.org/)'s [Steane Code](https://errorcorrectionzoo.org/c/steane), the logical qubits are:
# $$
# \begin{align*}
# \left| 0 \right\rangle_L
# &= \frac{1}{\sqrt{8}} (\left|0000000\right\rangle + \left|0111100\right\rangle + \left|1011010\right\rangle + \left|1100110\right\rangle \\
# &\qquad + \left|1101001\right\rangle + \left|1010101\right\rangle + \left|0110011\right\rangle + \left|0001111\right\rangle), \\
# \left| 1 \right\rangle_L
# &= \frac{1}{\sqrt{8}} (\left|1110000\right\rangle + \left|1001100\right\rangle + \left|0101010\right\rangle + \left|0010110\right\rangle \\
# &\quad + \left|0011001\right\rangle + \left|0100101\right\rangle + \left|1000011\right\rangle + \left|1111111\right\rangle).
# \end{align*}
# $$

# %% [markdown]
# #### Testing the Encoding Circuit

# %% [markdown]
# Again, let's check that the encoder works.

# %%
logical7_0 = Statevector(encoder7)

# %% [markdown]
# All scalar coefficients must be real:

# %%
for x in logical7_0.data:
    if x.imag != 0:
        print(f"{x} is not real!")
        break
else:
    print("All real!")

# %% [markdown]
# The also must be either $0$ or $1/\sqrt{8}$:

# %%
a = 8 * logical7_0.data ** 2
np.all(np.isclose(a, 0) | np.isclose(a, 1))

# %% [markdown]
# Let's check the values:

# %%
for i, coef in enumerate(np.sqrt(8) * logical7_0.data):
    if coef != 0:
        print(f"{np.rint(coef.real).astype(int):>3}: {binary_digits(i, 7)}")

# %% [markdown]
# It matches $\left|0\right\rangle_L$, as expected.

# %% [markdown]
# Now $\left|1\right\rangle_L$:

# %%
quantum_register = QuantumRegister(size=7, name="x")
circ = QuantumCircuit(quantum_register)

circ.x(0)
circ.compose(encoder7, inplace=True)

logical7_1 = Statevector(circ)

# %% [markdown]
# Scalar coefficients must be real and either $0$ or $\frac{1}{\sqrt{8}}$:

# %%
for x in logical7_0.data:
    if x.imag != 0:
        print(f"{x} is not real!")
        break
else:
    print("All real!")

# %%
a = 8 * logical7_0.data ** 2
np.all(np.isclose(a, 0) | np.isclose(a, 1))

# %% [markdown]
# Here is the representation:

# %%
for i, coef in enumerate(np.sqrt(8) * logical7_1.data):
    if coef != 0:
        print(f"{np.rint(coef.real).astype(int):>3}: {binary_digits(i, 7)}")

# %% [markdown]
# Again, it matches the logical qubit $\left|1\right\rangle_L$.

# %% [markdown]
# Finally, we can use our `test_logical_qubit` function to test them against the stabilizer group:

# %%
test_logical_qubit(encoder7, g7)

# %% [markdown]
# ### Error Correction

# %% [markdown]
# We can use `stabilizer_error_correction_circ` to create the error-correcting circuit for Steane's code:

# %%
code7_circuit = stabilizer_error_correction_circ(encoder7, g7_gates_str, barrier=True)

code7_circuit.draw("mpl")

# %% [markdown]
# And `error_table` will give the error table:

# %%
decode7_dict = error_table(g7_gates_str)

for error, syndrome in decode7_dict.items():
    print(f"{error[0]}_{error[1]}: {syndrome}")


# %% [markdown]
# With error table, we can manually create the function to add the error-correction part of the circuit:

# %%
def add_7_qubit_correction(circuit, qubits, syndromes, decode_dict):
    """
    Adds correction for Steane's code circuit (in-place) using a decoding table.

    INPUT:
    * circuit: the circuit containing the Steane's code;
    * qubtis: register for the qubits and checks;
    * syndromes: syndromes containing the measurements;
    * decode_dict: dictionary containing the decoding table.
    """
    for error, values in decode_dict.items():
        with circuit.if_test((syndromes[0], values[0])):
            with circuit.if_test((syndromes[1], values[1])):
                with circuit.if_test((syndromes[2], values[2])):
                    with circuit.if_test((syndromes[3], values[3])):
                        with circuit.if_test((syndromes[4], values[4])):
                            with circuit.if_test((syndromes[5], values[5])):
                                if error[0] == "x":
                                    circuit.x(qubits[error[1]])
                                elif error[0] == "z":
                                    circuit.z(qubits[error[1]])
                                elif error[0] == "y":
                                    circuit.y(qubits[error[1]])


# %% [markdown]
# Let's test it:

# %%
add_7_qubit_correction(code7_circuit, code7_circuit.qubits[0:7], code7_circuit.clbits, decode7_dict)

code7_circuit.draw("mpl")

# %% [markdown]
# Let's use `test_error_full` to test the error-correction:

# %%
test_error_full(encoder7, g7_gates_str, add_7_qubit_correction)


# %% [markdown]
# Again, it works for every Pauli error in a single qubit.

# %% [markdown]
# ### Generating Function and Testing

# %% [markdown]
# We can use `n_qubit_plus_errors_circ` to generate a circuit that produces the Steane's code for one qubit, with Pauli errors appearing in each encoded qubit with probability $3p$:

# %%
def one_qubit_plus_errors_7_circ(qubit, p, barrier=False):
    """
    Produces a circuit that prepares the state for 1 qubits to be encoded using
    the Steane's code, followed by the encoder, random individual, errors in each
    qubit of X, Y, or Z with probability p for each one and each qubit, followed
    by the correction, decoding, and measuments.

    INPUTS:
    * qubits: a list/tuple of booleans, or intergers 0 or 1, that represent the
              qubits to be encoded;
    * p: the probability that an X, Y, or Z error occurs in a physical qubit
         (so the probability that at least one error occur in a physical qubit
         is 3 * p, and and each error has the same probability);
    * barrier: a boolan determining if barriers should be added between encoding
               and steps of encoding.  (Default: False.)

    OUTPUTS:
    * a circuit that prepares the state for a qubit to be encoded, followed by the
      encoder, random individual errors in each qubit of X, Y, or Z with probability
      p for each one and each qubit, followed by the correction, decoding, and
      measuments;
    * a list of strings with errors for each qubit, with "i" representing the identity.
    """

    g7_gates_str = [
        ["i", "i", "i", "x", "x", "x", "x"],
        ["i", "x", "x", "i", "i", "x", "x"],
        ["x", "i", "x", "i", "x", "i", "x"],
        ["i", "i", "i", "z", "z", "z", "z"],
        ["i", "z", "z", "i", "i", "z", "z"],
        ["z", "i", "z", "i", "z", "i", "z"],
    ]

    return n_qubit_plus_errors_circ(
        1,
        g7_gates_str,
        steanes_encoder(),
        add_7_qubit_correction,
        [qubit],
        p,
        barrier=barrier,
    )


# %% [markdown]
# Let's test it:

# %%
qubit = 1
p = 0.05

circ, error = one_qubit_plus_errors_7_circ(qubit, p, barrier=True)
circ.draw("mpl")

# %% [markdown]
# Let's check the error introduced:

# %%
error


# %% [markdown]
# To perform the tests, we need to introduce the simulation.  We use `n_qubit_plus_errors_sim` as before:

# %%
def one_qubit_plus_errors_7_sim(qubit, p, shots=10):
    """
    Produces a circuit that prepares the state for 1 qubits to be encoded using
    the Steane's code, followed by the encoder, random individual errors in each
    qubit of X, Y, or Z with probability p for each one and each qubit, followed
    by the correction, decoding, and measuments, then runs a simulation and
    verifies if the error was corrected.

    INPUTS:
    * qubits: a list/tuple of booleans, or intergers 0 or 1, that represent the
              qubits to be encoded;
    * p: the probability that an X, Y, or Z error occurs in a physical qubit
         (so the probability that at least one error occur in a physical qubit
         is 3 * p, and and each error has the same probability);
    * shots: number of shots to be used in the simulation. (Default: 10.)

    OUTPUTS:
    * a boolean, with True if the circuit corrected the error;
    * a list of strings with errors for each qubit, with "i" representing the identity.
    """
    g7_gates_str = [
        ["i", "i", "i", "x", "x", "x", "x"],
        ["i", "x", "x", "i", "i", "x", "x"],
        ["x", "i", "x", "i", "x", "i", "x"],
        ["i", "i", "i", "z", "z", "z", "z"],
        ["i", "z", "z", "i", "i", "z", "z"],
        ["z", "i", "z", "i", "z", "i", "z"],
    ]

    return n_qubit_plus_errors_sim(
        1,
        g7_gates_str,
        steanes_encoder(),
        add_7_qubit_correction,
        [qubit],
        p,
    )


# %% [markdown]
# So, let's run one test:

# %%
qubit = 0
p = 0.1

one_qubit_plus_errors_7_sim(qubit, p)

# %% [markdown]
# Let's collect data for different values of $p$ for the Steane's code.  (**Note:** It can take a long time to run it!)

# %%
# %%time
max_prob = 0.17
step = 0.01

xs = np.arange(0, max_prob + step, step)
ys = np.zeros_like(xs)

number_of_tries = 100

for i, p in enumerate(xs):
    count = 0
    for _ in range(number_of_tries):
        qubit = np.random.randint(0, 2)
        res, _ = one_qubit_plus_errors_7_sim(qubit, p)
        if res:
            count += 1
        ys[i] = count / number_of_tries

# %% [markdown]
# Here are is the table with the percentage of correctly decoded pairs of qubits:

# %%
print(f"{'p':^6} | percentage")
print("------ | ---------- ")
for x, y in zip(xs, ys):
    print(f"{x:^6.2f} | {y:^10.2f}")

# %% [markdown]
# Here is the corresponding plot:

# %%
ys_est = lower_probability(7, xs)

plt.plot(xs, ys_est, alpha=0.4, label="Estimates")
plt.plot(xs, ys, "--o", label="Empirical Probability")

plt.title("Percentage of Pauli Errors Corrected")
plt.xlabel("$p$ (Probability of Pauli Error)")
plt.ylabel("Percentage Corrected")
plt.legend()

# plt.savefig("steane.png")

plt.show()

# %% [markdown]
# Below is the hard coded data from one run, with $1000$ repetitions per $p$:

# %%
xs_found_1000_steanes = np.array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ,
       0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17])
ys_found_1000_steanes = np.array([1.   , 0.987, 0.94 , 0.889, 0.838, 0.762, 0.678, 0.611, 0.503,
       0.472, 0.401, 0.384, 0.327, 0.282, 0.257, 0.251, 0.208, 0.196])

# %% [markdown]
# Here is the table for that run:
#
# |   $p$   | percentage  |
# |---------|:-----------:|
# | $0.00$  |   $1.000$   |
# | $0.01$  |   $0.990$   |
# | $0.02$  |   $0.948$   |
# | $0.03$  |   $0.944$   |
# | $0.04$  |   $0.867$   |
# | $0.05$  |   $0.812$   |
# | $0.06$  |   $0.754$   |
# | $0.07$  |   $0.680$   |
# | $0.08$  |   $0.666$   |
# | $0.09$  |   $0.603$   |
# | $0.10$  |   $0.526$   |
# | $0.11$  |   $0.509$   |
# | $0.12$  |   $0.489$   |
# | $0.13$  |   $0.439$   |
# | $0.14$  |   $0.419$   |
# | $0.15$  |   $0.385$   |
# | $0.16$  |   $0.380$   |
# | $0.17$  |   $0.333$   |
#
# Here is the corresponding graph:
#
# <img src="steane-1000.png" alt="Percentage Corrected for One Run"/>

# %% [markdown]
# ## References

# %% [markdown]
# * [Wikipedia](https://en.wikipedia.org/)'s [Quantum Five-Qubit Error Correcting Code](https://en.wikipedia.org/wiki/Five-qubit_error_correcting_code)
# * Nielsen and Chuang's [Quantum Computation and Quantum Information](https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview)
# *  [Stack Exchange](https://quantumcomputing.stackexchange.com/)'s thread [Nielsen&Chuang 5-qubit quantum error-correction encoding gate](https://quantumcomputing.stackexchange.com/questions/14264/nielsenchuang-5-qubit-quantum-error-correction-encoding-gate)
# *  [Bernard Zygelman](https://www.physics.unlv.edu/~bernard/)'s [Five and Seven Qubit Codes](https://www.physics.unlv.edu/~bernard/MATH_book/Chap9/Notebook9_3.pdf)
# * [Wikipedia](https://en.wikipedia.org/)'s [Steane code](https://en.wikipedia.org/wiki/Steane_code)
# *  [MITRE STEM](https://stem.mitre.org/)'s [Steane's Error Correction Code](https://stem.mitre.org/quantum/error-correction-codes/steane-ecc.html)
# * [EC Zoo](https://errorcorrectionzoo.org/)'s [Steane Code](https://errorcorrectionzoo.org/c/steane)
