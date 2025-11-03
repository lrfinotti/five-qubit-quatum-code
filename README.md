# Stabilizer Codes

## Description

In this mini project for the [Erdős Institute](https://www.erdosinstitute.org/)'s [Fall 2025 Quantum Computing Boot Camp](https://www.erdosinstitute.org/programs/fall-2025/quantum-computing-boot-camp), we explore examples of [stabilizer codes](https://en.wikipedia.org/wiki/Stabilizer_code), more specifically, the [Five-Qubit Error Correcting Code](https://en.wikipedia.org/wiki/Five-qubit_error_correcting_code) and the [Steane's Code](https://en.wikipedia.org/wiki/Steane_code).  (Hopefully soon I will add an example of a [CSS Code](https://en.wikipedia.org/wiki/CSS_code) as well.)

In the Jupyter notebook we construct the encoding and error-correction circuit and run examples where errors are randomly introduced.  We then collect data for the success probability.  For the Five-Qubit Code we encode two qubits, and for the Steane's code only a single one.

## Dependencies

We use [Qiskit](https://www.ibm.com/quantum/qiskit), [Qiskit-Aer](https://github.com/Qiskit/qiskit-aer), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org), and [pylatexenc](https://github.com/phfaist/pylatexenc).  These can be installed with

```
pip install qiskit qiskit_aer numpy matplotlib pylatexenc
```
