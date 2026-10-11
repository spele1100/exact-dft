"""exact-dft: exact DFT over cyclotomic fields.

Component-level reference implementation of the constructions in OpenAI's
*"An Explicit Power-Saving for the Exact Discrete Fourier Transform"*
(Sept 25, 2026): exact Newton/Toeplitz compilation, sector-packing
synchronization, prime-product working lengths with Bluestein chirp, and a
scaled demo of the sec-2 tensor network with an honest lambda(h) budget.

Everything here is EXACT (cyclotomic arithmetic, bit-exact differential
tests against a naive oracle). It does not reproduce the paper's asymptotic
power saving — the constants are galactic and the paper itself gives "no
useful crossover estimate". It is slower than any real FFT library. Its
value: exactness, structural fidelity, and pedagogy.

Quickstart:
    >>> import exact_dft
    >>> y = exact_dft.exact_dft([1, 2, 3, 4])   # exact F_4 x, no rounding
    >>> y[1].as_complex()
    (-2+2j)                                       # exact: -2 + 2i

See examples/exact_demo.py and the README for the full surface.
"""
from ._core import (  # noqa: F401
    Cyclotomic,
    zeta,
    gauss,
    exact_dft,
    exact_convolve,
    oracle_dft,
    working_length,
    compile_report,
    network_report,
    network_exchange_demo,
    gmp_backend,
)

__version__ = "0.1.0"
__all__ = [
    "Cyclotomic",
    "zeta",
    "gauss",
    "exact_dft",
    "exact_convolve",
    "oracle_dft",
    "working_length",
    "compile_report",
    "network_report",
    "network_exchange_demo",
    "gmp_backend",
]
