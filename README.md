# exact-dft

Component-level **exact** implementation of the constructions in OpenAI's
*"An Explicit Power-Saving for the Exact Discrete Fourier Transform"*
(Sept 25, 2026, `openai/math` preprint).

**Honest positioning.** This is a faithful reference implementation of the
paper's *components* under exact algebraic arithmetic (cyclotomic fields
Q(zeta_d) with rational coefficient vectors). It does **not** reproduce the
paper's asymptotic power saving — the constants are galactic (m = 10^6,
W* = 2^71; the authors themselves state there is "no useful crossover
estimate" and disclaim floating-point-stability and bit-complexity
implications). It is slower than any real FFT library. Its value is
exactness, structural fidelity, and pedagogy.

## Status

- **M1 (done):** L0 exact rational + cyclotomic ring Q(zeta_d) (Phi_d via
  recursive division, differentially tested against an independent
  Mobius-formula table), L1 naive exact O(n^2) DFT oracle + linear
  convolution oracle. All tests green.
- M2: Newton–Toeplitz exact-width compiler with borrowed-coordinate replay.
- M3: mixed-radix sector-packing traversal.
- M4: all-lengths driver (prime products + Bluestein chirp).
- M5: scaled tensor network demo.
- M6: full pybind11 surface + packaging.

## Build & test (C++ layers)

```bash
cmake -S . -B build -DEXACT_DFT_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Python module (M6, works already in skeleton form):

```bash
pip install .   # needs pybind11 + scikit-build-core
```

## License

MIT.
