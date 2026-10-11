# exact-dft

**Exact discrete Fourier transforms over cyclotomic fields** — a
component-level reference implementation of the constructions in OpenAI's
paper *"An Explicit Power-Saving for the Exact Discrete Fourier Transform"*
(Sept 25, 2026, `openai/math` preprint).

**Honest positioning, up front.** This package computes DFTs *exactly* —
bit-exact rational arithmetic in cyclotomic rings ℚ(ζ_d), no floating point,
no rounding, ever. It does **not** reproduce the paper's asymptotic power
saving: the constants are galactic (m = 10⁶, W\* = 2⁷¹, base network
dimension ~4×10¹⁵) and the paper itself states there is *"no useful
crossover estimate"* and explicitly disclaims floating-point-stability and
bit-complexity implications. `exact-dft` is slower than any real FFT
library, and always will be at feasible sizes. Its value is **exactness,
structural fidelity to the paper, and pedagogy**.

## What's implemented (layer by layer, mirroring the paper)

| Layer | Paper section | What it does |
|---|---|---|
| L0 | — | Exact rationals (int64/int128 primary, GMP optional via `-DEXACT_DFT_USE_GMP=ON`), cyclotomic ring ℚ(ζ_d) = ℚ[t]/(Φ_d), Φ_d via recursive division, differentially tested against an independent Möbius-formula table |
| L1 | — | Naive exact O(n²) DFT oracle — *the* bit-exact reference for every later layer |
| L2 | §3 | Newton–Toeplitz exact-width compiler: F_r(u) = N·D_N⁻¹·Nᵀ, triangular Toeplitz words on exactly r coordinates, **borrowed-coordinate Bennett replay** restoring garbage scratch (the paper's distinctive lemma, fuzz-tested with adversarial scratch contents) |
| L3 | §4 | Synchronization: mixed-radix digit DFS (< 2R prefix nodes), sector packing with the four-state S/Q/o/I update in O(R), scheduled tensor transform (⊗_j F_{r_j}) over composite rings |
| L5 | §5 | All-lengths driver: odd-prime prefix product R ≤ 2n + power-of-two padding to working length **L ∈ [2n, 4n)**, Good–Thomas coprime-axis assembly, exact radix-2 for the 2-axis, **Bluestein chirp** (via ζ_{2n} directly — algebraically equivalent simplification), exact convolution (Cor. 1.3) |
| L4 | §2 | Scaled tensor-network demo: the verbatim 8-row invocation schedule and 3-stage bank exchange (h configurable, default 4), exact frame-change algebra (Φ_U, directional kernels, Lemma net:edge), and the honest **λ(h)/Δ(h) dimension budget** reproducing the paper's h = 100 integers exactly |
| L6 | — | pybind11 bindings + packaging (this Python package) |

Every layer has its own CTest suite; all differential tests are **bit-exact
against the oracle — no tolerances anywhere in the exact path**.

## Installation

```bash
pip install .            # int64 backend, no dependencies beyond the build
```

Large n (roughly n > 48) overflows int64 coefficient widths; build with GMP
for the big-number backend:

```bash
pip install . --config-settings=cmake.define.EXACT_DFT_USE_GMP=ON  # needs libgmp-dev
```

## Quickstart

```python
>>> import exact_dft as ed
>>> y = ed.exact_dft([1, 2, 3, 4])          # exact F_4·x
>>> y[0].as_complex()                        # Gaussian rationals only
(10+0j)                                      # exact: sum of inputs
>>> y[1].as_complex()
(-2+2j)                                      # exact: -2 + 2i, not a float
>>> ed.exact_dft([1, 1, 1])[1].coeffs()      # general outputs live in Q(zeta_n)
[(0, 1), (1, 1)]                             # = zeta_3, exactly
```

Complex inputs are converted **exactly**: every finite double is a dyadic
rational p·2^e, so the conversion is lossless (0.1 becomes the exact binary
value `3602879701896397/36028797018963968`, which is what "exact" requires).

General ring elements:

```python
>>> z = ed.zeta(5)            # primitive 5th root in Q(zeta_5)
>>> (z.pow(5)).is_zero()      # zeta^5 = 1 exactly
False                                         # = 1, not zero -- correct!
>>> ed.oracle_dft(5, [z, ed.Cyclotomic(5), ed.Cyclotomic(5), ed.Cyclotomic(5), ed.Cyclotomic(5)])[0] == z
True
```

The scaled network demo (paper §2):

```python
>>> ed.network_exchange_demo(4)   # bank exchange with GARBAGE aux wires
{'h': 4, 'v': 4, 'bank_entries': 64, 'invocations': 48, 'verified_exact': True}
>>> rep = ed.network_report(4)     # lambda(h) budget
>>> rep['Delta'], rep['verdict']   # Delta = -1792 < 0 ...
(-1792, 'NO power saving at this h (lambda > m, theta > 1)')
```

Full runnable demo: `python examples/exact_demo.py` (exact DFT n = 16 and
n = 45 + the network budget, printed nicely).

## The honest λ(h) report (paper §2.5)

`network_report(h)` computes the paper's dimension budget exactly (in
128-bit integers): W (wires), s (directional steps),
**Δ = Wm − s = 2v²(v − 3h(h+1))** with v = C(h,3), m = h³, the 2-power
padding W\*, and **λ(h) = m − Δ/W\***, θ = log_m λ. At the paper's
h = 100 this reproduces their integers character-for-character:

- W = 1873807244643542670000, Δ = 6871402692000000, W\* = 2⁷¹
- λ = 999999.99999709…, θ = 0.99999999999978939 (= 1 − 2.1×10⁻¹³)

**and there is no saving at feasible sizes**: Δ > 0 first requires
v > 3h(h+1), i.e. h ≥ 22 (C(22,3) = 1540 > 1518). At the default h = 4 the
network *loses*: λ = 65.75 > m = 64, θ = 1.0065. Even where Δ > 0,
executing the frame-change layer needs arrays of 2^{h³} entries. This is an
asymptotic existence mechanism, demonstrated exactly — never a speedup.
The repo's tests verify the frame-change *algebra* (Lemma net:edge, the
terminal ratios Φ_𝒟 = C^{⊗m} and the exceptional-route R_u·C^{⊗m}) exactly
at test scale m ∈ {4, 6, 8} over ℚ(i).

## Performance: what to honestly expect

Exact means dense rational arithmetic in ℚ(ζ_D) with D = lcm(2n, L) — ring
multiplication is O(φ(D)²) rational ops, and coefficients *grow*. Practical
frontiers (measured on the M-era test box, single core):

- **int64 backend**: exact DFTs safe through **n ≈ 48** (chirp
  intermediates overflow int64 beyond; the L2 compiler is safe for
  widths r ≤ 64).
- **GMP backend**: removes the width limit but not the quadratic ring cost —
  n = 54 takes ~130 s, n = 101 takes hours. The frontier is the
  quadratic-ring cost, not the register width.
- **Working-length rule**: L ∈ [2n, 4n) always (prime prefix + one
  power of two), exactly as the paper's elementary bounds require.

If you need fast approximate DFTs, use `numpy.fft` or FFTW. If you need to
*prove* something about a transform of a small rational vector — or study
the paper's constructions — this is the tool.

## API surface

- `exact_dft(x)` — exact F_n·x; `x` a list of complex (exact dyadic
  conversion) or `Cyclotomic` elements (mixed rings allowed; the transform
  ring is the lcm closure)
- `oracle_dft(n, x)` — naive exact reference (inputs in ring n)
- `exact_convolve(a, b)` — exact linear convolution (Corollary 1.3)
- `zeta(d)`, `gauss(d, re, im)`, `Cyclotomic` (with `+ − ×`, `pow`, `==`,
  `coeffs()`, `as_complex()` for Gaussian rationals)
- `working_length(n)`, `compile_report(n)` — structure / op-count
  introspection
- `network_report(h)`, `network_exchange_demo(h)` — §2 budget and demo

## Building & testing from source (C++ layers)

```bash
cmake -S . -B build -DEXACT_DFT_BUILD_TESTS=ON
cmake --build build && ctest --test-dir build --output-on-failure

# GMP backend config (big rationals):
cmake -S . -B build-gmp -DEXACT_DFT_BUILD_TESTS=ON -DEXACT_DFT_USE_GMP=ON
cmake --build build-gmp && ctest --test-dir build-gmp
```

Both backends run the same suites against the same bit-exact oracle, so
int64/GMP agreement on the overlap range is enforced by construction.

## Deviations from the paper (documented judgment calls)

1. **Galactic constants → configurable h** (default 4) for the §2 network;
   the saving is *reported* (λ(h)), never claimed. Full telescoping over
   2^m-entry arrays is not executed (galactic); its two exactness-critical
   ingredients — the scalar exchange with arbitrary auxiliaries and the
   frame-change algebra — are implemented and tested exactly.
2. **Exactness model**: the paper's exact-ℂ model is realized concretely as
   ℚ(ζ_d) arithmetic; inputs outside the cyclotomic closure are exact
   dyadic conversions of doubles (lossless) — never silently rounded.
3. **L2 execution**: the replay/borrowed-coordinate structure is kept (it
   is the correctness-critical part); the fixed slot-pattern padding needed
   only for cross-width synchronization proofs is dropped (the scheduler
   consumes per-width words directly).
4. **Chirp via ζ_{2n} directly** rather than the paper's working-length
   formulation — algebraically equivalent, exactness unaffected.
5. **No bit-complexity or stability claims** — per the paper's own
   disclaimer.

## License

MIT. The paper is cited, not reproduced; see the
[`openai/math`](https://github.com/openai/math) repository for the original.
