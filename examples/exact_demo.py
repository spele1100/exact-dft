#!/usr/bin/env python3
"""exact-dft demo: exact DFTs + the scaled sec-2 network budget.

Runs (no floating point anywhere in the transform pipeline):

  1. A length-n exact DFT of a rational-complex vector through the full
     pipeline: Newton/Toeplitz compiled axes, sector-packing scheduling,
     working length L in [2n, 4n), Bluestein chirp. Outputs are printed as
     exact fractions and y[0] is verified to equal sum(x) BIT-EXACTLY.

  2. A second length (odd, exercising the chirp on coprime mixes).

  3. The scaled tensor-network demo (paper sec 2): the scalar bank exchange
     with garbage-filled auxiliary wires + the honest lambda(h) budget —
     the headline number is that there is NO power saving at feasible h.

Usage:
    python exact_demo.py [n1 n2]
Defaults (16, 45) finish in seconds on the default int64 build; n <= 48 is
the int64 frontier, larger n needs the GMP build (-DEXACT_DFT_USE_GMP=ON).
"""
import sys

import exact_dft as ed


def show(c, label):
    parts = [f"({num}/{den})" + ("" if i == 0 else ("z" if i == 1 else f"z^{i}"))
             for i, (num, den) in enumerate(c.coeffs()) if num != 0]
    print(f"    {label} = " + (" + ".join(parts) or "0") + f"   [in Q(zeta_{c.d})]")


def demo_dft(n):
    print(f"\n=== Exact DFT, n = {n} ===")
    wl = ed.working_length(n)
    radices = " * ".join(map(str, wl["odd_primes"])) or "1"
    print(f"  working length L = {wl['L']} = ({radices}) * 2^{wl['two_exp']}"
          f"   (rule: 2n <= L < 4n)")
    rep = ed.compile_report(n)
    print(f"  composite ring Q(zeta_D), D = {rep['ring']}, phi(D) = {rep['phi_ring']},"
          f" compiled axes r = {[a['r'] for a in rep['axes']]}")

    x = [complex(i % 5 - 2, (3 * i) % 7 - 3) for i in range(n)]  # integers
    y = ed.exact_dft(x)

    # Bit-exact sanity identity: y[0] = sum_j x_j * zeta_n^{j*0} = sum_j x_j,
    # and y[1] = sum_j x_j zeta_n^j (reconstructed exactly in the driver ring).
    D = y[0].d
    z = ed.zeta(D)
    zn = z.pow(D // n)  # primitive n-th root inside Q(zeta_D)
    sum_all = ed.gauss(D, 0.0, 0.0)
    sum_rot = ed.gauss(D, 0.0, 0.0)
    zp = ed.gauss(D, 1.0, 0.0)
    for c in x:
        sum_all = sum_all + ed.gauss(D, c.real, c.imag)
    for c in x:
        sum_rot = sum_rot + ed.gauss(D, c.real, c.imag) * zp
        zp = zp * zn
    assert y[0] == sum_all, "y[0] != sum(x) — exactness violated"
    assert y[1] == sum_rot, "y[1] != sum_j x_j zeta^j — exactness violated"
    print("  y[0] == sum(x) and y[1] == sum_j x_j*zeta^j: verified BIT-EXACTLY")

    print("  first exact outputs:")
    for k in range(min(4, n)):
        show(y[k], f"y[{k}]")
    return y


def demo_network(h=4):
    print(f"\n=== Scaled sec-2 tensor network, h = {h} ===")
    demo = ed.network_exchange_demo(h)
    print(f"  bank exchange: v = C({h},3) = {demo['v']} triples per bank axis, "
          f"{demo['bank_entries']} bank entries,\n"
          f"  {demo['invocations']} invocations of the 8-row schedule")
    print(f"  (X, Y) -> (-Y, X) with GARBAGE in every auxiliary wire, "
          f"verified exactly: {demo['verified_exact']}")

    rep = ed.network_report(h)
    print("  dimension budget (paper eqs net:wire-count / net:s / net:delta /"
          " net:padding):")
    for line in rep["lines"]:
        print("    " + line)
    print("""
  HONEST SCALING STATEMENT: at h = 4 (and at every h <= 21) Delta <= 0 —
  this network needs MORE directional steps than the trivial per-direction
  baseline. The paper's saving first appears at h = 22 and at the paper's
  h = 100 it is theta = 1 - 2.1e-13 with m = h^3 = 10^6 (arrays of 2^10^6
  entries) and W* = 2^71 wires. The paper itself gives "no useful crossover
  estimate" and disclaims floating-point-stability and bit-complexity
  implications. This demo certifies the MECHANISM is exact — never that it
  is fast.""")


def main():
    args = [int(a) for a in sys.argv[1:]]
    n1, n2 = args[:2] if len(args) >= 2 else (16, 45)
    print(f"exact-dft {ed.__version__} demo   (GMP backend: {ed.gmp_backend})")
    demo_dft(n1)
    demo_dft(n2)
    demo_network(4)
    print("\nAll results exact. No floating point was used in any transform.")


if __name__ == "__main__":
    main()
