// exact-dft — M4 tests: all-lengths driver + Bluestein chirp (paper sec 5).
//
// Differential standards (spec L5, no tolerances anywhere):
//   * bit-exact vs the L1 naive oracle on the full fuzz matrix;
//   * bit-exact chirp path vs direct composite-ring path on overlaps;
//   * oracle-free column identity (driver(e_j) = (zeta_n^{jk})_k) for
//     larger sizes where the O(n^2) oracle in Q(zeta_n) is still cheap but
//     the driver exercises the full chirp machinery;
//   * exact_convolve vs naive linear convolution;
//   * backend-independent literal check (n = 4 -> 1, i, -1, -i), anchoring
//     bit-exact agreement of the int64 and GMP builds on the overlap.
//
// Size policy: the driver's cost is O(L * sum p) ring multiplications in the
// composite ring Q(zeta_D), D = lcm(2n, L), each O(phi(D)^2) rational ops.
// phi(D) is dominated by the part of n coprime to the working length, so
// the GMP build covers n <= 50 for full oracle diffs plus spot sizes 60/63
// (and 120 columns) where n shares factors with L; the int64 build runs the
// same suite through n = 48 (its coefficient frontier — chirp intermediates
// overflow int64 beyond that; GMP removes the width limit only, not the
// quadratic ring cost).
#include "exact/dft.hpp"
#include "exact/lengths.hpp"
#include "exact/oracle_dft.hpp"

#include <array>
#include <cstdio>
#include <random>
#include <vector>

using ed::Cyclotomic;
using ed::Rational;

static int g_checks = 0;
#define CHECK(cond)                                                        \
    do {                                                                   \
        ++g_checks;                                                        \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, \
                         #cond);                                           \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

// Random dense element of Q(zeta_d): independent random rational coefficients
// (a strictly larger input class than rational-complex; exercises the ring).
static Cyclotomic rand_elem(int d, std::mt19937& rng) {
    std::uniform_int_distribution<int64_t> dist(-9, 9);
    Cyclotomic c(d);
    std::vector<Rational> co((size_t)ed::detail::euler_phi(d), Rational(0));
    for (auto& q : co) {
        int64_t den = dist(rng);
        if (den == 0) den = 7;
        q = Rational(dist(rng), den);
    }
    return Cyclotomic(d, std::move(co));
}

// Random input vector of length n in ring n (oracle's ring).
static std::vector<Cyclotomic> rand_vec_ring_n(int n, std::mt19937& rng) {
    std::vector<Cyclotomic> v;
    v.reserve(n);
    const bool dense = n % 2 == 1;  // odd rings: dense rational elements
    for (int i = 0; i < n; ++i) {
        if (dense) {
            v.push_back(rand_elem(n, rng));
        } else {
            // even n: 4 | 2n always, host i in Q(zeta_n) when possible,
            // else plain rationals
            if (n % 4 == 0) {
                std::uniform_int_distribution<int64_t> dist(-9, 9);
                v.push_back(ed::embed_complex(n, Rational(dist(rng), 7),
                                              Rational(dist(rng), 7)));
            } else {
                std::uniform_int_distribution<int64_t> dist(-9, 9);
                v.push_back(
                    Cyclotomic::rational(n, Rational(dist(rng), 5)) +
                    Cyclotomic::rational(n, Rational(dist(rng), 5)) *
                        Cyclotomic::zeta(n).pow((i * 3) % n));
            }
        }
    }
    return v;
}

// ---------------------------------------------------------------------------
// Working-length selection invariants (paper sec 5).
// ---------------------------------------------------------------------------
static void test_working_length() {
    for (int n = 1; n <= 300; ++n) {
        auto w = ed::working_length(n);
        CHECK(2 * n <= w.L && w.L < 4 * n);
        long long R = 1;
        for (size_t i = 0; i < w.odd_primes.size(); ++i) {
            CHECK(ed::is_prime_ti(w.odd_primes[i]));
            R *= w.odd_primes[i];
        }
        CHECK(R == w.R_odd);
        CHECK(w.R_odd % 2 == 1);
        CHECK(w.L == (int)(w.R_odd << w.two_exp));
        // minimality of e and maximality of ell
        CHECK(w.L / 2 < 2 * n);
        long long next = 1;
        for (int p : w.odd_primes) next *= p;
        // next odd prime after the last one would exceed the bound
        int np = w.odd_primes.empty() ? 3 : w.odd_primes.back() + 2;
        while (!ed::is_prime_ti(np)) np += 2;
        CHECK(next * np > 2 * n);
        ++g_checks;
    }
}

// Backend-independent literal anchor: F_4 * (0,1,0,0) = (1, i, -1, -i)
// (column 1 of F_4 = powers of zeta_4).
// Ring of the driver: L = 12 (R=3, e=2), D = lcm(8, 12) = 24; i = zeta_24^6.
static void test_literal_n4() {
    std::vector<Cyclotomic> x(4, Cyclotomic(4));
    x[1] = Cyclotomic::integer(4, 1);
    auto y = ed::exact_dft(4, x);
    const Cyclotomic z = Cyclotomic::zeta(24);
    CHECK(y[0] == Cyclotomic::integer(24, 1));
    CHECK(y[1] == z.pow(6));
    CHECK(y[2] == z.pow(12));
    CHECK(y[3] == z.pow(18));
}

// ---------------------------------------------------------------------------
// Full differential vs oracle: driver == embed(oracle), bit-exact.
// ---------------------------------------------------------------------------
static void diff_one(int n, std::mt19937& rng) {
    auto x = rand_vec_ring_n(n, rng);
    auto want = ed::oracle_dft(n, x);
    ed::ExactDFT drv(n);
    auto y = drv.dft(x);
    CHECK((int)y.size() == n);
    for (int k = 0; k < n; ++k) {
        auto wk = ed::embed(want[k], drv.ring());
        if (!(y[k] == wk)) {
            std::fprintf(stderr, "driver/oracle mismatch n=%d k=%d\n", n, k);
            CHECK(false);
        }
    }
    ++g_checks;
}

static void test_oracle_diff() {
    std::mt19937 rng(20261011);
    for (int n = 1; n <= 24; ++n) diff_one(n, rng);
    for (int n : {28, 30, 32, 36, 40, 42, 45, 48}) diff_one(n, rng);
    if (ed::kGMPBackend)
        diff_one(50, rng);  // beyond int64 build's list; n=54 would be phi(D)=864, ~minutes — skipped (runtime)
}

// ---------------------------------------------------------------------------
// Chirp path vs direct composite-ring path (both bit-exact, rings differ).
// ---------------------------------------------------------------------------
static void test_chirp_vs_direct() {
    std::mt19937 rng(4242);
    for (int n : {2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 21, 24, 25, 27,
                  30, 32, 35, 36, 40, 42, 45, 48, 50, 60}) {
        if (!ed::kGMPBackend && n > 48) continue;
        auto x = rand_vec_ring_n(n, rng);
        auto direct = ed::direct_dft(n, x);            // ring n
        ed::ExactDFT drv(n);
        auto y = drv.dft(x);                           // ring D
        for (int k = 0; k < n; ++k)
            CHECK(y[k] == ed::embed(direct[k], drv.ring()));
        ++g_checks;
    }
}

// ---------------------------------------------------------------------------
// Oracle-free column identity at larger sizes: driver(e_j) = (zeta^{jk})_k.
// Exercises the full chirp pipeline where the O(n^2)-in-Q(zeta_n) oracle is
// unnecessary; columns are computed directly from the driver's ring root.
// ---------------------------------------------------------------------------
static void column_check(int n, const std::vector<int>& cols) {
    ed::ExactDFT drv(n);
    const Cyclotomic zn = Cyclotomic::zeta(drv.ring()).pow(drv.ring() / n);  // zeta_n in ring D
    for (int j : cols) {
        std::vector<Cyclotomic> e(n, Cyclotomic(n));
        e[j] = Cyclotomic::integer(n, 1);
        auto y = drv.dft(e);
        for (int k = 0; k < n; ++k)
            CHECK(y[k] == zn.pow(((int64_t)j * k) % n));
        ++g_checks;
    }
}

static void test_columns() {
    // 64/81 would need ring D = lcm(128,210) resp. lcm(162,420) with
    // phi 3072/2592: O(phi^2) ring multiplication makes those hours — the
    // recorded runtime observation (see M4 report). Sizes here share
    // factors with the working length, keeping D small while still
    // exercising the full chirp pipeline beyond the oracle-diff range.
    column_check(60, {0, 1, 59});
    if (ed::kGMPBackend)
        column_check(63, {0, 1, 62});  // D = 1260, phi = 288
}

// ---------------------------------------------------------------------------
// Exact convolution vs naive linear convolution.
// ---------------------------------------------------------------------------
static void test_convolve() {
    std::mt19937 rng(99);
    // Inputs in small rings (rational ring 1, and ring 8 where 8 | L keeps D = L):
    // the convolution needs only the working-length ring, not zeta_{2n}.
    for (auto& t : std::vector<std::array<int, 3>>{
             {1, 1, 1}, {2, 3, 1}, {5, 4, 1}, {7, 7, 4}, {12, 9, 1}, {16, 16, 1},
             {20, 31, 1}, {40, 40, 1}}) {
        const int la = t[0], lb = t[1], r = t[2];
        std::vector<Cyclotomic> av, bv;
        auto rc = [&](int i) {
            return Cyclotomic::rational(r, Rational((int)(rng() % 13) - 6, 5));
        };
        for (int i = 0; i < la; ++i) av.push_back(rc(i));
        for (int i = 0; i < lb; ++i) bv.push_back(rc(i));
        auto want = ed::oracle_conv_linear(av, bv);
        auto got = ed::exact_convolve(av, bv);
        CHECK(got.size() == want.size());
        const int D = got.empty() ? 1 : got[0].d();
        for (size_t k = 0; k < want.size(); ++k)
            CHECK(got[k] == ed::embed(want[k], D));
        ++g_checks;
    }
}

int main() {
    test_working_length();
    test_literal_n4();
    test_oracle_diff();
    test_chirp_vs_direct();
    test_columns();
    test_convolve();
    std::printf("test_driver: OK (%d checks; backend=%s)\n", g_checks,
                ed::kGMPBackend ? "GMP" : "int64");
    return 0;
}
