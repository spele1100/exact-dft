// exact-dft — M2 tests: Newton/Toeplitz exact-width compiler + replay.
// Differential vs oracle_dft: BIT-EXACT equality, no tolerances (spec L2).
#include "exact/newton_toeplitz.hpp"
#include "exact/oracle_dft.hpp"

#include <cstdio>
#include <random>
#include <set>
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

// Deterministic random vector in Q(zeta_r): a + b*zeta^c (covers rationals
// and zeta-multiples; no need for i in the ring).
static std::vector<Cyclotomic> rand_vec(int r, std::mt19937& rng) {
    std::uniform_int_distribution<int64_t> num(-15, 15);
    std::uniform_int_distribution<int> den(1, 9);
    std::uniform_int_distribution<int> expo(0, r - 1);
    std::vector<Cyclotomic> v((size_t)r, Cyclotomic(r));
    const Cyclotomic z = Cyclotomic::zeta(r);
    for (int i = 0; i < r; ++i)
        v[i] = Cyclotomic::rational(r, Rational(num(rng), den(rng))) +
               Cyclotomic::rational(r, Rational(num(rng), den(rng))) * z.pow(expo(rng));
    return v;
}

// ---------------------------------------------------------------------------
static void test_inverse() {
    for (int d : {3, 4, 5, 8, 12, 16}) {
        const Cyclotomic z = Cyclotomic::zeta(d);
        CHECK(z * ed::cyclotomic_inverse(z) == Cyclotomic::integer(d, 1));
        CHECK(ed::cyclotomic_inverse(ed::cyclotomic_inverse(z)) == z);
    }
    // NOTE: the generic elimination-based inverse overflows int64 for dense
    // random elements around d ~ 12-16 (documented fragility); the compiler
    // constructor uses the closed form inverse_one_minus instead. Exercise it
    // only on well-conditioned elements here.
    for (int d : {5, 9, 12}) {
        const Cyclotomic z = Cyclotomic::zeta(d);
        for (auto c : {Cyclotomic::rational(d, Rational(2)) + Cyclotomic::rational(d, Rational(3)) * z,
                       Cyclotomic::rational(d, Rational(1, 7)) - z * z,
                       z + z.pow(d - 1) + Cyclotomic::rational(d, Rational(5, 3))})
            CHECK(c * ed::cyclotomic_inverse(c) == Cyclotomic::integer(d, 1));
    }
    // Closed form for 1/(1 - zeta^s): check against the generic inverse.
    for (int d : {5, 8, 9, 12}) {
        const Cyclotomic u = Cyclotomic::zeta(d);
        for (int s = 1; s < d; ++s) {
            const Cyclotomic om = Cyclotomic::integer(d, 1) - u.pow(s);
            CHECK(ed::inverse_one_minus(u, s) * om == Cyclotomic::integer(d, 1));
        }
    }
    bool threw = false;
    try { ed::cyclotomic_inverse(Cyclotomic(5)); } catch (...) { threw = true; }
    CHECK(threw);
}

// t_j = H_j^{-1} (ring inverse) and g = series reciprocal of t:
// (t * g)_k = delta_{k0} for k < r  =>  g inverts every triangular block.
static void test_reciprocal_roundtrip() {
    for (int r = 2; r <= 16; ++r) {
        ed::NewtonToeplitz nt(r);
        for (int j = 0; j < r; ++j)
            CHECK(nt.t()[j] * nt.H()[j] == Cyclotomic::integer(r, 1));
        for (int k = 0; k < r; ++k) {
            Cyclotomic conv(r);
            for (int j = 0; j <= k; ++j) conv += nt.t()[j] * nt.g()[k - j];
            CHECK(conv == Cyclotomic::integer(r, k == 0 ? 1 : 0));
        }
    }
}

// Newton matrix identity: N = diag(H) T diag((-1)^k u^{k(k-1)/2}) equals
// N_ik = prod_{j<k} (u^i - u^j)  (paper eq loc:newton), small r.
static void test_newton_matrix() {
    for (int r = 2; r <= 8; ++r) {
        ed::NewtonToeplitz nt(r);
        const Cyclotomic u = Cyclotomic::zeta(r);
        // D_N^{-1} entries invert D_N exactly.
        for (int k = 0; k < r; ++k)
            CHECK(nt.DNinv()[k] * (nt.H()[k] * nt.D2()[k]) == Cyclotomic::integer(r, 1));
        for (int i = 0; i < r; ++i)
            for (int k = 0; k < r; ++k) {
                Cyclotomic nik(r);
                if (i >= k) {
                    nik = Cyclotomic::integer(r, 1);
                    for (int j = 0; j < k; ++j) nik = nik * (u.pow(i) - u.pow(j));
                }
                // N_ik = H_i * t_{i-k} * D2_k
                Cyclotomic got = (i >= k) ? nt.H()[i] * nt.t()[i - k] * nt.D2()[k]
                                           : Cyclotomic(r);
                CHECK(got == nik);
            }
    }
}

// Lower triangular Toeplitz vs direct triangular matvec, bit-exact.
static void test_toeplitz_vs_direct() {
    std::mt19937 rng(99);
    for (int r : {2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 20, 24, 32, 40, 48}) {
        ed::NewtonToeplitz nt(r);
        auto x = rand_vec(r, rng);
        auto got = x;
        nt.toeplitz_lower(got);
        for (int i = 0; i < r; ++i) {
            Cyclotomic want(r);
            for (int j = 0; j <= i; ++j) want += nt.t()[i - j] * x[j];
            CHECK(got[i] == want);
        }
    }
}

// Replay unit test on a hand-built DAG: y += Mx, z restored bit-exact,
// including z correlated with x (paper: arbitrary borrowed contents).
static void test_replay_garbage() {
    const int d = 5;
    const int E = 4, A = 3, G = 6;  // sources, targets, gates
    // arena: [0,E) = x, [E, E+A) = y, [E+A, E+A+G) = z (borrowed, garbage).
    const int ZOFF = E + A;
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int64_t> dist(-9, 9);
    auto rc = [&] { return Cyclotomic::rational(d, Rational(dist(rng), 3)); };

    for (int trial = 0; trial < 8; ++trial) {
        // Random linear DAG: gates 0..2 read sources only; gates 3..5 read a
        // source and an earlier gate; outputs read gates 3..5.
        struct P { int coord; bool from_source; Cyclotomic c; };
        std::vector<std::vector<P>> gp(G);
        for (int g = 0; g < G; ++g) {
            gp[g].push_back({(int)((dist(rng) % E + E) % E), true, rc()});
            if (g < 3) gp[g].push_back({(int)((dist(rng) % E + E) % E), true, rc()});
            else       gp[g].push_back({ZOFF + (g - 3), false, rc()});
        }
        struct O { int target; int coord; bool from_source; Cyclotomic c; };
        std::vector<O> outs;
        for (int i = 0; i < A; ++i) outs.push_back({E + i, ZOFF + 3 + i, false, rc()});

        std::vector<Cyclotomic> arena;
        for (int i = 0; i < E + A; ++i) arena.push_back(rc());
        // Adversarial garbage in the borrowed coordinates.
        for (int g = 0; g < G; ++g) {
            if (trial % 4 == 2) arena.push_back(arena[g % E]);              // copy of x
            else if (trial % 4 == 3) arena.push_back(arena[E + g % A] * arena[(g + 1) % E]);
            else if (trial == 6) arena.push_back(Cyclotomic::rational(d, Rational(999999, 7)));
            else arena.push_back(rc());                                     // random
        }
        auto before = arena;

        // Reference: pure DAG evaluation on zeroed z, added into y.
        std::vector<Cyclotomic> gates(G, Cyclotomic(d));
        for (int g = 0; g < G; ++g)
            for (const P& p : gp[g])
                gates[g] += p.c * (p.from_source ? arena[p.coord] : gates[p.coord - ZOFF]);
        for (const O& o : outs) before[o.target] += o.c * gates[o.coord - ZOFF];

        ed::ReplayDAG dag;
        for (int g = 0; g < G; ++g) {
            dag.gate_coord.push_back(ZOFF + g);
            dag.preds.emplace_back();
            for (const P& p : gp[g])
                dag.preds.back().push_back({p.coord, p.from_source, p.c});
        }
        for (const O& o : outs)
            dag.outs.push_back({o.target, o.coord, o.from_source, o.c});

        ed::NTReport rep;
        dag.run(arena, rep);

        for (int i = 0; i < A; ++i) CHECK(arena[E + i] == before[E + i]);
        for (int i = 0; i < E; ++i) CHECK(arena[i] == before[i]);       // x untouched
        for (int g = 0; g < G; ++g)                                      // z restored
            CHECK(arena[ZOFF + g] == before[ZOFF + g]);
    }
}

// Full pipeline differential: compiled F_r x == oracle, BIT-EXACT.
static void test_differential() {
    std::mt19937 rng(20261011);
    int cases = 0;
    for (int r = 1; r <= 16; ++r) {
        ed::NewtonToeplitz nt(r);
        for (int trial = 0; trial < 3; ++trial) {
            auto x = rand_vec(r, rng);
            auto want = ed::oracle_dft(r, x);
            nt.apply(x);
            for (int k = 0; k < r; ++k) CHECK(x[k] == want[k]);
            ++cases;
        }
    }
    CHECK(cases == 48);
}

static void test_differential_extras() {
    std::mt19937 rng(777);
    // Random dense inputs: int64 exact through r = 48 (growth is
    // input-dependent beyond; see test_growth_frontier).
    for (int r : {17, 19, 23, 25, 27, 29, 31, 32, 36, 40, 44, 48}) {
        ed::NewtonToeplitz nt(r);
        for (int trial = 0; trial < 2; ++trial) {
            auto x = rand_vec(r, rng);
            auto want = ed::oracle_dft(r, x);
            nt.apply(x);
            for (int k = 0; k < r; ++k) CHECK(x[k] == want[k]);
        }
        // basis vector: column of F = powers of zeta.
        const Cyclotomic z = Cyclotomic::zeta(r);
        for (int j : {0, r / 2, r - 1}) {
            std::vector<Cyclotomic> e((size_t)r, Cyclotomic(r));
            e[j] = Cyclotomic::integer(r, 1);
            nt.apply(e);
            for (int k = 0; k < r; ++k) CHECK(e[k] == z.pow(((int64_t)j * k) % r));
        }
    }
    // Mild inputs (small rationals, zeta multiples): exact to the int64
    // frontier r = 64.
    for (int r : {54, 60, 64}) {
        ed::NewtonToeplitz nt(r);
        const Cyclotomic z = Cyclotomic::zeta(r);
        std::vector<Cyclotomic> x((size_t)r, Cyclotomic(r));
        for (int i = 0; i < r; ++i)
            x[i] = Cyclotomic::rational(r, Rational(i % 7 - 3, 5)) +
                   Cyclotomic::rational(r, Rational(1, 3)) * z.pow((i * 5) % r);
        auto want = ed::oracle_dft(r, x);
        nt.apply(x);
        for (int k = 0; k < r; ++k) CHECK(x[k] == want[k]);
    }
}

// Borrowed/shifted scratch: the r-coordinate word embedded at different
// offsets in a larger arena leaves all neighbors untouched (width claim).
static void test_shifted_arena() {
    std::mt19937 rng(55);
    const int r = 12, PADL = 5, PADR = 7;
    ed::NewtonToeplitz nt(r);
    auto x = rand_vec(r, rng);
    auto want = x;
    nt.apply(want);
    for (int shift : {PADL, PADL + 3}) {
        std::vector<Cyclotomic> arena((size_t)(shift + r + PADR), Cyclotomic(r));
        for (int i = 0; i < shift + r + PADR; ++i)
            arena[i] = Cyclotomic::rational(r, Rational((int64_t)(i * 31 % 17) - 8, 3));
        auto garbage = arena;
        for (int i = 0; i < r; ++i) arena[shift + i] = x[i];
        // Apply on the sub-vector (word executes on its own coordinates).
        std::vector<Cyclotomic> sub(arena.begin() + shift, arena.begin() + shift + r);
        nt.apply(sub);
        for (int i = 0; i < r; ++i) {
            CHECK(sub[i] == want[i]);
            arena[shift + i] = sub[i];
        }
        for (int i = 0; i < shift; ++i) CHECK(arena[i] == garbage[i]);
        for (int i = shift + r; i < shift + r + PADR; ++i) CHECK(arena[i] == garbage[i]);
    }
}

// Coefficient growth + int64 frontier probe (informs GMP timing, spec risk).
static void test_growth_frontier() {
    std::printf("  growth frontier (prepared scalars + apply intermediates; %s backend):\n",
                ed::kGMPBackend ? "GMP" : "int64");
    int exact_up_to = 0, first_overflow = -1;
    for (int r : {8, 16, 24, 32, 48, 60, 64, 65, 72, 96, 128}) {
        try {
            auto rep = ed::compile_report(r);
            std::printf("    r=%3d OK   max|num| bits=%lld  max den bits=%lld  (shears=%lld replays=%lld pairs=%lld)\n",
                        r, rep.max_num_bits, rep.max_den_bits,
                        rep.shears, rep.replays, rep.cross_pairs);
            exact_up_to = std::max(exact_up_to, r);
        } catch (const std::overflow_error&) {
            std::printf("    r=%3d OVERFLOW (int64 rationals; GMP option territory)\n", r);
            if (first_overflow < 0) first_overflow = r;
        }
    }
    CHECK(exact_up_to >= 64);  // covers the spec's core exact range
    if (!ed::kGMPBackend)
        CHECK(first_overflow == 65 || first_overflow == -1);  // known int64 frontier
    else
        CHECK(first_overflow == -1);  // GMP: no overflow through r = 128
}

int main() {
    test_inverse();
    test_reciprocal_roundtrip();
    test_newton_matrix();
    test_toeplitz_vs_direct();
    test_replay_garbage();
    test_differential();
    test_differential_extras();
    test_shifted_arena();
    test_growth_frontier();
    std::printf("test_newton_toeplitz: OK (%d checks)\n", g_checks);
    return 0;
}
