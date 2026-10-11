// exact-dft — M1 tests: L0 cyclotomic ring Q(zeta_d).
// Differential vs independent Mobius-formula ground truth (phi_table.hpp).
#include "exact/cyclotomic.hpp"
#include "exact/oracle_dft.hpp"
#include "phi_table.hpp"

#include <cassert>
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

static void test_phi_against_table() {
    for (const auto& entry : ed::test::phi_table()) {
        const int d = entry.d;
        const Cyclotomic z(d);
        const auto& impl = z.phi();  // dense monic Zpoly
        // degree == phi(d) cross-check (Euler totient closed form)
        CHECK((int64_t)impl.size() - 1 == ed::detail::euler_phi(d));
        // coefficient-by-coefficient vs Mobius-formula table (sparse)
        std::vector<int64_t> dense(impl.size(), 0);
        for (const auto& t : entry.terms) {
            CHECK(t.idx < (int)dense.size());
            CHECK(t.den == 1);  // Phi_d is monic integer
            dense[t.idx] = t.num;
        }
        CHECK((int64_t)dense.size() == (int64_t)impl.size());
        for (size_t i = 0; i < impl.size(); ++i) CHECK(dense[i] == impl[i]);
        ++g_checks;
    }
}

static void test_zeta_order_and_primitivity() {
    for (int d = 1; d <= 100; ++d) {
        const Cyclotomic z = Cyclotomic::zeta(d);
        CHECK(z.pow(d) == Cyclotomic::integer(d, 1));  // zeta^d = 1
        for (int k = 1; k < d; ++k)                    // primitivity: t^k != 1
            CHECK(z.pow(k) != Cyclotomic::integer(d, 1));
    }
    // spot larger orders incl. the cap region behavior (correctness, not perf)
    for (int d : {128, 210, 256, 360, 512, 1024, 2310}) {
        const Cyclotomic z = Cyclotomic::zeta(d);
        CHECK(z.pow(d) == Cyclotomic::integer(d, 1));
        // primitivity: only need to rule out proper divisors' quotient orders
        for (int k : ed::detail::divisors_of(d)) {
            if (k == d) continue;
            CHECK(z.pow(k) != Cyclotomic::integer(d, 1));
        }
    }
}

static Cyclotomic rand_elem(int d, std::mt19937& rng) {
    // Denominators drawn from 1..6 so every denominator in sight divides
    // lcm(1..6) = 60: chained products stay comfortably inside int64. (With
    // random denominators up to 24 the *exact reduced* coefficients have
    // legitimately astronomical denominators — coefficient growth is a
    // documented spec risk; int64 + bounded fuzz is the M1 stance, GMP the
    // later fallback.)
    std::uniform_int_distribution<int64_t> num_dist(-24, 24);
    std::uniform_int_distribution<int64_t> den_dist(1, 6);
    std::vector<Rational> c((size_t)ed::detail::euler_phi(d));
    for (auto& q : c) q = Rational(num_dist(rng), den_dist(rng));
    return Cyclotomic(d, std::move(c));
}

static void test_ring_axioms_fuzz() {
    std::mt19937 rng(777);
    for (int d : {1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 20, 24, 30}) {
        for (int iter = 0; iter < 200; ++iter) {
            Cyclotomic a = rand_elem(d, rng), b = rand_elem(d, rng),
                        c = rand_elem(d, rng);
            CHECK(a + b == b + a);
            CHECK(a * b == b * a);
            CHECK((a + b) + c == a + (b + c));
            CHECK((a * b) * c == a * (b * c));
            CHECK(a * (b + c) == a * b + a * c);
            CHECK(a * Cyclotomic::integer(d, 1) == a);
            CHECK(a + Cyclotomic::integer(d, 0) == a);
            CHECK(a - a == Cyclotomic::integer(d, 0));
        }
    }
}

static void test_reduction_idempotent() {
    // Reduction mod Phi_d of an already-reduced vector is a no-op (round trip).
    std::mt19937 rng(99);
    for (int d : {2, 3, 5, 8, 12, 30, 60}) {
        const Cyclotomic a = rand_elem(d, rng);
        std::vector<Rational> again = a.reduce(a.coeffs());
        CHECK(again == a.coeffs());
    }
    // Sanity: t^phi-related identity, e.g. in Q(zeta_5): t^4 = -(1+t+t^2+t^3)
    const Cyclotomic z5 = Cyclotomic::zeta(5);
    Cyclotomic sum(5);
    for (int k = 0; k < 5; ++k) sum += z5.pow(k);
    CHECK(sum.is_zero());  // 1 + t + ... + t^4 = 0 in Q(zeta_5)
}

static void test_ring_mixing_rejected() {
    bool threw = false;
    try {
        Cyclotomic::zeta(3) + Cyclotomic::zeta(5);
    } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { Cyclotomic::zeta(0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

static void test_embedding() {
    // i embeds as zeta_4: zeta(4)^1 = i
    const Cyclotomic i = Cyclotomic::zeta(4);
    CHECK(i * i == Cyclotomic::integer(4, -1));
    CHECK(ed::embed_complex(8, Rational(1), Rational(1)) ==
          Cyclotomic::rational(8, Rational(1)) + Cyclotomic::zeta(8).pow(2));
}

int main() {
    test_phi_against_table();
    test_zeta_order_and_primitivity();
    test_ring_axioms_fuzz();
    test_reduction_idempotent();
    test_ring_mixing_rejected();
    test_embedding();
    std::printf("test_cyclotomic: OK (%d checks)\n", g_checks);
    return 0;
}
