// exact-dft — M1 tests: L0 rational arithmetic (plain assert-based, per spec).
#include "exact/rational.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

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

static void test_construction() {
    CHECK(Rational(0).is_zero());
    CHECK(Rational(6, 4) == Rational(3, 2));
    CHECK(Rational(-6, 4) == Rational(-3, 2));
    CHECK(Rational(6, -4) == Rational(-3, 2));
    CHECK(Rational(-6, -4) == Rational(3, 2));
    CHECK(Rational(INT64_MIN + 1).num == INT64_MIN + 1);
    bool threw = false;
    try { Rational(1, 0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

static void test_arithmetic() {
    CHECK(Rational(1, 2) + Rational(1, 3) == Rational(5, 6));
    CHECK(Rational(1, 2) - Rational(1, 3) == Rational(1, 6));
    CHECK(Rational(1, 2) * Rational(2, 3) == Rational(1, 3));
    CHECK(Rational(1, 2) / Rational(3, 4) == Rational(2, 3));
    CHECK(-Rational(1, 2) == Rational(-1, 2));
    CHECK(Rational(7).is_one() == false);
    CHECK(Rational(5, 5).is_one());
    CHECK(Rational(0, 7).is_zero());
}

static void test_axiom_fuzz() {
    std::mt19937 rng(12345);
    auto rr = [&]() {
        std::uniform_int_distribution<int64_t> dist(-1000000, 1000000);
        int64_t den = dist(rng);
        if (den == 0) den = 1;
        return Rational(dist(rng), den);
    };
    for (int i = 0; i < 5000; ++i) {
        Rational a = rr(), b = rr(), c = rr();
        CHECK(a + b == b + a);                      // commutativity
        CHECK(a * b == b * a);
        CHECK((a + b) + c == a + (b + c));          // associativity
        CHECK((a * b) * c == a * (b * c));
        CHECK(a * (b + c) == a * b + a * c);        // distributivity
        CHECK(a + Rational(0) == a);                // identities
        CHECK(a * Rational(1) == a);
        CHECK(a + (-a) == Rational(0));
        CHECK(a - a == Rational(0));
        if (!b.is_zero()) CHECK((a / b) * b == a);  // division inverse
        CHECK((a - b) + b == a);
    }
}

static void test_overflow_detection() {
    bool threw = false;
    try {
        Rational big = Rational(INT64_MAX, 1);
        Rational r = big + big;  // overflows int64
        (void)r;
    } catch (const std::overflow_error&) { threw = true; }
    CHECK(threw);
    threw = false;
    try {
        Rational(INT64_MAX, 1) * Rational(INT64_MAX, 1);
    } catch (const std::overflow_error&) { threw = true; }
    CHECK(threw);
    // 128-bit reduction must rescue representable results:
    CHECK(Rational(1LL << 40, 1) / Rational(1LL << 40, 1) == Rational(1));
    CHECK(Rational((int64_t)1 << 55) / Rational((int64_t)1 << 54) == Rational(2));
}

int main() {
    test_construction();
    test_arithmetic();
    test_axiom_fuzz();
    test_overflow_detection();
    std::printf("test_rational: OK (%d checks)\n", g_checks);
    return 0;
}
