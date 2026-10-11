// exact-dft — M1 tests: L1 naive exact DFT oracle over Q(zeta_n).
// Hand-computed small cases + structural identities (F^2 = n*rev, F^4 = n^2).
#include "exact/oracle_dft.hpp"

#include <cmath>
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

static std::vector<Cyclotomic> rand_vec(int n, std::mt19937& rng) {
    std::uniform_int_distribution<int64_t> dist(-20, 20);
    std::vector<Cyclotomic> v;
    v.reserve(n);
    const bool can_host_i = (n % 4 == 0);  // i = zeta_n^{n/4} needs 4 | n
    for (int i = 0; i < n; ++i)
        v.push_back(ed::embed_complex(
            n, Rational(dist(rng), 7),
            can_host_i ? Rational(dist(rng), 7) : Rational(0)));
    return v;
}

static void test_n1() {
    auto y = ed::oracle_dft(1, {Cyclotomic::rational(1, Rational(3, 4))});
    CHECK(y.size() == 1);
    CHECK(y[0] == Cyclotomic::rational(1, Rational(3, 4)));
}

static void test_n2_hand() {
    // F_2 = [[1,1],[1,-1]]; x = (a, b) -> (a+b, a-b)
    const Rational a(3, 5), b(-1, 7);
    auto y = ed::oracle_dft(2, {Cyclotomic::rational(2, a), Cyclotomic::rational(2, b)});
    CHECK(y[0] == Cyclotomic::rational(2, a + b));
    CHECK(y[1] == Cyclotomic::rational(2, a - b));
}

static void test_n4_hand() {
    // x = (1,0,0,0): F x = (1,1,1,1)
    auto e0 = ed::oracle_dft(4, {Cyclotomic::integer(4, 1), Cyclotomic::integer(4, 0),
                                 Cyclotomic::integer(4, 0), Cyclotomic::integer(4, 0)});
    for (int k = 0; k < 4; ++k) CHECK(e0[k] == Cyclotomic::integer(4, 1));
    // x = (0,1,0,0): F x = (1, i, -1, -i) = zeta_4^k
    auto e1 = ed::oracle_dft(4, {Cyclotomic::integer(4, 0), Cyclotomic::integer(4, 1),
                                 Cyclotomic::integer(4, 0), Cyclotomic::integer(4, 0)});
    const Cyclotomic z = Cyclotomic::zeta(4);
    for (int k = 0; k < 4; ++k) CHECK(e1[k] == z.pow(k));
    // x = (1,i,-1,-i) (i.e. z^k): F x = 4 * e_3  (sum_j z^{j(k+1)} = 4 iff k=3)
    std::vector<Cyclotomic> zk;
    for (int k = 0; k < 4; ++k) zk.push_back(z.pow(k));
    auto y = ed::oracle_dft(4, zk);
    for (int k = 0; k < 4; ++k)
        CHECK(y[k] == Cyclotomic::integer(4, k == 3 ? 4 : 0));
}

static void test_basis_columns() {
    // Spec L1 test: F_n applied to basis vectors recovers columns = powers of zeta_n.
    for (int n : {2, 3, 4, 5, 6, 8, 9, 12, 16, 30}) {
        const Cyclotomic z = Cyclotomic::zeta(n);
        for (int j = 0; j < n; ++j) {
            std::vector<Cyclotomic> e((size_t)n, Cyclotomic::integer(n, 0));
            e[j] = Cyclotomic::integer(n, 1);
            auto y = ed::oracle_dft(n, e);
            for (int k = 0; k < n; ++k) CHECK(y[k] == z.pow((int64_t)j * k % n));
        }
    }
}

static void test_fourth_power_identity() {
    // F^2 x = n * reverse(x); F^4 x = n^2 * x  (cyclic identities, spec L1).
    std::mt19937 rng(4242);
    for (int n : {2, 4, 6, 8, 12, 16, 20, 24}) {
        auto x = rand_vec(n, rng);
        auto y = ed::oracle_dft(n, x);
        auto y2 = ed::oracle_dft(n, y);
        // reverse: x_0, x_{n-1}, ..., x_1
        std::vector<Cyclotomic> rev;
        rev.push_back(x[0]);
        for (int i = n - 1; i >= 1; --i) rev.push_back(x[i]);
        for (int k = 0; k < n; ++k)
            CHECK(y2[k] == rev[k] * Cyclotomic::integer(n, n));
        auto y4 = ed::oracle_dft(n, ed::oracle_dft(n, y2));  // F^4 x = F^2(F^2 x)
        for (int k = 0; k < n; ++k)
            CHECK(y4[k] == x[k] * Cyclotomic::integer(n, (int64_t)n * n));
    }
}

static void test_rational_complex_consistency() {
    // At n=4 the root w = zeta_4 = i lives in Q(i), so DFT outputs stay in
    // span{1, t} and exactly-embedded values must match a float DFT closely.
    std::mt19937 rng(2024);
    const int n = 4;
    auto x = rand_vec(n, rng);
    auto y = ed::oracle_dft(n, x);
    for (int k = 0; k < n; ++k) {
        double sre = 0, sim = 0;
        const double arg = 2.0 * M_PI * k / n;
        for (int j = 0; j < n; ++j) {
            const double w_re = std::cos(arg * j), w_im = std::sin(arg * j);
            const double re = x[j].coeffs()[0].to_double();
            const double im = x[j].coeffs()[1].to_double();  // t^1 slot = i
            sre += re * w_re - im * w_im;
            sim += re * w_im + im * w_re;
        }
        const double yre = y[k].coeffs()[0].to_double();
        const double yim = y[k].coeffs()[1].to_double();
        CHECK(std::abs(yre - sre) < 1e-9);
        CHECK(std::abs(yim - sim) < 1e-9);
    }
}

static void test_convolution() {
    std::vector<Cyclotomic> a{Cyclotomic::integer(6, 1), Cyclotomic::integer(6, 2),
                              Cyclotomic::integer(6, 3)};
    std::vector<Cyclotomic> b{Cyclotomic::integer(6, 4), Cyclotomic::integer(6, 5)};
    auto c = ed::oracle_conv_linear(a, b);
    CHECK(c.size() == 4);
    CHECK(c[0] == Cyclotomic::integer(6, 4));
    CHECK(c[1] == Cyclotomic::integer(6, 13));
    CHECK(c[2] == Cyclotomic::integer(6, 22));
    CHECK(c[3] == Cyclotomic::integer(6, 15));
}

int main() {
    test_n1();
    test_n2_hand();
    test_n4_hand();
    test_basis_columns();
    test_fourth_power_identity();
    test_rational_complex_consistency();
    test_convolution();
    std::printf("test_oracle: OK (%d checks)\n", g_checks);
    return 0;
}
