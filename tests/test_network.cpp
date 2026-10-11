// exact-dft — M5 tests: scaled tensor-network demo (paper sec 2).
//
// Differential standards (no tolerances anywhere):
//   * invocation (eq:net:schedule): forward sends (x, y, A, c) ->
//     (x, y + x, A, c) with ARBITRARY garbage preloaded into every aux
//     wire (the paper's distinctive claim, Lemma net:scalar); inverse
//     (rows reversed, signs flipped) sends y -> y - x, aux restored.
//   * three-stage exchange: (X_d, Y_d) -> (-Y_d, X_d), every invocation's
//     aux garbage restored (bank-level, h = 4 and 5).
//   * frames (sec 2.2, even m in {4, 6, 8}, exact over Q(i) = Q(zeta_4)):
//       - Phi_0 = Id;
//       - Phi_D = C^{otimes m} vs naive per-bit application;
//       - Lemma net:edge: Phi_V Phi_U^{-1} = product of |residual|
//         directional kernels K_z for random nested nondegenerate U < V;
//       - exceptional route (Prop net:finite-interface): for unit u,
//         Phi_{u^perp} Phi_u^{-1} = R_u Phi_D.
//   * dimension budget (sec 2.5): internal consistency Delta =
//     2 v^2 (v - 3h(h+1)); the paper's h = 100 integers W, s, Delta and
//     W* = 2^71 reproduced EXACTLY; lambda in (1, m) and theta =
//     1 - 2e-13 at h = 100; no saving for h <= 21, first saving at h = 22.
#include "exact/network.hpp"

#include <cstdio>
#include <random>
#include <vector>

using ed::Cyclotomic;
using ed::Rational;
using ed::net::frames::Mask;
using ed::net::i128_str;

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

static Cyclotomic rand_gauss(std::mt19937& rng, int d = 4) {
    std::uniform_int_distribution<int64_t> dist(-9, 9);
    return Cyclotomic::rational(d, Rational(dist(rng), 7)) +
           Cyclotomic::rational(d, Rational(dist(rng), 5)) * Cyclotomic::zeta(4);
}

// ---------------------------------------------------------------------------
// Scalar network (sec 2.1)
// ---------------------------------------------------------------------------
static void test_invocation(int h, int ring) {
    const ed::net::ScalarNetwork net(h);
    const int v = net.v();
    std::mt19937 rng(h * 1000 + ring);
    const auto rand_elem = [&rng, ring]() {
        // dense random element of the ring (garbage class, not just Q(i))
        std::uniform_int_distribution<int64_t> dist(-9, 9);
        const int64_t phi = ed::detail::euler_phi(ring);
        Cyclotomic c(ring);
        std::vector<Rational> co((size_t)phi, Rational(0));
        for (auto& q : co) {
            int64_t den = dist(rng);
            if (den == 0) den = 3;
            q = Rational(dist(rng), den);
        }
        return Cyclotomic(ring, std::move(co));
    };
    for (int trial = 0; trial < 3; ++trial) {
        std::vector<Cyclotomic> x, y, A((size_t)v * v), c(h + 1);
        for (int i = 0; i < v; ++i) x.push_back(rand_elem());
        for (int i = 0; i < v; ++i) y.push_back(rand_elem());
        for (auto& a : A) a = rand_elem();  // garbage EVERYWHERE (neighbors and not)
        for (auto& q : c) q = rand_elem();
        const auto x0 = x, y0 = y, A0 = A, c0 = c;

        auto yplus = y0;
        for (int i = 0; i < v; ++i) yplus[i] += x0[i];
        auto yminus = y0;
        for (int i = 0; i < v; ++i) yminus[i] -= x0[i];

        auto xx = x, yy = y, AA = A, cc = c;
        net.invoke(xx, yy, AA, cc, true);
        CHECK(xx == x0);
        CHECK(yy == yplus);
        CHECK(AA == A0);  // every aux wire restored despite the garbage
        CHECK(cc == c0);

        // inverse invocation from the ORIGINAL state: y -> y - x.
        auto xi = x, yi = y, Ai = A, ci = c;
        net.invoke(xi, yi, Ai, ci, false);
        CHECK(xi == x0);
        CHECK(yi == yminus);
        CHECK(Ai == A0);
        CHECK(ci == c0);

        // forward then inverse round-trips exactly.
        net.invoke(xx, yy, AA, cc, false);
        CHECK(yy == y0);
        CHECK(AA == A0);
        CHECK(cc == c0);
    }
}

static void test_exchange(int h) {
    const ed::net::ScalarNetwork net(h);
    const int v = net.v();
    std::mt19937 rng(h * 77);
    std::vector<Cyclotomic> X, Y;
    for (int i = 0; i < v * v * v; ++i) X.push_back(rand_gauss(rng));
    for (int i = 0; i < v * v * v; ++i) Y.push_back(rand_gauss(rng));
    const auto X0 = X, Y0 = Y;
    // Aux garbage: adversarial values that would poison a zero-assuming
    // schedule (nonzero everywhere, including non-neighbor slots).
    auto garbage = [&](std::vector<Cyclotomic>& A, std::vector<Cyclotomic>& c) {
        static std::mt19937 g(h * 31);
        for (auto& a : A) a = rand_gauss(g);
        for (auto& q : c) q = rand_gauss(g);
    };
    net.exchange(X, Y, garbage);
    // (X_d, Y_d) -> (-Y_d, X_d), bit-exact.
    for (int i = 0; i < v * v * v; ++i) {
        CHECK(X[i] == -Y0[i]);
        CHECK(Y[i] == X0[i]);
    }
    // Exchange is an involution up to sign: running twice returns -(X, Y).
    net.exchange(X, Y, garbage);
    for (int i = 0; i < v * v * v; ++i) {
        CHECK(X[i] == -X0[i]);
        CHECK(Y[i] == -Y0[i]);
    }
}

// ---------------------------------------------------------------------------
// Frames (sec 2.2), even m, exact over Q(i)
// ---------------------------------------------------------------------------
static std::vector<Cyclotomic> rand_array(std::mt19937& rng, size_t n) {
    std::vector<Cyclotomic> f;
    f.reserve(n);
    for (size_t i = 0; i < n; ++i) f.push_back(rand_gauss(rng));
    return f;
}

static void test_orthonormal(int m, std::mt19937& rng) {
    using namespace ed::net::frames;
    for (int trial = 0; trial < 8; ++trial) {
        // Random nondegenerate subspace containing a unit vector: take
        // random vectors, then force-include a coordinate unit.
        std::vector<Mask> basis;
        std::uniform_int_distribution<Mask> dist(0, (Mask)((1u << m) - 1));
        basis.push_back((Mask)1 << (trial % m));  // guaranteed unit
        for (int i = 0; i < m / 2; ++i) {
            Mask w = dist(rng);
            if (w) basis.push_back(w);
        }
        auto piv = echelon(basis);
        // Nondegeneracy check via the Gram matrix over the echelon basis.
        const size_t k = piv.size();
        bool nondeg = true;
        {
            // Solve Gram * coeff = e_i solvability for all i: rank(Gram) = k.
            std::vector<std::uint64_t> aug;  // k rows, k+1 cols
            for (size_t i = 0; i < k; ++i) {
                std::uint64_t row = 0;
                for (size_t j = 0; j < k; ++j)
                    if (dot(piv[i], piv[j])) row |= (std::uint64_t)1 << j;
                aug.push_back(row);
            }
            size_t rank = 0;
            for (size_t col = 0; col < k && rank < k; ++col) {
                ssize_t sel = -1;
                for (size_t r = rank; r < k; ++r)
                    if ((aug[r] >> col) & 1) { sel = (ssize_t)r; break; }
                if (sel < 0) continue;
                std::swap(aug[rank], aug[sel]);
                for (size_t r = rank + 1; r < k; ++r)
                    if ((aug[r] >> col) & 1) aug[r] ^= aug[rank];
                ++rank;
            }
            nondeg = (rank == k);
        }
        if (!nondeg) continue;  // random space happened to be degenerate; skip
        auto onb = orthonormal_basis(piv);
        CHECK(onb.size() == k);
        for (Mask b : piv) CHECK(in_span(onb, b));
        for (Mask a : onb) CHECK(unit_norm(a));
    }
}

static void test_terminal_frames(int m, std::mt19937& rng) {
    using namespace ed::net::frames;
    const size_t n = size_t(1) << m;
    // Phi_0 = Id (empty subspace: q = 0).
    {
        auto f = rand_array(rng, n);
        auto g = phi(f, {});
        CHECK(g == f);
    }
    // Phi_D = C^{otimes m} vs naive per-bit application.
    {
        auto f = rand_array(rng, n);
        std::vector<Mask> full;
        for (int b = 0; b < m; ++b) full.push_back((Mask)1 << b);
        auto g = phi(f, full);
        auto naive = naive_tensor_C(f);
        CHECK(g == naive);
    }
    // Projection sanity on a small nondegenerate ONB: {e0, e0^e1} -> units,
    // pairwise orthogonal, correct cardinality.
    {
        auto onb = orthonormal_basis({(Mask)1, (Mask)3});
        CHECK(onb.size() == 2);
        for (Mask a : onb) CHECK(unit_norm(a));
        CHECK(dot(onb[0], onb[1]) == 0);
        CHECK(in_span(onb, (Mask)2));  // e1 = e0 + (e0^e1)
    }
}

static void test_edge_lemma(int m, std::mt19937& rng) {
    using namespace ed::net::frames;
    const size_t n = size_t(1) << m;
    for (int trial = 0; trial < 4; ++trial) {
        // Random nested nondegenerate U < V: U = random ONB-containing space,
        // E = another one inside a fresh set of coordinates; V = U + E.
        std::uniform_int_distribution<Mask> dist(0, (Mask)((1u << m) - 1));
        std::vector<Mask> Ub{(Mask)1, (Mask)3};  // nondeg, units present
        for (int i = 0; i < 1; ++i) {
            Mask w = dist(rng) & ~((Mask)7);
            if (w && !in_span(Ub, w)) Ub.push_back(w);
        }
        auto gram_full_rank = [](const std::vector<Mask>& piv) {
            const size_t k = piv.size();
            std::vector<std::uint64_t> aug;
            for (size_t i = 0; i < k; ++i) {
                std::uint64_t row = 0;
                for (size_t j = 0; j < k; ++j)
                    if (dot(piv[i], piv[j])) row |= (std::uint64_t)1 << j;
                aug.push_back(row);
            }
            size_t rank = 0;
            for (size_t col = 0; col < k && rank < k; ++col) {
                ssize_t sel = -1;
                for (size_t r = rank; r < k; ++r)
                    if ((aug[r] >> col) & 1) { sel = (ssize_t)r; break; }
                if (sel < 0) continue;
                std::swap(aug[rank], aug[sel]);
                for (size_t r = rank + 1; r < k; ++r)
                    if ((aug[r] >> col) & 1) aug[r] ^= aug[rank];
                ++rank;
            }
            return rank == k;
        };
        if (!gram_full_rank(echelon(Ub))) continue;  // U must be nondegenerate
        auto U_onb = orthonormal_basis(Ub);
        // E inside U^perp containing a unit (Lemma net:edge needs an
        // orthonormal basis of the residual): random vectors projected off
        // U, plus coordinate units projected off U that remain units.
        std::vector<Mask> Eb;
        for (int i = 0; i < 2; ++i) {
            Mask w = dist(rng);
            w ^= project(U_onb, w);
            if (w) Eb.push_back(w);
        }
        {
            // A coordinate unit e stays a unit after projection iff its
            // projection is even-norm; search for one to guarantee E has a
            // unit vector.
            bool added = false;
            for (int i = 0; i < m && !added; ++i) {
                Mask e = (Mask)1 << i;
                Mask w = e ^ project(U_onb, e);
                if (unit_norm(w) && !in_span(Eb, w)) { Eb.push_back(w); added = true; }
            }
            if (!added) continue;
        }
        auto E_piv = echelon(Eb);
        if (E_piv.empty()) continue;
        if (!gram_full_rank(E_piv)) continue;  // E must be nondegenerate too
        std::vector<Mask> Vb = Ub;
        Vb.insert(Vb.end(), E_piv.begin(), E_piv.end());
        auto V_onb = orthonormal_basis(Vb);
        auto f = rand_array(rng, n);

        // Phi_V Phi_U^{-1} f  ==  product of |residual| kernels K_{z_j}.
        auto lhs = phi(phi(f, U_onb, /*inverse=*/true), V_onb);
        auto res = residual_basis(U_onb, Vb);
        CHECK(!res.empty());
        auto res_onb = orthonormal_basis(res);  // E nondegenerate with units
        auto rhs = f;
        for (Mask z : res_onb) {
            CHECK(unit_norm(z));
            rhs = kernel(rhs, z);
        }
        CHECK(lhs == rhs);
    }
}

static void test_exceptional_route(int m, std::mt19937& rng) {
    using namespace ed::net::frames;
    const size_t n = size_t(1) << m;
    const std::vector<Mask> full = [&] {
        std::vector<Mask> b;
        for (int i = 0; i < m; ++i) b.push_back((Mask)1 << i);
        return b;
    }();
    for (int trial = 0; trial < 4; ++trial) {
        Mask u = (Mask)1 << (trial % m) | (Mask)1 << ((trial * 3 + 1) % m) |
                 (Mask)1 << ((trial * 5 + 2) % m);
        u &= (Mask)((1u << m) - 1);
        if (!unit_norm(u)) u ^= 1;  // odd weight
        CHECK(unit_norm(u));
        auto u_perp = orthogonal_complement({u}, m);
        auto U_onb = std::vector<Mask>{u};
        auto u_perp_onb = orthonormal_basis(u_perp);
        CHECK((int)u_perp_onb.size() == m - 1);
        auto f = rand_array(rng, n);
        // Phi_{u^perp} Phi_u^{-1} = R_u Phi_D   (Prop net:finite-interface:
        // the exceptional X_d -> Y_d route has frame ratio R_u C^{otimes m}).
        auto lhs = phi(phi(f, U_onb, true), u_perp_onb);
        auto rhs = translate(phi(f, full), u);
        CHECK(lhs == rhs);
    }
}

// ---------------------------------------------------------------------------
// Dimension budget (sec 2.5) — reproduces the paper's exact integers.
// ---------------------------------------------------------------------------
static void test_accounting() {
    using ed::net::network_report;
    using ed::net::i128;
    // Internal consistency (already enforced inside network_report, checked
    // here again for the record at a few sizes).
    for (int h : {3, 4, 5, 10, 21, 22, 50, 100}) {
        auto r = network_report(h);
        CHECK(r.W * r.m - r.s == r.Delta);
        CHECK(r.Delta == 2 * r.v * r.v * (r.v - 3 * (i128)h * (h + 1)));
        CHECK(r.S == r.s + (r.Wstar - r.W) * r.m);
        CHECK(r.Wstar >= r.W && r.Wstar / 2 < r.W);
    }
    // The paper's h = 100 numbers, character for character (eq:net:wire-count,
    // eq:net:s, eq:net:delta, eq:net:padding).
    {
        auto r = network_report(100);
        CHECK(i128_str(r.W) == "1873807244643542670000");
        CHECK(i128_str(r.s) == "1873807244636671267308000000");
        CHECK(i128_str(r.Delta) == "6871402692000000");
        CHECK(r.Wstar == ((i128)1 << 71));
        CHECK(i128_str(r.m) == "1000000");
        CHECK(r.saving);
        // lambda = m - Delta/2^71 in (1, m); theta = 1 - 2e-13.
        CHECK(r.lambda > 1.0L && r.lambda < 1000000.0L);
        // theta = 1 - Delta/(W* ln m) ~ 1 - 2.11e-13 (paper: ~2e-13 below 1).
        CHECK(r.theta > 0.99999999999978L);
        CHECK(r.theta < 0.99999999999980L);
        CHECK(r.theta < 1.0L);
        std::printf("h=100: lambda = %.17Lg, theta = %.17Lg\n", r.lambda, r.theta);
    }
    // No saving below h = 22; first saving at h = 22 (C(22,3) = 1540 > 1518).
    for (int h = 3; h <= 21; ++h) CHECK(!network_report(h).saving);
    CHECK(network_report(22).saving);
    CHECK(ed::net::minimal_saving_h() == 22);
    // Default demo regime: h = 4 has NO saving — lambda > m, theta > 1.
    {
        auto r = network_report(4);
        CHECK(!r.saving);
        CHECK(r.lambda > (long double)r.m);
        CHECK(r.theta > 1.0L);
        std::printf("h=4: v=%s m=%s lambda=%.6Lg theta=%.6Lg (NO saving)\n",
                    i128_str(r.v).c_str(), i128_str(r.m).c_str(), r.lambda, r.theta);
        for (auto& line : r.lines())
            std::printf("  %s\n", line.c_str());
    }
}

int main() {
    test_invocation(4, 4);   // Q(i): the network's actual scalar field
    test_invocation(4, 5);   // dense ring garbage (Q(zeta_5)) — aux values
                             // are arbitrary ring elements, not just Q(i)
    test_invocation(6, 4);   // v = 20, deeper neighbor structure
    test_exchange(4);
    test_exchange(5);        // v = 10
    std::mt19937 rng(12345);
    for (int m : {4, 6, 8}) {
        test_orthonormal(m, rng);
        test_terminal_frames(m, rng);
        test_edge_lemma(m, rng);
        test_exceptional_route(m, rng);
    }
    test_accounting();
    std::printf("test_network: all %d checks passed\n", g_checks);
    return 0;
}
