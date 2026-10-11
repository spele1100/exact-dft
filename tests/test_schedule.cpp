// exact-dft — M3 tests: sector-packing traversal + scheduling (paper sec 4).
// Differential vs brute-force lexicographic packing and vs a naive per-axis
// tensor oracle: BIT-EXACT equality, no tolerances (spec L3).
#include "exact/oracle_dft.hpp"
#include "exact/schedule.hpp"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <numeric>
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

// ---------------------------------------------------------------------------
// Brute-force packing: sectors ordered lexicographically by block choices,
// entries within a sector by t-digits (first axis most significant), each
// entry mapped to its original mixed-radix address. This directly encodes
// the paper's definition (sec 4.2), independently of eq:sector-start /
// eq:sector-updates.
// ---------------------------------------------------------------------------
static std::vector<int> brute_pack(const std::vector<int>& radices,
                                   const std::vector<ed::SectorTables>& tab) {
    const int ell = (int)radices.size();
    long long R = 1;
    for (int r : radices) R *= r;
    std::vector<long long> tau(ell + 1, 1);
    for (int j = ell - 1; j >= 0; --j) tau[j] = tau[j + 1] * radices[j];

    std::vector<std::vector<ed::SectorTables::Block>> blocks(ell);
    for (int j = 0; j < ell; ++j) blocks[j] = tab[j].blocks();

    // Recursive enumeration of block choices (lexicographic in axis order)
    // x t-digits; o = sum_j t_j * prod_{i>j} q_i computed at the leaf from
    // the chosen suffix products (independent positional formula).
    struct Entry {
        std::vector<int> blockIdx;
        long long o;
        int orig;
    };
    std::vector<Entry> out;
    std::vector<int> choices(ell, 0), tdig(ell, 0);
    std::function<void(int)> go = [&](int j) {
        if (j == ell) {
            std::vector<int> qs(ell + 1, 1);
            for (int i = ell - 1; i >= 0; --i)
                qs[i] = qs[i + 1] * blocks[i][choices[i]].q;
            long long o = 0;
            for (int i = 0; i < ell; ++i)
                o += (long long)tdig[i] * qs[i + 1];
            int orig = 0;
            for (int i = 0; i < ell; ++i)
                orig = orig * radices[i] + blocks[i][choices[i]].digits[(size_t)tdig[i]];
            out.push_back({choices, o, orig});
            return;
        }
        for (size_t b = 0; b < blocks[j].size(); ++b) {
            choices[j] = (int)b;
            for (int t = 0; t < blocks[j][b].q; ++t) {
                tdig[j] = t;
                go(j + 1);
            }
        }
    };
    go(0);

    std::sort(out.begin(), out.end(), [](const Entry& a, const Entry& b) {
        if (a.blockIdx != b.blockIdx) return a.blockIdx < b.blockIdx;
        return a.o < b.o;
    });
    CHECK((long long)out.size() == R);
    std::vector<int> perm;
    perm.reserve(out.size());
    for (const Entry& e : out) perm.push_back(e.orig);
    return perm;
}

static void check_tables_consistent(const ed::SectorTables& tb) {
    for (int a = 0; a < tb.r; ++a) {
        CHECK(tb.p[a] >= 0 && tb.p[a] + tb.q[a] <= tb.r);
        CHECK(tb.q[a] == 1 || tb.q[a] == 2);
        CHECK(tb.t[a] >= 0 && tb.t[a] < tb.q[a]);
    }
    // Blocks partition the coordinates: p-offsets consistent.
    for (int a = 0; a < tb.r; ++a)
        for (int b = a + 1; b < tb.r; ++b) {
            const bool same = tb.p[a] == tb.p[b];
            if (same) { CHECK(tb.q[a] == tb.q[b]); CHECK(tb.t[a] != tb.t[b]); }
        }
}

// ---------------------------------------------------------------------------
static void test_sector_packing() {
    struct Case {
        std::vector<int> radices;
        std::vector<std::vector<std::pair<int, int>>> pairSets;  // per-axis
    };
    // Identity tables: packing must be the identity permutation.
    {
        std::vector<int> radices = {2, 3, 5};
        std::vector<ed::SectorTables> tab;
        for (int r : radices) tab.push_back(ed::SectorTables::identity(r));
        ed::SectorPacker pk(radices, tab);
        const auto& perm = pk.packed_to_orig();
        CHECK(perm.size() == 30u);
        for (int i = 0; i < 30; ++i) CHECK(perm[i] == i);
        const auto& sec = pk.sectors();
        CHECK(sec.size() == 30u);
        for (int i = 0; i < 30; ++i) { CHECK(sec[i].S == i); CHECK(sec[i].Q == 1); }
    }
    // Pair tables, including nonconsecutive pairs, multiple axes.
    std::mt19937 rng(7);
    const std::vector<std::vector<int>> radixSets = {
        {2, 3}, {3, 5}, {2, 3, 5}, {4, 5}, {2, 2, 7}, {3, 4, 5}};
    for (const auto& radices : radixSets) {
        long long R = 1;
        for (int r : radices) R *= r;
        for (int trial = 0; trial < 6; ++trial) {
            std::vector<ed::SectorTables> tab;
            for (int r : radices) {
                if (trial == 0 || r < 2) {
                    tab.push_back(ed::SectorTables::identity(r));
                } else {
                    std::vector<int> digits(r);
                    std::iota(digits.begin(), digits.end(), 0);
                    std::shuffle(digits.begin(), digits.end(), rng);
                    std::vector<std::pair<int, int>> pairs;
                    int used = 0;
                    while (used + 2 <= r && (int)pairs.size() < r / 3) {
                        pairs.emplace_back(digits[used], digits[used + 1]);
                        used += 2;
                    }
                    tab.push_back(ed::SectorTables::from_pairs(r, pairs));
                }
            }
            for (const auto& tb : tab) check_tables_consistent(tb);

            ed::SectorPacker pk(radices, tab);
            const auto& perm = pk.packed_to_orig();
            const auto& inv = pk.orig_to_packed();
            const auto& sec = pk.sectors();

            // Bijectivity + round-trip pack -> unpack identity.
            CHECK((long long)perm.size() == R);
            CHECK((long long)inv.size() == R);
            std::vector<char> seen(R, 0);
            for (long long s = 0; s < R; ++s) {
                CHECK(perm[s] >= 0 && perm[s] < R);
                CHECK(!seen[perm[s]]);
                seen[perm[s]] = 1;
                CHECK(inv[perm[s]] == (int)s);   // round-trip
            }
            // Sector widths sum to R; starts strictly increasing (monotone
            // lexicographic packed order); widths = 2^pairs.
            long long sum = 0;
            for (size_t i = 0; i < sec.size(); ++i) {
                sum += sec[i].Q;
                CHECK(sec[i].Q == (1LL << sec[i].pairs));
                if (i) CHECK(sec[i - 1].S < sec[i].S);
            }
            CHECK(sum == R);
            // Sector starts match brute force: packed block of sector i
            // occupies [S, S+Q) in packed order.
            const std::vector<int> want = brute_pack(radices, tab);
            CHECK(perm == want);
            // O(R) timing sanity: digit-DFS extensions < 2R (eq:prefix-nodes).
            CHECK(pk.state_updates() < 2 * R);
        }
    }
    std::printf("  sector packing: ok\n");
}

// ---------------------------------------------------------------------------
// Naive tensor oracle: apply F_{r_j} along axis j by explicit matrix-vector
// products with zeta_L^{(L/r_j) i k} entries (independent of the compiled
// Newton/Toeplitz word).
// ---------------------------------------------------------------------------
static std::vector<Cyclotomic> naive_tensor(const std::vector<int>& radices,
                                            std::vector<Cyclotomic> x) {
    const int ell = (int)radices.size();
    const int L = (int)std::accumulate(radices.begin(), radices.end(), 1,
                                       [](int a, int b) { return std::lcm(a, b); });
    std::vector<long long> tau(ell + 1, 1);
    for (int j = ell - 1; j >= 0; --j) tau[j] = tau[j + 1] * radices[j];
    // tau[j] = prod_{i>=j} r_i; the STRIDE of digit j (paper tau_j) is
    // tau[j+1].
    for (int j = 0; j < ell; ++j) {
        if (radices[j] == 1) continue;
        const Cyclotomic zj = Cyclotomic::zeta(L).pow(L / radices[j]);
        std::vector<Cyclotomic> y = x;
        // enumerate groups: fix all digits except j
        std::vector<int> digits(ell, 0);
        std::function<void(int, long long)> rec = [&](int i, long long base) {
            if (i == ell) {
                for (int k = 0; k < radices[j]; ++k) {
                    Cyclotomic acc(L);
                    for (int m = 0; m < radices[j]; ++m)
                        acc += x[(size_t)(base + (long long)m * tau[j + 1])] *
                               zj.pow(((int64_t)m * k) % radices[j]);
                    y[(size_t)(base + (long long)k * tau[j + 1])] = std::move(acc);
                }
                return;
            }
            if (i == j) { rec(i + 1, base); return; }
            for (int a = 0; a < radices[i]; ++a)
                rec(i + 1, base + (long long)a * tau[i + 1]);
        };
        rec(0, 0);
        x = std::move(y);
    }
    return x;
}

static std::vector<Cyclotomic> rand_vec(int L, long long n, std::mt19937& rng) {
    std::uniform_int_distribution<int64_t> num(-9, 9);
    std::uniform_int_distribution<int> den(1, 7);
    std::uniform_int_distribution<int> expo(0, L - 1);
    std::vector<Cyclotomic> v((size_t)n, Cyclotomic(L));
    const Cyclotomic z = Cyclotomic::zeta(L);
    for (long long i = 0; i < n; ++i)
        v[(size_t)i] = Cyclotomic::rational(L, Rational(num(rng), den(rng))) +
                       Cyclotomic::rational(L, Rational(num(rng), den(rng))) *
                           z.pow(expo(rng));
    return v;
}

static void test_tensor_fourier() {
    const std::vector<std::vector<int>> cases = {
        {2, 3}, {3, 5}, {2, 3, 5}, {3, 4, 5}, {2, 5, 7}, {4, 9}, {5, 8}};
    std::mt19937 rng(11);
    for (const auto& radices : cases) {
        long long R = 1;
        for (int r : radices) R *= r;
        ed::TensorFourier tf(radices);
        CHECK(tf.R() == R);
        const int L = tf.ring();
        for (int r : radices) CHECK(L % r == 0);

        // Random inputs, multiple trials.
        for (int trial = 0; trial < 3; ++trial) {
            std::vector<Cyclotomic> x = rand_vec(L, R, rng);
            std::vector<Cyclotomic> x2 = x;
            tf.apply(x2);
            std::vector<Cyclotomic> want = naive_tensor(radices, std::move(x));
            CHECK(x2.size() == want.size());
            for (long long i = 0; i < R; ++i) CHECK(x2[(size_t)i] == want[(size_t)i]);
        }

        // Basis vectors: tensor of basis vectors = outer product of columns.
        {
            std::vector<Cyclotomic> basis((size_t)R, Cyclotomic(L));
            basis[3 % (size_t)R] = Cyclotomic::integer(L, 1);
            std::vector<Cyclotomic> want = naive_tensor(radices, basis);
            tf.apply(basis);
            for (long long i = 0; i < R; ++i) CHECK(basis[(size_t)i] == want[(size_t)i]);
        }

        // Traversal-order independence: permuted axis application order.
        {
            std::vector<Cyclotomic> a = rand_vec(L, R, rng);
            std::vector<Cyclotomic> b = a;
            std::vector<int> order(radices.size());
            std::iota(order.begin(), order.end(), 0);
            std::shuffle(order.begin(), order.end(), rng);
            ed::TensorFourier tf2(radices);
            tf2.apply(a);
            ed::TensorFourier tf3(radices);
            tf3.apply(b, &order);
            for (long long i = 0; i < R; ++i) CHECK(a[(size_t)i] == b[(size_t)i]);
        }
        std::printf("  tensor radices {%s}: R=%lld ok\n",
                    [&] { std::string s; for (int r : radices) s += std::to_string(r) + ","; return s; }().c_str(), R);
    }
}

// Cross-check on a coprime set: tensor transform composed with zero-padding
// equals the plain oracle DFT at the product length (Good-Thomas style
// sanity: F_R restricted to coprime axes is exactly the tensor map under the
// CRT digit reindexing; here digits are mixed-radix, so instead verify the
// single-axis special case ell=1 reproduces oracle_dft).
static void test_single_axis_matches_oracle() {
    for (int r : {3, 5, 7, 8, 9, 12}) {
        ed::TensorFourier tf({r});
        CHECK(tf.ring() == r);
        std::mt19937 rng(r);
        std::vector<Cyclotomic> x = rand_vec(r, r, rng);
        std::vector<Cyclotomic> want = ed::oracle_dft(r, x);
        tf.apply(x);
        CHECK(x.size() == want.size());
        for (int i = 0; i < r; ++i) CHECK(x[(size_t)i] == want[(size_t)i]);
    }
    std::printf("  single-axis vs oracle: ok\n");
}

static void test_embed() {
    // zeta_r embeds to zeta_D^{D/r}; powers and field ops preserved.
    for (auto [d, D] : {std::pair<int, int>{3, 12}, {4, 20}, {5, 60}, {8, 24}}) {
        const Cyclotomic z = Cyclotomic::zeta(d);
        const Cyclotomic ze = ed::embed(z, D);
        CHECK(ze == Cyclotomic::zeta(D).pow(D / d));
        CHECK(ze.pow(d) == Cyclotomic::integer(D, 1));
        for (int k = 1; k < d; ++k)  // primitivity preserved
            CHECK(ze.pow(k) != Cyclotomic::integer(D, 1));
        // ring homomorphism on random elements: embed(a*b) == embed(a)*embed(b)
        std::mt19937 rng(d + D);
        for (int t = 0; t < 5; ++t) {
            std::vector<Cyclotomic> v = rand_vec(d, 2, rng);
            CHECK(ed::embed(v[0] * v[1], D) == ed::embed(v[0], D) * ed::embed(v[1], D));
            CHECK(ed::embed(v[0] + v[1], D) == ed::embed(v[0], D) + ed::embed(v[1], D));
        }
    }
    std::printf("  embed: ok\n");
}

static void test_schedule_report() {
    // Small axes compile to direct words (no shears); a width-16 axis uses
    // the chunked replay path.
    auto rp = ed::schedule_report({2, 3, 5});
    CHECK(rp.R == 30);
    CHECK(rp.L == 30);
    CHECK(rp.ell == 3);
    CHECK((long long)rp.axes.size() == 3);
    long long shears = 0, direct = 0;
    for (const auto& a : rp.axes) { shears += a.shears; direct += a.direct_blocks; }
    CHECK(shears == 0);  // r <= 5: direct word fallback
    CHECK(direct > 0);
    CHECK(rp.groups == 15 + 10 + 6);  // R/2 + R/3 + R/5 word invocations
    CHECK(rp.prefix_extensions < 2 * rp.R);

    auto rp2 = ed::schedule_report({9, 5});
    CHECK(rp2.R == 45);
    long long shears2 = 0;
    for (const auto& a : rp2.axes) shears2 += a.shears;
    CHECK(shears2 > 0);  // width-9 axis runs the replayed chunk compiler
    std::printf("  schedule_report: R=%lld/%lld shears=%lld/%lld\n", rp.R,
                rp2.R, shears, shears2);
}

int main() {
    test_embed();
    test_sector_packing();
    test_single_axis_matches_oracle();
    test_tensor_fourier();
    test_schedule_report();
    std::printf("test_schedule: all %d checks passed\n", g_checks);
    return 0;
}
