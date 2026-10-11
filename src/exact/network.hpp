// exact-dft — L4: scaled tensor-network demo (paper sec 2, network.tex).
//
// The paper's sec 2 has two halves:
//
//   (a) sec 2.1 "A scalar network with arbitrary auxiliary values": two
//       banks (X_d), (Y_d) indexed by triples of 3-subsets d in T^3, an
//       8-row invocation schedule (eq:net:schedule) with per-invocation
//       side wires A_ST (ordered neighboring pairs) and central wires
//       c_1..c_h, c_*. Lemma net:scalar: a forward invocation sends
//       (x, y, A, c) -> (x, y + x, A, c) for ARBITRARY initial auxiliary
//       values; three stages (forward X->Y, inverse Y->X, forward X->Y)
//       exchange the banks as (X_d, Y_d) -> (-Y_d, X_d) and restore every
//       auxiliary value. This is implemented VERBATIM below with h
//       configurable (default 4: v = C(4,3) = 4, N0 = v^3 = 64) instead of
//       the paper's h = 100 (v = 161700, N0 ~ 4.2e15).
//
//   (b) secs 2.2-2.4: frame changes turn each scalar wire into an array
//       indexed by F_2^m (m = h^3): the normalized Walsh-conjugated phase
//       operators Phi_U = J diag(i^{q_U}) J (eq:net:frame) with
//       q_U(x) = wt(P_U x) mod 4; Lemma net:edge: Phi_V Phi_U^{-1} factors
//       into |residual| directional kernels C_z = a Id + b R_z (a = (1+i)/2,
//       b = (1-i)/2); terminal labels Phi_D = C^{otimes m}. Executing the
//       full telescoping network needs 2^m-entry arrays — galactic for any
//       h >= 3 — so we implement the *algebra* of the frame changes exactly
//       (subspaces of F_2^m, orthonormal bases per the paper's constructive
//       proof of Lemma net:residuals, q_U, Phi_U, kernels) and verify the
//       two exactness-critical identities (Lemma net:edge; the terminal /
//       exceptional-route frame ratios of Prop net:finite-interface) at
//       test scale m in {4, 6, 8} where arrays are feasible. Everything is
//       exact over Q(i) = Q(zeta_4): for even m the normalized Walsh matrix
//       J = W / 2^{m/2} has rational entries, so no irrational appears.
//
//   (c) sec 2.5: the dimension budget. Delta = W m - s = 2 v^2 (v - 3h(h+1))
//       (eq:net:delta), the padding W_* = 2^71 and the critical multiplier
//       lambda = m - Delta / W_* (eq:net:padding), theta = log_m lambda.
//       network_report(h) computes all of it in __int128 and — this is the
//       honest-positioning requirement — reports when there is NO saving:
//       Delta > 0 requires v > 3h(h+1), which first happens at h = 22
//       (C(22,3) = 1540 > 3*22*23 = 1518). At the default h = 4 the network
//       uses MORE directional steps than the trivial per-direction baseline
//       (lambda > m, theta > 1). Even where Delta > 0, executing the
//       network needs arrays of 2^{h^3} entries, and at the paper's h = 100
//       the saving is theta = 1 - 2e-13 with "no useful crossover estimate"
//       (the paper's own words). This module demonstrates the mechanism;
//       it does not and cannot demonstrate a speedup.
#pragma once
#include "cyclotomic.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace ed {
namespace net {

using i128 = __int128;

inline std::string i128_str(i128 v) {
    if (v == 0) return "0";
    bool neg = v < 0;
    unsigned __int128 u = neg ? (unsigned __int128)(-(v + 1)) + 1u : (unsigned __int128)v;
    std::string s;
    while (u > 0) { s += char('0' + (int)(u % 10)); u /= 10; }
    if (neg) s += '-';
    return std::string(s.rbegin(), s.rend());
}

inline i128 choose3(i128 n) {
    return n * (n - 1) * (n - 2) / 6;  // exact for n >= 0 (product of 3 consecutive)
}

// ---------------------------------------------------------------------------
// Dimension accounting (eq:net:parameters / net:wire-count / net:s /
// net:delta / net:padding), __int128, h in [3, 128].
// ---------------------------------------------------------------------------
struct NetworkReport {
    int h = 0;
    i128 v = 0;        // C(h,3) triples
    i128 m = 0;        // h^3
    i128 N0 = 0;       // v^3 bank entries per bank
    i128 I0 = 0;       // 3 v^2 invocations
    i128 d_nbr = 0;    // C(h-3,3) + 3(h-3) neighbors per triple
    i128 W = 0;        // physical wires (eq:net:wire-count)
    i128 s = 0;        // total residual dimensions (eq:net:s)
    i128 Delta = 0;    // W m - s (eq:net:delta)
    i128 Wstar = 0;    // least power of two >= W
    i128 S = 0;        // s + (Wstar - W) m = Wstar m - Delta
    bool saving = false;      // Delta > 0
    long double lambda = 0;   // S / Wstar
    long double theta = 0;    // log_m lambda

    std::vector<std::string> lines() const;  // human-readable honest report
};

inline NetworkReport network_report(int h) {
    if (h < 3 || h > 128)
        throw std::invalid_argument("network_report: h must be in [3, 128]");
    NetworkReport r;
    r.h = h;
    r.v = choose3(h);
    r.m = (i128)h * h * h;
    r.N0 = r.v * r.v * r.v;
    r.I0 = 3 * r.v * r.v;
    r.d_nbr = choose3(h - 3) + 3 * (h - 3);
    r.W = 2 * r.N0 + r.I0 * (r.v * r.d_nbr + h + 1);           // eq:net:wire-count
    r.s = r.W * r.m - 2 * r.N0 + 2 * r.I0 * (h + 1) * h;       // eq:net:s
    r.Delta = r.W * r.m - r.s;                                  // eq:net:delta
    if (r.Delta != 2 * r.v * r.v * (r.v - 3 * (i128)h * (h + 1)))
        throw std::logic_error("network_report: Delta bookkeeping mismatch");
    i128 ws = 1;
    while (ws < r.W) ws *= 2;
    r.Wstar = ws;
    r.S = r.s + (r.Wstar - r.W) * r.m;                          // eq:net:padding
    r.saving = r.Delta > 0;
    r.lambda = ((long double)r.S) / ((long double)r.Wstar);
    r.theta = std::log((double)r.lambda) / std::log((double)r.m);
    return r;
}

inline std::vector<std::string> NetworkReport::lines() const {
    std::vector<std::string> L;
    auto add = [&](const std::string& k, const std::string& v) {
        L.push_back(k + ": " + v);
    };
    add("h", std::to_string(h));
    add("v = C(h,3)", i128_str(v));
    add("m = h^3", i128_str(m));
    add("N0 = v^3 (bank entries)", i128_str(N0));
    add("I0 = 3 v^2 (invocations)", i128_str(I0));
    add("d_nbr (neighbors per triple)", i128_str(d_nbr));
    add("W (physical wires)", i128_str(W));
    add("s (directional steps)", i128_str(s));
    add("Delta = W m - s", i128_str(Delta));
    add("W* (padding power of two)", i128_str(Wstar));
    add("S = s + (W*-W) m", i128_str(S));
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.15Lg", lambda);
    add("lambda = S/W*", buf);
    std::snprintf(buf, sizeof buf, "%.15Lg", theta);
    add("theta = log_m lambda", buf);
    if (!saving) {
        L.push_back("VERDICT: NO POWER SAVING at h = " + std::to_string(h) +
                    ": v = " + i128_str(v) + " <= 3h(h+1), so Delta <= 0 and the "
                    "network needs MORE directional steps than the trivial "
                    "per-direction baseline (lambda > m, theta > 1).");
    } else {
        L.push_back("VERDICT: Delta > 0 (first at h = 22), but executing the "
                    "network requires arrays of 2^m = 2^" + i128_str(m) +
                    " entries; the saving is an asymptotic curiosity with no "
                    "useful crossover estimate (paper sec 1), not a usable "
                    "algorithm at any feasible size.");
    }
    return L;
}

// Smallest h with Delta > 0 (22 for the paper's formulas).
inline int minimal_saving_h() {
    for (int h = 3; h <= 128; ++h)
        if (network_report(h).saving) return h;
    return -1;
}

// ---------------------------------------------------------------------------
// Scalar network (sec 2.1), verbatim 8-row schedule, h configurable.
// Values live in any Cyclotomic ring d (tests use d = 4 = Q(i) and d = 5).
// ---------------------------------------------------------------------------
class ScalarNetwork {
  public:
    explicit ScalarNetwork(int h) : h_(h) {
        if (h < 3) throw std::invalid_argument("ScalarNetwork: h >= 3");
        // T = 3-subsets of {0..h-1}, lexicographic.
        for (int a = 0; a < h; ++a)
            for (int b = a + 1; b < h; ++b)
                for (int c = b + 1; c < h; ++c) T_.push_back({a, b, c});
        v_ = (int)T_.size();
        if (v_ != (int)choose3(h))
            throw std::logic_error("ScalarNetwork: triple enumeration mismatch");
        // Neighbor lists: distinct triples with even intersection (size 0 or 2).
        nbr_.assign(v_, {});
        for (int s = 0; s < v_; ++s)
            for (int t = 0; t < v_; ++t) {
                if (s == t) continue;
                int inter = 0;
                for (int e : T_[s])
                    for (int f : T_[t]) inter += (e == f) ? 1 : 0;
                if (inter % 2 == 0) nbr_[s].push_back(t);
            }
    }

    int h() const { return h_; }
    int v() const { return v_; }
    const std::vector<std::array<int, 3>>& triples() const { return T_; }
    const std::vector<int>& neighbors(int s) const { return nbr_[s]; }
    int64_t bank_index(int p, int q, int r) const {
        return (int64_t)p * v_ * v_ + (int64_t)q * v_ + r;
    }

    // One invocation (eq:net:schedule), in place. x = logical source list
    // (length v), y = target list, A = side wires as a v*v matrix indexed
    // A[s*v + t] over ORDERED neighboring pairs (other entries never
    // touched), c = h+1 central wires (c[0..h-1] = c_1..c_h, c[h] = c_*).
    // forward = true: rows 0..7, net y += x, A and c restored.
    // forward = false: rows 7..0 with flipped signs (the paper's "inverse
    // forward invocation"), net y -= x, A and c restored.
    // Initial auxiliary values are ARBITRARY (Lemma net:scalar).
    void invoke(std::vector<Cyclotomic>& x, std::vector<Cyclotomic>& y,
                std::vector<Cyclotomic>& A, std::vector<Cyclotomic>& c,
                bool forward) const {
        const int d = x.at(0).d();
        const auto check = [&](const std::vector<Cyclotomic>& w, const char* what) {
            if ((int)w.size() != v_ || w.at(0).d() != d) {
                throw std::invalid_argument(std::string("invoke: ") + what +
                                            " length/ring mismatch");
            }
        };
        check(x, "x");
        check(y, "y");
        if ((int)A.size() != v_ * v_ || A.at(0).d() != d)
            throw std::invalid_argument("invoke: A length/ring mismatch");
        if ((int)c.size() != h_ + 1 || c.at(0).d() != d)
            throw std::invalid_argument("invoke: c length/ring mismatch");

        const Rational half(1, 2);
        // (JA)_S = -sum_{T nbr S} ((|S cap T| - 1)/2) A_{ST}, from current A.
        auto JA = [&](int s) {
            Cyclotomic acc(d);
            for (int t : nbr_[s]) {
                int inter = 0;
                for (int e : T_[s])
                    for (int f : T_[t]) inter += (e == f) ? 1 : 0;
                acc += A[(size_t)s * v_ + t] * Cyclotomic::rational(
                    d, Rational(-(inter - 1)) * half);
            }
            return acc;
        };
        // (Rc)_S = (sum_{j in S} c_j - c_*)/2, from current c.
        auto Rc = [&](int s) {
            Cyclotomic acc = -c[h_];  // -c_*
            for (int j : T_[s]) acc += c[j];
            return acc * Cyclotomic::rational(d, half);
        };
        // Gx_j = sum_{T contains j} x_T (j in 0..h-1); Gx_* = sum_T x_T.
        auto Gx = [&]() {
            std::vector<Cyclotomic> g(h_ + 1, Cyclotomic(d));
            for (int t = 0; t < v_; ++t) {
                if (x[t].is_zero()) continue;
                for (int j : T_[t]) g[j] += x[t];
                g[h_] += x[t];
            }
            return g;
        };

        // Row r with sign sigma (+1 forward, -1 inverse): adds sigma times
        // the row's update to the touched wires, reading CURRENT values.
        auto row = [&](int r, int sigma) {
            const Cyclotomic sgn = Cyclotomic::integer(d, sigma);
            switch (r) {
                case 0:  // y <- y - J A
                    for (int s = 0; s < v_; ++s) y[s] -= sgn * JA(s);
                    break;
                case 1:  // y <- y - R c
                    for (int s = 0; s < v_; ++s) y[s] -= sgn * Rc(s);
                    break;
                case 2:  // A <- A + V x   (side wires only)
                    for (int s = 0; s < v_; ++s)
                        for (int t : nbr_[s])
                            A[(size_t)s * v_ + t] += sgn * x[t];
                    break;
                case 3: {  // c <- c + G x
                    auto g = Gx();
                    for (int j = 0; j <= h_; ++j) c[j] += sgn * g[j];
                    break;
                }
                case 4:  // y <- y + R c
                    for (int s = 0; s < v_; ++s) y[s] += sgn * Rc(s);
                    break;
                case 5:  // y <- y + J A
                    for (int s = 0; s < v_; ++s) y[s] += sgn * JA(s);
                    break;
                case 6: {  // c <- c - G x
                    auto g = Gx();
                    for (int j = 0; j <= h_; ++j) c[j] -= sgn * g[j];
                    break;
                }
                case 7:  // A <- A - V x
                    for (int s = 0; s < v_; ++s)
                        for (int t : nbr_[s])
                            A[(size_t)s * v_ + t] -= sgn * x[t];
                    break;
                default: throw std::logic_error("invoke: bad row");
            }
        };

        if (forward)
            for (int r = 0; r < 8; ++r) row(r, +1);
        else
            for (int r = 7; r >= 0; --r) row(r, -1);
    }

    // Three-stage bank exchange (Lemma net:scalar):
    //   stage 1 (j = 0): forward X -> Y        (Y += X)
    //   stage 2 (j = 1): inverse Y -> X        (X -= Y)
    //   stage 3 (j = 2): forward X -> Y        (Y += X)
    // net (X_d, Y_d) -> (-Y_d, X_d). Banks are X, Y of length v^3 (index
    // bank_index). Every invocation allocates its OWN aux wires, initialized
    // by aux_init (which may install arbitrary garbage — that is the point);
    // all of them come out restored.
    using AuxInit = std::function<void(std::vector<Cyclotomic>& A,
                                       std::vector<Cyclotomic>& c)>;
    void exchange(std::vector<Cyclotomic>& X, std::vector<Cyclotomic>& Y,
                  const AuxInit& aux_init) const {
        const int64_t n0 = (int64_t)v_ * v_ * v_;
        if ((int64_t)X.size() != n0 || (int64_t)Y.size() != n0)
            throw std::invalid_argument("exchange: bank size != v^3");
        if (X.at(0).d() != Y.at(0).d())
            throw std::invalid_argument("exchange: bank rings must match");
        const int d = X[0].d();
        const int stage_src_bank[3] = {0, 1, 0};   // stage j: source bank (0=X, 1=Y)
        const bool stage_fwd[3] = {true, false, true};
        for (int j = 0; j < 3; ++j) {
            auto& src = stage_src_bank[j] == 0 ? X : Y;
            auto& tgt = stage_src_bank[j] == 0 ? Y : X;
            // Fix the two positions other than j; vary d_j over T.
            for (int p = 0; p < v_; ++p) {
                for (int q = 0; q < v_; ++q) {
                    std::vector<Cyclotomic> x(v_), y(v_);
                    for (int r = 0; r < v_; ++r) {
                        // Bank entry with triple index r at position j and
                        // p, q at the other two positions in order.
                        const int64_t src_idx =
                            j == 0 ? bank_index(r, p, q)
                                   : (j == 1 ? bank_index(p, r, q)
                                             : bank_index(p, q, r));
                        x[r] = src[src_idx];
                        y[r] = tgt[src_idx];  // same (p,q,r) slot in the other bank
                    }
                    std::vector<Cyclotomic> A((size_t)v_ * v_, Cyclotomic(d));
                    std::vector<Cyclotomic> c(h_ + 1, Cyclotomic(d));
                    if (aux_init) aux_init(A, c);
                    invoke(x, y, A, c, stage_fwd[j]);
                    for (int r = 0; r < v_; ++r) {
                        const int64_t idx =
                            j == 0 ? bank_index(r, p, q)
                                   : (j == 1 ? bank_index(p, r, q)
                                             : bank_index(p, q, r));
                        src[idx] = x[r];
                        tgt[idx] = y[r];
                    }
                }
            }
        }
    }

  private:
    int h_ = 0, v_ = 0;
    std::vector<std::array<int, 3>> T_;
    std::vector<std::vector<int>> nbr_;
};

// ---------------------------------------------------------------------------
// Frame changes (secs 2.2-2.3) at test scale: subspaces of F_2^m, m <= 16.
// ---------------------------------------------------------------------------
namespace frames {

using Mask = uint32_t;  // a vector of F_2^m as a bit mask (bit i = coordinate i)

inline int dot(Mask a, Mask b) { return __builtin_popcount(a & b) & 1; }
inline bool unit_norm(Mask a) { return __builtin_popcount(a) & 1; }

// Leading (highest) set bit of a nonzero vector.
inline Mask lead_bit(Mask p) { return (Mask)1u << (31 - __builtin_clz(p)); }

// Reduce a basis to echelon form; returns independent generators, sorted by
// leading bit (distinct leads, so value-descending order = lead-descending).
inline std::vector<Mask> echelon(std::vector<Mask> basis) {
    std::vector<Mask> piv;
    for (Mask v : basis) {
        while (v != 0) {
            // Cancel v's leading bit against the pivot with the same lead.
            bool canceled = false;
            for (size_t i = 0; i < piv.size(); ++i)
                if (lead_bit(piv[i]) == lead_bit(v)) { v ^= piv[i]; canceled = true; break; }
            if (!canceled) break;  // v reduced w.r.t. the pivot set
        }
        if (v) piv.push_back(v);
    }
    std::sort(piv.begin(), piv.end(), std::greater<Mask>());
    return piv;
}

inline bool in_span(const std::vector<Mask>& basis, Mask w) {
    for (Mask p : echelon(basis)) {  // sorted lead-descending, distinct leads
        if (w & lead_bit(p)) w ^= p;
    }
    return w == 0;
}

// Orthonormal basis of a nondegenerate subspace given to contain a unit
// vector — the constructive proof of Lemma net:residuals: split unit lines,
// split alternating planes, absorb each plane into a retained unit line via
// the triple (w+p, w+q, w+p+q).
inline std::vector<Mask> orthonormal_basis(std::vector<Mask> basis) {
    auto piv = echelon(std::move(basis));
    std::vector<Mask> units;      // orthonormal unit lines
    std::vector<std::array<Mask, 2>> planes;  // alternating planes (p, q), p.p=q.q=0, p.q=1

    // Split unit vectors greedily; what remains is alternating (or zero).
    std::vector<Mask> cur = piv;
    while (true) {
        // Find a unit vector in span(cur): bounded search over combinations
        // (Gray-code accumulation; dims here are <= 16, spans <= 65536).
        const size_t k = cur.size();
        if (k == 0) break;
        const size_t n = size_t(1) << k;
        Mask found = 0;
        {
            std::vector<Mask> acc(n, 0);
            for (size_t i = 1; i < n; ++i) {
                const size_t j = __builtin_ctzll(i);  // lowest set bit
                acc[i] = acc[i & (i - 1)] ^ cur[j];
                if (unit_norm(acc[i])) { found = acc[i]; break; }
            }
        }
        if (found == 0) {
            // Remaining space is alternating & nondegenerate: split planes.
            std::vector<Mask> rem = cur;
            while (!rem.empty()) {
                const Mask p = rem.back();
                rem.pop_back();
                Mask q = 0;
                bool ok = false;
                for (Mask cand : rem)
                    if (dot(p, cand) == 1) { q = cand; ok = true; break; }
                // Search combinations if no single basis vector pairs with p.
                if (!ok) {
                    const size_t kk = rem.size();
                    std::vector<Mask> acc(size_t(1) << kk, 0);
                    for (size_t i = 1; i < (size_t(1) << kk) && !ok; ++i) {
                        const size_t j = __builtin_ctzll(i);
                        acc[i] = acc[i & (i - 1)] ^ rem[j];
                        if (dot(p, acc[i]) == 1) { q = acc[i]; ok = true; }
                    }
                }
                if (!ok) throw std::logic_error("orthonormal_basis: degenerate space");
                planes.push_back({p, q});
                // Remove span{p, q}: project the rest off the plane.
                std::vector<Mask> next;
                for (Mask b : rem) {
                    const Mask b2 = b ^ (Mask)dot(b, p) * q ^ (Mask)dot(b, q) * p;
                    if (b2 != 0) next.push_back(b2);
                }
                rem = echelon(std::move(next));
            }
            break;
        }
        units.push_back(found);
        // Complement of <found> inside span(cur): project off.
        std::vector<Mask> next;
        for (Mask b : cur) {
            const Mask b2 = b ^ (Mask)dot(b, found) * found;
            if (b2 != 0) next.push_back(b2);
        }
        cur = echelon(std::move(next));
    }

    // Absorb each alternating plane into an already-emitted unit line w
    // (paper: net:residuals proof): replace w in place by the pairwise-
    // orthogonal unit triple (w+p, w+q, w+p+q), which spans <w, p, q>.
    // Every unit emitted so far is orthogonal to the remaining planes, so
    // any of them may serve as w; net growth is +2 vectors per plane.
    std::vector<Mask> out = units;
    for (auto& pq : planes) {
        if (out.empty())
            throw std::logic_error("orthonormal_basis: no unit to absorb a plane");
        const Mask w = out.back();
        out.pop_back();
        out.insert(out.end(), {w ^ pq[0], w ^ pq[1], w ^ pq[0] ^ pq[1]});
    }
    // Sanity: pairwise orthogonal unit vectors, correct cardinality.
    for (Mask a : out) {
        if (!unit_norm(a)) throw std::logic_error("orthonormal_basis: non-unit output");
        for (Mask b : out)
            if (a != b && dot(a, b) != 0)
                throw std::logic_error("orthonormal_basis: non-orthogonal output");
    }
    return out;
}

// Orthogonal complement basis (in F_2^m).
inline std::vector<Mask> orthogonal_complement(const std::vector<Mask>& basis, int m) {
    // Solve dot(v, b) = 0 for all b by basis-free scan over coordinates.
    // Standard: build parity-check via echelon over the m coordinates.
    std::vector<std::vector<int>> rows;  // each row: constraint over m coords
    for (Mask b : basis) {
        std::vector<int> r;
        for (int i = 0; i < m; ++i)
            if ((b >> i) & 1) r.push_back(i);
        rows.push_back(std::move(r));
    }
    // Gaussian elimination on the m unknowns.
    std::vector<int> pivcol;
    std::vector<std::uint64_t> mat;  // bitmask over m columns
    for (auto& r : rows) {
        std::uint64_t w = 0;
        for (int c : r) w |= (std::uint64_t)1 << c;
        mat.push_back(w);
    }
    for (int c = 0; c < m && (int)mat.size() < m; ++c) {
        int sel = -1;
        for (size_t i = pivcol.size(); i < mat.size(); ++i)
            if ((mat[i] >> c) & 1) { sel = (int)i; break; }
        if (sel < 0) continue;
        std::swap(mat[pivcol.size()], mat[sel]);
        for (size_t i = pivcol.size() + 1; i < mat.size(); ++i)
            if ((mat[i] >> c) & 1) mat[i] ^= mat[pivcol.size()];
        pivcol.push_back(c);
    }
    // Free columns -> nullspace vectors (each = e_c + combination of pivots).
    std::vector<bool> is_piv(m, false);
    for (int c : pivcol) is_piv[c] = true;
    std::vector<Mask> comp;
    for (int c = 0; c < m; ++c) {
        if (is_piv[c]) continue;
        Mask v = (Mask)1 << c;
        for (size_t i = 0; i < pivcol.size(); ++i)
            if ((mat[i] >> c) & 1) v |= (Mask)1 << pivcol[i];
        comp.push_back(v);
    }
    return comp;
}

// Projection onto U along U^perp (needs U nondegenerate): P x = sum z (z.x).
inline Mask project(const std::vector<Mask>& onb, Mask x) {
    Mask p = 0;
    for (Mask z : onb)
        if (dot(z, x)) p ^= z;
    return p;
}

// ---------------------------------------------------------------------------
// Exact array operators over Q(i) = Q(zeta_4), m EVEN (J = W/2^{m/2} then
// has rational entries; sec 2.2's identities hold with no irrational).
// ---------------------------------------------------------------------------
inline Cyclotomic ipow(int q) {  // i^q, q mod 4
    q &= 3;
    return q == 0 ? Cyclotomic::integer(4, 1)
         : q == 1 ? Cyclotomic::zeta(4)
         : q == 2 ? Cyclotomic::integer(4, -1)
                  : -Cyclotomic::zeta(4);
}

// Unnormalized Walsh transform in place: g[y] = sum_x (-1)^{x.y} f[x]
// (fast Walsh-Hadamard over the m address bits).
inline void walsh(std::vector<Cyclotomic>& f) {
    const int m = (int)__builtin_ctzll((unsigned long long)f.size());
    for (int b = 0; b < m; ++b) {
        const size_t step = (size_t)1 << b;
        for (size_t x = 0; x < f.size(); x += step * 2)
            for (size_t i = x; i < x + step; ++i) {
                const Cyclotomic u = f[i], w = f[i + step];
                f[i] = u + w;
                f[i + step] = u - w;
            }
    }
}

// q_U(x) = wt(P_U x) mod 4 (eq:net:frame).
inline int q_of(const std::vector<Mask>& onb, Mask x) {
    return __builtin_popcount(project(onb, x)) & 3;
}

// Phi_U = J diag(i^{q_U}) J with J the NORMALIZED Walsh matrix: implemented
// exactly as 2^{-m} W D W (J^2 = Id). inverse=true uses i^{-q_U}.
inline std::vector<Cyclotomic> phi(const std::vector<Cyclotomic>& f,
                                   const std::vector<Mask>& onb, bool inverse = false) {
    const size_t n = f.size();
    const int m = (int)__builtin_ctzll((unsigned long long)n);
    if (m <= 0 || (size_t(1) << m) != n || m % 2 != 0 || m > 30)
        throw std::invalid_argument("phi: need even m, array size 2^m");
    std::vector<Cyclotomic> g = f;
    walsh(g);
    for (size_t x = 0; x < n; ++x) {
        int q = q_of(onb, (Mask)x);
        if (inverse) q = (4 - q) & 3;
        if (q) g[x] = g[x] * ipow(q);
    }
    walsh(g);
    const Rational scale = Rational(1, 1 << m);
    for (auto& v : g) v = v * Cyclotomic::rational(4, scale);
    return g;
}

// Naive C^{otimes m}: one C per address bit (a = (1+i)/2, b = (1-i)/2).
inline std::vector<Cyclotomic> naive_tensor_C(const std::vector<Cyclotomic>& f) {
    const size_t n = f.size();
    const int m = (int)__builtin_ctzll((unsigned long long)n);
    const Cyclotomic a = Cyclotomic::rational(4, Rational(1, 2)) * (Cyclotomic::integer(4, 1) + Cyclotomic::zeta(4));
    const Cyclotomic b = Cyclotomic::rational(4, Rational(1, 2)) * (Cyclotomic::integer(4, 1) - Cyclotomic::zeta(4));
    std::vector<Cyclotomic> g = f;
    for (int bit = 0; bit < m; ++bit) {
        const size_t step = (size_t)1 << bit;
        for (size_t x = 0; x < n; x += step * 2)
            for (size_t i = x; i < x + step; ++i) {
                const Cyclotomic u = g[i], w = g[i + step];
                g[i] = a * u + b * w;
                g[i + step] = b * u + a * w;
            }
    }
    return g;
}

// Directional kernel K_z = a' Id + b' R_z, R_z f (x) = f(x + z):
// a' = (1 + i^{wt z})/2, b' = (1 - i^{wt z})/2 for ODD wt(z) (Lemma net:edge:
// wt(z) odd, so i^{wt z} = i or -i and K_z is C_z or its sign-flipped form;
// C_z^{-1} = R_z C_z only enters the paper's inverse-product bookkeeping —
// the operator itself is this single kernel).
inline std::vector<Cyclotomic> kernel(const std::vector<Cyclotomic>& f, Mask z) {
    if (!unit_norm(z))
        throw std::invalid_argument("kernel: z must have odd weight");
    const int w = __builtin_popcount(z) & 3;
    const Cyclotomic ip = ipow(w);
    const Rational half(1, 2);
    const Cyclotomic ap = (Cyclotomic::integer(4, 1) + ip) * Cyclotomic::rational(4, half);
    const Cyclotomic bp = (Cyclotomic::integer(4, 1) - ip) * Cyclotomic::rational(4, half);
    std::vector<Cyclotomic> g(f.size());
    for (size_t x = 0; x < f.size(); ++x)
        g[x] = ap * f[x] + bp * f[x ^ z];
    return g;
}

// Translation R_z: (R_z f)(x) = f(x + z).
inline std::vector<Cyclotomic> translate(const std::vector<Cyclotomic>& f, Mask z) {
    std::vector<Cyclotomic> g(f.size());
    for (size_t x = 0; x < f.size(); ++x) g[x] = f[x ^ z];
    return g;
}

// Orthonormal basis of the residual of nested U < V: the orthogonal
// complement of U inside V (Lemma net:edge's "residual").
inline std::vector<Mask> residual_basis(const std::vector<Mask>& U_onb,
                                        const std::vector<Mask>& V_basis) {
    // residual = V cap U^perp: project each V generator off U (v + P_U v =
    // v - P_U v over F_2), landing in U^perp. Since U < V nondegenerate,
    // V = U perp residual.
    std::vector<Mask> res;
    for (Mask v : V_basis) {
        const Mask vr = v ^ project(U_onb, v);
        if (vr != 0) res.push_back(vr);
    }
    res = echelon(std::move(res));
    // Verify U perp residual reconstructs V.
    std::vector<Mask> all = U_onb;
    all.insert(all.end(), res.begin(), res.end());
    for (Mask v : V_basis)
        if (!in_span(all, v)) throw std::logic_error("residual_basis: span mismatch");
    for (Mask r : res)
        for (Mask u : U_onb)
            if (dot(r, u) != 0)
                throw std::logic_error("residual_basis: residual not orthogonal to U");
    return res;
}

}  // namespace frames
}  // namespace net
}  // namespace ed
