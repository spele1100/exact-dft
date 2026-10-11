// exact-dft — L3: synchronization / scheduling (paper section 4,
// "Synchronizing tensor factors in linear array work").
//
//   * Linear-time traversal rule (sec 4.1): mixed-radix digit DFS with the
//     first digit most significant visits < 2R prefix nodes (eq
//     eq:prefix-nodes); every fixed-work per-extension state update is O(R)
//     total. Implemented by SectorPacker's four-state DFS.
//   * Packing sectors (sec 4.2, Lemma lem:sector-address): a layer of
//     disjoint pairs on an axis partitions it into blocks; the sector start
//     formula S = sum_j (prod_{i<j} q_i) p_j tau_j (eq eq:sector-start) and
//     the four-state update (S, Q, o, I) (eq eq:sector-updates) generate the
//     packing permutation, its inverse, and the sector list in O(R).
//   * Synchronized transform (sec 4.3, Prop prop:tensor-fourier): the tensor
//     product (tensor_j F_{r_j}) is computed by compiling each axis with the
//     L2 exact-width compiler and scheduling the words on the R = prod r_j
//     coordinates through the same mixed-radix traversal. All scalars live
//     in the composite ring Q(zeta_L), L = lcm(radices), via ed::embed.
//
// Deviations from the paper (continuing the M2 judgment call #3):
//   * The paper interleaves slot-by-slot across axes (tensor eq
//     eq:tensor-multiply) so simultaneous pair layers become independent
//     tensor-power calls on packed sectors. We execute each axis's compiled
//     word to completion before the next axis (words on distinct axes act on
//     disjoint index sets and commute exactly), so sector packing is not
//     needed for correctness of the tensor transform — it is exposed and
//     differentially tested as the standalone O(R) machinery it is, ready
//     for a slot-interleaved executor if pair-layer coloring is added.
//   * Within-axis pair layers still run serially (M2 deviation), so the
//     default axis tables are all-singleton (identity packing).
#pragma once
#include "cyclotomic.hpp"
#include "newton_toeplitz.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace ed {

// ---------------------------------------------------------------------------
// Per-axis local block tables (paper sec 4.2: "These local tables are part
// of the compiled layer"). For each actual axis digit a in [0, r):
//   p[a] = number of coordinates in preceding blocks,
//   q[a] = width of a's block (1 = singleton, 2 = pair),
//   t[a] = position within the block.
// ---------------------------------------------------------------------------
struct SectorTables {
    int r = 0;
    std::vector<int> p, q, t;

    // All-singleton blocks in coordinate order (identity packing: every
    // sector has width 1 and S(a) = a itself).
    static SectorTables identity(int r) {
        if (r < 1) throw std::invalid_argument("SectorTables: r must be >= 1");
        SectorTables s;
        s.r = r;
        s.p.resize(r); s.q.assign(r, 1); s.t.assign(r, 0);
        for (int a = 0; a < r; ++a) s.p[a] = a;
        return s;
    }

    // Disjoint ordered pairs first (given order; first coordinate is t=0),
    // then the unused digits as singletons in ascending order. Pairs may be
    // nonconsecutive in the original coordinate order (paper fig:sectors).
    static SectorTables from_pairs(int r, const std::vector<std::pair<int, int>>& pairs) {
        if (r < 1) throw std::invalid_argument("SectorTables: r must be >= 1");
        std::vector<int> block(r, -1);  // digit -> block index
        std::vector<int> bt(r, -1);     // digit -> position within block
        for (size_t b = 0; b < pairs.size(); ++b) {
            const auto& [x, y] = pairs[b];
            if (x < 0 || x >= r || y < 0 || y >= r || x == y || block[x] >= 0 ||
                block[y] >= 0)
                throw std::invalid_argument("SectorTables: invalid pair set");
            block[x] = (int)b; block[y] = (int)b;
            bt[x] = 0; bt[y] = 1;
        }
        int next = (int)pairs.size();
        for (int a = 0; a < r; ++a)
            if (block[a] < 0) { block[a] = next++; bt[a] = 0; }
        // Block start offsets: p of a block = number of coordinates in all
        // preceding blocks = for pairs 2*index, for singletons
        // 2*pairs + (index - pairs).
        SectorTables s;
        s.r = r;
        s.p.resize(r); s.q.resize(r); s.t.resize(r);
        for (int a = 0; a < r; ++a) {
            const int b = block[a];
            s.q[a] = b < (int)pairs.size() ? 2 : 1;
            s.p[a] = b < (int)pairs.size() ? 2 * b : 2 * (int)pairs.size() + (b - (int)pairs.size());
            s.t[a] = bt[a];
        }
        return s;
    }

    // Blocks in chosen order: (p, q, digits in t-order). Digits of one
    // block share the same p, so group by p; pairs may be nonconsecutive.
    struct Block { int p, q; std::vector<int> digits; };
    std::vector<Block> blocks() const {
        std::vector<int> idx(r, -1);  // p value -> block index (p < r always)
        std::vector<Block> bs;
        for (int a = 0; a < r; ++a) {
            if (idx[p[a]] < 0) {
                idx[p[a]] = (int)bs.size();
                bs.push_back({p[a], q[a], {}});
            }
            bs[idx[p[a]]].digits.push_back(a);
        }
        for (Block& b : bs)
            std::sort(b.digits.begin(), b.digits.end(),
                      [&](int x, int y) { return t[x] < t[y]; });
        // Block order must agree with the p-offsets assigned at construction
        // (p = coordinates in PRECEDING blocks), i.e. ascending p.
        std::sort(bs.begin(), bs.end(),
                  [](const Block& x, const Block& y) { return x.p < y.p; });
        return bs;
    }
};

// One sector: packed start S, width Q = 2^k, k = pair count (paper Lemma
// lem:sector-address).
struct SectorInfo {
    long long S = 0, Q = 1;
    int pairs = 0;
};

// ---------------------------------------------------------------------------
// SectorPacker: packing permutation + sector enumeration via the four-state
// DFS (S, Q, o, I) and the block-choice DFS. O(R) state updates.
// ---------------------------------------------------------------------------
class SectorPacker {
  public:
    SectorPacker(std::vector<int> radices, std::vector<SectorTables> tables)
        : r_(std::move(radices)), tab_(std::move(tables)) {
        ell_ = (int)r_.size();
        if (ell_ == 0) throw std::invalid_argument("SectorPacker: no axes");
        R_ = 1;
        for (int j = 0; j < ell_; ++j) {
            if (r_[j] < 1) throw std::invalid_argument("SectorPacker: radix >= 1");
            if (tab_[j].r != r_[j])
                throw std::invalid_argument("SectorPacker: table/axis mismatch");
            // Overflow-safe for test sizes.
            R_ *= r_[j];
        }
        tau_.assign(ell_ + 1, 1);
        for (int j = ell_ - 1; j >= 0; --j) tau_[j] = tau_[j + 1] * r_[j];
    }

    long long R() const { return R_; }
    int ell() const { return ell_; }
    // Number of digit-DFS prefix extensions performed (paper eq
    // eq:prefix-nodes: < 2R). Exposed for the O(R) timing sanity test.
    long long state_updates() const { return state_updates_; }

    // packed -> original address permutation, by the four-state digit DFS
    // (eq eq:sector-updates): packed address = S + o at the leaf.
    const std::vector<int>& packed_to_orig() {
        if (!perm_done_) {
            perm_.assign(R_, -1);
            dfs_digits(0, 0, 1, 0, 0);
            perm_done_ = true;
        }
        return perm_;
    }

    // Inverse permutation (original -> packed), also O(R).
    const std::vector<int>& orig_to_packed() {
        packed_to_orig();
        if (!inv_done_) {
            inv_.assign(R_, -1);
            for (long long s = 0; s < R_; ++s) inv_[perm_[s]] = (int)s;
            inv_done_ = true;
        }
        return inv_;
    }

    // Sectors in lexicographic block-choice order (block-choice DFS,
    // O(#block prefixes) <= O(R)).
    const std::vector<SectorInfo>& sectors() {
        if (!sectors_done_) {
            sectors_.clear();
            dfs_blocks(0, 0, 1, 0);
            sectors_done_ = true;
        }
        return sectors_;
    }

  private:
    void dfs_digits(int j, long long S, long long Q, long long o, long long I) {
        if (j == ell_) {
            perm_[S + o] = (int)I;
            return;
        }
        for (int a = 0; a < r_[j]; ++a) {
            ++state_updates_;
            const SectorTables& tb = tab_[j];
            dfs_digits(j + 1,
                       S + Q * (long long)tb.p[a] * tau_[j + 1],
                       Q * tb.q[a],
                       o * tb.q[a] + tb.t[a],
                       I * r_[j] + a);
        }
    }

    void dfs_blocks(int j, long long S, long long Q, int pairs) {
        if (j == ell_) {
            sectors_.push_back({S, Q, pairs});
            return;
        }
        for (const SectorTables::Block& b : tab_[j].blocks()) {
            dfs_blocks(j + 1, S + Q * (long long)b.p * tau_[j + 1], Q * b.q,
                       pairs + (b.q == 2 ? 1 : 0));
        }
    }

    std::vector<int> r_;
    std::vector<SectorTables> tab_;
    int ell_ = 0;
    long long R_ = 1, state_updates_ = 0;
    std::vector<long long> tau_;
    std::vector<int> perm_, inv_;
    std::vector<SectorInfo> sectors_;
    bool perm_done_ = false, inv_done_ = false, sectors_done_ = false;
};

// ---------------------------------------------------------------------------
// TensorFourier: (tensor_j F_{r_j}) on R = prod r_j coordinates over the
// composite ring Q(zeta_L), L = lcm(radices), via per-axis L2 compiled words
// (scalars embedded) scheduled by the mixed-radix traversal.
// ---------------------------------------------------------------------------
class TensorFourier {
  public:
    explicit TensorFourier(std::vector<int> radices) : r_(std::move(radices)) {
        ell_ = (int)r_.size();
        if (ell_ == 0) throw std::invalid_argument("TensorFourier: no axes");
        R_ = 1; L_ = 1;
        for (int j = 0; j < ell_; ++j) {
            if (r_[j] < 1) throw std::invalid_argument("TensorFourier: radix >= 1");
            R_ *= r_[j];
            L_ = std::lcm(L_, r_[j]);
        }
        for (int j = 0; j < ell_; ++j)
            words_.emplace_back(r_[j], r_[j] == 1 ? 0 : (int)L_);
        tau_.assign(ell_ + 1, 1);
        for (int j = ell_ - 1; j >= 0; --j) tau_[j] = tau_[j + 1] * r_[j];
    }

    long long R() const { return R_; }
    int ring() const { return (int)L_; }
    int ell() const { return ell_; }
    const std::vector<NTReport>& axis_reports() const { return axis_rep_; }

    // Apply (tensor_j F_{r_j}) in place. x must have R entries, all in ring
    // Q(zeta_L). axis_order optionally permutes the axis application order
    // (default 0..ell-1) — used by the traversal-order-independence test;
    // the result is identical because per-axis words touch disjoint index
    // sets and commute.
    void apply(std::vector<Cyclotomic>& x, const std::vector<int>* axis_order = nullptr) {
        if ((long long)x.size() != R_)
            throw std::invalid_argument("TensorFourier::apply: length != prod radices");
        for (const Cyclotomic& c : x)
            if (c.d() != L_)
                throw std::invalid_argument("TensorFourier::apply: ring mismatch");
        std::vector<int> order;
        if (axis_order) {
            order = *axis_order;
            if ((int)order.size() != ell_) throw std::invalid_argument("bad axis order");
        } else {
            order.resize(ell_);
            std::iota(order.begin(), order.end(), 0);
        }
        std::vector<Cyclotomic> sub;
        for (int j : order) {
            if (r_[j] == 1) continue;  // F_1 = id
            sub.clear();
            apply_axis_dfs(x, j, 0, 0, sub);
            axis_rep_.push_back(words_[j].report());
        }
    }

    // Introspection (spec: schedule_report / feeds compile_report(n) in M4).
    struct Report {
        long long R = 0, L = 1, ell = 0;
        long long groups = 0;          // axis-word invocations
        long long prefix_extensions = 0;  // digit-DFS state updates (< 2R)
        std::vector<NTReport> axes;    // per-axis compiled-word reports
    };

    Report report() const {
        Report rp;
        rp.R = R_; rp.L = L_; rp.ell = ell_;
        rp.groups = groups_;
        rp.prefix_extensions = prefix_extensions_;
        rp.axes = axis_rep_;
        return rp;
    }

  private:
    // Enumerate all digit tuples with axis j varying (mixed-radix DFS over
    // the other axes, O(R/r_j) leaves; each leaf applies the compiled word
    // on the r_j strided coordinates base + k*tau_j).
    void apply_axis_dfs(std::vector<Cyclotomic>& x, int j, int i, long long base,
                        std::vector<Cyclotomic>& sub) {
        if (i == ell_) {
            ++groups_;
            sub.resize(r_[j]);
            for (int k = 0; k < r_[j]; ++k) sub[k] = x[base + (long long)k * tau_[j + 1]];
            words_[j].apply(sub);
            for (int k = 0; k < r_[j]; ++k)
                x[base + (long long)k * tau_[j + 1]] = std::move(sub[k]);
            return;
        }
        if (i == j) {  // digit of the varying axis fixed to 0 in the base
            apply_axis_dfs(x, j, i + 1, base, sub);
            return;
        }
        for (int a = 0; a < r_[i]; ++a) {
            ++prefix_extensions_;
            apply_axis_dfs(x, j, i + 1, base + (long long)a * tau_[i + 1], sub);
        }
    }

    std::vector<int> r_;
    int ell_ = 0;
    long long R_ = 1, L_ = 1, groups_ = 0, prefix_extensions_ = 0;
    std::vector<long long> tau_;
    std::vector<NewtonToeplitz> words_;
    std::vector<NTReport> axis_rep_;
};

// Convenience for bindings/tests: compile + scheduled-run on deterministic
// input, returning the introspection report (spec compile_report(n) composes
// this with the M5 driver in M4+).
inline typename TensorFourier::Report schedule_report(const std::vector<int>& radices) {
    TensorFourier tf(radices);
    std::vector<Cyclotomic> x((size_t)tf.R(), Cyclotomic(tf.ring()));
    for (long long i = 0; i < tf.R(); ++i)
        x[(size_t)i] = Cyclotomic::rational(tf.ring(),
                                            Rational((int)(i % 7) + 1, 5));
    tf.apply(x);
    return tf.report();
}

}  // namespace ed
