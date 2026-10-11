// exact-dft — L2: exact-width Newton/Toeplitz compiler + Bennett-style replay.
//
// Paper: "Compiling a Fourier transform on its own coordinates" (local.tex).
//   Lemma loc:newton-reduction: F_r(u) = N D_N^{-1} N^T with
//     N = diag(H_i) * T * diag((-1)^k u^{k(k-1)/2}),
//     H_j = prod_{s=1..j}(1-u^s),  T unit lower-triangular Toeplitz, T_ik = H_{i-k}^{-1},
//     D_N = diag(N_kk) = diag(H_k * (-1)^k u^{k(k-1)/2}).
//   Lemma loc:toeplitz: T_v = [[I,0],[E,I]] * diag(T_p, T_{v-p}), E = B T_p^{-1},
//     cross block applied through displacement-rank chunks (3 explicit outer
//     products, Kailath–Kung–Morf), each chunk implemented by a linear DAG and
//     run by Lemma loc:replay on coordinates BORROWED from the same v-coordinate
//     block (arbitrary initial contents, restored exactly).
//   Lemma loc:replay (Bennett compute–copy–reverse): forward sweep
//     z -> Uz + Tx, output reads into y, undo sweep, then source-free sweep and
//     subtract FUz, undo again. Net: (x,y,z) -> (x, y + Mx, z), z arbitrary.
//
// Deviations from the paper (per spec judgment call #3, documented):
//   * The paper's convolution DAGs use power-of-two-root FFT circuits with
//     O((a+e)log(a+e)) gates and bounded fan-out; we build direct sum-chain
//     DAGs (one shear per term of the two-convolution formula). Exactness is
//     identical; only the asymptotic gate count / layer bound differs. As a
//     consequence we never need power-of-two roots: all scalars live in
//     Q(zeta_r), the same ring as the oracle, so differential tests compare
//     bit-exactly with no ring conversion.
//   * Shears execute serially as immediate vector operations; we do not
//     materialize the disjoint-pair shear-layer coloring (it matters only for
//     the synchronization width proof, M3+). Layer counts are reported as
//     shear op counts instead.
//
// Width guarantee kept: `apply` transforms data on EXACTLY r coordinates
// (the input vector, in place). Gate coordinates are borrowed from the same
// block and restored, per the fit test g + a + e <= v (paper eq loc:fit).
#pragma once
#include "cyclotomic.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace ed {

// ---------------------------------------------------------------------------
// Field inverse in K = Q(zeta_d): Gaussian elimination on the multiplication
// matrix. NOTE: intermediate minors grow fast; int64 rationals overflow
// around d ~ 12-16 for dense elements. Kept for tests/small use — the
// NewtonToeplitz constructor below uses the closed form for 1/(1-zeta^s)
// instead, which produces already-reduced minimal-size scalars.
// ---------------------------------------------------------------------------
inline Cyclotomic cyclotomic_inverse(const Cyclotomic& a) {
    if (a.is_zero()) throw std::invalid_argument("cyclotomic_inverse: zero");
    const int d = a.d();
    const size_t n = (size_t)a.degree();
    if (n == 1) return Cyclotomic::rational(d, Rational(1) / a.coeffs()[0]);

    // Augmented matrix [M | I]; column j of M is a * t^j.
    std::vector<std::vector<Rational>> M(n, std::vector<Rational>(2 * n, Rational(0)));
    for (size_t j = 0; j < n; ++j) {
        // Column j = a * t^j, reduced mod Phi_d: shift a's coefficients by j.
        std::vector<Rational> shifted(n + j, Rational(0));
        for (size_t i = 0; i < n; ++i) shifted[i + j] = a.coeffs()[i];
        std::vector<Rational> col = a.reduce(std::move(shifted));
        for (size_t i = 0; i < n; ++i) M[i][j] = col[i];
        M[j][n + j] = Rational(1);
    }
    for (size_t col = 0; col < n; ++col) {
        size_t piv = col;
        while (piv < n && M[piv][col].is_zero()) ++piv;
        if (piv == n) throw std::logic_error("cyclotomic_inverse: singular (bug)");
        std::swap(M[col], M[piv]);
        const Rational inv = Rational(1) / M[col][col];
        for (size_t k = col; k < 2 * n; ++k) M[col][k] *= inv;
        for (size_t row = 0; row < n; ++row) {
            if (row == col || M[row][col].is_zero()) continue;
            const Rational f = M[row][col];
            for (size_t k = col; k < 2 * n; ++k) M[row][k] -= f * M[col][k];
        }
    }
    std::vector<Rational> x(n);
    for (size_t i = 0; i < n; ++i) x[i] = M[i][n];  // x = M^{-1} e_0 (first column)
    Cyclotomic res(d, std::move(x));
    if (a * res != Cyclotomic::integer(d, 1))
        throw std::logic_error("cyclotomic_inverse: verification failed");
    return res;
}

// Closed form: for x a primitive m-th root of unity,
//   1/(1-x) = -(1/m) * sum_{k=1..m-1} k x^k   (x != 1).
// Gives the already-reduced inverse of 1 - zeta_r^s with denominator m = ord.
inline Cyclotomic inverse_one_minus(const Cyclotomic& u, int s) {
    const int r = u.d();
    const int m = r / std::gcd(r, s);
    if (m == 1) throw std::invalid_argument("inverse_one_minus: u^s = 1");
    const Cyclotomic w = u.pow(s);  // order m
    Cyclotomic acc(r);
    Cyclotomic wp = Cyclotomic::integer(r, 1);
    for (int k = 1; k < m; ++k) {
        wp = wp * w;
        acc += wp * Cyclotomic::rational(r, Rational(k));
    }
    return acc * Cyclotomic::rational(r, Rational(-1, m));
}

// ---------------------------------------------------------------------------
// Introspection report (spec L2 compile_report).
// ---------------------------------------------------------------------------
struct NTReport {
    int r = 0;
    long long shears = 0;        // shear operations executed (all replay rounds)
    long long gate_coords = 0;   // borrowed gate coordinates assigned
    long long replays = 0;       // Lemma loc:replay invocations
    long long cross_pairs = 0;   // chunk pairs processed
    long long direct_blocks = 0; // direct-word fallbacks (paper's v-below-threshold case)
    int64_t max_num = 0;         // growth of prepared scalars (|numerator| max)
    int64_t max_den = 1;         // growth of prepared scalars (denominator max)

    void note_scalar(const Cyclotomic& c) {
        for (const Rational& q : c.coeffs()) {
            max_num = std::max(max_num, q.num < 0 ? -q.num : q.num);
            max_den = std::max(max_den, q.den);
        }
    }
};

// ---------------------------------------------------------------------------
// Linear DAG + Bennett replay (Lemma loc:replay, direct-execution form).
//
// A gate is a sequence of single-predecessor shears INTO a borrowed gate
// coordinate; the paper's 2-predecessor gates are split. from_source edges
// read live x coordinates; gate edges read earlier gate coordinates.
// ---------------------------------------------------------------------------
struct ReplayDAG {
    struct Pred {
        int coord;              // arena index
        bool from_source;       // true = live source coordinate, false = gate coordinate
        Cyclotomic coeff;
    };
    struct Out {
        int target;             // arena index of a y coordinate
        int coord;              // arena index of source (gate) or x coordinate
        bool from_source;
        Cyclotomic coeff;
    };
    std::vector<int> gate_coord;              // arena index per gate (topological order)
    std::vector<std::vector<Pred>> preds;     // per gate
    std::vector<Out> outs;

    void run(std::vector<Cyclotomic>& arena, NTReport& rep) const {
        const size_t G = gate_coord.size();
        // Sweep 1: z -> Uz + Tx (all edges).
        for (size_t g = 0; g < G; ++g)
            for (const Pred& p : preds[g])
                if (!p.coeff.is_zero()) {
                    arena[gate_coord[g]] += p.coeff * arena[p.coord];
                    ++rep.shears;
                }
        // Output round 1: y += F(Uz + Tx) + Dx.
        for (const Out& o : outs)
            if (!o.coeff.is_zero()) {
                arena[o.target] += o.coeff * arena[o.coord];
                ++rep.shears;
            }
        // Undo sweep 1.
        for (size_t g = G; g-- > 0;)
            for (size_t k = preds[g].size(); k-- > 0;) {
                const Pred& p = preds[g][k];
                if (!p.coeff.is_zero()) {
                    arena[gate_coord[g]] -= p.coeff * arena[p.coord];
                    ++rep.shears;
                }
            }
        // Sweep 2: source edges omitted -> computes Uz only.
        for (size_t g = 0; g < G; ++g)
            for (const Pred& p : preds[g])
                if (!p.from_source && !p.coeff.is_zero()) {
                    arena[gate_coord[g]] += p.coeff * arena[p.coord];
                    ++rep.shears;
                }
        // Output round 2: subtract F(Uz). Direct-source outputs omitted.
        for (const Out& o : outs)
            if (!o.from_source && !o.coeff.is_zero()) {
                arena[o.target] -= o.coeff * arena[o.coord];
                ++rep.shears;
            }
        // Undo sweep 2.
        for (size_t g = G; g-- > 0;)
            for (size_t k = preds[g].size(); k-- > 0;) {
                const Pred& p = preds[g][k];
                if (!p.from_source && !p.coeff.is_zero()) {
                    arena[gate_coord[g]] -= p.coeff * arena[p.coord];
                    ++rep.shears;
                }
            }
        ++rep.replays;
    }
};

// ---------------------------------------------------------------------------
// NewtonToeplitz: compiled F_r(zeta_r) on exactly r coordinates.
// ---------------------------------------------------------------------------
class NewtonToeplitz {
  public:
    explicit NewtonToeplitz(int r) : r_(r) {
        if (r < 1 || r > kMaxCyclotomicOrder)
            throw std::invalid_argument("NewtonToeplitz: r out of range");
        rep_.r = r;
        if (r == 1) return;
        const Cyclotomic u = Cyclotomic::zeta(r);
        const Cyclotomic one = Cyclotomic::integer(r, 1);

        // H_j = prod_{s=1..j} (1 - u^s)  (all nonzero: u primitive).
        H_.assign(r, Cyclotomic(r));
        H_[0] = one;
        Cyclotomic cur = one;
        for (int j = 1; j < r; ++j) {
            cur = cur * (one - u.pow(j));
            H_[j] = cur;
        }

        // Toeplitz sequence t_j = H_j^{-1} (RING inverse, paper Lemma
        // loc:newton-reduction: T_ik = H_{i-k}^{-1}); obtained via the closed
        // form products invH below. No field-division elimination needed.
        std::vector<Cyclotomic> invH(r, Cyclotomic(r));
        invH[0] = one;
        for (int j = 1; j < r; ++j) invH[j] = invH[j - 1] * inverse_one_minus(u, j);
        t_ = invH;

        // Reciprocal series g of t (paper eq loc:reciprocal; g_0 = 1):
        // (t * g)_k = delta_{k0}; g gives T_p^{-1} for every block.
        g_.assign(r, Cyclotomic(r));
        g_[0] = one;
        for (int k = 1; k < r; ++k) {
            Cyclotomic acc(r);
            for (int j = 1; j <= k; ++j) acc += t_[j] * g_[k - j];
            g_[k] = -acc;
        }

        // Diagonal factors of N and D_N^{-1}.
        D1_.assign(r, Cyclotomic(r));   // diag(H_i)
        D2_.assign(r, Cyclotomic(r));   // diag((-1)^k u^{k(k-1)/2})
        DNinv_.assign(r, Cyclotomic(r));
        for (int k = 0; k < r; ++k) {
            D1_[k] = H_[k];
            Cyclotomic d2 = u.pow((int64_t)k * (k - 1) / 2 % r);
            if (k % 2 == 1) d2 = -d2;
            D2_[k] = d2;
            // 1/(H_k d2) = (1/H_k) * u^{-k(k-1)/2} * (-1)^k.
            DNinv_[k] = invH[k] * u.pow((r - (int64_t)k * (k - 1) / 2 % r) % r);
            if (k % 2 == 1) DNinv_[k] = -DNinv_[k];
        }
        for (const auto& v : {&H_, &t_, &g_, &D1_, &D2_, &DNinv_})
            for (const Cyclotomic& c : *v) rep_.note_scalar(c);
    }

    int r() const { return r_; }
    const NTReport& report() const { return rep_; }

    // Apply F_r(zeta_r) * x exactly, in place, on exactly the r coordinates
    // of x. All entries of x must live in ring Q(zeta_r).
    void apply(std::vector<Cyclotomic>& x) const {
        if ((int)x.size() != r_) throw std::invalid_argument("apply: length != r");
        for (const Cyclotomic& c : x)
            if (c.d() != r_) throw std::invalid_argument("apply: ring mismatch");
        if (r_ == 1) return;
        // N^T x = D2 (T^T (D1 x)); T^T = J T J (J = exchange matrix).
        for (int i = 0; i < r_; ++i) x[i] = D1_[i] * x[i];
        std::reverse(x.begin(), x.end());
        toeplitz_lower(x);
        std::reverse(x.begin(), x.end());
        for (int k = 0; k < r_; ++k) x[k] = (D2_[k] * DNinv_[k]) * x[k];
        // N (D_N^{-1} N^T x) = D1 (T (D2 ...)).
        for (int k = 0; k < r_; ++k) x[k] = D2_[k] * x[k];
        toeplitz_lower(x);
        for (int i = 0; i < r_; ++i) x[i] = D1_[i] * x[i];
    }

    // Lower triangular Toeplitz T (sequence t_) applied to a full width-r
    // vector, in place, using the recursive exact-width compiler.
    void toeplitz_lower(std::vector<Cyclotomic>& x) const {
        if ((int)x.size() != r_) throw std::invalid_argument("toeplitz_lower: width");
        toeplitz_(x, 0, r_);
    }

    // --- exposed prepared scalars (tests, compile_report) ---
    const std::vector<Cyclotomic>& H() const { return H_; }
    const std::vector<Cyclotomic>& t() const { return t_; }
    const std::vector<Cyclotomic>& g() const { return g_; }
    const std::vector<Cyclotomic>& D2() const { return D2_; }
    const std::vector<Cyclotomic>& DNinv() const { return DNinv_; }

  private:
    // Direct word fallback (paper: rows in decreasing order; lower
    // coordinates still hold original values).
    void direct_(std::vector<Cyclotomic>& X, int off, int v) const {
        for (int i = v - 1; i >= 0; --i) {
            Cyclotomic acc = X[off + i] * t_[0];
            for (int j = 0; j < i; ++j) acc += X[off + j] * t_[i - j];
            X[off + i] = std::move(acc);
        }
        ++rep_.direct_blocks;
    }

    void toeplitz_(std::vector<Cyclotomic>& X, int off, int v) const {
        if (v <= 0) return;
        // Admissible chunk size b: fit test g + a + e <= v with our gate
        // count g = 3(a+e) and worst pair a = e = b  =>  8b <= v.
        const int b = v / 8;
        if (b < 1) {  // v < 8: no admissible chunking; direct word.
            direct_(X, off, v);
            return;
        }
        const int p = v / 2;
        toeplitz_(X, off, p);        // children run on disjoint blocks
        toeplitz_(X, off + p, v - p);
        cross_(X, off, v, p, b);     // y2 += E * y1
    }

    // Cross update: for rows [p,v), cols [0,p):
    //   E_ij = sum_{nu=j}^{p-1} t_{i-nu} g_{nu-j}   (paper eq loc:cross),
    // applied chunkwise via the displacement identity
    //   E_ij - E_{i-1,j-1} = -t_{i-p} g_{p-j}      (eq loc:displacement-entry)
    // and the 3 explicit outer products (first row/col corrections).
    void cross_(std::vector<Cyclotomic>& X, int off, int v, int p, int b) const {
        const int q = v - p;
        const int nsrc = (p + b - 1) / b;
        const int ntgt = (q + b - 1) / b;

        // Rolling row of E via the displacement recurrence; keep full rows at
        // target-chunk starts (first local row of each chunk's M) and column
        // entries at source-chunk starts (first local column of each M).
        std::vector<std::vector<Cyclotomic>> rowE(ntgt);      // rowE[tau][j], j < p
        std::vector<std::vector<Cyclotomic>> colE(nsrc);      // colE[sigma][i-p], i < v
        std::vector<Cyclotomic> row(p, Cyclotomic(r_));
        // Base row i = p: E_p,j = sum_{nu=j}^{p-1} t_{p-nu} g_{nu-j}.
        for (int j = 0; j < p; ++j) {
            Cyclotomic acc(r_);
            for (int nu = j; nu < p; ++nu) acc += t_[p - nu] * g_[nu - j];
            row[j] = std::move(acc);
        }
        for (int i = p; i < v; ++i) {
            if (i > p) {
                // Interior (j >= 1): E_ij = E_{i-1,j-1} - t_{i-p} g_{p-j}.
                // First column (j = 0) is NOT covered by the displacement
                // identity; computed directly: E_i0 = sum_{nu<p} t_{i-nu} g_nu.
                const Cyclotomic tv = t_[i - p];
                std::vector<Cyclotomic> next(p, Cyclotomic(r_));
                for (int j = 1; j < p; ++j) next[j] = row[j - 1] - tv * g_[p - j];
                Cyclotomic e0(r_);
                for (int nu = 0; nu < p; ++nu) e0 += t_[i - nu] * g_[nu];
                next[0] = std::move(e0);
                row = std::move(next);
            }
            const int li = i - p;
            if (li % b == 0) rowE[li / b] = row;
            for (int s = 0; s < nsrc; ++s) {
                const int sb = s * b;
                if (sb < p) colE[s].push_back(row[sb]);
            }
        }

        for (int tau = 0; tau < ntgt; ++tau) {
            const int ta = p + tau * b;
            const int a = std::min(b, v - ta);
            for (int sig = 0; sig < nsrc; ++sig) {
                const int sb = sig * b;
                const int e = std::min(b, p - sb);
                apply_chunk_(X, off, v, p, ta, a, sb, e, rowE[tau], colE[sig]);
                ++rep_.cross_pairs;
            }
        }
    }

    // One chunk pair: M = submatrix of E (a x e) applied to source chunk
    // [sb, sb+e) added into target chunk [ta, ta+a), all coordinates local
    // to the block [off, off+v). Gate coordinates borrowed from the block.
    void apply_chunk_(std::vector<Cyclotomic>& X, int off, int v, int p, int ta, int a,
                      int sb, int e, const std::vector<Cyclotomic>& rowStart,
                      const std::vector<Cyclotomic>& colStart) const {
        // Branch vectors of the displacement decomposition.
        // M_ij = E_{ta+i, sb+j}; rowStart[j] = E_{ta, j}; colStart[i] = E_{p+i, sb}.
        std::vector<Cyclotomic> v1(a, Cyclotomic(r_)), w1(e, Cyclotomic(r_));
        for (int i = 0; i < a; ++i) v1[i] = -t_[(ta + i) - p];
        for (int j = 0; j < e; ++j) w1[j] = g_[p - (sb + j)];
        // R = (M - J_a M J_e^T) - v1 w1^T: zero in the interior; only first
        // row/column survive.
        std::vector<Cyclotomic> R0(e, Cyclotomic(r_)), Rc(a, Cyclotomic(r_));
        for (int j = 0; j < e; ++j) R0[j] = rowStart[sb + j] - v1[0] * w1[j];
        for (int i = 0; i < a; ++i) Rc[i] = colStart[ta + i - p] - v1[i] * w1[0];
        // branch 2: e_0 R_{0,*};  branch 3: (R_{*,0} - R_00 e_0) e_0^T.
        // v2 = e_0, w2 = R0;  v3 = (0, Rc[1..]), w3 = e_0.

        // ---- Build the replay DAG: 3 branches x (e b-gates + a c-gates) ----
        ReplayDAG dag;
        const long long need = 3 * (a + e);
        // Reserved: source and target chunk coordinates (block-local).
        auto reserved = [&](int c) {
            return (c >= sb && c < sb + e) || (c >= ta && c < ta + a);
        };
        std::vector<int> gates;
        gates.reserve((size_t)need);
        for (int c = 0; c < v && (long long)gates.size() < need; ++c)
            if (!reserved(c)) gates.push_back(c);
        if ((long long)gates.size() < need)
            throw std::logic_error("apply_chunk: fit violated (bug)");
        rep_.gate_coords += need;

        auto branch = [&]( const std::vector<Cyclotomic>& vv,
                          const std::vector<Cyclotomic>& ww) {
            const int base = (int)dag.gate_coord.size();
            // b-gates: b_l = sum_{j=l..e-1} w_{j-l} x_j  (shears from sources).
            for (int l = 0; l < e; ++l) {
                dag.gate_coord.push_back(off + gates[base + l]);
                dag.preds.emplace_back();
                for (int j = l; j < e; ++j)
                    if (!ww[j - l].is_zero())
                        dag.preds.back().push_back(
                            {off + sb + j, true, ww[j - l]});
            }
            // c-gates: c_i = sum_l v_{i-l} b_l  (shears from b-gates).
            for (int i = 0; i < a; ++i) {
                dag.gate_coord.push_back(off + gates[base + e + i]);
                dag.preds.emplace_back();
                for (int l = 0; l <= std::min(i, e - 1); ++l)
                    if (i - l < (int)vv.size() && !vv[i - l].is_zero())
                        dag.preds.back().push_back(
                            {off + gates[base + l], false, vv[i - l]});
            }
            // Outputs: y_i += c_i (gate reads; no direct-source outputs => D=0).
            for (int i = 0; i < a; ++i)
                dag.outs.push_back({off + ta + i, off + gates[base + e + i],
                                    false, Cyclotomic::integer(r_, 1)});
        };
        branch(v1, w1);
        {   // branch 2: v2 = e_0, w2 = R0
            std::vector<Cyclotomic> v2(a, Cyclotomic(r_));
            v2[0] = Cyclotomic::integer(r_, 1);
            branch(v2, R0);
        }
        {   // branch 3: v3 = (0, Rc[1], ...), w3 = e_0
            std::vector<Cyclotomic> v3(a, Cyclotomic(r_));
            for (int i = 1; i < a; ++i) v3[i] = Rc[i];
            std::vector<Cyclotomic> w3(e, Cyclotomic(r_));
            w3[0] = Cyclotomic::integer(r_, 1);
            branch(v3, w3);
        }
#ifdef ED_CHUNK_DEBUG
        auto before_ = X;
#endif
        dag.run(X, rep_);
#ifdef ED_CHUNK_DEBUG
        auto after_ = X;
        // verify: delta on targets == M * sources, M from direct cross formula
        for (int i = 0; i < a; ++i) {
            Cyclotomic want(r_);
            for (int j = 0; j < e; ++j) {
                Cyclotomic Eij(r_);
                for (int nu = sb + j; nu < p; ++nu) Eij += t_[(ta + i) - nu] * g_[nu - (sb + j)];
                want += Eij * X[off + sb + j];
            }
            if (!(after_[off + ta + i] - before_[off + ta + i] == want)) {
                std::fprintf(stderr, "CHUNK BUG r=%d ta=%d sb=%d a=%d e=%d i=%d\n", r_, ta, sb, a, e, i);
                if (a == 1 && e == 1) {
                    auto pr=[](const char*n, const Cyclotomic&c){ std::fprintf(stderr,"  %s:",n); for(auto&q:c.coeffs()) std::fprintf(stderr," %lld/%lld",(long long)q.num,(long long)q.den); std::fprintf(stderr,"\n"); };
                    Cyclotomic Eij(r_);
                    for (int nu = sb; nu < p; ++nu) Eij += t_[(ta) - nu] * g_[nu - sb];
                    pr("M_direct", Eij);
                    pr("rowStart[sb]", rowStart[sb]);
                    pr("colStart[ta-p]", colStart[ta - p]);
                    pr("v1[0]", v1[0]); pr("w1[0]", w1[0]);
                    pr("R0[0]", R0[0]); pr("Rc[0]", Rc[0]);
                }
                std::exit(2);
            }
        }
#endif
    }

    int r_;
    std::vector<Cyclotomic> H_, t_, g_, D1_, D2_, DNinv_;
    mutable NTReport rep_;
};

// Convenience for bindings/tests: compile + run on deterministic input,
// returning the introspection report (spec L2 compile_report).
inline NTReport compile_report(int r) {
    NewtonToeplitz nt(r);
    if (r > 1) {
        std::vector<Cyclotomic> x((size_t)r, Cyclotomic(r));
        for (int i = 0; i < r; ++i)
            x[i] = Cyclotomic::rational(r, Rational(i % 7 + 1, 5));
        nt.apply(x);
    }
    return nt.report();
}

}  // namespace ed
