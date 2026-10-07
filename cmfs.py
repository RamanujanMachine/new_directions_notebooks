"""The CMFs of the history notebooks that ramanujantools does not provide.

ramanujantools supplies the pFq and Meijer G CMFs. The others are:

* integral CMFs, built by `integral_cmf` from a weight prod_i f_i(t)^(e_i):
  Rhin's log 2 CMF, Hata's pi CMF, and Salikhov's pi CMF K(a,b,g), which
  also carries the Zeilberger--Zudilin and Bai constructions;
* Mignotte's pi CMF, a Meijer G CMF on a fibre over Q(i);
* Eskandari's Catalan CMF, built from a recurrence.

ramanujantools walks only rational matrices, so CMFs over Q(i) are realified:
a row S = Re S + i Im S becomes (Re S, Im S). ramanujantools checks
conservation when it constructs each CMF.
"""
import sympy as sp
from ramanujantools import Matrix
from ramanujantools.cmf import CMF
from ramanujantools.cmf.meijer_g import MeijerG

t = sp.Symbol("t")


def integral_cmf(weight):
    """The CMF of the moments S_j = int t^j w(t) dt; see `integral_matrices`."""
    return CMF(integral_matrices(weight))


def integral_matrices(weight):
    """The axis matrices of the moments S_j = int t^j w(t) dt, j = 0, ..., deg D - 2.

    `weight` maps each factor f_i of w = prod_i f_i^(e_i) to its exponent e_i,
    which is x + c (raising the axis x multiplies w by f_i), -x + c (raising x
    divides w by f_i) or a constant. The path runs between roots of
    D = prod_i f_i, so integrating (D r w)' gives, for every polynomial r,
        int (D r' + (D' + D w'/w) r) w dt = 0.
    Taking r = t^k reduces every power of t to the moment basis 1, ..., t^(deg D - 2).
    The rows of the walk are the moment vectors (S_0, S_1, ...).
    """
    D = sp.expand(sp.prod(weight))
    rank = sp.degree(D, t) - 1
    R = sp.diff(D, t) + sum(e * sp.cancel(D / f) * sp.diff(f, t) for f, e in weight.items())

    def coordinates(p):
        """The coefficients of p(t) in the moment basis, modulo the relations."""
        p = sp.Poly(p, t)
        while p.degree() >= rank:
            k = p.degree() - rank
            relation = sp.Poly(k * D * t ** (k - 1) + R * t**k, t)
            p = sp.Poly(sp.cancel(p.as_expr() - p.LC() / relation.LC() * relation.as_expr()), t)
        return [sp.cancel(p.nth(j)) for j in range(rank)]

    def multiplication(f):
        """Row j, column i: the coefficient of t^j in f(t) t^i."""
        columns = [coordinates(f * t**i) for i in range(rank)]
        return Matrix(rank, rank, lambda j, i: columns[i][j])

    matrices = {}
    for f, e in weight.items():
        for x in sp.sympify(e).free_symbols:
            if e.coeff(x) == 1:
                matrices[x] = multiplication(f)
            else:
                matrices[x] = multiplication(f).subs({x: x + 1}).inverse()
    return {x: M.applyfunc(sp.factor) for x, M in matrices.items()}


def realify(matrices):
    """Axis matrices over Q(i) as real matrices of twice the size, acting on rows (Re S, Im S)."""
    def real(M):
        re = M.applyfunc(lambda x: sp.cancel(sp.re(sp.expand_complex(x))))
        im = M.applyfunc(lambda x: sp.cancel(sp.im(sp.expand_complex(x))))
        return Matrix.vstack(Matrix.hstack(re, im), Matrix.hstack(-im, re))

    real_axes = {x: sp.Symbol(x.name, real=True) for x in matrices}
    back = {y: x for x, y in real_axes.items()}
    return {x: real(M.subs(real_axes)).subs(back) for x, M in matrices.items()}


def rhin_log2():
    """Rhin's (1987) log 2 CMF: axes (e1, e2, b), rank 4.

    w = (t(1-t))^e1 (6t^2-5t+1)^e2 / (1+t)^b on [0, 1].
    """
    e1, e2, b = sp.symbols("e1 e2 b")
    return integral_cmf({t * (1 - t): e1, 6 * t**2 - 5 * t + 1: e2, 1 + t: -b})


def hata_pi():
    """Hata's (1993) pi CMF: axes (u, v, w, m), rank 3 over Q(i), realified to rank 6.

    J(u,v,w,m) = int_1^(1+i) (z-1)^u (z-2)^v (z-1-i)^w / z^m dz, written in t = z.
    """
    u, v, w, m = sp.symbols("u v w m")
    return CMF(realify(integral_matrices({t - 1: u, t - 2: v, t - 1 - sp.I: w, t: -m})))


def mignotte_pi():
    """Mignotte's (1974) pi CMF: axes (a0, ..., a5), rank 6 over Q(i), realified to rank 12.

    ramanujantools' Meijer G CMF G^(6,0)_(6,6)(z) on the fibre z = -i, b_0 = ... = b_5 = 0,
    whose state is (F, theta F, ..., theta^5 F) with
    F = G^(6,0)_(6,6)(-i | a_0, ..., a_5; 0, ..., 0).
    """
    meijer = MeijerG(6, 0, 6, 6)
    fibre = {sp.Symbol("z"): -sp.I, **{b: 0 for b in MeijerG.b_axes(6)}}
    return CMF(realify({a: meijer.M(a).subs(fibre) for a in MeijerG.a_axes(6)}))


def salikhov_pi():
    """Salikhov's (2008) pi CMF K(a,b,g): axes (a, b, g), rank 3.

    K(a,b,g) = -i int_(-1-2i)^(-1+2i) x^(2g) (x^4+6x^2+25)^a / (x^2-25)^b dx.
    In s = x^2 the weight is s^(g-1/2) (s^2+6s+25)^a (s-25)^(-b), written in t = s,
    and the moment basis 1, s, s^2 gives the row (K(a,b,g), K(a,b,g+1), K(a,b,g+2)).
    It carries Salikhov, Zeilberger--Zudilin and Bai.
    """
    a, b, g = sp.symbols("a b g")
    return integral_cmf({t: g - sp.Rational(1, 2), t**2 + 6 * t + 25: a, t - 25: -b})


def eskandari_catalan():
    """Eskandari's (2026) Catalan CMF: axes (k, m), rank 8.

    Eskandari's sequence u_k satisfies a_0(k) u_k + a_1(k) u_(k+1) + a_2(k) u_(k+2) = 0.
    Its binomial transforms U(k, m) = sum_i C(m, i) (-16)^i u_(k+i) satisfy a
    nine-term recurrence sum_j c_j(k, m) U(k+j, m) = 0 in k. M_k is its companion
    matrix and M_m = I - 16 M_k, since U(k, m+1) = U(k, m) - 16 U(k+1, m).
    (ramanujantools reserves the symbol n, so the sequence index is k here.)
    """
    k, m = sp.symbols("k m")
    p = lambda x: 20 * x**2 - 8 * x + 1
    q = lambda x: 3520 * x**6 + 5632 * x**5 + 2064 * x**4 - 384 * x**3 - 156 * x**2 + 16 * x + 7
    base = (-(2 * k + 1) ** 2 * (2 * k + 2) ** 2 * p(k + 2), q(k + 1), (2 * k + 3) ** 2 * (2 * k + 4) ** 2 * p(k + 1))

    c = [sp.S.Zero] * 9
    for r, a_r in enumerate(base):
        difference = sp.expand(a_r)  # the j-th forward difference of a_r in k
        for j in range(7):
            choose = sp.binomial(m + 6, j).expand(func=True)
            for h in range(7 - j):
                c[r + j + h] += choose * (-16) ** (j + h) * sp.binomial(6 - j, h) * difference
            difference = sp.expand(difference.subs(k, k + 1) - difference)
    c = [sp.expand(ck) for ck in c]

    M_k = Matrix(8, 8, lambda i, j: (1 if i == j + 1 else 0) if j < 7 else -c[i] / c[8])
    return CMF({k: M_k, m: Matrix.eye(8) - 16 * M_k})
