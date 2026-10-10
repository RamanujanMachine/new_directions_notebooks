"""The CMFs of the history notebooks that ramanujantools does not provide.

ramanujantools supplies the pFq and Meijer G CMFs. The others are:

* integral CMFs, built by `integral_cmf` from a weight prod_i f_i(t)^(e_i):
  Rhin's log 2 CMF, Hata's pi CMF, and Salikhov's pi CMF K(a,b,g), which
  also carries the Zeilberger--Zudilin and Bai constructions. Rhin's full
  log 2 field, whose symbolic matrices are too large to build, is given by
  its axis matrices at integer points (`integral_steps`);
* Mignotte's pi CMF, a Meijer G CMF on a fibre over Q(i), and the dual of
  Mahler's, the same field with eleven a-axes;
* Eskandari's Catalan CMF, built from a recurrence;
* the Dvornicich--Viola zeta(2) and zeta(3) CMFs, built by `functional_cmf`
  from the recurrence of Beukers' integrals.

ramanujantools walks only rational matrices, so CMFs over Q(i) are realified:
a row S = Re S + i Im S becomes (Re S, Im S). ramanujantools checks
conservation when it constructs each CMF.
"""
import sympy as sp
from flint import fmpq, fmpq_mat, fmpq_poly
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


def integral_steps(weight):
    """The axis matrices of `integral_matrices(weight)` at integer points, without symbolic matrices.

    Returns M(x, point): the axis matrix M_x at the integer point `point` (a dict
    over the axes), as an exact python-flint fmpq_mat. It is the reduction of
    `integral_matrices` with the exponents evaluated at `point` first, so a step
    costs the same however many axes the field has. This is for fields whose
    symbolic matrices are too large to build, such as `rhin_log2_full`; walk them
    one axis step at a time.
    """
    def rational(c):
        c = sp.Rational(c)
        return fmpq(int(c.p), int(c.q))

    def flint(f):
        return fmpq_poly([rational(c) for c in reversed(sp.Poly(f, t).all_coeffs())])

    factors = {f: flint(f) for f in weight}
    D = fmpq_poly([1])
    for p in factors.values():
        D *= p
    rank = D.degree() - 1
    parts = {f: D // p * p.derivative() for f, p in factors.items()}  # D f'/f
    T = fmpq_poly([0, 1])

    def multiplication(f, point):
        """Row j, column i: the coefficient of t^j in f(t) t^i, at `point`."""
        R = D.derivative()
        for g, e in weight.items():
            R += parts[g] * rational(sp.sympify(e).subs(point))
        relations = [R * T**k + (k * D * T ** (k - 1) if k else 0) for k in range(factors[f].degree())]
        columns = []
        for i in range(rank):
            p = factors[f] * T**i
            while p.degree() >= rank:
                relation = relations[p.degree() - rank]
                if relation.degree() < p.degree():
                    raise ZeroDivisionError(f"the reduction has a pole at {point}")
                p -= p[p.degree()] / relation[p.degree()] * relation
            columns.append([p[j] for j in range(rank)])
        return fmpq_mat([[columns[i][j] for i in range(rank)] for j in range(rank)])

    def M(x, point):
        for f, e in weight.items():
            if sp.sympify(e).coeff(x) == 1:
                return multiplication(f, point)
            if sp.sympify(e).coeff(x) == -1:
                return multiplication(f, {**point, x: point[x] + 1}).inv()
        raise KeyError(x)

    return M


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


def rhin_log2_full():
    """Rhin's (1987) full log 2 field: axes (e1, ..., e6, b), rank 14, at integer points.

    w = p_1^e1 ... p_6^e6 / (1+t)^b on [0, 1], with the polynomials of Rhin's Appendix 1:
    p_1, ..., p_6 = t(1-t), t^2+2t-1, 6t^2-5t+1, 7t^2-6t+1, 13t^2-11t+2, 2t^4+3t^3+3t^2-5t+1.
    The face e2 = e4 = e5 = e6 = 0 is `rhin_log2` (whose e2 is e3 here). The symbolic
    14 x 14 matrices in seven variables are too large to build, so this returns
    M(x, point) of `integral_steps`, the axis matrices at integer points.
    """
    e1, e2, e3, e4, e5, e6, b = sp.symbols("e1:7 b")
    p = (t * (1 - t), t**2 + 2 * t - 1, 6 * t**2 - 5 * t + 1, 7 * t**2 - 6 * t + 1,
         13 * t**2 - 11 * t + 2, 2 * t**4 + 3 * t**3 + 3 * t**2 - 5 * t + 1)
    return integral_steps({**dict(zip(p, (e1, e2, e3, e4, e5, e6))), 1 + t: -b})


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


def mahler_pi(m=10):
    """The dual of Mahler's (1953) pi CMF: axes (a0, ..., am), rank m+1 over Q(i), realified.

    Mahler's CMF is ramanujantools' Meijer G CMF G^(m+1,0)_(m+1,m+1)(z) on the fibre
    z = -i, b_0 = ... = b_m = 0: the field of `mignotte_pi`, with Mahler's m = 10 in place
    of Mignotte's 5. Its state is (F, theta F, ..., theta^m F). On the fibre the monic
    equation [-i prod_j (theta - a_j + 1) - theta^(m+1)] / (-1 - i) has the coefficients
    (1 + i) s_k / 2, s_k those of prod_j (theta - a_j + 1), so its theta-companion matrix
    is C = C_re + i C_im with the last columns of C_re and C_im both -s / 2.
    The a_j-lowering matrix is -(C - (a_j - 1) I), and the a_j-raising matrix is its
    inverse at a + e_j; inverting it symbolically at rank 11 is slow, and so is expanding
    the rank-11 equation with `MeijerG`. The dual CMF (inverse transposes) needs neither:
    its a_j matrix at a is the transposed realified lowering matrix at a + e_j. Its walks
    are the inverse transposes of the walks of the realified Mahler CMF, exactly as
    `.dual()` of that CMF would give; m = 5 gives the dual of `mignotte_pi`.
    Unlike the other CMFs here it is built without ramanujantools' symbolic conservation
    check, which at rank 22 in 11 variables does not finish in reasonable time. For m = 5
    the check passes; for m = 10 conservation is checked exactly at regular points, and
    it follows in general from the commuting Mellin shifts of the Meijer G function.
    """
    rank = m + 1
    a = MeijerG.a_axes(rank)
    theta = sp.Symbol("theta")
    duals = {}
    for j, x in enumerate(a):
        # prod_i (theta - a_i + 1) at the point a + e_j, collected by powers of theta
        product = sp.prod(
            (sp.Poly(theta - y + 1 - int(y == x), theta, *a) for y in a), start=sp.Poly(1, theta, *a)
        )
        s = [{} for _ in range(rank + 1)]
        for (k, *powers), c in product.as_dict().items():
            s[k][tuple(powers)] = c
        half = [-sp.Poly.from_dict(s[k], *a).as_expr() / 2 for k in range(rank)]
        C_re = Matrix(rank, rank, lambda i, k: half[i] if k == rank - 1 else int(i == k + 1))
        C_im = Matrix(rank, rank, lambda i, k: half[i] if k == rank - 1 else 0)
        # the lowering matrix -(C - a_j I) at a + e_j, transposed
        re, im = (x * Matrix.eye(rank) - C_re).transpose(), -C_im.transpose()
        # the transpose of the realified lowering matrix [[Re, Im], [-Im, Re]]
        duals[x] = Matrix.vstack(Matrix.hstack(re, -im), Matrix.hstack(im, re))
    return CMF(duals, validate=False)


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


def functional_cmf(recurrence, axis, factors):
    """The CMF of S(e_0, e_1, ...) = L(t^(e_0) prod_i f_i(t)^(e_i)) for a moment functional L.

    L is a linear functional on Q[t] whose moments J_l = L(t^l) satisfy
        sum_s p_s(l) J_(l+s) = 0  for every l >= 1,
    given as `recurrence` = {s: p_s(l)}, polynomials in the symbol l, s >= -1.
    `axis` is the exponent e_0 of t and `factors` maps each polynomial f_i(t)
    to its exponent e_i, a symbol (raising it multiplies the integrand by f_i).

    With theta = t d/dt the recurrence says L(Omega phi) = 0 for every polynomial
    phi with phi(0) = 0, where Omega = sum_s t^s p_s(theta). Taking
    phi = t^(e_0+1) prod_i f_i^(e_i+r), r = max_s deg p_s, gives
        Omega phi = t^(e_0) prod_i f_i^(e_i) Q(t),  Q = sum_j q_j t^j,
    so sum_j q_j S(e_0+j, e_1, ...) = 0 for all e_0, e_i >= 0. M_(e_0) is its
    companion matrix and M_(e_i) = f_i(M_(e_0)) for linear f_i.
    The rows of the walk are the vectors (S(e_0, ...), ..., S(e_0+deg Q-1, ...)).
    """
    l = sp.Symbol("l")
    r = max(sp.degree(p, l) for p in recurrence.values())
    log_derivative = axis + 1 + sum((e + r) * t * sp.diff(f, t) / f for f, e in factors.items())
    powers = [sp.Integer(1)]  # theta^j phi = phi * powers[j]
    for _ in range(r):
        powers.append(sp.cancel(log_derivative * powers[-1] + t * sp.diff(powers[-1], t)))
    omega = sum(t**s * sum(sp.Poly(p, l).nth(j) * powers[j] for j in range(r + 1)) for s, p in recurrence.items())
    Q = sp.Poly(sp.cancel(t * sp.prod(f**r for f in factors) * omega), t)
    rank = Q.degree()
    q = [sp.factor(Q.nth(j)) for j in range(rank + 1)]

    M = Matrix(rank, rank, lambda i, j: (1 if i == j + 1 else 0) if j < rank - 1 else -q[i] / q[rank])
    matrices = {axis: M}
    for f, e in factors.items():
        c0, c1 = sp.Poly(f, t).nth(0), sp.Poly(f, t).nth(1)
        assert sp.degree(f, t) == 1
        matrices[e] = c0 * Matrix.eye(rank) + c1 * M
    return CMF(matrices)


def dvornicich_viola_zeta3():
    """Dvornicich--Viola's (1987) zeta(3) CMF: axes (e0, e1, e2), rank 8.

    S(e0, e1, e2) = int_[0,1]^3 f^e0 (1-34f)^e1 (1-35f)^e2 dx dy dz / (1-(1-xy)z),
    f = x(1-x)y(1-y)z(1-z)/(1-(1-xy)z). Its moments J_l = S(l, 0, 0) are Beukers'
    integrals, J_l = 2(a_l zeta(3) - b_l) with Apery's numbers a_l, b_l:
        (l+1)^3 J_(l+1) - (34l^3+51l^2+27l+5) J_l + l^3 J_(l-1) = 0.
    The state is (S(e0, e1, e2), ..., S(e0+7, e1, e2)).
    """
    l = sp.Symbol("l")
    e0, e1, e2 = sp.symbols("e0 e1 e2")
    recurrence = {1: (l + 1) ** 3, 0: -(34 * l**3 + 51 * l**2 + 27 * l + 5), -1: l**3}
    return functional_cmf(recurrence, e0, {1 - 34 * t: e1, 1 - 35 * t: e2})


def dvornicich_viola_zeta2():
    """Dvornicich--Viola's (1987) zeta(2) CMF: axes (e0, e1, e2), rank 6.

    S(e0, e1, e2) = int_[0,1]^2 f^e0 (1-11f)^e1 (1-12f)^e2 dx dy / (1-xy),
    f = x(1-x)y(1-y)/(1-xy). Its moments K_l = S(l, 0, 0) are Beukers' integrals,
    K_l = (-1)^l (a_l zeta(2) - b_l) with Apery's numbers a_l, b_l:
        (l+1)^2 K_(l+1) + (11l^2+11l+3) K_l - l^2 K_(l-1) = 0.
    The state is (S(e0, e1, e2), ..., S(e0+5, e1, e2)).
    """
    l = sp.Symbol("l")
    e0, e1, e2 = sp.symbols("e0 e1 e2")
    recurrence = {1: (l + 1) ** 2, 0: 11 * l**2 + 11 * l + 3, -1: -(l**2)}
    return functional_cmf(recurrence, e0, {1 - 11 * t: e1, 1 - 12 * t: e2})
