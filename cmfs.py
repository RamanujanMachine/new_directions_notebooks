"""Custom CMFs and independent moment references for the history notebooks.

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
  from the recurrence of Beukers' integrals, and walked at integer points with
  `axis_steps`.
* independent reference implementations (the notebooks now use pFq directly):
  the Alladi--Robinson logarithmic and quadratic moment fields, the
  Heimonen--Matala-aho--Vaananen Jacobi-Pade moment field, and Hata's
  pi/sqrt(3) integral in rational rank-3 coordinates; Hata's (1992)
  rank-2 hypergeometric remainders for algebraic logarithms and pi combinations.
  `local_checks/verify_hypergeometric_replacements.py` compares all sampled fractions with
  these references; the quadratic and Hata (1993) replacements use even indices.
* Zudilin's (2002) zeta(5) coefficient seeds and determinant readout for the
  full 15-axis pFq(8, 7, 1) field, plus its rank-3 recurrence for cross-checks.

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


def gauss_arctan_pi_seeds(taylor_terms=0):
    """Rows P,Q for pi from the Gauss CF of the Gregory-series tail.

    pFq(2,1,-1), start (m+1/2,1;m+3/2), direction (1,1;2).
    A walk of k steps reads the source convergent M=2k as (P W)e0/(Q W)e0.
    Source: Borwein--Choi--Pigulla (2005), equations (8), (10), (12).
    """
    m = sp.sympify(taylor_terms)
    if m.is_Integer is not True or m < 0:
        raise ValueError("taylor_terms must be a nonnegative integer")
    partial = sum((-1)**j * sp.Rational(1, 2*j+1) for j in range(int(m)))
    a = m + sp.Rational(1, 2)
    A = (2*m+1)*(-1)**m / 4
    B = -(2*m+1)*(-1)**m * partial
    # F=A*pi+B and theta F=a/2-a*F at z=-1.
    return Matrix([[-B, a*B-a/2]]), Matrix([[A, -a*A]])


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


def axis_steps(cmf):
    """The axis matrices of `cmf` at integer points, as exact python-flint fmpq_mat.

    Returns M(x, point), like `integral_steps`. Each entry p/q of each symbolic
    axis matrix is compiled once into integer polynomials, so a step costs integer
    evaluations instead of a sympy substitution. This is for walks of tens of
    thousands of axis steps, such as the Dvornicich--Viola staircases.
    """
    axes = list(cmf.matrices)
    compiled = {}
    for x, matrix in cmf.matrices.items():
        rows = []
        for row in matrix.tolist():
            entries = []
            for entry in row:
                p, q = sp.fraction(sp.cancel(sp.together(entry)))
                cp, p = sp.Poly(p, *axes).clear_denoms()  # p = cp * numerator, integral
                cq, q = sp.Poly(q, *axes).clear_denoms()
                scale = sp.Rational(cq) / sp.Rational(cp)
                entries.append((
                    sp.lambdify(axes, p.as_expr(), modules=[{}]),
                    sp.lambdify(axes, q.as_expr(), modules=[{}]),
                    fmpq(int(scale.p), int(scale.q)),
                ))
            rows.append(entries)
        compiled[x] = rows

    def M(x, point):
        values = [int(point[y]) for y in axes]
        return fmpq_mat([[c * fmpq(int(p(*values)), int(q(*values))) for p, q, c in row] for row in compiled[x]])

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


def alladi_log(a):
    """Rank-2 CMF for E_n = int_0^1 t^n(1-t)^n/(a+t)^(n+1) dt.

    The start is (u,v,w)=(1,1,2), the direction is (1,1,1), and the
    moment state is (E(u,v,w), E(u+1,v,w)).
    """
    u, v, w = sp.symbols("u v w")
    return integral_cmf({t: u, 1 - t: v, a + t: -w})


def alladi_log_start(a):
    """Rows C,B at (1,1,2), for moments C+B*log((a+1)/a)."""
    return (Matrix([[-2, 3*a + sp.Rational(1, 2)]]),
            Matrix([[2*a + 1, -a*(3*a + 2)]]))


def alladi_quadratic(a):
    """Rank-3 CMF for E_n = int_0^1 t^n(1-t)^n/(t^2+a)^(n+1) dt."""
    u, v, w = sp.symbols("u v w")
    return integral_cmf({t: u, 1 - t: v, t**2 + a: -w})


def alladi_quadratic_start(a):
    """Rows C,B,H at (1,1,2), for moments C+B*c+H*log((a+1)/a).

    Here c=atan(1/sqrt(a))/sqrt(a). Along the diagonal, H vanishes in
    the first component. Rational multiples of c include pi/sqrt(3).
    """
    return (Matrix([[sp.Rational(1, 2*a), 0, -sp.Rational(3, 2)]]),
            Matrix([[-sp.Rational(1, 2), sp.Rational(1, 2),
                     sp.Rational(3*a, 2)]]),
            Matrix([[0, -sp.Rational(1, 2), sp.Rational(1, 2)]]))


def jacobi_pade(a):
    """Rank-2 CMF for Jacobi-Pade remainders of 2F1(1,1/2;3/2;-1/a).

    J_j(U,V,W) = int_0^1 t^(U+j)(1-t)^V(1+t/a)^(-W) dt.
    At (U,V,W)=(n-1/2,m,n+1), J_0 is proportional to the Pade remainder
    with l=n and m=floor(beta*n). The regular start (n,m)=(1,1) is
    (U,V,W)=(1/2,1,2).
    """
    U, V, W = sp.symbols("U V W")
    return integral_cmf({t: U,
                         1 - t: V, 1 + t/sp.Integer(a): -W})


def hata_hypergeometric(s):
    """Hata (1992), Eq. (3.10), after n integrations by parts.

    Moments J_j = int_0^1 t^(U+j)(1-t)^V/(1-t/s)^W dt, j=0,1.
    For source index n and m=floor(n/lambda), use
    (U,V,W)=(n-delta,n+m,n+1). Common scalar factors from integration
    by parts and denominator clearing cancel in the rational readout.
    """
    U, V, W = sp.symbols("U V W")
    return integral_cmf({t: U, 1-t: V, 1-t/sp.Integer(s): -W})


def hata_hypergeometric_start(s, delta):
    """Rows C,B at source n=m=0, J=C+B*2F1(1,1-delta;2-delta;1/s)."""
    b = 1-sp.Rational(delta)
    return Matrix([[0, -s/b]]), Matrix([[1/b, s/b]])


def jacobi_pade_start(a):
    """Rows C,B at n=m=1, for moments C+B*F(-1/a), F=2F1(1,1/2;3/2;z)."""
    return (Matrix([[-3*a**2, 5*a**3 + sp.Rational(4*a**2, 3)]]),
            Matrix([[a*(3*a + 1), -a**2*(5*a + 3)]]))


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


def hata_pi_sqrt3():
    """Hata (1993), Theorem 1.2: rank-3 rational moment CMF.

    Put lambda=exp(pi*i/3), h=(1+lambda)/2 and z=h*(1+t).
    Hata's integral of F(z)^n dz/z is h^(3*n) times J_0, where
    J_j = int_{-i/sqrt(3)}^{i/sqrt(3)}
          t^(v+j)*(t^2+1/3)^u/(1+t)^w dt.
    The source index n is (u,v,w)=(2*n,2*n,3*n+1).
    The state is (J_0,J_1,J_2); the contour avoids t=-1.
    """
    u, v, w = sp.symbols("u v w")
    return integral_cmf({t**2 + sp.Rational(1, 3): u, t: v, 1 + t: -w})


def hata_pi_sqrt3_start():
    """Rows C,B at (2,2,4), with J/(i*sqrt(3))=C+B*pi/sqrt(3).

    Integrate the rational partial fractions of
    t^(2+j)*(t^2+1/3)^2/(1+t)^4 for j=0,1,2. The logarithm's
    endpoint difference is i*pi/3; its residue therefore contributes B=R/3.
    The rational approximant at index n is -(C*walk)[0]/(B*walk)[0].
    """
    return (Matrix([[sp.Rational(370, 27), -sp.Rational(682, 27),
                     sp.Rational(17086, 405)]]),
            Matrix([[-sp.Rational(68, 9), sp.Rational(376, 27),
                     -sp.Rational(628, 27)]]))


class RhinLog3CMF:
    """Full rank-8, seven-axis moment field, evaluated at integer points.

    Rhin (1987), as explicitly reconstructed in Zudilin (2004), Section 3.4.
    The walk follows floor(N*trajectory[axis]) from start, raising polynomial
    axes before the pole order. It uses the same exact integration-by-parts
    matrices as integral_cmf, without expanding large symbolic entries.
    """
    def __init__(self):
        self.axes = sp.symbols('e1:7 b')
        self.factors = (t-1, t-sp.Rational(2,3), t-sp.Rational(4,3), 5*t-4,
                        17*t**2-34*t+16, 19*t**2-36*t+16)
        self.step = integral_steps({**dict(zip(self.factors,self.axes[:-1])),
                                    t:-self.axes[-1]})

    def walk(self, trajectory, depths, start):
        point = dict(start)
        W = fmpq_mat([[int(i==j) for j in range(8)] for i in range(8)])
        walks = []
        for N in depths:
            target = {a:start[a]+int(sp.floor(N*trajectory[a])) for a in self.axes}
            for a in self.axes:
                if target[a] < point[a]:
                    raise ValueError('walk requires nondecreasing depths and coordinates')
                while point[a] < target[a]:
                    W = W*self.step(a,point)
                    point[a] += 1
            walks.append(Matrix([[sp.Rational(str(W[i,j])) for j in range(8)] for i in range(8)]))
        return walks


def rhin_log3_seeds():
    """Rows for log 3 = log(4/3)-2 log(2/3), at all e_i=0,b=1."""
    a,b = sp.Rational(2,3),sp.Rational(4,3)
    P = Matrix([[0]+[-((b**j-1)-2*(a**j-1))/j for j in range(1,8)]])
    Q = Matrix([[1]+[0]*7])
    return P,Q


def salikhov_log3():
    """Salikhov (2007): y=(x-35)^2 moment field, rank 3, four axes.

    Weight y^u(y-25)^v(y-49)^w/(1225-y)^m, start (-1/2,0,0,1),
    direction (1,1,1,2). Source integrals are one half of these y moments.
    """
    u, v, w, m = sp.symbols("u v w m")
    return integral_cmf({t:u, t-25:v, t-49:w, 1225-t:-m})


def salikhov_log3_seeds():
    """Rows P,Q for log 3 = 2 log(3/2)+log(4/3), from source index 0.

    For endpoints h=5,7 the source moments are C_h+B log((35+h)/(35-h)).
    Seeds use the original dx moments; the common y-substitution factor cancels.
    """
    def C(h):
        return Matrix([[0, -h, -sp.Rational(h**3,3)-1225*h]])
    B = Matrix([[sp.Rational(1225**j,70) for j in range(3)]])
    return -(2*C(7)+C(5)), B


def huttner_logs():
    """Huttner (2003), Corollary 3.1(4), after z=1-t; rank 3, four axes.

    Weight (z-1)^u (z-2/3)^v (z-1/2)^w / z^m.
    Source n=1 starts at (2,2,2,3), direction (2,2,2,3).
    The common source sign (-1)^(n+1) cancels from rational readouts.
    """
    u, v, w, m = sp.symbols("u v w m")
    return integral_cmf({t-1: u, t-sp.Rational(2,3): v,
                         t-sp.Rational(1,2): w, t: -m})


def huttner_logs_seeds(ratio):
    """P,Q with log(ratio) approximated by (P W)e0/(Q W)e0.

    At source n=1 the moments on the oriented interval are C-B*log(ratio).
    Here the residue row returned as Q is B, since log(end/start)=-log(ratio).
    """
    ratio = sp.Rational(ratio)
    if ratio in (2, 3):
        P32, Q = huttner_logs_seeds(sp.Rational(3,2))
        P43, _ = huttner_logs_seeds(sp.Rational(4,3))
        # log 2 = log(3/2)+log(4/3); log 3 = 2 log(3/2)+log(4/3).
        return (ratio-1)*P32 + P43, Q
    if ratio == sp.Rational(3,2):
        left, right = sp.S.One, sp.Rational(2,3)
    elif ratio == sp.Rational(4,3):
        left, right = sp.Rational(2,3), sp.Rational(1,2)
    else:
        raise ValueError("ratio must be 3/2, 4/3, 2, or 3")
    polynomial = sp.Poly(((t-1)*(t-sp.Rational(2,3))*(t-sp.Rational(1,2)))**2,t)
    C, B = [], []
    for j in range(3):
        C.append(sum(polynomial.nth(k)*(right**(k+j-2)-left**(k+j-2))/(k+j-2)
                     for k in range(7) if k+j != 2))
        B.append(polynomial.nth(2-j))
    return Matrix([C]), Matrix([B])


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


def zudilin_catalan_seeds(variant="original"):
    """Numerator/denominator rows for two Zudilin (2002) 3F2(1) recoveries.

    Both start at source index 1 and have direction (1,1,1;2,2).
    Original: start (2,2,3/2;7/2,7/2), source math/0201024, Theorem 3.
    Modified: start (1,1,1/2;5/2,5/2), source math/0210423, Theorem 2.
    Multiplication by the source's beta prefactor gives its exact p_N,q_N.
    """
    if variant == "original":
        return (Matrix([[-sp.Rational(2925,32), sp.Rational(13275,64)]]),
                Matrix([[-sp.Rational(1575,16), sp.Rational(7425,32)]]))
    if variant == "modified":
        return (Matrix([[sp.Rational(45,4), -sp.Rational(117,8)]]),
                Matrix([[sp.Rational(27,2), -sp.Rational(63,4)]]))
    raise ValueError("variant must be original or modified")


def zudilin_zeta5():
    """Rank-3 one-axis recurrence CMF, Zudilin (2002), math/0206178, (1).

    A row (u_k,u_(k+1),u_(k+2)) advances by right multiplication.
    Start k=0, direction +1; the first column at depth N reads u_N.
    This is a one-dimensional encoding, not a multiaxis extension.
    """
    k = sp.Symbol("k")
    n = k + 2
    def a0(n):
        return 41218*n**3 - 48459*n**2 + 20010*n - 2871
    a1 = 2*(48802112*n**9 + 89030880*n**8 + 36002654*n**7
            - 24317344*n**6 - 19538418*n**5 + 1311365*n**4
            + 3790503*n**3 + 460056*n**2 - 271701*n - 60291)
    a2 = (3874492*n**8 - 2617900*n**7 - 3144314*n**6 + 2947148*n**5
          + 647130*n**4 - 1182926*n**3 + 115771*n**2 + 170716*n - 44541)
    d = (n + 1)**6 * a0(n)
    return CMF({k: Matrix([
        [0, 0, 4*(n-1)**4*(2*n-1)*(2*n-3)*a0(n+1)/d],
        [1, 0, 4*(2*n-1)*a2/d],
        [0, 1, -a1/d],
    ])})


def apery_seeds(zeta_index):
    """Source-index-2 numerator/denominator rows for the full pFq CMFs.

    zeta(2): 3F2(-N,-N,N+1;1,1;1), rank 2.
    zeta(3): 4F3(-N,-N,N+1,N+1;1,1,1;1), rank 3.
    The first-column trajectories satisfy Apery's second-order recurrence
    identically; these rows match b_2,b_3 and a_2,a_3 respectively.
    """
    if zeta_index == 2:
        return Matrix([[sp.Rational(125, 4), sp.Rational(79, 2)]]), Matrix([[19, 24]])
    if zeta_index == 3:
        return Matrix([[sp.Rational(351, 4), sp.Rational(2683, 24), 0]]), Matrix([[73, 93, 0]])
    raise ValueError("zeta_index must be 2 or 3")


def zudilin_zeta23_seeds():
    """Coefficient rows (1,zeta(1),zeta(2),zeta(3)) of F_2,theta F_2,theta^2 F_2.

    F_N=4F3(N+1,N+1,N+1,N+1;2N+2,2N+2,2N+2;1).
    Source: Zudilin (2002), math/0206178, Section 3, Theorem 4.
    """
    m = sp.Symbol("m")
    term = (sp.factorial(5)**3/sp.factorial(2)**4 * (m+1)*(m+2)
            / sp.prod((m+i)**3 for i in range(3, 6)))
    rows = [[sp.S(0)]*3 for _ in range(4)]
    for j in range(3):
        for term_j in sp.Add.make_args(sp.apart(m**j*term, m)):
            numerator, denominator = term_j.as_numer_denom()
            poly = sp.Poly(denominator, m)
            power = poly.degree()
            shift = poly.nth(power-1)/(power*poly.LC())
            coefficient = numerator/poly.LC()
            rows[0][j] -= coefficient*sp.harmonic(shift-1, power)
            rows[power][j] += coefficient
    assert rows[1] == [0]*3
    return fmpq_mat([[fmpq(int(c.p), int(c.q)) for c in row] for row in rows])


def zudilin_zeta23_forms(walk, index, seeds):
    """Return (p2,p3,q)/C_N^2 from the full 7-axis walk of length index-2.

    The paper's series are -C_N F_N and C_N(theta+N+1)F_N,
    C_N=N!^6/(2N+1)!^3. Determinants eliminate the other zeta value.
    """
    state = seeds*fmpq_mat(walk.rows, walk.cols,
                          [fmpq(int(c.p), int(c.q)) for c in walk])
    a = [state[i, 0] for i in range(4)]
    b = [(index+1)*state[i, 0]+state[i, 1] for i in range(4)]
    assert a[1] == b[1] == 0
    q = a[3]*b[2]-b[3]*a[2]
    return b[3]*a[0]-a[3]*b[0], a[2]*b[0]-b[2]*a[0], q


def zudilin_zeta23_readout(walk, index, seeds, zeta_index):
    """Reduced rational approximation to either zeta(2) or zeta(3)."""
    if zeta_index not in (2, 3):
        raise ValueError("zeta_index must be 2 or 3")
    p2, p3, q = zudilin_zeta23_forms(walk, index, seeds)
    return (p2 if zeta_index == 2 else p3)/q


def zudilin_zeta5_seeds():
    """Numerator and denominator rows in the normalization of Theorem 1."""
    return ([fmpq(0), fmpq(87, 2), fmpq(-1190161, 64)],
            [fmpq(-1), fmpq(42), fmpq(-17934)])


def zudilin_zeta5_hypergeometric_seeds():
    """Coefficient rows of (F, theta F, ..., theta^6 F) at source n=2.

    F = 8F7(8,3,3,3,3,3,3,3;6,6,6,6,6,6,6;1).
    Rows are 1, zeta(1), ..., zeta(7); the zeta(1) row vanishes.
    Each derivative is summed by rational partial fractions. All seven
    derivatives converge at this anchor (parametric excess is 13).
    """
    m = sp.Symbol("m")
    term = (sp.factorial(5)**7 / (sp.factorial(7)*sp.factorial(2)**7)
            * sp.prod(m+i for i in range(1, 8))
            / sp.prod((m+i)**7 for i in range(3, 6)))
    rows = [[sp.S(0)]*7 for _ in range(8)]
    for j in range(7):
        for term_j in sp.Add.make_args(sp.apart(m**j*term, m)):
            numerator, denominator = term_j.as_numer_denom()
            poly = sp.Poly(denominator, m)
            power = poly.degree()
            shift = poly.nth(power-1) / (power*poly.LC())
            coefficient = numerator/poly.LC()
            rows[0][j] -= coefficient*sp.harmonic(shift-1, power)
            rows[power][j] += coefficient
    assert rows[1] == [0]*7
    return fmpq_mat([[fmpq(int(c.p), int(c.q)) for c in row] for row in rows])


def zudilin_zeta5_hypergeometric_forms(walk, index, seeds):
    """Coefficient vectors of r_n/C_n and rtilde_n/C_n, source (7).

    The walk starts at n=2, so its depth is index-2. For m=k-n-1,
    r_n/C_n=(theta+3n/2+1)F_n and
    rtilde_n/C_n=-(theta+n+1)(theta+2n+1)(theta+3n/2+1)F_n.
    C_n=n!^11 (3n+1)!/(2n+1)!^7; it cancels from the final ratio.
    """
    state = seeds * fmpq_mat(walk.rows, walk.cols,
                            [fmpq(int(c.p), int(c.q)) for c in walk])
    a, b, c = fmpq(3*index+2, 2), index+1, 2*index+1
    r = [a*state[i, 0]+state[i, 1] for i in range(8)]
    rt = [-(a*b*c*state[i, 0]+(a*b+a*c+b*c)*state[i, 1]
            +(a+b+c)*state[i, 2]+state[i, 3]) for i in range(8)]
    return r, rt


def zudilin_zeta5_hypergeometric_readout(walk, index, seeds):
    """Source (15): cancel zeta(3) using the two coefficient determinants."""
    r, rt = zudilin_zeta5_hypergeometric_forms(walk, index, seeds)
    assert all(r[i] == rt[i] == 0 for i in (1, 2, 4, 6, 7))
    return (r[3]*rt[0]-rt[3]*r[0]) / (r[5]*rt[3]-rt[5]*r[3])


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
