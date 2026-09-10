mod=998244353
def multiply_naive(a, b):
    n = len(a)
    m = len(b)
    out_len = max(n, m)
    res = [0] * out_len
    for i in range(n):
        ai = a[i] % mod
        if ai == 0:
            continue
        for j in range(m):
            k = i + j
            if k >= out_len:
                continue
            res[k] = (res[k] + ai * b[j]) % mod
    return res
def multiply_sparse(a, b):
    n = len(a)
    m = len(b)
    out_len = max(n, m)

    nz_a = [(i, a[i] % mod) for i in range(n) if a[i] % mod != 0]
    nz_b = [(j, b[j] % mod) for j in range(m) if b[j] % mod != 0]

    res = [0] * out_len
    for i, ai in nz_a:
        for j, bj in nz_b:
            k = i + j
            if k >= out_len:
                continue
            res[k] = (res[k] + ai * bj) % mod
    return res
def fps_div_naive(a, b, deg=-1):
    if deg == -1:
        deg = max(len(a), len(b))
    if not b or b[0] % mod == 0:
        raise ZeroDivisionError("b[0] == 0 mod mod")

    inv_b0 = pow(b[0] % mod, mod - 2, mod)
    q = [0] * deg

    for n in range(deg):
        s = a[n] % mod if n < len(a) else 0
        upper = min(n, len(b) - 1)
        for i in range(1, upper + 1):
            s = (s - b[i] * q[n - i]) % mod
        q[n] = s * inv_b0 % mod

    return q
def fps_div_sparse(a, b, deg=-1):
    if deg == -1:
        deg = max(len(a), len(b))
    if not b or b[0] % mod == 0:
        raise ZeroDivisionError("b[0] == 0 mod mod")

    inv_b0 = pow(b[0] % mod, mod - 2, mod)

    b_nz = {i: bi % mod for i, bi in enumerate(b) if bi % mod != 0}

    q = [0] * deg
    for n in range(deg):
        s = a[n] % mod if n < len(a) else 0
        for i, bi in b_nz.items():
            if i == 0 or i > n:
                continue
            s = (s - bi * q[n - i]) % mod
        q[n] = s * inv_b0 % mod

    return q
#↑library be Chat GPT
import math
MOD=mod
FPS_W=[]
def setw(k):
    global FPS_W
    k -= 1
    if len(FPS_W) >= (1 << k):
        return
    arg = math.pi / (1 << k)
    FPS_W = [0.0] * (1 << (k + 1))
    btr = [0] * (1 << k)
    for i in range(1 << k):
        j = (btr[i >> 1] >> 1) + ((i & 1) << (k - 1))
        btr[i] = j
        FPS_W[i * 2], FPS_W[i * 2 + 1] = math.cos(j * arg), math.sin(j * arg)

def _fft(a):
    def mul(ar, ai, br, bi):
        return ar * br - ai * bi, ar * bi + ai * br
    k=len(a).bit_length()+1
    setw(k)
    if k & 1:
        v = 1 << k
        for j in range(0, v):
            a[j], a[j + v] = a[j] + a[j + v], a[j] - a[j + v]
    u, v = 1 << (k & 1), 1 << (k - 2 - (k & 1))
    while v:
        for jh in range(u):
            j0, j1 = jh * v * 8, jh * v * 8 + 2 * v
            j2, j3 = jh * v * 8 + 4 * v, jh * v * 8 + 6 * v
            w1r, w1i = FPS_W[jh << 1], FPS_W[(jh << 1) + 1]
            w2r, w2i = FPS_W[jh << 2], FPS_W[(jh << 2) + 1]
            w3r, w3i = mul(w1r, w1i, w2r, w2i)
            for j in range(0, 2 * v, 2):
                t0r, t0i = a[j0 + j], a[j0 + j + 1]
                t1r, t1i = mul(a[j1 + j], a[j1 + j + 1], w2r, w2i)
                t2r, t2i = mul(a[j2 + j], a[j2 + j + 1], w1r, w1i)
                t3r, t3i = mul(a[j3 + j], a[j3 + j + 1], w3r, w3i)
                a[j0 + j + 0] = (t0r + t2r) + (t1r + t3r)
                a[j0 + j + 1] = (t0i + t2i) + (t1i + t3i)
                a[j1 + j + 0] = (t0r + t2r) - (t1r + t3r)
                a[j1 + j + 1] = (t0i + t2i) - (t1i + t3i)
                a[j2 + j + 0] = (t0r - t2r) + (t3i - t1i)
                a[j2 + j + 1] = (t0i - t2i) + (t1r - t3r)
                a[j3 + j + 0] = (t0r - t2r) - (t3i - t1i)
                a[j3 + j + 1] = (t0i - t2i) - (t1r - t3r)
        u, v = u << 2, v >> 2

def _ifft(a):
    def mul(ar, ai, br, bi):
        return ar * br - ai * bi, ar * bi + ai * br
    k=len(a).bit_length()+1
    setw(k)
    u, v = 1 << (k - 2), 1
    while u:
        for jh in range(u):
            j0, j1 = jh * v * 8, jh * v * 8 + 2 * v
            j2, j3 = jh * v * 8 + 4 * v, jh * v * 8 + 6 * v
            w1r, w1i = FPS_W[jh << 1], -FPS_W[(jh << 1) + 1]
            w2r, w2i = FPS_W[jh << 2], -FPS_W[(jh << 2) + 1]
            w3r, w3i = mul(w1r, w1i, w2r, w2i)
            for j in range(0, 2 * v, 2):
                t0r, t0i = a[j0 + j], a[j0 + j + 1]
                t1r, t1i = a[j1 + j], a[j1 + j + 1]
                t2r, t2i = a[j2 + j], a[j2 + j + 1]
                t3r, t3i = a[j3 + j], a[j3 + j + 1]
                a[j0 + j] = (t0r + t1r) + (t2r + t3r)
                a[j0 + j + 1] = (t0i + t1i) + (t2i + t3i)
                a[j1 + j], a[j1 + j + 1] = mul(
                    (t0r - t1r) + (t2i - t3i), (t0i - t1i) + (t3r - t2r), w2r, w2i
                )
                a[j2 + j], a[j2 + j + 1] = mul(
                    (t0r + t1r) - (t2r + t3r), (t0i + t1i) - (t2i + t3i), w1r, w1i
                )
                a[j3 + j], a[j3 + j + 1] = mul(
                    (t0r - t1r) - (t2i - t3i), (t0i - t1i) - (t3r - t2r), w3r, w3i
                )
        u, v = u >> 2, v << 2
    if k & 1:
        u = 1 << k
        for j in range(u):
            a[j], a[j + u] = a[j] + a[j + u], a[j] - a[j + u]
def ntt(a) -> None:
    if len(a) <= 1: return
    _fft(a)

def intt(a) -> None:
    if len(a) <= 1: return
    _ifft(a)
    iv = pow(len(a), MOD - 2, MOD)
    for i, x in enumerate(a): a[i] = x * iv % MOD

def multiply(s: list, t: list) -> list:
    n, m = len(s), len(t)
    l = n + m - 1
    if min(n, m) <= 60:
        a = [0] * l
        for i, x in enumerate(s):
            for j, y in enumerate(t):
                a[i + j] += x * y
        return [x % MOD for x in a]
    z = 1 << (l - 1).bit_length()
    a = s + [0] * (z - n)
    b = t + [0] * (z - m)
    _fft(a)
    _fft(b)
    for i, x in enumerate(b): a[i] = a[i] * x % MOD
    _ifft(a)
    a[l:] = []
    iz = pow(z, MOD - 2, MOD)
    return [x * iz % MOD for x in a]

def pow2(s: list) -> list:
    n = len(s)
    l = (n << 1) - 1
    if n <= 60:
        a = [0] * l
        for i, x in enumerate(s):
            for j, y in enumerate(s):
                a[i + j] += x * y
        return [x % MOD for x in a]
    z = 1 << (l - 1).bit_length()
    a = s + [0] * (z - n)
    _fft(a)
    for i, x in enumerate(a): a[i] = x * x % MOD
    _ifft(a)
    a[l:] = []
    iz = pow(z, MOD - 2, MOD)
    return [x * iz % MOD for x in a]

def ntt_doubling(a: list) -> None:
    M = len(a)
    b = a[:]
    intt(b)
    r = 1
    zeta = pow(3, (MOD - 1) // (M << 1), MOD)
    for i, x in enumerate(b):
        b[i] = x * r % MOD
        r = r * zeta % MOD
    ntt(b)
    a += b

# https://nyaannyaan.github.io/library/fps/formal-power-series.hpp
def shrink(a: list) -> None:
    while a and not a[-1]: a.pop()

def fps_add(a: list, b: list) -> list:
    if len(a) < len(b):
        res = b[::]
        for i, x in enumerate(a): res[i] += x
    else:
        res = a[::]
        for i, x in enumerate(b): res[i] += x
    return [x % MOD for x in res]

def fps_add_scalar(a: list, k: int) -> list:
    res = a[:]
    res[0] = (res[0] + k) % MOD
    return res

def fps_sub(a: list, b: list) -> list:
    if len(a) < len(b):
        res = b[::]
        for i, x in enumerate(a): res[i] -= x
        res = fps_neg(res)
    else:
        res = a[::]
        for i, x in enumerate(b): res[i] -= x
    return [x % MOD for x in res]

def fps_sub_scalar(a: list, k: int) -> list:
    return fps_add_scalar(a, -k)

def fps_neg(a: list) -> list:
    return [MOD - x if x else 0 for x in a]

def fps_mul_scalar(a: list, k: int) -> list:
    return [x * k % MOD for x in a]

def fps_matmul(a: list, b: list) -> list:
    'not verified'
    return [x * b[i] % MOD for i, x in enumerate(a)]

def fps_div(a: list, b: list) -> list:
    if len(a) < len(b): return []
    n = len(a) - len(b) + 1
    cnt = 0
    if len(b) > 64:
        return multiply(a[::-1][:n], fps_inv(b[::-1], n))[:n][::-1]
    f, g = a[::], b[::]
    while g and not g[-1]:
        g.pop()
        cnt += 1
    coef = pow(g[-1], MOD - 2, MOD)
    g = fps_mul_scalar(g, coef)
    deg = len(f) - len(g) + 1
    gs = len(g)
    quo = [0] * deg
    for i in range(deg)[::-1]:
        quo[i] = x = f[i + gs - 1] % MOD
        for j, y in enumerate(g):
            f[i + j] -= x * y
    return fps_mul_scalar(quo, coef) + [0] * cnt

def fps_mod(a: list, b: list) -> list:
    res = fps_sub(a, multiply(fps_div(a, b),  b))
    while res and not res[-1]: res.pop()
    return res

def fps_divmod(a: list, b: list):
    q = fps_div(a, b)
    r = fps_sub(a, multiply(q, b))
    while r and not r[-1]: r.pop()
    return q, r

def fps_eval(a: list, x: int) -> int:
    r = 0; w = 1
    for v in a:
        r += w * v % MOD
        w = w * x % MOD
    return r % MOD

def fps_inv(a: list, deg: int=-1) -> list:
    # assert(self[0] != 0)
    if deg == -1: deg = len(a)
    res = [0] * deg
    res[0] = pow(a[0], MOD - 2, MOD)
    d = 1
    while d < deg:
        f = [0] * (d << 1)
        tmp = min(len(a), d << 1)
        f[:tmp] = a[:tmp]
        g = [0] * (d << 1)
        g[:d] = res[:d]
        ntt(f)
        ntt(g)
        for i, x in enumerate(g): f[i] = f[i] * x % MOD
        intt(f)
        f[:d] = [0] * d
        ntt(f)
        for i, x in enumerate(g): f[i] = f[i] * x % MOD
        intt(f)
        for j in range(d, min(d << 1, deg)):
            if f[j]: res[j] = MOD - f[j]
            else: res[j] = 0
        d <<= 1
    return res

def fps_pow(a: list, k: int, deg=-1) -> list:
    n = len(a)
    if deg == -1: deg = n
    if k == 0:
        if not deg: return []
        ret = [0] * deg
        ret[0] = 1
        return ret
    for i, x in enumerate(a):
        if x:
            rev = pow(x, MOD - 2, MOD)
            ret = fps_mul_scalar(fps_exp(fps_mul_scalar(fps_log(fps_mul_scalar(a, rev)[i:], deg),  k), deg), pow(x, k, MOD))
            ret[:0] = [0] * (i * k)
            if len(ret) < deg:
                ret[len(ret):] = [0] * (deg - len(ret))
                return ret
            return ret[:deg]
        if (i + 1) * k >= deg: break
    return [0] * deg

def fps_exp(a: list, deg=-1) -> list:
    # assert(not self or self[0] == 0)
    if deg == -1: deg = len(a)
    inv = [0, 1]

    def inplace_integral(F: list) -> list:
        n = len(F)
        while len(inv) <= n:
            j, k = divmod(MOD, len(inv))
            inv.append((-inv[k] * j) % MOD)
        return [0] + [x * inv[i + 1] % MOD for i, x in enumerate(F)]

    def inplace_diff(F: list) -> list:
        return [x * i % MOD for i, x in enumerate(F) if i]

    b = [1, (a[1] if 1 < len(a) else 0)]
    c = [1]
    z1 = []
    z2 = [1, 1]
    m = 2
    while m < deg:
        y = b + [0] * m
        ntt(y)
        z1 = z2
        z = [y[i] * p % MOD for i, p in enumerate(z1)]
        intt(z)
        z[:m >> 1] = [0] * (m >> 1)
        ntt(z)
        for i, p in enumerate(z1): z[i] = z[i] * (-p) % MOD
        intt(z)
        c[m >> 1:] = z[m >> 1:]
        z2 = c + [0] * m
        ntt(z2)
        tmp = min(len(a), m)
        x = a[:tmp] + [0] * (m - tmp)
        x = inplace_diff(x)
        x.append(0)
        ntt(x)
        for i, p in enumerate(x): x[i] = y[i] * p % MOD
        intt(x)
        for i, p in enumerate(b):
            if not i: continue
            x[i - 1] -= p * i % MOD
        x += [0] * m
        for i in range(m - 1): x[m + i], x[i] = x[i], 0
        ntt(x)
        for i, p in enumerate(z2): x[i] = x[i] * p % MOD
        intt(x)
        x.pop()
        x = inplace_integral(x)
        x[:m] = [0] * m
        for i in range(m, min(len(a), m << 1)): x[i] += a[i]
        ntt(x)
        for i, p in enumerate(y): x[i] = x[i] * p % MOD
        intt(x)
        b[m:] = x[m:]
        m <<= 1
    return b[:deg]

def fps_log(a: list, deg=-1) -> list:
    # assert(a[0] == 1)
    if deg == -1: deg = len(a)
    return fps_integral(multiply(fps_diff(a), fps_inv(a, deg))[:deg - 1])

def fps_integral(a: list) -> list:
    n = len(a)
    res = [0] * (n + 1)
    if n: res[1] = 1
    for i in range(2, n + 1):
        j, k = divmod(MOD, i)
        res[i] = (-res[k] * j) % MOD
    for i, x in enumerate(a): res[i + 1] = res[i + 1] * x % MOD
    return res

def fps_diff(a: list) -> list:
    return [i * x % MOD for i, x in enumerate(a) if i]