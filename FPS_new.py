# -*- coding: utf-8 -*-
"""
fps.py — 形式的冪級数 (Formal Power Series) ライブラリ
=====================================================

使い方::

    from fps import FPS

    a = FPS([1, 2, 3])                 # mod は省略すると 998244353
    b = FPS([1, 2, 3], 998244353)      # 明示的に指定してもよい
    c = FPS([1, 2, 3], 167772161)      # 他の NTT 素数を使う
    d = FPS([1, 2, 3], -1)             # -1 : 3 素数の CRT 合成で
                                        #      -2**63 〜 2**63 程度の
                                        #      巨大整数のまま畳み込みできるモード

    print(a * b)         # 畳み込み
    print(a + b, a - b)  # 加減算
    print(a.inv(10))     # 逆元 (先頭 10 項)
    print(a.log(10), a.exp(10), a.pow(5, 10))
    print(a.diff(), a.integral())
    print(a.eval(3))
    q, r = divmod(a, b)  # 多項式除算

サポートしている mod
--------------------
* ``998244353``  (デフォルト, g=3)
* ``754974721``  (g=11)
* ``167772161``  (g=3)
* ``469762049``  (g=3)
* 上記以外でも「NTT-friendly な素数」(mod-1 が 2 を十分な回数で割り切れる素数) なら
  自動で原始根を探索してテーブルを構築するので指定可能。
* ``-1``  … 上記のうち 998244353 を除く 3 素数 (754974721 / 167772161 / 469762049)
  を使い、畳み込み結果を中国剰余定理 (Garner のアルゴリズム) で合成する。
  3 素数の積は約 5.95×10^25 (2^86 相当) あるので、真の値の絶対値がその半分未満
  (実用上は -2**63 〜 2**63 程度) に収まっていれば mod を取らずに厳密な整数として
  計算できる（atcoder library の convolution_ll と同じ考え方）。

  ``inv`` / ``log`` / ``exp`` / ``pow`` / ``//`` (除算) / ``%`` (剰余) /
  ``divmod`` / ``integral`` のような「体の割り算」を要する操作も、mod=-1 で
  そのまま使える。裏側では 3 つの NTT 素数それぞれを法として独立に同じ計算を
  行い、出てきた 3 通りの結果を CRT で 1 つの整数列に合成している。
  つまり「途中で log/exp/inv などを使うが、最終的な答えは-2**63〜2**63 に
  収まることが分かっている整数列がほしい」というユースケースにそのまま使える::

      bell = FPS([0, 1], -1).exp(30)      # Bell数を厳密な整数として計算
      inv  = FPS([1, 1], -1).inv(30)      # 1/(1+x) を厳密な整数係数で

  これが正しく機能するのは、計算中に出てくる分母 (a[0] や 1..deg の整数など)
  がどの 3 素数 (どれも ~10^8〜10^9) でも 0 にならない場合。EGF 的な組合せ計算
  (階乗・二項係数・Bell数・分割数など) では実用上まず問題にならないが、
  理論的な保証はしていない (エラーハンドリングを弱くした実装のため、割り切れて
  しまった場合は例外を出さず黙って不正な値を返す)。
"""

from typing import List, Optional, Tuple, Union

Number = int

# ---------------------------------------------------------------------------
# 1. 原始根探索 & NTT テーブル (root/iroot/rate2/rate3/imag/iimag) の汎用構築
# ---------------------------------------------------------------------------


def _primitive_root(mod: int) -> int:
    """mod (素数) の原始根を一つ返す。"""
    if mod == 2:
        return 1
    m = mod - 1
    factors = []
    d, mm = 2, m
    while d * d <= mm:
        if mm % d == 0:
            factors.append(d)
            while mm % d == 0:
                mm //= d
        d += 1
    if mm > 1:
        factors.append(mm)
    g = 2
    while True:
        if all(pow(g, m // p, mod) != 1 for p in factors):
            return g
        g += 1


def _build_ntt_tables(mod: int, g: Optional[int] = None):
    """
    与えられた NTT-friendly な素数 mod に対して
        IMAG, IIMAG, rate2, irate2, rate3, irate3, rank2
    を計算する。

    rate2 / rate3 系列は「index 0 はダミー、有効なデータは index 1 から」という
    元の (yosupo 140558 由来の) 実装の添字規約に合わせてある。
    これは 998244353 の場合、ハードコードされていた _rate2 / _rate3 の値と
    完全一致することを確認済み。
    """
    if g is None:
        g = _primitive_root(mod)

    m = mod - 1
    rank2 = 0
    while m % 2 == 0:
        m //= 2
        rank2 += 1
    if rank2 < 2:
        raise ValueError(
            f"mod={mod} は NTT に使うには不適切です (mod-1 の 2 冪部分が小さすぎます)"
        )

    root = [0] * (rank2 + 1)
    iroot = [0] * (rank2 + 1)
    root[rank2] = pow(g, (mod - 1) >> rank2, mod)
    iroot[rank2] = pow(root[rank2], mod - 2, mod)
    for i in range(rank2 - 1, -1, -1):
        root[i] = root[i + 1] * root[i + 1] % mod
        iroot[i] = iroot[i + 1] * iroot[i + 1] % mod

    IMAG = root[2]
    IIMAG = iroot[2]

    rate2 = [0] * (rank2 + 1)
    irate2 = [0] * (rank2 + 1)
    prod, iprod = 1, 1
    for i in range(0, rank2 - 1):
        rate2[i + 1] = root[i + 2] * prod % mod
        irate2[i + 1] = iroot[i + 2] * iprod % mod
        prod = prod * iroot[i + 2] % mod
        iprod = iprod * root[i + 2] % mod

    rate3 = [0] * rank2
    irate3 = [0] * rank2
    prod, iprod = 1, 1
    for i in range(0, rank2 - 2):
        rate3[i + 1] = root[i + 3] * prod % mod
        irate3[i + 1] = iroot[i + 3] * iprod % mod
        prod = prod * iroot[i + 3] % mod
        iprod = iprod * root[i + 3] % mod

    return IMAG, IIMAG, rate2, irate2, rate3, irate3, rank2


def _make_fft_ifft(mod, IMAG, IIMAG, rate2, irate2, rate3, irate3):
    """radix-4 iterative NTT (元コードそのまま。テーブルだけ差し替え可能にした)。"""

    def _fft(a):
        n = len(a)
        h = (n - 1).bit_length()
        le = 0
        for le in range(0, h - 1, 2):
            p = 1 << (h - le - 2)
            rot = 1
            for s in range(1 << le):
                rot2 = rot * rot % mod
                rot3 = rot2 * rot % mod
                offset = s << (h - le)
                for i in range(p):
                    a0 = a[i + offset]
                    a1 = a[i + offset + p] * rot
                    a2 = a[i + offset + p * 2] * rot2
                    a3 = a[i + offset + p * 3] * rot3
                    a1na3imag = (a1 - a3) % mod * IMAG
                    a[i + offset] = (a0 + a2 + a1 + a3) % mod
                    a[i + offset + p] = (a0 + a2 - a1 - a3) % mod
                    a[i + offset + p * 2] = (a0 - a2 + a1na3imag) % mod
                    a[i + offset + p * 3] = (a0 - a2 - a1na3imag) % mod
                rot = rot * rate3[(~s & -~s).bit_length()] % mod
        if h - le & 1:
            rot = 1
            for s in range(1 << (h - 1)):
                offset = s << 1
                l = a[offset]
                r = a[offset + 1] * rot
                a[offset] = (l + r) % mod
                a[offset + 1] = (l - r) % mod
                rot = rot * rate2[(~s & -~s).bit_length()] % mod

    def _ifft(a):
        n = len(a)
        h = (n - 1).bit_length()
        le = h
        for le in range(h, 1, -2):
            p = 1 << (h - le)
            irot = 1
            for s in range(1 << (le - 2)):
                irot2 = irot * irot % mod
                irot3 = irot2 * irot % mod
                offset = s << (h - le + 2)
                for i in range(p):
                    a0 = a[i + offset]
                    a1 = a[i + offset + p]
                    a2 = a[i + offset + p * 2]
                    a3 = a[i + offset + p * 3]
                    a2na3iimag = (a2 - a3) * IIMAG % mod
                    a[i + offset] = (a0 + a1 + a2 + a3) % mod
                    a[i + offset + p] = (a0 - a1 + a2na3iimag) * irot % mod
                    a[i + offset + p * 2] = (a0 + a1 - a2 - a3) * irot2 % mod
                    a[i + offset + p * 3] = (a0 - a1 - a2na3iimag) * irot3 % mod
                irot = irot * irate3[(~s & -~s).bit_length()] % mod
        if le & 1:
            p = 1 << (h - 1)
            for i in range(p):
                l = a[i]
                r = a[i + p]
                a[i] = l + r if l + r < mod else l + r - mod
                a[i + p] = l - r if l - r >= 0 else l - r + mod

    return _fft, _ifft


# ---------------------------------------------------------------------------
# 2. 1 つの素数 mod に対する演算一式をまとめたコンテキスト (テーブルはキャッシュ)
# ---------------------------------------------------------------------------


class NTTContext:
    """
    1 つの NTT-friendly prime に対する root テーブルと、それを使った
    畳み込み / 逆元 / log / exp / pow / 除算 などをまとめたクラス。

    同じ mod に対しては使い回す (テーブル構築コストを避けるため mod ごとに
    1 個だけインスタンスを作ってキャッシュする)。
    """

    _cache = {}

    def __new__(cls, mod: int, g: Optional[int] = None):
        key = mod
        if key in cls._cache:
            return cls._cache[key]
        self = object.__new__(cls)
        self.mod = mod
        (
            self.IMAG,
            self.IIMAG,
            self.rate2,
            self.irate2,
            self.rate3,
            self.irate3,
            self.rank2,
        ) = _build_ntt_tables(mod, g)
        self._fft, self._ifft = _make_fft_ifft(
            mod, self.IMAG, self.IIMAG, self.rate2, self.irate2, self.rate3, self.irate3
        )
        cls._cache[key] = self
        return self

    # --- 基本 NTT ---
    def ntt(self, a: list) -> None:
        if len(a) <= 1:
            return
        self._fft(a)

    def intt(self, a: list) -> None:
        if len(a) <= 1:
            return
        self._ifft(a)
        iv = pow(len(a), self.mod - 2, self.mod)
        for i, x in enumerate(a):
            a[i] = x * iv % self.mod

    # --- 畳み込み ---
    def multiply(self, s: list, t: list) -> list:
        mod = self.mod
        n, m = len(s), len(t)
        l = n + m - 1
        if l <= 0:
            return []
        if min(n, m) <= 60:
            a = [0] * l
            for i, x in enumerate(s):
                if x == 0:
                    continue
                for j, y in enumerate(t):
                    a[i + j] += x * y
            return [x % mod for x in a]
        z = 1 << (l - 1).bit_length()
        a = s + [0] * (z - n)
        b = t + [0] * (z - m)
        self._fft(a)
        self._fft(b)
        for i, x in enumerate(b):
            a[i] = a[i] * x % mod
        self._ifft(a)
        del a[l:]
        iz = pow(z, mod - 2, mod)
        return [x * iz % mod for x in a]

    def pow2(self, s: list) -> list:
        """自乗 (s の畳み込みを自分自身と行う)。"""
        return self.multiply(s, s)

    # --- 加減算 (体でなくても良いが、mod で reduce する) ---
    def fps_add(self, a: list, b: list) -> list:
        mod = self.mod
        if len(a) < len(b):
            res = b[:]
            for i, x in enumerate(a):
                res[i] += x
        else:
            res = a[:]
            for i, x in enumerate(b):
                res[i] += x
        return [x % mod for x in res]

    def fps_sub(self, a: list, b: list) -> list:
        mod = self.mod
        if len(a) < len(b):
            res = b[:]
            for i, x in enumerate(a):
                res[i] -= x
            res = [(-x) % mod for x in res]
        else:
            res = a[:]
            for i, x in enumerate(b):
                res[i] -= x
        return [x % mod for x in res]

    def fps_neg(self, a: list) -> list:
        mod = self.mod
        return [mod - x if x else 0 for x in a]

    def fps_mul_scalar(self, a: list, k: int) -> list:
        mod = self.mod
        k %= mod
        return [x * k % mod for x in a]

    # --- 除算 / 逆元 ---
    def fps_div(self, a: list, b: list) -> list:
        mod = self.mod
        if len(a) < len(b):
            return []
        n = len(a) - len(b) + 1
        if len(b) > 64:
            return self.multiply(a[::-1][:n], self.fps_inv(b[::-1], n))[:n][::-1]
        f, g = a[:], b[:]
        cnt = 0
        while g and not g[-1]:
            g.pop()
            cnt += 1
        coef = pow(g[-1], mod - 2, mod)
        g = self.fps_mul_scalar(g, coef)
        deg = len(f) - len(g) + 1
        gs = len(g)
        quo = [0] * deg
        for i in range(deg)[::-1]:
            quo[i] = x = f[i + gs - 1] % mod
            for j, y in enumerate(g):
                f[i + j] -= x * y
        return self.fps_mul_scalar(quo, coef) + [0] * cnt

    def fps_mod(self, a: list, b: list) -> list:
        res = self.fps_sub(a, self.multiply(self.fps_div(a, b), b))
        while res and not res[-1]:
            res.pop()
        return res

    def fps_divmod(self, a: list, b: list) -> Tuple[list, list]:
        q = self.fps_div(a, b)
        r = self.fps_sub(a, self.multiply(q, b))
        while r and not r[-1]:
            r.pop()
        return q, r

    def fps_inv(self, a: list, deg: int = -1) -> list:
        mod = self.mod
        if deg == -1:
            deg = len(a)
        res = [0] * deg
        res[0] = pow(a[0], mod - 2, mod)
        d = 1
        while d < deg:
            f = [0] * (d << 1)
            tmp = min(len(a), d << 1)
            f[:tmp] = a[:tmp]
            g = [0] * (d << 1)
            g[:d] = res[:d]
            self.ntt(f)
            self.ntt(g)
            for i, x in enumerate(g):
                f[i] = f[i] * x % mod
            self.intt(f)
            f[:d] = [0] * d
            self.ntt(f)
            for i, x in enumerate(g):
                f[i] = f[i] * x % mod
            self.intt(f)
            for j in range(d, min(d << 1, deg)):
                res[j] = mod - f[j] if f[j] else 0
            d <<= 1
        return res

    # --- log / exp / pow ---
    def fps_diff(self, a: list) -> list:
        mod = self.mod
        return [i * x % mod for i, x in enumerate(a) if i]

    def fps_integral(self, a: list) -> list:
        mod = self.mod
        n = len(a)
        res = [0] * (n + 1)
        if n:
            res[1] = 1
        for i in range(2, n + 1):
            j, k = divmod(mod, i)
            res[i] = (-res[k] * j) % mod
        for i, x in enumerate(a):
            res[i + 1] = res[i + 1] * x % mod
        return res

    def fps_log(self, a: list, deg: int = -1) -> list:
        if deg == -1:
            deg = len(a)
        return self.fps_integral(self.multiply(self.fps_diff(a), self.fps_inv(a, deg))[: deg - 1])

    def fps_exp(self, a: list, deg: int = -1) -> list:
        mod = self.mod
        if deg == -1:
            deg = len(a)
        inv = [0, 1]

        def inplace_integral(F):
            n = len(F)
            while len(inv) <= n:
                j, k = divmod(mod, len(inv))
                inv.append((-inv[k] * j) % mod)
            return [0] + [x * inv[i + 1] % mod for i, x in enumerate(F)]

        def inplace_diff(F):
            return [x * i % mod for i, x in enumerate(F) if i]

        b = [1, (a[1] if 1 < len(a) else 0)]
        c = [1]
        z1: list = []
        z2 = [1, 1]
        m = 2
        while m < deg:
            y = b + [0] * m
            self.ntt(y)
            z1 = z2
            z = [y[i] * p % mod for i, p in enumerate(z1)]
            self.intt(z)
            z[: m >> 1] = [0] * (m >> 1)
            self.ntt(z)
            for i, p in enumerate(z1):
                z[i] = z[i] * (-p) % mod
            self.intt(z)
            c[m >> 1 :] = z[m >> 1 :]
            z2 = c + [0] * m
            self.ntt(z2)
            tmp = min(len(a), m)
            x = a[:tmp] + [0] * (m - tmp)
            x = inplace_diff(x)
            x.append(0)
            self.ntt(x)
            for i, p in enumerate(x):
                x[i] = y[i] * p % mod
            self.intt(x)
            for i, p in enumerate(b):
                if not i:
                    continue
                x[i - 1] -= p * i % mod
            x += [0] * m
            for i in range(m - 1):
                x[m + i], x[i] = x[i], 0
            self.ntt(x)
            for i, p in enumerate(z2):
                x[i] = x[i] * p % mod
            self.intt(x)
            x.pop()
            x = inplace_integral(x)
            x[:m] = [0] * m
            for i in range(m, min(len(a), m << 1)):
                x[i] += a[i]
            self.ntt(x)
            for i, p in enumerate(y):
                x[i] = x[i] * p % mod
            self.intt(x)
            b[m:] = x[m:]
            m <<= 1
        return b[:deg]

    def fps_pow(self, a: list, k: int, deg: int = -1) -> list:
        mod = self.mod
        n = len(a)
        if deg == -1:
            deg = n
        if k == 0:
            if not deg:
                return []
            ret = [0] * deg
            ret[0] = 1
            return ret
        for i, x in enumerate(a):
            if x:
                rev = pow(x, mod - 2, mod)
                shifted = self.fps_mul_scalar(a, rev)[i:]
                lg = self.fps_log(shifted, deg)
                lgk = self.fps_mul_scalar(lg, k)
                ex = self.fps_exp(lgk, deg)
                ret = self.fps_mul_scalar(ex, pow(x, k, mod))
                ret[:0] = [0] * (i * k)
                if len(ret) < deg:
                    ret += [0] * (deg - len(ret))
                    return ret
                return ret[:deg]
            if (i + 1) * k >= deg:
                break
        return [0] * deg

    def fps_eval(self, a: list, x: int) -> int:
        mod = self.mod
        r, w = 0, 1
        for v in a:
            r += w * v % mod
            w = w * x % mod
        return r % mod


# ---------------------------------------------------------------------------
# 3. mod=-1 : 3 素数 CRT による「巨大整数のまま畳み込む」モード
# ---------------------------------------------------------------------------

# atcoder library の convolution_ll と同じ組み合わせ。
# 積は約 5.95e25 (≈2^86) あるので、真の値が -2**63〜2**63 程度に収まっていれば
# mod を取らずに厳密な畳み込みができる。
_CRT_PRIMES: Tuple[Tuple[int, int], ...] = (
    (754974721, 11),
    (167772161, 3),
    (469762049, 3),
)

# NOTE(速度対策): 以前はこの3素数分の NTT テーブルを import 時に無条件で構築して
# いたが、mod=-1 (CRT 巨大整数モード) を一度も使わない大多数の呼び出し
# (デフォルトの 998244353 だけ使うケースなど) にとって完全に無駄なコストだった
# ので、実際に mod=-1 の FPS が作られた最初のタイミングまで遅延させる。
_CRT_STATE = None  # (ctxs, P0, P1, P2, M0M1, INV_P0_MOD_P1, INV_M0M1_MOD_P2, MODULUS)


def _get_crt_state():
    global _CRT_STATE
    if _CRT_STATE is None:
        ctxs = [NTTContext(p, g) for p, g in _CRT_PRIMES]
        P0, P1, P2 = (p for p, _ in _CRT_PRIMES)
        M0M1 = P0 * P1
        inv_p0_mod_p1 = pow(P0, P1 - 2, P1)
        inv_m0m1_mod_p2 = pow(M0M1 % P2, P2 - 2, P2)
        modulus = P0 * P1 * P2
        _CRT_STATE = (ctxs, P0, P1, P2, M0M1, inv_p0_mod_p1, inv_m0m1_mod_p2, modulus)
    return _CRT_STATE


def _crt_modulus() -> int:
    """mod=-1 モードで厳密に復元できる範囲 (この値の約半分未満) を返す。"""
    return _get_crt_state()[7]


def _crt_combine(r0: list, r1: list, r2: list) -> list:
    """
    3 つの素数それぞれを法とした結果を Garner のアルゴリズムで 1 つの厳密な
    整数列に合成する。3 素数の積は ≈2^86 あるので、真の値がその半分未満
    (実用上 -2**63〜2**63 程度なら十分収まる) であれば厳密に復元できる。

    r0/r1/r2 の長さが (末尾 0 の trim 具合の違いなどで) 揃っていないことも
    あるので、その場合は 0 埋めしてから合成する。
    """
    n = max(len(r0), len(r1), len(r2))
    if len(r0) < n:
        r0 = r0 + [0] * (n - len(r0))
    if len(r1) < n:
        r1 = r1 + [0] * (n - len(r1))
    if len(r2) < n:
        r2 = r2 + [0] * (n - len(r2))
    _, P0, P1, P2, M0M1, inv_p0_mod_p1, inv_m0m1_mod_p2, modulus = _get_crt_state()
    half = modulus >> 1
    out = [0] * n
    for idx in range(n):
        x0, x1, x2 = r0[idx], r1[idx], r2[idx]
        t1 = (x1 - x0) * inv_p0_mod_p1 % P1
        t2 = (x2 - x0 - t1 * P0) * inv_m0m1_mod_p2 % P2
        x = x0 + t1 * P0 + t2 * M0M1
        if x > half:
            x -= modulus
        out[idx] = x
    return out


def _multiply_bigint(a: List[int], b: List[int]) -> List[int]:
    """任意精度整数のリスト同士の畳み込みを、3 素数 NTT + CRT で厳密に計算する。"""
    (ctx0, ctx1, ctx2), P0, P1, P2, *_ = _get_crt_state()
    r0 = ctx0.multiply([x % P0 for x in a], [x % P0 for x in b])
    r1 = ctx1.multiply([x % P1 for x in a], [x % P1 for x in b])
    r2 = ctx2.multiply([x % P2 for x in a], [x % P2 for x in b])
    return _crt_combine(r0, r1, r2)


def _crt_field_op(opname: str, a: List[int], *args) -> list:
    """
    inv/log/exp/pow/integral のような「体の割り算」を要する操作を、3 つの
    NTT 素数それぞれの下で独立に (mod p_i として) 実行し、結果を CRT で
    合成して 1 つの厳密な整数列に戻す。

    真に成り立つのは、計算の途中で出てくる分母 (逆元を取る対象) がどの
    p_i でも 0 にならない場合 — 具体的には a[0] (inv/log/pow の場合) や
    1..deg の整数 (integral の場合) が 3 つの素数 (どれも ~10^8〜10^9 の
    大きさ) では通常まず割り切れないので、EGF 的な組合せ計算 (n!, 二項係数,
    Bell数, 分割数 の指数型母関数など) では実用上問題なく機能する。
    ここではその割り切れチェックはしていない (エラーハンドリングは弱くした
    実装なので、割り切れてしまった場合は黙って不正な結果を返す)。
    """
    (ctx0, ctx1, ctx2), P0, P1, P2, *_ = _get_crt_state()
    r0 = getattr(ctx0, opname)([x % P0 for x in a], *args)
    r1 = getattr(ctx1, opname)([x % P1 for x in a], *args)
    r2 = getattr(ctx2, opname)([x % P2 for x in a], *args)
    return _crt_combine(r0, r1, r2)


def _crt_field_op2(opname: str, a: List[int], b: List[int], *args) -> list:
    """div / mod のように 2 引数かつ単一のリストを返す field 演算版の _crt_field_op。"""
    (ctx0, ctx1, ctx2), P0, P1, P2, *_ = _get_crt_state()
    r0 = getattr(ctx0, opname)([x % P0 for x in a], [x % P0 for x in b], *args)
    r1 = getattr(ctx1, opname)([x % P1 for x in a], [x % P1 for x in b], *args)
    r2 = getattr(ctx2, opname)([x % P2 for x in a], [x % P2 for x in b], *args)
    return _crt_combine(r0, r1, r2)


def _crt_divmod(a: List[int], b: List[int]):
    """fps_divmod の mod=-1 版。商・余りをそれぞれ独立に CRT 合成する。"""
    (ctx0, ctx1, ctx2), P0, P1, P2, *_ = _get_crt_state()
    q0, r0 = ctx0.fps_divmod([x % P0 for x in a], [x % P0 for x in b])
    q1, r1 = ctx1.fps_divmod([x % P1 for x in a], [x % P1 for x in b])
    q2, r2 = ctx2.fps_divmod([x % P2 for x in a], [x % P2 for x in b])
    return _crt_combine(q0, q1, q2), _crt_combine(r0, r1, r2)


# ---------------------------------------------------------------------------
# 4. FPS クラス本体
# ---------------------------------------------------------------------------

BIG = -1  # FPS(coeffs, -1) の -1 に名前を付けたもの (FPS(coeffs, FPS.BIG) とも書ける)


class FPS:
    """
    形式的冪級数 (係数リスト + mod) を表すクラス。

        a = FPS([1, 2, 3])                 # mod=998244353 (デフォルト)
        b = FPS([1, 2, 3], 167772161)      # 他の NTT 素数
        c = FPS([1, 2, 3], -1)             # 3 素数 CRT の「巨大整数」モード
    """

    __slots__ = ("a", "mod", "_ctx")

    def __init__(self, coeffs: Union[List[int], "FPS"], mod: int = 998244353):
        if isinstance(coeffs, FPS):
            coeffs = coeffs.a
        self.mod = mod
        if mod == -1:
            self._ctx = None
            self.a = [x for x in coeffs]
        else:
            self._ctx = NTTContext(mod)
            m = self._ctx.mod
            self.a = [x % m for x in coeffs]

    @classmethod
    def _wrap(cls, a: list, mod: int, ctx: Optional["NTTContext"] = None) -> "FPS":
        """
        速度対策用の「信頼済み」内部コンストラクタ。

        通常の __init__ は「ユーザーから渡された生のリストを mod で正規化する」
        ために毎回コピー + 全要素の % 演算 (+以前は int() 呼び出しも) を行うが、
        ctx.multiply / ctx.fps_add / ctx.fps_pow などライブラリ内部の関数が返す
        リストは既に mod で reduce 済みであることが保証されているので、
        その再検証パスは完全に無駄な O(n) の二度手間だった。
        FPS 同士の演算結果をラップするときは、検証をスキップしてこちらを使う。
        (エラーチェックは弱くなるが、その代わり全ての演算 (+, -, *, inv, log,
        exp, pow, diff, integral, ...) で余分なコピー & mod 演算が消える。)
        """
        self = object.__new__(cls)
        self.a = a
        self.mod = mod
        self._ctx = ctx
        return self

    # ---- ヘルパ ----
    def _same_mode(self, other: "FPS") -> "FPS":
        if not isinstance(other, FPS):
            raise TypeError(f"FPS 同士でないと演算できません: {type(other)}")
        if other.mod != self.mod:
            raise ValueError(f"mod が異なる FPS 同士は演算できません: {self.mod} vs {other.mod}")
        return other

    def shrink(self) -> "FPS":
        while self.a and not self.a[-1]:
            self.a.pop()
        return self

    def resize(self, n: int) -> "FPS":
        if len(self.a) < n:
            self.a += [0] * (n - len(self.a))
        else:
            del self.a[n:]
        return self

    # ---- dunder: コンテナ的な振る舞い ----
    def __len__(self):
        return len(self.a)

    def __iter__(self):
        return iter(self.a)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return FPS._wrap(self.a[i], self.mod, self._ctx)
        return self.a[i] if 0 <= i < len(self.a) else 0

    def __setitem__(self, i, v):
        if i >= len(self.a):
            self.a += [0] * (i + 1 - len(self.a))
        self.a[i] = v if self.mod == -1 else (v % self.mod)

    def __repr__(self):
        return f"FPS({self.a}, mod={self.mod})"

    def __eq__(self, other):
        if not isinstance(other, FPS) or other.mod != self.mod:
            return NotImplemented
        n = max(len(self.a), len(other.a))
        A = self.a + [0] * (n - len(self.a))
        B = other.a + [0] * (n - len(other.a))
        return A == B

    # ---- 加減算 ----
    def __add__(self, other):
        other = self._same_mode(other)
        if self.mod == -1:
            n = max(len(self.a), len(other.a))
            A = self.a + [0] * (n - len(self.a))
            B = other.a + [0] * (n - len(other.a))
            return FPS._wrap([x + y for x, y in zip(A, B)], -1)
        return FPS._wrap(self._ctx.fps_add(self.a, other.a), self.mod, self._ctx)

    def __sub__(self, other):
        other = self._same_mode(other)
        if self.mod == -1:
            n = max(len(self.a), len(other.a))
            A = self.a + [0] * (n - len(self.a))
            B = other.a + [0] * (n - len(other.a))
            return FPS._wrap([x - y for x, y in zip(A, B)], -1)
        return FPS._wrap(self._ctx.fps_sub(self.a, other.a), self.mod, self._ctx)

    def __neg__(self):
        if self.mod == -1:
            return FPS._wrap([-x for x in self.a], -1)
        return FPS._wrap(self._ctx.fps_neg(self.a), self.mod, self._ctx)

    # ---- 乗算 (畳み込み / スカラー倍) ----
    def __mul__(self, other):
        if isinstance(other, int):
            if self.mod == -1:
                return FPS._wrap([x * other for x in self.a], -1)
            return FPS._wrap(self._ctx.fps_mul_scalar(self.a, other), self.mod, self._ctx)
        other = self._same_mode(other)
        if self.mod == -1:
            return FPS._wrap(_multiply_bigint(self.a, other.a), -1)
        return FPS._wrap(self._ctx.multiply(self.a, other.a), self.mod, self._ctx)

    __rmul__ = __mul__

    def square(self) -> "FPS":
        """自乗 (a*a を 1 回の畳み込みで)。"""
        if self.mod == -1:
            return FPS._wrap(_multiply_bigint(self.a, self.a), -1)
        return FPS._wrap(self._ctx.pow2(self.a), self.mod, self._ctx)

    # ---- 除算 (多項式除算) ----
    def __floordiv__(self, other):
        other = self._same_mode(other)
        if self.mod == -1:
            return FPS._wrap(_crt_field_op2("fps_div", self.a, other.a), -1)
        return FPS._wrap(self._ctx.fps_div(self.a, other.a), self.mod, self._ctx)

    def __mod__(self, other):
        other = self._same_mode(other)
        if self.mod == -1:
            return FPS._wrap(_crt_field_op2("fps_mod", self.a, other.a), -1)
        return FPS._wrap(self._ctx.fps_mod(self.a, other.a), self.mod, self._ctx)

    def __divmod__(self, other):
        other = self._same_mode(other)
        if self.mod == -1:
            q, r = _crt_divmod(self.a, other.a)
            return FPS._wrap(q, -1), FPS._wrap(r, -1)
        q, r = self._ctx.fps_divmod(self.a, other.a)
        return FPS._wrap(q, self.mod, self._ctx), FPS._wrap(r, self.mod, self._ctx)

    # ---- 逆元 / log / exp / pow / 微積分 / 評価 ----
    def inv(self, deg: int = -1) -> "FPS":
        if self.mod == -1:
            return FPS._wrap(_crt_field_op("fps_inv", self.a, deg), -1)
        return FPS._wrap(self._ctx.fps_inv(self.a, deg), self.mod, self._ctx)

    def log(self, deg: int = -1) -> "FPS":
        if self.mod == -1:
            return FPS._wrap(_crt_field_op("fps_log", self.a, deg), -1)
        return FPS._wrap(self._ctx.fps_log(self.a, deg), self.mod, self._ctx)

    def exp(self, deg: int = -1) -> "FPS":
        if self.mod == -1:
            return FPS._wrap(_crt_field_op("fps_exp", self.a, deg), -1)
        return FPS._wrap(self._ctx.fps_exp(self.a, deg), self.mod, self._ctx)

    def pow(self, k: int, deg: int = -1) -> "FPS":
        if self.mod == -1:
            return FPS._wrap(_crt_field_op("fps_pow", self.a, k, deg), -1)
        return FPS._wrap(self._ctx.fps_pow(self.a, k, deg), self.mod, self._ctx)

    __pow__ = pow

    def diff(self) -> "FPS":
        # 微分は割り算を要らないので mod=-1 でも常に厳密に計算できる
        if self.mod == -1:
            return FPS._wrap([i * x for i, x in enumerate(self.a) if i], -1)
        return FPS._wrap(self._ctx.fps_diff(self.a), self.mod, self._ctx)

    def integral(self) -> "FPS":
        if self.mod == -1:
            return FPS._wrap(_crt_field_op("fps_integral", self.a), -1)
        return FPS._wrap(self._ctx.fps_integral(self.a), self.mod, self._ctx)

    def eval(self, x: int) -> int:
        if self.mod == -1:
            r, w = 0, 1
            for v in self.a:
                r += w * v
                w *= x
            return r
        return self._ctx.fps_eval(self.a, x)


def __getattr__(name):
    # CRT_MODULUS は mod=-1 系のテーブルが要る時だけ計算する (PEP 562)。
    if name == "CRT_MODULUS":
        return _crt_modulus()
    raise AttributeError(name)


__all__ = ["FPS", "NTTContext", "CRT_MODULUS", "BIG"]