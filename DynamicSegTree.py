#↓library by Gemini
import typing

class DynamicSegTree:
    def __init__(self,
                 op,
                 e,
                 max_bit: int = 62) -> None:
        self._op = op
        self._e = e
        self._n = 2**max_bit
        
        # n 以上の最小の 2 冪を求める (ビット演算用)
        self._log = max_bit
        self._size = 1 << self._log
        
        # リストによる管理 (0番は単位元を返すダミー/Noneの代わり)
        # v[0] = e, l[0] = 0, r[0] = 0 としておくことで、
        # 存在しないノードへのアクセスを単純化できる
        self._v = [e]
        self._l = [0]
        self._r = [0]
        self._root = self._new_node()

    def _new_node(self) -> int:
        idx = len(self._v)
        self._v.append(self._e)
        self._l.append(0)
        self._r.append(0)
        return idx

    def set(self, p: int, x) -> None:
        assert 0 <= p < self._n
        
        path = []
        curr = self._root
        # 上から下へ降りながらノードを作成
        for i in range(self._log - 1, -1, -1):
            path.append(curr)
            if (p >> i) & 1:
                if not self._r[curr]: self._r[curr] = self._new_node()
                curr = self._r[curr]
            else:
                if not self._l[curr]: self._l[curr] = self._new_node()
                curr = self._l[curr]
        
        self._v[curr] = x
        # 下から上へ値を更新
        for node in reversed(path):
            self._v[node] = self._op(self._v[self._l[node]], self._v[self._r[node]])

    def get(self, p: int):
        assert 0 <= p < self._n
        curr = self._root
        for i in range(self._log - 1, -1, -1):
            if not curr: break
            curr = self._r[curr] if (p >> i) & 1 else self._l[curr]
        return self._v[curr] if curr else self._e

    def prod(self, left: int, right: int):
        assert 0 <= left <= right <= self._n
        if left == right: return self._e
        
        # prod は範囲が複雑なため再帰が安全だが、スタックで擬似的に反復化可能
        # ここでは ACL の使用感に合わせつつ、高速な再帰で実装
        return self._prod_recursive(self._root, 0, self._size, left, right)

    def _prod_recursive(self, node: int, l: int, r: int, ql: int, qr: int):
        if not node or qr <= l or r <= ql:
            return self._e
        if ql <= l and r <= qr:
            return self._v[node]
        mid = (l + r) >> 1
        return self._op(self._prod_recursive(self._l[node], l, mid, ql, qr),
                        self._prod_recursive(self._r[node], mid, r, ql, qr))

    def all_prod(self):
        return self._v[self._root]

    def max_right(self, left: int, f: typing.Callable) -> int:
        assert 0 <= left <= self._n
        assert f(self._e)
        if left == self._n: return self._n
        
        sm = self._e
        return self._max_right_rec(self._root, 0, self._size, left, f, sm)[1]

    def _max_right_rec(self, node: int, l: int, r: int, ql: int, f, sm):
        if r <= ql:
            return sm, r
        if ql <= l:
            nxt = self._op(sm, self._v[node])
            if f(nxt):
                return nxt, r
            if r - l == 1:
                return sm, l
        
        mid = (l + r) >> 1
        res_sm, res_idx = self._max_right_rec(self._l[node], l, mid, ql, f, sm)
        if res_idx < mid:
            return res_sm, res_idx
        return self._max_right_rec(self._r[node], mid, r, ql, f, res_sm)

    def min_left(self, right: int, f: typing.Callable) -> int:
        assert 0 <= right <= self._n
        assert f(self._e)
        if right == 0: return 0
        
        sm = self._e
        return self._min_left_rec(self._root, 0, self._size, right, f, sm)[1]

    def _min_left_rec(self, node: int, l: int, r: int, qr: int, f, sm):
        if qr <= l:
            return sm, l
        if r <= qr:
            nxt = self._op(self._v[node], sm)
            if f(nxt):
                return nxt, l
            if r - l == 1:
                return sm, r
        
        mid = (l + r) >> 1
        res_sm, res_idx = self._min_left_rec(self._r[node], mid, r, qr, f, sm)
        if res_idx > mid:
            return res_sm, res_idx
        return self._min_left_rec(self._l[node], l, mid, qr, f, res_sm)
