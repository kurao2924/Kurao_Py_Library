from typing import Optional
from typing import Optional
from itertools import accumulate


class BitVector:
    # --- ここに型宣言を追加 ---
    B: list[int]
    acc: list[int]
    # -----------------------
    def __init__(self, B):
        """累積和を使用したビットベクトル

        Args:
            B (list[int]): 要素は0 or 1
        """
        self.B = B
        self.acc = list(accumulate(B))

    def __len__(self) -> int:
        """元の配列の長さ

        Returns:
            int: 元の配列の長さ
        """
        return len(self.B)

    def __getitem__(self, i: int) -> int:
        """B[i]を取得

        Args:
            i (int): i番目の要素

        Returns:
            int: B[i]
        """
        return self.B[i]

    def rank0(self, i: int) -> int:
        """元の配列B[0..i)の0の個数

        Args:
            i (int): 上限

        Returns:
            int: 0の個数
        """
        if i <= 0:
            return 0
        i = min(i, len(self))
        return i - self.rank1(i)

    def rank1(self, i: int) -> int:
        """元の配列B[0..i)の1の個数

        Args:
            i (int): 上限

        Returns:
            int: 1の個数
        """
        if i <= 0:
            return 0
        i = min(i, len(self))
        return self.acc[i - 1]

    def rank0_all(self) -> int:
        """元の配列Bの0の個数

        Returns:
            int: 0の個数
        """
        return self.rank0(len(self.B))

    def rank1_all(self) -> int:
        """元の配列Bの1の個数

        Returns:
            int: 1の個数
        """
        return self.rank1(len(self.B))

    def select0(self, k: int) -> Optional[int]:
        """0がk番目に現れるindexを返す

        Args:
            k (int): k番目. 1-indexed.

        Returns:
            Optional[int]: 該当のindex. 存在しない場合はNone
        """
        if (k <= 0) or (k > self.rank0_all()):
            return None

        left, right = 0, len(self)
        while (right - left) > 1:
            mid = (left + right) // 2
            if self.rank0(mid) < k:
                left = mid
            else:
                right = mid
        return left

    def select1(self, k: int) -> Optional[int]:
        """1がk番目に現れるindexを返す

        Args:
            k (int): k番目. 1-indexed.

        Returns:
            Optional[int]: 該当のindex. 存在しない場合はNone
        """
        if (k <= 0) or (k > self.rank1_all()):
            return None

        left, right = 0, len(self)
        while (right - left) > 1:
            mid = (left + right) // 2
            if self.rank1(mid) < k:
                left = mid
            else:
                right = mid
        return left

class WaveletMatrix:
    def __init__(self, T: list[int]):
        """WaveletMatrix

        Args:
            T (list[int]): 整数列
        """
        self.bit_size: int = len(bin(max(T)))-2 if T else 0
        self.wavelet_matrix: list[BitVector] = self._build(T)

    def _build(self, T: list[int]) -> list[BitVector]:
        """WaveletMatrixを構築する

        Args:
            T (list[int]): 元の配列

        Returns:
            list[BitVector]: WaveletMatrix
        """
        wavelet_matrix: list[BitVector] = []
        for digit in range(self.bit_size)[::-1]:
            zeros = [t for t in T if not self._get_i_bit(t, digit + 1)]
            ones = [t for t in T if self._get_i_bit(t, digit + 1)]

            T = zeros + ones
            wavelet_matrix.append(BitVector([self._get_i_bit(t, digit) for t in T]))
        return wavelet_matrix

    def __len__(self) -> int:
        """len(T).

        Returns:
            int: len(T)
        """
        if len(self.wavelet_matrix) == 0:
            return 0
        return len(self.wavelet_matrix[0])

    def __getitem__(self, i: int) -> int:
        """元の配列T[i]を返す

        Args:
            i (int): index

        Returns:
            int: T[i]
        """
        return self.access(i)

    def _get_i_bit(self, x: int, digit: int) -> int:
        """xの(下から数えて)digit桁目のbitを計算

        Args:
            x (int): 整数
            digit (int): 桁数

        Returns:
            int: 0 or 1
        """
        return 1 & (x >> digit)

    def _next_range(self, B: BitVector, bit: int, left: int, right: int) -> tuple[int, int]:
        """B[j][left..right)におけるbit(0 or 1)が, B[j+1]においてどの範囲になるかを計算

        Args:
            B (BitVector): Bit Vector B[j].
            bit (int): 範囲に含まれるbit. 0 or 1.
            left (int): Bの範囲の下限.
            right (int): Bの範囲の上限.

        Returns:
            (int, int): B[j+1]の[left..right)
        """
        if bit == 0:
            left = B.rank0(left)
            right = B.rank0(right)
        else:
            left = B.rank0_all() + B.rank1(left)
            right = B.rank0_all() + B.rank1(right)
        return (left, right)

    def access(self, i: int) -> int:
        """元の配列T[i]を返す

        Args:
            i (int): index

        Returns:
            int: T[i]
        """
        x = 0
        for digit, B in enumerate(self.wavelet_matrix):
            bit = B[i]
            x |= bit << (self.bit_size - digit - 1)

            if bit == 0:
                i = B.rank0(i + 1) - 1
            else:
                i = B.rank0_all() + B.rank1(i + 1) - 1
        return x

    def rank(self, x: int, right: int) -> int:
        """T[0..right)における, xの出現回数を返す

        Args:
            x (int): 対象の要素
            right (int): Tの範囲の上限

        Returns:
            int: 出現回数
        """
        return self.rank_range(x, 0, right)

    def rank_range(self, x: int, left: int, right: int) -> int:
        """T[left..right)におけるxの出現回数を返す

        Args:
            x (int): 対象の要素
            left (int): Tの範囲の下限
            right (int): Tの範囲の上限

        Returns:
            int: 出現回数
        """
        if (left >= right) or (len(bin(x)) - 2 > self.bit_size) or (x < 0):
            return 0

        left = max(left, 0)
        right = min(right, len(self))
        for digit, B in enumerate(self.wavelet_matrix):
            x_bit = self._get_i_bit(x, self.bit_size - digit - 1)
            left, right = self._next_range(B, x_bit, left, right)
        return right - left

    def select(self, x: int, k: int) -> Optional[int]:
        """元の配列Tのk個目のxの出現位置 (index) を返す

        Args:
            x (int): 対象の要素.
            k (int): 何個目の出現か. 1-index.

        Returns:
            Optional[int]: index. 該当要素が存在しない場合はNone.
        """
        if not (0 <= x < 1 << self.bit_size):
            return None
        if k <= 0:
            return None

        # xが最終的な配列のどの範囲に入るか計算
        left, right = 0, len(self)
        for digit, B in enumerate(self.wavelet_matrix):
            x_bit = self._get_i_bit(x, self.bit_size - digit - 1)
            left, right = self._next_range(B, x_bit, left, right)

        if left >= right:
            return None

        # 上記範囲のk番目が, 元の配列で何番目か計算
        index = left + k - 1
        for digit, B in enumerate(self.wavelet_matrix[::-1]):
            if index is None:
                return None

            x_bit = self._get_i_bit(x, digit)
            if x_bit == 0:
                index = B.select0(index + 1)
            else:
                index = B.select1(index - (B.rank0_all() - 1))
        return index

    def quantile(self, left: int, right: int, k: int) -> Optional[int]:
        """元の配列T[left..right)の中のk番目に小さい値を返す

        Args:
            left (int): Tの範囲の下限
            right (int): Tの範囲の上限
            k (int): 何番目の要素か, 1-index

        Returns:
            Optional[int]: k番目に小さい値. 該当要素が存在しない場合はNone.
        """
        if (left >= right) or (k <= 0) or (right - left < k):
            return None

        left = max(left, 0)
        right = min(right, len(self))

        x = 0
        for digit, B in enumerate(self.wavelet_matrix):
            num_zeros = B.rank0(right) - B.rank0(left)

            # 該当要素が0の中か or 1の中か
            bit = 0 if k <= num_zeros else 1
            left, right = self._next_range(B, bit, left, right)

            if bit == 1:
                k -= num_zeros
                x |= bit << (self.bit_size - digit - 1)
        return x

    def range_freq_to(self, left: int, right: int, upper: int) -> int:
        """T[left..right)の0 <= x < upper となるxの個数を計算する

        Args:
            left (int): Tの範囲の下限
            right (int): Tの範囲の上限
            upper (int): 要素の上限

        Returns:
            int: 0 <= x < upperとなる要素の数
        """
        if (left >= right) or (upper <= 0):
            return 0

        # 全ての要素が < upper
        if len(bin(upper))-2 > self.bit_size:
            return right - left

        cnt = 0
        left = max(left, 0)
        right = min(right, len(self))
        for digit, B in enumerate(self.wavelet_matrix):
            upper_bit = self._get_i_bit(upper, self.bit_size - digit - 1)
            if upper_bit == 1:
                cnt += B.rank0(right) - B.rank0(left)

            left, right = self._next_range(B, upper_bit, left, right)

        return cnt

    def range_freq_from(self, left: int, right: int, lower: int) -> int:
        """T[left..right)のlower <= x となるxの個数を計算する

        Args:
            left (int): Tの範囲の下限
            right (int): Tの範囲の上限
            lower (int): 要素の下限

        Returns:
            int: lower <= x となる要素の数
        """
        if (left >= right) or (lower > (1 << self.bit_size)):
            return 0

        left = max(left, 0)
        right = min(right, len(self))
        return (right - left) - self.range_freq_to(left, right, lower)

    def range_freq(self, left: int, right: int, lower: int, upper: int) -> int:
        """T[left..right)のlower <= x < upper となるxの個数を計算する

        Args:
            left (int): Tの範囲の下限
            right (int): Tの範囲の上限
            lower (int): 要素の下限
            upper (int): 要素の上限

        Returns:
            int: lower <= x < upperとなる要素の数
        """
        if (left >= right) or (lower >= upper) or (lower > (1 << self.bit_size)):
            return 0

        return self.range_freq_to(left, right, upper) - self.range_freq_to(
            left, right, lower
        )

    def prev_value(self, left: int, right: int, upper: int) -> Optional[int]:
        """T[left..right)の中で, 0 <= x < upperを満たす最大のxを返す

        Args:
            left (int): Tの範囲の下限
            right (int): Tの範囲の上限
            upper (int): 要素の上限

        Returns:
            Optional[int]: 0 <= x < upperを満たす最大のx. 存在しない場合None.
        """
        # T[left..right)内の, 0 <= x < upperのxの個数
        cnt = self.range_freq_to(left, right, upper)
        if cnt == 0:
            return None
        return self.quantile(left, right, cnt)

    def next_value(self, left: int, right: int, lower: int) -> Optional[int]:
        """T[left..right)の中で, lower <= x を満たす最小のxを返す

        Args:
            left (int): Tの範囲の下限
            right (int): Tの範囲の上限
            lower (int): 要素の下限

        Returns:
            int: lower <= x を満たす最小のx
        """
        # T[left..right)内の, lower <= xとなるxの個数
        cnt = self.range_freq_from(left, right, lower)
        if cnt == 0:
            return None
        return self.quantile(left, right, right - left - cnt + 1)