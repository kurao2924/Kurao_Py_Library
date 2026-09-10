def hilbert_order(x: int, y: int) -> int:
    # x, y の最大値から必要な 2^k を決定
    maxc = max(x, y)
    k = maxc.bit_length()
    maxn = 1 << k

    d = 0
    s = maxn >> 1
    while s:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0

        d += s * s * ((rx * 3) ^ ry)

        if ry == 0:
            if rx == 1:
                x = maxn - 1 - x
                y = maxn - 1 - y
            x, y = y, x

        s >>= 1

    return d