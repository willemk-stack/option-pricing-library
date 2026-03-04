import numpy as np


def isotonic_regression(y: np.ndarray, *, increasing: bool = True) -> np.ndarray:
    """
    Unweighted isotonic regression using Pool-Adjacent-Violators (PAV).
    Minimizes sum (y_hat - y)^2 subject to y_hat monotone.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if y.size == 0:
        return y.copy()

    if not increasing:
        # decreasing fit == increasing fit of negated values, then negate back
        return -isotonic_regression(-y, increasing=True)

    # Each point starts as its own block
    # We'll maintain a stack of blocks with (start, end_exclusive, sum, count)
    starts = []
    ends = []
    sums = []
    counts = []

    for i, yi in enumerate(y):
        starts.append(i)
        ends.append(i + 1)
        sums.append(float(yi))
        counts.append(1)

        # While monotonicity is violated: mean(prev) > mean(curr), merge blocks
        while len(sums) >= 2:
            m_prev = sums[-2] / counts[-2]
            m_curr = sums[-1] / counts[-1]
            if m_prev <= m_curr:
                break
            # merge last two blocks
            sums[-2] += sums[-1]
            counts[-2] += counts[-1]
            ends[-2] = ends[-1]
            # pop last
            starts.pop()
            ends.pop()
            sums.pop()
            counts.pop()

    # Expand block means back to full length
    y_fit = np.empty_like(y)
    for s, e, sm, c in zip(starts, ends, sums, counts, strict=True):
        y_fit[s:e] = sm / c
    return y_fit
