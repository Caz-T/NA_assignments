import numpy as np


def quadratic_fitting(x, y):



def diokein_method(a, b, c, f, n):
    # Algorithm 3.12 in textbook
    for i in range(1, n):
        temp = a[i] / b[i - 1]
        b[i] -= temp * c[i]
        f[i] -= temp * f[i - 1]
    ans = np.zeros(n, dtype=np.float_)
    ans[n - 1] = f[n - 1] / b[n - 1]
    for i in range(n - 2, -1, -1):
        ans[i] = (f[i] - c[i] * ans[i + 1]) / b[i]
    return ans


if __name__ == "__main__":
    # Section 1
    t = np.array(list(x / 2 for x in range(2, 17)), dtype=np.float_)
    y = np.array([
        33.4, 79.5, 122.65, 159.05, 189.15, 214.15, 238.65, 252.2, 267.55, 280.5, 296.65, 301.65, 310.40, 318.15, 325.15
    ], dtype=np.float_)



