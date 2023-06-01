import numpy as np


def cholesky(matrix):
    """
    Can be applied to symmetric positive definite matrices only.
    """
    assert matrix.shape[0] == matrix.shape[1]
    shape = matrix.shape[0]
    for i in range(shape):
        for j in range(i):
            matrix[i, i] -= matrix[i, j] ** 2
        matrix[i, i] = np.sqrt(matrix[i, i])
        for j in range(i + 1, shape):
            for k in range(i):
                matrix[j, i] -= matrix[i, k] * matrix[j, k]
            matrix[j, i] /= matrix[i, i]
            # set unused elements to 0
            matrix[i, j] = 0.0
    return matrix


def forward_substitution(l, b):
    # Algorithm 3.7
    dim = l.shape[0]
    ans = np.zeros((dim, 1), dtype=np.float_)
    for i in range(dim):
        if l[i, i] == 0:
            print("Error")
            return None
        ans[i, 0] = b[i, 0]
        for j in range(i):
            ans[i, 0] -= l[i, j] * ans[j, 0]
        ans[i, 0] /= l[i, i]
    return ans


def backward_substitution(u, b):
    # Algorithm 3.2
    dim = u.shape[0]
    ans = np.zeros((dim, 1), dtype=np.float_)
    for i in range(dim - 1, -1, -1):
        if u[i, i] == 0:
            print("Error")
            return None
        ans[i, 0] = b[i, 0]
        for j in range(dim - 1, i, -1):
            ans[i, 0] -= u[i, j] * ans[j, 0]
        ans[i, 0] /= u[i, i]
    return ans


def quadratic_fitting(t, f):
    assert t.shape[0] == f.shape[0]
    dim = t.shape[0]
    matrix_a = np.mat([
        [1, t[i], np.float_power(t[i], 2)]
        for i in range(dim)
    ], dtype=np.float_)
    gram = matrix_a.transpose() * matrix_a
    rhs = matrix_a.transpose() * f
    lower = cholesky(gram)
    solution = backward_substitution(lower.transpose(), forward_substitution(lower, rhs))
    return solution


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

    print(quadratic_fitting(t, y))


