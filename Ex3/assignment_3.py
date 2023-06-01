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


def hilbert(n: int):
    return np.mat([[1.0 / (i + j) for j in range(n)] for i in range(1, n + 1)])


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


def solve_hilbert_system(dim, disturb=None):
    std_vector = np.ones((dim, 1))
    rhs = std_vector if disturb is None else std_vector + disturb
    lower = cholesky(hilbert(dim))
    solution = backward_substitution(lower.transpose(), forward_substitution(lower, rhs))
    residual = rhs - hilbert(dim) * solution
    return np.linalg.norm(residual, np.inf), np.linalg.norm(solution - std_vector, np.inf)


if __name__ == '__main__':
    # Part 1: n = 10
    print(solve_hilbert_system(10))

    # Part 2: add disturbance
    results = [solve_hilbert_system(10, np.random.rand(10, 1) * np.float_power(10, -7)) for _ in range(100)]
    print(tuple(np.mean(list(u[i] for u in results)) for i in range(2)))

    # Part 3: varying dimensions
    n_list = [8, 12, 13, 14]
    result = [solve_hilbert_system(d) for d in n_list]
    for i, res in enumerate(result):
        print("N = {}: residual = {}, delta = {}".format(n_list[i], res[0], res[1]))

