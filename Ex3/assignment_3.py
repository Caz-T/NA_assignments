import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def solve_hilbert_system(dim, disturb=None):
    std_vector = np.ones((dim, 1))
    rhs = std_vector if disturb is None else std_vector + disturb
    lower = cholesky(hilbert(dim))
    inv_lower = np.linalg.inv(lower)
    solution = inv_lower.transpose() * inv_lower * rhs
    residual = rhs - hilbert(dim) * solution
    return np.linalg.norm(residual, np.inf), np.linalg.norm(solution - std_vector, np.inf)


if __name__ == '__main__':
    # Part 1: n = 10
    print(solve_hilbert_system(10))

    # Part 2: add disturbance
    results = [solve_hilbert_system(10, np.random.rand(10, 1) * np.float_power(10, -7)) for _ in range(100)]
    print(tuple(np.mean(list(u[i] for u in results)) for i in range(2)))

    # Part 3: varying dimensions
    result = [solve_hilbert_system(d) for d in [8, 12, 13, 14]]
    print(result)

