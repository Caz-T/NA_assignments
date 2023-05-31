import time

import numpy as np


def power_iteration(matrix):
    result = np.random.rand(matrix.shape[1], 1)
    eigenvalue = np.float_(0.0)
    while True:
        eigenvector = matrix * result
        eigenvalue = np.linalg.norm(eigenvector, np.inf)
        normalised = eigenvector / eigenvalue
        if np.linalg.norm(normalised - result, 2) < np.float_power(10, -5):
            break
        result = normalised
    return eigenvalue


def is_quasi_upper(matrix, threshold=0.05):
    assert matrix.shape[0] == matrix.shape[1]
    dim = matrix.shape[0]
    alarm = False
    for i in range(dim):
        for j in range(i + 1, dim):
            if abs(matrix[j, i]) > threshold:
                if j != i + 1 or alarm:
                    return False
                alarm = True
            elif j == i + 1:
                alarm = False
    return True


def get_eigvals_from_quasi(matrix, threshold):
    # compute eigenvalues based on diagonal elements/2-dim blocks

    def get_dim2_eig(m):
        assert m.shape[0] == m.shape[1] == 2
        dt = np.sqrt(np.float_power(m[0, 0] - m[1, 1], 2) + 4.0 * m[1, 0] * m[0, 1])
        return [(m[0, 0] + m[1, 1] + dt) / 2, (m[0, 0] + m[1, 1] - dt) / 2]

    evs = []
    i = 0
    while i < matrix.shape[0]:
        if i == matrix.shape[0] - 1:
            evs.append(matrix[i, i])
            break
        if np.abs(matrix[i + 1, i]) < threshold:
            evs.append(matrix[i, i])
            i += 1
            continue
        evs.extend(get_dim2_eig(matrix[np.ix_([i, i + 1], [i, i + 1])]))
        i += 2

    return evs


def basic_qr(matrix, verbose=False):
    threshold = 0.05
    while not is_quasi_upper(matrix, threshold):
        q, r = np.linalg.qr(matrix)
        matrix = r * q
        if verbose:
            print(matrix)

    return get_eigvals_from_quasi(matrix, threshold)


def shifted_qr(matrix, verbose=False):
    threshold = 0.05
    assert matrix.shape[0] == matrix.shape[1]
    dim = matrix.shape[0]
    while not is_quasi_upper(matrix, threshold):
        s = matrix[dim - 1, dim - 1]
        q, r = np.linalg.qr(matrix - s * np.identity(dim))
        matrix = r * q + s * np.identity(dim)
        if verbose:
            print(matrix)

    return get_eigvals_from_quasi(matrix, threshold)


if __name__ == '__main__':
    # Section 1
    a = np.mat([
        [5, -4, 1],
        [-4, 6, -4],
        [1, -4, 7],
    ], dtype=np.float_)
    ev = np.linalg.eigvals(a)
    print("Matrix A: power iteration yields {:.6f}, eigvals yields {:.6f}"
          .format(power_iteration(a), np.linalg.norm(ev, np.inf)))
    b = np.mat([
        [25, -41, 10, -6],
        [-41, 68, -17, 10],
        [10, -17, 5, -3],
        [-6, 10, -3, 2],
    ], dtype=np.float_)
    ev = np.linalg.eigvals(b)
    print("Matrix B: power iteration yields {:.6f}, eigvals yields {:.6f}"
          .format(power_iteration(b), np.linalg.norm(ev, np.inf)))

    # Section 2
    a = np.mat([
        [.5, .5, .5, .5],
        [.5, .5, -.5, -.5],
        [.5, -.5, .5, -.5],
        [.5, -.5, -.5, .5],
    ])
    # matrix A is an orthonormal matrix, thus the following line will take forever to execute
    # evs = basic_qr(a)
    # print("Basic-QR yields eigenvalues {}".format(evs))

    # Section 3
    evs = shifted_qr(a, True)
    print("Shifting-QR yields eigenvalues {}".format(evs))
