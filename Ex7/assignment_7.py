import numpy as np


def composite_simpson(f, a, b, h):
    ans = np.float_(0.0)
    while a + h < b:
        ans += f(a) + 4.0 * f(a + h / 2) + f(a + h)
        a += h
    ans *= h
    ans += (b - a) * (f(a) + 4.0 * f((a + b) / 2) + f(b))
    ans /= 6
    return ans


def composite_gaussian(f, a, b, n):

    def x(i):
        return a + i * h + h / 2

    h = (b - a) / n
    delta = 0.5 / np.sqrt(3) * h
    return sum(f(x(i) - delta) + f(x(i) + delta) for i in range(n)) * h / 2


if __name__ == '__main__':
    ref_ln2 = np.log(2.0)
    ref_pi = np.pi

    # Section 1
    step1 = 0.027832
    step2 = 0.019680
    ln2_s = composite_simpson(lambda t: 1 / t, 1, 2, step1)
    pi_s = 4 * composite_simpson(lambda t: 1 / (1 + t * t), 0, 1, step2)

    # Section 2
    ln2_g = composite_gaussian(lambda t: 1 / t, 1, 2, 33)
    pi_g = 4 * composite_gaussian(lambda t: 1 / (1 + t * t), 0, 1, 46)
    print("Reference value: ln(2) = {}, pi = {}".format(ref_ln2, ref_pi))
    print()
    print("Composite simpson yields: ln(2) = {}, pi = {}".format(ln2_s, pi_s))
    print("Absolute error: {} & {}".format(abs(ref_ln2 - ln2_s), abs(ref_pi - pi_s)))
    print()
    print("Composite gaussian yields: ln(2) = {}, pi = {}".format(ln2_g, pi_g))
    print("Absolute error: {} & {}".format(abs(ref_ln2 - ln2_g), abs(ref_pi - pi_g)))

