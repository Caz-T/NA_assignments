import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Section 1
def section_1():

    def estimate_der(step):
        return (np.sin(step + np.float64(1.0)) - np.sin(np.float64(1.0))) / step

    true_value = np.cos(1.0)
    xranges = np.float_power(10, np.arange(-16, 1))
    error_bounds = np.array(list(np.abs(estimate_der(ind) - true_value) for ind in xranges))
    plt.figure(figsize=(10, 7))
    plt.plot(xranges, error_bounds)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(np.float_power(10, -17), 10.0)
    plt.ylim(np.float_power(10, -17), 10.0)
    plt.xlabel("Step length")
    plt.ylabel("Error bound")
    plt.plot([np.float_power(10, -17), np.float_power(10, 1)], [np.float_power(10, 0), np.float_power(10, -17)], "b:,")
    plt.plot([np.float_power(10, -17), np.float_power(10, 1)], [np.float_power(10, -17), np.float_power(10, 1)], "b:,")
    plt.title("Error margin of sin'(x)")
    plt.savefig('1_1.png')
    plt.close()


def section_2():
    ans = np.float32(0)
    prevans = np.float32(0)
    n = 1
    start_time = time.time()
    while True:
        prevans = ans
        ans += np.float32(1.0 / n)
        if ans == prevans:
            break
        n += 1
    end_time = time.time()
    print("{} single-precision iterations in {} seconds, yielding {:.6f}".format(n, end_time - start_time, ans))

    double_ans = np.float_(0)
    for i in tqdm(range(n - 1)):
        double_ans += np.float_(1.0 / (i + 1))
    print("Double precision calculations yield {:.6f}".format(double_ans))
    print("Result difference: {}".format(double_ans - ans))


if __name__ == '__main__':
    section_2()




