import time
import numpy as np
import npufunc_directory.npufunc as npufunc

def timeit(f):
    start = time.perf_counter(); f(); return time.perf_counter() - start


x = np.linspace(0.1, 0.9, 10000000)

print(timeit(lambda : npufunc.logit(x)))
