#!/usr/bin/env python3
import numpy as np
import time
import timeit

N = 10000
sizes = [4, 6, 8, 10, 14, 20, 27, 38, 52, 71, 98, 135, 186, 256]

for n in sizes:
    a = np.random.randn(n).astype("f")
    b = np.random.randn(n).astype("f")
    dt = timeit.timeit(lambda: a @ b, number=N, timer=time.perf_counter_ns)
    print(f"numpy,dot,{n},{dt//N}")

for n in sizes:
    a = np.random.randn(n, n).astype("f").T
    b = np.random.randn(n).astype("f")
    dt = timeit.timeit(lambda: a @ b, number=N, timer=time.perf_counter_ns)
    print(f"numpy,gemv,{n},{dt//N}")

for n in sizes:
    a = np.random.randn(n, n).astype("f").T
    b = np.random.randn(n, n).astype("f").T
    dt = timeit.timeit(lambda: a @ b, number=N, timer=time.perf_counter_ns)
    print(f"numpy,gemm,{n},{dt//N}")

for n in sizes:
    a = np.random.randn(n).astype("f")
    b = np.random.randn(n).astype("f")
    dt = timeit.timeit(lambda: np.outer(a, b), number=N, timer=time.perf_counter_ns)
    print(f"numpy,ger,{n},{dt//N}")
