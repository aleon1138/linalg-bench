#!/usr/bin/env python
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt


t = pd.read_csv(sys.argv[1], names=["lib", "func", "dim", "nsec"])

libs = sorted(t["lib"].unique())
funcs = sorted(t["func"].unique())

for (i, func) in enumerate(funcs):
    plt.subplot(2, 2, i + 1)
    t0 = t[t["func"] == func]

    for lib in libs:
        t1 = t0[t0["lib"] == lib]
        plt.loglog(t1["dim"], t1["nsec"], label=lib)
    plt.legend()
    plt.title(func.upper())

plt.show()
