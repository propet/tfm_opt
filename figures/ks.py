import numpy as np
import matplotlib.pyplot as plt

import scienceplots
plt.style.use(["science", "ieee"])
plt.rcParams.update({'figure.dpi': '300'})


def ks(g, rho=100):
    sum = 0
    for gi in g:
        sum += np.exp(rho * gi)
    ks = 1/rho * np.log(sum)
    return ks


x = np.linspace(-10, 10, 1000)
y = np.maximum(0, x)
ks_values_rho_1 = [ ks([0, xi], rho=1) for xi in x ]
ks_values_rho_10 = [ ks([0, xi], rho=10) for xi in x ]


plt.figure(figsize=(10, 5))
plt.plot(x, y, label="max(0, x)")
plt.plot(x, ks_values_rho_1, label=r"KS(0, x; $\rho$=1)")
plt.plot(x, ks_values_rho_10, label=r"KS(0, x; $\rho$=10)")
plt.xlabel("x")
plt.grid(True)
plt.legend()
plt.show()
