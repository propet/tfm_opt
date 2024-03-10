import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee"])
plt.rcParams.update({"figure.dpi": "300"})


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def smooth_step(x, mu_0, mu_1):
    return sigmoid(x) * mu_0 + (1 - sigmoid(x)) * mu_1


ETA_CHARGE = 0.8
ETA_DISCHARGE = 0.9

x = np.linspace(-10, 10, 1000)
y = np.maximum(0, x)
y = [smooth_step(xi, ETA_CHARGE, 1 / ETA_DISCHARGE) for xi in x]

plt.figure(figsize=(10, 5))
plt.plot(x, y)
plt.xlabel(r"$P_{bat}$")
plt.ylabel(r"$\eta$")
plt.grid(True)
plt.legend()
plt.show()
