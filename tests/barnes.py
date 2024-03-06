# First party modules
from pyoptsparse import SLSQP, Optimization
import numpy as np

a = np.array(
    [
        75.196,
        -3.8112,
        0.12694,
        -2.0567e-3,
        1.0345e-5,
        -6.8306,
        0.030234,
        -1.28134e-3,
        3.5256e-5,
        -2.266e-7,
        0.25645,
        -3.4604e-3,
        1.3514e-5,
        -28.106,
        -5.2375e-6,
        -6.3e-8,
        7.0e-10,
        3.4054e-4,
        -1.6638e-6,
        -2.8673,
        0.0005,
    ]
)


# rst begin objfunc
def objfunc(xdict):
    x = xdict["xvars"]
    funcs = {}

    y1 = x[0] * x[1]
    y2 = y1 * x[0]
    y3 = x[1] ** 2
    y4 = x[0] ** 2

    funcs["obj"] = (
        a[0]
        + a[1] * x[0]
        + a[2] * y4
        + a[3] * y4 * x[0]
        + a[4] * y4**2
        + a[5] * x[1]
        + a[6] * y1
        + a[7] * y1 * x[0]
        + a[8] * y1 * y4
        + a[9] * y2 * y4
        + a[10] * y3
        + a[11] * x[1] * y3
        + a[12] * y3**2
        + a[13] / (x[1] + 2)
        + a[14] * y3 * y4
        + a[15] * y1 * y4 * x[1]
        + a[16] * y1 * y3 * y4
        + a[17] * x[0] * y3
        + a[18] * y1 * y3
        + a[19] * np.exp(a[20] * y1)
    )

    funcs["cons"] = [
        1 - y1 / 700,
        y4 / 25**2 - x[1] / 5,
        x[0] / 500 - 0.11 - (x[1] / 50 - 1) ** 2,
    ]

    fail = False

    return funcs, fail


# Optimization Object
optProb = Optimization("barnes", objfunc)

# Design Variables
optProb.addVarGroup("xvars", 2, "c", value=0)

# Constraints
# Equality constraints are specified using the same values for lower and upper bounds
# None equals to -infinity to lower, and +infinity to upper
optProb.addConGroup("cons", 3, lower=None, upper=0.0)

# rst begin addObj
# Objective
optProb.addObj("obj")

# rst begin print
# Check optimization problem
print(optProb)

# rst begin OPT
# Optimizer
optOptions = {"IPRINT": -1}
opt = SLSQP(options=optOptions)

# rst begin solve
# Solve
sol = opt(optProb, sens="FD")

# rst begin check
# Check Solution
print(sol)
