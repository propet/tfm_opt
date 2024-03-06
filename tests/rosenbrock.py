# First party modules
from pyoptsparse import SLSQP, Optimization


# rst begin objfunc
def objfunc(xdict):
    x = xdict["xvars"]
    funcs = {}
    funcs["obj"] = (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    fail = False

    return funcs, fail


# rst begin optProb
# Optimization Object
optProb = Optimization("Rosenbrock function", objfunc)

# rst begin addVar
# Design Variables
optProb.addVarGroup("xvars", 2, "c", value=0)

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
