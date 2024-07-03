import numpy as np
from scikits.odes.dae import dae


SOLVER = 'ida'
extra_options = {'old_api': False}
tout = np.linspace(0, 10, 1000)
y_initial = np.array([0.5, 0.8660254037])
ydot_initial = np.array([-0.6160254037844386, 0.0])


def right_hand_side(t, y, ydot, residue):
    """
    \dot{x}(t) = -x(t) + y(t) \\
    0 = x(t)^2 + y(t)^2 - 1
    """
    residue[0] = ydot[0] + y[0] - y[1]
    residue[1] = y[0]**2 + y[1]**2 - 1


def jacobian(t, y, ydot, residual, cj, J):
    J[0, 0] = cj + 1
    J[0, 1] = -1
    J[1, 0] = 2 * y[0]
    J[1, 1] = 2 * y[1]


dae_solver = dae(SOLVER, right_hand_side, jacfn=jacobian, **extra_options)
output = dae_solver.solve(tout, y_initial, ydot_initial)
# print(output)
print("iterations: ", len(output.values.y))
print(output.values.y)
