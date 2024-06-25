import sys
import petsc4py
import numpy as np
from petsc4py import PETSc
import matplotlib.pyplot as plt

# Initialize PETSc and parse command-line options
petsc4py.init(sys.argv)

# Enable detailed logging
PETSc.Log.begin()


class DAE(object):
    def __init__(self):
        r"""
        Solve DAE with adjoint sensitivities
            .. math::
            \dot{x}(t) = -x(t) + y(t) \\
            0 = x(t)^2 + y(t)^2 - 1
        """

        self.comm = PETSc.COMM_WORLD
        self.t0 = 0.0
        self.tf = 10.0
        self.dt = 0.001

        # Setup vectors
        self.u = self.create_state_vector()
        self.r = PETSc.Vec().createSeq(1, comm=self.comm)  # quadrature value

        # Setup matrices
        self.Ju = PETSc.Mat().createDense([2, 2], comm=self.comm)
        self.Judot = PETSc.Mat().createDense([2, 2], comm=self.comm)

        # Set up monitors to store the solution
        self.sol_t = []
        self.sol_u = []

        # TimeStep PETSc object
        self.ts = self.create_ts()


    def create_state_vector(self):
        u = PETSc.Vec().createSeq(2, comm=self.comm)  # states
        # Initial conditions
        u[0] = 0.5
        u[1] = 0.8660254037
        return u


    def evalIFunction(self, ts, t, u, udot, f):
        r"""
        Implicit functions
        """
        f[0] = udot[0] + u[0] - u[1]
        f[1] = u[0]**2 + u[1]**2 - 1
        return True


    def evalIJacobian(self, ts, t, u, udot, shift, Ju, Judot):
        r"""
        Compute the derivatives of the implicit function F with respect to states u

        shift is the derivative of of u_dot with respect to u,
        whose value depends on the chosen discretization for the derivative (ODE integrator)

        .. math::
            \frac{dF}{du^n} = \frac{\partial F}{\partial \dot{u}} \bigg|_{u^n} \frac{\partial \dot{u}}{\partial u} \bigg|_{u^n} + \frac{\partial F}{\partial u} \bigg|_{u^n}
        """
        Ju[0, 0] = shift + 1
        Ju[0, 1] = -1
        Ju[1, 0] = 2 * u[0]
        Ju[1, 1] = 2 * u[1]
        Ju.assemble()

        Judot[0, 0] = 1
        Judot[0, 1] = 0
        Judot[1, 0] = 0
        Judot[1, 1] = 0
        Judot.assemble()
        return True


    def monitor(self, ts, step, time, u):
        self.sol_t.append(time)
        self.sol_u.append(u.getArray().copy())
        return 0


    def create_ts(self):
        ts = PETSc.TS().create(comm=self.comm)
        ts.setProblemType(ts.ProblemType.NONLINEAR)
        ts.setType(PETSc.TS.Type.THETA)  # Type of time integrator
        ts.setTheta(0.5)  # Crank-Nicolson scheme
        ts.setIFunction(self.evalIFunction, self.u)
        ts.setIJacobian(self.evalIJacobian, self.Ju, self.Judot)
        return ts


    def run(self):
        self.ts.setTime(self.t0)
        self.ts.setMaxTime(self.tf)
        self.ts.setTimeStep(self.dt)
        self.ts.setSaveTrajectory()
        self.ts.setFromOptions()
        self.ts.setMonitor(self.monitor)

        # Set up SNES and KSP options
        snes = self.ts.getSNES()
        snes.setTolerances(rtol=1e-8, atol=1e-8, max_it=100)
        ksp = snes.getKSP()
        ksp.setType('gmres')
        pc = ksp.getPC()
        pc.setType('jacobi')

        # Solve the DAE
        try:
            self.ts.solve(self.u)
        except PETSc.Error as e:
            if e.ierr == 71:
                print("PETSc Error 71: Likely a singular Jacobian. Try adjusting initial conditions or solver parameters.")
            else:
                print(f"PETSc Error {e.ierr}: {e}")
            sys.exit(1)


    def plot(self):
        # Convert solution to numpy arrays
        sol_t = np.array(self.sol_t)
        sol_u = np.array(self.sol_u)

        # Print final solution
        print("Final time:", sol_t[-1])
        print("Final solution:", sol_u[-1])

        # Plot the solution
        plt.figure(figsize=(10, 6))
        plt.plot(sol_t, sol_u[:, 0], label='x(t)')
        plt.plot(sol_t, sol_u[:, 1], label='y(t)')
        plt.xlabel('Time')
        plt.ylabel('Solution')
        plt.title('DAE Solution')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    dae = DAE()
    dae.run()
    print(dae.u.view())
    dae.plot()
