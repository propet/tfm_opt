import sys
import petsc4py
import numpy as np
from petsc4py import PETSc
# import pdb

# Initialize PETSc and parse command-line options
petsc4py.init(sys.argv)

# Enable detailed logging
# PETSc.Log.begin()


# import traceback
# import logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# def exception_handler(exc_type, exc_value, exc_traceback):
#     logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
# sys.excepthook = exception_handler


class DAE(object):
    def __init__(self):
        r"""
        Solve DAE with adjoint sensitivities
        .. math::
            \dot{u}_0 + p_0 u_0 - p_1 * u_1 * (1 - u_0 - u_1) = 0 \\
            \dot{u}_1 - p_1 u_0^2 + u_1 = 0 \\
            u_2 - 1 + u_0 + u_1 = 0

        And cost function:
        .. math::
            \Psi(y_0, p) = \Phi(y_F, p) + \int_{t_0}^{t_F} r(y(t), p, t) \, dt \\
            \Psi = \int_0^T (u_0 + u_1 + u_2) dt
        """

        self.comm = PETSc.COMM_WORLD
        self.t0 = 0.0
        self.tf = 1.0
        self.dt = 0.001

        # Setup vectors
        self.p = self.create_parameters_vector()
        self.u = self.create_state_vector()

        # self.r = PETSc.Vec().createSeq(1, comm=self.comm)  # quadrature value
        # self.r.setUp()
        self.r = PETSc.Vec().create(comm=self.comm)
        self.r.setSizes(1)
        self.r.setFromOptions()
        self.r.setUp()

        # Setup matrices
        self.Ju = PETSc.Mat().createDense([3, 3], comm=self.comm)
        self.Ju.setUp()
        self.Jp = PETSc.Mat().createDense([3, 2], comm=self.comm)
        self.Jp.setUp()

        self.drdu = PETSc.Mat().createDense([1, 3], comm=self.comm)
        self.drdu.setUp()
        self.drdp = PETSc.Mat().createDense([1, 2], comm=self.comm)
        self.drdp.setUp()

        # TimeStep PETSc object
        self.ts = self.create_ts()

    def create_parameters_vector(self):
        p = PETSc.Vec().createSeq(2, comm=self.comm)  # parameters
        p[0] = 1.0
        p[1] = 2.0
        p.assemble()
        return p

    def create_state_vector(self):
        u = PETSc.Vec().createSeq(3, comm=self.comm)  # states
        # Initial conditions
        u[0] = 2.0
        u[1] = 2.0
        u[2] = -3.0
        u.assemble()
        return u


    def create_adjoint_variables(self):
        # Derivatives of the cost function with respect the two states and two parameters
        adj_lambda = PETSc.Vec().createSeq(3, comm=self.comm)  # adjoint variables lambda: ∂Psi/∂u
        adj_mu = PETSc.Vec().createSeq(2, comm=self.comm)  # adjoint variable mu: ∂Psi/∂p

        # Terminal conditions
        # Initialize to the derivative of the final state term of the cost function
        # with respect the states(adj_lambda) and parameters(adj_mu)
        # Since the integral term of the cost function we'll add it as another ODE
        # to the augmente DAE
        # .. math:
        #     adj_lambda = \frac{\partial \Phi(y_F, p)}{\partial u_N}
        #     adj_mu = \frac{\partial \Phi(y_F, p)}{\partial p}
        # Since we don't have a Phi term, the initializations are 0
        adj_lambda[0] = 0.0
        adj_lambda[1] = 0.0
        adj_lambda[2] = 0.0
        adj_lambda.assemble()
        adj_mu[0] = 0.0
        adj_mu[1] = 0.0
        adj_mu.assemble()

        return adj_lambda, adj_mu


    def evalIFunction(self, ts, t, u, udot, f):
        r"""
        Implicit functions
        """
        f[0] = udot[0] + self.p[0] * u[0] - self.p[1] * u[1] * (1.0 - u[0] - u[1])
        f[1] = udot[1] - self.p[1] * u[0] ** 2.0 + u[1]
        f[2] = u[2] - 1.0 + u[0] + u[1]
        f.assemble()
        return True


    def evalIJacobian(self, ts, t, u, udot, shift, Ju, P):
        r"""
        Compute the derivatives of the implicit function F with respect to states u

        shift is the derivative of of u_dot with respect to u,
        whose value depends on the chosen discretization for the derivative (ODE integrator)

        .. math::
            \frac{dF}{du^n} = \frac{\partial F}{\partial \dot{u}} \bigg|_{u^n} \frac{\partial \dot{u}}{\partial u} \bigg|_{u^n} + \frac{\partial F}{\partial u} \bigg|_{u^n}
        """
        Ju[0, 0] = shift + self.p[0] + self.p[1] * u[1]
        Ju[0, 1] = -self.p[1] * u[0] + self.p[1] * 2.0 * u[1]
        Ju[0, 2] = 0.0
        Ju[1, 0] = -self.p[1] * 2.0 * u[0]
        Ju[1, 1] = shift + 1.0
        Ju[1, 2] = 0.0
        Ju[2, 0] = 1.0
        Ju[2, 1] = 1.0
        Ju[2, 2] = 1.0
        Ju.assemble()
        return True


    def evalIJacobianP(self, ts, t, u, udot, shift, Jp):
        r"""
        Compute the derivatives of the implicit function F with respect to parameters p
        """
        Jp[0, 0] = u[0]
        Jp[0, 1] = -u[1] + u[0] * u[1] + u[1] ** 2.0
        Jp[1, 0] = 0.0
        Jp[1, 1] = -u[0] ** 2.0
        Jp[2, 0] = 0.0
        Jp[2, 1] = 0.0
        Jp.assemble()
        return True


    def costIntegration(self, ts, t, u, r):
        r[0] = u[0] + u[1] + u[2]
        r.assemble()
        return True


    def costIntegrationJacobian(self, ts, t, u, drdu, P):
        drdu[0, 0] = 1.0
        drdu[0, 1] = 1.0
        drdu[0, 2] = 1.0
        drdu.assemble()
        return True


    def costIntegrationJacobianP(self, ts, t, u, drdp):
        drdp[0, 0] = 0.0
        drdp[0, 1] = 0.0
        drdp.assemble()
        return True


    def create_ts(self):
        ts = PETSc.TS().create(comm=self.comm)
        ts.setProblemType(ts.ProblemType.NONLINEAR)
        ts.setType(PETSc.TS.Type.THETA)  # Type of time integrator
        ts.setTheta(0.5)  # Crank-Nicolson scheme
        ts.setIFunction(self.evalIFunction, self.u)
        ts.setIJacobian(self.evalIJacobian, self.Ju, self.Ju)
        ts.setIJacobianP(self.evalIJacobianP, self.Jp)
        return ts


    def set_quadts(self):
        try:
            # print(f"self.r type: {type(self.r)}, size: {self.r.getSize()}")
            # # self.quadts.setRHSFunction(self.costIntegration, self.r)
            self.quadts.setRHSFunction(self.costIntegration, None)
            # # self.quadts.setIFunction(self.costIntegration, self.r)
            # print("RHS function set for quadrature TS.")
        except PETSc.Error as e:
            print(f"Error setting RHS function for quadrature TS: {e}")
            raise

        try:
            # self.quadts.setRHSJacobian(self.costIntegrationJacobian, self.drdu, self.drdu_p)
            self.quadts.setRHSJacobian(self.costIntegrationJacobian, self.drdu)
            print("RHS Jacobian set for quadrature TS.")
        except PETSc.Error as e:
            print(f"Error setting RHS Jacobian for quadrature TS: {e}")
            raise

        try:
            self.quadts.setRHSJacobianP(self.costIntegrationJacobianP, self.drdp)
            print("RHS JacobianP set for quadrature TS.")
        except PETSc.Error as e:
            print(f"Error setting RHS JacobianP for quadrature TS: {e}")
            raise


    def run(self):
        self.ts.setTime(self.t0)
        self.ts.setMaxTime(self.tf)
        self.ts.setTimeStep(self.dt)
        self.ts.setSaveTrajectory()
        self.ts.setFromOptions()

        # Add monitoring
        def monitor(ts, step, time, u):
            # print(f"Step {step}, Time {time}, Solution: {u.array}")
            return 0
        self.ts.setMonitor(monitor)

        # Solve
        self.ts.solve(self.u)
        print("Forward solve complete.")
        print(f"Final u: {self.u.getArray()}")
        print(f"self.r type: {type(self.r)}, size: {self.r.getSize()}")


    def get_adjoint(self):
        self.adj_lambda, self.adj_mu = self.create_adjoint_variables()
        self.ts.setCostGradients(self.adj_lambda, self.adj_mu)
        self.quadts = self.ts.createQuadratureTS(forward=False)
        self.set_quadts()
        self.ts.adjointSetUp()
        self.ts.adjointSolve()


if __name__ == "__main__":
    # pdb.set_trace()
    dae = DAE()
    dae.run()
    print(dae.u.view())
    dae.get_adjoint()
    print(dae.adj_lambda)
    print(dae.adj_mu)
