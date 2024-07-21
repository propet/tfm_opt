import numpy as np
from pyoptsparse import Optimization
from custom_types import DesignVariableInfo, DesignVariables, Parameters, ConstraintInfo
from typing import Callable, List, Dict, Union
from pyoptsparse import OPT
from pyoptsparse import ALPSO, CONMIN, IPOPT, NLPQLP, NSGA2, PSQP, ParOpt, SLSQP, SNOPT


class Opt:
    optProb: Optimization
    name: str
    constraints_info: List[ConstraintInfo]
    design_variables_info: List[DesignVariableInfo]
    parameters: Parameters
    objective: Callable[["Opt", DesignVariables], np.ndarray]
    objective_name: str = "obj"
    optimizer: ALPSO | CONMIN | IPOPT | NLPQLP | NSGA2 | PSQP | ParOpt | SLSQP | SNOPT

    def __init__(
        self,
        name: str,
        objective: Callable[["Opt", DesignVariables], np.ndarray],
    ):
        self.name = name
        self.objective = objective
        self.constraints_info = []
        self.design_variables_info = []
        self.parameters = {}

    def print(self):
        print(self.optProb)

    def problem_wrapper(self):
        def problem(design_variables: DesignVariables):
            funcs: Dict[str, Union[float, np.ndarray]] = {}

            # Cost function
            funcs[self.objective_name] = self.objective(self, design_variables)

            # Constraint functions
            for constraint_info in self.constraints_info:
                funcs[constraint_info["name"]] = constraint_info["function"](self, design_variables)

            fail = False
            return funcs, fail

        return problem

    def sens_wrapper(self, sens_function: Callable[["Opt", DesignVariables, List], Dict]):
        def sens(design_variables: DesignVariables, func_values):
            return sens_function(self, design_variables, func_values)

        return sens

    def add_constraint_info(self, constraint: ConstraintInfo) -> None:
        self.constraints_info.append(constraint)

    def add_design_variables_info(self, design_variable: DesignVariableInfo) -> None:
        self.design_variables_info.append(design_variable)

    def add_parameters(self, parameters: Parameters) -> None:
        self.parameters = parameters

    def add_optimizer(self, optimizer_type: str, options: Dict) -> None:
        self.optimizer = OPT(optimizer_type, options=options)

    def setup(self):
        self.optProb = Optimization(self.name, self.problem_wrapper())

        # Add design variables
        for design_variable_info in self.design_variables_info:
            self.optProb.addVarGroup(
                design_variable_info["name"],
                design_variable_info["n_params"],
                design_variable_info["type"],
                lower=design_variable_info["lower"],
                upper=design_variable_info["upper"],
                value=design_variable_info["initial_value"],
                scale=design_variable_info.get("scale", 1.0),  # Use 1.0 as default if "scale" is not present
            )

        # Add constraints
        for constraint_info in self.constraints_info:
            self.optProb.addConGroup(
                constraint_info["name"],
                constraint_info["n_constraints"],
                lower=constraint_info["lower"],
                upper=constraint_info["upper"],
                scale=constraint_info.get("scale", 1.0),  # Use 1.0 as default if "scale" is not present
                wrt=constraint_info.get("wrt", None),
                jac=constraint_info.get("jac", None),
            )

        # Objective
        self.optProb.addObj(self.objective_name)

    def optimize(self, sens: Union[str, Callable[["Opt", DesignVariables, List], Dict]], sensStep=None):
        if isinstance(sens, str):
            sol = self.optimizer(self.optProb, sens=sens, sensStep=sensStep)
        else:
            sol = self.optimizer(self.optProb, sens=self.sens_wrapper(sens))
        return sol
