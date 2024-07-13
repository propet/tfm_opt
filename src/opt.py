import numpy as np
from pyoptsparse import Optimization
from custom_types import DesignVariableInfo, DesignVariables, Parameters, ConstraintInfo
from typing import Callable, List, Dict, Union


class Opt:
    optProb: Optimization
    name: str
    constraints_info: List[ConstraintInfo]
    design_variables_info: List[DesignVariableInfo]
    parameters: Parameters
    objective: Callable[[DesignVariables, Parameters], np.ndarray]
    objective_name: str = "obj"

    def __init__(
        self,
        name: str,
        objective: Callable[[DesignVariables, Parameters], np.ndarray],
    ):
        self.name = name
        self.objective = objective
        self.constraints_info = []
        self.design_variables_info = []
        self.parameters = {}

    def print(self):
        print(self.optProb)

    def problem_wrapper(self, parameters: Parameters):
        def problem(design_variables: DesignVariables):
            funcs: Dict[str, Union[float, np.ndarray]] = {}

            # Cost function
            funcs[self.objective_name] = self.objective(design_variables, parameters)

            # Constraint functions
            for constraint_info in self.constraints_info:
                funcs[constraint_info["name"]] = constraint_info["function"](design_variables, parameters)

            fail = False
            return funcs, fail

        return problem

    def add_constraint_info(self, constraint: ConstraintInfo) -> None:
        self.constraints_info.append(constraint)

    def add_design_variables_info(self, design_variable: DesignVariableInfo) -> None:
        self.design_variables_info.append(design_variable)

    def add_parameters(self, parameters: Parameters) -> None:
        self.parameters = parameters

    def setup(self):
        self.optProb = Optimization(self.name, self.problem_wrapper(self.parameters))

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
