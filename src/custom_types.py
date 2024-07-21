import numpy as np
from typing import TypedDict, Dict, List, Union, Optional, Callable, Any, NotRequired
from typing_extensions import NotRequired
from numpy.typing import NDArray
import scipy.sparse as sp
# from opt import Opt


"""
Plot types: PlotData its dependencies
"""


class ArrayData(TypedDict):
    x: Union[List[int], List[float], NDArray[np.int_], NDArray[np.float_]]
    y: Union[List[int], List[float], NDArray[np.int_], NDArray[np.float_]]
    label: Union[str, None]


class RequiredAxesData(TypedDict):
    i: int
    j: int
    arrays_data: List[ArrayData]


class AxesData(RequiredAxesData, total=False):
    title: Optional[str]
    xlabel: Optional[str]
    ylabel: Optional[str]


class PlotData(TypedDict):
    rows: int
    columns: int
    axes_data: List[AxesData]


Parameters = Dict[str, Any]
DesignVariables = Dict[str, np.ndarray]


class DesignVariableInfo(TypedDict):
    name: str
    n_params: int
    type: str
    lower: Union[float, List[float], None]
    upper: Union[float, List[float], None]
    initial_value: float
    scale: NotRequired[float]


class ConstraintInfo(TypedDict):
    name: str
    n_constraints: int
    lower: Union[float, List[float], None]
    upper: Union[float, List[float], None]
    scale: NotRequired[float]
    function: Callable[["Opt", DesignVariables], np.ndarray]
    wrt: NotRequired[List[str]]
    # jac: NotRequired[Dict[str, Union[np.ndarray, sp.spmatrix]]]
    jac: NotRequired[Dict[str, Any]]

