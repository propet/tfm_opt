import numpy as np
from typing import TypedDict, List, Union, Optional
from numpy.typing import NDArray


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
