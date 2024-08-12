# heuristic algorithm to find suboptimal solutions
from typing import Any

import numpy as np
import gurobipy as gp

from dataIO import ROOT
from ground import Ground
from track import Track
from train import Train


def find_track_heuristic(gd: Ground) -> Track:
    lb, ub = gd.get_absolute_e_range()
    le, ue = gd.get_envelope()
    e_min, e_max = np.max([lb, le], axis=0), np.min([ub, ue], axis=0)

    pass


def get_vao_solution_from_track(track: Track) -> dict[str, dict[int | tuple, int | float]]:
    pass


def get_ek_heuristic(trk: Track, tr: Train, max_running_time: float, is_uphill_dir: bool) -> dict[str]:
    pass


def get_eetc_solution_from_ek(ek: np.ndarray, trk: Track, tr: Train) -> dict[str, dict[int | tuple, int | float]]:
    pass


# main function for ANS1
def get_feasible_track_from_relaxed_elevations(relaxed_elevations: np.ndarray) -> Track:
    # the first step is to get current VPI locations
    # (absolute VPI points derived from relaxed elevations without considerations of feasibility)
    vpi_points: np.ndarray = get_vpi_points_from_relaxed_elevations(relaxed_elevations)

    # the second step is to check slope length constraints
    # just delete the nearest VPI point that violate the slope length constraint
    vpi_points: np.ndarray = update_vpi_points_with_slope_len_constraint(vpi_points)

    # the third step is to check gradient constraints
    vpi_points: np.ndarray = update_vpi_points_with_gradient_constraint(vpi_points)

    # transform vpi_points to track
    track = get_track_from_vpi(vpi_points)

    return track


def get_vpi_points_from_relaxed_elevations(relaxed_elevations: np.ndarray) -> np.ndarray:
    vpi_ix: list[int] = []
    vpi_y: list[float] = []
    for ix, (l, m, n) in enumerate(zip(relaxed_elevations[:-2], relaxed_elevations[1:-1], relaxed_elevations[2:])):
        if l + n != 2 * m:
            vpi_ix.append(ix + 1)
            vpi_y.append(m)
    new_vpi = np.array([vpi_ix, vpi_y]).T
    return new_vpi

