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


def local_search_callback(where, model: gp.Model):
    # if where == gp.GRB.Callback.MIPSOL:  # Integer solution found.
    #
    #     # Retrieve incumbent solution
    #     IsResourceGet = model.cbGetSolution(IsResource)
    #     DoesBurnGet = model.cbGetSolution(DoesBurn)
    #     IsResourceV = {n: round(IsResourceGet[n]) for n in IsResourceGet}
    #     DoesBurnV = {n: round(DoesBurnGet[n]) for n in DoesBurnGet}
    #
    #
    #     Incumbent = set(  # Set of nodes with a resource
    #         n for n in Nodes for t in ResAtTime if IsResourceV[n, t] > .5)

    pass

