import os
from typing import Callable, Union

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from matplotlib import pyplot as plt

from dataIO import ROOT
from ground import Ground
from track import Track
from train import Train

CONST: dict[str, float] = {
    'epsilon': 10e-5,
    'e_u': 2.7777777777778 * 10e-7,  # e_u: 1 J = e_u x 1 kWh
    'lambda': 0.8,  # lambda: the average price of electricity in RMB per kWh. 按照居民用电差不多0.8元一度
    'eta': 0.85,  # eta: the conversion factor from electricity to kinetic energy. 按照王青元教授开会时候说的，0.85。
}


def add_vao_variables(model: gp.Model, ground: Ground) -> list[gp.Var]:
    S = ground.num_s
    iy_min, iy_max = ground.iy_min, ground.iy_max

    e = model.addVars(range(0, S + 2), vtype=GRB.CONTINUOUS, lb=iy_min, ub=iy_max, name='e')
    pi = model.addVars(range(1, S + 1), vtype=GRB.BINARY, name='pi')
    C = model.addVars(range(1, S + 1), vtype=GRB.CONTINUOUS, name='C')
    C6tn_e = model.addVars(range(1, S + 1), vtype=GRB.CONTINUOUS, name='C^{tn,e}')
    gamma = model.addVars(range(1, 4), range(1, S + 1), vtype=GRB.CONTINUOUS, lb=iy_min, ub=iy_max, name='gamma')
    z1 = model.addVars(range(1, S + 2), vtype=GRB.BINARY, name='z1')
    z2 = model.addVars(range(1, S + 1), vtype=GRB.BINARY, name='z2')
    z3 = model.addVars(range(1, S + 1), vtype=GRB.BINARY, name='z3')
    z4 = model.addVars(range(1, S + 1), vtype=GRB.BINARY, name='z4')
    z5 = model.addVars(range(1, S + 1), vtype=GRB.BINARY, name='z5')

    return [e, pi, C, C6tn_e, gamma, z1, z2, z3, z4, z5]


def add_vao_constraints(model: gp.Model, ground: Ground, variables: dict[str, gp.Var]) -> None:
    e, pi, C, C6tn_e, gamma, z1, z2, z3, z4, z5 = variables.values()

    S = ground.num_s
    ds, de = ground.ds, ground.de
    e6g = ground.e6g
    iy_min, iy_max = ground.iy_min, ground.iy_max
    ht_min, hb_min = ground.ht_min, ground.hb_min
    epsilon = CONST['epsilon']
    sigma = ground.sigma

    #   construction cost related. C_s
    model.addConstrs((
        C[s] / (de * ds) + ground.c_fill * e6g[s] ==
        z1[s] * (ground.c_tn / de - ground.c_cut * e6g[s]) +
        z2[s] * (ground.c_cut + ground.c_fill) * e6g[s] +
        z3[s] * (ground.c_fill - ground.c_bg) * e6g[s] +
        gamma[1, s] * ground.c_cut -
        gamma[2, s] * (ground.c_cut + ground.c_fill) +
        gamma[3, s] * (ground.c_bg - ground.c_fill) +
        ground.c_fill * e[s]
        for s in range(1, S + 1)),
        name="vao-Cs"
    )

    # auxiliary binary variables, z1, z2, z3
    model.addConstrs((e[s] - e6g[s] + ht_min / de <= (iy_max - e6g[s] + ht_min / de) * (1 - z1[s])
                      for s in range(1, S + 1)), name='vao-z1_linear1')
    model.addConstrs(
        (e[s] - e6g[s] + ht_min / de >= epsilon + (iy_min - e6g[s] + ht_min / de - epsilon) * z1[s]
         for s in range(1, S + 1)), name='vao-z1_linear2')
    model.addConstrs((e[s] - e6g[s] <= (iy_max - e6g[s]) * (1 - z2[s])
                      for s in range(1, S + 1)), name='vao-z2_linear1')
    model.addConstrs((e[s] - e6g[s] >= epsilon + (iy_min - e6g[s] - epsilon) * z2[s]
                      for s in range(1, S + 1)), name='vao-z2_linear2')
    model.addConstrs((e[s] - e6g[s] - hb_min / de <= (iy_max - e6g[s] - hb_min / de) * z3[s] - epsilon
                      for s in range(1, S + 1)), name='vao-z3_linear1')
    model.addConstrs((e[s] - e6g[s] - hb_min / de >= (iy_min - e6g[s] - hb_min / de) * (1 - z3[s])
                      for s in range(1, S + 1)), name='vao-z3_linear2')

    # auxiliary continuous variables, gamma 123
    for i, z in zip([1, 2, 3], [z1, z2, z3]):
        model.addConstrs((gamma[i, s] <= iy_max * z[s] for s in range(1, S + 1)), name=f"vao-gamma{i}_linear1")
        model.addConstrs((gamma[i, s] >= iy_min * z[s] for s in range(1, S + 1)), name=f'vao-gamma{i}_linear2')
        model.addConstrs((gamma[i, s] <= e[s] - iy_min * (1 - z[s]) for s in range(1, S + 1)),
                         name=f'vao-gamma{i}_linear3')
        model.addConstrs((gamma[i, s] >= e[s] - iy_max * (1 - z[s]) for s in range(1, S + 1)),
                         name=f'vao-gamma{i}_linear4')

    # tunnel entrance cost
    model.addConstrs((C6tn_e[s] == ground.c6e_tn * (z4[s] - z5[s]) for s in range(1, S + 1)), name='vao-Ctne')
    # tunnel entrance cost linearize constraints
    model.addConstrs((z4[s] >= z1[s] for s in range(1, S + 1)), name='vao-z4_linear1')
    model.addConstrs((z4[s] >= z1[s + 1] for s in range(1, S + 1)), name='vao-z4_linear2')
    model.addConstrs((z5[s] <= z1[s] for s in range(1, S + 1)), name='vao-z5_linear1')
    model.addConstrs((z5[s] <= z1[s + 1] for s in range(1, S + 1)), name='vao-z5_linear2')

    # VPI points are chosen from potential VPI locations
    model.addConstrs((pi[s] <= ground.potential_vpi[s] for s in range(1, S + 1)), name='vao-pi')

    # elevation diff between two consecutive intervals
    model.addConstrs(
        (e[s + 1] + e[s - 1] - 2 * e[s] >= - ground.di_max * ds / de * pi[s] for s in range(1, S + 1)),
        name='vao-ele_diff1')
    model.addConstrs(
        (e[s + 1] + e[s - 1] - 2 * e[s] <= ground.di_max * ds / de * pi[s] for s in range(1, S + 1)),
        name='vao-ele_diff2')

    # max gradient range
    model.addConstrs((e[s + 1] - e[s] >= - ground.i_max * ds / de for s in range(0, S + 1)),
                     name='vao-grad_rg1')
    model.addConstrs((e[s + 1] - e[s] <= ground.i_max * ds / de for s in range(0, S + 1)), name='vao-grad_rg2')

    # minimum slope length
    model.addConstrs(
        (gp.quicksum(pi[s] for s in range(i, i + sigma)) <= 1 for i in range(1, S - sigma + 2)),
        name="vao-min_slope_len")

    # start and end points of the track
    model.addConstr(e[0] == e6g[0], name='vao-start_sta')
    model.addConstr(e[S + 1] == e6g[S + 1], name='vao-end_sta')
    model.addConstr(e[1] == e6g[0], name='vao-start_plat')
    model.addConstr(e[S] == e6g[S + 1], name='vao-end_plat')

    # end point is not a tunnel
    model.addConstr(z1[S + 1] == 0, name='vao-end_z1')
    return


def add_vao_logic_cuts(model: gp.Model, ground: Ground, variables: dict[str, gp.Var]):
    z1, z2, z3 = variables['z1'], variables['z2'], variables['z3']
    model.addConstrs((z2[s] + z3[s] <= 1 for s in range(1, ground.num_s + 1)), name='vao-LC1')
    model.addConstrs((z2[s] >= z1[s] for s in range(1, ground.num_s + 1)), name='vao-LC2')
    return


def add_vao_valid_inequalities(model: gp.Model, ground: Ground, variables: dict[str, gp.Var]):
    e = variables['e']
    lb, ub = ground.get_absolute_e_range()
    le, ue = ground.get_envelope()
    for s in range(ground.num_s + 2):
        # determine the actual range of e
        e_s_min, e_s_max = lb[s], ub[s]
        if (ue[s] < e_s_max) & (ue[s] > e_s_min):
            e_s_max = ue[s]
        if (le[s] < e_s_max) & (le[s] > e_s_min):
            e_s_min = le[s]
        model.addConstr(e[s] >= e_s_min, name=f"vao-VI_lower[{s}]")
        model.addConstr(e[s] <= e_s_max, name=f"vao-VI_upper[{s}]")
    return


def add_tc_variables(
        model: gp.Model,
        ground: Ground,
        train: Train,
        is_uphill_dir: bool = True
) -> list[gp.Var]:
    upd = "u" if is_uphill_dir else "d"
    S = ground.num_s
    I_MAX = ground.i_max
    traction_e: np.array = train.traction_e
    brake_e: np.array = train.brake_e
    max_traction = traction_e[:, 1].max()
    max_brake = brake_e[:, 1].max()
    max_resist = train.max_resist
    max_control = max(max_traction, max_brake)
    max_Ek = traction_e[:, 0].max()

    u = model.addVars(range(1, S + 1), vtype=GRB.CONTINUOUS, lb=-max_brake, ub=max_traction, name=f"u^{upd}")
    kappa = model.addVars(range(1, S + 1), vtype=GRB.CONTINUOUS, ub=max_control, name=f"kappa^{upd}")
    E_k = model.addVars(range(1, S + 2), vtype=GRB.CONTINUOUS, lb=0, ub=max_Ek, name=f"E_k^{upd}")
    f_tra = model.addVars(range(1, S + 1), vtype=GRB.CONTINUOUS, ub=max_traction, name=f"f_tra^{upd}")
    f_bra = model.addVars(range(1, S + 1), vtype=GRB.CONTINUOUS, ub=max_brake, name=f"f_bra^{upd}")
    omega_0 = model.addVars(range(1, S + 1), vtype=GRB.CONTINUOUS, lb=0, ub=max_resist, name=f'omega_0^{upd}')
    omega_i = model.addVars(range(1, S + 1), vtype=GRB.CONTINUOUS, lb=-I_MAX * 1000, ub=I_MAX * 1000,
                            name=f'omega_i^{upd}')
    omega_r = model.addVars(range(1, S + 1), vtype=GRB.CONTINUOUS, name=f'omega_r^{upd}')
    omega_tn = model.addVars(range(1, S + 1), vtype=GRB.CONTINUOUS, lb=0, name=f'omega_tn^{upd}')
    c = model.addVars(range(1, S + 1), vtype=GRB.CONTINUOUS, lb=-100, name=f'c^{upd}')
    t = model.addVars(range(1, S + 1), vtype=GRB.CONTINUOUS, lb=0, name=f't^{upd}')
    f_pwa_v = model.addVars(range(2, S + 1), vtype=GRB.CONTINUOUS, name=f'f_PWA_v^{upd}')

    return [u, kappa, E_k, f_tra, f_bra, omega_0, omega_i, omega_r, omega_tn, c, t, f_pwa_v]


def add_tc_constraints(model: gp.Model,
                       ground: Ground,
                       train: Train,
                       variables: dict[str, gp.Var],
                       vao_variables: dict[str, gp.Var | np.ndarray],
                       is_uphill_dir: bool = True
                       ):
    S = ground.num_s
    u, kappa, E_k, f_tra, f_bra, omega_0, omega_i, omega_r, omega_tn, c, t, f_pwa_v = variables.values()
    e, z1 = vao_variables['e'], vao_variables['z1']  # could be either parameters or variables
    upd = "u" if is_uphill_dir else "d"

    for i in range(1, S + 1):
        #   absolute value of control variables, to calculate the energy
        model.addGenConstrAbs(kappa[i], u[i], name=f"tc-kappa^{upd}[{i}]")
        #   maximum traction PWA constraints
        model.addGenConstrPWL(E_k[i], f_tra[i], train.traction_e[:, 0].tolist(), train.traction_e[:, 1].tolist(),
                              name=f'tc-pwl_tra^{upd}[{i}]')
        #   maximum brake PWA constraints
        model.addGenConstrPWL(E_k[i], f_bra[i], train.brake_e[:, 0].tolist(), train.brake_e[:, 1].tolist(),
                              name=f'tc-pwl_bra^{upd}[{i}]')

    #       unit basic resistance affine using Strahl formula
    model.addConstrs(
        (omega_0[i] == train.r0_strahl + train.r2_strahl * E_k[i] for i in range(1, S + 1)),
        name=f'tc-w0^{upd}')

    #       unit control force range
    model.addConstrs((u[i] >= -f_bra[i] for i in range(1, S + 1)), name=f'tc-u_lb^{upd}')
    model.addConstrs((u[i] <= f_tra[i] for i in range(1, S + 1)), name=f'tc-u_ub^{upd}')

    #       grad resist
    _grad_pr = ground.de / ground.ds * 1000
    if is_uphill_dir:
        model.addConstrs((omega_i[i] == _grad_pr * (e[i] - e[i + 1]) for i in range(1, S + 1)), name=f'tc-wi^{upd}')
    else:
        model.addConstrs((omega_i[i] == _grad_pr * (e[i + 1] - e[i]) for i in range(1, S + 1)), name=f'tc-wi^{upd}')

    #       curve resist
    model.addConstrs((omega_r[i] == ground.curve_resist[i, 1] for i in range(1, S + 1)), name=f'tc-wr^{upd}')

    #       tunnel resist
    if not isinstance(z1, gp.tupledict):  # z1 is considered a constant parameter in this case. (VAO input)
        model.addConstrs(
            (omega_tn[i] == train.r2 * train.r_tn * E_k[i] * z1[i] for i in range(1, S + 1)), name=f'tc-wtn^{upd}')
    else:
        # linearize
        phi = model.addVars(range(1, S + 1), vtype=GRB.CONTINUOUS, lb=0, name=f'phi^{upd}')
        model.addConstrs((omega_tn[i] == train.r2 * train.r_tn * phi[i] for i in range(1, S + 1)),
                         name=f'tc-wtn-1^{upd}')
        model.addConstrs((phi[i] <= ground.ek_lim[i, 1] * z1[i] for i in range(1, S + 1)), name=f'tc-wtn-2^{upd}')
        # model.addConstrs((phi[i] >= 0 for i in range(1, S + 1)), name=f'cs-34-3-{upd}')
        model.addConstrs((phi[i] <= E_k[i] for i in range(1, S + 1)), name=f'tc-wtn-4^{upd}')
        model.addConstrs((phi[i] >= E_k[i] - ground.ek_lim[i, 1] * (1 - z1[i]) for i in range(1, S + 1)),
                         name=f'tc-wtn-5^{upd}')

    #    unit joint force
    model.addConstrs(
        (c[i] == u[i] - omega_0[i] - omega_r[i] - omega_i[i] - omega_tn[i] for i in range(1, S + 1)),
        name=f'tc-c^{upd}')

    #    train motion
    _multiplier = 2 * ground.ds * 3.6 ** 2 / ((1 + train.rho) * 102)
    _multiplier2 = 1 if is_uphill_dir else -1
    model.addConstrs((E_k[i] - E_k[i + 1] == _multiplier2 * _multiplier * c[i] for i in range(1, S + 1)),
                     name=f'tc-train_motion^{upd}')

    #     piecewise linear function to affine the f_t
    for i in range(2, S + 1):
        model.addGenConstrPWL(E_k[i], f_pwa_v[i], train.PWL_EK, train.PWL_F_EK,
                              name=f"tc_pwl_t^{upd}[{i}]")

    #    running time
    model.addConstr(t[1] == 2 * ground.ds * f_pwa_v[2], name=f"tc_t1^{upd}")
    model.addConstr(t[S] == 2 * ground.ds * f_pwa_v[S], name=f"tc_tS^{upd}")
    model.addConstrs((t[i] == ground.ds / 2 * (f_pwa_v[i] + f_pwa_v[i + 1]) for i in range(2, S)), name=f"tc_t^{upd}")

    #    speed limit
    model.addConstrs((E_k[i] == 0 for i in [1, S + 1]), name=f"tc_Ek^{upd}")
    model.addConstrs((E_k[i] <= ground.ek_lim[i, 1] for i in range(2, S + 1)), name=f'tc_Ek^{upd}')

    return


def add_tc_constraints_max_time(
        model: gp.Model,
        ground: Ground,
        train: Train,
        variables: dict[str, gp.Var],
        vao_variables=None,
        is_uphill_dir: bool = True):
    # section running time restriction
    t_max = ground.time[train.name]["TU_MAX"] if is_uphill_dir else ground.time[train.name]["TD_MAX"]
    upd = "u" if is_uphill_dir else "d"
    model.addConstr(variables['t'].sum() <= t_max, name=f"tc_T^{upd}")
    return


def add_tc_valid_inequalities(
        model: gp.Model,
        ground: Ground,
        train: Train,
        variables: dict[str, gp.Var],
        vao_variables=None,
        is_uphill_dir: bool = True):
    # section running time restriction
    t_max = ground.time[train.name]["TU_MAX"] if is_uphill_dir else ground.time[train.name]["TD_MAX"]
    upd = "u" if is_uphill_dir else "d"
    model.addConstr(variables['t'].sum() >= t_max * 0.98, name=f"tc-VI_{upd}")
    return


class ModelNotSolvedException(Exception):
    pass


class OptimizationModel:
    """
    input name, get self.name = f"{ground.name}__{train_name}__{name}"
        e.g. gd_gaoyan__CRH380AL__vao_LC_VI, gd_gaoyan__CRH380AL__eetc_tcVI, gd2__HXD2__eetc-vao_LC_VI_tcVI
    """

    def __init__(self, name: str, ground: Ground, train: Train = None):
        train_name = "" if train is None else train.name
        self.name: str = f"{ground.name}__{train_name}__{name}"  # vao, eetc, sotc, eetc-vao, ...
        self.model: gp.Model = gp.Model(name)
        self.model._variable_groups: dict = {}  # for callback use.
        self.ground: Ground = ground
        self.train: Train = train
        self.directory = f"{ROOT}\\Cases\\{self.name}"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.log_file_path = f"{self.directory}\\{self.name}.log"

        # dictionaries of dictionaries. {var_group_name: {name: variable, ...}, ...}
        self.variable_groups = {}  # vao, tc_u, tc_d
        self._vao_variable_names = ['e', 'pi', 'C', 'C6tn_e', 'gamma', 'z1', 'z2', 'z3', 'z4', 'z5']
        self._tc_variable_names = ["u", "kappa", "E_k", "f_tra", "f_bra", "omega_0", "omega_i", "omega_r", "omega_tn",
                                   "c", "t", "f_pwa_v"]
        return

    def add_vars(self, variable_group_name: str = None, variable_function: Callable = None, **kwargs):
        """add variables to the model, and save variable in self.variable_groups (dict)"""
        self.variable_groups[variable_group_name] = {}
        variables = variable_function(model=self.model, ground=self.ground, **kwargs)
        if variable_group_name not in ['vao', 'tc_u', 'tc_d']:
            raise KeyError("variable_group_name must be in 'vao', 'tc_u', 'tc_d'. ")
        variable_names = self._vao_variable_names if variable_group_name == "vao" else self._tc_variable_names
        for var_name, variable in zip(variable_names, variables):
            self.variable_groups[variable_group_name][var_name] = variable
        self.model._variable_groups[variable_group_name] = self.variable_groups[variable_group_name]
        return

    def add_variables(self, **kwargs):
        pass

    def add_constraints(self, **kwargs):
        pass

    def set_objectives(self, gurobi_linear_expression: gp.LinExpr):
        self.model.setObjective(gurobi_linear_expression)
        return

    def set_warm_start(self, warm_start_data: dict[str, dict[str, gp.Var | float | int]]):
        if warm_start_data is None:
            return
        for grp, grp_variables in warm_start_data.items():
            for var_name, variables in grp_variables.items():
                for var_key in variables.keys():
                    to_assign = variables[var_key]
                    if isinstance(to_assign, gp.Var):
                        to_assign = to_assign.X
                    elif isinstance(to_assign, (int, float)):
                        pass
                    else:
                        raise ValueError("warm_start_data should only contain integers, floats, or gp.Var.")
                    self.variable_groups[grp][var_name][var_key].Start = to_assign

                    # # add absolute range for debugging
                    # self.model.addRange(
                    #     self.variable_groups[grp][var_name][var_key], to_assign, to_assign,
                    #     name=f"BOUND{grp}_{var_name}[{var_key}]")

        return

    def set_parameters(self, save_on: bool = True, **kwargs):
        # set log file parameter
        if save_on:
            self.model.setParam("LogFile", self.log_file_path)
        else:
            self.model.setParam("OutputFlag", 0)
        # set other user-defined parameters
        for key, value in kwargs.items():
            self.model.setParam(key, value)
        return

    def opt(self, callback_function: Callable = None, save_on: bool = True, **kwargs):
        self.set_parameters(save_on=save_on, **kwargs)
        if callback_function is None:
            self.model.optimize()
        else:
            self.model.optimize(callback_function)
        if save_on:
            try:
                self.save_optimization_info()
                self.plot_results()
            except Exception as e:
                print(e)
        return

    def get_brief_results_decorated_txt(self):
        txt = ""
        txt += '>>' * 20 + f' {self.name} ' + '<<' * 20 + '\n'

        if "vao" in self.variable_groups.keys():
            txt += '>>' * 10 + f' vao results ' + '<<' * 10 + '\n'
            for _show_variable in ['e', 'pi', 'z1']:
                _show_variable_info = self.model.getAttr('X', self.variable_groups['vao'][_show_variable])
                txt += f"{_show_variable}:\n{_show_variable_info}\n"
            txt += f"total tunnel number: \t{self.variable_groups['vao']['z1'].sum().getValue()}\n"
            Cc = self.variable_groups['vao']['C'].sum().getValue() + \
                 self.variable_groups['vao']['C6tn_e'].sum().getValue()
            txt += f"constructionCost:\t{Cc}\n"
            txt += '>>' * 10 + f' vao results ' + '<<' * 10 + '\n'

        if len(self.variable_groups) > 1:
            operation_cost_total = 0
            _multiplier = CONST['e_u'] * self.train.M_t * self.train.g * self.ground.ds / CONST['eta']
            for direction in ['_u', '_d']:
                tc_dir = f"tc{direction}"
                NumTrainsRunning_dir = self.ground.N_tr_up[
                    self.train.name] if direction == '_u' else self.ground.N_tr_down[self.train.name]
                if tc_dir in self.variable_groups.keys():
                    txt += '>>' * 10 + f' {tc_dir} results ' + '<<' * 10 + '\n'
                    sectionRunningTime = self.variable_groups[tc_dir]['t'].sum().getValue()
                    txt += f"T{direction}:\t{sectionRunningTime}\n"
                    tractionCostPerTrain = _multiplier * self.variable_groups[tc_dir]['kappa'].sum().getValue()
                    auxiliaryCostPerTrain = self.train.mu * sectionRunningTime
                    txt += f"tractionCostPerTrain{direction}: \t{tractionCostPerTrain}\n"
                    txt += f"auxiliaryCostPerTrain{direction}: \t{auxiliaryCostPerTrain}\n"
                    txt += f"No.TrainsRunning{direction}: \t{NumTrainsRunning_dir}\n"
                    operation_cost_total += NumTrainsRunning_dir * CONST['lambda'] \
                                            * (tractionCostPerTrain + auxiliaryCostPerTrain)
                    txt += '>>' * 10 + f' {tc_dir} results ' + '<<' * 10 + '\n'
            txt += f"operationCostTotal: \t{operation_cost_total}\n"
        txt += '>>' * 20 + f' {self.name} ' + '<<' * 20 + '\n'
        return txt

    def get_detail_results_decorated_txt(self):
        txt = ""
        txt += '>>' * 20 + f' DETAIL RESULT ' + '<<' * 20 + '\n'
        txt += '>>' * 10 + f' VARIABLES ' + '<<' * 10 + '\n'
        for var in self.model.getVars():
            txt += (f"{var.VarName}, VType: {var.VType}, LB: {var.LB}, UB: "
                    f"{var.UB}, ObjCoefficient: {var.Obj}, value: {var.X}\n")
        txt += ">>" * 10 + f' CONSTRAINTS ' + '<<' * 10 + '\n'
        for constr in self.model.getConstrs():
            expr = self.model.getRow(constr)  # 获取约束的左侧表达式
            lhs_value = 0  # 初始化左侧表达式的值

            # 遍历表达式中的所有项
            for i in range(expr.size()):
                var = expr.getVar(i)
                coeff = expr.getCoeff(i)
                lhs_value += var.X * coeff

            txt += (f"{constr.ConstrName} with SLACK {constr.Slack:.4f}: "
                    f"{self.model.getRow(constr)} = {lhs_value} ||{constr.Sense}=|| {constr.rhs}\n")
        txt += '>>' * 20 + f' DETAIL RESULT ' + '<<' * 20 + '\n'
        return txt

    def save_optimization_info(self, print_out: bool = True):
        save_name = os.path.join(self.directory, f"{self.name}")
        self.model.write(f"{save_name}.mps")
        self.model.write(f"{save_name}.lp")
        if self.model.Status == GRB.INFEASIBLE:
            self.model.computeIIS()
            self.model.write(f"{save_name}.ilp")
        else:
            self.model.write(f"{save_name}.json")  # solution file

            # append into log files
            with open(self.log_file_path, "a") as f:
                brief_res = self.get_brief_results_decorated_txt()
                if print_out:
                    print(brief_res)
                f.write(brief_res)
                f.write("\n\n\n\n")
                f.write(self.get_detail_results_decorated_txt())

        return

    def plot_results(self):
        """Plot the results of either vao or tc or both"""
        return


class VAO(OptimizationModel):
    def __init__(self, ground: Ground, LC_ON: bool = True, VI_on: bool = True, plot_ground: bool = True):
        name = "vao"
        if LC_ON:
            name += "_LC"
        if VI_on:
            name += "_VI"
        super().__init__(name=name, ground=ground, train=None)
        self.add_variables()

        self.add_constraints(constraint_func=add_vao_constraints)
        if LC_ON:
            self.add_constraints(constraint_func=add_vao_logic_cuts)
        if VI_on:
            self.add_constraints(constraint_func=add_vao_valid_inequalities)

        obj_exp = gp.LinExpr()
        obj_exp += gp.quicksum(
            self.variable_groups['vao']['C'][s] +
            self.variable_groups['vao']['C6tn_e'][s]
            for s in range(1, self.ground.num_s + 1)
        )
        self.set_objectives(gurobi_linear_expression=obj_exp)
        if plot_ground:
            self.ground.plot_ground_with_envelope().savefig(
                f"{self.directory}\\{self.name}-ground_profile.pdf", dpi=600
            )
        return

    def add_variables(self):
        super().add_vars(
            variable_group_name="vao", variable_function=add_vao_variables)
        return

    def add_constraints(self, constraint_func: Callable):
        constraint_func(
            model=self.model, ground=self.ground,
            variables=self.variable_groups["vao"])
        return

    def optimize(self, callback_function: Callable = None, save_on: bool = True,
                 IntegralityFocus=1, NumericFocus=1, Cuts=2,
                 IntFeasTol=1e-07, MIPGap=0, TimeLimit=3600 * 2, **kwargs):
        super().opt(
            callback_function=callback_function,
            save_on=save_on,
            IntegralityFocus=IntegralityFocus,
            NumericFocus=NumericFocus,
            Cuts=Cuts,
            IntFeasTol=IntFeasTol,
            MIPGap=MIPGap,
            TimeLimit=TimeLimit,
            **kwargs)
        return

    def get_track(self) -> Track:
        if (self.model.Status != GRB.Status.OPTIMAL) & (self.model.Status != GRB.Status.TIME_LIMIT):
            raise Exception("Model is not solved!")
        e = np.array([_.X for _ in self.variable_groups['vao']['e'].values()])
        pi = np.array([_.X for _ in self.variable_groups['vao']['pi'].values()])
        z1 = np.array([_.X for _ in self.variable_groups['vao']['z1'].values()])
        track = Track(e=e, pi=pi, z1=z1, ground=self.ground)
        return track

    def plot_results(self):
        track = self.get_track()
        fig = track.plot_ground_track()
        fig.savefig(self.directory + f"\\{self.name}-track_profile.pdf", dpi=600)
        return


class TC(OptimizationModel):
    def __init__(self, train: Train, track: Track, is_ee: bool = True, tcVI_on: bool = True,
                 warm_start_data: dict[str, dict[str, gp.Var | float | int]] = None):
        name = "eetc" if is_ee else "sotc"
        tcVI_on = tcVI_on if is_ee else False
        if tcVI_on:
            name += "_tcVI"
        if warm_start_data is not None:
            name += "_WS"
        super().__init__(name=name, ground=track.ground, train=train)
        self.add_variables()

        vao_variables: dict[str, np.ndarray] = {"e": track.e, "z1": track.z1}
        self.add_constraints(
            constraint_func=add_tc_constraints, vao_variables=vao_variables)
        if is_ee:
            self.add_constraints(constraint_func=add_tc_constraints_max_time)
        if tcVI_on:
            self.add_constraints(
                constraint_func=add_tc_valid_inequalities, vao_variables=vao_variables)

        obj_exp = gp.LinExpr()
        _pr_obj = CONST['e_u'] * self.train.M_t * self.train.g * self.ground.ds / CONST['eta']
        n_tr_up = self.ground.N_tr_up[self.train.name]
        n_tr_down = self.ground.N_tr_down[self.train.name]
        # auxiliary energy cost
        obj_exp += gp.quicksum(self.variable_groups['tc_u']['t']) * self.train.mu * n_tr_up
        obj_exp += gp.quicksum(self.variable_groups['tc_d']['t']) * self.train.mu * n_tr_down
        # traction energy cost
        if is_ee:
            obj_exp += gp.quicksum(self.variable_groups['tc_u']['kappa']) * _pr_obj * n_tr_up
            obj_exp += gp.quicksum(self.variable_groups['tc_d']['kappa']) * _pr_obj * n_tr_down
        obj_exp = obj_exp * CONST['lambda']
        self.model.setObjective(obj_exp, GRB.MINIMIZE)
        self.set_warm_start(warm_start_data)
        pass

    def add_variables(self):
        super().add_vars(
            variable_group_name="tc_u",
            variable_function=add_tc_variables,
            train=self.train,
            is_uphill_dir=True)
        super().add_vars(
            variable_group_name="tc_d",
            variable_function=add_tc_variables,
            train=self.train,
            is_uphill_dir=False)
        return

    def add_constraints(self, constraint_func: Callable, vao_variables: dict[str, gp.Var | np.ndarray] = None):
        """adding constraints in both uphill and downhill directions"""
        constraint_func(
            model=self.model,
            ground=self.ground,
            train=self.train,
            variables=self.variable_groups["tc_u"],
            vao_variables=vao_variables,
            is_uphill_dir=True
        )
        constraint_func(
            model=self.model,
            ground=self.ground,
            train=self.train,
            variables=self.variable_groups["tc_d"],
            vao_variables=vao_variables,
            is_uphill_dir=False
        )
        return

    def optimize(self, callback_function: Callable = None, save_on: bool = True,
                 IntegralityFocus=1, NumericFocus=1, Cuts=2,
                 IntFeasTol=1e-07, MIPGap=0, TimeLimit=3600 * 2, **kwargs):
        super().opt(
            callback_function=callback_function,
            save_on=save_on,
            IntegralityFocus=IntegralityFocus,
            NumericFocus=NumericFocus,
            Cuts=Cuts,
            IntFeasTol=IntFeasTol,
            MIPGap=MIPGap,
            TimeLimit=TimeLimit,
            **kwargs)
        return

    def plot_results(self):
        """Plot the results of either vao or tc or both"""
        S = self.ground.num_s
        distances = (np.arange(0, S + 1) * self.ground.ds +
                     self.ground.bottom_left_corner_coordinates[0])
        plt.figure(figsize=(8, 4))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212, sharex=ax1)
        for dr in ['_u', '_d']:
            tip_shape = 'x' if dr == "_u" else "o"
            v_arr = np.sqrt(
                [self.variable_groups[f"tc{dr}"]['E_k'][i].X for i in range(1, S + 2)])
            ax1.plot(distances, v_arr, f"{tip_shape}-", label=f"v{dr}")
            u_arr = np.array(
                [self.variable_groups[f'tc{dr}']['u'][i].X for i in range(1, S + 1)])
            ax2.plot(distances[:-1], u_arr, f"{tip_shape}-", label=f"u{dr}")
        ax1.legend(fontsize="small")
        ax1.set_ylabel("Velocity (km/h)", fontsize="small")
        ax2.legend(fontsize="small")
        ax2.set_xlabel("Horizontal location (m)", fontsize="small")
        ax2.set_ylabel("Unit control force (N/kN)", fontsize="small")
        plt.tight_layout()
        plt.savefig(self.directory + f"\\{self.name}-speed_profile.pdf", dpi=600)
        return


class EETC_VAO(OptimizationModel):
    def __init__(self,
                 ground: Ground, train: Train,
                 LC_on: bool = True, VI_on: bool = True,
                 tcVI_on: bool = True,
                 warm_start_data: dict[str, dict[str, gp.Var | float | int]] = None):
        name = f"eetc-vao"
        if LC_on:
            name += "_LC"
        if VI_on:
            name += "_VI"
        if tcVI_on:
            name += "_tcVI"
        if warm_start_data is not None:
            name += "_WS"
        super().__init__(name, ground, train)

        self.add_variables()

        self.add_constraints(constraint_func=add_vao_constraints, variable_group_name="vao")
        self.add_constraints(constraint_func=add_tc_constraints, variable_group_name="tc")
        self.add_constraints(constraint_func=add_tc_constraints_max_time, variable_group_name="tc")

        if LC_on:
            self.add_constraints(constraint_func=add_vao_logic_cuts, variable_group_name="vao")
        if VI_on:
            self.add_constraints(constraint_func=add_vao_valid_inequalities, variable_group_name="vao")
        if tcVI_on:
            self.add_constraints(constraint_func=add_tc_valid_inequalities, variable_group_name="tc")

        obj_exp = gp.LinExpr()
        obj_exp += gp.quicksum(
            self.variable_groups['vao']['C'][s] + self.variable_groups['vao']['C6tn_e'][s]
            for s in range(1, self.ground.num_s + 1)
        )
        _pr_obj = CONST['e_u'] * self.train.M_t * self.train.g * self.ground.ds / CONST['eta']
        n_tr_up = self.ground.N_tr_up[self.train.name]
        n_tr_down = self.ground.N_tr_down[self.train.name]
        # auxiliary energy cost
        obj_exp += gp.quicksum(self.variable_groups['tc_u']['t']) * self.train.mu * n_tr_up
        obj_exp += gp.quicksum(self.variable_groups['tc_d']['t']) * self.train.mu * n_tr_down
        # traction energy cost
        obj_exp += gp.quicksum(self.variable_groups['tc_u']['kappa']) * _pr_obj * n_tr_up
        obj_exp += gp.quicksum(self.variable_groups['tc_d']['kappa']) * _pr_obj * n_tr_down
        obj_exp = obj_exp * CONST['lambda']
        self.model.setObjective(obj_exp, GRB.MINIMIZE)

        self.set_warm_start(warm_start_data)
        return

    def add_variables(self):
        super().add_vars(variable_group_name="vao", variable_function=add_vao_variables)
        super().add_vars(
            variable_group_name="tc_u",
            variable_function=add_tc_variables,
            train=self.train,
            is_uphill_dir=True)
        super().add_vars(
            variable_group_name="tc_d",
            variable_function=add_tc_variables,
            train=self.train,
            is_uphill_dir=False)
        return

    def add_constraints(self, constraint_func: Callable, variable_group_name: str = "tc_u"):
        if variable_group_name == "vao":
            constraint_func(model=self.model, ground=self.ground, variables=self.variable_groups["vao"])
        else:
            constraint_func(
                model=self.model,
                ground=self.ground,
                train=self.train,
                variables=self.variable_groups["tc_u"],
                vao_variables=self.variable_groups["vao"],
                is_uphill_dir=True
            )
            constraint_func(
                model=self.model,
                ground=self.ground,
                train=self.train,
                variables=self.variable_groups["tc_d"],
                vao_variables=self.variable_groups["vao"],
                is_uphill_dir=False
            )
        pass

    @staticmethod
    def callback(model, where):
        if where == gp.GRB.Callback.MIPSOL:  # Integer solution found.
            e = model.cbGetSolution(model._variable_groups["vao"]["e"])
            pi = model.cbGetSolution(model._variable_groups["vao"]['pi'])
            z1 = model.cbGetSolution(model._variable_groups["vao"]['z1'])

        # e = model.model._variable_groups["vao"]
        ...

    def optimize(self, callback_function: Callable = None, save_on: bool = True,
                 IntegralityFocus=1, NumericFocus=1, Cuts=2,
                 IntFeasTol=1e-07, MIPGap=0, TimeLimit=3600 * 2, **kwargs):
        super().opt(
            callback_function=callback_function,
            save_on=save_on,
            IntegralityFocus=IntegralityFocus,
            NumericFocus=NumericFocus,
            Cuts=Cuts,
            IntFeasTol=IntFeasTol,
            MIPGap=MIPGap,
            TimeLimit=TimeLimit,
            **kwargs)
        return

    def get_track(self) -> Track:
        if (self.model.Status != GRB.Status.OPTIMAL) & (self.model.Status != GRB.Status.TIME_LIMIT):
            raise ModelNotSolvedException("Model is not solved!")
        e = np.array([_.X for _ in self.variable_groups['vao']['e'].values()])
        pi = np.array([_.X for _ in self.variable_groups['vao']['pi'].values()])
        z1 = np.array([_.X for _ in self.variable_groups['vao']['z1'].values()])
        track = Track(e=e, pi=pi, z1=z1, ground=self.ground)
        return track

    def plot_results(self):
        # plot track profile
        try:
            track = self.get_track()
        except ModelNotSolvedException as e:
            print(e)
            return
        fig = track.plot_ground_track()
        fig.savefig(self.directory + f"\\{self.name}-track_profile.pdf", dpi=600)

        # plot velocity and control curves
        S = self.ground.num_s
        distances = (np.arange(0, S + 1) * self.ground.ds +
                     self.ground.bottom_left_corner_coordinates[0])
        plt.figure(figsize=(8, 4))
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212, sharex=ax1)
        for dr in ['_u', '_d']:
            tip_shape = 'x' if dr == "_u" else "o"
            v_arr = np.sqrt(
                [self.variable_groups[f"tc{dr}"]['E_k'][i].X for i in range(1, S + 2)])
            ax1.plot(distances, v_arr, f"{tip_shape}-", label=f"v{dr}")
            u_arr = np.array(
                [self.variable_groups[f'tc{dr}']['u'][i].X for i in range(1, S + 1)])
            ax2.plot(distances[:-1], u_arr, f"{tip_shape}-", label=f"u{dr}")
        ax1.legend(fontsize="small")
        ax1.set_ylabel("Velocity (km/h)", fontsize="small")
        ax2.legend(fontsize="small")
        ax2.set_xlabel("Horizontal location (m)", fontsize="small")
        ax2.set_ylabel("Unit control force (N/kN)", fontsize="small")
        plt.tight_layout()
        plt.savefig(self.directory + f"\\{self.name}-speed_profile.pdf", dpi=600)

        return


def get_all_ground_sotc(train: Train):
    for i in range(1, 7):
        gd = Ground(name=f"gd{i}")
        vao = VAO(ground=gd, VI_on=True, LC_ON=True, plot_ground=False)
        vao.optimize(save_on=False)
        track = vao.get_track()
        sotc = TC(train=train, track=track, is_ee=False)
        sotc.optimize(save_on=True)
    return


def get_all_train_sotc(ground: Ground):
    vao = VAO(ground=ground, VI_on=True, LC_ON=True, plot_ground=True)
    vao.optimize(save_on=True)
    track = vao.get_track()
    for train in ["CRH380AL", "HXD1D", "HXD2"]:
        tr = Train(name=train)
        sotc = TC(train=tr, track=track, is_ee=False)
        sotc.optimize(save_on=True)
    return


def one_case_routine(ground: Ground, train: Train, warm_start_case: str = "sotc"):
    if warm_start_case == "":
        ev = EETC_VAO(ground=ground, train=train)
        ev.optimize(save_on=True)
        return
    vao = VAO(ground=ground, VI_on=True, LC_ON=True, plot_ground=True)
    vao.optimize(save_on=True)
    track = vao.get_track()
    if warm_start_case == "sotc":
        tc = TC(train=train, track=track, is_ee=False)
        tc.optimize(save_on=True)
    elif warm_start_case == "eetc":
        tc = TC(train=train, track=track, is_ee=True, tcVI_on=True)
        tc.optimize(save_on=True, TimeLimit=1800)
    else:
        raise ValueError("warm_start_case must be \"sotc\" or \"eetc\".")

    variables_dict: dict = {**vao.variable_groups, **tc.variable_groups}
    ev = EETC_VAO(ground=ground, train=train, warm_start_data=variables_dict)
    ev.optimize(save_on=True)
    return


def main():
    # vao = VAO(ground=Ground("gd2"), VI_on=False, LC_ON=False)
    # vao.optimize()
    # vao2 = VAO(ground=Ground("gd2"), VI_on=True, LC_ON=False)
    # vao2.optimize()
    # vao3 = VAO(ground=Ground("gd2"), VI_on=True, LC_ON=True)
    # vao3.optimize()
    # track = vao.get_track()
    # sotc = TC(train=Train("CRH380AL"), track=track, is_ee=False)
    # sotc.optimize()
    # eetc1 = TC(train=Train("CRH380AL"), track=track, is_ee=True, tcVI_on=False, warm_start_data=sotc.variable_groups)
    # eetc1.optimize()
    # eetc2 = TC(train=Train("CRH380AL"), track=track, is_ee=True, tcVI_on=True)
    # eetc2.optimize()
    # get_all_ground_sotc(Train("HXD2"))
    # gd = Ground(name="gd_gaoyan", type_="real")
    # get_all_train_sotc(ground=gd)
    for train in ["CRH380AL", "HXD1D", "HXD2"]:
        tr = Train(name=train)
        for i in range(1, 7):
            ev = EETC_VAO(ground=Ground(f"gd{i}"), train=tr, LC_on=True, VI_on=True, tcVI_on=False)
            ev.optimize(save_on=True)

    gd = Ground(name="gd_gaoyan", type_="real")
    for train in ["CRH380AL", "HXD1D", "HXD2"]:
        tr = Train(name=train)
        ev = EETC_VAO(ground=gd, train=tr, LC_on=True, VI_on=True, tcVI_on=False)
        ev.optimize(save_on=True)

    pass


if __name__ == '__main__':
    main()
