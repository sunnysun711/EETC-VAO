import os
from typing import Any, Callable

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from matplotlib import pyplot as plt

from dataIO import ROOT
from ground import Ground
from track import Track
from train import Train

CONST: dict[str, float] = {
    'epsilon': 10e-5,
    'e_u': 2.7777777777778 * 10e-7,
    'lambda': 0.8,
    'eta': 0.85,
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


def add_vao_constraints(model: gp.Model, variables: dict[str, gp.Var], ground: Ground) -> None:
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
    # 用之前在遗传算法中得出的e的取值范围的方法，加入那些约束，同时增加判断条件，进而限制一些z和gamma取值。
    # # lower and upper bound of the vao solution, size = S.
    #     # upper bound constrained by: 1) ground max, 2) max_grad from start, 3) max_grad from end.
    #     ub_max_ele = np.ones_like(case.E6G[1:-1]) * case.E6G.max()
    #     ub_imax_start = np.arange(start=0, stop=case.NUM_S) * case.I_MAX * case.DS / case.DE + case.E6G[0]
    #     ub_imax_end = np.arange(start=case.NUM_S - 1, stop=-1, step=-1) * case.I_MAX * case.DS / case.DE + case.E6G[-1]
    #     ub = np.vstack((ub_max_ele, ub_imax_start, ub_imax_end)).min(axis=0)
    #     # lower bound likewise
    #     lb_min_ele = np.ones_like(case.E6G[1:-1]) * case.E6G.min()
    #     lb_imax_start = np.arange(start=0, stop=case.NUM_S) * - case.I_MAX * (case.DS / case.DE) + case.E6G[0]
    #     lb_imax_end = np.arange(start=case.NUM_S - 1, stop=-1, step=-1) * - case.I_MAX * (case.DS / case.DE) + case.E6G[-1]
    #     lb = np.vstack((lb_min_ele, lb_imax_start, lb_imax_end)).max(axis=0)


    pass


def add_tc_constraints(model: gp.Model,
                       ground: Ground,
                       train: Train,
                       variables: dict[str, gp.Var],
                       vao_variables: dict[str, gp.Var | np.array],
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
    model.addConstrs((E_k[i] - E_k[i + 1] == _multiplier2 * _multiplier * c[i] for i in range(1, S + 1)))

    #     piecewise linear function to affine the f_t
    for i in range(2, S + 1):
        model.addGenConstrPWL(E_k[i], f_pwa_v[i], train.PWL_EK.tolist(), train.PWL_F_EK.tolist(),
                              name=f"tc_pwl_t^{upd}[{i}]")

    #    running time
    model.addConstr(t[1] == 2 * ground.ds * f_pwa_v[2], name=f"tc_t1^{upd}")
    model.addConstr(t[S] == 2 * ground.ds * f_pwa_v[S], name=f"tc_tS^{upd}")
    model.addConstrs((t[i] == ground.ds / 2 * (f_pwa_v[i] + f_pwa_v[i + 1]) for i in range(2, S)), name=f"tc_t^{upd}")

    #    speed limit
    model.addConstrs((E_k[i] == 0 for i in [1, S + 1]), name=f"tc_Ek^{upd}")
    model.addConstrs((E_k[i] <= ground.ek_lim[i, 1] for i in range(2, S + 1)), name=f'tc_Ek^{upd}')

    #    T_max
    t_max = ground.TU_MAX if is_uphill_dir else ground.TD_MAX
    model.addConstr(t.sum() <= t_max, name=f"tc_T^{upd}")
    pass


def add_tc_valid_inequalities(model: gp.Model, ground: Ground, variables: dict[str, gp.Var], is_uphill_dir: bool):
    # section running time restriction
    t_max = ground.TU_MAX if is_uphill_dir else ground.TD_MAX
    upd = "u" if is_uphill_dir else "d"
    model.addConstrs(variables['t'].sum() >= t_max - 1, name=f"tc-VI_{upd}")
    return


class OptimizationModel:
    def __init__(self, name: str, ground: Ground, train: Train = None):
        self.name: str = f"{ground.name}__{train.name}__{name}"  # vao, eetc, sotc, eetc-vao, ...
        self.model: gp.Model = gp.Model(name)
        self.ground: Ground = ground
        self.train: Train = train
        self.directory = f"{ROOT}\\Cases\\{self.name}"
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.log_file_path = f"{self.directory}\\{name}.log"

        # dictionaries of dictionaries. {var_group_name: {name: variable, ...}, ...}
        self.variable_groups = {}  # vao, tc_u, tc_d
        self._vao_variable_names = ['e', 'pi', 'C', 'C6tn_e', 'gamma', 'z1', 'z2', 'z3', 'z4', 'z5']
        self._tc_variable_names = ["u", "kappa", "E_k", "f_tra", "f_bra", "omega_0", "omega_i", "omega_r", "omega_tn",
                                   "c", "t", "f_pwa_v"]
        return

    def add_vars(self, variable_group_name: str, add_variables_func: Callable, **kwargs):
        self.variable_groups[variable_group_name] = {}
        variables = add_variables_func(model=self.model, cp=self.cp, **kwargs)
        if variable_group_name not in ['vao', 'tc_u', 'tc_d']:
            raise KeyError("variable_group_name must be in 'vao', 'tc_u', 'tc_d'. ")
        variable_names = self._vao_variable_names if variable_group_name == "vao" else self._tc_variable_names
        for var_name, variable in zip(variable_names, variables):
            self.variable_groups[variable_group_name][var_name] = variable
        return

    def add_constrs(self, variable_group_name: str, add_constrs_func: Callable, **kwargs):
        add_constrs_func(model=self.model, cp=self.cp, variables=self.variable_groups[variable_group_name], **kwargs)
        return

    def set_objectives(self, gurobi_linear_expression: gp.LinExpr):
        self.model.setObjective(gurobi_linear_expression)
        return

    def set_warm_start(self, warm_start_model: dict | "OptimizationModel"):
        if warm_start_model is None:
            return
        if isinstance(warm_start_model, OptimizationModel):
            for grp, variables in warm_start_model.variable_groups.items():
                for var_name, variable in variables.items():
                    for i in variable.keys():
                        self.variable_groups[grp][var_name][i].Start = variable[i].X

                        ## add absolute range for debugging
                        # self.model.addRange(
                        #     self.variable_groups[grp][var_name][i], variable[i].X, variable[i].X,
                        #     name=f"BOUND{grp}_{var_name}[{i}]")

        elif isinstance(warm_start_model, dict):  # result dict
            for grp, variables in warm_start_model.items():
                for var_name, variable in variables.items():
                    for i in variable.keys():
                        self.variable_groups[grp][var_name][i].Start = variable[i]
        else:
            raise ValueError("warm_start_model must be OptimizationModel or dict. ")
        self.model.update()
        return

    def set_parameters(self, **kwargs):
        # set log file parameter
        self.model.setParam("LogFile", self.log_file_path)
        # set other user-defined parameters
        for key, value in kwargs.items():
            self.model.setParam(key, value)
        return

    def optimize(self, callback_function: Callable = None, **kwargs):
        self.set_parameters(**kwargs)
        if callback_function is None:
            self.model.optimize()
        else:
            self.model.optimize(callback_function)
        self.save_optimization_info()
        self.plot_results()
        return

    def get_brief_results_decorated_txt(self):
        txt = ""
        txt += '>>' * 20 + f' {self.case_name} result - {self.name} ' + '<<' * 20 + '\n'

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

        operation_cost_total = 0
        _multiplier = self.cp.E_U * self.cp.M_T * self.cp.GRAVITY * self.cp.DS / self.cp.ETA
        for direction in ['_u', '_d']:
            tc_dir = f"tc{direction}"
            NumTrainsRunning_dir = self.cp.N_TR_UP if direction == '_u' else self.cp.N_TR_DOWN
            if tc_dir in self.variable_groups.keys():
                txt += '>>' * 10 + f' {tc_dir} results ' + '<<' * 10 + '\n'
                sectionRunningTime = self.variable_groups[tc_dir]['t'].sum().getValue()
                txt += f"T{direction}:\t{sectionRunningTime}\n"
                tractionCostPerTrain = _multiplier * self.variable_groups[tc_dir]['kappa'].sum().getValue()
                auxiliaryCostPerTrain = self.cp.MU * sectionRunningTime
                txt += f"tractionCostPerTrain{direction}: \t{tractionCostPerTrain}\n"
                txt += f"auxiliaryCostPerTrain{direction}: \t{auxiliaryCostPerTrain}\n"
                txt += f"No.TrainsRunning{direction}: \t{NumTrainsRunning_dir}\n"
                operation_cost_total += NumTrainsRunning_dir * self.cp.LAMBDA \
                                        * (tractionCostPerTrain + auxiliaryCostPerTrain)
                txt += '>>' * 10 + f' {tc_dir} results ' + '<<' * 10 + '\n'
        txt += f"operationCostTotal: \t{operation_cost_total}\n"
        txt += '>>' * 20 + f' {self.case_name} result - {self.name} ' + '<<' * 20 + '\n'
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
            # 获取约束的左侧表达式
            expr = self.model.getRow(constr)

            # 初始化左侧表达式的值
            lhs_value = 0

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
        # save results in mps, lp, ilp, json files
        self.model.write(f"{save_name}.mps")
        self.model.write(f"{save_name}.lp")
        if self.model.Status == GRB.INFEASIBLE:
            self.model.computeIIS()
            self.model.write(f"{save_name}.ilp")
        else:  # solution file
            self.model.write(f"{save_name}.json")

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
        S = self.ground.num_s
        if "vao" in self.variable_groups.keys():
            e = np.array([self.variable_groups['vao']['e'][i].X for i in range(0, S + 2)])
            pi = np.array([self.variable_groups['vao']['pi'][i].X for i in range(0, S + 2)])
            track = Track(e=e, pi=pi, ground=self.ground)
            fig = track.plot_ground_track()
            fig.savefig(self.directory + f"\\{self.name}-track_profile.pdf", dpi=600)

        if "tc_u" in self.variable_groups.keys() or "tc_d" in self.variable_groups.keys():
            # plot velocity and control
            distances = np.arange(1, S + 2) * self.ground.ds
            plt.figure(figsize=(12, 8))
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212, sharex=ax1)
            for dr in ['_u', '_d']:
                tip_shape = 'x' if dr == "_u" else "o"
                v_arr = np.sqrt([self.variable_groups[f"tc{dr}"]['E_k'][i].X for i in range(1, S + 2)])
                ax1.plot(distances, v_arr, f"{tip_shape}-", label=f"v{dr}")
                u_arr = np.array([self.variable_groups[f'tc{dr}']['u'][i].X for i in range(1, S + 1)])
                ax2.plot(distances[:-1], u_arr, f"{tip_shape}-", label=f"u{dr}")
            ax1.legend()
            ax2.legend()
            plt.savefig(self.directory + f"\\{self.name}-speed_profile.pdf", dpi=600)
        return


def main():
    pass


if __name__ == '__main__':
    main()
