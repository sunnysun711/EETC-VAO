# heuristic algorithm to find feasible solutions
from typing import Any

import numpy as np
from scipy.stats import truncnorm

from ground import Ground, interpolate_vpi_ixy
from track import Track, gen_track_from_file
from train import Train


def select_in_range(left: float | np.ndarray, right: float | np.ndarray,
                    ratio: float | np.ndarray = None, triangular_mode: float | np.ndarray = None,
                    truncnorm_mu: float = None, truncnorm_sigma: float = 1.) -> float | np.ndarray:
    """
    Selects a value or array of values within a specified range, using uniform,
    triangular, or truncated normal distribution based on the provided parameters.

    :param left: Lower limit.
    :param right: Upper limit, must be larger than left.
    :param ratio: A ratio between 0 and 1 for linear interpolation between left and right.
    :param triangular_mode: The mode (peak) of the triangular distribution.
                            The value must fulfill the condition left <= mode <= right.
    :param truncnorm_mu: A float for the mean of the standard normal distribution to be
                         truncated. If provided, values are drawn from a truncated normal
                         distribution.
    :param truncnorm_sigma: A float for the standard deviation of the normal distribution
                            to be truncated. Default is 1.
    :return: A scalar or an ndarray of selected values.

    Notes:
    ------
    - `ratio` has the highest priority. If provided, it will be used to generate the values.
    - `triangular_mode` and `truncnorm_mu` are mutually exclusive. Only one of them should be provided.
    - If neither `ratio`, `triangular_mode`, nor `truncnorm_mu` is provided,
      the selection will be uniform within the specified range.
    """

    # Check for mutual exclusivity of triangular_mode and truncnorm_params
    if triangular_mode is not None and truncnorm_mu is not None:
        raise ValueError("`triangular_mode` and `truncnorm_params` cannot both be provided. Choose one.")

    if left == right:  # range closed
        return left

    if ratio is not None:  # Use ratio for linear interpolation between left and right
        selected = left + ratio * (right - left)
    elif triangular_mode is not None:  # Use triangular distribution with the given mode
        selected = np.random.triangular(left=left, mode=triangular_mode, right=right)
    elif truncnorm_mu is not None:  # Use truncated normal distribution with given mu and sigma
        a, b = (left - truncnorm_mu) / truncnorm_sigma, (right - truncnorm_mu) / truncnorm_sigma  # Standardized bounds
        selected = truncnorm(a, b, loc=truncnorm_mu, scale=truncnorm_sigma).rvs(size=np.shape(left))
    else:  # Default to uniform distribution
        selected = np.random.uniform(left, right)
    return selected


def calculate_ratio(value, min_, max_):
    """if min_ == max_, return 0, not DivideByZeroError."""
    mask = np.isclose(max_, min_)
    ratio = np.zeros_like(value, dtype=float)
    if isinstance(value, float):
        if not mask:
            ratio = (value - min_) / (max_ - min_)
    else:
        ratio[~mask] = (value[~mask] - min_[~mask]) / (max_[~mask] - min_[~mask])
    return ratio


def gen_track(ground: Ground, display_on: bool = True, triangular_select: bool = True) -> Track:
    """generate a Track object from a Ground object."""
    # get envelope
    lb, ub = ground.get_absolute_e_range()
    le, ue = ground.get_envelope()
    e_min, e_max = np.max([lb, le], axis=0), np.min([ub, ue], axis=0)

    # check feasibility
    if (e_max - e_min < 0).any():
        locations_not_feasible = np.where(e_max - e_min < 0)[0] + 1
        raise Exception(f'Upper bound is smaller than lower bound!\nub: {e_max}\nlb: {e_min}.\n'
                        f'locations_not_feasible: {locations_not_feasible}')

    # generate vao solution by choose random vpi and assign random elevations in the forward direction,
    # where elevation range is determined sequentially.
    # the final result should have the size of S. Only VPI points have elevation values, non-VPI points are zero.
    # example: [40, 0, 0, 45, 0, 0, 0, 60, 0, 60] (sigma is 3, S is 10), VPI points are: 1, 4, 8, 10.
    p_vpi_locs = np.where(ground.potential_vpi == 1)[0][1:]  # potential vpi locations, exclude the first one.
    skip_chances = 0.25  # when deciding whether to choose this vpi, there is skip_chances to skip this vpi point.

    while True:
        vpi_sol = np.full(ground.num_s, None)
        vpi_sol[0], vpi_sol[-1] = ground.e6g[1], ground.e6g[-2]  # first and last vpi point

        current_loc, destination_loc = 1, ground.num_s
        current_ele, destination_ele = ground.e6g[1], ground.e6g[-2]
        previous_gradient = 0

        for s_ in p_vpi_locs:
            step = int(s_ - current_loc)
            if (current_loc != 1) & (s_ != destination_loc) & (step < ground.sigma):
                # not the first slope & not the last slope & slope length too small
                continue
            if (s_ != destination_loc) & (np.random.random() < skip_chances):
                # not the last slope & just randomly skip some vpi points
                continue

            # find a feasible range of elevations at this location s_
            # 1. lb and ub
            upper_ele, lower_ele = e_max[s_ - 1], e_min[s_ - 1]
            # 2. i_max
            i_max_ele = step * ground.ds * ground.i_max / ground.de + current_ele
            i_min_ele = step * ground.ds * - ground.i_max / ground.de + current_ele
            # 3. di_max
            i_max, i_min = previous_gradient + ground.di_max, previous_gradient - ground.di_max
            di_max_ele = step * ground.ds * i_max / ground.de + current_ele
            di_min_ele = step * ground.ds * i_min / ground.de + current_ele
            max_eles, min_eles = [upper_ele, i_max_ele, di_max_ele], [lower_ele, i_min_ele, di_min_ele]
            # 4. special cases.
            if (current_loc == 1) & (step < ground.sigma):  # first slope connect with station 1 platform
                max_eles.append(current_ele)
                min_eles.append(current_ele)
            elif destination_loc - s_ < ground.sigma:  # last slope connect with station 2 platform
                max_eles.append(destination_ele)
                min_eles.append(destination_ele)
                # also must be between di_max range with level track (station 2 platform)
                di_max_ele_from_end = (s_ - current_loc) * ground.ds * (0 + ground.di_max) / ground.de + current_ele
                di_min_ele_from_end = (s_ - current_loc) * ground.ds * (0 - ground.di_max) / ground.de + current_ele
                max_eles.append(di_max_ele_from_end)
                min_eles.append(di_min_ele_from_end)

            # final range
            max_ele, min_ele = min(max_eles), max(min_eles)

            if max_ele < min_ele:  # check gradient range feasibility
                if display_on:
                    print(f'\033[32m' + f'[VAO generation] Not feasible at: {s_}/{ground.num_s}' + f'\033[0m')
                break

            # select ele
            if min_ele == max_ele:
                selected_ele = select_in_range(left=min_ele, right=max_ele)
            elif triangular_select:
                ground_ele = ground.e6g[s_]
                triangular_mode = min(max(ground_ele, min_ele), max_ele)
                selected_ele = select_in_range(left=min_ele, right=max_ele, triangular_mode=triangular_mode)
            else:
                selected_ele = select_in_range(left=min_ele, right=max_ele)
            vpi_sol[s_ - 1] = selected_ele
            # update previous gradient, set current loc and ele
            previous_gradient = (selected_ele - current_ele) * ground.de / (ground.ds * step)
            current_loc = s_
            current_ele = selected_ele
        else:  # for loop over without break, which means a feasible solution is found.
            break
    if display_on:
        print(f"\033[1;33m" + "[VAO generation] Solution found!" + f"\033[0m")

    # add station 1 and station 2 platforms
    vpi_sol = np.insert(vpi_sol, 0, vpi_sol[0])
    vpi_sol = np.append(vpi_sol, vpi_sol[-1])

    # get vpi_ixy
    vpi_ix = np.where(vpi_sol != None)
    vpi_iy = vpi_sol[vpi_ix]
    sol = interpolate_vpi_ixy(np.vstack((vpi_ix, vpi_iy)))
    sol_track = Track(e=sol, ground=ground)
    return sol_track


def get_v_anchors(S: int, DS: float, tr: Train, T_MAX: float, v_lim_real: np.ndarray, is_downhill_dir: bool = True,
                  acc_ratio: float = 0.5, anchor_ratio: float = 0.1, lower_ratio: float = 0.65) -> np.ndarray[float]:
    """
    Calculate the anchor velocities for a train moving through a series of intervals based
    on velocity limits, acceleration, deceleration, and a time constraint.

    :param S: Total number of intervals.
    :param DS: Delta_s, length of each interval, meter.
    :param tr: Train object that contains attributes like maximum acceleration and deceleration.
    :param T_MAX: Maximum running time, second.
    :param v_lim_real: Velocity limit, km/h. Size: S+1.
    :param is_downhill_dir: A flag for whether the train is moving downhill.
    :param acc_ratio: The ratio of used acceleration to the maximum train acceleration,
        ranging from 0 to 1. Both acceleration and deceleration are considered.
    :param anchor_ratio: A ratio between 0 and 1 that determines the positioning of the
        anchor velocity between the calculated lower bound and the velocity limit. A
        value closer to 0 moves the anchor closer to the lower bound, while a value
        closer to 1 moves it closer to the velocity limit.
    :param lower_ratio: A ratio between 0 and 1 that determines the ratio of the lower
        anchors to the calculated lower anchors.

    :return: A v_anchor array with size S+1 (same as the final v_arr)
    """
    # exceptions:
    if acc_ratio > 1 or acc_ratio < 0:
        raise ValueError('acc_ratio must be between 0 and 1')
    if anchor_ratio > 1 or anchor_ratio < 0:
        raise ValueError('anchor_ratio must be between 0 and 1')

    acc = tr.max_acc * acc_ratio  # m/s^2
    dec = tr.max_dec * acc_ratio  # m/s^2

    # check equation feasibility:
    delta = T_MAX ** 2 - 2 * DS * S * (1 / acc + 1 / dec)
    # print(f"{delta} = {T_MAX} ** 2 - 2 * {DS} * {S} * (1 / {acc} + 1 / {dec})")
    while delta < 0:
        if acc > tr.max_acc or dec > tr.max_dec:
            acc /= 1.1  # roll back to a previous step
            dec /= 1.1  # roll back to a previous step
            T_MAX = (2 * DS * S * (1 / acc + 1 / dec)) ** 0.5  # force T_MAX to make it feasible
            print(f"T_MAX is altered to {T_MAX} to calculate the v_anchor.")
            delta = 0
            break
        acc *= 1.1  # try to increase acc and dec
        dec *= 1.1
        delta = T_MAX ** 2 - 2 * DS * S * (1 / acc + 1 / dec)

    # calculate ta, tc, td, and vc for the T_MAX situation.
    vc = (T_MAX - delta ** 0.5) / (1 / acc + 1 / dec)  # cruising v (m/s)
    ta = vc / acc  # acceleration time
    td = vc / dec  # deceleration time
    tc = T_MAX - ta - td  # cruising time

    # calculate the control scheme switching points locations: s1: acc-cru, s2: cru-dec
    s1 = 0.5 * acc * ta ** 2 if is_downhill_dir else 0.5 * dec * td ** 2
    s2 = DS * S - 0.5 * dec * td ** 2 if is_downhill_dir else DS * S - 0.5 * acc * ta ** 2
    v_lower_anchors = []
    for s in range(S + 1):
        if s * DS <= s1:  # accelerating
            v_lower_anchor = (2 * acc * s * DS) ** 0.5 if is_downhill_dir else (2 * dec * s * DS) ** 0.5
        elif s * DS >= s2:  # decelerating
            v_lower_anchor = (2 * dec * DS * (S - s)) ** 0.5 if is_downhill_dir else (2 * acc * DS * (S - s)) ** 0.5
        else:  # cruising
            v_lower_anchor = vc
        v_lower_anchors.append(v_lower_anchor)  # m/s
    v_lower_anchors = np.array(v_lower_anchors) * 3.6  # km/h

    # make sure that the lower anchors are not greater than the real velocity limit.
    v_lower_anchors = np.minimum(v_lower_anchors, v_lim_real) * lower_ratio

    v_anchors = v_lower_anchors + anchor_ratio * (v_lim_real - v_lower_anchors)

    #########################
    #### For visualizing ####
    #########################
    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('TkAgg')
    # plt.plot(v_lim_real, "o-", markersize=2)
    # plt.plot(v_lower_anchors, "o-", markersize=2)
    # plt.plot(v_anchors, "o-", markersize=2)
    # plt.show()
    # print("v_upper_anchors: \n", v_lim_real)
    # print("v_lower_anchors: \n", v_lower_anchors)
    # print("v_anchors: \n", v_anchors)

    return v_anchors


def simulate_tc(track: Track, tr: Train, is_downhill_dir: bool, v_select_ratio: np.ndarray | float | int = None,
                use_v_anchor: bool = True, v_step_wise: float = 0.01, display_on: bool = True, **kwargs):
    """
    Simulates train control along a track section, adjusting velocity and control based
    on track conditions and train properties.

    :param track: Track object. used attributes: ground, e, z1.
    :param tr: Train object. used attributes: name. used methods: unit_w0, unit_wtn,
        next_v_range, get_unit_control_from_v
    :param is_downhill_dir: A flag indicating whether the simulation is in the downhill direction.
    :param v_select_ratio: optional, a specific selection ratio or array of ratios
        that influence how the train's velocity is controlled. If None, a random
        control sequence is generated. If a float or int, this ratio is used consistently
        across the simulation. Apply 1 to get the speed limit.
    :param use_v_anchor: optional, a flag indicating whether to use velocity anchors.
        If True, velocity anchors are first generated via ``get_v_anchors``.
    :param v_step_wise: optional, default is 0.01, a step size for deducing the velocity
        limits, both upper and lower limits.
    :param display_on: optional, a flag for enabling print statements that show progress
        and any issues during the simulation.
    :param kwargs: optional keyword arguments for adjusting ``get_v_anchors`` parameters.

    :return: tuple of three numpy arrays:
        - ``v_arr``: Array of velocities for each segment of the track. Size: S+1.
        - ``u_arr``: Array of unit control forces for each segment of the track. Size: S.
        - ``v_select_ratio``: Array of ratios used for velocity selection. Size: S.
        - ``omegas``: Array of resistances for each segment of the track. Shape: (S, 4).

    Details:
    -------
    The function simulates train movement across a track by determining optimal velocities and control settings
    based on track elevation changes, curvature, tunnel presence, and other resistive factors. It ensures the
    train's velocity does not exceed safety limits and adjusts for the most efficient operation. Simulation
    accounts for changes in direction and the potential need to reverse adjustments if computed velocities
    are not feasible under the given constraints.

    Example:
    -------
    To simulate train control for a given track and train configuration:

    >>> track = gen_track(ground=Ground("gd2"))
    >>> train = Train("HXD1D")
    >>> velocities, controls, ratios, omegas = simulate_tc(track, train, is_downhill_dir=True)

    To get the actual speed limit:

    >>> speed_limit, _, _, _ = simulate_tc(track, train, is_downhill_dir=True, v_select_ratio=1)
    """
    S = track.ground.num_s
    DS, DE = track.ground.ds, track.ground.de
    T_MAX = track.ground.time[tr.name]["TD_MAX"] if is_downhill_dir else track.ground.time[tr.name]["TU_MAX"]

    curve_resist_arr = track.ground.curve_resist[1:, 1]  # curve resist, size: S
    track_ele = track.e.copy()  # size: S+2
    is_tunnel_arr = np.insert(track.z1.copy(), 0, 0).astype(bool)  # size: S+2
    v_limit_arr = track.ground.ek_lim[:, 1].copy() ** 0.5  # size: S+1
    v_limit_arr = np.minimum(v_limit_arr, tr.max_v)  # km/h
    v_min_arr = np.zeros_like(v_limit_arr)  # size: S+1
    v_limit_arr[0], v_limit_arr[-1] = 0, 0
    v_min_arr[0], v_min_arr[-1] = 0, 0

    v_anchors = np.full_like(v_limit_arr, None)
    if isinstance(v_select_ratio, (float, int)):  # use a fixed ratio value for speed selection
        v_select_ratio = np.ones(S) * v_select_ratio
    elif v_select_ratio is not None:  # use a fixed ratio array for speed selection
        pass
    else:  # random or using v_anchors
        v_select_ratio = np.full(S, None)
        if use_v_anchor:
            # get the SOTC result, the v_arr happens to be the real v_lim.
            v_lim_real, _, _, _ = simulate_tc(
                track, tr, is_downhill_dir, v_select_ratio=1, use_v_anchor=False, display_on=False)
            # use kwargs to adjust v_anchors
            v_anchors = get_v_anchors(
                S=S, DS=DS, tr=tr, T_MAX=T_MAX, v_lim_real=v_lim_real, is_downhill_dir=is_downhill_dir, **kwargs)
            # print(v_anchors)
            v_limit_arr = np.minimum(v_limit_arr, v_lim_real)  # update v limit

    if not is_downhill_dir:
        track_ele = track_ele[::-1]
        is_tunnel_arr = is_tunnel_arr[::-1]
        v_limit_arr = v_limit_arr[::-1]
        curve_resist_arr = curve_resist_arr[::-1]
        v_select_ratio = v_select_ratio[::-1]

    # records list
    v_range_record = []  # 1. [current_v, next_v_lower_bound, next_v_upper_bound]
    v_record = [0]  # 2. velocity
    u_record = []  # 3. unit control
    w_0 = tr.unit_w0(v=0, unit="km/h")
    w_i = (track_ele[1] - track_ele[0]) * DE / DS * 1000
    w_r = curve_resist_arr[0]
    w_tn = tr.unit_wtn(v=0, is_tunnel=is_tunnel_arr[0])
    w_record = [[w_0, w_i, w_r, w_tn]]  # 4. resistances

    # equal to elevation's index s. Subsection s_ is bounded by current_v (left) and next_v (right)
    s_ = 1  # starting point
    while s_ != S + 1:
        current_v = v_record[-1]
        # calculate v_range  -> [v, lb, ub]
        next_v_limit, next_v_min = v_limit_arr[s_], v_min_arr[s_]
        v_range = tr.next_v_range(
            ds=DS, speed_limit=next_v_limit, speed_min=next_v_min, v=current_v,
            w_0=w_record[-1][0], w_i=w_record[-1][1], w_r=w_record[-1][2], w_tn=w_record[-1][3])[0]
        # check v_range feasibility
        if v_range[1] > v_range[2]:  # over speed guaranteed
            if display_on:
                print(f"\033[32m"
                      f"[Train Control Simulation] Over speed at loc {s_} with v and v_range: {v_range}."
                      f"\033[0m")
            # modify the speed limit in left side of s_
            step_wise_v = np.arange(start=current_v, stop=0, step=-v_step_wise)
            test_v_range = tr.next_v_range(
                ds=DS, speed_min=next_v_min, speed_limit=next_v_limit, v=step_wise_v,
                w_0=w_record[-1][0], w_i=w_record[-1][1], w_r=w_record[-1][2], w_tn=w_record[-1][3])
            new_speed_limit_here = test_v_range[test_v_range[:, 1] <= test_v_range[:, 2], 0].max()
            v_limit_arr[s_ - 1] = new_speed_limit_here  # alter v limit
            # backward to the previous step
            w_record.pop()
            v_range_record.pop()
            v_record.pop()
            u_record.pop()
        elif (v_range[2] == 0) & (s_ != S):  # dwell in subsection guaranteed
            # modify the speed min in left side of s_
            step_wise_v = np.arange(start=current_v, stop=tr.max_v, step=v_step_wise)
            test_v_range = tr.next_v_range(
                ds=DS, speed_min=v_range_record[-1][1], speed_limit=v_range_record[-1][2], v=step_wise_v,
                w_0=w_record[-1][0], w_i=w_record[-1][1], w_r=w_record[-1][2], w_tn=w_record[-1][3])
            new_speed_min_here = test_v_range[test_v_range[:, 2] > 0, 0].min()
            v_min_arr[s_ - 1] = new_speed_min_here  # alter v min
            # backward to the previous step
            w_record.pop()
            v_range_record.pop()
            v_record.pop()
            u_record.pop()
        else:  # next_v_range is feasible
            v_sel_ratio, v_anchor = v_select_ratio[s_ - 1], v_anchors[s_]
            v_selected = select_in_range(left=v_range[1], right=v_range[2], ratio=v_sel_ratio, truncnorm_mu=v_anchor)
            # print(s_, v_range, v_selected, v_range[1] <= v_selected <= v_range[2])
            control_ = tr.get_unit_control_from_v(
                ds=DS, v_selected=v_selected, v=current_v,
                w_0=w_record[-1][0], w_i=w_record[-1][1], w_r=w_record[-1][2], w_tn=w_record[-1][3])
            # update record
            v_range_record.append(list(v_range))
            v_record.append(v_selected)
            u_record.append(control_)

            if s_ != S:  # if this is not the last step, calculate next step resistances
                w_0 = tr.unit_w0(v=v_selected)
                w_i = (track_ele[s_ + 1] - track_ele[s_]) * DE / DS * 1000
                w_r = curve_resist_arr[s_]
                w_tn = tr.unit_wtn(v=v_selected, is_tunnel=is_tunnel_arr[s_ + 1])
                # update w_record
                w_record.append([w_0, w_i, w_r, w_tn])
        s_ = len(v_record)
    if display_on:
        print(f"\033[1;33m" + f"[Train Control Simulation] Complete!" + f"\033[0m")

    # final results
    # print(np.array(v_range_record))  # to see the processing steps, uncomment this line
    v_arr, u_arr = np.array(v_record), np.array(u_record)
    v_range_arr = np.array(v_range_record)
    v_select_ratio = calculate_ratio(value=v_arr[1:], min_=v_range_arr[:, 1], max_=v_range_arr[:, 2])
    if not is_downhill_dir:
        v_arr, u_arr, v_select_ratio = v_arr[::-1], u_arr[::-1], v_select_ratio[::-1]
        w_record = w_record[::-1]
    # print(v_range_record)
    return v_arr, u_arr, v_select_ratio, np.array(w_record)


def gen_tc_sol(trk: Track, tr: Train, is_ee: bool = True, display_on: bool = False) -> list[np.ndarray]:
    """
    :param trk:
    :param tr:
    :param is_ee:
    :param display_on:
    :return: list of ndarray, [vd_arr, cd_arr, wd_arr, vu_arr, cu_arr, wu_arr]
        - ``vd_arr``: Array of velocities for each segment of the track in the downhill direction. Size: S+1.
        - ``ud_arr``: Array of unit control forces for each segment of the track in the downhill direction. Size: S.
        - ``wd_arr``: Array of resistances for each segment of the track in the downhill direction. Shape: (S, 4).
        - ``vu_arr``: Array of velocities for each segment of the track in the uphill direction. Size: S+1.
        - ``uu_arr``: Array of unit control forces for each segment of the track in the uphill direction. Size: S.
        - ``wu_arr``: Array of resistances for each segment of the track in the uphill direction. Shape: (S, 4).
    """
    if is_ee:
        res = []
        for _dir in ["D", "U"]:
            is_down = True if _dir == "D" else False
            T_MAX = trk.ground.time[tr.name][f"T{_dir}_MAX"]
            if display_on:
                print("Original T_MAX: ", T_MAX)
            lower_ratio = 0.65
            loop_count = 0
            while True:
                loop_count += 1
                if loop_count > 2:  # need to adjust params
                    if T <= T_MAX * 0.98:
                        lower_ratio -= 0.05
                    elif T > T_MAX:
                        lower_ratio += 0.05
                    if display_on:
                        print(f"lower_ratio is altered to {lower_ratio}.")
                    loop_count = 0
                v_arr, u_arr, _, w_arr = simulate_tc(
                    track=trk, tr=tr, is_downhill_dir=is_down, display_on=display_on, lower_ratio=lower_ratio)
                T = cal_time_from_v(v_arr, tr.PWL_EK, tr.PWL_F_EK, trk.ground.ds, trk.ground.num_s)
                # print(_dir, T, T_MAX)
                if T_MAX >= T >= T_MAX * 0.98:
                    if display_on:
                        print(">>>>>>>>>>>>>>>>>>>>>> Success:\t", _dir, T, T_MAX, T / T_MAX)
                    res.append(v_arr)
                    res.append(u_arr)
                    res.append(w_arr)
                    break
    else:  # sotc
        res = []
        for _dir in ["D", "U"]:
            is_down = True if _dir == "D" else False
            v_arr, u_arr, _, w_arr = simulate_tc(
                track=trk, tr=tr, is_downhill_dir=is_down, v_select_ratio=1, display_on=display_on)
            T = cal_time_from_v(v_arr, tr.PWL_EK, tr.PWL_F_EK, trk.ground.ds, trk.ground.num_s)
            if display_on:
                print(">>>>>>>>>>>>>>>>>>>>>> Success:\t", _dir, " SOTC time:\t", T)
            res.append(v_arr)
            res.append(u_arr)
            res.append(w_arr)

    return res


def get_tc_variables_from_v_u_w(
        v_arr: np.ndarray, u_arr: np.ndarray, w_arr: np.ndarray,
        tr: Train, trk: Track, is_downhill_dir: bool = True) -> dict[str, dict[int, float]]:
    """
    :param v_arr:
    :param u_arr:
    :param w_arr:
    :param tr:
    :param trk:
    :param is_downhill_dir:
    :return: {var_name: {s index: value}}
    """
    num_s = trk.ground.num_s
    ds, de = trk.ground.ds, trk.ground.de

    u = {int(i + 1): u_arr[i] for i in range(num_s)}
    kappa = {int(i + 1): np.abs(u_arr[i]) for i in range(num_s)}
    E_k = {int(i + 1): v_arr[i] ** 2 for i in range(num_s + 1)}
    if is_downhill_dir:
        f_tra = {i: np.interp(E_k[i], tr.traction_e[:, 0], tr.traction_e[:, 1]) for i in range(1, num_s + 1)}
        f_bra = {i: np.interp(E_k[i], tr.brake_e[:, 0], tr.brake_e[:, 1]) for i in range(1, num_s + 1)}
        phi = {i: E_k[i] * trk.z1[i - 1] for i in range(1, num_s + 1)}
    else:
        f_tra = {i: np.interp(E_k[i + 1], tr.traction_e[:, 0], tr.traction_e[:, 1]) for i in range(1, num_s + 1)}
        f_bra = {i: np.interp(E_k[i + 1], tr.brake_e[:, 0], tr.brake_e[:, 1]) for i in range(1, num_s + 1)}
        phi = {i: E_k[i + 1] * trk.z1[i - 1] for i in range(1, num_s + 1)}
    omega_0 = {i: w_arr[i - 1, 0] for i in range(1, num_s + 1)}
    omega_i = {i: w_arr[i - 1, 1] for i in range(1, num_s + 1)}
    omega_r = {i: w_arr[i - 1, 2] for i in range(1, num_s + 1)}
    omega_tn = {i: w_arr[i - 1, 3] for i in range(1, num_s + 1)}
    c = {i: u[i] - (omega_0[i] + omega_i[i] + omega_r[i] + omega_tn[i]) for i in range(1, num_s + 1)}
    f_pwa_v = {i: np.interp(E_k[i], tr.PWL_EK, tr.PWL_F_EK) for i in range(2, num_s + 1)}
    t = {}
    for s in range(1, num_s + 1):
        if s == 1:
            t[s] = 2 * ds * f_pwa_v[2]
        elif s == num_s:
            t[s] = 2 * ds * f_pwa_v[num_s]
        else:
            t[s] = ds / 2 * (f_pwa_v[s] + f_pwa_v[s + 1])

    ##############
    # debug mode #
    ##############
    debug = False
    if debug:
        print("=======> is_downhill_dir: ", is_downhill_dir)
        print("tc-w0", end="\t")
        _mul = 1 if is_downhill_dir else -1
        print(np.array(
            [np.abs((E_k[i + 1] - E_k[i]) * _mul - 23.9733629301 * c[i]) < 1e5 for i in range(1, num_s + 1)]
        ).all())
        print("u_lb", end="\t")
        print(np.array([u[i] >= - f_bra[i] for i in range(1, num_s + 1)]).all())
        print("u_ub", end="\t")
        print(np.array([(u[i] - f_tra[i]) < 1e-7 for i in range(1, num_s + 1)]).all())
        print("omega_r", end="\t")
        print(np.array([omega_r[i] == trk.ground.curve_resist[i, 1] for i in range(1, num_s + 1)]).all())
        print("omega_tn", end="\t")
        print(np.array([omega_tn[i] == tr.r2 * tr.r_tn * phi[i] for i in range(1, num_s + 1)]).all())
        if is_downhill_dir:
            print("omega_0", end="\t")
            print(np.array([omega_0[i] == tr.r0_strahl + tr.r2_strahl * E_k[i] for i in range(1, num_s + 1)]).all())
            print("omega_i", end="\t")
            print(
                np.array([omega_i[i] == (trk.e[i] - trk.e[i - 1]) * de / ds * 1000 for i in range(1, num_s + 1)]).all())
        else:
            print("omega_0", end="\t")
            print(np.array([omega_0[i] == tr.r0_strahl + tr.r2_strahl * E_k[i + 1] for i in range(1, num_s + 1)]).all())
            print("omega_i", end="\t")
            print(
                np.array([omega_i[i] == (trk.e[i] - trk.e[i + 1]) * de / ds * 1000 for i in range(1, num_s + 1)]).all())
        print("2as==ek - ek", end="\t")
        print(np.array([np.abs(2 * ds * 3.6 ** 2 / ((1 + tr.rho) * 102) * c[i] * _mul
                               - (E_k[i + 1] - E_k[i])) < 1e-7 for i in range(1, num_s + 1)]).all())

    res = {
        'u': u,
        'kappa': kappa,
        "E_k": E_k,
        "f_tra": f_tra,
        "f_bra": f_bra,
        "omega_0": omega_0,
        "omega_i": omega_i,
        "omega_r": omega_r,
        "omega_tn": omega_tn,
        "c": c,
        "t": t,
        "f_pwa_v": f_pwa_v,
        "phi": phi,
    }
    return res


def gen_warm_start_data(ground: Ground, tr: Train, vao_sol_file: str = None,
                        display_on: bool = False) -> dict[str, dict[str, dict[int, Any]]]:
    """
    get warm start data (dict) for the EETC_VAO model. if vao_sol_file provided, load track from file.
    :return: {grp: {var_name: {var_key: value,...},...},...}
    """
    if vao_sol_file is None:
        trk = gen_track(ground=ground, display_on=display_on, triangular_select=True)
    else:
        trk = gen_track_from_file(ground=ground, json_file=vao_sol_file)
    vd_arr, ud_arr, wd_arr, vu_arr, uu_arr, wu_arr = gen_tc_sol(trk=trk, tr=tr, display_on=display_on)

    ######################
    #### for debug #######
    ######################
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('TkAgg')
    # trk.plot_ground_track()
    #
    # plt.figure(figsize=(8, 4))
    # distances = np.arange(0, ground.num_s + 1) * ground.ds + ground.bottom_left_corner_coordinates[0]
    # ax1 = plt.subplot(211)
    # ax2 = plt.subplot(212, sharex=ax1)
    #
    # marker_size = 0
    #
    # ax1.plot(distances, vd_arr, marker='o', markersize=marker_size, ls="-", label="v_d")
    # ax1.plot(distances, vu_arr, marker='x', markersize=marker_size, ls="-", label="v_u")
    #
    # ax2.plot(distances[:-1], ud_arr, marker='o', markersize=marker_size, ls="-", label="u_d")
    # ax2.plot(distances[:-1], uu_arr, marker='x', markersize=marker_size, ls="-", label="u_u")
    #
    # ax1.legend(fontsize="small")
    # ax1.set_ylabel("Velocity (km/h)", fontsize="small")
    # ax2.legend(fontsize="small")
    # ax2.set_xlabel("Horizontal location (m)", fontsize="small")
    # ax2.set_ylabel("Unit control force (N/kN)", fontsize="small")
    # plt.tight_layout()
    # plt.show()

    warm_start_data: dict = {
        "vao": trk.get_warm_start_data(),
        "tc_u": get_tc_variables_from_v_u_w(
            v_arr=vu_arr, u_arr=uu_arr, w_arr=wu_arr, tr=tr, trk=trk, is_downhill_dir=False),
        "tc_d": get_tc_variables_from_v_u_w(
            v_arr=vd_arr, u_arr=ud_arr, w_arr=wd_arr, tr=tr, trk=trk, is_downhill_dir=True)
    }
    return warm_start_data


def cal_time_from_v(v_arr: np.ndarray, PWL_EK: list, PWL_F_EK: list, ds: float, num_s: int):
    """calculate time from v_arr using approximation method in the proposed model."""
    E_k = {int(i + 1): v_arr[i] ** 2 for i in range(num_s + 1)}
    f_pwa_v = {i: np.interp(E_k[i], PWL_EK, PWL_F_EK) for i in range(2, num_s + 1)}

    t = {}
    for i in range(1, num_s + 1):
        if i == 1:
            t[i] = 2 * ds * f_pwa_v[2]
        elif i == num_s:
            t[i] = 2 * ds * f_pwa_v[num_s]
        else:
            t[i] = ds / 2 * (f_pwa_v[i] + f_pwa_v[i + 1])
    return sum(t.values())


def main():
    pass


if __name__ == '__main__':
    main()
