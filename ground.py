from typing import Any

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from dataIO import read_data, get_random_seed

# Notes:
# ------
# points, x, y: real coordinates
# ixy, ix, iy: coordinates indexed by ground.ds and ground.de
# vpi_ixy, vpi_ix, vpi_iy: only vpi coordinates

RANDOM_GROUND_NAMES: list[str] = list(read_data("ground_data\\random_ground_data.json").keys())
REAL_GROUND_NAMES: list[str] = list(read_data("ground_data\\real_ground_data.json").keys())


def _running_mean(x: np.array, N: int) -> np.ndarray:
    cum_sum = np.cumsum(np.insert(x, 0, 0))
    return (cum_sum[N:] - cum_sum[:-N]) / N


def generate_random_ground_points(
        distance_meter: float,
        max_elevation_meter: float,
        num_vpi: int = 100,
        N_smooth: int = 10,
        seed: int = None,
        is_x_location_random: bool = False,
        fluctuation_range: int = 0
) -> np.ndarray:
    seed = get_random_seed(title='gen_rand_ground') if seed is None else seed
    np.random.seed(seed)

    y = np.random.random(size=num_vpi + N_smooth - 1)
    # smooth the random elevation
    y = _running_mean(y, N=N_smooth)
    y = y / (y.max() - y.min()) * max_elevation_meter  # scale to certain range
    y = y - y.min()

    if is_x_location_random:
        x = np.array(
            [0, *sorted(np.random.random(size=num_vpi - 2)), 1]
        ) * distance_meter
    else:
        x = np.arange(y.size) / y.size * \
            (distance_meter + distance_meter / num_vpi)

    if fluctuation_range:
        fluctuation = np.random.random(
            size=x.size) * fluctuation_range * 2 - fluctuation_range
        x += fluctuation

    return np.array([x, y]).T


def calculate_elevation(points: np.array, x: float) -> float:
    """
    Calculates the elevation at a given x-coordinate based on a series of points.

    :return: The interpolated y-coordinate (elevation) at the given x-coordinate. If the
        x-coordinate exactly matches one of the points, the corresponding elevation
        is returned. Otherwise, linear interpolation is used to estimate the elevation.
    """
    i_s = np.searchsorted(points[:, 0], x)
    if x in points[:, 0]:
        return points[i_s, 1]
    else:
        p1 = points[i_s - 1, :]
        p2 = points[i_s, :]
        return p1[1] + (x - p1[0]) * (p2[1] - p1[1]) / (p2[0] - p1[0])


def discretize_points(points: np.array, dx: float, dy: float) -> np.ndarray:
    """
    Discretizes a series of points into a grid with specified x and y intervals.

    :return: A 2D numpy array where each row represents a discretized point with the
        first column as the x-coordinate and the second column as the y-coordinate.
        The y-coordinates are rounded down to the nearest multiple of `dy`.
    """
    x = np.arange(points[0, 0], points[-1, 0] + dx, dx)
    x = x[:-1] if x[-1] > points[-1, 0] else x
    y = np.array([calculate_elevation(points, i) // dy * dy for i in x])
    return np.array([x, y]).T


def get_bottom_left_corner_coordinates(points: np.array) -> tuple[float, float]:
    """
    Finds the bottom-left corner (minimum x and y coordinates) of a series of points.

    :return: A tuple containing the minimum x-coordinate and the minimum y-coordinate
        from the given points, representing the bottom-left corner.
    """
    x, y = points.T
    return x.min(), y.min()


def shift_point_coordinates(points: np.array, bottom_left_corner_coordinates: tuple[float, float]) -> np.ndarray:
    x, y = points.T
    new_x = x - bottom_left_corner_coordinates[0]
    new_y = y - bottom_left_corner_coordinates[1]
    new_points = np.array([new_x, new_y]).T
    return new_points


def from_points_index_to_discrete_points(ixy: np.array, bottom_left_corner_coordinates: tuple[float],
                                         dx: float, dy: float) -> np.ndarray:
    """

    :param ixy: A 2D numpy array where each row represents a point index with the first
        column as the x-index (ix) and the second column as the y-index (iy) in the discretized grid.
    :param bottom_left_corner_coordinates: A tuple containing the x, y coordinates of the
        bottom-left corner of the original points.
    :param dx:
    :param dy:
    :return: A 2D numpy array where each row represents the actual point coordinates
        corresponding to the given indices in the discretized grid (ixy).
    """
    x = ixy[:, 0] * dx + bottom_left_corner_coordinates[0]
    y = ixy[:, 1] * dy + bottom_left_corner_coordinates[1]
    return np.array([x, y]).T


def from_ie_to_e(ie: np.array, bottom_left_corner_coordinates: tuple[float], dy: float) -> np.ndarray:
    """Converts y-indices (iy) in the discretized grid back into actual y-coordinates (elevations)."""
    return ie * dy + bottom_left_corner_coordinates[1]


def get_minmax_gradient_list_from_p_vpi(
        sigma: int, p_vpi: np.ndarray,
        i_max: float, di_max: float,
        start_from_left: bool = True,
        get_max_grad: bool = True) -> list[float]:
    """
    Generates a list of minimum or maximum gradients for a given series of potential vertical
    points of intersection (P_VPI), ensuring that the gradients do not exceed specified limits.

    This function calculates a sequence of gradients based on the input parameters, while
    considering the maximum allowable gradient (`i_max`) and the maximum allowable difference
    between successive gradients (`di_max`). The calculation ensures that each slope section
    contains at least `sigma` intervals, where each interval has a length of `ds`.

    Parameters:
    -----------
    sigma : int
        The minimum number of intervals (`ds`) that a slope section must contain.

    p_vpi : np.array
        A numpy array of binary values representing the potential vertical points of
        intersection (VPI). A value of 1 indicates a VPI, and 0 indicates no VPI.

    i_max : float
        The maximum allowable gradient in the system.

    di_max : float
        The maximum allowable difference between successive gradients.

    start_from_left : bool, optional (default=True)
        If True, the gradient calculation starts from the left (beginning of the array).
        If False, the calculation starts from the right (end of the array).

    get_max_grad : bool, optional (default=True)
        If True, the function calculates the maximum possible gradients.
        If False, it calculates the minimum possible gradients.

    Returns:
    --------
    minmax_grad : list[float]
        A list of calculated gradients, constrained by the input limits (`i_max` and `di_max`),
        ensuring that each slope section contains at least `sigma` intervals.
    """
    current_slope_length_in_ds = sigma
    current_gradient = 0

    _multiplier = 1 if get_max_grad else -1
    if start_from_left:
        get_index = lambda _i: _i
    else:
        get_index = lambda _i: -1 - _i
    minmax_grad: list[float] = [_multiplier * i_max for _ in range(p_vpi.size - 1)]  # size is num_s+1
    potential_vpi = p_vpi if start_from_left else p_vpi[::-1]
    for i, is_vpi in enumerate(potential_vpi):
        # print(i, is_vpi, end="\t")
        current_slope_length_in_ds += 1
        if is_vpi & (current_slope_length_in_ds >= sigma):
            minmax_gradient_from_di = current_gradient + di_max * _multiplier
            current_slope_length_in_ds = 0
            if (minmax_gradient_from_di >= i_max) | (minmax_gradient_from_di <= -i_max):
                break
            minmax_grad[get_index(i)] = minmax_gradient_from_di
            current_gradient = minmax_gradient_from_di
        else:
            minmax_grad[get_index(i)] = current_gradient
        # print(minmax_grad[get_index(i)])
    return minmax_grad


def get_e_from_grad_list(
        e_anchor: float,
        gradients: list[float],
        ds: float, de: float,
        from_left: bool) -> np.ndarray:
    """
    Computes a list of elevations based on an anchor elevation and a list of gradients.

    This function calculates a series of elevation values starting from an anchor point (`e_anchor`),
    using a list of gradients that represent the slope between points. The calculation can proceed
    from left to right or right to left, depending on the `from_left` parameter.

    Parameters:
    -----------
    e_anchor : float
        The starting elevation at the anchor point.

    gradients : list[float]
        A list of gradient values (slopes) between successive points. Positive values indicate
        an upward slope, while negative values indicate a downward slope.

    ds : float
        The distance between successive points in the same unit as the elevation.

    de : float
        A scaling factor for the elevation difference, used to normalize the effect of the gradients.

    from_left : bool
        If True, the calculation proceeds from the left (i.e., using the gradients as provided).
        If False, the calculation proceeds from the right (i.e., using the gradients in reverse order).

    Returns:
    --------
    np.ndarray
        A numpy array of computed elevation values, starting from `e_anchor` and adjusted based on
        the provided gradients.
    """
    gradients = gradients if from_left else gradients[::-1]
    e_list: list[float] = [e_anchor]
    current_e = e_anchor
    for grad in gradients:
        current_e += grad * ds / de
        e_list.append(current_e)
    e_list = e_list if from_left else e_list[::-1]
    return np.array(e_list)


def get_envelope(series: np.ndarray, step: int, is_max: bool = True):
    """
    Computes the envelope of a given time series by finding either the maximum or minimum value
    within each segment of the series, depending on the value of `is_max`. The function returns
    an array that represents the envelope, which includes the positions and values of these extrema.

    Parameters:
    -----------
    series : np.ndarray
        The input time series data as a 1D numpy array.

    step : int
        The length of each segment in which the time series is divided. The envelope will
        be computed by finding the extreme value (maximum or minimum) in each segment.

    is_max : bool, optional (default=True)
        If True, the function will compute the upper envelope (maximum values).
        If False, the function will compute the lower envelope (minimum values).

    Returns:
    --------
    envelope : np.ndarray
        A 2x(num_loops+2) numpy array, vpi_ixy like, where the first row contains the indices of the
        envelope points, and the second row contains the corresponding values. The envelope
        includes the first and last points of the original series to ensure the envelope covers
        the entire range of the series.
    """
    num_loops = round(series.size / step)
    envelope = np.zeros((2, num_loops))
    f_index = np.argmax if is_max else np.argmin
    for k in range(num_loops - 1):
        segment = series[k * step:(k + 1) * step]
        index = f_index(segment)
        envelope[1, k] = segment[index]
        envelope[0, k] = index + step * k
    segment = series[(num_loops - 1) * step:]
    index = f_index(segment)
    envelope[1, num_loops - 1] = segment[index]
    envelope[0, num_loops - 1] = index + step * (num_loops - 1)
    # add first and last data
    envelope = np.hstack(([[0], [series[0]]], envelope.tolist(), [[series.size - 1], [series[-1]]]))
    # drop duplicate columns
    if envelope[0, 0] == envelope[0, 1]:
        envelope = envelope[:, 1:]
    if envelope[0, -1] == envelope[0, -2]:
        envelope = envelope[:, :-1]
    return envelope


def interpolate_vpi_ixy(vpi_ixy: np.array) -> np.ndarray:
    """
    Interpolates the vpi_ixy data to create a full time series.

    This function takes a vpi_ixy, which consists of sparse points representing the
    extreme values of a time series, and interpolates these points to produce a complete
    time series with values at every time index. Linear interpolation is used to fill in
    the gaps between the known vpi points.

    Parameters:
    -----------
    vpi_ixy : np.array
        A 2xN numpy array where the first row contains the indices of the vpi points,
        and the second row contains the corresponding values of the time series at those points.

    Returns:
    --------
    full_data_interpolated : np.ndarray
        A 1D numpy array representing the time series with interpolated values between
        the original vpi points. The length of this array matches the highest index
        in the vpi ixy points.
    """
    num_s = int(vpi_ixy[0, -1])
    full_data = np.full(num_s + 1, np.nan)
    for point in vpi_ixy.T:
        full_data[int(point[0])] = point[1]
    time_indices = np.arange(0, len(full_data))
    valid_indices = ~np.isnan(full_data)
    full_data_interpolated = np.interp(time_indices, time_indices[valid_indices], full_data[valid_indices])
    return full_data_interpolated


class Ground:
    def __init__(self, name: str) -> None:
        self.name: str = name
        if name in RANDOM_GROUND_NAMES:
            type_ = 'random'
        elif name in REAL_GROUND_NAMES:
            type_ = 'real'
        else:
            raise ValueError(f"Ground {name} not found.")
        data = read_data(f"ground_data\\{type_}_ground_data.json")[name]
        self.data: dict[str, Any] = data

        self.ds: float = data['ds']
        self.de: float = data['de']
        self.i_max: float = data['i_max']
        self.di_max: float = data['di_max']
        self.hb_min: float = data['hb_min']
        self.ht_min: float = data["ht_min"]
        self.lk_under: float = data["lk_under"]
        self.c_cut: float = data["c_cut"]
        self.c_fill: float = data["c_fill"]
        self.c_bg: float = data["c_bg"]
        self.c_tn: float = data["c_tn"]
        self.c6e_tn: float = data["c6e_tn"]
        self.N_tr_up: dict[str, int] = data["N_tr_up"]
        self.N_tr_down: dict[str, int] = data["N_tr_down"]
        self.envelope_step: int = data["envelope_step"]
        self.time: dict[str, dict[str, int]] = data["time"]  # first key: train type; second key: "TD_MAX", ...

        self.points: np.ndarray = self.generate_points()  # shape (num_s, 2)
        self.bottom_left_corner_coordinates: tuple[float] = (0.0, 0.0)
        self.points_index: np.ndarray = self.generate_point_index()  # shape (num_s, 2)

        # useful attributes for cases
        self.sigma: int = int(np.ceil(self.data["lk_under"] / self.ds))
        self.e6g: np.ndarray = np.hstack(
            (self.points_index[0, 1], self.points_index[:, 1], self.points_index[-1, 1]))  # shape (num_s+2, )
        self.iy_min: int = self.points_index[:, 1].min()
        self.iy_max: int = self.points_index[:, 1].max()
        self.potential_vpi: np.ndarray = self.get_potential_vpi_array()  # shape (num_s+2, )
        self.ek_lim: np.ndarray = self.get_ek_lim_array()  # shape (num_s+1, 2)
        self.curve_resist: np.ndarray = self.get_curve_resist_array()  # shape (num_s+1, 2)
        self.num_s: int = self.e6g.size - 2
        return

    def generate_points(self) -> np.ndarray:
        if "points" in self.data:
            return np.array(self.data["points"])
        if "seed" in self.data:
            points = generate_random_ground_points(
                distance_meter=self.data['distance'],
                max_elevation_meter=self.data['elevation'],
                num_vpi=self.data['n_vpi'],
                N_smooth=self.data['n_smooth'],
                seed=self.data['seed'],
                is_x_location_random=self.data['x_loc_rand'],
                fluctuation_range=self.data['fl_range']
            )
            return points
        if "point_arr_file" in self.data:
            points = read_data(f"ground_data\\{self.data['point_arr_file']}").values
            return points

    def generate_point_index(self) -> np.ndarray:
        # 处理points转换，变成ix，iy和左下角坐标值
        discrete_points = discretize_points(
            points=self.points, dx=self.ds, dy=self.de)
        self.bottom_left_corner_coordinates = \
            get_bottom_left_corner_coordinates(points=discrete_points)
        discrete_points = shift_point_coordinates(
            points=discrete_points,
            bottom_left_corner_coordinates=self.bottom_left_corner_coordinates)
        points_index = np.array(
            [discrete_points[:, 0] / self.ds,
             discrete_points[:, 1] / self.de]).astype(int).T
        return points_index

    def get_discrete_points(self) -> np.ndarray:
        dis_points = from_points_index_to_discrete_points(
            ixy=self.points_index,
            bottom_left_corner_coordinates=self.bottom_left_corner_coordinates,
            dx=self.ds, dy=self.de
        )
        return dis_points

    def get_ek_lim_array(self) -> np.ndarray:
        v_lim_compact: np.ndarray = np.array(self.data["v_lim"])
        speedLimitArr = self.points_index.copy()
        for row in v_lim_compact:
            loc, limit = row
            ix = (loc - self.bottom_left_corner_coordinates[0]) / self.ds
            speedLimitArr[np.where(speedLimitArr[:, 0] >= ix), 1] = limit
        speedLimitArr[:, 1] = speedLimitArr[:, 1] ** 2
        # access through EkLimitArr[i, 1]
        EkLimitArr = np.vstack((speedLimitArr[0, :], speedLimitArr))
        return EkLimitArr

    def get_curve_resist_array(self) -> np.ndarray:
        curve_radius_compact: np.array = np.array(self.data["curve"])
        curveRadArr = self.points_index.copy().astype(float)
        curveRadArr[:, 1] = np.inf
        for row in curve_radius_compact:
            start, end, rad = row
            ix_start = (start - self.bottom_left_corner_coordinates[0]) / self.ds
            ix_end = (end - self.bottom_left_corner_coordinates[0]) / self.ds
            curveRadArr[np.where((curveRadArr[:, 0] <= ix_end) & (
                    curveRadArr[:, 0] >= ix_start)), 1] = rad
        # calculate the curve resist
        curveRadArr[:, 1] = 600 / curveRadArr[:, 1]
        # access through curveResistArr[i, 1]
        curveResistArr = np.vstack((curveRadArr[0, :], curveRadArr))
        return curveResistArr

    def get_potential_vpi_array(self) -> np.ndarray:
        iys_diff3 = np.diff(self.points_index[:, 1], n=3)
        p_vpi_array = np.zeros(self.points_index.shape[0])
        for i, iy_diff in enumerate(iys_diff3[:-1]):
            ix = i + 2
            next_iy_diff = iys_diff3[i + 1]
            if iy_diff * next_iy_diff < 0:
                p_vpi_array[ix] = 1
        p_vpi_array[0], p_vpi_array[-1] = 1, 1
        p_vpi_array = np.hstack((0, p_vpi_array, 0))
        return p_vpi_array.astype(int)

    def get_stair_plot_data(self) -> np.ndarray:
        """Return a np.ndarray with shape of (2, num_s+2), first row locations, second row elevations"""
        x = self.points_index[:, 0] * self.ds + self.bottom_left_corner_coordinates[0]
        x = np.array([x, x]).T.flatten()

        y = self.points_index[:, 1] * self.de + self.bottom_left_corner_coordinates[1]
        y = np.array([y, y]).T.flatten()

        x = np.append(x, x[-1] + self.ds)[1:]
        return np.array([x, y])  # shape is (2 x n)

    def get_absolute_e_range(self) -> np.ndarray:
        """return numpy array with shape of (2, num_s+2), first row lower bound"""
        max_grad_from_left: list[float] = get_minmax_gradient_list_from_p_vpi(
            sigma=self.sigma, p_vpi=self.potential_vpi, i_max=self.i_max, di_max=self.di_max,
            start_from_left=True, get_max_grad=True)
        max_grad_from_right: list[float] = get_minmax_gradient_list_from_p_vpi(
            sigma=self.sigma, p_vpi=self.potential_vpi, i_max=self.i_max, di_max=self.di_max,
            start_from_left=False, get_max_grad=True)
        max_e_from_left: np.ndarray = get_e_from_grad_list(
            e_anchor=float(self.e6g[0]), gradients=max_grad_from_left,
            ds=self.ds, de=self.de, from_left=True)
        max_e_from_right: np.ndarray = get_e_from_grad_list(
            e_anchor=float(self.e6g[-1]), gradients=max_grad_from_right,
            ds=self.ds, de=self.de, from_left=False)
        max_e_from_iy_max = np.ones_like(self.e6g) * self.iy_max
        ub = np.vstack((max_e_from_left, max_e_from_right, max_e_from_iy_max)).min(axis=0)

        min_grad_from_left: list[float] = get_minmax_gradient_list_from_p_vpi(
            sigma=self.sigma, p_vpi=self.potential_vpi, i_max=self.i_max, di_max=self.di_max,
            start_from_left=True, get_max_grad=False)
        min_grad_from_right: list[float] = get_minmax_gradient_list_from_p_vpi(
            sigma=self.sigma, p_vpi=self.potential_vpi, i_max=self.i_max, di_max=self.di_max,
            start_from_left=False, get_max_grad=False)
        min_e_from_left: np.ndarray = get_e_from_grad_list(
            e_anchor=float(self.e6g[0]), gradients=min_grad_from_left,
            ds=self.ds, de=self.de, from_left=True)
        min_e_from_right: np.ndarray = get_e_from_grad_list(
            e_anchor=float(self.e6g[-1]), gradients=min_grad_from_right,
            ds=self.ds, de=self.de, from_left=False)
        min_e_from_iy_max = np.ones_like(self.e6g) * self.iy_min
        lb = np.vstack((min_e_from_left, min_e_from_right, min_e_from_iy_max)).max(axis=0)

        return np.array([lb, ub])

    def get_envelope(self) -> np.ndarray:
        """return numpy array with shape of (2, num_s+2), first row lower bound"""
        if self.envelope_step is None:
            num_envelope: int = 10  # to be tuned
            self.envelope_step = max(int(self.sigma * 2), self.e6g.size // num_envelope)
        upper_envelope = get_envelope(series=self.e6g[1:-1], step=self.envelope_step, is_max=True)
        lower_envelope = get_envelope(series=self.e6g[1:-1], step=self.envelope_step, is_max=False)
        upper_envelope_full = interpolate_vpi_ixy(vpi_ixy=upper_envelope)
        upper_envelope_full = np.hstack((upper_envelope_full[0], upper_envelope_full, upper_envelope_full[-1]))
        lower_envelope_full = interpolate_vpi_ixy(vpi_ixy=lower_envelope)
        lower_envelope_full = np.hstack((lower_envelope_full[0], lower_envelope_full, lower_envelope_full[-1]))

        # if the lower envelope has greater gradients than i_max, smooth it
        e_diff_range = self.i_max * self.ds / self.de
        # forward check
        for i in range(self.num_s):
            # lower
            le1, le2 = lower_envelope_full[i], lower_envelope_full[i + 1]
            le2_min, le2_max = le1 - e_diff_range, le1 + e_diff_range
            le2 = np.max([le2, le2_min])
            le2 = np.min([le2, le2_max])
            lower_envelope_full[i + 1] = le2
            # upper
            ue1, ue2 = upper_envelope_full[i], upper_envelope_full[i + 1]
            ue2_min, ue2_max = ue1 - e_diff_range, ue1 + e_diff_range
            ue2 = np.max([ue2, ue2_min])
            ue2 = np.min([ue2, ue2_max])
            upper_envelope_full[i + 1] = ue2

        # backward check
        for i in range(1, self.num_s + 1):
            # lower
            le1, le2 = lower_envelope_full[-i - 2], lower_envelope_full[-i - 1]
            le1_min, le1_max = le2 - e_diff_range, le2 + e_diff_range
            le1 = np.max([le1, le1_min])
            le1 = np.min([le1, le1_max])
            lower_envelope_full[-i - 2] = le1
            # upper
            ue1, ue2 = upper_envelope_full[-i - 2], upper_envelope_full[-i - 1]
            ue1_min, ue1_max = ue2 - e_diff_range, ue2 + e_diff_range
            ue1 = np.max([ue1, ue1_min])
            ue1 = np.min([ue1, ue1_max])
            upper_envelope_full[-i - 2] = ue1

        # if the envelope exceeds actual ground, change to ground.
        # upper_envelope_full = np.vstack((self.e6g, upper_envelope_full)).max(axis=0)
        lower_envelope_full = np.vstack((self.e6g, lower_envelope_full)).min(axis=0)

        return np.array([lower_envelope_full, upper_envelope_full])

    def plot_ground_with_envelope(self) -> plt.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=150)

        locations = self.points_index[:, 0] * self.ds + self.bottom_left_corner_coordinates[0]
        locations = np.hstack((locations[0] - self.ds, locations, locations[-1] + self.ds))

        ground_ele = from_ie_to_e(
            ie=self.e6g, bottom_left_corner_coordinates=self.bottom_left_corner_coordinates, dy=self.de)
        ax.plot(locations, ground_ele, label="Ground", c="lightgray")
        lb, ub = self.get_absolute_e_range()
        lb = from_ie_to_e(ie=lb, bottom_left_corner_coordinates=self.bottom_left_corner_coordinates, dy=self.de)
        ub = from_ie_to_e(ie=ub, bottom_left_corner_coordinates=self.bottom_left_corner_coordinates, dy=self.de)
        le, ue = self.get_envelope()
        le = from_ie_to_e(ie=le, bottom_left_corner_coordinates=self.bottom_left_corner_coordinates, dy=self.de)
        ue = from_ie_to_e(ie=ue, bottom_left_corner_coordinates=self.bottom_left_corner_coordinates, dy=self.de)

        ax.plot(locations, lb, label="Lowerbound", c="r", marker="", lw=0.75)
        ax.plot(locations, ub, label="Upperbound", c="r", marker="", lw=0.75)
        ax.plot(locations, ue, label=f"Upper envelope", c="b", lw=0.75)
        ax.plot(locations, le, label=f"Lower envelope", c="b", lw=0.75)

        ax.legend(handles=[Line2D([0], [0], color="lightgray", lw=1, label='Ground'),
                           Line2D([0], [0], color="r", lw=0.75, label='Absolute bounds'),
                           Line2D([0], [0], color="b", lw=0.75, label='Envelope bounds')],
                  fontsize="small")
        ax.set_xlabel("Horizontal location (m)", fontsize="small")
        ax.set_ylabel("Elevation (m)", fontsize="small")
        plt.tight_layout()
        plt.show()
        return fig


def main():
    pass


if __name__ == '__main__':
    main()
