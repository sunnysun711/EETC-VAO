import numpy as np
from matplotlib import pyplot as plt

from dataIO import read_data, get_random_seed


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
            [0, *sorted(np.random.random(size=num_vpi - 2)), 1]) * distance_meter
    else:
        x = np.arange(y.size) / y.size * \
            (distance_meter + distance_meter / num_vpi)

    if fluctuation_range:
        fluctuation = np.random.random(
            size=x.size) * fluctuation_range * 2 - fluctuation_range
        x += fluctuation

    return np.array([x, y]).T


def calculate_elevation(points: np.array, x: float) -> float:
    i_s = np.searchsorted(points[:, 0], x)
    if x in points[:, 0]:
        return points[i_s, 1]
    else:
        p1 = points[i_s - 1, :]
        p2 = points[i_s, :]
        return p1[1] + (x - p1[0]) * (p2[1] - p1[1]) / (p2[0] - p1[0])


def discretize_points(points: np.array, dx: float, dy: float) -> np.ndarray:
    x = np.arange(points[0, 0], points[-1, 0] + dx, dx)
    x = x[:-1] if x[-1] > points[-1, 0] else x
    y = np.array([calculate_elevation(points, i) // dy * dy for i in x])
    return np.array([x, y]).T


def shift_point_coordinates(points: np.array) -> tuple[np.ndarray, tuple[float, float]]:
    x, y = points.T
    bottom_left_corner_coordinates = x.min(), y.min()
    x = x - x.min()
    y = y - y.min()
    new_points = np.array([x, y]).T
    return new_points, bottom_left_corner_coordinates


def from_points_index_to_discrete_points(
        points_index: np.array,
        bottom_left_corner_coordinates: tuple[float, float],
        dx: float,
        dy: float) -> np.ndarray:
    new_x = points_index[:, 0] * dx + bottom_left_corner_coordinates[0]
    new_y = points_index[:, 1] * dy + bottom_left_corner_coordinates[1]
    return np.array([new_x, new_y]).T


def get_minmax_gradient_list_from_p_vpi(
        sigma: int, p_vpi: np.array,
        i_max: float, di_max: float,
        start_from_left: bool = True,
        get_max_grad: bool = True) -> list[float]:
    current_slope_length_in_ds = sigma
    current_gradient = 0

    _multiplier = 1 if get_max_grad else -1
    if start_from_left:
        get_index = lambda _i: _i
    else:
        get_index = lambda _i: -1 - _i
    # get_index = lambda _i: _i if start_from_left else lambda _i: -1 - _i
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


def get_e_from_grad_list(e_anchor: float, gradients: list[float], ds: float, de: float, from_left: bool) -> np.ndarray:
    gradients = gradients if from_left else gradients[::-1]
    e_list: list[float] = [e_anchor]
    current_e = e_anchor
    for grad in gradients:
        current_e += grad * ds / de
        e_list.append(current_e)
    e_list = e_list if from_left else e_list[::-1]
    return np.array(e_list)


def get_envelope(series: np.ndarray, step: int, is_max: bool = True):
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
    return envelope


def interpolate_envelope(envelope: np.ndarray) -> np.ndarray:
    num_s = int(envelope[0, -1])
    full_data = np.full(num_s + 1, np.nan)
    for point in envelope.T:
        full_data[int(point[0])] = point[1]
    time_indices = np.arange(0, len(full_data))
    valid_indices = ~np.isnan(full_data)
    full_data_interpolated = np.interp(time_indices, time_indices[valid_indices], full_data[valid_indices])
    return full_data_interpolated


class Ground:
    def __init__(self, name: str, type_: str = "random") -> None:
        self.name: str = name
        self.ds: float = None
        self.de: float = None
        self.i_max: float = None
        self.di_max: float = None
        self.hb_min: float = None
        self.ht_min: float = None
        self.lk_under: float = None
        self.c_cut: float = None
        self.c_fill: float = None
        self.c_bg: float = None
        self.c_tn: float = None
        self.c6e_tn: float = None
        self.N_tr_up: int = None
        self.N_tr_down: int = None
        self.time: dict[str, dict[str, int]] = None  # first key: train type; second key: TD_MAX

        if type_ in ["random", "real"]:
            data = read_data(f"ground_data\\{type_}_ground_data.json")[name]
        else:
            raise ValueError(f"type_ should be 'random' or 'real'.")

        for k, v in data.items():
            if isinstance(v, list):
                try:
                    v = np.array(v)
                except ValueError:
                    pass
            setattr(self, k, v)

        self.points: np.ndarray = self.generate_points()  # shape (num_s, 2)
        self.bottom_left_corner_coordinates: tuple[float] = None
        # also update bottom_left_corner_coordinates
        self.points_index: np.ndarray = self.generate_point_index()  # shape (num_s, 2)

        # useful attributes for cases
        self.sigma: int = int(np.ceil(self.__dict__["lk_under"] / self.ds))
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
        if "points" in self.__dict__:
            return self.__dict__["points"]
        if "seed" in self.__dict__:
            points = generate_random_ground_points(
                distance_meter=self.__dict__['distance'],
                max_elevation_meter=self.__dict__['elevation'],
                num_vpi=self.__dict__['n_vpi'],
                N_smooth=self.__dict__['n_smooth'],
                seed=self.__dict__['seed'],
                is_x_location_random=self.__dict__['x_loc_rand'],
                fluctuation_range=self.__dict__['fl_range']
            )
            return points
        if "point_arr_file" in self.__dict__:
            points = read_data(f"ground_data\\{self.__dict__['point_arr_file']}").values
            return points

    def generate_point_index(self) -> np.ndarray:
        # 处理points转换，变成ix，iy和左下角坐标值
        discrete_points = discretize_points(points=self.points, dx=self.ds, dy=self.de)
        discrete_points, bottom_left_corner_coordinates = shift_point_coordinates(points=discrete_points)

        # update attribute
        self.bottom_left_corner_coordinates = bottom_left_corner_coordinates
        points_index = np.array(
            [discrete_points[:, 0] / self.ds,
             discrete_points[:, 1] / self.de]
        ).astype(int).T

        return points_index

    def get_discrete_points(self) -> np.ndarray:
        dis_points = from_points_index_to_discrete_points(
            points_index=self.points_index,
            bottom_left_corner_coordinates=self.bottom_left_corner_coordinates,
            dx=self.ds, dy=self.de
        )
        return dis_points

    def get_ek_lim_array(self) -> np.ndarray:
        v_lim_compact: np.ndarray = self.__dict__["v_lim"]
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
        curve_radius_compact: np.array = self.__dict__["curve"]
        curveRadArr = self.points_index.copy().astype(float)
        curveRadArr[:, 1] = np.inf
        for row in curve_radius_compact:
            start, end, rad = row
            ix_start = (
                               start - self.bottom_left_corner_coordinates[0]) / self.ds
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
        x = self.points_index[:, 0] * self.ds + self.bottom_left_corner_coordinates[0]
        x = np.array([x, x]).T.flatten()

        y = self.points_index[:, 1] * self.de + self.bottom_left_corner_coordinates[1]
        y = np.array([y, y]).T.flatten()

        x = np.append(x, x[-1] + self.ds)[1:]
        return np.array([x, y])  # shape is (2 x n)

    def get_absolute_e_range(self) -> np.ndarray:
        """return numpy array with shape of (2, num_s+2), first row lower bound"""
        return get_e_range_from_ground(ground=self)

    def get_envelope(self) -> np.ndarray:
        """return numpy array with shape of (2, num_s+2), first row lower bound"""
        return get_envelope_from_ground(ground=self)


def get_e_range_from_ground(ground: Ground) -> np.array:
    max_grad_from_left: list[float] = get_minmax_gradient_list_from_p_vpi(
        sigma=ground.sigma, p_vpi=ground.potential_vpi, i_max=ground.i_max, di_max=ground.di_max,
        start_from_left=True, get_max_grad=True
    )
    max_grad_from_right: list[float] = get_minmax_gradient_list_from_p_vpi(
        sigma=ground.sigma, p_vpi=ground.potential_vpi, i_max=ground.i_max, di_max=ground.di_max,
        start_from_left=False, get_max_grad=True
    )
    max_e_from_left: np.ndarray = get_e_from_grad_list(
        e_anchor=float(ground.e6g[0]), gradients=max_grad_from_left, ds=ground.ds, de=ground.de, from_left=True)
    max_e_from_right: np.ndarray = get_e_from_grad_list(
        e_anchor=float(ground.e6g[-1]), gradients=max_grad_from_right, ds=ground.ds, de=ground.de, from_left=False)
    max_e_from_iy_max = np.ones_like(ground.e6g) * ground.iy_max
    ub = np.vstack((max_e_from_left, max_e_from_right, max_e_from_iy_max)).min(axis=0)

    min_grad_from_left: list[float] = get_minmax_gradient_list_from_p_vpi(
        sigma=ground.sigma, p_vpi=ground.potential_vpi, i_max=ground.i_max, di_max=ground.di_max,
        start_from_left=True, get_max_grad=False
    )
    min_grad_from_right: list[float] = get_minmax_gradient_list_from_p_vpi(
        sigma=ground.sigma, p_vpi=ground.potential_vpi, i_max=ground.i_max, di_max=ground.di_max,
        start_from_left=False, get_max_grad=False
    )
    min_e_from_left: np.ndarray = get_e_from_grad_list(
        e_anchor=float(ground.e6g[0]), gradients=min_grad_from_left, ds=ground.ds, de=ground.de, from_left=True)
    min_e_from_right: np.ndarray = get_e_from_grad_list(
        e_anchor=float(ground.e6g[-1]), gradients=min_grad_from_right, ds=ground.ds, de=ground.de, from_left=False)
    min_e_from_iy_max = np.ones_like(ground.e6g) * ground.iy_min
    lb = np.vstack((min_e_from_left, min_e_from_right, min_e_from_iy_max)).max(axis=0)

    return np.array([lb, ub])


def get_envelope_from_ground(ground: Ground) -> np.ndarray:
    envelope_step: int = max(ground.sigma * 2, ground.e6g.size // 15)  # to be tuned
    upper_envelope = get_envelope(series=ground.e6g, step=envelope_step, is_max=True)
    lower_envelope = get_envelope(series=ground.e6g, step=envelope_step, is_max=False)
    upper_envelope_full = interpolate_envelope(envelope=upper_envelope)
    lower_envelope_full = interpolate_envelope(envelope=lower_envelope)

    return np.array([lower_envelope_full, upper_envelope_full])





def main():
    gd = Ground(name="gd4", type_="random")
    # gd = Ground(name="gd_gaoyan", type_="real")
    print(gd.num_s)

    # x, y = gd.get_stair_plot_data()
    # x_raw, y_raw = gd.points.T
    # import matplotlib
    # matplotlib.use('TkAgg')
    # plt.plot(x, y, label="stair plot")
    # plt.plot(x_raw, y_raw, label="line plot")
    # plt.legend()
    # plt.show()

    for key in gd.__dict__:
        value = gd.__dict__[key]
        if isinstance(value, np.ndarray):
            print(key, value.shape)

    # print(gd.e6g)
    get_e_range_from_ground(ground=gd)
    pass


if __name__ == '__main__':
    main()
