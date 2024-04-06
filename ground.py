import numpy as np

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
        x = np.array([0, *sorted(np.random.random(size=num_vpi - 2)), 1]) * distance_meter
    else:
        x = np.arange(y.size) / y.size * (distance_meter + distance_meter / num_vpi)

    if fluctuation_range:
        fluctuation = np.random.random(size=x.size) * fluctuation_range * 2 - fluctuation_range
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


class Ground:
    def __init__(self, name: str, type_: str = "random") -> None:
        self.ds: float = None
        self.de: float = None
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
        self.points: np.ndarray = self.generate_points()
        self.bottom_left_corner_coordinate: tuple[float] = None
        self.points_index: np.ndarray = self.generate_point_index()  # also update bottom_left_corner_coordinate

    def generate_points(self) -> np.ndarray:
        if "point_arr" in self.__dict__:
            return self.__dict__["point_arr"]
        if "seed" in self.__dict__:
            point_arr = generate_random_ground_points(
                distance_meter=self.__dict__['distance'],
                max_elevation_meter=self.__dict__['elevation'],
                num_vpi=self.__dict__['n_vpi'],
                N_smooth=self.__dict__['n_smooth'],
                seed=self.__dict__['seed'],
                is_x_location_random=self.__dict__['x_loc_rand'],
                fluctuation_range=self.__dict__['fl_range']
            )
            return point_arr
        if "point_arr_file" in self.__dict__:
            data = read_data(f"ground_data\\{self.__dict__['point_arr_file']}").values
            return data

    def generate_point_index(self) -> np.ndarray:
        # 处理points转换，变成ix，iy和左下角坐标值
        discrete_points = discretize_points(points=self.points, dx=self.ds, dy=self.de)
        discrete_points, bottom_left_corner_coordinate = shift_point_coordinates(points=discrete_points)

        # update attribute
        self.bottom_left_corner_coordinate = bottom_left_corner_coordinate

        points_index = np.array([discrete_points[:, 0] / self.ds, discrete_points[:, 1] / self.de]).astype(int).T
        return points_index

    def get_v_lim_array(self) -> np.ndarray:
        ...

    def get_curve_resist_array(self) -> np.ndarray:
        ...

    def get_potential_vpi_array(self) -> np.ndarray:
        iys_diff3 = np.diff(self.points_index[:, 1], n=3)
        p_vpi_array = np.zeros(self.points_index.shape[0])
        for i, iy_diff in enumerate(iys_diff3[:-1]):
            ix = i + 2
            next_iy_diff = iys_diff3[i + 1]
            if iy_diff * next_iy_diff < 0:
                p_vpi_array[ix] = 1
        p_vpi_array[0], p_vpi_array[-1] = 1, 1
        return p_vpi_array.astype(int)


def main():
    gd = Ground(name="gd1")
    # print(gd.__dict__)
    print(gd.get_potential_vpi_array())
    pass


if __name__ == '__main__':
    main()
