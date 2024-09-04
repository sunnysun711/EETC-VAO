import numpy as np

from dataIO import read_data


class Train:
    def __init__(self, name: str):
        self.name: str = name
        data = read_data(f"train_data\\{name}.json")
        self.data: dict = data
        self.M_t: float = data["M_t"]  # Mass. ton
        self.rho: float = data["rho"]  # rotating inertia
        self.mu: float = data["mu"]  # the energy consumption of auxiliary power. kWh/s
        self.g: float = data["g"]  # gravity acceleration. m/s^2
        self.r0: float | None = data["r0"]
        self.r1: float | None = data["r1"]
        self.r2: float = data["r2"]
        self.r_tn: float = data["r_tn"]
        self.r0_strahl: float = data["r0_strahl"]
        self.r2_strahl: float = data["r2_strahl"]
        self.PWL_EK: list = data["PWL_EK"]  # piecewise linear Ek
        self.PWL_F_EK: list = data["PWL_F_EK"]  # piecewise linear F_Ek
        self.max_v: float = data["max_v"]  # maximum velocity. km/h
        self.max_resist: float = data["max_resist"]  # maximum resistance. N/kN
        self.traction_e: np.ndarray = np.array(data["traction_e"])  # shape (N, 2) first col: Ek, second col: unit_force
        self.brake_e: np.ndarray = np.array(data["brake_e"])  # shape (N, 2) first col: Ek, second col: unit_force

        # max acceleration m/s^2
        self.max_acc = self.traction_e[:, 1].max() / (102 * (1 + self.rho))
        self.max_dec = self.brake_e[:, 1].max() / (102 * (1 + self.rho))

        # running attributes
        self.v: float | None = None
        self.u_f: float | None = None
        self.u_b: float | None = None
        self.w_j: float | None = None
        self.w_0: float | None = None
        self.w_i: float | None = None
        self.w_r: float | None = None
        self.w_tn: float | None = None
        self.is_tunnel = None
        return

    def reset_running_attr(self):
        """
        reset all running attributes to None
        :return:
        """
        self.v = None
        self.u_f = None
        self.u_b = None
        self.w_j = None
        self.w_0 = None
        self.w_i = None
        self.w_r = None
        self.w_tn = None
        self.is_tunnel = None
        return

    def unit_w0(self, v: float | np.ndarray, unit: str = "km/h"):
        if unit == "km/h":
            pass
        elif unit == "m/s":
            v = v * 3.6
        else:
            raise ValueError(f"unit {unit} is not supported.")
        return self.r0_strahl + self.r2_strahl * v ** 2

    def unit_wtn(self, v: float | np.ndarray, unit: str = "km/h", is_tunnel: bool = True):
        if unit == "km/h":
            pass
        elif unit == "m/s":
            v = v * 3.6
        else:
            raise ValueError(f"unit {unit} is not supported.")
        if is_tunnel:
            w_tn = self.r_tn * self.r2 * v ** 2
        else:
            w_tn = np.zeros_like(v)
        return w_tn

    def next_v_range(self, ds, speed_limit, speed_min, v=None, w_0=None, w_i=None, w_r=None, w_tn=None):
        """
        deduct possible velocity range at next delta s. considering speed_limit, speed_min and resistances.
        :param ds: delta s. meter.
        :param speed_limit: max speed allowed. km/h.
        :param speed_min: minimum speed allowed. km/h.
        :param v: int, float, list, np.array. velocity km/h.
        :param w_0: basic resistance. N/kN
        :param w_i: gradient resistance. N/kN
        :param w_r: curve resistance. N/kN
        :param w_tn: tunnel resistance. N/kN
        :return: np.array, [[v, lb, ub], ...]
        """
        v = np.array(v) if v is not None else self.v
        w_0 = w_0 if w_0 is not None else self.w_0
        w_i = w_i if w_i is not None else self.w_i
        w_r = w_r if w_r is not None else self.w_r
        w_tn = w_tn if w_tn is not None else self.w_tn
        max_traction = self.unit_traction(v=v)
        max_brake = self.unit_brake(v=v)

        resist = w_i + w_r + w_0 + w_tn
        joint_force_min = -max_brake - resist
        joint_force_max = max_traction - resist

        next_ek_min = 2 * ds * 3.6 ** 2 / (102 * (1 + self.rho)) * joint_force_min + v ** 2
        next_ek_max = 2 * ds * 3.6 ** 2 / (102 * (1 + self.rho)) * joint_force_max + v ** 2

        if v.size == 1:  # v is a float
            # ek greater than 0
            next_ek_min = max(0, next_ek_min)
            next_ek_max = max(0, next_ek_max)
            # must not exceed speed_min, speed_limit range
            next_v_min = max(np.sqrt(next_ek_min), speed_min)
            next_v_max = min(np.sqrt(next_ek_max), speed_limit)
        else:  # v is iterable
            # ek greater than 0
            next_ek_min[next_ek_min < 0] = 0
            next_ek_max[next_ek_max < 0] = 0
            next_v_min = np.sqrt(next_ek_min)
            next_v_max = np.sqrt(next_ek_max)
            # must not exceed speed_min, speed_limit range
            next_v_min[next_v_min < speed_min] = speed_min
            next_v_max[next_v_max > speed_limit] = speed_limit

        next_v_range = np.vstack((v, next_v_min, next_v_max)).T
        return next_v_range
        pass

    def unit_traction(self, v, unit: str = "km/h"):
        if unit == "km/h":
            pass
        elif unit == "m/s":
            v = v * 3.6
        else:
            raise ValueError(f"unit {unit} is not supported.")
        ek = v ** 2
        data = self.traction_e
        return np.interp(ek, data[:, 0], data[:, 1])

    def unit_brake(self, v, unit: str = "km/h"):
        if unit == "km/h":
            pass
        elif unit == "m/s":
            v = v * 3.6
        else:
            raise ValueError(f"unit {unit} is not supported.")
        ek = v ** 2
        data = self.brake_e
        return np.interp(ek, data[:, 0], data[:, 1])

    def get_unit_control_from_v(self, ds, v_selected, v=None, w_i=None, w_r=None, w_0=None, w_tn=None):
        """

        :param ds: delta s.
        :param v_selected:
        :param v: must be non-iterable.
        :param w_i:
        :param w_r:
        :param w_0:
        :param w_tn:
        :return:
        """
        v = v if v is not None else self.v
        v_selected = np.array(v_selected)
        w_i = w_i if w_i is not None else self.w_i
        w_r = w_r if w_r is not None else self.w_r
        w_tn = w_tn if w_tn is not None else self.w_tn
        w_0 = w_0 if w_0 is not None else self.w_0

        joint_force = 102 * (1 + self.rho) * (v_selected ** 2 - v ** 2) / (2 * ds * 3.6 ** 2)
        unit_control = joint_force + w_0 + w_i + w_r + w_tn

        # check feasibility?
        tol = 1e-6
        max_traction = self.unit_traction(v=v) + tol
        max_brake = self.unit_brake(v=v) + tol
        if unit_control < -max_brake or unit_control > max_traction:
            raise ValueError(
                f"unit control {unit_control} is not feasible. Should in range (-{max_brake}, {max_traction}).")
        return unit_control


def main():
    print(Train("HXD1D").max_acc)
    print(Train("HXD2").max_acc)
    print(Train("CRH380AL").max_acc)
    print(Train("HXD1D").max_dec)
    print(Train("HXD2").max_dec)
    print(Train("CRH380AL").max_dec)
    pass


if __name__ == "__main__":
    main()
