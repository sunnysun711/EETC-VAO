import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from get_opt_results import get_track_ele_from_file
from ground import Ground


def get_track_type(e6g: np.array, e: np.array, ht_min: float, hb_min: float):
    cut, fill, tn, bg = [], [], [], []
    for i in range(e6g.size):
        g = e6g[i]
        t = e[i]
        if g == t:
            continue
        elif g - t > ht_min:
            tn.append([i, g - t])
        elif g - t > 0:
            cut.append([i, g - t])
        elif t - g > hb_min:
            bg.append([i, t - g])
        elif t - g > 0:
            fill.append([i, t - g])
    cut, fill, tn, bg = np.array(cut), np.array(fill), np.array(tn), np.array(bg)
    res = {'cut': cut, 'fill': fill, 'tn': tn, 'bg': bg}
    return res


def get_pi_from_e(e: np.ndarray, IntFeasTol: float = 1e-07) -> np.ndarray:
    """
    get vpi points ix locations (pi) from elevations of the track (track.e)
    :param e: elevations of the track (track.e), shape of (S+2, )
    :param IntFeasTol: tolerance for integer feasibility, default 1e-07, which should be the same as gurobi IntFeasTol.
    :return: vpi points ix locations (pi), shape of (S, )
    """
    # Calculate the differences (delta_e)
    gradients = np.diff(e)

    # # Create an array of zeros with the same length as e
    # pi = np.zeros(e.size - 2, dtype=int)

    # Check for changes in delta_e and mark them in the delta_change array
    pi = (np.abs(gradients[1:] - gradients[:-1]) > IntFeasTol).astype(int)
    return pi


class Track:
    def __init__(self, e: np.ndarray, pi: np.ndarray = None, z1: np.ndarray = None, ground: Ground = None) -> None:
        self.e: np.ndarray = e  # shape (S+2, ), index: [0, S+1]
        if pi is None:
            pi = get_pi_from_e(e)
        self.pi: np.ndarray = pi  # shape (S, ), index: [1, S]
        self.track_type = get_track_type(
            e6g=ground.e6g,
            e=e,
            ht_min=ground.ht_min / ground.de,
            hb_min=ground.hb_min / ground.de
        )
        if z1 is None:
            z1 = np.zeros(ground.num_s + 1)
            tn = self.track_type['tn']
            if tn.size != 0:
                tn_ix = tn[:, 0].astype(int)
                z1[tn_ix - 1] = 1
        self.z1: np.ndarray = z1  # shape (S+1, ), index: [1, S+1]

        if ground is not None and ground.num_s != self.e.size - 2:
            raise Exception(f"Track is not compatible with ground {ground.name}!")
        self.ground: Ground = ground
        return

    def get_stair_plot_data(self) -> tuple[np.array, np.array]:
        ground_stair_points = self.ground.get_stair_plot_data()
        track_x = ground_stair_points[0].copy()
        track_y = self.e[1:-1].copy() * self.ground.de + self.ground.bottom_left_corner_coordinates[1]
        track_y = np.array([track_y, track_y]).T.flatten()

        return ground_stair_points, np.array([track_x, track_y])

    def plot_ground_track(self, plt_show: bool = False) -> plt.Figure:
        track_type = get_track_type(
            e6g=self.ground.e6g * self.ground.de,
            e=self.e * self.ground.de,
            hb_min=self.ground.hb_min, ht_min=self.ground.ht_min)

        fig, ax = plt.subplots(1, 1, figsize=(8, 3), dpi=150)
        ground_stair_points, track_stair_points = self.get_stair_plot_data()
        ax.plot(ground_stair_points[0], ground_stair_points[1], lw=0.5, label="ground", c="lightgray")
        ax.plot(track_stair_points[0], track_stair_points[1], lw=0.5, label="track", c="black")
        # legend of the figure
        legend_elements = [Line2D([0], [0], color="lightgray", lw=1, label='Ground'),
                           Line2D([0], [0], color="black", lw=1, label='Track')]
        track_type_colors: dict[str, str] = {'cut': 'b', 'fill': 'r', 'tn': 'g', 'bg': 'y'}
        for key in track_type.keys():
            cl = track_type_colors[key]
            if len(track_type[key]) == 0:
                continue
            data = track_type[key][:, 0].astype(int)
            for x in data:
                real_x = (x - 1) * self.ground.ds + self.ground.bottom_left_corner_coordinates[0]
                ax.fill_between(
                    [real_x, real_x + self.ground.ds],
                    self.ground.e6g[x] * self.ground.de + self.ground.bottom_left_corner_coordinates[1],
                    self.e[x] * self.ground.de + self.ground.bottom_left_corner_coordinates[1],
                    facecolor=cl,
                    alpha=0.5)
        # add shades legend
        legend_elements.extend([Patch(facecolor=track_type_colors['cut'], alpha=0.5, label='Earth cut'),
                                Patch(facecolor=track_type_colors['fill'], alpha=0.5, label='Earth fill'),
                                Patch(facecolor=track_type_colors['tn'], alpha=0.5, label='Tunnel'),
                                Patch(facecolor=track_type_colors['bg'], alpha=0.5, label='Bridge')])
        ax.legend(handles=legend_elements, fontsize="small")
        ax.set_xlabel("Horizontal location (m)", fontsize="small")
        ax.set_ylabel("Elevation (m)", fontsize="small")
        plt.tight_layout()
        if plt_show:
            plt.show()
        return fig

    def get_warm_start_data(self) -> dict[str, dict]:
        """
        :return:
        e: range(0, S+2)
        pi: range(1, S+1)
        z1: range(1, S+2)
        z2: range(1, S+1)
        z3: range(1, S+1)
        z4: range(1, S+1)
        z5: range(1, S+1)
        gamma: (range(1, 4), range(1, S+1))
        C: range(1, S+1)
        C6tn_e: range(1, S+1)
        """
        num_s = self.ground.num_s
        res = {
            'e': {i: self.e[i] for i in range(num_s + 2)},
            'pi': {int(i + 1): int(self.pi[i]) for i in range(num_s)},
            'z1': {int(i + 1): int(self.z1[i]) for i in range(num_s + 1)}
        }

        z2 = np.zeros(num_s)
        z2[self.ground.e6g[1:-1] > self.e[1:-1]] = 1
        res['z2'] = {int(i + 1): int(z2[i]) for i in range(num_s)}

        z3 = np.zeros(num_s)
        z3[self.ground.e6g[1:-1] <= self.e[1:-1] - self.ground.hb_min / self.ground.de] = 1
        res['z3'] = {int(i + 1): int(z3[i]) for i in range(num_s)}

        z4 = np.maximum(self.z1[1:], self.z1[:-1])
        z5 = np.minimum(self.z1[1:], self.z1[:-1])
        res['z4'] = {int(i + 1): int(z4[i]) for i in range(num_s)}
        res['z5'] = {int(i + 1): int(z5[i]) for i in range(num_s)}

        gamma1 = self.z1[:-1] * self.e[1:-1]
        gamma2 = z2 * self.e[1:-1]
        gamma3 = z3 * self.e[1:-1]
        gamma = np.vstack((gamma1, gamma2, gamma3))
        res['gamma'] = {(int(i + 1), int(j + 1)): gamma[i, j] for i in range(3) for j in range(num_s)}

        C = {}
        for s in range(1, num_s + 1):
            C[s] = (res['z1'][s] * (self.ground.c_tn / self.ground.de - self.ground.c_cut * self.ground.e6g[s]) + \
                    res['z2'][s] * (self.ground.c_cut + self.ground.c_fill) * self.ground.e6g[s] + \
                    res['z3'][s] * (self.ground.c_fill - self.ground.c_bg) * self.ground.e6g[s] + \
                    res['gamma'][1, s] * self.ground.c_cut - \
                    res['gamma'][2, s] * (self.ground.c_cut + self.ground.c_fill) + \
                    res['gamma'][3, s] * (self.ground.c_bg - self.ground.c_fill) + \
                    self.ground.c_fill * self.e[s] - \
                    self.ground.c_fill * self.ground.e6g[s]) * (self.ground.de * self.ground.ds)
        res['C'] = C

        C6tn_e = self.ground.c6e_tn * (z4 - z5)
        res['C6tn_e'] = {int(i + 1): C6tn_e[i] for i in range(num_s)}

        return res


def gen_track_from_file(ground: Ground, json_file: str) -> Track:
    """generate a Track object from a json VAO solution file."""
    e = get_track_ele_from_file(file=json_file, S=ground.num_s)
    return Track(ground=ground, e=e)


def main():
    # ground = Ground("gd_gaoyan")
    ground = Ground("gd2")

    from heuristic import gen_track
    import matplotlib
    matplotlib.use("TkAgg")
    track = gen_track(ground=ground, display_on=True)
    track.get_warm_start_data()
    track.plot_ground_track()

    pass


if __name__ == "__main__":
    main()
