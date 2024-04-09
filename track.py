import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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


class Track:
    def __init__(self, e: np.array, pi: np.array = None, ground: Ground = None) -> None:
        self.e: np.array = e  # shape (S+2, )
        self.pi: np.array = pi  # shape (S, )
        self.ground: Ground = ground

        pass

    def get_stair_plot_data(self) -> tuple[np.array, np.array]:
        ground_stair_points = self.ground.get_stair_plot_data()
        track_x = ground_stair_points[0].copy()
        track_y = self.e[1:-1].copy() * self.ground.de + self.ground.bottom_left_corner_coordinates[1]
        track_y = np.array([track_y, track_y]).T.flatten()

        return ground_stair_points, np.array([track_x, track_y])

    def plot_ground_track(self) -> plt.Figure:
        track_type = get_track_type(
            e6g=self.ground.e6g * self.ground.de,
            e=self.e * self.ground.de,
            hb_min=self.ground.hb_min, ht_min=self.ground.ht_min)

        fig, ax = plt.subplots(1, 1, figsize=(14, 5), dpi=150)
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
                ax.fill_between([real_x, real_x + self.ground.ds],
                                self.ground.e6g[x] * self.ground.de + self.ground.bottom_left_corner_coordinates[1],
                                self.e[x] * self.ground.de + self.ground.bottom_left_corner_coordinates[1],
                                facecolor=cl,
                                alpha=0.5)
        # add shades legend
        legend_elements.extend([Patch(facecolor=track_type_colors['cut'], alpha=0.5, label='Earth cut'),
                                Patch(facecolor=track_type_colors['fill'], alpha=0.5, label='Earth fill'),
                                Patch(facecolor=track_type_colors['tn'], alpha=0.5, label='Tunnel'),
                                Patch(facecolor=track_type_colors['bg'], alpha=0.5, label='Bridge')])
        ax.legend(handles=legend_elements)
        ax.set_xlabel("Horizontal Location")
        ax.set_ylabel("Elevation")
        plt.show()
        return fig


def main():
    ground = Ground("gd_gaoyan", "real")
    # print(ground.e6g)
    # e = ground.e6g.copy() + np.random.rand(ground.num_s + 2) * 100 - 50
    # e[0], e[-1] = ground.e6g[0], ground.e6g[-1]
    e = np.linspace(ground.e6g[0], ground.e6g[-1], ground.num_s + 2)
    # print(e)

    track = Track(ground=ground, e=e, pi=np.zeros_like(e))
    # print(track.get_stair_plot_data())
    g, t = track.get_stair_plot_data()

    matplotlib.use("TkAgg")
    track.plot_ground_track()

    pass


if __name__ == "__main__":
    main()
