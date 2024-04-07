import numpy as np

from dataIO import read_data


class Train:
    def __init__(self, name: str):
        data = read_data(f"train_data\\{name}.json")
        self.name: str = name
        self.M_t: float = None
        self.r0: float | None = None
        self.r1: float | None = None
        self.r2: float | None = None
        self.r_tn: float | None = None
        self.r0_strahl: float = None
        self.r2_strahl: float = None
        self.rho: float = None
        self.mu: float = None
        self.g: float = None
        self.PWL_EK: np.ndarray = None
        self.PWL_F_EK: np.ndarray = None
        self.max_v: float = None
        self.max_resist: float = None
        self.traction_e: np.ndarray = None
        self.brake_e: np.ndarray = None

        for k, v in data.items():
            if isinstance(v, list):
                try:
                    v = np.array(v)
                except ValueError:
                    pass
            setattr(self, k, v)


def main():
    pass


if __name__ == "__main__":
    main()
