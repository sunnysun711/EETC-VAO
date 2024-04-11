import numpy as np

from dataIO import read_data


class Train:
    def __init__(self, name: str):
        self.name: str = name
        data = read_data(f"train_data\\{name}.json")
        self.data: dict = data
        self.M_t: float = data["M_t"]
        self.rho: float = data["rho"]
        self.mu: float = data["mu"]
        self.g: float = data["g"]
        self.r0: float | None = data["r0"]
        self.r1: float | None = data["r1"]
        self.r2: float = data["r2"]
        self.r_tn: float = data["r_tn"]
        self.r0_strahl: float = data["r0_strahl"]
        self.r2_strahl: float = data["r2_strahl"]
        self.PWL_EK: list = data["PWL_EK"]
        self.PWL_F_EK: list = data["PWL_F_EK"]
        self.max_v: float = data["max_v"]
        self.max_resist: float = data["max_resist"]
        self.traction_e: np.ndarray = np.array(data["traction_e"])
        self.brake_e: np.ndarray = np.array(data["brake_e"])
        return


def main():
    pass


if __name__ == "__main__":
    main()
