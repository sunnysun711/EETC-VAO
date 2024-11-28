# Integrating Energy-Efficient Train Control in Railway Vertical Alignment Optimization

This repository contains the code implementation for the academic paper titled "Integrating Energy-Efficient Train Control in Railway Vertical Alignment Optimization: A Novel Mixed-Integer Linear Programming Approach". The paper is currently under revision for potential publication in Transportation Research Part C.

## Project Overview

This research presents a novel approach to optimize railway track design and train control strategies simultaneously. The project aims to minimize construction costs and energy consumption while satisfying various operational constraints, using a Mixed-Integer Linear Programming (MILP) model.

## Key Components

1. Vertical Alignment Optimization (VAO)
2. Energy-Efficient Train Control (EETC)
3. Integrated VAO-EETC optimization
4. Ground profile generation and analysis
5. Heuristic algorithms for efficient solution generation

## Key Features

- Novel MILP approach for integrating EETC with VAO
- Generation and analysis of random and real ground profiles
- Support for multiple train types (e.g., CRH380AL, HXD1D, HXD2)
- Consideration of various constraints such as gradient limits, curve resistance, and tunnel effects
- Efficient heuristic algorithms for generating initial VAO and EETC solutions

## File Structure

- `ground.py`: Defines the `Ground` class for handling ground profile data and calculations
- `track.py`: Implements the `Track` class for track design and analysis
- `train.py`: Contains the `Train` class with train-specific parameters and methods
- `heuristic.py`: Provides efficient heuristic algorithms for generating VAO and EETC solutions
- `optimize.py`: Implements the main optimization models (VAO, EETC, EETC-VAO) using MILP
- `dataIO.py`: Handles data input/output operations

## Data Files

- `random_ground_data.json`: Contains parameters for generating random ground profiles
- `real_ground_data.json`: Stores data for real ground profiles
- Train-specific JSON files (e.g., `CRH380AL.json`, `HXD1D.json`, `HXD2.json`): Contain train parameters

## Usage

To use this system, you'll need to:

1. Set up the required environment (Python with necessary libraries such as NumPy, Matplotlib, and Gurobi)
2. Choose or generate a ground profile
3. Select a train type
4. Run the desired optimization model (VAO, EETC, or integrated EETC-VAO)

Example usage:

```python
from ground import Ground
from train import Train
from optimize import EETC_VAO
from heuristic import gen_warm_start_data

# Initialize ground and train objects
ground = Ground("gd2")
train = Train("CRH380AL")

# Generate initial solutions using heuristic algorithms
warm_start_data = gen_warm_start_data(ground, train)

# Run integrated EETC-VAO optimization
model = EETC_VAO(ground=ground, train=train, LC_on=True, VI_on=True, tcVI_on=True, warm_start_data=warm_start_data)
model.optimize()

# Analyze and visualize results
# model.plot_results()  # this will automatically execute inside EETC_VAO.optimize()
```

## Heuristic Algorithms

The `heuristic.py` file contains efficient algorithms for generating initial solutions:

1. VAO solution generation: Creates feasible vertical alignments considering various constraints
2. EETC solution generation: Develops energy-efficient train control strategies based on a given VAO solution

These heuristic solutions can be used as warm starts for the MILP model, potentially reducing computation time and improving solution quality.

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
- Gurobi Optimizer

## Associated Research

This code repository is associated with the academic paper "Integrating Energy-Efficient Train Control in Railway Vertical Alignment Optimization: A Novel Mixed-Integer Linear Programming Approach". Please cite the paper as below:
```text
@article{SUN2025104943,
title = {Integrating Energy-Efficient Train Control in railway Vertical Alignment Optimization: A novel Mixed-Integer Linear Programming approach},
journal = {Transportation Research Part C: Emerging Technologies},
volume = {171},
pages = {104943},
year = {2025},
issn = {0968-090X},
doi = {https://doi.org/10.1016/j.trc.2024.104943},
url = {https://www.sciencedirect.com/science/article/pii/S0968090X24004649},
author = {Yichen Sun and Shaoquan Ni and Dingjun Chen and Qing He and Shuangting Xu and Yan Gao and Tao Chen},
keywords = {Railway, Vertical Alignment Optimization, Energy-Efficient Train Control, Mixed-Integer Linear Programming, Optimization, Lifecycle cost},
}
```

## License

This project is licensed under a custom academic use license - see the [LICENSE](LICENSE) file for details.

Note: Use of this software requires explicit permission from the author and appropriate citation of the associated research paper.

## Contact

For questions or further information, please contact SUN Yichen at [yichensun@my.swjtu.edu.cn].
