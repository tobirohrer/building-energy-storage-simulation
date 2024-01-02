# Building Energy Storage Simulation

[![Documentation Status](https://readthedocs.org/projects/building-energy-storage-simulation/badge/?version=latest)](https://building-energy-storage-simulation.readthedocs.io/en/latest/)

<img src="docs/imgs/overview.drawio.png" alt="isolated" width="600"/>

The Building Energy Storage Simulation serves as OpenAI gym (now [gymnasium](https://github.com/Farama-Foundation/Gymnasium)) environment 
for Reinforcement Learning. The environment represents a building with an energy storage (in form of a battery) and a 
solar energy system. The building is connected to a power grid with time varying electricity prices. The task is to 
control the energy storage so that the total cost of electricity are minimized.

The inspiration of this project and the data profiles come from the [CityLearn](https://github.com/intelligent-environments-lab/CityLearn) environment. Anyhow, this project focuses on the ease of usage and the simplicity of its implementation. Therefore, this project serves as playground for those who want to get started with reinforcement learning for energy management system control.

## Installation

By using pip just: 

```
pip install building-energy-storage-simulation
```

or if you want to continue developing the package:

```
git clone https://github.com/tobirohrer/building-energy-storage-simulation.git && cd building-energy-storage-simulation
pip install -e .[dev]
```

## Usage

```python
from building_energy_storage_simulation import Environment, BuildingSimulation

simulation = BuildingSimulation()
env = Environment(building_simulation=simulation)

env.reset()
env.step(1)
...
```

**Important note:** This environment is implemented by using [gymnasium](https://github.com/Farama-Foundation/Gymnasium) (the proceeder of OpenAI gym). Meaning, if you are using a reinforcement learning library like [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) make sure it supports [gymnasium](https://github.com/Farama-Foundation/Gymnasium) environments. 

## Task Description

The simulation contains a building with an energy load profile attached to it. The load is always automatically covered by

- primarily using electricity generated by the solar energy system,
- and secondary by using the remaining required electricity "from the grid"

When energy is taken from the grid, costs are incurred that can vary depending on the time (if a price profile is passed
as `electricity_price` to `BuildingSimulation`). The simulated building contains a battery which be controlled by 
**charging** and **discharging** energy. The goal is to find control strategies optimize the use of the energy storage
by e.g. charging whenever electricity prices are high or whenever there is a surplus of solar generation. It is important 
to note that no energy can be fed into the grid. This means any surplus of solar energy which is not used to charge the
battery is considered lost.

### Reward

$$ r_t = -1 * electricity\_consumed_t * electricity\_price_t$$ 

Note, that the term `electricity_consumed` cannot be negative. This means, excess energy from the solar 
energy system which is not consumed by the electricity load or by charging the battery is considered lost 
(`electricity_consumed` is 0 in this case). 
 
### Action Space

| Action   | Min      | Max    |
|----------|----------|--------|
| Charge   | -1       | 1      |

The actions lie in the interval of [-1;1]. The action represents a fraction of the maximum energy which can be retrieved from the battery (or used to charge the battery) per time step.

- 1 means maximum charging the battery. The maximum charge per time step is defined by the parameter `max_battery_charge_per_timestep`.
- -1 means maximum discharging the battery, meaning "gaining" electricity out of the battery
- 0 means don't charge or discharge

### Observation Space

| Index       | Observation                           | Min                       | Max                          |
|-------------|---------------------------------------|---------------------------|------------------------------|
| 0           | State of Charge (in kWh)              | 0                         | `battery_capacity`           |
| [1; n]      | Forecast Electric Load (in kWh)       | Min of Load Profile       | Max of Load Profile          |
| [n+1; 2*n]  | Forecast Solar Generation (in kWh)    | Min of Generation Profile | Max of Generation Profile    |
| [2n+1; 3*n] | Electricity Price (in € cent per kWh) | Min of Price Profile      | Max of Price Profile         |


The length of the observation depends on the length of the forecast ($n$) used. By default, the simulation uses a forecast length of 4. 
This means 4 time steps of an electric load forecast, 4 time steps of a solar generation forecast and 4 time steps of the 
electric price profile are included in the observation. 
In addition to that, the information about the current state of charge of the battery is contained in the observation.

The length of the forecast can be defined by setting the parameter `num_forecasting_steps` of the `Environment()`.


### Episode Ends

The episode ends if the `max_timesteps` of the `Environment()` are reached.

## Example Solutions

The folder [example_solutions](example_solutions) contains three different example solutions to solve the problem 
described.

1. By applying deep reinforcement learning using the framework [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).
2. By formulating the problem as optimal control problem (OCP) using [pyomo](http://www.pyomo.org/). In this case, it 
   is assumed that the forecast for the price, load and generation data for the whole period is available. 
3. By model predictive control, which solves the optimal control problem formulation from 2. in each time step in a closed loop manner.
   In contrast to 2. only a forecast of a fixed length is given in each iteration. 

Note that the execution of the example solutions requires additional dependencies which are not specified inside `setup.py`. 
Therefore, make sure to install the required python packages defined in `requirements.txt`. Additionally, an installation 
of the `ipopt` solver is required in order to solve the optimal control problem 
(by using conda, simply run `conda install -c conda-forge ipopt`). 

## Code Documentation

The documentation is available at [https://building-energy-storage-simulation.readthedocs.io/](https://building-energy-storage-simulation.readthedocs.io/en/master/)

## Contribute & Contact

As I just started with this project, I am very happy for any kind of
contribution! In case you want to contribute, or if you have any
questions, contact me via
[discord](https://discord.com/users/tobirohrer#8654).

