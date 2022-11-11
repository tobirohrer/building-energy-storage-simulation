# Building Energy Storage Simulation

<img src="docs/imgs/overview.drawio.png" alt="isolated" width="600"/>

The Building Energy Storage Simulation serves as open Source OpenAI gym (now [gymnasium](https://github.com/Farama-Foundation/Gymnasium)) environment for Reinforcement Learning. The environment represents a building with an energy storage (in form of a battery) and solar energy system. The aim is to control the energy storage in such a way that the energy of the solar system can be used optimally. 

The inspiration of this project and the data profiles come from the [CityLearn](https://github.com/intelligent-environments-lab/CityLearn) environment.

## Installation

1. clone it to your local filesystem: `git clone https://github.com/tobirohrer/building-energy-storage-simulation.git`		
2. cd into the cloned repository: `cd python-project-template`
3. optional:

	4. create new python environment: `conda create -n <env_name> python=3.X`
	5. activate new python environment: `conda activate <env_name>`
6. install the package by: `pip install .` (If you want to run tests, build the sphinx documentation locally, or want to continue developing, install it via `pip install -e .[docs,tests]`)
7. done :)

## Usage 

```
from building_energy_storage_simulation import Environment
env = Environment()
env.reset()
env.step(42)
...
```

