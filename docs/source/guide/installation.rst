.. _usage:

Usage
=====

.. _installation:

Install Stable Release
----------------------

To install the building-energy-storage-simulation with pip, execute:

.. code-block:: console

   (.venv) $ pip install building-energy-storage-simulation

Install Development Version
---------------------------

To contribute to this project, with support for running tests and building the documentation.

.. code-block:: bash

    git clone https://github.com/tobirohrer/building-energy-storage-simulation.git && cd building-energy-storage-simulation
    pip install -e .[docs,tests]

Example Usage
-------------

To use the building-energy-storage-simulation, first install it using pip:

.. code-block:: python

    from building_energy_storage_simulation import Environment, BuildingSimulation

    simulation = BuildingSimulation()
    env = Environment(building_simulation=simulation)

    env.reset()
    env.step(1)
    ...
