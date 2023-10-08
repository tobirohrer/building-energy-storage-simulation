.. building-energy-storage-simulation documentation master file, created by
   sphinx-quickstart on Wed Dec 21 13:41:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to building-energy-storage-simulation's documentation!
==============================================================

The `Building Energy Storage Simulation <https://github.com/tobirohrer/building-energy-storage-simulation>`_ serves as open source OpenAI gym (now
`gymnasium <https://github.com/Farama-Foundation/Gymnasium>`_) environment for reinforcement learning. The environment
represents a building with an energy storage (in form of a battery) and a solar energy system. The aim is to control
the energy storage in such a way that the energy of the solar system can be used optimally.

A more detailed description of the task itself and the description of the markov decision process, containing
information about the action space, observation space and reward can be found
`here <https://github.com/tobirohrer/building-energy-storage-simulation>`_.

Contents
--------

.. toctree::
   :maxdepth: 1
   :Caption: API

   modules/environment
   modules/simulation
   modules/battery

Thanks To
---------

The inspiration of this project and the data profiles come from the
`CityLearn <https://github.com/intelligent-environments-lab/CityLearn>`_ environment. Anyhow, this project focuses on
the ease of usage and the simplicity of its implementation.

